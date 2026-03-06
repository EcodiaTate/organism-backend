"""
EcodiaOS — Soma Temporal Derivative Engine

Computes velocity (S'), acceleration (S''), and jerk (S''') of the organism
state trajectory using Savitzky-Golay filtering for noise-robust
differentiation at three time scales:

  fast   ~ 1s   (window=7,   polyorder=3)  — function-level transients
  medium ~ 10s  (window=67,  polyorder=3)  — system interaction dynamics
  slow   ~ 100s (window=667, polyorder=3)  — drift and long-term trends

Returns per-system breakdown plus whole-organism L2 norms, and identifies
which system dominates each derivative at each scale (the localization signal).

Pure numerical computation. No I/O. Budget: ~1ms on typical trajectory sizes.
"""

from __future__ import annotations

import math
from collections import deque

import structlog

from systems.soma.types import (
    SYSTEM_STATE_DIM,
    DerivativeSnapshot,
    OrganismStateVector,
)

logger = structlog.get_logger("systems.soma.temporal_derivatives")


# ─── Savitzky-Golay Coefficient Precomputation ──────────────────────

# SG coefficients for first, second, and third derivatives.
# Precomputed once at module load for the three scale windows.
# Uses the Gram polynomial / matrix pseudoinverse approach.


def _sg_coefficients(window_size: int, polyorder: int, deriv: int) -> list[float]:
    """Compute Savitzky-Golay convolution coefficients for a given derivative.

    Uses the matrix pseudoinverse approach: build Vandermonde matrix J,
    compute (J^T J)^{-1} J^T, extract the row for the requested derivative
    order scaled by deriv!.

    For the window sizes we use (7, 67, 667), this runs once at import.
    """
    if window_size < polyorder + 1:
        if deriv == 1 and window_size >= 2:
            return [-1.0, 1.0]
        return [0.0] * max(window_size, 1)

    m = window_size
    half = m // 2

    p = polyorder + 1
    # Build J^T J
    jtj: list[list[float]] = [[0.0] * p for _ in range(p)]
    for i in range(-half, half + 1):
        for r in range(p):
            for c in range(p):
                jtj[r][c] += float(i) ** (r + c)

    # Invert J^T J using Gauss-Jordan elimination
    n = p
    aug: list[list[float]] = [
        [jtj[r][c] if c < n else (1.0 if c - n == r else 0.0) for c in range(2 * n)]
        for r in range(n)
    ]

    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-15:
            continue
        for c in range(2 * n):
            aug[col][c] /= pivot
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for c in range(2 * n):
                    aug[row][c] -= factor * aug[col][c]

    inv_jtj: list[list[float]] = [[aug[r][c + n] for c in range(n)] for r in range(n)]

    # Compute SG coefficients for derivative order `deriv`
    factorial = 1
    for i in range(1, deriv + 1):
        factorial *= i

    coeffs: list[float] = []
    for j in range(-half, half + 1):
        val = 0.0
        for k in range(p):
            val += inv_jtj[deriv][k] * float(j) ** k
        coeffs.append(val * factorial)

    return coeffs


# Scale definitions matching the spec
DERIVATIVE_SCALES: dict[str, dict[str, int]] = {
    "fast": {"window": 7, "polyorder": 3},
    "medium": {"window": 67, "polyorder": 3},
    "slow": {"window": 667, "polyorder": 3},
}

# Precompute SG coefficients for all scales and derivative orders
_SG_CACHE: dict[tuple[str, int], list[float]] = {}

for _scale_name, _params in DERIVATIVE_SCALES.items():
    for _deriv_order in (1, 2, 3):
        _SG_CACHE[(_scale_name, _deriv_order)] = _sg_coefficients(
            _params["window"], _params["polyorder"], _deriv_order,
        )


# ─── Engine ─────────────────────────────────────────────────────────


class TemporalDerivativeEngine:
    """Computes multi-scale temporal derivatives of the organism state trajectory.

    Maintains a rolling history of OrganismStateVectors and applies
    Savitzky-Golay convolution to compute velocity, acceleration, and jerk
    at fast (~1s), medium (~10s), and slow (~100s) scales.
    """

    def __init__(self, history_size: int = 2000) -> None:
        self._history: deque[OrganismStateVector] = deque(maxlen=history_size)
        self._system_order: list[str] = []

    def push(self, state_vector: OrganismStateVector) -> None:
        """Add a new state vector to the rolling history."""
        self._history.append(state_vector)
        for sid in state_vector.systems:
            if sid not in self._system_order:
                self._system_order.append(sid)

    @property
    def history_length(self) -> int:
        return len(self._history)

    def compute(self) -> DerivativeSnapshot:
        """Compute velocity, acceleration, and jerk at all three scales.

        Returns a DerivativeSnapshot with per-system breakdown and
        whole-organism L2 norms.
        """
        timestamp = self._history[-1].timestamp if self._history else 0.0

        snap = DerivativeSnapshot(timestamp=timestamp)

        for scale_name, params in DERIVATIVE_SCALES.items():
            window = params["window"]
            if len(self._history) < window:
                # Not enough history for this scale — emit zeros
                snap.velocity[scale_name] = {}
                snap.acceleration[scale_name] = {}
                snap.jerk[scale_name] = {}
                snap.organism_velocity_norm[scale_name] = 0.0
                snap.organism_acceleration_norm[scale_name] = 0.0
                snap.organism_jerk_norm[scale_name] = 0.0
                snap.dominant_system_velocity[scale_name] = ""
                snap.dominant_system_acceleration[scale_name] = ""
                snap.dominant_system_jerk[scale_name] = ""
                continue

            # Extract the most recent `window` state vectors
            recent = list(self._history)[-window:]

            vel_by_sys, acc_by_sys, jrk_by_sys = self._convolve_scale(
                recent, scale_name,
            )

            snap.velocity[scale_name] = vel_by_sys
            snap.acceleration[scale_name] = acc_by_sys
            snap.jerk[scale_name] = jrk_by_sys

            # Whole-organism L2 norms
            snap.organism_velocity_norm[scale_name] = self._organism_norm(vel_by_sys)
            snap.organism_acceleration_norm[scale_name] = self._organism_norm(acc_by_sys)
            snap.organism_jerk_norm[scale_name] = self._organism_norm(jrk_by_sys)

            # Dominant system localization
            snap.dominant_system_velocity[scale_name] = self._dominant_system(vel_by_sys)
            snap.dominant_system_acceleration[scale_name] = self._dominant_system(acc_by_sys)
            snap.dominant_system_jerk[scale_name] = self._dominant_system(jrk_by_sys)

        return snap

    def _convolve_scale(
        self,
        vectors: list[OrganismStateVector],
        scale_name: str,
    ) -> tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ]:
        """Apply SG convolution at a given scale for all three derivative orders.

        Returns (velocity, acceleration, jerk) dicts mapping system_id to
        7-element derivative vectors.
        """
        vel_coeffs = _SG_CACHE[(scale_name, 1)]
        acc_coeffs = _SG_CACHE[(scale_name, 2)]
        jrk_coeffs = _SG_CACHE[(scale_name, 3)]

        n = len(vectors)
        vel: dict[str, list[float]] = {}
        acc: dict[str, list[float]] = {}
        jrk: dict[str, list[float]] = {}

        for sid in self._system_order:
            # Build time series for this system: n x 7
            series: list[list[float]] = []
            for sv in vectors:
                s = sv.systems.get(sid)
                series.append(s.to_list() if s else [0.0] * SYSTEM_STATE_DIM)

            # Convolve each feature dimension with the SG coefficients
            vel_vec: list[float] = [0.0] * SYSTEM_STATE_DIM
            acc_vec: list[float] = [0.0] * SYSTEM_STATE_DIM
            jrk_vec: list[float] = [0.0] * SYSTEM_STATE_DIM

            for d in range(SYSTEM_STATE_DIM):
                for i in range(n):
                    val = series[i][d]
                    if i < len(vel_coeffs):
                        vel_vec[d] += vel_coeffs[i] * val
                    if i < len(acc_coeffs):
                        acc_vec[d] += acc_coeffs[i] * val
                    if i < len(jrk_coeffs):
                        jrk_vec[d] += jrk_coeffs[i] * val

            vel[sid] = vel_vec
            acc[sid] = acc_vec
            jrk[sid] = jrk_vec

        return vel, acc, jrk

    @staticmethod
    def _organism_norm(by_system: dict[str, list[float]]) -> float:
        """Compute the L2 norm across all systems' derivative vectors."""
        total_sq = 0.0
        for vec in by_system.values():
            for v in vec:
                total_sq += v * v
        return math.sqrt(total_sq) if total_sq > 0 else 0.0

    @staticmethod
    def _dominant_system(by_system: dict[str, list[float]]) -> str:
        """Find the system with the largest L2 norm derivative."""
        best_sid = ""
        best_norm = 0.0
        for sid, vec in by_system.items():
            norm_sq = sum(v * v for v in vec)
            if norm_sq > best_norm:
                best_norm = norm_sq
                best_sid = sid
        return best_sid
