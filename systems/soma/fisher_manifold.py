"""
EcodiaOS — Soma Fisher Information Manifold

The organism's state distributions define a statistical manifold. The Fisher
information matrix provides a Riemannian metric — the organism's intrinsic
sense of "how different does this feel from normal."

For a Gaussian approximation of the state distribution, the Fisher information
matrix equals the inverse covariance: F = Sigma_inv. This gives us a natural
metric that downweights noisy dimensions (high variance, low Fisher info) and
upweights signal dimensions (low variance, high Fisher info).

Designed for the deep analysis path (~75s interval). The baseline is learned
during initial operation and frozen after calibration.

Dependencies: numpy, scipy, scikit-learn (Ledoit-Wolf shrinkage).
"""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np
import structlog
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf

logger = structlog.get_logger("systems.soma.fisher_manifold")


# ─── Output Types ────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class GeodesicDeviation:
    """Result of measuring how far the current state is from the healthy baseline."""

    scalar: float
    """Overall deviation magnitude (Mahalanobis distance to nearest baseline point)."""

    direction: np.ndarray
    """Unit vector pointing from nearest baseline point to current state.
    Shows which dimensions are distorted and by how much."""

    dominant_systems: list[int]
    """Dimension indices contributing most to the deviation, sorted by contribution."""

    percentile: float
    """Where this deviation falls in the historical distribution [0, 100].
    95th percentile means only 5% of observed states deviated more."""


# ─── Fisher Manifold ─────────────────────────────────────────────


class FisherManifold:
    """
    Maintains the Riemannian state manifold with Fisher information metric.

    The manifold is approximated empirically:
    - Collect state vectors over a rolling window
    - Estimate the Fisher information matrix (precision matrix = Sigma_inv)
      using Ledoit-Wolf shrinkage for numerical stability
    - Compute geodesic distances using Mahalanobis distance
    - Track geodesic deviation from the learned "healthy" trajectory

    State vectors are numpy arrays of configurable dimension. The manifold
    adapts to whatever dimensionality the upstream state constructor provides.
    """

    def __init__(
        self,
        window_size: int = 2000,
        baseline_capacity: int = 5000,
        calibration_threshold: int = 1000,
        min_samples_for_fisher: int = 50,
        deviation_history_size: int = 5000,
    ) -> None:
        # Rolling window of recent state vectors for current Fisher estimation
        self._window: deque[np.ndarray] = deque(maxlen=window_size)

        # Healthy baseline trajectory — frozen after calibration
        self._baseline: deque[np.ndarray] = deque(maxlen=baseline_capacity)
        self._baseline_locked: bool = False
        self._calibration_threshold = calibration_threshold

        # Fisher information matrices (precision matrices)
        self._current_fisher: np.ndarray | None = None
        self._baseline_fisher: np.ndarray | None = None

        # Baseline center (mean of baseline vectors)
        self._baseline_center: np.ndarray | None = None

        # Minimum samples before we attempt Fisher estimation
        self._min_samples = min_samples_for_fisher

        # Historical deviation magnitudes for percentile computation
        self._deviation_history: deque[float] = deque(maxlen=deviation_history_size)

        # Dimensionality — inferred from first state vector
        self._dim: int | None = None

        # Cached (N, D) matrix of baseline vectors for vectorized deviation.
        # Invalidated (set to None) whenever the baseline deque is modified.
        self._baseline_matrix: np.ndarray | None = None

        # Ledoit-Wolf estimator (reusable, stateless)
        self._lw = LedoitWolf(assume_centered=False)

    @property
    def dimension(self) -> int | None:
        """Dimensionality of the state vectors, or None if not yet inferred."""
        return self._dim

    @property
    def baseline_locked(self) -> bool:
        return self._baseline_locked

    @property
    def baseline_size(self) -> int:
        return len(self._baseline)

    @property
    def window_size(self) -> int:
        return len(self._window)

    @property
    def has_fisher(self) -> bool:
        return self._current_fisher is not None

    @property
    def window_vectors(self) -> deque[np.ndarray]:
        """Read-only view of the rolling window of state vectors."""
        return self._window

    # ─── Core Update ─────────────────────────────────────────────

    def update(self, state_vector: np.ndarray) -> None:
        """
        Add a new state observation.

        During calibration (baseline not locked), vectors are added to both
        the rolling window and the baseline. After calibration, only the
        rolling window is updated.

        Recomputes the Fisher information matrix from the rolling window.
        """
        vec = np.asarray(state_vector, dtype=np.float64).ravel()

        # Infer dimensionality from first vector; self-reset on expansion
        if self._dim is None:
            self._dim = vec.shape[0]
            logger.info("fisher_manifold_initialized", dimension=self._dim)
        elif vec.shape[0] != self._dim:
            # State space expanded (new systems registered). Reset internal
            # buffers and adopt the new dimensionality rather than discarding
            # every subsequent vector.
            logger.info(
                "fisher_manifold_dimension_reset",
                old=self._dim,
                new=vec.shape[0],
            )
            self._dim = vec.shape[0]
            self._window.clear()
            self._baseline.clear()
            self._baseline_locked = False
            self._current_fisher = None
            self._baseline_fisher = None
            self._baseline_center = None
            self._baseline_matrix = None
            self._deviation_history.clear()

        self._window.append(vec)

        # During calibration, also feed the baseline
        if not self._baseline_locked:
            self._baseline.append(vec)
            if len(self._baseline) >= self._calibration_threshold:
                self.lock_baseline()

        # Recompute Fisher info from rolling window
        if len(self._window) >= self._min_samples:
            self._current_fisher = self._estimate_fisher(self._window)

    def lock_baseline(self) -> None:
        """
        Freeze the baseline trajectory. Called automatically when enough
        calibration vectors have been collected, or manually.
        """
        if self._baseline_locked:
            return

        if len(self._baseline) < self._min_samples:
            logger.warning(
                "baseline_lock_failed",
                reason="insufficient_samples",
                have=len(self._baseline),
                need=self._min_samples,
            )
            return

        baseline_fisher = self._estimate_fisher(self._baseline)
        if baseline_fisher is None:
            logger.warning(
                "baseline_lock_failed",
                reason="fisher_estimation_failed",
                samples=len(self._baseline),
            )
            return

        self._baseline_fisher = baseline_fisher
        self._baseline_center = np.mean(
            np.array(list(self._baseline)), axis=0,
        )
        self._baseline_matrix = None  # invalidate cache
        self._baseline_locked = True

        logger.info(
            "baseline_locked",
            samples=len(self._baseline),
            dimension=self._dim,
        )

    def update_baseline(self, state_vector: np.ndarray) -> None:
        """
        Slow adaptation of the baseline during sleep/recovery.
        Only callable when baseline is locked — adds new vectors and
        recomputes the baseline Fisher matrix.
        """
        if not self._baseline_locked:
            return

        vec = np.asarray(state_vector, dtype=np.float64).ravel()
        if self._dim is not None and vec.shape[0] != self._dim:
            return

        self._baseline.append(vec)
        new_fisher = self._estimate_fisher(self._baseline)
        if new_fisher is not None:
            self._baseline_fisher = new_fisher
        self._baseline_center = np.mean(
            np.array(list(self._baseline)), axis=0,
        )
        self._baseline_matrix = None  # invalidate cache

    # ─── Geodesic Distance ───────────────────────────────────────

    def geodesic_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Geodesic distance between two states using the Fisher-Rao metric.

        d(a, b) = sqrt((a - b)^T F (a - b))

        This is the Mahalanobis distance with Fisher info as precision matrix.
        Falls back to Euclidean distance if Fisher matrix is not yet available.
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()

        fisher = self._current_fisher
        if fisher is None:
            return float(np.linalg.norm(a - b))

        try:
            return float(mahalanobis(a, b, fisher))
        except (ValueError, np.linalg.LinAlgError):
            return float(np.linalg.norm(a - b))

    # ─── Geodesic Deviation ──────────────────────────────────────

    def geodesic_deviation(self, current: np.ndarray) -> GeodesicDeviation | None:
        """
        Measure how far the current state is from the nearest point
        on the baseline trajectory, using the Fisher metric (not Euclidean).

        Uses a vectorized Mahalanobis computation:
            d²(x, b_i) = (x - b_i)^T F (x - b_i)
        over all baseline points at once, avoiding the O(N) Python loop
        that would violate the 5ms inline budget.

        Returns None if baseline is not yet established.
        """
        if not self._baseline_locked or self._baseline_fisher is None:
            return None

        current = np.asarray(current, dtype=np.float64).ravel()
        fisher = self._baseline_fisher

        # Use cached baseline matrix; rebuild if stale (None or shape mismatch).
        if self._baseline_matrix is None or self._baseline_matrix.shape[0] != len(self._baseline):
            self._baseline_matrix = np.array(list(self._baseline))  # (N, D)

        B = self._baseline_matrix  # (N, D)
        diff_all = current[np.newaxis, :] - B  # (N, D)

        # d²_i = diff_i @ F @ diff_i  — batched as (N, D) @ (D, D) = (N, D), then row-dot
        Fdiff = diff_all @ fisher  # (N, D)
        sq_dists = np.einsum("ij,ij->i", Fdiff, diff_all)  # (N,)
        # Clamp numerical negatives from floating-point error
        sq_dists = np.maximum(sq_dists, 0.0)
        dists = np.sqrt(sq_dists)  # (N,)

        nearest_idx = int(np.argmin(dists))
        min_dist = float(dists[nearest_idx])
        nearest_point = B[nearest_idx]

        # Direction vector: normalized difference in state space
        diff = current - nearest_point
        diff_norm = np.linalg.norm(diff)
        direction = diff / diff_norm if diff_norm > 1e-12 else np.zeros_like(diff)

        # Per-dimension contribution to deviation:
        # weighted_diff = F @ diff gives the Fisher-weighted displacement.
        # Contribution of dimension i = |weighted_diff_i * diff_i|
        weighted_diff = fisher @ diff
        contributions = np.abs(weighted_diff * diff)

        # Sort dimension indices by contribution descending
        dominant = np.argsort(contributions)[::-1].tolist()

        # Percentile in historical distribution
        self._deviation_history.append(min_dist)
        if len(self._deviation_history) > 1:
            sorted_hist = sorted(self._deviation_history)
            rank = np.searchsorted(sorted_hist, min_dist, side="right")
            percentile = float(100.0 * rank / len(sorted_hist))
        else:
            percentile = 50.0

        return GeodesicDeviation(
            scalar=min_dist,
            direction=direction,
            dominant_systems=dominant,
            percentile=percentile,
        )

    # ─── Fisher Estimation ───────────────────────────────────────

    def _estimate_fisher(
        self, vectors: deque[np.ndarray],
    ) -> np.ndarray | None:
        """
        Estimate the Fisher information matrix from a collection of state vectors.

        For a Gaussian approximation: F = Sigma_inv (precision matrix).
        Uses Ledoit-Wolf shrinkage for numerical stability — this regularizes
        the covariance estimate when dimensionality is high relative to
        sample count, preventing singular or near-singular matrices.

        Returns the precision matrix, or None on failure.
        """
        if len(vectors) < self._min_samples:
            return None

        X = np.array(list(vectors))  # (n_samples, n_features)

        try:
            self._lw.fit(X)
            precision = self._lw.precision_  # Sigma_inv = Fisher info
            return np.asarray(precision, dtype=np.float64)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("fisher_estimation_failed", error=str(exc))
            return None
