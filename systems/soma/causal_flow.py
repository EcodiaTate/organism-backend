"""
EcodiaOS - Soma Causal Flow Engine (Transfer Entropy)

Computes directed causal influence between all system pairs using transfer
entropy. Detects when information flow deviates from the expected
architectural topology.

Transfer entropy (Schreiber, 2000):
    TE(X->Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)
    = how much does knowing X's recent past reduce uncertainty about
      Y's next state, beyond what Y's own past tells you.

Uses discretized entropy estimation for speed: bin continuous values into
discrete bins, then compute conditional entropies from count tables.

Runs every ~15s (100 theta cycles), parallelized across system pairs.

Dependencies: numpy
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("systems.soma.causal_flow")


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CausalAnomaly:
    """A deviation from the expected causal topology."""

    source_system: str
    target_system: str
    expected_te: float
    actual_te: float
    deviation_sigma: float  # How many sigma from expected
    interpretation: str  # Human-readable diagnosis


@dataclass(slots=True)
class CausalFlowMap:
    """Result of a single transfer entropy computation across all system pairs."""

    timestamp: float = 0.0

    # The directed causal influence graph: te_matrix[i][j] = TE from i to j
    te_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    system_ids: list[str] = field(default_factory=list)

    # Deviations from expected topology
    unexpected_influences: list[CausalAnomaly] = field(default_factory=list)
    missing_influences: list[CausalAnomaly] = field(default_factory=list)
    reversed_influences: list[CausalAnomaly] = field(default_factory=list)

    # Derivative: which relationships are forming/dissolving
    forming_links: list[tuple[str, str, float]] = field(default_factory=list)
    dissolving_links: list[tuple[str, str, float]] = field(default_factory=list)

    # Systems ordered by net causal outflow
    causal_hierarchy: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for signal composition and logging."""
        return {
            "timestamp": self.timestamp,
            "system_count": len(self.system_ids),
            "unexpected_count": len(self.unexpected_influences),
            "missing_count": len(self.missing_influences),
            "reversed_count": len(self.reversed_influences),
            "forming_links": len(self.forming_links),
            "dissolving_links": len(self.dissolving_links),
            "hierarchy_top_3": self.causal_hierarchy[:3],
        }


# ---------------------------------------------------------------------------
# Expected causal topology (from EcodiaOS architecture)
# ---------------------------------------------------------------------------

# (source, target) -> expected TE strength classification
# "strong": high expected TE (tight causal coupling)
# "moderate": medium expected TE
# "weak": low expected TE

EXPECTED_TOPOLOGY: dict[tuple[str, str], str] = {
    ("atune", "nova"): "strong",
    ("nova", "axon"): "strong",
    ("nova", "thread"): "moderate",
    ("thymos", "nova"): "strong",
    ("soma", "thymos"): "strong",
    ("soma", "nova"): "moderate",
    ("evo", "nova"): "moderate",
    ("evo", "axon"): "moderate",
    ("evo", "soma"): "weak",
    ("synapse", "nova"): "moderate",
    ("synapse", "atune"): "moderate",
    ("synapse", "soma"): "moderate",
    ("axon", "atune"): "moderate",
    ("oikos", "soma"): "moderate",
    ("thread", "nova"): "moderate",
    ("eis", "axon"): "moderate",
    ("equor", "axon"): "moderate",
    ("oneiros", "soma"): "weak",
    ("voxis", "atune"): "weak",
}

# Strength thresholds for TE values
_TE_STRONG_THRESHOLD: float = 0.15
_TE_MODERATE_THRESHOLD: float = 0.05
_TE_WEAK_THRESHOLD: float = 0.02
_TE_NOISE_FLOOR: float = 0.005

# Number of bins for discretizing continuous time series
_N_BINS: int = 8


# ---------------------------------------------------------------------------
# Discretized entropy helpers
# ---------------------------------------------------------------------------


def _discretize(series: np.ndarray, n_bins: int = _N_BINS) -> np.ndarray:
    """Discretize a continuous time series into integer bins."""
    if len(series) == 0:
        return np.array([], dtype=np.int32)

    mn, mx = float(np.min(series)), float(np.max(series))
    if mx - mn < 1e-12:
        return np.zeros(len(series), dtype=np.int32)

    binned = np.clip(
        ((series - mn) / (mx - mn) * (n_bins - 1)).astype(np.int32),
        0,
        n_bins - 1,
    )
    return binned


def _conditional_entropy(
    target: np.ndarray,
    *conditions: np.ndarray,
    n_bins: int = _N_BINS,
) -> float:
    """
    Compute H(target | conditions) using discrete bin counts.

    All arrays must be integer-valued (discretized) and same length.
    """
    if len(target) == 0:
        return 0.0

    n = len(target)

    # Build joint count table using integer hashing
    joint: dict[tuple[int, ...], np.ndarray] = {}

    for i in range(n):
        cond_key = tuple(int(c[i]) for c in conditions)
        if cond_key not in joint:
            joint[cond_key] = np.zeros(n_bins, dtype=np.float64)
        joint[cond_key][int(target[i])] += 1.0

    # H(target | conditions) = sum_c P(c) * H(target | c)
    total_h = 0.0
    for counts in joint.values():
        total = counts.sum()
        if total < 1:
            continue
        p = counts / total
        p_nonzero = p[p > 0]
        h = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
        total_h += (total / n) * h

    return total_h


def _transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag_k: int = 1,
    n_bins: int = _N_BINS,
) -> float:
    """
    Compute transfer entropy TE(source -> target) using discretized bins.

    TE(X->Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)

    Uses lag_k steps of history. Higher TE = stronger causal influence.
    """
    n = min(len(source), len(target))
    if n <= lag_k + 1:
        return 0.0

    src_d = _discretize(source[:n], n_bins)
    tgt_d = _discretize(target[:n], n_bins)

    # Build lagged arrays
    target_future = tgt_d[lag_k:]
    target_past = tgt_d[lag_k - 1 : -1] if lag_k > 0 else tgt_d[:len(target_future)]
    source_past = src_d[lag_k - 1 : -1] if lag_k > 0 else src_d[:len(target_future)]

    min_len = min(len(target_future), len(target_past), len(source_past))
    target_future = target_future[:min_len]
    target_past = target_past[:min_len]
    source_past = source_past[:min_len]

    if min_len < 10:
        return 0.0

    h_y_given_ypast = _conditional_entropy(target_future, target_past, n_bins=n_bins)
    h_y_given_ypast_xpast = _conditional_entropy(
        target_future, target_past, source_past, n_bins=n_bins,
    )

    te = max(0.0, h_y_given_ypast - h_y_given_ypast_xpast)
    return te


# ---------------------------------------------------------------------------
# CausalFlowEngine
# ---------------------------------------------------------------------------


class CausalFlowEngine:
    """
    Computes transfer entropy between all system pairs to map the
    real-time directed causal influence topology.

    Each system's summary time series (scalar: call_rate or a
    principal-component projection of its state slice) is buffered.
    Every ``compute_interval_cycles`` theta cycles, TE is computed
    for all pairs and compared to the expected architecture.

    Anomalies are reported when:
    - An unexpected link appears (high TE where none expected)
    - An expected link weakens (low TE where strong expected)
    - A link reverses direction (TE(A->B) was > TE(B->A), now opposite)
    """

    def __init__(
        self,
        history_length: int = 200,
        lag_k: int = 5,
        compute_interval_cycles: int = 100,
        n_bins: int = _N_BINS,
    ) -> None:
        self._history_length = history_length
        self._lag_k = lag_k
        self._compute_interval = compute_interval_cycles
        self._n_bins = n_bins

        # Per-system time series buffers
        self._system_series: dict[str, deque[float]] = {}
        self._system_ids: list[str] = []  # Stable ordering

        # Expected topology
        self._expected: dict[tuple[str, str], str] = dict(EXPECTED_TOPOLOGY)

        # Previous TE matrix (for detecting forming/dissolving links)
        self._prev_te_matrix: np.ndarray | None = None
        self._prev_system_ids: list[str] = []

        # Last result
        self._last_map: CausalFlowMap | None = None
        self._cycle_counter: int = 0

    # -- Public API --------------------------------------------------------

    def push_system_value(self, system_id: str, value: float) -> None:
        """
        Buffer a scalar summary value for a system. Called every theta cycle.

        The value should be a representative scalar (e.g. call_rate,
        success_rate, or a principal component of the system's state slice).
        """
        if system_id not in self._system_series:
            self._system_series[system_id] = deque(maxlen=self._history_length)
            self._system_ids = sorted(self._system_series.keys())

        self._system_series[system_id].append(value)

    def tick(self) -> None:
        """Advance the cycle counter by one. Call once per theta cycle, not once per system."""
        self._cycle_counter += 1

    def should_compute(self) -> bool:
        """Whether enough cycles have passed for a new computation."""
        return self._cycle_counter >= self._compute_interval

    def compute_causal_flow(self) -> CausalFlowMap:
        """
        Compute transfer entropy between all system pairs.

        Returns a CausalFlowMap with the TE matrix, anomalies, forming/
        dissolving links, and causal hierarchy.
        """
        self._cycle_counter = 0
        ts = time.monotonic()

        systems = list(self._system_ids)
        n = len(systems)

        if n < 2:
            return CausalFlowMap(timestamp=ts, system_ids=systems)

        te_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            src_buf = self._system_series.get(systems[i])
            if src_buf is None or len(src_buf) < self._lag_k + 10:
                continue
            src = np.array(list(src_buf), dtype=np.float64)

            for j in range(n):
                if i == j:
                    continue
                tgt_buf = self._system_series.get(systems[j])
                if tgt_buf is None or len(tgt_buf) < self._lag_k + 10:
                    continue
                tgt = np.array(list(tgt_buf), dtype=np.float64)

                min_len = min(len(src), len(tgt))
                te_matrix[i, j] = _transfer_entropy(
                    src[-min_len:],
                    tgt[-min_len:],
                    lag_k=self._lag_k,
                    n_bins=self._n_bins,
                )

        unexpected, missing, reversed_links = self._detect_anomalies(
            te_matrix, systems,
        )
        forming, dissolving = self._detect_link_dynamics(te_matrix, systems)
        hierarchy = self._compute_hierarchy(te_matrix, systems)

        flow_map = CausalFlowMap(
            timestamp=ts,
            te_matrix=te_matrix,
            system_ids=systems,
            unexpected_influences=unexpected,
            missing_influences=missing,
            reversed_influences=reversed_links,
            forming_links=forming,
            dissolving_links=dissolving,
            causal_hierarchy=hierarchy,
        )

        self._prev_te_matrix = te_matrix.copy()
        self._prev_system_ids = list(systems)
        self._last_map = flow_map

        if unexpected or missing or reversed_links:
            logger.info(
                "causal_flow_anomalies",
                unexpected=len(unexpected),
                missing=len(missing),
                reversed=len(reversed_links),
                hierarchy_top=hierarchy[:3],
            )

        return flow_map

    @property
    def last_map(self) -> CausalFlowMap | None:
        return self._last_map

    @property
    def system_count(self) -> int:
        return len(self._system_ids)

    # -- Internals ---------------------------------------------------------

    def _detect_anomalies(
        self,
        te_matrix: np.ndarray,
        systems: list[str],
    ) -> tuple[list[CausalAnomaly], list[CausalAnomaly], list[CausalAnomaly]]:
        """Compare TE matrix against expected topology."""
        unexpected: list[CausalAnomaly] = []
        missing: list[CausalAnomaly] = []
        reversed_links: list[CausalAnomaly] = []

        n = len(systems)
        sys_to_idx = {s: i for i, s in enumerate(systems)}

        # Check all expected links
        for (src, tgt), strength in self._expected.items():
            i = sys_to_idx.get(src)
            j = sys_to_idx.get(tgt)
            if i is None or j is None:
                continue

            actual_te = te_matrix[i, j]
            expected_te = self._strength_to_threshold(strength)

            # Check if expected link is missing/weakened
            if actual_te < expected_te * 0.3:
                deviation = (expected_te - actual_te) / max(expected_te, 0.01)
                missing.append(CausalAnomaly(
                    source_system=src,
                    target_system=tgt,
                    expected_te=expected_te,
                    actual_te=actual_te,
                    deviation_sigma=deviation,
                    interpretation=(
                        f"Expected {strength} influence {src}->{tgt} "
                        f"weakened to {actual_te:.4f} (expected >={expected_te:.4f})"
                    ),
                ))

            # Check for reversal
            reverse_te = te_matrix[j, i]
            if reverse_te > actual_te * 2.0 and reverse_te > _TE_MODERATE_THRESHOLD:
                reversed_links.append(CausalAnomaly(
                    source_system=src,
                    target_system=tgt,
                    expected_te=expected_te,
                    actual_te=actual_te,
                    deviation_sigma=(reverse_te - actual_te) / max(expected_te, 0.01),
                    interpretation=(
                        f"Reversed: {tgt}->{src} ({reverse_te:.4f}) "
                        f"now stronger than {src}->{tgt} ({actual_te:.4f})"
                    ),
                ))

        # Check for unexpected strong links
        expected_pairs = set(self._expected.keys())
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pair = (systems[i], systems[j])
                if pair in expected_pairs:
                    continue
                te_val = te_matrix[i, j]
                if te_val > _TE_STRONG_THRESHOLD:
                    unexpected.append(CausalAnomaly(
                        source_system=systems[i],
                        target_system=systems[j],
                        expected_te=0.0,
                        actual_te=te_val,
                        deviation_sigma=te_val / _TE_MODERATE_THRESHOLD,
                        interpretation=(
                            f"Unexpected strong influence {systems[i]}->{systems[j]} "
                            f"(TE={te_val:.4f}) not in expected architecture"
                        ),
                    ))

        return unexpected, missing, reversed_links

    def _detect_link_dynamics(
        self,
        te_matrix: np.ndarray,
        systems: list[str],
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
        """Detect forming and dissolving links via TE matrix derivative."""
        forming: list[tuple[str, str, float]] = []
        dissolving: list[tuple[str, str, float]] = []

        if self._prev_te_matrix is None or self._prev_system_ids != systems:
            return forming, dissolving

        n = len(systems)
        if self._prev_te_matrix.shape != (n, n):
            return forming, dissolving

        delta = te_matrix - self._prev_te_matrix

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = delta[i, j]
                if d > _TE_WEAK_THRESHOLD:
                    forming.append((systems[i], systems[j], round(float(d), 6)))
                elif d < -_TE_WEAK_THRESHOLD:
                    dissolving.append((systems[i], systems[j], round(float(abs(d)), 6)))

        forming.sort(key=lambda x: x[2], reverse=True)
        dissolving.sort(key=lambda x: x[2], reverse=True)

        return forming[:10], dissolving[:10]

    @staticmethod
    def _compute_hierarchy(
        te_matrix: np.ndarray,
        systems: list[str],
    ) -> list[str]:
        """Order systems by net causal outflow (outgoing TE - incoming TE)."""
        n = len(systems)
        if n == 0:
            return []

        outflow = te_matrix.sum(axis=1)
        inflow = te_matrix.sum(axis=0)
        net = outflow - inflow

        order = np.argsort(-net)
        return [systems[int(i)] for i in order]

    @staticmethod
    def _strength_to_threshold(strength: str) -> float:
        """Map expected strength classification to TE threshold."""
        if strength == "strong":
            return _TE_STRONG_THRESHOLD
        if strength == "moderate":
            return _TE_MODERATE_THRESHOLD
        return _TE_WEAK_THRESHOLD
