"""
EcodiaOS — Soma Causal Emergence Engine

Measures whether the organism is more than its parts. Effective information
at macro level minus micro level directly quantifies the Coherence
constitutional drive.

Micro-level: event-type to event-type transition probability matrix from
Synapse events.
Macro-level: discretize state vectors into N macro states (KMeans), build
state to state TPM.

EI(X->Y) = log(N) - mean(H(row)) for each TPM.
Causal emergence = macro EI - micro EI.
Positive = coherent organism. Negative = fragmentation.

Runs every ~15s (100 theta cycles). Not on the critical path.

Dependencies: numpy, scikit-learn (KMeans)
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("systems.soma.emergence")

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EmergenceReport:
    """Result of a single causal emergence computation."""

    timestamp: float = 0.0
    micro_ei: float = 0.0  # Effective information at micro level
    macro_ei: float = 0.0  # Effective information at macro level
    causal_emergence: float = 0.0  # macro_ei - micro_ei
    coherence_signal: float = 0.5  # Normalized to [0, 1] for drive system

    # Which macro-state transitions contribute most to emergence
    dominant_transitions: list[tuple[int, int, float]] = field(default_factory=list)

    # Trend tracking
    emergence_velocity: float = 0.0  # d(emergence)/dt
    emergence_trend: str = "stable"  # "increasing", "stable", "declining", "critical"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for signal composition and logging."""
        return {
            "timestamp": self.timestamp,
            "micro_ei": round(self.micro_ei, 6),
            "macro_ei": round(self.macro_ei, 6),
            "causal_emergence": round(self.causal_emergence, 6),
            "coherence_signal": round(self.coherence_signal, 4),
            "emergence_velocity": round(self.emergence_velocity, 6),
            "emergence_trend": self.emergence_trend,
            "dominant_transition_count": len(self.dominant_transitions),
        }


# ---------------------------------------------------------------------------
# Micro-level event tracking
# ---------------------------------------------------------------------------

# Maximum number of distinct event types to track
_MAX_EVENT_TYPES: int = 256

# Laplace smoothing for TPM rows (prevents zero-entropy rows)
_LAPLACE_ALPHA: float = 1e-6


def _row_entropy(row: np.ndarray) -> float:
    """Shannon entropy of a single probability row (base-2)."""
    nonzero = row[row > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _effective_information(tpm: np.ndarray) -> float:
    """
    Compute effective information for a transition probability matrix.

    EI = log2(N) - mean_i(H(TPM[i, :]))

    N = number of states (columns).
    H(TPM[i,:]) = Shannon entropy of the i-th row.
    """
    n = tpm.shape[1]
    if n == 0:
        return 0.0

    log_n = math.log2(n)
    row_entropies = np.array([_row_entropy(tpm[i]) for i in range(tpm.shape[0])])
    mean_h = float(np.mean(row_entropies)) if len(row_entropies) > 0 else 0.0

    return max(0.0, log_n - mean_h)


# ---------------------------------------------------------------------------
# CausalEmergenceEngine
# ---------------------------------------------------------------------------


class CausalEmergenceEngine:
    """
    Computes effective information at micro and macro levels to quantify
    organismal coherence.

    The micro level tracks event-type to event-type transitions from Synapse
    events. The macro level discretizes state vectors into N macro states
    via KMeans and builds a state to state TPM.

    Produces ``coherence_signal`` in [0, 1] for the Coherence constitutional
    drive in Thymos.
    """

    def __init__(
        self,
        n_macro_states: int = 32,
        micro_history_size: int = 5000,
        macro_history_size: int = 2000,
        compute_interval_cycles: int = 100,
    ) -> None:
        self._n_macro_states = n_macro_states

        # -- Micro-level tracking --
        self._event_type_to_idx: dict[str, int] = {}
        self._next_event_idx: int = 0
        self._last_micro_idx: int | None = None
        self._micro_counts: np.ndarray = np.zeros(
            (_MAX_EVENT_TYPES, _MAX_EVENT_TYPES), dtype=np.float64,
        )
        self._micro_total_transitions: int = 0

        # -- Macro-level tracking --
        self._macro_buffer: deque[np.ndarray] = deque(maxlen=macro_history_size)
        self._macro_vector_dim: int | None = None  # expected dim; None until first observation
        self._quantizer: Any = None  # sklearn KMeans instance
        self._quantizer_fitted: bool = False
        self._quantizer_lock = threading.Lock()
        self._last_macro_idx: int | None = None
        self._macro_counts: np.ndarray = np.zeros(
            (n_macro_states, n_macro_states), dtype=np.float64,
        )
        self._macro_total_transitions: int = 0
        self._min_fit_samples: int = max(n_macro_states * 4, 128)

        # -- Emergence tracking --
        self._emergence_history: deque[float] = deque(maxlen=100)
        self._last_report: EmergenceReport | None = None
        self._cycle_counter: int = 0
        self._compute_interval = compute_interval_cycles

    # -- Public API --------------------------------------------------------

    def observe_micro(self, event_type: str) -> None:
        """
        Record a micro-level event transition. Called on every Synapse event.

        Each event contributes one transition: previous_type to current_type.
        """
        idx = self._get_or_create_event_idx(event_type)
        if idx is None:
            return  # Exceeded max event types

        if self._last_micro_idx is not None:
            self._micro_counts[self._last_micro_idx, idx] += 1.0
            self._micro_total_transitions += 1

        self._last_micro_idx = idx

    def observe_macro(self, state_vector: np.ndarray) -> None:
        """
        Record a macro-level state observation. Called every theta cycle.

        Buffers the state vector. If the quantizer is fitted, also records
        the macro-level transition.
        """
        dim = state_vector.shape[0] if state_vector.ndim == 1 else state_vector.size
        if self._macro_vector_dim is None:
            self._macro_vector_dim = dim
        elif dim != self._macro_vector_dim:
            # Dimension expanded (new systems registered) — discard stale buffer
            # and reset quantizer so it retrains on homogeneous data.
            self._macro_buffer.clear()
            self._macro_vector_dim = dim
            with self._quantizer_lock:
                self._quantizer = None
                self._quantizer_fitted = False
            self._last_macro_idx = None
            self._macro_counts = np.zeros(
                (self._n_macro_states, self._n_macro_states), dtype=np.float64,
            )
            self._macro_total_transitions = 0

        self._macro_buffer.append(state_vector)
        self._cycle_counter += 1

        if not self._quantizer_fitted:
            if len(self._macro_buffer) >= self._min_fit_samples:
                self._fit_quantizer()
            return

        macro_idx = self._quantize(state_vector)
        if macro_idx is not None and self._last_macro_idx is not None:
            self._macro_counts[self._last_macro_idx, macro_idx] += 1.0
            self._macro_total_transitions += 1
        self._last_macro_idx = macro_idx

    def should_compute(self) -> bool:
        """Whether enough cycles have passed for a new computation."""
        return self._cycle_counter >= self._compute_interval

    def compute_emergence(self) -> EmergenceReport:
        """
        Compute effective information at micro and macro levels.

        Returns an EmergenceReport with the causal emergence value and
        the normalized coherence_signal for the constitutional drive.
        """
        self._cycle_counter = 0
        ts = time.monotonic()

        micro_ei = self._compute_micro_ei()
        macro_ei = self._compute_macro_ei()
        emergence = macro_ei - micro_ei

        # Normalize coherence signal to [0, 1] via sigmoid
        coherence = 1.0 / (1.0 + math.exp(-emergence))

        self._emergence_history.append(emergence)
        velocity = self._compute_velocity()
        trend = self._classify_trend(emergence, velocity)
        dominant = self._find_dominant_transitions()

        report = EmergenceReport(
            timestamp=ts,
            micro_ei=micro_ei,
            macro_ei=macro_ei,
            causal_emergence=emergence,
            coherence_signal=coherence,
            dominant_transitions=dominant,
            emergence_velocity=velocity,
            emergence_trend=trend,
        )

        self._last_report = report

        if abs(emergence) > 0.5 or trend in ("declining", "critical"):
            logger.info(
                "emergence_computed",
                micro_ei=round(micro_ei, 4),
                macro_ei=round(macro_ei, 4),
                emergence=round(emergence, 4),
                coherence=round(coherence, 4),
                trend=trend,
            )

        return report

    @property
    def last_report(self) -> EmergenceReport | None:
        return self._last_report

    @property
    def coherence_signal(self) -> float:
        """Current coherence signal [0, 1]. 0.5 if not yet computed."""
        if self._last_report is not None:
            return self._last_report.coherence_signal
        return 0.5

    @property
    def quantizer_fitted(self) -> bool:
        return self._quantizer_fitted

    # -- Sleep-cycle recalibration -----------------------------------------

    def refit_quantizer(self) -> None:
        """
        Re-fit the macro-state quantizer during Oneiros sleep.

        Resets the macro transition counts but preserves micro transitions
        (which accumulate over the organism's lifetime).
        """
        if len(self._macro_buffer) >= self._min_fit_samples:
            self._fit_quantizer()
            self._macro_counts[:] = 0.0
            self._macro_total_transitions = 0
            self._last_macro_idx = None
            logger.info("emergence_quantizer_refit_during_sleep")

    # -- Internals ---------------------------------------------------------

    def _get_or_create_event_idx(self, event_type: str) -> int | None:
        """Map event type string to index, creating if needed."""
        if event_type in self._event_type_to_idx:
            return self._event_type_to_idx[event_type]
        if self._next_event_idx >= _MAX_EVENT_TYPES:
            return None
        idx = self._next_event_idx
        self._event_type_to_idx[event_type] = idx
        self._next_event_idx += 1
        return idx

    def _compute_micro_ei(self) -> float:
        """Compute effective information at the micro (event) level."""
        n = self._next_event_idx
        if n < 2 or self._micro_total_transitions < 10:
            return 0.0
        tpm = self._counts_to_tpm(self._micro_counts[:n, :n])
        return _effective_information(tpm)

    def _compute_macro_ei(self) -> float:
        """Compute effective information at the macro (system-state) level."""
        n = self._n_macro_states
        if not self._quantizer_fitted or self._macro_total_transitions < 10:
            return 0.0
        tpm = self._counts_to_tpm(self._macro_counts[:n, :n])
        return _effective_information(tpm)

    @staticmethod
    def _counts_to_tpm(counts: np.ndarray) -> np.ndarray:
        """Convert transition count matrix to probability matrix with Laplace smoothing."""
        smoothed = counts + _LAPLACE_ALPHA
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return np.asarray(smoothed / row_sums)

    def _fit_quantizer(self) -> None:
        """Fit KMeans quantizer on buffered state vectors."""
        try:
            from sklearn.cluster import KMeans  # type: ignore[import-untyped]

            data = np.array(list(self._macro_buffer), dtype=np.float64)
            n_clusters = min(self._n_macro_states, len(data))
            if n_clusters < 2:
                return

            with self._quantizer_lock:
                km = KMeans(
                    n_clusters=n_clusters,
                    n_init=3,
                    max_iter=50,
                    random_state=42,
                )
                km.fit(data)
                self._quantizer = km
                self._quantizer_fitted = True

                if n_clusters != self._n_macro_states:
                    self._n_macro_states = n_clusters
                    self._macro_counts = np.zeros(
                        (n_clusters, n_clusters), dtype=np.float64,
                    )
                    self._macro_total_transitions = 0
                    self._last_macro_idx = None

            logger.info("emergence_quantizer_fitted", n_clusters=n_clusters, samples=len(data))

        except Exception as exc:
            logger.warning("emergence_quantizer_fit_failed", error=str(exc))

    def _quantize(self, state_vector: np.ndarray) -> int | None:
        """Quantize a state vector to its nearest macro state."""
        with self._quantizer_lock:
            if self._quantizer is None:
                return None
            try:
                label: int = int(self._quantizer.predict(state_vector.reshape(1, -1))[0])
                return label
            except Exception:
                return None

    def _compute_velocity(self) -> float:
        """Compute d(emergence)/dt from recent history."""
        if len(self._emergence_history) < 3:
            return 0.0
        recent = list(self._emergence_history)
        window = min(5, len(recent))
        vals = recent[-window:]
        if len(vals) < 2:
            return 0.0
        return (vals[-1] - vals[0]) / max(1, len(vals) - 1)

    @staticmethod
    def _classify_trend(emergence: float, velocity: float) -> str:
        """Classify the emergence trend."""
        if emergence < -1.0:
            return "critical"
        if velocity < -0.05:
            return "declining"
        if velocity > 0.05:
            return "increasing"
        return "stable"

    def _find_dominant_transitions(self) -> list[tuple[int, int, float]]:
        """Find the macro transitions that contribute most to macro EI."""
        if not self._quantizer_fitted or self._macro_total_transitions < 10:
            return []

        n = self._n_macro_states
        tpm = self._counts_to_tpm(self._macro_counts[:n, :n])
        log_n = math.log2(n) if n > 0 else 0.0

        contributions: list[tuple[int, int, float]] = []
        for i in range(n):
            row_h = _row_entropy(tpm[i])
            row_contribution = (log_n - row_h) / n
            if row_contribution > 0.01:
                j = int(np.argmax(tpm[i]))
                contributions.append((i, j, round(row_contribution, 6)))

        contributions.sort(key=lambda x: x[2], reverse=True)
        return contributions[:10]
