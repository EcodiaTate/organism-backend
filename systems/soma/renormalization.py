"""
EcodiaOS -- Soma Renormalization Engine

Examines the signal stream at multiple time scales simultaneously to detect
where self-similarity breaks.

Break interpretation:
  0.1s to 1s  => function_level      (deadlocks, hot loops)
  1s to 100s  => system_interaction  (feedback instability, resource starvation)
  100s+       => drift               (constitutional erosion, economic decline)

Dependencies: numpy only. No LLM, no DB, no network.
Budget: periodic (medium path, every ~100 cycles / 15s).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import structlog

logger = structlog.get_logger("systems.soma.renormalization")


SCALES: list[float] = [0.1, 1.0, 10.0, 100.0, 1000.0]

_STATS_BUFFER_SIZE = 200
_MIN_SAMPLES_FOR_OPERATOR = 30
_MAX_FIXED_POINTS = 8
_CENTROID_LR = 0.05
_FIXED_POINT_MERGE_RADIUS = 0.10
_BREAK_THRESHOLD = 0.20
_REFIT_INTERVAL = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ScaleWindow(NamedTuple):
    """Time-stamped observation for routing into scale windows."""

    timestamp: float           # Unix time (seconds)
    state_vector: np.ndarray   # 1D float array, shape (n_dims,)
    event_type: str            # e.g. synapse_event, nova_decision


@dataclass
class ScaleStatistics:
    """Statistics computed over one rolling window at one scale."""

    scale: float
    window_start: float
    window_end: float
    n_samples: int
    mean: np.ndarray                 # shape (n_dims,)
    variance: np.ndarray             # shape (n_dims,)
    correlation_matrix: np.ndarray   # shape (n_dims, n_dims)
    event_entropy: float             # Shannon H over event-type histogram
    spectral_energy: np.ndarray      # shape (n_freq_bins,), normalised PSD

    def to_feature_vector(self) -> np.ndarray:
        """Flatten to fixed-size 1D vector for operator fitting.
        Layout: [mean | variance | upper-triangle correlations | event_entropy | spectral]
        """
        n = len(self.mean)
        triu_i, triu_j = np.triu_indices(n, k=1)
        corr_vec = self.correlation_matrix[triu_i, triu_j]
        return np.concatenate([
            self.mean,
            self.variance,
            corr_vec,
            [self.event_entropy],
            self.spectral_energy,
        ])


@dataclass
class RGFixedPoint:
    """A stable operating mode discovered in coarse-scale statistics space."""

    center: np.ndarray    # centroid in feature-vector space
    stability: float      # in [0, 1]; higher = more stable
    basin_size: float     # approximate attraction radius
    label: str            # heuristic: "active", "sleeping", "economic"
    n_visits: int = 0
    first_seen: float = field(default_factory=time.time)


@dataclass
class RGFlowReport:
    """Output of one renormalization group analysis run."""

    timestamp: float
    # Self-similarity score per scale transition: (fine_s, coarse_s) -> [0, 1]
    similarity_breaks: dict[tuple[float, float], float]
    anomaly_scale: float | None
    # "function_level" | "system_interaction" | "drift"
    anomaly_scale_interpretation: str
    fixed_points: list[RGFixedPoint]
    fixed_point_drift: float           # mean centroid displacement since last check
    scale_statistics: dict[float, ScaleStatistics | None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * float(np.log(p))
    return h


def _compute_psd(series: np.ndarray, n_freq_bins: int = 16) -> np.ndarray:
    """Estimate normalised PSD via FFT with Hann window."""
    n = len(series)
    if n < 4:
        return np.zeros(n_freq_bins)
    window = np.hanning(n)
    fft_vals = np.fft.rfft(series * window)
    psd = (np.abs(fft_vals) ** 2) / n
    n_fft = len(psd)
    bin_size = max(1, n_fft // n_freq_bins)
    binned = np.array([
        float(np.sum(psd[i * bin_size: (i + 1) * bin_size]))
        for i in range(n_freq_bins)
    ])
    total = binned.sum()
    if total > 1e-12:
        binned /= total
    return binned


def compute_scale_statistics(
    observations: list[ScaleWindow],
    scale: float,
    n_freq_bins: int = 16,
) -> ScaleStatistics | None:
    """Compute ScaleStatistics from observations. Returns None if < 3 samples."""
    if len(observations) < 3:
        return None

    timestamps = [o.timestamp for o in observations]
    vectors = np.array([o.state_vector for o in observations], dtype=float)   # (N, D)

    event_counts: dict[str, int] = {}
    for o in observations:
        event_counts[o.event_type] = event_counts.get(o.event_type, 0) + 1

    mean = vectors.mean(axis=0)
    variance = vectors.var(axis=0)

    n_dims = vectors.shape[1]
    corr = np.eye(n_dims)
    stds = np.sqrt(variance)
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            if stds[i] > 1e-9 and stds[j] > 1e-9:
                c = float(np.corrcoef(vectors[:, i], vectors[:, j])[0, 1])
                corr[i, j] = c
                corr[j, i] = c

    norms = np.linalg.norm(vectors, axis=1)
    spectral = _compute_psd(norms, n_freq_bins=n_freq_bins)

    return ScaleStatistics(
        scale=scale,
        window_start=timestamps[0],
        window_end=timestamps[-1],
        n_samples=len(observations),
        mean=mean,
        variance=variance,
        correlation_matrix=corr,
        event_entropy=_shannon_entropy(event_counts),
        spectral_energy=spectral,
    )


# ---------------------------------------------------------------------------
# Coarse-Graining Operator
# ---------------------------------------------------------------------------


class CoarseGrainingOperator:
    """
    Linear map W: fine_feat -> coarse_feat, fitted via least squares.

    Self-similarity score = ||W @ fine - actual_coarse|| / ||actual_coarse||
    in [0, 1]. Score 0 = perfect self-similarity. Score 1 = no correlation (anomaly).
    """

    def __init__(self) -> None:
        self._W: np.ndarray | None = None
        self._fine_buf: list[np.ndarray] = []
        self._coarse_buf: list[np.ndarray] = []
        self._fitted = False
        self._residual_history: deque[float] = deque(maxlen=50)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def add_sample(self, fine_feat: np.ndarray, coarse_feat: np.ndarray) -> None:
        self._fine_buf.append(fine_feat.copy())
        self._coarse_buf.append(coarse_feat.copy())
        if len(self._fine_buf) > _STATS_BUFFER_SIZE:
            self._fine_buf.pop(0)
            self._coarse_buf.pop(0)

    def fit(self) -> bool:
        """Fit via least squares. Returns True on success."""
        if len(self._fine_buf) < _MIN_SAMPLES_FOR_OPERATOR:
            return False
        X = np.array(self._fine_buf)    # (N, D)
        Y = np.array(self._coarse_buf)  # (N, D)
        try:
            # Solve X W_T = Y; W_T shape (D, D), W = W_T.T
            W_T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            self._W = W_T.T
            self._fitted = True
            return True
        except np.linalg.LinAlgError:
            return False

    def predict_similarity_break(
        self,
        fine_feat: np.ndarray,
        actual_coarse_feat: np.ndarray,
    ) -> float:
        """Return relative residual norm in [0, 1]. 0.0 if operator not yet fitted."""
        if not self._fitted or self._W is None:
            return 0.0
        predicted = self._W @ fine_feat
        norm_actual = float(np.linalg.norm(actual_coarse_feat))
        if norm_actual < 1e-9:
            residual = 0.0
        else:
            residual = float(np.linalg.norm(predicted - actual_coarse_feat) / norm_actual)
        clipped = float(np.clip(residual, 0.0, 1.0))
        self._residual_history.append(clipped)
        return clipped

    @property
    def mean_residual(self) -> float:
        if not self._residual_history:
            return 0.0
        return float(np.mean(list(self._residual_history)))


# ---------------------------------------------------------------------------
# Fixed Point Tracker (online k-means)
# ---------------------------------------------------------------------------


def _infer_label(centre: np.ndarray, n_dims: int) -> str:
    mean_val = float(centre[:n_dims].mean()) if len(centre) >= n_dims else float(centre.mean())
    if mean_val > 0.65:
        return "active"
    if mean_val < 0.25:
        return "sleeping"
    return "economic"


class FixedPointTracker:
    """Online k-means over coarse-scale feature vectors to track stable operating modes."""

    def __init__(self, max_centres: int = _MAX_FIXED_POINTS) -> None:
        self._centres: list[np.ndarray] = []
        self._spreads: list[float] = []
        self._visits: list[int] = []
        self._max_centres = max_centres
        self._prev_centres: list[np.ndarray] = []

    def observe(self, feat: np.ndarray) -> None:
        """Update clustering with a new coarse-scale feature vector."""
        if not self._centres:
            self._centres.append(feat.copy())
            self._spreads.append(0.0)
            self._visits.append(1)
            return

        dists = [float(np.linalg.norm(feat - c)) for c in self._centres]
        nearest_idx = int(np.argmin(dists))
        nearest_dist = dists[nearest_idx]

        if nearest_dist < _FIXED_POINT_MERGE_RADIUS or len(self._centres) >= self._max_centres:
            lr = _CENTROID_LR
            self._centres[nearest_idx] = (1 - lr) * self._centres[nearest_idx] + lr * feat
            self._spreads[nearest_idx] = (1 - lr) * self._spreads[nearest_idx] + lr * nearest_dist
            self._visits[nearest_idx] += 1
        else:
            self._centres.append(feat.copy())
            self._spreads.append(0.0)
            self._visits.append(1)

    def fixed_points(self, n_dims: int) -> list[RGFixedPoint]:
        fps: list[RGFixedPoint] = []
        for centre, spread, visits in zip(self._centres, self._spreads, self._visits, strict=False):
            stability = float(np.clip(
                1.0 - spread / max(_FIXED_POINT_MERGE_RADIUS, 1e-6), 0.0, 1.0
            ))
            fps.append(RGFixedPoint(
                center=centre.copy(),
                stability=stability,
                basin_size=max(float(spread * 1.5), 0.05),
                label=_infer_label(centre, n_dims),
                n_visits=visits,
            ))
        return fps

    def compute_drift(self) -> float:
        """Mean L2 displacement of centres since last call. Updates snapshot."""
        if not self._prev_centres or len(self._prev_centres) != len(self._centres):
            self._prev_centres = [c.copy() for c in self._centres]
            return 0.0
        drifts = [
            float(np.linalg.norm(self._centres[i] - self._prev_centres[i]))
            for i in range(len(self._centres))
        ]
        self._prev_centres = [c.copy() for c in self._centres]
        return float(np.mean(drifts)) if drifts else 0.0


# ---------------------------------------------------------------------------
# Anomaly interpretation
# ---------------------------------------------------------------------------


def _interpret_anomaly_scale(scale_s: float) -> str:
    if scale_s < 1.0:
        return "function_level"
    if scale_s < 100.0:
        return "system_interaction"
    return "drift"


# ---------------------------------------------------------------------------
# RenormalizationEngine
# ---------------------------------------------------------------------------


class RenormalizationEngine:
    """
    Multi-scale self-similarity analyser for EcodiaOS Soma.

    Ingestion:   call observe(timestamp, state_vector, event_type) per signal event.
    Analysis:    call compute_rg_flow() on the medium path (~100 cycles / 15s).
    Calibration: coarse-graining operators are fitted lazily after
                 _MIN_SAMPLES_FOR_OPERATOR paired samples; force-fit via calibrate()
                 after Oneiros sleep when extended data has accumulated.
    """

    def __init__(
        self,
        scales: list[float] | None = None,
        n_freq_bins: int = 16,
        refit_interval: int = _REFIT_INTERVAL,
    ) -> None:
        self._scales = scales or SCALES
        self._n_freq_bins = n_freq_bins
        self._refit_interval = refit_interval

        self._windows: dict[float, deque[ScaleWindow]] = {
            s: deque(maxlen=500) for s in self._scales
        }
        self._stats_history: dict[float, deque] = {
            s: deque(maxlen=_STATS_BUFFER_SIZE) for s in self._scales
        }
        self._scale_pairs = list(zip(self._scales[:-1], self._scales[1:], strict=False))
        self._cg_operators: dict[tuple[float, float], CoarseGrainingOperator] = {
            pair: CoarseGrainingOperator() for pair in self._scale_pairs
        }
        self._fp_tracker = FixedPointTracker()
        self._analysis_count = 0
        self._last_report: RGFlowReport | None = None
        self._n_dims: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def observe(
        self,
        timestamp: float,
        state_vector: np.ndarray,
        event_type: str = "generic",
    ) -> None:
        """Route a signal observation to all scale windows."""
        if self._n_dims == 0:
            self._n_dims = len(state_vector)
        sw = ScaleWindow(timestamp=timestamp, state_vector=state_vector, event_type=event_type)
        for scale in self._scales:
            self._windows[scale].append(sw)

    def _window_for_scale(self, scale: float, now: float) -> list[ScaleWindow]:
        cutoff = now - scale
        return [obs for obs in self._windows[scale] if obs.timestamp >= cutoff]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compute_rg_flow(self) -> RGFlowReport:
        """
        Run the full renormalization group analysis. Returns RGFlowReport.

        Steps:
          1. Compute statistics at each scale from current rolling windows.
          2. For each adjacent (fine, coarse) pair: feed sample, measure break.
          3. Refit operators on schedule.
          4. Identify anomaly scale (largest break above threshold).
          5. Update fixed point tracking on coarsest scale.
          6. Compute centroid drift.
          7. Return RGFlowReport.
        """
        now = time.time()
        self._analysis_count += 1

        # 1. Per-scale statistics
        current_stats: dict[float, ScaleStatistics | None] = {}
        for scale in self._scales:
            window = self._window_for_scale(scale, now)
            stats = compute_scale_statistics(window, scale, n_freq_bins=self._n_freq_bins)
            current_stats[scale] = stats
            if stats is not None:
                self._stats_history[scale].append(stats)

        # 2. Paired samples + similarity breaks
        similarity_breaks: dict[tuple[float, float], float] = {}
        for fine_s, coarse_s in self._scale_pairs:
            fine_stats = current_stats.get(fine_s)
            coarse_stats = current_stats.get(coarse_s)
            if fine_stats is None or coarse_stats is None:
                continue
            fine_feat = fine_stats.to_feature_vector()
            coarse_feat = coarse_stats.to_feature_vector()
            op = self._cg_operators[(fine_s, coarse_s)]
            op.add_sample(fine_feat, coarse_feat)
            similarity_breaks[(fine_s, coarse_s)] = op.predict_similarity_break(
                fine_feat, coarse_feat
            )

        # 3. Refit on schedule
        if self._analysis_count % self._refit_interval == 0:
            for op in self._cg_operators.values():
                op.fit()

        # 4. Anomaly scale detection
        anomaly_scale: float | None = None
        anomaly_interpretation = ""
        max_pair: tuple[float, float] = (0.0, 0.0)
        if similarity_breaks:
            max_pair = max(similarity_breaks, key=lambda k: similarity_breaks[k])
            if similarity_breaks[max_pair] > _BREAK_THRESHOLD:
                anomaly_scale = max_pair[0]
                anomaly_interpretation = _interpret_anomaly_scale(anomaly_scale)

        # 5. Fixed point tracking (coarsest scale)
        coarsest_stats = current_stats.get(self._scales[-1])
        if coarsest_stats is not None:
            self._fp_tracker.observe(coarsest_stats.to_feature_vector())

        fixed_points = self._fp_tracker.fixed_points(self._n_dims)
        drift = self._fp_tracker.compute_drift()

        report = RGFlowReport(
            timestamp=now,
            similarity_breaks=similarity_breaks,
            anomaly_scale=anomaly_scale,
            anomaly_scale_interpretation=anomaly_interpretation,
            fixed_points=fixed_points,
            fixed_point_drift=drift,
            scale_statistics=current_stats,
        )
        self._last_report = report

        if anomaly_scale is not None:
            logger.info(
                "rg_flow_anomaly_detected",
                anomaly_scale_s=anomaly_scale,
                interpretation=anomaly_interpretation,
                max_break=similarity_breaks.get(max_pair, 0.0),
                fixed_point_drift=round(drift, 4),
            )

        return report

    @property
    def last_report(self) -> RGFlowReport | None:
        return self._last_report

    def calibrate(self) -> None:
        """Force-fit all operators from accumulated data. Call after Oneiros sleep."""
        fitted = sum(1 for op in self._cg_operators.values() if op.fit())
        logger.info("rg_calibration_complete", operators_fitted=fitted)
