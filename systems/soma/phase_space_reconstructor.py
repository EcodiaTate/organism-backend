"""
EcodiaOS — Soma Phase Space Reconstructor

Implements Takens delay-embedding reconstruction for 7 key scalar metrics.
Estimates correlation dimension (Grassberger-Procaccia 1983) and largest
Lyapunov exponent (Rosenstein et al. 1993). Runs on the slow path (~75s).

All algorithms use numpy only — no LLM, no DB, no network.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger("systems.soma.phase_space_reconstructor")

# ─── Configuration ────────────────────────────────────────────────

DEFAULT_TARGET_METRICS: list[str] = [
    "nova.decision_latency_ms",
    "axon.execution_success_rate",
    "oikos.economic_ratio",
    "evo.hypothesis_confidence_mean",
    "synapse.cycle_latency_ms",
    "atune.salience_mean",
    "thymos.drive_pressure_total",
]

_SERIES_BUFFER = 2000        # max samples retained per metric
_AMI_MAX_LAG = 50            # max lag for AMI delay estimation
_FNN_MAX_DIM = 10            # max embedding dimension to test
_FNN_THRESHOLD = 0.01        # FNN% below this => accept dimension
_GP_N_EPS = 30               # number of epsilon values for GP correlation integral
_LLE_N_NEIGHBOURS = 5        # nearest neighbours for Rosenstein LLE
_LLE_N_STEPS = 20            # divergence tracking steps
_MIN_SERIES_LENGTH = 100     # minimum samples before reconstruction
_MAX_PREDICTABILITY_CYCLES = 1000


# ─── Output Types ─────────────────────────────────────────────────

@dataclass
class AttractorDiagnosis:
    """Per-metric reconstruction result."""
    metric: str
    timestamp: float
    embedding_delay: int
    embedding_dimension: int
    correlation_dimension: float
    dimension_trend: str          # "stable" | "increasing" | "collapsing"
    largest_lyapunov: float
    lyapunov_interpretation: str  # "chaotic" | "stable" | "dissipative"
    predictability_horizon_cycles: int


@dataclass
class PhaseSpaceReport:
    """Full reconstruction run result."""
    timestamp: float
    diagnoses: dict[str, AttractorDiagnosis]
    skipped_metrics: list[str]


# ─── AMI Delay Estimation ─────────────────────────────────────────

def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 16) -> float:
    """
    Estimate mutual information I(X;Y) via joint histogram.
    Returns MI in nats (natural log base).
    """
    # Joint 2-D histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_xy / hist_xy.sum()

    # Marginals
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    # Avoid log(0): only compute where joint prob > 0
    mask = p_xy > 0
    mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / (p_x * p_y)[mask]))
    return float(mi)


def estimate_delay(series: np.ndarray, max_lag: int = _AMI_MAX_LAG) -> int:
    """
    Estimate optimal delay tau via Average Mutual Information (Fraser & Swinney 1986).

    Returns the lag at the first local minimum of AMI(lag).
    Falls back to lag=1 if no minimum is found.
    """
    n = len(series)
    ami_values: list[float] = []

    for lag in range(1, min(max_lag + 1, n // 2)):
        x = series[:-lag]
        y = series[lag:]
        ami_values.append(_mutual_information(x, y))

    if len(ami_values) < 2:
        return 1

    # First local minimum
    for i in range(1, len(ami_values) - 1):
        if ami_values[i] < ami_values[i - 1] and ami_values[i] < ami_values[i + 1]:
            return i + 1  # lag is 1-indexed

    # No clear minimum: return lag at global minimum
    return int(np.argmin(ami_values)) + 1


# ─── Delay Embedding ──────────────────────────────────────────────

def delay_embed(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Construct Takens delay-embedding matrix.

    Returns shape (N-(dim-1)*tau, dim) where each row is
    [x(t), x(t+tau), ..., x(t+(dim-1)*tau)].
    """
    n = len(series)
    span = (dim - 1) * tau
    if n <= span:
        return np.empty((0, dim))
    rows = n - span
    indices = np.arange(dim) * tau  # [0, tau, 2*tau, ...]
    idx = np.arange(rows)[:, None] + indices[None, :]
    return series[idx]


# ─── FNN Embedding Dimension ──────────────────────────────────────

def estimate_embedding_dimension(
    series: np.ndarray,
    tau: int,
    max_dim: int = _FNN_MAX_DIM,
    fnn_threshold: float = _FNN_THRESHOLD,
    rtol: float = 10.0,
) -> int:
    """
    Estimate minimum embedding dimension via False Nearest Neighbours
    (Kennel, Brown & Abarbanel 1992).

    Returns smallest d in [1, max_dim] where FNN fraction < fnn_threshold.
    Falls back to max_dim if threshold never met.
    """
    for d in range(1, max_dim):
        embedded_d = delay_embed(series, d, tau)
        embedded_d1 = delay_embed(series, d + 1, tau)

        n = min(len(embedded_d), len(embedded_d1))
        if n < 10:
            return d

        pts_d = embedded_d[:n]
        pts_d1 = embedded_d1[:n]

        false_neighbours = 0
        checked = 0

        # Sample up to 200 points for speed
        sample_idx = np.random.choice(n, min(200, n), replace=False)

        for i in sample_idx:
            # Find nearest neighbour in d dimensions (excluding self)
            dists_d = np.linalg.norm(pts_d - pts_d[i], axis=1)
            dists_d[i] = np.inf
            nn_idx = int(np.argmin(dists_d))
            nn_dist = dists_d[nn_idx]

            if nn_dist < 1e-12:
                continue

            # Check if it becomes a false neighbour in d+1 dimensions
            np.linalg.norm(pts_d1[i] - pts_d1[nn_idx])
            extra = abs(pts_d1[i, -1] - pts_d1[nn_idx, -1])

            if extra / nn_dist > rtol:
                false_neighbours += 1
            checked += 1

        if checked == 0:
            continue

        fnn_fraction = false_neighbours / checked
        if fnn_fraction < fnn_threshold:
            return d + 1

    return max_dim


# ─── Grassberger-Procaccia Correlation Dimension ──────────────────

def estimate_correlation_dimension(
    embedded: np.ndarray,
    n_eps: int = _GP_N_EPS,
    subsample: int = 300,
) -> float:
    """
    Estimate correlation dimension D2 via Grassberger-Procaccia (1983).

    C(eps) = fraction of point pairs within distance eps.
    D2 = slope of log C(eps) vs log eps in the scaling region.
    Returns float in [0, embedding_dim]. Returns 0.0 on failure.
    """
    n = len(embedded)
    if n < 20:
        return 0.0

    # Subsample for speed
    if n > subsample:
        idx = np.random.choice(n, subsample, replace=False)
        pts = embedded[idx]
    else:
        pts = embedded

    n_pts = len(pts)

    # Pairwise distances (upper triangle only)
    dists: list[float] = []
    for i in range(n_pts):
        d = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1)
        dists.extend(d.tolist())

    if not dists:
        return 0.0

    dists_arr = np.array(dists)
    d_min = dists_arr[dists_arr > 0].min() if (dists_arr > 0).any() else 1e-6
    d_max = dists_arr.max()

    if d_max <= d_min:
        return 0.0

    epsilons = np.geomspace(d_min, d_max, n_eps)
    n_pairs = len(dists_arr)

    log_eps: list[float] = []
    log_c: list[float] = []

    for eps in epsilons:
        c = float(np.sum(dists_arr < eps)) / n_pairs
        if c > 0:
            log_eps.append(np.log(eps))
            log_c.append(np.log(c))

    if len(log_eps) < 4:
        return 0.0

    log_eps_arr = np.array(log_eps)
    log_c_arr = np.array(log_c)

    # Use middle 50% of the scaling range for slope estimation
    lo = len(log_eps_arr) // 4
    hi = 3 * len(log_eps_arr) // 4
    if hi - lo < 2:
        lo, hi = 0, len(log_eps_arr)

    coeffs = np.polyfit(log_eps_arr[lo:hi], log_c_arr[lo:hi], 1)
    d2 = float(coeffs[0])
    return max(0.0, d2)


# ─── Rosenstein Largest Lyapunov Exponent ─────────────────────────

def estimate_largest_lyapunov(
    embedded: np.ndarray,
    n_steps: int = _LLE_N_STEPS,
    min_temporal_separation: int = 10,
) -> float:
    """
    Estimate the largest Lyapunov exponent via Rosenstein et al. (1993).

    For each point, find the nearest neighbour with temporal separation
    >= min_temporal_separation, then track the log-divergence over n_steps.
    LLE = slope of <ln d_k(i)> vs k via linear regression.

    Positive LLE => sensitive dependence (chaotic); negative => contracting.
    Returns 0.0 on failure.
    """
    n = len(embedded)
    if n < n_steps + min_temporal_separation + 10:
        return 0.0

    # Subsample for speed — use up to 300 reference points
    ref_limit = min(n - n_steps - min_temporal_separation, 300)
    if ref_limit < 5:
        return 0.0

    ref_indices = np.random.choice(
        n - n_steps - min_temporal_separation, ref_limit, replace=False
    )

    log_divs: np.ndarray = np.zeros(n_steps)
    counts: np.ndarray = np.zeros(n_steps, dtype=int)

    for i in ref_indices:
        # Find nearest neighbour with temporal separation constraint
        nn_idx = -1
        nn_dist = np.inf
        for j in range(n - n_steps):
            if abs(j - i) < min_temporal_separation:
                continue
            d = float(np.linalg.norm(embedded[i] - embedded[j]))
            if d < nn_dist:
                nn_dist = d
                nn_idx = j

        if nn_idx < 0 or nn_dist < 1e-12:
            continue

        # Track divergence
        for k in range(n_steps):
            if i + k >= n or nn_idx + k >= n:
                break
            d_k = float(np.linalg.norm(embedded[i + k] - embedded[nn_idx + k]))
            if d_k > 0:
                log_divs[k] += np.log(d_k)
                counts[k] += 1

    # Average log divergence
    valid = counts > 0
    if valid.sum() < 4:
        return 0.0

    avg_log_div = np.where(valid, log_divs / np.maximum(counts, 1), np.nan)
    steps = np.arange(n_steps)[valid].astype(float)
    vals = avg_log_div[valid]

    # LLE = slope via linear regression
    coeffs = np.polyfit(steps, vals, 1)
    return float(coeffs[0])


# ─── Classification Helpers ───────────────────────────────────────

def _classify_dimension_trend(
    current_dim: float,
    history: list[float],
) -> str:
    """
    Return "increasing", "collapsing", or "stable" based on recent
    dimension estimates (requires >= 3 history points).
    """
    if len(history) < 3:
        return "stable"
    recent = history[-5:]
    if len(recent) < 2:
        return "stable"
    slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
    if slope > 0.1:
        return "increasing"
    if slope < -0.1:
        return "collapsing"
    return "stable"


def _interpret_lyapunov(lle: float) -> str:
    """Map LLE to qualitative label."""
    if lle > 0.02:
        return "chaotic"
    if lle < -0.02:
        return "dissipative"
    return "stable"


def _predictability_horizon(lle: float) -> int:
    """
    Estimate predictability horizon in cycles as ~1/|lambda_1|.
    Capped at _MAX_PREDICTABILITY_CYCLES.
    """
    if abs(lle) < 1e-6:
        return _MAX_PREDICTABILITY_CYCLES
    horizon = int(1.0 / abs(lle))
    return min(horizon, _MAX_PREDICTABILITY_CYCLES)


# ─── PhaseSpaceReconstructor ──────────────────────────────────────

class PhaseSpaceReconstructor:
    """
    Maintains rolling time-series buffers for target metrics and reconstructs
    the phase-space attractor geometry on demand (~75s slow-path cycle).

    Usage:
        psr = PhaseSpaceReconstructor()
        psr.push_metric("nova.decision_latency_ms", 42.3)
        report = psr.reconstruct_all()
    """

    TARGET_METRICS: list[str] = DEFAULT_TARGET_METRICS

    def __init__(
        self,
        target_metrics: list[str] | None = None,
        series_buffer: int = _SERIES_BUFFER,
    ) -> None:
        metrics = target_metrics if target_metrics is not None else self.TARGET_METRICS
        self._buffer_size = series_buffer
        self._series: dict[str, deque[float]] = {
            m: deque(maxlen=series_buffer) for m in metrics
        }
        # Per-metric dimension history for trend detection
        self._dim_history: dict[str, list[float]] = {m: [] for m in metrics}

    # ─── Data Ingestion ───────────────────────────────────────────

    def push_metric(self, metric: str, value: float) -> None:
        """Append a single scalar observation. O(1). Auto-registers unknown metrics."""
        if metric not in self._series:
            self._series[metric] = deque(maxlen=self._buffer_size)
            self._dim_history[metric] = []
        self._series[metric].append(float(value))

    def push_metrics(self, values: dict[str, float]) -> None:
        """Append multiple metrics at once."""
        for metric, value in values.items():
            self.push_metric(metric, value)

    def add_metric(self, metric: str) -> None:
        """Register a new metric to track."""
        if metric not in self._series:
            self._series[metric] = deque(maxlen=self._buffer_size)
            self._dim_history[metric] = []

    def series_length(self, metric: str) -> int:
        """Return current number of samples for a metric."""
        return len(self._series.get(metric, []))

    # ─── Reconstruction ───────────────────────────────────────────

    def reconstruct_all(self) -> PhaseSpaceReport:
        """
        Run full phase-space reconstruction for all registered metrics.
        Skips metrics with insufficient data.
        """
        now = time.time()
        diagnoses: dict[str, AttractorDiagnosis] = {}
        skipped: list[str] = []

        for metric in list(self._series.keys()):
            diag = self.reconstruct_metric(metric)
            if diag is None:
                skipped.append(metric)
            else:
                diagnoses[metric] = diag

        logger.debug(
            "phase_space_reconstruction_complete",
            n_diagnosed=len(diagnoses),
            n_skipped=len(skipped),
        )
        return PhaseSpaceReport(
            timestamp=now,
            diagnoses=diagnoses,
            skipped_metrics=skipped,
        )

    def reconstruct_metric(self, metric: str) -> AttractorDiagnosis | None:
        """
        Reconstruct attractor for a single metric.
        Returns None if insufficient data (<_MIN_SERIES_LENGTH samples).
        """
        buf = self._series.get(metric)
        if buf is None or len(buf) < _MIN_SERIES_LENGTH:
            return None

        series = np.array(buf, dtype=float)

        # Normalize to [0, 1] to make algorithms scale-invariant
        s_min, s_max = series.min(), series.max()
        if s_max - s_min < 1e-10:
            # Constant series — degenerate attractor
            return AttractorDiagnosis(
                metric=metric,
                timestamp=time.time(),
                embedding_delay=1,
                embedding_dimension=1,
                correlation_dimension=0.0,
                dimension_trend="stable",
                largest_lyapunov=0.0,
                lyapunov_interpretation="stable",
                predictability_horizon_cycles=_MAX_PREDICTABILITY_CYCLES,
            )

        series = (series - s_min) / (s_max - s_min)

        try:
            tau = estimate_delay(series)
            dim = estimate_embedding_dimension(series, tau)
            embedded = delay_embed(series, dim, tau)

            if len(embedded) < 20:
                return None

            d2 = estimate_correlation_dimension(embedded)
            lle = estimate_largest_lyapunov(embedded)

            # Update dimension history for trend detection
            hist = self._dim_history.setdefault(metric, [])
            hist.append(d2)
            if len(hist) > 20:
                hist.pop(0)

            trend = _classify_dimension_trend(d2, hist)
            interp = _interpret_lyapunov(lle)
            horizon = _predictability_horizon(lle)

            return AttractorDiagnosis(
                metric=metric,
                timestamp=time.time(),
                embedding_delay=tau,
                embedding_dimension=dim,
                correlation_dimension=d2,
                dimension_trend=trend,
                largest_lyapunov=lle,
                lyapunov_interpretation=interp,
                predictability_horizon_cycles=horizon,
            )

        except Exception:
            logger.warning("reconstruction_failed", metric=metric, exc_info=True)
            return None
