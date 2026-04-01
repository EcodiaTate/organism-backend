"""
Fovea - Economic Prediction Model

Tracks revenue and cost expectations using exponential moving averages,
computes prediction errors when actuals diverge from expectations, and
detects trend divergence (not just level divergence) before starvation fires.

This module belongs in Fovea because prediction error is Fovea's job.
Oikos tracks the actuals; Fovea predicts them and measures the delta.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger("systems.fovea.economic_model")

# Smoothing factor for exponential moving average.
# α=0.3 responds within ~3 observations while remaining stable.
_EMA_ALPHA = 0.3

# Minimum observations before error is meaningful (avoid false alarms on boot).
_MIN_OBSERVATIONS = 3

# Trend window: number of consecutive observations to compute velocity.
_TREND_WINDOW = 5

# Per-source decay: if a source hasn't reported in this many cycles, reduce
# its expected contribution rather than keeping a stale high prediction.
_SOURCE_STALE_CYCLES = 6


class _SourceTracker:
    """Per-revenue-source prediction tracker."""

    def __init__(
        self,
        source: str,
        *,
        ema_alpha: float = _EMA_ALPHA,
        stale_cycles: int = _SOURCE_STALE_CYCLES,
        min_observations: int = _MIN_OBSERVATIONS,
    ) -> None:
        self.source = source
        self.predicted: float = 0.0          # EMA prediction for this source
        self.last_actual: float = 0.0        # Most recent actual from Oikos
        self.observation_count: int = 0
        self.cycles_since_update: int = 0    # Staleness counter
        self._history: deque[float] = deque(maxlen=_TREND_WINDOW)
        self._alpha: float = ema_alpha
        self._stale_cycles: int = stale_cycles
        self._min_observations: int = min_observations

    def update(self, actual: float) -> float:
        """Update with a new actual value; return normalised prediction error [0, 1]."""
        self.cycles_since_update = 0
        self.last_actual = actual
        self.observation_count += 1
        self._history.append(actual)

        if self.observation_count <= 1:
            # Seed: no prior - accept actual as prediction, no error yet.
            self.predicted = actual
            return 0.0

        raw_error = abs(self.predicted - actual)
        # Normalise against the larger of predicted/actual to stay in [0, 1].
        denominator = max(abs(self.predicted), abs(actual), 1e-6)
        normalised = min(1.0, raw_error / denominator)

        # Update EMA prediction AFTER computing the error for this cycle.
        self.predicted = self._alpha * actual + (1.0 - self._alpha) * self.predicted
        return normalised

    def age(self) -> None:
        """Called each Oikos vitality cycle for sources that didn't report."""
        self.cycles_since_update += 1
        # Gradually decay our expectation toward zero so stale sources don't
        # produce persistent phantom errors.
        if self.cycles_since_update > self._stale_cycles:
            self.predicted *= 0.9

    @property
    def trend_velocity(self) -> float:
        """
        Revenue velocity from recent history: positive = growing, negative = shrinking.

        Returns the average delta between consecutive observations, normalised
        by the mean absolute value so it's scale-independent.
        """
        hist = list(self._history)
        if len(hist) < 2:
            return 0.0
        deltas = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
        mean_delta = sum(deltas) / len(deltas)
        mean_abs = sum(abs(h) for h in hist) / len(hist)
        if mean_abs < 1e-9:
            return 0.0
        return mean_delta / mean_abs  # dimensionless, positive = growth

    @property
    def is_sufficiently_observed(self) -> bool:
        return self.observation_count >= self._min_observations


class EconomicPredictionModel:
    """
    Tracks revenue and cost expectations; computes prediction errors.

    Design principles:
    - Uses EMA - simple, no ARIMA, no external deps.
    - Tracks per-source revenue so localised failures (e.g. only yield drops)
      are distinguishable from global revenue collapse.
    - Detects TREND divergence (declining revenue that is still above threshold)
      in addition to LEVEL divergence (revenue already below expected).
    - Stays entirely in-process; no I/O. Feed it from _on_economic_vitality().
    """

    def __init__(self) -> None:
        # Aggregate 24h rolling predictions (from Oikos ECONOMIC_VITALITY payload)
        self._predicted_revenue_24h: float = 0.0
        self._predicted_costs_24h: float = 0.0
        self._predicted_efficiency: float = 1.0

        # Per-source trackers keyed by RevenueStream value string
        self._source_trackers: dict[str, _SourceTracker] = {}

        # Aggregate observation history for trend computation
        self._revenue_history: deque[float] = deque(maxlen=_TREND_WINDOW)
        self._efficiency_history: deque[float] = deque(maxlen=_TREND_WINDOW)

        # Running error state (updated by update_from_economic_vitality)
        self._revenue_error: float = 0.0
        self._cost_error: float = 0.0
        self._efficiency_error: float = 0.0
        self._trend_error: float = 0.0

        # The source with the largest per-source error in the last update
        self.worst_source: str = ""
        self.worst_source_error: float = 0.0

        # Count of observations (gate for premature error signals)
        self._observation_count: int = 0

        # ── LEARNABLE economic model parameters (Evo/Simula can tune) ──
        self._ema_alpha: float = _EMA_ALPHA
        self._min_observations: int = _MIN_OBSERVATIONS
        self._trend_window: int = _TREND_WINDOW
        self._source_stale_cycles: int = _SOURCE_STALE_CYCLES
        self._composite_weights: dict[str, float] = {
            "revenue": 0.35,
            "efficiency": 0.25,
            "trend": 0.25,
            "worst_source": 0.15,
        }

        self._logger = logger.bind(component="economic_prediction_model")

    # ── LEARNABLE parameter API (AUTONOMY) ──

    def adjust_param(self, name: str, value: float) -> bool:
        """Adjust an economic model parameter. Called by Evo ADJUST_BUDGET."""
        if name == "ema_alpha":
            self._ema_alpha = max(0.01, min(0.99, value))
        elif name == "min_observations":
            self._min_observations = max(1, min(20, int(value)))
        elif name == "trend_window":
            self._trend_window = max(2, min(50, int(value)))
        elif name == "source_stale_cycles":
            self._source_stale_cycles = max(1, min(50, int(value)))
        elif name.startswith("composite_weight_"):
            weight_key = name[len("composite_weight_"):]
            if weight_key in self._composite_weights:
                self._composite_weights[weight_key] = max(0.0, min(1.0, value))
                # Re-normalise composite weights
                total = sum(self._composite_weights.values())
                if total > 0:
                    for k in self._composite_weights:
                        self._composite_weights[k] /= total
            else:
                return False
        else:
            return False
        logger.info("economic_param_adjusted", name=name, value=round(value, 4))
        return True

    def get_learnable_params(self) -> dict[str, float]:
        """Return all learnable economic model parameters for introspection."""
        params: dict[str, float] = {
            "ema_alpha": self._ema_alpha,
            "min_observations": float(self._min_observations),
            "trend_window": float(self._trend_window),
            "source_stale_cycles": float(self._source_stale_cycles),
        }
        for k, v in self._composite_weights.items():
            params[f"composite_weight_{k}"] = v
        return params

    def export_learnable_params(self) -> dict[str, float]:
        """Export for genome inheritance."""
        return self.get_learnable_params()

    def import_learnable_params(self, params: dict[str, float]) -> None:
        """Import from parent genome."""
        for name, value in params.items():
            self.adjust_param(name, value)

    # ------------------------------------------------------------------
    # Primary update entry point
    # ------------------------------------------------------------------

    def update_from_economic_vitality(self, payload: dict[str, Any]) -> float:
        """
        Called when ECONOMIC_VITALITY arrives from Oikos.

        Extracts actuals, compares to predictions, updates EMA predictions,
        and returns the composite normalised prediction error magnitude [0, 1].
        """
        self._observation_count += 1

        # Extract actuals - Oikos sends Decimal-serialised strings or floats
        try:
            revenue_24h = float(payload.get("revenue_24h", 0) or 0)
            costs_24h = float(payload.get("costs_24h", 0) or 0)
            efficiency = float(payload.get("metabolic_efficiency", 1) or 1)
            revenue_by_source: dict[str, Any] = payload.get("revenue_by_source") or {}
        except (TypeError, ValueError):
            self._logger.warning("economic_vitality_parse_error", payload_keys=list(payload.keys()))
            return 0.0

        # Age all per-source trackers (mark unseen sources as stale)
        for tracker in self._source_trackers.values():
            tracker.age()

        # Per-source errors
        worst_source = ""
        worst_error = 0.0
        for source_key, amount in revenue_by_source.items():
            try:
                amount_f = float(amount or 0)
            except (TypeError, ValueError):
                continue
            if source_key not in self._source_trackers:
                self._source_trackers[source_key] = _SourceTracker(
                    source_key,
                    ema_alpha=self._ema_alpha,
                    stale_cycles=self._source_stale_cycles,
                    min_observations=self._min_observations,
                )
            tracker = self._source_trackers[source_key]
            src_error = tracker.update(amount_f)
            if tracker.is_sufficiently_observed and src_error > worst_error:
                worst_error = src_error
                worst_source = source_key

        self.worst_source = worst_source
        self.worst_source_error = worst_error

        # Aggregate-level revenue error
        if self._observation_count <= 1:
            # Seed run - accept actuals as baseline, no error yet
            self._predicted_revenue_24h = revenue_24h
            self._predicted_costs_24h = costs_24h
            self._predicted_efficiency = efficiency
            self._revenue_history.append(revenue_24h)
            self._efficiency_history.append(efficiency)
            return 0.0

        self._revenue_error = self._normalised_error(
            self._predicted_revenue_24h, revenue_24h
        )
        self._cost_error = self._normalised_error(
            self._predicted_costs_24h, costs_24h
        )
        self._efficiency_error = self._normalised_error(
            self._predicted_efficiency, efficiency
        )

        # Trend error: is the trend WORSE than predicted?
        self._revenue_history.append(revenue_24h)
        self._efficiency_history.append(efficiency)
        self._trend_error = self._compute_trend_error()

        # Update EMA predictions
        alpha = self._ema_alpha
        self._predicted_revenue_24h = (
            alpha * revenue_24h + (1.0 - alpha) * self._predicted_revenue_24h
        )
        self._predicted_costs_24h = (
            alpha * costs_24h + (1.0 - alpha) * self._predicted_costs_24h
        )
        self._predicted_efficiency = (
            alpha * efficiency + (1.0 - alpha) * self._predicted_efficiency
        )

        # Composite error: weighted combination of revenue, efficiency, trend, worst-source
        cw = self._composite_weights
        composite = max(
            self._revenue_error * cw["revenue"]
            + self._efficiency_error * cw["efficiency"]
            + self._trend_error * cw["trend"]
            + worst_error * cw["worst_source"],
            0.0,
        )
        composite = min(1.0, composite)

        self._logger.debug(
            "economic_prediction_updated",
            revenue_error=round(self._revenue_error, 3),
            efficiency_error=round(self._efficiency_error, 3),
            trend_error=round(self._trend_error, 3),
            worst_source=worst_source,
            worst_source_error=round(worst_error, 3),
            composite=round(composite, 3),
        )
        return composite

    def update_from_revenue_event(self, source: str, amount: float) -> None:
        """
        Track an individual revenue event for per-source prediction.

        Called on fine-grained revenue events (e.g. individual bounty receipts)
        to update the per-source EMA between Oikos vitality snapshots.
        """
        if source not in self._source_trackers:
            self._source_trackers[source] = _SourceTracker(
                source,
                ema_alpha=self._ema_alpha,
                stale_cycles=self._source_stale_cycles,
                min_observations=self._min_observations,
            )
        self._source_trackers[source].update(amount)

    # ------------------------------------------------------------------
    # Properties - current error state
    # ------------------------------------------------------------------

    @property
    def revenue_prediction_error(self) -> float:
        """Current aggregate revenue divergence [0, 1]."""
        return self._revenue_error

    @property
    def cost_prediction_error(self) -> float:
        """Current aggregate cost divergence [0, 1]."""
        return self._cost_error

    @property
    def efficiency_trend_error(self) -> float:
        """Error in predicted metabolic efficiency trend vs actual [0, 1]."""
        return self._trend_error

    @property
    def is_warmed_up(self) -> bool:
        """True once enough observations exist for meaningful errors."""
        return self._observation_count >= self._min_observations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalised_error(predicted: float, actual: float) -> float:
        """Normalised absolute error relative to the larger value [0, 1]."""
        raw = abs(predicted - actual)
        denominator = max(abs(predicted), abs(actual), 1e-6)
        return min(1.0, raw / denominator)

    def _compute_trend_error(self) -> float:
        """
        Detect when the TREND is diverging from prediction even if level is ok.

        Computes the velocity of revenue change over _TREND_WINDOW observations.
        - Revenue declining consistently → positive trend error (bad)
        - Revenue stable or growing → near-zero trend error
        - Revenue volatile (high variance) → moderate trend error

        Returns [0, 1]; 0 = trend as expected, 1 = severe unexpected decline.
        """
        rev_hist = list(self._revenue_history)
        if len(rev_hist) < 2:
            return 0.0

        mean_rev = sum(rev_hist) / len(rev_hist)
        if mean_rev < 1e-6:
            return 0.0

        # Velocity: average delta between consecutive observations
        deltas = [rev_hist[i] - rev_hist[i - 1] for i in range(1, len(rev_hist))]
        velocity = sum(deltas) / len(deltas)

        # Normalise velocity by mean revenue so it's scale-independent
        normalised_velocity = velocity / mean_rev  # negative = declining

        # Variance: high variance = unpredictable = moderate error signal
        variance = sum((r - mean_rev) ** 2 for r in rev_hist) / len(rev_hist)
        std_dev = math.sqrt(variance)
        normalised_std = min(1.0, std_dev / (mean_rev + 1e-6))

        # Trend error is high when:
        # 1. Revenue is declining (negative velocity), OR
        # 2. Revenue is unusually volatile (high std)
        decline_signal = max(0.0, -normalised_velocity)  # only penalise decline
        volatility_signal = normalised_std * 0.5

        return min(1.0, decline_signal + volatility_signal)

    def build_re_context(self) -> dict[str, Any]:
        """
        Build reasoning context for RE training examples.

        Returns a dict describing the current prediction state - what was
        predicted, what happened, and where the worst divergence occurred.
        """
        source_summary = {
            src: {
                "predicted": round(t.predicted, 4),
                "actual": round(t.last_actual, 4),
                "trend_velocity": round(t.trend_velocity, 4),
                "observation_count": t.observation_count,
                "cycles_since_update": t.cycles_since_update,
            }
            for src, t in self._source_trackers.items()
            if t.observation_count > 0
        }
        return {
            "predicted_revenue_24h": round(self._predicted_revenue_24h, 4),
            "predicted_costs_24h": round(self._predicted_costs_24h, 4),
            "predicted_efficiency": round(self._predicted_efficiency, 4),
            "revenue_error": round(self._revenue_error, 4),
            "cost_error": round(self._cost_error, 4),
            "efficiency_trend_error": round(self._trend_error, 4),
            "worst_source": self.worst_source,
            "worst_source_error": round(self.worst_source_error, 4),
            "source_breakdown": source_summary,
            "observation_count": self._observation_count,
        }
