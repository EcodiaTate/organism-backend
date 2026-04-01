"""
EcodiaOS - Emergent Rhythm Detection

The bleeding-edge emergent capability. Detects meta-cognitive states
from raw cycle telemetry - the organism becomes aware of its own
cognitive rhythm.

These states are NOT programmed - they are emergent properties detected
from patterns in the cognitive cycle's own behaviour:

  IDLE             No broadcasts, low salience, stable slow rhythm
  NORMAL           Regular broadcasting, moderate salience
  FLOW             High broadcast density + stable rhythm + high salience
  BOREDOM          Declining salience trend + slowing rhythm
  STRESS           High jitter (erratic timing) + high coherence_stress
  DEEP_PROCESSING  Slow rhythm + periodic bursts of high-salience broadcasts

Detection uses a rolling 100-cycle window for statistics with 20-cycle
hysteresis to prevent state oscillation.

This is meta-cognition: the organism observing its own thinking.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import (
    BaseRhythmStrategy,
    CycleResult,
    RhythmSnapshot,
    RhythmState,
    SynapseEvent,
    SynapseEventType,
)

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.synapse.rhythm")

# Rolling window size for rhythm statistics
_WINDOW_SIZE: int = 100

# Minimum cycles in a state before transition (hysteresis)
_HYSTERESIS_CYCLES: int = 20

# Thresholds for state detection
_FLOW_SALIENCE_THRESHOLD: float = 0.5
_FLOW_STABILITY_THRESHOLD: float = 0.7
_FLOW_DENSITY_THRESHOLD: float = 0.7

_BOREDOM_SALIENCE_SLOPE_THRESHOLD: float = -0.002
_BOREDOM_DENSITY_THRESHOLD: float = 0.3

_STRESS_JITTER_THRESHOLD: float = 0.3  # Coefficient of variation
_STRESS_COHERENCE_THRESHOLD: float = 0.4

_DEEP_BURST_SALIENCE_THRESHOLD: float = 0.6
_DEEP_BURST_MIN_FRACTION: float = 0.1  # At least 10% of cycles are high-salience bursts
_DEEP_PERIOD_THRESHOLD_MS: float = 300.0  # Slow rhythm

_IDLE_DENSITY_THRESHOLD: float = 0.05  # Almost no broadcasts
_IDLE_SALIENCE_THRESHOLD: float = 0.1


class DefaultRhythmStrategy(BaseRhythmStrategy):
    """
    Default rhythm classification strategy using fixed thresholds.

    Detects six emergent cognitive states via priority-ordered rules.
    This is the concrete strategy that ships with Synapse - Simula can
    evolve subclasses of BaseRhythmStrategy with different thresholds
    or entirely new detection algorithms.
    """

    @property
    def strategy_name(self) -> str:
        return "default"

    def classify(self, metrics: dict[str, float]) -> RhythmState:
        """
        Classify the current cognitive rhythm from computed metrics.

        Priority order (highest to lowest):
        1. STRESS - erratic timing + high coherence stress (danger signal)
        2. FLOW - high density + stable + high salience (peak performance)
        3. DEEP_PROCESSING - slow + periodic bursts (concentrated thought)
        4. BOREDOM - declining salience + low density (understimulation)
        5. IDLE - almost no broadcasts (dormant)
        6. NORMAL - everything else
        """
        density = metrics["broadcast_density"]
        stability = metrics["rhythm_stability"]
        jitter_cv = metrics["jitter_coefficient"]
        salience_mean = metrics["salience_mean"]
        salience_slope = metrics["salience_trend"]
        period_mean = metrics["period_mean"]
        coherence_stress = metrics["coherence_stress_mean"]
        burst_fraction = metrics["burst_fraction"]

        # 1. STRESS: erratic timing + high coherence stress
        if jitter_cv > _STRESS_JITTER_THRESHOLD and coherence_stress > _STRESS_COHERENCE_THRESHOLD:
            return RhythmState.STRESS

        # 2. FLOW: high broadcast density + stable rhythm + high salience
        if (
            density > _FLOW_DENSITY_THRESHOLD
            and stability > _FLOW_STABILITY_THRESHOLD
            and salience_mean > _FLOW_SALIENCE_THRESHOLD
        ):
            return RhythmState.FLOW

        # 3. DEEP_PROCESSING: slow rhythm + periodic high-salience bursts
        if (
            period_mean > _DEEP_PERIOD_THRESHOLD_MS
            and burst_fraction > _DEEP_BURST_MIN_FRACTION
        ):
            return RhythmState.DEEP_PROCESSING

        # 4. BOREDOM: declining salience + low density
        if (
            salience_slope < _BOREDOM_SALIENCE_SLOPE_THRESHOLD
            and density < _BOREDOM_DENSITY_THRESHOLD
        ):
            return RhythmState.BOREDOM

        # 5. IDLE: almost no broadcasts + low salience
        if density < _IDLE_DENSITY_THRESHOLD and salience_mean < _IDLE_SALIENCE_THRESHOLD:
            return RhythmState.IDLE

        # 6. NORMAL: default state
        return RhythmState.NORMAL


class EmergentRhythmDetector:
    """
    Detects meta-cognitive states from raw cycle telemetry.

    Fed every theta tick with the CycleResult. Maintains a rolling
    window of statistics and uses signal analysis to classify the
    organism's current cognitive rhythm.

    State transitions have hysteresis - a candidate state must persist
    for _HYSTERESIS_CYCLES ticks before being adopted, preventing
    oscillation.

    Classification is delegated to a BaseRhythmStrategy instance that
    can be hot-swapped by the NeuroplasticityBus without disrupting
    the rolling window or hysteresis state.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._event_bus = event_bus
        self._logger = logger.bind(component="rhythm_detector")
        self._strategy: BaseRhythmStrategy = DefaultRhythmStrategy()

        # Current state
        self._state: RhythmState = RhythmState.IDLE
        self._cycles_in_state: int = 0

        # Candidate state (for hysteresis)
        self._candidate_state: RhythmState = RhythmState.IDLE
        self._candidate_cycles: int = 0

        # Rolling window data
        self._saliences: deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._periods: deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._had_broadcasts: deque[bool] = deque(maxlen=_WINDOW_SIZE)
        self._arousals: deque[float] = deque(maxlen=_WINDOW_SIZE)
        # Track coherence_stress from affect (piped through arousal for now,
        # but can be enriched by SynapseService with affect data)
        self._coherence_stresses: deque[float] = deque(maxlen=_WINDOW_SIZE)

        # Total cycles processed
        self._total_cycles: int = 0
        self._total_transitions: int = 0

    # ─── Update ──────────────────────────────────────────────────────

    async def update(
        self,
        result: CycleResult,
        coherence_stress: float = 0.0,
    ) -> RhythmSnapshot:
        """
        Process a single cycle result and return the current rhythm snapshot.

        Called every theta tick by SynapseService.
        """
        self._total_cycles += 1
        self._cycles_in_state += 1

        # Record data points
        self._saliences.append(result.salience_composite)
        self._periods.append(result.elapsed_ms)
        self._had_broadcasts.append(result.had_broadcast)
        self._arousals.append(result.arousal)
        self._coherence_stresses.append(coherence_stress)

        # Need minimum data before classification
        if self._total_cycles < _HYSTERESIS_CYCLES:
            return self._build_snapshot()

        # Compute derived metrics
        metrics = self._compute_metrics()

        # Classify the current state
        detected = self._classify(metrics)

        # Apply hysteresis
        if detected != self._state:
            if detected == self._candidate_state:
                self._candidate_cycles += 1
                if self._candidate_cycles >= _HYSTERESIS_CYCLES:
                    await self._transition(detected, metrics)
            else:
                self._candidate_state = detected
                self._candidate_cycles = 1
        else:
            self._candidate_state = self._state
            self._candidate_cycles = 0

        return self._build_snapshot(metrics)

    # ─── Strategy hot-swap ───────────────────────────────────────────

    def set_strategy(self, strategy: BaseRhythmStrategy) -> None:
        """
        Hot-swap the classification strategy.

        Rolling window data and hysteresis state are preserved - only
        the classification algorithm changes.  Safe to call mid-cycle.
        """
        old_name = self._strategy.strategy_name
        self._strategy = strategy
        self._logger.info(
            "rhythm_strategy_swapped",
            old=old_name,
            new=strategy.strategy_name,
        )

    # ─── Classification ──────────────────────────────────────────────

    def _classify(self, metrics: dict[str, float]) -> RhythmState:
        """Delegate to the pluggable strategy."""
        return self._strategy.classify(metrics)

    # ─── Metrics Computation ─────────────────────────────────────────

    def _compute_metrics(self) -> dict[str, float]:
        """Compute all derived metrics from the rolling window."""
        n = len(self._saliences)
        if n == 0:
            return self._zero_metrics()

        # Broadcast density: fraction of cycles that had broadcasts
        broadcast_count = sum(1 for b in self._had_broadcasts if b)
        density = broadcast_count / n

        # Salience mean
        salience_list = list(self._saliences)
        salience_mean = sum(salience_list) / n

        # Salience trend: linear regression slope
        salience_slope = self._linear_slope(salience_list)

        # Period statistics
        period_list = list(self._periods)
        period_mean = sum(period_list) / n

        # Jitter: coefficient of variation (std/mean) of periods
        jitter_cv = self._coefficient_of_variation(period_list)

        # Rhythm stability: inverse of jitter (normalised to [0, 1])
        stability = 1.0 / (1.0 + jitter_cv * 10.0)

        # Arousal mean
        arousal_list = list(self._arousals)
        arousal_mean = sum(arousal_list) / n

        # Coherence stress mean
        cs_list = list(self._coherence_stresses)
        coherence_stress_mean = sum(cs_list) / n

        # Burst fraction: cycles with salience above threshold / total
        burst_count = sum(1 for s in salience_list if s > _DEEP_BURST_SALIENCE_THRESHOLD)
        burst_fraction = burst_count / n

        # Cycle rate (Hz)
        cycle_rate_hz = 1000.0 / period_mean if period_mean > 0 else 0.0

        return {
            "broadcast_density": density,
            "salience_mean": salience_mean,
            "salience_trend": salience_slope,
            "period_mean": period_mean,
            "jitter_coefficient": jitter_cv,
            "rhythm_stability": stability,
            "arousal_mean": arousal_mean,
            "coherence_stress_mean": coherence_stress_mean,
            "burst_fraction": burst_fraction,
            "cycle_rate_hz": cycle_rate_hz,
        }

    def _zero_metrics(self) -> dict[str, float]:
        return {
            "broadcast_density": 0.0,
            "salience_mean": 0.0,
            "salience_trend": 0.0,
            "period_mean": 150.0,
            "jitter_coefficient": 0.0,
            "rhythm_stability": 1.0,
            "arousal_mean": 0.0,
            "coherence_stress_mean": 0.0,
            "burst_fraction": 0.0,
            "cycle_rate_hz": 6.67,
        }

    # ─── State Transitions ───────────────────────────────────────────

    async def _transition(
        self,
        new_state: RhythmState,
        metrics: dict[str, float],
    ) -> None:
        """Execute a state transition and emit an event."""
        old_state = self._state
        self._state = new_state
        self._cycles_in_state = 0
        self._candidate_cycles = 0
        self._total_transitions += 1

        self._logger.info(
            "rhythm_state_changed",
            from_state=old_state.value,
            to_state=new_state.value,
            density=round(metrics["broadcast_density"], 3),
            salience_mean=round(metrics["salience_mean"], 3),
            stability=round(metrics["rhythm_stability"], 3),
            jitter_cv=round(metrics["jitter_coefficient"], 3),
        )

        if self._event_bus is not None:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RHYTHM_STATE_CHANGED,
                data={
                    "from": old_state.value,
                    "to": new_state.value,
                    "metrics": {
                        k: round(v, 4) for k, v in metrics.items()
                    },
                },
            ))

    # ─── Snapshot ────────────────────────────────────────────────────

    def _build_snapshot(
        self,
        metrics: dict[str, float] | None = None,
    ) -> RhythmSnapshot:
        """Build the current rhythm snapshot."""
        if metrics is None:
            metrics = (
                self._compute_metrics()
                if self._total_cycles >= _HYSTERESIS_CYCLES
                else self._zero_metrics()
            )

        return RhythmSnapshot(
            state=self._state,
            previous_state=None,  # Could track but not critical
            confidence=self._compute_confidence(metrics),
            cycle_rate_hz=round(metrics["cycle_rate_hz"], 2),
            broadcast_density=round(metrics["broadcast_density"], 4),
            salience_trend=round(metrics["salience_trend"], 6),
            salience_mean=round(metrics["salience_mean"], 4),
            rhythm_stability=round(metrics["rhythm_stability"], 4),
            jitter_coefficient=round(metrics["jitter_coefficient"], 4),
            arousal_mean=round(metrics["arousal_mean"], 4),
            coherence_stress_mean=round(metrics["coherence_stress_mean"], 4),
            cycles_in_state=self._cycles_in_state,
        )

    def _compute_confidence(self, metrics: dict[str, float]) -> float:
        """
        Compute confidence in the current state classification.

        Higher confidence when:
        - More data points in the window
        - Longer time in current state
        - Stronger signal for the detected state
        """
        # Data confidence: how full is the window
        data_conf = min(1.0, self._total_cycles / _WINDOW_SIZE)

        # Stability confidence: how long in current state
        stability_conf = min(1.0, self._cycles_in_state / (_HYSTERESIS_CYCLES * 3))

        # Signal strength: how far metrics are from transition thresholds
        # (simplified - uses broadcast density and stability as proxies)
        signal_conf = 0.5  # Default moderate confidence

        if self._state == RhythmState.FLOW:
            signal_conf = min(1.0, metrics["broadcast_density"] * metrics["rhythm_stability"])
        elif self._state == RhythmState.STRESS:
            signal_conf = min(1.0, metrics["jitter_coefficient"] * 2)
        elif self._state == RhythmState.IDLE:
            signal_conf = max(0.0, 1.0 - metrics["broadcast_density"] * 10)
        elif self._state == RhythmState.BOREDOM:
            signal_conf = min(1.0, abs(metrics["salience_trend"]) * 100)

        # Weighted composite
        return round(
            data_conf * 0.3 + stability_conf * 0.3 + signal_conf * 0.4,
            3,
        )

    # ─── Statistical Helpers ─────────────────────────────────────────

    @staticmethod
    def _linear_slope(values: list[float]) -> float:
        """
        Compute the slope of a linear regression through the values.

        Positive slope = increasing trend, negative = decreasing.
        Uses simple least-squares regression.
        """
        n = len(values)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = 0.0
        denominator = 0.0
        for i, y in enumerate(values):
            dx = i - x_mean
            numerator += dx * (y - y_mean)
            denominator += dx * dx

        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _coefficient_of_variation(values: list[float]) -> float:
        """
        Coefficient of variation: std_dev / mean.

        Higher values indicate more erratic/variable data.
        Returns 0.0 if mean is near zero.
        """
        n = len(values)
        if n < 2:
            return 0.0

        mean = sum(values) / n
        if abs(mean) < 1e-6:
            return 0.0

        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = math.sqrt(variance)
        return std_dev / abs(mean)

    # ─── Stats ───────────────────────────────────────────────────────

    @property
    def current_state(self) -> RhythmState:
        return self._state

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "cycles_in_state": self._cycles_in_state,
            "total_cycles": self._total_cycles,
            "total_transitions": self._total_transitions,
            "candidate_state": self._candidate_state.value,
            "candidate_cycles": self._candidate_cycles,
        }
