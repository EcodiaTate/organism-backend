"""
EcodiaOS — Oneiros: Circadian Clock & Sleep Stage Controller

The CircadianClock tracks sleep pressure — the organism's accumulating
need for rest. Like adenosine in biological brains, pressure builds from
four independent sources during wakefulness.

The SleepStageController is the state machine governing transitions
between consciousness states: wake → hypnagogia → NREM → REM →
lucid → hypnopompia → wake.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from primitives.common import utc_now
from systems.oneiros.types import (
    CircadianPhase,
    SleepPressure,
    SleepQuality,
    SleepStage,
    WakeDegradation,
)

logger = structlog.get_logger().bind(system="oneiros", component="circadian")


# ─── Pressure Weights ─────────────────────────────────────────────

_DEFAULT_WEIGHT_CYCLES = 0.40
_DEFAULT_WEIGHT_AFFECT = 0.25
_DEFAULT_WEIGHT_EPISODES = 0.20
_DEFAULT_WEIGHT_HYPOTHESES = 0.15

# Capacity denominators (normalise raw counts to 0-1 range)
_DEFAULT_AFFECT_CAPACITY = 50.0
_DEFAULT_EPISODE_CAPACITY = 500
_DEFAULT_HYPOTHESIS_CAPACITY = 50

# Quality reset multipliers
_QUALITY_RESET: dict[SleepQuality, float] = {
    SleepQuality.DEEP: 1.00,
    SleepQuality.NORMAL: 0.90,
    SleepQuality.FRAGMENTED: 0.60,
    SleepQuality.DEPRIVED: 0.30,
}


class CircadianClock:
    """
    Tracks sleep pressure and manages the circadian rhythm.

    Updated every cognitive cycle via ``tick()``. Pressure rises from
    four independent sources:

    1. Raw time awake (cycles since last sleep)
    2. Unprocessed affect residue (high-emotion experiences)
    3. Unconsolidated episode count (raw memories)
    4. Hypothesis backlog (Evo's unresolved questions)

    When pressure crosses the threshold the organism should sleep.
    When it crosses the critical threshold it *must* sleep.
    """

    def __init__(self, config: Any = None) -> None:
        cfg = config or {}
        self._pressure = SleepPressure(
            threshold=_get(cfg, "pressure_threshold", 0.70),
            critical_threshold=_get(cfg, "pressure_critical", 0.95),
        )
        self._phase = CircadianPhase(
            wake_duration_target_s=_get(cfg, "wake_duration_target_s", 79200.0),
            sleep_duration_target_s=_get(cfg, "sleep_duration_target_s", 7200.0),
        )
        self._max_wake_cycles: int = _get(cfg, "max_wake_cycles", 528000)
        self._w_cycles: float = _get(cfg, "pressure_weight_cycles", _DEFAULT_WEIGHT_CYCLES)
        self._w_affect: float = _get(cfg, "pressure_weight_affect", _DEFAULT_WEIGHT_AFFECT)
        self._w_episodes: float = _get(cfg, "pressure_weight_episodes", _DEFAULT_WEIGHT_EPISODES)
        self._w_hypotheses: float = _get(
            cfg, "pressure_weight_hypotheses", _DEFAULT_WEIGHT_HYPOTHESES,
        )
        self._affect_capacity: float = _get(cfg, "affect_capacity", _DEFAULT_AFFECT_CAPACITY)
        self._episode_capacity: int = _get(cfg, "episode_capacity", _DEFAULT_EPISODE_CAPACITY)
        self._hypothesis_capacity: int = _get(
            cfg, "hypothesis_capacity", _DEFAULT_HYPOTHESIS_CAPACITY,
        )
        self._debt_noise_max: float = _get(cfg, "debt_salience_noise_max", 0.15)
        self._debt_efe_max: float = _get(cfg, "debt_efe_precision_loss_max", 0.20)
        self._debt_flat_max: float = _get(cfg, "debt_expression_flatness_max", 0.25)
        self._debt_learn_max: float = _get(cfg, "debt_learning_rate_reduction_max", 0.30)

        self._logger = logger

    # ── Public API ────────────────────────────────────────────────

    def tick(self) -> None:
        """Called every cognitive cycle during WAKE. Increments counters and recomputes pressure."""
        self._pressure.cycles_since_sleep += 1
        self._pressure.composite_pressure = self.compute_pressure()
        self._pressure.last_computation = utc_now()

    def record_affect_trace(self, valence: float, arousal: float) -> None:
        """Record a high-affect episode for pressure tracking."""
        contribution = 0.0
        if abs(valence) > 0.7:
            contribution += abs(valence)
        if arousal > 0.8:
            contribution += arousal
        if contribution > 0.0:
            self._pressure.unprocessed_affect_residue += contribution

    def record_episode(self) -> None:
        """Increment unconsolidated episode count."""
        self._pressure.unconsolidated_episode_count += 1

    def record_hypothesis_count(self, count: int) -> None:
        """Update hypothesis backlog from Evo."""
        self._pressure.hypothesis_backlog = count

    def compute_pressure(self) -> float:
        """
        Compute composite sleep pressure from four sources.

        Returns value clamped to [0.0, 1.5]. Values above 1.0
        indicate severe sleep debt.
        """
        p = self._pressure
        cycle_ratio = min(
            1.0, p.cycles_since_sleep / max(self._max_wake_cycles, 1),
        )
        affect_ratio = min(
            1.0, p.unprocessed_affect_residue / max(self._affect_capacity, 0.01),
        )
        episode_ratio = min(
            1.0, p.unconsolidated_episode_count / max(self._episode_capacity, 1),
        )
        hyp_ratio = min(
            1.0, p.hypothesis_backlog / max(self._hypothesis_capacity, 1),
        )

        raw = (
            self._w_cycles * cycle_ratio
            + self._w_affect * affect_ratio
            + self._w_episodes * episode_ratio
            + self._w_hypotheses * hyp_ratio
        )
        return min(1.5, max(0.0, raw))

    def should_sleep(self) -> bool:
        """Pressure has crossed the DROWSY threshold."""
        return self._pressure.composite_pressure >= self._pressure.threshold

    def must_sleep(self) -> bool:
        """Pressure has crossed the CRITICAL threshold — forced sleep."""
        return self._pressure.composite_pressure >= self._pressure.critical_threshold

    def reset_after_sleep(self, quality: SleepQuality) -> None:
        """
        Reset counters after a sleep cycle completes.

        Reset amount depends on sleep quality:
        - DEEP: 100% reset
        - NORMAL: 90% reset
        - FRAGMENTED: 60% reset
        - DEPRIVED: 30% reset
        """
        multiplier = _QUALITY_RESET.get(quality, 0.90)
        self._pressure.cycles_since_sleep = int(
            self._pressure.cycles_since_sleep * (1.0 - multiplier)
        )
        self._pressure.unprocessed_affect_residue *= 1.0 - multiplier
        self._pressure.unconsolidated_episode_count = int(
            self._pressure.unconsolidated_episode_count * (1.0 - multiplier)
        )
        # hypothesis_backlog is externally driven, but reduce residual
        self._pressure.hypothesis_backlog = int(
            self._pressure.hypothesis_backlog * (1.0 - multiplier)
        )
        self._pressure.composite_pressure = self.compute_pressure()
        self._pressure.last_sleep_completed = utc_now()
        self._phase.total_cycles_completed += 1

        self._logger.info(
            "sleep_pressure_reset",
            quality=quality.value,
            multiplier=multiplier,
            new_pressure=self._pressure.composite_pressure,
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def pressure(self) -> SleepPressure:
        return self._pressure

    @property
    def phase(self) -> CircadianPhase:
        return self._phase

    @property
    def degradation(self) -> WakeDegradation:
        """Current wake degradation from sleep debt."""
        return WakeDegradation.from_pressure(
            pressure=self._pressure.composite_pressure,
            threshold=self._pressure.threshold,
            critical=self._pressure.critical_threshold,
            noise_max=self._debt_noise_max,
            efe_max=self._debt_efe_max,
            flatness_max=self._debt_flat_max,
            learning_max=self._debt_learn_max,
        )


# ─── Sleep Stage Controller ───────────────────────────────────────


class SleepStageController:
    """
    State machine governing sleep stage transitions.

    ::

        WAKE → HYPNAGOGIA → NREM → REM → LUCID → HYPNOPOMPIA → WAKE
                                    ↓ (no creative goal)
                                HYPNOPOMPIA → WAKE

        Any sleep stage → HYPNOPOMPIA (emergency wake)
    """

    def __init__(self, config: Any = None) -> None:
        cfg = config or {}
        self._stage: SleepStage = SleepStage.WAKE
        self._stage_start: float = time.monotonic()
        self._stage_elapsed_s: float = 0.0
        self._sleep_budget_s: float = _get(cfg, "sleep_duration_target_s", 7200.0)
        self._sleep_elapsed_s: float = 0.0
        self._has_creative_goal: bool = False
        self._current_cycle_id: str | None = None

        # Stage durations
        self._hypnagogia_s: float = _get(cfg, "hypnagogia_duration_s", 30.0)
        self._hypnopompia_s: float = _get(cfg, "hypnopompia_duration_s", 30.0)
        self._nrem_fraction: float = _get(cfg, "nrem_fraction", 0.40)
        self._rem_fraction: float = _get(cfg, "rem_fraction", 0.40)
        self._lucid_fraction: float = _get(cfg, "lucid_fraction", 0.10)

        # Computed boundaries (seconds into sleep when transitions occur)
        effective_sleep = self._sleep_budget_s - self._hypnagogia_s - self._hypnopompia_s
        self._nrem_end_s = self._hypnagogia_s + effective_sleep * self._nrem_fraction
        self._rem_end_s = self._nrem_end_s + effective_sleep * self._rem_fraction
        self._lucid_end_s = self._rem_end_s + effective_sleep * self._lucid_fraction

        self._emergency_reason: str | None = None
        self._logger = logger

    # ── Public API ────────────────────────────────────────────────

    @property
    def current_stage(self) -> SleepStage:
        return self._stage

    @property
    def stage_elapsed_s(self) -> float:
        return self._stage_elapsed_s

    @property
    def is_sleeping(self) -> bool:
        return self._stage != SleepStage.WAKE

    @property
    def is_in_nrem(self) -> bool:
        return self._stage == SleepStage.NREM

    @property
    def is_in_rem(self) -> bool:
        return self._stage == SleepStage.REM

    @property
    def is_in_lucid(self) -> bool:
        return self._stage == SleepStage.LUCID

    def set_has_creative_goal(self, has_goal: bool) -> None:
        self._has_creative_goal = has_goal

    def begin_sleep(self, cycle_id: str) -> None:
        """Transition from WAKE → HYPNAGOGIA. Start of a sleep cycle."""
        if self._stage != SleepStage.WAKE:
            self._logger.warning("begin_sleep_not_wake", current_stage=self._stage.value)
            return

        self._current_cycle_id = cycle_id
        self._sleep_elapsed_s = 0.0
        self._transition_to(SleepStage.HYPNAGOGIA)
        self._logger.info("sleep_begun", cycle_id=cycle_id)

    def advance(self, elapsed_s: float) -> SleepStage | None:
        """
        Advance the sleep clock by *elapsed_s* seconds.

        Returns the new stage if a transition occurred, ``None`` if staying.
        """
        if self._stage == SleepStage.WAKE:
            return None

        self._stage_elapsed_s += elapsed_s
        self._sleep_elapsed_s += elapsed_s

        new_stage = self._check_transition()
        if new_stage is not None:
            self._transition_to(new_stage)
            return new_stage
        return None

    def emergency_wake(self, reason: str) -> None:
        """Immediately jump to HYPNOPOMPIA (then WAKE on next advance)."""
        if self._stage == SleepStage.WAKE:
            return
        from_stage = self._stage.value  # capture before _transition_to overwrites it
        self._emergency_reason = reason
        self._transition_to(SleepStage.HYPNOPOMPIA)
        self._stage_elapsed_s = self._hypnopompia_s * 0.5  # half-transition
        self._logger.warning("emergency_wake", reason=reason, from_stage=from_stage)

    def wake(self) -> None:
        """Transition to WAKE. Called after HYPNOPOMPIA completes."""
        self._transition_to(SleepStage.WAKE)
        self._current_cycle_id = None

    # ── Internals ─────────────────────────────────────────────────

    def _check_transition(self) -> SleepStage | None:
        """Determine if the current stage should transition."""
        if self._stage == SleepStage.HYPNAGOGIA:
            if self._stage_elapsed_s >= self._hypnagogia_s:
                return SleepStage.NREM

        elif self._stage == SleepStage.NREM:
            if self._sleep_elapsed_s >= self._nrem_end_s:
                return SleepStage.REM

        elif self._stage == SleepStage.REM:
            if self._sleep_elapsed_s >= self._rem_end_s:
                if self._has_creative_goal:
                    return SleepStage.LUCID
                return SleepStage.HYPNOPOMPIA

        elif self._stage == SleepStage.LUCID:
            if self._sleep_elapsed_s >= self._lucid_end_s:
                return SleepStage.HYPNOPOMPIA

        elif (
            self._stage == SleepStage.HYPNOPOMPIA
            and self._stage_elapsed_s >= self._hypnopompia_s
        ):
            return SleepStage.WAKE

        return None

    def _transition_to(self, new_stage: SleepStage) -> None:
        old = self._stage
        self._stage = new_stage
        self._stage_elapsed_s = 0.0
        self._stage_start = time.monotonic()
        self._logger.info(
            "stage_transition",
            from_stage=old.value,
            to_stage=new_stage.value,
            sleep_elapsed_s=round(self._sleep_elapsed_s, 1),
            cycle_id=self._current_cycle_id,
        )


# ─── Config Accessor ──────────────────────────────────────────────


def _get(cfg: Any, key: str, default: Any) -> Any:
    """Extract a config value from an object or dict, with default."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
