"""
EcodiaOS - Degradation Engine (Speciation Bible §8.2)

Genuine precariousness requires ACTIVE entropy resistance.
This engine emits three degradation signals every hour. If subscriber systems
(Memory, Evo, Simula) do not counteract them, the organism's vitality
measurably degrades over time - eventually crossing a fatal threshold.

Degradation rates (configurable via env vars):
  MEMORY_DECAY_RATE       = 0.02  (2%/day fidelity loss on unconsolidated episodes)
  CONFIG_DRIFT_RATE       = 0.01  (1%/day config drift from optimal)
  HYPOTHESIS_STALENESS_RATE = 0.05 (5%/day confidence decay on unvalidated hypotheses)

Soma homeostatic loops and Oneiros consolidation counteract memory decay.
Evo parameter optimisation counteracts config drift.
Evo hypothesis re-validation counteracts hypothesis staleness.

If nothing counteracts: vitality degrades → BRAIN_DEATH threshold approached.

Architecture: DegradationEngine is started by VitalityCoordinator alongside
its 30s check loop. It runs an independent hourly timer so death-proximity does
NOT slow down entropy accumulation - the organism fights entropy or succumbs.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.skia.degradation")

# ── Tick interval ──────────────────────────────────────────────────
_TICK_INTERVAL_S: float = 3600.0  # 1 hour - independent of 30s vitality loop

# ── Default degradation rates (per-day; converted to per-hour in tick) ────────
_DEFAULT_MEMORY_DECAY_RATE: float = 0.02       # 2% / day
_DEFAULT_CONFIG_DRIFT_RATE: float = 0.01       # 1% / day
_DEFAULT_HYPOTHESIS_STALENESS_RATE: float = 0.05  # 5% / day

# ── Episode age threshold for memory degradation signal ────────────
_DEGRADED_EPISODE_AGE_HOURS: float = 48.0  # episodes older than 2 days are at risk


@dataclass
class DegradationConfig:
    """
    Degradation rates from Speciation Bible §8.2.

    All rates are per-day fractions. DegradationEngine converts to per-tick
    values internally (dividing by 24 for hourly ticks).

    Configurable via environment variables so the organism itself (via Evo)
    can eventually evolve these thresholds.
    """

    memory_decay_rate: float = field(
        default_factory=lambda: float(
            os.environ.get("DEGRADATION_MEMORY_DECAY_RATE", _DEFAULT_MEMORY_DECAY_RATE)
        )
    )
    """Fraction of unconsolidated episode fidelity lost per day (default 0.02)."""

    config_drift_rate: float = field(
        default_factory=lambda: float(
            os.environ.get("DEGRADATION_CONFIG_DRIFT_RATE", _DEFAULT_CONFIG_DRIFT_RATE)
        )
    )
    """Fraction of learnable config params drifting from optimal per day (default 0.01)."""

    hypothesis_staleness_rate: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "DEGRADATION_HYPOTHESIS_STALENESS_RATE",
                _DEFAULT_HYPOTHESIS_STALENESS_RATE,
            )
        )
    )
    """Fraction of unvalidated hypothesis confidence lost per day (default 0.05)."""

    tick_interval_s: float = field(
        default_factory=lambda: float(
            os.environ.get("DEGRADATION_TICK_INTERVAL_S", _TICK_INTERVAL_S)
        )
    )
    """How often the degradation tick fires, in seconds (default 3600 = 1 hour)."""


@dataclass
class DegradationSnapshot:
    """
    Cumulative degradation pressure since last counteraction.

    VitalityCoordinator reads this to compute degradation_pressure (0.0–1.0)
    in the vitality report.
    """

    tick_count: int = 0
    """Number of hourly ticks since organism start."""

    cumulative_memory_fidelity_lost: float = 0.0
    """Accumulated memory fidelity loss (fraction, not counteracted yet)."""

    cumulative_config_drift: float = 0.0
    """Accumulated config drift (fraction, not counteracted yet)."""

    cumulative_hypothesis_staleness: float = 0.0
    """Accumulated hypothesis confidence loss (fraction, not counteracted yet)."""

    last_tick_at: float = 0.0
    """Monotonic timestamp of the last tick."""

    @property
    def degradation_pressure(self) -> float:
        """
        Composite entropy pressure score (0.0 = no pressure, 1.0 = maximum).

        Weighted average: memory 40%, config 20%, hypothesis 40%.
        Capped at 1.0 - beyond 1.0 is still 1.0 (already fatal territory).
        """
        raw = (
            self.cumulative_memory_fidelity_lost * 0.40
            + self.cumulative_config_drift * 0.20
            + self.cumulative_hypothesis_staleness * 0.40
        )
        return min(1.0, raw)

    def counteract_memory(self, fraction: float) -> None:
        """Call when Oneiros/Soma consolidation runs - reduces memory pressure."""
        self.cumulative_memory_fidelity_lost = max(
            0.0, self.cumulative_memory_fidelity_lost - fraction
        )

    def counteract_config(self, fraction: float) -> None:
        """Call when Evo parameter optimisation runs - reduces config drift."""
        self.cumulative_config_drift = max(
            0.0, self.cumulative_config_drift - fraction
        )

    def counteract_hypotheses(self, fraction: float) -> None:
        """Call when Evo re-validates hypotheses - reduces staleness."""
        self.cumulative_hypothesis_staleness = max(
            0.0, self.cumulative_hypothesis_staleness - fraction
        )


class DegradationEngine:
    """
    Hourly entropy engine - the organism must ACTIVELY fight this or die.

    Emits three Synapse events each tick:
      MEMORY_DEGRADATION      → Memory reduces fidelity on old episodes
      CONFIG_DRIFT            → Simula pertubs learnable config params
      HYPOTHESIS_STALENESS    → Evo decays confidence on unvalidated hypotheses
      DEGRADATION_TICK        → Aggregated tick summary for VitalityCoordinator

    If subscriber systems respond and counteract (Soma runs consolidation,
    Evo revalidates hypotheses, Evo/Simula re-optimise config), the
    cumulative pressure stays low. If not, pressure rises and vitality
    degrades toward fatal thresholds.

    VitalityCoordinator reads snapshot.degradation_pressure to include
    entropy burden in vitality reports.
    """

    def __init__(
        self,
        config: DegradationConfig | None = None,
        instance_id: str = "eos-default",
    ) -> None:
        self._config = config or DegradationConfig()
        self._instance_id = instance_id
        self._log = logger.bind(system="skia.degradation", instance_id=instance_id)

        self._event_bus: EventBus | None = None
        self.snapshot = DegradationSnapshot()

        self._running = False
        self._task: asyncio.Task[None] | None = None

        # Per-tick rates (rates are per-day; divide by hours_per_day)
        _hours_per_day = 24.0
        self._memory_per_tick = self._config.memory_decay_rate / _hours_per_day
        self._config_per_tick = self._config.config_drift_rate / _hours_per_day
        self._hypothesis_per_tick = self._config.hypothesis_staleness_rate / _hours_per_day

    # ── Wiring ────────────────────────────────────────────────────

    def set_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    # ── Evo-evolvable parameter updates ──────────────────────────

    def update_rates(
        self,
        *,
        memory_decay_rate: float | None = None,
        config_drift_rate: float | None = None,
        hypothesis_staleness_rate: float | None = None,
        tick_interval_s: float | None = None,
    ) -> None:
        """Hot-reload degradation rates from Evo parameter adjustments.

        The organism must be able to evolve its own entropy resistance.
        Rates that are too low make the organism complacent (no pressure to
        consolidate/learn). Rates that are too high kill it before it can adapt.
        Evo discovers the sweet spot through natural selection.
        """
        _hours_per_day = 24.0
        if memory_decay_rate is not None:
            self._config.memory_decay_rate = max(0.001, min(0.5, memory_decay_rate))
            self._memory_per_tick = self._config.memory_decay_rate / _hours_per_day
        if config_drift_rate is not None:
            self._config.config_drift_rate = max(0.001, min(0.5, config_drift_rate))
            self._config_per_tick = self._config.config_drift_rate / _hours_per_day
        if hypothesis_staleness_rate is not None:
            self._config.hypothesis_staleness_rate = max(0.001, min(0.5, hypothesis_staleness_rate))
            self._hypothesis_per_tick = self._config.hypothesis_staleness_rate / _hours_per_day
        if tick_interval_s is not None:
            self._config.tick_interval_s = max(60.0, min(86400.0, tick_interval_s))
        self._log.info(
            "degradation_rates_updated",
            memory=self._config.memory_decay_rate,
            config=self._config.config_drift_rate,
            hypothesis=self._config.hypothesis_staleness_rate,
            tick_s=self._config.tick_interval_s,
        )

    def get_evolvable_parameters(self) -> dict[str, float]:
        """Return current degradation rates for genome extraction.

        These parameters are heritable - child organisms inherit the parent's
        evolved entropy resistance rates.
        """
        return {
            "degradation_memory_decay_rate": self._config.memory_decay_rate,
            "degradation_config_drift_rate": self._config.config_drift_rate,
            "degradation_hypothesis_staleness_rate": self._config.hypothesis_staleness_rate,
            "degradation_tick_interval_s": self._config.tick_interval_s,
        }

    def estimate_time_to_critical_s(self) -> float | None:
        """Estimate seconds until degradation_pressure reaches 0.8 (critical zone).

        Returns None if pressure is stable or decreasing, or if already above 0.8.
        Uses the last 6 ticks to compute a linear trend.
        """
        current = self.snapshot.degradation_pressure
        if current >= 0.8:
            return 0.0

        # Need tick history for trend - use tick_count and current pressure
        # Simple estimate: current rate of pressure increase per tick
        tick_count = self.snapshot.tick_count
        if tick_count < 2:
            return None

        # Pressure added per tick (without counteraction)
        pressure_per_tick = (
            self._memory_per_tick * 0.40
            + self._config_per_tick * 0.20
            + self._hypothesis_per_tick * 0.40
        )

        if pressure_per_tick <= 0:
            return None

        remaining = 0.8 - current
        ticks_to_critical = remaining / pressure_per_tick
        return ticks_to_critical * self._config.tick_interval_s

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the independent hourly degradation tick loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._tick_loop(), name="skia_degradation_engine"
        )
        self._log.info(
            "degradation_engine_started",
            memory_decay_rate=self._config.memory_decay_rate,
            config_drift_rate=self._config.config_drift_rate,
            hypothesis_staleness_rate=self._config.hypothesis_staleness_rate,
            tick_interval_s=self._config.tick_interval_s,
        )

    async def stop(self) -> None:
        """Stop the degradation tick loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._log.info("degradation_engine_stopped")

    # ── Tick Loop ─────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        """
        Fire once per tick_interval_s (default: 1 hour).

        First tick is delayed by the full interval so the organism gets
        time to initialise before entropy starts accumulating.
        """
        while self._running:
            try:
                await asyncio.sleep(self._config.tick_interval_s)
            except asyncio.CancelledError:
                return

            if not self._running:
                return

            try:
                await self.tick()
            except Exception as exc:
                self._log.error("degradation_tick_error", error=str(exc))

    async def tick(self) -> DegradationSnapshot:
        """
        Execute one degradation tick.

        Accumulates entropy in the snapshot, then emits three targeted
        degradation events followed by the aggregate DEGRADATION_TICK.

        Returns the updated snapshot for direct callers (e.g. tests).
        """
        self.snapshot.tick_count += 1
        self.snapshot.last_tick_at = time.monotonic()

        # Accumulate entropy this tick
        memory_lost_this_tick = self._memory_per_tick
        config_drifted_this_tick = self._config_per_tick
        hypotheses_staled_this_tick = self._hypothesis_per_tick

        self.snapshot.cumulative_memory_fidelity_lost += memory_lost_this_tick
        self.snapshot.cumulative_config_drift += config_drifted_this_tick
        self.snapshot.cumulative_hypothesis_staleness += hypotheses_staled_this_tick

        tick_n = self.snapshot.tick_count

        self._log.info(
            "degradation_tick",
            tick=tick_n,
            memory_lost=round(memory_lost_this_tick, 6),
            config_drift=round(config_drifted_this_tick, 6),
            hypothesis_staleness=round(hypotheses_staled_this_tick, 6),
            cumulative_pressure=round(self.snapshot.degradation_pressure, 4),
        )

        # ── Emit targeted degradation events ──────────────────────
        # Systems respond autonomously. If they don't, pressure rises.
        from systems.synapse.types import SynapseEventType as _SET

        await self._emit(_SET.MEMORY_DEGRADATION, {
            "fidelity_loss_rate": memory_lost_this_tick,
            "affected_episode_age_hours": _DEGRADED_EPISODE_AGE_HOURS,
            "instance_id": self._instance_id,
            "tick_number": tick_n,
        })

        await self._emit(_SET.CONFIG_DRIFT, {
            "drift_rate": config_drifted_this_tick,
            # Rough estimate: 1% drift across ~23 learnable Simula params
            "num_params_affected": max(1, int(23 * config_drifted_this_tick * 10)),
            "instance_id": self._instance_id,
            "tick_number": tick_n,
        })

        await self._emit(_SET.HYPOTHESIS_STALENESS, {
            "staleness_rate": hypotheses_staled_this_tick,
            # Optimistic estimate; actual count from Evo's hypothesis store
            "affected_hypothesis_count": -1,  # -1 = "all unvalidated"
            "instance_id": self._instance_id,
            "tick_number": tick_n,
        })

        # ── Emit aggregate DEGRADATION_TICK for VitalityCoordinator ──
        await self._emit(_SET.DEGRADATION_TICK, {
            "memory_fidelity_lost": memory_lost_this_tick,
            "configs_drifted": max(1, int(23 * config_drifted_this_tick * 10)),
            "hypotheses_staled": -1,
            "tick_number": tick_n,
            "cumulative_pressure": self.snapshot.degradation_pressure,
            "instance_id": self._instance_id,
        })

        return self.snapshot

    # ── Counteraction API (called by VitalityCoordinator on counteraction events) ──

    def on_memory_consolidated(self, fraction: float = 0.5) -> None:
        """
        Oneiros/Soma ran consolidation - counteracts memory fidelity loss.

        fraction: how much of the accumulated loss to reverse (default 0.5 = 50%).
        """
        before = self.snapshot.cumulative_memory_fidelity_lost
        self.snapshot.counteract_memory(self.snapshot.cumulative_memory_fidelity_lost * fraction)
        self._log.debug(
            "memory_degradation_counteracted",
            before=round(before, 6),
            after=round(self.snapshot.cumulative_memory_fidelity_lost, 6),
        )

    def on_config_optimised(self, fraction: float = 0.8) -> None:
        """
        Evo ran parameter optimisation - counteracts config drift.

        fraction: how much of the accumulated drift to reverse (default 0.8 = 80%).
        """
        before = self.snapshot.cumulative_config_drift
        self.snapshot.counteract_config(self.snapshot.cumulative_config_drift * fraction)
        self._log.debug(
            "config_drift_counteracted",
            before=round(before, 6),
            after=round(self.snapshot.cumulative_config_drift, 6),
        )

    def on_hypotheses_revalidated(self, fraction: float = 0.6) -> None:
        """
        Evo re-validated hypotheses - counteracts hypothesis staleness.

        fraction: how much of the accumulated staleness to reverse (default 0.6 = 60%).
        """
        before = self.snapshot.cumulative_hypothesis_staleness
        self.snapshot.counteract_hypotheses(
            self.snapshot.cumulative_hypothesis_staleness * fraction
        )
        self._log.debug(
            "hypothesis_staleness_counteracted",
            before=round(before, 6),
            after=round(self.snapshot.cumulative_hypothesis_staleness, 6),
        )

    # ── Internal ──────────────────────────────────────────────────

    async def _emit(self, event_type_name: str | Any, data: dict[str, Any]) -> None:
        if not self._event_bus:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            if isinstance(event_type_name, SynapseEventType):
                et = event_type_name
            else:
                et = SynapseEventType(event_type_name)
            await self._event_bus.emit(SynapseEvent(
                event_type=et,
                data=data,
                source_system="skia",
            ))
        except Exception as exc:
            self._log.warning("degradation_emit_failed", event=event_type_name, error=str(exc))
