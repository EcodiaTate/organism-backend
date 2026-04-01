"""
EcodiaOS - Synapse Service

The autonomic nervous system. Synapse is the heartbeat of EOS - it drives
the cognitive cycle clock, monitors system health, allocates resources,
detects emergent cognitive rhythms, and measures cross-system coherence.

Synapse is invisible when it works. It is the heartbeat, the circulation,
the autonomic regulation that keeps everything alive. You don't notice
your nervous system until it fails - and Synapse is designed never to fail.

Zero LLM tokens consumed. Pure computation, monitoring, coordination.

Lifecycle:
  initialize()          - build all sub-systems
  register_system()     - register a cognitive system for management
  start_clock()         - start the theta rhythm
  start_health_monitor()- start background health polling
  stop()                - graceful shutdown
  health()              - self-health report

The _on_cycle callback (called by CognitiveClock after every tick):
  1. Feed CycleResult into EmergentRhythmDetector (every cycle)
  2. Feed broadcast data into CoherenceMonitor (every cycle)
  3. Trigger CoherenceMonitor.compute() (every 50 cycles)
  4. Trigger ResourceAllocator.capture_snapshot() (every 33 cycles)
  5. Record telemetry to MetricCollector
  6. Emit CYCLE_COMPLETED event to Redis for Alive
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.re_training import RETrainingExample
from systems.synapse.clock import CognitiveClock
from systems.synapse.coherence import CoherenceMonitor
from systems.synapse.degradation import DegradationManager
from systems.synapse.event_bus import EventBus
from systems.synapse.health import HealthMonitor
from systems.synapse.metabolism import MetabolicTracker
from systems.synapse.resources import ResourceAllocator
from systems.synapse.rhythm import EmergentRhythmDetector
from systems.synapse.types import (
    BaseResourceAllocator,
    BaseRhythmStrategy,
    ClockState,
    CoherenceSnapshot,
    CycleResult,
    MetabolicSnapshot,
    MetabolicState,
    RhythmSnapshot,
    SomaTickEvent,
    SynapseEvent,
    SynapseEventType,
)

if TYPE_CHECKING:
    from clients.model_hotswap import HotSwapManager
    from clients.redis import RedisClient
    from config import SynapseConfig
    from core.hotreload import NeuroplasticityBus
    from systems.fovea.gateway import AtuneService
    from telemetry.metrics import MetricCollector

logger = structlog.get_logger("systems.synapse")

# How often to compute coherence (in cycles)
_COHERENCE_INTERVAL: int = 50

# How often to capture a resource snapshot (in cycles)
_RESOURCE_SNAPSHOT_INTERVAL: int = 33

# How often to rebalance resource allocations (in cycles)
_REBALANCE_INTERVAL: int = 100

# How often to snapshot metabolic state and emit pressure events (in cycles)
_METABOLIC_INTERVAL: int = 50

# Burn rate threshold (USD/hour) above which METABOLIC_PRESSURE fires
_METABOLIC_PRESSURE_THRESHOLD_USD_HR: float = 1.0


class SynapseService:
    """
    Synapse - the EOS autonomic nervous system.

    Coordinates six sub-systems:
      CognitiveClock          - theta rhythm driving Atune
      HealthMonitor           - background health polling
      ResourceAllocator       - adaptive resource budgets
      DegradationManager      - graceful fallback on failure
      EmergentRhythmDetector  - meta-cognitive state detection
      CoherenceMonitor        - cross-system integration quality
      EventBus                - dual-output event publication
    """

    system_id: str = "synapse"

    def __init__(
        self,
        atune: AtuneService,
        config: SynapseConfig,
        redis: RedisClient | None = None,
        metrics: MetricCollector | None = None,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
        hot_swap_manager: HotSwapManager | None = None,
    ) -> None:
        self._atune = atune
        self._config = config
        self._redis = redis
        self._metrics = metrics
        self._neuroplasticity_bus = neuroplasticity_bus
        self._hot_swap_manager = hot_swap_manager
        self._logger = logger.bind(system="synapse")
        self._initialized: bool = False
        self._stopping: bool = False  # Set during stop() to suppress false clock-death alerts

        # Sub-systems
        self._event_bus = EventBus(redis=redis)
        self._clock = CognitiveClock(atune=atune, config=config)
        self._health = HealthMonitor(config=config, event_bus=self._event_bus)
        self._resources: BaseResourceAllocator = ResourceAllocator()
        self._degradation = DegradationManager(
            event_bus=self._event_bus,
            health_monitor=self._health,
        )
        self._rhythm = EmergentRhythmDetector(event_bus=self._event_bus)
        self._coherence = CoherenceMonitor(event_bus=self._event_bus)
        self._metabolism = MetabolicTracker()

        # Cycle counter for periodic sub-system triggers
        self._cycle_count: int = 0

        # Grid metabolism state - updated by GridMetabolismSensor events
        self._grid_state: MetabolicState = MetabolicState.NORMAL

        # Thymos (wired after construction) - used to feed the CognitiveStallSentinel
        # per theta tick so it can detect catatonic / stalled organism states.
        self._thymos: Any = None

        # Benchmarks (wired after construction) - receives KPI snapshot every 50 cycles
        # so the organism can self-measure heartbeat health and coherence quality.
        self._benchmarks: Any = None

        # Soma reference for organism telemetry (emotions, interoception state).
        # Stored here in addition to _clock so the 50-cycle block can query it.
        self._soma: Any = None

        # Nova + Evo references for CognitiveStallSentinel activity deltas.
        # Counters from the *previous* cycle are tracked here so we can derive
        # "did nova produce an intent this cycle?" without modifying nova/evo.
        self._nova: Any = None
        self._evo: Any = None
        self._nova_intents_prev: int = 0
        self._evo_evidence_prev: int = 0

        # Track previous rhythm state so we can detect STRESS transitions and
        # emit FOVEA_PREDICTION_ERROR on entry - rhythm state = self-awareness signal.
        self._prev_rhythm_state: str = ""

        # Cached starvation level from Oikos METABOLIC_PRESSURE events.
        # Relayed on every SomaTick so downstream systems read it without polling.
        self._cached_starvation_level: str = "nominal"

        # Economic state from Oikos - used to derive events_per_dollar metrics
        self._cached_burn_rate_usd: float = 0.0
        self._cached_liquid_balance_usd: float = 0.0

        # Interoception signal cache - updated by INTEROCEPTIVE_ALERT subscription.
        # Used to populate OrganismTelemetry without blocking the 50-cycle emit.
        self._interoception_cache: dict[str, Any] = {
            "error_rate_per_min": None,
            "cascade_pressure": False,
            "latency_spike_active": False,
        }

        # Persona handle cache - updated reactively on PERSONA_CREATED / PERSONA_EVOLVED.
        # Included in ORGANISM_TELEMETRY so Nova knows the organism's public identity.
        self._cached_persona_handle: str | None = None

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Build all sub-systems and wire inter-dependencies."""
        if self._initialized:
            return

        # Wire health → degradation
        self._health.set_degradation_manager(self._degradation)

        # Set the per-cycle callback on the clock
        self._clock.set_on_cycle(self._on_cycle)

        # Wire event bus into clock so it can emit THETA_CYCLE_START/OVERRUN (Spec 09 §18 P7/P8)
        self._clock.set_event_bus(self._event_bus)

        # Subscribe to grid metabolism changes to adapt clock timing
        self._event_bus.subscribe(
            SynapseEventType.GRID_METABOLISM_CHANGED,
            self._on_grid_metabolism_changed,
        )

        # Cache starvation level from Oikos so it can be relayed on every SomaTick
        self._event_bus.subscribe(
            SynapseEventType.METABOLIC_PRESSURE,
            self._on_metabolic_pressure_for_relay,
        )

        # Subscribe to Oikos economic state for events_per_dollar / cost_per_event metrics
        self._event_bus.subscribe(
            SynapseEventType.ECONOMIC_STATE_UPDATED,
            self._on_economic_state_updated,
        )

        # M1: Subscribe to Oikos revenue and starvation events to drive MetabolicTracker
        # reactively instead of requiring external injection (Spec 09 §18 Gap / §20 Gap #6).
        self._event_bus.subscribe(
            SynapseEventType.REVENUE_INJECTED,
            self._on_oikos_revenue_injected,
        )
        self._event_bus.subscribe(
            SynapseEventType.STARVATION_WARNING,
            self._on_oikos_starvation_warning,
        )

        # Subscribe to INTEROCEPTIVE_ALERT to cache pressure signals for OrganismTelemetry
        self._event_bus.subscribe(
            SynapseEventType.INTEROCEPTIVE_ALERT,
            self._on_interoceptive_alert_for_cache,
        )

        # Subscribe to persona events to cache the organism's public handle for telemetry
        self._event_bus.subscribe(
            SynapseEventType.PERSONA_CREATED,
            self._on_persona_update,
        )
        self._event_bus.subscribe(
            SynapseEventType.PERSONA_EVOLVED,
            self._on_persona_update,
        )

        # Register with NeuroplasticityBus for hot-reload of allocators & rhythm strategies
        if self._neuroplasticity_bus is not None:
            self._neuroplasticity_bus.register(
                base_class=BaseResourceAllocator,
                registration_callback=self._on_allocator_evolved,
                system_id="synapse",
            )
            self._neuroplasticity_bus.register(
                base_class=BaseRhythmStrategy,
                registration_callback=self._on_rhythm_strategy_evolved,
                system_id="synapse",
            )

        self._initialized = True
        self._logger.info("synapse_initialized")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma service into the clock (step 0 of theta cycle) and cache for telemetry."""
        self._clock.set_soma(soma)
        self._soma = soma

    def set_thymos(self, thymos: Any) -> None:
        """
        Wire Thymos so the CognitiveStallSentinel is fed every theta tick.

        Must be called after thymos.initialize() so the sentinel is built.
        The actual record_cycle() call happens in _on_cycle() after every tick.
        """
        self._thymos = thymos
        self._logger.info("thymos_wired_to_synapse", system="synapse")

    def set_benchmarks(self, benchmarks: Any) -> None:
        """Wire Benchmarks so Synapse can emit KPI snapshots every 50 cycles."""
        self._benchmarks = benchmarks
        self._logger.info("benchmarks_wired_to_synapse")

    def set_nova(self, nova: Any) -> None:
        """Wire Nova so _on_cycle can read intent-count deltas for the stall sentinel."""
        self._nova = nova
        self._nova_intents_prev = getattr(nova, "_total_intents_issued", 0)
        self._logger.info("nova_wired_to_synapse_for_stall_sentinel", system="synapse")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so _on_cycle can read evidence-count deltas for the stall sentinel."""
        self._evo = evo
        self._evo_evidence_prev = getattr(evo, "_total_evidence_evaluations", 0)
        self._logger.info("evo_wired_to_synapse_for_stall_sentinel", system="synapse")

    def set_instance_id(self, instance_id: str) -> None:
        """
        Set the organism's instance identity on the EventBus (Spec 09 §18 M4/SG4).

        Stamps every future event with instance_id and namespaces the Redis channel
        to `synapse_events:{instance_id}` so Federation/Mitosis instances don't
        cross-pollute each other's pub/sub streams.

        Must be called before start_clock() to take effect on the first tick.
        Typically wired from AppConfig.instance_id at startup.
        """
        self._event_bus.set_instance_id(instance_id)

    def set_hot_swap_manager(self, manager: HotSwapManager) -> None:
        """Wire the model hot-swap manager for probation monitoring."""
        self._hot_swap_manager = manager

    def register_system(self, system: Any) -> None:
        """
        Register a cognitive system for health monitoring and degradation.

        The system must have:
          - system_id: str
          - async health() -> dict[str, Any]
        """
        self._health.register(system)
        self._degradation.register_system(system)
        # Update coherence monitor with total system count
        self._coherence.set_total_systems(len(self._health.get_all_records()))

    async def start_clock(self) -> None:
        """Start the cognitive cycle clock (the heartbeat)."""
        if not self._initialized:
            raise RuntimeError("SynapseService.initialize() must be called first")

        clock_task = self._clock.start()

        # Supervision callback: if the supervised clock task exits unexpectedly
        # (e.g., exhausted restarts), emit CRITICAL on the bus so Thymos/Alive
        # can detect a permanently stopped heartbeat.
        def _on_clock_done(task: asyncio.Task[None]) -> None:
            if self._stopping or task.cancelled():
                return  # Normal shutdown - not an error
            exc = task.exception()
            if exc is not None:
                self._logger.critical(
                    "clock_task_died_unexpectedly",
                    error=str(exc),
                )
            else:
                self._logger.critical(
                    "clock_task_exited_unexpectedly",
                    note="Cognitive clock stopped without being asked to - organism heartbeat lost",
                )
            asyncio.create_task(
                self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_FAILED,
                    data={
                        "system_id": "synapse",
                        "task": "cognitive_clock",
                        "reason": "clock_exhausted_restarts",
                    },
                )),
                name="synapse_clock_death_event",
            )

        clock_task.add_done_callback(_on_clock_done)

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CLOCK_STARTED,
            data={"period_ms": self._config.cycle_period_ms},
        ))

        self._logger.info(
            "clock_started",
            period_ms=self._config.cycle_period_ms,
            min_ms=self._config.min_cycle_period_ms,
            max_ms=self._config.max_cycle_period_ms,
        )

    async def start_health_monitor(self) -> None:
        """Start the background health monitoring loop."""
        if not self._initialized:
            raise RuntimeError("SynapseService.initialize() must be called first")

        self._health.start()
        self._logger.info(
            "health_monitor_started",
            interval_ms=self._config.health_check_interval_ms,
        )

    async def stop(self) -> None:
        """Graceful shutdown of all sub-systems."""
        self._stopping = True
        self._logger.info("synapse_stopping")

        # Deregister from NeuroplasticityBus
        if self._neuroplasticity_bus is not None:
            self._neuroplasticity_bus.deregister(BaseResourceAllocator)
            self._neuroplasticity_bus.deregister(BaseRhythmStrategy)

        await self._clock.stop()
        await self._health.stop()

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CLOCK_STOPPED,
            data={"total_cycles": self._cycle_count},
        ))

        self._logger.info(
            "synapse_stopped",
            total_cycles=self._cycle_count,
            rhythm_state=self._rhythm.current_state.value,
            coherence=self._coherence.latest.composite,
        )

    # ─── Health ──────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Self-health report (implements ManagedSystem protocol)."""
        # Compute events_per_dollar and cost_per_event from cached economic state
        total_events = self._event_bus._total_emitted
        events_per_dollar = 0.0
        cost_per_event = 0.0
        if self._cached_burn_rate_usd > 0 and total_events > 0:
            cost_per_event = self._cached_burn_rate_usd / max(1, total_events)
        if self._cached_burn_rate_usd > 0:
            events_per_dollar = total_events / max(0.001, self._cached_burn_rate_usd)

        report: dict[str, Any] = {
            "status": "healthy" if self._initialized else "starting",
            "cycle_count": self._cycle_count,
            "safe_mode": self._health.is_safe_mode,
            "rhythm_state": self._rhythm.current_state.value,
            "coherence_composite": self._coherence.latest.composite,
            "metabolic_deficit_usd": round(self._metabolism.rolling_deficit_usd, 6),
            "burn_rate_usd_per_hour": round(self._metabolism.burn_rate_usd_per_hour, 4),
            "events_per_dollar": round(events_per_dollar, 2),
            "cost_per_event": round(cost_per_event, 8),
        }
        if self._hot_swap_manager is not None:
            report["hot_swap"] = await self._hot_swap_manager.health()
        return report

    # ─── Safe Mode ───────────────────────────────────────────────────

    @property
    def is_safe_mode(self) -> bool:
        return self._health.is_safe_mode

    async def set_safe_mode(self, enabled: bool, reason: str = "") -> None:
        """Manually toggle safe mode (admin API)."""
        await self._health.set_safe_mode(enabled, reason)
        if enabled:
            self._clock.pause()
            await self._emit_evolutionary_observable(
                "degradation_triggered", 1.0, is_novel=True,
                metadata={"reason": reason or "safe_mode_entered"},
            )
        else:
            self._clock.resume()
        # ── RE training: degradation transition ──
        asyncio.ensure_future(self._emit_re_training_example(
            category="degradation_transition",
            instruction="Decide whether to enter or exit safe mode (graceful degradation).",
            input_context=f"enabled={enabled}, reason={reason[:200]}",
            output=f"safe_mode={'entered' if enabled else 'exited'}, clock={'paused' if enabled else 'resumed'}",
            outcome_quality=0.4 if enabled else 0.8,
        ))

    # ─── Clock Control (admin API) ───────────────────────────────────

    def pause_clock(self) -> None:
        """Pause the cognitive cycle clock (e.g., for maintenance)."""
        self._clock.pause()

    def resume_clock(self) -> None:
        """Resume the cognitive cycle clock."""
        self._clock.resume()

    def set_clock_speed(self, hz: float) -> None:
        """Override base clock frequency (1–20 Hz)."""
        self._clock.set_speed(hz)

    # ─── Accessors ───────────────────────────────────────────────────

    @property
    def clock_state(self) -> ClockState:
        return self._clock.state

    @property
    def rhythm_snapshot(self) -> RhythmSnapshot:
        return self._rhythm._build_snapshot()

    @property
    def coherence_snapshot(self) -> CoherenceSnapshot:
        return self._coherence.latest

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def metabolic_snapshot(self) -> MetabolicSnapshot:
        return self._metabolism.snapshot()

    @property
    def metabolic_deficit(self) -> float:
        """Current rolling deficit in USD - how much the organism owes."""
        return self._metabolism.rolling_deficit_usd

    @property
    def metabolism(self) -> MetabolicTracker:
        """Direct access for callers that need log_usage or inject_revenue."""
        return self._metabolism


    async def inject_revenue(
        self,
        amount_usd: float,
        source: str = "external",
    ) -> None:
        """
        Record incoming revenue and emit REVENUE_INJECTED on the event bus.

        This is the preferred entry point for revenue injections. It keeps
        the MetabolicTracker updated AND ensures Memory encodes the event as
        a salience=1.0 episode so the organism learns what actions lead to income.

        Args:
            amount_usd: Revenue amount in USD.
            source: Human-readable label for the revenue origin
                    (e.g. "stripe", "on-chain-fee", "client-payment").
        """
        from systems.synapse.types import SynapseEvent, SynapseEventType

        self._metabolism.inject_revenue(amount_usd)

        event = SynapseEvent(
            event_type=SynapseEventType.REVENUE_INJECTED,
            data={
                "amount_usd": round(amount_usd, 8),
                "source": source,
                "new_deficit_usd": round(self._metabolism.rolling_deficit_usd, 6),
            },
            source_system="synapse",
        )
        try:
            await self._event_bus.emit(event)
        except Exception as exc:
            logger.error("revenue_injected_event_emit_failed", error=str(exc))
    # ─── Stats ───────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "cycle_count": self._cycle_count,
            "clock": self._clock.state.model_dump(),
            "health": self._health.stats,
            "degradation": self._degradation.stats,
            "resources": self._resources.stats,
            "rhythm": self._rhythm.stats,
            "coherence": self._coherence.stats,
            "metabolism": self._metabolism.stats,
            "event_bus": self._event_bus.stats,
        }

    # ─── Grid Metabolism Reaction ────────────────────────────────────

    async def _on_grid_metabolism_changed(self, event: SynapseEvent) -> None:
        """
        Adapt the cognitive clock's base period to the physical grid's carbon
        intensity.

        CONSERVATION: 150 ms → 1 000 ms - drastically reduce idle CPU cycles
                      while the grid is carbon-heavy.
        NORMAL:       restore the configured base period.
        GREEN_SURPLUS: restore the configured base period (clock is already at
                       full speed; heavy work is triggered in Oneiros/Simula).
        """
        raw = event.data.get("state", "")
        try:
            new_state = MetabolicState(raw)
        except ValueError:
            self._logger.warning(
                "grid_metabolism_unknown_state",
                raw_state=raw,
            )
            return

        old_state = self._grid_state
        if new_state == old_state:
            return

        self._grid_state = new_state

        if new_state == MetabolicState.CONSERVATION:
            # Slow the heartbeat dramatically to shed idle CPU load
            self._clock.set_speed(1.0)  # 1 Hz → 1 000 ms period
            self._logger.info(
                "grid_conservation_clock_throttled",
                from_state=old_state.value,
                to_state=new_state.value,
                new_period_ms=1000,
            )
            # Spec 09 §18 P6 - CONSERVATION_MODE_ENTERED event
            with contextlib.suppress(Exception):
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.CONSERVATION_MODE_ENTERED,
                    source_system="synapse",
                    data={
                        "trigger": "grid_carbon_intensity",
                        "new_period_ms": 1000,
                        "from_state": old_state.value,
                    },
                ))
        else:
            # NORMAL or GREEN_SURPLUS - restore configured base frequency
            configured_hz = 1000.0 / self._config.cycle_period_ms
            self._clock.set_speed(configured_hz)
            self._logger.info(
                "grid_recovery_clock_restored",
                from_state=old_state.value,
                to_state=new_state.value,
                new_period_ms=self._config.cycle_period_ms,
            )
            # Spec 09 §18 P6 - CONSERVATION_MODE_EXITED event (only if was CONSERVATION)
            if old_state == MetabolicState.CONSERVATION:
                with contextlib.suppress(Exception):
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.CONSERVATION_MODE_EXITED,
                        source_system="synapse",
                        data={
                            "restored_period_ms": self._config.cycle_period_ms,
                            "to_state": new_state.value,
                        },
                    ))

    async def _on_metabolic_pressure_for_relay(self, event: SynapseEvent) -> None:
        """Cache starvation_level from Oikos METABOLIC_PRESSURE for SomaTick relay."""
        level = event.data.get("starvation_level", "")
        if level:
            self._cached_starvation_level = level

    async def _on_economic_state_updated(self, event: SynapseEvent) -> None:
        """
        Cache Oikos economic state for Synapse-level efficiency metrics.

        Enables computation of events_per_dollar and cost_per_event.
        """
        burn_rate = event.data.get("burn_rate_usd")
        if burn_rate is not None:
            try:
                self._cached_burn_rate_usd = float(str(burn_rate))
            except (ValueError, TypeError):
                pass
        balance = event.data.get("liquid_balance_usd")
        if balance is not None:
            try:
                self._cached_liquid_balance_usd = float(str(balance))
            except (ValueError, TypeError):
                pass

    async def _on_oikos_revenue_injected(self, event: SynapseEvent) -> None:
        """
        React to Oikos REVENUE_INJECTED by forwarding the amount to MetabolicTracker.

        This closes the M1 gap (Spec 09 §18): MetabolicTracker's rolling deficit
        was previously only updated via direct `inject_revenue()` calls, making
        the deficit grow unboundedly when Oikos earned revenue without notifying
        Synapse.  By subscribing here, any revenue Oikos earns - bounty payouts,
        yield harvests, client fees - is immediately reflected in the deficit.

        We do NOT re-emit REVENUE_INJECTED (the MetabolicTracker.inject_revenue()
        already emits it), so there is no double-emission risk.
        """
        amount_raw = event.data.get("amount_usd", 0.0)
        try:
            amount = float(str(amount_raw))
        except (ValueError, TypeError):
            return
        if amount <= 0.0:
            return

        source = str(event.data.get("source", "oikos"))
        # MetabolicTracker.inject_revenue() updates the rolling deficit and emits
        # REVENUE_INJECTED itself - we only call the tracker, not inject_revenue()
        # on self (which would double-emit).
        self._metabolism.inject_revenue(amount)
        self._logger.info(
            "metabolic_revenue_injected_reactively",
            amount_usd=round(amount, 8),
            source=source,
            new_deficit_usd=round(self._metabolism.rolling_deficit_usd, 6),
        )

    async def _on_oikos_starvation_warning(self, event: SynapseEvent) -> None:
        """
        React to Oikos STARVATION_WARNING by caching the level for SomaTick relay.

        The cached level is read each theta tick by `_on_cycle` (via SomaTickEvent)
        and propagated to Soma so allostatic control can account for economic danger
        without polling Oikos directly (Spec 09 §18 M1).
        """
        level = str(event.data.get("starvation_level", "")).strip()
        if not level:
            return
        self._cached_starvation_level = level
        self._logger.warning(
            "starvation_warning_received",
            starvation_level=level,
            runway_days=event.data.get("runway_days", "unknown"),
            liquid_balance_usd=event.data.get("liquid_balance_usd", "unknown"),
        )

    async def _on_interoceptive_alert_for_cache(self, event: SynapseEvent) -> None:
        """
        Cache interoception signal state so ORGANISM_TELEMETRY can include it
        without reading it from a separate source in the hot-path 50-cycle block.
        """
        alert_type = str(event.data.get("alert_type", ""))
        if alert_type == "error_rate":
            self._interoception_cache["error_rate_per_min"] = event.data.get("value")
        elif alert_type == "cascade":
            self._interoception_cache["cascade_pressure"] = True
        elif alert_type == "latency":
            self._interoception_cache["latency_spike_active"] = True

    async def _on_persona_update(self, event: SynapseEvent) -> None:
        """Cache the organism's public handle for inclusion in ORGANISM_TELEMETRY."""
        handle = event.data.get("handle") if event.data else None
        if handle:
            self._cached_persona_handle = handle

    # ─── NeuroplasticityBus callbacks ────────────────────────────────

    def _on_allocator_evolved(self, new_allocator: BaseResourceAllocator) -> None:
        """
        Hot-swap the resource allocator when Simula evolves a new one.

        The old allocator's accumulated load observations are intentionally
        discarded - evolved logic starts with fresh observations.  The swap
        happens between cycles so the active theta tick is never disrupted.
        """
        old_name = self._resources.allocator_name
        self._resources = new_allocator
        self._logger.info(
            "resource_allocator_evolved",
            old=old_name,
            new=new_allocator.allocator_name,
        )

    def _on_rhythm_strategy_evolved(self, new_strategy: BaseRhythmStrategy) -> None:
        """
        Hot-swap the rhythm classification strategy.

        The EmergentRhythmDetector's rolling window and hysteresis state
        are preserved - only the classification algorithm changes.  This
        means the new strategy immediately has a full 100-cycle window
        of data to classify against, rather than starting cold.
        """
        self._rhythm.set_strategy(new_strategy)
        self._logger.info(
            "rhythm_strategy_evolved",
            new=new_strategy.strategy_name,
        )

    # ─── Per-Cycle Callback ──────────────────────────────────────────

    async def _on_cycle(self, result: CycleResult) -> None:
        """
        Called by CognitiveClock after every theta tick.

        This is the central integration point where all sub-systems
        are fed cycle telemetry. The work here must be lightweight -
        heavy computation is deferred to periodic triggers.
        """
        self._cycle_count = result.cycle_number

        # ── 1. Feed rhythm detector (every cycle) ──
        coherence_stress = 0.0
        with contextlib.suppress(Exception):
            coherence_stress = self._atune.current_affect.coherence_stress

        await self._rhythm.update(result, coherence_stress=coherence_stress)

        # Detect rhythm state transitions → emit anomaly and narrative signals.
        current_rhythm = self._rhythm.current_state.value
        if current_rhythm != self._prev_rhythm_state:
            # Any significant state transition goes to Thread as an Episode so
            # the organism's narrative identity records its own cognitive rhythm changes.
            thread_worthy = {"stress", "flow", "deep_processing", "boredom"}
            if current_rhythm in thread_worthy or self._prev_rhythm_state in thread_worthy:
                with contextlib.suppress(Exception):
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.EPISODE_STORED,
                        source_system="synapse",
                        data={
                            "episode_id": f"rhythm_transition_{result.cycle_number}",
                            "source": "synapse:rhythm",
                            "summary": (
                                f"Cognitive rhythm shifted from {self._prev_rhythm_state!r} "
                                f"to {current_rhythm!r} at cycle {result.cycle_number}."
                            ),
                            "salience": 0.7 if current_rhythm == "stress" else 0.4,
                        },
                    ))

            # On STRESS entry, also signal Fovea: rhythm instability = self-model divergence.
            if current_rhythm == "stress":
                with contextlib.suppress(Exception):
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.FOVEA_PREDICTION_ERROR,
                        source_system="synapse",
                        data={
                            "source_system": "synapse",
                            "domain": "rhythm",
                            "magnitude": 0.75,
                            "trigger_id": f"stress_entry_cycle_{result.cycle_number}",
                            "coherence_stress": round(coherence_stress, 3),
                            "cycle": result.cycle_number,
                        },
                    ))

            await self._emit_evolutionary_observable(
                "rhythm_transition", 1.0, is_novel=True,
                metadata={
                    "from": self._prev_rhythm_state,
                    "to": current_rhythm,
                    "cycle": result.cycle_number,
                },
            )

        # ── RE training: rhythm classification ──
        if current_rhythm != self._prev_rhythm_state:
            asyncio.ensure_future(self._emit_re_training_example(
                category="rhythm_classification",
                instruction="Classify emergent cognitive rhythm from cycle telemetry and detect state transitions.",
                input_context=f"prev={self._prev_rhythm_state}, cycle={result.cycle_number}, arousal={result.arousal:.2f}, coherence_stress={coherence_stress:.2f}",
                output=f"rhythm={current_rhythm}, transition=True",
                outcome_quality=0.7 if current_rhythm not in ("stress",) else 0.3,
            ))
        self._prev_rhythm_state = current_rhythm

        # Push rhythm state to Atune so meta-attention can modulate salience
        # weights based on the organism's emergent cognitive state.
        with contextlib.suppress(Exception):  # Non-critical - meta-attention falls back to "normal"
            self._atune.set_rhythm_state(self._rhythm.current_state.value)

        # ── 2. Feed coherence monitor (every cycle) ──
        source = ""
        if result.had_broadcast and result.broadcast_id:
            source = result.broadcast_id[:8]  # Use broadcast ID prefix as source proxy

        # Drain handler latencies from EventBus so CoherenceMonitor can compute
        # real system_resonance and response_synchrony (not the 0.5 defaults).
        latency_sets = self._event_bus.drain_handler_latencies()
        # Flatten all dispatch rounds into a single per-cycle latency set
        cycle_latencies: list[float] = []
        responding = 0
        for lat_set in latency_sets:
            cycle_latencies.extend(lat_set)
            responding = max(responding, len(lat_set))

        self._coherence.record_broadcast(
            source=source,
            salience=result.salience_composite,
            had_content=result.had_broadcast,
            response_latencies=cycle_latencies if cycle_latencies else None,
            responding_systems=responding,
        )

        # ── 3. Periodic: compute coherence → adapt clock ──
        if self._cycle_count % _COHERENCE_INTERVAL == 0:
            snapshot = await self._coherence.compute()

            # Use coherence to modulate clock speed: low coherence → slow
            # down to give systems time to resynchronize.
            if snapshot is not None:
                # Activate drag when composite drops below 0.4
                if snapshot.composite < 0.4:
                    drag = (0.4 - snapshot.composite) / 0.4  # 0→1 as coherence drops
                    self._clock.set_coherence_drag(drag)

                    # Emit FOVEA_PREDICTION_ERROR: organism coherence diverged from
                    # expected baseline - this is a structural self-model anomaly.
                    with contextlib.suppress(Exception):
                        await self._event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.FOVEA_PREDICTION_ERROR,
                            source_system="synapse",
                            data={
                                "source_system": "synapse",
                                "domain": "coherence",
                                "magnitude": round(drag, 3),
                                "trigger_id": f"coherence_drop_cycle_{self._cycle_count}",
                                "coherence_composite": round(snapshot.composite, 3),
                                "phi": round(snapshot.phi, 3),
                                "cycle": self._cycle_count,
                            },
                        ))
                else:
                    self._clock.set_coherence_drag(0.0)

            # Emit COHERENCE_SNAPSHOT unconditionally every coherence interval
            # so Benchmarks and other consumers always receive the data.
            if snapshot is not None:
                with contextlib.suppress(Exception):
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.COHERENCE_SNAPSHOT,
                        source_system="synapse",
                        data={
                            "system_resonance": snapshot.system_resonance,
                            "response_synchrony": snapshot.response_synchrony,
                            "phi_approximation": snapshot.phi_approximation,
                            "broadcast_diversity": snapshot.broadcast_diversity,
                            "composite": snapshot.composite,
                            "window_cycles": snapshot.window_cycles,
                            "event_throughput": round(
                                self._event_bus._total_emitted / max(1, self._cycle_count), 3
                            ),
                            "cycle": self._cycle_count,
                        },
                    ))

            # Emit Benchmarks KPI so the organism's self-performance model
            # stays current on heartbeat quality, coherence, and metabolism.
            if self._benchmarks is not None:
                try:
                    clock_st = self._clock.state
                    await self._benchmarks.record_kpi(
                        system="synapse",
                        metrics={
                            "cycle_rate_hz": round(clock_st.actual_rate_hz, 3),
                            "coherence_composite": self._coherence.latest.composite,
                            "rhythm_state": self._rhythm.current_state.value,
                            "burn_rate_usd_per_hour": round(
                                self._metabolism.burn_rate_usd_per_hour, 4
                            ),
                            "clock_overrun_count": clock_st.overrun_count,
                            "safe_mode": self._health.is_safe_mode,
                        },
                    )
                except Exception:
                    pass  # KPI emission must never block the cycle

        # ── 4. Periodic: resource snapshot ──
        if self._cycle_count % _RESOURCE_SNAPSHOT_INTERVAL == 0:
            self._resources.capture_snapshot()

        # ── 5. Periodic: rebalance resources ──
        if self._cycle_count % _REBALANCE_INTERVAL == 0:
            self._resources.rebalance(self._clock.state.current_period_ms)
            # Emit RESOURCE_REBALANCE so Alive/Benchmarks can observe allocation changes (Spec 09 §18 P5)
            _alloc_data = {
                "allocations": self._resources.stats.get("allocations", {}),
                "cycle": self._cycle_count,
                "period_ms": round(self._clock.state.current_period_ms, 2),
            }
            with contextlib.suppress(Exception):
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RESOURCE_REBALANCE,
                    source_system="synapse",
                    data=_alloc_data,
                ))
            # Also emit RESOURCE_REBALANCED (spec_checker canonical name)
            with contextlib.suppress(Exception):
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RESOURCE_REBALANCED,
                    source_system="synapse",
                    data=_alloc_data,
                ))
            # Emit RESOURCE_PRESSURE when total CPU is above threshold
            _snap = self._resources.stats
            _cpu = _snap.get("total_cpu_percent", 0.0) if isinstance(_snap, dict) else 0.0
            if _cpu > 80.0:
                with contextlib.suppress(Exception):
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.RESOURCE_PRESSURE,
                        source_system="synapse",
                        data={
                            "total_cpu_percent": _cpu,
                            "cycle": self._cycle_count,
                            "pressure_level": "high" if _cpu > 90.0 else "elevated",
                        },
                    ))
            await self._emit_evolutionary_observable(
                "allocation_rebalanced", 1.0, is_novel=True,
                metadata={"cycle": self._cycle_count},
            )
            # ── RE training: resource allocation decision ──
            asyncio.ensure_future(self._emit_re_training_example(
                category="resource_allocation",
                instruction="Rebalance resource allocations across cognitive systems based on cycle telemetry.",
                input_context=f"cycle={self._cycle_count}, period_ms={self._clock.state.current_period_ms}",
                output=f"rebalanced=True, rhythm={self._rhythm.current_state.value}",
                outcome_quality=0.7,
            ))

        # ── 5b. Periodic: metabolic snapshot + pressure event ──
        if self._cycle_count % _METABOLIC_INTERVAL == 0:
            meta_snap = self._metabolism.snapshot()

            # Emit METABOLIC_SNAPSHOT unconditionally so consumers always receive data
            with contextlib.suppress(Exception):
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.METABOLIC_SNAPSHOT,
                    source_system="synapse",
                    data={
                        "rolling_deficit_usd": meta_snap.rolling_deficit_usd,
                        "burn_rate_usd_per_hour": meta_snap.burn_rate_usd_per_hour,
                        "api_cost_usd_per_hour": meta_snap.api_cost_usd_per_hour,
                        "infra_cost_usd_per_hour": meta_snap.infra_cost_usd_per_hour,
                        "total_calls": meta_snap.total_calls,
                        "per_system_cost_usd": meta_snap.per_system_cost_usd,
                        "cycle": self._cycle_count,
                    },
                ))

            if meta_snap.burn_rate_usd_per_hour > _METABOLIC_PRESSURE_THRESHOLD_USD_HR:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.METABOLIC_PRESSURE,
                    data={
                        "rolling_deficit_usd": meta_snap.rolling_deficit_usd,
                        "burn_rate_usd_per_hour": meta_snap.burn_rate_usd_per_hour,
                        "api_cost_usd_per_hour": meta_snap.api_cost_usd_per_hour,
                        "infra_cost_usd_per_hour": meta_snap.infra_cost_usd_per_hour,
                        "total_calls": meta_snap.total_calls,
                        "per_system_cost_usd": meta_snap.per_system_cost_usd,
                    },
                ))
            # Reset window accumulator so next interval is a clean delta
            self._metabolism.reset_window()

            # ── 5c. ORGANISM_TELEMETRY - unified state broadcast ──────────────
            # Build once per 50-cycle interval using already-computed snapshots.
            # Gracefully degrades: missing sub-systems leave fields at defaults.
            with contextlib.suppress(Exception):
                from primitives.telemetry import OrganismTelemetry, SystemHealthSummary

                coherence_snap = self._coherence.latest
                resources_snap = self._resources.stats
                health_records = self._health.stats.get("systems", {})

                # Per-system health summaries
                _health_summaries: dict[str, SystemHealthSummary] = {
                    sid: SystemHealthSummary(
                        latency_ema_ms=info.get("latency_ema_ms", 0.0),
                        consecutive_misses=info.get("consecutive_misses", 0),
                        restart_count=info.get("restart_count", 0),
                        status=info.get("status", "healthy"),
                    )
                    for sid, info in health_records.items()
                }

                # Per-system CPU from resource allocator
                _cpu_per_system: dict[str, float] = dict(
                    resources_snap.get("system_loads", {})
                )
                _total_cpu = (
                    resources_snap.get("snapshot", {}).get("total_cpu_percent", 0.0)
                    if isinstance(resources_snap, dict)
                    else 0.0
                )

                # Active emotions from Soma (non-blocking, best-effort)
                _emotions: list[str] = []
                if self._soma is not None:
                    try:
                        _active = self._soma.get_active_emotions()
                        _emotions = [e.name for e in _active] if _active else []
                    except Exception:
                        pass

                # Compute two-ledger dependency ratio for organism self-awareness.
                # api_burn = per-token charges from organism's wallet
                # infra_burn = RunPod compute billed to human (subsidised cost)
                _api_burn = meta_snap.api_cost_usd_per_hour
                _infra_burn = meta_snap.infra_cost_usd_per_hour
                _total_burn = meta_snap.burn_rate_usd_per_hour
                _dependency_ratio = (
                    _infra_burn / _total_burn if _total_burn > 0 else 0.0
                )

                _telemetry = OrganismTelemetry(
                    # Metabolic
                    burn_rate_usd_per_hour=_total_burn,
                    runway_hours=meta_snap.hours_until_depleted,
                    per_provider_cost=dict(meta_snap.per_provider_cost_usd),
                    infra_cost_usd_per_hour=_infra_burn,
                    infra_resources=dict(
                        getattr(meta_snap, "infra_resources", {}) or {}
                    ),
                    # Two-ledger
                    api_burn_rate_usd_per_hour=_api_burn,
                    dependency_ratio=_dependency_ratio,
                    # Coherence
                    phi=coherence_snap.phi_approximation,
                    resonance=coherence_snap.system_resonance,
                    diversity=coherence_snap.broadcast_diversity,
                    synchrony=coherence_snap.response_synchrony,
                    coherence_composite=coherence_snap.composite,
                    # Rhythm
                    rhythm_state=self._rhythm.current_state.value,
                    cycles_in_rhythm_state=self._rhythm.stats.get("cycles_in_state", 0),
                    # Health + resources
                    health=_health_summaries,
                    cpu_per_system=_cpu_per_system,
                    total_cpu_pct=_total_cpu,
                    # Emotions
                    emotions=_emotions,
                    # Interoception (from cache updated by INTEROCEPTIVE_ALERT)
                    error_rate_per_min=self._interoception_cache["error_rate_per_min"],
                    cascade_pressure=self._interoception_cache["cascade_pressure"],
                    latency_spike_active=self._interoception_cache["latency_spike_active"],
                    # Persona - public identity handle (None until PersonaEngine seals)
                    persona_handle=self._cached_persona_handle,
                    # Provenance
                    cycle_number=self._cycle_count,
                )
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.ORGANISM_TELEMETRY,
                    source_system="synapse",
                    data=_telemetry.model_dump(mode="json"),
                ))
                # Reset cascade/latency flags after each broadcast so they don't
                # persist across cycles when the pressure is gone.
                self._interoception_cache["cascade_pressure"] = False
                self._interoception_cache["latency_spike_active"] = False

        # ── 6. Record telemetry ──
        if self._metrics is not None:
            try:
                await self._metrics.record(
                    "synapse", "cycle.latency_ms", result.elapsed_ms,
                )
                await self._metrics.record(
                    "synapse", "cycle.period_ms", result.budget_ms,
                )
                await self._metrics.record(
                    "synapse", "cycle.arousal", result.arousal,
                )
                if result.had_broadcast:
                    await self._metrics.record(
                        "synapse", "cycle.salience", result.salience_composite,
                    )
            except Exception:
                pass  # Telemetry failures must never block the cycle

        # ── 7. Emit cycle event to Redis for Alive ──
        cycle_data: dict[str, Any] = {
            "cycle": result.cycle_number,
            "elapsed_ms": result.elapsed_ms,
            "period_ms": result.budget_ms,
            "arousal": result.arousal,
            "had_broadcast": result.had_broadcast,
            "salience": result.salience_composite,
            "rhythm": self._rhythm.current_state.value,
            "metabolic_deficit_usd": self._metabolism.rolling_deficit_usd,
            "burn_rate_usd_per_hour": round(
                self._metabolism.burn_rate_usd_per_hour, 6,
            ),
        }
        if result.somatic is not None:
            cycle_data["soma"] = {
                "urgency": result.somatic.urgency,
                "dominant_error": result.somatic.dominant_error,
                "arousal_sensed": result.somatic.arousal_sensed,
                "energy_sensed": result.somatic.energy_sensed,
                "nearest_attractor": result.somatic.nearest_attractor,
                "trajectory_heading": result.somatic.trajectory_heading,
            }

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CYCLE_COMPLETED,
            data=cycle_data,
        ))

        # ── 8. Model hot-swap probation monitor ──
        # Runs every cycle during probation. The rollback check is O(1) -
        # it just compares an error counter ratio against a threshold.
        if self._hot_swap_manager is not None:
            try:
                await self._hot_swap_manager.on_cycle()
            except Exception as exc:
                self._logger.error(
                    "hot_swap_on_cycle_error",
                    error=str(exc),
                )

        # ── 9. Feed CognitiveStallSentinel ──
        # The stall sentinel detects "catatonic organism" - cycles that run
        # but accomplish nothing.  It lives in Thymos; we feed it here so it
        # gets a data point on every theta tick without needing a poll loop.
        if self._thymos is not None:
            sentinel = getattr(self._thymos, "_cognitive_stall_sentinel", None)
            if sentinel is not None:
                # Derive per-cycle activity deltas from live counters.
                nova_now = getattr(self._nova, "_total_intents_issued", self._nova_intents_prev)
                evo_now = getattr(self._evo, "_total_evidence_evaluations", self._evo_evidence_prev)
                nova_had_intent = nova_now > self._nova_intents_prev
                evo_had_evidence = evo_now > self._evo_evidence_prev
                self._nova_intents_prev = nova_now
                self._evo_evidence_prev = evo_now

                try:
                    stall_incidents = sentinel.record_cycle(
                        had_broadcast=result.had_broadcast,
                        nova_had_intent=nova_had_intent,
                        evo_had_evidence=evo_had_evidence,
                        atune_had_percept=result.had_broadcast,
                    )
                    if stall_incidents:
                        self._logger.warning(
                            "cognitive_stall_incidents_detected",
                            count=len(stall_incidents),
                            cycle=result.cycle_number,
                        )
                        for incident in stall_incidents:
                            asyncio.create_task(
                                self._thymos.on_incident(incident),
                                name=f"thymos_stall_{incident.id[:8]}",
                            )
                    elif self._cycle_count % 100 == 0:
                        self._logger.debug(
                            "cognitive_stall_sentinel_fed",
                            cycle=result.cycle_number,
                            had_broadcast=result.had_broadcast,
                            nova_had_intent=nova_had_intent,
                            evo_had_evidence=evo_had_evidence,
                        )
                except Exception as stall_exc:
                    self._logger.error(
                        "cognitive_stall_sentinel_error",
                        error=str(stall_exc),
                        cycle=result.cycle_number,
                    )

        # ── 10. Emit SomaTickEvent for stateless consumers ──
        if result.somatic is not None:
            # Inject cached starvation level so downstream systems read it per-tick
            result.somatic.starvation_level = self._cached_starvation_level
            soma_tick_data = SomaTickEvent(
                cycle_number=result.cycle_number,
                somatic_state=result.somatic,
            ).model_dump(mode="json")
            # EIS anomaly_detector expects a "drives" key with dict[str, float]
            # of interoceptive dimension values for drive-state baseline tracking.
            soma_tick_data["drives"] = dict(result.somatic.precision_weights)
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SOMA_TICK,
                data=soma_tick_data,
            ))

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an evolutionary observable event via the event bus."""
        bus = self._event_bus
        if bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from primitives.common import SystemID

            obs = EvolutionaryObservable(
                source_system=SystemID.SYNAPSE,
                instance_id="",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="synapse",
                data=obs.model_dump(mode="json"),
            )
            await bus.emit(event)
        except Exception:
            pass  # Evolutionary telemetry must never block the cycle

    async def _emit_re_training_example(
        self,
        *,
        category: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float = 0.5,
        reasoning_trace: str = "",
        alternatives_considered: list[str] | None = None,
        latency_ms: int = 0,
    ) -> None:
        """Fire-and-forget RE training example onto the event bus."""
        if self._event_bus is None:
            return
        try:
            from decimal import Decimal

            example = RETrainingExample(
                source_system=SystemID.SYNAPSE,
                category=category,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=max(0.0, min(1.0, outcome_quality)),
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives_considered or [],
                latency_ms=latency_ms,
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="synapse",
                data=example.model_dump(mode="json"),
            ))
        except Exception:
            pass  # Never block the cognitive cycle
