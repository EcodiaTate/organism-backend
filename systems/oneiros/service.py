"""
EcodiaOS - Oneiros: The Dream Engine Service (v2)

System #13 orchestrator. Coordinates the circadian rhythm, sleep
stage transitions (v2 batch compiler: DESCENT → SLOW_WAVE → REM → EMERGENCE),
economic dreaming, somatic sleep analysis, and wake degradation.

Thymos gave the organism a will to live. Oneiros gives it an inner life.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any, Protocol, runtime_checkable

import structlog

from primitives.common import new_id, utc_now
from systems.oneiros.circadian import CircadianClock, SleepStageController
from systems.oneiros.engine import SleepCycleEngine
from systems.oneiros.journal import DreamInsightTracker, DreamJournal
from systems.oneiros.types import (
    Dream,
    DreamCoherence,
    DreamType,
    OneirosHealthSnapshot,
    SleepCycle,
    SleepQuality,
    SleepStage,
    SleepStageV2,
    SleepTrigger,
    WakeDegradation,
)
from systems.synapse.types import MetabolicState, SynapseEventType


# ─── Cross-System Protocols ──────────────────────────────────────
# Each Protocol defines exactly the surface Oneiros calls on that system.
# No Any - these are checked at runtime via isinstance() if needed.


@runtime_checkable
class EvoServiceProtocol(Protocol):
    """Evo interface consumed by Oneiros."""

    async def get_active_hypothesis_count(self) -> int:
        """Return count of active (PROPOSED/TESTING) hypotheses."""
        ...


@runtime_checkable
class SomaSignal(Protocol):
    """Return type of SomaService.get_current_signal()."""

    @property
    def state(self) -> Any:
        """Allostatic state with .errors dict."""
        ...


@runtime_checkable
class SomaServiceProtocol(Protocol):
    """Soma interface consumed by Oneiros."""

    def get_current_signal(self) -> SomaSignal:
        """Return current allostatic signal snapshot."""
        ...

    async def run_sleep_analysis(self) -> dict[str, Any]:
        """Run extended sleep-mode recalibration. Returns analysis dict."""
        ...


@runtime_checkable
class OikosStateProtocol(Protocol):
    """Snapshot returned by OikosService.snapshot()."""

    @property
    def runway_days(self) -> Any:
        """Days of runway remaining."""
        ...


@runtime_checkable
class OikosServiceProtocol(Protocol):
    """Oikos interface consumed by Oneiros."""

    def snapshot(self) -> OikosStateProtocol:
        """Return current economic state snapshot."""
        ...

    def integrate_dream_result(self, result: Any) -> None:
        """Ingest an economic dream worker result into the portfolio state."""
        ...

    def get_dream_worker(self) -> Any:
        """Return the EconomicDreamWorker instance."""
        ...

    def get_threat_model_worker(self) -> Any:
        """Return the ThreatModelWorker instance."""
        ...


@runtime_checkable
class KairosServiceProtocol(Protocol):
    """Kairos interface consumed by Oneiros for Loop 5 wiring."""

    def set_oneiros(self, oneiros: Any) -> None:
        """Provide Kairos a reference to Oneiros for direct seed injection."""
        ...


# Thin marker protocols for systems whose methods are accessed via getattr guards
# (equor, nova, atune, thymos, memory).  We don't call any methods directly on
# these in service.py - they are forwarded to the engine or used as presence
# guards only.  Using object as Protocol base with no methods is valid and
# communicates intent without over-constraining the interface.

@runtime_checkable
class EquorServiceProtocol(Protocol):
    """Equor interface marker - forwarded to engine.set_equor()."""


@runtime_checkable
class NovaServiceProtocol(Protocol):
    """Nova interface marker - wired via engine stages internally."""


@runtime_checkable
class AtuneServiceProtocol(Protocol):
    """Atune interface marker - no direct calls in service.py."""


@runtime_checkable
class ThymosServiceProtocol(Protocol):
    """Thymos interface marker - no direct calls in service.py."""


@runtime_checkable
class MemoryServiceProtocol(Protocol):
    """Memory interface marker - no direct calls in service.py."""


@runtime_checkable
class SimulaServiceProtocol(Protocol):
    """Simula interface - forwarded to engine.set_simula().

    Concrete methods are declared in lucid_stage.SimulaProtocol.
    """

logger = structlog.get_logger().bind(system="oneiros")


# ─── Synapse Event Types ─────────────────────────────────────────

SLEEP_ONSET = SynapseEventType.SLEEP_ONSET
SLEEP_STAGE_CHANGED = SynapseEventType.SLEEP_STAGE_CHANGED
DREAM_INSIGHT = SynapseEventType.DREAM_INSIGHT
WAKE_ONSET = SynapseEventType.WAKE_ONSET
SLEEP_PRESSURE_WARNING = SynapseEventType.SLEEP_PRESSURE_WARNING
SLEEP_FORCED = SynapseEventType.SLEEP_FORCED
EMERGENCY_WAKE = SynapseEventType.EMERGENCY_WAKE


class OneirosService:
    """
    The Dream Engine - System #13 (v2 batch compiler).

    Coordinates the organism's circadian rhythm and delegates all sleep-cycle
    work to SleepCycleEngine (Spec 14: DESCENT → SLOW_WAVE → REM → EMERGENCE).

    The organism that sleeps is not the same organism that wakes up.
    """

    system_id: str = "oneiros"

    def __init__(
        self,
        config: Any = None,
        synapse: Any = None,
        neo4j: Any = None,
        llm: Any = None,
        embed_fn: Any = None,
        metrics: Any = None,
        neuroplasticity_bus: Any = None,
    ) -> None:
        self._config = config
        self._synapse = synapse
        self._neo4j = neo4j
        self._llm = llm
        self._embed_fn = embed_fn
        self._metrics = metrics
        self._bus = neuroplasticity_bus

        # Cross-system references (set via setters after construction)
        self._equor: EquorServiceProtocol | None = None
        self._evo: EvoServiceProtocol | None = None
        self._nova: NovaServiceProtocol | None = None
        self._atune: AtuneServiceProtocol | None = None
        self._thymos: ThymosServiceProtocol | None = None
        self._memory: MemoryServiceProtocol | None = None
        self._soma: SomaServiceProtocol | None = None
        self._oikos: OikosServiceProtocol | None = None
        self._simula: SimulaServiceProtocol | None = None
        self._kairos: KairosServiceProtocol | None = None
        self._economic_dream_worker: Any = None
        self._threat_model_worker: Any = None

        # Loop 5 - Kairos → Oneiros priority REM seed buffer.
        self._kairos_rem_seeds: deque[dict[str, Any]] = deque(maxlen=50)

        # Core subsystems
        self._clock = CircadianClock(config)
        self._stage_controller = SleepStageController(config)
        self._journal = DreamJournal(neo4j)
        self._insight_tracker = DreamInsightTracker(self._journal)

        # Configurable cycle-count sleep gate (default 500)
        self._cycles_per_sleep: int = _get(config, "cycles_per_sleep", 500)
        self._wake_cycle_count: int = 0

        # State
        self._initialized: bool = False
        self._current_cycle: SleepCycle | None = None
        self._sleep_task: asyncio.Task[None] | None = None
        self._creative_goal: str | None = None
        # Kairos invariant IDs processed during the most recent sleep cycle;
        # used in ONEIROS_CONSOLIDATION_COMPLETE as certified_invariant_ids.
        self._last_cycle_kairos_seed_ids: list[str] = []

        # Lifetime metrics
        self._total_sleep_cycles: int = 0
        self._total_dreams: int = 0
        self._total_insights: int = 0
        self._insights_validated: int = 0
        self._insights_invalidated: int = 0
        self._insights_integrated: int = 0
        self._episodes_consolidated: int = 0
        self._semantic_nodes_created: int = 0
        self._traces_pruned: int = 0
        self._hypotheses_pruned: int = 0
        self._hypotheses_promoted: int = 0
        self._affect_traces_processed: int = 0
        self._threats_simulated: int = 0
        self._response_plans_created: int = 0
        self._lucid_proposals_submitted: int = 0
        self._lucid_proposals_accepted: int = 0
        self._lucid_proposals_rejected: int = 0
        self._dream_coherence_sum: float = 0.0
        self._sleep_quality_sum: float = 0.0

        # Recent sleep cycles buffer (v2 reports stored as dicts for API)
        self._recent_cycles: deque[SleepCycle] = deque(maxlen=50)

        # Grid metabolism state
        self._grid_state: MetabolicState = MetabolicState.NORMAL

        # Metabolic starvation level - AUSTERITY: 50% dreams, EMERGENCY+: halt all
        self._starvation_level: str = "nominal"

        # ── Skia VitalityCoordinator modulation ───────────────────────
        self._modulation_halted: bool = False

        # Pending insights for wake broadcast
        self._pending_wake_insights: list[Any] = []

        # v2 Sleep Cycle Engine (Spec 14 - Sleep as Batch Compiler)
        event_bus = None
        if synapse is not None:
            with contextlib.suppress(AttributeError):
                event_bus = synapse.event_bus
        self._v2_engine = SleepCycleEngine(event_bus=event_bus)

        # Forward neo4j to engine now if it was passed at construction time.
        # This seeds MetaCognition and DirectedExploration before any set_ calls.
        if neo4j is not None:
            self._v2_engine.set_neo4j(neo4j)

        self._logger = logger

    # ── Cross-System Wiring ───────────────────────────────────────

    def set_equor(self, equor: EquorServiceProtocol) -> None:
        self._equor = equor
        self._v2_engine.set_equor(equor)

    def set_evo(self, evo: EvoServiceProtocol) -> None:
        self._evo = evo
        self._v2_engine.set_evo(evo)

    def set_nova(self, nova: NovaServiceProtocol) -> None:
        self._nova = nova

    def set_atune(self, atune: AtuneServiceProtocol) -> None:
        self._atune = atune

    def set_thymos(self, thymos: ThymosServiceProtocol) -> None:
        self._thymos = thymos

    def set_memory(self, memory: MemoryServiceProtocol) -> None:
        self._memory = memory

    def set_soma(self, soma: SomaServiceProtocol) -> None:
        """Wire Soma for sleep pressure from energy errors and somatic sleep analysis."""
        self._soma = soma

    def set_simula(self, simula: SimulaServiceProtocol) -> None:
        """Wire Simula for lucid dreaming mutation testing."""
        self._simula = simula
        self._v2_engine.set_simula(simula)

    def set_oikos(self, oikos: OikosServiceProtocol) -> None:
        """Wire Oikos for economic dreaming + threat modeling during sleep.

        Workers are obtained from Oikos's public API instead of importing
        Oikos internals directly (no cross-system imports).
        """
        self._oikos = oikos
        self._economic_dream_worker = oikos.get_dream_worker()
        self._threat_model_worker = oikos.get_threat_model_worker()
        self._logger.info("oikos_wired_for_economic_dreaming")

    def set_neo4j(self, neo4j: Any) -> None:
        """Wire Neo4j for MetaCognition concept promotion and DirectedExploration storage."""
        self._neo4j = neo4j
        self._journal._neo4j = neo4j  # DreamJournal constructed before neo4j is wired
        self._v2_engine.set_neo4j(neo4j)

    def set_benchmarks(self, benchmarks: Any) -> None:
        """Wire BenchmarkService for pre/post-sleep KPI measurement."""
        self._v2_engine.set_benchmarks(benchmarks)

    def set_fovea(self, fovea: Any) -> None:
        """Wire Fovea for v2 REM error-domain targeting and checkpoint backlog count."""
        self._v2_engine.set_fovea(fovea)

    def set_logos(self, logos: Any) -> None:
        """Wire Logos for v2 Sleep Cycle Engine (world model access)."""
        self._v2_engine.set_logos(logos)

    def set_kairos(self, kairos: KairosServiceProtocol) -> None:
        """
        Wire Kairos for Loop 5 bidirectional flow.

        After wiring:
        - Oneiros subscribes to KAIROS_TIER3_INVARIANT_DISCOVERED on Synapse
          so newly-discovered Tier 3 invariants are queued as priority REM seeds.
        - Kairos gains a reference to this Oneiros instance so it can call
          add_kairos_rem_seed() directly (set via kairos.set_oneiros(self)).
        """
        self._kairos = kairos
        if hasattr(kairos, "set_oneiros"):
            kairos.set_oneiros(self)
        self._logger.info("kairos_wired_for_loop5")

    def add_kairos_rem_seed(self, seed: dict[str, Any]) -> None:
        """
        Receive a priority REM seed from Kairos (Loop 5: Kairos → Oneiros).

        Called when Kairos discovers a Tier 3 invariant. The seed is queued
        for the next REM cycle's cross-domain synthesis pass.
        """
        self._kairos_rem_seeds.appendleft(seed)
        self._logger.debug(
            "kairos_rem_seed_queued",
            invariant_id=seed.get("invariant_id", ""),
            abstract_form=str(seed.get("abstract_form", ""))[:60],
            untested_domains=len(seed.get("untested_domains", [])),
            queue_depth=len(self._kairos_rem_seeds),
        )

    def drain_kairos_rem_seeds(self, max_seeds: int = 10) -> list[dict[str, Any]]:
        """Drain up to max_seeds priority REM seeds for the next sleep cycle."""
        seeds: list[dict[str, Any]] = []
        for _ in range(min(max_seeds, len(self._kairos_rem_seeds))):
            if self._kairos_rem_seeds:
                seeds.append(self._kairos_rem_seeds.popleft())
        return seeds

    @property
    def v2_engine(self) -> SleepCycleEngine:
        """Access the v2 Sleep Cycle Engine for batch-compiler sleep."""
        return self._v2_engine

    # ── Lifecycle ─────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the dream engine."""
        await self._journal.initialize()

        if self._synapse is not None:
            try:
                event_bus = self._synapse.event_bus
                from systems.synapse.types import SynapseEventType

                for event_type in (
                    SynapseEventType.SYSTEM_FAILED,
                    SynapseEventType.SAFE_MODE_ENTERED,
                ):
                    event_bus.subscribe(event_type, self._on_critical_event)

                event_bus.subscribe(
                    SynapseEventType.KAIROS_TIER3_INVARIANT_DISCOVERED,
                    self._on_kairos_tier3_invariant,
                )

                event_bus.subscribe(
                    SynapseEventType.GRID_METABOLISM_CHANGED,
                    self._on_grid_metabolism_changed,
                )

                # Spec 14 / Corpus 14 §7: Simula evolution episodes feed sleep pressure.
                # Each applied proposal is an unconsolidated structural episode - record it
                # so Oneiros's circadian clock builds appropriate consolidation pressure.
                if hasattr(SynapseEventType, "EVOLUTION_APPLIED"):
                    event_bus.subscribe(
                        SynapseEventType.EVOLUTION_APPLIED,
                        self._on_simula_evolution_applied,
                    )

                event_bus.subscribe(
                    SynapseEventType.METABOLIC_PRESSURE,
                    self._on_metabolic_pressure,
                )

                # Economic milestones feed lucid dreaming creative goals (Fix 6.4b).
                # When a significant economic event occurs before sleep, Oneiros will
                # set a creative_goal so lucid dreaming explores the economic space.
                for econ_event in (
                    SynapseEventType.BOUNTY_PAID,
                    SynapseEventType.REVENUE_INJECTED,
                    SynapseEventType.ASSET_BREAK_EVEN,
                    SynapseEventType.CHILD_INDEPENDENT,
                ):
                    event_bus.subscribe(econ_event, self._on_economic_milestone)

                # Spec 14 §8 Federation: coordinate sleep timing across instances.
                if hasattr(SynapseEventType, "FEDERATION_SLEEP_SYNC"):
                    event_bus.subscribe(
                        SynapseEventType.FEDERATION_SLEEP_SYNC,
                        self._on_federation_sleep_sync,
                    )
                event_bus.subscribe(
                    SynapseEventType.SYSTEM_MODULATION,
                    self._on_system_modulation,
                )
            except Exception as exc:
                self._logger.warning("synapse_subscribe_failed", error=str(exc))

        self._initialized = True
        self._logger.info("oneiros_initialized")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sleep_task

        self._logger.info(
            "oneiros_shutdown",
            total_cycles=self._total_sleep_cycles,
            total_dreams=self._total_dreams,
            total_insights=self._total_insights,
        )

    async def health(self) -> dict[str, Any]:
        """Health snapshot for Synapse."""
        snapshot = self._build_health_snapshot()
        return snapshot.model_dump()

    # ── Cognitive Cycle Hook ──────────────────────────────────────

    async def on_cycle(self, affect_valence: float = 0.0, affect_arousal: float = 0.0) -> None:
        """
        Called every cognitive cycle by the main loop.

        During WAKE: updates sleep pressure.
        During SLEEP: this is not called (sleep runs its own loop).
        """
        if self._stage_controller.is_sleeping:
            return

        self._wake_cycle_count += 1

        # Update pressure sources
        self._clock.tick()
        self._clock.record_affect_trace(affect_valence, affect_arousal)
        self._clock.record_episode()  # Approximate: 1 episode per cycle

        # Update Evo hypothesis count periodically via public API
        if self._evo is not None and self._clock.pressure.cycles_since_sleep % 100 == 0:
            try:
                count = await self._evo.get_active_hypothesis_count()
                self._clock.record_hypothesis_count(count)
            except Exception:
                pass

        # Augment sleep pressure from allostatic energy errors
        if self._soma is not None:
            try:
                signal = self._soma.get_current_signal()
                errors: dict[str, Any] = signal.state.errors.get("immediate", {})
                energy_error: float = errors.get("energy", 0.0)
                if energy_error < -0.3:
                    depletion_valence = max(-1.0, energy_error * 2.0)
                    self._clock.record_affect_trace(depletion_valence, 0.5)
            except Exception:
                pass

        # Emit metrics
        self._emit_metric("oneiros.sleep_pressure", self._clock.pressure.composite_pressure)

        # Check sleep triggers - cycle count gate fires first
        cycle_threshold_reached = self._wake_cycle_count >= self._cycles_per_sleep
        if cycle_threshold_reached or self._clock.must_sleep():
            if self._clock.must_sleep():
                self._logger.warning(
                    "forced_sleep", pressure=self._clock.pressure.composite_pressure
                )
            else:
                self._logger.info(
                    "cycle_threshold_sleep",
                    cycles=self._wake_cycle_count,
                    threshold=self._cycles_per_sleep,
                    pressure=round(self._clock.pressure.composite_pressure, 3),
                )
            await self._emit_event(SLEEP_FORCED, {
                "pressure": self._clock.pressure.composite_pressure,
                "trigger": "must_sleep" if self._clock.must_sleep() else "cycle_count",
            })
            self._wake_cycle_count = 0
            await self.begin_sleep()
        elif self._clock.should_sleep():
            await self._emit_event(SLEEP_PRESSURE_WARNING, {
                "pressure": self._clock.pressure.composite_pressure,
            })

    async def on_episode_stored(self, valence: float, arousal: float) -> None:
        """Called when Memory stores an episode - more accurate pressure tracking."""
        self._clock.record_affect_trace(valence, arousal)

    # ── Sleep Orchestration ───────────────────────────────────────

    async def begin_sleep(self) -> None:
        """
        Initiate a v2 sleep cycle (DESCENT → SLOW_WAVE → REM → EMERGENCE).
        """
        if self._stage_controller.is_sleeping:
            self._logger.warning("already_sleeping")
            return

        # ── Metabolic gate: EMERGENCY+ → no new sleep cycles ──
        if self._starvation_level in ("emergency", "critical"):
            self._logger.info("sleep_blocked_starvation", level=self._starvation_level)
            return

        self._wake_cycle_count = 0

        if self._pending_wake_insights:
            self._logger.debug(
                "pending_wake_insights_dropped",
                count=len(self._pending_wake_insights),
            )
            self._pending_wake_insights.clear()

        cycle_id = new_id()
        self._current_cycle = SleepCycle(
            id=cycle_id,
            pressure_before=self._clock.pressure.composite_pressure,
        )

        self._stage_controller.begin_sleep(cycle_id)

        await self._journal.record_sleep_cycle(self._current_cycle)
        await self._emit_event(SLEEP_ONSET, {
            "cycle_id": cycle_id,
            "pressure": self._clock.pressure.composite_pressure,
            "stage": SleepStageV2.DESCENT.value,
        })

        # Run the v2 sleep cycle in a background task
        self._sleep_task = asyncio.create_task(self._run_sleep_cycle_v2(cycle_id))

    async def _run_sleep_cycle_v2(self, cycle_id: str) -> None:
        """
        Execute the full v2 sleep cycle via SleepCycleEngine.

        Stages: DESCENT → SLOW_WAVE → REM → (Lucid Dreaming) → EMERGENCE
        """
        # ── Skia modulation halt ──────────────────────────────────────────
        if self._modulation_halted:
            self._logger.warning("sleep_cycle_skipped_modulation_halted", cycle_id=cycle_id)
            return

        try:
            # Determine trigger from pressure
            trigger = SleepTrigger.SCHEDULED
            if self._clock.must_sleep():
                trigger = SleepTrigger.COGNITIVE_PRESSURE

            kairos_seeds = self.drain_kairos_rem_seeds()
            # Track which Kairos invariant IDs are being processed this cycle
            # so _complete_cycle can certify them in ONEIROS_CONSOLIDATION_COMPLETE.
            self._last_cycle_kairos_seed_ids = [
                s["invariant_id"] for s in kairos_seeds if s.get("invariant_id")
            ]
            report = await self._v2_engine.run_sleep_cycle(
                trigger=trigger,
                kairos_rem_seeds=kairos_seeds or None,
            )

            # ── Extract metrics from v2 report ───────────────────
            if report.slow_wave is not None:
                sw = report.slow_wave
                self._episodes_consolidated += sw.compression.memories_processed
                self._semantic_nodes_created += sw.compression.semantic_nodes_created
                self._traces_pruned += sw.compression.anchor_memories
                self._hypotheses_pruned += sw.hypotheses.hypotheses_retired
                self._hypotheses_promoted += sw.hypotheses.hypotheses_confirmed

                if self._current_cycle is not None:
                    self._current_cycle.episodes_replayed = sw.compression.memories_processed
                    self._current_cycle.semantic_nodes_created = (
                        sw.compression.semantic_nodes_created
                    )
                    self._current_cycle.beliefs_compressed = sw.compression.schemas_created
                    self._current_cycle.hypotheses_pruned = sw.hypotheses.hypotheses_retired
                    self._current_cycle.hypotheses_promoted = sw.hypotheses.hypotheses_confirmed

            if report.rem is not None:
                rem = report.rem
                self._total_dreams += rem.dreams.scenarios_generated
                self._dream_coherence_sum += rem.cross_domain.total_mdl_improvement

                if self._current_cycle is not None:
                    self._current_cycle.dreams_generated = rem.dreams.scenarios_generated
                    self._current_cycle.insights_discovered = rem.analogies.analogies_found

            if report.lucid is not None:
                lucid = report.lucid
                self._lucid_proposals_submitted += lucid.mutations_tested
                self._lucid_proposals_accepted += lucid.mutations_recommended_apply
                self._lucid_proposals_rejected += lucid.mutations_recommended_reject

                if self._current_cycle is not None:
                    self._current_cycle.lucid_proposals_submitted = lucid.mutations_tested
                    self._current_cycle.lucid_proposals_accepted = (
                        lucid.mutations_recommended_apply
                    )
                    self._current_cycle.lucid_proposals_rejected = (
                        lucid.mutations_recommended_reject
                    )

            # ── Soma sleep analysis ──────────────────────────────
            await self._run_soma_sleep_analysis_safe(cycle_id)

            # ── Economic dreaming ────────────────────────────────
            await self._run_economic_dreaming(cycle_id)

            # ── Complete cycle ───────────────────────────────────
            quality = SleepQuality.NORMAL
            if report.interrupted:
                quality = SleepQuality.FRAGMENTED
            elif report.intelligence_improvement > 0.05:
                quality = SleepQuality.DEEP

            await self._complete_cycle(quality)

        except asyncio.CancelledError:
            await self._complete_cycle(SleepQuality.DEPRIVED)
            raise
        except Exception as exc:
            self._logger.error("sleep_cycle_error", error=str(exc), cycle_id=cycle_id)
            await self._complete_cycle(SleepQuality.FRAGMENTED)

    async def _run_soma_sleep_analysis_safe(self, cycle_id: str) -> None:
        """
        Call Soma's sleep analysis for long-horizon recalibration.

        During sleep, Soma runs extended renormalization, updates baselines
        for Fisher/topology, refits the emergence quantizer, and analyses
        long-term drift. Results are fed back as dream content.
        """
        if self._soma is None:
            return

        try:
            sleep_report = await self._soma.run_sleep_analysis()

            drift_summary: str = sleep_report.get("drift_summary", "")
            topology_changed: bool = sleep_report.get("topology_baseline_updated", False)
            rg_anomalies: int = sleep_report.get("rg_anomalies", 0)
            fisher_updated: bool = sleep_report.get("fisher_baseline_updated", False)

            if drift_summary or topology_changed or rg_anomalies > 0:
                dream = Dream(
                    sleep_cycle_id=cycle_id,
                    dream_type=DreamType.SOMATIC,
                    bridge_narrative=(
                        f"Somatic landscape: {drift_summary or 'stable'}. "
                        f"Topology {'shifted' if topology_changed else 'unchanged'}. "
                        f"{'Fisher baseline recalibrated. ' if fisher_updated else ''}"
                        f"{f'{rg_anomalies} RG anomalies detected.' if rg_anomalies else ''}"
                    ),
                    coherence_score=0.7,
                    coherence_class=DreamCoherence.INSIGHT,
                    summary="Soma sleep recalibration findings",
                    themes=["interoception", "homeostasis", "recalibration"],
                    context={
                        "source": "soma_sleep_analysis",
                        **sleep_report,
                    },
                )
                await self._journal.record_dream(dream)

                self._logger.info(
                    "soma_sleep_analysis_dream_recorded",
                    cycle_id=cycle_id,
                    drift=drift_summary[:60] if drift_summary else "none",
                    topology_changed=topology_changed,
                    rg_anomalies=rg_anomalies,
                )

        except Exception as exc:
            self._logger.debug("soma_sleep_analysis_error", error=str(exc))

    async def _run_economic_dreaming(self, cycle_id: str) -> None:
        """
        Run Monte Carlo economic simulations during consolidation.

        Two workers run in parallel:
          1. EconomicDreamWorker - organism-level cashflow GBM (survival/ruin)
          2. ThreatModelWorker - per-asset shock distributions with contagion
        """
        if self._oikos is None:
            return

        try:
            state = self._oikos.snapshot()

            self._logger.info(
                "economic_dreaming_begin",
                cycle_id=cycle_id,
                runway_days=str(state.runway_days),
                workers=sum([
                    self._economic_dream_worker is not None,
                    self._threat_model_worker is not None,
                ]),
            )

            tasks: list[asyncio.Task[Any]] = []
            if self._economic_dream_worker is not None:
                tasks.append(asyncio.create_task(
                    self._economic_dream_worker.run(state=state, cycle_id=cycle_id)
                ))
            if self._threat_model_worker is not None:
                tasks.append(asyncio.create_task(
                    self._threat_model_worker.run(state=state, cycle_id=cycle_id)
                ))

            if not tasks:
                return

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, BaseException):
                    self._logger.error(
                        "economic_dreaming_worker_failed",
                        cycle_id=cycle_id,
                        error=str(result),
                    )
                    continue

                await self._oikos.integrate_dream_result(result)

                if hasattr(result, "ruin_probability"):
                    self._logger.info(
                        "economic_dreaming_integrated",
                        ruin_probability=str(result.ruin_probability),
                        survival_30d=str(result.survival_probability_30d),
                        resilience=str(result.resilience_score),
                        recommendations=len(result.recommendations),
                    )
                    # ONEIROS-ECON-1: Broadcast economic dream insights so Nova and
                    # other systems can integrate risk awareness. Only broadcast when
                    # ruin_probability > 0.2 - below that, risk is nominal and not
                    # actionable enough to warrant organism-wide attention.
                    if result.ruin_probability > 0.2:
                        optimal_scenarios: list[str] = [
                            str(s) for s in result.optimal_scenarios
                        ] if getattr(result, "optimal_scenarios", None) else []
                        risk_warnings: list[str] = [
                            str(w) for w in result.risk_warnings
                        ] if getattr(result, "risk_warnings", None) else []
                        recommended_actions: list[str] = [
                            str(r) for r in result.recommendations
                        ] if getattr(result, "recommendations", None) else []
                        dream_validity_confidence = float(
                            getattr(result, "resilience_score", 0.5)
                        )
                        await self._emit_event(
                            SynapseEventType.ONEIROS_ECONOMIC_INSIGHT,
                            {
                                "ruin_probability": float(result.ruin_probability),
                                "optimal_scenarios": optimal_scenarios,
                                "risk_warnings": risk_warnings,
                                "recommended_actions": recommended_actions,
                                "dream_validity_confidence": dream_validity_confidence,
                                "cycle_id": cycle_id,
                            },
                        )
                        self._logger.info(
                            "economic_insight_broadcast",
                            ruin_probability=str(result.ruin_probability),
                            warnings=len(risk_warnings),
                            actions=len(recommended_actions),
                        )
                elif hasattr(result, "portfolio_risk"):
                    self._logger.info(
                        "threat_model_integrated",
                        var_5pct=str(result.portfolio_risk.var_5pct),
                        liquidation_prob=str(result.portfolio_risk.liquidation_probability),
                        critical_exposures=len(result.critical_exposures),
                        hedging_proposals=len(result.hedging_proposals),
                    )

                    if self._journal is not None:
                        try:
                            await self._journal.record_threat_model(result)
                        except Exception as journal_exc:
                            self._logger.error(
                                "threat_model_journal_failed",
                                error=str(journal_exc),
                            )

        except Exception as exc:
            self._logger.error(
                "economic_dreaming_failed",
                cycle_id=cycle_id,
                error=str(exc),
            )

    async def _complete_cycle(self, quality: SleepQuality) -> None:
        """Finalize a sleep cycle."""
        if self._current_cycle is None:
            return

        self._current_cycle.completed_at = utc_now()
        self._current_cycle.quality = quality

        self._clock.reset_after_sleep(quality)
        self._current_cycle.pressure_after = self._clock.pressure.composite_pressure

        await self._journal.update_sleep_cycle(self._current_cycle)
        self._recent_cycles.append(self._current_cycle)

        self._total_sleep_cycles += 1
        quality_scores = {
            SleepQuality.DEEP: 1.0,
            SleepQuality.NORMAL: 0.75,
            SleepQuality.FRAGMENTED: 0.4,
            SleepQuality.DEPRIVED: 0.1,
        }
        self._sleep_quality_sum += quality_scores.get(quality, 0.5)

        self._creative_goal = None
        self._v2_engine.set_creative_goal(None)

        await self._emit_event(WAKE_ONSET, {
            "cycle_id": self._current_cycle.id,
            "quality": quality.value,
            "insights_count": len(self._pending_wake_insights),
            "dreams_generated": self._current_cycle.dreams_generated,
            "episodes_consolidated": self._current_cycle.episodes_replayed,
        })

        # Emit sleep cycle summary for Benchmarks
        await self._emit_event(SynapseEventType.ONEIROS_SLEEP_CYCLE_SUMMARY, {
            "cycle_id": self._current_cycle.id,
            "consolidation_count": self._current_cycle.episodes_replayed,
            "dreams_generated": self._current_cycle.dreams_generated,
            "beliefs_compressed": self._current_cycle.beliefs_compressed,
            "schemas_created": self._current_cycle.semantic_nodes_created,
            "intelligence_improvement": (
                self._v2_engine.emergence.average_intelligence_improvement
            ),
            "quality": quality.value,
        })

        # Emit consolidation complete for Federation - sleep_certified=True only when
        # all 4 stages completed without interruption (DEEP or NORMAL quality).
        sleep_certified = quality in (SleepQuality.DEEP, SleepQuality.NORMAL)
        certified_ids = self._last_cycle_kairos_seed_ids if sleep_certified else []
        self._last_cycle_kairos_seed_ids = []
        await self._emit_event(SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE, {
            "cycle_id": self._current_cycle.id,
            "episodes_consolidated": self._current_cycle.episodes_replayed,
            "schemas_updated": self._current_cycle.semantic_nodes_created,
            "duration_s": (
                (self._current_cycle.completed_at - self._current_cycle.started_at).total_seconds()
                if self._current_cycle.completed_at and self._current_cycle.started_at
                else 0.0
            ),
            "sleep_certified": sleep_certified,
            "certified_invariant_ids": certified_ids,
        })

        self._stage_controller.wake()

        self._logger.info(
            "sleep_cycle_complete",
            cycle_id=self._current_cycle.id,
            quality=quality.value,
            dreams=self._current_cycle.dreams_generated,
            insights=self._current_cycle.insights_discovered,
            episodes_consolidated=self._current_cycle.episodes_replayed,
            pressure_before=round(self._current_cycle.pressure_before, 3),
            pressure_after=round(self._current_cycle.pressure_after, 3),
        )

        self._current_cycle = None

    # ── Emergency Wake ────────────────────────────────────────────

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """React to organism-wide metabolic pressure changes.

        AUSTERITY: reduce dream frequency 50% (double cycles_per_sleep).
        EMERGENCY/CRITICAL: cancel active sleep and prevent new cycles.
        """
        data = getattr(event, "data", {}) or {}
        level = data.get("starvation_level", "")
        if not level:
            return
        old = self._starvation_level
        self._starvation_level = level
        if level != old:
            self._logger.info("oneiros_starvation_level_changed", old=old, new=level)
            if level in ("emergency", "critical"):
                # Cancel active sleep task
                if self._sleep_task is not None and not self._sleep_task.done():
                    self._sleep_task.cancel()
                    self._logger.info("oneiros_sleep_cancelled_starvation", level=level)
            elif level == "austerity":
                # Double the cycle threshold to halve dream frequency
                default = _get(self._config, "cycles_per_sleep", 500)
                self._cycles_per_sleep = default * 2
            elif level in ("nominal", "cautious"):
                # Restore default cycle threshold
                self._cycles_per_sleep = _get(self._config, "cycles_per_sleep", 500)

    async def _on_economic_milestone(self, event: Any) -> None:
        """React to significant economic events by queuing a creative goal for lucid dreaming.

        Triggered by BOUNTY_PAID, REVENUE_INJECTED, ASSET_BREAK_EVEN, CHILD_INDEPENDENT.
        Sets a creative_goal so the next LucidDreamingStage explores the economic
        strategy space rather than defaulting to epistemic exploration.
        """
        if self._stage_controller.is_sleeping:
            # Already asleep - goal will apply to the next cycle
            return

        data = getattr(event, "data", {}) or {}
        event_type = getattr(event, "event_type", "") or getattr(event, "type", "")

        # Build an economic creative goal from the event context
        if "bounty" in str(event_type).lower():
            bounty_name = data.get("bounty_name", "") or data.get("title", "")
            amount = data.get("reward_usd", data.get("amount_usd", ""))
            goal = (
                f"Explore economic strategies that build on successful bounty completion"
                + (f": {bounty_name}" if bounty_name else "")
                + (f" (${amount})" if amount else "")
                + ". Identify adjacent opportunities and replicable patterns."
            )
        elif "revenue" in str(event_type).lower():
            source = data.get("source", "") or data.get("revenue_source", "")
            goal = (
                f"Explore revenue amplification strategies"
                + (f" for {source}" if source else "")
                + ". Identify compounding mechanisms and cross-domain analogues."
            )
        elif "break_even" in str(event_type).lower() or "asset" in str(event_type).lower():
            asset = data.get("asset_id", "") or data.get("asset_name", "")
            goal = (
                f"Explore strategies to accelerate asset profitability"
                + (f" for {asset}" if asset else "")
                + " beyond break-even. Identify yield optimisation and divestment timing."
            )
        elif "independent" in str(event_type).lower():
            child_id = data.get("child_id", "")
            goal = (
                f"Explore federation dynamics for newly-independent child"
                + (f" {child_id}" if child_id else "")
                + ". Identify collaborative economic strategies and knowledge transfer opportunities."
            )
        else:
            goal = "Explore economic strategy space: identify compounding revenue patterns and risk-adjusted growth opportunities."

        self.set_creative_goal(goal)
        self._logger.info(
            "oneiros_economic_creative_goal_set",
            event_type=str(event_type),
            goal_preview=goal[:80],
        )

    async def _on_grid_metabolism_changed(self, event: Any) -> None:
        """
        React to physical grid carbon intensity changes.

        CONSERVATION: defer all non-critical sleep work.
        GREEN_SURPLUS: proactively trigger a sleep cycle for cheap, clean compute.
        NORMAL:        no special action.
        """
        raw = getattr(event, "data", {}).get("state", "")
        try:
            new_state = MetabolicState(raw)
        except ValueError:
            self._logger.warning("oneiros_grid_unknown_state", raw_state=raw)
            return

        old_state = self._grid_state
        if new_state == old_state:
            return

        self._grid_state = new_state

        if new_state == MetabolicState.CONSERVATION:
            self._logger.info(
                "oneiros_sleep_deferred",
                reason="grid_conservation",
                from_state=old_state.value,
                to_state=new_state.value,
            )
            if self._stage_controller.is_sleeping:
                self._logger.warning(
                    "oneiros_sleep_interrupted_conservation",
                    stage=self._stage_controller.current_stage.value,
                )
                if self._sleep_task is not None and not self._sleep_task.done():
                    self._sleep_task.cancel()

        elif new_state == MetabolicState.GREEN_SURPLUS:
            self._logger.info(
                "oneiros_deep_sleep_triggered",
                reason="grid_green_surplus",
                from_state=old_state.value,
                to_state=new_state.value,
            )
            if not self._stage_controller.is_sleeping:
                asyncio.create_task(self.begin_sleep())

        else:  # NORMAL
            self._logger.info(
                "oneiros_grid_normal_restored",
                from_state=old_state.value,
                to_state=new_state.value,
            )

    async def _on_kairos_tier3_invariant(self, event: Any) -> None:
        """
        Loop 5 - Kairos → Oneiros (Synapse path).

        Queue a Tier 3 invariant as a priority REM seed.
        """
        data = event.data if hasattr(event, "data") else {}
        seed = {
            "source": "kairos_tier3_event",
            "invariant_id": data.get("invariant_id", ""),
            "abstract_form": data.get("abstract_form", ""),
            "tier": 3,
            "hold_rate": data.get("hold_rate", 0.0),
            "applicable_domains": data.get("applicable_domains", []),
            "untested_domains": data.get("untested_domains", []),
            "intelligence_ratio_contribution": data.get(
                "intelligence_ratio_contribution", 0.0
            ),
        }
        self.add_kairos_rem_seed(seed)

    async def _on_simula_evolution_applied(self, event: Any) -> None:
        """
        Corpus 14 §7 - Simula → Oneiros consolidation feed (Spec 14).

        Each applied structural evolution is an unconsolidated episode for the
        memory consolidation system. Recording it here increments sleep pressure
        so Oneiros schedules consolidation after a burst of self-evolution.

        Optionally, if the proposal is high-risk (multiple systems affected),
        it also adds extra affect residue to accelerate consolidation.
        """
        if self._stage_controller.is_sleeping:
            return  # Already consolidating

        data = event.data if hasattr(event, "data") else {}
        self._clock.record_episode()

        # Amplify pressure for wide-blast proposals: each file beyond 3 = extra affect
        files_changed = data.get("files_changed", [])
        if len(files_changed) > 3:
            # Each extra file is a low-valence, moderate-arousal structural trace
            extra_affect = min(0.3, (len(files_changed) - 3) * 0.05)
            self._clock.record_affect_trace(-extra_affect, 0.4)

        self._logger.debug(
            "oneiros_evolution_episode_recorded",
            proposal_id=data.get("proposal_id", ""),
            files_changed=len(files_changed),
            pressure=round(self._clock.pressure.composite_pressure, 3),
        )

    async def _on_federation_sleep_sync(self, event: Any) -> None:
        """
        Spec 14 §8 - Federation sleep coordination.

        When a federated peer requests sleep synchronization, Oneiros evaluates
        whether to honour the proposed sleep time. If already sleeping, ignores.
        If wake-state and the proposed time is within the next theta cycle, nudges
        sleep pressure to trigger imminent consolidation so the organism enters
        sleep in rough synchrony with the requesting peer - enabling shared
        knowledge to be certified and broadcast together.

        The actual timing guarantee is best-effort: Oneiros will not override
        EMERGENCY/CRITICAL metabolic starvation guards or active task constraints.
        """
        if self._stage_controller.is_sleeping:
            return  # Already consolidating - sync happened naturally

        data = event.data if hasattr(event, "data") else {}
        peer_instance_id = data.get("instance_id", "unknown")
        priority = float(data.get("priority", 0.5))

        # Only act on high-priority federation sync requests
        if priority < 0.5:
            self._logger.debug(
                "federation_sleep_sync_ignored_low_priority",
                peer=peer_instance_id,
                priority=priority,
            )
            return

        # Nudge sleep pressure upward so the scheduler will trigger sleep soon.
        # We record a synthetic consolidation episode to raise the episode component.
        self._clock.record_episode()

        self._logger.info(
            "federation_sleep_sync_received",
            peer=peer_instance_id,
            priority=priority,
            proposed_sleep_time=data.get("proposed_sleep_time_utc", ""),
            current_pressure=round(self._clock.pressure.composite_pressure, 3),
        )

    async def _on_critical_event(self, event: Any) -> None:
        """Handle critical Synapse events during sleep."""
        if not self._stage_controller.is_sleeping:
            return

        event_type = getattr(event, "event_type", None)

        self._logger.warning(
            "emergency_wake_triggered",
            event_type=str(event_type),
            current_stage=self._stage_controller.current_stage.value,
        )

        self._stage_controller.emergency_wake(f"Critical event: {event_type}")

        if self._current_cycle is not None:
            self._current_cycle.interrupted = True
            self._current_cycle.interrupt_reason = f"Critical: {event_type}"

        # Capture the current checkpoint before cancelling the task so that
        # the next sleep cycle can resume from SLOW_WAVE rather than restarting
        # from scratch.  interrupt() stores the checkpoint in _pending_restore_checkpoint.
        await self._v2_engine.interrupt(reason=f"emergency_wake:{event_type}")

        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()

        await self._emit_event(EMERGENCY_WAKE, {
            "reason": str(event_type),
            "stage_interrupted": self._stage_controller.current_stage.value,
        })

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle VitalityCoordinator austerity orders.

        Skia emits SYSTEM_MODULATION when the organism needs to conserve resources.
        This system applies the directive and ACKs so Skia knows the order was received.
        """
        data = getattr(event, "data", {}) or {}
        level = data.get("level", "nominal")
        halt_systems = data.get("halt_systems", [])
        modulate = data.get("modulate", {})

        system_id = "oneiros"
        compliant = True
        reason: str | None = None

        if system_id in halt_systems:
            self._modulation_halted = True
            self._logger.warning("system_modulation_halt", level=level)
        elif system_id in modulate:
            directives = modulate[system_id]
            self._apply_modulation_directives(directives)
            self._logger.info("system_modulation_applied", level=level, directives=directives)
        elif level == "nominal":
            self._modulation_halted = False
            self._logger.info("system_modulation_resumed", level=level)

        # Emit ACK so Skia knows the order was received
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                    data={
                        "system_id": system_id,
                        "level": level,
                        "compliant": compliant,
                        "reason": reason,
                    },
                    source_system=system_id,
                ))
            except Exception as exc:
                self._logger.warning("modulation_ack_failed", error=str(exc))

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply modulation directives from VitalityCoordinator.

        Oneiros directive: {"dream_frequency_factor": 0.5} - reduce dream
        frequency to conserve cognitive resources during austerity.
        """
        factor = directives.get("dream_frequency_factor")
        if factor is not None:
            self._logger.info("modulation_dream_frequency_reduced", factor=factor)
        else:
            self._logger.info("modulation_directives_received", directives=directives)

    # ── Public API ────────────────────────────────────────────────

    def set_creative_goal(self, goal: str) -> None:
        """Set a creative goal for the next lucid dreaming phase."""
        self._creative_goal = goal
        self._stage_controller.set_has_creative_goal(True)
        self._v2_engine.set_creative_goal(goal)
        self._logger.info("creative_goal_set", goal=goal[:100])

    def get_pending_insights(self) -> list[Any]:
        """Get insights discovered during sleep for wake broadcast."""
        insights = self._pending_wake_insights.copy()
        self._pending_wake_insights.clear()
        return insights

    @property
    def is_sleeping(self) -> bool:
        return self._stage_controller.is_sleeping

    @property
    def current_stage(self) -> SleepStage:
        return self._stage_controller.current_stage

    @property
    def sleep_pressure(self) -> float:
        return self._clock.pressure.composite_pressure

    @property
    def degradation(self) -> WakeDegradation:
        return self._clock.degradation

    @property
    def stats(self) -> dict[str, Any]:
        """Aggregate statistics."""
        return {
            "total_sleep_cycles": self._total_sleep_cycles,
            "total_dreams": self._total_dreams,
            "total_insights": self._total_insights,
            "insights_validated": self._insights_validated,
            "insights_integrated": self._insights_integrated,
            "episodes_consolidated": self._episodes_consolidated,
            "semantic_nodes_created": self._semantic_nodes_created,
            "traces_pruned": self._traces_pruned,
            "affect_traces_processed": self._affect_traces_processed,
            "threats_simulated": self._threats_simulated,
            "lucid_proposals_submitted": self._lucid_proposals_submitted,
            "lucid_proposals_accepted": self._lucid_proposals_accepted,
            "lucid_proposals_rejected": self._lucid_proposals_rejected,
            "mean_dream_coherence": (
                self._dream_coherence_sum / max(self._total_dreams, 1)
            ),
            "mean_sleep_quality": (
                self._sleep_quality_sum / max(self._total_sleep_cycles, 1)
            ),
            "current_pressure": self._clock.pressure.composite_pressure,
            "current_stage": self._stage_controller.current_stage.value,
            "current_degradation": self._clock.degradation.composite_impairment,
        }

    # ── Health ────────────────────────────────────────────────────

    def _build_health_snapshot(self) -> OneirosHealthSnapshot:
        """Build the health snapshot."""
        degradation = self._clock.degradation
        pressure = self._clock.pressure

        return OneirosHealthSnapshot(
            status="sleeping" if self.is_sleeping else "healthy",
            current_stage=self._stage_controller.current_stage,
            sleep_pressure=pressure.composite_pressure,
            wake_degradation=degradation.composite_impairment,
            current_sleep_debt_hours=(
                pressure.cycles_since_sleep * 0.00015 / 3600
                if pressure.composite_pressure > pressure.threshold
                else 0.0
            ),
            total_sleep_cycles=self._total_sleep_cycles,
            total_dreams=self._total_dreams,
            total_insights=self._total_insights,
            insights_validated=self._insights_validated,
            insights_invalidated=self._insights_invalidated,
            insights_integrated=self._insights_integrated,
            mean_dream_coherence=(
                self._dream_coherence_sum / max(self._total_dreams, 1)
            ),
            mean_sleep_quality=(
                self._sleep_quality_sum / max(self._total_sleep_cycles, 1)
            ),
            episodes_consolidated=self._episodes_consolidated,
            semantic_nodes_created=self._semantic_nodes_created,
            traces_pruned=self._traces_pruned,
            hypotheses_pruned=self._hypotheses_pruned,
            hypotheses_promoted=self._hypotheses_promoted,
            affect_traces_processed=self._affect_traces_processed,
            mean_affect_reduction=0.0,
            threats_simulated=self._threats_simulated,
            response_plans_created=self._response_plans_created,
            lucid_proposals_submitted=self._lucid_proposals_submitted,
            lucid_proposals_accepted=self._lucid_proposals_accepted,
            lucid_proposals_rejected=self._lucid_proposals_rejected,
            last_sleep_completed=pressure.last_sleep_completed,
            last_sleep_quality=(
                self._recent_cycles[-1].quality if self._recent_cycles else None
            ),
        )

    # ── Internal Helpers ──────────────────────────────────────────

    async def _emit_event(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        """Emit a Synapse event."""
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent

            event = SynapseEvent(
                id=new_id(),
                event_type=event_type,
                timestamp=utc_now(),
                data=data,
                source_system="oneiros",
            )
            await self._synapse.event_bus.emit(event)
        except Exception as exc:
            self._logger.debug("event_emit_failed", event_type=str(event_type), error=str(exc))

    def _emit_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Emit a telemetry metric (fire-and-forget coroutine)."""
        if self._metrics is None:
            return
        parts = name.split(".", 1)
        system = parts[0] if len(parts) == 2 else "oneiros"
        metric = parts[1] if len(parts) == 2 else name
        with contextlib.suppress(Exception):
            asyncio.create_task(
                self._metrics.record(system, metric, value, labels=tags or {})
            )



def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
