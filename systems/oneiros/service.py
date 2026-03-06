"""
EcodiaOS — Oneiros: The Dream Engine Service (v2)

System #13 orchestrator. Coordinates the circadian rhythm, sleep
stage transitions (v2 batch compiler: DESCENT → SLOW_WAVE → REM → EMERGENCE),
economic dreaming, somatic sleep analysis, and wake degradation.

Thymos gave the organism a will to live. Oneiros gives it an inner life.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any

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
    The Dream Engine — System #13 (v2 batch compiler).

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
        self._equor: Any = None
        self._evo: Any = None
        self._nova: Any = None
        self._atune: Any = None
        self._thymos: Any = None
        self._memory: Any = None
        self._soma: Any = None
        self._oikos: Any = None
        self._simula: Any = None
        self._kairos: Any = None
        self._economic_dream_worker: Any = None
        self._threat_model_worker: Any = None

        # Loop 5 — Kairos → Oneiros priority REM seed buffer.
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

        # Pending insights for wake broadcast
        self._pending_wake_insights: list[Any] = []

        # v2 Sleep Cycle Engine (Spec 14 — Sleep as Batch Compiler)
        event_bus = None
        if synapse is not None:
            with contextlib.suppress(AttributeError):
                event_bus = synapse.event_bus
        self._v2_engine = SleepCycleEngine(event_bus=event_bus)

        self._logger = logger

    # ── Cross-System Wiring ───────────────────────────────────────

    def set_equor(self, equor: Any) -> None:
        self._equor = equor
        self._v2_engine.set_equor(equor)

    def set_evo(self, evo: Any) -> None:
        self._evo = evo
        self._v2_engine.set_evo(evo)

    def set_nova(self, nova: Any) -> None:
        self._nova = nova

    def set_atune(self, atune: Any) -> None:
        self._atune = atune

    def set_thymos(self, thymos: Any) -> None:
        self._thymos = thymos

    def set_memory(self, memory: Any) -> None:
        self._memory = memory

    def set_soma(self, soma: Any) -> None:
        """Wire Soma for sleep pressure from energy errors and somatic sleep analysis."""
        self._soma = soma

    def set_simula(self, simula: Any) -> None:
        """Wire Simula for lucid dreaming mutation testing."""
        self._simula = simula
        self._v2_engine.set_simula(simula)

    def set_oikos(self, oikos: Any) -> None:
        """Wire Oikos for economic dreaming + threat modeling during sleep."""
        from systems.oikos.dream_worker import EconomicDreamWorker
        from systems.oikos.threat_model_worker import ThreatModelWorker

        self._oikos = oikos
        self._economic_dream_worker = EconomicDreamWorker(
            config=oikos._config,
        )
        if oikos._config.threat_model_enabled:
            self._threat_model_worker = ThreatModelWorker(
                config=oikos._config,
            )
        self._logger.info("oikos_wired_for_economic_dreaming")

    def set_fovea(self, fovea: Any) -> None:
        """Wire Fovea for v2 REM error-domain targeting and checkpoint backlog count."""
        self._v2_engine.set_fovea(fovea)

    def set_logos(self, logos: Any) -> None:
        """Wire Logos for v2 Sleep Cycle Engine (world model access)."""
        self._v2_engine.set_logos(logos)

    def set_kairos(self, kairos: Any) -> None:
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

        # Update Evo hypothesis count periodically
        if self._evo is not None and self._clock.pressure.cycles_since_sleep % 100 == 0:
            try:
                if hasattr(self._evo, "_hypothesis_engine"):
                    engine = self._evo._hypothesis_engine
                    count = len(getattr(engine, "_hypotheses", {}))
                    self._clock.record_hypothesis_count(count)
            except Exception:
                pass

        # Augment sleep pressure from allostatic energy errors
        if self._soma is not None:
            try:
                signal = self._soma.get_current_signal()
                errors = signal.state.errors.get("immediate", {})
                energy_error = errors.get("energy", 0.0)
                if energy_error < -0.3:
                    depletion_valence = max(-1.0, energy_error * 2.0)
                    self._clock.record_affect_trace(depletion_valence, 0.5)
            except Exception:
                pass

        # Emit metrics
        self._emit_metric("oneiros.sleep_pressure", self._clock.pressure.composite_pressure)

        # Check sleep triggers — cycle count gate fires first
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
        """Called when Memory stores an episode — more accurate pressure tracking."""
        self._clock.record_affect_trace(valence, arousal)

    # ── Sleep Orchestration ───────────────────────────────────────

    async def begin_sleep(self) -> None:
        """
        Initiate a v2 sleep cycle (DESCENT → SLOW_WAVE → REM → EMERGENCE).
        """
        if self._stage_controller.is_sleeping:
            self._logger.warning("already_sleeping")
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
        try:
            # Determine trigger from pressure
            trigger = SleepTrigger.SCHEDULED
            if self._clock.must_sleep():
                trigger = SleepTrigger.COGNITIVE_PRESSURE

            kairos_seeds = self.drain_kairos_rem_seeds()
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

        run_sleep_analysis = getattr(self._soma, "run_sleep_analysis", None)
        if run_sleep_analysis is None:
            return

        try:
            sleep_report = await run_sleep_analysis()

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
          1. EconomicDreamWorker — organism-level cashflow GBM (survival/ruin)
          2. ThreatModelWorker — per-asset shock distributions with contagion
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

                self._oikos.integrate_dream_result(result)

                if hasattr(result, "ruin_probability"):
                    self._logger.info(
                        "economic_dreaming_integrated",
                        ruin_probability=str(result.ruin_probability),
                        survival_30d=str(result.survival_probability_30d),
                        resilience=str(result.resilience_score),
                        recommendations=len(result.recommendations),
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

        await self._emit_event(WAKE_ONSET, {
            "cycle_id": self._current_cycle.id,
            "quality": quality.value,
            "insights_count": len(self._pending_wake_insights),
            "dreams_generated": self._current_cycle.dreams_generated,
            "episodes_consolidated": self._current_cycle.episodes_replayed,
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
        Loop 5 — Kairos → Oneiros (Synapse path).

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

        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()

        await self._emit_event(EMERGENCY_WAKE, {
            "reason": str(event_type),
            "stage_interrupted": self._stage_controller.current_stage.value,
        })

    # ── Public API ────────────────────────────────────────────────

    def set_creative_goal(self, goal: str) -> None:
        """Set a creative goal for the next lucid dreaming phase."""
        self._creative_goal = goal
        self._stage_controller.set_has_creative_goal(True)
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


def _make_hedge_condition(target_symbol_lower: str) -> Any:
    """
    Create a condition lambda for a hedge procedure template.

    Fires when a broadcast contains economic/price content mentioning
    the target asset symbol.
    """
    def condition(broadcast: object) -> bool:
        try:
            content = getattr(broadcast, "content", None)
            if content is None:
                return False
            content_text = ""
            for attr in ("content", "text", "summary"):
                val = getattr(content, attr, None)
                if isinstance(val, str) and val:
                    content_text = val.lower()
                    break
            if not content_text:
                return False
            has_symbol = target_symbol_lower in content_text
            has_economic = any(
                kw in content_text
                for kw in ("price", "crash", "liquidat", "depeg", "drop",
                           "decline", "risk", "alert", "flash", "exploit")
            )
            return has_symbol and has_economic
        except Exception:
            return False
    return condition


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
