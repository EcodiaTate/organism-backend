"""
EcodiaOS — Evo Consolidation Orchestrator

The "sleep mode" of the learning system. Runs every 6 hours or 10,000
cognitive cycles — whichever comes first.

Ten phases (spec Section VII):
  1.    Memory consolidation       — delegate to MemoryService
  2.    Hypothesis review          — integrate supported, archive refuted/stale
  2.5   Belief aging               — flag stale beliefs for re-verification
  2.75  Belief consolidation       — harden high-confidence beliefs into read-only nodes
  2.8   Genetic fixation           — compress stable beliefs into inheritable genome
  3.    Schema induction           — propose new entity/relation types from clusters
  4.    Procedure extraction       — codify mature action sequences as Procedures
  5.    Parameter optimisation     — apply supported parameter hypotheses
  6.    Self-model update          — recompute capability and effectiveness metrics
  7.    Drift data feed            — send effectiveness data to Equor's drift detector
  8.    Evolution proposals        — flag structural changes that warrant Simula review

Performance budget: ≤60 seconds end-to-end (spec Section X).

Guard rails:
  - Velocity limits enforced by ParameterTuner
  - Evo cannot touch Equor evaluation logic (EVO_CONSTRAINTS)
  - Evolution proposals are submitted to Simula, not applied directly
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.evo.types import (
    VELOCITY_LIMITS,
    ConsolidationResult,
    EvolutionProposal,
    HypothesisCategory,
    HypothesisStatus,
    MutationType,
    PatternContext,
    SchemaInduction,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from systems.evo.belief_consolidation import BeliefConsolidationScanner
    from systems.evo.belief_halflife import BeliefAgingScanner
    from systems.evo.cognitive_niche import NicheRegistry
    from systems.evo.genetic_memory import GenomeExtractor
    from systems.evo.hypothesis import HypothesisEngine
    from systems.evo.meta_learning import MetaLearningEngine
    from systems.evo.niche_forking import NicheForkingEngine
    from systems.evo.parameter_tuner import ParameterTuner
    from systems.evo.pressure import EvolutionaryPressureSystem
    from systems.evo.procedure_codifier import ProcedureCodifier
    from systems.evo.procedure_extractor import ProcedureExtractor
    from systems.evo.schema_induction import SchemaInductionEngine
    from systems.evo.self_model import SelfModelManager
    from systems.evo.speciation import SpeciationEngine
    from systems.evo.tournament import TournamentEngine
    from systems.memory.service import MemoryService
    from systems.simula.coevolution.causal_surgery import (
        CausalFailureAnalyzer,
    )
    from systems.simula.coevolution.causal_surgery_types import (
        FailurePattern,
    )
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()

_CONSOLIDATION_INTERVAL_HOURS: int = 6
_CONSOLIDATION_CYCLE_THRESHOLD: int = 10_000


class ConsolidationOrchestrator:
    """
    Drives the full consolidation pipeline — the organism dreaming.

    Receives the active hypothesis list and pattern context from EvoService.
    Coordinates all sub-systems through the 8-phase pipeline.
    """

    def __init__(
        self,
        hypothesis_engine: HypothesisEngine,
        parameter_tuner: ParameterTuner,
        procedure_extractor: ProcedureExtractor,
        self_model_manager: SelfModelManager,
        memory: MemoryService | None = None,
        logos: Any | None = None,
        simula_callback: Callable[..., Any] | None = None,
        belief_aging_scanner: BeliefAgingScanner | None = None,
        belief_consolidation_scanner: BeliefConsolidationScanner | None = None,
        tournament_engine: TournamentEngine | None = None,
        genome_extractor: GenomeExtractor | None = None,
        causal_surgery_analyzer: CausalFailureAnalyzer | None = None,
        procedure_codifier: ProcedureCodifier | None = None,
        event_bus: EventBus | None = None,
        schema_induction_engine: SchemaInductionEngine | None = None,
        meta_learning_engine: MetaLearningEngine | None = None,
        speciation_engine: SpeciationEngine | None = None,
        niche_forking_engine: NicheForkingEngine | None = None,
        pressure_system: EvolutionaryPressureSystem | None = None,
        niche_registry: NicheRegistry | None = None,
    ) -> None:
        self._hypotheses = hypothesis_engine
        self._tuner = parameter_tuner
        self._extractor = procedure_extractor
        self._codifier = procedure_codifier
        self._self_model = self_model_manager
        self._memory = memory
        self._logos = logos
        self._simula_callback = simula_callback
        self._belief_aging = belief_aging_scanner
        self._belief_consolidation = belief_consolidation_scanner
        self._tournament = tournament_engine
        self._genome_extractor = genome_extractor
        self._causal_surgery = causal_surgery_analyzer
        self._event_bus = event_bus
        self._schema_engine = schema_induction_engine
        self._meta_learning = meta_learning_engine
        self._speciation = speciation_engine
        self._niche_forking = niche_forking_engine
        self._pressure = pressure_system
        self._niche_registry = niche_registry
        self._detected_failure_patterns: list[FailurePattern] = []
        self._logger = logger.bind(system="evo.consolidation")

        self._last_run_at = utc_now() - timedelta(hours=_CONSOLIDATION_INTERVAL_HOURS)
        self._total_runs: int = 0

    def should_run(self, cycle_count: int, cycles_since_last: int) -> bool:
        """
        Return True if consolidation is due.
        Triggers on:
          - 6 hours elapsed since last run
          - 10,000 cognitive cycles since last run
        """
        hours_elapsed = (utc_now() - self._last_run_at).total_seconds() / 3600
        if hours_elapsed >= _CONSOLIDATION_INTERVAL_HOURS:
            return True
        return cycles_since_last >= _CONSOLIDATION_CYCLE_THRESHOLD

    async def run(self, pattern_context: PatternContext) -> ConsolidationResult:
        """
        Execute the full 8-phase consolidation pipeline.
        Returns a ConsolidationResult summary.

        Never raises — all phases handle their own exceptions.
        """
        self._logger.info("consolidation_starting")
        start = time.monotonic()
        result = ConsolidationResult(triggered_at=utc_now())

        # ── Phase 1: Memory Consolidation ────────────────────────────────────
        await self._phase_memory_consolidation()

        # ── Snapshot supported hypotheses BEFORE Phase 2 removes them ─────────
        # Phase 2 integrates/archives supported hypotheses which removes them
        # from the active list. Phases 3 and 5 need these hypotheses, so we
        # snapshot them first.
        self._supported_snapshot = list(self._hypotheses.get_supported())

        # ── Phase 2: Hypothesis Review ────────────────────────────────────────
        integrated, archived = await self._phase_hypothesis_review()
        result.hypotheses_evaluated = integrated + archived
        result.hypotheses_integrated = integrated
        result.hypotheses_archived = archived

        # ── Phase 2 (tournament): Hypothesis Tournaments ──────────────────────
        t_active, t_converged = await self._phase_tournament_update()
        result.tournaments_active = t_active
        result.tournaments_converged = t_converged

        # ── Phase 2.5: Belief Aging ───────────────────────────────────────────
        aging_result = await self._phase_belief_aging()
        result.beliefs_stale = aging_result.beliefs_stale
        result.beliefs_critical = aging_result.beliefs_critical

        # ── Phase 2.75: Belief Consolidation ──────────────────────────────────
        consolidation_result = await self._phase_belief_consolidation()
        result.beliefs_consolidated = consolidation_result.beliefs_consolidated
        result.foundation_conflicts = consolidation_result.foundation_conflicts

        # ── Phase 2.8: Genetic Fixation ───────────────────────────────────────
        genome_result = await self._phase_genetic_fixation()
        result.genome_candidates_fixed = genome_result.candidates_fixed
        result.genome_size_bytes = genome_result.genome_size_bytes

        # ── Phase 2.9: Cognitive Speciation ───────────────────────────────────
        speciation_result = await self._phase_speciation()
        result.niches_created = speciation_result.niches_created
        result.niches_extinct = speciation_result.niches_extinct
        result.speciation_events = len(speciation_result.events)
        result.ring_species_detected = speciation_result.ring_species_detected

        # ── Phase 2.95: Niche Forking (Cognitive Organogenesis) ───────────────
        forking_result = await self._phase_niche_forking()
        result.niche_forks_proposed = forking_result.proposals_generated
        result.worldview_forks = forking_result.worldview_forks

        # ── Phase 3: Schema Induction ─────────────────────────────────────────
        schemas_induced = await self._phase_schema_induction()
        result.schemas_induced = schemas_induced

        # ── Phase 4: Procedure Extraction ─────────────────────────────────────
        self._extractor.begin_cycle()
        procedures_extracted = await self._phase_procedure_extraction(pattern_context)
        result.procedures_extracted = procedures_extracted

        # ── Phase 5: Parameter Optimisation ───────────────────────────────────
        self._tuner.begin_cycle()
        adj_count, total_delta = await self._phase_parameter_optimisation()
        result.parameters_adjusted = adj_count
        result.total_parameter_delta = total_delta

        # ── Phase 6: Self-Model Update ────────────────────────────────────────
        await self._phase_self_model_update()
        result.self_model_updated = True

        # ── Phase 6.25: Meta-Learning Update ──────────────────────────────────
        await self._phase_meta_learning_update()

        # ── Phase 6.5: Causal Failure Pattern Detection (Prompt #16) ─────────
        self._detected_failure_patterns = await self._phase_failure_pattern_detection()

        # ── Phase 7: Drift Data Feed ──────────────────────────────────────────
        await self._phase_drift_feed()

        # ── Phase 8: Evolution Proposals ──────────────────────────────────────
        await self._phase_evolution_proposals()

        # ── Housekeeping ──────────────────────────────────────────────────────
        pattern_context.reset()
        self._last_run_at = utc_now()
        self._total_runs += 1

        result.duration_ms = int((time.monotonic() - start) * 1000)
        self._logger.info(
            "consolidation_complete",
            duration_ms=result.duration_ms,
            hypotheses_integrated=result.hypotheses_integrated,
            hypotheses_archived=result.hypotheses_archived,
            procedures_extracted=result.procedures_extracted,
            parameters_adjusted=result.parameters_adjusted,
            total_parameter_delta=round(result.total_parameter_delta, 4),
            beliefs_stale=result.beliefs_stale,
            beliefs_critical=result.beliefs_critical,
            beliefs_consolidated=result.beliefs_consolidated,
            foundation_conflicts=result.foundation_conflicts,
            tournaments_active=result.tournaments_active,
            tournaments_converged=result.tournaments_converged,
            genome_candidates_fixed=result.genome_candidates_fixed,
            genome_size_bytes=result.genome_size_bytes,
            speciation_events=result.speciation_events,
            niches_created=result.niches_created,
            niches_extinct=result.niches_extinct,
            niche_forks_proposed=result.niche_forks_proposed,
        )
        return result

    # ─── Phases ───────────────────────────────────────────────────────────────

    async def _phase_memory_consolidation(self) -> None:
        """
        Phase 1: Graph maintenance (Memory) then batch compression (Logos).

        Memory handles salience decay, community detection, semantic compression,
        and entity deduplication. Logos handles the MDL / world-model compression
        cascade. Oneiros owns the deeper Memory Ladder during SLOW_WAVE sleep.
        """
        if self._memory is not None:
            try:
                await self._memory.consolidate()
                self._logger.info("memory_consolidation_complete")
            except Exception as exc:
                self._logger.error("memory_consolidation_failed", error=str(exc))

        if self._logos is not None:
            try:
                compression_report = await self._logos.run_batch_compression(
                    force=True, max_items=100,
                )
                self._logger.info(
                    "logos_compression_complete",
                    items=compression_report.items_processed,
                    bits_saved=round(compression_report.bits_saved, 1),
                )
            except Exception as exc:
                self._logger.warning("logos_compression_failed", error=str(exc))

    async def _phase_hypothesis_review(self) -> tuple[int, int]:
        """
        Phase 2: Review all active hypotheses.
          - SUPPORTED → attempt integration (calls HypothesisEngine.integrate_hypothesis)
          - REFUTED   → archive
          - Stale     → archive
        Returns (integrated_count, archived_count).
        """
        integrated = 0
        archived = 0

        all_hypotheses = self._hypotheses.get_all_active()

        for h in all_hypotheses:
            try:
                if h.status == HypothesisStatus.SUPPORTED:
                    success = await self._hypotheses.integrate_hypothesis(h)
                    if success:
                        integrated += 1
                        # Persist the integrated hypothesis as a :Belief node
                        # with domain-aware half-life metadata
                        await self._persist_belief_from_hypothesis(h)
                        # Feed meta-learning: successful integration
                        if self._meta_learning is not None:
                            self._meta_learning.record_hypothesis_outcome(
                                h.id, h.source_detector or "unknown", "integrated",
                            )
                elif h.status == HypothesisStatus.REFUTED:
                    await self._hypotheses.archive_hypothesis(h, reason="refuted")
                    archived += 1
                    if self._meta_learning is not None:
                        self._meta_learning.record_hypothesis_outcome(
                            h.id, h.source_detector or "unknown", "refuted",
                        )
                elif self._hypotheses.is_stale(h):
                    await self._hypotheses.archive_hypothesis(h, reason="stale")
                    archived += 1
                    if self._meta_learning is not None:
                        self._meta_learning.record_hypothesis_outcome(
                            h.id, h.source_detector or "unknown", "stale",
                        )
            except Exception as exc:
                self._logger.warning(
                    "hypothesis_review_failed",
                    hypothesis_id=h.id,
                    error=str(exc),
                )

        return integrated, archived

    async def _phase_tournament_update(self) -> tuple[int, int]:
        """
        Phase 2 (tournament): Update hypothesis tournaments.

        Three sub-steps:
          1. Check convergence on running tournaments (declare winners)
          2. Archive converged tournaments (persist to Neo4j)
          3. Detect and create new tournaments from hypothesis clusters

        Returns (active_tournament_count, newly_converged_count).
        """
        if self._tournament is None:
            return 0, 0

        try:
            # Step 1: Check convergence on running tournaments
            converged = await self._tournament.check_convergence()

            # Step 2: Archive converged tournaments
            await self._tournament.archive_converged()

            # Step 3: Detect and create new tournaments from hypothesis clusters
            new_tournaments = self._tournament.detect_and_create_tournaments()
            if new_tournaments:
                self._logger.info(
                    "tournaments_created",
                    count=len(new_tournaments),
                    tournament_ids=[t.id for t in new_tournaments],
                )

            active_count = len(self._tournament.get_active_tournaments())
            converged_count = len(converged)

            return active_count, converged_count

        except Exception as exc:
            self._logger.error("tournament_update_phase_failed", error=str(exc))
            return 0, 0

    async def _phase_belief_aging(self) -> Any:
        """
        Phase 2.5: Scan beliefs for half-life decay and flag stale ones.

        Identifies beliefs that have crossed their domain-specific half-life
        threshold and marks them for re-verification. Returns a BeliefAgingResult.
        """
        from systems.evo.belief_halflife import BeliefAgingResult

        if self._belief_aging is None:
            return BeliefAgingResult()

        try:
            aging_result = await self._belief_aging.scan_stale_beliefs()

            if aging_result.beliefs_stale > 0:
                self._logger.info(
                    "belief_aging_stale_detected",
                    stale=aging_result.beliefs_stale,
                    critical=aging_result.beliefs_critical,
                    top_stale=[
                        {
                            "id": s.belief_id,
                            "domain": s.domain,
                            "age_factor": s.age_factor,
                            "elapsed_days": s.elapsed_days,
                        }
                        for s in aging_result.stale_beliefs[:5]
                    ],
                )

            return aging_result
        except Exception as exc:
            self._logger.error("belief_aging_phase_failed", error=str(exc))
            return BeliefAgingResult()

    async def _phase_belief_consolidation(self) -> Any:
        """
        Phase 2.75: Harden high-confidence, low-volatility beliefs into
        read-only :ConsolidatedBelief reference nodes.

        Also checks for foundation conflicts — hypotheses that contradict
        consolidated beliefs are logged at high severity.

        Returns a BeliefConsolidationResult.
        """
        from systems.evo.types import BeliefConsolidationResult

        if self._belief_consolidation is None:
            return BeliefConsolidationResult()

        try:
            # Consolidate eligible beliefs
            consolidation_result = await self._belief_consolidation.scan_and_consolidate()

            # Check for foundation conflicts with active hypotheses
            all_active = self._hypotheses.get_all_active()
            conflicts = await self._belief_consolidation.check_foundation_conflicts(
                all_active
            )

            consolidation_result.foundation_conflicts = len(conflicts)
            consolidation_result.conflicts = conflicts

            if conflicts:
                self._logger.warning(
                    "foundation_conflicts_detected",
                    count=len(conflicts),
                    conflicts=[
                        {
                            "hypothesis_id": c.hypothesis_id,
                            "consolidated_belief_id": c.consolidated_belief_id,
                            "severity": c.severity,
                        }
                        for c in conflicts[:5]
                    ],
                )

            return consolidation_result
        except Exception as exc:
            self._logger.error("belief_consolidation_phase_failed", error=str(exc))
            return BeliefConsolidationResult()

    async def _phase_genetic_fixation(self) -> Any:
        """
        Phase 2.8: Extract stable beliefs into an inheritable genome.

        Identifies hypotheses meeting genetic fixation criteria (confidence >=
        0.95, volatility < 0.1, age > 30 days, no contradictions) and compresses
        them into a BeliefGenome that child instances can inherit at birth.

        Only runs if the instance is mature enough (>10,000 episodes, >100
        confirmed hypotheses). Returns a GenomeExtractionResult.
        """
        from systems.evo.types import GenomeExtractionResult

        if self._genome_extractor is None:
            return GenomeExtractionResult()

        try:
            genome, extraction_result = await self._genome_extractor.extract_genome()

            if genome is not None:
                self._logger.info(
                    "genetic_fixation_complete",
                    genome_id=genome.id,
                    fixed_count=extraction_result.candidates_fixed,
                    genome_bytes=extraction_result.genome_size_bytes,
                )

            return extraction_result
        except Exception as exc:
            self._logger.error("genetic_fixation_phase_failed", error=str(exc))
            return GenomeExtractionResult()

    async def _phase_speciation(self) -> Any:
        """
        Phase 2.9: Cognitive Speciation — evolve new ways of thinking.

        Runs five speciation mechanisms: allopatric (domain divergence),
        sympatric (same-domain worldview splits), adaptive radiation
        (novel domain pressure), parapatric (boundary zone divergence),
        and ring species detection (circular incompatibility).

        Creates and evolves cognitive niches — isolated hypothesis
        ecosystems with their own fitness landscapes.
        """
        from systems.evo.speciation import SpeciationResult

        if self._speciation is None or self._pressure is None:
            return SpeciationResult()

        try:
            # Gather inputs
            all_active = list(self._hypotheses.get_active())
            species = self._pressure.detect_species(all_active)

            # Collect hypothesis fitness scores
            hypothesis_fitness: dict[str, float] = {}
            for h in all_active:
                score = self._pressure.score_hypothesis_fitness(h)
                hypothesis_fitness[h.id] = score.fitness

            result = await self._speciation.run_speciation_cycle(
                hypotheses=all_active,
                species=species,
                hypothesis_fitness=hypothesis_fitness,
            )

            if result.events:
                self._logger.info(
                    "speciation_phase_complete",
                    events=len(result.events),
                    niches_created=result.niches_created,
                    niches_extinct=result.niches_extinct,
                    ring_species=result.ring_species_detected,
                    duration_ms=result.duration_ms,
                )

            return result
        except Exception as exc:
            self._logger.error("speciation_phase_failed", error=str(exc))
            return SpeciationResult()

    async def _phase_niche_forking(self) -> Any:
        """
        Phase 2.95: Niche Forking — cognitive organogenesis.

        Examines mature niches and proposes architectural forks:
        specialized detectors, evidence functions, consolidation
        strategies, schema topologies, or full worldview forks.

        The organism growing new cognitive organs.
        """
        from systems.evo.niche_forking import NicheForkingResult

        if self._niche_forking is None:
            return NicheForkingResult()

        try:
            all_active = list(self._hypotheses.get_active())
            result = await self._niche_forking.run_forking_cycle(
                hypotheses=all_active,
            )

            if result.proposals_generated > 0:
                self._logger.info(
                    "niche_forking_phase_complete",
                    proposals=result.proposals_generated,
                    detector_forks=result.detector_forks,
                    evidence_forks=result.evidence_forks,
                    consolidation_forks=result.consolidation_forks,
                    schema_forks=result.schema_forks,
                    worldview_forks=result.worldview_forks,
                    duration_ms=result.duration_ms,
                )

            return result
        except Exception as exc:
            self._logger.error("niche_forking_phase_failed", error=str(exc))
            return NicheForkingResult()

    async def _phase_schema_induction(self) -> int:
        """
        Phase 3: Induce new schema elements from the knowledge graph.

        If SchemaInductionEngine is wired, runs the full 3-strategy pipeline:
          1. Graph motif mining → entity type proposals
          2. Analogical structure mapping → abstract relation types
          3. MDL filtering → keep only schemas that compress

        Also feeds supported WORLD_MODEL hypotheses as additional schema seeds.
        Falls back to the legacy per-hypothesis MERGE if no engine is available.

        Returns the count of schemas induced.
        """
        supported = getattr(self, "_supported_snapshot", [])

        # ── New engine path ───────────────────────────────────────────────
        if self._schema_engine is not None:
            try:
                result = await self._schema_engine.induce(
                    supported_hypotheses=supported,
                )
                total = (
                    result.entity_types_discovered
                    + result.relation_types_discovered
                    + result.compositions_discovered
                )
                self._logger.info(
                    "schema_induction_engine_complete",
                    entities=result.entity_types_discovered,
                    relations=result.relation_types_discovered,
                    compositions=result.compositions_discovered,
                    mdl_gain=round(result.mdl_total_gain_bits, 2),
                    duration_ms=result.duration_ms,
                )
                return total
            except Exception as exc:
                self._logger.error(
                    "schema_induction_engine_failed",
                    error=str(exc),
                )
                # Fall through to legacy path

        # ── Legacy fallback ───────────────────────────────────────────────
        schemas_induced = 0
        for h in supported:
            if h.category != HypothesisCategory.WORLD_MODEL:
                continue
            if h.proposed_mutation is None:
                continue
            if h.proposed_mutation.type != MutationType.SCHEMA_ADDITION:
                continue

            schema = SchemaInduction(
                entities=[{"name": h.proposed_mutation.target, "description": h.statement}],
                source_hypothesis=h.id,
            )
            success = await self._apply_schema_induction(schema)
            if success:
                schemas_induced += 1

        return schemas_induced

    async def _phase_procedure_extraction(self, context: PatternContext) -> int:
        """
        Phase 4: Extract procedures from mature action-sequence patterns.
        Returns the count of new procedures extracted.
        """
        # Get action-sequence patterns that have hit the occurrence threshold (≥3)
        all_patterns = context.get_mature_sequences(min_occurrences=3)

        extracted = 0
        for pattern in all_patterns:
            procedure = await self._extractor.extract_procedure(pattern)
            if procedure is not None:
                extracted += 1

        # Also run ProcedureCodifier — converts mature (Intent, Outcome) pairs
        # accumulated via on_action_completed into Procedure objects.
        if self._codifier is not None:
            codified = await self._codifier.codify()
            extracted += len(codified)

        return extracted

    async def _phase_parameter_optimisation(self) -> tuple[int, float]:
        """
        Phase 5: Apply supported parameter hypotheses.
        Uses the pre-Phase-2 snapshot so hypotheses are available after integration.
        Velocity-limited to prevent lurching changes.
        Returns (adjustment_count, total_absolute_delta).
        """
        supported = getattr(self, "_supported_snapshot", [])
        candidates: list[Any] = []

        for h in supported:
            if h.category != HypothesisCategory.PARAMETER:
                continue
            adj = self._tuner.propose_adjustment(h)
            if adj is not None:
                candidates.append(adj)

        if not candidates:
            return 0, 0.0

        # Check velocity limit for the batch
        allowed, reason = self._tuner.check_velocity_limit(candidates)
        if not allowed:
            self._logger.warning("parameter_velocity_limit", reason=reason)
            # Apply as many as we can without exceeding total limit
            limit = VELOCITY_LIMITS["max_total_parameter_delta_per_cycle"]
            running_delta = 0.0
            filtered = []
            for adj in sorted(candidates, key=lambda a: abs(a.delta), reverse=False):
                if running_delta + abs(adj.delta) <= limit:
                    filtered.append(adj)
                    running_delta += abs(adj.delta)
            candidates = filtered

        applied = 0
        total_delta = 0.0
        for adj in candidates:
            await self._tuner.apply_adjustment(adj)
            applied += 1
            total_delta += abs(adj.delta)

        return applied, total_delta

    async def _phase_self_model_update(self) -> None:
        """Phase 6: Recompute self-model from recent outcome episodes."""
        try:
            await self._self_model.update()
        except Exception as exc:
            self._logger.error("self_model_update_failed", error=str(exc))

    async def _phase_meta_learning_update(self) -> None:
        """
        Phase 6.25: Meta-learning — Evo learns about HOW it learns.

        Adapts detector sensitivity, evidence thresholds, and hypothesis
        quality scoring based on accumulated outcome statistics. Adjustments
        are velocity-limited at half the normal rate to prevent meta-instability.
        """
        if self._meta_learning is None:
            return
        try:
            report = self._meta_learning.update()
            if report.changes:
                self._logger.info(
                    "meta_learning_adjustments",
                    adjustments=report.adjustments_made,
                    changes=report.changes,
                    threshold_multiplier=report.learning_rate.threshold_multiplier,
                    false_positive_rate=report.learning_rate.false_positive_rate,
                )
        except Exception as exc:
            self._logger.error("meta_learning_update_failed", error=str(exc))

    async def _phase_drift_feed(self) -> None:
        """
        Phase 7: Feed effectiveness data to Equor's drift detector.
        Drift detection is handled by Equor; we just update the data it reads.
        The self-model stats written to the Self node are what Equor reads.
        """
        stats = self._self_model.get_current()
        self._logger.debug(
            "drift_data_available",
            success_rate=round(stats.success_rate, 3),
            mean_alignment=round(stats.mean_alignment, 3),
        )
        # Equor reads from Self node directly; no active push needed in Phase 7.
        # Future: could publish a Synapse event for Equor to act on immediately.

    async def _phase_evolution_proposals(self) -> None:
        """
        Phase 8: Submit structural change proposals to Simula for warranted cases.

        Pipeline:
          1. Collect hypotheses pointing to architectural change
          2. Build lightweight EvolutionProposal objects
          3. Sort by evidence score (highest first) as a pre-ranking heuristic
          4. For high-confidence candidates (confidence >= 0.9, evidence_score >= 8.0):
             emit EVOLUTION_CANDIDATE on Synapse so Simula can receive via event bus
          5. Submit to Simula via bridge callback — each proposal is scored by
             Simula's ArchitectureEFEScorer during process_proposal() and
             its EFE is persisted as a :ProposalEFE node for monitoring

        The full EFE ranking happens inside Simula's pipeline. Evo submits
        in evidence-strength order; Simula attaches EFE scores, persists them,
        and the Equor review queue can consume them ranked by EFE.
        """
        # Use the pre-Phase-2 snapshot: by the time Phase 8 runs, integrate_hypothesis
        # has already removed SUPPORTED hypotheses from the active registry.  Phases 3
        # and 5 already follow this pattern; Phase 8 must too.
        supported = getattr(self, "_supported_snapshot", [])
        evolution_candidates = [
            h for h in supported
            if (
                h.proposed_mutation is not None
                and h.proposed_mutation.type == MutationType.EVOLUTION_PROPOSAL
                and h.evidence_score > 5.0
            )
        ]

        if not evolution_candidates:
            return

        # Sort by evidence score descending (strongest evidence first)
        evolution_candidates.sort(key=lambda h: h.evidence_score, reverse=True)

        self._logger.info(
            "evolution_proposals_phase8",
            candidates=len(evolution_candidates),
            top_evidence=evolution_candidates[0].evidence_score if evolution_candidates else 0,
        )

        for h in evolution_candidates:
            if h.proposed_mutation is None:
                continue
            proposal = EvolutionProposal(
                description=h.proposed_mutation.description or h.statement,
                rationale=h.statement,
                supporting_hypotheses=[h.id],
            )
            self._logger.info(
                "evolution_proposal_generated",
                hypothesis_id=h.id,
                description=proposal.description[:80],
            )

            # Compute normalised confidence from evidence score
            # (mirrors the sigmoid used in belief_store: min(0.99, 0.5 + score * 0.05))
            confidence = min(0.99, 0.5 + h.evidence_score * 0.05)

            # High-confidence candidates (>= 0.9) emit an EVOLUTION_CANDIDATE Synapse
            # event so Simula can receive them via event-bus subscription in addition
            # to the direct callback path below.
            if confidence >= 0.9 and self._event_bus is not None:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.EVOLUTION_CANDIDATE,
                        source_system="evo.consolidation",
                        data={
                            "hypothesis_id": h.id,
                            "hypothesis_statement": h.statement,
                            "evidence_score": h.evidence_score,
                            "confidence": round(confidence, 3),
                            "mutation_type": h.proposed_mutation.type.value,
                            "mutation_target": h.proposed_mutation.target,
                            "mutation_description": h.proposed_mutation.description,
                            "supporting_episodes": h.supporting_episodes,
                        },
                    )
                )
                self._logger.info(
                    "evolution_candidate_emitted",
                    hypothesis_id=h.id,
                    confidence=round(confidence, 3),
                    evidence_score=round(h.evidence_score, 2),
                )

            # Submit to Simula via bridge callback — only when the event-bus path
            # was not used, to avoid duplicate delivery into process_proposal().
            emitted_via_event_bus = (
                confidence >= 0.9 and self._event_bus is not None
            )
            if not emitted_via_event_bus and self._simula_callback is not None:
                try:
                    result = await self._simula_callback(proposal, [h])
                    self._logger.info(
                        "evolution_proposal_submitted_to_simula",
                        hypothesis_id=h.id,
                        result_status=getattr(result, "status", "unknown"),
                    )
                except Exception as exc:
                    self._logger.error(
                        "simula_submission_failed",
                        hypothesis_id=h.id,
                        error=str(exc),
                    )

        # Causal self-surgery: targeted proposals from failure patterns (Prompt #16)
        await self._phase_causal_surgery_proposals()

    async def _phase_failure_pattern_detection(self) -> list[FailurePattern]:
        """
        Phase 6.5: Detect recurring failure patterns from RegretStats.

        Uses CausalFailureAnalyzer to query Neo4j for high-regret resolved
        counterfactuals clustered by (policy_type, goal_domain).
        Only triggers when high_regret_count >= 5.
        """
        if self._causal_surgery is None:
            return []

        regret = self._self_model.get_current().regret
        if regret.high_regret_count < 5:
            return []

        try:
            patterns = await self._causal_surgery.detect_failure_patterns(regret)
            if patterns:
                self._logger.info(
                    "failure_patterns_phase6_5",
                    count=len(patterns),
                    top_regret=round(patterns[0].mean_regret, 2),
                )
            return patterns
        except Exception as exc:
            self._logger.error("failure_pattern_detection_failed", error=str(exc))
            return []

    async def _phase_causal_surgery_proposals(self) -> None:
        """
        Phase 8 extension: Generate surgical evolution proposals from
        causal failure analysis (Prompt #16).

        For each detected failure pattern (max 3 per cycle):
          1. Run CausalFailureAnalyzer.analyze_failure_pattern()
          2. If a best_intervention is found, translate to EvolutionProposal
          3. Submit to Simula via bridge callback
        """
        if self._causal_surgery is None or not self._detected_failure_patterns:
            return

        for pattern in self._detected_failure_patterns[:3]:
            try:
                surgery_result = await self._causal_surgery.analyze_failure_pattern(
                    pattern,
                )
                if surgery_result.best_intervention is None:
                    continue

                # Build a lightweight Evo proposal with the surgical description
                intervention = surgery_result.best_intervention
                intervention.failure_pattern_id = pattern.pattern_id

                description = (
                    f"Causal surgery: {intervention.direction.value} "
                    f"{intervention.parameter}"
                )
                if intervention.condition:
                    description += f" ({intervention.condition})"

                proposal = EvolutionProposal(
                    description=description,
                    rationale=(
                        f"Causal analysis of {intervention.episodes_analyzed} "
                        f"failure episodes identified {intervention.parameter} "
                        f"as the critical intervention point "
                        f"({intervention.intervention_success_rate:.0%} success rate, "
                        f"confidence={intervention.confidence:.2f})"
                    ),
                    supporting_hypotheses=[],
                )

                self._logger.info(
                    "causal_surgery_proposal_generated",
                    pattern_id=pattern.pattern_id,
                    parameter=intervention.parameter,
                    direction=intervention.direction.value,
                    success_rate=round(intervention.intervention_success_rate, 2),
                )

                if self._simula_callback is not None:
                    # REVIEW: bridge.translate_surgical_proposal() could produce a
                    # richer EvolutionProposal (with ATE, counterfactuals, etc.) but
                    # the consolidation path doesn't have access to the bridge or to
                    # the raw intervention object at callback time. Currently the
                    # description text carries enough signal for rule-based category
                    # inference ("parameter" → ADJUST_BUDGET). If surgical precision
                    # is needed, wire bridge into ConsolidationOrchestrator.
                    result = await self._simula_callback(proposal, [])
                    self._logger.info(
                        "causal_surgery_proposal_submitted",
                        pattern_id=pattern.pattern_id,
                        result_status=getattr(result, "status", "unknown"),
                    )

            except Exception as exc:
                self._logger.error(
                    "causal_surgery_proposal_failed",
                    pattern_id=pattern.pattern_id,
                    error=str(exc),
                )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    async def _persist_belief_from_hypothesis(self, hypothesis: Any) -> None:
        """
        Convert an integrated hypothesis into a persisted :Belief node
        with domain-aware half-life metadata.
        """
        if self._memory is None:
            return
        try:
            from systems.memory.belief_store import (
                store_belief_from_hypothesis,
            )

            await store_belief_from_hypothesis(
                neo4j=self._memory._neo4j,
                hypothesis_id=hypothesis.id,
                statement=hypothesis.statement,
                category=hypothesis.category.value,
                evidence_score=hypothesis.evidence_score,
                supporting_episodes=hypothesis.supporting_episodes,
            )
        except Exception as exc:
            self._logger.warning(
                "belief_persistence_from_hypothesis_failed",
                hypothesis_id=hypothesis.id,
                error=str(exc),
            )

    async def _apply_schema_induction(self, schema: SchemaInduction) -> bool:
        """Apply schema induction to the Memory graph."""
        if self._memory is None or not schema.entities:
            return False
        try:
            for entity_spec in schema.entities:
                name = entity_spec.get("name", "")
                description = entity_spec.get("description", "")
                if name:
                    await self._memory._neo4j.execute_write(
                        """
                        MERGE (et:EvoEntityType {name: $name})
                        SET et.description = $description,
                            et.source_hypothesis = $source_hypothesis,
                            et.induced_at = datetime()
                        """,
                        {
                            "name": name,
                            "description": description,
                            "source_hypothesis": schema.source_hypothesis,
                        },
                    )
            return True
        except Exception as exc:
            self._logger.warning("schema_induction_failed", error=str(exc))
            return False

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_runs": self._total_runs,
            "last_run_at": self._last_run_at.isoformat(),
        }
