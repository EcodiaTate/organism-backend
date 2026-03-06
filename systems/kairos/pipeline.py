"""
EcodiaOS -- Kairos: Pipeline Orchestrator

Wires the Kairos causal mining pipeline to the EOS event bus.

Phases A+B (existing): Correlation mining, causal direction, confounder
analysis, context invariance testing.

Phase C (new): Invariant distillation — variable abstraction, tautology test,
minimality test, domain mapping. Tier 3 promotion with broadcast.

Phase D (new): Counter-invariant detection — violation scanning, clustering,
scope refinement. Intelligence contribution ledger — per-invariant accounting.

Events broadcast:
  KAIROS_CAUSAL_CANDIDATE_GENERATED, KAIROS_CAUSAL_DIRECTION_ACCEPTED,
  KAIROS_CONFOUNDER_DISCOVERED, KAIROS_INVARIANT_CANDIDATE,
  KAIROS_INVARIANT_DISTILLED, KAIROS_TIER3_INVARIANT_DISCOVERED,
  KAIROS_COUNTER_INVARIANT_FOUND, KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE

Integrations: Logos (world model), Nexus (federation sharing of Tier 3),
  Telos (intelligence ratio step change signal).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.kairos.causal_direction import CausalDirectionTester
from systems.kairos.confounder import ConfounderAnalyzer
from systems.kairos.context_invariance import ContextInvarianceTester
from systems.kairos.correlation_miner import CorrelationMiner
from systems.kairos.counter_invariant import CounterInvariantDetector
from systems.kairos.hierarchy import CausalHierarchy
from systems.kairos.intelligence_ledger import IntelligenceContributionLedger
from systems.kairos.invariant_distiller import InvariantDistiller
from systems.kairos.types import (
    CausalCandidatePayload,
    CausalDirectionPayload,
    CausalDirectionResult,
    CausalInvariant,
    CausalRule,
    ConfounderDiscoveredPayload,
    CorrelationCandidate,
    CounterInvariantPayload,
    IntelligenceRatioStepChangePayload,
    InvarianceVerdict,
    InvariantCandidatePayload,
    InvariantDistilledPayload,
    KairosConfig,
    Tier3InvariantPayload,
)

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

logger = structlog.get_logger("kairos.pipeline")


class KairosEvoPipeline:
    """
    Subscribes to Evo hypothesis output. When Evo generates a hypothesis
    that is a causal claim, feed it into Stage 1 as a pre-seeded
    correlation candidate.
    """

    def __init__(self) -> None:
        self._causal_claims_received: int = 0
        self._seeded_candidates: list[CorrelationCandidate] = []

    def process_evolution_candidate(
        self, event_data: dict[str, Any],
    ) -> CorrelationCandidate | None:
        """
        Check if an Evo hypothesis is a causal claim. If so, convert it
        to a pre-seeded CorrelationCandidate for Stage 1.
        """
        statement = event_data.get("hypothesis_statement", "")
        category = event_data.get("category", "")

        if category and category != "world_model":
            return None

        causal_keywords = [
            "causes", "leads to", "produces", "results in",
            "drives", "triggers", "induces", "generates",
            "increases", "decreases", "affects",
        ]

        is_causal = any(kw in statement.lower() for kw in causal_keywords)
        if not is_causal:
            return None

        self._causal_claims_received += 1

        hypothesis_id = event_data.get("hypothesis_id", "")
        confidence = event_data.get("confidence", 0.0)

        candidate = CorrelationCandidate(
            variable_a=f"evo_hyp_{hypothesis_id}_cause",
            variable_b=f"evo_hyp_{hypothesis_id}_effect",
            mean_correlation=confidence,
            cross_context_variance=0.0,
            context_count=1,
        )

        self._seeded_candidates.append(candidate)

        logger.info(
            "evo_causal_claim_seeded",
            hypothesis_id=hypothesis_id,
            statement=statement[:100],
            confidence=confidence,
        )

        return candidate

    @property
    def seeded_candidates(self) -> list[CorrelationCandidate]:
        return list(self._seeded_candidates)

    def clear_seeded(self) -> None:
        self._seeded_candidates.clear()


class KairosPipeline:
    """
    Main Kairos pipeline orchestrator.

    Manages the full lifecycle:
    1. Event subscriptions (Fovea, Evo, Memory)
    2. Pipeline execution (Stages 1-3 + context invariance)
    3. Phase C: Invariant distillation + domain mapping + Tier 3 promotion
    4. Phase D: Counter-invariant detection + intelligence ledger
    5. Synapse event broadcasting (8 event types)
    6. Logos world model integration
    7. Nexus federation sharing (Tier 3 invariants)
    """

    system_id: str = "kairos"

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()

        # Pipeline stages (Phases A+B)
        self._correlation_miner = CorrelationMiner(self._config)
        self._direction_tester = CausalDirectionTester(self._config)
        self._confounder_analyzer = ConfounderAnalyzer(self._config)
        self._invariance_tester = ContextInvarianceTester(self._config)
        self._hierarchy = CausalHierarchy(self._config)
        self._evo_pipeline = KairosEvoPipeline()

        # Phase C: Distillation
        self._distiller = InvariantDistiller(self._config)

        # Phase D: Counter-invariants + Intelligence ledger
        self._counter_detector = CounterInvariantDetector(self._config)
        self._intelligence_ledger = IntelligenceContributionLedger(self._config)

        # Register Tier 3 callback on hierarchy
        self._hierarchy.on_tier3_discovered(self._on_tier3_promotion)

        # External dependencies (wired after construction)
        self._event_bus: EventBus | None = None
        self._logos_ingest: Any = None
        self._nexus_share: Any = None  # Nexus fragment sharing protocol
        self._oneiros: Any = None  # Loop 5: Oneiros REM seed injection

        # Metrics
        self._pipeline_runs: int = 0
        self._fovea_events_received: int = 0
        self._evo_events_received: int = 0
        self._cross_domain_events_received: int = 0
        self._invariants_created: int = 0
        self._tier3_discoveries: int = 0
        self._counter_invariants_found: int = 0
        self._step_changes: int = 0

    # --- Wiring ---

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire the Synapse event bus and register subscriptions."""
        self._event_bus = event_bus
        self._register_subscriptions()
        logger.info("kairos_event_bus_wired")

    def set_logos(self, logos_ingest: Any) -> None:
        """Wire the Logos invariant ingestion protocol."""
        self._logos_ingest = logos_ingest
        logger.info("kairos_logos_wired")

    def set_nexus(self, nexus_share: Any) -> None:
        """Wire the Nexus fragment sharing protocol for Tier 3 federation."""
        self._nexus_share = nexus_share
        logger.info("kairos_nexus_wired")

    def set_oneiros(self, oneiros: Any) -> None:
        """Wire Oneiros for Loop 5 REM seed injection on Tier 3 discoveries."""
        self._oneiros = oneiros
        logger.info("kairos_oneiros_wired")

    def _register_subscriptions(self) -> None:
        """Subscribe to relevant Synapse events."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEventType

        self._event_bus.subscribe(
            SynapseEventType.FOVEA_PREDICTION_ERROR,
            self._on_fovea_prediction_error,
        )
        self._event_bus.subscribe(
            SynapseEventType.EVOLUTION_CANDIDATE,
            self._on_evolution_candidate,
        )
        self._event_bus.subscribe(
            SynapseEventType.EPISODE_STORED,
            self._on_episode_stored,
        )
        self._event_bus.subscribe(
            SynapseEventType.CROSS_DOMAIN_MATCH_FOUND,
            self._on_cross_domain_match,
        )

        logger.info("kairos_subscriptions_registered", events=4)

    # --- Event handlers ---

    async def _on_fovea_prediction_error(self, event: SynapseEvent) -> None:
        """
        Handle Fovea prediction errors routed to Kairos.

        Fovea is Kairos's PRIMARY input: a causal_error > threshold means the
        world model failed to predict a causal relationship.  We convert the
        error payload into a pre-seeded CorrelationCandidate so the next
        pipeline run starts with these causal surprises already in Stage 1.
        """
        data = event.data
        routes = data.get("routes", [])
        causal_error = data.get("causal_error", 0.0)

        if "kairos" not in routes and causal_error <= self._config.causal_error_route_threshold:
            return

        self._fovea_events_received += 1

        percept_id = data.get("percept_id", "")
        dominant = data.get("dominant_error_type", "unknown")

        # Build a pre-seeded CorrelationCandidate from the causal surprise.
        # variable_a = the cause side Fovea expected; variable_b = the effect.
        # mean_correlation is set to causal_error (higher surprise → stronger
        # signal that a real causal relationship exists to be mined).
        candidate = CorrelationCandidate(
            variable_a=f"fovea_causal_cause:{percept_id}",
            variable_b=f"fovea_causal_effect:{percept_id}",
            mean_correlation=causal_error,
            cross_context_variance=0.0,  # Single observation — variance unknown
            context_count=1,
        )
        self._correlation_miner.add_preseed(candidate)

        logger.debug(
            "fovea_causal_error_received",
            percept_id=percept_id,
            causal_error=causal_error,
            dominant_error=dominant,
            candidate_id=candidate.id,
        )

    async def _on_evolution_candidate(self, event: SynapseEvent) -> None:
        """Handle Evo evolution candidates that might be causal claims."""
        self._evo_events_received += 1
        self._evo_pipeline.process_evolution_candidate(event.data)

    async def _on_episode_stored(self, event: SynapseEvent) -> None:
        """
        Handle new episodes for observation buffering.

        Memory episodes are part of the observation stream that feeds Stage 1.
        Extract variable pairs from the episode's context and content, then
        pre-seed the correlation miner so the next pipeline run includes them.
        """
        data = event.data if hasattr(event, "data") else {}
        context_id = data.get("context_id", data.get("episode_id", ""))
        observations = data.get("observations", [])

        if not context_id or not observations:
            return

        # Buffer observations keyed by context — the next run_pipeline() call
        # will include them via the observations_by_context argument.
        # For pre-seeding: if the episode contains causal annotations, create
        # CorrelationCandidates directly.
        causal_annotations = data.get("causal_annotations", [])
        for annotation in causal_annotations:
            cause = annotation.get("cause", "")
            effect = annotation.get("effect", "")
            strength = float(annotation.get("strength", 0.5))
            if cause and effect:
                candidate = CorrelationCandidate(
                    variable_a=f"episode:{cause}",
                    variable_b=f"episode:{effect}",
                    mean_correlation=strength,
                    cross_context_variance=0.0,
                    context_count=1,
                )
                self._correlation_miner.add_preseed(candidate)

        logger.debug(
            "episode_stored_processed",
            context_id=context_id,
            observation_count=len(observations),
            causal_annotations=len(causal_annotations),
        )

    async def _on_cross_domain_match(self, event: SynapseEvent) -> None:
        """
        Loop 5 — Oneiros → Kairos.

        Oneiros REM emits CROSS_DOMAIN_MATCH_FOUND when it discovers a structural
        isomorphism between two domains.  A cross-domain match IS an invariance
        candidate: if the same abstract structure appears in domain A and domain B,
        that is a substrate-level invariant waiting to be mined.

        Convert the match to a pre-seeded CorrelationCandidate so the next
        run_pipeline() call tests it through the full causal pipeline.
        """
        data = event.data if hasattr(event, "data") else {}
        self._cross_domain_events_received += 1

        domain_a = data.get("domain_a", "")
        domain_b = data.get("domain_b", "")
        iso_score = float(data.get("isomorphism_score", 0.0))

        candidate = CorrelationCandidate(
            variable_a=f"oneiros_domain:{domain_a}",
            variable_b=f"oneiros_domain:{domain_b}",
            mean_correlation=iso_score,
            cross_context_variance=0.0,
            context_count=2,
        )
        self._correlation_miner.add_preseed(candidate)

        logger.debug(
            "cross_domain_match_preseeded",
            domain_a=domain_a,
            domain_b=domain_b,
            iso_score=round(iso_score, 3),
        )

    # --- Pipeline execution ---

    async def run_pipeline(
        self,
        observations_by_context: dict[str, list[dict[str, Any]]],
        temporal_events: list[dict[str, Any]] | None = None,
        axon_logs: list[dict[str, Any]] | None = None,
        known_domains: list[str] | None = None,
        total_model_length: float = 0.0,
    ) -> dict[str, Any]:
        """
        Run the full Kairos pipeline (Phases A-D).

        Args:
            observations_by_context: context_id to list of observation dicts.
            temporal_events: Temporal event stream for direction testing.
            axon_logs: Axon execution logs for intervention asymmetry (Phase B).
            known_domains: All domains in the world model (Phase C domain mapping).
            total_model_length: Total world model description length in bits (Phase D).

        Returns:
            Pipeline run summary.
        """
        self._pipeline_runs += 1
        temporal_events = temporal_events or []
        axon_logs = axon_logs or []
        known_domains = known_domains or []

        # ── Stage 1: Correlation Mining ──────────────────────────
        candidates = await self._correlation_miner.mine(observations_by_context)

        # Add Evo-seeded candidates
        candidates.extend(self._evo_pipeline.seeded_candidates)
        self._evo_pipeline.clear_seeded()

        for candidate in candidates:
            await self._emit_candidate_generated(candidate)

        # ── Stage 2: Causal Direction Testing ────────────────────
        accepted_directions: list[CausalDirectionResult] = []
        for candidate in candidates:
            obs_pairs = self._extract_observation_pairs(
                candidate, observations_by_context
            )

            result = await self._direction_tester.test_direction(
                candidate, temporal_events, axon_logs, obs_pairs
            )
            if result.accepted:
                accepted_directions.append(result)
                await self._emit_direction_accepted(result)

        # ── Stage 3: Confounder Analysis ─────────────────────────
        clean_directions: list[CausalDirectionResult] = []
        for direction_result in accepted_directions:
            confounder_result = await self._confounder_analyzer.analyze(
                direction_result, observations_by_context
            )

            if confounder_result.is_confounded:
                await self._emit_confounder_discovered(confounder_result)
            else:
                clean_directions.append(direction_result)

        # ── Stage 5: Context Invariance Testing ──────────────────
        # (Stage 4 — Mechanism Extraction — deferred to future phase)
        invariants_created = 0
        new_invariants: list[CausalInvariant] = []
        for direction_result in clean_directions:
            rule = CausalRule(
                cause_variable=direction_result.cause,
                effect_variable=direction_result.effect,
                direction_confidence=direction_result.confidence,
                domain="",
                observation_count=direction_result.candidate.context_count,
                source_candidate_id=direction_result.candidate.id,
            )

            invariance_result = await self._invariance_tester.test_invariance(
                rule, observations_by_context
            )

            if invariance_result.verdict in (
                InvarianceVerdict.STRONG_INVARIANT,
                InvarianceVerdict.CONDITIONAL_INVARIANT,
            ):
                invariant = self._hierarchy.create_from_rule(rule, invariance_result)
                self._invariants_created += 1
                invariants_created += 1
                new_invariants.append(invariant)
                await self._emit_invariant_candidate(
                    invariant, rule, invariance_result,
                )
                await self._ingest_to_logos(invariant, rule)

        # ── Phase C: Invariant Distillation ──────────────────────
        distillations = 0
        tautologies_rejected = 0
        domains_mapped = 0

        for invariant in new_invariants:
            distill_result = await self._distiller.distill(invariant, known_domains)
            distillations += 1
            if distill_result.is_tautological:
                tautologies_rejected += 1
                continue
            domains_mapped += len(distill_result.untested_domains)
            await self._emit_invariant_distilled(invariant, distill_result)

            # Re-evaluate tier after distillation (Phase C unlocks Tier 3)
            self._hierarchy.promote_if_eligible(invariant.id)

        # Also distill existing undistilled invariants
        for invariant in self._hierarchy.get_all():
            if not invariant.distilled:
                distill_result = await self._distiller.distill(invariant, known_domains)
                distillations += 1
                if not distill_result.is_tautological:
                    domains_mapped += len(distill_result.untested_domains)
                    await self._emit_invariant_distilled(invariant, distill_result)
                    self._hierarchy.promote_if_eligible(invariant.id)

        # ── Phase D: Counter-Invariant Detection ─────────────────
        violations_found = 0
        refinements_made = 0

        for invariant in self._hierarchy.get_all():
            violations = await self._counter_detector.scan_for_violations(
                invariant, observations_by_context
            )
            violations_found += len(violations)

            if violations:
                clusters = await self._counter_detector.cluster_violations(
                    invariant, violations
                )
                refined = await self._counter_detector.refine_scope(
                    invariant, clusters, observations_by_context
                )
                if refined is not None:
                    refinements_made += 1
                    self._counter_invariants_found += 1
                    await self._emit_counter_invariant_found(invariant, refined)

        # ── Phase D: Intelligence Contribution Ledger ────────────
        all_invariants = self._hierarchy.get_all()
        if all_invariants:
            contributions = self._intelligence_ledger.compute_all(
                all_invariants, observations_by_context, total_model_length
            )

            # Check for step changes
            for contribution in contributions:
                inv = self._hierarchy._find(contribution.invariant_id)
                if inv is not None:
                    old_ratio = contribution.intelligence_ratio_without
                    is_step, delta = self._intelligence_ledger.detect_step_change(
                        inv, old_ratio
                    )
                    if is_step:
                        self._step_changes += 1
                        await self._emit_intelligence_ratio_step_change(
                            inv, old_ratio, contribution.intelligence_ratio_contribution, delta,
                            "pipeline_run",
                        )

        summary = {
            "pipeline_run": self._pipeline_runs,
            "stage1_candidates": len(candidates),
            "stage2_accepted": len(accepted_directions),
            "stage3_clean": len(clean_directions),
            "invariants_created": invariants_created,
            "phase_c": {
                "distillations": distillations,
                "tautologies_rejected": tautologies_rejected,
                "domains_mapped": domains_mapped,
            },
            "phase_d": {
                "violations_found": violations_found,
                "refinements_made": refinements_made,
                "step_changes": self._step_changes,
            },
            "hierarchy": self._hierarchy.summary(),
            "intelligence_ledger": self._intelligence_ledger.summary(),
        }

        logger.info("pipeline_run_complete", **summary)
        return summary

    # --- Helpers ---

    @staticmethod
    def _extract_observation_pairs(
        candidate: CorrelationCandidate,
        observations_by_context: dict[str, list[dict[str, Any]]],
    ) -> list[tuple[float, float]] | None:
        """Extract paired (a, b) observations for the ANM test."""
        for obs_list in observations_by_context.values():
            pairs: list[tuple[float, float]] = []
            for obs in obs_list:
                a = obs.get(candidate.variable_a)
                b = obs.get(candidate.variable_b)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    pairs.append((float(a), float(b)))
            if len(pairs) >= 10:
                return pairs
        return None

    # --- Logos integration ---

    async def _ingest_to_logos(self, invariant: Any, rule: CausalRule) -> None:
        """Feed a confirmed invariant into the Logos world model."""
        if self._logos_ingest is None:
            return

        from systems.logos.types import EmpiricalInvariant

        logos_invariant = EmpiricalInvariant(
            statement=invariant.abstract_form,
            domain=rule.domain or "general",
            observation_count=rule.observation_count,
            confidence=invariant.invariance_hold_rate,
            source="kairos",
        )

        self._logos_ingest.ingest_invariant(logos_invariant)

        logger.info(
            "invariant_ingested_to_logos",
            invariant_id=invariant.id,
            statement=invariant.abstract_form[:80],
        )

        # Emit WORLD_MODEL_UPDATED to close Loop 3:
        # Fovea's weight learner listens for this event to recalibrate its
        # causal_error component — fewer causal errors → higher Kairos SNR.
        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.WORLD_MODEL_UPDATED,
                    source_system="kairos",
                    data={
                        "update_type": "invariant_integrated",
                        "schemas_added": 0,
                        "priors_updated": 0,
                        "causal_updates": 1,
                        "invariant_id": invariant.id,
                        "invariant_tier": invariant.tier,
                        "source": "kairos",
                    },
                )
            )

    # --- Synapse event emission ---

    async def _emit_candidate_generated(self, candidate: CorrelationCandidate) -> None:
        """Broadcast KAIROS_CAUSAL_CANDIDATE_GENERATED."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = CausalCandidatePayload(
            candidate_id=candidate.id,
            variable_a=candidate.variable_a,
            variable_b=candidate.variable_b,
            mean_correlation=candidate.mean_correlation,
            cross_context_variance=candidate.cross_context_variance,
            context_count=candidate.context_count,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_CAUSAL_CANDIDATE_GENERATED,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    async def _emit_direction_accepted(self, result: CausalDirectionResult) -> None:
        """Broadcast KAIROS_CAUSAL_DIRECTION_ACCEPTED."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        methods_agreed = sum(
            1 for tr in result.test_results
            if tr.direction == result.direction and tr.confidence > 0.1
        )

        payload = CausalDirectionPayload(
            result_id=result.id,
            cause=result.cause,
            effect=result.effect,
            direction=result.direction.value,
            confidence=result.confidence,
            methods_agreed=methods_agreed,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_CAUSAL_DIRECTION_ACCEPTED,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    async def _emit_confounder_discovered(self, confounder_result: Any) -> None:
        """Broadcast KAIROS_CONFOUNDER_DISCOVERED."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = ConfounderDiscoveredPayload(
            result_id=confounder_result.id,
            original_cause=confounder_result.original_pair.cause,
            original_effect=confounder_result.original_pair.effect,
            confounders=[c.variable for c in confounder_result.confounding_variables],
            mdl_improvement=confounder_result.mdl_improvement,
            is_spurious=confounder_result.is_confounded,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_CONFOUNDER_DISCOVERED,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    async def _emit_invariant_candidate(
        self,
        invariant: CausalInvariant,
        rule: CausalRule,
        invariance_result: Any,
    ) -> None:
        """Broadcast KAIROS_INVARIANT_CANDIDATE — Stage 5 strong invariant."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = InvariantCandidatePayload(
            invariant_id=invariant.id,
            cause=rule.cause_variable,
            effect=rule.effect_variable,
            hold_rate=invariance_result.hold_rate,
            context_count=invariance_result.context_count,
            verdict=invariance_result.verdict.value,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_INVARIANT_CANDIDATE,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    async def _emit_invariant_distilled(
        self,
        invariant: CausalInvariant,
        distill_result: Any,
    ) -> None:
        """Broadcast KAIROS_INVARIANT_DISTILLED — Stage 6 complete."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = InvariantDistilledPayload(
            invariant_id=invariant.id,
            abstract_form=invariant.abstract_form,
            domain_count=invariant.domain_count,
            is_minimal=distill_result.is_minimal,
            untested_domain_count=len(distill_result.untested_domains),
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_INVARIANT_DISTILLED,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    # --- Phase C+D: Tier 3 promotion callback ---

    def _on_tier3_promotion(self, invariant: CausalInvariant) -> None:
        """
        Called by CausalHierarchy when an invariant is promoted to Tier 3.

        This is the highest-priority event in the system:
        1. Broadcast TIER3_INVARIANT_DISCOVERED on Synapse
        2. Feed to Logos for deepest-layer world model integration
        3. Feed to Nexus for immediate federation sharing
        4. Signal intelligence ratio step change to Telos
        """
        self._tier3_discoveries += 1

        logger.info(
            "tier3_invariant_event",
            invariant_id=invariant.id,
            abstract_form=invariant.abstract_form[:80],
            domain_count=invariant.domain_count,
            substrate_count=invariant.substrate_count,
        )

        # These are async operations; schedule them without blocking the sync callback
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._handle_tier3_async(invariant))
        except RuntimeError:
            logger.warning(
                "tier3_async_scheduling_failed",
                invariant_id=invariant.id,
                reason="no_running_event_loop",
            )

    async def _handle_tier3_async(self, invariant: CausalInvariant) -> None:
        """Async handler for Tier 3 promotion — broadcasts and integrations."""
        await self._emit_tier3_discovered(invariant)
        await self._share_with_nexus(invariant)
        await self._inject_oneiros_rem_seed(invariant)

    # --- Nexus federation sharing ---

    async def _share_with_nexus(self, invariant: CausalInvariant) -> None:
        """Share a Tier 3 invariant with Nexus for immediate federation broadcast."""
        if self._nexus_share is None:
            return

        from systems.nexus.types import ShareableWorldModelFragment

        fragment = ShareableWorldModelFragment(
            source_instance_id="local",
            abstract_structure={
                "type": "causal_invariant",
                "tier": 3,
                "abstract_form": invariant.abstract_form,
                "hold_rate": invariant.invariance_hold_rate,
                "domains": [d.domain for d in invariant.applicable_domains],
                "substrates": list(
                    {d.substrate for d in invariant.applicable_domains if d.substrate}
                ),
            },
            domain_labels=[d.domain for d in invariant.applicable_domains],
            observations_explained=sum(
                d.observation_count for d in invariant.applicable_domains
            ),
            description_length=invariant.description_length_bits,
            compression_ratio=(
                invariant.intelligence_ratio_contribution
                if invariant.intelligence_ratio_contribution > 0
                else 1.0
            ),
        )

        try:
            self._nexus_share.share_fragment(fragment)
            logger.info(
                "tier3_shared_with_nexus",
                invariant_id=invariant.id,
                fragment_id=fragment.fragment_id,
            )
        except Exception:
            logger.exception(
                "nexus_sharing_failed",
                invariant_id=invariant.id,
            )

    # --- Loop 5: Oneiros REM seed injection ---

    async def _inject_oneiros_rem_seed(self, invariant: CausalInvariant) -> None:
        """
        Loop 5 — Kairos → Oneiros (direct path).

        Inject a Tier 3 invariant as a priority REM seed so the next Oneiros
        dream cycle asks: "what other domains might this substrate-level
        pattern apply to?" — closing the Oneiros ↔ Kairos discovery loop.
        """
        if self._oneiros is None:
            return

        seed = {
            "source": "kairos_tier3",
            "invariant_id": invariant.id,
            "abstract_form": invariant.abstract_form,
            "tier": invariant.tier,
            "hold_rate": invariant.invariance_hold_rate,
            "applicable_domains": [d.domain for d in invariant.applicable_domains],
            "untested_domains": list(invariant.untested_domains),
            "intelligence_ratio_contribution": invariant.intelligence_ratio_contribution,
        }

        if hasattr(self._oneiros, "add_kairos_rem_seed"):
            self._oneiros.add_kairos_rem_seed(seed)
            logger.info(
                "tier3_injected_as_oneiros_rem_seed",
                invariant_id=invariant.id,
                abstract_form=invariant.abstract_form[:60],
            )

    # --- Phase C+D Synapse event emission ---

    async def _emit_tier3_discovered(self, invariant: CausalInvariant) -> None:
        """Broadcast KAIROS_TIER3_INVARIANT_DISCOVERED — highest priority."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = Tier3InvariantPayload(
            invariant_id=invariant.id,
            abstract_form=invariant.abstract_form,
            domain_count=invariant.domain_count,
            substrate_count=invariant.substrate_count,
            hold_rate=invariant.invariance_hold_rate,
            description_length_bits=invariant.description_length_bits,
            intelligence_ratio_contribution=invariant.intelligence_ratio_contribution,
            applicable_domains=[d.domain for d in invariant.applicable_domains],
            untested_domains=invariant.untested_domains,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_TIER3_INVARIANT_DISCOVERED,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    async def _emit_counter_invariant_found(
        self, invariant: CausalInvariant, refined: Any
    ) -> None:
        """Broadcast KAIROS_COUNTER_INVARIANT_FOUND."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = CounterInvariantPayload(
            invariant_id=invariant.id,
            violation_count=invariant.violation_count,
            boundary_condition=refined.boundary_condition,
            excluded_feature=refined.excluded_feature,
            original_hold_rate=refined.original_hold_rate,
            refined_hold_rate=refined.refined_hold_rate,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_COUNTER_INVARIANT_FOUND,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    async def _emit_intelligence_ratio_step_change(
        self,
        invariant: CausalInvariant,
        old_ratio: float,
        new_ratio: float,
        delta: float,
        cause: str,
    ) -> None:
        """Broadcast KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = IntelligenceRatioStepChangePayload(
            invariant_id=invariant.id,
            old_ratio=old_ratio,
            new_ratio=new_ratio,
            delta=delta,
            cause=cause,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    # --- Health (ManagedSystemProtocol) ---

    async def health(self) -> dict[str, Any]:
        """Return health status for Synapse monitoring."""
        return {
            "status": "healthy",
            "pipeline_runs": self._pipeline_runs,
            "fovea_events_received": self._fovea_events_received,
            "evo_events_received": self._evo_events_received,
            "invariants_created": self._invariants_created,
            "tier3_discoveries": self._tier3_discoveries,
            "counter_invariants_found": self._counter_invariants_found,
            "step_changes": self._step_changes,
            "hierarchy": self._hierarchy.summary(),
            "intelligence_ledger": self._intelligence_ledger.summary(),
            "correlation_miner": {
                "pairs_evaluated": self._correlation_miner.total_pairs_evaluated,
                "candidates_found": self._correlation_miner.total_candidates_found,
            },
            "direction_tester": {
                "tests_run": self._direction_tester.total_tests_run,
                "accepted": self._direction_tester.total_accepted,
            },
            "confounder_analyzer": {
                "analyses_run": self._confounder_analyzer.total_analyses_run,
                "confounders_found": self._confounder_analyzer.total_confounders_found,
            },
            "distiller": {
                "distillations_run": self._distiller.total_distillations_run,
                "tautologies_rejected": self._distiller.total_tautologies_rejected,
                "domains_mapped": self._distiller.total_domains_mapped,
            },
            "counter_detector": {
                "scans_run": self._counter_detector.total_scans_run,
                "violations_found": self._counter_detector.total_violations_found,
                "refinements_made": self._counter_detector.total_refinements_made,
            },
        }

    @property
    def hierarchy(self) -> CausalHierarchy:
        return self._hierarchy

    @property
    def distiller(self) -> InvariantDistiller:
        return self._distiller

    @property
    def counter_detector(self) -> CounterInvariantDetector:
        return self._counter_detector

    @property
    def intelligence_ledger(self) -> IntelligenceContributionLedger:
        return self._intelligence_ledger
