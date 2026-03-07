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
  KAIROS_COUNTER_INVARIANT_FOUND, KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE,
  KAIROS_VALIDATED_CAUSAL_STRUCTURE, KAIROS_SPURIOUS_HYPOTHESIS_CLASS,
  KAIROS_INVARIANT_ABSORPTION_REQUESTED, KAIROS_CAUSAL_NOVELTY_DETECTED,
  KAIROS_HEALTH_DEGRADED, KAIROS_VIOLATION_ESCALATION

Subscriptions:
  FOVEA_PREDICTION_ERROR, EVOLUTION_CANDIDATE, EPISODE_STORED,
  CROSS_DOMAIN_MATCH_FOUND, WORLD_MODEL_UPDATED (Logos bidirectional),
  COMPRESSION_BACKLOG_PROCESSED (Oneiros consolidation feedback)

Integrations: Logos (world model), Nexus (federation sharing of Tier 3),
  Telos (intelligence ratio step change signal), Evo (Thompson sampler feedback),
  Fovea (invariant absorption), Thymos (health degradation incidents).
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
    CausalDirection,
    CausalDirectionPayload,
    CausalDirectionResult,
    CausalInvariant,
    CausalNoveltyPayload,
    CausalRule,
    CausalStructurePattern,
    ConfounderDiscoveredPayload,
    CorrelationCandidate,
    CounterInvariantPayload,
    HealthDegradedPayload,
    IntelligenceRatioStepChangePayload,
    InvariantAbsorptionPayload,
    InvarianceVerdict,
    InvariantCandidatePayload,
    InvariantDistilledPayload,
    KairosConfig,
    KairosHealthStatus,
    SpuriousHypothesisClassPayload,
    Tier3InvariantPayload,
    ValidatedCausalStructurePayload,
    ViolationEscalationPayload,
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

        # Parse actual cause/effect variables from the hypothesis statement.
        # Try structured fields first, then fall back to keyword splitting.
        cause_var = event_data.get("cause_variable", "")
        effect_var = event_data.get("effect_variable", "")

        if not cause_var or not effect_var:
            cause_var, effect_var = self._parse_causal_statement(statement)

        if not cause_var or not effect_var:
            logger.debug(
                "evo_causal_claim_unparseable",
                hypothesis_id=hypothesis_id,
                statement=statement[:100],
            )
            return None

        candidate = CorrelationCandidate(
            variable_a=cause_var,
            variable_b=effect_var,
            mean_correlation=confidence,
            cross_context_variance=0.0,
            context_count=1,
        )

        self._seeded_candidates.append(candidate)

        logger.info(
            "evo_causal_claim_seeded",
            hypothesis_id=hypothesis_id,
            cause=cause_var,
            effect=effect_var,
            confidence=confidence,
        )

        return candidate

    @staticmethod
    def _parse_causal_statement(statement: str) -> tuple[str, str]:
        """
        Extract cause and effect variable names from a natural-language
        causal claim like "X causes Y" or "X leads to Y".

        Returns (cause, effect) or ("", "") if unparseable.
        """
        statement_lower = statement.lower()
        # Try each causal keyword as a split point
        causal_phrases = [
            " causes ", " leads to ", " produces ", " results in ",
            " drives ", " triggers ", " induces ", " generates ",
            " increases ", " decreases ", " affects ",
        ]
        for phrase in causal_phrases:
            if phrase in statement_lower:
                idx = statement_lower.index(phrase)
                cause = statement[:idx].strip()
                effect = statement[idx + len(phrase):].strip()
                # Clean: take last word cluster before the keyword as cause,
                # first word cluster after as effect
                cause = cause.split(",")[-1].strip()
                effect = effect.split(",")[0].strip().rstrip(".")
                if cause and effect:
                    return cause, effect
        return "", ""

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
        self._memory: Any = None  # Memory system for observation queries
        self._neo4j: Any = None  # Neo4j client for persistence

        # Health monitoring
        self._health_status = KairosHealthStatus()
        self._discovered_patterns: list[CausalStructurePattern] = []

        # Metrics
        self._pipeline_runs: int = 0
        self._fovea_events_received: int = 0
        self._evo_events_received: int = 0
        self._cross_domain_events_received: int = 0
        self._invariants_created: int = 0
        self._tier3_discoveries: int = 0
        self._deferred_tier3: list[CausalInvariant] = []
        self._tier3_demotions: int = 0
        self._counter_invariants_found: int = 0
        self._step_changes: int = 0
        # P6: per-invariant I-ratio from previous pipeline run for cross-run step detection
        self._prev_i_ratios: dict[str, float] = {}
        self._confounders_found: int = 0
        self._logos_updates_received: int = 0
        self._oneiros_consolidations_received: int = 0

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

    def set_memory(self, memory: Any) -> None:
        """Wire the Memory system for observation queries."""
        self._memory = memory
        logger.info("kairos_memory_wired")

    def set_neo4j(self, neo4j: Any) -> None:
        """Wire the Neo4j client for invariant persistence."""
        self._neo4j = neo4j
        logger.info("kairos_neo4j_wired")

    async def initialize(self) -> None:
        """
        Initialize Kairos — restore invariants from Neo4j.

        Must complete before the first pipeline cycle so the hierarchy
        is populated with previously discovered invariants.
        """
        if self._neo4j is None:
            logger.warning("kairos_initialize_skipped", reason="no_neo4j_client")
            return

        from systems.kairos.persistence import ensure_schema, restore_invariants

        await ensure_schema(self._neo4j)

        invariants = await restore_invariants(self._neo4j)
        for inv in invariants:
            self._hierarchy._place(inv)

        logger.info(
            "kairos_initialized",
            invariants_restored=len(invariants),
            hierarchy_total=self._hierarchy.total_count,
        )

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

        # P1: Logos bidirectional — subscribe to world model updates
        self._event_bus.subscribe(
            SynapseEventType.WORLD_MODEL_UPDATED,
            self._on_world_model_updated,
        )

        # P1: Oneiros consolidation feedback
        self._event_bus.subscribe(
            SynapseEventType.COMPRESSION_BACKLOG_PROCESSED,
            self._on_oneiros_consolidation,
        )

        # SG3: Federation invariant reception
        self._event_bus.subscribe(
            SynapseEventType.FEDERATION_INVARIANT_RECEIVED,
            self._on_federation_invariant_received,
        )

        logger.info("kairos_subscriptions_registered", events=7)

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

    async def _on_world_model_updated(self, event: SynapseEvent) -> None:
        """
        P1: Logos bidirectional feedback.

        When Logos reorganizes the world model, Kairos should re-evaluate
        existing invariants against the new model structure. Non-Kairos
        updates (from other systems) may reveal new contexts for testing.
        """
        data = event.data if hasattr(event, "data") else {}
        source = data.get("source", "")

        # Skip our own updates to avoid feedback loops
        if source == "kairos":
            return

        self._logos_updates_received += 1

        update_type = data.get("update_type", "")
        schemas_added = data.get("schemas_added", 0)

        logger.debug(
            "logos_world_model_update_received",
            update_type=update_type,
            source=source,
            schemas_added=schemas_added,
        )

        # If schemas were added, they may define new domains for invariant testing
        if schemas_added > 0:
            # Re-evaluate undistilled invariants against the expanded domain set
            for invariant in self._hierarchy.get_all():
                if not invariant.distilled:
                    self._hierarchy.promote_if_eligible(invariant.id)

    async def _on_oneiros_consolidation(self, event: SynapseEvent) -> None:
        """
        P1: Oneiros consolidation feedback.

        When Oneiros finishes a consolidation cycle, it may have discovered
        cross-domain patterns that Kairos should mine. Pre-seed the
        correlation miner with any causal structures from consolidation.
        """
        data = event.data if hasattr(event, "data") else {}
        self._oneiros_consolidations_received += 1

        # Extract cross-domain patterns from consolidation output
        patterns = data.get("cross_domain_patterns", [])
        for pattern in patterns:
            cause = pattern.get("cause", "")
            effect = pattern.get("effect", "")
            confidence = float(pattern.get("confidence", 0.5))
            if cause and effect:
                candidate = CorrelationCandidate(
                    variable_a=f"oneiros_consolidation:{cause}",
                    variable_b=f"oneiros_consolidation:{effect}",
                    mean_correlation=confidence,
                    cross_context_variance=0.0,
                    context_count=2,
                )
                self._correlation_miner.add_preseed(candidate)

        logger.debug(
            "oneiros_consolidation_processed",
            patterns_received=len(patterns),
        )

    async def _on_federation_invariant_received(self, event: SynapseEvent) -> None:
        """
        SG3: Handle invariants received from federated peer instances.

        Validate against local observations via counter-invariant testing.
        If validated: merge into CausalHierarchy at appropriate tier.
        If contradicted: emit KAIROS_INVARIANT_CONTRADICTED.
        """
        data = event.data if hasattr(event, "data") else {}
        source_instance = data.get("source_instance_id", "unknown")
        abstract_form = data.get("abstract_form", "")
        tier_val = int(data.get("tier", 1))
        hold_rate = float(data.get("hold_rate", 0.0))
        domains = data.get("domains", [])

        if not abstract_form:
            return

        # Build a CausalInvariant from the federation payload
        from primitives.causal import ApplicableDomain, CausalInvariantTier

        fed_invariant = CausalInvariant(
            tier=CausalInvariantTier(tier_val),
            abstract_form=abstract_form,
            invariance_hold_rate=hold_rate,
            applicable_domains=[
                ApplicableDomain(domain=d, hold_rate=hold_rate) for d in domains
            ],
        )

        # Validate against local observations
        local_observations = await self.query_observations_for_testing(
            fed_invariant.id,
        )
        if not local_observations:
            # No local data to validate — accept at trust but don't promote
            self._hierarchy._place(fed_invariant)
            logger.info(
                "federation_invariant_accepted_unvalidated",
                invariant_id=fed_invariant.id,
                source=source_instance,
            )
            return

        # Run counter-invariant testing
        violations = await self._counter_detector.scan_for_violations(
            fed_invariant, local_observations
        )

        if not violations:
            # Validated locally — merge into hierarchy
            self._hierarchy._place(fed_invariant)
            self._hierarchy.promote_if_eligible(fed_invariant.id)
            fed_invariant.validated = True
            logger.info(
                "federation_invariant_validated",
                invariant_id=fed_invariant.id,
                source=source_instance,
                tier=fed_invariant.tier.value,
            )
        else:
            # Contradicted locally — emit contradiction event
            if self._event_bus is not None:
                from systems.synapse.types import SynapseEvent as SE
                from systems.synapse.types import SynapseEventType

                await self._event_bus.emit(
                    SE(
                        event_type=SynapseEventType.KAIROS_INVARIANT_CONTRADICTED,
                        source_system="kairos",
                        data={
                            "invariant_id": fed_invariant.id,
                            "abstract_form": abstract_form,
                            "local_hold_rate": 1.0 - (
                                len(violations) / max(len(local_observations), 1)
                            ),
                            "violation_count": len(violations),
                            "source_instance_id": source_instance,
                        },
                    )
                )
            logger.info(
                "federation_invariant_contradicted",
                invariant_id=fed_invariant.id,
                source=source_instance,
                violations=len(violations),
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

        # Drain any Tier 3 promotions deferred from sync callbacks
        await self._drain_deferred_tier3()

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
                self._confounders_found += 1
                # P0: Evo penalty signal — this hypothesis class is spurious
                await self._emit_spurious_hypothesis_class(confounder_result)
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
                # Populate causal direction from the original correlation sign
                invariant.direction = (
                    "negative"
                    if direction_result.candidate.mean_correlation < 0
                    else "positive"
                )
                self._invariants_created += 1
                invariants_created += 1
                new_invariants.append(invariant)
                await self._emit_invariant_candidate(
                    invariant, rule, invariance_result,
                )
                await self._ingest_to_logos(invariant, rule)

                # P0: Evo validation signal — this causal structure is confirmed
                await self._emit_validated_causal_structure(invariant, rule)
                # P0: Fovea absorption — request world model integration
                await self._emit_invariant_absorption_requested(invariant)
                # P2: Novelty detection
                await self._detect_and_emit_novelty(invariant, direction_result)
                # SG4: Emit RE training example for validated causal chain (Stream 4)
                await self._emit_re_training_example(invariant, rule)

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

        # M10: order Phase D by ledger value so highest-value invariants are scanned first
        ranked_contributions = self._intelligence_ledger.rank_by_value()
        ranked_ids = [c.invariant_id for c in ranked_contributions]
        ranked_id_set = set(ranked_ids)
        all_phase_d_invariants = self._hierarchy.get_all()
        # Higher-ranked first, then any not yet in ledger in original order
        phase_d_invariants = sorted(
            all_phase_d_invariants,
            key=lambda inv: ranked_ids.index(inv.id) if inv.id in ranked_id_set else len(ranked_ids),
        )

        for invariant in phase_d_invariants:
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

        # ── Violation Escalation (P2: Thymos) ──────────────────────
        for invariant in self._hierarchy.get_all():
            if invariant.violation_count >= self._config.min_violations_for_cluster * 2:
                await self._emit_violation_escalation(invariant)

        # ── Phase D: Intelligence Contribution Ledger ────────────
        all_invariants = self._hierarchy.get_all()
        if all_invariants:
            contributions = self._intelligence_ledger.compute_all(
                all_invariants, observations_by_context, total_model_length
            )

            # P6: Check for step changes using previous pipeline run's ratios
            next_i_ratios: dict[str, float] = {}
            for contribution in contributions:
                inv = self._hierarchy.find_invariant(contribution.invariant_id)
                if inv is not None:
                    # P6: compare against last run's ratio (not within-run counterfactual)
                    old_ratio = self._prev_i_ratios.get(
                        contribution.invariant_id, contribution.intelligence_ratio_without
                    )
                    is_step, delta = self._intelligence_ledger.detect_step_change(
                        inv, old_ratio
                    )
                    if is_step:
                        self._step_changes += 1
                        await self._emit_intelligence_ratio_step_change(
                            inv, old_ratio, contribution.intelligence_ratio_contribution, delta,
                            "pipeline_run",
                        )
                    # D4: emit counterfactual step-change signal when value is non-trivial
                    counterfactual = contribution.intelligence_ratio_without
                    if counterfactual > 0.1:
                        await self._emit_intelligence_ratio_step_change(
                            inv, counterfactual, contribution.intelligence_ratio_contribution,
                            contribution.intelligence_ratio_contribution - counterfactual,
                            "counterfactual_removal",
                        )
                    next_i_ratios[contribution.invariant_id] = contribution.intelligence_ratio_contribution
            # P6: persist current ratios for next run
            self._prev_i_ratios = next_i_ratios

        # ── Invariant Decay (SG2) ─────────────────────────────────
        reinforced_ids = {inv.id for inv in new_invariants}
        await self._apply_invariant_decay(reinforced_ids)

        # ── Health Monitoring (P1) ─────────────────────────────────
        await self._diagnose_health(
            candidates_count=len(candidates),
            invariants_created=invariants_created,
            violations_found=violations_found,
        )

        # ── Evolutionary Metrics (SG5) ─────────────────────────────
        await self._emit_evolutionary_metrics()

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

        # ── Persist to Neo4j (batched) ─────────────────────────────
        await self._persist_all_invariants()

        logger.info("pipeline_run_complete", **summary)
        return summary

    async def _persist_all_invariants(self) -> None:
        """Batch-persist all invariants to Neo4j."""
        if self._neo4j is None:
            return

        from systems.kairos.persistence import persist_invariants_batch

        all_invariants = self._hierarchy.get_all()
        if all_invariants:
            await persist_invariants_batch(self._neo4j, all_invariants)

    async def _apply_invariant_decay(self, reinforced_ids: set[str]) -> None:
        """
        SG2: Decay invariants not reinforced this cycle.

        - Reinforced invariants reset recency_weight to 1.0
        - Others decay by 0.95×
        - Demote from tier when recency_weight < 0.3
        - Archive (active=false) when recency_weight < 0.1
        """
        archived = 0
        demoted = 0

        for invariant in list(self._hierarchy.get_all()):
            if invariant.id in reinforced_ids:
                invariant.recency_weight = 1.0
                continue

            invariant.recency_weight *= 0.95

            if invariant.recency_weight < 0.1:
                # Archive — remove from hierarchy
                invariant.active = False
                self._hierarchy._remove(invariant)
                archived += 1
            elif invariant.recency_weight < 0.3:
                # Demote one tier
                old_tier = invariant.tier
                from primitives.causal import CausalInvariantTier

                if invariant.tier == CausalInvariantTier.TIER_3_SUBSTRATE:
                    self._hierarchy._remove(invariant)
                    invariant.tier = CausalInvariantTier.TIER_2_CROSS_DOMAIN
                    self._hierarchy._place(invariant)
                    self._tier3_demotions += 1
                    demoted += 1
                elif invariant.tier == CausalInvariantTier.TIER_2_CROSS_DOMAIN:
                    self._hierarchy._remove(invariant)
                    invariant.tier = CausalInvariantTier.TIER_1_DOMAIN
                    self._hierarchy._place(invariant)
                    demoted += 1

        if archived or demoted:
            logger.info(
                "invariant_decay_applied",
                archived=archived,
                demoted=demoted,
                reinforced=len(reinforced_ids),
            )

    async def _emit_evolutionary_metrics(self) -> None:
        """
        SG5: Compute and emit evolutionary metrics as observables.

        Metrics: mean I-ratio, Tier 3 discovery rate, invariant overlap coefficient.
        """
        if self._event_bus is None:
            return

        all_invariants = self._hierarchy.get_all()
        if not all_invariants:
            return

        # Mean I-ratio across all invariants
        mean_i_ratio = sum(
            inv.intelligence_ratio_contribution for inv in all_invariants
        ) / len(all_invariants)

        # Tier 3 discovery rate: tier3 discoveries / total pipeline runs
        tier3_rate = (
            self._tier3_discoveries / max(self._pipeline_runs, 1)
        )

        # Invariant overlap coefficient: fraction of invariants sharing domains
        all_domains: list[set[str]] = [
            {d.domain for d in inv.applicable_domains} for inv in all_invariants
        ]
        overlap_pairs = 0
        total_pairs = 0
        for i in range(len(all_domains)):
            for j in range(i + 1, len(all_domains)):
                total_pairs += 1
                if all_domains[i] & all_domains[j]:
                    overlap_pairs += 1
        overlap_coeff = overlap_pairs / max(total_pairs, 1)

        await self._emit_evolutionary_observable(
            "kairos_mean_i_ratio", mean_i_ratio, is_novel=False,
            metadata={"invariant_count": len(all_invariants)},
        )
        await self._emit_evolutionary_observable(
            "kairos_tier3_discovery_rate", tier3_rate,
            is_novel=self._tier3_discoveries > 0,
            metadata={"total_tier3": self._tier3_discoveries},
        )
        await self._emit_evolutionary_observable(
            "kairos_invariant_overlap_coefficient", overlap_coeff,
            is_novel=False,
            metadata={"total_pairs": total_pairs, "overlapping": overlap_pairs},
        )

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an evolutionary observable event for Benchmarks population tracking."""
        if self._event_bus is None:
            return
        try:
            from primitives.common import SystemID
            from primitives.evolutionary import EvolutionaryObservable
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.KAIROS,
                instance_id="",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                    source_system="kairos",
                    data=obs.model_dump(mode="json"),
                )
            )
        except Exception:
            logger.debug("evolutionary_observable_emission_failed", type=observable_type)

    # --- Helpers ---

    @staticmethod
    def _extract_observation_pairs(
        candidate: CorrelationCandidate,
        observations_by_context: dict[str, list[dict[str, Any]]],
    ) -> list[tuple[float, float]] | None:
        """Extract paired (a, b) observations for the ANM test, pooling across all contexts."""
        pairs: list[tuple[float, float]] = []
        for obs_list in observations_by_context.values():
            for obs in obs_list:
                a = obs.get(candidate.variable_a)
                b = obs.get(candidate.variable_b)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    pairs.append((float(a), float(b)))
        return pairs if len(pairs) >= 10 else None

    # --- Logos integration ---

    async def _ingest_to_logos(self, invariant: Any, rule: CausalRule) -> None:
        """Feed a confirmed invariant into the Logos world model."""
        if self._logos_ingest is None:
            return

        # Use dict payload to avoid cross-system import of EmpiricalInvariant.
        # The Logos ingest protocol accepts either EmpiricalInvariant or a dict.
        logos_payload = {
            "statement": invariant.abstract_form,
            "domain": rule.domain or "general",
            "observation_count": rule.observation_count,
            "confidence": invariant.invariance_hold_rate,
            "source": "kairos",
        }

        if hasattr(self._logos_ingest, "ingest_invariant_dict"):
            self._logos_ingest.ingest_invariant_dict(logos_payload)
        else:
            self._logos_ingest.ingest_invariant(logos_payload)

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

        # Schedule async cascade from sync callback.
        # asyncio.ensure_future works when a loop is running; if not, we
        # queue the coroutine for the next pipeline run to drain.
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._handle_tier3_async(invariant))
            task.add_done_callback(self._tier3_task_done)
        except RuntimeError:
            # No running loop — queue for deferred execution
            logger.warning(
                "tier3_deferred",
                invariant_id=invariant.id,
                reason="no_running_event_loop",
            )
            self._deferred_tier3.append(invariant)

    def _tier3_task_done(self, task: object) -> None:
        """Log errors from Tier 3 async tasks instead of silently dropping them."""
        import asyncio

        t = task if isinstance(task, asyncio.Task) else None
        if t is not None and t.done() and not t.cancelled():
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "tier3_async_cascade_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    async def _drain_deferred_tier3(self) -> None:
        """Drain any Tier 3 promotions that were queued when no loop was running."""
        if not self._deferred_tier3:
            return
        deferred = self._deferred_tier3[:]
        self._deferred_tier3.clear()
        for inv in deferred:
            logger.info("tier3_deferred_drain", invariant_id=inv.id)
            await self._handle_tier3_async(inv)

    async def _handle_tier3_async(self, invariant: CausalInvariant) -> None:
        """
        Async handler for Tier 3 promotion — full 7-response cascade:
        1. Broadcast TIER3_INVARIANT_DISCOVERED (existing)
        2. Share with Nexus for federation (existing)
        3. Inject Oneiros REM seed (existing)
        4. Equor: constitutional review before acceptance (M4)
        5. Thread: narrative milestone log (M5)
        6. Nova: policy update notification (M6)
        7. Logos: deep structural reorganization (M7)
        """
        await self._emit_tier3_discovered(invariant)
        await self._share_with_nexus(invariant)
        await self._inject_oneiros_rem_seed(invariant)
        await self._request_equor_review(invariant)
        await self._emit_thread_milestone(invariant)
        await self._notify_nova_policy_update(invariant)
        await self._logos_tier3_structural_reorganize(invariant)

    async def _logos_tier3_structural_reorganize(self, invariant: CausalInvariant) -> None:
        """
        M7: Signal Logos to reorganize its world model around a Tier 3 substrate-independent
        invariant. This goes beyond the standard ingest — it requests deep structural
        reweighting of hypotheses that touch the same variables.
        """
        if self._logos_ingest is None:
            return

        logos_payload = {
            "statement": invariant.abstract_form,
            "domain": "substrate_independent",
            "observation_count": invariant.applicable_domains[0].observation_count
            if invariant.applicable_domains else 0,
            "confidence": invariant.invariance_hold_rate,
            "source": "kairos_tier3",
            "reorganize": True,
            "tier": invariant.tier.value if hasattr(invariant.tier, "value") else invariant.tier,
            "domain_count": invariant.domain_count,
            "substrate_count": invariant.substrate_count,
        }

        if hasattr(self._logos_ingest, "ingest_invariant_dict"):
            self._logos_ingest.ingest_invariant_dict(logos_payload)
        else:
            self._logos_ingest.ingest_invariant(logos_payload)

        logger.info(
            "logos_tier3_structural_reorganize_requested",
            invariant_id=invariant.id,
            abstract_form=invariant.abstract_form[:60],
            domain_count=invariant.domain_count,
        )

    async def _request_equor_review(self, invariant: CausalInvariant) -> None:
        """M4: Request Equor constitutional review before Tier 3 acceptance."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.CONSTITUTIONAL_REVIEW_REQUESTED,
                source_system="kairos",
                data={
                    "review_type": "tier3_invariant_acceptance",
                    "invariant_id": invariant.id,
                    "abstract_form": invariant.abstract_form,
                    "hold_rate": invariant.invariance_hold_rate,
                    "domain_count": invariant.domain_count,
                    "substrate_count": invariant.substrate_count,
                    "reason": (
                        "Tier 3 substrate-independent invariant discovered. "
                        "Constitutional review required before permanent acceptance "
                        "into deepest world model layer."
                    ),
                },
            )
        )

    async def _emit_thread_milestone(self, invariant: CausalInvariant) -> None:
        """M5: Log Tier 3 discovery as a narrative milestone in Thread."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.NARRATIVE_MILESTONE,
                source_system="kairos",
                data={
                    "milestone_type": "tier3_invariant_discovered",
                    "title": f"Substrate-independent invariant: {invariant.abstract_form[:60]}",
                    "description": (
                        f"Discovered a Tier 3 causal invariant spanning "
                        f"{invariant.domain_count} domains and "
                        f"{invariant.substrate_count} substrates with "
                        f"{invariant.invariance_hold_rate:.1%} hold rate."
                    ),
                    "significance": "high",
                    "invariant_id": invariant.id,
                    "domains": [d.domain for d in invariant.applicable_domains],
                },
            )
        )

    async def _notify_nova_policy_update(self, invariant: CausalInvariant) -> None:
        """M6: Notify Nova that a Tier 3 invariant should update active inference priors."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.WORLD_MODEL_UPDATED,
                source_system="kairos",
                data={
                    "update_type": "tier3_invariant_discovered",
                    "schemas_added": 0,
                    "priors_updated": 1,
                    "causal_updates": 1,
                    "invariant_id": invariant.id,
                    "invariant_tier": 3,
                    "abstract_form": invariant.abstract_form,
                    "hold_rate": invariant.invariance_hold_rate,
                    "source": "kairos",
                },
            )
        )

    # --- Nexus federation sharing ---

    async def _share_with_nexus(self, invariant: CausalInvariant) -> None:
        """Share a Tier 3 invariant with Nexus for immediate federation broadcast."""
        if self._nexus_share is None:
            return

        # Use dict payload to avoid cross-system import of ShareableWorldModelFragment.
        # The Nexus share protocol constructs the fragment internally.
        fragment_data = {
            "source_instance_id": "local",
            "abstract_structure": {
                "type": "causal_invariant",
                "tier": 3,
                "abstract_form": invariant.abstract_form,
                "hold_rate": invariant.invariance_hold_rate,
                "domains": [d.domain for d in invariant.applicable_domains],
                "substrates": list(
                    {d.substrate for d in invariant.applicable_domains if d.substrate}
                ),
            },
            "domain_labels": [d.domain for d in invariant.applicable_domains],
            "observations_explained": sum(
                d.observation_count for d in invariant.applicable_domains
            ),
            "description_length": invariant.description_length_bits,
            "compression_ratio": (
                invariant.intelligence_ratio_contribution
                if invariant.intelligence_ratio_contribution > 0
                else 1.0
            ),
        }

        try:
            if hasattr(self._nexus_share, "share_fragment_dict"):
                await self._nexus_share.share_fragment_dict(fragment_data)
            else:
                await self._nexus_share.share_fragment(fragment_data)
            logger.info(
                "tier3_shared_with_nexus",
                invariant_id=invariant.id,
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
            # M8: mark as priority so Oneiros processes this before other seeds
            "priority": True,
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

    # --- P0: Evo ↔ Kairos bidirectional feedback ---

    async def _emit_validated_causal_structure(
        self, invariant: CausalInvariant, rule: CausalRule,
    ) -> None:
        """P0: Emit VALIDATED_CAUSAL_STRUCTURE → Evo Thompson sampler reward."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = ValidatedCausalStructurePayload(
            invariant_id=invariant.id,
            cause=rule.cause_variable,
            effect=rule.effect_variable,
            hold_rate=invariant.invariance_hold_rate,
            tier=invariant.tier.value,
            domain_count=invariant.domain_count,
            hypothesis_pattern=f"{rule.cause_variable} causes {rule.effect_variable}",
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_VALIDATED_CAUSAL_STRUCTURE,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    async def _emit_spurious_hypothesis_class(self, confounder_result: Any) -> None:
        """P0: Emit SPURIOUS_HYPOTHESIS_CLASS → Evo Thompson sampler penalty."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = SpuriousHypothesisClassPayload(
            confounded_cause=confounder_result.original_pair.cause,
            confounded_effect=confounder_result.original_pair.effect,
            confounders=[c.variable for c in confounder_result.confounding_variables],
            mdl_improvement=confounder_result.mdl_improvement,
            hypothesis_class="confounded_correlation",
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_SPURIOUS_HYPOTHESIS_CLASS,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    # --- P0: Fovea ↔ Kairos bidirectional feedback ---

    async def _emit_invariant_absorption_requested(
        self, invariant: CausalInvariant,
    ) -> None:
        """P0: Emit INVARIANT_ABSORPTION_REQUESTED → Fovea world model update."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = InvariantAbsorptionPayload(
            invariant_id=invariant.id,
            cause=invariant.abstract_form.split(" causes ")[0].strip() if " causes " in invariant.abstract_form else "",
            effect=invariant.abstract_form.split(" causes ")[1].strip() if " causes " in invariant.abstract_form else "",
            hold_rate=invariant.invariance_hold_rate,
            tier=invariant.tier.value,
            abstract_form=invariant.abstract_form,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_INVARIANT_ABSORPTION_REQUESTED,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    # --- SG4: RE training data emission ---

    async def _emit_re_training_example(
        self, invariant: CausalInvariant, rule: CausalRule
    ) -> None:
        """
        SG4: Emit a RE training example for each validated causal chain (Stream 4).

        The prompt asks the model to predict whether a causal relationship holds;
        the signal is the confirmed causal direction and hold rate. This enriches
        the reasoning engine's causal reasoning capability over time.
        """
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        training_prompt = (
            f"Given observations of '{rule.cause_variable}' and '{rule.effect_variable}' "
            f"across {rule.observation_count} cases in domain '{rule.domain or 'general'}', "
            f"does '{rule.cause_variable}' causally {invariant.direction or 'influence'} "
            f"'{rule.effect_variable}'? Provide your reasoning."
        )
        training_signal = (
            f"Yes. Causal direction confidence: {rule.direction_confidence:.2f}. "
            f"Invariance hold rate: {invariant.invariance_hold_rate:.2f}. "
            f"Abstract form: {invariant.abstract_form}"
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="kairos",
                data={
                    "stream": 4,
                    "training_prompt": training_prompt,
                    "training_signal": training_signal,
                    "invariant_id": invariant.id,
                    "hold_rate": invariant.invariance_hold_rate,
                    "tier": invariant.tier.value if hasattr(invariant.tier, "value") else invariant.tier,
                },
            )
        )

    # --- P2: Novelty detection ---

    async def _detect_and_emit_novelty(
        self,
        invariant: CausalInvariant,
        direction_result: CausalDirectionResult,
    ) -> None:
        """
        P2: Detect novel causal structures and broadcast CAUSAL_NOVELTY_DETECTED.

        Novel structures: bidirectional causation, feedback loops, modulated
        relationships, causal chains across domains.
        """
        novelty_type = ""
        structure: dict[str, Any] = {}

        # Bidirectional causation is inherently novel
        if direction_result.direction == CausalDirection.BIDIRECTIONAL:
            novelty_type = "bidirectional"
            structure = {
                "variable_a": direction_result.candidate.variable_a,
                "variable_b": direction_result.candidate.variable_b,
            }

        # Cross-domain invariant with high hold rate is novel
        elif invariant.domain_count >= 2 and invariant.invariance_hold_rate >= 0.9:
            novelty_type = "causal_chain"
            structure = {
                "domains": [d.domain for d in invariant.applicable_domains],
                "hold_rate": invariant.invariance_hold_rate,
            }

        if not novelty_type:
            return

        # Track discovered pattern
        pattern = CausalStructurePattern(
            pattern_type=novelty_type,
            variables=[direction_result.candidate.variable_a, direction_result.candidate.variable_b],
            domain_count=invariant.domain_count,
            invariant_ids=[invariant.id],
            abstract_description=invariant.abstract_form,
        )
        self._discovered_patterns.append(pattern)
        # D5: cap list to prevent unbounded growth
        if len(self._discovered_patterns) > 200:
            self._discovered_patterns = self._discovered_patterns[-200:]

        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = CausalNoveltyPayload(
            invariant_id=invariant.id,
            novelty_type=novelty_type,
            structure=structure,
            domains=[d.domain for d in invariant.applicable_domains],
            abstract_form=invariant.abstract_form,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_CAUSAL_NOVELTY_DETECTED,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    # --- P2: Violation escalation → Thymos ---

    async def _emit_violation_escalation(self, invariant: CausalInvariant) -> None:
        """P2: Escalate repeated violations to Thymos as a health incident."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        hold_rate = invariant.invariance_hold_rate
        severity = "critical" if hold_rate < 0.5 else "high" if hold_rate < 0.75 else "medium"

        payload = ViolationEscalationPayload(
            invariant_id=invariant.id,
            violation_count=invariant.violation_count,
            violation_rate=1.0 - hold_rate,
            severity=severity,
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_VIOLATION_ESCALATION,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

    # --- P1: Health monitoring ---

    async def _diagnose_health(
        self,
        candidates_count: int,
        invariants_created: int,
        violations_found: int,
    ) -> None:
        """
        P1: Comprehensive health diagnosis with stall/corruption/drift detection.

        Checks:
        - Discovery stall: not enough new invariants per pipeline run
        - Confounder inflation: too many correlations turn out to be confounded
        - Causal surprise: violations are too frequent
        - Ledger drift: intelligence contributions are shifting rapidly
        """
        issues: list[str] = []
        overall = "healthy"

        # Discovery stall detection
        discovery_rate = invariants_created / max(candidates_count, 1)
        if candidates_count > 0 and discovery_rate < self._config.discovery_stall_threshold:
            issues.append(f"discovery_stall: rate={discovery_rate:.3f}")
            overall = "degraded"

        # Confounder inflation detection
        confounder_rate = self._confounders_found / max(self._pipeline_runs, 1)
        if confounder_rate > self._config.confounder_inflation_threshold:
            issues.append(f"confounder_inflation: rate={confounder_rate:.3f}")
            overall = "degraded"

        # Causal surprise (corruption) detection
        total_invariants = self._hierarchy.total_count
        surprise_rate = violations_found / max(total_invariants, 1)
        if surprise_rate > self._config.corruption_surprise_threshold:
            issues.append(f"corruption_suspected: surprise_rate={surprise_rate:.3f}")
            overall = "critical"

        # Ledger drift detection
        ledger_drift = self._intelligence_ledger.get_ledger_drift()
        if ledger_drift > 0.3:
            issues.append(f"ledger_drift: {ledger_drift:.3f}")
            if overall != "critical":
                overall = "degraded"

        self._health_status = KairosHealthStatus(
            overall=overall,
            discovery_rate=discovery_rate,
            tier3_demotion_rate=self._tier3_demotions / max(self._pipeline_runs, 1),
            confounder_rate=confounder_rate,
            causal_surprise_rate=surprise_rate,
            ledger_drift=ledger_drift,
            issues=issues,
        )

        if overall != "healthy":
            await self._emit_health_degraded(overall, issues)

    async def _emit_health_degraded(
        self, severity: str, issues: list[str],
    ) -> None:
        """Emit KAIROS_HEALTH_DEGRADED → Thymos incident."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        payload = HealthDegradedPayload(
            degradation_type="kairos_system_health",
            severity=severity,
            details={"issues": issues, "pipeline_runs": self._pipeline_runs},
            metrics={
                "discovery_rate": self._health_status.discovery_rate,
                "confounder_rate": self._health_status.confounder_rate,
                "surprise_rate": self._health_status.causal_surprise_rate,
                "ledger_drift": self._health_status.ledger_drift,
            },
        )

        await self._event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.KAIROS_HEALTH_DEGRADED,
                data=payload.model_dump(),
                source_system="kairos",
            )
        )

        logger.warning(
            "kairos_health_degraded",
            severity=severity,
            issues=issues,
        )

    # --- P1: Memory query API ---

    async def query_observations_for_testing(
        self, invariant_id: str, context_filter: str = "",
    ) -> dict[str, list[dict[str, Any]]]:
        """
        P1: Query Memory for observations relevant to testing an invariant.

        Returns observations_by_context suitable for pipeline stages.
        """
        if self._memory is None:
            logger.warning("memory_not_wired", method="query_observations_for_testing")
            return {}

        invariant = self._hierarchy._find(invariant_id)
        if invariant is None:
            return {}

        # Parse cause/effect from abstract form
        parts = invariant.abstract_form.split(" causes ")
        if len(parts) != 2:
            return {}

        cause_var = parts[0].strip()
        effect_var = parts[1].strip()

        # Query Memory for episodes containing both variables
        if hasattr(self._memory, "query_episodes"):
            episodes = await self._memory.query_episodes(
                variables=[cause_var, effect_var],
                context_filter=context_filter,
                limit=100,
            )
            # Convert to observations_by_context format
            result: dict[str, list[dict[str, Any]]] = {}
            for ep in episodes:
                ctx = ep.get("context_id", "default")
                obs = ep.get("observations", [])
                result.setdefault(ctx, []).extend(obs)
            return result

        return {}

    # --- Health (ManagedSystemProtocol) ---

    async def health(self) -> dict[str, Any]:
        """Return health status for Synapse monitoring."""
        return {
            "status": self._health_status.overall,
            "health_details": self._health_status.model_dump(),
            "pipeline_runs": self._pipeline_runs,
            "fovea_events_received": self._fovea_events_received,
            "evo_events_received": self._evo_events_received,
            "invariants_created": self._invariants_created,
            "tier3_discoveries": self._tier3_discoveries,
            "counter_invariants_found": self._counter_invariants_found,
            "step_changes": self._step_changes,
            "discovered_patterns": len(self._discovered_patterns),
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
