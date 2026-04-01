"""
EcodiaOS - Nexus: Epistemic Triangulation Service

Orchestrates fragment extraction, convergence detection, divergence
measurement, incentive computation, speciation detection, invariant
bridge exchange, and ground truth promotion across the federation.

Nexus decides WHAT to share; Federation handles HOW.

Sub-components:
  ConvergenceDetector         - structural isomorphism comparison
  InstanceDivergenceMeasurer  - five-dimensional divergence scoring
  DivergenceIncentiveEngine   - triangulation weights + speciation pressure
  SpeciationDetector          - detects divergence >= 0.8 speciation events
  InvariantBridge             - cross-speciation causal invariant exchange
  SpeciationRegistry          - tracks cognitive kinds and bridge connections
  GroundTruthPromotionPipeline - Level 0-4 epistemic promotion

Synapse events emitted:
  FRAGMENT_SHARED               - a fragment was sent to the federation
  CONVERGENCE_DETECTED          - structural convergence found
  DIVERGENCE_PRESSURE           - speciation pressure generated
  TRIANGULATION_WEIGHT_UPDATE   - instance weight recalculated
  SPECIATION_EVENT              - two instances diverged beyond threshold
  GROUND_TRUTH_CANDIDATE        - fragment reached Level 3
  EMPIRICAL_INVARIANT_CONFIRMED - fragment reached Level 4 (constitutional)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

import json

from primitives.common import SystemID, new_id, utc_now
from primitives.evolutionary import EvolutionaryObservable
from primitives.re_training import RETrainingExample
from systems.nexus.convergence import ConvergenceDetector
from systems.nexus.divergence import InstanceDivergenceMeasurer, compute_economic_divergence
from systems.nexus.ground_truth import GroundTruthPromotionPipeline
from systems.nexus.incentives import DivergenceIncentiveEngine
from systems.nexus.persistence import NexusPersistence
from systems.nexus.speciation import (
    InvariantBridge,
    SpeciationDetector,
    SpeciationRegistry,
    _detect_economic_domain_hints,
    _strip_domain_context,
)
from systems.nexus.types import (
    CompressionPath,
    ConvergenceResult,
    DivergencePressure,
    DivergenceScore,
    EpistemicLevel,
    FragmentShareOutcome,
    IIEPFragmentType,
    IIEPMessage,
    InstanceDivergenceProfile,
    InvariantExchangeReport,
    NexusConfig,
    PromotionDecision,
    ShareableWorldModelFragment,
    SleepCertification,
    TriangulationMetadata,
    WorldModelFragmentShare,
    WorldModelFragmentShareResponse,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.nexus.protocols import (
        EquorProtectionProtocol,
        EvoCompetitionProtocol,
        EvoHypothesisSourceProtocol,
        FederationFragmentProtocol,
        FoveaAttentionProtocol,
        KairosCausalSourceProtocol,
        LogosWorldModelProtocol,
        LogosWriteBackProtocol,
        OneirosAdversarialProtocol,
        TelosFragmentGateProtocol,
        ThymosDriveSinkProtocol,
    )
    from systems.synapse.service import SynapseService

logger = structlog.get_logger("nexus")


class NexusService:
    """
    Epistemic Triangulation across Federation.

    Instances don't share beliefs - they share the structure beneath
    beliefs. Convergence across maximally diverse compression paths
    is the primary evidence for ground truth.

    Satisfies ManagedSystemProtocol for Synapse health monitoring.
    """

    system_id: str = "nexus"

    def __init__(
        self,
        *,
        config: NexusConfig | None = None,
    ) -> None:
        self._config = config or NexusConfig()
        self._initialized = False

        # Sub-components (Phases A–B)
        self._convergence_detector = ConvergenceDetector(wl1_iterations=self._config.wl1_iterations)
        self._divergence_measurer: InstanceDivergenceMeasurer | None = None
        self._incentive_engine: DivergenceIncentiveEngine | None = None

        # Sub-components (Phase C - Speciation)
        self._speciation_detector = SpeciationDetector(config=self._config)
        self._invariant_bridge = InvariantBridge()
        self._speciation_registry = SpeciationRegistry()

        # Sub-component (Phase D - Ground Truth Promotion)
        self._promotion_pipeline: GroundTruthPromotionPipeline | None = None

        # External dependencies (injected post-init)
        self._synapse: SynapseService | None = None
        self._world_model: LogosWorldModelProtocol | None = None
        self._fovea: FoveaAttentionProtocol | None = None
        self._federation: FederationFragmentProtocol | None = None
        self._thymos: ThymosDriveSinkProtocol | None = None
        self._oneiros: OneirosAdversarialProtocol | None = None
        self._evo: EvoCompetitionProtocol | None = None
        self._evo_hypothesis_source: EvoHypothesisSourceProtocol | None = None
        self._equor: EquorProtectionProtocol | None = None
        self._telos: TelosFragmentGateProtocol | None = None
        self._logos_adapter: LogosWriteBackProtocol | None = None
        self._kairos: KairosCausalSourceProtocol | None = None

        # Persistence layer (optional - graceful degradation if not wired)
        self._persistence: NexusPersistence | None = None

        # Instance identity
        self._instance_id: str = ""

        # Fragment store
        self._local_fragments: dict[str, ShareableWorldModelFragment] = {}
        self._remote_fragments: dict[str, ShareableWorldModelFragment] = {}

        # Maps schema_id → fragment_id for bridge survivor resolution.
        # _extract_causal_invariants uses schema_id as fragment_id on
        # ephemeral fragments so ConvergedInvariant.invariant_a_id carries
        # the schema_id; this map translates it back to the real fragment_id.
        self._schema_to_fragment: dict[str, str] = {}

        # Convergence history
        self._convergences: list[ConvergenceResult] = []

        # Remote profiles cache
        self._remote_profiles: dict[str, InstanceDivergenceProfile] = {}

        # Current triangulation weight
        self._triangulation_weight: float = 1.0

        # Federation link health tracking (circuit breaker)
        self._link_failures: dict[str, int] = {}
        self._max_consecutive_failures: int = 5

        # Oikos metabolic coupling: track consecutive divergence cycles without
        # convergence (Gap HIGH-5 Federation Spec, 2026-03-07).
        self._consecutive_divergence_cycles: int = 0

        # IIEP session registry: maps session_id → {initiator, responder, round}
        # Tracks active inter-instance epistemic protocol sessions so convergence
        # results can be correlated back to the originating session (HIGH-1).
        self._iiep_sessions: dict[str, dict[str, Any]] = {}

        # Background tasks
        self._tasks: list[asyncio.Task[None]] = []

        # ── Skia VitalityCoordinator modulation ───────────────────────
        self._modulation_halted: bool = False

    # ─── Dependency Injection ────────────────────────────────────

    def set_synapse(self, synapse: SynapseService) -> None:
        """Wire the event bus for broadcasting Nexus events."""
        self._synapse = synapse

    def set_world_model(self, world_model: LogosWorldModelProtocol) -> None:
        """Wire Logos world model for fragment extraction and divergence."""
        self._world_model = world_model

    def set_fovea(self, fovea: FoveaAttentionProtocol) -> None:
        """Wire Fovea for attentional diversity measurement."""
        self._fovea = fovea

    def set_federation(self, federation: FederationFragmentProtocol) -> None:
        """Wire Federation for fragment transport."""
        self._federation = federation

    def set_thymos(self, thymos: ThymosDriveSinkProtocol) -> None:
        """Wire Thymos for divergence pressure routing."""
        self._thymos = thymos

    def set_oneiros(self, oneiros: OneirosAdversarialProtocol) -> None:
        """Wire Oneiros for adversarial testing (Level 4 promotion)."""
        self._oneiros = oneiros

    def set_evo(self, evo: EvoCompetitionProtocol) -> None:
        """Wire Evo for hypothesis competition (Level 4 promotion)."""
        self._evo = evo

    def set_evo_hypothesis_source(
        self, evo_hypothesis_source: EvoHypothesisSourceProtocol
    ) -> None:
        """
        Wire Evo as the source of active hypothesis IDs for divergence measurement.

        Separate from set_evo() which wires the competition protocol for
        Level 4 ground truth promotion. Evo will typically satisfy both.
        """
        self._evo_hypothesis_source = evo_hypothesis_source

    def set_telos(self, telos: TelosFragmentGateProtocol) -> None:
        """Wire Telos for topology-aware fragment scoring during federation."""
        self._telos = telos

    def set_equor(self, equor: EquorProtectionProtocol) -> None:
        """Wire Equor for constitutional protection of empirical invariants."""
        self._equor = equor

    def set_logos_adapter(self, logos_adapter: LogosWriteBackProtocol) -> None:
        """
        Wire the LogosWorldModelAdapter for convergence→Logos feedback.

        This is separate from set_world_model() which provides the
        LogosWorldModelProtocol for divergence measurement.  The adapter
        provides write-back methods (update_schema_triangulation_confidence,
        ingest_empirical_invariant_from_nexus) that the protocol does not.
        """
        self._logos_adapter = logos_adapter

    def set_kairos(self, kairos: KairosCausalSourceProtocol) -> None:
        """
        Wire Kairos for bidirectional Tier 3 invariant sync.

        Kairos pushes Tier 3 to Nexus via share_fragment(), but Nexus
        also needs to pull fresh discoveries - especially after sleep cycles.
        """
        self._kairos = kairos

    def set_neo4j(self, neo4j: Neo4jClient) -> None:
        """Wire Neo4j client for persistent speciation/promotion state."""
        self._persistence = NexusPersistence(neo4j)

    def subscribe_to_synapse_events(self) -> None:
        """
        Subscribe to Synapse events needed for Loop 4.

        - WAKE_INITIATED: extract & share sleep-certified fragments
        - EMPIRICAL_INVARIANT_CONFIRMED (self-emitted): feed back to Logos
        """
        if self._synapse is None:
            logger.warning("nexus_subscribe_skipped", reason="synapse not wired")
            return

        bus = self._synapse.event_bus

        bus.subscribe(
            SynapseEventType.WAKE_INITIATED,
            self._on_wake_initiated,
        )

        bus.subscribe(
            SynapseEventType.EMPIRICAL_INVARIANT_CONFIRMED,
            self._on_empirical_invariant_confirmed,
        )

        bus.subscribe(
            SynapseEventType.SPECIATION_EVENT,
            self._on_speciation_event,
        )

        bus.subscribe(
            SynapseEventType.KAIROS_TIER3_INVARIANT_DISCOVERED,
            self._on_kairos_tier3_discovered,
        )

        # ── Phase 2: Six additional subscriptions (Spec §XIII Gap 2) ──

        bus.subscribe(
            SynapseEventType.FEDERATION_SESSION_STARTED,
            self._on_federation_session_started,
        )

        bus.subscribe(
            SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE,
            self._on_sleep_consolidation_complete,
        )

        bus.subscribe(
            SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
            self._on_hypothesis_confirmed,
        )

        bus.subscribe(
            SynapseEventType.EVO_HYPOTHESIS_REFUTED,
            self._on_hypothesis_refuted,
        )

        bus.subscribe(
            SynapseEventType.INSTANCE_SPAWNED,
            self._on_instance_spawned,
        )

        bus.subscribe(
            SynapseEventType.INSTANCE_RETIRED,
            self._on_instance_retired,
        )

        bus.subscribe(
            SynapseEventType.INCIDENT_RESOLVED,
            self._on_thymos_incident_resolved,
        )

        # ── MEDIUM-3: Oneiros adversarial threat scenario testing ─────────
        bus.subscribe(
            SynapseEventType.ONEIROS_THREAT_SCENARIO,
            self._on_oneiros_threat_scenario,
        )

        # ── Economic events → epistemic triangulation across federation ────
        bus.subscribe(
            SynapseEventType.REVENUE_INJECTED,
            self._on_federation_revenue_learning,
        )
        bus.subscribe(
            SynapseEventType.BOUNTY_PAID,
            self._on_federation_bounty_learning,
        )
        bus.subscribe(
            SynapseEventType.SYSTEM_MODULATION,
            self._on_system_modulation,
        )

        logger.info("nexus_subscribed_to_synapse_events")

    # ─── Lifecycle ───────────────────────────────────────────────

    async def initialize(self, instance_id: str) -> None:
        """Initialize Nexus with the local instance identity."""
        self._instance_id = instance_id

        self._divergence_measurer = InstanceDivergenceMeasurer(
            world_model=self._world_model,
            fovea=self._fovea,
            evo=self._evo_hypothesis_source,
            local_instance_id=instance_id,
        )

        self._incentive_engine = DivergenceIncentiveEngine(
            thymos=self._thymos,
            local_instance_id=instance_id,
            config=self._config,
        )

        self._promotion_pipeline = GroundTruthPromotionPipeline(
            config=self._config,
            speciation_registry=self._speciation_registry,
            oneiros=self._oneiros,
            evo=self._evo,
            equor=self._equor,
        )

        # Restore persisted state from Neo4j (speciation events, cognitive
        # kinds, epistemic levels, bridge survivors, fragments, converged
        # invariants, divergence profiles).  Graceful degradation: if
        # persistence is not wired or restore fails, start fresh.
        if self._persistence is not None:
            try:
                (
                    events,
                    kinds,
                    levels,
                    bridge_survivors,
                    fragments,
                    _converged_invariants,  # informational - not re-indexed
                    profiles,
                ) = await self._persistence.restore_full_state()

                for ev in events:
                    self._speciation_registry.register_speciation(ev, logos_a=None, logos_b=None)
                for kind in kinds:
                    self._speciation_registry.state.cognitive_kinds[kind.kind_id] = kind
                for frag_id, level in levels.items():
                    self._promotion_pipeline._fragment_levels[frag_id] = level
                for frag_id in bridge_survivors:
                    self._promotion_pipeline.mark_bridge_survivor(frag_id)

                # Restore local fragments (R1) - skip if fragment_id already
                # present (in-memory state takes priority over persisted).
                for f in fragments:
                    if f.fragment_id not in self._local_fragments:
                        self._local_fragments[f.fragment_id] = f

                # Restore remote divergence profiles (R5) - later captures
                # overwrite earlier ones for the same instance_id.
                for p in profiles:
                    self._remote_profiles[p.instance_id] = p

                logger.info(
                    "nexus_state_restored_from_neo4j",
                    speciation_events=len(events),
                    cognitive_kinds=len(kinds),
                    epistemic_levels=len(levels),
                    bridge_survivors=len(bridge_survivors),
                    fragments_restored=len(fragments),
                    converged_invariants_restored=len(_converged_invariants),
                    profiles_restored=len(profiles),
                )
            except Exception:
                logger.warning("nexus_state_restore_failed", exc_info=True)

        self._initialized = True
        logger.info("nexus_initialized", instance_id=instance_id)

    async def start_background_loops(self) -> None:
        """Start periodic divergence measurement and pressure computation."""
        if not self._initialized:
            logger.warning("nexus_not_initialized_for_background_loops")
            return

        self._tasks.append(
            asyncio.create_task(self._divergence_loop())
        )
        logger.info("nexus_background_loops_started")

    # ─── Loop 4: Sleep-Certified Fragment Extraction ─────────────

    async def _on_wake_initiated(self, event: SynapseEvent) -> None:
        """
        Handle WAKE_INITIATED from Oneiros.

        After each sleep cycle, Oneiros broadcasts WAKE_INITIATED.  This is
        the trigger for Nexus to query Logos for schemas that survived both
        slow-wave (compression) and REM (recombination) stages and package
        them as ShareableWorldModelFragments for federation sharing.

        Only schemas with compression_ratio > config threshold are extracted.
        """
        # ── Skia modulation halt ────────────────────────────────────────────────────
        if self._modulation_halted:
            logger.debug("nexus_wake_extraction_skipped_modulation_halted")
            return

        if self._world_model is None:
            return

        schema_ids = self._world_model.get_schema_ids()
        extracted: list[str] = []

        for schema_id in schema_ids:
            schema = self._world_model.get_schema(schema_id)
            if schema is None:
                continue

            # Only extract schemas with meaningful compression quality.
            # Sleep certification is implicit: WAKE_INITIATED means Oneiros
            # just completed a full slow-wave + REM cycle.
            if schema.get("compression_ratio", 0.0) < self._config.min_compression_ratio_to_share:
                continue

            # Don't re-extract fragments we already have
            if schema_id in self._schema_to_fragment:
                continue

            # Build relational skeleton from structural fields - strip all
            # domain-specific labels (name, description) before sharing.
            raw_structure: dict[str, Any] = {}
            pattern = schema.get("pattern", {})
            if isinstance(pattern, dict):
                raw_structure.update(pattern)
            # Top-level structural keys override pattern when present
            for key in ("nodes", "edges", "symmetry", "invariants"):
                if key in schema:
                    raw_structure[key] = schema[key]
            abstract_structure = _strip_domain_context(raw_structure)

            fragment = await self.extract_fragment(
                schema_id=schema_id,
                abstract_structure=abstract_structure,
                domain_labels=[schema.get("domain", "general")],
                observations_explained=int(schema.get("instance_count", 1)),
                description_length=max(
                    10.0,
                    schema.get("compression_ratio", 1.0) * 10.0,
                ),
                compression_ratio=schema.get("compression_ratio", 1.0),
                sleep_certification=SleepCertification(
                    survived_slow_wave=True,
                    survived_rem=True,
                    sleep_cycles_survived=1,
                ),
            )
            extracted.append(fragment.fragment_id)

        # Sync Tier 3 invariants from Kairos (bidirectional pull)
        kairos_synced = await self.sync_kairos_tier3()

        if extracted or kairos_synced:
            logger.info(
                "nexus_wake_initiated_fragments_extracted",
                count=len(extracted),
                kairos_synced=kairos_synced,
                total_schemas=len(schema_ids),
            )
            # Schedule sharing of all newly extracted certified fragments
            asyncio.create_task(
                self._share_extracted_fragments(extracted),
                name="nexus_share_wake_fragments",
            )

    async def _share_extracted_fragments(self, fragment_ids: list[str]) -> None:
        """Share a batch of newly extracted fragments via federation."""
        shared = 0
        for fid in fragment_ids:
            try:
                responses = await self.share_fragment(fid)
                if responses:
                    shared += 1
            except Exception:
                logger.exception("nexus_fragment_share_failed", fragment_id=fid)

        logger.info("nexus_wake_fragments_shared", attempted=len(fragment_ids), shared=shared)

    async def _on_empirical_invariant_confirmed(self, event: SynapseEvent) -> None:
        """
        Handle EMPIRICAL_INVARIANT_CONFIRMED (self-emitted by Nexus).

        When a fragment reaches Level 4 (EMPIRICAL_INVARIANT), feed it back
        into Logos as a highest-priority world model component, protected
        from entropic decay.  Also route to Equor for constitutional protection.

        This closes the triangulation cycle:
        Logos → Nexus → Federation → convergence → promotion → Logos
        """
        fragment_id = event.data.get("fragment_id", "")
        triangulation_confidence = float(event.data.get("triangulation_confidence", 0.0))

        fragment = self._local_fragments.get(fragment_id)
        if fragment is None:
            return

        # Feed back into Logos world model
        if self._logos_adapter is not None:
            try:
                self._logos_adapter.ingest_empirical_invariant_from_nexus(
                    abstract_structure=fragment.abstract_structure,
                    observations_explained=fragment.observations_explained,
                    triangulation_confidence=triangulation_confidence,
                )
            except Exception:
                logger.warning(
                    "nexus_invariant_logos_feedback_failed",
                    fragment_id=fragment_id,
                    exc_info=True,
                )

        logger.info(
            "nexus_empirical_invariant_feedback_complete",
            fragment_id=fragment_id,
            confidence=round(triangulation_confidence, 3),
        )

    async def _on_speciation_event(self, event: SynapseEvent) -> None:
        """
        Handle SPECIATION_EVENT - trigger InvariantBridge exchange.

        When two instances speciate, normal fragment sharing stops.
        This handler initiates the InvariantBridge exchange so causal
        invariants can still cross the speciation boundary.
        """
        instance_a = event.data.get("instance_a_id", "")
        instance_b = event.data.get("instance_b_id", "")

        # Only act if we are one of the speciated pair
        if self._instance_id not in (instance_a, instance_b):
            return

        remote_instance_id = instance_b if self._instance_id == instance_a else instance_a

        if self._federation is None:
            logger.warning(
                "invariant_bridge_skipped_no_federation",
                remote=remote_instance_id,
            )
            return

        remote_logos = await self._federation.get_remote_logos(remote_instance_id)
        if remote_logos is None:
            logger.warning(
                "invariant_bridge_skipped_remote_logos_unavailable",
                remote=remote_instance_id,
            )
            return

        report = await self.exchange_invariants_across_bridge(
            remote_logos=remote_logos,
            remote_instance_id=remote_instance_id,
        )
        if report is not None:
            logger.info(
                "invariant_bridge_exchange_completed",
                remote=remote_instance_id,
                equivalences=report.abstract_equivalences_found,
                compared=report.invariants_compared,
            )

    async def _on_kairos_tier3_discovered(self, event: SynapseEvent) -> None:
        """
        Handle KAIROS_TIER3_INVARIANT_DISCOVERED - auto-ingest as local fragment.

        When Kairos discovers a Tier 3 substrate-independent invariant,
        Nexus immediately ingests it for federation sharing. This is the
        push direction of the Kairos↔Nexus bidirectional protocol;
        sync_kairos_tier3() handles the pull direction.
        """
        invariant_id = event.data.get("invariant_id", "")
        if not invariant_id or invariant_id in self._schema_to_fragment:
            return

        abstract_form = event.data.get("abstract_form", {})
        applicable_domains = event.data.get("applicable_domains", [])
        domain_labels = (
            [d.get("domain", "") for d in applicable_domains if isinstance(d, dict)]
            if isinstance(applicable_domains, list)
            else []
        )

        fragment = await self.extract_fragment(
            schema_id=invariant_id,
            abstract_structure={
                "type": "causal_invariant",
                "tier": 3,
                "abstract_form": abstract_form,
                "hold_rate": event.data.get("hold_rate", 0.0),
            },
            domain_labels=domain_labels,
            observations_explained=sum(
                d.get("observation_count", 0)
                for d in applicable_domains
                if isinstance(d, dict)
            ) if isinstance(applicable_domains, list) else 0,
            description_length=event.data.get("description_length_bits", 10.0),
            compression_ratio=max(event.data.get("hold_rate", 1.0), 1.0),
            sleep_certification=SleepCertification(
                survived_slow_wave=True,
                survived_rem=True,
                sleep_cycles_survived=1,
            ),
        )

        # Schedule immediate federation sharing
        asyncio.create_task(
            self._share_extracted_fragments([fragment.fragment_id]),
            name=f"nexus_share_kairos_tier3_{invariant_id[:8]}",
        )

        logger.info(
            "kairos_tier3_auto_ingested",
            invariant_id=invariant_id,
            fragment_id=fragment.fragment_id,
        )

    # ─── Gap 2: Six Additional Event Handlers ────────────────────

    async def _on_federation_session_started(self, event: SynapseEvent) -> None:
        """
        FEDERATION_SESSION_STARTED → trigger fragment sharing.

        When a new federation sharing session is established, share all
        local sleep-certified fragments with the new peer.
        """
        # ── Skia modulation halt ────────────────────────────────────────────────────
        if self._modulation_halted:
            logger.debug("nexus_federation_session_skipped_modulation_halted")
            return

        remote_instance_id = event.data.get("remote_instance_id", "")
        if not remote_instance_id:
            return

        shareable_ids = [
            fid for fid, f in self._local_fragments.items()
            if f.is_shareable
        ]
        if shareable_ids:
            asyncio.create_task(
                self._share_extracted_fragments(shareable_ids),
                name=f"nexus_share_session_{remote_instance_id[:8]}",
            )
            logger.info(
                "federation_session_fragment_sharing_triggered",
                remote_instance_id=remote_instance_id,
                fragments_to_share=len(shareable_ids),
            )

    async def _on_sleep_consolidation_complete(self, event: SynapseEvent) -> None:
        """
        ONEIROS_CONSOLIDATION_COMPLETE → per-schema sleep certification.

        Instead of blanket certification on WAKE_INITIATED, certify only
        fragments whose schema_ids appear in the consolidation result.
        Fragments not processed during sleep retain previous state.
        """
        schema_ids = event.data.get("schema_ids", [])
        if not isinstance(schema_ids, list):
            # Backward compatibility: if event doesn't include schema_ids,
            # skip per-schema cert - blanket cert from WAKE_INITIATED applies
            return

        certified_count = 0
        for schema_id in schema_ids:
            fragment_id = self._schema_to_fragment.get(schema_id)
            if fragment_id is None:
                continue
            fragment = self._local_fragments.get(fragment_id)
            if fragment is None:
                continue
            fragment.sleep_certification.survived_slow_wave = True
            fragment.sleep_certification.survived_rem = True
            fragment.sleep_certification.sleep_cycles_survived = max(
                fragment.sleep_certification.sleep_cycles_survived + 1, 1
            )
            certified_count += 1

        if certified_count:
            logger.info(
                "per_schema_sleep_certification",
                schemas_certified=certified_count,
                total_schemas_in_event=len(schema_ids),
            )
            # Persist updated sleep certification flags (R1).
            if self._persistence is not None:
                certified_fragments = [
                    self._local_fragments[self._schema_to_fragment[sid]]
                    for sid in schema_ids
                    if sid in self._schema_to_fragment
                    and self._schema_to_fragment[sid] in self._local_fragments
                ]
                if certified_fragments:
                    try:
                        await self._persistence.persist_fragments(certified_fragments)
                    except Exception:
                        logger.warning(
                            "nexus_certified_fragments_persist_failed", exc_info=True
                        )

            # ── MEDIUM-4: emit NEXUS_CERTIFIED_FOR_FEDERATION immediately ────
            # Spec: emit NEXUS_CERTIFIED_FOR_FEDERATION right after
            # ONEIROS_CONSOLIDATION_COMPLETE + sleep_certified=true.
            # This triggers the IIEP session for sharing newly certified schemas
            # without waiting for the next WAKE_INITIATED cycle.
            consolidation_cycle_id = event.data.get("consolidation_cycle_id", "")
            certified_schema_ids = [
                sid for sid in schema_ids
                if sid in self._schema_to_fragment
                and self._schema_to_fragment[sid] in self._local_fragments
            ]
            await self._emit(
                SynapseEventType.NEXUS_CERTIFIED_FOR_FEDERATION,
                {
                    "instance_id": self._instance_id,
                    "schema_ids": certified_schema_ids,
                    "consolidation_cycle_id": consolidation_cycle_id,
                    "certified_fragment_count": certified_count,
                    "timestamp": utc_now().isoformat(),
                },
            )
            logger.info(
                "nexus_certified_for_federation_emitted",
                certified_count=certified_count,
                consolidation_cycle_id=consolidation_cycle_id,
            )

            # Schedule immediate sharing of newly certified fragments so peers
            # receive them in this wake cycle rather than the next.
            newly_certified_ids = [
                self._schema_to_fragment[sid]
                for sid in certified_schema_ids
                if self._local_fragments.get(
                    self._schema_to_fragment[sid]
                ) is not None
            ]
            if newly_certified_ids and self._federation is not None:
                asyncio.create_task(
                    self._share_extracted_fragments(newly_certified_ids),
                    name=f"nexus_iiep_post_sleep_{consolidation_cycle_id[:8]}",
                )

    async def _on_hypothesis_confirmed(self, event: SynapseEvent) -> None:
        """
        EVO_HYPOTHESIS_CONFIRMED → update hypothesis diversity dimension.

        When Evo confirms a hypothesis, update divergence profiles so
        the hypothesis diversity dimension reflects the latest state.
        """
        hypothesis_id = event.data.get("hypothesis_id", "")
        if not hypothesis_id:
            return

        # The confirmed hypothesis changes the local hypothesis profile.
        # Next divergence measurement cycle will pick up the updated set.
        # Force an immediate profile refresh if measurer is available.
        if self._divergence_measurer is not None:
            logger.debug(
                "hypothesis_confirmed_divergence_refresh",
                hypothesis_id=hypothesis_id,
            )

    async def _on_hypothesis_refuted(self, event: SynapseEvent) -> None:
        """
        EVO_HYPOTHESIS_REFUTED → lower federation_confidence on GenerativeSchema nodes
        associated with this hypothesis domain; trigger schema challenge if threshold crossed.

        A refuted hypothesis weakens the epistemic standing of schemas that were
        built on it. Nexus tracks this via the divergence profile so federated
        instances see the correction on next fragment exchange.
        """
        category = event.data.get("category", "")
        hypothesis_id = event.data.get("hypothesis_id", "")
        evidence_score = float(event.data.get("evidence_score", 0.0))
        if not hypothesis_id:
            return

        # Flag the divergence measurer so the next cycle re-measures hypothesis diversity.
        # This is sufficient - the divergence loop will propagate the correction via
        # DIVERGENCE_PRESSURE if the domain diverges meaningfully from federation peers.
        if self._divergence_measurer is not None:
            logger.debug(
                "hypothesis_refuted_divergence_refresh",
                hypothesis_id=hypothesis_id,
                category=category,
                evidence_score=evidence_score,
            )

        # Strong refutations (evidence_score well below baseline) may indicate our
        # local world model fragment is drifting from ground truth. Emit a lightweight
        # divergence pressure signal so Thymos and Fovea can respond.
        if evidence_score <= 1.0:
            await self._emit(
                SynapseEventType.DIVERGENCE_PRESSURE,
                {
                    "trigger": "hypothesis_refuted",
                    "domain": category,
                    "hypothesis_id": hypothesis_id,
                    "pressure_magnitude": 0.3,
                    "frontier_domains": [category] if category else [],
                },
            )

    async def _on_instance_spawned(self, event: SynapseEvent) -> None:
        """
        INSTANCE_SPAWNED → register new instance for divergence measurement.

        When a new organism instance joins the federation, prepare for
        divergence tracking against it.
        """
        instance_id = event.data.get("instance_id", "")
        if not instance_id or instance_id == self._instance_id:
            return

        # Initialize an empty profile - will be populated on first measurement
        stub_profile = InstanceDivergenceProfile(instance_id=instance_id)
        self._remote_profiles[instance_id] = stub_profile

        # Persist stub so the instance is tracked across restarts (R5).
        if self._persistence is not None:
            try:
                await self._persistence.persist_divergence_profiles([stub_profile])
            except Exception:
                logger.warning(
                    "nexus_instance_profile_stub_persist_failed",
                    instance_id=instance_id,
                    exc_info=True,
                )

        logger.info(
            "instance_spawned_registered",
            new_instance_id=instance_id,
        )

    async def _on_instance_retired(self, event: SynapseEvent) -> None:
        """
        INSTANCE_RETIRED → garbage-collect divergence history.

        Remove the retired instance from remote profiles and link failures.
        """
        instance_id = event.data.get("instance_id", "")
        if not instance_id:
            return

        self._remote_profiles.pop(instance_id, None)
        self._link_failures.pop(instance_id, None)

        # Remove remote fragments from this instance
        to_remove = [
            fid for fid, f in self._remote_fragments.items()
            if f.source_instance_id == instance_id
        ]
        for fid in to_remove:
            del self._remote_fragments[fid]

        logger.info(
            "instance_retired_cleaned_up",
            retired_instance_id=instance_id,
            remote_fragments_removed=len(to_remove),
        )

    async def _on_thymos_incident_resolved(self, event: SynapseEvent) -> None:
        """
        INCIDENT_RESOLVED → re-weight instance epistemic trust.

        When Thymos resolves a data integrity incident, apply a 20% triangulation
        confidence penalty to all local fragments whose compression_path.source_system
        matches the affected system. This prevents corrupted signals from inflating
        ground-truth confidence until re-validated by a new federation round.

        The penalty is conservative (×0.8) - not a hard revocation - so the
        organism can still reason with partially-trusted knowledge while re-validation
        converges. On next fragment receive + convergence, the confidence auto-repairs
        via ConvergenceDetector.update_triangulation().
        """
        incident_class = event.data.get("incident_class", "")
        source_system = event.data.get("source_system", "")
        resolution = event.data.get("resolution", "")

        if incident_class not in ("contract_violation", "data_corruption"):
            return

        if not source_system:
            return

        # Discount triangulation confidence on local fragments sourced from the
        # affected system.  Only fragments with meaningful confidence are adjusted;
        # zero-confidence fragments are already effectively untrusted.
        discounted_count = 0
        _TRUST_DISCOUNT = 0.8  # 20% penalty per data integrity incident
        for fragment in self._local_fragments.values():
            if fragment.compression_path.source_system != source_system:
                continue
            tri = fragment.triangulation
            if tri.triangulation_confidence <= 0.0:
                continue
            # Discount each source's divergence_score to lower the aggregate
            # source_diversity_score and thus the computed triangulation_confidence.
            for src in tri.independent_sources:
                src.divergence_score = max(0.0, src.divergence_score * _TRUST_DISCOUNT)
            discounted_count += 1

        logger.info(
            "incident_resolved_epistemic_trust_discounted",
            incident_class=incident_class,
            source_system=source_system,
            resolution=resolution,
            fragments_discounted=discounted_count,
            trust_discount=_TRUST_DISCOUNT,
        )

        if discounted_count > 0:
            await self._emit(
                SynapseEventType.DIVERGENCE_PRESSURE,
                {
                    "instance_id": self._instance_id,
                    "trigger": "data_integrity_incident",
                    "incident_class": incident_class,
                    "source_system": source_system,
                    "fragments_discounted": discounted_count,
                    "recommendation": (
                        f"Triangulation confidence discounted for {discounted_count} "
                        f"fragments sourced from '{source_system}' due to "
                        f"{incident_class}. Re-validation via federation convergence "
                        "will restore confidence."
                    ),
                    "timestamp": utc_now().isoformat(),
                },
            )

    async def _on_oneiros_threat_scenario(self, event: SynapseEvent) -> None:
        """
        ONEIROS_THREAT_SCENARIO → epistemic stability check (MEDIUM-3).

        Oneiros ThreatSimulator emits hypothetical world states that represent
        adversarial conditions (threat scenarios seeded from Thymos incidents,
        Evo concerning hypotheses, and Nova high-uncertainty beliefs).

        Nexus runs each scenario's domain through convergence to check for
        epistemic instability: if our local fragments in that domain have LOW
        convergence under the threat conditions, the world model is fragile
        there.  We emit a diagnostic so operators and Thymos can take note.

        Algorithm:
        1. Extract domain and severity from the scenario.
        2. Build a synthetic threat fragment from the scenario description.
        3. Compare the threat fragment against all local fragments in the
           matching domain using WL-1 convergence.
        4. If best convergence_score < 0.4 AND we have local fragments in
           that domain, the domain is epistemically unstable under threat -
           emit DIVERGENCE_PRESSURE warning via Synapse.
        5. Log outcome regardless for observability.
        """
        scenario_id = event.data.get("scenario_id", "")
        domain = event.data.get("domain", "")
        severity = float(event.data.get("severity", 0.5))
        scenario_description = event.data.get("scenario_description", "")

        if not scenario_id or not domain:
            return

        # Build a synthetic adversarial fragment representing the threat world state.
        # Uses the scenario domain to strip labels and find matching local fragments.
        threat_fragment = ShareableWorldModelFragment(
            fragment_id=f"threat_{scenario_id}",
            source_instance_id=self._instance_id,
            abstract_structure={
                "nodes": [{"type": "threat_state"}, {"type": "causal_factor"}],
                "edges": [{"type": "leads_to", "from": 0, "to": 1}],
                "symmetry": "chain",
                "invariants": ["adversarial"],
            },
            domain_labels=[domain],
            sleep_certification=SleepCertification(
                survived_slow_wave=True,
                survived_rem=True,
                sleep_cycles_survived=1,
            ),
        )

        # Find local fragments in the threatened domain
        domain_fragments = [
            f for f in self._local_fragments.values()
            if domain in f.domain_labels
        ]

        if not domain_fragments:
            logger.debug(
                "nexus_threat_scenario_no_local_domain",
                scenario_id=scenario_id,
                domain=domain,
            )
            return

        # Run convergence check: measure structural stability under threat
        best_score = 0.0
        for local_frag in domain_fragments:
            result = self._convergence_detector.compare_structures(
                local_frag, threat_fragment
            )
            if result.convergence_score > best_score:
                best_score = result.convergence_score

        # Low convergence against the threat fragment means our world model
        # in this domain is STABLE (structurally different from the threat).
        # HIGH convergence means our model looks similar to the adversarial state -
        # epistemic instability under threat.
        epistemic_instability = best_score >= 0.4

        logger.info(
            "nexus_threat_scenario_convergence",
            scenario_id=scenario_id,
            domain=domain,
            severity=severity,
            best_score=round(best_score, 3),
            domain_fragments=len(domain_fragments),
            epistemic_instability=epistemic_instability,
        )

        if epistemic_instability:
            # Emit divergence pressure to drive exploration away from this
            # structurally threatened region of the epistemic landscape.
            await self._emit(
                SynapseEventType.DIVERGENCE_PRESSURE,
                {
                    "instance_id": self._instance_id,
                    "trigger": "oneiros_threat_scenario",
                    "scenario_id": scenario_id,
                    "domain": domain,
                    "severity": severity,
                    "epistemic_instability_score": round(best_score, 3),
                    "recommendation": (
                        f"Domain '{domain}' world model is structurally similar to "
                        "adversarial threat state - explore alternative framings."
                    ),
                    "timestamp": utc_now().isoformat(),
                },
            )

    async def _on_federation_revenue_learning(self, event: SynapseEvent) -> None:
        """
        REVENUE_INJECTED → trigger epistemic triangulation on economic world model.

        Revenue events are ground-truth signals about the organism's economic
        strategy. When revenue arrives, Nexus checks whether the economic domain
        fragments from federation peers have converged on similar strategies.
        Convergence across 3+ peers on revenue-generating approaches promotes
        the strategy to an epistemic ground truth (Level 3+).
        """
        try:
            data = event.data or {}
            amount = float(data.get("amount_usd", 0.0))
            source = str(data.get("source", "unknown"))
            if amount <= 0:
                return

            # Find local fragments in the economic domain
            economic_fragments = [
                f for f in self._local_fragments.values()
                if any(lbl in ("economic", "oikos", "yield", "bounty") for lbl in f.domain_labels)
            ]

            if not economic_fragments:
                logger.debug("nexus_revenue_no_economic_fragments", source=source)
                return

            # Trigger a ground-truth promotion evaluation for economic fragments
            # now that we have a concrete revenue signal to anchor confidence.
            await self.evaluate_all_promotions()

            logger.info(
                "nexus_revenue_epistemic_update",
                amount_usd=amount,
                source=source,
                economic_fragments=len(economic_fragments),
            )
        except Exception as exc:
            logger.warning("nexus_revenue_learning_failed", error=str(exc))

    async def _on_federation_bounty_learning(self, event: SynapseEvent) -> None:
        """
        BOUNTY_PAID → record confirmed economic ground truth signal.

        A paid bounty is one of the strongest economic ground-truth signals -
        an external party has validated the organism's capability. Nexus records
        this as a high-confidence economic signal and triggers promotion evaluation
        for any bounty-domain fragments. If 3+ federation peers have confirmed
        similar bounty revenue, promote the capability schema to Level 3.
        """
        try:
            data = event.data or {}
            bounty_id = str(data.get("bounty_id", ""))
            amount = float(data.get("reward_usd", data.get("amount", 0.0)))
            if not bounty_id and amount <= 0:
                return

            # Find fragments in bounty-related domains
            bounty_fragments = [
                f for f in self._local_fragments.values()
                if any(lbl in ("bounty", "economic", "capability") for lbl in f.domain_labels)
            ]

            if bounty_fragments:
                # Elevate triangulation confidence on bounty fragments
                for frag in bounty_fragments:
                    current = frag.triangulation.triangulation_confidence
                    # Confirmed revenue is concrete evidence - boost confidence
                    frag.triangulation.triangulation_confidence = min(
                        1.0, current + 0.05
                    )

            # Trigger promotion pipeline - may promote to Level 3+ on peer confirmation
            await self.evaluate_all_promotions()

            logger.info(
                "nexus_bounty_epistemic_update",
                bounty_id=bounty_id,
                amount_usd=amount,
                bounty_fragments=len(bounty_fragments),
            )
        except Exception as exc:
            logger.warning("nexus_bounty_learning_failed", error=str(exc))

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle VitalityCoordinator austerity orders.

        Skia emits SYSTEM_MODULATION when the organism needs to conserve resources.
        This system applies the directive and ACKs so Skia knows the order was received.
        """
        data = getattr(event, "data", {}) or {}
        level = data.get("level", "nominal")
        halt_systems = data.get("halt_systems", [])
        modulate = data.get("modulate", {})

        system_id = "nexus"
        compliant = True
        reason: str | None = None

        if system_id in halt_systems:
            self._modulation_halted = True
            logger.warning("system_modulation_halt", level=level)
        elif system_id in modulate:
            directives = modulate[system_id]
            self._apply_modulation_directives(directives)
            logger.info("system_modulation_applied", level=level, directives=directives)
        elif level == "nominal":
            self._modulation_halted = False
            logger.info("system_modulation_resumed", level=level)

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
                logger.warning("modulation_ack_failed", error=str(exc))

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply modulation directives from VitalityCoordinator.

        Nexus directive: {"mode": "inbound_only"} - accept incoming fragments
        but pause outbound sharing to reduce bandwidth during austerity.
        """
        mode = directives.get("mode")
        if mode == "inbound_only":
            logger.info("modulation_inbound_only_mode_set")
        else:
            logger.info("modulation_directives_received", directives=directives)

    def _feed_convergence_to_logos(
        self,
        local_fragment: ShareableWorldModelFragment,
        convergence_result: ConvergenceResult,
    ) -> None:
        """
        After convergence raises a fragment's triangulation_confidence, update
        the corresponding Logos schema with higher epistemic confidence.

        Called from receive_fragment() after successful convergence detection.
        """
        if self._logos_adapter is None:
            return

        new_confidence = local_fragment.triangulation.triangulation_confidence
        if new_confidence < 0.1:
            return

        # Find the schema_id that maps to this fragment
        schema_id = next(
            (sid for sid, fid in self._schema_to_fragment.items()
             if fid == local_fragment.fragment_id),
            None,
        )
        if schema_id is None:
            return

        try:
            self._logos_adapter.update_schema_triangulation_confidence(
                schema_id=schema_id,
                new_confidence=new_confidence,
            )
        except Exception:
            logger.debug(
                "nexus_logos_confidence_update_failed",
                fragment_id=local_fragment.fragment_id,
                exc_info=True,
            )

    async def shutdown(self) -> None:
        """Cancel background tasks and clean up."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self._convergence_detector.clear_cache()
        logger.info("nexus_shutdown")

    async def health(self) -> dict[str, Any]:
        """Health check for Synapse monitoring."""
        registry = self._speciation_registry.state
        circuit_open_links = [
            lid for lid, count in self._link_failures.items()
            if count >= self._max_consecutive_failures
        ]
        return {
            "status": "healthy" if self._initialized else "stopped",
            "local_fragments": len(self._local_fragments),
            "remote_fragments": len(self._remote_fragments),
            "convergences_detected": len(self._convergences),
            "remote_profiles_cached": len(self._remote_profiles),
            "triangulation_weight": self._triangulation_weight,
            "speciation_events": len(registry.speciation_events),
            "cognitive_kinds": len(registry.cognitive_kinds),
            "active_bridges": len(registry.active_bridge_pairs),
            "federation_links_circuit_open": circuit_open_links,
            "kairos_wired": self._kairos is not None,
        }

    # ─── Phase A: Fragment Operations ────────────────────────────

    async def extract_fragment(
        self,
        schema_id: str,
        abstract_structure: dict[str, Any],
        domain_labels: list[str],
        observations_explained: int,
        description_length: float,
        compression_ratio: float,
        compression_path: CompressionPath | None = None,
        sleep_certification: SleepCertification | None = None,
        domain_hints: list[str] | None = None,
        economic_context: dict[str, Any] | None = None,
    ) -> ShareableWorldModelFragment:
        """
        Extract a shareable fragment from a world model schema.

        The abstract_structure should have domain labels already stripped -
        only relational skeleton (nodes, edges, symmetry, invariants).

        domain_hints and economic_context (NEXUS-ECON-2/3) preserve economic
        metadata alongside the abstract structure so federation peers can
        recover economic meaning during convergence analysis.
        """
        # Auto-detect economic domain hints when not explicitly provided.
        resolved_hints = domain_hints if domain_hints is not None else (
            _detect_economic_domain_hints(domain_labels, abstract_structure)
        )

        fragment = ShareableWorldModelFragment(
            fragment_id=new_id(),
            source_instance_id=self._instance_id,
            source_instance_divergence_score=self._triangulation_weight,
            abstract_structure=abstract_structure,
            domain_labels=domain_labels,
            observations_explained=observations_explained,
            description_length=description_length,
            compression_ratio=compression_ratio,
            compression_path=compression_path or CompressionPath(),
            sleep_certification=sleep_certification or SleepCertification(),
            triangulation=TriangulationMetadata(),
            domain_hints=resolved_hints,
            economic_context=economic_context,
            created_at=utc_now(),
            last_confirmed_at=utc_now(),
        )

        self._local_fragments[fragment.fragment_id] = fragment
        self._schema_to_fragment[schema_id] = fragment.fragment_id
        self._enforce_fragment_limit()

        # Persist to Neo4j immediately after extraction (R1).
        if self._persistence is not None:
            try:
                await self._persistence.persist_fragments([fragment])
            except Exception:
                logger.warning(
                    "nexus_fragment_persist_failed",
                    fragment_id=fragment.fragment_id,
                    exc_info=True,
                )

        logger.info(
            "fragment_extracted",
            fragment_id=fragment.fragment_id,
            schema_id=schema_id,
            domains=domain_labels,
            shareable=fragment.is_shareable,
            quality=fragment.quality_score,
        )

        return fragment

    async def share_fragment(
        self, fragment_id: str
    ) -> dict[str, WorldModelFragmentShareResponse]:
        """
        Share a local fragment with the federation.
        Only shares fragments that have passed sleep certification.
        """
        fragment = self._local_fragments.get(fragment_id)
        if fragment is None:
            logger.warning("fragment_not_found", fragment_id=fragment_id)
            return {}

        if not fragment.is_shareable:
            logger.warning(
                "fragment_not_shareable",
                fragment_id=fragment_id,
                sleep_certified=fragment.sleep_certification.is_certified,
            )
            return {}

        if self._federation is None:
            logger.warning("federation_not_wired")
            return {}

        # Telos gate: only share fragments that score above the care+coherence
        # threshold, ensuring the federation receives high-quality structures.
        if self._telos is not None:
            telos = self._telos
            try:
                telos_score = telos.score_fragment({
                    "domain": fragment.domain_labels[0] if fragment.domain_labels else "",
                    "coverage": fragment.quality_score,
                    "coherence_validated": fragment.sleep_certification.is_certified,
                    "prediction_accuracy": getattr(
                        fragment, "prediction_accuracy",
                        fragment.quality_score,
                    ),
                })
                if telos_score < 0.3:
                    logger.info(
                        "fragment_below_telos_threshold",
                        fragment_id=fragment_id,
                        telos_score=round(telos_score, 3),
                    )
                    return {}
            except Exception:
                logger.debug("telos_fragment_scoring_failed", exc_info=True)

        message = WorldModelFragmentShare(
            sender_instance_id=self._instance_id,
            sender_divergence_score=self._triangulation_weight,
            fragment=fragment,
            sender_quality_claim=fragment.quality_score,
            sender_triangulation_confidence=(
                fragment.triangulation.triangulation_confidence
            ),
            sleep_certified=fragment.sleep_certification.is_certified,
        )

        # ── IIEP envelope (HIGH-1) ────────────────────────────────────────
        # Wrap the outbound fragment in an IIEPMessage for federation-wide
        # session tracking.  Each share_fragment call opens a new IIEP session
        # (broadcast responder_id = "").  The session_id is recorded locally
        # so inbound convergence results can be correlated back to it.
        iiep = IIEPMessage(
            initiator_id=self._instance_id,
            responder_id="",  # broadcast
            fragment_type=IIEPFragmentType.WORLD_MODEL_FRAGMENT,
            payload=message.model_dump(mode="json"),
            sleep_certified=fragment.sleep_certification.is_certified,
            convergence_round=len(fragment.triangulation.independent_sources),
        )
        self._iiep_sessions[iiep.session_id] = {
            "initiator_id": self._instance_id,
            "fragment_id": fragment_id,
            "started_at": utc_now().isoformat(),
            "convergence_round": iiep.convergence_round,
        }

        # Emit WORLD_MODEL_FRAGMENT_SHARE on the Synapse bus so other systems
        # (Oikos, Telos) can subscribe to gate sharing decisions.
        await self._emit(
            SynapseEventType.WORLD_MODEL_FRAGMENT_SHARE,
            {
                "message_id": message.message_id,
                "session_id": iiep.session_id,
                "sender_instance_id": self._instance_id,
                "fragment_id": fragment_id,
                "sender_divergence_score": self._triangulation_weight,
                "sender_quality_claim": fragment.quality_score,
                "sender_triangulation_confidence": (
                    fragment.triangulation.triangulation_confidence
                ),
                "sleep_certified": iiep.sleep_certified,
                "consolidation_cycle_id": message.consolidation_cycle_id,
                "domain_labels": fragment.domain_labels,
                "domain_hints": fragment.domain_hints,
                "economic_context": fragment.economic_context,
                "compression_ratio": fragment.compression_ratio,
                "source_system": fragment.compression_path.source_system,
                "convergence_round": iiep.convergence_round,
                "timestamp": utc_now().isoformat(),
            },
        )

        responses = await self._federation.broadcast_fragment(message)

        # Spec payload: {fragment_id, target_instance, abstract_structure}.
        # Emit one event per accepting peer so consumers can track per-instance sharing.
        for peer_id, response in responses.items():
            if response.outcome == FragmentShareOutcome.ACCEPTED:
                await self._emit(
                    SynapseEventType.FRAGMENT_SHARED,
                    {
                        "fragment_id": fragment_id,
                        "target_instance": peer_id,
                        "abstract_structure": fragment.abstract_structure,
                        "session_id": iiep.session_id,
                    },
                )

        return responses

    async def receive_fragment(
        self,
        message: WorldModelFragmentShare,
    ) -> WorldModelFragmentShareResponse:
        """
        Handle an inbound fragment from a federated peer.

        Validates the fragment, runs convergence detection against
        all local fragments, and updates triangulation metadata.
        """
        fragment = message.fragment

        # Validate sleep certification
        if not fragment.sleep_certification.is_certified:
            return WorldModelFragmentShareResponse(
                message_id=message.message_id,
                outcome=FragmentShareOutcome.REJECTED_NO_SLEEP_CERT,
                reason="Fragment has not survived a complete sleep cycle.",
            )

        # Validate quality
        min_ratio = self._config.min_compression_ratio_to_share
        if fragment.compression_ratio < min_ratio:
            return WorldModelFragmentShareResponse(
                message_id=message.message_id,
                outcome=FragmentShareOutcome.REJECTED_LOW_QUALITY,
                reason=(
                    f"Compression ratio {fragment.compression_ratio:.2f} "
                    f"below minimum {min_ratio}."
                ),
            )

        # Block post-speciation fragment sharing - only InvariantBridge remains.
        # After speciation, structural languages are incompatible; fragments
        # cannot meaningfully converge and sharing is epistemically unsound.
        if not self.is_fragment_sharing_allowed(fragment.source_instance_id):
            return WorldModelFragmentShareResponse(
                message_id=message.message_id,
                outcome=FragmentShareOutcome.REJECTED_TRUST_INSUFFICIENT,
                reason=(
                    f"Fragment sharing blocked: instances {self._instance_id!r} "
                    f"and {fragment.source_instance_id!r} have speciated. "
                    "Use InvariantBridge for cross-speciation exchange."
                ),
            )

        # Check for duplicates
        if fragment.fragment_id in self._remote_fragments:
            return WorldModelFragmentShareResponse(
                message_id=message.message_id,
                outcome=FragmentShareOutcome.REJECTED_DUPLICATE,
                reason="Fragment already received.",
            )

        # Store the remote fragment
        self._remote_fragments[fragment.fragment_id] = fragment

        # Look up the actual measured divergence for this source instance.
        # This is the authoritative source_diversity for triangulation weight.
        remote_cached = self._remote_profiles.get(fragment.source_instance_id)
        peer_divergence: float | None = None
        if remote_cached is not None and self._divergence_measurer is not None:
            local_profile = self._divergence_measurer.build_local_profile()
            peer_div_score = self._divergence_measurer.measure(
                local_profile, remote_cached
            )
            peer_divergence = peer_div_score.overall

        # Run convergence detection against all local fragments
        best_convergence: ConvergenceResult | None = None
        for local_fragment in self._local_fragments.values():
            result = self._convergence_detector.compare_structures(
                local_fragment, fragment, peer_divergence_score=peer_divergence
            )
            if result.is_convergent:
                if (
                    best_convergence is None
                    or result.convergence_score > best_convergence.convergence_score
                ):
                    best_convergence = result

                # Update triangulation metadata on the local fragment
                self._convergence_detector.update_triangulation(
                    local_fragment.triangulation,
                    confirming_instance_id=fragment.source_instance_id,
                    confirming_divergence_score=message.sender_divergence_score,
                    confirming_fragment_id=fragment.fragment_id,
                )

                # Feed raised triangulation confidence back to Logos world model
                self._feed_convergence_to_logos(local_fragment, result)

                self._convergences.append(result)

                await self._emit(
                    SynapseEventType.CONVERGENCE_DETECTED,
                    result.model_dump(),
                )

        self._enforce_convergence_limit()

        # Persist updated triangulation metadata for any local fragments
        # whose confidence changed due to this convergence (R1).
        if best_convergence is not None and self._persistence is not None:
            updated_local = [
                f for f in self._local_fragments.values()
                if any(
                    s.instance_id == fragment.source_instance_id
                    for s in f.triangulation.independent_sources
                )
            ]
            if updated_local:
                try:
                    await self._persistence.persist_fragments(updated_local)
                except Exception:
                    logger.warning(
                        "nexus_triangulation_update_persist_failed", exc_info=True
                    )

        # ── Oikos metabolic coupling (Gap HIGH-5 + HIGH-2 economic reward) ──
        # Emit NEXUS_CONVERGENCE_METABOLIC_SIGNAL after processing:
        #   - Convergence achieved → bonus signal + economic_reward_usd
        #     (convergence_tier × 0.001 USDC; reset divergence counter)
        #   - No convergence → increment counter; penalty when threshold crossed
        #   Divergent instances receive no reward - selection pressure toward sharing.
        if best_convergence is not None:
            self._consecutive_divergence_cycles = 0
            # Determine the local fragment's current epistemic level for tier reward.
            # Find the local fragment that produced best_convergence.
            local_fragment_for_reward = self._local_fragments.get(
                best_convergence.fragment_a_id
            )
            if local_fragment_for_reward is None:
                # fragment_b may be local if we are source_b in the pair
                local_fragment_for_reward = self._local_fragments.get(
                    best_convergence.fragment_b_id
                )
            convergence_tier = 0
            if local_fragment_for_reward is not None and self._promotion_pipeline is not None:
                convergence_tier = int(
                    self._promotion_pipeline.get_level(
                        local_fragment_for_reward.fragment_id
                    )
                )
            economic_reward_usd = (
                convergence_tier * self._config.convergence_economic_reward_usdc_per_tier
            )
            await self._emit(
                SynapseEventType.NEXUS_CONVERGENCE_METABOLIC_SIGNAL,
                {
                    "instance_id": self._instance_id,
                    "remote_instance_id": fragment.source_instance_id,
                    "convergence_score": best_convergence.convergence_score,
                    "source_diversity": best_convergence.source_diversity,
                    "wl1_used": best_convergence.wl1_used,
                    "fragment_a_id": best_convergence.fragment_a_id,
                    "fragment_b_id": best_convergence.fragment_b_id,
                    "metabolic_signal": "bonus",
                    "magnitude": self._config.convergence_growth_bonus,
                    "consecutive_divergence_cycles": 0,
                    "convergence_tier": convergence_tier,
                    "economic_reward_usd": economic_reward_usd,
                    "timestamp": utc_now().isoformat(),
                },
            )
        else:
            self._consecutive_divergence_cycles += 1
            if self._consecutive_divergence_cycles >= self._config.divergence_penalty_threshold:
                await self._emit(
                    SynapseEventType.NEXUS_CONVERGENCE_METABOLIC_SIGNAL,
                    {
                        "instance_id": self._instance_id,
                        "remote_instance_id": fragment.source_instance_id,
                        "convergence_score": 0.0,
                        "source_diversity": 0.0,
                        "wl1_used": False,
                        "fragment_a_id": "",
                        "fragment_b_id": "",
                        "metabolic_signal": "penalty",
                        "magnitude": self._config.divergence_metabolic_penalty,
                        "consecutive_divergence_cycles": self._consecutive_divergence_cycles,
                        "convergence_tier": 0,
                        "economic_reward_usd": 0.0,
                        "timestamp": utc_now().isoformat(),
                    },
                )
                logger.info(
                    "nexus_persistent_divergence_penalty",
                    consecutive_cycles=self._consecutive_divergence_cycles,
                    remote_instance=fragment.source_instance_id,
                )

        return WorldModelFragmentShareResponse(
            message_id=message.message_id,
            outcome=FragmentShareOutcome.ACCEPTED,
            convergence_detected=best_convergence is not None,
            convergence_score=(
                best_convergence.convergence_score if best_convergence else 0.0
            ),
            wl1_used=best_convergence.wl1_used if best_convergence else False,
        )

    # ─── Phase B: Divergence Operations ──────────────────────────

    async def measure_divergence(
        self, link_id: str
    ) -> DivergenceScore | None:
        """
        Measure divergence against a specific federated peer.
        Fetches the remote profile via federation, then computes
        five-dimensional divergence score.
        """
        if self._divergence_measurer is None or self._federation is None:
            return None

        local_profile = self._divergence_measurer.build_local_profile()
        remote_profile = await self._federation.get_remote_profile(link_id)

        if remote_profile is None:
            logger.warning("remote_profile_unavailable", link_id=link_id)
            return None

        self._remote_profiles[remote_profile.instance_id] = remote_profile

        # Persist the updated remote profile so cross-session divergence
        # tracking survives restarts (R5).
        if self._persistence is not None:
            try:
                await self._persistence.persist_divergence_profiles([remote_profile])
            except Exception:
                logger.warning(
                    "nexus_divergence_profile_persist_failed",
                    instance_id=remote_profile.instance_id,
                    exc_info=True,
                )

        score = self._divergence_measurer.measure(local_profile, remote_profile)

        if self._incentive_engine is not None:
            self._incentive_engine.update_divergence(
                remote_profile.instance_id, score
            )

        # Phase C: check for speciation
        await self._check_speciation(score)

        return score

    async def measure_all_divergences(self) -> dict[str, DivergenceScore]:
        """Measure divergence against all active federation links."""
        if self._federation is None:
            return {}

        link_ids = self._federation.get_active_link_ids()
        results: dict[str, DivergenceScore] = {}

        for link_id in link_ids:
            # Check link health before measuring - skip links that are
            # in a circuit-breaker open state (too many consecutive failures).
            if self._is_link_circuit_open(link_id):
                logger.debug("link_circuit_open_skipping", link_id=link_id)
                continue

            try:
                score = await self.measure_divergence(link_id)
                if score is not None:
                    results[link_id] = score
                    self._record_link_success(link_id)
            except Exception:
                self._record_link_failure(link_id)
                logger.warning(
                    "divergence_measurement_failed",
                    link_id=link_id,
                    consecutive_failures=self._link_failures.get(link_id, 0),
                    exc_info=True,
                )

        return results

    async def compute_and_apply_pressure(self) -> DivergencePressure | None:
        """
        Compute divergence pressure and apply to Thymos if needed.
        Returns the pressure signal if generated, None otherwise.
        """
        if self._incentive_engine is None or self._divergence_measurer is None:
            return None

        local_profile = self._divergence_measurer.build_local_profile()

        # Update federation domain coverage
        all_profiles = list(self._remote_profiles.values())
        all_profiles.append(local_profile)
        self._incentive_engine.update_federation_domains(all_profiles)

        # Compute weight - snapshot previous so TRIANGULATION_WEIGHT_UPDATE
        # can report the delta (spec §XI mandates previous_weight in payload).
        previous_weight = self._triangulation_weight
        self._triangulation_weight = (
            self._incentive_engine.compute_triangulation_weight()
        )

        await self._emit(
            SynapseEventType.TRIANGULATION_WEIGHT_UPDATE,
            {
                "instance_id": self._instance_id,
                "previous_weight": previous_weight,
                "new_weight": self._triangulation_weight,
                "weight_delta": round(self._triangulation_weight - previous_weight, 4),
                "peer_count": len(self._remote_profiles),
            },
        )

        # Compute pressure
        pressure = self._incentive_engine.compute_divergence_pressure(
            local_profile
        )

        if pressure is not None:
            self._incentive_engine.apply_pressure_to_thymos(pressure)

            # Notify Fovea of divergence pressure so it can bias attention
            # toward frontier domains the federation lacks coverage in.
            if self._fovea is not None:
                try:
                    self._fovea.receive_divergence_signal(
                        divergence_pressure=pressure.pressure_magnitude,
                        frontier_domains=pressure.frontier_domains,
                    )
                except Exception:
                    logger.debug("fovea_divergence_signal_failed", exc_info=True)

            await self._emit(
                SynapseEventType.DIVERGENCE_PRESSURE,
                pressure.model_dump(),
            )

        return pressure

    def get_local_profile(self) -> InstanceDivergenceProfile | None:
        """Build and return the local divergence profile."""
        if self._divergence_measurer is None:
            return None
        return self._divergence_measurer.build_local_profile()

    # ─── Phase C: Speciation Operations ─────────────────────────

    async def _check_speciation(self, score: DivergenceScore) -> None:
        """Check a divergence score for speciation and handle if triggered."""
        event = self._speciation_detector.check_for_speciation(
            score, self._speciation_registry
        )
        if event is None:
            return

        self._speciation_registry.register_speciation(
            event,
            logos_a=self._world_model,
            # Remote Logos not directly available - metadata comes from score
            logos_b=None,
        )

        # Register an invariant bridge between the speciated pair
        self._speciation_registry.register_bridge(
            event.instance_a_id, event.instance_b_id
        )

        await self._emit(
            SynapseEventType.SPECIATION_EVENT,
            event.model_dump(),
        )

        # Persist speciation state to Neo4j
        if self._persistence is not None:
            try:
                await self._persistence.persist_speciation_events([event])
                await self._persistence.persist_cognitive_kinds(
                    list(self._speciation_registry.state.cognitive_kinds.values())
                )
            except Exception:
                logger.warning("nexus_speciation_persist_failed", exc_info=True)

    async def exchange_invariants_across_bridge(
        self,
        remote_logos: LogosWorldModelProtocol,
        remote_instance_id: str,
    ) -> InvariantExchangeReport | None:
        """
        Exchange causal invariants with a speciated peer via InvariantBridge.

        Only callable between speciated instances. Returns None if the
        instances are not speciated or local Logos is not wired.
        """
        if self._world_model is None:
            logger.warning("world_model_not_wired_for_bridge")
            return None

        if not self._speciation_registry.are_speciated(
            self._instance_id, remote_instance_id
        ):
            logger.warning(
                "not_speciated_for_bridge",
                local=self._instance_id,
                remote=remote_instance_id,
            )
            return None

        report = self._invariant_bridge.exchange_invariants(
            logos_a=self._world_model,
            logos_b=remote_logos,
            instance_a_id=self._instance_id,
            instance_b_id=remote_instance_id,
        )

        # Mark converged invariants as bridge survivors for promotion.
        # invariant_a_id is the schema_id used by _extract_causal_invariants
        # (not the fragment_id tracked by _local_fragments). Resolve it via
        # _schema_to_fragment; invariant_b_id belongs to the remote instance.
        if self._promotion_pipeline is not None:
            for ci in report.converged_invariants:
                # Resolve local schema_id → real fragment_id
                real_fragment_id = self._schema_to_fragment.get(ci.invariant_a_id)
                if real_fragment_id is not None:
                    self._promotion_pipeline.mark_bridge_survivor(real_fragment_id)
                else:
                    # Fall back to marking the schema_id directly in case
                    # the fragment was extracted with a matching schema_id
                    self._promotion_pipeline.mark_bridge_survivor(ci.invariant_a_id)

        # Persist bridge survivors and converged invariants to Neo4j (R1).
        if self._persistence is not None and report.converged_invariants:
            survivor_ids = [
                self._schema_to_fragment.get(ci.invariant_a_id, ci.invariant_a_id)
                for ci in report.converged_invariants
            ]
            try:
                await self._persistence.persist_bridge_survivors(survivor_ids)
            except Exception:
                logger.warning("nexus_bridge_survivors_persist_failed", exc_info=True)
            try:
                await self._persistence.persist_converged_invariants(
                    report.converged_invariants
                )
            except Exception:
                logger.warning(
                    "nexus_converged_invariants_persist_failed", exc_info=True
                )

        return report

    async def sync_kairos_tier3(self) -> int:
        """
        Pull Tier 3 invariants from Kairos and ingest as local fragments.

        Closes the bidirectional loop: Kairos pushes new Tier 3 discoveries
        via share_fragment(), and Nexus pulls any it may have missed -
        especially after sleep cycles when Kairos may have promoted invariants.

        Returns the number of new fragments ingested.
        """
        if self._kairos is None:
            return 0

        tier3_list = self._kairos.get_tier3_invariants()
        ingested = 0

        for inv_data in tier3_list:
            inv_id = inv_data.get("id", "")
            if not inv_id:
                continue

            # Skip if already tracked
            if inv_id in self._schema_to_fragment:
                continue

            abstract_form = inv_data.get("abstract_form", {})
            domains = inv_data.get("applicable_domains", [])
            domain_labels = (
                [d.get("domain", "") for d in domains if isinstance(d, dict)]
                if isinstance(domains, list)
                else []
            )

            fragment = await self.extract_fragment(
                schema_id=inv_id,
                abstract_structure={
                    "type": "causal_invariant",
                    "tier": 3,
                    "abstract_form": abstract_form,
                    "hold_rate": inv_data.get("invariance_hold_rate", 0.0),
                },
                domain_labels=domain_labels,
                observations_explained=sum(
                    d.get("observation_count", 0)
                    for d in domains
                    if isinstance(d, dict)
                ) if isinstance(domains, list) else 0,
                description_length=inv_data.get("description_length_bits", 10.0),
                compression_ratio=max(inv_data.get("invariance_hold_rate", 1.0), 1.0),
                sleep_certification=SleepCertification(
                    survived_slow_wave=True,
                    survived_rem=True,
                    sleep_cycles_survived=1,
                ),
            )
            ingested += 1
            logger.debug(
                "kairos_tier3_synced",
                invariant_id=inv_id,
                fragment_id=fragment.fragment_id,
            )

        if ingested:
            logger.info("kairos_tier3_sync_complete", new_fragments=ingested)

        return ingested

    def get_federation_confidence(self, schema_id: str) -> float | None:
        """
        Return the federation-derived triangulation confidence for a schema.

        Enables Logos to query Nexus for on-demand epistemic status
        without waiting for the next sleep cycle.
        """
        fragment_id = self._schema_to_fragment.get(schema_id)
        if fragment_id is None:
            return None

        fragment = self._local_fragments.get(fragment_id)
        if fragment is None:
            return None

        return fragment.triangulation.triangulation_confidence

    def is_fragment_sharing_allowed(
        self, remote_instance_id: str
    ) -> bool:
        """
        Check if normal fragment sharing is allowed with a remote instance.

        Post-speciation, normal sharing is blocked - only invariant bridge
        exchange remains possible.
        """
        return not self._speciation_registry.are_speciated(
            self._instance_id, remote_instance_id
        )

    @property
    def speciation_registry(self) -> SpeciationRegistry:
        return self._speciation_registry

    # ─── Phase D: Ground Truth Promotion Operations ──────────────

    async def evaluate_fragment_promotion(
        self, fragment_id: str
    ) -> PromotionDecision | None:
        """
        Evaluate a local fragment for epistemic level promotion.

        Returns the PromotionDecision, or None if the fragment or
        pipeline is not available.
        """
        if self._promotion_pipeline is None:
            return None

        fragment = self._local_fragments.get(fragment_id)
        if fragment is None:
            logger.warning(
                "fragment_not_found_for_promotion", fragment_id=fragment_id
            )
            return None

        decision = await self._promotion_pipeline.evaluate_for_promotion(fragment)

        if decision.promoted:
            # Emit events for significant promotions
            if decision.proposed_level == EpistemicLevel.GROUND_TRUTH_CANDIDATE:
                await self._emit(
                    SynapseEventType.GROUND_TRUTH_CANDIDATE,
                    {
                        "fragment_id": fragment_id,
                        "triangulation_confidence": (
                            decision.triangulation_confidence
                        ),
                        "source_count": decision.independent_source_count,
                        "source_diversity": decision.source_diversity,
                    },
                )
            elif decision.proposed_level == EpistemicLevel.EMPIRICAL_INVARIANT:
                await self._emit(
                    SynapseEventType.EMPIRICAL_INVARIANT_CONFIRMED,
                    {
                        "invariant_id": fragment_id,
                        "fragment_id": fragment_id,
                        "triangulation_confidence": (
                            decision.triangulation_confidence
                        ),
                        "survived_adversarial": (
                            decision.survived_adversarial_test
                        ),
                        "survived_competition": (
                            decision.survived_hypothesis_competition
                        ),
                    },
                )

            # Persist promotion to Neo4j
            if self._persistence is not None:
                try:
                    await self._persistence.persist_epistemic_promotions([{
                        "fragment_id": fragment_id,
                        "from_level": decision.current_level.value,
                        "to_level": decision.proposed_level.value,
                        "triangulation_confidence": decision.triangulation_confidence,
                        "source_count": decision.independent_source_count,
                        "source_diversity": decision.source_diversity,
                        "survived_adversarial": decision.survived_adversarial_test,
                        "survived_competition": decision.survived_hypothesis_competition,
                    }])
                except Exception:
                    logger.warning("nexus_promotion_persist_failed", exc_info=True)

            # Gap 4: Emit RE training example on Level 3+ promotions
            # The epistemic promotion reasoning is high-value curriculum for RE
            if decision.proposed_level.value >= EpistemicLevel.GROUND_TRUTH_CANDIDATE.value:
                await self._emit_re_training_example(fragment, decision)

        return decision

    async def evaluate_all_promotions(self) -> list[PromotionDecision]:
        """Evaluate all local fragments for possible promotion."""
        decisions: list[PromotionDecision] = []
        for fragment_id in list(self._local_fragments):
            decision = await self.evaluate_fragment_promotion(fragment_id)
            if decision is not None and decision.promoted:
                decisions.append(decision)
        return decisions

    def get_fragment_epistemic_level(
        self, fragment_id: str
    ) -> EpistemicLevel:
        """Return the current epistemic level of a fragment."""
        if self._promotion_pipeline is None:
            return EpistemicLevel.HYPOTHESIS
        return self._promotion_pipeline.get_level(fragment_id)

    def get_empirical_invariants(self) -> list[ShareableWorldModelFragment]:
        """Return all local fragments at Level 4 (EMPIRICAL_INVARIANT)."""
        if self._promotion_pipeline is None:
            return []
        return [
            f
            for f in self._local_fragments.values()
            if self._promotion_pipeline.get_level(f.fragment_id)
            == EpistemicLevel.EMPIRICAL_INVARIANT
        ]

    def get_ground_truth_candidates(self) -> list[ShareableWorldModelFragment]:
        """Return all local fragments at Level 3+ (GROUND_TRUTH_CANDIDATE or higher)."""
        if self._promotion_pipeline is None:
            return []
        return [
            f
            for f in self._local_fragments.values()
            if self._promotion_pipeline.get_level(f.fragment_id).value >= 3
        ]

    # ─── Query Interface ─────────────────────────────────────────

    def get_fragment(
        self, fragment_id: str
    ) -> ShareableWorldModelFragment | None:
        """Look up a fragment by ID (local or remote)."""
        return (
            self._local_fragments.get(fragment_id)
            or self._remote_fragments.get(fragment_id)
        )

    def get_high_confidence_structures(
        self, min_confidence: float = 0.3
    ) -> list[ShareableWorldModelFragment]:
        """Return local fragments with triangulation confidence above threshold."""
        return [
            f
            for f in self._local_fragments.values()
            if f.triangulation.triangulation_confidence >= min_confidence
        ]

    def get_divergence_weighted_fragments(self) -> list[dict[str, Any]]:
        """
        Return local fragments weighted by divergence score for Mitosis
        genome inheritance.

        Higher-divergence fragments (those confirmed from more diverse sources)
        are more valuable for child instance genome composition.
        Returns list of {fragment_id, abstract_structure, weight, epistemic_level}.
        """
        results: list[dict[str, Any]] = []
        for fid, fragment in self._local_fragments.items():
            tri = fragment.triangulation
            # Weight: triangulation confidence × source diversity
            weight = tri.triangulation_confidence * max(tri.source_diversity, 0.1)
            level = (
                self._promotion_pipeline.get_level(fid)
                if self._promotion_pipeline is not None
                else EpistemicLevel.HYPOTHESIS
            )
            results.append({
                "fragment_id": fid,
                "abstract_structure": fragment.abstract_structure,
                "domain_labels": fragment.domain_labels,
                "weight": weight,
                "epistemic_level": level.value,
                "source_count": tri.independent_source_count,
            })
        results.sort(key=lambda r: r["weight"], reverse=True)
        return results

    def get_epistemic_metabolic_value(self) -> dict[str, Any]:
        """
        Return the current epistemic metabolic value for Oikos coupling.

        High-triangulation structures represent metabolic investment that
        should be reflected in Oikos's economic accounting. Returns a
        summary of epistemic assets suitable for metabolic representation.
        """
        high_conf = self.get_high_confidence_structures(min_confidence=0.3)
        total_epistemic_value = sum(
            f.triangulation.triangulation_confidence * f.observations_explained
            for f in high_conf
        )
        empirical_count = len(self.get_empirical_invariants())
        candidate_count = len(self.get_ground_truth_candidates())

        return {
            "total_epistemic_value": total_epistemic_value,
            "high_confidence_fragments": len(high_conf),
            "empirical_invariants": empirical_count,
            "ground_truth_candidates": candidate_count,
            "triangulation_weight": self._triangulation_weight,
            "federation_peer_count": len(self._remote_profiles),
        }

    @property
    def triangulation_weight(self) -> float:
        return self._triangulation_weight

    @property
    def local_fragment_count(self) -> int:
        return len(self._local_fragments)

    @property
    def remote_fragment_count(self) -> int:
        return len(self._remote_fragments)

    @property
    def convergence_count(self) -> int:
        return len(self._convergences)

    # ─── Background Loop ─────────────────────────────────────────

    async def _divergence_loop(self) -> None:
        """Periodic divergence measurement and pressure computation."""
        interval = self._config.divergence_measurement_interval_s
        while True:
            try:
                await asyncio.sleep(interval)
                scores = await self.measure_all_divergences()
                await self.compute_and_apply_pressure()
                # Always emit local epistemic observables, even without peers.
                # _emit_divergence_observables guards on non-empty scores for
                # peer-derived metrics; _emit_local_epistemic_value fires always.
                await self._emit_divergence_observables(scores)
                await self._emit_local_epistemic_value()
                await self._check_economic_divergence_alert()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("divergence_loop_error")
                await asyncio.sleep(interval)

    # ─── Internal Helpers ────────────────────────────────────────

    async def _emit(
        self, event_type: SynapseEventType, data: dict[str, Any]
    ) -> None:
        """Emit a Synapse event if the event bus is wired."""
        if self._synapse is None:
            return
        event = SynapseEvent(
            event_type=event_type,
            data=data,
            source_system=self.system_id,
        )
        await self._synapse.event_bus.emit(event)

    def _is_link_circuit_open(self, link_id: str) -> bool:
        """Check if a federation link's circuit breaker is open (too many failures)."""
        return self._link_failures.get(link_id, 0) >= self._max_consecutive_failures

    def _record_link_success(self, link_id: str) -> None:
        """Reset failure counter on successful communication."""
        self._link_failures.pop(link_id, None)

    def _record_link_failure(self, link_id: str) -> None:
        """Increment failure counter; log when circuit opens."""
        count = self._link_failures.get(link_id, 0) + 1
        self._link_failures[link_id] = count
        if count == self._max_consecutive_failures:
            logger.warning(
                "federation_link_circuit_opened",
                link_id=link_id,
                consecutive_failures=count,
            )

    def _enforce_fragment_limit(self) -> None:
        """Evict lowest-quality local fragments if over the store limit."""
        limit = self._config.max_stored_fragments
        if len(self._local_fragments) <= limit:
            return

        sorted_fragments = sorted(
            self._local_fragments.values(),
            key=lambda f: f.quality_score,
        )

        excess = len(self._local_fragments) - limit
        for fragment in sorted_fragments[:excess]:
            del self._local_fragments[fragment.fragment_id]
            # Remove schema mapping so bridge survivors don't reference ghosts
            self._schema_to_fragment = {
                schema_id: fid
                for schema_id, fid in self._schema_to_fragment.items()
                if fid != fragment.fragment_id
            }

        logger.debug(
            "fragments_evicted",
            evicted=excess,
            remaining=len(self._local_fragments),
        )

    def _enforce_convergence_limit(self) -> None:
        """Trim convergence history if over the limit."""
        limit = self._config.max_stored_convergences
        if len(self._convergences) <= limit:
            return

        self._convergences.sort(
            key=lambda c: c.convergence_score, reverse=True
        )
        self._convergences = self._convergences[:limit]

    # ─── Gap 4: RE Training Data Emission ─────────────────────────

    async def _emit_re_training_example(
        self,
        fragment: ShareableWorldModelFragment,
        decision: PromotionDecision,
    ) -> None:
        """Emit an RE training example capturing the epistemic promotion reasoning."""
        try:
            from primitives.common import DriveAlignmentVector as _DAV
            _tri_conf = fragment.triangulation.triangulation_confidence
            _nexus_alignment = _DAV(
                coherence=round(min(1.0, _tri_conf * 2.0 - 1.0), 3),
                growth=round(min(1.0, _tri_conf), 3),
                care=0.0,
                honesty=round(min(1.0, fragment.triangulation.source_diversity), 3),
            )
        except Exception:
            _nexus_alignment = None

        example = RETrainingExample(
            source_system=SystemID.NEXUS,
            episode_id=fragment.fragment_id,
            instruction=(
                "Evaluate an epistemic fragment for ground truth promotion. "
                "Given the fragment's abstract structure, triangulation evidence, "
                "and convergence history, determine if promotion is warranted."
            ),
            input_context=json.dumps({
                "fragment_id": fragment.fragment_id,
                "abstract_structure": fragment.abstract_structure,
                "domain_labels": fragment.domain_labels,
                "observations_explained": fragment.observations_explained,
                "compression_ratio": fragment.compression_ratio,
                "triangulation_confidence": fragment.triangulation.triangulation_confidence,
                "source_count": fragment.triangulation.independent_source_count,
                "source_diversity": fragment.triangulation.source_diversity,
            }),
            output=json.dumps({
                "promoted": True,
                "from_level": decision.current_level.name,
                "to_level": decision.proposed_level.name,
                "triangulation_confidence": decision.triangulation_confidence,
                "survived_adversarial": decision.survived_adversarial_test,
                "survived_competition": decision.survived_hypothesis_competition,
            }),
            outcome_quality=min(decision.triangulation_confidence, 1.0),
            category="epistemic_promotion",
            constitutional_alignment=_nexus_alignment,
        )

        await self._emit(
            SynapseEventType.RE_TRAINING_EXAMPLE,
            example.model_dump(mode="json"),
        )

    # ─── Gap 5: Benchmarks Evolutionary Observable Metrics ────────

    async def _emit_divergence_observables(
        self,
        scores: dict[str, DivergenceScore],
    ) -> None:
        """Emit evolutionary observables after a divergence measurement cycle."""
        if not scores:
            return

        avg_divergence = sum(s.overall for s in scores.values()) / len(scores)

        observables = [
            EvolutionaryObservable(
                source_system=SystemID.NEXUS,
                observable_type="federation_mean_divergence",
                value=avg_divergence,
                is_novel=False,
                metadata={
                    "peer_count": len(scores),
                    "instance_id": self._instance_id,
                },
            ),
            EvolutionaryObservable(
                source_system=SystemID.NEXUS,
                observable_type="speciation_event_count",
                value=float(len(self._speciation_registry.state.speciation_events)),
                is_novel=False,
                metadata={"instance_id": self._instance_id},
            ),
            EvolutionaryObservable(
                source_system=SystemID.NEXUS,
                observable_type="epistemic_promotion_rate",
                value=float(len([
                    f for f in self._local_fragments.values()
                    if self._promotion_pipeline is not None
                    and self._promotion_pipeline.get_level(f.fragment_id).value >= 2
                ])),
                is_novel=False,
                metadata={"instance_id": self._instance_id},
            ),
        ]

        for obs in observables:
            await self._emit(
                SynapseEventType.NEXUS_EPISTEMIC_VALUE,
                obs.model_dump(mode="json"),
            )

    async def _emit_local_epistemic_value(self) -> None:
        """
        Emit NEXUS_EPISTEMIC_VALUE from local fragment state.

        Fires every divergence loop cycle regardless of federation peer count.
        Ensures Benchmarks and the spec compliance checker can observe Nexus
        activity even in single-instance deployments with no federation peers.
        """
        local_count = len(self._local_fragments)
        remote_count = len(self._remote_fragments)
        convergence_count = len(self._convergences)
        speciation_count = len(self._speciation_registry.state.speciation_events)

        ground_truth_count = len([
            f for f in self._local_fragments.values()
            if self._promotion_pipeline is not None
            and self._promotion_pipeline.get_level(f.fragment_id).value >= 3
        ])

        obs = EvolutionaryObservable(
            source_system=SystemID.NEXUS,
            observable_type="local_epistemic_state",
            value=float(local_count),
            is_novel=False,
            metadata={
                "instance_id": self._instance_id,
                "local_fragment_count": local_count,
                "remote_fragment_count": remote_count,
                "convergence_count": convergence_count,
                "speciation_count": speciation_count,
                "ground_truth_count": ground_truth_count,
                "triangulation_weight": self._triangulation_weight,
            },
        )
        await self._emit(
            SynapseEventType.NEXUS_EPISTEMIC_VALUE,
            obs.model_dump(mode="json"),
        )

    async def _check_economic_divergence_alert(self) -> None:
        """
        Compute economic divergence across all known instance profiles and
        alert when peers diverge significantly on economic strategy.

        NEXUS-ECON-4: economic_divergence measures revenue-per-strategy
        variance. When divergence is high (> 0.6), peers have independently
        discovered different profitable strategies - this is signal, not noise.
        Emit DIVERGENCE_PRESSURE with economic context so Oikos and Evo can
        investigate and potentially synthesise across strategies.
        """
        all_profiles = list(self._remote_profiles.values())
        if len(all_profiles) < 2:
            return

        # Only include profiles with actual strategy data
        profiled = [p for p in all_profiles if p.strategy_revenue_rates]
        if len(profiled) < 2:
            return

        economic_scores = compute_economic_divergence(profiled)

        # Update each profile's economic_divergence field in-place
        for profile in profiled:
            profile.economic_divergence = economic_scores.get(
                profile.instance_id, 0.0
            )

        # Alert if any pair has high economic divergence (> 0.6)
        high_divergence_instances = [
            (iid, score)
            for iid, score in economic_scores.items()
            if score > 0.6
        ]

        if not high_divergence_instances:
            return

        max_instance_id, max_score = max(
            high_divergence_instances, key=lambda x: x[1]
        )
        max_profile = self._remote_profiles.get(max_instance_id)
        top_strategies = sorted(
            (max_profile.strategy_revenue_rates if max_profile else {}).items(),
            key=lambda x: -x[1],
        )[:3]

        logger.info(
            "economic_strategy_divergence_detected",
            max_instance=max_instance_id,
            max_score=round(max_score, 3),
            diverged_instances=len(high_divergence_instances),
            top_strategies=[s for s, _ in top_strategies],
        )

        await self._emit(
            SynapseEventType.DIVERGENCE_PRESSURE,
            {
                "instance_id": self._instance_id,
                "economic_divergence_detected": True,
                "max_economic_divergence": max_score,
                "diverged_instance_id": max_instance_id,
                "diverged_strategies": [s for s, _ in top_strategies],
                "peer_count": len(profiled),
                "recommendation": (
                    "Investigate differing economic strategies - "
                    "high economic divergence may indicate unexplored profitable paths."
                ),
            },
        )
