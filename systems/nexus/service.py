"""
EcodiaOS — Nexus: Epistemic Triangulation Service

Orchestrates fragment extraction, convergence detection, divergence
measurement, incentive computation, speciation detection, invariant
bridge exchange, and ground truth promotion across the federation.

Nexus decides WHAT to share; Federation handles HOW.

Sub-components:
  ConvergenceDetector         — structural isomorphism comparison
  InstanceDivergenceMeasurer  — five-dimensional divergence scoring
  DivergenceIncentiveEngine   — triangulation weights + speciation pressure
  SpeciationDetector          — detects divergence >= 0.8 speciation events
  InvariantBridge             — cross-speciation causal invariant exchange
  SpeciationRegistry          — tracks cognitive kinds and bridge connections
  GroundTruthPromotionPipeline — Level 0-4 epistemic promotion

Synapse events emitted:
  FRAGMENT_SHARED               — a fragment was sent to the federation
  CONVERGENCE_DETECTED          — structural convergence found
  DIVERGENCE_PRESSURE           — speciation pressure generated
  TRIANGULATION_WEIGHT_UPDATE   — instance weight recalculated
  SPECIATION_EVENT              — two instances diverged beyond threshold
  GROUND_TRUTH_CANDIDATE        — fragment reached Level 3
  EMPIRICAL_INVARIANT_CONFIRMED — fragment reached Level 4 (constitutional)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.nexus.convergence import ConvergenceDetector
from systems.nexus.divergence import InstanceDivergenceMeasurer
from systems.nexus.ground_truth import GroundTruthPromotionPipeline
from systems.nexus.incentives import DivergenceIncentiveEngine
from systems.nexus.speciation import (
    InvariantBridge,
    SpeciationDetector,
    SpeciationRegistry,
    _strip_domain_context,
)
from systems.nexus.types import (
    CompressionPath,
    ConvergenceResult,
    DivergencePressure,
    DivergenceScore,
    EpistemicLevel,
    FragmentShareOutcome,
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
    from systems.nexus.protocols import (
        EquorProtectionProtocol,
        EvoCompetitionProtocol,
        EvoHypothesisSourceProtocol,
        FederationFragmentProtocol,
        FoveaAttentionProtocol,
        LogosWorldModelProtocol,
        OneirosAdversarialProtocol,
        ThymosDriveSinkProtocol,
    )
    from systems.synapse.service import SynapseService

logger = structlog.get_logger("nexus")


class NexusService:
    """
    Epistemic Triangulation across Federation.

    Instances don't share beliefs — they share the structure beneath
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
        self._convergence_detector = ConvergenceDetector()
        self._divergence_measurer: InstanceDivergenceMeasurer | None = None
        self._incentive_engine: DivergenceIncentiveEngine | None = None

        # Sub-components (Phase C — Speciation)
        self._speciation_detector = SpeciationDetector(config=self._config)
        self._invariant_bridge = InvariantBridge()
        self._speciation_registry = SpeciationRegistry()

        # Sub-component (Phase D — Ground Truth Promotion)
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
        self._telos: Any = None  # TelosService — for fragment score gating
        self._logos_adapter: Any = None  # LogosWorldModelAdapter — for write-back

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

        # Background tasks
        self._tasks: list[asyncio.Task[None]] = []

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

    def set_telos(self, telos: Any) -> None:
        """Wire Telos for topology-aware fragment scoring during federation."""
        self._telos = telos

    def set_equor(self, equor: EquorProtectionProtocol) -> None:
        """Wire Equor for constitutional protection of empirical invariants."""
        self._equor = equor

    def set_logos_adapter(self, logos_adapter: Any) -> None:
        """
        Wire the LogosWorldModelAdapter for convergence→Logos feedback.

        This is separate from set_world_model() which provides the
        LogosWorldModelProtocol for divergence measurement.  The adapter
        provides write-back methods (update_schema_triangulation_confidence,
        ingest_empirical_invariant_from_nexus) that the protocol does not.
        """
        self._logos_adapter = logos_adapter

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
        )

        self._promotion_pipeline = GroundTruthPromotionPipeline(
            config=self._config,
            speciation_registry=self._speciation_registry,
            oneiros=self._oneiros,
            evo=self._evo,
            equor=self._equor,
        )

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

            # Build relational skeleton from structural fields — strip all
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

            fragment = self.extract_fragment(
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

        if extracted:
            logger.info(
                "nexus_wake_initiated_fragments_extracted",
                count=len(extracted),
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
        logos_adapter = getattr(self, "_logos_adapter", None)
        if logos_adapter is not None:
            try:
                logos_adapter.ingest_empirical_invariant_from_nexus(
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
        logos_adapter = getattr(self, "_logos_adapter", None)
        if logos_adapter is None:
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
            logos_adapter.update_schema_triangulation_confidence(
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
        }

    # ─── Phase A: Fragment Operations ────────────────────────────

    def extract_fragment(
        self,
        schema_id: str,
        abstract_structure: dict[str, Any],
        domain_labels: list[str],
        observations_explained: int,
        description_length: float,
        compression_ratio: float,
        compression_path: CompressionPath | None = None,
        sleep_certification: SleepCertification | None = None,
    ) -> ShareableWorldModelFragment:
        """
        Extract a shareable fragment from a world model schema.

        The abstract_structure should have domain labels already stripped —
        only relational skeleton (nodes, edges, symmetry, invariants).
        """
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
            created_at=utc_now(),
            last_confirmed_at=utc_now(),
        )

        self._local_fragments[fragment.fragment_id] = fragment
        self._schema_to_fragment[schema_id] = fragment.fragment_id
        self._enforce_fragment_limit()

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
        telos = getattr(self, "_telos", None)
        if telos is not None:
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

        # Block post-speciation fragment sharing — only InvariantBridge remains.
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

        return WorldModelFragmentShareResponse(
            message_id=message.message_id,
            outcome=FragmentShareOutcome.ACCEPTED,
            convergence_detected=best_convergence is not None,
            convergence_score=(
                best_convergence.convergence_score if best_convergence else 0.0
            ),
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
            score = await self.measure_divergence(link_id)
            if score is not None:
                results[link_id] = score

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

        # Compute weight
        self._triangulation_weight = (
            self._incentive_engine.compute_triangulation_weight()
        )

        await self._emit(
            SynapseEventType.TRIANGULATION_WEIGHT_UPDATE,
            {
                "instance_id": self._instance_id,
                "new_weight": self._triangulation_weight,
                "peer_count": len(self._remote_profiles),
            },
        )

        # Compute pressure
        pressure = self._incentive_engine.compute_divergence_pressure(
            local_profile
        )

        if pressure is not None:
            self._incentive_engine.apply_pressure_to_thymos(pressure)

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
            # Remote Logos not directly available — metadata comes from score
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

        return report

    def is_fragment_sharing_allowed(
        self, remote_instance_id: str
    ) -> bool:
        """
        Check if normal fragment sharing is allowed with a remote instance.

        Post-speciation, normal sharing is blocked — only invariant bridge
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
                await self.measure_all_divergences()
                await self.compute_and_apply_pressure()
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
