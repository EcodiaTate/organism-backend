"""
EcodiaOS -- Logos: Universal Compression Engine (Service)

Cross-cutting compression layer.  Cognitive budget, MDL scoring,
four-stage compression cascade, entropic decay, Schwarzschild threshold
detection, and intelligence metric broadcasting.

Phases A-D:
  A: CognitiveBudget, MDLEstimator, pressure, IntelligenceMetrics
  B: HolographicEncoder, WorldModel, Memory integration
  C: CompressionCascade (4-stage), EntropicDecayEngine, Anchor memories
  D: SchwarzchildCognitionDetector (5 indicators, self-prediction loop)
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID, new_id, utc_now
from primitives.re_training import RETrainingExample
from systems.logos.budget import CognitiveBudgetManager
from systems.logos.cascade import CompressionCascade
from systems.logos.decay import EntropicDecayEngine, MemoryStoreProtocol
from systems.logos.persistence import LogosPersistence
from systems.logos.holographic import HolographicEncoder
from systems.logos.mdl import MDLEstimator
from systems.logos.schwarzschild import SchwarzchildCognitionDetector
from systems.logos.types import (
    CascadeResult,
    CompressionCycleReport,
    CompressionStage,
    EmpiricalInvariant,
    ExperienceDelta,
    GenerativeSchema,
    IntelligenceMetrics,
    LogosConfig,
    LogosFitnessRecord,
    MemoryTier,
    Prediction,
    RawExperience,
    SchwarzchildStatus,
    WorldModelUpdate,
)
from systems.logos.world_model import WorldModel
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.synapse.service import SynapseService

logger = structlog.get_logger("logos")


class LogosService:
    """
    The Universal Compression Engine.

    Orchestrates the cognitive budget, MDL scoring, four-stage compression
    cascade, entropic decay, Schwarzschild threshold detection, and
    intelligence metric broadcasting.

    Satisfies ManagedSystemProtocol, FoveaPredictionProtocol,
    TelosMetricsProtocol, OneirosCompressionHooks, KairosInvariantProtocol.
    """

    system_id: str = "logos"

    def __init__(
        self,
        *,
        config: LogosConfig | None = None,
    ) -> None:
        self._config = config or LogosConfig()
        self._initialized = False

        # Phase A: Budget + MDL
        self._budget = CognitiveBudgetManager(
            total_budget=self._config.total_budget_ku
        )
        self._world_model = WorldModel()
        self._mdl_estimator = MDLEstimator(world_model=self._world_model)

        # Phase B: Holographic encoder
        self._holographic_encoder = HolographicEncoder(
            self._world_model,
            discard_threshold=self._config.holographic_discard_threshold,
            update_threshold=self._config.world_model_update_threshold,
        )

        # Phase C: Compression cascade + entropic decay
        self._cascade = CompressionCascade(
            self._holographic_encoder,
            self._world_model,
            salience_threshold=self._config.episodic_salience_threshold,
            pattern_merge_threshold=self._config.pattern_merge_threshold,
        )
        self._decay_engine = EntropicDecayEngine(
            self._mdl_estimator,
            self._cascade,
            eviction_threshold=self._config.eviction_survival_threshold,
            reinforcement_threshold=self._config.reinforcement_survival_threshold,
            reinforcement_factor=self._config.reinforcement_factor,
            anchor_compression_threshold=self._config.anchor_compression_ratio_threshold,
            access_decay_rate=self._config.access_decay_rate,
            contradiction_decay_multiplier=self._config.contradiction_decay_multiplier,
        )

        # Phase D: Schwarzschild cognition detector
        self._schwarzschild = SchwarzchildCognitionDetector(
            self._world_model,
            threshold_self_prediction=self._config.schwarzschild_self_prediction,
            threshold_intelligence_ratio=self._config.schwarzschild_intelligence_ratio,
            threshold_hypothesis_ratio=self._config.schwarzschild_hypothesis_ratio,
            self_prediction_window=self._config.self_prediction_window,
        )
        self._schwarzschild_fired = False  # Fires once, ever
        self._schwarzschild_approaching_emitted = False  # Fires once when indicators cross 80%

        # External references (protocol-based DI)
        self._synapse: SynapseService | None = None
        self._memory_store: MemoryStoreProtocol | None = None
        self._persistence: LogosPersistence | None = None

        # Background tasks
        self._pressure_task: asyncio.Task[None] | None = None
        self._metrics_task: asyncio.Task[None] | None = None
        self._decay_task: asyncio.Task[None] | None = None
        self._schwarzschild_task: asyncio.Task[None] | None = None
        # A3: fire-and-forget anchor emit tasks, cancelled on shutdown
        self._fire_forget_tasks: set[asyncio.Task[None]] = set()

        # Latest metrics + schwarzschild snapshot
        self._latest_metrics = IntelligenceMetrics()
        self._latest_schwarzschild = SchwarzchildStatus()

        # SG3/SG4: instance ID for fitness time-series (set via set_instance_id)
        self._instance_id: str = "logos_default"

        # Anchor memory IDs (protected from eviction)
        self._anchor_ids: set[str] = set()

        # Sleep state (compression paused during Oneiros sleep)
        self._sleep_active: bool = False

        # Budget emergency debounce (max 1 per 30s)
        self._last_budget_emergency_at: float = 0.0

        # Contradiction tracking {item_id: count}
        self._contradiction_counts: dict[str, int] = {}

        # HIGH-4 / MEDIUM-7: theta clock cycle counter
        # decay runs every 100 cycles; self-prediction runs every 50 cycles
        self._theta_cycle_count: int = 0

    # ─── Lifecycle ───────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the Logos engine. Idempotent. Restores world model from Neo4j."""
        if self._initialized:
            return

        logger.info(
            "logos_initializing",
            total_budget=self._config.total_budget_ku,
            pressure_interval=self._config.pressure_broadcast_interval_s,
            metrics_interval=self._config.metrics_broadcast_interval_s,
        )

        # M3: Restore world model from Neo4j on startup
        if self._persistence is not None:
            try:
                restored = await self._persistence.restore_world_model(self._world_model)
                logger.info("logos_world_model_restored_from_neo4j", nodes=restored)
            except Exception as exc:
                logger.warning("logos_neo4j_restore_failed", error=str(exc))

        self._initialized = True
        logger.info("logos_initialized")

    async def start(self) -> None:
        """Start background loops. Call after Synapse is wired."""
        if self._synapse is None:
            logger.warning("logos_start_no_synapse", note="Broadcasting disabled")
            return

        self._pressure_task = asyncio.create_task(
            self._pressure_broadcast_loop(), name="logos_pressure_broadcast"
        )
        self._metrics_task = asyncio.create_task(
            self._metrics_broadcast_loop(), name="logos_metrics_broadcast"
        )
        # Phase C: decay engine loop
        self._decay_task = asyncio.create_task(
            self._decay_loop(), name="logos_decay"
        )
        # Phase D: schwarzschild measurement + self-prediction loop
        self._schwarzschild_task = asyncio.create_task(
            self._schwarzschild_loop(), name="logos_schwarzschild"
        )
        logger.info("logos_all_loops_started")

    async def shutdown(self) -> None:
        """Gracefully stop all background tasks."""
        tasks = (
            self._pressure_task,
            self._metrics_task,
            self._decay_task,
            self._schwarzschild_task,
        )
        for task in tasks:
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        # A3: cancel tracked fire-and-forget tasks (anchor emits, RE training emits)
        for ff_task in list(self._fire_forget_tasks):
            if not ff_task.done():
                ff_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ff_task
        self._fire_forget_tasks.clear()
        logger.info("logos_shutdown")

    async def health(self) -> dict[str, Any]:
        """Health check for Synapse monitoring."""
        return {
            "status": "healthy" if self._initialized else "stopped",
            "cognitive_pressure": self._budget.total_pressure,
            "compression_urgency": self._budget.compression_urgency,
            "intelligence_ratio": self._world_model.measure_intelligence_ratio(),
            "world_model_schemas": len(self._world_model.generative_schemas),
            "world_model_complexity_bits": self._world_model.current_complexity,
            "schwarzschild_met": self._schwarzschild.threshold_met,
            "anchor_memories": len(self._anchor_ids),
        }

    # ─── Setter Wiring (protocol-based DI) ───────────────────────

    def set_synapse(self, synapse: SynapseService) -> None:
        """Wire Synapse for event broadcasting and subscribe to inbound events."""
        self._synapse = synapse
        self._subscribe_all_inbound(synapse)
        logger.info("logos_synapse_wired")

    def set_memory_store(self, store: MemoryStoreProtocol) -> None:
        """Wire a MemoryStoreProtocol for the decay engine."""
        self._memory_store = store
        logger.info("logos_memory_store_wired")

    def set_neo4j(self, neo4j: Any) -> None:
        """Wire Neo4j for world model persistence (M3)."""
        self._persistence = LogosPersistence(neo4j)
        logger.info("logos_neo4j_wired")

    def set_instance_id(self, instance_id: str) -> None:
        """Set the EOS instance ID for fitness time-series tagging (SG3/SG4)."""
        self._instance_id = instance_id

    # ─── Inbound event subscriptions ─────────────────────────────

    def _subscribe_all_inbound(self, synapse: SynapseService) -> None:
        """Subscribe to all 9 inbound Synapse events (Spec 21 §M2).

        Previously only FOVEA_PREDICTION_ERROR was subscribed. This wires
        the full ecosystem: memory consolidation, hypothesis lifecycle,
        schema induction, Kairos invariants, sleep lifecycle, and mitosis.
        """
        subscriptions: list[tuple[SynapseEventType, Any]] = [
            (SynapseEventType.FOVEA_PREDICTION_ERROR, self._on_fovea_prediction_error),
            (SynapseEventType.MEMORY_CONSOLIDATED, self._on_memory_consolidated),
            (SynapseEventType.EVO_HYPOTHESIS_CONFIRMED, self._on_hypothesis_confirmed),
            (SynapseEventType.EVO_HYPOTHESIS_REFUTED, self._on_hypothesis_rejected),
            (SynapseEventType.SCHEMA_INDUCED, self._on_schema_induced),
            (SynapseEventType.KAIROS_TIER3_INVARIANT_DISCOVERED, self._on_kairos_tier3_invariant),
            (SynapseEventType.SLEEP_INITIATED, self._on_sleep_started),
            (SynapseEventType.WAKE_ONSET, self._on_sleep_complete),
            (SynapseEventType.INSTANCE_SPAWNED, self._on_instance_spawned),
            # Spec 21 / Spec 26: prune retired-instance world model data
            (SynapseEventType.INSTANCE_RETIRED, self._on_instance_retired),
            # Spec 09 / Spec 29: VitalityCoordinator austerity - pause compression
            (SynapseEventType.SYSTEM_MODULATION, self._on_system_modulation),
            # HIGH-4: theta clock - decay every 100 cycles, self-prediction every 50
            (SynapseEventType.THETA_CYCLE_START, self._on_theta_cycle),
        ]
        subscribed = 0
        for event_type, handler in subscriptions:
            try:
                synapse.event_bus.subscribe(event_type, handler)
                subscribed += 1
            except (ValueError, AttributeError):
                logger.debug(
                    "logos_subscription_skipped",
                    event_type=event_type.value,
                    reason="event_type_unavailable",
                )
        logger.info("logos_subscriptions_wired", count=subscribed)

    async def _on_fovea_prediction_error(self, event: Any) -> None:
        """Handle a FOVEA_PREDICTION_ERROR: feed the error back as a compression delta.

        Only high-salience errors justify a world model update - low-salience
        errors are noise. The threshold is set at 0.3 precision-weighted salience.
        """
        data = event.data if hasattr(event, "data") else {}
        salience = data.get("precision_weighted_salience", 0.0)

        if salience < 0.3:
            return

        # Build a RawExperience from the prediction error and push it through
        # the compression cascade.  This is the "prediction error IS the delta"
        # insight from the Integration Manifold.
        content = {
            "content_error": data.get("content_error", 0.0),
            "temporal_error": data.get("temporal_error", 0.0),
            "magnitude_error": data.get("magnitude_error", 0.0),
            "source_error": data.get("source_error", 0.0),
            "category_error": data.get("category_error", 0.0),
            "causal_error": data.get("causal_error", 0.0),
        }
        ctx = {
            "source": "fovea_prediction_error",
            "percept_id": data.get("percept_id", ""),
            "dominant_error_type": data.get("dominant_error_type", ""),
        }
        raw = RawExperience(
            context=ctx,
            content=content,
            # HIGH-2: use spec formula; salience * 100 was an underestimate
            raw_complexity=self._mdl_estimator.compute_raw_complexity({**ctx, **content}),
            source_system="fovea",
        )

        try:
            await self.process_experience(raw)
            logger.debug(
                "fovea_error_fed_to_cascade",
                salience=round(salience, 4),
                percept_id=data.get("percept_id", ""),
            )
        except Exception as exc:
            # A2 fix: log full error context so Thymos can classify the incident.
            # We do NOT re-raise - Synapse event handlers must not crash the bus.
            logger.error(
                "fovea_error_cascade_failed",
                error=str(exc),
                exc_type=type(exc).__name__,
                percept_id=data.get("percept_id", ""),
                salience=round(salience, 4),
                exc_info=True,
            )

    # ─── Inbound: MEMORY_CONSOLIDATED → rescore distilled items ──

    async def _on_memory_consolidated(self, event: Any) -> None:
        """Rescore distilled items after memory consolidation and update coverage."""
        data = event.data if hasattr(event, "data") else {}
        coverage_delta = data.get("coverage_delta", 0.0)
        consolidated_count = data.get("consolidated_count", 0)

        if consolidated_count > 0:
            # Consolidation freed capacity - update coverage tracking
            self._world_model.coverage = min(
                1.0, self._world_model.coverage + coverage_delta
            )
            logger.debug(
                "memory_consolidation_received",
                consolidated=consolidated_count,
                coverage_delta=coverage_delta,
            )

    # ─── Inbound: EVO_HYPOTHESIS_CONFIRMED → reinforce schema ──

    async def _on_hypothesis_confirmed(self, event: Any) -> None:
        """Increase MDL weight for confirmed hypothesis schemas."""
        data = event.data if hasattr(event, "data") else {}
        hypothesis_id = data.get("hypothesis_id", "")
        statement = data.get("statement", "")

        # Find and reinforce matching schemas in the world model
        for schema in self._world_model.generative_schemas.values():
            if statement and statement in schema.description:
                schema.instance_count += 1
                schema.last_instantiated = utc_now()
                logger.debug(
                    "hypothesis_confirmed_schema_reinforced",
                    schema_id=schema.id,
                    hypothesis_id=hypothesis_id,
                )
                break

    # ─── Inbound: EVO_HYPOTHESIS_REFUTED → contradiction decay ─

    async def _on_hypothesis_rejected(self, event: Any) -> None:
        """Trigger contradiction decay on items related to the refuted hypothesis."""
        data = event.data if hasattr(event, "data") else {}
        hypothesis_id = data.get("hypothesis_id", "")
        statement = data.get("statement", "")

        # Record contradictions on related items via decay engine
        if self._memory_store is not None and statement:
            try:
                items = await self._memory_store.get_all_scored_items()
                for item in items:
                    content_str = str(item.content)
                    if statement and any(
                        term in content_str
                        for term in statement.split()[:5]
                        if len(term) > 3
                    ):
                        self._decay_engine.record_contradiction(item.id)
                        logger.debug(
                            "hypothesis_rejected_contradiction_recorded",
                            item_id=item.id,
                            hypothesis_id=hypothesis_id,
                        )
            except Exception as exc:
                logger.warning(
                    "hypothesis_rejection_decay_failed",
                    error=str(exc),
                )

    # ─── Inbound: SCHEMA_INDUCED → score via MDL and integrate ─

    async def _on_schema_induced(self, event: Any) -> None:
        """Score a newly induced schema via MDL; integrate if MDL > 1.0."""
        data = event.data if hasattr(event, "data") else {}
        schema_id = data.get("schema_id", "")
        description = data.get("description", "")
        domain = data.get("domain", "general")
        instance_count = data.get("instance_count", 1)

        # Estimate MDL score: description length vs instances covered
        desc_length = len(description) * 4.5 if description else 100.0
        observation_complexity = instance_count * desc_length
        mdl_ratio = observation_complexity / max(desc_length, 1.0)

        if mdl_ratio > 1.0:
            schema = GenerativeSchema(
                id=schema_id or new_id(),
                name=f"induced_{len(self._world_model.generative_schemas)}",
                domain=domain,
                description=description,
                instance_count=instance_count,
                compression_ratio=mdl_ratio,
            )
            self._world_model.register_schema(schema)
            logger.info(
                "schema_induced_integrated",
                schema_id=schema.id,
                mdl_ratio=round(mdl_ratio, 2),
            )
        else:
            logger.debug(
                "schema_induced_rejected",
                schema_id=schema_id,
                mdl_ratio=round(mdl_ratio, 2),
            )

    # ─── Inbound: KAIROS_TIER3_INVARIANT → immediate integration ─

    async def _on_kairos_tier3_invariant(self, event: Any) -> None:
        """Immediately integrate Tier 3 substrate-independent invariants as high-confidence priors."""
        data = event.data if hasattr(event, "data") else {}
        invariant_id = data.get("invariant_id", "")
        abstract_form = data.get("abstract_form", "")
        hold_rate = data.get("hold_rate", 1.0)

        invariant = EmpiricalInvariant(
            id=invariant_id or new_id(),
            statement=abstract_form,
            domain="cross_domain",
            observation_count=data.get("domain_count", 0),
            confidence=hold_rate,
            source="kairos_tier3",
        )
        self._world_model.ingest_invariant(invariant)
        logger.info(
            "kairos_tier3_invariant_integrated",
            invariant_id=invariant.id,
            hold_rate=hold_rate,
        )

    # ─── Inbound: SLEEP_INITIATED → pause compression queue ────

    async def _on_sleep_started(self, event: Any) -> None:
        """Pause real-time compression queue during sleep (Oneiros takes over)."""
        self._sleep_active = True
        logger.info("logos_sleep_started_compression_paused")

    # ─── Inbound: WAKE_ONSET → resume compression queue ────────

    async def _on_sleep_complete(self, event: Any) -> None:
        """Resume compression queue after sleep; rescore sleep-touched items."""
        self._sleep_active = False
        logger.info("logos_sleep_complete_compression_resumed")

    # ─── Inbound: INSTANCE_SPAWNED → clone world model (M5) ───

    async def _on_instance_spawned(self, event: Any) -> None:
        """Clone world model snapshot for a spawned child instance."""
        data = event.data if hasattr(event, "data") else {}
        instance_id = data.get("instance_id", "")
        snap = self._world_model.snapshot()
        logger.info(
            "world_model_snapshot_for_child",
            child_instance_id=instance_id,
            schemas=len(snap.get("generative_schemas", {})),
            causal_links=len(snap.get("causal_links", {})),
        )
        # The snapshot is available for the Mitosis genome orchestrator to pick up.
        # In a full implementation, this would be written to a shared store or
        # emitted as an event payload.

    # ─── Inbound: INSTANCE_RETIRED → prune retired-instance data ──

    async def _on_instance_retired(self, event: Any) -> None:
        """Prune world model data that originated from a retired child instance.

        Spec 21 §M5 / Spec 26: when a child instance is retired, Logos should
        remove or decay any generative schemas or causal priors that were seeded
        exclusively from that instance's data - preventing stale genome inheritance
        from polluting the parent world model indefinitely.
        """
        data = event.data if hasattr(event, "data") else {}
        retired_instance_id = data.get("instance_id", "")
        if not retired_instance_id:
            return

        # Evict schemas whose sole source is the retired instance
        to_remove: list[str] = [
            sid
            for sid, schema in self._world_model.generative_schemas.items()
            if schema.source_system == retired_instance_id
        ]
        for sid in to_remove:
            self._world_model.generative_schemas.pop(sid, None)
            self._anchor_ids.discard(sid)

        logger.info(
            "logos_instance_retired_pruned",
            retired_instance_id=retired_instance_id,
            schemas_removed=len(to_remove),
        )

    # ─── Inbound: SYSTEM_MODULATION → austerity compliance ────────

    async def _on_system_modulation(self, event: Any) -> None:
        """Respond to VitalityCoordinator austerity directives.

        Spec 09 / Spec 29: SYSTEM_MODULATION carries halt_systems / preserve_systems
        lists. If logos is in halt_systems, pause compression (same as sleep gate).
        Always emit SYSTEM_MODULATION_ACK so VitalityCoordinator can track compliance.
        """
        data = event.data if hasattr(event, "data") else {}
        halt_systems: list[str] = data.get("halt_systems", [])
        level: str = data.get("level", "nominal")

        compliant = False
        if "logos" in halt_systems or level in ("safe_mode", "emergency"):
            self._sleep_active = True  # reuse sleep gate - no new compressions
            compliant = True
            logger.warning(
                "logos_modulation_halted",
                level=level,
                halt_systems=halt_systems,
            )
        elif not halt_systems and level == "nominal":
            # Recovery - resume if we were previously halted by modulation
            self._sleep_active = False
            compliant = True

        # Emit ACK so VitalityCoordinator knows Logos heard the directive
        if self._synapse is not None:
            try:
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                    data={
                        "system_id": "logos",
                        "level": level,
                        "compliant": compliant,
                        "reason": "compression_paused" if compliant and "logos" in halt_systems else None,
                    },
                ))
            except Exception as exc:
                logger.warning("logos_modulation_ack_failed", error=str(exc))

    # ─── Inbound: THETA_CYCLE_START → decay every 100, self-pred every 50 ─

    async def _on_theta_cycle(self, event: Any) -> None:
        """HIGH-4 / MEDIUM-7: driven by the Synapse theta clock.

        - Every 100 cycles: trigger a decay pass on low-salience nodes
          (decay engine on memory_store if wired; skip gracefully if not).
        - Every 50 cycles: run a Schwarzschild self-prediction cycle and
          store self_prediction_accuracy for Benchmarks.
        """
        self._theta_cycle_count += 1

        # Every 50 cycles: Schwarzschild self-prediction (MEDIUM-7)
        if self._theta_cycle_count % 50 == 0:
            try:
                record = await self._schwarzschild.run_self_prediction_cycle()
                # Store accuracy in latest status so Benchmarks can poll it
                # via health() or via INTELLIGENCE_METRICS broadcast
                self._latest_schwarzschild = SchwarzchildStatus(
                    self_prediction_accuracy=self._schwarzschild._compute_self_prediction_accuracy(),
                    intelligence_ratio=self._world_model.measure_intelligence_ratio(),
                    hypothesis_ratio=self._schwarzschild._compute_generative_surplus(),
                    novel_concept_rate=float(len(self._schwarzschild._novel_schema_ids)),
                    cross_domain_transfers=len(self._schwarzschild._cross_domain_transfers),
                    compression_acceleration=self._schwarzschild._compute_compression_velocity(),
                    novel_structures=len(self._schwarzschild._novel_schema_ids),
                    threshold_met=self._schwarzschild.threshold_met,
                )
                logger.debug(
                    "theta_self_prediction_cycle",
                    cycle=self._theta_cycle_count,
                    accuracy=record.accuracy,
                    self_pred_rolling=self._latest_schwarzschild.self_prediction_accuracy,
                )
            except Exception as exc:
                logger.warning("theta_self_prediction_failed", error=str(exc))

        # Every 100 cycles: decay pass (HIGH-4)
        if self._theta_cycle_count % 100 == 0:
            if self._memory_store is not None and not self._sleep_active:
                try:
                    await self._decay_engine.run_decay_cycle(
                        self._memory_store,
                        self._anchor_ids,
                        max_items=50,  # Light pass - theta-driven, not pressure-driven
                    )
                    logger.debug(
                        "theta_decay_cycle_complete",
                        cycle=self._theta_cycle_count,
                    )
                except Exception as exc:
                    logger.warning("theta_decay_cycle_failed", error=str(exc))

    # ─── Public API: Budget ──────────────────────────────────────

    @property
    def budget(self) -> CognitiveBudgetManager:
        return self._budget

    @property
    def world_model(self) -> WorldModel:
        return self._world_model

    @property
    def mdl_estimator(self) -> MDLEstimator:
        return self._mdl_estimator

    @property
    def holographic_encoder(self) -> HolographicEncoder:
        return self._holographic_encoder

    @property
    def cascade(self) -> CompressionCascade:
        return self._cascade

    @property
    def decay_engine(self) -> EntropicDecayEngine:
        return self._decay_engine

    @property
    def schwarzschild_detector(self) -> SchwarzchildCognitionDetector:
        return self._schwarzschild

    @property
    def latest_schwarzschild(self) -> SchwarzchildStatus:
        return self._latest_schwarzschild

    def update_tier_utilization(self, tier: MemoryTier, count: float) -> None:
        """Called by Memory/Evo to report current utilization per tier."""
        self._budget.update_utilization(tier, count)

    def try_admit(self, tier: MemoryTier, amount: float = 1.0) -> bool:
        """
        Compression-first memory admission.
        Nothing enters long-term memory without budget approval.
        Emits LOGOS_BUDGET_ADMISSION_DENIED when the tier ceiling is breached.
        """
        admitted = self._budget.increment(tier, amount)
        if not admitted and self._synapse is not None:
            asyncio.create_task(  # noqa: RUF006 - fire-and-forget, tracked via _fire_forget_tasks
                self._emit_budget_admission_denied(tier, amount)
            )
        return admitted

    async def _emit_budget_admission_denied(
        self, tier: MemoryTier, amount: float
    ) -> None:
        """Emit LOGOS_BUDGET_ADMISSION_DENIED when a knowledge unit is rejected."""
        if self._synapse is None:
            return
        state = self._budget.current_utilization_state
        tier_used = state.get(tier.value, 0.0)
        tier_limit = self._budget.state.tier_budget(tier)
        with contextlib.suppress(Exception):
            await self._synapse.event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.LOGOS_BUDGET_ADMISSION_DENIED,
                    source_system=SystemID.LOGOS,
                    data={
                        "tier": tier.value,
                        "requested_ku": amount,
                        "tier_used_ku": tier_used,
                        "tier_limit_ku": tier_limit,
                        "tier_utilization_pct": (tier_used / tier_limit * 100) if tier_limit > 0 else 100.0,
                        "total_pressure": self._budget.total_pressure,
                        "compression_urgency": self._budget.compression_urgency,
                        "recommendation": "compression_required" if self._budget.needs_compression() else "tier_rebalance",
                    },
                )
            )

    def release(self, tier: MemoryTier, amount: float = 1.0) -> None:
        """Release budget on eviction or compression."""
        self._budget.decrement(tier, amount)

    # ─── Public API: Holographic Encoding (Phase B) ──────────────

    async def encode_experience(
        self, raw_experience: RawExperience
    ) -> ExperienceDelta:
        """
        Holographically encode a raw experience.

        This is the entry point for all new experiences entering the
        compression cascade. The encoder computes the delta between
        what the world model predicted and what actually happened.
        """
        return await self._holographic_encoder.encode(raw_experience)

    async def integrate_delta(self, delta: ExperienceDelta) -> WorldModelUpdate:
        """
        Integrate an experience delta into the world model.

        Called by external systems (e.g. Oneiros) after holographic encoding.
        Emits WORLD_MODEL_UPDATED when the delta produces a structural change
        (not just a redundant/discard event).
        """
        update = await self._world_model.integrate(delta)

        # Only broadcast structural changes - not redundant-discard no-ops
        if self._synapse is not None and not delta.discard_after_encoding:
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.WORLD_MODEL_UPDATED,
                source_system=self.system_id,
                data={
                    "update_type": update.update_type.value,
                    "schemas_added": update.schemas_added,
                    "priors_updated": update.priors_updated,
                    "causal_updates": update.causal_links_added + update.causal_links_revised,
                    "coverage_delta": update.coverage_delta,
                    "complexity_delta": update.complexity_delta,
                },
            ))

        # CRITICAL-1: Persist (:WorldModel) node + [:COMPRESSES] links to Neo4j.
        # Source IDs come from the delta's experience_id (episode) - the cascade
        # sets this when a salient episode reaches Stage 4.
        if self._persistence is not None and not delta.discard_after_encoding:
            try:
                source_ep_ids = [delta.experience_id] if delta.experience_id else None
                await self._persistence.persist_integration(
                    update, delta,
                    source_episode_ids=source_ep_ids,
                )
            except Exception as exc:
                logger.warning("world_model_integration_persist_failed", error=str(exc))

            # M3: Batch-persist full world model snapshot
            await self._persist_world_model_if_wired()

        return update

    async def process_experience(
        self, raw_experience: RawExperience
    ) -> CascadeResult:
        """
        Run a raw experience through the full four-stage compression
        cascade.  This is the primary entry point for all new experiences.

        The cascade: holographic encoding -> episodic compression ->
        semantic distillation -> world model integration.

        During Oneiros sleep (_sleep_active=True), real-time compression is
        paused - Oneiros drives offline compression via run_batch_compression().
        Experiences received during sleep are discarded as redundant (not stored).

        Returns the CascadeResult with per-stage metrics, anchor flags,
        and compression ratios.
        """
        if self._sleep_active:
            logger.debug(
                "logos_experience_skipped_during_sleep",
                source=raw_experience.source_system,
            )
            return CascadeResult(
                experience_id=raw_experience.id,
                stage_reached=CompressionStage.HOLOGRAPHIC_ENCODING,
                is_irreducible=False,
                anchor_memory=False,
                compression_ratio=0.0,
            )

        result = await self._cascade.run(raw_experience)

        # Feed compression ratio to Schwarzschild detector
        if result.compression_ratio > 0:
            self._schwarzschild.record_compression_ratio(result.compression_ratio)
            self._schwarzschild.record_data_arrival()

        # HIGH-3: Update cognitive budget utilization with this compression's cost/gain.
        # cost_ku = bits consumed by the new compressed form (world_model_complexity delta)
        # gain_ku = bits freed from source material (bits_saved)
        if result.world_model_update is not None:
            wm_update = result.world_model_update
            cost_ku = max(wm_update.complexity_delta, 0.0)
            gain_ku = max(result.bits_saved, 0.0)
            self._budget.record_compression_operation(
                cost_ku=cost_ku,
                gain_ku=gain_ku,
                tier=MemoryTier.WORLD_MODEL,
            )

        # Track hypothesis generation from world model updates
        if result.world_model_update is not None:
            wm = result.world_model_update
            if wm.schemas_added > 0:
                self._schwarzschild.record_hypothesis_generated(wm.schemas_added)

        # Broadcast WORLD_MODEL_UPDATED if integration occurred
        if result.world_model_update is not None:
            wm = result.world_model_update
            if self._synapse is not None:
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.WORLD_MODEL_UPDATED,
                    source_system=self.system_id,
                    data={
                        "update_type": wm.update_type.value,
                        "schemas_added": wm.schemas_added,
                        "priors_updated": wm.priors_updated,
                        "causal_updates": wm.causal_links_added + wm.causal_links_revised,
                        "coverage_delta": wm.coverage_delta,
                        "complexity_delta": wm.complexity_delta,
                        # Surfaced for Telos/Kairos/Equor - was previously invisible
                        "invariants_tested": wm.invariants_tested,
                        "invariants_violated": wm.invariants_violated,
                    },
                ))

                # Emit dedicated invariant violation event so Kairos/Equor/Thymos react.
                # Previously only a WARNING log - now escalated to the organism.
                if wm.invariants_violated > 0:
                    try:
                        await self._synapse.event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.LOGOS_INVARIANT_VIOLATED,
                            source_system=self.system_id,
                            data={
                                "invariants_violated": wm.invariants_violated,
                                "invariants_tested": wm.invariants_tested,
                                "experience_id": raw_experience.id,
                                "update_type": wm.update_type.value,
                                "recommendation": (
                                    "Kairos should re-validate affected invariants. "
                                    "Equor should review if violated invariants are constitutional."
                                ),
                            },
                        ))
                    except (ValueError, AttributeError):
                        pass  # Event type may not be registered in older deployments

            # CRITICAL-1: Persist (:WorldModel) node + [:COMPRESSES] links
            if self._persistence is not None and result.delta is not None:
                try:
                    source_ep_ids = (
                        [result.delta.experience_id] if result.delta.experience_id else None
                    )
                    await self._persistence.persist_integration(
                        wm, result.delta,
                        source_episode_ids=source_ep_ids,
                    )
                except Exception as exc:
                    logger.warning(
                        "world_model_integration_persist_failed",
                        error=str(exc),
                        experience_id=raw_experience.id,
                    )

            # M3: Batch-persist world model snapshot to Neo4j
            await self._persist_world_model_if_wired()

        # Handle anchor memories
        if result.anchor_memory and result.compressed_item_id:
            self.mark_anchor(result.compressed_item_id)

        # SG5: Emit RE training example for high-MDL compressions
        if result.compression_ratio > 2.0 and result.world_model_update is not None:
            wm = result.world_model_update
            self._emit_re_training_example(
                instruction="Compress experience into world model update",
                input_context=f"source={raw_experience.source_system} complexity={raw_experience.raw_complexity}",
                output=f"compression_ratio={result.compression_ratio:.2f} schemas_added={wm.schemas_added} coverage_delta={wm.coverage_delta:.4f}",
                category="compression_reasoning",
                quality=min(result.compression_ratio / 10.0, 1.0),
            )

        return result

    # ─── FoveaPredictionProtocol ─────────────────────────────────

    async def predict(self, context: dict[str, Any]) -> Prediction:
        """Generate a world model prediction for Fovea."""
        return await self._world_model.predict(context)

    def get_historical_accuracy(self, domain: str | None = None) -> float:
        """Historical prediction accuracy for Fovea calibration."""
        return self._world_model.get_historical_accuracy(domain)

    def get_context_stability_age(self, context_key: str) -> float:
        """Context prior stability age for Fovea confidence."""
        return self._world_model.get_context_stability_age(context_key)

    # ─── TelosMetricsProtocol ────────────────────────────────────

    def get_intelligence_ratio(self) -> float:
        """I = K(reality_modeled) / K(model)."""
        return self._world_model.measure_intelligence_ratio()

    def get_generative_schemas(self) -> dict[str, Any]:
        """Return the world model's generative schema registry (for Oneiros)."""
        return self._world_model.generative_schemas

    def get_causal_structure(self) -> Any:
        """Return the world model's causal graph structure (for Oneiros)."""
        return self._world_model.causal_structure

    def get_current_complexity(self) -> float:
        """Return current world model complexity in MDL bits (for Oneiros)."""
        return self._world_model.current_complexity

    def get_compression_stats(self) -> dict[str, float]:
        """Compression statistics for Telos drive modulation."""
        return {
            "cognitive_pressure": self._budget.total_pressure,
            "compression_urgency": self._budget.compression_urgency,
            "compression_efficiency": self._latest_metrics.compression_efficiency,
            "world_model_coverage": self._world_model.coverage,
            "world_model_complexity": self._world_model.current_complexity,
        }

    def get_latest_metrics(self) -> IntelligenceMetrics:
        """Full intelligence metrics snapshot."""
        return self._latest_metrics

    # ─── OneirosCompressionHooks ─────────────────────────────────

    async def run_batch_compression(
        self,
        *,
        force: bool = False,
        max_items: int = 100,
    ) -> CompressionCycleReport:
        """
        Run a batch compression cycle via the entropic decay engine.

        During Oneiros sleep, called with force=True to bypass
        the normal pressure threshold check.
        """
        start = time.monotonic()

        if not force and not self._budget.needs_compression():
            return CompressionCycleReport(
                items_processed=0,
                cycle_duration_ms=(time.monotonic() - start) * 1000,
            )

        # Run the decay engine if a memory store is wired
        decay_report = None
        if self._memory_store is not None:
            decay_report = await self._decay_engine.run_decay_cycle(
                self._memory_store,
                self._anchor_ids,
                max_items=max_items,
            )

        # Build the compression cycle report
        elapsed = (time.monotonic() - start) * 1000
        # MDL improvement: ratio of bits freed to world model complexity.
        # Freeing 1% of complexity worth of redundant bits = 0.01 improvement.
        bits_freed = decay_report.total_bits_freed if decay_report else 0.0
        wm_complexity = max(self._world_model.current_complexity, 1.0)
        mdl_improvement = bits_freed / wm_complexity

        report = CompressionCycleReport(
            items_processed=decay_report.total_items_scanned if decay_report else 0,
            items_evicted=len(decay_report.evicted) if decay_report else 0,
            items_distilled=len(decay_report.distilled) if decay_report else 0,
            items_reinforced=len(decay_report.reinforced) if decay_report else 0,
            anchors_created=sum(
                1 for r in (decay_report.reinforced if decay_report else [])
                if r in self._anchor_ids
            ),
            bits_saved=bits_freed,
            mdl_improvement=mdl_improvement,
            cycle_duration_ms=elapsed,
        )

        # Release budget for evicted items - decrement the correct tier (P7 fix)
        if decay_report:
            for item_id in decay_report.evicted:
                tier = self._item_type_to_tier(
                    decay_report.evicted_item_types.get(item_id, "")
                )
                self._budget.decrement(tier, 1.0)

        # Broadcast COMPRESSION_CYCLE_COMPLETE with real data
        if self._synapse is not None:
            # Surface evicted item IDs (previously only aggregate count was visible).
            # Organism (Memory/Thread/Kairos) can now react to specific losses.
            evicted_ids: list[str] = []
            evicted_with_types: list[dict[str, str]] = []
            if decay_report is not None:
                evicted_ids = list(decay_report.evicted)[:20]  # cap at 20 for event size
                evicted_with_types = [
                    {
                        "id": item_id,
                        "type": decay_report.evicted_item_types.get(item_id, "unknown"),
                    }
                    for item_id in evicted_ids
                ]
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.COMPRESSION_CYCLE_COMPLETE,
                source_system=self.system_id,
                data={
                    "items_evicted": report.items_evicted,
                    "items_distilled": report.items_distilled,
                    "mdl_improvement": report.mdl_improvement,
                    "bits_saved": report.bits_saved,
                    "anchors_created": report.anchors_created,
                    # Newly surfaced - was previously invisible to organism
                    "evicted_item_ids": evicted_ids,
                    "evicted_items": evicted_with_types,
                    "total_evicted_this_cycle": len(evicted_ids),
                    "eviction_truncated": report.items_evicted > 20,
                },
            ))

        return report

    # ─── KairosInvariantProtocol ─────────────────────────────────

    def ingest_invariant(self, invariant: EmpiricalInvariant) -> None:
        """Ingest a causal invariant from Kairos into the world model."""
        self._world_model.ingest_invariant(invariant)

    # ─── Anchor Memory Management ────────────────────────────────

    def mark_anchor(
        self,
        item_id: str,
        *,
        information_content: float = 1.0,
        domain: str = "general",
        reason: str = "irreducible_novelty",
    ) -> None:
        """Mark an item as an anchor memory (never evicted)."""
        self._anchor_ids.add(item_id)
        logger.info(
            "anchor_memory_marked",
            item_id=item_id,
            info_content=information_content,
            domain=domain,
            reason=reason,
        )

        if self._synapse is not None:
            task = asyncio.create_task(
                self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.ANCHOR_MEMORY_CREATED,
                    source_system=self.system_id,
                    data={
                        "memory_id": item_id,
                        "information_content": information_content,
                        "domain": domain,
                        "reason": reason,
                    },
                )),
                name=f"logos_anchor_{item_id[:8]}",
            )
            # A3 fix: register in managed set so shutdown() can cancel it
            self._fire_forget_tasks.add(task)

            def _on_anchor_done(t: asyncio.Task[None]) -> None:
                self._fire_forget_tasks.discard(t)
                if not t.cancelled() and t.exception() is not None:
                    logger.warning("anchor_emit_failed", error=str(t.exception()))

            task.add_done_callback(_on_anchor_done)

        # SG5: Emit RE training example for anchor memories
        self._emit_re_training_example(
            instruction="Identify irreducibly novel information",
            input_context=f"domain={domain} info_content={information_content}",
            output=f"anchor_memory_created: {item_id}",
            category="compression_reasoning",
            quality=min(information_content, 1.0),
        )

    def is_anchor(self, item_id: str) -> bool:
        return item_id in self._anchor_ids

    # ─── Background Loops ────────────────────────────────────────

    async def _pressure_broadcast_loop(self) -> None:
        """Broadcast COGNITIVE_PRESSURE every 30s. Emit BUDGET_EMERGENCY when >= 0.90."""
        while True:
            try:
                await asyncio.sleep(self._config.pressure_broadcast_interval_s)
                if self._synapse is None:
                    continue

                payload = self._budget.pressure_payload()
                # Add tier_utilization (P1) and raw utilization (HIGH-3)
                tier_utilization: dict[str, float] = {}
                for tier in MemoryTier:
                    tier_utilization[tier.value] = self._budget.state.tier_pressure(tier)
                payload["tier_utilization"] = tier_utilization
                payload["current_utilization"] = self._budget.current_utilization_state

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.COGNITIVE_PRESSURE,
                    source_system=self.system_id,
                    data=payload,
                ))

                # M1: Emit BUDGET_EMERGENCY when utilization >= 0.90 (debounced 30s)
                if self._budget.is_emergency():
                    now = time.monotonic()
                    if now - self._last_budget_emergency_at >= 30.0:
                        self._last_budget_emergency_at = now
                        tier_overages: dict[str, float] = {
                            tier.value: self._budget.state.tier_pressure(tier)
                            for tier in MemoryTier
                            if self._budget.state.tier_pressure(tier) > 0.90
                        }
                        await self._synapse.event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.BUDGET_EMERGENCY,
                            source_system=self.system_id,
                            data={
                                "utilization": self._budget.total_pressure,
                                "tier_overages": tier_overages,
                                "recommended_action": "emergency_compression",
                            },
                        ))
                        logger.warning(
                            "budget_emergency_emitted",
                            utilization=self._budget.total_pressure,
                        )

                    # SG1: Trigger immediate eviction at critical boundary
                    if self._budget.is_critical():
                        await self._trigger_critical_eviction()

                logger.debug(
                    "cognitive_pressure_broadcast",
                    pressure=payload["pressure"],
                    urgency=payload["urgency"],
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("pressure_broadcast_error", error=str(exc))

    async def _metrics_broadcast_loop(self) -> None:
        """Broadcast INTELLIGENCE_METRICS every 60s."""
        previous_metrics: IntelligenceMetrics | None = None

        while True:
            try:
                await asyncio.sleep(self._config.metrics_broadcast_interval_s)
                if self._synapse is None:
                    continue

                metrics = self._compute_intelligence_metrics(previous_metrics)
                self._latest_metrics = metrics
                previous_metrics = metrics

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.INTELLIGENCE_METRICS,
                    source_system=self.system_id,
                    data=metrics.model_dump(),
                ))

                # SG3/SG4: append immutable fitness record to Neo4j time-series
                if self._persistence is not None:
                    fitness = LogosFitnessRecord(
                        instance_id=self._instance_id,
                        intelligence_ratio=metrics.intelligence_ratio,
                        compression_efficiency=metrics.compression_efficiency,
                        world_model_coverage=metrics.world_model_coverage,
                        cognitive_pressure=metrics.cognitive_pressure,
                        schema_count=len(self._world_model.generative_schemas),
                        anchor_count=len(self._anchor_ids),
                        schwarzschild_met=metrics.schwarzschild_threshold_met,
                    )
                    try:
                        await self._persistence.persist_fitness_record(fitness)
                    except Exception as fit_exc:
                        logger.warning(
                            "logos_fitness_persist_failed", error=str(fit_exc)
                        )

                logger.debug(
                    "intelligence_metrics_broadcast",
                    intelligence_ratio=metrics.intelligence_ratio,
                    cognitive_pressure=metrics.cognitive_pressure,
                    prediction_accuracy=metrics.prediction_accuracy,
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("metrics_broadcast_error", error=str(exc))

    async def _decay_loop(self) -> None:
        """Run entropic decay on a configurable schedule."""
        while True:
            try:
                await asyncio.sleep(self._config.decay_cycle_interval_s)
                await self.run_batch_compression()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("decay_loop_error", error=str(exc))

    async def _schwarzschild_loop(self) -> None:
        """Run Schwarzschild measurement + self-prediction periodically."""
        while True:
            try:
                await asyncio.sleep(
                    self._config.schwarzschild_measurement_interval_s
                )

                # Self-prediction cycle
                await self._schwarzschild.run_self_prediction_cycle()

                # Cross-domain transfer detection
                self._schwarzschild.detect_cross_domain_transfers()

                # Novel structure detection
                self._schwarzschild.detect_novel_structures()

                # Full measurement
                status = await self._schwarzschild.measure()
                self._latest_schwarzschild = status

                # Progressive warning: emit LOGOS_SCHWARZSCHILD_APPROACHING when any
                # indicator crosses 80% of its threshold. Gives the organism foresight
                # rather than a binary surprise. Fires once per instance lifetime.
                if not self._schwarzschild_approaching_emitted and not self._schwarzschild_fired:
                    ir_thresh = self._config.schwarzschild_intelligence_ratio
                    sp_thresh = self._config.schwarzschild_self_prediction
                    hr_thresh = self._config.schwarzschild_hypothesis_ratio
                    ir_frac = status.intelligence_ratio / max(ir_thresh, 1e-9)
                    sp_frac = status.self_prediction_accuracy / max(sp_thresh, 1e-9)
                    hr_frac = status.hypothesis_ratio / max(hr_thresh, 1e-9)
                    closest_frac = max(ir_frac, sp_frac, hr_frac)
                    if closest_frac >= 0.80 and self._synapse is not None:
                        self._schwarzschild_approaching_emitted = True
                        closest = (
                            "intelligence_ratio" if ir_frac == closest_frac
                            else ("self_prediction" if sp_frac == closest_frac else "hypothesis_ratio")
                        )
                        try:
                            await self._synapse.event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.LOGOS_SCHWARZSCHILD_APPROACHING,
                                source_system=self.system_id,
                                data={
                                    "intelligence_ratio": status.intelligence_ratio,
                                    "self_prediction_accuracy": status.self_prediction_accuracy,
                                    "hypothesis_ratio": status.hypothesis_ratio,
                                    "closest_indicator": closest,
                                    "fraction_to_threshold": round(closest_frac, 3),
                                    "message": (
                                        "Schwarzschild threshold approaching - cognitive reorganization "
                                        f"is imminent. Closest indicator: {closest} at "
                                        f"{round(closest_frac * 100, 1)}% of threshold."
                                    ),
                                },
                            ))
                        except (ValueError, AttributeError):
                            pass
                        logger.info(
                            "schwarzschild_approaching",
                            closest_indicator=closest,
                            fraction=round(closest_frac, 3),
                        )

                # Fire SCHWARZSCHILD_THRESHOLD_MET once, ever
                if status.threshold_met and not self._schwarzschild_fired:
                    self._schwarzschild_fired = True
                    if self._synapse is not None:
                        await self._synapse.event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.SCHWARZSCHILD_THRESHOLD_MET,
                            source_system=self.system_id,
                            data={
                                "timestamp": status.measured_at.isoformat(),
                                "intelligence_ratio": status.intelligence_ratio,
                                "self_prediction_accuracy": status.self_prediction_accuracy,
                                "hypothesis_ratio": status.hypothesis_ratio,
                                "cross_domain_transfers": status.cross_domain_transfers,
                                "novel_structures": status.novel_structures,
                                "compression_acceleration": status.compression_acceleration,
                            },
                        ))
                    logger.critical(
                        "schwarzschild_event_fired",
                        intelligence_ratio=status.intelligence_ratio,
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("schwarzschild_loop_error", error=str(exc))

    def _emit_re_training_example(
        self,
        *,
        instruction: str,
        input_context: str,
        output: str,
        category: str = "compression_reasoning",
        quality: float = 0.8,
    ) -> None:
        """SG5: Emit an RE training example for high-value compression events."""
        if self._synapse is None:
            return

        example = RETrainingExample(
            source_system=SystemID.LOGOS,
            instruction=instruction,
            input_context=input_context,
            output=output,
            outcome_quality=quality,
            category=category,
        )

        async def _emit() -> None:
            if self._synapse is not None:
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                    source_system=self.system_id,
                    data=example.model_dump(mode="json"),
                ))

        # A3 fix: register in managed set so shutdown() can cancel it
        re_task = asyncio.create_task(_emit(), name="logos_re_training")
        self._fire_forget_tasks.add(re_task)
        re_task.add_done_callback(self._fire_forget_tasks.discard)

    async def _persist_world_model_if_wired(self) -> None:
        """M3: Batch-persist world model to Neo4j if persistence is wired."""
        if self._persistence is None:
            return
        try:
            await self._persistence.persist_world_model(self._world_model)
        except Exception as exc:
            logger.warning("neo4j_persist_failed", error=str(exc))

    async def _trigger_critical_eviction(self) -> None:
        """SG1: Synchronous eviction when critical boundary (>= 0.95) crossed.

        Instead of waiting for the 300s decay timer, trigger immediate eviction.
        Eviction is logged irreversibly (SG2).
        """
        if self._memory_store is None:
            return

        report = await self._decay_engine.run_decay_cycle(
            self._memory_store,
            self._anchor_ids,
            max_items=50,  # Smaller batch for synchronous path
        )

        if report.evicted:
            # P7: decrement the correct tier per item
            for item_id in report.evicted:
                tier = self._item_type_to_tier(
                    report.evicted_item_types.get(item_id, "")
                )
                self._budget.decrement(tier, 1.0)

            # SG2: Immutable audit log of eviction
            if self._synapse is not None:
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.COMPRESSION_CYCLE_COMPLETE,
                    source_system=self.system_id,
                    data={
                        "items_evicted": len(report.evicted),
                        "items_distilled": len(report.distilled),
                        "trigger": "critical_eviction",
                        "bits_freed": report.total_bits_freed,
                    },
                ))
            logger.warning(
                "critical_eviction_triggered",
                evicted=len(report.evicted),
                bits_freed=report.total_bits_freed,
            )

    @staticmethod
    def _item_type_to_tier(item_type: str) -> MemoryTier:
        """
        Map a KnowledgeItemType string to the corresponding MemoryTier for budget
        decrement.  Defaults to EPISODIC when the type is unrecognised (P7 fix).

        Spec 21 §III: each memory tier has its own budget allocation.
        Evictions must release from the correct tier so pressure metrics stay accurate.
        """
        _MAP: dict[str, MemoryTier] = {
            "episode": MemoryTier.EPISODIC,
            "semantic_node": MemoryTier.SEMANTIC,
            "hypothesis": MemoryTier.HYPOTHESIS,
            "procedure": MemoryTier.PROCEDURAL,
            "schema": MemoryTier.SEMANTIC,  # Schemas live in semantic tier
        }
        return _MAP.get(item_type, MemoryTier.EPISODIC)

    def _compute_hypothesis_confirmation_rate(self) -> float:
        """
        Fraction of world-model schemas that have been confirmed (instance_count > 1).
        A schema with only 1 instance was just created and not yet confirmed.
        A schema with multiple instances has been matched again = hypothesis confirmed.
        """
        schemas = self._world_model.generative_schemas
        if not schemas:
            return 0.0
        confirmed = sum(1 for s in schemas.values() if s.instance_count > 1)
        return confirmed / len(schemas)

    def _compute_intelligence_metrics(
        self, previous: IntelligenceMetrics | None
    ) -> IntelligenceMetrics:
        """Compute the full intelligence metrics snapshot."""
        ir = self._world_model.measure_intelligence_ratio()
        encoder = self._holographic_encoder
        cascade = self._cascade
        sc = self._latest_schwarzschild

        # Compression efficiency: ratio of cascade items that achieved ratio > 1.0
        total = cascade.total_cascaded
        efficient = cascade.total_distilled + cascade.total_integrated
        compression_efficiency = efficient / max(total, 1)

        metrics = IntelligenceMetrics(
            timestamp=utc_now(),
            # Primary
            intelligence_ratio=ir,
            cognitive_pressure=self._budget.total_pressure,
            compression_efficiency=compression_efficiency,
            # World model quality
            world_model_coverage=self._world_model.coverage,
            world_model_complexity=self._world_model.current_complexity,
            prediction_accuracy=self._world_model.prediction_accuracy,
            # Learning velocity
            schema_growth_rate=self._world_model.schema_growth_rate,
            # hypothesis_confirmation_rate: fraction of schemas that have been
            # extended (i.e., a hypothesis was confirmed by a new instance).
            # Approximated as schemas_extended / total schemas; not world model
            # prediction_accuracy (which is a different metric).
            hypothesis_confirmation_rate=self._compute_hypothesis_confirmation_rate(),
            cross_domain_transfers_today=sc.cross_domain_transfers,
            # Schwarzschild proximity (Phase D: real data)
            self_prediction_accuracy=sc.self_prediction_accuracy,
            hypothesis_generation_ratio=sc.hypothesis_ratio,
            schwarzschild_threshold_met=sc.threshold_met,
            # Compression throughput
            experiences_holographically_encoded=encoder.total_encoded,
            experiences_discarded_as_redundant=encoder.total_discarded,
            anchor_memories_created=len(self._anchor_ids),
        )

        # Compute deltas from previous snapshot
        if previous is not None:
            metrics.intelligence_ratio_delta = ir - previous.intelligence_ratio
            metrics.coverage_delta = (
                self._world_model.coverage - previous.world_model_coverage
            )
            metrics.compression_efficiency_delta = (
                compression_efficiency - previous.compression_efficiency
            )

        return metrics
