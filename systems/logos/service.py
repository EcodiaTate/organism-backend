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

from primitives.common import utc_now
from systems.logos.budget import CognitiveBudgetManager
from systems.logos.cascade import CompressionCascade
from systems.logos.decay import EntropicDecayEngine, MemoryStoreProtocol
from systems.logos.holographic import HolographicEncoder
from systems.logos.mdl import MDLEstimator
from systems.logos.schwarzschild import SchwarzchildCognitionDetector
from systems.logos.types import (
    CascadeResult,
    CompressionCycleReport,
    EmpiricalInvariant,
    ExperienceDelta,
    IntelligenceMetrics,
    LogosConfig,
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

        # External references (protocol-based DI)
        self._synapse: SynapseService | None = None
        self._memory: Any = None  # MemoryService
        self._memory_store: MemoryStoreProtocol | None = None

        # Background tasks
        self._pressure_task: asyncio.Task[None] | None = None
        self._metrics_task: asyncio.Task[None] | None = None
        self._decay_task: asyncio.Task[None] | None = None
        self._schwarzschild_task: asyncio.Task[None] | None = None

        # Latest metrics + schwarzschild snapshot
        self._latest_metrics = IntelligenceMetrics()
        self._latest_schwarzschild = SchwarzchildStatus()

        # Anchor memory IDs (protected from eviction)
        self._anchor_ids: set[str] = set()

    # ─── Lifecycle ───────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the Logos engine. Idempotent."""
        if self._initialized:
            return

        logger.info(
            "logos_initializing",
            total_budget=self._config.total_budget_ku,
            pressure_interval=self._config.pressure_broadcast_interval_s,
            metrics_interval=self._config.metrics_broadcast_interval_s,
        )

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
        """Wire Synapse for event broadcasting and subscribe to Fovea errors."""
        self._synapse = synapse
        self._subscribe_to_fovea_errors(synapse)
        logger.info("logos_synapse_wired")

    def set_memory(self, memory: Any) -> None:
        """Wire Memory for consolidation integration."""
        self._memory = memory
        logger.info("logos_memory_wired")

    def set_memory_store(self, store: MemoryStoreProtocol) -> None:
        """Wire a MemoryStoreProtocol for the decay engine."""
        self._memory_store = store
        logger.info("logos_memory_store_wired")

    # ─── Loop 1 feedback: Fovea prediction errors → compression ──

    def _subscribe_to_fovea_errors(self, synapse: SynapseService) -> None:
        """Subscribe to FOVEA_PREDICTION_ERROR to close the Loop 1 feedback cycle.

        When Fovea detects a prediction error, the error IS the delta
        that Logos needs — high-salience errors indicate the world model
        failed to predict reality and must be updated.
        """
        try:
            synapse.event_bus.subscribe(
                SynapseEventType("fovea_prediction_error"),
                self._on_fovea_prediction_error,
            )
            logger.info("logos_subscribed_to_fovea_errors")
        except (ValueError, AttributeError):
            logger.debug(
                "logos_fovea_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_fovea_prediction_error(self, event: Any) -> None:
        """Handle a FOVEA_PREDICTION_ERROR: feed the error back as a compression delta.

        Only high-salience errors justify a world model update — low-salience
        errors are noise. The threshold is set at 0.3 precision-weighted salience.
        """
        data = event.data if hasattr(event, "data") else {}
        salience = data.get("precision_weighted_salience", 0.0)

        if salience < 0.3:
            return

        # Build a RawExperience from the prediction error and push it through
        # the compression cascade.  This is the "prediction error IS the delta"
        # insight from the Integration Manifold.
        raw = RawExperience(
            context={
                "source": "fovea_prediction_error",
                "percept_id": data.get("percept_id", ""),
                "dominant_error_type": data.get("dominant_error_type", ""),
            },
            content={
                "content_error": data.get("content_error", 0.0),
                "temporal_error": data.get("temporal_error", 0.0),
                "magnitude_error": data.get("magnitude_error", 0.0),
                "source_error": data.get("source_error", 0.0),
                "category_error": data.get("category_error", 0.0),
                "causal_error": data.get("causal_error", 0.0),
            },
            raw_complexity=salience * 100.0,
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
            logger.warning(
                "fovea_error_cascade_failed",
                error=str(exc),
            )

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
        """
        return self._budget.increment(tier, amount)

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

        # Only broadcast structural changes — not redundant-discard no-ops
        if self._synapse is not None and not delta.discard_after_encoding:
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.WORLD_MODEL_UPDATED,
                source_system=self.system_id,
                data={
                    "update_type": update.update_type.value,
                    "schemas_added": update.schemas_added,
                    "priors_updated": update.priors_updated,
                    "causal_updates": update.causal_links_added + update.causal_links_revised,
                },
            ))

        return update

    async def process_experience(
        self, raw_experience: RawExperience
    ) -> CascadeResult:
        """
        Run a raw experience through the full four-stage compression
        cascade.  This is the primary entry point for all new experiences.

        The cascade: holographic encoding -> episodic compression ->
        semantic distillation -> world model integration.

        Returns the CascadeResult with per-stage metrics, anchor flags,
        and compression ratios.
        """
        result = await self._cascade.run(raw_experience)

        # Feed compression ratio to Schwarzschild detector
        if result.compression_ratio > 0:
            self._schwarzschild.record_compression_ratio(result.compression_ratio)
            self._schwarzschild.record_data_arrival()

        # Track hypothesis generation from world model updates
        if result.world_model_update is not None:
            wm = result.world_model_update
            if wm.schemas_added > 0:
                self._schwarzschild.record_hypothesis_generated(wm.schemas_added)

        # Broadcast WORLD_MODEL_UPDATED if integration occurred
        if result.world_model_update is not None and self._synapse is not None:
            wm = result.world_model_update
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.WORLD_MODEL_UPDATED,
                source_system=self.system_id,
                data={
                    "update_type": wm.update_type.value,
                    "schemas_added": wm.schemas_added,
                    "priors_updated": wm.priors_updated,
                    "causal_updates": wm.causal_links_added + wm.causal_links_revised,
                },
            ))

        # Handle anchor memories
        if result.anchor_memory and result.compressed_item_id:
            self.mark_anchor(result.compressed_item_id)

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

        # Release budget for evicted items
        if decay_report:
            for _ in decay_report.evicted:
                self._budget.decrement(MemoryTier.EPISODIC, 1.0)

        # Broadcast COMPRESSION_CYCLE_COMPLETE with real data
        if self._synapse is not None:
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.COMPRESSION_CYCLE_COMPLETE,
                source_system=self.system_id,
                data={
                    "items_evicted": report.items_evicted,
                    "items_distilled": report.items_distilled,
                    "mdl_improvement": report.mdl_improvement,
                    "bits_saved": report.bits_saved,
                    "anchors_created": report.anchors_created,
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
    ) -> None:
        """Mark an item as an anchor memory (never evicted)."""
        self._anchor_ids.add(item_id)
        logger.info(
            "anchor_memory_marked",
            item_id=item_id,
            info_content=information_content,
            domain=domain,
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
                    },
                )),
                name=f"logos_anchor_{item_id[:8]}",
            )
            def _on_anchor_done(t: asyncio.Task[None]) -> None:
                if not t.cancelled() and t.exception() is not None:
                    logger.warning("anchor_emit_failed", error=str(t.exception()))

            task.add_done_callback(_on_anchor_done)

    def is_anchor(self, item_id: str) -> bool:
        return item_id in self._anchor_ids

    # ─── Background Loops ────────────────────────────────────────

    async def _pressure_broadcast_loop(self) -> None:
        """Broadcast COGNITIVE_PRESSURE every 30s."""
        while True:
            try:
                await asyncio.sleep(self._config.pressure_broadcast_interval_s)
                if self._synapse is None:
                    continue

                payload = self._budget.pressure_payload()
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.COGNITIVE_PRESSURE,
                    source_system=self.system_id,
                    data=payload,
                ))
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
