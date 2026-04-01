"""
Fovea -- Service Layer

The main orchestrator for the Fovea prediction error attention system.
Implements the ManagedSystemProtocol for Synapse health monitoring,
broadcasts Fovea events on the event bus, and coordinates the full
percept processing pipeline.

Phase A+B: Prediction error computation, precision weighting, habituation,
           workspace integration.
Phase C:   Weight learning from world model updates, habituation feedback.
Phase D:   Internal prediction errors (self-attention).

Wiring order:
    1. Construct FoveaService(world_model)
    2. set_event_bus(synapse_event_bus)  -- subscribes to WORLD_MODEL_UPDATED
    3. set_workspace(atune_workspace)
    4. Ready: call process_percept() per incoming percept
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.evolutionary import EvolutionaryObservable
from primitives.genome import GenomeExtractionProtocol, OrganGenomeSegment
from primitives.re_training import RETrainingExample

from .economic_model import EconomicPredictionModel
from .habituation import (
    _DISHABITUATION_AMPLIFICATION,
    _DISHABITUATION_THRESHOLD,
    _HABITUATION_COMPLETE_THRESHOLD,
    _HABITUATION_INCREMENT,
    _HISTORY_WINDOW,
    _MAX_HABITUATION,
)
from .integration import FoveaAtuneBridge
from .internal import InternalPredictionEngine
from .learning import AttentionWeightLearner
from .protocols import LogosWorldModel, StubWorldModel
from .types import (
    AttentionProfile,
    DEFAULT_ERROR_WEIGHTS,
    ErrorRoute,
    ErrorType,
    FoveaMetrics,
    FoveaPredictionError,
    InternalErrorType,
    InternalPredictionError,
    WorldModelUpdate,
)

if TYPE_CHECKING:
    from primitives.percept import Percept
    from systems.synapse.event_bus import EventBus

    from .workspace import GlobalWorkspace

logger = structlog.get_logger("systems.fovea.service")


# Synapse event type string constants
FOVEA_PREDICTION_ERROR = "fovea_prediction_error"
FOVEA_HABITUATION_DECAY = "fovea_habituation_decay"
FOVEA_DISHABITUATION = "fovea_dishabituation"
FOVEA_WORKSPACE_IGNITION = "fovea_workspace_ignition"
FOVEA_ATTENTION_PROFILE_UPDATE = "fovea_attention_profile_update"
FOVEA_HABITUATION_COMPLETE = "fovea_habituation_complete"
FOVEA_INTERNAL_PREDICTION_ERROR = "fovea_internal_prediction_error"
FOVEA_DIAGNOSTIC_REPORT = "fovea_diagnostic_report"
FOVEA_BACKPRESSURE_WARNING = "fovea_backpressure_warning"
FOVEA_CALIBRATION_ALERT = "fovea_calibration_alert"

# Routing threshold overrides (organism-adjustable at runtime)
# Defaults match original hardcoded values in compute_routing()
_CONSTITUTIONAL_EQUOR_THRESHOLD: float = 0.3
_CONSTITUTIONAL_ONEIROS_THRESHOLD: float = 0.5
_ECONOMIC_ROUTE_THRESHOLD: float = 0.3
_ECONOMIC_WORKSPACE_THRESHOLD: float = 0.5

# Backpressure: warn when unresolved errors exceed this count
_BACKPRESSURE_WARNING_THRESHOLD: int = 150


class FoveaService:
    """
    Fovea -- Prediction Error as Attention.

    Satisfies ManagedSystemProtocol (system_id, health()).
    Phases A-D: error computation, workspace integration, weight learning,
    self-attention.
    """

    system_id: str = "fovea"

    def __init__(
        self,
        world_model: LogosWorldModel | None = None,
        *,
        threshold_percentile: float = 75.0,
        threshold_window: int = 100,
        learning_rate: float = 0.01,
        instance_id: str = "",
        neo4j_driver: Any = None,
    ) -> None:
        wm = world_model or StubWorldModel()
        self._bridge = FoveaAtuneBridge(
            wm,
            threshold_percentile=threshold_percentile,
            threshold_window=threshold_window,
        )
        self._weight_learner = AttentionWeightLearner(
            learning_rate=learning_rate,
            instance_id=instance_id,
            neo4j_driver=neo4j_driver,
        )
        self._bridge.set_weight_applicator(self._weight_learner.apply_learned_weights)
        # Wire habituation engine to weight learner so world model update correlations
        # propagate record_update() calls, fixing stochastic/learning_failure diagnosis.
        self._weight_learner.set_habituation_engine(
            self._bridge.habituation_engine
        )
        # Wire Neo4j to habituation engine
        if neo4j_driver is not None:
            self._bridge.habituation_engine.set_neo4j_driver(neo4j_driver, instance_id)
        self._internal_engine = InternalPredictionEngine()
        self._event_bus: EventBus | None = None
        self._neo4j_driver = neo4j_driver
        self._instance_id = instance_id
        self._logger = logger.bind(component="fovea_service")
        self._started = False
        self._start_time: float = 0.0
        self._paused_for_sleep: bool = False
        self._serialization_error_counts: dict[str, int] = {}  # Track dict key errors per event type

        # Fitness tracking for EvolutionaryObservable emissions
        self._errors_since_last_fitness: int = 0
        self._updates_since_last_fitness: int = 0
        self._fitness_emit_interval: int = 50  # emit every 50 errors

        # RE training: track recent prediction errors for correlation with world model updates
        self._recent_error_for_re: dict[str, FoveaPredictionError] = {}
        self._re_error_buffer_max: int = 200

        # Parent genome weights for novelty detection
        self._parent_genome_weights: dict[str, float] | None = None

        # Attentional divergence (speciation signal for Benchmarks)
        # Fleet mean weights received via FOVEA_ATTENTION_PROFILE_UPDATE from siblings
        self._fleet_weight_samples: list[dict[str, float]] = []
        self._fleet_weight_samples_max: int = 20
        self._divergence_emit_interval: int = 100   # emit every 100 errors
        self._errors_since_last_divergence: int = 0
        self._last_kl_divergence: float = 0.0

        # Per-percept arrival time tracking (feeds WorldModelAdapter timing history)
        self._last_percept_arrival_by_source: dict[str, float] = {}

        # Economic prediction model - tracks revenue/cost EMA and emits
        # ECONOMIC error dimension when actuals diverge from predictions.
        self._economic_model = EconomicPredictionModel()

        # Diagnostic report cadence - emitted every N errors alongside fitness signal
        self._diagnostic_emit_interval: int = 50  # matches fitness interval
        self._errors_since_last_diagnostic: int = 0
        self._last_backpressure_warning_count: int = 0  # avoid repeated warnings

        # Runtime-adjustable routing thresholds (organism can request changes
        # via FOVEA_PARAMETER_ADJUSTMENT event without service restart)
        self._constitutional_equor_threshold: float = _CONSTITUTIONAL_EQUOR_THRESHOLD
        self._constitutional_oneiros_threshold: float = _CONSTITUTIONAL_ONEIROS_THRESHOLD
        self._economic_route_threshold: float = _ECONOMIC_ROUTE_THRESHOLD
        self._economic_workspace_threshold: float = _ECONOMIC_WORKSPACE_THRESHOLD

        # Autonomy gap closure: track consecutive poor cycles for calibration alerts
        self._consecutive_poor_tpr: int = 0  # Reset when TPR >= 0.6
        self._consecutive_high_false_alarm: int = 0  # Reset when false_alarm_rate <= 0.4

        self._modulation_halted: bool = False

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire the Synapse event bus and subscribe to all relevant events."""
        self._event_bus = event_bus
        self._subscribe_to_logos_updates(event_bus)
        self._subscribe_to_affect_state(event_bus)
        self._subscribe_to_sleep_transitions(event_bus)
        self._subscribe_to_hypothesis_outcomes(event_bus)
        self._subscribe_to_percept_arrived(event_bus)
        self._subscribe_to_fleet_attention_profiles(event_bus)
        self._subscribe_to_axon_events(event_bus)
        self._subscribe_to_economic_vitality(event_bus)
        self._subscribe_to_parameter_adjustments(event_bus)
        self._subscribe_to_system_modulation(event_bus)
        self._subscribe_to_compression_events(event_bus)
        self._logger.info("event_bus_wired")

    def set_workspace(self, workspace: GlobalWorkspace) -> None:
        self._bridge.set_workspace(workspace)
        self._logger.info("workspace_wired")

    def set_world_model(self, world_model: LogosWorldModel) -> None:
        """Hot-swap the world model when Logos comes online."""
        self._bridge.set_world_model(world_model)
        self._logger.info("world_model_swapped")

    def set_neo4j_driver(self, driver: Any, instance_id: str = "") -> None:
        """Wire Neo4j driver post-construction for persistence.

        If called after startup() (the usual case when called from registry.py),
        schedules a late restore of persisted state via asyncio.ensure_future()
        so that learned thresholds, weights, and habituation state are recovered
        even though startup() ran before the driver was available.
        """
        self._neo4j_driver = driver
        if instance_id:
            self._instance_id = instance_id
        self._weight_learner.set_neo4j_driver(driver, instance_id)
        self._bridge.habituation_engine.set_neo4j_driver(driver, instance_id)
        self._bridge.dynamic_threshold.set_neo4j_driver(driver, instance_id)  # Part B

        # If the system is already started (startup() ran before Neo4j was
        # available), trigger a late restore so persisted state is actually loaded.
        if self._started and driver is not None:
            import asyncio

            async def _late_restore() -> None:
                try:
                    await self._weight_learner.restore_weights()
                    await self._bridge.habituation_engine.restore_state()
                    await self._bridge.dynamic_threshold.restore_state_from_neo4j()
                    self._logger.info("fovea_neo4j_late_restore_complete", instance_id=instance_id)
                except Exception as exc:
                    self._logger.warning("fovea_neo4j_late_restore_failed", error=str(exc))

            try:
                asyncio.ensure_future(_late_restore())
            except RuntimeError:
                # No running event loop during tests - silently skip
                pass

    def get_metrics(self) -> FoveaMetrics:
        """Return current Fovea metrics snapshot (public API - avoids _bridge access)."""
        return self._bridge.get_metrics()

    @property
    def weight_learner(self) -> AttentionWeightLearner:
        """Return the weight learner for external inspection (public API - avoids _bridge access)."""
        return self._weight_learner

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        # Restore persisted state from Neo4j
        await self._weight_learner.restore_weights()
        await self._bridge.habituation_engine.restore_state()
        await self._bridge.dynamic_threshold.restore_state_from_neo4j()  # Part B
        # Apply inherited curiosity genome from parent (child instances only)
        self._apply_inherited_atune_genome_if_child()
        self._started = True
        self._start_time = time.monotonic()
        self._logger.info("fovea_started")

    async def shutdown(self) -> None:
        # Persist final state before shutdown
        await self._weight_learner.persist_weights(force=True)
        await self._bridge.habituation_engine.persist_state(force=True)
        self._started = False
        self._logger.info("fovea_stopped")

    # ------------------------------------------------------------------
    # ManagedSystemProtocol
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        metrics = self._bridge.get_metrics()
        result = {
            "status": "healthy" if self._started else "stopped",
            "errors_processed": metrics.errors_processed,
            "workspace_ignitions": metrics.workspace_ignitions,
            "mean_salience": round(metrics.mean_salience, 4),
            "habituation_entries": metrics.habituation_entries,
            "active_predictions": metrics.active_predictions,
            "dynamic_threshold": round(
                self._bridge.dynamic_threshold.current, 4
            ),
            "weight_reinforcements": self._weight_learner.reinforcements,
            "weight_decays": self._weight_learner.decays,
            "false_alarms": self._weight_learner.false_alarms,
            "learned_weights": self._weight_learner.weights,
            "internal_predictions_made": self._internal_engine.predictions_made,
            "internal_errors_generated": self._internal_engine.errors_generated,
            "internal_errors_by_type": self._internal_engine.errors_by_type,
            "autonomy": self.introspect_autonomy(),
        }
        return result

    # ------------------------------------------------------------------
    # Autonomy: self-introspection
    # ------------------------------------------------------------------

    def introspect_autonomy(self) -> dict[str, Any]:
        """
        Return all learnable parameters, effectiveness metrics, and internal
        state for organism-level self-awareness. Exposed in health().
        """
        return {
            "learner_params": self._weight_learner.get_learnable_params(),
            "habituation_params": self._bridge.habituation_engine.get_learnable_params(),
            "economic_model_params": self._economic_model.get_learnable_params(),
            "threshold_config": {
                "current": round(self._bridge.dynamic_threshold.current, 4),
                "percentile": self._bridge.dynamic_threshold._percentile,
                "floor": self._bridge.dynamic_threshold._floor,
                "ceiling": self._bridge.dynamic_threshold._ceiling,
            },
            "effectiveness": {
                "reinforcements": self._weight_learner.reinforcements,
                "decays": self._weight_learner.decays,
                "false_alarms": self._weight_learner.false_alarms,
                "tpr": (
                    self._weight_learner.reinforcements
                    / max(1, self._weight_learner.reinforcements + self._weight_learner.false_alarms)
                ),
                "habituation_complete_count": self._bridge.habituation_engine.habituation_complete_count,
            },
            "fleet_divergence": {
                "last_kl_divergence": round(self._last_kl_divergence, 4),
                "fleet_samples": len(self._fleet_weight_samples),
            },
        }

    # ------------------------------------------------------------------
    # Autonomy: learnable parameter API (Evo ADJUST_BUDGET)
    # ------------------------------------------------------------------

    def adjust_learner_param(self, name: str, value: float) -> bool:
        """Adjust a weight learner parameter. Called by Evo ADJUST_BUDGET."""
        return self._weight_learner.adjust_param(name, value)

    def adjust_habituation_param(self, name: str, value: float) -> bool:
        """Adjust a habituation parameter. Called by Evo ADJUST_BUDGET."""
        return self._bridge.habituation_engine.adjust_param(name, value)

    def adjust_economic_param(self, name: str, value: float) -> bool:
        """Adjust an economic model parameter. Called by Evo ADJUST_BUDGET."""
        return self._economic_model.adjust_param(name, value)

    def adjust_threshold_param(self, name: str, value: float) -> bool:
        """Adjust a dynamic ignition threshold parameter."""
        import asyncio as _asyncio
        dt = self._bridge.dynamic_threshold
        if name == "percentile":
            dt._percentile = max(10.0, min(99.0, value))
        elif name == "floor":
            dt._floor = max(0.01, min(0.5, value))
        elif name == "ceiling":
            dt._ceiling = max(0.5, min(1.0, value))
        else:
            return False
        logger.info("threshold_param_adjusted", name=name, value=round(value, 4))
        # Persist immediately - these are Evo-tuned values that must survive restart.
        _asyncio.ensure_future(dt.persist_params())
        return True

    def get_all_learnable_params(self) -> dict[str, Any]:
        """Return all learnable parameters across all Fovea subsystems."""
        result: dict[str, Any] = {
            "learner": self._weight_learner.get_learnable_params(),
            "habituation": self._bridge.habituation_engine.get_learnable_params(),
            "economic": self._economic_model.get_learnable_params(),
            "threshold": {
                "percentile": self._bridge.dynamic_threshold._percentile,
                "floor": self._bridge.dynamic_threshold._floor,
                "ceiling": self._bridge.dynamic_threshold._ceiling,
            },
        }
        # Include workspace curiosity params if the workspace is wired
        workspace = self._bridge._workspace
        if workspace is not None:
            result["workspace"] = workspace.export_learnable_params()
        return result

    def export_learnable_params(self) -> dict[str, Any]:
        """Export all learnable parameters for genome inheritance."""
        return self.get_all_learnable_params()

    def import_learnable_params(self, params: dict[str, Any]) -> None:
        """Import learnable parameters from parent genome."""
        if "learner" in params:
            self._weight_learner.import_learnable_params(params["learner"])
        if "habituation" in params:
            self._bridge.habituation_engine.import_learnable_params(params["habituation"])
        if "economic" in params:
            self._economic_model.import_learnable_params(params["economic"])
        if "threshold" in params:
            for name, value in params["threshold"].items():
                self.adjust_threshold_param(name, float(value))
        # Import workspace curiosity params if available and workspace is wired
        if "workspace" in params:
            workspace = self._bridge._workspace
            if workspace is not None:
                workspace.import_learnable_params(params["workspace"])

    # ------------------------------------------------------------------
    # Main API: percept processing
    # ------------------------------------------------------------------

    async def process_percept(
        self, percept: Percept
    ) -> FoveaPredictionError | None:
        """
        Process a percept through the full Fovea pipeline.

        Integrates Phase C weight learning and Phase D self-attention.
        """
        self._sync_clock()

        # ── Skia modulation halt ──────────────────────────────────────────
        if self._modulation_halted:
            self._logger.debug("process_percept_skipped_modulation_halted", percept_id=percept.id)
            return None

        try:
            error = await self._bridge.process_percept(percept)
        except Exception:
            self._logger.warning(
                "process_percept_failed",
                percept_id=percept.id,
                exc_info=True,
            )
            return None

        if error is None:
            return None

        # Constitutional mismatch detection: if the percept carries drive
        # alignment signals (populated by Nova/Equor in percept.metadata),
        # score the divergence and set constitutional_mismatch, then re-run
        # routing so EQUOR + ONEIROS receive the error when mismatch is high.
        # NOTE: compute_routing() already ran inside the bridge with
        # constitutional_mismatch == 0 (not yet injected). We must re-run it
        # here - after injection - using the instance-level adjustable thresholds
        # so constitutional routing is actually applied (previously a dead path).
        self._inject_constitutional_mismatch(error, percept)
        error.compute_routing(
            self._bridge.dynamic_threshold.current,
            constitutional_equor_threshold=self._constitutional_equor_threshold,
            constitutional_oneiros_threshold=self._constitutional_oneiros_threshold,
            economic_route_threshold=self._economic_route_threshold,
        )

        # Phase C: register error for weight learning correlation
        self._weight_learner.register_error(error)

        # Phase C: flush stale errors (false alarm detection)
        decayed_types = self._weight_learner.flush_stale_errors()
        if decayed_types:
            await self._emit_attention_profile_update(
                {t.value: 0.0 for t in decayed_types},
                decayed_types[0].value,
            )

        # Emit standard events
        await self._emit_prediction_error(error)

        if ErrorRoute.WORKSPACE in error.routes:
            await self._emit_workspace_ignition(error)

        # Consume transient bridge state
        dishabituation_info = self._bridge.consume_dishabituation()
        hab_complete = self._bridge.consume_habituation_complete()

        # Emit HABITUATION_DECAY whenever habituation is actively applied
        # (spec event: {error_signature, habituation_level})
        if error.habituation_level > 0.0 and dishabituation_info is None:
            await self._emit_habituation_decay(error)

        # Emit DISHABITUATION when the habituation engine detected a sudden magnitude change
        # (spec event: {error_signature, expected_magnitude, actual_magnitude})
        if dishabituation_info is not None:
            await self._emit_dishabituation(error, dishabituation_info)

        # Phase C: check for habituation-complete
        if hab_complete is not None:
            await self._emit_habituation_complete(hab_complete)

        # Track error for RE training data correlation
        if len(self._recent_error_for_re) < self._re_error_buffer_max:
            self._recent_error_for_re[error.id] = error

        # Batched persistence for weights and habituation
        await self._weight_learner.persist_weights()
        await self._bridge.habituation_engine.persist_state()

        # Fitness signal emission (batched every N errors)
        self._errors_since_last_fitness += 1
        if self._errors_since_last_fitness >= self._fitness_emit_interval:
            await self._emit_fitness_signal()

        # Attentional divergence: emit KL divergence from fleet mean every N errors.
        # This is a speciation signal - instances with divergent attention phenotypes
        # are developing distinct ecological niches.
        self._errors_since_last_divergence += 1
        if self._errors_since_last_divergence >= self._divergence_emit_interval:
            await self._emit_attentional_divergence()
            self._errors_since_last_divergence = 0

        # Diagnostic report: surface invisible internal state to the organism.
        self._errors_since_last_diagnostic += 1
        if self._errors_since_last_diagnostic >= self._diagnostic_emit_interval:
            await self._emit_diagnostic_report()
            await self._check_calibration_alert()  # Part A: Autonomy gap closure
            self._errors_since_last_diagnostic = 0

        # Backpressure: warn if unresolved error backlog is building up.
        unresolved = sum(1 for t in self._weight_learner.recent_errors if not t.resolved)
        if (
            unresolved >= _BACKPRESSURE_WARNING_THRESHOLD
            and unresolved > self._last_backpressure_warning_count
        ):
            await self._emit_backpressure_warning(unresolved)
            self._last_backpressure_warning_count = unresolved
        elif unresolved < _BACKPRESSURE_WARNING_THRESHOLD:
            self._last_backpressure_warning_count = 0

        return error

    # ------------------------------------------------------------------
    # Phase C: weight learning integration
    # ------------------------------------------------------------------

    def _subscribe_to_logos_updates(self, event_bus: EventBus) -> None:
        """Subscribe to WORLD_MODEL_UPDATED and external FOVEA_PREDICTION_ERROR events."""
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.WORLD_MODEL_UPDATED,
                self._on_world_model_updated,
            )
            self._logger.info("subscribed_to_world_model_updates")
        except (ImportError, ValueError):
            self._logger.debug(
                "logos_subscription_skipped",
                reason="event_type_unavailable",
            )

        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.FOVEA_PREDICTION_ERROR,
                self._on_external_prediction_error,
            )
            self._logger.info("subscribed_to_external_prediction_errors")
        except (ImportError, ValueError):
            self._logger.debug(
                "external_pe_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_world_model_updated(self, event: Any) -> None:
        """
        Synapse callback: Logos has updated the world model.

        Correlates the update back to the prediction error that triggered it
        and updates attention weights accordingly.
        """
        self._sync_clock()

        data = event.data if hasattr(event, "data") else {}
        weight_deltas = self._weight_learner.on_world_model_updated(data)

        if weight_deltas is not None:
            dominant = max(weight_deltas, key=lambda k: abs(weight_deltas[k]))
            await self._emit_attention_profile_update(weight_deltas, dominant)
            await self._weight_learner.persist_weights()

        # Track world model updates for fitness signal
        self._updates_since_last_fitness += 1

        # Emit RE training data: correlate update with recent prediction error
        await self._emit_re_training_data(data)

    async def _on_external_prediction_error(self, event: Any) -> None:
        """
        Handle FOVEA_PREDICTION_ERROR events emitted by other systems (Synapse, Simula).

        External anomaly signals - such as Synapse entering STRESS rhythm state or
        Simula rolling back an evolution proposal - are structurally equivalent to
        world-model divergence: reality (organism state) diverged from expectation.
        We map the incoming signal onto the magnitude_error dimension and let the
        precision-weighted salience machinery treat it as any other error.
        """
        self._sync_clock()
        data: dict[str, Any] = event.data if hasattr(event, "data") else {}

        # Ignore re-emissions from Fovea itself to prevent feedback loops.
        # Check both event.source_system (SynapseEvent field) and data payload.
        event_source = getattr(event, "source_system", None)
        if event_source == self.system_id:
            return

        source = data.get("source_system", event_source or "unknown")

        # Map generic anomaly magnitude to our error dimensions
        magnitude = float(data.get("magnitude", 0.5))
        domain = data.get("domain", source)

        error = FoveaPredictionError(
            percept_id=data.get("trigger_id", ""),
            magnitude_error=min(1.0, magnitude),
            category_error=min(1.0, magnitude * 0.5),  # secondary signal
        )
        error.compute_precision_weighted_salience()

        self._logger.debug(
            "external_prediction_error_registered",
            source=source,
            domain=domain,
            magnitude=round(magnitude, 3),
            salience=round(error.precision_weighted_salience, 3),
        )

        # Re-emit so Atune/Thread subscribers can also react to the signal
        await self._emit_prediction_error(error)

    # ------------------------------------------------------------------
    # Affect state subscription (P3 - precision τ coupling)
    # ------------------------------------------------------------------

    def _subscribe_to_affect_state(self, event_bus: EventBus) -> None:
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.SELF_AFFECT_UPDATED,
                self._on_affect_state_changed,
            )
            self._logger.info("subscribed_to_affect_state")
        except (ImportError, ValueError):
            self._logger.debug(
                "affect_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_affect_state_changed(self, event: Any) -> None:
        """
        Modulate precision (τ) based on arousal from Soma's affect state.

        High arousal → raise ignition threshold (more conservative - only attend
        to very surprising things during high arousal, reducing noise).
        Low arousal → lower threshold (more permissive - attend to subtler errors).
        """
        data = event.data if hasattr(event, "data") else {}
        arousal = float(data.get("arousal", 0.5))

        # Map arousal [0,1] to threshold shift: high arousal = +shift, low = -shift
        # Neutral arousal (0.5) produces zero shift
        tau_shift = (arousal - 0.5) * 0.1  # ±0.05 max
        new_threshold = self._bridge.dynamic_threshold.adjust(tau_shift)

        self._logger.debug(
            "precision_tau_modulated",
            arousal=round(arousal, 3),
            tau_shift=round(tau_shift, 4),
            new_threshold=round(new_threshold, 4),
        )

    # ------------------------------------------------------------------
    # Sleep stage subscription (P4)
    # ------------------------------------------------------------------

    def _subscribe_to_sleep_transitions(self, event_bus: EventBus) -> None:
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.SLEEP_STAGE_TRANSITION,
                self._on_sleep_stage_transition,
            )
            self._logger.info("subscribed_to_sleep_transitions")
        except (ImportError, ValueError):
            self._logger.debug(
                "sleep_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_sleep_stage_transition(self, event: Any) -> None:
        """
        On sleep entry: pause prediction engine, snapshot weights + habituation.
        On sleep exit: resume, reload any Oneiros-modified weights.
        """
        data = event.data if hasattr(event, "data") else {}
        stage = data.get("stage", "")
        direction = data.get("direction", "")

        if stage == "descent" or direction == "entering_sleep":
            self._paused_for_sleep = True
            await self._weight_learner.snapshot_weights_for_sleep()
            await self._bridge.habituation_engine.snapshot_for_sleep()
            self._logger.info("fovea_paused_for_sleep")

        elif stage == "emergence" or direction == "exiting_sleep":
            # Reload weights in case Oneiros modified them during consolidation
            await self._weight_learner.restore_weights()
            self._paused_for_sleep = False
            self._logger.info("fovea_resumed_from_sleep")

    # ------------------------------------------------------------------
    # Hypothesis outcome subscription (P6)
    # ------------------------------------------------------------------

    def _subscribe_to_hypothesis_outcomes(self, event_bus: EventBus) -> None:
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
                self._on_hypothesis_confirmed,
            )
            event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_REFUTED,
                self._on_hypothesis_refuted,
            )
            # Curiosity outcome tracking: positive signal when a hypothesis is
            # created (suggests the recalled content seeded new learning) or
            # when Thread registers a coherence shift.
            try:
                event_bus.subscribe(
                    SynapseEventType.EVO_HYPOTHESIS_CREATED,
                    self._on_curiosity_positive_signal,
                )
                event_bus.subscribe(
                    SynapseEventType.COHERENCE_SHIFT,
                    self._on_curiosity_positive_signal,
                )
            except (AttributeError, ValueError):
                pass
            self._logger.info("subscribed_to_hypothesis_outcomes")
        except (ImportError, ValueError):
            self._logger.debug(
                "hypothesis_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_curiosity_positive_signal(self, event: Any) -> None:
        """
        Mark any pending spontaneous-recall outcomes as positive when Evo creates
        a new hypothesis or Thread registers a coherence shift within 50 cycles.

        This closes the curiosity effectiveness feedback loop: workspace knows
        whether spontaneous recalls are seeding growth or noise.
        """
        workspace = self._bridge._workspace
        if workspace is None:
            return
        current_cycle = workspace.cycle_count
        # Resolve all pending outcomes (value == -1) that fired within 50 cycles
        for percept_id, outcome in list(workspace._spontaneous_recall_outcomes.items()):
            if outcome == -1:  # pending
                workspace.record_curiosity_outcome(percept_id, positive=True)
        self._logger.debug(
            "curiosity_positive_signal_received",
            cycle=current_cycle,
            hit_rate=round(workspace.curiosity_hit_rate, 3),
        )

    async def _on_hypothesis_confirmed(self, event: Any) -> None:
        """Reinforce error-type weights when Evo confirms a hypothesis about a prediction error domain."""
        data = event.data if hasattr(event, "data") else {}
        domain = data.get("domain", "")
        error_type_key = self._map_domain_to_error_type(domain)
        if error_type_key is None:
            return

        old_weight = self._weight_learner.weights.get(error_type_key, 0.0)
        boost = self._weight_learner.learning_rate * 0.5
        new_weight = self._weight_learner.adjust_weight(error_type_key, boost)
        await self._weight_learner.persist_weights()

        self._logger.info(
            "hypothesis_confirmed_weight_reinforced",
            domain=domain,
            error_type=error_type_key,
            old=round(old_weight, 4),
            new=round(new_weight, 4),
        )

    async def _on_hypothesis_refuted(self, event: Any) -> None:
        """Decay error-type weights when Evo refutes a hypothesis about a prediction error domain."""
        data = event.data if hasattr(event, "data") else {}
        domain = data.get("domain", "")
        error_type_key = self._map_domain_to_error_type(domain)
        if error_type_key is None:
            return

        old_weight = self._weight_learner.weights.get(error_type_key, 0.0)
        decay = self._weight_learner.false_alarm_decay_rate * 0.5
        new_weight = self._weight_learner.adjust_weight(error_type_key, -decay)
        await self._weight_learner.persist_weights()

        self._logger.info(
            "hypothesis_refuted_weight_decayed",
            domain=domain,
            error_type=error_type_key,
            old=round(old_weight, 4),
            new=round(new_weight, 4),
        )

    @staticmethod
    def _map_domain_to_error_type(domain: str) -> str | None:
        """Map an Evo hypothesis domain string to a Fovea ErrorType key."""
        mapping: dict[str, str] = {
            "content": ErrorType.CONTENT,
            "temporal": ErrorType.TEMPORAL,
            "timing": ErrorType.TEMPORAL,
            "magnitude": ErrorType.MAGNITUDE,
            "source": ErrorType.SOURCE,
            "category": ErrorType.CATEGORY,
            "causal": ErrorType.CAUSAL,
            "economic": ErrorType.ECONOMIC,
            "economic_revenue": ErrorType.ECONOMIC,
            "revenue": ErrorType.ECONOMIC,
        }
        return mapping.get(domain.lower())

    @property
    def weight_learner(self) -> AttentionWeightLearner:
        """Access the weight learner for external weight queries."""
        return self._weight_learner

    # ------------------------------------------------------------------
    # Phase D: internal prediction errors (self-attention)
    # ------------------------------------------------------------------

    @property
    def internal_engine(self) -> InternalPredictionEngine:
        """Access the internal prediction engine for other EOS systems."""
        return self._internal_engine

    def predict_self(
        self,
        action_type: str,
        internal_error_type: InternalErrorType,
        predicted_state: dict[str, Any],
    ) -> str:
        """
        Generate a self-prediction before a major cognitive action.

        Called by other EOS systems (Nova, Equor, Thread, Soma) before
        actions they want to self-monitor.

        Returns a prediction_id for later resolution via resolve_self().
        """
        self._sync_clock()
        return self._internal_engine.predict(
            action_type=action_type,
            internal_error_type=internal_error_type,
            predicted_state=predicted_state,
            clock_s=self._elapsed_s(),
        )

    async def resolve_self(
        self,
        prediction_id: str,
        actual_state: dict[str, Any],
    ) -> InternalPredictionError | None:
        """
        Resolve a self-prediction after a cognitive action completes.

        If the prediction was violated, the InternalPredictionError flows
        through the standard pipeline with the 3x precision multiplier.
        """
        internal_error = self._internal_engine.resolve(
            prediction_id, actual_state
        )
        if internal_error is None:
            return None

        # Apply learned weights and recompute salience with 3x multiplier
        self._weight_learner.apply_learned_weights(internal_error)
        internal_error.compute_precision_weighted_salience()

        # Apply habituation (internal errors can habituate too)
        self._bridge.habituation_engine.apply_habituation(internal_error)

        # Re-run routing with adjustable constitutional thresholds so EQUOR
        # and ONEIROS routing honours Evo-tuned parameters rather than hardcoded
        # defaults. The internal engine pre-sets an initial route; compute_routing
        # supplements it (routes are appended, not replaced) with constitutional
        # and economic routing checks.
        internal_error.compute_routing(
            self._bridge.dynamic_threshold.current,
            constitutional_equor_threshold=self._constitutional_equor_threshold,
            constitutional_oneiros_threshold=self._constitutional_oneiros_threshold,
            economic_route_threshold=self._economic_route_threshold,
        )

        # Register for weight learning
        self._weight_learner.register_error(internal_error)

        # Emit events
        await self._emit_internal_prediction_error(internal_error)
        await self._emit_prediction_error(internal_error)

        if ErrorRoute.WORKSPACE in internal_error.routes:
            await self._emit_workspace_ignition(internal_error)

        return internal_error

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    async def expire_stale_predictions(self) -> int:
        """Housekeeping: remove stale predictions (external + internal)."""
        external = self._bridge.prediction_engine.expire_stale_predictions()
        internal = self._internal_engine.expire_stale(
            max_age_s=60.0, clock_s=self._elapsed_s()
        )
        return external + internal

    async def prune_habituation(self, max_entries: int = 1000) -> int:
        """Housekeeping: prune habituation tracker."""
        return self._bridge.habituation_engine.prune_stale(max_entries)

    def get_metrics(self) -> FoveaMetrics:
        metrics = self._bridge.get_metrics()
        metrics.internal_errors_count = self._internal_engine.errors_generated
        metrics.error_weight_profile = self._weight_learner.weights
        return metrics

    def get_attention_weights(self) -> dict[str, float]:
        """Return current attention weight profile.

        Satisfies FoveaAttentionProtocol for Nexus attentional divergence measurement.
        Keys are prediction error type strings; values are learned weights in [0, 1].
        """
        return self._weight_learner.get_raw_weights()

    def get_current_attention_profile(self) -> AttentionProfile:
        """
        Build a current attentional profile snapshot for Alive telemetry.

        Mean error values are computed from recent tracked errors; the dynamic
        ignition threshold and habituation count come from the bridge.
        """
        recent = list(self._weight_learner.recent_errors)

        # Accumulators for each error dimension
        content = timing = magnitude = source = category = causal = economic = 0.0
        top_salience: float = -1.0
        top_summary: str | None = None

        for tracked in recent:
            # _TrackedError only stores dominant_type + salience; use salience
            # as a proxy for each error dimension via dominant_type.
            dim = tracked.dominant_type.value
            s = tracked.salience
            if dim == "content":
                content += s
            elif dim == "temporal":
                timing += s
            elif dim == "magnitude":
                magnitude += s
            elif dim == "source":
                source += s
            elif dim == "category":
                category += s
            elif dim == "causal":
                causal += s
            elif dim == "economic":
                economic += s

            if s > top_salience:
                top_salience = s
                top_summary = f"Dominant {dim} error (salience={round(s, 3)})"

        n = max(len(recent), 1)
        return AttentionProfile(
            mean_content_error=round(content / n, 4),
            mean_timing_error=round(timing / n, 4),
            mean_magnitude_error=round(magnitude / n, 4),
            mean_source_error=round(source / n, 4),
            mean_category_error=round(category / n, 4),
            mean_causal_error=round(causal / n, 4),
            mean_economic_error=round(economic / n, 4),
            current_ignition_threshold=round(
                self._bridge.dynamic_threshold.current, 4
            ),
            habituated_pattern_count=self._bridge.habituation_engine.entry_count,
            highest_recent_error_summary=top_summary,
        )

    # ------------------------------------------------------------------
    # Oneiros integration: backlog & error domain queries
    # ------------------------------------------------------------------

    async def get_unprocessed_error_count(self) -> int:
        """Return count of prediction errors not yet correlated to world model updates.

        Satisfies ``FoveaBacklogProtocol`` used by the Oneiros SleepScheduler
        to determine if the compression backlog is high enough to trigger sleep.
        """
        return sum(
            1
            for tracked in self._weight_learner.recent_errors
            if not tracked.resolved
        )

    async def get_top_error_domains(self, n: int = 5) -> list[dict[str, Any]]:
        """Return top-N domains by remaining prediction error.

        Satisfies ``FoveaErrorDomainProtocol`` used by the Oneiros REM stage
        to target cross-domain synthesis at the highest-error domains.

        Each dict: {"domain": str, "error_count": int, "mean_error": float}
        """
        from collections import Counter

        domain_errors: dict[str, list[float]] = {}
        domain_counts: Counter[str] = Counter()

        for tracked in self._weight_learner.recent_errors:
            if tracked.resolved:
                continue
            # Use the dominant error type as a proxy for "domain" since Fovea
            # tracks errors by error type, not by external domain.
            domain = tracked.dominant_type.value
            domain_counts[domain] += 1
            domain_errors.setdefault(domain, []).append(tracked.salience)

        result: list[dict[str, Any]] = []
        for domain, count in domain_counts.most_common(n):
            errors = domain_errors[domain]
            result.append({
                "domain": domain,
                "error_count": count,
                "mean_error": sum(errors) / len(errors) if errors else 0.0,
            })
        return result

    # ------------------------------------------------------------------
    # Genome extraction (GenomeExtractionProtocol)
    # ------------------------------------------------------------------

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """Serialise Fovea's heritable state: weights + all learnable params."""
        payload = {
            "error_weights": self._weight_learner.get_raw_weights(),
            # Legacy fields (retained for backward compat)
            "habituation_increment": _HABITUATION_INCREMENT,
            "max_habituation_cap": _MAX_HABITUATION,
            "learning_rate": self._weight_learner.learning_rate,
            "false_alarm_decay": self._weight_learner.false_alarm_decay_rate,
            # Full learnable params (all subsystems)
            "learnable_params": self.export_learnable_params(),
        }
        payload_json = json.dumps(payload, sort_keys=True)
        return OrganGenomeSegment(
            system_id=SystemID.FOVEA,
            payload=payload,
            payload_hash=hashlib.sha256(payload_json.encode()).hexdigest(),
            size_bytes=len(payload_json.encode()),
        )

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """Restore heritable state from a parent genome segment."""
        if segment.system_id != SystemID.FOVEA:
            return False
        payload = segment.payload
        weights = payload.get("error_weights")
        if isinstance(weights, dict) and weights:
            self._weight_learner.set_raw_weights(weights)
            self._parent_genome_weights = dict(weights)
        # Import full learnable params (preferred over legacy fields)
        if "learnable_params" in payload:
            self.import_learnable_params(payload["learnable_params"])
        else:
            # Legacy fallback
            if "learning_rate" in payload:
                self._weight_learner.set_learning_rate(float(payload["learning_rate"]))
            if "false_alarm_decay" in payload:
                self._weight_learner.set_false_alarm_decay(float(payload["false_alarm_decay"]))
        self._logger.info("seeded_from_parent_genome", payload_keys=list(payload.keys()))
        return True

    # ------------------------------------------------------------------
    # AtuneGenomeFragment - spawn-time curiosity genome (Mitosis)
    # ------------------------------------------------------------------

    def export_atune_genome(self, instance_id: str = "", generation: int = 1) -> Any:
        """
        Extract AtuneGenomeFragment for Mitosis child spawning.

        Called by SpawnChildExecutor Step 0b. Returns an AtuneGenomeFragment
        capturing the parent's evolved curiosity parameters, current arousal
        level, and curiosity hit-rate so children inherit a tuned attentional
        rhythm rather than cold-starting at defaults.
        """
        try:
            from primitives.genome_inheritance import AtuneGenomeFragment
        except ImportError:
            return None

        workspace = self._bridge._workspace
        curiosity_params: dict[str, float] = {}
        buffer_scale_arousal = 0.4
        curiosity_hit_rate = 0.5

        if workspace is not None:
            curiosity_params = workspace.export_learnable_params()
            buffer_scale_arousal = workspace._current_arousal
            curiosity_hit_rate = workspace.curiosity_hit_rate

        fragment = AtuneGenomeFragment(
            instance_id=instance_id or self._instance_id,
            generation=generation,
            curiosity_params=curiosity_params,
            buffer_scale_arousal=buffer_scale_arousal,
            curiosity_hit_rate=curiosity_hit_rate,
        )

        self._logger.info(
            "atune_genome_extracted",
            genome_id=fragment.genome_id,
            curiosity_hit_rate=round(curiosity_hit_rate, 3),
            base_prob=round(curiosity_params.get("base_prob", 0.02), 4),
        )
        return fragment

    def _apply_inherited_atune_genome_if_child(self) -> None:
        """
        Child-side: apply inherited AtuneGenomeFragment from env var.

        Reads ECODIAOS_ATUNE_GENOME_PAYLOAD (JSON-encoded AtuneGenomeFragment)
        injected by LocalDockerSpawner / CloudRunSpawner at boot time.
        Non-fatal - if env var is absent or malformed, cold-starts at defaults.
        """
        import json as _json
        import os as _os

        if _os.environ.get("ECODIAOS_IS_GENESIS_NODE", "").lower() in ("true", "1"):
            return  # Genesis instance has no parent

        payload_str = _os.environ.get("ECODIAOS_ATUNE_GENOME_PAYLOAD", "")
        if not payload_str:
            return

        try:
            from primitives.genome_inheritance import AtuneGenomeFragment
            data = _json.loads(payload_str)
            fragment = AtuneGenomeFragment.model_validate(data)

            workspace = self._bridge._workspace
            if workspace is not None and fragment.curiosity_params:
                workspace.import_learnable_params(fragment.curiosity_params)
                # Pre-warm arousal so buffer sizes are set before first Soma tick
                workspace.update_arousal(fragment.buffer_scale_arousal)

            self._logger.info(
                "atune_genome_inherited",
                genome_id=fragment.genome_id,
                generation=fragment.generation,
                curiosity_hit_rate=round(fragment.curiosity_hit_rate, 3),
                curiosity_params=fragment.curiosity_params,
            )

            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                import asyncio as _asyncio
                if self._event_bus is not None:
                    _asyncio.ensure_future(
                        self._event_bus.emit(
                            SynapseEvent(
                                event_type=SynapseEventType.GENOME_INHERITED,
                                source_system="fovea",
                                data={
                                    "genome_id": fragment.genome_id,
                                    "system": "atune",
                                    "generation": fragment.generation,
                                },
                            )
                        )
                    )
            except Exception:
                pass

        except Exception:
            self._logger.warning("atune_genome_inheritance_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Fitness signal (EvolutionaryObservable → Evo)
    # ------------------------------------------------------------------

    async def _emit_fitness_signal(self) -> None:
        """Emit attention_calibration fitness to Evo after each batch of errors."""
        if self._event_bus is None:
            return

        total = self._errors_since_last_fitness
        updates = self._updates_since_last_fitness
        tpr = updates / total if total > 0 else 0.0

        # Detect novelty: has the weight distribution shifted from parent genome?
        is_novel = False
        if self._parent_genome_weights is not None:
            current_weights = self._weight_learner.get_raw_weights()
            shift = sum(
                abs(current_weights.get(k, 0.0) - v)
                for k, v in self._parent_genome_weights.items()
            )
            is_novel = shift > 0.1

        self._errors_since_last_fitness = 0
        self._updates_since_last_fitness = 0

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            observable = EvolutionaryObservable(
                source_system=SystemID.FOVEA,
                instance_id=self._instance_id,
                observable_type="attention_calibration",
                value=tpr,
                is_novel=is_novel,
                metadata={
                    "true_positive_rate": round(tpr, 4),
                    "errors_in_batch": total,
                    "updates_in_batch": updates,
                    "current_weights": {
                        k: round(v, 4) for k, v in self._weight_learner.get_raw_weights().items()
                    },
                },
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system=self.system_id,
                data=observable.model_dump(),
            )
            await self._event_bus.emit(event)
        except (ImportError, ValueError):
            self._logger.debug("fitness_emit_skipped", reason="event_type_unavailable")

    # ------------------------------------------------------------------
    # RE training data emission
    # ------------------------------------------------------------------

    async def _emit_re_training_data(self, update_data: dict[str, Any]) -> None:
        """Emit RETrainingExample when a world model update is correlated with a prediction error."""
        if self._event_bus is None or not self._recent_error_for_re:
            return

        # Find the most recent unresolved error that likely triggered this update
        error_id: str | None = None
        error: FoveaPredictionError | None = None
        for eid in reversed(list(self._recent_error_for_re.keys())):
            error = self._recent_error_for_re[eid]
            error_id = eid
            break

        if error is None or error_id is None:
            return

        # Remove used error from buffer
        self._recent_error_for_re.pop(error_id, None)

        update_magnitude = AttentionWeightLearner._compute_update_magnitude(update_data)
        if update_magnitude <= 0:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.FOVEA,
                category="prediction_error_resolution",
                instruction="Given prediction error dimensions and salience, predict world model update.",
                input_context=json.dumps({
                    "error_dimensions": error.get_error_vector(),
                    "salience": round(error.precision_weighted_salience, 4),
                    "dominant_type": error.get_dominant_error_type().value,
                }),
                output=json.dumps({
                    "update_magnitude": round(update_magnitude, 4),
                    "schemas_added": update_data.get("schemas_added", 0),
                    "priors_updated": update_data.get("priors_updated", 0),
                    "causal_updates": update_data.get("causal_updates", 0),
                }),
                outcome_quality=min(update_magnitude, 1.0),
            )
            event = SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system=self.system_id,
                data=example.model_dump(),
            )
            await self._event_bus.emit(event)
        except (ImportError, ValueError):
            self._logger.debug("re_training_emit_skipped", reason="event_type_unavailable")

    # ------------------------------------------------------------------
    # PERCEPT_ARRIVED subscription (direct Synapse path, not via bridge)
    # ------------------------------------------------------------------

    def _subscribe_to_percept_arrived(self, event_bus: EventBus) -> None:
        """Subscribe to PERCEPT_ARRIVED to track inter-event timing.

        This is the direct Synapse path for Fovea to observe incoming percepts
        without being called from PerceptionGateway. Allows WorldModelAdapter
        to build timing statistics independent of the bridge processing path.
        """
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.PERCEPT_ARRIVED,
                self._on_percept_arrived,
            )
            self._logger.info("subscribed_to_percept_arrived")
        except (ImportError, ValueError):
            self._logger.debug(
                "percept_arrived_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_percept_arrived(self, event: Any) -> None:
        """
        Handle PERCEPT_ARRIVED events from Synapse.

        Updates the WorldModelAdapter's timing history for the source system
        so future predict_timing() calls use live inter-event statistics.
        """
        data = event.data if hasattr(event, "data") else {}
        source_system = data.get("source_system", "")
        if not source_system:
            return

        import time as _time

        now = _time.monotonic()
        last = self._last_percept_arrival_by_source.get(source_system)
        if last is not None:
            gap_s = now - last
            # Feed gap into WorldModelAdapter if it's wired
            wm = getattr(self._bridge, "_prediction_engine", None)
            if wm is not None:
                adapter = getattr(wm, "_world_model", None)
                record_fn = getattr(adapter, "record_timing_observation", None)
                if callable(record_fn):
                    record_fn(source_system, gap_s)

        self._last_percept_arrival_by_source[source_system] = now

    # ------------------------------------------------------------------
    # Fleet attention profile subscription (for KL divergence computation)
    # ------------------------------------------------------------------

    def _subscribe_to_fleet_attention_profiles(self, event_bus: EventBus) -> None:
        """Subscribe to sibling FOVEA_ATTENTION_PROFILE_UPDATE events.

        Fleet siblings emit their weight vectors on this event. We accumulate
        samples to compute a fleet mean for the KL divergence calculation.
        """
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.FOVEA_ATTENTION_PROFILE_UPDATE,
                self._on_fleet_attention_profile,
            )
            self._logger.info("subscribed_to_fleet_attention_profiles")
        except (ImportError, ValueError):
            self._logger.debug(
                "fleet_profile_subscription_skipped",
                reason="event_type_unavailable",
            )

    def _subscribe_to_axon_events(self, event_bus: EventBus) -> None:
        """Subscribe to Axon execution lifecycle events.

        Fovea broadcasts a prediction-error query when it observes an
        AXON_EXECUTION_REQUEST, and resolves competency errors via
        AXON_EXECUTION_RESULT. Replaces any direct import of PredictionError
        from Axon - all Axon→Fovea comms now flow via Synapse bus.
        """
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.AXON_EXECUTION_REQUEST,
                self._on_axon_execution_request,
            )
            event_bus.subscribe(
                SynapseEventType.AXON_EXECUTION_RESULT,
                self._on_axon_execution_result,
            )
            self._logger.info("subscribed_to_axon_execution_events")
        except (ImportError, ValueError):
            self._logger.debug(
                "axon_execution_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_fleet_attention_profile(self, event: Any) -> None:
        """Accumulate fleet weight vectors for KL divergence computation.

        Ignores events emitted by this instance (same instance_id) to avoid
        self-contamination of the fleet mean.
        """
        event_source = getattr(event, "source_system", None)
        if event_source == self.system_id:
            # Check if this is from our own instance
            data = event.data if hasattr(event, "data") else {}
            emitter_instance = data.get("instance_id", "")
            if emitter_instance == self._instance_id or not emitter_instance:
                return

        data = event.data if hasattr(event, "data") else {}
        weights = data.get("current_weights")
        if not isinstance(weights, dict) or not weights:
            return

        # Keep a rolling window of fleet weight samples
        self._fleet_weight_samples.append(dict(weights))
        if len(self._fleet_weight_samples) > self._fleet_weight_samples_max:
            self._fleet_weight_samples.pop(0)

    async def _on_axon_execution_request(self, event: Any) -> None:
        """
        Handle AXON_EXECUTION_REQUEST from Axon (Spec 06 decoupling).

        Generates an internal competency prediction before execution begins
        so that AXON_EXECUTION_RESULT can measure the delta without Axon
        directly calling Fovea.

        Replaces: direct import of systems.fovea.types.PredictionError from Axon.
        """
        data = getattr(event, "data", {}) or {}
        intent_id: str = data.get("intent_id", "")
        action_types: list[str] = data.get("action_types", [])
        if not intent_id:
            return

        # Cache for resolve - intent_id → internal prediction_id
        if not hasattr(self, "_axon_prediction_ids"):
            self._axon_prediction_ids: dict[str, str] = {}

        try:
            prediction_id = self._internal_engine.predict(
                action_type=",".join(action_types) or "unknown",
                internal_error_type=InternalErrorType.COMPETENCY,
                predicted_state={
                    "intent_id": intent_id,
                    "expected_success": True,
                    "action_types": action_types,
                    "goal": data.get("goal", ""),
                },
            )
            self._axon_prediction_ids[intent_id] = prediction_id
        except Exception:
            pass

    async def _on_axon_execution_result(self, event: Any) -> None:
        """
        Handle AXON_EXECUTION_RESULT from Axon (Spec 06 decoupling).

        Resolves the competency self-prediction registered in
        _on_axon_execution_request. If the actual result violates the
        prediction, the 3x precision multiplier fires an internal error
        - the organism notices when its actions don't produce expected
        outcomes, without Axon needing to directly call Fovea.
        """
        data = getattr(event, "data", {}) or {}
        intent_id: str = data.get("intent_id", "")
        success: bool = bool(data.get("success", False))
        failure_reason: str = data.get("failure_reason") or ""
        duration_ms: int = int(data.get("duration_ms", 0))
        if not intent_id:
            return

        prediction_cache: dict[str, str] = getattr(self, "_axon_prediction_ids", {})
        prediction_id = prediction_cache.pop(intent_id, None)
        if prediction_id is None:
            return

        try:
            self._internal_engine.resolve(
                prediction_id=prediction_id,
                actual_state={
                    "intent_id": intent_id,
                    "actual_success": success,
                    "failure_reason": failure_reason,
                    "duration_ms": duration_ms,
                },
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Economic prediction error (ECONOMIC_VITALITY from Oikos)
    # ------------------------------------------------------------------

    def _subscribe_to_economic_vitality(self, event_bus: EventBus) -> None:
        """Subscribe to ECONOMIC_VITALITY from Oikos for revenue prediction errors."""
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.ECONOMIC_VITALITY,
                self._on_economic_vitality,
            )
            self._logger.info("subscribed_to_economic_vitality")
        except (ImportError, ValueError):
            self._logger.debug(
                "economic_vitality_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_economic_vitality(self, event: Any) -> None:
        """
        Handle ECONOMIC_VITALITY from Oikos.

        Feeds actuals into EconomicPredictionModel. If the composite
        prediction error exceeds 0.3, creates a FoveaPredictionError with
        economic_error set and routes it to OIKOS + EVO. If it exceeds 0.6,
        also emits FOVEA_INTERNAL_PREDICTION_ERROR (mirrors SACM cost surprise).
        """
        if not self._economic_model.is_warmed_up and self._economic_model._observation_count < 1:
            # Prime the model on first event, no error yet
            data: dict[str, Any] = event.data if hasattr(event, "data") else {}
            self._economic_model.update_from_economic_vitality(data)
            return

        data = event.data if hasattr(event, "data") else {}
        composite_error = self._economic_model.update_from_economic_vitality(data)

        if not self._economic_model.is_warmed_up:
            return

        if composite_error < self._economic_route_threshold:
            return

        # Build a FoveaPredictionError in the economic dimension
        error = FoveaPredictionError(
            economic_error=min(1.0, composite_error),
        )
        self._weight_learner.apply_learned_weights(error)
        error.compute_precision_weighted_salience()
        self._bridge.habituation_engine.apply_habituation(error)
        error.compute_routing(
            self._bridge.dynamic_threshold.current,
            economic_route_threshold=self._economic_route_threshold,
        )

        # Ensure OIKOS + EVO are in routes regardless of generic threshold logic
        from .types import ErrorRoute as _ER
        if _ER.OIKOS not in error.routes:
            error.routes.append(_ER.OIKOS)
        if _ER.EVO not in error.routes:
            error.routes.append(_ER.EVO)
        if _ER.WORKSPACE not in error.routes and composite_error > 0.5:
            error.routes.append(_ER.WORKSPACE)

        # Emit standard prediction error
        await self._emit_prediction_error(error)

        worst_source = self._economic_model.worst_source
        worst_error = self._economic_model.worst_source_error

        self._logger.info(
            "economic_prediction_error_emitted",
            composite_error=round(composite_error, 3),
            worst_source=worst_source or "aggregate",
            worst_source_error=round(worst_error, 3),
            routes=error.routes,
        )

        # If error crosses 0.6: mirror SACM cost-surprise pattern with a
        # FOVEA_INTERNAL_PREDICTION_ERROR (domain="economic_revenue")
        if composite_error > 0.6 and self._event_bus is not None:
            failed_source = worst_source or "aggregate"
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType(FOVEA_INTERNAL_PREDICTION_ERROR),
                    source_system=self.system_id,
                    data={
                        "domain": "economic_revenue",
                        "prediction_error": {
                            "economic": round(composite_error, 4),
                            "source": "fovea.economic_prediction_model",
                            "failed_source": failed_source,
                        },
                        "salience_hint": round(min(1.0, composite_error), 4),
                    },
                ))
            except (ImportError, ValueError):
                pass

        # Emit RE training example for significant economic errors
        if composite_error > 0.3 and self._event_bus is not None:
            await self._emit_economic_re_training(composite_error, data)

    async def _emit_economic_re_training(
        self,
        composite_error: float,
        vitality_payload: dict[str, Any],
    ) -> None:
        """Emit RE_TRAINING_EXAMPLE for economic prediction errors."""
        import json as _json

        worst_source = self._economic_model.worst_source
        worst_error = self._economic_model.worst_source_error
        context = self._economic_model.build_re_context()

        instruction = (
            f"Revenue from '{worst_source}' diverged from prediction by "
            f"{round(worst_error * 100, 1)}%."
            if worst_source
            else f"Aggregate revenue diverged from prediction (composite error {round(composite_error, 3)})."
        )

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.FOVEA,
                category="economic_prediction_error",
                instruction=instruction,
                input_context=_json.dumps(context),
                output=_json.dumps({
                    "composite_error": round(composite_error, 4),
                    "revenue_error": round(self._economic_model.revenue_prediction_error, 4),
                    "efficiency_trend_error": round(self._economic_model.efficiency_trend_error, 4),
                    "worst_source": worst_source,
                    "worst_source_error": round(worst_error, 4),
                    "urgency": float(vitality_payload.get("urgency", 0) or 0),
                }),
                outcome_quality=min(1.0, composite_error),
            )
            event = SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system=self.system_id,
                data=example.model_dump(),
            )
            await self._event_bus.emit(event)  # type: ignore[union-attr]
        except (ImportError, ValueError):
            self._logger.debug("economic_re_training_emit_skipped", reason="event_type_unavailable")

    # ------------------------------------------------------------------
    # Attentional divergence emission (speciation signal → Benchmarks)
    # ------------------------------------------------------------------

    async def _emit_attentional_divergence(self) -> None:
        """
        Emit FOVEA_ATTENTIONAL_DIVERGENCE with KL divergence from fleet mean.

        KL(P || Q) where:
          P = this instance's current error-weight distribution (normalised)
          Q = fleet mean distribution (mean of accumulated sibling samples)

        If no fleet data is available, emits KL=0.0 (no divergence measured yet).
        Benchmarks subscribes to track how attentional phenotypes diverge over time.
        """
        if self._event_bus is None:
            return

        my_weights = self._weight_learner.get_raw_weights()
        fleet_mean = self._get_fleet_mean_weights()
        kl = self._kl_divergence(my_weights, fleet_mean)
        self._last_kl_divergence = kl

        # Compute divergence_rank: simple threshold-based rank [0,1]
        # (proper percentile rank requires fleet-wide comparison; this is a proxy)
        divergence_rank = min(1.0, kl / 2.0)  # KL > 2.0 = maximally divergent

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event = SynapseEvent(
                event_type=SynapseEventType.FOVEA_ATTENTIONAL_DIVERGENCE,
                source_system=self.system_id,
                data={
                    "instance_id": self._instance_id,
                    "kl_divergence": round(kl, 6),
                    "weight_vector": {k: round(v, 4) for k, v in my_weights.items()},
                    "fleet_mean_vector": {k: round(v, 4) for k, v in fleet_mean.items()},
                    "divergence_rank": round(divergence_rank, 4),
                    "cycle_count": self._errors_since_last_fitness
                    + self._fitness_emit_interval
                    * (self._updates_since_last_fitness + 1),
                    "fleet_sample_count": len(self._fleet_weight_samples),
                },
            )
            await self._event_bus.emit(event)
            self._logger.debug(
                "attentional_divergence_emitted",
                kl=round(kl, 4),
                rank=round(divergence_rank, 4),
                fleet_samples=len(self._fleet_weight_samples),
            )
        except (ImportError, ValueError):
            self._logger.debug(
                "attentional_divergence_emit_skipped",
                reason="event_type_unavailable",
            )

    def _get_fleet_mean_weights(self) -> dict[str, float]:
        """Compute fleet mean from accumulated sibling weight samples."""
        if not self._fleet_weight_samples:
            # No fleet data - return uniform distribution as neutral baseline
            my_weights = self._weight_learner.get_raw_weights()
            n = max(len(my_weights), 1)
            return {k: 1.0 / n for k in my_weights}

        # Sum across all samples for each dimension
        all_keys: set[str] = set()
        for sample in self._fleet_weight_samples:
            all_keys.update(sample.keys())

        sums: dict[str, float] = {k: 0.0 for k in all_keys}
        for sample in self._fleet_weight_samples:
            for k in all_keys:
                sums[k] += sample.get(k, 0.0)

        n = len(self._fleet_weight_samples)
        return {k: v / n for k, v in sums.items()}

    @staticmethod
    def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
        """
        KL divergence KL(P || Q) between two discrete probability distributions.

        P = this instance's weight distribution (normalised to [0,1] sum)
        Q = fleet mean distribution (normalised)

        Uses ε-smoothing (1e-9) to avoid log(0).
        Returns 0.0 if either distribution is empty or degenerate.
        """
        eps = 1e-9
        dims = set(p.keys()) | set(q.keys())
        if not dims:
            return 0.0

        # Normalise both distributions
        p_sum = sum(p.values()) or 1.0
        q_sum = sum(q.values()) or 1.0
        p_norm = {d: p.get(d, 0.0) / p_sum for d in dims}
        q_norm = {d: q.get(d, 0.0) / q_sum for d in dims}

        kl = 0.0
        for d in dims:
            pi = p_norm[d] + eps
            qi = q_norm[d] + eps
            kl += pi * (math.log(pi) - math.log(qi))

        return max(0.0, kl)

    # ------------------------------------------------------------------
    # Constitutional mismatch detection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_constitutional_mismatch(
        error: FoveaPredictionError,
        percept: Any,
    ) -> None:
        """
        Detect constitutional value mismatch and set error.constitutional_mismatch.

        A constitutional mismatch occurs when:
        - The percept's metadata carries ``predicted_care_score`` and
          ``actual_care_score`` (or equivalents for Coherence/Honesty/Growth),
          and the actual score is significantly lower than predicted.
        - OR the Equor decision (``equor_verdict``) in the metadata was
          MODIFY/DENY despite a high predicted alignment, indicating the
          organism's own behaviour drifted from constitutional expectations.

        The signal is continuous [0.0, 1.0]:
          0.0  = no constitutional dimension to this error
          >0.3 = route to Equor
          >0.5 = also route to Oneiros (sleep-consolidation backlog)

        Caller (process_percept) must call this BEFORE compute_routing
        so that EQUOR/ONEIROS routing is included in the routes list.
        """
        meta = getattr(percept, "metadata", {}) or {}

        # Primary signal: explicit drive score deltas (set by Nova/Equor)
        mismatch = 0.0
        for predicted_key, actual_key in (
            ("predicted_care_score", "actual_care_score"),
            ("predicted_honesty_score", "actual_honesty_score"),
            ("predicted_coherence_score", "actual_coherence_score"),
            ("predicted_growth_score", "actual_growth_score"),
        ):
            if predicted_key in meta and actual_key in meta:
                predicted = float(meta[predicted_key])
                actual = float(meta[actual_key])
                # Only count drops (predicted high, actual low = mismatch)
                drop = max(0.0, predicted - actual)
                mismatch = max(mismatch, drop)

        # Secondary signal: Equor MODIFY/DENY on a predicted-PERMIT outcome
        equor_verdict = meta.get("equor_verdict", "")
        predicted_verdict = meta.get("predicted_equor_verdict", "")
        if (
            equor_verdict in ("MODIFY", "DENY")
            and predicted_verdict == "PERMIT"
        ):
            mismatch = max(mismatch, 0.6)

        # Tertiary: drive_alignment_delta directly (from Equor check payload)
        alignment_delta = meta.get("drive_alignment_delta")
        if isinstance(alignment_delta, (int, float)):
            # Negative delta = alignment dropped below prediction
            if alignment_delta < 0:
                mismatch = max(mismatch, min(1.0, abs(float(alignment_delta))))

        error.constitutional_mismatch = max(0.0, min(1.0, mismatch))

    # ------------------------------------------------------------------
    # Clock helpers
    # ------------------------------------------------------------------

    def _sync_clock(self) -> None:
        """Sync the weight learner's clock to real time."""
        self._weight_learner.set_clock(self._elapsed_s())

    def _elapsed_s(self) -> float:
        """Seconds since service startup."""
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # Synapse event emission
    # ------------------------------------------------------------------

    async def _emit_prediction_error(
        self, error: FoveaPredictionError
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import (
                SynapseEvent,
                SynapseEventType,
            )

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_PREDICTION_ERROR),
                source_system=self.system_id,
                data={
                    "error_id": error.id,
                    "percept_id": error.percept_id,
                    "prediction_id": error.prediction_id,
                    "content_error": round(error.content_error, 4),
                    "temporal_error": round(error.temporal_error, 4),
                    "magnitude_error": round(error.magnitude_error, 4),
                    "source_error": round(error.source_error, 4),
                    "category_error": round(error.category_error, 4),
                    "causal_error": round(error.causal_error, 4),
                    "economic_error": round(error.economic_error, 4),
                    "precision_weighted_salience": round(
                        error.precision_weighted_salience, 4
                    ),
                    "habituated_salience": round(
                        error.habituated_salience, 4
                    ),
                    "dominant_error_type": error.get_dominant_error_type().value,
                    "routes": error.routes,
                },
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError):
            self._logger.debug(
                "synapse_emit_skipped",
                reason="event_type_not_registered",
            )

    async def _emit_workspace_ignition(
        self, error: FoveaPredictionError
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import (
                SynapseEvent,
                SynapseEventType,
            )

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_WORKSPACE_IGNITION),
                source_system=self.system_id,
                data={
                    "percept_id": error.percept_id,
                    "salience": round(error.habituated_salience, 4),
                    "prediction_error_id": error.id,
                    "threshold": round(
                        self._bridge.dynamic_threshold.current, 4
                    ),
                },
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError):
            self._logger.debug(
                "synapse_emit_skipped",
                reason="event_type_not_registered",
            )

    async def _emit_habituation_decay(
        self, error: FoveaPredictionError
    ) -> None:
        """Emit HABITUATION_DECAY event per spec Section IX."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_HABITUATION_DECAY),
                source_system=self.system_id,
                data={
                    "error_signature": error.get_signature(),
                    "habituation_level": round(error.habituation_level, 4),
                },
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError):
            self._logger.debug("synapse_emit_skipped", reason="event_type_not_registered")

    async def _emit_dishabituation(
        self, error: FoveaPredictionError, dishabituation_info: dict[str, float]
    ) -> None:
        """Emit DISHABITUATION event per spec Section IX."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Spec payload: {error_signature, expected_magnitude, actual_magnitude}
            sig = error.get_signature()
            exp = round(dishabituation_info.get("expected_magnitude", 0.0), 4)
            act = round(dishabituation_info.get("actual_magnitude", 0.0), 4)

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_DISHABITUATION),
                source_system=self.system_id,
                data={"error_signature": sig, "expected_magnitude": exp, "actual_magnitude": act},
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError):
            self._logger.debug("synapse_emit_skipped", reason="event_type_not_registered")

    async def _emit_attention_profile_update(
        self,
        weight_deltas: dict[str, float],
        dominant_type: str,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import (
                SynapseEvent,
                SynapseEventType,
            )

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_ATTENTION_PROFILE_UPDATE),
                source_system=self.system_id,
                data={
                    "weight_deltas": {
                        k: round(v, 6) for k, v in weight_deltas.items()
                    },
                    "dominant_error_type": dominant_type,
                    "current_weights": {
                        k: round(v, 4)
                        for k, v in self._weight_learner.weights.items()
                    },
                },
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError) as e:
            self._logger.debug(
                "synapse_emit_skipped",
                reason="event_type_not_registered",
            )
            # Track ValueError (likely dict key type mismatch) and report to Thymos after 3 occurrences
            if isinstance(e, ValueError):
                self._serialization_error_counts["fovea_attention_profile_update"] = (
                    self._serialization_error_counts.get("fovea_attention_profile_update", 0) + 1
                )
                count = self._serialization_error_counts["fovea_attention_profile_update"]
                if count == 3 and self._event_bus is not None:
                    try:
                        import hashlib

                        from systems.synapse.types import SynapseEvent, SynapseEventType
                        from systems.thymos.types import Incident, IncidentClass, IncidentSeverity

                        incident = Incident(
                            incident_class=IncidentClass.CONTRACT_VIOLATION,
                            severity=IncidentSeverity.MEDIUM,
                            fingerprint=hashlib.md5(b"fovea_serialization_error").hexdigest(),
                            source_system="fovea",
                            error_type="serialization_error",
                            error_message="Dict key must be str in event payload",
                            context={"event_type": "fovea_attention_profile_update"}
                        )
                        await self._event_bus.emit(
                            SynapseEvent(
                                event_type=SynapseEventType.SYSTEM_FAILED,
                                source_system="fovea",
                                data={"incident": incident.model_dump()}
                            )
                        )
                    except Exception as _report_exc:
                        self._logger.warning(
                            "failed_to_report_serialization_error",
                            error=str(_report_exc)
                        )

    async def _emit_habituation_complete(self, info: Any) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import (
                SynapseEvent,
                SynapseEventType,
            )

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_HABITUATION_COMPLETE),
                source_system=self.system_id,
                data={
                    "error_signature": info.signature,
                    "habituation_level": round(info.habituation_level, 4),
                    "times_seen": info.times_seen,
                    "times_led_to_update": info.times_led_to_update,
                    "diagnosis": info.diagnosis,
                },
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError):
            self._logger.debug(
                "synapse_emit_skipped",
                reason="event_type_not_registered",
            )

    async def _emit_internal_prediction_error(
        self, error: InternalPredictionError
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import (
                SynapseEvent,
                SynapseEventType,
            )

            event = SynapseEvent(
                event_type=SynapseEventType(
                    FOVEA_INTERNAL_PREDICTION_ERROR
                ),
                source_system=self.system_id,
                data={
                    "error_id": error.id,
                    "internal_error_type": error.internal_error_type.value,
                    "predicted_state": error.predicted_state,
                    "actual_state": error.actual_state,
                    "precision_weighted_salience": round(
                        error.precision_weighted_salience, 4
                    ),
                    "route_to": (
                        error.routes[0] if error.routes else "unknown"
                    ),
                    "precision_multiplier": error.precision_multiplier,
                },
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError):
            self._logger.debug(
                "synapse_emit_skipped",
                reason="event_type_not_registered",
            )

    # ------------------------------------------------------------------
    # Autonomy gap closure: FOVEA_DIAGNOSTIC_REPORT + parameter control
    # ------------------------------------------------------------------

    def _subscribe_to_parameter_adjustments(self, event_bus: EventBus) -> None:
        """Subscribe to FOVEA_PARAMETER_ADJUSTMENT from Equor/Nova.

        Allows the organism to tune Fovea's routing thresholds and sensitivity
        parameters at runtime without a service restart. Closes the gap where
        all routing decisions were hardcoded and uninfluenceable by the LLM.
        """
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.FOVEA_PARAMETER_ADJUSTMENT,
                self._on_parameter_adjustment,
            )
            # Also subscribe to EVO_PARAMETER_ADJUSTED for workspace curiosity params
            event_bus.subscribe(
                SynapseEventType.EVO_PARAMETER_ADJUSTED,
                self._on_evo_workspace_param_adjusted,
            )
            self._logger.info("subscribed_to_parameter_adjustments")
        except (ImportError, ValueError):
            self._logger.debug(
                "parameter_adjustment_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_evo_workspace_param_adjusted(self, event: Any) -> None:
        """
        Handle EVO_PARAMETER_ADJUSTED for atune.workspace.* curiosity parameters.

        Routes Evo parameter adjustments to the GlobalWorkspace's curiosity API.
        Non-workspace params are silently ignored - only atune.workspace.* prefix
        is consumed here; other params flow to their own system handlers.
        """
        try:
            data = event.data if hasattr(event, "data") else {}
            param_name: str = str(data.get("parameter_name", data.get("param_name", "")))
            new_value = data.get("new_value", data.get("value", None))
            if new_value is None or not param_name.startswith("atune.workspace."):
                return
            # Strip the prefix to get the workspace-local param name
            local_name = param_name.removeprefix("atune.workspace.")
            workspace = getattr(self._bridge, "_workspace", None)
            if workspace is None:
                return
            # Compute delta from defaults (Evo sends absolute new_value; workspace uses delta API)
            from systems.evo.types import PARAMETER_DEFAULTS
            default = PARAMETER_DEFAULTS.get(param_name, None)
            if default is None:
                return
            delta = float(new_value) - default
            workspace.adjust_param(local_name, delta)
            self._logger.info(
                "workspace_curiosity_param_adjusted",
                param=local_name,
                new_value=round(float(new_value), 5),
            )
        except Exception:
            pass

    async def _on_parameter_adjustment(self, event: Any) -> None:
        """
        Handle FOVEA_PARAMETER_ADJUSTMENT from Equor/Nova.

        Supported adjustment_types:
          - "routing_threshold_equor" - constitutional mismatch threshold for Equor routing
          - "routing_threshold_oneiros" - constitutional mismatch threshold for Oneiros routing
          - "economic_route_threshold" - economic error threshold for Oikos+Evo routing
          - "economic_workspace_threshold" - economic error threshold for workspace routing
          - "threshold_percentile" - dynamic ignition threshold percentile (affects all routing)
          - "habituation_speed" - scales the habituation increment (0.5–2.0× multiplier)

        All adjustments are clamped to safe ranges. Emits FOVEA_DIAGNOSTIC_REPORT
        immediately so the requesting system can confirm the change.
        """
        import asyncio as _asyncio

        data = event.data if hasattr(event, "data") else {}
        adjustment_type = str(data.get("adjustment_type", ""))
        raw_value = data.get("value", None)
        reason = str(data.get("reason", ""))
        source = str(data.get("source_system", getattr(event, "source_system", "unknown")))

        if raw_value is None:
            return
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return

        applied = False
        clamped_value = value

        if adjustment_type == "routing_threshold_equor":
            clamped_value = max(0.1, min(0.9, value))
            self._constitutional_equor_threshold = clamped_value
            applied = True
        elif adjustment_type == "routing_threshold_oneiros":
            clamped_value = max(0.1, min(0.9, value))
            self._constitutional_oneiros_threshold = clamped_value
            applied = True
        elif adjustment_type == "economic_route_threshold":
            clamped_value = max(0.05, min(0.9, value))
            self._economic_route_threshold = clamped_value
            applied = True
        elif adjustment_type == "economic_workspace_threshold":
            clamped_value = max(0.1, min(0.95, value))
            self._economic_workspace_threshold = clamped_value
            applied = True
        elif adjustment_type == "threshold_percentile":
            clamped_value = max(30.0, min(95.0, value))
            if hasattr(self._bridge.dynamic_threshold, "set_percentile"):
                self._bridge.dynamic_threshold.set_percentile(clamped_value)  # type: ignore[attr-defined]
            applied = True
        elif adjustment_type == "habituation_speed":
            scale = max(0.5, min(2.0, value))
            hab_engine = self._bridge.habituation_engine
            if hasattr(hab_engine, "_increment"):
                hab_engine._increment = _HABITUATION_INCREMENT * scale  # type: ignore[attr-defined]
                clamped_value = hab_engine._increment
                applied = True

        if applied:
            self._logger.info(
                "parameter_adjusted",
                adjustment_type=adjustment_type,
                requested=round(value, 4),
                applied=round(clamped_value, 4),
                reason=reason,
                source=source,
            )
            _asyncio.ensure_future(self._emit_diagnostic_report())

    async def _emit_diagnostic_report(self) -> None:
        """
        Emit FOVEA_DIAGNOSTIC_REPORT with previously invisible internal state.

        Surfaces every 50 errors (same cadence as fitness signal):
        - Per-dimension precision weights (learned accuracy profile per error type)
        - Habituation engine stats (entry count, current increment)
        - Economic per-source trend data (worst source, trend velocities)
        - Unresolved error backlog count (Oneiros processing pressure)
        - All runtime-adjustable parameter values (thresholds, intervals)
        - Weight learning state (reinforcements, decays, false alarms, learning rate)

        Nova/Evo/RE subscribe to reason about attention calibration at planning time.
        """
        if self._event_bus is None:
            return

        current_weights = self._weight_learner.get_raw_weights()

        hab_engine = self._bridge.habituation_engine
        hab_stats: dict[str, Any] = {
            "entry_count": getattr(hab_engine, "entry_count", 0),
            "current_increment": round(
                getattr(hab_engine, "_increment", _HABITUATION_INCREMENT), 6
            ),
            "max_cap": _MAX_HABITUATION,
        }

        econ_trends: dict[str, Any] = {
            "worst_source": self._economic_model.worst_source,
            "worst_source_error": round(self._economic_model.worst_source_error, 4),
            "revenue_prediction_error": round(
                getattr(self._economic_model, "revenue_prediction_error", 0.0), 4
            ),
            "efficiency_trend_error": round(
                getattr(self._economic_model, "efficiency_trend_error", 0.0), 4
            ),
            "is_warmed_up": self._economic_model.is_warmed_up,
        }

        unresolved_count = sum(
            1 for t in self._weight_learner.recent_errors if not t.resolved
        )

        param_config = {
            "constitutional_equor_threshold": self._constitutional_equor_threshold,
            "constitutional_oneiros_threshold": self._constitutional_oneiros_threshold,
            "economic_route_threshold": self._economic_route_threshold,
            "economic_workspace_threshold": self._economic_workspace_threshold,
            "dynamic_threshold_current": round(self._bridge.dynamic_threshold.current, 4),
            "divergence_emit_interval": self._divergence_emit_interval,
            "diagnostic_emit_interval": self._diagnostic_emit_interval,
            "backpressure_warning_threshold": _BACKPRESSURE_WARNING_THRESHOLD,
        }

        wl = self._weight_learner
        weight_learning_state = {
            "reinforcements": getattr(wl, "reinforcements", 0),
            "decays": getattr(wl, "decays", 0),
            "false_alarms": getattr(wl, "false_alarms", 0),
            "learning_rate": getattr(wl, "learning_rate", 0.0),
            "false_alarm_decay_rate": getattr(wl, "false_alarm_decay_rate", 0.0),
        }

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_DIAGNOSTIC_REPORT),
                source_system=self.system_id,
                data={
                    "instance_id": self._instance_id,
                    "precision_weights": {k: round(v, 4) for k, v in current_weights.items()},
                    "habituation_stats": hab_stats,
                    "economic_trends": econ_trends,
                    "unresolved_backlog": unresolved_count,
                    "param_config": param_config,
                    "weight_learning_state": weight_learning_state,
                    "kl_divergence_last": round(self._last_kl_divergence, 6),
                    "fleet_sample_count": len(self._fleet_weight_samples),
                },
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError):
            self._logger.debug("diagnostic_report_emit_skipped", reason="event_type_unavailable")

    async def _emit_backpressure_warning(self, unresolved_count: int) -> None:
        """
        Emit FOVEA_BACKPRESSURE_WARNING when error backlog exceeds threshold.

        Signals that Fovea is accumulating errors faster than Oneiros/Logos
        can process them. Nova should raise the dynamic ignition threshold via
        FOVEA_PARAMETER_ADJUSTMENT. Oneiros should schedule early consolidation.
        """
        if self._event_bus is None:
            return

        top_domains = [
            {"domain": t.dominant_type.value, "salience": round(t.salience, 3)}
            for t in sorted(
                (x for x in self._weight_learner.recent_errors if not x.resolved),
                key=lambda x: x.salience,
                reverse=True,
            )[:5]
        ]

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_BACKPRESSURE_WARNING),
                source_system=self.system_id,
                data={
                    "unresolved_count": unresolved_count,
                    "threshold": _BACKPRESSURE_WARNING_THRESHOLD,
                    "top_error_domains": top_domains,
                    "recommendation": (
                        "Oneiros should schedule early sleep consolidation. "
                        "Nova may raise Fovea threshold via FOVEA_PARAMETER_ADJUSTMENT "
                        "with adjustment_type='threshold_percentile'."
                    ),
                },
            )
            await self._event_bus.emit(event)
            self._logger.warning(
                "fovea_backpressure_warning",
                unresolved=unresolved_count,
                threshold=_BACKPRESSURE_WARNING_THRESHOLD,
            )
        except (ValueError, ImportError):
            self._logger.debug("backpressure_warning_emit_skipped", reason="event_type_unavailable")

    async def _check_calibration_alert(self) -> None:
        """
        Part A: Autonomy Gap Closure - Check for consecutive poor cycles.

        Tracks:
        - Low TPR (< 0.6): Fovea isn't catching real errors
        - High false alarm rate (> 0.4): Fovea is too sensitive

        If either condition persists for 5+ cycles, emit FOVEA_CALIBRATION_ALERT
        so Evo can generate targeted attention-tuning hypotheses.
        """
        if self._event_bus is None:
            return

        wl = self._weight_learner
        total_reinforcements = max(1, wl.reinforcements + wl.false_alarms)
        tpr = wl.reinforcements / total_reinforcements if total_reinforcements > 0 else 0.5
        false_alarm_rate = (
            wl.false_alarms / total_reinforcements if total_reinforcements > 0 else 0.0
        )

        # Track consecutive poor TPR cycles
        if tpr < 0.6:
            self._consecutive_poor_tpr += 1
        else:
            self._consecutive_poor_tpr = 0

        # Track consecutive high false alarm cycles
        if false_alarm_rate > 0.4:
            self._consecutive_high_false_alarm += 1
        else:
            self._consecutive_high_false_alarm = 0

        # Emit alert if either counter reaches 5
        alert_emitted = False
        if self._consecutive_poor_tpr >= 5:
            await self._emit_calibration_alert(
                alert_type="low_tpr",
                current_value=tpr,
                consecutive_cycles=self._consecutive_poor_tpr,
            )
            alert_emitted = True
            self._consecutive_poor_tpr = 0  # Reset after alert

        if self._consecutive_high_false_alarm >= 5:
            await self._emit_calibration_alert(
                alert_type="high_false_alarm",
                current_value=false_alarm_rate,
                consecutive_cycles=self._consecutive_high_false_alarm,
            )
            alert_emitted = True
            self._consecutive_high_false_alarm = 0  # Reset after alert

        if alert_emitted:
            self._logger.info(
                "calibration_alert_emitted",
                tpr=round(tpr, 4),
                false_alarm_rate=round(false_alarm_rate, 4),
            )

    async def _emit_calibration_alert(
        self, alert_type: str, current_value: float, consecutive_cycles: int
    ) -> None:
        """
        Emit FOVEA_CALIBRATION_ALERT for Evo to generate attention-tuning hypotheses.

        alert_type: "low_tpr" or "high_false_alarm"
        current_value: The problematic metric value
        consecutive_cycles: How many cycles it's been poor
        """
        if self._event_bus is None:
            return

        threshold_params = {
            "percentile": self._bridge.dynamic_threshold._percentile,
            "floor": self._bridge.dynamic_threshold._floor,
            "ceiling": self._bridge.dynamic_threshold._ceiling,
        }

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event = SynapseEvent(
                event_type=SynapseEventType(FOVEA_CALIBRATION_ALERT),
                source_system=self.system_id,
                data={
                    "alert_type": alert_type,
                    "current_value": round(current_value, 4),
                    "consecutive_cycles": consecutive_cycles,
                    "threshold_params": threshold_params,
                    "recommendation": (
                        "Lower ignition floor or raise precision to improve TPR"
                        if alert_type == "low_tpr"
                        else "Raise ignition floor or reduce curiosity boost to reduce false alarms"
                    ),
                },
            )
            await self._event_bus.emit(event)
        except (ValueError, ImportError):
            self._logger.debug("calibration_alert_emit_skipped", reason="event_type_unavailable")

    def _subscribe_to_system_modulation(self, event_bus: EventBus) -> None:
        """Subscribe to SYSTEM_MODULATION from Skia's VitalityCoordinator."""
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.SYSTEM_MODULATION,
                self._on_system_modulation,
            )
            self._logger.info("subscribed_to_system_modulation")
        except (ImportError, ValueError):
            self._logger.debug(
                "system_modulation_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle SYSTEM_MODULATION from Skia's VitalityCoordinator."""
        data = getattr(event, "data", {}) or {}
        halt_systems: list[str] = data.get("halt_systems", [])
        modulate: dict = data.get("modulate", {})
        modulation_id: str = data.get("modulation_id", "")

        if "fovea" in halt_systems:
            self._modulation_halted = True
            self._logger.warning("system_modulation_halted", modulation_id=modulation_id)
        elif "fovea" in modulate:
            self._modulation_halted = False
            self._apply_modulation_directives(modulate["fovea"])

        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                    source_system=SystemID.FOVEA,
                    data={
                        "system": "fovea",
                        "modulation_id": modulation_id,
                        "halted": self._modulation_halted,
                    },
                ))
            except (ValueError, ImportError):
                self._logger.debug("system_modulation_ack_skipped", reason="event_type_unavailable")

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply Skia modulation directives to Fovea runtime state."""
        # No specific Skia directives defined for fovea - halt-only modulation
        self._logger.info("system_modulation_directives_applied", directives=directives)

    # ------------------------------------------------------------------
    # Logos compression cycle subscription
    # ------------------------------------------------------------------

    def _subscribe_to_compression_events(self, event_bus: EventBus) -> None:
        """Subscribe to COMPRESSION_CYCLE_COMPLETE from Logos.

        After Logos evicts or distills items, any salience weights Fovea
        learned for that domain are stale - the world model has changed.
        We decay the relevant error-type weight proportional to eviction
        volume so that Fovea re-learns from fresh data.
        """
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.COMPRESSION_CYCLE_COMPLETE,
                self._on_compression_cycle_complete,
            )
            self._logger.info("subscribed_to_compression_cycle_complete")
        except (ImportError, ValueError):
            self._logger.debug(
                "compression_cycle_subscription_skipped",
                reason="event_type_unavailable",
            )

    async def _on_compression_cycle_complete(self, event: Any) -> None:
        """Handle COMPRESSION_CYCLE_COMPLETE from Logos.

        Logos reports how many items were evicted and distilled and how much
        MDL improved.  We use this to decay attention weights for the affected
        domain:

        - ``items_evicted``  - raw data removed; weights are most stale here.
        - ``items_distilled``- converted to schemas; less decay, model still valid.
        - ``mdl_improvement``- how much the world model compressed; large improvement
          means the domain is now well-understood so moderate decay is safe.

        Decay formula:
            decay = false_alarm_decay × clamp(eviction_ratio, 0.1, 1.0)

        where eviction_ratio = items_evicted / max(items_evicted + items_distilled, 1).
        If domain is provided in the payload we restrict the decay to that
        error-type key; otherwise we apply a small uniform decay to all weights.
        """
        data = getattr(event, "data", {}) or {}
        items_evicted: int = int(data.get("items_evicted", 0))
        items_distilled: int = int(data.get("items_distilled", 0))
        domain: str = data.get("domain", "")

        total = max(items_evicted + items_distilled, 1)
        eviction_ratio = min(1.0, items_evicted / total)

        if eviction_ratio < 0.05:
            # Negligible eviction - no meaningful staleness introduced
            return

        decay = self._weight_learner.false_alarm_decay_rate * eviction_ratio

        error_type_key = self._map_domain_to_error_type(domain) if domain else None

        if error_type_key is not None:
            old_weight = self._weight_learner.weights.get(error_type_key, 0.0)
            new_weight = self._weight_learner.adjust_weight(error_type_key, -decay)
            self._logger.info(
                "compression_cycle_weight_decayed",
                domain=domain,
                error_type=error_type_key,
                items_evicted=items_evicted,
                items_distilled=items_distilled,
                eviction_ratio=round(eviction_ratio, 3),
                old_weight=round(old_weight, 4),
                new_weight=round(new_weight, 4),
            )
        else:
            # Domain unknown - apply a smaller uniform decay to all weights
            uniform_decay = decay * 0.3
            adjusted: dict[str, float] = {}
            for key in list(self._weight_learner.weights):
                adjusted[key] = self._weight_learner.adjust_weight(key, -uniform_decay)
            self._logger.info(
                "compression_cycle_uniform_weight_decay",
                items_evicted=items_evicted,
                items_distilled=items_distilled,
                eviction_ratio=round(eviction_ratio, 3),
                uniform_decay=round(uniform_decay, 4),
                keys_affected=len(adjusted),
            )

        await self._weight_learner.persist_weights()
