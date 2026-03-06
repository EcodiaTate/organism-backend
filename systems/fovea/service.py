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

import time
from typing import TYPE_CHECKING, Any

import structlog

from .integration import FoveaAtuneBridge
from .internal import InternalPredictionEngine
from .learning import AttentionWeightLearner
from .protocols import LogosWorldModel, StubWorldModel
from .types import (
    AttentionProfile,
    ErrorRoute,
    FoveaMetrics,
    FoveaPredictionError,
    InternalErrorType,
    InternalPredictionError,
)

if TYPE_CHECKING:
    from primitives.percept import Percept
    from systems.atune.workspace import GlobalWorkspace
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.fovea.service")


# Synapse event type string constants
FOVEA_PREDICTION_ERROR = "fovea_prediction_error"
FOVEA_HABITUATION_DECAY = "fovea_habituation_decay"
FOVEA_DISHABITUATION = "fovea_dishabituation"
FOVEA_WORKSPACE_IGNITION = "fovea_workspace_ignition"
FOVEA_ATTENTION_PROFILE_UPDATE = "fovea_attention_profile_update"
FOVEA_HABITUATION_COMPLETE = "fovea_habituation_complete"
FOVEA_INTERNAL_PREDICTION_ERROR = "fovea_internal_prediction_error"


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
    ) -> None:
        wm = world_model or StubWorldModel()
        self._bridge = FoveaAtuneBridge(
            wm,
            threshold_percentile=threshold_percentile,
            threshold_window=threshold_window,
        )
        self._weight_learner = AttentionWeightLearner(learning_rate=learning_rate)
        self._bridge.set_weight_applicator(self._weight_learner.apply_learned_weights)
        # Wire habituation engine to weight learner so world model update correlations
        # propagate record_update() calls, fixing stochastic/learning_failure diagnosis.
        self._weight_learner.set_habituation_engine(
            self._bridge.habituation_engine
        )
        self._internal_engine = InternalPredictionEngine()
        self._event_bus: EventBus | None = None
        self._logger = logger.bind(component="fovea_service")
        self._started = False
        self._start_time: float = 0.0
        self._serialization_error_counts: dict[str, int] = {}  # Track dict key errors per event type

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire the Synapse event bus and subscribe to WORLD_MODEL_UPDATED."""
        self._event_bus = event_bus
        self._subscribe_to_logos_updates(event_bus)
        self._logger.info("event_bus_wired")

    def set_workspace(self, workspace: GlobalWorkspace) -> None:
        self._bridge.set_workspace(workspace)
        self._logger.info("workspace_wired")

    def set_world_model(self, world_model: LogosWorldModel) -> None:
        """Hot-swap the world model when Logos comes online."""
        self._bridge._prediction_engine._world_model = world_model
        self._bridge._precision_computer._world_model = world_model
        self._logger.info("world_model_swapped")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        self._started = True
        self._start_time = time.monotonic()
        self._logger.info("fovea_started")

    async def shutdown(self) -> None:
        self._started = False
        self._logger.info("fovea_stopped")

    # ------------------------------------------------------------------
    # ManagedSystemProtocol
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        metrics = self._bridge.get_metrics()
        return {
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
        }

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

        # Emit HABITUATION_DECAY whenever habituation is actively applied
        # (spec event: {error_signature, habituation_level})
        if error.habituation_level > 0.0 and self._bridge._last_dishabituation is None:
            await self._emit_habituation_decay(error)

        # Emit DISHABITUATION when the habituation engine detected a sudden magnitude change
        # (spec event: {error_signature, expected_magnitude, actual_magnitude})
        dishabituation_info = self._bridge._last_dishabituation
        if dishabituation_info is not None:
            await self._emit_dishabituation(error, dishabituation_info)
            self._bridge._last_dishabituation = None

        # Phase C: check for habituation-complete
        hab_complete = self._bridge._last_habituation_complete
        if hab_complete is not None:
            await self._emit_habituation_complete(hab_complete)
            self._bridge._last_habituation_complete = None

        return error

    # ------------------------------------------------------------------
    # Phase C: weight learning integration
    # ------------------------------------------------------------------

    def _subscribe_to_logos_updates(self, event_bus: EventBus) -> None:
        """Subscribe to WORLD_MODEL_UPDATED events from Logos."""
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

    def get_current_attention_profile(self) -> AttentionProfile:
        """
        Build a current attentional profile snapshot for Alive telemetry.

        Mean error values are computed from recent tracked errors; the dynamic
        ignition threshold and habituation count come from the bridge.
        """
        recent = list(self._weight_learner._recent_errors)

        # Accumulators for each error dimension
        content = timing = magnitude = source = category = causal = 0.0
        top_salience: float = -1.0
        top_summary: str | None = None

        for tracked in recent:
            content += 0.0   # _TrackedError only stores dominant_type + salience
            # We use salience as a proxy for each error dimension via dominant_type
            # to avoid coupling to the full error object (not stored in tracker).
            # A zero-floor per-dimension approach: weight salience by dominance.
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
            for tracked in self._weight_learner._recent_errors
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

        for tracked in self._weight_learner._recent_errors:
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
