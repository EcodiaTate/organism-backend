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
from .habituation import _HABITUATION_INCREMENT, _MAX_HABITUATION
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

        # Economic prediction model — tracks revenue/cost EMA and emits
        # ECONOMIC error dimension when actuals diverge from predictions.
        self._economic_model = EconomicPredictionModel()

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
        self._logger.info("event_bus_wired")

    def set_workspace(self, workspace: GlobalWorkspace) -> None:
        self._bridge.set_workspace(workspace)
        self._logger.info("workspace_wired")

    def set_world_model(self, world_model: LogosWorldModel) -> None:
        """Hot-swap the world model when Logos comes online."""
        self._bridge.set_world_model(world_model)
        self._logger.info("world_model_swapped")

    def set_neo4j_driver(self, driver: Any, instance_id: str = "") -> None:
        """Wire Neo4j driver post-construction for persistence."""
        self._neo4j_driver = driver
        if instance_id:
            self._instance_id = instance_id
        self._weight_learner.set_neo4j_driver(driver, instance_id)
        self._bridge.habituation_engine.set_neo4j_driver(driver, instance_id)

    def get_metrics(self) -> FoveaMetrics:
        """Return current Fovea metrics snapshot (public API — avoids _bridge access)."""
        return self._bridge.get_metrics()

    @property
    def weight_learner(self) -> AttentionWeightLearner:
        """Return the weight learner for external inspection (public API — avoids _bridge access)."""
        return self._weight_learner

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        # Restore persisted state from Neo4j
        await self._weight_learner.restore_weights()
        await self._bridge.habituation_engine.restore_state()
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

        # Constitutional mismatch detection: if the percept carries drive
        # alignment signals (populated by Nova/Equor in percept.metadata),
        # score the divergence and set constitutional_mismatch before routing.
        # This gates EQUOR + ONEIROS routing in compute_routing().
        self._inject_constitutional_mismatch(error, percept)

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
        # This is a speciation signal — instances with divergent attention phenotypes
        # are developing distinct ecological niches.
        self._errors_since_last_divergence += 1
        if self._errors_since_last_divergence >= self._divergence_emit_interval:
            await self._emit_attentional_divergence()
            self._errors_since_last_divergence = 0

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

        External anomaly signals — such as Synapse entering STRESS rhythm state or
        Simula rolling back an evolution proposal — are structurally equivalent to
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
    # Affect state subscription (P3 — precision τ coupling)
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

        High arousal → raise ignition threshold (more conservative — only attend
        to very surprising things during high arousal, reducing noise).
        Low arousal → lower threshold (more permissive — attend to subtler errors).
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
            self._logger.info("subscribed_to_hypothesis_outcomes")
        except (ImportError, ValueError):
            self._logger.debug(
                "hypothesis_subscription_skipped",
                reason="event_type_unavailable",
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
        """Serialise Fovea's heritable state: weights + habituation params."""
        payload = {
            "error_weights": self._weight_learner.get_raw_weights(),
            "habituation_increment": _HABITUATION_INCREMENT,
            "max_habituation_cap": _MAX_HABITUATION,
            "learning_rate": self._weight_learner.learning_rate,
            "false_alarm_decay": self._weight_learner.false_alarm_decay_rate,
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
        if "learning_rate" in payload:
            self._weight_learner.set_learning_rate(float(payload["learning_rate"]))
        if "false_alarm_decay" in payload:
            self._weight_learner.set_false_alarm_decay(float(payload["false_alarm_decay"]))
        self._logger.info("seeded_from_parent_genome", payload_keys=list(payload.keys()))
        return True

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
        from Axon — all Axon→Fovea comms now flow via Synapse bus.
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

        # Cache for resolve — intent_id → internal prediction_id
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
        — the organism notices when its actions don't produce expected
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

        if composite_error < 0.3:
            return

        # Build a FoveaPredictionError in the economic dimension
        error = FoveaPredictionError(
            economic_error=min(1.0, composite_error),
        )
        self._weight_learner.apply_learned_weights(error)
        error.compute_precision_weighted_salience()
        self._bridge.habituation_engine.apply_habituation(error)
        error.compute_routing(self._bridge.dynamic_threshold.current)

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
            # No fleet data — return uniform distribution as neutral baseline
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
