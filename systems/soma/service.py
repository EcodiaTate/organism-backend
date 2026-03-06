"""
EcodiaOS — Soma Service (The Interoceptive Predictive Substrate)

The organism's felt sense of being alive. Soma predicts internal states,
computes the gap between where the organism is and where it needs to be,
and emits the allostatic signals that make every other system care about
staying viable.

Cognitive cycle role (step 0 — SENSE):
  Soma runs FIRST in every theta cycle, before Atune. It reads from all
  systems, predicts multi-horizon states, computes allostatic errors,
  and emits an AllostaticSignal that downstream systems consume.

Iron Rules:
  - Total cycle budget: 5ms. Soma is the fastest system in the organism.
  - No LLM calls. No database calls. No network calls during cycle.
  - All reads are in-memory from system references.
  - If Soma fails, the organism degrades gracefully to pre-Soma behaviour.
  - Soma is advisory, not commanding — systems MAY ignore the signal.

Interface:
  initialize()              — wire system refs, load config, seed attractors
  run_cycle()               — main theta cycle entry (sense → predict → emit)
  get_current_state()       — last computed interoceptive state
  get_current_signal()      — last emitted allostatic signal
  get_somatic_marker()      — snapshot for memory stamping
  get_errors()              — allostatic errors per horizon per dimension
  get_phase_position()      — attractor, bifurcation, trajectory info
  get_developmental_stage() — current maturation stage
  create_somatic_marker()   — create marker from current state
  somatic_rerank()          — boost memory candidates by somatic similarity
  update_dynamics_matrix()  — Evo updates cross-dimension coupling
  update_emotion_regions()  — Evo refines emotion boundaries
  shutdown()                — graceful teardown
  health()                  — self-health report for Synapse
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from systems.soma.adaptive_setpoints import AdaptiveSetpointLearner
from systems.soma.allostatic_controller import AllostaticController
from systems.soma.autonomic_protocol import AutonomicAction, AutonomicProtocol
from systems.soma.base import BaseAllostaticRegulator, BaseSomaPredictor
from systems.soma.cascade_predictor import CascadePredictor, CascadeSnapshot
from systems.soma.causal_flow import CausalFlowEngine, CausalFlowMap
from systems.soma.counterfactual import CounterfactualEngine
from systems.soma.curvature_analyzer import CurvatureAnalyzer, CurvatureMap
from systems.soma.developmental import DevelopmentalManager
from systems.soma.emergence import CausalEmergenceEngine, EmergenceReport
from systems.soma.emotions import ActiveEmotion, EmotionDetector, VoxisInteroceptiveInterface
from systems.soma.feedback_loops import ALLOSTATIC_LOOP_MAP, get_active_loop_signals
from systems.soma.fisher_manifold import FisherManifold, GeodesicDeviation
from systems.soma.interoceptive_broadcaster import (
    BroadcasterThresholds,
    InteroceptiveBroadcaster,
)
from systems.soma.interoceptor import Interoceptor
from systems.soma.loop_executor import LoopDispatch, LoopExecutor
from systems.soma.phase_space import PhaseSpaceModel
from systems.soma.phase_space_reconstructor import (
    PhaseSpaceReconstructor,
    PhaseSpaceReport,
)
from systems.soma.predictor import InteroceptivePredictor
from systems.soma.renormalization import RenormalizationEngine, RGFlowReport
from systems.soma.signal_buffer import SignalBuffer
from systems.soma.somatic_memory import SomaticMemoryIntegration
from systems.soma.state_vector import StateVectorConstructor
from systems.soma.temporal_depth import (
    TemporalDepthManager,
    build_financial_snapshot_from_oikos,
)
from systems.soma.temporal_derivatives import TemporalDerivativeEngine
from systems.soma.topology import (
    PersistenceDiagnosis,
    TopologicalAnalyzer,
)
from systems.soma.types import (
    ALL_DIMENSIONS,
    EMOTION_REGIONS,
    AllostaticSignal,
    CounterfactualTrace,
    DerivativeSnapshot,
    DevelopmentalStage,
    InteroceptiveDimension,
    InteroceptivePercept,
    InteroceptiveState,
    OrganismStateVector,
    SomaticMarker,
)

if TYPE_CHECKING:
    from config import SomaConfig
    from core.hotreload import NeuroplasticityBus
    from systems.soma.exteroception.types import ExteroceptivePressure
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

logger = structlog.get_logger("systems.soma")


class SomaService:
    """
    Soma — the EOS interoceptive predictive substrate.

    Coordinates twelve sub-systems:
      Interoceptor       — reads 9D sensed state from all systems (2ms)
      Predictor          — multi-horizon generative model (1ms)
      AllostaticCtl      — setpoint management, urgency, signal construction (0.5ms)
      PhaseSpace         — attractor detection, bifurcation mapping (2ms every 100 cycles)
      SomaticMemory      — marker creation, embodied retrieval reranking (1ms)
      TemporalDepth      — multi-scale prediction, temporal dissonance
      Counterfactual     — Oneiros REM counterfactual replay
      Developmental      — stage gating, maturation triggers
      SignalBuffer       — Phase A: bounded ring buffer for all Synapse events + logs
      StateVectorCtor    — Phase A: aggregates signals into per-system feature vectors
      DerivativeEngine   — Phase A: SG-filtered velocity/accel/jerk at 3 time scales
      Broadcaster        — Phase A: composes interoceptive percepts when thresholds exceeded
    """

    system_id: str = "soma"

    def __init__(
        self,
        config: SomaConfig,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
    ) -> None:
        self._config = config
        self._bus = neuroplasticity_bus

        # Sub-systems
        self._interoceptor = Interoceptor()
        self._predictor: BaseSomaPredictor = InteroceptivePredictor(
            buffer_size=config.trajectory_buffer_size,
            ewm_span=config.prediction_ewm_span,
        )
        self._controller: BaseAllostaticRegulator = AllostaticController(
            adaptation_alpha=config.setpoint_adaptation_alpha,
            urgency_threshold=config.urgency_threshold,
        )
        self._phase_space = PhaseSpaceModel(
            max_attractors=config.max_attractors,
            min_dwell_cycles=config.attractor_min_dwell_cycles,
            detection_enabled=config.bifurcation_detection_enabled,
        )
        self._somatic_memory = SomaticMemoryIntegration(
            rerank_boost=config.somatic_rerank_boost,
        )
        self._temporal_depth = TemporalDepthManager(config=config)
        self._counterfactual = CounterfactualEngine()
        self._developmental = DevelopmentalManager(
            initial_stage=DevelopmentalStage(config.initial_stage),
        )

        # ── Closed-loop regulation engines ──
        self._loop_executor = LoopExecutor()
        self._adaptive_setpoints = AdaptiveSetpointLearner(
            enabled=config.initial_stage != DevelopmentalStage.REFLEXIVE.value,
        )
        self._cascade_predictor = CascadePredictor()
        self._autonomic_protocol = AutonomicProtocol()

        # Cached results from new modules
        self._last_cascade_snapshot: CascadeSnapshot | None = None
        self._last_autonomic_actions: list[AutonomicAction] = []
        self._last_loop_dispatches: list[LoopDispatch] = []

        # Latest exogenous stress value — injected by ExteroceptionService via
        # inject_external_stress(); interoceptor reads it each sense() pass.
        self._external_stress: float = 0.0
        self._interoceptor.set_soma(self)

        # Cross-modal synesthesia — exteroceptive pressure (per-dimension)
        # Injected by ExteroceptionService, applied as deltas to sensed state
        self._exteroceptive_pressure: ExteroceptivePressure | None = None

        # State
        self._synapse_ref: Any = None  # Cached for hot-swap forwarding
        self._oikos_ref: Any = None  # Oikos service ref for financial snapshot reads
        self._financial_refresh_counter: int = 0
        self._cycle_count: int = 0
        self._current_state: InteroceptiveState | None = None
        self._current_signal: AllostaticSignal = AllostaticSignal.default()
        self._last_cycle_duration_ms: float = 0.0
        self._cycle_durations: deque[float] = deque(maxlen=100)
        self._phase_space_update_counter: int = 0
        self._enabled: bool = config.cycle_enabled

        # Stagger counters for deferred operations (Task 3)
        self._fisher_cycle_counter: int = 0
        self._fisher_cycle_interval: int = 10   # Fisher manifold: every 10 cycles
        self._vulnerability_cycle_counter: int = 0
        self._vulnerability_cycle_interval: int = 20  # vulnerability map: every 20 cycles
        self._body_scan_cycle_counter: int = 0
        self._body_scan_cycle_interval: int = 50  # full body scan: every 50 cycles

        # Timing breakdown accumulator for DEBUG log every 50 cycles (Task 1)
        self._breakdown_log_interval: int = 50

        # Warmup period tracking (Task 1)
        # Suppress soma_cycle_slow warnings during first 50 cycles (cold start of Fisher manifold)
        self._warmup_cycle_count: int = 50
        self._warmup_complete: bool = False

        # Emotion regions (Evo can update)
        self._emotion_regions: dict[str, dict[str, Any]] = dict(EMOTION_REGIONS)

        # Emergent emotion detector — runs every cycle (lightweight pattern matching)
        self._emotion_detector = EmotionDetector(emotion_regions=self._emotion_regions)
        self._voxis_interface = VoxisInteroceptiveInterface(self._emotion_detector)
        self._current_emotions: list[ActiveEmotion] = []

        # ── Phase A: Homeostatic Manifold engines ──
        self._manifold_enabled: bool = config.manifold_enabled
        self._signal_buffer = SignalBuffer(max_size=config.signal_buffer_size)
        self._state_vector_ctor = StateVectorConstructor()
        self._derivative_engine = TemporalDerivativeEngine(
            history_size=config.derivative_history_size,
        )
        self._broadcaster = InteroceptiveBroadcaster(
            thresholds=BroadcasterThresholds(
                velocity_norm_threshold=config.broadcaster_velocity_threshold,
                acceleration_norm_threshold=config.broadcaster_acceleration_threshold,
                jerk_norm_threshold=config.broadcaster_jerk_threshold,
                error_rate_threshold=config.broadcaster_error_rate_threshold,
                entropy_divergence_threshold=config.broadcaster_entropy_threshold,
                fast_slow_divergence_threshold=config.broadcaster_fast_slow_divergence,
            ),
        )
        self._event_bus_ref: EventBus | None = None
        self._last_window_time: float = 0.0
        self._current_state_vector: OrganismStateVector | None = None
        self._current_derivatives: DerivativeSnapshot | None = None
        self._current_percept: InteroceptivePercept | None = None

        # ── Phase B–E: Advanced Analysis Engines ──
        self._fisher_manifold = FisherManifold(
            window_size=config.fisher_window_size,
            baseline_capacity=config.fisher_baseline_capacity,
            calibration_threshold=config.fisher_calibration_threshold,
            min_samples_for_fisher=config.fisher_min_samples,
        )
        self._last_fisher_system_order: list[str] | None = None
        self._fisher_config = {
            "window_size": config.fisher_window_size,
            "baseline_capacity": config.fisher_baseline_capacity,
            "calibration_threshold": config.fisher_calibration_threshold,
            "min_samples_for_fisher": config.fisher_min_samples,
        }
        self._curvature_analyzer = CurvatureAnalyzer(
            k_neighbors=config.curvature_k_neighbors,
        )
        self._topological_analyzer = TopologicalAnalyzer(
            subsample_rate=config.topology_subsample_rate,
            compute_interval_cycles=config.deep_path_interval_cycles,
            max_homology_dim=config.topology_max_homology_dim,
        )
        self._emergence_engine = CausalEmergenceEngine(
            n_macro_states=config.emergence_n_macro_states,
            compute_interval_cycles=config.medium_path_interval_cycles,
        )
        self._causal_flow_engine = CausalFlowEngine(
            history_length=config.causal_flow_history_length,
            lag_k=config.causal_flow_lag_k,
            compute_interval_cycles=config.medium_path_interval_cycles,
        )
        self._renormalization_engine = RenormalizationEngine(
            scales=config.renormalization_scales,
        )
        self._phase_space_reconstructor = PhaseSpaceReconstructor(
            series_buffer=config.psr_series_buffer,
        )
        self._fisher_deviation_threshold: float = config.fisher_deviation_broadcast_threshold

        # Path counters (independent of phase_space_update_counter)
        self._medium_path_counter: int = 0
        self._deep_path_counter: int = 0
        self._medium_path_interval: int = config.medium_path_interval_cycles
        self._deep_path_interval: int = config.deep_path_interval_cycles

        # Cached results from analysis paths
        self._last_geodesic_deviation: GeodesicDeviation | None = None
        self._last_emergence_report: EmergenceReport | None = None
        self._last_causal_flow_map: CausalFlowMap | None = None
        self._last_rg_flow_report: RGFlowReport | None = None
        self._last_persistence_diagnosis: PersistenceDiagnosis | None = None
        self._last_phase_space_report: PhaseSpaceReport | None = None
        self._last_curvature_map: CurvatureMap | None = None

        # Background task handles for medium/deep paths
        self._medium_path_task: asyncio.Task[None] | None = None
        self._deep_path_task: asyncio.Task[None] | None = None

        # Fisher manifold input cache — skip update if delta < threshold (Task 2)
        self._fisher_last_flat: np.ndarray | None = None
        self._fisher_cache_delta_threshold: float = 0.01
        self._fisher_cache_max_age_s: float = 5.0  # never serve stale beyond 5s
        self._fisher_last_updated_ts: float = 0.0

    async def initialize(self) -> None:
        """Initialize sub-systems and register with NeuroplasticityBus."""
        self._temporal_depth.set_stage(self._developmental.stage)
        self._counterfactual.set_dynamics(self._predictor.dynamics_matrix)

        # Task 3: Pre-initialize Fisher manifold with known system list to avoid cold-start spike
        # Build a default manifold state from the registered system order
        if self._manifold_enabled:
            self._prewarm_fisher_manifold()

        # Register hot-reloadable strategies with the NeuroplasticityBus
        if self._bus is not None:
            self._bus.register(
                base_class=BaseSomaPredictor,
                registration_callback=self._on_predictor_evolved,
                system_id="soma",
            )
            self._bus.register(
                base_class=BaseAllostaticRegulator,
                registration_callback=self._on_regulator_evolved,
                system_id="soma",
            )

        logger.info(
            "soma_initialized",
            stage=self._developmental.stage.value,
            attractors=self._phase_space.attractor_count,
            manifold_enabled=self._manifold_enabled,
        )

    async def shutdown(self) -> None:
        """Graceful teardown — deregister from NeuroplasticityBus."""
        # Cancel background analysis tasks
        for task in (self._medium_path_task, self._deep_path_task):
            if task is not None and not task.done():
                task.cancel()
        if self._bus is not None:
            self._bus.deregister(BaseSomaPredictor)
            self._bus.deregister(BaseAllostaticRegulator)
        logger.info("soma_shutdown", cycle_count=self._cycle_count)

    def _prewarm_fisher_manifold(self) -> None:
        """
        Task 3: Pre-initialize Fisher manifold with known system dimensions at startup.

        Instead of cold-starting at dim=0 on the first cycle, we seed the manifold
        with zero vectors at the expected dimensionality from the registered system
        order. This avoids the dim=0→42 expansion spike in the first cycle.
        """
        try:
            system_order = self._state_vector_ctor.system_order
            if not system_order:
                return  # No systems registered yet

            expected_dim = len(system_order)

            # Initialize with zero-state vectors at the expected dimension
            # This "warms up" the Fisher metric estimator without actual state data
            zero_vec = np.zeros(expected_dim, dtype=np.float64)

            # Push a few dummy observations to establish the window and baseline
            for _ in range(3):
                self._fisher_manifold.update(zero_vec)

            # Set the system order baseline so first real update doesn't trigger expansion
            self._last_fisher_system_order = list(system_order)

            logger.info(
                "soma_fisher_prewarmed",
                system_count=len(system_order),
                initial_dim=expected_dim,
            )
        except Exception as e:
            logger.debug(
                "soma_fisher_prewarm_failed",
                error=str(e),
            )

    async def health(self) -> dict[str, Any]:
        """Health check for Synapse monitoring."""
        mean_duration = (
            sum(self._cycle_durations) / len(self._cycle_durations)
            if self._cycle_durations
            else 0.0
        )
        ext_pressure = self._exteroceptive_pressure
        result: dict[str, Any] = {
            "status": "healthy" if self._enabled else "degraded",
            "cycle_count": self._cycle_count,
            "last_cycle_ms": round(self._last_cycle_duration_ms, 3),
            "mean_cycle_ms": round(mean_duration, 3),
            "stage": self._developmental.stage.value,
            "attractors": self._phase_space.attractor_count,
            "urgency": round(self._current_signal.urgency, 3),
            "dominant_error": self._current_signal.dominant_error.value,
            "exteroceptive_ambient_stress": round(
                ext_pressure.ambient_stress if ext_pressure else 0.0, 3
            ),
            "exteroceptive_modalities": (
                [m.value for m in ext_pressure.active_modalities]
                if ext_pressure
                else []
            ),
        }

        # Phase A manifold health
        if self._manifold_enabled:
            result["manifold"] = {
                "signal_buffer_size": self._signal_buffer.size,
                "signals_ingested": self._signal_buffer.total_ingested,
                "derivative_history": self._derivative_engine.history_length,
                "known_systems": len(self._state_vector_ctor.system_order),
                "has_percept": self._current_percept is not None,
                "percept_urgency": (
                    round(self._current_percept.urgency, 3)
                    if self._current_percept
                    else 0.0
                ),
            }

            # Phase B–E analysis engine health
            result["analysis_engines"] = {
                "fisher_has_metric": self._fisher_manifold.has_fisher,
                "fisher_baseline_locked": self._fisher_manifold.baseline_locked,
                "fisher_window_size": self._fisher_manifold.window_size,
                "topology_window_size": self._topological_analyzer.window_size,
                "topology_baseline_locked": self._topological_analyzer.baseline_locked,
                "emergence_coherence": round(self._emergence_engine.coherence_signal, 3),
                "emergence_quantizer_fitted": self._emergence_engine.quantizer_fitted,
                "causal_flow_systems": self._causal_flow_engine.system_count,
                "geodesic_deviation": (
                    round(self._last_geodesic_deviation.scalar, 3)
                    if self._last_geodesic_deviation else None
                ),
            }

        # Closed-loop regulation health
        result["regulation"] = {
            "loop_executor_active": len(self._last_loop_dispatches),
            "adaptive_setpoints_profiles": len(self._adaptive_setpoints.get_all_profiles()),
            "cascade_predictor_accuracy": round(
                self._cascade_predictor.prediction_accuracy, 3,
            ),
            "cascade_risk": (
                round(self._last_cascade_snapshot.total_cascade_risk, 3)
                if self._last_cascade_snapshot else 0.0
            ),
            "autonomic_actions_recent": len(self._last_autonomic_actions),
            "autonomic_cooldowns_active": sum(
                1 for v in self._autonomic_protocol._cooldowns.values() if v > 0
            ),
        }

        return result

    # ─── System Wiring ──────────────────────────────────────────

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire in the Synapse EventBus for manifold signal ingestion."""
        self._event_bus_ref = event_bus

        # Subscribe to all events for the manifold signal buffer
        if self._manifold_enabled:
            event_bus.subscribe_all(self._on_synapse_event)

    def set_atune(self, atune: Any) -> None:
        self._interoceptor.set_atune(atune)
        self._loop_executor.set_atune(atune)
        self._autonomic_protocol.set_atune(atune)

    def set_synapse(self, synapse: Any) -> None:
        self._interoceptor.set_synapse(synapse)
        # Forward to controller if it supports metabolic sensing (MetabolicAllostaticRegulator)
        if hasattr(self._controller, "set_synapse"):
            self._controller.set_synapse(synapse)  # type: ignore[union-attr]
        self._synapse_ref = synapse
        self._loop_executor.set_synapse(synapse)
        self._autonomic_protocol.set_synapse(synapse)

    def set_nova(self, nova: Any) -> None:
        self._interoceptor.set_nova(nova)
        self._loop_executor.set_nova(nova)
        self._autonomic_protocol.set_nova(nova)

    def set_thymos(self, thymos: Any) -> None:
        self._interoceptor.set_thymos(thymos)
        self._loop_executor.set_thymos(thymos)
        self._autonomic_protocol.set_thymos(thymos)

    def set_equor(self, equor: Any) -> None:
        self._interoceptor.set_equor(equor)

    def set_token_budget(self, budget: Any) -> None:
        self._interoceptor.set_token_budget(budget)

    def set_telos(self, telos: Any) -> None:
        """
        Wire Telos for Loop 6 bidirectional integration.

        Soma reads Telos's last EffectiveIntelligenceReport to supplement
        the CONFIDENCE dimension (effective_I → generative model fitness)
        and COHERENCE dimension (alignment_gap → value-coherence signal).
        """
        self._interoceptor.set_telos(telos)

    def set_oikos(self, oikos: Any) -> None:
        """Wire Oikos service for financial snapshot reads (temporal depth expansion)."""
        self._oikos_ref = oikos

    def set_evo(self, evo: Any) -> None:
        """Wire Evo for loop executor hypothesis dispatch and autonomic learning."""
        self._loop_executor.set_evo(evo)
        self._autonomic_protocol.set_evo(evo)

    def set_oneiros(self, oneiros: Any) -> None:
        """Wire Oneiros for sleep pressure loop and autonomic sleep requests."""
        self._loop_executor.set_oneiros(oneiros)
        self._autonomic_protocol.set_oneiros(oneiros)

    def set_thread(self, thread: Any) -> None:
        """Wire Thread for narrative coherence loop."""
        self._loop_executor.set_thread(thread)

    def set_alive(self, alive: Any) -> None:
        """Wire Alive for visualization loop."""
        self._loop_executor.set_alive(alive)

    def set_voxis(self, voxis: Any) -> None:
        """Wire Voxis for expression feedback loop."""
        self._loop_executor.set_voxis(voxis)

    def set_memory(self, memory: Any) -> None:
        """Wire Memory for salience decay loop."""
        self._loop_executor.set_memory(memory)

    # ─── NeuroplasticityBus Callbacks ─────────────────────────────

    def _on_predictor_evolved(self, predictor: BaseSomaPredictor) -> None:
        """
        Hot-swap the interoceptive predictor in the live service.

        Called by NeuroplasticityBus when a new BaseSomaPredictor subclass
        is discovered. The swap is atomic — any in-flight cycle that already
        captured a reference to the old predictor completes normally.
        """
        self._predictor = predictor
        self._counterfactual.set_dynamics(predictor.dynamics_matrix)
        logger.info(
            "soma_predictor_hot_reloaded",
            predictor=type(predictor).__name__,
        )

    def _on_regulator_evolved(self, regulator: BaseAllostaticRegulator) -> None:
        """
        Hot-swap the allostatic controller in the live service.

        Called by NeuroplasticityBus when a new BaseAllostaticRegulator subclass
        is discovered. Preserves current context by re-applying it, and forwards
        the cached Synapse reference so MetabolicAllostaticRegulator can read
        MetabolicSnapshot immediately after the swap.
        """
        self._controller = regulator
        # Re-wire Synapse if the new regulator supports metabolic sensing
        synapse = getattr(self, "_synapse_ref", None)
        if synapse is not None and hasattr(regulator, "set_synapse"):
            regulator.set_synapse(synapse)  # type: ignore[union-attr]
        logger.info(
            "soma_regulator_hot_reloaded",
            regulator=type(regulator).__name__,
        )

    # ─── Manifold Signal Ingestion ─────────────────────────────────

    async def _on_synapse_event(self, event: SynapseEvent) -> None:
        """EventBus callback — ingest every Synapse event into the signal buffer."""
        self._signal_buffer.ingest_synapse_event(event)
        # Feed micro-level transition tracking for causal emergence
        self._emergence_engine.observe_micro(event.event_type.value)

    # ─── Core Cycle ──────────────────────────────────────────────

    async def run_cycle(self) -> AllostaticSignal:
        """
        Main theta cycle entry. Called by Synapse BEFORE Atune.

        Pipeline:
          1. Sense — read 9D state from interoceptors (<=2ms)
          2. Buffer — push into trajectory ring buffer
          3. Predict — multi-horizon forecasts (<=1ms)
          4. Compute errors — predicted - setpoint per horizon per dim
          5. Compute error rates — d(error)/dt
          6. Compute temporal dissonance — moment vs session divergence
          7. Compute urgency — max(|errors|) * max(|error_rates|)
          8. Update phase space — every N cycles only (<=2ms)
          9. Build and emit AllostaticSignal
          10. Evaluate developmental transition

        Total budget: <=5ms.
        """
        if not self._enabled:
            return self._current_signal

        start = time.perf_counter()
        self._cycle_count += 1

        # Per-operation timing accumulators (Task 1)
        _t_allostatic_ms: float = 0.0
        _t_fisher_ms: float = 0.0
        _t_vulnerability_ms: float = 0.0
        _t_signal_ms: float = 0.0

        try:
            # ── Allostatic computation ────────────────────────────────
            _t0 = time.perf_counter()

            # 1. Sense
            sensed = self._interoceptor.sense()

            # 1b. Apply exteroceptive pressure deltas (Cross-Modal Synesthesia)
            # External world events nudge the sensed state before prediction,
            # so the organism's forecasts incorporate the "felt weather".
            sensed = self._apply_exteroceptive_pressure(sensed)

            # 1c. Apply financial affect bias (Temporal Depth Expansion)
            # Projects treasury trajectory → TTD → affect deltas.
            # Refreshes the financial snapshot from Oikos every N cycles.
            self._refresh_financial_snapshot_if_due()
            self._temporal_depth.tick_financial()
            sensed = self._temporal_depth.apply_affect_bias_to_sensed(sensed)

            # 2. Buffer
            self._predictor.push_state(sensed)

            # 3. Tick setpoints (EMA toward targets)
            if self._developmental.setpoint_adaptation_enabled():
                self._controller.tick_setpoints()

            # 4. Predict
            available_horizons = self._temporal_depth.available_horizons
            predictions = self._predictor.predict_all_horizons(sensed, available_horizons)

            # 5. Compute allostatic errors
            setpoints = self._controller.setpoints
            errors = self._predictor.compute_allostatic_errors(predictions, setpoints)

            # 6. Compute error rates
            moment_errors = errors.get("moment", {d: 0.0 for d in ALL_DIMENSIONS})
            error_rates = self._predictor.compute_error_rates(moment_errors)

            # 7. Compute temporal dissonance
            dissonance = self._temporal_depth.compute_dissonance(predictions)

            # 8. Compute urgency
            urgency = self._controller.compute_urgency(errors, error_rates)

            # 9. Find dominant error
            dominant_dim, dominant_mag = self._controller.find_dominant_error(errors)

            # 10. Compute precision weights
            # Build intermediate state for precision computation
            state = InteroceptiveState(
                sensed=sensed,
                setpoints=setpoints,
                predicted=predictions,
                errors=errors,
                error_rates=error_rates,
                precision={d: 1.0 for d in ALL_DIMENSIONS},
                max_error_magnitude=dominant_mag,
                dominant_error=dominant_dim,
                temporal_dissonance=dissonance,
                urgency=urgency,
            )

            # 11. Update phase space (every N cycles)
            self._phase_space_update_counter += 1
            if (
                self._developmental.phase_space_enabled()
                and self._phase_space_update_counter >= self._config.phase_space_update_interval
            ):
                velocity = self._predictor.compute_velocity()
                self._phase_space.update(self._predictor.raw_trajectory, velocity)
                self._phase_space_update_counter = 0

            _t_allostatic_ms = (time.perf_counter() - _t0) * 1000

            # ── Signal emission ───────────────────────────────────────
            _t0 = time.perf_counter()

            # 12. Build signal
            phase_dict = self._phase_space.snapshot_dict()
            signal = self._controller.build_signal(state, phase_dict, self._cycle_count)

            # 12b. Populate financial temporal depth fields on the signal
            signal.financial_ttd_days = self._temporal_depth.current_ttd_days
            signal.financial_affect_bias = self._temporal_depth.current_affect_bias

            # Update current state reference on the signal's state
            signal.state.precision = signal.precision_weights
            self._current_state = signal.state
            self._current_signal = signal

            # 12c. Detect emergent emotions (lightweight pattern match, every cycle)
            self._current_emotions = self._emotion_detector.detect(signal.state)

            # 13. Evaluate developmental transitions (every 1000 cycles)
            if self._cycle_count % 1000 == 0:
                mean_conf = sensed.get(InteroceptiveDimension.CONFIDENCE, 0.5)
                promoted = self._developmental.evaluate_transition(
                    cycle_count=self._cycle_count,
                    mean_confidence=mean_conf,
                    attractor_count=self._phase_space.attractor_count,
                    bifurcation_count=self._phase_space.bifurcation_count,
                )
                if promoted:
                    self._temporal_depth.set_stage(self._developmental.stage)

            # ── Closed-loop regulation (new modules) ──────────────────
            # 14. Adaptive setpoint learning — observe lived state near attractors
            nearest_attractor = self._phase_space.get_nearest_attractor_label(sensed)
            learned = self._adaptive_setpoints.observe(
                signal.state, nearest_attractor, self._cycle_count,
            )
            if learned is not None and hasattr(self._controller, "apply_learned_setpoints"):
                self._controller.apply_learned_setpoints(learned)  # type: ignore[union-attr]

            # 15. Execute active feedback loops (parametric nudges to target systems)
            active_loops = get_active_loop_signals(moment_errors, error_rates)
            if active_loops:
                self._last_loop_dispatches = self._loop_executor.execute(
                    signal, active_loops,
                )

            # 16. Autonomic regulation (reflexive actions, no deliberation)
            self._last_autonomic_actions = self._autonomic_protocol.evaluate(
                signal, self._last_cascade_snapshot,
            )

            _t_signal_ms = (time.perf_counter() - _t0) * 1000

            # ── Vulnerability / Fisher manifold (staggered) ───────────
            # Phase A — Manifold fast path (every cycle — cheap signal aggregation)
            if self._manifold_enabled:
                await self._run_manifold_fast_path()

            # Phase B — Fisher manifold: every 10 cycles (Task 3)
            # The manifold update + Ledoit-Wolf estimation is the primary slow path.
            _t0 = time.perf_counter()
            self._fisher_cycle_counter += 1
            if self._manifold_enabled and self._current_state_vector is not None:
                if self._fisher_cycle_counter >= self._fisher_cycle_interval:
                    self._fisher_cycle_counter = 0
                    self._run_fisher_fast_path(self._current_state_vector)
                    logger.debug(
                        "soma_fisher_recomputed",
                        cycle=self._cycle_count,
                    )
                else:
                    cycles_left = self._fisher_cycle_interval - self._fisher_cycle_counter
                    logger.debug(
                        "soma_fisher_cached",
                        cycle=self._cycle_count,
                        cycles_until_refresh=cycles_left,
                    )
            _t_fisher_ms = (time.perf_counter() - _t0) * 1000

            # Vulnerability assessment placeholder counter (Task 3)
            # (Actual vulnerability_map call site — kept for future use.)
            _t0 = time.perf_counter()
            self._vulnerability_cycle_counter += 1
            if self._vulnerability_cycle_counter >= self._vulnerability_cycle_interval:
                self._vulnerability_cycle_counter = 0
                # Vulnerability map runs inside medium/deep paths already;
                # counter is maintained here so the stagger contract is explicit.
                logger.debug("soma_vulnerability_interval_tick", cycle=self._cycle_count)
            _t_vulnerability_ms = (time.perf_counter() - _t0) * 1000

            # Body scan counter (Task 3) — full somatic scan every 50 cycles
            self._body_scan_cycle_counter += 1
            if self._body_scan_cycle_counter >= self._body_scan_cycle_interval:
                self._body_scan_cycle_counter = 0
                logger.debug("soma_body_scan_interval_tick", cycle=self._cycle_count)

            # Schedule medium path (every N cycles, async)
            self._medium_path_counter += 1
            if self._manifold_enabled and self._medium_path_counter >= self._medium_path_interval:
                self._medium_path_counter = 0
                if self._medium_path_task is None or self._medium_path_task.done():
                    self._medium_path_task = asyncio.create_task(
                        self._run_medium_path(), name="soma_medium_path",
                    )

            # Schedule deep path (every N cycles, background)
            self._deep_path_counter += 1
            if self._manifold_enabled and self._deep_path_counter >= self._deep_path_interval:
                self._deep_path_counter = 0
                if self._deep_path_task is None or self._deep_path_task.done():
                    self._deep_path_task = asyncio.create_task(
                        self._run_deep_path(), name="soma_deep_path",
                    )

        except Exception as exc:
            logger.error("soma_cycle_error", error=str(exc), cycle=self._cycle_count)
            # Emit default signal on failure — graceful degradation
            self._current_signal = AllostaticSignal.default()
            self._current_signal.cycle_number = self._cycle_count

        # Track timing
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._last_cycle_duration_ms = elapsed_ms
        self._cycle_durations.append(elapsed_ms)

        # Log per-operation breakdown every 50 cycles at DEBUG level (Task 1)
        if self._cycle_count % self._breakdown_log_interval == 0:
            logger.debug(
                "soma_cycle_breakdown",
                allostatic_ms=round(_t_allostatic_ms, 3),
                fisher_ms=round(_t_fisher_ms, 3),
                vulnerability_ms=round(_t_vulnerability_ms, 3),
                signal_ms=round(_t_signal_ms, 3),
                total_ms=round(elapsed_ms, 3),
                cycle=self._cycle_count,
            )

        # Task 1+2: Suppress soma_cycle_slow during warmup or heavy external operations
        # (1) Warmup suppression: first 50 cycles are cold-start initialization period
        in_warmup = not self._warmup_complete and self._cycle_count < self._warmup_cycle_count

        # (2) Synapse activity suppression: check if external systems are running expensive ops
        # (embedding, LLM call, etc.) that legitimately cause clock overrun
        suppressed_by_synapse = False
        if self._synapse_ref is not None:
            try:
                get_active = getattr(self._synapse_ref, "get_active_systems", lambda: [])
                active_systems: list[str] = get_active()
                # If Batches (embedding) or Atune (LLM) are active, suppress the warning
                heavy_ops = {"Batches", "Atune"}
                if any(sys in heavy_ops for sys in active_systems):
                    suppressed_by_synapse = True
            except Exception:
                pass  # Silently ignore if Synapse query fails

        # Emit warning only if genuinely slow AND not suppressed
        if elapsed_ms > 20.0 and not in_warmup and not suppressed_by_synapse:
            logger.warning("soma_cycle_slow", elapsed_ms=round(elapsed_ms, 2))

        # Mark warmup complete on first cycle past threshold
        if in_warmup and self._cycle_count >= self._warmup_cycle_count:
            self._warmup_complete = True
            logger.info("soma_warmup_complete", cycle=self._cycle_count)

        return self._current_signal

    # ─── Query Methods ───────────────────────────────────────────

    def get_current_state(self) -> InteroceptiveState | None:
        """Returns the last computed interoceptive state."""
        return self._current_state

    def get_current_signal(self) -> AllostaticSignal:
        """Returns the last emitted allostatic signal."""
        return self._current_signal

    def get_somatic_marker(self) -> SomaticMarker:
        """Returns a somatic marker from the current state for memory stamping."""
        return self.create_somatic_marker()

    def get_errors(self) -> dict[str, dict[InteroceptiveDimension, float]]:
        """Returns allostatic errors per horizon per dimension."""
        if self._current_state is not None:
            return self._current_state.errors
        return {}

    def get_predictions(self) -> dict[str, dict[InteroceptiveDimension, float]]:
        """Returns all horizon predictions from the last computed state."""
        if self._current_state is not None:
            return self._current_state.predicted
        return {}

    def get_phase_position(self) -> dict[str, Any]:
        """Returns nearest attractor, distance to bifurcation, trajectory heading."""
        return self._phase_space.snapshot_dict()

    def get_developmental_stage(self) -> DevelopmentalStage:
        """Returns the current maturation stage."""
        return self._developmental.stage

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def urgency(self) -> float:
        return self._current_signal.urgency

    @property
    def urgency_threshold(self) -> float:
        return self._controller.urgency_threshold

    @property
    def dominant_error(self) -> InteroceptiveDimension:
        return self._current_signal.dominant_error

    @property
    def precision_weights(self) -> dict[InteroceptiveDimension, float]:
        return self._current_signal.precision_weights

    # ─── Emotion Detection ───────────────────────────────────────

    def get_active_emotions(self) -> list[ActiveEmotion]:
        """Return currently active emotion regions with intensities."""
        return list(self._current_emotions)

    def get_interoceptive_report_for_voxis(self) -> dict[str, Any]:
        """
        Build a report of the organism's felt state for Voxis consumption.
        Returns raw interoceptive state + detected emotion regions.
        """
        if self._current_state is None:
            return {}
        return self._voxis_interface.get_interoceptive_report(
            self._current_state,
            self._phase_space.snapshot_dict(),
        )

    # ─── Feedback Loop Queries ───────────────────────────────────

    def get_allostatic_loops(self) -> dict[str, dict[str, Any]]:
        """Return all 15 allostatic loop definitions."""
        return {name: loop.to_dict() for name, loop in ALLOSTATIC_LOOP_MAP.items()}

    def get_active_loops(self, threshold: float = 0.1) -> dict[str, dict[str, float]]:
        """
        Return loops currently firing based on error magnitudes.
        Each entry includes error, rate, and coupled_strength.
        """
        if self._current_state is None:
            return {}
        moment_errors = self._current_state.errors.get("moment", {})
        return get_active_loop_signals(
            moment_errors,
            self._current_state.error_rates,
            threshold=threshold,
        )

    def get_loops_for_system(self, system_id: str) -> list[dict[str, Any]]:
        """Return loop definitions consumed by a specific system."""
        from systems.soma.feedback_loops import get_loops_for_consumer
        return [loop.to_dict() for loop in get_loops_for_consumer(system_id)]

    # ─── Memory Integration ──────────────────────────────────────

    def create_somatic_marker(self) -> SomaticMarker:
        """
        Snapshot current state as a somatic marker. Budget: 1ms.
        Called by Memory when storing traces.
        """
        if self._current_state is None:
            return SomaticMarker()

        attractor_label = self._phase_space.get_nearest_attractor_label(
            self._current_state.sensed,
        )
        return self._somatic_memory.create_marker(
            self._current_state,
            attractor_label=attractor_label,
        )

    def somatic_rerank(
        self,
        candidates: list[Any],
        current_state: InteroceptiveState | None = None,
    ) -> list[Any]:
        """
        Boost memories with similar somatic markers.
        Up to +30% salience boost based on somatic similarity.

        When temporal depth financial projection is active, additionally
        modulates reranking to boost exploration memories (high TTD) or
        revenue memories (low TTD).
        """
        state = current_state or self._current_state
        if state is None:
            return candidates
        financial_modifiers = self._temporal_depth.get_rerank_modifiers()
        return self._somatic_memory.somatic_rerank(
            candidates, state, financial_modifiers=financial_modifiers,
        )

    # ─── Evo Integration ─────────────────────────────────────────

    def update_dynamics_matrix(self, new_dynamics: list[list[float]]) -> None:
        """Evo updates the 9x9 cross-dimension coupling matrix."""
        self._predictor.update_dynamics(new_dynamics)
        self._counterfactual.set_dynamics(self._predictor.dynamics_matrix)

    def update_emotion_regions(self, updated_regions: dict[str, dict[str, Any]]) -> None:
        """Evo refines emotion region boundaries."""
        self._emotion_regions.update(updated_regions)
        self._emotion_detector.update_regions(updated_regions)

    # ─── Counterfactual (Oneiros Integration) ────────────────────

    def generate_counterfactual(
        self,
        decision_id: str,
        actual_trajectory: list[dict[InteroceptiveDimension, float]],
        alternative_description: str,
        alternative_initial_impact: dict[InteroceptiveDimension, float],
        num_steps: int = 10,
    ) -> CounterfactualTrace:
        """
        Generate a counterfactual interoceptive trajectory.
        Only available at REFLECTIVE stage and above.
        """
        if not self._developmental.counterfactual_enabled():
            return CounterfactualTrace(
                decision_id=decision_id,
                lesson="Counterfactual not available at current developmental stage",
            )

        return self._counterfactual.generate_counterfactual(
            decision_id=decision_id,
            actual_trajectory=actual_trajectory,
            alternative_description=alternative_description,
            alternative_initial_impact=alternative_initial_impact,
            setpoints=self._controller.setpoints,
            num_steps=num_steps,
        )

    # ─── Context Management ──────────────────────────────────────

    def set_context(self, context: str) -> None:
        """
        Switch allostatic context. Called when the organism's activity changes.

        Contexts: "conversation", "deep_processing", "recovery", "exploration"
        """
        if self._developmental.setpoint_adaptation_enabled():
            self._controller.set_context(context)

    # ─── Cross-Modal Synesthesia ──────────────────────────────────

    def inject_external_stress(self, stress: float) -> None:
        """
        Accept a normalised [0, 1] economic stress value from the external
        volatility sensor and blend it into the interoceptive state.

        Called by ExteroceptionService from its background poll task.
        Intentionally synchronous and allocation-free — not called from the theta cycle.

        Effect:
          TEMPORAL_PRESSURE is modulated upward by up to 0.3 units at
          stress=1.0, keeping it within the declared valid range [0, 1].
          The Interoceptor's next sense() pass will read the updated value
          from ``_external_stress`` via ``_sense_temporal_pressure()``.
        """
        self._external_stress = max(0.0, min(1.0, stress))

    def inject_exteroceptive_pressure(self, pressure: ExteroceptivePressure) -> None:
        """Accept per-dimension exteroceptive pressure from ExteroceptionService.

        The pressure is stored and applied as additive deltas to the sensed
        state during the next theta cycle (step 1b in run_cycle). This is
        the new multi-dimensional synesthesia channel — external market
        volatility, sentiment, and fear/greed each push specific dimensions
        rather than collapsing into a single scalar stress value.

        Synchronous, allocation-free, safe to call from background tasks.
        """
        self._exteroceptive_pressure = pressure

    @property
    def external_stress(self) -> float:
        """Latest exogenous stress value injected by the volatility sensor."""
        return self._external_stress

    @property
    def exteroceptive_pressure(self) -> ExteroceptivePressure | None:
        """Latest per-dimension exteroceptive pressure, or None before first injection."""
        return self._exteroceptive_pressure

    def _refresh_financial_snapshot_if_due(self) -> None:
        """
        Periodically read Oikos EconomicState and inject a fresh snapshot
        into the TemporalDepthManager. This runs outside the critical
        prediction path — snapshot construction is cheap (pure getattr).
        """
        if not self._temporal_depth.financial_enabled or self._oikos_ref is None:
            return

        self._financial_refresh_counter += 1
        if self._financial_refresh_counter < self._config.temporal_depth_refresh_interval_cycles:
            return

        self._financial_refresh_counter = 0

        # Read economic state snapshot (in-memory, no I/O)
        economic_state = getattr(self._oikos_ref, "snapshot", lambda: None)()
        if economic_state is None:
            return

        snapshot = build_financial_snapshot_from_oikos(economic_state)
        snapshot.timestamp_cycle = self._cycle_count
        self._temporal_depth.inject_financial_snapshot(snapshot)

    def _apply_exteroceptive_pressure(
        self,
        sensed: dict[InteroceptiveDimension, float],
    ) -> dict[InteroceptiveDimension, float]:
        """Apply exteroceptive pressure deltas to the sensed state.

        Each dimension is nudged by the corresponding pressure delta,
        then re-clamped to its valid range. This ensures external data
        never pushes a dimension outside its declared bounds.

        The pressure values are already clamped by the mapping engine
        (max_total_pressure = 0.25 per dimension), so external sources
        can never override the organism's internal baseline by more
        than 25%.
        """
        if self._exteroceptive_pressure is None:
            return sensed

        from systems.soma.types import DIMENSION_RANGES

        result = dict(sensed)
        for dim, delta in self._exteroceptive_pressure.pressures.items():
            if abs(delta) < 0.0001:
                continue
            current = result.get(dim, 0.0)
            lo, hi = DIMENSION_RANGES.get(dim, (0.0, 1.0))
            result[dim] = max(lo, min(hi, current + delta))

        return result

    # ─── Phase A: Manifold Fast Path ─────────────────────────────

    async def _run_manifold_fast_path(self) -> None:
        """Per-cycle manifold analysis: signals → state vector → derivatives → percept.

        Runs inside the try block of run_cycle so failures degrade gracefully
        without breaking the existing allostatic pipeline.
        """
        now = time.monotonic()

        # Drain signals since last window
        signals = self._signal_buffer.get_window(self._last_window_time)
        self._last_window_time = now

        if not signals:
            return

        # Construct organism state vector from this cycle's signals
        window_duration = 0.15  # ~theta cycle duration in seconds
        state_vector = self._state_vector_ctor.construct(
            signals, window_duration, self._cycle_count,
        )
        self._current_state_vector = state_vector

        # Push into derivative engine history
        self._derivative_engine.push(state_vector)

        # Compute multi-scale derivatives (only when we have enough history)
        if self._derivative_engine.history_length >= 7:  # minimum for fast scale
            derivatives = self._derivative_engine.compute()
            self._current_derivatives = derivatives

            # Compose interoceptive percept if thresholds exceeded
            percept = self._broadcaster.compose_percept(
                derivatives, state_vector,
            )
            self._current_percept = percept

            # Broadcast percept through Synapse EventBus
            if percept is not None and self._event_bus_ref is not None:
                from systems.synapse.types import (
                    SynapseEvent as SynEvent,
                )
                from systems.synapse.types import (
                    SynapseEventType,
                )

                event = SynEvent(
                    event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                    source_system="soma",
                    data={
                        "percept_id": percept.percept_id,
                        "urgency": percept.urgency,
                        "sensation_type": percept.sensation_type.value,
                        "description": percept.description,
                        "epicenter_system": percept.epicenter_system,
                        "affected_systems": percept.affected_systems,
                        "recommended_action": percept.recommended_action.value,
                    },
                )
                await self._event_bus_ref.emit(event)

    # ─── Phase B: Fisher Fast Path (inline, staggered every 10 cycles) ──

    def _run_fisher_fast_path(self, state_vector: OrganismStateVector) -> None:
        """Update Fisher manifold and check geodesic deviation.

        Called every `_fisher_cycle_interval` cycles (default: 10) — not every
        cycle. The Fisher update itself (Ledoit-Wolf on 50+ samples) was the
        primary source of soma_cycle_slow events.

        Manifold expansion policy (Task 2):
        - Dimension grows (new system registered): pad existing window vectors
          with zeros for the new dimensions and continue — no full reset.
        - Dimension shrinks (system removed): full reset required because the
          existing Fisher metric is no longer valid for the smaller space.

        Input-delta caching (Task 2):
        - If the L2 delta from the last computed flat vector is < 0.01 and the
          last update was < 5s ago, skip the manifold.update() call entirely.
          The geodesic deviation from the previous cycle remains valid.
        """
        try:
            system_order = self._state_vector_ctor.system_order

            flat = np.array(
                state_vector.to_flat_array(system_order), dtype=np.float64,
            )
            if flat.size == 0:
                return

            now_ts = time.monotonic()

            # ── Incremental manifold expansion instead of full reset ──
            if self._last_fisher_system_order != system_order:
                prev = self._last_fisher_system_order
                old_dim = len(prev) if prev else 0
                new_dim = flat.size

                if new_dim > old_dim:
                    # Dimension grew — pad existing window vectors and keep history.
                    added = [s for s in system_order if prev and s not in prev]
                    logger.info(
                        "fisher_manifold_expand",
                        old_dim=old_dim,
                        new_dim=new_dim,
                        added_systems=added,
                    )
                    pad_width = new_dim - old_dim
                    existing_window = list(self._fisher_manifold.window_vectors)
                    if existing_window:
                        padding = np.zeros(pad_width, dtype=np.float64)
                        padded = [np.concatenate([v, padding]) for v in existing_window]
                        # Rebuild manifold with padded history — preserves calibration.
                        new_manifold = FisherManifold(
                            window_size=self._fisher_config["window_size"],
                            baseline_capacity=self._fisher_config["baseline_capacity"],
                            calibration_threshold=self._fisher_config["calibration_threshold"],
                            min_samples_for_fisher=self._fisher_config["min_samples_for_fisher"],
                        )
                        for padded_vec in padded:
                            new_manifold.update(padded_vec)
                        self._fisher_manifold = new_manifold
                    else:
                        # No prior history — just reset (first few cycles)
                        self._fisher_manifold = FisherManifold(
                            window_size=self._fisher_config["window_size"],
                            baseline_capacity=self._fisher_config["baseline_capacity"],
                            calibration_threshold=self._fisher_config["calibration_threshold"],
                            min_samples_for_fisher=self._fisher_config["min_samples_for_fisher"],
                        )
                    # Invalidate cached flat so we don't skip the first update
                    self._fisher_last_flat = None
                else:
                    # Dimension shrank (system removed) — full reset required.
                    logger.info(
                        "fisher_manifold_reset",
                        old_dim=old_dim,
                        new_dim=new_dim,
                        reason="dimension_shrank",
                    )
                    self._fisher_manifold = FisherManifold(
                        window_size=self._fisher_config["window_size"],
                        baseline_capacity=self._fisher_config["baseline_capacity"],
                        calibration_threshold=self._fisher_config["calibration_threshold"],
                        min_samples_for_fisher=self._fisher_config["min_samples_for_fisher"],
                    )
                    self._fisher_last_flat = None

                self._last_fisher_system_order = list(system_order)
                self._fisher_last_updated_ts = 0.0  # force update on next call

            # ── Input-delta cache: skip manifold update if state hasn't changed ──
            age_s = now_ts - self._fisher_last_updated_ts
            skip_update = False
            if (
                self._fisher_last_flat is not None
                and flat.shape == self._fisher_last_flat.shape
                and age_s < self._fisher_cache_max_age_s
            ):
                delta = float(np.linalg.norm(flat - self._fisher_last_flat))
                if delta < self._fisher_cache_delta_threshold:
                    skip_update = True
                    logger.debug(
                        "soma_fisher_input_cached",
                        delta=round(delta, 5),
                        age_s=round(age_s, 3),
                        cycle=self._cycle_count,
                    )

            if not skip_update:
                self._fisher_manifold.update(flat)
                self._fisher_last_flat = flat.copy()
                self._fisher_last_updated_ts = now_ts

            # Feed into other engines that need per-cycle state
            self._topological_analyzer.push_state(flat)
            self._emergence_engine.observe_macro(flat)

            # Feed causal flow engine with per-system scalar summaries
            for sys_id, slc in state_vector.systems.items():
                vals = slc.to_list()
                self._causal_flow_engine.push_system_value(
                    sys_id, float(np.mean(vals)),
                )
            # Advance causal flow cycle counter once per theta cycle (not once per system)
            self._causal_flow_engine.tick()

            # Feed renormalization engine
            self._renormalization_engine.observe(
                timestamp=time.time(),
                state_vector=flat,
                event_type="theta_cycle",
            )

            # Feed phase space reconstructor with key metrics
            for sys_id, slc in state_vector.systems.items():
                vals = slc.to_list()
                for i, feat_val in enumerate(vals):
                    metric_key = f"{sys_id}.feat_{i}"
                    self._phase_space_reconstructor.push_metric(metric_key, feat_val)

            # Geodesic deviation check
            deviation = self._fisher_manifold.geodesic_deviation(flat)
            if deviation is not None:
                self._last_geodesic_deviation = deviation
                if deviation.scalar > self._fisher_deviation_threshold:
                    self._broadcast_fisher_deviation(deviation)

        except Exception as exc:
            logger.debug("fisher_fast_path_error", error=str(exc))

    def _broadcast_fisher_deviation(self, deviation: GeodesicDeviation) -> None:
        """Emit an interoceptive percept when Fisher deviation exceeds threshold."""
        if self._event_bus_ref is None:
            return

        from systems.soma.types import InteroceptiveAction, SensationType
        from systems.synapse.types import (
            SynapseEvent as SynEvent,
        )
        from systems.synapse.types import (
            SynapseEventType,
        )

        percept_data = {
            "percept_id": f"fisher_deviation_{self._cycle_count}",
            "urgency": min(1.0, deviation.scalar / 5.0),
            "sensation_type": SensationType.GEOMETRIC_DEVIATION.value,
            "description": (
                f"Fisher geodesic deviation {deviation.scalar:.2f} "
                f"(p{deviation.percentile:.0f}) — dominant dims "
                f"{deviation.dominant_systems[:3]}"
            ),
            "epicenter_system": "manifold",
            "affected_systems": [str(d) for d in deviation.dominant_systems[:5]],
            "recommended_action": (
                InteroceptiveAction.ATTEND_INWARD.value
                if deviation.percentile > 90
                else InteroceptiveAction.NONE.value
            ),
        }

        event = SynEvent(
            event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
            source_system="soma",
            data=percept_data,
        )
        asyncio.create_task(self._event_bus_ref.emit(event), name="soma_fisher_broadcast")

    # ─── Medium Path (every ~100 cycles, async) ──────────────────

    async def _run_medium_path(self) -> None:
        """Derivatives at all scales → causal flow → emergence → renormalization.

        Runs asynchronously every ~100 theta cycles (~15s). Each engine
        is isolated — one failure doesn't kill the others.
        """
        # Causal flow
        try:
            if self._causal_flow_engine.should_compute():
                flow_map = self._causal_flow_engine.compute_causal_flow()
                self._last_causal_flow_map = flow_map
                # Feed TE matrix to cascade predictor for structural anticipation
                self._cascade_predictor.update_causal_graph(
                    flow_map.te_matrix, flow_map.system_ids,
                )
        except Exception as exc:
            logger.debug("medium_path_causal_flow_error", error=str(exc))

        # Cascade prediction — propagate current stresses through causal graph
        try:
            if self._current_state is not None:
                self._cascade_predictor.update_system_stresses_from_state(
                    self._current_state,
                )
                self._last_cascade_snapshot = self._cascade_predictor.predict(
                    self._cycle_count,
                )
        except Exception as exc:
            logger.debug("medium_path_cascade_predictor_error", error=str(exc))

        # Causal emergence
        try:
            if self._emergence_engine.should_compute():
                report = self._emergence_engine.compute_emergence()
                self._last_emergence_report = report
        except Exception as exc:
            logger.debug("medium_path_emergence_error", error=str(exc))

        # Renormalization
        try:
            rg_report = self._renormalization_engine.compute_rg_flow()
            self._last_rg_flow_report = rg_report
        except Exception as exc:
            logger.debug("medium_path_renormalization_error", error=str(exc))

        # Broadcast medium-path results if warranted
        self._maybe_broadcast_medium_results()

    def _maybe_broadcast_medium_results(self) -> None:
        """Emit percept if medium-path analysis found something notable."""
        if self._event_bus_ref is None:
            return

        emergence = self._last_emergence_report
        rg = self._last_rg_flow_report
        causal = self._last_causal_flow_map

        # Emergence declining or critical
        if emergence is not None and emergence.emergence_trend in ("declining", "critical"):
            from systems.soma.types import InteroceptiveAction, SensationType
            from systems.synapse.types import (
                SynapseEvent as SynEvent,
            )
            from systems.synapse.types import (
                SynapseEventType,
            )

            event = SynEvent(
                event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                source_system="soma",
                data={
                    "percept_id": f"emergence_{emergence.emergence_trend}_{self._cycle_count}",
                    "urgency": 0.8 if emergence.emergence_trend == "critical" else 0.5,
                    "sensation_type": SensationType.COHERENCE_DECLINE.value,
                    "description": (
                        f"Causal emergence {emergence.emergence_trend}: "
                        f"CE={emergence.causal_emergence:.3f}, "
                        f"coherence={emergence.coherence_signal:.3f}"
                    ),
                    "epicenter_system": "emergence",
                    "affected_systems": [],
                    "recommended_action": InteroceptiveAction.ATTEND_INWARD.value,
                },
            )
            asyncio.create_task(self._event_bus_ref.emit(event), name="soma_emergence_broadcast")

        # RG anomaly detected
        if rg is not None and rg.anomaly_scale is not None:
            from systems.soma.types import InteroceptiveAction, SensationType
            from systems.synapse.types import (
                SynapseEvent as SynEvent,
            )
            from systems.synapse.types import (
                SynapseEventType,
            )

            event = SynEvent(
                event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                source_system="soma",
                data={
                    "percept_id": f"rg_anomaly_{self._cycle_count}",
                    "urgency": 0.6,
                    "sensation_type": SensationType.SCALE_ANOMALY.value,
                    "description": (
                        f"Scale self-similarity break at {rg.anomaly_scale}s "
                        f"({rg.anomaly_scale_interpretation})"
                    ),
                    "epicenter_system": "renormalization",
                    "affected_systems": [],
                    "recommended_action": InteroceptiveAction.ATTEND_INWARD.value,
                },
            )
            asyncio.create_task(self._event_bus_ref.emit(event), name="soma_rg_broadcast")

        # Causal topology anomalies: unexpected or reversed influence links
        if causal is not None and (causal.unexpected_influences or causal.reversed_influences):
            from systems.soma.types import InteroceptiveAction, SensationType
            from systems.synapse.types import (
                SynapseEvent as SynEvent,
            )
            from systems.synapse.types import (
                SynapseEventType,
            )

            anomaly_count = len(causal.unexpected_influences) + len(causal.reversed_influences)
            top_anomaly = (
                causal.unexpected_influences[0]
                if causal.unexpected_influences
                else causal.reversed_influences[0]
            )
            event = SynEvent(
                event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                source_system="soma",
                data={
                    "percept_id": f"causal_anomaly_{self._cycle_count}",
                    "urgency": min(0.9, 0.3 + anomaly_count * 0.1),
                    "sensation_type": SensationType.CAUSAL_DISRUPTION.value,
                    "description": (
                        f"Causal topology: {anomaly_count} anomaly(s) — "
                        f"{top_anomaly.interpretation}"
                    ),
                    "epicenter_system": top_anomaly.source_system,
                    "affected_systems": [top_anomaly.target_system],
                    "recommended_action": InteroceptiveAction.ATTEND_INWARD.value,
                },
            )
            asyncio.create_task(self._event_bus_ref.emit(event), name="soma_causal_broadcast")

    # ─── Deep Path (every ~500 cycles, background) ───────────────

    async def _run_deep_path(self) -> None:
        """Persistent homology → attractors → Lyapunov → Ricci curvature.

        Runs in background every ~500 theta cycles (~75s). Compute-heavy
        but not on the critical path.
        """
        # Persistent homology
        try:
            if self._topological_analyzer.should_compute():
                diagnosis = self._topological_analyzer.compute_persistence()
                self._last_persistence_diagnosis = diagnosis
        except Exception as exc:
            logger.debug("deep_path_topology_error", error=str(exc))

        # Phase space reconstruction (attractors + Lyapunov)
        try:
            report = self._phase_space_reconstructor.reconstruct_all()
            self._last_phase_space_report = report
        except Exception as exc:
            logger.debug("deep_path_psr_error", error=str(exc))

        # Ricci curvature
        try:
            if self._current_state_vector is not None and self._fisher_manifold.has_fisher:
                system_order = self._state_vector_ctor.system_order
                flat = np.array(
                    self._current_state_vector.to_flat_array(system_order),
                    dtype=np.float64,
                )
                if flat.size > 0:
                    curvature = self._curvature_analyzer.analyze(flat, self._fisher_manifold)
                    self._last_curvature_map = curvature
        except Exception as exc:
            logger.debug("deep_path_curvature_error", error=str(exc))

    # ─── Sleep Path (called by Oneiros during sleep) ─────────────

    async def run_sleep_analysis(self) -> dict[str, Any]:
        """Extended analysis during Oneiros sleep cycle.

        Performs:
        1. Extended-scale renormalization with slow time windows
        2. Fisher manifold baseline recalibration
        3. Topological barcode baseline update
        4. Coarse-graining operator refit
        5. Macro-state quantizer refit
        6. Long-term drift analysis

        Returns a summary dict for Thread narrative and dream content.
        """
        results: dict[str, Any] = {}

        # 1. Extended renormalization calibration
        try:
            self._renormalization_engine.calibrate()
            results["renormalization_calibrated"] = True
        except Exception as exc:
            logger.debug("sleep_rg_calibrate_error", error=str(exc))
            results["renormalization_calibrated"] = False

        # 2. Fisher baseline recalibration (slow drift adaptation)
        try:
            if self._current_state_vector is not None:
                system_order = self._state_vector_ctor.system_order
                flat = np.array(
                    self._current_state_vector.to_flat_array(system_order),
                    dtype=np.float64,
                )
                if flat.size > 0:
                    self._fisher_manifold.update_baseline(flat)
            results["fisher_baseline_updated"] = True
        except Exception as exc:
            logger.debug("sleep_fisher_baseline_error", error=str(exc))
            results["fisher_baseline_updated"] = False

        # 3. Topological barcode baseline update
        try:
            self._topological_analyzer.unlock_baseline()
            self._topological_analyzer.update_baseline()
            self._topological_analyzer.lock_baseline()
            results["topology_baseline_updated"] = True
        except Exception as exc:
            logger.debug("sleep_topology_baseline_error", error=str(exc))
            results["topology_baseline_updated"] = False

        # 4. Emergence quantizer refit
        try:
            self._emergence_engine.refit_quantizer()
            results["emergence_quantizer_refit"] = True
        except Exception as exc:
            logger.debug("sleep_emergence_refit_error", error=str(exc))
            results["emergence_quantizer_refit"] = False

        # 5. Long-term drift analysis
        try:
            drift_summary: dict[str, Any] = {}
            if self._last_rg_flow_report is not None:
                drift_summary["fixed_point_drift"] = round(
                    self._last_rg_flow_report.fixed_point_drift, 4,
                )
                drift_summary["n_fixed_points"] = len(
                    self._last_rg_flow_report.fixed_points,
                )
            if self._last_persistence_diagnosis is not None:
                drift_summary["topological_health"] = round(
                    self._last_persistence_diagnosis.topological_health, 4,
                )
            if self._last_emergence_report is not None:
                drift_summary["coherence_signal"] = round(
                    self._last_emergence_report.coherence_signal, 4,
                )
                drift_summary["emergence_trend"] = self._last_emergence_report.emergence_trend
            results["drift_analysis"] = drift_summary

            # Derive summary string and rg_anomaly count expected by Oneiros
            rg_anomalies = 0
            if self._last_rg_flow_report is not None and self._last_rg_flow_report.anomaly_scale is not None:
                rg_anomalies = 1
            results["rg_anomalies"] = rg_anomalies

            # Build a human-readable drift_summary string for dream narrative
            parts: list[str] = []
            if "fixed_point_drift" in drift_summary:
                parts.append(f"drift={drift_summary['fixed_point_drift']:.3f}")
            if "emergence_trend" in drift_summary:
                parts.append(f"emergence={drift_summary['emergence_trend']}")
            if "topological_health" in drift_summary:
                parts.append(f"topology={drift_summary['topological_health']:.3f}")
            results["drift_summary"] = ", ".join(parts) if parts else ""
        except Exception as exc:
            logger.debug("sleep_drift_analysis_error", error=str(exc))
            results.setdefault("rg_anomalies", 0)
            results.setdefault("drift_summary", "")

        # 6. Full deep path run (takes advantage of sleep's compute budget)
        await self._run_deep_path()
        results["deep_path_completed"] = True

        logger.info("soma_sleep_analysis_complete", results=results)
        return results

    # ─── Coherence Signal (for Thymos constitutional drive) ──────

    @property
    def coherence_signal(self) -> float:
        """Normalized [0, 1] causal emergence coherence for Thymos.

        0.5 = neutral (no emergence data yet).
        Higher = organism is more coherent (macro > micro).
        Lower = fragmentation.
        """
        return self._emergence_engine.coherence_signal

    # ─── Analysis Query Methods ──────────────────────────────────

    def latest_analysis(self) -> dict[str, Any]:
        """Return the most recent results from all analysis paths."""
        result: dict[str, Any] = {}

        if self._last_geodesic_deviation is not None:
            result["geodesic_deviation"] = {
                "scalar": round(self._last_geodesic_deviation.scalar, 4),
                "percentile": round(self._last_geodesic_deviation.percentile, 1),
                "dominant_systems": self._last_geodesic_deviation.dominant_systems[:5],
            }

        if self._last_emergence_report is not None:
            result["emergence"] = self._last_emergence_report.to_dict()

        if self._last_causal_flow_map is not None:
            result["causal_flow"] = self._last_causal_flow_map.to_dict()

        if self._last_rg_flow_report is not None:
            rg = self._last_rg_flow_report
            result["renormalization"] = {
                "anomaly_scale": rg.anomaly_scale,
                "interpretation": rg.anomaly_scale_interpretation,
                "fixed_point_drift": round(rg.fixed_point_drift, 4),
                "n_fixed_points": len(rg.fixed_points),
            }

        if self._last_persistence_diagnosis is not None:
            result["topology"] = self._last_persistence_diagnosis.to_dict()

        if self._last_curvature_map is not None:
            cm = self._last_curvature_map
            result["curvature"] = {
                "overall": round(cm.overall_scalar_curvature, 4),
                "most_vulnerable_region": cm.most_vulnerable_region,
                "n_vulnerable_pairs": len(cm.vulnerable_pairs),
            }

        if self._last_phase_space_report is not None:
            psr = self._last_phase_space_report
            chaotic_metrics = [
                m for m, d in psr.diagnoses.items()
                if d.lyapunov_interpretation == "chaotic"
            ]
            result["phase_space"] = {
                "n_diagnosed": len(psr.diagnoses),
                "n_skipped": len(psr.skipped_metrics),
                "chaotic_metrics": chaotic_metrics,
            }

        return result

    def vulnerability_map(self) -> dict[str, Any]:
        """Return a map of the organism's current vulnerability landscape."""
        vulns: dict[str, Any] = {}

        # Curvature-based fragility
        if self._last_curvature_map is not None:
            cm = self._last_curvature_map
            fragile_dims = {
                d: round(c, 4)
                for d, c in cm.per_system_curvature.items()
                if c < -0.1
            }
            vulns["fragile_dimensions"] = fragile_dims
            vulns["vulnerable_pairs"] = [
                (a, b, round(c, 4))
                for a, b, c in cm.vulnerable_pairs[:5]
            ]

        # Causal flow anomalies
        if self._last_causal_flow_map is not None:
            cf = self._last_causal_flow_map
            vulns["unexpected_influences"] = [
                {
                    "source": a.source_system,
                    "target": a.target_system,
                    "te": round(a.actual_te, 4),
                }
                for a in cf.unexpected_influences[:5]
            ]
            vulns["missing_influences"] = [
                {
                    "source": a.source_system,
                    "target": a.target_system,
                    "expected": round(a.expected_te, 4),
                    "actual": round(a.actual_te, 4),
                }
                for a in cf.missing_influences[:5]
            ]

        # Topological breaches
        if self._last_persistence_diagnosis is not None:
            pd = self._last_persistence_diagnosis
            vulns["topological_breaches"] = len(pd.breaches)
            vulns["topological_fractures"] = len(pd.fractures)
            vulns["novel_cycles"] = len(pd.novel_cycles)

        # Chaotic metrics (positive Lyapunov)
        if self._last_phase_space_report is not None:
            chaotic = [
                {
                    "metric": d.metric,
                    "lyapunov": round(d.largest_lyapunov, 4),
                    "horizon": d.predictability_horizon_cycles,
                }
                for d in self._last_phase_space_report.diagnoses.values()
                if d.largest_lyapunov > 0.02
            ]
            vulns["chaotic_metrics"] = chaotic

        return vulns

    # ─── Phase A: Query Methods ──────────────────────────────────

    def get_organism_state_vector(self) -> OrganismStateVector | None:
        """Returns the last computed organism state vector."""
        return self._current_state_vector

    def get_derivatives(self) -> DerivativeSnapshot | None:
        """Returns the last computed multi-scale derivative snapshot."""
        return self._current_derivatives

    def get_interoceptive_percept(self) -> InteroceptivePercept | None:
        """Returns the last composed interoceptive percept, or None if healthy."""
        return self._current_percept

    @property
    def signal_buffer(self) -> SignalBuffer:
        """Direct access to the signal buffer for external log injection."""
        return self._signal_buffer

    # ─── Closed-Loop Regulation: Query Methods ───────────────────

    def get_cascade_snapshot(self) -> CascadeSnapshot | None:
        """Returns the last cascade risk forecast from structural anticipation."""
        return self._last_cascade_snapshot

    def get_autonomic_actions(self) -> list[AutonomicAction]:
        """Returns the most recent autonomic regulatory actions."""
        return list(self._last_autonomic_actions)

    def get_loop_dispatches(self) -> list[LoopDispatch]:
        """Returns the most recent feedback loop dispatches."""
        return list(self._last_loop_dispatches)

    @property
    def adaptive_setpoints(self) -> AdaptiveSetpointLearner:
        """Direct access to the adaptive setpoint learner for Oneiros consolidation."""
        return self._adaptive_setpoints

    @property
    def cascade_predictor(self) -> CascadePredictor:
        """Direct access to the cascade predictor for external stress injection."""
        return self._cascade_predictor

    @property
    def autonomic_protocol(self) -> AutonomicProtocol:
        """Direct access to the autonomic protocol for external monitoring."""
        return self._autonomic_protocol

    @property
    def loop_executor(self) -> LoopExecutor:
        """Direct access to loop executor for Evo learning reads."""
        return self._loop_executor
