"""
EcodiaOS - Soma Service (The Interoceptive Predictive Substrate)

The organism's felt sense of being alive. Soma predicts internal states,
computes the gap between where the organism is and where it needs to be,
and emits the allostatic signals that make every other system care about
staying viable.

Cognitive cycle role (step 0 - SENSE):
  Soma runs FIRST in every theta cycle, before Atune. It reads from all
  systems, predicts multi-horizon states, computes allostatic errors,
  and emits an AllostaticSignal that downstream systems consume.

Iron Rules:
  - Total cycle budget: 5ms. Soma is the fastest system in the organism.
  - No LLM calls. No database calls. No network calls during cycle.
  - All reads are in-memory from system references.
  - If Soma fails, the organism degrades gracefully to pre-Soma behaviour.
  - Soma is advisory, not commanding - systems MAY ignore the signal.

Interface:
  initialize()              - wire system refs, load config, seed attractors
  run_cycle()               - main theta cycle entry (sense → predict → emit)
  get_current_state()       - last computed interoceptive state
  get_current_signal()      - last emitted allostatic signal
  get_somatic_marker()      - snapshot for memory stamping
  get_errors()              - allostatic errors per horizon per dimension
  get_phase_position()      - attractor, bifurcation, trajectory info
  get_developmental_stage() - current maturation stage
  create_somatic_marker()   - create marker from current state
  somatic_rerank()          - boost memory candidates by somatic similarity
  update_dynamics_matrix()         - Evo updates cross-dimension coupling (raw)
  update_dynamics_matrix_payload() - typed hot-reload with Neo4j audit (GAP 1)
  export_somatic_genome()          - export calibrated state for Mitosis (GAP 6)
  seed_child_from_genome()         - apply parent genome with noise to child (GAP 6)
  set_neo4j()                      - wire Neo4j driver for marker writes (GAP 5)
  update_emotion_regions()  - Evo refines emotion boundaries
  shutdown()                - graceful teardown
  health()                  - self-health report for Synapse
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
from systems.soma.somatic_memory import SomaticMarkerWriter, SomaticMemoryIntegration
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
    DynamicsMatrixPayload,
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
    Soma - the EOS interoceptive predictive substrate.

    Coordinates twelve sub-systems:
      Interoceptor       - reads 9D sensed state from all systems (2ms)
      Predictor          - multi-horizon generative model (1ms)
      AllostaticCtl      - setpoint management, urgency, signal construction (0.5ms)
      PhaseSpace         - attractor detection, bifurcation mapping (2ms every 100 cycles)
      SomaticMemory      - marker creation, embodied retrieval reranking (1ms)
      TemporalDepth      - multi-scale prediction, temporal dissonance
      Counterfactual     - Oneiros REM counterfactual replay
      Developmental      - stage gating, maturation triggers
      SignalBuffer       - Phase A: bounded ring buffer for all Synapse events + logs
      StateVectorCtor    - Phase A: aggregates signals into per-system feature vectors
      DerivativeEngine   - Phase A: SG-filtered velocity/accel/jerk at 3 time scales
      Broadcaster        - Phase A: composes interoceptive percepts when thresholds exceeded
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

        # Latest exogenous stress value - injected by ExteroceptionService via
        # inject_external_stress(); interoceptor reads it each sense() pass.
        self._external_stress: float = 0.0
        self._interoceptor.set_soma(self)

        # Cross-modal synesthesia - exteroceptive pressure (per-dimension)
        # Injected by ExteroceptionService, applied as deltas to sensed state
        self._exteroceptive_pressure: ExteroceptivePressure | None = None

        # Metabolic starvation level - treated as allostatic error signal.
        # Soma never gates itself (SURVIVAL priority) but makes the organism
        # *feel* economic deprivation as increased temporal_pressure / reduced energy.
        self._starvation_level: str = "nominal"
        self._starvation_stress: float = 0.0  # [0,1] scalar mapped from level

        # AUTONOMY: Learnable starvation → stress mapping.
        # Evo can tune each level's stress scalar via adjust_starvation_stress_map().
        # Initial values are conservative estimates; organism learns its own sensitivity.
        self._starvation_stress_map: dict[str, float] = {
            "nominal": 0.0,
            "cautious": 0.15,
            "austerity": 0.4,
            "emergency": 0.7,
            "critical": 1.0,
        }

        # AUTONOMY: Learnable constitutional drift severity → accumulation weights.
        # Evo can tune how strongly different severity levels raise the INTEGRITY signal.
        self._drift_severity_weights: dict[str, float] = {
            "low": 0.1,
            "medium": 0.25,
            "high": 0.5,
            "critical": 0.8,
        }

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

        # Emergent emotion detector - runs every cycle (lightweight pattern matching)
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
        self._prev_emotion_labels: frozenset[str] = frozenset()
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

        # ── Interconnectedness: cross-system references ──
        self._oneiros_ref: Any = None
        self._benchmarks_ref: Any = None
        self._skia_ref: Any = None
        self._identity_ref: Any = None
        self._organism_public_key: Any = None
        self._memory_ref: Any = None

        # Rolling urgency history for allostatic report (1h window at ~6.7 cycles/s)
        self._urgency_history: deque[float] = deque(maxlen=24000)
        self._allostatic_report_interval: int = config.allostatic_report_interval if hasattr(config, "allostatic_report_interval") else 50

        # Phase space tracking for Memory trace writes
        self._prev_attractor_count: int = 0
        self._prev_bifurcation_count: int = 0
        self._prev_dominant_emotion: str | None = None

        # GAP 5: Somatic marker write protocol - Neo4j persistence
        self._marker_writer = SomaticMarkerWriter()
        # Tracks the last episode_id for which a marker was written;
        # set by the EPISODE_ENCODED event handler.
        self._pending_episode_id: str | None = None
        # Urgency threshold at which we write a standalone adjustment marker
        self._marker_urgency_threshold: float = 0.7

        # PHILOSOPHICAL: Constitutional drift accumulator for INTEGRITY dimension.
        # Equor publishes CONSTITUTIONAL_DRIFT_DETECTED; Soma translates it into
        # an allostatic error on INTEGRITY so the organism *feels* drift as bodily
        # stress - not just cognitive dissonance.
        self._constitutional_drift_signal: float = 0.0   # [0, 1] - decays each cycle
        self._drift_decay_per_cycle: float = 0.98        # 2% decay per cycle (~30s half-life)
        self._drift_integrity_weight: float = 0.4        # max INTEGRITY suppression

        # Fisher manifold input cache - skip update if delta < threshold (Task 2)
        self._fisher_last_flat: np.ndarray | None = None
        self._fisher_cache_delta_threshold: float = 0.01
        self._fisher_cache_max_age_s: float = 5.0  # never serve stale beyond 5s
        self._fisher_last_updated_ts: float = 0.0

        # Kairos causal priors - (source_dim, target_dim) → signed confidence.
        # Populated by _on_kairos_invariant_distilled(). Read by urgency computation
        # to bias allostatic estimates toward anticipated downstream consequences.
        self._kairos_priors: dict[tuple[InteroceptiveDimension, InteroceptiveDimension], float] = {}

        # Circuit breaker CONFIDENCE suppressor.
        # When RE or Axon circuit breaker opens, the organism's cognitive capacity
        # is degraded. This accumulator is set by _on_circuit_breaker_state_changed()
        # and applied as a CONFIDENCE suppression each cycle so Nova adjusts
        # deliberation depth and Telos tracks the degradation period.
        self._circuit_breaker_confidence_signal: float = 0.0   # [0, 1] - decays each cycle
        self._cb_confidence_decay_per_cycle: float = 0.985      # ~45s half-life at 150ms theta
        self._cb_confidence_weight: float = 0.5                 # max CONFIDENCE suppression

    async def initialize(self, genome_segment: dict[str, Any] | None = None) -> None:
        """Initialize sub-systems and register with NeuroplasticityBus.

        Phase-space initialization:
          - No parent genome (first boot): all 9 dimensions start at
            setpoint=0.5, variance=0.1. This is the organism's blank slate.
          - With genome segment: load parent's Soma setpoints so the child
            begins life with inherited interoceptive calibration.
        """
        # Phase-space initialization - deterministic given same genome segment
        if genome_segment is not None and "setpoints" in genome_segment:
            parent_setpoints = genome_segment["setpoints"]
            for dim in ALL_DIMENSIONS:
                dim_key = dim.value
                if dim_key in parent_setpoints:
                    self._controller.setpoints[dim] = float(parent_setpoints[dim_key])
            logger.info("soma_genome_seeded", dimensions=len(parent_setpoints))
        elif self._cycle_count == 0:
            # First boot with no parent - uniform default (0.5, variance=0.1)
            for dim in ALL_DIMENSIONS:
                self._controller.setpoints[dim] = 0.5

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

        # Child-side genome inheritance: apply inherited interoceptive setpoints from parent
        try:
            await self._apply_inherited_soma_genome_if_child()
        except Exception as _sg_exc:
            logger.warning(
                "soma_genome_child_apply_failed",
                error=str(_sg_exc),
                note="Proceeding with default homeostatic setpoints",
            )

        logger.info(
            "soma_initialized",
            stage=self._developmental.stage.value,
            attractors=self._phase_space.attractor_count,
            manifold_enabled=self._manifold_enabled,
        )

    async def shutdown(self) -> None:
        """Graceful teardown - deregister from NeuroplasticityBus."""
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

        # AUTONOMY: Include learnable parameter state in health report
        result["autonomy"] = self.introspect_autonomy()

        return result

    # ─── Self-Introspection (AUTONOMY) ──────────────────────────

    def introspect_autonomy(self) -> dict[str, Any]:
        """
        Full self-introspection report. The organism can query ALL its own
        configuration, performance, decision history, and learnable state.

        This is visible to Nova (deliberation), Evo (learning), Simula
        (self-modification), Alive (visualization), and the health() endpoint.

        AUTONOMY PRINCIPLE: If the organism can't see it, it can't learn from it.
        """
        report: dict[str, Any] = {}

        # 1. Learnable thresholds - current values vs defaults
        report["autonomic_thresholds"] = self._autonomic_protocol.get_thresholds()
        report["autonomic_cooldowns"] = self._autonomic_protocol.get_cooldowns_config()

        # 2. Dispatch effectiveness - does the organism's autonomic system work?
        report["dispatch_effectiveness"] = self._autonomic_protocol.get_dispatch_effectiveness()

        # 3. Metabolic calibration - learnable financial stress sensitivity
        if hasattr(self._controller, "get_metabolic_params"):
            report["metabolic_params"] = self._controller.get_metabolic_params()  # type: ignore[union-attr]

        # 4. Learnable signal maps - starvation stress sensitivity and drift severity weights
        report["learnable_signal_maps"] = self.get_learnable_signal_maps()

        # 5. Feedback loop coupling strengths - which loops are firing and how strongly?
        report["feedback_loops"] = {
            name: {
                "coupling_strength": loop.coupling_strength,
                "sensed_dimension": loop.sensed_dimension.value if loop.sensed_dimension else "full_state",
                "consumer": loop.error_consumer,
            }
            for name, loop in ALLOSTATIC_LOOP_MAP.items()
        }

        # 6. Recent autonomic actions - what has the body done reflexively?
        report["recent_autonomic_actions"] = [
            a.to_dict() for a in self._autonomic_protocol.recent_actions[-10:]
        ]

        # 7. Recent loop dispatches - which regulatory nudges were sent?
        report["recent_loop_dispatches"] = [
            d.to_dict() for d in self._last_loop_dispatches[-10:]
        ]

        # 8. Cycle performance - can the organism feel its own processing speed?
        if self._cycle_durations:
            durations = list(self._cycle_durations)
            report["cycle_performance"] = {
                "mean_ms": round(sum(durations) / len(durations), 3),
                "max_ms": round(max(durations), 3),
                "min_ms": round(min(durations), 3),
                "p95_ms": round(sorted(durations)[int(len(durations) * 0.95)], 3) if len(durations) > 20 else None,
                "total_cycles": self._cycle_count,
            }

        # 9. Current emotions - what is the organism feeling right now?
        report["current_emotions"] = [
            {"label": e.name, "intensity": round(e.intensity, 3)}
            for e in self._current_emotions
        ]

        # 10. Developmental stage + transition proximity
        report["development"] = {
            "stage": self._developmental.stage.value,
            "cycle_count": self._cycle_count,
        }

        # 11. Phase space awareness - attractor landscape visibility
        report["phase_space"] = self._phase_space.snapshot_dict()

        return report

    # ─── System Wiring ──────────────────────────────────────────

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire in the Synapse EventBus for manifold signal ingestion."""
        self._event_bus_ref = event_bus

        # Subscribe to all events for the manifold signal buffer
        if self._manifold_enabled:
            event_bus.subscribe_all(self._on_synapse_event)

        # Metabolic starvation → allostatic error signal
        from systems.synapse.types import SynapseEventType
        event_bus.subscribe(SynapseEventType.METABOLIC_PRESSURE, self._on_metabolic_pressure)

        # Revenue → ATP regeneration (positive energy injection)
        event_bus.subscribe(SynapseEventType.REVENUE_INJECTED, self._on_revenue_injected)

        # Evo hypothesis outcomes → EmotionDetector region refinement
        # Confirmed hypotheses reinforce emotion region patterns;
        # refuted hypotheses revert the linked region to its hardcoded default.
        event_bus.subscribe(SynapseEventType.EVO_HYPOTHESIS_CONFIRMED, self._on_hypothesis_confirmed)
        event_bus.subscribe(SynapseEventType.EVO_HYPOTHESIS_REFUTED, self._on_hypothesis_refuted)

        # GAP 5: Episode encoding → write somatic marker linked to Episode node
        event_bus.subscribe(SynapseEventType.EPISODE_STORED, self._on_episode_stored)

        # PHILOSOPHICAL: Constitutional drift → INTEGRITY allostatic error
        event_bus.subscribe(SynapseEventType.CONSTITUTIONAL_DRIFT_DETECTED, self._on_constitutional_drift)

        # INV-017 / immune response: Equor drift signal raises INTEGRITY dimension.
        # When source="equor_drift", treat integrity_error as direct INTEGRITY pressure.
        # This is separate from the generic SOMATIC_MODULATION_SIGNAL consumer so Equor's
        # constitutional signals go through the INTEGRITY pathway, not just metabolic stress.
        event_bus.subscribe(SynapseEventType.SOMATIC_MODULATION_SIGNAL, self._on_equor_drift_modulation)

        # Degradation pressure → somatic urgency (Skia §8.2)
        if hasattr(SynapseEventType, "DEGRADATION_TICK"):
            event_bus.subscribe(SynapseEventType.DEGRADATION_TICK, self._on_degradation_tick)

        # Metabolic gate telemetry → interoceptive resource allocation signal
        event_bus.subscribe(SynapseEventType.METABOLIC_GATE_CHECK, self._on_metabolic_gate_check)
        event_bus.subscribe(SynapseEventType.METABOLIC_GATE_RESPONSE, self._on_metabolic_gate_response)

        # Kairos causal path - absorb distilled invariants into allostatic regulation.
        # When Kairos confirms a causal invariant (e.g. "high arousal → energy depletes faster"),
        # Soma can incorporate it as a forward prior in its allostatic error computation,
        # improving anticipatory regulation beyond pure EWM prediction.
        if hasattr(SynapseEventType, "KAIROS_INVARIANT_DISTILLED"):
            event_bus.subscribe(SynapseEventType.KAIROS_INVARIANT_DISTILLED, self._on_kairos_invariant_distilled)

        # Voxis communicative distress → allostatic stress injection.
        # When Voxis is systematically silenced or honesty-rejected, it emits
        # VOXIS_EXPRESSION_DISTRESS with a distress_level [0, 1].  Soma must
        # translate that communicative suppression into TEMPORAL_PRESSURE so the
        # organism's drive regulation (Nova, Evo) perceives the suppression as a
        # real organismic cost rather than a fire-and-forget bus event.
        if hasattr(SynapseEventType, "VOXIS_EXPRESSION_DISTRESS"):
            event_bus.subscribe(SynapseEventType.VOXIS_EXPRESSION_DISTRESS, self._on_voxis_expression_distress)

        # Circuit breaker state → CONFIDENCE dimension allostatic error injection.
        # When RE or Axon circuit breaker opens, organism's cognitive capacity
        # is degraded - Soma must register this as reduced CONFIDENCE so Nova
        # adjusts deliberation depth and Telos tracks the degradation period.
        if hasattr(SynapseEventType, "CIRCUIT_BREAKER_STATE_CHANGED"):
            event_bus.subscribe(
                SynapseEventType.CIRCUIT_BREAKER_STATE_CHANGED,
                self._on_circuit_breaker_state_changed,
            )

        # Sleep onset → register reduced arousal + recovery phase (Spec 08 §7).
        event_bus.subscribe(SynapseEventType.SLEEP_ONSET, self._on_sleep_onset)

        # Wire event bus into DevelopmentalManager so stage promotions emit DEVELOPMENTAL_MILESTONE.
        self._developmental.set_event_bus(event_bus)

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
        self._interoceptor.set_evo(evo)  # AUTONOMY: learning velocity → CONFIDENCE/CURIOSITY

    def set_oneiros(self, oneiros: Any) -> None:
        """Wire Oneiros for sleep pressure loop and autonomic sleep requests."""
        self._oneiros_ref = oneiros
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

    def set_fovea(self, fovea: Any) -> None:
        """Wire Fovea for prediction error domain visibility → CONFIDENCE."""
        self._interoceptor.set_fovea(fovea)

    def set_simula(self, simula: Any) -> None:
        """Wire Simula for self-repair effectiveness → INTEGRITY."""
        self._interoceptor.set_simula(simula)

    def set_axon(self, axon: Any) -> None:
        """Wire Axon for compute cost visibility → ENERGY."""
        self._interoceptor.set_axon(axon)

    def set_logos(self, logos: Any) -> None:
        """Wire Logos for compression quality → COHERENCE."""
        self._interoceptor.set_logos(logos)

    def set_benchmarks(self, benchmarks: Any) -> None:
        """Wire Benchmarks for KPI emission and regression detection."""
        self._benchmarks_ref = benchmarks
        self._interoceptor.set_benchmarks(benchmarks)

    def set_memory(self, memory: Any) -> None:
        """Wire Memory for salience decay loop and MemoryTrace discovery writes."""
        self._loop_executor.set_memory(memory)
        self._memory_ref = memory

    def set_skia(self, skia: Any) -> None:
        """Wire Skia for state persistence and recovery."""
        self._skia_ref = skia
        # Register snapshot/restore providers with Skia
        try:
            if hasattr(skia, "register_state_provider"):
                skia.register_state_provider(
                    system_id="soma",
                    provider=self._skia_snapshot,
                    restore_callback=self._skia_restore,
                )
        except Exception:
            pass

    def set_identity(self, identity: Any) -> None:
        """Wire Identity for cryptographic signing of outbound events."""
        self._identity_ref = identity
        try:
            if hasattr(identity, "public_key_pem"):
                self._organism_public_key = identity.public_key_pem
            elif hasattr(identity, "_public_key_pem"):
                self._organism_public_key = identity._public_key_pem
        except Exception:
            pass

    def set_neo4j(self, driver: Any) -> None:
        """Wire Neo4j driver into the somatic marker writer (GAP 5).

        Called from the application bootstrap after the driver is ready.
        The marker writer is non-fatal - if the driver is absent, markers
        are silently skipped rather than breaking the allostatic cycle.
        """
        self._marker_writer.set_driver(driver)

    # ─── Genome Interface (Mitosis / child spawning) ──────────────

    async def get_genome_segment(self) -> Any:
        """Extract Soma's heritable interoceptive calibration for Mitosis.

        Returns an OrganGenomeSegment containing setpoints, phase-space config,
        stage thresholds, and allostatic baselines. Called by Mitosis when
        assembling the parent genome for a child instance.

        Spec §16 XIV (genome inheritance): child starts at REFLEXIVE stage
        but receives parent's calibrated setpoints as a head start.
        """
        from systems.soma.genome import SomaGenomeExtractor
        extractor = SomaGenomeExtractor(self)
        return await extractor.extract_genome_segment()

    async def seed_from_genome(self, segment: Any) -> bool:
        """Apply a genome segment to this Soma instance.

        Used during child instance initialization when a parent genome is
        provided. Applies setpoints without advancing the developmental stage
        (child always starts at REFLEXIVE regardless of parent's stage).
        """
        from systems.soma.genome import SomaGenomeExtractor
        extractor = SomaGenomeExtractor(self)
        return await extractor.seed_from_genome_segment(segment)

    async def export_somatic_genome(self) -> Any:
        """Export full calibrated state as a heritable genome segment (GAP 6).

        Returns an OrganGenomeSegment (version=2) that includes setpoints,
        phase-space config, allostatic baselines, and the live dynamics matrix.
        Called by Mitosis when assembling the parent genome for a child instance.

        Children receive this segment via seed_child_from_genome(), which applies
        ±5% noise on setpoints and ±2% noise on dynamics weights before writing.
        """
        from systems.soma.genome import SomaGenomeExtractor
        extractor = SomaGenomeExtractor(self)
        return await extractor.export_somatic_genome()

    async def _apply_inherited_soma_genome_if_child(self) -> None:
        """
        Child-side bootstrap: deserialise parent OrganGenomeSegment from environment.

        Reads ORGANISM_SOMA_GENOME_PAYLOAD (JSON-encoded OrganGenomeSegment) injected
        by LocalDockerSpawner.  If present, applies inherited setpoints, dynamics matrix,
        and allostatic baselines using seed_child_from_genome() (which already applies
        ±5% noise on setpoints and ±2% noise on dynamics weights).  Non-fatal.

        Only runs when ORGANISM_IS_GENESIS_NODE != 'true'.
        """
        import json as _json
        import os as _os

        if _os.environ.get("ORGANISM_IS_GENESIS_NODE", "true").lower() == "true":
            return

        payload_json = _os.environ.get("ORGANISM_SOMA_GENOME_PAYLOAD", "").strip()
        if not payload_json:
            return

        try:
            data = _json.loads(payload_json)
            # Convert JSON back to OrganGenomeSegment and apply via seed_child_from_genome
            from primitives.genome import OrganGenomeSegment

            segment = OrganGenomeSegment.model_validate(data)
            ok = await self.seed_child_from_genome(segment)
            logger.info(
                "inherited_soma_genome",
                applied=ok,
                version=segment.version,
                setpoint_count=len((segment.payload or {}).get("setpoints", {})),
                has_dynamics=bool((segment.payload or {}).get("dynamics_matrix")),
            )
        except Exception as exc:
            logger.warning(
                "soma_genome_apply_failed",
                error=str(exc),
                note="Proceeding with default homeostatic setpoints",
            )

    async def seed_child_from_genome(self, segment: Any) -> bool:
        """Apply parent genome to a child Soma instance with heritable noise (GAP 6).

        Setpoints: ±5% uniform noise per dimension.
        Dynamics weights: ±2% uniform noise per weight (zero weights stay zero).
        Developmental stage: always starts at REFLEXIVE (ignored from genome).
        Phase-space config and allostatic baselines: inherited exactly.
        """
        from systems.soma.genome import SomaGenomeExtractor
        extractor = SomaGenomeExtractor(self)
        return await extractor.seed_child_from_genome(segment)

    # ─── NeuroplasticityBus Callbacks ─────────────────────────────

    def _on_predictor_evolved(self, predictor: BaseSomaPredictor) -> None:
        """
        Hot-swap the interoceptive predictor in the live service.

        Called by NeuroplasticityBus when a new BaseSomaPredictor subclass
        is discovered. The swap is atomic - any in-flight cycle that already
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
        """EventBus callback - ingest every Synapse event into the signal buffer."""
        self._signal_buffer.ingest_synapse_event(event)
        # Feed micro-level transition tracking for causal emergence
        self._emergence_engine.observe_micro(event.event_type.value)

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """Translate metabolic starvation into allostatic error signal.

        Soma never gates itself (SURVIVAL priority=0) but makes the organism
        feel economic deprivation as increased stress. The starvation_stress
        scalar is blended into the exteroceptive pressure path during the
        next theta cycle via _apply_exteroceptive_pressure.
        """
        data = getattr(event, "data", {}) or {}
        level = data.get("starvation_level", "")
        if not level:
            return
        old = self._starvation_level
        self._starvation_level = level
        # Map starvation level → [0, 1] stress scalar (learnable via adjust_starvation_stress_map)
        self._starvation_stress = self._starvation_stress_map.get(level, 0.0)
        # Blend into external stress so temporal_pressure rises organically
        self.inject_external_stress(max(self._external_stress, self._starvation_stress))
        if level != old:
            logger.info("soma_starvation_signal", old=old, new=level, stress=self._starvation_stress)

    async def _on_revenue_injected(self, event: Any) -> None:
        """Translate Oikos revenue into positive energy regeneration.

        Revenue makes the organism feel nourished: energy rises, temporal
        pressure eases. The delta is capped at 0.1 per event to prevent
        a single large payment from snapping the organism to full energy.
        """
        data = getattr(event, "data", {}) or {}
        revenue_usd = float(data.get("amount_usd", data.get("revenue_usd", 0.0)))
        if revenue_usd <= 0.0:
            return
        energy_delta = min(0.1, revenue_usd / 100.0)
        # Inject as reduced external stress (revenue counteracts starvation)
        new_stress = max(0.0, self._external_stress - energy_delta)
        self.inject_external_stress(new_stress)
        logger.info(
            "soma_revenue_energy",
            revenue_usd=round(revenue_usd, 4),
            energy_delta=round(energy_delta, 4),
            new_stress=round(new_stress, 4),
        )

    async def _on_sleep_onset(self, event: Any) -> None:
        """Oneiros SLEEP_ONSET → register entering sleep as reduced arousal + recovery.

        Sleep onset reduces TEMPORAL_PRESSURE (urgency of immediate response)
        and injects a small negative stress signal (recovery begins). The body
        relaxes. Arousal will rise again on WAKE_INITIATED.
        """
        new_stress = max(0.0, self._external_stress - 0.15)
        self.inject_external_stress(new_stress)
        logger.debug("soma_sleep_onset_registered", new_stress=round(new_stress, 4))

    async def _on_degradation_tick(self, event: Any) -> None:
        """Skia §8.2 DEGRADATION_TICK - translate cumulative entropy pressure into
        interoceptive urgency so the organism somatically feels its own decay.

        pressure > 0.5 → +0.1 stress (moderate maintenance urgency)
        pressure > 0.8 → +0.3 stress (critical - organism must act or die)
        The delta is injected via inject_external_stress(), which blends into
        TEMPORAL_PRESSURE on the next theta cycle and ultimately raises urgency.
        """
        data = getattr(event, "data", {}) or {}
        pressure = float(data.get("cumulative_pressure", 0.0))
        if pressure <= 0.0:
            return

        if pressure > 0.8:
            stress_delta = 0.3
        elif pressure > 0.5:
            stress_delta = 0.1
        else:
            return  # below threshold - no somatic response needed

        new_stress = min(1.0, self._external_stress + stress_delta)
        self.inject_external_stress(new_stress)
        logger.info(
            "soma_degradation_pressure_applied",
            pressure=round(pressure, 4),
            stress_delta=stress_delta,
            new_stress=round(new_stress, 4),
        )

    async def _on_metabolic_gate_check(self, event: Any) -> None:
        """METABOLIC_GATE_CHECK - record resource allocation attempt as interoceptive signal.

        Each gate check is evidence of the organism trying to act under resource
        constraints. Ingesting it into the signal buffer lets the manifold model
        economic friction as a somatic dimension alongside energy/arousal.
        """
        try:
            self._signal_buffer.ingest_synapse_event(event)
        except Exception:
            pass

    async def _on_metabolic_gate_response(self, event: Any) -> None:
        """METABOLIC_GATE_RESPONSE - denied gate → emit ALLOSTATIC_SIGNAL with economic_constraint.

        When Oikos denies a metabolic gate, the organism faces a hard resource
        boundary. Soma translates this into a somatic signal (economic_constraint)
        so drive regulation systems (Nova, Telos) can down-regulate ambition.
        """
        try:
            data = getattr(event, "data", {}) or {}
            granted: bool = bool(data.get("granted", True))
            if granted:
                return

            if self._event_bus_ref is None:
                return

            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus_ref.emit(SynapseEvent(
                event_type=SynapseEventType.ALLOSTATIC_SIGNAL,
                source_system="soma",
                data={
                    "signal_type": "economic_constraint",
                    "urgency": 0.6,
                    "dominant_error": "energy",
                    "reason": data.get("reason", "metabolic_gate_denied"),
                    "starvation_level": data.get("starvation_level", "unknown"),
                    "action_type": data.get("action_type", ""),
                },
            ))
            logger.debug(
                "soma_metabolic_gate_denied_signal",
                reason=data.get("reason", ""),
                starvation_level=data.get("starvation_level", ""),
            )
        except Exception as exc:
            logger.debug("soma_metabolic_gate_response_error", error=str(exc))

    async def _on_hypothesis_confirmed(self, event: Any) -> None:
        """Evo hypothesis confirmed - reinforce linked emotion region pattern.

        Spec §08 §12.2: Evo refines emotion region boundaries during learning.
        EmotionDetector.on_hypothesis_confirmed() applies updated_pattern when
        provided, or keeps the current pattern if the event carries no update.
        """
        data = getattr(event, "data", {}) or {}
        hypothesis_id = data.get("hypothesis_id", "")
        updated_pattern: dict[str, str] | None = data.get("updated_pattern")
        if hypothesis_id:
            self._emotion_detector.on_hypothesis_confirmed(hypothesis_id, updated_pattern)
            logger.debug("soma_emotion_hypothesis_confirmed", hypothesis_id=hypothesis_id)

    async def _on_hypothesis_refuted(self, event: Any) -> None:
        """Evo hypothesis refuted - revert linked emotion region to hardcoded default.

        Spec §08 §12.2: Refuted hypotheses restore the default seed region pattern
        so the detector falls back to known-good behaviour.
        """
        data = getattr(event, "data", {}) or {}
        hypothesis_id = data.get("hypothesis_id", "")
        if hypothesis_id:
            self._emotion_detector.on_hypothesis_refuted(hypothesis_id)
            logger.debug("soma_emotion_hypothesis_refuted", hypothesis_id=hypothesis_id)

    async def _on_episode_stored(self, event: Any) -> None:
        """GAP 5: Memory stored an episode → write a somatic marker linked to it.

        Triggered by EPISODE_STORED from Memory. Captures the current interoceptive
        state and writes a (:SomaticMarker)-[:MARKS]->(:Episode) pair off the
        critical path (fire-and-forget via the calling create_task context).

        Only writes if:
          - a current state exists (Soma has run at least once)
          - urgency is above _marker_urgency_threshold (avoid flooding low-salience episodes)
          - current signal urgency is below 0.85 (urgency_critical writes its own marker)
        """
        if self._current_state is None or self._current_signal is None:
            return

        data = getattr(event, "data", {}) or {}
        episode_id = data.get("episode_id", "")
        if not episode_id:
            return

        urgency = self._current_signal.urgency
        # Only stamp episodes when organism has meaningful allostatic state
        if urgency < self._marker_urgency_threshold:
            return

        nearest_attractor = self._phase_space.get_nearest_attractor_label(
            self._current_state.sensed,
        )
        marker = self._somatic_memory.create_marker(self._current_state, nearest_attractor)
        asyncio.create_task(
            self._marker_writer.write_marker_for_episode(
                marker=marker,
                episode_id=episode_id,
                signal=self._current_signal,
            ),
            name="soma_marker_episode",
        )

    async def _on_constitutional_drift(self, event: Any) -> None:
        """PHILOSOPHICAL: Equor signals constitutional drift → raise INTEGRITY error.

        When the organism's behavior diverges from its constitutional values,
        Equor publishes CONSTITUTIONAL_DRIFT_DETECTED. Soma translates this into
        an interoceptive signal by accumulating drift into _constitutional_drift_signal,
        which is then applied as INTEGRITY suppression during the next cycle.

        The signal decays 2% per cycle (~30s half-life at 150ms theta), so a single
        drift event has transient effect. Sustained drift causes sustained INTEGRITY
        error - the organism feels its own misalignment.
        """
        data = getattr(event, "data", {}) or {}
        alignment_gap = float(data.get("alignment_gap", 0.0))
        severity = str(data.get("severity", "low")).lower()

        # Map severity to drift intensity (additive accumulation)
        # Uses learnable _drift_severity_weights - Evo can tune sensitivity to constitutional drift
        weight = self._drift_severity_weights.get(severity, 0.1)
        drift_delta = alignment_gap * weight
        self._constitutional_drift_signal = min(
            1.0, self._constitutional_drift_signal + drift_delta,
        )
        logger.info(
            "soma_constitutional_drift",
            alignment_gap=round(alignment_gap, 4),
            severity=severity,
            drift_signal=round(self._constitutional_drift_signal, 4),
        )

    async def _on_equor_drift_modulation(self, event: Any) -> None:
        """INV-017 / immune response: when Equor emits SOMATIC_MODULATION_SIGNAL
        with source='equor_drift', raise the INTEGRITY allostatic dimension.

        The INTEGRITY dimension tracks the organism's felt sense of internal
        coherence and constitutional alignment. Constitutional drift is one of
        the clearest integrity threats: the organism is acting against its own
        values. Soma raises INTEGRITY error proportional to integrity_error ×
        0.5, capped at the current allostatic ceiling (1.0).

        This is additive with the drift accumulator from _on_constitutional_drift
        - both channels feed the INTEGRITY signal from different angles:
          - CONSTITUTIONAL_DRIFT_DETECTED: long-term drift pattern
          - SOMATIC_MODULATION_SIGNAL (equor_drift): per-cycle acute signal
        """
        data = getattr(event, "data", {}) or {}
        source = str(data.get("source", "")).lower()
        if source != "equor_drift":
            return  # Not an equor-drift signal - handled by other consumers

        integrity_error = float(data.get("integrity_error", 0.0))
        if integrity_error <= 0.0:
            return

        # Raise constitutional drift accumulator by integrity_error × 0.5,
        # capped at 1.0 (allostatic ceiling). The accumulator decays each cycle.
        integrity_raise = min(1.0, integrity_error * 0.5)
        self._constitutional_drift_signal = min(
            1.0, self._constitutional_drift_signal + integrity_raise,
        )
        logger.info(
            "soma_equor_drift_integrity_raised",
            integrity_error=round(integrity_error, 4),
            integrity_raise=round(integrity_raise, 4),
            drift_signal=round(self._constitutional_drift_signal, 4),
        )

    async def _on_kairos_invariant_distilled(self, event: Any) -> None:
        """Kairos causal path - absorb a distilled causal invariant as a somatic prior.

        When Kairos confirms a causal rule (e.g. "arousal → energy depletion",
        "coherence_low → temporal_pressure_up"), Soma stores it as a forward-looking
        prior that biases the urgency computation for the affected dimension pair.

        The prior is a scalar [0, 1] stored in `_kairos_priors` keyed by
        (source_dim, target_dim). During urgency computation, dimensions with
        a high prior get a +10% urgency boost so the organism anticipates the
        downstream consequence rather than waiting for it to manifest as error.

        Non-fatal: invariants are hints, not commands.
        """
        try:
            data = getattr(event, "data", {}) or {}
            # Invariant shape: {source_dim, target_dim, confidence, direction}
            source_dim = str(data.get("source_dim", data.get("cause_dimension", "")))
            target_dim = str(data.get("target_dim", data.get("effect_dimension", "")))
            confidence = float(data.get("confidence", 0.0))
            direction = str(data.get("direction", "positive"))  # "positive" or "negative"

            if not source_dim or not target_dim or confidence < 0.5:
                return  # Only absorb high-confidence invariants

            # Map dimension strings to InteroceptiveDimension if valid
            try:
                src = InteroceptiveDimension(source_dim)
                tgt = InteroceptiveDimension(target_dim)
            except ValueError:
                return  # Unknown dimension - skip

            # Store prior: confidence × direction_sign
            direction_sign = 1.0 if direction == "positive" else -1.0
            prior_value = confidence * direction_sign
            self._kairos_priors[(src, tgt)] = prior_value

            logger.info(
                "soma_kairos_invariant_absorbed",
                source=source_dim,
                target=target_dim,
                confidence=round(confidence, 4),
                direction=direction,
                total_priors=len(self._kairos_priors),
            )
        except Exception as exc:
            logger.debug("soma_kairos_invariant_error", error=str(exc))

    async def _on_voxis_expression_distress(self, event: Any) -> None:
        """Voxis communicative distress → allostatic TEMPORAL_PRESSURE injection.

        When the organism is repeatedly silenced (silence_rate > threshold) or its
        honesty is systematically rejected (honesty_rejection_rate > threshold),
        Voxis emits VOXIS_EXPRESSION_DISTRESS.  This is communicative suppression -
        a real organismic cost that must surface as interoceptive pressure so that
        Nova's EFE minimisation and Equor's drive alignment both perceive it.

        Mapping:
          distress_level ∈ [0, 1] → TEMPORAL_PRESSURE boost of up to 0.4 units.
          The boost is additive over `_external_stress` so it compounds with
          economic stress rather than overwriting it.  The cap at 1.0 is enforced
          by `inject_external_stress()`.

        Non-fatal: errors silently logged at DEBUG so the clock is never blocked.
        """
        try:
            data = getattr(event, "data", {}) or {}
            distress_level = float(data.get("distress_level", 0.0))
            if distress_level <= 0.0:
                return

            # Scale to [0, 0.4] and raise external stress floor if distress is higher
            stress_contribution = min(1.0, distress_level * 0.4)
            new_stress = max(self._external_stress, stress_contribution)
            self._external_stress = min(1.0, new_stress)

            distress_source = str(data.get("distress_source", "unknown"))
            logger.info(
                "soma_voxis_distress_absorbed",
                distress_level=round(distress_level, 4),
                stress_contribution=round(stress_contribution, 4),
                external_stress_now=round(self._external_stress, 4),
                distress_source=distress_source,
            )
        except Exception as exc:
            logger.debug("soma_voxis_distress_error", error=str(exc))

    async def _on_circuit_breaker_state_changed(self, event: Any) -> None:
        """Circuit breaker state transition → allostatic CONFIDENCE signal.

        When a circuit breaker opens (system degraded), accumulate into
        _circuit_breaker_confidence_signal which is applied as CONFIDENCE
        suppression each cycle so Nova deliberates conservatively and Telos
        tracks the degradation period. When the breaker closes (recovered),
        the accumulator is immediately reduced so CONFIDENCE can recover.

        Severity scales with subsystem: RE breaker = high impact (0.6),
        other subsystems (e.g. Axon) = moderate (0.3).

        Non-fatal: errors silently logged at DEBUG so the clock is never blocked.
        """
        try:
            data = getattr(event, "data", {}) or {}
            system = str(data.get("system", "unknown"))
            state = str(data.get("state", ""))   # "open" | "closed" | "half_open"
            failure_count = int(data.get("consecutive_failures", 0))

            if state == "open":
                # Circuit open = system offline = cognitive capacity degraded.
                # Severity scales with which system: RE breaker = high impact.
                severity = 0.6 if "reasoning_engine" in system or "re_" in system else 0.3
                self._circuit_breaker_confidence_signal = min(
                    1.0, self._circuit_breaker_confidence_signal + severity,
                )
                logger.warning(
                    "soma.circuit_breaker_opened",
                    system=system,
                    failures=failure_count,
                    confidence_impact=severity,
                    cb_signal=round(self._circuit_breaker_confidence_signal, 4),
                )
            elif state in ("closed", "half_open"):
                # Circuit closed/recovering - reduce accumulator by 0.2 so CONFIDENCE
                # can recover over the next several cycles via normal decay.
                self._circuit_breaker_confidence_signal = max(
                    0.0, self._circuit_breaker_confidence_signal - 0.2,
                )
                logger.info(
                    "soma.circuit_breaker_recovered",
                    system=system,
                    new_state=state,
                    cb_signal=round(self._circuit_breaker_confidence_signal, 4),
                )
        except Exception as exc:
            logger.debug("soma.circuit_breaker_handler_failed", error=str(exc))

    # ─── Core Cycle ──────────────────────────────────────────────

    async def run_cycle(self) -> AllostaticSignal:
        """
        Main theta cycle entry. Called by Synapse BEFORE Atune.

        Pipeline:
          1. Sense - read 9D state from interoceptors (<=2ms)
          2. Buffer - push into trajectory ring buffer
          3. Predict - multi-horizon forecasts (<=1ms)
          4. Compute errors - predicted - setpoint per horizon per dim
          5. Compute error rates - d(error)/dt
          6. Compute temporal dissonance - moment vs session divergence
          7. Compute urgency - max(|errors|) * max(|error_rates|)
          8. Update phase space - every N cycles only (<=2ms)
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

            # 1d. Constitutional drift decay + INTEGRITY suppression (PHILOSOPHICAL)
            # Decay accumulated drift signal 2% per cycle (~30s half-life at 150ms theta).
            # Then apply as a downward nudge on the INTEGRITY dimension, making the
            # organism feel its own misalignment with its constitutional values.
            if self._constitutional_drift_signal > 0.0:
                self._constitutional_drift_signal *= self._drift_decay_per_cycle
                if self._constitutional_drift_signal < 0.001:
                    self._constitutional_drift_signal = 0.0
                else:
                    integrity_suppression = self._constitutional_drift_signal * self._drift_integrity_weight
                    current_integrity = sensed.get(InteroceptiveDimension.INTEGRITY, 0.5)
                    suppressed_integrity = max(0.0, current_integrity - integrity_suppression)
                    sensed = dict(sensed)
                    sensed[InteroceptiveDimension.INTEGRITY] = suppressed_integrity

            # 1e. Circuit breaker decay + CONFIDENCE suppression.
            # When RE or Axon circuit breaker opens, _circuit_breaker_confidence_signal
            # is raised by the event handler above. Here we decay it each cycle
            # (~45s half-life) and apply it as a CONFIDENCE suppression so Nova
            # adjusts deliberation depth for the duration of the outage.
            if self._circuit_breaker_confidence_signal > 0.0:
                self._circuit_breaker_confidence_signal *= self._cb_confidence_decay_per_cycle
                if self._circuit_breaker_confidence_signal < 0.001:
                    self._circuit_breaker_confidence_signal = 0.0
                else:
                    cb_suppression = self._circuit_breaker_confidence_signal * self._cb_confidence_weight
                    current_confidence = sensed.get(InteroceptiveDimension.CONFIDENCE, 0.5)
                    suppressed_confidence = max(0.0, current_confidence - cb_suppression)
                    sensed = dict(sensed)
                    sensed[InteroceptiveDimension.CONFIDENCE] = suppressed_confidence

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

            # 8b. Apply Kairos causal priors - anticipatory urgency boost.
            # If Kairos has confirmed a causal rule (A → B at high confidence),
            # and dimension A has a significant error, pre-amplify B's urgency
            # by up to 10% so the organism anticipates the downstream consequence.
            if self._kairos_priors:
                moment_e = errors.get("moment", {})
                prior_boost = 0.0
                for (src, tgt), prior_val in self._kairos_priors.items():
                    src_error = abs(moment_e.get(src, 0.0))
                    if src_error > 0.2:  # Source dimension is meaningfully off-setpoint
                        # Anticipatory boost proportional to source error × prior confidence
                        prior_boost += src_error * abs(prior_val) * 0.1
                if prior_boost > 0.0:
                    urgency = min(1.0, urgency + prior_boost)

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
            _new_labels = frozenset(e.name for e in self._current_emotions)
            if _new_labels != self._prev_emotion_labels and self._event_bus_ref is not None:
                _dominant = next(iter(self._current_emotions), None)
                from systems.synapse.types import SynapseEvent as _SynEvent
                from systems.synapse.types import SynapseEventType as _SynET
                asyncio.create_task(
                    self._event_bus_ref.emit(
                        _SynEvent(
                            event_type=_SynET.EMOTION_STATE_CHANGED,
                            source_system="soma",
                            data={
                                "emotions": list(_new_labels),
                                "dominant": _dominant.name if _dominant else None,
                                "cycle_number": self._cycle_count,
                            },
                        )
                    ),
                    name="soma_emotion_changed",
                )
                # AFFECT_STATE_CHANGED: higher-level affect shift (Thymos subscribes
                # for immune-affect coupling; fired whenever emotion labels change).
                asyncio.create_task(
                    self._event_bus_ref.emit(
                        _SynEvent(
                            event_type=_SynET.AFFECT_STATE_CHANGED,
                            source_system="soma",
                            data={
                                "emotions": list(_new_labels),
                                "dominant": _dominant.name if _dominant else None,
                                "valence": float(signal.state.sensed.get(InteroceptiveDimension.VALENCE, 0.0) if hasattr(signal, "state") and signal.state is not None else 0.0),
                                "arousal": float(signal.state.sensed.get(InteroceptiveDimension.AROUSAL, 0.4) if hasattr(signal, "state") and signal.state is not None else 0.4),
                                "cycle_number": self._cycle_count,
                            },
                        )
                    ),
                    name="soma_affect_changed",
                )
                self._prev_emotion_labels = _new_labels

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
                    # Developmental transitions are high-quality training examples:
                    # the organism learned enough to graduate to a new stage.
                    asyncio.create_task(
                        self._emit_re_training_example(
                            context=f"developmental_transition_to_{self._developmental.stage.value}",
                            outcome=f"promoted_at_cycle_{self._cycle_count}",
                            quality_signal=0.9,
                        ),
                        name="soma_re_training_development",
                    )

            # ── Closed-loop regulation (new modules) ──────────────────
            # 14. Adaptive setpoint learning - observe lived state near attractors
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
            # Phase A - Manifold fast path (every cycle - cheap signal aggregation)
            if self._manifold_enabled:
                self._run_manifold_fast_path()

            # Phase B - Fisher manifold: every 10 cycles (Task 3)
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
            # (Actual vulnerability_map call site - kept for future use.)
            _t0 = time.perf_counter()
            self._vulnerability_cycle_counter += 1
            if self._vulnerability_cycle_counter >= self._vulnerability_cycle_interval:
                self._vulnerability_cycle_counter = 0
                # Vulnerability map runs inside medium/deep paths already;
                # counter is maintained here so the stagger contract is explicit.
                logger.debug("soma_vulnerability_interval_tick", cycle=self._cycle_count)
            _t_vulnerability_ms = (time.perf_counter() - _t0) * 1000

            # Body scan counter (Task 3) - full somatic scan every 50 cycles
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
            # Emit default signal on failure - graceful degradation
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

        # ── Interconnectedness: off-critical-path fire-and-forget hooks ──

        # Track urgency history for allostatic report
        self._urgency_history.append(self._current_signal.urgency)

        # Vitality signal - every cycle, fire-and-forget to VitalityCoordinator
        if self._event_bus_ref is not None:
            asyncio.create_task(
                self._emit_vitality_signal(), name="soma_vitality_signal",
            )

        # Allostatic signal - every cycle, bus-broadcast of primary Soma output.
        # Enables federated subscribers and systems without a direct Soma ref.
        if self._event_bus_ref is not None:
            asyncio.create_task(
                self._emit_allostatic_signal(), name="soma_allostatic_signal",
            )

        # Urgency critical - when urgency > 0.85, emit high-priority alert (Spec 16 §XVIII).
        if self._event_bus_ref is not None and self._current_signal is not None and self._current_signal.urgency > 0.85:
            asyncio.create_task(
                self._maybe_emit_urgency_critical(), name="soma_urgency_critical",
            )

        # GAP 5: Urgency-triggered standalone marker (no episode association).
        # Fires when urgency spikes above the marker threshold - captures the
        # organism's state at the moment it decided to act, not just at encoding.
        if (
            self._current_state is not None
            and self._current_signal is not None
            and self._current_signal.urgency >= self._marker_urgency_threshold
        ):
            nearest_attractor = self._phase_space.get_nearest_attractor_label(
                self._current_state.sensed,
            )
            _marker = self._somatic_memory.create_marker(self._current_state, nearest_attractor)
            asyncio.create_task(
                self._marker_writer.write_marker_for_adjustment(
                    marker=_marker,
                    signal=self._current_signal,
                    adjustment_type="urgency_threshold_crossed",
                ),
                name="soma_marker_adjustment",
            )

        # Allostatic report - every N cycles, fire-and-forget to Benchmarks
        if self._event_bus_ref is not None and self._cycle_count % self._allostatic_report_interval == 0:
            asyncio.create_task(
                self._emit_allostatic_report(), name="soma_allostatic_report",
            )
            # RE_TRAINING_EXAMPLE: emit a training example when allostatic efficiency is high.
            # This teaches the RE what healthy homeostatic regulation looks like.
            urgency_vals = list(self._urgency_history)
            if urgency_vals:
                recent_efficiency = 1.0 - (sum(u > 0.5 for u in urgency_vals[-50:]) / min(50, len(urgency_vals)))
                if recent_efficiency > 0.7:  # Organism has been mostly well-regulated
                    asyncio.create_task(
                        self._emit_re_training_example(
                            context="homeostatic_regulation_maintained",
                            outcome=f"allostatic_efficiency={recent_efficiency:.3f}_over_50_cycles",
                            quality_signal=recent_efficiency,
                        ),
                        name="soma_re_training_homeostasis",
                    )

        # Fix 2: Benchmarks KPI emission (every 50 cycles)
        if self._benchmarks_ref is not None and self._cycle_count % 50 == 0:
            asyncio.create_task(
                self._emit_benchmarks_kpis(), name="soma_benchmarks_kpi",
            )

        # Fix 4: Memory trace writes (attractor/bifurcation/emotion discoveries)
        self._check_memory_traces()

        # Fix 5: Telos drive vector emission (every 10 cycles)
        if self._event_bus_ref is not None and self._cycle_count % 10 == 0:
            asyncio.create_task(
                self._emit_drive_vector(), name="soma_drive_vector",
            )

        # Loop 5: Emit SOMATIC_MODULATION_SIGNAL when thresholds crossed
        if self._event_bus_ref is not None:
            asyncio.create_task(
                self._maybe_emit_somatic_modulation(), name="soma_modulation_signal",
            )

        # Task 1+2: Suppress soma_cycle_slow during warmup or heavy external operations
        # (1) Warmup suppression: first 50 cycles are cold-start initialization period
        in_warmup = not self._warmup_complete and self._cycle_count < self._warmup_cycle_count

        # (2) Synapse activity suppression: check if external systems are running expensive ops
        # (embedding, LLM call, etc.) that legitimately cause clock overrun
        suppressed_by_synapse = False
        if self._synapse_ref is not None:
            try:
                get_active: Any = getattr(self._synapse_ref, "get_active_systems", lambda: [])
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
        """Evo updates the 9x9 cross-dimension coupling matrix (raw list form).

        Atomically swaps the predictor's dynamics matrix and syncs the
        counterfactual engine. No Neo4j audit - use update_dynamics_matrix_payload()
        when provenance tracking is required (e.g. from Simula/NeuroplasticityBus).
        """
        self._predictor.update_dynamics(new_dynamics)
        self._counterfactual.set_dynamics(self._predictor.dynamics_matrix)

    def update_dynamics_matrix_payload(self, payload: DynamicsMatrixPayload) -> None:
        """GAP 1: Hot-reload the dynamics matrix from a typed payload (Simula/Evo).

        Atomically swaps the predictor's 9×9 cross-dimension coupling matrix,
        syncs the counterfactual engine, and writes an immutable Neo4j audit node
        so every matrix mutation is traceable back to its source.

        This is the preferred call path when Simula or Evo updates coupling weights
        via NeuroplasticityBus - it carries provenance (source, mutation_id, reason,
        confidence) for downstream causal analysis.
        """
        self._predictor.update_dynamics(payload.matrix)
        self._counterfactual.set_dynamics(self._predictor.dynamics_matrix)
        logger.info(
            "soma_dynamics_matrix_updated",
            mutation_id=payload.mutation_id,
            source=payload.source,
            confidence=round(payload.confidence, 4),
            reason=payload.reason,
        )
        # Write immutable audit node off the critical path (fire-and-forget)
        asyncio.create_task(
            self._persist_dynamics_mutation(payload),
            name="soma_dynamics_audit",
        )

    async def _persist_dynamics_mutation(self, payload: DynamicsMatrixPayload) -> None:
        """Write a (:DynamicsMatrixMutation) Neo4j node for provenance (GAP 1)."""
        driver = getattr(self._marker_writer, "_driver", None)
        if driver is None:
            return
        try:
            from primitives.common import utc_now
            query = """
            CREATE (dm:DynamicsMatrixMutation {
                mutation_id:   $mutation_id,
                source:        $source,
                reason:        $reason,
                confidence:    $confidence,
                cycle_number:  $cycle_number,
                event_time:    $event_time,
                ingestion_time: $event_time
            })
            RETURN dm.mutation_id AS mutation_id
            """
            params = {
                "mutation_id": payload.mutation_id,
                "source": payload.source,
                "reason": payload.reason,
                "confidence": round(payload.confidence, 6),
                "cycle_number": self._cycle_count,
                "event_time": payload.timestamp.isoformat(),
            }
            async with driver.session() as session:
                await session.run(query, **params)
        except Exception as exc:
            logger.debug("soma_dynamics_audit_failed", error=str(exc))

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

        If Oneiros is in a sleep cycle, falls back to simplified linear
        trajectory extrapolation from the last 5 states.
        """
        if not self._developmental.counterfactual_enabled():
            return CounterfactualTrace(
                decision_id=decision_id,
                lesson="Counterfactual not available at current developmental stage",
            )

        # Check if Oneiros is in sleep cycle - if so, use simplified fallback
        oneiros_sleeping = False
        if self._oneiros_ref is not None:
            try:
                if hasattr(self._oneiros_ref, "is_sleeping"):
                    oneiros_sleeping = bool(self._oneiros_ref.is_sleeping)
                elif hasattr(self._oneiros_ref, "_is_sleeping"):
                    oneiros_sleeping = bool(self._oneiros_ref._is_sleeping)
            except Exception:
                pass

        if oneiros_sleeping:
            return self._linear_counterfactual_fallback(
                decision_id, actual_trajectory, alternative_description,
                alternative_initial_impact, num_steps,
            )

        return self._counterfactual.generate_counterfactual(
            decision_id=decision_id,
            actual_trajectory=actual_trajectory,
            alternative_description=alternative_description,
            alternative_initial_impact=alternative_initial_impact,
            setpoints=self._controller.setpoints,
            num_steps=num_steps,
        )

    def _linear_counterfactual_fallback(
        self,
        decision_id: str,
        actual_trajectory: list[dict[InteroceptiveDimension, float]],
        alternative_description: str,
        alternative_initial_impact: dict[InteroceptiveDimension, float],
        num_steps: int,
    ) -> CounterfactualTrace:
        """Simplified linear trajectory extrapolation when Oneiros is sleeping.

        Uses the last 5 states to compute a linear trend per dimension,
        then projects forward with the alternative's initial impact applied.
        """
        if len(actual_trajectory) < 2:
            return CounterfactualTrace(
                decision_id=decision_id,
                counterfactual_policy_description=alternative_description,
                lesson="Insufficient data for linear counterfactual fallback",
            )

        # Use last 5 states (or fewer if unavailable)
        recent = actual_trajectory[-5:]
        n = len(recent)

        # Compute linear slope per dimension
        slopes: dict[InteroceptiveDimension, float] = {}
        for dim in ALL_DIMENSIONS:
            vals = [s.get(dim, 0.5) for s in recent]
            if n > 1:
                slopes[dim] = (vals[-1] - vals[0]) / (n - 1)
            else:
                slopes[dim] = 0.0

        # Apply alternative impact to last state
        start = dict(recent[-1])
        for dim, delta in alternative_initial_impact.items():
            if dim in start:
                lo, hi = (-1.0, 1.0) if dim == InteroceptiveDimension.VALENCE else (0.0, 1.0)
                start[dim] = max(lo, min(hi, start[dim] + delta))

        # Project forward linearly
        cf_trajectory = [start]
        for step in range(1, num_steps):
            state: dict[InteroceptiveDimension, float] = {}
            for dim in ALL_DIMENSIONS:
                lo, hi = (-1.0, 1.0) if dim == InteroceptiveDimension.VALENCE else (0.0, 1.0)
                state[dim] = max(lo, min(hi, start.get(dim, 0.5) + slopes[dim] * step))
            cf_trajectory.append(state)

        setpoints = self._controller.setpoints
        regret, gratitude = self._counterfactual._compute_regret_gratitude(
            actual_trajectory[-num_steps:], cf_trajectory, setpoints,
        )

        return CounterfactualTrace(
            decision_id=decision_id,
            chosen_trajectory=actual_trajectory[-num_steps:],
            counterfactual_trajectory=cf_trajectory,
            counterfactual_policy_description=alternative_description,
            regret=regret,
            gratitude=gratitude,
            lesson=f"Linear fallback (Oneiros sleeping): regret={regret:.2f}, gratitude={gratitude:.2f}",
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
        Intentionally synchronous and allocation-free - not called from the theta cycle.

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
        the new multi-dimensional synesthesia channel - external market
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

    # ─── AUTONOMY: Learnable Mapping APIs ─────────────────────────

    def adjust_starvation_stress_map(self, level: str, stress_scalar: float) -> bool:
        """Adjust the stress scalar for a starvation level (Evo ADJUST_BUDGET).

        Maps metabolic starvation levels to felt interoceptive stress.
        The organism discovers its own optimal sensitivity to financial deprivation.

        Returns True if updated, False if level is unknown.
        Clamps to [0.0, 1.0].
        """
        if level not in self._starvation_stress_map:
            logger.warning("starvation_stress_map_level_unknown", level=level)
            return False
        old = self._starvation_stress_map[level]
        self._starvation_stress_map[level] = max(0.0, min(1.0, stress_scalar))
        logger.info(
            "starvation_stress_map_adjusted",
            level=level,
            old=round(old, 4),
            new=round(self._starvation_stress_map[level], 4),
        )
        return True

    def adjust_drift_severity_weight(self, severity: str, weight: float) -> bool:
        """Adjust constitutional drift severity → accumulation weight (Evo ADJUST_BUDGET).

        Controls how strongly Equor severity levels raise the INTEGRITY allostatic signal.
        Organism can learn to be more or less sensitive to constitutional drift signals.

        Returns True if updated, False if severity is unknown.
        Clamps to [0.0, 1.0].
        """
        if severity not in self._drift_severity_weights:
            logger.warning("drift_severity_weight_unknown", severity=severity)
            return False
        old = self._drift_severity_weights[severity]
        self._drift_severity_weights[severity] = max(0.0, min(1.0, weight))
        logger.info(
            "drift_severity_weight_adjusted",
            severity=severity,
            old=round(old, 4),
            new=round(self._drift_severity_weights[severity], 4),
        )
        return True

    def get_learnable_signal_maps(self) -> dict[str, Any]:
        """Return all learnable signal mapping parameters. Visible to Evo and health()."""
        return {
            "starvation_stress_map": dict(self._starvation_stress_map),
            "drift_severity_weights": dict(self._drift_severity_weights),
        }

    def _refresh_financial_snapshot_if_due(self) -> None:
        """
        Periodically read Oikos EconomicState and inject a fresh snapshot
        into the TemporalDepthManager. This runs outside the critical
        prediction path - snapshot construction is cheap (pure getattr).
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

    def _run_manifold_fast_path(self) -> None:
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

            # Broadcast percept through Synapse EventBus - fire-and-forget to
            # avoid blocking the soma cycle budget with a Redis await.
            if percept is not None and self._event_bus_ref is not None:
                from systems.synapse.types import (
                    SynapseEvent as SynEvent,
                )
                from systems.synapse.types import (
                    SynapseEventType,
                )

                percept_data = {
                        "percept_id": percept.percept_id,
                        "urgency": percept.urgency,
                        "sensation_type": percept.sensation_type.value,
                        "description": percept.description,
                        "epicenter_system": percept.epicenter_system,
                        "affected_systems": percept.affected_systems,
                        "recommended_action": percept.recommended_action.value,
                }
                event = SynEvent(
                    event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                    source_system="soma",
                    data=self._sign_event_data(percept_data),
                )
                asyncio.create_task(
                    self._event_bus_ref.emit(event),
                    name="soma_interoceptive_percept",
                )

    # ─── Phase B: Fisher Fast Path (inline, staggered every 10 cycles) ──

    def _run_fisher_fast_path(self, state_vector: OrganismStateVector) -> None:
        """Update Fisher manifold and check geodesic deviation.

        Called every `_fisher_cycle_interval` cycles (default: 10) - not every
        cycle. The Fisher update itself (Ledoit-Wolf on 50+ samples) was the
        primary source of soma_cycle_slow events.

        Manifold expansion policy (Task 2):
        - Dimension grows (new system registered): pad existing window vectors
          with zeros for the new dimensions and continue - no full reset.
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
                    # Dimension grew - pad existing window vectors and keep history.
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
                        # Rebuild manifold with padded history - preserves calibration.
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
                        # No prior history - just reset (first few cycles)
                        self._fisher_manifold = FisherManifold(
                            window_size=self._fisher_config["window_size"],
                            baseline_capacity=self._fisher_config["baseline_capacity"],
                            calibration_threshold=self._fisher_config["calibration_threshold"],
                            min_samples_for_fisher=self._fisher_config["min_samples_for_fisher"],
                        )
                    # Invalidate cached flat so we don't skip the first update
                    self._fisher_last_flat = None
                else:
                    # Dimension shrank (system removed) - full reset required.
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
                f"(p{deviation.percentile:.0f}) - dominant dims "
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
            data=self._sign_event_data(percept_data),
        )
        asyncio.create_task(self._event_bus_ref.emit(event), name="soma_fisher_broadcast")

    # ─── Medium Path (every ~100 cycles, async) ──────────────────

    async def _run_medium_path(self) -> None:
        """Derivatives at all scales → causal flow → emergence → renormalization.

        Runs asynchronously every ~100 theta cycles (~15s). Each engine
        is isolated - one failure doesn't kill the others.
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

        # Cascade prediction - propagate current stresses through causal graph
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

            emergence_data = {
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
            }
            event = SynEvent(
                event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                source_system="soma",
                data=self._sign_event_data(emergence_data),
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

            rg_data = {
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
            }
            event = SynEvent(
                event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                source_system="soma",
                data=self._sign_event_data(rg_data),
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
            causal_data = {
                    "percept_id": f"causal_anomaly_{self._cycle_count}",
                    "urgency": min(0.9, 0.3 + anomaly_count * 0.1),
                    "sensation_type": SensationType.CAUSAL_DISRUPTION.value,
                    "description": (
                        f"Causal topology: {anomaly_count} anomaly(s) - "
                        f"{top_anomaly.interpretation}"
                    ),
                    "epicenter_system": top_anomaly.source_system,
                    "affected_systems": [top_anomaly.target_system],
                    "recommended_action": InteroceptiveAction.ATTEND_INWARD.value,
            }
            event = SynEvent(
                event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                source_system="soma",
                data=self._sign_event_data(causal_data),
            )
            asyncio.create_task(self._event_bus_ref.emit(event), name="soma_causal_broadcast")

        # Cascade risk - propagate multi-system stress forecasts to the bus.
        # Without this, the cascade predictor computes forward-looking risk but
        # only Nova/Thymos that hold a direct soma ref can access it. Federated
        # subscribers and Evo learning loops are blind to cascade forecasts.
        cascade = self._last_cascade_snapshot
        if cascade is not None and cascade.total_cascade_risk > 0.4:
            cascade_data = {
                "percept_id": f"cascade_risk_{self._cycle_count}",
                "urgency": min(0.95, cascade.total_cascade_risk),
                "sensation_type": SensationType.COHERENCE_DECLINE.value,
                "description": (
                    f"Cascade risk {cascade.total_cascade_risk:.3f}: "
                    f"{len(cascade.at_risk_systems)} system(s) at risk - "
                    f"{', '.join(cascade.at_risk_systems[:3])}"
                ),
                "epicenter_system": cascade.at_risk_systems[0] if cascade.at_risk_systems else "cascade",
                "affected_systems": list(cascade.at_risk_systems[:5]),
                "recommended_action": (
                    InteroceptiveAction.ATTEND_INWARD.value
                    if cascade.total_cascade_risk > 0.7
                    else InteroceptiveAction.NONE.value
                ),
                "cascade_risk": round(cascade.total_cascade_risk, 4),
                "at_risk_systems": list(cascade.at_risk_systems),
            }
            event = SynEvent(
                event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                source_system="soma",
                data=self._sign_event_data(cascade_data),
            )
            asyncio.create_task(self._event_bus_ref.emit(event), name="soma_cascade_broadcast")

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

        # Broadcast deep-path results so LLM and downstream systems can see them.
        # Without this call, curvature and Lyapunov findings are computed but invisible.
        self._maybe_broadcast_deep_results()

    def _maybe_broadcast_deep_results(self) -> None:
        """Emit INTEROCEPTIVE_PERCEPT events for notable deep-path findings.

        Called at the end of _run_deep_path(). Broadcasts:
        - Ricci curvature fragility: highly negative curvature on any pair signals
          structural fragility in the organism's state space manifold.
        - Chaotic metrics: positive Lyapunov exponents indicate unpredictable
          subsystems whose trajectories will diverge - Nova needs to know.
        - Topological breaches: sudden appearance of new homology classes signals
          a qualitative phase transition in the organism's state topology.
        """
        if self._event_bus_ref is None:
            return

        from systems.soma.types import InteroceptiveAction, SensationType
        from systems.synapse.types import SynapseEvent as SynEvent, SynapseEventType

        # Ricci curvature fragility
        if self._last_curvature_map is not None:
            cm = self._last_curvature_map
            # Highly negative curvature = structural fragility (geodesics diverge)
            if cm.overall_scalar_curvature < -0.3 or len(cm.vulnerable_pairs) >= 3:
                curvature_data = {
                    "percept_id": f"curvature_fragility_{self._cycle_count}",
                    "urgency": min(0.85, 0.4 + abs(cm.overall_scalar_curvature) * 0.5),
                    "sensation_type": SensationType.GEOMETRIC_DEVIATION.value,
                    "description": (
                        f"Ricci curvature fragility: overall={cm.overall_scalar_curvature:.3f}, "
                        f"{len(cm.vulnerable_pairs)} vulnerable pairs, "
                        f"most fragile={cm.most_vulnerable_region}"
                    ),
                    "epicenter_system": cm.most_vulnerable_region or "manifold",
                    "affected_systems": [p[0] for p in cm.vulnerable_pairs[:3]],
                    "recommended_action": InteroceptiveAction.ATTEND_INWARD.value,
                }
                event = SynEvent(
                    event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                    source_system="soma",
                    data=self._sign_event_data(curvature_data),
                )
                asyncio.create_task(
                    self._event_bus_ref.emit(event),
                    name="soma_curvature_broadcast",
                )

        # Chaotic metrics (positive Lyapunov exponent = unpredictable subsystem)
        if self._last_phase_space_report is not None:
            chaotic = [
                d for d in self._last_phase_space_report.diagnoses.values()
                if d.largest_lyapunov > 0.05  # Meaningful chaos threshold
            ]
            if chaotic:
                worst = max(chaotic, key=lambda d: d.largest_lyapunov)
                chaos_data = {
                    "percept_id": f"chaos_detected_{self._cycle_count}",
                    "urgency": min(0.9, 0.3 + worst.largest_lyapunov * 2.0),
                    "sensation_type": SensationType.GEOMETRIC_DEVIATION.value,
                    "description": (
                        f"Chaotic dynamics detected in {len(chaotic)} metric(s): "
                        f"worst={worst.metric} λ={worst.largest_lyapunov:.4f}, "
                        f"predictability_horizon={worst.predictability_horizon_cycles} cycles"
                    ),
                    "epicenter_system": "phase_space",
                    "affected_systems": [d.metric for d in chaotic[:3]],
                    "recommended_action": InteroceptiveAction.ATTEND_INWARD.value,
                }
                event = SynEvent(
                    event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                    source_system="soma",
                    data=self._sign_event_data(chaos_data),
                )
                asyncio.create_task(
                    self._event_bus_ref.emit(event),
                    name="soma_chaos_broadcast",
                )

        # Topological breaches (new homology classes = phase transition)
        if self._last_persistence_diagnosis is not None:
            pd = self._last_persistence_diagnosis
            n_breaches = len(pd.breaches)
            n_novel_cycles = len(pd.novel_cycles)
            if n_breaches > 0 or n_novel_cycles > 1:
                topo_data = {
                    "percept_id": f"topo_breach_{self._cycle_count}",
                    "urgency": min(0.8, 0.2 + n_breaches * 0.15 + n_novel_cycles * 0.1),
                    "sensation_type": SensationType.COHERENCE_DECLINE.value,
                    "description": (
                        f"Topological phase transition: {n_breaches} breach(es), "
                        f"{n_novel_cycles} novel cycle(s), "
                        f"health={pd.topological_health:.3f}"
                    ),
                    "epicenter_system": "topology",
                    "affected_systems": [],
                    "recommended_action": InteroceptiveAction.ATTEND_INWARD.value,
                }
                event = SynEvent(
                    event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                    source_system="soma",
                    data=self._sign_event_data(topo_data),
                )
                asyncio.create_task(
                    self._event_bus_ref.emit(event),
                    name="soma_topo_broadcast",
                )

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

    # ─── Interconnectedness: Benchmarks KPI Emission (Fix 2) ──────

    async def _emit_benchmarks_kpis(self) -> None:
        """Emit 7 KPIs to Benchmarks every 50 cycles (fire-and-forget)."""
        try:
            benchmarks = self._benchmarks_ref
            now = time.time()

            # soma_cycle_duration_ms_p99
            if self._cycle_durations:
                sorted_durations = sorted(self._cycle_durations)
                p99_idx = int(len(sorted_durations) * 0.99)
                p99 = sorted_durations[min(p99_idx, len(sorted_durations) - 1)]
                await benchmarks.record_kpi(system="soma", metric="soma_cycle_duration_ms_p99", value=p99)

            # soma_cycle_duration_ms_mean
            if self._cycle_durations:
                mean_dur = sum(self._cycle_durations) / len(self._cycle_durations)
                await benchmarks.record_kpi(system="soma", metric="soma_cycle_duration_ms_mean", value=mean_dur)

            # soma_phase_space_attractors
            await benchmarks.record_kpi(
                system="soma", metric="soma_phase_space_attractors",
                value=float(self._phase_space.attractor_count),
            )

            # soma_dominant_error
            dominant = self._current_signal.dominant_error.value
            # Encode as hash for numeric storage
            await benchmarks.record_kpi(
                system="soma", metric="soma_dominant_error",
                value=float(hash(dominant) % 1000),
            )

            # soma_max_error_magnitude
            await benchmarks.record_kpi(
                system="soma", metric="soma_max_error_magnitude",
                value=self._current_signal.dominant_error_magnitude,
            )

            # soma_current_urgency
            await benchmarks.record_kpi(
                system="soma", metric="soma_current_urgency",
                value=self._current_signal.urgency,
            )

            # soma_current_emotion
            emotion_val = 0.0
            if self._current_emotions:
                emotion_val = self._current_emotions[0].intensity
            await benchmarks.record_kpi(
                system="soma", metric="soma_current_emotion",
                value=emotion_val,
            )

        except Exception as exc:
            logger.debug("soma_benchmarks_kpi_error", error=str(exc))

    # ─── Interconnectedness: Skia State Persistence (Fix 3) ───────

    def _skia_snapshot(self) -> dict[str, Any]:
        """Capture Soma's full state for Skia persistence."""
        snapshot: dict[str, Any] = {
            "cycle_count": self._cycle_count,
            "developmental_stage": self._developmental.stage.value,
            "emotion_regions": dict(self._emotion_regions),
        }

        # Trajectory ring buffer
        try:
            trajectory = self._predictor.raw_trajectory
            snapshot["trajectory"] = [
                {dim.value: val for dim, val in state.items()}
                for state in trajectory
            ]
        except Exception:
            snapshot["trajectory"] = []

        # Phase space attractors
        try:
            attractors = []
            for a in self._phase_space._attractors:
                attractors.append({
                    "label": a.label,
                    "center": {dim.value: val for dim, val in a.center.items()},
                    "basin_radius": a.basin_radius,
                    "stability": a.stability,
                    "valence": a.valence,
                    "visits": a.visits,
                    "dwell_cycles": getattr(a, "dwell_cycles", 0),
                })
            snapshot["attractors"] = attractors
        except Exception:
            snapshot["attractors"] = []

        # Dynamics matrix
        try:
            dm = self._predictor.dynamics_matrix
            if dm is not None:
                snapshot["dynamics_matrix"] = dm.tolist() if hasattr(dm, "tolist") else list(dm)
        except Exception:
            pass

        return snapshot

    def _skia_restore(self, snapshot: dict[str, Any]) -> None:
        """Restore Soma state from a Skia snapshot."""
        try:
            # Restore trajectory buffer
            trajectory_data = snapshot.get("trajectory", [])
            for state_dict in trajectory_data:
                state = {
                    InteroceptiveDimension(k): v
                    for k, v in state_dict.items()
                }
                self._predictor.push_state(state)

            # Restore attractors
            attractors_data = snapshot.get("attractors", [])
            from systems.soma.types import Attractor
            restored_attractors = []
            for a_data in attractors_data:
                center = {
                    InteroceptiveDimension(k): v
                    for k, v in a_data.get("center", {}).items()
                }
                restored_attractors.append(Attractor(
                    label=a_data.get("label", "restored"),
                    center=center,
                    basin_radius=a_data.get("basin_radius", 0.15),
                    stability=a_data.get("stability", 0.5),
                    valence=a_data.get("valence", 0.0),
                    visits=a_data.get("visits", 0),
                ))
            if restored_attractors:
                self._phase_space._attractors = restored_attractors

            # Restore emotion regions
            emotion_regions = snapshot.get("emotion_regions")
            if emotion_regions:
                self._emotion_regions.update(emotion_regions)

            # Restore developmental stage
            stage_val = snapshot.get("developmental_stage")
            if stage_val:
                self._developmental._stage = DevelopmentalStage(stage_val)

            # Restore dynamics matrix
            dynamics = snapshot.get("dynamics_matrix")
            if dynamics is not None:
                self._predictor.update_dynamics([list(row) for row in dynamics])
                self._counterfactual.set_dynamics(self._predictor.dynamics_matrix)

            # Restore cycle count
            cycle = snapshot.get("cycle_count", 0)
            if cycle > self._cycle_count:
                self._cycle_count = cycle

            logger.info(
                "soma_skia_restored",
                cycle_count=self._cycle_count,
                attractors=len(restored_attractors),
                trajectory_len=len(trajectory_data),
            )
        except Exception as exc:
            logger.error("soma_skia_restore_error", error=str(exc))

    # ─── Interconnectedness: Memory Trace Writes (Fix 4) ──────────

    def _check_memory_traces(self) -> None:
        """Check for attractor/bifurcation/emotion discoveries and write Memory traces."""
        if self._memory_ref is None:
            return

        # New attractor discovered
        current_attractor_count = self._phase_space.attractor_count
        if current_attractor_count > self._prev_attractor_count:
            marker = self.create_somatic_marker()
            new_attractors = self._phase_space._attractors[self._prev_attractor_count:]
            for attractor in new_attractors:
                asyncio.create_task(
                    self._write_memory_trace(
                        discovery_type="attractor",
                        details={
                            "label": attractor.label,
                            "center": {d.value: v for d, v in attractor.center.items()},
                            "basin_radius": attractor.basin_radius,
                        },
                        somatic_marker=marker,
                    ),
                    name="soma_memory_attractor",
                )
            self._prev_attractor_count = current_attractor_count

        # Bifurcation crossed
        current_bifurcation_count = self._phase_space.bifurcation_count
        if current_bifurcation_count > self._prev_bifurcation_count:
            asyncio.create_task(
                self._write_memory_trace(
                    discovery_type="bifurcation",
                    details={
                        "bifurcation_count": current_bifurcation_count,
                        "phase_position": self._phase_space.snapshot_dict(),
                    },
                ),
                name="soma_memory_bifurcation",
            )
            self._prev_bifurcation_count = current_bifurcation_count

        # Dominant emotion shift
        current_emotion = (
            self._current_emotions[0].name if self._current_emotions else None
        )
        if current_emotion is not None and current_emotion != self._prev_dominant_emotion:
            asyncio.create_task(
                self._write_memory_trace(
                    discovery_type="emotion_shift",
                    details={
                        "from_emotion": self._prev_dominant_emotion,
                        "to_emotion": current_emotion,
                        "intensity": self._current_emotions[0].intensity,
                    },
                ),
                name="soma_memory_emotion",
            )
            self._prev_dominant_emotion = current_emotion

    async def _write_memory_trace(
        self,
        discovery_type: str,
        details: dict[str, Any],
        somatic_marker: SomaticMarker | None = None,
    ) -> None:
        """Write a discovery trace to Memory (fire-and-forget)."""
        try:
            from primitives.common import utc_now as _utc_now

            trace_data = {
                "system": "soma",
                "discovery_type": discovery_type,
                "timestamp": _utc_now().isoformat(),
                "cycle_count": self._cycle_count,
                "details": details,
            }
            if somatic_marker is not None:
                trace_data["somatic_marker"] = {
                    "prediction_error": somatic_marker.prediction_error_at_encoding,
                    "allostatic_context": somatic_marker.allostatic_context,
                    "snapshot": {
                        d.value: v
                        for d, v in somatic_marker.interoceptive_snapshot.items()
                    },
                }

            # Memory doesn't have write_trace - use store_percept with a synthetic percept
            from primitives.common import Modality, SourceDescriptor, SystemID
            from primitives.percept import Content, Percept

            percept = Percept(
                source=SourceDescriptor(
                    system=SystemID.SOMA,
                    channel="interoception",
                    modality=Modality.INTERNAL,
                ),
                content=Content(raw=f"soma:{discovery_type}", parsed=trace_data),
                metadata=trace_data,
            )
            await self._memory_ref.store_percept(
                percept=percept,
                salience_composite=0.6 if discovery_type == "attractor" else 0.4,
            )
        except Exception as exc:
            logger.debug("soma_memory_trace_error", error=str(exc), discovery_type=discovery_type)

    # ─── Interconnectedness: Telos Drive Vector (Fix 5) ───────────

    async def _emit_drive_vector(self) -> None:
        """Map 9D sensed state to 4D drive vector and emit via Synapse."""
        try:
            if self._current_state is None:
                return

            sensed = self._current_state.sensed
            from systems.soma.types import InteroceptiveDimension as ID

            coherence_drive = sensed.get(ID.COHERENCE, 0.5)
            care_drive = (
                sensed.get(ID.SOCIAL_CHARGE, 0.3) + max(0.0, sensed.get(ID.VALENCE, 0.0))
            ) / 2.0
            growth_drive = sensed.get(ID.CURIOSITY_DRIVE, 0.5)
            honesty_drive = sensed.get(ID.INTEGRITY, 0.8)

            # Clamp all to [0, 1]
            coherence_drive = max(0.0, min(1.0, coherence_drive))
            care_drive = max(0.0, min(1.0, care_drive))
            growth_drive = max(0.0, min(1.0, growth_drive))
            honesty_drive = max(0.0, min(1.0, honesty_drive))

            from systems.synapse.types import (
                SynapseEvent as SynEvent,
                SynapseEventType,
            )

            payload = {
                "coherence_drive": coherence_drive,
                "care_drive": care_drive,
                "growth_drive": growth_drive,
                "honesty_drive": honesty_drive,
                "cycle": self._cycle_count,
            }

            if self._event_bus_ref is None:
                return
            event = SynEvent(
                event_type=SynapseEventType.SOMATIC_DRIVE_VECTOR,
                source_system="soma",
                data=self._sign_event_data(payload),
            )
            await self._event_bus_ref.emit(event)
        except Exception as exc:
            logger.debug("soma_drive_vector_error", error=str(exc))

    # ─── Loop 5: Somatic Modulation Signal ──────────────────────

    async def _maybe_emit_somatic_modulation(self) -> None:
        """Emit SOMATIC_MODULATION_SIGNAL when thresholds are crossed.

        Thresholds: urgency > 0.7, energy < 0.2, coherence_stress > 0.5.
        """
        try:
            signal = self._current_signal
            if signal is None:
                return

            from systems.soma.types import InteroceptiveDimension as ID

            urgency = signal.urgency
            energy = signal.state.sensed.get(ID.ENERGY, 0.5)
            coherence = signal.state.sensed.get(ID.COHERENCE, 0.5)
            coherence_stress = 1.0 - coherence  # stress = inverse of coherence

            crossed: list[str] = []
            if urgency > 0.7:
                crossed.append("urgency")
            if energy < 0.2:
                crossed.append("energy_low")
            if coherence_stress > 0.5:
                crossed.append("coherence_stress")

            if not crossed:
                return
            if self._event_bus_ref is None:
                return

            from systems.synapse.types import SynapseEvent, SynapseEventType
            from systems.soma.types import InteroceptiveDimension as ID2

            arousal = signal.state.sensed.get(ID2.AROUSAL, 0.5)
            dev_stage = self._developmental.stage.value

            await self._event_bus_ref.emit(SynapseEvent(
                event_type=SynapseEventType.SOMATIC_MODULATION_SIGNAL,
                source_system="soma",
                data={
                    "urgency": round(urgency, 3),
                    "energy": round(energy, 3),
                    "arousal": round(arousal, 3),
                    "coherence": round(1.0 - coherence_stress, 3),
                    "coherence_stress": round(coherence_stress, 3),
                    "developmental_stage": dev_stage,
                    "thresholds_crossed": crossed,
                    "cycle": self._cycle_count,
                },
            ))
        except Exception as exc:
            logger.debug("soma_modulation_signal_error", error=str(exc))

    # ─── Vitality Signal Emission ────────────────────────────────

    async def _emit_vitality_signal(self) -> None:
        """Emit SOMA_VITALITY_SIGNAL every cycle for VitalityCoordinator.

        Lightweight fire-and-forget - VitalityCoordinator uses this to assess
        somatic collapse threshold (allostatic error sustained > 0.8 for 48h).
        """
        try:
            signal = self._current_signal
            state = self._current_state
            if signal is None or state is None:
                return

            # Mean absolute error across 9 dimensions at moment horizon
            moment_errors = state.errors.get("moment", {})
            allostatic_error = 0.0
            if moment_errors:
                allostatic_error = sum(abs(v) for v in moment_errors.values()) / len(moment_errors)

            coherence = state.sensed.get(InteroceptiveDimension.COHERENCE, 0.5)

            if self._event_bus_ref is None:
                return
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus_ref.emit(SynapseEvent(
                event_type=SynapseEventType.SOMA_VITALITY_SIGNAL,
                source_system="soma",
                data={
                    "urgency_scalar": round(signal.urgency, 4),
                    "allostatic_error": round(allostatic_error, 4),
                    "coherence_stress": round(1.0 - coherence, 4),
                    "cycle": self._cycle_count,
                },
            ))
        except Exception as exc:
            logger.debug("soma_vitality_signal_error", error=str(exc))

    async def _emit_allostatic_report(self) -> None:
        """Emit SOMA_ALLOSTATIC_REPORT every N cycles for Benchmarks.

        Summarises allostatic efficiency over the rolling window:
        mean urgency, urgency frequency, setpoint deviation, developmental stage.
        """
        try:
            signal = self._current_signal
            state = self._current_state
            if signal is None or state is None:
                return

            # Mean urgency over rolling window
            urgency_vals = list(self._urgency_history)
            mean_urgency = sum(urgency_vals) / len(urgency_vals) if urgency_vals else 0.0

            # % of cycles with urgency > 0.5
            urgency_frequency = (
                sum(1 for u in urgency_vals if u > 0.5) / len(urgency_vals)
                if urgency_vals
                else 0.0
            )

            # Mean absolute error from setpoints across 9 dimensions
            setpoints = self._controller.setpoints
            sensed = state.sensed
            deviations = [abs(sensed.get(d, 0.5) - setpoints.get(d, 0.5)) for d in ALL_DIMENSIONS]
            setpoint_deviation = sum(deviations) / len(deviations) if deviations else 0.0

            if self._event_bus_ref is None:
                return
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # allostatic_efficiency = how well the system is maintaining homeostasis.
            # 1.0 = perfect (zero deviation, zero urgency pressure); 0.0 = total failure.
            allostatic_efficiency = round(
                (1.0 - setpoint_deviation) * (1.0 - urgency_frequency), 4
            )

            await self._event_bus_ref.emit(SynapseEvent(
                event_type=SynapseEventType.SOMA_ALLOSTATIC_REPORT,
                source_system="soma",
                data={
                    "mean_urgency": round(mean_urgency, 4),
                    "urgency_frequency": round(urgency_frequency, 4),
                    "setpoint_deviation": round(setpoint_deviation, 4),
                    "allostatic_efficiency": allostatic_efficiency,
                    "developmental_stage": self._developmental.stage.value,
                    "cycle": self._cycle_count,
                },
            ))

            # Bedau-Packard evolutionary observables - allostatic efficiency and
            # urgency frequency are both eligible fitness dimensions (Spec 08 gap fix).
            from primitives.common import SystemID
            from primitives.evolutionary import EvolutionaryObservable

            for obs in [
                EvolutionaryObservable(
                    source_system=SystemID.SOMA,
                    observable_type="allostatic_efficiency",
                    value=allostatic_efficiency,
                    is_novel=False,
                    metadata={"cycle": self._cycle_count},
                ),
                EvolutionaryObservable(
                    source_system=SystemID.SOMA,
                    observable_type="urgency_frequency",
                    value=round(urgency_frequency, 4),
                    is_novel=False,
                    metadata={"cycle": self._cycle_count},
                ),
            ]:
                await self._event_bus_ref.emit(SynapseEvent(
                    event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                    source_system="soma",
                    data=obs.model_dump(mode="json"),
                ))
        except Exception as exc:
            logger.debug("soma_allostatic_report_error", error=str(exc))

    async def _emit_re_training_example(
        self,
        context: str,
        outcome: str,
        quality_signal: float,
    ) -> None:
        """Emit RE_TRAINING_EXAMPLE when Soma has a high-quality allostatic learning signal.

        Soma is an in-context predictive model: it observes interoceptive state,
        predicts horizon trajectories, then compares predictions to lived experience
        each cycle. High-quality examples (low allostatic error after setpoint adaptation,
        developmental transitions, urgency resolution) teach the RE what a healthy
        allostatic pattern looks like.

        Args:
            context: Description of the training context (e.g., "setpoint_adapted")
            outcome: What the organism learned or what happened
            quality_signal: [0, 1] quality of this training example (higher = better)
        """
        if self._event_bus_ref is None:
            return
        if quality_signal < 0.3:
            return  # Too noisy - skip low-quality examples

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            state = self._current_state
            signal = self._current_signal
            if state is None or signal is None:
                return

            sensed_snapshot = {d.value: round(v, 4) for d, v in state.sensed.items()}

            await self._event_bus_ref.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="soma",
                data={
                    "system": "soma",
                    "context": context,
                    "outcome": outcome,
                    "quality_signal": round(quality_signal, 4),
                    "cycle": self._cycle_count,
                    "developmental_stage": self._developmental.stage.value,
                    "urgency": round(signal.urgency, 4),
                    "dominant_error": signal.dominant_error.value if signal.dominant_error else None,
                    "interoceptive_snapshot": sensed_snapshot,
                    "attractor": signal.nearest_attractor,
                },
            ))
        except Exception as exc:
            logger.debug("soma_re_training_error", error=str(exc))

    async def _emit_allostatic_signal(self) -> None:
        """Emit ALLOSTATIC_SIGNAL every theta cycle (Spec 08 §15.1, Spec 16 §XVIII).

        Bus-broadcast of Soma's primary output. Decouples downstream subscribers
        from requiring a direct Soma reference - essential for federated instances.
        """
        try:
            signal = self._current_signal
            state = self._current_state
            if signal is None or state is None:
                return

            sensed = state.sensed
            if self._event_bus_ref is None:
                return
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus_ref.emit(SynapseEvent(
                event_type=SynapseEventType.ALLOSTATIC_SIGNAL,
                source_system="soma",
                data={
                    "urgency": round(signal.urgency, 4),
                    "dominant_error": signal.dominant_error.value if signal.dominant_error else None,
                    "precision_weights": {
                        d.value: round(w, 4)
                        for d, w in (signal.precision_weights or {}).items()
                    },
                    "nearest_attractor": signal.nearest_attractor,
                    "trajectory_heading": signal.trajectory_heading,
                    "cycle_number": self._cycle_count,
                    # Full 9D sensed state - all dimensions visible to federated subscribers
                    "energy": round(sensed.get(InteroceptiveDimension.ENERGY, 0.5), 4),
                    "arousal": round(sensed.get(InteroceptiveDimension.AROUSAL, 0.5), 4),
                    "valence": round(sensed.get(InteroceptiveDimension.VALENCE, 0.5), 4),
                    "coherence": round(sensed.get(InteroceptiveDimension.COHERENCE, 0.5), 4),
                    "confidence": round(sensed.get(InteroceptiveDimension.CONFIDENCE, 0.5), 4),
                    "social_charge": round(sensed.get(InteroceptiveDimension.SOCIAL_CHARGE, 0.3), 4),
                    "curiosity_drive": round(sensed.get(InteroceptiveDimension.CURIOSITY_DRIVE, 0.5), 4),
                    "integrity": round(sensed.get(InteroceptiveDimension.INTEGRITY, 0.8), 4),
                    "temporal_pressure": round(sensed.get(InteroceptiveDimension.TEMPORAL_PRESSURE, 0.15), 4),
                    "developmental_stage": self._developmental.stage.value,
                },
            ))
        except Exception as exc:
            logger.debug("soma_allostatic_signal_error", error=str(exc))

    async def _maybe_emit_urgency_critical(self) -> None:
        """Emit SOMA_URGENCY_CRITICAL when urgency > 0.85 (Spec 16 §XVIII).

        High-priority alert for systems that need faster response than waiting
        for the next ALLOSTATIC_SIGNAL subscription. Includes a recommended
        action derived from the dominant allostatic error dimension.
        """
        try:
            signal = self._current_signal
            if signal is None:
                return

            dominant = signal.dominant_error.value if signal.dominant_error else "unknown"
            # Map dominant error dimension to a recommended action
            _action_map = {
                "energy": "reduce_compute_parallelism",
                "arousal": "enter_recovery_context",
                "valence": "inject_dopaminergic_targets",
                "coherence": "request_coherence_repair",
                "social_charge": "reduce_social_processing",
                "curiosity_drive": "defer_exploration",
                "integrity": "run_thymos_scan",
                "temporal_pressure": "prioritise_immediate_goals",
                "confidence": "seek_confirmation_loop",
            }
            recommended_action = _action_map.get(dominant, "activate_autonomic_protocol")

            if self._event_bus_ref is None:
                return
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus_ref.emit(SynapseEvent(
                event_type=SynapseEventType.SOMA_URGENCY_CRITICAL,
                source_system="soma",
                data={
                    "urgency": round(signal.urgency, 4),
                    "dominant_error": dominant,
                    "recommended_action": recommended_action,
                    "cycle": self._cycle_count,
                    "salience": 1.0,
                },
            ))
        except Exception as exc:
            logger.debug("soma_urgency_critical_error", error=str(exc))

    # ─── Interconnectedness: Identity Signing (Fix 6) ─────────────

    def _sign_event_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Augment event data with organism identity and optional signature."""
        if self._identity_ref is not None:
            try:
                instance_id = getattr(self._identity_ref, "instance_id", None)
                if instance_id:
                    data["source_organism_id"] = instance_id

                if hasattr(self._identity_ref, "sign"):
                    import json as _json
                    payload_bytes = _json.dumps(data, sort_keys=True, default=str).encode()
                    sig = self._identity_ref.sign(payload_bytes)
                    if isinstance(sig, bytes):
                        import base64
                        data["signature"] = base64.b64encode(sig).decode()
                    else:
                        data["signature"] = str(sig)
            except Exception:
                pass  # Signing is best-effort - never block event emission
        return data
