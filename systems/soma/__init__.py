"""
EcodiaOS — Soma (Interoceptive Predictive Substrate)

The organism's felt sense of being alive. Predicts internal states,
computes allostatic errors, and emits signals that drive regulation.

Includes the Cross-Modal Synesthesia layer (exteroception/) that maps
external world events into somatic pressure.
"""

from systems.soma.adaptive_setpoints import AdaptiveSetpointLearner, AttractorSetpointProfile
from systems.soma.autonomic_protocol import AutonomicAction, AutonomicProtocol
from systems.soma.base import BaseAllostaticRegulator, BaseSomaPredictor
from systems.soma.cascade_predictor import CascadeForecast, CascadePredictor, CascadeSnapshot
from systems.soma.causal_flow import CausalAnomaly, CausalFlowEngine, CausalFlowMap
from systems.soma.curvature_analyzer import CurvatureAnalyzer, CurvatureMap
from systems.soma.emergence import CausalEmergenceEngine, EmergenceReport
from systems.soma.emotions import ActiveEmotion, EmotionDetector, VoxisInteroceptiveInterface
from systems.soma.exteroception import (
    ExteroceptionService,
    ExteroceptiveMappingEngine,
    ExteroceptiveModality,
    ExteroceptivePressure,
    ExteroceptiveReading,
    MarketDataAdapter,
    NewsSentimentAdapter,
)
from systems.soma.feedback_loops import ALLOSTATIC_LOOP_MAP, AllostaticLoop
from systems.soma.fisher_manifold import FisherManifold, GeodesicDeviation
from systems.soma.healing import HealingOutcome, HealingReport, HealingSnapshot, HealingVerifier
from systems.soma.loop_executor import LoopDispatch, LoopExecutor
from systems.soma.metabolic_regulator import MetabolicAllostaticRegulator
from systems.soma.phase_space_reconstructor import (
    AttractorDiagnosis,
    PhaseSpaceReconstructor,
    PhaseSpaceReport,
)
from systems.soma.renormalization import RenormalizationEngine, RGFlowReport
from systems.soma.sensors import ExternalVolatilitySensor
from systems.soma.service import SomaService
from systems.soma.topology import PersistenceDiagnosis, TopologicalAnalyzer, TopologicalFeature
from systems.soma.types import (
    AllostaticSignal,
    Attractor,
    Bifurcation,
    CounterfactualTrace,
    DerivativeSnapshot,
    DevelopmentalStage,
    InteroceptiveAction,
    InteroceptiveDimension,
    InteroceptivePercept,
    InteroceptiveState,
    OrganismStateVector,
    SensationType,
    SignalSource,
    SomaSignal,
    SomaticMarker,
    SystemStateSlice,
)

__all__ = [
    # adaptive_setpoints
    "AdaptiveSetpointLearner",
    "AttractorSetpointProfile",
    # autonomic_protocol
    "AutonomicAction",
    "AutonomicProtocol",
    # base
    "BaseAllostaticRegulator",
    "BaseSomaPredictor",
    # cascade_predictor
    "CascadeForecast",
    "CascadePredictor",
    "CascadeSnapshot",
    # causal_flow
    "CausalAnomaly",
    "CausalFlowEngine",
    "CausalFlowMap",
    # curvature_analyzer
    "CurvatureAnalyzer",
    "CurvatureMap",
    # emergence
    "CausalEmergenceEngine",
    "EmergenceReport",
    # emotions
    "ActiveEmotion",
    "EmotionDetector",
    "VoxisInteroceptiveInterface",
    # feedback_loops
    "ALLOSTATIC_LOOP_MAP",
    "AllostaticLoop",
    # loop_executor
    "LoopDispatch",
    "LoopExecutor",
    # exteroception
    "ExteroceptiveMappingEngine",
    "ExteroceptiveModality",
    "ExteroceptivePressure",
    "ExteroceptiveReading",
    "ExteroceptionService",
    "MarketDataAdapter",
    "NewsSentimentAdapter",
    # fisher_manifold
    "FisherManifold",
    "GeodesicDeviation",
    # healing
    "HealingOutcome",
    "HealingReport",
    "HealingSnapshot",
    "HealingVerifier",
    # metabolic_regulator
    "MetabolicAllostaticRegulator",
    # phase_space_reconstructor
    "AttractorDiagnosis",
    "PhaseSpaceReconstructor",
    "PhaseSpaceReport",
    # renormalization
    "RenormalizationEngine",
    "RGFlowReport",
    # sensors
    "ExternalVolatilitySensor",
    # service
    "SomaService",
    # topology
    "PersistenceDiagnosis",
    "TopologicalAnalyzer",
    "TopologicalFeature",
    # types
    "AllostaticSignal",
    "Attractor",
    "Bifurcation",
    "CounterfactualTrace",
    "DerivativeSnapshot",
    "DevelopmentalStage",
    "InteroceptiveAction",
    "InteroceptiveDimension",
    "InteroceptivePercept",
    "InteroceptiveState",
    "OrganismStateVector",
    "SensationType",
    "SignalSource",
    "SomaSignal",
    "SomaticMarker",
    "SystemStateSlice",
]
