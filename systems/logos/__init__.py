"""
EcodiaOS -- Logos (System #21)

The Universal Compression Engine.  Cognitive budget, MDL scoring,
four-stage compression cascade, entropic decay, Schwarzschild
threshold detection, and intelligence metric broadcasting.
"""

from systems.logos.budget import CognitiveBudgetManager
from systems.logos.cascade import CompressionCascade
from systems.logos.decay import EntropicDecayEngine, MemoryStoreProtocol
from systems.logos.holographic import HolographicEncoder
from systems.logos.mdl import MDLEstimator
from systems.logos.protocols import (
    FoveaPredictionProtocol,
    KairosInvariantProtocol,
    OneirosCompressionHooks,
    TelosMetricsProtocol,
)
from systems.logos.schwarzschild import SchwarzchildCognitionDetector
from systems.logos.service import LogosService
from systems.logos.types import (
    CascadeResult,
    CausalLink,
    CognitiveBudgetState,
    CompressionCycleReport,
    CompressionStage,
    CrossDomainTransfer,
    DecayReport,
    DecayType,
    EmpiricalInvariant,
    ExperienceDelta,
    GenerativeSchema,
    IntelligenceMetrics,
    KnowledgeItemType,
    LogosConfig,
    MDLScore,
    MemoryTier,
    Prediction,
    PriorDistribution,
    RawExperience,
    SalientEpisode,
    SchwarzchildIndicators,
    SchwarzchildStatus,
    SelfPredictionRecord,
    SemanticDelta,
    SemanticExtraction,
    StageMetrics,
    WorldModelUpdate,
    WorldModelUpdateType,
)
from systems.logos.world_model import CausalGraph, WorldModel

__all__ = [
    # Service
    "LogosService",
    # Subsystems (Phases A-D)
    "CognitiveBudgetManager",
    "HolographicEncoder",
    "MDLEstimator",
    "WorldModel",
    "CausalGraph",
    "CompressionCascade",
    "EntropicDecayEngine",
    "SchwarzchildCognitionDetector",
    # Protocols
    "FoveaPredictionProtocol",
    "TelosMetricsProtocol",
    "OneirosCompressionHooks",
    "KairosInvariantProtocol",
    "MemoryStoreProtocol",
    # Config
    "LogosConfig",
    # Types -- Budget & MDL
    "CognitiveBudgetState",
    "MDLScore",
    "MemoryTier",
    "KnowledgeItemType",
    # Types -- Experience & Delta
    "RawExperience",
    "SemanticDelta",
    "ExperienceDelta",
    # Types -- Compression Cascade
    "CompressionStage",
    "StageMetrics",
    "SalientEpisode",
    "SemanticExtraction",
    "CascadeResult",
    "CompressionCycleReport",
    # Types -- World Model
    "GenerativeSchema",
    "CausalLink",
    "PriorDistribution",
    "EmpiricalInvariant",
    "Prediction",
    "WorldModelUpdate",
    "WorldModelUpdateType",
    # Types -- Decay
    "DecayType",
    "DecayReport",
    # Types -- Schwarzschild
    "SelfPredictionRecord",
    "CrossDomainTransfer",
    "SchwarzchildIndicators",
    "SchwarzchildStatus",
    # Types -- Intelligence
    "IntelligenceMetrics",
]
