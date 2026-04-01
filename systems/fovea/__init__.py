"""
Fovea - The Unified Perception Engine.

EOS's predictive coding layer and perception pipeline. Fovea computes
prediction errors against the world model, manages the Global Workspace,
handles input normalisation, and broadcasts winning content to all cognitive
systems.

Fovea provides prediction error decomposition as the salience signal.
Salience IS prediction error: what the model got wrong is what gets attended to.

This module is the canonical source for all perception-pipeline types,
including types formerly defined in systems.atune.
"""

from .block_competition import BlockCompetitionMonitor
from .extraction import ExtractionLLMClient, extract_entities_and_relations
from .gateway import AtuneConfig, AtuneService, GatewayConfig, PerceptionGateway
from .gradient import (
    compute_gradient_attention_analytical,
    compute_gradient_attention_numerical,
    compute_token_salience,
)
from .habituation import HabituationCompleteInfo, HabituationEngine, SignatureStats
from .helpers import (
    clamp,
    compute_hash_chain,
    cosine_similarity,
    detect_causal_language,
    detect_conflict,
    detect_direct_address,
    detect_distress,
    detect_positive_emotion,
    detect_risk_patterns,
    detect_urgency,
    estimate_consequence_scope,
    estimate_temporal_proximity,
    hash_content,
    match_keyword_set,
)
from .integration import DynamicIgnitionThreshold, FoveaAtuneBridge
from .internal import InternalPredictionEngine
from .learning import AttentionWeightLearner
from .normalisation import (
    CHANNEL_NORMALISERS,
    ChannelNormaliser,
    NormalisationLLMClient,
    normalise,
    normalise_enriched,
)
from .precision import PrecisionWeightComputer
from .prediction import FoveaPredictionEngine
from .protocols import LogosWorldModel, LogosWorldModelAdapter, StubWorldModel
from .service import FoveaService
from .types import (
    # Types migrated from Atune
    ActiveGoalSummary,
    # Fovea-native types
    ActivePrediction,
    Alert,
    AttentionContext,
    AttentionProfile,
    AtuneCache,
    EntityCandidate,
    ErrorRoute,
    ErrorType,
    ExtractionResult,
    FoveaCache,
    FoveaMetrics,
    FoveaPredictionError,
    GradientAttentionVector,
    HeadMomentum,
    InputChannel,
    InternalErrorType,
    InternalPredictionError,
    LearnedPattern,
    MemoryContext,
    MetaContext,
    NormalisedPercept,
    PendingDecision,
    PerceptContext,
    PredictionError,
    PredictionErrorDirection,
    RawInput,
    RelationCandidate,
    RiskCategory,
    SalienceVector,
    SentimentAnalysis,
    SystemLoad,
    ThreatAnnotationLocal,
    ThreatTrajectory,
    WorkspaceBroadcast,
    WorkspaceCandidate,
    WorkspaceContext,
    WorkspaceContribution,
    WorldModelUpdate,
)
from .workspace import BroadcastSubscriber, GlobalWorkspace, WorkspaceMemoryClient

__all__ = [
    # Service
    "FoveaService",
    # Gateway (perception orchestrator, formerly AtuneService)
    "PerceptionGateway",
    "GatewayConfig",
    "AtuneService",
    "AtuneConfig",
    # Block competition
    "BlockCompetitionMonitor",
    # Core engines (Phase A+B)
    "FoveaPredictionEngine",
    "PrecisionWeightComputer",
    "HabituationEngine",
    # Phase C: Weight learning
    "AttentionWeightLearner",
    "HabituationCompleteInfo",
    "SignatureStats",
    # Phase D: Self-attention
    "InternalPredictionEngine",
    # Integration
    "FoveaAtuneBridge",
    "DynamicIgnitionThreshold",
    # Workspace
    "GlobalWorkspace",
    "BroadcastSubscriber",
    "WorkspaceMemoryClient",
    # Normalisation
    "normalise",
    "normalise_enriched",
    "NormalisationLLMClient",
    "ChannelNormaliser",
    "CHANNEL_NORMALISERS",
    # Extraction
    "extract_entities_and_relations",
    "ExtractionLLMClient",
    # Gradient attention
    "compute_token_salience",
    "compute_gradient_attention_analytical",
    "compute_gradient_attention_numerical",
    # Helpers
    "cosine_similarity",
    "clamp",
    "hash_content",
    "compute_hash_chain",
    "detect_risk_patterns",
    "detect_distress",
    "detect_conflict",
    "detect_positive_emotion",
    "detect_urgency",
    "detect_causal_language",
    "detect_direct_address",
    "match_keyword_set",
    "estimate_consequence_scope",
    "estimate_temporal_proximity",
    # Protocols
    "LogosWorldModel",
    "LogosWorldModelAdapter",
    "StubWorldModel",
    # Fovea-native types
    "ActivePrediction",
    "AttentionProfile",
    "ErrorRoute",
    "ErrorType",
    "FoveaMetrics",
    "FoveaPredictionError",
    "InternalErrorType",
    "InternalPredictionError",
    "PerceptContext",
    "WorldModelUpdate",
    # Migrated types (formerly atune.types)
    "ActiveGoalSummary",
    "Alert",
    "AttentionContext",
    "AtuneCache",
    "EntityCandidate",
    "ExtractionResult",
    "FoveaCache",
    "GradientAttentionVector",
    "HeadMomentum",
    "InputChannel",
    "LearnedPattern",
    "MemoryContext",
    "MetaContext",
    "NormalisedPercept",
    "PendingDecision",
    "PredictionError",
    "PredictionErrorDirection",
    "RawInput",
    "RelationCandidate",
    "RiskCategory",
    "SalienceVector",
    "SentimentAnalysis",
    "SystemLoad",
    "ThreatAnnotationLocal",
    "ThreatTrajectory",
    "WorkspaceBroadcast",
    "WorkspaceCandidate",
    "WorkspaceContext",
    "WorkspaceContribution",
]
