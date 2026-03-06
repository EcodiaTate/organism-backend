"""
Atune — Perception Gateway & Global Workspace.

EOS's sensory cortex and consciousness.  Receives all input, passes it
to Fovea for prediction-error salience scoring, and broadcasts the winning
content to all cognitive systems via the Global Workspace.

Salience is owned by Fovea. Affect state is owned by Soma.
"""

from .service import AtuneConfig, AtuneService
from .types import (
    ActiveGoalSummary,
    Alert,
    AttentionContext,
    AtuneCache,
    EntityCandidate,
    ExtractionResult,
    HeadMomentum,
    InputChannel,
    LearnedPattern,
    MemoryContext,
    MetaContext,
    NormalisedPercept,
    PendingDecision,
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
)
from .workspace import BroadcastSubscriber, GlobalWorkspace

__all__ = [
    # Service
    "AtuneService",
    "AtuneConfig",
    # Workspace
    "GlobalWorkspace",
    "BroadcastSubscriber",
    # Types
    "ActiveGoalSummary",
    "Alert",
    "AttentionContext",
    "AtuneCache",
    "EntityCandidate",
    "ExtractionResult",
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
