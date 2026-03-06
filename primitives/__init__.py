"""
EcodiaOS — Shared Primitives

The lingua franca of the organism. Every system communicates through these types.
"""

from primitives.affect import AffectDelta, AffectState
from primitives.belief import Belief
from primitives.common import (
    AutonomyLevel,
    ConsolidationLevel,
    DriveAlignmentVector,
    EntityType,
    HealthStatus,
    Modality,
    ResourceBudget,
    SalienceVector,
    SourceDescriptor,
    SystemID,
    Verdict,
    new_id,
    utc_now,
)
from primitives.constitutional import ConstitutionalCheck, InvariantResult
from primitives.expression import Expression, ExpressionStrategy, PersonalityVector
from primitives.federation import (
    SHARING_PERMISSIONS,
    TRUST_THRESHOLDS,
    VIOLATION_MULTIPLIER,
    AssistanceRequest,
    AssistanceResponse,
    FederationInteraction,
    FederationLink,
    FederationLinkStatus,
    FilteredKnowledge,
    InstanceIdentityCard,
    InteractionOutcome,
    KnowledgeItem,
    KnowledgeRequest,
    KnowledgeResponse,
    KnowledgeType,
    PrivacyLevel,
    TrustLevel,
    TrustPolicy,
    ViolationType,
)
from primitives.governance import AmendmentProposal, GovernanceRecord
from primitives.intent import (
    Action,
    ActionSequence,
    DecisionTrace,
    EthicalClearance,
    GoalDescriptor,
    Intent,
)
from primitives.memory_trace import (
    Community,
    ConstitutionNode,
    Entity,
    Episode,
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    MemoryTrace,
    MentionRelation,
    RetrievalResult,
    SelfNode,
    SemanticRelation,
)
from primitives.percept import Content, Percept, Provenance
from primitives.telemetry import InstanceHealth, MetricPoint, SystemHealth

__all__ = [
    # Common
    "SystemID", "Modality", "EntityType", "ConsolidationLevel", "AutonomyLevel",
    "Verdict", "HealthStatus", "DriveAlignmentVector", "ResourceBudget",
    "SalienceVector", "SourceDescriptor", "new_id", "utc_now",
    # Percept
    "Percept", "Content", "Provenance",
    # Affect
    "AffectState", "AffectDelta",
    # Memory
    "Episode", "Entity", "Community", "SelfNode", "ConstitutionNode",
    "MentionRelation", "SemanticRelation", "MemoryTrace",
    "MemoryRetrievalRequest", "MemoryRetrievalResponse", "RetrievalResult",
    # Belief
    "Belief",
    # Intent
    "Intent", "GoalDescriptor", "Action", "ActionSequence",
    "EthicalClearance", "DecisionTrace",
    # Constitutional
    "ConstitutionalCheck", "InvariantResult",
    # Expression
    "Expression", "ExpressionStrategy", "PersonalityVector",
    # Governance
    "AmendmentProposal", "GovernanceRecord",
    # Telemetry
    "MetricPoint", "SystemHealth", "InstanceHealth",
    # Federation
    "InstanceIdentityCard", "FederationLink", "FederationLinkStatus",
    "TrustLevel", "TrustPolicy", "ViolationType", "InteractionOutcome",
    "FederationInteraction", "KnowledgeType", "PrivacyLevel",
    "KnowledgeItem", "KnowledgeRequest", "KnowledgeResponse",
    "FilteredKnowledge", "AssistanceRequest", "AssistanceResponse",
    "SHARING_PERMISSIONS", "TRUST_THRESHOLDS", "VIOLATION_MULTIPLIER",
]
