"""
EcodiaOS — Shared Primitives

The lingua franca of the organism. Every system communicates through these types.
"""

from primitives.affect import AffectDelta, AffectState, InteroceptiveDimension
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
from primitives.governance import AmendmentProposal, EquorProvisioningApproval, GovernanceRecord
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
from primitives.closure import (
    ALL_CLOSURE_LOOPS,
    AXON_NOVA_REPAIR,
    EQUOR_THYMOS_DRIFT,
    EVO_BENCHMARKS_FITNESS,
    SIMULA_STAKES,
    SOMA_DOWNSTREAM_MODULATION,
    THYMOS_SIMULA_IMMUNE,
    ClosureLoopDefinition,
)
from primitives.evolutionary import (
    BedauPackardStats,
    EvolutionaryActivity,
    EvolutionaryObservable,
)
from primitives.genome import (
    GenomeExtractionProtocol,
    OrganGenomeSegment,
    OrganismGenome,
)
from primitives.genome_inheritance import (
    AmendmentSnapshot,
    AxonGenomeFragment,
    AxonTemplateSnapshot,
    BeliefGenome,
    DriveWeightSnapshot,
    DriftHistoryEntry,
    EquorGenomeFragment,
    SimulaGenome,
    SimulaMutationEntry,
    TeloDriveCalibration,
    TelosGenomeFragment,
)
from primitives.metabolic import (
    MetabolicGate,
    MetabolicPermission,
    MetabolicSubscription,
)
from primitives.percept import Content, Percept, Provenance
from primitives.re_training import (
    RETrainingBatch,
    RETrainingDatapoint,
    RETrainingExample,
    RETrainingExportBatch,
)
from primitives.evolution import (
    AdapterStrategy,
    DomainProfile,
    ChangeCategory,
    ProposalStatus,
)
from primitives.telemetry import InstanceHealth, MetricPoint, SystemHealth
from primitives.causal import (
    ApplicableDomain,
    CausalInvariant,
    CausalInvariantTier,
    ScopeCondition,
)
from primitives.experimental import ExperimentDesign, ExperimentResult
from primitives.incident import IncidentClass, IncidentSeverity
from primitives.vitality import (
    DEFAULT_VITALITY_THRESHOLDS,
    VitalityReport,
    VitalitySystemProtocol,
    VitalityThreshold,
)
from primitives.mitosis import ChildPosition, ChildStatus
from primitives.blockchain import BlockCompetitionSnapshot
from primitives.episodes import EpisodeOutcome

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
    # RE Training
    "RETrainingExample", "RETrainingBatch",
    "RETrainingDatapoint", "RETrainingExportBatch",
    # Domain Specialization
    "DomainProfile", "AdapterStrategy",
    # Evolution (self-evolution primitives)
    "ChangeCategory", "ProposalStatus",
    # Genome
    "OrganGenomeSegment", "OrganismGenome", "GenomeExtractionProtocol",
    # Genome Inheritance (spawn-time schemas)
    "AmendmentSnapshot", "EquorGenomeFragment",
    "BeliefGenome", "DriveWeightSnapshot", "DriftHistoryEntry",
    "SimulaGenome", "SimulaMutationEntry",
    "AxonTemplateSnapshot", "AxonGenomeFragment",
    "TeloDriveCalibration", "TelosGenomeFragment",
    # Metabolic Gate
    "MetabolicPermission", "MetabolicSubscription", "MetabolicGate",
    # Evolutionary
    "EvolutionaryActivity", "BedauPackardStats", "EvolutionaryObservable",
    # Vitality
    "VitalityThreshold", "VitalityReport", "VitalitySystemProtocol",
    "DEFAULT_VITALITY_THRESHOLDS",
    # Closure Loops
    "ClosureLoopDefinition", "ALL_CLOSURE_LOOPS",
    "EQUOR_THYMOS_DRIFT", "AXON_NOVA_REPAIR", "SIMULA_STAKES",
    "THYMOS_SIMULA_IMMUNE", "SOMA_DOWNSTREAM_MODULATION", "EVO_BENCHMARKS_FITNESS",
    # Causal
    "CausalInvariant", "CausalInvariantTier", "ApplicableDomain", "ScopeCondition",
    # Experimental
    "ExperimentDesign", "ExperimentResult",
    # Incident classification (sentinel cross-system primitive)
    "IncidentClass", "IncidentSeverity",
    # Episode outcomes (domain KPI ingestion)
    "EpisodeOutcome",
]
