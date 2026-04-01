"""
EcodiaOS - Simula: Self-Evolution System

The organism's capacity for metamorphosis. Where Evo adjusts the knobs,
Simula redesigns the dashboard.

Public API:
  SimulaService              - main service, wired in main.py
  EvoSimulaBridge            - translates Evo proposals to Simula format
  EvolutionAnalyticsEngine   - evolution quality tracking
  ProposalIntelligence       - dedup, prioritize, dependency analysis
  SimulaCodeAgent            - agentic code generation engine
  EvolutionHistoryManager    - immutable evolution history in Neo4j
  EvolutionProposal          - submitted by Evo when a hypothesis reaches SUPPORTED
  ProposalResult             - outcome of process_proposal()
  CodeChangeResult           - output of the code agent
  ChangeCategory             - taxonomy of allowed (and forbidden) change types
  ChangeSpec                 - formal specification of what to change
  EnrichedSimulationResult   - deep multi-strategy simulation output

Stage 1 enhancements:
  1A: Extended-thinking model routing for governance/high-risk proposals
  1B: Voyage-code-3 embeddings for semantic dedup + find_similar + Neo4j vector index
  1C: KVzip-inspired context compression for agentic tool loops

Stage 2 enhancements (Formal Verification Core):
  2A: Dafny proof-carrying code with Clover pattern
  2B: LLM + Z3 invariant discovery loop
  2C: Static analysis gates (Bandit / Semgrep)
  2D: AgentCoder pattern - test/code separation pipeline

Stage 3 enhancements (Incremental & Learning):
  3A: Salsa incremental verification - dependency-aware memoization
  3B: SWE-grep agentic retrieval - multi-hop code search
  3C: LILO library learning - abstraction extraction from successful proposals

Inspector - Zero-Day Discovery Engine:
  TargetWorkspace      - workspace abstraction (internal/external)
  AttackSurface        - discovered entry point
  VulnerabilityReport  - proven vulnerability + PoC
  InspectionResult           - aggregated hunt results
  InspectorConfig         - authorization and resource limits
"""

# Stage 2D: AgentCoder agents
from systems.simula.agents.test_designer import TestDesignerAgent
from systems.simula.agents.test_executor import TestExecutorAgent
from systems.simula.analytics import EvolutionAnalyticsEngine
from systems.simula.bridge import EvoSimulaBridge
from systems.simula.code_agent import SimulaCodeAgent
from systems.simula.constraint_checker import ConstraintSatisfactionChecker
from systems.simula.evolution_types import (
    FORBIDDEN,
    GOVERNANCE_REQUIRED,
    SELF_APPLICABLE,
    SIMULA_IRON_RULES,
    CanaryDeploymentPlan,
    CanaryTrafficStep,
    CategorySuccessRate,
    CautionAdjustment,
    ChangeCategory,
    ChangeSpec,
    CodeChangeResult,
    ConfigVersion,
    ConstraintViolation,
    CounterfactualResult,
    DependencyImpact,
    EnrichedSimulationResult,
    EvolutionAnalytics,
    EvolutionProposal,
    EvolutionRecord,
    EvoProposalEnriched,
    HealthStatus,
    ProposalCluster,
    ProposalPriority,
    ProposalResult,
    ProposalStatus,
    ResourceCostEstimate,
    RiskLevel,
    SimulaMetrics,
    SimulationResult,
    TriageResult,
    TriageStatus,
)

# Genetic Memory: Genome extraction, seeding, and lineage tracking
from systems.simula.genome import SimulaGenomeExtractor, SimulaGenomeSeeder
from systems.simula.genome_types import (
    CategoryAnalyticsRecord,
    EFECalibrationPoint,
    GenerationRecord,
    GenomeComponent,
    GRPOTrainingExample,
    LibraryAbstractionRecord,
    LineageEvent,
    LineageEventType,
    MutationRecord,
    PopulationSnapshot,
    ProcedureRecord,
    SimulaGenome,
    SimulaGenomeExtractionResult,
    SimulaGenomeSeedingResult,
)
from systems.simula.history import EvolutionHistoryManager

# Inspector: Zero-Day Discovery Engine
from systems.simula.inspector import (
    AttackSurface,
    AttackSurfaceType,
    InspectionResult,
    InspectorConfig,
    TargetType,
    TargetWorkspace,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from systems.simula.learning.lilo import LiloLibraryEngine
from systems.simula.lineage import EvolutionLineageTracker
from systems.simula.proposal_intelligence import ProposalIntelligence
from systems.simula.retrieval.swe_grep import SweGrepRetriever
from systems.simula.service import SimulaService

# Stage 2: Verification bridges
from systems.simula.verification.dafny_bridge import DafnyBridge

# Stage 3: Engines
from systems.simula.verification.incremental import IncrementalVerificationEngine
from systems.simula.verification.static_analysis import StaticAnalysisBridge

# Stages 2 + 3: Verification types
from systems.simula.verification.types import (
    DAFNY_TRIGGERABLE_CATEGORIES,
    AbstractionExtractionResult,
    # Stage 3C: LILO Library Learning
    AbstractionKind,
    # Stage 2A: Dafny
    AgentCoderIterationResult,
    AgentCoderResult,
    CachedVerificationResult,
    CloverRoundResult,
    DafnyVerificationResult,
    DafnyVerificationStatus,
    DiscoveredInvariant,
    FormalVerificationResult,
    FunctionSignature,
    IncrementalVerificationResult,
    InvariantKind,
    InvariantVerificationResult,
    InvariantVerificationStatus,
    LibraryAbstraction,
    LibraryStats,
    RetrievalHop,
    # Stage 3B: SWE-grep Retrieval
    RetrievalToolKind,
    RetrievedContext,
    StaticAnalysisFinding,
    StaticAnalysisResult,
    StaticAnalysisSeverity,
    SweGrepResult,
    TestDesignResult,
    TestExecutionResult,
    # Stage 3A: Incremental Verification
    VerificationCacheStatus,
    VerificationCacheTier,
)
from systems.simula.verification.z3_bridge import Z3Bridge

__all__ = [
    # Services
    "SimulaService",
    "SimulaCodeAgent",
    "EvolutionHistoryManager",
    "EvoSimulaBridge",
    "EvolutionAnalyticsEngine",
    "ProposalIntelligence",
    # Core types
    "ChangeCategory",
    "ChangeSpec",
    "CodeChangeResult",
    "ConfigVersion",
    "EvolutionProposal",
    "EvolutionRecord",
    "ProposalResult",
    "ProposalStatus",
    "RiskLevel",
    "SimulationResult",
    # Enriched types
    "EnrichedSimulationResult",
    "CautionAdjustment",
    "CounterfactualResult",
    "DependencyImpact",
    "ResourceCostEstimate",
    "EvoProposalEnriched",
    "ProposalPriority",
    "ProposalCluster",
    "CategorySuccessRate",
    "EvolutionAnalytics",
    "TriageStatus",
    "TriageResult",
    # Canary deployment
    "CanaryDeploymentPlan",
    "CanaryTrafficStep",
    # Constraint satisfaction
    "ConstraintSatisfactionChecker",
    "ConstraintViolation",
    # Health & metrics
    "HealthStatus",
    "SimulaMetrics",
    # Constants
    "FORBIDDEN",
    "GOVERNANCE_REQUIRED",
    "SELF_APPLICABLE",
    "SIMULA_IRON_RULES",
    # Stage 2: Verification types
    "DafnyVerificationStatus",
    "CloverRoundResult",
    "DafnyVerificationResult",
    "InvariantKind",
    "InvariantVerificationStatus",
    "DiscoveredInvariant",
    "InvariantVerificationResult",
    "StaticAnalysisSeverity",
    "StaticAnalysisFinding",
    "StaticAnalysisResult",
    "TestDesignResult",
    "TestExecutionResult",
    "AgentCoderIterationResult",
    "AgentCoderResult",
    "FormalVerificationResult",
    "DAFNY_TRIGGERABLE_CATEGORIES",
    # Stage 2: Bridges
    "DafnyBridge",
    "Z3Bridge",
    "StaticAnalysisBridge",
    # Stage 2D: Agents
    "TestDesignerAgent",
    "TestExecutorAgent",
    # Stage 3A: Incremental Verification
    "VerificationCacheStatus",
    "VerificationCacheTier",
    "FunctionSignature",
    "CachedVerificationResult",
    "IncrementalVerificationResult",
    "IncrementalVerificationEngine",
    # Stage 3B: SWE-grep Retrieval
    "RetrievalToolKind",
    "RetrievalHop",
    "RetrievedContext",
    "SweGrepResult",
    "SweGrepRetriever",
    # Stage 3C: LILO Library Learning
    "AbstractionKind",
    "LibraryAbstraction",
    "AbstractionExtractionResult",
    "LibraryStats",
    "LiloLibraryEngine",
    # Inspector: Zero-Day Discovery Engine
    "TargetWorkspace",
    "TargetType",
    "AttackSurface",
    "AttackSurfaceType",
    "VulnerabilityReport",
    "VulnerabilitySeverity",
    "InspectionResult",
    "InspectorConfig",
    # Genetic Memory: Genome & Lineage
    "SimulaGenome",
    "SimulaGenomeExtractor",
    "SimulaGenomeSeeder",
    "SimulaGenomeExtractionResult",
    "SimulaGenomeSeedingResult",
    "EvolutionLineageTracker",
    "GenerationRecord",
    "PopulationSnapshot",
    "LineageEvent",
    "LineageEventType",
    "GenomeComponent",
    "MutationRecord",
    "LibraryAbstractionRecord",
    "GRPOTrainingExample",
    "EFECalibrationPoint",
    "CategoryAnalyticsRecord",
    "ProcedureRecord",
]
