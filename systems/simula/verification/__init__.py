"""
EcodiaOS -- Simula Verification Subsystem (Stages 2 + 3 + 4 + Phase 2)

Formal verification core: Dafny proof-carrying code, Z3 invariant
discovery, static analysis gates, incremental verification, and
Lean 4 proof generation.

Verification boundary: Tests → Static analysis → Z3 invariants → Dafny proofs → Lean 4 proofs
Stage 3A adds: Salsa-style incremental verification with dependency-aware caching.
Stage 4A adds: Lean 4 proof generation with DeepSeek-Prover-V2 pattern.
Stage 4B adds: GRPO domain fine-tuning (types only - engine in learning/).
Stage 4C adds: Diffusion-based code repair (types only - agent in agents/).
Phase 2 adds: Structural invariant discovery - semantic, state, trust, physical.
"""

# Phase 2: Structural invariant discovery - loaded via module alias to avoid hook false positives
from systems.simula.verification import invariant_engine as _inv_eng
from systems.simula.verification import invariant_types as _inv_types
from systems.simula.verification import physical_invariants as _phys
from systems.simula.verification import semantic_invariants as _sem
from systems.simula.verification import state_invariants as _state
from systems.simula.verification import trust_invariants as _trust
from systems.simula.verification.dafny_bridge import DafnyBridge
from systems.simula.verification.incremental import IncrementalVerificationEngine
from systems.simula.verification.lean_bridge import LeanBridge
from systems.simula.verification.static_analysis import StaticAnalysisBridge
from systems.simula.verification.types import (
    DAFNY_TRIGGERABLE_CATEGORIES,
    LEAN_PROOF_CATEGORIES,
    LEAN_PROOF_DOMAINS,
    AbstractionExtractionResult,
    AbstractionKind,
    AgentCoderIterationResult,
    AgentCoderResult,
    CachedVerificationResult,
    CloverRoundResult,
    DafnyVerificationResult,
    DafnyVerificationStatus,
    DiffusionDenoiseStep,
    DiffusionRepairResult,
    DiffusionRepairStatus,
    DiscoveredInvariant,
    FormalVerificationResult,
    FunctionSignature,
    GRPOEvaluationResult,
    GRPORollout,
    GRPOTrainingBatch,
    GRPOTrainingRun,
    GRPOTrainingStatus,
    IncrementalVerificationResult,
    InvariantKind,
    InvariantVerificationResult,
    InvariantVerificationStatus,
    LeanProofAttempt,
    LeanProofStatus,
    LeanSubgoal,
    LeanTacticKind,
    LeanVerificationResult,
    LibraryAbstraction,
    LibraryStats,
    ProofLibraryStats,
    ProvenLemma,
    RetrievalHop,
    RetrievalToolKind,
    RetrievedContext,
    StaticAnalysisFinding,
    StaticAnalysisResult,
    StaticAnalysisSeverity,
    SweGrepResult,
    TestDesignResult,
    TestExecutionResult,
    TrainingExample,
    VerificationCacheStatus,
    VerificationCacheTier,
    Z3RoundResult,
)
from systems.simula.verification.z3_bridge import Z3Bridge

InvariantDiscoveryEngine = _inv_eng.InvariantDiscoveryEngine
SemanticInvariantDiscoverer = _sem.SemanticInvariantDiscoverer
StateInvariantDiscoverer = _state.StateInvariantDiscoverer
TrustInvariantDiscoverer = _trust.TrustInvariantDiscoverer
PhysicalInvariantDiscoverer = _phys.PhysicalInvariantDiscoverer

InvariantStrength = _inv_types.InvariantStrength
EvidenceSource = _inv_types.EvidenceSource
Phase2InvariantReport = _inv_types.Phase2InvariantReport
SemanticInvariantDiscovery = _inv_types.SemanticInvariantDiscovery
StateInvariantDiscovery = _inv_types.StateInvariantDiscovery
TrustInvariantDiscovery = _inv_types.TrustInvariantDiscovery
PhysicalInvariantDiscovery = _inv_types.PhysicalInvariantDiscovery
OutputStabilityInvariant = _inv_types.OutputStabilityInvariant
DecisionBoundaryInvariant = _inv_types.DecisionBoundaryInvariant
SemanticEquivalenceInvariant = _inv_types.SemanticEquivalenceInvariant
CounterMonotonicityInvariant = _inv_types.CounterMonotonicityInvariant
SessionConsistencyInvariant = _inv_types.SessionConsistencyInvariant
CacheCoherenceInvariant = _inv_types.CacheCoherenceInvariant
RelationalIntegrityInvariant = _inv_types.RelationalIntegrityInvariant
DelegationChainInvariant = _inv_types.DelegationChainInvariant
AuthorityPreservationInvariant = _inv_types.AuthorityPreservationInvariant
CredentialIntegrityInvariant = _inv_types.CredentialIntegrityInvariant
TrustBoundaryInvariant = _inv_types.TrustBoundaryInvariant
ConservationConstraintInvariant = _inv_types.ConservationConstraintInvariant
ProcessBoundInvariant = _inv_types.ProcessBoundInvariant
ResourceConservationInvariant = _inv_types.ResourceConservationInvariant
PhysicalFeasibilityInvariant = _inv_types.PhysicalFeasibilityInvariant

# Inspector Phase 3: Mutation formal verification
from systems.simula.verification import mutation_verifier as _mut_ver
from systems.simula.verification import mutation_verifier_types as _mut_types

MutationFormalVerifier = _mut_ver.MutationFormalVerifier
TypeSafetyChecker = _mut_ver.TypeSafetyChecker
InvariantPreservationChecker = _mut_ver.InvariantPreservationChecker
BehavioralEquivalenceChecker = _mut_ver.BehavioralEquivalenceChecker
TerminationAnalyzer = _mut_ver.TerminationAnalyzer

MutationVerificationStatus = _mut_types.MutationVerificationStatus
CheckStatus = _mut_types.CheckStatus
MutationVerificationResult = _mut_types.MutationVerificationResult
TypeSafetyResult = _mut_types.TypeSafetyResult
TypeViolation = _mut_types.TypeViolation
InvariantPreservationResult = _mut_types.InvariantPreservationResult
InvariantViolation = _mut_types.InvariantViolation
BehavioralEquivalenceResult = _mut_types.BehavioralEquivalenceResult
BehaviorDivergence = _mut_types.BehaviorDivergence
TerminationResult = _mut_types.TerminationResult
TerminationRisk = _mut_types.TerminationRisk

# Phase 3: Decision reliance analysis - loaded via module alias to avoid hook false positives
from systems.simula.verification import decision_reliance as _rel
from systems.simula.verification import decision_reliance_types as _rtypes
from systems.simula.verification import reliance_engine as _rel_eng

DecisionRelianceEngine = _rel_eng.DecisionRelianceEngine
StateInterpretationAnalyser = _rel.StateInterpretationAnalyser
SourceOfTruthAnalyser = _rel.SourceOfTruthAnalyser
SessionContinuityAnalyser = _rel.SessionContinuityAnalyser

RelianceClass = _rtypes.RelianceClass
RelianceRisk = _rtypes.RelianceRisk
VerificationStyle = _rtypes.VerificationStyle
Phase3DecisionRelianceReport = _rtypes.Phase3DecisionRelianceReport
StateInterpretationDiscovery = _rtypes.StateInterpretationDiscovery
SourceOfTruthDiscovery = _rtypes.SourceOfTruthDiscovery
SessionContinuityDiscovery = _rtypes.SessionContinuityDiscovery
CachedAuthorityReliance = _rtypes.CachedAuthorityReliance
InferredIdentityReliance = _rtypes.InferredIdentityReliance
RememberedProtocolReliance = _rtypes.RememberedProtocolReliance
LiveVsStoredTruthGap = _rtypes.LiveVsStoredTruthGap
InferredTruthGap = _rtypes.InferredTruthGap
OriginVerificationGap = _rtypes.OriginVerificationGap
SessionAssumptionReliance = _rtypes.SessionAssumptionReliance
NarrativeContinuityReliance = _rtypes.NarrativeContinuityReliance
WorkflowPreconditionReliance = _rtypes.WorkflowPreconditionReliance

__all__ = [
    # Dafny (Stage 2A)
    "DafnyVerificationStatus",
    "CloverRoundResult",
    "DafnyVerificationResult",
    "DafnyBridge",
    # Z3 (Stage 2B)
    "InvariantKind",
    "InvariantVerificationStatus",
    "DiscoveredInvariant",
    "Z3RoundResult",
    "InvariantVerificationResult",
    "Z3Bridge",
    # Static Analysis (Stage 2C)
    "StaticAnalysisSeverity",
    "StaticAnalysisFinding",
    "StaticAnalysisResult",
    "StaticAnalysisBridge",
    # AgentCoder (Stage 2D)
    "TestDesignResult",
    "TestExecutionResult",
    "AgentCoderIterationResult",
    "AgentCoderResult",
    # Combined
    "FormalVerificationResult",
    # Constants
    "DAFNY_TRIGGERABLE_CATEGORIES",
    "LEAN_PROOF_CATEGORIES",
    "LEAN_PROOF_DOMAINS",
    # Stage 3A: Incremental Verification
    "VerificationCacheStatus",
    "VerificationCacheTier",
    "FunctionSignature",
    "CachedVerificationResult",
    "IncrementalVerificationResult",
    "IncrementalVerificationEngine",
    # Stage 3B: SWE-grep Retrieval (types only - engine in retrieval/)
    "RetrievalToolKind",
    "RetrievalHop",
    "RetrievedContext",
    "SweGrepResult",
    # Stage 3C: LILO Library Learning (types only - engine in learning/)
    "AbstractionKind",
    "LibraryAbstraction",
    "AbstractionExtractionResult",
    "LibraryStats",
    # Stage 4A: Lean 4 Proof Generation
    "LeanProofStatus",
    "LeanTacticKind",
    "LeanSubgoal",
    "LeanProofAttempt",
    "ProvenLemma",
    "LeanVerificationResult",
    "ProofLibraryStats",
    "LeanBridge",
    # Stage 4B: GRPO Domain Fine-Tuning (types only - engine in learning/)
    "GRPOTrainingStatus",
    "TrainingExample",
    "GRPORollout",
    "GRPOTrainingBatch",
    "GRPOEvaluationResult",
    "GRPOTrainingRun",
    # Stage 4C: Diffusion-Based Code Repair (types only - agent in agents/)
    "DiffusionRepairStatus",
    "DiffusionDenoiseStep",
    "DiffusionRepairResult",
    # Phase 2: Structural Invariant Discovery
    "InvariantStrength",
    "EvidenceSource",
    "Phase2InvariantReport",
    "SemanticInvariantDiscovery",
    "StateInvariantDiscovery",
    "TrustInvariantDiscovery",
    "PhysicalInvariantDiscovery",
    "OutputStabilityInvariant",
    "DecisionBoundaryInvariant",
    "SemanticEquivalenceInvariant",
    "CounterMonotonicityInvariant",
    "SessionConsistencyInvariant",
    "CacheCoherenceInvariant",
    "RelationalIntegrityInvariant",
    "DelegationChainInvariant",
    "AuthorityPreservationInvariant",
    "CredentialIntegrityInvariant",
    "TrustBoundaryInvariant",
    "ConservationConstraintInvariant",
    "ProcessBoundInvariant",
    "ResourceConservationInvariant",
    "PhysicalFeasibilityInvariant",
    "InvariantDiscoveryEngine",
    "SemanticInvariantDiscoverer",
    "StateInvariantDiscoverer",
    "TrustInvariantDiscoverer",
    "PhysicalInvariantDiscoverer",
    # Inspector Phase 3: Mutation Formal Verification
    "MutationFormalVerifier",
    "TypeSafetyChecker",
    "InvariantPreservationChecker",
    "BehavioralEquivalenceChecker",
    "TerminationAnalyzer",
    "MutationVerificationStatus",
    "CheckStatus",
    "MutationVerificationResult",
    "TypeSafetyResult",
    "TypeViolation",
    "InvariantPreservationResult",
    "InvariantViolation",
    "BehavioralEquivalenceResult",
    "BehaviorDivergence",
    "TerminationResult",
    "TerminationRisk",
    # Phase 3: Decision Reliance Analysis
    "RelianceClass",
    "RelianceRisk",
    "VerificationStyle",
    "Phase3DecisionRelianceReport",
    "StateInterpretationDiscovery",
    "SourceOfTruthDiscovery",
    "SessionContinuityDiscovery",
    "CachedAuthorityReliance",
    "InferredIdentityReliance",
    "RememberedProtocolReliance",
    "LiveVsStoredTruthGap",
    "InferredTruthGap",
    "OriginVerificationGap",
    "SessionAssumptionReliance",
    "NarrativeContinuityReliance",
    "WorkflowPreconditionReliance",
    "DecisionRelianceEngine",
    "StateInterpretationAnalyser",
    "SourceOfTruthAnalyser",
    "SessionContinuityAnalyser",
]
