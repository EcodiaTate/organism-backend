"""
EcodiaOS - Simula Inspector: Vulnerability Analysis Engine

Inspector inverts Simula's internal verification logic - instead of proving
code is *correct*, it identifies potential vulnerabilities by translating Z3 SAT
counterexamples into proof-of-concept demonstrations.

Public API (Phases 1-11):
  InspectorService             - full pipeline orchestrator (Phase 7)
  TargetWorkspace           - abstraction for internal or external codebases
  TargetIngestor            - clones repos, builds graphs, maps inspection targets
  VulnerabilityProver       - proves vulnerabilities via Z3 constraint inversion
  InspectorRepairOrchestrator  - autonomous patch generation + Z3 re-verification
  InspectorSafetyGates         - PoC execution, workspace isolation, config validation (Phase 11)
  SafetyResult              - outcome of a safety gate check
  InspectorAnalyticsEmitter    - structured event logging + TSDB persistence (Phase 9)
  InspectorAnalyticsView       - aggregate vulnerability analytics with time-windowed trends
  InspectorAnalyticsStore      - durable TimescaleDB event storage + historical queries
  InspectorEvent               - structured analytics event data model
  TargetType                - INTERNAL_EOS | EXTERNAL_REPO
  AttackSurface             - discovered evidenceable entry point
  VulnerabilityReport       - proven vulnerability with Z3 counterexample + PoC
  InspectionResult                - aggregated results from a full inspection
  InspectorConfig              - authorization and resource limits
  RemediationResult         - outcome of autonomous vulnerability remediation

Advanced Features (Phase 12):
  MultiLanguageSurfaceDetector   - Go, Rust, TypeScript inspection target detection
  EvidenceChainAnalyzer           - multi-vulnerability chain discovery
  EvidenceChain                   - discovered chain of evidenceable vulnerabilities
  UnpatchedVulnerabilityMarketplace             - cryptographic peer-to-peer vulnerability trading
  MarketplaceVulnerabilityListing - marketplace listing with encrypted PoC
  MarketplacePurchaseAgreement    - atomic transaction for unpatched vulnerability sale
  AutonomousPatchingOrchestrator  - auto-generate GitHub pull requests with patches
  GitHubPRConfig                 - GitHub API configuration for PR submission
  GitHubPRResult                 - result of PR submission attempt
  ContinuousHuntingScheduler     - recurring hunts with cron-based scheduling
  ScheduledHuntConfig            - configuration for a scheduled hunt
  HuntScheduleRun                - single execution of a scheduled hunt

Phase 1 Observability (Cross-Layer Substrate):
  CorrelationContext    - UUID threading through all observability phases (1–8)
  KernelEventType       - enum of kernel event types (fork, exit, syscall, file, socket)
  KernelEvent           - structured kernel/eBPF event with CorrelationContext
  InteractionEdge       - single directed interaction between two service nodes
  InteractionGraph      - process-to-service topology for a proposal run
  KERNEL_EVENTS_SCHEMA  - TimescaleDB DDL for kernel_events hypertable (in analytics.py)
  ProcessLifecycleMonitor - /proc-based process fork/exit poller (in observer/observer.py)
  InteractionGraphBuilder - builds topology graph from ring buffer + lifecycle events

Phase 2 Runtime Instrumentation (Behaviour Visibility):
  RuntimeTracer                  - function-level + basic-block tracing (settrace / sys.monitoring)
  FaultClassifier                - crash/signal taxonomy: OOB, UAF, lifetime, type confusion
  ControlIntegrityScorer         - per-run control integrity score [0, 1]
  RuntimeInstrumentationEngine   - top-level Phase 2 orchestrator
  TraceEvent                     - single instrumentation event (call/return/bb/branch/fault/signal)
  TraceEventKind                 - enum of trace event granularity levels
  FaultClass                     - fault taxonomy enum (OOB, UAF_DANGLING, LIFETIME, TYPE, …)
  RunCategory                    - NORMAL | FAILURE | CRASH
  ControlFlowTrace               - complete control-flow summary for one run
  BasicBlockTrace                - aggregated BB coverage for one run
  FaultObservation               - classified fault with location + heuristic transition point
  TraceDataset                   - labelled collection of normal + failure runs
  ControlIntegrityScore          - per-run CIS with influence-permissive transition list
  InfluencePermissiveTransition  - single detected steerable call-edge
  FaultClassificationReport      - aggregate fault taxonomy for a dataset
  Phase2Result                   - top-level Phase 2 output (dataset + report + scores)

Phase 5 Trust-Graph Influence Expansion (Post-boundary Propagation):
  TrustAnalyzer          - top-level Phase 5 orchestrator → Phase5Result
  TrustGraphBuilder      - four-pass trust graph construction from Phase 3/4 artifacts
  PropagationEngine      - BFS propagation + privilege gradient + corridor extraction
  TrustGraph             - directed graph of principals, services, resources, roles, credentials, sessions
  TrustNode              - single node with kind + privilege_value + steerability_adjacent flag
  TrustEdge              - directed trust relationship with kind, strength, traversal_cost
  TrustNodeKind          - PRINCIPAL | SERVICE | RESOURCE | ROLE | CREDENTIAL | SESSION
  TrustEdgeKind          - AUTHENTICATION | DELEGATION | ASSUMED_TRUST | IMPLICIT_PERMISSION | …
  TrustStrength          - EXPLICIT | VERIFIED | IMPLICIT | BLIND
  PrivilegeImpact        - CRITICAL | HIGH | MEDIUM | LOW | NONE
  CorridorRiskTier       - CRITICAL | HIGH | MEDIUM | LOW
  FootholdBinding        - links Phase 4 ConditionSet → trust-graph entry node
  PropagationPath        - a complete path from foothold to terminal with per-hop steps
  PropagationStep        - single hop (edge kind + strength + privilege delta)
  PropagationSimulation  - per-foothold BFS result with all paths + corridors
  ReachabilityMap        - aggregate across all footholds; ranked corridors
  ExpansionCorridor      - ranked propagation pathway (the primary Phase 5 deliverable)
  Phase5Result           - top-level output with exit criterion

Phase 6 Protocol State-Machine Stress (Boundary Failure Discovery):
  ProtocolAnalyzer       - top-level Phase 6 orchestrator → Phase6Result
  ProtocolFsmBuilder     - four-pass FSM construction from Phase 3/4/5 artifacts
  BoundaryTransitionEngine - DFS path extraction targeting boundary states/transitions
  BoundaryStressEngine   - scenario generation via 14 MutationStrategies
  ScenarioReplayer       - model-level scenario replay + failure/mismatch detection
  StateCoverageTracker   - per-FSM state/transition coverage accumulator
  ProtocolFsm            - explicit FSM (states, transitions, guards, counters, timers)
  ProtocolFsmState       - FSM state node with counters, timers, layer, boundary flag
  ProtocolTransition     - FSM edge with guards, counter increments, layer, boundary flag
  TransitionGuard        - guard predicate with declared boundary_value for stress targeting
  FsmCounter             - numeric counter with max_value for overflow/wrap stress
  FsmTimer               - timer slot with timeout_ms for expiry-boundary stress
  StressScenario         - valid state-trace targeting a specific boundary condition
  StateStep              - single step in a scenario trace with full counter/timer snapshot
  BoundaryFailure        - anomaly observed at a boundary with state-path provenance
  InterpretationMismatch - layer-level disagreement at a shared state boundary
  FailureAtBoundaryEntry - triad: state path → inconsistent transition → anomaly
  FailureAtBoundaryDataset - complete failure dataset for one target
  StateCoverageReport    - which states/transitions were exercised + coverage ratios
  ScenarioLibrary        - indexed collection of StressScenarios by BoundaryKind
  BoundaryKind           - taxonomy of 14 boundary condition categories
  MutationStrategy       - 14 mutation strategies for scenario generation
  ProtocolFamily         - 7 protocol families (TLS/QUIC, HTTP, OAuth, etc.)
  TransitionInterpretation - 7 interpretation layers (PARSER/HANDLER/CRYPTO/…)
  ScenarioResult         - outcome of executing a scenario (BOUNDARY_FAILURE/DESYNC/…)
  Phase6Result           - top-level output with exit criterion

Phase 3 Binary Analysis + Fragment Catalog (Static Understanding):
  StaticAnalyzer         - top-level Phase 3 orchestrator → Phase3Result
  CfgBuilder             - recovers StaticCFG from source files or compiled binaries
  FragmentCatalogBuilder - extracts + indexes CodeFragments from a StaticCFG
  TraceMapper            - maps runtime ControlFlowTraces into the static CFG
  StaticCFG              - complete inter-procedural control-flow graph
  StaticFunction         - single function with basic blocks + call edges
  BasicBlock             - maximal straight-line instruction sequence
  ControlEdge            - CFG edge (direct call / indirect / branch / return)
  EdgeKind               - enum of CFG edge types
  FragmentCatalog        - indexed catalog of reusable code fragments
  CodeFragment           - single reusable instruction sequence / call chain
  FragmentSemantics      - semantic category (MEMORY_WRITE, INDIRECT_BRANCH, …)
  AnalysisBackend        - PYTHON_AST | REGEX | CAPSTONE | ANGR | STUB
  ExecutionAtlas         - per-target picture (CFG + catalog + hot paths + FA regions)
  HotPath                - frequently-executed block sequence from normal runs
  FailureAdjacentRegion  - subgraph only exercised in failure/crash runs
  TraceStaticMapping     - runtime trace → static CFG mapping for one run
  TracedBlockMapping     - single runtime bb_id → static BasicBlock mapping
  Phase3Result           - top-level Phase 3 output (atlas + aggregate stats)

Standalone Observer:
  observer/                       - eBPF network event observer (sidecar container)
  observer/observer.py            - BCC-based daemon printing PID/comm/flow to stdout
  observer/bpftrace_probes.bt     - bpftrace quick-validation probes
  observer/docker-compose.observer.yml - standalone compose for the observer sidecar
"""

from systems.simula.inspector.advanced import (
    AttackPathAnalyzer,
    AutonomousPatchingOrchestrator,
    ContinuousHuntingScheduler,
    GitHubPRConfig,
    GitHubPRResult,
    HuntScheduleRun,
    LanguageType,
    MultiLanguageSurfaceDetector,
    ScheduledHuntConfig,
    VulnerabilitySequence,
)
from systems.simula.inspector.analytics import (
    InspectorAnalyticsEmitter,
    InspectorAnalyticsStore,
    InspectorAnalyticsView,
    InspectorEvent,
)
from systems.simula.inspector.boundary_stress_engine import (
    BoundaryStressEngine,
    ScenarioReplayer,
    StateCoverageTracker,
)
from systems.simula.inspector.cfg_builder import CfgBuilder
from systems.simula.inspector.constraint_solver import ConstraintEngine

# Phase 4: Constraint Reasoning + Steerability Model
from systems.simula.inspector.constraint_types import (
    ConditionSet,
    Constraint,
    ConstraintKind,
    ConstraintSet,
    Invariant,
    InvariantSet,
    InvariantStrength,
    Phase4Result,
    StateVariable,
    StateVariableKind,
    SteerabilityClass,
    SteerabilityModel,
    SteerableRegion,
    TransitionExplanation,
    ViolationMechanism,
)
from systems.simula.inspector.control_integrity import (
    ControlIntegrityScorer,
    RuntimeInstrumentationEngine,
)
from systems.simula.inspector.cross_service_tracer import CrossServiceTracer
from systems.simula.inspector.detonation import LiveDetonationChamber
from systems.simula.inspector.ebpf_programs import BpfProgramType
from systems.simula.inspector.fault_classifier import FaultClassifier
from systems.simula.inspector.fragment_catalog import FragmentCatalogBuilder
from systems.simula.inspector.inference import InvariantInferencer
from systems.simula.inspector.ingestor import TargetIngestor
from systems.simula.inspector.propagation_engine import PropagationEngine
from systems.simula.inspector.protocol_analyzer import ProtocolAnalyzer
from systems.simula.inspector.protocol_state_machine import (
    BoundaryTransitionEngine,
    ProtocolFsmBuilder,
)

# Phase 6: Protocol State-Machine Stress
from systems.simula.inspector.protocol_types import (
    BoundaryFailure,
    BoundaryKind,
    FailureAtBoundaryDataset,
    FailureAtBoundaryEntry,
    FsmCounter,
    FsmTimer,
    InterpretationMismatch,
    MutationStrategy,
    Phase6Result,
    ProtocolFamily,
    ProtocolFsm,
    ProtocolFsmState,
    ProtocolTransition,
    ScenarioLibrary,
    ScenarioResult,
    StateCoverageReport,
    StateStep,
    StressScenario,
    TransitionCoverageRecord,
    TransitionGuard,
    TransitionInterpretation,
)
from systems.simula.inspector.prover import VulnerabilityProver
from systems.simula.inspector.remediation import InspectorRepairOrchestrator, RepairAgent
from systems.simula.inspector.runtime_tracer import RuntimeTracer

# Phase 2: Runtime Behaviour Instrumentation
from systems.simula.inspector.runtime_types import (
    BasicBlockTrace,
    ControlFlowTrace,
    ControlIntegrityScore,
    FaultClass,
    FaultClassificationReport,
    FaultObservation,
    InfluencePermissiveTransition,
    Phase2Result,
    RunCategory,
    TraceDataset,
    TraceEvent,
    TraceEventKind,
)
from systems.simula.inspector.safety import InspectorSafetyGates, SafetyResult
from systems.simula.inspector.scope import BountyScope, ScopeEnforcer
from systems.simula.inspector.service import InspectorService
from systems.simula.inspector.slicer import SemanticSlicer
from systems.simula.inspector.state_model import StateModelExtractor
from systems.simula.inspector.static_analyzer import StaticAnalyzer

# Phase 3: Binary Analysis + Fragment Catalog
from systems.simula.inspector.static_types import (
    AnalysisBackend,
    BasicBlock,
    CodeFragment,
    ControlEdge,
    EdgeKind,
    ExecutionAtlas,
    FailureAdjacentRegion,
    FragmentCatalog,
    FragmentSemantics,
    HotPath,
    Phase3Result,
    StaticCFG,
    StaticFunction,
    TracedBlockMapping,
    TraceStaticMapping,
)
from systems.simula.inspector.steerability_analyzer import SteerabilityAnalyzer
from systems.simula.inspector.taint_client import TaintCollectorClient
from systems.simula.inspector.taint_flow_linker import (
    TaintChainGraph,
    TaintEdge,
    TaintEntryPoint,
    TaintFlowLinker,
    TaintRegistry,
    extract_tokens_from_payload,
)
from systems.simula.inspector.taint_types import (
    # Phase 1: Cross-Layer Observability Substrate
    CorrelationContext,
    # Existing taint types
    CrossServiceAttackSurface,
    FlowType,
    InteractionEdge,
    InteractionGraph,
    KernelEvent,
    KernelEventType,
    SinkType,
    TaintCollectorStatus,
    TaintFlow,
    TaintGraph,
    TaintGraphNode,
    TaintLevel,
    TaintSink,
    TaintSource,
)
from systems.simula.inspector.topology import TopologyContext, TopologyDetonationChamber
from systems.simula.inspector.trace_mapper import TraceMapper
from systems.simula.inspector.tracer import CallGraphTracer
from systems.simula.inspector.trust_analyzer import TrustAnalyzer
from systems.simula.inspector.trust_graph import TrustGraphBuilder

# Phase 5: Trust-Graph Influence Expansion
from systems.simula.inspector.trust_types import (
    CorridorRiskTier,
    ExpansionCorridor,
    FootholdBinding,
    Phase5Result,
    PrivilegeImpact,
    PropagationPath,
    PropagationSimulation,
    PropagationStep,
    ReachabilityMap,
    TrustEdge,
    TrustEdgeKind,
    TrustGraph,
    TrustNode,
    TrustNodeKind,
    TrustStrength,
)
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    InspectionResult,
    InspectorConfig,
    RemediationAttempt,
    RemediationResult,
    RemediationStatus,
    TargetType,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from systems.simula.inspector.workspace import TargetWorkspace

__all__ = [
    # Core (Phases 1-11)
    "LiveDetonationChamber",
    "TopologyContext",
    "TopologyDetonationChamber",
    "InspectorService",
    "BountyScope",
    "InvariantInferencer",
    "ScopeEnforcer",
    "SemanticSlicer",
    "CallGraphTracer",
    "InspectorSafetyGates",
    "SafetyResult",
    "TargetWorkspace",
    "TargetIngestor",
    "VulnerabilityProver",
    "InspectorRepairOrchestrator",
    "RepairAgent",
    "InspectorAnalyticsEmitter",
    "InspectorAnalyticsView",
    "InspectorAnalyticsStore",
    "InspectorEvent",
    "TargetType",
    "AttackSurface",
    "AttackSurfaceType",
    "VulnerabilityClass",
    "VulnerabilityReport",
    "VulnerabilitySeverity",
    "InspectionResult",
    "InspectorConfig",
    "RemediationStatus",
    "RemediationAttempt",
    "RemediationResult",
    # eBPF Taint Tracking
    "BpfProgramType",
    "CrossServiceAttackSurface",
    "CrossServiceTracer",
    "FlowType",
    "SinkType",
    "TaintCollectorClient",
    "TaintCollectorStatus",
    "TaintFlow",
    "TaintGraph",
    "TaintGraphNode",
    "TaintLevel",
    "TaintSink",
    "TaintSource",
    # Flow Linker (user-space token-level chain tracking)
    "TaintChainGraph",
    "TaintEdge",
    "TaintEntryPoint",
    "TaintFlowLinker",
    "TaintRegistry",
    "extract_tokens_from_payload",
    # Phase 1: Cross-Layer Observability Substrate
    "CorrelationContext",
    "InteractionEdge",
    "InteractionGraph",
    "KernelEvent",
    "KernelEventType",
    # Advanced (Phase 12)
    "LanguageType",
    "MultiLanguageSurfaceDetector",
    "VulnerabilitySequence",
    "AttackPathAnalyzer",
    "AutonomousPatchingOrchestrator",
    "GitHubPRConfig",
    "GitHubPRResult",
    "ContinuousHuntingScheduler",
    "ScheduledHuntConfig",
    "HuntScheduleRun",
    # Phase 2: Runtime Behaviour Instrumentation
    "RuntimeTracer",
    "FaultClassifier",
    "ControlIntegrityScorer",
    "RuntimeInstrumentationEngine",
    "TraceEvent",
    "TraceEventKind",
    "FaultClass",
    "RunCategory",
    "ControlFlowTrace",
    "BasicBlockTrace",
    "FaultObservation",
    "TraceDataset",
    "ControlIntegrityScore",
    "InfluencePermissiveTransition",
    "FaultClassificationReport",
    "Phase2Result",
    # Phase 3: Binary Analysis + Fragment Catalog
    "AnalysisBackend",
    "BasicBlock",
    "CfgBuilder",
    "CodeFragment",
    "ControlEdge",
    "EdgeKind",
    "ExecutionAtlas",
    "FailureAdjacentRegion",
    "FragmentCatalog",
    "FragmentCatalogBuilder",
    "FragmentSemantics",
    "HotPath",
    "Phase3Result",
    "StaticAnalyzer",
    "StaticCFG",
    "StaticFunction",
    "TraceMapper",
    "TraceStaticMapping",
    "TracedBlockMapping",
    # Phase 4: Constraint Reasoning + Steerability
    "ConditionSet",
    "Constraint",
    "ConstraintEngine",
    "ConstraintKind",
    "ConstraintSet",
    "Invariant",
    "InvariantSet",
    "InvariantStrength",
    "Phase4Result",
    "SteerabilityAnalyzer",
    "SteerabilityClass",
    "SteerabilityModel",
    "SteerableRegion",
    "StateModelExtractor",
    "StateVariable",
    "StateVariableKind",
    "TransitionExplanation",
    "ViolationMechanism",
    # Phase 5: Trust-Graph Influence Expansion
    "TrustAnalyzer",
    "TrustGraphBuilder",
    "PropagationEngine",
    "TrustGraph",
    "TrustNode",
    "TrustEdge",
    "TrustNodeKind",
    "TrustEdgeKind",
    "TrustStrength",
    "PrivilegeImpact",
    "CorridorRiskTier",
    "FootholdBinding",
    "PropagationPath",
    "PropagationStep",
    "PropagationSimulation",
    "ReachabilityMap",
    "ExpansionCorridor",
    "Phase5Result",
    # Phase 6: Protocol State-Machine Stress
    "ProtocolAnalyzer",
    "ProtocolFsmBuilder",
    "BoundaryTransitionEngine",
    "BoundaryStressEngine",
    "ScenarioReplayer",
    "StateCoverageTracker",
    "BoundaryFailure",
    "BoundaryKind",
    "FailureAtBoundaryDataset",
    "FailureAtBoundaryEntry",
    "FsmCounter",
    "FsmTimer",
    "InterpretationMismatch",
    "MutationStrategy",
    "Phase6Result",
    "ProtocolFamily",
    "ProtocolFsm",
    "ProtocolFsmState",
    "ProtocolTransition",
    "ScenarioLibrary",
    "ScenarioResult",
    "StateStep",
    "StateCoverageReport",
    "StressScenario",
    "TransitionCoverageRecord",
    "TransitionGuard",
    "TransitionInterpretation",
]
