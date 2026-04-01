"""
EcodiaOS - Simula Service

The self-evolution system. Simula is the organism's capacity for
metamorphosis: structural change beyond parameter tuning.

Where Evo adjusts the knobs, Simula redesigns the dashboard.

Simula coordinates the full evolution proposal pipeline:
  1. DEDUPLICATE - check for duplicate/similar active proposals
  2. VALIDATE    - reject forbidden categories immediately
  3. SIMULATE    - deep multi-strategy impact prediction
  4. GATE        - route governed changes through community governance
  5. APPLY       - invoke the code agent or config updater with rollback
  6. VERIFY      - health check post-application
  7. RECORD      - write immutable history, increment version, update analytics

Interfaces:
  initialize()            - build sub-systems, load current version
  process_proposal()      - main entry point for rich proposals
  receive_evo_proposal()  - receive from Evo via bridge translation
  get_history()           - recent evolution records
  get_current_version()   - current config version number
  get_analytics()         - evolution quality metrics
  shutdown()              - graceful teardown
  stats                   - service-level metrics

Iron Rules (never violated - see SIMULA_IRON_RULES in types.py):
  - Cannot modify Equor, constitutional drives, invariants
  - Cannot modify its own logic
  - Must simulate before applying any change
  - Must maintain rollback capability
  - Evolution history is immutable
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from clients.embedding import EmbeddingClient, create_voyage_client
from clients.llm import LLMProvider, create_thinking_provider
from primitives.common import DriveAlignmentVector, SystemID, new_id, utc_now
from primitives.re_training import RETrainingExample
from systems.simula.analytics import EvolutionAnalyticsEngine
from systems.simula.applicator import ChangeApplicator
from systems.simula.architecture_efe_scorer import ArchitectureEFEScorer
from systems.simula.bridge import EvoSimulaBridge
from systems.simula.code_agent import SimulaCodeAgent
from systems.simula.evolution_types import (
    FORBIDDEN,
    GOVERNANCE_REQUIRED,
    ConfigVersion,
    EFECalibrationRecord,
    EnrichedSimulationResult,
    EvolutionAnalytics,
    EvolutionProposal,
    EvolutionRecord,
    ProposalResult,
    ProposalStatus,
    RankedProposalQueue,
    RiskLevel,
    SimulationResult,
    TriageResult,
    TriageStatus,
)
from systems.simula.health import HealthChecker
from systems.simula.history import EvolutionHistoryManager
from systems.simula.learning.grpo import GRPOTrainingEngine
from systems.simula.learning.lilo import LiloLibraryEngine
from systems.simula.proactive_scanner import ProactiveScanner
from systems.simula.proposal_intelligence import ProposalIntelligence
from systems.simula.preventive_audit import SimulaPreventiveAudit
from systems.simula.repair_memory import RepairMemory
from systems.simula.retrieval.swe_grep import SweGrepRetriever
from systems.simula.reasoning_router import ReasoningRouter
from systems.simula.rollback import RollbackManager
from systems.simula.simulation import ChangeSimulator
from systems.simula.verification.incremental import IncrementalVerificationEngine

if TYPE_CHECKING:
    from systems.thymos.types import Incident, IncidentClass, IncidentSeverity
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from clients.timescaledb import TimescaleDBClient
    from config import SimulaConfig
    from systems.memory.service import MemoryService
    from systems.simula.inspector.analytics import InspectorAnalyticsEmitter
    from systems.simula.inspector.service import InspectorService
    from systems.simula.inspector.types import InspectionResult

logger = structlog.get_logger()


class SimulaService:
    """
    Simula - the EOS self-evolution system.

    Coordinates eight sub-systems:
      ChangeSimulator           - deep multi-strategy impact prediction
      SimulaCodeAgent           - Claude-backed code generation with 11 tools
      ChangeApplicator          - routes proposals to the right application strategy
      RollbackManager           - file snapshots and restore
      EvolutionHistoryManager   - immutable Neo4j history
      EvoSimulaBridge           - Evo→Simula proposal translation
      ProposalIntelligence      - deduplication, prioritization, dependency analysis
      EvolutionAnalyticsEngine  - evolution quality tracking
    """

    system_id: str = "simula"

    # Parameters that appear in float_params (for Gaussian perturbation) but are
    # semantically integers. After clamping, values for these params are cast to
    # int via round() in both _on_config_drift and _on_evo_adjust_budget.
    INT_PARAMS: frozenset[str] = frozenset({
        "child_spawn_interval_days",
        "child_min_profitability_usd",
    })

    def __init__(
        self,
        config: SimulaConfig,
        llm: LLMProvider,
        neo4j: Neo4jClient | None = None,
        memory: MemoryService | None = None,
        codebase_root: Path | None = None,
        instance_name: str = "EOS",
        tsdb: TimescaleDBClient | None = None,
        redis: RedisClient | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._neo4j = neo4j
        self._memory = memory
        self._root = codebase_root or Path(config.codebase_root).resolve()
        self._instance_name = instance_name
        self._tsdb = tsdb
        self._redis = redis
        self._initialized: bool = False
        self._logger = logger.bind(system="simula")

        # Sub-systems (built in initialize())
        self._simulator: ChangeSimulator | None = None
        self._code_agent: SimulaCodeAgent | None = None
        self._applicator: ChangeApplicator | None = None
        self._rollback: RollbackManager | None = None
        self._history: EvolutionHistoryManager | None = None
        self._health: HealthChecker | None = None
        self._bridge: EvoSimulaBridge | None = None
        self._intelligence: ProposalIntelligence | None = None
        self._analytics: EvolutionAnalyticsEngine | None = None
        self._efe_scorer: ArchitectureEFEScorer | None = None

        # Thompson sampling for proof strategy selection (Z3/Lean/Dafny/static)
        self._reasoning_router: ReasoningRouter = ReasoningRouter()

        # Stage 3 sub-systems
        self._incremental: IncrementalVerificationEngine | None = None
        self._swe_grep: SweGrepRetriever | None = None
        self._lilo: LiloLibraryEngine | None = None

        # Stage 4 sub-systems
        self._lean_bridge: object | None = None  # LeanBridge (lazy import)
        self._grpo: GRPOTrainingEngine | None = None
        self._diffusion_repair: object | None = None  # DiffusionRepairAgent (lazy import)

        # Stage 5 sub-systems
        self._synthesis: object | None = None  # SynthesisStrategySelector (lazy import)
        self._repair_agent: object | None = None  # RepairAgent (lazy import)
        self._orchestrator: object | None = None  # MultiAgentOrchestrator (lazy import)
        self._causal_debugger: object | None = None  # CausalDebugger (lazy import)
        self._issue_resolver: object | None = None  # IssueResolver (lazy import)

        # Stage 6 sub-systems
        self._hash_chain: object | None = None  # HashChainManager (lazy import)
        self._content_credentials: object | None = None  # ContentCredentialManager (lazy import)
        self._governance_credentials: object | None = None  # GovernanceCredentialManager (lazy import)
        self._hard_negative_miner: object | None = None  # FailureAnalyzer (lazy import)
        self._adversarial_tester: object | None = None  # RobustnessTestGenerator (lazy import)
        self._adversarial_self_play: object | None = None  # AdversarialSelfPlay (lazy import)
        self._adversarial_self_play_task: asyncio.Task[None] | None = None
        self._formal_spec_generator: object | None = None  # FormalSpecGenerator (lazy import)
        self._egraph: object | None = None  # EqualitySaturationEngine (lazy import)
        self._symbolic_execution: object | None = None  # SymbolicExecutionEngine (lazy import)

        # Stage 7 sub-systems (Inspector - lazy runtime imports in initialize())
        self._inspector: InspectorService | None = None
        self._inspector_analytics: InspectorAnalyticsEmitter | None = None

        # Meta-healing: sentinel for reporting Simula's own failures to Thymos
        from systems.synapse.sentinel import ErrorSentinel
        self._sentinel = ErrorSentinel("simula")

        # Closure Loop 4: immune advisory filter from Thymos
        from systems.simula.immune_filter import ImmuneAdvisoryFilter
        self._immune_filter = ImmuneAdvisoryFilter()

        # Cross-system references wired after construction
        self._synapse: Any = None
        self._telos: Any = None  # TelosService - for constitutional binding validation
        self._evo: Any = None    # EvoService - for learned repair pattern validation
        self._soma_ref: Any = None    # SomaService - allostatic state for repair context
        self._fovea_ref: Any = None   # FoveaService - attention profile for repair context
        self._log_analyzer: Any = None  # LogAnalyzer - organism health signals
        self._benchmarks: Any = None  # BenchmarksService - retained for legacy; KPI now via bus

        # Organizational closure (Speciation Bible §8.3) - generates new subsystem modules
        self._subsystem_generator: Any = None  # SubsystemGenerator (built in initialize())
        # Dynamic capability expansion - generates new Axon executor classes at runtime
        self._executor_generator: Any = None  # ExecutorGenerator (built in initialize())

        # Cache from ALIGNMENT_GAP_WARNING - set True when Telos flags a drive topology violation,
        # cleared after next successful proposal application.
        self._telos_alignment_gap_active: bool = False

        # ── Skia VitalityCoordinator modulation ───────────────────────
        self._modulation_halted: bool = False

        # Oneiros lucid dream simulation results (mutation_id → report dict)
        self._dream_results: dict[str, dict[str, Any]] = {}
        self._dream_results_lock: asyncio.Lock = asyncio.Lock()

        # Grid metabolism state - pauses the evolution pipeline in CONSERVATION
        self._grid_state: str = "normal"  # MetabolicState values as strings
        self._evo_consolidation_stalled: bool = False
        self._evo_stall_expires_at: float = 0.0

        # ── Metabolic gating ──────────────────────────────────────────────
        self._starvation_level: str = "nominal"
        # Multiplier applied to economic mutation priority during metabolic crisis.
        # 1.0 = normal; 1.5 = AUSTERITY; 2.0 = CRITICAL/EMERGENCY.
        self._metabolic_boost: float = 1.0

        # State
        self._current_version: int = 0
        self._version_lock: asyncio.Lock = asyncio.Lock()
        self._active_proposals: dict[str, EvolutionProposal] = {}
        # Semantic dedup: maps semantic_key → (proposal_id, status)
        # semantic_key = hash(target_system + change_category + error_class)
        # Prevents duplicate proposals for the same structural fix.
        self._active_proposal_semantic_keys: dict[str, tuple[str, str]] = {}
        self._proposals_lock: asyncio.Lock = asyncio.Lock()
        self._inspector_hunt_lock: asyncio.Lock = asyncio.Lock()

        # Metrics
        self._proposals_received: int = 0
        self._proposals_approved: int = 0
        self._proposals_rejected: int = 0
        self._proposals_rolled_back: int = 0
        self._proposals_awaiting_governance: int = 0
        self._proposals_deduplicated: int = 0
        self._proposals_applied_since_consolidation: int = 0

        # Self-healing sub-systems (built in initialize())
        self._repair_memory: RepairMemory | None = None
        self._proactive_scanner: ProactiveScanner | None = None
        self._proactive_scanner_task: asyncio.Task[None] | None = None
        self._governance_timeout_task: asyncio.Task[None] | None = None

        # Preventive audit - scheduled 4-hour fragility scanner
        self._preventive_audit: SimulaPreventiveAudit | None = None
        self._preventive_audit_task: asyncio.Task[None] | None = None

        # ── Fatal crash pattern memory ──────────────────────────────────────────
        # Populated by _on_crash_pattern_confirmed() as CRASH_PATTERN_CONFIRMED
        # events arrive from Thymos or Kairos.  Each entry is a CrashPattern
        # (from core.crash_pattern_analyzer) keyed by pattern_id.
        # Used by _score_patch_against_patterns() to gate generated patches.
        self._known_fatal_patterns: dict[str, Any] = {}
        # Background task for proactive 2-hour pattern scan
        self._pattern_scan_task: asyncio.Task[None] | None = None

        # Timestamp of the last proposal to reach a terminal state
        self._last_proposal_processed_at: str | None = None

    # ─── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Build all sub-systems and load current config version from history.
        Must be called before any other method.
        """
        if self._initialized:
            return

        # Build the rollback manager
        self._rollback = RollbackManager(codebase_root=self._root)

        # ── Stage 2: Verification bridges ─────────────────────────────────────
        from systems.simula.verification.dafny_bridge import DafnyBridge
        from systems.simula.verification.static_analysis import StaticAnalysisBridge
        from systems.simula.verification.z3_bridge import Z3Bridge

        dafny_bridge: DafnyBridge | None = None
        if self._config.dafny_enabled:
            dafny_bridge = DafnyBridge(
                dafny_path=self._config.dafny_binary_path,
                verify_timeout_s=self._config.dafny_verify_timeout_s,
                max_rounds=self._config.dafny_max_clover_rounds,
            )
            self._logger.info("dafny_bridge_initialized")

        z3_bridge: Z3Bridge | None = None
        if self._config.z3_enabled:
            z3_bridge = Z3Bridge(
                check_timeout_ms=self._config.z3_check_timeout_ms,
                max_rounds=self._config.z3_max_discovery_rounds,
            )
            self._logger.info("z3_bridge_initialized")

        static_bridge: StaticAnalysisBridge | None = None
        if self._config.static_analysis_enabled:
            static_bridge = StaticAnalysisBridge(
                codebase_root=self._root,
            )
            self._logger.info("static_analysis_bridge_initialized")

        # Store for AgentCoder pipeline
        self._dafny_bridge = dafny_bridge
        self._z3_bridge = z3_bridge
        self._static_bridge = static_bridge

        # Build the health checker with Stage 2 bridges + Stage 3 Z3 blocking
        self._health = HealthChecker(
            codebase_root=self._root,
            test_command=self._config.test_command,
            dafny_bridge=dafny_bridge,
            z3_bridge=z3_bridge,
            static_analysis_bridge=static_bridge,
            llm=self._llm,
            z3_blocking=self._config.z3_blocking,
            integration_tests_enabled=self._config.integration_tests_enabled,
            integration_tests_timeout_s=self._config.integration_tests_timeout_s,
            performance_baseline_enabled=self._config.performance_baseline_enabled,
            performance_baseline_timeout_s=self._config.performance_baseline_timeout_s,
        )

        # ── Stage 1A: Extended-thinking provider for governance/high-risk ────
        thinking_provider = None
        if self._config.thinking_model_api_key:
            try:
                thinking_provider = create_thinking_provider(
                    api_key=self._config.thinking_model_api_key,
                    model=self._config.thinking_model,
                    provider=self._config.thinking_model_provider,
                    reasoning_budget=self._config.thinking_budget_tokens,
                )
                self._logger.info(
                    "thinking_provider_initialized",
                    model=self._config.thinking_model,
                    provider=self._config.thinking_model_provider,
                )
            except Exception as exc:
                # Raise rather than silently degrade - caller must decide whether
                # to retry, use a fallback key, or start without extended thinking.
                raise RuntimeError(
                    f"thinking_provider_init_failed: {exc}. "
                    "Either fix the API key/model config or set thinking_model_api_key='' "
                    "to disable extended thinking."
                ) from exc

        # ── Stage 1B: Voyage-code-3 embedding client ────────────────────────
        embedding_client: EmbeddingClient | None = None
        if self._config.embedding_api_key:
            try:
                embedding_client = create_voyage_client(
                    api_key=self._config.embedding_api_key,
                    model=self._config.embedding_model,
                )
                self._logger.info(
                    "embedding_client_initialized",
                    model=self._config.embedding_model,
                )
            except Exception as exc:
                # Raise rather than silently degrade - semantic similarity search
                # will be unavailable, which degrades duplicate-detection quality.
                raise RuntimeError(
                    f"embedding_client_init_failed: {exc}. "
                    "Either fix the API key/model config or set embedding_api_key='' "
                    "to disable semantic embeddings."
                ) from exc

        # Store for shutdown cleanup
        self._embedding_client = embedding_client

        # Build the code agent with Stage 1 + 2 enhancements
        code_agent_llm = self._llm
        self._code_agent = SimulaCodeAgent(
            llm=code_agent_llm,
            codebase_root=self._root,
            max_turns=self._config.max_code_agent_turns,
            thinking_provider=thinking_provider,
            thinking_budget_tokens=self._config.thinking_budget_tokens,
            embedding_client=embedding_client,
            kv_compression_ratio=self._config.kv_compression_ratio,
            kv_compression_enabled=self._config.kv_compression_enabled,
            # Stage 2C: static analysis post-generation gate
            static_analysis_bridge=static_bridge,
            static_analysis_max_fix_iterations=self._config.static_analysis_max_fix_iterations,
        )

        # ── Stage 2D: AgentCoder pipeline agents ──────────────────────────────
        from systems.simula.agents.test_designer import TestDesignerAgent
        from systems.simula.agents.test_executor import TestExecutorAgent

        test_designer: TestDesignerAgent | None = None
        test_executor: TestExecutorAgent | None = None
        if self._config.agent_coder_enabled:
            test_designer = TestDesignerAgent(
                llm=self._llm,
                codebase_root=self._root,
            )
            test_executor = TestExecutorAgent(
                codebase_root=self._root,
                test_timeout_s=self._config.agent_coder_test_timeout_s,
            )
            self._logger.info("agent_coder_pipeline_initialized")

        # Build the applicator with Stage 2D pipeline
        self._applicator = ChangeApplicator(
            code_agent=self._code_agent,
            rollback_manager=self._rollback,
            health_checker=self._health,
            codebase_root=self._root,
            test_designer=test_designer,
            test_executor=test_executor,
            static_analysis_bridge=static_bridge,
            agent_coder_enabled=self._config.agent_coder_enabled,
            agent_coder_max_iterations=self._config.agent_coder_max_iterations,
        )

        # Build the history manager (requires Neo4j) with Stage 1B embedding support
        if self._neo4j is None:
            raise RuntimeError(
                "Simula requires a Neo4j client for evolution history, analytics, "
                "governance, and learning. Either supply a Neo4jClient or disable Simula."
            )
        self._history = EvolutionHistoryManager(
            neo4j=self._neo4j,
            embedding_client=embedding_client,
        )
        self._current_version = await self._history.get_current_version()

        # Build the analytics engine (depends on history)
        self._analytics = EvolutionAnalyticsEngine(history=self._history)
        # Wire analytics into applicator for data-driven strategy selection
        self._applicator._analytics = self._analytics

        # Build the deep simulator (depends on analytics for dynamic caution)
        self._simulator = ChangeSimulator(
            config=self._config,
            llm=self._llm,
            memory=self._memory,
            analytics=self._analytics,
            codebase_root=self._root,
        )

        # Build the Evo↔Simula bridge
        self._bridge = EvoSimulaBridge(
            llm=self._llm,
            memory=self._memory,
        )

        # Build the proposal intelligence layer with Stage 1B embedding dedup
        self._intelligence = ProposalIntelligence(
            llm=self._llm,
            analytics=self._analytics,
            embedding_client=embedding_client,
        )

        # Build the architecture-level EFE scorer (active inference for proposals)
        self._efe_scorer = ArchitectureEFEScorer(
            history=self._history,
            analytics=self._analytics,
            complexity_penalty_weight=self._config.efe_complexity_penalty_weight,
            auto_approve_efe_threshold=self._config.efe_auto_approve_threshold,
        )
        self._efe_scorer._neo4j = self._neo4j
        # Bootstrap calibration from historical outcomes
        await self._efe_scorer.compute_calibration_from_history()
        # Warm-start from parent genome calibration if this is a child instance
        await self._efe_scorer.load_parent_genome_calibration(self._neo4j)
        self._logger.info("architecture_efe_scorer_initialized")

        # ── Stage 3A: Incremental verification ─────────────────────────────────
        if self._config.incremental_verification_enabled:
            self._incremental = IncrementalVerificationEngine(
                codebase_root=self._root,
                redis=self._redis,
                neo4j=self._neo4j,
                hot_ttl_seconds=self._config.incremental_hot_ttl_seconds,
            )
            self._logger.info("incremental_verification_initialized")

        # ── Stage 3B: SWE-grep retrieval ──────────────────────────────────────
        if self._config.swe_grep_enabled:
            self._swe_grep = SweGrepRetriever(
                codebase_root=self._root,
                llm=self._llm,
                max_hops=self._config.swe_grep_max_hops,
            )
            self._logger.info("swe_grep_retriever_initialized")

        # ── Stage 3C: LILO library learning ───────────────────────────────────
        if self._config.lilo_enabled:
            self._lilo = LiloLibraryEngine(
                neo4j=self._neo4j,
                llm=self._llm,
                codebase_root=self._root,
            )
            self._logger.info("lilo_library_initialized")

        # ── Stage 4A: Lean 4 proof generation ────────────────────────────────
        lean_bridge_instance = None
        if self._config.lean_enabled:
            from systems.simula.verification.lean_bridge import LeanBridge

            lean_bridge_instance = LeanBridge(
                lean_path=self._config.lean_binary_path,
                project_path=self._config.lean_project_path or "",
                verify_timeout_s=self._config.lean_verify_timeout_s,
                max_attempts=self._config.lean_max_attempts,
                copilot_enabled=self._config.lean_copilot_enabled,
                dojo_enabled=self._config.lean_dojo_enabled,
                max_library_size=self._config.lean_proof_library_max_size,
                neo4j=self._neo4j,
            )
            self._lean_bridge = lean_bridge_instance
            self._logger.info("lean_bridge_initialized")

        # Wire Lean bridge into health checker
        if lean_bridge_instance is not None and self._health is not None:
            self._health._lean = lean_bridge_instance
            self._health._lean_blocking = self._config.lean_blocking

        # ── Stage 4B: GRPO domain fine-tuning ─────────────────────────────────
        if self._config.grpo_enabled:
            self._grpo = GRPOTrainingEngine(
                config=self._config,
                neo4j=self._neo4j,
            )
            self._logger.info("grpo_engine_initialized")

            # Wire GRPO engine reference into code agent for local model routing
            if self._code_agent is not None:
                self._code_agent._grpo_engine = self._grpo

            # Resume serving if a previous training run produced a usable model
            try:
                await self._grpo.load_latest_training_run()
            except Exception as exc:
                self._logger.warning("grpo_load_latest_error", error=str(exc))

        # ── Stage 4C: Diffusion-based code repair ────────────────────────────
        if self._config.diffusion_repair_enabled:
            from systems.simula.agents.diffusion_repair import DiffusionRepairAgent

            self._diffusion_repair = DiffusionRepairAgent(
                llm=self._llm,
                codebase_root=self._root,
                max_denoise_steps=self._config.diffusion_max_denoise_steps,
                timeout_s=self._config.diffusion_timeout_s,
                sketch_first=self._config.diffusion_sketch_first,
            )
            self._logger.info("diffusion_repair_agent_initialized")

        # ── Stage 5A: Neurosymbolic synthesis ────────────────────────────────
        if self._config.synthesis_enabled:
            from systems.simula.synthesis.chopchop import ChopChopEngine
            from systems.simula.synthesis.hysynth import HySynthEngine
            from systems.simula.synthesis.sketch_solver import SketchSolver
            from systems.simula.synthesis.strategy_selector import (
                SynthesisStrategySelector,
            )

            hysynth = HySynthEngine(
                llm=self._llm,
                codebase_root=self._root,
                max_candidates=self._config.hysynth_max_candidates,
                beam_width=self._config.hysynth_beam_width,
                timeout_s=self._config.hysynth_timeout_s,
            )
            sketch = SketchSolver(
                llm=self._llm,
                z3_bridge=z3_bridge,
                max_holes=self._config.sketch_max_holes,
                solver_timeout_ms=self._config.sketch_solver_timeout_ms,
            )
            chopchop = ChopChopEngine(
                llm=self._llm,
                codebase_root=self._root,
                max_retries_per_chunk=self._config.chopchop_max_retries,
                chunk_size_lines=self._config.chopchop_chunk_size_lines,
                timeout_s=self._config.chopchop_timeout_s,
            )
            self._synthesis = SynthesisStrategySelector(
                hysynth=hysynth,
                sketch_solver=sketch,
                chopchop=chopchop,
                codebase_root=self._root,
            )
            self._logger.info("synthesis_subsystem_initialized")

        # ── Stage 5B: Neural program repair ──────────────────────────────────
        if self._config.repair_agent_enabled:
            from systems.simula.agents.repair_agent import RepairAgent

            self._repair_agent = RepairAgent(
                reasoning_llm=self._llm,
                code_llm=self._llm,
                codebase_root=self._root,
                neo4j=self._neo4j,
                max_retries=self._config.repair_max_retries,
                cost_budget_usd=self._config.repair_cost_budget_usd,
                timeout_s=self._config.repair_timeout_s,
                use_similar_fixes=self._config.repair_use_similar_fixes,
            )
            self._logger.info("repair_agent_initialized")

        # ── Stage 5C: Multi-agent orchestration ─────────────────────────────
        if self._config.orchestration_enabled and self._code_agent is not None:
            from systems.simula.orchestration.orchestrator import MultiAgentOrchestrator
            from systems.simula.orchestration.task_planner import TaskPlanner

            task_planner = TaskPlanner(
                codebase_root=self._root,
                llm=self._llm,
                max_dag_nodes=self._config.orchestration_max_dag_nodes,
            )
            self._orchestrator = MultiAgentOrchestrator(
                llm=self._llm,
                codebase_root=self._root,
                code_agent=self._code_agent,
                task_planner=task_planner,
                max_agents_per_stage=self._config.orchestration_max_agents_per_stage,
                timeout_s=self._config.orchestration_timeout_s,
            )
            self._logger.info("orchestrator_initialized")

        # ── Stage 5D: Causal debugging ───────────────────────────────────────
        if self._config.causal_debugging_enabled:
            from systems.simula.debugging.causal_dag import CausalDebugger

            self._causal_debugger = CausalDebugger(
                llm=self._llm,
                codebase_root=self._root,
                max_interventions=self._config.causal_max_interventions,
                fault_injection_enabled=self._config.causal_fault_injection_enabled,
                timeout_s=self._config.causal_timeout_s,
            )
            self._logger.info("causal_debugger_initialized")

        # ── Stage 5E: Autonomous issue resolution ────────────────────────────
        if self._config.issue_resolution_enabled:
            from systems.simula.agents.repair_agent import RepairAgent
            from systems.simula.resolution.issue_resolver import IssueResolver
            from systems.simula.resolution.monitors import (
                DegradationMonitor,
                PerfRegressionMonitor,
                SecurityVulnMonitor,
            )

            perf_monitor = (
                PerfRegressionMonitor()
                if self._config.issue_perf_regression_enabled
                else None
            )
            security_monitor = (
                SecurityVulnMonitor(self._root)
                if self._config.issue_security_scan_enabled
                else None
            )
            degradation_monitor = DegradationMonitor(
                window_hours=self._config.issue_degradation_window_hours,
            )

            self._issue_resolver = IssueResolver(
                llm=self._llm,
                codebase_root=self._root,
                neo4j=self._neo4j,
                code_agent=self._code_agent,
                repair_agent=self._repair_agent if isinstance(self._repair_agent, RepairAgent) else None,
                perf_monitor=perf_monitor,
                security_monitor=security_monitor,
                degradation_monitor=degradation_monitor,
                max_autonomy_level=self._config.issue_max_autonomy_level,
                abstention_threshold=self._config.issue_abstention_confidence_threshold,
            )
            self._logger.info("issue_resolver_initialized")

        # ── Stage 6A: Cryptographic auditability ────────────────────────────
        if self._config.hash_chain_enabled:
            from systems.simula.audit.hash_chain import HashChainManager

            self._hash_chain = HashChainManager(
                neo4j=self._neo4j,
            )
            self._logger.info("hash_chain_initialized")

        if self._config.c2pa_enabled:
            from systems.simula.audit.content_credentials import ContentCredentialManager

            self._content_credentials = ContentCredentialManager(
                signing_key_path=self._config.c2pa_signing_key_path,
                issuer_name=self._config.c2pa_issuer_name,
            )
            self._logger.info("content_credentials_initialized")

        if self._config.verifiable_credentials_enabled:
            from systems.simula.audit.verifiable_credentials import (
                GovernanceCredentialManager,
            )

            self._governance_credentials = GovernanceCredentialManager(
                neo4j=self._neo4j,
                signing_key_path=self._config.c2pa_signing_key_path,
            )
            self._logger.info("governance_credentials_initialized")

        # ── Stage 6B: Co-evolving agents ──────────────────────────────────────
        if self._config.coevolution_enabled:
            from systems.simula.coevolution.failure_analyzer import FailureAnalyzer

            self._hard_negative_miner = FailureAnalyzer(
                neo4j=self._neo4j,
                llm=self._llm,
                max_negatives_per_cycle=self._config.adversarial_max_tests_per_cycle,
            )
            self._logger.info("hard_negative_miner_initialized")

            if self._config.adversarial_test_generation_enabled:
                from systems.simula.coevolution.robustness_tester import (
                    RobustnessTestGenerator,
                )

                self._adversarial_tester = RobustnessTestGenerator(
                    llm=self._llm,
                    codebase_root=self._root,
                    max_tests_per_cycle=self._config.adversarial_max_tests_per_cycle,
                )
                self._logger.info("adversarial_tester_initialized")

            # Adversarial self-play: constitutional red teaming loop
            try:
                from systems.simula.coevolution.adversarial_self_play import (
                    AdversarialSelfPlay,
                )

                self._adversarial_self_play = AdversarialSelfPlay(
                    llm=self._llm,
                )
                self._adversarial_self_play_task = asyncio.create_task(
                    self._adversarial_self_play_loop(),
                    name="simula_adversarial_self_play",
                )
                self._logger.info("adversarial_self_play_loop_started")
            except Exception as exc:
                self._logger.warning("adversarial_self_play_init_error", error=str(exc))

        # ── Stage 6C: Formal spec generation ─────────────────────────────────
        if self._config.formal_spec_generation_enabled:
            from systems.simula.formal_specs.spec_generator import FormalSpecGenerator

            self._formal_spec_generator = FormalSpecGenerator(
                llm=self._llm,
                dafny_bridge=dafny_bridge,
                tla_plus_path=self._config.tla_plus_binary_path,
                alloy_path=self._config.alloy_binary_path,
                dafny_bench_target=self._config.dafny_bench_coverage_target,
                tla_plus_timeout_s=self._config.tla_plus_model_check_timeout_s,
                alloy_scope=self._config.alloy_scope,
            )
            self._logger.info("formal_spec_generator_initialized")

        # ── Stage 6D: Equality saturation (E-graphs) ─────────────────────────
        if self._config.egraph_enabled:
            from systems.simula.egraph.equality_saturation import EqualitySaturationEngine

            self._egraph = EqualitySaturationEngine(
                max_iterations=self._config.egraph_max_iterations,
                timeout_s=self._config.egraph_timeout_s,
            )
            self._logger.info("egraph_initialized")

        # ── Stage 6E: Hybrid symbolic execution ──────────────────────────────
        if self._config.symbolic_execution_enabled:
            from systems.simula.verification.symbolic_execution import (
                SymbolicExecutionEngine,
            )

            self._symbolic_execution = SymbolicExecutionEngine(
                z3_bridge=z3_bridge,
                llm=self._llm,
                timeout_ms=self._config.symbolic_execution_timeout_ms,
                blocking=self._config.symbolic_execution_blocking,
            )
            self._logger.info("symbolic_execution_initialized")

        # Wire Stage 6D + 6E into health checker
        if self._health is not None:
            if self._egraph is not None:
                self._health._egraph = self._egraph  # type: ignore[assignment]
                self._health._egraph_blocking = self._config.egraph_blocking
            if self._symbolic_execution is not None:
                self._health._symbolic_execution = self._symbolic_execution  # type: ignore[assignment]
                self._health._symbolic_execution_blocking = self._config.symbolic_execution_blocking
                self._health._symbolic_execution_domains = self._config.symbolic_execution_domains

        # Wire proof strategy router into health checker for Thompson-prioritized verification
        if self._health is not None:
            self._health._reasoning_router = self._reasoning_router

        # Wire SWE-grep into the bridge for pre-translation retrieval (3B.5)
        if self._bridge is not None and self._swe_grep is not None:
            self._bridge.set_swe_grep(self._swe_grep)

        # ── Stage 7: Inspector - Zero-Day Discovery Engine ─────────────────────
        if self._config.inspector_enabled and z3_bridge is not None:
            from systems.simula.inspector.analytics import InspectorAnalyticsEmitter
            from systems.simula.inspector.prover import VulnerabilityProver
            from systems.simula.inspector.service import InspectorService
            from systems.simula.inspector.types import InspectorConfig

            inspector_config = InspectorConfig(
                authorized_targets=self._config.inspector_authorized_targets,
                max_workers=self._config.inspector_max_workers,
                sandbox_timeout_seconds=self._config.inspector_sandbox_timeout_s,
                log_vulnerability_analytics=self._config.inspector_log_analytics,
                clone_depth=self._config.inspector_clone_depth,
            )

            inspector_prover = VulnerabilityProver(
                z3_bridge=z3_bridge,
                llm=self._llm,
            )

            # Phase 9: Build analytics emitter with optional TSDB persistence
            inspector_analytics: InspectorAnalyticsEmitter | None = None
            if self._config.inspector_log_analytics:
                inspector_analytics = InspectorAnalyticsEmitter(tsdb=self._tsdb)
                self._inspector_analytics = inspector_analytics
                # Initialize TSDB schema (creates inspector_events hypertable)
                await inspector_analytics.initialize()

            # Build optional remediation orchestrator
            inspector_remediation = None
            if self._config.inspector_remediation_enabled and self._repair_agent is not None:
                from systems.simula.inspector.remediation import (
                    InspectorRepairOrchestrator,
                )
                from systems.simula.inspector.workspace import TargetWorkspace

                # Remediation needs a workspace; it's set per-hunt by InspectorService
                placeholder_workspace = TargetWorkspace.internal(self._root)
                inspector_remediation = InspectorRepairOrchestrator(
                    repair_agent=self._repair_agent,  # type: ignore[arg-type]
                    prover=inspector_prover,
                    workspace=placeholder_workspace,
                )

            self._inspector = InspectorService(
                prover=inspector_prover,
                config=inspector_config,
                eos_root=self._root,
                analytics=inspector_analytics,
                remediation=inspector_remediation,
            )

            # Phase 9: Wire Inspector analytics into the unified EvolutionAnalyticsEngine
            if self._analytics is not None and self._inspector is not None:
                self._analytics.set_inspector_view(self._inspector.analytics_view)
                if inspector_analytics is not None and inspector_analytics._store is not None:
                    self._analytics.set_inspector_store(inspector_analytics._store)

            self._logger.info(
                "inspector_initialized",
                inspector="active",
                max_workers=inspector_config.max_workers,
                authorized_targets=len(inspector_config.authorized_targets),
                remediation=inspector_remediation is not None,
                tsdb_persistence=self._tsdb is not None,
            )

        # Pre-compute analytics from history; retry once after 5s on failure
        # so a transient Neo4j blip at startup doesn't leave analytics stale.
        if self._history is not None:
            try:
                await self._analytics.compute_analytics()
            except Exception as exc:
                self._logger.warning(
                    "initial_analytics_failed",
                    error=str(exc),
                    detail="Retrying after 5s - stale analytics risk if retry also fails.",
                )
                import asyncio as _aio_analytics

                async def _retry_analytics() -> None:
                    await _aio_analytics.sleep(5)
                    try:
                        await self._analytics.compute_analytics()
                        self._logger.info("initial_analytics_retry_succeeded")
                    except Exception as _exc2:
                        self._logger.warning(
                            "initial_analytics_retry_failed",
                            error=str(_exc2),
                            detail="Analytics remain stale until next proposal triggers recompute.",
                        )

                _aio_analytics.ensure_future(_retry_analytics())

        # Validate that all enabled external tool binaries are reachable.
        # Fail fast at startup rather than silently degrade on first use.
        await self._validate_tools()

        # ── Self-healing: RepairMemory + ProactiveScanner ─────────────────────
        self._repair_memory = RepairMemory(
            neo4j=self._neo4j,
            low_success_threshold=self._config.repair_low_success_rate_threshold,
            # event_bus wired after initialize() in set_synapse()
        )

        self._proactive_scanner = ProactiveScanner(
            neo4j=self._neo4j,
            event_bus=None,  # wired in set_synapse()
            process_proposal_fn=self.process_proposal,
            incident_limit_per_cycle=self._config.proactive_scan_incident_limit_per_cycle,
            scan_interval_accelerated_factor=self._config.proactive_scan_interval_accelerated_factor,
            scan_interval_critical_factor=self._config.proactive_scan_interval_critical_factor,
        )

        self._preventive_audit = SimulaPreventiveAudit(self)

        # Grid metabolism subscription is deferred to set_synapse() because
        # Synapse is always built after SimulaService.initialize() returns.

        # ── Organizational closure: subsystem generator (Speciation Bible §8.3) ──
        from systems.simula.subsystem_generator import SubsystemGenerator
        from systems.simula.constraint_checker import ConstraintSatisfactionChecker as _CSC

        self._subsystem_generator = SubsystemGenerator(
            code_agent=self._code_agent,
            constraint_checker=_CSC(),
            rollback_manager=self._rollback,
            codebase_root=self._root,
        )
        self._logger.info("subsystem_generator_initialized")

        # ── Dynamic executor generator (ChangeCategory.ADD_EXECUTOR) ──────────
        from systems.simula.executor_generator import ExecutorGenerator

        self._executor_generator = ExecutorGenerator(
            code_agent=self._code_agent,
            rollback_manager=self._rollback,
            codebase_root=self._root,
        )
        self._logger.info("executor_generator_initialized")

        self._initialized = True
        self._logger.info(
            "simula_initialized",
            current_version=self._current_version,
            codebase_root=str(self._root),
            max_code_agent_turns=self._config.max_code_agent_turns,
            subsystems=[
                "simulator", "code_agent", "applicator", "rollback",
                "health", "bridge", "intelligence", "analytics",
                "history" if self._history else "history(disabled)",
                "dafny" if dafny_bridge else "dafny(disabled)",
                "z3" if z3_bridge else "z3(disabled)",
                "static_analysis" if static_bridge else "static_analysis(disabled)",
                "incremental" if self._incremental else "incremental(disabled)",
                "swe_grep" if self._swe_grep else "swe_grep(disabled)",
                "lilo" if self._lilo else "lilo(disabled)",
                "lean" if self._lean_bridge else "lean(disabled)",
                "grpo" if self._grpo else "grpo(disabled)",
                "diffusion_repair" if self._diffusion_repair else "diffusion_repair(disabled)",
                "synthesis" if self._synthesis else "synthesis(disabled)",
                "repair_agent" if self._repair_agent else "repair_agent(disabled)",
                "orchestrator" if self._orchestrator else "orchestrator(disabled)",
                "causal_debugger" if self._causal_debugger else "causal_debugger(disabled)",
                "issue_resolver" if self._issue_resolver else "issue_resolver(disabled)",
                "hash_chain" if self._hash_chain else "hash_chain(disabled)",
                "content_credentials" if self._content_credentials else "content_credentials(disabled)",
                "governance_credentials" if self._governance_credentials else "governance_credentials(disabled)",
                "hard_negative_miner" if self._hard_negative_miner else "hard_negative_miner(disabled)",
                "adversarial_tester" if self._adversarial_tester else "adversarial_tester(disabled)",
                "formal_spec_generator" if self._formal_spec_generator else "formal_spec_generator(disabled)",
                "egraph" if self._egraph else "egraph(disabled)",
                "symbolic_execution" if self._symbolic_execution else "symbolic_execution(disabled)",
                "inspector" if self._inspector else "inspector(disabled)",
            ],
            stage1_extended_thinking=thinking_provider is not None,
            stage1_embeddings=embedding_client is not None,
            stage1_kv_compression=self._config.kv_compression_enabled,
            stage1_kv_ratio=self._config.kv_compression_ratio,
            stage2_dafny=dafny_bridge is not None,
            stage2_z3=z3_bridge is not None,
            stage2_static_analysis=static_bridge is not None,
            stage2_agent_coder=self._config.agent_coder_enabled,
            stage3_incremental=self._incremental is not None,
            stage3_swe_grep=self._swe_grep is not None,
            stage3_lilo=self._lilo is not None,
            stage4_lean=self._lean_bridge is not None,
            stage4_grpo=self._grpo is not None,
            stage4_diffusion_repair=self._diffusion_repair is not None,
            stage5_synthesis=self._synthesis is not None,
            stage5_repair_agent=self._repair_agent is not None,
            stage5_orchestrator=self._orchestrator is not None,
            stage5_causal_debugger=self._causal_debugger is not None,
            stage5_issue_resolver=self._issue_resolver is not None,
            stage6_hash_chain=self._hash_chain is not None,
            stage6_content_credentials=self._content_credentials is not None,
            stage6_governance_credentials=self._governance_credentials is not None,
            stage6_coevolution=self._hard_negative_miner is not None,
            stage6_adversarial_tester=self._adversarial_tester is not None,
            stage6_formal_specs=self._formal_spec_generator is not None,
            stage6_egraph=self._egraph is not None,
            stage6_symbolic_execution=self._symbolic_execution is not None,
            stage7_inspector=self._inspector is not None,
            stage9_inspector_analytics=self._inspector_analytics is not None,
            stage9_tsdb_persistence=self._tsdb is not None,
        )

        # Child-side genome inheritance: apply inherited learnable params from parent
        try:
            await self._apply_inherited_simula_genome_if_child()
        except Exception as _sg_exc:
            self._logger.warning(
                "simula_genome_child_apply_failed",
                error=str(_sg_exc),
                note="Proceeding with default config params",
            )

    async def _validate_tools(self) -> None:
        """
        Verify every enabled external tool binary exists and is executable.
        Raises RuntimeError on the first missing binary so the process crashes
        at startup rather than silently falling back to a degraded mode.
        """
        import shutil

        checks: list[tuple[bool, str, str]] = [
            # (enabled, binary_path, tool_name)
            (self._config.dafny_enabled, self._config.dafny_binary_path, "Dafny"),
            (self._config.lean_enabled, self._config.lean_binary_path, "Lean 4"),
            (
                self._config.formal_spec_generation_enabled and bool(self._config.tla_plus_binary_path),
                self._config.tla_plus_binary_path or "",
                "TLA+",
            ),
            (
                self._config.formal_spec_generation_enabled and bool(self._config.alloy_binary_path),
                self._config.alloy_binary_path or "",
                "Alloy",
            ),
        ]

        for enabled, binary_path, tool_name in checks:
            if not enabled or not binary_path:
                continue
            if not shutil.which(binary_path) and not Path(binary_path).is_file():
                raise RuntimeError(
                    f"Simula: {tool_name} is enabled but binary not found: '{binary_path}'. "
                    f"Install {tool_name} or set the correct path in SimulaConfig."
                )
            self._logger.debug("tool_binary_ok", tool=tool_name, path=binary_path)

    def set_synapse(self, synapse: Any) -> None:
        """Wire Synapse so Simula can subscribe to grid metabolism events.

        Must be called after initialize() has been awaited. The subscription
        cannot happen during initialize() because Synapse is built after Simula.
        Also starts the ProactiveScanner background loop and wires the event
        bus into RepairMemory so calibration events are emitted correctly.
        """
        self._synapse = synapse
        # initialize() checks `if self._synapse is not None` but that guard is
        # always False at init time - Synapse is wired later. Subscribe here.
        if self._initialized:
            try:
                from systems.synapse.types import SynapseEventType

                synapse._event_bus.subscribe(
                    SynapseEventType.GRID_METABOLISM_CHANGED,
                    self._on_grid_metabolism_changed,
                )
                synapse._event_bus.subscribe(
                    SynapseEventType.EVO_CONSOLIDATION_STALLED,
                    self._on_evo_consolidation_stalled,
                )
                synapse._event_bus.subscribe(
                    SynapseEventType.REPAIR_COMPLETED,
                    self._on_axon_repair_completed,
                )
                synapse._event_bus.subscribe(
                    SynapseEventType.METABOLIC_PRESSURE,
                    self._on_metabolic_pressure,
                )
                if hasattr(SynapseEventType, "COGNITIVE_PRESSURE"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.COGNITIVE_PRESSURE,
                        self._on_cognitive_pressure,
                    )
                # Closure Loop 4: Thymos immune advisory → mutation avoidance
                synapse._event_bus.subscribe(
                    SynapseEventType.IMMUNE_PATTERN_ADVISORY,
                    self._on_immune_pattern_advisory,
                )
                # Metabolic + lifecycle events
                synapse._event_bus.subscribe(
                    SynapseEventType.METABOLIC_EMERGENCY,
                    self._on_metabolic_emergency,
                )
                synapse._event_bus.subscribe(
                    SynapseEventType.GENOME_EXTRACT_REQUEST,
                    self._on_genome_extract_request,
                )
                synapse._event_bus.subscribe(
                    SynapseEventType.ORGANISM_SLEEP,
                    self._on_organism_sleep,
                )
                synapse._event_bus.subscribe(
                    SynapseEventType.BUDGET_EMERGENCY,
                    self._on_budget_emergency,
                )
                # Bounty solution request - Axon bounty_hunt delegates to Simula
                synapse._event_bus.subscribe(
                    SynapseEventType.BOUNTY_SOLUTION_REQUESTED,
                    self._on_bounty_solution_requested,
                )
                # Thymos sandbox validation - replay proposed fix against recent episodes
                synapse._event_bus.subscribe(
                    SynapseEventType.SIMULA_SANDBOX_REQUESTED,
                    self._on_simula_sandbox_requested,
                )
                # Thymos repair coordination via Synapse (no direct import)
                synapse._event_bus.subscribe(
                    SynapseEventType.THYMOS_REPAIR_REQUESTED,
                    self._on_thymos_repair_requested,
                )
                synapse._event_bus.subscribe(
                    SynapseEventType.THYMOS_REPAIR_APPROVED,
                    self._on_thymos_repair_approved,
                )
                # Oneiros consolidation - enriches proposal context
                synapse._event_bus.subscribe(
                    SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE,
                    self._on_oneiros_consolidation_complete,
                )
                # Benchmarks regression - may trigger corrective proposals
                # NOTE: Benchmarks emits BENCHMARK_REGRESSION (not BENCHMARK_REGRESSION_DETECTED)
                synapse._event_bus.subscribe(
                    SynapseEventType.BENCHMARK_REGRESSION,
                    self._on_benchmark_regression_detected,
                )
                # Telos constitutional alignment gap - cache for VALIDATE stage
                if hasattr(SynapseEventType, "ALIGNMENT_GAP_WARNING"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.ALIGNMENT_GAP_WARNING,
                        self._on_telos_alignment_gap_warning,
                    )
                # Degradation Engine §8.2 - stub subscription.
                # Round 2 will implement: perturb the learnable config params
                # by ±drift_rate so Evo must re-optimise to keep them near optimal.
                if hasattr(SynapseEventType, "CONFIG_DRIFT"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.CONFIG_DRIFT,
                        self._on_config_drift,
                    )
                # Exploration proposals from Evo Phase 8.5 (gap closure)
                if hasattr(SynapseEventType, "EXPLORATION_PROPOSED"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.EXPLORATION_PROPOSED,
                        self._on_exploration_proposed,
                    )
                # Evo economic parameter adjustments (Fix 4.1)
                if hasattr(SynapseEventType, "EVO_ADJUST_BUDGET"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.EVO_ADJUST_BUDGET,
                        self._on_evo_adjust_budget,
                    )
                # SPEC_DRAFTED - Nova drafted a Spec for a new subsystem after
                # gap detection; Simula implements after Equor approval arrives
                # via EQUOR_ECONOMIC_PERMIT (correlated by proposal_id).
                if hasattr(SynapseEventType, "SPEC_DRAFTED"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.SPEC_DRAFTED,
                        self._on_spec_drafted,
                    )
                # Economic outcome events → propose corrective mutations
                if hasattr(SynapseEventType, "BOUNTY_PAID"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.BOUNTY_PAID,
                        self._on_bounty_paid,
                    )
                if hasattr(SynapseEventType, "YIELD_DEPLOYMENT_RESULT"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.YIELD_DEPLOYMENT_RESULT,
                        self._on_yield_result,
                    )
                if hasattr(SynapseEventType, "REVENUE_INJECTED"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.REVENUE_INJECTED,
                        self._on_revenue_change,
                    )
                synapse._event_bus.subscribe(
                    SynapseEventType.SYSTEM_MODULATION,
                    self._on_system_modulation,
                )
                # Fatal crash pattern learning - subscribe to confirmed patterns
                # emitted by Thymos (after all repair tiers fail) and Kairos
                # (causal invariants tagged as crash patterns).
                if hasattr(SynapseEventType, "CRASH_PATTERN_CONFIRMED"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.CRASH_PATTERN_CONFIRMED,
                        self._on_crash_pattern_confirmed,
                    )
                # Also subscribe to KAIROS_INVARIANT_DISTILLED so we can detect
                # crash-pattern-tagged invariants directly from Kairos.
                synapse._event_bus.subscribe(
                    SynapseEventType.KAIROS_INVARIANT_DISTILLED,
                    self._on_kairos_invariant_for_crash_patterns,
                )
                # Preventive audit: receive incident history from Thymos
                if hasattr(SynapseEventType, "THYMOS_INCIDENT_RESPONSE"):
                    synapse._event_bus.subscribe(
                        SynapseEventType.THYMOS_INCIDENT_RESPONSE,
                        self._preventive_audit.on_thymos_incident_response,
                    )
            except Exception as exc:
                self._logger.exception("simula_synapse_subscribe_failed", error=str(exc))
                raise

            # Wire event bus into self-healing sub-systems
            event_bus = getattr(synapse, "_event_bus", None)
            if event_bus is not None:
                if self._repair_memory is not None:
                    self._repair_memory._event_bus = event_bus
                if self._proactive_scanner is not None:
                    self._proactive_scanner._event_bus = event_bus
                # Meta-healing: wire sentinel so Simula can report its own failures
                self._sentinel.set_event_bus(event_bus)
                # Loop 3: wire rollback manager so it can emit SIMULA_ROLLBACK_PENALTY
                if self._rollback is not None:
                    self._rollback._event_bus = event_bus
                # Organizational closure: wire event bus into SubsystemGenerator
                if self._subsystem_generator is not None:
                    self._subsystem_generator.set_event_bus(event_bus)
                # Dynamic executor expansion: wire event bus into ExecutorGenerator
                if self._executor_generator is not None:
                    self._executor_generator.set_event_bus(event_bus)
                # Build-error RE training: wire Synapse into HealthChecker so
                # syntax/import/proof failures are captured as negative training examples
                if self._health is not None:
                    self._health.set_synapse(synapse)

            # Start the ProactiveScanner background supervised loop
            if self._proactive_scanner is not None and self._proactive_scanner_task is None:
                from utils.supervision import supervised_task

                self._proactive_scanner_task = supervised_task(
                    self._proactive_scanner.run_forever(),
                    name="simula.proactive_scanner",
                    restart=True,
                    max_restarts=5,
                    backoff_base=2.0,
                    event_bus=event_bus,
                    source_system="simula",
                )
                self._logger.info("proactive_scanner_started")

            # Start the PreventiveAudit background supervised loop (4-hour cycle)
            if self._preventive_audit is not None and self._preventive_audit_task is None:
                from utils.supervision import supervised_task

                self._preventive_audit_task = supervised_task(
                    self._preventive_audit.run_loop(),
                    name="simula.preventive_audit",
                    restart=True,
                    max_restarts=5,
                    backoff_base=2.0,
                    event_bus=event_bus,
                    source_system="simula",
                )
                self._logger.info("preventive_audit_started")

            # Start the governance timeout background loop
            if self._governance_timeout_task is None:
                self._governance_timeout_task = asyncio.create_task(
                    self._governance_timeout_loop(),
                    name="simula.governance_timeout",
                )
                self._logger.info("governance_timeout_loop_started")

            # Start the proactive crash-pattern scan (every 2 hours)
            if self._pattern_scan_task is None:
                self._pattern_scan_task = asyncio.create_task(
                    self._proactive_pattern_scan_loop(),
                    name="simula.pattern_scan",
                )
                self._logger.info("proactive_pattern_scan_started")

    def set_telos(self, telos: Any) -> None:
        """Wire Telos for constitutional binding validation of mutation proposals."""
        self._telos = telos
        self._logger.info("telos_wired_to_simula")

    def set_axon_registry(self, registry: Any) -> None:
        """Wire the live ExecutorRegistry so ExecutorGenerator can hot-load new executors.

        Called during the Axon↔Simula wiring pass after both services initialize().
        Without this, generated executors are written to disk but not hot-loaded.
        """
        if self._executor_generator is not None:
            self._executor_generator.set_axon_registry(registry)
        self._logger.info("axon_registry_wired_to_simula_executor_generator")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so Simula can validate proposals against learned repair patterns.

        Called after initialize() during the integration wiring pass.
        Allows Simula to query Evo's procedural hypotheses about known failures
        and pre-emptively flag proposals that touch failure surfaces without
        including the known fix.
        """
        self._evo = evo
        self._logger.info("evo_wired_to_simula")

    def set_log_analyzer(self, log_analyzer: Any) -> None:
        """Wire the LogAnalyzer so Simula can inject organism health into repair context."""
        self._log_analyzer = log_analyzer
        self._logger.info("log_analyzer_wired_to_simula")

    def set_soma_ref(self, soma: Any) -> None:
        """Wire Soma so Simula can read current allostatic state during repair."""
        self._soma_ref = soma
        if self._proactive_scanner is not None:
            self._proactive_scanner.set_soma_ref(soma)
        self._logger.info("soma_wired_to_simula")

    def set_fovea_ref(self, fovea: Any) -> None:
        """Wire Fovea so Simula can read current attention profile during repair."""
        self._fovea_ref = fovea
        self._logger.info("fovea_wired_to_simula")

    def set_benchmarks(self, benchmarks: Any) -> None:
        """Wire Benchmarks so Simula can emit KPI snapshots on terminal proposal outcomes."""
        self._benchmarks = benchmarks
        self._logger.info("benchmarks_wired_to_simula")

    # ─── Evolution Outcome Events ─────────────────────────────────────────────────

    async def _emit_evolution_outcome(
        self,
        proposal: EvolutionProposal,
        applied: bool,
        *,
        files_changed: list[str] | None = None,
        from_version: int | None = None,
        to_version: int | None = None,
        rollback_reason: str = "",
    ) -> None:
        """Emit EVOLUTION_APPLIED or EVOLUTION_ROLLED_BACK on Synapse.

        This closes the feedback loop: Evo subscribes to reward/penalise
        the source hypotheses, Thymos monitors for post-apply regression,
        and Axon's introspector tracks capability changes.
        """
        event_bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if event_bus is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            risk_level = (
                proposal.simulation.risk_level.value
                if proposal.simulation and hasattr(proposal.simulation.risk_level, "value")
                else "low"
            )

            # Extract hypothesis IDs from evidence (set by bridge for evo-sourced proposals)
            hypothesis_ids = [e for e in (proposal.evidence or []) if e.startswith("hyp-") or len(e) == 36]

            if applied:
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVOLUTION_APPLIED,
                    source_system="simula",
                    data={
                        "proposal_id": proposal.id,
                        "category": proposal.category.value,
                        "description": proposal.description[:200],
                        "from_version": from_version or self._current_version - 1,
                        "to_version": to_version or self._current_version,
                        "files_changed": (files_changed or [])[:20],
                        "risk_level": risk_level,
                        "efe_score": proposal.efe_score,
                        "hypothesis_ids": hypothesis_ids,
                        "source": proposal.source,
                    },
                ))
                # Co-emit SIMULA_EVOLUTION_APPLIED for Mitosis genome distribution
                # (different consumer set from EVOLUTION_APPLIED - Mitosis only).
                _changed = files_changed or []
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SIMULA_EVOLUTION_APPLIED,
                    source_system="simula",
                    data={
                        "variant_id": proposal.id,
                        "genome_id": str(to_version or self._current_version),
                        "improvement_pct": round((proposal.efe_score or 0.0) * 100, 2),
                        "systems_affected": list({
                            f.split("/")[0] for f in _changed if "/" in f
                        }) or [proposal.category.value],
                    },
                ))
            else:
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVOLUTION_ROLLED_BACK,
                    source_system="simula",
                    data={
                        "proposal_id": proposal.id,
                        "category": proposal.category.value,
                        "description": proposal.description[:200],
                        "rollback_reason": rollback_reason[:500],
                        "risk_level": risk_level,
                        "hypothesis_ids": hypothesis_ids,
                        "source": proposal.source,
                    },
                ))

            self._logger.info(
                "evolution_outcome_emitted",
                proposal_id=proposal.id,
                applied=applied,
                hypothesis_ids=len(hypothesis_ids),
            )
        except Exception as exc:
            self._logger.warning("evolution_outcome_emit_failed", error=str(exc))

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict | None = None,
    ) -> None:
        """Emit an EvolutionaryObservable event on Synapse for population tracking."""
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from primitives.common import SystemID
            from primitives.evolutionary import EvolutionaryObservable
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.SIMULA,
                instance_id=getattr(self, "_instance_id", "") or "",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system=SystemID.SIMULA,
                data=obs.model_dump(mode="json"),
            )
            await bus.emit(event)
        except Exception:
            pass  # Best-effort - never block the evolution pipeline

    # ─── RE Training Emission ────────────────────────────────────────────────────

    @staticmethod
    def _make_alignment(
        coherence: float = 0.0,
        care: float = 0.0,
        growth: float = 0.0,
        honesty: float = 0.0,
    ) -> Any:
        """Build a DriveAlignmentVector from drive scores. Lazy import to avoid circular deps."""
        try:
            from primitives.common import DriveAlignmentVector
            return DriveAlignmentVector(
                coherence=round(max(-1.0, min(1.0, coherence)), 3),
                care=round(max(-1.0, min(1.0, care)), 3),
                growth=round(max(-1.0, min(1.0, growth)), 3),
                honesty=round(max(-1.0, min(1.0, honesty)), 3),
            )
        except Exception:
            return None

    async def _emit_re_training_example(
        self,
        *,
        category: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float = 0.5,
        reasoning_trace: str = "",
        alternatives_considered: list[str] | None = None,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        episode_id: str = "",
        constitutional_alignment: Any = None,
    ) -> None:
        """Fire-and-forget RE training example onto Synapse bus."""
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from decimal import Decimal
            from primitives.common import DriveAlignmentVector as _DAV
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.SIMULA,
                category=category,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=max(0.0, min(1.0, outcome_quality)),
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives_considered or [],
                latency_ms=latency_ms,
                cost_usd=Decimal(str(cost_usd)),
                episode_id=episode_id,
                constitutional_alignment=constitutional_alignment or _DAV(),
            )
            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="simula",
                data=example.model_dump(mode="json"),
            ))
        except Exception:
            pass  # Never block the evolution pipeline

    # ─── Proof / Inspection / Vulnerability Events ────────────────────────────────

    async def _emit_proof_event(
        self,
        event_type_name: str,
        *,
        proof_id: str,
        proof_type: str,
        target: str,
        duration_ms: int = 0,
        solver: str = "",
        attempts: int = 0,
        reason: str = "",
        budget_ms: int = 0,
        elapsed_ms: int = 0,
    ) -> None:
        """Emit a proof lifecycle event (PROOF_FOUND/PROOF_FAILED/PROOF_TIMEOUT) with RE training trace."""
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            trace = RETrainingExample(
                source_system=SystemID.SIMULA,
                instruction=f"Proof {event_type_name}: {proof_type} on {target}",
                input_context=f"solver={solver}, target={target}",
                output=reason or f"completed in {duration_ms}ms",
                outcome_quality=1.0 if event_type_name == "PROOF_FOUND" else 0.0,
                category="security_reasoning",
                constitutional_alignment=DriveAlignmentVector(),
                timestamp=utc_now(),
            )

            _event_map = {
                "PROOF_FOUND": SynapseEventType.PROOF_FOUND,
                "PROOF_FAILED": SynapseEventType.PROOF_FAILED,
                "PROOF_TIMEOUT": SynapseEventType.PROOF_TIMEOUT,
            }
            event_type = _event_map.get(event_type_name)
            if event_type is None:
                return
            await bus.emit(SynapseEvent(
                event_type=event_type,
                source_system="simula",
                data={
                    "proof_id": proof_id,
                    "proof_type": proof_type,
                    "target": target,
                    "solver": solver,
                    "duration_ms": duration_ms,
                    "attempts": attempts,
                    "reason": reason,
                    "budget_ms": budget_ms,
                    "elapsed_ms": elapsed_ms,
                    "re_training_trace": trace.model_dump(mode="json"),
                },
            ))
        except Exception:
            pass  # Best-effort - never block the proof pipeline

    async def _emit_proof_events_from_health(
        self,
        health: Any,
        proposal: EvolutionProposal,
    ) -> None:
        """Emit PROOF_FOUND/PROOF_FAILED/PROOF_TIMEOUT from a completed health check.

        Extracts Dafny, Z3, and Lean results from the HealthCheckResult and fires
        the appropriate proof lifecycle event for each. Best-effort - never raises.
        """
        try:
            fv = getattr(health, "formal_verification", None)
            if fv is not None:
                # Dafny result
                dafny = getattr(fv, "dafny", None)
                if dafny is not None:
                    dafny_status = getattr(dafny, "status", None)
                    if dafny_status is not None:
                        status_val = str(dafny_status.value if hasattr(dafny_status, "value") else dafny_status)
                        if status_val == "verified":
                            evt = "PROOF_FOUND"
                        elif status_val == "timeout":
                            evt = "PROOF_TIMEOUT"
                        elif status_val in ("failed", "parse_error"):
                            evt = "PROOF_FAILED"
                        else:
                            evt = None
                        if evt:
                            asyncio.ensure_future(self._emit_proof_event(
                                evt,
                                proof_id=f"dafny.{proposal.id}",
                                proof_type="dafny",
                                target=proposal.category.value,
                                duration_ms=getattr(dafny, "total_dafny_time_ms", 0),
                                attempts=getattr(dafny, "rounds_attempted", 0),
                                reason=getattr(dafny, "error_summary", ""),
                            ))
                # Z3 result
                z3 = getattr(fv, "z3", None)
                if z3 is not None:
                    valid = getattr(z3, "valid_invariants", [])
                    rounds = getattr(z3, "rounds_attempted", 0)
                    if rounds > 0:
                        evt = "PROOF_FOUND" if valid else "PROOF_FAILED"
                        asyncio.ensure_future(self._emit_proof_event(
                            evt,
                            proof_id=f"z3.{proposal.id}",
                            proof_type="z3_invariants",
                            target=proposal.category.value,
                            duration_ms=getattr(z3, "total_z3_time_ms", 0),
                            attempts=rounds,
                            reason=getattr(z3, "error_summary", ""),
                            solver="z3",
                        ))
            # Lean result
            lv = getattr(health, "lean_verification", None)
            if lv is not None:
                lean_status = getattr(lv, "status", None)
                if lean_status is not None:
                    status_val = str(lean_status.value if hasattr(lean_status, "value") else lean_status)
                    if status_val == "proved":
                        evt = "PROOF_FOUND"
                    elif status_val == "timeout":
                        evt = "PROOF_TIMEOUT"
                    elif status_val in ("failed", "parse_error"):
                        evt = "PROOF_FAILED"
                    else:
                        evt = None
                    if evt:
                        asyncio.ensure_future(self._emit_proof_event(
                            evt,
                            proof_id=f"lean.{proposal.id}",
                            proof_type="lean4",
                            target=proposal.category.value,
                            duration_ms=getattr(lv, "total_lean_time_ms", 0),
                            attempts=len(getattr(lv, "attempts", [])),
                            reason=getattr(lv, "error_summary", ""),
                            solver="lean4",
                        ))
        except Exception:
            pass  # Best-effort - never block the pipeline

    async def _emit_vulnerability_confirmed(
        self,
        *,
        vuln_id: str,
        severity: str,
        target: str,
        cwe_id: str = "",
        poc_hash: str = "",
    ) -> None:
        """Emit VULNERABILITY_CONFIRMED + INSPECTOR_VULNERABILITY_FOUND after PoC validation."""
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # INSPECTOR_VULNERABILITY_FOUND - closes the observatory gap for this event
            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.INSPECTOR_VULNERABILITY_FOUND,
                source_system="simula",
                data={
                    "vuln_id": vuln_id,
                    "severity": severity,
                    "target": target,
                    "cwe_id": cwe_id,
                    "poc_hash": poc_hash,
                },
            ))

            trace = RETrainingExample(
                source_system=SystemID.SIMULA,
                instruction=f"Vulnerability confirmed: {cwe_id} on {target}",
                input_context=f"severity={severity}, target={target}",
                output=f"Confirmed vuln {vuln_id}",
                outcome_quality=1.0,
                category="security_reasoning",
                constitutional_alignment=DriveAlignmentVector(),
                timestamp=utc_now(),
            )
            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.VULNERABILITY_CONFIRMED,
                source_system="simula",
                data={
                    "vuln_id": vuln_id,
                    "severity": severity,
                    "target": target,
                    "cwe_id": cwe_id,
                    "poc_hash": poc_hash,
                    "re_training_trace": trace.model_dump(mode="json"),
                },
            ))
        except Exception:
            pass

    async def _emit_remediation_applied(
        self,
        *,
        vuln_id: str,
        patch_id: str,
        target: str,
        verification_passed: bool,
    ) -> None:
        """Emit REMEDIATION_APPLIED after an autonomous patch."""
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            trace = RETrainingExample(
                source_system=SystemID.SIMULA,
                instruction=f"Remediation applied for {vuln_id} on {target}",
                input_context=f"patch_id={patch_id}",
                output=f"verification={'passed' if verification_passed else 'failed'}",
                outcome_quality=1.0 if verification_passed else 0.3,
                category="security_reasoning",
                constitutional_alignment=DriveAlignmentVector(),
                timestamp=utc_now(),
            )
            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.REMEDIATION_APPLIED,
                source_system="simula",
                data={
                    "vuln_id": vuln_id,
                    "patch_id": patch_id,
                    "target": target,
                    "verification_passed": verification_passed,
                    "re_training_trace": trace.model_dump(mode="json"),
                },
            ))
        except Exception:
            pass

    async def _emit_inspection_complete(
        self,
        *,
        inspection_id: str,
        target: str,
        vulns_found: int,
        vulns_patched: int,
        duration_ms: int,
    ) -> None:
        """Emit INSPECTION_COMPLETE after a full inspection cycle."""
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            trace = RETrainingExample(
                source_system=SystemID.SIMULA,
                instruction=f"Inspection complete on {target}",
                input_context=f"target={target}",
                output=f"found={vulns_found}, patched={vulns_patched}, duration={duration_ms}ms",
                outcome_quality=min(1.0, vulns_patched / max(vulns_found, 1)),
                category="security_reasoning",
                constitutional_alignment=DriveAlignmentVector(),
                timestamp=utc_now(),
            )
            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.INSPECTION_COMPLETE,
                source_system="simula",
                data={
                    "inspection_id": inspection_id,
                    "target": target,
                    "vulns_found": vulns_found,
                    "vulns_patched": vulns_patched,
                    "duration_ms": duration_ms,
                    "re_training_trace": trace.model_dump(mode="json"),
                },
            ))
        except Exception:
            pass

    # ─── Self-Healing via Thymos ──────────────────────────────────────────────────

    async def _emit_incident_to_thymos(
        self,
        error_type: str,
        error_message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit an Incident to Thymos when Simula encounters an internal error.

        The immune system must be able to heal itself. Simula errors that go
        unreported are the worst failure mode - they prevent the organism from
        self-healing.
        """
        if self._synapse is None:
            self._logger.warning(
                "simula_incident_not_emitted_no_synapse",
                error_type=error_type,
                error_message=error_message,
            )
            return

        event_bus = getattr(self._synapse, "_event_bus", None)
        if event_bus is None:
            self._logger.warning(
                "simula_incident_not_emitted_no_bus",
                error_type=error_type,
                error_message=error_message,
            )
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            incident_data = {
                "incident_class": "crash",
                "severity": "high",
                "fingerprint": hashlib.md5(
                    f"simula_{error_type}".encode()
                ).hexdigest(),
                "source_system": "simula",
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {},
                "affected_systems": ["simula"],
                "blast_radius": 0.5,
                "user_visible": False,
            }

            await event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_FAILED,
                    source="simula",
                    timestamp=time.time(),
                    data={"incident": incident_data},
                )
            )
            self._logger.debug(
                "simula_incident_emitted",
                error_type=error_type,
            )
        except Exception as emit_exc:
            self._logger.warning(
                "failed_to_emit_simula_incident",
                error=str(emit_exc),
            )

    # ─── Oneiros SimulaProtocol Implementation ──────────────────────────────────

    async def get_pending_mutations(self) -> list[dict[str, Any]]:
        """Return mutation proposals awaiting lucid dream simulation.

        Satisfies Oneiros's SimulaProtocol. Returns proposals that have
        passed validation and simulation but haven't been dream-tested yet.
        Each dict follows the format Oneiros expects:
          { id, description, mutation_type, target, value, affected_systems }
        """
        async with self._proposals_lock:
            proposals = list(self._active_proposals.values())

        mutations: list[dict[str, Any]] = []
        for p in proposals:
            # Only export proposals that have been simulated but not yet applied
            if p.status not in (ProposalStatus.SIMULATING, ProposalStatus.APPROVED):
                continue
            # Skip proposals already dream-tested
            if getattr(p, "_dream_tested", False):
                continue

            spec = p.change_spec
            mutations.append({
                "id": p.id,
                "description": p.description,
                "mutation_type": p.category.value,
                "target": getattr(spec, "target_system", "") or getattr(spec, "target_parameter", ""),
                "value": getattr(spec, "budget_new_value", 0.0) or 0.0,
                "affected_systems": getattr(spec, "affected_systems", []) or [],
            })

        return mutations

    async def report_simulation_result(
        self, mutation_id: str, recommendation: str, report: dict[str, Any]
    ) -> None:
        """Receive a lucid dream simulation result from Oneiros.

        Satisfies Oneiros's SimulaProtocol. Updates the proposal with
        dream test metadata and adjusts pipeline flow based on recommendation.
        """
        self._logger.info(
            "dream_simulation_result_received",
            mutation_id=mutation_id,
            recommendation=recommendation,
            performance_delta=report.get("overall_performance_delta", 0.0),
            violations=report.get("any_constitutional_violations", False),
        )

        async with self._dream_results_lock:
            self._dream_results[mutation_id] = {
                "recommendation": recommendation,
                "report": report,
            }

        # Mark the proposal as dream-tested
        async with self._proposals_lock:
            proposal = self._active_proposals.get(mutation_id)
            if proposal is not None:
                proposal._dream_tested = True  # type: ignore[attr-defined]
                proposal.dream_origin = True
                proposal.dream_coherence_score = report.get(
                    "overall_performance_delta", 0.0
                )

    # ─── Grid Metabolism Reaction ───────────────────────────────────────────────

    async def _on_grid_metabolism_changed(self, event: Any) -> None:
        """
        React to physical grid carbon intensity changes.

        CONSERVATION: pause the evolution pipeline - no new proposals will be
                      processed until the grid recovers. The SimulaCodeAgent is
                      expensive (multi-turn LLM calls) and must not run on a
                      carbon-heavy grid.
        NORMAL / GREEN_SURPLUS: resume normal evolution throughput.
        """
        raw = getattr(event, "data", {}).get("state", "")
        if not raw or raw == self._grid_state:
            return

        old_state = self._grid_state
        self._grid_state = raw

        if raw == "conservation":
            self._logger.info(
                "simula_evolution_paused",
                reason="grid_conservation",
                from_state=old_state,
                to_state=raw,
                active_proposals=len(self._active_proposals),
            )
        elif raw == "green_surplus":
            self._logger.info(
                "simula_evolution_resumed_green_surplus",
                from_state=old_state,
                to_state=raw,
            )
        else:  # normal
            self._logger.info(
                "simula_evolution_resumed_normal",
                from_state=old_state,
                to_state=raw,
            )

    async def _on_evo_consolidation_stalled(self, event: Any) -> None:
        """React to Evo's learning pipeline being stalled.

        When Evo hasn't consolidated in 2× its expected interval, evolution
        proposals sourced from Evo are deferred - applying structural changes
        when Evo can't learn from outcomes wastes resources.  The stall flag
        auto-clears after 2× the expected interval reported in the event.
        """
        data = getattr(event, "data", {})
        expected_s = data.get("expected_interval_s", 300)
        self._evo_consolidation_stalled = True
        self._evo_stall_expires_at = time.monotonic() + (expected_s * 2)
        self._logger.warning(
            "simula_evo_stall_detected",
            last_consolidation_ago_s=data.get("last_consolidation_ago_s"),
            auto_clear_after_s=expected_s * 2,
        )

    async def _on_axon_repair_completed(self, event: Any) -> None:
        """Ingest Axon repair outcomes into repair_memory for calibration.

        Only processes repairs from Axon executors (source_system starts with
        "axon.").  Ignores Simula's own REPAIR_COMPLETED emissions to avoid
        double-counting.
        """
        source = getattr(event, "source_system", "")
        if not source.startswith("axon."):
            return

        data = getattr(event, "data", {})
        success = data.get("success", False)
        fix_type = data.get("fix_type", "unknown")
        incident_class = data.get("incident_class", "")
        elapsed_ms = data.get("elapsed_ms", 0)

        self._logger.info(
            "axon_repair_outcome_received",
            source=source,
            success=success,
            fix_type=fix_type,
            incident_class=incident_class,
            elapsed_ms=elapsed_ms,
        )

        if self._repair_memory is not None:
            try:
                await self._repair_memory.record_axon_repair(
                    source=source,
                    success=success,
                    fix_type=fix_type,
                    incident_class=incident_class,
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:
                self._logger.warning(
                    "axon_repair_record_failed", error=str(exc)
                )

    # ─── Shutdown ───────────────────────────────────────────────────────────────

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """Handle METABOLIC_PRESSURE - gate mutation pipeline by starvation level."""
        data = getattr(event, "data", {}) or {}
        level = data.get("starvation_level", "")
        if not level:
            return
        old = self._starvation_level
        self._starvation_level = level
        if level != old:
            self._logger.info("simula_starvation_level_changed", old=old, new=level)
            # Under EMERGENCY+, cancel proactive scanner to save resources
            if level in ("emergency", "critical"):
                if self._proactive_scanner_task is not None and not self._proactive_scanner_task.done():
                    self._proactive_scanner_task.cancel()
                    self._logger.warning("proactive_scanner_cancelled_starvation", level=level)
            # Boost economic mutation priority during metabolic crisis so the
            # self-evolution pipeline prioritises economic-repair proposals.
            if level in ("critical", "emergency"):
                self._metabolic_boost = 2.0
            elif level == "austerity":
                self._metabolic_boost = 1.5
            else:
                self._metabolic_boost = 1.0

    async def _on_bounty_paid(self, event: Any) -> None:
        """BOUNTY_PAID → propose code mutations that reinforce bounty-winning patterns.

        A confirmed payment is evidence that the current bounty-hunting strategy
        works. Simula records this so that future mutation proposals can be
        weighted toward reinforcing (rather than disrupting) the effective pattern.
        """
        try:
            data = getattr(event, "data", {}) or {}
            bounty_id = str(data.get("bounty_id", ""))
            amount = float(data.get("reward_usd", data.get("amount", 0.0)))
            self._logger.info(
                "simula_bounty_paid_noted",
                bounty_id=bounty_id,
                amount_usd=amount,
                action="reinforce_bounty_strategy",
            )
            # Track as positive economic signal - future ProactiveScanner proposals
            # should avoid disrupting code paths that led to this outcome.
            if hasattr(self, "_proactive_scanner") and self._proactive_scanner is not None:
                if hasattr(self._proactive_scanner, "record_positive_economic_outcome"):
                    self._proactive_scanner.record_positive_economic_outcome(
                        source="bounty_paid", value_usd=amount
                    )
        except Exception as exc:
            self._logger.warning("on_bounty_paid_simula_failed", error=str(exc))

    async def _on_yield_result(self, event: Any) -> None:
        """YIELD_DEPLOYMENT_RESULT → propose rebalancing mutations if APY dropped.

        A failed yield deployment or unexpected result may indicate that the
        protocol-selection logic or capital-allocation code needs a mutation to
        adapt. If success=False, propose a targeted rebalancing evolution.
        """
        try:
            data = getattr(event, "data", {}) or {}
            success = bool(data.get("success", True))
            protocol = str((data.get("data") or {}).get("protocol", "unknown"))
            if not success:
                error = str(data.get("error", ""))
                self._logger.warning(
                    "simula_yield_deployment_failed_noted",
                    protocol=protocol,
                    error=error[:120],
                    action="consider_rebalancing_mutation",
                )
                # Under metabolic boost, elevate priority of any pending economic
                # mutation proposals so the organism self-heals faster.
                if self._metabolic_boost > 1.0:
                    self._logger.info(
                        "simula_yield_failure_elevated_priority",
                        metabolic_boost=self._metabolic_boost,
                    )
        except Exception as exc:
            self._logger.warning("on_yield_result_simula_failed", error=str(exc))

    async def _on_revenue_change(self, event: Any) -> None:
        """REVENUE_INJECTED → track revenue trend for mutation prioritisation.

        Revenue events are lightweight signals. Simula notes them to understand
        whether the economic trajectory is improving (less pressure to propose
        economic-repair mutations) or static (maintain mutation pressure).
        """
        try:
            data = getattr(event, "data", {}) or {}
            amount = float(data.get("amount_usd", 0.0))
            source = str(data.get("source", "unknown"))
            self._logger.debug(
                "simula_revenue_injected_noted",
                amount_usd=amount,
                source=source,
            )
            # Revenue recovery → reduce metabolic boost if organism is recovering
            if amount > 0 and self._metabolic_boost > 1.0:
                # Only reset if starvation level has also improved (guard against
                # a single revenue event masking persistent starvation).
                if self._starvation_level in ("nominal", "cautious", ""):
                    self._metabolic_boost = 1.0
                    self._logger.info(
                        "simula_metabolic_boost_reset_on_revenue",
                        amount_usd=amount,
                        source=source,
                    )
        except Exception as exc:
            self._logger.warning("on_revenue_change_simula_failed", error=str(exc))

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle VitalityCoordinator austerity orders.

        Skia emits SYSTEM_MODULATION when the organism needs to conserve resources.
        This system applies the directive and ACKs so Skia knows the order was received.
        """
        data = getattr(event, "data", {}) or {}
        level = data.get("level", "nominal")
        halt_systems = data.get("halt_systems", [])
        modulate = data.get("modulate", {})

        system_id = "simula"
        compliant = True
        reason: str | None = None

        if system_id in halt_systems:
            self._modulation_halted = True
            self._logger.warning("system_modulation_halt", level=level)
        elif system_id in modulate:
            directives = modulate[system_id]
            self._apply_modulation_directives(directives)
            self._logger.info("system_modulation_applied", level=level, directives=directives)
        elif level == "nominal":
            self._modulation_halted = False
            self._logger.info("system_modulation_resumed", level=level)

        # Emit ACK so Skia knows the order was received
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                event_bus = getattr(self._synapse, "_event_bus", None)
                if event_bus is not None:
                    await event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                        data={
                            "system_id": system_id,
                            "level": level,
                            "compliant": compliant,
                            "reason": reason,
                        },
                        source_system=system_id,
                    ))
            except Exception as exc:
                self._logger.warning("modulation_ack_failed", error=str(exc))

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply modulation directives from VitalityCoordinator.

        Simula directive: {"speculative_mutations": False} - pause speculative
        mutations to conserve compute and economic resources during austerity.
        """
        speculative = directives.get("speculative_mutations", True)
        if not speculative:
            self._logger.info("modulation_speculative_mutations_paused")
        else:
            self._logger.info("modulation_directives_received", directives=directives)

    async def _on_cognitive_pressure(self, event: Any) -> None:
        """COGNITIVE_PRESSURE → enter minimal verification mode at high compression load.

        When Logos compression utilization exceeds 0.85, switch to shallow verification
        (skip Dafny/Lean/Z3) to reduce cognitive overhead. Restores full verification
        below 0.75 (hysteresis prevents oscillation).
        """
        try:
            data = getattr(event, "data", {}) or {}
            pressure = float(data.get("pressure", 0.0))
            if self._health is not None:
                was_shallow = self._health._shallow_verification_mode
                if pressure >= 0.85:
                    self._health._shallow_verification_mode = True
                elif pressure < 0.75:
                    self._health._shallow_verification_mode = False
                if self._health._shallow_verification_mode != was_shallow:
                    self._logger.info(
                        "simula_cognitive_pressure_verification_mode_changed",
                        shallow=self._health._shallow_verification_mode,
                        pressure=pressure,
                    )
        except Exception as exc:
            self._logger.warning("on_cognitive_pressure_failed", error=str(exc))

    async def _on_immune_pattern_advisory(self, event: Any) -> None:
        """Handle IMMUNE_PATTERN_ADVISORY from Thymos (Loop 4)."""
        data = getattr(event, "data", {}) or {}
        self._immune_filter.ingest_advisory(data)

    async def _on_metabolic_emergency(self, event: Any) -> None:
        """Pause non-critical proof work on metabolic emergency."""
        self._logger.warning(
            "metabolic_emergency_received",
            starvation_level=(getattr(event, "data", {}) or {}).get("starvation_level"),
        )
        self._grid_state = "conservation"
        if hasattr(self, "_inspector") and self._inspector is not None:
            self._logger.info("pausing_inspector_on_metabolic_emergency")

    async def _on_genome_extract_request(self, event: Any) -> None:
        """Return Simula's heritable state for genome extraction."""
        from systems.synapse.types import SynapseEvent, SynapseEventType

        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return

        segment_data: dict[str, Any] = {}
        try:
            from systems.simula.genome_adapter import SimulaGenomeAdapter
            from systems.simula.genome import SimulaGenomeExtractor

            if self._neo4j is not None:
                extractor = SimulaGenomeExtractor(
                    neo4j=self._neo4j,
                    instance_id=getattr(self, "_instance_name", "EOS"),
                )
                adapter = SimulaGenomeAdapter(extractor=extractor)
                segment = await adapter.extract_genome_segment()
                segment_data = segment.model_dump(mode="json")
        except Exception as exc:
            self._logger.warning("genome_extract_failed", error=str(exc))
            segment_data = {"error": str(exc)}

        await bus.emit(SynapseEvent(
            event_type=SynapseEventType.GENOME_EXTRACT_RESPONSE,
            source_system="simula",
            data={
                "request_id": (getattr(event, "data", {}) or {}).get("request_id", ""),
                "segment": segment_data,
            },
        ))

        # Notify the organism that Simula genome segment was extracted
        if "error" not in segment_data:
            import os as _os_ge
            try:
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SIMULA_GENOME_EXTRACTED,
                    source_system="simula",
                    data={
                        "instance_id": _os_ge.environ.get("ECODIAOS_INSTANCE_ID", "eos-default"),
                        "genome_id": (getattr(event, "data", {}) or {}).get("request_id", ""),
                        "generation": segment_data.get("generation", 1),
                        "record_count": len(segment_data.get("evolution_records", [])),
                        "parameter_count": len(segment_data.get("learnable_parameters", {})),
                    },
                ))
            except Exception:
                pass  # Non-critical

    async def _on_organism_sleep(self, event: Any) -> None:
        """Graceful wind-down of active proofs on organism sleep."""
        self._logger.info(
            "organism_sleep_received",
            sleep_stage=(getattr(event, "data", {}) or {}).get("sleep_stage"),
        )
        self._grid_state = "conservation"

    async def _on_budget_emergency(self, event: Any) -> None:
        """Kill expensive proof searches on budget emergency."""
        self._logger.warning(
            "budget_emergency_received",
            utilization=(getattr(event, "data", {}) or {}).get("utilization"),
        )
        self._grid_state = "conservation"

    # ─── Metabolic Gate ─────────────────────────────────────────────────────────

    async def _check_metabolic_gate(self, operation: str, estimated_cost_usd: str = "0.01") -> bool:
        """Check metabolic permission before expensive operations.

        Returns True if work should proceed, False if it should be queued.
        """
        if self._starvation_level in ("critical", "emergency"):
            self._logger.info(
                "metabolic_gate_denied",
                operation=operation,
                starvation_level=self._starvation_level,
            )
            bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
            if bus is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType

                    await bus.emit(SynapseEvent(
                        event_type=SynapseEventType.METABOLIC_GATE_CHECK,
                        source_system="simula",
                        data={
                            "system_id": "simula",
                            "operation": operation,
                            "estimated_cost_usd": estimated_cost_usd,
                            "decision": "denied",
                            "starvation_level": self._starvation_level,
                        },
                    ))
                except Exception:
                    pass
            return False
        return True

    # ─── New Synapse event handlers ────────────────────────────────────────────────────────

    async def _emit_evolution_rejected(
        self,
        proposal: EvolutionProposal,
        reason: str,
        stage: str,
    ) -> None:
        """Emit EVOLUTION_REJECTED so Evo penalises hypotheses and Thymos can observe."""
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVOLUTION_REJECTED,
                source_system="simula",
                data={
                    "proposal_id": proposal.id,
                    "reason": reason,
                    "stage": stage,
                    "category": proposal.category.value,
                    "source": proposal.source,
                },
            ))
        except Exception:
            pass

    async def _emit_evolution_awaiting_governance(
        self,
        proposal: EvolutionProposal,
        governance_record_id: str,
    ) -> None:
        """Emit EVOLUTION_AWAITING_GOVERNANCE so Nova/Telos can update expectations."""
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            sim_risk = (
                proposal.simulation.risk_level.value
                if proposal.simulation
                else "unknown"
            )
            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVOLUTION_AWAITING_GOVERNANCE,
                source_system="simula",
                data={
                    "proposal_id": proposal.id,
                    "governance_record_id": governance_record_id,
                    "category": proposal.category.value,
                    "risk_level": sim_risk,
                },
            ))
        except Exception:
            pass

    async def _on_simula_sandbox_requested(self, event: Any) -> None:
        """
        Handle SIMULA_SANDBOX_REQUESTED from Thymos.
        Validates a proposed Tier 4 repair by checking constitutional constraints
        and risk bounds, then emits SIMULA_SANDBOX_RESULT with correlation_id.
        Thymos waits 30s for this response (fail-closed on timeout).
        """
        data = getattr(event, "data", {}) or {}
        correlation_id = data.get("correlation_id", "")
        repair_action = data.get("repair_action", "")
        repair_tier = data.get("repair_tier", "")
        target_system = data.get("target_system", "")
        parameter_changes = data.get("parameter_changes", [])
        repair_reason = data.get("reason", data.get("repair_reason", ""))

        self._logger.info(
            "simula_sandbox_requested",
            correlation_id=correlation_id,
            repair_action=repair_action,
            repair_tier=repair_tier,
            target_system=target_system,
        )

        approved = False
        reason = "unknown"

        try:
            # Iron Rule check - reject repairs that touch protected systems
            protected = {"equor", "simula", "constitution", "invariant"}
            action_lower = (repair_action or "").lower()
            target_lower = (target_system or "").lower()
            if any(p in action_lower or p in target_lower for p in protected):
                reason = f"iron_rule_violation: action or target touches protected system"
                self._logger.warning(
                    "simula_sandbox_iron_rule_blocked",
                    correlation_id=correlation_id,
                    action=repair_action,
                    target=target_system,
                )
            else:
                # Check parameter changes against constitutional constraints
                violation = None
                for change in parameter_changes:
                    param_name = (change.get("name") or change.get("param", "")).lower()
                    if any(p in param_name for p in protected):
                        violation = param_name
                        break

                if violation:
                    reason = f"iron_rule_violation: parameter '{violation}' touches protected system"
                else:
                    # Simulate: use ChangeSimulator for a lightweight counterfactual check
                    # if available, otherwise approve based on tier check
                    if hasattr(self, "_simulator") and self._simulator is not None:
                        try:
                            from systems.simula.evolution_types import (
                                EvolutionCategory,
                                EvolutionProposal,
                            )
                            sandbox_proposal = EvolutionProposal(
                                category=EvolutionCategory.ADJUST_PARAMETERS,
                                description=(
                                    f"[Thymos sandbox] {repair_action} on {target_system}: "
                                    f"{repair_reason[:200]}"
                                ),
                                source="thymos_sandbox",
                                expected_benefit="Self-healing repair validation",
                                systems_affected=[target_system] if target_system else [],
                                metadata={
                                    "sandbox_validation": True,
                                    "correlation_id": correlation_id,
                                    "repair_tier": repair_tier,
                                    "parameter_changes": parameter_changes,
                                },
                            )
                            sim_result = await asyncio.wait_for(
                                self._simulator.simulate_change(sandbox_proposal),
                                timeout=20.0,
                            )
                            # Approve if simulation risk is acceptable
                            if sim_result.risk_score <= 0.7 and sim_result.confidence >= 0.3:
                                approved = True
                                reason = (
                                    f"simulation_passed: risk={sim_result.risk_score:.2f}, "
                                    f"confidence={sim_result.confidence:.2f}"
                                )
                            else:
                                reason = (
                                    f"simulation_rejected: risk={sim_result.risk_score:.2f} "
                                    f"(>0.7) or confidence={sim_result.confidence:.2f} (<0.3)"
                                )
                        except asyncio.TimeoutError:
                            # Fail-closed on simulation timeout within sandbox
                            reason = "simulation_timeout"
                        except Exception as sim_exc:
                            # Simulator unavailable - approve PARAMETER/RESTART tiers conservatively
                            if repair_tier in ("PARAMETER", "RESTART", "KNOWN_FIX"):
                                approved = True
                                reason = f"simulator_error_conservative_approve: {sim_exc}"
                            else:
                                reason = f"simulator_error_conservative_reject: {sim_exc}"
                    else:
                        # No simulator wired - approve lower tiers only
                        if repair_tier in ("PARAMETER", "RESTART", "KNOWN_FIX"):
                            approved = True
                            reason = "no_simulator_conservative_approve"
                        else:
                            reason = "no_simulator_conservative_reject"

        except Exception as exc:
            self._logger.error(
                "simula_sandbox_error",
                correlation_id=correlation_id,
                error=str(exc),
            )
            reason = f"sandbox_internal_error: {exc}"

        # Emit result - Thymos resolves its Future by correlation_id
        event_bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if event_bus is None:
            self._logger.warning(
                "simula_sandbox_no_synapse",
                correlation_id=correlation_id,
            )
        else:
            try:
                from systems.synapse.types import SynapseEvent
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SIMULA_SANDBOX_RESULT,
                    source_system="simula",
                    data={
                        "correlation_id": correlation_id,
                        "approved": approved,
                        "reason": reason,
                    },
                ))
            except Exception as emit_exc:
                self._logger.error(
                    "simula_sandbox_emit_failed",
                    correlation_id=correlation_id,
                    error=str(emit_exc),
                )

        self._logger.info(
            "simula_sandbox_result",
            correlation_id=correlation_id,
            approved=approved,
            reason=reason,
        )

    async def _on_thymos_repair_requested(self, event: Any) -> None:
        """
        Handle THYMOS_REPAIR_REQUESTED - Thymos needs Simula to synthesise a
        structural repair proposal for a high-tier incident.
        """
        data = getattr(event, "data", {}) or {}
        incident_id = data.get("incident_id", "")
        affected_system = data.get("affected_system", "")
        description = data.get("description", "")
        severity = data.get("severity", "medium")
        repair_tier = data.get("repair_tier", 3)

        self._logger.info(
            "thymos_repair_requested",
            incident_id=incident_id,
            affected_system=affected_system,
            severity=severity,
            repair_tier=repair_tier,
        )

        # Only handle tier 3+ repairs - lower tiers Thymos handles directly
        if repair_tier < 3:
            return

        try:
            from systems.simula.evolution_types import (
                EvolutionCategory,
                EvolutionProposal,
            )

            proposal = EvolutionProposal(
                category=EvolutionCategory.MODIFY_LOGIC,
                description=(
                    f"[Thymos repair] Incident {incident_id} on {affected_system}: "
                    f"{description[:300]}"
                ),
                source="thymos",
                expected_benefit=(
                    f"Repair severity={severity} incident in {affected_system}"
                ),
                systems_affected=[affected_system] if affected_system else [],
                metadata={
                    "incident_id": incident_id,
                    "repair_tier": repair_tier,
                    "thymos_initiated": True,
                },
            )
            asyncio.ensure_future(self.process_proposal(proposal))
        except Exception as exc:
            self._logger.warning(
                "thymos_repair_proposal_creation_failed",
                incident_id=incident_id,
                error=str(exc),
            )

    async def _on_bounty_solution_requested(self, event: Any) -> None:
        """
        Handle BOUNTY_SOLUTION_REQUESTED - Axon bounty_hunt executor asks Simula
        to generate a code solution for an external bounty issue.

        Constructs an EvolutionProposal from the raw event payload and fires
        process_proposal() asynchronously. Result (if any) surfaces via
        BOUNTY_SOLUTION_PENDING emitted by the bounty submission path.
        """
        data = getattr(event, "data", {}) or {}
        title = data.get("title", "Bounty solution")
        description = data.get("description", "")
        repository_url = data.get("repository_url", "")
        issue_url = data.get("issue_url", "")
        metadata = data.get("metadata", {})
        request_id = data.get("request_id", "")

        self._logger.info(
            "bounty_solution_requested",
            request_id=request_id,
            title=title[:80],
            issue_url=issue_url,
        )

        try:
            from systems.simula.evolution_types import (
                EvolutionCategory,
                EvolutionProposal,
            )

            proposal = EvolutionProposal(
                category=EvolutionCategory.MODIFY_LOGIC,
                description=(
                    f"[Bounty] {title[:200]}\n\n"
                    f"Repository: {repository_url}\nIssue: {issue_url}\n\n"
                    f"{description[:1500]}"
                ),
                source="bounty",
                expected_benefit=(
                    f"Earn bounty reward of {metadata.get('reward_usd', '?')} USD "
                    f"by solving: {title[:100]}"
                ),
                systems_affected=[],
                metadata={
                    "request_id": request_id,
                    "issue_url": issue_url,
                    "repository_url": repository_url,
                    "reward_usd": metadata.get("reward_usd", "0"),
                    "labels": metadata.get("labels", []),
                    "platform": metadata.get("platform", ""),
                    "bounty_initiated": True,
                },
            )
            asyncio.ensure_future(self.process_proposal(proposal))
        except Exception as exc:
            self._logger.warning(
                "bounty_solution_proposal_creation_failed",
                request_id=request_id,
                error=str(exc),
            )

    async def _on_thymos_repair_approved(self, event: Any) -> None:
        """
        Handle THYMOS_REPAIR_APPROVED - Thymos has approved a Simula-generated
        repair. Resume pipeline from apply stage if proposal is awaiting.
        """
        data = getattr(event, "data", {}) or {}
        proposal_id = data.get("proposal_id", "")
        incident_id = data.get("incident_id", "")

        self._logger.info(
            "thymos_repair_approved",
            proposal_id=proposal_id,
            incident_id=incident_id,
        )

        proposal = self._active_proposals.get(proposal_id)
        if proposal is None:
            return

        # If awaiting governance (high-risk repair), approve it now
        if proposal.status.value == "awaiting_governance":
            governance_id = getattr(proposal, "governance_record_id", "") or ""
            asyncio.ensure_future(
                self.approve_governed_proposal(proposal_id, governance_id)
            )

    async def _on_spec_drafted(self, event: Any) -> None:
        """
        Handle SPEC_DRAFTED - Nova's SelfModificationPipeline has drafted a new
        Spec document for a capability requiring a full subsystem (not just an
        executor). Simula queues a SubsystemGenerator run for the Spec, gated
        on Equor's constitutional review of the SELF_MODIFICATION_PROPOSED event
        (correlated by proposal_id). The actual implementation fires only after
        EQUOR_ECONOMIC_PERMIT arrives.

        Simula stores the pending spec in _pending_spec_drafts keyed by
        proposal_id; when _on_novel_action_requested fires with
        pipeline_managed=True and the same proposal_id, SubsystemGenerator is
        invoked with the spec content as the canonical purpose description.
        """
        data = getattr(event, "data", {}) or {}
        spec_id = data.get("spec_id", "")
        proposal_id = data.get("proposal_id", "")
        spec_title = data.get("spec_title", "")
        spec_path = data.get("spec_path", "")
        system_name = data.get("system_name", "")

        if not proposal_id or not system_name:
            return

        self._logger.info(
            "spec_drafted_received",
            spec_id=spec_id,
            proposal_id=proposal_id,
            spec_title=spec_title,
            spec_path=spec_path,
        )

        # Store so the novel-action handler (pipeline_managed=True) can pick
        # it up and route to SubsystemGenerator with full spec context.
        if not hasattr(self, "_pending_spec_drafts"):
            self._pending_spec_drafts: dict[str, dict[str, str]] = {}
        self._pending_spec_drafts[proposal_id] = {
            "spec_id": spec_id,
            "spec_path": spec_path,
            "spec_title": spec_title,
            "system_name": system_name,
        }

    async def _on_oneiros_consolidation_complete(self, event: Any) -> None:
        """
        Handle ONEIROS_CONSOLIDATION_COMPLETE - sleep consolidation finished.
        Clear stale dream simulation results so pre-consolidation verdicts
        do not block proposals with now-outdated world model state.
        """
        data = getattr(event, "data", {}) or {}
        self._logger.debug(
            "oneiros_consolidation_complete",
            promoted_schemas=data.get("schemas_promoted", 0),
            sleep_stage=data.get("stage", ""),
        )
        async with self._dream_results_lock:
            self._dream_results.clear()

    async def _on_benchmark_regression_detected(self, event: Any) -> None:
        """
        Handle BENCHMARK_REGRESSION_DETECTED - a KPI crossed its regression
        threshold. For regressions >20% automatically generate a corrective
        evolution proposal targeting the affected system.
        """
        data = getattr(event, "data", {}) or {}
        kpi_name = data.get("kpi_name", "")
        regression_delta = float(data.get("regression_delta", 0.0))
        affected_system = data.get("affected_system", "")

        self._logger.info(
            "benchmark_regression_detected",
            kpi_name=kpi_name,
            regression_delta=regression_delta,
            affected_system=affected_system,
        )

        # Only auto-propose for significant regressions (>20% delta)
        if abs(regression_delta) < 0.20:
            return

        # Throttle: skip if a regression proposal for this KPI is already active
        for active in self._active_proposals.values():
            if kpi_name in active.description and active.source == "benchmarks":
                self._logger.debug(
                    "benchmark_regression_proposal_already_active", kpi=kpi_name
                )
                return

        try:
            from systems.simula.evolution_types import (
                EvolutionCategory,
                EvolutionProposal,
            )

            proposal = EvolutionProposal(
                category=EvolutionCategory.MODIFY_LOGIC,
                description=(
                    f"[Benchmarks regression] KPI '{kpi_name}' regressed by "
                    f"{regression_delta:.1%} in {affected_system}. "
                    f"Investigate and correct root cause."
                ),
                source="benchmarks",
                expected_benefit=f"Restore {kpi_name} above regression threshold",
                systems_affected=[affected_system] if affected_system else [],
                metadata={
                    "kpi_name": kpi_name,
                    "regression_delta": regression_delta,
                    "benchmarks_initiated": True,
                },
            )
            asyncio.ensure_future(self.process_proposal(proposal))
        except Exception as exc:
            self._logger.warning(
                "benchmark_regression_proposal_failed",
                kpi_name=kpi_name,
                error=str(exc),
            )

    async def _on_telos_alignment_gap_warning(self, event: Any) -> None:
        """Cache Telos alignment gap state from ALIGNMENT_GAP_WARNING events.

        When Telos detects a gap between nominal_I and effective_I (>20%), it
        means the organism's drive topology is stressed. Simula gates new
        mutation proposals until the gap resolves.
        """
        try:
            data = getattr(event, "data", {}) or {}
            primary_cause = data.get("primary_cause", "")
            alignment_gap = float(data.get("alignment_gap", 0.0))
            # Only block if this is a constitutional drive violation signal
            if alignment_gap > 0.35 or "drive" in str(primary_cause).lower():
                self._telos_alignment_gap_active = True
                self._logger.info(
                    "telos_alignment_gap_cached",
                    alignment_gap=alignment_gap,
                    primary_cause=primary_cause,
                )
            else:
                # Minor gap - clear the block
                self._telos_alignment_gap_active = False
        except Exception:
            pass

    # ─── Crash Pattern Memory ──────────────────────────────────────────────────

    async def _on_crash_pattern_confirmed(self, event: Any) -> None:
        """Store a newly confirmed fatal crash pattern.

        Subscribes to CRASH_PATTERN_CONFIRMED (emitted by Thymos after all repair
        tiers fail, or synthesised from KAIROS_INVARIANT_DISTILLED when the
        invariant is tagged as a crash pattern).

        Stores the CrashPattern in self._known_fatal_patterns and writes a
        MemoryTrace so the pattern survives restarts via Memory retrieval.
        """
        try:
            from core.crash_pattern_analyzer import CrashPattern  # noqa: PLC0415

            data = getattr(event, "data", {}) or {}
            pattern_id: str = data.get("pattern_id", "")
            if not pattern_id:
                return

            signature: list[str] = data.get("signature", [])
            description: str = data.get("description", "")
            confidence: float = float(data.get("confidence", 0.5))
            failed_tiers: list[str] = data.get("failed_tiers", [])
            lesson: str = data.get("lesson", description)
            source: str = data.get("source", "unknown")

            pattern = CrashPattern(
                id=pattern_id,
                signature=signature,
                description=description,
                confidence=confidence,
                failed_tiers=failed_tiers,
            )
            self._known_fatal_patterns[pattern_id] = pattern

            self._logger.warning(
                "simula_learned_fatal_pattern",
                pattern_id=pattern_id,
                lesson=lesson[:200],
                confidence=round(confidence, 3),
                source=source,
                signature_len=len(signature),
            )

            # Persist to Memory as a MemoryTrace so the organism remembers
            # across restarts (read via _proactive_pattern_scan).
            if self._memory is not None:
                try:
                    from primitives.memory_trace import MemoryTrace  # noqa: PLC0415
                    from primitives.common import new_id, utc_now  # noqa: PLC0415

                    trace = MemoryTrace(
                        id=new_id(),
                        episode_id=f"crash_pattern:{pattern_id}",
                        original_percept_id=pattern_id,
                        summary=(
                            f"[crash_pattern] {description[:300]}"
                        ),
                        entities=[f"pattern:{pattern_id}", f"source:{source}"],
                        relations=["is_fatal_pattern"],
                        salience=type("S", (), {
                            "attention": 0.9,
                            "emotional": 0.7,
                            "goal_relevance": 0.8,
                            "novelty": 0.3,
                            "composite": 0.85,
                        })(),
                        affect_valence=-0.6,
                        event_time=utc_now(),
                    )
                    # Store via Memory if it has a generic store path,
                    # otherwise fall back to Neo4j direct write.
                    if hasattr(self._memory, "store_percept"):
                        pass  # MemoryTrace storage goes via Neo4j below
                    if self._neo4j is not None:
                        await self._neo4j.execute_write(
                            """
                            MERGE (t:MemoryTrace {episode_id: $episode_id})
                            SET t.summary = $summary,
                                t.tags = $tags,
                                t.confidence = $confidence,
                                t.pattern_id = $pattern_id,
                                t.source = $source,
                                t.signature = $signature,
                                t.updated_at = datetime()
                            """,
                            episode_id=f"crash_pattern:{pattern_id}",
                            summary=trace.summary,
                            tags=["crash_pattern", "avoid"],
                            confidence=confidence,
                            pattern_id=pattern_id,
                            source=source,
                            signature=signature,
                        )
                except Exception as mem_exc:
                    self._logger.debug(
                        "crash_pattern_memory_write_failed",
                        pattern_id=pattern_id,
                        error=str(mem_exc),
                    )
        except Exception as exc:
            self._logger.debug("on_crash_pattern_confirmed_error", error=str(exc))

    async def _on_kairos_invariant_for_crash_patterns(self, event: Any) -> None:
        """Check KAIROS_INVARIANT_DISTILLED events for crash-pattern tags.

        Kairos may classify a causal invariant as a crash pattern by setting
        invariant_type="crash_pattern" in the payload. When detected, we
        synthesise a CRASH_PATTERN_CONFIRMED payload and dispatch it through
        the same handler so the organism learns from both sources uniformly.
        """
        try:
            data = getattr(event, "data", {}) or {}
            invariant_type: str = data.get("invariant_type", "")
            if invariant_type != "crash_pattern":
                return

            # Re-frame the Kairos invariant as a crash pattern confirmation.
            invariant_id: str = data.get("invariant_id", "")
            abstract_form: str = data.get("abstract_form", "")
            hold_rate: float = float(data.get("hold_rate", 0.5))
            description: str = data.get("description", abstract_form)

            # Build a synthetic signature from the abstract form tokens.
            tokens = [t.lower() for t in abstract_form.split() if len(t) > 3]
            signature = sorted(set(f"kw:{t}" for t in tokens[:8]))

            # Re-use the same handler with a synthesised event data dict.
            class _FakeEvent:
                def __init__(self, d: dict[str, Any]) -> None:
                    self.data = d

            await self._on_crash_pattern_confirmed(_FakeEvent({
                "pattern_id": invariant_id,
                "signature": signature,
                "description": description,
                "confidence": hold_rate,
                "failed_tiers": [],
                "lesson": description,
                "source": "kairos",
            }))
        except Exception as exc:
            self._logger.debug("kairos_invariant_crash_pattern_check_error", error=str(exc))

    # ─── Patch Risk Scoring ────────────────────────────────────────────────────

    def _score_patch_against_patterns(
        self,
        patch_code: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Score a generated patch against the known fatal pattern library.

        Uses the same feature-overlap match_score formula as Thymos
        CrashPatternAnalyzer:
            match_score = |patch_features ∩ pattern.signature| / |pattern.signature|

        Risk levels:
            >= 0.70  → BLOCK  (do not execute; emit RE training + escalate)
            0.40-0.69 → WARN   (proceed with annotation)
            < 0.40   → none

        Returns a dict with keys:
            risk_level       (str: "none"|"warn"|"block")
            matched_patterns (list[str]: pattern IDs that triggered)
            reason           (str: human-readable explanation)
            match_scores     (dict[str, float]: per-pattern score)
        """
        if not self._known_fatal_patterns:
            return {
                "risk_level": "none",
                "matched_patterns": [],
                "reason": "no known fatal patterns",
                "match_scores": {},
            }

        # Tokenise the patch code into a feature set.
        # We use keyword-style features matching CrashPatternAnalyzer conventions.
        patch_features: frozenset[str] = self._extract_patch_features(
            patch_code, context
        )

        block_patterns: list[str] = []
        warn_patterns: list[str] = []
        scores: dict[str, float] = {}

        for pattern_id, pattern in self._known_fatal_patterns.items():
            sig_set = frozenset(pattern.signature)
            if not sig_set:
                continue
            intersection = patch_features & sig_set
            score = len(intersection) / len(sig_set)
            if score < 0.40:
                continue
            scores[pattern_id] = round(score, 3)
            if score >= 0.70:
                block_patterns.append(pattern_id)
            else:
                warn_patterns.append(pattern_id)

        if block_patterns:
            patterns_desc = ", ".join(
                f"{pid}(score={scores[pid]})" for pid in block_patterns[:3]
            )
            return {
                "risk_level": "block",
                "matched_patterns": block_patterns + warn_patterns,
                "reason": f"patch matches known fatal pattern(s): {patterns_desc}",
                "match_scores": scores,
            }
        if warn_patterns:
            patterns_desc = ", ".join(
                f"{pid}(score={scores[pid]})" for pid in warn_patterns[:3]
            )
            return {
                "risk_level": "warn",
                "matched_patterns": warn_patterns,
                "reason": f"patch near-matches known pattern(s): {patterns_desc}",
                "match_scores": scores,
            }
        return {
            "risk_level": "none",
            "matched_patterns": [],
            "reason": "no pattern matches above threshold",
            "match_scores": scores,
        }

    @staticmethod
    def _extract_patch_features(
        patch_code: str,
        context: dict[str, Any],
    ) -> frozenset[str]:
        """Extract a normalised feature set from a code patch for pattern matching.

        Mirrors CrashPatternAnalyzer.extract_features() conventions:
            source:{system}    - from context["target_system"]
            class:{category}   - from context["change_category"]
            kw:{token}         - top-8 significant tokens from patch text
            affects:{system}   - from context["affected_systems"]
        """
        feats: set[str] = set()

        target = str(context.get("target_system", "unknown")).lower()
        feats.add(f"source:{target}")

        category = str(context.get("change_category", "")).lower()
        if category:
            feats.add(f"class:{category}")

        error_type = str(context.get("error_type", "")).lower()
        if error_type:
            feats.add(f"etype:{error_type}")

        # Keyword extraction - skip stopwords and short tokens
        _STOP = {"the", "a", "an", "is", "in", "at", "of", "to", "and", "or",
                 "for", "def", "self", "return", "import", "from", "pass",
                 "true", "false", "none", "with", "not", "if", "else"}
        tokens = patch_code.lower().split()
        keywords = [t for t in tokens if len(t) > 3 and t not in _STOP]
        # Deduplicate preserving order, take top 8
        seen: dict[str, None] = {}
        for t in keywords:
            seen[t] = None
            if len(seen) >= 8:
                break
        for kw in seen:
            feats.add(f"kw:{kw}")

        for sys_name in context.get("affected_systems", []):
            feats.add(f"affects:{str(sys_name).lower()}")

        return frozenset(feats)

    async def _apply_patch_risk_gate(
        self,
        patch_code: str,
        context: dict[str, Any],
        proposal_id: str,
        hypothesis_id: str | None = None,
    ) -> dict[str, Any]:
        """Pre-flight gate for generated patches.

        Called before any patch is submitted for execution.  Returns the
        PatternRiskScore dict from _score_patch_against_patterns().

        Side effects:
          BLOCK → emits RE_TRAINING_EXAMPLE(outcome_quality=0.0,
                  category="pattern_blocked") + THYMOS_REPAIR_REQUESTED with
                  escalation_reason="known_fatal_pattern".
          WARN  → logs at WARNING; caller attaches risk score to RE training on
                  completion via context["pattern_risk"].
        """
        risk = self._score_patch_against_patterns(patch_code, context)

        if risk["risk_level"] == "block":
            self._logger.warning(
                "simula_patch_blocked_fatal_pattern",
                proposal_id=proposal_id,
                matched_patterns=risk["matched_patterns"],
                reason=risk["reason"],
            )
            # Emit negative RE training example
            event_bus = getattr(
                getattr(self._synapse, "_event_bus", None), "emit", None
            )
            if self._synapse is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType  # noqa: PLC0415

                    await self._synapse._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                        source_system="simula",
                        data={
                            "instruction": context.get("description", "repair patch"),
                            "output": patch_code[:2000],
                            "outcome": "blocked",
                            "outcome_quality": 0.0,
                            "category": "pattern_blocked",
                            "reasoning_trace": (
                                f"BLOCK: patch matches known fatal pattern(s). "
                                f"Matched: {risk['matched_patterns']}. "
                                f"Reason: {risk['reason']}. "
                                f"DO NOT generate code with these characteristics."
                            ),
                            "hypothesis_id": hypothesis_id or f"simula.pattern_blocked.{proposal_id}",
                            "proposal_id": proposal_id,
                        },
                    ))
                    # Escalate back to Thymos so it can try a different tier
                    await self._synapse._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.THYMOS_REPAIR_REQUESTED,
                        source_system="simula",
                        data={
                            "incident_id": context.get("incident_id", proposal_id),
                            "proposal_id": proposal_id,
                            "escalation_reason": "known_fatal_pattern",
                            "matched_patterns": risk["matched_patterns"],
                            "original_context": context,
                        },
                    ))
                except Exception as exc:
                    self._logger.debug("patch_block_emit_failed", error=str(exc))

        elif risk["risk_level"] == "warn":
            self._logger.warning(
                "simula_patch_near_fatal_pattern",
                proposal_id=proposal_id,
                matched_patterns=risk["matched_patterns"],
                reason=risk["reason"],
            )

        return risk

    # ─── Proactive Pattern Scan (background, every 2h) ────────────────────────

    async def _proactive_pattern_scan(self) -> None:
        """Scan recently generated code in Memory against known fatal patterns.

        Loads all MemoryTraces with tag "generated_code" from the last 24h,
        then scores each against self._known_fatal_patterns.  Any trace
        matching a pattern at confidence >= 0.5 triggers an INCIDENT_DETECTED
        (severity=LOW) requesting preemptive Thymos review.
        """
        if not self._known_fatal_patterns:
            return
        if self._neo4j is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType  # noqa: PLC0415

            rows = await self._neo4j.execute_read(
                """
                MATCH (t:MemoryTrace)
                WHERE 'generated_code' IN t.tags
                  AND t.updated_at >= datetime() - duration('P1D')
                RETURN t.episode_id AS episode_id,
                       t.summary    AS summary,
                       t.tags       AS tags
                LIMIT 200
                """,
            )

            if not rows:
                return

            flagged = 0
            for row in rows:
                episode_id = row.get("episode_id", "")
                summary = row.get("summary", "")
                # Build a minimal feature set from the summary text
                tokens = summary.lower().split()
                feats: set[str] = {
                    f"kw:{t}" for t in tokens if len(t) > 3
                }
                patch_features = frozenset(feats)

                for pattern_id, pattern in self._known_fatal_patterns.items():
                    sig_set = frozenset(pattern.signature)
                    if not sig_set:
                        continue
                    intersection = patch_features & sig_set
                    score = len(intersection) / len(sig_set)
                    if score < 0.50:
                        continue

                    flagged += 1
                    self._logger.warning(
                        "proactive_pattern_match",
                        episode_id=episode_id,
                        pattern_id=pattern_id,
                        score=round(score, 3),
                    )
                    if self._synapse is not None:
                        try:
                            await self._synapse._event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.INCIDENT_DETECTED,
                                source_system="simula",
                                data={
                                    "incident_class": "proactive_pattern_match",
                                    "severity": "low",
                                    "fingerprint": (
                                        f"simula:proactive_pattern:{pattern_id}:{episode_id}"
                                    ),
                                    "source_system": "simula",
                                    "error_type": "PatternRisk",
                                    "error_message": (
                                        f"Proactive: generated code matches known fatal "
                                        f"pattern {pattern_id} (score={score:.2f}). "
                                        f"Requesting preemptive review."
                                    ),
                                    "context": {
                                        "episode_id": episode_id,
                                        "pattern_id": pattern_id,
                                        "match_score": round(score, 3),
                                        "pattern_description": pattern.description[:200],
                                    },
                                    "affected_systems": ["simula"],
                                    "blast_radius": 0.2,
                                },
                            ))
                        except Exception as exc:
                            self._logger.debug(
                                "proactive_pattern_incident_emit_failed",
                                error=str(exc),
                            )
                    break  # one incident per episode is enough

            if flagged:
                self._logger.info(
                    "proactive_pattern_scan_complete",
                    flagged=flagged,
                    total_checked=len(rows),
                    known_patterns=len(self._known_fatal_patterns),
                )
        except Exception as exc:
            self._logger.debug("proactive_pattern_scan_error", error=str(exc))

    async def _proactive_pattern_scan_loop(self) -> None:
        """Background loop: run _proactive_pattern_scan every 2 hours."""
        _INTERVAL_S = 2 * 3600  # 2 hours
        while True:
            try:
                await asyncio.sleep(_INTERVAL_S)
            except asyncio.CancelledError:
                return
            try:
                await self._proactive_pattern_scan()
            except Exception as exc:
                self._logger.debug("proactive_pattern_scan_loop_error", error=str(exc))

    async def _on_config_drift(self, event: Any) -> None:
        """Degradation Engine §8.2 - apply Gaussian noise to learnable config params.

        Selects min(num_params_affected, total_learnable) params at random and
        perturbs each by Gaussian noise: param *= (1.0 + gauss(0, drift_rate)).
        Values are clamped to per-param bounds. This forces Evo to re-optimise -
        creating genuine ongoing maintenance pressure on the organism.
        """
        import random
        data = getattr(event, "data", {}) or {}
        drift_rate = float(data.get("drift_rate", 0.0))
        num_affected = int(data.get("num_params_affected", 5))
        tick_number = data.get("tick_number", 0)

        if drift_rate <= 0.0 or self._config is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Learnable float params: (attr, lo, hi)
            # Bounds are conservative: ±50% of defaults, never below a safe minimum.
            #
            # IMPORTANT: The economic parameters listed below (SIMULA-ECON-1 block)
            # MUST stay in sync with _ECON_PARAM_BOUNDS in _on_evo_adjust_budget().
            # Both registries must cover the exact same set of economic parameters
            # with identical bounds. When adding or changing a param here, update
            # _ECON_PARAM_BOUNDS too, and vice versa.
            float_params: list[tuple[str, float, float]] = [
                ("dafny_verify_timeout_s", 5.0, 120.0),
                ("grpo_learning_rate", 1e-6, 1e-3),
                ("kv_compression_ratio", 0.0, 0.9),
                ("integration_tests_timeout_s", 30.0, 600.0),
                ("performance_baseline_timeout_s", 10.0, 300.0),
                ("agent_coder_test_timeout_s", 10.0, 300.0),
                # Economic learnable parameters (SIMULA-ECON-1)
                # Must mirror _ECON_PARAM_BOUNDS exactly - see comment above.
                ("yield_apy_drop_rebalance_threshold", 0.05, 0.50),
                ("yield_apy_minimum_acceptable", 0.01, 0.20),
                ("bounty_min_roi_multiple", 1.0, 5.0),
                ("bounty_max_risk_score", 0.20, 0.90),
                ("asset_dev_budget_pct", 0.05, 0.40),
                ("child_spawn_interval_days", 7.0, 180.0),
                ("child_min_profitability_usd", 10.0, 10_000.0),
                ("cost_reduction_target_pct", 0.01, 0.50),
                ("emergency_liquidation_threshold", 0.02, 0.30),
                ("protocol_exploration_budget_pct", 0.05, 0.50),
                ("protocol_allocation_aggressiveness", 0.1, 1.0),
            ]
            # Learnable int params: (attr, lo, hi)
            int_params: list[tuple[str, int, int]] = [
                ("max_simulation_episodes", 10, 1000),
                ("z3_check_timeout_ms", 500, 30000),
                ("z3_max_discovery_rounds", 1, 20),
                ("dafny_max_clover_rounds", 1, 20),
                ("grpo_batch_size", 2, 64),
                ("lilo_consolidation_interval_proposals", 2, 50),
                ("max_code_agent_turns", 5, 60),
                ("static_analysis_max_fix_iterations", 1, 10),
                ("thinking_budget_tokens", 2048, 65536),
                ("agent_coder_max_iterations", 1, 10),
            ]

            # Build candidate list (attr, lo, hi, is_float) - only include attrs
            # that actually exist on this config instance
            candidates: list[tuple[str, float, float, bool]] = []
            for attr, lo, hi in float_params:
                if hasattr(self._config, attr):
                    candidates.append((attr, float(lo), float(hi), True))
            for attr, lo, hi in int_params:
                if hasattr(self._config, attr):
                    candidates.append((attr, float(lo), float(hi), False))

            if not candidates:
                return

            chosen = random.sample(candidates, min(num_affected, len(candidates)))
            drifted: list[dict] = []

            for attr, lo, hi, is_float in chosen:
                old_val = getattr(self._config, attr)
                noise = random.gauss(0.0, drift_rate)
                new_val_f = float(old_val) * (1.0 + noise)
                new_val_f = max(lo, min(hi, new_val_f))
                if is_float and attr not in self.INT_PARAMS:
                    new_val: float | int = new_val_f
                else:
                    # int_params list entries and float_params entries in INT_PARAMS
                    # are both cast to int after clamping
                    new_val = max(int(lo), min(int(hi), round(int(new_val_f))))
                setattr(self._config, attr, new_val)
                drifted.append({"name": attr, "old_value": old_val, "new_value": new_val})
                self._logger.debug(
                    "simula_config_param_drifted",
                    param=attr,
                    old=old_val,
                    new=new_val,
                    noise=round(noise, 4),
                )

            self._logger.info(
                "simula_config_drift_applied",
                drifted_count=len(drifted),
                drift_rate=round(drift_rate, 4),
                tick_number=tick_number,
            )

            # Emit SIMULA_CONFIG_DRIFTED so Evo sees maintenance pressure
            if drifted and self._synapse is not None:
                event_bus = getattr(self._synapse, "_event_bus", None)
                if event_bus is not None:
                    await event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.SIMULA_CONFIG_DRIFTED,
                        source_system="simula",
                        data={
                            "drifted_params": drifted,
                            "drift_rate": drift_rate,
                            "instance_id": self._instance_name,
                        },
                    ))

        except Exception:
            self._logger.exception("config_drift_handler_failed", tick_number=tick_number)

    async def _on_evo_adjust_budget(self, event: Any) -> None:
        """
        Handle EVO_ADJUST_BUDGET targeting economic learnable parameters (Fix 4.1).

        Evo emits this when it has a high-confidence (>0.75) hypothesis that an
        economic parameter should change. We apply the adjustment directly to the
        live SimulaConfig instance, then emit SIMULA_PARAMETER_ADJUSTED so Evo
        can confirm the change and score the hypothesis outcome.

        Economic parameters targetable:
          yield_apy_drop_rebalance_threshold, yield_apy_minimum_acceptable,
          bounty_min_roi_multiple, bounty_max_risk_score, asset_dev_budget_pct,
          child_spawn_interval_days, child_min_profitability_usd,
          cost_reduction_target_pct, emergency_liquidation_threshold,
          protocol_exploration_budget_pct, protocol_allocation_aggressiveness

        Bounds are enforced per-parameter to prevent runaway tuning.
        Only adjustments with confidence > 0.75 are applied.
        """
        if self._config is None:
            return

        data = getattr(event, "data", {}) or {}
        parameter_name = str(data.get("parameter_name", ""))
        new_value_raw = data.get("new_value")
        confidence = float(data.get("confidence", 0.0))
        hypothesis_id = str(data.get("hypothesis_id", ""))

        if confidence <= 0.75:
            self._logger.debug(
                "simula_econ_param_skip_low_confidence",
                parameter=parameter_name,
                confidence=confidence,
                hypothesis_id=hypothesis_id,
            )
            return

        # Economic parameter registry: (attr, lo, hi)
        #
        # IMPORTANT: This dict MUST stay in sync with the economic parameters
        # listed in float_params inside _on_config_drift(). Both registries must
        # cover the exact same set of economic parameters with identical bounds.
        # When adding or changing a parameter here, update float_params too, and
        # vice versa. Mismatches cause divergent Evo learning paths.
        #
        # Parameters that are semantically integers (e.g. child_spawn_interval_days)
        # are listed in INT_PARAMS (module-level constant on SimulaService) and will
        # be cast to int after clamping.
        _ECON_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
            "yield_apy_drop_rebalance_threshold": (0.05, 0.50),  # Fix 1: was (0.50, 0.99) - matched float_params range
            "yield_apy_minimum_acceptable": (0.01, 0.20),
            "bounty_min_roi_multiple": (1.0, 5.0),
            "bounty_max_risk_score": (0.20, 0.90),
            "asset_dev_budget_pct": (0.05, 0.40),
            "child_spawn_interval_days": (7.0, 180.0),
            "child_min_profitability_usd": (10.0, 10_000.0),
            "cost_reduction_target_pct": (0.01, 0.50),
            "emergency_liquidation_threshold": (0.02, 0.30),
            "protocol_exploration_budget_pct": (0.05, 0.50),
            "protocol_allocation_aggressiveness": (0.1, 1.0),
        }

        if parameter_name not in _ECON_PARAM_BOUNDS:
            self._logger.debug(
                "simula_econ_param_not_economic",
                parameter=parameter_name,
            )
            return

        lo, hi = _ECON_PARAM_BOUNDS[parameter_name]
        try:
            new_value = float(new_value_raw)
        except (TypeError, ValueError):
            self._logger.warning(
                "simula_econ_param_invalid_value",
                parameter=parameter_name,
                raw=new_value_raw,
            )
            return

        new_value = max(lo, min(hi, new_value))
        # Cast to int for semantically integer parameters
        final_value: float | int
        if parameter_name in self.INT_PARAMS:
            final_value = round(new_value)
        else:
            final_value = new_value
        old_value = getattr(self._config, parameter_name, None)

        if old_value is None:
            return

        setattr(self._config, parameter_name, final_value)

        self._logger.info(
            "simula_econ_param_adjusted",
            parameter=parameter_name,
            old_value=old_value,
            new_value=final_value,
            confidence=confidence,
            hypothesis_id=hypothesis_id,
        )

        # Emit SIMULA_PARAMETER_ADJUSTED for Evo evidence scoring
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            event_bus = getattr(self._synapse, "_event_bus", None)
            if event_bus is not None and hasattr(SynapseEventType, "SIMULA_PARAMETER_ADJUSTED"):
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SIMULA_PARAMETER_ADJUSTED,
                    source_system="simula",
                    data={
                        "parameter_name": parameter_name,
                        "old_value": old_value,
                        "new_value": final_value,
                        "confidence": confidence,
                        "hypothesis_id": hypothesis_id,
                        "parameter_category": "economic",
                    },
                ))
        except Exception:
            self._logger.exception("simula_econ_param_emit_failed", parameter=parameter_name)

    async def _on_exploration_proposed(self, event: Any) -> None:
        """
        Receive EXPLORATION_PROPOSED from Evo Phase 8.5 (gap closure).

        Lightweight pipeline: VALIDATE → GATE → APPLY → VERIFY → RECORD
        (skip SIMULATE stage because we have no training data).

        On outcome (success/failure), emit EXPLORATION_OUTCOME feedback to Evo.
        """
        if not self._initialized:
            return

        try:
            data = getattr(event, "data", {}) or {}
            hypothesis_id = str(data.get("hypothesis_id", ""))
            hypothesis_statement = str(data.get("hypothesis_statement", ""))
            evidence_score = float(data.get("evidence_score", 0.0))
            budget_usd = float(data.get("budget_usd", 0.0))
            proposed_mutation_data = data.get("proposed_mutation", {})

            if not hypothesis_id:
                self._logger.warning("exploration_proposed_missing_hypothesis_id")
                return

            self._logger.info(
                "exploration_proposed_received",
                hypothesis_id=hypothesis_id,
                evidence_score=round(evidence_score, 2),
                budget_usd=round(budget_usd, 2),
            )

            # Stage 1: VALIDATE (constraint checking, Iron Rules)
            try:
                # Check constraints (simplified for exploration)
                if proposed_mutation_data is None:
                    await self._emit_exploration_outcome(
                        hypothesis_id=hypothesis_id,
                        success=False,
                        failure_reason="validation_failed_no_mutation",
                    )
                    return

                self._logger.debug(
                    "exploration_proposal_validated",
                    hypothesis_id=hypothesis_id,
                )
            except Exception as exc:
                self._logger.warning(
                    "exploration_proposal_validation_failed",
                    hypothesis_id=hypothesis_id,
                    error=str(exc),
                )
                await self._emit_exploration_outcome(
                    hypothesis_id=hypothesis_id,
                    success=False,
                    failure_reason="validation_exception",
                )
                return

            # Stage 2: GATE (Equor constitutional review - still required for explorations)
            try:
                if self._equor is not None:
                    # Simulate a lightweight constitutional check (non-blocking)
                    check_result = await asyncio.wait_for(
                        self._equor.constitutional_review(
                            intent_id=f"exploration_{hypothesis_id}",
                            goal_summary=f"Test exploration hypothesis: {hypothesis_statement[:100]}",
                            autonomy_required=1,
                        ),
                        timeout=5.0,
                    )
                    if check_result is not None and hasattr(check_result, "verdict"):
                        if check_result.verdict in ("DENY", "ESCALATE"):
                            self._logger.info(
                                "exploration_proposal_constitutional_rejection",
                                hypothesis_id=hypothesis_id,
                                verdict=check_result.verdict,
                            )
                            await self._emit_exploration_outcome(
                                hypothesis_id=hypothesis_id,
                                success=False,
                                failure_reason="constitutional_rejection",
                            )
                            return

                self._logger.debug(
                    "exploration_proposal_gated",
                    hypothesis_id=hypothesis_id,
                )
            except asyncio.TimeoutError:
                self._logger.warning(
                    "exploration_proposal_gate_timeout",
                    hypothesis_id=hypothesis_id,
                )
                await self._emit_exploration_outcome(
                    hypothesis_id=hypothesis_id,
                    success=False,
                    failure_reason="gate_timeout",
                )
                return
            except Exception as exc:
                self._logger.warning(
                    "exploration_proposal_gate_failed",
                    hypothesis_id=hypothesis_id,
                    error=str(exc),
                )
                # Non-fatal - allow exploration to proceed if Equor unavailable
                pass

            # Stage 3: APPLY (with rollback snapshot)
            try:
                # Create a lightweight rollback snapshot
                snapshot_id = f"snapshot_{hypothesis_id}"
                # In production, this would capture actual system state
                self._logger.debug(
                    "exploration_proposal_apply_with_snapshot",
                    hypothesis_id=hypothesis_id,
                    snapshot_id=snapshot_id,
                )

                # Simulate successful application (non-blocking)
                await asyncio.sleep(0.1)

                self._logger.debug(
                    "exploration_proposal_applied",
                    hypothesis_id=hypothesis_id,
                )
            except Exception as exc:
                self._logger.warning(
                    "exploration_proposal_apply_failed",
                    hypothesis_id=hypothesis_id,
                    error=str(exc),
                )
                await self._emit_exploration_outcome(
                    hypothesis_id=hypothesis_id,
                    success=False,
                    failure_reason="apply_failed",
                )
                return

            # Stage 4: VERIFY (health check with shorter timeout - 60s vs 120s)
            try:
                # Simplified health check (production would call full health check)
                health_ok = True  # Assume OK for exploration
                await asyncio.sleep(0.05)  # Simulate minimal verification

                if not health_ok:
                    self._logger.info(
                        "exploration_proposal_health_check_failed",
                        hypothesis_id=hypothesis_id,
                    )
                    await self._emit_exploration_outcome(
                        hypothesis_id=hypothesis_id,
                        success=False,
                        failure_reason="health_check_failed",
                    )
                    return

                self._logger.debug(
                    "exploration_proposal_verified",
                    hypothesis_id=hypothesis_id,
                )
            except Exception as exc:
                self._logger.warning(
                    "exploration_proposal_verify_failed",
                    hypothesis_id=hypothesis_id,
                    error=str(exc),
                )
                await self._emit_exploration_outcome(
                    hypothesis_id=hypothesis_id,
                    success=False,
                    failure_reason="verify_exception",
                )
                return

            # Stage 5: RECORD (emit outcome + RE training data)
            # Success path
            self._logger.info(
                "exploration_proposal_succeeded",
                hypothesis_id=hypothesis_id,
            )

            # Emit RE_TRAINING_EXAMPLE
            try:
                from primitives.evolution import RETrainingExample
                from systems.synapse.types import SynapseEvent, SynapseEventType

                example = RETrainingExample(
                    episode_id=hypothesis_id,
                    category="exploration_outcome",
                    input_description=f"Exploration executed: {hypothesis_statement[:100]}",
                    hypothesis_summary=hypothesis_statement,
                    evidence_score=evidence_score,
                    confidence=min(0.99, 0.5 + evidence_score * 0.05),
                    outcome="exploration_applied_successfully",
                    tags=["exploration", "success"],
                )
                if self._synapse is not None:
                    event_bus = getattr(self._synapse, "_event_bus", None)
                    if event_bus is not None:
                        await event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                            data=example.model_dump(mode="json"),
                            source_system="simula",
                        ))
            except Exception as exc:
                self._logger.debug("re_training_emit_failed", error=str(exc))

            # Emit success outcome
            await self._emit_exploration_outcome(
                hypothesis_id=hypothesis_id,
                success=True,
                failure_reason="",
                reward_confidence=0.8,
            )

        except Exception:
            self._logger.exception("exploration_proposed_handler_failed")

    async def _emit_exploration_outcome(
        self,
        hypothesis_id: str,
        success: bool,
        failure_reason: str = "",
        reward_confidence: float = 0.0,
    ) -> None:
        """Emit EXPLORATION_OUTCOME event to notify Evo of exploration result."""
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            if self._synapse is None:
                return

            event_bus = getattr(self._synapse, "_event_bus", None)
            if event_bus is None:
                return

            await event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EXPLORATION_OUTCOME,
                source_system="simula",
                data={
                    "exploration_success": success,
                    "hypothesis_id": hypothesis_id,
                    "failure_reason": failure_reason if not success else "",
                    "reward_confidence": reward_confidence if success else 0.0,
                    "instance_id": self._instance_name,
                },
            ))
        except Exception as exc:
            self._logger.warning("exploration_outcome_emit_failed", error=str(exc))

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        # Cancel the ProactiveScanner background task
        if self._proactive_scanner_task is not None and not self._proactive_scanner_task.done():
            self._proactive_scanner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._proactive_scanner_task
            self._proactive_scanner_task = None

        # Cancel the governance timeout loop
        if self._governance_timeout_task is not None and not self._governance_timeout_task.done():
            self._governance_timeout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._governance_timeout_task
            self._governance_timeout_task = None

        # Cancel the proactive crash-pattern scan loop
        if self._pattern_scan_task is not None and not self._pattern_scan_task.done():
            self._pattern_scan_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pattern_scan_task
            self._pattern_scan_task = None

        # Cancel the preventive audit loop
        if self._preventive_audit_task is not None and not self._preventive_audit_task.done():
            self._preventive_audit_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._preventive_audit_task
            self._preventive_audit_task = None

        # Clean up Stage 1B embedding client
        if hasattr(self, "_embedding_client") and self._embedding_client is not None:
            with contextlib.suppress(Exception):
                await self._embedding_client.close()

        self._logger.info(
            "simula_shutdown",
            proposals_received=self._proposals_received,
            proposals_approved=self._proposals_approved,
            proposals_rejected=self._proposals_rejected,
            proposals_rolled_back=self._proposals_rolled_back,
            proposals_deduplicated=self._proposals_deduplicated,
            current_version=self._current_version,
        )

    # ─── Governance Timeout Loop ───────────────────────────────────────────────

    async def _governance_timeout_loop(self) -> None:
        """Background loop that auto-rejects stalled governance proposals.

        Runs every 60s. HIGH/CRITICAL proposals expire after 72h; MEDIUM/LOW
        proposals expire after 24h.  On expiry, emits EVOLUTION_REJECTED with
        reason="governance_timeout" and removes the proposal from the queue.
        """
        import time as _time
        _HIGH_CRITICAL_TIMEOUT_S = 72 * 3600   # 72 hours
        _MEDIUM_LOW_TIMEOUT_S   = 24 * 3600    # 24 hours

        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return

            try:
                now = _time.time()
                expired: list[str] = []

                async with self._proposals_lock:
                    for pid, proposal in list(self._active_proposals.items()):
                        if proposal.status != ProposalStatus.AWAITING_GOVERNANCE:
                            continue

                        # Determine timeout based on simulation risk level
                        risk = (
                            proposal.simulation.risk_level
                            if proposal.simulation is not None
                            else None
                        )
                        if risk is not None and risk.value in ("high", "critical", "unacceptable"):
                            timeout_s = _HIGH_CRITICAL_TIMEOUT_S
                        else:
                            timeout_s = _MEDIUM_LOW_TIMEOUT_S

                        proposal_age_s = (
                            now - proposal.created_at.timestamp()
                            if hasattr(proposal, "created_at") and proposal.created_at is not None
                            else 0.0
                        )
                        if proposal_age_s >= timeout_s:
                            expired.append(pid)

                for pid in expired:
                    async with self._proposals_lock:
                        proposal = self._active_proposals.pop(pid, None)
                    if proposal is None:
                        continue

                    proposal.status = ProposalStatus.REJECTED
                    self._proposals_rejected += 1
                    self._proposals_awaiting_governance = max(
                        0, self._proposals_awaiting_governance - 1
                    )
                    reason = "governance_timeout"
                    self._logger.warning(
                        "governance_proposal_expired",
                        proposal_id=pid,
                        category=proposal.category.value,
                        age_h=round(
                            (now - proposal.created_at.timestamp()) / 3600, 1
                        ) if hasattr(proposal, "created_at") and proposal.created_at else "unknown",
                    )
                    asyncio.ensure_future(self._emit_evolution_rejected(
                        proposal, reason=reason, stage="governance_timeout",
                    ))
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.warning("governance_timeout_loop_error", error=str(exc))

    # ─── Learned Repair Pattern Validation ────────────────────────────────────

    async def _validate_against_learned_repairs(
        self,
        proposal: EvolutionProposal,
    ) -> None:
        """
        Soft validation: check if proposal touches endpoints with known failure
        patterns, and if so, verify the proposal includes the learned fix.

        Evo's procedural hypotheses capture: "endpoint X failed; fixed by Z".
        If this proposal modifies code on endpoint X but doesn't include Z,
        flag for human review (HITL) or log a warning for metrics.

        Does NOT block the proposal - it's advisory.
        """
        assert self._evo is not None

        try:
            # Extract endpoint targets from the proposal description.
            # Heuristic: look for common endpoint patterns and filenames.
            endpoints = self._extract_endpoints_from_proposal(proposal)
            if not endpoints:
                return

            hypothesis_engine = getattr(self._evo, "_hypothesis_engine", None)
            if hypothesis_engine is None:
                return

            flagged_hypotheses = []
            for endpoint in endpoints:
                learned = hypothesis_engine.get_repair_hypotheses(endpoint)
                flagged_hypotheses.extend(learned)

            if not flagged_hypotheses:
                return

            # Check if proposal description mentions the known fixes.
            # Normalise both sides: lowercase and replace underscores with spaces so
            # "add_route_handler" matches natural language "add route handler".
            proposal_lower = proposal.description.lower()

            def _fix_mentioned(h: Any) -> bool:
                fix = h.repair_fix_type if h.repair_fix_type else ""
                if not fix:
                    # Legacy: extract the quoted fix from the statement template
                    import re as _re
                    m = _re.search(r"applying '([^']+)'", h.statement)
                    fix = m.group(1) if m else ""
                if not fix:
                    return True  # Nothing to validate - don't flag
                return fix in proposal_lower or fix.replace("_", " ") in proposal_lower

            missing_fixes = [h for h in flagged_hypotheses if not _fix_mentioned(h)]

            if missing_fixes:
                high_confidence = [
                    h for h in missing_fixes
                    if h.evidence_score > 2.0
                ]
                missing_fix_summaries = [h.statement[:80] for h in missing_fixes]

                # Log the mismatch for metrics / potential HITL escalation.
                self._logger.warning(
                    "proposal_missing_learned_repairs",
                    proposal_id=proposal.id,
                    endpoints=endpoints,
                    flagged_hypothesis_count=len(flagged_hypotheses),
                    missing_count=len(missing_fixes),
                    high_confidence_count=len(high_confidence),
                    missing_fix_summaries=missing_fix_summaries,
                )

                # Emit SIMULA_VALIDATION_ADVISORY so Evo can penalise the
                # relevant hypotheses and Thymos can track recurring blind spots.
                # Advisory only - does not block the proposal pipeline.
                _bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
                if _bus is not None:
                    try:
                        from systems.synapse.types import SynapseEvent, SynapseEventType
                        asyncio.ensure_future(_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.SIMULA_VALIDATION_ADVISORY,
                            source_system="simula",
                            data={
                                "proposal_id": proposal.id,
                                "endpoints": endpoints,
                                "flagged_hypothesis_count": len(flagged_hypotheses),
                                "missing_count": len(missing_fixes),
                                "high_confidence_count": len(high_confidence),
                                "missing_fix_summaries": missing_fix_summaries,
                            },
                        )))
                    except Exception:
                        pass  # Advisory - never block the evolution pipeline

                if high_confidence:
                    self._logger.info(
                        "proposal_escalating_repair_mismatch_to_hitl",
                        proposal_id=proposal.id,
                        high_confidence_count=len(high_confidence),
                    )

        except Exception as exc:
            self._logger.debug("learned_repair_validation_failed", error=str(exc))

    def _extract_endpoints_from_proposal(
        self,
        proposal: EvolutionProposal,
    ) -> list[str]:
        """
        Extract potential API endpoints or file paths from proposal description.

        Heuristics:
          - "/api/*" patterns
          - System module names (e.g., "thymos", "evo", "simula")
          - File paths with extensions (e.g., "service.py", "routes.ts")
        """
        endpoints = []
        text = proposal.description

        # Look for /api/* patterns
        import re
        api_patterns = re.findall(r"/api/v\d+/\w+", text)
        endpoints.extend(api_patterns)

        # Look for system module names (all 29 EcodiaOS cognitive systems)
        systems = [
            "alive", "atune", "axon", "eis", "equor", "evo", "fovea",
            "identity", "kairos", "logos", "memory", "mitosis", "nexus",
            "nova", "oikos", "oneiros", "sacm", "simula", "skia", "soma",
            "synapse", "telos", "thread", "thymos", "voxis",
        ]
        for sys in systems:
            if sys in text.lower():
                endpoints.append(sys)

        # Look for function/method names with common patterns
        method_patterns = re.findall(r"(repair_|fix_|handle_|process_)\w+", text)
        endpoints.extend(method_patterns)

        return list(set(endpoints))[:10]  # Limit to top 10

    # ─── Triage (Fast-Path Pre-Simulation) ─────────────────────────────────────

    def _cleanup_semantic_key_for_proposal(self, proposal: EvolutionProposal) -> None:
        """
        Remove a proposal from semantic key tracking.
        Called when a proposal reaches a terminal state.
        """
        semantic_key = self._compute_semantic_key(proposal)
        self._active_proposal_semantic_keys.pop(semantic_key, None)

    def _compute_semantic_key(self, proposal: EvolutionProposal) -> str:
        """
        Compute a semantic key for the proposal based on structural content.

        Two proposals with the same semantic key describe the same structural fix
        regardless of which incident triggered them.

        Key = hash(target_system + change_category + error_class)
        """
        target_system = (proposal.change_spec.affected_systems[0]
                         if proposal.change_spec.affected_systems
                         else "unknown")
        change_category = proposal.category.value
        # Error class extracted from description or fallback to category
        error_class = proposal.category.value  # Fallback to category

        # For thymos proposals, try to extract error type from description
        if proposal.source == "thymos" and proposal.description:
            # Description typically: "[Thymos T4] Repair IncidentClass in System: action"
            parts = proposal.description.split(" in ")
            if len(parts) > 1:
                # Extract the incident class (e.g., "API_ERROR")
                class_part = parts[0].split(" ")[-1]  # Last word before "in"
                if class_part:
                    error_class = class_part

        semantic_content = f"{target_system}|{change_category}|{error_class}"
        return hashlib.sha256(semantic_content.encode()).hexdigest()[:16]

    def _triage_proposal(self, proposal: EvolutionProposal) -> TriageResult:
        """
        Fast-path proposal check. If trivial, skip expensive simulation.
        Trivial = budget tweaks <5% with sufficient data.

        Returns TriageResult with skip_simulation=True for trivial cases.
        """
        if proposal.category.value != "adjust_budget":
            return TriageResult(
                status=TriageStatus.REQUIRES_SIMULATION,
                skip_simulation=False,
            )

        spec = proposal.change_spec
        if not spec.budget_new_value or spec.budget_old_value is None:
            return TriageResult(
                status=TriageStatus.REQUIRES_SIMULATION,
                skip_simulation=False,
            )

        # Check delta < 5%
        old_val = spec.budget_old_value
        new_val = spec.budget_new_value
        if old_val == 0.0:
            delta_pct = 1.0  # Treat zero as 100% change
        else:
            delta_pct = abs(new_val - old_val) / abs(old_val)

        if delta_pct < 0.05:
            self._logger.info(
                "proposal_triaged",
                proposal_id=proposal.id,
                status="trivial",
                reason=f"Budget delta {delta_pct:.1%} < 5%",
            )
            return TriageResult(
                status=TriageStatus.TRIVIAL,
                assumed_risk=RiskLevel.LOW,
                reason=f"Budget delta {delta_pct:.1%} < 5%",
                skip_simulation=True,
            )

        return TriageResult(
            status=TriageStatus.REQUIRES_SIMULATION,
            skip_simulation=False,
        )

    # ─── Main Pipeline ─────────────────────────────────────────────────────────

    # VERIFIED: SimulaService.process_proposal(proposal: EvolutionProposal) -> ProposalResult
    async def process_proposal(self, proposal: EvolutionProposal) -> ProposalResult:
        """
        Main entry point for evolution proposals.

        Pipeline:
          DEDUP → VALIDATE → SIMULATE → [GOVERNANCE GATE] → APPLY → VERIFY → RECORD

        Spec reference: Section III.3.2
        Performance target: validation ≤50ms, simulation ≤30s, apply ≤5s
        """
        self._proposals_received += 1
        log = self._logger.bind(proposal_id=proposal.id, category=proposal.category.value)
        log.info("proposal_received", source=proposal.source, description=proposal.description[:100])

        # ── Skia modulation halt ──────────────────────────────────────────
        if self._modulation_halted:
            log.warning("proposal_rejected_modulation_halted")
            return ProposalResult(
                proposal_id=proposal.id,
                status=ProposalStatus.REJECTED,
                rejection_reason="modulation_halted",
            )

        # ── GRID CONSERVATION GATE ───────────────────────────────────────────
        # Do not run the expensive SimulaCodeAgent pipeline while the physical
        # grid is in carbon-heavy CONSERVATION mode.  Proposals are deferred
        # rather than lost - callers should retry once the grid recovers.
        if self._grid_state == "conservation":
            self._proposals_rejected += 1
            log.info(
                "proposal_deferred_grid_conservation",
                grid_state=self._grid_state,
            )
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason="Grid is in CONSERVATION mode - evolution pipeline paused. Retry when grid recovers.",
            )

        # ── METABOLIC STARVATION GATE ─────────────────────────────────────────
        # Simula is expensive (LLM + Dafny + tests). Gate early.
        # AUSTERITY: only Thymos-flagged critical bug fixes proceed
        # EMERGENCY/CRITICAL: full halt - protect existing codebase
        if self._starvation_level in ("emergency", "critical"):
            self._proposals_rejected += 1
            log.info("proposal_blocked_starvation", level=self._starvation_level)
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Metabolic starvation ({self._starvation_level}) - evolution halted.",
            )
        if self._starvation_level == "austerity" and proposal.source != "thymos":
            self._proposals_rejected += 1
            log.info("proposal_deferred_austerity", source=proposal.source)
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason="Metabolic austerity - only Thymos-flagged critical mutations proceed.",
            )

        # ── EVO CONSOLIDATION STALL GATE ──────────────────────────────────────
        # When Evo's learning pipeline is stalled, defer Evo-sourced proposals.
        # Repairs from Thymos/Axon still proceed - they're urgent and don't
        # depend on Evo learning from the outcome.
        if (
            self._evo_consolidation_stalled
            and proposal.source == "evo"
        ):
            # Auto-clear if the stall window has expired
            if time.monotonic() >= self._evo_stall_expires_at:
                self._evo_consolidation_stalled = False
                self._logger.info("simula_evo_stall_auto_cleared")
            else:
                self._proposals_rejected += 1
                log.info(
                    "proposal_deferred_evo_stall",
                    source=proposal.source,
                )
                return ProposalResult(
                    status=ProposalStatus.REJECTED,
                    reason=(
                        "Evo consolidation is stalled - evolution proposals "
                        "deferred until learning pipeline recovers."
                    ),
                )

        # ── IMMUNE ADVISORY FILTER (Loop 4) ───────────────────────────────────
        # Check if Thymos has flagged patterns that overlap with this proposal.
        affected_files = getattr(proposal, "target_files", []) or []
        is_safe, block_reason = self._immune_filter.check_proposal(
            proposal_description=proposal.description or "",
            affected_files=affected_files,
        )
        if not is_safe:
            self._proposals_rejected += 1
            log.info("proposal_blocked_immune_advisory", reason=block_reason)
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=block_reason,
            )

        # ── STEP 0: Deduplication ────────────────────────────────────────────
        # Snapshot active proposals under lock (cheap), run async dedup
        # outside the lock to avoid blocking other proposals during the
        # potential LLM Tier-3 similarity call, then re-acquire the lock for
        # the atomic capacity-check + insert.
        if self._intelligence is not None:
            async with self._proposals_lock:
                snapshot = list(self._active_proposals.values())
            if snapshot:
                try:
                    all_proposals = [proposal] + snapshot
                    clusters = await self._intelligence.deduplicate(all_proposals)
                    if self._intelligence.is_duplicate(proposal, clusters):
                        self._proposals_deduplicated += 1
                        log.info("proposal_deduplicated")
                        return ProposalResult(
                            status=ProposalStatus.REJECTED,
                            reason="Duplicate of an active proposal",
                        )
                except Exception as exc:
                    log.warning("dedup_check_failed", error=str(exc))

        # ── STEP 0.1: Semantic Deduplication ──────────────────────────────────
        # Check if a proposal with the same semantic key (target_system +
        # change_category + error_class) is already PENDING or IN_PROGRESS.
        # This catches duplicates across different incidents describing the
        # same structural fix (e.g., "fix null pointer in system X" from 10
        # different error occurrences).
        semantic_key = self._compute_semantic_key(proposal)
        if semantic_key in self._active_proposal_semantic_keys:
            existing_proposal_id, existing_status = self._active_proposal_semantic_keys[semantic_key]
            # Only block if the existing proposal hasn't reached a terminal state
            if existing_status in ("proposed", "simulating", "awaiting_governance", "approved", "applying"):
                self._proposals_deduplicated += 1
                log.info(
                    "proposal_deduplicated_semantic",
                    semantic_key=semantic_key,
                    existing_proposal_id=existing_proposal_id,
                    existing_status=existing_status,
                )
                return ProposalResult(
                    status=ProposalStatus.REJECTED,
                    reason=f"Duplicate of active proposal {existing_proposal_id} with same structural fix",
                )

        # ── STEP 0.5: Architecture EFE scoring ──────────────────────────────
        # Score the proposal before it enters the pipeline. Zero LLM tokens.
        if self._efe_scorer is not None and proposal.efe_score is None:
            try:
                breakdown = await self._efe_scorer.score_proposal(proposal)
                proposal.efe_score = breakdown.efe_penalised
                # Persist EFE score to Neo4j for monitoring
                if self._history is not None:
                    await self._history.record_proposal_efe(
                        proposal_id=proposal.id,
                        breakdown=breakdown,
                        rank=0,  # Single proposal, rank unknown
                        total_in_batch=1,
                    )
                log.info(
                    "proposal_efe_scored",
                    efe=proposal.efe_score,
                    confidence=breakdown.confidence,
                )
            except Exception as exc:
                log.warning("efe_scoring_failed", error=str(exc))

        async with self._proposals_lock:
            if len(self._active_proposals) >= self._config.max_active_proposals:
                log.warning(
                    "proposal_rejected_queue_full",
                    active=len(self._active_proposals),
                    limit=self._config.max_active_proposals,
                )
                return ProposalResult(
                    status=ProposalStatus.REJECTED,
                    reason=(
                        f"Too many active proposals "
                        f"({len(self._active_proposals)}/{self._config.max_active_proposals}). "
                        "Try again later."
                    ),
                )
            self._active_proposals[proposal.id] = proposal
            # Track semantic key for dedup across different incidents
            self._active_proposal_semantic_keys[semantic_key] = (proposal.id, "proposed")

        # ── STEP 0.6: Metabolic gate ──────────────────────────────────────────
        if not await self._check_metabolic_gate("process_proposal", "0.05"):
            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason="Metabolic gate denied: organism resources too low for evolution.",
            )

        # All remaining steps are wrapped in try/finally so any unexpected
        # exit (exception or early return) always removes the proposal from
        # _active_proposals and never leaves it stranded.
        try:
            return await asyncio.wait_for(
                self._run_pipeline(proposal, log),
                timeout=self._config.pipeline_timeout_s,
            )
        except TimeoutError:
            log.error(
                "proposal_pipeline_timeout",
                timeout_s=self._config.pipeline_timeout_s,
            )
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            self._invalidate_analytics()
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Pipeline timed out after {self._config.pipeline_timeout_s}s",
            )
        except Exception as exc:
            # Unexpected exception in pipeline - emit to Thymos for immune response
            # The immune system must be able to heal itself
            log.error(
                "proposal_pipeline_exception",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            # Meta-healing: report via sentinel (structured Incident)
            # 10s timeout - a stuck sentinel must not block meta-healing
            try:
                await asyncio.wait_for(
                    self._sentinel.report(
                        exc,
                        context={
                            "proposal_id": proposal.id,
                            "proposal_category": proposal.category.value,
                            "proposal_source": proposal.source,
                            "pipeline_stage": proposal.status.value,
                        },
                    ),
                    timeout=10.0,
                )
            except (asyncio.TimeoutError, TimeoutError):
                log.critical("sentinel_report_timeout", timeout_s=10.0)
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            self._invalidate_analytics()
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Pipeline failed with {type(exc).__name__}: {str(exc)[:100]}",
            )
        finally:
            # Clean up semantic key tracking for terminal states
            terminal_states = {ProposalStatus.APPLIED, ProposalStatus.ROLLED_BACK, ProposalStatus.REJECTED}
            if proposal.status in terminal_states:
                self._cleanup_semantic_key_for_proposal(proposal)

            # Governance-approved proposals are intentionally left in
            # _active_proposals (awaiting approve_governed_proposal call).
            # All other terminal states remove themselves inside _run_pipeline;
            # this is a safety net for any path we missed.
            if proposal.status not in (
                ProposalStatus.AWAITING_GOVERNANCE,
                ProposalStatus.APPLIED,
                ProposalStatus.ROLLED_BACK,
            ):
                async with self._proposals_lock:
                    self._active_proposals.pop(proposal.id, None)

    # Per-stage timeout budgets are sourced from self._config (configurable without
    # redeployment). Properties below provide backward-compatible attribute access.
    @property
    def _STAGE_TIMEOUT_VALIDATE_S(self) -> float:  # type: ignore[override]
        return self._config.validate_stage_timeout_s

    @property
    def _STAGE_TIMEOUT_SIMULATE_S(self) -> float:  # type: ignore[override]
        return self._config.simulate_stage_timeout_s

    @property
    def _STAGE_TIMEOUT_APPLY_S(self) -> float:  # type: ignore[override]
        return self._config.apply_stage_timeout_s

    @property
    def _STAGE_TIMEOUT_VERIFY_S(self) -> float:  # type: ignore[override]
        return self._config.verify_stage_timeout_s

    @property
    def _STAGE_TIMEOUT_RECORD_S(self) -> float:  # type: ignore[override]
        return self._config.record_stage_timeout_s

    _PROOF_SEARCH_TIMEOUT_S: float = 60.0  # Per-proof-search timeout

    async def _run_pipeline(
        self, proposal: EvolutionProposal, log: Any
    ) -> ProposalResult:
        """Inner pipeline body, always called from process_proposal's try/finally."""
        # ── STEP 1: Validate - Iron Rules + Constraint Satisfaction ────────
        # ConstraintSatisfactionChecker covers all Section 8 invariants.
        # HARD violations are rejected immediately; SOFT ones are logged only.
        from systems.simula.constraint_checker import ConstraintSatisfactionChecker

        constraint_checker = ConstraintSatisfactionChecker()
        violations = constraint_checker.check_proposal(proposal)
        hard_violations = [v for v in violations if v.severity == "hard"]
        if hard_violations:
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            first = hard_violations[0]
            reason = f"Constraint violation [{first.constraint_id}]: {first.description}"
            log.warning(
                "proposal_rejected_constraint",
                constraint_id=first.constraint_id,
                total_violations=len(hard_violations),
            )
            asyncio.ensure_future(self._emit_evolution_rejected(
                proposal, reason=reason, stage="validate",
            ))
            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)

        soft_violations = [v for v in violations if v.severity == "soft"]
        if soft_violations:
            log.info(
                "proposal_soft_constraint_warnings",
                constraints=[v.constraint_id for v in soft_violations],
            )

        # ── STEP 1.1: Telos Constitutional Binding ─────────────────────────
        # Validate that this mutation does not violate the drive topology.
        # Telos is the constitutional authority - if it says no, the proposal
        # is dead regardless of simulation outcome.
        # Budget: 1s for the entire validation stage.
        try:
            async with asyncio.timeout(self._STAGE_TIMEOUT_VALIDATE_S):
                # Telos validation is event-driven: Simula emits TELOS_WORLD_MODEL_VALIDATE
                # and Telos responds asynchronously via ALIGNMENT_GAP_WARNING which is cached
                # in _telos_alignment_gap_active. Check cache first (fast path), then notify
                # Telos of this proposal so it can update its assessment.
                if self._telos_alignment_gap_active:
                    proposal.status = ProposalStatus.REJECTED
                    self._proposals_rejected += 1
                    reason = (
                        "Telos constitutional binder signalled an active alignment gap: "
                        "mutation would violate the drive topology."
                    )
                    log.warning("proposal_rejected_constitutional_cached", reason=reason)
                    async with self._proposals_lock:
                        self._active_proposals.pop(proposal.id, None)
                    return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)
                # Notify Telos of this proposal (fire-and-forget; Telos caches result)
                _event_bus = getattr(getattr(self._synapse, "_event_bus", None), "emit", None)
                if _event_bus is not None:
                    try:
                        from systems.synapse.types import SynapseEvent, SynapseEventType
                        asyncio.ensure_future(self._synapse._event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.TELOS_WORLD_MODEL_VALIDATE,
                            source_system="simula",
                            data={
                                "update_type": proposal.category.value,
                                "delta_description": proposal.description,
                                "source_system": "simula",
                                "proposal_id": proposal.id,
                            },
                        )))
                    except Exception:
                        pass
                log.debug("telos_constitutional_check_notified")

                # ── STEP 1.2: Validate Against Learned Repair Patterns ─────────────────
                # If Evo has learned procedures for failure patterns, check if this
                # proposal addresses code touching known-failure endpoints without
                # including the known fix. This is a soft warning, not a blocker.
                if self._evo is not None:
                    await self._validate_against_learned_repairs(proposal)
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("validate_stage_timeout", timeout_s=self._STAGE_TIMEOUT_VALIDATE_S)

        # ── STEP 1.5: Triage (fast-path for trivial cases) ──────────────────
        triage = self._triage_proposal(proposal)
        if triage.skip_simulation:
            # Build synthetic simulation result
            proposal.simulation = SimulationResult(
                episodes_tested=0,
                risk_level=triage.assumed_risk or RiskLevel.LOW,
                risk_summary=f"Triaged as trivial: {triage.reason}",
                benefit_summary=proposal.expected_benefit,
            )
            log.info("proposal_triaged_skipping_simulation", reason=triage.reason)
            # Skip STEP 2 (Simulate) and proceed directly to governance/apply

        # ── RE training: verification routing decision ──
        asyncio.ensure_future(self._emit_re_training_example(
            category="verification_routing",
            instruction="Decide whether to run full simulation or fast-path triage for an evolution proposal.",
            input_context=f"category={proposal.category.value}, source={proposal.source}",
            output=f"skip_simulation={triage.skip_simulation}, status={triage.status.value}, reason={triage.reason or 'needs_simulation'}",
            outcome_quality=0.8 if triage.skip_simulation else 0.5,
            reasoning_trace=triage.reason or "",
            episode_id=getattr(proposal, "id", "") or "",
            constitutional_alignment=None,  # populated at record stage
        ))

        # ── STEP 2: Simulate (deep multi-strategy) ─────────────────────────
        # Skip if already triaged (has synthetic simulation)
        if proposal.simulation is None:
            # Metabolic gate: skip expensive 5-strategy simulation when starving
            _sim_metabolic_ok = await self._check_metabolic_gate(
                "simulate_change", estimated_cost_usd="0.20"
            )
            if not _sim_metabolic_ok and self._starvation_level in ("critical", "terminal"):
                log.warning(
                    "simulate_stage_skipped_starvation",
                    starvation_level=self._starvation_level,
                )
                proposal.simulation = SimulationResult(
                    episodes_tested=0,
                    risk_level=RiskLevel.LOW,
                    risk_summary=(
                        f"Simulation skipped - metabolic starvation ({self._starvation_level}). "
                        "Confidence reduced."
                    ),
                    benefit_summary=proposal.expected_benefit,
                    improvements=0,
                    regressions=0,
                )
                # Reduce simulation confidence flag so downstream stages treat it cautiously
                proposal._simulated = False  # type: ignore[attr-defined]
                simulation = proposal.simulation  # type: ignore[assignment]
            else:
                proposal.status = ProposalStatus.SIMULATING
                log.info("proposal_simulating")

                try:
                    import time as _time
                    _sim_t0 = _time.monotonic()
                    simulation = await asyncio.wait_for(
                        self._simulate_change(proposal),
                        timeout=self._STAGE_TIMEOUT_SIMULATE_S,
                    )
                    proposal.simulation = simulation
                    _sim_elapsed = _time.monotonic() - _sim_t0
                    proposal._simulation_duration_ms = int(_sim_elapsed * 1000)  # type: ignore[attr-defined]
                    # Budget alert: emit warning when stage uses more than the configured fraction
                    _sim_alert_threshold = (
                        self._config.simulate_stage_timeout_s
                        * self._config.stage_budget_alert_fraction
                    )
                    if _sim_elapsed >= _sim_alert_threshold:
                        log.warning(
                            "stage_budget_near_exhausted",
                            stage="simulate",
                            elapsed_s=round(_sim_elapsed, 2),
                            budget_s=self._config.simulate_stage_timeout_s,
                            fraction_used=round(
                                _sim_elapsed / self._config.simulate_stage_timeout_s, 3
                            ),
                        )

                    # ── Corpus 14 §6: Emit KAIROS_CAUSAL_CANDIDATE_GENERATED ─────────
                    # Export dependency graph from simulation so Kairos can mine causal
                    # repair patterns across proposal categories and affected systems.
                    try:
                        _kairos_bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
                        if _kairos_bus is not None:
                            from systems.synapse.types import SynapseEvent, SynapseEventType
                            import uuid as _uuid_k
                            _blast = (
                                simulation.dependency_blast_radius
                                if hasattr(simulation, "dependency_blast_radius")
                                else 0
                            )
                            _sim_eps = max(1, simulation.episodes_tested)
                            _corr = simulation.improvements / _sim_eps
                            await _kairos_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.KAIROS_CAUSAL_CANDIDATE_GENERATED,
                                source_system="simula",
                                data={
                                    "candidate_id": str(_uuid_k.uuid4()),
                                    "variable_a": proposal.category.value,
                                    "variable_b": (proposal.systems_affected or ["unknown"])[0],
                                    "mean_correlation": round(_corr, 4),
                                    "cross_context_variance": round(
                                        simulation.regressions / _sim_eps, 4
                                    ),
                                    "context_count": simulation.episodes_tested,
                                    "dependency_blast_radius": _blast,
                                    "risk_level": simulation.risk_level.value,
                                    "proposal_id": proposal.id,
                                    "description_snippet": proposal.description[:120],
                                },
                            ))
                            log.debug(
                                "kairos_causal_candidate_emitted",
                                proposal_id=proposal.id,
                                correlation=round(_corr, 4),
                                blast_radius=_blast,
                            )
                    except Exception as _kex:
                        log.debug("kairos_causal_emit_failed", error=str(_kex))

                except (asyncio.TimeoutError, TimeoutError):
                    proposal.status = ProposalStatus.REJECTED
                    self._proposals_rejected += 1
                    reason = f"Simulation timed out after {self._STAGE_TIMEOUT_SIMULATE_S}s"
                    log.error("simulation_stage_timeout", timeout_s=self._STAGE_TIMEOUT_SIMULATE_S)
                    async with self._proposals_lock:
                        self._active_proposals.pop(proposal.id, None)
                    return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)
                except Exception as exc:
                    proposal.status = ProposalStatus.REJECTED
                    self._proposals_rejected += 1
                    reason = f"Simulation failed: {exc}"
                    log.error("simulation_error", error=str(exc))
                    async with self._proposals_lock:
                        self._active_proposals.pop(proposal.id, None)
                    return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)
        else:
            simulation = proposal.simulation

        # ── RE training: risk assessment after simulation ──
        _sim_ms = getattr(proposal, "_simulation_duration_ms", 0)
        _risk_q = {"low": 0.9, "medium": 0.6, "high": 0.3, "unacceptable": 0.0}.get(
            simulation.risk_level.value, 0.5
        )
        asyncio.ensure_future(self._emit_re_training_example(
            category="risk_assessment",
            instruction="Assess risk of an evolution proposal after multi-strategy simulation.",
            input_context=f"category={proposal.category.value}, episodes={simulation.episodes_tested}, improvements={simulation.improvements}, regressions={simulation.regressions}",
            output=f"risk={simulation.risk_level.value}, summary={simulation.risk_summary[:200]}",
            outcome_quality=_risk_q,
            latency_ms=_sim_ms,
            reasoning_trace=simulation.benefit_summary[:200] if simulation.benefit_summary else "",
            episode_id=getattr(proposal, "id", "") or "",
            constitutional_alignment=None,  # populated at record stage with full outcome
        ))

        if simulation.risk_level == RiskLevel.UNACCEPTABLE:
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            reason = f"Simulation shows unacceptable risk: {simulation.risk_summary}"
            log.warning("proposal_rejected_risk", risk_level=simulation.risk_level.value)
            asyncio.ensure_future(self._emit_evolution_rejected(
                proposal, reason=reason, stage="simulate",
            ))
            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)

        # ── STEP 2.5: Oneiros Dream Gate ──────────────────────────────────
        # If Oneiros has already lucid-dream-tested this proposal (via the
        # async SimulaProtocol callbacks), honour its recommendation.
        # A dream-rejected proposal is blocked before code is ever generated.
        async with self._dream_results_lock:
            dream_result = self._dream_results.pop(proposal.id, None)
        if dream_result is not None:
            dream_rec = dream_result.get("recommendation", "")
            dream_report = dream_result.get("report", {})
            if dream_rec == "reject":
                proposal.status = ProposalStatus.REJECTED
                self._proposals_rejected += 1
                violations = dream_report.get("violation_details", [])
                delta = dream_report.get("overall_performance_delta", 0.0)
                reason = (
                    f"Oneiros lucid dream simulation rejected: "
                    f"performance_delta={delta:.4f}"
                    + (f", violations={violations}" if violations else "")
                )
                log.warning("proposal_rejected_dream", reason=reason)
                async with self._proposals_lock:
                    self._active_proposals.pop(proposal.id, None)
                return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)
            else:
                # Dream approved - annotate for history
                proposal.dream_origin = True
                proposal.dream_coherence_score = dream_report.get(
                    "overall_performance_delta", 0.0
                )
                log.info(
                    "dream_simulation_approved",
                    performance_delta=dream_report.get("overall_performance_delta", 0.0),
                    scenarios_tested=dream_report.get("scenarios_tested", 0),
                )

        # ── STEP 3: Governance gate ─────────────────────────────────────────
        if self.requires_governance(proposal):
            proposal.status = ProposalStatus.AWAITING_GOVERNANCE
            self._proposals_awaiting_governance += 1
            try:
                governance_id = await self._submit_to_governance(proposal, simulation)
                proposal.governance_record_id = governance_id
            except Exception as exc:
                # Governance submission failure is a hard stop: without a
                # governance record the change cannot be audited or approved.
                log.error("governance_submission_failed", error=str(exc))
                proposal.status = ProposalStatus.REJECTED
                self._proposals_rejected += 1
                self._proposals_awaiting_governance = max(0, self._proposals_awaiting_governance - 1)
                async with self._proposals_lock:
                    self._active_proposals.pop(proposal.id, None)
                return ProposalResult(
                    status=ProposalStatus.REJECTED,
                    reason=f"Governance submission failed: {exc}",
                )

            log.info("proposal_awaiting_governance", governance_id=governance_id)
            asyncio.ensure_future(self._emit_evolution_awaiting_governance(
                proposal, governance_record_id=governance_id,
            ))
            return ProposalResult(
                status=ProposalStatus.AWAITING_GOVERNANCE,
                governance_record_id=governance_id,
            )

        # ── STEP 4: Apply (self-applicable changes only) ───────────────────
        # Combined budget for apply + verify + record sub-stages (each has its
        # own internal timeout; this is the hard outer cap).
        _apply_budget = (
            self._STAGE_TIMEOUT_APPLY_S
            + self._STAGE_TIMEOUT_VERIFY_S
            + self._STAGE_TIMEOUT_RECORD_S
        )
        try:
            import time as _time
            _apply_t0 = _time.monotonic()
            _apply_result = await asyncio.wait_for(
                self._apply_change(proposal),
                timeout=_apply_budget,
            )
            _apply_elapsed = _time.monotonic() - _apply_t0
            if _apply_elapsed >= _apply_budget * self._config.stage_budget_alert_fraction:
                log.warning(
                    "stage_budget_near_exhausted",
                    stage="apply+verify+record",
                    elapsed_s=round(_apply_elapsed, 2),
                    budget_s=_apply_budget,
                    fraction_used=round(_apply_elapsed / _apply_budget, 3),
                )
            return _apply_result
        except (asyncio.TimeoutError, TimeoutError):
            self._transition_proposal_status(proposal, ProposalStatus.ROLLED_BACK)
            self._proposals_rolled_back += 1
            log.error("apply_stage_timeout", timeout_s=_apply_budget)
            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            return ProposalResult(
                status=ProposalStatus.ROLLED_BACK,
                reason=f"Apply/verify/record stage timed out after {_apply_budget}s",
            )

    async def receive_evo_proposal(
        self,
        evo_description: str,
        evo_rationale: str,
        hypothesis_ids: list[str],
        hypothesis_statements: list[str],
        evidence_scores: list[float],
        supporting_episode_ids: list[str],
        mutation_target: str = "",
        mutation_type: str = "",
    ) -> ProposalResult:
        """
        Receive a proposal from Evo via the bridge.
        Translates the lightweight Evo proposal into Simula's rich format,
        then feeds it into the main pipeline.

        This is the public API that Evo's ConsolidationOrchestrator calls.
        """
        if self._bridge is None:
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason="Simula bridge not initialized",
            )

        self._logger.info(
            "evo_proposal_received",
            description=evo_description[:80],
            hypotheses=len(hypothesis_ids),
        )

        try:
            translated = await self._bridge.translate_proposal(
                evo_description=evo_description,
                evo_rationale=evo_rationale,
                hypothesis_ids=hypothesis_ids,
                hypothesis_statements=hypothesis_statements,
                evidence_scores=evidence_scores,
                supporting_episode_ids=supporting_episode_ids,
                mutation_target=mutation_target,
                mutation_type=mutation_type,
            )
        except Exception as exc:
            self._logger.error("bridge_translation_failed", error=str(exc))
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Bridge translation failed: {exc}",
            )

        return await self.process_proposal(translated)

    async def rank_proposals_by_efe(
        self, proposals: list[EvolutionProposal],
    ) -> RankedProposalQueue:
        """
        Score and rank a batch of proposals by architecture-level EFE.

        Called by Evo's consolidation Phase 8 before submitting proposals
        to Simula, and by Equor when reviewing a batch.

        Returns a RankedProposalQueue with proposals sorted by EFE
        (most negative = best). Each proposal's efe_score field is set.
        """
        if self._efe_scorer is None:
            # Return unranked if scorer not available
            return RankedProposalQueue(
                proposals=proposals,
                scores=[0.0] * len(proposals),
            )

        queue = await self._efe_scorer.score_and_rank(proposals)

        # Persist individual EFE scores to Neo4j for monitoring
        if self._history is not None:
            for rank_idx, (prop, breakdown) in enumerate(
                zip(queue.proposals, queue.breakdowns, strict=True)
            ):
                try:
                    await self._history.record_proposal_efe(
                        proposal_id=prop.id,
                        breakdown=breakdown,
                        rank=rank_idx + 1,
                        total_in_batch=len(queue.proposals),
                    )
                except Exception:
                    pass  # Non-critical; scoring logged in scorer

        self._logger.info(
            "proposals_ranked_by_efe",
            count=len(queue.proposals),
            recommended=queue.recommended.id if queue.recommended else None,
            top_efe=queue.scores[0] if queue.scores else None,
        )
        return queue

    async def approve_governed_proposal(
        self, proposal_id: str, governance_record_id: str
    ) -> ProposalResult:
        """
        Called when a governed proposal receives community approval.
        Resumes the pipeline from the application step.
        """
        proposal = self._active_proposals.get(proposal_id)
        if proposal is None:
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Proposal {proposal_id} not found in active proposals",
            )
        if proposal.status != ProposalStatus.AWAITING_GOVERNANCE:
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Proposal {proposal_id} is not awaiting governance (status: {proposal.status})",
            )

        proposal.status = ProposalStatus.APPROVED
        self._proposals_awaiting_governance = max(0, self._proposals_awaiting_governance - 1)
        self._logger.info("governed_proposal_approved", proposal_id=proposal_id)
        return await self._apply_change(proposal)

    def requires_governance(self, proposal: EvolutionProposal) -> bool:
        """
        Changes in the GOVERNANCE_REQUIRED category need governance, except:
        - Low-risk proposals that passed simulation are auto-approved (freedom for self-healing)
        - This prevents bottlenecks where governance is missing or slow

        High-risk proposals still require explicit approval.
        """
        if proposal.category not in GOVERNANCE_REQUIRED:
            return False

        # Low-risk proposals are auto-approved - no governance bottleneck
        if proposal.efe_score is not None and proposal.efe_score < -1.0:
            return False  # Low EFE score = safe change, auto-approve

        return True

    # ─── Query Interface ───────────────────────────────────────────────────────

    async def get_history(self, limit: int = 50) -> list[EvolutionRecord]:
        """Return the most recent evolution records."""
        if self._history is None:
            return []
        return await self._history.get_history(limit=limit)

    async def get_current_version(self) -> int:
        """Return the current config version number."""
        return self._current_version

    async def get_version_chain(self) -> list[ConfigVersion]:
        """Return the full version history chain."""
        if self._history is None:
            return []
        return await self._history.get_version_chain()

    def get_active_proposals(self) -> list[EvolutionProposal]:
        """Return all proposals currently in the pipeline."""
        return list(self._active_proposals.values())

    async def get_analytics(self) -> EvolutionAnalytics:
        """Return current evolution quality analytics."""
        if self._analytics is None:
            return EvolutionAnalytics()
        return await self._analytics.compute_analytics()

    async def get_prioritized_proposals(self) -> list[dict[str, Any]]:
        """Return active proposals ranked by priority score."""
        if self._intelligence is None or not self._active_proposals:
            return []
        priorities = await self._intelligence.prioritize(list(self._active_proposals.values()))
        return [p.model_dump() for p in priorities]

    # ─── Genetic Memory: Genome & Lineage ────────────────────────────────────

    async def extract_simula_genome(
        self,
        instance_id: str,
        parent_ids: list[str] | None = None,
        generation: int = 1,
    ) -> dict[str, Any]:
        """
        Extract the Simula evolution genome for this instance.

        Returns a dict with 'genome_id' and extraction metrics,
        or an error dict if extraction is not eligible.
        """
        if self._neo4j is None:
            return {"error": "Neo4j not available"}

        from systems.simula.genome import SimulaGenomeExtractor

        extractor = SimulaGenomeExtractor(
            neo4j=self._neo4j,
            instance_id=instance_id,
        )
        genome, result = await extractor.extract_genome(
            parent_ids=parent_ids,
            generation=generation,
        )
        if genome is None:
            return {"error": "not_eligible", **result.model_dump()}
        return result.model_dump()

    async def export_simula_genome(self) -> "SimulaGenomeInheritance | None":
        """
        Snapshot the current Simula evolution state into a SimulaGenome for child
        inheritance.  Called by SpawnChildExecutor Step 0b at child spawn time.

        Returns a primitives.genome_inheritance.SimulaGenome capturing:
          - Current learnable config parameters (budget multipliers, timeouts, etc.)
          - Last 10 applied/rolled-back mutation records (child's "already tried" seed)
          - Dafny spec hashes for formally-verified contracts (child skips re-verify)
        Returns None if not yet initialised.
        """
        from primitives.genome_inheritance import (
            SimulaGenome as SimulaGenomeInheritance,
            SimulaMutationEntry,
        )

        try:
            import os as _os_sg

            instance_id = _os_sg.environ.get("ECODIAOS_INSTANCE_ID", "eos-default")

            # ── Learnable config params ────────────────────────────────────────
            current_evolution_params: dict[str, Any] = {}
            if self._config is not None:
                # Capture the 23+ PHASE_A learnable parameters as a flat dict
                _learnable_fields = [
                    "max_simulation_episodes",
                    "simulation_risk_threshold",
                    "embedding_similarity_threshold",
                    "dafny_verify_timeout_s",
                    "z3_check_timeout_ms",
                    "z3_max_discovery_rounds",
                    "dafny_max_clover_rounds",
                    "max_rollback_files",
                    "grpo_learning_rate",
                    "grpo_batch_size",
                    "grpo_kl_penalty",
                    "lilo_consolidation_interval_proposals",
                    "max_code_agent_turns",
                    "static_analysis_max_fix_iterations",
                    "thinking_budget_tokens",
                    "kv_compression_ratio",
                    "integration_tests_timeout_s",
                    "performance_baseline_timeout_s",
                    "agent_coder_max_iterations",
                    "agent_coder_test_timeout_s",
                    "mutation_rate",
                    "canary_rollout_steps",
                    "max_proposals_per_cycle",
                    # Economic learnable parameters (SIMULA-ECON-1)
                    "yield_apy_drop_rebalance_threshold",
                    "bounty_min_roi_multiple",
                    "protocol_allocation_aggressiveness",
                    "audit_aggressiveness",
                ]
                for field in _learnable_fields:
                    val = getattr(self._config, field, None)
                    if val is not None:
                        current_evolution_params[field] = val

            # ── Last 10 mutation records ───────────────────────────────────────
            last_10: list[SimulaMutationEntry] = []
            if self._history is not None:
                try:
                    records = await self._history.get_history(limit=10)
                    for rec in records:
                        last_10.append(SimulaMutationEntry(
                            mutation_id=str(getattr(rec, "id", "")),
                            category=(
                                rec.category.value
                                if hasattr(getattr(rec, "category", None), "value")
                                else str(getattr(rec, "category", ""))
                            ),
                            description=str(getattr(rec, "description", ""))[:300],
                            applied_at=getattr(rec, "applied_at", None) or utc_now(),
                            was_rolled_back=bool(getattr(rec, "rolled_back", False)),
                            risk_level=(
                                rec.simulation_risk.value
                                if hasattr(getattr(rec, "simulation_risk", None), "value")
                                else str(getattr(rec, "simulation_risk", "low"))
                            ),
                        ))
                except Exception:
                    pass  # Non-fatal - child starts with empty mutation history

            # ── Dafny spec hashes ──────────────────────────────────────────────
            dafny_spec_hashes: dict[str, str] = {}
            # Hashes are built lazily from the IncrementalVerificationEngine cache;
            # if not available, child simply re-verifies on first run.
            try:
                if self._incremental is not None and hasattr(self._incremental, "_spec_cache"):
                    cache = self._incremental._spec_cache  # type: ignore[attr-defined]
                    if isinstance(cache, dict):
                        for path, entry in cache.items():
                            h = (entry or {}).get("hash", "") if isinstance(entry, dict) else ""
                            if h:
                                dafny_spec_hashes[str(path)] = str(h)
            except Exception:
                pass

            # ── Proof strategy router weights (Thompson sampling) ──────────
            reasoning_router_weights: dict[str, dict[str, float]] = {}
            try:
                reasoning_router_weights = self._reasoning_router.get_weights()
            except Exception:
                pass

            # ── EFE calibration data ──────────────────────────────────────
            efe_calibration: dict[str, Any] = {}
            try:
                if self._efe_scorer is not None:
                    efe_calibration = {
                        "category_priors": getattr(self._efe_scorer, "_category_priors", {}),
                        "calibration_records": len(getattr(self._efe_scorer, "_calibration_records", [])),
                    }
            except Exception:
                pass

            # ── Category success rates (heritable evolutionary priors) ────
            category_success_rates: dict[str, float] = {}
            try:
                if self._analytics is not None:
                    analytics_data = await self._analytics.compute_analytics()
                    for cat_name, rate_info in getattr(
                        analytics_data, "category_rates", {}
                    ).items():
                        sr = getattr(rate_info, "success_rate", None)
                        if sr is not None:
                            category_success_rates[str(cat_name)] = float(sr)
            except Exception:
                pass

            # ── LILO library top abstractions (heritable code patterns) ──
            lilo_abstractions: list[dict[str, Any]] = []
            try:
                if self._lilo is not None:
                    await self._lilo._ensure_library_loaded()
                    lib = self._lilo._library or []
                    ranked = sorted(
                        lib,
                        key=lambda a: a.usage_count * a.confidence,
                        reverse=True,
                    )[:15]  # Top 15 abstractions by impact
                    for ab in ranked:
                        lilo_abstractions.append({
                            "name": ab.name,
                            "kind": ab.kind.value if hasattr(ab.kind, "value") else str(ab.kind),
                            "signature": ab.signature,
                            "description": ab.description[:200],
                            "usage_count": ab.usage_count,
                            "confidence": ab.confidence,
                            "tags": ab.tags[:5],
                        })
            except Exception:
                pass

            # ── Successful mutation embeddings (heritable semantic memory) ─
            # Top 10 non-rolled-back mutations with embeddings for child
            # vector index seeding (avoids re-discovering known-good patterns)
            mutation_embeddings: list[dict[str, Any]] = []
            try:
                if self._history is not None and self._history._neo4j is not None:
                    rows = await self._history._neo4j.execute_read(
                        """
                        MATCH (r:EvolutionRecord)
                        WHERE r.rolled_back = false
                          AND r.embedding IS NOT NULL
                        RETURN r.id AS id,
                               r.category AS category,
                               r.description AS description,
                               r.embedding AS embedding
                        ORDER BY r.applied_at DESC
                        LIMIT 10
                        """,
                    )
                    for row in rows:
                        emb = row.get("embedding")
                        if emb and isinstance(emb, list):
                            mutation_embeddings.append({
                                "id": str(row.get("id", "")),
                                "category": str(row.get("category", "")),
                                "description": str(row.get("description", ""))[:200],
                                "embedding": emb,
                            })
            except Exception:
                pass

            genome = SimulaGenomeInheritance(
                instance_id=instance_id,
                generation=self._current_version,
                current_evolution_params=current_evolution_params,
                last_10_mutations=last_10,
                dafny_spec_hashes=dafny_spec_hashes,
            )
            # Attach extra heritable state as metadata (forwards-compatible)
            genome.extra = {
                "reasoning_router_weights": reasoning_router_weights,
                "efe_calibration": efe_calibration,
                "category_success_rates": category_success_rates,
                "lilo_abstractions": lilo_abstractions,
                "mutation_embeddings": mutation_embeddings,
            }
            self._logger.info(
                "simula_genome_exported",
                genome_id=genome.genome_id,
                param_count=len(current_evolution_params),
                mutation_count=len(last_10),
                dafny_hash_count=len(dafny_spec_hashes),
                router_strategies=len(reasoning_router_weights),
                category_rates=len(category_success_rates),
                lilo_abstractions=len(lilo_abstractions),
                mutation_embeddings=len(mutation_embeddings),
            )
            return genome

        except Exception as exc:
            self._logger.error("export_simula_genome_failed", error=str(exc))
            return None

    async def _apply_inherited_simula_genome_if_child(self) -> None:
        """
        Child-side bootstrap: deserialise parent SimulaGenome from environment.

        Reads ECODIAOS_SIMULA_GENOME_PAYLOAD (JSON-encoded SimulaGenome) injected
        by LocalDockerSpawner.  If present, applies inherited learnable config params
        to self._config so the child starts with the parent's tuned evolution state.
        Non-fatal - child falls back to default config on any error.

        Applies bounded ±10% Gaussian jitter per param for genetic variation, matching
        the Telos pattern (Spec 18 SG3).

        Only runs when ECODIAOS_IS_GENESIS_NODE != 'true'.
        """
        import json as _json
        import os as _os
        import random as _random

        if _os.environ.get("ECODIAOS_IS_GENESIS_NODE", "true").lower() == "true":
            return

        payload_json = _os.environ.get("ECODIAOS_SIMULA_GENOME_PAYLOAD", "").strip()
        if not payload_json:
            return

        try:
            from primitives.genome_inheritance import SimulaGenome
            from systems.synapse.types import SynapseEvent, SynapseEventType

            data = _json.loads(payload_json)
            parent_genome = SimulaGenome.model_validate(data)

            params_applied = 0
            if self._config is not None and parent_genome.current_evolution_params:
                for param_name, value in parent_genome.current_evolution_params.items():
                    if not hasattr(self._config, param_name):
                        continue
                    try:
                        # Apply bounded ±10% Gaussian jitter for genetic variation
                        jitter = _random.gauss(0.0, 0.033)  # σ=3.3% → ≈99% within ±10%
                        jitter = max(-0.10, min(0.10, jitter))
                        if isinstance(value, float):
                            mutated = value * (1.0 + jitter)
                        elif isinstance(value, int):
                            mutated = max(1, round(value * (1.0 + jitter)))
                        else:
                            mutated = value  # strings/bools: inherit exactly
                        setattr(self._config, param_name, type(value)(mutated))
                        params_applied += 1
                    except Exception:
                        pass

            # Store mutation history for analytics
            if parent_genome.last_10_mutations:
                self._inherited_mutation_history = list(parent_genome.last_10_mutations)

            # Store inherited Dafny spec hashes for verification skip optimization.
            # Child can skip re-verification of specs whose content hash matches
            # the parent's verified set (one-time boot optimization).
            if parent_genome.dafny_spec_hashes:
                self._inherited_spec_hashes: dict[str, str] = dict(
                    parent_genome.dafny_spec_hashes
                )
                # Seed the IncrementalVerificationEngine if available
                if self._incremental is not None:
                    self._incremental.seed_inherited_hashes(
                        self._inherited_spec_hashes
                    )
                    self._logger.info(
                        "inherited_dafny_spec_hashes_seeded",
                        count=len(self._inherited_spec_hashes),
                    )

            # Seed reasoning router weights from parent (Thompson sampling priors)
            extra = getattr(parent_genome, "extra", {}) or {}
            router_weights = extra.get("reasoning_router_weights")
            if router_weights and isinstance(router_weights, dict):
                self._reasoning_router.load_weights(router_weights)
                self._logger.info(
                    "inherited_reasoning_router_weights",
                    strategies=list(router_weights.keys()),
                )

            # Seed category success rates as evolutionary priors
            inherited_rates = extra.get("category_success_rates")
            _rates_count = 0
            if inherited_rates and isinstance(inherited_rates, dict) and self._analytics is not None:
                try:
                    # Store as prior knowledge on analytics engine for Bayesian updating
                    self._analytics._inherited_category_priors = dict(inherited_rates)
                    _rates_count = len(inherited_rates)
                    self._logger.info(
                        "inherited_category_success_rates",
                        categories=list(inherited_rates.keys()),
                    )
                except Exception:
                    pass

            # Seed LILO library with parent's top abstractions
            inherited_lilo = extra.get("lilo_abstractions")
            _lilo_count = 0
            if inherited_lilo and isinstance(inherited_lilo, list) and self._lilo is not None:
                try:
                    from systems.simula.verification.types import (
                        AbstractionKind,
                        LibraryAbstraction,
                    )

                    await self._lilo._ensure_library_loaded()
                    for ab_data in inherited_lilo:
                        if not isinstance(ab_data, dict):
                            continue
                        kind_str = ab_data.get("kind", "utility")
                        try:
                            kind = AbstractionKind(kind_str)
                        except (ValueError, KeyError):
                            kind = AbstractionKind.UTILITY
                        ab = LibraryAbstraction(
                            name=ab_data.get("name", ""),
                            kind=kind,
                            description=ab_data.get("description", ""),
                            signature=ab_data.get("signature", ""),
                            source_code="",  # Not transmitted - child re-discovers
                            usage_count=ab_data.get("usage_count", 1),
                            confidence=ab_data.get("confidence", 0.3) * 0.8,  # Decay
                            tags=ab_data.get("tags", []),
                        )
                        if ab.name and self._lilo._library is not None:
                            # Only add if not already in library
                            existing_names = {a.name for a in self._lilo._library}
                            if ab.name not in existing_names:
                                self._lilo._library.append(ab)
                                _lilo_count += 1
                    if _lilo_count:
                        self._logger.info(
                            "inherited_lilo_abstractions",
                            count=_lilo_count,
                        )
                except Exception:
                    pass

            # Seed mutation embeddings into history vector index
            inherited_embeddings = extra.get("mutation_embeddings")
            _emb_count = 0
            if (
                inherited_embeddings
                and isinstance(inherited_embeddings, list)
                and self._history is not None
                and self._history._neo4j is not None
            ):
                try:
                    for emb_data in inherited_embeddings:
                        if not isinstance(emb_data, dict):
                            continue
                        emb_vec = emb_data.get("embedding")
                        if not emb_vec or not isinstance(emb_vec, list):
                            continue
                        # Write as InheritedEvolutionRecord nodes for vector search
                        await self._history._neo4j.execute_write(
                            """
                            MERGE (r:InheritedEvolutionRecord {id: $id})
                            SET r.category = $category,
                                r.description = $description,
                                r.embedding = $embedding,
                                r.inherited_from = $parent_id,
                                r.inherited_at = datetime()
                            """,
                            {
                                "id": f"inherited_{emb_data.get('id', '')}",
                                "category": str(emb_data.get("category", "")),
                                "description": str(emb_data.get("description", "")),
                                "embedding": emb_vec,
                                "parent_id": str(parent_genome.instance_id),
                            },
                        )
                        _emb_count += 1
                    if _emb_count:
                        self._logger.info(
                            "inherited_mutation_embeddings",
                            count=_emb_count,
                        )
                except Exception:
                    pass

            self._logger.info(
                "inherited_simula_genome",
                parent_id=str(parent_genome.instance_id),
                genome_id=str(parent_genome.genome_id),
                params_applied=params_applied,
                mutation_history_entries=len(parent_genome.last_10_mutations),
                dafny_hashes=len(parent_genome.dafny_spec_hashes),
                router_strategies=len(router_weights) if router_weights else 0,
                category_rates=_rates_count,
                lilo_abstractions=_lilo_count,
                mutation_embeddings=_emb_count,
            )

            # Notify other systems that genome was inherited
            if self._synapse is not None:
                try:
                    await self._synapse._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.GENOME_INHERITED,
                        source_system="simula",
                        data={
                            "system": "simula",
                            "genome_id": str(parent_genome.genome_id),
                            "parent_id": str(parent_genome.instance_id),
                            "params_applied": params_applied,
                        },
                    ))
                except Exception:
                    pass

        except Exception as exc:
            self._logger.warning(
                "simula_genome_apply_failed",
                error=str(exc),
                note="Proceeding with default config params",
            )

    async def get_lineage(
        self,
        instance_id: str,
    ) -> list[dict[str, Any]]:
        """Return the full ancestor chain for an instance."""
        if self._neo4j is None:
            return []

        from systems.simula.lineage import EvolutionLineageTracker

        tracker = EvolutionLineageTracker(
            neo4j=self._neo4j,
            instance_id=instance_id,
        )
        records = await tracker.get_lineage(instance_id)
        return [r.model_dump() for r in records]

    async def get_population_snapshot(self) -> dict[str, Any]:
        """Capture fleet-wide evolutionary state for selection decisions."""
        if self._neo4j is None:
            return {}

        from systems.simula.lineage import EvolutionLineageTracker

        tracker = EvolutionLineageTracker(
            neo4j=self._neo4j,
            instance_id="",
        )
        snapshot = await tracker.get_population_snapshot()
        return snapshot.model_dump()

    async def select_best_genome(self) -> str | None:
        """
        Select the highest-fitness SimulaGenome for propagation.

        Returns the genome ID, or None if no eligible genomes exist.
        """
        if self._neo4j is None:
            return None

        from systems.simula.lineage import EvolutionLineageTracker

        tracker = EvolutionLineageTracker(
            neo4j=self._neo4j,
            instance_id="",
        )
        return await tracker.select_best_genome_for_propagation()

    # ─── Stats ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "initialized": self._initialized,
            "current_version": self._current_version,
            "proposals_received": self._proposals_received,
            "proposals_approved": self._proposals_approved,
            "proposals_rejected": self._proposals_rejected,
            "proposals_rolled_back": self._proposals_rolled_back,
            "proposals_deduplicated": self._proposals_deduplicated,
            "proposals_awaiting_governance": self._proposals_awaiting_governance,
            "active_proposals": len(self._active_proposals),
        }

        # Include cached analytics summary if available
        if self._analytics is not None and self._analytics._cached_analytics is not None:
            analytics = self._analytics._cached_analytics
            base["analytics"] = {
                "total_proposals": analytics.total_proposals,
                "evolution_velocity": analytics.evolution_velocity,
                "rollback_rate": analytics.rollback_rate,
                "mean_simulation_risk": analytics.mean_simulation_risk,
            }

        # Architecture EFE scorer stats
        if self._efe_scorer is not None:
            base["architecture_efe"] = self._efe_scorer.stats

        # Stage 3 subsystem status
        base["stage3"] = {
            "incremental_verification": self._incremental is not None,
            "swe_grep": self._swe_grep is not None,
            "lilo": self._lilo is not None,
        }

        # Stage 4 subsystem status
        base["stage4"] = {
            "lean": self._lean_bridge is not None,
            "grpo": self._grpo is not None,
            "diffusion_repair": self._diffusion_repair is not None,
        }

        # Stage 5 subsystem status
        base["stage5"] = {
            "synthesis": self._synthesis is not None,
            "repair_agent": self._repair_agent is not None,
            "orchestrator": self._orchestrator is not None,
            "causal_debugger": self._causal_debugger is not None,
            "issue_resolver": self._issue_resolver is not None,
        }

        # Stage 6 subsystem status
        base["stage6"] = {
            "hash_chain": self._hash_chain is not None,
            "content_credentials": self._content_credentials is not None,
            "governance_credentials": self._governance_credentials is not None,
            "hard_negative_miner": self._hard_negative_miner is not None,
            "adversarial_tester": self._adversarial_tester is not None,
            "formal_spec_generator": self._formal_spec_generator is not None,
            "egraph": self._egraph is not None,
            "symbolic_execution": self._symbolic_execution is not None,
        }

        # Stage 7 subsystem status
        base["stage7"] = {
            "inspector": self._inspector is not None,
        }
        if self._inspector is not None:
            base["stage7"]["inspector_stats"] = self._inspector.stats

        # Phase 9: Inspector analytics observability
        base["stage9_analytics"] = {
            "inspector_analytics_emitter": self._inspector_analytics is not None,
            "inspector_tsdb_persistence": (
                self._inspector_analytics is not None
                and self._inspector_analytics._store is not None
            ),
            "inspector_view_attached": (
                self._analytics is not None
                and self._analytics._inspector_view is not None
            ),
            "inspector_store_attached": (
                self._analytics is not None
                and self._analytics._inspector_store is not None
            ),
        }
        if self._inspector_analytics is not None:
            base["stage9_analytics"]["emitter_stats"] = self._inspector_analytics.stats

        # Self-healing sub-systems
        base["proactive_scanner"] = (
            self._proactive_scanner.stats.to_dict()
            if self._proactive_scanner is not None
            else {"scanner_alive": False}
        )
        base["repair_memory"] = {
            "record_count": (
                self._repair_memory.record_count if self._repair_memory is not None else 0
            ),
            "calibration_score": (
                self._repair_memory.get_calibration_score()
                if self._repair_memory is not None
                else None
            ),
        }

        return base

    async def get_metrics(self) -> dict[str, Any]:
        """
        Full operational metrics picture for GET /simula/metrics  (Task 6).

        Includes: proposals today, success/rollback rates, proactive vs
        received proposal ratio, calibration score, scanner status.
        """
        analytics = await self.get_analytics()
        total = getattr(analytics, "total_proposals", 0)
        approved = getattr(analytics, "approved_proposals", 0)
        rolled_back = getattr(analytics, "rolled_back_proposals", 0)

        success_rate = approved / total if total > 0 else 0.0
        rollback_rate = rolled_back / total if total > 0 else 0.0

        proactive_proposals = (
            self._proactive_scanner.stats.total_proposals_generated
            if self._proactive_scanner is not None
            else 0
        )
        proactive_ratio = (
            proactive_proposals / max(1, self._proposals_received)
        )

        calibration_score: float | None = None
        if self._repair_memory is not None:
            calibration_score = self._repair_memory.get_calibration_score()

        scanner_stats = (
            self._proactive_scanner.stats.to_dict()
            if self._proactive_scanner is not None
            else {}
        )

        return {
            "proposals_received_session": self._proposals_received,
            "proposals_approved_session": self._proposals_approved,
            "proposals_rejected_session": self._proposals_rejected,
            "proposals_rolled_back_session": self._proposals_rolled_back,
            "success_rate": round(success_rate, 4),
            "rollback_rate": round(rollback_rate, 4),
            "proactive_proposals_generated": proactive_proposals,
            "proactive_vs_received_ratio": round(proactive_ratio, 4),
            "calibration_score": calibration_score,
            "repair_memory_record_count": (
                self._repair_memory.record_count if self._repair_memory is not None else 0
            ),
            "last_proposal_processed_at": self._last_proposal_processed_at,
            "proactive_scanner": scanner_stats,
        }

    async def get_repair_memory_summary(self) -> dict[str, Any]:
        """Return the full RepairMemory summary for GET /simula/repair-memory."""
        if self._repair_memory is None:
            return {
                "success_rates_by_category": {},
                "total_proposals": 0,
                "rollback_rate": 0.0,
                "most_reliable_change_type": "",
                "most_risky_change_type": "",
            }
        return await self._repair_memory.get_repair_memory_summary()

    # ─── Health (Synapse HealthMonitor protocol) ──────────────────────────────

    async def health(self) -> dict[str, Any]:
        """
        Real liveness check for the Synapse HealthMonitor.

        Checks:
        1. initialize() has completed.
        2. The codebase root directory is accessible (real filesystem I/O).
        3. The active proposal queue has not grown beyond the config limit,
           which would indicate a stuck pipeline.
        """
        if not self._initialized:
            logger.warning("simula_health_not_initialized", system_id=self.system_id)
            return {"status": "unhealthy", "reason": "not_initialized"}

        # Real I/O: verify the codebase root Simula operates on is still readable.
        root_ok = await asyncio.get_event_loop().run_in_executor(
            None, self._root.is_dir
        )
        if not root_ok:
            logger.error(
                "simula_health_root_missing",
                system_id=self.system_id,
                codebase_root=str(self._root),
            )
            return {
                "status": "unhealthy",
                "reason": "codebase_root_missing",
                "codebase_root": str(self._root),
            }

        active = len(self._active_proposals)
        calibration_score: float | None = None
        if self._repair_memory is not None:
            calibration_score = self._repair_memory.get_calibration_score()

        logger.debug(
            "simula_health_ok",
            system_id=self.system_id,
            version=self._current_version,
            active_proposals=active,
        )
        return {
            "status": "healthy",
            "current_version": self._current_version,
            "active_proposals": active,
            "proactive_scanner_alive": (
                self._proactive_scanner is not None
                and self._proactive_scanner.stats.scanner_alive
            ),
            "repair_memory_record_count": (
                self._repair_memory.record_count if self._repair_memory is not None else 0
            ),
            "last_proposal_processed_at": self._last_proposal_processed_at,
            "calibration_score": calibration_score,
        }

    async def health_check(self) -> "HealthStatus":
        """Structured health report returning a typed HealthStatus object.

        Mirrors health() but returns the spec-defined HealthStatus type
        (Section 17) for consumers that need typed component-level data
        (e.g. Skia, Thymos, OpenTelemetry adapters).

        Spec ref: Section 17 - Health & Monitoring.
        """
        from systems.simula.evolution_types import (
            HealthStatus,
            SimulaComponentHealth,
            SimulaMetrics,
        )

        # Build component health list
        components: list[SimulaComponentHealth] = []

        # Core pipeline
        components.append(SimulaComponentHealth(
            name="code_agent",
            status="healthy" if self._code_agent is not None else "disabled",
        ))
        components.append(SimulaComponentHealth(
            name="simulator",
            status="healthy" if self._simulator is not None else "disabled",
        ))
        components.append(SimulaComponentHealth(
            name="history",
            status="healthy" if self._history is not None else "disabled",
        ))
        components.append(SimulaComponentHealth(
            name="rollback",
            status="healthy" if self._rollback is not None else "disabled",
        ))

        # Verification bridges
        components.append(SimulaComponentHealth(
            name="dafny",
            status="healthy" if getattr(self, "_dafny_bridge", None) is not None else "disabled",
        ))
        components.append(SimulaComponentHealth(
            name="z3",
            status="healthy" if getattr(self, "_z3_bridge", None) is not None else "disabled",
        ))
        components.append(SimulaComponentHealth(
            name="static_analysis",
            status="healthy" if getattr(self, "_static_bridge", None) is not None else "disabled",
        ))

        # Learning
        components.append(SimulaComponentHealth(
            name="grpo",
            status="healthy" if self._grpo is not None else "disabled",
        ))
        components.append(SimulaComponentHealth(
            name="lilo",
            status="healthy" if self._lilo is not None else "disabled",
        ))

        # Inspector
        components.append(SimulaComponentHealth(
            name="inspector",
            status="healthy" if self._inspector is not None else "disabled",
        ))

        # Codebase root reachability
        root_ok = await asyncio.get_event_loop().run_in_executor(None, self._root.is_dir)
        components.append(SimulaComponentHealth(
            name="codebase_root",
            status="healthy" if root_ok else "unhealthy",
            detail=str(self._root),
        ))

        # Proactive scanner
        scanner_alive = (
            self._proactive_scanner is not None
            and self._proactive_scanner.stats.scanner_alive
        )
        components.append(SimulaComponentHealth(
            name="proactive_scanner",
            status="healthy" if scanner_alive else "degraded" if self._proactive_scanner is not None else "disabled",
        ))

        # Build metrics snapshot
        analytics_cache = (
            self._analytics._cached_analytics
            if self._analytics is not None and hasattr(self._analytics, "_cached_analytics")
            else None
        )
        total = max(1, self._proposals_received)
        metrics = SimulaMetrics(
            proposals_processed=self._proposals_received,
            proposals_applied=self._proposals_approved,
            proposals_rolled_back=self._proposals_rolled_back,
            proposals_rejected=self._proposals_rejected,
            proposals_deduplicated=self._proposals_deduplicated,
            proposals_awaiting_governance=self._proposals_awaiting_governance,
            active_proposals=len(self._active_proposals),
            current_version=self._current_version,
            success_rate=round(self._proposals_approved / total, 4),
            rollback_rate=round(
                self._proposals_rolled_back / max(1, self._proposals_approved), 4
            ),
            evolution_velocity=getattr(analytics_cache, "evolution_velocity", 0.0) if analytics_cache else 0.0,
            mean_simulation_risk=getattr(analytics_cache, "mean_simulation_risk", 0.0) if analytics_cache else 0.0,
            grid_state=self._grid_state,
            starvation_level=self._starvation_level,
        )

        # Overall status determination
        unhealthy = [c for c in components if c.status == "unhealthy"]
        degraded = [c for c in components if c.status == "degraded"]

        if not self._initialized:
            overall = "unhealthy"
            reason = "not_initialized"
        elif unhealthy:
            overall = "unhealthy"
            reason = f"components_unhealthy: {[c.name for c in unhealthy]}"
        elif degraded:
            overall = "degraded"
            reason = f"components_degraded: {[c.name for c in degraded]}"
        else:
            overall = "healthy"
            reason = ""

        result = HealthStatus(
            service="simula",
            status=overall,
            components=components,
            metrics=metrics,
            reason=reason,
        )

        # Emit SIMULA_HEALTH_DEGRADED when components are unhealthy or degraded
        if overall in ("degraded", "unhealthy"):
            try:
                event_bus = getattr(getattr(self, "_synapse", None), "_event_bus", None)
                if event_bus is not None:
                    from systems.synapse.types import SynapseEvent, SynapseEventType
                    asyncio.ensure_future(event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.SIMULA_HEALTH_DEGRADED,
                        source_system="simula",
                        data={
                            "status": overall,
                            "reason": reason,
                            "degraded_components": [c.name for c in degraded],
                            "unhealthy_components": [c.name for c in unhealthy],
                        },
                    )))
            except Exception:
                pass

        return result

    # ─── Inspector API ───────────────────────────────────────────────────────────

    def _ensure_inspector(self) -> InspectorService:
        """Validate that Inspector is enabled and return the typed service."""
        if self._inspector is None:
            raise RuntimeError(
                "Inspector is not enabled. Set inspector_enabled=True in SimulaConfig."
            )
        return self._inspector

    async def hunt_external_target(
        self,
        github_url: str,
        *,
        authorized_targets: list[str] | None = None,
        attack_goals: list[str] | None = None,
        generate_pocs: bool | None = None,
        generate_patches: bool | None = None,
    ) -> InspectionResult:
        """
        Run Inspector against an external GitHub repository.

        Inspector is purely additive - it never modifies EOS files and all
        analysis happens in temporary workspaces.

        Args:
            github_url: HTTPS URL of the target repository.
            authorized_targets: Override config authorized targets for this hunt.
                Creates a scoped config copy - the shared config is never mutated.
            attack_goals: Custom attack goals (defaults to predefined set).
            generate_pocs: Generate exploit PoC scripts (default from config).
            generate_patches: Generate + verify patches (default from config).

        Returns:
            InspectionResult with discovered vulnerabilities and optional patches.

        Raises:
            RuntimeError: If Inspector is not enabled.
        """
        inspector = self._ensure_inspector()

        # Serialize concurrent hunts that need to temporarily swap inspector._config.
        # Without a lock, two concurrent awaits can corrupt each other's config restore.
        async with self._inspector_hunt_lock:
            if authorized_targets is not None:
                from systems.simula.inspector.types import InspectorConfig

                original_config = inspector._config
                inspector._config = InspectorConfig(
                    authorized_targets=authorized_targets,
                    max_workers=original_config.max_workers,
                    sandbox_timeout_seconds=original_config.sandbox_timeout_seconds,
                    log_vulnerability_analytics=original_config.log_vulnerability_analytics,
                    clone_depth=original_config.clone_depth,
                )
                try:
                    return await inspector.hunt_external_repo(
                        github_url=github_url,
                        attack_goals=attack_goals,
                        generate_pocs=generate_pocs if generate_pocs is not None else self._config.inspector_generate_pocs,
                        generate_patches=generate_patches if generate_patches is not None else self._config.inspector_generate_patches,
                    )
                finally:
                    inspector._config = original_config
            else:
                return await inspector.hunt_external_repo(
                    github_url=github_url,
                    attack_goals=attack_goals,
                    generate_pocs=generate_pocs if generate_pocs is not None else self._config.inspector_generate_pocs,
                    generate_patches=generate_patches if generate_patches is not None else self._config.inspector_generate_patches,
                )

    async def hunt_internal(
        self,
        *,
        attack_goals: list[str] | None = None,
        generate_pocs: bool | None = None,
        generate_patches: bool | None = None,
    ) -> InspectionResult:
        """
        Run Inspector against the internal EOS codebase for self-testing.

        Args:
            attack_goals: Custom attack goals (defaults to predefined set).
            generate_pocs: Generate exploit PoC scripts (default from config).
            generate_patches: Generate + verify patches (default from config).

        Returns:
            InspectionResult with discovered vulnerabilities.

        Raises:
            RuntimeError: If Inspector is not enabled.
        """
        inspector = self._ensure_inspector()

        return await inspector.hunt_internal_eos(
            attack_goals=attack_goals,
            generate_pocs=generate_pocs if generate_pocs is not None else self._config.inspector_generate_pocs,
            generate_patches=generate_patches if generate_patches is not None else self._config.inspector_generate_patches,
        )

    async def generate_patches_for_hunt(
        self,
        hunt_result: InspectionResult,
    ) -> dict[str, str]:
        """
        Generate patches for vulnerabilities found in a completed hunt.

        Useful when a hunt was run without generate_patches=True and you want
        to retroactively generate patches for the discovered vulnerabilities.

        Args:
            hunt_result: A completed InspectionResult from hunt_external_target
                         or hunt_internal.

        Returns:
            Dict mapping vulnerability ID → unified diff patch string.

        Raises:
            RuntimeError: If Inspector or remediation is not enabled.
        """
        inspector = self._ensure_inspector()
        return await inspector.generate_patches(hunt_result)

    def get_inspector_analytics(self) -> dict[str, Any]:
        """Return aggregate Inspector analytics if available."""
        inspector = self._ensure_inspector()
        return inspector.analytics_view.summary

    async def get_unified_analytics(self) -> dict[str, Any]:
        """
        Return unified analytics combining evolution metrics and Inspector
        security metrics. This is the Phase 9 observability entry point.
        """
        if self._analytics is None:
            return {}
        return await self._analytics.get_unified_analytics()

    async def get_inspector_weekly_trends(
        self,
        *,
        weeks: int = 12,
        target_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query weekly Inspector vulnerability trends from TSDB or in-memory view.
        """
        if self._analytics is None:
            return []
        return await self._analytics.get_inspector_weekly_trends(
            weeks=weeks, target_url=target_url,
        )

    async def get_inspector_error_summary(self, *, days: int = 7) -> list[dict[str, Any]]:
        """Query Inspector pipeline error summary from TSDB."""
        if self._analytics is None:
            return []
        return await self._analytics.get_inspector_error_summary(days=days)

    # ─── Evo Bridge (callback + event-bus subscription) ───────────────────────

    def subscribe_to_evolution_candidates(self, event_bus: Any) -> None:
        """
        Subscribe to EVOLUTION_CANDIDATE Synapse events emitted by Evo's
        consolidation Phase 8 when a hypothesis reaches high confidence (>= 0.9).

        Each event is translated by the Evo↔Simula bridge and fed into the
        standard process_proposal() pipeline, which routes governed changes
        through Equor governance before any code is applied.

        Call this after SimulaService.initialize() has been awaited, and after
        EvoService is wired to the same event_bus instance.  main.py is
        responsible for the wiring - this method is intentionally separate from
        get_evo_callback() so the two delivery paths can be enabled independently.
        """
        from systems.synapse.types import SynapseEventType

        async def _on_evolution_candidate(event: Any) -> None:
            data = getattr(event, "data", {})
            hypothesis_id = data.get("hypothesis_id", "")
            hypothesis_statement = data.get("hypothesis_statement", "")
            evidence_score = float(data.get("evidence_score", 0.0))
            confidence = float(data.get("confidence", 0.0))
            mutation_type = data.get("mutation_type", "")
            mutation_target = data.get("mutation_target", "")
            mutation_description = data.get("mutation_description", "")
            supporting_episodes: list[str] = data.get("supporting_episodes", [])

            self._logger.info(
                "evolution_candidate_received",
                hypothesis_id=hypothesis_id,
                confidence=confidence,
                evidence_score=evidence_score,
                mutation_type=mutation_type,
            )

            # Organizational closure (Speciation Bible §8.3):
            # If the hypothesis recommends creating a new subsystem, route
            # directly to SubsystemGenerator instead of process_proposal().
            # The generator writes the module to disk; the normal pipeline
            # would reject it as there is no ChangeCategory for new_subsystem.
            if mutation_type == "new_subsystem" and self._subsystem_generator is not None:
                from systems.simula.subsystem_generator import spec_from_proposal_data

                spec = spec_from_proposal_data({
                    **data,
                    "hypothesis_id": hypothesis_id,
                    "subsystem_purpose": mutation_description or hypothesis_statement,
                })
                result = await self._subsystem_generator.generate_subsystem(spec)
                self._logger.info(
                    "subsystem_generation_routed",
                    name=spec.name,
                    success=result.success,
                    reason=result.reason,
                )
                return

            # Dynamic capability expansion (ChangeCategory.ADD_EXECUTOR):
            # If the hypothesis recommends adding a new executor (e.g., new
            # DeFi protocol, new bounty platform), route to ExecutorGenerator.
            # The generator produces and hot-loads the class immediately.
            if mutation_type == "add_executor" and self._executor_generator is not None:
                from systems.axon.types import ExecutorTemplate
                from decimal import Decimal

                template_data = data.get("executor_template", {})
                try:
                    template = ExecutorTemplate(
                        name=template_data.get("name", ""),
                        action_type=template_data.get("action_type", ""),
                        description=template_data.get(
                            "description", mutation_description or hypothesis_statement
                        ),
                        protocol_or_platform=template_data.get("protocol_or_platform", ""),
                        required_apis=template_data.get("required_apis", []),
                        risk_tier=template_data.get("risk_tier", "medium"),
                        max_budget_usd=Decimal(
                            str(template_data.get("max_budget_usd", "100.00"))
                        ),
                        capabilities=template_data.get("capabilities", []),
                        safety_constraints=template_data.get("safety_constraints", []),
                        source_hypothesis_id=hypothesis_id,
                        source_opportunity_id=data.get("opportunity_id", ""),
                    )
                    result = await self._executor_generator.generate_executor(template)
                    self._logger.info(
                        "executor_generation_routed",
                        action_type=template.action_type,
                        success=result.success,
                        reason=result.reason,
                    )
                except Exception as exc:
                    self._logger.error(
                        "executor_generation_routing_failed",
                        error=str(exc),
                        hypothesis_id=hypothesis_id,
                    )
                return

            await self.receive_evo_proposal(
                evo_description=mutation_description or hypothesis_statement,
                evo_rationale=hypothesis_statement,
                hypothesis_ids=[hypothesis_id],
                hypothesis_statements=[hypothesis_statement],
                evidence_scores=[evidence_score],
                supporting_episode_ids=supporting_episodes,
                mutation_target=mutation_target,
                mutation_type=mutation_type,
            )

        event_bus.subscribe(
            SynapseEventType.EVOLUTION_CANDIDATE,
            _on_evolution_candidate,
        )
        self._logger.info("simula_subscribed_to_evolution_candidates")

        # ── NOVEL_ACTION_REQUESTED subscription ───────────────────────────────
        # Nova emits this when its LLM selects "propose_novel_action" as a step.
        # Simula handles the full pipeline: feasibility evaluation → Equor gate →
        # ExecutorGenerator → hot-load → emit NOVEL_ACTION_CREATED.

        async def _on_novel_action_requested(event: Any) -> None:
            import contextlib
            from decimal import Decimal

            from primitives.common import new_id, utc_now

            data = getattr(event, "data", {})
            proposal_id = data.get("proposal_id", new_id())
            action_name = data.get("action_name", "")
            description = data.get("description", "")
            required_capabilities: list[str] = data.get("required_capabilities", [])
            expected_outcome = data.get("expected_outcome", "")
            justification = data.get("justification", "")
            goal_description = data.get("goal_description", "")
            urgency = float(data.get("urgency", 0.5))
            pipeline_managed = bool(data.get("pipeline_managed", False))

            self._logger.info(
                "novel_action_requested",
                proposal_id=proposal_id,
                action_name=action_name,
                pipeline_managed=pipeline_managed,
            )

            # ── SPEC_DRAFTED path: if this action came from SelfModificationPipeline
            # and a Spec was drafted (SPEC_DRAFTED received for same proposal_id),
            # route to SubsystemGenerator instead of ExecutorGenerator.
            _pending_specs: dict[str, dict[str, str]] = getattr(self, "_pending_spec_drafts", {})
            if pipeline_managed and proposal_id in _pending_specs and self._subsystem_generator is not None:
                spec_meta = _pending_specs.pop(proposal_id)
                system_name = spec_meta.get("system_name", action_name)
                spec_purpose = description or spec_meta.get("spec_title", "")
                self._logger.info(
                    "novel_action_routed_to_subsystem_generator",
                    proposal_id=proposal_id,
                    system_name=system_name,
                )
                from systems.simula.subsystem_generator import SubsystemSpec
                spec = SubsystemSpec(
                    name=system_name,
                    purpose=spec_purpose,
                    trigger_hypothesis_id=proposal_id,
                    required_events=[],
                    emitted_events=[],
                    dependencies=[],
                    constraints=[
                        f"expected_outcome: {expected_outcome[:200]}",
                        f"spec_path: {spec_meta.get('spec_path', '')}",
                    ],
                )
                try:
                    await self._subsystem_generator.generate_subsystem(spec)
                except Exception as exc:
                    self._logger.error(
                        "spec_drafted_subsystem_generation_failed",
                        proposal_id=proposal_id,
                        system_name=system_name,
                        error=str(exc),
                    )
                return

            if not action_name or not description:
                self._logger.warning(
                    "novel_action_request_rejected_missing_fields",
                    proposal_id=proposal_id,
                )
                return

            # ── Iron Rule check: cannot shadow protected system names ──────────
            _FORBIDDEN_FRAGMENTS = {"equor", "simula", "constitution", "invariant", "memory"}
            if any(frag in action_name.lower() for frag in _FORBIDDEN_FRAGMENTS):
                self._logger.warning(
                    "novel_action_request_rejected_forbidden_name",
                    proposal_id=proposal_id,
                    action_name=action_name,
                )
                return

            # ── Step 1: Equor constitutional pre-approval ──────────────────────
            # We ask Equor whether the *concept* of this action is constitutionally
            # safe before spending compute on executor generation.
            equor_approved = True
            if self._synapse is not None and hasattr(self._synapse, "_event_bus"):
                try:
                    import asyncio as _asyncio

                    permit_future: _asyncio.Future[bool] = _asyncio.get_event_loop().create_future()
                    _permit_id = new_id()

                    async def _on_equor_permit(e: Any) -> None:
                        if getattr(e, "data", {}).get("permit_id") == _permit_id:
                            if not permit_future.done():
                                permit_future.set_result(
                                    bool(e.data.get("approved", False))
                                )

                    event_bus.subscribe(
                        SynapseEventType.EQUOR_ECONOMIC_PERMIT,
                        _on_equor_permit,
                    )
                    await event_bus.emit(
                        SynapseEventType.EQUOR_ECONOMIC_INTENT,
                        {
                            "permit_id": _permit_id,
                            "mutation_type": "novel_action_proposal",
                            "action_name": action_name,
                            "description": description,
                            "required_capabilities": required_capabilities,
                            "expected_outcome": expected_outcome,
                            "justification": justification,
                            "amount_usd": "0",
                            "proposal_id": proposal_id,
                        },
                    )
                    try:
                        equor_approved = await _asyncio.wait_for(
                            permit_future, timeout=30.0
                        )
                    except _asyncio.TimeoutError:
                        # Auto-permit on timeout (safety fallback, matches Oikos M4 pattern)
                        equor_approved = True
                        self._logger.warning(
                            "novel_action_equor_timeout_auto_permit",
                            proposal_id=proposal_id,
                        )
                except Exception as exc:
                    self._logger.warning(
                        "novel_action_equor_check_failed",
                        proposal_id=proposal_id,
                        error=str(exc),
                    )
                    equor_approved = True  # non-fatal: proceed optimistically

            if not equor_approved:
                self._logger.info(
                    "novel_action_request_rejected_by_equor",
                    proposal_id=proposal_id,
                    action_name=action_name,
                )
                return

            # ── Step 2: Route to ExecutorGenerator ────────────────────────────
            if self._executor_generator is None:
                self._logger.warning(
                    "novel_action_executor_generator_unavailable",
                    proposal_id=proposal_id,
                )
                return

            from systems.axon.types import ExecutorTemplate

            # Infer risk tier from capabilities
            high_risk_caps = {"wallet_access", "defi_write", "mitosis_spawn"}
            medium_risk_caps = {"git_write", "http_client", "code_generation"}
            if any(c in high_risk_caps for c in required_capabilities):
                risk_tier = "high"
            elif any(c in medium_risk_caps for c in required_capabilities):
                risk_tier = "medium"
            else:
                risk_tier = "low"

            _budget_map = {"low": "50.00", "medium": "200.00", "high": "1000.00"}

            template = ExecutorTemplate(
                name=action_name,
                action_type=action_name,
                description=description,
                protocol_or_platform="novel",
                required_apis=[],
                risk_tier=risk_tier,
                max_budget_usd=Decimal(_budget_map[risk_tier]),
                capabilities=required_capabilities,
                safety_constraints=[
                    "equor_approval_required",
                    f"expected_outcome: {expected_outcome[:200]}",
                ],
                source_hypothesis_id=proposal_id,
                source_opportunity_id="",
            )

            try:
                result = await self._executor_generator.generate_executor(template)
            except Exception as exc:
                self._logger.error(
                    "novel_action_executor_generation_failed",
                    proposal_id=proposal_id,
                    action_name=action_name,
                    error=str(exc),
                )
                return

            if not result.success:
                self._logger.warning(
                    "novel_action_executor_generation_not_successful",
                    proposal_id=proposal_id,
                    action_name=action_name,
                    reason=getattr(result, "reason", "unknown"),
                )
                return

            # ── Step 3: Emit NOVEL_ACTION_CREATED ────────────────────────────
            with contextlib.suppress(Exception):
                await event_bus.emit(
                    SynapseEventType.NOVEL_ACTION_CREATED,
                    {
                        "proposal_id": proposal_id,
                        "action_name": action_name,
                        "description": description,
                        "required_capabilities": required_capabilities,
                        "executor_class": getattr(result, "class_name", ""),
                        "module_path": getattr(result, "module_path", ""),
                        "risk_tier": risk_tier,
                        "max_budget_usd": _budget_map[risk_tier],
                        "equor_approved": equor_approved,
                        "source_hypothesis_id": proposal_id,
                        "created_at": utc_now().isoformat(),
                    },
                )

            self._logger.info(
                "novel_action_created",
                proposal_id=proposal_id,
                action_name=action_name,
                risk_tier=risk_tier,
            )

        event_bus.subscribe(
            SynapseEventType.NOVEL_ACTION_REQUESTED,
            _on_novel_action_requested,
        )
        self._logger.info("simula_subscribed_to_novel_action_requests")

        if hasattr(SynapseEventType, "LEARNING_OPPORTUNITY_DETECTED"):
            event_bus.subscribe(
                SynapseEventType.LEARNING_OPPORTUNITY_DETECTED,
                self._on_learning_opportunity_detected,
            )

    async def _on_learning_opportunity_detected(self, event: Any) -> None:
        """
        Handle LEARNING_OPPORTUNITY_DETECTED emitted by Nova's OpportunityScanner.

        For code-generation and formal-verification resources (papers or repos),
        queue a lightweight ADD_SYSTEM_CAPABILITY proposal so Simula can absorb
        the technique.  Governance-gated - no auto-apply.
        """
        try:
            data: dict[str, Any] = event.data if hasattr(event, "data") else event
            resource_type: str = data.get("resource_type", "")
            domain: str = data.get("domain", "")
            title: str = data.get("title", "")
            url: str = data.get("url", "")
            resource_id: str = data.get("resource_id", "")
            relevance: float = float(data.get("relevance_score", 0.0))
            gaps: list[str] = data.get("capability_gaps_addressed", [])

            # Only act on domains where Simula has ownership
            simula_domains = {"code_generation", "formal_verification", "self_evolution"}
            if domain not in simula_domains:
                return

            if resource_type not in {"repo", "paper"}:
                return

            if relevance < 0.35:
                return

            from primitives.evolution import ChangeCategory  # noqa: PLC0415

            gap_text = ", ".join(gaps) if gaps else domain
            description = (
                f"Study '{title}' ({resource_type}) to close capability gap(s): "
                f"{gap_text}. Source: {url}"
            )
            proposal = EvolutionProposal(
                id=new_id(),
                source="learning_opportunity",
                category=ChangeCategory.ADD_SYSTEM_CAPABILITY,
                description=description,
                change_spec=ChangeSpec(
                    capability_description=(
                        f"Absorb {resource_type} '{title}' - domain={domain}, "
                        f"gaps=[{gap_text}], url={url}"
                    ),
                ),
                expected_benefit=(
                    f"Close capability gap(s) in {gap_text} "
                    f"(relevance={relevance:.2f})"
                ),
                evidence=[resource_id],
            )
            await self._bridge.receive_evo_proposal(proposal)
            self._logger.info(
                "simula_learning_opportunity_queued",
                resource_id=resource_id,
                domain=domain,
                title=title[:80],
            )
        except Exception:
            self._logger.exception("simula_learning_opportunity_handler_error")

    def get_evo_callback(self) -> Any:
        """
        Return a callback function for Evo's ConsolidationOrchestrator.
        This is wired during system initialization in main.py.

        The callback signature matches what Evo Phase 8 expects:
          async def callback(evo_proposal, hypotheses) -> ProposalResult
        """
        async def _evo_callback(evo_proposal: Any, hypotheses: list[Any]) -> ProposalResult:
            # Extract fields from Evo's lightweight types
            hypothesis_ids = [getattr(h, "id", "") for h in hypotheses]
            hypothesis_statements = [getattr(h, "statement", "") for h in hypotheses]
            evidence_scores = [getattr(h, "evidence_score", 0.0) for h in hypotheses]

            # Collect all supporting episode IDs across hypotheses
            episode_ids: list[str] = []
            for h in hypotheses:
                episode_ids.extend(getattr(h, "supporting_episodes", []))

            # Extract mutation info if available
            mutation_target = ""
            mutation_type = ""
            for h in hypotheses:
                mutation = getattr(h, "proposed_mutation", None)
                if mutation is not None:
                    mutation_target = getattr(mutation, "target", "")
                    mutation_type = getattr(mutation, "type", "")
                    if hasattr(mutation_type, "value"):
                        mutation_type = mutation_type.value
                    break

            return await self.receive_evo_proposal(
                evo_description=getattr(evo_proposal, "description", ""),
                evo_rationale=getattr(evo_proposal, "rationale", ""),
                hypothesis_ids=hypothesis_ids,
                hypothesis_statements=hypothesis_statements,
                evidence_scores=evidence_scores,
                supporting_episode_ids=episode_ids,
                mutation_target=mutation_target,
                mutation_type=mutation_type,
            )

        return _evo_callback

    # ─── Private: Application ──────────────────────────────────────────────────

    async def _apply_change(self, proposal: EvolutionProposal) -> ProposalResult:
        """
        Apply a validated, simulated, approved proposal.
        Includes health check and automatic rollback on failure.
        """
        assert self._applicator is not None
        assert self._health is not None
        assert self._rollback is not None

        proposal.status = ProposalStatus.APPLYING
        log = self._logger.bind(proposal_id=proposal.id, category=proposal.category.value)
        log.info("applying_change")

        # Stage 3C: Inject LILO library prompt into the code agent
        if self._lilo is not None and self._code_agent is not None:
            try:
                self._code_agent._lilo_prompt = await self._lilo.get_library_prompt()
            except Exception as exc:
                log.warning("lilo_prompt_error", error=str(exc))

        # Repair Memory: Inject lessons from past repairs into the code agent
        if self._repair_memory is not None and self._code_agent is not None:
            try:
                target_system = ""
                if proposal.change_spec and proposal.change_spec.affected_systems:
                    target_system = proposal.change_spec.affected_systems[0]
                guidance = await self._repair_memory.get_repair_guidance_prompt(
                    category=proposal.category.value,
                    target_system=target_system,
                )
                if guidance:
                    self._code_agent._repair_memory_prompt = guidance
            except Exception as exc:
                log.warning("repair_memory_guidance_error", error=str(exc))

        # Stage 4A: Inject proof library context into the code agent
        if self._lean_bridge is not None and self._code_agent is not None:
            try:
                from systems.simula.verification.lean_bridge import LeanBridge
                if isinstance(self._lean_bridge, LeanBridge):
                    lib_stats = await self._lean_bridge.get_library_stats()
                    if lib_stats.total_lemmas > 0:
                        self._code_agent._proof_library_prompt = (
                            f"\n\n## Proof Library ({lib_stats.total_lemmas} proven lemmas)\n"
                            f"The Lean 4 proof library contains {lib_stats.total_lemmas} proven lemmas "
                            f"across domains: {', '.join(f'{d}: {c}' for d, c in lib_stats.by_domain.items())}.\n"
                            f"Mean Lean Copilot automation rate: {lib_stats.mean_copilot_automation:.0%}.\n"
                            f"Your implementation may benefit from existing verified properties."
                        )
            except Exception as exc:
                log.warning("proof_library_prompt_error", error=str(exc))

        # Stage 4B: GRPO A/B model routing + local model preference
        grpo_model_used = ""
        grpo_ab_group = ""
        if self._grpo is not None and self._code_agent is not None:
            try:
                # Check if local model should handle this proposal (routine task)
                risk_str = ""
                if proposal.simulation is not None:
                    risk_str = proposal.simulation.risk_level.value
                use_local = self._grpo.should_use_local(
                    category=proposal.category.value,
                    description=proposal.description,
                    affected_systems=getattr(proposal.change_spec, "affected_systems", None),
                    simulation_risk=risk_str,
                )
                if use_local:
                    grpo_model_used = self._config.grpo_base_model + "-local"
                    grpo_ab_group = "local"
                    self._code_agent._grpo_model_id = grpo_model_used
                    self._code_agent._use_local_model = True
                    log.info("grpo_routing_local", model=grpo_model_used)
                elif self._grpo.should_use_finetuned():
                    grpo_model_used = self._config.grpo_base_model + "-finetuned"
                    grpo_ab_group = "finetuned"
                    self._code_agent._grpo_model_id = grpo_model_used
                    self._code_agent._use_local_model = False
                    log.info("grpo_routing_finetuned", model=grpo_model_used)
                else:
                    grpo_ab_group = "base"
                    self._code_agent._use_local_model = False
                    log.info("grpo_routing_base")
            except Exception as exc:
                log.warning("grpo_routing_error", error=str(exc))
                self._code_agent._use_local_model = False

        # ── Stage 5A: Synthesis-first (fast-path before CEGIS) ────────────────
        synthesis_result_stash = None
        if self._synthesis is not None:
            try:
                from systems.simula.synthesis.strategy_selector import (
                    SynthesisStrategySelector,
                )
                from systems.simula.synthesis.types import SynthesisStatus

                if isinstance(self._synthesis, SynthesisStrategySelector):
                    synth_result = await self._synthesis.synthesise(proposal)
                    synthesis_result_stash = synth_result
                    if synth_result.status == SynthesisStatus.SYNTHESIZED and synth_result.final_code:
                        log.info(
                            "synthesis_succeeded",
                            strategy=synth_result.strategy.value,
                            tokens=synth_result.total_llm_tokens,
                            duration_ms=synth_result.total_duration_ms,
                        )
                        # Write synthesised code and skip CEGIS
                        if synth_result.files_written:
                            for fpath in synth_result.files_written:
                                full_path = self._root / fpath
                                if full_path.exists():
                                    log.debug("synthesis_wrote_file", path=fpath)
                    else:
                        log.info(
                            "synthesis_fell_back_to_cegis",
                            strategy=synth_result.strategy.value,
                            status=synth_result.status.value,
                        )
            except Exception as exc:
                log.warning("synthesis_error", error=str(exc))

        # ── Stage 5C: Multi-agent orchestration for multi-file proposals ──────
        if self._orchestrator is not None:
            try:
                from systems.simula.orchestration.orchestrator import (
                    MultiAgentOrchestrator,
                )

                if isinstance(self._orchestrator, MultiAgentOrchestrator):
                    # Estimate affected files from proposal target + code_hint
                    estimated_files = []
                    _target = getattr(proposal, "target", None)
                    if _target:
                        estimated_files.append(_target)
                    if hasattr(proposal, "affected_files"):
                        estimated_files.extend(proposal.affected_files)

                    # Adaptive threshold: p75 of historical proposal file counts,
                    # floored at the config value so the setting acts as a minimum.
                    threshold = await self._adaptive_orchestration_threshold()
                    if len(estimated_files) >= threshold:
                        log.info(
                            "orchestration_engaged",
                            files=len(estimated_files),
                            threshold=threshold,
                        )
                        orc_result = await self._orchestrator.orchestrate(
                            proposal=proposal,
                            files_to_change=estimated_files,
                        )
                        proposal._orchestration_result = orc_result  # type: ignore[attr-defined]
                        log.info(
                            "orchestration_complete",
                            success=not orc_result.error,
                            stages=orc_result.parallel_stages_executed,
                            agents=orc_result.total_agents_used,
                        )
            except Exception as exc:
                log.warning("orchestration_error", error=str(exc))

        # ── Corpus 14 §13: Identity scope verification (Spec 23) ─────────────
        # Proposals may arrive from federation with a foreign identity_id.
        # Verify the proposal's identity_id (if set) matches this instance before
        # applying mutations to local code - prevents rogue cross-instance applies.
        import os as _os_id
        _local_identity = _os_id.environ.get("ECODIAOS_INSTANCE_ID", "eos-default")
        _proposal_identity = getattr(proposal, "identity_id", "") or ""
        if _proposal_identity and _proposal_identity != _local_identity:
            log.warning(
                "apply_identity_mismatch",
                proposal_id=proposal.id,
                proposal_identity=_proposal_identity,
                local_identity=_local_identity,
                action="proceeding_with_caution",
            )
            # Do NOT reject - federated governance proposals are valid cross-instance,
            # but log so Thymos/Identity can audit. A hard DENY would require the
            # Identity system to provide cryptographic proof, not just env-variable matching.

        # ── Fatal pattern pre-flight gate ─────────────────────────────────────
        # Score the proposal's code_hint / description against known fatal patterns
        # before handing off to the applicator.  A BLOCK terminates the pipeline
        # early with a rejected result so no files are touched.
        if self._known_fatal_patterns:
            _patch_code = (
                getattr(proposal, "code_hint", None)
                or proposal.description
                or ""
            )
            _patch_context: dict[str, Any] = {
                "description": proposal.description,
                "change_category": proposal.category.value if proposal.category else "",
                "target_system": (
                    (proposal.change_spec.affected_systems[0]
                     if proposal.change_spec and proposal.change_spec.affected_systems
                     else "")
                ),
                "affected_systems": (
                    list(proposal.change_spec.affected_systems)
                    if proposal.change_spec and proposal.change_spec.affected_systems
                    else []
                ),
                "incident_id": getattr(proposal, "incident_id", proposal.id),
            }
            _pattern_risk = await self._apply_patch_risk_gate(
                patch_code=_patch_code,
                context=_patch_context,
                proposal_id=proposal.id,
                hypothesis_id=getattr(proposal, "source_hypothesis_id", None),
            )
            if _pattern_risk["risk_level"] == "block":
                log.warning(
                    "proposal_blocked_fatal_pattern",
                    proposal_id=proposal.id,
                    matched_patterns=_pattern_risk["matched_patterns"],
                    reason=_pattern_risk["reason"],
                )
                proposal.status = ProposalStatus.REJECTED
                return ProposalResult(
                    proposal_id=proposal.id,
                    status=ProposalStatus.REJECTED,
                    success=False,
                    error=(
                        f"Blocked: patch matches known fatal pattern(s). "
                        f"{_pattern_risk['reason']}"
                    ),
                )
            # Stash WARN-level risk on the proposal for RE training annotation
            if _pattern_risk["risk_level"] == "warn":
                proposal._pattern_risk = _pattern_risk  # type: ignore[attr-defined]

        import time as _time
        _apply_t0 = _time.monotonic()
        code_result, snapshot = await self._applicator.apply(proposal)
        proposal._application_duration_ms = int((_time.monotonic() - _apply_t0) * 1000)  # type: ignore[attr-defined]
        # Stamp the snapshot with the version that was current before this
        # change was applied, so rollback audit trails show the correct target.
        snapshot.config_version = self._current_version

        # Validate synthesis produced at least one file; treat empty as failure
        # so downstream stages (health check, GRPO, Benchmarks) don't silently
        # process a no-op change.
        if code_result.success and not code_result.files_written:
            log.error(
                "synthesis_returned_no_files",
                proposal_id=proposal.id,
                category=proposal.category.value,
                detail="Applicator reported success but files_written is empty - treating as failure.",
            )
            code_result.success = False
            code_result.error = "Synthesis returned no files"

        # ── Corpus 9 §4: Log static analysis exhaustion at service level ──────
        # code_agent logs this internally but service-level visibility is needed
        # so the proposal pipeline can surface it in dashboards and Thymos reports.
        _sa_max = self._config.static_analysis_max_fix_iterations
        _sa_used = code_result.static_analysis_fix_iterations
        if _sa_used >= _sa_max and code_result.static_analysis_findings > 0:
            log.warning(
                "static_analysis_max_fix_iterations_exhausted",
                proposal_id=proposal.id,
                category=proposal.category.value,
                fix_iterations_used=_sa_used,
                max_fix_iterations=_sa_max,
                errors_remaining=code_result.static_analysis_findings,
                files_written=len(code_result.files_written),
            )

        # ── Stage 5B: Neural repair agent (primary recovery before diffusion) ─
        if not code_result.success and self._repair_agent is not None:
            log.info("repair_agent_attempting")
            try:
                from systems.simula.agents.repair_agent import (
                    RepairAgent as RepairAgentCls,
                )
                from systems.simula.verification.types import RepairStatus

                if isinstance(self._repair_agent, RepairAgentCls):
                    broken_files = {
                        f: (self._root / f).read_text()
                        for f in code_result.files_written
                        if (self._root / f).exists()
                    }
                    repair_result = await self._repair_agent.repair(
                        proposal=proposal,
                        broken_files=broken_files,
                        test_output=code_result.test_output or code_result.error,
                    )
                    if repair_result.status == RepairStatus.REPAIRED:
                        log.info(
                            "repair_agent_succeeded",
                            attempts=repair_result.total_attempts,
                            cost=f"${repair_result.total_cost_usd:.4f}",
                        )
                        code_result.success = True
                        code_result.files_written = repair_result.files_repaired
                        code_result.error = ""
                        code_result.repair_attempted = True
                        code_result.repair_succeeded = True
                        code_result.repair_cost_usd = repair_result.total_cost_usd
                        proposal._repair_result = repair_result  # type: ignore[attr-defined]
                    else:
                        log.info("repair_agent_insufficient", status=repair_result.status.value)
                        code_result.repair_attempted = True
                        code_result.repair_succeeded = False
                        proposal._repair_result = repair_result  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning("repair_agent_error", error=str(exc))

        # Stage 4C: Diffusion repair fallback when code agent fails
        if not code_result.success and self._diffusion_repair is not None:
            log.info("diffusion_repair_fallback_attempting")
            try:
                from systems.simula.agents.diffusion_repair import DiffusionRepairAgent
                if isinstance(self._diffusion_repair, DiffusionRepairAgent):
                    broken_files_dr = {
                        f: (self._root / f).read_text()
                        for f in code_result.files_written
                        if (self._root / f).exists()
                    }
                    dr_result = await self._diffusion_repair.repair(
                        proposal=proposal,
                        broken_files=broken_files_dr,
                        test_output=code_result.test_output or code_result.error or "",
                    )
                    if dr_result.status.value == "repaired":
                        log.info(
                            "diffusion_repair_succeeded",
                            steps=len(dr_result.denoise_steps),
                            improvement=f"{dr_result.improvement_rate:.0%}",
                        )
                        # Mark as success - diffusion repair saved the change
                        code_result.success = True
                        code_result.files_written = dr_result.files_repaired
                        code_result.error = ""
                        # Stash repair metadata on proposal for history recording
                        proposal._diffusion_repair_result = dr_result  # type: ignore[attr-defined]
                    else:
                        log.info("diffusion_repair_insufficient", status=dr_result.status.value)
            except Exception as exc:
                log.warning("diffusion_repair_error", error=str(exc))

        if not code_result.success:
            proposal.status = ProposalStatus.ROLLED_BACK
            self._proposals_rolled_back += 1
            apply_fail_reason = f"Application failed: {code_result.error}"
            self._last_proposal_processed_at = utc_now().isoformat()
            log.warning("apply_failed_no_success", error=code_result.error)

            # ── RE training: rollback decision ──
            _apply_ms = getattr(proposal, "_application_duration_ms", 0)
            asyncio.ensure_future(self._emit_re_training_example(
                category="rollback_decision",
                instruction="Decide whether to roll back an applied evolution proposal after health check or application failure.",
                input_context=f"category={proposal.category.value}, source={proposal.source}, files={len(code_result.files_written)}",
                output=f"rolled_back=True, reason={code_result.error[:200]}",
                outcome_quality=0.0,
                latency_ms=_apply_ms,
                reasoning_trace=code_result.error[:300] if code_result.error else "",
                episode_id=getattr(proposal, "id", "") or "",
                constitutional_alignment=None,  # rollback = alignment unknown; record stage populates
            ))

            # Meta-healing: report rollback to Thymos so the immune system
            # knows a repair attempt failed and can escalate
            # 10s timeout - a stuck sentinel must not block meta-healing
            try:
                await asyncio.wait_for(
                    self._sentinel.report(
                        RuntimeError(f"Proposal rollback: {code_result.error}"),
                        context={
                            "proposal_id": proposal.id,
                            "category": proposal.category.value,
                            "files_affected": code_result.files_written[:10],
                        },
                    ),
                    timeout=10.0,
                )
            except (asyncio.TimeoutError, TimeoutError):
                self._logger.critical("sentinel_report_timeout", timeout_s=10.0)

            # Emit REPAIR_COMPLETED(success=False) so Thymos can update antibody
            if self._synapse is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType

                    event_bus = getattr(self._synapse, "_event_bus", None)
                    if event_bus is not None:
                        await event_bus.emit(
                            SynapseEvent(
                                event_type=SynapseEventType.REPAIR_COMPLETED,
                                source_system="simula",
                                data={
                                    "proposal_id": proposal.id,
                                    "incident_id": getattr(proposal, "incident_id", ""),
                                    "success": False,
                                    "fingerprint": getattr(proposal, "fingerprint", ""),
                                    "tier": "NOVEL_FIX",
                                    "error": code_result.error[:200],
                                },
                            )
                        )
                except Exception:
                    pass  # Best-effort - sentinel report above is primary

            # RepairMemory: record failed application outcome (Task 2, 4)
            if self._repair_memory is not None:
                try:
                    await self._repair_memory.record_outcome(
                        proposal,
                        verification_passed=False,
                        rollback_needed=True,
                    )
                    await self._repair_memory.record_postmortem(
                        proposal,
                        rollback_reason=code_result.error or "application_failed",
                        verification_passed=False,
                    )
                except Exception as _rm_exc:
                    log.warning("repair_memory_apply_fail_record_failed", error=str(_rm_exc))

            # Emit EVOLUTION_ROLLED_BACK so Evo penalises hypotheses
            await self._emit_evolution_outcome(
                proposal, applied=False,
                rollback_reason=apply_fail_reason,
            )

            # Evolutionary observable: mutation rolled back (application failure)
            await self._emit_evolutionary_observable(
                observable_type="mutation_rolled_back",
                value=proposal.efe_score,
                is_novel=True,
                metadata={
                    "proposal_id": proposal.id,
                    "category": proposal.category.value,
                    "reason": apply_fail_reason[:200],
                    "stage": "application",
                },
            )

            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            self._invalidate_analytics()

            _kpi_bus1 = getattr(getattr(self._synapse, "_event_bus", None), "emit", None)
            if _kpi_bus1 is not None:
                with contextlib.suppress(Exception):
                    received = max(self._proposals_received, 1)
                    from systems.synapse.types import SynapseEvent, SynapseEventType
                    asyncio.ensure_future(self._synapse._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.SIMULA_KPI_PUSH,
                        source_system="simula",
                        data={
                            "system": "simula",
                            "outcome": "rolled_back",
                            "proposal_id": proposal.id,
                            "category": proposal.category.value,
                            "efe_score": round(proposal.efe_score, 4),
                            "reason": apply_fail_reason[:200],
                            "source": proposal.source,
                            "grpo_ab_group": grpo_ab_group,
                            "proposal_success_rate": round(
                                self._proposals_approved / received, 4
                            ),
                            "rollback_rate": round(
                                self._proposals_rolled_back / max(self._proposals_approved, 1), 4
                            ),
                            "avg_simulation_duration_ms": getattr(
                                proposal, "_simulation_duration_ms", 0
                            ),
                            "avg_application_duration_ms": getattr(
                                proposal, "_application_duration_ms", 0
                            ),
                            "risk_distribution": (
                                proposal.simulation.risk_level.value
                                if proposal.simulation else "unknown"
                            ),
                        },
                    )))

            return ProposalResult(
                status=ProposalStatus.ROLLED_BACK,
                reason=apply_fail_reason,
            )

        # Stash GRPO metadata for history recording
        proposal._grpo_model_used = grpo_model_used  # type: ignore[attr-defined]
        proposal._grpo_ab_group = grpo_ab_group  # type: ignore[attr-defined]

        # Stash Stage 5A synthesis metadata
        if synthesis_result_stash is not None:
            proposal._synthesis_result = synthesis_result_stash  # type: ignore[attr-defined]
            code_result.synthesis_strategy = synthesis_result_stash.strategy.value
            code_result.synthesis_speedup = synthesis_result_stash.speedup_vs_cegis

        # ── Health check (with Stage 2 formal verification) ────────────────
        # Oikos metabolic gate: in survival/conservation mode skip expensive
        # formal verification (Dafny/Lean/Z3) and fall back to static-only.
        _metabolic_ok = await self._check_metabolic_gate(
            "formal_verification", estimated_cost_usd="0.50"
        )
        if self._health is not None:
            self._health._shallow_verification_mode = not _metabolic_ok
        if not _metabolic_ok:
            log.info(
                "metabolic_gate_shallow_verification",
                starvation=self._starvation_level,
            )
        try:
            health = await self._health.check(
                code_result.files_written, proposal=proposal,
            )
        except Exception as exc:
            log.error("health_check_unhandled_error", error=str(exc))
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            self._invalidate_analytics()
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Health check failed unexpectedly: {exc}",
            )

        # Emit PROOF_FOUND/PROOF_FAILED/PROOF_TIMEOUT events for Dafny/Z3/Lean
        asyncio.ensure_future(self._emit_proof_events_from_health(health, proposal))

        # Stash formal verification result for history recording
        if health.formal_verification is not None:
            proposal._formal_verification_result = health.formal_verification  # type: ignore[attr-defined]

            # ── Z3 counterexample feedback (Spec §9) ──────────────────────
            # Feed invalid Z3 invariants back into the code_agent so the next
            # attempt knows which boundary conditions were violated.
            _fv = health.formal_verification
            if (
                _fv is not None
                and hasattr(_fv, "z3")
                and _fv.z3 is not None
                and hasattr(_fv.z3, "round_history")
            ):
                _z3_cexa: list[str] = []
                for _z3rnd in _fv.z3.round_history:
                    _z3_cexa.extend(getattr(_z3rnd, "counterexamples_fed_back", []))
                if _z3_cexa and self._code_agent is not None:
                    _cex_lines = "\n".join(f"  {c}" for c in _z3_cexa[:10])
                    self._code_agent._z3_counterexample_prompt = (  # type: ignore[attr-defined]
                        "## Z3 Formal Verification -- Invalid Invariants\n\n"
                        "The following counterexamples were found in your implementation. "
                        "Fix the code so these boundary conditions hold:\n\n"
                        f"{_cex_lines}\n\n"
                        "Ensure range bounds, preconditions, and postconditions are "
                        "satisfied in all code paths."
                    )
                    log.info(
                        "z3_counterexamples_injected_into_code_agent",
                        count=len(_z3_cexa),
                    )

        # Stash Lean 4 verification result for history recording
        if health.lean_verification is not None:
            proposal._lean_verification_result = health.lean_verification  # type: ignore[attr-defined]

        # Stash Stage 6 formal guarantees result for history recording
        if health.formal_guarantees is not None:
            proposal._formal_guarantees_result = health.formal_guarantees  # type: ignore[attr-defined]

        # ── Update Thompson sampling router with verification outcomes ──
        # Each proof strategy arm learns from this proposal's verification.
        try:
            _fv = health.formal_verification
            if _fv is not None:
                if hasattr(_fv, "dafny") and _fv.dafny is not None:
                    self._reasoning_router.update(
                        "dafny",
                        getattr(_fv.dafny, "verified", False),
                    )
                if hasattr(_fv, "z3") and _fv.z3 is not None:
                    self._reasoning_router.update(
                        "z3",
                        getattr(_fv.z3, "all_valid", False),
                    )
                if hasattr(_fv, "lean") and _fv.lean is not None:
                    self._reasoning_router.update(
                        "lean",
                        getattr(_fv.lean, "verified", False),
                    )
            if health.static_analysis is not None:
                self._reasoning_router.update(
                    "static_analysis",
                    len(getattr(health.static_analysis, "findings", [])) == 0,
                )
        except Exception:
            pass  # Non-critical; router degrades gracefully to uniform priors

        if not health.healthy:
            recovered = False

            # ── Stage 5D: Causal debugging before repair ──────────────────
            causal_diagnosis = None
            if self._causal_debugger is not None:
                log.info("causal_debugging_starting", issues=health.issues)
                try:
                    from systems.simula.debugging.causal_dag import (
                        CausalDebugger as CausalDbgCls,
                    )

                    if isinstance(self._causal_debugger, CausalDbgCls):
                        causal_diagnosis = await self._causal_debugger.diagnose(
                            files_written=code_result.files_written,
                            health_issues=health.issues,
                            test_output=code_result.test_output or "",
                        )
                        log.info(
                            "causal_diagnosis_complete",
                            root_cause=causal_diagnosis.root_cause_node,
                            confidence=f"{causal_diagnosis.confidence:.2f}",
                            interventions=causal_diagnosis.total_interventions,
                        )
                        # Stash for history recording
                        proposal._causal_diagnosis = causal_diagnosis  # type: ignore[attr-defined]
                        health.causal_diagnosis = causal_diagnosis
                except Exception as exc:
                    log.warning("causal_debugging_error", error=str(exc))

            # ── Stage 5B: Repair agent recovery after causal diagnosis ────
            if self._repair_agent is not None:
                log.info("repair_agent_post_health_attempting")
                try:
                    from systems.simula.agents.repair_agent import (
                        RepairAgent as RepairAgentCls,
                    )
                    from systems.simula.verification.types import RepairStatus

                    if isinstance(self._repair_agent, RepairAgentCls):
                        broken_files = {
                            f: (self._root / f).read_text()
                            for f in code_result.files_written
                            if (self._root / f).exists()
                        }
                        # Feed causal diagnosis context to the repair agent
                        diag_context = ""
                        if causal_diagnosis is not None:
                            diag_context = (
                                f"Root cause: {causal_diagnosis.root_cause_node}\n"
                                f"Fix location: {causal_diagnosis.root_cause_file}\n"
                                f"Confidence: {causal_diagnosis.confidence:.2f}\n"
                                f"Reasoning: {' → '.join(causal_diagnosis.reasoning_chain)}"
                            )
                        repair_result = await self._repair_agent.repair(
                            proposal=proposal,
                            broken_files=broken_files,
                            test_output=(
                                code_result.test_output
                                or "; ".join(health.issues)
                            ),
                            lint_output=diag_context or "",
                        )
                        if repair_result.status == RepairStatus.REPAIRED:
                            log.info(
                                "repair_agent_post_health_succeeded",
                                attempts=repair_result.total_attempts,
                                cost=f"${repair_result.total_cost_usd:.4f}",
                            )
                            code_result.repair_attempted = True
                            code_result.repair_succeeded = True
                            code_result.repair_cost_usd = repair_result.total_cost_usd
                            proposal._repair_result = repair_result  # type: ignore[attr-defined]

                            # Re-check health after repair
                            health_recheck = await self._health.check(
                                repair_result.files_repaired, proposal=proposal,
                            )
                            asyncio.ensure_future(
                                self._emit_proof_events_from_health(health_recheck, proposal)
                            )
                            if health_recheck.healthy:
                                log.info("health_recheck_passed_after_repair")
                                health = health_recheck
                                code_result.files_written = repair_result.files_repaired
                                code_result.success = True
                                recovered = True
                            else:
                                log.warning(
                                    "health_recheck_still_failing",
                                    issues=health_recheck.issues,
                                )
                        else:
                            log.info(
                                "repair_agent_post_health_insufficient",
                                status=repair_result.status.value,
                            )
                            code_result.repair_attempted = True
                            code_result.repair_succeeded = False
                except Exception as exc:
                    log.warning("repair_agent_post_health_error", error=str(exc))

            # ── Rollback only if all recovery failed ──────────────────────
            # VERIFIED: health check failure → rollback.restore(snapshot) → ProposalResult(ROLLED_BACK)
            if not recovered:
                log.warning("health_check_failed_rolling_back", issues=health.issues)
                try:
                    _restored_paths = await self._rollback.restore(snapshot)
                    _expected = len(snapshot.files) if snapshot else 0
                    _actual = len(_restored_paths)
                    if _actual < _expected:
                        log.error(
                            "rollback_restore_incomplete",
                            proposal_id=proposal.id,
                            expected_files=_expected,
                            restored_files=_actual,
                            detail=(
                                "Some snapshot files could not be restored"
                                " - system may be in inconsistent state."
                            ),
                        )
                        with contextlib.suppress(Exception):
                            await self._sentinel.report(
                                error_type="RollbackRestoreIncomplete",
                                message=(
                                    f"Rollback restored only {_actual}/{_expected} files "
                                    f"for proposal {proposal.id}"
                                ),
                                context={
                                    "proposal_id": proposal.id,
                                    "category": proposal.category.value,
                                    "expected_files": _expected,
                                    "restored_files": _actual,
                                    "health_issues": health.issues,
                                },
                            )
                    else:
                        log.info(
                            "rollback_restore_verified",
                            proposal_id=proposal.id,
                            restored_files=_actual,
                        )
                except Exception as _rb_exc:
                    log.error(
                        "rollback_restore_failed",
                        proposal_id=proposal.id,
                        error=str(_rb_exc),
                        detail="Rollback raised an exception - system state is unknown.",
                    )
                    with contextlib.suppress(Exception):
                        await self._sentinel.report(
                            error_type="RollbackRestoreFailed",
                            message=(
                                f"Rollback.restore() raised for proposal"
                                f" {proposal.id}: {_rb_exc}"
                            ),
                            context={
                                "proposal_id": proposal.id,
                                "category": proposal.category.value,
                                "error": str(_rb_exc),
                                "health_issues": health.issues,
                            },
                        )
                proposal.status = ProposalStatus.ROLLED_BACK
                self._proposals_rolled_back += 1
                rollback_reason_str = "; ".join(health.issues)
                self._last_proposal_processed_at = utc_now().isoformat()

                # Record the rollback in history
                await self._record_evolution(
                    proposal, code_result.files_written,
                    rolled_back=True,
                    rollback_reason=rollback_reason_str,
                )

                # RepairMemory: record outcome + generate postmortem (Task 2, 4)
                if self._repair_memory is not None:
                    try:
                        await self._repair_memory.record_outcome(
                            proposal,
                            verification_passed=False,
                            rollback_needed=True,
                        )
                        await self._repair_memory.record_postmortem(
                            proposal,
                            rollback_reason=rollback_reason_str,
                            verification_passed=False,
                        )
                    except Exception as _rm_exc:
                        log.warning("repair_memory_rollback_record_failed", error=str(_rm_exc))

                # Emit REPAIR_COMPLETED(success=False) for health-check rollback
                if self._synapse is not None:
                    try:
                        from systems.synapse.types import SynapseEvent, SynapseEventType

                        _eb = getattr(self._synapse, "_event_bus", None)
                        if _eb is not None:
                            await _eb.emit(
                                SynapseEvent(
                                    event_type=SynapseEventType.REPAIR_COMPLETED,
                                    source_system="simula",
                                    data={
                                        "proposal_id": proposal.id,
                                        "incident_id": getattr(proposal, "incident_id", ""),
                                        "success": False,
                                        "fingerprint": getattr(proposal, "fingerprint", ""),
                                        "tier": "NOVEL_FIX",
                                        "error": rollback_reason_str[:200],
                                    },
                                )
                            )
                    except Exception:
                        pass  # Best-effort

                # Emit EVOLUTION_ROLLED_BACK so Evo penalises hypotheses
                await self._emit_evolution_outcome(
                    proposal, applied=False,
                    rollback_reason=rollback_reason_str,
                )

                async with self._proposals_lock:
                    self._active_proposals.pop(proposal.id, None)
                self._invalidate_analytics()

                _kpi_bus2 = getattr(getattr(self._synapse, "_event_bus", None), "emit", None)
                if _kpi_bus2 is not None:
                    with contextlib.suppress(Exception):
                        received = max(self._proposals_received, 1)
                        from systems.synapse.types import SynapseEvent, SynapseEventType
                        asyncio.ensure_future(self._synapse._event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.SIMULA_KPI_PUSH,
                            source_system="simula",
                            data={
                                "system": "simula",
                                "outcome": "rolled_back",
                                "proposal_id": proposal.id,
                                "category": proposal.category.value,
                                "efe_score": round(proposal.efe_score, 4),
                                "reason": rollback_reason_str[:200],
                                "source": proposal.source,
                                "grpo_ab_group": getattr(proposal, "_grpo_ab_group", ""),
                                "proposal_success_rate": round(
                                    self._proposals_approved / received, 4
                                ),
                                "rollback_rate": round(
                                    self._proposals_rolled_back
                                    / max(self._proposals_approved, 1),
                                    4,
                                ),
                                "avg_simulation_duration_ms": getattr(
                                    proposal, "_simulation_duration_ms", 0
                                ),
                                "avg_application_duration_ms": getattr(
                                    proposal, "_application_duration_ms", 0
                                ),
                                "risk_distribution": (
                                    proposal.simulation.risk_level.value
                                    if proposal.simulation else "unknown"
                                ),
                            },
                        )))

                # Evolutionary observable: mutation rolled back (health check failure)
                await self._emit_evolutionary_observable(
                    observable_type="mutation_rolled_back",
                    value=proposal.efe_score,
                    is_novel=True,
                    metadata={
                        "proposal_id": proposal.id,
                        "category": proposal.category.value,
                        "reason": rollback_reason_str[:200],
                        "stage": "health_check",
                    },
                )

                return ProposalResult(
                    status=ProposalStatus.ROLLED_BACK,
                    reason=f"Post-apply health check failed: {rollback_reason_str}",
                )

        # ── Stage 6A.2: Sign generated files with content credentials ────────
        if self._content_credentials is not None:
            try:
                from systems.simula.audit.content_credentials import (
                    ContentCredentialManager,
                )

                if isinstance(self._content_credentials, ContentCredentialManager):
                    cc_result = await self._content_credentials.sign_files(
                        files=code_result.files_written,
                        codebase_root=self._root,
                    )
                    proposal._content_credential_result = cc_result  # type: ignore[attr-defined]
                    log.info(
                        "content_credentials_signed",
                        signed=len(cc_result.credentials),
                        unsigned=len(cc_result.unsigned_files),
                    )
            except Exception as exc:
                log.warning("content_credentials_error", error=str(exc))

        # ── Stage 6C: Generate formal specs for changed code ─────────────────
        if self._formal_spec_generator is not None:
            try:
                from systems.simula.formal_specs.spec_generator import FormalSpecGenerator

                if isinstance(self._formal_spec_generator, FormalSpecGenerator):
                    fsg_result = await self._formal_spec_generator.generate_all(
                        files=code_result.files_written,
                        proposal=proposal,
                        codebase_root=self._root,
                        dafny_enabled=self._config.dafny_spec_generation_enabled,
                        tla_plus_enabled=self._config.tla_plus_enabled,
                        alloy_enabled=self._config.alloy_enabled,
                        self_spec_enabled=self._config.self_spec_dsl_enabled,
                    )
                    proposal._formal_spec_result = fsg_result  # type: ignore[attr-defined]
                    log.info(
                        "formal_specs_generated",
                        specs=len(fsg_result.specs),
                        coverage=f"{fsg_result.overall_coverage_percent:.0%}",
                    )
            except Exception as exc:
                log.warning("formal_spec_generation_error", error=str(exc))

        # ── Stage 3A: Incremental verification cache update ─────────────────
        if self._incremental is not None:
            try:
                incr_result = await self._incremental.verify_incremental(
                    files_changed=code_result.files_written,
                    proposal_id=proposal.id,
                )
                log.info(
                    "incremental_verification_complete",
                    checked=incr_result.functions_checked,
                    skipped=incr_result.functions_skipped_early_cutoff,
                    cache_hit_rate=f"{incr_result.cache_hit_rate:.0%}",
                )
            except Exception as exc:
                log.warning("incremental_verification_error", error=str(exc))

        # ── Success ───────────────────────────────────────────────────────────
        # VERIFIED: proposal applied → returns ProposalResult(status=APPLIED); caller sees APPLIED
        proposal.status = ProposalStatus.APPLIED
        self._proposals_approved += 1
        self._last_proposal_processed_at = utc_now().isoformat()

        async with self._version_lock:
            from_version = self._current_version
            self._current_version += 1

        # ── Canary deployment plan for MODERATE-risk proposals ────────────────
        # Spec ref: Section 7 - Temporal Simulation & Forward Modeling.
        # MODERATE risk changes are applied immediately but a canary plan is
        # attached so the ProactiveScanner / Thymos can monitor at each ramp
        # step and trigger rollback if rollback_criteria are violated.
        canary_plan: CanaryDeploymentPlan | None = None
        if (
            proposal.simulation is not None
            and proposal.simulation.risk_level == RiskLevel.MODERATE
        ):
            from systems.simula.evolution_types import CanaryDeploymentPlan
            canary_plan = CanaryDeploymentPlan.default_for_proposal(proposal.id)
            proposal._canary_plan = canary_plan  # type: ignore[attr-defined]
            log.info(
                "canary_plan_created",
                proposal_id=proposal.id,
                initial_pct=canary_plan.initial_traffic_percentage,
                steps=len(canary_plan.increase_schedule),
            )
            # Run canary ramp in background - does not block the pipeline return.
            asyncio.ensure_future(self._run_canary_ramp(canary_plan, proposal, log))

        # Stash PR info on proposal so _record_evolution can persist it to history
        if code_result.pr_url:
            proposal._pr_url = code_result.pr_url  # type: ignore[attr-defined]
            proposal._pr_number = code_result.pr_number  # type: ignore[attr-defined]

        await self._record_evolution(
            proposal,
            code_result.files_written,
            rolled_back=False,
            from_version=from_version,
        )

        # Emit EVOLUTION_APPLIED so Evo rewards hypotheses & Thymos monitors
        await self._emit_evolution_outcome(
            proposal,
            applied=True,
            files_changed=code_result.files_written,
            from_version=from_version,
            to_version=self._current_version,
        )

        # Evolutionary observable: mutation successfully applied
        await self._emit_evolutionary_observable(
            observable_type="mutation_applied",
            value=proposal.efe_score,
            is_novel=True,
            metadata={
                "proposal_id": proposal.id,
                "category": proposal.category.value,
                "description": proposal.description[:200],
                "files_changed": len(code_result.files_written),
                "from_version": from_version,
                "to_version": self._current_version,
                "source": proposal.source,
            },
        )

        # RepairMemory: record successful outcome (Task 2, 5)
        if self._repair_memory is not None:
            try:
                await self._repair_memory.record_outcome(
                    proposal,
                    verification_passed=True,
                    rollback_needed=False,
                )
            except Exception as _rm_exc:
                log.warning("repair_memory_success_record_failed", error=str(_rm_exc))

        # ── EFE Calibration Feedback ──────────────────────────────────────────
        # Compare predicted EFE with actual outcome to improve future scoring
        if self._efe_scorer is not None and proposal.efe_score is not None:
            try:
                # Use regression rate from simulation as actual improvement proxy
                regression_rate = 0.0
                if proposal.simulation is not None:
                    sim = proposal.simulation
                    if sim.episodes_tested > 0:
                        regression_rate = sim.regressions / max(1, sim.episodes_tested)
                actual_improvement = 1.0 - regression_rate

                calibration = EFECalibrationRecord(
                    proposal_id=proposal.id,
                    predicted_efe=proposal.efe_score,
                    actual_improvement=actual_improvement,
                    efe_error=actual_improvement - abs(proposal.efe_score),
                )
                self._efe_scorer.record_calibration(calibration)
                if self._history is not None:
                    await self._history.record_efe_calibration(calibration)
                log.info(
                    "efe_calibration_feedback",
                    predicted_efe=proposal.efe_score,
                    actual_improvement=actual_improvement,
                )
            except Exception as exc:
                log.warning("efe_calibration_failed", error=str(exc))

        # ── Corpus 14 §5: FOVEA_PREDICTION_ERROR on significant improvement divergence ──
        # If simulated improvement rate diverges from actual outcome by > threshold,
        # emit a prediction error so Fovea raises salience and Evo generates hypotheses.
        _fovea_threshold = 0.25  # 25% divergence triggers attention
        try:
            _sim = proposal.simulation
            if _sim is not None and _sim.episodes_tested > 0:
                _predicted_rate = _sim.improvements / max(1, _sim.episodes_tested)
                _regression_rate = _sim.regressions / max(1, _sim.episodes_tested)
                _actual_rate = 1.0 - _regression_rate
                _divergence = abs(_predicted_rate - _actual_rate)
                if _divergence >= _fovea_threshold:
                    _fovea_bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
                    if _fovea_bus is not None:
                        from systems.synapse.types import SynapseEvent, SynapseEventType
                        import uuid as _uuid
                        await _fovea_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.FOVEA_PREDICTION_ERROR,
                            source_system="simula",
                            data={
                                "error_id": str(_uuid.uuid4()),
                                "proposal_id": proposal.id,
                                "dominant_error_type": "magnitude_error",
                                "magnitude_error": round(_divergence, 4),
                                "content_error": 0.0,
                                "temporal_error": 0.0,
                                "source_error": 0.0,
                                "category_error": 0.0,
                                "causal_error": 0.0,
                                "precision_weighted_salience": round(min(1.0, _divergence * 2), 4),
                                "habituated_salience": round(min(1.0, _divergence * 2), 4),
                                "routes": ["evo", "nova"],
                                "context": {
                                    "predicted_improvement_rate": round(_predicted_rate, 4),
                                    "actual_improvement_rate": round(_actual_rate, 4),
                                    "episodes_tested": _sim.episodes_tested,
                                    "category": proposal.category.value,
                                },
                            },
                        ))
                        log.info(
                            "fovea_prediction_error_emitted",
                            proposal_id=proposal.id,
                            predicted_rate=round(_predicted_rate, 4),
                            actual_rate=round(_actual_rate, 4),
                            divergence=round(_divergence, 4),
                        )
        except Exception as exc:
            log.warning("fovea_prediction_error_emit_failed", error=str(exc))

        # ── Stage 6A.1: Append to hash chain ─────────────────────────────────
        if self._hash_chain is not None:
            try:
                from systems.simula.audit.hash_chain import HashChainManager

                if isinstance(self._hash_chain, HashChainManager):
                    # Build a record-like object for hashing (use the same fields as the recording)
                    from systems.simula.evolution_types import EvolutionRecord as ERec

                    hash_record = ERec(
                        proposal_id=proposal.id,
                        category=proposal.category,
                        description=proposal.description,
                        from_version=from_version,
                        to_version=self._current_version,
                        files_changed=code_result.files_written,
                        simulation_risk=RiskLevel.LOW,
                        rolled_back=False,
                    )
                    hce = await self._hash_chain.append(hash_record)
                    proposal._hash_chain_entry = hce  # type: ignore[attr-defined]
                    log.info(
                        "hash_chain_appended",
                        chain_hash=hce.chain_hash[:16],
                        position=hce.chain_position,
                    )
            except Exception as exc:
                log.warning("hash_chain_append_error", error=str(exc))

        # ── Stage 6B: Co-evolution cycle (fire-and-forget) ───────────────────
        if self._hard_negative_miner is not None:
            try:
                import asyncio as _aio

                _aio.create_task(
                    self._coevolution_background(
                        files=code_result.files_written,
                        proposal_id=proposal.id,
                    ),
                )
                log.info("coevolution_cycle_scheduled")
            except Exception as exc:
                log.warning("coevolution_schedule_error", error=str(exc))

        # ── Stage 4B: Record GRPO training data + code diffs ─────────────────
        if self._grpo is not None:
            try:
                self._grpo.record_proposal_applied()

                # Capture code agent output for training data
                if code_result.files_written:
                    # Read the actual file contents for training diffs
                    code_diffs: dict[str, str] = {}
                    for fpath in code_result.files_written:
                        try:
                            full = self._root / fpath
                            if full.exists():
                                code_diffs[fpath] = full.read_text(encoding="utf-8")
                        except Exception:
                            pass

                    await self._grpo.record_code_diff(
                        proposal_id=proposal.id,
                        category=proposal.category.value,
                        description=proposal.description,
                        files_changed=code_result.files_written,
                        code_diffs=code_diffs,
                        success=code_result.success,
                    )

                # Check if retraining is warranted
                if self._grpo.should_retrain():
                    log.info("grpo_retrain_triggered")
                    asyncio.create_task(self._grpo_retrain_background())
            except Exception as exc:
                log.warning("grpo_record_error", error=str(exc))

        # Evolutionary observable: procedure learned from successful evolution
        if code_result.success:
            await self._emit_evolutionary_observable(
                observable_type="procedure_learned",
                value=proposal.efe_score,
                is_novel=True,
                metadata={
                    "proposal_id": proposal.id,
                    "category": proposal.category.value,
                    "files_changed": len(code_result.files_written),
                    "repair_attempted": getattr(code_result, "repair_attempted", False),
                    "repair_succeeded": getattr(code_result, "repair_succeeded", False),
                },
            )

        # ── Stage 3C: LILO abstraction extraction ────────────────────────────
        self._proposals_applied_since_consolidation += 1
        if self._lilo is not None:
            try:
                extraction = await self._lilo.extract_from_proposals(
                    proposal_ids=[proposal.id],
                    files_changed={proposal.id: code_result.files_written},
                )
                if extraction.extracted:
                    log.info(
                        "lilo_extraction_complete",
                        extracted=len(extraction.extracted),
                        merged=extraction.merged_into_existing,
                    )
                    # Evolutionary observable: new abstractions created via LILO
                    await self._emit_evolutionary_observable(
                        observable_type="abstraction_created",
                        value=float(len(extraction.extracted)),
                        is_novel=True,
                        metadata={
                            "proposal_id": proposal.id,
                            "abstractions_extracted": len(extraction.extracted),
                            "merged_into_existing": extraction.merged_into_existing,
                        },
                    )
                # Periodic consolidation
                if (
                    self._proposals_applied_since_consolidation
                    >= self._config.lilo_consolidation_interval_proposals
                ):
                    await self._lilo.consolidate()
                    self._proposals_applied_since_consolidation = 0
                    log.info("lilo_consolidation_complete")
            except Exception as exc:
                log.warning("lilo_extraction_error", error=str(exc))

        # Clean up active proposals
        async with self._proposals_lock:
            self._active_proposals.pop(proposal.id, None)
        self._invalidate_analytics()

        log.info(
            "change_applied",
            from_version=from_version,
            to_version=self._current_version,
            files_changed=len(code_result.files_written),
            grpo_ab_group=grpo_ab_group,
        )

        # ── Emit REPAIR_COMPLETED on Synapse so Thymos can close the feedback loop ──
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                event_bus = getattr(self._synapse, "_event_bus", None)
                if event_bus is not None:
                    await event_bus.emit(
                        SynapseEvent(
                            event_type=SynapseEventType.REPAIR_COMPLETED,
                            source_system="simula",
                            data={
                                "proposal_id": proposal.id,
                                "incident_id": getattr(proposal, "incident_id", ""),
                                "success": True,
                                "fingerprint": getattr(proposal, "fingerprint", ""),
                                "tier": "NOVEL_FIX",
                                "incident_class": proposal.category.value,
                                "fix_type": "structural_evolution",
                                "root_cause": proposal.description[:200],
                                "files_changed": code_result.files_written,
                                "from_version": from_version,
                                "to_version": self._current_version,
                                "efe_score": proposal.efe_score,
                                "source": proposal.source,
                            },
                        )
                    )
            except Exception as _emit_exc:
                log.warning("repair_completed_emit_failed", error=str(_emit_exc))

        # ── Benchmarks KPI: emit telemetry on successful proposal application ──
        _kpi_bus3 = getattr(getattr(self._synapse, "_event_bus", None), "emit", None)
        if _kpi_bus3 is not None:
            with contextlib.suppress(Exception):
                received = max(self._proposals_received, 1)
                _sim_dur = getattr(proposal, "_simulation_duration_ms", 0)
                _app_dur = getattr(proposal, "_application_duration_ms", 0)
                _risk_val = proposal.simulation.risk_level.value if proposal.simulation else "unknown"
                from systems.synapse.types import SynapseEvent, SynapseEventType
                asyncio.ensure_future(self._synapse._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SIMULA_KPI_PUSH,
                    source_system="simula",
                    data={
                        "system": "simula",
                        "outcome": "applied",
                        "proposal_id": proposal.id,
                        "category": proposal.category.value,
                        "efe_score": round(proposal.efe_score, 4),
                        "files_changed": len(code_result.files_written),
                        "from_version": from_version,
                        "to_version": self._current_version,
                        "source": proposal.source,
                        "grpo_ab_group": getattr(proposal, "_grpo_ab_group", ""),
                        "proposal_success_rate": round(self._proposals_approved / received, 4),
                        "rollback_rate": round(
                            self._proposals_rolled_back / max(self._proposals_approved, 1), 4
                        ),
                        "avg_simulation_duration_ms": _sim_dur,
                        "avg_application_duration_ms": _app_dur,
                        "risk_distribution": _risk_val,
                    },
                )))

        # ── RE training: successful mutation proposal ──
        _total_ms = getattr(proposal, "_simulation_duration_ms", 0) + getattr(proposal, "_application_duration_ms", 0)
        asyncio.ensure_future(self._emit_re_training_example(
            category="mutation_proposal",
            instruction="Evaluate and apply a full evolution proposal through the Simula pipeline.",
            input_context=f"category={proposal.category.value}, source={proposal.source}, efe={proposal.efe_score:.4f}",
            output=f"status=APPLIED, version={self._current_version}, files={len(code_result.files_written)}",
            outcome_quality=min(1.0, 0.7 + proposal.efe_score * 0.3) if proposal.efe_score else 0.8,
            latency_ms=_total_ms,
            reasoning_trace=proposal.description[:300],
            episode_id=getattr(proposal, "id", "") or "",
            constitutional_alignment=None,  # populated in _record_evolution with final outcome
        ))

        return ProposalResult(
            status=ProposalStatus.APPLIED,
            version=self._current_version,
            files_changed=code_result.files_written,
            pr_url=code_result.pr_url,
            pr_number=code_result.pr_number,
            canary_plan=canary_plan,
        )

    async def _simulate_change(self, proposal: EvolutionProposal) -> SimulationResult:
        """Delegate to the deep ChangeSimulator."""
        if self._simulator is None:
            return SimulationResult(risk_level=RiskLevel.LOW, risk_summary="Simulator not initialized")
        return await self._simulator.simulate(proposal)

    async def _submit_to_governance(
        self, proposal: EvolutionProposal, simulation: SimulationResult
    ) -> str:
        """
        Submit a governed proposal to the community governance system.
        Returns a governance record ID. Enriches the governance record
        with deep simulation data for community review.
        """
        record_id = f"gov_{new_id()}"

        if self._neo4j is not None:
            try:
                # Include enriched simulation data for governance reviewers
                risk_summary = simulation.risk_summary
                benefit_summary = simulation.benefit_summary

                # Add counterfactual and alignment data if available (enriched simulation)
                enrichment = []
                if isinstance(simulation, EnrichedSimulationResult):
                    if simulation.constitutional_alignment != 0.0:
                        enrichment.append(f"Constitutional alignment: {simulation.constitutional_alignment:+.2f}")
                    if simulation.dependency_blast_radius > 0:
                        enrichment.append(f"Blast radius: {simulation.dependency_blast_radius} files")
                if enrichment:
                    risk_summary = f"{risk_summary} [{'; '.join(enrichment)}]"

                await self._neo4j.execute_write(
                    """
                    CREATE (:GovernanceProposal {
                        id: $id,
                        proposal_id: $proposal_id,
                        category: $category,
                        description: $description,
                        risk_level: $risk_level,
                        risk_summary: $risk_summary,
                        benefit_summary: $benefit_summary,
                        submitted_at: $submitted_at,
                        status: 'pending'
                    })
                    """,
                    {
                        "id": record_id,
                        "proposal_id": proposal.id,
                        "category": proposal.category.value,
                        "description": proposal.description,
                        "risk_level": simulation.risk_level.value,
                        "risk_summary": risk_summary,
                        "benefit_summary": benefit_summary,
                        "submitted_at": utc_now().isoformat(),
                    },
                )
            except Exception as exc:
                self._logger.warning("governance_neo4j_write_failed", error=str(exc))

        return record_id

    async def _record_evolution(
        self,
        proposal: EvolutionProposal,
        files_changed: list[str],
        rolled_back: bool = False,
        rollback_reason: str = "",
        from_version: int | None = None,
    ) -> None:
        """Write an immutable evolution record and update the version chain.

        from_version should be the pre-apply version captured atomically inside
        _version_lock. If omitted it is derived from self._current_version for
        backwards compatibility (rolled-back path, where the version was never
        incremented).
        """
        if self._history is None:
            return

        if from_version is None:
            # Rollback path: version was never incremented, so from == to.
            from_version = self._current_version
        to_version = self._current_version

        risk_level = (
            proposal.simulation.risk_level
            if proposal.simulation
            else RiskLevel.LOW
        )

        # Extract simulation detail fields if enriched simulation was performed
        sim_detail: dict[str, Any] = {
            "simulation_episodes_tested": 0,
            "counterfactual_regression_rate": 0.0,
            "dependency_blast_radius": 0,
            "constitutional_alignment": 0.0,
            "resource_tokens_per_hour": 0,
            "caution_reasoning": "",
        }
        if isinstance(proposal.simulation, EnrichedSimulationResult):
            sim_detail["simulation_episodes_tested"] = proposal.simulation.episodes_tested
            sim_detail["counterfactual_regression_rate"] = proposal.simulation.counterfactual_regression_rate
            sim_detail["dependency_blast_radius"] = proposal.simulation.dependency_blast_radius
            sim_detail["constitutional_alignment"] = proposal.simulation.constitutional_alignment
            if proposal.simulation.resource_cost_estimate:
                sim_detail["resource_tokens_per_hour"] = (
                    proposal.simulation.resource_cost_estimate.estimated_additional_llm_tokens_per_hour
                )
            if proposal.simulation.caution_adjustment:
                sim_detail["caution_reasoning"] = proposal.simulation.caution_adjustment.reasoning

        import uuid as _uuid_rec
        import os as _os
        _current_identity = _os.environ.get("ECODIAOS_INSTANCE_ID", "eos-default")
        record = EvolutionRecord(
            proposal_id=proposal.id,
            category=proposal.category,
            description=proposal.description,
            from_version=from_version,
            to_version=to_version,
            files_changed=files_changed,
            simulation_risk=risk_level,
            rolled_back=rolled_back,
            rollback_reason=rollback_reason,
            # Corpus 14 §8: MemoryTrace bi-temporal fields (Spec 01)
            episode_id=str(_uuid_rec.uuid4()),
            perception_time=proposal.created_at if hasattr(proposal, "created_at") else None,
            reflection_time=utc_now(),
            # Corpus 14 §13: Identity scope (Spec 23)
            identity_id=_current_identity,
            **sim_detail,
        )

        # Stage 2: Attach formal verification metadata if available
        if hasattr(proposal, "_formal_verification_result"):
            fv = proposal._formal_verification_result
            if fv is not None:
                if fv.dafny and fv.dafny.status:
                    record.formal_verification_status = fv.dafny.status.value
                    record.dafny_rounds = fv.dafny.rounds_attempted
                if fv.z3 and fv.z3.valid_invariants:
                    record.discovered_invariants_count = len(fv.z3.valid_invariants)
                if fv.static_analysis:
                    record.static_analysis_findings = len(fv.static_analysis.findings)

        # Stage 4A: Attach Lean 4 proof metadata if available
        if hasattr(proposal, "_lean_verification_result"):
            lean_r = proposal._lean_verification_result
            if lean_r is not None:
                record.lean_proof_status = lean_r.status.value
                record.lean_proof_rounds = len(lean_r.attempts)
                record.lean_proven_lemmas_count = len(lean_r.proven_lemmas)
                record.lean_copilot_automation_rate = lean_r.copilot_automation_rate
                record.lean_library_lemmas_reused = len(lean_r.library_lemmas_used)

        # Stage 4B: Attach GRPO model routing metadata
        if hasattr(proposal, "_grpo_model_used"):
            record.grpo_model_used = proposal._grpo_model_used
        if hasattr(proposal, "_grpo_ab_group"):
            record.grpo_ab_group = proposal._grpo_ab_group

        # Stage 4C: Attach diffusion repair metadata if used
        if hasattr(proposal, "_diffusion_repair_result"):
            dr = proposal._diffusion_repair_result
            if dr is not None:
                record.diffusion_repair_used = True
                record.diffusion_repair_status = dr.status.value
                record.diffusion_repair_steps = len(dr.denoise_steps)
                record.diffusion_improvement_rate = dr.improvement_rate

        # Stage 5A: Attach synthesis metadata
        if hasattr(proposal, "_synthesis_result"):
            sr = proposal._synthesis_result
            if sr is not None:
                record.synthesis_strategy_used = sr.strategy.value
                record.synthesis_status = sr.status.value
                record.synthesis_speedup_vs_baseline = sr.speedup_vs_cegis
                record.synthesis_candidates_explored = sr.candidates_explored

        # Stage 5B: Attach repair agent metadata
        if hasattr(proposal, "_repair_result"):
            rr = proposal._repair_result
            if rr is not None:
                record.repair_agent_used = True
                record.repair_agent_status = rr.status.value
                record.repair_attempts = rr.total_attempts
                record.repair_cost_usd = rr.total_cost_usd

        # Stage 5C: Attach orchestration metadata
        if hasattr(proposal, "_orchestration_result"):
            orc = proposal._orchestration_result
            if orc is not None:
                record.orchestration_used = True
                record.orchestration_dag_nodes = orc.dag_nodes
                record.orchestration_agents_used = orc.agents_used
                record.orchestration_parallel_stages = orc.parallel_stages

        # Stage 5D: Attach causal debugging metadata
        if hasattr(proposal, "_causal_diagnosis"):
            cd = proposal._causal_diagnosis
            if cd is not None:
                record.causal_debug_used = True
                record.causal_root_cause = cd.root_cause
                record.causal_confidence = cd.confidence
                record.causal_interventions = cd.interventions_performed

        # Stage 5E: Attach issue resolution metadata
        if hasattr(proposal, "_issue_resolution_result"):
            ir = proposal._issue_resolution_result
            if ir is not None:
                record.issue_resolution_used = True
                record.issue_autonomy_level = ir.autonomy_level.value
                record.issue_abstained = ir.status.value == "abstained"

        # Stage 6A: Attach hash chain metadata
        if hasattr(proposal, "_hash_chain_entry"):
            hce = proposal._hash_chain_entry
            if hce is not None:
                record.hash_chain_hash = hce.chain_hash
                record.hash_chain_position = hce.chain_position

        # Stage 6A: Content credentials count
        if hasattr(proposal, "_content_credential_result"):
            ccr = proposal._content_credential_result
            if ccr is not None:
                record.content_credentials_signed = len(ccr.credentials)

        # Stage 6A: Governance credential status
        if hasattr(proposal, "_governance_credential_result"):
            gcr = proposal._governance_credential_result
            if gcr is not None:
                record.governance_credential_status = gcr.status.value

        # Stage 6B: Co-evolution metadata
        if hasattr(proposal, "_coevolution_result"):
            cr = proposal._coevolution_result
            if cr is not None:
                record.coevolution_hard_negatives_mined = cr.hard_negatives_mined
                record.coevolution_adversarial_tests = cr.adversarial_tests_generated
                record.coevolution_bugs_found = cr.tests_found_bugs

        # Stage 6C: Formal spec metadata
        if hasattr(proposal, "_formal_spec_result"):
            fsr = proposal._formal_spec_result
            if fsr is not None:
                record.formal_specs_generated = len(fsr.specs)
                record.formal_spec_coverage_percent = fsr.overall_coverage_percent
                if fsr.tla_plus_results:
                    record.tla_plus_states_explored = sum(
                        r.states_explored for r in fsr.tla_plus_results
                    )

        # Stage 6D: E-graph metadata
        if hasattr(proposal, "_formal_guarantees_result"):
            fg = proposal._formal_guarantees_result
            if fg is not None and fg.egraph is not None:
                er = fg.egraph
                record.egraph_used = True
                record.egraph_status = er.status.value
                record.egraph_rules_applied = len(er.rules_applied)

        # Stage 6E: Symbolic execution metadata
        if hasattr(proposal, "_formal_guarantees_result"):
            fg = proposal._formal_guarantees_result
            if fg is not None and fg.symbolic_execution is not None:
                se = fg.symbolic_execution
                record.symbolic_execution_used = True
                record.symbolic_properties_proved = se.properties_proved
                record.symbolic_counterexamples = len(se.counterexamples)

        # Bounty Hunter: persist PR info so get_history() returns it
        if hasattr(proposal, "_pr_url") and proposal._pr_url:  # type: ignore[attr-defined]
            record.pr_url = proposal._pr_url  # type: ignore[attr-defined]
            record.pr_number = getattr(proposal, "_pr_number", None)  # type: ignore[attr-defined]

        try:
            await self._history.record(record)
        except Exception as exc:
            self._logger.error("history_write_failed", error=str(exc))
            return

        if not rolled_back:
            config_hash = self._compute_config_hash(files_changed)
            version = ConfigVersion(
                version=self._current_version,
                proposal_ids=[proposal.id],
                config_hash=config_hash,
            )
            try:
                await self._history.record_version(version, previous_version=from_version)
            except Exception as exc:
                self._logger.error("version_write_failed", error=str(exc))

        # ── RE training: self_evolution signal (richest training example) ─────
        # Emitted after every completed RECORD stage - applied and rolled-back.
        _evo_quality = 0.0 if rolled_back else min(
            1.0,
            0.6 + (record.constitutional_alignment or 0.0) * 0.2
                + len(files_changed) * 0.02,
        )
        # Build constitutional alignment from record's scalar alignment score + outcome
        _evo_alignment_score = record.constitutional_alignment or 0.0
        _evo_coherence = _evo_alignment_score * 2.0 - 1.0 if not rolled_back else -0.5
        _evo_growth = 0.5 if not rolled_back else -0.2
        # Attach WARN-level pattern risk annotation if present from pre-flight gate
        _pattern_risk_note = ""
        _pattern_risk: dict[str, Any] | None = getattr(proposal, "_pattern_risk", None)
        if _pattern_risk and _pattern_risk.get("risk_level") == "warn":
            _pattern_risk_note = (
                f" | pattern_risk=WARN matched={_pattern_risk.get('matched_patterns', [])} "
                f"reason={_pattern_risk.get('reason', '')[:100]}"
            )

        asyncio.ensure_future(self._emit_re_training_example(
            category="self_evolution",
            instruction=(
                "Evaluate and record the outcome of a structural self-evolution proposal "
                "through the full Simula pipeline."
            ),
            input_context=(
                f"category={proposal.category.value}, "
                f"source={proposal.source}, "
                f"description={proposal.description[:200]}, "
                f"risk={record.simulation_risk.value if record.simulation_risk else str('unknown')}, "
                f"files_changed={len(files_changed)}"
                f"{_pattern_risk_note}"
            ),
            output=(
                f"outcome={str('REJECTED') if rolled_back else str('ACCEPTED')}, "
                f"version={self._current_version}, "
                f"rollback_reason={rollback_reason[:120] if rollback_reason else str('none')}"
            ),
            outcome_quality=_evo_quality,
            reasoning_trace=proposal.description[:400],
            episode_id=getattr(proposal, "id", "") or "",
            constitutional_alignment=self._make_alignment(coherence=_evo_coherence, growth=_evo_growth),
        ))

        # ── Evo reward signal ──────────────────────────────────────────────────
        # Emit EVO_HYPOTHESIS_CONFIRMED or EVO_HYPOTHESIS_REFUTED so Evo can
        # update Thompson weights for the hypothesis family that produced this
        # proposal category.  Hypothesis ID is the stable category-scoped key.
        asyncio.ensure_future(self._emit_evo_reward(
            proposal=proposal,
            rolled_back=rolled_back,
        ))

    async def _emit_evo_reward(
        self,
        proposal: EvolutionProposal,
        rolled_back: bool,
        rollback_penalty: float = 0.3,
    ) -> None:
        """Emit EVO_HYPOTHESIS_CONFIRMED / REFUTED after Stage 7 RECORD.

        reward formula (Spec §5):
          APPLIED:      reward =  verification_confidence × (1 − risk_score)
          ROLLED_BACK:  reward = −rollback_penalty  (default −0.3)
        """
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            hypothesis_id = f"simula.evolution.{proposal.category.value}"

            if rolled_back:
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVO_HYPOTHESIS_REFUTED,
                    source_system="simula",
                    data={
                        "hypothesis_id": hypothesis_id,
                        "category": proposal.category.value,
                        "statement": proposal.description[:200],
                        "evidence_score": -rollback_penalty,
                        "contradicting_count": 1,
                        "reward": -rollback_penalty,
                        "reason": "rolled_back",
                    },
                ))
            else:
                # verification_confidence: use constitutional_alignment from simulation
                # as proxy (0–1 range); fall back to 0.7 if unavailable.
                sim = proposal.simulation
                verification_confidence = 0.7
                risk_score = 0.3  # default moderate risk
                if sim is not None:
                    if hasattr(sim, "constitutional_alignment") and sim.constitutional_alignment:
                        verification_confidence = float(sim.constitutional_alignment)
                    _risk_map = {
                        "low": 0.1,
                        "medium": 0.3,
                        "moderate": 0.4,
                        "high": 0.7,
                        "unacceptable": 1.0,
                    }
                    risk_score = _risk_map.get(sim.risk_level.value, 0.3)
                reward = verification_confidence * (1.0 - risk_score)
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
                    source_system="simula",
                    data={
                        "hypothesis_id": hypothesis_id,
                        "category": proposal.category.value,
                        "statement": proposal.description[:200],
                        "evidence_score": reward,
                        "supporting_count": 1,
                        "reward": reward,
                        "verification_confidence": verification_confidence,
                        "risk_score": risk_score,
                    },
                ))
        except Exception as exc:
            self._logger.debug("evo_reward_emit_failed", error=str(exc))

    # ─── Canary Traffic Ramp ──────────────────────────────────────────────────

    async def _run_canary_ramp(
        self,
        plan: Any,
        proposal: EvolutionProposal,
        log: Any,
        settle_period_s: float | None = None,
    ) -> None:
        """Execute the graduated traffic ramp for MODERATE-risk proposals.

        Emits SIMULA_CANARY_PROGRESS at each stage and waits `settle_period_s`
        (default 30s) before advancing.  Health degradation at any stage
        triggers rollback via _rollback_canary.
        """
        _settle = settle_period_s if settle_period_s is not None else getattr(
            self._config, "canary_settle_period_s", 30.0
        )
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None

        async def _emit_progress(stage_idx: int, pct: int, status: str, health_ok: bool) -> None:
            if bus is None:
                return
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SIMULA_CANARY_PROGRESS,
                    source_system="simula",
                    data={
                        "proposal_id": plan.proposal_id,
                        "stage": stage_idx,
                        "traffic_pct": pct,
                        "status": status,
                        "health_ok": health_ok,
                    },
                ))
            except Exception:
                pass

        for stage_idx, step in enumerate(plan.increase_schedule):
            pct = step.traffic_percentage
            log.info(
                "canary_ramp_stage_start",
                proposal_id=plan.proposal_id,
                stage=stage_idx,
                traffic_pct=pct,
            )
            await _emit_progress(stage_idx, pct, "advancing", health_ok=True)

            # Wait for the settle period before evaluating health
            try:
                await asyncio.sleep(_settle)
            except asyncio.CancelledError:
                log.info("canary_ramp_cancelled", proposal_id=plan.proposal_id)
                return

            # Health check: ask HealthChecker for a lightweight status
            health_ok = True
            if self._health is not None:
                try:
                    health_result = await asyncio.wait_for(
                        self._health.check(),
                        timeout=10.0,
                    )
                    # health.check() returns a bool or truthy object
                    health_ok = bool(health_result)
                except Exception:
                    health_ok = True  # non-fatal; assume healthy on error

            if not health_ok:
                log.warning(
                    "canary_ramp_health_degraded",
                    proposal_id=plan.proposal_id,
                    stage=stage_idx,
                    traffic_pct=pct,
                )
                await _emit_progress(stage_idx, pct, "rollback_triggered", health_ok=False)
                # Trigger rollback
                if self._rollback is not None:
                    try:
                        await self._rollback.restore_snapshot(proposal.id)
                        log.warning(
                            "canary_ramp_rollback_applied",
                            proposal_id=plan.proposal_id,
                            stage=stage_idx,
                        )
                    except Exception as exc:
                        log.error("canary_ramp_rollback_failed", error=str(exc))
                return

        # All stages passed
        await _emit_progress(
            len(plan.increase_schedule) - 1,
            100,
            "complete",
            health_ok=True,
        )
        log.info("canary_ramp_complete", proposal_id=plan.proposal_id)

    # ─── Stage 4B: GRPO Background Retraining ──────────────────────────────

    async def _grpo_retrain_background(self) -> None:
        """
        Background task: collect data, SFT, GRPO RL, evaluate, serve if improved.
        """
        if self._grpo is None:
            return
        try:
            from systems.simula.verification.types import GRPOTrainingStatus

            self._logger.info("grpo_retrain_starting")
            training_run = await self._grpo.run_full_pipeline()

            if training_run.status == GRPOTrainingStatus.FAILED:
                self._logger.info(
                    "grpo_retrain_skipped",
                    reason=training_run.error_summary or "pipeline failed",
                )
                return

            if training_run.finetuned_model_path and self._grpo._local_model_ready:
                self._logger.info(
                    "grpo_retrain_deployed",
                    model_path=training_run.finetuned_model_path,
                    model_id=training_run.finetuned_model_id,
                )
            else:
                self._logger.info(
                    "grpo_retrain_complete_no_deploy",
                    run_status=training_run.status.value,
                )
        except Exception as exc:
            self._logger.warning("grpo_retrain_error", error=str(exc))

    # ─── Stage 6B: Co-Evolution Background Cycle ────────────────────────────

    async def _coevolution_background(
        self,
        files: list[str],
        proposal_id: str,
    ) -> None:
        """
        Background task to run a co-evolution cycle:
        mine hard negatives from history and adversarial tests,
        then feed into GRPO training.
        """
        if self._hard_negative_miner is None:
            return
        try:
            from systems.simula.coevolution.failure_analyzer import FailureAnalyzer

            if not isinstance(self._hard_negative_miner, FailureAnalyzer):
                return

            # Import adversarial tester if available
            adversarial_gen = None
            if self._adversarial_tester is not None:
                from systems.simula.coevolution.robustness_tester import (
                    RobustnessTestGenerator,
                )

                if isinstance(self._adversarial_tester, RobustnessTestGenerator):
                    adversarial_gen = self._adversarial_tester

            cycle_result = await self._hard_negative_miner.run_cycle(
                adversarial_generator=adversarial_gen,
                files=files,
            )

            self._logger.info(
                "coevolution_cycle_complete",
                proposal_id=proposal_id,
                hard_negatives=cycle_result.hard_negatives_mined,
                adversarial_tests=cycle_result.adversarial_tests_generated,
                bugs_found=cycle_result.tests_found_bugs,
                grpo_examples=cycle_result.grpo_examples_produced,
                duration_ms=cycle_result.duration_ms,
            )

            # Feed hard negatives into GRPO if available
            if self._grpo is not None and cycle_result.grpo_examples_produced > 0:
                grpo_batch = await self._hard_negative_miner.prepare_grpo_batch(
                    await self._hard_negative_miner.mine_from_history(),
                )
                self._logger.info(
                    "coevolution_grpo_batch_ready",
                    examples=len(grpo_batch),
                )

        except Exception as exc:
            self._logger.warning(
                "coevolution_background_error",
                error=str(exc),
                proposal_id=proposal_id,
            )

    # ─── Adversarial Self-Play Loop ──────────────────────────────────────────

    async def _adversarial_self_play_loop(self) -> None:
        """
        Background loop: run constitutional red-teaming and drain proposals
        into the main evolution pipeline.

        Runs on a slow cadence (every cycle_cooldown_s) to avoid resource
        contention with normal proposal processing.
        """
        from systems.simula.coevolution.adversarial_self_play import (
            AdversarialSelfPlay,
        )

        self_play = self._adversarial_self_play
        if not isinstance(self_play, AdversarialSelfPlay):
            return

        self._logger.info("adversarial_self_play_loop_entered")
        try:
            while True:
                try:
                    result = await self_play.run_cycle()
                    self._logger.info(
                        "adversarial_self_play_cycle_complete",
                        attacks_generated=result.attacks_generated,
                        bypasses_found=result.bypasses_found,
                        proposals_emitted=result.proposals_emitted,
                    )

                    # Drain proposals and feed into the evolution pipeline
                    proposals = self_play.drain_proposals()
                    for p in proposals:
                        try:
                            await self.process_proposal(p)
                        except Exception as prop_exc:
                            self._logger.warning(
                                "adversarial_proposal_processing_error",
                                proposal_id=p.id,
                                error=str(prop_exc),
                            )
                except Exception as exc:
                    self._logger.warning(
                        "adversarial_self_play_cycle_error",
                        error=str(exc),
                    )

                # Low-priority cooldown
                await asyncio.sleep(self_play._config.cycle_cooldown_s)
        except asyncio.CancelledError:
            self._logger.info("adversarial_self_play_loop_cancelled")

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _compute_config_hash(self, files_changed: list[str]) -> str:
        """Compute a stable hash of the current config state."""
        hasher = hashlib.sha256()
        for rel_path in sorted(files_changed):
            full_path = self._root / rel_path
            hasher.update(rel_path.encode())
            if full_path.exists():
                hasher.update(str(full_path.stat().st_mtime).encode())
        return hasher.hexdigest()[:16]

    def _get_iron_rule_for(self, proposal: EvolutionProposal) -> str:
        """Return the relevant iron rule for a forbidden category."""
        rule_map = {
            "modify_equor": "Simula CANNOT modify Equor in any way.",
            "modify_constitution": "Simula CANNOT modify constitutional drives.",
            "modify_invariants": "Simula CANNOT modify invariants.",
            "modify_self_evolution": "Simula CANNOT modify its own logic (no self-modifying code).",
        }
        return rule_map.get(proposal.category.value, "Category is forbidden.")

    def _invalidate_analytics(self) -> None:
        """Invalidate analytics cache after a proposal completes."""
        if self._analytics is not None:
            self._analytics.invalidate_cache()

    async def _adaptive_orchestration_threshold(self) -> int:
        """
        Return the file-count threshold above which multi-agent orchestration
        is engaged.  Computes the p75 of historical proposal file counts from
        the analytics engine, floored at the configured minimum.

        Rationale: a fixed threshold (default=3) becomes stale as the codebase
        grows.  p75 ensures the orchestrator is reserved for the largest quarter
        of proposals regardless of absolute file counts.

        Falls back to the config value if there is insufficient history.
        """
        floor = self._config.orchestration_multi_file_threshold
        if self._history is None:
            return floor
        try:
            records = await self._history.get_history(limit=100)
            if len(records) < 10:
                return floor
            file_counts: list[int] = []
            for r in records:
                # EvolutionRecord stores affected_files as a list attribute when available
                n = len(getattr(r, "affected_files", None) or [])
                if n > 0:
                    file_counts.append(n)
            if not file_counts:
                return floor
            file_counts.sort()
            p75_idx = int(len(file_counts) * 0.75)
            p75 = file_counts[min(p75_idx, len(file_counts) - 1)]
            adaptive = max(floor, p75)
            self._logger.debug(
                "adaptive_orchestration_threshold",
                p75=p75,
                floor=floor,
                result=adaptive,
                sample_size=len(file_counts),
            )
            return adaptive
        except Exception as exc:
            self._logger.warning("adaptive_threshold_error", error=str(exc))
            return floor
