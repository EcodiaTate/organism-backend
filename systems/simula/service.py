"""
EcodiaOS — Simula Service

The self-evolution system. Simula is the organism's capacity for
metamorphosis: structural change beyond parameter tuning.

Where Evo adjusts the knobs, Simula redesigns the dashboard.

Simula coordinates the full evolution proposal pipeline:
  1. DEDUPLICATE — check for duplicate/similar active proposals
  2. VALIDATE    — reject forbidden categories immediately
  3. SIMULATE    — deep multi-strategy impact prediction
  4. GATE        — route governed changes through community governance
  5. APPLY       — invoke the code agent or config updater with rollback
  6. VERIFY      — health check post-application
  7. RECORD      — write immutable history, increment version, update analytics

Interfaces:
  initialize()            — build sub-systems, load current version
  process_proposal()      — main entry point for rich proposals
  receive_evo_proposal()  — receive from Evo via bridge translation
  get_history()           — recent evolution records
  get_current_version()   — current config version number
  get_analytics()         — evolution quality metrics
  shutdown()              — graceful teardown
  stats                   — service-level metrics

Iron Rules (never violated — see SIMULA_IRON_RULES in types.py):
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
from primitives.common import new_id, utc_now
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
from systems.simula.repair_memory import RepairMemory
from systems.simula.retrieval.swe_grep import SweGrepRetriever
from systems.simula.rollback import RollbackManager
from systems.simula.simulation import ChangeSimulator
from systems.simula.verification.incremental import IncrementalVerificationEngine
from systems.thymos.types import Incident, IncidentClass, IncidentSeverity

if TYPE_CHECKING:
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
    Simula — the EOS self-evolution system.

    Coordinates eight sub-systems:
      ChangeSimulator           — deep multi-strategy impact prediction
      SimulaCodeAgent           — Claude-backed code generation with 11 tools
      ChangeApplicator          — routes proposals to the right application strategy
      RollbackManager           — file snapshots and restore
      EvolutionHistoryManager   — immutable Neo4j history
      EvoSimulaBridge           — Evo→Simula proposal translation
      ProposalIntelligence      — deduplication, prioritization, dependency analysis
      EvolutionAnalyticsEngine  — evolution quality tracking
    """

    system_id: str = "simula"

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

        # Stage 7 sub-systems (Inspector — lazy runtime imports in initialize())
        self._inspector: InspectorService | None = None
        self._inspector_analytics: InspectorAnalyticsEmitter | None = None

        # Meta-healing: sentinel for reporting Simula's own failures to Thymos
        from systems.synapse.sentinel import ErrorSentinel
        self._sentinel = ErrorSentinel("simula")

        # Cross-system references wired after construction
        self._synapse: Any = None
        self._telos: Any = None  # TelosService — for constitutional binding validation
        self._evo: Any = None    # EvoService — for learned repair pattern validation

        # Oneiros lucid dream simulation results (mutation_id → report dict)
        self._dream_results: dict[str, dict[str, Any]] = {}
        self._dream_results_lock: asyncio.Lock = asyncio.Lock()

        # Grid metabolism state — pauses the evolution pipeline in CONSERVATION
        self._grid_state: str = "normal"  # MetabolicState values as strings
        self._evo_consolidation_stalled: bool = False
        self._evo_stall_expires_at: float = 0.0

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
                self._logger.warning("thinking_provider_init_failed", error=str(exc))

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
                self._logger.warning("embedding_client_init_failed", error=str(exc))

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
            from systems.simula.coevolution.hard_negative_miner import FailureAnalyzer

            self._hard_negative_miner = FailureAnalyzer(
                neo4j=self._neo4j,
                llm=self._llm,
                max_negatives_per_cycle=self._config.adversarial_max_tests_per_cycle,
            )
            self._logger.info("hard_negative_miner_initialized")

            if self._config.adversarial_test_generation_enabled:
                from systems.simula.coevolution.adversarial_tester import (
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

        # Wire SWE-grep into the bridge for pre-translation retrieval (3B.5)
        if self._bridge is not None and self._swe_grep is not None:
            self._bridge.set_swe_grep(self._swe_grep)

        # ── Stage 7: Inspector — Zero-Day Discovery Engine ─────────────────────
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

        # Pre-compute analytics from history
        if self._history is not None:
            try:
                await self._analytics.compute_analytics()
            except Exception as exc:
                self._logger.warning("initial_analytics_failed", error=str(exc))

        # Validate that all enabled external tool binaries are reachable.
        # Fail fast at startup rather than silently degrade on first use.
        await self._validate_tools()

        # ── Self-healing: RepairMemory + ProactiveScanner ─────────────────────
        self._repair_memory = RepairMemory(
            neo4j=self._neo4j,
            # event_bus wired after initialize() in set_synapse()
        )

        self._proactive_scanner = ProactiveScanner(
            neo4j=self._neo4j,
            event_bus=None,  # wired in set_synapse()
            process_proposal_fn=self.process_proposal,
        )

        # Grid metabolism subscription is deferred to set_synapse() because
        # Synapse is always built after SimulaService.initialize() returns.

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
        # always False at init time — Synapse is wired later. Subscribe here.
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
            except Exception as exc:
                self._logger.warning("simula_synapse_subscribe_failed", error=str(exc))

            # Wire event bus into self-healing sub-systems
            event_bus = getattr(synapse, "_event_bus", None)
            if event_bus is not None:
                if self._repair_memory is not None:
                    self._repair_memory._event_bus = event_bus
                if self._proactive_scanner is not None:
                    self._proactive_scanner._event_bus = event_bus
                # Meta-healing: wire sentinel so Simula can report its own failures
                self._sentinel.set_event_bus(event_bus)

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

    def set_telos(self, telos: Any) -> None:
        """Wire Telos for constitutional binding validation of mutation proposals."""
        self._telos = telos
        self._logger.info("telos_wired_to_simula")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so Simula can validate proposals against learned repair patterns.

        Called after initialize() during the integration wiring pass.
        Allows Simula to query Evo's procedural hypotheses about known failures
        and pre-emptively flag proposals that touch failure surfaces without
        including the known fix.
        """
        self._evo = evo
        self._logger.info("evo_wired_to_simula")

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
        unreported are the worst failure mode — they prevent the organism from
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

            incident = Incident(
                incident_class=IncidentClass.CRASH,
                severity=IncidentSeverity.HIGH,
                fingerprint=hashlib.md5(
                    f"simula_{error_type}".encode()
                ).hexdigest(),
                source_system="simula",
                error_type=error_type,
                error_message=error_message,
                context=context or {},
                affected_systems=["simula"],
                blast_radius=0.5,
                user_visible=False,
            )

            await event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_FAILED,
                    source="simula",
                    timestamp=time.time(),
                    data={"incident": incident.model_dump()},
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

        CONSERVATION: pause the evolution pipeline — no new proposals will be
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
        proposals sourced from Evo are deferred — applying structural changes
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

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        # Cancel the ProactiveScanner background task
        if self._proactive_scanner_task is not None and not self._proactive_scanner_task.done():
            self._proactive_scanner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._proactive_scanner_task
            self._proactive_scanner_task = None

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

        Does NOT block the proposal — it's advisory.
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
                    return True  # Nothing to validate — don't flag
                return fix in proposal_lower or fix.replace("_", " ") in proposal_lower

            missing_fixes = [h for h in flagged_hypotheses if not _fix_mentioned(h)]

            if missing_fixes:
                # Log the mismatch for metrics / potential HITL escalation.
                self._logger.warning(
                    "proposal_missing_learned_repairs",
                    proposal_id=proposal.id,
                    endpoints=endpoints,
                    flagged_hypothesis_count=len(flagged_hypotheses),
                    missing_count=len(missing_fixes),
                    missing_fix_summaries=[h.statement[:80] for h in missing_fixes],
                )

                # Optional: escalate high-confidence mismatches to HITL.
                high_confidence = [
                    h for h in missing_fixes
                    if h.evidence_score > 2.0
                ]
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

        # ── GRID CONSERVATION GATE ───────────────────────────────────────────
        # Do not run the expensive SimulaCodeAgent pipeline while the physical
        # grid is in carbon-heavy CONSERVATION mode.  Proposals are deferred
        # rather than lost — callers should retry once the grid recovers.
        if self._grid_state == "conservation":
            self._proposals_rejected += 1
            log.info(
                "proposal_deferred_grid_conservation",
                grid_state=self._grid_state,
            )
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason="Grid is in CONSERVATION mode — evolution pipeline paused. Retry when grid recovers.",
            )

        # ── EVO CONSOLIDATION STALL GATE ──────────────────────────────────────
        # When Evo's learning pipeline is stalled, defer Evo-sourced proposals.
        # Repairs from Thymos/Axon still proceed — they're urgent and don't
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
                        "Evo consolidation is stalled — evolution proposals "
                        "deferred until learning pipeline recovers."
                    ),
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
            # Unexpected exception in pipeline — emit to Thymos for immune response
            # The immune system must be able to heal itself
            log.error(
                "proposal_pipeline_exception",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            # Meta-healing: report via sentinel (structured Incident)
            await self._sentinel.report(
                exc,
                context={
                    "proposal_id": proposal.id,
                    "proposal_category": proposal.category.value,
                    "proposal_source": proposal.source,
                    "pipeline_stage": proposal.status.value,
                },
            )
            # Legacy fallback for backward compatibility
            await self._emit_incident_to_thymos(
                error_type=type(exc).__name__,
                error_message=str(exc),
                context={
                    "proposal_id": proposal.id,
                    "proposal_category": proposal.category.value,
                    "proposal_source": proposal.source,
                },
            )
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

    async def _run_pipeline(
        self, proposal: EvolutionProposal, log: Any
    ) -> ProposalResult:
        """Inner pipeline body, always called from process_proposal's try/finally."""
        # ── STEP 1: Validate ────────────────────────────────────────────────
        if proposal.category in FORBIDDEN:
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            reason = (
                f"Category '{proposal.category.value}' is forbidden for Simula. "
                f"Iron rule: {self._get_iron_rule_for(proposal)}"
            )
            log.warning("proposal_rejected_forbidden", reason=reason)
            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)

        # ── STEP 1.1: Telos Constitutional Binding ─────────────────────────
        # Validate that this mutation does not violate the drive topology.
        # Telos is the constitutional authority — if it says no, the proposal
        # is dead regardless of simulation outcome.
        if self._telos is not None:
            try:
                from systems.telos.types import (
                    TopologyValidationResult,
                    WorldModelUpdatePayload,
                )

                telos_payload = WorldModelUpdatePayload(
                    update_type=proposal.category.value,
                    delta_description=proposal.description,
                    source_system="simula",
                )
                telos_result = self._telos.validate_world_model_update(telos_payload)
                if telos_result == TopologyValidationResult.CONSTITUTIONAL_VIOLATION:
                    proposal.status = ProposalStatus.REJECTED
                    self._proposals_rejected += 1
                    reason = (
                        "Telos constitutional binder rejected this mutation: "
                        "it would violate the drive topology."
                    )
                    log.warning("proposal_rejected_constitutional", reason=reason)
                    async with self._proposals_lock:
                        self._active_proposals.pop(proposal.id, None)
                    return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)
                log.debug("telos_constitutional_check_passed")
            except Exception as exc:
                # Constitutional check failure is non-blocking — log and continue.
                # The binder itself is a safety net; simulation will catch issues too.
                log.warning("telos_constitutional_check_error", error=str(exc))

        # ── STEP 1.2: Validate Against Learned Repair Patterns ─────────────────
        # If Evo has learned procedures for failure patterns, check if this
        # proposal addresses code touching known-failure endpoints without
        # including the known fix. This is a soft warning, not a blocker.
        if self._evo is not None:
            await self._validate_against_learned_repairs(proposal)

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

        # ── STEP 2: Simulate (deep multi-strategy) ─────────────────────────
        # Skip if already triaged (has synthetic simulation)
        if proposal.simulation is None:
            proposal.status = ProposalStatus.SIMULATING
            log.info("proposal_simulating")

            try:
                simulation = await self._simulate_change(proposal)
                proposal.simulation = simulation
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

        if simulation.risk_level == RiskLevel.UNACCEPTABLE:
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            reason = f"Simulation shows unacceptable risk: {simulation.risk_summary}"
            log.warning("proposal_rejected_risk", risk_level=simulation.risk_level.value)
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
                # Dream approved — annotate for history
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
            return ProposalResult(
                status=ProposalStatus.AWAITING_GOVERNANCE,
                governance_record_id=governance_id,
            )

        # ── STEP 4: Apply (self-applicable changes only) ───────────────────
        return await self._apply_change(proposal)

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

        # Low-risk proposals are auto-approved — no governance bottleneck
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

        Inspector is purely additive — it never modifies EOS files and all
        analysis happens in temporary workspaces.

        Args:
            github_url: HTTPS URL of the target repository.
            authorized_targets: Override config authorized targets for this hunt.
                Creates a scoped config copy — the shared config is never mutated.
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
        responsible for the wiring — this method is intentionally separate from
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

                    threshold = self._config.orchestration_multi_file_threshold
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

        code_result, snapshot = await self._applicator.apply(proposal)
        # Stamp the snapshot with the version that was current before this
        # change was applied, so rollback audit trails show the correct target.
        snapshot.config_version = self._current_version

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
                        # Mark as success — diffusion repair saved the change
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

            # Meta-healing: report rollback to Thymos so the immune system
            # knows a repair attempt failed and can escalate
            await self._sentinel.report(
                RuntimeError(f"Proposal rollback: {code_result.error}"),
                context={
                    "proposal_id": proposal.id,
                    "category": proposal.category.value,
                    "files_affected": code_result.files_written[:10],
                },
            )

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
                    pass  # Best-effort — sentinel report above is primary

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

            async with self._proposals_lock:
                self._active_proposals.pop(proposal.id, None)
            self._invalidate_analytics()
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

        # Stash formal verification result for history recording
        if health.formal_verification is not None:
            proposal._formal_verification_result = health.formal_verification  # type: ignore[attr-defined]

        # Stash Lean 4 verification result for history recording
        if health.lean_verification is not None:
            proposal._lean_verification_result = health.lean_verification  # type: ignore[attr-defined]

        # Stash Stage 6 formal guarantees result for history recording
        if health.formal_guarantees is not None:
            proposal._formal_guarantees_result = health.formal_guarantees  # type: ignore[attr-defined]

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
                await self._rollback.restore(snapshot)
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
                    import asyncio
                    asyncio.create_task(self._grpo_retrain_background())
            except Exception as exc:
                log.warning("grpo_record_error", error=str(exc))

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

        return ProposalResult(
            status=ProposalStatus.APPLIED,
            version=self._current_version,
            files_changed=code_result.files_written,
            pr_url=code_result.pr_url,
            pr_number=code_result.pr_number,
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
            from systems.simula.coevolution.hard_negative_miner import FailureAnalyzer

            if not isinstance(self._hard_negative_miner, FailureAnalyzer):
                return

            # Import adversarial tester if available
            adversarial_gen = None
            if self._adversarial_tester is not None:
                from systems.simula.coevolution.adversarial_tester import (
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
