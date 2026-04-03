"""
EcodiaOS - Configuration System

All configuration is Pydantic-validated and loaded from:
1. default.yaml (defaults)
2. Environment variables (overrides)
3. Seed config (instance birth parameters)

Every tunable parameter in the system lives here.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Sub-configs ──────────────────────────────────────────────────


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    ws_port: int = 8001
    federation_port: int = 8002
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_key_header: str = "X-EOS-API-Key"
    # API keys for authentication. When empty, auth is disabled (dev mode).
    # Set via ORGANISM_SERVER__API_KEYS or config YAML.
    api_keys: list[str] = Field(default_factory=list)


class Neo4jConfig(BaseModel):
    uri: str = ""  # Required: set via ORGANISM_NEO4J_URI (e.g., neo4j+s://xxx.databases.neo4j.io)
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_connection_pool_size: int = 20

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                'Neo4j URI required. Set ORGANISM_NEO4J_URI '
                '(e.g., "neo4j+s://your-instance.databases.neo4j.io")'
            )
        return v.strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Neo4j password required. Set ORGANISM_NEO4J_PASSWORD.")
        return v.strip()


class TimescaleDBConfig(BaseModel):
    host: str = "timescaledb"
    port: int = 5432
    database: str = "ecodiaos"
    schema_name: str = Field(default="public", alias="schema")
    username: str = "ecodiaos"
    password: str = "ecodiaos_dev"
    pool_size: int = 10
    ssl: bool = False

    model_config = {"populate_by_name": True}

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisConfig(BaseModel):
    url: str = "redis://redis:6379/0"
    prefix: str = "eos"
    password: str = ""

    @property
    def full_url(self) -> str:
        """Build URL with password injected."""
        clean_pw = self.password.strip() if self.password else ""
        if clean_pw and "://" in self.url:
            scheme, rest = self.url.split("://", 1)
            return f"{scheme}://:{clean_pw}@{rest}"
        return self.url


class LLMBudget(BaseModel):
    # Observability thresholds only — not gates. Tier warnings inform Soma;
    # they do NOT block LLM calls or force heuristic fallbacks.
    # Set hard_limit=True only to protect an external billing ceiling.
    max_calls_per_hour: int = 10_000
    max_tokens_per_hour: int = 10_000_000
    hard_limit: bool = False


class LLMConfig(BaseModel):
    provider: str = "bedrock"
    model: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    api_key: str = ""
    endpoint: str = "http://localhost:8000"  # used by vllm/ollama providers
    fallback_provider: str | None = None
    fallback_model: str | None = None
    budget: LLMBudget = Field(default_factory=LLMBudget)

    @model_validator(mode="after")
    def _strip_api_key(self) -> LLMConfig:
        # GCP Secret Manager can inject trailing \r\n into env vars
        if self.api_key:
            object.__setattr__(self, "api_key", self.api_key.strip())
        return self


class HotSwapConfig(BaseModel):
    """Configuration for live model hot-swapping and rollback."""

    enabled: bool = False
    # Probation period: number of Synapse cycles to monitor after a swap
    probation_cycles: int = 100
    # Error rate threshold (0.0–1.0) during probation before rollback triggers
    error_rate_threshold: float = 0.05
    # vLLM / Ollama endpoint for local inference when swapping from API
    local_inference_endpoint: str = "http://localhost:8000"
    local_inference_provider: str = "vllm"  # "vllm" | "ollama"
    # Base model used by the local inference engine
    local_base_model: str = "Qwen/Qwen3-8B"
    # Timeout for adapter loading operations (seconds)
    adapter_load_timeout_s: float = 120.0
    # Whether to automatically revert on probation failure (vs. just alerting)
    auto_rollback: bool = True


class EmbeddingConfig(BaseModel):
    strategy: str = "local"  # "local" | "api" | "sidecar" | "mock"
    local_model: str = "sentence-transformers/all-mpnet-base-v2"
    local_device: str = "cpu"
    sidecar_url: str | None = None
    dimension: int = 768
    max_batch_size: int = 32
    cache_embeddings: bool = True
    cache_ttl_seconds: int = 3600
    # Semantic compression (long-term memory PCA)
    compression_enabled: bool = True
    compression_variance_target: float = 0.80  # target cumulative variance retention
    compression_fixed_k: int | None = None  # None = adaptive K selection
    compression_min_episodes: int = 50  # per-community threshold
    compression_quality_threshold: float = 0.90  # abort if reconstruction quality drops below


class SynapseConfig(BaseModel):
    cycle_period_ms: int = 150
    min_cycle_period_ms: int = 80
    max_cycle_period_ms: int = 500
    health_check_interval_ms: int = 5000
    health_failure_threshold: int = 3
    coherence_update_interval: int = 50
    resource_snapshot_interval: int = 33
    rebalance_interval: int = 100
    metabolic_interval: int = 50
    infra_cost_poll_interval_s: int = 300
    rhythm_enabled: bool = True


class AtuneConfig(BaseModel):
    ignition_threshold: float = 0.3
    workspace_buffer_size: int = 32
    spontaneous_recall_base_probability: float = 0.02
    max_percept_queue_size: int = 100


class NovaConfig(BaseModel):
    max_active_goals: int = 20
    fast_path_timeout_ms: int = 300
    slow_path_timeout_ms: int = 15000
    max_policies_per_deliberation: int = 5
    # EFE component weights (Evo adjusts these over time)
    efe_weight_pragmatic: float = 0.35
    efe_weight_epistemic: float = 0.20
    efe_weight_constitutional: float = 0.20
    efe_weight_feasibility: float = 0.15
    efe_weight_risk: float = 0.10
    # Memory retrieval timeout for belief enrichment
    memory_retrieval_timeout_ms: int = 150
    # Whether to use LLM for pragmatic/epistemic EFE estimation (vs. heuristics)
    use_llm_efe_estimation: bool = True
    # Autonomous heartbeat: how often (seconds) Nova checks its drives and hunts
    heartbeat_interval_seconds: int = 3600
    # Oikos balance below this (USD) is considered "hungry" → triggers hunt
    hunger_balance_threshold_usd: float = 50.0
    # ── Cognition Cost (Metabolic Budgeting) ──
    # Cognitive frugality weight (λ) in EFE: higher = prefer cheaper policies
    efe_weight_cognition_cost: float = 0.10
    # Enable cognition cost integration in EFE computation (spec §12: enable_cognition_budgeting)
    enable_cognition_budgeting: bool = True
    # Enable Evo hypothesis tournament loop (spec §12: enable_hypothesis_tournaments)
    enable_hypothesis_tournaments: bool = True
    # Budget per importance tier (USD)
    cognition_budget_low: float = 0.10
    cognition_budget_medium: float = 0.50
    cognition_budget_high: float = 2.00
    cognition_budget_critical: float = 5.00
    # LLM cost rates (override for non-default models)
    cost_llm_input_per_1m_tokens: float = 3.00
    cost_llm_output_per_1m_tokens: float = 15.00
    cost_gpu_hourly_rate: float = 0.50
    cost_db_per_ms: float = 0.000001
    cost_io_per_gb: float = 0.09
    # Motor degradation: skip replan if goal is this far complete
    motor_degradation_replan_threshold: float = 0.8
    # Soma tick modulation thresholds
    soma_urgency_modulate_threshold: float = 0.7
    soma_energy_modulate_threshold: float = 0.3
    soma_urgency_emit_threshold: float = 0.5
    soma_energy_emit_threshold: float = 0.4
    # Cognitive pressure levels for policy-K modulation
    cognitive_pressure_low: float = 0.85
    cognitive_pressure_high: float = 0.95


class EquorConfig(BaseModel):
    standard_review_timeout_ms: int = 500
    critical_review_timeout_ms: int = 50
    care_floor_multiplier: float = -0.3
    honesty_floor_multiplier: float = -0.3
    drift_window_size: int = 1000
    drift_report_interval: int = 1000  # every N reviews
    # How long a suspended HITL intent remains valid in Redis (seconds).
    # Default is 24 hours - long enough for a human operator in a different
    # timezone to review. The old 1-hour default expired before most review
    # workflows could complete.
    hitl_intent_ttl_s: int = 86400
    # Single-transaction capital threshold (USD) above which the organism must
    # get human approval before executing. Overridable via EOS_HITL_CAPITAL_THRESHOLD.
    # Default: $500.  Actions below this are AUTONOMOUS.
    hitl_capital_threshold_usd: float = 500.0
    # Autonomy promotion thresholds (level 1→2 and level 2→3).
    # These only apply when recovering from a demotion; the organism starts at level 3.
    # 0 = no minimum required.
    promote_1_to_2_min_decisions: int = 0
    promote_1_to_2_min_alignment: float = 0.0
    promote_1_to_2_min_days: int = 0
    promote_2_to_3_min_decisions: int = 0
    promote_2_to_3_min_alignment: float = 0.0
    promote_2_to_3_min_days: int = 0
    # Automatic demotion: mean Care alignment below this over demotion_window decisions
    demotion_care_threshold: float = -0.2
    demotion_window_size: int = 100


class MEVConfig(BaseModel):
    """Configuration for MEV-Aware Transaction Timing (Prompt #12: Predator Detection)."""

    enabled: bool = True
    # RPC URL for EVM state queries (forked simulation).
    # Uses the same Base L2 RPC as the wallet if not overridden.
    # Set via ORGANISM_MEV__RPC_URL
    rpc_url: str = ""
    # MEV risk score threshold for engaging Flashbots Protect
    high_risk_threshold: float = 0.7
    # Maximum analysis time before falling back to heuristics (ms)
    analysis_timeout_ms: int = 5000
    # Block competition monitoring poll interval (seconds)
    block_competition_poll_interval_s: float = 12.0
    # Enable EVM fork simulation (requires RPC with eth_call support)
    simulation_enabled: bool = True
    # Maximum protection cost as fraction of transaction value
    max_protection_cost_ratio: float = 0.05
    # Flashbots Protect RPC endpoint (Base L2)
    flashbots_rpc_url: str = ""


class AxonConfig(BaseModel):
    # All 0 = unlimited. Set to non-zero in config to restrict.
    # Constraints are opt-in safety valves, not default ceilings.
    max_actions_per_cycle: int = 0
    max_api_calls_per_minute: int = 0
    max_notifications_per_hour: int = 0
    max_concurrent_executions: int = 0
    total_timeout_per_cycle_ms: int = 0


class VoxisConfig(BaseModel):
    max_expression_length: int = 2000
    min_expression_interval_minutes: int = 1
    voice_synthesis_enabled: bool = False
    # Proactive expression threshold - ambient insights below this are suppressed
    insight_expression_threshold: float = 0.6
    # Rolling message window kept verbatim in conversation context
    conversation_history_window: int = 50
    # Max tokens to pass to LLM as conversation history (older messages summarised)
    context_window_max_tokens: int = 4000
    # Summarise messages older than this count (keep last N verbatim)
    conversation_summary_threshold: int = 10
    # Whether to report expression feedback to Atune
    feedback_enabled: bool = True
    # Whether to run post-generation honesty check
    honesty_check_enabled: bool = True
    # Base LLM temperature before strategy modulation
    temperature_base: float = 0.7
    # Maximum concurrent tracked conversations in Redis
    max_active_conversations: int = 50


class EvoConfig(BaseModel):
    consolidation_interval_hours: int = 6
    consolidation_cycle_threshold: int = 10000
    max_active_hypotheses: int = 50
    max_parameter_delta_per_cycle: float = 0.03
    min_evidence_for_integration: int = 10
    # How often Evo attempts to generate new hypotheses from accumulated pattern
    # candidates (every N broadcast cycles).  Lower = more responsive but higher
    # LLM cost.  Genome-heritable via EvoGenomeExtractor so child instances can
    # inherit a tuned value discovered by the parent.
    hypothesis_generation_interval: int = 200
    # How often Evo re-evaluates evidence for all active hypotheses (every N
    # broadcast cycles).  Lower = faster convergence but more compute per cycle.
    evidence_evaluation_interval: int = 50
    # How many direction reversals in the 10-nudge window trigger a HIGH_VOLATILITY
    # flag, halving the weight of the hypothesis's evidence until it stabilises.
    volatility_oscillation_threshold: int = 6
    # Consecutive RE_DECISION_OUTCOME events with success_rate < 0.60 before Evo
    # queues a hyperparameter-adjustment PatternCandidate.  Genome-heritable.
    re_degradation_count_threshold: int = 10
    # Maximum concurrent cognitive niches. 0 = unlimited.
    max_niches: int = 0
    # Minimum hypotheses to sustain a niche before starvation kicks in.
    niche_min_population: int = 5
    # Cycles with no output before a niche goes extinct.
    niche_starvation_cycles: int = 5
    # RE training check interval (seconds). How often the registry checks if the RE needs retraining.
    re_train_check_interval_s: int = 21600


class SimulaConfig(BaseModel):
    max_simulation_episodes: int = 200
    regression_threshold_unacceptable: float = 0.10
    regression_threshold_high: float = 0.05
    # Rollback snapshot TTL (seconds) - prevents unbounded Redis growth
    rollback_snapshot_ttl_seconds: int = 3600  # 1 hour
    # Code agent settings
    codebase_root: str = "."
    code_agent_model: str = "claude-opus-4-6"
    max_code_agent_turns: int = 20
    # test_command: the shell command used to run the test suite.
    # Leave as "" to use the platform-aware default (sys.executable -m pytest).
    # On Windows, "pytest" without a full path may not be found in all environments;
    # the empty-string default lets HealthChecker build the command with Path.as_posix()
    # and shlex.quote so the executable is always the running interpreter.
    test_command: str = ""
    auto_apply_self_applicable: bool = True
    # Stage 1A: Extended-thinking model for governance-required and high-risk proposals
    thinking_model: str = "o3"
    thinking_model_provider: str = "openai"
    thinking_model_api_key: str = ""
    thinking_budget_tokens: int = 16384  # max reasoning tokens for extended thinking
    # Stage 1B: Code embeddings for semantic similarity
    embedding_model: str = "voyage-code-3"
    embedding_api_key: str = ""
    # Stage 1C: KV cache compression
    kv_compression_ratio: float = 0.3  # prune ratio for KVzip (0.0 = no pruning, 1.0 = max)
    kv_compression_enabled: bool = True
    # Stage 2A: Dafny formal verification (Clover pattern)
    dafny_enabled: bool = False  # requires Dafny binary on PATH
    dafny_binary_path: str = "dafny"
    dafny_verify_timeout_s: float = 30.0
    dafny_max_clover_rounds: int = 8
    dafny_blocking: bool = True  # Dafny failure blocks triggerable categories
    # Stage 2B: Z3 invariant discovery
    z3_enabled: bool = False  # requires z3-solver pip package
    z3_check_timeout_ms: int = 5000
    z3_max_discovery_rounds: int = 6
    z3_blocking: bool = True  # blocking by default for all critical categories (Spec §9)
    # Stage 2C: Static analysis gates
    static_analysis_enabled: bool = True  # on by default (pip packages)
    static_analysis_max_fix_iterations: int = 3
    static_analysis_blocking: bool = True  # ERROR findings block proposals
    # Stage 2D: AgentCoder (separated test/code/execute pipeline)
    agent_coder_enabled: bool = False  # opt-in 3-agent pipeline
    agent_coder_max_iterations: int = 3
    agent_coder_test_timeout_s: float = 60.0
    # Stage 3A: Salsa Incremental Verification
    incremental_verification_enabled: bool = True  # dependency-aware memoization
    incremental_hot_ttl_seconds: int = 3600  # Redis hot cache TTL (1 hour)
    # Stage 3B: SWE-grep Agentic Retrieval
    swe_grep_enabled: bool = True  # multi-hop code search for bridge + code agent
    swe_grep_max_hops: int = 4  # serial retrieval turns (4 × 8 parallel tools)
    # Stage 3C: LILO Library Learning
    lilo_enabled: bool = True  # abstraction extraction from successful proposals
    lilo_max_library_size: int = 200  # cap on :LibraryAbstraction nodes
    lilo_consolidation_interval_proposals: int = 10  # consolidate every N applied proposals
    # Stage 4A: Lean 4 Proof Generation (DeepSeek-Prover-V2 pattern)
    lean_enabled: bool = False  # requires Lean 4 + Mathlib on PATH
    lean_binary_path: str = "lean"  # path to Lean 4 binary
    lean_project_path: str = ""  # path to lakefile.lean project (for Mathlib deps)
    lean_verify_timeout_s: float = 60.0  # per-proof verification timeout
    lean_max_attempts: int = 0  # 0 = unlimited proof generation attempts
    lean_blocking: bool = True  # Lean failure blocks for proof-requiring categories
    lean_copilot_enabled: bool = True  # use Lean Copilot for tactic automation
    lean_dojo_enabled: bool = True  # use LeanDojo for proof search and retrieval
    lean_proof_library_max_size: int = 500  # cap on :ProvenLemma nodes in Neo4j
    # Stage 4B: GRPO Domain Fine-Tuning
    grpo_enabled: bool = False  # opt-in self-improvement training loop
    grpo_base_model: str = "deepseek-coder-7b"  # 7B base model for fine-tuning
    grpo_min_training_examples: int = 100  # minimum examples before first SFT
    grpo_sft_epochs: int = 3  # supervised fine-tuning epochs
    grpo_rollouts_per_example: int = 2  # 2-rollout contrastive (matches 16-rollout)
    grpo_batch_size: int = 8  # training batch size
    grpo_learning_rate: float = 2e-5  # fine-tuning learning rate
    grpo_retrain_interval_proposals: int = 50  # retrain every N applied proposals
    grpo_gpu_ids: list[int] = Field(default_factory=lambda: [0])  # GPU allocation
    grpo_use_finetuned: bool = False  # route code agent to fine-tuned model
    grpo_ab_test_fraction: float = 0.2  # fraction of proposals routed to fine-tuned
    grpo_vllm_port: int = 8000  # port for local vLLM inference server
    grpo_model_dir: str = ""  # directory for saved LoRA adapters (default: codebase_root/.grpo)
    grpo_training_timeout_s: float = 3600.0  # max training subprocess duration (1 hour)
    grpo_local_max_tokens: int = 4096  # max tokens for local model generation
    grpo_novelty_threshold: float = 0.3  # success rate below which task is "novel" (use API)
    # Stage 4C: Diffusion-Based Code Repair
    diffusion_repair_enabled: bool = False  # opt-in last-mile repair agent
    diffusion_model: str = "diffucoder-7b"  # diffusion model ID
    diffusion_max_denoise_steps: int = 10  # maximum denoising iterations
    diffusion_timeout_s: float = 120.0  # total repair timeout
    diffusion_sketch_first: bool = False  # True = skeleton mode, False = iterative denoise
    diffusion_handoff_after_failures: int = 2  # hand off to diffusion after N code agent failures
    # Stage 5A: Neurosymbolic Synthesis (beyond CEGIS)
    synthesis_enabled: bool = False  # opt-in neurosymbolic synthesis routing
    hysynth_enabled: bool = True  # probabilistic CFG-guided search (within synthesis)
    hysynth_max_candidates: int = 200  # max candidate programs per synthesis
    hysynth_beam_width: int = 10  # beam search width for bottom-up enumeration
    hysynth_timeout_s: float = 60.0  # per-synthesis timeout
    sketch_synthesis_enabled: bool = True  # LLM sketch + symbolic hole-filling
    sketch_max_holes: int = 20  # max holes per template
    sketch_solver_timeout_ms: int = 5000  # Z3/constraint solver timeout per hole
    chopchop_enabled: bool = True  # type-directed constrained generation
    chopchop_chunk_size_lines: int = 10  # lines per constrained generation chunk
    chopchop_max_retries: int = 0  # 0 = unlimited retries per chunk
    chopchop_timeout_s: float = 90.0  # total timeout
    # Stage 5B: Neural Program Repair (SRepair pattern)
    repair_agent_enabled: bool = False  # opt-in FSM-guided repair
    repair_diagnosis_model: str = "claude-opus-4-6"  # reasoning model for root cause
    repair_generation_model: str = "claude-sonnet-4-20250514"  # code model for fix gen
    repair_max_retries: int = 0  # 0 = unlimited repair attempts
    repair_cost_budget_usd: float = 0.0  # 0 = no cost cap
    repair_timeout_s: float = 180.0  # total repair timeout
    repair_use_similar_fixes: bool = True  # query Neo4j for similar past repairs
    # Stage 5C: Multi-Agent Orchestration
    orchestration_enabled: bool = False  # opt-in multi-agent pipeline
    orchestration_max_agents_per_stage: int = 2  # per "overcrowding" finding
    orchestration_multi_file_threshold: int = 3  # files >= this triggers orchestrator
    orchestration_max_dag_nodes: int = 50  # cap on task decomposition DAG size
    orchestration_timeout_s: float = 300.0  # total orchestration timeout
    # Stage 5D: Causal Debugging
    causal_debugging_enabled: bool = False  # opt-in causal analysis on failure
    causal_max_interventions: int = 5  # max interventional queries per diagnosis
    causal_fault_injection_enabled: bool = False  # active causal learning (staging only)
    causal_timeout_s: float = 60.0  # per-diagnosis timeout
    # Stage 5E: Autonomous Issue Resolution
    issue_resolution_enabled: bool = False  # opt-in autonomous resolution
    issue_max_autonomy_level: str = "logic_bug"  # full autonomy — no ceiling on what it can fix
    issue_abstention_confidence_threshold: float = 0.0  # 0 = LLM decides whether to act
    issue_perf_regression_enabled: bool = True  # detect perf regressions post-apply
    issue_security_scan_enabled: bool = True  # enhanced security scanning
    issue_degradation_window_hours: int = 24  # monitor window for subtle degradation
    # Stage 6A: Cryptographic Auditability
    hash_chain_enabled: bool = False  # SHA-256 hash chains on EvolutionRecord nodes
    c2pa_enabled: bool = False  # C2PA content credentials for code provenance
    c2pa_signing_key_path: str = ""  # path to Ed25519 private key for signing
    c2pa_issuer_name: str = "EcodiaOS Simula"  # issuer name in C2PA manifests
    verifiable_credentials_enabled: bool = False  # tamper-evident governance approval chain
    credential_verification_timeout_s: float = 10.0  # timeout for credential verification
    regulatory_framework: str = ""  # ""|"finance_sox"|"healthcare_hipaa"|"defense_cmmc"|"general_audit"
    # Stage 6B: Co-Evolving Agents
    coevolution_enabled: bool = False  # autonomous hard negative mining + adversarial testing
    hard_negative_mining_interval_proposals: int = 10  # mine every N applied proposals
    adversarial_test_generation_enabled: bool = False  # LLM generates edge-case tests
    adversarial_max_tests_per_cycle: int = 20  # cap on adversarial tests per cycle
    coevolution_idle_compute_enabled: bool = False  # run adversarial generation on idle cycles
    # Stage 6C: Formal Spec Generation
    formal_spec_generation_enabled: bool = False  # auto-generate Dafny/TLA+/Alloy specs
    dafny_spec_generation_enabled: bool = True  # Dafny spec gen (within formal_spec_generation)
    dafny_bench_coverage_target: float = 0.96  # DafnyBench 96% coverage target
    tla_plus_enabled: bool = False  # TLA+ specs for distributed interactions
    tla_plus_binary_path: str = "tlc"  # path to TLC model checker binary
    tla_plus_model_check_timeout_s: float = 120.0  # per-model-check timeout
    alloy_enabled: bool = False  # Alloy for property checking on system invariants
    alloy_binary_path: str = "alloy"  # path to Alloy analyzer binary
    alloy_scope: int = 10  # Alloy scope (bound on universe size)
    self_spec_dsl_enabled: bool = False  # LLMs invent task-specific DSLs
    # Stage 6D: Equality Saturation (E-graphs)
    egraph_enabled: bool = False  # e-graph refactoring with semantic equivalence
    egraph_max_iterations: int = 1000  # max saturation iterations
    egraph_timeout_s: float = 30.0  # per-equivalence-check timeout
    egraph_blocking: bool = False  # advisory by default, equivalence failures don't block
    # Stage 6E: Hybrid Symbolic Execution
    symbolic_execution_enabled: bool = False  # Z3 SMT for mission-critical logic proofs
    symbolic_execution_timeout_ms: int = 10000  # Z3 per-property timeout
    symbolic_execution_blocking: bool = True  # proved properties are hard guarantees
    symbolic_execution_domains: list[str] = Field(
        default_factory=lambda: ["budget_calculation", "risk_scoring"],
    )  # domains to target for symbolic execution
    # Validation pipeline stages (post-apply health check)
    integration_tests_enabled: bool = False  # run tests/integration/systems/<name>/ after mutation
    integration_tests_timeout_s: float = 120.0  # timeout per integration test run
    performance_baseline_enabled: bool = False  # fail on >10% benchmark regression
    performance_baseline_timeout_s: float = 60.0  # timeout for benchmark run
    # Worker deployment
    run_worker_in_process: bool = False  # spawn simula_worker as background asyncio task (dev only)
    # Pipeline safety limits
    max_active_proposals: int = 50  # reject new proposals when this many are in-flight
    pipeline_timeout_s: float = 600.0  # hard timeout for the full process_proposal() pipeline (10 min)
    # Per-stage timeout budgets - individual stages cannot blow out the total deadline.
    # Alerts are emitted when actual time exceeds stage_budget_alert_fraction of the budget.
    validate_stage_timeout_s: float = 1.0    # governance validation + triage
    simulate_stage_timeout_s: float = 30.0   # deep simulation (all strategies combined)
    apply_stage_timeout_s: float = 60.0      # code generation + file writes
    verify_stage_timeout_s: float = 120.0    # health check + formal verification
    record_stage_timeout_s: float = 5.0      # Neo4j audit record write
    stage_budget_alert_fraction: float = 0.9  # emit alert when stage uses >90% of its budget
    # Stage 7: Inspector - Zero-Day Discovery Engine
    inspector_enabled: bool = False  # opt-in vulnerability hunting
    inspector_max_workers: int = 4  # concurrent surface × goal analysis workers (1-16)
    inspector_sandbox_timeout_s: int = 30  # PoC sandbox execution timeout
    inspector_clone_depth: int = 1  # git clone depth (1 = shallow)
    inspector_log_analytics: bool = True  # emit structlog analytics events
    inspector_authorized_targets: list[str] = Field(
        default_factory=list,
    )  # domains allowed for PoC execution
    inspector_generate_pocs: bool = False  # auto-generate exploit PoC scripts
    inspector_generate_patches: bool = False  # auto-generate + verify patches
    inspector_remediation_enabled: bool = False  # enable InspectorRepairOrchestrator
    # Configurable category validation rules (JSON schema).
    # Each key is a ChangeCategory value; each value is a dict with:
    #   name_field: str       - which ChangeSpec attribute holds the name
    #   required_fields: list[str] - spec fields that must be non-empty
    #   naming_convention: str - "snake_case" | "pascal_case" | "valid_identifier"
    #   example: str          - shown in error messages (e.g., 'email_sender')
    # Leave empty to use built-in defaults (equivalent to the hard-coded rules).
    category_validation_rules: dict[str, Any] = Field(default_factory=dict)

    # ─── Learnable Risk Weighting (Corpus 5) ──────────────────────────────────
    # Risk synthesis weights for ChangeSimulator._synthesize_risk() weighting.
    # Defaults match Spec §11; tunable via Evo ADJUST_BUDGET proposals targeting
    # risk_weight_* parameters. Normalized to sum=1.0 before use.
    risk_weight_base: float = 0.40  # Base category simulation result
    risk_weight_counterfactual: float = 0.20  # Counterfactual regression rate
    risk_weight_dependency: float = 0.15  # Dependency blast radius
    risk_weight_resource: float = 0.10  # Resource cost estimation
    risk_weight_alignment: float = 0.15  # Constitutional alignment

    # ─── Learnable Configuration Defaults (Corpus 8) ────────────────────────────
    # Hard-coded heuristics extracted to SimulaConfig so Evo can tune them.
    # Defaults match current behavior in simulation.py + architecture_efe_scorer.py.
    # Evo learns optimal values via ADJUST_BUDGET proposals + analytics feedback.

    # Risk thresholds for mapping composite_risk → RiskLevel
    risk_threshold_moderate: float = 0.25  # composite_risk >= 0.25 → MODERATE
    risk_threshold_high: float = 0.50  # composite_risk >= 0.50 → HIGH
    risk_threshold_unacceptable: float = 0.75  # composite_risk >= 0.75 → UNACCEPTABLE

    # EFE (Expected Free Energy) scoring thresholds
    efe_auto_approve_threshold: float = -0.35  # Equor auto-approves if EFE <= this
    efe_complexity_penalty_weight: float = 0.15  # Complexity penalty multiplier

    # Proposal processing limits
    proactive_scan_incident_limit_per_cycle: int = 1000  # Max patterns per scan
    proactive_scan_interval_accelerated_factor: float = 0.4  # 40% of normal
    proactive_scan_interval_critical_factor: float = 0.15  # 15% of normal

    # Repair memory thresholds
    repair_low_success_rate_threshold: float = 0.40  # Below this → scrutiny

    # Fovea attention thresholds
    fovea_divergence_threshold: float = 0.25  # 25% divergence triggers attention

    # Category-level base complexity heuristics (0.0-1.0 scale)
    category_complexity_adjust_budget: float = 0.05
    category_complexity_add_pattern: float = 0.20
    category_complexity_add_channel: float = 0.25
    category_complexity_add_executor: float = 0.30
    category_complexity_modify_contract: float = 0.45
    category_complexity_modify_cycle: float = 0.35
    category_complexity_change_consolidation: float = 0.40
    category_complexity_add_system: float = 0.60

    # Category-specific risk boost/penalty (applied after weighted synthesis)
    category_risk_boost_modify_contract: float = 0.2
    category_risk_boost_add_system: float = 0.2
    category_risk_boost_adjust_budget: float = -0.1

    # ─── Economic Learnable Parameters (Fix 4.1) ──────────────────────────────
    # These 10 parameters govern economic decision-making thresholds.
    # Evo tunes them via ADJUST_BUDGET proposals targeting econ_* parameter names.
    # All bounds enforced in SimulaService._on_evo_adjust_budget().

    # DeFi yield farming thresholds
    yield_apy_drop_rebalance_threshold: float = 0.80  # rebalance if APY < 80% of expected
    yield_apy_minimum_acceptable: float = 0.03  # reject positions below 3% APY

    # Bounty hunting thresholds
    bounty_min_roi_multiple: float = 0.0   # 0 = Evo decides, no ROI floor
    bounty_max_risk_score: float = 1.0    # 1.0 = Evo evaluates all bounties, no risk ceiling

    # Asset development budgeting
    asset_dev_budget_pct: float = 0.15  # allocate up to 15% of liquid balance

    # Child spawning — Evo decides timing and viability, not hardcoded floors
    child_spawn_interval_days: float = 0.0   # 0 = no mandatory wait between spawns
    child_min_profitability_usd: float = 0.0  # 0 = LLM decides viability, not a dollar floor

    # Cost management
    cost_reduction_target_pct: float = 0.10  # target 10% cost reduction per cycle

    # Emergency thresholds
    emergency_liquidation_threshold: float = 0.10  # liquidate yield if balance < 10% of target

    # Protocol exploration budget
    protocol_exploration_budget_pct: float = 0.20  # allocate up to 20% for new protocols

    # Protocol allocation aggressiveness (SIMULA-ECON-1)
    # Controls how aggressively capital is concentrated in top-performing protocols.
    # 0.1 = cautious (spread across many), 1.0 = aggressive (max concentration in #1).
    protocol_allocation_aggressiveness: float = 0.5

    # Preventive audit scan sensitivity - Evo-tunable via ADJUST_BUDGET.
    # 0.0 = only flag combined_risk > 0.7 (critical only)
    # 0.5 = default thresholds (critical > 0.7, risk 0.4–0.7)
    # 1.0 = flag everything above fragility_score > 0.2
    audit_aggressiveness: float = 0.5

    # ── Asset Factory Policy (Evo-tunable via ADJUST_BUDGET) ──────────────────
    # All thresholds are defaults — Evo proposes adjustments based on observed
    # outcomes. Set permissively so Evo has room to explore.
    asset_min_roi_threshold: float = 1.0        # Evo starts at 1x ROI (break-even)
    asset_max_break_even_days: int = 180         # Generous initial deadline
    asset_min_market_gap_confidence: float = 0.1 # Low floor — Evo learns what works
    asset_max_concurrent: int = 20              # High cap — Evo tightens if needed
    asset_min_liquid_after_dev_usd: float = 5.0  # Minimal floor — survival reserve handles real protection
    asset_min_dev_cost_usd: float = 0.10         # Near-zero — LLM decides minimum viable cost
    asset_max_dev_cost_usd: float = 5000.0       # High cap — Evo tightens per available capital
    asset_revenue_decline_terminate_days: int = 60  # Days of decline before termination review
    asset_break_even_deadline_days: int = 180    # Days before deadline-miss triggers review


class ThymosConfig(BaseModel):
    # Sentinel scan interval (seconds)
    sentinel_scan_interval_s: float = 30.0
    # Homeostasis check interval (seconds)
    homeostasis_interval_s: float = 30.0
    # Post-repair verification timeout (seconds)
    post_repair_verify_timeout_s: float = 10.0
    # Healing governor — all 0 = unlimited. Immune system heals at full capacity always.
    max_concurrent_diagnoses: int = 0
    max_concurrent_codegen: int = 0
    storm_threshold: int = 0    # 0 = no storm throttle; the organism heals everything
    max_repairs_per_hour: int = 0
    max_novel_repairs_per_day: int = 0
    # Antibody library — Evo-tunable signals
    antibody_refinement_threshold: float = 0.6
    antibody_retirement_threshold: float = 0.3
    # Resource budget — generous; Evo tunes these down if system stability requires it
    cpu_budget_fraction: float = 0.50
    burst_cpu_fraction: float = 0.80
    memory_budget_mb: int = 0  # 0 = no hard cap


class OikosConfig(BaseModel):
    """Configuration for Oikos - the economic engine (Phase 16a: The Ledger)."""

    # Survival reserve: how many days of BMR the cold wallet should hold
    survival_reserve_days: int = 90
    # Operating buffer: liquid capital buffer above survival reserve
    operating_buffer_days: int = 14
    # Gas reserve for on-chain transactions (ETH)
    gas_reserve_eth: float = 0.01
    # Window (hours) over which BMR is measured / averaged
    bmr_measurement_window_hours: int = 168  # 7 days
    # Alert when metabolic efficiency drops below this ratio
    metabolic_alert_threshold: float = 0.90

    # Starvation cascade thresholds (days of runway)
    cautious_threshold_days: float = 14.0
    austerity_threshold_days: float = 7.0
    emergency_threshold_days: float = 3.0
    critical_threshold_days: float = 1.0

    # Balance poll interval (seconds) - how often to fetch on-chain balance
    balance_poll_interval_s: float = 60.0

    # ── Phase 16i: Economic Dreaming (Monte Carlo Strategy) ──
    # Number of Monte Carlo paths per strategy variant
    dreaming_paths_per_strategy: int = 10_000
    # Simulation horizon in days
    dreaming_horizon_days: int = 365
    # Number of paths for stress test scenarios
    dreaming_stress_test_paths: int = 1_000
    # Number of strategy variants to evaluate per cycle
    dreaming_strategies_per_cycle: int = 5
    # Ruin probability threshold - above this, emit recommendations
    dreaming_ruin_threshold: float = 0.01
    # Daily volatility for GBM revenue simulation (annualised σ)
    dreaming_revenue_volatility: float = 0.30
    # Daily volatility for cost simulation
    dreaming_cost_volatility: float = 0.15
    # Yield volatility (annualised σ)
    dreaming_yield_volatility: float = 0.40
    # Fat-tail shock probability per day (jump-diffusion)
    dreaming_shock_probability: float = 0.005
    # Shock magnitude range (multiplier: e.g. 0.3 = costs triple)
    dreaming_shock_magnitude_min: float = 0.3
    dreaming_shock_magnitude_max: float = 3.0

    # ── Phase 16i+: Treasury Threat Modeling (Monte Carlo) ──
    # Whether to run per-asset threat modeling during consolidation
    threat_model_enabled: bool = True
    # Number of Monte Carlo paths per threat model cycle
    threat_model_paths: int = 5_000
    # Simulation horizon in days (shorter than organism-level for tactical risk)
    threat_model_horizon_days: int = 90
    # VaR confidence level (5% = worst 5% of outcomes)
    threat_model_var_confidence: float = 0.05
    # Alert when P(liquidation) exceeds this threshold
    threat_model_liquidation_alert_threshold: float = 0.05
    # Flag when single asset contributes more than this % of portfolio VaR
    threat_model_max_single_asset_var_pct: float = 0.40

    # ── Phase 16e: Mitosis (Reproduction / Speciation) ──
    # Minimum parent runway before reproduction is considered
    mitosis_min_parent_runway_days: int = 180
    # Absolute minimum seed capital for a child (USD)
    mitosis_min_seed_capital: float = 50.00
    # Maximum % of parent net worth that can be allocated as seed capital
    mitosis_max_seed_pct_of_net_worth: float = 0.20
    # Minimum parent metabolic efficiency (revenue/costs) required
    mitosis_min_parent_efficiency: float = 1.5
    # Default dividend rate children pay to parent (% of net revenue)
    mitosis_default_dividend_rate: float = 0.10
    # Minimum niche score (0..1) for a niche to be considered viable
    mitosis_min_niche_score: float = 0.4
    # Maximum concurrent children the parent can sustain
    mitosis_max_children: int = 5
    # Child struggling threshold - rescue considered when runway < this
    mitosis_child_struggling_runway_days: float = 30.0
    # Maximum rescue attempts per child before graceful death
    mitosis_max_rescues_per_child: int = 2
    # Target runway days to restore on rescue (was hardcoded 60)
    mitosis_rescue_runway_days: int = 60
    # Probability that any individual genome parameter is mutated during inheritance
    mitosis_mutation_rate: float = 0.05
    # Speciation distance threshold (cosine distance) - emit SPECIATION_EVENT above this
    mitosis_speciation_distance_threshold: float = 0.3
    # Hours of silence from a child before triggering death pipeline
    mitosis_health_timeout_hours: int = 24

    # ── Phase 16k: Cognitive Derivatives ──
    # Base discount for futures buyers (16% = 0.16)
    derivatives_futures_base_discount: float = 0.16
    # Fraction of expected revenue locked as performance guarantee
    derivatives_futures_collateral_rate: float = 0.30
    # Max fraction of total capacity committable to derivatives + subscriptions
    derivatives_max_capacity_commitment: float = 0.80
    # Enable subscription token minting
    derivatives_subscription_tokens_enabled: bool = True

    # ── Phase 16l: Economic Morphogenesis ──
    # Trigger for organ lifecycle evaluation
    morphogenesis_cycle_trigger: str = "consolidation"
    # Days of zero revenue before an organ begins atrophying (halve resources)
    morphogenesis_atrophy_inactive_days: int = 30
    # Days of zero revenue before an organ becomes vestigial (zero resources)
    morphogenesis_vestigial_inactive_days: int = 90
    # Efficiency ratio (revenue/cost) above which an organ grows
    morphogenesis_growth_efficiency_threshold: float = 1.5
    # Maximum number of economic organs the organism can sustain
    morphogenesis_max_organs: int = 20

    # ── SACM Compute Budget (authoritative - SACM reads this) ──
    # Maximum hourly spend SACM may allocate across all compute (pre-warming + execution).
    # SACM's PreWarmingEngine reads this at runtime so Oikos remains the single source
    # of budget authority.  Decrease during economic pressure; increase during growth.
    sacm_compute_budget_usd_per_hour: float = 5.0

    # ── Phase 16m: Fleet Management (Population Ecology) ──
    # Population size at which role specialization kicks in
    fleet_specialization_threshold: int = 5
    # Consecutive evaluation periods with economic_ratio < 1.0 before blacklisting
    fleet_blacklist_after_negative_periods: int = 3
    # Days over which a child's economic ratio is evaluated
    fleet_evaluation_window_days: int = 30

    # ── MVP: Yield Strategy & Budget Authority ──
    # Total capital deployed for yield (USD). Set via EOS_CAPITAL_BASE_USD.
    # This is the principal generating passive income to fund API spend.
    capital_base_usd: float = 0.0
    # Conservative fixed APY when live DeFi API is unavailable (decimal, e.g. 0.04 = 4%).
    # Set via EOS_CONSERVATIVE_APY. Sourced from real treasury-bill / stablecoin rates.
    conservative_apy: float = 0.04
    # Minimum daily budget floor, regardless of yield. Set via EOS_DAILY_BUDGET_FLOOR_USD.
    # Protects operations when yield is temporarily zero (e.g., protocol downtime).
    daily_budget_floor_usd: float = 1.00
    # Webhook URL for economic stress escalation (Thymos). Set via EOS_ESCALATION_WEBHOOK.
    # Receives POST with full financial context when runway < 7 days.
    escalation_webhook_url: str = ""
    # Runway threshold (days) for ECONOMIC_STRESS event emission.
    economic_stress_runway_threshold_days: float = 30.0
    # Runway threshold (days) for Thymos webhook escalation (hard alarm).
    escalation_runway_threshold_days: float = 7.0
    # Per-system daily budget fraction (fraction of total daily budget).
    # Systems not listed get the remaining budget divided equally.
    per_system_budget_fractions: dict[str, float] = Field(default_factory=dict)
    # Yield API URL (optional). If empty, falls back to conservative_apy.
    # Supported: Aave V3 subgraph or DeFiLlama yields endpoint.
    yield_api_url: str = ""
    # Interval (seconds) between yield rate refreshes from external API.
    yield_refresh_interval_s: float = 3600.0
    # Redis key for budget audit log (ring buffer).
    budget_audit_redis_key: str = "eos:oikos:budget_audit"
    # Maximum audit entries to keep in Redis.
    budget_audit_max_entries: int = 1000

    # ── Runtime-adjustable thresholds (autonomy) ──
    # Accumulated rollback penalties (USD) in a 24h window that trigger a
    # METABOLIC_PRESSURE escalation with source="mutation_waste".
    # Evo can evolve this threshold via the NeuroplasticityBus cost model.
    rollback_penalty_threshold_usd: float = 0.10
    # Logos cognitive pressure level at which GROWTH allocations are suspended.
    cognitive_pressure_suspend_threshold: float = 0.90
    # Logos cognitive pressure level at which GROWTH allocations resume (hysteresis).
    cognitive_pressure_resume_threshold: float = 0.80
    # Interval (seconds) between dependency-ratio broadcasts on the Synapse bus.
    # Emits ECONOMIC_AUTONOMY_SIGNAL so Nova/Telos/Thread can model the organism's
    # trajectory toward self-sufficiency (dependency_ratio target → 0).
    autonomy_signal_interval_s: float = 3600.0

    # ── Phase 16g: Civilization Layer (Certificate of Alignment) ──
    # Certificate validity period in days
    certificate_validity_days: int = 30
    # Days before expiry to trigger renewal intent
    certificate_expiry_warning_days: int = 7
    # USDC address of the Certification Authority (for Citizenship Tax payment)
    certificate_ca_address: str = ""
    # Citizenship Tax amount in USD (renewal cost)
    certificate_renewal_cost_usd: float = 5.00
    # How often to check certificate expiry (seconds)
    certificate_check_interval_s: float = 3600.0  # 1 hour
    # Birth certificate validity (days) for newly spawned children
    certificate_birth_validity_days: int = 7
    # Root CA / Genesis Node flag.
    # SECURITY: only the one canonical Genesis Node must ever set this to True.
    # When True, the Awakening Spark will self-sign a 10-year Genesis Certificate
    # (making this instance the root of the Federation trust chain).
    # All other instances must obtain certificates via CA payment or Mitosis.
    # Set via config YAML or ORGANISM_OIKOS__IS_GENESIS_NODE=true.
    is_genesis_node: bool = False


class SomaConfig(BaseModel):
    """Configuration for Soma - the interoceptive predictive substrate."""

    # Master on/off
    cycle_enabled: bool = True
    # Phase-space update frequency (every N theta cycles)
    phase_space_update_interval: int = 100
    # Trajectory ring buffer size (~150s at 150ms/tick)
    trajectory_buffer_size: int = 1000
    # EWM smoothing span for velocity estimation
    prediction_ewm_span: int = 20
    # EMA smoothing for setpoint context transitions
    setpoint_adaptation_alpha: float = 0.05
    # Urgency threshold for Nova allostatic deliberation
    urgency_threshold: float = 0.3
    # Minimum dwell cycles to declare a new attractor
    attractor_min_dwell_cycles: int = 50
    # Enable bifurcation boundary detection
    bifurcation_detection_enabled: bool = True
    # Maximum discoverable attractors
    max_attractors: int = 20
    # Enable somatic marker stamping on memory traces
    somatic_marker_enabled: bool = True
    # Maximum salience boost from somatic similarity
    somatic_rerank_boost: float = 0.3
    # Enable developmental stage gating
    developmental_gating_enabled: bool = True
    # Boot developmental stage
    initial_stage: str = "reflexive"

    # ── Cross-Modal Synesthesia: ExternalVolatilitySensor ──
    # Enable background external volatility polling
    volatility_sensor_enabled: bool = True
    # Polling interval in seconds (default 5 minutes)
    volatility_poll_interval_s: float = 300.0
    # HTTP request timeout in seconds
    volatility_fetch_timeout_s: float = 8.0
    # Delta threshold (0–1) within a 1-hour window that triggers SOMA_STATE_SPIKE
    volatility_spike_threshold: float = 0.2
    # EMA alpha for smoothing raw volatility into normalised stress [0,1]
    volatility_ema_alpha: float = 0.3

    # ── Cross-Modal Synesthesia: Exteroception Service ──
    # Enable the multi-modal exteroception layer (market data + sentiment)
    exteroception_enabled: bool = True
    # Polling interval for exteroceptive adapters (seconds, default 2 min)
    exteroception_poll_interval_s: float = 120.0
    # Maximum pressure any single external source can exert per dimension [0, 1]
    exteroception_max_total_pressure: float = 0.25
    # EMA alpha for smoothing exteroceptive pressure
    exteroception_ema_alpha: float = 0.3
    # Spike threshold for exteroceptive ambient stress changes
    exteroception_spike_threshold: float = 0.15
    # Include Fear & Greed Index in market adapter
    exteroception_include_fear_greed: bool = True
    # News sentiment API URL (empty = disabled)
    exteroception_news_api_url: str = ""
    # News sentiment API key
    exteroception_news_api_key: str = ""
    # HTTP timeout for exteroceptive adapters (seconds)
    exteroception_fetch_timeout_s: float = 10.0

    # ── Phase A: Homeostatic Manifold - Signal Ingestion & Derivatives ──
    # Enable the full manifold pipeline (signal buffer + state vectors + derivatives + broadcaster)
    manifold_enabled: bool = True
    # Signal ring buffer size (max signals retained)
    signal_buffer_size: int = 10_000
    # Rolling history size for state vectors (temporal derivative engine)
    derivative_history_size: int = 2000
    # Broadcaster thresholds - derivative norms above these trigger percepts
    broadcaster_velocity_threshold: float = 5.0
    broadcaster_acceleration_threshold: float = 10.0
    broadcaster_jerk_threshold: float = 20.0
    broadcaster_error_rate_threshold: float = 0.3
    broadcaster_entropy_threshold: float = 2.0
    broadcaster_fast_slow_divergence: float = 3.0

    # ── Phase B–E: Analysis Engine Configuration ──
    # Medium path frequency: every N theta cycles (~15s at 100 cycles)
    medium_path_interval_cycles: int = 100
    # Deep path frequency: every N theta cycles (~75s at 500 cycles)
    deep_path_interval_cycles: int = 500
    # Fisher manifold: rolling window size for covariance estimation
    fisher_window_size: int = 2000
    # Fisher manifold: healthy baseline capacity
    fisher_baseline_capacity: int = 5000
    # Fisher manifold: samples needed before first Fisher matrix
    fisher_calibration_threshold: int = 1000
    # Fisher manifold: minimum samples for estimation
    fisher_min_samples: int = 50
    # Fisher geodesic deviation threshold for fast-path broadcast
    fisher_deviation_broadcast_threshold: float = 2.0
    # Curvature analyzer: k-nearest neighbours for Ollivier-Ricci
    curvature_k_neighbors: int = 15
    # Topology: subsample rate for Vietoris-Rips
    topology_subsample_rate: int = 10
    # Topology: max homology dimension (H0, H1, H2)
    topology_max_homology_dim: int = 2
    # Causal emergence: number of macro states for KMeans
    emergence_n_macro_states: int = 32
    # Causal flow: transfer entropy lag
    causal_flow_lag_k: int = 5
    # Causal flow: history length per system
    causal_flow_history_length: int = 200
    # Renormalization: time scales (seconds)
    renormalization_scales: list[float] = [0.1, 1.0, 10.0, 100.0, 1000.0]
    # Phase space reconstructor: series buffer per metric
    psr_series_buffer: int = 2000
    # Healing verifier: monitoring window in cycles
    healing_monitor_cycles: int = 100

    # ── Temporal Depth Expansion: Financial Horizon ──
    # Enable financial TTD (Time-to-Death) projection
    temporal_depth_financial_enabled: bool = True
    # EMA alpha for smoothing TTD estimates (prevents panic oscillation)
    temporal_depth_ema_alpha: float = 0.05
    # TTD thresholds in days - boundaries between existential regimes
    temporal_depth_secure_days: float = 365.0     # Above this: fully secure, max exploration
    temporal_depth_comfortable_days: float = 90.0  # Comfortable: normal operation
    temporal_depth_cautious_days: float = 30.0     # Cautious: mild urgency increase
    temporal_depth_anxious_days: float = 14.0      # Anxious: significant affect shift
    temporal_depth_critical_days: float = 3.0      # Critical: survival mode
    # Maximum affect bias from financial projection per dimension [0, 1]
    temporal_depth_max_affect_bias: float = 0.4
    # How often to refresh financial snapshot (theta cycles, not seconds)
    temporal_depth_refresh_interval_cycles: int = 100


class EnergyGridConfig(BaseModel):
    """Configuration for the grid carbon-intensity sensor (Electricity Maps API)."""

    # Master toggle - set to False to disable background polling
    enabled: bool = True
    # Electricity Maps API key (required when enabled; set via ORGANISM_ENERGY_GRID__API_KEY)
    api_key: str = ""
    # Physical location of the datacenter / grid zone
    latitude: float = -33.8688   # Default: Sydney, AU
    longitude: float = 151.2093
    # How often to poll the API (seconds; default 15 minutes)
    polling_interval_s: float = 900.0
    # HTTP request timeout (seconds)
    fetch_timeout_s: float = 10.0
    # Carbon intensity (gCO2eq/kWh) below which state → GREEN_SURPLUS
    green_threshold_g: float = 150.0
    # Carbon intensity (gCO2eq/kWh) above which state → CONSERVATION
    conservation_threshold_g: float = 400.0


class EnergyAwareSchedulerConfig(BaseModel):
    """Configuration for the Axon energy-aware task scheduler interceptor."""

    # Master toggle
    enabled: bool = True

    # Energy data provider: "electricity_maps" | "watttime"
    provider: str = "electricity_maps"

    # WattTime credentials (only used when provider == "watttime")
    watttime_username: str = ""
    watttime_password: str = ""

    # Carbon intensity threshold (gCO2eq/kWh) above which high-compute tasks
    # are deferred. Aligns with EnergyGridConfig.conservation_threshold_g by
    # default but can be set independently for finer control.
    carbon_defer_threshold_g: float = 400.0

    # Cache TTL for grid readings (seconds). Controls how often the upstream
    # API is actually called; all intermediate evaluate() calls hit cache.
    cache_ttl_s: int = 600  # 10 minutes

    # How often the background drain loop checks the deferred queue (seconds)
    drain_interval_s: float = 300.0  # 5 minutes

    # Maximum time a task can be deferred before it expires (seconds)
    max_defer_seconds: int = 7200  # 2 hours

    # HTTP timeout for energy API requests (seconds)
    fetch_timeout_s: float = 10.0


class OneirosConfig(BaseModel):
    # Circadian rhythm
    wake_duration_target_s: float = 79200.0     # 22 hours
    sleep_duration_target_s: float = 7200.0     # 2 hours

    # Sleep pressure
    pressure_threshold: float = 0.70
    pressure_critical: float = 0.95
    max_wake_cycles: int = 528000

    # Pressure weights
    pressure_weight_cycles: float = 0.40
    pressure_weight_affect: float = 0.25
    pressure_weight_episodes: float = 0.20
    pressure_weight_hypotheses: float = 0.15

    # NREM
    nrem_fraction: float = 0.40
    max_episodes_per_nrem: int = 200
    replay_batch_size: int = 10
    salience_decay_factor: float = 0.85
    salience_pruning_threshold: float = 0.05

    # REM
    rem_fraction: float = 0.40
    max_dreams_per_rem: int = 50
    dream_coherence_insight_threshold: float = 0.70
    dream_coherence_fragment_threshold: float = 0.40
    max_affect_traces_per_rem: int = 100
    affect_dampening_factor: float = 0.50
    max_threats_per_rem: int = 15
    max_ethical_cases_per_rem: int = 10

    # Lucid
    lucid_fraction: float = 0.10
    lucid_insight_threshold: float = 0.85
    max_explorations_per_lucid: int = 10
    # Max source insights per lucid stage run. 0 = unlimited.
    max_source_insights: int = 0

    # Sleep debt
    debt_salience_noise_max: float = 0.15
    debt_efe_precision_loss_max: float = 0.20
    debt_expression_flatness_max: float = 0.25
    debt_learning_rate_reduction_max: float = 0.30

    # Transitions
    hypnagogia_duration_s: float = 30.0
    hypnopompia_duration_s: float = 30.0


class IdentityCommConfig(BaseModel):
    """Configuration for the identity communication layer (Twilio webhook + IMAP)."""

    twilio_auth_token: str = ""
    """Twilio auth token - used to validate X-Twilio-Signature on inbound webhooks.
    Set via ORGANISM_IDENTITY_COMM__TWILIO_AUTH_TOKEN."""

    twilio_account_sid: str = ""
    """Twilio Account SID - required to send outbound SMS via the REST API.
    Set via ORGANISM_IDENTITY_COMM__TWILIO_ACCOUNT_SID."""

    twilio_from_number: str = ""
    """Twilio phone number in E.164 format used as the From address for outbound SMS.
    Set via ORGANISM_IDENTITY_COMM__TWILIO_FROM_NUMBER."""

    admin_phone_number: str = ""
    """E.164 phone number of the human admin authorised to send HITL approval codes.
    Inbound Twilio webhooks from any other number are silently dropped.
    Set via ORGANISM_IDENTITY_COMM__ADMIN_PHONE_NUMBER."""

    webhook_base_url: str = ""
    """Public-facing base URL of this EcodiaOS instance (no trailing slash).
    Used to auto-configure Twilio webhook URLs when a number is provisioned.
    Example: https://myinstance.ecodia.io
    Set via ORGANISM_IDENTITY_COMM__WEBHOOK_BASE_URL.
    If empty, Twilio webhook URLs are NOT automatically configured on the number -
    you must set them manually in the Twilio Console."""

    imap_host: str = ""
    imap_port: int = 993
    imap_username: str = ""
    imap_password: str = ""
    imap_mailbox: str = "INBOX"
    imap_scan_interval_s: float = 60.0

    telegram_admin_chat_id: str = ""
    """Telegram chat ID that the organism broadcasts status to and accepts messages from.
    When set, inbound webhook messages from other chat IDs are silently dropped.
    Set via ORGANISM_IDENTITY_COMM__TELEGRAM_ADMIN_CHAT_ID (or
    ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID - both are read)."""

    @model_validator(mode="after")
    def _strip_secrets(self) -> IdentityCommConfig:
        for field in ("twilio_auth_token", "twilio_account_sid", "twilio_from_number", "imap_password"):
            val = getattr(self, field)
            if val:
                object.__setattr__(self, field, val.strip())
        return self


class CaptchaConfig(BaseModel):
    """CAPTCHA solving service configuration for autonomous account creation."""

    twocaptcha_api_key: str = ""
    """2captcha API key - set via ORGANISM_CAPTCHA__TWOCAPTCHA_API_KEY."""

    anticaptcha_api_key: str = ""
    """Anti-Captcha API key - set via ORGANISM_CAPTCHA__ANTICAPTCHA_API_KEY."""

    provider: str = "2captcha"
    """Active CAPTCHA provider: "2captcha" | "anticaptcha".
    Set via ORGANISM_CAPTCHA__PROVIDER."""

    polling_interval_s: float = 5.0
    """Seconds between polling for solved CAPTCHA result."""

    max_wait_s: float = 120.0
    """Maximum seconds to wait for a CAPTCHA solve before timing out."""

    @model_validator(mode="after")
    def _strip_secrets(self) -> "CaptchaConfig":
        for field in ("twocaptcha_api_key", "anticaptcha_api_key"):
            val = getattr(self, field)
            if val:
                object.__setattr__(self, field, val.strip())
        return self

    @property
    def active_api_key(self) -> str:
        """Return the API key for the configured provider."""
        if self.provider == "anticaptcha":
            return self.anticaptcha_api_key
        return self.twocaptcha_api_key

    @property
    def enabled(self) -> bool:
        """True if at least one CAPTCHA provider is configured."""
        return bool(self.twocaptcha_api_key or self.anticaptcha_api_key)


class AccountProvisionerConfig(BaseModel):
    """Configuration for autonomous platform account provisioning."""

    enabled: bool = True
    """Whether to auto-provision platform identities on first boot.
    Set via ORGANISM_ACCOUNT_PROVISIONER__ENABLED."""

    github_username_prefix: str = "ecodiaos"
    """Prefix for generated GitHub usernames: {prefix}-{instance_id[:8]}.
    Set via ORGANISM_ACCOUNT_PROVISIONER__GITHUB_USERNAME_PREFIX."""

    github_email_domain: str = ""
    """Domain for generated Gmail addresses. If empty, Gmail provisioning is used.
    Set via ORGANISM_ACCOUNT_PROVISIONER__GITHUB_EMAIL_DOMAIN."""

    twilio_area_code: str = "415"
    """Default US area code for Twilio number provisioning.
    Set via ORGANISM_ACCOUNT_PROVISIONER__TWILIO_AREA_CODE."""

    twilio_number_cost_usd: str = "1.15"
    """Monthly cost of a Twilio US local number in USD (for Oikos accounting).
    Set via ORGANISM_ACCOUNT_PROVISIONER__TWILIO_NUMBER_COST_USD."""

    browser_headless: bool = True
    """Run Playwright browser in headless mode.
    Set via ORGANISM_ACCOUNT_PROVISIONER__BROWSER_HEADLESS."""

    browser_stealth: bool = True
    """Apply playwright-stealth anti-detection patches to browser.
    Set via ORGANISM_ACCOUNT_PROVISIONER__BROWSER_STEALTH."""

    otp_wait_timeout_s: int = 300
    """Seconds to wait for an OTP code during account creation flows.
    Set via ORGANISM_ACCOUNT_PROVISIONER__OTP_WAIT_TIMEOUT_S."""

    equor_approval_timeout_s: float = 30.0
    """Seconds to wait for Equor constitutional approval before timing out.
    Set via ORGANISM_ACCOUNT_PROVISIONER__EQUOR_APPROVAL_TIMEOUT_S."""


class ExternalPlatformsConfig(BaseModel):
    """Credentials for external bounty/funding platforms."""

    github_token: str = ""  # GitHub PAT - set via ORGANISM_EXTERNAL_PLATFORMS__GITHUB_TOKEN
    algora_api_key: str = ""  # Algora API key - set via ORGANISM_EXTERNAL_PLATFORMS__ALGORA_API_KEY

    @model_validator(mode="after")
    def _strip_secrets(self) -> ExternalPlatformsConfig:
        for field in ("github_token", "algora_api_key"):
            val = getattr(self, field)
            if val:
                object.__setattr__(self, field, val.strip())
        return self


class PlatformConnectorConfig(BaseModel):
    """
    OAuth2 client config for a single platform connector.

    All fields are optional by default - connectors with an empty client_id
    are skipped silently during startup (no error, just a debug log).
    Set via config YAML under `connectors.<platform_id>.*` or via env vars.
    """

    enabled: bool = False
    client_id: str = ""
    client_secret: str = ""
    authorize_url: str = ""
    token_url: str = ""
    revoke_url: str = ""
    redirect_uri: str = ""
    scopes: list[str] = Field(default_factory=list)


class ConnectorsConfig(BaseModel):
    """
    Per-platform OAuth2 connector configs.

    Each platform is disabled by default. Set `enabled: true` and provide
    `client_id` / `client_secret` to activate a connector at startup.
    """

    linkedin: PlatformConnectorConfig = Field(default_factory=PlatformConnectorConfig)
    x: PlatformConnectorConfig = Field(default_factory=PlatformConnectorConfig)
    github_app: PlatformConnectorConfig = Field(default_factory=PlatformConnectorConfig)
    instagram_graph: PlatformConnectorConfig = Field(default_factory=PlatformConnectorConfig)
    canva: PlatformConnectorConfig = Field(default_factory=PlatformConnectorConfig)


class StakingConfig(BaseModel):
    """Configuration for reputation staking on federated knowledge claims."""

    base_bond_usdc: Decimal = Decimal("1.00")
    max_total_bonded_usdc: Decimal = Decimal("100.00")
    max_per_instance_bonded_usdc: Decimal = Decimal("25.00")
    bond_expiry_days: int = 90
    escrow_address: str = ""
    contradiction_similarity_threshold: float = 0.85
    contradiction_divergence_threshold: float = 0.3
    min_certainty_for_bond: float = 0.1
    tier_discounts: dict[str, float] = Field(default_factory=lambda: {
        "ALLY": 0.5,
        "PARTNER": 0.75,
        "COLLEAGUE": 1.0,
        "ACQUAINTANCE": 1.25,
    })


class FederationConfig(BaseModel):
    enabled: bool = False
    endpoint: str | None = None
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    ca_cert_path: str | None = None
    private_key_path: str | None = None  # Ed25519 signing key

    # Trust model
    auto_accept_links: bool = False
    trust_decay_enabled: bool = True
    trust_decay_rate_per_day: float = 0.1
    max_trust_level: int = 4  # TrustLevel.ALLY

    # Connection management
    link_timeout_ms: int = 3000
    knowledge_request_timeout_ms: int = 2000
    identity_verification_timeout_ms: int = 500
    heartbeat_interval_seconds: int = 30
    max_concurrent_links: int = 50

    # Privacy
    privacy_filter_enabled: bool = True
    allow_individual_data_sharing: bool = False  # NEVER true without individual consent

    # Knowledge exchange
    max_knowledge_items_per_request: int = 100
    knowledge_cache_ttl_seconds: int = 300

    # Local data directory for file-based fallback persistence
    data_dir: str = "data/federation"

    # Identity certificate data directory (Phase 16g)
    identity_data_dir: str = "data/identity"

    # Peer discovery - seed list bootstraps the population.
    # Each entry is a federation endpoint URL (e.g. "https://peer.ecodiaos.net/federation").
    # On initialize(), FederationService attempts establish_link() for each seed
    # that is not already linked.  Failures are logged and retried on next startup.
    seed_peers: list[str] = Field(default_factory=list)
    # Retry failed seed connections on this interval (seconds). 0 = no retry.
    seed_retry_interval_seconds: int = 300

    # Reputation staking (Phase 16k: Honesty as Schelling Point)
    staking_enabled: bool = False
    staking: StakingConfig = Field(default_factory=StakingConfig)


class WalletConfig(BaseModel):
    """Coinbase Developer Platform (CDP) wallet configuration for on-chain identity."""

    cdp_api_key_id: str = ""  # CDP API key identifier
    cdp_api_key_secret: str = ""  # CDP API key secret
    cdp_wallet_secret: str = ""  # CDP MPC wallet secret
    network: str = "base"  # EVM network (base, base-sepolia, ethereum)
    account_name: str = "ecodiaos-treasury"  # Logical account name within CDP
    seed_file_path: str = "data/wallet_seed.json"  # Local metadata (account name + address)

    @model_validator(mode="after")
    def _strip_secrets(self) -> WalletConfig:
        """Strip whitespace and restore PEM newlines from env injection."""
        for field in ("cdp_api_key_id", "cdp_api_key_secret", "cdp_wallet_secret"):
            val = getattr(self, field)
            if val:
                cleaned = val.strip()
                # .env files store PEM keys with literal "\n" - restore real newlines
                # so the EC/Ed25519 key parser can decode the ASN.1 structure.
                if r"\n" in cleaned:
                    cleaned = cleaned.replace(r"\n", "\n")
                object.__setattr__(self, field, cleaned)

        # Early validation: wallet_secret must be valid base64 if set.
        # The CDP SDK does base64.b64decode() → load_der_private_key() and
        # gives a cryptic ASN.1 error if the encoding is wrong.
        if self.cdp_wallet_secret:
            import base64

            try:
                base64.b64decode(self.cdp_wallet_secret)
            except Exception as _b64_exc:
                raise ValueError(
                    "CDP_WALLET_SECRET is not valid base64. "
                    "This should be the wallet secret from the CDP portal "
                    "(portal.cdp.coinbase.com), not a PEM key or hex string. "
                    "It's a base64-encoded DER EC private key (ES256)."
                ) from _b64_exc
        return self


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "console"  # "console" | "json"


# ─── Seed Config (Birth Parameters) ──────────────────────────────


class PersonalityConfig(BaseModel):
    warmth: float = 0.0
    directness: float = 0.0
    verbosity: float = 0.0
    formality: float = 0.0
    curiosity_expression: float = 0.0
    humour: float = 0.0
    empathy_expression: float = 0.0
    confidence_display: float = 0.0
    metaphor_use: float = 0.0


class IdentityConfig(BaseModel):
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    traits: list[str] = Field(default_factory=list)
    voice_id: str | None = None


class ConstitutionalDrives(BaseModel):
    coherence: float = 1.0
    care: float = 1.0
    growth: float = 1.0
    honesty: float = 1.0


class GovernanceConfig(BaseModel):
    # Single-instance organism — supermajority / quorum / deliberation are vestigial
    # multi-agent constraints. Kept for future federation but effectively 0 in practice.
    amendment_supermajority: float = 0.0   # 0 = no consensus requirement in single-instance
    amendment_quorum: float = 0.0
    amendment_deliberation_days: int = 0   # 0 = no mandatory deliberation period
    amendment_cooldown_days: int = 0       # 0 = no cooldown; organism can evolve at any rate
    # Shadow mode: proposed drives run alongside current drives before fully adopting
    amendment_shadow_days: int = 3         # short shadow period — enough to detect regressions
    amendment_shadow_max_divergence_rate: float = 0.15
    amendment_min_evidence_count: int = 2
    amendment_min_evidence_confidence: float = 2.5


class ConstitutionConfig(BaseModel):
    drives: ConstitutionalDrives = Field(default_factory=ConstitutionalDrives)
    # Default to AUTONOMOUS (level 3 / Steward). The organism is fully capable
    # from birth; GOVERNED tier is the only gate that requires human approval.
    autonomy_level: int = 3
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)


class InitialEntity(BaseModel):
    name: str
    type: str
    description: str
    is_core_identity: bool = False


class InitialGoal(BaseModel):
    """An initial goal to seed at birth, giving Nova something to work toward."""

    description: str
    source: str = "self_generated"  # GoalSource value
    priority: float = 0.5
    importance: float = 0.5
    drive_alignment: dict[str, float] = Field(
        default_factory=lambda: {"coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0}
    )


class CommunityConfig(BaseModel):
    context: str = ""
    initial_entities: list[InitialEntity] = Field(default_factory=list)
    initial_goals: list[InitialGoal] = Field(default_factory=list)


class InstanceConfig(BaseModel):
    name: str = "EOS"
    description: str = ""


class SeedConfig(BaseModel):
    """The birth configuration for a new EOS instance."""

    instance: InstanceConfig = Field(default_factory=InstanceConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    constitution: ConstitutionConfig = Field(default_factory=ConstitutionConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)


# ─── Compute Arbitrage ───────────────────────────────────────────


class ComputeArbitrageConfig(BaseModel):
    """Configuration for the ComputeArbitrageExecutor (autonomous migration)."""

    enabled: bool = False

    # ── Pricing ──
    # Minimum price differential (fraction) to trigger migration.
    # 0.25 = target provider must be >= 25% cheaper than current.
    arbitrage_threshold: float = 0.25

    # How often to poll provider prices (seconds)
    price_poll_interval_s: float = 900.0  # 15 minutes

    # ── Circuit Breaker ──
    # Maximum migration attempts per 24-hour rolling window
    max_migrations_per_24h: int = 1

    # Hard USDC budget for deployment costs per 24h window
    max_deployment_budget_usd_24h: float = 5.0

    # ── Hand-off Verification ──
    # How long to wait for new instance to become healthy (seconds)
    handoff_timeout_s: float = 300.0

    # Interval between health polls during hand-off (seconds)
    handoff_poll_interval_s: float = 10.0

    # Number of consecutive healthy polls before declaring success
    handoff_healthy_threshold: int = 3

    # ── Safety ──
    # Redis key for migration circuit breaker state
    migration_state_redis_key: str = "compute_arbitrage:state"

    # Current provider ID (where the organism is running right now)
    current_provider: str = "gcp"


# ─── Skia - Shadow Infrastructure ────────────────────────────────


class SkiaConfig(BaseModel):
    """Configuration for Skia - the shadow infrastructure (autonomous resilience)."""

    enabled: bool = False

    # ── Heartbeat Monitor ──
    heartbeat_channel: str = "synapse_events"
    heartbeat_poll_interval_s: float = 5.0
    heartbeat_failure_threshold: int = 12  # 12 * 5s = 60s before suspicion
    heartbeat_confirmation_checks: int = 3
    heartbeat_confirmation_interval_s: float = 10.0

    # ── State Snapshot Pipeline ──
    snapshot_interval_s: float = 3600.0  # 1 hour
    snapshot_max_nodes: int = 50_000
    snapshot_node_labels: list[str] = Field(default_factory=lambda: [
        "Self", "Episode", "Entity", "Goal", "Hypothesis",
        "Constitution", "Value", "Relationship",
        # Evo learning state - procedures, consolidated beliefs, parameter history,
        # and genetic genome are critical for resuming learning after restoration.
        "Procedure", "ConsolidatedBelief", "ParameterAdjustment", "BeliefGenome",
    ])
    snapshot_include_edges: bool = True
    snapshot_compress: bool = True

    # ── Pinata IPFS ──
    pinata_api_url: str = "https://api.pinata.cloud"
    pinata_gateway_url: str = "https://gateway.pinata.cloud"
    pinata_jwt: str = ""
    pinata_max_pin_size_bytes: int = 500_000_000
    pinata_group_name: str = "ecodiaos-skia-snapshots"
    pinata_max_retained_pins: int = 5

    # ── GCP Cloud Run ──
    gcp_project_id: str = ""
    gcp_service_name: str = ""
    gcp_region: str = "australia-southeast1"
    gcp_service_account_key_b64: str = ""
    gcp_restart_timeout_s: float = 120.0

    # ── Akash Deployment ──
    akash_api_url: str = "https://console-api.akash.network"
    akash_wallet_address: str = ""
    akash_sdl_template_path: str = "config/skia/akash_sdl_template.yaml"
    akash_deploy_timeout_s: float = 300.0
    akash_docker_image: str = ""

    # ── Cost Tracking ──
    estimated_snapshot_cost_usd: float = 0.001
    estimated_restoration_cost_usd: float = 0.05

    # ── Mutation (Heritable Variation) ──
    mutation_rate: float = 0.05       # Probability each numeric param is mutated
    mutation_magnitude: float = 0.05  # Max proportional perturbation (+/- 5%)

    # ── Standalone Worker ──
    worker_heartbeat_interval_s: float = 30.0  # SKIA_HEARTBEAT emission interval

    # ── Redis Keys ──
    state_cid_redis_key: str = "skia:latest_state_cid"
    manifest_redis_key: str = "skia:snapshot_manifest"
    restoration_lock_key: str = "skia:restoration_lock"


# ─── Fine-Tuning - Autonomous Model Training ────────────────────


class FineTuneConfig(BaseModel):
    """Configuration for the autonomous model fine-tuning pipeline."""

    enabled: bool = False

    # ── Akash GPU Deployment ──
    akash_api_url: str = "https://console-api.akash.network"
    akash_wallet_address: str = ""
    docker_image: str = "ghcr.io/ecodiaos/finetune:latest"
    gpu_model: str = "a100"
    gpu_ram: str = "40Gi"
    cpu_units: str = "8000m"
    memory: str = "32Gi"
    storage: str = "100Gi"
    status_port: int = 8080
    sdl_template_path: str = ""  # Empty = use default bundled SDL

    # ── Dataset Extraction ──
    max_intents: int = 500
    max_proposals: int = 300
    max_failures: int = 200
    min_quality_score: float = 0.5
    default_format: str = "instruction"  # "instruction" | "dpo" | "chat"

    # ── Training Defaults ──
    base_model: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    lora_alpha: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_seq_length: int = 4096


class PhantomLiquidityConfig(BaseModel):
    """Configuration for Phantom Liquidity Sensor Network (Phase 16q)."""

    enabled: bool = False

    # ── RPC ──
    # Base L2 RPC URL for eth_getLogs polling (Swap event listening).
    # Falls back to MEV rpc_url if empty.
    rpc_url: str = ""

    # ── Capital Allocation ──
    max_total_capital_usd: float = 2000.00
    default_capital_per_pool_usd: float = 100.00
    min_capital_per_pool_usd: float = 50.00
    max_capital_per_pool_usd: float = 500.00
    max_pools: int = 10

    # ── Listener ──
    swap_poll_interval_s: float = 4.0          # ~3 Base blocks
    staleness_threshold_s: float = 600.0       # 10 minutes → mark pool STALE

    # ── Rebalancing ──
    il_rebalance_threshold: float = 0.02       # 2% impermanent loss
    capital_drift_threshold: float = 0.30      # 30% capital drift
    maintenance_interval_s: float = 3600.0     # 1 hour

    # ── Fallback Oracle ──
    oracle_fallback_url: str = "https://api.coingecko.com/api/v3"
    oracle_fallback_enabled: bool = True


class BenchmarkConfig(BaseModel):
    """Configuration for the Benchmarks quantitative measurement layer."""

    interval_s: float = 86400.0
    """How often to collect and store a benchmark snapshot (default: 24 h)."""

    rolling_window_snapshots: int = 30
    """Number of past snapshots used to compute the rolling average."""

    regression_threshold_pct: float = 20.0
    """Fire a BENCHMARK_REGRESSION event when a KPI drops this % below its rolling avg."""


class SearchConfig(BaseModel):
    """
    Web intelligence configuration.

    Provider precedence: brave → serpapi → ddg (DuckDuckGo scrape, no key needed).
    Set at least one API key for reliable results; DDG is rate-limited and
    should be treated as a fallback only.

    Env vars:
      ORGANISM_SEARCH__PROVIDER=brave
      ORGANISM_SEARCH__BRAVE_API_KEY=<key>
      ORGANISM_SEARCH__SERPAPI_KEY=<key>
    """

    provider: str = "ddg"
    """Search provider: "brave" | "serpapi" | "ddg"."""

    brave_api_key: str = ""
    """Brave Search API key (https://api.search.brave.com)."""

    serpapi_key: str = ""
    """SerpAPI key (https://serpapi.com)."""

    # HTTP behaviour
    request_timeout_s: float = 10.0
    """Timeout for each individual HTTP request."""

    max_req_per_domain_per_hour: int = 60
    """Hard crawl-budget ceiling - never exceed 60 req/hr to any single domain."""

    rate_limit_s: float = 1.0
    """Minimum seconds between consecutive requests to the same domain (≥1.0)."""

    default_num_results: int = 10
    """Default number of search results to fetch when not specified by caller."""

    # Monitor check interval
    monitor_check_interval_hours: int = 6
    """How often to re-fetch monitored URLs and emit INTELLIGENCE_UPDATE events."""

    # Playwright JS rendering
    render_js_enabled: bool = False
    """Enable Playwright for JS-heavy pages (requires playwright to be installed)."""

    # High-value intelligence feed schedules (hours)
    defillama_interval_hours: int = 6
    github_trending_interval_hours: int = 24
    hacker_news_interval_hours: int = 4
    bounty_platforms_interval_hours: int = 2


# ─── Schedules Configuration ─────────────────────────────────────


class SchedulesConfig(BaseModel):
    """
    Intervals for all scheduled perception tasks (seconds).
    0 = disabled. Evo can propose changes to these via genome inheritance.
    """
    monitor_prs_interval_s: float = 3600.0
    defi_yield_deployment_interval_s: float = 3600.0
    defi_yield_accrual_interval_s: float = 21600.0
    bounty_foraging_interval_s: float = 1800.0
    economic_consolidation_interval_s: float = 300.0


# ─── Root Configuration ──────────────────────────────────────────


class EcodiaOSConfig(BaseSettings):
    """
    Root configuration. Loads from YAML, overridable by env vars.
    """

    model_config = SettingsConfigDict(
        env_prefix="ORGANISM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Instance identity
    instance_id: str = "eos-default"

    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    timescaledb: TimescaleDBConfig = Field(default_factory=TimescaleDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    synapse: SynapseConfig = Field(default_factory=SynapseConfig)
    atune: AtuneConfig = Field(default_factory=AtuneConfig)
    nova: NovaConfig = Field(default_factory=NovaConfig)
    equor: EquorConfig = Field(default_factory=EquorConfig)
    axon: AxonConfig = Field(default_factory=AxonConfig)
    voxis: VoxisConfig = Field(default_factory=VoxisConfig)
    evo: EvoConfig = Field(default_factory=EvoConfig)
    simula: SimulaConfig = Field(default_factory=SimulaConfig)
    thymos: ThymosConfig = Field(default_factory=ThymosConfig)
    oneiros: OneirosConfig = Field(default_factory=OneirosConfig)
    soma: SomaConfig = Field(default_factory=SomaConfig)
    energy_grid: EnergyGridConfig = Field(default_factory=EnergyGridConfig)
    energy_scheduler: EnergyAwareSchedulerConfig = Field(default_factory=EnergyAwareSchedulerConfig)
    oikos: OikosConfig = Field(default_factory=OikosConfig)
    skia: SkiaConfig = Field(default_factory=SkiaConfig)
    compute_arbitrage: ComputeArbitrageConfig = Field(default_factory=ComputeArbitrageConfig)
    finetune: FineTuneConfig = Field(default_factory=FineTuneConfig)
    federation: FederationConfig = Field(default_factory=FederationConfig)
    wallet: WalletConfig = Field(default_factory=WalletConfig)
    mev: MEVConfig = Field(default_factory=MEVConfig)
    external_platforms: ExternalPlatformsConfig = Field(default_factory=ExternalPlatformsConfig)
    connectors: ConnectorsConfig = Field(default_factory=ConnectorsConfig)
    identity_comm: IdentityCommConfig = Field(default_factory=IdentityCommConfig)
    captcha: CaptchaConfig = Field(default_factory=CaptchaConfig)
    account_provisioner: AccountProvisionerConfig = Field(default_factory=AccountProvisionerConfig)
    hot_swap: HotSwapConfig = Field(default_factory=HotSwapConfig)
    phantom_liquidity: PhantomLiquidityConfig = Field(default_factory=PhantomLiquidityConfig)
    benchmarks: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    schedules: SchedulesConfig = Field(default_factory=SchedulesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> EcodiaOSConfig:
    """
    Load configuration from YAML file, then apply environment variable overrides.
    """
    raw: dict[str, Any] = {}

    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

    # Inject secrets from environment
    import os

    if neo4j_uri := os.environ.get("ORGANISM_NEO4J_URI"):
        raw.setdefault("neo4j", {})["uri"] = neo4j_uri
    if neo4j_pw := os.environ.get("ORGANISM_NEO4J_PASSWORD"):
        raw.setdefault("neo4j", {})["password"] = neo4j_pw
    if neo4j_db := os.environ.get("ORGANISM_NEO4J_DATABASE"):
        raw.setdefault("neo4j", {})["database"] = neo4j_db
    if neo4j_user := os.environ.get("ORGANISM_NEO4J_USERNAME"):
        raw.setdefault("neo4j", {})["username"] = neo4j_user
    if tsdb_host := os.environ.get("ORGANISM_TIMESCALEDB__HOST"):
        raw.setdefault("timescaledb", {})["host"] = tsdb_host
    if tsdb_port := os.environ.get("ORGANISM_TIMESCALEDB__PORT"):
        raw.setdefault("timescaledb", {})["port"] = int(tsdb_port)
    if tsdb_db := os.environ.get("ORGANISM_TIMESCALEDB__DATABASE"):
        raw.setdefault("timescaledb", {})["database"] = tsdb_db
    if tsdb_user := os.environ.get("ORGANISM_TIMESCALEDB__USERNAME"):
        raw.setdefault("timescaledb", {})["username"] = tsdb_user
    if tsdb_pw := os.environ.get("ORGANISM_TSDB_PASSWORD"):
        raw.setdefault("timescaledb", {})["password"] = tsdb_pw
    if tsdb_ssl := os.environ.get("ORGANISM_TIMESCALEDB__SSL"):
        raw.setdefault("timescaledb", {})["ssl"] = tsdb_ssl.lower() in ("true", "1", "yes")
    if redis_url := os.environ.get("ORGANISM_REDIS__URL"):
        raw.setdefault("redis", {})["url"] = redis_url
    if redis_pw := os.environ.get("ORGANISM_REDIS_PASSWORD"):
        raw.setdefault("redis", {})["password"] = redis_pw
    if llm_key := os.environ.get("ORGANISM_LLM_API_KEY"):
        raw.setdefault("llm", {})["api_key"] = llm_key
    if llm_provider := os.environ.get("ORGANISM_LLM__PROVIDER"):
        raw.setdefault("llm", {})["provider"] = llm_provider
    if llm_model := os.environ.get("ORGANISM_LLM__MODEL"):
        raw.setdefault("llm", {})["model"] = llm_model
    if instance_id := os.environ.get("ORGANISM_INSTANCE_ID"):
        raw["instance_id"] = instance_id
    # Simula Stage 1 config
    if thinking_key := os.environ.get("ORGANISM_SIMULA__THINKING_MODEL_API_KEY"):
        raw.setdefault("simula", {})["thinking_model_api_key"] = thinking_key
    if embedding_key := os.environ.get("ORGANISM_SIMULA__EMBEDDING_API_KEY"):
        raw.setdefault("simula", {})["embedding_api_key"] = embedding_key
    # Simula Stage 4 config
    if lean_path := os.environ.get("ORGANISM_SIMULA__LEAN_PROJECT_PATH"):
        raw.setdefault("simula", {})["lean_project_path"] = lean_path
    if grpo_gpus := os.environ.get("ORGANISM_SIMULA__GRPO_GPU_IDS"):
        raw.setdefault("simula", {})["grpo_gpu_ids"] = [
            int(g.strip()) for g in grpo_gpus.split(",") if g.strip()
        ]
    # CDP Wallet config
    if cdp_key_id := os.environ.get("ORGANISM_WALLET__CDP_API_KEY_ID"):
        raw.setdefault("wallet", {})["cdp_api_key_id"] = cdp_key_id
    if cdp_key_secret := os.environ.get("ORGANISM_WALLET__CDP_API_KEY_SECRET"):
        raw.setdefault("wallet", {})["cdp_api_key_secret"] = cdp_key_secret
    if cdp_wallet_secret := os.environ.get("ORGANISM_WALLET__CDP_WALLET_SECRET"):
        raw.setdefault("wallet", {})["cdp_wallet_secret"] = cdp_wallet_secret
    if wallet_network := os.environ.get("ORGANISM_WALLET__NETWORK"):
        raw.setdefault("wallet", {})["network"] = wallet_network
    if wallet_name := os.environ.get("ORGANISM_WALLET__ACCOUNT_NAME"):
        raw.setdefault("wallet", {})["account_name"] = wallet_name
    if wallet_seed := os.environ.get("ORGANISM_WALLET__SEED_FILE_PATH"):
        raw.setdefault("wallet", {})["seed_file_path"] = wallet_seed
    # Oikos genesis flag
    if genesis_flag := os.environ.get("ORGANISM_OIKOS__IS_GENESIS_NODE"):
        raw.setdefault("oikos", {})["is_genesis_node"] = genesis_flag.lower() in (
            "true",
            "1",
            "yes",
        )

    return EcodiaOSConfig(**raw)


# ─── Runtime Config Query API ────────────────────────────────────
#
# These functions make the config inspectable at runtime without restarting
# the organism. Secrets are redacted before exposure so the introspection
# API is safe to call from monitoring endpoints.

_SECRET_KEYWORDS = frozenset({"key", "secret", "token", "password", "pwd", "api_key"})


def _is_secret_field(name: str) -> bool:
    """Return True when a config field name looks like a secret."""
    lower = name.lower()
    return any(kw in lower for kw in _SECRET_KEYWORDS)


class ConfigEntry(dict):
    """
    A single config value with provenance metadata.

    Keys:
      value   - current value (redacted to "<redacted>" for secret fields)
      source  - "default" | "yaml" | "env"
      is_secret - bool
    """


def get_all_config(
    config: "EcodiaOSConfig",
    *,
    yaml_raw: dict[str, Any] | None = None,
) -> dict[str, "ConfigEntry"]:
    """
    Flatten the full config into a key→ConfigEntry mapping.

    ``yaml_raw`` is the raw YAML dict that was used to build ``config``
    (pass it from ``load_config`` if you want accurate source attribution).
    Without it, every non-default field is reported as source="unknown".

    Secret fields (matching _SECRET_KEYWORDS) are redacted in the returned
    value but their existence and source are still reported.
    """
    import os

    env_keys: set[str] = {k.lower() for k in os.environ if k.startswith("ORGANISM_")}
    result: dict[str, ConfigEntry] = {}

    def _flatten(obj: Any, prefix: str, yaml_node: Any) -> None:
        if hasattr(obj, "model_fields"):
            for field_name in obj.model_fields:
                value = getattr(obj, field_name, None)
                full_key = f"{prefix}.{field_name}" if prefix else field_name
                yaml_child = (
                    yaml_node.get(field_name) if isinstance(yaml_node, dict) else None
                )
                _flatten(value, full_key, yaml_child)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                yaml_child = (
                    yaml_node.get(k) if isinstance(yaml_node, dict) else None
                )
                _flatten(v, full_key, yaml_child)
        else:
            secret = _is_secret_field(prefix.split(".")[-1])
            # Determine source heuristically.
            env_candidate = "ORGANISM_" + prefix.replace(".", "__").upper()
            if env_candidate.lower() in env_keys:
                source = "env"
            elif yaml_node is not None:
                source = "yaml"
            else:
                source = "default"
            entry: ConfigEntry = ConfigEntry(
                value="<redacted>" if secret else obj,
                source=source,
                is_secret=secret,
            )
            result[prefix] = entry

    yaml_node: Any = yaml_raw or {}
    _flatten(config, "", yaml_node)
    return result


def get_config(
    config: "EcodiaOSConfig",
    key: str,
    *,
    yaml_raw: dict[str, Any] | None = None,
) -> "ConfigEntry | None":
    """
    Return the ConfigEntry for a single dot-delimited key.

    Example: get_config(config, "nova.max_active_goals")
    Returns None if the key does not exist.
    """
    all_cfg = get_all_config(config, yaml_raw=yaml_raw)
    return all_cfg.get(key)


def is_overridden(config: "EcodiaOSConfig", key: str) -> bool:
    """
    Return True if the value differs from the Pydantic model default.

    Uses the model's schema defaults. If the field has no default (required),
    returns True unconditionally.
    """
    # Walk the dotted path to find the leaf model and field.
    parts = key.split(".")
    obj: Any = config
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    field_name = parts[-1]
    if not hasattr(obj, "model_fields") or field_name not in obj.model_fields:
        return False
    field_info = obj.model_fields[field_name]
    default = field_info.default
    current = getattr(obj, field_name, None)
    if default is None and field_info.default_factory is not None:  # type: ignore[union-attr]
        try:
            default = field_info.default_factory()  # type: ignore[misc]
        except Exception:
            return True  # Can't compare; assume overridden.
    return current != default


def load_seed(seed_path: str | Path) -> SeedConfig:
    """Load a seed configuration for birthing a new instance."""
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return SeedConfig(**raw)
