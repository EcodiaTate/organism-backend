"""
EcodiaOS -- Simula Verification Types (Stages 2 + 3 + 4 + 5B + 6)

Pydantic models for the formal verification core:
  - Stage 2A: Dafny proof-carrying code (Clover pattern)
  - Stage 2B: LLM + Z3 invariant discovery
  - Stage 2C: Static analysis gates (Bandit / Semgrep)
  - Stage 2D: AgentCoder pattern (test/code separation)
  - Stage 3A: Salsa incremental verification (dependency-aware memoization)
  - Stage 3B: SWE-grep agentic retrieval
  - Stage 3C: LILO library learning
  - Stage 4A: Lean 4 proof generation (DeepSeek-Prover-V2 pattern)
  - Stage 4B: GRPO domain fine-tuning (self-improvement via execution feedback)
  - Stage 4C: Diffusion-based code repair (last-mile denoising)
  - Stage 5B: Neural program repair (SRepair pattern - separate diagnosis from generation)
  - Stage 6A: Cryptographic auditability (hash chains, C2PA, verifiable credentials)
  - Stage 6B: Co-evolving agents (hard negatives, adversarial testing)
  - Stage 6C: Formal spec generation (Dafny, TLA+, Alloy, Self-Spec DSL)
  - Stage 6D: Equality saturation (e-graphs for semantic equivalence)
  - Stage 6E: Hybrid symbolic execution (Z3 SMT for mission-critical logic)

All types use EOSBaseModel for consistency with the rest of Simula.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now
from systems.simula.evolution_types import ChangeCategory

# ── Stage 2A: Dafny Proof-Carrying Code ──────────────────────────────────────


class DafnyVerificationStatus(enum.StrEnum):
    """Outcome of a Dafny verification attempt."""

    VERIFIED = "verified"
    FAILED = "failed"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    SKIPPED = "skipped"


class CloverRoundResult(EOSBaseModel):
    """Result of one round in the Clover iteration loop."""

    round_number: int
    spec_generated: str = ""
    implementation_generated: str = ""
    dafny_stdout: str = ""
    dafny_stderr: str = ""
    dafny_exit_code: int = -1
    verified: bool = False
    errors: list[str] = Field(default_factory=list)
    llm_tokens_used: int = 0


class DafnyVerificationResult(EOSBaseModel):
    """
    Aggregate result of the Clover-pattern Dafny verification.

    The Clover loop iterates: LLM generates Dafny spec + implementation,
    Dafny verifies, errors fed back, until verified or max rounds reached.
    """

    status: DafnyVerificationStatus = DafnyVerificationStatus.SKIPPED
    rounds_attempted: int = 0
    rounds_max: int = 8
    final_spec: str = ""
    final_implementation: str = ""
    round_history: list[CloverRoundResult] = Field(default_factory=list)
    proof_obligations: list[str] = Field(default_factory=list)
    total_llm_tokens: int = 0
    total_dafny_time_ms: int = 0
    verification_time_ms: int = 0
    error_summary: str = ""
    verified_at: datetime = Field(default_factory=utc_now)


# ── Stage 2B: LLM + Z3 Invariant Discovery ──────────────────────────────────


class InvariantKind(enum.StrEnum):
    """Classification of discovered invariants."""

    LOOP_INVARIANT = "loop_invariant"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    RANGE_BOUND = "range_bound"
    MONOTONICITY = "monotonicity"
    RELATIONSHIP = "relationship"


class InvariantVerificationStatus(enum.StrEnum):
    """Outcome of Z3 checking a candidate invariant."""

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


class DiscoveredInvariant(EOSBaseModel):
    """A single invariant discovered by the LLM + Z3 loop."""

    kind: InvariantKind
    expression: str
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)
    target_function: str = ""
    target_file: str = ""
    status: InvariantVerificationStatus = InvariantVerificationStatus.UNKNOWN
    counterexample: str = ""
    confidence: float = 0.0


class Z3RoundResult(EOSBaseModel):
    """Result of one round in the Z3 invariant discovery loop."""

    round_number: int
    candidate_invariants: list[DiscoveredInvariant] = Field(default_factory=list)
    valid_count: int = 0
    invalid_count: int = 0
    unknown_count: int = 0
    counterexamples_fed_back: list[str] = Field(default_factory=list)
    llm_tokens_used: int = 0
    z3_time_ms: int = 0


class InvariantVerificationResult(EOSBaseModel):
    """
    Aggregate result of the LLM + Z3 invariant discovery.

    The discovery loop: LLM generates candidate invariants,
    Z3 checks them, counterexamples fed back, iterate.
    """

    status: InvariantVerificationStatus = InvariantVerificationStatus.SKIPPED
    rounds_attempted: int = 0
    rounds_max: int = 6
    discovered_invariants: list[DiscoveredInvariant] = Field(default_factory=list)
    valid_invariants: list[DiscoveredInvariant] = Field(default_factory=list)
    round_history: list[Z3RoundResult] = Field(default_factory=list)
    total_llm_tokens: int = 0
    total_z3_time_ms: int = 0
    verification_time_ms: int = 0
    error_summary: str = ""
    verified_at: datetime = Field(default_factory=utc_now)


# ── Stage 2C: Static Analysis Gates ─────────────────────────────────────────


class StaticAnalysisSeverity(enum.StrEnum):
    """Severity level for static analysis findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class StaticAnalysisFinding(EOSBaseModel):
    """A single finding from static analysis (Bandit / Semgrep)."""

    tool: str
    rule_id: str = ""
    severity: StaticAnalysisSeverity = StaticAnalysisSeverity.INFO
    file_path: str = ""
    line: int = 0
    column: int = 0
    message: str = ""
    fixable: bool = False
    cwe: str = ""


class StaticAnalysisResult(EOSBaseModel):
    """Aggregate static analysis result across all tools."""

    findings: list[StaticAnalysisFinding] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    fixable_count: int = 0
    tools_run: list[str] = Field(default_factory=list)
    fix_rate: float = 0.0
    analysis_time_ms: int = 0


# ── Stage 2D: AgentCoder Pattern ────────────────────────────────────────────


class TestDesignResult(EOSBaseModel):
    """Output from the TestDesigner agent."""

    test_files: dict[str, str] = Field(default_factory=dict)
    test_count: int = 0
    coverage_targets: list[str] = Field(default_factory=list)
    design_reasoning: str = ""
    llm_tokens_used: int = 0


class TestExecutionResult(EOSBaseModel):
    """Structured output from the TestExecutor agent."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    coverage_percent: float = 0.0
    failure_details: list[str] = Field(default_factory=list)
    raw_output: str = ""
    execution_time_ms: int = 0


class AgentCoderIterationResult(EOSBaseModel):
    """Result of one iteration in the 3-agent AgentCoder pipeline."""

    iteration: int
    test_design: TestDesignResult | None = None
    code_generation_success: bool = False
    code_generation_files: list[str] = Field(default_factory=list)
    test_execution: TestExecutionResult | None = None
    all_tests_passed: bool = False


class AgentCoderResult(EOSBaseModel):
    """Aggregate result of the AgentCoder pipeline."""

    iterations: list[AgentCoderIterationResult] = Field(default_factory=list)
    total_iterations: int = 0
    final_pass_rate: float = 0.0
    converged: bool = False
    total_llm_tokens: int = 0
    total_time_ms: int = 0


# ── Combined Formal Verification Result ─────────────────────────────────────


class FormalVerificationResult(EOSBaseModel):
    """
    Combined result of all formal verification stages.

    Attached to HealthCheckResult and used by SimulaService
    for pass/fail decision-making.

    Dafny is blocking for triggerable categories.
    Z3 is advisory by default (graduates to blocking in Stage 3).
    Static analysis is blocking for ERROR-severity findings.
    """

    dafny: DafnyVerificationResult | None = None
    z3: InvariantVerificationResult | None = None
    static_analysis: StaticAnalysisResult | None = None
    passed: bool = True
    blocking_issues: list[str] = Field(default_factory=list)
    advisory_issues: list[str] = Field(default_factory=list)
    total_verification_time_ms: int = 0


# ── Constants ────────────────────────────────────────────────────────────────


DAFNY_TRIGGERABLE_CATEGORIES: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.MODIFY_CONTRACT,
    ChangeCategory.ADD_SYSTEM_CAPABILITY,
})


# ── Stage 3A: Salsa Incremental Verification ─────────────────────────────────


class VerificationCacheStatus(enum.StrEnum):
    """Status of a cached verification result."""

    HIT = "hit"
    MISS = "miss"
    STALE = "stale"
    INVALIDATED = "invalidated"


class VerificationCacheTier(enum.StrEnum):
    """Which cache layer holds the result."""

    HOT = "hot"       # Redis - fast, ephemeral
    COLD = "cold"     # Neo4j - durable, slower
    NONE = "none"     # Not cached


class FunctionSignature(EOSBaseModel):
    """
    Unique identity of a function for incremental verification.
    The content_hash enables early cutoff: if the hash hasn't changed,
    skip all downstream re-verification.
    """

    file_path: str
    function_name: str
    content_hash: str  # SHA-256 of function source
    start_line: int = 0
    end_line: int = 0
    imports: list[str] = Field(default_factory=list)  # direct dependencies
    importers: list[str] = Field(default_factory=list)  # reverse dependencies


class CachedVerificationResult(EOSBaseModel):
    """
    A verification result stored in the incremental cache.
    Keyed by (file_path, function_name, content_hash).
    """

    signature: FunctionSignature
    formal_verification: FormalVerificationResult | None = None
    test_passed: bool = True
    static_analysis_clean: bool = True
    cached_at: datetime = Field(default_factory=utc_now)
    ttl_seconds: int = 3600  # 1 hour default, overridable
    version_id: int = 0  # MVCC version for concurrent proposals


class IncrementalVerificationResult(EOSBaseModel):
    """
    Aggregate result of incremental verification for a proposal.
    Tracks what was re-verified vs skipped via early cutoff.
    """

    functions_checked: int = 0
    functions_skipped_early_cutoff: int = 0
    functions_cache_hit: int = 0
    functions_re_verified: int = 0
    cache_hit_rate: float = 0.0
    total_time_ms: int = 0
    # Per-function detail
    results: list[CachedVerificationResult] = Field(default_factory=list)
    invalidated_functions: list[str] = Field(default_factory=list)
    # MVCC metadata
    proposal_version: int = 0
    concurrent_proposals: int = 0


# ── Stage 3B: SWE-grep Agentic Retrieval ──────────────────────────────────────


class RetrievalToolKind(enum.StrEnum):
    """Tools available to the SWE-grep retrieval agent."""

    GREP = "grep"
    GLOB = "glob"
    READ_FILE = "read_file"
    AST_QUERY = "ast_query"


class RetrievalHop(EOSBaseModel):
    """One hop in the multi-hop retrieval trace."""

    hop_number: int
    tool_used: RetrievalToolKind
    query: str
    files_found: list[str] = Field(default_factory=list)
    snippets_collected: int = 0
    tokens_used: int = 0
    latency_ms: int = 0


class RetrievedContext(EOSBaseModel):
    """A single piece of retrieved context (file snippet or API doc)."""

    source: str  # file path or doc URL
    content: str
    relevance_score: float = 0.0
    context_type: str = "code"  # "code" | "api_doc" | "spec" | "test"
    start_line: int = 0
    end_line: int = 0


class SweGrepResult(EOSBaseModel):
    """
    Aggregate result of SWE-grep agentic retrieval.
    Multi-hop: 4 serial turns × 8 parallel tools per turn.
    """

    contexts: list[RetrievedContext] = Field(default_factory=list)
    hops: list[RetrievalHop] = Field(default_factory=list)
    total_hops: int = 0
    total_files_searched: int = 0
    total_snippets: int = 0
    total_tokens: int = 0
    total_time_ms: int = 0
    # Comparison with embedding-based search
    precision_vs_embedding: float | None = None


# ── Stage 3C: LILO Library Learning ───────────────────────────────────────────


class AbstractionKind(enum.StrEnum):
    """Classification of discovered code abstractions."""

    UTILITY_FUNCTION = "utility_function"
    PATTERN_TEMPLATE = "pattern_template"
    ERROR_HANDLER = "error_handler"
    VALIDATION_GUARD = "validation_guard"
    DATA_TRANSFORM = "data_transform"
    INTEGRATION_ADAPTER = "integration_adapter"


class LibraryAbstraction(EOSBaseModel):
    """
    A reusable code abstraction extracted from successful evolution proposals.
    Stored in Neo4j as :LibraryAbstraction nodes linked to :EvolutionRecord.

    The LILO loop:
      1. LLM generates code for proposals
      2. Stitch-like extraction identifies common lambda-abstractions
      3. AutoDoc-style naming gives them meaningful identifiers
      4. Stored in the library for reuse in future proposals
    """

    name: str  # human-readable name (e.g., "safe_dict_merge")
    kind: AbstractionKind
    description: str  # one-line what it does
    signature: str  # function signature (e.g., "def safe_dict_merge(a: dict, b: dict) -> dict")
    source_code: str  # full implementation
    source_proposal_ids: list[str] = Field(default_factory=list)
    usage_count: int = 0
    confidence: float = 0.0  # how often it's been successfully reused
    tags: list[str] = Field(default_factory=list)  # e.g., ["error_handling", "dict"]
    created_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime | None = None


class AbstractionExtractionResult(EOSBaseModel):
    """Result of extracting abstractions from a set of proposals."""

    extracted: list[LibraryAbstraction] = Field(default_factory=list)
    merged_into_existing: int = 0
    pruned: int = 0
    total_proposals_analyzed: int = 0
    total_time_ms: int = 0


class LibraryStats(EOSBaseModel):
    """Statistics about the abstraction library."""

    total_abstractions: int = 0
    by_kind: dict[str, int] = Field(default_factory=dict)
    total_usage_count: int = 0
    mean_confidence: float = 0.0
    last_consolidated: datetime | None = None


# ── Stage 4A: Lean 4 Proof Generation ──────────────────────────────────────


class LeanProofStatus(enum.StrEnum):
    """Outcome of a Lean 4 proof attempt."""

    PROVED = "proved"
    FAILED = "failed"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    SKIPPED = "skipped"
    PARTIAL = "partial"  # some subgoals proved, others pending


class LeanTacticKind(enum.StrEnum):
    """Classification of Lean 4 tactics used in proofs."""

    SIMP = "simp"           # simplification lemmas
    OMEGA = "omega"         # linear arithmetic
    DECIDE = "decide"       # decidable propositions
    AESOP = "aesop"         # automated reasoning
    LINARITH = "linarith"   # linear arithmetic reasoning
    RING = "ring"           # ring normalization
    NORM_NUM = "norm_num"   # numeric normalization
    EXACT = "exact"         # exact term construction
    APPLY = "apply"         # apply a theorem
    INTRO = "intro"         # introduce hypotheses
    CASES = "cases"         # case analysis
    INDUCTION = "induction" # structural induction
    CUSTOM = "custom"       # LLM-generated tactic


class LeanSubgoal(EOSBaseModel):
    """A single subgoal in a Lean 4 proof decomposition."""

    index: int
    description: str = ""
    lean_statement: str = ""
    tactic_used: LeanTacticKind = LeanTacticKind.CUSTOM
    tactic_code: str = ""
    proved: bool = False
    error: str = ""
    copilot_automated: bool = False  # True if Lean Copilot solved it


class LeanProofAttempt(EOSBaseModel):
    """Result of one proof attempt in the DeepSeek-Prover-V2 loop."""

    attempt_number: int
    skeleton_code: str = ""
    subgoals: list[LeanSubgoal] = Field(default_factory=list)
    subgoals_proved: int = 0
    subgoals_total: int = 0
    lean_stdout: str = ""
    lean_stderr: str = ""
    lean_exit_code: int = -1
    fully_proved: bool = False
    errors: list[str] = Field(default_factory=list)
    llm_tokens_used: int = 0
    copilot_steps: int = 0  # number of steps automated by Lean Copilot


class ProvenLemma(EOSBaseModel):
    """
    A proven Lean lemma stored in the proof library.
    Reusable across proposals - linked to :EvolutionRecord in Neo4j.
    """

    name: str  # Lean lemma name (e.g., "risk_score_bounded")
    statement: str  # Lean theorem statement
    proof: str  # Full Lean proof code
    domain: str = ""  # "risk_scoring" | "governance_gating" | "budget" | "alignment"
    target_function: str = ""  # Python function this proves properties about
    dependencies: list[str] = Field(default_factory=list)  # other lemma names used
    source_proposal_id: str = ""
    proved_at: datetime = Field(default_factory=utc_now)
    reuse_count: int = 0


class LeanVerificationResult(EOSBaseModel):
    """
    Aggregate result of Lean 4 proof generation.

    DeepSeek-Prover-V2 pattern:
      1. LLM generates proof skeleton with subgoal decomposition
      2. Each subgoal filled via tactic-level proof search
      3. Lean Copilot automates up to 74.2% of tactic steps
      4. LeanDojo provides proof search and retrieval from Mathlib

    Target: risk scoring, governance gating, constitutional alignment
    all get machine-checked Lean proofs.
    """

    status: LeanProofStatus = LeanProofStatus.SKIPPED
    attempts: list[LeanProofAttempt] = Field(default_factory=list)
    max_attempts: int = 5
    final_proof: str = ""
    final_statement: str = ""
    proven_lemmas: list[ProvenLemma] = Field(default_factory=list)
    library_lemmas_used: list[str] = Field(default_factory=list)  # reused from proof library
    total_subgoals: int = 0
    subgoals_proved: int = 0
    copilot_automation_rate: float = 0.0  # fraction solved by Lean Copilot
    total_llm_tokens: int = 0
    total_lean_time_ms: int = 0
    verification_time_ms: int = 0
    error_summary: str = ""
    proved_at: datetime = Field(default_factory=utc_now)


class ProofLibraryStats(EOSBaseModel):
    """Statistics about the Lean proof library."""

    total_lemmas: int = 0
    by_domain: dict[str, int] = Field(default_factory=dict)
    total_reuse_count: int = 0
    mean_copilot_automation: float = 0.0
    last_updated: datetime | None = None


# ── Stage 4B: GRPO Domain Fine-Tuning ──────────────────────────────────────


class GRPOTrainingStatus(enum.StrEnum):
    """Status of a GRPO training run."""

    PENDING = "pending"
    COLLECTING = "collecting"     # collecting training data
    SFT_RUNNING = "sft_running"   # cold-start supervised fine-tuning
    GRPO_RUNNING = "grpo_running" # RL fine-tuning
    EVALUATING = "evaluating"     # A/B evaluation
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingExample(EOSBaseModel):
    """
    One training example for GRPO: a code agent session with binary outcome.
    Collected from Neo4j evolution history.
    """

    proposal_id: str
    category: str = ""
    change_spec_text: str = ""
    system_prompt: str = ""
    code_output: str = ""
    files_written: list[str] = Field(default_factory=list)
    # Binary correctness signal from Simula's own pipeline
    tests_passed: bool = False
    lint_passed: bool = False
    formal_verification_passed: bool = False
    health_check_passed: bool = False
    rolled_back: bool = False
    # Composite reward: 1.0 = all passed + no rollback, 0.0 = failed
    reward: float = 0.0


class GRPORollout(EOSBaseModel):
    """
    One rollout in the GRPO contrastive pair.
    2-rollout contrastive matches 16-rollout performance (per 2-GRPO finding).
    """

    rollout_index: int  # 0 or 1
    code_output: str = ""
    tests_passed: bool = False
    formal_verification_passed: bool = False
    reward: float = 0.0
    tokens_generated: int = 0


class GRPOTrainingBatch(EOSBaseModel):
    """A batch of contrastive rollout pairs for GRPO training."""

    batch_id: str = ""
    examples: list[TrainingExample] = Field(default_factory=list)
    rollout_pairs: list[tuple[GRPORollout, GRPORollout]] = Field(default_factory=list)
    mean_reward_positive: float = 0.0
    mean_reward_negative: float = 0.0
    contrastive_gap: float = 0.0  # positive - negative reward
    total_tokens: int = 0


class GRPOEvaluationResult(EOSBaseModel):
    """A/B evaluation: fine-tuned vs base model."""

    base_model_pass_at_1: float = 0.0
    finetuned_model_pass_at_1: float = 0.0
    improvement_percent: float = 0.0
    test_proposals_count: int = 0
    base_model_mean_reward: float = 0.0
    finetuned_model_mean_reward: float = 0.0
    statistically_significant: bool = False
    evaluated_at: datetime = Field(default_factory=utc_now)


class GRPOTrainingRun(EOSBaseModel):
    """
    Aggregate result of a GRPO training run.

    Pipeline:
      1. Collect training data from Neo4j evolution history
      2. Cold-start SFT on successful code agent outputs
      3. GRPO RL loop: 2-rollout contrastive pairs
      4. A/B deploy: fine-tuned vs base, measure pass@1
      5. Continuous: execution feedback → periodic retraining on idle compute

    The reward signal is binary correctness from Simula's own
    test/verify pipeline - no human labeling needed.
    """

    status: GRPOTrainingStatus = GRPOTrainingStatus.PENDING
    # Data collection
    total_examples_collected: int = 0
    positive_examples: int = 0  # tests passed + no rollback
    negative_examples: int = 0
    # SFT phase
    sft_examples_used: int = 0
    sft_epochs: int = 0
    sft_final_loss: float = 0.0
    # GRPO phase
    grpo_iterations: int = 0
    grpo_batches_processed: int = 0
    grpo_mean_contrastive_gap: float = 0.0
    # Evaluation
    evaluation: GRPOEvaluationResult | None = None
    # Model metadata
    base_model_id: str = ""
    finetuned_model_id: str = ""
    finetuned_model_path: str = ""
    # Resource usage
    total_gpu_hours: float = 0.0
    total_training_tokens: int = 0
    training_time_ms: int = 0
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    error_summary: str = ""


# ── Stage 4C: Diffusion-Based Code Repair ──────────────────────────────────


class DiffusionRepairStatus(enum.StrEnum):
    """Status of a diffusion repair attempt."""

    REPAIRED = "repaired"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class DiffusionDenoiseStep(EOSBaseModel):
    """One denoising step in the diffusion repair process."""

    step_number: int
    noise_level: float = 1.0  # 1.0 = max noise, 0.0 = clean
    code_snapshot: str = ""
    tests_passed: int = 0
    tests_total: int = 0
    lint_errors: int = 0
    improvement_delta: float = 0.0  # change in test pass rate from previous step
    tokens_used: int = 0


class DiffusionRepairResult(EOSBaseModel):
    """
    Result of diffusion-based code repair.

    Integration: after CEGIS/code-agent exhausts max iterations,
    hand off to diffusion repair for last-mile fixes.

    Two modes:
      1. Iterative denoising: progressively repair broken code
         via denoising steps (DiffuCoder-style)
      2. Sketch-first: diffusion generates code skeleton,
         code agent fills implementation details (Tree Diffusion)

    Benchmark: compare against standard autoregressive repair.
    """

    status: DiffusionRepairStatus = DiffusionRepairStatus.SKIPPED
    mode: str = "iterative_denoise"  # "iterative_denoise" | "sketch_first"
    # Repair process
    denoise_steps: list[DiffusionDenoiseStep] = Field(default_factory=list)
    total_steps: int = 0
    # Input/output
    original_code: str = ""
    repaired_code: str = ""
    files_repaired: list[str] = Field(default_factory=list)
    # Verification
    tests_passed_before: int = 0
    tests_passed_after: int = 0
    tests_total: int = 0
    lint_clean: bool = False
    formal_verification_passed: bool = False
    # Metrics
    repair_success: bool = False
    improvement_rate: float = 0.0  # tests_passed_after / tests_total
    total_llm_tokens: int = 0
    total_time_ms: int = 0
    # Benchmark comparison
    autoregressive_would_pass: bool | None = None  # None = not benchmarked
    error_summary: str = ""


# ── Stage 4: Categories requiring Lean proofs ──────────────────────────────

LEAN_PROOF_CATEGORIES: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.MODIFY_CONTRACT,
    ChangeCategory.ADD_SYSTEM_CAPABILITY,
    ChangeCategory.ADJUST_BUDGET,
    ChangeCategory.MODIFY_CYCLE_TIMING,
})

LEAN_PROOF_DOMAINS: list[str] = [
    "risk_scoring",
    "governance_gating",
    "constitutional_alignment",
    "budget_calculation",
]


# ── Stage 5B: Neural Program Repair (SRepair pattern) ──────────────────────


class RepairPhase(enum.StrEnum):
    """FSM states for the SRepair-style repair agent."""

    DIAGNOSE = "diagnose"
    LOCALIZE = "localize"
    GENERATE_FIX = "generate_fix"
    VERIFY = "verify"
    ACCEPT = "accept"
    REJECT = "reject"


class RepairStatus(enum.StrEnum):
    """Terminal outcome of a repair attempt."""

    REPAIRED = "repaired"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    BUDGET_EXCEEDED = "budget_exceeded"


class FaultLocation(EOSBaseModel):
    """A specific location in the codebase identified as a fault source."""

    file_path: str
    function_name: str = ""
    class_name: str = ""
    line_start: int = 0
    line_end: int = 0
    confidence: float = 0.0
    reasoning: str = ""


class DiagnosisResult(EOSBaseModel):
    """Output of the DIAGNOSE phase - reasoning-model analysis of the failure."""

    error_category: str = ""  # "syntax"|"type"|"logic"|"runtime"|"test"|"import"
    root_cause_hypothesis: str = ""
    affected_components: list[str] = Field(default_factory=list)
    stack_trace_summary: str = ""
    similar_past_fixes: list[str] = Field(default_factory=list)  # evolution record IDs
    reasoning_tokens: int = 0
    confidence: float = 0.0


class LocalizationResult(EOSBaseModel):
    """Output of the LOCALIZE phase - narrowing down fault locations."""

    fault_locations: list[FaultLocation] = Field(default_factory=list)
    search_tools_used: list[str] = Field(default_factory=list)
    files_examined: int = 0
    narrowed_from_files: int = 0  # started with this many candidates
    narrowed_to_files: int = 0  # ended with this many


class FixGenerationResult(EOSBaseModel):
    """Output of the GENERATE_FIX phase - code model produces a patch."""

    fix_description: str = ""
    files_modified: list[str] = Field(default_factory=list)
    diff_summary: str = ""
    code_tokens: int = 0
    alternative_fixes_considered: int = 0
    error: str = ""


class RepairAttempt(EOSBaseModel):
    """One attempt within the repair loop (may retry up to max_retries)."""

    attempt_number: int = 0
    phase: RepairPhase = RepairPhase.DIAGNOSE
    diagnosis: DiagnosisResult | None = None
    localization: LocalizationResult | None = None
    fix_generation: FixGenerationResult | None = None
    tests_passed: bool = False
    lint_clean: bool = False
    type_check_clean: bool = False
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str = ""


class RepairResult(EOSBaseModel):
    """Final aggregated result of the neural repair agent."""

    status: RepairStatus = RepairStatus.SKIPPED
    attempts: list[RepairAttempt] = Field(default_factory=list)
    total_attempts: int = 0
    successful_attempt: int | None = None  # which attempt succeeded (0-indexed)
    files_repaired: list[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_reasoning_tokens: int = 0
    total_code_tokens: int = 0
    diagnosis_summary: str = ""
    fix_summary: str = ""


# ── Stage 6A: Cryptographic Auditability ─────────────────────────────────────


class HashChainStatus(enum.StrEnum):
    """Status of a hash chain verification."""

    VALID = "valid"
    BROKEN = "broken"
    UNVERIFIED = "unverified"
    GENESIS = "genesis"


class ContentCredentialStatus(enum.StrEnum):
    """Status of C2PA content credential verification."""

    SIGNED = "signed"
    VERIFIED = "verified"
    INVALID = "invalid"
    UNSIGNED = "unsigned"
    EXPIRED = "expired"


class VerifiableCredentialStatus(enum.StrEnum):
    """Status of governance verifiable credential verification."""

    VALID = "valid"
    REVOKED = "revoked"
    EXPIRED = "expired"
    UNVERIFIED = "unverified"


class RegulatoryFramework(enum.StrEnum):
    """Supported regulatory audit frameworks."""

    FINANCE_SOX = "finance_sox"
    HEALTHCARE_HIPAA = "healthcare_hipaa"
    DEFENSE_CMMC = "defense_cmmc"
    GENERAL_AUDIT = "general_audit"


class HashChainEntry(EOSBaseModel):
    """One link in the SHA-256 hash chain for EvolutionRecord auditability."""

    record_id: str
    previous_hash: str = ""  # "" for genesis entry
    content_hash: str  # SHA-256 of the record's canonical fields
    chain_hash: str  # SHA-256(previous_hash + content_hash)
    chain_position: int = 0
    verified_at: datetime = Field(default_factory=utc_now)


class HashChainVerificationResult(EOSBaseModel):
    """Aggregate result of walking and verifying the hash chain."""

    status: HashChainStatus = HashChainStatus.UNVERIFIED
    chain_length: int = 0
    entries_verified: int = 0
    break_position: int = -1  # -1 = no break found
    root_hash: str = ""
    tip_hash: str = ""
    duration_ms: int = 0


class ContentCredential(EOSBaseModel):
    """
    C2PA-style content credential for code provenance.

    Every generated file carries an authorship proof:
    content hash + Ed25519 signature + issuer metadata.
    """

    file_path: str
    content_hash: str  # SHA-256 of file content
    issuer: str = "EcodiaOS Simula"
    signature: str = ""  # hex-encoded Ed25519 signature
    algorithm: str = "Ed25519"
    c2pa_manifest_json: str = ""  # JSON-encoded C2PA manifest
    created_at: datetime = Field(default_factory=utc_now)


class ContentCredentialResult(EOSBaseModel):
    """Aggregate result of signing or verifying a batch of files."""

    status: ContentCredentialStatus = ContentCredentialStatus.UNSIGNED
    credentials: list[ContentCredential] = Field(default_factory=list)
    unsigned_files: list[str] = Field(default_factory=list)
    verified_count: int = 0
    invalid_count: int = 0
    duration_ms: int = 0


class GovernanceCredential(EOSBaseModel):
    """
    Verifiable Credential for governance decisions.

    Each governance approval/rejection carries a signed, tamper-evident
    credential that forms an auditable approval chain.
    """

    governance_record_id: str
    proposal_id: str
    approver_id: str = ""
    decision: str = ""  # "approved"|"rejected"|"deferred"
    signature: str = ""  # hex-encoded Ed25519 signature
    signed_payload_hash: str = ""  # SHA-256 of the signed payload
    credential_chain_json: str = ""  # JSON-encoded chain of prior credentials
    issued_at: datetime = Field(default_factory=utc_now)


class GovernanceCredentialResult(EOSBaseModel):
    """Aggregate result of verifying governance credentials for a proposal."""

    status: VerifiableCredentialStatus = VerifiableCredentialStatus.UNVERIFIED
    credentials: list[GovernanceCredential] = Field(default_factory=list)
    chain_verified: bool = False
    chain_length: int = 0
    duration_ms: int = 0


# ── Stage 6B: Co-Evolving Agents ─────────────────────────────────────────────


class FailureCaseSource(enum.StrEnum):
    """Where a hard negative training example originated."""

    ROLLBACK_HISTORY = "rollback_history"
    HEALTH_FAILURE = "health_failure"
    FORMAL_VERIFICATION_FAILURE = "formal_verification_failure"
    ADVERSARIAL_GENERATION = "adversarial_generation"


class FailureCaseExample(EOSBaseModel):
    """
    A hard negative example for GRPO training.

    Hard negatives are code-generation failures that the model should learn
    to avoid: rollbacks, verification failures, health check crashes.
    """

    source: FailureCaseSource
    proposal_id: str = ""
    category: str = ""
    failure_reason: str = ""
    code_context: str = ""  # the code that was generated (and failed)
    expected_output: str = ""  # what correct output would look like
    adversarial_input: str = ""  # the adversarial prompt/test that triggered failure
    mined_at: datetime = Field(default_factory=utc_now)


class RobustnessTestResult(EOSBaseModel):
    """Result of one adversarial test generation cycle."""

    tests_generated: int = 0
    tests_executed: int = 0
    tests_found_bugs: int = 0
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    coverage_delta: float = 0.0
    test_files_written: list[str] = Field(default_factory=list)
    bug_descriptions: list[str] = Field(default_factory=list)
    duration_ms: int = 0


class CoevolutionCycleResult(EOSBaseModel):
    """Aggregate result of one co-evolution cycle (mine + test + feed GRPO)."""

    hard_negatives_mined: int = 0
    adversarial_tests_generated: int = 0
    tests_found_bugs: int = 0
    grpo_examples_produced: int = 0
    coverage_growth_percent: float = 0.0
    duration_ms: int = 0


# ── Stage 6C: Formal Spec Generation ─────────────────────────────────────────


class FormalSpecKind(enum.StrEnum):
    """Kind of formal specification generated."""

    DAFNY = "dafny"
    TLA_PLUS = "tla_plus"
    ALLOY = "alloy"
    SELF_SPEC_DSL = "self_spec_dsl"


class FormalSpecStatus(enum.StrEnum):
    """Status of a formal spec generation + verification attempt."""

    GENERATED = "generated"
    VERIFIED = "verified"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class FormalSpecResult(EOSBaseModel):
    """Result of generating and verifying one formal specification."""

    kind: FormalSpecKind = FormalSpecKind.DAFNY
    status: FormalSpecStatus = FormalSpecStatus.SKIPPED
    spec_source: str = ""  # the generated specification text
    target_function: str = ""
    target_file: str = ""
    coverage_percent: float = 0.0
    verified: bool = False
    verification_output: str = ""
    llm_tokens_used: int = 0
    duration_ms: int = 0


class TlaPlusModelCheckResult(EOSBaseModel):
    """Result of TLC model checking a TLA+ specification."""

    status: FormalSpecStatus = FormalSpecStatus.SKIPPED
    spec_source: str = ""
    system_name: str = ""
    states_explored: int = 0
    distinct_states: int = 0
    violations: list[str] = Field(default_factory=list)
    deadlocks_found: int = 0
    duration_ms: int = 0


class AlloyCheckResult(EOSBaseModel):
    """Result of Alloy analyzer checking system properties."""

    status: FormalSpecStatus = FormalSpecStatus.SKIPPED
    model_source: str = ""
    instances_found: int = 0
    counterexamples: list[str] = Field(default_factory=list)
    scope: int = 10  # Alloy scope (bound on universe size)
    duration_ms: int = 0


class SelfSpecDSL(EOSBaseModel):
    """
    A task-specific DSL invented by the LLM for novel proposal categories.

    Self-Spec: when no existing formal method fits, the system invents
    a domain-specific language to specify the expected behavior.
    """

    dsl_name: str = ""
    grammar_source: str = ""  # BNF/PEG grammar of the DSL
    example_programs: list[str] = Field(default_factory=list)
    target_category: str = ""
    coverage_rate: float = 0.0
    llm_tokens_used: int = 0


class FormalSpecGenerationResult(EOSBaseModel):
    """Aggregate result of all formal spec generation for a proposal."""

    specs: list[FormalSpecResult] = Field(default_factory=list)
    overall_coverage_percent: float = 0.0
    tla_plus_results: list[TlaPlusModelCheckResult] = Field(default_factory=list)
    alloy_results: list[AlloyCheckResult] = Field(default_factory=list)
    self_spec_dsls: list[SelfSpecDSL] = Field(default_factory=list)
    total_llm_tokens: int = 0
    total_duration_ms: int = 0


# ── Stage 6D: Equality Saturation (E-graphs) ─────────────────────────────────


class EGraphStatus(enum.StrEnum):
    """Status of an e-graph equality saturation run."""

    SATURATED = "saturated"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    FAILED = "failed"
    SKIPPED = "skipped"


class RewriteRule(EOSBaseModel):
    """One algebraic rewrite rule for the e-graph engine."""

    name: str
    pattern: str  # AST pattern to match (s-expression notation)
    replacement: str  # replacement pattern
    condition: str = ""  # optional guard condition


class EGraphEquivalenceResult(EOSBaseModel):
    """
    Result of checking semantic equivalence via equality saturation.

    Two code snippets are equivalent if, after applying rewrite rules
    to saturation, they reside in the same e-class in the e-graph.
    """

    status: EGraphStatus = EGraphStatus.SKIPPED
    original_hash: str = ""
    rewritten_hash: str = ""
    semantically_equivalent: bool = False
    rules_applied: list[str] = Field(default_factory=list)
    iterations: int = 0
    e_class_count: int = 0
    e_node_count: int = 0
    duration_ms: int = 0


# ── Stage 6E: Hybrid Symbolic Execution ──────────────────────────────────────


class SymbolicExecutionStatus(enum.StrEnum):
    """Outcome of a symbolic execution property check."""

    PROVED = "proved"
    COUNTEREXAMPLE = "counterexample"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


class SymbolicDomain(enum.StrEnum):
    """Domains targeted for symbolic execution (mission-critical logic)."""

    BUDGET_CALCULATION = "budget_calculation"
    ACCESS_CONTROL = "access_control"
    RISK_SCORING = "risk_scoring"
    GOVERNANCE_GATING = "governance_gating"
    CONSTITUTIONAL_ALIGNMENT = "constitutional_alignment"


class PathCondition(EOSBaseModel):
    """A path condition explored during symbolic execution."""

    condition_expr: str  # Z3 expression as string
    satisfiable: bool = True
    model_values: dict[str, str] = Field(default_factory=dict)  # variable -> concrete value


class SymbolicProperty(EOSBaseModel):
    """A property to be checked via Z3 symbolic execution."""

    domain: SymbolicDomain
    property_name: str
    z3_encoding: str  # Python code that constructs Z3 expressions
    human_description: str = ""
    target_function: str = ""
    target_file: str = ""
    status: SymbolicExecutionStatus = SymbolicExecutionStatus.SKIPPED
    counterexample: str = ""


class SymbolicExecutionResult(EOSBaseModel):
    """Aggregate result of symbolic execution across all checked properties."""

    status: SymbolicExecutionStatus = SymbolicExecutionStatus.SKIPPED
    properties_checked: int = 0
    properties_proved: int = 0
    properties_failed: int = 0
    counterexamples: list[str] = Field(default_factory=list)
    path_conditions_explored: int = 0
    properties: list[SymbolicProperty] = Field(default_factory=list)
    z3_time_ms: int = 0
    duration_ms: int = 0


# ── Stage 6 Combined Result ──────────────────────────────────────────────────


class FormalGuaranteesResult(EOSBaseModel):
    """
    Combined result of all Stage 6 formal guarantee checks.

    Attached to HealthCheckResult for pass/fail decision-making.
    E-graph equivalence is advisory by default.
    Symbolic execution is blocking for proved properties.
    """

    hash_chain: HashChainVerificationResult | None = None
    content_credentials: ContentCredentialResult | None = None
    governance_credentials: GovernanceCredentialResult | None = None
    formal_specs: FormalSpecGenerationResult | None = None
    egraph: EGraphEquivalenceResult | None = None
    symbolic_execution: SymbolicExecutionResult | None = None
    passed: bool = True
    blocking_issues: list[str] = Field(default_factory=list)
    advisory_issues: list[str] = Field(default_factory=list)
    total_duration_ms: int = 0
