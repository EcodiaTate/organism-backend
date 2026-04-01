"""
EcodiaOS -- Simula Evolution Types  (formerly inspector_types.py)

All data types internal to the Simula self-evolution system.
Simula is the organism's capacity for metamorphosis: structural change
beyond parameter tuning. These types model the full lifecycle of an
evolution proposal -- from reception through simulation, governance,
application, and immutable history.

Namespace: systems.simula.evolution_types
Distinct from: systems.simula.inspector.types  (vulnerability-discovery types)
Do NOT import Inspector types here.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
    utc_now,
)

# --- Enums -------------------------------------------------------------------


# Re-exported from primitives so Thymos and other systems can import these
# without a cross-system violation.  All other code that already imports
# from this module continues to work unchanged.
from primitives.evolution import ChangeCategory as ChangeCategory  # noqa: E402
from primitives.evolution import ProposalStatus as ProposalStatus  # noqa: E402


class ProposalSource(enum.StrEnum):
    """Source of proposal origin - used instead of string comparisons."""
    EVO = "evo"
    THYMOS = "thymos"
    AXON = "axon"
    PROACTIVE = "proactive"


class RiskLevel(enum.StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class ImpactType(enum.StrEnum):
    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    NEUTRAL = "neutral"


class TriageStatus(enum.StrEnum):
    """Status of proposal triage (fast-path pre-simulation check)."""
    TRIVIAL = "trivial"
    REQUIRES_SIMULATION = "requires_simulation"


# --- Models -----------------------------------------------------------------


class ChangeSpec(EOSBaseModel):
    """
    Formal specification of what to change.
    One model covers every ChangeCategory -- fields are optional by category.
    """

    # ADD_EXECUTOR
    executor_name: str | None = None
    executor_description: str | None = None
    executor_action_type: str | None = None
    executor_input_schema: dict[str, Any] | None = None

    # ADD_INPUT_CHANNEL
    channel_name: str | None = None
    channel_type: str | None = None
    channel_description: str | None = None

    # ADD_PATTERN_DETECTOR
    detector_name: str | None = None
    detector_description: str | None = None
    detector_pattern_type: str | None = None

    # ADJUST_BUDGET
    budget_parameter: str | None = None
    budget_old_value: float | None = None
    budget_new_value: float | None = None

    # MODIFY_CONTRACT
    contract_changes: list[str] = Field(default_factory=list)

    # ADD_SYSTEM_CAPABILITY
    capability_description: str | None = None

    # MODIFY_CYCLE_TIMING
    timing_parameter: str | None = None
    timing_old_value: float | None = None
    timing_new_value: float | None = None

    # CHANGE_CONSOLIDATION
    consolidation_schedule: str | None = None

    # CONSTITUTIONAL_AMENDMENT
    amendment_proposed_drives: dict[str, float] | None = None
    amendment_rationale: str | None = None
    amendment_evidence_hypothesis_ids: list[str] = Field(default_factory=list)

    # Cross-cutting
    affected_systems: list[str] = Field(default_factory=list)
    additional_context: str = ""
    code_hint: str = ""  # optional hint of what the code should look like


class SimulationDifference(EOSBaseModel):
    """Describes how one episode's outcome would differ under the proposed change."""

    episode_id: str
    original_outcome: str
    simulated_outcome: str
    impact: ImpactType
    reasoning: str = ""


class SimulationResult(EOSBaseModel):
    """Aggregate outcome of simulating a proposal against recent episodes."""

    episodes_tested: int = 0
    differences: int = 0
    improvements: int = 0
    regressions: int = 0
    neutral_changes: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    risk_summary: str = ""
    benefit_summary: str = ""
    simulated_at: datetime = Field(default_factory=utc_now)


class CautionAdjustment(EOSBaseModel):
    """
    Transparent caution adjustment logic explaining WHY a proposal's risk
    was bumped. Returned by EvolutionAnalyticsEngine.should_increase_caution().
    """

    should_adjust: bool
    magnitude: float  # 0.0-0.5 additive risk bump
    factors: dict[str, float] = Field(default_factory=dict)  # {factor_name: contribution}
    reasoning: str = ""


class TriageResult(EOSBaseModel):
    """Result of fast-path proposal triage (pre-simulation check)."""

    status: TriageStatus
    assumed_risk: RiskLevel | None = None
    reason: str = ""
    skip_simulation: bool = False


class ProposalResult(EOSBaseModel):
    """Final outcome recorded once a proposal reaches a terminal state."""

    status: ProposalStatus
    reason: str = ""
    error: str = ""  # Truncated to 1000 chars if necessary (see validator below)
    version: int | None = None
    governance_record_id: str | None = None
    files_changed: list[str] = Field(default_factory=list)
    # Bounty Hunter integration: PR submitted on behalf of this proposal
    pr_url: str = ""
    pr_number: int | None = None
    # Canary deployment plan attached when risk_level == MODERATE (Spec §7)
    canary_plan: "CanaryDeploymentPlan | None" = None

    def model_post_init(self, __context: Any) -> None:
        """Cap error field at 1000 chars to prevent unbounded truncation."""
        if len(self.error) > 1000:
            self.error = self.error[:997] + "..."


class EvolutionProposal(Identified, Timestamped):
    """
    The full proposal lifecycle object -- richer than Evo's simplified version.
    Owns the proposal from receipt through simulation, governance, and application.
    """

    source: str  # "evo" | "governance" | "bounty"
    category: ChangeCategory
    description: str
    change_spec: ChangeSpec
    evidence: list[str] = Field(default_factory=list)  # hypothesis IDs / episode IDs
    expected_benefit: str = ""
    risk_assessment: str = ""
    status: ProposalStatus = ProposalStatus.PROPOSED
    simulation: SimulationResult | None = None
    governance_record_id: str | None = None
    result: ProposalResult | None = None
    # Bounty Hunter integration: trace back to the originating bounty & target repo
    source_bounty_id: str | None = None
    target_repository_url: str | None = None
    # Path to the cloned external workspace (str so no pathlib import needed here).
    # When set, Simula's code agent and applicator operate on this path instead of
    # the organism's own codebase_root.
    workspace_root: str | None = None
    # Architecture-level active inference: EFE score from ArchitectureEFEScorer.
    # Lower (more negative) = better expected value. None = not yet scored.
    efe_score: float | None = None
    # Lucid dreaming origin: True when this proposal was generated from a
    # high-coherence REM dream insight during sleep. Dream-origin proposals
    # are based on generalizations from accumulated experience rather than
    # external input, so they carry inherently lower novelty risk.
    dream_origin: bool = False
    # The coherence score of the originating dream insight (0.0-1.0).
    # Only set when dream_origin is True.
    dream_coherence_score: float | None = None
    # Sleep cycle that produced this proposal.
    dream_sleep_cycle_id: str | None = None
    # ── Corpus 14 §11: Per-proposal token budget tracking ─────────────────────
    # Token budget allocated to this proposal. Starts at the system max and is
    # decremented as LLM calls consume tokens across simulation/counterfactual/etc.
    # 0 means unlimited (budget tracking not active).
    token_budget: int = 0       # 0 = unlimited; >0 = per-proposal cap
    tokens_consumed: int = 0    # cumulative tokens used across all stages

    @property
    def remaining_token_budget(self) -> int:
        """Remaining token budget. Returns maxsize when unlimited."""
        import sys
        if self.token_budget <= 0:
            return sys.maxsize
        return max(0, self.token_budget - self.tokens_consumed)


class FileSnapshot(EOSBaseModel):
    """
    One file's state immediately before a change was applied, enabling rollback.
    content is None when the file did not previously exist -- rollback deletes it.
    """

    path: str  # absolute path
    content: str | None  # None means file did not exist before
    existed: bool = True


class ConfigSnapshot(Identified, Timestamped):
    """Full snapshot of all affected files captured before applying a change."""

    proposal_id: str
    files: list[FileSnapshot] = Field(default_factory=list)
    config_version: int  # the version at snapshot time


class ConfigVersion(EOSBaseModel):
    """Tracks one step in the config version chain."""

    version: int
    timestamp: datetime = Field(default_factory=utc_now)
    proposal_ids: list[str] = Field(default_factory=list)  # evolution proposal IDs
    config_hash: str  # SHA256 hash of the canonical config state


class EvolutionRecord(Identified, Timestamped):
    """Immutable history entry written to Neo4j after each successful application."""

    proposal_id: str
    category: ChangeCategory
    description: str
    from_version: int
    to_version: int
    files_changed: list[str] = Field(default_factory=list)
    simulation_risk: RiskLevel
    applied_at: datetime = Field(default_factory=utc_now)
    rolled_back: bool = False
    rollback_reason: str = ""
    # Simulation detail persisted for audit trail and learning
    simulation_episodes_tested: int = 0
    counterfactual_regression_rate: float = 0.0
    dependency_blast_radius: int = 0
    constitutional_alignment: float = 0.0
    resource_tokens_per_hour: int = 0
    caution_reasoning: str = ""
    # Stage 2: Formal verification metadata
    formal_verification_status: str = ""  # "verified"|"failed"|"skipped"|""
    discovered_invariants_count: int = 0
    dafny_rounds: int = 0
    static_analysis_findings: int = 0
    # Stage 4A: Lean 4 proof metadata
    lean_proof_status: str = ""  # "proved"|"failed"|"timeout"|"skipped"|""
    lean_proof_rounds: int = 0
    lean_proven_lemmas_count: int = 0
    lean_copilot_automation_rate: float = 0.0
    lean_library_lemmas_reused: int = 0
    # Stage 4B: GRPO fine-tuning metadata
    grpo_model_used: str = ""  # "" = base model, else fine-tuned model id
    grpo_ab_group: str = ""  # "base"|"finetuned"|""
    # Stage 4C: Diffusion repair metadata
    diffusion_repair_used: bool = False
    diffusion_repair_status: str = ""  # "repaired"|"partial"|"failed"|"skipped"|""
    diffusion_repair_steps: int = 0
    diffusion_improvement_rate: float = 0.0
    # Stage 5A: Neurosymbolic synthesis metadata
    synthesis_strategy_used: str = ""  # "hysynth"|"sketch_solve"|"chopchop"|"cegis_fallback"|""
    synthesis_status: str = ""  # "synthesized"|"partial"|"failed"|"timeout"|"skipped"|""
    synthesis_speedup_vs_baseline: float = 0.0
    synthesis_candidates_explored: int = 0
    # Stage 5B: Neural repair metadata
    repair_agent_used: bool = False
    repair_agent_status: str = ""  # "repaired"|"partial"|"failed"|"timeout"|"skipped"|"budget_exceeded"|""
    repair_attempts: int = 0
    repair_cost_usd: float = 0.0
    # Stage 5C: Orchestration metadata
    orchestration_used: bool = False
    orchestration_dag_nodes: int = 0
    orchestration_agents_used: int = 0
    orchestration_parallel_stages: int = 0
    # Stage 5D: Causal debugging metadata
    causal_debug_used: bool = False
    causal_root_cause: str = ""
    causal_confidence: float = 0.0
    causal_interventions: int = 0
    # Stage 5E: Issue resolution metadata
    issue_resolution_used: bool = False
    issue_autonomy_level: str = ""  # "lint"|"dependency"|"test_fix"|"logic_bug"|""
    issue_abstained: bool = False
    # Stage 6A: Cryptographic auditability metadata
    hash_chain_hash: str = ""  # SHA-256 chain hash for this record
    hash_chain_position: int = 0  # position in the hash chain
    content_credentials_signed: int = 0  # number of files signed with C2PA
    governance_credential_status: str = ""  # "valid"|"revoked"|"expired"|"unverified"|""
    # Stage 6B: Co-evolution metadata
    coevolution_hard_negatives_mined: int = 0
    coevolution_adversarial_tests: int = 0
    coevolution_bugs_found: int = 0
    # Stage 6C: Formal spec generation metadata
    formal_specs_generated: int = 0
    formal_spec_coverage_percent: float = 0.0
    tla_plus_states_explored: int = 0
    # Stage 6D: E-graph metadata
    egraph_used: bool = False
    egraph_status: str = ""  # "saturated"|"partial"|"timeout"|"failed"|"skipped"|""
    egraph_rules_applied: int = 0
    # Stage 6E: Symbolic execution metadata
    symbolic_execution_used: bool = False
    symbolic_properties_proved: int = 0
    symbolic_counterexamples: int = 0
    # Prompt #16: Causal self-surgery metadata
    causal_surgery_used: bool = False
    causal_surgery_parameter: str = ""     # e.g. "nova.efe.pragmatic"
    causal_surgery_success_rate: float = 0.0
    causal_surgery_condition: str = ""     # e.g. "when policy_type == 'high_leverage'"
    # Bounty Hunter integration: PR created from this evolution
    pr_url: str = ""
    pr_number: int | None = None
    # ── Corpus 14 §8: MemoryTrace standardization (Spec 01 bi-temporal episode format) ──
    # EvolutionRecord participates in Memory's bi-temporal knowledge graph.
    # episode_id links this record to its MemoryTrace in Neo4j.
    # perception_time = when the proposal was first received (event_time in bi-temporal).
    # reflection_time = when the outcome was recorded (ingestion_time in bi-temporal).
    episode_id: str = ""           # UUID linking to MemoryTrace node in Neo4j
    perception_time: datetime | None = None   # proposal created_at (event_time)
    reflection_time: datetime | None = None   # applied_at (ingestion_time)
    # ── Corpus 14 §13: Identity scope (Spec 23) ───────────────────────────────
    # Records which instance produced this mutation for cryptographic audit.
    # Before applying, service verifies this matches the current instance_id.
    identity_id: str = ""          # ECODIAOS_INSTANCE_ID of the recording instance


class CodeChangeResult(EOSBaseModel):
    """What the code agent returns after implementing a structural change."""

    success: bool
    files_written: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str = ""
    lint_passed: bool = True
    tests_passed: bool = True
    test_output: str = ""
    # LLM token budget tracking
    total_tokens: int = 0          # cumulative input+output tokens across all turns
    system_prompt_tokens: int = 0  # tokens consumed by system prompt (static budget)
    # Stage 1A: Extended-thinking model metrics
    used_extended_thinking: bool = False
    reasoning_tokens: int = 0
    # Stage 1C: KVzip context compression metrics
    kv_compression_ratio: float = 0.0  # 0.0 = no savings, 1.0 = maximum
    kv_messages_compressed: int = 0
    kv_original_tokens: int = 0
    kv_compressed_tokens: int = 0
    # Stage 2C: Static analysis metrics
    static_analysis_findings: int = 0
    static_analysis_fix_iterations: int = 0
    # Stage 2D: AgentCoder metrics
    agent_coder_iterations: int = 0
    test_designer_test_count: int = 0
    # Stage 4B: GRPO model routing metrics
    grpo_model_used: str = ""
    grpo_ab_group: str = ""  # "base"|"finetuned"|""
    # Stage 4C: Diffusion repair metrics
    diffusion_repair_attempted: bool = False
    diffusion_repair_succeeded: bool = False
    # Stage 5A: Synthesis metrics
    synthesis_strategy: str = ""  # "hysynth"|"sketch_solve"|"chopchop"|"cegis_fallback"|""
    synthesis_speedup: float = 0.0
    # Stage 5B: Repair metrics
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_cost_usd: float = 0.0
    # Stage 5C: Orchestration metrics
    orchestration_used: bool = False
    orchestration_agents: int = 0
    # Bounty Hunter integration: PR submitted from this code change
    pr_url: str = ""
    pr_number: int | None = None


class HealthCheckResult(EOSBaseModel):
    """Result of a post-apply codebase health check."""

    healthy: bool
    issues: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now)
    # Inspector Phase 3: Mutation formal verification (pre-apply, between tests and formal verification)
    mutation_verification: object | None = None  # MutationVerificationResult
    # Stage 2: Formal verification result (attached when verification runs)
    formal_verification: object | None = None  # FormalVerificationResult
    # Stage 4A: Lean 4 proof verification result (attached when Lean verification runs)
    lean_verification: object | None = None  # LeanVerificationResult
    # Stage 5D: Causal debugging result (attached when causal debug runs)
    causal_diagnosis: object | None = None  # CausalDiagnosis
    # Stage 6: Formal guarantees result (attached when Stage 6 checks run)
    formal_guarantees: object | None = None  # FormalGuaranteesResult


# --- Enriched Simulation Models ----------------------------------------------


class CounterfactualResult(EOSBaseModel):
    """
    Result of asking: 'If this change had existed during episode X,
    what would have been different?'

    Batched into a single LLM call across multiple episodes for
    token efficiency (~800 tokens per 30-episode batch).
    """

    episode_id: str
    would_have_triggered: bool = False
    predicted_outcome: str = ""
    impact: ImpactType = ImpactType.NEUTRAL
    confidence: float = 0.5
    reasoning: str = ""


class DependencyImpact(EOSBaseModel):
    """
    A file or module affected by a proposed change, discovered
    via static import-graph analysis (zero LLM tokens).
    """

    file_path: str
    impact_type: str = "import_dependency"  # "direct_modification" | "import_dependency" | "test_coverage"
    risk_contribution: float = 0.0


class ResourceCostEstimate(EOSBaseModel):
    """
    Heuristic estimation of the ongoing resource cost a change
    would add to the running system. Computed without LLM calls.
    """

    estimated_additional_llm_tokens_per_hour: int = 0
    estimated_additional_compute_ms_per_cycle: int = 0
    estimated_memory_mb: float = 0.0
    budget_headroom_percent: float = 100.0


class EnrichedSimulationResult(SimulationResult):
    """
    Extended simulation result with deep multi-strategy analysis.
    Produced by the upgraded ChangeSimulator, consumed by SimulaService
    for richer risk/benefit decision-making.
    """

    counterfactuals: list[CounterfactualResult] = Field(default_factory=list)
    dependency_impacts: list[DependencyImpact] = Field(default_factory=list)
    resource_cost_estimate: ResourceCostEstimate | None = None
    constitutional_alignment: float = 0.0
    counterfactual_regression_rate: float = 0.0
    dependency_blast_radius: int = 0
    caution_adjustment: CautionAdjustment | None = None


# --- Canary Deployment Models ------------------------------------------------


class CanaryTrafficStep(EOSBaseModel):
    """One step in a canary traffic ramp schedule.

    Spec ref: Section 7 - Canary Deployment.
    """

    traffic_percentage: int  # 0–100
    hold_duration: str        # e.g. "1 hour", "2 hours"
    rollback_criteria: list[str] = Field(default_factory=list)


class CanaryDeploymentPlan(EOSBaseModel):
    """Graduated traffic-ramp plan for MODERATE-risk proposals.

    When simulation yields RiskLevel.MODERATE, the change is applied
    immediately but traffic is shifted incrementally.  The plan is stored
    on the EvolutionRecord so the ProactiveScanner and Thymos can monitor
    health at each ramp step and trigger rollback if criteria are violated.

    Spec ref: Section 7 - Temporal Simulation & Forward Modeling.
    """

    proposal_id: str
    initial_traffic_percentage: int = 10
    increase_schedule: list[CanaryTrafficStep] = Field(default_factory=list)
    rollback_criteria: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    rolled_back: bool = False
    rollback_reason: str = ""

    @classmethod
    def default_for_proposal(cls, proposal_id: str) -> "CanaryDeploymentPlan":
        """Build the standard 4-step ramp plan from spec §7."""
        return cls(
            proposal_id=proposal_id,
            initial_traffic_percentage=10,
            increase_schedule=[
                CanaryTrafficStep(
                    traffic_percentage=10,
                    hold_duration="1 hour",
                    rollback_criteria=["error_rate > 5%", "latency_p99 > 200ms"],
                ),
                CanaryTrafficStep(
                    traffic_percentage=25,
                    hold_duration="2 hours",
                    rollback_criteria=["error_rate > 5%", "latency_p99 > 200ms"],
                ),
                CanaryTrafficStep(
                    traffic_percentage=50,
                    hold_duration="4 hours",
                    rollback_criteria=["error_rate > 3%", "latency_p99 > 150ms"],
                ),
                CanaryTrafficStep(
                    traffic_percentage=100,
                    hold_duration="ongoing",
                    rollback_criteria=[],
                ),
            ],
            rollback_criteria=["error_rate > 5%", "latency_p99 > 200ms"],
        )


# --- Constraint Satisfaction Models ------------------------------------------


class ConstraintViolation(EOSBaseModel):
    """A specific Iron Rule or invariant violated by a proposal.

    Spec ref: Section 8 - Constraint Satisfaction in Imagined Scenarios.
    """

    constraint_id: str   # e.g. "equor_immutability", "drive_normalization"
    description: str = ""
    severity: str = "hard"  # "hard" (blocks proposal) | "soft" (advisory)


# --- Structured Health & Metrics Models --------------------------------------


class SimulaComponentHealth(EOSBaseModel):
    """Health status of a single Simula sub-system component.

    Spec ref: Section 17 - Health & Monitoring.
    """

    name: str
    status: str  # "healthy" | "degraded" | "disabled" | "unhealthy"
    detail: str = ""


class SimulaMetrics(EOSBaseModel):
    """Operational KPI snapshot for the Simula system.

    Spec ref: Section 17 - Health & Monitoring.
    """

    proposals_processed: int = 0
    proposals_applied: int = 0
    proposals_rolled_back: int = 0
    proposals_rejected: int = 0
    proposals_deduplicated: int = 0
    proposals_awaiting_governance: int = 0
    active_proposals: int = 0
    current_version: int = 0

    success_rate: float = 0.0    # applied / processed
    rollback_rate: float = 0.0   # rolled_back / applied

    evolution_velocity: float = 0.0  # proposals applied per hour (from analytics)
    mean_simulation_risk: float = 0.0

    grid_state: str = "normal"
    starvation_level: str = "nominal"


class HealthStatus(EOSBaseModel):
    """Full liveness + component health report for GET /simula/health.

    Spec ref: Section 17 - Health & Monitoring.
    """

    service: str = "simula"
    status: str  # "healthy" | "degraded" | "unhealthy"
    components: list[SimulaComponentHealth] = Field(default_factory=list)
    metrics: SimulaMetrics = Field(default_factory=SimulaMetrics)
    reason: str = ""
    checked_at: datetime = Field(default_factory=utc_now)


# --- Bridge Models -----------------------------------------------------------


class EvoProposalEnriched(EOSBaseModel):
    """
    Evo proposal enriched with hypothesis evidence and inferred context.
    Produced by EvoSimulaBridge, consumed by SimulaService.translate().
    """

    evo_description: str
    evo_rationale: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    hypothesis_statements: list[str] = Field(default_factory=list)
    evidence_scores: list[float] = Field(default_factory=list)
    supporting_episode_ids: list[str] = Field(default_factory=list)
    mutation_target: str = ""
    mutation_type: str = ""
    inferred_category: ChangeCategory | None = None
    inferred_change_spec: ChangeSpec | None = None


# --- Proposal Intelligence Models --------------------------------------------


class ProposalPriority(EOSBaseModel):
    """
    Priority score for a proposal, enabling intelligent processing order.
    Higher priority_score = process first.

    Formula: evidence_strength * expected_impact / max(0.1, estimated_risk * estimated_cost)
    """

    proposal_id: str
    priority_score: float = 0.0
    evidence_strength: float = 0.0
    expected_impact: float = 0.0
    estimated_risk: float = 0.0
    estimated_cost: float = 0.0
    reasoning: str = ""


class ProposalCluster(EOSBaseModel):
    """
    Group of semantically similar proposals that could be deduplicated.
    Detected via cheap heuristics first, LLM only for ambiguous cases.
    """

    representative_id: str
    member_ids: list[str] = Field(default_factory=list)
    similarity_scores: list[float] = Field(default_factory=list)
    merge_recommendation: str = ""


# --- Analytics Models --------------------------------------------------------


class CategorySuccessRate(EOSBaseModel):
    """Success rate tracking for a specific change category."""

    category: ChangeCategory
    total: int = 0
    approved: int = 0
    rejected: int = 0
    rolled_back: int = 0

    @property
    def success_rate(self) -> float:
        return self.approved / max(1, self.total)

    @property
    def rollback_rate(self) -> float:
        return self.rolled_back / max(1, self.total)


class EvolutionAnalytics(EOSBaseModel):
    """
    Aggregate evolution quality metrics computed from Neo4j history.
    Enables Simula to learn from its own performance over time.
    Zero LLM tokens -- pure computation from stored records.
    """

    category_rates: dict[str, CategorySuccessRate] = Field(default_factory=dict)
    total_proposals: int = 0
    evolution_velocity: float = 0.0  # proposals per day
    mean_simulation_risk: float = 0.0
    rollback_rate: float = 0.0
    recent_rollback_rates: dict[str, float] = Field(default_factory=dict)  # per-category 7-day rate
    last_updated: datetime = Field(default_factory=utc_now)


# --- Architecture EFE Models -------------------------------------------------


class ProposalEFEBreakdown(EOSBaseModel):
    """
    Detailed EFE breakdown for a single evolution proposal.
    Mirrors Nova's EFE decomposition but adapted for structural changes.
    """

    pragmatic_value: float = 0.0       # Fraction of historical failures addressed
    epistemic_value: float = 0.0       # Entropy reduction from the proposed change
    complexity_penalty: float = 0.0    # Cost of added parameters / code paths / state
    efe_raw: float = 0.0               # -(pragmatic + epistemic)
    efe_penalised: float = 0.0         # efe_raw - (complexity * penalty_weight)
    confidence: float = 0.5            # Scorer confidence in this estimate
    reasoning: str = ""


class RankedProposalQueue(EOSBaseModel):
    """
    A batch of proposals ranked by architecture-level EFE.
    Produced by ArchitectureEFEScorer, consumed by Equor and SimulaService.
    """

    proposals: list[EvolutionProposal] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)           # EFE scores (most negative = best)
    breakdowns: list[ProposalEFEBreakdown] = Field(default_factory=list)
    recommended: EvolutionProposal | None = None                # Top-ranked proposal
    scored_at: datetime = Field(default_factory=utc_now)


class RiskWeightVector(EOSBaseModel):
    """
    Learnable weights for the risk synthesis composite score.

    Default values match the original hard-coded spec (§11):
      base=0.40, counterfactual=0.20, dependency=0.15, resource=0.10, alignment=0.15

    Evo tunes these by writing (:RiskWeights) nodes to Neo4j via
    ChangeSimulator.persist_weights() / ChangeSimulator.load_weights().
    The weights are normalised to sum=1.0 before use; any individual weight
    must stay in [0.05, 0.70] to prevent degenerate collapse.
    """

    w_base: float = 0.40
    w_counterfactual: float = 0.20
    w_dependency: float = 0.15
    w_resource: float = 0.10
    w_alignment: float = 0.15
    updated_at: datetime = Field(default_factory=utc_now)

    def normalised(self) -> "RiskWeightVector":
        """Return a copy with weights normalised to sum=1.0."""
        total = self.w_base + self.w_counterfactual + self.w_dependency + self.w_resource + self.w_alignment
        if total == 0:
            return RiskWeightVector()
        return RiskWeightVector(
            w_base=self.w_base / total,
            w_counterfactual=self.w_counterfactual / total,
            w_dependency=self.w_dependency / total,
            w_resource=self.w_resource / total,
            w_alignment=self.w_alignment / total,
        )


class EFECalibrationRecord(EOSBaseModel):
    """
    Feedback record comparing predicted EFE with actual post-application outcome.
    Persisted as :ProposalEFE nodes in Neo4j for calibration learning.
    """

    proposal_id: str
    predicted_efe: float = 0.0
    actual_improvement: float = 0.0    # Measured change in failure rate post-apply
    efe_error: float = 0.0             # actual_improvement - predicted_efe
    pragmatic_predicted: float = 0.0
    epistemic_predicted: float = 0.0
    complexity_predicted: float = 0.0
    applied_at: datetime = Field(default_factory=utc_now)
    measured_at: datetime = Field(default_factory=utc_now)


# --- Constants ---------------------------------------------------------------

SELF_APPLICABLE: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.ADD_EXECUTOR,
    ChangeCategory.ADD_INPUT_CHANNEL,
    ChangeCategory.ADD_PATTERN_DETECTOR,
    ChangeCategory.ADJUST_BUDGET,
})

GOVERNANCE_REQUIRED: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.MODIFY_CONTRACT,
    ChangeCategory.ADD_SYSTEM_CAPABILITY,
    ChangeCategory.MODIFY_CYCLE_TIMING,
    ChangeCategory.CHANGE_CONSOLIDATION,
    ChangeCategory.CONSTITUTIONAL_AMENDMENT,
})

FORBIDDEN: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.MODIFY_CONSTITUTION,
    ChangeCategory.MODIFY_INVARIANTS,
    ChangeCategory.MODIFY_SELF_EVOLUTION,
})

SIMULA_IRON_RULES: list[str] = [
    "Simula CANNOT modify constitutional drives.",
    "Simula CANNOT modify invariants.",
    "Simula CANNOT modify its own logic (no self-modifying code).",
    "Simula CANNOT bypass governance for governed changes.",
    "Simula CANNOT apply changes without rollback capability.",
    "Simula CANNOT delete evolution history records.",
    "Simula MUST simulate before applying any change.",
    "Simula MUST maintain version continuity -- no identity-breaking changes.",
]

# Paths the code agent is NEVER allowed to write to
FORBIDDEN_WRITE_PATHS: list[str] = [
    "systems/simula",
    "ecodiaos/primitives/constitutional.py",
    "ecodiaos/primitives/common.py",
    "ecodiaos/config.py",
]
