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


class ChangeCategory(enum.StrEnum):
    ADD_EXECUTOR = "add_executor"
    ADD_INPUT_CHANNEL = "add_input_channel"
    ADD_PATTERN_DETECTOR = "add_pattern_detector"
    ADJUST_BUDGET = "adjust_budget"
    MODIFY_CONTRACT = "modify_contract"
    ADD_SYSTEM_CAPABILITY = "add_system_capability"
    MODIFY_CYCLE_TIMING = "modify_cycle_timing"
    CHANGE_CONSOLIDATION = "change_consolidation"
    # BUG_FIX: runtime errors that Simula can autonomously fix
    # (missing attributes, unregistered executors, KeyErrors in config, method signature mismatches).
    # Low blast radius by definition — no governance gate required. Skips heavy simulation.
    BUG_FIX = "bug_fix"
    # Constitutional amendment: changes drive weights via the formal
    # amendment pipeline (deliberation -> shadow -> vote -> adoption).
    # Governance-required, not FORBIDDEN -- the community vote path
    # provides the safety boundary that FORBIDDEN enforces for direct mods.
    CONSTITUTIONAL_AMENDMENT = "constitutional_amendment"
    MODIFY_EQUOR = "modify_equor"
    MODIFY_CONSTITUTION = "modify_constitution"
    MODIFY_INVARIANTS = "modify_invariants"
    MODIFY_SELF_EVOLUTION = "modify_self_evolution"


class ProposalStatus(enum.StrEnum):
    PROPOSED = "proposed"
    SIMULATING = "simulating"
    AWAITING_GOVERNANCE = "awaiting_governance"
    APPROVED = "approved"
    APPLYING = "applying"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


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
    version: int | None = None
    governance_record_id: str | None = None
    files_changed: list[str] = Field(default_factory=list)
    # Bounty Hunter integration: PR submitted on behalf of this proposal
    pr_url: str = ""
    pr_number: int | None = None


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


class CodeChangeResult(EOSBaseModel):
    """What the code agent returns after implementing a structural change."""

    success: bool
    files_written: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str = ""
    lint_passed: bool = True
    tests_passed: bool = True
    test_output: str = ""
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
    ChangeCategory.MODIFY_EQUOR,
    ChangeCategory.MODIFY_CONSTITUTION,
    ChangeCategory.MODIFY_INVARIANTS,
    ChangeCategory.MODIFY_SELF_EVOLUTION,
})

SIMULA_IRON_RULES: list[str] = [
    "Simula CANNOT modify Equor in any way.",
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
    "systems/equor",
    "systems/simula",
    "ecodiaos/primitives/constitutional.py",
    "ecodiaos/primitives/common.py",
    "ecodiaos/config.py",
]
