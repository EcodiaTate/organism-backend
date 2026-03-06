"""
EcodiaOS — Kairos Type Definitions

All data types for the Causal Invariant Mining system.

Kairos mines the hierarchy of knowledge:
  Correlations → Causal rules → Context-invariant causal rules → Substrate-independent invariants

Each level up is exponentially more compressed and more generative.
A single Tier 3 invariant can generate predictions across every domain it touches.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003 — Pydantic needs this at runtime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now

# ─── Enums ────────────────────────────────────────────────────────────


class PipelineStage(enum.StrEnum):
    """Seven stages of the Kairos causal mining pipeline."""

    CORRELATION_MINING = "correlation_mining"
    CAUSAL_DIRECTION = "causal_direction"
    CONFOUNDER_ANALYSIS = "confounder_analysis"
    MECHANISM_EXTRACTION = "mechanism_extraction"
    CONTEXT_INVARIANCE = "context_invariance"
    INVARIANT_DISTILLATION = "invariant_distillation"
    DOMAIN_MAPPING = "domain_mapping"


class CausalDirection(enum.StrEnum):
    """Possible causal direction outcomes."""

    A_CAUSES_B = "A_causes_B"
    B_CAUSES_A = "B_causes_A"
    BIDIRECTIONAL = "bidirectional"
    NO_CAUSAL = "no_causal"


class CausalInvariantTier(enum.IntEnum):
    """Three tiers of causal invariant generality."""

    TIER_1_DOMAIN = 1        # Holds within a single domain
    TIER_2_CROSS_DOMAIN = 2  # Holds across multiple distinct domains
    TIER_3_SUBSTRATE = 3     # Holds regardless of substrate — deepest layer


class InvarianceVerdict(enum.StrEnum):
    """Outcome of context invariance testing."""

    STRONG_INVARIANT = "strong_invariant"          # hold_rate >= 0.95
    CONDITIONAL_INVARIANT = "conditional_invariant"  # hold_rate >= 0.75
    CONTEXT_SPECIFIC = "context_specific"           # hold_rate < 0.75
    INSUFFICIENT_CONTEXTS = "insufficient_contexts"  # < 5 contexts available


class DirectionTestMethod(enum.StrEnum):
    """Methods used to determine causal direction."""

    TEMPORAL_PRECEDENCE = "temporal_precedence"
    INTERVENTION_ASYMMETRY = "intervention_asymmetry"
    ADDITIVE_NOISE_MODEL = "additive_noise_model"


# ─── Stage 1: Correlation Mining ──────────────────────────────────────


class ContextCorrelation(EOSBaseModel):
    """Correlation measurement for a variable pair within a single context."""

    context_id: str
    correlation: float = 0.0
    sample_count: int = 0


class CorrelationCandidate(Identified):
    """
    A cross-context correlation candidate that passed the consistency filter.

    Selection criteria:
    - mean |r| > 0.3 (non-trivial correlation)
    - cross-context variance < 0.1 (consistent across contexts)
    """

    variable_a: str
    variable_b: str
    mean_correlation: float = 0.0
    cross_context_variance: float = 0.0
    context_count: int = 0
    context_correlations: list[ContextCorrelation] = Field(default_factory=list)
    discovered_at: datetime = Field(default_factory=utc_now)

    @property
    def abs_mean_correlation(self) -> float:
        return abs(self.mean_correlation)


# ─── Stage 2: Causal Direction Testing ────────────────────────────────


class DirectionTestResult(EOSBaseModel):
    """Result of a single direction test method."""

    method: DirectionTestMethod
    direction: CausalDirection
    confidence: float = 0.0
    evidence_count: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class TemporalEvidence(EOSBaseModel):
    """Temporal ordering evidence between two variables."""

    a_precedes_b_count: int = 0
    b_precedes_a_count: int = 0
    simultaneous_count: int = 0
    mean_lag_ms: float = 0.0

    @property
    def total(self) -> int:
        return self.a_precedes_b_count + self.b_precedes_a_count + self.simultaneous_count

    @property
    def a_precedes_b_ratio(self) -> float:
        if self.total == 0:
            return 0.0
        return self.a_precedes_b_count / self.total

    @property
    def b_precedes_a_ratio(self) -> float:
        if self.total == 0:
            return 0.0
        return self.b_precedes_a_count / self.total


class InterventionEvidence(EOSBaseModel):
    """Evidence from Axon intervention logs about causal asymmetry."""

    interventions_on_a: int = 0
    b_changed_after_a_intervention: int = 0
    interventions_on_b: int = 0
    a_changed_after_b_intervention: int = 0

    @property
    def a_causes_b_score(self) -> float:
        """P(B changes | do(A)) — high means A→B is supported."""
        if self.interventions_on_a == 0:
            return 0.0
        return self.b_changed_after_a_intervention / self.interventions_on_a

    @property
    def b_causes_a_score(self) -> float:
        """P(A changes | do(B)) — high means B→A is supported."""
        if self.interventions_on_b == 0:
            return 0.0
        return self.a_changed_after_b_intervention / self.interventions_on_b


class AdditiveNoiseResult(EOSBaseModel):
    """
    Additive noise model test result.

    For the correct causal direction X→Y, the residuals of regressing Y on X
    are independent of X. For the wrong direction, they are dependent.
    """

    residual_independence_a_to_b: float = 0.0  # Higher = more independent = A→B supported
    residual_independence_b_to_a: float = 0.0  # Higher = more independent = B→A supported


class CausalDirectionResult(Identified):
    """
    Combined result of all direction tests for a correlation candidate.

    Acceptance requires confidence > 0.6 and at least 2 methods agreeing.
    """

    candidate: CorrelationCandidate
    direction: CausalDirection = CausalDirection.NO_CAUSAL
    confidence: float = 0.0
    accepted: bool = False
    test_results: list[DirectionTestResult] = Field(default_factory=list)
    temporal_evidence: TemporalEvidence | None = None
    intervention_evidence: InterventionEvidence | None = None
    noise_model_result: AdditiveNoiseResult | None = None
    tested_at: datetime = Field(default_factory=utc_now)

    @property
    def cause(self) -> str:
        if self.direction == CausalDirection.A_CAUSES_B:
            return self.candidate.variable_a
        elif self.direction == CausalDirection.B_CAUSES_A:
            return self.candidate.variable_b
        return ""

    @property
    def effect(self) -> str:
        if self.direction == CausalDirection.A_CAUSES_B:
            return self.candidate.variable_b
        elif self.direction == CausalDirection.B_CAUSES_A:
            return self.candidate.variable_a
        return ""


# ─── Stage 3: Confounder Analysis ─────────────────────────────────────


class ConfirmedConfounder(EOSBaseModel):
    """A confirmed confounding variable between two other variables."""

    variable: str
    conditional_independence_score: float = 0.0
    partial_correlation_residual: float = 0.0
    mdl_improvement: float = 0.0  # Bits saved by removing the spurious edge


class ConfounderResult(Identified):
    """
    Result of confounder analysis for a causal direction result.

    If confounders are found, the causal relationship is spurious:
    removing the A→B edge and keeping C→A + C→B saves description length.
    """

    original_pair: CausalDirectionResult
    is_confounded: bool = False
    confounding_variables: list[ConfirmedConfounder] = Field(default_factory=list)
    adjusted_correlation: float = 0.0  # Correlation after conditioning on confounders
    mdl_improvement: float = 0.0  # Total bits saved by removing spurious connection
    pc_edges_removed: int = 0  # Edges removed during PC algorithm
    pc_edges_oriented: int = 0  # Edges oriented during PC algorithm
    analyzed_at: datetime = Field(default_factory=utc_now)


# ─── Context Invariance Testing ───────────────────────────────────────


class CausalRule(Identified, Timestamped):
    """
    A confirmed causal rule that passed direction testing and confounder analysis.

    This is the input to context invariance testing (Stage 5).
    """

    cause_variable: str
    effect_variable: str
    direction_confidence: float = 0.0
    domain: str = ""
    mechanism: str = ""  # Natural language description of the causal mechanism
    observation_count: int = 0
    source_candidate_id: str = ""  # ID of the CorrelationCandidate that spawned this


class ContextTestResult(EOSBaseModel):
    """Result of testing a causal rule in a single context."""

    context_id: str
    rule_holds: bool = False
    correlation_in_context: float = 0.0
    sample_count: int = 0


class ScopeCondition(EOSBaseModel):
    """A condition under which a causal rule holds or fails."""

    condition: str  # Natural language description
    holds_when: bool = True  # True = rule holds when condition is met
    distinguishing_feature: str = ""
    context_ids: list[str] = Field(default_factory=list)


class InvarianceTestResult(Identified):
    """Result of testing a causal rule across multiple contexts."""

    rule: CausalRule
    verdict: InvarianceVerdict = InvarianceVerdict.INSUFFICIENT_CONTEXTS
    hold_rate: float = 0.0
    context_count: int = 0
    contexts_tested: list[ContextTestResult] = Field(default_factory=list)
    failing_contexts: list[str] = Field(default_factory=list)
    scope_conditions: list[ScopeCondition] = Field(default_factory=list)
    tested_at: datetime = Field(default_factory=utc_now)


# ─── Causal Hierarchy ─────────────────────────────────────────────────


class ApplicableDomain(EOSBaseModel):
    """A domain where a causal invariant has been tested and holds."""

    domain: str
    substrate: str = ""  # physical, biological, computational, social, economic
    hold_rate: float = 0.0
    observation_count: int = 0


class CausalInvariant(Identified, Timestamped):
    """
    A confirmed causal invariant — the most compressed form of causal knowledge.

    One invariant generates predictions across every domain it touches.
    Tier 3 invariants are architectural events for the world model.
    """

    tier: CausalInvariantTier = CausalInvariantTier.TIER_1_DOMAIN
    abstract_form: str = ""  # The invariant statement in its most abstract form
    concrete_instances: list[str] = Field(default_factory=list)  # CausalRule IDs
    applicable_domains: list[ApplicableDomain] = Field(default_factory=list)
    invariance_hold_rate: float = 0.0
    scope_conditions: list[ScopeCondition] = Field(default_factory=list)
    intelligence_ratio_contribution: float = 0.0
    description_length_bits: float = 0.0
    source_rule_id: str = ""  # ID of the CausalRule that was promoted

    # Phase C: Distillation fields
    distilled: bool = False  # True after InvariantDistiller has processed this
    variable_roles: dict[str, str] = Field(default_factory=dict)
    """Maps concrete variable names to abstract roles (e.g. "price" -> "quantity")."""
    is_tautological: bool = False  # Set by tautology test
    is_minimal: bool = False  # Set by minimality test
    untested_domains: list[str] = Field(default_factory=list)
    """Domains where the abstract form matches but the invariant hasn't been tested."""

    # Phase D: Counter-invariant tracking
    violation_count: int = 0
    refined_scope: str = ""  # Natural language boundary condition
    last_violation_check: datetime | None = None

    @property
    def domain_count(self) -> int:
        return len(self.applicable_domains)

    @property
    def substrate_count(self) -> int:
        return len({d.substrate for d in self.applicable_domains if d.substrate})

    @property
    def cross_domain_transfer_value(self) -> float:
        """Number of untested domains × invariant coverage."""
        return len(self.untested_domains) * self.invariance_hold_rate


# ─── Phase C: Invariant Distillation ────────────────────────────────────


class DistillationResult(Identified):
    """Result of distilling an invariant to its minimal abstract form."""

    invariant_id: str
    original_form: str = ""
    abstract_form: str = ""
    variable_roles: dict[str, str] = Field(default_factory=dict)
    is_tautological: bool = False
    is_minimal: bool = False
    parts_removed: int = 0  # Number of redundant parts stripped in minimality
    untested_domains: list[str] = Field(default_factory=list)
    distilled_at: datetime = Field(default_factory=utc_now)


# ─── Phase D: Counter-Invariant Detection ───────────────────────────────


class InvariantViolation(Identified):
    """A single observed violation of an accepted invariant."""

    invariant_id: str
    violating_observation: dict[str, Any] = Field(default_factory=dict)
    violation_context: str = ""  # Context/domain where the violation occurred
    expected_direction: str = ""  # What the invariant predicted
    observed_value: float = 0.0  # What was actually observed
    distinguishing_features: dict[str, float] = Field(default_factory=dict)
    """Features of the violating context that differ from holding contexts."""
    detected_at: datetime = Field(default_factory=utc_now)


class ViolationCluster(EOSBaseModel):
    """A cluster of violations sharing a common distinguishing feature."""

    invariant_id: str
    violations: list[str] = Field(default_factory=list)  # InvariantViolation IDs
    common_feature: str = ""
    feature_threshold: float = 0.0
    """The feature value above/below which the invariant breaks."""
    is_significant: bool = False  # True if feature is a real modulator, not noise
    cluster_size: int = 0


class RefinedScope(EOSBaseModel):
    """A refined scope for an invariant after counter-invariant analysis."""

    invariant_id: str
    original_hold_rate: float = 0.0
    refined_hold_rate: float = 0.0
    boundary_condition: str = ""
    """Natural language: 'holds everywhere EXCEPT contexts with feature X > threshold'."""
    excluded_feature: str = ""
    excluded_threshold: float = 0.0
    contexts_excluded: int = 0


# ─── Phase D: Intelligence Contribution Ledger ──────────────────────────


class IntelligenceContribution(EOSBaseModel):
    """Per-invariant accounting of contribution to the intelligence ratio."""

    invariant_id: str
    observations_covered: int = 0
    """How many observations does this invariant explain?"""
    description_savings: float = 0.0
    """Description length WITHOUT the invariant minus WITH it (in bits)."""
    invariant_length: float = 0.0
    """Description length of the invariant itself (in bits)."""
    intelligence_ratio_contribution: float = 0.0
    """savings / invariant_length — how much this invariant compresses."""
    intelligence_ratio_without: float = 0.0
    """What the overall I-ratio would be if this invariant were removed."""
    computed_at: datetime = Field(default_factory=utc_now)


# ─── Synapse Event Payloads ───────────────────────────────────────────


class CausalCandidatePayload(EOSBaseModel):
    """Payload for KAIROS_CAUSAL_CANDIDATE_GENERATED event."""

    candidate_id: str
    variable_a: str
    variable_b: str
    mean_correlation: float
    cross_context_variance: float
    context_count: int


class CausalDirectionPayload(EOSBaseModel):
    """Payload for KAIROS_CAUSAL_DIRECTION_ACCEPTED event."""

    result_id: str
    cause: str
    effect: str
    direction: str
    confidence: float
    methods_agreed: int


class ConfounderDiscoveredPayload(EOSBaseModel):
    """Payload for KAIROS_CONFOUNDER_DISCOVERED event."""

    result_id: str
    original_cause: str
    original_effect: str
    confounders: list[str]
    mdl_improvement: float
    is_spurious: bool


class Tier3InvariantPayload(EOSBaseModel):
    """Payload for KAIROS_TIER3_INVARIANT_DISCOVERED — highest-priority Kairos event."""

    invariant_id: str
    abstract_form: str
    domain_count: int
    substrate_count: int
    hold_rate: float
    description_length_bits: float
    intelligence_ratio_contribution: float
    applicable_domains: list[str]
    untested_domains: list[str]


class CounterInvariantPayload(EOSBaseModel):
    """Payload for KAIROS_COUNTER_INVARIANT_FOUND."""

    invariant_id: str
    violation_count: int
    boundary_condition: str
    excluded_feature: str
    original_hold_rate: float
    refined_hold_rate: float


class InvariantCandidatePayload(EOSBaseModel):
    """Payload for KAIROS_INVARIANT_CANDIDATE — Stage 5 strong invariant."""

    invariant_id: str
    cause: str
    effect: str
    hold_rate: float
    context_count: int
    verdict: str


class InvariantDistilledPayload(EOSBaseModel):
    """Payload for KAIROS_INVARIANT_DISTILLED — Stage 6 complete."""

    invariant_id: str
    abstract_form: str
    domain_count: int
    is_minimal: bool
    untested_domain_count: int


class IntelligenceRatioStepChangePayload(EOSBaseModel):
    """Payload for KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE."""

    invariant_id: str
    old_ratio: float
    new_ratio: float
    delta: float
    cause: str  # "tier3_discovered", "counter_invariant_refined", "domain_expanded"


# ─── Pipeline Configuration ──────────────────────────────────────────


class KairosConfig(EOSBaseModel):
    """Configuration for the Kairos causal invariant mining system."""

    # Stage 1: Correlation Mining
    min_cross_context_count: int = 3
    min_abs_mean_correlation: float = 0.3
    max_cross_context_variance: float = 0.1

    # Stage 2: Direction Testing
    min_direction_confidence: float = 0.6
    min_temporal_evidence_count: int = 5
    temporal_precedence_threshold: float = 0.6  # > 60% consistency required

    # Stage 3: Confounder Analysis
    conditional_independence_threshold: float = 0.05  # p-value for independence test
    min_partial_correlation_drop: float = 0.5  # Drop in |r| when conditioning

    # Context Invariance
    min_contexts_for_invariance: int = 5
    strong_invariant_threshold: float = 0.95
    conditional_invariant_threshold: float = 0.75

    # Tier promotion
    tier2_min_domains: int = 2
    tier3_min_domains: int = 4
    tier3_min_substrates: int = 3

    # Fovea integration
    causal_error_route_threshold: float = 0.15  # Matches Fovea routing

    # Phase C: Distillation
    tautology_min_variables: int = 2  # Invariants with fewer variables are suspect
    minimality_hold_rate_tolerance: float = 0.02  # Max drop when removing a part
    tier3_min_contexts: int = 5  # Minimum contexts for Tier 3 promotion

    # Phase D: Counter-invariant detection
    violation_significance_threshold: float = 0.3  # Min feature diff to be "significant"
    min_violations_for_cluster: int = 3  # Min violations to form a cluster
    min_cluster_size_for_refinement: int = 3  # Min cluster size to refine scope

    # Pipeline timing
    mining_interval_s: float = 300.0  # How often to run the full pipeline
