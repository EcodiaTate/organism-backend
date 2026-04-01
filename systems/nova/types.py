"""
EcodiaOS - Nova Internal Types

All types internal to Nova's decision and planning system.
These are richer than the shared primitives - they carry the full
cognitive context needed for deliberation, goal tracking, and EFE scoring.

Design notes:
- BeliefState is Nova's internal model of the world. It is NOT the shared
  Belief primitive (which represents a single probability distribution).
- Goal is Nova's rich internal goal structure. When an Intent is dispatched,
  it carries a GoalDescriptor (from primitives/intent.py), which is a lean
  summary suitable for cross-system communication.
- Policy is Nova's internal candidate action plan, distinct from Intent.
  Intents are finalised, Equor-reviewed plans; Policies are candidates.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003
from typing import Any

from pydantic import Field

from primitives.affect import AffectState  # noqa: TC003
from primitives.common import (
    DriveAlignmentVector,
    EOSBaseModel,
    Identified,
    Timestamped,
    new_id,
    utc_now,
)

# ─── Belief State ─────────────────────────────────────────────────


class EntityBelief(EOSBaseModel):
    """Nova's belief about a single entity in the world."""

    entity_id: str
    name: str = ""
    entity_type: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    # 0.0 = completely uncertain, 1.0 = certain
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    last_observed: datetime = Field(default_factory=utc_now)
    # Percept/episode IDs that support this belief
    source_episodes: list[str] = Field(default_factory=list)


class ContextBelief(EOSBaseModel):
    """Nova's belief about the current conversational/situational context."""

    summary: str = ""
    domain: str = ""           # e.g., "technical", "emotional", "social"
    is_active_dialogue: bool = False
    user_intent_estimate: str = ""
    # Surprise level - how different is this from predictions?
    prediction_error_magnitude: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SelfBelief(EOSBaseModel):
    """Nova's beliefs about EOS's own state and capabilities."""

    # Map of capability name → confidence (0-1)
    capabilities: dict[str, float] = Field(default_factory=dict)
    # Estimated cognitive load (0-1)
    cognitive_load: float = Field(default=0.0, ge=0.0, le=1.0)
    # Confidence in own current beliefs overall
    epistemic_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Estimated goal completion capacity (can we take on more?)
    goal_capacity_remaining: float = Field(default=1.0, ge=0.0, le=1.0)


class IndividualBelief(EOSBaseModel):
    """Nova's beliefs about a specific individual."""

    individual_id: str
    name: str = ""
    # Estimated emotional state (valence estimate)
    estimated_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Confidence in that estimate
    valence_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    # Estimated engagement level (0-1)
    engagement_level: float = Field(default=0.5, ge=0.0, le=1.0)
    # General trust in the interaction
    relationship_trust: float = Field(default=0.5, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=utc_now)


class BeliefState(EOSBaseModel):
    """
    Nova's complete world model.

    This is the cognitive map - the best current estimate of world state.
    Updated continuously from workspace broadcasts and Memory retrieval.
    Drives all deliberation: which goals are relevant, which policies can work,
    what the expected free energy of each action is.

    Variational free energy (VFE) field tracks the aggregate prediction error:
        VFE ≈ Σ_i (1 - confidence_i) × salience_i
    Lower VFE = beliefs are well-supported = less surprise = better organism state.
    """

    # ── World model ──
    entities: dict[str, EntityBelief] = Field(default_factory=dict)

    # ── Situation model ──
    current_context: ContextBelief = Field(default_factory=ContextBelief)
    active_individual_ids: list[str] = Field(default_factory=list)
    individual_beliefs: dict[str, IndividualBelief] = Field(default_factory=dict)

    # ── Self model ──
    self_belief: SelfBelief = Field(default_factory=SelfBelief)

    # ── Metadata ──
    last_updated: datetime = Field(default_factory=utc_now)
    # Overall belief confidence (mean precision across all beliefs)
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Current variational free energy estimate
    free_energy: float = Field(default=0.5, ge=0.0, le=1.0)

    def compute_free_energy(self) -> float:
        """
        Compute variational free energy as the precision-weighted prediction error.

        VFE ≈ 1 - mean(confidence) across salient beliefs.
        Lower is better (well-supported beliefs = low surprise).
        This is a tractable approximation of the full VFE functional.
        """
        confidences: list[float] = [self.overall_confidence]
        confidences.extend(e.confidence for e in self.entities.values())
        confidences.append(self.current_context.confidence)
        if self.individual_beliefs:
            confidences.extend(b.valence_confidence for b in self.individual_beliefs.values())
        mean_confidence = sum(confidences) / max(1, len(confidences))
        return 1.0 - mean_confidence


# ─── Free Energy Budget ──────────────────────────────────────────


class FreeEnergyBudget(EOSBaseModel):
    """
    Information-theoretic pressure valve for Nova's cognitive cycle.

    Tracks cumulative prediction error (in nats) per budget window. When
    accumulated surprise crosses the threshold fraction of the total budget,
    Nova triggers an emergency consolidation interrupt and optionally reduces
    policy generation diversity (K candidates) to conserve cognitive resources.

    Nats are the natural unit of information (1 nat ≈ 1.443 bits). Prediction
    error magnitudes (0-1 from Atune) are converted to nats via the mapping:
        nats = -ln(1 - magnitude)   (clamped so magnitude < 0.999)
    This maps 0→0, 0.5→0.693, 0.9→2.303, 0.99→4.605 nats.

    Budget resets after consolidation completes or when the time window elapses.
    """

    # Total surprise budget per window (tunable via Evo)
    budget_nats: float = Field(default=5.0, ge=0.1, le=50.0)
    # Accumulated prediction error this window
    spent_nats: float = Field(default=0.0, ge=0.0)
    # Fraction at which consolidation interrupt fires (0.8 = 80% spent)
    threshold_fraction: float = Field(default=0.8, ge=0.1, le=1.0)
    # Reduced K for policy generation when budget is pressured (> threshold * 0.6)
    reduced_k: int = Field(default=2, ge=1, le=5)
    # Normal K for policy generation (restored after budget reset)
    normal_k: int = Field(default=5, ge=2, le=10)
    # Number of consolidation interrupts triggered this session
    interrupts_triggered: int = Field(default=0, ge=0)
    # Whether the budget is currently exhausted (waiting for consolidation)
    is_exhausted: bool = False

    @property
    def threshold_nats(self) -> float:
        """The nats threshold at which consolidation fires."""
        return self.budget_nats * self.threshold_fraction

    @property
    def remaining_nats(self) -> float:
        """How much surprise budget remains."""
        return max(0.0, self.budget_nats - self.spent_nats)

    @property
    def utilisation(self) -> float:
        """Fraction of budget consumed (0-1)."""
        if self.budget_nats <= 0:
            return 1.0
        return min(1.0, self.spent_nats / self.budget_nats)

    @property
    def is_pressured(self) -> bool:
        """True when budget usage exceeds 60% of threshold - reduce K."""
        return self.spent_nats > self.threshold_nats * 0.6

    @staticmethod
    def magnitude_to_nats(magnitude: float) -> float:
        """
        Convert prediction error magnitude (0-1) to nats of surprise.

        Uses the information-theoretic mapping: I = -ln(1 - p)
        where p is the prediction error magnitude (probability of being wrong).
        """
        import math
        clamped = min(magnitude, 0.999)
        if clamped <= 0.0:
            return 0.0
        return -math.log(1.0 - clamped)

    def accumulate(self, magnitude: float) -> float:
        """
        Add prediction error to the budget. Returns nats added.
        Does NOT check threshold - caller must check would_exhaust() first.
        """
        nats = self.magnitude_to_nats(magnitude)
        self.spent_nats += nats
        return nats

    def would_exhaust(self, magnitude: float) -> bool:
        """Check if adding this percept's error would cross the threshold."""
        nats = self.magnitude_to_nats(magnitude)
        return (self.spent_nats + nats) > self.threshold_nats

    def reset(self) -> None:
        """Reset the budget after consolidation or time window expiry."""
        self.spent_nats = 0.0
        self.is_exhausted = False

    @property
    def effective_k(self) -> int:
        """Policy generation K, reduced when budget is pressured."""
        if self.is_pressured or self.is_exhausted:
            return self.reduced_k
        return self.normal_k


class BeliefDelta(EOSBaseModel):
    """
    A structured change to the belief state.
    Produced by belief update operations and used for goal progress assessment.
    """

    entity_updates: dict[str, EntityBelief] = Field(default_factory=dict)
    entity_additions: dict[str, EntityBelief] = Field(default_factory=dict)
    entity_removals: list[str] = Field(default_factory=list)
    context_update: ContextBelief | None = None
    individual_updates: dict[str, IndividualBelief] = Field(default_factory=dict)
    prediction_error_magnitude: float = Field(default=0.0, ge=0.0, le=1.0)
    contradicted_belief_ids: list[str] = Field(default_factory=list)

    def involves_belief_conflict(self) -> bool:
        """True if this delta contains contradictions with existing beliefs."""
        return len(self.contradicted_belief_ids) > 0 or self.prediction_error_magnitude > 0.6

    def is_empty(self) -> bool:
        return (
            not self.entity_updates
            and not self.entity_additions
            and not self.entity_removals
            and self.context_update is None
            and not self.individual_updates
        )


# ─── Goal Types ───────────────────────────────────────────────────


class GoalStatus(enum.StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class GoalSource(enum.StrEnum):
    USER_REQUEST = "user_request"
    SELF_GENERATED = "self_generated"
    GOVERNANCE = "governance"
    CARE_RESPONSE = "care_response"
    MAINTENANCE = "maintenance"
    EPISTEMIC = "epistemic"


class Goal(Identified, Timestamped):
    """
    A living goal structure. Goals are not tasks - they are desires.
    Priority, urgency, and importance shift with context.
    """

    description: str
    target_domain: str = ""
    # The specific success state we want to reach (natural language)
    success_criteria: str = ""

    # ── Priority ──
    priority: float = Field(default=0.5, ge=0.0, le=1.0)   # Dynamic, recomputed each cycle
    urgency: float = Field(default=0.3, ge=0.0, le=1.0)    # Time sensitivity
    importance: float = Field(default=0.5, ge=0.0, le=1.0) # Constitutional weight

    # ── Drive alignment ──
    drive_alignment: DriveAlignmentVector = Field(default_factory=DriveAlignmentVector)

    # ── Source & Lifecycle ──
    source: GoalSource = GoalSource.USER_REQUEST
    status: GoalStatus = GoalStatus.ACTIVE
    deadline: datetime | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Dependencies ──
    depends_on: list[str] = Field(default_factory=list)  # Goal IDs
    blocks: list[str] = Field(default_factory=list)

    # ── Tracking ──
    intents_issued: list[str] = Field(default_factory=list)
    evidence_of_progress: list[str] = Field(default_factory=list)  # Episode IDs


class PriorityContext(EOSBaseModel):
    """Context needed for dynamic goal priority computation."""

    current_affect: AffectState = Field(default_factory=AffectState.neutral)
    drive_weights: dict[str, float] = Field(
        default_factory=lambda: {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
    )
    goal_statuses: dict[str, str] = Field(default_factory=dict)  # goal_id → status
    # Episode timestamps for staleness computation
    episode_timestamps: dict[str, datetime] = Field(default_factory=dict)


# ─── Policy Types ─────────────────────────────────────────────────


class PolicyStep(EOSBaseModel):
    """A single step in a policy's execution plan."""

    action_type: str  # "express" | "observe" | "request_info" | "store" | "wait" | "federate"
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    expected_duration_ms: int = 1000


class Policy(Identified):
    """
    A candidate course of action.
    Generated by the PolicyGenerator and scored by the EFEEvaluator.
    Policies are Nova-internal; they become Intents after Equor review.
    """

    name: str
    # "deliberate" | "express" | "observe" | "defer" | "do_nothing" | etc.
    type: str = "deliberate"
    description: str = ""              # Human-readable description of the policy
    reasoning: str = ""
    steps: list[PolicyStep] = Field(default_factory=list)
    fallback_steps: list[PolicyStep] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    epistemic_value_description: str = ""
    estimated_effort: str = "medium"    # "none" | "low" | "medium" | "high"
    time_horizon: str = "short"         # "immediate" | "short" | "medium" | "long"
    # Set by EFEEvaluator
    efe_score: float | None = None


# ─── EFE Scoring ──────────────────────────────────────────────────


class PragmaticEstimate(EOSBaseModel):
    """How well a policy achieves the goal."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Estimated probability of goal achievement
    success_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""


class EpistemicEstimate(EOSBaseModel):
    """How much uncertainty a policy reduces."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    # How many uncertain beliefs would this policy test?
    uncertainties_addressed: int = 0
    expected_info_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    # Is this genuinely exploring new territory?
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)


class RiskEstimate(EOSBaseModel):
    """Expected harm from executing a policy."""

    expected_harm: float = Field(default=0.0, ge=0.0, le=1.0)
    reversibility: float = Field(default=1.0, ge=0.0, le=1.0)  # 1.0 = fully reversible
    identified_risks: list[str] = Field(default_factory=list)


class EFEScore(EOSBaseModel):
    """
    The complete Expected Free Energy decomposition for a policy.

    G(π) = -[pragmatic_value + epistemic_value + constitutional_alignment + feasibility]
           + risk_penalty
           + λ * cognition_cost_term

    Lower total = more preferred policy (active inference convention).
    The cognition_cost_term penalises policies that are expensive to
    deliberate on, aligning cognitive investment with decision importance.
    """

    # Component scores (all 0-1, higher = better)
    pragmatic: PragmaticEstimate = Field(default_factory=PragmaticEstimate)
    epistemic: EpistemicEstimate = Field(default_factory=EpistemicEstimate)
    constitutional_alignment: float = Field(default=0.5, ge=0.0, le=1.0)
    feasibility: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: RiskEstimate = Field(default_factory=RiskEstimate)

    # Cognition cost (metabolic cost of deliberating on this policy)
    cognition_cost_usd: float = 0.0
    # Normalised cognition cost term (cost / budget, clamped to [0,1])
    cognition_cost_term: float = 0.0

    # Weighted total (lower = preferred)
    total: float = 0.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""


class EFEWeights(EOSBaseModel):
    """
    Weights for EFE components.
    Starting defaults match spec; Evo adjusts them over time.

    cognition_cost (λ): the "cognitive frugality" weight. Higher values
    make the system prefer cheaper-to-deliberate policies. Default 0.1
    means cognition cost has a small but meaningful influence on policy
    selection. Evo tunes this based on cost-adjusted outcome quality.
    """

    pragmatic: float = 0.35
    epistemic: float = 0.20
    constitutional: float = 0.20
    feasibility: float = 0.15
    risk: float = 0.10
    cognition_cost: float = 0.10  # λ - cognitive frugality tunable


# ─── Situation Assessment ─────────────────────────────────────────


class SituationAssessment(EOSBaseModel):
    """
    The output of the fast/slow routing decision.
    Determines which deliberation path to take.
    """

    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    belief_conflict: bool = False
    requires_deliberation: bool = False
    # Was a matching fast-path procedure found in memory?
    has_matching_procedure: bool = False
    # The broadcast precision (importance signal from Atune)
    broadcast_precision: float = Field(default=0.5, ge=0.0, le=1.0)


# ─── Pending Intent & Outcome Tracking ───────────────────────────


class PendingIntent(EOSBaseModel):
    """Tracks an intent that has been dispatched and is awaiting outcome."""

    intent_id: str
    goal_id: str
    routed_to: str  # "voxis" | "axon"
    dispatched_at: datetime = Field(default_factory=utc_now)
    policy_name: str = ""
    executors: list[str] = Field(default_factory=list)
    # Tournament metadata: set when this intent's source hypothesis is in an A/B tournament
    tournament_id: str | None = None
    tournament_hypothesis_id: str | None = None


class IntentOutcome(EOSBaseModel):
    """The result of executing an intent."""

    intent_id: str
    success: bool
    episode_id: str = ""
    failure_reason: str = ""
    # Any new information revealed by execution
    new_observations: list[str] = Field(default_factory=list)


# ─── Decision Record (Observability) ─────────────────────────────


class DecisionRecord(EOSBaseModel):
    """
    Full record of a deliberation cycle for observability and Evo learning.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    broadcast_id: str = ""
    path: str = ""  # "fast" | "slow" | "do_nothing" | "no_goal" | "budget_exhausted"
    situation_assessment: SituationAssessment = Field(default_factory=SituationAssessment)
    goal_id: str | None = None
    goal_description: str = ""
    policies_generated: int = 0
    selected_policy_name: str = ""
    efe_scores: dict[str, float] = Field(default_factory=dict)
    equor_verdict: str = ""
    intent_dispatched: bool = False
    latency_ms: int = 0
    # Free energy budget state at time of decision
    fe_budget_spent_nats: float | None = None
    fe_budget_remaining_nats: float | None = None
    fe_budget_interrupt: bool = False
    # Cognition cost tracking (metabolic budgeting)
    cognition_cost_total_usd: float | None = None
    cognition_budget_allocated_usd: float | None = None
    cognition_budget_remaining_usd: float | None = None
    cognition_budget_utilisation: float | None = None
    cognition_budget_importance: str | None = None
    cognition_budget_borrowed_usd: float | None = None
    cognition_budget_early_stop: bool = False
    # Hypothesis tournament context (when the selected policy's source hypothesis
    # is in an active A/B tournament)
    tournament_id: str | None = None
    tournament_hypothesis_id: str | None = None

    # RE training pipeline metadata (Spec §21)
    # re_training_eligible: True when this record has a meaningful outcome
    # signal (slow path with equor verdict). Gated to avoid poisoning the
    # training queue with fast-path or budget-exhausted noise.
    re_training_eligible: bool = False
    # model_used: "claude" | "re" - which model generated the slow-path
    # policies. "claude" until Thompson sampling routes to RE.
    model_used: str = "claude"


# ─── Counterfactual Record ────────────────────────────────────────


class CounterfactualRecord(EOSBaseModel):
    """
    A rejected policy archived for counterfactual regret analysis.

    Created during Nova's slow-path deliberation for every non-selected,
    non-do-nothing policy. Persisted to Neo4j as :Episode:Counterfactual
    nodes, then resolved with regret scores when the actual outcome arrives.
    """

    id: str = Field(default_factory=new_id)
    intent_id: str  # The chosen intent this was an alternative to
    decision_record_id: str  # Link to DecisionRecord for traceability
    goal_id: str = ""
    goal_description: str = ""

    # The rejected policy (serialised snapshot)
    policy_name: str
    policy_type: str = "deliberate"
    policy_description: str = ""
    policy_reasoning: str = ""

    # EFE component scores at decision time
    efe_total: float = 0.0
    estimated_pragmatic_value: float = 0.0  # PragmaticEstimate.score
    estimated_epistemic_value: float = 0.0  # EpistemicEstimate.score
    constitutional_alignment: float = 0.0
    feasibility: float = 0.0
    risk_expected_harm: float = 0.0

    # Chosen policy's scores (for regret computation without re-reading graph)
    chosen_policy_name: str = ""
    chosen_efe_total: float = 0.0

    timestamp: datetime = Field(default_factory=utc_now)

    # Regret (populated later by process_outcome)
    resolved: bool = False
    actual_outcome_success: bool | None = None
    actual_pragmatic_value: float | None = None  # Derived from outcome
    # actual_pragmatic - estimated_pragmatic (positive = counterfactual was better)
    regret: float | None = None
