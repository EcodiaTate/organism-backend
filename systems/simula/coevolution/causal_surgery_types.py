"""
EcodiaOS -- Causal Self-Surgery Types (Prompt #16)

Episode-level causal DAG types for targeted failure recovery.
Distinct from debugging/types.py which models AST-level causal graphs.

These types model the cognitive decision chain:
  Percept → Belief Update → Goal → Policy → Action → Outcome

and the do-calculus interventions that identify which system variable,
at which decision point, caused a recurring failure pattern.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
    new_id,
    utc_now,
)

# ── Enums ────────────────────────────────────────────────────────────────────


class CognitiveStage(enum.StrEnum):
    """Stages in the cognitive decision chain (one per episode)."""

    PERCEPT = "percept"                    # What was perceived (Atune)
    BELIEF_UPDATE = "belief_update"        # How beliefs changed (Nova belief updater)
    GOAL_SELECTION = "goal_selection"      # Which goal was prioritized (Nova goal manager)
    POLICY_GENERATION = "policy_generation"  # Which policies were considered (Nova)
    POLICY_SELECTION = "policy_selection"   # Which policy was chosen (Nova EFE)
    EQUOR_REVIEW = "equor_review"          # Constitutional check (Equor)
    ACTION_EXECUTION = "action_execution"  # What was executed (Axon)
    OUTCOME = "outcome"                    # What happened (Memory/Atune)


class InterventionDirection(enum.StrEnum):
    """Direction of a proposed parameter intervention."""

    INCREASE = "increase"
    DECREASE = "decrease"
    SET_VALUE = "set_value"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"


# ── Cognitive Causal Chain ───────────────────────────────────────────────────


class CognitiveNode(EOSBaseModel):
    """
    A node in the episode-level cognitive causal DAG.

    Each node represents a snapshot of system state at one stage of the
    cognitive cycle. The ``variables`` dict maps tunable parameter names
    (e.g. ``nova.efe.pragmatic``) to their observed values at this stage.
    """

    node_id: str
    stage: CognitiveStage
    episode_id: str
    system: str  # which EOS system: "atune", "nova", "equor", "axon", etc.
    description: str = ""
    variables: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class CognitiveEdge(EOSBaseModel):
    """
    A directed causal edge between two nodes in the cognitive DAG.

    ``causal_influence`` is the estimated strength of causal transmission
    (used as a path coefficient in the linear SCM approximation).
    ``variables_changed`` tracks deltas between the two nodes.
    """

    from_node: str  # node_id
    to_node: str    # node_id
    causal_influence: float = 1.0  # [0, 1] strength of causal link
    mechanism: str = ""  # e.g. "salience_threshold_exceeded", "efe_selection"
    variables_changed: dict[str, float] = Field(default_factory=dict)


class CognitiveCausalDAG(Identified, Timestamped):
    """
    Complete causal DAG for one episode's cognitive chain.

    Reconstructed from Neo4j episode data (FOLLOWED_BY links,
    Counterfactual ALTERNATIVE_TO relationships, stored properties).
    """

    episode_id: str
    nodes: list[CognitiveNode] = Field(default_factory=list)
    edges: list[CognitiveEdge] = Field(default_factory=list)
    outcome_success: bool = False
    outcome_value: float = 0.0        # pragmatic value of the outcome
    free_energy_start: float = 0.0
    free_energy_end: float = 0.0
    policy_type: str = ""             # from the counterfactual or episode source
    regret: float = 0.0               # from resolved counterfactual


# ── Failure Pattern ──────────────────────────────────────────────────────────


class FailurePattern(EOSBaseModel):
    """
    A detected recurring failure pattern across multiple episodes.

    Detected by clustering high-regret resolved counterfactuals
    by (policy_type, goal_domain) in Neo4j.
    """

    pattern_id: str = Field(default_factory=new_id)
    description: str  # e.g. "high_leverage AND volatile_market => crash"
    condition_predicates: list[str] = Field(default_factory=list)
    matching_episode_ids: list[str] = Field(default_factory=list)
    policy_type: str = ""
    goal_domain: str = ""
    occurrence_count: int = 0
    mean_regret: float = 0.0
    first_observed: datetime = Field(default_factory=utc_now)
    last_observed: datetime = Field(default_factory=utc_now)


# ── Counterfactual / Intervention ────────────────────────────────────────────


class CounterfactualScenario(EOSBaseModel):
    """
    One counterfactual scenario: "what if we intervened at node X?"

    Computes the Average Treatment Effect (ATE):
      ATE = E[outcome | do(X = x')] - E[outcome | do(X = x)]
    """

    scenario_id: str = Field(default_factory=new_id)
    episode_id: str
    intervention_node_id: str          # which CognitiveNode to intervene on
    intervention_stage: CognitiveStage
    parameter: str = ""                # e.g. "nova.efe.pragmatic"
    original_variables: dict[str, float] = Field(default_factory=dict)
    counterfactual_variables: dict[str, float] = Field(default_factory=dict)
    original_outcome_value: float = 0.0
    counterfactual_outcome_value: float = 0.0
    ate: float = 0.0                   # counterfactual - original
    outcome_flipped: bool = False      # did intervention flip success/failure?
    confidence: float = 0.5
    reasoning: str = ""


class CausalInterventionPoint(Identified, Timestamped):
    """
    The key output: identifies exactly which system variable to change
    and with what confidence that change would have prevented failure.

    Translates directly into a surgical EvolutionProposal via Bridge.
    """

    system: str  # e.g. "nova", "atune", "memory"
    parameter: str  # e.g. "nova.efe.pragmatic", "atune.head.risk.weight"
    direction: InterventionDirection
    suggested_value: float | None = None
    current_value: float | None = None

    # Evidence
    episodes_analyzed: int = 0
    episodes_where_intervention_helps: int = 0
    intervention_success_rate: float = 0.0  # helped / analyzed
    mean_ate: float = 0.0
    confidence: float = 0.5

    # Cost estimation (optional)
    estimated_normal_cost: float = 0.0  # outcome degradation in normal conditions

    # Source
    failure_pattern_id: str = ""
    counterfactual_scenarios: list[CounterfactualScenario] = Field(
        default_factory=list
    )
    reasoning: str = ""

    # Conditional constraint (for targeted proposals)
    condition: str = ""  # e.g. "when market.volatility > 0.7"
    is_conditional: bool = False


# ── Aggregate Result ─────────────────────────────────────────────────────────


class CausalSurgeryResult(Identified, Timestamped):
    """Aggregate result of one causal surgery analysis pass."""

    failure_pattern: FailurePattern
    dags_built: int = 0
    counterfactuals_evaluated: int = 0
    intervention_points: list[CausalInterventionPoint] = Field(
        default_factory=list
    )
    best_intervention: CausalInterventionPoint | None = None
    total_duration_ms: int = 0
    llm_tokens_used: int = 0
