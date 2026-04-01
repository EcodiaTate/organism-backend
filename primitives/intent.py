"""
EcodiaOS - Intent Primitive

The fundamental unit of planned action.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from primitives.common import (
    AutonomyLevel,
    EOSBaseModel,
    Identified,
    ResourceBudget,
    SystemID,
    Timestamped,
    Verdict,
)


class GoalDescriptor(EOSBaseModel):
    """What an intent is trying to achieve."""

    description: str
    target_domain: str = ""
    success_criteria: dict[str, Any] = Field(default_factory=dict)


class Action(EOSBaseModel):
    """A single step in an action sequence."""

    executor: str = ""        # e.g., "communicate.text", "data.store"
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 5000
    rollback: dict[str, Any] | None = None
    parallel_group: str | None = Field(
        default=None,
        description=(
            "Steps sharing the same parallel_group execute concurrently. "
            "None means sequential (default). Nova assigns groups when "
            "steps are provably independent."
        ),
    )


class ActionSequence(EOSBaseModel):
    """An ordered plan of actions with contingencies."""

    steps: list[Action] = Field(default_factory=list)
    contingencies: dict[str, list[Action]] = Field(default_factory=dict)


class EthicalClearance(EOSBaseModel):
    """Result of Equor's review."""

    status: Verdict = Verdict.APPROVED
    equor_trace_id: str | None = None
    modifications: list[str] = Field(default_factory=list)
    reasoning: str = ""


class DecisionTrace(EOSBaseModel):
    """Full explainability trace for why this intent was chosen."""

    reasoning: str = ""
    alternatives_considered: list[dict[str, Any]] = Field(default_factory=list)
    free_energy_scores: dict[str, float] = Field(default_factory=dict)


class Intent(Identified, Timestamped):
    """
    The fundamental unit of planned action.
    Created by Nova, reviewed by Equor, executed by Axon.
    """

    goal: GoalDescriptor
    plan: ActionSequence = Field(default_factory=ActionSequence)
    expected_free_energy: float = 0.0
    ethical_clearance: EthicalClearance = Field(default_factory=EthicalClearance)
    autonomy_level_required: int = AutonomyLevel.ADVISOR
    autonomy_level_granted: int = AutonomyLevel.ADVISOR
    budget: ResourceBudget = Field(default_factory=ResourceBudget)
    priority: float = 0.5
    created_by: SystemID = SystemID.NOVA
    decision_trace: DecisionTrace = Field(default_factory=DecisionTrace)
