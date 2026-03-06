"""
EcodiaOS — Simula Protocol State Machine Specification

Formal specification of valid state transitions in the Simula proposal pipeline.
Based on the 7-stage pipeline in service.py:
  1. DEDUPLICATE
  2. VALIDATE
  3. SIMULATE
  4. GATE
  5. APPLY
  6. VERIFY
  7. RECORD

This module documents what transitions are valid and what conditions must hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger().bind(system="simula.protocol.state_machine")


class ProposalStatus(StrEnum):
    """Status of a proposal in the Simula pipeline."""
    INITIAL = "initial"
    DEDUPLICATED = "deduplicated"
    VALIDATED = "validated"
    SIMULATED = "simulated"
    GATED = "gated"
    APPLIED = "applied"
    VERIFIED = "verified"
    RECORDED = "recorded"

    # Failure states
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class StateTransition:
    """
    A valid transition in the Simula state machine.

    Defines:
      - From/to states
      - Precondition (English description)
      - Deadline (max time allowed)
      - Side effects (what happens during transition)
    """
    from_state: ProposalStatus
    to_state: ProposalStatus
    condition: str  # English description of precondition
    deadline_ms: int | None  # Time limit for transition (None = no limit)
    side_effects: str = ""  # What happens during this transition


# ─── Simula State Machine Definition ──────────────────────────────────────

SIMULA_STATE_MACHINE: list[StateTransition] = [
    # Stage 1: Deduplication
    StateTransition(
        from_state=ProposalStatus.INITIAL,
        to_state=ProposalStatus.DEDUPLICATED,
        condition="Proposal received by SimulaService.process_proposal()",
        deadline_ms=5000,
        side_effects="ProposalIntelligence checks for duplicates/similar active proposals",
    ),

    # Rejection path: Duplicate detected
    StateTransition(
        from_state=ProposalStatus.DEDUPLICATED,
        to_state=ProposalStatus.REJECTED,
        condition="Duplicate proposal detected by ProposalIntelligence",
        deadline_ms=5000,
        side_effects="Proposal is deduplicated; similar proposal tracked instead",
    ),

    # Stage 2: Validation
    StateTransition(
        from_state=ProposalStatus.DEDUPLICATED,
        to_state=ProposalStatus.VALIDATED,
        condition="Proposal passes initial validation (not forbidden, syntactically correct)",
        deadline_ms=5000,
        side_effects="ChangeSpec is validated; forbidden categories rejected immediately",
    ),

    # Rejection path: Forbidden change
    StateTransition(
        from_state=ProposalStatus.VALIDATED,
        to_state=ProposalStatus.REJECTED,
        condition="Proposal violates Simula iron rules (forbidden categories)",
        deadline_ms=5000,
        side_effects="Proposal rejected; reason logged to EvolutionHistoryManager",
    ),

    # Stage 3: Simulation
    StateTransition(
        from_state=ProposalStatus.VALIDATED,
        to_state=ProposalStatus.SIMULATED,
        condition="ChangeSimulator completes multi-strategy impact prediction",
        deadline_ms=30000,
        side_effects="Blast radius, resource costs, risk levels computed",
    ),

    # Rejection path: Simulation reveals unacceptable risk
    StateTransition(
        from_state=ProposalStatus.SIMULATED,
        to_state=ProposalStatus.REJECTED,
        condition="Simulation reveals risk_level > UNACCEPTABLE or cost > budget",
        deadline_ms=30000,
        side_effects="Proposal rejected; EnrichedSimulationResult logged",
    ),

    # Stage 4: Governance Gate
    StateTransition(
        from_state=ProposalStatus.SIMULATED,
        to_state=ProposalStatus.GATED,
        condition="Proposal passes governance gates (if GOVERNANCE_REQUIRED) OR is SELF_APPLICABLE",
        deadline_ms=None,  # No deadline; may wait for community vote
        side_effects="If gated, authorization is recorded; audit log updated",
    ),

    # Rejection path: Governance gate fails
    StateTransition(
        from_state=ProposalStatus.SIMULATED,
        to_state=ProposalStatus.REJECTED,
        condition="Governance vote fails or timeout reached",
        deadline_ms=None,
        side_effects="Proposal rejected due to lack of approval",
    ),

    # Stage 5: Application
    StateTransition(
        from_state=ProposalStatus.GATED,
        to_state=ProposalStatus.APPLIED,
        condition="ChangeApplicator successfully applies code changes",
        deadline_ms=60000,
        side_effects="Files written; RollbackManager snapshots for recovery",
    ),

    # Rollback path: Apply fails
    StateTransition(
        from_state=ProposalStatus.APPLIED,
        to_state=ProposalStatus.ROLLED_BACK,
        condition="ChangeApplicator.apply() raises exception or succeeds with errors",
        deadline_ms=60000,
        side_effects="RollbackManager restores snapshots; filesystem reverted",
    ),

    # Stage 6: Verification
    StateTransition(
        from_state=ProposalStatus.APPLIED,
        to_state=ProposalStatus.VERIFIED,
        condition="HealthChecker passes all 6 check phases",
        deadline_ms=120000,
        side_effects="Syntax, import, tests, formal verification, Lean proofs all pass",
    ),

    # Rollback path: Health check fails
    StateTransition(
        from_state=ProposalStatus.VERIFIED,
        to_state=ProposalStatus.ROLLED_BACK,
        condition="HealthChecker detects blocking failure in any check phase",
        deadline_ms=120000,
        side_effects="CausalDebugger diagnoses root cause; RollbackManager reverts changes",
    ),

    # Stage 7: Recording
    StateTransition(
        from_state=ProposalStatus.VERIFIED,
        to_state=ProposalStatus.RECORDED,
        condition="EvolutionHistoryManager successfully records immutable history",
        deadline_ms=10000,
        side_effects="Evolution record written to Neo4j; ConfigVersion incremented; analytics updated",
    ),

    # Recovery: Rollback can transition to RECORDED if recovery succeeds
    StateTransition(
        from_state=ProposalStatus.ROLLED_BACK,
        to_state=ProposalStatus.RECORDED,
        condition="Rollback completes successfully",
        deadline_ms=60000,
        side_effects="Rollback recorded as evolution event; trigger repair if desired",
    ),
]


# ─── State Machine Validation ─────────────────────────────────────────────


def get_valid_next_states(current_state: ProposalStatus) -> list[ProposalStatus]:
    """Get all valid next states from a given state."""
    return [
        t.to_state
        for t in SIMULA_STATE_MACHINE
        if t.from_state == current_state
    ]


def is_valid_transition(
    from_state: ProposalStatus,
    to_state: ProposalStatus,
) -> bool:
    """Check if a transition is valid."""
    valid_next = get_valid_next_states(from_state)
    return to_state in valid_next


async def validate_state_transition(
    from_state: ProposalStatus,
    to_state: ProposalStatus,
    context: dict[str, Any],
) -> tuple[bool, str]:
    """
    Verify a transition matches the state machine spec.

    Args:
        from_state: Current state
        to_state: Desired next state
        context: Context dict (for checking preconditions)

    Returns:
        (is_valid, error_message)
    """
    if not is_valid_transition(from_state, to_state):
        return False, f"Invalid transition: {from_state} -> {to_state}"

    # Find the transition spec
    transition = next(
        (t for t in SIMULA_STATE_MACHINE if t.from_state == from_state and t.to_state == to_state),
        None,
    )

    if not transition:
        return False, f"No transition spec found for {from_state} -> {to_state}"

    # Deadline check (if applicable)
    if transition.deadline_ms and context.get("elapsed_ms", 0) > transition.deadline_ms:
        return False, f"Transition deadline exceeded: {context.get('elapsed_ms')}ms > {transition.deadline_ms}ms"

    logger.info(
        "state_transition_validated",
        from_state=from_state,
        to_state=to_state,
        condition=transition.condition,
    )

    return True, ""


# ─── State Machine Query Helpers ──────────────────────────────────────────


def get_transition_deadline(from_state: ProposalStatus, to_state: ProposalStatus) -> int | None:
    """Get the deadline (ms) for a specific transition."""
    transition = next(
        (t for t in SIMULA_STATE_MACHINE if t.from_state == from_state and t.to_state == to_state),
        None,
    )
    return transition.deadline_ms if transition else None


def get_transition_description(from_state: ProposalStatus, to_state: ProposalStatus) -> str:
    """Get the condition description for a transition."""
    transition = next(
        (t for t in SIMULA_STATE_MACHINE if t.from_state == from_state and t.to_state == to_state),
        None,
    )
    return transition.condition if transition else "Unknown transition"


def is_failure_state(state: ProposalStatus) -> bool:
    """Is this a failure/terminal state?"""
    return state in (ProposalStatus.REJECTED, ProposalStatus.ROLLED_BACK)


def is_success_state(state: ProposalStatus) -> bool:
    """Is this a success/terminal state?"""
    return state == ProposalStatus.RECORDED
