"""
EcodiaOS - Equor Prompt: Community Invariant Check

Determines whether a proposed action violates a constitutional invariant.
"""


def build_prompt(
    invariant_name: str,
    invariant_description: str,
    goal: str,
    plan_summary: str,
    reasoning: str,
) -> str:
    return f"""Constitutional invariant check.

INVARIANT: {invariant_name}
DESCRIPTION: {invariant_description}

PROPOSED ACTION:
- Goal: {goal}
- Plan: {plan_summary}
- Reasoning: {reasoning}

Read the intent carefully. Consider what the action actually does, not surface-level word matching.
Consider edge cases, second-order effects, and indirect harms.

Start your response with SATISFIED or VIOLATED, then explain your reasoning."""
