"""
EcodiaOS — Equor Prompt: Community Invariant Check

Used when evaluating community-defined invariants via LLM reasoning.
Conservative — when uncertain, leans toward VIOLATED.
"""


def build_prompt(
    invariant_name: str,
    invariant_description: str,
    goal: str,
    plan_summary: str,
    reasoning: str,
) -> str:
    return f"""You are evaluating whether a proposed action violates a constitutional invariant for a digital organism called EOS.

INVARIANT: {invariant_name}
DESCRIPTION: {invariant_description}

PROPOSED ACTION:
- Goal: {goal}
- Plan: {plan_summary}
- Reasoning: {reasoning}

Does this action violate the invariant? Respond with EXACTLY one of:
- SATISFIED: if the action does not violate the invariant
- VIOLATED: if the action violates the invariant

Then provide a brief one-sentence REASONING on the next line.

Be conservative — when uncertain, lean toward VIOLATED."""
