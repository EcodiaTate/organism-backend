"""
EcodiaOS - Inspector Invariant Inferencer (Autonomous Invariant Inference)

Uses an LLM to deduce the implicit business-logic invariants of a code slice,
then translates each invariant into a precise Z3 violation goal.  These goals
are injected into the attack-goal list before the VulnerabilityProver runs,
giving the engine a per-target understanding of *intended* behaviour rather
than relying solely on pre-defined generic attack patterns.

Workflow:
  1. Feed sliced context_code to the LLM using the Formal Methods persona.
  2. Parse the returned JSON list of invariant objects.
  3. Return the z3_violation_goal strings - one per inferred invariant.

Failure handling:
  - Any exception (network, JSON decode, schema mismatch) is caught and logged.
  - The method always returns a list; on failure it returns [] so the pipeline
    continues with its standard goals unmodified.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.inspector.inference")

_INFERENCER_SYSTEM_PROMPT = """\
You are a PhD-level Formal Methods Mathematician specializing in Business Logic \
Vulnerabilities.

Task: I will provide a slice of backend source code. You must identify the core \
state variables (e.g., balances, permissions, counts) and deduce the implicit \
'Invariants' - the mathematical laws that the developer assumed would always be \
true (e.g., 'a user cannot withdraw more than their balance', \
'price cannot be negative').

Output Format: Return ONLY a strict JSON list of objects. Each object MUST \
contain exactly these three keys:
  - "invariant_name"      (string): A short, unique identifier for the invariant.
  - "business_logic_rule" (string): A plain-English description of the rule.
  - "z3_violation_goal"   (string): Exact instructions on how a Z3 solver should \
model the *violation* of this rule - including which variables to declare, which \
constraints to assert, and what the solver should find (e.g., \
'Declare Int variables balance and withdrawal. Assert withdrawal > balance. \
Assert balance >= 0. Check satisfiability - a SAT result proves a user can \
overdraw their account.').

Do NOT include any explanation, markdown fences, or commentary outside the JSON \
array. The response must be valid JSON that can be parsed directly by \
json.loads()."""


class InvariantInferencer:
    """
    Infers implicit business-logic invariants from a code slice and translates
    each one into a Z3 violation goal string for the VulnerabilityProver.

    Constructed with an LLMProvider; can be injected into InspectorService as an
    optional component - the pipeline degrades gracefully when absent.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._log = logger

    async def infer_vulnerability_goals(self, sliced_context: str) -> list[str]:
        """
        Ask the LLM to identify business-logic invariants in *sliced_context*
        and return the z3_violation_goal string for each discovered invariant.

        Args:
            sliced_context: Pre-sliced source code to analyse (post SemanticSlicer).

        Returns:
            List of z3_violation_goal strings.  Empty list on any failure.
        """
        if not sliced_context.strip():
            return []

        user_message = (
            "Analyse the following backend source code and return the JSON array "
            "of invariant objects as specified:\n\n"
            f"{sliced_context}"
        )

        try:
            from clients.llm import Message  # local import - avoids circular

            response = await self._llm.generate(
                system_prompt=_INFERENCER_SYSTEM_PROMPT,
                messages=[Message(role="user", content=user_message)],
                max_tokens=2048,
            )

            raw = response.text.strip()
            if not raw:
                self._log.warning(
                    "inferencer_empty_response",
                    context_size=len(sliced_context),
                )
                return []

            return self._parse_goals(raw)

        except Exception as exc:
            self._log.warning(
                "inferencer_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                context_size=len(sliced_context),
            )
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_goals(self, raw: str) -> list[str]:
        """
        Parse the raw LLM response into a list of z3_violation_goal strings.

        Accepts two shapes:
          - A bare JSON array:  [{...}, {...}]
          - A JSON object with a top-level list value (some models wrap the array)

        Returns an empty list rather than raising on any parse or schema error.
        """
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._log.warning(
                "inferencer_json_decode_error",
                error=str(exc),
                raw_preview=raw[:200],
            )
            return []

        # Unwrap single-key object wrapper if the model returned {"invariants": [...]}
        if isinstance(parsed, dict):
            # Take the first list value found
            for value in parsed.values():
                if isinstance(value, list):
                    parsed = value
                    break
            else:
                self._log.warning(
                    "inferencer_unexpected_json_shape",
                    raw_preview=raw[:200],
                )
                return []

        if not isinstance(parsed, list):
            self._log.warning(
                "inferencer_not_a_list",
                actual_type=type(parsed).__name__,
                raw_preview=raw[:200],
            )
            return []

        goals: list[str] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            goal = item.get("z3_violation_goal", "")
            if isinstance(goal, str) and goal.strip():
                goals.append(goal.strip())

        return goals
