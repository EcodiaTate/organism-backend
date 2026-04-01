"""
EcodiaOS - Inspector Semantic Slicer (Phase 4: State Explosion Mitigation)

Implements neurosymbolic backwards program slicing: uses a focused LLM pass
to strip boilerplate, logging, telemetry, and irrelevant branches from an
execution trace before it reaches Z3.

Without slicing, 10,000+ line execution traces produce Z3 constraint systems
too large to solve in finite time. The slicer acts as a scalpel, reducing
context to only the sink→source data-flow path relevant to the attack goal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.inspector.slicer")

_SLICE_THRESHOLD = 1000  # characters; skip slicing for trivially small inputs

_SLICER_SYSTEM_PROMPT = """\
You are a strict Static Analysis engine performing Backwards Program Slicing.

Task: I will give you a large execution trace and a specific attack goal \
(e.g., 'SQL Injection' or 'Auth Bypass'). You must identify the 'Sink' (the \
vulnerable execution point) and trace the data flow backwards to the 'Source' \
(user input).

Action: You MUST DELETE all logging, telemetry, unrelated UI/rendering code, \
dead code, and irrelevant branches. You must RETURN ONLY the minimal, \
contiguous code slice necessary for a formal verification engine to understand \
the exact data flow of the vulnerability. DO NOT alter the actual logic, \
variable names, or syntax of the remaining code.

Output ONLY the sliced code block. No explanation, no markdown fences, no \
commentary - just the raw code."""


class SemanticSlicer:
    """
    Strips noise from execution traces via a focused LLM backwards-slicing pass.

    Designed to sit immediately before the Z3 prover in the hunt pipeline.
    For traces under the threshold it is a no-op; for large traces it reduces
    context to only the sink→source data-flow path relevant to the attack goal.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._log = logger

    async def slice_context(self, context_code: str, attack_goal: str) -> str:
        """
        Perform backwards program slicing on *context_code* for *attack_goal*.

        If the code is below the threshold, returns it unchanged (no LLM call).
        If the LLM call fails for any reason (timeout, API error, empty response),
        returns the original context_code so the pipeline is never blocked.

        Args:
            context_code: Raw source / execution trace to slice.
            attack_goal:  The attacker objective (e.g., "SQL Injection in user input").

        Returns:
            Sliced code string (never shorter than 1 char if input is non-empty).
        """
        if len(context_code) < _SLICE_THRESHOLD:
            return context_code

        user_message = (
            f"Attack Goal: {attack_goal}\n\n"
            f"Execution Trace:\n{context_code}"
        )

        try:
            from clients.llm import Message  # local import - avoids circular

            response = await self._llm.generate(
                system_prompt=_SLICER_SYSTEM_PROMPT,
                messages=[Message(role="user", content=user_message)],
                max_tokens=4096,
            )
            sliced = response.text.strip()
            if not sliced:
                self._log.warning(
                    "slicer_empty_response",
                    attack_goal=attack_goal[:80],
                    original_size=len(context_code),
                )
                return context_code
            return sliced

        except Exception as exc:
            self._log.warning(
                "slicer_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                attack_goal=attack_goal[:80],
                original_size=len(context_code),
            )
            return context_code
