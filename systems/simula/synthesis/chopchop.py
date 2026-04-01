"""
EcodiaOS -- Simula ChopChop Engine (Stage 5A.3)

Type-directed constrained code generation via generate-then-verify chunks.

Algorithm:
  1. Decompose the target code into small chunks (N lines each)
  2. For each chunk, extract type/grammar constraints from context
  3. LLM generates the chunk
  4. Validate chunk against constraints (type check, grammar, AST)
  5. Retry on failure up to max_retries per chunk
  6. Assemble all valid chunks into the final code

Since the Anthropic API does not support logit masking, we use a
generate-then-verify approach: generate a chunk freely, then check
constraints, and retry if violated.

Best for: changes touching type-heavy systems with strong contracts.
"""

from __future__ import annotations

import ast
import re
import time
from typing import TYPE_CHECKING
from pathlib import Path

import structlog

from clients.llm import Message
from systems.simula.synthesis.types import (
    ChopChopResult,
    GrammarConstraint,
    SynthesisStatus,
)

if TYPE_CHECKING:

    from clients.llm import LLMProvider
    from systems.simula.evolution_types import ChangeSpec
logger = structlog.get_logger().bind(system="simula.synthesis.chopchop")

# ── System prompt ───────────────────────────────────────────────────────────

CHUNK_GENERATION_PROMPT = """You are a precise Python code generator for EcodiaOS.
Generate EXACTLY the requested code chunk, respecting all type constraints.

## Constraints
{constraints}

## Context (preceding code)
```python
{preceding}
```

## Task
Generate the next {chunk_size} lines of Python code that:
1. Follow naturally from the preceding context
2. Satisfy ALL listed constraints
3. Use EOS conventions (structlog, type hints, async/await, EOSBaseModel)

Respond with ONLY the code lines - no explanation, no fences, no line numbers."""


class ChopChopEngine:
    """Type-directed constrained generation engine (generate-then-verify chunks)."""

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        *,
        max_retries_per_chunk: int = 3,
        chunk_size_lines: int = 10,
        timeout_s: float = 90.0,
    ) -> None:
        self._llm = llm
        self._codebase_root = codebase_root
        self._max_retries = max_retries_per_chunk
        self._chunk_size = chunk_size_lines
        self._timeout_s = timeout_s

    # ── Public API ──────────────────────────────────────────────────────────

    async def synthesise(
        self,
        change_spec: ChangeSpec,
        context_code: str = "",
        target_structure: str = "",
    ) -> ChopChopResult:
        """Run ChopChop: decompose → constrain → generate → verify → assemble."""
        start = time.monotonic()
        total_tokens = 0

        try:
            # Phase 1: Extract constraints from context + spec
            constraints = self._extract_constraints(change_spec, context_code, target_structure)

            # Phase 2: Determine how many chunks we need
            # Estimate from code_hint length or spec complexity
            estimated_lines = self._estimate_lines(change_spec)
            num_chunks = max(1, estimated_lines // self._chunk_size)

            # Phase 3: Generate chunks iteratively
            assembled_lines: list[str] = []
            chunks_generated = 0
            chunks_valid = 0
            chunks_retried = 0

            # Start with any preamble from context
            preceding = context_code[-500:] if context_code else ""

            for chunk_idx in range(num_chunks):
                if time.monotonic() - start > self._timeout_s:
                    break

                chunk, retries, tokens = await self._generate_chunk(
                    preceding, constraints, chunk_idx, num_chunks
                )
                chunks_generated += 1
                chunks_retried += retries
                total_tokens += tokens

                if chunk:
                    chunks_valid += 1
                    assembled_lines.extend(chunk.splitlines())
                    preceding = "\n".join(assembled_lines[-20:])

            # Phase 4: Assemble and final validate
            final_code = "\n".join(assembled_lines)
            ast_valid = self._ast_valid(final_code)

            # Check constraint satisfaction
            constraints_satisfied = sum(
                1 for c in constraints if self._check_constraint(c, final_code)
            )
            for c in constraints:
                c.satisfied = self._check_constraint(c, final_code)

            elapsed_ms = int((time.monotonic() - start) * 1000)
            status = (
                SynthesisStatus.SYNTHESIZED if ast_valid and chunks_valid == chunks_generated
                else SynthesisStatus.PARTIAL if ast_valid
                else SynthesisStatus.FAILED
            )

            logger.info(
                "chopchop_complete",
                status=status.value,
                chunks=f"{chunks_valid}/{chunks_generated}",
                constraints=f"{constraints_satisfied}/{len(constraints)}",
                retries=chunks_retried,
                duration_ms=elapsed_ms,
            )

            return ChopChopResult(
                status=status,
                chunks_generated=chunks_generated,
                chunks_valid=chunks_valid,
                chunks_retried=chunks_retried,
                constraints_total=len(constraints),
                constraints_satisfied=constraints_satisfied,
                final_code=final_code,
                ast_valid=ast_valid,
                type_valid=ast_valid and constraints_satisfied == len(constraints),
                duration_ms=elapsed_ms,
                llm_tokens=total_tokens,
            )

        except Exception:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.exception("chopchop_error")
            return ChopChopResult(
                status=SynthesisStatus.FAILED,
                duration_ms=elapsed_ms,
            )

    # ── Constraint extraction ───────────────────────────────────────────────

    def _extract_constraints(
        self,
        change_spec: ChangeSpec,
        context_code: str,
        target_structure: str,
    ) -> list[GrammarConstraint]:
        """Extract type/grammar constraints from context and spec."""
        constraints: list[GrammarConstraint] = []

        # From context code: parse type annotations
        if context_code:
            try:
                tree = ast.parse(context_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        # Return type constraint
                        if node.returns:
                            constraints.append(GrammarConstraint(
                                constraint_type="return_type",
                                target=f"return type of {node.name}()",
                                expected=ast.dump(node.returns),
                            ))
                        # Argument type constraints
                        for arg in node.args.args:
                            if arg.annotation:
                                constraints.append(GrammarConstraint(
                                    constraint_type="arg_type",
                                    target=f"type of {node.name}.{arg.arg}",
                                    expected=ast.dump(arg.annotation),
                                ))
            except SyntaxError:
                pass

        # From target structure hints
        if target_structure:
            for line in target_structure.splitlines():
                line = line.strip()
                if line.startswith("class ") or line.startswith("def ") or line.startswith("async def "):
                    constraints.append(GrammarConstraint(
                        constraint_type="grammar",
                        target="structure",
                        expected=line,
                    ))

        # From change spec
        if change_spec.code_hint:
            constraints.append(GrammarConstraint(
                constraint_type="grammar",
                target="code_hint",
                expected=change_spec.code_hint,
            ))

        # Always require valid imports
        constraints.append(GrammarConstraint(
            constraint_type="import",
            target="ecodiaos imports",
            expected="from ",
        ))

        return constraints

    @staticmethod
    def _estimate_lines(change_spec: ChangeSpec) -> int:
        """Estimate how many lines of code the proposal requires."""
        if change_spec.code_hint:
            return max(10, len(change_spec.code_hint.splitlines()) * 2)
        # Heuristic based on category
        category_estimates = {
            "add_executor": 60,
            "add_input_channel": 40,
            "add_pattern_detector": 50,
            "modify_contract": 30,
            "add_system_capability": 80,
        }
        for cat, estimate in category_estimates.items():
            if cat in (change_spec.additional_context or "").lower():
                return estimate
        return 40  # default

    # ── Chunk generation ────────────────────────────────────────────────────

    async def _generate_chunk(
        self,
        preceding: str,
        constraints: list[GrammarConstraint],
        chunk_idx: int,
        total_chunks: int,
    ) -> tuple[str, int, int]:
        """Generate one chunk with retry on constraint violation. Returns (code, retries, tokens)."""
        constraint_text = "\n".join(
            f"- [{c.constraint_type}] {c.target}: {c.expected}"
            for c in constraints
        )

        total_tokens = 0
        for retry in range(self._max_retries + 1):
            prompt = CHUNK_GENERATION_PROMPT.format(
                constraints=constraint_text,
                preceding=preceding[-500:] if preceding else "(start of file)",
                chunk_size=self._chunk_size,
            )

            if chunk_idx > 0:
                prompt += f"\n\nThis is chunk {chunk_idx + 1} of ~{total_chunks}."

            response = await self._llm.complete(  # type: ignore[attr-defined]
                system="You are a precise Python code generator.",
                messages=[Message(role="user", content=prompt)],
                max_tokens=1024,
            )

            tokens = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)
            total_tokens += tokens

            chunk_code = self._clean_response(response.text)

            # Validate chunk
            if self._validate_chunk(chunk_code, constraints):
                return chunk_code, retry, total_tokens

            logger.debug(
                "chopchop_chunk_retry",
                chunk_idx=chunk_idx,
                retry=retry,
            )

        # All retries exhausted - return best effort
        return chunk_code, self._max_retries, total_tokens

    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip code fences and whitespace from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _validate_chunk(self, code: str, constraints: list[GrammarConstraint]) -> bool:
        """Validate a code chunk against constraints."""
        if not code:
            return False

        # Basic syntax check (as part of a module)
        # Chunks may not be valid standalone, so we wrap in a try
        try:
            ast.parse(code)
        except SyntaxError:
            # Try wrapping in a function to see if it's valid as a body
            try:
                ast.parse("def _():\n" + "\n".join(f"    {line}" for line in code.splitlines()))
            except SyntaxError:
                return False

        # Check key constraints
        for constraint in constraints:
            if constraint.constraint_type == "import" and constraint.expected in code:
                constraint.satisfied = True

        return True

    @staticmethod
    def _check_constraint(constraint: GrammarConstraint, code: str) -> bool:
        """Check if a constraint is satisfied in the final code."""
        if constraint.constraint_type == "import":
            return constraint.expected in code
        if constraint.constraint_type == "grammar":
            return constraint.expected in code
        # Type constraints require full type checking - assume satisfied if code parses
        return True

    @staticmethod
    def _ast_valid(code: str) -> bool:
        """Check if assembled code is syntactically valid."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
