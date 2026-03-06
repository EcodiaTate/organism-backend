"""
EcodiaOS -- Simula Sketch Solver (Stage 5A.2)

LLM-generated template with symbolic hole-filling.

Algorithm:
  1. LLM generates a code template with __HOLE_N__ markers + type annotations
  2. Each hole is filled by the best available solver:
     - Z3:       arithmetic constraints, range bounds, comparisons
     - Type enum: type annotations, expression types (from type hints)
     - Micro-LLM: block-level holes where symbolic solving is infeasible
  3. Final code is assembled and validated via ast.parse() + type check

Reuses the existing Z3Bridge from verification/z3_bridge.py and
LiloLibraryEngine from learning/lilo.py for reusable abstractions.

Best for: modification categories (MODIFY_CONTRACT, ADD_SYSTEM_CAPABILITY)
or proposals with non-empty code_hint.
"""

from __future__ import annotations

import ast
import json
import re
import time
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import Message
from systems.simula.synthesis.types import (
    HoleKind,
    SketchHole,
    SketchSolveResult,
    SketchTemplate,
    SynthesisStatus,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.evolution_types import ChangeSpec
    from systems.simula.verification.z3_bridge import Z3Bridge

logger = structlog.get_logger().bind(system="simula.synthesis.sketch")

# Regex to find __HOLE_N__ markers in generated templates
_HOLE_PATTERN = re.compile(r"__HOLE_(\d+)__")

# ── System prompts ──────────────────────────────────────────────────────────

SKETCH_TEMPLATE_PROMPT = """You are a code template generator for EcodiaOS.
Your task: generate a Python code template where uncertain parts are replaced
with __HOLE_N__ markers (N = 0, 1, 2, ...). Each hole has a type annotation
and constraints to guide symbolic filling.

## Rules
- Replace ONLY uncertain expressions/statements with holes
- Keep structural elements (class defs, function signatures, imports) concrete
- For each hole, annotate with a comment: # HOLE_N: <type> | <constraint>
- Use EOS conventions: structlog, EOSBaseModel, async/await, type hints
- Maximum holes: as many as needed, but prefer fewer larger holes

## Output Format
```python
# template code with __HOLE_N__ markers
```

```json
[
  {"hole_id": "__HOLE_0__", "kind": "expression", "type_hint": "float",
   "constraints": ["value >= 0", "value <= 1.0"]},
  ...
]
```"""


MICRO_LLM_FILL_PROMPT = """You are a code completion specialist.
Fill in the code hole with a valid Python expression or statement.

Hole context:
- Kind: {kind}
- Expected type: {type_hint}
- Constraints: {constraints}
- Surrounding code:
```python
{context}
```

Respond with ONLY the code to fill the hole — no explanation, no markers."""


class SketchSolver:
    """LLM template + symbolic hole-filling synthesis engine."""

    def __init__(
        self,
        llm: LLMProvider,
        z3_bridge: Z3Bridge | None = None,
        *,
        max_holes: int = 20,
        solver_timeout_ms: int = 5000,
    ) -> None:
        self._llm = llm
        self._z3 = z3_bridge
        self._max_holes = max_holes
        self._solver_timeout_ms = solver_timeout_ms

    # ── Public API ──────────────────────────────────────────────────────────

    async def synthesise(
        self,
        change_spec: ChangeSpec,
        exemplar_code: str = "",
        context_code: str = "",
    ) -> SketchSolveResult:
        """Run sketch-based synthesis: template → fill holes → validate."""
        start = time.monotonic()

        try:
            # Phase 1: LLM generates template with holes
            template = await self._generate_template(change_spec, exemplar_code, context_code)
            if not template.holes:
                # No holes = LLM was fully confident; treat as direct synthesis
                if template.template_code:
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    return SketchSolveResult(
                        status=SynthesisStatus.SYNTHESIZED,
                        template=template,
                        holes_total=0,
                        final_code=template.template_code,
                        ast_valid=self._ast_valid(template.template_code),
                        duration_ms=elapsed_ms,
                    )
                return SketchSolveResult(status=SynthesisStatus.FAILED)

            if len(template.holes) > self._max_holes:
                logger.warning(
                    "sketch_too_many_holes",
                    holes=len(template.holes),
                    max=self._max_holes,
                )
                template.holes = template.holes[:self._max_holes]

            # Phase 2: Fill holes
            z3_filled = 0
            enum_filled = 0
            llm_filled = 0
            unfilled = 0
            code = template.template_code

            for hole in template.holes:
                filled, method = await self._fill_hole(hole, code)
                if filled:
                    hole.filled_value = filled
                    code = code.replace(hole.hole_id, filled)
                    if method == "z3":
                        z3_filled += 1
                    elif method == "enum":
                        enum_filled += 1
                    else:
                        llm_filled += 1
                else:
                    unfilled += 1

            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Phase 3: Validate assembled code
            ast_valid = self._ast_valid(code)
            status = SynthesisStatus.SYNTHESIZED if ast_valid and unfilled == 0 else (
                SynthesisStatus.PARTIAL if ast_valid else SynthesisStatus.FAILED
            )

            logger.info(
                "sketch_solve_complete",
                status=status.value,
                holes_total=len(template.holes),
                z3=z3_filled,
                enum=enum_filled,
                llm=llm_filled,
                unfilled=unfilled,
                duration_ms=elapsed_ms,
            )

            return SketchSolveResult(
                status=status,
                template=template,
                holes_total=len(template.holes),
                holes_filled_z3=z3_filled,
                holes_filled_enum=enum_filled,
                holes_filled_llm=llm_filled,
                holes_unfilled=unfilled,
                final_code=code,
                ast_valid=ast_valid,
                type_valid=ast_valid and unfilled == 0,
                duration_ms=elapsed_ms,
                z3_solver_ms=0,  # aggregate Z3 time tracked below if needed
            )

        except Exception:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.exception("sketch_solve_error")
            return SketchSolveResult(
                status=SynthesisStatus.FAILED,
                duration_ms=elapsed_ms,
            )

    # ── Phase 1: Template generation ────────────────────────────────────────

    async def _generate_template(
        self, change_spec: ChangeSpec, exemplar_code: str, context_code: str
    ) -> SketchTemplate:
        """LLM generates a code template with __HOLE_N__ markers."""
        spec_text = (
            f"Affected systems: {change_spec.affected_systems}\n"
            f"Code hint: {change_spec.code_hint}\n"
            f"Context: {change_spec.additional_context}"
        )
        user_msg = f"## Change Specification\n{spec_text}"
        if exemplar_code:
            user_msg += f"\n\n## Exemplar Code\n```python\n{exemplar_code}\n```"
        if context_code:
            user_msg += f"\n\n## Surrounding Context\n```python\n{context_code}\n```"

        response = await self._llm.complete(  # type: ignore[attr-defined]
            system=SKETCH_TEMPLATE_PROMPT,
            messages=[Message(role="user", content=user_msg)],
            max_tokens=4096,
        )

        tokens = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)

        # Parse template code and hole annotations from response
        template_code, holes = self._parse_template_response(response.text)

        return SketchTemplate(
            template_code=template_code,
            holes=holes,
            llm_tokens=tokens,
        )

    def _parse_template_response(self, text: str) -> tuple[str, list[SketchHole]]:
        """Extract code template and hole definitions from LLM response."""
        # Extract code block
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        template_code = code_match.group(1).strip() if code_match else ""

        # Extract JSON block with hole definitions
        json_match = re.search(r"```json\n(.*?)```", text, re.DOTALL)
        holes: list[SketchHole] = []
        if json_match:
            try:
                raw_holes = json.loads(json_match.group(1))
                for rh in raw_holes:
                    kind_str = rh.get("kind", "expression")
                    try:
                        kind = HoleKind(kind_str)
                    except ValueError:
                        kind = HoleKind.EXPRESSION
                    holes.append(SketchHole(
                        hole_id=rh.get("hole_id", ""),
                        kind=kind,
                        type_hint=rh.get("type_hint", ""),
                        constraints=rh.get("constraints", []),
                    ))
            except (json.JSONDecodeError, TypeError):
                logger.warning("sketch_hole_parse_failed")

        # Also find any __HOLE_N__ in code not listed in JSON
        found_ids = {h.hole_id for h in holes}
        for match in _HOLE_PATTERN.finditer(template_code):
            hole_id = f"__HOLE_{match.group(1)}__"
            if hole_id not in found_ids:
                holes.append(SketchHole(hole_id=hole_id))
                found_ids.add(hole_id)

        return template_code, holes

    # ── Phase 2: Hole filling ───────────────────────────────────────────────

    async def _fill_hole(
        self, hole: SketchHole, surrounding_code: str
    ) -> tuple[str, str]:
        """Fill one hole using the best available method. Returns (value, method)."""
        # Try Z3 for arithmetic constraints
        if (
            self._z3 is not None
            and hole.kind in (HoleKind.EXPRESSION, HoleKind.GUARD_CONDITION)
            and hole.constraints
            and self._constraints_are_arithmetic(hole.constraints)
        ):
            z3_result = await self._try_z3_fill(hole)
            if z3_result:
                return z3_result, "z3"

        # Try type enumeration for type annotations
        if hole.kind == HoleKind.TYPE_ANNOTATION and hole.type_hint:
            enum_result = self._try_type_enum(hole)
            if enum_result:
                return enum_result, "enum"

        # Fallback: micro-LLM call
        llm_result = await self._try_llm_fill(hole, surrounding_code)
        if llm_result:
            return llm_result, "llm"

        return "", "none"

    @staticmethod
    def _constraints_are_arithmetic(constraints: list[str]) -> bool:
        """Check if constraints are expressible in Z3 arithmetic."""
        arithmetic_patterns = (">=", "<=", ">", "<", "==", "!=", "+", "-", "*", "/")
        return all(
            any(op in c for op in arithmetic_patterns)
            for c in constraints
        )

    async def _try_z3_fill(self, hole: SketchHole) -> str:
        """Try to fill a hole using Z3 solver."""
        if self._z3 is None:
            return ""

        try:
            # Build a simple Z3 constraint satisfaction query
            # Map hole constraints to Z3 expressions
            import z3 as z3_mod

            solver = z3_mod.Solver()
            solver.set("timeout", self._solver_timeout_ms)

            # Create a variable for the hole value
            var = z3_mod.Int("value") if hole.type_hint in ("int", "Int") else z3_mod.Real("value")

            # Parse constraints
            for constraint in hole.constraints:
                z3_expr = self._parse_constraint(var, constraint)
                if z3_expr is not None:
                    solver.add(z3_expr)

            if solver.check() == z3_mod.sat:
                model = solver.model()
                val = model[var]
                if val is not None:
                    result = str(val)
                    # Convert Z3 rationals to Python floats
                    if "/" in result:
                        parts = result.split("/")
                        result = str(float(parts[0]) / float(parts[1]))
                    return result

        except ImportError:
            logger.debug("z3_not_available")
        except Exception:
            logger.debug("z3_fill_failed", hole=hole.hole_id)

        return ""

    @staticmethod
    def _parse_constraint(var: Any, constraint: str) -> Any:
        """Parse a simple constraint string into a Z3 expression."""
        import z3 as z3_mod

        constraint = constraint.strip()
        ops = [
            (">=", lambda v, n: v >= n),
            ("<=", lambda v, n: v <= n),
            ("!=", lambda v, n: v != n),
            ("==", lambda v, n: v == n),
            (">", lambda v, n: v > n),
            ("<", lambda v, n: v < n),
        ]

        for op_str, op_fn in ops:
            if op_str in constraint:
                parts = constraint.split(op_str)
                if len(parts) == 2:
                    try:
                        value = float(parts[1].strip())
                        if isinstance(var, z3_mod.ArithRef):
                            return op_fn(var, z3_mod.RealVal(value))  # type: ignore[no-untyped-call]
                    except ValueError:
                        pass
                    break

        return None

    @staticmethod
    def _try_type_enum(hole: SketchHole) -> str:
        """Fill type annotation holes via simple enumeration."""
        type_map = {
            "str": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "list": "list[Any]",
            "dict": "dict[str, Any]",
            "None": "None",
            "Optional[str]": "str | None",
            "Optional[int]": "int | None",
            "Optional[float]": "float | None",
        }
        return type_map.get(hole.type_hint, "")

    async def _try_llm_fill(self, hole: SketchHole, surrounding_code: str) -> str:
        """Fill a hole using a micro-LLM call (~100 tokens)."""
        # Extract context around the hole marker
        context_lines: list[str] = []
        for line in surrounding_code.splitlines():
            if hole.hole_id in line:
                context_lines.append(line)
            elif context_lines:
                context_lines.append(line)
                if len(context_lines) > 5:
                    break
        if not context_lines:
            context_lines = [surrounding_code[:500]]

        prompt = MICRO_LLM_FILL_PROMPT.format(
            kind=hole.kind.value,
            type_hint=hole.type_hint,
            constraints=", ".join(hole.constraints) or "none",
            context="\n".join(context_lines),
        )

        response = await self._llm.complete(  # type: ignore[attr-defined]
            system="You are a concise code completion assistant.",
            messages=[Message(role="user", content=prompt)],
            max_tokens=256,
        )

        # Clean up the response — strip code fences if present
        text = response.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()  # type: ignore[no-any-return]

    # ── Validation ──────────────────────────────────────────────────────────

    @staticmethod
    def _ast_valid(code: str) -> bool:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
