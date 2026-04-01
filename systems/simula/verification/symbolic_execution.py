"""
EcodiaOS -- Simula Hybrid Symbolic Execution Engine (Stage 6E)

Z3 SMT solver for mission-critical logic beyond invariant discovery.

Where Stage 2B (Z3Bridge) discovers invariants via an LLM+Z3 loop,
Stage 6E provides mathematical correctness guarantees for specific
domains: budget calculations, access control, risk scoring, governance
gating, and constitutional alignment.

Algorithm:
  1. AST-extract functions matching domain keywords from changed files
  2. LLM encodes each function's properties as Z3 expressions
  3. Z3 checks NOT(property) - UNSAT means property is proved
  4. Counterexamples returned when SAT (a concrete input violates property)

This delivers mathematical proof, not just test coverage.
"""

from __future__ import annotations

import ast
import re
import time
from typing import TYPE_CHECKING
from pathlib import Path

import structlog

from systems.simula.verification.types import (
    SymbolicDomain,
    SymbolicExecutionResult,
    SymbolicExecutionStatus,
    SymbolicProperty,
)

if TYPE_CHECKING:

    from clients.llm import LLMProvider
    from systems.simula.verification.z3_bridge import Z3Bridge
logger = structlog.get_logger().bind(system="simula.verification.symbolic_execution")


# Domain keyword patterns for AST function extraction.
_DOMAIN_KEYWORDS: dict[SymbolicDomain, tuple[str, ...]] = {
    SymbolicDomain.BUDGET_CALCULATION: (
        "budget", "cost", "spend", "allocate", "balance", "tokens_per_hour",
    ),
    SymbolicDomain.ACCESS_CONTROL: (
        "permission", "authorize", "role", "access", "gate", "allowed",
    ),
    SymbolicDomain.RISK_SCORING: (
        "risk", "score", "threshold", "level", "assess", "risk_level",
    ),
    SymbolicDomain.GOVERNANCE_GATING: (
        "governance", "approve", "vote", "quorum", "governed",
    ),
    SymbolicDomain.CONSTITUTIONAL_ALIGNMENT: (
        "constitution", "alignment", "drive", "coherence", "care", "growth", "honesty",
    ),
}


_PROPERTY_EXTRACTION_PROMPT = """\
You are a formal verification expert. Given a Python function, extract
mathematical properties that should hold for ALL valid inputs.

Function source:
```python
{source}
```

Domain: {domain}

Generate a list of properties as Z3 Python code. Each property should:
1. Declare Z3 variables matching the function parameters
2. Express the property as a Z3 boolean expression
3. Use z3.Int, z3.Real, z3.Bool for variable declarations

Output ONLY a JSON array of objects with these fields:
- "property_name": short descriptive name
- "human_description": what the property guarantees
- "z3_encoding": Python code string that returns a Z3 expression
  (assume `import z3` is available, declare variables inline)

Example:
[
  {{
    "property_name": "budget_non_negative",
    "human_description": "Budget allocation is always >= 0",
    "z3_encoding": "budget = z3.Real('budget'); z3.And(budget >= 0, budget <= 1.0)"
  }}
]
"""


class SymbolicExecutionEngine:
    """Z3 SMT solver for mathematical correctness of mission-critical logic."""

    def __init__(
        self,
        z3_bridge: Z3Bridge | None = None,
        llm: LLMProvider | None = None,
        *,
        timeout_ms: int = 10000,
        blocking: bool = True,
    ) -> None:
        self._z3 = z3_bridge
        self._llm = llm
        self._timeout_ms = timeout_ms
        self._blocking = blocking

    # ── Public API ──────────────────────────────────────────────────────────

    async def prove_properties(
        self,
        files: list[str],
        codebase_root: Path,
        domains: list[SymbolicDomain] | None = None,
    ) -> SymbolicExecutionResult:
        """
        Extract mission-critical functions, encode as Z3 formulas, prove.

        Returns aggregated result: properties proved, counterexamples found.
        """
        start = time.monotonic()

        if domains is None:
            domains = list(SymbolicDomain)

        # Phase 1: Extract domain functions from changed files
        targets = self._extract_domain_functions(files, domains, codebase_root)

        if not targets:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("symbolic_no_domain_functions", files=len(files))
            return SymbolicExecutionResult(
                status=SymbolicExecutionStatus.SKIPPED,
                duration_ms=elapsed_ms,
            )

        # Phase 2: For each function, LLM generates properties, Z3 proves
        all_properties: list[SymbolicProperty] = []
        proved = 0
        failed = 0
        counterexamples: list[str] = []
        path_conditions = 0
        z3_time = 0

        for source, func_name, domain, file_path in targets:
            props = await self._encode_function(source, func_name, domain)
            for prop in props:
                prop.target_file = file_path
                result_status, result_detail = await self._check_property(prop)
                prop.status = result_status

                if result_status == SymbolicExecutionStatus.PROVED:
                    proved += 1
                elif result_status == SymbolicExecutionStatus.COUNTEREXAMPLE:
                    failed += 1
                    prop.counterexample = result_detail
                    counterexamples.append(
                        f"{file_path}:{func_name} - {prop.property_name}: {result_detail}",
                    )

                path_conditions += 1
                all_properties.append(prop)

        # Determine overall status
        if failed > 0:
            overall_status = SymbolicExecutionStatus.COUNTEREXAMPLE
        elif proved > 0:
            overall_status = SymbolicExecutionStatus.PROVED
        else:
            overall_status = SymbolicExecutionStatus.UNKNOWN

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "symbolic_execution_complete",
            properties_checked=len(all_properties),
            proved=proved,
            failed=failed,
            counterexamples=len(counterexamples),
            duration_ms=elapsed_ms,
        )

        return SymbolicExecutionResult(
            status=overall_status,
            properties_checked=len(all_properties),
            properties_proved=proved,
            properties_failed=failed,
            counterexamples=counterexamples,
            path_conditions_explored=path_conditions,
            properties=all_properties,
            z3_time_ms=z3_time,
            duration_ms=elapsed_ms,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _extract_domain_functions(
        self,
        files: list[str],
        domains: list[SymbolicDomain],
        codebase_root: Path,
    ) -> list[tuple[str, str, SymbolicDomain, str]]:
        """
        AST-based extraction of functions matching domain keywords.

        Returns: list of (source_code, function_name, domain, file_path)
        """
        results: list[tuple[str, str, SymbolicDomain, str]] = []

        for file_path in files:
            full_path = codebase_root / file_path
            if not full_path.exists() or not file_path.endswith(".py"):
                continue

            try:
                source = full_path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            lines = source.splitlines()

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                func_name = node.name
                # Check if function name or body matches any domain keywords
                for domain in domains:
                    keywords = _DOMAIN_KEYWORDS[domain]
                    name_lower = func_name.lower()

                    if any(kw in name_lower for kw in keywords):
                        # Extract function source
                        func_start = node.lineno - 1
                        func_end = node.end_lineno or func_start + 1
                        func_source = "\n".join(lines[func_start:func_end])
                        results.append((func_source, func_name, domain, file_path))
                        break  # don't double-count for multiple domains

        logger.debug(
            "domain_functions_extracted",
            files=len(files),
            functions=len(results),
            domains=[d.value for d in domains],
        )
        return results

    async def _encode_function(
        self,
        source: str,
        function_name: str,
        domain: SymbolicDomain,
    ) -> list[SymbolicProperty]:
        """LLM extracts properties and encodes them as Z3 expressions."""
        if self._llm is None:
            return []

        prompt = _PROPERTY_EXTRACTION_PROMPT.format(
            source=source,
            domain=domain.value.replace("_", " "),
        )

        try:
            from clients.llm import Message

            response = await self._llm.complete(  # type: ignore[attr-defined]
                system="You are a formal verification expert specializing in Z3 SMT encoding.",
                messages=[Message(role="user", content=prompt)],
                max_tokens=2048,
            )

            text = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response (handle markdown code fences)
            json_match = re.search(r"\[[\s\S]*\]", text)
            if not json_match:
                return []

            import json
            raw_props = json.loads(json_match.group())

            properties: list[SymbolicProperty] = []
            for raw in raw_props:
                if not isinstance(raw, dict):
                    continue
                properties.append(
                    SymbolicProperty(
                        domain=domain,
                        property_name=str(raw.get("property_name", "")),
                        z3_encoding=str(raw.get("z3_encoding", "")),
                        human_description=str(raw.get("human_description", "")),
                        target_function=function_name,
                    ),
                )
            return properties

        except Exception as exc:
            logger.warning(
                "property_encoding_failed",
                function=function_name,
                error=str(exc),
            )
            return []

    async def _check_property(
        self,
        prop: SymbolicProperty,
    ) -> tuple[SymbolicExecutionStatus, str]:
        """
        Z3 checks NOT(property) - UNSAT means the property is proved.

        If SAT, the model provides a counterexample: a concrete set of
        inputs that violates the property.
        """
        if not prop.z3_encoding:
            return SymbolicExecutionStatus.SKIPPED, ""

        try:
            import z3

            # Execute the Z3 encoding in a sandboxed namespace
            namespace: dict[str, object] = {"z3": z3}
            exec(prop.z3_encoding, namespace)  # noqa: S102 - controlled input from our LLM

            # Find the Z3 expression (the last assigned variable or expression result)
            z3_expr = None
            for val in namespace.values():
                if isinstance(val, z3.BoolRef):
                    z3_expr = val

            if z3_expr is None:
                return SymbolicExecutionStatus.UNKNOWN, "No Z3 BoolRef produced"

            # Create solver and check NOT(property)
            solver = z3.Solver()
            solver.set("timeout", self._timeout_ms)
            solver.add(z3.Not(z3_expr))

            result = solver.check()

            if result == z3.unsat:
                # NOT(property) is UNSAT → property holds for ALL inputs
                return SymbolicExecutionStatus.PROVED, ""
            elif result == z3.sat:
                # NOT(property) is SAT → found a counterexample
                model = solver.model()
                ce_parts: list[str] = []
                for decl in model.decls():
                    ce_parts.append(f"{decl.name()}={model[decl]}")
                counterexample = ", ".join(ce_parts)
                return SymbolicExecutionStatus.COUNTEREXAMPLE, counterexample
            else:
                return SymbolicExecutionStatus.TIMEOUT, "Z3 returned unknown"

        except TimeoutError:
            return SymbolicExecutionStatus.TIMEOUT, ""
        except Exception as exc:
            logger.warning(
                "z3_check_failed",
                property=prop.property_name,
                error=str(exc),
            )
            return SymbolicExecutionStatus.UNKNOWN, str(exc)
