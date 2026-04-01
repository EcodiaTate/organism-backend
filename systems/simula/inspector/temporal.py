"""
EcodiaOS - Inspector Temporal Engine (Phase 3)

Proves race conditions and double-spend vulnerabilities by modelling
TWO concurrent threads in Z3's integer/boolean arithmetic.

The insight: a transactional function is vulnerable to a race condition
when two threads can both read the pre-write state, both pass a guard
check, and both commit - yielding a final state that violates a critical
invariant (e.g., balance goes negative, inventory count goes below zero).

The encoding does NOT use Z3 quantifiers or ForAll - it stays purely
propositional so the solver terminates quickly.

Pipeline per attack surface:
  1. LLM reads context_code → writes a two-thread Z3 script (as JSON)
  2. We evaluate the script in a sandboxed z3 namespace
  3. If SAT → the invariant can be violated under concurrent access
  4. LLM generates a Python asyncio reproduction script
  5. Return VulnerabilityReport(vulnerability_class=RACE_CONDITION)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import Message
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.verification.z3_bridge import Z3Bridge

logger = structlog.get_logger().bind(system="simula.inspector.temporal")

# ── LLM timeouts ─────────────────────────────────────────────────────────────

_ENCODING_TIMEOUT_S = 60
_POC_TIMEOUT_S = 60

# ── Concurrency encoding system prompt ───────────────────────────────────────

_CONCURRENCY_ENCODING_SYSTEM_PROMPT = """\
You are a Formal Verification Engineer specialising in concurrent systems.

I will give you source code for a transactional function (e.g. a withdrawal,
purchase, stock-decrement, or any read-modify-write operation).

Your task: write a Z3 Python expression that models TWO concurrent threads
(Thread 1 and Thread 2) executing this function simultaneously and proves that
both can succeed while violating a critical invariant.

## Encoding Rules

1. Declare INITIAL STATE variables (e.g. initial_balance, initial_stock).
   Use z3.Int for integer quantities, z3.Real for currency amounts.

2. Declare what EACH THREAD READS before any write occurs:
     t1_read  - the value Thread 1 reads from shared state
     t2_read  - the value Thread 2 reads from shared state
   The interleave constraint: both threads read BEFORE either writes.
   Express this as: t1_read == initial_value, t2_read == initial_value

3. Model each thread's GUARD CHECK (e.g. "sufficient balance"):
     t1_guard = (t1_read >= cost)
     t2_guard = (t2_read >= cost)

4. Model each thread's COMPUTED WRITE (the new state each would commit):
     t1_write = initial_value - cost   (only meaningful when guard passes)
     t2_write = initial_value - cost

5. The SAT GOAL - assert ALL of:
   a. Both guards pass (both threads believe they can proceed)
   b. Both threads commit (t1_success == True, t2_success == True)
   c. The final state violates the invariant
      (e.g. t1_write < 0  OR  t1_write + t2_write - initial_value < 0)

6. Use ONLY: z3.And, z3.Or, z3.Not, z3.Implies, z3.If, and comparison
   operators on declared variables. Do NOT use z3.ForAll or z3.Exists.

## Output Format

Respond with ONLY a JSON object - no markdown, no extra text:
{
  "variable_declarations": {
    "initial_balance": "Real",
    "cost": "Real",
    "t1_read": "Real",
    "t2_read": "Real",
    "t1_write": "Real",
    "t2_write": "Real",
    "t1_success": "Bool",
    "t2_success": "Bool"
  },
  "z3_expression": "z3.And(t1_read == initial_balance, t2_read == initial_balance, t1_read >= cost, t2_read >= cost, t1_success == True, t2_success == True, t1_write == initial_balance - cost, t2_write == initial_balance - cost, initial_balance - 2*cost < 0, initial_balance >= 0, cost > 0)",
  "invariant_violated": "final balance goes negative when two withdrawals race",
  "reasoning": "Both threads read initial_balance before either writes; both see sufficient funds and both commit, but the real final balance (initial - 2*cost) is negative."
}"""


# ── PoC script generation prompt ─────────────────────────────────────────────

_RACE_POC_SYSTEM_PROMPT = """\
You are a security verification engineer generating async race-condition
reproduction scripts from Z3 concurrency counterexamples.

Given:
- The Z3 counterexample (concrete variable assignments)
- The attack surface (entry point, HTTP method, route)
- The invariant that was violated

Generate a Python script that:
1. Fires TWO simultaneous requests to the endpoint using asyncio.gather.
2. Asserts that both requests "succeeded" (e.g. HTTP 200) when under correct
   concurrency control only ONE should succeed.
3. Includes a timeout of 5 seconds per request.
4. Targets localhost only.

## Rules
- Use ONLY: asyncio, httpx, json, sys, time (no other imports).
- TARGET_URL defaults to "http://localhost:8000"  # Run against local dev server only
- Structure:
    MODULE DOCSTRING labelled "Security Unit Test: Race Condition - <invariant>"
    TARGET_URL constant
    async def build_requests() -> tuple[httpx.Request, httpx.Request]
    async def run_race_test() -> None  - sends both, asserts, prints result
    if __name__ == "__main__": asyncio.run(run_race_test())
- The assertion must FAIL (raise AssertionError) when both requests succeed
  AND the server response indicates the invariant was violated (e.g., both
  return 200 with success=True).
- Add a comment above the assert explaining the invariant.
- Handle httpx.TimeoutException and httpx.ConnectError gracefully (print,
  do not raise).
- The script must be syntactically valid Python 3.10+.

Respond with ONLY the Python source code. No markdown fences."""


# ── ConcurrencyProver ─────────────────────────────────────────────────────────


_MAX_REFLEXION_RETRIES = 3


class ConcurrencyProver:
    """
    Proves race conditions and double-spend vulnerabilities using Z3.

    Models two concurrent threads reading shared state before either
    writes, then checks whether both can complete while violating a
    critical invariant (balance < 0, stock < 0, count > limit, etc.).
    """

    def __init__(
        self,
        llm: LLMProvider,
        z3_bridge: Z3Bridge,
        *,
        check_timeout_ms: int = 10_000,
    ) -> None:
        self._llm = llm
        self._z3 = z3_bridge
        self._check_timeout_ms = check_timeout_ms
        self._log = logger

    async def prove_race_condition(
        self,
        attack_surface: AttackSurface,
        target_url: str = "unknown",
    ) -> VulnerabilityReport | None:
        """
        Attempt to prove a race condition exists for this attack surface.

        Returns a VulnerabilityReport if a race condition is provable,
        None if the surface appears safe or the model is inconclusive.

        Args:
            attack_surface: The surface to analyse (must have context_code).
            target_url: Target URL for tagging the report.
        """
        start = time.monotonic()
        log = self._log.bind(
            entry_point=attack_surface.entry_point,
            surface_type=attack_surface.surface_type.value,
            target_url=target_url,
        )

        if not attack_surface.context_code:
            log.debug("temporal_skipped_no_context")
            return None

        log.info("temporal_proof_started")

        # Step 1: LLM encodes the two-thread race as a Z3 expression
        encoding = await self._encode_race_condition(attack_surface)
        if encoding is None:
            log.warning("temporal_encoding_failed")
            return None

        z3_expr, var_decls, invariant_violated, reasoning = encoding

        # Step 2: Run Z3 - SAT means the race condition is provable
        status, counterexample = self._check_race_constraints(z3_expr, var_decls)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if status != "sat":
            log.info(
                "temporal_proof_inconclusive",
                status=status,
                detail=counterexample[:200],
                elapsed_ms=elapsed_ms,
            )
            return None

        log.info(
            "race_condition_proved",
            counterexample=counterexample[:200],
            invariant_violated=invariant_violated,
            elapsed_ms=elapsed_ms,
        )

        # Escalate severity for financial/smart-contract surfaces
        severity = _severity_for_surface(attack_surface)

        report = VulnerabilityReport(
            target_url=target_url,
            vulnerability_class=VulnerabilityClass.RACE_CONDITION,
            severity=severity,
            attack_surface=attack_surface,
            attack_goal=(
                f"Race condition: two concurrent threads both complete the "
                f"transaction while violating invariant - {invariant_violated}"
            ),
            z3_counterexample=counterexample,
            z3_constraints_code=z3_expr,
        )

        # Step 3: Generate the asyncio reproduction script
        poc_code = await self._generate_race_poc(report, invariant_violated)
        if poc_code:
            report.proof_of_concept_code = poc_code

        log.info(
            "temporal_report_built",
            vuln_id=report.id,
            severity=severity.value,
            has_poc=report.has_poc,
        )
        return report

    # ── Encoding ─────────────────────────────────────────────────────────────

    async def _encode_race_condition(
        self,
        surface: AttackSurface,
    ) -> tuple[str, dict[str, str], str, str] | None:
        """
        Reflexion loop: ask the LLM to encode the two-thread race as Z3
        constraints, attempt to execute the result, and feed any error back
        as a follow-up user message so the LLM can correct it.

        Catches SyntaxError, NameError, Z3Exception, and general eval errors.
        Maintains full conversation history across all retry attempts.

        Returns (z3_expression, variable_declarations, invariant_violated,
                 reasoning) or None if all retries are exhausted.
        """
        messages: list[Message] = [
            Message(role="user", content=_build_encoding_prompt(surface))
        ]

        for attempt in range(1, _MAX_REFLEXION_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    self._llm.generate(
                        system_prompt=_CONCURRENCY_ENCODING_SYSTEM_PROMPT,
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.2 if attempt == 1 else 0.1,
                    ),
                    timeout=_ENCODING_TIMEOUT_S,
                )
            except TimeoutError:
                self._log.warning(
                    "temporal_encoding_timeout",
                    attempt=attempt,
                    entry_point=surface.entry_point,
                    timeout_s=_ENCODING_TIMEOUT_S,
                )
                return None
            except Exception as exc:
                self._log.error(
                    "temporal_encoding_llm_error",
                    attempt=attempt,
                    entry_point=surface.entry_point,
                    error=str(exc),
                )
                return None

            assistant_text = response.text
            messages.append(Message(role="assistant", content=assistant_text))

            # Try to parse the JSON structure from the LLM response
            parsed = _parse_encoding_response(assistant_text, self._log)
            if parsed is None:
                error_msg = (
                    "Execution failed with the following error: "
                    "Your response could not be parsed as a valid JSON object. "
                    "Please correct the Z3 script and output the fixed version "
                    "as a JSON object with exactly these keys: "
                    "variable_declarations, z3_expression, invariant_violated, reasoning."
                )
                self._log.debug(
                    "temporal_reflexion_parse_error",
                    attempt=attempt,
                    max_retries=_MAX_REFLEXION_RETRIES,
                    entry_point=surface.entry_point,
                )
                messages.append(Message(role="user", content=error_msg))
                continue

            z3_expr, var_decls, invariant_violated, reasoning = parsed

            # Attempt to execute the Z3 expression to catch runtime errors
            execution_error = _validate_z3_expression(z3_expr, var_decls)
            if execution_error is None:
                # Success - expression is syntactically and semantically valid
                if attempt > 1:
                    self._log.info(
                        "temporal_reflexion_succeeded",
                        attempt=attempt,
                        entry_point=surface.entry_point,
                    )
                return parsed

            # Feed the execution error back to the LLM for self-correction
            error_msg = (
                f"Execution failed with the following error: {execution_error}. "
                "Please correct the Z3 script and output the fixed version. "
                "Common issues:\n"
                "- All variable names in z3_expression must exactly match "
                "those declared in variable_declarations\n"
                "- Use z3.And, z3.Or, z3.Not - not Python and/or/not\n"
                "- Do NOT use z3.ForAll or z3.Exists\n"
                "- Bool variables use == True/False, not bare references\n"
                "Respond with ONLY a corrected JSON object."
            )
            self._log.debug(
                "temporal_reflexion_execution_error",
                attempt=attempt,
                max_retries=_MAX_REFLEXION_RETRIES,
                error=execution_error,
                entry_point=surface.entry_point,
            )
            messages.append(Message(role="user", content=error_msg))

        self._log.error(
            "prover_reflexion_failed",
            max_retries=_MAX_REFLEXION_RETRIES,
            entry_point=surface.entry_point,
        )
        return None

    # ── Z3 constraint check ───────────────────────────────────────────────────

    def _check_race_constraints(
        self,
        z3_expr_code: str,
        variable_declarations: dict[str, str],
    ) -> tuple[str, str]:
        """
        Evaluate the two-thread Z3 expression and check satisfiability.

        SAT means both threads can complete while violating the invariant.

        Returns:
            ("sat", counterexample) | ("unsat", "") | ("unknown", detail)
        """
        try:
            import z3 as z3_lib
        except ImportError:
            return "unknown", "z3-solver not installed"

        solver = z3_lib.Solver()
        solver.set("timeout", self._check_timeout_ms)

        z3_vars: dict[str, Any] = {}
        for name, z3_type in variable_declarations.items():
            if z3_type == "Int":
                z3_vars[name] = z3_lib.Int(name)
            elif z3_type == "Real":
                z3_vars[name] = z3_lib.Real(name)
            elif z3_type == "Bool":
                z3_vars[name] = z3_lib.Bool(name)
            else:
                z3_vars[name] = z3_lib.Real(name)

        namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
        try:
            expr = eval(z3_expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:
            return "unknown", f"expression eval error: {exc}"

        if not isinstance(expr, z3_lib.BoolRef):
            return "unknown", "expression did not produce a z3.BoolRef"

        solver.add(expr)
        result = solver.check()

        if result == z3_lib.sat:
            model = solver.model()
            parts: list[str] = []
            try:
                for decl in model.decls():
                    parts.append(f"{decl.name()}={model[decl]}")
            except Exception:
                pass
            return "sat", ", ".join(sorted(parts))

        if result == z3_lib.unsat:
            return "unsat", ""

        return "unknown", "solver timeout or unknown"

    # ── PoC generation ────────────────────────────────────────────────────────

    async def _generate_race_poc(
        self,
        report: VulnerabilityReport,
        invariant_violated: str,
    ) -> str:
        """
        Generate an asyncio race-condition reproduction script.

        The script fires two simultaneous requests via asyncio.gather and
        asserts that the server enforces the invariant under load.
        """
        surface = report.attack_surface
        method = (surface.http_method or "POST").upper()
        route = surface.route_pattern or f"/{surface.entry_point}"

        user_prompt = "\n".join([
            "## Race Condition Proof",
            f"Entry point  : {surface.entry_point}",
            f"HTTP method  : {method}",
            f"Route pattern: {route}",
            f"Invariant violated: {invariant_violated}",
            "",
            "## Z3 Counterexample",
            report.z3_counterexample,
            "",
            "Generate the async Python reproduction script as specified.",
        ])

        try:
            response = await asyncio.wait_for(
                self._llm.generate(
                    system_prompt=_RACE_POC_SYSTEM_PROMPT,
                    messages=[Message(role="user", content=user_prompt)],
                    max_tokens=1536,
                    temperature=0.2,
                ),
                timeout=_POC_TIMEOUT_S,
            )
        except (TimeoutError, Exception) as exc:
            self._log.warning(
                "temporal_poc_generation_failed",
                vuln_id=report.id,
                error=str(exc),
            )
            return ""

        code = response.text.strip()
        # Strip markdown fences if the model wrapped them anyway
        if code.startswith("```"):
            lines = code.splitlines()
            code = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        # Safety: reject scripts that import forbidden modules
        if _poc_is_safe(code):
            return code

        self._log.warning(
            "temporal_poc_safety_rejected",
            vuln_id=report.id,
        )
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

_SMART_CONTRACT_AND_FINANCIAL = frozenset({
    AttackSurfaceType.SMART_CONTRACT_PUBLIC,
})

_FORBIDDEN_POC_IMPORTS = frozenset({
    "subprocess", "socket", "ctypes", "pickle", "shelve",
    "marshal", "shutil", "tempfile", "multiprocessing",
    "threading", "os.system", "os.popen",
})


def _severity_for_surface(surface: AttackSurface) -> VulnerabilitySeverity:
    """
    Race conditions on financial/smart-contract surfaces are CRITICAL;
    all others are HIGH (data integrity impact).
    """
    if surface.surface_type in _SMART_CONTRACT_AND_FINANCIAL:
        return VulnerabilitySeverity.CRITICAL
    return VulnerabilitySeverity.HIGH


def _build_encoding_prompt(surface: AttackSurface) -> str:
    lines = [
        "## Attack Surface",
        f"Entry point : {surface.entry_point}",
        f"Surface type: {surface.surface_type.value}",
        f"File        : {surface.file_path}",
    ]
    if surface.http_method:
        lines.append(f"HTTP method : {surface.http_method}")
    if surface.route_pattern:
        lines.append(f"Route       : {surface.route_pattern}")

    lines += [
        "",
        "## Source Code",
        "```",
        (surface.context_code or "(no context)")[:4000],
        "```",
        "",
        "Model TWO concurrent threads executing this function simultaneously.",
        "Encode the race condition as a Z3 expression and output the JSON.",
    ]
    return "\n".join(lines)


def _parse_encoding_response(
    raw: str,
    log: Any,
) -> tuple[str, dict[str, str], str, str] | None:
    """
    Extract z3_expression, variable_declarations, invariant_violated,
    and reasoning from the LLM JSON response.

    Tries direct parse then first-object extraction as fallback.
    """
    text = raw.strip()

    data: dict[str, Any] | None = None

    # Strategy 1: whole response is JSON
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            data = candidate
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first {...} block
    if data is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                candidate = json.loads(text[start: end + 1])
                if isinstance(candidate, dict):
                    data = candidate
            except json.JSONDecodeError:
                pass

    if data is None:
        log.warning("temporal_encoding_parse_failed", raw=raw[:300])
        return None

    z3_expr: str = data.get("z3_expression", "")
    var_decls: dict[str, str] = data.get("variable_declarations", {})
    invariant: str = data.get("invariant_violated", "invariant violated")
    reasoning: str = data.get("reasoning", "")

    if not z3_expr or not var_decls:
        log.warning(
            "temporal_encoding_incomplete",
            has_expr=bool(z3_expr),
            has_decls=bool(var_decls),
        )
        return None

    return z3_expr, var_decls, invariant, reasoning


def _poc_is_safe(code: str) -> bool:
    """Return True if the PoC script does not import forbidden modules."""
    return all(forbidden not in code for forbidden in _FORBIDDEN_POC_IMPORTS)


def _validate_z3_expression(
    z3_expr_code: str,
    variable_declarations: dict[str, str],
) -> str | None:
    """
    Attempt to evaluate the Z3 expression in a sandboxed namespace.

    Returns None if the expression is valid and produces a z3.BoolRef,
    or an error message string if it raises SyntaxError, NameError,
    a Z3Exception, or any other execution error.
    """
    try:
        import z3 as z3_lib
    except ImportError:
        return None  # Cannot validate without z3 installed; assume valid

    z3_vars: dict[str, Any] = {}
    for name, z3_type in variable_declarations.items():
        if z3_type == "Int":
            z3_vars[name] = z3_lib.Int(name)
        elif z3_type == "Real":
            z3_vars[name] = z3_lib.Real(name)
        elif z3_type == "Bool":
            z3_vars[name] = z3_lib.Bool(name)
        else:
            z3_vars[name] = z3_lib.Real(name)

    namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
    try:
        expr = eval(z3_expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    if not isinstance(expr, z3_lib.BoolRef):
        return f"expression produced {type(expr).__name__}, expected z3.BoolRef"

    return None
