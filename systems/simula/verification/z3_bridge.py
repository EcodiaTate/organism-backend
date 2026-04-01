"""
EcodiaOS -- Simula Z3 Invariant Discovery Bridge (Stage 2B)

Tight loop: reasoning model generates candidate invariants,
Z3 checks them, counterexamples are fed back, iterate.

Target domains:
  - Budget calculations (ADJUST_BUDGET proposals)
  - Risk scoring (ChangeSimulator risk synthesis)
  - Governance gating thresholds
  - Constitutional alignment bounds

Uses z3-solver Python bindings (z3.Solver, z3.Int, z3.Real, etc.)
rather than subprocess invocation for lower latency.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import Message
from systems.simula.verification.types import (
    DiscoveredInvariant,
    InvariantKind,
    InvariantVerificationResult,
    InvariantVerificationStatus,
    Z3RoundResult,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.verification.z3")


# ── Z3 Discovery System Prompt ───────────────────────────────────────────────

Z3_DISCOVERY_SYSTEM_PROMPT = """You are a formal invariant discovery assistant for EcodiaOS.
Your task: analyze Python code and propose invariants that can be verified by Z3.

## Output Format
Respond with a JSON array of invariant objects. Each object has:
- "kind": one of "precondition", "postcondition", "range_bound",
  "monotonicity", "relationship", "loop_invariant"
- "expression": human-readable invariant statement (e.g., "risk_score is always in [0.0, 1.0]")
- "z3_expression": Z3 Python expression using z3.And, z3.Or, z3.Not, z3.Implies, etc.
  Variables are pre-declared as z3.Int or z3.Real - just reference them by name.
  Example: "z3.And(risk_score >= 0, risk_score <= 1)"
- "variable_declarations": dict mapping variable names to z3 types ("Int", "Real", "Bool")
  Example: {"risk_score": "Real", "episode_count": "Int"}
- "target_function": fully qualified function name
- "confidence": float 0.0-1.0, your confidence this invariant holds

## EcodiaOS Domain Knowledge
- Risk scores: [0.0, 1.0]
- Budget values: non-negative reals
- Drive alignment: [-1.0, 1.0]
- Regression rates: [0.0, 1.0]
- Episode counts: non-negative integers
- Priority scores: non-negative reals
- Priority = evidence_strength * expected_impact / max(0.1, risk * cost)
- Regression threshold: unacceptable (0.10) > high (0.05) > moderate > low

## Rules
- Only propose invariants you are >60% confident about
- Prefer simple, verifiable invariants over complex ones
- Each invariant must be independently checkable by Z3
- Use only z3.And, z3.Or, z3.Not, z3.Implies, z3.If, comparison operators
- Do NOT use z3.ForAll or z3.Exists (keep it propositional)

Respond with ONLY the JSON array, no other text."""


Z3_REFINEMENT_TEMPLATE = """Some of your proposed invariants were refuted by Z3.

## Counterexamples (round {round_number}/{max_rounds})
{counterexamples}

## Previously Proposed Invariants
{previous_invariants}

Revise the invalid invariants based on the counterexamples.
Strengthen conditions, add bounds, or propose alternative invariants.
Respond with ONLY a JSON array of revised/new invariant objects."""


# ── Z3Bridge ─────────────────────────────────────────────────────────────────


class Z3Bridge:
    """
    Manages Z3 invariant checking and the LLM-driven discovery loop.

    Uses z3-solver Python bindings directly for low-latency checking.
    The LLM generates candidate invariants as Z3 Python expressions,
    which are evaluated in a sandboxed namespace.
    """

    def __init__(
        self,
        check_timeout_ms: int = 5000,
        max_rounds: int = 6,
    ) -> None:
        self._check_timeout_ms = check_timeout_ms
        self._max_rounds = max_rounds
        self._log = logger

    def check_invariant(
        self,
        z3_expr_code: str,
        variable_declarations: dict[str, str],
    ) -> tuple[InvariantVerificationStatus, str]:
        """
        Check a single invariant expression using Z3.

        The invariant is checked by asserting its negation:
        if NOT(invariant) is UNSAT, the invariant is universally valid.

        Args:
            z3_expr_code: Python code using z3 API that evaluates to a z3.BoolRef.
                Example: "z3.And(risk_score >= 0, risk_score <= 1)"
            variable_declarations: Mapping of variable names to Z3 types.
                Example: {"risk_score": "Real", "budget": "Real"}

        Returns:
            (status, counterexample_or_error).
        """
        try:
            import z3 as z3_lib
        except ImportError:
            return InvariantVerificationStatus.UNKNOWN, "z3-solver not installed"

        solver = z3_lib.Solver()
        solver.set("timeout", self._check_timeout_ms)

        # Create Z3 variables from declarations
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

        # Evaluate the Z3 expression in a sandboxed namespace
        namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
        try:
            expr = eval(z3_expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:
            return InvariantVerificationStatus.UNKNOWN, f"expression error: {exc}"

        # Check if the expression is a valid Z3 BoolRef
        if not isinstance(expr, z3_lib.BoolRef):
            return InvariantVerificationStatus.UNKNOWN, "expression did not produce a z3.BoolRef"

        # Check negation: if NOT(invariant) is UNSAT, invariant holds universally
        solver.add(z3_lib.Not(expr))
        result = solver.check()

        if result == z3_lib.unsat:
            return InvariantVerificationStatus.VALID, ""
        elif result == z3_lib.sat:
            model = solver.model()
            cex_parts = []
            for d in model.decls():
                cex_parts.append(f"{d.name()}={model[d]}")
            counterexample = ", ".join(cex_parts)
            return InvariantVerificationStatus.INVALID, counterexample
        else:
            return InvariantVerificationStatus.UNKNOWN, "solver timeout or unknown"

    async def run_discovery_loop(
        self,
        llm: LLMProvider,
        python_source: str,
        target_functions: list[str],
        domain_context: str = "",
    ) -> InvariantVerificationResult:
        """
        LLM generates candidate invariants, Z3 checks, counterexamples
        fed back. Iterates up to max_rounds.

        Args:
            llm: LLM provider for invariant generation.
            python_source: The Python source containing target functions.
            target_functions: Function names to discover invariants for.
            domain_context: Description of the domain (budget, risk, etc.).

        Returns:
            InvariantVerificationResult with all discovered invariants.
        """
        result = InvariantVerificationResult(rounds_max=self._max_rounds)
        start = time.monotonic()
        all_valid: list[DiscoveredInvariant] = []

        # Build initial prompt
        prompt = self._build_discovery_prompt(
            python_source, target_functions, domain_context,
        )
        messages: list[Message] = [Message(role="user", content=prompt)]

        for round_num in range(1, self._max_rounds + 1):
            self._log.info(
                "z3_discovery_round_start",
                round=round_num,
                max_rounds=self._max_rounds,
                targets=target_functions,
            )

            # LLM generates candidate invariants
            try:
                response = await llm.generate(
                    system_prompt=Z3_DISCOVERY_SYSTEM_PROMPT,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
            except Exception as exc:
                self._log.error("z3_llm_error", round=round_num, error=str(exc))
                result.error_summary = f"LLM call failed on round {round_num}: {exc}"
                break

            # Parse invariants from LLM response
            candidates = self._parse_invariants(response.text)
            if not candidates:
                self._log.warning("z3_no_candidates_parsed", round=round_num)
                # Append retry instruction to existing history; replacing loses code context
                messages.append(Message(role="assistant", content=response.text))
                messages.append(Message(role="user", content=(
                    "Your response was not valid JSON. Please respond with ONLY "
                    "a JSON array of invariant objects as specified."
                )))
                continue

            round_result = Z3RoundResult(
                round_number=round_num,
                llm_tokens_used=getattr(response, "total_tokens", 0),
            )
            counterexamples: list[str] = []

            # Check each invariant via Z3
            z3_start = time.monotonic()
            for inv in candidates:
                if not inv.z3_expression:
                    inv.status = InvariantVerificationStatus.UNKNOWN
                    round_result.unknown_count += 1
                    continue

                status, cex = self.check_invariant(
                    inv.z3_expression,
                    inv.variable_declarations,
                )
                inv.status = status
                inv.counterexample = cex

                if status == InvariantVerificationStatus.VALID:
                    round_result.valid_count += 1
                    all_valid.append(inv)
                elif status == InvariantVerificationStatus.INVALID:
                    round_result.invalid_count += 1
                    counterexamples.append(
                        f"  - '{inv.expression}': counterexample {{ {cex} }}"
                    )
                else:
                    round_result.unknown_count += 1

            round_result.z3_time_ms = int((time.monotonic() - z3_start) * 1000)
            round_result.candidate_invariants = candidates
            round_result.counterexamples_fed_back = counterexamples
            result.round_history.append(round_result)
            result.total_llm_tokens += round_result.llm_tokens_used
            result.total_z3_time_ms += round_result.z3_time_ms

            self._log.info(
                "z3_discovery_round_done",
                round=round_num,
                valid=round_result.valid_count,
                invalid=round_result.invalid_count,
                unknown=round_result.unknown_count,
            )

            if not counterexamples:
                # All invariants valid or unknown - stop iterating
                break

            # Feed counterexamples back to LLM
            prev_json = json.dumps(
                [{"expression": inv.expression, "z3_expression": inv.z3_expression}
                 for inv in candidates],
                indent=2,
            )
            feedback = Z3_REFINEMENT_TEMPLATE.format(
                round_number=round_num,
                max_rounds=self._max_rounds,
                counterexamples="\n".join(counterexamples),
                previous_invariants=prev_json,
            )
            messages = [Message(role="user", content=feedback)]

        # Build final result
        result.rounds_attempted = len(result.round_history)
        result.discovered_invariants = [
            inv
            for r in result.round_history
            for inv in r.candidate_invariants
        ]
        result.valid_invariants = all_valid
        result.status = (
            InvariantVerificationStatus.VALID if all_valid
            else InvariantVerificationStatus.UNKNOWN
        )
        result.verification_time_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "z3_discovery_complete",
            rounds=result.rounds_attempted,
            total_discovered=len(result.discovered_invariants),
            total_valid=len(all_valid),
        )
        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_discovery_prompt(
        self,
        python_source: str,
        target_functions: list[str],
        domain_context: str,
    ) -> str:
        """Build initial prompt asking LLM to propose invariants."""
        parts = [
            "Analyze the following Python code and propose formal invariants "
            "that can be verified by Z3.",
            "",
            f"## Target Functions: {', '.join(target_functions)}",
            "",
            "## Python Source",
            f"```python\n{python_source[:6000]}\n```",
        ]
        if domain_context:
            parts.extend(["", "## Domain Context", domain_context])

        parts.extend([
            "",
            "Propose invariants covering: range bounds, preconditions, "
            "postconditions, and relationships between variables.",
            "",
            "Respond with ONLY a JSON array of invariant objects.",
        ])
        return "\n".join(parts)

    def _parse_invariants(self, llm_text: str) -> list[DiscoveredInvariant]:
        """
        Parse LLM response to extract candidate invariants.
        Expects a JSON array of invariant objects.
        """
        # Try to find JSON array in the response
        text = llm_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        # Find the JSON array
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            self._log.warning("z3_parse_no_json_array", text=text[:200])
            return []

        json_str = text[start_idx:end_idx + 1]
        try:
            raw_list = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self._log.warning("z3_parse_json_error", error=str(exc))
            return []

        if not isinstance(raw_list, list):
            return []

        invariants: list[DiscoveredInvariant] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            try:
                kind_str = item.get("kind", "relationship")
                try:
                    kind = InvariantKind(kind_str)
                except ValueError:
                    kind = InvariantKind.RELATIONSHIP

                inv = DiscoveredInvariant(
                    kind=kind,
                    expression=item.get("expression", ""),
                    z3_expression=item.get("z3_expression", ""),
                    variable_declarations=item.get("variable_declarations", {}),
                    target_function=item.get("target_function", ""),
                    confidence=float(item.get("confidence", 0.5)),
                )
                if inv.expression and inv.z3_expression:
                    invariants.append(inv)
            except Exception as exc:
                self._log.debug("z3_parse_item_error", error=str(exc))
                continue

        return invariants
