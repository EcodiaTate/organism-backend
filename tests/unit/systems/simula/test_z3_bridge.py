"""
Unit tests for Simula Z3Bridge (Stage 2B).

Tests Z3 invariant checking, sandboxed evaluation, discovery loop,
and invariant parsing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from systems.simula.verification.types import (
    InvariantKind,
    InvariantVerificationResult,
    InvariantVerificationStatus,
)
from systems.simula.verification.z3_bridge import Z3Bridge

# ── Z3 Invariant Checking ────────────────────────────────────────────────────


class TestCheckInvariant:
    """Test check_invariant with real z3-solver."""

    def test_valid_invariant_always_positive(self):
        """x * x >= 0 is always true for reals → VALID."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, counterexample = bridge.check_invariant(
            z3_expr_code="x * x >= 0",
            variable_declarations={"x": "Real"},
        )
        assert status == InvariantVerificationStatus.VALID
        assert counterexample == ""

    def test_valid_invariant_bounded_sum(self):
        """If 0 <= x <= 1 and 0 <= y <= 1, then x + y <= 2 → VALID."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, counterexample = bridge.check_invariant(
            z3_expr_code="z3.Implies(z3.And(x >= 0, x <= 1, y >= 0, y <= 1), x + y <= 2)",
            variable_declarations={"x": "Real", "y": "Real"},
        )
        assert status == InvariantVerificationStatus.VALID

    def test_invalid_invariant_with_counterexample(self):
        """x + 1 > 0 is NOT always true for all integers → INVALID with counterexample."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, counterexample = bridge.check_invariant(
            z3_expr_code="x + 1 > 0",
            variable_declarations={"x": "Int"},
        )
        assert status == InvariantVerificationStatus.INVALID
        assert counterexample != ""

    def test_simple_true_invariant_int(self):
        """x == x is always true → VALID."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, _ = bridge.check_invariant(
            z3_expr_code="x == x",
            variable_declarations={"x": "Int"},
        )
        assert status == InvariantVerificationStatus.VALID

    def test_simple_false_invariant(self):
        """x > x is never true → INVALID."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, _ = bridge.check_invariant(
            z3_expr_code="x > x",
            variable_declarations={"x": "Int"},
        )
        assert status == InvariantVerificationStatus.INVALID

    def test_bool_variable_invariant(self):
        """z3.Or(b, z3.Not(b)) is a tautology → VALID."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, _ = bridge.check_invariant(
            z3_expr_code="z3.Or(b, z3.Not(b))",
            variable_declarations={"b": "Bool"},
        )
        assert status == InvariantVerificationStatus.VALID

    def test_empty_expression_returns_unknown(self):
        """Empty expression should not crash, returns UNKNOWN."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, _ = bridge.check_invariant(
            z3_expr_code="",
            variable_declarations={"x": "Int"},
        )
        assert status == InvariantVerificationStatus.UNKNOWN


class TestSandboxSafety:
    """Ensure Z3 eval sandbox blocks dangerous code."""

    def test_sandbox_blocks_import(self):
        """Attempt to import os should be blocked."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, _ = bridge.check_invariant(
            z3_expr_code="__import__('os').system('echo hacked')",
            variable_declarations={"x": "Int"},
        )
        # Should return UNKNOWN due to eval failure, not execute the code
        assert status == InvariantVerificationStatus.UNKNOWN

    def test_sandbox_blocks_builtins(self):
        """Builtins like open() should be blocked."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, _ = bridge.check_invariant(
            z3_expr_code="open('/etc/passwd')",
            variable_declarations={"x": "Int"},
        )
        assert status == InvariantVerificationStatus.UNKNOWN

    def test_unknown_variable_type(self):
        """Unknown variable type should fail gracefully."""
        bridge = Z3Bridge(check_timeout_ms=5000)
        status, _ = bridge.check_invariant(
            z3_expr_code="x > 0",
            variable_declarations={"x": "ComplexNumber"},
        )
        assert status == InvariantVerificationStatus.UNKNOWN


# ── Invariant Parsing ─────────────────────────────────────────────────────────


class TestInvariantParsing:
    def test_parse_valid_json_invariants(self):
        bridge = Z3Bridge()
        text = """
Here are the invariants:
```json
[
    {
        "kind": "precondition",
        "expression": "budget >= 0",
        "z3_expression": "budget >= 0",
        "variable_declarations": {"budget": "Real"},
        "target_function": "adjust_budget"
    }
]
```
"""
        invariants = bridge._parse_invariants(text)
        assert len(invariants) == 1
        assert invariants[0].kind == InvariantKind.PRECONDITION
        assert invariants[0].expression == "budget >= 0"
        assert invariants[0].target_function == "adjust_budget"

    def test_parse_multiple_invariants(self):
        bridge = Z3Bridge()
        text = """
```json
[
    {
        "kind": "precondition",
        "expression": "x >= 0",
        "z3_expression": "x >= 0",
        "variable_declarations": {"x": "Int"},
        "target_function": "f"
    },
    {
        "kind": "postcondition",
        "expression": "result > x",
        "z3_expression": "result > x",
        "variable_declarations": {"x": "Int", "result": "Int"},
        "target_function": "f"
    }
]
```
"""
        invariants = bridge._parse_invariants(text)
        assert len(invariants) == 2
        assert invariants[0].kind == InvariantKind.PRECONDITION
        assert invariants[1].kind == InvariantKind.POSTCONDITION

    def test_parse_no_json_block(self):
        bridge = Z3Bridge()
        text = "No JSON here at all."
        invariants = bridge._parse_invariants(text)
        assert len(invariants) == 0

    def test_parse_invalid_json(self):
        bridge = Z3Bridge()
        text = "```json\n{invalid json\n```"
        invariants = bridge._parse_invariants(text)
        assert len(invariants) == 0


# ── Discovery Loop ────────────────────────────────────────────────────────────


class TestDiscoveryLoop:
    @pytest.mark.asyncio
    async def test_discovery_loop_finds_valid_invariants(self):
        """LLM suggests invariants, Z3 validates them."""
        bridge = Z3Bridge(check_timeout_ms=5000, max_rounds=2)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = """
Here are the invariants:
```json
[
    {
        "kind": "range_bound",
        "expression": "x * x >= 0",
        "z3_expression": "x * x >= 0",
        "variable_declarations": {"x": "Real"},
        "target_function": "compute"
    }
]
```
"""
        mock_response.tool_calls = []
        mock_llm.complete_with_tools.return_value = mock_response

        result = await bridge.run_discovery_loop(
            llm=mock_llm,
            python_source="def compute(x): return x * x",
            target_functions=["compute"],
            domain_context="Returns the square of x",
        )

        assert isinstance(result, InvariantVerificationResult)
        assert result.rounds_attempted >= 1
        assert len(result.discovered_invariants) >= 1
        # x*x >= 0 should be valid
        valid = [
            i for i in result.discovered_invariants
            if i.status == InvariantVerificationStatus.VALID
        ]
        assert len(valid) >= 1

    @pytest.mark.asyncio
    async def test_discovery_loop_handles_llm_error(self):
        """LLM raises exception - loop should handle gracefully."""
        bridge = Z3Bridge(check_timeout_ms=5000, max_rounds=1)

        mock_llm = AsyncMock()
        mock_llm.complete_with_tools.side_effect = RuntimeError("LLM unavailable")

        result = await bridge.run_discovery_loop(
            llm=mock_llm,
            python_source="def f(): pass",
            target_functions=["f"],
            domain_context="Test",
        )

        assert isinstance(result, InvariantVerificationResult)
        assert result.rounds_attempted == 0 or len(result.discovered_invariants) == 0
