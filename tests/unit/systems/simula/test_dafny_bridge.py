"""
Unit tests for Simula DafnyBridge (Stage 2A).

Tests Dafny subprocess invocation, Clover loop iteration,
template selection, and output parsing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from systems.simula.verification.dafny_bridge import DafnyBridge
from systems.simula.verification.templates import TEMPLATES, get_template
from systems.simula.verification.types import (
    DafnyVerificationStatus,
)

# ── DafnyBridge Tests ─────────────────────────────────────────────────────────


class TestDafnyBridgeAvailability:
    @pytest.mark.asyncio
    async def test_check_available_returns_true_when_binary_exists(self, tmp_path: Path):
        bridge = DafnyBridge(dafny_path="echo", verify_timeout_s=5.0)
        # 'echo' is always available
        result = await bridge.check_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_available_returns_false_for_missing_binary(self):
        bridge = DafnyBridge(dafny_path="/nonexistent/dafny_xyz_fake", verify_timeout_s=5.0)
        result = await bridge.check_available()
        assert result is False


class TestDafnyVerification:
    @pytest.mark.asyncio
    async def test_verify_dafny_source_success(self, tmp_path: Path):
        """Mock a successful dafny verify subprocess."""
        bridge = DafnyBridge(dafny_path="dafny", verify_timeout_s=5.0)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"Dafny program verified\n", b"")
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            verified, stdout, stderr, exit_code = await bridge.verify_dafny_source(
                "method Foo() returns (x: int) ensures x >= 0 { x := 1; }"
            )
        assert verified is True
        assert exit_code == 0
        assert "verified" in stdout.lower()

    @pytest.mark.asyncio
    async def test_verify_dafny_source_failure(self, tmp_path: Path):
        """Mock a failed dafny verify."""
        bridge = DafnyBridge(dafny_path="dafny", verify_timeout_s=5.0)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"Error: postcondition might not hold\n", b"")
        )
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            verified, stdout, stderr, exit_code = await bridge.verify_dafny_source(
                "method Foo() returns (x: int) ensures x >= 0 { x := -1; }"
            )
        assert verified is False
        assert exit_code != 0

    @pytest.mark.asyncio
    async def test_verify_dafny_source_timeout(self, tmp_path: Path):
        """Timeout should kill process and return False."""
        bridge = DafnyBridge(dafny_path="dafny", verify_timeout_s=0.01)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            side_effect=TimeoutError()
        )
        mock_proc.kill = MagicMock()

        # After kill, communicate should return empty
        async def post_kill_communicate():
            return b"", b""
        mock_proc.communicate.side_effect = [TimeoutError()]

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # The bridge should handle timeout gracefully
            verified, stdout, stderr, exit_code = await bridge.verify_dafny_source(
                "method Hang() { while true {} }"
            )
        assert verified is False


class TestCloverLoop:
    @pytest.mark.asyncio
    async def test_clover_loop_verifies_on_first_round(self):
        """LLM generates valid Dafny on first try."""
        bridge = DafnyBridge(dafny_path="dafny", verify_timeout_s=5.0, max_rounds=3)

        # Mock LLM to return valid Dafny
        mock_llm = AsyncMock()
        mock_llm.complete_with_tools = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = (
            "Here is the Dafny specification:\n\n"
            "```dafny\n"
            "method Verify(x: int) returns (y: int)\n"
            "  ensures y >= 0\n"
            "{ y := if x >= 0 then x else -x; }\n"
            "```"
        )
        mock_response.tool_calls = []
        mock_llm.complete_with_tools.return_value = mock_response

        # Mock successful verification
        with patch.object(bridge, "verify_dafny_source", return_value=(True, "verified", "", 0)):
            result = await bridge.run_clover_loop(
                llm=mock_llm,
                python_source="def verify(x): return abs(x)",
                function_name="verify",
                context="Return the absolute value",
            )

        assert result.status == DafnyVerificationStatus.VERIFIED
        assert result.rounds_attempted == 1
        assert len(result.round_history) == 1
        assert result.round_history[0].verified is True

    @pytest.mark.asyncio
    async def test_clover_loop_iterates_on_failure(self):
        """LLM generates invalid Dafny, gets feedback, then succeeds."""
        bridge = DafnyBridge(dafny_path="dafny", verify_timeout_s=5.0, max_rounds=3)

        mock_llm = AsyncMock()
        dafny_code = (
            "```dafny\n"
            "method M() returns (x: int) ensures x >= 0 { x := 1; }\n"
            "```"
        )
        mock_response = MagicMock()
        mock_response.text = dafny_code
        mock_response.tool_calls = []
        mock_llm.complete_with_tools.return_value = mock_response

        # First call fails, second succeeds
        call_count = 0

        async def mock_verify(source):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (False, "Error: postcondition might not hold", "", 1)
            return (True, "verified", "", 0)

        with patch.object(bridge, "verify_dafny_source", side_effect=mock_verify):
            result = await bridge.run_clover_loop(
                llm=mock_llm,
                python_source="def m(): return 1",
                function_name="m",
                context="Returns 1",
            )

        assert result.status == DafnyVerificationStatus.VERIFIED
        assert result.rounds_attempted == 2

    @pytest.mark.asyncio
    async def test_clover_loop_fails_after_max_rounds(self):
        """LLM never generates valid Dafny - fails after max rounds."""
        bridge = DafnyBridge(dafny_path="dafny", verify_timeout_s=5.0, max_rounds=2)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "```dafny\nmethod M() { }\n```"
        mock_response.tool_calls = []
        mock_llm.complete_with_tools.return_value = mock_response

        with patch.object(
            bridge, "verify_dafny_source",
            return_value=(False, "Error", "", 1),
        ):
            result = await bridge.run_clover_loop(
                llm=mock_llm,
                python_source="def m(): pass",
                function_name="m",
                context="Does nothing",
            )

        assert result.status == DafnyVerificationStatus.FAILED
        assert result.rounds_attempted == 2


class TestDafnyOutputParsing:
    def test_parse_dafny_fenced_block(self):
        bridge = DafnyBridge(dafny_path="dafny")
        text = (
            "Here is the spec:\n"
            "```dafny\n"
            "method M() returns (x: int)\n"
            "  ensures x >= 0\n"
            "{ x := 1; }\n"
            "```\n"
            "Done."
        )
        parsed = bridge._parse_dafny_output(text)
        assert "method M()" in parsed
        assert "ensures x >= 0" in parsed

    def test_parse_dafny_no_block(self):
        bridge = DafnyBridge(dafny_path="dafny")
        text = "No code here, just text."
        parsed = bridge._parse_dafny_output(text)
        assert parsed == ""


# ── Template Tests ────────────────────────────────────────────────────────────


class TestTemplates:
    def test_all_templates_are_non_empty(self):
        for key, template in TEMPLATES.items():
            assert template.strip(), f"Template '{key}' is empty"

    def test_get_template_by_spec_type(self):
        result = get_template("irrelevant", spec_type="budget_adjustment")
        assert result is not None
        assert "VerifyBudgetAdjustment" in result

    def test_get_template_by_category(self):
        result = get_template("adjust_budget")
        assert result is not None
        assert "Budget" in result

    def test_get_template_governance_category(self):
        result = get_template("modify_contract")
        assert result is not None
        assert "Governance" in result or "governance" in result

    def test_get_template_returns_none_for_unknown(self):
        result = get_template("unknown_category_xyz")
        assert result is None

    def test_template_count(self):
        assert len(TEMPLATES) >= 6

    def test_budget_template_has_postcondition(self):
        template = TEMPLATES["budget_adjustment"]
        assert "ensures" in template

    def test_risk_scoring_template_has_datatype(self):
        template = TEMPLATES["risk_scoring"]
        assert "datatype RiskLevel" in template

    def test_governance_gate_partition_lemma(self):
        template = TEMPLATES["governance_gate"]
        assert "lemma CategoryPartition" in template
