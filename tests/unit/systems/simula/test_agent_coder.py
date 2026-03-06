"""
Unit tests for Simula AgentCoder Pipeline (Stage 2D).

Tests TestDesigner, TestExecutor, and the 3-agent pipeline.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from systems.simula.agents.test_designer import TestDesignerAgent
from systems.simula.agents.test_executor import TestExecutorAgent
from systems.simula.verification.types import (
    TestDesignResult,
    TestExecutionResult,
)

# ── TestDesigner ──────────────────────────────────────────────────────────────


class TestTestDesigner:
    def _make_proposal(self) -> MagicMock:
        """Create a mock EvolutionProposal."""
        proposal = MagicMock()
        proposal.category.value = "add_executor"
        proposal.description = "Add a new HTTP executor"
        proposal.expected_benefit = "Enables HTTP-based actions"
        proposal.change_spec.target_system = "axon"
        proposal.change_spec.files_to_modify = ["src/systems/axon/executors/http.py"]
        proposal.change_spec.description = "New HTTP executor for external API calls"
        return proposal

    @pytest.mark.asyncio
    async def test_design_tests_returns_result(self, tmp_path: Path):
        """Designer should return a TestDesignResult with test files."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = (
            "Here are the tests:\n\n"
            "```tests/unit/systems/axon/test_http_executor.py\n"
            "import pytest\n\n"
            "class TestHttpExecutor:\n"
            "    @pytest.mark.asyncio\n"
            "    async def test_execute_success(self):\n"
            "        assert True\n\n"
            "    @pytest.mark.asyncio\n"
            "    async def test_execute_timeout(self):\n"
            "        assert True\n"
            "```\n\n"
            "Coverage targets:\n"
            "- HttpExecutor.execute\n"
            "- HttpExecutor.__init__\n"
        )
        mock_response.tool_calls = []
        mock_llm.complete_with_tools.return_value = mock_response

        designer = TestDesignerAgent(llm=mock_llm, codebase_root=tmp_path)
        proposal = self._make_proposal()
        result = await designer.design_tests(proposal)

        assert isinstance(result, TestDesignResult)
        assert len(result.test_files) == 1
        assert "tests/unit/systems/axon/test_http_executor.py" in result.test_files
        assert result.test_count == 2
        assert len(result.coverage_targets) >= 1

    @pytest.mark.asyncio
    async def test_design_tests_with_tools(self, tmp_path: Path):
        """Designer uses tools to explore codebase before generating tests."""
        # Create some files for the tools to find
        src_dir = tmp_path / "src" / "ecodiaos"
        src_dir.mkdir(parents=True)
        (src_dir / "example.py").write_text("def hello(): pass\n", encoding="utf-8")

        mock_llm = AsyncMock()
        # First call: uses tools
        tool_response = MagicMock()
        tool_response.text = "Let me explore the codebase first."
        tool_call = MagicMock()
        tool_call.id = "tc_1"
        tool_call.name = "list_directory"
        tool_call.input = {"path": "src/ecodiaos"}
        tool_response.tool_calls = [tool_call]

        # Second call: generates tests (no more tool calls)
        final_response = MagicMock()
        final_response.text = (
            "```tests/unit/test_example.py\n"
            "def test_hello():\n"
            "    assert True\n"
            "```"
        )
        final_response.tool_calls = []

        mock_llm.complete_with_tools.side_effect = [tool_response, final_response]

        designer = TestDesignerAgent(llm=mock_llm, codebase_root=tmp_path)
        proposal = self._make_proposal()
        result = await designer.design_tests(proposal)

        assert len(result.test_files) == 1
        assert result.test_count == 1


class TestTestDesignerParsing:
    def test_parse_test_files(self):
        text = (
            "```tests/unit/test_a.py\n"
            "def test_one(): pass\n"
            "```\n\n"
            "```tests/unit/test_b.py\n"
            "def test_two(): pass\n"
            "def test_three(): pass\n"
            "```"
        )
        files = TestDesignerAgent._parse_test_files(text)
        assert len(files) == 2
        assert "tests/unit/test_a.py" in files
        assert "tests/unit/test_b.py" in files

    def test_parse_test_files_no_blocks(self):
        files = TestDesignerAgent._parse_test_files("No code blocks here.")
        assert len(files) == 0

    def test_count_tests(self):
        test_files = {
            "test_a.py": "def test_one(): pass\ndef test_two(): pass\n",
            "test_b.py": "async def test_three(): pass\n",
        }
        count = TestDesignerAgent._count_tests(test_files)
        assert count == 3

    def test_extract_coverage_targets(self):
        text = (
            "Coverage targets:\n"
            "- HttpExecutor.execute\n"
            "- HttpExecutor.__init__\n"
            "- validate_url\n"
        )
        targets = TestDesignerAgent._extract_coverage_targets(text)
        assert len(targets) == 3
        assert "HttpExecutor.execute" in targets


# ── TestExecutor ──────────────────────────────────────────────────────────────


class TestTestExecutor:
    @pytest.mark.asyncio
    async def test_execute_empty_tests(self, tmp_path: Path):
        executor = TestExecutorAgent(codebase_root=tmp_path)
        result = await executor.execute_tests({})
        assert isinstance(result, TestExecutionResult)
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_write_test_files(self, tmp_path: Path):
        executor = TestExecutorAgent(codebase_root=tmp_path)
        test_files = {
            "tests/test_example.py": "def test_one():\n    assert True\n",
        }
        paths = executor._write_test_files(test_files)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].read_text(encoding="utf-8") == "def test_one():\n    assert True\n"

    @pytest.mark.asyncio
    async def test_cleanup_test_files(self, tmp_path: Path):
        executor = TestExecutorAgent(codebase_root=tmp_path)
        test_files = {
            "tests/test_cleanup.py": "def test_x(): pass\n",
        }
        executor._write_test_files(test_files)
        assert (tmp_path / "tests" / "test_cleanup.py").exists()

        executor._cleanup_test_files()
        assert not (tmp_path / "tests" / "test_cleanup.py").exists()

    def test_parse_json_report(self, tmp_path: Path):
        executor = TestExecutorAgent(codebase_root=tmp_path)
        report_path = tmp_path / ".pytest_report.json"
        report_data = {
            "summary": {
                "passed": 5,
                "failed": 2,
                "error": 1,
                "total": 8,
            },
            "tests": [
                {
                    "nodeid": "tests/test_a.py::test_pass",
                    "outcome": "passed",
                },
                {
                    "nodeid": "tests/test_a.py::test_fail",
                    "outcome": "failed",
                    "call": {
                        "longrepr": "AssertionError: assert 1 == 2",
                    },
                },
            ],
        }
        report_path.write_text(json.dumps(report_data), encoding="utf-8")

        result = executor._parse_json_report(report_path, "raw output here")
        assert result.passed == 5
        assert result.failed == 2
        assert result.errors == 1
        assert result.total == 8
        assert len(result.failure_details) == 1
        assert "test_fail" in result.failure_details[0]

    def test_parse_json_report_missing_file(self, tmp_path: Path):
        executor = TestExecutorAgent(codebase_root=tmp_path)
        result = executor._parse_json_report(
            tmp_path / "nonexistent.json", "raw",
        )
        assert result.total == 0

    def test_parse_stdout_output(self):
        stdout = (
            "tests/test_a.py::test_one PASSED\n"
            "tests/test_a.py::test_two FAILED\n"
            "\n"
            "=== 4 passed, 1 failed in 0.5s ===\n"
        )
        result = TestExecutorAgent._parse_stdout_output(stdout, stdout)
        assert result.passed == 4
        assert result.failed == 1

    def test_parse_stdout_all_passed(self):
        stdout = "=== 10 passed in 1.2s ===\n"
        result = TestExecutorAgent._parse_stdout_output(stdout, stdout)
        assert result.passed == 10
        assert result.failed == 0

    def test_format_failures_for_feedback_no_failures(self):
        result = TestExecutionResult(passed=5, total=5)
        text = TestExecutorAgent.format_failures_for_feedback(result)
        assert "All 5 tests passed" in text

    def test_format_failures_for_feedback_with_failures(self):
        result = TestExecutionResult(
            passed=3, failed=2, errors=0, total=5,
            failure_details=[
                "test_a::test_one: AssertionError",
                "test_a::test_two: TypeError",
            ],
        )
        text = TestExecutorAgent.format_failures_for_feedback(result)
        assert "3 passed, 2 failed" in text
        assert "Do NOT modify the test files" in text
        assert "AssertionError" in text

    def test_format_failures_for_no_tests(self):
        result = TestExecutionResult()
        text = TestExecutorAgent.format_failures_for_feedback(result)
        assert "No tests were executed" in text
