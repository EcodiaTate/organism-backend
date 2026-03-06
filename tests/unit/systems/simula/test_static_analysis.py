"""
Unit tests for Simula StaticAnalysisBridge (Stage 2C).

Tests Bandit/Semgrep subprocess invocation, JSON output parsing,
severity mapping, and feedback formatting.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from systems.simula.verification.static_analysis import StaticAnalysisBridge
from systems.simula.verification.types import (
    StaticAnalysisFinding,
    StaticAnalysisResult,
    StaticAnalysisSeverity,
)

# ── Bandit Output Parsing ────────────────────────────────────────────────────


class TestBanditParsing:
    def test_parse_bandit_with_findings(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        bandit_output = json.dumps({
            "results": [
                {
                    "test_id": "B301",
                    "issue_severity": "HIGH",
                    "issue_text": "Use of pickle detected",
                    "filename": str(tmp_path / "src/module.py"),
                    "line_number": 42,
                    "col_offset": 0,
                    "issue_cwe": {"id": "CWE-502"},
                },
                {
                    "test_id": "B101",
                    "issue_severity": "LOW",
                    "issue_text": "Use of assert detected",
                    "filename": str(tmp_path / "src/utils.py"),
                    "line_number": 10,
                    "col_offset": 4,
                },
            ],
        })
        findings = bridge._parse_bandit_output(
            bandit_output,
            original_files=["src/module.py", "src/utils.py"],
        )
        assert len(findings) == 2
        assert findings[0].tool == "bandit"
        assert findings[0].rule_id == "B301"
        assert findings[0].severity == StaticAnalysisSeverity.ERROR
        assert findings[0].message == "Use of pickle detected"
        assert findings[0].cwe == "CWE-502"

        assert findings[1].severity == StaticAnalysisSeverity.INFO
        assert findings[1].fixable is False  # INFO is not fixable

    def test_parse_bandit_empty_output(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        findings = bridge._parse_bandit_output("", original_files=[])
        assert findings == []

    def test_parse_bandit_invalid_json(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        findings = bridge._parse_bandit_output(
            "not valid json", original_files=[],
        )
        assert findings == []

    def test_parse_bandit_no_results_key(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        findings = bridge._parse_bandit_output(
            json.dumps({"errors": []}), original_files=[],
        )
        assert findings == []

    def test_parse_bandit_severity_mapping(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        for bandit_sev, expected_sev in [
            ("HIGH", StaticAnalysisSeverity.ERROR),
            ("MEDIUM", StaticAnalysisSeverity.WARNING),
            ("LOW", StaticAnalysisSeverity.INFO),
            ("UNDEFINED", StaticAnalysisSeverity.INFO),
        ]:
            output = json.dumps({
                "results": [{
                    "test_id": "B001",
                    "issue_severity": bandit_sev,
                    "issue_text": "Test",
                    "filename": "test.py",
                    "line_number": 1,
                }],
            })
            findings = bridge._parse_bandit_output(output, original_files=["test.py"])
            assert findings[0].severity == expected_sev, f"Failed for {bandit_sev}"


# ── Semgrep Output Parsing ───────────────────────────────────────────────────


class TestSemgrepParsing:
    def test_parse_semgrep_with_findings(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        semgrep_output = json.dumps({
            "results": [
                {
                    "check_id": "python.security.sql-injection",
                    "path": str(tmp_path / "src/db.py"),
                    "start": {"line": 25, "col": 8},
                    "end": {"line": 25, "col": 40},
                    "extra": {
                        "severity": "ERROR",
                        "message": "Possible SQL injection",
                        "is_fixable": True,
                    },
                },
            ],
        })
        findings = bridge._parse_semgrep_output(
            semgrep_output,
            original_files=["src/db.py"],
        )
        assert len(findings) == 1
        assert findings[0].tool == "semgrep"
        assert findings[0].rule_id == "python.security.sql-injection"
        assert findings[0].severity == StaticAnalysisSeverity.ERROR
        assert findings[0].fixable is True
        assert findings[0].line == 25

    def test_parse_semgrep_empty_output(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        findings = bridge._parse_semgrep_output("", original_files=[])
        assert findings == []

    def test_parse_semgrep_severity_mapping(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        for sg_sev, expected in [
            ("ERROR", StaticAnalysisSeverity.ERROR),
            ("WARNING", StaticAnalysisSeverity.WARNING),
            ("INFO", StaticAnalysisSeverity.INFO),
        ]:
            output = json.dumps({
                "results": [{
                    "check_id": "test",
                    "path": "f.py",
                    "start": {"line": 1, "col": 1},
                    "extra": {"severity": sg_sev, "message": "test"},
                }],
            })
            findings = bridge._parse_semgrep_output(output, original_files=["f.py"])
            assert findings[0].severity == expected


# ── Run All (Integration) ────────────────────────────────────────────────────


class TestRunAll:
    @pytest.mark.asyncio
    async def test_run_all_empty_files(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        result = await bridge.run_all([])
        assert isinstance(result, StaticAnalysisResult)
        assert len(result.findings) == 0

    @pytest.mark.asyncio
    async def test_run_all_no_python_files(self, tmp_path: Path):
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        result = await bridge.run_all(["data.json", "readme.md"])
        assert isinstance(result, StaticAnalysisResult)
        assert len(result.findings) == 0

    @pytest.mark.asyncio
    async def test_run_all_tools_not_installed(self, tmp_path: Path):
        """When bandit/semgrep are not installed, should return empty results."""
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)
        # Both tools will raise FileNotFoundError → empty findings
        result = await bridge.run_all(["test.py"])
        assert isinstance(result, StaticAnalysisResult)

    @pytest.mark.asyncio
    async def test_run_all_merges_findings(self, tmp_path: Path):
        """Mock both tools returning findings, verify merge."""
        bridge = StaticAnalysisBridge(codebase_root=tmp_path)

        bandit_findings = [
            StaticAnalysisFinding(
                tool="bandit", rule_id="B001",
                severity=StaticAnalysisSeverity.ERROR,
                message="issue 1", file_path="test.py", line=1,
                fixable=True,
            ),
        ]
        semgrep_findings = [
            StaticAnalysisFinding(
                tool="semgrep", rule_id="S001",
                severity=StaticAnalysisSeverity.WARNING,
                message="issue 2", file_path="test.py", line=5,
            ),
        ]

        with (
            patch.object(bridge, "_run_bandit", return_value=bandit_findings),
            patch.object(bridge, "_run_semgrep", return_value=semgrep_findings),
        ):
            result = await bridge.run_all(["test.py"])

        assert result.error_count == 1
        assert result.warning_count == 1
        assert result.fixable_count == 1
        assert len(result.findings) == 2
        assert "bandit" in result.tools_run
        assert "semgrep" in result.tools_run


# ── Feedback Formatting ──────────────────────────────────────────────────────


class TestFeedbackFormatting:
    def test_format_no_findings(self):
        result = StaticAnalysisResult()
        text = StaticAnalysisBridge.format_findings_for_feedback(result)
        assert "No static analysis findings" in text

    def test_format_with_findings(self):
        result = StaticAnalysisResult(
            findings=[
                StaticAnalysisFinding(
                    tool="bandit", rule_id="B301",
                    severity=StaticAnalysisSeverity.ERROR,
                    message="Use of pickle",
                    file_path="src/module.py", line=42,
                ),
                StaticAnalysisFinding(
                    tool="semgrep", rule_id="S001",
                    severity=StaticAnalysisSeverity.WARNING,
                    message="Debug print",
                    file_path="src/utils.py", line=10,
                ),
            ],
            error_count=1,
            warning_count=1,
        )
        text = StaticAnalysisBridge.format_findings_for_feedback(result)
        assert "2 issue(s)" in text
        assert "1 errors" in text
        assert "ERROR" in text
        assert "WARN" in text
        assert "BLOCKING" in text
        assert "src/module.py:42" in text

    def test_format_without_errors_no_blocking(self):
        result = StaticAnalysisResult(
            findings=[
                StaticAnalysisFinding(
                    tool="bandit", rule_id="B101",
                    severity=StaticAnalysisSeverity.WARNING,
                    message="Assert used",
                    file_path="test.py", line=1,
                ),
            ],
            error_count=0,
            warning_count=1,
        )
        text = StaticAnalysisBridge.format_findings_for_feedback(result)
        assert "BLOCKING" not in text
