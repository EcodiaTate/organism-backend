"""
EcodiaOS -- Simula Static Analysis Bridge (Stage 2C)

Subprocess runners for security and quality static analysis:
  - Bandit: Python security vulnerability scanner
  - Semgrep: Pattern-based static analysis (when available)

The bridge runs tools in parallel, parses JSON output, and
produces a unified StaticAnalysisResult. Findings are fed back
to the code agent as counterexamples for iterative repair.

Integration: post-generation gate in code_agent.py and
             verification phase in health.py.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import structlog

from systems.simula.verification.types import (
    StaticAnalysisFinding,
    StaticAnalysisResult,
    StaticAnalysisSeverity,
)

logger = structlog.get_logger().bind(system="simula.verification.static_analysis")


# ── Severity Mapping ─────────────────────────────────────────────────────────

_BANDIT_SEVERITY_MAP: dict[str, StaticAnalysisSeverity] = {
    "HIGH": StaticAnalysisSeverity.ERROR,
    "MEDIUM": StaticAnalysisSeverity.WARNING,
    "LOW": StaticAnalysisSeverity.INFO,
    "UNDEFINED": StaticAnalysisSeverity.INFO,
}

_SEMGREP_SEVERITY_MAP: dict[str, StaticAnalysisSeverity] = {
    "ERROR": StaticAnalysisSeverity.ERROR,
    "WARNING": StaticAnalysisSeverity.WARNING,
    "INFO": StaticAnalysisSeverity.INFO,
}


# ── StaticAnalysisBridge ─────────────────────────────────────────────────────


class StaticAnalysisBridge:
    """
    Runs Bandit and Semgrep static analysis tools on generated code.

    Both tools are invoked as subprocesses with JSON output parsing.
    The bridge runs them in parallel and merges results.
    """

    def __init__(
        self,
        codebase_root: Path,
        bandit_timeout_s: float = 30.0,
        semgrep_timeout_s: float = 60.0,
        # Inspector: allow overriding the workspace root for external target analysis
        workspace_root: Path | None = None,
    ) -> None:
        self._root = workspace_root or codebase_root
        self._bandit_timeout_s = bandit_timeout_s
        self._semgrep_timeout_s = semgrep_timeout_s
        self._log = logger

    async def run_all(self, files: list[str]) -> StaticAnalysisResult:
        """
        Run all available static analysis tools on the given files.
        Returns a unified StaticAnalysisResult.
        """
        if not files:
            return StaticAnalysisResult()

        start = time.monotonic()

        # Filter to Python files only
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            return StaticAnalysisResult()

        # Run tools in parallel
        bandit_task = asyncio.create_task(self._run_bandit(py_files))
        semgrep_task = asyncio.create_task(self._run_semgrep(py_files))

        bandit_findings = await bandit_task
        semgrep_findings = await semgrep_task

        # Merge findings
        all_findings = bandit_findings + semgrep_findings
        tools_run: list[str] = []
        if bandit_findings is not None:
            tools_run.append("bandit")
        if semgrep_findings is not None:
            tools_run.append("semgrep")

        error_count = sum(
            1 for f in all_findings if f.severity == StaticAnalysisSeverity.ERROR
        )
        warning_count = sum(
            1 for f in all_findings if f.severity == StaticAnalysisSeverity.WARNING
        )
        info_count = sum(
            1 for f in all_findings if f.severity == StaticAnalysisSeverity.INFO
        )
        fixable_count = sum(1 for f in all_findings if f.fixable)
        fix_rate = fixable_count / max(1, len(all_findings))

        result = StaticAnalysisResult(
            findings=all_findings,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            fixable_count=fixable_count,
            tools_run=tools_run,
            fix_rate=fix_rate,
            analysis_time_ms=int((time.monotonic() - start) * 1000),
        )

        self._log.info(
            "static_analysis_complete",
            files=len(py_files),
            findings=len(all_findings),
            errors=error_count,
            warnings=warning_count,
            tools=tools_run,
        )
        return result

    async def _run_bandit(self, files: list[str]) -> list[StaticAnalysisFinding]:
        """Run Bandit security scanner on files, return findings."""
        abs_files = [str(self._root / f) for f in files]

        try:
            proc = await asyncio.create_subprocess_exec(
                "bandit", "-f", "json", "-ll", *abs_files,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._bandit_timeout_s,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                self._log.warning("bandit_timeout")
                return []

            output = stdout.decode("utf-8", errors="replace")
            return self._parse_bandit_output(output, files)

        except FileNotFoundError:
            self._log.debug("bandit_not_installed")
            return []
        except Exception as exc:
            self._log.warning("bandit_error", error=str(exc))
            return []

    async def _run_semgrep(self, files: list[str]) -> list[StaticAnalysisFinding]:
        """Run Semgrep pattern analysis on files, return findings."""
        abs_files = [str(self._root / f) for f in files]

        try:
            proc = await asyncio.create_subprocess_exec(
                "semgrep", "--json", "--config", "auto", *abs_files,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._semgrep_timeout_s,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                self._log.warning("semgrep_timeout")
                return []

            output = stdout.decode("utf-8", errors="replace")
            return self._parse_semgrep_output(output, files)

        except FileNotFoundError:
            self._log.debug("semgrep_not_installed")
            return []
        except Exception as exc:
            self._log.warning("semgrep_error", error=str(exc))
            return []

    def _parse_bandit_output(
        self, output: str, original_files: list[str],
    ) -> list[StaticAnalysisFinding]:
        """Parse Bandit JSON output into findings."""
        if not output.strip():
            return []

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            self._log.warning("bandit_parse_error", output=output[:200])
            return []

        findings: list[StaticAnalysisFinding] = []
        for result in data.get("results", []):
            severity_str = result.get("issue_severity", "UNDEFINED")
            severity = _BANDIT_SEVERITY_MAP.get(severity_str, StaticAnalysisSeverity.INFO)

            # Relativize the file path
            file_path = result.get("filename", "")
            for orig in original_files:
                if file_path.endswith(orig) or orig in file_path:
                    file_path = orig
                    break

            finding = StaticAnalysisFinding(
                tool="bandit",
                rule_id=result.get("test_id", ""),
                severity=severity,
                file_path=file_path,
                line=result.get("line_number", 0),
                column=result.get("col_offset", 0),
                message=result.get("issue_text", ""),
                fixable=severity != StaticAnalysisSeverity.INFO,
                cwe=(
                    result.get("issue_cwe", {}).get("id", "")
                    if isinstance(result.get("issue_cwe"), dict)
                    else ""
                ),
            )
            findings.append(finding)

        return findings

    def _parse_semgrep_output(
        self, output: str, original_files: list[str],
    ) -> list[StaticAnalysisFinding]:
        """Parse Semgrep JSON output into findings."""
        if not output.strip():
            return []

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            self._log.warning("semgrep_parse_error", output=output[:200])
            return []

        findings: list[StaticAnalysisFinding] = []
        for result in data.get("results", []):
            severity_str = result.get("extra", {}).get("severity", "INFO")
            severity = _SEMGREP_SEVERITY_MAP.get(
                severity_str.upper(), StaticAnalysisSeverity.INFO,
            )

            file_path = result.get("path", "")
            for orig in original_files:
                if file_path.endswith(orig) or orig in file_path:
                    file_path = orig
                    break

            finding = StaticAnalysisFinding(
                tool="semgrep",
                rule_id=result.get("check_id", ""),
                severity=severity,
                file_path=file_path,
                line=result.get("start", {}).get("line", 0),
                column=result.get("start", {}).get("col", 0),
                message=result.get("extra", {}).get("message", ""),
                fixable=result.get("extra", {}).get("is_fixable", False),
            )
            findings.append(finding)

        return findings

    @staticmethod
    def format_findings_for_feedback(result: StaticAnalysisResult) -> str:
        """
        Format static analysis findings as text for code agent feedback.

        Used in the post-generation gate: findings become tool results
        that the code agent uses to fix issues.
        """
        if not result.findings:
            return "No static analysis findings."

        lines = [
            f"Static analysis found {len(result.findings)} issue(s) "
            f"({result.error_count} errors, {result.warning_count} warnings):",
            "",
        ]
        for f in result.findings:
            severity_marker = {
                StaticAnalysisSeverity.ERROR: "ERROR",
                StaticAnalysisSeverity.WARNING: "WARN ",
                StaticAnalysisSeverity.INFO: "INFO ",
            }.get(f.severity, "     ")
            lines.append(
                f"  [{severity_marker}] {f.file_path}:{f.line} "
                f"({f.tool}/{f.rule_id}) {f.message}"
            )

        if result.error_count > 0:
            lines.extend([
                "",
                "BLOCKING: ERROR-severity findings must be fixed before the "
                "proposal can proceed. Focus on security vulnerabilities first.",
            ])

        return "\n".join(lines)
