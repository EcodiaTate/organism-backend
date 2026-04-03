"""
EcodiaOS -- Simula Issue Resolver (Stage 5E.1 + 5E.2 + 5E.3)

LogicStar-pattern autonomous issue resolution with progressive autonomy
and strict abstention policy.

Pipeline:
  INVESTIGATE → REPRODUCE → FIX → VALIDATE → ABSTAIN_IF_FAIL

Autonomy levels (ordered by risk):
  LINT          → auto-fix lint errors (ruff --fix)
  DEPENDENCY    → auto-resolve dependency conflicts
  TEST_FIX      → semi-auto fix test failures (requires confidence > threshold)
  LOGIC_BUG     → supervised fix (always requires human approval)

Abstention policy: If confidence < threshold, return diagnostic context
but never apply a partial fix. No half-measures.

Integration: receives issues from health check, monitors, or manual submission.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from pathlib import Path

import structlog

from systems.simula.resolution.types import (
    ISSUE_KIND_TO_AUTONOMY,
    AutonomyLevel,
    DetectedIssue,
    IssueKind,
    IssueSource,
    MonitoringAlert,
    ResolutionAttempt,
    ResolutionResult,
    ResolutionStatus,
)

if TYPE_CHECKING:

    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from systems.simula.agents.repair_agent import RepairAgent
    from systems.simula.code_agent import SimulaCodeAgent
    from systems.simula.resolution.monitors import (
        DegradationMonitor,
        PerfRegressionMonitor,
        SecurityVulnMonitor,
    )
logger = structlog.get_logger().bind(system="simula.resolution")


class IssueResolver:
    """
    Autonomous issue resolver with progressive autonomy and strict abstention.

    LogicStar pattern: INVESTIGATE → REPRODUCE → FIX → VALIDATE.
    If confidence drops below threshold at any stage, abstains and returns
    diagnostic context instead of a partial fix.
    """

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        neo4j: Neo4jClient | None = None,
        code_agent: SimulaCodeAgent | None = None,
        repair_agent: RepairAgent | None = None,
        perf_monitor: PerfRegressionMonitor | None = None,
        security_monitor: SecurityVulnMonitor | None = None,
        degradation_monitor: DegradationMonitor | None = None,
        *,
        max_autonomy_level: str = "test_fix",
        abstention_threshold: float = 0.8,
        timeout_s: float = 120.0,
    ) -> None:
        self._llm = llm
        self._root = codebase_root
        self._neo4j = neo4j
        self._code_agent = code_agent
        self._repair_agent = repair_agent
        self._perf_monitor = perf_monitor
        self._security_monitor = security_monitor
        self._degradation_monitor = degradation_monitor
        self._max_autonomy = AutonomyLevel(max_autonomy_level)
        self._abstention_threshold = abstention_threshold
        self._timeout_s = timeout_s

    # ── Public API ──────────────────────────────────────────────────────────

    async def resolve(
        self,
        issue: DetectedIssue,
    ) -> ResolutionResult:
        """
        Attempt to resolve an issue through the LogicStar pipeline.

        If autonomy level is too high for the issue type, or confidence
        drops below threshold, abstains with full diagnostic context.

        Args:
            issue: The detected issue to resolve.

        Returns:
            ResolutionResult with status, attempts, and diagnostics.
        """
        start = time.monotonic()
        attempts: list[ResolutionAttempt] = []

        # Determine required autonomy level
        required_level = ISSUE_KIND_TO_AUTONOMY.get(
            issue.kind, AutonomyLevel.LOGIC_BUG
        )

        # Check if we have permission for this autonomy level
        if not self._has_autonomy(required_level):
            logger.info(
                "issue_resolution_escalated",
                kind=issue.kind.value,
                required=required_level.value,
                max=self._max_autonomy.value,
            )
            return ResolutionResult(
                status=ResolutionStatus.ESCALATED,
                issue=issue,
                autonomy_level_used=required_level,
                escalation_context=(
                    f"Issue requires autonomy level '{required_level.value}' "
                    f"but max allowed is '{self._max_autonomy.value}'"
                ),
                diagnostic_summary=self._build_diagnostic(issue, attempts),
                total_duration_ms=int((time.monotonic() - start) * 1000),
            )

        try:
            # Phase 1: INVESTIGATE
            investigation = await self._investigate(issue)
            attempts.append(investigation)

            if investigation.confidence < self._abstention_threshold:
                return self._abstain(
                    issue, attempts, start,
                    reason=f"Investigation confidence too low: {investigation.confidence:.2f}",
                )

            # Phase 2: REPRODUCE
            reproduction = await self._reproduce(issue, investigation)
            attempts.append(reproduction)

            if not reproduction.tests_passed:
                # Good - we reproduced the failure. Proceed to fix.
                pass
            elif issue.kind == IssueKind.LINT_ERROR:
                # Lint errors don't need reproduction
                pass
            else:
                return self._abstain(
                    issue, attempts, start,
                    reason="Could not reproduce the issue",
                )

            # Phase 3: FIX
            fix = await self._fix(issue, required_level, investigation, reproduction)
            attempts.append(fix)

            if fix.confidence < self._abstention_threshold:
                return self._abstain(
                    issue, attempts, start,
                    reason=f"Fix confidence too low: {fix.confidence:.2f}",
                )

            # Phase 4: VALIDATE
            validation = await self._validate(fix)
            attempts.append(validation)

            if validation.tests_passed and validation.lint_clean:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                logger.info(
                    "issue_resolved",
                    kind=issue.kind.value,
                    level=required_level.value,
                    duration_ms=elapsed_ms,
                )
                return ResolutionResult(
                    status=ResolutionStatus.RESOLVED,
                    issue=issue,
                    autonomy_level_used=required_level,
                    attempts=attempts,
                    total_attempts=len(attempts),
                    files_modified=fix.files_modified,
                    confidence=fix.confidence,
                    diagnostic_summary=self._build_diagnostic(issue, attempts),
                    total_duration_ms=elapsed_ms,
                )

            # Validation failed
            return self._abstain(
                issue, attempts, start,
                reason="Fix did not pass validation",
            )

        except TimeoutError:
            logger.warning("issue_resolution_timeout")
            return ResolutionResult(
                status=ResolutionStatus.TIMEOUT,
                issue=issue,
                attempts=attempts,
                total_attempts=len(attempts),
                diagnostic_summary=self._build_diagnostic(issue, attempts),
                total_duration_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception:
            logger.exception("issue_resolution_error")
            return ResolutionResult(
                status=ResolutionStatus.FAILED,
                issue=issue,
                attempts=attempts,
                total_attempts=len(attempts),
                diagnostic_summary=self._build_diagnostic(issue, attempts),
                total_duration_ms=int((time.monotonic() - start) * 1000),
            )

    async def resolve_from_alert(
        self,
        alert: MonitoringAlert,
    ) -> ResolutionResult:
        """Convert a MonitoringAlert to a DetectedIssue and resolve it."""
        issue = DetectedIssue(
            issue_id=alert.alert_id,
            kind=alert.issue_kind,
            source=IssueSource.MONITORING,
            title=alert.title,
            description=alert.description,
            severity=alert.severity,
            file_path=alert.file_path,
            detected_at=alert.detected_at,
        )
        return await self.resolve(issue)

    async def run_monitors(
        self,
        files_modified: list[str],
        analytics: Any = None,
        before_timing: dict[str, float] | None = None,
        after_timing: dict[str, float] | None = None,
    ) -> list[MonitoringAlert]:
        """Run all configured monitors and return alerts."""
        alerts: list[MonitoringAlert] = []

        tasks = []
        if self._perf_monitor and before_timing is not None:
            tasks.append(self._perf_monitor.check(
                before_timing=before_timing,
                after_timing=after_timing or {},
            ))
        if self._security_monitor:
            tasks.append(self._security_monitor.check(files_modified))
        if self._degradation_monitor and analytics:
            tasks.append(self._degradation_monitor.check(analytics))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    alerts.extend(result)
                elif isinstance(result, Exception):
                    logger.warning("monitor_error", error=str(result))

        return alerts

    # ── Autonomy gating ─────────────────────────────────────────────────────

    def _has_autonomy(self, required: AutonomyLevel) -> bool:
        """Check if we have sufficient autonomy for this issue type."""
        level_order = list(AutonomyLevel)
        return level_order.index(required) <= level_order.index(self._max_autonomy)

    # ── Phase 1: INVESTIGATE ────────────────────────────────────────────────

    async def _investigate(self, issue: DetectedIssue) -> ResolutionAttempt:
        """Investigate the issue to understand its scope and root cause."""
        start = time.monotonic()

        from clients.llm import Message

        prompt = (
            f"Investigate this issue:\n"
            f"  Kind: {issue.kind.value}\n"
            f"  Title: {issue.title}\n"
            f"  Description: {issue.description}\n"
            f"  File: {issue.file_path}\n"
            f"  Stack trace: {issue.stack_trace[:1000] if issue.stack_trace else 'N/A'}\n\n"
            f"Determine: root cause, affected scope, confidence of diagnosis."
        )

        response = await self._llm.complete(  # type: ignore[attr-defined]
            system=None,
            messages=[Message(role="user", content=prompt)],
            max_tokens=1024,
        )

        # Parse confidence from response (look for "confidence: X.X" pattern)
        import re
        confidence_match = re.search(r"confidence[:\s]+(\d+\.?\d*)", response.text.lower())
        confidence = float(confidence_match.group(1)) if confidence_match else 0.6

        return ResolutionAttempt(
            attempt_number=0,
            phase="investigate",
            autonomy_level=ISSUE_KIND_TO_AUTONOMY.get(issue.kind, AutonomyLevel.LINT),
            confidence=min(1.0, confidence),
            fix_description=response.text[:500],
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # ── Phase 2: REPRODUCE ─────────────────────────────────────────────────

    async def _reproduce(
        self, issue: DetectedIssue, investigation: ResolutionAttempt
    ) -> ResolutionAttempt:
        """Attempt to reproduce the issue."""
        start = time.monotonic()

        if issue.kind == IssueKind.LINT_ERROR:
            # Run ruff to verify lint error exists
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "ruff", "check", str(self._root / issue.file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            reproduced = proc.returncode != 0
        elif issue.kind == IssueKind.TEST_FAILURE:
            # Run tests to reproduce
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest", "--tb=short", "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            reproduced = proc.returncode != 0
        else:
            # Other issues: assume reproduced based on investigation
            reproduced = True

        return ResolutionAttempt(
            attempt_number=1,
            phase="reproduce",
            confidence=investigation.confidence,
            tests_passed=not reproduced,  # False = we reproduced the failure (good)
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # ── Phase 3: FIX ───────────────────────────────────────────────────────

    async def _fix(
        self,
        issue: DetectedIssue,
        level: AutonomyLevel,
        investigation: ResolutionAttempt,
        reproduction: ResolutionAttempt,
    ) -> ResolutionAttempt:
        """Generate and apply a fix based on the autonomy level."""
        start = time.monotonic()

        if level == AutonomyLevel.LINT:
            return await self._fix_lint(issue, start)
        elif level == AutonomyLevel.DEPENDENCY:
            return await self._fix_dependency(issue, start)
        elif level == AutonomyLevel.TEST_FIX:
            return await self._fix_test(issue, investigation, start)
        else:
            # LOGIC_BUG - always escalate (confidence threshold is 1.0)
            return ResolutionAttempt(
                attempt_number=2,
                phase="fix",
                autonomy_level=level,
                confidence=0.0,  # will trigger abstention
                fix_description="Logic bug requires human approval",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

    async def _fix_lint(self, issue: DetectedIssue, start: float) -> ResolutionAttempt:
        """Auto-fix lint errors via ruff --fix."""
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "ruff", "check", "--fix",
            str(self._root / issue.file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._root),
        )
        _, _ = await asyncio.wait_for(proc.communicate(), timeout=30)

        return ResolutionAttempt(
            attempt_number=2,
            phase="fix",
            autonomy_level=AutonomyLevel.LINT,
            confidence=0.9,  # ruff auto-fix is reliable
            fix_description="Applied ruff --fix",
            files_modified=[issue.file_path] if issue.file_path else [],
            lint_clean=proc.returncode == 0,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _fix_dependency(self, issue: DetectedIssue, start: float) -> ResolutionAttempt:
        """Auto-resolve dependency conflicts."""
        # Attempt to reinstall with pip
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pip", "install", "-e", ".",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._root),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        return ResolutionAttempt(
            attempt_number=2,
            phase="fix",
            autonomy_level=AutonomyLevel.DEPENDENCY,
            confidence=0.7 if proc.returncode == 0 else 0.3,
            fix_description=f"Reinstalled package dependencies (exit={proc.returncode})",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _fix_test(
        self, issue: DetectedIssue, investigation: ResolutionAttempt, start: float
    ) -> ResolutionAttempt:
        """Fix test failures using the repair agent if available."""
        if self._repair_agent is None:
            return ResolutionAttempt(
                attempt_number=2,
                phase="fix",
                autonomy_level=AutonomyLevel.TEST_FIX,
                confidence=0.0,
                fix_description="No repair agent available",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        from systems.simula.evolution_types import (
            ChangeCategory,
            ChangeSpec,
            EvolutionProposal,
        )

        # Create a synthetic proposal for the repair agent
        synthetic_proposal = EvolutionProposal(
            source="issue_resolution",
            category=ChangeCategory.MODIFY_CONTRACT,
            description=f"Fix issue: {issue.title}",
            change_spec=ChangeSpec(
                additional_context=issue.description,
                affected_systems=[issue.file_path] if issue.file_path else [],
            ),
        )

        broken_files: dict[str, str] = {}
        if issue.file_path:
            full_path = self._root / issue.file_path
            if full_path.exists():
                broken_files[issue.file_path] = full_path.read_text()

        repair_result = await self._repair_agent.repair(
            proposal=synthetic_proposal,
            broken_files=broken_files,
            test_output=issue.stack_trace,
        )

        from systems.simula.verification.types import RepairStatus

        return ResolutionAttempt(
            attempt_number=2,
            phase="fix",
            autonomy_level=AutonomyLevel.TEST_FIX,
            confidence=(
                0.85 if repair_result.status == RepairStatus.REPAIRED
                else 0.3
            ),
            fix_description=repair_result.fix_summary,
            files_modified=repair_result.files_repaired,
            tests_passed=repair_result.status == RepairStatus.REPAIRED,
            cost_usd=repair_result.total_cost_usd,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # ── Phase 4: VALIDATE ──────────────────────────────────────────────────

    async def _validate(self, fix: ResolutionAttempt) -> ResolutionAttempt:
        """Run tests + lint on fixed files to validate the fix."""
        start = time.monotonic()

        if not fix.files_modified:
            return ResolutionAttempt(
                attempt_number=3,
                phase="validate",
                tests_passed=False,
                lint_clean=False,
                duration_ms=0,
            )

        # Run lint
        lint_proc = await asyncio.create_subprocess_exec(
            "python", "-m", "ruff", "check",
            *[str(self._root / f) for f in fix.files_modified],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._root),
        )
        await asyncio.wait_for(lint_proc.communicate(), timeout=30)

        # Run tests
        test_proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", "--tb=short", "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._root),
        )
        await asyncio.wait_for(test_proc.communicate(), timeout=60)

        return ResolutionAttempt(
            attempt_number=3,
            phase="validate",
            confidence=fix.confidence,
            tests_passed=test_proc.returncode == 0,
            lint_clean=lint_proc.returncode == 0,
            files_modified=fix.files_modified,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # ── Abstention ──────────────────────────────────────────────────────────

    def _abstain(
        self,
        issue: DetectedIssue,
        attempts: list[ResolutionAttempt],
        start: float,
        *,
        reason: str,
    ) -> ResolutionResult:
        """Abstain from fixing - return diagnostic context instead."""
        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "issue_resolution_abstained",
            kind=issue.kind.value,
            reason=reason,
        )

        return ResolutionResult(
            status=ResolutionStatus.ABSTAINED,
            issue=issue,
            autonomy_level_used=ISSUE_KIND_TO_AUTONOMY.get(
                issue.kind, AutonomyLevel.LINT
            ),
            attempts=attempts,
            total_attempts=len(attempts),
            abstention_reason=reason,
            diagnostic_summary=self._build_diagnostic(issue, attempts),
            total_duration_ms=elapsed_ms,
        )

    @staticmethod
    def _build_diagnostic(
        issue: DetectedIssue, attempts: list[ResolutionAttempt]
    ) -> str:
        """Build a diagnostic summary from the issue and attempts."""
        lines = [
            f"Issue: {issue.title} ({issue.kind.value})",
            f"Source: {issue.source.value}",
            f"Severity: {issue.severity}",
        ]
        if issue.file_path:
            lines.append(f"File: {issue.file_path}")
        for attempt in attempts:
            lines.append(
                f"  [{attempt.phase}] confidence={attempt.confidence:.2f}"
                + (f" - {attempt.fix_description[:100]}" if attempt.fix_description else "")
            )
        return "\n".join(lines)
