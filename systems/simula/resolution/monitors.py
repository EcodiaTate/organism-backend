"""
EcodiaOS -- Simula Monitors (Stage 5E.4)

Monitoring subsystems that detect issues proactively:
  - PerfRegressionMonitor: Compares test timing before/after apply
  - SecurityVulnMonitor:   Static analysis + CVE pattern matching
  - DegradationMonitor:    Tracks metrics over time from analytics

Each monitor produces MonitoringAlerts that feed into the IssueResolver.
"""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path

import structlog

from primitives.common import new_id
from systems.simula.resolution.types import (
    IssueKind,
    MonitoringAlert,
)

logger = structlog.get_logger().bind(system="simula.resolution.monitors")


# ── Performance Regression Monitor ──────────────────────────────────────────


class PerfRegressionMonitor:
    """
    Detect performance regressions by comparing test timing before/after.

    Collects test duration metrics from pytest --durations output and
    flags significant slowdowns (>20% increase in total test time).
    """

    def __init__(
        self,
        *,
        regression_threshold: float = 0.20,
    ) -> None:
        self._threshold = regression_threshold

    async def check(
        self,
        before_timing: dict[str, float] | None = None,
        after_timing: dict[str, float] | None = None,
        before_total_s: float = 0.0,
        after_total_s: float = 0.0,
    ) -> list[MonitoringAlert]:
        """
        Compare before/after test timings and flag regressions.

        Args:
            before_timing: {test_name: duration_s} before the change.
            after_timing: {test_name: duration_s} after the change.
            before_total_s: Total test suite time before.
            after_total_s: Total test suite time after.

        Returns:
            List of alerts for detected regressions.
        """
        alerts: list[MonitoringAlert] = []

        # Check total suite time
        if before_total_s > 0 and after_total_s > 0:
            increase = (after_total_s - before_total_s) / before_total_s
            if increase > self._threshold:
                alerts.append(MonitoringAlert(
                    alert_id=f"perf_{new_id()[:8]}",
                    monitor_type="perf_regression",
                    issue_kind=IssueKind.PERF_REGRESSION,
                    title="Test suite performance regression",
                    description=(
                        f"Total test time increased by {increase:.1%}: "
                        f"{before_total_s:.1f}s → {after_total_s:.1f}s"
                    ),
                    severity="high" if increase > 0.5 else "medium",
                    metric_name="total_test_duration_s",
                    metric_value=after_total_s,
                    threshold=before_total_s * (1 + self._threshold),
                ))

        # Check individual test regressions
        if before_timing and after_timing:
            for test_name, after_time in after_timing.items():
                before_time = before_timing.get(test_name, 0)
                if before_time > 0.1:  # ignore very fast tests
                    increase = (after_time - before_time) / before_time
                    if increase > self._threshold * 2:  # higher bar for individual tests
                        alerts.append(MonitoringAlert(
                            alert_id=f"perf_{new_id()[:8]}",
                            monitor_type="perf_regression",
                            issue_kind=IssueKind.PERF_REGRESSION,
                            title=f"Test '{test_name}' slowed by {increase:.1%}",
                            description=(
                                f"{test_name}: {before_time:.3f}s → {after_time:.3f}s"
                            ),
                            severity="medium",
                            metric_name="test_duration_s",
                            metric_value=after_time,
                            threshold=before_time * (1 + self._threshold * 2),
                        ))

        if alerts:
            logger.info("perf_regressions_detected", count=len(alerts))

        return alerts


# ── Security Vulnerability Monitor ──────────────────────────────────────────


# Known dangerous patterns (simplified CVE matching)
_SECURITY_PATTERNS: list[tuple[str, str, str]] = [
    (r"eval\s*\(", "Potential code injection via eval()", "high"),
    (r"exec\s*\(", "Potential code injection via exec()", "high"),
    (r"subprocess\.(?:call|run|Popen)\s*\(.*shell\s*=\s*True", "Shell injection risk", "high"),
    (r"pickle\.loads?\s*\(", "Deserialization vulnerability via pickle", "high"),
    (r"yaml\.load\s*\([^)]*\)(?!.*Loader)", "Unsafe YAML loading (no Loader specified)", "medium"),
    (r"__import__\s*\(", "Dynamic import - potential code injection", "medium"),
    (r"os\.system\s*\(", "Shell command execution via os.system()", "medium"),
    (r"tempfile\.mk(?:stemp|dtemp)\s*\((?![^)]*dir=)", "Temp file without explicit directory", "low"),
    (r"hashlib\.md5\s*\(", "Weak hash function (MD5)", "low"),
    (r"hashlib\.sha1\s*\(", "Weak hash function (SHA1)", "low"),
]


class SecurityVulnMonitor:
    """
    Detect security vulnerabilities via pattern matching on modified files.

    Supplements the existing static_analysis.py with CVE-style pattern
    matching for common Python security issues.
    """

    def __init__(self, codebase_root: Path) -> None:
        self._root = codebase_root
        self._patterns = [
            (re.compile(p), desc, sev) for p, desc, sev in _SECURITY_PATTERNS
        ]

    async def check(
        self,
        files_modified: list[str],
    ) -> list[MonitoringAlert]:
        """
        Scan modified files for security vulnerability patterns.

        Args:
            files_modified: Files to scan (relative paths).

        Returns:
            List of alerts for detected vulnerabilities.
        """
        alerts: list[MonitoringAlert] = []

        for file_path in files_modified:
            full_path = self._root / file_path
            if not full_path.exists() or full_path.suffix != ".py":
                continue

            try:
                content = full_path.read_text()
            except OSError:
                continue

            for pattern, description, severity in self._patterns:
                for match in pattern.finditer(content):
                    # Find line number
                    line_num = content[:match.start()].count("\n") + 1
                    alerts.append(MonitoringAlert(
                        alert_id=f"sec_{new_id()[:8]}",
                        monitor_type="security_vuln",
                        issue_kind=IssueKind.SECURITY_VULN,
                        title=description,
                        description=(
                            f"{description} at {file_path}:{line_num}\n"
                            f"Match: {match.group(0)[:80]}"
                        ),
                        severity=severity,
                        file_path=file_path,
                    ))

        if alerts:
            logger.warning(
                "security_vulns_detected",
                count=len(alerts),
                files=files_modified,
            )

        return alerts


# ── Degradation Monitor ─────────────────────────────────────────────────────


class DegradationMonitor:
    """
    Detect gradual degradation in evolution quality metrics.

    Tracks rollback rates, success rates, and risk levels over a
    configurable window from the EvolutionAnalyticsEngine.
    """

    def __init__(
        self,
        *,
        window_hours: int = 24,
        rollback_rate_threshold: float = 0.3,
        success_rate_threshold: float = 0.5,
    ) -> None:
        self._window_hours = window_hours
        self._rollback_threshold = rollback_rate_threshold
        self._success_threshold = success_rate_threshold

    async def check(
        self,
        analytics: Any = None,
    ) -> list[MonitoringAlert]:
        """
        Check analytics for degradation patterns.

        Args:
            analytics: EvolutionAnalytics object from analytics engine.

        Returns:
            List of alerts for detected degradation.
        """
        alerts: list[MonitoringAlert] = []

        if analytics is None:
            return alerts

        # Check overall rollback rate
        rollback_rate = getattr(analytics, "rollback_rate", 0.0)
        if rollback_rate > self._rollback_threshold:
            alerts.append(MonitoringAlert(
                alert_id=f"deg_{new_id()[:8]}",
                monitor_type="degradation",
                issue_kind=IssueKind.DEGRADATION,
                title="High rollback rate detected",
                description=(
                    f"Overall rollback rate is {rollback_rate:.1%} "
                    f"(threshold: {self._rollback_threshold:.1%})"
                ),
                severity="high",
                metric_name="rollback_rate",
                metric_value=rollback_rate,
                threshold=self._rollback_threshold,
            ))

        # Check per-category recent rollback rates
        recent_rates = getattr(analytics, "recent_rollback_rates", {})
        for category, rate in recent_rates.items():
            if rate > self._rollback_threshold * 1.5:
                alerts.append(MonitoringAlert(
                    alert_id=f"deg_{new_id()[:8]}",
                    monitor_type="degradation",
                    issue_kind=IssueKind.DEGRADATION,
                    title=f"Category '{category}' degradation",
                    description=(
                        f"Recent rollback rate for {category}: {rate:.1%}"
                    ),
                    severity="medium",
                    metric_name=f"rollback_rate_{category}",
                    metric_value=rate,
                    threshold=self._rollback_threshold * 1.5,
                ))

        # Check evolution velocity (stall detection)
        velocity = getattr(analytics, "evolution_velocity", 0.0)
        total = getattr(analytics, "total_proposals", 0)
        if total > 10 and velocity < 0.1:  # less than 1 proposal per 10 days
            alerts.append(MonitoringAlert(
                alert_id=f"deg_{new_id()[:8]}",
                monitor_type="degradation",
                issue_kind=IssueKind.DEGRADATION,
                title="Evolution velocity stall",
                description=(
                    f"Evolution velocity is {velocity:.2f} proposals/day "
                    f"(expected > 0.1)"
                ),
                severity="low",
                metric_name="evolution_velocity",
                metric_value=velocity,
                threshold=0.1,
            ))

        if alerts:
            logger.info("degradation_detected", count=len(alerts))

        return alerts
