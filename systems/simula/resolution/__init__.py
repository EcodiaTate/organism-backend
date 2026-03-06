"""
EcodiaOS -- Simula Autonomous Issue Resolution Subsystem (Stage 5E)

Progressive-autonomy issue resolver with strict abstention policy.
Receives issues from health check, monitors, CI/CD, or manual submission.

  IssueResolver  — LogicStar pattern: investigate → reproduce → fix → validate → abstain
  Monitors       — Perf regression, security vuln, degradation detection
"""

from systems.simula.resolution.issue_resolver import IssueResolver
from systems.simula.resolution.monitors import (
    DegradationMonitor,
    PerfRegressionMonitor,
    SecurityVulnMonitor,
)
from systems.simula.resolution.types import (
    AUTONOMY_LEVEL_THRESHOLDS,
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

__all__ = [
    # Engines
    "IssueResolver",
    "PerfRegressionMonitor",
    "SecurityVulnMonitor",
    "DegradationMonitor",
    # Types
    "AutonomyLevel",
    "IssueKind",
    "IssueSource",
    "ResolutionStatus",
    "DetectedIssue",
    "ResolutionAttempt",
    "ResolutionResult",
    "MonitoringAlert",
    "AUTONOMY_LEVEL_THRESHOLDS",
    "ISSUE_KIND_TO_AUTONOMY",
]
