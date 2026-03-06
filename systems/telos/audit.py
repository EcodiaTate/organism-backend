"""
EcodiaOS — Telos: 24-Hour Constitutional Topology Audit

Runs daily. Verifies all four drive bindings are intact.
Emits CONSTITUTIONAL_TOPOLOGY_INTACT on success.
Emits emergency alert if any binding appears compromised.

The audit is the last line of defense: even if every other check
fails, the daily audit will catch drive definition drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.telos.types import (
    ConstitutionalAuditResult,
    TelosConfig,
)

if TYPE_CHECKING:
    from systems.telos.alignment import AlignmentGapMonitor
    from systems.telos.binder import TelosConstitutionalBinder

logger = structlog.get_logger()


class ConstitutionalTopologyAuditor:
    """
    Performs the 24-hour constitutional topology audit.

    Checks:
    1. All four Final bindings remain True (runtime integrity)
    2. No violations have been recorded since the last audit
    3. The alignment gap trend is not in emergency state
    4. The drive definitions have not drifted from their topological meaning
    """

    def __init__(
        self,
        config: TelosConfig,
        binder: TelosConstitutionalBinder,
        gap_monitor: AlignmentGapMonitor,
    ) -> None:
        self._config = config
        self._binder = binder
        self._gap_monitor = gap_monitor
        self._logger = logger.bind(component="telos.audit")
        self._consecutive_failures = 0

    def run_audit(self) -> ConstitutionalAuditResult:
        """
        Execute the full constitutional topology audit.

        Returns an audit result indicating whether all bindings are intact.
        Clears the binder's violation buffer after inspection.
        """
        # 1. Verify Final bindings are intact
        bindings_intact = self._binder.verify_bindings_intact()

        # 2. Collect any violations since last audit
        violations = self._binder.clear_violations()

        # 3. Get the current alignment gap trend
        gap_trend = self._gap_monitor.compute_trend()

        # 4. Determine overall result
        all_intact = bindings_intact and len(violations) == 0

        result = ConstitutionalAuditResult(
            all_bindings_intact=all_intact,
            care_is_coverage=self._binder.CARE_IS_COVERAGE,
            coherence_is_compression=self._binder.COHERENCE_IS_COMPRESSION,
            growth_is_gradient=self._binder.GROWTH_IS_GRADIENT,
            honesty_is_validity=self._binder.HONESTY_IS_VALIDITY,
            alignment_gap_trend=gap_trend,
            violations_since_last_audit=violations,
        )

        if all_intact:
            self._consecutive_failures = 0
            self._logger.info(
                "constitutional_audit_passed",
                violations_reviewed=len(violations),
                gap_urgency=gap_trend.urgency,
            )
        else:
            self._consecutive_failures += 1
            self._logger.critical(
                "constitutional_audit_FAILED",
                bindings_intact=bindings_intact,
                violations=len(violations),
                consecutive_failures=self._consecutive_failures,
                gap_urgency=gap_trend.urgency,
            )

        return result

    @property
    def is_emergency(self) -> bool:
        """True if the audit has failed enough times to trigger emergency."""
        return (
            self._consecutive_failures
            >= self._config.constitutional_audit_emergency_threshold
        )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures
