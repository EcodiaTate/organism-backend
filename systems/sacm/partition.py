"""
EcodiaOS - SACM Partition Recovery Policy

When a remote provider becomes unreachable mid-execution (network partition,
provider crash, region outage), the execution engine consults this policy to
decide the best recovery action given the workload's remaining deadline.

Decision tree:

  remaining = deadline - elapsed

  1. remaining < abort_grace_s
     → ABORT (no time left for any recovery)

  2. remaining > reprovision_overhead_s + wait_budget_s
     → WAIT (enough slack to wait for the original provider, then still
       reprovision if it doesn't come back)

  3. remaining > reprovision_overhead_s
     → REPROVISION (no time to wait, but enough to spin up elsewhere)

  4. estimated_duration_s <= local_fallback_max_duration_s
     AND remaining > estimated_duration_s
     → LOCAL_FALLBACK (workload is small enough to run locally)

  5. Otherwise
     → ABORT

The policy is stateless - it reads config thresholds and workload metadata,
returning a pure RecoveryAction value.  Retry counting and circuit-breaking
live in the execution layer.
"""

from __future__ import annotations

import enum
from typing import Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped
from systems.sacm.config import SACMPreWarmConfig
from systems.sacm.workload import WorkloadDescriptor, WorkloadPriority

logger = structlog.get_logger("systems.sacm.partition")


# ─── Recovery Action ─────────────────────────────────────────────────


class RecoveryActionKind(enum.StrEnum):
    """The four possible recovery strategies."""

    WAIT = "wait"
    """Hold the workload and wait for the original provider to recover."""

    REPROVISION = "reprovision"
    """Immediately re-place the workload on a different provider."""

    LOCAL_FALLBACK = "local_fallback"
    """Execute the workload locally (only viable for small workloads)."""

    ABORT = "abort"
    """Cancel the workload - no recovery path is feasible."""


class RecoveryAction(EOSBaseModel):
    """
    The policy's recommendation for how to handle a partition event.

    Consumers (typically RemoteExecutionManager) inspect `kind` and act
    accordingly.  The `detail` field provides human-readable reasoning
    for logging and dashboards.
    """

    kind: RecoveryActionKind
    detail: str = ""
    remaining_deadline_s: float = 0.0
    """Seconds remaining before the workload's hard deadline."""

    estimated_wait_s: float = 0.0
    """If kind=WAIT, how long the policy suggests waiting."""

    reprovision_attempt: int = 0
    """Current reprovision attempt number (1-based). 0 if not applicable."""

    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Partition Event ─────────────────────────────────────────────────


class PartitionEvent(Identified, Timestamped):
    """
    Record of a detected partition between SACM and a remote provider.

    Created by the execution layer and passed to the recovery policy.
    """

    workload_id: str
    provider_id: str
    elapsed_s: float
    """Seconds since the workload was dispatched to the provider."""

    previous_reprovision_attempts: int = 0
    """How many times we've already tried to reprovision this workload."""

    last_heartbeat_age_s: float = 0.0
    """Seconds since the last successful heartbeat from the provider."""

    error: str = ""
    """Transport-level error message, if available."""


# ─── Recovery Policy ─────────────────────────────────────────────────


class PartitionRecoveryPolicy:
    """
    Stateless policy that maps a (workload, partition_event) pair to a
    RecoveryAction.

    Usage:
        policy = PartitionRecoveryPolicy(config)
        action = policy.evaluate(workload, event)

        match action.kind:
            case RecoveryActionKind.WAIT:
                await asyncio.sleep(action.estimated_wait_s)
            case RecoveryActionKind.REPROVISION:
                new_plan = optimize_placement(workload, oracle.snapshot)
                ...
            case RecoveryActionKind.LOCAL_FALLBACK:
                result = await local_executor(workload)
            case RecoveryActionKind.ABORT:
                workload.status = WorkloadStatus.FAILED
    """

    def __init__(self, config: SACMPreWarmConfig | None = None) -> None:
        cfg = config or SACMPreWarmConfig()
        self._cfg = cfg.partition_recovery
        self._log = logger.bind(component="sacm.partition_recovery")

    def evaluate(
        self,
        workload: WorkloadDescriptor,
        event: PartitionEvent,
    ) -> RecoveryAction:
        """
        Compute the best recovery action given the workload constraints
        and the partition event details.

        The decision uses the workload's max_latency_s as the hard
        deadline.  If the workload has no latency constraint
        (max_latency_s == 0), the engine uses estimated_duration_s × 3
        as a generous soft deadline.
        """
        deadline = self._effective_deadline(workload)
        remaining = max(0.0, deadline - event.elapsed_s)

        self._log.info(
            "evaluating_partition",
            workload_id=workload.id,
            provider_id=event.provider_id,
            elapsed_s=round(event.elapsed_s, 1),
            deadline_s=round(deadline, 1),
            remaining_s=round(remaining, 1),
            previous_attempts=event.previous_reprovision_attempts,
        )

        # Gate 1: no time at all
        if remaining < self._cfg.abort_grace_s:
            return self._abort(
                remaining,
                f"remaining {remaining:.1f}s < abort_grace {self._cfg.abort_grace_s:.1f}s",
            )

        # Gate 2: enough slack to wait + reprovision as fallback
        wait_plus_reprovision = self._cfg.wait_budget_s + self._cfg.reprovision_overhead_s
        if remaining > wait_plus_reprovision:
            return RecoveryAction(
                kind=RecoveryActionKind.WAIT,
                remaining_deadline_s=remaining,
                estimated_wait_s=self._cfg.wait_budget_s,
                detail=(
                    f"Sufficient slack ({remaining:.1f}s remaining, need "
                    f"{wait_plus_reprovision:.1f}s for wait+reprovision). "
                    f"Waiting {self._cfg.wait_budget_s:.1f}s for provider "
                    f"{event.provider_id} to recover."
                ),
            )

        # Gate 3: enough time to reprovision (but not wait first)
        if (
            remaining > self._cfg.reprovision_overhead_s
            and event.previous_reprovision_attempts < self._cfg.max_reprovision_attempts
        ):
            attempt = event.previous_reprovision_attempts + 1
            return RecoveryAction(
                kind=RecoveryActionKind.REPROVISION,
                remaining_deadline_s=remaining,
                reprovision_attempt=attempt,
                detail=(
                    f"Reprovisioning (attempt {attempt}/{self._cfg.max_reprovision_attempts}). "
                    f"{remaining:.1f}s remaining, reprovision overhead "
                    f"{self._cfg.reprovision_overhead_s:.1f}s."
                ),
            )

        # Gate 4: local fallback for small workloads
        if self._can_fall_back_locally(workload, remaining):
            return RecoveryAction(
                kind=RecoveryActionKind.LOCAL_FALLBACK,
                remaining_deadline_s=remaining,
                detail=(
                    f"Local fallback viable: estimated duration "
                    f"{workload.estimated_duration_s:.1f}s fits within "
                    f"{remaining:.1f}s remaining and is under local max "
                    f"{self._cfg.local_fallback_max_duration_s:.1f}s."
                ),
            )

        # Gate 5: nothing left
        return self._abort(
            remaining,
            (
                f"No recovery path: {remaining:.1f}s remaining, "
                f"reprovision needs {self._cfg.reprovision_overhead_s:.1f}s, "
                f"attempts exhausted ({event.previous_reprovision_attempts}"
                f"/{self._cfg.max_reprovision_attempts}), "
                f"workload too large for local fallback "
                f"({workload.estimated_duration_s:.1f}s > "
                f"{self._cfg.local_fallback_max_duration_s:.1f}s)."
            ),
        )

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _effective_deadline(workload: WorkloadDescriptor) -> float:
        """
        Derive the hard deadline in seconds from workload constraints.

        If the workload has an explicit max_latency_s, use that.
        Otherwise, use 3× estimated_duration_s as a generous soft cap.
        Critical workloads get a tighter 2× multiplier.
        """
        if workload.has_latency_constraint:
            return workload.max_latency_s

        multiplier = 2.0 if workload.priority == WorkloadPriority.CRITICAL else 3.0
        return workload.estimated_duration_s * multiplier

    def _can_fall_back_locally(
        self,
        workload: WorkloadDescriptor,
        remaining_s: float,
    ) -> bool:
        """Check if the workload is small enough for local execution."""
        return (
            workload.estimated_duration_s <= self._cfg.local_fallback_max_duration_s
            and remaining_s > workload.estimated_duration_s
        )

    def _abort(self, remaining: float, detail: str) -> RecoveryAction:
        self._log.warning("partition_abort", remaining_s=round(remaining, 1), detail=detail)
        return RecoveryAction(
            kind=RecoveryActionKind.ABORT,
            remaining_deadline_s=remaining,
            detail=detail,
        )
