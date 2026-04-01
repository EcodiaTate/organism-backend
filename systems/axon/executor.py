"""
EcodiaOS - Axon Executor ABC

All action executors implement this interface.

An Executor is a stateless (or near-stateless) handler for one category of action.
It knows:
  - What action type it handles (action_type)
  - How to validate parameters for that action (validate_params)
  - How to perform the action (execute)
  - Whether it can undo the action (rollback, if reversible=True)

Executors are registered with the ExecutorRegistry at startup.
They receive an ExecutionContext that carries credentials, intent, and affect
state - but they must not mutate it.

Design principle: an Executor does exactly one thing, precisely, reliably.
The deliberation (what to do) happens in Nova. The judgement (whether to do it)
happens in Equor. Axon just does it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    RollbackResult,
    ValidationResult,
)


class Executor(ABC):
    """
    Base class for all Axon action executors.

    Subclass this and register with the ExecutorRegistry to add new capabilities.
    """

    # ── Identity ─────────────────────────────────────────────────
    action_type: str = ""           # Must be unique - used as registry key
    description: str = ""           # Human-readable capability description

    # ── Safety ───────────────────────────────────────────────────
    required_autonomy: int = 1      # Minimum AutonomyLevel required to execute
    reversible: bool = False        # If True, override rollback()
    max_duration_ms: int = 5000     # Hard execution timeout
    rate_limit: RateLimit = RateLimit.per_minute(30)

    # ── Budget / Atune participation ─────────────────────────────
    # Internal executors (store_insight, observe, wait, store) should not
    # consume the per-cycle action budget - the budget exists to throttle
    # external actions (GitHub, LLM, financial).  Set to False on internal
    # executors so they never starve external work like hunt_bounties.
    counts_toward_budget: bool = True
    # Internal executors should not broadcast their outcomes back into
    # Atune's workspace - doing so creates a tight feedback loop where
    # every store_insight triggers a new broadcast → deliberation → store_insight.
    emits_to_atune: bool = True

    @abstractmethod
    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Perform the action described by params.

        Must complete within max_duration_ms.
        Must not raise - return ExecutionResult(success=False, error=...) on failure.
        May return new_observations that will be fed back as Percepts.
        """
        ...

    @abstractmethod
    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """
        Validate that params are sufficient and well-formed for this executor.

        Called before rate-limit checks, before context assembly, before execution.
        Must be fast (synchronous-level latency) - no I/O.
        """
        ...

    async def rollback(
        self,
        execution_id: str,
        context: ExecutionContext,
    ) -> RollbackResult:
        """
        Attempt to undo a previously completed execution.

        Override this if reversible=True. The default returns not-supported,
        which is correct for most executors - only data mutation actions
        (create_record, update_record, schedule_event) should implement this.
        """
        return RollbackResult(success=False, reason="Rollback not supported by this executor")

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"action_type={self.action_type!r} "
            f"autonomy={self.required_autonomy} "
            f"reversible={self.reversible}>"
        )
