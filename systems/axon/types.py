"""
EcodiaOS — Axon Internal Types

All types internal to Axon's action execution system.

Design notes:
- AxonOutcome is Axon's rich internal result. It carries step-level detail,
  world-state changes, and rollback metadata. When reporting back to Nova,
  it is converted to IntentOutcome (from nova/types.py), which is the
  cross-system outcome primitive.
- ExecutionContext is assembled per-execution — it carries the approved intent,
  Equor verdict, scoped credentials, and current affect. Executors receive it
  but cannot modify it.
- ScopedCredentials wraps time-limited tokens issued per-execution. Executors
  never see raw secrets.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003 — Pydantic requires at runtime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.affect import AffectState
from primitives.common import EOSBaseModel, Identified, Timestamped, new_id, utc_now

if TYPE_CHECKING:

    from primitives.constitutional import ConstitutionalCheck
    from primitives.intent import Intent
# ─── Enums ────────────────────────────────────────────────────────


class ExecutionStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ROLLED_BACK = "rolled_back"
    TIMED_OUT = "timed_out"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    DEFERRED = "deferred"  # Queued for later (e.g. sleep safety)
    SUSPENDED_HITL = "suspended_hitl"  # Paused awaiting human-in-the-loop


class FailureReason(enum.StrEnum):
    UNKNOWN_ACTION_TYPE = "unknown_action_type"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    INSUFFICIENT_AUTONOMY = "insufficient_autonomy"
    TIMEOUT = "timeout"
    STEP_FAILED = "step_failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CREDENTIAL_ERROR = "credential_error"
    EXECUTION_EXCEPTION = "execution_exception"
    TRANSACTION_SHIELD_REJECTED = "transaction_shield_rejected"


class CircuitStatus(enum.StrEnum):
    CLOSED = "closed"      # Normal — executions allowed
    OPEN = "open"          # Tripped — executions blocked
    HALF_OPEN = "half_open"  # Recovering — limited executions allowed


# ─── Primitive Execution Types ────────────────────────────────────


class ValidationResult(EOSBaseModel):
    """Result of parameter validation before execution."""

    valid: bool
    reason: str = ""
    field_errors: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def ok(cls) -> ValidationResult:
        return cls(valid=True)

    @classmethod
    def fail(cls, reason: str, **field_errors: str) -> ValidationResult:
        return cls(valid=False, reason=reason, field_errors=field_errors)


class RollbackResult(EOSBaseModel):
    """Result of attempting to roll back a completed step."""

    success: bool
    reason: str = ""
    side_effects_reversed: list[str] = Field(default_factory=list)


class ExecutionResult(EOSBaseModel):
    """
    The result of executing a single action step.

    The data dict carries step-specific output — query results, created IDs,
    API responses. Callers should not depend on specific keys without checking
    the executor's documentation.
    """

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    # Human-readable descriptions of what changed in the world
    side_effects: list[str] = Field(default_factory=list)
    # New observations to feed back as Percepts (e.g., API response content)
    new_observations: list[str] = Field(default_factory=list)


class StepOutcome(EOSBaseModel):
    """
    The recorded outcome of a single plan step.
    Collected across all steps to form the full AxonOutcome.
    """

    step_index: int
    action_type: str   # executor name (e.g., "store_insight", "call_api")
    description: str
    result: ExecutionResult
    duration_ms: int
    rolled_back: bool = False


# ─── Execution Context ────────────────────────────────────────────


class ScopedCredentials(EOSBaseModel):
    """
    Time-limited, scope-restricted tokens for external services.
    Issued per-execution by CredentialStore — executors never see raw secrets.
    """

    tokens: dict[str, str] = Field(default_factory=dict)
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None

    def get(self, service: str) -> str | None:
        return self.tokens.get(service)


class ExecutionContext(EOSBaseModel):
    """
    The complete context for one intent execution.
    Assembled once, passed to all executors — read-only from executor perspective.
    """

    execution_id: str = Field(default_factory=new_id)
    intent: Intent
    equor_check: ConstitutionalCheck
    credentials: ScopedCredentials = Field(default_factory=ScopedCredentials)
    instance_id: str = "eos-default"
    affect_state: AffectState = Field(default_factory=AffectState.neutral)
    started_at: datetime = Field(default_factory=utc_now)

    model_config = {"arbitrary_types_allowed": True}


# ─── Execution Budget & Rate Limiting ────────────────────────────


@dataclass
class ExecutionBudget:
    """
    Per-cycle limits on action execution.
    Non-negotiable safety valve — cannot be overridden by Nova or Simula.
    Only governance (Equor amendment) can change these limits.
    """

    max_actions_per_cycle: int = 5
    max_api_calls_per_minute: int = 30
    max_notifications_per_hour: int = 10
    max_federation_messages_per_hour: int = 50
    max_concurrent_executions: int = 3
    total_timeout_per_cycle_ms: int = 30_000


@dataclass
class RateLimit:
    """Rate limit definition for an executor."""

    max_calls: int
    window_seconds: int

    @classmethod
    def unlimited(cls) -> RateLimit:
        return cls(max_calls=10_000, window_seconds=1)

    @classmethod
    def per_minute(cls, max_calls: int) -> RateLimit:
        return cls(max_calls=max_calls, window_seconds=60)

    @classmethod
    def per_hour(cls, max_calls: int) -> RateLimit:
        return cls(max_calls=max_calls, window_seconds=3600)

    @classmethod
    def per_day(cls, max_calls: int) -> RateLimit:
        return cls(max_calls=max_calls, window_seconds=86400)


# ─── Circuit Breaker State ────────────────────────────────────────


@dataclass
class CircuitState:
    """Per-executor circuit breaker state."""

    status: CircuitStatus = CircuitStatus.CLOSED
    consecutive_failures: int = 0
    tripped_at: float = 0.0
    half_open_calls: int = 0


# ─── Execution Request & Outcome ─────────────────────────────────


class ExecutionRequest(EOSBaseModel):
    """
    The input to AxonService.execute().
    Carries the approved Intent plus the Equor verdict that cleared it.
    """

    intent: Intent
    equor_check: ConstitutionalCheck
    timeout_ms: int = 30_000

    model_config = {"arbitrary_types_allowed": True}


class AxonOutcome(EOSBaseModel):
    """
    Axon's rich execution outcome.

    This is Axon-internal and carries full step-level detail.
    When reporting back to Nova via IntentOutcome, it is converted to
    the shared primitive (nova.types.IntentOutcome).
    """

    intent_id: str
    execution_id: str
    success: bool
    partial: bool = False
    status: ExecutionStatus = ExecutionStatus.PENDING
    failure_reason: str = ""
    error: str = ""
    step_outcomes: list[StepOutcome] = Field(default_factory=list)
    duration_ms: int = 0
    world_state_changes: list[str] = Field(default_factory=list)
    new_observations: list[str] = Field(default_factory=list)
    episode_id: str = ""

    def classify_failure(self) -> str:
        """Derive a failure reason from step outcomes."""
        if self.failure_reason:
            return self.failure_reason
        for step in self.step_outcomes:
            if not step.result.success and step.result.error:
                return step.result.error[:100]
        return FailureReason.STEP_FAILED.value

    def collect_world_changes(self) -> list[str]:
        """Aggregate side effects across all steps."""
        changes: list[str] = []
        for step in self.step_outcomes:
            changes.extend(step.result.side_effects)
        return changes

    def collect_new_observations(self) -> list[str]:
        """Aggregate new observations from all steps."""
        obs: list[str] = []
        for step in self.step_outcomes:
            obs.extend(step.result.new_observations)
        return obs


# ─── Audit Record ─────────────────────────────────────────────────


class AuditRecord(Identified, Timestamped):
    """
    Permanent record of every action taken by Axon.
    Stored as (:GovernanceRecord {type: "action_audit"}) in the Memory graph.

    Parameters are hashed, not stored raw, to protect sensitive data.
    """

    execution_id: str
    intent_id: str
    equor_verdict: str
    equor_reasoning: str
    action_type: str
    # SHA-256 of JSON-serialised parameters — never raw
    parameters_hash: str
    target: str            # What system/entity was acted upon
    result: str            # "success" | "failure" | "partial" | "rolled_back"
    duration_ms: int
    affect_state: AffectState = Field(default_factory=AffectState.neutral)
    autonomy_level: int = 1

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_outcome(
        cls,
        outcome: AxonOutcome,
        context: ExecutionContext,
        parameters: dict[str, Any],
        action_type: str,
    ) -> AuditRecord:
        params_json = json.dumps(parameters, sort_keys=True, default=str)
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()

        if outcome.success:
            result = "success"
        elif outcome.partial:
            result = "partial"
        else:
            result = "failure"

        return cls(
            execution_id=outcome.execution_id,
            intent_id=outcome.intent_id,
            equor_verdict=context.equor_check.verdict.value
            if hasattr(context.equor_check, "verdict")
            else "approved",
            equor_reasoning=context.equor_check.reasoning
            if hasattr(context.equor_check, "reasoning")
            else "",
            action_type=action_type,
            parameters_hash=params_hash,
            target=str(
                next(
                    (
                        step_params.get("target")
                        or step_params.get("to")
                        or step_params.get("destination")
                        for step_params in parameters.values()
                        if isinstance(step_params, dict)
                        and (
                            step_params.get("target")
                            or step_params.get("to")
                            or step_params.get("destination")
                        )
                    ),
                    action_type,
                )
            ),
            result=result,
            duration_ms=outcome.duration_ms,
            affect_state=context.affect_state,
            autonomy_level=context.intent.autonomy_level_granted,
        )
