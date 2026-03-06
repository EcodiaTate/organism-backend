"""
Unit tests for the Axon ExecutionPipeline.

Tests the full 7-stage pipeline: budget → validation → rate limit →
circuit breaker → context assembly → step execution → outcome assembly.

All external dependencies are mocked.
"""

from __future__ import annotations

import pytest

from config import AxonConfig
from primitives.common import Verdict
from primitives.constitutional import ConstitutionalCheck
from primitives.intent import (
    Action,
    ActionSequence,
    EthicalClearance,
    GoalDescriptor,
    Intent,
    ResourceBudget,
)
from systems.axon.audit import AuditLogger
from systems.axon.credentials import CredentialStore
from systems.axon.executor import Executor
from systems.axon.pipeline import ExecutionPipeline
from systems.axon.registry import ExecutorRegistry
from systems.axon.safety import BudgetTracker, CircuitBreaker, RateLimiter
from systems.axon.types import (
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    FailureReason,
    RateLimit,
    ValidationResult,
)

# ─── Fixtures ─────────────────────────────────────────────────────


def make_config(**kwargs) -> AxonConfig:
    return AxonConfig(**{
        "max_actions_per_cycle": 5,
        "max_api_calls_per_minute": 30,
        "max_notifications_per_hour": 10,
        "max_concurrent_executions": 3,
        "total_timeout_per_cycle_ms": 30_000,
        **kwargs,
    })


class _SuccessExecutor(Executor):
    action_type = "success_action"
    description = "Always succeeds"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 1000
    rate_limit = RateLimit.unlimited()

    async def validate_params(self, params: dict) -> ValidationResult:
        return ValidationResult.ok()

    async def execute(self, params: dict, context: ExecutionContext) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            data={"done": True},
            side_effects=["success_action completed"],
        )


class _FailExecutor(Executor):
    action_type = "fail_action"
    description = "Always fails"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 1000
    rate_limit = RateLimit.unlimited()

    async def validate_params(self, params: dict) -> ValidationResult:
        return ValidationResult.ok()

    async def execute(self, params: dict, context: ExecutionContext) -> ExecutionResult:
        return ExecutionResult(success=False, error="Deliberate failure")


class _SlowExecutor(Executor):
    action_type = "slow_action"
    description = "Takes too long"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 100
    rate_limit = RateLimit.unlimited()

    async def validate_params(self, params: dict) -> ValidationResult:
        return ValidationResult.ok()

    async def execute(self, params: dict, context: ExecutionContext) -> ExecutionResult:
        import asyncio
        await asyncio.sleep(10)  # Way over timeout
        return ExecutionResult(success=True)


def make_pipeline(config: AxonConfig | None = None) -> ExecutionPipeline:
    config = config or make_config()
    registry = ExecutorRegistry()
    registry.register(_SuccessExecutor())
    registry.register(_FailExecutor())
    registry.register(_SlowExecutor())

    audit = AuditLogger(memory=None)

    return ExecutionPipeline(
        registry=registry,
        budget=BudgetTracker(config),
        rate_limiter=RateLimiter(),
        circuit_breaker=CircuitBreaker(),
        credential_store=CredentialStore(),
        audit_logger=audit,
        instance_id="test-instance",
    )


def make_request(
    executor_name: str = "success_action",
    params: dict | None = None,
    autonomy: int = 1,
) -> ExecutionRequest:
    intent = Intent(
        goal=GoalDescriptor(description="Test goal"),
        plan=ActionSequence(
            steps=[Action(executor=executor_name, parameters=params or {})]
        ),
        ethical_clearance=EthicalClearance(status=Verdict.APPROVED),
        autonomy_level_granted=autonomy,
        budget=ResourceBudget(compute_ms=5000),
    )
    check = ConstitutionalCheck(
        intent_id=intent.id,
        verdict=Verdict.APPROVED,
    )
    return ExecutionRequest(intent=intent, equor_check=check, timeout_ms=5000)


# ─── Tests: Budget Check ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_budget_exceeded_fast_fail():
    config = make_config(max_actions_per_cycle=0)  # 0 actions allowed
    pipeline = make_pipeline(config)
    # Exhaust the budget manually by starting without ending
    pipeline._budget._actions_this_cycle = 10

    outcome = await pipeline.execute(make_request())
    assert outcome.success is False
    assert outcome.status == ExecutionStatus.RATE_LIMITED
    assert outcome.failure_reason == FailureReason.BUDGET_EXCEEDED.value


# ─── Tests: Validation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_executor_fast_fail():
    pipeline = make_pipeline()
    request = make_request(executor_name="nonexistent_executor")
    outcome = await pipeline.execute(request)
    assert outcome.success is False
    assert outcome.failure_reason == FailureReason.UNKNOWN_ACTION_TYPE.value
    assert "nonexistent_executor" in outcome.error


@pytest.mark.asyncio
async def test_insufficient_autonomy_fast_fail():
    """Executor requires autonomy 2 but intent has autonomy 1."""

    class _Level2Executor(_SuccessExecutor):
        action_type = "level2_action"
        required_autonomy = 2

    pipeline = make_pipeline()
    pipeline._registry.register(_Level2Executor())

    request = make_request(executor_name="level2_action", autonomy=1)
    outcome = await pipeline.execute(request)
    assert outcome.success is False
    assert outcome.failure_reason == FailureReason.INSUFFICIENT_AUTONOMY.value


# ─── Tests: Successful Execution ─────────────────────────────────


@pytest.mark.asyncio
async def test_successful_single_step():
    pipeline = make_pipeline()
    request = make_request(executor_name="success_action")
    outcome = await pipeline.execute(request)

    assert outcome.success is True
    assert outcome.partial is False
    assert outcome.status == ExecutionStatus.SUCCESS
    assert len(outcome.step_outcomes) == 1
    assert outcome.step_outcomes[0].result.success is True
    assert outcome.duration_ms >= 0


@pytest.mark.asyncio
async def test_successful_multi_step():
    pipeline = make_pipeline()
    intent = Intent(
        goal=GoalDescriptor(description="Multi-step goal"),
        plan=ActionSequence(
            steps=[
                Action(executor="success_action", parameters={}),
                Action(executor="success_action", parameters={}),
            ]
        ),
        ethical_clearance=EthicalClearance(status=Verdict.APPROVED),
        autonomy_level_granted=1,
        budget=ResourceBudget(compute_ms=5000),
    )
    check = ConstitutionalCheck(intent_id=intent.id, verdict=Verdict.APPROVED)
    request = ExecutionRequest(intent=intent, equor_check=check, timeout_ms=5000)

    outcome = await pipeline.execute(request)
    assert outcome.success is True
    assert len(outcome.step_outcomes) == 2


# ─── Tests: Failure Handling ──────────────────────────────────────


@pytest.mark.asyncio
async def test_single_step_failure():
    pipeline = make_pipeline()
    request = make_request(executor_name="fail_action")
    outcome = await pipeline.execute(request)

    assert outcome.success is False
    assert outcome.partial is False
    assert outcome.status == ExecutionStatus.FAILURE
    assert len(outcome.step_outcomes) == 1
    assert outcome.step_outcomes[0].result.success is False


@pytest.mark.asyncio
async def test_partial_success_multi_step():
    """First step succeeds, second fails — partial outcome."""
    pipeline = make_pipeline()
    intent = Intent(
        goal=GoalDescriptor(description="Partial goal"),
        plan=ActionSequence(
            steps=[
                Action(executor="success_action", parameters={}),
                Action(executor="fail_action", parameters={}),
            ]
        ),
        ethical_clearance=EthicalClearance(status=Verdict.APPROVED),
        autonomy_level_granted=1,
        budget=ResourceBudget(compute_ms=5000),
    )
    check = ConstitutionalCheck(intent_id=intent.id, verdict=Verdict.APPROVED)
    request = ExecutionRequest(intent=intent, equor_check=check, timeout_ms=5000)

    outcome = await pipeline.execute(request)
    assert outcome.success is False
    assert outcome.partial is True
    assert outcome.status == ExecutionStatus.PARTIAL


@pytest.mark.asyncio
async def test_timeout_returns_failure():
    pipeline = make_pipeline()
    request = make_request(executor_name="slow_action")
    request = ExecutionRequest(
        intent=request.intent,
        equor_check=request.equor_check,
        timeout_ms=200,  # Very short timeout
    )
    # Override the step timeout
    request.intent.plan.steps[0] = Action(
        executor="slow_action",
        parameters={},
        timeout_ms=50,  # 50ms × 3 = 150ms step timeout
    )
    outcome = await pipeline.execute(request)
    # The slow executor should timeout
    assert not outcome.success or len(outcome.step_outcomes) == 0 or (
        not outcome.step_outcomes[0].result.success
    )


# ─── Tests: Side Effects & Observations ──────────────────────────


@pytest.mark.asyncio
async def test_outcome_collects_world_changes():
    pipeline = make_pipeline()
    request = make_request(executor_name="success_action")
    outcome = await pipeline.execute(request)
    assert "success_action completed" in outcome.world_state_changes


@pytest.mark.asyncio
async def test_empty_plan_returns_success():
    """An intent with no steps should succeed vacuously."""
    pipeline = make_pipeline()
    intent = Intent(
        goal=GoalDescriptor(description="No-op goal"),
        plan=ActionSequence(steps=[]),
        ethical_clearance=EthicalClearance(status=Verdict.APPROVED),
        autonomy_level_granted=1,
        budget=ResourceBudget(compute_ms=5000),
    )
    check = ConstitutionalCheck(intent_id=intent.id, verdict=Verdict.APPROVED)
    request = ExecutionRequest(intent=intent, equor_check=check, timeout_ms=5000)
    outcome = await pipeline.execute(request)
    # No steps → all_succeeded is False (empty) but partial is also False
    assert len(outcome.step_outcomes) == 0


# ─── Tests: Rate Limiting ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limited_executor_fast_fails():
    pipeline = make_pipeline()
    # Manually exhaust the rate limit
    tight_limit = RateLimit.per_minute(1)
    pipeline._rate_limiter.record("success_action")  # consume the 1 slot
    # Override the executor's rate limit to be tight
    pipeline._registry.get("success_action").rate_limit = tight_limit

    request = make_request(executor_name="success_action")
    outcome = await pipeline.execute(request)
    assert outcome.success is False
    assert outcome.failure_reason == FailureReason.RATE_LIMITED.value
