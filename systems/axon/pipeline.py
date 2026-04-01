"""
EcodiaOS - Axon Execution Pipeline

The pipeline is the core of Axon - it takes an Intent and executes it, but ONLY
after verifying Equor's constitutional verdict.

Pipeline stages:
  0. Equor gate - MANDATORY: reject intents that Equor did not approve
  1. Budget check - verify the per-cycle execution budget allows another action
  2. Validation - validate parameters for all steps against their executors
  3. Rate limit check - verify each executor is within its rate limit
  4. Circuit breaker check - verify no executor has a tripped circuit
  5. Context assembly - gather credentials, build ExecutionContext
  6. Step execution - execute each step with timeout, progress tracking, and rollback
  7. Outcome assembly - collect results, compute success/partial/failure
  8. Outcome delivery - async: audit log + deliver to Nova

All stages must complete within the total_timeout_per_cycle_ms budget.
Steps that exceed their per-executor timeout are cancelled and reported as failures.

If Equor's verdict is anything other than APPROVED or MODIFIED, the pipeline halts
immediately - no steps are executed, no credentials are issued.
If a step fails, the pipeline rolls back reversible steps and reports the failure -
it does not re-evaluate whether the intent was correct.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import Verdict
from systems.axon.types import (
    AxonOutcome,
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    FailureReason,
    RollbackResult,
    StepOutcome,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.axon.audit import AuditLogger
    from systems.axon.credentials import CredentialStore
    from systems.axon.registry import ExecutorRegistry
    from systems.axon.safety import BudgetTracker, CircuitBreaker, RateLimiter
    from systems.axon.shield import TransactionShield
    from systems.nova.service import NovaService

logger = structlog.get_logger()


class ExecutionPipeline:
    """
    The Axon execution pipeline.

    Receives approved ExecutionRequests and runs them through the 7-stage
    execution pipeline, delivering outcomes to Nova asynchronously.
    """

    def __init__(
        self,
        registry: ExecutorRegistry,
        budget: BudgetTracker,
        rate_limiter: RateLimiter,
        circuit_breaker: CircuitBreaker,
        credential_store: CredentialStore,
        audit_logger: AuditLogger,
        instance_id: str = "eos-default",
        shield: TransactionShield | None = None,
    ) -> None:
        self._registry = registry
        self._budget = budget
        self._rate_limiter = rate_limiter
        self._circuit_breaker = circuit_breaker
        self._credential_store = credential_store
        self._audit = audit_logger
        self._instance_id = instance_id
        self._shield = shield
        self._logger = logger.bind(system="axon.pipeline")
        self._nova: NovaService | None = None
        self._atune = None  # AtuneService -- for feeding outcomes as percepts

    def set_nova(self, nova: NovaService) -> None:
        self._nova = nova

    def set_atune(self, atune: Any) -> None:
        self._atune = atune

    def set_event_bus(self, event_bus: Any) -> None:
        """Wire the Synapse event bus for error reporting to Thymos."""
        self._event_bus = event_bus

    async def execute(self, request: ExecutionRequest) -> AxonOutcome:
        """
        Execute an approved Intent through all pipeline stages.

        Returns an AxonOutcome regardless of whether execution succeeded -
        failures are reported, not raised.
        """
        intent = request.intent
        start_time = time.monotonic()

        from primitives.common import new_id
        execution_id = new_id()

        self._logger.info(
            "pipeline_start",
            execution_id=execution_id,
            intent_id=intent.id,
            goal=intent.goal.description[:60],
            steps=len(intent.plan.steps),
        )

        # ── STAGE 0: Equor constitutional gate ────────────────────
        # This is the trust boundary. No intent may execute unless Equor
        # explicitly approved or conditionally modified it. Any other
        # verdict (BLOCKED, DEFERRED, SUSPENDED_AWAITING_HUMAN) halts
        # the pipeline before any side-effects occur.
        allowed_verdicts = frozenset({Verdict.APPROVED, Verdict.MODIFIED})
        equor = request.equor_check

        if equor is None or equor.verdict not in allowed_verdicts:
            verdict_label = equor.verdict.value if equor is not None else "missing"
            reasoning = equor.reasoning if equor is not None else "No constitutional check provided"
            self._logger.critical(
                "equor_gate_blocked",
                execution_id=execution_id,
                intent_id=intent.id,
                verdict=verdict_label,
                reasoning=reasoning[:200],
            )
            return self._fast_fail(
                intent_id=intent.id,
                execution_id=execution_id,
                status=ExecutionStatus.FAILURE,
                failure_reason="equor_constitutional_block",
                error="Blocked by Equor constitutional gate",
                start_time=start_time,
            )

        # ── STAGE 1: Budget check ──────────────────────────────────
        # Internal executors (store_insight, observe, wait, store) are exempt
        # from budget consumption - the budget exists to throttle external
        # actions and should never starve them.
        # An empty plan has no steps at all - treat as non-exempt so the budget
        # check still runs. `all()` of an empty iterable returns True, which
        # would silently bypass the budget for empty plans.
        all_exempt = bool(intent.plan.steps) and all(
            (ex := self._registry.get(step.executor)) is not None
            and not ex.counts_toward_budget
            for step in intent.plan.steps
        )
        if not all_exempt:
            allowed, reason = self._budget.can_execute()
            if not allowed:
                return self._fast_fail(
                    intent_id=intent.id,
                    execution_id=execution_id,
                    status=ExecutionStatus.RATE_LIMITED,
                    failure_reason=FailureReason.BUDGET_EXCEEDED.value,
                    error=reason,
                    start_time=start_time,
                )

        # ── STAGE 2: Validation ───────────────────────────────────
        for step in intent.plan.steps:
            executor = self._registry.get(step.executor)
            if executor is None:
                # Report to Thymos via Synapse event bus
                if self._event_bus is not None:
                    try:
                        await self._event_bus.emit(
                            SynapseEvent(
                                event_type=SynapseEventType.SYSTEM_FAILED,
                                source_system="axon",
                                data={"incident": {
                                    "incident_class": "crash",
                                    "severity": "high",
                                    "fingerprint": hashlib.md5(f"unknown_executor_{step.executor}".encode()).hexdigest(),
                                    "source_system": "axon",
                                    "error_type": "missing_executor",
                                    "error_message": f"No executor registered for '{step.executor}'",
                                    "context": {
                                        "action_type": step.executor,
                                        "intent_id": intent.id,
                                        "execution_id": execution_id,
                                    },
                                }},
                            )
                        )
                    except Exception as _emit_exc:
                        self._logger.warning(
                            "failed_to_emit_unknown_executor_incident",
                            error=str(_emit_exc)
                        )
                return self._fast_fail(
                    intent_id=intent.id,
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILURE,
                    failure_reason=FailureReason.UNKNOWN_ACTION_TYPE.value,
                    error=f"No executor registered for '{step.executor}'",
                    start_time=start_time,
                )

            try:
                validation = await executor.validate_params(step.parameters)
            except Exception as exc:
                return self._fast_fail(
                    intent_id=intent.id,
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILURE,
                    failure_reason=FailureReason.VALIDATION_ERROR.value,
                    error=(
                        f"Step '{step.executor}' validate_params raised: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    start_time=start_time,
                )
            if not validation.valid:
                return self._fast_fail(
                    intent_id=intent.id,
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILURE,
                    failure_reason=FailureReason.VALIDATION_ERROR.value,
                    error=f"Step '{step.executor}' validation failed: {validation.reason}",
                    start_time=start_time,
                )

            # Autonomy check
            if executor.required_autonomy > intent.autonomy_level_granted:
                return self._fast_fail(
                    intent_id=intent.id,
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILURE,
                    failure_reason=FailureReason.INSUFFICIENT_AUTONOMY.value,
                    error=(
                        f"Executor '{step.executor}' requires autonomy level "
                        f"{executor.required_autonomy}, but intent has "
                        f"{intent.autonomy_level_granted}"
                    ),
                    start_time=start_time,
                )

        # ── STAGE 3: Rate limit check ─────────────────────────────
        for step in intent.plan.steps:
            executor = self._registry.get(step.executor)
            if executor and not self._rate_limiter.check(
                executor.action_type, executor.rate_limit
            ):
                return self._fast_fail(
                    intent_id=intent.id,
                    execution_id=execution_id,
                    status=ExecutionStatus.RATE_LIMITED,
                    failure_reason=FailureReason.RATE_LIMITED.value,
                    error=f"Rate limit exceeded for executor '{step.executor}'",
                    start_time=start_time,
                )
            # Sub-limit check: API calls/min and notifications/hr (Spec §5.3)
            if executor and hasattr(self._budget, "can_execute_action_type"):
                sub_ok, sub_reason = self._budget.can_execute_action_type(
                    executor.action_type
                )
                if not sub_ok:
                    return self._fast_fail(
                        intent_id=intent.id,
                        execution_id=execution_id,
                        status=ExecutionStatus.RATE_LIMITED,
                        failure_reason=FailureReason.RATE_LIMITED.value,
                        error=sub_reason,
                        start_time=start_time,
                    )

        # ── STAGE 4: Circuit breaker check ───────────────────────
        for step in intent.plan.steps:
            executor = self._registry.get(step.executor)
            if executor and not self._circuit_breaker.can_execute(executor.action_type):
                return self._fast_fail(
                    intent_id=intent.id,
                    execution_id=execution_id,
                    status=ExecutionStatus.CIRCUIT_OPEN,
                    failure_reason=FailureReason.CIRCUIT_OPEN.value,
                    error=f"Circuit breaker open for executor '{step.executor}'",
                    start_time=start_time,
                )

        # ── STAGE 5: Context assembly ─────────────────────────────
        credentials = await self._credential_store.get_for_intent(intent)
        context = ExecutionContext(
            execution_id=execution_id,
            intent=intent,
            equor_check=request.equor_check,
            credentials=credentials,
            instance_id=self._instance_id,
        )

        # ── STAGE 5.5: Transaction shield ────────────────────────
        if self._shield is not None:
            from systems.axon.shield import SHIELDED_EXECUTORS

            for step in intent.plan.steps:
                if step.executor in SHIELDED_EXECUTORS:
                    sim = await self._shield.evaluate(
                        action_type=step.executor,
                        params=step.parameters,
                        context=context,
                    )
                    if not sim.passed:
                        self._logger.warning(
                            "shield_rejected",
                            execution_id=execution_id,
                            executor=step.executor,
                            reason=sim.revert_reason,
                        )
                        # Emit AXON_SHIELD_REJECTION so Thymos gets real-time
                        # incident channel for shield blocks (not just post-mortem).
                        if hasattr(self, "_event_bus") and self._event_bus is not None:
                            await self._event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.AXON_SHIELD_REJECTION,
                                source_system="axon",
                                data={
                                    "execution_id": execution_id,
                                    "executor": step.executor,
                                    "intent_id": intent.id,
                                    "rejection_reason": sim.revert_reason or "unknown",
                                    "check_type": self._classify_shield_rejection(sim),
                                    "params": {
                                        k: v for k, v in step.parameters.items()
                                        if k not in ("private_key", "seed", "mnemonic")
                                    },
                                },
                            ))
                        return self._fast_fail(
                            intent_id=intent.id,
                            execution_id=execution_id,
                            status=ExecutionStatus.FAILURE,
                            failure_reason=FailureReason.TRANSACTION_SHIELD_REJECTED.value,
                            error=f"Transaction shield rejected: {sim.revert_reason}",
                            start_time=start_time,
                        )

        # ── STAGE 6: Step execution ───────────────────────────────
        # Supports parallel execution: steps sharing the same parallel_group
        # run concurrently. Steps with no group run sequentially.
        # Only charge the budget for intents that contain budget-tracked executors.
        _charges_budget = not all_exempt
        if _charges_budget:
            self._budget.begin_execution(intent_id=intent.id)
        step_outcomes: list[StepOutcome] = []

        try:
            # Group steps into execution batches: parallel groups run together,
            # ungrouped steps run individually in order.
            batches = _group_steps_into_batches(intent.plan.steps)
            abort = False

            for batch in batches:
                if abort:
                    break

                # Calculate remaining time budget
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                remaining_ms = request.timeout_ms - elapsed_ms

                if remaining_ms <= 0:
                    # Budget exhausted -- skip remaining batches
                    for idx, step in batch:
                        executor = self._registry.get(step.executor)
                        action_type = executor.action_type if executor else step.executor
                        step_outcomes.append(StepOutcome(
                            step_index=idx,
                            action_type=action_type,
                            description=step.parameters.get("description", step.executor),
                            result=ExecutionResult(
                                success=False,
                                error=f"Skipped: total budget ({request.timeout_ms}ms) exhausted",
                            ),
                            duration_ms=0,
                        ))
                    self._logger.warning(
                        "pipeline_budget_exhausted_skipping_remaining",
                        execution_id=execution_id,
                        elapsed_ms=elapsed_ms,
                        budget_ms=request.timeout_ms,
                        steps_remaining=sum(len(b) for b in batches),
                    )
                    break

                if len(batch) == 1:
                    # Sequential execution (single step)
                    idx, step = batch[0]
                    step_result, should_abort = await self._execute_single_step(
                        idx, step, context, execution_id, start_time, request.timeout_ms,
                    )
                    step_outcomes.append(step_result)
                    if should_abort:
                        steps_to_rollback = step_outcomes[:-1]
                        rollback_results = await _rollback_completed(
                            step_outcomes=steps_to_rollback,
                            registry=self._registry,
                            context=context,
                        )
                        self._logger.warning(
                            "pipeline_step_failed_rolling_back",
                            execution_id=execution_id,
                            step_index=idx,
                            error=(step_result.result.error or "")[:100],
                            rollback_results=len(rollback_results),
                        )
                        # Mark rolled-back steps in their outcomes
                        for so in steps_to_rollback:
                            so.rolled_back = True
                        await self._emit_rollback_re_example(
                            execution_id=execution_id,
                            context=context,
                            failed_step=step_result,
                            rolled_back_steps=steps_to_rollback,
                            rollback_results=rollback_results,
                            trigger="sequential_step_failure",
                        )
                        abort = True
                else:
                    # Parallel execution (multiple steps in same group)
                    self._logger.debug(
                        "parallel_batch_start",
                        execution_id=execution_id,
                        batch_size=len(batch),
                        group=batch[0][1].parallel_group,
                    )
                    batch_outcomes = await self._execute_parallel_batch(
                        batch, context, execution_id, start_time, request.timeout_ms,
                    )
                    step_outcomes.extend(batch_outcomes)

                    # Check for failures in parallel batch
                    failed = [o for o in batch_outcomes if not o.result.success]
                    if failed:
                        # Check if all failed steps allow continue_on_failure
                        must_abort = any(
                            not batch[i][1].parameters.get("continue_on_failure", False)
                            for i, o in enumerate(batch_outcomes)
                            if not o.result.success
                        )
                        if must_abort:
                            rollback_results = await _rollback_completed(
                                step_outcomes=step_outcomes,
                                registry=self._registry,
                                context=context,
                            )
                            for so in step_outcomes:
                                if so.result.success:
                                    so.rolled_back = True
                            if failed:
                                await self._emit_rollback_re_example(
                                    execution_id=execution_id,
                                    context=context,
                                    failed_step=failed[0],
                                    rolled_back_steps=[s for s in step_outcomes if s.result.success],
                                    rollback_results=rollback_results,
                                    trigger="parallel_batch_failure",
                                )
                            abort = True

        finally:
            if _charges_budget:
                self._budget.end_execution()

        # ── STAGE 7: Outcome assembly ─────────────────────────────
        total_duration_ms = int((time.monotonic() - start_time) * 1000)
        all_succeeded = bool(step_outcomes) and all(
            s.result.success for s in step_outcomes
        )
        any_succeeded = any(s.result.success for s in step_outcomes)
        partial = any_succeeded and not all_succeeded

        outcome = AxonOutcome(
            intent_id=intent.id,
            execution_id=execution_id,
            success=all_succeeded,
            partial=partial,
            status=ExecutionStatus.SUCCESS if all_succeeded else (
                ExecutionStatus.PARTIAL if partial else ExecutionStatus.FAILURE
            ),
            step_outcomes=step_outcomes,
            duration_ms=total_duration_ms,
        )
        outcome.world_state_changes = outcome.collect_world_changes()
        outcome.new_observations = outcome.collect_new_observations()
        if not all_succeeded:
            outcome.failure_reason = outcome.classify_failure()

        self._logger.info(
            "pipeline_complete",
            execution_id=execution_id,
            intent_id=intent.id,
            success=all_succeeded,
            partial=partial,
            steps_completed=len(step_outcomes),
            duration_ms=total_duration_ms,
        )

        # ── STAGE 8: Audit + Nova delivery (concurrent) ───────────
        await asyncio.gather(
            self._audit.log(outcome, context),
            self._deliver_to_nova(outcome),
            self._contribute_to_atune(outcome),
            return_exceptions=True,
        )

        return outcome

    async def _execute_single_step(
        self,
        idx: int,
        step: Any,
        context: ExecutionContext,
        execution_id: str,
        pipeline_start: float,
        total_timeout_ms: int,
    ) -> tuple[StepOutcome, bool]:
        """Execute a single step sequentially. Returns (outcome, should_abort)."""
        executor = self._registry.get(step.executor)
        if executor is None:
            return StepOutcome(
                step_index=idx,
                action_type=step.executor,
                description=step.parameters.get("description", step.executor),
                result=ExecutionResult(
                    success=False,
                    error=f"Missing executor: {step.executor}",
                ),
                duration_ms=0,
            ), True

        elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
        remaining_ms = total_timeout_ms - elapsed_ms
        step_timeout_ms = min(
            step.timeout_ms * 3,
            executor.max_duration_ms,
            remaining_ms,
        )
        step_timeout_ms = max(step_timeout_ms, 100)

        step_start = time.monotonic()
        result = await _run_step_with_timeout(
            executor=executor,
            params=step.parameters,
            context=context,
            timeout_ms=step_timeout_ms,
        )
        step_duration_ms = int((time.monotonic() - step_start) * 1000)

        self._rate_limiter.record(executor.action_type)
        self._circuit_breaker.record_result(executor.action_type, result.success)
        # Record for sub-limit sliding windows (API calls/min, notifications/hr)
        if result.success and hasattr(self._budget, "record_action_type"):
            self._budget.record_action_type(executor.action_type)

        outcome = StepOutcome(
            step_index=idx,
            action_type=executor.action_type,
            description=step.parameters.get("description", step.executor),
            result=result,
            duration_ms=step_duration_ms,
        )

        self._logger.debug(
            "step_complete",
            execution_id=execution_id,
            step_index=idx,
            action_type=executor.action_type,
            success=result.success,
            duration_ms=step_duration_ms,
        )

        should_abort = (
            not result.success
            and not step.parameters.get("continue_on_failure", False)
        )
        return outcome, should_abort

    async def _execute_parallel_batch(
        self,
        batch: list[tuple[int, Any]],
        context: ExecutionContext,
        execution_id: str,
        pipeline_start: float,
        total_timeout_ms: int,
    ) -> list[StepOutcome]:
        """Execute a batch of independent steps concurrently."""
        elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
        remaining_ms = total_timeout_ms - elapsed_ms

        async def run_one(idx: int, step: Any) -> StepOutcome:
            executor = self._registry.get(step.executor)
            if executor is None:
                return StepOutcome(
                    step_index=idx,
                    action_type=step.executor,
                    description=step.parameters.get("description", step.executor),
                    result=ExecutionResult(
                    success=False,
                    error=f"Missing executor: {step.executor}",
                ),
                    duration_ms=0,
                )

            step_timeout_ms = min(
                step.timeout_ms * 3,
                executor.max_duration_ms,
                remaining_ms,
            )
            step_timeout_ms = max(step_timeout_ms, 100)

            step_start = time.monotonic()
            result = await _run_step_with_timeout(
                executor=executor,
                params=step.parameters,
                context=context,
                timeout_ms=step_timeout_ms,
            )
            step_duration_ms = int((time.monotonic() - step_start) * 1000)

            self._rate_limiter.record(executor.action_type)
            self._circuit_breaker.record_result(executor.action_type, result.success)
            if result.success and hasattr(self._budget, "record_action_type"):
                self._budget.record_action_type(executor.action_type)

            self._logger.debug(
                "parallel_step_complete",
                execution_id=execution_id,
                step_index=idx,
                action_type=executor.action_type,
                success=result.success,
                duration_ms=step_duration_ms,
            )

            return StepOutcome(
                step_index=idx,
                action_type=executor.action_type,
                description=step.parameters.get("description", step.executor),
                result=result,
                duration_ms=step_duration_ms,
            )

        tasks = [run_one(idx, step) for idx, step in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outcomes: list[StepOutcome] = []
        for i, (idx, step) in enumerate(batch):
            r = results[i]
            if isinstance(r, BaseException):
                outcomes.append(StepOutcome(
                    step_index=idx,
                    action_type=step.executor,
                    description=step.parameters.get("description", step.executor),
                    result=ExecutionResult(
                        success=False,
                        error=f"Parallel step raised: {type(r).__name__}: {r}",
                    ),
                    duration_ms=0,
                ))
            else:
                outcomes.append(r)
        return outcomes

    async def _deliver_to_nova(self, outcome: AxonOutcome) -> None:
        """
        Deliver the execution outcome to Nova via Synapse event bus.

        Emits ACTION_EXECUTED or ACTION_FAILED so Nova can update beliefs
        without requiring a direct import of nova.types. All consumers
        (Nova, Evo, Fovea) subscribe to these events - no direct call needed.

        If the event bus is not wired, logs a warning. The direct Nova fallback
        has been removed to enforce bus-first architecture (AV2 resolved).
        """
        if not hasattr(self, "_event_bus") or self._event_bus is None:
            self._logger.warning(
                "nova_delivery_skipped_no_event_bus",
                intent_id=outcome.intent_id,
                execution_id=outcome.execution_id,
                note="Set event bus via set_event_bus() to enable Nova outcome delivery",
            )
            return

        event_type = (
            SynapseEventType.ACTION_EXECUTED if outcome.success
            else SynapseEventType.ACTION_FAILED
        )
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=event_type,
                source_system="axon",
                data={
                    "intent_id": outcome.intent_id,
                    "execution_id": outcome.execution_id,
                    "success": outcome.success,
                    "episode_id": outcome.episode_id,
                    "failure_reason": outcome.failure_reason,
                    "new_observations": outcome.new_observations,
                    "duration_ms": outcome.duration_ms,
                    "step_outcomes": [
                        {
                            "action_type": s.action_type,
                            "success": s.result.success,
                            "error": s.result.error,
                            "duration_ms": s.duration_ms,
                        }
                        for s in outcome.step_outcomes
                    ],
                },
            ))
        except Exception as exc:
            self._logger.error(
                "nova_delivery_event_emit_failed",
                intent_id=outcome.intent_id,
                error=str(exc),
            )

    async def _contribute_to_atune(self, outcome: AxonOutcome) -> None:
        """
        Feed the execution outcome into Atune's workspace as a self-perception.

        The organism perceives its own actions: successes are routine (low salience),
        failures are salient and demand attention.

        Internal executors (store_insight, update_goal, trigger_consolidation) are
        excluded - broadcasting their outcomes creates a tight feedback loop where
        every store_insight triggers a new workspace broadcast → deliberation →
        store_insight ad infinitum.
        """
        if self._atune is None:
            return

        # Skip Atune contribution if all steps are from executors that opt out.
        if outcome.step_outcomes:
            all_silent = all(
                (ex := self._registry.get(so.action_type)) is not None
                and not ex.emits_to_atune
                for so in outcome.step_outcomes
            )
            if all_silent:
                return

        if outcome.success:
            content = (
                f"Action completed: {outcome.execution_id} "
                f"({len(outcome.step_outcomes)} steps)"
            )
            priority = 0.25  # Routine - success is expected
        else:
            content = (
                f"Action failed: {outcome.execution_id} - "
                f"{outcome.failure_reason or 'unknown'}"
            )
            priority = 0.55  # Failure is salient and demands attention

        try:
            # Use dict payload to avoid importing fovea.types at runtime
            self._atune.contribute({
                "system": "axon",
                "content": content,
                "priority": priority,
                "reason": "action_outcome",
            })
        except Exception as exc:
            self._logger.debug("atune_contribution_failed", error=str(exc))

    async def _emit_rollback_re_example(
        self,
        execution_id: str,
        context: ExecutionContext,
        failed_step: StepOutcome,
        rolled_back_steps: list[StepOutcome],
        rollback_results: list[RollbackResult],
        trigger: str,
    ) -> None:
        """
        Emit a dedicated RE training example for rollback decisions.

        Rollback examples are among the highest-value curriculum because they
        capture the organism correcting its own mistakes: the moment a step
        fails, the principled decision to undo prior side-effects, and the
        downstream consequences if it had not.
        """
        if not hasattr(self, "_event_bus") or self._event_bus is None:
            return
        try:
            import json as _json

            from primitives.common import DriveAlignmentVector, SystemID
            from primitives.re_training import RETrainingExample

            reversible_count = sum(
                1 for r in rollback_results if r.success
            )
            non_reversible_count = len(rollback_results) - reversible_count

            # Describe each rollback attempt
            rollback_detail_parts = []
            for i, (so, rr) in enumerate(zip(rolled_back_steps, rollback_results)):
                status = "REVERSED" if rr.success else f"IRREVERSIBLE({rr.reason[:60]})"
                side_effects_note = (
                    f", side_effects_reversed={rr.side_effects_reversed}"
                    if rr.side_effects_reversed else ""
                )
                rollback_detail_parts.append(
                    f"  [{i}] {so.action_type}: {status}{side_effects_note}"
                )
            rollback_detail = "\n".join(rollback_detail_parts) if rollback_detail_parts else "  (no prior steps to roll back)"

            # Constitutional alignment from the Equor check
            equor_alignment = None
            equor_verdict = "approved"
            if context.equor_check is not None:
                equor_alignment = getattr(context.equor_check, "drive_alignment", None)
                equor_verdict = (
                    context.equor_check.verdict.value
                    if hasattr(context.equor_check.verdict, "value")
                    else str(context.equor_check.verdict)
                )

            reasoning_trace = "\n".join([
                f"1. TRIGGER: {trigger} - step '{failed_step.action_type}' failed: {(failed_step.result.error or 'unknown')[:200]}",
                f"2. ROLLBACK DECISION: {len(rolled_back_steps)} prior step(s) completed with side-effects that must be reversed",
                f"3. REVERSIBILITY SCAN: {reversible_count} reversible, {non_reversible_count} non-reversible (financial/external ops cannot be undone)",
                f"4. ROLLBACK EXECUTION (LIFO order):\n{rollback_detail}",
                f"5. NET STATE: {reversible_count}/{len(rolled_back_steps)} side-effects reversed; {non_reversible_count} permanent side-effects remain",
                f"6. EQUOR CONTEXT: original verdict={equor_verdict}; rollback does not require re-evaluation (Axon self-correction)",
            ])

            alternatives = [
                f"Alternative: continue_on_failure=True would skip rollback and let downstream steps proceed with partial state - risks downstream incoherence",
                f"Alternative: re-queue the intent with a patched plan after root cause of '{failed_step.action_type}' failure is diagnosed",
            ]
            if non_reversible_count > 0:
                alternatives.append(
                    f"WARNING: {non_reversible_count} non-reversible step(s) cannot be undone - "
                    f"manual compensation (refund, notification, record correction) may be required"
                )

            if non_reversible_count > 0:
                counterfactual = (
                    f"If rollback had not been triggered after '{failed_step.action_type}' failed, "
                    f"the {len(rolled_back_steps)} prior step(s) would have left persistent side-effects "
                    f"with no follow-through, corrupting downstream intent coherence. "
                    f"Note: {non_reversible_count} non-reversible side-effect(s) persist regardless."
                )
            elif reversible_count > 0:
                counterfactual = (
                    f"If rollback had not been triggered, {reversible_count} completed step(s) would "
                    f"have left dangling state changes (records created, tasks scheduled) with no "
                    f"corresponding completion - requiring manual cleanup or producing silent inconsistencies."
                )
            else:
                counterfactual = (
                    f"No prior steps had side-effects to reverse. Rollback was a no-op, but the "
                    f"abort signal correctly prevented the remaining steps from executing in broken state."
                )

            structured_output = _json.dumps({
                "trigger": trigger,
                "failed_executor": failed_step.action_type,
                "failure_error": (failed_step.result.error or "")[:200],
                "prior_steps_count": len(rolled_back_steps),
                "rollback_attempted": len(rollback_results),
                "rollback_succeeded": reversible_count,
                "rollback_failed_non_reversible": non_reversible_count,
                "rollback_correct": True,  # Axon always rolls back on failure - this is curriculum for why
            })

            example = RETrainingExample(
                source_system=SystemID.AXON,
                episode_id=context.execution_id,
                category="rollback_decision",
                instruction=(
                    f"Step '{failed_step.action_type}' failed mid-execution. "
                    f"Decide whether to roll back {len(rolled_back_steps)} completed prior step(s) "
                    f"and execute the rollback in LIFO order."
                ),
                input_context=(
                    f"execution_id={context.execution_id}, "
                    f"intent_id={context.intent.id}, "
                    f"failed_step={failed_step.action_type}, "
                    f"error={(failed_step.result.error or '')[:200]!r}, "
                    f"prior_steps=[{', '.join(s.action_type for s in rolled_back_steps)}], "
                    f"trigger={trigger}"
                ),
                output=structured_output,
                outcome_quality=1.0,  # Rollback itself is always the correct decision - quality measures the decision, not the failure
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives,
                constitutional_alignment=equor_alignment or DriveAlignmentVector(),
                counterfactual=counterfactual,
            )

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon",
                data=example.model_dump(mode="json"),
            ))

            # Also emit AXON_ROLLBACK_INITIATED so Thymos can track
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.AXON_ROLLBACK_INITIATED,
                source_system="axon",
                data={
                    "execution_id": execution_id,
                    "intent_id": context.intent.id,
                    "failed_step": failed_step.action_type,
                    "steps_rolled_back": reversible_count,
                    "non_reversible_steps": non_reversible_count,
                    "trigger": trigger,
                },
            ))

        except Exception:
            self._logger.debug("rollback_re_emit_failed", exc_info=True)

    @staticmethod
    def _classify_shield_rejection(sim: Any) -> str:
        """Classify the shield rejection type from SimulationResult."""
        reason = (sim.revert_reason or "").lower()
        if "blacklist" in reason:
            return "blacklist"
        if "slippage" in reason:
            return "slippage"
        if "gas" in reason or "roi" in reason:
            return "gas_roi"
        if "mev" in reason or "sandwich" in reason or "frontrun" in reason:
            return "mev"
        return "unknown"

    def _fast_fail(
        self,
        intent_id: str,
        execution_id: str,
        status: ExecutionStatus,
        failure_reason: str,
        error: str,
        start_time: float,
    ) -> AxonOutcome:
        """Create a fast-fail outcome without executing any steps."""
        duration_ms = int((time.monotonic() - start_time) * 1000)
        self._logger.warning(
            "pipeline_fast_fail",
            intent_id=intent_id,
            execution_id=execution_id,
            status=status.value,
            failure_reason=failure_reason,
            error=error[:100],
        )
        return AxonOutcome(
            intent_id=intent_id,
            execution_id=execution_id,
            success=False,
            status=status,
            failure_reason=failure_reason,
            error=error,
            duration_ms=duration_ms,
        )


# ─── Helpers ──────────────────────────────────────────────────────


def _group_steps_into_batches(
    steps: list[Any],
) -> list[list[tuple[int, Any]]]:
    """
    Group steps into execution batches based on parallel_group.

    Steps with the same non-None parallel_group are batched together for
    concurrent execution. Steps with no group run sequentially (batch of 1).
    Consecutive steps sharing a group form one batch; the ordering of batches
    preserves the original step order.
    """
    if not steps:
        return []

    batches: list[list[tuple[int, Any]]] = []
    current_group: str | None = None
    current_batch: list[tuple[int, Any]] = []

    for i, step in enumerate(steps):
        group = getattr(step, "parallel_group", None)

        if group is not None and group == current_group:
            # Continue the current parallel batch
            current_batch.append((i, step))
        else:
            # Flush the previous batch
            if current_batch:
                batches.append(current_batch)
            current_batch = [(i, step)]
            current_group = group

    if current_batch:
        batches.append(current_batch)

    return batches


async def _run_step_with_timeout(
    executor: Any,
    params: dict[str, Any],
    context: ExecutionContext,
    timeout_ms: int,
) -> ExecutionResult:
    """Run a single step executor with timeout enforcement."""
    try:
        return await asyncio.wait_for(
            executor.execute(params, context),
            timeout=timeout_ms / 1000,
        )
    except TimeoutError:
        return ExecutionResult(
            success=False,
            error=f"Step timed out after {timeout_ms}ms",
        )
    except Exception as exc:
        return ExecutionResult(
            success=False,
            error=f"Step raised exception: {type(exc).__name__}: {exc}",
        )


async def _rollback_completed(
    step_outcomes: list[StepOutcome],
    registry: ExecutorRegistry,
    context: ExecutionContext,
) -> list[RollbackResult]:
    """
    Attempt to rollback completed steps in reverse order (most recent first).
    Best-effort - non-reversible steps are reported as not-supported.
    """
    results: list[RollbackResult] = []

    for step_outcome in reversed(step_outcomes):
        if not step_outcome.result.success:
            continue  # Failed steps don't need rollback

        executor = registry.get(step_outcome.action_type)
        if executor is None or not executor.reversible:
            results.append(RollbackResult(
                success=False,
                reason=f"Executor '{step_outcome.action_type}' is not reversible",
            ))
            continue

        try:
            rollback_result = await executor.rollback(
                context.execution_id,
                context,
            )
            results.append(rollback_result)
        except Exception as exc:
            results.append(RollbackResult(
                success=False,
                reason=f"Rollback exception: {exc}",
            ))
            logger.error(
                "rollback_failed",
                step_index=step_outcome.step_index,
                action_type=step_outcome.action_type,
                error=str(exc),
            )

    return results
