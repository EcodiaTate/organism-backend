"""
EcodiaOS - Axon Fast-Path Executor

The terminal stage of the Arbitrage Reflex Arc. Receives FastPathIntents
from Atune's MarketPatternDetector and executes them directly against the
appropriate Axon executor, bypassing the standard 8-stage ExecutionPipeline.

Trust model:
  The FastPathIntent carries a ConstitutionalTemplate's pre-approval.
  Equor reviewed the strategy class at template registration time; we
  don't re-review individual executions. Instead, we enforce:
    1. max_capital ceiling (from the template)
    2. Executor-level rate limiting (from the standard rate limiter)
    3. Per-template circuit breaker (from the TemplateLibrary)

What we skip (for speed):
  - Nova's policy generation (0-10s savings)
  - Equor's full 8-stage verdict pipeline (up to 800ms savings)
  - Equor's community invariant LLM checks (up to 400ms savings)
  - EFE scoring (200ms savings)

What we keep (for safety):
  - Rate limiting per executor
  - Capital ceiling enforcement
  - Circuit breaker per template
  - Full audit logging
  - Outcome delivery to TemplateLibrary (for circuit breaker)
  - Atune contribution (for self-perception loop)

Latency target: ≤150ms from FastPathIntent to executor.execute() return.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id
from primitives.fast_path import FastPathIntent, FastPathOutcome
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    ScopedCredentials,
)

if TYPE_CHECKING:
    from systems.axon.audit import AuditLogger
    from systems.axon.registry import ExecutorRegistry
    from systems.axon.safety import RateLimiter
    from systems.equor.template_library import TemplateLibrary

logger = structlog.get_logger()


class FastPathExecutor:
    """
    Executes FastPathIntents with sub-200ms latency.

    Bypasses Nova deliberation and Equor review. Safety is maintained via:
      - Capital ceiling (template.max_capital_per_execution)
      - Rate limiting (standard Axon rate limiter)
      - Circuit breaker (TemplateLibrary.record_failure/record_success)
      - Full audit trail
    """

    def __init__(
        self,
        registry: ExecutorRegistry,
        template_library: TemplateLibrary,
        rate_limiter: RateLimiter,
        audit_logger: AuditLogger,
        instance_id: str = "eos-default",
    ) -> None:
        self._registry = registry
        self._templates = template_library
        self._rate_limiter = rate_limiter
        self._audit = audit_logger
        self._instance_id = instance_id
        self._logger = logger.bind(system="axon.fast_path")

        # Metrics
        self._total_executions: int = 0
        self._successful_executions: int = 0
        self._failed_executions: int = 0
        self._capital_deployed: float = 0.0
        self._total_latency_ms: int = 0

    async def execute(self, intent: FastPathIntent) -> FastPathOutcome:
        """
        Execute a FastPathIntent against the appropriate Axon executor.

        Returns FastPathOutcome with latency breakdown and execution result.
        Never raises - failures are captured in the outcome.
        """
        start = time.monotonic()
        self._total_executions += 1
        execution_id = new_id()

        self._logger.info(
            "fast_path_execute_start",
            execution_id=execution_id,
            template_id=intent.template_id,
            executor=intent.executor_type,
            max_capital=intent.max_capital,
        )

        # ── Gate 1: Template still active and fresh? ────────────────
        # Use get() for O(1) lookup but apply staleness check ourselves -
        # drives may have changed since template registration.
        template = self._templates.get(intent.template_id)
        if template is None or not template.active:
            return self._fail(
                intent, execution_id, start,
                error=f"Template '{intent.template_id}' not found or inactive",
            )

        # Reject stale templates: drives or invariants may have shifted since
        # the last Equor review.  60s matches TemplateLibrary._STALENESS_WINDOW_S.
        from primitives.common import utc_now as _utc_now
        _age_s = (_utc_now() - template.last_approved_at).total_seconds()
        _FAST_PATH_MAX_AGE_S = 60.0
        if _age_s > _FAST_PATH_MAX_AGE_S:
            return self._fail(
                intent, execution_id, start,
                error=(
                    f"Template '{intent.template_id}' is stale "
                    f"({_age_s:.0f}s since last Equor review; max {_FAST_PATH_MAX_AGE_S:.0f}s). "
                    "Equor must re-evaluate the template before it can be used."
                ),
            )

        # ── Gate 2: Capital ceiling ────────────────────────────────
        amount_str = intent.execution_params.get("amount", "0")
        try:
            requested_capital = float(amount_str)
        except (ValueError, TypeError):
            requested_capital = 0.0

        if requested_capital > intent.max_capital:
            return self._fail(
                intent, execution_id, start,
                error=(
                    f"Requested capital ${requested_capital:.2f} exceeds "
                    f"template ceiling ${intent.max_capital:.2f}"
                ),
            )

        # ── Gate 3: Executor exists? ───────────────────────────────
        executor = self._registry.get(intent.executor_type)
        if executor is None:
            return self._fail(
                intent, execution_id, start,
                error=f"No executor registered for '{intent.executor_type}'",
            )

        # ── Gate 4: Rate limit ─────────────────────────────────────
        if not self._rate_limiter.check(executor.action_type, executor.rate_limit):
            return self._fail(
                intent, execution_id, start,
                error=f"Rate limit exceeded for executor '{executor.action_type}'",
            )

        # ── Execute ────────────────────────────────────────────────
        pattern_match_ms = int((time.monotonic() - start) * 1000)

        # Build a minimal ExecutionContext for the executor
        context = self._build_context(intent, execution_id)

        exec_start = time.monotonic()
        try:
            result = await executor.execute(intent.execution_params, context)
        except Exception as exc:
            result = ExecutionResult(
                success=False,
                error=f"Executor raised: {type(exc).__name__}: {exc}",
            )
        exec_ms = int((time.monotonic() - exec_start) * 1000)

        # Record rate limit usage
        self._rate_limiter.record(executor.action_type)

        # ── Outcome assembly ───────────────────────────────────────
        total_ms = int((time.monotonic() - start) * 1000)

        capital_deployed = requested_capital if result.success else 0.0
        try:
            capital_returned = float(result.data.get("capital_returned", 0.0))
        except (ValueError, TypeError):
            capital_returned = 0.0

        outcome = FastPathOutcome(
            intent_id=intent.id,
            template_id=intent.template_id,
            execution_id=execution_id,
            success=result.success,
            error=result.error,
            total_latency_ms=total_ms,
            pattern_match_ms=pattern_match_ms,
            execution_ms=exec_ms,
            capital_deployed=capital_deployed,
            capital_returned=capital_returned,
            executor_type=intent.executor_type,
            execution_data=result.data,
        )

        # ── Post-execution bookkeeping ─────────────────────────────
        if result.success:
            self._successful_executions += 1
            self._capital_deployed += capital_deployed
            self._templates.record_success(intent.template_id, capital_deployed)
        else:
            self._failed_executions += 1
            self._templates.record_failure(intent.template_id)

        self._total_latency_ms += total_ms

        self._logger.info(
            "fast_path_execute_complete",
            execution_id=execution_id,
            template_id=intent.template_id,
            success=result.success,
            execution_path="fast_path",
            total_latency_ms=total_ms,
            pattern_match_ms=pattern_match_ms,
            execution_ms=exec_ms,
            capital_deployed=capital_deployed,
            error=result.error[:100] if result.error else "",
        )

        # Fire-and-forget audit logging
        try:
            await self._log_audit(intent, outcome, context)
        except Exception as exc:
            self._logger.warning("fast_path_audit_failed", error=str(exc))

        return outcome

    def _build_context(
        self,
        intent: FastPathIntent,
        execution_id: str,
    ) -> ExecutionContext:
        """
        Build a minimal ExecutionContext for the fast-path executor.

        Uses a synthetic Intent since the executor interface expects one,
        but the actual routing already happened outside the standard pipeline.
        """
        from primitives.common import Verdict
        from primitives.constitutional import ConstitutionalCheck
        from primitives.intent import (
            Action,
            ActionSequence,
            EthicalClearance,
            GoalDescriptor,
            Intent,
        )

        # Build a synthetic Intent for executor compatibility
        synthetic_intent = Intent(
            id=intent.id,
            goal=GoalDescriptor(
                description=f"Fast-path execution: template {intent.template_id}",
                target_domain="defi_arbitrage",
            ),
            plan=ActionSequence(
                steps=[
                    Action(
                        executor=intent.executor_type,
                        parameters=intent.execution_params,
                        timeout_ms=150,
                    )
                ]
            ),
            ethical_clearance=EthicalClearance(
                status=Verdict.APPROVED,
                reasoning=f"Pre-approved via constitutional template {intent.template_id}",
            ),
            autonomy_level_required=3,  # DeFi requires STEWARD
            autonomy_level_granted=3,
        )

        # Build a synthetic Equor check (template pre-approval)
        equor_check = ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.APPROVED,
            confidence=intent.approval_confidence,
            reasoning=f"Pre-approved constitutional template: {intent.template_id}",
        )

        return ExecutionContext(
            execution_id=execution_id,
            intent=synthetic_intent,
            equor_check=equor_check,
            credentials=ScopedCredentials(),
            instance_id=self._instance_id,
        )

    async def _log_audit(
        self,
        intent: FastPathIntent,
        outcome: FastPathOutcome,
        context: ExecutionContext,
    ) -> None:
        """Log the fast-path execution to the audit trail."""
        from systems.axon.types import AxonOutcome, StepOutcome

        # Convert to AxonOutcome for audit compatibility
        step_outcome = StepOutcome(
            step_index=0,
            action_type=intent.executor_type,
            description=f"Fast-path: {intent.template_id}",
            result=ExecutionResult(
                success=outcome.success,
                data=outcome.execution_data,
                error=outcome.error,
                side_effects=[
                    f"Fast-path execution via template {intent.template_id}, "
                    f"capital deployed: ${outcome.capital_deployed:.2f}"
                ],
            ),
            duration_ms=outcome.execution_ms,
        )

        axon_outcome = AxonOutcome(
            intent_id=intent.id,
            execution_id=outcome.execution_id,
            success=outcome.success,
            status=ExecutionStatus.SUCCESS if outcome.success else ExecutionStatus.FAILURE,
            step_outcomes=[step_outcome],
            duration_ms=outcome.total_latency_ms,
        )

        await self._audit.log(axon_outcome, context)

    def _fail(
        self,
        intent: FastPathIntent,
        execution_id: str,
        start: float,
        error: str,
    ) -> FastPathOutcome:
        """Create a fast failure outcome."""
        total_ms = int((time.monotonic() - start) * 1000)
        self._failed_executions += 1
        self._templates.record_failure(intent.template_id)

        self._logger.warning(
            "fast_path_gate_rejected",
            execution_id=execution_id,
            template_id=intent.template_id,
            error=error[:200],
            latency_ms=total_ms,
        )

        return FastPathOutcome(
            intent_id=intent.id,
            template_id=intent.template_id,
            execution_id=execution_id,
            success=False,
            error=error,
            total_latency_ms=total_ms,
            executor_type=intent.executor_type,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "success_rate": (
                self._successful_executions / max(1, self._total_executions)
            ),
            "capital_deployed": self._capital_deployed,
            "avg_latency_ms": (
                self._total_latency_ms / max(1, self._total_executions)
            ),
        }
