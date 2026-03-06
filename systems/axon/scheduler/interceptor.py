"""
EcodiaOS — Energy-Aware Scheduler Interceptor

Sits between Nova's intent approval and Axon's execution pipeline.
When a high-compute task is queued (REM/GRPO, model fine-tuning, bounty
solving, code generation), the interceptor checks the current grid carbon
intensity. If above a configurable threshold, the task is pushed to a
"deferred sleep cycle" queue and executed later when the grid is clean.

Architecture:
    Nova → Equor → **EnergyInterceptor** → Axon Pipeline
                         ↕
                   EnergyCache (Redis)
                         ↕
                   EnergyProvider (Electricity Maps | WattTime)

The interceptor is invoked by AxonService.execute() before the pipeline
runs. It does NOT replace the pipeline — it gates entry to it.

A background drain loop periodically checks the deferred queue and
re-submits tasks whose grid conditions have improved.

Iron Rules:
  - Interceptor decision is synchronous from the caller's perspective
    (cache lookup, threshold comparison — no LLM calls).
  - Deferred tasks are never silently dropped. They expire visibly after
    max_defer_seconds with an audit log entry.
  - The interceptor can be disabled at runtime via config toggle.
  - Financial and safety-critical tasks (wallet_transfer, request_funding)
    are NEVER deferred — only compute-heavy tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.scheduler.types import (
    ComputeIntensity,
    DeferralReason,
    DeferredTask,
    InterceptDecision,
)

if TYPE_CHECKING:
    from config import EnergyAwareSchedulerConfig
    from systems.axon.scheduler.energy_cache import EnergyCache
    from systems.axon.types import ExecutionRequest
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.axon.scheduler.interceptor")


# ─── Action Classification ────────────────────────────────────────

# Executor action_types that are compute-heavy (multi-turn LLM, GPU training).
# These are the only ones eligible for deferral.
_HIGH_COMPUTE_ACTIONS: frozenset[str] = frozenset({
    "axon.solve_bounty",      # Full Simula pipeline: clone → code agent → PR
    "trigger_consolidation",  # Memory consolidation (LLM-driven)
    "analyse",                # Multi-hop analysis with LLM
    "deploy_asset",           # Asset creation (generation + deployment)
})

# These are CRITICAL and must NEVER be deferred, regardless of grid state.
_NEVER_DEFER_ACTIONS: frozenset[str] = frozenset({
    "wallet_transfer",
    "request_funding",
    "defi_yield",
    "respond_text",
    "send_notification",
    "post_message",
    "set_reminder",
    "spawn_child",
    "collect_dividend",
    "request_telecom",
})

# Intent goal keywords that signal REM/GRPO/fine-tuning workloads.
# Checked against Intent.goal.description when executor-level classification
# is insufficient (e.g. a generic "call_api" step that actually triggers
# a training job).
_HIGH_COMPUTE_KEYWORDS: frozenset[str] = frozenset({
    "grpo",
    "fine-tune",
    "fine_tune",
    "finetune",
    "training",
    "rem_dream",
    "dream_generation",
    "model_training",
    "code_mutation",
    "evolution_proposal",
})


def classify_intent(request: ExecutionRequest) -> ComputeIntensity:
    """
    Classify an ExecutionRequest's compute intensity.

    Checks both the executor action_types in the plan and keywords in the
    goal description.
    """
    intent = request.intent

    # Check step executors
    for step in intent.plan.steps:
        if step.executor in _HIGH_COMPUTE_ACTIONS:
            return ComputeIntensity.HIGH

    # Check goal description for training/REM keywords
    goal_lower = intent.goal.description.lower()
    for keyword in _HIGH_COMPUTE_KEYWORDS:
        if keyword in goal_lower:
            return ComputeIntensity.CRITICAL

    # Check step parameters for high-compute hints
    for step in intent.plan.steps:
        params = step.parameters
        if params.get("compute_intensity") in ("high", "critical"):
            return ComputeIntensity.HIGH
        if params.get("requires_gpu", False):
            return ComputeIntensity.CRITICAL

    return ComputeIntensity.LOW


def is_never_defer(request: ExecutionRequest) -> bool:
    """True if ANY step in the intent uses a never-defer executor."""
    return any(step.executor in _NEVER_DEFER_ACTIONS for step in request.intent.plan.steps)


# ─── Interceptor ──────────────────────────────────────────────────


class EnergyAwareInterceptor:
    """
    Gates high-compute tasks based on grid carbon intensity.

    Usage:
        interceptor = EnergyAwareInterceptor(config, cache)
        decision = await interceptor.evaluate(request)
        if decision.should_defer:
            await interceptor.defer(request, decision)
        else:
            outcome = await axon_pipeline.execute(request)

    The interceptor also runs a background drain loop that periodically
    checks the deferred queue and re-submits tasks.
    """

    def __init__(
        self,
        config: EnergyAwareSchedulerConfig,
        cache: EnergyCache,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._event_bus = event_bus
        self._log = logger.bind(component="energy_interceptor")

        # Background drain task
        self._drain_task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Callback for re-submitting released tasks — wired by AxonService
        self._resubmit_callback: Any = None

        # Metrics
        self._total_evaluated: int = 0
        self._total_deferred: int = 0
        self._total_released: int = 0
        self._total_expired: int = 0

    # ─── Lifecycle ────────────────────────────────────────────────

    def set_resubmit_callback(self, callback: Any) -> None:
        """
        Set the callback used to re-submit released deferred tasks.

        Typically ``axon_service.execute``.
        """
        self._resubmit_callback = callback

    async def start(self) -> None:
        """Start the background drain loop."""
        if not self._config.enabled:
            self._log.info("interceptor_disabled")
            return
        if self._running:
            return
        self._running = True
        self._drain_task = asyncio.create_task(
            self._drain_loop(),
            name="axon.energy_interceptor.drain",
        )
        self._log.info(
            "interceptor_started",
            carbon_threshold_g=self._config.carbon_defer_threshold_g,
            drain_interval_s=self._config.drain_interval_s,
            max_defer_s=self._config.max_defer_seconds,
        )

    async def stop(self) -> None:
        """Cancel the background drain loop."""
        self._running = False
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
        self._drain_task = None
        self._log.info(
            "interceptor_stopped",
            total_evaluated=self._total_evaluated,
            total_deferred=self._total_deferred,
            total_released=self._total_released,
        )

    # ─── Evaluation ───────────────────────────────────────────────

    async def evaluate(self, request: ExecutionRequest) -> InterceptDecision:
        """
        Decide whether an ExecutionRequest should be deferred.

        Returns an InterceptDecision. The caller (AxonService) uses
        ``decision.should_defer`` to route the request.
        """
        self._total_evaluated += 1

        if not self._config.enabled:
            return InterceptDecision(should_defer=False, reason="interceptor_disabled")

        # Never defer safety-critical tasks
        if is_never_defer(request):
            return InterceptDecision(
                should_defer=False,
                reason="never_defer_action",
                compute_intensity=ComputeIntensity.LOW,
            )

        # Classify compute intensity
        intensity = classify_intent(request)
        if intensity in (ComputeIntensity.LOW, ComputeIntensity.MODERATE):
            return InterceptDecision(
                should_defer=False,
                reason="low_compute",
                compute_intensity=intensity,
            )

        # High/critical compute — check grid
        reading = await self._cache.get_reading()
        if reading is None:
            # No grid data available — execute anyway (fail-open)
            self._log.debug("no_grid_data_fail_open", intent_id=request.intent.id)
            return InterceptDecision(
                should_defer=False,
                reason="no_grid_data",
                compute_intensity=intensity,
            )

        threshold = self._config.carbon_defer_threshold_g
        current_g = reading.carbon_intensity_g

        if current_g >= threshold:
            self._log.info(
                "task_should_defer",
                intent_id=request.intent.id,
                carbon_g=round(current_g, 1),
                threshold_g=threshold,
                intensity=intensity,
                goal=request.intent.goal.description[:60],
            )
            return InterceptDecision(
                should_defer=True,
                reason=f"carbon_intensity {current_g:.0f}g >= threshold {threshold:.0f}g",
                compute_intensity=intensity,
                current_carbon_g=current_g,
                threshold_carbon_g=threshold,
            )

        return InterceptDecision(
            should_defer=False,
            reason="grid_ok",
            compute_intensity=intensity,
            current_carbon_g=current_g,
            threshold_carbon_g=threshold,
        )

    # ─── Deferral ─────────────────────────────────────────────────

    async def defer(
        self, request: ExecutionRequest, decision: InterceptDecision
    ) -> DeferredTask:
        """
        Park a high-compute task in the deferred sleep-cycle queue.

        Returns the DeferredTask record for audit/tracking.
        """
        task = DeferredTask(
            intent_id=request.intent.id,
            execution_request_json=request.model_dump(mode="json"),
            compute_intensity=decision.compute_intensity,
            deferral_reason=DeferralReason.CARBON_INTENSITY,
            carbon_intensity_at_deferral=decision.current_carbon_g or 0.0,
            max_defer_seconds=self._config.max_defer_seconds,
        )

        await self._cache.defer_task(task)
        self._total_deferred += 1

        # Emit event for observability
        await self._emit_deferred_event(task)

        return task

    # ─── Background Drain ─────────────────────────────────────────

    async def _drain_loop(self) -> None:
        """
        Background loop that periodically checks the deferred queue
        and re-submits tasks whose grid conditions have improved.
        """
        import time

        while self._running:
            loop_start = time.monotonic()
            try:
                await self._drain_once()
            except Exception as exc:
                self._log.warning("drain_error", error=str(exc))

            elapsed = time.monotonic() - loop_start
            sleep_s = max(0.0, self._config.drain_interval_s - elapsed)
            try:
                await asyncio.sleep(sleep_s)
            except asyncio.CancelledError:
                break

    async def _drain_once(self) -> None:
        """Single pass: release tasks that can now execute."""
        released, expired_count = await self._cache.release_ready_tasks(
            carbon_threshold_g=self._config.carbon_defer_threshold_g,
        )
        self._total_expired += expired_count
        if not released:
            return

        self._total_released += len(released)

        for task in released:
            self._log.info(
                "resubmitting_deferred_task",
                task_id=task.id,
                intent_id=task.intent_id,
                deferred_for_s=int(
                    (task.released_at - task.deferred_at).total_seconds()
                )
                if task.released_at
                else 0,
                release_carbon_g=task.release_carbon_intensity,
            )
            await self._resubmit(task)

    async def _resubmit(self, task: DeferredTask) -> None:
        """Re-submit a released deferred task to the Axon pipeline."""
        if self._resubmit_callback is None:
            self._log.warning(
                "no_resubmit_callback",
                task_id=task.id,
                hint="Wire via interceptor.set_resubmit_callback(axon.execute)",
            )
            return

        try:
            from systems.axon.types import ExecutionRequest
            request = ExecutionRequest(**task.execution_request_json)
            await self._resubmit_callback(request)
        except Exception as exc:
            self._log.error(
                "resubmit_failed",
                task_id=task.id,
                intent_id=task.intent_id,
                error=str(exc),
            )

    # ─── Events ───────────────────────────────────────────────────

    async def _emit_deferred_event(self, task: DeferredTask) -> None:
        """Emit a Synapse event when a task is deferred."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.TASK_ENERGY_DEFERRED,
                    data={
                        "task_id": task.id,
                        "intent_id": task.intent_id,
                        "compute_intensity": task.compute_intensity,
                        "carbon_intensity_g": task.carbon_intensity_at_deferral,
                        "max_defer_seconds": task.max_defer_seconds,
                    },
                    source_system="axon.energy_interceptor",
                )
            )
        except Exception as exc:
            # Event emission is non-fatal
            self._log.debug("event_emit_failed", error=str(exc))

    # ─── Diagnostics ──────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        """Return interceptor diagnostics."""
        cache_stats = await self._cache.stats()
        return {
            "enabled": self._config.enabled,
            "carbon_defer_threshold_g": self._config.carbon_defer_threshold_g,
            "drain_interval_s": self._config.drain_interval_s,
            "max_defer_seconds": self._config.max_defer_seconds,
            "total_evaluated": self._total_evaluated,
            "total_deferred": self._total_deferred,
            "total_released": self._total_released,
            "total_expired": self._total_expired,
            "drain_loop_running": self._running,
            "cache": cache_stats,
        }
