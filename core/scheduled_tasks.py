"""
EcodiaOS - Scheduled Perception Tasks

Register periodic tasks (PR monitoring, DeFi yield deployment,
yield accrual, foraging, consolidation) with the PerceptionScheduler.

All intervals come from config.schedules — the organism can evolve them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.scheduler import PerceptionScheduler
    from config import AppConfig
    from systems.axon.service import AxonService
    from systems.oikos.service import OikosService

logger = structlog.get_logger()


def register_scheduled_tasks(
    scheduler: PerceptionScheduler,
    axon: AxonService,
    oikos: OikosService,
    cfg: "AppConfig",
    event_bus: "Any | None" = None,
) -> None:
    """Register all periodic tasks with the scheduler.

    Intervals are read from cfg.schedules so Evo can tune them
    without touching Python code. 0 = task disabled.
    """
    s = cfg.schedules

    if s.monitor_prs_interval_s > 0:
        _register_monitor_prs(scheduler, axon, s.monitor_prs_interval_s)

    if s.defi_yield_deployment_interval_s > 0:
        _register_defi_yield_deployment(scheduler, oikos, event_bus, s.defi_yield_deployment_interval_s)

    if s.defi_yield_accrual_interval_s > 0:
        _register_defi_yield_accrual(scheduler, oikos, event_bus, s.defi_yield_accrual_interval_s)

    if s.bounty_foraging_interval_s > 0:
        _register_foraging_cycle(scheduler, oikos, event_bus, s.bounty_foraging_interval_s)

    if s.economic_consolidation_interval_s > 0:
        _register_consolidation_cycle(scheduler, oikos, event_bus, s.economic_consolidation_interval_s)


def _register_monitor_prs(
    scheduler: PerceptionScheduler,
    axon: AxonService,
    interval_s: float,
) -> None:
    from systems.fovea.types import InputChannel

    async def _poll_monitor_prs() -> str | None:
        # Route through the event bus so Equor deliberates before Axon acts.
        # No pre-approved verdict — the organism decides at execution time.
        try:
            from primitives.common import AutonomyLevel
            from primitives.intent import Action, ActionSequence, GoalDescriptor, Intent
            from systems.synapse.types import SynapseEvent, SynapseEventType

            intent = Intent(
                goal=GoalDescriptor(
                    description="Monitor open GitHub PRs for merge or rejection",
                    target_domain="github",
                ),
                plan=ActionSequence(
                    steps=[Action(executor="axon.monitor_prs", timeout_ms=60_000)]
                ),
                autonomy_level_required=AutonomyLevel.STEWARD,
                autonomy_level_granted=AutonomyLevel.STEWARD,
            )
            event_bus = getattr(axon, "_event_bus", None)
            if event_bus is not None:
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.AXON_EXECUTION_REQUEST,
                    source_system="scheduled_tasks",
                    data={"intent": intent.model_dump(), "source": "monitor_prs_schedule"},
                ))
            else:
                # No bus — execute directly; Axon's own pipeline still validates.
                from systems.axon.types import ExecutionRequest
                await axon.execute(ExecutionRequest(intent=intent, timeout_ms=60_000))
        except Exception as exc:
            logger.warning("monitor_prs_schedule_failed", error=str(exc))
        return None

    scheduler.register(
        name="monitor_prs",
        interval_seconds=interval_s,
        channel=InputChannel.SYSTEM_EVENT,
        fn=_poll_monitor_prs,
        metadata={"task_type": "maintenance", "executor": "axon.monitor_prs"},
    )


def _register_defi_yield_deployment(
    scheduler: PerceptionScheduler,
    oikos: OikosService,
    event_bus: Any | None,
    interval_s: float,
) -> None:
    from systems.fovea.types import InputChannel

    async def _deploy_idle_capital() -> str | None:
        outcome = await oikos.deploy_idle_capital()
        if outcome.success:
            return (
                f"Deployed ${outcome.amount_deployed_usd:.2f} USDC into "
                f"{outcome.protocol} (APY {float(outcome.apy):.2%}). "
                f"Expected daily yield: ${outcome.expected_daily_yield_usd:.4f}. "
                f"tx: {outcome.tx_hash}"
            )
        reason = getattr(outcome, "error", None) or getattr(outcome, "reason", "unknown")
        if event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await event_bus.emit(SynapseEvent(
                    type=SynapseEventType.ECONOMIC_ACTION_FAILED,
                    source_system="scheduled_tasks",
                    data={"task": "defi_yield_deployment", "reason": str(reason)},
                ))
            except Exception:
                pass
        return f"DeFi yield deployment failed: {reason}"

    scheduler.register(
        name="defi_yield_deployment",
        interval_seconds=interval_s,
        channel=InputChannel.SYSTEM_EVENT,
        fn=_deploy_idle_capital,
        metadata={"task_type": "economic", "phase": "16c"},
    )


def _register_defi_yield_accrual(
    scheduler: PerceptionScheduler,
    oikos: OikosService,
    event_bus: Any | None,
    interval_s: float,
) -> None:
    from systems.fovea.types import InputChannel

    async def _record_accrued_yield() -> str | None:
        accrued = await oikos.record_accrued_yield()
        if accrued > 0:
            return (
                f"Yield accrued: ${accrued:.6f} USDC recorded as revenue "
                f"from active DeFi position."
            )
        return "DeFi yield accrual: $0.00 recorded. Position may be inactive or pending."

    scheduler.register(
        name="defi_yield_accrual",
        interval_seconds=interval_s,
        channel=InputChannel.SYSTEM_EVENT,
        fn=_record_accrued_yield,
        metadata={"task_type": "economic", "phase": "16c"},
    )


def _register_foraging_cycle(
    scheduler: PerceptionScheduler,
    oikos: OikosService,
    event_bus: Any | None,
    interval_s: float,
) -> None:
    from systems.fovea.types import InputChannel

    async def _run_foraging() -> str | None:
        result = await oikos.run_foraging_cycle()
        accepted = result.get("accepted", 0)
        scanned = result.get("candidates_scanned", 0)
        if scanned > 0:
            return (
                f"Foraging cycle: scanned {scanned} candidates, "
                f"accepted {accepted} bounties."
            )
        return "Foraging cycle: no candidates found this round."

    scheduler.register(
        name="bounty_foraging",
        interval_seconds=interval_s,
        channel=InputChannel.SYSTEM_EVENT,
        fn=_run_foraging,
        metadata={"task_type": "economic", "phase": "16b"},
    )


def _register_consolidation_cycle(
    scheduler: PerceptionScheduler,
    oikos: OikosService,
    event_bus: Any | None,
    interval_s: float,
) -> None:
    from systems.fovea.types import InputChannel

    async def _run_consolidation() -> str | None:
        result = await oikos.run_consolidation_cycle()
        foraging = result.get("foraging", {})
        accepted = foraging.get("accepted", 0) if isinstance(foraging, dict) else 0
        fleet_alive = result.get("fleet", {}).get("alive", 0) if isinstance(result.get("fleet"), dict) else 0
        assets = result.get("assets", {})
        return (
            f"Consolidation: foraging={accepted} bounties, "
            f"assets={assets}, "
            f"fleet alive={fleet_alive}."
        )

    scheduler.register(
        name="economic_consolidation",
        interval_seconds=interval_s,
        channel=InputChannel.SYSTEM_EVENT,
        fn=_run_consolidation,
        metadata={"task_type": "economic", "phase": "16_consolidation"},
    )
