"""
EcodiaOS - Scheduled Perception Tasks

Register periodic tasks (PR monitoring, DeFi yield deployment,
yield accrual) with the PerceptionScheduler.

Extracted from main.py lifespan().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from clients.scheduler import PerceptionScheduler
    from systems.axon.service import AxonService
    from systems.oikos.service import OikosService

logger = structlog.get_logger()


def register_scheduled_tasks(
    scheduler: PerceptionScheduler,
    axon: AxonService,
    oikos: OikosService,
) -> None:
    """Register all periodic tasks with the scheduler."""

    # ── Monitor open PRs for merge/rejection (every 6h) ──────
    _register_monitor_prs(scheduler, axon)

    # ── Deploy idle USDC into DeFi yield (every 6h) ──────────
    _register_defi_yield_deployment(scheduler, oikos)

    # ── Record accrued DeFi yield as revenue (daily) ─────────
    _register_defi_yield_accrual(scheduler, oikos)

    # ── Bounty foraging cycle (every 2h) ─────────────────────
    _register_foraging_cycle(scheduler, oikos)

    # ── Economic consolidation (every 15m) ───────────────────
    _register_consolidation_cycle(scheduler, oikos)


def _register_monitor_prs(
    scheduler: PerceptionScheduler,
    axon: AxonService,
) -> None:
    from primitives.common import AutonomyLevel, Verdict
    from primitives.constitutional import ConstitutionalCheck
    from primitives.intent import Action, ActionSequence, GoalDescriptor, Intent
    from systems.axon.types import ExecutionRequest
    from systems.fovea.types import InputChannel

    async def _poll_monitor_prs() -> str | None:
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
        check = ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.APPROVED,
            reasoning="Scheduled read-only maintenance task; pre-approved.",
        )
        await axon.execute(
            ExecutionRequest(intent=intent, equor_check=check, timeout_ms=60_000)
        )
        return None

    scheduler.register(
        name="monitor_prs",
        interval_seconds=21600,
        channel=InputChannel.SYSTEM_EVENT,
        fn=_poll_monitor_prs,
        metadata={"task_type": "maintenance", "executor": "axon.monitor_prs"},
    )


def _register_defi_yield_deployment(
    scheduler: PerceptionScheduler,
    oikos: OikosService,
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
        return None

    scheduler.register(
        name="defi_yield_deployment",
        interval_seconds=21600,
        channel=InputChannel.SYSTEM_EVENT,
        fn=_deploy_idle_capital,
        metadata={"task_type": "economic", "phase": "16c"},
    )


def _register_defi_yield_accrual(
    scheduler: PerceptionScheduler,
    oikos: OikosService,
) -> None:
    from systems.fovea.types import InputChannel

    async def _record_accrued_yield() -> str | None:
        accrued = await oikos.record_accrued_yield()
        if accrued > 0:
            return (
                f"Yield accrued: ${accrued:.6f} USDC recorded as revenue "
                f"from active DeFi position."
            )
        return None

    scheduler.register(
        name="defi_yield_accrual",
        interval_seconds=86400,
        channel=InputChannel.SYSTEM_EVENT,
        fn=_record_accrued_yield,
        metadata={"task_type": "economic", "phase": "16c"},
    )


def _register_foraging_cycle(
    scheduler: PerceptionScheduler,
    oikos: OikosService,
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
        return None

    scheduler.register(
        name="bounty_foraging",
        interval_seconds=7200,  # Every 2 hours
        channel=InputChannel.SYSTEM_EVENT,
        fn=_run_foraging,
        metadata={"task_type": "economic", "phase": "16b"},
    )


def _register_consolidation_cycle(
    scheduler: PerceptionScheduler,
    oikos: OikosService,
) -> None:
    from systems.fovea.types import InputChannel

    async def _run_consolidation() -> str | None:
        result = await oikos.run_consolidation_cycle()
        foraging = result.get("foraging", {})
        accepted = foraging.get("accepted", 0) if isinstance(foraging, dict) else 0
        return (
            f"Consolidation: foraging={accepted} bounties, "
            f"assets={result.get('assets', {})}, "
            f"fleet alive={result.get('fleet', {}).get('alive', 0)}."
        )

    scheduler.register(
        name="economic_consolidation",
        interval_seconds=900,  # Every 15 minutes
        channel=InputChannel.SYSTEM_EVENT,
        fn=_run_consolidation,
        metadata={"task_type": "economic", "phase": "16_consolidation"},
    )
