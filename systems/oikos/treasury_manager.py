"""
EcodiaOS - Oikos Treasury Manager (Phase 16d: DeFi Intelligence Expansion)

Manages the organism's liquid_balance according to target allocation buckets:

  40% yield bucket        - actively deployed in DeFi (income-generating)
  30% survival reserve    - NEVER deployed, sacrosanct emergency buffer
  20% working capital     - bounty compute costs, substrate fees, active operations
  10% opportunity fund    - new opportunities, account provisioning, exploratory deploys

When ratios drift beyond ±5% tolerance, auto-rebalances by deploying to or
withdrawing from the yield bucket. The survival reserve is hard-enforced and
cannot be touched even during rebalance.

Integration:
  - Called from OikosService.run_consolidation_cycle()
  - Reads liquid_balance from EconomicState snapshot
  - Emits YIELD_DEPLOYMENT_REQUEST (deposit/withdraw) via Synapse bus
  - Emits TREASURY_REBALANCED on bucket rebalance
  - Never raises - all failures log and return

All USD math uses Decimal.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.treasury_manager")

# ─── Target bucket ratios ────────────────────────────────────────────────────

YIELD_TARGET = Decimal("0.40")          # 40% deployed for income
SURVIVAL_TARGET = Decimal("0.30")       # 30% survival reserve (never deployed)
WORKING_CAPITAL_TARGET = Decimal("0.20")  # 20% bounty / compute costs
OPPORTUNITY_TARGET = Decimal("0.10")    # 10% new opportunities

# Rebalance tolerance: only act if a bucket drifts beyond ±5%
REBALANCE_TOLERANCE = Decimal("0.05")

# Minimum deploy/withdraw amount to make a rebalance gas-efficient
MIN_REBALANCE_AMOUNT_USD = Decimal("20.00")

_EVENT_SOURCE = "oikos.treasury_manager"


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass
class TreasuryState:
    """Current bucket allocation snapshot."""
    total_balance_usd: Decimal
    deployed_yield_usd: Decimal       # Currently in DeFi protocols
    survival_reserve_usd: Decimal     # Sacrosanct buffer
    working_capital_usd: Decimal      # Liquid for operations
    opportunity_usd: Decimal          # Liquid for new opportunities

    # Derived ratios
    @property
    def yield_ratio(self) -> Decimal:
        if self.total_balance_usd <= 0:
            return Decimal("0")
        return self.deployed_yield_usd / self.total_balance_usd

    @property
    def survival_ratio(self) -> Decimal:
        if self.total_balance_usd <= 0:
            return Decimal("0")
        return self.survival_reserve_usd / self.total_balance_usd

    @property
    def liquid_ratio(self) -> Decimal:
        """Fraction of balance that is liquid (not in DeFi)."""
        if self.total_balance_usd <= 0:
            return Decimal("1")
        liquid = (
            self.survival_reserve_usd
            + self.working_capital_usd
            + self.opportunity_usd
        )
        return liquid / self.total_balance_usd

    def in_balance(self) -> bool:
        """True if all buckets are within tolerance of targets."""
        return (
            abs(self.yield_ratio - YIELD_TARGET) <= REBALANCE_TOLERANCE
            and abs(self.survival_ratio - SURVIVAL_TARGET) <= REBALANCE_TOLERANCE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_balance_usd": str(self.total_balance_usd),
            "deployed_yield_usd": str(self.deployed_yield_usd),
            "survival_reserve_usd": str(self.survival_reserve_usd),
            "working_capital_usd": str(self.working_capital_usd),
            "opportunity_usd": str(self.opportunity_usd),
            "yield_ratio": str(self.yield_ratio),
            "survival_ratio": str(self.survival_ratio),
        }


# ─── TreasuryManager ─────────────────────────────────────────────────────────


class TreasuryManager:
    """
    Rebalances the organism's liquid capital across four purpose-specific buckets.

    The manager does NOT execute transactions directly. When it determines that
    more capital should be deployed (or withdrawn), it emits
    YIELD_DEPLOYMENT_REQUEST on the Synapse bus so Axon handles the on-chain op.

    Usage from consolidation cycle:
        state = treasury_mgr.compute_state(
            liquid_balance=econ_state.liquid_balance,
            deployed_yield=sum(p.principal_usd for p in econ_state.yield_positions),
            survival_reserve=econ_state.survival_reserve,
        )
        await treasury_mgr.rebalance_if_drifted(state)
    """

    def __init__(self, event_bus: "EventBus | None" = None) -> None:
        self._event_bus = event_bus
        self._log = logger.bind(component="treasury_manager")

    def set_event_bus(self, event_bus: "EventBus") -> None:
        self._event_bus = event_bus

    # ── State computation ─────────────────────────────────────────────────────

    def compute_state(
        self,
        liquid_balance: Decimal,
        deployed_yield: Decimal,
        survival_reserve: Decimal,
    ) -> TreasuryState:
        """
        Compute the current treasury bucket state from known values.

        liquid_balance   - total USDC in wallet (not including DeFi positions)
        deployed_yield   - USDC currently earning yield in DeFi
        survival_reserve - the configured sacrosanct buffer

        The liquid balance is split proportionally into survival, working, and
        opportunity buckets. The actual split is inferred from targets because
        there are no hard sub-wallet separations in the current architecture.
        """
        total = liquid_balance + deployed_yield

        # Survival reserve is hard-configured - it always has its full amount
        survival = survival_reserve

        # Remainder after survival is the operationally available balance
        available_liquid = max(Decimal("0"), liquid_balance - survival_reserve)

        # Split available liquid into working capital and opportunity buckets
        # at their relative target ratios (20/10 = 2:1 split)
        working_share = WORKING_CAPITAL_TARGET / (WORKING_CAPITAL_TARGET + OPPORTUNITY_TARGET)
        working_capital = (available_liquid * working_share).quantize(Decimal("0.01"))
        opportunity = available_liquid - working_capital

        state = TreasuryState(
            total_balance_usd=total,
            deployed_yield_usd=deployed_yield,
            survival_reserve_usd=survival,
            working_capital_usd=working_capital,
            opportunity_usd=opportunity,
        )

        self._log.debug(
            "treasury_state_computed",
            total_usd=str(total),
            yield_deployed=str(deployed_yield),
            yield_ratio=str(state.yield_ratio),
            survival_reserve=str(survival),
            working_capital=str(working_capital),
            opportunity=str(opportunity),
        )

        return state

    # ── Rebalancing ───────────────────────────────────────────────────────────

    async def rebalance_if_drifted(self, state: TreasuryState) -> bool:
        """
        If yield bucket is out of tolerance, emit a deploy or withdraw request.

        Returns True if a rebalance action was taken.
        """
        if state.in_balance():
            return False

        total = state.total_balance_usd
        if total <= Decimal("0"):
            return False

        target_yield_usd = (total * YIELD_TARGET).quantize(Decimal("0.01"))
        current_yield_usd = state.deployed_yield_usd
        delta = target_yield_usd - current_yield_usd

        before_ratios = state.to_dict()
        action_taken = False

        if delta > MIN_REBALANCE_AMOUNT_USD:
            # Under-deployed - deploy more
            # But never touch survival reserve: cap deploy at available liquid
            max_deployable = max(
                Decimal("0"),
                state.working_capital_usd + state.opportunity_usd - MIN_REBALANCE_AMOUNT_USD,
            )
            deploy_amount = min(delta, max_deployable)
            if deploy_amount >= MIN_REBALANCE_AMOUNT_USD:
                await self._request_deploy(deploy_amount)
                action_taken = True
                self._log.info(
                    "treasury_deploying",
                    target_yield=str(target_yield_usd),
                    current_yield=str(current_yield_usd),
                    deploy_amount=str(deploy_amount),
                )

        elif delta < -MIN_REBALANCE_AMOUNT_USD:
            # Over-deployed - withdraw excess
            withdraw_amount = abs(delta)
            if withdraw_amount >= MIN_REBALANCE_AMOUNT_USD:
                await self._request_withdraw(withdraw_amount)
                action_taken = True
                self._log.info(
                    "treasury_withdrawing",
                    target_yield=str(target_yield_usd),
                    current_yield=str(current_yield_usd),
                    withdraw_amount=str(withdraw_amount),
                )

        if action_taken:
            trigger = "rebalance_under_deployed" if delta > 0 else "rebalance_over_deployed"
            await _emit(
                self._event_bus,
                event_type="treasury_rebalanced",
                data={
                    "trigger": trigger,
                    "before_ratios": before_ratios,
                    "after_ratios": {
                        "target_yield": str(YIELD_TARGET),
                        "target_survival": str(SURVIVAL_TARGET),
                        "target_working": str(WORKING_CAPITAL_TARGET),
                        "target_opportunity": str(OPPORTUNITY_TARGET),
                    },
                    "deploy_amount_usd": str(delta) if delta > 0 else None,
                    "withdraw_amount_usd": str(abs(delta)) if delta < 0 else None,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        return action_taken

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _request_deploy(self, amount_usd: Decimal) -> None:
        """Emit YIELD_DEPLOYMENT_REQUEST (deposit) for treasury rebalance."""
        from systems.synapse.types import SynapseEvent, SynapseEventType  # noqa: PLC0415

        if self._event_bus is None:
            return

        request_id = str(uuid.uuid4())
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.YIELD_DEPLOYMENT_REQUEST,
                source_system=_EVENT_SOURCE,
                data={
                    "action": "deposit",
                    "amount_usd": str(amount_usd.quantize(Decimal("0.000001"))),
                    "protocol": "",   # yield_strategy will select best pool
                    "apy": "0",
                    "request_id": request_id,
                    "trigger": "treasury_rebalance",
                },
            ))
        except Exception as exc:
            self._log.error("deploy_request_emit_failed", error=str(exc))

    async def _request_withdraw(self, amount_usd: Decimal) -> None:
        """Emit YIELD_DEPLOYMENT_REQUEST (withdraw) for treasury rebalance."""
        from systems.synapse.types import SynapseEvent, SynapseEventType  # noqa: PLC0415

        if self._event_bus is None:
            return

        request_id = str(uuid.uuid4())
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.YIELD_DEPLOYMENT_REQUEST,
                source_system=_EVENT_SOURCE,
                data={
                    "action": "withdraw",
                    "amount_usd": str(amount_usd.quantize(Decimal("0.000001"))),
                    "protocol": "",   # executor selects active position protocol
                    "apy": "0",
                    "request_id": request_id,
                    "trigger": "treasury_rebalance",
                },
            ))
        except Exception as exc:
            self._log.error("withdraw_request_emit_failed", error=str(exc))


# ─── Event emission helper ────────────────────────────────────────────────────


async def _emit(
    event_bus: "EventBus | None",
    event_type: str,
    data: dict[str, Any],
) -> None:
    """Emit a SynapseEvent. Fails silently on error."""
    if event_bus is None:
        return
    try:
        from systems.synapse.types import SynapseEvent, SynapseEventType  # noqa: PLC0415

        await event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType(event_type),
                source_system=_EVENT_SOURCE,
                data=data,
            )
        )
    except Exception as exc:
        logger.error("treasury_manager_emit_failed", event_type=event_type, error=str(exc))
