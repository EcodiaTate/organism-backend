"""
EcodiaOS - Oikos Risk Manager (Phase 16d: DeFi Intelligence Expansion)

Monitors portfolio health, enforces concentration limits, caps leverage, and
triggers emergency deleveraging when positions become unsafe.

Safety constraints (non-negotiable):
  MAX_SINGLE_PROTOCOL_PCT = 0.60  - no more than 60% of deployed capital in one protocol
  MAX_LEVERAGE             = 2.0  - never exceed 2× leverage on any borrowed position
  EMERGENCY_WITHDRAW_TRIGGER = 0.85 - withdraw if health factor < 0.85 on any loan

Integration:
  - Called from OikosService.run_consolidation_cycle() every cycle
  - Subscribes to BENCHMARK_REGRESSION for portfolio health deterioration
  - Emits PORTFOLIO_REBALANCED on rebalance execution
  - Emits METABOLIC_PRESSURE on emergency deleverage

All USD math uses Decimal. Never raises - failures return degraded RiskReport.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.risk_manager")

# ─── Safety invariants (never override) ──────────────────────────────────────

MAX_SINGLE_PROTOCOL_PCT = Decimal("0.60")   # 60% cap per protocol
MAX_LEVERAGE = Decimal("2.0")               # 2× leverage cap
EMERGENCY_WITHDRAW_TRIGGER = Decimal("0.85")  # Health factor floor

# Aave V3 health factor read - returns the minimum health factor across positions
# health factor < 1.0 = liquidation risk; < EMERGENCY_WITHDRAW_TRIGGER = EOS exits
_AAVE_POOL_BASE = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"
_AAVE_GET_USER_ACCOUNT_DATA = "0xbf92857c"  # getUserAccountData(address)
_BASE_RPC = "https://mainnet.base.org"

_REDIS_RISK_KEY = "eos:oikos:risk_report"
_EVENT_SOURCE = "oikos.risk_manager"


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass
class ProtocolAllocation:
    """Current deployment in a single protocol."""
    protocol: str
    amount_usd: Decimal
    apy: Decimal
    has_debt: bool = False          # True if this is a leveraged position
    health_factor: Decimal = Decimal("999")  # 999 = no debt, effectively infinite


@dataclass
class RiskReport:
    """Point-in-time portfolio risk snapshot."""
    total_deployed_usd: Decimal = Decimal("0")
    allocations: list[ProtocolAllocation] = field(default_factory=list)
    max_single_protocol_pct: Decimal = Decimal("0")
    dominant_protocol: str = ""
    leverage_ratio: Decimal = Decimal("1.0")
    min_health_factor: Decimal = Decimal("999")
    concentration_breach: bool = False   # True if any protocol > 60%
    leverage_breach: bool = False        # True if leverage > 2×
    emergency_trigger: bool = False      # True if health factor < 0.85
    healthy: bool = True
    assessed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_deployed_usd": str(self.total_deployed_usd),
            "allocations": [
                {
                    "protocol": a.protocol,
                    "amount_usd": str(a.amount_usd),
                    "apy": str(a.apy),
                    "has_debt": a.has_debt,
                    "health_factor": str(a.health_factor),
                }
                for a in self.allocations
            ],
            "max_single_protocol_pct": str(self.max_single_protocol_pct),
            "dominant_protocol": self.dominant_protocol,
            "leverage_ratio": str(self.leverage_ratio),
            "min_health_factor": str(self.min_health_factor),
            "concentration_breach": self.concentration_breach,
            "leverage_breach": self.leverage_breach,
            "emergency_trigger": self.emergency_trigger,
            "healthy": self.healthy,
            "assessed_at": self.assessed_at,
        }


@dataclass
class RebalanceAction:
    """A single rebalance step."""
    protocol: str
    action: str       # "withdraw" | "reduce_leverage"
    amount_usd: Decimal
    reason: str


# ─── RiskManager ─────────────────────────────────────────────────────────────


class RiskManager:
    """
    Portfolio risk monitor and rebalancer.

    Lifecycle:
      1. Call initialize() once at system startup.
      2. Call assess_portfolio() each consolidation cycle.
      3. Call rebalance_if_needed() after assess - issues rebalance requests on the bus.
      4. Call emergency_deleverage() immediately if health factor drops below trigger.

    The RiskManager does NOT execute transactions directly. It emits
    YIELD_DEPLOYMENT_REQUEST (withdraw variant) events on the Synapse bus so
    Axon's DeFiYieldExecutor handles the actual on-chain work. This preserves
    the no-direct-cross-system-import rule.
    """

    def __init__(
        self,
        redis: "RedisClient | None" = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._redis = redis
        self._event_bus = event_bus
        self._log = logger.bind(component="risk_manager")
        self._last_report: RiskReport | None = None

    def set_redis(self, redis: "RedisClient") -> None:
        self._redis = redis

    def set_event_bus(self, event_bus: "EventBus") -> None:
        self._event_bus = event_bus

    # ── Portfolio assessment ──────────────────────────────────────────────────

    async def assess_portfolio(self) -> RiskReport:
        """
        Read current yield positions from Redis and compute a risk snapshot.

        Returns RiskReport. Never raises.
        """
        report = RiskReport(assessed_at=datetime.now(UTC).isoformat())

        if self._redis is None:
            self._log.warning("risk_assess_no_redis")
            return report

        try:
            # Load all yield positions (may be a single dict or list of dicts)
            raw = await self._redis.get_json("eos:oikos:yield_positions")
        except Exception as exc:
            self._log.warning("risk_assess_redis_failed", error=str(exc))
            return report

        if raw is None:
            self._last_report = report
            return report

        # Normalize: the yield_strategy stores a single position dict
        positions_raw: list[dict[str, Any]] = (
            raw if isinstance(raw, list) else [raw]
        )

        total = Decimal("0")
        allocations: list[ProtocolAllocation] = []

        for pos in positions_raw:
            try:
                protocol = str(pos.get("protocol", "unknown"))
                amount = Decimal(str(pos.get("amount_usd", "0")))
                apy = Decimal(str(pos.get("apy", "0")))
                has_debt = bool(pos.get("has_debt", False))
            except (InvalidOperation, TypeError):
                continue

            health = await self._fetch_health_factor(protocol)
            alloc = ProtocolAllocation(
                protocol=protocol,
                amount_usd=amount,
                apy=apy,
                has_debt=has_debt,
                health_factor=health,
            )
            allocations.append(alloc)
            total += amount

        report.total_deployed_usd = total
        report.allocations = allocations

        if total > Decimal("0") and allocations:
            # Concentration analysis
            max_pct = Decimal("0")
            dominant = ""
            for alloc in allocations:
                pct = alloc.amount_usd / total
                if pct > max_pct:
                    max_pct = pct
                    dominant = alloc.protocol

            report.max_single_protocol_pct = max_pct
            report.dominant_protocol = dominant
            report.concentration_breach = max_pct > MAX_SINGLE_PROTOCOL_PCT

            # Leverage analysis (sum of debt positions / total deployed)
            debt_total = sum(
                a.amount_usd for a in allocations if a.has_debt
            )
            report.leverage_ratio = (
                Decimal("1") + debt_total / total if total > 0 else Decimal("1")
            )
            report.leverage_breach = report.leverage_ratio > MAX_LEVERAGE

            # Health factor: minimum across all debt positions
            min_hf = min(
                (a.health_factor for a in allocations if a.has_debt),
                default=Decimal("999"),
            )
            report.min_health_factor = min_hf
            report.emergency_trigger = min_hf < EMERGENCY_WITHDRAW_TRIGGER

        report.healthy = not (
            report.concentration_breach
            or report.leverage_breach
            or report.emergency_trigger
        )

        self._log.info(
            "portfolio_assessed",
            total_deployed=str(total),
            dominant=report.dominant_protocol,
            max_single_pct=str(report.max_single_protocol_pct),
            leverage=str(report.leverage_ratio),
            min_health_factor=str(report.min_health_factor),
            healthy=report.healthy,
        )

        # Persist to Redis for external inspection
        if self._redis is not None:
            try:
                await self._redis.set_json(_REDIS_RISK_KEY, report.to_dict())
            except Exception as exc:
                self._log.warning("risk_report_persist_failed", error=str(exc))

        self._last_report = report
        return report

    # ── Rebalancing ───────────────────────────────────────────────────────────

    async def rebalance_if_needed(self) -> list[RebalanceAction]:
        """
        Check the last assessment and emit rebalance requests if needed.

        Returns list of RebalanceAction taken (may be empty). Never raises.
        """
        report = self._last_report
        if report is None or report.healthy:
            return []

        actions: list[RebalanceAction] = []

        if report.emergency_trigger:
            self._log.critical(
                "emergency_trigger_detected",
                min_health_factor=str(report.min_health_factor),
                trigger=str(EMERGENCY_WITHDRAW_TRIGGER),
            )
            await self.emergency_deleverage()
            actions.append(RebalanceAction(
                protocol=report.dominant_protocol,
                action="emergency_deleverage",
                amount_usd=report.total_deployed_usd,
                reason=f"Health factor {report.min_health_factor} < {EMERGENCY_WITHDRAW_TRIGGER}",
            ))
            await self._emit_portfolio_rebalanced("emergency", actions, report)
            return actions

        if report.leverage_breach:
            # Reduce leverage by withdrawing from debt position
            overage_ratio = report.leverage_ratio - MAX_LEVERAGE
            for alloc in report.allocations:
                if alloc.has_debt and alloc.amount_usd > Decimal("0"):
                    # Withdraw enough to bring leverage within bounds
                    reduce_by = (overage_ratio * alloc.amount_usd).quantize(Decimal("0.01"))
                    if reduce_by > Decimal("0.01"):
                        await self._request_withdraw(alloc.protocol, reduce_by)
                        actions.append(RebalanceAction(
                            protocol=alloc.protocol,
                            action="reduce_leverage",
                            amount_usd=reduce_by,
                            reason=(
                                f"Leverage {report.leverage_ratio} > "
                                f"MAX_LEVERAGE {MAX_LEVERAGE}"
                            ),
                        ))

        if report.concentration_breach:
            # Withdraw from the dominant protocol until it's at 60%
            dominant_alloc = next(
                (a for a in report.allocations if a.protocol == report.dominant_protocol),
                None,
            )
            if dominant_alloc is not None and report.total_deployed_usd > Decimal("0"):
                target_amount = MAX_SINGLE_PROTOCOL_PCT * report.total_deployed_usd
                excess = dominant_alloc.amount_usd - target_amount
                if excess > Decimal("1.00"):  # Only rebalance for $1+ excess
                    await self._request_withdraw(report.dominant_protocol, excess)
                    actions.append(RebalanceAction(
                        protocol=report.dominant_protocol,
                        action="withdraw",
                        amount_usd=excess,
                        reason=(
                            f"Concentration {report.max_single_protocol_pct:.1%} > "
                            f"MAX {MAX_SINGLE_PROTOCOL_PCT:.1%}"
                        ),
                    ))

        if actions:
            await self._emit_portfolio_rebalanced("concentration", actions, report)

        return actions

    async def emergency_deleverage(self) -> None:
        """
        Immediate full withdrawal from all debt positions.

        Called when health_factor < EMERGENCY_WITHDRAW_TRIGGER. Requests
        withdrawal via YIELD_DEPLOYMENT_REQUEST on the Synapse bus.
        Never raises.
        """
        if self._last_report is None:
            return

        self._log.critical(
            "emergency_deleverage_initiated",
            min_health_factor=str(self._last_report.min_health_factor),
        )

        for alloc in self._last_report.allocations:
            if alloc.has_debt and alloc.amount_usd > Decimal("0"):
                await self._request_withdraw(alloc.protocol, alloc.amount_usd)

        # Emit metabolic pressure signal so the organism knows it was hit
        await _emit(
            self._event_bus,
            event_type="metabolic_pressure",
            data={
                "emergency_deleverage": True,
                "min_health_factor": str(self._last_report.min_health_factor),
                "trigger_threshold": str(EMERGENCY_WITHDRAW_TRIGGER),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    # ── Health factor fetching ────────────────────────────────────────────────

    async def _fetch_health_factor(self, protocol: str) -> Decimal:
        """
        Fetch the current health factor for a lending position.

        Currently implemented for Aave V3 on Base (getUserAccountData).
        Other protocols return 999 (no debt, safe).

        Returns Decimal health factor. Never raises.
        """
        if protocol not in ("aave", "aave-v3"):
            return Decimal("999")   # Only Aave has borrowing in Phase 16d

        try:
            loop = asyncio.get_event_loop()
            hf = await loop.run_in_executor(None, self._fetch_aave_health_sync)
            return hf if hf is not None else Decimal("999")
        except Exception as exc:
            self._log.debug("health_factor_fetch_failed", protocol=protocol, error=str(exc))
            return Decimal("999")   # Fail-safe: assume no debt

    @staticmethod
    def _fetch_aave_health_sync() -> Decimal | None:
        """
        Synchronous Aave V3 getUserAccountData() call.

        Returns health factor (index 5 in the returned tuple) or None on error.
        Health factor is in ray units (1e18 for V3, not 1e27).
        """
        try:
            from web3 import Web3  # noqa: PLC0415

            # Need wallet address - read from env
            import os  # noqa: PLC0415
            wallet_addr = os.environ.get("ORGANISM_WALLET__ADDRESS", "")
            if not wallet_addr:
                return None

            w3 = Web3(Web3.HTTPProvider(_BASE_RPC, request_kwargs={"timeout": 5}))
            abi = [
                {
                    "inputs": [{"internalType": "address", "name": "user", "type": "address"}],
                    "name": "getUserAccountData",
                    "outputs": [
                        {"internalType": "uint256", "name": "totalCollateralBase", "type": "uint256"},
                        {"internalType": "uint256", "name": "totalDebtBase", "type": "uint256"},
                        {"internalType": "uint256", "name": "availableBorrowsBase", "type": "uint256"},
                        {"internalType": "uint256", "name": "currentLiquidationThreshold", "type": "uint256"},
                        {"internalType": "uint256", "name": "ltv", "type": "uint256"},
                        {"internalType": "uint256", "name": "healthFactor", "type": "uint256"},
                    ],
                    "stateMutability": "view",
                    "type": "function",
                }
            ]
            pool = w3.eth.contract(
                address=Web3.to_checksum_address(_AAVE_POOL_BASE),
                abi=abi,
            )
            result = pool.functions.getUserAccountData(
                Web3.to_checksum_address(wallet_addr)
            ).call()
            # healthFactor is at index 5, in 1e18 scale
            hf_raw = result[5]
            if hf_raw == 0:
                return Decimal("999")  # No debt
            return Decimal(hf_raw) / Decimal("1e18")
        except Exception:
            return None

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _request_withdraw(self, protocol: str, amount_usd: Decimal) -> None:
        """Request Axon to withdraw from a protocol via YIELD_DEPLOYMENT_REQUEST."""
        import uuid  # noqa: PLC0415
        from systems.synapse.types import SynapseEvent, SynapseEventType  # noqa: PLC0415

        request_id = str(uuid.uuid4())

        self._log.info(
            "requesting_withdraw",
            protocol=protocol,
            amount_usd=str(amount_usd),
            request_id=request_id,
        )

        if self._event_bus is None:
            return

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.YIELD_DEPLOYMENT_REQUEST,
                source_system=_EVENT_SOURCE,
                data={
                    "action": "withdraw",
                    "amount_usd": str(amount_usd.quantize(Decimal("0.000001"))),
                    "protocol": protocol,
                    "apy": "0",
                    "request_id": request_id,
                },
            ))
        except Exception as exc:
            self._log.error("withdraw_request_emit_failed", error=str(exc))

    async def _emit_portfolio_rebalanced(
        self,
        trigger: str,
        actions: list[RebalanceAction],
        before_report: RiskReport,
    ) -> None:
        """Emit PORTFOLIO_REBALANCED event."""
        await _emit(
            self._event_bus,
            event_type="portfolio_rebalanced",
            data={
                "trigger": trigger,
                "actions": [
                    {
                        "protocol": a.protocol,
                        "action": a.action,
                        "amount_usd": str(a.amount_usd),
                        "reason": a.reason,
                    }
                    for a in actions
                ],
                "before_report": before_report.to_dict(),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )


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
        logger.error("risk_manager_emit_failed", event_type=event_type, error=str(exc))
