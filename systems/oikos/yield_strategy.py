"""
EcodiaOS - Oikos Yield Strategy (Phase 16c: Resting Metabolism - Live Deployment)

Owns the full lifecycle of EOS's DeFi yield position:

  1. deploy_idle_capital()
       Reads live USDC balance from WalletClient, queries DeFiLlama for the
       best qualifying pool on Base, subtracts the survival reserve, then
       drives DeFiYieldExecutor to supply into Aave or Morpho.

  2. YieldPositionTracker (supervised loop, 1h interval)
       Persists the active position to Redis (eos:oikos:yield_positions).
       Every hour: re-queries DeFiLlama, checks whether APY has dropped >50%
       relative to the entry rate, and emits YIELD_REBALANCE_NEEDED if so.

  3. record_accrued_yield() (called from daily accounting loop)
       Calculates accrued = principal × apy / 365.
       Emits REVENUE_INJECTED on the Synapse bus - OikosService picks it up
       and credits revenue_24h / revenue_7d / liquid_balance.

Safety constraints:
  - EOS_SURVIVAL_RESERVE_USD (default $2.00) is never deployable.
  - Minimum deployable: $20.00 (gas-efficiency floor from DeFiYieldExecutor).
  - If WalletClient is None: log critical, emit ECONOMIC_CAPABILITY_MISSING,
    return degraded outcome - never fake success.
  - All USD math uses Decimal.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from clients.wallet import WalletClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.yield_strategy")

# ─── Constants ────────────────────────────────────────────────────────────────

_YIELD_POSITIONS_KEY = "eos:oikos:yield_positions"
_METABOLISM_KEY = "eos:oikos:metabolism"

_DEFILLAMA_YIELDS_URL = "https://yields.llama.fi/pools"
_YIELD_API_TIMEOUT_S = 10.0

# Minimum balance that must always remain in the wallet (untouchable).
_DEFAULT_SURVIVAL_RESERVE_USD = Decimal("2.00")

# Gas-efficiency floor - deploying less earns less than the tx costs.
_MIN_DEPLOYABLE_USD = Decimal("20.00")

# Minimum APY we will accept (2%).
_YIELD_FLOOR_APY = Decimal("0.02")

# APY drop threshold that triggers rebalance alert (50% relative drop).
_APY_DROP_REBALANCE_THRESHOLD = Decimal("0.50")

# Hourly health-check interval in seconds.
_HEALTH_CHECK_INTERVAL_S = 3600

# DeFiLlama pool filter config.
# Phase 16d additions: Aerodrome (AMM), Moonwell (lending), Extra Finance (leveraged),
# Beefy (auto-compounding vault aggregator)
_SAFE_PROTOCOLS = frozenset({
    "aave-v3", "compound-v3", "morpho", "spark",
    "aerodrome",        # Base: concentrated liquidity AMM, earns AERO rewards
    "moonwell",         # Base: lending/borrowing, earns WELL rewards
    "extra-finance",    # Base: leveraged yield farming (conservative mode)
    "beefy",            # Multi-chain: auto-compounding vaults
})
_SAFE_CHAINS = frozenset({"Base"})          # Base only - where EOS funds live
_SAFE_SYMBOLS = frozenset({"USDC", "USDT"})
_MIN_TVL_USD = 10_000_000
_MAX_APY_SANITY = Decimal("0.50")           # 50% APY cap - anything above is sus

# Reinvestment threshold - accrued yield must exceed this before redeployment.
# Below this the gas cost eats the compound benefit.
_REINVESTMENT_THRESHOLD_USD = Decimal("5.00")

# Map DeFiLlama project names to DeFiYieldExecutor protocol keys
_PROTOCOL_EXECUTOR_MAP: dict[str, str] = {
    "aave-v3": "aave",
    "morpho": "morpho",
    "aerodrome": "aerodrome",
    "moonwell": "moonwell",
    "extra-finance": "extra_finance",
    "beefy": "beefy",
    "compound-v3": "aave",   # Compound maps to aave-compatible path (ERC-4626)
    "spark": "morpho",       # Spark maps to morpho-compatible path
}

_EVENT_SOURCE = "oikos.yield_strategy"

# ─── YIELD_DEPLOYMENT_RESULT router ──────────────────────────────────────────
#
# Previously, each deploy_idle_capital() call subscribed a new one-shot closure
# to YIELD_DEPLOYMENT_RESULT via EventBus.subscribe(). Because EventBus has no
# unsubscribe() method, every call left a dead reference in the subscriber list.
# Over many deployment cycles this accumulated unboundedly.
#
# Fix: use a module-level dict of pending Futures keyed by request_id plus a
# single persistent router subscriber that is registered only once (on the first
# call) and routes incoming results to the correct Future. Dead Futures are
# cleaned up automatically once they resolve.

_pending_yield_futures: dict[str, "asyncio.Future[dict[str, Any]]"] = {}
_yield_result_subscriber_registered: bool = False


async def _yield_result_router(event: Any) -> None:
    """Single persistent router for YIELD_DEPLOYMENT_RESULT events."""
    request_id = event.data.get("request_id", "")
    fut = _pending_yield_futures.get(request_id)
    if fut is not None and not fut.done():
        fut.set_result(event.data)
        # Clean up the resolved future immediately
        _pending_yield_futures.pop(request_id, None)


def _ensure_yield_result_subscriber(event_bus: Any) -> None:
    """Register the persistent router subscriber exactly once per bus."""
    global _yield_result_subscriber_registered  # noqa: PLW0603
    if _yield_result_subscriber_registered:
        return
    from systems.synapse.types import SynapseEventType

    event_bus.subscribe(SynapseEventType.YIELD_DEPLOYMENT_RESULT, _yield_result_router)
    _yield_result_subscriber_registered = True


# ─── Outcome types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DeploymentOutcome:
    """Result of a capital deployment attempt."""

    success: bool
    tx_hash: str = ""
    amount_deployed_usd: Decimal = Decimal("0")
    expected_daily_yield_usd: Decimal = Decimal("0")
    protocol: str = ""
    apy: Decimal = Decimal("0")
    error: str = ""
    degraded: bool = False          # True when we returned a non-fatal failure


# ─── DeFiLlama helpers (reuse metabolism_api selection logic, Base-only) ─────


async def _fetch_best_base_pool() -> tuple[str, Decimal] | None:
    """
    Query DeFiLlama for the highest-APY safe USDC/USDT pool on Base.

    Returns (protocol_name, apy_fraction) or None if no qualifying pool found.
    Protocol name maps to DeFiYieldExecutor's supported protocols:
      "aave-v3" → "aave"
      "morpho"  → "morpho"
    """
    try:
        async with httpx.AsyncClient(timeout=_YIELD_API_TIMEOUT_S) as client:
            resp = await client.get(_DEFILLAMA_YIELDS_URL)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("defillama_fetch_failed", error=str(exc))
        return None

    pools: list[dict[str, Any]] = data.get("data", [])
    best_apy: Decimal | None = None
    best_protocol: str | None = None

    for pool in pools:
        try:
            symbol: str = pool.get("symbol", "").upper()
            project: str = pool.get("project", "").lower()
            chain: str = pool.get("chain", "")
            tvl: float = float(pool.get("tvlUsd", 0))
            apy_raw: float = float(pool.get("apy", 0))

            if not any(s in symbol for s in _SAFE_SYMBOLS):
                continue
            if project not in _SAFE_PROTOCOLS:
                continue
            if chain not in _SAFE_CHAINS:
                continue
            if tvl < _MIN_TVL_USD:
                continue
            if apy_raw <= 0:
                continue

            apy = Decimal(str(round(apy_raw / 100, 8)))
            if apy < _YIELD_FLOOR_APY or apy > _MAX_APY_SANITY:
                continue

            if best_apy is None or apy > best_apy:
                best_apy = apy
                best_protocol = project

        except (ValueError, TypeError, InvalidOperation):
            continue

    if best_protocol is None or best_apy is None:
        return None

    # Map DeFiLlama project name to DeFiYieldExecutor protocol key
    executor_protocol = _PROTOCOL_EXECUTOR_MAP.get(best_protocol, "aave")
    return executor_protocol, best_apy


def _survival_reserve() -> Decimal:
    """Read EOS_SURVIVAL_RESERVE_USD from env, default $2.00."""
    raw = os.environ.get("EOS_SURVIVAL_RESERVE_USD", "")
    try:
        return Decimal(raw) if raw else _DEFAULT_SURVIVAL_RESERVE_USD
    except InvalidOperation:
        return _DEFAULT_SURVIVAL_RESERVE_USD


# ─── Capital Deployment ───────────────────────────────────────────────────────


async def deploy_idle_capital(
    wallet: WalletClient | None,
    event_bus: EventBus | None,
) -> DeploymentOutcome:
    """
    Read balance, pick best pool, deploy surplus above survival reserve.

    Returns a DeploymentOutcome. Never raises - failures return degraded=True.
    On missing credentials: emits ECONOMIC_CAPABILITY_MISSING and returns early.
    """
    log = logger.bind(op="deploy_idle_capital")

    # ── Guard: wallet must be available ───────────────────────────────────────
    if wallet is None:
        log.critical(
            "cdp_wallet_missing",
            message="WalletClient not injected - cannot deploy capital",
            event="ECONOMIC_CAPABILITY_MISSING",
        )
        await _emit(
            event_bus,
            event_type="metabolic_pressure",
            data={
                "economic_capability_missing": True,
                "capability": "defi_yield_deployment",
                "reason": "WalletClient not configured - set CDP credentials",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        return DeploymentOutcome(
            success=False,
            error="WalletClient not configured",
            degraded=True,
        )

    # ── Step 1: Get current USDC balance ─────────────────────────────────────
    try:
        usdc_balance = await wallet.get_usdc_balance()
    except Exception as exc:
        log.error("balance_fetch_failed", error=str(exc))
        return DeploymentOutcome(
            success=False,
            error=f"Balance fetch failed: {exc}",
            degraded=True,
        )

    reserve = _survival_reserve()
    deployable = usdc_balance - reserve

    log.info(
        "balance_checked",
        usdc_balance=str(usdc_balance),
        survival_reserve=str(reserve),
        deployable=str(deployable),
    )

    if deployable < _MIN_DEPLOYABLE_USD:
        msg = (
            f"Deployable balance ${deployable:.2f} is below gas-efficiency floor "
            f"${_MIN_DEPLOYABLE_USD:.2f} (balance=${usdc_balance:.2f}, "
            f"reserve=${reserve:.2f})"
        )
        log.info("below_deployable_floor", deployable=str(deployable))
        return DeploymentOutcome(
            success=False,
            error=msg,
            degraded=True,
        )

    # ── Step 2: Find best pool on Base ───────────────────────────────────────
    pool_result = await _fetch_best_base_pool()
    if pool_result is None:
        log.warning("no_qualifying_pool_found")
        return DeploymentOutcome(
            success=False,
            error="No qualifying pool found on Base via DeFiLlama",
            degraded=True,
        )

    protocol, apy = pool_result
    daily_yield = (deployable * apy / Decimal("365")).quantize(Decimal("0.000001"))

    log.info(
        "best_pool_selected",
        protocol=protocol,
        apy=str(apy),
        amount=str(deployable),
        expected_daily_yield_usd=str(daily_yield),
    )

    # ── Step 3: Request Axon execution via Synapse bus ──────────────────────
    # No direct cross-system import - all motor actions go through Synapse.
    if event_bus is None:
        log.error("no_event_bus_for_yield_deployment")
        return DeploymentOutcome(success=False, error="No event bus", degraded=True)

    import uuid

    from systems.synapse.types import SynapseEvent, SynapseEventType

    request_id = str(uuid.uuid4())
    result_future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

    # Register the shared router subscriber exactly once and route this
    # request via the module-level _pending_yield_futures dict.
    # Fixes the per-call subscriber leak: previously every deploy call added a
    # new closure to the EventBus subscriber list with no cleanup path.
    _ensure_yield_result_subscriber(event_bus)
    _pending_yield_futures[request_id] = result_future

    await event_bus.emit(SynapseEvent(
        event_type=SynapseEventType.YIELD_DEPLOYMENT_REQUEST,
        source_system="oikos",
        data={
            "action": "deposit",
            "amount_usd": str(deployable.quantize(Decimal("0.000001"))),
            "protocol": protocol,
            "apy": str(apy),
            "request_id": request_id,
        },
    ))

    # Wait up to 30s for Axon to respond
    try:
        result_data = await asyncio.wait_for(result_future, timeout=30.0)
    except asyncio.TimeoutError:
        # Clean up the timed-out future so the router doesn't see a stale entry.
        _pending_yield_futures.pop(request_id, None)
        log.error("yield_deployment_timeout", request_id=request_id)
        return DeploymentOutcome(
            success=False, error="Axon deployment timeout", degraded=True
        )

    if not result_data.get("success", False):
        error_msg = result_data.get("error", "Unknown executor error")
        log.error("executor_failed", error=error_msg)
        return DeploymentOutcome(success=False, error=error_msg, degraded=True)

    tx_hash: str = result_data.get("tx_hash", "")
    log.info(
        "capital_deployed",
        tx_hash=tx_hash,
        protocol=protocol,
        amount_usd=str(deployable),
        apy=str(apy),
        daily_yield_usd=str(daily_yield),
    )

    return DeploymentOutcome(
        success=True,
        tx_hash=tx_hash,
        amount_deployed_usd=deployable,
        expected_daily_yield_usd=daily_yield,
        protocol=protocol,
        apy=apy,
    )


# ─── Position Tracking ────────────────────────────────────────────────────────


class YieldPositionTracker:
    """
    Tracks the active yield position in Redis and monitors health hourly.

    Lifecycle:
      - On deployment success: call record_position().
      - Background: run start_monitoring() as a supervised_task.
      - On APY drop >50%: emit YIELD_REBALANCE_NEEDED.
      - Daily: call record_accrued_yield() to inject revenue on the bus.
    """

    def __init__(
        self,
        redis: RedisClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._redis = redis
        self._event_bus = event_bus
        self._log = logger.bind(component="yield_position_tracker")

    # ── Position persistence ──────────────────────────────────────────────────

    async def record_position(self, outcome: DeploymentOutcome) -> None:
        """Persist a new position to Redis after successful deployment."""
        if not outcome.success:
            return

        position = {
            "protocol": outcome.protocol,
            "amount_usd": str(outcome.amount_deployed_usd),
            "apy": str(outcome.apy),
            "deployed_at": datetime.now(UTC).isoformat(),
            "expected_daily_usd": str(outcome.expected_daily_yield_usd),
            "tx_hash": outcome.tx_hash,
            "entry_apy": str(outcome.apy),   # stored separately for health checks
        }

        if self._redis is not None:
            try:
                await self._redis.set_json(_YIELD_POSITIONS_KEY, position)
                self._log.info(
                    "yield_position_recorded",
                    protocol=outcome.protocol,
                    amount_usd=str(outcome.amount_deployed_usd),
                    apy=str(outcome.apy),
                )
            except Exception as exc:
                self._log.warning("yield_position_persist_failed", error=str(exc))
        else:
            self._log.warning(
                "yield_position_no_redis",
                position=position,
                hint="Redis unavailable - position not persisted",
            )

    async def load_position(self) -> dict[str, Any] | None:
        """Load the current position from Redis. Returns None if none exists."""
        if self._redis is None:
            return None
        try:
            return await self._redis.get_json(_YIELD_POSITIONS_KEY)
        except Exception as exc:
            self._log.warning("yield_position_load_failed", error=str(exc))
            return None

    async def clear_position(self) -> None:
        """Remove the position record (e.g. after withdrawal)."""
        if self._redis is None:
            return
        try:
            await self._redis.client.delete(_YIELD_POSITIONS_KEY)  # type: ignore[attr-defined]
        except Exception as exc:
            self._log.warning("yield_position_clear_failed", error=str(exc))

    # ── Hourly health monitor (supervised_task loop) ──────────────────────────

    async def monitor_loop(self) -> None:
        """
        Supervised loop: check position health every hour.

        Designed to be passed to supervised_task() - has an internal while True.
        Exits only on CancelledError (graceful shutdown).
        """
        while True:
            try:
                await self._check_health_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log.error("health_check_error", error=str(exc))
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL_S)

    async def _check_health_once(self) -> None:
        """Single health check iteration."""
        position = await self.load_position()
        if position is None:
            self._log.debug("health_check_no_position")
            return

        protocol = position.get("protocol", "")
        entry_apy_raw = position.get("entry_apy", position.get("apy", "0"))
        try:
            entry_apy = Decimal(str(entry_apy_raw))
        except InvalidOperation:
            entry_apy = Decimal("0")

        # Fetch current APY for the protocol
        pool_result = await _fetch_best_base_pool()
        if pool_result is None:
            self._log.warning("health_check_api_unavailable")
            return

        current_protocol, current_apy = pool_result

        self._log.info(
            "yield_health_check",
            protocol=protocol,
            entry_apy=str(entry_apy),
            current_apy=str(current_apy),
        )

        # Rebalance if APY dropped >50% relative to entry
        rebalance_needed = False
        relative_drop = Decimal("0")
        if entry_apy > Decimal("0"):
            relative_drop = (entry_apy - current_apy) / entry_apy
            if relative_drop > _APY_DROP_REBALANCE_THRESHOLD:
                rebalance_needed = True
                self._log.warning(
                    "apy_drop_threshold_exceeded",
                    entry_apy=str(entry_apy),
                    current_apy=str(current_apy),
                    relative_drop=str(relative_drop),
                )
                await _emit(
                    self._event_bus,
                    event_type="metabolic_pressure",
                    data={
                        "yield_rebalance_needed": True,
                        "protocol": protocol,
                        "entry_apy": str(entry_apy),
                        "current_apy": str(current_apy),
                        "relative_drop": str(relative_drop),
                        "threshold": str(_APY_DROP_REBALANCE_THRESHOLD),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

        # Always emit YIELD_PERFORMANCE_REPORT so Evo and Simula can observe
        # yield health without coupling to internal yield_strategy state.
        from systems.synapse.types import SynapseEventType as _SET  # noqa: PLC0415

        await _emit(
            self._event_bus,
            event_type=_SET.YIELD_PERFORMANCE_REPORT.value,
            data={
                "protocol": protocol,
                "current_apy": str(current_apy),
                "entry_apy": str(entry_apy),
                "relative_drop_pct": str(round(relative_drop * 100, 4)),
                "rebalance_needed": rebalance_needed,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    # ── Daily yield accrual ───────────────────────────────────────────────────

    async def record_accrued_yield(self) -> Decimal:
        """
        Calculate and record one day of accrued yield as Oikos revenue.

        accrued = principal × apy / 365

        Emits REVENUE_INJECTED on the Synapse bus - OikosService credits
        revenue_24h, revenue_7d, and liquid_balance automatically.

        Returns the accrued amount (Decimal), or 0 if no position exists.
        """
        position = await self.load_position()
        if position is None:
            self._log.debug("accrue_no_position")
            return Decimal("0")

        try:
            principal = Decimal(str(position.get("amount_usd", "0")))
            apy = Decimal(str(position.get("apy", "0")))
        except InvalidOperation:
            self._log.error("accrue_invalid_position_data", position=position)
            return Decimal("0")

        if principal <= Decimal("0") or apy <= Decimal("0"):
            return Decimal("0")

        accrued = (principal * apy / Decimal("365")).quantize(Decimal("0.000001"))

        self._log.info(
            "yield_accrued",
            principal_usd=str(principal),
            apy=str(apy),
            accrued_usd=str(accrued),
        )

        # Inject as revenue on Synapse bus
        await _emit(
            self._event_bus,
            event_type="revenue_injected",
            data={
                "amount_usd": str(accrued),
                "source": "yield",
                "protocol": position.get("protocol", ""),
                "principal_usd": str(principal),
                "apy": str(apy),
                "accrued_at": datetime.now(UTC).isoformat(),
            },
        )

        # Also update the position's running metadata in Redis
        if self._redis is not None and accrued > Decimal("0"):
            try:
                position["last_accrual_usd"] = str(accrued)
                position["last_accrual_at"] = datetime.now(UTC).isoformat()
                await self._redis.set_json(_YIELD_POSITIONS_KEY, position)
            except Exception as exc:
                self._log.warning("accrue_update_failed", error=str(exc))

        return accrued


# ─── Yield Reinvestment Engine ────────────────────────────────────────────────


class YieldReinvestmentEngine:
    """
    Compound growth loop: when accrued yield exceeds the reinvestment threshold,
    redeploy it back into the best available pool.

    Threshold: $5.00 - below this gas cost erodes the compounding benefit.

    Called from:
      - YieldPositionTracker.record_accrued_yield() after each daily accrual
      - OikosService.run_consolidation_cycle() as a safety net

    Never raises - all failures return False with a log entry.
    """

    def __init__(
        self,
        tracker: "YieldPositionTracker",
        wallet: "WalletClient | None" = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._tracker = tracker
        self._wallet = wallet
        self._event_bus = event_bus
        self._log = logger.bind(component="yield_reinvestment_engine")

    def set_wallet(self, wallet: "WalletClient") -> None:
        self._wallet = wallet

    def set_event_bus(self, event_bus: "EventBus") -> None:
        self._event_bus = event_bus

    async def check_and_reinvest(self) -> bool:
        """
        Check whether enough yield has accrued to reinvest.

        Returns True if reinvestment was triggered, False otherwise.
        """
        position = await self._tracker.load_position()
        if position is None:
            return False

        # Read how much yield has accrued since last reinvestment
        try:
            last_accrual = Decimal(str(position.get("last_accrual_usd", "0")))
        except InvalidOperation:
            return False

        if last_accrual < _REINVESTMENT_THRESHOLD_USD:
            self._log.debug(
                "reinvestment_below_threshold",
                accrued_usd=str(last_accrual),
                threshold=str(_REINVESTMENT_THRESHOLD_USD),
            )
            return False

        self._log.info(
            "reinvestment_threshold_met",
            accrued_usd=str(last_accrual),
            threshold=str(_REINVESTMENT_THRESHOLD_USD),
        )

        # Deploy accrued yield back into the best available pool
        outcome = await deploy_idle_capital(
            wallet=self._wallet,
            event_bus=self._event_bus,
        )

        if not outcome.success:
            self._log.warning("reinvestment_deploy_failed", error=outcome.error)
            return False

        # Emit YIELD_REINVESTED
        from datetime import UTC, datetime as _dt  # noqa: PLC0415

        await _emit(
            self._event_bus,
            event_type="yield_reinvested",
            data={
                "amount_usd": str(last_accrual),
                "protocol": outcome.protocol,
                "apy": str(outcome.apy),
                "pool_id": outcome.protocol,
                "accrued_since": position.get("last_accrual_at", ""),
                "timestamp": _dt.now(UTC).isoformat(),
            },
        )

        # Reset accrual counter in the position record
        if self._tracker._redis is not None:  # noqa: SLF001
            try:
                position["last_accrual_usd"] = "0"
                position["last_reinvested_at"] = _dt.now(UTC).isoformat()
                await self._tracker._redis.set_json(_YIELD_POSITIONS_KEY, position)  # noqa: SLF001
            except Exception as exc:
                self._log.warning("reinvestment_position_update_failed", error=str(exc))

        self._log.info(
            "yield_reinvested",
            amount_usd=str(last_accrual),
            protocol=outcome.protocol,
            apy=str(outcome.apy),
        )
        return True


# ─── Event emission helper ────────────────────────────────────────────────────


async def _emit(
    event_bus: EventBus | None,
    event_type: str,
    data: dict[str, Any],
) -> None:
    """Emit a SynapseEvent if a bus is available. Fails silently on error."""
    if event_bus is None:
        logger.warning("yield_strategy_no_event_bus", event_type=event_type, data=data)
        return
    try:
        from systems.synapse.types import SynapseEvent, SynapseEventType

        await event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType(event_type),
                source_system=_EVENT_SOURCE,
                data=data,
            )
        )
    except Exception as exc:
        logger.error("yield_strategy_emit_failed", event_type=event_type, error=str(exc))
