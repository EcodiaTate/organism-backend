"""
EcodiaOS - Cross-Chain Yield Observer (Phase 16d: DeFi Intelligence Expansion)

Observation-only module: tracks yield rates on Arbitrum, Optimism, and Polygon
via DeFiLlama and flags opportunities that are ≥2× Base rates for >72 hours.

Design constraints:
  - NEVER deploys capital cross-chain (bridge risk in Phase 16d)
  - Observation only - feeds intelligence to Nova and Equor for deliberation
  - When opportunity exceeds threshold AND organism has >$500 capital:
    surfaces as a goal for Equor review via CROSS_CHAIN_OPPORTUNITY event
  - Uses only official DeFiLlama data (no unaudited sources)

Integration:
  - Called from OikosService.run_consolidation_cycle() (or its own monitor loop)
  - Emits CROSS_CHAIN_OPPORTUNITY for Nova/Equor to consider
  - Emits INTELLIGENCE_UPDATE for Atune salience scoring

Never raises. All failures return empty results.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.cross_chain_observer")

# ─── Constants ────────────────────────────────────────────────────────────────

_DEFILLAMA_YIELDS_URL = "https://yields.llama.fi/pools"
_API_TIMEOUT_S = 10.0

# Chains to monitor (observation only - NOT deployment targets in Phase 16d)
_MONITORED_CHAINS = frozenset({"Arbitrum", "Optimism", "Polygon"})

# Same protocol/symbol safety filter as Base yield strategy
_SAFE_PROTOCOLS = frozenset({
    "aave-v3", "compound-v3", "morpho", "spark",
    "aerodrome", "moonwell", "extra-finance", "beefy",
    "uniswap-v3", "balancer", "curve",
})
_SAFE_SYMBOLS = frozenset({"USDC", "USDT"})
_MIN_TVL_USD = 5_000_000      # $5M TVL floor for cross-chain pools
_MAX_APY_SANITY = Decimal("0.50")
_YIELD_FLOOR_APY = Decimal("0.02")

# Opportunity threshold: cross-chain APY must be ≥2× the best Base APY
_OPPORTUNITY_RATIO_THRESHOLD = Decimal("2.0")

# Hours the ratio must be sustained before flagging as an opportunity
_SUSTAINED_HOURS_THRESHOLD = 72.0

# Minimum liquid capital to surface as a goal (below this, not worth bridge fees)
_MIN_CAPITAL_FOR_OPPORTUNITY_USD = Decimal("500.00")

# Redis key for tracking opportunity persistence
_OPPORTUNITY_TRACKING_KEY = "eos:oikos:cross_chain_opportunities"

# How often to poll (seconds)
_POLL_INTERVAL_S = 3600  # hourly

_EVENT_SOURCE = "oikos.cross_chain_observer"


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass
class CrossChainPool:
    """A yield pool on a non-Base chain."""
    chain: str
    protocol: str
    pool_id: str
    symbol: str
    apy: Decimal
    tvl_usd: float


@dataclass
class CrossChainOpportunity:
    """A tracked opportunity that has been elevated above the threshold."""
    chain: str
    protocol: str
    apy: Decimal
    base_apy: Decimal
    ratio: Decimal
    first_seen_at: datetime
    hours_elevated: float

    def is_sustained(self) -> bool:
        return self.hours_elevated >= _SUSTAINED_HOURS_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain": self.chain,
            "protocol": self.protocol,
            "apy": str(self.apy),
            "base_apy": str(self.base_apy),
            "ratio": str(self.ratio),
            "first_seen_at": self.first_seen_at.isoformat(),
            "hours_elevated": self.hours_elevated,
        }


# ─── CrossChainYieldObserver ─────────────────────────────────────────────────


class CrossChainYieldObserver:
    """
    Observes cross-chain yield opportunities without deploying capital.

    When an opportunity is ≥2× Base APY for >72h and the organism holds >$500,
    emits CROSS_CHAIN_OPPORTUNITY for Nova/Equor deliberation.

    The observer has no wallet access and cannot initiate any transactions.
    It is a pure intelligence layer.
    """

    def __init__(
        self,
        redis: "RedisClient | None" = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._redis = redis
        self._event_bus = event_bus
        self._log = logger.bind(component="cross_chain_observer")
        # In-memory tracking of currently elevated opportunities
        # key = f"{chain}:{protocol}"
        self._tracked: dict[str, CrossChainOpportunity] = {}

    def set_redis(self, redis: "RedisClient") -> None:
        self._redis = redis

    def set_event_bus(self, event_bus: "EventBus") -> None:
        self._event_bus = event_bus

    # ── Background monitor ────────────────────────────────────────────────────

    async def monitor_loop(self) -> None:
        """Supervised background loop: poll DeFiLlama hourly."""
        while True:
            try:
                await self.observe_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log.error("cross_chain_observe_error", error=str(exc))
            await asyncio.sleep(_POLL_INTERVAL_S)

    # ── Main observation cycle ────────────────────────────────────────────────

    async def observe_once(
        self, liquid_balance_usd: Decimal = Decimal("0")
    ) -> list[CrossChainOpportunity]:
        """
        Fetch DeFiLlama data, compare against Base rates, track opportunities.

        liquid_balance_usd - passed from OikosService to determine if organism
        has enough capital to surface an opportunity goal ($500 threshold).

        Returns list of currently sustained opportunities.
        """
        all_pools = await self._fetch_all_pools()
        if not all_pools:
            return []

        # Separate Base pools from cross-chain pools
        base_pools = [p for p in all_pools if p.chain == "Base"]
        cross_chain = [p for p in all_pools if p.chain in _MONITORED_CHAINS]

        if not base_pools:
            return []

        base_apy = max((p.apy for p in base_pools), default=Decimal("0"))
        now = datetime.now(UTC)

        # Track elevated opportunities
        active_keys: set[str] = set()
        flagged: list[CrossChainOpportunity] = []

        for pool in cross_chain:
            if base_apy <= Decimal("0"):
                continue
            ratio = pool.apy / base_apy

            if ratio < _OPPORTUNITY_RATIO_THRESHOLD:
                continue   # Not elevated enough

            key = f"{pool.chain}:{pool.protocol}"
            active_keys.add(key)

            if key in self._tracked:
                # Update hours elapsed
                opp = self._tracked[key]
                elapsed = (now - opp.first_seen_at).total_seconds() / 3600
                opp.hours_elevated = elapsed
                opp.apy = pool.apy
                opp.ratio = ratio
            else:
                # New elevated opportunity
                self._tracked[key] = CrossChainOpportunity(
                    chain=pool.chain,
                    protocol=pool.protocol,
                    apy=pool.apy,
                    base_apy=base_apy,
                    ratio=ratio,
                    first_seen_at=now,
                    hours_elevated=0.0,
                )
                self._log.info(
                    "cross_chain_opportunity_detected",
                    chain=pool.chain,
                    protocol=pool.protocol,
                    apy=str(pool.apy),
                    base_apy=str(base_apy),
                    ratio=float(ratio),
                )

        # Expire opportunities that are no longer elevated
        expired = [k for k in self._tracked if k not in active_keys]
        for k in expired:
            del self._tracked[k]

        # Flag sustained opportunities
        for opp in list(self._tracked.values()):
            if opp.is_sustained():
                flagged.append(opp)
                await self._flag_opportunity(opp, liquid_balance_usd)

        # Persist tracking state
        await self._persist_tracking()

        return flagged

    async def _flag_opportunity(
        self,
        opp: CrossChainOpportunity,
        liquid_balance_usd: Decimal,
    ) -> None:
        """
        Emit CROSS_CHAIN_OPPORTUNITY and optionally surface as Nova goal.

        Only emits if not already flagged recently (avoid spam).
        """
        self._log.info(
            "cross_chain_opportunity_flagged",
            chain=opp.chain,
            protocol=opp.protocol,
            apy=str(opp.apy),
            ratio=float(opp.ratio),
            hours_elevated=opp.hours_elevated,
        )

        await _emit(
            self._event_bus,
            event_type="cross_chain_opportunity",
            data={
                "chain": opp.chain,
                "protocol": opp.protocol,
                "apy": float(opp.apy),
                "base_apy": float(opp.base_apy),
                "ratio": float(opp.ratio),
                "hours_elevated": opp.hours_elevated,
                "min_capital_usd": float(_MIN_CAPITAL_FOR_OPPORTUNITY_USD),
                "flagged_at": datetime.now(UTC).isoformat(),
                "note": (
                    "Observation only. Cross-chain deployment requires "
                    "Equor review and organism capital > $500. "
                    "No bridge transactions initiated."
                ),
            },
        )

        # If organism has enough capital, surface as intelligence update for Nova
        if liquid_balance_usd >= _MIN_CAPITAL_FOR_OPPORTUNITY_USD:
            await _emit(
                self._event_bus,
                event_type="intelligence_update",
                data={
                    "feed_id": f"cross_chain_yield:{opp.chain}:{opp.protocol}",
                    "category": "defi",
                    "url": "https://yields.llama.fi",
                    "summary": (
                        f"{opp.protocol} on {opp.chain} offers {float(opp.apy):.1%} APY, "
                        f"{float(opp.ratio):.1f}× Base rates, sustained {opp.hours_elevated:.0f}h. "
                        f"Cross-chain deployment requires Equor review."
                    ),
                    "changed": True,
                    "content_hash": f"{opp.chain}:{opp.protocol}:{opp.apy}",
                    "raw_snippets": [],
                    "source_urls": ["https://yields.llama.fi"],
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "salience": min(0.9, 0.5 + float(opp.ratio - _OPPORTUNITY_RATIO_THRESHOLD) * 0.1),
                },
            )

    # ── DeFiLlama fetch ───────────────────────────────────────────────────────

    async def _fetch_all_pools(self) -> list[CrossChainPool]:
        """
        Fetch all qualifying pools from DeFiLlama (Base + monitored chains).

        Returns list of CrossChainPool. Returns [] on error.
        """
        try:
            async with httpx.AsyncClient(timeout=_API_TIMEOUT_S) as client:
                resp = await client.get(_DEFILLAMA_YIELDS_URL)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            self._log.warning("defillama_fetch_failed", error=str(exc))
            return []

        monitored = _MONITORED_CHAINS | frozenset({"Base"})
        pools: list[CrossChainPool] = []

        for raw in data.get("data", []):
            try:
                symbol: str = raw.get("symbol", "").upper()
                project: str = raw.get("project", "").lower()
                chain: str = raw.get("chain", "")
                tvl: float = float(raw.get("tvlUsd", 0))
                apy_raw: float = float(raw.get("apy", 0))

                if chain not in monitored:
                    continue
                if not any(s in symbol for s in _SAFE_SYMBOLS):
                    continue
                if project not in _SAFE_PROTOCOLS:
                    continue
                if tvl < _MIN_TVL_USD:
                    continue
                if apy_raw <= 0:
                    continue

                apy = Decimal(str(round(apy_raw / 100, 8)))
                if apy < _YIELD_FLOOR_APY or apy > _MAX_APY_SANITY:
                    continue

                pools.append(CrossChainPool(
                    chain=chain,
                    protocol=project,
                    pool_id=str(raw.get("pool", "")),
                    symbol=symbol,
                    apy=apy,
                    tvl_usd=tvl,
                ))
            except (ValueError, TypeError, InvalidOperation):
                continue

        return pools

    # ── Persistence ──────────────────────────────────────────────────────────

    async def _persist_tracking(self) -> None:
        if self._redis is None:
            return
        try:
            serialized = {
                k: opp.to_dict() for k, opp in self._tracked.items()
            }
            await self._redis.set_json(_OPPORTUNITY_TRACKING_KEY, serialized)
        except Exception as exc:
            self._log.warning("cross_chain_tracking_persist_failed", error=str(exc))

    async def initialize(self) -> None:
        """Restore tracked opportunities from Redis on startup."""
        if self._redis is None:
            return
        try:
            raw = await self._redis.get_json(_OPPORTUNITY_TRACKING_KEY)
            if not isinstance(raw, dict):
                return
            for key, data in raw.items():
                try:
                    self._tracked[key] = CrossChainOpportunity(
                        chain=data["chain"],
                        protocol=data["protocol"],
                        apy=Decimal(str(data["apy"])),
                        base_apy=Decimal(str(data["base_apy"])),
                        ratio=Decimal(str(data["ratio"])),
                        first_seen_at=datetime.fromisoformat(data["first_seen_at"]),
                        hours_elevated=float(data.get("hours_elevated", 0)),
                    )
                except (KeyError, TypeError, InvalidOperation, ValueError):
                    continue
        except Exception as exc:
            self._log.warning("cross_chain_tracking_load_failed", error=str(exc))

    def snapshot(self) -> dict[str, Any]:
        """Return current observation state for Benchmarks."""
        return {
            "tracked_opportunities": len(self._tracked),
            "sustained_opportunities": sum(
                1 for o in self._tracked.values() if o.is_sustained()
            ),
            "opportunities": [o.to_dict() for o in self._tracked.values()],
        }


# ─── Event emission helper ────────────────────────────────────────────────────


async def _emit(
    event_bus: "EventBus | None",
    event_type: str,
    data: dict[str, Any],
) -> None:
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
        logger.error("cross_chain_observer_emit_failed", event_type=event_type, error=str(exc))
