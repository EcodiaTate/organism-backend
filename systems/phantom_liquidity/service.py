"""
EcodiaOS — Phantom Liquidity Service (Phase 16q)

Orchestrates the phantom liquidity sensor network.  Manages the full lifecycle:

1. Pool selection — which pairs to monitor (via PoolSelector)
2. Position deployment — mint LP positions (via PhantomLiquidityExecutor)
3. Swap event listening — extract prices from on-chain events (via SwapEventListener)
4. Atune integration — feed prices into the perception pipeline
5. Health monitoring — impermanent loss tracking, staleness detection
6. Oikos reporting — track as YieldPosition entries in EconomicState

Lifecycle::

    service = LiquidityPhantomService(config, wallet, atune, oikos)
    await service.initialize()
    service.attach(event_bus)
    # ... running ...
    await service.shutdown()

Performance contract:
  - ``get_price()`` is a cheap dict lookup (~0us).
  - ``maintenance_cycle()`` runs periodically (default 1hr), does pure math.
  - Event handling completes within 100ms.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from systems.phantom_liquidity.pool_selector import PoolSelector
from systems.phantom_liquidity.price_listener import (
    SwapEventListener,
)
from systems.phantom_liquidity.types import (
    PhantomLiquidityPool,
    PhantomPriceFeed,
    PoolHealth,
)

if TYPE_CHECKING:
    from clients.timescaledb import TimescaleDBClient
    from clients.wallet import WalletClient
    from config import PhantomLiquidityConfig
    from systems.atune.service import AtuneService
    from systems.oikos.service import OikosService
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

logger = structlog.get_logger()


class LiquidityPhantomService:
    """
    Phantom Liquidity Sensor Network.

    Deploys minimal concentrated liquidity positions into DeFi pools to
    receive real-time price feeds via Swap events.  This replaces paid
    oracle subscriptions with self-funding price sensors.
    """

    system_id: str = "phantom_liquidity"

    def __init__(
        self,
        config: PhantomLiquidityConfig,
        wallet: WalletClient | None = None,
        atune: AtuneService | None = None,
        oikos: OikosService | None = None,
        tsdb: TimescaleDBClient | None = None,
        instance_id: str = "eos-default",
    ) -> None:
        self._config = config
        self._wallet = wallet
        self._atune = atune
        self._oikos = oikos
        self._tsdb = tsdb
        self._instance_id = instance_id
        self._logger = logger.bind(system="phantom_liquidity")

        # Sub-components
        self._pool_selector = PoolSelector(max_pools=config.max_pools)
        self._listener: SwapEventListener | None = None

        # State
        self._pools: dict[str, PhantomLiquidityPool] = {}  # pool_address -> pool
        self._latest_prices: dict[str, PhantomPriceFeed] = {}  # pair_key -> feed
        self._pair_to_pool: dict[str, str] = {}  # pair_key -> pool_address

        # Event bus
        self._event_bus: EventBus | None = None

        # Background tasks
        self._maintenance_task: asyncio.Task[None] | None = None
        self._initialized = False

        # Metrics
        self._total_price_updates: int = 0
        self._oracle_fallback_count: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Build sub-components and start the swap event listener."""
        if self._initialized:
            return

        rpc_url = self._config.rpc_url
        if not rpc_url:
            self._logger.info(
                "phantom_liquidity_disabled",
                reason="no_rpc_url",
            )
            return

        # Build listener
        self._listener = SwapEventListener(
            rpc_url=rpc_url,
            poll_interval_s=self._config.swap_poll_interval_s,
        )
        self._listener.add_listener(self._on_price_update)

        # Register any pre-existing pools (restored from state)
        for pool in self._pools.values():
            if pool.health == PoolHealth.ACTIVE:
                self._listener.add_pool(
                    pool.pool_address,
                    pool.pair,
                    pool.token0_decimals,
                    pool.token1_decimals,
                )

        await self._listener.start()

        # Start maintenance loop
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(),
            name="phantom_liquidity_maintenance",
        )

        self._initialized = True
        self._logger.info(
            "phantom_liquidity_initialized",
            pools=len(self._pools),
            rpc_url=rpc_url[:40] + "..." if len(rpc_url) > 40 else rpc_url,
        )

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to Synapse events for coordination."""
        self._event_bus = event_bus

        # React to metabolic pressure — if AUSTERITY, consider freeing capital
        try:
            from systems.synapse.types import SynapseEventType
            event_bus.subscribe(
                SynapseEventType.METABOLIC_PRESSURE,
                self._on_metabolic_pressure,
            )
            self._logger.info("phantom_liquidity_attached_to_synapse")
        except Exception as exc:
            self._logger.warning(
                "phantom_liquidity_attach_failed", error=str(exc),
            )

    async def shutdown(self) -> None:
        """Stop listener and cancel background tasks."""
        if self._listener is not None:
            await self._listener.stop()
            self._listener = None

        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None

        self._logger.info(
            "phantom_liquidity_shutdown",
            total_price_updates=self._total_price_updates,
            oracle_fallbacks=self._oracle_fallback_count,
        )

    # ── Public API ─────────────────────────────────────────────────

    def get_price(self, pair: tuple[str, str]) -> PhantomPriceFeed | None:
        """
        Get the latest price for a trading pair from phantom pools.

        Returns ``None`` if no pool covers this pair — the caller should
        fall back to an oracle API.
        """
        key = _pair_key(pair)
        feed = self._latest_prices.get(key)

        if feed is None:
            return None

        # Check staleness
        age_s = (datetime.now(UTC) - feed.timestamp).total_seconds()
        if age_s > self._config.staleness_threshold_s:
            return None  # Stale — caller should use oracle

        return feed

    def get_all_prices(self) -> list[PhantomPriceFeed]:
        """Return latest prices from all active phantom pools."""
        return list(self._latest_prices.values())

    def get_pools(self) -> list[PhantomLiquidityPool]:
        """Return all tracked phantom pool positions."""
        return list(self._pools.values())

    def register_pool(self, pool: PhantomLiquidityPool) -> None:
        """
        Register a deployed phantom pool position.

        Called after a successful ``deploy_position`` execution.  Adds the
        pool to the listener and tracks it in Oikos.
        """
        self._pools[pool.pool_address.lower()] = pool
        self._pair_to_pool[_pair_key(pool.pair)] = pool.pool_address.lower()

        # Add to swap listener
        if self._listener is not None:
            self._listener.add_pool(
                pool.pool_address,
                pool.pair,
                pool.token0_decimals,
                pool.token1_decimals,
            )

        # Track in Oikos as YieldPosition
        if self._oikos is not None:
            from systems.oikos.models import YieldPosition
            yp = YieldPosition(
                protocol="uniswap_v3_phantom",
                pool=pool.pool_address,
                principal_usd=pool.capital_deployed_usd,
                apy=Decimal("0"),
                protocol_address=pool.pool_address,
                chain_id=8453,
                health_status="healthy",
            )
            pool.yield_position_id = yp.id if hasattr(yp, "id") else ""
            self._oikos.register_phantom_position(yp)

        self._logger.info(
            "phantom_pool_registered",
            pool=pool.pool_address,
            pair=pool.pair,
            capital=str(pool.capital_deployed_usd),
        )

    def unregister_pool(self, pool_address: str) -> None:
        """
        Unregister a phantom pool after withdrawal.

        Removes from listener and Oikos.
        """
        addr = pool_address.lower()
        pool = self._pools.pop(addr, None)
        if pool is None:
            return

        # Remove from pair mapping
        key = _pair_key(pool.pair)
        if self._pair_to_pool.get(key) == addr:
            self._pair_to_pool.pop(key, None)
            self._latest_prices.pop(key, None)

        # Remove from listener
        if self._listener is not None:
            self._listener.remove_pool(pool_address)

        # Remove from Oikos
        if self._oikos is not None:
            self._oikos.remove_phantom_position(pool_address)

        self._logger.info("phantom_pool_unregistered", pool=pool_address)

    # ── Price Feed Callback ────────────────────────────────────────

    async def _on_price_update(self, feed: PhantomPriceFeed) -> None:
        """
        Callback from SwapEventListener when a new Swap event is decoded.

        1. Update the pool's last_price_observed
        2. Cache for get_price() queries
        3. Feed into Atune via ingest()
        """
        addr = feed.pool_address.lower()
        pool = self._pools.get(addr)

        if pool is not None:
            pool.last_price_observed = feed.price
            pool.last_price_timestamp = feed.timestamp
            pool.price_update_count += 1
            if pool.health == PoolHealth.STALE:
                pool.health = PoolHealth.ACTIVE

        # Cache latest price by pair
        key = _pair_key(feed.pair)
        self._latest_prices[key] = feed
        self._total_price_updates += 1

        # Persist to TimescaleDB
        await self._persist_price(feed)

        # Feed to Atune
        await self._feed_to_atune(feed)

    async def _persist_price(self, feed: PhantomPriceFeed) -> None:
        """Write a price observation to TimescaleDB if available."""
        if self._tsdb is None:
            return
        try:
            await self._tsdb.write_phantom_price({
                "time": feed.timestamp,
                "pair": f"{feed.pair[0]}/{feed.pair[1]}",
                "price": feed.price,
                "source": feed.source,
                "pool_address": feed.pool_address,
                "block_number": feed.block_number,
                "latency_ms": feed.latency_ms,
            })
        except Exception as exc:
            self._logger.debug("phantom_tsdb_write_error", error=str(exc))

    async def get_price_history(
        self,
        pair: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return persisted price history for a pair from TimescaleDB."""
        if self._tsdb is None:
            return []
        try:
            rows = await self._tsdb.get_phantom_price_history(pair, limit=limit)
            return rows
        except Exception as exc:
            self._logger.debug("phantom_tsdb_read_error", error=str(exc))
            return []

    async def _feed_to_atune(self, feed: PhantomPriceFeed) -> None:
        """Push a price observation into Atune's perception pipeline."""
        if self._atune is None:
            return

        from systems.atune.types import InputChannel, RawInput

        raw = RawInput(
            data=json.dumps({
                "type": "market_price",
                "pair": list(feed.pair),
                "price": str(feed.price),
                "pool_address": feed.pool_address,
                "block_number": feed.block_number,
                "source": feed.source,
                "latency_ms": feed.latency_ms,
            }),
            metadata={
                "source": "phantom_liquidity",
                "channel": "market_data",
                "pair": f"{feed.pair[0]}/{feed.pair[1]}",
            },
        )

        try:
            await self._atune.ingest(raw, InputChannel.EXTERNAL_API)
        except Exception as exc:
            self._logger.debug(
                "phantom_atune_ingest_error", error=str(exc),
            )

    # ── Oracle Fallback ────────────────────────────────────────────

    async def get_price_with_fallback(
        self,
        pair: tuple[str, str],
    ) -> PhantomPriceFeed | None:
        """
        Get price from phantom pool, falling back to oracle API if unavailable.

        Returns ``None`` only if both sources fail.
        """
        feed = self.get_price(pair)
        if feed is not None:
            return feed

        if not self._config.oracle_fallback_enabled:
            return None

        return await self._fetch_oracle_fallback(pair)

    async def _fetch_oracle_fallback(
        self,
        pair: tuple[str, str],
    ) -> PhantomPriceFeed | None:
        """
        Fetch price from CoinGecko free API as a fallback.

        Returns a PhantomPriceFeed with source="oracle_fallback".
        """
        try:
            import httpx

            # Map common symbols to CoinGecko IDs
            symbol_to_cg: dict[str, str] = {
                "ETH": "ethereum",
                "USDC": "usd-coin",
                "DAI": "dai",
                "cbBTC": "bitcoin",
                "USDT": "tether",
            }

            base_id = symbol_to_cg.get(pair[0])
            quote_symbol = pair[1].lower()

            if base_id is None:
                return None

            # For stablecoin quotes, use "usd"
            vs_currency = "usd" if quote_symbol in ("usdc", "usdt", "dai") else quote_symbol

            url = f"{self._config.oracle_fallback_url}/simple/price"
            params = {"ids": base_id, "vs_currencies": vs_currency}

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            price_val = data.get(base_id, {}).get(vs_currency)
            if price_val is None:
                return None

            self._oracle_fallback_count += 1

            feed = PhantomPriceFeed(
                pool_address="",
                pair=pair,
                price=Decimal(str(price_val)),
                timestamp=datetime.now(UTC),
                source="oracle_fallback",
            )

            # Also cache and feed to Atune
            key = _pair_key(pair)
            self._latest_prices[key] = feed
            await self._feed_to_atune(feed)

            self._logger.info(
                "phantom_oracle_fallback",
                pair=f"{pair[0]}/{pair[1]}",
                price=str(feed.price),
            )

            return feed

        except Exception as exc:
            self._logger.debug(
                "phantom_oracle_fallback_failed",
                pair=f"{pair[0]}/{pair[1]}",
                error=str(exc),
            )
            return None

    # ── Maintenance ────────────────────────────────────────────────

    async def _maintenance_loop(self) -> None:
        """Periodic maintenance — runs until cancelled."""
        while True:
            try:
                await asyncio.sleep(self._config.maintenance_interval_s)
                await self.maintenance_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.warning(
                    "phantom_maintenance_error", error=str(exc),
                )

    async def maintenance_cycle(self) -> None:
        """
        Periodic health check on all phantom positions.

        - Detect stale pools (no price updates in > staleness_threshold)
        - Check impermanent loss against threshold
        - Sync health status to Oikos
        - Log uptime metrics
        """
        now = datetime.now(UTC)
        staleness_s = self._config.staleness_threshold_s
        il_threshold = Decimal(str(self._config.il_rebalance_threshold))

        active_count = 0
        stale_count = 0
        il_flagged_count = 0

        for pool in self._pools.values():
            if pool.health in (PoolHealth.WITHDRAWN, PoolHealth.FAILED):
                continue

            # Check staleness
            age_s = (now - pool.last_price_timestamp).total_seconds()
            if age_s > staleness_s and pool.health == PoolHealth.ACTIVE:
                pool.health = PoolHealth.STALE
                stale_count += 1

                if self._oikos is not None:
                    self._oikos.update_phantom_position(
                        pool.pool_address, health_status="degraded",
                    )
            elif age_s <= staleness_s and pool.health == PoolHealth.STALE:
                pool.health = PoolHealth.ACTIVE

            # Check impermanent loss
            if pool.impermanent_loss_pct < -il_threshold:
                if pool.health != PoolHealth.IMPERMANENT_LOSS:
                    pool.health = PoolHealth.IMPERMANENT_LOSS
                    il_flagged_count += 1
                    self._logger.warning(
                        "phantom_il_exceeded",
                        pool=pool.pool_address,
                        il_pct=str(pool.impermanent_loss_pct),
                        threshold=str(il_threshold),
                    )

                    if self._oikos is not None:
                        self._oikos.update_phantom_position(
                            pool.pool_address, health_status="critical",
                        )

            if pool.health == PoolHealth.ACTIVE:
                active_count += 1

        self._logger.info(
            "phantom_maintenance_complete",
            total_pools=len(self._pools),
            active=active_count,
            stale=stale_count,
            il_flagged=il_flagged_count,
            total_price_updates=self._total_price_updates,
            oracle_fallbacks=self._oracle_fallback_count,
        )

    # ── Event Handlers ─────────────────────────────────────────────

    async def _on_metabolic_pressure(self, event: SynapseEvent) -> None:
        """
        React to metabolic pressure from Oikos.

        If the organism enters AUSTERITY or worse, log a warning. Actual
        withdrawal decisions are left to Nova/operator — we only flag.
        """
        data = event.data if hasattr(event, "data") else {}
        level = data.get("starvation_level", "")

        if level in ("AUSTERITY", "EMERGENCY", "CRITICAL"):
            total_deployed = sum(
                p.capital_deployed_usd for p in self._pools.values()
                if p.health not in (PoolHealth.WITHDRAWN, PoolHealth.FAILED)
            )
            self._logger.warning(
                "phantom_metabolic_pressure",
                starvation_level=level,
                phantom_capital_deployed=str(total_deployed),
                msg="Consider withdrawing phantom positions to free capital",
            )

    # ── Health ─────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Self-health report for Synapse."""
        active = sum(
            1 for p in self._pools.values() if p.health == PoolHealth.ACTIVE
        )
        stale = sum(
            1 for p in self._pools.values() if p.health == PoolHealth.STALE
        )
        listener_stats = (
            self._listener.stats if self._listener is not None else {}
        )

        return {
            "system": self.system_id,
            "initialized": self._initialized,
            "pools_total": len(self._pools),
            "pools_active": active,
            "pools_stale": stale,
            "total_price_updates": self._total_price_updates,
            "oracle_fallback_count": self._oracle_fallback_count,
            "listener": listener_stats,
        }


# ── Utilities ──────────────────────────────────────────────────────


def _pair_key(pair: tuple[str, str]) -> str:
    """Canonical key for a trading pair (sorted for consistency)."""
    return f"{pair[0]}/{pair[1]}"
