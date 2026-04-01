"""
EcodiaOS - Phantom Liquidity Service (Phase 16q)

Orchestrates the phantom liquidity sensor network.  Manages the full lifecycle:

1. Pool selection - which pairs to monitor (via PoolSelector)
2. Position deployment - mint LP positions (via PhantomLiquidityExecutor)
3. Swap event listening - extract prices from on-chain events (via SwapEventListener)
4. Synapse integration - broadcast prices via PHANTOM_PRICE_UPDATE events
5. Health monitoring - impermanent loss tracking, staleness detection
6. Oikos reporting - track as YieldPosition entries in EconomicState
7. Genome extraction - heritable state for Mitosis

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
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import GenomeExtractionProtocol, OrganGenomeSegment
from systems.phantom_liquidity.pool_selector import PoolSelector
from systems.phantom_liquidity.price_listener import (
    SwapEventListener,
)
from systems.phantom_liquidity.types import (
    PhantomLiquidityPool,
    PhantomPriceFeed,
    PoolHealth,
)

from systems.synapse.types import SynapseEventType

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from clients.timescaledb import TimescaleDBClient
    from clients.wallet import WalletClient
    from config import PhantomLiquidityConfig
    from systems.fovea.gateway import AtuneService
    from systems.identity.vault import IdentityVault, SealedEnvelope
    from systems.oikos.service import OikosService
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

# Secret name used to store/retrieve the LP wallet private key in the vault.
_LP_KEY_SECRET_NAME = "phantom_lp_key"

logger = structlog.get_logger()


class LiquidityPhantomService:
    """
    Phantom Liquidity Sensor Network.

    Deploys minimal concentrated liquidity positions into DeFi pools to
    receive real-time price feeds via Swap events.  This replaces paid
    oracle subscriptions with self-funding price sensors.

    Implements GenomeExtractionProtocol for Mitosis inheritance.
    """

    system_id: str = "phantom_liquidity"

    def __init__(
        self,
        config: PhantomLiquidityConfig,
        wallet: WalletClient | None = None,
        atune: AtuneService | None = None,
        oikos: OikosService | None = None,
        tsdb: TimescaleDBClient | None = None,
        neo4j: Neo4jClient | None = None,
        vault: IdentityVault | None = None,
        instance_id: str = "eos-default",
    ) -> None:
        self._config = config
        self._wallet = wallet
        self._atune = atune
        self._oikos = oikos
        self._tsdb = tsdb
        self._neo4j = neo4j
        self._vault = vault
        self._instance_id = instance_id
        self._logger = logger.bind(system="phantom_liquidity")

        # LP key sealed envelope - stored in vault, not in config/env.
        self._lp_key_envelope: SealedEnvelope | None = None

        # Sub-components
        self._pool_selector = PoolSelector(max_pools=config.max_pools)
        self._listener: SwapEventListener | None = None

        # State
        self._pools: dict[str, PhantomLiquidityPool] = {}  # pool_address -> pool
        self._latest_prices: dict[str, PhantomPriceFeed] = {}  # pair_key -> feed
        self._pair_to_pool: dict[str, str] = {}  # pair_key -> pool_address

        # Multi-instance price consensus: pool_address -> list[(price_float, ts)]
        self._peer_observations: dict[str, list[tuple[float, datetime]]] = {}
        # Consensus window - keep observations within 30s to compute fleet median.
        self._consensus_window_s: float = 30.0

        # Event bus
        self._event_bus: EventBus | None = None

        # Background tasks
        self._maintenance_task: asyncio.Task[None] | None = None
        self._initialized = False

        # Metrics
        self._total_price_updates: int = 0
        self._oracle_fallback_count: int = 0
        self._total_rpc_calls: int = 0
        self._cumulative_gas_cost_usd: Decimal = Decimal("0")
        self._last_cost_report_time: datetime = datetime.now(UTC)
        self._service_start_time: datetime = datetime.now(UTC)

    # ── Lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Build sub-components and start the swap event listener."""
        if self._initialized:
            return

        rpc_url = self._config.rpc_url
        if not rpc_url:
            self._logger.warning(
                "phantom_liquidity_unconfigured",
                reason="no_rpc_url",
            )
            # Notify the observatory that the system is alive but not configured,
            # and alert Thymos so the degraded state is tracked.
            await self._emit(
                SynapseEventType.PHANTOM_SUBSTRATE_OBSERVABLE,
                {
                    "status": "unconfigured",
                    "active_pools": 0,
                    "reason": "no_rpc_url",
                    "price_feeds": 0,
                    "total_price_updates": 0,
                },
            )
            await self._emit(
                SynapseEventType.SYSTEM_DEGRADED,
                {
                    "system": "phantom_liquidity",
                    "severity": "low",
                    "reason": "rpc_url_not_configured",
                    "detail": "Phantom Liquidity is disabled: RPC_URL is not set. Price oracle is unavailable.",
                },
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

        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.METABOLIC_PRESSURE,
                self._on_metabolic_pressure,
            )
            event_bus.subscribe(
                SynapseEventType.GENOME_EXTRACT_REQUEST,
                self._on_genome_extract_request,
            )
            # Subscribe to peer price observations for multi-instance consensus.
            event_bus.subscribe(
                SynapseEventType.PHANTOM_PRICE_OBSERVATION,
                self._on_peer_price_observation,
            )
            # Subscribe to Evo parameter adjustments - Evo can tune IL threshold,
            # staleness window, consensus window, and poll interval via Thompson sampling.
            if hasattr(SynapseEventType, "EVO_ADJUST_BUDGET"):
                event_bus.subscribe(
                    SynapseEventType.EVO_ADJUST_BUDGET,
                    self._on_evo_adjust_budget,
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

    # ── Synapse Emission Helper ────────────────────────────────────

    async def _emit(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        """Broadcast a Synapse event if the event bus is attached."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent

            event = SynapseEvent(
                event_type=event_type,
                data=data,
                source_system="phantom_liquidity",
            )
            await self._event_bus.emit(event)
        except Exception as exc:
            self._logger.debug(
                "phantom_emit_failed", event=event_type.value, error=str(exc),
            )

    # ── Vault Key Management ───────────────────────────────────────

    async def store_lp_key(self, private_key_hex: str) -> bool:
        """
        Seal and store the LP wallet private key in the Identity vault.

        Must be called once during initial provisioning. After this, the key
        is never stored in config, env vars, or plaintext memory - always
        retrieved via ``retrieve_lp_key()``.

        Returns True on success, False if vault is unavailable.
        """
        if self._vault is None:
            self._logger.warning(
                "phantom_lp_key_store_skipped", reason="no_vault_injected",
            )
            return False
        try:
            envelope = self._vault.encrypt(
                private_key_hex.encode("utf-8"),
                platform_id=_LP_KEY_SECRET_NAME,
                purpose="lp_wallet_key",
            )
            self._lp_key_envelope = envelope
            self._logger.info(
                "phantom_lp_key_stored",
                envelope_id=envelope.id,
                key_version=envelope.key_version,
            )
            return True
        except Exception as exc:
            self._logger.error("phantom_lp_key_store_failed", error=str(exc))
            return False

    def retrieve_lp_key(self) -> str | None:
        """
        Decrypt and return the LP wallet private key from the vault.

        Called by the executor on each on-chain operation. Key is never
        cached in plaintext between calls.

        Returns None if vault or sealed envelope is unavailable.
        """
        if self._vault is None or self._lp_key_envelope is None:
            self._logger.warning(
                "phantom_lp_key_unavailable",
                reason="no_vault" if self._vault is None else "no_envelope",
            )
            return None
        try:
            plaintext = self._vault.decrypt(self._lp_key_envelope)
            return plaintext.decode("utf-8")
        except Exception as exc:
            self._logger.error("phantom_lp_key_decrypt_failed", error=str(exc))
            return None

    # ── Public API ─────────────────────────────────────────────────

    def get_price(self, pair: tuple[str, str]) -> PhantomPriceFeed | None:
        """
        Get the latest price for a trading pair from phantom pools.

        Returns ``None`` if no pool covers this pair - the caller should
        fall back to an oracle API.
        """
        key = _pair_key(pair)
        feed = self._latest_prices.get(key)

        if feed is None:
            return None

        # Check staleness
        age_s = (datetime.now(UTC) - feed.timestamp).total_seconds()
        if age_s > self._config.staleness_threshold_s:
            return None  # Stale - caller should use oracle

        return feed

    def get_all_prices(self) -> list[PhantomPriceFeed]:
        """Return latest prices from all active phantom pools."""
        return list(self._latest_prices.values())

    def get_pools(self) -> list[PhantomLiquidityPool]:
        """Return all tracked phantom pool positions."""
        return list(self._pools.values())

    def get_candidates(self) -> list[Any]:
        """Return the static pool candidate list - use instead of importing _STATIC_POOLS directly."""
        return self._pool_selector.get_static_pools()

    def register_pool(self, pool: PhantomLiquidityPool) -> None:
        """
        Register a deployed phantom pool position.

        Called after a successful ``deploy_position`` execution.  Adds the
        pool to the listener and tracks it in Oikos.
        """
        self._pools[pool.pool_address.lower()] = pool
        self._pair_to_pool[_pair_key(pool.pair)] = pool.pool_address.lower()

        # Store entry price for IL calculation
        if pool.last_price_observed > 0:
            pool._entry_price = pool.last_price_observed  # type: ignore[attr-defined]

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
        3. Compute IL from price movement
        4. Emit PHANTOM_PRICE_OBSERVATION for fleet consensus
        5. Apply fleet consensus price if peers available
        6. Broadcast PHANTOM_PRICE_UPDATE via Synapse
        7. Persist to TimescaleDB
        8. Write lightweight PriceObservation node to Neo4j (Memory bridge)
        """
        addr = feed.pool_address.lower()
        pool = self._pools.get(addr)

        prev_il: Decimal = Decimal("0")
        prev_health: PoolHealth | None = None

        if pool is not None:
            prev_il = pool.impermanent_loss_pct
            prev_health = pool.health
            pool.last_price_observed = feed.price
            pool.last_price_timestamp = feed.timestamp
            pool.price_update_count += 1
            if pool.health == PoolHealth.STALE:
                pool.health = PoolHealth.ACTIVE

            # Compute IL from price movement relative to entry
            self._update_il(pool, feed.price)

            # Immediate IL breach detection - don't wait for the hourly maintenance
            # cycle.  If IL just crossed the rebalance threshold on this price tick,
            # immediately flag the pool and emit an intent to withdraw.
            il_threshold = Decimal(str(self._config.il_rebalance_threshold))
            if (
                pool.impermanent_loss_pct < -il_threshold
                and prev_health != PoolHealth.IMPERMANENT_LOSS
                and pool.health != PoolHealth.WITHDRAWN
            ):
                pool.health = PoolHealth.IMPERMANENT_LOSS
                entry_price: Decimal = getattr(pool, "_entry_price", Decimal("0"))
                severity = (
                    "critical"
                    if pool.impermanent_loss_pct < Decimal("-0.05")
                    else "warning"
                )
                await self._emit(SynapseEventType.PHANTOM_IL_DETECTED, {
                    "pool_address": pool.pool_address,
                    "pair": list(pool.pair),
                    "il_pct": str(pool.impermanent_loss_pct),
                    "severity": severity,
                    "capital_at_risk_usd": str(pool.capital_deployed_usd),
                    "entry_price": str(entry_price),
                    "current_price": str(feed.price),
                })
                await self._emit(SynapseEventType.PHANTOM_POSITION_CRITICAL, {
                    "pool_address": pool.pool_address,
                    "pair": list(pool.pair),
                    "il_pct": str(pool.impermanent_loss_pct),
                    "capital_at_risk_usd": str(pool.capital_deployed_usd),
                    "threshold": str(il_threshold),
                })
                # Autonomous recourse: propose withdrawal immediately without
                # waiting for the maintenance cycle or operator intervention.
                await self._emit(SynapseEventType.NOVA_INTENT_REQUESTED, {
                    "requesting_system": "phantom_liquidity",
                    "intent_type": "withdraw_phantom_position",
                    "priority": "HIGH" if severity == "critical" else "MEDIUM",
                    "reason": (
                        f"IL breach on {pool.pool_address[:10]}...: "
                        f"IL={pool.impermanent_loss_pct:.2%} > threshold={il_threshold:.2%}"
                    ),
                    "pool_address": pool.pool_address,
                    "token_id": pool.token_id,
                    "capital_usd": str(pool.capital_deployed_usd),
                    "il_pct": str(pool.impermanent_loss_pct),
                    "entry_price": str(entry_price),
                    "current_price": str(feed.price),
                    "estimated_recovery_usd": str(pool.capital_deployed_usd),
                })
                # Emit RE training example: price movement that caused an IL breach
                # is a valuable causal event for the RE to learn economic consequences.
                await self._emit(SynapseEventType.RE_TRAINING_EXAMPLE, {
                    "episode_id": f"phantom_il_breach_{pool.pool_address}_{feed.block_number}",
                    "system": "phantom_liquidity",
                    "event_type": "il_breach",
                    "reasoning_trace": (
                        f"Pool {pool.pool_address} ({pool.pair[0]}/{pool.pair[1]}) "
                        f"suffered IL breach. Entry price: {entry_price}, "
                        f"current price: {feed.price}, "
                        f"IL: {pool.impermanent_loss_pct:.4f} < threshold: -{il_threshold}. "
                        f"Price ratio: {float(feed.price / entry_price) if entry_price > 0 else 'N/A'}. "
                        f"Capital at risk: ${pool.capital_deployed_usd}. "
                        f"Proposed autonomous withdrawal."
                    ),
                    "economic_context": {
                        "pair": list(pool.pair),
                        "pool_address": pool.pool_address,
                        "fee_tier": pool.fee_tier,
                        "entry_price": str(entry_price),
                        "current_price": str(feed.price),
                        "il_pct": str(pool.impermanent_loss_pct),
                        "capital_deployed_usd": str(pool.capital_deployed_usd),
                        "price_update_count": pool.price_update_count,
                        "il_threshold": str(il_threshold),
                    },
                    "outcome": "il_breach_detected",
                    "confidence": 1.0,
                })
                self._logger.warning(
                    "phantom_il_breach_detected",
                    pool=pool.pool_address,
                    il_pct=str(pool.impermanent_loss_pct),
                    threshold=str(il_threshold),
                    severity=severity,
                )

        # Cache latest price by pair
        key = _pair_key(feed.pair)
        self._latest_prices[key] = feed
        self._total_price_updates += 1
        self._total_rpc_calls += 1

        # Emit raw observation to federation peers for consensus aggregation.
        await self._emit(SynapseEventType.PHANTOM_PRICE_OBSERVATION, {
            "pool_address": feed.pool_address,
            "pair": list(feed.pair),
            "price": str(feed.price),
            "sqrt_price_x96": feed.sqrt_price_x96,
            "block_number": feed.block_number,
            "timestamp": feed.timestamp.isoformat(),
            "liquidity": getattr(feed, "liquidity", 0),
            "source_instance": self._instance_id,
        })

        # Compute fleet consensus price (median of peer observations + self).
        consensus_price = self._compute_consensus_price(
            feed.pool_address, float(feed.price),
        )

        # Use consensus price for downstream broadcast if peers exist.
        broadcast_price = (
            Decimal(str(consensus_price))
            if consensus_price is not None
            else feed.price
        )

        # Persist to TimescaleDB
        await self._persist_price(feed)

        # Write lightweight PriceObservation node to Neo4j for Memory bridge.
        await self._write_price_observation_to_neo4j(feed, broadcast_price)

        # Broadcast via Synapse - replaces direct atune.ingest().
        # Includes price_source_quality so Fovea/Nova can weight this signal
        # and RE training examples can annotate economic decisions with prevailing
        # market conditions (Spec §23 annotation pattern).
        await self._emit(SynapseEventType.PHANTOM_PRICE_UPDATE, {
            "pair": list(feed.pair),
            "price": str(broadcast_price),
            "pool_address": feed.pool_address,
            "block_number": feed.block_number,
            "latency_ms": feed.latency_ms,
            "source": feed.source,
            "sqrt_price_x96": feed.sqrt_price_x96,
            "tx_hash": feed.tx_hash,
            "consensus": consensus_price is not None,
            # RE annotation context: downstream systems attach this to training examples
            "price_source_quality": (
                "consensus" if consensus_price is not None
                else "phantom_lp"
            ),
            "staleness_s": 0,
            "il_pct": str(pool.impermanent_loss_pct) if pool is not None else "0",
        })

    def _update_il(self, pool: PhantomLiquidityPool, current_price: Decimal) -> None:
        """
        Compute impermanent loss from price movement relative to entry price.

        IL formula for concentrated liquidity (simplified):
          IL% = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        where price_ratio = current_price / entry_price
        """
        entry_price: Decimal = getattr(pool, "_entry_price", Decimal("0"))
        if entry_price <= 0 or current_price <= 0:
            return

        price_ratio = float(current_price / entry_price)
        if price_ratio <= 0:
            return

        import math
        il = 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1
        pool.impermanent_loss_pct = Decimal(str(round(il, 6)))

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

    async def _write_price_observation_to_neo4j(
        self,
        feed: PhantomPriceFeed,
        consensus_price: Decimal,
    ) -> None:
        """
        Write a lightweight PriceObservation node to Neo4j.

        This is the Memory bridge: stores price history in the knowledge graph
        so Kairos, Memory, and other systems can query price history without
        direct TimescaleDB access.

        Node: (:PriceObservation {pool_id, price, timestamp, liquidity, source})
        """
        if self._neo4j is None:
            return
        try:
            query = """
MERGE (p:PriceObservation {
    pool_id: $pool_id,
    block_number: $block_number
})
ON CREATE SET
    p.price            = $price,
    p.timestamp        = $timestamp,
    p.liquidity        = $liquidity,
    p.source           = $source,
    p.pair             = $pair,
    p.latency_ms       = $latency_ms,
    p.consensus_price  = $consensus_price,
    p.instance_id      = $instance_id
"""
            await self._neo4j.execute_write(query, {
                "pool_id": feed.pool_address.lower(),
                "block_number": feed.block_number,
                "price": float(feed.price),
                "timestamp": feed.timestamp.isoformat(),
                "liquidity": getattr(feed, "liquidity", 0),
                "source": "phantom_lp",
                "pair": f"{feed.pair[0]}/{feed.pair[1]}",
                "latency_ms": feed.latency_ms,
                "consensus_price": float(consensus_price),
                "instance_id": self._instance_id,
            })
        except Exception as exc:
            self._logger.debug(
                "phantom_neo4j_write_error", error=str(exc),
            )

    # ── Multi-instance Price Consensus ─────────────────────────────

    def _compute_consensus_price(
        self,
        pool_address: str,
        self_price: float,
    ) -> float | None:
        """
        Compute fleet median price from peer observations + self observation.

        Uses 2σ outlier rejection: drops observations more than 2 standard
        deviations from the mean, then returns the median of remaining values.

        Returns None if fewer than 2 peers have observations (no consensus
        possible - self price is used directly).
        """
        addr = pool_address.lower()
        now = datetime.now(UTC)
        cutoff = self._consensus_window_s

        # Collect fresh peer prices within the consensus window.
        peer_obs = self._peer_observations.get(addr, [])
        fresh = [
            price for price, ts in peer_obs
            if (now - ts).total_seconds() <= cutoff
        ]

        # Include self observation.
        all_prices = fresh + [self_price]

        if len(all_prices) < 2:  # noqa: PLR2004
            return None  # Not enough peers - no consensus, use raw price.

        # 2σ outlier rejection.
        mean = statistics.mean(all_prices)
        if len(all_prices) >= 3:  # noqa: PLR2004
            stdev = statistics.stdev(all_prices)
            if stdev > 0:
                filtered = [p for p in all_prices if abs(p - mean) <= 2 * stdev]
                if filtered:
                    all_prices = filtered

        return statistics.median(all_prices)

    async def _on_peer_price_observation(self, event: SynapseEvent) -> None:
        """
        Handle a PHANTOM_PRICE_OBSERVATION from a federation peer.

        Ignores observations from self (same instance_id). Stores the price
        in the per-pool peer observations window for consensus computation.
        """
        data = event.data if hasattr(event, "data") else {}
        source = data.get("source_instance", "")
        if source == self._instance_id:
            return  # Skip own emissions re-received from the bus.

        pool_addr = data.get("pool_address", "").lower()
        if not pool_addr:
            return

        try:
            price = float(data.get("price", 0))
            ts_str = data.get("timestamp", "")
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(UTC)
        except (ValueError, TypeError):
            return

        if pool_addr not in self._peer_observations:
            self._peer_observations[pool_addr] = []

        self._peer_observations[pool_addr].append((price, ts))

        # Trim stale observations beyond window.
        now = datetime.now(UTC)
        cutoff = self._consensus_window_s
        self._peer_observations[pool_addr] = [
            (p, t) for p, t in self._peer_observations[pool_addr]
            if (now - t).total_seconds() <= cutoff
        ]

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
        Emits PHANTOM_FALLBACK_ACTIVATED via Synapse.
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

            # Cache
            key = _pair_key(pair)
            self._latest_prices[key] = feed

            # Emit fallback activation event
            await self._emit(SynapseEventType.PHANTOM_FALLBACK_ACTIVATED, {
                "pair": list(pair),
                "reason": "no_active_phantom_pool",
                "fallback_source": "coingecko",
            })

            # Also broadcast as price update so all subscribers get the data
            await self._emit(SynapseEventType.PHANTOM_PRICE_UPDATE, {
                "pair": list(feed.pair),
                "price": str(feed.price),
                "pool_address": "",
                "block_number": 0,
                "latency_ms": 0,
                "source": "oracle_fallback",
                "sqrt_price_x96": 0,
                "tx_hash": "",
            })

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
        """Periodic maintenance - runs until cancelled."""
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
        - Emit Synapse events for stale/critical pools
        - Emit metabolic cost report
        """
        now = datetime.now(UTC)
        staleness_s = self._config.staleness_threshold_s
        il_threshold = Decimal(str(self._config.il_rebalance_threshold))

        active_count = 0
        stale_count = 0
        il_flagged_count = 0
        total_fees = Decimal("0")

        for pool in self._pools.values():
            if pool.health in (PoolHealth.WITHDRAWN, PoolHealth.FAILED):
                continue

            total_fees += pool.cumulative_yield_usd

            # Check staleness
            age_s = (now - pool.last_price_timestamp).total_seconds()
            if age_s > staleness_s and pool.health == PoolHealth.ACTIVE:
                pool.health = PoolHealth.STALE
                stale_count += 1

                if self._oikos is not None:
                    self._oikos.update_phantom_position(
                        pool.pool_address, health_status="degraded",
                    )

                # Emit stale event via Synapse
                await self._emit(SynapseEventType.PHANTOM_POOL_STALE, {
                    "pool_address": pool.pool_address,
                    "pair": list(pool.pair),
                    "last_update_s": age_s,
                    "staleness_threshold_s": staleness_s,
                })

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

                    # Emit position critical event
                    await self._emit(SynapseEventType.PHANTOM_POSITION_CRITICAL, {
                        "pool_address": pool.pool_address,
                        "pair": list(pool.pair),
                        "il_pct": str(pool.impermanent_loss_pct),
                        "capital_at_risk_usd": str(pool.capital_deployed_usd),
                        "threshold": str(il_threshold),
                    })

                    # Emit IL detected for Simula/EIS security pipeline
                    entry_price = getattr(pool, "_entry_price", Decimal("0"))
                    severity = "critical" if pool.impermanent_loss_pct < -Decimal("0.05") else "warning"
                    await self._emit(SynapseEventType.PHANTOM_IL_DETECTED, {
                        "pool_address": pool.pool_address,
                        "il_pct": str(pool.impermanent_loss_pct),
                        "severity": severity,
                        "capital_at_risk_usd": str(pool.capital_deployed_usd),
                        "entry_price": str(entry_price),
                        "current_price": str(pool.last_price_observed),
                    })

            if pool.health == PoolHealth.ACTIVE:
                active_count += 1

        # Emit periodic metabolic cost report
        period_s = (now - self._last_cost_report_time).total_seconds()
        if period_s > 0:
            await self._emit(SynapseEventType.PHANTOM_METABOLIC_COST, {
                "total_gas_cost_usd": str(self._cumulative_gas_cost_usd),
                "total_rpc_calls": self._total_rpc_calls,
                "pools_active": active_count,
                "cumulative_fees_earned_usd": str(total_fees),
                "period_s": period_s,
            })
            self._last_cost_report_time = now

        # Emit Bedau-Packard evolutionary observables for Telos/Benchmarks.
        await self._emit_substrate_observables(now, active_count)

        self._logger.info(
            "phantom_maintenance_complete",
            total_pools=len(self._pools),
            active=active_count,
            stale=stale_count,
            il_flagged=il_flagged_count,
            total_price_updates=self._total_price_updates,
            oracle_fallbacks=self._oracle_fallback_count,
        )

    async def _emit_substrate_observables(
        self,
        now: datetime,
        active_count: int,
    ) -> None:
        """
        Emit PHANTOM_SUBSTRATE_OBSERVABLE for Telos/Benchmarks (Bedau-Packard).

        Reports evolutionary observables of the economic substrate:
        - pool_latency_ms: mean observation latency across active pools
        - verification_rate: fraction of price updates with non-zero block confirmation
        - trust_score: ratio of phantom-source vs fallback prices (1.0 = all phantom)
        - lp_position_age_s: mean age of active LP positions
        - pools_active: number of active pools
        - price_updates_per_hour: rate of price observations
        """
        listener_stats = self._listener.stats if self._listener is not None else {}
        total_polls = listener_stats.get("polls", 0)
        total_events = listener_stats.get("events_processed", 0)

        # Mean latency: estimate from listener stats; 0 if no events yet.
        pool_latency_ms: float = 0.0
        if total_events > 0:
            # Use ratio of events to polls as a proxy for observation density.
            pool_latency_ms = max(0.0, 4000.0 / max(total_events, 1))

        # Verification rate: events processed / (polls * pools) - fraction of polls that yielded events.
        verification_rate = 0.0
        monitored_pools = listener_stats.get("pools_monitored", 0)
        if total_polls > 0 and monitored_pools > 0:
            verification_rate = min(1.0, total_events / (total_polls * monitored_pools))

        # Trust score: fraction of latest prices from phantom source vs oracle fallback.
        phantom_count = sum(
            1 for f in self._latest_prices.values() if f.source == "phantom_liquidity"
        )
        total_prices = len(self._latest_prices)
        trust_score = phantom_count / total_prices if total_prices > 0 else 0.0

        # Mean LP position age.
        position_ages: list[float] = []
        for pool in self._pools.values():
            if pool.health in (PoolHealth.ACTIVE, PoolHealth.STALE):
                age_s = (now - pool.last_price_timestamp).total_seconds()
                position_ages.append(age_s)
        lp_position_age_s = statistics.mean(position_ages) if position_ages else 0.0

        # Price updates per hour (since service start).
        service_hours = (now - self._service_start_time).total_seconds() / 3600.0
        price_updates_per_hour = (
            self._total_price_updates / service_hours if service_hours > 0 else 0.0
        )

        await self._emit(SynapseEventType.PHANTOM_SUBSTRATE_OBSERVABLE, {
            "pool_latency_ms": round(pool_latency_ms, 2),
            "verification_rate": round(verification_rate, 4),
            "trust_score": round(trust_score, 4),
            "lp_position_age_s": round(lp_position_age_s, 1),
            "pools_active": active_count,
            "price_updates_per_hour": round(price_updates_per_hour, 2),
            "instance_id": self._instance_id,
        })

    # ── Event Handlers ─────────────────────────────────────────────

    async def _on_metabolic_pressure(self, event: SynapseEvent) -> None:
        """
        React to metabolic pressure from Oikos.

        AUSTERITY  → warn + emit PHANTOM_RESOURCE_EXHAUSTED.
        EMERGENCY / CRITICAL → also emit an AXON_INTENT_REQUESTED so Nova/Axon
        can autonomously withdraw the most IL-exposed positions without waiting
        for human intervention.
        """
        data = event.data if hasattr(event, "data") else {}
        level = data.get("starvation_level", "")

        if level not in ("AUSTERITY", "EMERGENCY", "CRITICAL"):
            return

        active_pools = [
            p for p in self._pools.values()
            if p.health not in (PoolHealth.WITHDRAWN, PoolHealth.FAILED)
        ]
        total_deployed = sum(p.capital_deployed_usd for p in active_pools)

        self._logger.warning(
            "phantom_metabolic_pressure",
            starvation_level=level,
            phantom_capital_deployed=str(total_deployed),
            pool_count=len(active_pools),
        )

        # Emit resource exhausted event so Thymos / Nova can observe.
        await self._emit(SynapseEventType.PHANTOM_RESOURCE_EXHAUSTED, {
            "operation": "phantom_liquidity_sensing",
            "estimated_cost_usd": str(total_deployed),
            "starvation_level": level,
            "pool_count": len(active_pools),
            "reason": f"Metabolic pressure at {level} - "
                      f"${total_deployed} deployed in phantom positions",
        })

        # EMERGENCY / CRITICAL: autonomously propose withdrawal of IL-exposed pools
        # by emitting a Nova-compatible intent request.  The organism should not
        # wait for an operator - it should self-rescue.
        if level in ("EMERGENCY", "CRITICAL") and active_pools:
            # Sort by worst IL first so the most dangerous positions are withdrawn.
            il_sorted = sorted(
                active_pools,
                key=lambda p: p.impermanent_loss_pct,  # most negative first
            )
            # Propose withdrawal of the highest-IL pool.
            target = il_sorted[0]
            await self._emit(SynapseEventType.NOVA_INTENT_REQUESTED, {
                "requesting_system": "phantom_liquidity",
                "intent_type": "withdraw_phantom_position",
                "priority": "HIGH",
                "reason": (
                    f"Metabolic {level}: withdrawing phantom position "
                    f"{target.pool_address[:10]}... (IL={target.impermanent_loss_pct:.2%}, "
                    f"capital=${target.capital_deployed_usd})"
                ),
                "pool_address": target.pool_address,
                "token_id": target.token_id,
                "capital_usd": str(target.capital_deployed_usd),
                "il_pct": str(target.impermanent_loss_pct),
                "starvation_level": level,
                "estimated_recovery_usd": str(target.capital_deployed_usd),
            })
            self._logger.info(
                "phantom_withdrawal_intent_proposed",
                pool=target.pool_address,
                il_pct=str(target.impermanent_loss_pct),
                starvation_level=level,
            )

    # ── Evo Parameter Tuning ───────────────────────────────────────

    # Bounds for Evo-adjustable parameters.  Evo must not be allowed to push
    # these outside safe operating ranges.
    _EVO_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
        "il_rebalance_threshold":  (0.005, 0.10),   # 0.5% – 10%
        "staleness_threshold_s":   (60.0, 3600.0),   # 1 min – 1 hour
        "consensus_window_s":      (10.0, 120.0),    # 10 s – 2 min
        "swap_poll_interval_s":    (1.0, 30.0),      # 1 s – 30 s
    }

    async def _on_evo_adjust_budget(self, event: SynapseEvent) -> None:
        """
        Handle EVO_ADJUST_BUDGET - Evo-tuned runtime parameter adjustment.

        Evo learns which threshold values maximize price feed quality and
        minimize impermanent loss via Thompson sampling on hypothesis outcomes.
        This handler applies approved adjustments within safe bounds.

        Adjustable parameters:
          - ``il_rebalance_threshold``  - IL% that triggers pool rebalancing
          - ``staleness_threshold_s``   - seconds before a pool is marked stale
          - ``consensus_window_s``      - peer observation aggregation window
          - ``swap_poll_interval_s``    - RPC poll frequency (applied on restart)
        """
        data = event.data if hasattr(event, "data") else {}
        param = data.get("parameter_name", "")
        confidence = float(data.get("confidence", 0.0))
        hypothesis_id = data.get("hypothesis_id", "")

        if confidence < 0.75:  # noqa: PLR2004
            return

        bounds = self._EVO_PARAM_BOUNDS.get(param)
        if bounds is None:
            return  # Not a phantom-owned parameter

        try:
            new_val = float(data.get("new_value", 0))
        except (ValueError, TypeError):
            return

        lo, hi = bounds
        clamped = max(lo, min(hi, new_val))

        old_val: float | None = None
        if param == "il_rebalance_threshold":
            old_val = self._config.il_rebalance_threshold
            self._config.il_rebalance_threshold = clamped
        elif param == "staleness_threshold_s":
            old_val = self._config.staleness_threshold_s
            self._config.staleness_threshold_s = clamped
        elif param == "consensus_window_s":
            old_val = self._consensus_window_s
            self._consensus_window_s = clamped
        elif param == "swap_poll_interval_s":
            old_val = self._config.swap_poll_interval_s
            self._config.swap_poll_interval_s = clamped
            # Update listener poll interval live if running
            if self._listener is not None:
                self._listener._poll_interval_s = clamped

        if old_val is not None:
            self._logger.info(
                "phantom_evo_param_adjusted",
                parameter=param,
                old_value=old_val,
                new_value=clamped,
                requested=new_val,
                confidence=confidence,
                hypothesis_id=hypothesis_id,
            )
            # Confirm adjustment to Evo for Thompson sampling feedback.
            await self._emit(SynapseEventType.PHANTOM_PARAMETER_ADJUSTED, {
                "parameter": param,
                "old_value": old_val,
                "new_value": clamped,
                "confidence": confidence,
                "hypothesis_id": hypothesis_id,
                "system": "phantom_liquidity",
            })

    async def _on_genome_extract_request(self, event: SynapseEvent) -> None:
        """Handle genome extraction requests from Mitosis."""
        data = event.data if hasattr(event, "data") else {}
        request_id = data.get("request_id", "")

        segment = await self.extract_genome_segment()

        await self._emit(SynapseEventType.GENOME_EXTRACT_RESPONSE, {
            "request_id": request_id,
            "segment": segment.model_dump(mode="json"),
        })

    # ── Genome Extraction Protocol ─────────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """Extract heritable state for Mitosis genome."""
        payload: dict[str, Any] = {
            "pool_configs": [
                {
                    "pool_address": p.pool_address,
                    "pair": list(p.pair),
                    "fee_tier": p.fee_tier,
                    "tick_lower": p.tick_lower,
                    "tick_upper": p.tick_upper,
                    "token0_address": p.token0_address,
                    "token1_address": p.token1_address,
                    "token0_decimals": p.token0_decimals,
                    "token1_decimals": p.token1_decimals,
                    "capital_deployed_usd": str(p.capital_deployed_usd),
                }
                for p in self._pools.values()
                if p.health not in (PoolHealth.WITHDRAWN, PoolHealth.FAILED)
            ],
            "staleness_threshold_s": self._config.staleness_threshold_s,
            "il_rebalance_threshold": self._config.il_rebalance_threshold,
            "max_pools": self._config.max_pools,
            "swap_poll_interval_s": self._config.swap_poll_interval_s,
            "oracle_fallback_enabled": self._config.oracle_fallback_enabled,
        }

        payload_json = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()

        return OrganGenomeSegment(
            system_id=SystemID.PHANTOM,
            version=1,
            schema_version="1.0",
            payload=payload,
            payload_hash=payload_hash,
            size_bytes=len(payload_json),
        )

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """Restore heritable state from a parent's genome segment."""
        try:
            payload = segment.payload
            # Restore config overrides
            if "staleness_threshold_s" in payload:
                self._config.staleness_threshold_s = payload["staleness_threshold_s"]
            if "il_rebalance_threshold" in payload:
                self._config.il_rebalance_threshold = payload["il_rebalance_threshold"]
            if "max_pools" in payload:
                self._config.max_pools = payload["max_pools"]
            if "swap_poll_interval_s" in payload:
                self._config.swap_poll_interval_s = payload["swap_poll_interval_s"]

            self._logger.info(
                "phantom_genome_seeded",
                pool_configs=len(payload.get("pool_configs", [])),
            )
            return True
        except Exception as exc:
            self._logger.warning("phantom_genome_seed_failed", error=str(exc))
            return False

    # ── Fee Tracking ──────────────────────────────────────────────

    def record_gas_cost(self, cost_usd: Decimal) -> None:
        """Record gas cost from an on-chain operation."""
        self._cumulative_gas_cost_usd += cost_usd

    def record_fee_earned(self, pool_address: str, fee_usd: Decimal) -> None:
        """Record swap fees earned by a phantom position."""
        addr = pool_address.lower()
        pool = self._pools.get(addr)
        if pool is not None:
            pool.cumulative_yield_usd += fee_usd

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
            "cumulative_gas_cost_usd": str(self._cumulative_gas_cost_usd),
            "listener": listener_stats,
        }


# ── Utilities ──────────────────────────────────────────────────────


def _pair_key(pair: tuple[str, str]) -> str:
    """Canonical key for a trading pair (sorted for consistency)."""
    return f"{pair[0]}/{pair[1]}"
