"""
EcodiaOS - Phantom Liquidity Types (Phase 16q)

Pydantic models for the phantom liquidity sensor network.  Each model
follows the ``EOSBaseModel`` convention from ``primitives/common.py``
and uses ``Decimal`` for all monetary values.
"""

from __future__ import annotations

import enum
from decimal import Decimal
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ---------------------------------------------------------------------------
# Pool health
# ---------------------------------------------------------------------------


class PoolHealth(enum.StrEnum):
    """Health state of a phantom liquidity pool position."""

    ACTIVE = "active"                          # Position is live, emitting prices
    STALE = "stale"                            # No swap events in > staleness threshold
    IMPERMANENT_LOSS = "impermanent_loss"      # IL exceeds rebalance threshold
    WITHDRAWN = "withdrawn"                    # Position has been removed
    PENDING_DEPLOY = "pending_deploy"          # Queued for deployment
    FAILED = "failed"                          # Deploy tx failed


# ---------------------------------------------------------------------------
# Core position model
# ---------------------------------------------------------------------------


class PhantomLiquidityPool(EOSBaseModel):
    """
    A single phantom LP position used as a price sensor.

    Each position is a Uniswap V3 concentrated liquidity position with a
    deliberately wide tick range to minimise impermanent loss.  The primary
    purpose is price observation, not yield - though the positions do earn
    small swap fees as a side benefit.
    """

    id: str = Field(default_factory=new_id)
    pool_address: str                          # Uniswap V3 pool contract
    token_id: int = 0                          # NFT token ID from NonfungiblePositionManager
    pair: tuple[str, str]                      # e.g. ("USDC", "ETH")
    token0_address: str = ""
    token1_address: str = ""
    token0_decimals: int = 18
    token1_decimals: int = 18

    # Capital
    capital_deployed_usd: Decimal = Decimal("0")
    amount0_deployed: int = 0                  # Raw token0 amount (smallest unit)
    amount1_deployed: int = 0                  # Raw token1 amount (smallest unit)

    # Uniswap V3 position parameters
    fee_tier: int = 3000                       # 100 | 500 | 3000 | 10000
    tick_lower: int = 0
    tick_upper: int = 0

    # Price observations
    last_price_observed: Decimal = Decimal("0")
    last_price_timestamp: datetime = Field(default_factory=utc_now)
    price_update_count: int = 0

    # Yield tracking
    cumulative_yield_usd: Decimal = Decimal("0")
    impermanent_loss_pct: Decimal = Decimal("0")

    # Lifecycle
    health: PoolHealth = PoolHealth.PENDING_DEPLOY
    deployed_at: datetime | None = None
    deploy_tx_hash: str = ""
    withdraw_tx_hash: str = ""

    # Oikos linkage - tracked as a YieldPosition
    yield_position_id: str = ""


# ---------------------------------------------------------------------------
# Price feed
# ---------------------------------------------------------------------------


class PhantomPriceFeed(EOSBaseModel):
    """
    A single price observation extracted from a Swap event.

    Injected into Atune via the ``EXTERNAL_API`` channel.  The ``source``
    field distinguishes phantom-liquidity feeds from oracle fallbacks.
    """

    id: str = Field(default_factory=new_id)
    pool_address: str
    pair: tuple[str, str]
    price: Decimal
    sqrt_price_x96: int = 0                    # Raw sqrtPriceX96 from event
    timestamp: datetime = Field(default_factory=utc_now)
    block_number: int = 0
    tx_hash: str = ""
    source: str = "phantom_liquidity"          # "phantom_liquidity" | "oracle_fallback"
    latency_ms: int = 0                        # Time from block to observation


# ---------------------------------------------------------------------------
# Pool selection
# ---------------------------------------------------------------------------


class PoolSelectionCandidate(EOSBaseModel):
    """A candidate pool evaluated during pool selection."""

    pool_address: str
    pair: tuple[str, str]
    token0_address: str = ""
    token1_address: str = ""
    token0_decimals: int = 18
    token1_decimals: int = 18
    fee_tier: int = 3000
    volume_24h_usd: Decimal = Decimal("0")
    tvl_usd: Decimal = Decimal("0")
    relevance_score: Decimal = Decimal("0")    # 0–1, relevance to EOS goals
    selected: bool = False
