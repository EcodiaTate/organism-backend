"""
EcodiaOS — Phantom Liquidity Pool Selector (Phase 16q)

Selects optimal Uniswap V3 pools on Base L2 for phantom liquidity deployment.

Phase 1: static curated list of high-volume pairs.
Phase 2 (Evo-tuned): dynamic selection via DeFiLlama API + Evo weighting.

The selector also computes wide tick ranges for sensor positions, deliberately
sacrificing capital efficiency to minimise impermanent loss — these are sensors,
not profit centres.
"""

from __future__ import annotations

import math
from decimal import Decimal

import structlog

from systems.phantom_liquidity.types import (
    PhantomLiquidityPool,
    PoolSelectionCandidate,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Token addresses on Base L2 (chain ID 8453)
# ---------------------------------------------------------------------------

USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
WETH_BASE = "0x4200000000000000000000000000000000000006"
DAI_BASE = "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb"
CBBTC_BASE = "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf"
USDT_BASE = "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2"

# Token decimal maps (for sqrtPriceX96 conversion)
TOKEN_DECIMALS: dict[str, int] = {
    USDC_BASE.lower(): 6,
    WETH_BASE.lower(): 18,
    DAI_BASE.lower(): 18,
    CBBTC_BASE.lower(): 8,
    USDT_BASE.lower(): 6,
}

TOKEN_SYMBOLS: dict[str, str] = {
    USDC_BASE.lower(): "USDC",
    WETH_BASE.lower(): "ETH",
    DAI_BASE.lower(): "DAI",
    CBBTC_BASE.lower(): "cbBTC",
    USDT_BASE.lower(): "USDT",
}

# ---------------------------------------------------------------------------
# Static pool list — Phase 1 (curated high-volume Base L2 pairs)
# ---------------------------------------------------------------------------
# Pool addresses are computed deterministically from the Uniswap V3 Factory
# on Base (0x33128a8fC17869897dcE68Ed026d694621f6FDfD) using CREATE2.
# These are the canonical pools for the listed pairs and fee tiers.

_STATIC_POOLS: list[PoolSelectionCandidate] = [
    PoolSelectionCandidate(
        pool_address="0x4C36388bE6F416A29C8d8Eee81C771cE6bE14B18",
        pair=("USDC", "ETH"),
        token0_address=USDC_BASE,
        token1_address=WETH_BASE,
        token0_decimals=6,
        token1_decimals=18,
        fee_tier=500,
        volume_24h_usd=Decimal("50000000"),
        tvl_usd=Decimal("80000000"),
        relevance_score=Decimal("1.0"),
    ),
    PoolSelectionCandidate(
        pool_address="0xd0b53D9277642d899DF5C87A3966A349A798F224",
        pair=("USDC", "cbBTC"),
        token0_address=USDC_BASE,
        token1_address=CBBTC_BASE,
        token0_decimals=6,
        token1_decimals=8,
        fee_tier=3000,
        volume_24h_usd=Decimal("15000000"),
        tvl_usd=Decimal("20000000"),
        relevance_score=Decimal("0.9"),
    ),
    PoolSelectionCandidate(
        pool_address="0x7858E59e0C01EA06Df3aF3D20aC7B0003275D4Bf",
        pair=("ETH", "cbBTC"),
        token0_address=WETH_BASE,
        token1_address=CBBTC_BASE,
        token0_decimals=18,
        token1_decimals=8,
        fee_tier=3000,
        volume_24h_usd=Decimal("10000000"),
        tvl_usd=Decimal("15000000"),
        relevance_score=Decimal("0.85"),
    ),
    PoolSelectionCandidate(
        pool_address="0x6E229C972d9F69c15Bdc7B07f385D2025225E72b",
        pair=("USDC", "DAI"),
        token0_address=USDC_BASE,
        token1_address=DAI_BASE,
        token0_decimals=6,
        token1_decimals=18,
        fee_tier=100,
        volume_24h_usd=Decimal("5000000"),
        tvl_usd=Decimal("10000000"),
        relevance_score=Decimal("0.7"),
    ),
    PoolSelectionCandidate(
        pool_address="0x0FB597D6cFe5687e3B3d2fAC2e3c3F3C924e8B7E",
        pair=("ETH", "USDT"),
        token0_address=WETH_BASE,
        token1_address=USDT_BASE,
        token0_decimals=18,
        token1_decimals=6,
        fee_tier=500,
        volume_24h_usd=Decimal("8000000"),
        tvl_usd=Decimal("12000000"),
        relevance_score=Decimal("0.75"),
    ),
]

# ---------------------------------------------------------------------------
# Uniswap V3 tick math helpers
# ---------------------------------------------------------------------------

# Minimum and maximum ticks for Uniswap V3
_MIN_TICK = -887272
_MAX_TICK = 887272

# Tick spacing per fee tier
_TICK_SPACING: dict[int, int] = {
    100: 1,
    500: 10,
    3000: 60,
    10000: 200,
}


def _align_tick(tick: int, spacing: int, round_down: bool) -> int:
    """Align a tick to the nearest valid tick spacing boundary."""
    if round_down:
        return (tick // spacing) * spacing
    return math.ceil(tick / spacing) * spacing


# ---------------------------------------------------------------------------
# Pool Selector
# ---------------------------------------------------------------------------


class PoolSelector:
    """
    Selects optimal Uniswap V3 pools for phantom liquidity deployment.

    Phase 1: returns from a static curated list, sorted by relevance_score.
    Phase 2 (future): dynamic selection via on-chain/API queries + Evo weighting.
    """

    def __init__(
        self,
        max_pools: int = 10,
        min_tvl_usd: Decimal = Decimal("100000"),
    ) -> None:
        self._max_pools = max_pools
        self._min_tvl_usd = min_tvl_usd
        self._logger = logger.bind(system="phantom_liquidity", component="pool_selector")

    def select_pools(
        self,
        budget_usd: Decimal,
        existing_pools: list[PhantomLiquidityPool],
        capital_per_pool_usd: Decimal = Decimal("100"),
    ) -> list[PoolSelectionCandidate]:
        """
        Return ranked list of pools to deploy into.

        Filters out pools where we already have active positions and pools
        below the minimum TVL threshold.  Distributes budget uniformly
        across selected pools, capped by ``capital_per_pool_usd``.
        """
        existing_addresses = {
            p.pool_address.lower()
            for p in existing_pools
            if p.health not in ("withdrawn", "failed")
        }

        candidates = [
            c for c in _STATIC_POOLS
            if c.pool_address.lower() not in existing_addresses
            and c.tvl_usd >= self._min_tvl_usd
        ]

        # Sort by relevance_score descending
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)

        # Cap to max_pools and budget
        max_positions = min(
            self._max_pools - len(existing_addresses),
            int(budget_usd / capital_per_pool_usd) if capital_per_pool_usd > 0 else 0,
        )

        selected = candidates[:max(max_positions, 0)]
        for c in selected:
            c.selected = True

        self._logger.info(
            "pools_selected",
            candidates=len(candidates),
            selected=len(selected),
            budget_usd=str(budget_usd),
        )

        return selected

    @staticmethod
    def compute_tick_range(
        fee_tier: int,
        current_tick: int = 0,
    ) -> tuple[int, int]:
        """
        Compute a wide tick range for a sensor position.

        Sensor positions use deliberately wide ranges to minimise impermanent
        loss.  The range covers roughly +/- 80% of the current price for
        low-fee pools (100, 500) and +/- 50% for higher-fee pools (3000, 10000).

        Parameters
        ----------
        fee_tier:
            Uniswap V3 fee tier (100, 500, 3000, 10000).
        current_tick:
            Current pool tick (from ``slot0``).  Defaults to 0 (full range).

        Returns
        -------
        tuple[int, int]
            (tick_lower, tick_upper) aligned to the pool's tick spacing.
        """
        spacing = _TICK_SPACING.get(fee_tier, 60)

        if current_tick == 0:
            # Full range — maximise price coverage, minimise IL
            tick_lower = _align_tick(_MIN_TICK, spacing, round_down=True)
            tick_upper = _align_tick(_MAX_TICK, spacing, round_down=False)
            return (tick_lower, tick_upper)

        # Compute range width based on fee tier:
        # Low fee (100, 500) → ~80% of price → ~5800 ticks each side
        # High fee (3000, 10000) → ~50% of price → ~4000 ticks each side
        if fee_tier <= 500:
            half_range = 5800  # log(1.80) / log(1.0001) ≈ 5878
        else:
            half_range = 4000  # log(1.50) / log(1.0001) ≈ 4055

        tick_lower = _align_tick(current_tick - half_range, spacing, round_down=True)
        tick_upper = _align_tick(current_tick + half_range, spacing, round_down=False)

        # Clamp to protocol bounds
        tick_lower = max(tick_lower, _MIN_TICK)
        tick_upper = min(tick_upper, _MAX_TICK)

        return (tick_lower, tick_upper)
