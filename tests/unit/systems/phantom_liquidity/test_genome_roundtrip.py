"""
Unit tests: Phantom Liquidity genome round-trip fidelity.

Verifies that GenomeExtractionProtocol.extract_genome_segment() serializes
and LiquidityPhantomService.seed_from_genome_segment() deserializes Phantom's
configuration without data loss.

No external dependencies required - runs fully in-process.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Minimal stub fixtures - no real infra needed
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> Any:
    """Return a minimal PhantomLiquidityConfig stub."""
    config = MagicMock()
    config.rpc_url = ""  # Disable listener so initialize() is a no-op.
    config.max_pools = 7
    config.staleness_threshold_s = 450.0
    config.il_rebalance_threshold = 0.03
    config.swap_poll_interval_s = 5.0
    config.oracle_fallback_enabled = True
    config.maintenance_interval_s = 3600
    config.oracle_fallback_url = "https://api.coingecko.com/api/v3"
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_pool(address: str, pair: tuple[str, str], fee_tier: int = 500) -> Any:
    """Return a minimal PhantomLiquidityPool stub."""
    from systems.phantom_liquidity.types import PhantomLiquidityPool, PoolHealth

    return PhantomLiquidityPool(
        pool_address=address,
        pair=pair,
        fee_tier=fee_tier,
        tick_lower=-887_200,
        tick_upper=887_200,
        token0_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        token1_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        token0_decimals=6,
        token1_decimals=18,
        capital_deployed_usd=Decimal("100"),
        health=PoolHealth.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_genome_roundtrip_config_fidelity() -> None:
    """
    extract_genome_segment → seed_from_genome_segment preserves all
    operational parameters without data loss.
    """
    from systems.phantom_liquidity.service import LiquidityPhantomService

    config = _make_config()
    service = LiquidityPhantomService(config=config, instance_id="test-instance")

    # Register a pool so pool_configs are included in the genome.
    pool = _make_pool("0x4C36388bE6F416A29C8d8Eee81C771cE6bE14B18", ("USDC", "ETH"))
    service._pools[pool.pool_address.lower()] = pool
    service._pair_to_pool["USDC/ETH"] = pool.pool_address.lower()

    # --- Phase 1: extract ---
    segment = await service.extract_genome_segment()

    assert segment.system_id is not None
    assert segment.payload is not None
    payload = segment.payload

    # Verify all config fields are present.
    assert payload["staleness_threshold_s"] == config.staleness_threshold_s
    assert payload["il_rebalance_threshold"] == config.il_rebalance_threshold
    assert payload["max_pools"] == config.max_pools
    assert payload["swap_poll_interval_s"] == config.swap_poll_interval_s
    assert payload["oracle_fallback_enabled"] == config.oracle_fallback_enabled

    # Verify pool config is captured.
    pool_configs = payload["pool_configs"]
    assert len(pool_configs) == 1
    pc = pool_configs[0]
    assert pc["pool_address"] == pool.pool_address
    assert pc["pair"] == list(pool.pair)
    assert pc["fee_tier"] == pool.fee_tier
    assert pc["tick_lower"] == pool.tick_lower
    assert pc["tick_upper"] == pool.tick_upper
    assert pc["token0_decimals"] == pool.token0_decimals
    assert pc["token1_decimals"] == pool.token1_decimals

    # --- Phase 2: verify payload hash ---
    expected_json = json.dumps(payload, sort_keys=True, default=str)
    expected_hash = hashlib.sha256(expected_json.encode()).hexdigest()
    assert segment.payload_hash == expected_hash, (
        "Payload hash mismatch - genome integrity check would fail"
    )

    # --- Phase 3: seed into a fresh service ---
    config2 = _make_config(
        staleness_threshold_s=999.0,  # Different from original
        il_rebalance_threshold=0.99,
        max_pools=1,
        swap_poll_interval_s=99.0,
    )
    service2 = LiquidityPhantomService(config=config2, instance_id="child-instance")

    success = await service2.seed_from_genome_segment(segment)
    assert success is True

    # After seeding, child should have parent's config values.
    assert service2._config.staleness_threshold_s == config.staleness_threshold_s
    assert service2._config.il_rebalance_threshold == config.il_rebalance_threshold
    assert service2._config.max_pools == config.max_pools
    assert service2._config.swap_poll_interval_s == config.swap_poll_interval_s


@pytest.mark.asyncio
async def test_genome_roundtrip_empty_pools() -> None:
    """
    Genome extraction works correctly with no active pools.
    pool_configs should be an empty list, not absent.
    """
    from systems.phantom_liquidity.service import LiquidityPhantomService

    config = _make_config()
    service = LiquidityPhantomService(config=config)

    segment = await service.extract_genome_segment()

    assert segment.payload["pool_configs"] == []
    assert segment.payload_hash is not None

    # Seed into another service - should succeed trivially.
    service2 = LiquidityPhantomService(config=_make_config(), instance_id="child")
    success = await service2.seed_from_genome_segment(segment)
    assert success is True


@pytest.mark.asyncio
async def test_genome_roundtrip_withdrawn_pools_excluded() -> None:
    """
    Withdrawn and FAILED pools are excluded from the genome.
    Only ACTIVE/STALE/etc. pools should be inherited by children.
    """
    from systems.phantom_liquidity.service import LiquidityPhantomService
    from systems.phantom_liquidity.types import PoolHealth

    config = _make_config()
    service = LiquidityPhantomService(config=config)

    active_pool = _make_pool("0xAAAA000000000000000000000000000000000001", ("USDC", "ETH"))
    withdrawn_pool = _make_pool("0xBBBB000000000000000000000000000000000002", ("ETH", "DAI"))
    withdrawn_pool.health = PoolHealth.WITHDRAWN
    failed_pool = _make_pool("0xCCCC000000000000000000000000000000000003", ("USDC", "DAI"))
    failed_pool.health = PoolHealth.FAILED

    service._pools[active_pool.pool_address.lower()] = active_pool
    service._pools[withdrawn_pool.pool_address.lower()] = withdrawn_pool
    service._pools[failed_pool.pool_address.lower()] = failed_pool

    segment = await service.extract_genome_segment()
    pool_configs = segment.payload["pool_configs"]

    # Only the active pool should appear.
    assert len(pool_configs) == 1
    assert pool_configs[0]["pool_address"] == active_pool.pool_address


@pytest.mark.asyncio
async def test_genome_roundtrip_payload_hash_changes_on_mutation() -> None:
    """
    Mutating any config field must produce a different payload hash.
    This verifies genome integrity detection works.
    """
    from systems.phantom_liquidity.service import LiquidityPhantomService

    config = _make_config()
    service = LiquidityPhantomService(config=config)

    segment_a = await service.extract_genome_segment()

    # Mutate one config value.
    service._config.staleness_threshold_s = 1234.0
    segment_b = await service.extract_genome_segment()

    assert segment_a.payload_hash != segment_b.payload_hash, (
        "Payload hash should differ after config mutation"
    )
