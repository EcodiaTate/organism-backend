"""
EcodiaOS — Phantom Liquidity REST Router (Phase 16q)

Exposes the Phantom Liquidity Sensor Network to the Next.js frontend.

Endpoints:
  GET  /api/v1/phantom-liquidity/health          — System health & listener metrics
  GET  /api/v1/phantom-liquidity/pools           — All deployed LP positions
  GET  /api/v1/phantom-liquidity/prices          — Latest cached price feeds
  GET  /api/v1/phantom-liquidity/price           — Price for a specific pair (?pair=USDC/ETH)
  GET  /api/v1/phantom-liquidity/price-history   — Historical prices from TimescaleDB
  GET  /api/v1/phantom-liquidity/config          — Current configuration parameters
  GET  /api/v1/phantom-liquidity/candidates      — Pool selection candidates
  GET  /api/v1/phantom-liquidity/tick-range      — Tick range visualizer for a pool/pair
  POST /api/v1/phantom-liquidity/price-fetch     — Force oracle fallback fetch for a pair
  POST /api/v1/phantom-liquidity/deploy          — Deploy a phantom LP position
  POST /api/v1/phantom-liquidity/withdraw        — Withdraw a deployed phantom position
  GET  /api/v1/phantom-liquidity/defillama-pools — Dynamic pool discovery via DeFiLlama
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

logger = structlog.get_logger("api.phantom_liquidity")

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svc(request: Request) -> Any:
    """Retrieve the phantom liquidity service or raise 503."""
    svc = getattr(request.app.state, "phantom_liquidity", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="Phantom liquidity service not initialised (check config.phantom_liquidity.enabled)",
        )
    return svc


def _pool_to_dict(pool: Any) -> dict[str, Any]:
    return {
        "id": pool.id,
        "pool_address": pool.pool_address,
        "token_id": pool.token_id,
        "pair": list(pool.pair),
        "token0_address": pool.token0_address,
        "token1_address": pool.token1_address,
        "token0_decimals": pool.token0_decimals,
        "token1_decimals": pool.token1_decimals,
        "fee_tier": pool.fee_tier,
        "tick_lower": pool.tick_lower,
        "tick_upper": pool.tick_upper,
        "capital_deployed_usd": str(pool.capital_deployed_usd),
        "amount0_deployed": pool.amount0_deployed,
        "amount1_deployed": pool.amount1_deployed,
        "last_price_observed": str(pool.last_price_observed),
        "last_price_timestamp": pool.last_price_timestamp.isoformat(),
        "price_update_count": pool.price_update_count,
        "cumulative_yield_usd": str(pool.cumulative_yield_usd),
        "impermanent_loss_pct": str(pool.impermanent_loss_pct),
        "health": pool.health.value if hasattr(pool.health, "value") else str(pool.health),
        "deployed_at": pool.deployed_at.isoformat() if pool.deployed_at else None,
        "deploy_tx_hash": pool.deploy_tx_hash,
        "withdraw_tx_hash": pool.withdraw_tx_hash,
        "yield_position_id": pool.yield_position_id,
    }


def _feed_to_dict(feed: Any) -> dict[str, Any]:
    return {
        "id": feed.id,
        "pool_address": feed.pool_address,
        "pair": list(feed.pair),
        "price": str(feed.price),
        "sqrt_price_x96": feed.sqrt_price_x96,
        "timestamp": feed.timestamp.isoformat(),
        "block_number": feed.block_number,
        "tx_hash": feed.tx_hash,
        "source": feed.source,
        "latency_ms": feed.latency_ms,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/phantom-liquidity/health")
async def get_health(request: Request) -> dict[str, Any]:
    """Full system health report including listener metrics."""
    svc = _svc(request)
    h = await svc.health()
    return {"status": "ok", "data": h}


@router.get("/api/v1/phantom-liquidity/pools")
async def get_pools(request: Request) -> dict[str, Any]:
    """Return all tracked phantom LP positions."""
    svc = _svc(request)
    pools = svc.get_pools()
    return {
        "status": "ok",
        "data": [_pool_to_dict(p) for p in pools],
        "total": len(pools),
    }


@router.get("/api/v1/phantom-liquidity/prices")
async def get_prices(request: Request) -> dict[str, Any]:
    """Return the latest price feed for every monitored pair."""
    svc = _svc(request)
    feeds = svc.get_all_prices()
    return {
        "status": "ok",
        "data": [_feed_to_dict(f) for f in feeds],
        "total": len(feeds),
    }


@router.get("/api/v1/phantom-liquidity/price")
async def get_price(
    request: Request,
    pair: str = Query(..., description="Trading pair, e.g. USDC/ETH"),
) -> dict[str, Any]:
    """Get the latest cached price for a specific pair."""
    svc = _svc(request)
    parts = pair.split("/")
    if len(parts) != 2:  # noqa: PLR2004
        raise HTTPException(status_code=400, detail="pair must be in 'TOKEN0/TOKEN1' format")

    token0, token1 = parts[0].strip().upper(), parts[1].strip().upper()
    feed = svc.get_price((token0, token1))
    if feed is None:
        return {"status": "ok", "data": None, "stale": True}

    return {"status": "ok", "data": _feed_to_dict(feed), "stale": False}


@router.get("/api/v1/phantom-liquidity/config")
async def get_config(request: Request) -> dict[str, Any]:
    """Return current phantom liquidity configuration parameters."""
    svc = _svc(request)
    cfg = svc._config  # noqa: SLF001  — intentional introspection for dashboard
    return {
        "status": "ok",
        "data": {
            "enabled": cfg.enabled,
            "rpc_url_set": bool(cfg.rpc_url),
            "max_total_capital_usd": cfg.max_total_capital_usd,
            "default_capital_per_pool_usd": cfg.default_capital_per_pool_usd,
            "min_capital_per_pool_usd": cfg.min_capital_per_pool_usd,
            "max_capital_per_pool_usd": cfg.max_capital_per_pool_usd,
            "max_pools": cfg.max_pools,
            "swap_poll_interval_s": cfg.swap_poll_interval_s,
            "staleness_threshold_s": cfg.staleness_threshold_s,
            "il_rebalance_threshold": cfg.il_rebalance_threshold,
            "capital_drift_threshold": cfg.capital_drift_threshold,
            "maintenance_interval_s": cfg.maintenance_interval_s,
            "oracle_fallback_enabled": cfg.oracle_fallback_enabled,
        },
    }


@router.get("/api/v1/phantom-liquidity/candidates")
async def get_candidates(request: Request) -> dict[str, Any]:
    """
    Return pool selection candidates from the static pool registry.

    Uses current pools as the 'existing' list to show which are already
    deployed vs available for selection.
    """
    svc = _svc(request)

    existing = svc.get_pools()
    cfg = svc._config  # noqa: SLF001
    candidates = svc._pool_selector.select_pools(  # noqa: SLF001
        budget_usd=Decimal(str(cfg.max_total_capital_usd)),
        existing_pools=existing,
        capital_per_pool_usd=Decimal(str(cfg.default_capital_per_pool_usd)),
    )
    return {
        "status": "ok",
        "data": [
            {
                "pool_address": c.pool_address,
                "pair": list(c.pair),
                "fee_tier": c.fee_tier,
                "volume_24h_usd": str(c.volume_24h_usd),
                "tvl_usd": str(c.tvl_usd),
                "relevance_score": str(c.relevance_score),
                "selected": c.selected,
            }
            for c in candidates
        ],
        "total": len(candidates),
    }


class PriceFetchRequest(BaseModel):
    pair: str  # e.g. "USDC/ETH"


@router.post("/api/v1/phantom-liquidity/price-fetch")
async def force_price_fetch(request: Request, body: PriceFetchRequest) -> dict[str, Any]:
    """
    Force an oracle fallback fetch for a pair (bypasses phantom pool cache).

    Useful for testing the CoinGecko fallback chain and for bootstrapping
    price data before phantom positions have received swap events.
    """
    svc = _svc(request)
    parts = body.pair.split("/")
    if len(parts) != 2:  # noqa: PLR2004
        raise HTTPException(status_code=400, detail="pair must be in 'TOKEN0/TOKEN1' format")

    token0, token1 = parts[0].strip().upper(), parts[1].strip().upper()

    try:
        feed = await svc.get_price_with_fallback((token0, token1))
    except Exception as exc:  # noqa: BLE001
        logger.warning("price_fetch_error", pair=body.pair, error=str(exc))
        return {"status": "error", "error": str(exc), "data": None}

    return {
        "status": "ok",
        "data": _feed_to_dict(feed) if feed else None,
        "stale": feed is None,
    }


# ---------------------------------------------------------------------------
# Price history (TimescaleDB)
# ---------------------------------------------------------------------------


@router.get("/api/v1/phantom-liquidity/price-history")
async def get_price_history(
    request: Request,
    pair: str = Query(..., description="Trading pair, e.g. USDC/ETH"),
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    """Return persisted price history for a pair from TimescaleDB (oldest-first for charting)."""
    svc = _svc(request)
    parts = pair.split("/")
    if len(parts) != 2:  # noqa: PLR2004
        raise HTTPException(status_code=400, detail="pair must be in 'TOKEN0/TOKEN1' format")

    canonical = f"{parts[0].strip().upper()}/{parts[1].strip().upper()}"
    rows = await svc.get_price_history(canonical, limit=limit)

    serialised = []
    for r in rows:
        serialised.append({
            "time": r["time"].isoformat() if hasattr(r["time"], "isoformat") else str(r["time"]),
            "pair": r["pair"],
            "price": r["price"],
            "source": r["source"],
            "pool_address": r["pool_address"],
            "block_number": r["block_number"],
            "latency_ms": r["latency_ms"],
        })

    serialised.reverse()  # oldest-first for charting

    return {
        "status": "ok",
        "data": serialised,
        "pair": canonical,
        "total": len(serialised),
    }


# ---------------------------------------------------------------------------
# Tick range visualizer
# ---------------------------------------------------------------------------


def _tick_to_price(tick: int, token0_decimals: int, token1_decimals: int) -> float:
    """Convert a Uniswap V3 tick to a human-readable price."""
    return (1.0001 ** tick) * (10 ** (token0_decimals - token1_decimals))


@router.get("/api/v1/phantom-liquidity/tick-range")
async def get_tick_range(
    request: Request,
    pool_address: str | None = Query(None),
    pair: str | None = Query(None, description="e.g. USDC/ETH"),
    fee_tier: int = Query(3000),
    current_tick: int = Query(0, description="Current pool tick (0 = full range)"),
) -> dict[str, Any]:
    """Tick range + price band data for the pool position visualizer."""
    svc = _svc(request)

    token0_decimals = 6
    token1_decimals = 18
    pair_label = pair or "UNKNOWN"

    if pool_address:
        pool = svc._pools.get(pool_address.lower())  # noqa: SLF001
        if pool:
            token0_decimals = pool.token0_decimals
            token1_decimals = pool.token1_decimals
            pair_label = f"{pool.pair[0]}/{pool.pair[1]}"
            fee_tier = pool.fee_tier
            if current_tick == 0 and pool.last_price_observed > 0:
                price_raw = float(pool.last_price_observed) / (
                    10 ** (token0_decimals - token1_decimals)
                )
                if price_raw > 0:
                    current_tick = int(math.log(price_raw) / math.log(1.0001))

    from systems.phantom_liquidity.pool_selector import PoolSelector

    tick_lower, tick_upper = PoolSelector.compute_tick_range(fee_tier, current_tick)

    steps = 20
    step_size = max((tick_upper - tick_lower) // steps, 1)
    ticks = [tick_lower + i * step_size for i in range(steps + 1)]

    return {
        "status": "ok",
        "data": {
            "pair": pair_label,
            "fee_tier": fee_tier,
            "current_tick": current_tick,
            "tick_lower": tick_lower,
            "tick_upper": tick_upper,
            "price_lower": _tick_to_price(tick_lower, token0_decimals, token1_decimals),
            "price_upper": _tick_to_price(tick_upper, token0_decimals, token1_decimals),
            "price_current": (
                _tick_to_price(current_tick, token0_decimals, token1_decimals)
                if current_tick != 0
                else None
            ),
            "token0_decimals": token0_decimals,
            "token1_decimals": token1_decimals,
            "tick_ladder": [
                {
                    "tick": t,
                    "price": _tick_to_price(t, token0_decimals, token1_decimals),
                }
                for t in ticks
            ],
        },
    }


# ---------------------------------------------------------------------------
# Position deployment
# ---------------------------------------------------------------------------


class DeployRequest(BaseModel):
    pool_address: str
    capital_usd: float = 100.0


@router.post("/api/v1/phantom-liquidity/deploy")
async def deploy_position(request: Request, body: DeployRequest) -> dict[str, Any]:
    """
    Deploy a phantom LP position into a Uniswap V3 pool on Base L2.

    Mints a concentrated liquidity position via NonfungiblePositionManager.
    """
    svc = _svc(request)

    wallet = getattr(request.app.state, "wallet", None)
    if wallet is None:
        raise HTTPException(status_code=503, detail="Wallet client not initialised")

    from systems.phantom_liquidity.pool_selector import (
        _STATIC_POOLS as STATIC_POOL_REGISTRY,  # noqa: PLC2701
    )
    from systems.phantom_liquidity.pool_selector import PoolSelector
    from systems.phantom_liquidity.types import PhantomLiquidityPool, PoolHealth

    candidate = next(
        (c for c in STATIC_POOL_REGISTRY if c.pool_address.lower() == body.pool_address.lower()),
        None,
    )
    if candidate is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pool {body.pool_address} not in registry. Use /candidates to list available pools.",
        )

    if body.pool_address.lower() in svc._pools:  # noqa: SLF001
        existing = svc._pools[body.pool_address.lower()]  # noqa: SLF001
        if existing.health not in (PoolHealth.WITHDRAWN, PoolHealth.FAILED):
            raise HTTPException(status_code=409, detail="Position already deployed for this pool")

    tick_lower, tick_upper = PoolSelector.compute_tick_range(candidate.fee_tier)

    from decimal import Decimal as _Decimal
    pool = PhantomLiquidityPool(
        pool_address=candidate.pool_address,
        pair=candidate.pair,
        token0_address=candidate.token0_address,
        token1_address=candidate.token1_address,
        token0_decimals=candidate.token0_decimals,
        token1_decimals=candidate.token1_decimals,
        fee_tier=candidate.fee_tier,
        tick_lower=tick_lower,
        tick_upper=tick_upper,
        capital_deployed_usd=_Decimal(str(body.capital_usd)),
        health=PoolHealth.PENDING_DEPLOY,
    )

    try:
        from systems.phantom_liquidity.executor import PhantomLiquidityExecutor

        executor = PhantomLiquidityExecutor(wallet=wallet)
        result = await executor.mint_position(pool)
        pool.token_id = result["token_id"]
        pool.deploy_tx_hash = result["tx_hash"]
        pool.amount0_deployed = result.get("amount0", 0)
        pool.amount1_deployed = result.get("amount1", 0)
        pool.health = PoolHealth.ACTIVE

        from primitives.common import utc_now
        pool.deployed_at = utc_now()

    except Exception as exc:
        pool.health = PoolHealth.FAILED
        logger.warning("phantom_deploy_failed", pool=body.pool_address, error=str(exc))
        return {"status": "error", "error": str(exc), "data": _pool_to_dict(pool)}

    svc.register_pool(pool)
    logger.info(
        "phantom_position_deployed",
        pool=pool.pool_address,
        pair=pool.pair,
        token_id=pool.token_id,
        tx_hash=pool.deploy_tx_hash,
    )
    return {"status": "ok", "data": _pool_to_dict(pool)}


# ---------------------------------------------------------------------------
# Withdrawal
# ---------------------------------------------------------------------------


class WithdrawRequest(BaseModel):
    pool_address: str


@router.post("/api/v1/phantom-liquidity/withdraw")
async def withdraw_position(request: Request, body: WithdrawRequest) -> dict[str, Any]:
    """
    Withdraw (burn) a phantom LP position from a Uniswap V3 pool.

    Burns the NFT, collects fees and remaining liquidity, then unregisters.
    """
    svc = _svc(request)

    wallet = getattr(request.app.state, "wallet", None)
    if wallet is None:
        raise HTTPException(status_code=503, detail="Wallet client not initialised")

    pool = svc._pools.get(body.pool_address.lower())  # noqa: SLF001
    if pool is None:
        raise HTTPException(status_code=404, detail="Pool not registered")

    from systems.phantom_liquidity.types import PoolHealth

    if pool.health == PoolHealth.WITHDRAWN:
        raise HTTPException(status_code=409, detail="Position already withdrawn")

    if pool.token_id == 0:
        raise HTTPException(
            status_code=409,
            detail="Position has no token ID — may not have been minted on-chain",
        )

    try:
        from systems.phantom_liquidity.executor import PhantomLiquidityExecutor

        executor = PhantomLiquidityExecutor(wallet=wallet)
        result = await executor.burn_position(pool)
        pool.withdraw_tx_hash = result["tx_hash"]
        pool.health = PoolHealth.WITHDRAWN

    except Exception as exc:
        logger.warning("phantom_withdraw_failed", pool=body.pool_address, error=str(exc))
        return {"status": "error", "error": str(exc), "data": _pool_to_dict(pool)}

    svc.unregister_pool(body.pool_address)
    logger.info(
        "phantom_position_withdrawn",
        pool=pool.pool_address,
        token_id=pool.token_id,
        tx_hash=pool.withdraw_tx_hash,
    )
    return {"status": "ok", "data": _pool_to_dict(pool)}


# ---------------------------------------------------------------------------
# DeFiLlama dynamic pool discovery
# ---------------------------------------------------------------------------


@router.get("/api/v1/phantom-liquidity/defillama-pools")
async def get_defillama_pools(
    request: Request,
    chain: str = Query("Base", description="Chain name as DeFiLlama uses it"),
    protocol: str = Query("uniswap-v3", description="Protocol slug"),
    min_tvl_usd: float = Query(100_000, description="Minimum TVL in USD"),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """
    Fetch live pool data from the DeFiLlama Yields API.

    Falls back to the static registry if DeFiLlama is unreachable.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://yields.llama.fi/pools")
            resp.raise_for_status()
            raw = resp.json()

        chain_lower = chain.lower()
        proto_lower = protocol.lower()

        filtered = [
            p for p in raw.get("data", [])
            if p.get("chain", "").lower() == chain_lower
            and p.get("project", "").lower() == proto_lower
            and float(p.get("tvlUsd", 0)) >= min_tvl_usd
        ]

        filtered.sort(key=lambda p: float(p.get("tvlUsd", 0)), reverse=True)
        filtered = filtered[:limit]

        result = []
        for p in filtered:
            symbol = p.get("symbol", "")
            parts = symbol.split("-") if "-" in symbol else symbol.split("/")
            pair = [parts[0].upper(), parts[1].upper()] if len(parts) >= 2 else [symbol, ""]  # noqa: PLR2004
            result.append({
                "pool_id": p.get("pool", ""),
                "pair": pair,
                "symbol": symbol,
                "tvl_usd": float(p.get("tvlUsd", 0)),
                "apy": float(p.get("apy", 0)),
                "apy_base": float(p.get("apyBase", 0) or 0),
                "volume_7d_usd": float(p.get("volumeUsd7d", 0) or 0),
                "il_risk": p.get("ilRisk", "unknown"),
                "stable_coin": p.get("stablecoin", False),
                "chain": p.get("chain", ""),
                "project": p.get("project", ""),
            })

        return {
            "status": "ok",
            "data": result,
            "total": len(result),
            "chain": chain,
            "protocol": protocol,
            "source": "defillama",
        }

    except Exception as exc:
        logger.warning("defillama_fetch_failed", error=str(exc))

        from systems.phantom_liquidity.pool_selector import _STATIC_POOLS as _SP  # noqa: PLC2701

        fallback = [
            {
                "pool_id": c.pool_address,
                "pair": list(c.pair),
                "symbol": f"{c.pair[0]}-{c.pair[1]}",
                "tvl_usd": float(c.tvl_usd),
                "apy": 0.0,
                "apy_base": 0.0,
                "volume_7d_usd": float(c.volume_24h_usd) * 7,
                "il_risk": "unknown",
                "stable_coin": False,
                "chain": "Base",
                "project": "uniswap-v3",
            }
            for c in _SP
        ]

        return {
            "status": "ok",
            "data": fallback,
            "total": len(fallback),
            "chain": chain,
            "protocol": protocol,
            "source": "static_fallback",
            "error": str(exc),
        }
