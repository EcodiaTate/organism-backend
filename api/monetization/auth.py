"""
EcodiaOS - Tollbooth Authentication & Rate Limiting

FastAPI dependencies that:
1. Extract the API key from the X-Tollbooth-Key header
2. Validate the key exists in the ledger
3. Enforce a sliding-window rate limit per key
4. Are composed into endpoint-level Depends() chains

The balance check (402 gating) happens *inside* each endpoint so the
cost can vary per product - this module only verifies identity + rate.
"""

from __future__ import annotations

import time
from typing import Annotated

import structlog
from fastapi import Depends, Header, HTTPException, Request

from api.monetization.ledger import (
    CreditLedger,  # noqa: TC001 - used at runtime in Depends()
)

logger = structlog.get_logger("api.monetization.auth")

# Header name for external API keys (distinct from the internal X-EOS-API-Key)
_HEADER = "X-Tollbooth-Key"

# Rate limit: requests per window
_RATE_LIMIT = 60
_RATE_WINDOW_SECONDS = 60


# ─── Dependencies ────────────────────────────────────────────────


def _get_ledger(request: Request) -> CreditLedger:
    """Resolve the CreditLedger from app state (set during lifespan)."""
    ledger: CreditLedger | None = getattr(request.app.state, "tollbooth_ledger", None)
    if ledger is None:
        raise HTTPException(status_code=503, detail="Tollbooth ledger unavailable.")
    return ledger


async def _extract_api_key(
    x_tollbooth_key: Annotated[str, Header(alias=_HEADER)],
    ledger: Annotated[CreditLedger, Depends(_get_ledger)],
) -> str:
    """Validate the API key header and return the key string."""
    if not x_tollbooth_key:
        raise HTTPException(status_code=401, detail="Missing API key.")

    if not await ledger.key_exists(x_tollbooth_key):
        raise HTTPException(status_code=403, detail="Invalid API key.")

    return x_tollbooth_key


async def _enforce_rate_limit(
    request: Request,
    api_key: Annotated[str, Depends(_extract_api_key)],
) -> str:
    """
    Sliding-window rate limiter using a Redis sorted set.

    Each request inserts a timestamp score; we count entries within
    the window.  Expired entries are pruned on each call.
    """
    from redis.asyncio import Redis as AsyncRedis  # noqa: TC002 - runtime import for type narrowing

    redis: AsyncRedis = request.app.state.redis.client
    prefix: str = request.app.state.redis._config.prefix
    window_key = f"{prefix}:tollbooth:rate:{api_key}"

    now = time.time()
    window_start = now - _RATE_WINDOW_SECONDS

    # Phase 1: prune expired entries and count current window (read-only).
    # We check the limit before adding so rejected requests don't consume a slot.
    pipe = redis.pipeline()
    pipe.zremrangebyscore(window_key, 0, window_start)
    pipe.zcard(window_key)
    results = await pipe.execute()

    count: int = results[1]
    if count >= _RATE_LIMIT:
        logger.warning("tollbooth_rate_limited", api_key=api_key[:8], count=count)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({_RATE_LIMIT} req/{_RATE_WINDOW_SECONDS}s).",
            headers={"Retry-After": str(_RATE_WINDOW_SECONDS)},
        )

    # Phase 2: request is allowed - record it and refresh TTL.
    pipe2 = redis.pipeline()
    pipe2.zadd(window_key, {str(now): now})
    pipe2.expire(window_key, _RATE_WINDOW_SECONDS * 2)
    await pipe2.execute()

    return api_key


# ─── Public dependency alias ─────────────────────────────────────

# Endpoints declare: `api_key: Annotated[str, Depends(require_tollbooth_key)]`
# This chains: header extraction → key validation → rate limit check.
require_tollbooth_key = _enforce_rate_limit
