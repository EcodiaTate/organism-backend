"""
EcodiaOS — Redis Client

Async Redis for ephemeral state: workspace buffers, active goals,
affect state cache, rate limiting, and pub/sub.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import orjson
import structlog
from redis.asyncio import Redis

if TYPE_CHECKING:
    from config import RedisConfig

logger = structlog.get_logger()


class RedisClient:
    """
    Async Redis client with key prefixing for multi-instance support.
    """

    def __init__(self, config: RedisConfig) -> None:
        self._config = config
        self._client: Redis | None = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        self._client = Redis.from_url(
            self._config.full_url,
            decode_responses=True,
        )
        # Verify connectivity
        await self._client.ping()
        logger.info("redis_connected", prefix=self._config.prefix)

    async def close(self) -> None:
        """Close the connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("redis_disconnected")

    @property
    def client(self) -> Redis:
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    def _key(self, key: str) -> str:
        """Prefix a key with the instance prefix."""
        return f"{self._config.prefix}:{key}"

    async def health_check(self) -> dict[str, Any]:
        """Check connectivity."""
        try:
            t0 = time.monotonic()
            await self.client.ping()
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            return {"status": "connected", "latency_ms": latency_ms}
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            return {"status": "disconnected", "error": str(e)}

    # ─── JSON Helpers ─────────────────────────────────────────────

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a JSON-serialisable value."""
        raw = orjson.dumps(value).decode()
        if ttl:
            await self.client.setex(self._key(key), ttl, raw)
        else:
            await self.client.set(self._key(key), raw)

    async def get_json(self, key: str) -> Any | None:
        """Retrieve a JSON value."""
        raw = await self.client.get(self._key(key))
        if raw is None:
            return None
        return orjson.loads(raw)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        await self.client.delete(self._key(key))

    # ─── List Operations (Workspace Buffer) ───────────────────────

    async def push(self, key: str, value: Any) -> None:
        """Push a JSON value to a list."""
        raw = orjson.dumps(value).decode()
        await self.client.rpush(self._key(key), raw)  # type: ignore[misc]

    async def pop_all(self, key: str) -> list[Any]:
        """Pop all items from a list atomically."""
        pipe = self.client.pipeline()
        k = self._key(key)
        pipe.lrange(k, 0, -1)
        pipe.delete(k)
        results = await pipe.execute()
        # results[0] is the lrange output; guard against an empty or malformed
        # response (e.g. connection drop mid-pipeline returns an empty list).
        if not results or not results[0]:
            return []
        return [orjson.loads(item) for item in results[0]]

    async def list_length(self, key: str) -> int:
        """Get list length."""
        return await self.client.llen(self._key(key))  # type: ignore[misc, no-any-return]

    # ─── Hash Operations (Active Goals, Conversations) ────────────

    async def hset(self, key: str, field: str, value: Any) -> None:
        """Set a hash field."""
        raw = orjson.dumps(value).decode()
        await self.client.hset(self._key(key), field, raw)  # type: ignore[misc]

    async def hget(self, key: str, field: str) -> Any | None:
        """Get a hash field."""
        raw = await self.client.hget(self._key(key), field)  # type: ignore[misc]
        if raw is None:
            return None
        return orjson.loads(raw)

    async def hgetall(self, key: str) -> dict[str, Any]:
        """Get all hash fields."""
        raw = await self.client.hgetall(self._key(key))  # type: ignore[misc]
        return {k: orjson.loads(v) for k, v in raw.items()}

    async def hdel(self, key: str, field: str) -> None:
        """Delete a hash field."""
        await self.client.hdel(self._key(key), field)  # type: ignore[misc]

    # ─── Pub/Sub ──────────────────────────────────────────────────

    async def publish(self, channel: str, message: Any) -> None:
        """Publish a message to a channel."""
        raw = orjson.dumps(message).decode()
        await self.client.publish(self._key(f"channel:{channel}"), raw)
