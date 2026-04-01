"""
EcodiaOS - Prompt Caching Layer

Semantic cache for LLM prompts using Redis.
Deduplicates repeated evaluations to reduce token spend.

Key strategy: SHA256(system + method + prompt) → deterministic
TTL: configurable per system (Nova 5min, Voxis 1min, Evo 1hour)

Cache hits avoid LLM calls entirely; misses fall through to normal path.
Hit rate target: >30% in active conversations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json as _json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger()

# Type variable for generic cache values
T = TypeVar('T')


class CacheStrategy(StrEnum):
    """Cache configuration per system."""
    AGGRESSIVE = "aggressive"  # Short TTL (1min), high hit rate expected
    NORMAL = "normal"          # Medium TTL (5min), balanced
    CONSERVATIVE = "conservative"  # Long TTL (1hour), low churn


@dataclass
class CacheEntry[T]:
    """A cached value with metadata."""
    value: T
    timestamp: float
    ttl_seconds: int
    system: str
    method: str
    hit_count: int = 0


class PromptCache:
    """
    Redis-backed semantic cache for LLM responses.

    Stores (prompt_hash → response, metadata) with configurable TTL.
    Tracks hit rate for observability.
    """

    def __init__(self, redis_client: Any, prefix: str = "eos:cache") -> None:
        """
        Args:
            redis_client: aioredis client (async)
            prefix: Redis key namespace
        """
        self._redis = redis_client
        self._prefix = prefix
        self._hit_count = 0
        self._miss_count = 0
        self._logger = logger.bind(component="prompt_cache")

    def _compute_key(
        self,
        system: str,
        method: str,
        prompt: str,
    ) -> str:
        """Compute cache key from system, method, prompt."""
        digest = hashlib.sha256(
            f"{system}:{method}:{prompt}".encode()
        ).hexdigest()
        return f"{self._prefix}:{system}:{method}:{digest}"

    async def get(
        self,
        system: str,
        method: str,
        prompt: str,
    ) -> Any | None:
        """
        Retrieve a cached response.

        Args:
            system: System name (e.g., 'nova', 'voxis')
            method: Method name (e.g., 'pragmatic_value', 'render')
            prompt: The full prompt text

        Returns:
            Cached value if found and valid, else None
        """
        key = self._compute_key(system, method, prompt)

        try:
            cached_bytes = await self._redis.get(key)
            if cached_bytes:
                self._hit_count += 1
                cached_data = _json.loads(cached_bytes)
                self._logger.debug(
                    "cache_hit",
                    system=system,
                    method=method,
                    hit_count=self._hit_count,
                )
                return cached_data.get("value")
        except Exception as exc:
            self._logger.warning(
                "cache_get_error",
                system=system,
                method=method,
                error=str(exc),
            )

        self._miss_count += 1
        return None

    async def set(
        self,
        system: str,
        method: str,
        prompt: str,
        value: Any,
        ttl_seconds: int = 300,
    ) -> None:
        """
        Store a response in the cache.

        Args:
            system: System name
            method: Method name
            prompt: The full prompt text
            value: The LLM response (typically string or parsed JSON)
            ttl_seconds: Time-to-live (300s = 5min default)
        """
        key = self._compute_key(system, method, prompt)

        try:
            cache_entry = {
                "value": value,
                "system": system,
                "method": method,
                "timestamp": asyncio.get_running_loop().time(),
            }
            await self._redis.setex(
                key,
                ttl_seconds,
                _json.dumps(cache_entry),
            )
            self._logger.debug(
                "cache_set",
                system=system,
                method=method,
                ttl_s=ttl_seconds,
            )
        except Exception as exc:
            self._logger.warning(
                "cache_set_error",
                system=system,
                method=method,
                error=str(exc),
            )

    async def clear_pattern(self, pattern: str = "*") -> int:
        """
        Clear cache entries matching a pattern.

        Uses SCAN (cursor-based) instead of KEYS to avoid O(N) blocking
        on large Redis keyspaces in production.

        Args:
            pattern: Redis glob pattern (e.g., "eos:cache:nova:*")

        Returns:
            Number of keys deleted
        """
        full_pattern = f"{self._prefix}:{pattern}"
        deleted = 0
        try:
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=full_pattern, count=100)
                if keys:
                    deleted += int(await self._redis.delete(*keys))
                if cursor == 0:
                    break
            if deleted:
                self._logger.info(
                    "cache_cleared",
                    pattern=pattern,
                    deleted_count=deleted,
                )
            return deleted
        except Exception as exc:
            self._logger.warning(
                "cache_clear_error",
                pattern=pattern,
                error=str(exc),
            )
            return 0

    async def clear_system(self, system: str) -> int:
        """Clear all cache entries for a system."""
        return await self.clear_pattern(f"{system}:*")

    def get_hit_rate(self) -> float:
        """Return cache hit rate (0.0–1.0)."""
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hit_count + self._miss_count
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "total_requests": total,
            "hit_rate": self.get_hit_rate(),
        }

    async def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hit_count = 0
        self._miss_count = 0


class TTLConfig:
    """TTL presets for each system."""

    # Nova systems - beliefs change fairly slowly
    NOVA_EFE_PRAGMATIC_S = 300      # 5 minutes
    NOVA_EFE_EPISTEMIC_S = 300
    NOVA_POLICY_S = 600              # 10 minutes

    # Voxis - personality/affect shift frequently in conversations
    VOXIS_EXPRESSION_S = 60           # 1 minute
    VOXIS_OUTLINE_S = 120             # 2 minutes

    # Evo - schema rarely changes mid-session
    EVO_HYPOTHESIS_S = 3600           # 1 hour
    EVO_INDUCTION_S = 3600

    # Thread - identity quite stable
    THREAD_SYNTHESIS_S = 21600        # 6 hours
    THREAD_COHERENCE_S = 21600

    # Equor - constitutional checks stable
    EQUOR_INVARIANT_S = 1800          # 30 minutes

    # Oneiros - sleep processing off-cycle
    ONEIROS_REFLECTION_S = 86400      # 24 hours
