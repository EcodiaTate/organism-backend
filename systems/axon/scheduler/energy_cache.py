"""
EcodiaOS - Energy Grid Cache

Redis-backed cache for grid carbon intensity readings. Sits between the
EnergyProvider clients and the scheduler interceptor so we never hit the
upstream API more than once per TTL window, regardless of how many
execution requests flow through the pipeline.

Two storage concerns:
  1. **Grid reading cache** - the latest GridReading, stored as a Redis
     JSON key with a configurable TTL (default 10 minutes). On cache miss,
     the provider is called and the result is stored.
  2. **Deferred task queue** - high-compute tasks parked until the grid
     recovers. Stored as a Redis list (FIFO) so multiple Axon instances
     share the same queue.

Falls back to in-memory storage if Redis is unavailable.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.scheduler.types import (
    DeferralStatus,
    DeferredTask,
    GridReading,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.axon.scheduler.energy_client import EnergyProvider

logger = structlog.get_logger("systems.axon.scheduler.energy_cache")

# Redis key names
_GRID_READING_KEY = "axon:scheduler:grid_reading"
_DEFERRED_QUEUE_KEY = "axon:scheduler:deferred_tasks"


class EnergyCache:
    """
    Caching layer for grid energy readings and the deferred task queue.

    Usage:
        cache = EnergyCache(provider=client, redis=redis_client, ttl_s=600)
        reading = await cache.get_reading()  # cached or fresh
        await cache.defer_task(task)
        released = await cache.release_ready_tasks(threshold_g=150.0)
    """

    def __init__(
        self,
        provider: EnergyProvider,
        redis: RedisClient | None = None,
        ttl_s: int = 600,
    ) -> None:
        self._provider = provider
        self._redis = redis
        self._ttl_s = ttl_s
        self._log = logger.bind(
            component="energy_cache",
            provider=provider.provider_name,
        )

        # In-memory fallback when Redis is unavailable
        self._mem_reading: GridReading | None = None
        self._mem_reading_ts: float = 0.0
        self._mem_deferred: deque[DeferredTask] = deque(maxlen=500)

    # ─── Grid Reading ─────────────────────────────────────────────

    async def get_reading(self) -> GridReading | None:
        """
        Return the latest grid reading (from cache or fresh fetch).

        Priority:
          1. Redis cache (if not expired)
          2. Fresh fetch from provider → store in Redis
          3. In-memory fallback if Redis is down
          4. None if the provider also fails
        """
        # Try Redis first
        cached = await self._get_cached_reading()
        if cached is not None:
            return cached

        # Cache miss - fetch from provider
        reading = await self._provider.fetch()
        if reading is None:
            # Provider failed - return stale in-memory reading if we have one
            if self._mem_reading is not None and not self._mem_reading.is_stale:
                self._log.debug("using_stale_memory_reading")
                return self._mem_reading
            return None

        # Store in both Redis and memory
        await self._store_reading(reading)
        return reading

    async def _get_cached_reading(self) -> GridReading | None:
        """Try to load a cached reading from Redis."""
        if self._redis is None:
            return self._get_memory_reading()

        try:
            data = await self._redis.get_json(_GRID_READING_KEY)
            if data is not None:
                self._log.debug("cache_hit")
                return GridReading(**data)
        except Exception as exc:
            self._log.debug("redis_get_failed", error=str(exc))

        return self._get_memory_reading()

    def _get_memory_reading(self) -> GridReading | None:
        """Return the in-memory cached reading if still within TTL."""
        if self._mem_reading is None:
            return None
        age = time.monotonic() - self._mem_reading_ts
        if age > self._ttl_s:
            return None
        return self._mem_reading

    async def _store_reading(self, reading: GridReading) -> None:
        """Store a reading in Redis (with TTL) and in-memory."""
        self._mem_reading = reading
        self._mem_reading_ts = time.monotonic()

        if self._redis is None:
            return

        try:
            await self._redis.set_json(
                _GRID_READING_KEY,
                reading.model_dump(mode="json"),
                ttl=self._ttl_s,
            )
            self._log.debug("cache_stored", ttl_s=self._ttl_s)
        except Exception as exc:
            self._log.debug("redis_set_failed", error=str(exc))

    # ─── Deferred Task Queue ──────────────────────────────────────

    async def defer_task(self, task: DeferredTask) -> None:
        """Push a task onto the deferred sleep-cycle queue."""
        if self._redis is not None:
            try:
                await self._redis.push(
                    _DEFERRED_QUEUE_KEY,
                    task.model_dump(mode="json"),
                )
                self._log.info(
                    "task_deferred",
                    task_id=task.id,
                    intent_id=task.intent_id,
                    carbon_g=task.carbon_intensity_at_deferral,
                )
                return
            except Exception as exc:
                self._log.warning("redis_defer_failed", error=str(exc))

        # Fallback to in-memory
        self._mem_deferred.append(task)
        self._log.info(
            "task_deferred_memory",
            task_id=task.id,
            queue_size=len(self._mem_deferred),
        )

    async def release_ready_tasks(
        self, carbon_threshold_g: float
    ) -> tuple[list[DeferredTask], int]:
        """
        Drain deferred tasks that are safe to execute now.

        A task is released if:
          - Current grid carbon intensity is below the threshold, OR
          - The task has expired (exceeded max_defer_seconds)

        Expired tasks are marked as EXPIRED and dropped.
        Returns (released_tasks, expired_count).
        """
        reading = await self.get_reading()
        current_carbon = reading.carbon_intensity_g if reading else None

        all_tasks = await self._drain_deferred_queue()
        released: list[DeferredTask] = []
        still_deferred: list[DeferredTask] = []
        expired_count: int = 0

        from primitives.common import utc_now as _utc_now
        for task in all_tasks:
            if task.is_expired:
                task.status = DeferralStatus.EXPIRED
                expired_count += 1
                age_s = int((_utc_now() - task.deferred_at).total_seconds())
                self._log.warning(
                    "deferred_task_expired",
                    task_id=task.id,
                    intent_id=task.intent_id,
                    deferred_s=age_s,
                )
                continue  # Drop expired tasks

            if current_carbon is not None and current_carbon < carbon_threshold_g:
                task.status = DeferralStatus.RELEASED
                task.release_carbon_intensity = current_carbon
                task.released_at = _utc_now()
                released.append(task)
                self._log.info(
                    "deferred_task_released",
                    task_id=task.id,
                    intent_id=task.intent_id,
                    carbon_g=current_carbon,
                    threshold_g=carbon_threshold_g,
                )
            else:
                still_deferred.append(task)

        # Re-queue tasks that are still deferred
        for task in still_deferred:
            await self.defer_task(task)

        return released, expired_count

    async def _drain_deferred_queue(self) -> list[DeferredTask]:
        """Atomically drain all tasks from the deferred queue."""
        if self._redis is not None:
            try:
                raw_items = await self._redis.pop_all(_DEFERRED_QUEUE_KEY)
                return [DeferredTask(**item) for item in raw_items]
            except Exception as exc:
                self._log.warning("redis_drain_failed", error=str(exc))

        # Fallback: drain in-memory queue
        tasks = list(self._mem_deferred)
        self._mem_deferred.clear()
        return tasks

    async def deferred_count(self) -> int:
        """Return the number of tasks currently in the deferred queue."""
        if self._redis is not None:
            try:
                return await self._redis.list_length(_DEFERRED_QUEUE_KEY)
            except Exception:
                pass
        return len(self._mem_deferred)

    # ─── Diagnostics ──────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        """Return cache diagnostics."""
        reading = self._mem_reading
        return {
            "provider": self._provider.provider_name,
            "ttl_s": self._ttl_s,
            "has_cached_reading": reading is not None,
            "cached_carbon_g": round(reading.carbon_intensity_g, 1)
            if reading
            else None,
            "cached_zone": reading.zone if reading else None,
            "deferred_queue_size": await self.deferred_count(),
            "redis_available": self._redis is not None,
        }
