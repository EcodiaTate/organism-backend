"""
EcodiaOS — Oikos Snapshot Writer (Revenue Timeseries Persistence)

Persists EconomicState snapshots to a Redis-backed ring buffer every 5 minutes
so the dashboard can render net worth, burn rate, and runway trends over time.

Architecture:
  - SnapshotWriter subscribes to a periodic asyncio task
  - Every SNAPSHOT_INTERVAL_SECONDS, it reads oikos.snapshot() and appends to Redis
  - Redis stores up to MAX_SNAPSHOTS entries as a JSON list (RPUSH/LTRIM pattern)
  - get_history(days) slices the ring buffer to the requested window

Thread-safety: NOT thread-safe. Designed for single-threaded asyncio event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.oikos.service import OikosService

logger = structlog.get_logger("oikos.snapshot_writer")

SNAPSHOT_INTERVAL_SECONDS: int = 300  # 5 minutes
MAX_SNAPSHOTS: int = 2016  # 7 days x 288 snapshots/day (every 5 min)
REDIS_KEY: str = "ecodiaos:oikos:snapshot_ring"


class SnapshotWriter:
    """
    Background task that persists OikosService snapshots to a Redis ring buffer.

    Usage:
        writer = SnapshotWriter(oikos=oikos_service, redis=redis_client)
        await writer.start()          # begin background loop
        await writer.stop()           # graceful shutdown
        history = await writer.get_history(days=7)
    """

    def __init__(self, oikos: OikosService, redis: RedisClient) -> None:
        self._oikos = oikos
        self._redis = redis
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._logger = logger.bind(component="snapshot_writer")

    async def start(self) -> None:
        """Start the background snapshot loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="oikos_snapshot_writer")
        self._logger.info("snapshot_writer_started", interval_s=SNAPSHOT_INTERVAL_SECONDS)

    async def stop(self) -> None:
        """Gracefully stop the background loop."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._logger.info("snapshot_writer_stopped")

    async def _loop(self) -> None:
        """Run snapshot every SNAPSHOT_INTERVAL_SECONDS."""
        while self._running:
            try:
                await self._write_snapshot()
            except Exception as exc:
                self._logger.warning("snapshot_write_failed", error=str(exc))
            await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)

    async def _write_snapshot(self) -> None:
        """Take a snapshot of the current economic state and append to the ring buffer."""
        state = self._oikos.snapshot()
        now = datetime.now(UTC)

        row: dict[str, str] = {
            "timestamp": now.isoformat(),
            "net_worth_usd": str(state.total_net_worth),
            "liquid_balance": str(state.liquid_balance),
            "burn_rate_usd_per_day": str(state.current_burn_rate.usd_per_day),
            "runway_days": str(state.runway_days),
            "revenue_24h": str(state.revenue_24h),
            "costs_24h": str(state.costs_24h),
            "net_income_24h": str(state.net_income_24h),
            "starvation_level": state.starvation_level.value,
        }

        # Append to right, trim to MAX_SNAPSHOTS
        try:
            existing_raw = await self._redis.get_json(REDIS_KEY)
            ring: list[dict[str, str]] = existing_raw if isinstance(existing_raw, list) else []
            ring.append(row)
            if len(ring) > MAX_SNAPSHOTS:
                ring = ring[-MAX_SNAPSHOTS:]
            await self._redis.set_json(REDIS_KEY, ring)
            self._logger.debug(
                "snapshot_written",
                ring_size=len(ring),
                net_worth=row["net_worth_usd"],
            )
        except Exception as exc:
            self._logger.warning("snapshot_redis_write_failed", error=str(exc))

    async def get_history(self, days: int = 7) -> list[dict[str, Any]]:
        """
        Return snapshots from the last N days, oldest first.

        Filters by timestamp so callers always get a clean window even if
        the ring buffer spans multiple day ranges.
        """
        try:
            raw = await self._redis.get_json(REDIS_KEY)
        except Exception as exc:
            self._logger.warning("snapshot_redis_read_failed", error=str(exc))
            return []

        if not isinstance(raw, list):
            return []

        cutoff = datetime.now(UTC) - timedelta(days=days)

        result: list[dict[str, Any]] = []
        for row in raw:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                if ts >= cutoff:
                    result.append(row)
            except Exception:
                continue

        return result
