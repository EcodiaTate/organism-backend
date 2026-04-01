"""
EcodiaOS - Oikos Snapshot Writer (Revenue Timeseries Persistence)

Persists EconomicState snapshots to a Redis-backed ring buffer every 5 minutes
so the dashboard can render net worth, burn rate, and runway trends over time.

Also writes each snapshot to TimescaleDB (table: oikos_economic_state, hypertable
on recorded_at) for long-term time-series queries. TimescaleDB writes are
non-fatal - a failure logs and continues.

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
    import asyncpg

    from clients.redis import RedisClient
    from systems.oikos.service import OikosService

logger = structlog.get_logger("oikos.snapshot_writer")

SNAPSHOT_INTERVAL_SECONDS: int = 300  # 5 minutes
MAX_SNAPSHOTS: int = 2016  # 7 days x 288 snapshots/day (every 5 min)
REDIS_KEY: str = "ecodiaos:oikos:snapshot_ring"

# CostSnapshot written to Neo4j every hour for trend analysis
_COST_SNAPSHOT_INTERVAL_S: int = 3600  # 1 hour


class SnapshotWriter:
    """
    Background task that persists OikosService snapshots to a Redis ring buffer
    and to TimescaleDB for long-term time-series queries.

    Usage:
        writer = SnapshotWriter(oikos=oikos_service, redis=redis_client)
        writer.set_timescale(pool)    # optional - enables TimescaleDB writes
        await writer.start()          # begin background loop
        await writer.stop()           # graceful shutdown
        history = await writer.get_history(days=7)
    """

    def __init__(self, oikos: OikosService, redis: RedisClient) -> None:
        self._oikos = oikos
        self._redis = redis
        self._timescale: asyncpg.Pool | None = None
        self._neo4j: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._logger = logger.bind(component="snapshot_writer")
        # Track time of last hourly cost snapshot to Neo4j
        self._last_cost_snapshot_time: float = 0.0

    def set_timescale(self, pool: asyncpg.Pool) -> None:
        """Inject an asyncpg connection pool for TimescaleDB writes."""
        self._timescale = pool
        self._logger.info("timescale_pool_wired")

    def set_neo4j(self, neo4j: Any) -> None:
        """Inject Neo4j client for hourly CostSnapshot writes."""
        self._neo4j = neo4j
        self._logger.info("neo4j_cost_snapshot_wired")

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

        # TimescaleDB persistence - non-fatal
        await self._write_timescale(state, now)

        # Hourly Neo4j CostSnapshot - enables trend analysis (costs increasing 5%/wk?)
        import time as _time
        elapsed_since_cost_snap = _time.monotonic() - self._last_cost_snapshot_time
        if elapsed_since_cost_snap >= _COST_SNAPSHOT_INTERVAL_S:
            await self._write_cost_snapshot_neo4j(state, now)
            self._last_cost_snapshot_time = _time.monotonic()

    async def _write_timescale(self, state: Any, now: datetime) -> None:
        """
        Write the current EconomicState snapshot to TimescaleDB.

        Table: oikos_economic_state (hypertable on recorded_at)
        Schema:
          recorded_at TIMESTAMPTZ, instance_id TEXT, balance_usdc NUMERIC,
          burn_rate NUMERIC, metabolic_efficiency NUMERIC, runway_days NUMERIC

        Non-fatal: any error is logged and swallowed.
        """
        if self._timescale is None:
            return

        try:
            instance_id: str = getattr(self._oikos, "_instance_id", "unknown")
            async with self._timescale.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO oikos_economic_state
                        (recorded_at, instance_id, balance_usdc, burn_rate,
                         metabolic_efficiency, runway_days)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    now,
                    instance_id,
                    float(state.liquid_balance),
                    float(state.current_burn_rate.usd_per_day),
                    float(state.metabolic_efficiency),
                    float(state.runway_days),
                )
            self._logger.debug(
                "timescale_snapshot_written",
                instance_id=instance_id,
                balance_usdc=str(state.liquid_balance),
            )
        except Exception as exc:
            self._logger.warning("timescale_snapshot_write_failed", error=str(exc))

    async def _write_cost_snapshot_neo4j(self, state: Any, now: datetime) -> None:
        """
        Write a (:CostSnapshot) node to Neo4j for hourly cost trend analysis.

        Schema:
          (:CostSnapshot {
              snapshot_id, timestamp, instance_id,
              api_cost_usd_per_hour, infra_cost_usd_per_hour, total_burn_usd_per_hour,
              balance_usd, runway_hours, dependency_ratio
          })

        Enables trend queries: "are costs increasing by 5% per week?"
        Non-fatal - any failure is logged and swallowed.
        """
        if self._neo4j is None:
            return

        with contextlib.suppress(Exception):
            instance_id: str = getattr(self._oikos, "_instance_id", "unknown")
            oikos_service = self._oikos

            # Pull two-ledger fields from oikos service state
            api_burn = float(getattr(oikos_service, "_api_burn_rate_usd_per_hour", 0.0))
            infra_burn = float(getattr(oikos_service, "_infra_burn_rate_usd_per_hour", 0.0))
            total_burn = api_burn + infra_burn
            balance = float(state.liquid_balance)
            runway_hours = balance / total_burn if total_burn > 0 else float("inf")
            dependency_ratio = infra_burn / total_burn if total_burn > 0 else 0.0

            from primitives.common import new_id
            snapshot_id = new_id()

            query = """
            CREATE (:CostSnapshot {
                snapshot_id: $snapshot_id,
                timestamp: $timestamp,
                instance_id: $instance_id,
                api_cost_usd_per_hour: $api_cost_usd_per_hour,
                infra_cost_usd_per_hour: $infra_cost_usd_per_hour,
                total_burn_usd_per_hour: $total_burn_usd_per_hour,
                balance_usd: $balance_usd,
                runway_hours: $runway_hours,
                dependency_ratio: $dependency_ratio
            })
            """
            props = {
                "snapshot_id": snapshot_id,
                "timestamp": now.isoformat(),
                "instance_id": instance_id,
                "api_cost_usd_per_hour": round(api_burn, 6),
                "infra_cost_usd_per_hour": round(infra_burn, 6),
                "total_burn_usd_per_hour": round(total_burn, 6),
                "balance_usd": round(balance, 4),
                "runway_hours": round(runway_hours, 2) if runway_hours != float("inf") else -1.0,
                "dependency_ratio": round(dependency_ratio, 4),
            }
            await self._neo4j.execute_write(query, props)
            self._logger.debug(
                "cost_snapshot_neo4j_written",
                snapshot_id=snapshot_id,
                total_burn_usd_hr=round(total_burn, 4),
                runway_hours=props["runway_hours"],
                dependency_ratio=round(dependency_ratio, 4),
            )

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
