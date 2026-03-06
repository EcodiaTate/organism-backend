"""
EcodiaOS — TimescaleDB Client

Async connection management for telemetry, metrics, and audit logs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import asyncpg
import structlog

if TYPE_CHECKING:
    from config import TimescaleDBConfig

logger = structlog.get_logger()

# Table DDL — works on any Postgres instance.
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    time        TIMESTAMPTZ NOT NULL,
    system      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    labels      JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_metrics_system_metric ON metrics (system, metric, time DESC);

CREATE TABLE IF NOT EXISTS audit_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    system      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    intent_id   UUID,
    details     JSONB NOT NULL,
    affect      JSONB,
    checksum    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_system ON audit_log (system, time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log (event_type, time DESC);

CREATE TABLE IF NOT EXISTS affect_history (
    time              TIMESTAMPTZ NOT NULL,
    valence           DOUBLE PRECISION,
    arousal           DOUBLE PRECISION,
    dominance         DOUBLE PRECISION,
    curiosity         DOUBLE PRECISION,
    care_activation   DOUBLE PRECISION,
    coherence_stress  DOUBLE PRECISION,
    source_event      TEXT
);

CREATE TABLE IF NOT EXISTS cycle_log (
    time            TIMESTAMPTZ NOT NULL,
    cycle_number    BIGINT NOT NULL,
    period_ms       INTEGER,
    actual_ms       INTEGER,
    broadcast       BOOLEAN,
    salience_max    DOUBLE PRECISION,
    systems_acked   INTEGER
);

CREATE TABLE IF NOT EXISTS phantom_price_history (
    time         TIMESTAMPTZ NOT NULL,
    pair         TEXT NOT NULL,
    price        DOUBLE PRECISION NOT NULL,
    source       TEXT NOT NULL DEFAULT 'phantom_liquidity',
    pool_address TEXT NOT NULL DEFAULT '',
    block_number BIGINT NOT NULL DEFAULT 0,
    latency_ms   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_phantom_price_pair ON phantom_price_history (pair, time DESC);
"""

# Hypertable conversions — requires the timescaledb extension.
HYPERTABLE_TABLES = ["metrics", "audit_log", "affect_history", "cycle_log", "phantom_price_history"]


class TimescaleDBClient:
    """
    Async TimescaleDB client with connection pooling.
    Handles metrics, audit logs, and affect history.
    """

    def __init__(self, config: TimescaleDBConfig) -> None:
        self._config = config
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool and initialise schema."""
        self._pool = await asyncpg.create_pool(
            dsn=self._config.dsn,
            min_size=2,
            max_size=self._config.pool_size,
            ssl="require" if self._config.ssl else None,
        )
        logger.info(
            "timescaledb_connected",
            host=self._config.host,
            database=self._config.database,
        )
        # Initialise schema
        await self._init_schema()

    async def _init_schema(self) -> None:
        """Create tables (plain Postgres) then promote to hypertables if the extension exists."""
        async with self.pool.acquire() as conn:
            # 1. Create tables + indexes — pure Postgres, always succeeds.
            for statement in TABLE_SQL.split(";"):
                stmt = statement.strip()
                if stmt:
                    await conn.execute(stmt)

            # 2. Check whether TimescaleDB is available before attempting hypertables.
            has_timescaledb = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
            )

            if has_timescaledb:
                for table in HYPERTABLE_TABLES:
                    try:
                        await conn.execute(
                            f"SELECT create_hypertable('{table}', 'time', if_not_exists => TRUE)"
                        )
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning("tsdb_hypertable_warning", table=table, error=str(e))
                logger.info("timescaledb_schema_initialised", hypertables=True)
            else:
                logger.info(
                    "timescaledb_schema_initialised",
                    hypertables=False,
                    note="timescaledb extension not available — tables created as plain Postgres",
                )


    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("timescaledb_disconnected")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("TimescaleDB client not connected. Call connect() first.")
        return self._pool

    async def health_check(self) -> dict[str, Any]:
        """Check connectivity."""
        try:
            t0 = time.monotonic()
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            return {"status": "connected", "latency_ms": latency_ms}
        except Exception as e:
            logger.error("tsdb_health_check_failed", error=str(e))
            return {"status": "disconnected", "error": str(e)}

    async def write_metrics(self, points: list[dict[str, Any]]) -> None:
        """Batch write metric points."""
        if not points:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO metrics (time, system, metric, value, labels)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                [
                    (p["time"], p["system"], p["metric"], p["value"],
                     str(p.get("labels", {})).replace("'", '"'))
                    for p in points
                ],
            )

    async def write_phantom_price(self, feed: dict[str, Any]) -> None:
        """Persist a single phantom price observation to phantom_price_history."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO phantom_price_history
                    (time, pair, price, source, pool_address, block_number, latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                feed["time"],
                feed["pair"],
                float(feed["price"]),
                feed.get("source", "phantom_liquidity"),
                feed.get("pool_address", ""),
                feed.get("block_number", 0),
                feed.get("latency_ms", 0),
            )

    async def get_phantom_price_history(
        self,
        pair: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return recent price history for a pair, newest-first."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, pair, price, source, pool_address, block_number, latency_ms
                FROM phantom_price_history
                WHERE pair = $1
                ORDER BY time DESC
                LIMIT $2
                """,
                pair,
                limit,
            )
        return [dict(r) for r in rows]

    async def write_affect(self, state: dict[str, Any]) -> None:
        """Write a single affect state snapshot."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO affect_history
                    (time, valence, arousal, dominance, curiosity,
                     care_activation, coherence_stress, source_event)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                state.get("time"),
                state.get("valence", 0.0),
                state.get("arousal", 0.0),
                state.get("dominance", 0.0),
                state.get("curiosity", 0.0),
                state.get("care_activation", 0.0),
                state.get("coherence_stress", 0.0),
                state.get("source_event", ""),
            )
