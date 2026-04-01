"""
EcodiaOS - Metric Collection

Central metric collector. Systems report metrics here.
Writes are batched and flushed to TimescaleDB periodically.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.timescaledb import TimescaleDBClient

logger = structlog.get_logger()


class MetricCollector:
    """
    Central metric collection service.

    Systems call record() to report metrics.
    The collector batches writes and flushes to TimescaleDB
    either when the buffer is full or on a timer.
    """

    def __init__(
        self,
        tsdb: TimescaleDBClient,
        flush_interval_ms: int = 1000,
        batch_size: int = 100,
    ) -> None:
        self._tsdb = tsdb
        self._flush_interval = flush_interval_ms / 1000.0
        self._batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def record(
        self,
        system: str,
        metric: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a metric data point."""
        self._buffer.append({
            "time": datetime.now(UTC),
            "system": system,
            "metric": metric,
            "value": value,
            "labels": labels or {},
        })

        if len(self._buffer) >= self._batch_size:
            await self.flush()

    async def flush(self) -> None:
        """Flush the buffer to TimescaleDB."""
        if not self._buffer:
            return

        batch = self._buffer[:]
        self._buffer.clear()

        try:
            await self._tsdb.write_metrics(batch)
        except Exception as e:
            logger.error("metric_flush_failed", error=str(e), batch_size=len(batch))
            # Put items back in buffer for retry (with size limit)
            self._buffer = batch[:self._batch_size] + self._buffer

    async def start_writer(self) -> None:
        """Start the periodic flush task."""
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.info("metric_writer_started", interval_ms=int(self._flush_interval * 1000))

    async def _flush_loop(self) -> None:
        """Background loop that flushes periodically."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self.flush()

    async def stop(self) -> None:
        """Stop the writer and flush remaining metrics."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.flush()
        logger.info("metric_writer_stopped")
