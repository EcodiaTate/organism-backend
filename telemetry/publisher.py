"""
EcodiaOS — Metrics Publisher

Background loop that reads metrics dicts from an asyncio.Queue and
publishes them to a Redis pub/sub channel for SSE consumers.

Extracted from main.py to avoid a circular import with core/registry.py.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger()

METRICS_CHANNEL = "ecodiaos:system:metrics"


async def publish_metrics_loop(
    redis_client: RedisClient,
    metrics_queue: asyncio.Queue[dict],
) -> None:
    """
    Continuously reads dicts from *metrics_queue* and publishes them
    to the Redis metrics channel.  Runs until cancelled.
    """
    while True:
        try:
            payload: dict = await metrics_queue.get()
            await redis_client.client.publish(METRICS_CHANNEL, json.dumps(payload))
            metrics_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("metrics_publish_error")
