"""
EcodiaOS — Synapse Event Bus

Dual-output event publication: in-memory callbacks for internal coordination
plus Redis pub/sub for the Alive WebSocket layer.

High-frequency events (CYCLE_COMPLETED at ~6.7Hz) skip in-memory callbacks
by default to avoid overwhelming listeners — they go to Redis only.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger("systems.synapse.event_bus")

# Callback signature: async def handler(event: SynapseEvent) -> None
EventCallback = Callable[[SynapseEvent], Coroutine[Any, Any, None]]

# Events that fire every cycle — too noisy for in-memory callbacks
_HIGH_FREQUENCY_EVENTS: frozenset[SynapseEventType] = frozenset({
    SynapseEventType.CYCLE_COMPLETED,
})

# Maximum time a callback gets before we log a warning and move on
_CALLBACK_TIMEOUT_S: float = 0.1

# Maximum recent events to keep in the ring buffer per event type
_RECENT_BUFFER_SIZE: int = 100

# Redis channel for Synapse events
_REDIS_CHANNEL: str = "synapse_events"


class EventBus:
    """
    Synapse inter-system event bus.

    Provides two delivery mechanisms:
    1. In-memory async callbacks — for internal system coordination (low latency)
    2. Redis pub/sub — for external consumers (Alive WebSocket, monitoring)

    High-frequency events (CYCLE_COMPLETED) only go to Redis to avoid
    callback overhead on every theta tick.
    """

    def __init__(self, redis: RedisClient | None = None) -> None:
        self._redis = redis
        self._logger = logger.bind(component="event_bus")

        # Per-type callback registrations
        self._subscribers: dict[SynapseEventType, list[EventCallback]] = defaultdict(list)
        # Catch-all subscribers (receive every event)
        self._global_subscribers: list[EventCallback] = []

        # Ring buffers for recent event history
        self._recent: dict[SynapseEventType, deque[SynapseEvent]] = defaultdict(
            lambda: deque(maxlen=_RECENT_BUFFER_SIZE)
        )

        # Metrics
        self._total_emitted: int = 0
        self._total_redis_failures: int = 0
        self._total_callback_timeouts: int = 0

    # ─── Subscription ────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: SynapseEventType,
        callback: EventCallback,
    ) -> None:
        """Register a callback for a specific event type."""
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        """Register a callback that receives every event (except high-frequency)."""
        self._global_subscribers.append(callback)

    # ─── Emission ────────────────────────────────────────────────────

    async def emit(self, event: SynapseEvent) -> None:
        """
        Publish an event to all registered listeners.

        In-memory callbacks fire first (with timeout protection),
        then Redis publication (fire-and-forget, failure-tolerant).

        High-frequency events skip in-memory callbacks entirely.
        """
        self._total_emitted += 1
        self._recent[event.event_type].append(event)

        is_high_freq = event.event_type in _HIGH_FREQUENCY_EVENTS

        # ── In-memory callbacks (skip for high-frequency events) ──
        if not is_high_freq:
            callbacks = list(self._subscribers.get(event.event_type, []))
            callbacks.extend(self._global_subscribers)

            if callbacks:
                await self._dispatch_callbacks(callbacks, event)

        # ── Redis publication ──
        await self._publish_redis(event)

    async def _dispatch_callbacks(
        self,
        callbacks: list[EventCallback],
        event: SynapseEvent,
    ) -> None:
        """Dispatch event to callbacks with per-callback timeout protection."""
        for callback in callbacks:
            try:
                await asyncio.wait_for(
                    callback(event),
                    timeout=_CALLBACK_TIMEOUT_S,
                )
            except TimeoutError:
                self._total_callback_timeouts += 1
                self._logger.warning(
                    "event_callback_timeout",
                    event_type=event.event_type.value,
                    callback=getattr(callback, "__name__", str(callback)),
                )
            except Exception as exc:
                self._logger.error(
                    "event_callback_error",
                    event_type=event.event_type.value,
                    error=str(exc),
                )

    async def _publish_redis(self, event: SynapseEvent) -> None:
        """Publish event to Redis. Failure is logged but never blocks."""
        if self._redis is None:
            return

        try:
            payload = {
                "id": event.id,
                "type": event.event_type.value,
                "ts": event.timestamp.isoformat(),
                "data": event.data,
                "source": event.source_system,
            }
            await self._redis.publish(_REDIS_CHANNEL, payload)
        except Exception as exc:
            self._total_redis_failures += 1
            # Log at debug for high-frequency events, warning otherwise
            if event.event_type in _HIGH_FREQUENCY_EVENTS:
                self._logger.debug("redis_publish_failed", error=str(exc))
            else:
                self._logger.warning(
                    "redis_publish_failed",
                    event_type=event.event_type.value,
                    error=str(exc),
                )

    # ─── Query ───────────────────────────────────────────────────────

    def recent(
        self,
        event_type: SynapseEventType,
        limit: int = 10,
    ) -> list[SynapseEvent]:
        """Return recent events of a given type (most recent first)."""
        buf = self._recent.get(event_type)
        if not buf:
            return []
        items = list(buf)
        items.reverse()
        return items[:limit]

    # ─── Stats ───────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_emitted": self._total_emitted,
            "redis_failures": self._total_redis_failures,
            "callback_timeouts": self._total_callback_timeouts,
            "subscriber_count": sum(
                len(v) for v in self._subscribers.values()
            ) + len(self._global_subscribers),
            "recent_buffer_sizes": {
                et.value: len(buf) for et, buf in self._recent.items()
            },
        }
