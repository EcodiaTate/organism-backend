"""
EcodiaOS - Synapse Event Bus

Dual-output event publication: in-memory callbacks for internal coordination
plus Redis pub/sub for the Alive WebSocket layer.

High-frequency events (CYCLE_COMPLETED at ~6.7Hz) skip in-memory callbacks
by default to avoid overwhelming listeners - they go to Redis only.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

import structlog

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger("systems.synapse.event_bus")

# Callback signature: async def handler(event: SynapseEvent) -> None
EventCallback = Callable[[SynapseEvent], Coroutine[Any, Any, None]]

# Events that were previously skipped for in-memory callbacks.
# CYCLE_COMPLETED fires at ~6.7 Hz. Keeping it here was preventing the
# EventTracer from observing it (observatory coverage gap). The set is now
# empty - the 100ms per-callback timeout is sufficient protection.
_HIGH_FREQUENCY_EVENTS: frozenset[SynapseEventType] = frozenset()

# Maximum time a callback gets before we log a warning and move on
_CALLBACK_TIMEOUT_S: float = 0.1

# Maximum recent events to keep in the ring buffer per event type
_RECENT_BUFFER_SIZE: int = 100

# Redis channel for Synapse events
_REDIS_CHANNEL: str = "synapse_events"

# Retry configuration for failed deliveries
_MAX_RETRIES: int = 3
_RETRY_BACKOFF_BASE_S: float = 0.05  # 50ms, 100ms, 200ms

# Dead-letter queue max size (FIFO eviction)
_DEAD_LETTER_MAX: int = 10_000

# Rate limiting defaults
_DEFAULT_RATE_LIMIT: float = 100.0  # events/second per system
_RATE_LIMIT_WARN_FRACTION: float = 0.8


class _TokenBucket:
    """O(1) token-bucket rate limiter."""

    __slots__ = ("_rate", "_capacity", "_tokens", "_last_refill")

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._capacity = rate  # 1-second burst
        self._tokens = rate
        self._last_refill = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0

    def try_consume(self) -> bool:
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    @property
    def fill_fraction(self) -> float:
        """How full the bucket is (0 = empty/rate-limited, 1 = full)."""
        return self._tokens / self._capacity if self._capacity > 0 else 0.0


# ─── Payload Validation Schemas ──────────────────────────────────────
# Gradual enforcement: log warning on validation failure, never reject.

class _CoherenceSnapshotPayload(BaseModel):
    system_resonance: float
    response_synchrony: float
    composite: float

class _MetabolicSnapshotPayload(BaseModel):
    rolling_deficit_usd: float
    burn_rate_usd_per_hour: float

class _CoherenceShiftPayload(BaseModel):
    direction: str
    shift: float
    composite: float

class _SystemFailedPayload(BaseModel):
    system_id: str

class _MetabolicPressurePayload(BaseModel):
    rolling_deficit_usd: float
    burn_rate_usd_per_hour: float

class _CycleCompletedPayload(BaseModel):
    cycle: int
    elapsed_ms: float

# Map event types to their expected payload schema
_PAYLOAD_SCHEMAS: dict[SynapseEventType, type[BaseModel]] = {
    SynapseEventType.COHERENCE_SNAPSHOT: _CoherenceSnapshotPayload,
    SynapseEventType.METABOLIC_SNAPSHOT: _MetabolicSnapshotPayload,
    SynapseEventType.COHERENCE_SHIFT: _CoherenceShiftPayload,
    SynapseEventType.SYSTEM_FAILED: _SystemFailedPayload,
    SynapseEventType.METABOLIC_PRESSURE: _MetabolicPressurePayload,
    SynapseEventType.CYCLE_COMPLETED: _CycleCompletedPayload,
}


class EventBus:
    """
    Synapse inter-system event bus.

    Provides two delivery mechanisms:
    1. In-memory async callbacks - for internal system coordination (low latency)
    2. Redis pub/sub - for external consumers (Alive WebSocket, monitoring)

    High-frequency events (CYCLE_COMPLETED) only go to Redis to avoid
    callback overhead on every theta tick.
    """

    def __init__(self, redis: RedisClient | None = None) -> None:
        self._redis = redis
        self._logger = logger.bind(component="event_bus")
        # Instance identity - set via set_instance_id() after construction.
        # When non-empty: stamped onto every emitted event and used to namespace
        # the Redis channel so Federation/Mitosis deployments don't cross-pollute.
        self._instance_id: str = ""
        self._redis_channel: str = _REDIS_CHANNEL

        # Per-type callback registrations - stored as (callback, timeout_s) pairs
        self._subscribers: dict[SynapseEventType, list[tuple[EventCallback, float]]] = defaultdict(list)
        # Catch-all subscribers (receive every event) - stored as (callback, timeout_s) pairs
        self._global_subscribers: list[tuple[EventCallback, float]] = []

        # Ring buffers for recent event history
        self._recent: dict[SynapseEventType, deque[SynapseEvent]] = defaultdict(
            lambda: deque(maxlen=_RECENT_BUFFER_SIZE)
        )

        # Metrics
        self._total_emitted: int = 0
        self._total_redis_failures: int = 0
        self._total_callback_timeouts: int = 0
        self._total_callback_failures: int = 0
        self._total_dead_lettered: int = 0

        # Per-event-type handler latency tracking for CoherenceMonitor
        # Cleared after each read via drain_handler_latencies()
        self._handler_latencies: list[list[float]] = []

        # Dead-letter queue: events that failed all retry attempts
        self._dead_letter_queue: deque[tuple[SynapseEvent, str]] = deque(maxlen=_DEAD_LETTER_MAX)

        # Per-system rate limiting (token bucket)
        self._rate_buckets: dict[str, _TokenBucket] = {}
        self._rate_limited_count: int = 0

    # ─── Subscription ────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: SynapseEventType,
        callback: EventCallback,
        timeout_s: float | None = None,
    ) -> None:
        """Register a callback for a specific event type.

        Args:
            event_type: The event type to listen for.
            callback: Async callable to invoke on delivery.
            timeout_s: Per-callback deadline in seconds.  Defaults to the
                global ``_CALLBACK_TIMEOUT_S`` (100 ms).  Pass a larger value
                for subscribers that legitimately need more time (e.g. Oikos
                metabolic handlers that touch Redis and emit secondary events).
        """
        effective_timeout = timeout_s if timeout_s is not None else _CALLBACK_TIMEOUT_S
        self._subscribers[event_type].append((callback, effective_timeout))

    def subscribe_all(self, callback: EventCallback, timeout_s: float | None = None) -> None:
        """Register a callback that receives every event (except high-frequency).

        Args:
            callback: Async callable to invoke on delivery.
            timeout_s: Per-callback deadline in seconds.  Defaults to the
                global ``_CALLBACK_TIMEOUT_S``.
        """
        effective_timeout = timeout_s if timeout_s is not None else _CALLBACK_TIMEOUT_S
        self._global_subscribers.append((callback, effective_timeout))

    # ─── Instance Identity ───────────────────────────────────────────

    def set_instance_id(self, instance_id: str) -> None:
        """
        Stamp all future events with this organism's instance identity and namespace
        the Redis channel to `synapse_events:{instance_id}` (Spec 09 §18 M4/SG4).

        In single-instance deployments this is a no-op (empty string keeps the
        default `synapse_events` channel).  In Federation/Mitosis deployments each
        child instance sets its own ID so pub/sub streams don't cross-pollute.

        Must be called before start_clock() to take effect on the first tick.
        """
        self._instance_id = instance_id
        if instance_id:
            self._redis_channel = f"{_REDIS_CHANNEL}:{instance_id}"
        else:
            self._redis_channel = _REDIS_CHANNEL
        self._logger.info(
            "event_bus_instance_id_set",
            instance_id=instance_id,
            redis_channel=self._redis_channel,
        )

    # ─── Emission ────────────────────────────────────────────────────

    def set_rate_limit(self, system_id: str, rate: float) -> None:
        """Configure per-system event rate limit (events/second)."""
        self._rate_buckets[system_id] = _TokenBucket(rate)

    async def emit(self, event: SynapseEvent) -> None:
        """
        Publish an event to all registered listeners.

        In-memory callbacks fire first (with timeout + retry protection),
        then Redis publication (fire-and-forget, failure-tolerant).

        High-frequency events skip in-memory callbacks entirely.
        Rate-limited events from a single source are dropped with a warning.
        """
        # ── Rate limiting per source system ──
        source = event.source_system
        if source:
            bucket = self._rate_buckets.get(source)
            if bucket is None:
                bucket = _TokenBucket(_DEFAULT_RATE_LIMIT)
                self._rate_buckets[source] = bucket
            if not bucket.try_consume():
                self._rate_limited_count += 1
                self._logger.warning(
                    "event_rate_limited",
                    source_system=source,
                    event_type=event.event_type.value,
                )
                # Emit rate-limited notification (avoid recursion by checking type)
                if event.event_type != SynapseEventType.SYSTEM_OVERLOADED:
                    await self._publish_redis(SynapseEvent(
                        event_type=SynapseEventType.SYSTEM_OVERLOADED,
                        data={
                            "reason": "rate_limited",
                            "source_system": source,
                            "event_type": event.event_type.value,
                        },
                        source_system="synapse",
                    ))
                return
            # Warn at 80% threshold
            if bucket.fill_fraction < (1.0 - _RATE_LIMIT_WARN_FRACTION):
                self._logger.debug(
                    "event_rate_limit_warning",
                    source_system=source,
                    fill_fraction=round(bucket.fill_fraction, 2),
                )

        # ── Payload validation (warn-only, never reject) ──
        schema = _PAYLOAD_SCHEMAS.get(event.event_type)
        if schema is not None:
            try:
                schema.model_validate(event.data)
            except ValidationError as ve:
                self._logger.warning(
                    "event_payload_validation_warning",
                    event_type=event.event_type.value,
                    errors=str(ve.errors()[:3]),
                )

        # ── Stamp instance identity (M4/SG4) ──
        if self._instance_id and not event.instance_id:
            event = event.model_copy(update={"instance_id": self._instance_id})

        self._total_emitted += 1
        self._recent[event.event_type].append(event)

        is_high_freq = event.event_type in _HIGH_FREQUENCY_EVENTS

        # ── In-memory callbacks (skip for high-frequency events) ──
        if not is_high_freq:
            entries = list(self._subscribers.get(event.event_type, []))
            entries.extend(self._global_subscribers)

            if entries:
                await self._dispatch_callbacks(entries, event)

        # ── Redis publication ──
        await self._publish_redis(event)

    async def _dispatch_callbacks(
        self,
        entries: list[tuple[EventCallback, float]],
        event: SynapseEvent,
    ) -> None:
        """Dispatch event to callbacks with per-callback timeout, retry, and latency tracking."""
        latencies: list[float] = []
        for callback, cb_timeout_s in entries:
            cb_name = getattr(callback, "__name__", str(callback))
            delivered = False
            last_error = ""
            for attempt in range(_MAX_RETRIES):
                t0 = time.perf_counter()
                try:
                    await asyncio.wait_for(
                        callback(event),
                        timeout=cb_timeout_s,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(elapsed_ms)
                    delivered = True
                    break
                except TimeoutError:
                    self._total_callback_timeouts += 1
                    last_error = "timeout"
                    self._logger.warning(
                        "event_callback_timeout",
                        event_type=event.event_type.value,
                        callback=cb_name,
                        attempt=attempt + 1,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    self._total_callback_failures += 1
                    self._logger.error(
                        "event_callback_error",
                        event_type=event.event_type.value,
                        callback=cb_name,
                        error=last_error,
                        attempt=attempt + 1,
                    )
                # Exponential backoff before retry
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_BACKOFF_BASE_S * (2 ** attempt))

            if not delivered:
                # Dead-letter after exhausting retries
                self._dead_letter_queue.append((event, f"callback={cb_name}: {last_error}"))
                self._total_dead_lettered += 1
                self._logger.error(
                    "event_dead_lettered",
                    event_type=event.event_type.value,
                    callback=cb_name,
                    reason=last_error,
                )
                # Emit delivery failure notification
                try:
                    await self._publish_redis(SynapseEvent(
                        event_type=SynapseEventType.SYSTEM_DEGRADED,
                        data={
                            "reason": "delivery_failed",
                            "event_type": event.event_type.value,
                            "callback": cb_name,
                            "error": last_error[:200],
                        },
                        source_system="synapse",
                    ))
                except Exception:
                    pass

        if latencies:
            self._handler_latencies.append(latencies)

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
            await self._redis.publish(self._redis_channel, payload)
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

    # ─── Latency / Dead-Letter / Replay ────────────────────────────────

    def drain_handler_latencies(self) -> list[list[float]]:
        """
        Return and clear accumulated per-dispatch handler latency sets.

        Each entry is a list of per-callback latencies (ms) from a single
        dispatch. CoherenceMonitor uses these for resonance/synchrony metrics.
        """
        result = self._handler_latencies
        self._handler_latencies = []
        return result

    @property
    def dead_letter_queue(self) -> list[tuple[SynapseEvent, str]]:
        """Return the current dead-letter queue contents (read-only snapshot)."""
        return list(self._dead_letter_queue)

    def replay_events(
        self,
        since: datetime | None = None,
        event_type: SynapseEventType | None = None,
        limit: int = 100,
    ) -> list[SynapseEvent]:
        """
        Replay recent events from the ring buffer for debugging.

        Args:
            since: Only return events after this timestamp.
            event_type: Filter to a specific event type. If None, searches all types.
            limit: Maximum number of events to return.
        """
        candidates: list[SynapseEvent] = []
        types_to_search = [event_type] if event_type else list(self._recent.keys())
        for et in types_to_search:
            buf = self._recent.get(et)
            if buf:
                candidates.extend(buf)
        if since is not None:
            candidates = [e for e in candidates if e.timestamp >= since]
        candidates.sort(key=lambda e: e.timestamp, reverse=True)
        return candidates[:limit]

    # ─── Stats ───────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_emitted": self._total_emitted,
            "redis_failures": self._total_redis_failures,
            "callback_timeouts": self._total_callback_timeouts,
            "callback_failures": self._total_callback_failures,
            "dead_lettered": self._total_dead_lettered,
            "rate_limited": self._rate_limited_count,
            "subscriber_count": sum(
                len(v) for v in self._subscribers.values()
            ) + len(self._global_subscribers),
            "recent_buffer_sizes": {
                et.value: len(buf) for et, buf in self._recent.items()
            },
        }
