"""
EcodiaOS — Skia Heartbeat Monitor

Observes the primary organism's Synapse heartbeat via Redis pub/sub.

Three-phase detection algorithm:

  Phase 1 — Observation (60s default):
    Subscribe to Redis synapse_events channel. Count consecutive missed
    heartbeats at the poll interval. If misses reach the threshold,
    transition to SUSPECTED_DEAD.

  Phase 2 — Suspicion:
    Transition state. Begin confirmation probes.

  Phase 3 — Confirmation (30s default):
    Run N active probes at intervals. Each probe checks:
      1. Is Redis itself alive? (PING)
      2. Did a new synapse_events message arrive?
    If Redis is unreachable → do NOT trigger restoration (it would fail).
    If ALL probes confirm no heartbeat → CONFIRMED_DEAD → trigger restoration.

Total minimum detection time: 90 seconds (observation + confirmation).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.skia.types import HeartbeatState, HeartbeatStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from redis.asyncio import Redis

    from config import SkiaConfig

logger = structlog.get_logger("systems.skia.heartbeat")

# Prefix used by RedisClient for the pub/sub channel
_REDIS_PREFIX = "eos"


class HeartbeatMonitor:
    """
    Observes the primary organism's Synapse heartbeat via Redis pub/sub.

    Can run embedded (for health reporting) or standalone (for actual monitoring).
    The ``on_death_confirmed`` callback is invoked exactly once per confirmed death.
    """

    def __init__(
        self,
        redis: Redis,
        config: SkiaConfig,
        on_death_confirmed: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._redis = redis
        self._config = config
        self._on_death_confirmed = on_death_confirmed
        self._state = HeartbeatState()
        self._last_event_time: float = time.monotonic()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._pubsub_task: asyncio.Task[None] | None = None
        self._death_triggered = False
        self._log = logger.bind(component="skia.heartbeat")

    @property
    def state(self) -> HeartbeatState:
        return self._state

    async def start(self) -> None:
        """Start both the pub/sub listener and the heartbeat checker."""
        self._running = True
        self._death_triggered = False
        self._last_event_time = time.monotonic()
        self._pubsub_task = asyncio.create_task(
            self._subscribe_loop(), name="skia_heartbeat_subscribe"
        )
        self._task = asyncio.create_task(
            self._check_loop(), name="skia_heartbeat_check"
        )
        self._log.info("heartbeat_monitor_started")

    async def stop(self) -> None:
        self._running = False
        for task in (self._task, self._pubsub_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._task = None
        self._pubsub_task = None
        self._log.info("heartbeat_monitor_stopped")

    # ── Redis pub/sub listener ────────────────────────────────────

    async def _subscribe_loop(self) -> None:
        """Listen for synapse_events on Redis pub/sub and update last_event_time."""
        channel = f"{_REDIS_PREFIX}:channel:{self._config.heartbeat_channel}"
        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            pubsub = self._redis.pubsub()
            try:
                await pubsub.subscribe(channel)
                backoff = 1.0
                self._log.info("pubsub_subscribed", channel=channel)

                while self._running:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )
                    if message is not None and message.get("type") == "message":
                        self._last_event_time = time.monotonic()
                        self._state.last_heartbeat_at = utc_now()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._running:
                    self._log.warning(
                        "pubsub_error", error=str(exc), retry_in_s=backoff
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
            finally:
                with contextlib.suppress(Exception):
                    await pubsub.unsubscribe(channel)
                    await pubsub.aclose()

    # ── Periodic heartbeat checker ────────────────────────────────

    async def _check_loop(self) -> None:
        """Periodic check for heartbeat presence."""
        while self._running:
            await asyncio.sleep(self._config.heartbeat_poll_interval_s)
            if not self._running:
                break

            self._state.last_check_at = utc_now()
            elapsed = time.monotonic() - self._last_event_time
            threshold_s = self._config.heartbeat_poll_interval_s * 2

            if elapsed < threshold_s:
                # Heartbeat present — organism is alive
                if self._state.status != HeartbeatStatus.ALIVE:
                    if self._state.status in (
                        HeartbeatStatus.SUSPECTED_DEAD,
                        HeartbeatStatus.CONFIRMED_DEAD,
                    ):
                        self._state.total_false_positives += 1
                    self._log.info(
                        "heartbeat_recovered",
                        was=self._state.status,
                        misses=self._state.consecutive_misses,
                    )
                self._state.consecutive_misses = 0
                self._state.consecutive_confirmations = 0
                self._state.status = HeartbeatStatus.ALIVE
                self._death_triggered = False
                continue

            # No heartbeat — increment miss counter
            self._state.consecutive_misses += 1

            if self._state.consecutive_misses < self._config.heartbeat_failure_threshold:
                continue

            # Threshold reached — enter suspicion
            if self._state.status == HeartbeatStatus.ALIVE:
                self._state.status = HeartbeatStatus.SUSPECTED_DEAD
                self._log.warning(
                    "heartbeat_suspected_dead",
                    misses=self._state.consecutive_misses,
                )

            # Run confirmation checks (blocks the check loop during confirmation)
            if self._state.status == HeartbeatStatus.SUSPECTED_DEAD:
                confirmed = await self._run_confirmation_checks()
                if confirmed and not self._death_triggered:
                    self._state.status = HeartbeatStatus.CONFIRMED_DEAD
                    self._state.total_deaths_detected += 1
                    self._death_triggered = True
                    self._log.error(
                        "heartbeat_confirmed_dead",
                        total_deaths=self._state.total_deaths_detected,
                    )
                    if self._on_death_confirmed:
                        await self._on_death_confirmed()

    # ── Active confirmation probes ────────────────────────────────

    async def _run_confirmation_checks(self) -> bool:
        """
        Run N active probes to confirm death.

        Returns True only if ALL probes confirm the organism is dead.
        Returns False if any check succeeds (heartbeat recovered) or
        if Redis itself is unreachable (different failure mode).
        """
        for i in range(self._config.heartbeat_confirmation_checks):
            await asyncio.sleep(self._config.heartbeat_confirmation_interval_s)
            if not self._running:
                return False

            # Probe 1: Is Redis itself alive?
            try:
                await self._redis.ping()
            except Exception:
                self._log.warning(
                    "redis_unreachable_during_confirmation",
                    check=i + 1,
                )
                # Redis down is a different problem — don't trigger restoration
                return False

            # Probe 2: Did a new heartbeat arrive while we waited?
            elapsed = time.monotonic() - self._last_event_time
            threshold_s = self._config.heartbeat_poll_interval_s * 2
            if elapsed < threshold_s:
                self._log.info(
                    "heartbeat_recovered_during_confirmation",
                    check=i + 1,
                )
                self._state.status = HeartbeatStatus.ALIVE
                self._state.consecutive_misses = 0
                return False

            self._state.consecutive_confirmations += 1
            self._log.info(
                "confirmation_check_failed",
                check=i + 1,
                total=self._config.heartbeat_confirmation_checks,
            )

        return True
