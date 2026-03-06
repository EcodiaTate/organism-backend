"""
EcodiaOS — Perception Scheduler

Runs named, cron-like background tasks that generate percepts on a fixed
interval.  Each task is a simple callable that returns text (or None to skip),
which is then ingested through Atune on the configured InputChannel.

Usage:
    scheduler = PerceptionScheduler(atune=atune_svc)
    scheduler.register(
        name="weather",
        interval_seconds=1800,          # every 30 minutes
        channel=InputChannel.EXTERNAL_API,
        fn=fetch_weather,               # async () -> str | None
    )
    await scheduler.start()
    ...
    await scheduler.stop()

Design notes:
- Each task runs independently; a slow task does not block others.
- Tasks fire immediately on first tick (no initial delay), then at interval.
- Exceptions in task callables are caught and logged — the scheduler
  continues running.
- Tasks can be added/removed at runtime (start() must be called first).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.atune.service import AtuneService
    from systems.atune.types import InputChannel

logger = structlog.get_logger("scheduler")

TaskFn = Callable[[], Coroutine[Any, Any, str | None]]


@dataclass
class _ScheduledTask:
    name: str
    interval_seconds: float
    channel: InputChannel
    fn: TaskFn
    metadata: dict[str, Any] = field(default_factory=dict)
    # runtime state
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _run_count: int = field(default=0, repr=False)
    _error_count: int = field(default=0, repr=False)


class PerceptionScheduler:
    """
    Drives scheduled perception by running background polling tasks
    and feeding results into Atune.

    Parameters
    ----------
    atune:
        AtuneService reference for ingestion.
    """

    def __init__(self, atune: AtuneService) -> None:
        self._atune = atune
        self._tasks: dict[str, _ScheduledTask] = {}
        self._started = False

    # ── Registration ──────────────────────────────────────────────

    def register(
        self,
        name: str,
        interval_seconds: float,
        channel: InputChannel,
        fn: TaskFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a new scheduled perception task.

        If the scheduler is already running, the task starts immediately.
        Re-registering an existing name replaces the old task.
        """
        if name in self._tasks:
            self.unregister(name)

        task = _ScheduledTask(
            name=name,
            interval_seconds=interval_seconds,
            channel=channel,
            fn=fn,
            metadata=metadata or {},
        )
        self._tasks[name] = task

        if self._started:
            task._task = asyncio.create_task(
                self._run_task(task), name=f"scheduler:{name}"
            )
            logger.info("scheduler_task_registered_live", name=name)

    def unregister(self, name: str) -> None:
        """Remove and cancel a scheduled task."""
        task = self._tasks.pop(name, None)
        if task is None:
            return
        if task._task is not None and not task._task.done():
            task._task.cancel()
        logger.info("scheduler_task_unregistered", name=name)

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all registered tasks."""
        self._started = True
        for task in self._tasks.values():
            if task._task is None or task._task.done():
                task._task = asyncio.create_task(
                    self._run_task(task), name=f"scheduler:{task.name}"
                )
        logger.info("scheduler_started", task_count=len(self._tasks))

    async def stop(self) -> None:
        """Cancel all running tasks."""
        self._started = False
        for task in self._tasks.values():
            if task._task is not None and not task._task.done():
                task._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task._task
        logger.info("scheduler_stopped", task_count=len(self._tasks))

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._started,
            "tasks": {
                name: {
                    "interval_seconds": t.interval_seconds,
                    "channel": str(t.channel),
                    "run_count": t._run_count,
                    "error_count": t._error_count,
                    "active": t._task is not None and not t._task.done(),
                }
                for name, t in self._tasks.items()
            },
        }

    # ── Internals ─────────────────────────────────────────────────

    async def _run_task(self, task: _ScheduledTask) -> None:
        from systems.atune.types import RawInput

        # Fire immediately on first run, then at interval
        first = True
        while True:
            if not first:
                try:
                    await asyncio.sleep(task.interval_seconds)
                except asyncio.CancelledError:
                    logger.debug("scheduler_task_cancelled", name=task.name)
                    return
            first = False

            try:
                text = await task.fn()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                task._error_count += 1
                logger.warning(
                    "scheduler_task_error",
                    name=task.name,
                    error=str(exc),
                )
                continue

            if text is None:
                logger.debug("scheduler_task_skipped", name=task.name)
                continue

            raw = RawInput(
                data=text,
                channel_id=f"scheduler:{task.name}",
                metadata={"scheduler_task": task.name, **task.metadata},
            )
            percept_id = await self._atune.ingest(raw, task.channel)
            task._run_count += 1

            if percept_id is not None:
                logger.debug(
                    "scheduler_task_ingested",
                    name=task.name,
                    percept_id=percept_id,
                )
            else:
                logger.debug("scheduler_task_queue_full", name=task.name)
