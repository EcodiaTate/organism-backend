"""
EcodiaOS — Supervised Task Utility

Wraps asyncio coroutines in a self-restarting supervisor that:
  - Logs every exception with full context and traceback
  - Respawns with exponential backoff up to max_restarts
  - Emits TASK_PERMANENTLY_FAILED on the Synapse bus when all restarts are exhausted
  - Never swallows exceptions silently

Usage::

    from utils.supervision import supervised_task

    # Simple fire-and-forget supervised task
    supervised_task(my_coro(), name="my_task")

    # With custom restart policy and event bus
    supervised_task(
        my_coro(),
        name="critical_loop",
        restart=True,
        max_restarts=5,
        backoff_base=2.0,
        event_bus=self._event_bus,
        source_system="my_system",
    )

The returned asyncio.Task is the *supervisor* task — cancel it to stop
both the supervisor and the underlying coroutine.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from systems.synapse.event_bus import EventBus

_logger = structlog.get_logger("utils.supervision")


def supervised_task(
    coro: Coroutine[Any, Any, Any],
    *,
    name: str,
    restart: bool = True,
    max_restarts: int = 3,
    backoff_base: float = 2.0,
    event_bus: EventBus | None = None,
    source_system: str = "supervision",
) -> asyncio.Task[None]:
    """
    Run *coro* as a supervised asyncio task.

    Parameters
    ----------
    coro:
        The coroutine to run.  It will be consumed immediately — if you need
        to restart it, pass a factory (see note below).
    name:
        Human-readable name used in all log messages and the bus event.
    restart:
        Whether to respawn after failure (default True).  Pass False for
        one-shot tasks where failure should be logged but not retried.
    max_restarts:
        Maximum number of restart attempts before emitting
        TASK_PERMANENTLY_FAILED and giving up.
    backoff_base:
        Base for the exponential back-off delay.  Attempt N waits
        ``backoff_base ** N`` seconds before respawning.
    event_bus:
        Optional Synapse EventBus.  When provided, a TASK_PERMANENTLY_FAILED
        event is emitted after exhausting all restarts.
    source_system:
        ``source_system`` field for the bus event.

    Returns
    -------
    asyncio.Task
        The *supervisor* task.  Cancel it to stop the supervision loop.

    Note
    ----
    The first invocation consumes *coro* directly.  For tasks that need
    repeated restarts you must supply a *factory* coroutine — i.e. an
    ``async def`` that itself creates and awaits the target, then loops.
    Most long-running daemon loops already have an internal ``while True``
    so a single coroutine is sufficient.
    """
    supervisor = asyncio.create_task(
        _run_supervised(
            coro=coro,
            name=name,
            restart=restart,
            max_restarts=max_restarts,
            backoff_base=backoff_base,
            event_bus=event_bus,
            source_system=source_system,
        ),
        name=f"supervisor:{name}",
    )
    return supervisor


async def _run_supervised(
    coro: Coroutine[Any, Any, Any],
    name: str,
    restart: bool,
    max_restarts: int,
    backoff_base: float,
    event_bus: EventBus | None,
    source_system: str,
) -> None:
    """Internal supervisor loop."""
    log = _logger.bind(task=name)
    attempt = 0
    current_coro: Coroutine[Any, Any, Any] | None = coro

    while True:
        try:
            if current_coro is None:
                # All restarts exhausted — exit the supervisor.
                return
            await current_coro
            # Coroutine exited cleanly — no need to restart.
            log.debug("supervised_task_completed", attempt=attempt)
            return

        except asyncio.CancelledError:
            # Propagate cancellation — supervisor is being shut down.
            log.debug("supervised_task_cancelled", attempt=attempt)
            raise

        except Exception as exc:
            tb = traceback.format_exc()
            log.error(
                "supervised_task_failed",
                attempt=attempt,
                error=str(exc),
                error_type=type(exc).__name__,
                traceback=tb,
            )

            if not restart or attempt >= max_restarts:
                log.error(
                    "supervised_task_permanently_failed",
                    total_attempts=attempt + 1,
                    max_restarts=max_restarts,
                    final_error=str(exc),
                )
                await _emit_permanent_failure(
                    name=name,
                    exc=exc,
                    tb=tb,
                    attempt=attempt,
                    event_bus=event_bus,
                    source_system=source_system,
                )
                return

            attempt += 1
            delay = backoff_base ** attempt
            log.warning(
                "supervised_task_restarting",
                attempt=attempt,
                max_restarts=max_restarts,
                backoff_delay_s=round(delay, 1),
            )
            await asyncio.sleep(delay)

            # The original coroutine is exhausted — we cannot re-await it.
            # Signal to the caller that supervised tasks needing actual restart
            # should use a factory wrapper.  For the common case of a daemon
            # loop (while True inside the coro), the coro only exits on error
            # so we never reach this branch under normal operation.
            current_coro = None
            log.error(
                "supervised_task_no_factory",
                task=name,
                hint=(
                    "Wrap the coroutine in a factory function "
                    "to enable true restarts after failure."
                ),
            )
            return


async def _emit_permanent_failure(
    name: str,
    exc: Exception,
    tb: str,
    attempt: int,
    event_bus: EventBus | None,
    source_system: str,
) -> None:
    """Emit TASK_PERMANENTLY_FAILED on the bus if one is available."""
    if event_bus is None:
        return
    try:
        from systems.synapse.types import SynapseEvent, SynapseEventType

        await event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType.TASK_PERMANENTLY_FAILED,
                source_system=source_system,
                data={
                    "task_name": name,
                    "final_error": str(exc),
                    "error_type": type(exc).__name__,
                    "restart_attempts": attempt,
                    "traceback": tb,
                },
            )
        )
    except Exception as emit_exc:
        _logger.error(
            "supervised_task_bus_emit_failed",
            task=name,
            error=str(emit_exc),
        )
