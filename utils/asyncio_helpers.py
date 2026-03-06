"""Shared asyncio task cancellation and startup helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_logger = structlog.get_logger("utils.asyncio")


async def cancel_and_wait_tasks(
    tasks: list[asyncio.Task[object]],
    timeout: float = 10.0,
    logger: structlog.BoundLogger | None = None,
) -> None:
    """Cancel all *tasks* and await them so finally-blocks execute.

    Args:
        tasks: The asyncio tasks to cancel.
        timeout: Maximum seconds to wait for cancellation to complete.
        logger: Optional structlog logger for warnings.
    """
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
    except TimeoutError:
        if logger:
            logger.warning(
                "task_cancellation_timeout",
                count=len(tasks),
                timeout=timeout,
            )


async def init_optional[T](
    name: str,
    init_fn: Callable[[], Awaitable[T]],
    timeout: float = 5.0,
) -> T | None:
    """Initialize an optional dependency with a timeout.

    Returns ``None`` and logs a warning if the dependency times out or
    raises, so startup continues without it.
    """
    try:
        async with asyncio.timeout(timeout):
            result = await init_fn()
            _logger.info(f"{name}_initialized")
            return result
    except TimeoutError:
        _logger.warning(f"{name}_init_timeout", timeout_sec=timeout)
        return None
    except Exception as exc:
        _logger.warning(f"{name}_init_failed", error=str(exc))
        return None
