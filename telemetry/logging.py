"""
EcodiaOS - Structured Logging

All logging via structlog. Every log entry includes system context.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from config import LoggingConfig


# ─── SSE Log Broadcast ───────────────────────────────────────────────────────
# A lightweight in-process fanout: the SSELogHandler enqueues every log record
# into all active subscriber queues. The /api/v1/admin/logs/stream endpoint
# creates a queue per connection and removes it on disconnect.

_log_subscribers: list[asyncio.Queue[dict[str, Any]]] = []
_MAX_QUEUE = 500  # cap per-subscriber to avoid unbounded memory


def subscribe_logs() -> asyncio.Queue[dict[str, Any]]:
    """Register a new SSE subscriber and return its queue."""
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUE)
    _log_subscribers.append(q)
    return q


def unsubscribe_logs(q: asyncio.Queue[dict[str, Any]]) -> None:
    """Remove a subscriber queue (called when the SSE connection closes)."""
    with contextlib.suppress(ValueError):
        _log_subscribers.remove(q)


def _format_record_time(record: logging.LogRecord) -> str:
    """ISO-8601 timestamp from the record's created float (no formatter needed)."""
    import datetime
    return (
        datetime.datetime.fromtimestamp(record.created, tz=datetime.UTC)
        .strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    )


class SSELogHandler(logging.Handler):
    """
    Standard-library logging handler that fans out each record to all
    active SSE subscriber queues as a structured dict.

    We deliberately use put_nowait + discard on full so that a slow browser
    never blocks the logging path.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if not _log_subscribers:
            return

        # Build a lean dict - only the fields we want to stream
        entry: dict[str, Any] = {
            "ts": _format_record_time(record),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Structlog attaches extra fields as record.__dict__ keys
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for k, v in record.__dict__.items():
            if k not in skip and not k.startswith("_"):
                try:
                    # Only include JSON-serialisable primitives
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        entry[k] = v
                except Exception:
                    pass

        for q in list(_log_subscribers):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                # Subscriber is too slow - drop oldest entry and retry
                try:
                    q.get_nowait()
                    q.put_nowait(entry)
                except Exception:
                    pass


class RedisStreamLogHandler(logging.Handler):
    """
    Logging handler that fire-and-forgets each log record into the
    LogAnalyzer's Redis Streams for real-time querying via /api/logs/*.

    Safe to attach before the analyzer is initialized - records are
    silently dropped until ``set_analyzer()`` is called.
    """

    def __init__(self) -> None:
        super().__init__()
        self._analyzer: Any | None = None

    def set_analyzer(self, analyzer: Any) -> None:
        self._analyzer = analyzer

    def emit(self, record: logging.LogRecord) -> None:
        if self._analyzer is None:
            return

        fields: dict[str, Any] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Structlog attaches extra fields as record.__dict__ keys
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for k, v in record.__dict__.items():
            if (
                k not in skip
                and not k.startswith("_")
                and (isinstance(v, (str, int, float, bool)) or v is None)
            ):
                fields[k] = v

        # Fire-and-forget into the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._analyzer.ingest_log(**fields))
        except RuntimeError:
            pass  # no running loop - startup phase, skip


# Module-level handle so registry can wire the analyzer after init
_redis_stream_handler: RedisStreamLogHandler | None = None


def get_redis_stream_handler() -> RedisStreamLogHandler | None:
    """Return the handler so the registry can call .set_analyzer()."""
    return _redis_stream_handler


def setup_logging(config: LoggingConfig, instance_id: str = "") -> None:
    """
    Configure structured logging for the entire application.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if instance_id:
        shared_processors.insert(
            0,
            structlog.processors.CallsiteParameterAdder(
                parameters=[structlog.processors.CallsiteParameter.FUNC_NAME]
            ),
        )

    renderer: Any
    if config.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    sse_handler = SSELogHandler()

    global _redis_stream_handler
    _redis_stream_handler = RedisStreamLogHandler()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.addHandler(sse_handler)
    root_logger.addHandler(_redis_stream_handler)
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Quiet noisy libraries
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
