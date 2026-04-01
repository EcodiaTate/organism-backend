"""
EcodiaOS - File Watcher Perception Channel

Watches a folder for `.txt` and `.md` files dropped by operators or external
tools.  Each file is read, ingested through Atune as a SYSTEM_EVENT percept,
then deleted (success) or renamed to `.failed` (ingest error).

Usage:
    watcher = FileWatcher(watch_dir=Path("config/percepts"), atune=atune_svc)
    await watcher.start()
    ...
    await watcher.stop()

Design notes:
- Pure asyncio polling (no watchdog dependency) - 1-second tick.
- Each file is processed once; concurrent ingestion is safe because
  Atune.ingest() is thread-safe and queue-bounded.
- Files processed in mtime order (oldest first) so manual batches
  land in the right temporal sequence.
- Metadata extracted from YAML front-matter when present:
    ---
    channel: external_api
    priority: 0.8
    source: weather_poller
    ---
    Body text starts here.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from systems.fovea.gateway import AtuneService

logger = structlog.get_logger("file_watcher")

_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_WATCHED_SUFFIXES = {".txt", ".md"}
_POLL_INTERVAL = 1.0  # seconds


def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """
    Extract optional YAML front-matter from the text.

    Returns (metadata_dict, body_text).  If no front-matter is present,
    returns ({}, original_text).
    """
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text

    import yaml  # only imported when front-matter is present

    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        meta = {}

    body = text[m.end():]
    return meta, body


class FileWatcher:
    """
    Background asyncio task that polls a directory for new percept files.

    Parameters
    ----------
    watch_dir:
        Directory to watch.  Created automatically if it does not exist.
    atune:
        AtuneService reference for ingestion.
    poll_interval:
        How often to scan the directory (seconds).  Default 1.0.
    """

    def __init__(
        self,
        watch_dir: Path,
        atune: AtuneService,
        poll_interval: float = _POLL_INTERVAL,
    ) -> None:
        self._dir = watch_dir
        self._atune = atune
        self._interval = poll_interval
        self._task: asyncio.Task[None] | None = None
        self._ingested = 0
        self._failed = 0

    # ── Public API ────────────────────────────────────────────────

    async def start(self) -> None:
        """Create watch directory and launch background polling task."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._poll_loop(), name="file_watcher")
        logger.info("file_watcher_started", watch_dir=str(self._dir))

    async def stop(self) -> None:
        """Cancel background task and wait for it to finish."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info(
            "file_watcher_stopped",
            ingested=self._ingested,
            failed=self._failed,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "watch_dir": str(self._dir),
            "ingested": self._ingested,
            "failed": self._failed,
            "running": self._task is not None and not self._task.done(),
        }

    # ── Internals ─────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._scan()
            except asyncio.CancelledError:
                logger.debug("file_watcher_poll_cancelled")
                return
            except Exception as exc:
                logger.warning("file_watcher_poll_error", error=str(exc))

    async def _scan(self) -> None:
        """Process all eligible files in mtime order (oldest first)."""
        try:
            candidates = sorted(
                (p for p in self._dir.iterdir() if p.suffix in _WATCHED_SUFFIXES),
                key=lambda p: p.stat().st_mtime,
            )
        except Exception as exc:
            logger.warning("file_watcher_scan_error", error=str(exc))
            return

        for path in candidates:
            await self._process(path)

    async def _process(self, path: Path) -> None:
        from systems.fovea.types import InputChannel, RawInput

        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("file_watcher_read_error", path=str(path), error=str(exc))
            return

        if not text:
            path.unlink(missing_ok=True)
            return

        meta, body = _parse_front_matter(text)

        # Resolve channel (default: system_event)
        channel_str = meta.pop("channel", "system_event")
        try:
            channel = InputChannel(channel_str)
        except ValueError:
            channel = InputChannel.SYSTEM_EVENT

        # Build metadata - attach filename and any extra front-matter keys
        metadata: dict[str, Any] = {
            "source_file": path.name,
            "ingestion_channel": "file_watcher",
            **meta,
        }

        raw = RawInput(
            data=body,
            channel_id=f"file:{path.name}",
            metadata=metadata,
        )

        percept_id = await self._atune.ingest(raw, channel)

        if percept_id is not None:
            self._ingested += 1
            path.unlink(missing_ok=True)
            logger.info(
                "file_watcher_ingested",
                file=path.name,
                percept_id=percept_id,
                channel=channel_str,
            )
        else:
            # Queue full - back off; file stays, will retry next tick
            logger.warning(
                "file_watcher_queue_full",
                file=path.name,
            )
