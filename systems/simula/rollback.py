"""
EcodiaOS -- Simula Rollback Manager

Before any change is applied, RollbackManager snapshots all files
that might be modified. If the post-apply health check fails -- or if
any exception occurs during application -- the manager restores the
codebase to its pre-change state.

Rollback target: <=2s (from spec).

Snapshots are persisted to Redis with TTL to prevent unbounded memory growth.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from systems.simula.errors import RollbackError
from systems.simula.evolution_types import ConfigSnapshot, FileSnapshot

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger().bind(system="simula.rollback")

# Cooldown between consecutive SIMULA_ROLLBACK_PENALTY emissions (seconds)
_ROLLBACK_PENALTY_COOLDOWN_S: float = 60.0


class RollbackManager:
    """
    Captures file snapshots before structural changes are applied
    and restores them if the post-apply health check fails.
    Snapshots are cached in Redis with TTL to prevent unbounded growth.
    """

    def __init__(
        self,
        codebase_root: Path,
        redis: RedisClient | None = None,
        snapshot_ttl_seconds: int = 3600,
        event_bus: Any = None,
    ) -> None:
        self._root = codebase_root
        self._redis = redis
        self._snapshot_ttl = snapshot_ttl_seconds
        self._log = logger
        self._event_bus: Any = event_bus
        self._last_penalty_emit_time: float = 0.0

    async def snapshot(self, proposal_id: str, paths: list[Path]) -> ConfigSnapshot:
        """""""""
        Read each file's current content and package into a ConfigSnapshot.
        Files that do not exist are recorded with existed=False so rollback
        knows to delete them rather than restore content.
        Snapshot is persisted to Redis with TTL.
        """""""""
        snapshots: list[FileSnapshot] = []
        for path in paths:
            abs_path = path if path.is_absolute() else self._root / path
            content = await self._read_file_safe(abs_path)
            existed = content is not None
            snapshots.append(
                FileSnapshot(
                    path=str(abs_path),
                    content=content,
                    existed=existed,
                )
            )
            self._log.debug(
                "snapshot_captured",
                path=str(abs_path),
                existed=existed,
                size=len(content) if content else 0,
            )

        # We need a config_version from outside context; use 0 as placeholder.
        # The service layer will update this before persisting.
        cfg_snapshot = ConfigSnapshot(
            proposal_id=proposal_id,
            files=snapshots,
            config_version=0,
        )

        # Persist to Redis with TTL to prevent unbounded growth
        if self._redis is not None:
            try:
                redis_key = f"simula:rollback:{proposal_id}"
                snapshot_json = cfg_snapshot.model_dump_json()
                await self._redis.set_with_ttl(
                    redis_key,
                    snapshot_json,
                    ttl_seconds=self._snapshot_ttl,
                )
                self._log.debug(
                    "snapshot_cached_redis",
                    proposal_id=proposal_id,
                    ttl_seconds=self._snapshot_ttl,
                )
            except Exception as exc:
                self._log.warning("snapshot_redis_cache_failed", error=str(exc))

        self._log.info(
            "snapshot_complete",
            proposal_id=proposal_id,
            files_captured=len(snapshots),
        )
        return cfg_snapshot

    async def restore(self, snapshot: ConfigSnapshot) -> list[str]:
        """""""""
        Restore all files to the state captured in the snapshot.
        Files that did not exist before are deleted.
        Returns the list of absolute paths that were restored.
        Each file restore is retried up to 3 times with exponential backoff.
        Raises RollbackError if any restore fails after all retries.
        """""""""
        restored: list[str] = []
        errors: list[str] = []

        for file_snap in snapshot.files:
            path = Path(file_snap.path)
            last_exc: Exception | None = None
            for _attempt in range(3):
                if _attempt > 0:
                    await asyncio.sleep(2 ** _attempt + random.uniform(0, 1))
                try:
                    if not file_snap.existed:
                        # File was created by the change -- delete it
                        if path.exists():
                            path.unlink()
                            self._log.info("rollback_deleted", path=str(path))
                        restored.append(str(path))
                    elif file_snap.content is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(file_snap.content, encoding="utf-8")
                        self._log.info("rollback_restored", path=str(path), attempt=_attempt)
                        restored.append(str(path))
                    else:
                        # Snapshot recorded file existed but content was unreadable - cannot restore
                        self._log.warning("rollback_skipped_no_content", path=str(path))
                    last_exc = None
                    break  # success
                except Exception as exc:
                    last_exc = exc
                    self._log.warning(
                        "rollback_restore_attempt_failed",
                        path=str(path),
                        attempt=_attempt,
                        error=str(exc),
                    )
            if last_exc is not None:
                msg = f"Failed to restore {path} after 3 attempts: {last_exc}"
                self._log.error("rollback_restore_failed", path=str(path), error=str(last_exc))
                errors.append(msg)

        if errors:
            raise RollbackError("Rollback incomplete. Failures: " + str(errors))

        self._log.info(
            "rollback_complete",
            proposal_id=snapshot.proposal_id,
            files_restored=len(restored),
        )

        # Emit SIMULA_ROLLBACK_PENALTY so Oikos can charge metabolic cost
        await self._emit_rollback_penalty(
            proposal_id=snapshot.proposal_id,
            files_restored=len(restored),
            risk_level=getattr(snapshot, "risk_level", "NORMAL"),
            reason=getattr(snapshot, "rollback_reason", "health_check_failed"),
        )

        return restored

    async def _emit_rollback_penalty(
        self,
        proposal_id: str,
        files_restored: int,
        risk_level: str,
        reason: str,
    ) -> None:
        """Emit SIMULA_ROLLBACK_PENALTY with metabolic cost for Oikos."""
        if self._event_bus is None:
            return

        now = time.monotonic()
        if now - self._last_penalty_emit_time < _ROLLBACK_PENALTY_COOLDOWN_S:
            return
        self._last_penalty_emit_time = now

        # Cost formula: base $0.10 + $0.02 per file, 2x for HIGH/CRITICAL
        cost = Decimal("0.10") + Decimal("0.02") * files_restored
        if risk_level.upper() in ("HIGH", "CRITICAL"):
            cost *= 2

        from systems.synapse.types import SynapseEvent, SynapseEventType

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SIMULA_ROLLBACK_PENALTY,
                source_system="simula",
                data={
                    "proposal_id": proposal_id,
                    "files_restored": files_restored,
                    "cost_usd": str(cost),
                    "risk_level": risk_level,
                    "reason": reason,
                },
            ))
            self._log.info(
                "rollback_penalty_emitted",
                proposal_id=proposal_id,
                cost_usd=str(cost),
                files_restored=files_restored,
            )
        except Exception as exc:
            self._log.warning("rollback_penalty_emit_failed", error=str(exc))

    async def _read_file_safe(self, path: Path) -> str | None:
        """""""""
        Read file content; return None if file does not exist.
        """""""""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception as exc:
            self._log.warning("snapshot_read_failed", path=str(path), error=str(exc))
            return None
