"""
EcodiaOS -- Simula Rollback Manager

Before any change is applied, RollbackManager snapshots all files
that might be modified. If the post-apply health check fails -- or if
any exception occurs during application -- the manager restores the
codebase to its pre-change state.

Rollback target: <=2s (from spec).
"""

from __future__ import annotations

from pathlib import Path

import structlog

from systems.simula.errors import RollbackError
from systems.simula.evolution_types import ConfigSnapshot, FileSnapshot

logger = structlog.get_logger().bind(system="simula.rollback")


class RollbackManager:
    """
    Captures file snapshots before structural changes are applied
    and restores them if the post-apply health check fails.
    """

    def __init__(self, codebase_root: Path) -> None:
        self._root = codebase_root
        self._log = logger

    async def snapshot(self, proposal_id: str, paths: list[Path]) -> ConfigSnapshot:
        """""""""
        Read each file's current content and package into a ConfigSnapshot.
        Files that do not exist are recorded with existed=False so rollback
        knows to delete them rather than restore content.
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
        Raises RollbackError if any restore fails.
        """""""""
        restored: list[str] = []
        errors: list[str] = []

        for file_snap in snapshot.files:
            path = Path(file_snap.path)
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
                    self._log.info("rollback_restored", path=str(path))
                    restored.append(str(path))
                else:
                    # Snapshot recorded file existed but content was unreadable — cannot restore
                    self._log.warning("rollback_skipped_no_content", path=str(path))
            except Exception as exc:
                msg = f"Failed to restore {path}: {exc}"
                self._log.error("rollback_restore_failed", path=str(path), error=str(exc))
                errors.append(msg)

        if errors:
            raise RollbackError("Rollback incomplete. Failures: " + str(errors))

        self._log.info(
            "rollback_complete",
            proposal_id=snapshot.proposal_id,
            files_restored=len(restored),
        )
        return restored

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
