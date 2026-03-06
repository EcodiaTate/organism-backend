"""
Unit tests for Simula RollbackManager.

Tests snapshot creation, file restoration, and error handling for
non-existent files and failed restores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from systems.simula.rollback import RollbackManager

if TYPE_CHECKING:
    from pathlib import Path

# ─── Tests ────────────────────────────────────────────────────────────────────


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_existing_file(self, tmp_path: Path):
        """Snapshot captures the content of an existing file."""
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("original content", encoding="utf-8")

        mgr = RollbackManager(codebase_root=tmp_path)
        snapshot = await mgr.snapshot("prop_001", [test_file])

        assert len(snapshot.files) == 1
        assert snapshot.files[0].content == "original content"
        assert snapshot.files[0].existed is True

    @pytest.mark.asyncio
    async def test_snapshot_nonexistent_file(self, tmp_path: Path):
        """Snapshot marks non-existent files so rollback can delete them."""
        missing = tmp_path / "does_not_exist.py"
        mgr = RollbackManager(codebase_root=tmp_path)
        snapshot = await mgr.snapshot("prop_002", [missing])

        assert len(snapshot.files) == 1
        assert snapshot.files[0].content is None
        assert snapshot.files[0].existed is False

    @pytest.mark.asyncio
    async def test_snapshot_directory_expands_files(self, tmp_path: Path):
        """If given a directory path, snapshot all .py files inside it."""
        d = tmp_path / "pkg"
        d.mkdir()
        (d / "a.py").write_text("aaa", encoding="utf-8")
        (d / "b.py").write_text("bbb", encoding="utf-8")
        (d / "c.txt").write_text("skip", encoding="utf-8")

        mgr = RollbackManager(codebase_root=tmp_path)
        snapshot = await mgr.snapshot("prop_003", [d])

        # The rollback manager should snapshot at least the .py files
        py_files = [f for f in snapshot.files if f.path.endswith(".py")]
        assert len(py_files) >= 2

    @pytest.mark.asyncio
    async def test_snapshot_proposal_id(self, tmp_path: Path):
        mgr = RollbackManager(codebase_root=tmp_path)
        snapshot = await mgr.snapshot("my_proposal", [])
        assert snapshot.proposal_id == "my_proposal"


class TestRestore:
    @pytest.mark.asyncio
    async def test_restore_modified_file(self, tmp_path: Path):
        """Restore reverts a modified file to its original content."""
        test_file = tmp_path / "test.py"
        test_file.write_text("original", encoding="utf-8")

        mgr = RollbackManager(codebase_root=tmp_path)
        snapshot = await mgr.snapshot("prop_001", [test_file])

        # Simulate a change
        test_file.write_text("modified", encoding="utf-8")
        assert test_file.read_text() == "modified"

        # Restore
        restored = await mgr.restore(snapshot)
        assert test_file.read_text() == "original"
        assert len(restored) >= 1

    @pytest.mark.asyncio
    async def test_restore_deletes_new_file(self, tmp_path: Path):
        """Restore deletes files that didn't exist at snapshot time."""
        new_file = tmp_path / "new.py"
        mgr = RollbackManager(codebase_root=tmp_path)

        # Snapshot when file doesn't exist
        snapshot = await mgr.snapshot("prop_002", [new_file])

        # File gets created by the code agent
        new_file.write_text("new content", encoding="utf-8")
        assert new_file.exists()

        # Restore should delete it
        await mgr.restore(snapshot)
        assert not new_file.exists()

    @pytest.mark.asyncio
    async def test_restore_empty_snapshot(self, tmp_path: Path):
        mgr = RollbackManager(codebase_root=tmp_path)
        snapshot = await mgr.snapshot("prop_003", [])
        restored = await mgr.restore(snapshot)
        assert restored == []
