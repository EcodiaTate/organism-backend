"""
Unit tests for Inspector TargetWorkspace.

Tests workspace creation, cleanup, factory methods, and isolation guarantees.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from systems.simula.inspector.workspace import TargetWorkspace

# ── Construction Tests ──────────────────────────────────────────────────────


class TestTargetWorkspaceConstruction:
    def test_valid_directory(self, tmp_path: Path):
        ws = TargetWorkspace(
            root=tmp_path,
            workspace_type="external_repo",
        )
        assert ws.root == tmp_path.resolve()
        assert ws.workspace_type == "external_repo"
        assert ws.temp_directory is None

    def test_nonexistent_directory_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            TargetWorkspace(
                root=Path("/nonexistent/path/to/nowhere"),
                workspace_type="external_repo",
            )

    def test_file_instead_of_directory_raises(self, tmp_path: Path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        with pytest.raises(NotADirectoryError, match="not a directory"):
            TargetWorkspace(root=file_path, workspace_type="external_repo")

    def test_with_temp_directory(self, tmp_path: Path):
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        ws = TargetWorkspace(
            root=tmp_path,
            workspace_type="external_repo",
            temp_directory=temp_dir,
        )
        assert ws.temp_directory == temp_dir


# ── Properties ──────────────────────────────────────────────────────────────


class TestTargetWorkspaceProperties:
    def test_is_external_true(self, tmp_path: Path):
        ws = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        assert ws.is_external is True

    def test_is_external_false_for_internal(self, tmp_path: Path):
        ws = TargetWorkspace(root=tmp_path, workspace_type="internal_eos")
        assert ws.is_external is False

    def test_repr(self, tmp_path: Path):
        ws = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        r = repr(ws)
        assert "TargetWorkspace" in r
        assert "external_repo" in r


# ── Cleanup ─────────────────────────────────────────────────────────────────


class TestTargetWorkspaceCleanup:
    def test_cleanup_removes_temp_directory(self, tmp_path: Path):
        temp_dir = tmp_path / "inspector_temp"
        temp_dir.mkdir()
        (temp_dir / "file.txt").write_text("data")

        ws = TargetWorkspace(
            root=tmp_path,
            workspace_type="external_repo",
            temp_directory=temp_dir,
        )

        ws.cleanup()
        assert not temp_dir.exists()

    def test_cleanup_noop_without_temp(self, tmp_path: Path):
        ws = TargetWorkspace(root=tmp_path, workspace_type="internal_eos")
        # Should not raise
        ws.cleanup()

    def test_cleanup_noop_if_temp_already_removed(self, tmp_path: Path):
        temp_dir = tmp_path / "already_gone"
        temp_dir.mkdir()

        ws = TargetWorkspace(
            root=tmp_path,
            workspace_type="external_repo",
            temp_directory=temp_dir,
        )

        # Pre-remove
        shutil.rmtree(temp_dir)
        assert not temp_dir.exists()

        # Should not raise
        ws.cleanup()


# ── Factory Methods ─────────────────────────────────────────────────────────


class TestTargetWorkspaceFactoryMethods:
    def test_from_local_path(self, tmp_path: Path):
        ws = TargetWorkspace.from_local_path(tmp_path)
        assert ws.root == tmp_path.resolve()
        assert ws.workspace_type == "external_repo"
        assert ws.temp_directory is None

    def test_internal(self, tmp_path: Path):
        ws = TargetWorkspace.internal(tmp_path)
        assert ws.root == tmp_path.resolve()
        assert ws.workspace_type == "internal_eos"
        assert ws.is_external is False
        assert ws.temp_directory is None

    @pytest.mark.asyncio
    async def test_from_github_url_success(self, tmp_path: Path):
        """Test cloning by mocking asyncio.create_subprocess_exec."""
        # The subprocess mock returns success (returncode=0)
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec, \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)):
            # Create the expected repo directory so TargetWorkspace validates
            (tmp_path / "repo").mkdir(exist_ok=True)

            ws = await TargetWorkspace.from_github_url(
                "https://github.com/test/repo",
                clone_depth=1,
            )

            assert ws.workspace_type == "external_repo"
            assert ws.temp_directory == tmp_path
            assert "repo" in str(ws.root)

            # Verify git clone was called with correct args
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[0]
            assert "git" in call_args
            assert "clone" in call_args
            assert "--depth" in call_args
            assert "https://github.com/test/repo" in call_args

    @pytest.mark.asyncio
    async def test_from_github_url_clone_failure(self, tmp_path: Path):
        """Test that a failed clone raises RuntimeError and cleans up."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 128
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"fatal: repository not found"),
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)):
            with pytest.raises(RuntimeError, match="git clone failed"):
                await TargetWorkspace.from_github_url("https://github.com/invalid/repo")

    @pytest.mark.asyncio
    async def test_clone_depth_respected(self, tmp_path: Path):
        """Verify custom clone_depth is passed to git."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec, \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)):
            (tmp_path / "repo").mkdir(exist_ok=True)

            await TargetWorkspace.from_github_url(
                "https://github.com/test/repo",
                clone_depth=5,
            )

            call_args = mock_exec.call_args[0]
            depth_idx = list(call_args).index("--depth")
            assert call_args[depth_idx + 1] == "5"
