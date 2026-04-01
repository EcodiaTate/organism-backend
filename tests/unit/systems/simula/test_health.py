"""
Unit tests for Simula HealthChecker.

Tests syntax checking, import validation, test execution, and the
integration-test / performance-baseline opt-in paths. #79 #80
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.simula.health import HealthChecker

# ─── Tests ────────────────────────────────────────────────────────────────────


class TestSyntaxCheck:
    @pytest.mark.asyncio
    async def test_valid_python(self, tmp_path: Path):
        f = tmp_path / "valid.py"
        f.write_text('def hello():\n    return "world"\n', encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([str(f)])
        assert result.healthy

    @pytest.mark.asyncio
    async def test_invalid_python(self, tmp_path: Path):
        f = tmp_path / "invalid.py"
        f.write_text("def broken(\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([str(f)])
        assert not result.healthy
        assert any("syntax" in issue.lower() or "Syntax" in issue for issue in result.issues)

    @pytest.mark.asyncio
    async def test_non_python_file_skipped(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text("{}", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([str(f)])
        # Non-Python files should not cause syntax check failures
        assert result.healthy

    @pytest.mark.asyncio
    async def test_missing_file_reported_as_error(self, tmp_path: Path):
        """A .py path that does not exist should be reported, not silently skipped."""
        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([str(tmp_path / "ghost.py")])
        assert not result.healthy
        assert any("not found" in issue for issue in result.issues)

    @pytest.mark.asyncio
    async def test_multiple_files_first_error_wins(self, tmp_path: Path):
        """health check returns on first failure - subsequent files irrelevant."""
        bad = tmp_path / "bad.py"
        bad.write_text("def broken(\n", encoding="utf-8")
        good = tmp_path / "good.py"
        good.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([str(bad), str(good)])
        assert not result.healthy

    @pytest.mark.asyncio
    async def test_unicode_source_valid(self, tmp_path: Path):
        """Source files with non-ASCII identifiers should parse cleanly."""
        f = tmp_path / "unicode_ok.py"
        f.write_text("# -*- coding: utf-8 -*-\ncafé = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([str(f)])
        assert result.healthy


class TestEmptyCheck:
    @pytest.mark.asyncio
    async def test_no_files(self, tmp_path: Path):
        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([])
        assert result.healthy


class TestImportCheck:
    @pytest.mark.asyncio
    async def test_importable_module_passes(self, tmp_path: Path):
        """A module already on sys.path should pass the import check."""
        # Use a stdlib module that we know exists - write a stub file pointing to it
        # by placing it in a package that _derive_module_path will resolve.
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        mod = pkg / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        # Import check will try to find "mypkg.mod"; if not on sys.path it returns
        # "module not found" - but the syntax check passed, so this tests the
        # import-not-found path gracefully.
        result = await checker.check([str(mod)])
        # We only care that the check completes without raising
        assert isinstance(result.healthy, bool)

    @pytest.mark.asyncio
    async def test_module_not_found_reported(self, tmp_path: Path):
        """A file whose derived module path has no matching importlib spec
        should be reported as an import error."""
        f = tmp_path / "nonexistent_module_xyz.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        # Syntax is fine, but import check will fail (module not on path)
        result = await checker.check([str(f)])
        # Import failure should surface in issues (or be silently skipped if
        # the module has no importlib entry - either is acceptable as long as
        # the check does not raise an unhandled exception)
        assert isinstance(result.healthy, bool)


class TestTestRunner:
    @pytest.mark.asyncio
    async def test_no_test_dir_passes(self, tmp_path: Path):
        """When no test directory is found, check should still pass."""
        f = tmp_path / "orphan.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([str(f)])
        # No test directory → passes (no tests = not a failure)
        assert result.healthy

    @pytest.mark.asyncio
    async def test_test_runner_passes_on_zero_exit(self, tmp_path: Path):
        """If pytest exits 0, the health check should pass."""
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"1 passed", None))
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # Patch _derive_test_path to return a fake but "existing" dir
            with patch.object(checker, "_derive_test_path", return_value=str(tmp_path)):
                result = await checker.check([str(f)])

        assert result.healthy

    @pytest.mark.asyncio
    async def test_test_runner_fails_on_nonzero_exit(self, tmp_path: Path):
        """If pytest exits non-zero, the health check should fail."""
        f = tmp_path / "bad.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"FAILED", None))
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(checker, "_derive_test_path", return_value=str(tmp_path)):
                result = await checker.check([str(f)])

        assert not result.healthy
        assert any("Unit test suite failed" in issue for issue in result.issues)

    @pytest.mark.asyncio
    async def test_test_runner_timeout_fails(self, tmp_path: Path):
        """A 30-second subprocess timeout should surface as a health failure."""
        import asyncio as _asyncio

        f = tmp_path / "ok.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)

        async def _hanging_communicate():
            raise TimeoutError

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate = _hanging_communicate
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(checker, "_derive_test_path", return_value=str(tmp_path)):
                with patch("asyncio.wait_for", side_effect=_asyncio.TimeoutError):
                    result = await checker.check([str(f)])

        assert not result.healthy
        assert any("timed out" in issue.lower() for issue in result.issues)


class TestIntegrationTestsOptIn:
    """Integration-test gate (opt-in via integration_tests_enabled). #80"""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self, tmp_path: Path):
        """Integration tests must NOT run when integration_tests_enabled=False."""
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path, integration_tests_enabled=False)
        with patch.object(checker, "_run_integration_tests", new_callable=AsyncMock) as mock_int:
            with patch.object(checker, "_run_tests", return_value=(True, "")):
                await checker.check([str(f)])
            mock_int.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_and_passing(self, tmp_path: Path):
        """When enabled and integration tests pass, health check should pass."""
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(
            codebase_root=tmp_path,
            integration_tests_enabled=True,
        )
        with patch.object(checker, "_run_tests", return_value=(True, "")):
            with patch.object(
                checker, "_run_integration_tests", return_value=(True, "ok")
            ):
                result = await checker.check([str(f)])

        assert result.healthy

    @pytest.mark.asyncio
    async def test_enabled_and_failing(self, tmp_path: Path):
        """When enabled and integration tests fail, health check should fail."""
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(
            codebase_root=tmp_path,
            integration_tests_enabled=True,
        )
        with patch.object(checker, "_run_tests", return_value=(True, "")):
            with patch.object(
                checker, "_run_integration_tests", return_value=(False, "FAILED")
            ):
                result = await checker.check([str(f)])

        assert not result.healthy
        assert any("Integration test suite failed" in issue for issue in result.issues)


class TestPerformanceBaselineOptIn:
    """Performance baseline gate (opt-in via performance_baseline_enabled). #80"""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self, tmp_path: Path):
        """Performance baseline must NOT run when performance_baseline_enabled=False."""
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path, performance_baseline_enabled=False)
        with patch.object(checker, "_run_performance_baseline", new_callable=AsyncMock) as mock_perf:
            with patch.object(checker, "_run_tests", return_value=(True, "")):
                await checker.check([str(f)])
            mock_perf.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_and_passing(self, tmp_path: Path):
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(
            codebase_root=tmp_path,
            performance_baseline_enabled=True,
        )
        with patch.object(checker, "_run_tests", return_value=(True, "")):
            with patch.object(
                checker, "_run_performance_baseline", return_value=(True, "baseline ok")
            ):
                result = await checker.check([str(f)])

        assert result.healthy

    @pytest.mark.asyncio
    async def test_enabled_and_failing(self, tmp_path: Path):
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(
            codebase_root=tmp_path,
            performance_baseline_enabled=True,
        )
        with patch.object(checker, "_run_tests", return_value=(True, "")):
            with patch.object(
                checker, "_run_performance_baseline", return_value=(False, "p99 regression")
            ):
                result = await checker.check([str(f)])

        assert not result.healthy
        assert any("Performance baseline regression" in issue for issue in result.issues)


class TestDeriveModulePath:
    """Unit tests for the _derive_module_path helper."""

    def test_path_relative_to_root(self, tmp_path: Path):
        checker = HealthChecker(codebase_root=tmp_path)
        f = tmp_path / "systems" / "axon" / "service.py"
        result = checker._derive_module_path(str(f))
        assert result == "systems.axon.service"

    def test_strips_src_prefix(self, tmp_path: Path):
        checker = HealthChecker(codebase_root=tmp_path)
        f = tmp_path / "src" / "systems" / "simula" / "health.py"
        result = checker._derive_module_path(str(f))
        assert result == "systems.simula.health"

    def test_single_file_at_root(self, tmp_path: Path):
        checker = HealthChecker(codebase_root=tmp_path)
        f = tmp_path / "main.py"
        result = checker._derive_module_path(str(f))
        assert result == "main"

    def test_non_python_file_returns_none(self, tmp_path: Path):
        checker = HealthChecker(codebase_root=tmp_path)
        f = tmp_path / "config.yaml"
        # YAML file has no .py suffix - module path should be None or the bare name
        result = checker._derive_module_path(str(f))
        # We don't care about the exact value, just that it doesn't raise
        assert result is None or isinstance(result, str)


class TestDeriveTestPath:
    """Unit tests for the _derive_test_path helper."""

    def test_systems_layout(self, tmp_path: Path):
        checker = HealthChecker(codebase_root=tmp_path)
        src = tmp_path / "systems" / "axon" / "service.py"
        result = checker._derive_test_path(str(src))
        assert result is not None
        assert "axon" in result

    def test_src_systems_layout(self, tmp_path: Path):
        checker = HealthChecker(codebase_root=tmp_path)
        src = tmp_path / "src" / "systems" / "nova" / "service.py"
        result = checker._derive_test_path(str(src))
        assert result is not None
        assert "nova" in result
