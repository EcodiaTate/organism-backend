"""
Unit tests for Simula HealthChecker.

Tests syntax checking, import validation, and test execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from systems.simula.health import HealthChecker

if TYPE_CHECKING:
    from pathlib import Path

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


class TestEmptyCheck:
    @pytest.mark.asyncio
    async def test_no_files(self, tmp_path: Path):
        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([])
        assert result.healthy
