"""
Integration tests for Simula Stage 2 verification pipeline.

Tests the full formal verification phase in HealthChecker with
mocked bridges, and validates pass/fail/skip scenarios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from systems.simula.health import HealthChecker
from systems.simula.verification.types import (
    DAFNY_TRIGGERABLE_CATEGORIES,
    DafnyVerificationResult,
    DafnyVerificationStatus,
    DiscoveredInvariant,
    FormalVerificationResult,
    InvariantKind,
    InvariantVerificationResult,
    InvariantVerificationStatus,
    StaticAnalysisFinding,
    StaticAnalysisResult,
    StaticAnalysisSeverity,
)


def _make_proposal(category_value: str = "add_executor") -> MagicMock:
    """Create a mock EvolutionProposal."""
    proposal = MagicMock()
    proposal.category.value = category_value
    proposal.description = "Test proposal"
    proposal.change_spec.target_system = "axon"
    proposal.change_spec.description = "Test change"
    proposal.change_spec.files_to_modify = []
    proposal.change_spec.affected_systems = []
    # Make the category object support 'in' checks with frozenset
    from systems.simula.evolution_types import ChangeCategory
    if category_value == "modify_contract":
        proposal.category = ChangeCategory.MODIFY_CONTRACT
    elif category_value == "add_system_capability":
        proposal.category = ChangeCategory.ADD_SYSTEM_CAPABILITY
    elif category_value == "add_executor":
        proposal.category = ChangeCategory.ADD_EXECUTOR
    return proposal


# ── Health Check with No Bridges ──────────────────────────────────────────────


class TestHealthCheckNoBridges:
    @pytest.mark.asyncio
    async def test_no_bridges_returns_healthy(self, tmp_path: Path):
        """Without verification bridges, health check skips formal verification."""
        f = tmp_path / "valid.py"
        f.write_text("x = 1\n", encoding="utf-8")

        checker = HealthChecker(codebase_root=tmp_path)
        result = await checker.check([str(f)])
        assert result.healthy
        assert result.formal_verification is None


# ── Health Check with Static Analysis ─────────────────────────────────────────


class TestHealthCheckStaticAnalysis:
    @pytest.mark.asyncio
    async def test_static_analysis_passes(self, tmp_path: Path):
        """Static analysis with no findings → healthy."""
        f = tmp_path / "clean.py"
        f.write_text("x = 1\n", encoding="utf-8")

        mock_sa = AsyncMock()
        mock_sa.run_all = AsyncMock(return_value=StaticAnalysisResult())

        checker = HealthChecker(
            codebase_root=tmp_path,
            static_analysis_bridge=mock_sa,
        )
        result = await checker.check([str(f)])
        assert result.healthy

    @pytest.mark.asyncio
    async def test_static_analysis_errors_block(self, tmp_path: Path):
        """Static analysis with ERROR findings → unhealthy."""
        f = tmp_path / "bad.py"
        f.write_text("import pickle\n", encoding="utf-8")

        mock_sa = AsyncMock()
        mock_sa.run_all = AsyncMock(return_value=StaticAnalysisResult(
            findings=[
                StaticAnalysisFinding(
                    tool="bandit", rule_id="B301",
                    severity=StaticAnalysisSeverity.ERROR,
                    message="Use of pickle",
                    file_path="bad.py", line=1,
                ),
            ],
            error_count=1,
        ))

        checker = HealthChecker(
            codebase_root=tmp_path,
            static_analysis_bridge=mock_sa,
        )
        result = await checker.check([str(f)])
        assert not result.healthy
        assert result.formal_verification is not None
        assert len(result.formal_verification.blocking_issues) > 0

    @pytest.mark.asyncio
    async def test_static_analysis_warnings_advisory(self, tmp_path: Path):
        """Static analysis with only warnings → healthy with advisory."""
        f = tmp_path / "warn.py"
        f.write_text("assert True\n", encoding="utf-8")

        mock_sa = AsyncMock()
        mock_sa.run_all = AsyncMock(return_value=StaticAnalysisResult(
            findings=[
                StaticAnalysisFinding(
                    tool="bandit", rule_id="B101",
                    severity=StaticAnalysisSeverity.WARNING,
                    message="Assert used",
                    file_path="warn.py", line=1,
                ),
            ],
            warning_count=1,
        ))

        checker = HealthChecker(
            codebase_root=tmp_path,
            static_analysis_bridge=mock_sa,
        )
        result = await checker.check([str(f)])
        assert result.healthy
        assert result.formal_verification is not None
        assert len(result.formal_verification.advisory_issues) > 0


# ── Health Check with Dafny ───────────────────────────────────────────────────


class TestHealthCheckDafny:
    @pytest.mark.asyncio
    async def test_dafny_skipped_for_non_triggerable(self, tmp_path: Path):
        """Dafny only runs for MODIFY_CONTRACT and ADD_SYSTEM_CAPABILITY."""
        f = tmp_path / "code.py"
        f.write_text("x = 1\n", encoding="utf-8")

        mock_dafny = AsyncMock()
        mock_llm = AsyncMock()

        checker = HealthChecker(
            codebase_root=tmp_path,
            dafny_bridge=mock_dafny,
            llm=mock_llm,
        )
        proposal = _make_proposal("add_executor")
        result = await checker.check([str(f)], proposal=proposal)
        assert result.healthy
        # Dafny task should not have been created for add_executor
        mock_dafny.check_available.assert_not_called()

    @pytest.mark.asyncio
    async def test_dafny_runs_for_modify_contract(self, tmp_path: Path):
        """Dafny runs for MODIFY_CONTRACT category."""
        f = tmp_path / "contract.py"
        f.write_text("x = 1\n", encoding="utf-8")

        mock_dafny = AsyncMock()
        mock_dafny.check_available = AsyncMock(return_value=True)
        mock_dafny.run_clover_loop = AsyncMock(return_value=DafnyVerificationResult(
            status=DafnyVerificationStatus.VERIFIED,
            rounds_attempted=1,
        ))
        mock_llm = AsyncMock()

        checker = HealthChecker(
            codebase_root=tmp_path,
            dafny_bridge=mock_dafny,
            llm=mock_llm,
        )
        proposal = _make_proposal("modify_contract")
        result = await checker.check([str(f)], proposal=proposal)
        assert result.healthy

    @pytest.mark.asyncio
    async def test_dafny_failure_blocks(self, tmp_path: Path):
        """Dafny failure blocks the proposal for triggerable categories."""
        f = tmp_path / "contract.py"
        f.write_text("x = 1\n", encoding="utf-8")

        mock_dafny = AsyncMock()
        mock_dafny.check_available = AsyncMock(return_value=True)
        mock_dafny.run_clover_loop = AsyncMock(return_value=DafnyVerificationResult(
            status=DafnyVerificationStatus.FAILED,
            rounds_attempted=8,
            error_summary="Postcondition might not hold",
        ))
        mock_llm = AsyncMock()

        checker = HealthChecker(
            codebase_root=tmp_path,
            dafny_bridge=mock_dafny,
            llm=mock_llm,
        )
        proposal = _make_proposal("modify_contract")
        result = await checker.check([str(f)], proposal=proposal)
        assert not result.healthy
        assert any("Dafny" in issue for issue in result.issues)


# ── Health Check with Z3 ─────────────────────────────────────────────────────


class TestHealthCheckZ3:
    @pytest.mark.asyncio
    async def test_z3_advisory_does_not_block(self, tmp_path: Path):
        """Z3 results are advisory — even with valid invariants, it passes."""
        f = tmp_path / "code.py"
        f.write_text("def f(x): return x * x\n", encoding="utf-8")

        mock_z3 = AsyncMock()
        mock_z3.run_discovery_loop = AsyncMock(return_value=InvariantVerificationResult(
            status=InvariantVerificationStatus.VALID,
            rounds_attempted=1,
            valid_invariants=[
                DiscoveredInvariant(
                    kind=InvariantKind.POSTCONDITION,
                    expression="result >= 0",
                    z3_expression="result >= 0",
                    status=InvariantVerificationStatus.VALID,
                ),
            ],
        ))
        mock_llm = AsyncMock()

        checker = HealthChecker(
            codebase_root=tmp_path,
            z3_bridge=mock_z3,
            llm=mock_llm,
        )
        proposal = _make_proposal("add_executor")
        result = await checker.check([str(f)], proposal=proposal)
        assert result.healthy
        assert result.formal_verification is not None
        assert len(result.formal_verification.advisory_issues) > 0


# ── Combined Verification ────────────────────────────────────────────────────


class TestCombinedVerification:
    @pytest.mark.asyncio
    async def test_all_bridges_pass(self, tmp_path: Path):
        """All verification bridges pass → healthy."""
        f = tmp_path / "code.py"
        f.write_text("def f(x): return x\n", encoding="utf-8")

        mock_dafny = AsyncMock()
        mock_dafny.check_available = AsyncMock(return_value=True)
        mock_dafny.run_clover_loop = AsyncMock(return_value=DafnyVerificationResult(
            status=DafnyVerificationStatus.VERIFIED, rounds_attempted=1,
        ))
        mock_z3 = AsyncMock()
        mock_z3.run_discovery_loop = AsyncMock(return_value=InvariantVerificationResult(
            status=InvariantVerificationStatus.VALID, rounds_attempted=1,
        ))
        mock_sa = AsyncMock()
        mock_sa.run_all = AsyncMock(return_value=StaticAnalysisResult())
        mock_llm = AsyncMock()

        checker = HealthChecker(
            codebase_root=tmp_path,
            dafny_bridge=mock_dafny,
            z3_bridge=mock_z3,
            static_analysis_bridge=mock_sa,
            llm=mock_llm,
        )
        proposal = _make_proposal("modify_contract")
        result = await checker.check([str(f)], proposal=proposal)
        assert result.healthy
        fv = result.formal_verification
        assert fv is not None
        assert fv.passed is True

    @pytest.mark.asyncio
    async def test_static_failure_overrides_dafny_success(self, tmp_path: Path):
        """Static analysis ERROR blocks even if Dafny passes."""
        f = tmp_path / "code.py"
        f.write_text("import pickle\n", encoding="utf-8")

        mock_dafny = AsyncMock()
        mock_dafny.check_available = AsyncMock(return_value=True)
        mock_dafny.run_clover_loop = AsyncMock(return_value=DafnyVerificationResult(
            status=DafnyVerificationStatus.VERIFIED, rounds_attempted=1,
        ))
        mock_sa = AsyncMock()
        mock_sa.run_all = AsyncMock(return_value=StaticAnalysisResult(
            findings=[
                StaticAnalysisFinding(
                    tool="bandit", severity=StaticAnalysisSeverity.ERROR,
                    message="Pickle", file_path="code.py", line=1,
                ),
            ],
            error_count=1,
        ))
        mock_llm = AsyncMock()

        checker = HealthChecker(
            codebase_root=tmp_path,
            dafny_bridge=mock_dafny,
            static_analysis_bridge=mock_sa,
            llm=mock_llm,
        )
        proposal = _make_proposal("modify_contract")
        result = await checker.check([str(f)], proposal=proposal)
        assert not result.healthy


# ── Type Verification ─────────────────────────────────────────────────────────


class TestVerificationTypes:
    def test_formal_verification_result_defaults(self):
        result = FormalVerificationResult()
        assert result.passed is True
        assert result.blocking_issues == []
        assert result.advisory_issues == []
        assert result.dafny is None
        assert result.z3 is None
        assert result.static_analysis is None

    def test_dafny_triggerable_categories(self):
        from systems.simula.evolution_types import ChangeCategory
        assert ChangeCategory.MODIFY_CONTRACT in DAFNY_TRIGGERABLE_CATEGORIES
        assert ChangeCategory.ADD_SYSTEM_CAPABILITY in DAFNY_TRIGGERABLE_CATEGORIES
        assert ChangeCategory.ADD_EXECUTOR not in DAFNY_TRIGGERABLE_CATEGORIES

    def test_dafny_verification_result_serialization(self):
        result = DafnyVerificationResult(
            status=DafnyVerificationStatus.VERIFIED,
            rounds_attempted=3,
            final_spec="method M() {}",
        )
        data = result.model_dump()
        assert data["status"] == "verified"
        assert data["rounds_attempted"] == 3

    def test_static_analysis_result_serialization(self):
        result = StaticAnalysisResult(
            findings=[
                StaticAnalysisFinding(
                    tool="bandit", rule_id="B301",
                    severity=StaticAnalysisSeverity.ERROR,
                    message="test",
                ),
            ],
            error_count=1,
        )
        data = result.model_dump()
        assert len(data["findings"]) == 1
        assert data["error_count"] == 1
