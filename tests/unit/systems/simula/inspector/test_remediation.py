"""
Unit tests for Inspector Autonomous Remediation (Phase 6).

Tests InspectorRepairOrchestrator: patch generation, Z3 re-verification loop,
cost budget enforcement, timeout handling, and diff generation.
All RepairAgent and VulnerabilityProver calls are mocked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from systems.simula.inspector.remediation import InspectorRepairOrchestrator
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    RemediationStatus,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from systems.simula.inspector.workspace import TargetWorkspace
from systems.simula.verification.types import RepairResult, RepairStatus

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_surface(file_path: str = "app/routes.py") -> AttackSurface:
    return AttackSurface(
        entry_point="get_user",
        surface_type=AttackSurfaceType.API_ENDPOINT,
        file_path=file_path,
        line_number=10,
        context_code="def get_user(id): return db.query(id)",
        http_method="GET",
        route_pattern="/api/user/{id}",
    )


def _make_vulnerability(
    file_path: str = "app/routes.py",
    vuln_class: VulnerabilityClass = VulnerabilityClass.BROKEN_ACCESS_CONTROL,
    severity: VulnerabilitySeverity = VulnerabilitySeverity.HIGH,
) -> VulnerabilityReport:
    return VulnerabilityReport(
        target_url="https://github.com/test/repo",
        vulnerability_class=vuln_class,
        severity=severity,
        attack_surface=_make_surface(file_path),
        attack_goal="User A can access User B's data",
        z3_counterexample="is_authenticated=True, requested_user_id=999",
    )


def _make_repair_agent(
    status: RepairStatus = RepairStatus.REPAIRED,
    cost: float = 0.05,
) -> MagicMock:
    agent = MagicMock()
    result = RepairResult(
        status=status,
        total_cost_usd=cost,
        fix_summary="Added authorization check",
        diagnosis_summary="Missing access control",
    )
    agent.repair = AsyncMock(return_value=result)
    return agent


def _make_prover(returns_none: bool = True) -> MagicMock:
    """
    Mock prover for re-verification.
    returns_none=True means UNSAT (vulnerability eliminated).
    returns_none=False means SAT (vulnerability still present).
    """
    prover = MagicMock()
    if returns_none:
        prover.prove_vulnerability = AsyncMock(return_value=None)
    else:
        prover.prove_vulnerability = AsyncMock(return_value=_make_vulnerability())
    return prover


def _setup_workspace_with_file(
    tmp_path: Path,
    file_path: str = "app/routes.py",
    content: str = "def get_user(id): return db.query(id)\n",
) -> TargetWorkspace:
    """Create a workspace with a vulnerable source file on disk."""
    full_path = tmp_path / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return TargetWorkspace(root=tmp_path, workspace_type="external_repo")


# ── Single Patch Generation ──────────────────────────────────────────────────


class TestGeneratePatch:
    @pytest.mark.asyncio
    async def test_successful_patch_verified(self, tmp_path: Path):
        """
        RepairAgent produces a fix → prover re-verifies (UNSAT) → PATCHED.
        """
        workspace = _setup_workspace_with_file(tmp_path)
        repair_agent = _make_repair_agent(status=RepairStatus.REPAIRED)
        prover = _make_prover(returns_none=True)  # UNSAT = vulnerability eliminated

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=workspace,
        )

        vuln = _make_vulnerability()
        result = await orchestrator.generate_patch(vuln)

        assert result.status == RemediationStatus.PATCHED
        assert result.vulnerability_id == vuln.id
        assert result.successful_attempt is not None
        assert result.total_attempts >= 1
        repair_agent.repair.assert_called()
        prover.prove_vulnerability.assert_called()

    @pytest.mark.asyncio
    async def test_repair_agent_fails_returns_failed(self, tmp_path: Path):
        """
        RepairAgent cannot produce a fix → FAILED.
        """
        workspace = _setup_workspace_with_file(tmp_path)
        repair_agent = _make_repair_agent(status=RepairStatus.FAILED)
        prover = _make_prover()

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=workspace,
            max_retries=2,
        )

        vuln = _make_vulnerability()
        result = await orchestrator.generate_patch(vuln)

        assert result.status == RemediationStatus.FAILED
        assert result.total_attempts == 2  # Tried both retries
        prover.prove_vulnerability.assert_not_called()  # Never got to re-verification

    @pytest.mark.asyncio
    async def test_reverification_sat_retries(self, tmp_path: Path):
        """
        RepairAgent fixes, but Z3 re-verification shows vulnerability persists.
        Orchestrator retries, second attempt succeeds.
        """
        workspace = _setup_workspace_with_file(tmp_path)
        repair_agent = _make_repair_agent(status=RepairStatus.REPAIRED)

        # First reverification returns SAT (still vulnerable), second returns UNSAT
        vuln_report = _make_vulnerability()
        prover = MagicMock()
        prover.prove_vulnerability = AsyncMock(
            side_effect=[vuln_report, None],  # SAT then UNSAT
        )

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=workspace,
            max_retries=2,
        )

        vuln = _make_vulnerability()
        result = await orchestrator.generate_patch(vuln)

        assert result.status == RemediationStatus.PATCHED
        assert result.successful_attempt == 1  # Second attempt (0-indexed)

    @pytest.mark.asyncio
    async def test_file_not_found_returns_failed(self, tmp_path: Path):
        """If the vulnerable file doesn't exist in workspace, return FAILED."""
        workspace = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        repair_agent = _make_repair_agent()
        prover = _make_prover()

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=workspace,
        )

        # File path doesn't exist on disk
        vuln = _make_vulnerability(file_path="nonexistent/module.py")
        result = await orchestrator.generate_patch(vuln)

        assert result.status == RemediationStatus.FAILED
        assert any("not found" in a.error.lower() for a in result.attempts)

    @pytest.mark.asyncio
    async def test_exception_during_repair_caught(self, tmp_path: Path):
        """Exceptions in the repair loop should be caught, not crash."""
        workspace = _setup_workspace_with_file(tmp_path)
        repair_agent = MagicMock()
        repair_agent.repair = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        prover = _make_prover()

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=workspace,
            max_retries=1,
        )

        vuln = _make_vulnerability()
        result = await orchestrator.generate_patch(vuln)

        assert result.status == RemediationStatus.FAILED
        assert any("error" in a.error.lower() for a in result.attempts)


# ── Batch Remediation ───────────────────────────────────────────────────────


class TestBatchRemediation:
    @pytest.mark.asyncio
    async def test_batch_processes_all(self, tmp_path: Path):
        workspace = _setup_workspace_with_file(tmp_path)
        repair_agent = _make_repair_agent(status=RepairStatus.REPAIRED)
        prover = _make_prover(returns_none=True)

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=workspace,
        )

        vulns = [
            _make_vulnerability(vuln_class=VulnerabilityClass.SQL_INJECTION, severity=VulnerabilitySeverity.CRITICAL),
            _make_vulnerability(vuln_class=VulnerabilityClass.XSS, severity=VulnerabilitySeverity.MEDIUM),
        ]

        results = await orchestrator.generate_patches_batch(vulns)

        assert len(results) == 2
        assert all(r.status == RemediationStatus.PATCHED for r in results.values())

    @pytest.mark.asyncio
    async def test_batch_processes_critical_first(self, tmp_path: Path):
        """Vulnerabilities should be processed in severity order (CRITICAL first)."""
        workspace = _setup_workspace_with_file(tmp_path)

        call_order: list[str] = []

        async def tracking_repair(proposal, broken_files, test_output=""):
            severity_str = proposal.description.split("(")[1].split(")")[0] if "(" in proposal.description else "?"
            call_order.append(severity_str)
            return RepairResult(status=RepairStatus.FAILED)

        repair_agent = MagicMock()
        repair_agent.repair = AsyncMock(side_effect=tracking_repair)
        prover = _make_prover()

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=workspace,
            max_retries=1,
        )

        vulns = [
            _make_vulnerability(severity=VulnerabilitySeverity.LOW),
            _make_vulnerability(severity=VulnerabilitySeverity.CRITICAL),
            _make_vulnerability(severity=VulnerabilitySeverity.HIGH),
        ]

        await orchestrator.generate_patches_batch(vulns)

        # Verify critical was processed first
        assert "critical" in call_order[0]


# ── Workspace Management ────────────────────────────────────────────────────


class TestWorkspaceManagement:
    def test_set_workspace(self, tmp_path: Path):
        ws1 = _setup_workspace_with_file(tmp_path / "ws1")
        ws2 = _setup_workspace_with_file(tmp_path / "ws2")
        repair_agent = _make_repair_agent()
        prover = _make_prover()

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=ws1,
        )
        assert orchestrator.workspace.root == ws1.root

        orchestrator.set_workspace(ws2)
        assert orchestrator.workspace.root == ws2.root


# ── Synthetic Proposal Building ──────────────────────────────────────────────


class TestSyntheticProposal:
    def test_proposal_contains_vulnerability_info(self, tmp_path: Path):
        workspace = _setup_workspace_with_file(tmp_path)
        repair_agent = _make_repair_agent()
        prover = _make_prover()

        orchestrator = InspectorRepairOrchestrator(
            repair_agent=repair_agent,
            prover=prover,
            workspace=workspace,
        )

        vuln = _make_vulnerability()
        proposal = orchestrator._build_synthetic_proposal(vuln)

        assert proposal.source == "inspector"
        assert "broken_access_control" in proposal.description
        assert "high" in proposal.description
        assert vuln.attack_surface.file_path in proposal.description


# ── Diff Generation ─────────────────────────────────────────────────────────


class TestDiffGeneration:
    def test_generates_unified_diff(self):
        original = "def get_user(id):\n    return db.query(id)\n"
        patched = "def get_user(id, current_user):\n    if id != current_user.id:\n        raise Forbidden()\n    return db.query(id)\n"

        diff = InspectorRepairOrchestrator._generate_diff(
            "app/routes.py", original, patched,
        )

        assert "---" in diff
        assert "+++" in diff
        assert "a/app/routes.py" in diff
        assert "b/app/routes.py" in diff
        assert "+    if id != current_user.id:" in diff

    def test_empty_diff_for_identical(self):
        code = "def foo(): pass\n"
        diff = InspectorRepairOrchestrator._generate_diff("f.py", code, code)
        assert diff == ""
