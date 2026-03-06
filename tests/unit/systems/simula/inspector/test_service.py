"""
Unit tests for InspectorService (Phase 7).

Tests the full hunting pipeline orchestrator: external repo hunting,
internal EOS hunting, PoC validation, metrics, history, and analytics
integration. All sub-systems (ingestor, prover, remediation) are mocked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.simula.inspector.service import (
    _FORBIDDEN_POC_MODULES,
    PREDEFINED_ATTACK_GOALS,
    InspectorService,
)
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    HuntResult,
    InspectorConfig,
    RemediationResult,
    RemediationStatus,
    TargetType,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from systems.simula.inspector.workspace import TargetWorkspace

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_surface(**overrides) -> AttackSurface:
    defaults = dict(
        entry_point="get_user",
        surface_type=AttackSurfaceType.API_ENDPOINT,
        file_path="app/routes.py",
        line_number=10,
        context_code="def get_user(id): return db.query(id)",
        http_method="GET",
        route_pattern="/api/user/{id}",
    )
    defaults.update(overrides)
    return AttackSurface(**defaults)


def _make_vuln_report(
    target_url: str = "https://github.com/test/repo",
    severity: VulnerabilitySeverity = VulnerabilitySeverity.HIGH,
    vuln_class: VulnerabilityClass = VulnerabilityClass.BROKEN_ACCESS_CONTROL,
) -> VulnerabilityReport:
    return VulnerabilityReport(
        target_url=target_url,
        vulnerability_class=vuln_class,
        severity=severity,
        attack_surface=_make_surface(),
        attack_goal="User A can access User B's data",
        z3_counterexample="is_authenticated=True, user_id=999",
    )


def _make_prover(vuln: VulnerabilityReport | None = None) -> MagicMock:
    prover = MagicMock()
    prover.prove_vulnerability = AsyncMock(return_value=vuln)
    return prover


def _make_config(**overrides) -> InspectorConfig:
    defaults = dict(
        authorized_targets=["localhost", "example.com"],
        max_workers=2,
        sandbox_timeout_seconds=10,
        log_vulnerability_analytics=True,
        clone_depth=1,
    )
    defaults.update(overrides)
    return InspectorConfig(**defaults)


def _make_analytics() -> MagicMock:
    analytics = MagicMock()
    analytics.flush = AsyncMock()
    analytics.emit_hunt_started = MagicMock()
    analytics.emit_attack_surface_discovered = MagicMock()
    analytics.emit_vulnerability_proved = MagicMock()
    analytics.emit_poc_generated = MagicMock()
    analytics.emit_patch_generated = MagicMock()
    analytics.emit_hunt_completed = MagicMock()
    analytics.emit_hunt_error = MagicMock()
    analytics.emit_surface_mapping_failed = MagicMock()
    analytics.emit_proof_timeout = MagicMock()
    analytics.stats = {"events_emitted": 0, "events_failed": 0}
    return analytics


def _make_remediation() -> MagicMock:
    remediation = MagicMock()
    remediation.set_workspace = MagicMock()
    remediation.generate_patches_batch = AsyncMock(return_value={})
    return remediation


def _make_service(
    *,
    prover: MagicMock | None = None,
    config: InspectorConfig | None = None,
    analytics: MagicMock | None = None,
    remediation: MagicMock | None = None,
    eos_root: Path | None = None,
) -> InspectorService:
    return InspectorService(
        prover=prover or _make_prover(),
        config=config or _make_config(),
        eos_root=eos_root,
        analytics=analytics,
        remediation=remediation,
    )


# ── Predefined Attack Goals ─────────────────────────────────────────────────


class TestPredefinedGoals:
    def test_has_seven_goals(self):
        assert len(PREDEFINED_ATTACK_GOALS) == 7

    def test_covers_key_vulnerability_classes(self):
        all_goals = " ".join(PREDEFINED_ATTACK_GOALS).lower()
        assert "unauthenticated" in all_goals
        assert "sql injection" in all_goals
        assert "access control" in all_goals
        assert "privilege escalation" in all_goals
        assert "reentrancy" in all_goals
        assert "redirect" in all_goals
        assert "race condition" in all_goals


# ── PoC Validation ──────────────────────────────────────────────────────────


class TestValidatePoc:
    def test_valid_poc_passes(self):
        service = _make_service()
        poc = "import requests\nrequests.get('http://localhost:8000/api/user/999')"
        assert service.validate_poc(poc) is True

    def test_unauthorized_target_rejected(self):
        service = _make_service(config=_make_config(authorized_targets=["allowed.com"]))
        poc = "import requests\nrequests.get('http://evil.com/api')"
        assert service.validate_poc(poc, authorized_target="evil.com") is False

    def test_authorized_target_accepted(self):
        service = _make_service(config=_make_config(authorized_targets=["allowed.com"]))
        poc = "import requests\nrequests.get('http://allowed.com/api')"
        assert service.validate_poc(poc, authorized_target="allowed.com") is True

    def test_syntax_error_rejected(self):
        service = _make_service()
        poc = "def foo(\n    # broken"
        assert service.validate_poc(poc) is False

    @pytest.mark.parametrize("module", list(_FORBIDDEN_POC_MODULES)[:5])
    def test_forbidden_modules_rejected(self, module: str):
        service = _make_service()
        poc = f"import {module}\n{module}.do_something()"
        assert service.validate_poc(poc) is False

    def test_from_import_forbidden_module_rejected(self):
        service = _make_service()
        poc = "from subprocess import call\ncall(['ls'])"
        assert service.validate_poc(poc) is False


# ── External Repo Hunting ───────────────────────────────────────────────────


class TestHuntExternalRepo:
    @pytest.mark.asyncio
    async def test_successful_hunt_returns_result(self, tmp_path: Path):
        """Full pipeline: clone → map → prove → report."""
        vuln = _make_vuln_report()
        prover = _make_prover(vuln=vuln)
        analytics = _make_analytics()

        service = _make_service(prover=prover, analytics=analytics)

        surfaces = [_make_surface(), _make_surface(entry_point="create_user")]

        mock_ingestor = MagicMock()
        mock_ingestor.workspace = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        mock_ingestor.map_attack_surfaces = AsyncMock(return_value=surfaces)
        mock_ingestor.extract_context_code = AsyncMock(return_value="")

        with patch(
            "systems.simula.inspector.service.TargetIngestor.ingest_from_github",
            AsyncMock(return_value=mock_ingestor),
        ):
            result = await service.hunt_external_repo(
                "https://github.com/test/repo",
                attack_goals=["SQL injection"],
            )

        assert isinstance(result, HuntResult)
        assert result.target_url == "https://github.com/test/repo"
        assert result.target_type == TargetType.EXTERNAL_REPO
        assert result.surfaces_mapped == 2
        assert result.completed_at is not None

        # Analytics emitted
        analytics.emit_hunt_started.assert_called_once()
        analytics.emit_hunt_completed.assert_called_once()
        analytics.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_clone_failure_returns_empty_result(self):
        """If git clone fails, return an empty HuntResult (no crash)."""
        service = _make_service()

        with patch(
            "systems.simula.inspector.service.TargetIngestor.ingest_from_github",
            AsyncMock(side_effect=RuntimeError("git clone failed")),
        ):
            result = await service.hunt_external_repo(
                "https://github.com/invalid/repo",
            )

        assert isinstance(result, HuntResult)
        assert result.surfaces_mapped == 0
        assert result.vulnerability_count == 0

    @pytest.mark.asyncio
    async def test_no_surfaces_returns_empty_result(self, tmp_path: Path):
        """If no attack surfaces found, return empty result."""
        service = _make_service()

        mock_ingestor = MagicMock()
        mock_ingestor.workspace = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        mock_ingestor.map_attack_surfaces = AsyncMock(return_value=[])

        with patch(
            "systems.simula.inspector.service.TargetIngestor.ingest_from_github",
            AsyncMock(return_value=mock_ingestor),
        ):
            result = await service.hunt_external_repo(
                "https://github.com/test/empty-repo",
            )

        assert result.surfaces_mapped == 0
        assert result.vulnerability_count == 0

    @pytest.mark.asyncio
    async def test_workspace_cleaned_up_on_completion(self, tmp_path: Path):
        """Workspace temp directory should be cleaned up after hunt."""
        service = _make_service()

        mock_ws = MagicMock(spec=TargetWorkspace)
        mock_ws.root = tmp_path
        mock_ws.workspace_type = "external_repo"
        mock_ws.cleanup = MagicMock()

        mock_ingestor = MagicMock()
        mock_ingestor.workspace = mock_ws
        mock_ingestor.map_attack_surfaces = AsyncMock(return_value=[])

        with patch(
            "systems.simula.inspector.service.TargetIngestor.ingest_from_github",
            AsyncMock(return_value=mock_ingestor),
        ):
            await service.hunt_external_repo("https://github.com/test/repo")

        mock_ws.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_cleaned_up_on_error(self, tmp_path: Path):
        """Workspace should be cleaned up even if pipeline raises."""
        service = _make_service()

        mock_ws = MagicMock(spec=TargetWorkspace)
        mock_ws.root = tmp_path
        mock_ws.workspace_type = "external_repo"
        mock_ws.cleanup = MagicMock()

        mock_ingestor = MagicMock()
        mock_ingestor.workspace = mock_ws
        mock_ingestor.map_attack_surfaces = AsyncMock(
            side_effect=RuntimeError("Scanner crash"),
        )

        with patch(
            "systems.simula.inspector.service.TargetIngestor.ingest_from_github",
            AsyncMock(return_value=mock_ingestor),
        ):
            await service.hunt_external_repo("https://github.com/test/repo")

        mock_ws.cleanup.assert_called_once()


# ── Internal EOS Hunting ─────────────────────────────────────────────────────


class TestHuntInternalEOS:
    @pytest.mark.asyncio
    async def test_requires_eos_root(self):
        """Should raise RuntimeError if eos_root not provided."""
        service = _make_service(eos_root=None)
        with pytest.raises(RuntimeError, match="eos_root"):
            await service.hunt_internal_eos()

    @pytest.mark.asyncio
    async def test_internal_hunt_uses_internal_workspace(self, tmp_path: Path):
        """Internal hunt should use TargetWorkspace.internal()."""
        vuln = _make_vuln_report(target_url="internal_eos")
        prover = _make_prover(vuln=vuln)
        service = _make_service(prover=prover, eos_root=tmp_path)

        with patch.object(
            service, "_run_hunt_pipeline",
            AsyncMock(return_value=HuntResult(
                target_url="internal_eos",
                target_type=TargetType.INTERNAL_EOS,
            )),
        ) as mock_pipeline:
            result = await service.hunt_internal_eos()

            assert result.target_url == "internal_eos"
            # Verify pipeline was called with internal workspace
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["target_type"] == TargetType.INTERNAL_EOS
            assert call_kwargs["target_url"] == "internal_eos"


# ── Patch Generation ────────────────────────────────────────────────────────


class TestGeneratePatches:
    @pytest.mark.asyncio
    async def test_requires_remediation(self):
        """Should raise RuntimeError if remediation is not available."""
        service = _make_service(remediation=None)
        result = HuntResult(
            target_url="url",
            target_type=TargetType.EXTERNAL_REPO,
            vulnerabilities_found=[_make_vuln_report()],
        )
        with pytest.raises(RuntimeError, match="Remediation"):
            await service.generate_patches(result)

    @pytest.mark.asyncio
    async def test_empty_vulnerabilities_returns_empty(self):
        """No vulnerabilities → empty patches dict."""
        remediation = _make_remediation()
        service = _make_service(remediation=remediation)
        result = HuntResult(
            target_url="url",
            target_type=TargetType.EXTERNAL_REPO,
        )
        patches = await service.generate_patches(result)
        assert patches == {}

    @pytest.mark.asyncio
    async def test_passes_vulnerabilities_to_remediation(self):
        """Should delegate to remediation.generate_patches_batch."""
        vuln = _make_vuln_report()
        remediation = _make_remediation()
        remediation.generate_patches_batch = AsyncMock(return_value={
            vuln.id: RemediationResult(
                vulnerability_id=vuln.id,
                status=RemediationStatus.PATCHED,
                final_patch_diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1,2 @@\n+# fixed",
            ),
        })
        analytics = _make_analytics()

        service = _make_service(remediation=remediation, analytics=analytics)
        result = HuntResult(
            target_url="url",
            target_type=TargetType.EXTERNAL_REPO,
            vulnerabilities_found=[vuln],
        )

        patches = await service.generate_patches(result)

        assert vuln.id in patches
        assert "fixed" in patches[vuln.id]
        remediation.generate_patches_batch.assert_called_once()
        analytics.emit_patch_generated.assert_called_once()


# ── Metrics & History ────────────────────────────────────────────────────────


class TestMetricsAndHistory:
    def test_initial_stats(self):
        service = _make_service()
        stats = service.stats
        assert stats["hunts_completed"] == 0
        assert stats["total_surfaces_mapped"] == 0
        assert stats["total_vulnerabilities_found"] == 0
        assert stats["total_patches_generated"] == 0
        assert stats["config"]["max_workers"] == 2

    def test_empty_history(self):
        service = _make_service()
        assert service.get_hunt_history() == []

    def test_analytics_view_initialized(self):
        service = _make_service()
        summary = service.analytics_view.summary
        assert summary["total_vulnerabilities"] == 0
        assert summary["total_hunts"] == 0


# ── Concurrent Proving ──────────────────────────────────────────────────────


class TestConcurrentProving:
    @pytest.mark.asyncio
    async def test_bounded_concurrency(self, tmp_path: Path):
        """Verify that concurrent workers are bounded by max_workers."""
        import asyncio

        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def counting_prove(surface, attack_goal, **kwargs):
            nonlocal active_count, max_active
            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)
            await asyncio.sleep(0.01)
            async with lock:
                active_count -= 1
            return None

        prover = MagicMock()
        prover.prove_vulnerability = AsyncMock(side_effect=counting_prove)

        config = _make_config(max_workers=2, sandbox_timeout_seconds=5)
        service = _make_service(prover=prover, config=config)

        # 6 surfaces × 1 goal = 6 tasks, but only 2 should run concurrently
        surfaces = [_make_surface(entry_point=f"func_{i}") for i in range(6)]

        # Call _prove_all directly
        log = MagicMock()
        log.info = MagicMock()
        log.warning = MagicMock()

        await service._prove_all(
            surfaces=surfaces,
            goals=["SQL injection"],
            target_url="url",
            generate_pocs=False,
            log=log,
        )

        assert max_active <= 2
