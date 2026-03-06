"""
Unit tests for Inspector domain types and validation logic.

Tests all data models, enums, field validators, and computed properties
from ``systems.simula.inspector.types``.
"""

from __future__ import annotations

import pytest

from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    HuntResult,
    InspectorConfig,
    RemediationAttempt,
    RemediationResult,
    RemediationStatus,
    TargetType,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)

# ── Enum Tests ──────────────────────────────────────────────────────────────


class TestTargetType:
    def test_values(self):
        assert TargetType.INTERNAL_EOS == "internal_eos"
        assert TargetType.EXTERNAL_REPO == "external_repo"

    def test_all_members(self):
        assert len(TargetType) == 2


class TestAttackSurfaceType:
    def test_all_12_surface_types(self):
        assert len(AttackSurfaceType) == 12
        assert AttackSurfaceType.API_ENDPOINT == "api_endpoint"
        assert AttackSurfaceType.MIDDLEWARE == "middleware"
        assert AttackSurfaceType.SMART_CONTRACT_PUBLIC == "smart_contract_public"
        assert AttackSurfaceType.FUNCTION_EXPORT == "function_export"
        assert AttackSurfaceType.CLI_COMMAND == "cli_command"
        assert AttackSurfaceType.WEBSOCKET_HANDLER == "websocket_handler"
        assert AttackSurfaceType.GRAPHQL_RESOLVER == "graphql_resolver"
        assert AttackSurfaceType.EVENT_HANDLER == "event_handler"
        assert AttackSurfaceType.DATABASE_QUERY == "database_query"
        assert AttackSurfaceType.FILE_UPLOAD == "file_upload"
        assert AttackSurfaceType.AUTH_HANDLER == "auth_handler"
        assert AttackSurfaceType.DESERIALIZATION == "deserialization"


class TestVulnerabilitySeverity:
    def test_values(self):
        assert VulnerabilitySeverity.LOW == "low"
        assert VulnerabilitySeverity.MEDIUM == "medium"
        assert VulnerabilitySeverity.HIGH == "high"
        assert VulnerabilitySeverity.CRITICAL == "critical"

    def test_all_4_levels(self):
        assert len(VulnerabilitySeverity) == 4


class TestVulnerabilityClass:
    def test_has_15_classes(self):
        assert len(VulnerabilityClass) == 16  # 15 specific + OTHER

    def test_owasp_top_classes_present(self):
        important = {
            "broken_authentication",
            "broken_access_control",
            "injection",
            "sql_injection",
            "cross_site_scripting",
            "privilege_escalation",
            "reentrancy",
        }
        values = {m.value for m in VulnerabilityClass}
        assert important.issubset(values)


class TestRemediationStatus:
    def test_all_terminal_statuses(self):
        expected = {"patched", "patch_unverified", "failed", "timeout", "budget_exceeded", "skipped"}
        actual = {s.value for s in RemediationStatus}
        assert actual == expected


# ── AttackSurface Tests ─────────────────────────────────────────────────────


class TestAttackSurface:
    def test_minimal_construction(self):
        surface = AttackSurface(
            entry_point="get_user",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="app/routes.py",
        )
        assert surface.entry_point == "get_user"
        assert surface.surface_type == AttackSurfaceType.API_ENDPOINT
        assert surface.file_path == "app/routes.py"
        assert surface.id  # auto-generated ULID
        assert surface.discovered_at is not None

    def test_full_construction(self):
        surface = AttackSurface(
            entry_point="get_user",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="app/routes.py",
            line_number=42,
            context_code="def get_user(id): ...",
            http_method="GET",
            route_pattern="/api/user/{id}",
        )
        assert surface.line_number == 42
        assert surface.http_method == "GET"
        assert surface.route_pattern == "/api/user/{id}"

    def test_optional_fields_default_to_none(self):
        surface = AttackSurface(
            entry_point="handler",
            surface_type=AttackSurfaceType.MIDDLEWARE,
            file_path="middleware.py",
        )
        assert surface.line_number is None
        assert surface.http_method is None
        assert surface.route_pattern is None
        assert surface.context_code == ""


# ── VulnerabilityReport Tests ───────────────────────────────────────────────


def _make_surface() -> AttackSurface:
    return AttackSurface(
        entry_point="get_user",
        surface_type=AttackSurfaceType.API_ENDPOINT,
        file_path="app/routes.py",
        line_number=10,
        context_code="def get_user(id): return db.query(id)",
        http_method="GET",
        route_pattern="/api/user/{id}",
    )


class TestVulnerabilityReport:
    def test_minimal_construction(self):
        report = VulnerabilityReport(
            target_url="https://github.com/example/repo",
            vulnerability_class=VulnerabilityClass.BROKEN_ACCESS_CONTROL,
            severity=VulnerabilitySeverity.HIGH,
            attack_surface=_make_surface(),
            attack_goal="User A can access User B's data",
            z3_counterexample="is_authenticated=True, requested_user_id=999",
        )
        assert report.id
        assert report.target_url == "https://github.com/example/repo"
        assert report.severity == VulnerabilitySeverity.HIGH
        assert not report.verified
        assert report.proof_of_concept_code == ""

    def test_severity_validator_accepts_valid(self):
        for severity in VulnerabilitySeverity:
            report = VulnerabilityReport(
                target_url="internal_eos",
                vulnerability_class=VulnerabilityClass.OTHER,
                severity=severity,
                attack_surface=_make_surface(),
                attack_goal="test",
                z3_counterexample="x=1",
            )
            assert report.severity == severity


# ── HuntResult Tests ────────────────────────────────────────────────────────


class TestHuntResult:
    def test_empty_result(self):
        result = HuntResult(
            target_url="https://github.com/test/repo",
            target_type=TargetType.EXTERNAL_REPO,
        )
        assert result.vulnerability_count == 0
        assert result.critical_count == 0
        assert result.high_count == 0
        assert result.surfaces_mapped == 0
        assert result.total_duration_ms == 0
        assert result.generated_patches == {}

    def test_with_vulnerabilities(self):
        vulns = [
            VulnerabilityReport(
                target_url="url",
                vulnerability_class=VulnerabilityClass.SQL_INJECTION,
                severity=VulnerabilitySeverity.CRITICAL,
                attack_surface=_make_surface(),
                attack_goal="SQL injection",
                z3_counterexample="x=1",
            ),
            VulnerabilityReport(
                target_url="url",
                vulnerability_class=VulnerabilityClass.XSS,
                severity=VulnerabilitySeverity.HIGH,
                attack_surface=_make_surface(),
                attack_goal="XSS",
                z3_counterexample="x=2",
            ),
            VulnerabilityReport(
                target_url="url",
                vulnerability_class=VulnerabilityClass.UNVALIDATED_REDIRECT,
                severity=VulnerabilitySeverity.LOW,
                attack_surface=_make_surface(),
                attack_goal="Redirect",
                z3_counterexample="x=3",
            ),
        ]
        result = HuntResult(
            target_url="url",
            target_type=TargetType.EXTERNAL_REPO,
            surfaces_mapped=5,
            vulnerabilities_found=vulns,
        )
        assert result.vulnerability_count == 3
        assert result.critical_count == 1
        assert result.high_count == 1


# ── InspectorConfig Tests ──────────────────────────────────────────────────────


class TestInspectorConfig:
    def test_defaults(self):
        config = InspectorConfig()
        assert config.authorized_targets == []
        assert config.max_workers == 4
        assert config.sandbox_timeout_seconds == 30
        assert config.log_vulnerability_analytics is True
        assert config.clone_depth == 1

    def test_max_workers_bounds(self):
        config = InspectorConfig(max_workers=1)
        assert config.max_workers == 1

        config = InspectorConfig(max_workers=16)
        assert config.max_workers == 16

        with pytest.raises(Exception):
            InspectorConfig(max_workers=0)

        with pytest.raises(Exception):
            InspectorConfig(max_workers=17)

    def test_sandbox_timeout_must_be_positive(self):
        with pytest.raises(Exception):
            InspectorConfig(sandbox_timeout_seconds=0)

        with pytest.raises(Exception):
            InspectorConfig(sandbox_timeout_seconds=-1)

    def test_authorized_targets_rejects_empty_strings(self):
        with pytest.raises(Exception):
            InspectorConfig(authorized_targets=["valid.com", ""])

        with pytest.raises(Exception):
            InspectorConfig(authorized_targets=["   "])

    def test_authorized_targets_accepts_valid(self):
        config = InspectorConfig(authorized_targets=["example.com", "test.local"])
        assert len(config.authorized_targets) == 2

    def test_clone_depth_minimum(self):
        with pytest.raises(Exception):
            InspectorConfig(clone_depth=0)


# ── Remediation Types Tests ─────────────────────────────────────────────────


class TestRemediationAttempt:
    def test_defaults(self):
        attempt = RemediationAttempt()
        assert attempt.attempt_number == 0
        assert attempt.patch_diff == ""
        assert attempt.patched_code == ""
        assert attempt.vulnerability_eliminated is False
        assert attempt.cost_usd == 0.0
        assert attempt.error == ""


class TestRemediationResult:
    def test_defaults(self):
        result = RemediationResult(vulnerability_id="vuln_123")
        assert result.vulnerability_id == "vuln_123"
        assert result.status == RemediationStatus.SKIPPED
        assert result.attempts == []
        assert result.total_attempts == 0
        assert result.successful_attempt is None
        assert result.final_patch_diff == ""
        assert result.total_cost_usd == 0.0

    def test_successful_remediation(self):
        result = RemediationResult(
            vulnerability_id="vuln_123",
            status=RemediationStatus.PATCHED,
            total_attempts=1,
            successful_attempt=0,
            final_patch_diff="--- a/file.py\n+++ b/file.py\n@@ ...",
            total_cost_usd=0.05,
        )
        assert result.status == RemediationStatus.PATCHED
        assert result.successful_attempt == 0
