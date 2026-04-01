"""
Unit tests for Inspector VulnerabilityProver (Phases 4 + 5).

Tests Z3 constraint inversion, vulnerability classification, severity mapping,
PoC generation, response parsing, and safety validation.
All Z3Bridge and LLM calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.simula.inspector.prover import (
    _VULN_SEVERITY_MAP,
    VulnerabilityProver,
)
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    InspectorConfig,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)

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


def _make_llm() -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.text = json.dumps({
        "variable_declarations": {
            "is_authenticated": "Bool",
            "requested_user_id": "Int",
            "current_user_id": "Int",
            "can_access_data": "Bool",
        },
        "z3_expression": (
            "z3.And(is_authenticated == True, "
            "requested_user_id != current_user_id, "
            "can_access_data == True)"
        ),
        "reasoning": "If authenticated user can access another user's data, access control is broken",
    })
    llm.generate = AsyncMock(return_value=response)
    llm.evaluate = AsyncMock(return_value=MagicMock(text='{"severity":"high","reasoning":"test"}'))
    return llm


def _make_z3_bridge() -> MagicMock:
    return MagicMock()


def _make_prover(
    llm: MagicMock | None = None,
    z3_bridge: MagicMock | None = None,
) -> VulnerabilityProver:
    return VulnerabilityProver(
        z3_bridge=z3_bridge or _make_z3_bridge(),
        llm=llm or _make_llm(),
        max_encoding_retries=2,
        check_timeout_ms=5_000,
    )


# ── Vulnerability Classification ────────────────────────────────────────────


class TestVulnerabilityClassification:
    """Test _classify_vulnerability for each vulnerability class."""

    @pytest.mark.parametrize("goal,expected_class", [
        ("Unauthenticated access to protected resource", VulnerabilityClass.BROKEN_AUTH),
        ("Authentication bypass via session fixation", VulnerabilityClass.BROKEN_AUTH),
        ("SQL injection in user input", VulnerabilityClass.SQL_INJECTION),
        ("Broken access control: user A can access user B's data", VulnerabilityClass.BROKEN_ACCESS_CONTROL),
        ("Privilege escalation: regular user can call admin function", VulnerabilityClass.PRIVILEGE_ESCALATION),
        ("Reentrancy: contract can call itself recursively", VulnerabilityClass.REENTRANCY),
        ("Unvalidated redirect", VulnerabilityClass.UNVALIDATED_REDIRECT),
        ("Race condition: concurrent access violates invariant", VulnerabilityClass.RACE_CONDITION),
        ("XSS via user profile page", VulnerabilityClass.XSS),
        ("SSRF through URL parameter", VulnerabilityClass.SSRF),
        ("Command injection via shell parameter", VulnerabilityClass.COMMAND_INJECTION),
        ("Insecure deserialization of pickle data", VulnerabilityClass.INSECURE_DESERIALIZATION),
        ("Path traversal via file download endpoint", VulnerabilityClass.PATH_TRAVERSAL),
        ("Information disclosure through error messages", VulnerabilityClass.INFORMATION_DISCLOSURE),
    ])
    def test_classifies_known_goals(self, goal: str, expected_class: VulnerabilityClass):
        prover = _make_prover()
        result = prover._classify_vulnerability(goal)
        assert result == expected_class

    def test_unknown_goal_returns_other(self):
        prover = _make_prover()
        result = prover._classify_vulnerability("Something completely unrelated to security")
        assert result == VulnerabilityClass.OTHER


# ── Severity Heuristics ─────────────────────────────────────────────────────


class TestSeverityHeuristics:
    def test_sql_injection_is_critical(self):
        assert _VULN_SEVERITY_MAP[VulnerabilityClass.SQL_INJECTION] == VulnerabilitySeverity.CRITICAL

    def test_command_injection_is_critical(self):
        assert _VULN_SEVERITY_MAP[VulnerabilityClass.COMMAND_INJECTION] == VulnerabilitySeverity.CRITICAL

    def test_reentrancy_is_critical(self):
        assert _VULN_SEVERITY_MAP[VulnerabilityClass.REENTRANCY] == VulnerabilitySeverity.CRITICAL

    def test_broken_auth_is_high(self):
        assert _VULN_SEVERITY_MAP[VulnerabilityClass.BROKEN_AUTH] == VulnerabilitySeverity.HIGH

    def test_xss_is_medium(self):
        assert _VULN_SEVERITY_MAP[VulnerabilityClass.XSS] == VulnerabilitySeverity.MEDIUM

    def test_unvalidated_redirect_is_low(self):
        assert _VULN_SEVERITY_MAP[VulnerabilityClass.UNVALIDATED_REDIRECT] == VulnerabilitySeverity.LOW

    @pytest.mark.asyncio
    async def test_smart_contract_reentrancy_escalated_to_critical(self):
        """Smart contract reentrancy should always be CRITICAL."""
        prover = _make_prover()
        surface = _make_surface(surface_type=AttackSurfaceType.SMART_CONTRACT_PUBLIC)
        severity = await prover._classify_severity(
            surface, "Reentrancy attack", "x=1",
            VulnerabilityClass.REENTRANCY,
        )
        assert severity == VulnerabilitySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_auth_handler_escalated_to_high(self):
        """Auth handler vulnerabilities should be at least HIGH."""
        prover = _make_prover()
        surface = _make_surface(surface_type=AttackSurfaceType.AUTH_HANDLER)
        severity = await prover._classify_severity(
            surface, "Open redirect", "x=1",
            VulnerabilityClass.UNVALIDATED_REDIRECT,
        )
        assert severity == VulnerabilitySeverity.HIGH


# ── Encoding Response Parsing ───────────────────────────────────────────────


class TestEncodingResponseParsing:
    def test_parses_valid_json(self):
        prover = _make_prover()
        response = json.dumps({
            "variable_declarations": {"x": "Int", "y": "Bool"},
            "z3_expression": "z3.And(x > 0, y == True)",
            "reasoning": "Test encoding",
        })
        result = prover._parse_encoding_response(response)
        assert result is not None
        z3_expr, decls, reasoning = result
        assert z3_expr == "z3.And(x > 0, y == True)"
        assert decls == {"x": "Int", "y": "Bool"}
        assert reasoning == "Test encoding"

    def test_strips_markdown_fences(self):
        prover = _make_prover()
        response = """```json
{
  "variable_declarations": {"x": "Int"},
  "z3_expression": "x > 0",
  "reasoning": "test"
}
```"""
        result = prover._parse_encoding_response(response)
        assert result is not None
        assert result[0] == "x > 0"

    def test_normalizes_type_names(self):
        prover = _make_prover()
        response = json.dumps({
            "variable_declarations": {"a": "int", "b": "bool", "c": "real"},
            "z3_expression": "a > 0",
            "reasoning": "test",
        })
        result = prover._parse_encoding_response(response)
        assert result is not None
        _, decls, _ = result
        assert decls["a"] == "Int"
        assert decls["b"] == "Bool"
        assert decls["c"] == "Real"

    def test_returns_none_for_missing_expression(self):
        prover = _make_prover()
        response = json.dumps({
            "variable_declarations": {"x": "Int"},
            "z3_expression": "",
            "reasoning": "test",
        })
        result = prover._parse_encoding_response(response)
        assert result is None

    def test_returns_none_for_garbage(self):
        prover = _make_prover()
        assert prover._parse_encoding_response("not json at all") is None

    def test_handles_json_with_surrounding_text(self):
        prover = _make_prover()
        response = """Here is my encoding:
{
  "variable_declarations": {"x": "Int"},
  "z3_expression": "x > 0",
  "reasoning": "test"
}
I hope this helps!"""
        result = prover._parse_encoding_response(response)
        assert result is not None
        assert result[0] == "x > 0"


# ── Severity Response Parsing ───────────────────────────────────────────────


class TestSeverityResponseParsing:
    def test_parses_valid_severity(self):
        prover = _make_prover()
        response = json.dumps({"severity": "HIGH", "reasoning": "test"})
        result = prover._parse_severity_response(response)
        assert result == VulnerabilitySeverity.HIGH

    def test_parses_lowercase(self):
        prover = _make_prover()
        response = json.dumps({"severity": "critical", "reasoning": "test"})
        result = prover._parse_severity_response(response)
        assert result == VulnerabilitySeverity.CRITICAL

    def test_returns_none_for_invalid(self):
        prover = _make_prover()
        assert prover._parse_severity_response("not json") is None
        assert prover._parse_severity_response('{"severity": "EXTREME"}') is None


# ── Z3 Constraint Checking (Direct) ────────────────────────────────────────


class TestZ3ConstraintChecking:
    """Test _check_exploit_constraints with real Z3 solver."""

    def test_sat_result_returns_counterexample(self):
        """SAT = vulnerability proven (The Inversion)."""
        prover = _make_prover()

        # This expression is always satisfiable - Z3 will find values
        z3_expr = "z3.And(is_authenticated == False, user_id != attacker_id)"
        var_decls = {
            "is_authenticated": "Bool",
            "user_id": "Int",
            "attacker_id": "Int",
        }

        try:
            import z3 as z3_lib  # noqa: F401
        except ImportError:
            pytest.skip("z3-solver not installed")

        status, counterexample = prover._check_exploit_constraints(z3_expr, var_decls)
        assert status == "sat"
        assert counterexample  # Should have variable assignments
        assert "=" in counterexample

    def test_unsat_result(self):
        """UNSAT = security property holds, no vulnerability."""
        prover = _make_prover()

        # Contradictory expression: x > 0 AND x < 0
        z3_expr = "z3.And(x > 0, x < 0)"
        var_decls = {"x": "Int"}

        try:
            import z3 as z3_lib  # noqa: F401
        except ImportError:
            pytest.skip("z3-solver not installed")

        status, counterexample = prover._check_exploit_constraints(z3_expr, var_decls)
        assert status == "unsat"
        assert counterexample == ""

    def test_invalid_expression_returns_unknown(self):
        """Invalid expression should return 'unknown' with error detail."""
        prover = _make_prover()
        status, detail = prover._check_exploit_constraints(
            "this is not valid z3 code", {"x": "Int"},
        )
        assert status == "unknown"
        assert detail  # Should contain error message

    def test_non_boolref_returns_unknown(self):
        """Expression that doesn't produce z3.BoolRef should return 'unknown'."""
        prover = _make_prover()

        try:
            import z3 as z3_lib  # noqa: F401
        except ImportError:
            pytest.skip("z3-solver not installed")

        status, detail = prover._check_exploit_constraints(
            "x + 1",  # produces IntRef, not BoolRef
            {"x": "Int"},
        )
        assert status == "unknown"
        assert "BoolRef" in detail


# ── Z3 Expression Validation ────────────────────────────────────────────────


class TestZ3ExpressionValidation:
    def test_valid_expression_passes(self):
        prover = _make_prover()

        try:
            import z3 as z3_lib  # noqa: F401
        except ImportError:
            pytest.skip("z3-solver not installed")

        error = prover._validate_z3_expression(
            "z3.And(x > 0, y == True)",
            {"x": "Int", "y": "Bool"},
        )
        assert error is None

    def test_invalid_expression_fails(self):
        prover = _make_prover()

        try:
            import z3 as z3_lib  # noqa: F401
        except ImportError:
            pytest.skip("z3-solver not installed")

        error = prover._validate_z3_expression(
            "z3.And(undeclared_var > 0)",
            {"x": "Int"},
        )
        assert error is not None


# ── Full prove_vulnerability Pipeline (Mocked) ─────────────────────────────


class TestProveVulnerability:
    @pytest.mark.asyncio
    async def test_sat_returns_vulnerability_report(self):
        """When Z3 returns SAT, prove_vulnerability returns a VulnerabilityReport."""
        prover = _make_prover()

        # Mock the internal constraint checking to return SAT
        with patch.object(
            prover, "_check_exploit_constraints",
            return_value=("sat", "is_authenticated=False, user_id=999"),
        ):
            report = await prover.prove_vulnerability(
                surface=_make_surface(),
                attack_goal="Unauthenticated access to protected resource",
                target_url="https://github.com/test/repo",
            )

        assert report is not None
        assert isinstance(report, VulnerabilityReport)
        assert report.target_url == "https://github.com/test/repo"
        assert report.vulnerability_class == VulnerabilityClass.BROKEN_AUTH
        assert report.z3_counterexample == "is_authenticated=False, user_id=999"
        assert not report.proof_of_concept_code  # generate_poc=False by default

    @pytest.mark.asyncio
    async def test_unsat_returns_none(self):
        """When Z3 returns UNSAT, no vulnerability exists."""
        prover = _make_prover()

        with patch.object(
            prover, "_check_exploit_constraints",
            return_value=("unsat", ""),
        ):
            report = await prover.prove_vulnerability(
                surface=_make_surface(),
                attack_goal="SQL injection in user input",
            )

        assert report is None

    @pytest.mark.asyncio
    async def test_unknown_returns_none(self):
        """Solver timeout/unknown returns None (inconclusive)."""
        prover = _make_prover()

        with patch.object(
            prover, "_check_exploit_constraints",
            return_value=("unknown", "solver timeout"),
        ):
            report = await prover.prove_vulnerability(
                surface=_make_surface(),
                attack_goal="SQL injection",
            )

        assert report is None

    @pytest.mark.asyncio
    async def test_encoding_failure_returns_none(self):
        """When LLM fails to encode the attack goal, return None."""
        llm = _make_llm()
        llm.generate = AsyncMock(side_effect=Exception("LLM down"))
        prover = _make_prover(llm=llm)

        report = await prover.prove_vulnerability(
            surface=_make_surface(),
            attack_goal="SQL injection",
        )
        assert report is None

    @pytest.mark.asyncio
    async def test_generates_poc_when_requested(self):
        """When generate_poc=True and vulnerability is proven, PoC is generated."""
        prover = _make_prover()

        poc_code = '''"""PoC: IDOR"""\nimport requests\nTARGET_URL = "http://localhost:8000"\ndef exploit():\n    return f"{TARGET_URL}/api/user/999", {}, None\n'''

        with patch.object(
            prover, "_check_exploit_constraints",
            return_value=("sat", "is_authenticated=True, user_id=999"),
        ), patch.object(
            prover, "generate_poc",
            AsyncMock(return_value=poc_code),
        ):
            report = await prover.prove_vulnerability(
                surface=_make_surface(),
                attack_goal="Broken access control",
                generate_poc=True,
                config=InspectorConfig(authorized_targets=["localhost"]),
            )

        assert report is not None
        assert report.proof_of_concept_code == poc_code


# ── Batch Proving ───────────────────────────────────────────────────────────


class TestBatchProving:
    @pytest.mark.asyncio
    async def test_batch_returns_all_proven(self):
        prover = _make_prover()

        call_count = 0

        async def mock_prove(surface, attack_goal, **kwargs):
            nonlocal call_count
            call_count += 1
            if "SQL injection" in attack_goal:
                return VulnerabilityReport(
                    target_url="url",
                    vulnerability_class=VulnerabilityClass.SQL_INJECTION,
                    severity=VulnerabilitySeverity.CRITICAL,
                    attack_surface=surface,
                    attack_goal=attack_goal,
                    z3_counterexample="input_sanitized=False",
                )
            return None

        with patch.object(prover, "prove_vulnerability", side_effect=mock_prove):
            results = await prover.prove_vulnerability_batch(
                surface=_make_surface(),
                attack_goals=[
                    "SQL injection in user input",
                    "Something not exploitable",
                    "Another SQL injection vector",
                ],
            )

        assert len(results) == 2  # Only SQL injection goals returned reports
        assert call_count == 3  # All three goals were attempted


# ── PoC Safety Validation ───────────────────────────────────────────────────


class TestPoCValidation:
    def test_valid_poc_accepted(self):
        prover = _make_prover()
        poc = '''\
import requests
import json

TARGET_URL = "http://localhost:8000"

def exploit():
    url = f"{TARGET_URL}/api/user/999"
    headers = {"Authorization": "Bearer token"}
    return url, headers, None

if __name__ == "__main__":
    exploit()
'''
        error = prover._validate_poc_safety(poc, [])
        assert error is None

    def test_forbidden_import_rejected(self):
        prover = _make_prover()
        poc = "import subprocess\nsubprocess.call(['rm', '-rf', '/'])"
        error = prover._validate_poc_safety(poc, [])
        assert error is not None
        assert "subprocess" in error.lower() or "forbidden" in error.lower()

    def test_socket_import_rejected(self):
        prover = _make_prover()
        poc = "import socket\ns = socket.socket()"
        error = prover._validate_poc_safety(poc, [])
        assert error is not None

    def test_syntax_error_detected(self):
        prover = _make_prover()
        poc = "def foo(\n    # syntax error"
        error = prover._validate_poc_syntax(poc)
        assert error is not None
