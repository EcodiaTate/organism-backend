"""
Unit tests for Simula Symbolic Execution Engine (Stage 6E).

Tests domain function extraction via AST keywords, property encoding via
LLM mock, prove_properties with various domains, counterexample detection,
timeout handling, and domain keyword matching.
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.simula.verification.symbolic_execution import (
    _DOMAIN_KEYWORDS,
    SymbolicExecutionEngine,
)
from systems.simula.verification.types import (
    SymbolicDomain,
    SymbolicExecutionResult,
    SymbolicExecutionStatus,
    SymbolicProperty,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_py(root: Path, rel: str, source: str) -> Path:
    """Write a Python file and return its path."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    return p


def _make_engine(
    llm: AsyncMock | None = None,
    z3_bridge: MagicMock | None = None,
    timeout_ms: int = 5000,
) -> SymbolicExecutionEngine:
    """Create an engine with optional mocked dependencies."""
    return SymbolicExecutionEngine(
        z3_bridge=z3_bridge,
        llm=llm,
        timeout_ms=timeout_ms,
    )


def _llm_json_response(properties: list[dict]) -> MagicMock:
    """Create a mock LLM response whose .content is JSON."""
    resp = MagicMock()
    resp.content = json.dumps(properties)
    return resp


def _build_mock_z3() -> types.ModuleType:
    """
    Build a fake z3 module with just enough to satisfy _check_property.

    Provides: BoolRef, Int, Real, Bool, Not, And, Or, Implies, Solver,
    and the sat/unsat/unknown sentinel values.
    """
    mock_z3 = types.ModuleType("z3")

    class FakeBoolRef:
        """Sentinel class so isinstance(val, z3.BoolRef) works."""
        def __init__(self, name: str = "", proved: bool = True) -> None:
            self._name = name
            self._proved = proved

    mock_z3.BoolRef = FakeBoolRef  # type: ignore[attr-defined]

    # Sentinels for solver results
    mock_z3.unsat = "unsat"  # type: ignore[attr-defined]
    mock_z3.sat = "sat"  # type: ignore[attr-defined]
    mock_z3.unknown = "unknown"  # type: ignore[attr-defined]

    def _make_var(sort: str):
        def factory(name: str) -> MagicMock:
            m = MagicMock(name=f"z3.{sort}({name!r})")
            m.__eq__ = lambda self, other: FakeBoolRef(name, proved=True)  # type: ignore[assignment]
            m.__lt__ = lambda self, other: FakeBoolRef(name, proved=False)  # type: ignore[assignment]
            m.__le__ = lambda self, other: FakeBoolRef(name, proved=False)  # type: ignore[assignment]
            m.__gt__ = lambda self, other: FakeBoolRef(name, proved=False)  # type: ignore[assignment]
            m.__ge__ = lambda self, other: FakeBoolRef(name, proved=True)  # type: ignore[assignment]
            m.__add__ = lambda self, other: m  # type: ignore[assignment]
            return m
        return factory

    mock_z3.Int = _make_var("Int")  # type: ignore[attr-defined]
    mock_z3.Real = _make_var("Real")  # type: ignore[attr-defined]
    mock_z3.Bool = _make_var("Bool")  # type: ignore[attr-defined]

    mock_z3.Not = lambda expr: FakeBoolRef("Not")  # type: ignore[attr-defined]
    mock_z3.And = lambda *args: FakeBoolRef("And")  # type: ignore[attr-defined]
    mock_z3.Or = lambda *args: FakeBoolRef("Or")  # type: ignore[attr-defined]
    mock_z3.Implies = lambda a, b: FakeBoolRef("Implies")  # type: ignore[attr-defined]

    return mock_z3


def _make_check_property_mock(
    status: SymbolicExecutionStatus,
    detail: str = "",
) -> AsyncMock:
    """Create an AsyncMock for _check_property returning a fixed result."""
    mock = AsyncMock(return_value=(status, detail))
    return mock


# ── Domain Function Extraction ───────────────────────────────────────────────


class TestDomainFunctionExtraction:
    """Test AST-based extraction of functions matching domain keywords."""

    def test_extracts_budget_function_by_name(self, tmp_path: Path):
        """A function named 'calculate_budget' matches BUDGET_CALCULATION domain."""
        _write_py(
            tmp_path,
            "finance/budget.py",
            """\
            def calculate_budget(amount: float, rate: float) -> float:
                return amount * rate
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["finance/budget.py"],
            domains=[SymbolicDomain.BUDGET_CALCULATION],
            codebase_root=tmp_path,
        )

        assert len(targets) == 1
        source, func_name, domain, file_path = targets[0]
        assert func_name == "calculate_budget"
        assert domain == SymbolicDomain.BUDGET_CALCULATION
        assert file_path == "finance/budget.py"
        assert "amount * rate" in source

    def test_extracts_access_control_function(self, tmp_path: Path):
        """A function named 'check_permission' matches ACCESS_CONTROL domain."""
        _write_py(
            tmp_path,
            "auth/permissions.py",
            """\
            def check_permission(user_role: str, resource: str) -> bool:
                return user_role == "admin"
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["auth/permissions.py"],
            domains=[SymbolicDomain.ACCESS_CONTROL],
            codebase_root=tmp_path,
        )

        assert len(targets) == 1
        assert targets[0][1] == "check_permission"
        assert targets[0][2] == SymbolicDomain.ACCESS_CONTROL

    def test_extracts_risk_scoring_function(self, tmp_path: Path):
        """A function named 'compute_risk_score' matches RISK_SCORING domain."""
        _write_py(
            tmp_path,
            "risk/scoring.py",
            """\
            def compute_risk_score(factors: list) -> float:
                return sum(factors) / len(factors)
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["risk/scoring.py"],
            domains=[SymbolicDomain.RISK_SCORING],
            codebase_root=tmp_path,
        )

        assert len(targets) == 1
        assert targets[0][1] == "compute_risk_score"
        assert targets[0][2] == SymbolicDomain.RISK_SCORING

    def test_extracts_governance_gating_function(self, tmp_path: Path):
        """A function named 'check_quorum' matches GOVERNANCE_GATING domain."""
        _write_py(
            tmp_path,
            "governance/voting.py",
            """\
            def check_quorum(votes: int, total: int) -> bool:
                return votes >= total // 2 + 1
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["governance/voting.py"],
            domains=[SymbolicDomain.GOVERNANCE_GATING],
            codebase_root=tmp_path,
        )

        assert len(targets) == 1
        assert targets[0][1] == "check_quorum"
        assert targets[0][2] == SymbolicDomain.GOVERNANCE_GATING

    def test_extracts_constitutional_alignment_function(self, tmp_path: Path):
        """A function named 'measure_alignment' matches CONSTITUTIONAL_ALIGNMENT."""
        _write_py(
            tmp_path,
            "alignment/checker.py",
            """\
            def measure_alignment(drive_scores: dict) -> float:
                return sum(drive_scores.values()) / len(drive_scores)
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["alignment/checker.py"],
            domains=[SymbolicDomain.CONSTITUTIONAL_ALIGNMENT],
            codebase_root=tmp_path,
        )

        assert len(targets) == 1
        assert targets[0][1] == "measure_alignment"
        assert targets[0][2] == SymbolicDomain.CONSTITUTIONAL_ALIGNMENT

    def test_ignores_non_matching_functions(self, tmp_path: Path):
        """Functions without domain keywords should not be extracted."""
        _write_py(
            tmp_path,
            "utils/helpers.py",
            """\
            def format_string(s: str) -> str:
                return s.strip()

            def process_data(data: list) -> list:
                return sorted(data)
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["utils/helpers.py"],
            domains=list(SymbolicDomain),
            codebase_root=tmp_path,
        )

        assert len(targets) == 0

    def test_ignores_non_python_files(self, tmp_path: Path):
        """Non-.py files should be skipped entirely."""
        (tmp_path / "readme.md").write_text("# Budget docs", encoding="utf-8")

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["readme.md"],
            domains=[SymbolicDomain.BUDGET_CALCULATION],
            codebase_root=tmp_path,
        )

        assert len(targets) == 0

    def test_ignores_missing_files(self, tmp_path: Path):
        """Non-existent files should be skipped without error."""
        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["nonexistent/module.py"],
            domains=[SymbolicDomain.BUDGET_CALCULATION],
            codebase_root=tmp_path,
        )

        assert len(targets) == 0

    def test_ignores_syntax_error_files(self, tmp_path: Path):
        """Files with syntax errors should be skipped gracefully."""
        _write_py(
            tmp_path,
            "broken/budget_calc.py",
            """\
            def calculate_budget(
                # missing closing paren and colon
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["broken/budget_calc.py"],
            domains=[SymbolicDomain.BUDGET_CALCULATION],
            codebase_root=tmp_path,
        )

        assert len(targets) == 0

    def test_extracts_async_function(self, tmp_path: Path):
        """Async functions should also be extracted when they match keywords."""
        _write_py(
            tmp_path,
            "async_auth/gate.py",
            """\
            async def authorize_request(token: str, resource: str) -> bool:
                return token == "valid"
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["async_auth/gate.py"],
            domains=[SymbolicDomain.ACCESS_CONTROL],
            codebase_root=tmp_path,
        )

        assert len(targets) == 1
        assert targets[0][1] == "authorize_request"

    def test_no_double_count_across_domains(self, tmp_path: Path):
        """A function matching multiple domains should only appear once (first match wins)."""
        _write_py(
            tmp_path,
            "multi/scorer.py",
            """\
            def assess_risk_level(value: float) -> str:
                if value > 0.8:
                    return "high"
                return "low"
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["multi/scorer.py"],
            domains=[SymbolicDomain.RISK_SCORING, SymbolicDomain.ACCESS_CONTROL],
            codebase_root=tmp_path,
        )

        # "assess" and "risk" and "level" match RISK_SCORING; should not double-count
        assert len(targets) == 1
        assert targets[0][2] == SymbolicDomain.RISK_SCORING

    def test_extracts_multiple_functions_from_one_file(self, tmp_path: Path):
        """Multiple matching functions in a single file should all be extracted."""
        _write_py(
            tmp_path,
            "finance/ops.py",
            """\
            def allocate_budget(total: float, share: float) -> float:
                return total * share

            def check_balance(current: float, minimum: float) -> bool:
                return current >= minimum
            """,
        )

        engine = _make_engine()
        targets = engine._extract_domain_functions(
            files=["finance/ops.py"],
            domains=[SymbolicDomain.BUDGET_CALCULATION],
            codebase_root=tmp_path,
        )

        assert len(targets) == 2
        func_names = {t[1] for t in targets}
        assert "allocate_budget" in func_names
        assert "check_balance" in func_names

    def test_filters_by_specified_domains_only(self, tmp_path: Path):
        """Only domains passed to the method should be considered."""
        _write_py(
            tmp_path,
            "mixed/service.py",
            """\
            def calculate_cost(items: list) -> float:
                return sum(items)

            def check_permission(user: str) -> bool:
                return user == "admin"
            """,
        )

        engine = _make_engine()
        # Only look for BUDGET_CALCULATION, not ACCESS_CONTROL
        targets = engine._extract_domain_functions(
            files=["mixed/service.py"],
            domains=[SymbolicDomain.BUDGET_CALCULATION],
            codebase_root=tmp_path,
        )

        assert len(targets) == 1
        assert targets[0][1] == "calculate_cost"


# ── Domain Keyword Matching ──────────────────────────────────────────────────


class TestDomainKeywordMatching:
    """Verify the domain keyword table covers expected function names."""

    def test_budget_keywords_cover_common_names(self):
        """BUDGET_CALCULATION keywords should match typical budget-related names."""
        keywords = _DOMAIN_KEYWORDS[SymbolicDomain.BUDGET_CALCULATION]
        assert "budget" in keywords
        assert "cost" in keywords
        assert "allocate" in keywords
        assert "balance" in keywords
        assert "spend" in keywords

    def test_access_control_keywords_cover_common_names(self):
        """ACCESS_CONTROL keywords should match typical auth names."""
        keywords = _DOMAIN_KEYWORDS[SymbolicDomain.ACCESS_CONTROL]
        assert "permission" in keywords
        assert "authorize" in keywords
        assert "role" in keywords
        assert "access" in keywords

    def test_risk_scoring_keywords_cover_common_names(self):
        """RISK_SCORING keywords should match typical risk assessment names."""
        keywords = _DOMAIN_KEYWORDS[SymbolicDomain.RISK_SCORING]
        assert "risk" in keywords
        assert "score" in keywords
        assert "threshold" in keywords

    def test_governance_keywords_cover_common_names(self):
        """GOVERNANCE_GATING keywords should match typical governance names."""
        keywords = _DOMAIN_KEYWORDS[SymbolicDomain.GOVERNANCE_GATING]
        assert "governance" in keywords
        assert "approve" in keywords
        assert "vote" in keywords
        assert "quorum" in keywords

    def test_constitutional_keywords_cover_common_names(self):
        """CONSTITUTIONAL_ALIGNMENT keywords should match alignment concepts."""
        keywords = _DOMAIN_KEYWORDS[SymbolicDomain.CONSTITUTIONAL_ALIGNMENT]
        assert "constitution" in keywords
        assert "alignment" in keywords
        assert "coherence" in keywords
        assert "care" in keywords
        assert "growth" in keywords

    def test_all_domains_have_keywords(self):
        """Every SymbolicDomain enum member should have keyword entries."""
        for domain in SymbolicDomain:
            assert domain in _DOMAIN_KEYWORDS, f"Missing keywords for {domain}"
            assert len(_DOMAIN_KEYWORDS[domain]) > 0, f"Empty keywords for {domain}"


# ── Property Encoding (LLM Mock) ────────────────────────────────────────────


class TestPropertyEncoding:
    """Test LLM-based property extraction with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_encode_function_parses_llm_json(self):
        """LLM JSON response should be parsed into SymbolicProperty objects."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "budget_non_negative",
                "human_description": "Budget allocation is always >= 0",
                "z3_encoding": "budget = z3.Real('budget'); z3.And(budget >= 0, budget <= 1.0)",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        props = await engine._encode_function(
            source="def allocate_budget(amount): return amount * 0.5",
            function_name="allocate_budget",
            domain=SymbolicDomain.BUDGET_CALCULATION,
        )

        assert len(props) == 1
        assert props[0].property_name == "budget_non_negative"
        assert props[0].human_description == "Budget allocation is always >= 0"
        assert "z3.Real" in props[0].z3_encoding
        assert props[0].domain == SymbolicDomain.BUDGET_CALCULATION
        assert props[0].target_function == "allocate_budget"

    @pytest.mark.asyncio
    async def test_encode_function_handles_multiple_properties(self):
        """Multiple properties from a single LLM response should all be parsed."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "score_bounded_above",
                "human_description": "Risk score never exceeds 1.0",
                "z3_encoding": "s = z3.Real('s'); s <= 1.0",
            },
            {
                "property_name": "score_bounded_below",
                "human_description": "Risk score is never negative",
                "z3_encoding": "s = z3.Real('s'); s >= 0.0",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        props = await engine._encode_function(
            source="def compute_risk(x): return max(0, min(1, x))",
            function_name="compute_risk",
            domain=SymbolicDomain.RISK_SCORING,
        )

        assert len(props) == 2
        names = {p.property_name for p in props}
        assert "score_bounded_above" in names
        assert "score_bounded_below" in names

    @pytest.mark.asyncio
    async def test_encode_function_handles_markdown_fenced_json(self):
        """LLM response wrapped in markdown code fences should still parse."""
        mock_llm = AsyncMock()
        resp = MagicMock()
        resp.content = (
            "Here are the properties:\n\n"
            "```json\n"
            '[{"property_name": "quorum_positive", '
            '"human_description": "Quorum must be positive", '
            '"z3_encoding": "q = z3.Int(\'q\'); q > 0"}]\n'
            "```\n"
        )
        mock_llm.complete.return_value = resp

        engine = _make_engine(llm=mock_llm)
        props = await engine._encode_function(
            source="def check_quorum(q): return q > 0",
            function_name="check_quorum",
            domain=SymbolicDomain.GOVERNANCE_GATING,
        )

        assert len(props) == 1
        assert props[0].property_name == "quorum_positive"

    @pytest.mark.asyncio
    async def test_encode_function_returns_empty_on_no_json(self):
        """If LLM returns no JSON array, result should be empty list."""
        mock_llm = AsyncMock()
        resp = MagicMock()
        resp.content = "I cannot generate properties for this function."
        mock_llm.complete.return_value = resp

        engine = _make_engine(llm=mock_llm)
        props = await engine._encode_function(
            source="def allocate_budget(x): return x",
            function_name="allocate_budget",
            domain=SymbolicDomain.BUDGET_CALCULATION,
        )

        assert props == []

    @pytest.mark.asyncio
    async def test_encode_function_returns_empty_on_llm_exception(self):
        """LLM failures should be caught and return empty list."""
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = RuntimeError("LLM connection failed")

        engine = _make_engine(llm=mock_llm)
        props = await engine._encode_function(
            source="def calculate_cost(x): return x * 2",
            function_name="calculate_cost",
            domain=SymbolicDomain.BUDGET_CALCULATION,
        )

        assert props == []

    @pytest.mark.asyncio
    async def test_encode_function_returns_empty_when_no_llm(self):
        """When no LLM is configured, encoding should return empty list."""
        engine = _make_engine(llm=None)
        props = await engine._encode_function(
            source="def budget_calc(x): return x",
            function_name="budget_calc",
            domain=SymbolicDomain.BUDGET_CALCULATION,
        )

        assert props == []

    @pytest.mark.asyncio
    async def test_encode_function_skips_non_dict_entries(self):
        """Non-dict entries in the JSON array should be silently skipped."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            "this is not a dict",
            42,
            {
                "property_name": "valid_prop",
                "human_description": "A valid property",
                "z3_encoding": "x = z3.Int('x'); x >= 0",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        props = await engine._encode_function(
            source="def spend_budget(x): return x",
            function_name="spend_budget",
            domain=SymbolicDomain.BUDGET_CALCULATION,
        )

        assert len(props) == 1
        assert props[0].property_name == "valid_prop"

    @pytest.mark.asyncio
    async def test_encode_function_uses_str_response_fallback(self):
        """If response has no .content attribute, str(response) is used."""
        mock_llm = AsyncMock()

        class PlainResponse:
            """A response object without a .content attribute."""
            def __str__(self) -> str:
                return json.dumps([
                    {
                        "property_name": "fallback_prop",
                        "human_description": "Fallback test",
                        "z3_encoding": "x = z3.Int('x'); x > 0",
                    },
                ])

        mock_llm.complete.return_value = PlainResponse()

        engine = _make_engine(llm=mock_llm)
        props = await engine._encode_function(
            source="def budget_check(x): return x > 0",
            function_name="budget_check",
            domain=SymbolicDomain.BUDGET_CALCULATION,
        )

        assert len(props) == 1
        assert props[0].property_name == "fallback_prop"


# ── Symbolic Execution (prove_properties with mocked _check_property) ────────


class TestSymbolicExecution:
    """Test prove_properties end-to-end with mocked LLM and _check_property."""

    @pytest.mark.asyncio
    async def test_prove_properties_skipped_when_no_domain_functions(self, tmp_path: Path):
        """When no files match domain keywords, result is SKIPPED."""
        _write_py(
            tmp_path,
            "utils/helper.py",
            """\
            def format_string(s: str) -> str:
                return s.strip()
            """,
        )

        engine = _make_engine(llm=AsyncMock())
        result = await engine.prove_properties(
            files=["utils/helper.py"],
            codebase_root=tmp_path,
        )

        assert isinstance(result, SymbolicExecutionResult)
        assert result.status == SymbolicExecutionStatus.SKIPPED
        assert result.properties_checked == 0

    @pytest.mark.asyncio
    async def test_prove_properties_skipped_for_empty_file_list(self, tmp_path: Path):
        """Empty file list should return SKIPPED."""
        engine = _make_engine(llm=AsyncMock())
        result = await engine.prove_properties(
            files=[],
            codebase_root=tmp_path,
        )

        assert result.status == SymbolicExecutionStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_prove_properties_proved_with_z3_unsat(self, tmp_path: Path):
        """When Z3 returns UNSAT for NOT(property), overall status is PROVED."""
        _write_py(
            tmp_path,
            "finance/budget.py",
            """\
            def allocate_budget(amount: float) -> float:
                return max(0.0, amount)
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "tautology_holds",
                "human_description": "A simple tautology: x == x",
                "z3_encoding": "x = z3.Real('x'); x == x",
            },
        ])

        engine = _make_engine(llm=mock_llm)

        # Mock _check_property to simulate Z3 proving the property
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.PROVED, ""),
        )

        result = await engine.prove_properties(
            files=["finance/budget.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.BUDGET_CALCULATION],
        )

        assert result.status == SymbolicExecutionStatus.PROVED
        assert result.properties_proved == 1
        assert result.properties_failed == 0
        assert len(result.counterexamples) == 0
        assert len(result.properties) == 1
        assert result.properties[0].status == SymbolicExecutionStatus.PROVED

    @pytest.mark.asyncio
    async def test_prove_properties_counterexample_with_z3_sat(self, tmp_path: Path):
        """When Z3 finds NOT(property) is SAT, status is COUNTEREXAMPLE."""
        _write_py(
            tmp_path,
            "risk/scorer.py",
            """\
            def compute_risk_score(x: float) -> float:
                return x * 2
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "score_always_small",
                "human_description": "Score is always <= 10",
                "z3_encoding": "x = z3.Int('x'); x <= 10",
            },
        ])

        engine = _make_engine(llm=mock_llm)

        # Mock _check_property to simulate Z3 finding a counterexample
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.COUNTEREXAMPLE, "x=11"),
        )

        result = await engine.prove_properties(
            files=["risk/scorer.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.RISK_SCORING],
        )

        assert result.status == SymbolicExecutionStatus.COUNTEREXAMPLE
        assert result.properties_failed == 1
        assert len(result.counterexamples) == 1
        assert "compute_risk_score" in result.counterexamples[0]
        assert result.properties[0].counterexample == "x=11"

    @pytest.mark.asyncio
    async def test_prove_properties_mixed_proved_and_failed(self, tmp_path: Path):
        """Mixed results: some proved, some with counterexamples. Overall is COUNTEREXAMPLE."""
        _write_py(
            tmp_path,
            "finance/allocator.py",
            """\
            def allocate_budget(amount: float) -> float:
                return amount * 0.5
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "tautology",
                "human_description": "x equals x is always true",
                "z3_encoding": "x = z3.Real('x'); x == x",
            },
            {
                "property_name": "impossible_bound",
                "human_description": "x is always less than 0",
                "z3_encoding": "x = z3.Int('x'); x < 0",
            },
        ])

        engine = _make_engine(llm=mock_llm)

        # First call returns PROVED, second returns COUNTEREXAMPLE
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                (SymbolicExecutionStatus.PROVED, ""),
                (SymbolicExecutionStatus.COUNTEREXAMPLE, "x=0"),
            ],
        )

        result = await engine.prove_properties(
            files=["finance/allocator.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.BUDGET_CALCULATION],
        )

        # Overall status is COUNTEREXAMPLE when any property fails
        assert result.status == SymbolicExecutionStatus.COUNTEREXAMPLE
        assert result.properties_proved == 1
        assert result.properties_failed == 1
        assert result.properties_checked == 2

    @pytest.mark.asyncio
    async def test_prove_properties_defaults_to_all_domains(self, tmp_path: Path):
        """When domains=None, all SymbolicDomain values are searched."""
        _write_py(
            tmp_path,
            "multi/service.py",
            """\
            def calculate_cost(x: float) -> float:
                return x * 1.1

            def authorize_user(role: str) -> bool:
                return role == "admin"
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "always_true",
                "human_description": "Tautology",
                "z3_encoding": "x = z3.Real('x'); x == x",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.PROVED, ""),
        )

        result = await engine.prove_properties(
            files=["multi/service.py"],
            codebase_root=tmp_path,
            domains=None,  # defaults to all
        )

        # Should have found functions in both BUDGET_CALCULATION and ACCESS_CONTROL
        assert result.properties_checked >= 2

    @pytest.mark.asyncio
    async def test_prove_properties_records_target_file(self, tmp_path: Path):
        """Each SymbolicProperty should have target_file set to the source file path."""
        _write_py(
            tmp_path,
            "auth/gate.py",
            """\
            def check_access(level: int) -> bool:
                return level > 0
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "level_tautology",
                "human_description": "x == x",
                "z3_encoding": "x = z3.Int('x'); x == x",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.PROVED, ""),
        )

        result = await engine.prove_properties(
            files=["auth/gate.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.ACCESS_CONTROL],
        )

        assert len(result.properties) == 1
        assert result.properties[0].target_file == "auth/gate.py"

    @pytest.mark.asyncio
    async def test_prove_properties_duration_is_set(self, tmp_path: Path):
        """Result should have a non-negative duration_ms."""
        _write_py(
            tmp_path,
            "finance/calc.py",
            """\
            def budget_total(a: float, b: float) -> float:
                return a + b
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "sum_commutative",
                "human_description": "a + b == b + a",
                "z3_encoding": "a = z3.Real('a'); b = z3.Real('b'); a + b == b + a",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.PROVED, ""),
        )

        result = await engine.prove_properties(
            files=["finance/calc.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.BUDGET_CALCULATION],
        )

        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_prove_properties_path_conditions_count(self, tmp_path: Path):
        """path_conditions_explored should equal the number of properties checked."""
        _write_py(
            tmp_path,
            "governance/approver.py",
            """\
            def approve_vote(yes: int, no: int) -> bool:
                return yes > no
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "prop1",
                "human_description": "Tautology",
                "z3_encoding": "x = z3.Int('x'); x == x",
            },
            {
                "property_name": "prop2",
                "human_description": "Another tautology",
                "z3_encoding": "y = z3.Int('y'); y == y",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.PROVED, ""),
        )

        result = await engine.prove_properties(
            files=["governance/approver.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.GOVERNANCE_GATING],
        )

        assert result.path_conditions_explored == result.properties_checked

    @pytest.mark.asyncio
    async def test_prove_properties_counterexample_format(self, tmp_path: Path):
        """Counterexample strings should include file:function and property name."""
        _write_py(
            tmp_path,
            "gov/gate.py",
            """\
            def governance_check(level: int) -> bool:
                return level > 0
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "always_high",
                "human_description": "Level always > 100",
                "z3_encoding": "x = z3.Int('x'); x > 100",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.COUNTEREXAMPLE, "level=0"),
        )

        result = await engine.prove_properties(
            files=["gov/gate.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.GOVERNANCE_GATING],
        )

        assert len(result.counterexamples) == 1
        ce = result.counterexamples[0]
        assert "gov/gate.py" in ce
        assert "governance_check" in ce
        assert "always_high" in ce
        assert "level=0" in ce


# ── Z3 _check_property Internals (with mocked z3 module) ────────────────────


class TestCheckProperty:
    """Test the _check_property method by mocking the z3 import."""

    @pytest.mark.asyncio
    async def test_check_property_skips_empty_encoding(self):
        """Empty z3_encoding should result in SKIPPED status without touching z3."""
        engine = _make_engine()
        prop = SymbolicProperty(
            domain=SymbolicDomain.BUDGET_CALCULATION,
            property_name="empty",
            z3_encoding="",
            human_description="No encoding",
        )

        status, detail = await engine._check_property(prop)
        assert status == SymbolicExecutionStatus.SKIPPED
        assert detail == ""

    @pytest.mark.asyncio
    async def test_check_property_proved_via_unsat(self):
        """When z3.Solver.check() returns unsat, status should be PROVED."""
        mock_z3 = _build_mock_z3()

        # Configure Solver mock
        mock_solver_instance = MagicMock()
        mock_solver_instance.check.return_value = mock_z3.unsat
        mock_solver_class = MagicMock(return_value=mock_solver_instance)
        mock_z3.Solver = mock_solver_class  # type: ignore[attr-defined]

        engine = _make_engine()
        prop = SymbolicProperty(
            domain=SymbolicDomain.BUDGET_CALCULATION,
            property_name="identity",
            z3_encoding="x = z3.Real('x'); result = (x == x)",
            human_description="x equals x",
        )

        with patch.dict(sys.modules, {"z3": mock_z3}):
            status, detail = await engine._check_property(prop)

        assert status == SymbolicExecutionStatus.PROVED
        assert detail == ""

    @pytest.mark.asyncio
    async def test_check_property_counterexample_via_sat(self):
        """When z3.Solver.check() returns sat, status should be COUNTEREXAMPLE."""
        mock_z3 = _build_mock_z3()

        # Configure model with a counterexample
        mock_decl = MagicMock()
        mock_decl.name.return_value = "x"
        mock_model = MagicMock()
        mock_model.decls.return_value = [mock_decl]
        mock_model.__getitem__ = lambda self, key: 42

        mock_solver_instance = MagicMock()
        mock_solver_instance.check.return_value = mock_z3.sat
        mock_solver_instance.model.return_value = mock_model
        mock_solver_class = MagicMock(return_value=mock_solver_instance)
        mock_z3.Solver = mock_solver_class  # type: ignore[attr-defined]

        engine = _make_engine()
        prop = SymbolicProperty(
            domain=SymbolicDomain.RISK_SCORING,
            property_name="always_negative",
            z3_encoding="x = z3.Int('x'); result = (x < 0)",
            human_description="x is always negative (false)",
        )

        with patch.dict(sys.modules, {"z3": mock_z3}):
            status, detail = await engine._check_property(prop)

        assert status == SymbolicExecutionStatus.COUNTEREXAMPLE
        assert "x=" in detail

    @pytest.mark.asyncio
    async def test_check_property_unknown_when_solver_returns_unknown(self):
        """When z3.Solver.check() returns neither sat nor unsat, status is TIMEOUT."""
        mock_z3 = _build_mock_z3()

        mock_solver_instance = MagicMock()
        mock_solver_instance.check.return_value = mock_z3.unknown
        mock_solver_class = MagicMock(return_value=mock_solver_instance)
        mock_z3.Solver = mock_solver_class  # type: ignore[attr-defined]

        engine = _make_engine()
        prop = SymbolicProperty(
            domain=SymbolicDomain.BUDGET_CALCULATION,
            property_name="hard_prop",
            z3_encoding="x = z3.Int('x'); result = (x == x)",
            human_description="A hard property",
        )

        with patch.dict(sys.modules, {"z3": mock_z3}):
            status, detail = await engine._check_property(prop)

        assert status == SymbolicExecutionStatus.TIMEOUT
        assert "Z3 returned unknown" in detail

    @pytest.mark.asyncio
    async def test_check_property_unknown_when_no_boolref_produced(self):
        """If exec produces no z3.BoolRef, result should be UNKNOWN."""
        mock_z3 = _build_mock_z3()

        # Override Int to return a plain MagicMock (not a BoolRef)
        plain_mock = MagicMock()
        # Ensure isinstance check fails for BoolRef
        mock_z3.Int = lambda name: plain_mock  # type: ignore[attr-defined]

        mock_solver_class = MagicMock()
        mock_z3.Solver = mock_solver_class  # type: ignore[attr-defined]

        engine = _make_engine()
        prop = SymbolicProperty(
            domain=SymbolicDomain.BUDGET_CALCULATION,
            property_name="no_bool",
            z3_encoding="x = z3.Int('x')",
            human_description="Declares variable but no boolean expression",
        )

        with patch.dict(sys.modules, {"z3": mock_z3}):
            status, detail = await engine._check_property(prop)

        assert status == SymbolicExecutionStatus.UNKNOWN
        assert "No Z3 BoolRef produced" in detail

    @pytest.mark.asyncio
    async def test_check_property_unknown_on_exec_error(self):
        """Malformed Z3 encoding that raises an exception should return UNKNOWN."""
        mock_z3 = _build_mock_z3()
        mock_z3.Solver = MagicMock()  # type: ignore[attr-defined]

        engine = _make_engine()
        prop = SymbolicProperty(
            domain=SymbolicDomain.BUDGET_CALCULATION,
            property_name="bad_code",
            z3_encoding="import z3; raise ValueError('intentional error')",
            human_description="Broken encoding",
        )

        with patch.dict(sys.modules, {"z3": mock_z3}):
            status, detail = await engine._check_property(prop)
        assert status == SymbolicExecutionStatus.UNKNOWN
        assert "intentional error" in detail

    @pytest.mark.asyncio
    async def test_check_property_counterexample_includes_multiple_vars(self):
        """Counterexample should include all variable values from the model."""
        mock_z3 = _build_mock_z3()

        mock_decl_a = MagicMock()
        mock_decl_a.name.return_value = "a"
        mock_decl_b = MagicMock()
        mock_decl_b.name.return_value = "b"
        mock_model = MagicMock()
        mock_model.decls.return_value = [mock_decl_a, mock_decl_b]
        mock_model.__getitem__ = lambda self, key: 99

        mock_solver_instance = MagicMock()
        mock_solver_instance.check.return_value = mock_z3.sat
        mock_solver_instance.model.return_value = mock_model
        mock_solver_class = MagicMock(return_value=mock_solver_instance)
        mock_z3.Solver = mock_solver_class  # type: ignore[attr-defined]

        engine = _make_engine()
        prop = SymbolicProperty(
            domain=SymbolicDomain.BUDGET_CALCULATION,
            property_name="sum_bounded",
            z3_encoding=(
                "a = z3.Int('a'); b = z3.Int('b'); "
                "result = z3.And(a >= 0, b >= 0)"
            ),
            human_description="Both values non-negative",
        )

        with patch.dict(sys.modules, {"z3": mock_z3}):
            status, detail = await engine._check_property(prop)

        assert status == SymbolicExecutionStatus.COUNTEREXAMPLE
        assert "a=" in detail
        assert "b=" in detail

    @pytest.mark.asyncio
    async def test_check_property_solver_timeout_is_set(self):
        """The Z3 solver should receive the engine's timeout_ms configuration."""
        mock_z3 = _build_mock_z3()

        mock_solver_instance = MagicMock()
        mock_solver_instance.check.return_value = mock_z3.unsat
        mock_solver_class = MagicMock(return_value=mock_solver_instance)
        mock_z3.Solver = mock_solver_class  # type: ignore[attr-defined]

        engine = _make_engine(timeout_ms=7777)
        prop = SymbolicProperty(
            domain=SymbolicDomain.BUDGET_CALCULATION,
            property_name="test_timeout_config",
            z3_encoding="x = z3.Real('x'); result = (x == x)",
            human_description="Verify solver timeout is set",
        )

        with patch.dict(sys.modules, {"z3": mock_z3}):
            await engine._check_property(prop)

        mock_solver_instance.set.assert_called_once_with("timeout", 7777)


# ── Timeout Handling ─────────────────────────────────────────────────────────


class TestTimeoutHandling:
    """Test that Z3 solver timeout is respected and reported correctly."""

    @pytest.mark.asyncio
    async def test_timeout_error_returns_timeout_status(self):
        """A TimeoutError during Z3 check should return TIMEOUT status."""
        mock_z3 = _build_mock_z3()
        mock_z3.Solver = MagicMock()  # type: ignore[attr-defined]

        engine = _make_engine()
        prop = SymbolicProperty(
            domain=SymbolicDomain.BUDGET_CALCULATION,
            property_name="timeout_prop",
            z3_encoding="import z3; raise TimeoutError('solver timed out')",
            human_description="Simulated timeout",
        )

        with patch.dict(sys.modules, {"z3": mock_z3}):
            status, detail = await engine._check_property(prop)
        assert status == SymbolicExecutionStatus.TIMEOUT
        assert detail == ""

    @pytest.mark.asyncio
    async def test_engine_timeout_ms_is_configurable(self):
        """The timeout_ms parameter should be stored on the engine."""
        engine = _make_engine(timeout_ms=30000)
        assert engine._timeout_ms == 30000

    @pytest.mark.asyncio
    async def test_default_timeout_ms(self):
        """Default timeout should be 10000ms."""
        engine = SymbolicExecutionEngine()
        assert engine._timeout_ms == 10000

    @pytest.mark.asyncio
    async def test_prove_properties_returns_unknown_when_all_encodings_fail(
        self, tmp_path: Path,
    ):
        """If all property checks return UNKNOWN, overall status is UNKNOWN."""
        _write_py(
            tmp_path,
            "finance/broken.py",
            """\
            def calculate_budget_total(x: float) -> float:
                return x
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "broken_property",
                "human_description": "This will fail to produce BoolRef",
                "z3_encoding": "import z3; x = z3.Int('x')",
            },
        ])

        engine = _make_engine(llm=mock_llm)

        # Mock _check_property to return UNKNOWN (simulating no BoolRef)
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.UNKNOWN, "No Z3 BoolRef produced"),
        )

        result = await engine.prove_properties(
            files=["finance/broken.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.BUDGET_CALCULATION],
        )

        # No proved, no failed counterexample -- status falls through to UNKNOWN
        assert result.status == SymbolicExecutionStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_prove_properties_timeout_property_does_not_count_as_proved(
        self, tmp_path: Path,
    ):
        """A TIMEOUT property should not increment proved or failed counts."""
        _write_py(
            tmp_path,
            "risk/hard.py",
            """\
            def risk_assessment(x: float) -> float:
                return x
            """,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _llm_json_response([
            {
                "property_name": "hard_property",
                "human_description": "Hard to decide",
                "z3_encoding": "x = z3.Int('x'); x > 0",
            },
        ])

        engine = _make_engine(llm=mock_llm)
        engine._check_property = AsyncMock(  # type: ignore[method-assign]
            return_value=(SymbolicExecutionStatus.TIMEOUT, "Z3 returned unknown"),
        )

        result = await engine.prove_properties(
            files=["risk/hard.py"],
            codebase_root=tmp_path,
            domains=[SymbolicDomain.RISK_SCORING],
        )

        assert result.properties_proved == 0
        assert result.properties_failed == 0
        assert result.properties_checked == 1


# ── Engine Configuration ─────────────────────────────────────────────────────


class TestEngineConfiguration:
    """Test engine initialisation and configuration."""

    def test_engine_stores_z3_bridge(self):
        """z3_bridge should be stored on the engine."""
        mock_bridge = MagicMock()
        engine = SymbolicExecutionEngine(z3_bridge=mock_bridge)
        assert engine._z3 is mock_bridge

    def test_engine_stores_llm(self):
        """LLM provider should be stored on the engine."""
        mock_llm = AsyncMock()
        engine = SymbolicExecutionEngine(llm=mock_llm)
        assert engine._llm is mock_llm

    def test_engine_stores_blocking_flag(self):
        """blocking flag should be stored on the engine."""
        engine = SymbolicExecutionEngine(blocking=False)
        assert engine._blocking is False

    def test_engine_defaults(self):
        """Default values: no z3, no llm, 10s timeout, blocking=True."""
        engine = SymbolicExecutionEngine()
        assert engine._z3 is None
        assert engine._llm is None
        assert engine._timeout_ms == 10000
        assert engine._blocking is True
