"""
Unit tests for Simula Formal Spec Generator (Stage 6C).

Tests Dafny spec generation, TLA+ model checking, Alloy property checking,
Self-Spec DSL generation, and the generate_all orchestration method.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.simula.evolution_types import ChangeCategory
from systems.simula.formal_specs.spec_generator import FormalSpecGenerator
from systems.simula.verification.types import (
    FormalSpecGenerationResult,
    FormalSpecKind,
    FormalSpecStatus,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_llm_response(content: str, input_tokens: int = 100, output_tokens: int = 200) -> MagicMock:
    """Create a mock LLM response with the expected attributes."""
    response = MagicMock()
    response.content = content
    response.input_tokens = input_tokens
    response.output_tokens = output_tokens
    return response


def _make_proposal(
    description: str = "Add caching layer for repeated queries",
    category: ChangeCategory = ChangeCategory.ADD_SYSTEM_CAPABILITY,
    target: str = "Nova",
    code_hint: str = "",
) -> MagicMock:
    """Create a mock EvolutionProposal with the fields spec_generator expects."""
    proposal = MagicMock()
    proposal.description = description
    proposal.category = category
    proposal.target = target

    change_spec = MagicMock()
    change_spec.code_hint = code_hint
    proposal.change_spec = change_spec

    return proposal


SAMPLE_PYTHON_SOURCE = '''\
def clamp(value: int, low: int, high: int) -> int:
    """Clamp value between low and high."""
    if value < low:
        return low
    if value > high:
        return high
    return value
'''

SAMPLE_DAFNY_SPEC = """\
method Clamp(value: int, low: int, high: int) returns (result: int)
    requires low <= high
    ensures low <= result <= high
    ensures (low <= value <= high) ==> result == value
    ensures value < low ==> result == low
    ensures value > high ==> result == high
{
    if value < low {
        result := low;
    } else if value > high {
        result := high;
    } else {
        result := value;
    }
}
"""

SAMPLE_TLA_SPEC = """\
---- MODULE Nova ----
VARIABLES state, cache

Init == state = "idle" /\\ cache = {}

Next == state' = "processing" /\\ cache' = cache

Spec == Init /\\ [][Next]_<<state, cache>>
====
"""

SAMPLE_ALLOY_MODEL = """\
sig System {
    state: one State
}

abstract sig State {}
one sig Idle, Processing extends State {}

fact {
    all s: System | s.state = Idle or s.state = Processing
}

assert NoInvalidState {
    all s: System | s.state in State
}

check NoInvalidState for 10
"""

SAMPLE_DSL_JSON = json.dumps({
    "dsl_name": "CacheSpec",
    "grammar": 'cache_rule ::= "CACHE" key "FOR" duration\nkey ::= STRING\nduration ::= NUMBER "s"',
    "examples": [
        'CACHE "user_profile" FOR 300s',
        'CACHE "query_result" FOR 60s',
        'CACHE "session_data" FOR 1800s',
    ],
    "description": "A DSL for specifying cache policies with key-based TTL rules.",
})


# ── TestDafnySpecGeneration ──────────────────────────────────────────────────


class TestDafnySpecGeneration:
    @pytest.mark.asyncio
    async def test_generates_dafny_spec_for_python_function(self, tmp_path: Path):
        """LLM returns valid Dafny code and result has GENERATED status when no bridge."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_DAFNY_SPEC))

        generator = FormalSpecGenerator(llm=mock_llm, dafny_bridge=None)
        results = await generator.generate_dafny_specs(["module.py"], tmp_path)

        assert len(results) == 1
        spec = results[0]
        assert spec.kind == FormalSpecKind.DAFNY
        assert spec.status == FormalSpecStatus.GENERATED
        assert spec.target_function == "clamp"
        assert spec.target_file == "module.py"
        assert spec.spec_source == SAMPLE_DAFNY_SPEC
        assert spec.verified is False
        assert spec.coverage_percent == 0.5
        assert spec.llm_tokens_used == 300

    @pytest.mark.asyncio
    async def test_dafny_spec_verified_with_bridge(self, tmp_path: Path):
        """When a DafnyBridge is provided and verifies, status is VERIFIED."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_DAFNY_SPEC))

        mock_dafny_result = MagicMock()
        mock_dafny_result.status = MagicMock()
        mock_dafny_result.status.value = "verified"
        mock_dafny_result.error_summary = ""

        mock_bridge = MagicMock()
        mock_bridge.verify_dafny_source = AsyncMock(return_value=mock_dafny_result)

        generator = FormalSpecGenerator(llm=mock_llm, dafny_bridge=mock_bridge)
        results = await generator.generate_dafny_specs(["module.py"], tmp_path)

        assert len(results) == 1
        spec = results[0]
        assert spec.status == FormalSpecStatus.VERIFIED
        assert spec.verified is True
        assert spec.coverage_percent == 1.0
        assert spec.verification_output == "Verified"

    @pytest.mark.asyncio
    async def test_dafny_spec_failed_verification_with_bridge(self, tmp_path: Path):
        """When DafnyBridge reports failure, status is GENERATED (not VERIFIED)."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_DAFNY_SPEC))

        mock_dafny_result = MagicMock()
        mock_dafny_result.status = MagicMock()
        mock_dafny_result.status.value = "failed"
        mock_dafny_result.error_summary = "postcondition might not hold"

        mock_bridge = MagicMock()
        mock_bridge.verify_dafny_source = AsyncMock(return_value=mock_dafny_result)

        generator = FormalSpecGenerator(llm=mock_llm, dafny_bridge=mock_bridge)
        results = await generator.generate_dafny_specs(["module.py"], tmp_path)

        assert len(results) == 1
        spec = results[0]
        assert spec.status == FormalSpecStatus.GENERATED
        assert spec.verified is False
        assert spec.verification_output == "postcondition might not hold"

    @pytest.mark.asyncio
    async def test_dafny_bridge_exception_is_handled(self, tmp_path: Path):
        """Exception in DafnyBridge.verify_dafny_source does not crash generation."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_DAFNY_SPEC))

        mock_bridge = MagicMock()
        mock_bridge.verify_dafny_source = AsyncMock(side_effect=RuntimeError("Dafny crashed"))

        generator = FormalSpecGenerator(llm=mock_llm, dafny_bridge=mock_bridge)
        results = await generator.generate_dafny_specs(["module.py"], tmp_path)

        assert len(results) == 1
        spec = results[0]
        assert spec.status == FormalSpecStatus.GENERATED
        assert spec.verified is False
        assert "Dafny verification error" in spec.verification_output

    @pytest.mark.asyncio
    async def test_skips_non_python_files(self, tmp_path: Path):
        """Non-.py files are silently skipped."""
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("not python", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        generator = FormalSpecGenerator(llm=mock_llm)
        results = await generator.generate_dafny_specs(["readme.txt"], tmp_path)

        assert results == []
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_missing_files(self, tmp_path: Path):
        """Files that don't exist on disk are silently skipped."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        generator = FormalSpecGenerator(llm=mock_llm)
        results = await generator.generate_dafny_specs(["nonexistent.py"], tmp_path)

        assert results == []
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_multiple_functions_in_one_file(self, tmp_path: Path):
        """Each function in a file gets its own Dafny spec."""
        source = '''\
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

async def fetch_data(url: str) -> str:
    return ""
'''
        py_file = tmp_path / "ops.py"
        py_file.write_text(source, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response("method Stub() {}"))

        generator = FormalSpecGenerator(llm=mock_llm)
        results = await generator.generate_dafny_specs(["ops.py"], tmp_path)

        assert len(results) == 3
        func_names = {r.target_function for r in results}
        assert func_names == {"add", "subtract", "fetch_data"}

    @pytest.mark.asyncio
    async def test_llm_failure_returns_failed_status(self, tmp_path: Path):
        """LLM exception results in FAILED status spec result."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        generator = FormalSpecGenerator(llm=mock_llm)
        results = await generator.generate_dafny_specs(["module.py"], tmp_path)

        assert len(results) == 1
        spec = results[0]
        assert spec.status == FormalSpecStatus.FAILED
        assert spec.target_function == "clamp"

    @pytest.mark.asyncio
    async def test_skips_files_with_syntax_errors(self, tmp_path: Path):
        """Python files with syntax errors are skipped without crashing."""
        py_file = tmp_path / "broken.py"
        py_file.write_text("def bad(:\n  pass", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        generator = FormalSpecGenerator(llm=mock_llm)
        results = await generator.generate_dafny_specs(["broken.py"], tmp_path)

        assert results == []
        mock_llm.complete.assert_not_called()


# ── TestTlaPlusGeneration ────────────────────────────────────────────────────


class TestTlaPlusGeneration:
    @pytest.mark.asyncio
    async def test_generates_tla_spec_and_model_checks_success(self):
        """Successful TLC run yields VERIFIED status with state counts."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_TLA_SPEC))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"42 states found\nModel checking complete\n", b""),
        )
        mock_proc.returncode = 0

        generator = FormalSpecGenerator(llm=mock_llm)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.generate_tla_plus_spec(
                system_name="Nova",
                interactions=["Cache lookup", "Cache invalidation"],
            )

        assert result.status == FormalSpecStatus.VERIFIED
        assert result.spec_source == SAMPLE_TLA_SPEC
        assert result.system_name == "Nova"
        assert result.states_explored == 42
        assert result.violations == []
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_tla_violations_yield_failed_status(self):
        """TLC output containing violations results in FAILED status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_TLA_SPEC))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(
                b"10 states found\nInvariant violation: CacheConsistency\n",
                b"",
            ),
        )
        mock_proc.returncode = 1

        generator = FormalSpecGenerator(llm=mock_llm)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.generate_tla_plus_spec(
                system_name="Nova",
                interactions=["Stale read"],
            )

        assert result.status == FormalSpecStatus.FAILED
        assert result.states_explored == 10
        assert len(result.violations) >= 1

    @pytest.mark.asyncio
    async def test_tla_timeout_yields_timeout_status(self):
        """TLC exceeding the timeout produces TIMEOUT status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_TLA_SPEC))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError("TLC too slow"))
        mock_proc.kill = MagicMock()

        generator = FormalSpecGenerator(llm=mock_llm, tla_plus_timeout_s=1.0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.generate_tla_plus_spec(
                system_name="Nova",
                interactions=["Timeout scenario"],
            )

        assert result.status == FormalSpecStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_tlc_not_found_yields_skipped(self):
        """Missing TLC binary produces SKIPPED status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_TLA_SPEC))

        generator = FormalSpecGenerator(llm=mock_llm, tla_plus_path="/nonexistent/tlc_fake")

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("tlc not found"),
        ):
            result = await generator.generate_tla_plus_spec(
                system_name="Nova",
                interactions=["Missing binary test"],
            )

        assert result.status == FormalSpecStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_tla_deadlock_detection(self):
        """TLC output reporting deadlock is captured in deadlocks_found."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_TLA_SPEC))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(
                b"5 states found\nDeadlock reached at state 5\nError: deadlock\n",
                b"",
            ),
        )
        mock_proc.returncode = 1

        generator = FormalSpecGenerator(llm=mock_llm)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.generate_tla_plus_spec(
                system_name="Nova",
                interactions=["Deadlock scenario"],
            )

        assert result.status == FormalSpecStatus.FAILED
        assert result.deadlocks_found >= 1

    @pytest.mark.asyncio
    async def test_llm_failure_returns_failed_tla_result(self):
        """LLM exception during TLA+ generation yields FAILED status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM crashed"))

        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_tla_plus_spec(
            system_name="Nova",
            interactions=["Error scenario"],
        )

        assert result.status == FormalSpecStatus.FAILED
        assert result.system_name == "Nova"


# ── TestAlloyCheck ───────────────────────────────────────────────────────────


class TestAlloyCheck:
    @pytest.mark.asyncio
    async def test_alloy_check_success(self, tmp_path: Path):
        """Successful Alloy run yields VERIFIED status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_ALLOY_MODEL))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"Executing check...\nNo instance found.\n", b""),
        )
        mock_proc.returncode = 0

        generator = FormalSpecGenerator(llm=mock_llm, alloy_scope=10)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.check_alloy_properties(
                properties=["All states are valid"],
                codebase_root=tmp_path,
            )

        assert result.status == FormalSpecStatus.VERIFIED
        assert result.model_source == SAMPLE_ALLOY_MODEL
        assert result.scope == 10
        assert result.counterexamples == []
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_alloy_counterexample_yields_failed(self, tmp_path: Path):
        """Alloy finding a counterexample results in FAILED status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_ALLOY_MODEL))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(
                b"Instance found.\nCounterexample: System$0 has invalid state\n",
                b"",
            ),
        )
        mock_proc.returncode = 1

        generator = FormalSpecGenerator(llm=mock_llm)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.check_alloy_properties(
                properties=["No invalid states"],
                codebase_root=tmp_path,
            )

        assert result.status == FormalSpecStatus.FAILED
        assert len(result.counterexamples) >= 1
        assert result.instances_found >= 1

    @pytest.mark.asyncio
    async def test_alloy_timeout_yields_timeout_status(self, tmp_path: Path):
        """Alloy analyzer exceeding timeout produces TIMEOUT status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_ALLOY_MODEL))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError("Alloy too slow"))
        mock_proc.kill = MagicMock()

        generator = FormalSpecGenerator(llm=mock_llm)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.check_alloy_properties(
                properties=["Timeout property"],
                codebase_root=tmp_path,
            )

        assert result.status == FormalSpecStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_alloy_not_found_yields_skipped(self, tmp_path: Path):
        """Missing Alloy binary produces SKIPPED status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_ALLOY_MODEL))

        generator = FormalSpecGenerator(
            llm=mock_llm, alloy_path="/nonexistent/alloy_fake",
        )

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("alloy not found"),
        ):
            result = await generator.check_alloy_properties(
                properties=["Missing binary"],
                codebase_root=tmp_path,
            )

        assert result.status == FormalSpecStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_llm_failure_returns_failed_alloy_result(self, tmp_path: Path):
        """LLM exception during Alloy generation yields FAILED status."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM failed"))

        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.check_alloy_properties(
            properties=["Error scenario"],
            codebase_root=tmp_path,
        )

        assert result.status == FormalSpecStatus.FAILED


# ── TestSelfSpecDSL ──────────────────────────────────────────────────────────


class TestSelfSpecDSL:
    @pytest.mark.asyncio
    async def test_generates_dsl_from_valid_json_response(self):
        """LLM returns valid JSON and DSL fields are populated."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=_make_llm_response(SAMPLE_DSL_JSON, input_tokens=50, output_tokens=150),
        )

        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_self_spec_dsl(
            category=ChangeCategory.ADD_SYSTEM_CAPABILITY,
            examples=["Add caching layer", "Add rate limiter"],
        )

        assert result.dsl_name == "CacheSpec"
        assert "cache_rule" in result.grammar_source
        assert len(result.example_programs) == 3
        assert result.target_category == "add_system_capability"
        assert result.llm_tokens_used == 200
        assert result.description != ""

    @pytest.mark.asyncio
    async def test_handles_json_embedded_in_text(self):
        """LLM may return JSON embedded in surrounding text."""
        text_with_json = f"Here is the DSL spec:\n{SAMPLE_DSL_JSON}\nEnd of spec."
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=_make_llm_response(text_with_json),
        )

        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_self_spec_dsl(
            category=ChangeCategory.ADD_SYSTEM_CAPABILITY,
            examples=["Example A"],
        )

        assert result.dsl_name == "CacheSpec"
        assert len(result.example_programs) == 3

    @pytest.mark.asyncio
    async def test_no_json_in_response_returns_empty_dsl(self):
        """When LLM response contains no parsable JSON, an empty DSL is returned."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=_make_llm_response("I can't generate a DSL for this."),
        )

        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_self_spec_dsl(
            category=ChangeCategory.ADJUST_BUDGET,
            examples=["Increase limit"],
        )

        assert result.dsl_name == ""
        assert result.grammar_source == ""
        assert result.example_programs == []
        assert result.target_category == "adjust_budget"

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_dsl(self):
        """LLM exception yields an empty DSL with category preserved."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_self_spec_dsl(
            category=ChangeCategory.MODIFY_CONTRACT,
            examples=["Contract change"],
        )

        assert result.dsl_name == ""
        assert result.target_category == "modify_contract"
        assert result.llm_tokens_used == 0

    @pytest.mark.asyncio
    async def test_partial_json_populates_available_fields(self):
        """JSON with only some fields still populates what's available."""
        partial_json = json.dumps({
            "dsl_name": "PartialSpec",
            "grammar": "rule ::= TOKEN",
        })
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=_make_llm_response(partial_json),
        )

        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_self_spec_dsl(
            category=ChangeCategory.ADD_EXECUTOR,
            examples=["Add new executor"],
        )

        assert result.dsl_name == "PartialSpec"
        assert result.grammar_source == "rule ::= TOKEN"
        assert result.example_programs == []


# ── TestGenerateAll ──────────────────────────────────────────────────────────


class TestGenerateAll:
    @pytest.mark.asyncio
    async def test_only_dafny_enabled(self, tmp_path: Path):
        """With only dafny_enabled, only Dafny specs are generated."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_DAFNY_SPEC))

        proposal = _make_proposal()
        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_all(
            files=["module.py"],
            proposal=proposal,
            codebase_root=tmp_path,
            dafny_enabled=True,
            tla_plus_enabled=False,
            alloy_enabled=False,
            self_spec_enabled=False,
        )

        assert isinstance(result, FormalSpecGenerationResult)
        assert len(result.specs) == 1
        assert result.tla_plus_results == []
        assert result.alloy_results == []
        assert result.self_spec_dsls == []
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_all_generators_enabled(self, tmp_path: Path):
        """When all flags are enabled, all generator types run in parallel."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()

        # The LLM.complete is called multiple times for different generators.
        # Return appropriate responses based on call order.
        dafny_resp = _make_llm_response(SAMPLE_DAFNY_SPEC)
        tla_resp = _make_llm_response(SAMPLE_TLA_SPEC)
        alloy_resp = _make_llm_response(SAMPLE_ALLOY_MODEL)
        dsl_resp = _make_llm_response(SAMPLE_DSL_JSON, input_tokens=50, output_tokens=150)
        mock_llm.complete = AsyncMock(
            side_effect=[dafny_resp, tla_resp, alloy_resp, dsl_resp],
        )

        # Mock subprocess calls for TLA+ and Alloy
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"10 states found\nModel checking complete\n", b""),
        )
        mock_proc.returncode = 0

        proposal = _make_proposal()
        generator = FormalSpecGenerator(llm=mock_llm)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.generate_all(
                files=["module.py"],
                proposal=proposal,
                codebase_root=tmp_path,
                dafny_enabled=True,
                tla_plus_enabled=True,
                alloy_enabled=True,
                self_spec_enabled=True,
            )

        assert len(result.specs) >= 1
        assert len(result.tla_plus_results) >= 1
        assert len(result.alloy_results) >= 1
        assert len(result.self_spec_dsls) >= 1

    @pytest.mark.asyncio
    async def test_no_generators_enabled(self, tmp_path: Path):
        """With all flags disabled, the result is empty."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        proposal = _make_proposal()
        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_all(
            files=["module.py"],
            proposal=proposal,
            codebase_root=tmp_path,
            dafny_enabled=False,
            tla_plus_enabled=False,
            alloy_enabled=False,
            self_spec_enabled=False,
        )

        assert result.specs == []
        assert result.tla_plus_results == []
        assert result.alloy_results == []
        assert result.self_spec_dsls == []
        assert result.overall_coverage_percent == 0.0
        assert result.total_llm_tokens == 0
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_coverage_calculation_all_verified(self, tmp_path: Path):
        """Coverage is 100% when all specs are verified."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_DAFNY_SPEC))

        mock_dafny_result = MagicMock()
        mock_dafny_result.status = MagicMock()
        mock_dafny_result.status.value = "verified"
        mock_dafny_result.error_summary = ""

        mock_bridge = MagicMock()
        mock_bridge.verify_dafny_source = AsyncMock(return_value=mock_dafny_result)

        proposal = _make_proposal()
        generator = FormalSpecGenerator(llm=mock_llm, dafny_bridge=mock_bridge)
        result = await generator.generate_all(
            files=["module.py"],
            proposal=proposal,
            codebase_root=tmp_path,
            dafny_enabled=True,
        )

        assert result.overall_coverage_percent == 1.0

    @pytest.mark.asyncio
    async def test_coverage_calculation_none_verified(self, tmp_path: Path):
        """Coverage is 0% when no specs are verified (bridge absent)."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=_make_llm_response(SAMPLE_DAFNY_SPEC))

        proposal = _make_proposal()
        generator = FormalSpecGenerator(llm=mock_llm, dafny_bridge=None)
        result = await generator.generate_all(
            files=["module.py"],
            proposal=proposal,
            codebase_root=tmp_path,
            dafny_enabled=True,
        )

        assert result.overall_coverage_percent == 0.0

    @pytest.mark.asyncio
    async def test_task_exception_is_handled_gracefully(self, tmp_path: Path):
        """If one generator task throws, others still succeed."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        call_count = 0

        async def side_effect_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            # First call (Dafny) succeeds, second call (TLA+) fails
            if call_count == 1:
                return _make_llm_response(SAMPLE_DAFNY_SPEC)
            raise RuntimeError("TLA+ generation exploded")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=side_effect_complete)

        proposal = _make_proposal()
        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_all(
            files=["module.py"],
            proposal=proposal,
            codebase_root=tmp_path,
            dafny_enabled=True,
            tla_plus_enabled=True,
        )

        # Dafny specs should still be present even though TLA+ failed
        assert len(result.specs) >= 1
        # TLA+ may have a FAILED result or be absent depending on error handling
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_token_accounting(self, tmp_path: Path):
        """Total token count aggregates across all spec generators."""
        py_file = tmp_path / "module.py"
        py_file.write_text(SAMPLE_PYTHON_SOURCE, encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=_make_llm_response(SAMPLE_DAFNY_SPEC, input_tokens=200, output_tokens=300),
        )

        proposal = _make_proposal()
        generator = FormalSpecGenerator(llm=mock_llm, dafny_bridge=None)
        result = await generator.generate_all(
            files=["module.py"],
            proposal=proposal,
            codebase_root=tmp_path,
            dafny_enabled=True,
        )

        # 200 input + 300 output = 500 tokens for the one function
        assert result.total_llm_tokens == 500

    @pytest.mark.asyncio
    async def test_empty_file_list_produces_empty_result(self, tmp_path: Path):
        """No files means no Dafny specs, zero coverage."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        proposal = _make_proposal()
        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_all(
            files=[],
            proposal=proposal,
            codebase_root=tmp_path,
            dafny_enabled=True,
        )

        assert result.specs == []
        assert result.overall_coverage_percent == 0.0

    @pytest.mark.asyncio
    async def test_generate_all_with_proposal_without_target(self, tmp_path: Path):
        """Proposal with no target attribute falls back to UnknownSystem for TLA+."""
        mock_llm = MagicMock()

        tla_resp = _make_llm_response(SAMPLE_TLA_SPEC)
        mock_llm.complete = AsyncMock(return_value=tla_resp)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"5 states found\n", b""),
        )
        mock_proc.returncode = 0

        proposal = MagicMock()
        proposal.description = "Test proposal"
        proposal.category = ChangeCategory.ADD_SYSTEM_CAPABILITY
        proposal.target = None  # No target set
        proposal.change_spec = None

        generator = FormalSpecGenerator(llm=mock_llm)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await generator.generate_all(
                files=[],
                proposal=proposal,
                codebase_root=tmp_path,
                dafny_enabled=False,
                tla_plus_enabled=True,
            )

        assert len(result.tla_plus_results) == 1
        assert result.tla_plus_results[0].system_name == "UnknownSystem"

    @pytest.mark.asyncio
    async def test_self_spec_uses_proposal_category_and_description(self, tmp_path: Path):
        """Self-spec DSL generation uses the proposal's category and description."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=_make_llm_response(SAMPLE_DSL_JSON, input_tokens=40, output_tokens=160),
        )

        proposal = _make_proposal(
            description="Implement budget throttling",
            category=ChangeCategory.ADJUST_BUDGET,
            code_hint="def throttle(budget): ...",
        )

        generator = FormalSpecGenerator(llm=mock_llm)
        result = await generator.generate_all(
            files=[],
            proposal=proposal,
            codebase_root=tmp_path,
            dafny_enabled=False,
            self_spec_enabled=True,
        )

        assert len(result.self_spec_dsls) == 1
        dsl = result.self_spec_dsls[0]
        assert dsl.target_category == "adjust_budget"
        assert dsl.dsl_name == "CacheSpec"
        assert result.total_llm_tokens == 200
