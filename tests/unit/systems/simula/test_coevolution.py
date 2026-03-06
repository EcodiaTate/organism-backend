"""
Unit tests for Simula Co-Evolution (Stage 6B).

Tests hard negative mining from rollback history and verification failures,
adversarial test generation, GRPO batch formatting, the full co-evolution
cycle, and coverage_growth metric tracking.
"""

from __future__ import annotations

# The types module uses `from __future__ import annotations` with `datetime`
# under TYPE_CHECKING, so Pydantic needs an explicit model_rebuild() to
# resolve the deferred `mined_at: datetime` field at runtime.
from datetime import datetime as _datetime  # noqa: E402
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from systems.simula.coevolution.adversarial_tester import (
    RobustnessTestGenerator,
)
from systems.simula.coevolution.hard_negative_miner import FailureAnalyzer

import systems.simula.verification.types as _vtypes
from systems.simula.verification.types import (
    CoevolutionCycleResult,
    FailureCaseExample,
    FailureCaseSource,
    RobustnessTestResult,
)

if TYPE_CHECKING:
    from pathlib import Path

_vtypes.datetime = _datetime  # type: ignore[attr-defined]
FailureCaseExample.model_rebuild()
RobustnessTestResult.model_rebuild()
CoevolutionCycleResult.model_rebuild()


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_negative(
    source: FailureCaseSource = FailureCaseSource.ROLLBACK_HISTORY,
    proposal_id: str = "prop-1",
    category: str = "modify_config",
    failure_reason: str = "test failed",
    code_context: str = "def broken(): pass",
    adversarial_input: str = "",
) -> FailureCaseExample:
    return FailureCaseExample(
        source=source,
        proposal_id=proposal_id,
        category=category,
        failure_reason=failure_reason,
        code_context=code_context,
        adversarial_input=adversarial_input,
    )


def _make_neo4j_mock(
    rollback_rows: list[dict[str, str]] | None = None,
    verification_rows: list[dict[str, str]] | None = None,
) -> AsyncMock:
    """Build a mock Neo4jClient whose execute_read returns rollback then verification rows."""
    neo4j = AsyncMock()
    side_effects: list[list[dict[str, str]]] = [
        rollback_rows if rollback_rows is not None else [],
        verification_rows if verification_rows is not None else [],
    ]
    neo4j.execute_read = AsyncMock(side_effect=side_effects)
    return neo4j


def _make_llm_mock(response_text: str = "") -> AsyncMock:
    """Build a mock LLMProvider whose complete method returns a response with .content."""
    llm = AsyncMock()
    response = MagicMock()
    response.content = response_text
    llm.complete = AsyncMock(return_value=response)
    return llm


# ── TestFailureAnalyzer ───────────────────────────────────────────────────


class TestFailureAnalyzer:
    @pytest.mark.asyncio
    async def test_mine_from_history_no_neo4j_returns_empty(self):
        """When neo4j is None, mine_from_history should return empty list."""
        miner = FailureAnalyzer(neo4j=None, llm=None)

        result = await miner.mine_from_history()

        assert result == []

    @pytest.mark.asyncio
    async def test_mine_from_history_rollbacks(self):
        """Rolled-back proposals should be mined as ROLLBACK_HISTORY negatives."""
        rollback_rows = [
            {
                "proposal_id": "prop-101",
                "category": "modify_config",
                "description": "Changed config parser",
                "rollback_reason": "Broke YAML loading",
            },
            {
                "proposal_id": "prop-102",
                "category": "add_feature",
                "description": "Added caching layer",
                "rollback_reason": "Cache invalidation bug",
            },
        ]
        neo4j = _make_neo4j_mock(rollback_rows=rollback_rows, verification_rows=[])

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)
        result = await miner.mine_from_history()

        assert len(result) == 2
        assert result[0].source == FailureCaseSource.ROLLBACK_HISTORY
        assert result[0].proposal_id == "prop-101"
        assert result[0].failure_reason == "Broke YAML loading"
        assert result[0].code_context == "Changed config parser"
        assert result[1].proposal_id == "prop-102"

    @pytest.mark.asyncio
    async def test_mine_from_history_verification_failures(self):
        """Verification-failed proposals should be mined as FORMAL_VERIFICATION_FAILURE."""
        verification_rows = [
            {
                "proposal_id": "prop-201",
                "category": "modify_contract",
                "description": "Updated risk scoring",
                "fv_status": "failed",
                "lean_status": "failed",
            },
        ]
        neo4j = _make_neo4j_mock(rollback_rows=[], verification_rows=verification_rows)

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)
        result = await miner.mine_from_history()

        assert len(result) == 1
        neg = result[0]
        assert neg.source == FailureCaseSource.FORMAL_VERIFICATION_FAILURE
        assert neg.proposal_id == "prop-201"
        assert "failed" in neg.failure_reason
        assert neg.code_context == "Updated risk scoring"

    @pytest.mark.asyncio
    async def test_mine_from_history_combines_both_sources(self):
        """Results from rollbacks and verification failures are combined."""
        rollback_rows = [
            {
                "proposal_id": "prop-301",
                "category": "modify_config",
                "description": "desc-301",
                "rollback_reason": "reason-301",
            },
        ]
        verification_rows = [
            {
                "proposal_id": "prop-302",
                "category": "modify_contract",
                "description": "desc-302",
                "fv_status": "failed",
                "lean_status": "failed",
            },
        ]
        neo4j = _make_neo4j_mock(
            rollback_rows=rollback_rows,
            verification_rows=verification_rows,
        )

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)
        result = await miner.mine_from_history()

        assert len(result) == 2
        sources = {neg.source for neg in result}
        assert FailureCaseSource.ROLLBACK_HISTORY in sources
        assert FailureCaseSource.FORMAL_VERIFICATION_FAILURE in sources

    @pytest.mark.asyncio
    async def test_mine_from_history_respects_max_cap(self):
        """Negatives should be capped at max_negatives_per_cycle."""
        rollback_rows = [
            {
                "proposal_id": f"prop-{i}",
                "category": "cat",
                "description": f"desc-{i}",
                "rollback_reason": f"reason-{i}",
            }
            for i in range(10)
        ]
        neo4j = _make_neo4j_mock(rollback_rows=rollback_rows, verification_rows=[])

        miner = FailureAnalyzer(neo4j=neo4j, llm=None, max_negatives_per_cycle=3)
        result = await miner.mine_from_history()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_mine_from_history_handles_missing_fields(self):
        """Rows with missing optional fields should not raise."""
        rollback_rows = [
            {
                "proposal_id": "prop-400",
                "category": "cat",
                # description and rollback_reason are missing
            },
        ]
        neo4j = _make_neo4j_mock(rollback_rows=rollback_rows, verification_rows=[])

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)
        result = await miner.mine_from_history()

        assert len(result) == 1
        assert result[0].failure_reason == ""
        assert result[0].code_context == ""

    @pytest.mark.asyncio
    async def test_prepare_grpo_batch_formats_correctly(self):
        """GRPO batch should have reward=0.0, source as string value, and all fields."""
        negatives = [
            _make_negative(
                source=FailureCaseSource.ROLLBACK_HISTORY,
                proposal_id="prop-1",
                category="modify_config",
                code_context="def broken(): pass",
                failure_reason="import error",
            ),
            _make_negative(
                source=FailureCaseSource.ADVERSARIAL_GENERATION,
                proposal_id="prop-2",
                category="add_feature",
                code_context="class Bad: ...",
                failure_reason="assertion failed",
            ),
        ]

        miner = FailureAnalyzer(neo4j=None, llm=None)
        batch = await miner.prepare_grpo_batch(negatives)

        assert len(batch) == 2

        first = batch[0]
        assert first["proposal_id"] == "prop-1"
        assert first["category"] == "modify_config"
        assert first["code_output"] == "def broken(): pass"
        assert first["failure_reason"] == "import error"
        assert first["reward"] == 0.0
        assert first["source"] == "rollback_history"

        second = batch[1]
        assert second["reward"] == 0.0
        assert second["source"] == "adversarial_generation"

    @pytest.mark.asyncio
    async def test_prepare_grpo_batch_empty_input(self):
        """An empty list of negatives should produce an empty GRPO batch."""
        miner = FailureAnalyzer(neo4j=None, llm=None)
        batch = await miner.prepare_grpo_batch([])

        assert batch == []

    @pytest.mark.asyncio
    async def test_mine_from_robustness_converts_bugs_to_negatives(self):
        """Adversarial bugs should be converted to ADVERSARIAL_GENERATION negatives."""
        miner = FailureAnalyzer(neo4j=None, llm=None)

        mock_generator = AsyncMock(spec=RobustnessTestGenerator)
        mock_generator.generate_robustness_tests = AsyncMock(
            return_value=RobustnessTestResult(
                tests_generated=5,
                tests_executed=5,
                tests_found_bugs=2,
                bug_descriptions=["FAILED test_edge_case", "FAILED test_overflow"],
            ),
        )

        result = await miner.mine_from_robustness(mock_generator, files=["app.py"])

        assert len(result) == 2
        assert all(
            neg.source == FailureCaseSource.ADVERSARIAL_GENERATION for neg in result
        )
        assert result[0].failure_reason == "FAILED test_edge_case"
        assert result[0].adversarial_input == "FAILED test_edge_case"
        assert result[1].failure_reason == "FAILED test_overflow"

    @pytest.mark.asyncio
    async def test_mine_from_robustness_no_bugs_returns_empty(self):
        """When adversarial tests find no bugs, result should be empty."""
        miner = FailureAnalyzer(neo4j=None, llm=None)

        mock_generator = AsyncMock(spec=RobustnessTestGenerator)
        mock_generator.generate_robustness_tests = AsyncMock(
            return_value=RobustnessTestResult(
                tests_generated=5,
                tests_executed=5,
                tests_found_bugs=0,
                bug_descriptions=[],
            ),
        )

        result = await miner.mine_from_robustness(mock_generator, files=["app.py"])

        assert result == []

    @pytest.mark.asyncio
    async def test_mine_from_robustness_passes_history_to_generator(self):
        """mine_from_robustness should pass past_failures from mine_from_history."""
        rollback_rows = [
            {
                "proposal_id": "prop-500",
                "category": "cat",
                "description": "d",
                "rollback_reason": "r",
            },
        ]
        neo4j = _make_neo4j_mock(rollback_rows=rollback_rows, verification_rows=[])

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)

        mock_generator = AsyncMock(spec=RobustnessTestGenerator)
        mock_generator.generate_robustness_tests = AsyncMock(
            return_value=RobustnessTestResult(
                tests_generated=1,
                tests_executed=1,
                tests_found_bugs=0,
                bug_descriptions=[],
            ),
        )

        await miner.mine_from_robustness(mock_generator, files=["a.py"])

        call_args = mock_generator.generate_robustness_tests.call_args
        past_failures = call_args.kwargs.get(
            "past_failures", call_args.args[1] if len(call_args.args) > 1 else None
        )
        # The past failures come from mine_from_history (1 rollback)
        assert past_failures is not None
        assert len(past_failures) == 1
        assert past_failures[0].proposal_id == "prop-500"


# ── TestRobustnessTestGenerator ────────────────────────────────────────────


class TestRobustnessTestGenerator:
    @pytest.mark.asyncio
    async def test_generate_tests_with_llm(self, tmp_path: Path):
        """LLM-generated test code should be written, parsed, and executed."""
        test_code = (
            "import pytest\n\n"
            "def test_edge_empty():\n"
            "    assert len([]) == 0\n\n"
            "def test_edge_none():\n"
            "    assert None is None\n"
        )
        llm = _make_llm_mock(response_text=test_code)

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
            max_tests_per_cycle=10,
            timeout_s=30.0,
        )

        # Patch _run_tests to avoid actual subprocess calls
        mock_result = RobustnessTestResult(
            tests_generated=2,
            tests_executed=2,
            tests_found_bugs=1,
            test_files_written=[str(tmp_path / "test_adv.py")],
            bug_descriptions=["FAILED test_edge_none"],
        )
        generator._run_tests = AsyncMock(return_value=mock_result)

        result = await generator.generate_robustness_tests(
            files=["module_a.py"],
            past_failures=[
                _make_negative(failure_reason="NoneType error"),
            ],
        )

        assert result.tests_generated == 2
        assert result.tests_executed == 2
        assert result.tests_found_bugs == 1
        assert len(result.bug_descriptions) == 1

    @pytest.mark.asyncio
    async def test_generate_tests_strips_markdown_fences(self, tmp_path: Path):
        """Markdown code fences in LLM output should be stripped before writing."""
        test_code_with_fences = (
            "```python\n"
            "def test_boundary():\n"
            "    assert 1 + 1 == 2\n"
            "```\n"
        )
        llm = _make_llm_mock(response_text=test_code_with_fences)

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )

        mock_result = RobustnessTestResult(tests_generated=1, tests_executed=1)
        generator._run_tests = AsyncMock(return_value=mock_result)

        await generator.generate_robustness_tests(files=["a.py"])

        # _run_tests should have been called with stripped code (no fences)
        called_code = generator._run_tests.call_args.args[0]
        assert "```" not in called_code
        assert "def test_boundary():" in called_code

    @pytest.mark.asyncio
    async def test_generate_tests_llm_failure_returns_empty_result(self, tmp_path: Path):
        """If the LLM call raises, return a result with zero tests."""
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("API down"))

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )

        result = await generator.generate_robustness_tests(files=["a.py"])

        assert result.tests_generated == 0
        assert result.tests_executed == 0
        assert result.tests_found_bugs == 0

    @pytest.mark.asyncio
    async def test_generate_tests_no_past_failures_uses_exploratory(self, tmp_path: Path):
        """When past_failures is empty, prompt should mention exploratory tests."""
        llm = _make_llm_mock(response_text="def test_explore(): pass")

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )
        generator._run_tests = AsyncMock(
            return_value=RobustnessTestResult(tests_generated=1),
        )

        await generator.generate_robustness_tests(files=["a.py"], past_failures=None)

        call_args = llm.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0].content
        assert "exploratory" in prompt_content.lower()

    @pytest.mark.asyncio
    async def test_generate_tests_limits_files_in_prompt(self, tmp_path: Path):
        """Only the first 5 files should be included in the LLM prompt."""
        llm = _make_llm_mock(response_text="def test_x(): pass")

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )
        generator._run_tests = AsyncMock(
            return_value=RobustnessTestResult(tests_generated=1),
        )

        many_files = [f"file_{i}.py" for i in range(20)]
        await generator.generate_robustness_tests(files=many_files)

        call_args = llm.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0].content
        # file_5 through file_19 should NOT be in the prompt
        assert "file_0.py" in prompt_content
        assert "file_4.py" in prompt_content
        assert "file_5.py" not in prompt_content

    @pytest.mark.asyncio
    async def test_generate_tests_limits_past_failures_in_prompt(self, tmp_path: Path):
        """Only the first 10 past failures should be included in the prompt."""
        llm = _make_llm_mock(response_text="def test_x(): pass")

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )
        generator._run_tests = AsyncMock(
            return_value=RobustnessTestResult(tests_generated=1),
        )

        many_failures = [
            _make_negative(
                category=f"cat_{i}",
                failure_reason=f"reason_{i}",
            )
            for i in range(20)
        ]

        await generator.generate_robustness_tests(
            files=["a.py"],
            past_failures=many_failures,
        )

        call_args = llm.complete.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0].content
        assert "reason_0" in prompt_content
        assert "reason_9" in prompt_content
        assert "reason_10" not in prompt_content

    @pytest.mark.asyncio
    async def test_generate_tests_updates_lifetime_stats(self, tmp_path: Path):
        """Lifetime stats should accumulate across multiple generation calls."""
        llm = _make_llm_mock(response_text="def test_x(): pass")

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )
        generator._run_tests = AsyncMock(
            return_value=RobustnessTestResult(
                tests_generated=4,
                tests_executed=4,
                tests_found_bugs=1,
            ),
        )

        await generator.generate_robustness_tests(files=["a.py"])
        await generator.generate_robustness_tests(files=["b.py"])

        assert generator._total_tests_generated == 8
        assert generator._total_bugs_found == 2

    @pytest.mark.asyncio
    async def test_coverage_growth_zero_when_no_tests(self, tmp_path: Path):
        """Coverage growth should be 0.0 when no tests have been generated."""
        llm = _make_llm_mock()
        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )

        growth = await generator.get_coverage_growth()

        assert growth == 0.0

    @pytest.mark.asyncio
    async def test_coverage_growth_ratio(self, tmp_path: Path):
        """Coverage growth = bugs_found / tests_generated."""
        llm = _make_llm_mock(response_text="def test_x(): pass")

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )

        # Manually set lifetime stats to test the ratio
        generator._total_tests_generated = 20
        generator._total_bugs_found = 5

        growth = await generator.get_coverage_growth()

        assert growth == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_run_tests_syntax_error(self, tmp_path: Path):
        """Syntax errors in generated test code should be handled gracefully."""
        llm = _make_llm_mock(response_text="def test_bad(:\n    pass")

        generator = RobustnessTestGenerator(
            llm=llm,
            codebase_root=tmp_path,
        )

        # Call the real _run_tests with broken syntax
        result = await generator._run_tests("def test_bad(:\n    pass")

        assert result.tests_generated == 0
        assert len(result.bug_descriptions) == 1
        assert "syntax" in result.bug_descriptions[0].lower()


# ── TestCoevolutionCycle ────────────────────────────────────────────────────


class TestCoevolutionCycle:
    @pytest.mark.asyncio
    async def test_full_cycle_no_adversarial(self):
        """Full cycle without adversarial generator: only history mining + GRPO batch."""
        rollback_rows = [
            {
                "proposal_id": "prop-600",
                "category": "modify_config",
                "description": "Changed parser",
                "rollback_reason": "Broke imports",
            },
        ]
        neo4j = _make_neo4j_mock(rollback_rows=rollback_rows, verification_rows=[])

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)
        result = await miner.run_cycle(adversarial_generator=None, files=None)

        assert isinstance(result, CoevolutionCycleResult)
        assert result.hard_negatives_mined == 1
        assert result.adversarial_tests_generated == 0
        assert result.tests_found_bugs == 0
        assert result.grpo_examples_produced == 1
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_full_cycle_with_adversarial(self):
        """Full cycle with adversarial generator should include adversarial negatives."""
        rollback_rows = [
            {
                "proposal_id": "prop-700",
                "category": "cat",
                "description": "desc",
                "rollback_reason": "reason",
            },
        ]
        # mine_from_history is called twice: once in run_cycle, once inside mine_from_robustness
        # Each call invokes _mine_rollbacks then _mine_verification_failures (2 neo4j queries each)
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(
            side_effect=[
                # First call to mine_from_history (in run_cycle)
                rollback_rows,  # _mine_rollbacks
                [],             # _mine_verification_failures
                # Second call to mine_from_history (inside mine_from_robustness)
                rollback_rows,  # _mine_rollbacks
                [],             # _mine_verification_failures
            ],
        )

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)

        mock_generator = AsyncMock(spec=RobustnessTestGenerator)
        mock_generator.generate_robustness_tests = AsyncMock(
            return_value=RobustnessTestResult(
                tests_generated=3,
                tests_executed=3,
                tests_found_bugs=2,
                bug_descriptions=["FAILED test_a", "FAILED test_b"],
            ),
        )

        result = await miner.run_cycle(
            adversarial_generator=mock_generator,
            files=["target.py"],
        )

        assert isinstance(result, CoevolutionCycleResult)
        assert result.hard_negatives_mined == 1  # from history
        assert result.adversarial_tests_generated == 2  # from adversarial (len of adv_negs)
        assert result.tests_found_bugs == 2
        # Total GRPO examples = history negatives + adversarial negatives
        assert result.grpo_examples_produced == 3
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_full_cycle_no_neo4j_no_adversarial(self):
        """Cycle with no neo4j and no adversarial generator: zero results."""
        miner = FailureAnalyzer(neo4j=None, llm=None)
        result = await miner.run_cycle()

        assert isinstance(result, CoevolutionCycleResult)
        assert result.hard_negatives_mined == 0
        assert result.adversarial_tests_generated == 0
        assert result.tests_found_bugs == 0
        assert result.grpo_examples_produced == 0
        assert result.coverage_growth_percent == 0.0

    @pytest.mark.asyncio
    async def test_coverage_growth_is_zero_without_adversarial(self):
        """Without adversarial generator, coverage_growth_percent should remain 0.0."""
        neo4j = _make_neo4j_mock(
            rollback_rows=[
                {
                    "proposal_id": "p",
                    "category": "c",
                    "description": "d",
                    "rollback_reason": "r",
                },
            ],
            verification_rows=[],
        )

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)
        result = await miner.run_cycle(adversarial_generator=None)

        assert result.coverage_growth_percent == 0.0

    @pytest.mark.asyncio
    async def test_grpo_batch_has_zero_reward_for_all_entries(self):
        """Every entry in a GRPO batch from hard negatives should have reward=0.0."""
        rollback_rows = [
            {
                "proposal_id": f"prop-{i}",
                "category": f"cat-{i}",
                "description": f"desc-{i}",
                "rollback_reason": f"reason-{i}",
            }
            for i in range(5)
        ]
        neo4j = _make_neo4j_mock(rollback_rows=rollback_rows, verification_rows=[])

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)
        negatives = await miner.mine_from_history()
        batch = await miner.prepare_grpo_batch(negatives)

        assert len(batch) == 5
        assert all(entry["reward"] == 0.0 for entry in batch)

    @pytest.mark.asyncio
    async def test_cycle_result_fields_are_consistent(self):
        """The GRPO examples produced should equal history + adversarial negatives."""
        neo4j = _make_neo4j_mock(
            rollback_rows=[
                {
                    "proposal_id": "p1",
                    "category": "c",
                    "description": "d",
                    "rollback_reason": "r",
                },
                {
                    "proposal_id": "p2",
                    "category": "c",
                    "description": "d",
                    "rollback_reason": "r",
                },
            ],
            verification_rows=[],
        )

        miner = FailureAnalyzer(neo4j=neo4j, llm=None)
        result = await miner.run_cycle()

        assert result.grpo_examples_produced == result.hard_negatives_mined
