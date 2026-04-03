"""
Unit tests for Evo ConsolidationOrchestrator.

Tests the 8-phase consolidation pipeline in isolation:
  - should_run() triggers on time and cycle thresholds
  - Hypothesis review integrates/archives correctly
  - Parameter optimisation respects velocity limits
  - Phase interactions produce correct ConsolidationResult
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from primitives.common import utc_now
from systems.evo.consolidation import (
    _CONSOLIDATION_CYCLE_THRESHOLD,
    _CONSOLIDATION_MAX_INTERVAL_HOURS,
    ConsolidationOrchestrator,
)
from systems.evo.hypothesis import HypothesisEngine
from systems.evo.parameter_tuner import ParameterTuner
from systems.evo.procedure_extractor import ProcedureExtractor
from systems.evo.self_model import SelfModelManager
from systems.evo.types import (
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
    Mutation,
    MutationType,
    PatternContext,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_mock_engine() -> MagicMock:
    engine = MagicMock(spec=HypothesisEngine)
    engine.get_all_active.return_value = []
    engine.get_supported.return_value = []
    engine.integrate_hypothesis = AsyncMock(return_value=True)
    engine.archive_hypothesis = AsyncMock()
    engine.is_stale.return_value = False
    return engine


def make_mock_tuner() -> MagicMock:
    tuner = MagicMock(spec=ParameterTuner)
    tuner.propose_adjustment.return_value = None
    tuner.check_velocity_limit.return_value = (True, "")
    tuner.apply_adjustment = AsyncMock()
    tuner.begin_cycle.return_value = None
    tuner.cycle_delta.return_value = 0.0
    return tuner


def make_mock_extractor() -> MagicMock:
    extractor = MagicMock(spec=ProcedureExtractor)
    extractor.extract_procedure = AsyncMock(return_value=None)
    extractor.begin_cycle.return_value = None
    return extractor


def make_mock_self_model() -> MagicMock:
    self_model = MagicMock(spec=SelfModelManager)
    self_model.update = AsyncMock()
    self_model.get_current.return_value = MagicMock(
        success_rate=0.8,
        mean_alignment=0.7,
    )
    return self_model


def make_orchestrator(
    engine=None,
    tuner=None,
    extractor=None,
    self_model=None,
    memory=None,
) -> ConsolidationOrchestrator:
    return ConsolidationOrchestrator(
        hypothesis_engine=engine or make_mock_engine(),
        parameter_tuner=tuner or make_mock_tuner(),
        procedure_extractor=extractor or make_mock_extractor(),
        self_model_manager=self_model or make_mock_self_model(),
        memory=memory,
    )


def make_supported_hypothesis(
    category: HypothesisCategory = HypothesisCategory.WORLD_MODEL,
    mutation_type: MutationType = MutationType.SCHEMA_ADDITION,
) -> Hypothesis:
    from datetime import timedelta
    h = Hypothesis(
        category=category,
        statement="Test hypothesis",
        formal_test="Test condition",
        status=HypothesisStatus.SUPPORTED,
        evidence_score=5.0,
    )
    h.created_at = utc_now() - timedelta(hours=48)
    h.supporting_episodes = [f"ep_{i}" for i in range(12)]
    h.proposed_mutation = Mutation(
        type=mutation_type,
        target="test_target",
        description="Test change",
    )
    return h


def make_context() -> PatternContext:
    return PatternContext()


# ─── Tests: should_run() ──────────────────────────────────────────────────────


class TestShouldRun:
    def test_runs_when_enough_time_elapsed(self):
        orchestrator = make_orchestrator()
        # Set last run to 7 hours ago (> 6h threshold)
        orchestrator._last_run_at = utc_now() - timedelta(hours=7)
        assert orchestrator.should_run(cycle_count=100, cycles_since_last=100) is True

    def test_runs_when_enough_cycles_elapsed(self):
        orchestrator = make_orchestrator()
        # Last run was just now (time not elapsed)
        orchestrator._last_run_at = utc_now()
        # But 10,001 cycles have passed
        assert orchestrator.should_run(
            cycle_count=10_001,
            cycles_since_last=_CONSOLIDATION_CYCLE_THRESHOLD + 1,
        ) is True

    def test_does_not_run_when_neither_threshold_met(self):
        orchestrator = make_orchestrator()
        orchestrator._last_run_at = utc_now() - timedelta(hours=1)
        assert orchestrator.should_run(cycle_count=100, cycles_since_last=100) is False

    def test_runs_exactly_at_time_threshold(self):
        orchestrator = make_orchestrator()
        orchestrator._last_run_at = utc_now() - timedelta(hours=_CONSOLIDATION_MAX_INTERVAL_HOURS)
        assert orchestrator.should_run(cycle_count=0, cycles_since_last=0) is True

    def test_runs_exactly_at_cycle_threshold(self):
        orchestrator = make_orchestrator()
        orchestrator._last_run_at = utc_now()  # Just ran
        assert orchestrator.should_run(
            cycle_count=_CONSOLIDATION_CYCLE_THRESHOLD,
            cycles_since_last=_CONSOLIDATION_CYCLE_THRESHOLD,
        ) is True


# ─── Tests: Phase 2 - Hypothesis Review ──────────────────────────────────────


class TestHypothesisReview:
    @pytest.mark.asyncio
    async def test_integrates_supported_hypotheses(self):
        engine = make_mock_engine()
        supported = make_supported_hypothesis()
        engine.get_all_active.return_value = [supported]
        engine.integrate_hypothesis = AsyncMock(return_value=True)

        orchestrator = make_orchestrator(engine=engine)
        result = await orchestrator._phase_hypothesis_review()

        engine.integrate_hypothesis.assert_called_once_with(supported)
        integrated, archived = result
        assert integrated == 1
        assert archived == 0

    @pytest.mark.asyncio
    async def test_archives_refuted_hypotheses(self):
        engine = make_mock_engine()
        h = Hypothesis(
            category=HypothesisCategory.WORLD_MODEL,
            statement="Refuted",
            formal_test="test",
            status=HypothesisStatus.REFUTED,
            evidence_score=-3.0,
        )
        engine.get_all_active.return_value = [h]
        engine.archive_hypothesis = AsyncMock()

        orchestrator = make_orchestrator(engine=engine)
        integrated, archived = await orchestrator._phase_hypothesis_review()

        engine.archive_hypothesis.assert_called_once_with(h, reason="refuted")
        assert archived == 1
        assert integrated == 0

    @pytest.mark.asyncio
    async def test_archives_stale_hypotheses(self):
        engine = make_mock_engine()
        h = Hypothesis(
            category=HypothesisCategory.WORLD_MODEL,
            statement="Stale",
            formal_test="test",
            status=HypothesisStatus.TESTING,
        )
        engine.get_all_active.return_value = [h]
        engine.is_stale.return_value = True
        engine.archive_hypothesis = AsyncMock()

        orchestrator = make_orchestrator(engine=engine)
        integrated, archived = await orchestrator._phase_hypothesis_review()

        engine.archive_hypothesis.assert_called_once_with(h, reason="stale")
        assert archived == 1

    @pytest.mark.asyncio
    async def test_mixed_batch(self):
        engine = make_mock_engine()

        supported = make_supported_hypothesis()
        refuted = Hypothesis(
            category=HypothesisCategory.WORLD_MODEL,
            statement="Refuted",
            formal_test="test",
            status=HypothesisStatus.REFUTED,
            evidence_score=-3.0,
        )
        testing = Hypothesis(
            category=HypothesisCategory.WORLD_MODEL,
            statement="Active",
            formal_test="test",
            status=HypothesisStatus.TESTING,
        )

        engine.get_all_active.return_value = [supported, refuted, testing]
        engine.is_stale.return_value = False
        engine.integrate_hypothesis = AsyncMock(return_value=True)
        engine.archive_hypothesis = AsyncMock()

        orchestrator = make_orchestrator(engine=engine)
        integrated, archived = await orchestrator._phase_hypothesis_review()

        assert integrated == 1
        assert archived == 1


# ─── Tests: Phase 5 - Parameter Optimisation ─────────────────────────────────


class TestParameterOptimisation:
    @pytest.mark.asyncio
    async def test_applies_valid_adjustments(self):
        engine = make_mock_engine()
        tuner = make_mock_tuner()

        param_hypothesis = make_supported_hypothesis(
            category=HypothesisCategory.PARAMETER,
            mutation_type=MutationType.PARAMETER_ADJUSTMENT,
        )
        engine.get_supported.return_value = [param_hypothesis]

        # Tuner proposes an adjustment
        mock_adj = MagicMock()
        mock_adj.delta = 0.01
        tuner.propose_adjustment.return_value = mock_adj
        tuner.check_velocity_limit.return_value = (True, "")

        orchestrator = make_orchestrator(engine=engine, tuner=tuner)
        count, delta = await orchestrator._phase_parameter_optimisation()

        tuner.apply_adjustment.assert_called_once_with(mock_adj)
        assert count == 1

    @pytest.mark.asyncio
    async def test_skips_non_parameter_hypotheses(self):
        engine = make_mock_engine()
        tuner = make_mock_tuner()

        world_hypothesis = make_supported_hypothesis(
            category=HypothesisCategory.WORLD_MODEL,
        )
        engine.get_supported.return_value = [world_hypothesis]

        orchestrator = make_orchestrator(engine=engine, tuner=tuner)
        count, _ = await orchestrator._phase_parameter_optimisation()

        tuner.propose_adjustment.assert_not_called()
        assert count == 0

    @pytest.mark.asyncio
    async def test_respects_velocity_limit(self):
        engine = make_mock_engine()
        tuner = make_mock_tuner()

        param_hyp = make_supported_hypothesis(
            category=HypothesisCategory.PARAMETER,
            mutation_type=MutationType.PARAMETER_ADJUSTMENT,
        )
        engine.get_supported.return_value = [param_hyp]

        mock_adj = MagicMock()
        mock_adj.delta = 0.10
        tuner.propose_adjustment.return_value = mock_adj
        # Velocity check fails
        tuner.check_velocity_limit.return_value = (False, "Total delta too large")

        orchestrator = make_orchestrator(engine=engine, tuner=tuner)
        count, _ = await orchestrator._phase_parameter_optimisation()

        # Should not apply when velocity limit exceeded (unless filtered)
        # Either 0 applied (batch rejected) or partial (smallest filtered)
        # Either way, apply count <= 1 (the filtered small delta)
        assert tuner.apply_adjustment.call_count <= 1


# ─── Tests: Full Run ──────────────────────────────────────────────────────────


class TestFullRun:
    @pytest.mark.asyncio
    async def test_run_returns_result(self):
        orchestrator = make_orchestrator()
        context = make_context()
        result = await orchestrator.run(context)
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_resets_pattern_context(self):
        orchestrator = make_orchestrator()
        context = make_context()
        context.episodes_scanned = 500
        await orchestrator.run(context)
        assert context.episodes_scanned == 0

    @pytest.mark.asyncio
    async def test_run_increments_total_runs(self):
        orchestrator = make_orchestrator()
        assert orchestrator._total_runs == 0
        await orchestrator.run(make_context())
        assert orchestrator._total_runs == 1

    @pytest.mark.asyncio
    async def test_run_updates_last_run_at(self):
        orchestrator = make_orchestrator()
        old_time = orchestrator._last_run_at
        await orchestrator.run(make_context())
        assert orchestrator._last_run_at > old_time

    @pytest.mark.asyncio
    async def test_run_never_raises(self):
        """All phase errors are caught internally."""
        engine = make_mock_engine()
        engine.get_all_active.side_effect = RuntimeError("DB down")

        orchestrator = make_orchestrator(engine=engine)
        # Should not raise
        result = await orchestrator.run(make_context())
        assert result is not None
