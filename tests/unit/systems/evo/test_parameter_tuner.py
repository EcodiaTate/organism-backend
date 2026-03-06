"""
Unit tests for Evo ParameterTuner.

Tests parameter defaults, adjustment proposals, velocity limiting,
and application mechanics.
"""

from __future__ import annotations

import pytest

from systems.evo.parameter_tuner import ParameterTuner
from systems.evo.types import (
    PARAMETER_DEFAULTS,
    TUNABLE_PARAMETERS,
    VELOCITY_LIMITS,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
    Mutation,
    MutationType,
    ParameterAdjustment,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_tuner() -> ParameterTuner:
    return ParameterTuner(memory=None)


def make_parameter_hypothesis(
    param: str = "nova.efe.pragmatic",
    delta: float = 0.01,
    *,
    status: HypothesisStatus = HypothesisStatus.SUPPORTED,
    supporting_count: int = 12,
    evidence_score: float = 4.0,
) -> Hypothesis:
    h = Hypothesis(
        category=HypothesisCategory.PARAMETER,
        statement=f"Adjust {param}",
        formal_test="If adjusted, performance improves",
        status=status,
        evidence_score=evidence_score,
        proposed_mutation=Mutation(
            type=MutationType.PARAMETER_ADJUSTMENT,
            target=param,
            value=delta,
            description=f"Adjust {param} by {delta}",
        ),
    )
    h.supporting_episodes = [f"ep_{i}" for i in range(supporting_count)]
    return h


# ─── Tests: Initial State ─────────────────────────────────────────────────────


class TestParameterTunerInitialState:
    def test_all_defaults_loaded(self):
        tuner = make_tuner()
        for name, default in PARAMETER_DEFAULTS.items():
            assert tuner.get_current_parameter(name) == pytest.approx(default)

    def test_unknown_parameter_returns_none(self):
        tuner = make_tuner()
        assert tuner.get_current_parameter("not.real.param") is None

    def test_get_all_parameters_returns_all(self):
        tuner = make_tuner()
        params = tuner.get_all_parameters()
        assert len(params) == len(PARAMETER_DEFAULTS)
        for name in PARAMETER_DEFAULTS:
            assert name in params


# ─── Tests: propose_adjustment ────────────────────────────────────────────────


class TestProposeAdjustment:
    def test_returns_none_for_non_parameter_hypothesis(self):
        tuner = make_tuner()
        h = make_parameter_hypothesis()
        h.category = HypothesisCategory.WORLD_MODEL
        assert tuner.propose_adjustment(h) is None

    def test_returns_none_for_non_supported_hypothesis(self):
        tuner = make_tuner()
        h = make_parameter_hypothesis(status=HypothesisStatus.TESTING)
        assert tuner.propose_adjustment(h) is None

    def test_returns_none_for_no_mutation(self):
        tuner = make_tuner()
        h = make_parameter_hypothesis()
        h.proposed_mutation = None
        assert tuner.propose_adjustment(h) is None

    def test_returns_none_for_wrong_mutation_type(self):
        tuner = make_tuner()
        h = make_parameter_hypothesis()
        h.proposed_mutation = Mutation(
            type=MutationType.PROCEDURE_CREATION,
            target="some_procedure",
        )
        assert tuner.propose_adjustment(h) is None

    def test_returns_none_for_unknown_parameter(self):
        tuner = make_tuner()
        h = make_parameter_hypothesis(param="not.real.param")
        assert tuner.propose_adjustment(h) is None

    def test_adjustment_respects_max_step(self):
        """Delta should be clamped to spec.step (and velocity limit)."""
        tuner = make_tuner()
        spec = TUNABLE_PARAMETERS["nova.efe.pragmatic"]
        # Propose a very large delta — should be clamped to step
        h = make_parameter_hypothesis(param="nova.efe.pragmatic", delta=10.0)
        adj = tuner.propose_adjustment(h)
        assert adj is not None
        assert abs(adj.delta) <= spec.step + 0.0001

    def test_adjustment_clamped_to_max_range(self):
        """Value should not exceed max_val."""
        tuner = make_tuner()
        spec = TUNABLE_PARAMETERS["nova.efe.pragmatic"]
        # Push current value to just below max
        tuner._values["nova.efe.pragmatic"] = spec.max_val - 0.001
        h = make_parameter_hypothesis(param="nova.efe.pragmatic", delta=0.02)
        adj = tuner.propose_adjustment(h)
        if adj is not None:
            assert adj.new_value <= spec.max_val

    def test_adjustment_clamped_to_min_range(self):
        """Value should not go below min_val."""
        tuner = make_tuner()
        spec = TUNABLE_PARAMETERS["nova.efe.pragmatic"]
        tuner._values["nova.efe.pragmatic"] = spec.min_val + 0.001
        h = make_parameter_hypothesis(param="nova.efe.pragmatic", delta=-0.1)
        adj = tuner.propose_adjustment(h)
        if adj is not None:
            assert adj.new_value >= spec.min_val

    def test_returns_none_if_no_meaningful_change(self):
        """If clamped delta is < 0.0001, adjustment should be skipped."""
        tuner = make_tuner()
        spec = TUNABLE_PARAMETERS["nova.efe.pragmatic"]
        # Pin value at max — positive delta results in no change
        tuner._values["nova.efe.pragmatic"] = spec.max_val
        h = make_parameter_hypothesis(param="nova.efe.pragmatic", delta=0.01)
        adj = tuner.propose_adjustment(h)
        assert adj is None

    def test_valid_adjustment_has_correct_fields(self):
        tuner = make_tuner()
        h = make_parameter_hypothesis(
            param="nova.efe.pragmatic",
            delta=0.01,
            evidence_score=5.0,
            supporting_count=15,
        )
        adj = tuner.propose_adjustment(h)
        assert adj is not None
        assert adj.parameter == "nova.efe.pragmatic"
        assert adj.hypothesis_id == h.id
        assert adj.evidence_score == pytest.approx(5.0)
        assert adj.supporting_count == 15
        assert adj.old_value == pytest.approx(PARAMETER_DEFAULTS["nova.efe.pragmatic"])


# ─── Tests: check_velocity_limit ──────────────────────────────────────────────


class TestVelocityLimit:
    def _make_adj(self, param: str, delta: float) -> ParameterAdjustment:
        return ParameterAdjustment(
            parameter=param,
            old_value=0.3,
            new_value=0.3 + delta,
            delta=delta,
            hypothesis_id="h1",
            evidence_score=4.0,
            supporting_count=10,
        )

    def test_allows_small_batch(self):
        tuner = make_tuner()
        adjustments = [
            self._make_adj("nova.efe.pragmatic", 0.02),
            self._make_adj("nova.efe.epistemic", 0.01),
        ]
        allowed, reason = tuner.check_velocity_limit(adjustments)
        assert allowed is True
        assert reason == ""

    def test_blocks_when_total_delta_exceeds_limit(self):
        tuner = make_tuner()
        limit = VELOCITY_LIMITS["max_total_parameter_delta_per_cycle"]
        # Create batch that exceeds limit
        adjustments = [self._make_adj("nova.efe.pragmatic", limit / 2 + 0.01)]
        adjustments.append(self._make_adj("nova.efe.epistemic", limit / 2 + 0.01))
        allowed, reason = tuner.check_velocity_limit(adjustments)
        assert allowed is False
        assert "Total parameter delta" in reason

    def test_empty_batch_always_allowed(self):
        tuner = make_tuner()
        allowed, reason = tuner.check_velocity_limit([])
        assert allowed is True


# ─── Tests: apply_adjustment ──────────────────────────────────────────────────


class TestApplyAdjustment:
    @pytest.mark.asyncio
    async def test_updates_value(self):
        tuner = make_tuner()
        original = tuner.get_current_parameter("nova.efe.pragmatic")
        adj = ParameterAdjustment(
            parameter="nova.efe.pragmatic",
            old_value=original,
            new_value=original + 0.01,
            delta=0.01,
            hypothesis_id="test",
            evidence_score=4.0,
            supporting_count=10,
        )
        await tuner.apply_adjustment(adj)
        assert tuner.get_current_parameter("nova.efe.pragmatic") == pytest.approx(original + 0.01)

    @pytest.mark.asyncio
    async def test_increments_total_count(self):
        tuner = make_tuner()
        assert tuner.stats["total_adjustments"] == 0

        adj = ParameterAdjustment(
            parameter="nova.efe.pragmatic",
            old_value=0.35,
            new_value=0.37,
            delta=0.02,
            hypothesis_id="h1",
            evidence_score=4.0,
            supporting_count=10,
        )
        await tuner.apply_adjustment(adj)
        assert tuner.stats["total_adjustments"] == 1

    @pytest.mark.asyncio
    async def test_begin_cycle_resets_cycle_tracking(self):
        tuner = make_tuner()
        adj = ParameterAdjustment(
            parameter="nova.efe.pragmatic",
            old_value=0.35,
            new_value=0.37,
            delta=0.02,
            hypothesis_id="h1",
            evidence_score=4.0,
            supporting_count=10,
        )
        await tuner.apply_adjustment(adj)
        assert tuner.stats["cycle_adjustments"] == 1

        tuner.begin_cycle()
        assert tuner.stats["cycle_adjustments"] == 0
        assert tuner.stats["total_adjustments"] == 1  # Total preserved

    @pytest.mark.asyncio
    async def test_cycle_delta_accumulates(self):
        tuner = make_tuner()
        tuner.begin_cycle()

        for i, param in enumerate(["nova.efe.pragmatic", "nova.efe.epistemic"]):
            adj = ParameterAdjustment(
                parameter=param,
                old_value=0.2,
                new_value=0.22,
                delta=0.02,
                hypothesis_id=f"h{i}",
                evidence_score=4.0,
                supporting_count=10,
            )
            await tuner.apply_adjustment(adj)

        assert tuner.cycle_delta() == pytest.approx(0.04)
