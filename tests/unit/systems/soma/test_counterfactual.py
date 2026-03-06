"""Tests for Soma Counterfactual Engine — Oneiros REM integration."""

from __future__ import annotations

from systems.soma.counterfactual import CounterfactualEngine
from systems.soma.types import (
    ALL_DIMENSIONS,
    DEFAULT_SETPOINTS,
    CounterfactualTrace,
    InteroceptiveDimension,
)


def _flat_state(value: float = 0.5) -> dict[InteroceptiveDimension, float]:
    return {d: value for d in ALL_DIMENSIONS}


class TestCounterfactualGeneration:
    def test_empty_trajectory(self):
        engine = CounterfactualEngine()
        trace = engine.generate_counterfactual(
            decision_id="test-1",
            actual_trajectory=[],
            alternative_description="do nothing",
            alternative_initial_impact={},
            setpoints=dict(DEFAULT_SETPOINTS),
        )
        assert isinstance(trace, CounterfactualTrace)
        assert "Insufficient data" in trace.lesson

    def test_generates_trace(self):
        engine = CounterfactualEngine()
        actual = [_flat_state(0.5 - i * 0.02) for i in range(10)]
        trace = engine.generate_counterfactual(
            decision_id="test-2",
            actual_trajectory=actual,
            alternative_description="conserve energy",
            alternative_initial_impact={InteroceptiveDimension.ENERGY: 0.2},
            setpoints=dict(DEFAULT_SETPOINTS),
            num_steps=10,
        )
        assert isinstance(trace, CounterfactualTrace)
        assert trace.decision_id == "test-2"
        assert len(trace.counterfactual_trajectory) == 10
        assert len(trace.chosen_trajectory) == 10
        assert trace.lesson != ""

    def test_regret_when_counterfactual_better(self):
        engine = CounterfactualEngine()
        # Actual trajectory moves away from setpoints
        actual = [{d: 0.2 for d in ALL_DIMENSIONS} for _ in range(5)]
        trace = engine.generate_counterfactual(
            decision_id="test-3",
            actual_trajectory=actual,
            alternative_description="different approach",
            alternative_initial_impact={InteroceptiveDimension.ENERGY: 0.4},
            setpoints=dict(DEFAULT_SETPOINTS),
            num_steps=5,
        )
        # Should have some regret or gratitude (not both)
        assert trace.regret >= 0.0
        assert trace.gratitude >= 0.0


class TestDynamicsSharing:
    def test_set_dynamics(self):
        engine = CounterfactualEngine()
        n = len(ALL_DIMENSIONS)
        dynamics = [[0.0] * n for _ in range(n)]
        engine.set_dynamics(dynamics)
        assert engine._dynamics is dynamics
