"""Tests for Soma Allostatic Controller — setpoints, urgency, signal construction."""

from __future__ import annotations

import pytest

from systems.soma.allostatic_controller import AllostaticController
from systems.soma.types import (
    ALL_DIMENSIONS,
    DEFAULT_SETPOINTS,
    AllostaticSignal,
    InteroceptiveDimension,
    InteroceptiveState,
)


def _make_controller(**kwargs) -> AllostaticController:
    return AllostaticController(
        adaptation_alpha=kwargs.get("alpha", 0.05),
        urgency_threshold=kwargs.get("threshold", 0.3),
    )


class TestSetpoints:
    def test_default_setpoints_loaded(self):
        c = _make_controller()
        sp = c.setpoints
        for dim in ALL_DIMENSIONS:
            assert dim in sp
            assert sp[dim] == DEFAULT_SETPOINTS[dim]

    def test_context_switch_updates_targets(self):
        c = _make_controller()
        c.set_context("conversation")
        # Tick many times to converge
        for _ in range(200):
            c.tick_setpoints()
        sp = c.setpoints
        assert sp[InteroceptiveDimension.SOCIAL_CHARGE] == pytest.approx(0.50, abs=0.01)

    def test_ema_smoothing(self):
        c = _make_controller(alpha=0.5)
        c.set_context("conversation")
        c.tick_setpoints()
        sp = c.setpoints
        # After 1 tick with alpha=0.5, should be halfway between old and new
        expected = DEFAULT_SETPOINTS[InteroceptiveDimension.SOCIAL_CHARGE] + 0.5 * (0.50 - DEFAULT_SETPOINTS[InteroceptiveDimension.SOCIAL_CHARGE])
        assert sp[InteroceptiveDimension.SOCIAL_CHARGE] == pytest.approx(expected, abs=0.01)


class TestUrgency:
    def test_zero_errors_zero_urgency(self):
        c = _make_controller()
        errors = {"moment": {d: 0.0 for d in ALL_DIMENSIONS}}
        rates = {d: 0.0 for d in ALL_DIMENSIONS}
        assert c.compute_urgency(errors, rates) == 0.0

    def test_high_error_produces_urgency(self):
        c = _make_controller()
        errors = {"moment": {d: 0.0 for d in ALL_DIMENSIONS}}
        errors["moment"][InteroceptiveDimension.ENERGY] = -0.5
        rates = {d: 0.0 for d in ALL_DIMENSIONS}
        rates[InteroceptiveDimension.ENERGY] = -0.3
        urgency = c.compute_urgency(errors, rates)
        assert urgency > 0.1

    def test_urgency_clamped_0_to_1(self):
        c = _make_controller()
        errors = {"moment": {d: 1.0 for d in ALL_DIMENSIONS}}
        rates = {d: 1.0 for d in ALL_DIMENSIONS}
        urgency = c.compute_urgency(errors, rates)
        assert 0.0 <= urgency <= 1.0


class TestDominantError:
    def test_finds_largest_error(self):
        c = _make_controller()
        errors = {"moment": {d: 0.0 for d in ALL_DIMENSIONS}}
        errors["moment"][InteroceptiveDimension.COHERENCE] = -0.8
        dim, mag = c.find_dominant_error(errors)
        assert dim == InteroceptiveDimension.COHERENCE
        assert mag == pytest.approx(0.8)

    def test_empty_errors(self):
        c = _make_controller()
        dim, mag = c.find_dominant_error({})
        assert dim == InteroceptiveDimension.ENERGY
        assert mag == 0.0


class TestPrecisionWeights:
    def test_weights_normalize_to_mean_1(self):
        c = _make_controller()
        state = InteroceptiveState(
            errors={"moment": {d: 0.1 for d in ALL_DIMENSIONS}},
            error_rates={d: 0.0 for d in ALL_DIMENSIONS},
        )
        weights = c.compute_precision_weights(state)
        assert len(weights) == 9
        mean = sum(weights.values()) / len(weights)
        assert mean == pytest.approx(1.0, abs=0.01)

    def test_higher_error_higher_precision(self):
        c = _make_controller()
        errors = {"moment": {d: 0.0 for d in ALL_DIMENSIONS}}
        errors["moment"][InteroceptiveDimension.INTEGRITY] = 0.8
        state = InteroceptiveState(
            errors=errors,
            error_rates={d: 0.0 for d in ALL_DIMENSIONS},
        )
        weights = c.compute_precision_weights(state)
        assert weights[InteroceptiveDimension.INTEGRITY] > weights[InteroceptiveDimension.ENERGY]


class TestBuildSignal:
    def test_builds_valid_signal(self):
        c = _make_controller()
        state = InteroceptiveState(
            sensed={d: 0.5 for d in ALL_DIMENSIONS},
            setpoints=dict(DEFAULT_SETPOINTS),
            errors={"moment": {d: 0.1 for d in ALL_DIMENSIONS}},
            error_rates={d: 0.0 for d in ALL_DIMENSIONS},
            temporal_dissonance={d: 0.0 for d in ALL_DIMENSIONS},
            urgency=0.1,
        )
        phase = {"nearest_attractor": "flow", "trajectory_heading": "toward_attractor", "distance_to_nearest_bifurcation": 0.5}
        signal = c.build_signal(state, phase, cycle_number=42)
        assert isinstance(signal, AllostaticSignal)
        assert signal.cycle_number == 42
        assert signal.nearest_attractor == "flow"
        assert signal.trajectory_heading == "toward_attractor"
        assert len(signal.precision_weights) == 9


class TestEnergyBurnRate:
    def test_stable_energy_no_burn(self):
        c = _make_controller()
        for _ in range(5):
            rate, exhaustion = c.track_energy_burn(0.5)
        assert abs(rate) < 0.01
        assert exhaustion is None

    def test_depleting_energy_predicts_exhaustion(self):
        c = _make_controller()
        for i in range(10):
            rate, exhaustion = c.track_energy_burn(0.5 - i * 0.03)
        assert rate < 0  # Burning energy
