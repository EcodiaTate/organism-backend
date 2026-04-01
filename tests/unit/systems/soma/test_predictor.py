"""Tests for Soma Interoceptive Predictor - multi-horizon generative model."""

from __future__ import annotations

import pytest

from systems.soma.predictor import InteroceptivePredictor
from systems.soma.types import (
    ALL_DIMENSIONS,
    DEFAULT_SETPOINTS,
    HORIZONS,
    InteroceptiveDimension,
)


def _make_predictor(**kwargs) -> InteroceptivePredictor:
    return InteroceptivePredictor(
        buffer_size=kwargs.get("buffer_size", 100),
        ewm_span=kwargs.get("ewm_span", 5),
    )


def _flat_state(value: float = 0.5) -> dict[InteroceptiveDimension, float]:
    return {d: value for d in ALL_DIMENSIONS}


class TestTrajectoryBuffer:
    def test_push_state(self):
        p = _make_predictor()
        p.push_state(_flat_state(0.5))
        assert p.trajectory_length == 1

    def test_buffer_maxlen(self):
        p = _make_predictor(buffer_size=10)
        for i in range(20):
            p.push_state(_flat_state(i / 20))
        assert p.trajectory_length == 10


class TestPrediction:
    def test_predict_returns_all_dimensions(self):
        p = _make_predictor()
        for _ in range(5):
            p.push_state(_flat_state(0.5))
        pred = p.predict("moment", _flat_state(0.5), list(HORIZONS.keys()))
        assert len(pred) == 9
        for dim in ALL_DIMENSIONS:
            assert dim in pred

    def test_predict_stable_state_stays_near(self):
        """If trajectory is stable (no change), predictions stay near current."""
        p = _make_predictor()
        for _ in range(20):
            p.push_state(_flat_state(0.5))
        pred = p.predict("moment", _flat_state(0.5), list(HORIZONS.keys()))
        for dim in ALL_DIMENSIONS:
            assert abs(pred[dim] - 0.5) < 0.05

    def test_predict_rising_trajectory(self):
        """If trajectory is rising, prediction should be above current."""
        p = _make_predictor()
        for i in range(20):
            p.push_state(_flat_state(0.3 + i * 0.01))
        current = _flat_state(0.49)
        pred = p.predict("moment", current, list(HORIZONS.keys()))
        # Should predict above current (extrapolating the rise)
        for dim in [InteroceptiveDimension.ENERGY]:
            assert pred[dim] >= current[dim]

    def test_predict_unavailable_horizon(self):
        """If horizon is not in available list, returns current state."""
        p = _make_predictor()
        p.push_state(_flat_state(0.5))
        pred = p.predict("narrative", _flat_state(0.5), ["immediate", "moment"])
        for dim in ALL_DIMENSIONS:
            assert pred[dim] == 0.5

    def test_predict_all_horizons(self):
        p = _make_predictor()
        for _ in range(10):
            p.push_state(_flat_state(0.5))
        preds = p.predict_all_horizons(_flat_state(0.5), list(HORIZONS.keys()))
        assert len(preds) == 6
        for h in HORIZONS:
            assert h in preds

    def test_values_clamped(self):
        """Predictions should be clamped to valid ranges."""
        p = _make_predictor()
        for _i in range(20):
            p.push_state(_flat_state(0.99))
        pred = p.predict("narrative", _flat_state(0.99), list(HORIZONS.keys()))
        for dim in ALL_DIMENSIONS:
            if dim == InteroceptiveDimension.VALENCE:
                assert -1.0 <= pred[dim] <= 1.0
            else:
                assert 0.0 <= pred[dim] <= 1.0


class TestAllostaticErrors:
    def test_errors_at_setpoint(self):
        """When predicted equals setpoint, error is 0."""
        p = _make_predictor()
        predictions = {"moment": dict(DEFAULT_SETPOINTS)}
        errors = p.compute_allostatic_errors(predictions, dict(DEFAULT_SETPOINTS))
        for dim in ALL_DIMENSIONS:
            assert errors["moment"][dim] == pytest.approx(0.0)

    def test_positive_error_overshoot(self):
        predictions = {"moment": {d: 0.8 for d in ALL_DIMENSIONS}}
        setpoints = {d: 0.5 for d in ALL_DIMENSIONS}
        p = _make_predictor()
        errors = p.compute_allostatic_errors(predictions, setpoints)
        for dim in ALL_DIMENSIONS:
            assert errors["moment"][dim] == pytest.approx(0.3)

    def test_negative_error_undershoot(self):
        predictions = {"moment": {d: 0.2 for d in ALL_DIMENSIONS}}
        setpoints = {d: 0.5 for d in ALL_DIMENSIONS}
        p = _make_predictor()
        errors = p.compute_allostatic_errors(predictions, setpoints)
        for dim in ALL_DIMENSIONS:
            assert errors["moment"][dim] == pytest.approx(-0.3)


class TestErrorRates:
    def test_stable_errors_rate_near_zero(self):
        p = _make_predictor()
        for _ in range(5):
            p.compute_error_rates({d: 0.1 for d in ALL_DIMENSIONS})
        rates = p.compute_error_rates({d: 0.1 for d in ALL_DIMENSIONS})
        for dim in ALL_DIMENSIONS:
            assert abs(rates[dim]) < 0.01

    def test_empty_history(self):
        p = _make_predictor()
        rates = p.compute_error_rates({d: 0.1 for d in ALL_DIMENSIONS})
        for dim in ALL_DIMENSIONS:
            assert rates[dim] == 0.0


class TestTemporalDissonance:
    def test_no_dissonance_when_aligned(self):
        p = _make_predictor()
        preds = {
            "moment": {d: 0.5 for d in ALL_DIMENSIONS},
            "session": {d: 0.5 for d in ALL_DIMENSIONS},
        }
        dissonance = p.compute_temporal_dissonance(preds)
        for dim in ALL_DIMENSIONS:
            assert dissonance[dim] == 0.0

    def test_positive_dissonance_temptation(self):
        """Positive = feels good now, heading bad later."""
        p = _make_predictor()
        preds = {
            "moment": {d: 0.8 for d in ALL_DIMENSIONS},
            "session": {d: 0.3 for d in ALL_DIMENSIONS},
        }
        dissonance = p.compute_temporal_dissonance(preds)
        for dim in ALL_DIMENSIONS:
            assert dissonance[dim] == pytest.approx(0.5)


class TestDynamicsMatrix:
    def test_update_dynamics_valid(self):
        p = _make_predictor()
        n = len(ALL_DIMENSIONS)
        new_dynamics = [[0.0] * n for _ in range(n)]
        p.update_dynamics(new_dynamics)
        assert p._dynamics == new_dynamics

    def test_update_dynamics_wrong_size(self):
        p = _make_predictor()
        old = [row[:] for row in p._dynamics]
        p.update_dynamics([[1.0, 2.0], [3.0, 4.0]])
        assert p._dynamics == old  # Should not change
