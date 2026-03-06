"""Tests for Soma type definitions, enums, and data models."""

from __future__ import annotations

import pytest

from systems.soma.types import (
    ALL_DIMENSIONS,
    DEFAULT_SETPOINTS,
    DIMENSION_RANGES,
    HORIZONS,
    NUM_DIMENSIONS,
    SEED_ATTRACTORS,
    STAGE_HORIZONS,
    AllostaticSignal,
    Attractor,
    Bifurcation,
    DevelopmentalStage,
    InteroceptiveDimension,
    InteroceptiveState,
    SomaticMarker,
    _clamp_dimension,
    stage_at_least,
)


class TestEnums:
    def test_interoceptive_dimension_has_9_values(self):
        assert len(InteroceptiveDimension) == 9

    def test_interoceptive_dimension_string_values(self):
        assert InteroceptiveDimension.ENERGY.value == "energy"
        assert InteroceptiveDimension.VALENCE.value == "valence"
        assert InteroceptiveDimension.TEMPORAL_PRESSURE.value == "temporal_pressure"

    def test_developmental_stage_has_5_values(self):
        assert len(DevelopmentalStage) == 5

    def test_developmental_stage_ordering(self):
        assert stage_at_least(DevelopmentalStage.GENERATIVE, DevelopmentalStage.REFLEXIVE)
        assert stage_at_least(DevelopmentalStage.REFLECTIVE, DevelopmentalStage.DELIBERATIVE)
        assert not stage_at_least(DevelopmentalStage.REFLEXIVE, DevelopmentalStage.ASSOCIATIVE)

    def test_all_dimensions_matches_enum(self):
        assert NUM_DIMENSIONS == 9
        assert len(ALL_DIMENSIONS) == 9


class TestConstants:
    def test_horizons_has_6_entries(self):
        assert len(HORIZONS) == 6
        assert "immediate" in HORIZONS
        assert "narrative" in HORIZONS

    def test_horizon_values_increasing(self):
        values = list(HORIZONS.values())
        for i in range(1, len(values)):
            assert values[i] > values[i - 1]

    def test_default_setpoints_has_all_dimensions(self):
        for dim in ALL_DIMENSIONS:
            assert dim in DEFAULT_SETPOINTS

    def test_dimension_ranges_covers_all_dimensions(self):
        for dim in ALL_DIMENSIONS:
            assert dim in DIMENSION_RANGES
            lo, hi = DIMENSION_RANGES[dim]
            assert lo < hi

    def test_seed_attractors_non_empty(self):
        assert len(SEED_ATTRACTORS) >= 6

    def test_stage_horizons_progressive(self):
        for i, stage in enumerate(DevelopmentalStage):
            horizons = STAGE_HORIZONS[stage]
            assert len(horizons) >= 2
            if i > 0:
                prev = list(DevelopmentalStage)[i - 1]
                assert len(horizons) >= len(STAGE_HORIZONS[prev])


class TestInteroceptiveState:
    def test_construction_defaults(self):
        state = InteroceptiveState()
        assert state.urgency == 0.0
        assert state.dominant_error == InteroceptiveDimension.ENERGY

    def test_to_marker_vector_returns_19d(self):
        state = InteroceptiveState(
            sensed={d: 0.5 for d in ALL_DIMENSIONS},
            errors={"moment": {d: 0.1 for d in ALL_DIMENSIONS}},
            max_error_magnitude=0.3,
        )
        vec = state.to_marker_vector()
        assert len(vec) == 19
        # 9 sensed values of 0.5
        assert all(v == 0.5 for v in vec[:9])
        # 9 error values of 0.1
        assert all(v == 0.1 for v in vec[9:18])
        # 1 PE magnitude
        assert vec[18] == 0.3

    def test_to_marker_vector_missing_errors(self):
        state = InteroceptiveState(sensed={d: 0.5 for d in ALL_DIMENSIONS})
        vec = state.to_marker_vector()
        assert len(vec) == 19
        # Errors default to 0.0
        assert all(v == 0.0 for v in vec[9:18])


class TestAllostaticSignal:
    def test_default_signal(self):
        signal = AllostaticSignal.default()
        assert signal.urgency == 0.0
        assert len(signal.precision_weights) == 9
        # All precision weights should be 1.0
        assert all(w == 1.0 for w in signal.precision_weights.values())
        # State should be at setpoints
        for dim in ALL_DIMENSIONS:
            assert signal.state.sensed[dim] == DEFAULT_SETPOINTS[dim]


class TestSomaticMarker:
    def test_to_vector_returns_19d(self):
        marker = SomaticMarker(
            interoceptive_snapshot={d: 0.5 for d in ALL_DIMENSIONS},
            allostatic_error_snapshot={d: 0.1 for d in ALL_DIMENSIONS},
            prediction_error_at_encoding=0.2,
            allostatic_context="flow",
        )
        vec = marker.to_vector()
        assert len(vec) == 19
        assert vec[18] == 0.2

    def test_empty_marker_vector(self):
        marker = SomaticMarker()
        vec = marker.to_vector()
        assert len(vec) == 19
        assert all(v == 0.0 for v in vec)


class TestAttractor:
    def test_distance_to(self):
        attractor = Attractor(
            label="test",
            center={d: 0.5 for d in ALL_DIMENSIONS},
        )
        state = {d: 0.5 for d in ALL_DIMENSIONS}
        assert attractor.distance_to(state) == 0.0

    def test_distance_to_offset(self):
        attractor = Attractor(
            label="test",
            center={d: 0.0 for d in ALL_DIMENSIONS},
        )
        state = {d: 1.0 for d in ALL_DIMENSIONS}
        dist = attractor.distance_to(state)
        assert dist == pytest.approx(3.0, abs=0.01)  # sqrt(9 * 1^2) = 3.0


class TestBifurcation:
    def test_distance_no_weights(self):
        bif = Bifurcation(
            label="test",
            dimensions=[InteroceptiveDimension.ENERGY],
            boundary_condition="energy < 0.25",
            pre_regime="flow",
            post_regime="anxiety",
        )
        dist = bif.distance({InteroceptiveDimension.ENERGY: 0.5})
        assert dist == float("inf")

    def test_distance_with_weights(self):
        bif = Bifurcation(
            label="test",
            dimensions=[InteroceptiveDimension.ENERGY],
            boundary_condition="energy < 0.25",
            pre_regime="flow",
            post_regime="anxiety",
            weights={InteroceptiveDimension.ENERGY: 1.0},
            bias=-0.25,
        )
        # Energy at 0.5: 1.0 * 0.5 - 0.25 = 0.25 (safe side)
        dist = bif.distance({InteroceptiveDimension.ENERGY: 0.5})
        assert dist == pytest.approx(0.25)

    def test_time_to_crossing(self):
        bif = Bifurcation(
            label="test",
            dimensions=[InteroceptiveDimension.ENERGY],
            boundary_condition="energy < 0.25",
            pre_regime="flow",
            post_regime="anxiety",
            weights={InteroceptiveDimension.ENERGY: 1.0},
            bias=-0.25,
        )
        state = {InteroceptiveDimension.ENERGY: 0.5}
        velocity = {InteroceptiveDimension.ENERGY: -0.1}  # Depleting
        ttc = bif.time_to_crossing(state, velocity)
        assert ttc is not None
        assert ttc == pytest.approx(2.5)  # 0.25 / 0.1


class TestClampDimension:
    def test_clamp_within_range(self):
        assert _clamp_dimension(InteroceptiveDimension.ENERGY, 0.5) == 0.5

    def test_clamp_below_range(self):
        assert _clamp_dimension(InteroceptiveDimension.ENERGY, -0.5) == 0.0

    def test_clamp_above_range(self):
        assert _clamp_dimension(InteroceptiveDimension.ENERGY, 1.5) == 1.0

    def test_valence_allows_negative(self):
        assert _clamp_dimension(InteroceptiveDimension.VALENCE, -0.5) == -0.5
        assert _clamp_dimension(InteroceptiveDimension.VALENCE, -1.5) == -1.0
