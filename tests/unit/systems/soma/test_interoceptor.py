"""Tests for Soma Interoceptor - 9D sensing pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from systems.soma.interoceptor import FALLBACK_VALUES, Interoceptor
from systems.soma.types import ALL_DIMENSIONS, InteroceptiveDimension


def _make_interoceptor(**kwargs) -> Interoceptor:
    """Create an Interoceptor with optional mock system references."""
    i = Interoceptor()
    if "atune" in kwargs:
        i.set_atune(kwargs["atune"])
    if "synapse" in kwargs:
        i.set_synapse(kwargs["synapse"])
    if "nova" in kwargs:
        i.set_nova(kwargs["nova"])
    if "thymos" in kwargs:
        i.set_thymos(kwargs["thymos"])
    if "equor" in kwargs:
        i.set_equor(kwargs["equor"])
    if "token_budget" in kwargs:
        i.set_token_budget(kwargs["token_budget"])
    return i


def _mock_atune(valence=0.3, arousal=0.5, curiosity=0.4, care=0.2, coherence_stress=0.1):
    """Create a mock Atune with affect manager."""
    affect = MagicMock()
    affect.valence = valence
    affect.arousal = arousal
    affect.curiosity = curiosity
    affect.care_activation = care
    affect.coherence_stress = coherence_stress

    affect_mgr = MagicMock()
    affect_mgr.current = affect

    atune = MagicMock()
    atune._affect_mgr = affect_mgr
    atune.mean_prediction_error = 0.2
    return atune


class TestSenseAllDimensions:
    def test_returns_all_9_dimensions(self):
        i = _make_interoceptor()
        state = i.sense()
        assert len(state) == 9
        for dim in ALL_DIMENSIONS:
            assert dim in state

    def test_all_values_in_valid_ranges(self):
        i = _make_interoceptor(atune=_mock_atune())
        state = i.sense()
        for dim in ALL_DIMENSIONS:
            if dim == InteroceptiveDimension.VALENCE:
                assert -1.0 <= state[dim] <= 1.0
            else:
                assert 0.0 <= state[dim] <= 1.0


class TestFallbacks:
    def test_no_systems_uses_fallbacks(self):
        i = _make_interoceptor()
        state = i.sense()
        # Most dimensions use explicit FALLBACK_VALUES when sources unavailable.
        # Two exceptions have computed defaults that differ:
        # - INTEGRITY: min(thymos=1.0, equor=1.0) = 1.0
        # - TEMPORAL_PRESSURE: goal_urgency(0.0) + arousal_boost(0.0) = 0.0
        computed_defaults = {
            InteroceptiveDimension.INTEGRITY: 1.0,
            InteroceptiveDimension.TEMPORAL_PRESSURE: 0.0,
        }
        for dim in ALL_DIMENSIONS:
            if dim in computed_defaults:
                assert state[dim] == computed_defaults[dim]
            else:
                assert state[dim] == FALLBACK_VALUES[dim]

    def test_partial_systems_fills_missing(self):
        """Only wire Atune; other dimensions should use fallbacks."""
        i = _make_interoceptor(atune=_mock_atune())
        state = i.sense()
        assert state[InteroceptiveDimension.AROUSAL] == 0.5  # From mock
        # Integrity: min(1.0, 1.0) = 1.0 when no thymos/equor wired
        assert state[InteroceptiveDimension.INTEGRITY] == 1.0


class TestEnergyInteroceptor:
    def test_reads_token_budget(self):
        budget = MagicMock()
        budget_status = MagicMock()
        budget_status.utilization = 0.3
        budget.get_status.return_value = budget_status

        i = _make_interoceptor(token_budget=budget)
        state = i.sense()
        assert state[InteroceptiveDimension.ENERGY] == pytest.approx(0.7)

    def test_fallback_when_budget_none(self):
        i = _make_interoceptor()
        state = i.sense()
        assert state[InteroceptiveDimension.ENERGY] == FALLBACK_VALUES[InteroceptiveDimension.ENERGY]


class TestArousalInteroceptor:
    def test_reads_affect_arousal(self):
        i = _make_interoceptor(atune=_mock_atune(arousal=0.7))
        state = i.sense()
        assert state[InteroceptiveDimension.AROUSAL] == 0.7


class TestValenceInteroceptor:
    def test_reads_affect_valence(self):
        i = _make_interoceptor(atune=_mock_atune(valence=-0.3))
        state = i.sense()
        assert state[InteroceptiveDimension.VALENCE] == -0.3


class TestConfidenceInteroceptor:
    def test_reads_mean_prediction_error(self):
        atune = _mock_atune()
        atune.mean_prediction_error = 0.4
        i = _make_interoceptor(atune=atune)
        state = i.sense()
        assert state[InteroceptiveDimension.CONFIDENCE] == pytest.approx(0.6)


class TestCoherenceInteroceptor:
    def test_reads_synapse_phi(self):
        synapse = MagicMock(spec=[])  # spec=[] prevents auto-created attrs
        synapse._coherence = MagicMock()
        synapse._coherence.current_phi = 0.8
        i = _make_interoceptor(synapse=synapse)
        state = i.sense()
        assert state[InteroceptiveDimension.COHERENCE] == 0.8


class TestIntegrityInteroceptor:
    def test_min_of_thymos_and_equor(self):
        thymos = MagicMock()
        thymos.current_health_score = 0.9
        equor = MagicMock()
        equor.constitutional_drift = 0.2  # 1.0 - 0.2 = 0.8
        i = _make_interoceptor(thymos=thymos, equor=equor)
        state = i.sense()
        assert state[InteroceptiveDimension.INTEGRITY] == pytest.approx(0.8)

    def test_thymos_lower(self):
        thymos = MagicMock()
        thymos.current_health_score = 0.5
        equor = MagicMock()
        equor.constitutional_drift = 0.1
        i = _make_interoceptor(thymos=thymos, equor=equor)
        state = i.sense()
        assert state[InteroceptiveDimension.INTEGRITY] == pytest.approx(0.5)


class TestTemporalPressureInteroceptor:
    def test_goal_urgency_plus_arousal(self):
        nova = MagicMock()
        nova.goal_urgency_max = 0.4
        atune = _mock_atune(arousal=0.5)
        i = _make_interoceptor(nova=nova, atune=atune)
        state = i.sense()
        # 0.4 + 0.5 * 0.3 = 0.55
        assert state[InteroceptiveDimension.TEMPORAL_PRESSURE] == pytest.approx(0.55)

    def test_clamped_to_1(self):
        nova = MagicMock()
        nova.goal_urgency_max = 0.9
        atune = _mock_atune(arousal=0.8)
        i = _make_interoceptor(nova=nova, atune=atune)
        state = i.sense()
        assert state[InteroceptiveDimension.TEMPORAL_PRESSURE] == 1.0
