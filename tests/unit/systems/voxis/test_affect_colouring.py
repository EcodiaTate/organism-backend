"""Unit tests for AffectColouringEngine."""

from __future__ import annotations

import pytest

from primitives.affect import AffectState
from systems.voxis.affect_colouring import AffectColouringEngine
from systems.voxis.types import StrategyParams


@pytest.fixture
def engine() -> AffectColouringEngine:
    return AffectColouringEngine()


@pytest.fixture
def base_strategy() -> StrategyParams:
    return StrategyParams(target_length=200, allows_questions=True)


def make_affect(**kwargs) -> AffectState:
    return AffectState(**kwargs)


class TestAffectApplication:
    def test_high_care_adds_attentive_marker(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(care_activation=0.8)
        result = engine.apply(base_strategy, affect)
        assert "attentive" in result.tone_markers

    def test_high_care_sets_shorter_sentences(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(care_activation=0.8)
        result = engine.apply(base_strategy, affect)
        assert result.sentence_length_preference == "shorter"

    def test_high_care_enables_wellbeing_check(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(care_activation=0.8)
        result = engine.apply(base_strategy, affect)
        assert result.include_wellbeing_check is True

    def test_high_arousal_reduces_target_length(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(arousal=0.9)
        result = engine.apply(base_strategy, affect)
        assert result.target_length < base_strategy.target_length

    def test_high_arousal_sets_energetic_pacing(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(arousal=0.9)
        result = engine.apply(base_strategy, affect)
        assert result.pacing == "energetic"

    def test_low_arousal_sets_reflective_pacing(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(arousal=0.1)
        result = engine.apply(base_strategy, affect)
        assert result.pacing == "reflective"
        assert "measured" in result.tone_markers

    def test_high_curiosity_adds_inquisitive_marker(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(curiosity=0.8)
        result = engine.apply(base_strategy, affect)
        assert "inquisitive" in result.tone_markers

    def test_negative_valence_increases_hedge_level(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(valence=-0.5)
        result = engine.apply(base_strategy, affect)
        assert result.hedge_level in ("moderate", "explicit")

    def test_negative_valence_sets_cautious_override(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(valence=-0.5)
        result = engine.apply(base_strategy, affect)
        assert result.confidence_display_override == "cautious"

    def test_high_coherence_stress_sets_explicit_uncertainty(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(coherence_stress=0.8)
        result = engine.apply(base_strategy, affect)
        assert result.uncertainty_acknowledgment == "explicit"
        assert "thoughtful" in result.tone_markers

    def test_positive_valence_low_stress_boosts_warmth(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(valence=0.5, coherence_stress=0.1)
        result = engine.apply(base_strategy, affect)
        assert result.warmth_boost > 0

    def test_apply_does_not_mutate_input(
        self, engine: AffectColouringEngine, base_strategy: StrategyParams
    ) -> None:
        affect = make_affect(care_activation=0.9, arousal=0.9)
        original_length = base_strategy.target_length
        _ = engine.apply(base_strategy, affect)
        assert base_strategy.target_length == original_length


class TestHonestyCheck:
    def test_forced_positivity_detected_with_negative_valence(
        self, engine: AffectColouringEngine
    ) -> None:
        affect = make_affect(valence=-0.5)
        passed, detail = engine.check_authenticity("I'm thrilled to help you today!", affect)
        assert passed is False
        assert detail is not None

    def test_false_calm_detected_with_high_stress(
        self, engine: AffectColouringEngine
    ) -> None:
        affect = make_affect(coherence_stress=0.8)
        passed, detail = engine.check_authenticity(
            "Everything is fine. No worries at all.", affect
        )
        assert passed is False
        assert detail is not None

    def test_authentic_expression_passes(self, engine: AffectColouringEngine) -> None:
        affect = make_affect(valence=-0.3, coherence_stress=0.7)
        passed, detail = engine.check_authenticity(
            "This is a complex situation. I'm not certain about the best path forward.", affect
        )
        assert passed is True
        assert detail is None

    def test_positive_expression_with_positive_affect_passes(
        self, engine: AffectColouringEngine
    ) -> None:
        affect = make_affect(valence=0.6, coherence_stress=0.1)
        passed, detail = engine.check_authenticity(
            "That's wonderful news - I'm really glad to hear it.", affect
        )
        assert passed is True
        assert detail is None
