"""Unit tests for PersonalityEngine."""

from __future__ import annotations

import pytest

from primitives.expression import PersonalityVector
from systems.voxis.personality import MAX_PERSONALITY_DELTA, PersonalityEngine
from systems.voxis.types import StrategyParams


@pytest.fixture
def neutral_engine() -> PersonalityEngine:
    return PersonalityEngine(PersonalityVector())


@pytest.fixture
def warm_direct_engine() -> PersonalityEngine:
    return PersonalityEngine(PersonalityVector(warmth=0.7, directness=0.6, empathy_expression=0.5))


@pytest.fixture
def base_strategy() -> StrategyParams:
    return StrategyParams(target_length=200)


class TestPersonalityApplication:
    def test_warmth_above_threshold_adds_warm_marker(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(warmth=0.5))
        result = engine.apply(base_strategy)
        assert "warm" in result.tone_markers

    def test_warmth_below_threshold_adds_measured_marker(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(warmth=-0.5))
        result = engine.apply(base_strategy)
        assert "measured" in result.tone_markers
        assert result.greeting_style == "professional"

    def test_directness_high_sets_conclusion_first(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(directness=0.6))
        result = engine.apply(base_strategy)
        assert result.structure == "conclusion_first"
        assert result.hedge_level == "minimal"

    def test_directness_low_sets_context_first(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(directness=-0.6))
        result = engine.apply(base_strategy)
        assert result.structure == "context_first"
        assert result.hedge_level == "moderate"

    def test_verbosity_positive_increases_target_length(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(verbosity=1.0))
        result = engine.apply(base_strategy)
        assert result.target_length > base_strategy.target_length

    def test_verbosity_negative_decreases_target_length(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(verbosity=-1.0))
        result = engine.apply(base_strategy)
        assert result.target_length < base_strategy.target_length

    def test_verbosity_target_length_clamped_above_minimum(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(verbosity=-1.0))
        short_strategy = StrategyParams(target_length=60)
        result = engine.apply(short_strategy)
        assert result.target_length >= 50

    def test_formality_high_sets_formal_register(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(formality=0.6))
        result = engine.apply(base_strategy)
        assert result.speech_register == "formal"
        assert result.contraction_use is False

    def test_formality_low_sets_casual_register(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(formality=-0.6))
        result = engine.apply(base_strategy)
        assert result.speech_register == "casual"
        assert result.contraction_use is True

    def test_curiosity_above_threshold_enables_followup_question(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(curiosity_expression=0.5))
        strategy_with_questions = StrategyParams(target_length=200, allows_questions=True)
        result = engine.apply(strategy_with_questions)
        assert result.include_followup_question is True

    def test_humour_enabled_when_context_appropriate(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(humour=0.6))
        strategy_with_humour_ctx = StrategyParams(
            target_length=200, context_appropriate_for_humour=True
        )
        result = engine.apply(strategy_with_humour_ctx)
        assert result.humour_allowed is True
        assert result.humour_probability > 0

    def test_humour_not_enabled_when_context_inappropriate(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(humour=0.8))
        result = engine.apply(base_strategy)  # context_appropriate_for_humour=False by default
        assert result.humour_allowed is False

    def test_empathy_high_sets_explicit_acknowledgment(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(empathy_expression=0.6))
        result = engine.apply(base_strategy)
        assert result.emotional_acknowledgment == "explicit"

    def test_empathy_low_sets_minimal_acknowledgment(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(empathy_expression=-0.6))
        result = engine.apply(base_strategy)
        assert result.emotional_acknowledgment == "minimal"

    def test_metaphor_use_enables_analogy(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(
            metaphor_use=0.6,
            thematic_references=["ecology", "systems theory"],
        ))
        result = engine.apply(base_strategy)
        assert result.analogy_encouraged is True
        assert "ecology" in result.preferred_analogy_domains

    def test_apply_does_not_mutate_input(self, base_strategy: StrategyParams) -> None:
        engine = PersonalityEngine(PersonalityVector(warmth=0.8, verbosity=0.8))
        original_length = base_strategy.target_length
        _ = engine.apply(base_strategy)
        assert base_strategy.target_length == original_length
        assert base_strategy.tone_markers == []


class TestPersonalityDelta:
    def test_delta_applied_within_bounds(self, neutral_engine: PersonalityEngine) -> None:
        new_vector = neutral_engine.apply_delta({"warmth": 0.05, "care_not_a_field": 0.5})
        # Warmth delta clamped to MAX_PERSONALITY_DELTA
        assert abs(new_vector.warmth - MAX_PERSONALITY_DELTA) < 0.001

    def test_delta_cannot_exceed_max(self, neutral_engine: PersonalityEngine) -> None:
        new_vector = neutral_engine.apply_delta({"warmth": 1.0})
        assert new_vector.warmth <= MAX_PERSONALITY_DELTA + 0.001

    def test_negative_delta_applied(self, warm_direct_engine: PersonalityEngine) -> None:
        original_warmth = warm_direct_engine.current.warmth
        new_vector = warm_direct_engine.apply_delta({"warmth": -0.03})
        assert new_vector.warmth < original_warmth

    def test_values_clamped_to_valid_range(self) -> None:
        engine = PersonalityEngine(PersonalityVector(warmth=0.99))
        new_vector = engine.apply_delta({"warmth": 0.03})
        assert new_vector.warmth <= 1.0

    def test_humour_clamped_to_zero_minimum(self) -> None:
        engine = PersonalityEngine(PersonalityVector(humour=0.01))
        new_vector = engine.apply_delta({"humour": -0.03})
        assert new_vector.humour >= 0.0

    def test_unchanged_dimensions_preserved(self, warm_direct_engine: PersonalityEngine) -> None:
        original = warm_direct_engine.current
        new_vector = warm_direct_engine.apply_delta({"warmth": 0.01})
        assert new_vector.directness == original.directness
        assert new_vector.verbosity == original.verbosity

    def test_original_engine_not_mutated(self, neutral_engine: PersonalityEngine) -> None:
        _ = neutral_engine.apply_delta({"warmth": 0.03})
        assert neutral_engine.current.warmth == 0.0
