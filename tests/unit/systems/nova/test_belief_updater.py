"""
Unit tests for Nova BeliefUpdater.

Tests Bayesian belief updating, context inference, entity management,
confidence decay, and outcome-based updates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from primitives.affect import AffectState
from systems.atune.types import (
    PredictionError,
    PredictionErrorDirection,
    SalienceVector,
    WorkspaceBroadcast,
)
from systems.nova.belief_updater import (
    BeliefUpdater,
    _bayesian_update_entity,
    _infer_domain,
    _is_dialogue,
    _update_individual_belief,
)
from systems.nova.types import BeliefState, EntityBelief, IndividualBelief

# ─── Fixtures ─────────────────────────────────────────────────────


def make_broadcast(
    content_text: str = "Hello there",
    precision: float = 0.5,
    care_activation: float = 0.0,
    novelty: float = 0.0,
    prediction_error_magnitude: float = 0.0,
) -> WorkspaceBroadcast:
    """Build a minimal WorkspaceBroadcast for testing."""
    affect = AffectState.neutral()
    affect = affect.model_copy(update={"care_activation": care_activation})

    scores = {}
    if novelty > 0:
        scores["novelty"] = novelty

    salience = SalienceVector(scores=scores, composite=precision)
    if prediction_error_magnitude > 0:
        salience = salience.model_copy(
            update={
                "prediction_error": PredictionError(
                    magnitude=prediction_error_magnitude,
                    direction=PredictionErrorDirection.NOVEL,
                )
            }
        )

    # Minimal content object
    content = MagicMock()
    content.content = MagicMock()
    content.content.content = content_text

    return WorkspaceBroadcast(
        content=content,
        salience=salience,
        affect=affect,
        precision=precision,
    )


@pytest.fixture
def updater() -> BeliefUpdater:
    return BeliefUpdater()


# ─── BeliefUpdater: Initial State ─────────────────────────────────


class TestInitialState:
    def test_initial_beliefs_empty(self, updater: BeliefUpdater) -> None:
        b = updater.beliefs
        assert len(b.entities) == 0
        assert len(b.individual_beliefs) == 0

    def test_initial_confidence_half(self, updater: BeliefUpdater) -> None:
        assert updater.beliefs.overall_confidence == 0.5

    def test_initial_free_energy_nonzero(self, updater: BeliefUpdater) -> None:
        # VFE = 1 - mean(confidences), so with 0.5 confidence → ~0.5
        assert 0.4 <= updater.beliefs.free_energy <= 0.6


# ─── BeliefUpdater: Context Updates ───────────────────────────────


class TestContextUpdates:
    def test_update_returns_belief_delta(self, updater: BeliefUpdater) -> None:
        broadcast = make_broadcast("I need help with something")
        delta = updater.update_from_broadcast(broadcast)
        assert delta is not None

    def test_context_summary_updated(self, updater: BeliefUpdater) -> None:
        broadcast = make_broadcast("I need help with my code today")
        updater.update_from_broadcast(broadcast)
        assert updater.beliefs.current_context.summary != ""

    def test_context_domain_technical(self, updater: BeliefUpdater) -> None:
        broadcast = make_broadcast("I have a bug in my algorithm")
        updater.update_from_broadcast(broadcast)
        assert updater.beliefs.current_context.domain == "technical"

    def test_context_domain_emotional(self, updater: BeliefUpdater) -> None:
        broadcast = make_broadcast("I feel really sad today")
        updater.update_from_broadcast(broadcast)
        assert updater.beliefs.current_context.domain == "emotional"

    def test_context_domain_social(self, updater: BeliefUpdater) -> None:
        broadcast = make_broadcast("our community needs to work together")
        updater.update_from_broadcast(broadcast)
        assert updater.beliefs.current_context.domain == "social"

    def test_context_confidence_increases_with_precision(self, updater: BeliefUpdater) -> None:
        initial_conf = updater.beliefs.current_context.confidence
        broadcast = make_broadcast(precision=0.9)
        updater.update_from_broadcast(broadcast)
        assert updater.beliefs.current_context.confidence >= initial_conf

    def test_prediction_error_from_salience(self, updater: BeliefUpdater) -> None:
        broadcast = make_broadcast(prediction_error_magnitude=0.8)
        delta = updater.update_from_broadcast(broadcast)
        assert delta.prediction_error_magnitude == pytest.approx(0.8)

    def test_high_prediction_error_triggers_conflict(self, updater: BeliefUpdater) -> None:
        broadcast = make_broadcast(prediction_error_magnitude=0.9)
        delta = updater.update_from_broadcast(broadcast)
        assert delta.involves_belief_conflict()


# ─── BeliefUpdater: Free Energy ───────────────────────────────────


class TestFreeEnergy:
    def test_free_energy_recomputed_after_update(self, updater: BeliefUpdater) -> None:
        broadcast = make_broadcast(precision=0.95)
        updater.update_from_broadcast(broadcast)
        # With high precision, context confidence rises → VFE drops
        assert 0.0 <= updater.beliefs.free_energy <= 1.0

    def test_free_energy_lower_with_high_confidence(self) -> None:
        """Higher confidence beliefs should produce lower VFE."""
        state = BeliefState(overall_confidence=0.9)
        low_vfe = state.compute_free_energy()
        state2 = BeliefState(overall_confidence=0.1)
        high_vfe = state2.compute_free_energy()
        assert low_vfe < high_vfe


# ─── Bayesian Entity Update ────────────────────────────────────────


class TestBayesianEntityUpdate:
    def test_confidence_increases_toward_one(self) -> None:
        prior = EntityBelief(entity_id="e1", confidence=0.5)
        updated = _bayesian_update_entity(prior, precision=0.6)
        # new = 0.5 + 0.6 * (1 - 0.5) = 0.5 + 0.3 = 0.8
        assert updated.confidence == pytest.approx(0.8)

    def test_high_confidence_saturates_at_one(self) -> None:
        prior = EntityBelief(entity_id="e1", confidence=0.99)
        updated = _bayesian_update_entity(prior, precision=1.0)
        assert updated.confidence <= 1.0

    def test_zero_precision_no_change(self) -> None:
        prior = EntityBelief(entity_id="e1", confidence=0.4)
        updated = _bayesian_update_entity(prior, precision=0.0)
        assert updated.confidence == pytest.approx(0.4)

    def test_last_observed_updated(self) -> None:
        prior = EntityBelief(
            entity_id="e1",
            confidence=0.5,
            last_observed=datetime(2020, 1, 1, tzinfo=UTC),
        )
        updated = _bayesian_update_entity(prior, precision=0.5)
        assert updated.last_observed > prior.last_observed

    def test_other_fields_preserved(self) -> None:
        prior = EntityBelief(entity_id="e1", name="Alice", entity_type="person", confidence=0.3)
        updated = _bayesian_update_entity(prior, precision=0.5)
        assert updated.name == "Alice"
        assert updated.entity_type == "person"


# ─── Individual Belief Update ─────────────────────────────────────


class TestIndividualBeliefUpdate:
    def _make_broadcast(self, valence: float = 0.0, arousal: float = 0.5) -> WorkspaceBroadcast:
        affect = AffectState.neutral().model_copy(update={"valence": valence, "arousal": arousal})
        return WorkspaceBroadcast(
            content=MagicMock(),
            salience=SalienceVector(composite=0.5),
            affect=affect,
            precision=0.6,
        )

    def test_creates_new_individual_belief(self) -> None:
        broadcast = self._make_broadcast(valence=0.8)
        result = _update_individual_belief("u1", None, broadcast, precision=0.5)
        assert result.individual_id == "u1"

    def test_valence_moves_toward_observation(self) -> None:
        current = IndividualBelief(individual_id="u1", estimated_valence=0.0)
        broadcast = self._make_broadcast(valence=1.0)
        updated = _update_individual_belief("u1", current, broadcast, precision=0.8)
        # Should move toward 1.0
        assert updated.estimated_valence > current.estimated_valence

    def test_valence_confidence_increases(self) -> None:
        current = IndividualBelief(individual_id="u1", valence_confidence=0.3)
        broadcast = self._make_broadcast()
        updated = _update_individual_belief("u1", current, broadcast, precision=0.5)
        assert updated.valence_confidence > current.valence_confidence

    def test_valence_clamped(self) -> None:
        current = IndividualBelief(individual_id="u1", estimated_valence=0.9)
        broadcast = self._make_broadcast(valence=1.0)
        updated = _update_individual_belief("u1", current, broadcast, precision=1.0)
        assert updated.estimated_valence <= 1.0


# ─── Outcome Updates ──────────────────────────────────────────────


class TestOutcomeUpdates:
    def test_success_increases_overall_confidence(self, updater: BeliefUpdater) -> None:
        initial = updater.beliefs.overall_confidence
        updater.update_from_outcome("Task completed", success=True, precision=0.7)
        assert updater.beliefs.overall_confidence >= initial

    def test_failure_decreases_overall_confidence(self, updater: BeliefUpdater) -> None:
        initial = updater.beliefs.overall_confidence
        updater.update_from_outcome("Task failed", success=False, precision=0.7)
        assert updater.beliefs.overall_confidence <= initial

    def test_success_increases_epistemic_confidence(self, updater: BeliefUpdater) -> None:
        initial = updater.beliefs.self_belief.epistemic_confidence
        updater.update_from_outcome("Succeeded", success=True)
        assert updater.beliefs.self_belief.epistemic_confidence >= initial

    def test_failure_decreases_epistemic_confidence(self, updater: BeliefUpdater) -> None:
        initial = updater.beliefs.self_belief.epistemic_confidence
        updater.update_from_outcome("Failed", success=False)
        assert updater.beliefs.self_belief.epistemic_confidence <= initial

    def test_confidence_clamped_above_zero(self, updater: BeliefUpdater) -> None:
        for _ in range(50):
            updater.update_from_outcome("Failed repeatedly", success=False, precision=1.0)
        assert updater.beliefs.overall_confidence >= 0.1


# ─── Entity Management ────────────────────────────────────────────


class TestEntityManagement:
    def test_inject_entity(self, updater: BeliefUpdater) -> None:
        updater.inject_entity("e1", "Alice", confidence=0.7)
        assert "e1" in updater.beliefs.entities
        assert updater.beliefs.entities["e1"].name == "Alice"

    def test_decay_reduces_confidence(self, updater: BeliefUpdater) -> None:
        updater.inject_entity("e1", "Bob", confidence=0.5)
        updater.decay_unobserved_entities()
        # Decay = 0.005 per cycle
        assert updater.beliefs.entities["e1"].confidence < 0.5

    def test_decay_prunes_low_confidence(self, updater: BeliefUpdater) -> None:
        updater.inject_entity("e_low", "LowConf", confidence=0.04)
        updater.decay_unobserved_entities()
        # Below 0.05 threshold → should be pruned
        assert "e_low" not in updater.beliefs.entities

    def test_decay_preserves_high_confidence(self, updater: BeliefUpdater) -> None:
        updater.inject_entity("e_high", "HighConf", confidence=0.9)
        updater.decay_unobserved_entities()
        assert "e_high" in updater.beliefs.entities


# ─── Domain Inference ─────────────────────────────────────────────


class TestDomainInference:
    def test_emotional_keywords(self) -> None:
        assert _infer_domain("I feel so sad and worried") == "emotional"

    def test_technical_keywords(self) -> None:
        assert _infer_domain("there's a bug in my code function") == "technical"

    def test_social_keywords(self) -> None:
        assert _infer_domain("our community needs to work together") == "social"

    def test_request_keywords(self) -> None:
        assert _infer_domain("can you help me please") == "request"

    def test_general_fallback(self) -> None:
        assert _infer_domain("the weather is nice today") == "general"


# ─── Dialogue Detection ───────────────────────────────────────────


class TestDialogueDetection:
    def test_question_mark_is_dialogue(self) -> None:
        assert _is_dialogue("What time is it?") is True

    def test_greeting_is_dialogue(self) -> None:
        assert _is_dialogue("Hello, how are you?") is True

    def test_empty_is_not_dialogue(self) -> None:
        assert _is_dialogue("") is False

    def test_long_text_not_dialogue(self) -> None:
        long_text = "word " * 50  # >200 chars without question or greeting
        # No question mark and doesn't start with greeting
        if "?" not in long_text and not any(
            long_text.lower().startswith(p) for p in ["hey", "hi", "hello", "eos,", "can you", "could you"]
        ):
            assert _is_dialogue(long_text) is False

    def test_short_message_is_dialogue(self) -> None:
        assert _is_dialogue("okay") is True  # len < 200
