"""Unit tests for ContentRenderer - EFE policy selection and temperature calibration."""

from __future__ import annotations

from primitives.affect import AffectState
from systems.voxis.renderer import (
    ExpressionPolicy,
    ExpressionPolicyClass,
    _compute_temperature,
    _derive_candidate_policies,
    _select_minimum_efe_policy,
)
from systems.voxis.types import ExpressionIntent, ExpressionTrigger, StrategyParams


def make_intent(
    trigger: ExpressionTrigger = ExpressionTrigger.NOVA_RESPOND,
    content: str = "test content",
    urgency: float = 0.5,
) -> ExpressionIntent:
    return ExpressionIntent(trigger=trigger, content_to_express=content, urgency=urgency)


def make_affect(**kwargs) -> AffectState:
    return AffectState(**kwargs)


class TestEFEPolicySelection:
    def test_derive_candidate_policies_returns_at_least_two(self) -> None:
        intent = make_intent(ExpressionTrigger.NOVA_RESPOND)
        affect = make_affect()
        base = StrategyParams()
        policies = _derive_candidate_policies(intent, affect, base)
        assert len(policies) >= 2

    def test_affiliative_policy_included_for_distress_trigger(self) -> None:
        intent = make_intent(ExpressionTrigger.ATUNE_DISTRESS)
        affect = make_affect(care_activation=0.8)
        base = StrategyParams()
        policies = _derive_candidate_policies(intent, affect, base)
        classes = [p.policy_class for p in policies]
        assert ExpressionPolicyClass.AFFILIATIVE in classes

    def test_epistemic_policy_included_when_high_curiosity(self) -> None:
        intent = make_intent(ExpressionTrigger.NOVA_RESPOND)
        affect = make_affect(curiosity=0.8)
        base = StrategyParams()
        policies = _derive_candidate_policies(intent, affect, base)
        classes = [p.policy_class for p in policies]
        assert ExpressionPolicyClass.EPISTEMIC in classes

    def test_minimum_efe_policy_selected(self) -> None:
        # Create two policies with known EFE scores
        p1 = ExpressionPolicy(
            policy_class=ExpressionPolicyClass.PRAGMATIC,
            strategy=StrategyParams(),
            coherence_alignment=0.9,
            care_alignment=0.9,
            growth_alignment=0.9,
            honesty_alignment=0.9,
            epistemic_value=0.1,
        )
        p2 = ExpressionPolicy(
            policy_class=ExpressionPolicyClass.EPISTEMIC,
            strategy=StrategyParams(),
            coherence_alignment=0.2,
            care_alignment=0.2,
            growth_alignment=0.2,
            honesty_alignment=0.2,
            epistemic_value=0.0,
        )
        weights = {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
        selected = _select_minimum_efe_policy([p1, p2], weights)
        assert selected is p1  # p1 has higher alignment = lower EFE (more negative)

    def test_care_drive_weight_biases_toward_affiliative(self) -> None:
        intent = make_intent(ExpressionTrigger.NOVA_RESPOND)
        affect = make_affect(care_activation=0.7)
        base = StrategyParams()
        policies = _derive_candidate_policies(intent, affect, base)
        # With high care weight, affiliative policy should score competitively
        weights = {"coherence": 0.5, "care": 2.0, "growth": 0.5, "honesty": 0.5}
        selected = _select_minimum_efe_policy(policies, weights)
        assert selected.policy_class == ExpressionPolicyClass.AFFILIATIVE

    def test_efe_score_computed_correctly(self) -> None:
        policy = ExpressionPolicy(
            policy_class=ExpressionPolicyClass.PRAGMATIC,
            strategy=StrategyParams(),
            coherence_alignment=1.0,
            care_alignment=1.0,
            growth_alignment=1.0,
            honesty_alignment=1.0,
            epistemic_value=0.0,
        )
        weights = {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
        efe = policy.compute_efe(weights)
        # All alignments = 1.0, weights = 1.0 each, sum = 4.0, normalised = 1.0
        # EFE = -(1.0 + 0.0) = -1.0
        assert abs(efe - (-1.0)) < 0.001


class TestTemperatureCalibration:
    def test_base_temperature_returned_for_neutral_state(self) -> None:
        strategy = StrategyParams(context_type="conversation")
        affect = make_affect(coherence_stress=0.0)
        temp = _compute_temperature(strategy, affect, base_temp=0.7)
        assert abs(temp - 0.7) < 0.01

    def test_high_coherence_stress_reduces_temperature(self) -> None:
        strategy = StrategyParams(context_type="conversation")
        low_stress = make_affect(coherence_stress=0.1)
        high_stress = make_affect(coherence_stress=0.9)
        temp_low = _compute_temperature(strategy, low_stress, base_temp=0.7)
        temp_high = _compute_temperature(strategy, high_stress, base_temp=0.7)
        assert temp_high < temp_low

    def test_celebration_context_increases_temperature(self) -> None:
        neutral = StrategyParams(context_type="conversation")
        celebration = StrategyParams(context_type="celebration")
        affect = make_affect(coherence_stress=0.0)
        temp_neutral = _compute_temperature(neutral, affect, base_temp=0.7)
        temp_celebration = _compute_temperature(celebration, affect, base_temp=0.7)
        assert temp_celebration > temp_neutral

    def test_warning_context_reduces_temperature(self) -> None:
        neutral = StrategyParams(context_type="conversation")
        warning = StrategyParams(context_type="warning")
        affect = make_affect(coherence_stress=0.0)
        temp_neutral = _compute_temperature(neutral, affect, base_temp=0.7)
        temp_warning = _compute_temperature(warning, affect, base_temp=0.7)
        assert temp_warning < temp_neutral

    def test_temperature_clamped_to_valid_range(self) -> None:
        strategy = StrategyParams(context_type="warning")
        affect = make_affect(coherence_stress=1.0)
        temp = _compute_temperature(strategy, affect, base_temp=0.7)
        assert 0.30 <= temp <= 1.00

    def test_humour_allowed_slightly_raises_temperature(self) -> None:
        without_humour = StrategyParams(context_type="conversation", humour_allowed=False)
        with_humour = StrategyParams(context_type="conversation", humour_allowed=True)
        affect = make_affect(coherence_stress=0.2)
        temp_without = _compute_temperature(without_humour, affect, base_temp=0.7)
        temp_with = _compute_temperature(with_humour, affect, base_temp=0.7)
        assert temp_with > temp_without
