"""
Unit tests for Nova EFEEvaluator.

Tests EFE component computation, weighted total, constitutional alignment,
feasibility estimation, risk estimation, and LLM/heuristic estimation.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from clients.llm import LLMProvider, LLMResponse
from primitives.affect import AffectState
from primitives.common import DriveAlignmentVector, new_id
from systems.nova.efe_evaluator import (
    EFEEvaluator,
    _compute_constitutional_alignment,
    _estimate_epistemic_heuristic,
    _estimate_feasibility,
    _estimate_pragmatic_heuristic,
    _estimate_risk,
    _identify_uncertain_domains,
    _parse_json_response,
)
from systems.nova.policy_generator import DO_NOTHING_EFE, make_do_nothing_policy
from systems.nova.types import (
    BeliefState,
    EFEWeights,
    Goal,
    GoalSource,
    Policy,
    PolicyStep,
    SelfBelief,
)

# ─── Fixtures ─────────────────────────────────────────────────────


def make_goal(
    description: str = "Respond helpfully",
    domain: str = "dialogue",
    care: float = 0.3,
    growth: float = 0.2,
) -> Goal:
    return Goal(
        id=new_id(),
        description=description,
        target_domain=domain,
        source=GoalSource.USER_REQUEST,
        success_criteria="User receives a helpful response",
        drive_alignment=DriveAlignmentVector(
            coherence=0.3, care=care, growth=growth, honesty=0.2
        ),
    )


def make_policy(
    name: str = "Express response",
    effort: str = "low",
    horizon: str = "immediate",
    action_types: list[str] | None = None,
) -> Policy:
    if action_types is None:
        action_types = ["express"]
    return Policy(
        id=new_id(),
        name=name,
        estimated_effort=effort,
        time_horizon=horizon,
        steps=[PolicyStep(action_type=at, description=f"Step: {at}") for at in action_types],
    )


def make_beliefs(
    epistemic_confidence: float = 0.7,
    cognitive_load: float = 0.2,
    overall_confidence: float = 0.7,
) -> BeliefState:
    self_belief = SelfBelief(
        epistemic_confidence=epistemic_confidence,
        cognitive_load=cognitive_load,
    )
    return BeliefState(self_belief=self_belief, overall_confidence=overall_confidence)


def make_llm_mock(pragmatic_response: dict | None = None, epistemic_response: dict | None = None) -> AsyncMock:
    llm = AsyncMock(spec=LLMProvider)
    pr = pragmatic_response or {"success_probability": 0.8, "confidence": 0.7, "reasoning": "Good match"}
    er = epistemic_response or {"info_gain": 0.4, "uncertainties_addressed": 1, "novelty": 0.3}

    call_count = 0

    async def evaluate_side_effect(prompt, max_tokens=200, temperature=0.2):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(text=json.dumps(pr), model="mock", input_tokens=50, output_tokens=50, finish_reason="stop")
        return LLMResponse(text=json.dumps(er), model="mock", input_tokens=50, output_tokens=50, finish_reason="stop")

    llm.evaluate.side_effect = evaluate_side_effect
    return llm


NEUTRAL_DRIVE_WEIGHTS = {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
NEUTRAL_AFFECT = AffectState.neutral()


# ─── Do-Nothing Policy ────────────────────────────────────────────


class TestDoNothingEFE:
    @pytest.mark.asyncio
    async def test_do_nothing_fixed_efe(self) -> None:
        llm = AsyncMock(spec=LLMProvider)
        evaluator = EFEEvaluator(llm=llm)
        policy = make_do_nothing_policy()
        score = await evaluator.evaluate(
            policy, make_goal(), make_beliefs(), NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS
        )
        assert score.total == pytest.approx(DO_NOTHING_EFE)

    @pytest.mark.asyncio
    async def test_do_nothing_fully_feasible(self) -> None:
        llm = AsyncMock(spec=LLMProvider)
        evaluator = EFEEvaluator(llm=llm)
        policy = make_do_nothing_policy()
        score = await evaluator.evaluate(
            policy, make_goal(), make_beliefs(), NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS
        )
        assert score.feasibility == 1.0

    @pytest.mark.asyncio
    async def test_do_nothing_zero_risk(self) -> None:
        llm = AsyncMock(spec=LLMProvider)
        evaluator = EFEEvaluator(llm=llm)
        policy = make_do_nothing_policy()
        score = await evaluator.evaluate(
            policy, make_goal(), make_beliefs(), NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS
        )
        assert score.risk.expected_harm == 0.0


# ─── EFE Formula ──────────────────────────────────────────────────


class TestEFEFormula:
    @pytest.mark.asyncio
    async def test_efe_uses_weights(self) -> None:
        """Verify EFE total is a weighted combination (not just sum)."""
        llm = AsyncMock(spec=LLMProvider)
        weights = EFEWeights(pragmatic=0.35, epistemic=0.20, constitutional=0.20, feasibility=0.15, risk=0.10)
        evaluator = EFEEvaluator(llm=llm, weights=weights, use_llm_estimation=False)
        policy = make_policy()
        score = await evaluator.evaluate(
            policy, make_goal(), make_beliefs(), NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS
        )
        # Total should be negative (beneficial actions dominate)
        assert score.total < 0.0

    @pytest.mark.asyncio
    async def test_high_risk_policy_higher_efe(self) -> None:
        """High-risk policies should have higher (worse) EFE than low-risk ones."""
        llm = AsyncMock(spec=LLMProvider)
        evaluator = EFEEvaluator(llm=llm, use_llm_estimation=False)
        beliefs = make_beliefs()
        goal = make_goal()

        risky = make_policy(action_types=["federate", "external_api"])
        safe = make_policy(action_types=["express"])

        score_risky = await evaluator.evaluate(risky, goal, beliefs, NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS)
        score_safe = await evaluator.evaluate(safe, goal, beliefs, NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS)

        assert score_risky.total > score_safe.total

    @pytest.mark.asyncio
    async def test_evaluate_all_sorted_by_efe(self) -> None:
        llm = AsyncMock(spec=LLMProvider)
        evaluator = EFEEvaluator(llm=llm, use_llm_estimation=False)
        goal = make_goal()
        beliefs = make_beliefs()
        policies = [
            make_do_nothing_policy(),
            make_policy(name="Express", action_types=["express"]),
            make_policy(name="Risky", action_types=["federate"]),
        ]
        results = await evaluator.evaluate_all(policies, goal, beliefs, NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS)
        totals = [s.total for _, s in results]
        assert totals == sorted(totals)

    @pytest.mark.asyncio
    async def test_evaluate_all_handles_exceptions(self) -> None:
        """Failed evaluations get neutral score and don't block others."""
        llm = AsyncMock(spec=LLMProvider)
        llm.evaluate.side_effect = [Exception("fail"), Exception("fail")]
        evaluator = EFEEvaluator(llm=llm, use_llm_estimation=True)
        goal = make_goal()
        beliefs = make_beliefs()
        policies = [make_policy(), make_do_nothing_policy()]
        results = await evaluator.evaluate_all(policies, goal, beliefs, NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS)
        assert len(results) == 2

    def test_update_weights(self) -> None:
        llm = AsyncMock(spec=LLMProvider)
        evaluator = EFEEvaluator(llm=llm)
        new_weights = EFEWeights(pragmatic=0.5, epistemic=0.1, constitutional=0.1, feasibility=0.2, risk=0.1)
        evaluator.update_weights(new_weights)
        assert evaluator.weights.pragmatic == pytest.approx(0.5)


# ─── Constitutional Alignment ─────────────────────────────────────


class TestConstitutionalAlignment:
    def test_aligned_goal_scores_high(self) -> None:
        goal = make_goal(care=0.9, growth=0.8)
        weights = {"coherence": 1.0, "care": 2.0, "growth": 1.5, "honesty": 1.0}
        policy = make_policy()
        score = _compute_constitutional_alignment(policy, weights, goal)
        assert score > 0.5

    def test_misaligned_goal_scores_low(self) -> None:
        # All zeros except honesty
        goal = Goal(
            id=new_id(),
            description="test",
            source=GoalSource.USER_REQUEST,
            drive_alignment=DriveAlignmentVector(coherence=0.0, care=0.0, growth=0.0, honesty=0.0),
        )
        weights = {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
        score = _compute_constitutional_alignment(make_policy(), weights, goal)
        assert score == pytest.approx(0.0)

    def test_score_clamped_between_zero_and_one(self) -> None:
        goal = make_goal(care=1.0, growth=1.0)
        weights = {"coherence": 100.0, "care": 100.0, "growth": 100.0, "honesty": 100.0}
        score = _compute_constitutional_alignment(make_policy(), weights, goal)
        assert 0.0 <= score <= 1.0


# ─── Feasibility Estimation ───────────────────────────────────────


class TestFeasibilityEstimation:
    def test_observe_only_always_feasible(self) -> None:
        policy = make_policy(action_types=["observe", "wait"])
        beliefs = make_beliefs()
        assert _estimate_feasibility(policy, beliefs) == 1.0

    def test_high_cognitive_load_reduces_feasibility(self) -> None:
        policy = make_policy(action_types=["express"])
        low_load = make_beliefs(cognitive_load=0.1)
        high_load = make_beliefs(cognitive_load=0.9)
        assert _estimate_feasibility(policy, low_load) > _estimate_feasibility(policy, high_load)

    def test_high_effort_reduces_feasibility(self) -> None:
        beliefs = make_beliefs(epistemic_confidence=0.8)
        low_effort = make_policy(effort="low")
        high_effort = make_policy(effort="high")
        assert _estimate_feasibility(low_effort, beliefs) > _estimate_feasibility(high_effort, beliefs)

    def test_feasibility_clamped_above_zero(self) -> None:
        policy = make_policy(effort="high", action_types=["express"])
        beliefs = make_beliefs(epistemic_confidence=0.1, cognitive_load=0.9)
        assert _estimate_feasibility(policy, beliefs) >= 0.1


# ─── Risk Estimation ──────────────────────────────────────────────


class TestRiskEstimation:
    def test_express_action_low_risk(self) -> None:
        policy = make_policy(action_types=["express"])
        risk = _estimate_risk(policy)
        assert risk.expected_harm < 0.5

    def test_federate_action_high_risk(self) -> None:
        policy = make_policy(action_types=["federate"])
        risk = _estimate_risk(policy)
        assert risk.expected_harm > 0.0

    def test_long_horizon_adds_risk(self) -> None:
        short = make_policy(horizon="immediate", action_types=["express"])
        long = make_policy(horizon="long", action_types=["express"])
        assert _estimate_risk(long).expected_harm > _estimate_risk(short).expected_harm

    def test_expressive_only_fully_reversible(self) -> None:
        policy = make_policy(action_types=["observe", "wait", "express"])
        risk = _estimate_risk(policy)
        assert risk.reversibility == 1.0

    def test_high_risk_actions_identified(self) -> None:
        policy = make_policy(action_types=["federate", "external_api"])
        risk = _estimate_risk(policy)
        assert len(risk.identified_risks) > 0

    def test_risk_capped_at_one(self) -> None:
        policy = make_policy(
            action_types=["federate", "external_api", "irreversible"],
            horizon="long",
        )
        risk = _estimate_risk(policy)
        assert risk.expected_harm <= 1.0


# ─── Pragmatic Heuristic ──────────────────────────────────────────


class TestPragmaticHeuristic:
    def test_high_effort_higher_pragmatic(self) -> None:
        goal = make_goal(domain="dialogue")
        high_effort = make_policy(effort="high")
        low_effort = make_policy(effort="none")
        p_high = _estimate_pragmatic_heuristic(high_effort, goal)
        p_low = _estimate_pragmatic_heuristic(low_effort, goal)
        assert p_high.score > p_low.score

    def test_express_boosts_dialogue_domain(self) -> None:
        goal = make_goal(domain="dialogue")
        policy = make_policy(effort="low", action_types=["express"])
        non_express = make_policy(effort="low", action_types=["observe"])
        p_express = _estimate_pragmatic_heuristic(policy, goal)
        p_observe = _estimate_pragmatic_heuristic(non_express, goal)
        assert p_express.score >= p_observe.score

    def test_heuristic_confidence_is_low(self) -> None:
        goal = make_goal()
        policy = make_policy()
        result = _estimate_pragmatic_heuristic(policy, goal)
        assert result.confidence <= 0.5  # Low confidence in heuristic


# ─── Epistemic Heuristic ──────────────────────────────────────────


class TestEpistemicHeuristic:
    def test_observe_steps_increase_epistemic(self) -> None:
        beliefs = make_beliefs()
        observer = make_policy(action_types=["observe", "request_info"])
        actor = make_policy(action_types=["express"])
        e_obs = _estimate_epistemic_heuristic(observer, beliefs)
        e_act = _estimate_epistemic_heuristic(actor, beliefs)
        assert e_obs.score > e_act.score

    def test_low_confidence_adds_uncertainty_bonus(self) -> None:
        high_conf = make_beliefs(overall_confidence=0.9)
        low_conf = make_beliefs(overall_confidence=0.2)
        policy = make_policy(action_types=["express"])
        e_high = _estimate_epistemic_heuristic(policy, high_conf)
        e_low = _estimate_epistemic_heuristic(policy, low_conf)
        assert e_low.score > e_high.score

    def test_score_clamped(self) -> None:
        beliefs = make_beliefs(overall_confidence=0.0)
        policy = make_policy(action_types=["observe"] * 5)
        result = _estimate_epistemic_heuristic(policy, beliefs)
        assert 0.0 <= result.score <= 1.0


# ─── JSON Parsing ─────────────────────────────────────────────────


class TestJsonParsing:
    def test_parses_valid_json(self) -> None:
        data = {"success_probability": 0.8, "confidence": 0.7}
        result = _parse_json_response(json.dumps(data))
        assert result == data

    def test_strips_markdown_fences(self) -> None:
        raw = "```json\n{\"key\": \"value\"}\n```"
        result = _parse_json_response(raw)
        assert result == {"key": "value"}

    def test_returns_none_on_invalid(self) -> None:
        result = _parse_json_response("not json")
        assert result is None

    def test_returns_none_on_empty(self) -> None:
        result = _parse_json_response("")
        assert result is None


# ─── Uncertain Domain Identification ─────────────────────────────


class TestUncertainDomains:
    def test_low_overall_confidence_flagged(self) -> None:
        beliefs = make_beliefs(overall_confidence=0.2)
        result = _identify_uncertain_domains(beliefs)
        assert "world model" in result.lower() or "overall" in result.lower()

    def test_high_confidence_no_uncertainty(self) -> None:
        beliefs = make_beliefs(overall_confidence=0.9, epistemic_confidence=0.9)
        result = _identify_uncertain_domains(beliefs)
        # With high confidence in all areas, returns no specific uncertainties
        assert "no specific" in result or result == "no specific uncertainties identified"


# ─── LLM-Based Estimation ─────────────────────────────────────────


class TestLLMEstimation:
    @pytest.mark.asyncio
    async def test_llm_pragmatic_used_when_enabled(self) -> None:
        llm = make_llm_mock(
            pragmatic_response={"success_probability": 0.9, "confidence": 0.8, "reasoning": "Strong match"},
            epistemic_response={"info_gain": 0.4, "uncertainties_addressed": 1, "novelty": 0.2},
        )
        evaluator = EFEEvaluator(llm=llm, use_llm_estimation=True)
        policy = make_policy()
        score = await evaluator.evaluate(
            policy, make_goal(), make_beliefs(), NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS
        )
        assert score.pragmatic.score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_heuristic_used_when_llm_disabled(self) -> None:
        llm = AsyncMock(spec=LLMProvider)
        evaluator = EFEEvaluator(llm=llm, use_llm_estimation=False)
        policy = make_policy(effort="medium")
        score = await evaluator.evaluate(
            policy, make_goal(), make_beliefs(), NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS
        )
        llm.evaluate.assert_not_called()
        assert 0.0 <= score.pragmatic.score <= 1.0

    @pytest.mark.asyncio
    async def test_falls_back_to_heuristic_on_llm_error(self) -> None:
        llm = AsyncMock(spec=LLMProvider)
        llm.evaluate.side_effect = Exception("LLM down")
        evaluator = EFEEvaluator(llm=llm, use_llm_estimation=True)
        policy = make_policy()
        # Should not raise - fallback to heuristic
        score = await evaluator.evaluate(
            policy, make_goal(), make_beliefs(), NEUTRAL_AFFECT, NEUTRAL_DRIVE_WEIGHTS
        )
        assert score is not None
        assert 0.0 <= score.pragmatic.score <= 1.0
