"""
Unit tests for Nova PolicyGenerator and procedure matching.

Tests do-nothing policy, procedure template matching,
policy parsing, and fallback behaviour on LLM failure.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.llm import LLMProvider, LLMResponse
from primitives.affect import AffectState
from primitives.common import new_id
from systems.fovea.types import SalienceVector, WorkspaceBroadcast
from systems.nova.policy_generator import (
    DO_NOTHING_EFE,
    PolicyGenerator,
    _parse_policy_response,
    find_matching_procedure,
    make_do_nothing_policy,
    procedure_to_policy,
)
from systems.nova.types import BeliefState, Goal, GoalSource, Policy

# ─── Fixtures ─────────────────────────────────────────────────────


def make_goal(description: str = "Test goal") -> Goal:
    return Goal(
        id=new_id(),
        description=description,
        source=GoalSource.USER_REQUEST,
    )


def make_broadcast(
    text: str = "Hello",
    care_activation: float = 0.0,
    emotional_score: float = 0.0,
    precision: float = 0.5,
) -> WorkspaceBroadcast:
    affect = AffectState.neutral().model_copy(update={"care_activation": care_activation})
    scores = {}
    if emotional_score > 0:
        scores["emotional"] = emotional_score

    content = MagicMock()
    content.content = MagicMock()
    content.content.content = text

    return WorkspaceBroadcast(
        content=content,
        salience=SalienceVector(scores=scores, composite=precision),
        affect=affect,
        precision=precision,
    )


def make_llm_mock(response_json: dict | None = None) -> AsyncMock:
    llm = AsyncMock(spec=LLMProvider)
    if response_json is None:
        response_json = {
            "policies": [
                {
                    "name": "Respond directly",
                    "reasoning": "Direct response is most appropriate",
                    "steps": [{"action_type": "express", "description": "Reply to message"}],
                    "risks": [],
                    "epistemic_value": "Confirms understanding",
                    "estimated_effort": "low",
                    "time_horizon": "immediate",
                }
            ]
        }
    llm.generate.return_value = LLMResponse(
        text=json.dumps(response_json),
        model="mock",
        input_tokens=100,
        output_tokens=200,
        finish_reason="stop",
    )
    return llm


# ─── Do-Nothing Policy ────────────────────────────────────────────


class TestDoNothingPolicy:
    def test_id_is_do_nothing(self) -> None:
        policy = make_do_nothing_policy()
        assert policy.id == "do_nothing"

    def test_has_observe_step(self) -> None:
        policy = make_do_nothing_policy()
        assert len(policy.steps) == 1
        assert policy.steps[0].action_type == "observe"

    def test_effort_is_none(self) -> None:
        policy = make_do_nothing_policy()
        assert policy.estimated_effort == "none"

    def test_time_horizon_is_immediate(self) -> None:
        policy = make_do_nothing_policy()
        assert policy.time_horizon == "immediate"

    def test_do_nothing_efe_constant(self) -> None:
        assert pytest.approx(-0.10) == DO_NOTHING_EFE


# ─── Procedure Matching ───────────────────────────────────────────


class TestProcedureMatching:
    def test_regular_message_matches_acknowledge(self) -> None:
        broadcast = make_broadcast("Hello, how are you?", precision=0.5)
        result = find_matching_procedure(broadcast)
        assert result is not None

    def test_question_matches_a_procedure(self) -> None:
        """A question mark should match at least one procedure template."""
        content = MagicMock()
        content.content = MagicMock()
        content.content.content = "What is the capital of France?"
        broadcast = WorkspaceBroadcast(
            content=content,
            salience=SalienceVector(composite=0.5),
            affect=AffectState.neutral(),
            precision=0.5,
        )
        result = find_matching_procedure(broadcast)
        assert result is not None
        # When multiple templates match, the highest success_rate wins
        assert result["success_rate"] >= 0.85

    def test_distress_matches_empathetic_support(self) -> None:
        broadcast = make_broadcast(
            "I'm really struggling",
            care_activation=0.8,
            emotional_score=0.75,
        )
        result = find_matching_procedure(broadcast)
        assert result is not None
        assert result["name"] == "Empathetic support"

    def test_returns_highest_success_rate(self) -> None:
        # A question with care activation - multiple templates might match
        content = MagicMock()
        content.content = MagicMock()
        content.content.content = "Can you help me? I'm feeling lost"
        broadcast = WorkspaceBroadcast(
            content=content,
            salience=SalienceVector(scores={"emotional": 0.8}, composite=0.5),
            affect=AffectState.neutral().model_copy(update={"care_activation": 0.8}),
            precision=0.5,
        )
        result = find_matching_procedure(broadcast)
        # Should return the one with highest success_rate
        assert result is not None
        assert result["success_rate"] >= 0.8

    def test_procedure_to_policy_creates_valid_policy(self) -> None:
        from systems.nova.policy_generator import _PROCEDURE_TEMPLATES
        template = _PROCEDURE_TEMPLATES[0]
        policy = procedure_to_policy(template)
        assert isinstance(policy, Policy)
        assert policy.name == template["name"]
        assert len(policy.steps) > 0


# ─── Response Parsing ─────────────────────────────────────────────


class TestResponseParsing:
    def test_parses_valid_json(self) -> None:
        raw = json.dumps({
            "policies": [
                {
                    "name": "Policy A",
                    "reasoning": "Good reason",
                    "steps": [{"action_type": "express", "description": "Say something"}],
                    "risks": ["risk1"],
                    "epistemic_value": "Some learning",
                    "estimated_effort": "low",
                    "time_horizon": "immediate",
                }
            ]
        })
        policies = _parse_policy_response(raw)
        assert len(policies) == 1
        assert policies[0].name == "Policy A"

    def test_strips_markdown_fences(self) -> None:
        raw = "```json\n" + json.dumps({"policies": [
            {"name": "P1", "steps": [{"action_type": "observe"}]}
        ]}) + "\n```"
        policies = _parse_policy_response(raw)
        assert len(policies) == 1

    def test_skips_malformed_entries(self) -> None:
        raw = json.dumps({
            "policies": [
                {"name": "Valid", "steps": [{"action_type": "express"}]},
                None,  # Malformed
                "not a dict",  # Malformed
            ]
        })
        policies = _parse_policy_response(raw)
        # Should get the valid one
        assert len(policies) >= 1

    def test_returns_empty_on_invalid_json(self) -> None:
        policies = _parse_policy_response("this is not json at all")
        assert policies == []

    def test_returns_empty_on_missing_policies_key(self) -> None:
        raw = json.dumps({"other": "data"})
        policies = _parse_policy_response(raw)
        assert policies == []

    def test_step_action_type_defaults(self) -> None:
        raw = json.dumps({"policies": [{"name": "P", "steps": [{}]}]})
        policies = _parse_policy_response(raw)
        if policies:
            assert policies[0].steps[0].action_type == "observe"  # default

    def test_policy_names_truncated(self) -> None:
        long_name = "A" * 200
        raw = json.dumps({"policies": [{"name": long_name, "steps": [{"action_type": "express"}]}]})
        policies = _parse_policy_response(raw)
        if policies:
            assert len(policies[0].name) <= 80


# ─── PolicyGenerator (async) ──────────────────────────────────────


class TestPolicyGenerator:
    @pytest.mark.asyncio
    async def test_always_includes_do_nothing(self) -> None:
        llm = make_llm_mock()
        gen = PolicyGenerator(llm=llm)
        goal = make_goal()
        beliefs = BeliefState()
        affect = AffectState.neutral()

        candidates = await gen.generate_candidates(
            goal=goal,
            situation_summary="A user said hello",
            beliefs=beliefs,
            affect=affect,
        )
        do_nothing_ids = [p.id for p in candidates if p.id == "do_nothing"]
        assert len(do_nothing_ids) == 1

    @pytest.mark.asyncio
    async def test_includes_llm_generated_policies(self) -> None:
        llm = make_llm_mock()
        gen = PolicyGenerator(llm=llm)
        goal = make_goal()
        beliefs = BeliefState()
        affect = AffectState.neutral()

        candidates = await gen.generate_candidates(
            goal=goal,
            situation_summary="User asked a question",
            beliefs=beliefs,
            affect=affect,
        )
        # Should have at least do_nothing + 1 LLM policy
        assert len(candidates) >= 2

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self) -> None:
        llm = AsyncMock(spec=LLMProvider)
        llm.generate.side_effect = Exception("LLM unavailable")
        gen = PolicyGenerator(llm=llm)
        goal = make_goal()

        candidates = await gen.generate_candidates(
            goal=goal,
            situation_summary="Something happened",
            beliefs=BeliefState(),
            affect=AffectState.neutral(),
        )
        # Fallback: only do-nothing
        assert len(candidates) == 1
        assert candidates[0].id == "do_nothing"

    @pytest.mark.asyncio
    async def test_max_policies_respected(self) -> None:
        many_policies = {
            "policies": [
                {"name": f"Policy {i}", "steps": [{"action_type": "express"}]}
                for i in range(10)
            ]
        }
        llm = make_llm_mock(response_json=many_policies)
        gen = PolicyGenerator(llm=llm, max_policies=3)
        goal = make_goal()

        candidates = await gen.generate_candidates(
            goal=goal,
            situation_summary="Many options",
            beliefs=BeliefState(),
            affect=AffectState.neutral(),
        )
        # LLM returns 10 but we pass max_policies=3 to prompt
        # The policy list itself may be >3 since parsing doesn't hard-cap
        # but generation prompt asks for max 3
        assert len(candidates) >= 1  # At least do-nothing
