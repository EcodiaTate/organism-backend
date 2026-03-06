"""
Unit tests for Evo HypothesisEngine.

Tests hypothesis lifecycle: generation, evidence accumulation,
status transitions, integration, and archival.
"""

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from primitives.common import utc_now
from primitives.memory_trace import Episode
from systems.evo.hypothesis import HypothesisEngine, _build_hypothesis, _parse_json_safe
from systems.evo.types import (
    EvidenceDirection,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
    MutationType,
    PatternCandidate,
    PatternType,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_llm(generate_response: str = "", evaluate_response: str = "") -> MagicMock:
    """Create a mock LLMProvider."""
    llm = MagicMock()
    gen_result = MagicMock()
    gen_result.text = generate_response
    llm.generate = AsyncMock(return_value=gen_result)

    eval_result = MagicMock()
    eval_result.text = evaluate_response
    llm.evaluate = AsyncMock(return_value=eval_result)
    return llm


def make_engine(llm=None) -> HypothesisEngine:
    return HypothesisEngine(
        llm=llm or make_llm(),
        instance_name="TestEOS",
        memory=None,
    )


def make_hypothesis(
    *,
    category: HypothesisCategory = HypothesisCategory.WORLD_MODEL,
    status: HypothesisStatus = HypothesisStatus.TESTING,
    evidence_score: float = 0.0,
    supporting_count: int = 0,
    created_offset_hours: float = 0.0,
) -> Hypothesis:
    h = Hypothesis(
        category=category,
        statement="Test hypothesis about something observable",
        formal_test="If X then Y should occur within Z",
        complexity_penalty=0.1,
        status=status,
        evidence_score=evidence_score,
    )
    h.supporting_episodes = [f"ep_{i}" for i in range(supporting_count)]
    if created_offset_hours != 0.0:
        h.created_at = utc_now() - timedelta(hours=created_offset_hours)
    return h


def make_episode(
    *,
    source: str = "text_chat",
    affect_valence: float = 0.0,
    salience_composite: float = 0.5,
    raw_content: str = "episode content",
) -> Episode:
    return Episode(
        source=source,
        raw_content=raw_content,
        summary=raw_content[:100],
        affect_valence=affect_valence,
        salience_composite=salience_composite,
    )


def make_pattern(count: int = 5) -> PatternCandidate:
    return PatternCandidate(
        type=PatternType.COOCCURRENCE,
        elements=["entity_a", "entity_b"],
        count=count,
        confidence=0.6,
    )


# ─── Tests: HypothesisEngine.generate_hypotheses ──────────────────────────────


class TestHypothesisGeneration:
    @pytest.mark.asyncio
    async def test_empty_patterns_returns_empty(self):
        engine = make_engine()
        result = await engine.generate_hypotheses(patterns=[])
        assert result == []

    @pytest.mark.asyncio
    async def test_parses_valid_llm_response(self):
        response = json.dumps({
            "hypotheses": [{
                "category": "world_model",
                "statement": "Entities A and B are correlated",
                "formal_test": "If A appears, B will appear within 3 episodes",
                "complexity": "low",
                "proposed_mutation": {
                    "type": "schema_addition",
                    "target": "CORRELATED_WITH",
                    "value": 0.0,
                    "description": "Add CORRELATED_WITH relation type",
                },
            }]
        })
        engine = make_engine(llm=make_llm(generate_response=response))
        result = await engine.generate_hypotheses(patterns=[make_pattern()])
        assert len(result) == 1
        assert result[0].category == HypothesisCategory.WORLD_MODEL
        assert "correlated" in result[0].statement.lower()
        assert result[0].proposed_mutation is not None

    @pytest.mark.asyncio
    async def test_handles_invalid_json_gracefully(self):
        engine = make_engine(llm=make_llm(generate_response="not json"))
        result = await engine.generate_hypotheses(patterns=[make_pattern()])
        assert result == []

    @pytest.mark.asyncio
    async def test_respects_max_active_limit(self):
        engine = make_engine()
        # Fill the registry to capacity
        for _i in range(50):
            h = make_hypothesis()
            engine._active[h.id] = h

        result = await engine.generate_hypotheses(patterns=[make_pattern()])
        assert result == []

    @pytest.mark.asyncio
    async def test_caps_at_three_per_batch(self):
        many_hypotheses = json.dumps({
            "hypotheses": [
                {
                    "category": "world_model",
                    "statement": f"Hypothesis {i}",
                    "formal_test": f"Test {i}",
                    "complexity": "low",
                    "proposed_mutation": None,
                }
                for i in range(10)
            ]
        })
        engine = make_engine(llm=make_llm(generate_response=many_hypotheses))
        result = await engine.generate_hypotheses(patterns=[make_pattern()])
        assert len(result) <= 3


# ─── Tests: HypothesisEngine.evaluate_evidence ────────────────────────────────


class TestEvidenceEvaluation:
    @pytest.mark.asyncio
    async def test_supports_increases_score(self):
        eval_response = json.dumps({
            "direction": "supports",
            "strength": 0.8,
            "reasoning": "Clear support",
        })
        engine = make_engine(llm=make_llm(evaluate_response=eval_response))
        h = make_hypothesis(evidence_score=0.0)
        engine._active[h.id] = h

        result = await engine.evaluate_evidence(h, make_episode())

        assert result.direction == EvidenceDirection.SUPPORTS
        assert result.strength == pytest.approx(0.8)
        assert h.evidence_score > 0.0
        assert len(h.supporting_episodes) == 1

    @pytest.mark.asyncio
    async def test_contradicts_decreases_score(self):
        eval_response = json.dumps({
            "direction": "contradicts",
            "strength": 0.5,
            "reasoning": "Clear contradiction",
        })
        engine = make_engine(llm=make_llm(evaluate_response=eval_response))
        h = make_hypothesis(evidence_score=2.0)
        engine._active[h.id] = h

        result = await engine.evaluate_evidence(h, make_episode())

        assert result.direction == EvidenceDirection.CONTRADICTS
        assert h.evidence_score < 2.0
        assert len(h.contradicting_episodes) == 1

    @pytest.mark.asyncio
    async def test_neutral_does_not_change_score(self):
        eval_response = json.dumps({
            "direction": "neutral",
            "strength": 0.3,
            "reasoning": "Unrelated",
        })
        engine = make_engine(llm=make_llm(evaluate_response=eval_response))
        h = make_hypothesis(evidence_score=1.5)
        engine._active[h.id] = h
        original_score = h.evidence_score

        result = await engine.evaluate_evidence(h, make_episode())

        assert result.direction == EvidenceDirection.NEUTRAL
        assert h.evidence_score == pytest.approx(original_score)

    @pytest.mark.asyncio
    async def test_transitions_to_supported(self):
        """Hypothesis transitions to SUPPORTED when score > 3 and episodes >= 10."""
        eval_response = json.dumps({
            "direction": "supports",
            "strength": 1.0,
            "reasoning": "Strong support",
        })
        engine = make_engine(llm=make_llm(evaluate_response=eval_response))
        h = make_hypothesis(evidence_score=2.5, supporting_count=9)
        engine._active[h.id] = h

        await engine.evaluate_evidence(h, make_episode())

        assert h.status == HypothesisStatus.SUPPORTED
        assert h.evidence_score > 3.0

    @pytest.mark.asyncio
    async def test_transitions_to_refuted(self):
        """Hypothesis transitions to REFUTED when score < -2."""
        eval_response = json.dumps({
            "direction": "contradicts",
            "strength": 1.0,
            "reasoning": "Strong contradiction",
        })
        engine = make_engine(llm=make_llm(evaluate_response=eval_response))
        h = make_hypothesis(evidence_score=-1.5)
        engine._active[h.id] = h

        await engine.evaluate_evidence(h, make_episode())

        assert h.status == HypothesisStatus.REFUTED

    @pytest.mark.asyncio
    async def test_complexity_penalty_applied(self):
        """Higher complexity penalty reduces evidence score boost."""
        eval_response = json.dumps({
            "direction": "supports",
            "strength": 1.0,
            "reasoning": "Support",
        })
        engine_low = make_engine(llm=make_llm(evaluate_response=eval_response))
        engine_high = make_engine(llm=make_llm(evaluate_response=eval_response))

        h_low = make_hypothesis(evidence_score=0.0)
        h_low.complexity_penalty = 0.05
        engine_low._active[h_low.id] = h_low

        h_high = make_hypothesis(evidence_score=0.0)
        h_high.complexity_penalty = 0.3
        engine_high._active[h_high.id] = h_high

        await engine_low.evaluate_evidence(h_low, make_episode())
        await engine_high.evaluate_evidence(h_high, make_episode())

        assert h_low.evidence_score > h_high.evidence_score

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        llm = make_llm()
        llm.evaluate = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        engine = make_engine(llm=llm)
        h = make_hypothesis()
        engine._active[h.id] = h

        result = await engine.evaluate_evidence(h, make_episode())
        assert result.direction == EvidenceDirection.NEUTRAL
        assert result.strength == 0.0


# ─── Tests: Integration & Archival ────────────────────────────────────────────


class TestHypothesisLifecycle:
    @pytest.mark.asyncio
    async def test_integrate_requires_supported_status(self):
        engine = make_engine()
        h = make_hypothesis(status=HypothesisStatus.TESTING)
        result = await engine.integrate_hypothesis(h)
        assert result is False

    @pytest.mark.asyncio
    async def test_integrate_requires_min_age(self):
        engine = make_engine()
        h = make_hypothesis(status=HypothesisStatus.SUPPORTED, created_offset_hours=1)
        result = await engine.integrate_hypothesis(h)
        assert result is False  # Only 1 hour old, needs 24

    @pytest.mark.asyncio
    async def test_integrate_succeeds_when_old_enough(self):
        engine = make_engine()
        h = make_hypothesis(
            status=HypothesisStatus.SUPPORTED,
            created_offset_hours=25,
        )
        engine._active[h.id] = h
        result = await engine.integrate_hypothesis(h)
        assert result is True
        assert h.status == HypothesisStatus.INTEGRATED
        assert h.id not in engine._active

    @pytest.mark.asyncio
    async def test_archive_marks_and_removes(self):
        engine = make_engine()
        h = make_hypothesis()
        engine._active[h.id] = h

        await engine.archive_hypothesis(h, reason="test")

        assert h.status == HypothesisStatus.ARCHIVED
        assert h.id not in engine._active

    def test_is_stale_after_max_age(self):
        engine = make_engine()
        h = make_hypothesis(status=HypothesisStatus.TESTING)
        h.last_evidence_at = utc_now() - timedelta(days=8)
        assert engine.is_stale(h, max_age_days=7) is True

    def test_is_not_stale_within_age(self):
        engine = make_engine()
        h = make_hypothesis(status=HypothesisStatus.TESTING)
        h.last_evidence_at = utc_now() - timedelta(days=3)
        assert engine.is_stale(h, max_age_days=7) is False

    def test_supported_hypothesis_not_stale(self):
        """Supported hypotheses are not considered stale (they're awaiting integration)."""
        engine = make_engine()
        h = make_hypothesis(status=HypothesisStatus.SUPPORTED)
        h.last_evidence_at = utc_now() - timedelta(days=30)
        assert engine.is_stale(h) is False


# ─── Tests: Helpers ───────────────────────────────────────────────────────────


class TestHelpers:
    def test_parse_json_safe_valid(self):
        result = _parse_json_safe('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_safe_with_markdown_fences(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_safe(text)
        assert result == {"key": "value"}

    def test_parse_json_safe_invalid_returns_empty(self):
        result = _parse_json_safe("not json at all")
        assert result == {}

    def test_build_hypothesis_valid(self):
        item = {
            "category": "parameter",
            "statement": "The risk salience head should have higher weight",
            "formal_test": "If risk.weight > 0.3, outcome avoidance rate improves",
            "complexity": "low",
            "proposed_mutation": {
                "type": "parameter_adjustment",
                "target": "atune.head.risk.weight",
                "value": 0.01,
                "description": "Increase risk head weight",
            },
        }
        h = _build_hypothesis(item)
        assert h.category == HypothesisCategory.PARAMETER
        assert h.proposed_mutation is not None
        assert h.proposed_mutation.type == MutationType.PARAMETER_ADJUSTMENT
        assert h.complexity_penalty == pytest.approx(0.05)

    def test_build_hypothesis_unknown_category_defaults(self):
        item = {
            "category": "not_real",
            "statement": "Something",
            "formal_test": "Something else",
            "complexity": "medium",
            "proposed_mutation": None,
        }
        h = _build_hypothesis(item)
        assert h.category == HypothesisCategory.WORLD_MODEL

    def test_stats_accumulate(self):
        engine = make_engine()
        assert engine.stats["proposed"] == 0
        engine._total_proposed = 3
        engine._total_supported = 2
        engine._total_refuted = 1
        engine._total_integrated = 1
        stats = engine.stats
        assert stats["proposed"] == 3
        assert stats["supported"] == 2
