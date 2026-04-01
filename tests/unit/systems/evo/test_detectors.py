"""
Unit tests for Evo pattern detectors.

Tests the four online pattern detectors independently.
All tests are synchronous via pytest.mark.asyncio where needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from primitives.affect import AffectState
from primitives.memory_trace import Episode
from systems.evo.detectors import (
    AffectPatternDetector,
    CooccurrenceDetector,
    SequenceDetector,
    TemporalDetector,
    _classify_source,
    _is_successful_action_outcome,
    build_default_detectors,
    hash_sequence,
)
from systems.evo.types import PatternContext, PatternType

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_episode(
    *,
    source: str = "text_chat",
    affect_valence: float = 0.3,
    affect_arousal: float = 0.4,
    salience_composite: float = 0.6,
    event_time: datetime | None = None,
) -> Episode:
    return Episode(
        source=source,
        modality="text",
        raw_content="test content",
        summary="test",
        salience_composite=salience_composite,
        affect_valence=affect_valence,
        affect_arousal=affect_arousal,
        event_time=event_time or datetime(2026, 2, 22, 14, 30, tzinfo=UTC),
    )


def make_context() -> PatternContext:
    return PatternContext()


# ─── Tests: CooccurrenceDetector ──────────────────────────────────────────────


class TestCooccurrenceDetector:
    @pytest.mark.asyncio
    async def test_no_entities_returns_empty(self):
        detector = CooccurrenceDetector()
        context = make_context()
        context.recent_entity_ids = []
        result = await detector.scan(make_episode(), context)
        assert result == []

    @pytest.mark.asyncio
    async def test_single_entity_returns_empty(self):
        detector = CooccurrenceDetector()
        context = make_context()
        context.recent_entity_ids = ["entity_a"]
        result = await detector.scan(make_episode(), context)
        assert result == []

    @pytest.mark.asyncio
    async def test_emits_at_threshold(self):
        detector = CooccurrenceDetector()
        detector.min_occurrences = 3
        context = make_context()
        context.recent_entity_ids = ["entity_a", "entity_b"]

        # Below threshold
        for _ in range(2):
            result = await detector.scan(make_episode(), context)
            assert result == []

        # At threshold - should emit
        result = await detector.scan(make_episode(), context)
        assert len(result) == 1
        assert result[0].type == PatternType.COOCCURRENCE
        assert set(result[0].elements) == {"entity_a", "entity_b"}
        assert result[0].count == 3

    @pytest.mark.asyncio
    async def test_emits_only_at_threshold_not_after(self):
        """Should emit exactly once at threshold, not on every subsequent scan."""
        detector = CooccurrenceDetector()
        detector.min_occurrences = 2
        context = make_context()
        context.recent_entity_ids = ["a", "b"]

        await detector.scan(make_episode(), context)  # count=1
        result_at_threshold = await detector.scan(make_episode(), context)  # count=2 → emit
        result_after = await detector.scan(make_episode(), context)  # count=3 → no emit

        assert len(result_at_threshold) == 1
        assert result_after == []

    @pytest.mark.asyncio
    async def test_pair_order_is_canonical(self):
        """Pair (b, a) and (a, b) should map to the same counter key."""
        detector = CooccurrenceDetector()
        detector.min_occurrences = 2
        context = make_context()

        context.recent_entity_ids = ["b", "a"]
        await detector.scan(make_episode(), context)

        context.recent_entity_ids = ["a", "b"]
        result = await detector.scan(make_episode(), context)

        assert len(result) == 1  # Same pair, threshold reached


# ─── Tests: SequenceDetector ──────────────────────────────────────────────────


class TestSequenceDetector:
    @pytest.mark.asyncio
    async def test_ignores_non_action_episodes(self):
        detector = SequenceDetector()
        context = make_context()
        episode = make_episode(source="text_chat", salience_composite=0.8)
        result = await detector.scan(episode, context)
        assert result == []

    @pytest.mark.asyncio
    async def test_ignores_failed_action_episodes(self):
        detector = SequenceDetector()
        context = make_context()
        # Failed action: negative valence
        episode = make_episode(source="axon:respond_text", affect_valence=-0.5, salience_composite=0.1)
        result = await detector.scan(episode, context)
        assert result == []

    @pytest.mark.asyncio
    async def test_accumulates_successful_sequences(self):
        detector = SequenceDetector()
        detector.min_occurrences = 3
        context = make_context()
        episode = make_episode(source="axon:respond_text", affect_valence=0.3, salience_composite=0.7)

        await detector.scan(episode, context)
        await detector.scan(episode, context)
        result = await detector.scan(episode, context)

        assert len(result) == 1
        assert result[0].type == PatternType.ACTION_SEQUENCE
        assert result[0].count == 3

    @pytest.mark.asyncio
    async def test_different_sequences_are_independent(self):
        detector = SequenceDetector()
        detector.min_occurrences = 2
        context = make_context()

        ep_a = make_episode(source="axon:respond_text", affect_valence=0.3, salience_composite=0.7)
        ep_b = make_episode(source="axon:store_insight", affect_valence=0.3, salience_composite=0.7)

        await detector.scan(ep_a, context)
        await detector.scan(ep_b, context)

        # Neither has reached threshold of 2 yet
        assert len(context.sequence_counts) == 2

    def test_hash_sequence_is_deterministic(self):
        seq = ["respond_text", "store_insight"]
        assert hash_sequence(seq) == hash_sequence(seq)

    def test_hash_sequence_differs_for_different_sequences(self):
        assert hash_sequence(["a", "b"]) != hash_sequence(["b", "a"])


# ─── Tests: TemporalDetector ──────────────────────────────────────────────────


class TestTemporalDetector:
    @pytest.mark.asyncio
    async def test_bins_by_hour(self):
        detector = TemporalDetector()
        context = make_context()

        ep = make_episode(
            source="text_chat",
            event_time=datetime(2026, 2, 22, 9, 0, tzinfo=UTC),  # 9am
        )
        await detector.scan(ep, context)
        assert context.temporal_bins["social_text::h9"] == 1

    @pytest.mark.asyncio
    async def test_bins_by_weekday(self):
        detector = TemporalDetector()
        context = make_context()

        ep = make_episode(
            source="axon:respond",
            event_time=datetime(2026, 2, 23, 10, 0, tzinfo=UTC),  # Monday = 0
        )
        await detector.scan(ep, context)
        weekday = ep.event_time.weekday()
        assert context.temporal_bins[f"action_outcome::d{weekday}"] == 1

    @pytest.mark.asyncio
    async def test_emits_when_above_baseline(self):
        detector = TemporalDetector()
        detector.min_occurrences = 2
        context = make_context()
        # Set a low baseline so 2 occurrences exceeds 2x baseline
        context.temporal_baselines["social_text"] = 0.5

        ep = make_episode(
            source="text_chat",
            event_time=datetime(2026, 2, 22, 9, 0, tzinfo=UTC),
        )
        await detector.scan(ep, context)
        result = await detector.scan(ep, context)

        # count=2, baseline=0.5 → 2/0.5=4.0 > 2.0 threshold
        assert any(r.type == PatternType.TEMPORAL for r in result)


# ─── Tests: AffectPatternDetector ─────────────────────────────────────────────


class TestAffectPatternDetector:
    @pytest.mark.asyncio
    async def test_ignores_missing_affect_states(self):
        detector = AffectPatternDetector()
        context = make_context()
        # No previous or current affect set
        result = await detector.scan(make_episode(), context)
        assert result == []

    @pytest.mark.asyncio
    async def test_ignores_small_affect_shifts(self):
        detector = AffectPatternDetector()
        context = make_context()
        context.previous_affect = AffectState(valence=0.0, arousal=0.2)
        context.current_affect = AffectState(valence=0.05, arousal=0.22)  # < 0.1 shift

        result = await detector.scan(make_episode(source="text_chat"), context)
        assert result == []

    @pytest.mark.asyncio
    async def test_accumulates_significant_shifts(self):
        detector = AffectPatternDetector()
        detector.min_occurrences = 3
        context = make_context()

        for _i in range(3):
            context.previous_affect = AffectState(valence=0.0, arousal=0.2)
            context.current_affect = AffectState(valence=0.4, arousal=0.2)  # +0.4 valence
            result = await detector.scan(make_episode(source="text_chat"), context)

        assert len(result) == 1
        assert result[0].type == PatternType.AFFECT_PATTERN
        assert result[0].metadata["mean_valence_delta"] == pytest.approx(0.4, abs=0.05)

    @pytest.mark.asyncio
    async def test_different_stimuli_tracked_separately(self):
        detector = AffectPatternDetector()
        context = make_context()
        context.previous_affect = AffectState(valence=0.0, arousal=0.2)
        context.current_affect = AffectState(valence=0.5, arousal=0.2)

        ep_text = make_episode(source="text_chat")
        ep_action = make_episode(source="axon:respond")

        await detector.scan(ep_text, context)
        await detector.scan(ep_action, context)

        assert len(context.affect_responses["social_text"]) == 1
        assert len(context.affect_responses["action_outcome"]) == 1


# ─── Tests: Helpers ───────────────────────────────────────────────────────────


class TestHelpers:
    def test_classify_source_text_chat(self):
        assert _classify_source("text_chat:main") == "social_text"

    def test_classify_source_axon(self):
        assert _classify_source("axon:respond_text") == "action_outcome"

    def test_classify_source_empty(self):
        assert _classify_source("") == "general"

    def test_classify_source_federation(self):
        assert _classify_source("federation:instance_b") == "federation"

    def test_is_successful_action_outcome_requires_axon_prefix(self):
        ep_success = make_episode(source="axon:respond", affect_valence=0.3, salience_composite=0.7)
        ep_text = make_episode(source="text_chat", affect_valence=0.3, salience_composite=0.7)
        assert _is_successful_action_outcome(ep_success) is True
        assert _is_successful_action_outcome(ep_text) is False


# ─── Tests: Default Detector Set ──────────────────────────────────────────────


def test_build_default_detectors_returns_four():
    detectors = build_default_detectors()
    assert len(detectors) == 4
    names = {d.name for d in detectors}
    assert names == {"cooccurrence", "action_sequence", "temporal", "affect_pattern"}
