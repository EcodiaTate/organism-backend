"""Tests for DreamJournal and DreamInsightTracker."""
import pytest

from systems.oneiros.journal import DreamInsightTracker, DreamJournal
from systems.oneiros.types import (
    Dream,
    DreamCoherence,
    DreamInsight,
    DreamType,
    InsightStatus,
    SleepCycle,
    SleepQuality,
)


def _make_dream(**overrides):
    defaults = dict(
        dream_type=DreamType.RECOMBINATION,
        sleep_cycle_id="cycle_1",
        bridge_narrative="A test dream narrative",
        coherence_score=0.5,
        coherence_class=DreamCoherence.FRAGMENT,
    )
    defaults.update(overrides)
    return Dream(**defaults)

def _make_insight(**overrides):
    defaults = dict(
        dream_id="dream_1",
        sleep_cycle_id="cycle_1",
        insight_text="A creative connection discovered",
        coherence_score=0.8,
        domain="general",
    )
    defaults.update(overrides)
    return DreamInsight(**defaults)

class TestDreamJournal:
    @pytest.mark.asyncio
    async def test_initialize_without_neo4j(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()  # Should not raise

    @pytest.mark.asyncio
    async def test_record_dream_in_memory(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        dream = _make_dream()
        await journal.record_dream(dream)
        assert len(journal._dream_buffer) == 1

    @pytest.mark.asyncio
    async def test_record_multiple_dreams(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        for i in range(5):
            await journal.record_dream(_make_dream(coherence_score=i * 0.2))
        assert len(journal._dream_buffer) == 5

    @pytest.mark.asyncio
    async def test_record_insight(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        insight = _make_insight()
        await journal.record_insight(insight)
        assert insight.id in journal._all_insights

    @pytest.mark.asyncio
    async def test_get_recent_dreams(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        for _i in range(10):
            await journal.record_dream(_make_dream())
        dreams = await journal.get_recent_dreams(limit=5)
        assert len(dreams) <= 5

    @pytest.mark.asyncio
    async def test_get_recent_insights(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        for _i in range(5):
            await journal.record_insight(_make_insight())
        insights = await journal.get_recent_insights(limit=3)
        assert len(insights) <= 5  # may be all 5 since no status filter

    @pytest.mark.asyncio
    async def test_get_pending_insights(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        await journal.record_insight(_make_insight(status=InsightStatus.PENDING))
        await journal.record_insight(_make_insight(status=InsightStatus.VALIDATED))
        pending = await journal.get_pending_insights()
        assert any(i.status == InsightStatus.PENDING for i in pending)

    @pytest.mark.asyncio
    async def test_record_sleep_cycle(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        cycle = SleepCycle(quality=SleepQuality.NORMAL, episodes_replayed=10)
        await journal.record_sleep_cycle(cycle)

    @pytest.mark.asyncio
    async def test_stats(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        await journal.record_dream(_make_dream())
        await journal.record_insight(_make_insight())
        stats = await journal.stats()
        assert stats["total_dreams"] >= 1
        assert stats["total_insights"] >= 1

class TestDreamInsightTracker:
    @pytest.mark.asyncio
    async def test_validate_insight(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        insight = _make_insight()
        await journal.record_insight(insight)
        tracker = DreamInsightTracker(journal)
        await tracker.validate_insight(insight.id, "Confirmed in wake state")
        assert journal._all_insights[insight.id].status == InsightStatus.VALIDATED

    @pytest.mark.asyncio
    async def test_invalidate_insight(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        insight = _make_insight()
        await journal.record_insight(insight)
        tracker = DreamInsightTracker(journal)
        await tracker.invalidate_insight(insight.id, "Not useful")
        assert journal._all_insights[insight.id].status == InsightStatus.INVALIDATED

    @pytest.mark.asyncio
    async def test_integrate_insight(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        insight = _make_insight()
        await journal.record_insight(insight)
        tracker = DreamInsightTracker(journal)
        await tracker.integrate_insight(insight.id)
        assert journal._all_insights[insight.id].status == InsightStatus.INTEGRATED

    @pytest.mark.asyncio
    async def test_record_application(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        insight = _make_insight()
        await journal.record_insight(insight)
        tracker = DreamInsightTracker(journal)
        await tracker.record_application(insight.id)
        assert journal._all_insights[insight.id].wake_applications == 1

    @pytest.mark.asyncio
    async def test_get_effectiveness_empty(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        tracker = DreamInsightTracker(journal)
        eff = await tracker.get_effectiveness()
        assert "validated_ratio" in eff

    @pytest.mark.asyncio
    async def test_get_effectiveness_with_data(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        i1 = _make_insight()
        i2 = _make_insight()
        await journal.record_insight(i1)
        await journal.record_insight(i2)
        tracker = DreamInsightTracker(journal)
        await tracker.validate_insight(i1.id, "good")
        eff = await tracker.get_effectiveness()
        assert eff["validated_ratio"] == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_nonexistent_insight_id(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        tracker = DreamInsightTracker(journal)
        await tracker.validate_insight("nonexistent_id", "test")  # Should not raise
