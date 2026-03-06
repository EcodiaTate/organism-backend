"""Unit tests for SilenceEngine."""

from __future__ import annotations

import pytest

from systems.voxis.silence import SilenceEngine
from systems.voxis.types import ExpressionTrigger, SilenceContext


def make_context(
    trigger: ExpressionTrigger,
    humans_conversing: bool = False,
    elapsed_minutes: float = 999.0,
    min_interval: float = 1.0,
    insight_value: float = 0.8,
    urgency: float = 0.5,
) -> SilenceContext:
    return SilenceContext(
        trigger=trigger,
        humans_actively_conversing=humans_conversing,
        minutes_since_last_expression=elapsed_minutes,
        min_expression_interval=min_interval,
        insight_value=insight_value,
        urgency=urgency,
    )


@pytest.fixture
def engine() -> SilenceEngine:
    return SilenceEngine(min_expression_interval_minutes=1.0)


class TestMandatoryTriggers:
    def test_direct_address_always_speaks(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.ATUNE_DIRECT_ADDRESS, humans_conversing=True, elapsed_minutes=0.1)
        result = engine.evaluate(ctx)
        assert result.speak is True

    def test_distress_always_speaks(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.ATUNE_DISTRESS, humans_conversing=True, elapsed_minutes=0.0)
        result = engine.evaluate(ctx)
        assert result.speak is True

    def test_warning_always_speaks(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.NOVA_WARN, humans_conversing=True, elapsed_minutes=0.0)
        result = engine.evaluate(ctx)
        assert result.speak is True


class TestNovaDeliberateTriggers:
    def test_nova_respond_always_speaks(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.NOVA_RESPOND)
        result = engine.evaluate(ctx)
        assert result.speak is True

    def test_nova_celebrate_always_speaks(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.NOVA_CELEBRATE)
        result = engine.evaluate(ctx)
        assert result.speak is True

    def test_nova_inform_suppressed_when_humans_conversing(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.NOVA_INFORM, humans_conversing=True)
        result = engine.evaluate(ctx)
        assert result.speak is False
        assert result.queue is True

    def test_nova_inform_suppressed_within_rate_limit(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.NOVA_INFORM, elapsed_minutes=0.3, min_interval=1.0)
        result = engine.evaluate(ctx)
        assert result.speak is False
        assert result.queue is True

    def test_nova_inform_speaks_when_clear(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.NOVA_INFORM, elapsed_minutes=5.0)
        result = engine.evaluate(ctx)
        assert result.speak is True


class TestAmbientTriggers:
    def test_ambient_insight_suppressed_when_humans_conversing(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.AMBIENT_INSIGHT, humans_conversing=True)
        result = engine.evaluate(ctx)
        assert result.speak is False

    def test_ambient_insight_suppressed_below_value_threshold(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.AMBIENT_INSIGHT, insight_value=0.3)
        result = engine.evaluate(ctx)
        assert result.speak is False

    def test_ambient_insight_suppressed_within_rate_limit(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.AMBIENT_INSIGHT, elapsed_minutes=0.5, insight_value=0.9)
        result = engine.evaluate(ctx)
        assert result.speak is False

    def test_ambient_insight_speaks_when_all_conditions_met(self, engine: SilenceEngine) -> None:
        ctx = make_context(
            ExpressionTrigger.AMBIENT_INSIGHT,
            insight_value=0.85,
            elapsed_minutes=10.0,
            humans_conversing=False,
        )
        result = engine.evaluate(ctx)
        assert result.speak is True

    def test_ambient_insight_below_threshold_not_queued(self, engine: SilenceEngine) -> None:
        ctx = make_context(ExpressionTrigger.AMBIENT_INSIGHT, insight_value=0.2, elapsed_minutes=10.0)
        result = engine.evaluate(ctx)
        assert result.speak is False
        assert result.queue is False


class TestStateTracking:
    def test_record_expression_updates_timer(self, engine: SilenceEngine) -> None:
        assert engine.minutes_since_last_expression > 100
        engine.record_expression()
        assert engine.minutes_since_last_expression < 1.0
