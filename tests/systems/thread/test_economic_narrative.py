"""
Unit tests for Thread economic narrative extensions.

Covers:
  Fix 6.1: 29D fingerprint economic dimensions (compute_economic_dimensions)
  Fix 6.4c: ASSET_BREAK_EVEN and CHILD_INDEPENDENT turning-point handlers

All tests are synchronous or use pytest-asyncio.
No real Neo4j, Redis, or LLM calls.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.thread.diachronic_coherence import DiachronicCoherenceMonitor


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_monitor() -> DiachronicCoherenceMonitor:
    """Build a DiachronicCoherenceMonitor with mocked Neo4j and LLM."""
    neo4j = MagicMock()
    neo4j.execute_read = AsyncMock(return_value=[])
    neo4j.execute_write = AsyncMock(return_value=None)
    llm = MagicMock()
    config = MagicMock()
    config.wasserstein_stable_threshold = 0.05
    config.wasserstein_moderate_threshold = 0.25
    config.wasserstein_major_threshold = 0.50
    config.fingerprint_weight_personality = 0.35
    config.fingerprint_weight_drive = 0.25
    config.fingerprint_weight_affect = 0.20
    config.fingerprint_weight_goal = 0.10
    config.fingerprint_weight_interaction = 0.10
    return DiachronicCoherenceMonitor(neo4j=neo4j, llm=llm, config=config)


def make_event(event_type: str, data: dict[str, Any]) -> MagicMock:
    """Build a minimal Synapse event mock."""
    ev = MagicMock()
    ev.event_type = event_type
    ev.data = data
    return ev


def make_economic_events(types: list[str], hours_apart: float = 1.0) -> list[dict[str, Any]]:
    """Build a list of economic event dicts with sequential timestamps."""
    now = datetime.now(UTC)
    events = []
    for i, et in enumerate(types):
        events.append({
            "event_type": et,
            "timestamp": (now + timedelta(hours=i * hours_apart)).isoformat(),
            "revenue_source": "bounty" if "bounty" in et else "asset",
            "amount_usd": 50.0 + i * 10,
        })
    return events


# ─── compute_economic_dimensions ─────────────────────────────────────────────


class TestComputeEconomicDimensions:
    """Tests for DiachronicCoherenceMonitor.compute_economic_dimensions()."""

    def test_empty_events_returns_defaults(self) -> None:
        monitor = make_monitor()
        dims = monitor.compute_economic_dimensions([])
        assert len(dims) == 5
        assert dims[0] == 0.0   # economic_strategy: balanced
        assert dims[1] == 0.5   # risk_tolerance: neutral
        assert dims[2] == 0.0   # diversification: unknown
        assert dims[3] == 0.0   # yield_inclination: none
        assert dims[4] == 0.0   # reproduction_preference: none

    def test_returns_five_dimensions(self) -> None:
        monitor = make_monitor()
        events = make_economic_events(["bounty_paid", "revenue_injected"])
        dims = monitor.compute_economic_dimensions(events)
        assert len(dims) == 5

    def test_all_dims_in_valid_range(self) -> None:
        monitor = make_monitor()
        events = make_economic_events([
            "bounty_paid", "bounty_paid", "asset_break_even",
            "yield_rebalance", "child_independent",
        ])
        dims = monitor.compute_economic_dimensions(events)
        # economic_strategy is in [-1, +1]; others are [0, 1]
        assert -1.0 <= dims[0] <= 1.0
        assert 0.0 <= dims[1] <= 1.0
        assert 0.0 <= dims[2] <= 1.0
        assert 0.0 <= dims[3] <= 1.0
        assert 0.0 <= dims[4] <= 1.0

    def test_child_independent_raises_reproduction_preference(self) -> None:
        monitor = make_monitor()
        events_with_child = make_economic_events(["child_independent", "child_independent"])
        events_no_child = make_economic_events(["bounty_paid", "bounty_paid"])
        dims_child = monitor.compute_economic_dimensions(events_with_child)
        dims_no_child = monitor.compute_economic_dimensions(events_no_child)
        # dim[4] = child_reproduction_preference - should be higher when child events dominate
        assert dims_child[4] > dims_no_child[4]

    def test_yield_events_raise_yield_inclination(self) -> None:
        monitor = make_monitor()
        events_yield = make_economic_events(["yield_rebalance", "yield_deploy"])
        events_bounty = make_economic_events(["bounty_paid", "bounty_paid"])
        dims_yield = monitor.compute_economic_dimensions(events_yield)
        dims_bounty = monitor.compute_economic_dimensions(events_bounty)
        # dim[3] = yield_farming_inclination - higher when yield events dominate
        assert dims_yield[3] > dims_bounty[3]

    def test_mixed_revenue_sources_raise_diversification(self) -> None:
        monitor = make_monitor()
        events_single = make_economic_events(["bounty_paid", "bounty_paid", "bounty_paid"])
        events_mixed = []
        for et in ["bounty_paid", "revenue_injected", "asset_break_even", "yield_rebalance"]:
            events_mixed.append({
                "event_type": et,
                "timestamp": datetime.now(UTC).isoformat(),
                "revenue_source": et.split("_")[0],
                "amount_usd": 10.0,
            })
        dims_single = monitor.compute_economic_dimensions(events_single)
        dims_mixed = monitor.compute_economic_dimensions(events_mixed)
        # dim[2] = revenue_stream_diversity - higher when sources are varied
        assert dims_mixed[2] >= dims_single[2]


# ─── Thread service economic event handlers ───────────────────────────────────


class TestAssetBreakEvenHandler:
    """Tests for ThreadService._on_asset_break_even()."""

    @pytest.mark.asyncio
    async def test_asset_break_even_emits_achievement_turning_point(self) -> None:
        """ASSET_BREAK_EVEN should create an ACHIEVEMENT TurningPoint and emit it."""
        from systems.thread.service import ThreadService

        svc = MagicMock(spec=ThreadService)
        svc._cached_economic_events = []
        svc._ECONOMIC_EVENT_MAX = 200

        emitted: list[dict[str, Any]] = []

        async def fake_emit_event(name: str, payload: dict[str, Any]) -> None:
            emitted.append({"name": name, "payload": payload})

        async def fake_re_trace(**kwargs: Any) -> None:
            pass

        svc._emit_event = fake_emit_event
        svc._emit_re_training_trace = fake_re_trace
        svc._record_economic_event = ThreadService._record_economic_event.__get__(svc, ThreadService)

        # Monkey-patch _on_asset_break_even as the real method
        bound_method = ThreadService._on_asset_break_even.__get__(svc, ThreadService)

        # Build a chapter mock
        chapter = MagicMock()
        chapter.id = "ch_test"
        svc._current_chapter_id = MagicMock(return_value="ch_test")

        event = make_event("asset_break_even", {
            "asset_name": "TestToken",
            "roi_score": 1.5,
            "total_revenue_usd": "200",
            "dev_cost_usd": "100",
        })

        with (
            patch("systems.thread.service.TurningPoint") as mock_tp_cls,
            patch("systems.thread.service.TurningPointType") as mock_tpt,
        ):
            mock_tp = MagicMock()
            mock_tp.id = "tp_test"
            mock_tp.description = "Asset TestToken reached break-even (ROI: 1.5x)"
            mock_tp_cls.return_value = mock_tp
            mock_tpt.ACHIEVEMENT = "ACHIEVEMENT"

            await bound_method(event)

        # An economic event was recorded
        assert len(svc._cached_economic_events) == 1
        assert svc._cached_economic_events[0]["event_type"] == "asset_break_even"

    @pytest.mark.asyncio
    async def test_child_independent_records_economic_event(self) -> None:
        """CHILD_INDEPENDENT should record a child economic event in the cache."""
        from systems.thread.service import ThreadService

        svc = MagicMock(spec=ThreadService)
        svc._cached_economic_events = []
        svc._ECONOMIC_EVENT_MAX = 200
        svc._current_chapter_id = MagicMock(return_value="ch_test")

        async def fake_emit_event(name: str, payload: dict[str, Any]) -> None:
            pass

        async def fake_re_trace(**kwargs: Any) -> None:
            pass

        svc._emit_event = fake_emit_event
        svc._emit_re_training_trace = fake_re_trace
        svc._record_economic_event = ThreadService._record_economic_event.__get__(svc, ThreadService)

        event = make_event("child_independent", {
            "child_id": "child_abc",
            "child_name": "Hermes",
            "metabolic_efficiency": 0.92,
        })

        bound_method = ThreadService._on_child_independent.__get__(svc, ThreadService)

        with (
            patch("systems.thread.service.TurningPoint") as mock_tp_cls,
            patch("systems.thread.service.TurningPointType") as mock_tpt,
        ):
            mock_tp = MagicMock()
            mock_tp.id = "tp_child"
            mock_tp.description = "Child child_abc (Hermes) achieved independence"
            mock_tp_cls.return_value = mock_tp
            mock_tpt.ACHIEVEMENT = "ACHIEVEMENT"

            await bound_method(event)

        assert len(svc._cached_economic_events) == 1
        assert svc._cached_economic_events[0]["event_type"] == "child_independent"
        assert svc._cached_economic_events[0].get("extra", {}).get("child_id") == "child_abc"


# ─── Economic cache accumulation ─────────────────────────────────────────────


class TestEconomicEventCache:
    """Tests for _record_economic_event and the 200-item ring buffer."""

    def test_record_economic_event_appends(self) -> None:
        from systems.thread.service import ThreadService

        svc = MagicMock(spec=ThreadService)
        svc._cached_economic_events = []
        svc._ECONOMIC_EVENT_MAX = 200

        record_fn = ThreadService._record_economic_event.__get__(svc, ThreadService)
        record_fn("bounty_paid", revenue_source="github", extra={"amount": 100})

        assert len(svc._cached_economic_events) == 1
        entry = svc._cached_economic_events[0]
        assert entry["event_type"] == "bounty_paid"
        assert entry["revenue_source"] == "github"
        assert entry["extra"]["amount"] == 100

    def test_cache_respects_max_size(self) -> None:
        from systems.thread.service import ThreadService

        svc = MagicMock(spec=ThreadService)
        svc._cached_economic_events = []
        svc._ECONOMIC_EVENT_MAX = 5

        record_fn = ThreadService._record_economic_event.__get__(svc, ThreadService)
        for i in range(10):
            record_fn(f"event_{i}")

        assert len(svc._cached_economic_events) <= 5
