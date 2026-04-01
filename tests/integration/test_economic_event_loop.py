"""
Integration tests for the economic event loop - DEAD-EVENTS-1 and RE-ECON-1 fixes.

Verifies that economic Synapse events (YIELD_DEPLOYMENT_RESULT, BOUNTY_REJECTED,
BOUNTY_PAID, REVENUE_INJECTED, METABOLIC_PRESSURE, YIELD_PERFORMANCE_REPORT) are
correctly consumed by the subscribing systems (Evo, Simula, Thread, Nexus) and
that the RE training exporter applies metabolic priority boosts during starvation.

No real Neo4j, Redis, or LLM calls - all external dependencies are mocked.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── Minimal EventBus stub ────────────────────────────────────────────────────


class _FakeEventBus:
    """Minimal in-process pub/sub for test isolation."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list] = {}
        self.emitted: list[Any] = []

    def subscribe(self, event_type: Any, handler: Any) -> None:
        key = str(event_type)
        self._subscribers.setdefault(key, []).append(handler)

    async def emit(self, event: Any) -> None:
        self.emitted.append(event)
        key = str(event.event_type)
        for handler in self._subscribers.get(key, []):
            await handler(event)


def _make_event(event_type_str: str, data: dict) -> Any:
    """Build a minimal SynapseEvent-like object for testing."""
    from systems.synapse.types import SynapseEvent, SynapseEventType

    return SynapseEvent(
        event_type=SynapseEventType(event_type_str),
        source_system="test",
        data=data,
    )


# ─── Fix 2.1: Evo - YIELD_DEPLOYMENT_RESULT + BOUNTY_REJECTED ────────────────


@pytest.mark.asyncio
async def test_evo_yield_deployment_result_success_emits_confirmed() -> None:
    """
    YIELD_DEPLOYMENT_RESULT (success=True) → EvoService emits EVO_HYPOTHESIS_CONFIRMED.
    """
    from systems.evo.service import EvoService

    bus = _FakeEventBus()
    svc = EvoService.__new__(EvoService)
    svc._initialized = True
    svc._logger = MagicMock()
    svc._event_bus = bus
    svc._bounty_outcomes: list[bool] = []

    # Wire the handler manually (mirrors register_on_synapse)
    from systems.synapse.types import SynapseEventType

    bus.subscribe(SynapseEventType.YIELD_DEPLOYMENT_RESULT, svc._on_yield_result)

    # Patch _scan_episode_online to be a no-op
    svc._scan_episode_online = AsyncMock()

    event = _make_event(
        "yield_deployment_result",
        {"request_id": "req-1", "success": True, "data": {"protocol": "aave-v3"}},
    )
    await bus.emit(event)

    # One EVO_HYPOTHESIS_CONFIRMED should have been emitted
    confirmed = [
        e for e in bus.emitted
        if str(e.event_type) == "evo_hypothesis_confirmed"
    ]
    assert len(confirmed) == 1
    assert confirmed[0].data["hypothesis_id"] == "evo.yield_protocol.aave-v3"
    assert confirmed[0].data["reward"] > 0


@pytest.mark.asyncio
async def test_evo_bounty_rejected_emits_refuted() -> None:
    """
    BOUNTY_REJECTED → EvoService emits EVO_HYPOTHESIS_REFUTED and appends False outcome.
    """
    from systems.evo.service import EvoService
    from systems.synapse.types import SynapseEventType

    bus = _FakeEventBus()
    svc = EvoService.__new__(EvoService)
    svc._initialized = True
    svc._logger = MagicMock()
    svc._event_bus = bus
    svc._bounty_outcomes: list[bool] = []
    svc._scan_episode_online = AsyncMock()
    svc._check_economic_parameter_adjustments = AsyncMock()

    bus.subscribe(SynapseEventType.BOUNTY_REJECTED, svc._on_bounty_rejected)

    event = _make_event(
        "bounty_rejected",
        {"bounty_id": "b-99", "reason": "equor_denied"},
    )
    await bus.emit(event)

    refuted = [
        e for e in bus.emitted
        if str(e.event_type) == "evo_hypothesis_refuted"
    ]
    assert len(refuted) == 1
    assert refuted[0].data["hypothesis_id"] == "evo.bounty_viability"
    assert refuted[0].data["reward"] < 0
    assert False in svc._bounty_outcomes


# ─── Fix 2.2: Simula - economic event subscriptions + _metabolic_boost ───────


@pytest.mark.asyncio
async def test_simula_metabolic_pressure_sets_boost() -> None:
    """
    METABOLIC_PRESSURE with starvation_level=critical → _metabolic_boost=2.0.
    """
    from systems.simula.service import SimulaService

    svc = SimulaService.__new__(SimulaService)
    svc._initialized = True
    svc._logger = MagicMock()
    svc._starvation_level = "nominal"
    svc._metabolic_boost = 1.0
    svc._proactive_scanner_task = None

    event = _make_event(
        "metabolic_pressure",
        {"starvation_level": "critical", "reason": "runway < 3d", "runway_days": "2.5"},
    )
    await svc._on_metabolic_pressure(event)

    assert svc._starvation_level == "critical"
    assert svc._metabolic_boost == 2.0


@pytest.mark.asyncio
async def test_simula_revenue_injected_resets_boost_when_nominal() -> None:
    """
    REVENUE_INJECTED when starvation is back to nominal → _metabolic_boost reset to 1.0.
    """
    from systems.simula.service import SimulaService

    svc = SimulaService.__new__(SimulaService)
    svc._initialized = True
    svc._logger = MagicMock()
    svc._starvation_level = "nominal"
    svc._metabolic_boost = 2.0  # Simulate prior crisis

    event = _make_event(
        "revenue_injected",
        {"amount_usd": 10.0, "source": "yield"},
    )
    await svc._on_revenue_change(event)

    assert svc._metabolic_boost == 1.0


# ─── Fix 2.5: RETrainingExporter - metabolic boost on economic examples ───────


@pytest.mark.asyncio
async def test_re_exporter_metabolic_pressure_sets_boost() -> None:
    """
    METABOLIC_PRESSURE event → RETrainingExporter._metabolic_boost updated correctly.
    """
    from core.re_training_exporter import RETrainingExporter

    bus = _FakeEventBus()
    exporter = RETrainingExporter(event_bus=bus)  # type: ignore[arg-type]

    assert exporter._metabolic_boost == 1.0

    event = _make_event(
        "metabolic_pressure",
        {"starvation_level": "austerity"},
    )
    await exporter._on_metabolic_pressure(event)

    assert exporter._starvation_level == "austerity"
    assert exporter._metabolic_boost == 1.5

    # CRITICAL should raise to 2.0
    event2 = _make_event("metabolic_pressure", {"starvation_level": "critical"})
    await exporter._on_metabolic_pressure(event2)
    assert exporter._metabolic_boost == 2.0

    # Nominal should reset
    event3 = _make_event("metabolic_pressure", {"starvation_level": "nominal"})
    await exporter._on_metabolic_pressure(event3)
    assert exporter._metabolic_boost == 1.0


@pytest.mark.asyncio
async def test_re_exporter_enrich_boosts_economic_confidence_during_starvation() -> None:
    """
    _enrich_batch during starvation (boost=2.0) raises confidence on economic examples
    while leaving non-economic examples unchanged.
    """
    from primitives.re_training import RETrainingDatapoint
    from core.re_training_exporter import RETrainingExporter

    bus = _FakeEventBus()
    exporter = RETrainingExporter(event_bus=bus)  # type: ignore[arg-type]
    exporter._metabolic_boost = 2.0  # Simulate CRITICAL starvation

    economic_dp = RETrainingDatapoint(
        source_system="oikos",
        example_type="economic.bounty_hunt",
        instruction="Should the organism accept this bounty?",
        input_context="Bounty ROI: 2.5x, capital available: $50",
        output_action="Accept bounty",
        outcome="success",
        reasoning_trace="Step 1: Check ROI. Step 2: Check capacity. Step 3: Accept.",
        confidence=0.5,
    )
    other_dp = RETrainingDatapoint(
        source_system="nova",
        example_type="goal_formulation",
        instruction="What should the organism do?",
        input_context="User asked a question",
        output_action="Respond to user",
        outcome="success",
        reasoning_trace="Step 1: Perceive. Step 2: Act.",
        confidence=0.5,
    )

    exporter._enrich_batch([economic_dp, other_dp])

    # Economic example should have boosted confidence (capped at 1.0)
    assert economic_dp.confidence == pytest.approx(min(1.0, 0.5 * 2.0), abs=1e-6)
    # Non-economic example should be unchanged
    assert other_dp.confidence == pytest.approx(0.5, abs=1e-6)


# ─── Fix 2.6: YIELD_PERFORMANCE_REPORT emitted after health check ────────────


@pytest.mark.asyncio
async def test_yield_performance_report_emitted_on_health_check() -> None:
    """
    YieldPositionTracker._check_health_once() → YIELD_PERFORMANCE_REPORT always emitted.
    """
    from systems.oikos.yield_strategy import YieldPositionTracker
    from decimal import Decimal

    bus = _FakeEventBus()
    tracker = YieldPositionTracker.__new__(YieldPositionTracker)
    tracker._event_bus = bus
    tracker._log = MagicMock()

    # Simulate a healthy position (no rebalance needed)
    mock_position = {
        "protocol": "morpho",
        "entry_apy": "0.05",
        "apy": "0.05",
    }

    with (
        patch.object(tracker, "load_position", new=AsyncMock(return_value=mock_position)),
        patch(
            "systems.oikos.yield_strategy._fetch_best_base_pool",
            new=AsyncMock(return_value=("morpho", Decimal("0.048"))),
        ),
    ):
        await tracker._check_health_once()

    reports = [
        e for e in bus.emitted
        if str(e.event_type) == "yield_performance_report"
    ]
    assert len(reports) == 1
    r = reports[0]
    assert r.data["protocol"] == "morpho"
    assert r.data["rebalance_needed"] is False
    assert "current_apy" in r.data
    assert "entry_apy" in r.data
    assert "relative_drop_pct" in r.data
    assert "timestamp" in r.data


@pytest.mark.asyncio
async def test_yield_performance_report_rebalance_needed_flag() -> None:
    """
    When APY drops >50%, YIELD_PERFORMANCE_REPORT.rebalance_needed=True.
    """
    from systems.oikos.yield_strategy import YieldPositionTracker
    from decimal import Decimal

    bus = _FakeEventBus()
    tracker = YieldPositionTracker.__new__(YieldPositionTracker)
    tracker._event_bus = bus
    tracker._log = MagicMock()

    mock_position = {
        "protocol": "aave-v3",
        "entry_apy": "0.10",  # 10% entry APY
        "apy": "0.10",
    }

    with (
        patch.object(tracker, "load_position", new=AsyncMock(return_value=mock_position)),
        patch(
            "systems.oikos.yield_strategy._fetch_best_base_pool",
            # 4% - a 60% relative drop from 10%
            new=AsyncMock(return_value=("aave-v3", Decimal("0.04"))),
        ),
    ):
        await tracker._check_health_once()

    reports = [
        e for e in bus.emitted
        if str(e.event_type) == "yield_performance_report"
    ]
    assert len(reports) == 1
    assert reports[0].data["rebalance_needed"] is True
