"""
Integration tests for economic federation gaps (2026-03-08).

Covers:
  ONEIROS-ECON-1  Dream broadcast: Oneiros emits ONEIROS_ECONOMIC_INSIGHT after
                  Monte Carlo dreaming when ruin_probability > 0.2.
  NEXUS-ECON-1    Convergence reward: Oikos credits reserves on
                  NEXUS_CONVERGENCE_METABOLIC_SIGNAL.
  NEXUS-ECON-2/3  Economic schema sharing: ShareableWorldModelFragment
                  carries domain_hints + economic_context.
  NEXUS-ECON-4    Economic divergence: InstanceDivergenceProfile.economic_divergence
                  populated from strategy_revenue_rates variance.
  NOVA-ECON-INS   Nova subscribes to ONEIROS_ECONOMIC_INSIGHT and updates
                  economic_ruin_risk belief.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.nexus.divergence import compute_economic_divergence
from systems.nexus.speciation import _detect_economic_domain_hints
from systems.nexus.types import InstanceDivergenceProfile, ShareableWorldModelFragment
from systems.synapse.types import SynapseEventType


# ─── Helpers ──────────────────────────────────────────────────────


def make_event(event_type: SynapseEventType, data: dict[str, Any]) -> MagicMock:
    """Create a lightweight mock SynapseEvent."""
    event = MagicMock()
    event.event_type = event_type
    event.data = data
    return event


# ─── ONEIROS-ECON-1: Dream Broadcast ─────────────────────────────


class TestOneirosEconomicDreamBroadcast:
    """
    Oneiros._run_economic_dreaming() must emit ONEIROS_ECONOMIC_INSIGHT when
    ruin_probability > 0.2.
    """

    @pytest.mark.asyncio
    async def test_broadcast_emitted_when_ruin_above_threshold(self) -> None:
        """Insight event fires when ruin_probability > 0.2."""
        from systems.oneiros.service import OneirosService

        service = OneirosService.__new__(OneirosService)
        service._logger = MagicMock()
        service._logger.info = MagicMock()
        service._logger.error = MagicMock()
        service._oikos = MagicMock()
        service._synapse = MagicMock()
        service._journal = None

        # Capture emitted events
        emitted: list[dict[str, Any]] = []

        async def fake_emit(event_type: SynapseEventType, data: dict) -> None:
            emitted.append({"event_type": event_type, "data": data})

        service._emit_event = fake_emit  # type: ignore[assignment]

        # Build a dream result with ruin_probability = 0.45 (above threshold)
        dream_result = MagicMock()
        dream_result.ruin_probability = 0.45
        dream_result.survival_probability_30d = 0.72
        dream_result.resilience_score = 0.63
        dream_result.recommendations = ["reduce_burn", "defer_yield"]
        dream_result.optimal_scenarios = ["austerity_mode"]
        dream_result.risk_warnings = ["runway_45_days"]

        oikos_state = MagicMock()
        oikos_state.runway_days = 45
        service._oikos.snapshot.return_value = oikos_state
        service._oikos.integrate_dream_result = AsyncMock()
        service._economic_dream_worker = MagicMock()
        service._economic_dream_worker.run = AsyncMock(return_value=dream_result)
        service._threat_model_worker = None

        await service._run_economic_dreaming("cycle-001")

        # Expect the ONEIROS_ECONOMIC_INSIGHT event was emitted
        insight_events = [
            e for e in emitted
            if e["event_type"] == SynapseEventType.ONEIROS_ECONOMIC_INSIGHT
        ]
        assert len(insight_events) == 1
        payload = insight_events[0]["data"]
        assert payload["ruin_probability"] == pytest.approx(0.45)
        assert payload["cycle_id"] == "cycle-001"
        assert "reduce_burn" in payload["recommended_actions"]
        assert "runway_45_days" in payload["risk_warnings"]
        assert "austerity_mode" in payload["optimal_scenarios"]

    @pytest.mark.asyncio
    async def test_no_broadcast_when_ruin_below_threshold(self) -> None:
        """Insight event is NOT emitted when ruin_probability <= 0.2."""
        from systems.oneiros.service import OneirosService

        service = OneirosService.__new__(OneirosService)
        service._logger = MagicMock()
        service._logger.info = MagicMock()
        service._logger.error = MagicMock()
        service._oikos = MagicMock()
        service._synapse = MagicMock()
        service._journal = None

        emitted: list[dict[str, Any]] = []

        async def fake_emit(event_type: SynapseEventType, data: dict) -> None:
            emitted.append({"event_type": event_type, "data": data})

        service._emit_event = fake_emit  # type: ignore[assignment]

        dream_result = MagicMock()
        dream_result.ruin_probability = 0.10  # Below threshold
        dream_result.survival_probability_30d = 0.95
        dream_result.resilience_score = 0.88
        dream_result.recommendations = []
        dream_result.optimal_scenarios = []
        dream_result.risk_warnings = []

        oikos_state = MagicMock()
        oikos_state.runway_days = 180
        service._oikos.snapshot.return_value = oikos_state
        service._oikos.integrate_dream_result = AsyncMock()
        service._economic_dream_worker = MagicMock()
        service._economic_dream_worker.run = AsyncMock(return_value=dream_result)
        service._threat_model_worker = None

        await service._run_economic_dreaming("cycle-002")

        insight_events = [
            e for e in emitted
            if e["event_type"] == SynapseEventType.ONEIROS_ECONOMIC_INSIGHT
        ]
        assert len(insight_events) == 0


# ─── NEXUS-ECON-1: Convergence Reward Crediting ───────────────────


class TestOikosConvergenceRewardCrediting:
    """
    Oikos._on_federation_convergence_reward() must credit economic_reward_usd
    to state.reserves_usd and broadcast REVENUE_INJECTED.
    """

    @pytest.mark.asyncio
    async def test_reward_credited_to_reserves(self) -> None:
        """Convergence reward increments reserves_usd."""
        from decimal import Decimal
        from systems.oikos.service import OikosService

        service = OikosService.__new__(OikosService)
        service._logger = MagicMock()
        service._logger.info = MagicMock()

        # Minimal state stub
        state = MagicMock()
        state.reserves_usd = Decimal("10.000")
        service._state = state

        service._total_revenue_usd = Decimal("100.000")
        service._event_bus = None  # suppress re-broadcast in this test
        service._record_revenue_entry = MagicMock()
        service._credit_revenue_source = MagicMock()
        service._recalculate_derived_metrics = MagicMock()

        event = make_event(
            SynapseEventType.NEXUS_CONVERGENCE_METABOLIC_SIGNAL,
            {
                "economic_reward_usd": 0.003,
                "convergence_tier": 3,
                "theme": "causal_structure",
                "peer_count": 5,
            },
        )

        await service._on_federation_convergence_reward(event)

        # Reserves should increase by the reward amount
        assert state.reserves_usd == Decimal("10.000") + Decimal("0.003")
        assert service._total_revenue_usd == Decimal("100.000") + Decimal("0.003")
        service._record_revenue_entry.assert_called_once()
        service._recalculate_derived_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_zero_reward_is_ignored(self) -> None:
        """Zero or negative reward does not modify state."""
        from systems.oikos.service import OikosService

        service = OikosService.__new__(OikosService)
        service._logger = MagicMock()
        service._logger.info = MagicMock()

        state = MagicMock()
        state.reserves_usd = Decimal("5.000")
        service._state = state
        service._recalculate_derived_metrics = MagicMock()

        event = make_event(
            SynapseEventType.NEXUS_CONVERGENCE_METABOLIC_SIGNAL,
            {"economic_reward_usd": 0.0, "convergence_tier": 0},
        )
        await service._on_federation_convergence_reward(event)

        # State must be untouched
        service._recalculate_derived_metrics.assert_not_called()
        assert state.reserves_usd == Decimal("5.000")

    @pytest.mark.asyncio
    async def test_revenue_injected_rebroadcast(self) -> None:
        """REVENUE_INJECTED is emitted after crediting the reward."""
        import asyncio
        from systems.oikos.service import OikosService
        from systems.synapse.types import SynapseEvent

        service = OikosService.__new__(OikosService)
        service._logger = MagicMock()
        service._logger.info = MagicMock()

        state = MagicMock()
        state.reserves_usd = Decimal("0.000")
        service._state = state
        service._total_revenue_usd = Decimal("0.000")
        service._record_revenue_entry = MagicMock()
        service._credit_revenue_source = MagicMock()
        service._recalculate_derived_metrics = MagicMock()

        emitted: list[SynapseEvent] = []
        event_bus = MagicMock()
        event_bus.emit = AsyncMock(side_effect=lambda e: emitted.append(e))
        service._event_bus = event_bus

        event = make_event(
            SynapseEventType.NEXUS_CONVERGENCE_METABOLIC_SIGNAL,
            {"economic_reward_usd": 0.002, "convergence_tier": 2, "theme": "x", "peer_count": 3},
        )
        await service._on_federation_convergence_reward(event)

        # Allow ensure_future to run
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        revenue_events = [
            e for e in emitted
            if e.event_type == SynapseEventType.REVENUE_INJECTED
        ]
        assert len(revenue_events) == 1
        assert revenue_events[0].data["source"] == "federation_convergence"
        assert revenue_events[0].data["convergence_tier"] == 2


# ─── NEXUS-ECON-2/3: Economic Schema Context Preservation ────────


class TestEconomicSchemaContextPreservation:
    """
    ShareableWorldModelFragment.domain_hints and economic_context preserve
    economic metadata for federation peers.
    """

    def test_domain_hints_detected_from_labels(self) -> None:
        """_detect_economic_domain_hints finds DeFi domains from labels."""
        hints = _detect_economic_domain_hints(
            domain_labels=["yield_farming", "aave"],
            structure={},
        )
        assert "economic" in hints
        assert "yield_farming" in hints
        assert "defi" in hints

    def test_domain_hints_detected_from_structure(self) -> None:
        """_detect_economic_domain_hints scans structure string values."""
        hints = _detect_economic_domain_hints(
            domain_labels=["cognitive"],
            structure={"description": "Uniswap LP concentrated liquidity position"},
        )
        assert "economic" in hints
        assert "defi" in hints

    def test_no_hints_for_non_economic_schema(self) -> None:
        """Pure cognitive schema produces no economic hints."""
        hints = _detect_economic_domain_hints(
            domain_labels=["narrative", "emotion"],
            structure={"description": "emotional regulation causal chain"},
        )
        # No economic keywords → empty list
        assert hints == []

    def test_shareable_fragment_stores_domain_hints(self) -> None:
        """ShareableWorldModelFragment accepts and stores domain_hints."""
        fragment = ShareableWorldModelFragment(
            fragment_id="frag-001",
            source_instance_id="inst-a",
            abstract_structure={"nodes": 3},
            domain_labels=["economic"],
            domain_hints=["economic", "yield_farming"],
            economic_context={"protocol": "Aave", "apy_pct": 8.5},
        )
        assert "economic" in fragment.domain_hints
        assert "yield_farming" in fragment.domain_hints
        assert fragment.economic_context is not None
        assert fragment.economic_context["protocol"] == "Aave"

    def test_shareable_fragment_defaults_empty_hints(self) -> None:
        """domain_hints defaults to empty list when not set."""
        fragment = ShareableWorldModelFragment(
            fragment_id="frag-002",
            source_instance_id="inst-b",
            abstract_structure={},
            domain_labels=["cognitive"],
        )
        assert fragment.domain_hints == []
        assert fragment.economic_context is None


# ─── NEXUS-ECON-4: Economic Divergence Measurement ───────────────


class TestEconomicDivergenceMeasurement:
    """
    InstanceDivergenceProfile.economic_divergence and compute_economic_divergence().
    """

    def test_zero_divergence_for_single_profile(self) -> None:
        """Single profile cannot diverge - score is 0.0."""
        profile = InstanceDivergenceProfile(
            instance_id="a",
            strategy_revenue_rates={"bounty": 5.0, "yield": 2.0},
        )
        scores = compute_economic_divergence([profile])
        assert scores["a"] == pytest.approx(0.0)

    def test_zero_divergence_for_identical_rates(self) -> None:
        """Two profiles with identical rates produce zero divergence."""
        p1 = InstanceDivergenceProfile(
            instance_id="a",
            strategy_revenue_rates={"bounty": 5.0, "yield": 2.0},
        )
        p2 = InstanceDivergenceProfile(
            instance_id="b",
            strategy_revenue_rates={"bounty": 5.0, "yield": 2.0},
        )
        scores = compute_economic_divergence([p1, p2])
        assert scores["a"] == pytest.approx(0.0)
        assert scores["b"] == pytest.approx(0.0)

    def test_high_divergence_for_opposite_strategies(self) -> None:
        """Instances specialised in opposite strategies produce high divergence."""
        p1 = InstanceDivergenceProfile(
            instance_id="bounty_specialist",
            strategy_revenue_rates={"bounty": 100.0, "yield": 0.0},
        )
        p2 = InstanceDivergenceProfile(
            instance_id="yield_specialist",
            strategy_revenue_rates={"bounty": 0.0, "yield": 100.0},
        )
        scores = compute_economic_divergence([p1, p2])
        # Both instances have only one non-zero strategy, opposite to each other
        assert scores["bounty_specialist"] > 0.5
        assert scores["yield_specialist"] > 0.5

    def test_divergence_score_in_unit_range(self) -> None:
        """All economic divergence scores must be in [0, 1]."""
        profiles = [
            InstanceDivergenceProfile(
                instance_id=f"inst-{i}",
                strategy_revenue_rates={"a": float(i * 10), "b": float((5 - i) * 7)},
            )
            for i in range(5)
        ]
        scores = compute_economic_divergence(profiles)
        for iid, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{iid} score {score} out of range"

    def test_economic_divergence_field_on_profile(self) -> None:
        """InstanceDivergenceProfile.economic_divergence is a valid float field."""
        profile = InstanceDivergenceProfile(
            instance_id="test",
            economic_divergence=0.75,
            strategy_revenue_rates={"yield": 10.0},
        )
        assert profile.economic_divergence == pytest.approx(0.75)

    def test_empty_strategy_rates_produce_zero_divergence(self) -> None:
        """Profiles with no strategy data produce zero scores."""
        p1 = InstanceDivergenceProfile(instance_id="a")
        p2 = InstanceDivergenceProfile(instance_id="b")
        scores = compute_economic_divergence([p1, p2])
        assert scores["a"] == pytest.approx(0.0)
        assert scores["b"] == pytest.approx(0.0)


# ─── Nova Economic Dream Insight Integration ──────────────────────


class TestNovaEconomicDreamInsight:
    """
    Nova._on_economic_dream_insight() integrates ruin risk into world model beliefs.
    """

    @pytest.mark.asyncio
    async def test_ruin_risk_belief_injected(self) -> None:
        """High ruin probability updates economic_ruin_risk belief entity."""
        from systems.nova.service import NovaService

        service = NovaService.__new__(NovaService)
        service._logger = MagicMock()
        service._logger.info = MagicMock()
        service._deliberation_engine = None  # disable immediate_deliberation

        # Mock belief updater
        belief_updater = MagicMock()
        beliefs = MagicMock()
        beliefs.entities = {}
        belief_updater.beliefs = beliefs
        service._belief_updater = belief_updater

        event = make_event(
            SynapseEventType.ONEIROS_ECONOMIC_INSIGHT,
            {
                "ruin_probability": 0.55,
                "risk_warnings": ["runway_30_days"],
                "recommended_actions": ["cut_burn_rate"],
                "dream_validity_confidence": 0.70,
                "cycle_id": "sleep-42",
            },
        )

        await service._on_economic_dream_insight(event)

        # Should inject economic_ruin_risk belief
        belief_updater.inject_entity.assert_called_once_with(
            entity_id="economic_ruin_risk",
            name="economic_ruin_risk",
            confidence=pytest.approx(0.55 * 0.70, abs=1e-6),
        )

    @pytest.mark.asyncio
    async def test_high_ruin_triggers_immediate_deliberation(self) -> None:
        """Ruin > 0.3 with valid confidence and recommendations triggers deliberation."""
        from systems.nova.service import NovaService

        service = NovaService.__new__(NovaService)
        service._logger = MagicMock()
        service._logger.info = MagicMock()

        belief_updater = MagicMock()
        belief_updater.beliefs = MagicMock()
        belief_updater.beliefs.entities = {}
        service._belief_updater = belief_updater

        deliberation_triggered: list[dict] = []

        async def fake_deliberation(reason: str, urgency: float = 0.7) -> None:
            deliberation_triggered.append({"reason": reason, "urgency": urgency})

        service._immediate_deliberation = fake_deliberation  # type: ignore[assignment]
        service._deliberation_engine = MagicMock()  # mark as wired

        event = make_event(
            SynapseEventType.ONEIROS_ECONOMIC_INSIGHT,
            {
                "ruin_probability": 0.65,
                "risk_warnings": ["critical_runway"],
                "recommended_actions": ["emergency_austerity"],
                "dream_validity_confidence": 0.80,
                "cycle_id": "sleep-99",
            },
        )

        await service._on_economic_dream_insight(event)
        # Allow the task to schedule
        await asyncio.sleep(0)

        assert len(deliberation_triggered) == 1
        assert deliberation_triggered[0]["reason"] == "economic_dream_ruin_warning"
        assert deliberation_triggered[0]["urgency"] == pytest.approx(0.65 * 0.80, abs=1e-6)

    @pytest.mark.asyncio
    async def test_low_confidence_no_deliberation(self) -> None:
        """Low dream_validity_confidence does not trigger immediate deliberation."""
        from systems.nova.service import NovaService

        service = NovaService.__new__(NovaService)
        service._logger = MagicMock()
        service._logger.info = MagicMock()

        belief_updater = MagicMock()
        belief_updater.beliefs = MagicMock()
        belief_updater.beliefs.entities = {}
        service._belief_updater = belief_updater

        deliberation_triggered: list[dict] = []

        async def fake_deliberation(reason: str, urgency: float = 0.7) -> None:
            deliberation_triggered.append({"reason": reason, "urgency": urgency})

        service._immediate_deliberation = fake_deliberation  # type: ignore[assignment]

        event = make_event(
            SynapseEventType.ONEIROS_ECONOMIC_INSIGHT,
            {
                "ruin_probability": 0.50,
                "risk_warnings": ["warning"],
                "recommended_actions": ["some_action"],
                "dream_validity_confidence": 0.30,  # Below 0.6 threshold
                "cycle_id": "sleep-10",
            },
        )

        await service._on_economic_dream_insight(event)
        await asyncio.sleep(0)

        assert len(deliberation_triggered) == 0
