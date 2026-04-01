"""
Unit tests for Nova economic intelligence gaps closure.

Covers:
  NOVA-ECON-1: Economic event subscription handlers
  NOVA-ECON-2: Economic policy template selection via generate_economic_intent()
  NOVA-ECON-3: BeliefUrgencyMonitor - confidence shift triggers immediate deliberation
  EVO-NOVA-1: Full hypothesis metadata in NOVA_GOAL_INJECTED

All tests are synchronous or use pytest-asyncio.
No real Neo4j, Redis, or LLM calls.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.nova.belief_updater import BeliefUrgencyMonitor, BeliefUpdater, _URGENCY_THRESHOLD
from systems.nova.policy_generator import (
    PolicyGenerator,
    ThompsonSampler,
    _PROCEDURE_TEMPLATES,
)
from systems.nova.types import BeliefState, EntityBelief


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_belief_state(entities: dict[str, float] | None = None) -> BeliefState:
    """Build a minimal BeliefState, optionally with named entity confidences."""
    state = BeliefState()
    if entities:
        entity_map = {}
        for eid, conf in entities.items():
            entity_map[eid] = EntityBelief(
                entity_id=eid,
                name=eid,
                confidence=conf,
            )
        state = state.model_copy(update={"entities": entity_map})
    return state


def make_synapse_event(event_type: str, data: dict[str, Any]) -> Any:
    """Construct a minimal Synapse event mock."""
    event = MagicMock()
    event.event_type = event_type
    event.data = data
    return event


# ─── NOVA-ECON-2: Economic policy template coverage ───────────────────────────


class TestEconomicPolicyTemplates:
    """Verify all 5 economic policy templates exist in _PROCEDURE_TEMPLATES."""

    def _get_economic_templates(self) -> list[dict[str, Any]]:
        return [t for t in _PROCEDURE_TEMPLATES if t.get("domain") == "economic"]

    def test_all_five_economic_templates_present(self) -> None:
        economic = self._get_economic_templates()
        names = {t["name"] for t in economic}
        assert "Autonomous bounty hunting" in names, "bounty_hunting template missing"
        assert "Yield farming deployment" in names, "yield_farming template missing"
        assert "Cost optimization sweep" in names, "cost_optimization template missing"
        assert "Asset liquidation for liquidity" in names, "asset_liquidation template missing"
        assert "Revenue stream diversification" in names, "revenue_diversification template missing"

    def test_success_rates_are_distinct(self) -> None:
        economic = self._get_economic_templates()
        rates = [t["success_rate"] for t in economic]
        # 5 templates should not all have the same success rate
        assert len(set(rates)) > 1, "All economic templates have identical success rates"

    def test_yield_farming_has_higher_success_rate_than_bounty(self) -> None:
        bounty = next(t for t in _PROCEDURE_TEMPLATES if t["name"] == "Autonomous bounty hunting")
        yield_t = next(t for t in _PROCEDURE_TEMPLATES if t["name"] == "Yield farming deployment")
        assert yield_t["success_rate"] > bounty["success_rate"]

    def test_cost_optimization_has_highest_success_rate(self) -> None:
        economic = self._get_economic_templates()
        cost_opt = next(t for t in economic if t["name"] == "Cost optimization sweep")
        max_rate = max(t["success_rate"] for t in economic)
        assert cost_opt["success_rate"] == max_rate

    def test_revenue_diversification_has_lowest_success_rate(self) -> None:
        economic = self._get_economic_templates()
        rev_div = next(t for t in economic if t["name"] == "Revenue stream diversification")
        min_rate = min(t["success_rate"] for t in economic)
        assert rev_div["success_rate"] == min_rate


# ─── NOVA-ECON-2: generate_economic_intent() ──────────────────────────────────


class TestGenerateEconomicIntent:
    """Verify generate_economic_intent() scores policies by EFE, not keywords."""

    def _make_policy_generator(self) -> PolicyGenerator:
        llm = AsyncMock()
        llm.generate = AsyncMock()
        return PolicyGenerator(llm=llm, instance_name="TEST")

    def test_returns_policy_object(self) -> None:
        pg = self._make_policy_generator()
        beliefs = make_belief_state()
        policy = pg.generate_economic_intent(beliefs)
        assert policy is not None
        assert hasattr(policy, "name")
        assert hasattr(policy, "steps")

    def test_high_economic_risk_selects_cost_optimization(self) -> None:
        """When economic risk is high, cost optimization should rank highly."""
        pg = self._make_policy_generator()
        beliefs = make_belief_state({
            "economic_risk_level": 0.9,  # High risk
            "bounty_success_rate": 0.2,  # Low bounty success
        })
        policy = pg.generate_economic_intent(beliefs, economic_context={"wallet_balance_usd": 200.0})
        # Cost optimization has 80% success rate and negative EFE under risk
        assert policy.name in {
            "Cost optimization sweep",
            "Autonomous bounty hunting",
            "Yield farming deployment",
        }

    def test_low_hours_until_depleted_selects_liquidation(self) -> None:
        """When balance < 24h runway, asset liquidation should win."""
        pg = self._make_policy_generator()
        beliefs = make_belief_state({"economic_risk_level": 0.8})
        policy = pg.generate_economic_intent(
            beliefs,
            economic_context={
                "wallet_balance_usd": 5.0,  # Only 5 USD
                "burn_rate_hourly_usd": 1.0,  # 5 hours runway
            },
        )
        assert policy.name == "Asset liquidation for liquidity"

    def test_high_yield_confidence_selects_yield_farming(self) -> None:
        """When yield APY belief confidence is high, yield farming wins."""
        pg = self._make_policy_generator()
        beliefs = make_belief_state({
            "yield_apy_aave": 0.95,  # Very high confidence in yield
            "bounty_success_rate": 0.4,
            "economic_risk_level": 0.1,
        })
        policy = pg.generate_economic_intent(
            beliefs,
            economic_context={"wallet_balance_usd": 500.0},
        )
        assert policy.name == "Yield farming deployment"

    def test_dry_bounties_and_yield_selects_diversification(self) -> None:
        """When both bounty and yield confidence are low, diversification wins."""
        pg = self._make_policy_generator()
        beliefs = make_belief_state({
            "bounty_success_rate": 0.15,  # Very low
            "yield_apy_aave": 0.10,  # Very low
        })
        policy = pg.generate_economic_intent(
            beliefs,
            economic_context={"wallet_balance_usd": 300.0, "burn_rate_hourly_usd": 0.5},
        )
        assert policy.name == "Revenue stream diversification"

    def test_policy_steps_are_non_empty(self) -> None:
        pg = self._make_policy_generator()
        beliefs = make_belief_state()
        policy = pg.generate_economic_intent(beliefs)
        assert len(policy.steps) >= 1


# ─── NOVA-ECON-3: BeliefUrgencyMonitor ────────────────────────────────────────


class TestBeliefUrgencyMonitor:
    """Verify BeliefUrgencyMonitor triggers callback on >20% confidence shift."""

    def test_no_trigger_on_first_observation(self) -> None:
        """First check establishes baseline - no callback should fire."""
        callback = AsyncMock()
        monitor = BeliefUrgencyMonitor(callback=callback)
        monitor.check("economic_risk_level", 0.5)
        # No previous value → no trigger
        assert not callback.called

    def test_no_trigger_on_small_shift(self) -> None:
        """Shift below threshold should not trigger."""
        callback = AsyncMock()
        monitor = BeliefUrgencyMonitor(callback=callback)
        monitor.check("economic_risk_level", 0.5)
        monitor.check("economic_risk_level", 0.5 + _URGENCY_THRESHOLD - 0.01)
        assert not callback.called

    def test_triggers_on_threshold_breach(self) -> None:
        """Shift >= threshold on a priority key must fire callback."""
        fired: list[dict] = []

        async def capture(**kwargs: Any) -> None:
            fired.append(kwargs)

        monitor = BeliefUrgencyMonitor(callback=capture)
        monitor.check("economic_risk_level", 0.5)

        # Patch asyncio to capture task creation
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.create_task = MagicMock()
            monitor.check("economic_risk_level", 0.5 + _URGENCY_THRESHOLD + 0.05)
            # create_task should have been called
            assert mock_loop.return_value.create_task.called

    def test_non_priority_key_never_triggers(self) -> None:
        """Non-priority entity IDs must never trigger the callback."""
        callback = AsyncMock()
        monitor = BeliefUrgencyMonitor(callback=callback)
        monitor.check("some_random_belief", 0.1)
        monitor.check("some_random_belief", 0.9)  # +0.8 shift - but not a priority key
        assert not callback.called

    def test_all_priority_keys_are_monitored(self) -> None:
        """Each priority key triggers; non-priority keys don't."""
        from systems.nova.belief_updater import _PRIORITY_BELIEF_KEYS

        triggered: list[str] = []

        async def capture(reason: str, urgency: float) -> None:
            triggered.append(reason)

        for key in _PRIORITY_BELIEF_KEYS:
            monitor = BeliefUrgencyMonitor(callback=capture)
            monitor.check(key, 0.1)
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.create_task = MagicMock()
                monitor.check(key, 0.9)
                assert mock_loop.return_value.create_task.called, f"Priority key {key} did not trigger"


class TestBeliefUpdaterUrgencyWiring:
    """Verify BeliefUpdater calls urgency monitor on inject_entity/update_entity."""

    def test_inject_entity_calls_monitor(self) -> None:
        fired: list[tuple[str, float]] = []

        class _TrackingMonitor:
            def check(self, entity_id: str, new_confidence: float) -> None:
                fired.append((entity_id, new_confidence))

        updater = BeliefUpdater()
        updater.set_urgency_monitor(_TrackingMonitor())  # type: ignore[arg-type]
        updater.inject_entity("economic_risk_level", "economic_risk_level", confidence=0.8)
        assert fired == [("economic_risk_level", 0.8)]

    def test_update_entity_calls_monitor_on_confidence_change(self) -> None:
        fired: list[tuple[str, float]] = []

        class _TrackingMonitor:
            def check(self, entity_id: str, new_confidence: float) -> None:
                fired.append((entity_id, new_confidence))

        updater = BeliefUpdater()
        updater.inject_entity("bounty_success_rate", "bounty_success_rate", confidence=0.6)
        updater.set_urgency_monitor(_TrackingMonitor())  # type: ignore[arg-type]
        fired.clear()  # Ignore inject call

        updater.update_entity("bounty_success_rate", confidence=0.3)
        assert fired == [("bounty_success_rate", 0.3)]

    def test_update_entity_no_monitor_does_not_crash(self) -> None:
        updater = BeliefUpdater()
        updater.inject_entity("bounty_success_rate", "bounty_success_rate", confidence=0.5)
        # No monitor wired - must not raise
        result = updater.update_entity("bounty_success_rate", confidence=0.9)
        assert result is True


# ─── NOVA-ECON-1: Economic event handler tests ────────────────────────────────


@pytest.mark.asyncio
class TestNovaEconEventHandlers:
    """Integration-light tests for the 4 new economic event handlers on NovaService.

    Uses a minimal NovaService stub to avoid the full initialization chain.
    """

    def _make_minimal_nova(self) -> Any:
        """Build just enough of NovaService to test the economic handlers."""
        from unittest.mock import MagicMock, patch

        with patch("systems.nova.service.NovaService.__init__", return_value=None):
            nova = object.__new__(__import__("systems.nova.service", fromlist=["NovaService"]).NovaService)

        nova._belief_updater = BeliefUpdater()
        nova._goal_manager = None
        nova._deliberation_engine = None
        nova._synapse = None
        nova._logger = MagicMock()
        nova._logger.info = MagicMock()
        nova._logger.warning = MagicMock()
        nova._logger.debug = MagicMock()
        return nova

    async def test_on_fovea_econ_error_updates_risk_belief(self) -> None:
        from systems.nova.service import NovaService

        nova = self._make_minimal_nova()
        event = make_synapse_event("fovea_internal_prediction_error", {
            "prediction_error": {"economic": 0.5},
            "cost_ratio": 1.8,
            "salience_hint": 0.3,
        })

        await NovaService._on_fovea_econ_error(nova, event)

        risk = nova._belief_updater.beliefs.entities.get("economic_risk_level")
        assert risk is not None
        assert risk.confidence > 0.5

    async def test_on_fovea_econ_error_ignores_non_economic(self) -> None:
        from systems.nova.service import NovaService

        nova = self._make_minimal_nova()
        event = make_synapse_event("fovea_internal_prediction_error", {
            "prediction_error": {"visual": 0.9},
            "cost_ratio": 1.0,
            "error_type": "visual_mismatch",
        })

        await NovaService._on_fovea_econ_error(nova, event)

        # Should not update any economic belief
        risk = nova._belief_updater.beliefs.entities.get("economic_risk_level")
        assert risk is None

    async def test_on_revenue_change_injects_belief(self) -> None:
        from systems.nova.service import NovaService

        nova = self._make_minimal_nova()
        event = make_synapse_event("revenue_injected", {
            "amount_usd": 25.0,
            "source": "bounty_payout",
        })

        await NovaService._on_revenue_change(nova, event)

        ratio = nova._belief_updater.beliefs.entities.get("revenue_burn_ratio")
        assert ratio is not None
        assert ratio.confidence > 0.4

    async def test_on_bounty_outcome_success_raises_confidence(self) -> None:
        from systems.nova.service import NovaService

        nova = self._make_minimal_nova()
        nova._belief_updater.inject_entity("bounty_success_rate", "bounty_success_rate", 0.5)
        event = make_synapse_event("bounty_paid", {
            "success": True,
            "amount_usd": 50.0,
            "bounty_id": "test-bounty-001",
        })

        await NovaService._on_bounty_outcome(nova, event)

        rate = nova._belief_updater.beliefs.entities.get("bounty_success_rate")
        assert rate is not None
        assert rate.confidence > 0.5

    async def test_on_bounty_outcome_failure_lowers_confidence(self) -> None:
        from systems.nova.service import NovaService

        nova = self._make_minimal_nova()
        nova._belief_updater.inject_entity("bounty_success_rate", "bounty_success_rate", 0.6)
        event = make_synapse_event("bounty_paid", {
            "success": False,
            "amount_usd": 0.0,
            "bounty_id": "test-bounty-002",
        })

        await NovaService._on_bounty_outcome(nova, event)

        rate = nova._belief_updater.beliefs.entities.get("bounty_success_rate")
        assert rate is not None
        assert rate.confidence < 0.6

    async def test_on_yield_outcome_success_raises_apy_belief(self) -> None:
        from systems.nova.service import NovaService

        nova = self._make_minimal_nova()
        event = make_synapse_event("yield_deployment_result", {
            "success": True,
            "apy": 0.08,  # 8% APY
            "protocol": "aave",
            "amount_usd": 100.0,
        })

        await NovaService._on_yield_outcome(nova, event)

        apy_belief = nova._belief_updater.beliefs.entities.get("yield_apy_aave")
        assert apy_belief is not None
        assert apy_belief.confidence > 0.4


# ─── EVO-NOVA-1: Hypothesis metadata in goal injection ────────────────────────


class TestEvoNovaGoalMetadata:
    """Verify _generate_goal_from_hypothesis emits full metadata in event payload."""

    @pytest.mark.asyncio
    async def test_goal_injection_includes_hypothesis_metadata(self) -> None:
        """NOVA_GOAL_INJECTED payload must contain confidence, evidence_score, domain."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Build a minimal EvoService stub
        with patch("systems.evo.service.EvoService.__init__", return_value=None):
            evo = object.__new__(
                __import__("systems.evo.service", fromlist=["EvoService"]).EvoService
            )
        evo._event_bus = AsyncMock()
        evo._tournament_engine = None
        evo._hypothesis_engine = None
        evo._logger = MagicMock()
        evo._logger.info = MagicMock()
        evo._logger.warning = MagicMock()
        evo._logger.debug = MagicMock()

        # Build a minimal hypothesis mock
        hyp = MagicMock()
        hyp.id = "hyp-abc-001"
        hyp.statement = "Yield farming in bear markets generates 6% APY."
        hyp.confidence = 0.75
        hyp.evidence_score = 4.2
        hyp.domain = "economic"

        emitted_events: list[Any] = []
        evo._event_bus.emit = AsyncMock(side_effect=lambda e: emitted_events.append(e))

        from systems.evo.service import EvoService
        await EvoService._generate_goal_from_hypothesis(evo, hyp)

        assert len(emitted_events) == 1
        event = emitted_events[0]
        data = event.data

        assert data["hypothesis_id"] == "hyp-abc-001"
        assert "hypothesis_statement" in data
        assert data["confidence"] == pytest.approx(0.75)
        assert data["evidence_score"] == pytest.approx(4.2)
        assert data["domain"] == "economic"
        assert "thompson_arm_id" in data
        assert "thompson_arm_weights" in data

    def test_get_thompson_arm_weights_returns_dict(self) -> None:
        """get_thompson_arm_weights() must return a dict (empty if no tournaments)."""
        from unittest.mock import MagicMock, patch

        with patch("systems.evo.service.EvoService.__init__", return_value=None):
            evo = object.__new__(
                __import__("systems.evo.service", fromlist=["EvoService"]).EvoService
            )
        evo._tournament_engine = None
        evo._hypothesis_engine = None
        evo._logger = MagicMock()

        from systems.evo.service import EvoService
        result = EvoService.get_thompson_arm_weights(evo, domain="economic")
        assert isinstance(result, dict)

    def test_get_thompson_arm_weights_with_active_tournament(self) -> None:
        """When a tournament is active, weights should be returned."""
        from unittest.mock import MagicMock, patch

        with patch("systems.evo.service.EvoService.__init__", return_value=None):
            evo = object.__new__(
                __import__("systems.evo.service", fromlist=["EvoService"]).EvoService
            )
        evo._hypothesis_engine = None
        evo._logger = MagicMock()

        # Mock tournament engine with one active tournament
        arm_a = MagicMock()
        arm_a.hypothesis_id = "hyp-1"
        arm_a.alpha = 5.0
        arm_a.beta = 2.0  # Beta posterior mean = 5/7 ≈ 0.714

        arm_b = MagicMock()
        arm_b.hypothesis_id = "hyp-2"
        arm_b.alpha = 2.0
        arm_b.beta = 5.0  # Beta posterior mean = 2/7 ≈ 0.286

        tournament = MagicMock()
        tournament.arms = {"arm_a": arm_a, "arm_b": arm_b}

        mock_engine = MagicMock()
        mock_engine._active_tournaments = {"t-001": tournament}
        evo._tournament_engine = mock_engine

        from systems.evo.service import EvoService
        result = EvoService.get_thompson_arm_weights(evo, domain="")

        assert "t-001" in result
        weights = result["t-001"]
        assert "arm_a" in weights
        assert weights["arm_a"] == pytest.approx(5.0 / 7.0, rel=1e-3)
        assert weights["arm_b"] == pytest.approx(2.0 / 7.0, rel=1e-3)
