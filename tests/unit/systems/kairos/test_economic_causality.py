"""
Unit tests for KAIROS-ECON-1: Economic causal mining pipeline.

Tests:
1. EconomicCausalMiner.discover_patterns() - all three heuristics
2. KairosPipeline economic event handlers (_on_economic_episode,
   _on_fovea_economic_error, _on_price_observation)
3. Nova._on_economic_causal_invariant() - belief update + EFE weight update
4. Oikos._emit_economic_episode_to_memory() - payload structure
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from systems.kairos.invariant_distiller import EconomicCausalMiner


# ─── EconomicCausalMiner Unit Tests ───────────────────────────────────────────


class TestEconomicCausalMinerProtocolSuccess:
    """Heuristic 1: Protocol success rates."""

    def _make_obs(
        self,
        protocol: str,
        success: bool,
        roi: float = 5.0,
        day: int = 0,
        eth_price: float = 0.0,
    ) -> dict:
        return {
            "action_type": "yield_deploy",
            "protocol": protocol,
            "success": success,
            "roi_pct": roi,
            "day_of_week": day,
            "eth_price_usd": eth_price,
            "causal_variables": {
                "roi_pct": roi,
                "day_of_week": float(day),
                "eth_price_usd": eth_price,
            },
        }

    def test_high_success_rate_protocol_detected(self) -> None:
        """Protocol with >70% success rate emits a positive invariant."""
        miner = EconomicCausalMiner()
        # 9 successes, 1 failure → 90% success
        obs = [self._make_obs("aave", True) for _ in range(9)]
        obs.append(self._make_obs("aave", False))
        patterns = miner.discover_patterns(obs)

        assert len(patterns) == 1
        p = patterns[0]
        assert p["invariant_type"] == "protocol_success_rate"
        assert p["cause"] == "protocol:aave"
        assert p["effect"] == "high_success_probability"
        assert p["direction"] == "positive"
        assert p["confidence"] >= 0.75
        assert p["sample_count"] == 10

    def test_low_success_rate_protocol_detected(self) -> None:
        """Protocol with <30% success rate emits a negative invariant."""
        miner = EconomicCausalMiner()
        # 1 success, 9 failures → 10% success
        obs = [self._make_obs("risky_protocol", False) for _ in range(9)]
        obs.append(self._make_obs("risky_protocol", True))
        patterns = miner.discover_patterns(obs)

        assert len(patterns) == 1
        p = patterns[0]
        assert p["direction"] == "negative"
        assert p["confidence"] >= 0.75
        assert p["metadata"]["success_rate"] == pytest.approx(0.1, abs=0.01)

    def test_protocol_below_min_sample_size_ignored(self) -> None:
        """Protocols with fewer than 5 observations are not emitted."""
        miner = EconomicCausalMiner()
        obs = [self._make_obs("small_protocol", True) for _ in range(4)]
        patterns = miner.discover_patterns(obs)
        assert patterns == []

    def test_neutral_protocol_not_emitted(self) -> None:
        """Protocol near 50% success rate produces no pattern."""
        miner = EconomicCausalMiner()
        obs = [self._make_obs("neutral", i % 2 == 0) for i in range(10)]
        patterns = miner.discover_patterns(obs)
        assert all(p["invariant_type"] != "protocol_success_rate" for p in patterns)

    def test_multiple_protocols_independent(self) -> None:
        """Two protocols with distinct patterns both get detected."""
        miner = EconomicCausalMiner()
        good_obs = [self._make_obs("aave", True) for _ in range(8)]
        bad_obs = [self._make_obs("bad_protocol", False) for _ in range(8)]
        bad_obs += [self._make_obs("bad_protocol", True)]  # 1/9 success
        patterns = miner.discover_patterns(good_obs + bad_obs)

        causes = {p["cause"] for p in patterns if p["invariant_type"] == "protocol_success_rate"}
        assert "protocol:aave" in causes
        assert "protocol:bad_protocol" in causes


class TestEconomicCausalMinerTimeOfWeek:
    """Heuristic 2: Time-of-week effects."""

    def _make_obs(self, day: int, roi: float) -> dict:
        return {
            "action_type": "bounty_hunt",
            "protocol": "github",
            "success": True,
            "roi_pct": roi,
            "day_of_week": day,
            "eth_price_usd": 0.0,
            "causal_variables": {"roi_pct": roi, "day_of_week": float(day), "eth_price_usd": 0.0},
        }

    def test_weekday_vs_weekend_effect_detected(self) -> None:
        """Significant ROI difference between weekday and weekend is detected."""
        miner = EconomicCausalMiner()
        # Weekdays (Mon=0 … Fri=4): high ROI
        weekday_obs = [self._make_obs(d % 5, 50.0) for d in range(20)]
        # Weekends (Sat=5, Sun=6): low ROI
        weekend_obs = [self._make_obs(5 + d % 2, 5.0) for d in range(10)]
        patterns = miner.discover_patterns(weekday_obs + weekend_obs)

        tok_patterns = [p for p in patterns if p["invariant_type"] == "time_of_week_effect"]
        assert len(tok_patterns) == 1
        p = tok_patterns[0]
        assert p["direction"] == "positive"  # weekdays > weekends
        assert p["confidence"] >= 0.60
        assert p["metadata"]["relative_diff_pct"] >= 30.0

    def test_no_time_effect_when_rois_similar(self) -> None:
        """No time-of-week invariant emitted when weekday/weekend ROI is similar."""
        miner = EconomicCausalMiner()
        weekday_obs = [self._make_obs(d % 5, 20.0) for d in range(15)]
        weekend_obs = [self._make_obs(5 + d % 2, 21.0) for d in range(8)]
        patterns = miner.discover_patterns(weekday_obs + weekend_obs)
        tok_patterns = [p for p in patterns if p["invariant_type"] == "time_of_week_effect"]
        assert tok_patterns == []

    def test_insufficient_weekend_obs_no_pattern(self) -> None:
        """Fewer than MIN_SAMPLE_SIZE weekend observations → no pattern."""
        miner = EconomicCausalMiner()
        weekday_obs = [self._make_obs(d % 5, 50.0) for d in range(15)]
        weekend_obs = [self._make_obs(5, 5.0) for _ in range(3)]  # only 3
        patterns = miner.discover_patterns(weekday_obs + weekend_obs)
        tok_patterns = [p for p in patterns if p["invariant_type"] == "time_of_week_effect"]
        assert tok_patterns == []


class TestEconomicCausalMinerPriceDependentYield:
    """Heuristic 3: Price-dependent yield."""

    def _make_obs(self, eth_price: float, roi: float) -> dict:
        return {
            "action_type": "yield_deploy",
            "protocol": "aave",
            "success": True,
            "roi_pct": roi,
            "day_of_week": 1,
            "eth_price_usd": eth_price,
            "causal_variables": {"roi_pct": roi, "day_of_week": 1.0, "eth_price_usd": eth_price},
        }

    def test_price_tier_effect_detected(self) -> None:
        """Low ETH price → significantly lower ROI is detected."""
        miner = EconomicCausalMiner()
        # Low price tier: $1000 → 2% ROI
        low_obs = [self._make_obs(1000.0, 2.0) for _ in range(10)]
        # Mid price tier: $2000 → 8% ROI
        mid_obs = [self._make_obs(2000.0, 8.0) for _ in range(10)]
        # High price tier: $3000 → 15% ROI
        high_obs = [self._make_obs(3000.0, 15.0) for _ in range(10)]

        patterns = miner.discover_patterns(low_obs + mid_obs + high_obs)
        price_patterns = [p for p in patterns if p["invariant_type"] == "price_dependent_yield"]
        assert len(price_patterns) >= 1
        # At least one tier should have high confidence
        assert any(p["confidence"] >= 0.60 for p in price_patterns)

    def test_no_pattern_when_rois_uniform(self) -> None:
        """No price-dependent pattern when ROI is flat across price tiers."""
        miner = EconomicCausalMiner()
        obs = [self._make_obs(1000.0 + i * 100, 10.0) for i in range(30)]
        patterns = miner.discover_patterns(obs)
        price_patterns = [p for p in patterns if p["invariant_type"] == "price_dependent_yield"]
        assert price_patterns == []

    def test_zero_eth_price_observations_excluded(self) -> None:
        """Observations with eth_price_usd=0 are excluded from price-tier mining."""
        miner = EconomicCausalMiner()
        obs = [self._make_obs(0.0, 5.0) for _ in range(20)]
        patterns = miner.discover_patterns(obs)
        price_patterns = [p for p in patterns if p["invariant_type"] == "price_dependent_yield"]
        assert price_patterns == []

    def test_insufficient_observations_no_pattern(self) -> None:
        """Fewer than 2×MIN_SAMPLE_SIZE price observations → no pattern."""
        miner = EconomicCausalMiner()
        # Only 8 obs, need 10 (2×5)
        obs = [self._make_obs(float(i * 500), float(i)) for i in range(8)]
        patterns = miner.discover_patterns(obs)
        price_patterns = [p for p in patterns if p["invariant_type"] == "price_dependent_yield"]
        assert price_patterns == []


class TestEconomicCausalMinerConfidenceFloor:
    """Patterns below confidence 0.6 are filtered."""

    def test_low_confidence_patterns_excluded(self) -> None:
        """Patterns with confidence < 0.6 are not returned."""
        miner = EconomicCausalMiner()
        # 71% success: confidence = 0.70 + (0.71 - 0.70) * 1.5 = 0.715 ≥ 0.6 ✓
        # But just barely above 70% threshold - should still be ≥ 0.6
        obs = [{"action_type": "yield_deploy", "protocol": "borderline",
                "success": i < 71, "roi_pct": 5.0, "day_of_week": 1,
                "eth_price_usd": 0.0,
                "causal_variables": {"roi_pct": 5.0, "day_of_week": 1.0, "eth_price_usd": 0.0}}
               for i in range(100)]
        patterns = miner.discover_patterns(obs)
        assert all(p["confidence"] >= 0.6 for p in patterns)


# ─── Pipeline Handler Tests ────────────────────────────────────────────────────


class TestKairosPipelineEconomicHandlers:
    """KairosPipeline._on_economic_episode and _on_price_observation."""

    def _make_pipeline(self) -> "KairosPipeline":
        from systems.kairos.pipeline import KairosPipeline
        return KairosPipeline()

    @pytest.mark.asyncio
    async def test_economic_episode_buffered(self) -> None:
        """Valid economic episodes are appended to _economic_observations."""
        pipeline = self._make_pipeline()
        event = MagicMock()
        event.data = {
            "action_type": "yield_deploy",
            "success": True,
            "causal_variables": {"roi_pct": 8.0},
            "causal_substrate": "economic",
        }
        await pipeline._on_economic_episode(event)
        assert len(pipeline._economic_observations) == 1
        assert pipeline._economic_events_received == 1

    @pytest.mark.asyncio
    async def test_episode_missing_causal_variables_dropped(self) -> None:
        """Episodes without causal_variables key are silently dropped."""
        pipeline = self._make_pipeline()
        event = MagicMock()
        event.data = {"action_type": "yield_deploy", "success": True}
        await pipeline._on_economic_episode(event)
        assert len(pipeline._economic_observations) == 0

    @pytest.mark.asyncio
    async def test_episode_buffer_capped_at_500(self) -> None:
        """Buffer does not exceed 500 observations."""
        pipeline = self._make_pipeline()
        for _ in range(600):
            event = MagicMock()
            event.data = {
                "action_type": "bounty_hunt",
                "success": True,
                "causal_variables": {"roi_pct": 5.0},
            }
            await pipeline._on_economic_episode(event)
        assert len(pipeline._economic_observations) == 500

    @pytest.mark.asyncio
    async def test_fovea_economic_error_seeds_candidate(self) -> None:
        """Economic Fovea errors with magnitude ≥ 0.3 seed a CorrelationCandidate."""
        from systems.kairos.pipeline import KairosPipeline

        pipeline = KairosPipeline()
        # Mock correlation miner to capture preseeds
        pipeline._correlation_miner = MagicMock()
        pipeline._correlation_miner.add_preseed = MagicMock()

        event = MagicMock()
        event.data = {
            "source_system": "oikos",
            "percept_id": "test123",
            "prediction_error": {"economic": 0.6},
        }
        await pipeline._on_fovea_economic_error(event)
        pipeline._correlation_miner.add_preseed.assert_called_once()

    @pytest.mark.asyncio
    async def test_fovea_non_economic_source_ignored(self) -> None:
        """Fovea errors from non-economic systems are silently ignored."""
        from systems.kairos.pipeline import KairosPipeline

        pipeline = KairosPipeline()
        pipeline._correlation_miner = MagicMock()
        pipeline._correlation_miner.add_preseed = MagicMock()

        event = MagicMock()
        event.data = {
            "source_system": "atune",
            "prediction_error": {"economic": 0.9},
        }
        await pipeline._on_fovea_economic_error(event)
        pipeline._correlation_miner.add_preseed.assert_not_called()

    @pytest.mark.asyncio
    async def test_price_observation_buffered(self) -> None:
        """Valid price observations with non-empty pair are buffered."""
        from systems.kairos.pipeline import KairosPipeline

        pipeline = KairosPipeline()
        event = MagicMock()
        event.data = {
            "pair": ["ETH", "USDC"],
            "price": "2500.00",
            "timestamp": "2026-03-08T12:00:00",
            "pool_address": "0xabc",
        }
        await pipeline._on_price_observation(event)
        assert len(pipeline._price_observations) == 1

    @pytest.mark.asyncio
    async def test_price_observation_empty_pair_ignored(self) -> None:
        """Price observations with empty pair are dropped."""
        from systems.kairos.pipeline import KairosPipeline

        pipeline = KairosPipeline()
        event = MagicMock()
        event.data = {"pair": [], "price": "2500.00"}
        await pipeline._on_price_observation(event)
        assert len(pipeline._price_observations) == 0


# ─── Nova Integration Tests ────────────────────────────────────────────────────


class TestNovaEconomicInvariantHandler:
    """Nova._on_economic_causal_invariant - belief update + EFE weight."""

    def _make_nova(self) -> "NovaService":
        from systems.nova.service import NovaService

        nova = NovaService.__new__(NovaService)
        nova._logger = structlog_stub()
        nova._efe_evaluator = MagicMock()
        nova._efe_evaluator.weights = MagicMock()
        nova._efe_evaluator.weights.pragmatic = 0.5
        nova._belief_updater = MagicMock()
        nova._belief_updater.upsert_entity = MagicMock()
        nova.update_efe_weights = MagicMock()
        return nova

    @pytest.mark.asyncio
    async def test_high_confidence_invariant_updates_efe(self) -> None:
        """Invariant with confidence ≥ 0.75 raises EFE pragmatic weight."""
        nova = self._make_nova()
        event = MagicMock()
        event.data = {
            "invariant_type": "protocol_success_rate",
            "cause": "protocol:aave",
            "effect": "high_success_probability",
            "confidence": 0.85,
            "direction": "positive",
            "sample_count": 20,
            "metadata": {"protocol": "aave"},
        }
        await nova._on_economic_causal_invariant(event)
        nova.update_efe_weights.assert_called_once()
        call_args = nova.update_efe_weights.call_args[0][0]
        assert "pragmatic" in call_args
        assert call_args["pragmatic"] > 0.5  # raised from baseline

    @pytest.mark.asyncio
    async def test_low_confidence_invariant_no_efe_change(self) -> None:
        """Invariant with confidence < 0.75 does not update EFE pragmatic weight."""
        nova = self._make_nova()
        event = MagicMock()
        event.data = {
            "invariant_type": "protocol_success_rate",
            "cause": "protocol:unknown",
            "effect": "high_success_probability",
            "confidence": 0.65,  # below 0.75 floor
            "direction": "positive",
            "sample_count": 5,
            "metadata": {},
        }
        await nova._on_economic_causal_invariant(event)
        nova.update_efe_weights.assert_not_called()

    @pytest.mark.asyncio
    async def test_invariant_stored_as_entity_belief(self) -> None:
        """Invariant above confidence 0.6 is stored in belief updater."""
        nova = self._make_nova()
        event = MagicMock()
        event.data = {
            "invariant_type": "time_of_week_effect",
            "cause": "weekday_vs_weekend",
            "effect": "roi_differential",
            "confidence": 0.80,
            "direction": "positive",
            "sample_count": 30,
            "metadata": {"relative_diff_pct": 45.0},
        }
        await nova._on_economic_causal_invariant(event)
        nova._belief_updater.upsert_entity.assert_called_once()
        call_arg = nova._belief_updater.upsert_entity.call_args[0][0]
        assert call_arg.entity_id == "causal_invariant.time_of_week_effect.weekday_vs_weekend"
        assert call_arg.confidence == pytest.approx(0.80)

    @pytest.mark.asyncio
    async def test_below_threshold_confidence_dropped(self) -> None:
        """Invariant with confidence < 0.60 is completely ignored."""
        nova = self._make_nova()
        event = MagicMock()
        event.data = {
            "invariant_type": "protocol_success_rate",
            "cause": "protocol:x",
            "effect": "something",
            "confidence": 0.50,
            "direction": "positive",
            "sample_count": 10,
            "metadata": {},
        }
        await nova._on_economic_causal_invariant(event)
        nova._belief_updater.upsert_entity.assert_not_called()
        nova.update_efe_weights.assert_not_called()


# ─── Oikos Episode Emission Tests ─────────────────────────────────────────────


class TestOikosEconomicEpisodeEmission:
    """OikosService._emit_economic_episode_to_memory - payload structure."""

    @pytest.mark.asyncio
    async def test_episode_event_emitted_with_required_fields(self) -> None:
        """Emitted OIKOS_ECONOMIC_EPISODE event has all causal annotation fields."""
        from systems.synapse.types import SynapseEventType

        emitted_events: list = []

        mock_bus = AsyncMock()
        mock_bus.emit = AsyncMock(side_effect=lambda e: emitted_events.append(e))

        mock_state = MagicMock()
        mock_state.starvation_level = MagicMock()
        mock_state.starvation_level.value = "nominal"
        mock_state.metabolic_efficiency = 1.2

        # Build a minimal OikosService-like object with just the needed methods
        from systems.oikos.service import OikosService

        svc = OikosService.__new__(OikosService)
        svc._event_bus = mock_bus
        svc._state = mock_state

        # Inject the method directly
        import types

        svc._emit_economic_episode_to_memory = types.MethodType(
            OikosService._emit_economic_episode_to_memory, svc
        )

        await svc._emit_economic_episode_to_memory(
            "yield_deploy",
            True,
            30.0,
            protocol="aave",
            capital_deployed_usd=500.0,
            gas_cost_usd=0.05,
            roi_pct=8.5,
        )

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.event_type == SynapseEventType.OIKOS_ECONOMIC_EPISODE
        assert event.source_system == "oikos"

        data = event.data
        assert data["action_type"] == "yield_deploy"
        assert data["success"] is True
        assert data["protocol"] == "aave"
        assert data["chain"] == "base"
        assert data["causal_substrate"] == "economic"
        assert "day_of_week" in data
        assert "hour_of_day" in data
        assert "causal_variable_importance" in data
        assert "causal_variables" in data
        assert data["roi_pct"] == pytest.approx(8.5)

    @pytest.mark.asyncio
    async def test_episode_not_emitted_when_no_event_bus(self) -> None:
        """Method is a no-op when _event_bus is None."""
        from systems.oikos.service import OikosService
        import types

        mock_state = MagicMock()
        mock_state.starvation_level = MagicMock()
        mock_state.starvation_level.value = "nominal"
        mock_state.metabolic_efficiency = 1.0

        svc = OikosService.__new__(OikosService)
        svc._event_bus = None
        svc._state = mock_state
        svc._emit_economic_episode_to_memory = types.MethodType(
            OikosService._emit_economic_episode_to_memory, svc
        )

        # Should complete without error
        await svc._emit_economic_episode_to_memory("bounty_hunt", False, 60.0)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def structlog_stub() -> MagicMock:
    """Return a structlog-compatible MagicMock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger
