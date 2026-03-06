"""Tests for the OneirosService orchestrator."""
import pytest

from systems.oneiros.service import OneirosService
from systems.oneiros.types import SleepStage


class TestOneirosService:
    @pytest.mark.asyncio
    async def test_initialization(self):
        svc = OneirosService()
        await svc.initialize()
        assert svc._initialized is True

    @pytest.mark.asyncio
    async def test_on_cycle_increments_pressure(self):
        svc = OneirosService()
        await svc.initialize()
        p0 = svc.sleep_pressure
        for _ in range(100):
            await svc.on_cycle()
        assert svc.sleep_pressure > p0

    @pytest.mark.asyncio
    async def test_initial_stage_is_wake(self):
        svc = OneirosService()
        await svc.initialize()
        assert svc.current_stage == SleepStage.WAKE

    @pytest.mark.asyncio
    async def test_is_sleeping_false_initially(self):
        svc = OneirosService()
        await svc.initialize()
        assert svc.is_sleeping is False

    @pytest.mark.asyncio
    async def test_degradation_zero_initially(self):
        svc = OneirosService()
        await svc.initialize()
        deg = svc.degradation
        assert deg.composite_impairment == 0.0
        assert deg.salience_noise == 0.0

    @pytest.mark.asyncio
    async def test_creative_goal_set(self):
        svc = OneirosService()
        await svc.initialize()
        svc.set_creative_goal("Explore pattern recognition")
        assert svc._creative_goal == "Explore pattern recognition"

    @pytest.mark.asyncio
    async def test_get_pending_insights_empty(self):
        svc = OneirosService()
        await svc.initialize()
        insights = svc.get_pending_insights()
        assert insights == []

    @pytest.mark.asyncio
    async def test_stats(self):
        svc = OneirosService()
        await svc.initialize()
        stats = svc.stats
        assert "total_sleep_cycles" in stats
        assert "total_dreams" in stats
        assert "current_pressure" in stats
        assert "current_stage" in stats
        assert stats["current_stage"] == "wake"

    @pytest.mark.asyncio
    async def test_health_snapshot(self):
        svc = OneirosService()
        await svc.initialize()
        health = await svc.health()
        assert "status" in health
        assert "current_stage" in health
        assert "sleep_pressure" in health

    @pytest.mark.asyncio
    async def test_set_cross_system_references(self):
        svc = OneirosService()
        svc.set_equor("mock_equor")
        svc.set_evo("mock_evo")
        svc.set_nova("mock_nova")
        svc.set_atune("mock_atune")
        svc.set_thymos("mock_thymos")
        svc.set_memory("mock_memory")
        assert svc._equor == "mock_equor"
        assert svc._evo == "mock_evo"
        assert svc._nova == "mock_nova"

    @pytest.mark.asyncio
    async def test_shutdown(self):
        svc = OneirosService()
        await svc.initialize()
        await svc.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_episode_stored(self):
        svc = OneirosService()
        await svc.initialize()
        await svc.on_episode_stored(0.9, 0.8)  # High affect
        assert svc._clock.pressure.unprocessed_affect_residue > 0

    @pytest.mark.asyncio
    async def test_system_id(self):
        svc = OneirosService()
        assert svc.system_id == "oneiros"
