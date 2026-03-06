"""Tests for Soma Service — the full interoceptive substrate orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config import SomaConfig
from systems.soma.service import SomaService
from systems.soma.types import (
    ALL_DIMENSIONS,
    AllostaticSignal,
    DevelopmentalStage,
    InteroceptiveDimension,
    InteroceptiveState,
    SomaticMarker,
)


def _make_config(**overrides) -> SomaConfig:
    defaults = {
        "cycle_enabled": True,
        "phase_space_update_interval": 5,  # Small for tests
        "trajectory_buffer_size": 50,
        "prediction_ewm_span": 5,
        "setpoint_adaptation_alpha": 0.05,
        "urgency_threshold": 0.3,
        "attractor_min_dwell_cycles": 5,
        "bifurcation_detection_enabled": True,
        "max_attractors": 20,
        "somatic_marker_enabled": True,
        "somatic_rerank_boost": 0.3,
        "developmental_gating_enabled": True,
        "initial_stage": "reflexive",
    }
    defaults.update(overrides)
    return SomaConfig(**defaults)


def _mock_atune():
    affect = MagicMock()
    affect.valence = 0.2
    affect.arousal = 0.4
    affect.curiosity = 0.5
    affect.care_activation = 0.3
    affect.coherence_stress = 0.1

    mgr = MagicMock()
    mgr.current = affect

    atune = MagicMock()
    atune._affect_mgr = mgr
    atune.mean_prediction_error = 0.2
    return atune


def _make_service(**overrides) -> SomaService:
    config = _make_config(**overrides)
    return SomaService(config=config)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self):
        service = _make_service()
        await service.initialize()
        assert service.cycle_count == 0
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_health(self):
        service = _make_service()
        await service.initialize()
        health = await service.health()
        assert health["status"] == "healthy"
        assert "cycle_count" in health
        assert "stage" in health


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_run_cycle_returns_signal(self):
        service = _make_service()
        await service.initialize()
        signal = await service.run_cycle()
        assert isinstance(signal, AllostaticSignal)
        assert service.cycle_count == 1

    @pytest.mark.asyncio
    async def test_run_cycle_with_systems(self):
        service = _make_service()
        service.set_atune(_mock_atune())
        await service.initialize()
        signal = await service.run_cycle()
        assert isinstance(signal, AllostaticSignal)
        # Should have sensed arousal from mock
        state = service.get_current_state()
        assert state is not None
        assert state.sensed[InteroceptiveDimension.AROUSAL] == 0.4

    @pytest.mark.asyncio
    async def test_multiple_cycles(self):
        service = _make_service()
        service.set_atune(_mock_atune())
        await service.initialize()
        for _ in range(10):
            signal = await service.run_cycle()
        assert service.cycle_count == 10
        assert isinstance(signal, AllostaticSignal)

    @pytest.mark.asyncio
    async def test_disabled_returns_default_signal(self):
        service = _make_service(cycle_enabled=False)
        await service.initialize()
        signal = await service.run_cycle()
        # Should return default signal
        assert signal.urgency == 0.0


class TestQueryMethods:
    @pytest.mark.asyncio
    async def test_get_current_state_none_before_cycle(self):
        service = _make_service()
        await service.initialize()
        assert service.get_current_state() is None

    @pytest.mark.asyncio
    async def test_get_current_state_after_cycle(self):
        service = _make_service()
        await service.initialize()
        await service.run_cycle()
        state = service.get_current_state()
        assert state is not None
        assert isinstance(state, InteroceptiveState)

    @pytest.mark.asyncio
    async def test_get_current_signal(self):
        service = _make_service()
        await service.initialize()
        signal = service.get_current_signal()
        assert isinstance(signal, AllostaticSignal)

    @pytest.mark.asyncio
    async def test_get_errors_empty_before_cycle(self):
        service = _make_service()
        await service.initialize()
        errors = service.get_errors()
        assert errors == {}

    @pytest.mark.asyncio
    async def test_get_phase_position(self):
        service = _make_service()
        await service.initialize()
        pos = service.get_phase_position()
        assert "trajectory_heading" in pos

    @pytest.mark.asyncio
    async def test_get_developmental_stage(self):
        service = _make_service()
        await service.initialize()
        stage = service.get_developmental_stage()
        assert stage == DevelopmentalStage.REFLEXIVE


class TestSomaticMarker:
    @pytest.mark.asyncio
    async def test_create_marker_before_cycle(self):
        service = _make_service()
        await service.initialize()
        marker = service.create_somatic_marker()
        assert isinstance(marker, SomaticMarker)

    @pytest.mark.asyncio
    async def test_create_marker_after_cycle(self):
        service = _make_service()
        await service.initialize()
        await service.run_cycle()
        marker = service.create_somatic_marker()
        assert isinstance(marker, SomaticMarker)
        vec = marker.to_vector()
        assert len(vec) == 19


class TestSomaticRerank:
    @pytest.mark.asyncio
    async def test_rerank_no_state(self):
        service = _make_service()
        await service.initialize()
        result = service.somatic_rerank([{"salience_score": 0.5}])
        # Should return unchanged without a current state
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_rerank_after_cycle(self):
        service = _make_service()
        await service.initialize()
        await service.run_cycle()
        result = service.somatic_rerank([{"salience_score": 0.5}])
        assert len(result) == 1


class TestEvoIntegration:
    @pytest.mark.asyncio
    async def test_update_dynamics_matrix(self):
        service = _make_service()
        await service.initialize()
        n = len(ALL_DIMENSIONS)
        new_dynamics = [[0.5 if i == j else 0.0 for j in range(n)] for i in range(n)]
        service.update_dynamics_matrix(new_dynamics)
        # Verify it propagated
        assert service._predictor._dynamics == new_dynamics

    @pytest.mark.asyncio
    async def test_update_emotion_regions(self):
        service = _make_service()
        await service.initialize()
        service.update_emotion_regions({"new_emotion": {"pattern": {}}})
        assert "new_emotion" in service._emotion_regions


class TestContextManagement:
    @pytest.mark.asyncio
    async def test_set_context_reflexive_ignored(self):
        """At REFLEXIVE stage, setpoint adaptation is disabled."""
        service = _make_service(initial_stage="reflexive")
        await service.initialize()
        # Should not crash
        service.set_context("conversation")

    @pytest.mark.asyncio
    async def test_set_context_deliberative(self):
        """At DELIBERATIVE stage, context switching is active."""
        service = _make_service(initial_stage="deliberative")
        await service.initialize()
        service.set_context("deep_processing")
        # Should not crash and controller should have the context


class TestDegradation:
    @pytest.mark.asyncio
    async def test_error_in_cycle_returns_default_signal(self):
        """If sensing fails, Soma should emit default signal (graceful degradation)."""
        service = _make_service()
        await service.initialize()
        # Break the interoceptor
        service._interoceptor.sense = MagicMock(side_effect=RuntimeError("broken"))
        signal = await service.run_cycle()
        # Should get default signal, not a crash
        assert isinstance(signal, AllostaticSignal)
        assert signal.urgency == 0.0
