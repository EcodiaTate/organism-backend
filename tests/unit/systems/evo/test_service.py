"""
Unit tests for EvoService.

Tests the top-level service interface: initialization, broadcast handling,
parameter querying, and consolidation triggering.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config import EvoConfig
from primitives.affect import AffectState
from systems.fovea.types import (
    MemoryContext,
    SalienceVector,
    WorkspaceBroadcast,
    WorkspaceContext,
)
from systems.evo.service import EvoService
from systems.evo.types import (
    PARAMETER_DEFAULTS,
    ConsolidationResult,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_config(**kwargs) -> EvoConfig:
    defaults = {
        "consolidation_interval_hours": 6,
        "consolidation_cycle_threshold": 10_000,
        "max_active_hypotheses": 50,
        "max_parameter_delta_per_cycle": 0.03,
        "min_evidence_for_integration": 10,
    }
    return EvoConfig(**{**defaults, **kwargs})


def make_llm() -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.text = '{"hypotheses": []}'
    llm.generate = AsyncMock(return_value=response)
    llm.evaluate = AsyncMock(return_value=response)
    return llm


def make_broadcast(
    *,
    affect: AffectState | None = None,
    entity_ids: list[str] | None = None,
) -> WorkspaceBroadcast:
    affect = affect or AffectState.neutral()
    traces = []
    if entity_ids:
        trace = MagicMock()
        trace.entities = entity_ids
        traces = [trace]

    ctx = WorkspaceContext(
        memory_context=MemoryContext(traces=traces),
    )
    return WorkspaceBroadcast(
        content=None,
        salience=SalienceVector(scores={}, composite=0.5),
        affect=affect,
        context=ctx,
    )


async def make_initialized_service(**config_kwargs) -> EvoService:
    service = EvoService(
        config=make_config(**config_kwargs),
        llm=make_llm(),
        memory=None,
        instance_name="TestEOS",
    )
    await service.initialize()
    return service


# ─── Tests: Initialization ────────────────────────────────────────────────────


class TestInitialization:
    @pytest.mark.asyncio
    async def test_initializes_without_error(self):
        service = await make_initialized_service()
        assert service.stats["initialized"] is True

    @pytest.mark.asyncio
    async def test_double_initialize_is_idempotent(self):
        service = await make_initialized_service()
        await service.initialize()  # Second call should be a no-op
        assert service.stats["initialized"] is True

    @pytest.mark.asyncio
    async def test_detectors_built(self):
        service = await make_initialized_service()
        assert len(service._detectors) == 4

    @pytest.mark.asyncio
    async def test_default_parameters_available(self):
        service = await make_initialized_service()
        for name, default in PARAMETER_DEFAULTS.items():
            assert service.get_parameter(name) == pytest.approx(default)

    @pytest.mark.asyncio
    async def test_returns_none_for_uninitialized(self):
        service = EvoService(
            config=make_config(),
            llm=make_llm(),
            memory=None,
        )
        # Not initialized - get_parameter should return None
        assert service.get_parameter("nova.efe.pragmatic") is None


# ─── Tests: receive_broadcast ─────────────────────────────────────────────────


class TestReceiveBroadcast:
    @pytest.mark.asyncio
    async def test_increments_broadcast_counter(self):
        service = await make_initialized_service()
        assert service.stats["total_broadcasts"] == 0
        await service.receive_broadcast(make_broadcast())
        assert service.stats["total_broadcasts"] == 1

    @pytest.mark.asyncio
    async def test_multiple_broadcasts(self):
        service = await make_initialized_service()
        for _ in range(5):
            await service.receive_broadcast(make_broadcast())
        assert service.stats["total_broadcasts"] == 5

    @pytest.mark.asyncio
    async def test_updates_affect_context(self):
        service = await make_initialized_service()
        affect = AffectState(valence=0.7, arousal=0.5)
        await service.receive_broadcast(make_broadcast(affect=affect))
        assert service._pattern_context.current_affect is not None
        assert service._pattern_context.current_affect.valence == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_tracks_previous_affect(self):
        service = await make_initialized_service()
        affect_1 = AffectState(valence=0.2, arousal=0.3)
        affect_2 = AffectState(valence=0.7, arousal=0.5)

        await service.receive_broadcast(make_broadcast(affect=affect_1))
        await service.receive_broadcast(make_broadcast(affect=affect_2))

        assert service._pattern_context.previous_affect is not None
        assert service._pattern_context.previous_affect.valence == pytest.approx(0.2)
        assert service._pattern_context.current_affect.valence == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_updates_entity_ids(self):
        service = await make_initialized_service()
        await service.receive_broadcast(
            make_broadcast(entity_ids=["entity_a", "entity_b"])
        )
        assert "entity_a" in service._pattern_context.recent_entity_ids
        assert "entity_b" in service._pattern_context.recent_entity_ids

    @pytest.mark.asyncio
    async def test_does_not_raise_when_not_initialized(self):
        service = EvoService(config=make_config(), llm=make_llm(), memory=None)
        # Should silently return, not raise
        await service.receive_broadcast(make_broadcast())

    @pytest.mark.asyncio
    async def test_does_not_raise_on_detector_error(self):
        """Evo failures must not interrupt the cognitive cycle."""
        service = await make_initialized_service()
        # Break a detector
        broken = MagicMock()
        broken.scan = AsyncMock(side_effect=RuntimeError("detector broken"))
        broken.name = "broken"
        service._detectors = [broken]

        await service.receive_broadcast(make_broadcast())
        # Counter still incremented - no exception propagated
        assert service.stats["total_broadcasts"] == 1


# ─── Tests: get_parameter ─────────────────────────────────────────────────────


class TestGetParameter:
    @pytest.mark.asyncio
    async def test_returns_known_parameter(self):
        service = await make_initialized_service()
        value = service.get_parameter("nova.efe.pragmatic")
        assert value == pytest.approx(PARAMETER_DEFAULTS["nova.efe.pragmatic"])

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown(self):
        service = await make_initialized_service()
        assert service.get_parameter("does.not.exist") is None

    @pytest.mark.asyncio
    async def test_get_all_parameters_returns_dict(self):
        service = await make_initialized_service()
        params = service.get_all_parameters()
        assert isinstance(params, dict)
        assert len(params) == len(PARAMETER_DEFAULTS)


# ─── Tests: run_consolidation ─────────────────────────────────────────────────


class TestRunConsolidation:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_initialized(self):
        service = EvoService(config=make_config(), llm=make_llm(), memory=None)
        result = await service.run_consolidation()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_consolidation_result(self):
        service = await make_initialized_service()
        result = await service.run_consolidation()
        assert result is not None
        assert isinstance(result, ConsolidationResult)

    @pytest.mark.asyncio
    async def test_increments_consolidation_counter(self):
        service = await make_initialized_service()
        assert service._total_consolidations == 0
        await service.run_consolidation()
        assert service._total_consolidations == 1

    @pytest.mark.asyncio
    async def test_resets_cycle_counter(self):
        service = await make_initialized_service()
        # Simulate cycles having passed
        service._cycles_since_consolidation = 5000
        await service.run_consolidation()
        assert service._cycles_since_consolidation == 0


# ─── Tests: shutdown ──────────────────────────────────────────────────────────


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_when_not_initialized(self):
        service = EvoService(config=make_config(), llm=make_llm(), memory=None)
        await service.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_shutdown_after_initialization(self):
        service = await make_initialized_service()
        await service.shutdown()  # Should not raise


# ─── Tests: Stats ─────────────────────────────────────────────────────────────


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_structure(self):
        service = await make_initialized_service()
        stats = service.stats
        assert "initialized" in stats
        assert "total_broadcasts" in stats
        assert "total_consolidations" in stats
        assert "hypothesis" in stats
        assert "parameter_tuner" in stats

    @pytest.mark.asyncio
    async def test_stats_reflect_activity(self):
        service = await make_initialized_service()
        for _ in range(3):
            await service.receive_broadcast(make_broadcast())
        assert service.stats["total_broadcasts"] == 3
