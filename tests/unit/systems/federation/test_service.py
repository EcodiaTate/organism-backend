"""
Unit tests for the FederationService orchestrator.

Tests initialization, link management, and health reporting.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config import FederationConfig
from systems.federation.service import FederationService

# ─── Fixtures ────────────────────────────────────────────────────


def make_config(**kwargs) -> FederationConfig:
    defaults = {
        "enabled": True,
        "endpoint": "https://localhost:8002",
    }
    return FederationConfig(**{**defaults, **kwargs})


def make_disabled_config() -> FederationConfig:
    return FederationConfig(enabled=False)


def make_redis() -> MagicMock:
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    return redis


def make_metrics() -> MagicMock:
    metrics = MagicMock()
    metrics.record = AsyncMock()
    return metrics


# ─── Initialization ─────────────────────────────────────────────


class TestFederationInitialization:
    """Test federation service initialization."""

    @pytest.mark.asyncio
    async def test_disabled_federation_initializes(self):
        service = FederationService(
            config=make_disabled_config(),
            instance_id="test-001",
        )
        await service.initialize()

        assert service._initialized
        assert service.identity_card is None  # Not built when disabled

    @pytest.mark.asyncio
    async def test_enabled_federation_builds_subsystems(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            metrics=make_metrics(),
            instance_id="test-001",
        )
        await service.initialize()

        assert service._initialized
        assert service._identity is not None
        assert service._trust is not None
        assert service._privacy is not None
        assert service._knowledge is not None
        assert service._coordination is not None
        assert service._channels is not None

    @pytest.mark.asyncio
    async def test_identity_card_created(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()

        card = service.identity_card
        assert card is not None
        assert card.instance_id == "test-001"
        assert card.protocol_version == "1.0"

    @pytest.mark.asyncio
    async def test_double_initialize_is_noop(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()
        await service.initialize()  # Should not error


# ─── Health ──────────────────────────────────────────────────────


class TestFederationHealth:
    """Test health reporting."""

    @pytest.mark.asyncio
    async def test_health_when_disabled(self):
        service = FederationService(
            config=make_disabled_config(),
            instance_id="test-001",
        )
        await service.initialize()

        health = await service.health()
        assert health["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_health_when_enabled(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()

        health = await service.health()
        assert health["status"] == "healthy"
        assert health["enabled"] is True
        assert health["active_links"] == 0

    @pytest.mark.asyncio
    async def test_stats_comprehensive(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()

        stats = service.stats
        assert "initialized" in stats
        assert "enabled" in stats
        assert "active_links" in stats
        assert "identity" in stats
        assert "trust" in stats
        assert "knowledge" in stats
        assert "coordination" in stats
        assert "channels" in stats
        assert "privacy" in stats


# ─── Link Queries ────────────────────────────────────────────────


class TestLinkQueries:
    """Test link query methods."""

    @pytest.mark.asyncio
    async def test_active_links_empty_initially(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()

        assert service.active_links == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_link_returns_none(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()

        assert service.get_link("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_link_by_instance_returns_none(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()

        assert service.get_link_by_instance("nonexistent") is None


# ─── Establish Link (failure cases) ─────────────────────────────


class TestEstablishLink:
    """Test link establishment failure paths."""

    @pytest.mark.asyncio
    async def test_disabled_federation_rejects_link(self):
        service = FederationService(
            config=make_disabled_config(),
            instance_id="test-001",
        )
        await service.initialize()

        result = await service.establish_link("https://remote:8002")
        assert "error" in result
        assert "disabled" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_no_endpoint_returns_error(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()

        # Connection will fail since no actual server
        result = await service.establish_link("https://nonexistent:8002")
        assert "error" in result


# ─── Withdraw Link ──────────────────────────────────────────────


class TestWithdrawLink:
    """Test link withdrawal."""

    @pytest.mark.asyncio
    async def test_withdraw_nonexistent_returns_error(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()

        result = await service.withdraw_link("nonexistent")
        assert "error" in result


# ─── Shutdown ────────────────────────────────────────────────────


class TestFederationShutdown:
    """Test graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_completes(self):
        service = FederationService(
            config=make_config(),
            redis=make_redis(),
            instance_id="test-001",
        )
        await service.initialize()
        await service.shutdown()  # Should not error

    @pytest.mark.asyncio
    async def test_shutdown_when_disabled(self):
        service = FederationService(
            config=make_disabled_config(),
            instance_id="test-001",
        )
        await service.initialize()
        await service.shutdown()  # Should not error
