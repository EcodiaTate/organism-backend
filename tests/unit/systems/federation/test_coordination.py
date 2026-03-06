"""
Unit tests for Federation Coordinated Action.

Tests assistance request handling, trust gating, and interaction recording.
"""

from __future__ import annotations

import pytest

from primitives.federation import (
    AssistanceRequest,
    FederationLink,
    FederationLinkStatus,
    InteractionOutcome,
    TrustLevel,
)
from systems.federation.coordination import CoordinationManager

# ─── Fixtures ────────────────────────────────────────────────────


def make_link(
    trust_level: TrustLevel = TrustLevel.COLLEAGUE,
    **kwargs,
) -> FederationLink:
    defaults = {
        "local_instance_id": "local-001",
        "remote_instance_id": "remote-001",
        "remote_endpoint": "https://remote:8002",
        "trust_level": trust_level,
        "trust_score": float(trust_level.value * 10),
        "status": FederationLinkStatus.ACTIVE,
    }
    return FederationLink(**{**defaults, **kwargs})


def make_request(description: str = "Help with farming patterns") -> AssistanceRequest:
    return AssistanceRequest(
        requesting_instance_id="remote-001",
        description=description,
        knowledge_domain="agriculture",
        urgency=0.5,
    )


# ─── Trust Gating ────────────────────────────────────────────────


class TestTrustGating:
    """Test that coordination requires COLLEAGUE+ trust."""

    @pytest.mark.asyncio
    async def test_colleague_can_request_assistance(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.COLLEAGUE)
        request = make_request()

        response, interaction = await manager.handle_request(request, link)

        assert response.accepted
        assert interaction.outcome == InteractionOutcome.SUCCESSFUL

    @pytest.mark.asyncio
    async def test_acquaintance_cannot_request_assistance(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        request = make_request()

        response, interaction = await manager.handle_request(request, link)

        assert not response.accepted
        assert "insufficient trust" in response.reason.lower()

    @pytest.mark.asyncio
    async def test_none_trust_cannot_request_assistance(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.NONE)
        request = make_request()

        response, interaction = await manager.handle_request(request, link)

        assert not response.accepted

    @pytest.mark.asyncio
    async def test_partner_can_request_assistance(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.PARTNER)
        request = make_request()

        response, interaction = await manager.handle_request(request, link)

        assert response.accepted


# ─── Equor & Nova Gates ──────────────────────────────────────────


class TestConstitutionalGates:
    """Test that Equor and Nova can block assistance."""

    @pytest.mark.asyncio
    async def test_equor_denial_blocks(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.ALLY)
        request = make_request()

        response, interaction = await manager.handle_request(
            request, link, equor_permitted=False
        )

        assert not response.accepted
        assert "constitutional" in response.reason.lower()

    @pytest.mark.asyncio
    async def test_nova_misalignment_blocks(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.ALLY)
        request = make_request()

        response, interaction = await manager.handle_request(
            request, link, nova_aligned=False
        )

        assert not response.accepted
        assert "priorities" in response.reason.lower()


# ─── Interaction Recording ───────────────────────────────────────


class TestInteractionRecording:
    """Test that assistance interactions are properly recorded."""

    @pytest.mark.asyncio
    async def test_accepted_builds_significant_trust(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.COLLEAGUE)
        request = make_request()

        response, interaction = await manager.handle_request(request, link)

        assert interaction.trust_value == 2.0  # Providing assistance is worth 2x
        assert interaction.outcome == InteractionOutcome.SUCCESSFUL

    @pytest.mark.asyncio
    async def test_declined_records_half_trust_value(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        request = make_request()

        response, interaction = await manager.handle_request(request, link)

        assert interaction.trust_value == 0.5
        assert interaction.outcome == InteractionOutcome.FAILED


# ─── Active Assistance Tracking ──────────────────────────────────


class TestActiveAssistance:
    """Test tracking of in-progress assistance requests."""

    @pytest.mark.asyncio
    async def test_accepted_request_tracked(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.COLLEAGUE)
        request = make_request()

        await manager.handle_request(request, link)

        assert request.id in manager._active_assistance
        assert manager.stats["active_assistance"] == 1

    @pytest.mark.asyncio
    async def test_completed_assistance_removed(self):
        manager = CoordinationManager()
        link = make_link(trust_level=TrustLevel.COLLEAGUE)
        request = make_request()

        await manager.handle_request(request, link)
        manager.complete_assistance(request.id)

        assert request.id not in manager._active_assistance
        assert manager.stats["active_assistance"] == 0


# ─── Stats ───────────────────────────────────────────────────────


class TestCoordinationStats:
    """Test coordination statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_accumulate(self):
        manager = CoordinationManager()

        # One accepted
        link = make_link(trust_level=TrustLevel.COLLEAGUE)
        await manager.handle_request(make_request(), link)

        # One declined (low trust)
        link_low = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        await manager.handle_request(make_request(), link_low)

        stats = manager.stats
        assert stats["requests_received"] == 2
        assert stats["requests_accepted"] == 1
        assert stats["requests_declined"] == 1

    def test_build_request_increments_sent(self):
        manager = CoordinationManager()
        manager.build_request("Help me", local_instance_id="local-001")
        assert manager.stats["requests_sent"] == 1
