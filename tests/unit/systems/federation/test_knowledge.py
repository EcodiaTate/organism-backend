"""
Unit tests for Federation Knowledge Exchange.

Tests trust-gated sharing, privacy filtering, and request/response handling.
"""

from __future__ import annotations

import pytest

from primitives.federation import (
    FederationLink,
    FederationLinkStatus,
    InteractionOutcome,
    KnowledgeRequest,
    KnowledgeType,
    TrustLevel,
)
from systems.federation.knowledge import KnowledgeExchangeManager
from systems.federation.privacy import PrivacyFilter

# ─── Fixtures ────────────────────────────────────────────────────


def make_link(
    trust_level: TrustLevel = TrustLevel.ACQUAINTANCE,
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


def make_request(
    knowledge_type: KnowledgeType = KnowledgeType.PUBLIC_ENTITIES,
    query: str = "test query",
) -> KnowledgeRequest:
    return KnowledgeRequest(
        requesting_instance_id="remote-001",
        knowledge_type=knowledge_type,
        query=query,
    )


# ─── Trust-Gated Sharing ────────────────────────────────────────


class TestTrustGatedSharing:
    """Test that knowledge sharing is properly gated by trust level."""

    @pytest.mark.asyncio
    async def test_acquaintance_can_get_public_entities(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        request = make_request(knowledge_type=KnowledgeType.PUBLIC_ENTITIES)

        response, interaction = await manager.handle_request(request, link)
        # Should be granted (even if no data comes back — trust level is correct)
        assert response.granted

    @pytest.mark.asyncio
    async def test_acquaintance_cannot_get_procedures(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        request = make_request(knowledge_type=KnowledgeType.PROCEDURES)

        response, interaction = await manager.handle_request(request, link)
        assert not response.granted
        assert "trust level" in response.reason.lower()

    @pytest.mark.asyncio
    async def test_colleague_can_get_procedures(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.COLLEAGUE)
        request = make_request(knowledge_type=KnowledgeType.PROCEDURES)

        response, interaction = await manager.handle_request(request, link)
        assert response.granted

    @pytest.mark.asyncio
    async def test_none_trust_gets_nothing(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.NONE)
        request = make_request(knowledge_type=KnowledgeType.PUBLIC_ENTITIES)

        response, interaction = await manager.handle_request(request, link)
        assert not response.granted

    @pytest.mark.asyncio
    async def test_partner_can_get_hypotheses(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.PARTNER)
        request = make_request(knowledge_type=KnowledgeType.HYPOTHESES)

        response, interaction = await manager.handle_request(request, link)
        assert response.granted

    @pytest.mark.asyncio
    async def test_only_ally_can_get_schema(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        # Partner cannot
        link_partner = make_link(trust_level=TrustLevel.PARTNER)
        request = make_request(knowledge_type=KnowledgeType.SCHEMA_STRUCTURES)
        response, _ = await manager.handle_request(request, link_partner)
        assert not response.granted

        # Ally can
        link_ally = make_link(trust_level=TrustLevel.ALLY)
        response, _ = await manager.handle_request(request, link_ally)
        assert response.granted


# ─── Equor Review Gate ───────────────────────────────────────────


class TestEquorReviewGate:
    """Test that Equor can block knowledge sharing."""

    @pytest.mark.asyncio
    async def test_equor_denial_blocks_sharing(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.ALLY)
        request = make_request(knowledge_type=KnowledgeType.PUBLIC_ENTITIES)

        response, interaction = await manager.handle_request(
            request, link, equor_permitted=False
        )
        assert not response.granted
        assert "constitutional" in response.reason.lower()
        assert interaction.outcome == InteractionOutcome.FAILED


# ─── Interaction Records ────────────────────────────────────────


class TestInteractionRecords:
    """Test that interactions are correctly recorded for trust scoring."""

    @pytest.mark.asyncio
    async def test_successful_interaction_recorded(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        request = make_request(knowledge_type=KnowledgeType.PUBLIC_ENTITIES)

        response, interaction = await manager.handle_request(request, link)

        assert interaction.outcome == InteractionOutcome.SUCCESSFUL
        assert interaction.interaction_type == "knowledge_request"
        assert interaction.direction == "inbound"

    @pytest.mark.asyncio
    async def test_denied_interaction_recorded(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.NONE)
        request = make_request(knowledge_type=KnowledgeType.PUBLIC_ENTITIES)

        response, interaction = await manager.handle_request(request, link)

        assert interaction.outcome == InteractionOutcome.FAILED


# ─── Stats ───────────────────────────────────────────────────────


class TestKnowledgeStats:
    """Test knowledge exchange statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_track_requests(self):
        manager = KnowledgeExchangeManager(
            memory=None,
            privacy_filter=PrivacyFilter(),
        )

        link = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        request = make_request(knowledge_type=KnowledgeType.PUBLIC_ENTITIES)
        await manager.handle_request(request, link)

        denied_request = make_request(knowledge_type=KnowledgeType.PROCEDURES)
        await manager.handle_request(denied_request, link)

        stats = manager.stats
        assert stats["requests_received"] == 2
        assert stats["requests_granted"] == 1
        assert stats["requests_denied"] == 1
