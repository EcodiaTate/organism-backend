"""
Unit tests for Federation Reputation Staking.

Tests bond creation, contradiction detection, forfeit, recovery,
budget enforcement, and stats.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from primitives.common import new_id, utc_now
from primitives.federation import (
    BondStatus,
    ContradictionEvidence,
    FederationLink,
    FederationLinkStatus,
    InstanceIdentityCard,
    KnowledgeItem,
    KnowledgeType,
    PrivacyLevel,
    TrustLevel,
)
from systems.federation.reputation_staking import ReputationStakingManager

# ─── Fixtures ────────────────────────────────────────────────────


def make_link(
    trust_level: TrustLevel = TrustLevel.COLLEAGUE,
    remote_instance_id: str = "remote-001",
    wallet_address: str = "0xABCD1234",
    **kwargs: object,
) -> FederationLink:
    defaults = {
        "local_instance_id": "local-001",
        "remote_instance_id": remote_instance_id,
        "remote_endpoint": "https://remote:8002",
        "trust_level": trust_level,
        "trust_score": float(trust_level.value * 10),
        "status": FederationLinkStatus.ACTIVE,
        "remote_identity": InstanceIdentityCard(
            instance_id=remote_instance_id,
            name="Remote Instance",
            wallet_address=wallet_address,
        ),
    }
    return FederationLink(**{**defaults, **kwargs})  # type: ignore[arg-type]


def make_item(
    content: dict | None = None,
    embedding: list[float] | None = None,
) -> KnowledgeItem:
    return KnowledgeItem(
        item_id=new_id(),
        knowledge_type=KnowledgeType.PUBLIC_ENTITIES,
        privacy_level=PrivacyLevel.PUBLIC,
        content=content or {"summary": "Test knowledge claim", "confidence": 0.8},
        embedding=embedding,
    )


def make_manager(
    wallet: AsyncMock | None = None,
    escrow_address: str = "0xESCROW",
    base_bond: Decimal = Decimal("10.00"),
    max_total: Decimal = Decimal("100.00"),
    max_per_instance: Decimal = Decimal("50.00"),
) -> ReputationStakingManager:
    config = MagicMock()
    config.base_bond_usdc = base_bond
    config.max_total_bonded_usdc = max_total
    config.max_per_instance_bonded_usdc = max_per_instance
    config.bond_expiry_days = 90
    config.escrow_address = escrow_address
    config.contradiction_similarity_threshold = 0.85
    config.contradiction_divergence_threshold = 0.3
    config.min_certainty_for_bond = 0.1
    config.tier_discounts = {
        "ALLY": 0.5,
        "PARTNER": 0.75,
        "COLLEAGUE": 1.0,
        "ACQUAINTANCE": 1.25,
    }

    return ReputationStakingManager(
        wallet=wallet,
        redis=None,
        metrics=None,
        config=config,
        escrow_address=escrow_address,
    )


def make_wallet_mock() -> AsyncMock:
    wallet = AsyncMock()
    tx_result = MagicMock()
    tx_result.tx_hash = "0xTX_HASH_ABC123"
    wallet.transfer = AsyncMock(return_value=tx_result)
    return wallet


# ─── Bond Creation ─────────────────────────────────────────────


class TestBondCreation:
    """Test bond creation including amount scaling, budgets, and failures."""

    @pytest.mark.asyncio
    async def test_bond_amount_scales_with_certainty(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet, base_bond=Decimal("10.00"))
        link = make_link(trust_level=TrustLevel.COLLEAGUE)
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.5)

        assert bond is not None
        # base=10 * certainty=0.5 * tier(COLLEAGUE)=1.0 = 5.00
        assert bond.bond_amount_usdc == Decimal("5.00")
        assert bond.status == BondStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_bond_amount_scales_with_full_certainty(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet, base_bond=Decimal("10.00"))
        link = make_link(trust_level=TrustLevel.COLLEAGUE)
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=1.0)

        assert bond is not None
        assert bond.bond_amount_usdc == Decimal("10.00")

    @pytest.mark.asyncio
    async def test_bond_tiered_discount_ally(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet, base_bond=Decimal("10.00"))
        link = make_link(trust_level=TrustLevel.ALLY)
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=1.0)

        assert bond is not None
        # base=10 * certainty=1.0 * tier(ALLY)=0.5 = 5.00
        assert bond.bond_amount_usdc == Decimal("5.00")

    @pytest.mark.asyncio
    async def test_bond_tiered_discount_acquaintance(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet, base_bond=Decimal("10.00"))
        link = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=1.0)

        assert bond is not None
        # base=10 * certainty=1.0 * tier(ACQUAINTANCE)=1.25 = 12.50
        assert bond.bond_amount_usdc == Decimal("12.50")

    @pytest.mark.asyncio
    async def test_bond_not_created_below_min_certainty(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.05)

        assert bond is None
        wallet.transfer.assert_not_called()

    @pytest.mark.asyncio
    async def test_bond_not_created_without_escrow_address(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet, escrow_address="")
        link = make_link()
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.8)

        assert bond is None

    @pytest.mark.asyncio
    async def test_budget_check_max_total(self):
        wallet = make_wallet_mock()
        manager = make_manager(
            wallet=wallet,
            base_bond=Decimal("60.00"),
            max_total=Decimal("100.00"),
            max_per_instance=Decimal("200.00"),  # High enough to not interfere
        )
        link = make_link(trust_level=TrustLevel.COLLEAGUE)

        # First bond: 60 * 1.0 * 1.0 = 60.00 -> total = 60.00
        item1 = make_item()
        bond1 = await manager.create_bond(item1, link, claim_certainty=1.0)
        assert bond1 is not None
        assert bond1.bond_amount_usdc == Decimal("60.00")

        # Second bond: 60 * 1.0 * 1.0 = 60.00 -> total would be 120 > 100
        item2 = make_item()
        bond2 = await manager.create_bond(item2, link, claim_certainty=1.0)
        assert bond2 is None

    @pytest.mark.asyncio
    async def test_budget_check_max_per_instance(self):
        wallet = make_wallet_mock()
        manager = make_manager(
            wallet=wallet,
            base_bond=Decimal("30.00"),
            max_total=Decimal("200.00"),
            max_per_instance=Decimal("50.00"),
        )
        link = make_link(trust_level=TrustLevel.COLLEAGUE)

        # First bond: 30.00
        bond1 = await manager.create_bond(make_item(), link, claim_certainty=1.0)
        assert bond1 is not None

        # Second bond: would bring instance total to 60 > 50
        bond2 = await manager.create_bond(make_item(), link, claim_certainty=1.0)
        assert bond2 is None

    @pytest.mark.asyncio
    async def test_bond_creation_wallet_failure_returns_escrow_failed(self):
        wallet = make_wallet_mock()
        wallet.transfer = AsyncMock(side_effect=Exception("Insufficient funds"))
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.8)

        assert bond is not None
        assert bond.status == BondStatus.ESCROW_FAILED
        assert bond.escrow_tx_hash == ""

    @pytest.mark.asyncio
    async def test_bond_stores_claim_hash(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item(content={"test": "data"})

        bond = await manager.create_bond(item, link, claim_certainty=0.8)

        assert bond is not None
        assert len(bond.claim_content_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_bond_stores_embedding(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        embedding = [0.1, 0.2, 0.3]
        item = make_item(embedding=embedding)

        bond = await manager.create_bond(item, link, claim_certainty=0.8)

        assert bond is not None
        assert bond.claim_embedding == embedding

    @pytest.mark.asyncio
    async def test_bond_has_correct_expiry(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item()

        before = utc_now()
        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        after = utc_now()

        assert bond is not None
        # Expiry should be ~90 days from now
        expected_min = before + timedelta(days=90)
        expected_max = after + timedelta(days=90)
        assert expected_min <= bond.bond_expires_at <= expected_max

    @pytest.mark.asyncio
    async def test_bond_records_escrow_tx_hash(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.8)

        assert bond is not None
        assert bond.escrow_tx_hash == "0xTX_HASH_ABC123"


# ─── Contradiction Detection ────────────────────────────────────


class TestContradictionDetection:
    """Test contradiction detection via embedding similarity and content divergence."""

    @pytest.mark.asyncio
    async def test_high_similarity_low_overlap_is_contradiction(self):
        """Same topic (high embedding sim) + different content = contradiction."""
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()

        # Create a bonded claim
        claim_embedding = [1.0, 0.0, 0.0]
        item = make_item(
            content={"summary": "Market sentiment is very positive"},
            embedding=claim_embedding,
        )
        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        # Inbound item with very similar embedding but different content
        inbound = make_item(
            content={"summary": "Market sentiment is extremely negative"},
            embedding=[0.99, 0.05, 0.0],  # cosine sim ~0.998 with [1,0,0]
        )

        contradictions = await manager.check_contradiction(
            inbound, source_instance_id="remote-001",
        )

        assert len(contradictions) == 1
        bond_result, evidence = contradictions[0]
        assert bond_result.id == bond.id
        assert evidence.similarity_score > 0.85

    @pytest.mark.asyncio
    async def test_high_similarity_same_hash_not_contradiction(self):
        """Same topic and identical content hash = reinforcement, not contradiction."""
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()

        content = {"summary": "Test claim"}
        claim_embedding = [1.0, 0.0, 0.0]
        item = make_item(content=content, embedding=claim_embedding)
        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        # Inbound with same content and similar embedding
        inbound = make_item(content=content, embedding=[0.99, 0.05, 0.0])

        contradictions = await manager.check_contradiction(
            inbound, source_instance_id="remote-001",
        )

        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_low_similarity_ignored(self):
        """Different topic (low embedding sim) = not a contradiction."""
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()

        item = make_item(
            content={"summary": "Weather is sunny"},
            embedding=[1.0, 0.0, 0.0],
        )
        await manager.create_bond(item, link, claim_certainty=0.8)

        # Completely different topic
        inbound = make_item(
            content={"summary": "Stock prices crashed"},
            embedding=[0.0, 0.0, 1.0],  # Orthogonal = 0.0 similarity
        )

        contradictions = await manager.check_contradiction(
            inbound, source_instance_id="remote-001",
        )

        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_no_embedding_skipped(self):
        """Items without embeddings cannot be checked for contradiction."""
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()

        item = make_item(embedding=[1.0, 0.0, 0.0])
        await manager.create_bond(item, link, claim_certainty=0.8)

        # Inbound item has no embedding
        inbound = make_item(embedding=None)

        contradictions = await manager.check_contradiction(
            inbound, source_instance_id="remote-001",
        )

        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_multiple_contradictions_detected(self):
        """One inbound item can contradict multiple bonded claims."""
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()

        embedding = [1.0, 0.0, 0.0]
        # Create two bonds on similar topics
        item1 = make_item(content={"a": "claim one"}, embedding=embedding)
        item2 = make_item(content={"b": "claim two"}, embedding=[0.99, 0.05, 0.0])
        await manager.create_bond(item1, link, claim_certainty=0.8)
        await manager.create_bond(item2, link, claim_certainty=0.8)

        # Inbound contradicts both (same topic, different content)
        inbound = make_item(
            content={"c": "contradicting everything"},
            embedding=[0.98, 0.1, 0.0],
        )

        contradictions = await manager.check_contradiction(
            inbound, source_instance_id="remote-001",
        )

        assert len(contradictions) == 2

    @pytest.mark.asyncio
    async def test_only_active_bonds_checked(self):
        """Forfeited or expired bonds are not checked for contradiction."""
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()

        item = make_item(
            content={"summary": "active claim"},
            embedding=[1.0, 0.0, 0.0],
        )
        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        # Manually mark as forfeited
        bond.status = BondStatus.FORFEITED

        inbound = make_item(
            content={"summary": "contradicting content"},
            embedding=[0.99, 0.05, 0.0],
        )

        contradictions = await manager.check_contradiction(
            inbound, source_instance_id="remote-001",
        )

        assert len(contradictions) == 0


# ─── Bond Forfeit ───────────────────────────────────────────────


class TestBondForfeit:
    """Test bond forfeit flow."""

    @pytest.mark.asyncio
    async def test_forfeit_transfers_usdc_to_remote(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item(embedding=[1.0, 0.0])

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        evidence = ContradictionEvidence(
            contradicting_item_id="inbound-001",
            contradicting_content_hash="abc123",
            similarity_score=0.92,
            source_instance_id="remote-001",
        )

        result = await manager.forfeit_bond(bond, evidence, "0xREMOTE_WALLET")

        assert result is True
        # Verify wallet.transfer was called for the forfeit
        # (First call was escrow, second is forfeit)
        assert wallet.transfer.call_count == 2
        forfeit_call = wallet.transfer.call_args
        assert forfeit_call.kwargs["destination_address"] == "0xREMOTE_WALLET"
        assert forfeit_call.kwargs["asset"] == "usdc"

    @pytest.mark.asyncio
    async def test_forfeit_updates_status(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item(embedding=[1.0, 0.0])

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        evidence = ContradictionEvidence(
            contradicting_item_id="inbound-001",
            contradicting_content_hash="abc123",
            similarity_score=0.92,
            source_instance_id="remote-001",
        )

        await manager.forfeit_bond(bond, evidence, "0xREMOTE")

        assert bond.status == BondStatus.FORFEITED

    @pytest.mark.asyncio
    async def test_forfeit_records_evidence(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item(embedding=[1.0, 0.0])

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        evidence = ContradictionEvidence(
            contradicting_item_id="inbound-001",
            contradicting_content_hash="abc123",
            similarity_score=0.92,
            explanation="Test contradiction",
            source_instance_id="remote-001",
        )

        await manager.forfeit_bond(bond, evidence, "0xREMOTE")

        assert bond.forfeit_evidence is not None
        assert bond.forfeit_evidence.explanation == "Test contradiction"
        assert bond.forfeit_evidence.similarity_score == 0.92

    @pytest.mark.asyncio
    async def test_forfeit_wallet_failure_bond_stays_active(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item(embedding=[1.0, 0.0])

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        # Make the forfeit transfer fail
        wallet.transfer = AsyncMock(side_effect=Exception("Network error"))

        evidence = ContradictionEvidence(
            contradicting_item_id="inbound-001",
            contradicting_content_hash="abc123",
            similarity_score=0.92,
            source_instance_id="remote-001",
        )

        result = await manager.forfeit_bond(bond, evidence, "0xREMOTE")

        assert result is False
        assert bond.status == BondStatus.ACTIVE  # Unchanged

    @pytest.mark.asyncio
    async def test_forfeit_skipped_without_wallet_address(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item(embedding=[1.0, 0.0])

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        evidence = ContradictionEvidence(
            contradicting_item_id="inbound-001",
            contradicting_content_hash="abc123",
            similarity_score=0.92,
            source_instance_id="remote-001",
        )

        result = await manager.forfeit_bond(bond, evidence, "")

        assert result is False
        assert bond.status == BondStatus.ACTIVE


# ─── Bond Recovery ──────────────────────────────────────────────


class TestBondRecovery:
    """Test expired bond recovery."""

    @pytest.mark.asyncio
    async def test_expired_bonds_recovered(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        # Manually expire the bond
        bond.bond_expires_at = utc_now() - timedelta(days=1)

        recovered = await manager.recover_expired_bonds()

        assert len(recovered) == 1
        assert recovered[0].id == bond.id
        assert recovered[0].status == BondStatus.EXPIRED_RETURNED
        assert recovered[0].return_tx_hash == "0xTX_HASH_ABC123"

    @pytest.mark.asyncio
    async def test_active_bonds_not_recovered(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None
        # Bond expires in 90 days - should NOT be recovered

        recovered = await manager.recover_expired_bonds()

        assert len(recovered) == 0
        assert bond.status == BondStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_already_resolved_bonds_skipped(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None

        # Mark as already forfeited and expire it
        bond.status = BondStatus.FORFEITED
        bond.bond_expires_at = utc_now() - timedelta(days=1)

        recovered = await manager.recover_expired_bonds()

        assert len(recovered) == 0

    @pytest.mark.asyncio
    async def test_recovery_wallet_failure_retries_next_cycle(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet)
        link = make_link()
        item = make_item()

        bond = await manager.create_bond(item, link, claim_certainty=0.8)
        assert bond is not None
        bond.bond_expires_at = utc_now() - timedelta(days=1)

        # Make the recovery transfer fail
        wallet.transfer = AsyncMock(side_effect=Exception("Timeout"))

        recovered = await manager.recover_expired_bonds()

        assert len(recovered) == 0
        assert bond.status == BondStatus.ACTIVE  # Still active, will retry


# ─── Stats ──────────────────────────────────────────────────────


class TestStakingStats:
    """Test staking statistics reporting."""

    @pytest.mark.asyncio
    async def test_stats_accurate_counts(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet, base_bond=Decimal("5.00"))
        link = make_link()

        # Create 3 bonds
        for _ in range(3):
            await manager.create_bond(make_item(), link, claim_certainty=1.0)

        stats = manager.stats
        assert stats["bonds_active"] == 3
        assert stats["total_bonded_usdc"] == "15.00"
        assert stats["bonds_forfeited"] == 0
        assert stats["bonds_expired_returned"] == 0

    @pytest.mark.asyncio
    async def test_forfeit_rate_calculation(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet, base_bond=Decimal("5.00"))
        link = make_link()

        # Create bonds
        bonds = []
        for _ in range(3):
            b = await manager.create_bond(make_item(), link, claim_certainty=1.0)
            assert b is not None
            bonds.append(b)

        # Expire one (resolved)
        bonds[0].status = BondStatus.EXPIRED_RETURNED
        # Forfeit one (resolved)
        bonds[1].status = BondStatus.FORFEITED

        stats = manager.stats
        # resolved = 1 expired + 1 forfeited = 2
        # forfeit_rate = 1 / 2 = 0.5
        assert stats["forfeit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_per_instance_breakdown(self):
        wallet = make_wallet_mock()
        manager = make_manager(
            wallet=wallet,
            base_bond=Decimal("5.00"),
            max_per_instance=Decimal("100.00"),
        )

        link_a = make_link(remote_instance_id="remote-A")
        link_b = make_link(remote_instance_id="remote-B")

        await manager.create_bond(make_item(), link_a, claim_certainty=1.0)
        await manager.create_bond(make_item(), link_a, claim_certainty=1.0)
        await manager.create_bond(make_item(), link_b, claim_certainty=1.0)

        stats = manager.stats
        per_instance = stats["per_instance"]

        assert "remote-A" in per_instance
        assert per_instance["remote-A"]["active_bonds"] == 2
        assert per_instance["remote-A"]["total_bonded_usdc"] == "10.00"

        assert "remote-B" in per_instance
        assert per_instance["remote-B"]["active_bonds"] == 1
        assert per_instance["remote-B"]["total_bonded_usdc"] == "5.00"

    @pytest.mark.asyncio
    async def test_total_bonded_usdc_excludes_resolved(self):
        wallet = make_wallet_mock()
        manager = make_manager(wallet=wallet, base_bond=Decimal("10.00"))
        link = make_link()

        bond1 = await manager.create_bond(make_item(), link, claim_certainty=1.0)
        bond2 = await manager.create_bond(make_item(), link, claim_certainty=1.0)
        assert bond1 is not None and bond2 is not None

        # Forfeit one bond
        bond1.status = BondStatus.FORFEITED

        assert manager.total_bonded_usdc == Decimal("10.00")  # Only bond2


# ─── Cosine Similarity ─────────────────────────────────────────


class TestCosineSimilarity:
    """Test the cosine similarity helper."""

    def test_identical_vectors(self):
        assert ReputationStakingManager._cosine_similarity(
            [1.0, 0.0], [1.0, 0.0],
        ) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert ReputationStakingManager._cosine_similarity(
            [1.0, 0.0], [0.0, 1.0],
        ) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert ReputationStakingManager._cosine_similarity(
            [1.0, 0.0], [-1.0, 0.0],
        ) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert ReputationStakingManager._cosine_similarity([], []) == 0.0

    def test_different_length_vectors(self):
        assert ReputationStakingManager._cosine_similarity(
            [1.0, 0.0], [1.0, 0.0, 0.0],
        ) == 0.0


# ─── Content Hashing ───────────────────────────────────────────


class TestContentHashing:
    """Test that content hashing is deterministic and canonical."""

    def test_hash_deterministic(self):
        content = {"b": 2, "a": 1}
        hash1 = ReputationStakingManager._hash_claim_content(content)
        hash2 = ReputationStakingManager._hash_claim_content(content)
        assert hash1 == hash2

    def test_hash_key_order_independent(self):
        """Sort keys ensures order independence."""
        hash1 = ReputationStakingManager._hash_claim_content({"a": 1, "b": 2})
        hash2 = ReputationStakingManager._hash_claim_content({"b": 2, "a": 1})
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        hash1 = ReputationStakingManager._hash_claim_content({"a": 1})
        hash2 = ReputationStakingManager._hash_claim_content({"a": 2})
        assert hash1 != hash2

    def test_hash_is_sha256(self):
        h = ReputationStakingManager._hash_claim_content({"test": True})
        assert len(h) == 64  # SHA-256 produces 64 hex characters
