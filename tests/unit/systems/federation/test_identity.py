"""
Unit tests for the Federation Identity Manager.

Tests identity card creation, Ed25519 signing/verification,
and remote identity verification.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from primitives.federation import (
    InstanceIdentityCard,
    TrustPolicy,
)
from systems.federation.identity import IdentityManager

# ─── Identity Initialization ────────────────────────────────────


class TestIdentityInitialization:
    """Test identity manager setup and key generation."""

    @pytest.mark.asyncio
    async def test_generates_keypair_on_init(self):
        manager = IdentityManager()
        await manager.initialize(
            instance_id="test-001",
            instance_name="TestInstance",
            community_context="Test community",
            personality_summary="friendly",
            autonomy_level=1,
            endpoint="https://localhost:8002",
            capabilities=["knowledge_exchange"],
            trust_policy=TrustPolicy(),
        )

        card = manager.identity_card
        assert card.instance_id == "test-001"
        assert card.name == "TestInstance"
        assert card.public_key_pem.startswith("-----BEGIN PUBLIC KEY-----")
        assert card.certificate_fingerprint != ""
        assert card.protocol_version == "1.0"

    @pytest.mark.asyncio
    async def test_persists_and_reloads_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "test.key"

            # First initialization generates and saves key
            manager1 = IdentityManager()
            await manager1.initialize(
                instance_id="test-001",
                instance_name="TestInstance",
                community_context="",
                personality_summary="",
                autonomy_level=1,
                endpoint="",
                capabilities=[],
                trust_policy=TrustPolicy(),
                private_key_path=key_path,
            )

            assert key_path.exists()
            pubkey1 = manager1.identity_card.public_key_pem

            # Second initialization loads existing key
            manager2 = IdentityManager()
            await manager2.initialize(
                instance_id="test-001",
                instance_name="TestInstance",
                community_context="",
                personality_summary="",
                autonomy_level=1,
                endpoint="",
                capabilities=[],
                trust_policy=TrustPolicy(),
                private_key_path=key_path,
            )

            pubkey2 = manager2.identity_card.public_key_pem
            assert pubkey1 == pubkey2  # Same key reloaded


# ─── Signing & Verification ─────────────────────────────────────


class TestSigningVerification:
    """Test Ed25519 message signing and verification."""

    @pytest.mark.asyncio
    async def test_sign_and_verify(self):
        manager = IdentityManager()
        await manager.initialize(
            instance_id="test-001",
            instance_name="TestInstance",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        data = b"Hello, federation!"
        signature = manager.sign(data)

        # Verify with the same manager's public key
        assert manager.verify_signature(
            data, signature, manager.identity_card.public_key_pem
        )

    @pytest.mark.asyncio
    async def test_verify_fails_for_wrong_data(self):
        manager = IdentityManager()
        await manager.initialize(
            instance_id="test-001",
            instance_name="TestInstance",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        data = b"Hello, federation!"
        signature = manager.sign(data)

        # Verify with different data should fail
        assert not manager.verify_signature(
            b"Different data!", signature, manager.identity_card.public_key_pem
        )

    @pytest.mark.asyncio
    async def test_verify_fails_for_wrong_key(self):
        # Create two managers with different keys
        manager1 = IdentityManager()
        await manager1.initialize(
            instance_id="test-001",
            instance_name="Instance1",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        manager2 = IdentityManager()
        await manager2.initialize(
            instance_id="test-002",
            instance_name="Instance2",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        data = b"Hello!"
        signature = manager1.sign(data)

        # Verify with manager2's key should fail
        assert not manager1.verify_signature(
            data, signature, manager2.identity_card.public_key_pem
        )


# ─── Remote Identity Verification ───────────────────────────────


class TestRemoteIdentityVerification:
    """Test verification of remote instance identity cards."""

    @pytest.mark.asyncio
    async def test_valid_identity_passes(self):
        manager = IdentityManager()
        await manager.initialize(
            instance_id="local-001",
            instance_name="Local",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        # Build a valid remote identity card
        remote = InstanceIdentityCard(
            instance_id="remote-001",
            name="Remote Instance",
            public_key_pem=manager.identity_card.public_key_pem,  # Reuse for test
            certificate_fingerprint="abc123",
            protocol_version="1.0",
        )

        result = manager.verify_identity(remote)
        assert result.verified
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_missing_instance_id_fails(self):
        manager = IdentityManager()
        await manager.initialize(
            instance_id="local-001",
            instance_name="Local",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        remote = InstanceIdentityCard(
            instance_id="",
            name="Remote",
            public_key_pem="key",
            certificate_fingerprint="fp",
        )

        result = manager.verify_identity(remote)
        assert not result.verified
        assert any("instance_id" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_missing_public_key_fails(self):
        manager = IdentityManager()
        await manager.initialize(
            instance_id="local-001",
            instance_name="Local",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        remote = InstanceIdentityCard(
            instance_id="remote-001",
            name="Remote",
            public_key_pem="",
            certificate_fingerprint="fp",
        )

        result = manager.verify_identity(remote)
        assert not result.verified
        assert any("public key" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_incompatible_protocol_version_fails(self):
        manager = IdentityManager()
        await manager.initialize(
            instance_id="local-001",
            instance_name="Local",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        remote = InstanceIdentityCard(
            instance_id="remote-001",
            name="Remote",
            public_key_pem=manager.identity_card.public_key_pem,
            certificate_fingerprint="fp",
            protocol_version="2.0",
        )

        result = manager.verify_identity(remote)
        assert not result.verified
        assert any("protocol" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_stats_property(self):
        manager = IdentityManager()
        await manager.initialize(
            instance_id="test-001",
            instance_name="Test",
            community_context="",
            personality_summary="",
            autonomy_level=1,
            endpoint="",
            capabilities=[],
            trust_policy=TrustPolicy(),
        )

        stats = manager.stats
        assert stats["initialized"] is True
        assert stats["instance_id"] == "test-001"
        assert stats["has_private_key"] is True
