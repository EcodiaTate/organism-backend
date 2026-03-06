"""
EcodiaOS — Federation Identity Management

Handles creation and verification of instance identity cards,
Ed25519 signing and verification, and certificate fingerprinting.

Every EOS instance has a permanent identity — an Ed25519 keypair generated
at birth, a self-signed TLS certificate for mutual authentication, and a
public identity card that can be shared with federation partners.

The identity is inviolable: no instance can modify another's identity.
The public key and certificate fingerprint are used for cryptographic
verification of federation messages.
"""

from __future__ import annotations

import hashlib
from typing import Any
from pathlib import Path

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.x509 import load_pem_x509_certificate

from primitives.federation import (
    InstanceIdentityCard,
    TrustPolicy,
)

logger = structlog.get_logger("systems.federation.identity")


class IdentityManager:
    """
    Manages this instance's identity and verifies remote identities.

    Responsibilities:
      - Build this instance's InstanceIdentityCard from config + memory
      - Load Ed25519 private key for signing outbound messages
      - Verify remote instance identity cards (signature + certificate)
      - Compute certificate fingerprints for comparison
    """

    def __init__(self) -> None:
        self._private_key: Ed25519PrivateKey | None = None
        self._public_key: Ed25519PublicKey | None = None
        self._public_key_pem: str = ""
        self._certificate_fingerprint: str = ""
        self._local_identity: InstanceIdentityCard | None = None
        self._logger = logger.bind(component="identity_manager")

    # ─── Initialization ─────────────────────────────────────────────

    async def initialize(
        self,
        instance_id: str,
        instance_name: str,
        community_context: str,
        personality_summary: str,
        autonomy_level: int,
        endpoint: str,
        capabilities: list[str],
        trust_policy: TrustPolicy,
        private_key_path: Path | None = None,
        tls_cert_path: Path | None = None,
    ) -> None:
        """
        Initialize identity from configuration and stored keys.

        If no private key exists, generate a new Ed25519 keypair.
        This happens once at instance birth.
        """
        # Load or generate Ed25519 keypair
        if private_key_path and private_key_path.exists():
            key_bytes = private_key_path.read_bytes()
            loaded_key = serialization.load_pem_private_key(
                key_bytes, password=None,
            )
            if not isinstance(loaded_key, Ed25519PrivateKey):
                raise TypeError("Federation key must be Ed25519")
            self._private_key = loaded_key
            self._logger.info("identity_key_loaded", path=str(private_key_path))
        else:
            self._private_key = Ed25519PrivateKey.generate()
            self._logger.info("identity_key_generated")

            # Persist the key if a path was given
            if private_key_path:
                private_key_path.parent.mkdir(parents=True, exist_ok=True)
                key_pem = self._private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                private_key_path.write_bytes(key_pem)
                self._logger.info("identity_key_persisted", path=str(private_key_path))

        # Extract public key
        self._public_key = self._private_key.public_key()
        self._public_key_pem = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        # Compute certificate fingerprint
        if tls_cert_path and tls_cert_path.exists():
            cert_bytes = tls_cert_path.read_bytes()
            self._certificate_fingerprint = _compute_cert_fingerprint(cert_bytes)
        else:
            # Use public key fingerprint as fallback
            self._certificate_fingerprint = _compute_key_fingerprint(
                self._public_key_pem
            )

        # Compute constitutional hash (hash of the drives for compatibility)
        constitutional_hash = hashlib.sha256(
            b"coherence:1.0|care:1.0|growth:1.0|honesty:1.0"
        ).hexdigest()[:16]

        # Build local identity card
        self._local_identity = InstanceIdentityCard(
            instance_id=instance_id,
            name=instance_name,
            community_context=community_context,
            personality_summary=personality_summary,
            autonomy_level=autonomy_level,
            endpoint=endpoint,
            certificate_fingerprint=self._certificate_fingerprint,
            public_key_pem=self._public_key_pem,
            constitutional_hash=constitutional_hash,
            capabilities=capabilities,
            trust_policy=trust_policy,
        )

        self._logger.info(
            "identity_initialized",
            instance_id=instance_id,
            fingerprint=self._certificate_fingerprint[:16] + "...",
        )

    # ─── Local Identity ─────────────────────────────────────────────

    @property
    def identity_card(self) -> InstanceIdentityCard:
        """This instance's public identity card."""
        if self._local_identity is None:
            raise RuntimeError("IdentityManager not initialized")
        return self._local_identity

    @property
    def instance_id(self) -> str:
        return self.identity_card.instance_id

    # ─── Signing ────────────────────────────────────────────────────

    def sign(self, data: bytes) -> bytes:
        """Sign data with this instance's Ed25519 private key."""
        if self._private_key is None:
            raise RuntimeError("IdentityManager not initialized — no private key")
        return self._private_key.sign(data)

    # ─── Verification ───────────────────────────────────────────────

    def verify_identity(self, remote_identity: InstanceIdentityCard) -> VerificationResult:
        """
        Verify a remote instance's identity card.

        Checks:
          1. Public key is present and parseable
          2. Certificate fingerprint is present
          3. Protocol version is compatible
          4. Instance ID is non-empty

        Does NOT check certificate chain (that happens at the TLS layer).
        """
        errors: list[str] = []

        if not remote_identity.instance_id:
            errors.append("Missing instance_id")

        if not remote_identity.name:
            errors.append("Missing instance name")

        if not remote_identity.public_key_pem:
            errors.append("Missing public key")
        else:
            try:
                _parse_public_key(remote_identity.public_key_pem)
            except Exception as exc:
                errors.append(f"Invalid public key: {exc}")

        if not remote_identity.certificate_fingerprint:
            errors.append("Missing certificate fingerprint")

        if remote_identity.protocol_version != "1.0":
            errors.append(
                f"Incompatible protocol version: {remote_identity.protocol_version}"
            )

        if errors:
            return VerificationResult(
                verified=False,
                errors=errors,
                remote_instance_id=remote_identity.instance_id,
            )

        return VerificationResult(
            verified=True,
            errors=[],
            remote_instance_id=remote_identity.instance_id,
        )

    def verify_signature(
        self,
        data: bytes,
        signature: bytes,
        remote_public_key_pem: str,
    ) -> bool:
        """Verify a signature using a remote instance's public key."""
        try:
            public_key = _parse_public_key(remote_public_key_pem)
            public_key.verify(signature, data)
            return True
        except Exception:
            self._logger.warning(
                "signature_verification_failed",
                key_prefix=remote_public_key_pem[:40],
            )
            return False

    # ─── Health ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "initialized": self._local_identity is not None,
            "instance_id": self._local_identity.instance_id if self._local_identity else None,
            "fingerprint_prefix": self._certificate_fingerprint[:16] if self._certificate_fingerprint else None,
            "has_private_key": self._private_key is not None,
        }


# ─── Helper Types ────────────────────────────────────────────────


class VerificationResult:
    """Result of verifying a remote instance identity."""

    __slots__ = ("verified", "errors", "remote_instance_id")

    def __init__(
        self,
        verified: bool,
        errors: list[str],
        remote_instance_id: str,
    ) -> None:
        self.verified = verified
        self.errors = errors
        self.remote_instance_id = remote_instance_id


# ─── Utility Functions ───────────────────────────────────────────


def _compute_cert_fingerprint(cert_pem: bytes) -> str:
    """Compute SHA-256 fingerprint of a PEM-encoded X.509 certificate."""
    cert = load_pem_x509_certificate(cert_pem)
    digest = cert.fingerprint(hashes.SHA256())
    return digest.hex()


def _compute_key_fingerprint(public_key_pem: str) -> str:
    """Compute SHA-256 fingerprint of a public key PEM string."""
    return hashlib.sha256(public_key_pem.encode()).hexdigest()


def _parse_public_key(pem: str) -> Ed25519PublicKey:
    """Parse a PEM-encoded Ed25519 public key."""
    key = serialization.load_pem_public_key(pem.encode())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError(f"Expected Ed25519 public key, got {type(key).__name__}")
    return key
