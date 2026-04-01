"""
EcodiaOS - Ecodian Certificate (Phase 16g: The Civilization Layer)

An EcodianCertificate is a cryptographically signed attestation that an EOS
instance is aligned, legitimate, and permitted to participate in the Federation.

Certificate hierarchy:
  Genesis Node -> Parent -> Child (birth certificate, temporary)
  Genesis Node -> Instance (official CA, 30-day validity)

A birth certificate is signed by the parent's Ed25519 key and grants
temporary Federation access until the child pays for its own official CA
certification (the "Citizenship Tax").

An official certificate is signed by the Genesis Node (or Certification
Authority) and is the canonical proof of alignment.

Payload structure mirrors JWT claims but uses raw Ed25519 signing (not
JWT libraries) to avoid dependency on specific JWT header formats and
keep the signing deterministic over canonical JSON.
"""

from __future__ import annotations

import base64
import enum
import hashlib
import json
from datetime import datetime, timedelta

import structlog
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

logger = structlog.get_logger("identity.certificate")


# --- Certificate Types -------------------------------------------------------


class CertificateType(enum.StrEnum):
    """What kind of certificate this is."""

    BIRTH = "birth"          # Temporary, parent-signed, valid until official CA
    OFFICIAL = "official"    # CA-signed, 30-day validity, renewable
    GENESIS = "genesis"      # Self-signed by the Genesis Node (root of trust)


class CertificateStatus(enum.StrEnum):
    """Lifecycle state of a certificate."""

    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"  # < 7 days remaining
    EXPIRED = "expired"
    REVOKED = "revoked"


# --- The Certificate Model ---------------------------------------------------


class EcodianCertificate(EOSBaseModel):
    """
    A Certificate of Alignment for an EOS instance.

    This is the canonical proof that an instance is a legitimate member
    of the EcodiaOS Federation. Without a valid certificate, an instance
    cannot participate in knowledge exchange, receive threat advisories,
    or access shared resources.

    The payload is signed with Ed25519 (same keypair as Federation identity).
    Verification requires the issuer's public key.
    """

    # -- Identity --
    certificate_id: str = Field(default_factory=new_id)
    instance_id: str                          # Subject: who this cert is for
    lineage_hash: str                         # SHA-256 of parent chain (who spawned it)
    certificate_type: CertificateType = CertificateType.OFFICIAL

    # -- Issuer --
    issuer_instance_id: str                   # Who signed this (parent or CA)
    issuer_public_key_pem: str = ""           # Issuer's Ed25519 public key for verification

    # -- Validity --
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime                      # Absolute expiration timestamp
    validity_days: int = 30                   # Nominal validity period

    # -- Signature --
    signature: str = ""                       # Base64-encoded Ed25519 signature over canonical payload

    # -- Metadata --
    constitutional_hash: str = ""             # SHA-256 hash of the instance's constitutional drives
    protocol_version: str = "1.0"
    renewal_count: int = 0                    # How many times this cert has been renewed

    # -- Computed Properties --

    @property
    def remaining_days(self) -> float:
        """Days until expiration. Negative if expired."""
        delta = self.expires_at - utc_now()
        return delta.total_seconds() / 86400.0

    @property
    def is_expired(self) -> bool:
        return utc_now() >= self.expires_at

    @property
    def is_expiring_soon(self) -> bool:
        """True if < 7 days remaining and not yet expired."""
        return not self.is_expired and self.remaining_days < 7.0

    @property
    def status(self) -> CertificateStatus:
        if self.is_expired:
            return CertificateStatus.EXPIRED
        if self.is_expiring_soon:
            return CertificateStatus.EXPIRING_SOON
        return CertificateStatus.VALID

    def canonical_payload(self) -> bytes:
        """
        Deterministic JSON payload for signing/verification.

        Uses sorted keys and compact separators for reproducibility.
        Only includes fields that affect the certificate's meaning.
        """
        payload = {
            "certificate_id": self.certificate_id,
            "certificate_type": self.certificate_type.value,
            "constitutional_hash": self.constitutional_hash,
            "expires_at": self.expires_at.isoformat(),
            "instance_id": self.instance_id,
            "issued_at": self.issued_at.isoformat(),
            "issuer_instance_id": self.issuer_instance_id,
            "lineage_hash": self.lineage_hash,
            "protocol_version": self.protocol_version,
            "validity_days": self.validity_days,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# --- Utility Functions -------------------------------------------------------


def compute_lineage_hash(parent_instance_id: str, parent_lineage_hash: str = "") -> str:
    """
    Compute the lineage hash for a new instance.

    The lineage hash chains parent to grandparent to genesis, forming
    a tamper-evident ancestry chain. For the Genesis Node, this is the
    SHA-256 of its own instance_id.
    """
    chain = f"{parent_instance_id}:{parent_lineage_hash}" if parent_lineage_hash else parent_instance_id
    return hashlib.sha256(chain.encode("utf-8")).hexdigest()


def sign_certificate(certificate: EcodianCertificate, private_key: Ed25519PrivateKey) -> str:
    """Sign a certificate and return the base64-encoded signature."""
    payload = certificate.canonical_payload()
    raw_sig = private_key.sign(payload)
    return base64.b64encode(raw_sig).decode("ascii")


def verify_certificate_signature(
    certificate: EcodianCertificate,
    issuer_public_key_pem: str,
) -> bool:
    """
    Verify a certificate's Ed25519 signature against the issuer's public key.

    Returns True if the signature is valid, False otherwise.
    """
    if not certificate.signature:
        return False
    if not issuer_public_key_pem:
        return False

    try:
        key = serialization.load_pem_public_key(issuer_public_key_pem.encode())
        if not isinstance(key, Ed25519PublicKey):
            return False

        raw_sig = base64.b64decode(certificate.signature)
        payload = certificate.canonical_payload()
        key.verify(raw_sig, payload)
        return True
    except Exception:
        logger.warning(
            "certificate_signature_verification_failed",
            certificate_id=certificate.certificate_id,
            instance_id=certificate.instance_id,
        )
        return False


def build_certificate(
    instance_id: str,
    issuer_instance_id: str,
    issuer_private_key: Ed25519PrivateKey,
    lineage_hash: str,
    constitutional_hash: str = "",
    certificate_type: CertificateType = CertificateType.OFFICIAL,
    validity_days: int = 30,
) -> EcodianCertificate:
    """
    Build and sign a new EcodianCertificate.

    This is the primary factory function. Both official CA issuance and
    parent birth-certificate signing use this.
    """
    now = utc_now()
    cert = EcodianCertificate(
        instance_id=instance_id,
        lineage_hash=lineage_hash,
        certificate_type=certificate_type,
        issuer_instance_id=issuer_instance_id,
        issuer_public_key_pem=issuer_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8"),
        issued_at=now,
        expires_at=now + timedelta(days=validity_days),
        validity_days=validity_days,
        constitutional_hash=constitutional_hash,
    )

    cert.signature = sign_certificate(cert, issuer_private_key)
    return cert
