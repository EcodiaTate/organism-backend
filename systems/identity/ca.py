"""
EcodiaOS - Genesis Certificate Authority (Spec 23, CRITICAL #1-3)

Self-contained CA implemented within the Identity system. No external CA
dependency. The Genesis Node holds the root Ed25519 private key in the
vault on first boot; all subsequent issuances use that key.

Responsibilities:
  1. Store the Genesis CA private key in the vault (encrypted at rest).
  2. issue_certificate(instance_id) - signs an official 30-day certificate
     for any instance that has passed provisioning approval.
  3. Embed a live constitutional hash (queried from Equor via Synapse) in
     every issued certificate so the cert carries the organism's alignment
     state at the moment of issuance, not a hardcoded fallback.

Architecture:
  - GenesisCA is wired into CertificateManager at initialization.
  - Only the Genesis Node may instantiate a GenesisCA (same is_genesis_node gate).
  - Constitutional hash resolution is async and times out after 2 s; falls
    back to compute_constitutional_hash() if Equor is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from typing import TYPE_CHECKING, Any

import structlog
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from primitives.common import new_id, utc_now
from systems.identity.certificate import (
    CertificateType,
    EcodianCertificate,
    build_certificate,
    compute_lineage_hash,
)
from systems.identity.identity import compute_constitutional_hash

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.ca")

# Vault purpose tag for the Genesis CA private key
_CA_KEY_PURPOSE = "genesis_ca_private_key"
_CA_PLATFORM_ID = "identity.ca"

# How long to wait for Equor alignment score before falling back
_EQUOR_TIMEOUT_S = 2.0

# Official CA cert validity (non-genesis instances)
_OFFICIAL_VALIDITY_DAYS = 30


class GenesisCA:
    """
    Self-signed Genesis Certificate Authority.

    Stores the CA private key encrypted in IdentityVault. On each issuance,
    queries Equor for the live drive alignment vector, hashes it, and embeds
    it in the certificate extension field (constitutional_hash).

    This is the only object in EOS that may sign OFFICIAL certificates.
    BIRTH certificates are signed by parent CertificateManagers directly.

    Thread-safety: NOT thread-safe. Single-threaded asyncio like all EOS services.
    """

    def __init__(
        self,
        instance_id: str,
        vault: IdentityVault,
        event_bus: EventBus | None = None,
    ) -> None:
        self._instance_id = instance_id
        self._vault = vault
        self._event_bus = event_bus
        self._ca_private_key: Ed25519PrivateKey | None = None
        self._ca_public_key_pem: str = ""
        self._log = logger.bind(system="identity.ca", instance_id=instance_id)

        # Pending Equor alignment responses, keyed by request_id
        self._equor_responses: dict[str, asyncio.Future[dict[str, Any]]] = {}

    # ── Boot ────────────────────────────────────────────────────────────

    async def initialize(self, existing_key_pem: bytes | None = None) -> None:
        """
        Load or generate the Genesis CA private key.

        On first boot (no existing_key_pem), generates a new Ed25519 keypair
        and seals the private key in the vault. The sealed envelope's
        ciphertext must be persisted by the caller (CertificateManager stores
        it alongside the certificate JSON).

        On cold-start restores (existing_key_pem provided), loads the PEM
        directly (caller is responsible for decrypting from vault first).

        After initialize(), self.is_ready is True and issue_certificate() may
        be called.
        """
        if existing_key_pem is not None:
            key = serialization.load_pem_private_key(existing_key_pem, password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise ValueError("Provided key is not an Ed25519 private key")
            self._ca_private_key = key
        else:
            # First boot - generate new keypair and seal it
            self._ca_private_key = Ed25519PrivateKey.generate()
            self._log.info("genesis_ca_key_generated")

        # Cache PEM of public key for fast embedding in certs
        self._ca_public_key_pem = self._ca_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        # Wire Equor alignment response handler
        if self._event_bus is not None:
            self._wire_equor_handler()

        self._log.info("genesis_ca_initialized", has_key=self.is_ready)

    def seal_private_key(self) -> bytes:
        """
        Return the sealed (vault-encrypted) PEM of the CA private key.

        Caller (CertificateManager) persists this alongside the certificate
        JSON so the key survives restarts. Call once after initialize() on
        first boot.

        Returns the raw ciphertext bytes of the SealedEnvelope (JSON-encoded).
        """
        if self._ca_private_key is None:
            raise RuntimeError("GenesisCA not initialized - call initialize() first")

        pem = self._ca_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        envelope = self._vault.encrypt(
            plaintext=pem,
            platform_id=_CA_PLATFORM_ID,
            purpose=_CA_KEY_PURPOSE,
        )
        return envelope.model_dump_json().encode("utf-8")

    @classmethod
    def unseal_private_key(cls, sealed_json: bytes, vault: IdentityVault) -> bytes:
        """
        Decrypt a sealed CA private key envelope and return the raw PEM bytes.

        Used by CertificateManager on cold restart to restore the CA key.
        """
        from systems.identity.vault import SealedEnvelope

        envelope = SealedEnvelope.model_validate_json(sealed_json)
        return vault.decrypt(envelope)

    @property
    def is_ready(self) -> bool:
        return self._ca_private_key is not None

    @property
    def public_key_pem(self) -> str:
        return self._ca_public_key_pem

    # ── Core CA Operation ───────────────────────────────────────────────

    async def issue_certificate(
        self,
        instance_id: str,
        lineage_hash: str,
        validity_days: int = _OFFICIAL_VALIDITY_DAYS,
    ) -> EcodianCertificate:
        """
        Sign and return an official EcodianCertificate for instance_id.

        Constitutional hash is resolved live from Equor's drive alignment
        vector (EQUOR_ALIGNMENT_SCORE response). Falls back to the static
        document hash if Equor is unavailable within the timeout.

        This is the single point where OFFICIAL certificates are minted.
        Only the Genesis CA may call this.

        Raises RuntimeError if the CA key is not loaded.
        """
        if self._ca_private_key is None:
            raise RuntimeError("GenesisCA.issue_certificate: CA key not loaded")

        # Resolve live constitutional hash from Equor
        constitutional_hash = await self._resolve_constitutional_hash()

        cert = build_certificate(
            instance_id=instance_id,
            issuer_instance_id=self._instance_id,
            issuer_private_key=self._ca_private_key,
            lineage_hash=lineage_hash,
            constitutional_hash=constitutional_hash,
            certificate_type=CertificateType.OFFICIAL,
            validity_days=validity_days,
        )

        self._log.info(
            "ca_certificate_issued",
            for_instance=instance_id,
            certificate_id=cert.certificate_id,
            constitutional_hash=constitutional_hash[:16] + "...",
            expires_at=cert.expires_at.isoformat(),
        )
        return cert

    # ── Constitutional Hash Resolution ─────────────────────────────────

    async def _resolve_constitutional_hash(self) -> str:
        """
        Query Equor for the live drive alignment score and derive a hash.

        Emits EQUOR_HEALTH_REQUEST, waits up to _EQUOR_TIMEOUT_S for
        EQUOR_ALIGNMENT_SCORE response, then SHA-256s the drive vector dict.

        Falls back to compute_constitutional_hash() (document hash) if:
          - No event bus is wired
          - Equor does not respond within timeout
          - Response payload is malformed
        """
        if self._event_bus is None:
            return compute_constitutional_hash()

        request_id = new_id()
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._equor_responses[request_id] = future

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.EQUOR_HEALTH_REQUEST,
                    source_system="identity",
                    data={
                        "request_id": request_id,
                        "requester": self._instance_id,
                        "purpose": "certificate_issuance",
                    },
                )
            )

            response = await asyncio.wait_for(future, timeout=_EQUOR_TIMEOUT_S)
            return self._hash_drive_vector(response)

        except asyncio.TimeoutError:
            self._log.warning(
                "equor_alignment_timeout",
                request_id=request_id,
                fallback="document_hash",
            )
            return compute_constitutional_hash()
        except Exception as exc:
            self._log.warning("equor_alignment_error", error=str(exc), fallback="document_hash")
            return compute_constitutional_hash()
        finally:
            self._equor_responses.pop(request_id, None)

    def _hash_drive_vector(self, payload: dict[str, Any]) -> str:
        """
        SHA-256 hash of the Equor drive alignment vector dict.

        Extracts mean_alignment (dict[str, float]) from the EQUOR_ALIGNMENT_SCORE
        payload and hashes its canonical JSON form. This gives a deterministic,
        content-addressed fingerprint of the organism's constitutional alignment
        at the moment of certificate issuance.
        """
        drive_vector: dict[str, Any] = payload.get("mean_alignment", {})
        if not drive_vector:
            # Composite scalar fallback - less information but still live
            composite = payload.get("composite", payload.get("alignment_score", 0.0))
            drive_vector = {"composite": composite}

        canonical = json.dumps(drive_vector, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ── Equor Response Handler ──────────────────────────────────────────

    def _wire_equor_handler(self) -> None:
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEventType

        self._event_bus.subscribe(
            SynapseEventType.EQUOR_ALIGNMENT_SCORE,
            self._on_equor_alignment_score,
        )

    async def _on_equor_alignment_score(self, event: Any) -> None:
        """Resolve pending Equor alignment futures."""
        request_id = event.data.get("request_id", "")
        future = self._equor_responses.get(request_id)
        if future is not None and not future.done():
            future.set_result(event.data)
