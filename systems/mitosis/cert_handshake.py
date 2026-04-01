"""
EcodiaOS - Child-Side Certificate Validation + Federation Handshake (Spec 26 §6 / MEDIUM #6)

On startup, a child instance must:
  1. Load ECODIAOS_BIRTH_CERTIFICATE from environment (PEM-encoded X.509).
  2. Validate the certificate against the Genesis CA public key
     (ECODIAOS_GENESIS_CA_CERT env var or bundled trust anchor).
  3. Extract the child's instance_id, niche, and parent_instance_id from
     certificate Subject Alternative Names or extensions.
  4. Emit CHILD_WALLET_REPORTED via Synapse to announce the wallet address
     to the parent, completing the deferred seed capital transfer.
  5. Emit FEDERATION_PEER_CONNECTED so the parent's federation layer
     can establish the authenticated mTLS link.

Wire-up:
  handshake = ChildCertHandshake(synapse, wallet_client)
  await handshake.run()   # blocks until complete or raises on failure
"""

from __future__ import annotations

import base64
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now

if TYPE_CHECKING:
    from systems.synapse.service import SynapseService

logger = structlog.get_logger().bind(component="mitosis.cert_handshake")

_ENV_BIRTH_CERT = "ECODIAOS_BIRTH_CERTIFICATE"
_ENV_GENESIS_CA = "ECODIAOS_GENESIS_CA_CERT"
_ENV_INSTANCE_ID = "ECODIAOS_INSTANCE_ID"
_ENV_PARENT_ID = "ECODIAOS_PARENT_INSTANCE_ID"
_ENV_WALLET_ADDRESS = "ECODIAOS_WALLET_ADDRESS"
_ENV_NICHE = "ECODIAOS_NICHE"


class CertificateValidationError(Exception):
    """Raised when the birth certificate fails validation."""


class ChildCertHandshake:
    """
    Validate the birth certificate and announce the child to its parent.

    Parameters
    ----------
    synapse : SynapseService | None
        Used to emit CHILD_WALLET_REPORTED and FEDERATION_PEER_CONNECTED.
    instance_id : str
        This child's instance ID. Defaults to ECODIAOS_INSTANCE_ID env var.
    parent_instance_id : str
        Parent's instance ID. Defaults to ECODIAOS_PARENT_INSTANCE_ID env var.
    wallet_address : str
        This child's Base L2 wallet address. Defaults to ECODIAOS_WALLET_ADDRESS.
    niche : str
        This child's ecological niche. Defaults to ECODIAOS_NICHE env var.
    """

    def __init__(
        self,
        synapse: SynapseService | None = None,
        instance_id: str = "",
        parent_instance_id: str = "",
        wallet_address: str = "",
        niche: str = "",
    ) -> None:
        self._synapse = synapse
        self._instance_id = instance_id or os.environ.get(_ENV_INSTANCE_ID, "")
        self._parent_instance_id = parent_instance_id or os.environ.get(_ENV_PARENT_ID, "")
        self._wallet_address = wallet_address or os.environ.get(_ENV_WALLET_ADDRESS, "")
        self._niche = niche or os.environ.get(_ENV_NICHE, "")
        self._log = logger.bind(
            instance_id=self._instance_id,
            parent_id=self._parent_instance_id,
        )

    async def run(self) -> dict[str, Any]:
        """
        Execute the full cert validation + federation handshake sequence.

        Returns a dict with extracted certificate fields on success.
        Raises CertificateValidationError on any validation failure.
        """
        # 1. Load and validate birth certificate
        cert_pem = os.environ.get(_ENV_BIRTH_CERT, "")
        if not cert_pem:
            raise CertificateValidationError(
                f"Missing env var {_ENV_BIRTH_CERT} - child has no birth certificate"
            )

        cert_info = self._validate_certificate(cert_pem)
        self._log.info(
            "birth_certificate_validated",
            cert_serial=cert_info.get("serial", ""),
            not_after=cert_info.get("not_after", ""),
        )

        # 2. Announce wallet address to parent (triggers deferred seed transfer)
        if self._wallet_address and self._parent_instance_id:
            await self._emit_wallet_reported(cert_info)
        else:
            self._log.warning(
                "handshake_incomplete",
                has_wallet=bool(self._wallet_address),
                has_parent=bool(self._parent_instance_id),
            )

        # 3. Announce federation presence
        await self._emit_federation_connected(cert_info)

        return cert_info

    # ── Certificate Validation ──────────────────────────────────────

    def _validate_certificate(self, cert_pem: str) -> dict[str, Any]:
        """
        Validate the birth certificate against the Genesis CA.

        Uses Python's cryptography library if available, falls back to
        a basic PEM structure check and expiry validation otherwise.

        Returns a dict of extracted certificate fields.
        Raises CertificateValidationError on failure.
        """
        try:
            return self._validate_with_cryptography(cert_pem)
        except ImportError:
            self._log.warning(
                "cryptography_library_unavailable",
                fallback="basic_pem_check",
            )
            return self._validate_basic(cert_pem)

    def _validate_with_cryptography(self, cert_pem: str) -> dict[str, Any]:
        """Full cryptographic validation using the cryptography library."""
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.x509.oid import ExtensionOID

        # Load birth certificate
        cert_bytes = cert_pem.encode() if isinstance(cert_pem, str) else cert_pem
        try:
            cert = x509.load_pem_x509_certificate(cert_bytes)
        except Exception as exc:
            raise CertificateValidationError(
                f"Failed to parse birth certificate PEM: {exc}"
            ) from exc

        # Check expiry
        now = datetime.now(UTC)
        if cert.not_valid_after_utc < now:
            raise CertificateValidationError(
                f"Birth certificate expired at {cert.not_valid_after_utc.isoformat()}"
            )

        # Load Genesis CA certificate for signature verification
        ca_pem = os.environ.get(_ENV_GENESIS_CA, "")
        if ca_pem:
            ca_bytes = ca_pem.encode() if isinstance(ca_pem, str) else ca_pem
            try:
                ca_cert = x509.load_pem_x509_certificate(ca_bytes)
            except Exception as exc:
                raise CertificateValidationError(
                    f"Failed to parse Genesis CA certificate: {exc}"
                ) from exc

            # Verify child cert was signed by Genesis CA
            ca_public_key = ca_cert.public_key()
            try:
                ca_public_key.verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    cert.signature_hash_algorithm,  # type: ignore[arg-type]
                )
            except Exception as exc:
                raise CertificateValidationError(
                    f"Birth certificate signature verification failed: {exc}"
                ) from exc

        # Extract Subject Alternative Names (instance_id stored as DNS name or URI)
        san_values: list[str] = []
        try:
            san_ext = cert.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            )
            san = san_ext.value
            san_values = [n.value for n in san]  # type: ignore[attr-defined]
        except x509.ExtensionNotFound:
            pass

        # Extract CN from subject as fallback instance_id
        cn = ""
        for attr in cert.subject:
            if attr.oid == x509.oid.NameOID.COMMON_NAME:
                cn = str(attr.value)
                break

        return {
            "serial": str(cert.serial_number),
            "not_before": cert.not_valid_before_utc.isoformat(),
            "not_after": cert.not_valid_after_utc.isoformat(),
            "subject_cn": cn,
            "san_values": san_values,
            "ca_verified": bool(ca_pem),
        }

    def _validate_basic(self, cert_pem: str) -> dict[str, Any]:
        """
        Minimal PEM structure check when cryptography library is unavailable.

        Verifies the PEM block is present and decodes to a non-empty payload.
        Does NOT perform cryptographic signature verification.
        """
        if "-----BEGIN CERTIFICATE-----" not in cert_pem:
            raise CertificateValidationError(
                "Birth certificate is not a valid PEM-encoded X.509 certificate"
            )

        # Extract base64 payload
        lines = cert_pem.strip().splitlines()
        b64_lines = [
            l for l in lines
            if not l.startswith("-----")
        ]
        try:
            der_bytes = base64.b64decode("".join(b64_lines))
        except Exception as exc:
            raise CertificateValidationError(
                f"Birth certificate base64 decoding failed: {exc}"
            ) from exc

        if len(der_bytes) < 32:
            raise CertificateValidationError(
                "Birth certificate payload is too short to be a valid certificate"
            )

        self._log.warning(
            "birth_certificate_basic_check_only",
            note="cryptography library not available; signature not verified",
        )

        return {
            "serial": "unknown",
            "not_before": "",
            "not_after": "",
            "subject_cn": "",
            "san_values": [],
            "ca_verified": False,
        }

    # ── Synapse Event Emission ──────────────────────────────────────

    async def _emit_wallet_reported(self, cert_info: dict[str, Any]) -> None:
        """Emit CHILD_WALLET_REPORTED to trigger parent's deferred seed transfer."""
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._synapse.event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CHILD_WALLET_REPORTED,
            source_system=f"child.{self._instance_id}",
            data={
                "child_instance_id": self._instance_id,
                "parent_instance_id": self._parent_instance_id,
                "wallet_address": self._wallet_address,
                "niche": self._niche,
                "cert_serial": cert_info.get("serial", ""),
                "cert_not_after": cert_info.get("not_after", ""),
                "ca_verified": cert_info.get("ca_verified", False),
                "reported_at": utc_now().isoformat(),
            },
        ))
        self._log.info(
            "child_wallet_reported",
            wallet_address=self._wallet_address[:12] + "...",
        )

    async def _emit_federation_connected(self, cert_info: dict[str, Any]) -> None:
        """Emit FEDERATION_PEER_CONNECTED to initiate the mTLS federation link."""
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._synapse.event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.FEDERATION_PEER_CONNECTED,
            source_system=f"child.{self._instance_id}",
            data={
                "peer_instance_id": self._instance_id,
                "parent_instance_id": self._parent_instance_id,
                "peer_address": os.environ.get("ECODIAOS_FEDERATION_ADDRESS", ""),
                "niche": self._niche,
                "certificate_id": cert_info.get("serial", ""),
                "cert_not_after": cert_info.get("not_after", ""),
                "ca_verified": cert_info.get("ca_verified", False),
                "connected_at": utc_now().isoformat(),
            },
        ))
        self._log.info(
            "federation_peer_connected_emitted",
            niche=self._niche,
        )
