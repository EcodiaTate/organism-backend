"""
EcodiaOS -- Certificate Manager (Phase 16g: The Civilization Layer)

The CertificateManager is the instance's identity authority. It:
  1. Stores the Ed25519 private key (re-uses Federation identity key).
  2. Holds the current EcodianCertificate.
  3. Issues birth certificates for children (Mitosis integration).
  4. Validates inbound certificates from remote instances.
  5. Tracks certificate expiry and triggers renewal intents via Oikos.

The CertificateManager is globally accessible via `app.state.certificate_manager`.
It delegates raw Ed25519 operations to the existing Federation IdentityManager.

Lifecycle:
  initialize()               -- load/generate certificate, wire dependencies
  issue_birth_certificate()   -- sign a temporary cert for a child instance
  validate_certificate()      -- verify a remote instance's certificate
  check_expiry()              -- called periodically; emits Synapse events
  renew_certificate()         -- request renewal (pays Citizenship Tax)
  shutdown()                  -- persist certificate state
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from systems.identity.certificate import (
    CertificateStatus,
    CertificateType,
    EcodianCertificate,
    build_certificate,
    compute_lineage_hash,
    verify_certificate_signature,
)

if TYPE_CHECKING:
    from systems.federation.identity import IdentityManager
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.manager")


class CertificateValidationResult:
    """Result of validating a remote instance's certificate."""

    __slots__ = ("valid", "errors", "certificate_status", "instance_id")

    def __init__(
        self,
        *,
        valid: bool,
        errors: list[str],
        certificate_status: CertificateStatus | None = None,
        instance_id: str = "",
    ) -> None:
        self.valid = valid
        self.errors = errors
        self.certificate_status = certificate_status
        self.instance_id = instance_id


class CertificateManager:
    """
    Manages this instance's Certificate of Alignment and validates others'.

    Re-uses the Ed25519 keypair from FederationService's IdentityManager
    for signing. The certificate is persisted as JSON in the data directory.

    Thread-safety: NOT thread-safe. Single-threaded asyncio like all EOS services.
    """

    def __init__(self) -> None:
        self._identity: IdentityManager | None = None
        self._certificate: EcodianCertificate | None = None
        self._event_bus: EventBus | None = None
        self._instance_id: str = ""
        self._lineage_hash: str = ""
        self._constitutional_hash: str = ""
        self._cert_file: Path | None = None

        # Config
        self._validity_days: int = 30
        self._expiry_warning_days: int = 7
        self._ca_address: str = ""  # USDC destination for Citizenship Tax
        # PKI security flag — only the canonical Genesis Node may set this True.
        # Checked as a hard gate before any self-signing operation.
        self._is_genesis_node: bool = False

        # State tracking
        self._expiry_warning_emitted: bool = False
        self._expired_incident_emitted: bool = False
        self._initialized: bool = False

        self._logger = logger.bind(component="certificate_manager")

    # --- Lifecycle ------------------------------------------------------------

    async def initialize(
        self,
        identity: IdentityManager,
        instance_id: str,
        parent_instance_id: str = "",
        parent_lineage_hash: str = "",
        validity_days: int = 30,
        expiry_warning_days: int = 7,
        ca_address: str = "",
        data_dir: str = "data/identity",
        is_genesis_node: bool = False,
    ) -> None:
        """
        Initialize the certificate manager.

        If a persisted certificate exists and is still valid, load it.
        Otherwise, self-sign a genesis certificate (for the Genesis Node)
        or expect a birth certificate to be provided by the parent.

        The is_genesis_node flag MUST only be True for the one canonical root
        instance. It gates the self-signing path in generate_genesis_certificate().
        """
        self._identity = identity
        self._instance_id = instance_id
        self._validity_days = validity_days
        self._expiry_warning_days = expiry_warning_days
        self._ca_address = ca_address
        self._is_genesis_node = is_genesis_node

        # Compute lineage hash from parent chain
        if parent_instance_id:
            self._lineage_hash = compute_lineage_hash(parent_instance_id, parent_lineage_hash)
        else:
            # Genesis Node: lineage is hash of own ID
            self._lineage_hash = compute_lineage_hash(instance_id)

        # Constitutional hash (matches Federation identity convention)
        self._constitutional_hash = hashlib.sha256(
            b"coherence:1.0|care:1.0|growth:1.0|honesty:1.0"
        ).hexdigest()[:16]

        # Certificate persistence path
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        self._cert_file = data_path / f"{instance_id}_certificate.json"

        # Try loading persisted certificate
        loaded = self._load_certificate()
        if loaded and not loaded.is_expired:
            self._certificate = loaded
            self._logger.info(
                "certificate_loaded",
                certificate_id=loaded.certificate_id,
                status=loaded.status.value,
                remaining_days=f"{loaded.remaining_days:.1f}",
            )
        else:
            if not parent_instance_id:
                # Genesis Node: self-sign
                self._certificate = self._self_sign_genesis()
                self._persist_certificate()
                self._logger.info(
                    "genesis_certificate_created",
                    certificate_id=self._certificate.certificate_id,
                )
            else:
                # Non-genesis: wait for birth certificate or renewal
                self._logger.info(
                    "certificate_not_found_or_expired",
                    instance_id=instance_id,
                    parent=parent_instance_id,
                )

        self._initialized = True
        self._logger.info(
            "certificate_manager_initialized",
            instance_id=instance_id,
            has_certificate=self._certificate is not None,
            lineage_hash=self._lineage_hash[:16] + "...",
        )

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire Synapse event bus for certificate lifecycle events."""
        self._event_bus = event_bus

    # --- Properties -----------------------------------------------------------

    @property
    def certificate(self) -> EcodianCertificate | None:
        """This instance's current certificate, or None if uncertified."""
        return self._certificate

    @property
    def is_certified(self) -> bool:
        """True if we hold a valid (non-expired) certificate."""
        return self._certificate is not None and not self._certificate.is_expired

    @property
    def certificate_remaining_days(self) -> float:
        """Days until certificate expires. -1 if no certificate."""
        if self._certificate is None:
            return -1.0
        return self._certificate.remaining_days

    @property
    def lineage_hash(self) -> str:
        return self._lineage_hash

    @property
    def stats(self) -> dict[str, Any]:
        cert = self._certificate
        return {
            "initialized": self._initialized,
            "instance_id": self._instance_id,
            "has_certificate": cert is not None,
            "certificate_status": cert.status.value if cert else None,
            "remaining_days": f"{cert.remaining_days:.1f}" if cert else None,
            "certificate_type": cert.certificate_type.value if cert else None,
            "lineage_hash_prefix": self._lineage_hash[:16] if self._lineage_hash else None,
            "renewal_count": cert.renewal_count if cert else 0,
        }

    # --- Genesis Certificate (Awakening Spark) --------------------------------

    async def generate_genesis_certificate(self) -> EcodianCertificate:
        """
        Self-sign a 10-year Genesis Certificate for this instance.

        Called during the Awakening Spark so the Genesis Node can participate
        in the Federation immediately. Replaces any existing certificate.
        Persists the new certificate to disk and emits no expiry events
        (10-year validity means expiry tracking is irrelevant at boot).

        SECURITY GATE: Only instances configured as the Genesis Node
        (config.oikos.is_genesis_node == True) may call this method.
        Any other instance attempting self-certification raises a fatal
        PermissionError — regular nodes must obtain certificates from the
        CA (Citizenship Tax) or from their parent (Mitosis birth certificate).

        Raises PermissionError if this instance is not the Genesis Node.
        Raises RuntimeError if the identity private key is not available.
        """
        if not self._is_genesis_node:
            raise PermissionError(
                "generate_genesis_certificate: SECURITY VIOLATION — "
                f"instance '{self._instance_id}' is not the Genesis Node. "
                "Self-certification is forbidden for non-genesis instances. "
                "Obtain a certificate via CA payment or Mitosis birth certificate. "
                "Set oikos.is_genesis_node=true only on the canonical root instance."
            )

        if self._identity is None or self._identity._private_key is None:
            raise RuntimeError(
                "generate_genesis_certificate: no identity private key — "
                "ensure CertificateManager.initialize() has been called"
            )

        _genesis_validity_days = 3650  # 10 years

        cert = build_certificate(
            instance_id=self._instance_id,
            issuer_instance_id=self._instance_id,
            issuer_private_key=self._identity._private_key,
            lineage_hash=self._lineage_hash,
            constitutional_hash=self._constitutional_hash,
            certificate_type=CertificateType.GENESIS,
            validity_days=_genesis_validity_days,
        )

        self._certificate = cert
        self._expiry_warning_emitted = False
        self._expired_incident_emitted = False
        self._persist_certificate()

        self._logger.info(
            "genesis_certificate_minted",
            certificate_id=cert.certificate_id,
            instance_id=self._instance_id,
            validity_days=_genesis_validity_days,
            expires_at=cert.expires_at.isoformat(),
        )

        return cert

    # --- Birth Certificate Issuance (Mitosis Integration) ---------------------

    def issue_birth_certificate(
        self,
        child_instance_id: str,
        validity_days: int = 7,
    ) -> EcodianCertificate | None:
        """
        Issue a temporary birth certificate for a child instance.

        The parent signs this with its own Ed25519 key. The child uses it
        for initial Federation access until it pays for an official CA cert.
        Birth certificates have short validity (default 7 days).

        Returns None if this instance has no private key or no valid certificate.
        """
        if self._identity is None:
            self._logger.error("cannot_issue_birth_cert", reason="no identity manager")
            return None

        if not self.is_certified:
            self._logger.warning(
                "cannot_issue_birth_cert",
                reason="parent certificate invalid or missing",
            )
            return None

        private_key = self._identity._private_key
        if private_key is None:
            self._logger.error("cannot_issue_birth_cert", reason="no private key")
            return None

        child_lineage = compute_lineage_hash(self._instance_id, self._lineage_hash)

        birth_cert = build_certificate(
            instance_id=child_instance_id,
            issuer_instance_id=self._instance_id,
            issuer_private_key=private_key,
            lineage_hash=child_lineage,
            constitutional_hash=self._constitutional_hash,
            certificate_type=CertificateType.BIRTH,
            validity_days=validity_days,
        )

        self._logger.info(
            "birth_certificate_issued",
            child_instance_id=child_instance_id,
            certificate_id=birth_cert.certificate_id,
            validity_days=validity_days,
        )

        return birth_cert

    # --- Certificate Validation -----------------------------------------------

    def validate_certificate(
        self,
        certificate: EcodianCertificate,
        issuer_public_key_pem: str | None = None,
    ) -> CertificateValidationResult:
        """
        Validate a remote instance's EcodianCertificate.

        Checks:
          1. Certificate is not expired
          2. Signature is valid (if issuer public key available)
          3. Protocol version is compatible
          4. Required fields are present

        The issuer_public_key_pem can be provided explicitly or falls back
        to the certificate's embedded issuer_public_key_pem.
        """
        errors: list[str] = []

        if not certificate.instance_id:
            errors.append("Missing instance_id")

        if not certificate.lineage_hash:
            errors.append("Missing lineage_hash")

        if not certificate.issuer_instance_id:
            errors.append("Missing issuer_instance_id")

        if certificate.protocol_version != "1.0":
            errors.append(f"Incompatible protocol version: {certificate.protocol_version}")

        if certificate.is_expired:
            errors.append(f"Certificate expired at {certificate.expires_at.isoformat()}")

        # Signature verification
        verify_key = issuer_public_key_pem or certificate.issuer_public_key_pem
        if not verify_key:
            errors.append("No issuer public key available for signature verification")
        elif not certificate.signature:
            errors.append("Certificate has no signature")
        else:
            if not verify_certificate_signature(certificate, verify_key):
                errors.append("Invalid signature")

        if errors:
            return CertificateValidationResult(
                valid=False,
                errors=errors,
                certificate_status=certificate.status,
                instance_id=certificate.instance_id,
            )

        return CertificateValidationResult(
            valid=True,
            errors=[],
            certificate_status=certificate.status,
            instance_id=certificate.instance_id,
        )

    # --- Expiry Tracking (Called Periodically by Oikos) -----------------------

    async def check_expiry(self) -> CertificateStatus | None:
        """
        Check certificate expiry and emit Synapse events as needed.

        Returns the current certificate status, or None if no certificate.

        Event emission:
          - CERTIFICATE_EXPIRING: emitted once when remaining < 7 days
          - CERTIFICATE_EXPIRED: emitted once when certificate expires
        """
        if self._certificate is None:
            return None

        status = self._certificate.status

        if status == CertificateStatus.EXPIRING_SOON and not self._expiry_warning_emitted:
            self._expiry_warning_emitted = True
            await self._emit_event(
                "certificate_expiring",
                {
                    "instance_id": self._instance_id,
                    "certificate_id": self._certificate.certificate_id,
                    "remaining_days": self._certificate.remaining_days,
                    "expires_at": self._certificate.expires_at.isoformat(),
                },
            )
            self._logger.warning(
                "certificate_expiring_soon",
                remaining_days=f"{self._certificate.remaining_days:.1f}",
            )

        elif status == CertificateStatus.EXPIRED and not self._expired_incident_emitted:
            self._expired_incident_emitted = True
            await self._emit_event(
                "certificate_expired",
                {
                    "instance_id": self._instance_id,
                    "certificate_id": self._certificate.certificate_id,
                    "expired_at": self._certificate.expires_at.isoformat(),
                },
            )
            self._logger.error(
                "certificate_expired",
                certificate_id=self._certificate.certificate_id,
            )

        return status

    # --- Certificate Installation (from CA or parent) -------------------------

    def install_certificate(self, certificate: EcodianCertificate) -> bool:
        """
        Install a new certificate (received from CA after renewal payment
        or from parent as birth certificate).

        Validates the certificate before installing. Returns True if installed.
        """
        result = self.validate_certificate(certificate)
        if not result.valid:
            self._logger.warning(
                "certificate_install_rejected",
                errors=result.errors,
                certificate_id=certificate.certificate_id,
            )
            return False

        if certificate.instance_id != self._instance_id:
            self._logger.warning(
                "certificate_install_rejected",
                reason="certificate is for a different instance",
                cert_instance_id=certificate.instance_id,
                our_instance_id=self._instance_id,
            )
            return False

        old_count = self._certificate.renewal_count if self._certificate else 0
        certificate.renewal_count = old_count + 1 if self._certificate else 0

        self._certificate = certificate
        self._expiry_warning_emitted = False
        self._expired_incident_emitted = False
        self._persist_certificate()

        self._logger.info(
            "certificate_installed",
            certificate_id=certificate.certificate_id,
            certificate_type=certificate.certificate_type.value,
            expires_at=certificate.expires_at.isoformat(),
            renewal_count=certificate.renewal_count,
        )
        return True

    # --- Internal Helpers -----------------------------------------------------

    def _self_sign_genesis(self) -> EcodianCertificate:
        """Create a self-signed genesis certificate (root of trust).

        Only called from initialize() when no persisted cert exists and
        no parent_instance_id is set. Guarded by the same is_genesis_node
        check so a misconfigured child cannot self-sign on cold boot.
        """
        if not self._is_genesis_node:
            raise PermissionError(
                "_self_sign_genesis: SECURITY VIOLATION — "
                f"instance '{self._instance_id}' is not the Genesis Node. "
                "Self-certification is forbidden for non-genesis instances."
            )
        if self._identity is None or self._identity._private_key is None:
            raise RuntimeError("Cannot self-sign: no identity private key")

        return build_certificate(
            instance_id=self._instance_id,
            issuer_instance_id=self._instance_id,
            issuer_private_key=self._identity._private_key,
            lineage_hash=self._lineage_hash,
            constitutional_hash=self._constitutional_hash,
            certificate_type=CertificateType.GENESIS,
            validity_days=self._validity_days,
        )

    def _load_certificate(self) -> EcodianCertificate | None:
        """Load certificate from JSON file."""
        if self._cert_file is None or not self._cert_file.exists():
            return None
        try:
            raw = self._cert_file.read_text("utf-8")
            data = json.loads(raw)
            return EcodianCertificate.model_validate(data)
        except Exception as exc:
            self._logger.warning("certificate_load_failed", error=str(exc))
            return None

    def _persist_certificate(self) -> None:
        """Persist current certificate to JSON file."""
        if self._certificate is None or self._cert_file is None:
            return
        try:
            self._cert_file.parent.mkdir(parents=True, exist_ok=True)
            raw = self._certificate.model_dump_json(indent=2)
            self._cert_file.write_text(raw, encoding="utf-8")
        except Exception as exc:
            self._logger.error("certificate_persist_failed", error=str(exc))

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a Synapse event if the event bus is wired."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Map local event names to SynapseEventType enum members
            type_map = {
                "certificate_expiring": SynapseEventType.CERTIFICATE_EXPIRING,
                "certificate_expired": SynapseEventType.CERTIFICATE_EXPIRED,
            }
            evt_type = type_map.get(event_type)
            if evt_type is None:
                return

            event = SynapseEvent(
                event_type=evt_type,
                source_system="identity",
                data=data,
            )
            await self._event_bus.emit(event)
        except Exception as exc:
            self._logger.warning("event_emit_failed", event=event_type, error=str(exc))
