"""
EcodiaOS -- Certificate Manager (Phase 16g: The Civilization Layer)

The CertificateManager is the instance's identity authority. It:
  1. Stores the Ed25519 private key (re-uses Federation identity key).
  2. Holds the current EcodianCertificate.
  3. Issues birth certificates for children (Mitosis integration).
  4. Validates inbound certificates from remote instances.
  5. Tracks certificate expiry and triggers renewal intents via Oikos.
  6. Hosts the GenesisCA (Genesis Node only) - signs official certificates.
  7. Persists :Certificate and :Identity nodes to Neo4j via CertificateNeo4jClient.

The CertificateManager is globally accessible via `app.state.certificate_manager`.
It delegates raw Ed25519 operations to the existing Federation IdentityManager.

Lifecycle:
  initialize()               -- load/generate certificate, wire dependencies
  issue_birth_certificate()   -- sign a temporary cert for a child instance
  validate_certificate()      -- verify a remote instance's certificate
  check_expiry()              -- called periodically; emits Synapse events
  renew_certificate()         -- full renewal lifecycle via GenesisCA
  install_certificate()       -- install CA-issued or birth certificate
  shutdown()                  -- persist certificate state
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.identity.certificate import (
    CertificateStatus,
    CertificateType,
    EcodianCertificate,
    build_certificate,
    compute_lineage_hash,
    verify_certificate_signature,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.federation.identity import IdentityManager
    from systems.identity.ca import GenesisCA
    from systems.identity.vault import IdentityVault
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.manager")


class CertificateNeo4jClient:
    """
    Writes :Certificate and :Identity nodes to Neo4j on issuance/renewal.

    All writes are soft - Neo4j unavailability is logged but never fatal.
    Nodes are linked: (:Identity)-[:HOLDS_CERTIFICATE]->(:Certificate).
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(component="cert_neo4j")

    async def persist_certificate(
        self,
        cert: EcodianCertificate,
        instance_id: str,
    ) -> None:
        now = utc_now().isoformat()
        try:
            await self._neo4j.execute_write(
                """
                MERGE (c:Certificate {certificate_id: $cert_id})
                ON CREATE SET
                    c.id = $id,
                    c.certificate_id = $cert_id,
                    c.instance_id = $instance_id,
                    c.issuer_instance_id = $issuer_id,
                    c.certificate_type = $cert_type,
                    c.constitutional_hash = $const_hash,
                    c.lineage_hash = $lineage_hash,
                    c.issued_at = datetime($issued_at),
                    c.expires_at = datetime($expires_at),
                    c.validity_days = $validity_days,
                    c.renewal_count = $renewal_count,
                    c.created_at = datetime($now)
                ON MATCH SET
                    c.renewal_count = $renewal_count,
                    c.updated_at = datetime($now)
                WITH c
                MERGE (i:Identity {instance_id: $instance_id})
                MERGE (i)-[:HOLDS_CERTIFICATE {
                    issued_at: datetime($issued_at),
                    created_at: datetime($now)
                }]->(c)
                """,
                {
                    "id": new_id(),
                    "cert_id": cert.certificate_id,
                    "instance_id": instance_id,
                    "issuer_id": cert.issuer_instance_id,
                    "cert_type": cert.certificate_type.value,
                    "const_hash": cert.constitutional_hash,
                    "lineage_hash": cert.lineage_hash,
                    "issued_at": cert.issued_at.isoformat(),
                    "expires_at": cert.expires_at.isoformat(),
                    "validity_days": cert.validity_days,
                    "renewal_count": cert.renewal_count,
                    "now": now,
                },
            )
            self._log.info(
                "certificate_persisted_neo4j",
                certificate_id=cert.certificate_id,
                instance_id=instance_id,
            )
        except Exception as exc:
            self._log.error("certificate_neo4j_persist_failed", error=str(exc))


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
        self._ca_key_file: Path | None = None  # Sealed Genesis CA key storage path

        # Injected dependencies (Genesis Node only)
        self._genesis_ca: GenesisCA | None = None
        self._vault: IdentityVault | None = None
        self._neo4j_client: CertificateNeo4jClient | None = None

        # Config
        self._validity_days: int = 30
        self._expiry_warning_days: int = 7
        self._ca_address: str = ""  # USDC destination for Citizenship Tax
        # PKI security flag - only the canonical Genesis Node may set this True.
        # Checked as a hard gate before any self-signing operation.
        self._is_genesis_node: bool = False

        # Pending provisioning approvals: instance_id -> lineage_hash
        self._pending_provisioning: dict[str, str] = {}

        # Pending Equor provisioning approval futures: child_id -> Future
        self._provisioning_approval_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}

        # State tracking
        self._expiry_warning_emitted: bool = False
        self._expired_incident_emitted: bool = False
        self._initialized: bool = False
        self._running: bool = False
        self._expiry_loop_task: asyncio.Task[None] | None = None

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
        vault: IdentityVault | None = None,
        neo4j: Neo4jClient | None = None,
    ) -> None:
        """
        Initialize the certificate manager.

        If a persisted certificate exists and is still valid, load it.
        Otherwise, self-sign a genesis certificate (for the Genesis Node)
        or expect a birth certificate to be provided by the parent.

        On Genesis Nodes, also boots the GenesisCA: loads the sealed CA key
        from disk (if present) or generates a new one on first boot and seals
        it into the vault for persistence.

        The is_genesis_node flag MUST only be True for the one canonical root
        instance. It gates the self-signing path in generate_genesis_certificate().
        """
        self._identity = identity
        self._instance_id = instance_id
        self._validity_days = validity_days
        self._expiry_warning_days = expiry_warning_days
        self._ca_address = ca_address
        self._is_genesis_node = is_genesis_node
        self._vault = vault

        if neo4j is not None:
            self._neo4j_client = CertificateNeo4jClient(neo4j)

        # Compute lineage hash from parent chain
        if parent_instance_id:
            self._lineage_hash = compute_lineage_hash(parent_instance_id, parent_lineage_hash)
        else:
            # Genesis Node: lineage is hash of own ID
            self._lineage_hash = compute_lineage_hash(instance_id)

        # Constitutional hash - computed dynamically from actual document
        from systems.identity.identity import compute_constitutional_hash

        self._constitutional_hash = compute_constitutional_hash()

        # Certificate persistence path
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        self._cert_file = data_path / f"{instance_id}_certificate.json"
        self._ca_key_file = data_path / f"{instance_id}_ca_key.sealed"

        # Boot GenesisCA on Genesis Node
        if is_genesis_node and vault is not None:
            await self._boot_genesis_ca(vault)

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
                if self._neo4j_client is not None:
                    await self._neo4j_client.persist_certificate(
                        self._certificate, self._instance_id
                    )
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
        self._running = True
        self._expiry_loop_task = asyncio.create_task(
            self._expiry_check_loop(), name="certificate_expiry_loop"
        )
        self._logger.info(
            "certificate_manager_initialized",
            instance_id=instance_id,
            has_certificate=self._certificate is not None,
            has_genesis_ca=self._genesis_ca is not None and self._genesis_ca.is_ready,
            lineage_hash=self._lineage_hash[:16] + "...",
        )

    async def shutdown(self) -> None:
        """Stop the background expiry loop gracefully."""
        self._running = False
        if self._expiry_loop_task is not None and not self._expiry_loop_task.done():
            self._expiry_loop_task.cancel()
            try:
                await self._expiry_loop_task
            except asyncio.CancelledError:
                pass
        self._logger.info("certificate_manager_shutdown")

    async def _expiry_check_loop(self) -> None:
        """Background loop: call check_expiry() every 24 hours."""
        _INTERVAL_S = 86400  # 24 hours
        while self._running:
            try:
                await asyncio.sleep(_INTERVAL_S)
                if self._running:
                    await self.check_expiry()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("certificate_expiry_check_failed", error=str(exc))

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire Synapse event bus for certificate lifecycle events."""
        self._event_bus = event_bus
        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEventType

        # HIGH #6: Identity listens for CHILD_SPAWNED - Mitosis emits, Identity issues cert
        self._event_bus.subscribe(
            SynapseEventType.CHILD_SPAWNED,
            self._on_child_spawned,
        )
        # CRITICAL #2: HITL gate - human approved instance provisioning
        self._event_bus.subscribe(
            SynapseEventType.EQUOR_HITL_APPROVED,
            self._on_equor_hitl_approved,
        )
        # M2 (Identity): Equor constitutional review response for child provisioning
        self._event_bus.subscribe(
            SynapseEventType.EQUOR_PROVISIONING_APPROVAL,
            self._on_equor_provisioning_approval,
        )

        # Wire GenesisCA event handler if present
        if self._genesis_ca is not None:
            self._genesis_ca._wire_equor_handler()

        self._logger.info("certificate_manager_events_subscribed")

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
        PermissionError - regular nodes must obtain certificates from the
        CA (Citizenship Tax) or from their parent (Mitosis birth certificate).

        Raises PermissionError if this instance is not the Genesis Node.
        Raises RuntimeError if the identity private key is not available.
        """
        if not self._is_genesis_node:
            raise PermissionError(
                "generate_genesis_certificate: SECURITY VIOLATION - "
                f"instance '{self._instance_id}' is not the Genesis Node. "
                "Self-certification is forbidden for non-genesis instances. "
                "Obtain a certificate via CA payment or Mitosis birth certificate. "
                "Set oikos.is_genesis_node=true only on the canonical root instance."
            )

        if self._identity is None or self._identity._private_key is None:
            raise RuntimeError(
                "generate_genesis_certificate: no identity private key - "
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
        known_certificates: dict[str, EcodianCertificate] | None = None,
    ) -> CertificateValidationResult:
        """
        Validate a remote instance's EcodianCertificate.

        Checks:
          1. Certificate is not expired
          2. Signature is valid (issuer public key from argument or embedded field)
          3. Protocol version is compatible
          4. Required fields are present
          5. Lineage chain walk - recursively verify each issuer up to Genesis CA,
             using known_certificates to resolve parent certs. Returns False if
             any link in the chain has an invalid signature or is expired.

        The issuer_public_key_pem can be provided explicitly or falls back
        to the certificate's embedded issuer_public_key_pem.

        known_certificates: optional mapping of instance_id -> EcodianCertificate
        for chain walking. Without it, only the immediate signature is verified.
        Genesis certs (self-signed: instance_id == issuer_instance_id) terminate
        the walk.
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

        # Immediate signature verification
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

        # Lineage chain walk - verify every ancestor up to Genesis CA
        if known_certificates:
            chain_errors = self._walk_certificate_chain(certificate, known_certificates)
            if chain_errors:
                return CertificateValidationResult(
                    valid=False,
                    errors=chain_errors,
                    certificate_status=certificate.status,
                    instance_id=certificate.instance_id,
                )

        return CertificateValidationResult(
            valid=True,
            errors=[],
            certificate_status=certificate.status,
            instance_id=certificate.instance_id,
        )

    def _walk_certificate_chain(
        self,
        certificate: EcodianCertificate,
        known_certificates: dict[str, EcodianCertificate],
        _visited: set[str] | None = None,
    ) -> list[str]:
        """
        Recursively walk the issuer chain up to the Genesis CA, verifying each link.

        Terminates when:
          - issuer_instance_id == instance_id (self-signed Genesis cert), OR
          - issuer_instance_id is not in known_certificates (chain truncated - not an error;
            we cannot verify what we don't have, so we trust as far as we can).

        Returns a list of error strings; empty list means the chain is valid.
        Detects cycles via the _visited set to prevent infinite loops.
        """
        if _visited is None:
            _visited = set()

        current = certificate
        errors: list[str] = []

        while True:
            issuer_id = current.issuer_instance_id

            # Genesis cert - self-signed root, chain terminates here
            if issuer_id == current.instance_id:
                # Verify the self-signature once more at the root
                if not verify_certificate_signature(current, current.issuer_public_key_pem):
                    errors.append(
                        f"Genesis cert self-signature invalid for instance {current.instance_id}"
                    )
                break

            # Cycle guard
            if issuer_id in _visited:
                errors.append(f"Certificate chain cycle detected at issuer {issuer_id}")
                break
            _visited.add(issuer_id)

            # Resolve issuer cert from known registry
            issuer_cert = known_certificates.get(issuer_id)
            if issuer_cert is None:
                # Chain is truncated - we accept what we can verify
                self._logger.debug(
                    "certificate_chain_truncated",
                    at_issuer=issuer_id,
                    subject=current.instance_id,
                )
                break

            # Verify the current cert's signature using the issuer's embedded public key
            issuer_key = issuer_cert.issuer_public_key_pem
            if not issuer_key:
                errors.append(f"Issuer cert for {issuer_id} has no public key")
                break

            if issuer_cert.is_expired:
                errors.append(
                    f"Issuer cert for {issuer_id} expired at {issuer_cert.expires_at.isoformat()}"
                )
                break

            if not verify_certificate_signature(current, issuer_key):
                errors.append(
                    f"Signature invalid: cert for {current.instance_id} "
                    f"failed verification against issuer {issuer_id}"
                )
                break

            # Walk up one level
            current = issuer_cert

        return errors

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

        # Notify Federation so it can update its cached certificate fingerprint.
        # Fire-and-forget since install_certificate() is synchronous.
        if self._event_bus is not None:
            asyncio.ensure_future(self._emit_event(
                "identity_certificate_rotated",
                {
                    "new_fingerprint": certificate.certificate_id,
                    "certificate_type": certificate.certificate_type.value,
                    "expires_at": certificate.expires_at.isoformat(),
                    "instance_id": self._instance_id,
                },
            ))

        return True

    # --- Certificate Renewal (MEDIUM #9 + HIGH #5) ----------------------------

    async def renew_certificate(self) -> bool:
        """
        Full renewal lifecycle for this instance's certificate.

        Flow:
          1. Emit CERTIFICATE_RENEWAL_REQUESTED (Oikos coordination signal).
          2. If we are the Genesis Node and have a GenesisCA, self-renew.
          3. Otherwise, queue a pending renewal - the CA (Genesis Node) will
             respond to EQUOR_HITL_APPROVED which triggers ca.issue_certificate().
          4. On success: install cert, persist to Neo4j, emit CERTIFICATE_RENEWED.

        Returns True if renewal succeeded (cert installed), False otherwise.
        """
        if self._certificate is not None:
            await self._emit_event(
                "certificate_renewal_requested",
                {
                    "instance_id": self._instance_id,
                    "certificate_id": self._certificate.certificate_id,
                    "reason": "renewal",
                    "expires_at": self._certificate.expires_at.isoformat(),
                },
            )

        # Genesis Node renews itself via GenesisCA
        if self._is_genesis_node and self._genesis_ca is not None and self._genesis_ca.is_ready:
            try:
                cert = await self._genesis_ca.issue_certificate(
                    instance_id=self._instance_id,
                    lineage_hash=self._lineage_hash,
                    validity_days=self._validity_days,
                )
                installed = self.install_certificate(cert)
                if installed and self._neo4j_client is not None:
                    await self._neo4j_client.persist_certificate(cert, self._instance_id)
                return installed
            except Exception as exc:
                self._logger.error("genesis_self_renewal_failed", error=str(exc))
                return False

        # Non-genesis: cannot self-renew; signal needs external CA
        self._logger.info(
            "renewal_queued_awaiting_ca",
            instance_id=self._instance_id,
        )
        return False

    # --- Synapse Event Handlers -----------------------------------------------

    async def _on_child_spawned(self, event: Any) -> None:
        """
        HIGH #6 + M2: Issue birth certificate when Mitosis emits CHILD_SPAWNED,
        gated by Equor constitutional review of the child's inherited drives.

        Flow:
          1. Extract child_id and inherited_drives from event payload.
          2. Emit CERTIFICATE_PROVISIONING_REQUEST to Equor for constitutional review.
          3. Await EQUOR_PROVISIONING_APPROVAL (30 s timeout).
          4a. Approved + no HITL required → issue birth cert directly.
          4b. Approved + HITL required → store in pending_provisioning; human
              approval via EQUOR_HITL_APPROVED continues the existing flow.
          4c. Rejected or timed out → emit PROVISIONING_REQUIRES_HUMAN_ESCALATION.

        Identity still owns cert issuance - Mitosis must NOT call
        issue_birth_certificate() directly.
        """
        child_id = event.data.get("child_instance_id", "")
        if not child_id:
            return

        inherited_drives: dict[str, Any] = event.data.get("inherited_drives", {})

        # Step 2 - ask Equor to review the child's drive alignment
        await self._emit_event(
            "certificate_provisioning_request",
            {
                "child_id": child_id,
                "provisioning_type": "birth_certificate",
                "inherited_drives": inherited_drives,
                "requires_amendment_approval": bool(inherited_drives),
            },
        )

        # Step 3 - wait for Equor's verdict (non-blocking via Future)
        approval = await self._wait_for_equor_approval(child_id, timeout_s=30)

        if approval is None or not approval.get("approved", False):
            # Rejected or timed out - flag for human review
            reason = "equor_timeout" if approval is None else approval.get("reason", "equor_rejection")
            self._logger.warning(
                "child_provisioning_equor_rejected",
                child_instance_id=child_id,
                reason=reason,
            )
            await self._emit_event(
                "provisioning_requires_human_escalation",
                {"child_id": child_id, "reason": reason},
            )
            return

        if approval.get("requires_hitl", False):
            # Equor approved constitutionally but wants a human to confirm.
            # Store lineage so _handle_provisioning_approved can pick it up.
            child_lineage = compute_lineage_hash(self._instance_id, self._lineage_hash)
            self.register_pending_provisioning(child_id, child_lineage)
            self._logger.info(
                "child_provisioning_awaiting_hitl",
                child_instance_id=child_id,
                required_amendments=approval.get("required_amendments", []),
            )
            return

        # Fast path - Equor approved, no HITL needed; issue birth cert immediately
        birth_cert = self.issue_birth_certificate(child_id)
        if birth_cert is None:
            self._logger.error(
                "birth_cert_issuance_failed_on_child_spawned",
                child_instance_id=child_id,
            )
            return

        if self._neo4j_client is not None:
            await self._neo4j_client.persist_certificate(birth_cert, child_id)

        await self._emit_event(
            "child_certificate_installed",
            {
                "child_instance_id": child_id,
                "certificate_id": birth_cert.certificate_id,
                "certificate_type": birth_cert.certificate_type.value,
                "expires_at": birth_cert.expires_at.isoformat(),
                "issuer_instance_id": birth_cert.issuer_instance_id,
                "constitutional_hash": approval.get("constitutional_hash", ""),
            },
        )
        self._logger.info(
            "birth_cert_issued_and_announced",
            child_instance_id=child_id,
            certificate_id=birth_cert.certificate_id,
            equor_reason=approval.get("reason", ""),
        )

    async def _wait_for_equor_approval(
        self, child_id: str, timeout_s: float = 30.0
    ) -> dict[str, Any] | None:
        """
        Await EQUOR_PROVISIONING_APPROVAL for child_id, up to timeout_s seconds.

        Uses an asyncio.Future keyed on child_id. The Future is resolved by
        _on_equor_provisioning_approval() when Equor emits the matching event.

        Returns the approval payload dict, or None on timeout.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._provisioning_approval_futures[child_id] = future
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._logger.warning(
                "equor_provisioning_approval_timeout",
                child_instance_id=child_id,
                timeout_s=timeout_s,
            )
            return None
        finally:
            self._provisioning_approval_futures.pop(child_id, None)

    async def _on_equor_provisioning_approval(self, event: Any) -> None:
        """
        Resolve the pending Future when Equor emits EQUOR_PROVISIONING_APPROVAL.

        Keyed on child_id so parallel provisioning requests don't collide.
        """
        child_id = event.data.get("child_id", "")
        future = self._provisioning_approval_futures.get(child_id)
        if future is not None and not future.done():
            future.set_result(event.data)

    async def _on_equor_hitl_approved(self, event: Any) -> None:
        """
        Handle EQUOR_HITL_APPROVED for both instance provisioning and Citizenship Tax.

        Equor emits this when a human operator approves a suspended intent.

        Handled approval_type values:
          - "instance_provisioning": Issue an official certificate for a pending
            child instance via GenesisCA (CRITICAL #2).
          - "citizenship_tax_paid": Oikos confirmed the child paid the Citizenship
            Tax - issue an official certificate for the child so it can rejoin
            Federation as a fully certified member (MEDIUM SG4).

        Only the Genesis Node (with a live GenesisCA) performs CA issuance here.
        Non-genesis nodes receiving this event ignore it (not their role).
        """
        approval_type = event.data.get("approval_type", "")

        if approval_type == "instance_provisioning":
            await self._handle_provisioning_approved(event)
        elif approval_type == "citizenship_tax_paid":
            await self._handle_citizenship_tax_approved(event)

    async def _handle_provisioning_approved(self, event: Any) -> None:
        """Issue official cert after HITL approval of new instance provisioning."""
        if not self._is_genesis_node or self._genesis_ca is None:
            return

        child_id = event.data.get("instance_id", "") or event.data.get("subject_instance_id", "")
        if not child_id:
            self._logger.warning(
                "hitl_provisioning_approved_no_instance_id",
                intent_id=event.data.get("intent_id", ""),
            )
            return

        # Resolve lineage for the child - use pending store or compute from our lineage
        child_lineage = self._pending_provisioning.pop(
            child_id,
            compute_lineage_hash(self._instance_id, self._lineage_hash),
        )

        try:
            cert = await self._genesis_ca.issue_certificate(
                instance_id=child_id,
                lineage_hash=child_lineage,
            )
        except Exception as exc:
            self._logger.error(
                "hitl_ca_issuance_failed",
                child_instance_id=child_id,
                error=str(exc),
            )
            return

        if self._neo4j_client is not None:
            await self._neo4j_client.persist_certificate(cert, child_id)

        # Announce the installation - child Identity subscribes to this event
        await self._emit_event(
            "child_certificate_installed",
            {
                "child_instance_id": child_id,
                "certificate_id": cert.certificate_id,
                "certificate_type": cert.certificate_type.value,
                "expires_at": cert.expires_at.isoformat(),
                "issuer_instance_id": cert.issuer_instance_id,
            },
        )

        self._logger.info(
            "official_cert_issued_after_hitl_approval",
            child_instance_id=child_id,
            certificate_id=cert.certificate_id,
        )

    async def _handle_citizenship_tax_approved(self, event: Any) -> None:
        """
        MEDIUM SG4: Issue official cert after Oikos confirms Citizenship Tax payment.

        Oikos emits EQUOR_HITL_APPROVED with approval_type=="citizenship_tax_paid"
        once the child's USDC Citizenship Tax transfer is confirmed on-chain.
        The Genesis CA then issues a fresh official certificate so the child can
        rejoin Federation as a fully certified member.

        Expected event.data fields:
          - instance_id / subject_instance_id: the child that paid the tax
          - lineage_hash (optional): child's lineage hash for the cert
          - validity_days (optional): override cert validity (defaults to CA default)
        """
        if not self._is_genesis_node or self._genesis_ca is None:
            return

        child_id = event.data.get("instance_id", "") or event.data.get("subject_instance_id", "")
        if not child_id:
            self._logger.warning(
                "citizenship_tax_approved_no_instance_id",
                intent_id=event.data.get("intent_id", ""),
            )
            return

        # Prefer an explicit lineage hash from the event; fall back to computing
        # from our own lineage (child spawned from genesis).
        child_lineage = event.data.get("lineage_hash", "") or self._pending_provisioning.pop(
            child_id,
            compute_lineage_hash(self._instance_id, self._lineage_hash),
        )

        validity_days = int(event.data.get("validity_days", 0)) or None  # None = CA default

        try:
            cert = await self._genesis_ca.issue_certificate(
                instance_id=child_id,
                lineage_hash=child_lineage,
                **({"validity_days": validity_days} if validity_days else {}),
            )
        except Exception as exc:
            self._logger.error(
                "citizenship_tax_ca_issuance_failed",
                child_instance_id=child_id,
                error=str(exc),
            )
            return

        if self._neo4j_client is not None:
            await self._neo4j_client.persist_certificate(cert, child_id)

        await self._emit_event(
            "child_certificate_installed",
            {
                "child_instance_id": child_id,
                "certificate_id": cert.certificate_id,
                "certificate_type": cert.certificate_type.value,
                "expires_at": cert.expires_at.isoformat(),
                "issuer_instance_id": cert.issuer_instance_id,
                "reason": "citizenship_tax_paid",
            },
        )

        self._logger.info(
            "official_cert_issued_after_citizenship_tax",
            child_instance_id=child_id,
            certificate_id=cert.certificate_id,
        )

    def register_pending_provisioning(self, child_instance_id: str, lineage_hash: str) -> None:
        """
        Register a child instance that is pending HITL provisioning approval.

        Called by the provisioning pipeline (Axon/Oikos) before submitting
        the Intent to Equor, so the CA has the lineage ready when approval arrives.
        """
        self._pending_provisioning[child_instance_id] = lineage_hash
        self._logger.info(
            "provisioning_registered",
            child_instance_id=child_instance_id,
        )

    # --- Internal Helpers -----------------------------------------------------

    async def _boot_genesis_ca(self, vault: IdentityVault) -> None:
        """
        Boot the GenesisCA on first boot or restore from vault-sealed key file.

        On first boot: generates new Ed25519 keypair, seals it, writes to disk.
        On cold restart: reads sealed bytes from disk, decrypts via vault, restores.
        """
        from systems.identity.ca import GenesisCA

        ca = GenesisCA(
            instance_id=self._instance_id,
            vault=vault,
            event_bus=self._event_bus,
        )

        if self._ca_key_file is not None and self._ca_key_file.exists():
            # Restore from sealed key file
            try:
                sealed_bytes = self._ca_key_file.read_bytes()
                pem = GenesisCA.unseal_private_key(sealed_bytes, vault)
                await ca.initialize(existing_key_pem=pem)
                self._logger.info("genesis_ca_restored_from_vault")
            except Exception as exc:
                self._logger.error("genesis_ca_restore_failed", error=str(exc))
                return
        else:
            # First boot - generate and seal
            await ca.initialize()
            if self._ca_key_file is not None:
                try:
                    sealed = ca.seal_private_key()
                    self._ca_key_file.write_bytes(sealed)
                    self._logger.info(
                        "genesis_ca_key_sealed",
                        path=str(self._ca_key_file),
                    )
                except Exception as exc:
                    self._logger.error("genesis_ca_key_seal_failed", error=str(exc))

        self._genesis_ca = ca

    def _self_sign_genesis(self) -> EcodianCertificate:
        """Create a self-signed genesis certificate (root of trust).

        Only called from initialize() when no persisted cert exists and
        no parent_instance_id is set. Guarded by the same is_genesis_node
        check so a misconfigured child cannot self-sign on cold boot.
        """
        if not self._is_genesis_node:
            raise PermissionError(
                "_self_sign_genesis: SECURITY VIOLATION - "
                f"instance '{self._instance_id}' is not the Genesis Node. "
                "Self-certification is forbidden for non-genesis instances."
            )
        if self._identity is None or self._identity._private_key is None:
            raise RuntimeError("Cannot self-sign: no identity private key")

        # MEDIUM #8: Genesis certificate must be 10 years (3650 days), not the
        # configured validity_days default (30). Hard-coded to prevent accidental
        # short-lived genesis certs on misconfigured deployments.
        _GENESIS_VALIDITY_DAYS = 3650

        return build_certificate(
            instance_id=self._instance_id,
            issuer_instance_id=self._instance_id,
            issuer_private_key=self._identity._private_key,
            lineage_hash=self._lineage_hash,
            constitutional_hash=self._constitutional_hash,
            certificate_type=CertificateType.GENESIS,
            validity_days=_GENESIS_VALIDITY_DAYS,
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
                "child_certificate_installed": SynapseEventType.CHILD_CERTIFICATE_INSTALLED,
                "certificate_renewal_requested": SynapseEventType.CERTIFICATE_RENEWAL_REQUESTED,
                "certificate_renewed": SynapseEventType.CERTIFICATE_RENEWED,
                "identity_certificate_rotated": SynapseEventType.IDENTITY_CERTIFICATE_ROTATED,
                "certificate_provisioning_request": SynapseEventType.CERTIFICATE_PROVISIONING_REQUEST,
                "provisioning_requires_human_escalation": SynapseEventType.PROVISIONING_REQUIRES_HUMAN_ESCALATION,
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
