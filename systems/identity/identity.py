"""
EcodiaOS - Identity System Service (Spec 23)

The organism's cryptographic identity authority. Manages:
  - Neo4j-persisted Identity node (organism UUID, constitutional hash, generation, lineage)
  - Dynamic constitutional hash computed from the actual constitutional document
  - Certificate renewal via CA endpoint (initially self-signed fallback)
  - Synapse connector events (IDENTITY_VERIFIED, IDENTITY_CHALLENGED, IDENTITY_EVOLVED)
  - Evo signals (IDENTITY_DRIFT_DETECTED) for population-level identity diversity
  - Lifecycle subscriptions (GENOME_EXTRACT_REQUEST, ORGANISM_SLEEP, ORGANISM_SPAWNED)

The Identity node in Neo4j is the organism's canonical self-record - it knows
what it is, where it came from, and how its constitution has evolved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID, new_id, utc_now
from systems.identity.communication import OTPCoordinator

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from config import EcodiaOSConfig
    from systems.identity.account_provisioner import AccountProvisioner
    from systems.identity.vault import IdentityVault
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.system")

# Default constitutional document path (relative to project root)
_DEFAULT_CONSTITUTION_PATH = ".claude/EcodiaOS_Identity_Document.md"

# Drift detection threshold - below this coherence score, emit IDENTITY_DRIFT_DETECTED
_DRIFT_THRESHOLD = 0.7


def compute_constitutional_hash(constitution_path: str | Path | None = None) -> str:
    """
    Compute SHA-256 hash of the actual constitutional document.

    Falls back to hashing the four immutable drive names if the document
    is not found, ensuring deterministic reproducibility.
    """
    if constitution_path is not None:
        path = Path(constitution_path)
        if path.exists():
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()

    # Fallback: hash the four immutable constitutional drives
    fallback = "Coherence|Care|Growth|Honesty"
    return hashlib.sha256(fallback.encode("utf-8")).hexdigest()


class IdentitySystem:
    """
    The organism's identity authority - Neo4j persistence, constitutional
    hash management, certificate lifecycle, and Synapse event integration.

    Thread-safety: NOT thread-safe. Single-threaded asyncio like all EOS services.
    """

    def __init__(self) -> None:
        self._neo4j: Neo4jClient | None = None
        self._event_bus: EventBus | None = None
        self._instance_id: str = ""
        self._generation: int = 1
        self._parent_instance_id: str | None = None
        self._birth_timestamp: str = ""
        self._constitutional_hash: str = ""
        self._constitution_path: str | Path = _DEFAULT_CONSTITUTION_PATH
        self._certificate_chain_ref: str = ""
        self._initialized: bool = False
        self._log = logger.bind(system="identity")
        self._otp_coordinator: OTPCoordinator = OTPCoordinator()
        self._account_provisioner: AccountProvisioner | None = None
        self._vault: IdentityVault | None = None
        self._full_config: EcodiaOSConfig | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def initialize(
        self,
        instance_id: str,
        neo4j: Neo4jClient | None = None,
        event_bus: EventBus | None = None,
        generation: int = 1,
        parent_instance_id: str | None = None,
        constitution_path: str | Path | None = None,
        certificate_chain_ref: str = "",
    ) -> None:
        """
        Boot the identity system.

        1. Compute constitutional hash from actual document
        2. Persist or update Identity node in Neo4j
        3. Wire Synapse subscriptions
        """
        self._instance_id = instance_id
        self._neo4j = neo4j
        self._event_bus = event_bus
        self._generation = generation
        self._parent_instance_id = parent_instance_id
        self._certificate_chain_ref = certificate_chain_ref
        self._birth_timestamp = utc_now().isoformat()

        if constitution_path is not None:
            self._constitution_path = constitution_path

        # Compute constitutional hash from actual document
        self._constitutional_hash = compute_constitutional_hash(self._constitution_path)

        # Persist Identity node to Neo4j
        await self._persist_identity_node()

        # Wire Synapse subscriptions
        self._subscribe_to_lifecycle_events()

        # Wire OTP coordinator if bus is available at init time
        if event_bus is not None:
            self._otp_coordinator.set_event_bus(event_bus)

        self._initialized = True
        self._log.info(
            "identity_system_initialized",
            instance_id=instance_id,
            generation=generation,
            constitutional_hash=self._constitutional_hash[:16] + "...",
            parent=parent_instance_id,
        )

        # Autonomous platform identity provisioning - runs in background on first boot.
        # Checks if this instance already has its own GitHub account / phone number,
        # and provisions them if not. Never blocks the boot sequence.
        if self._account_provisioner is not None:
            import asyncio
            asyncio.ensure_future(self._run_platform_provisioning())

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire Synapse event bus after initialization."""
        self._event_bus = event_bus
        self._subscribe_to_lifecycle_events()
        self._otp_coordinator.set_event_bus(event_bus)
        if self._account_provisioner is not None:
            self._account_provisioner.set_event_bus(event_bus)

    def set_neo4j(self, neo4j: Neo4jClient) -> None:
        """Wire Neo4j client after initialization."""
        self._neo4j = neo4j
        if self._account_provisioner is not None:
            self._account_provisioner.set_neo4j(neo4j)

    def set_vault(self, vault: "IdentityVault") -> None:
        """Wire IdentityVault for account credential sealing."""
        self._vault = vault

    def set_full_config(self, config: "EcodiaOSConfig") -> None:
        """Wire full EcodiaOSConfig for account provisioner configuration."""
        self._full_config = config

    def set_account_provisioner(self, provisioner: "AccountProvisioner") -> None:
        """Wire the AccountProvisioner (called by registry after init)."""
        self._account_provisioner = provisioner

    # ── Platform Provisioning ───────────────────────────────────────────

    async def _run_platform_provisioning(self) -> None:
        """
        Background task: provision platform identities on first boot.

        Runs after IdentitySystem.initialize() completes, so the rest of the
        boot sequence is never blocked. Failures are logged but never fatal.
        """
        try:
            if self._account_provisioner is None:
                return
            await self._account_provisioner.provision_platform_identities()
        except Exception as exc:
            self._log.warning(
                "platform_provisioning_error",
                error=str(exc),
                instance_id=self._instance_id,
            )

    @property
    def account_provisioner(self) -> "AccountProvisioner | None":
        """Access the AccountProvisioner for direct provisioning calls."""
        return self._account_provisioner

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def constitutional_hash(self) -> str:
        return self._constitutional_hash

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def parent_instance_id(self) -> str | None:
        return self._parent_instance_id

    @property
    def otp_coordinator(self) -> OTPCoordinator:
        """Unified OTP coordination layer - use to await codes from any channel."""
        return self._otp_coordinator

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "instance_id": self._instance_id,
            "generation": self._generation,
            "constitutional_hash": self._constitutional_hash[:16] + "..." if self._constitutional_hash else None,
            "parent_instance_id": self._parent_instance_id,
            "birth_timestamp": self._birth_timestamp,
            "certificate_chain_ref": self._certificate_chain_ref or None,
        }

    # ── Neo4j Persistence ──────────────────────────────────────────────

    async def _persist_identity_node(self) -> None:
        """
        MERGE the Identity node in Neo4j.

        Creates on first boot, updates constitutional_hash and generation
        on subsequent runs. Birth timestamp and parent are immutable after
        creation.
        """
        if self._neo4j is None:
            self._log.warning("neo4j_unavailable", action="persist_identity_node")
            return

        now = utc_now().isoformat()
        try:
            await self._neo4j.execute_write(
                """
                MERGE (i:Identity {instance_id: $instance_id})
                ON CREATE SET
                    i.id = $id,
                    i.instance_id = $instance_id,
                    i.constitutional_hash = $constitutional_hash,
                    i.generation = $generation,
                    i.parent_instance_id = $parent_instance_id,
                    i.birth_timestamp = datetime($birth_timestamp),
                    i.certificate_chain_ref = $certificate_chain_ref,
                    i.created_at = datetime($now),
                    i.updated_at = datetime($now)
                ON MATCH SET
                    i.constitutional_hash = $constitutional_hash,
                    i.generation = $generation,
                    i.certificate_chain_ref = $certificate_chain_ref,
                    i.updated_at = datetime($now)
                """,
                {
                    "id": new_id(),
                    "instance_id": self._instance_id,
                    "constitutional_hash": self._constitutional_hash,
                    "generation": self._generation,
                    "parent_instance_id": self._parent_instance_id or "",
                    "birth_timestamp": self._birth_timestamp,
                    "certificate_chain_ref": self._certificate_chain_ref,
                    "now": now,
                },
            )

            # Create lineage edge to parent if spawned
            if self._parent_instance_id:
                await self._neo4j.execute_write(
                    """
                    MATCH (child:Identity {instance_id: $child_id})
                    MERGE (parent:Identity {instance_id: $parent_id})
                    MERGE (child)-[:SPAWNED_FROM {
                        generation: $generation,
                        created_at: datetime($now)
                    }]->(parent)
                    """,
                    {
                        "child_id": self._instance_id,
                        "parent_id": self._parent_instance_id,
                        "generation": self._generation,
                        "now": now,
                    },
                )

            self._log.info("identity_node_persisted", instance_id=self._instance_id)

        except Exception as exc:
            self._log.error("identity_node_persist_failed", error=str(exc))

    async def load_identity_from_neo4j(self) -> bool:
        """
        Restore identity state from Neo4j on cold boot.

        Returns True if an Identity node was found and loaded.
        """
        if self._neo4j is None:
            return False

        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (i:Identity {instance_id: $instance_id})
                RETURN i.constitutional_hash AS constitutional_hash,
                       i.generation AS generation,
                       i.parent_instance_id AS parent_instance_id,
                       i.birth_timestamp AS birth_timestamp,
                       i.certificate_chain_ref AS certificate_chain_ref
                LIMIT 1
                """,
                {"instance_id": self._instance_id},
            )
            if not rows:
                return False

            row = rows[0]
            self._constitutional_hash = str(row.get("constitutional_hash", ""))
            self._generation = int(row.get("generation", 1))
            parent = row.get("parent_instance_id", "")
            self._parent_instance_id = parent if parent else None
            bt = row.get("birth_timestamp")
            self._birth_timestamp = bt.isoformat() if bt else ""
            self._certificate_chain_ref = str(row.get("certificate_chain_ref", ""))

            self._log.info(
                "identity_loaded_from_neo4j",
                instance_id=self._instance_id,
                generation=self._generation,
            )
            return True

        except Exception as exc:
            self._log.error("identity_load_failed", error=str(exc))
            return False

    # ── Constitutional Hash Management ─────────────────────────────────

    async def recompute_constitutional_hash(
        self,
        constitution_path: str | Path | None = None,
        reason: str = "periodic_recheck",
    ) -> bool:
        """
        Recompute the constitutional hash from the document on disk.

        If the hash changed, persists to Neo4j and emits
        CONSTITUTIONAL_HASH_CHANGED and IDENTITY_EVOLVED events.

        Returns True if the hash changed.
        """
        path = constitution_path or self._constitution_path
        new_hash = compute_constitutional_hash(path)
        old_hash = self._constitutional_hash

        if new_hash == old_hash:
            return False

        self._constitutional_hash = new_hash
        await self._persist_identity_node()

        now = utc_now().isoformat()

        await self._emit_event(
            "CONSTITUTIONAL_HASH_CHANGED",
            {
                "instance_id": self._instance_id,
                "old_hash": old_hash,
                "new_hash": new_hash,
                "timestamp": now,
            },
        )

        await self._emit_event(
            "IDENTITY_EVOLVED",
            {
                "instance_id": self._instance_id,
                "old_hash": old_hash,
                "new_hash": new_hash,
                "generation": self._generation,
                "reason": reason,
                "timestamp": now,
            },
        )

        self._log.info(
            "constitutional_hash_changed",
            old_hash=old_hash[:16] + "...",
            new_hash=new_hash[:16] + "...",
            reason=reason,
        )

        return True

    # ── Identity Verification ──────────────────────────────────────────

    async def verify_identity(self) -> dict[str, Any]:
        """
        Confirm this organism's identity - certificate validity + constitutional hash.

        Emits IDENTITY_VERIFIED on success.
        """
        now = utc_now().isoformat()
        result = {
            "instance_id": self._instance_id,
            "constitutional_hash": self._constitutional_hash,
            "generation": self._generation,
            "timestamp": now,
            "verified": True,
        }

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.IDENTITY_VERIFIED, result)

        self._log.info("identity_verified", instance_id=self._instance_id)
        return result

    async def handle_identity_challenge(
        self,
        challenger: str,
        challenge_type: str = "federation",
    ) -> dict[str, Any]:
        """
        Respond to an identity challenge from a remote or internal system.

        Emits IDENTITY_CHALLENGED, then verifies and returns proof.
        """
        now = utc_now().isoformat()

        await self._emit_event(
            "IDENTITY_CHALLENGED",
            {
                "instance_id": self._instance_id,
                "challenger": challenger,
                "challenge_type": challenge_type,
                "timestamp": now,
            },
        )

        # Perform verification
        return await self.verify_identity()

    # ── Certificate Renewal ────────────────────────────────────────────

    async def renew_certificate(
        self,
        certificate_manager: Any | None = None,
    ) -> bool:
        """
        Renew the organism's certificate via CertificateManager.

        Delegates to CertificateManager.renew_certificate() which handles:
          - GenesisCA issuance (Genesis Node self-renewal)
          - CERTIFICATE_RENEWAL_REQUESTED emission (Oikos coordination)
          - Neo4j persistence via CertificateNeo4jClient
          - CERTIFICATE_RENEWED emission on success

        On success, updates certificate_chain_ref in the Identity node.
        Emits CERTIFICATE_RENEWED here only when manager is unavailable
        but in-process signing succeeds (legacy fallback path).
        """
        if certificate_manager is None:
            self._log.warning("renew_certificate_skipped", reason="no_certificate_manager")
            return False

        try:
            success = await certificate_manager.renew_certificate()

            if success:
                cert = certificate_manager.certificate
                if cert is not None:
                    self._certificate_chain_ref = cert.certificate_id
                    await self._persist_identity_node()

                    now = utc_now().isoformat()
                    await self._emit_event(
                        "CERTIFICATE_RENEWED",
                        {
                            "instance_id": self._instance_id,
                            "certificate_id": cert.certificate_id,
                            "expires_at": cert.expires_at.isoformat(),
                            "renewal_count": cert.renewal_count,
                            "timestamp": now,
                        },
                    )

                    self._log.info(
                        "certificate_renewed",
                        certificate_id=cert.certificate_id,
                        expires_at=cert.expires_at.isoformat(),
                    )

            return success

        except Exception as exc:
            self._log.error("renew_certificate_failed", error=str(exc))
            return False

    # ── Evo Signals ────────────────────────────────────────────────────

    async def check_constitutional_coherence(
        self,
        coherence_score: float,
        drift_dimensions: dict[str, float] | None = None,
    ) -> bool:
        """
        Check constitutional coherence against threshold.

        If coherence_score < _DRIFT_THRESHOLD, emits IDENTITY_DRIFT_DETECTED
        for Evo to incorporate into population-level identity diversity metrics.

        Returns True if drift was detected.
        """
        if coherence_score >= _DRIFT_THRESHOLD:
            return False

        now = utc_now().isoformat()
        await self._emit_event(
            "IDENTITY_DRIFT_DETECTED",
            {
                "instance_id": self._instance_id,
                "coherence_score": coherence_score,
                "threshold": _DRIFT_THRESHOLD,
                "drift_dimensions": drift_dimensions or {},
                "timestamp": now,
            },
        )

        self._log.warning(
            "identity_drift_detected",
            coherence_score=coherence_score,
            threshold=_DRIFT_THRESHOLD,
        )
        return True

    # ── Lifecycle Event Handlers ───────────────────────────────────────

    def _subscribe_to_lifecycle_events(self) -> None:
        """Subscribe to organism lifecycle events via Synapse."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEventType

        self._event_bus.subscribe(
            SynapseEventType.GENOME_EXTRACT_REQUEST,
            self._handle_genome_extract_request,
        )
        self._event_bus.subscribe(
            SynapseEventType.ORGANISM_SLEEP,
            self._handle_organism_sleep,
        )
        self._event_bus.subscribe(
            SynapseEventType.ORGANISM_SPAWNED,
            self._handle_organism_spawned,
        )
        # HIGH #6: Subscribe to CHILD_SPAWNED (Mitosis → Identity)
        # CertificateManager.set_event_bus() also subscribes to this to issue birth certs.
        # IdentitySystem's handler only persists lineage in Neo4j - no cert issuance here.
        self._event_bus.subscribe(
            SynapseEventType.CHILD_SPAWNED,
            self._handle_child_spawned,
        )
        # Receive confirmed cert installations to update certificate_chain_ref
        self._event_bus.subscribe(
            SynapseEventType.CHILD_CERTIFICATE_INSTALLED,
            self._handle_child_certificate_installed,
        )

        self._log.info("lifecycle_subscriptions_wired")

    async def _handle_genome_extract_request(self, event: Any) -> None:
        """Respond to GENOME_EXTRACT_REQUEST with identity genome segment."""
        try:
            from systems.identity.genome import IdentityGenomeExtractor

            extractor = IdentityGenomeExtractor(identity_system=self)
            segment = await extractor.extract_genome_segment()

            from systems.synapse.types import SynapseEvent, SynapseEventType

            response = SynapseEvent(
                event_type=SynapseEventType.GENOME_EXTRACT_RESPONSE,
                source_system="identity",
                data={
                    "request_id": event.data.get("request_id", ""),
                    "segment": segment.model_dump(mode="json"),
                },
            )
            if self._event_bus is not None:
                await self._event_bus.emit(response)

        except Exception as exc:
            self._log.error("genome_extract_request_failed", error=str(exc))

    async def _handle_organism_sleep(self, event: Any) -> None:
        """Persist identity state when organism enters sleep."""
        self._log.info("organism_sleep_received", trigger=event.data.get("trigger", ""))
        await self._persist_identity_node()

    async def _handle_organism_spawned(self, event: Any) -> None:
        """Initialize child identity with parent lineage when a new organism is spawned."""
        child_id = event.data.get("child_instance_id", "")
        parent_id = event.data.get("parent_instance_id", "")
        child_generation = int(event.data.get("generation", self._generation + 1))

        if not child_id or parent_id != self._instance_id:
            return

        self._log.info(
            "child_organism_spawned",
            child_id=child_id,
            generation=child_generation,
        )

        # Record the lineage relationship in Neo4j
        if self._neo4j is not None:
            now = utc_now().isoformat()
            try:
                await self._neo4j.execute_write(
                    """
                    MERGE (child:Identity {instance_id: $child_id})
                    ON CREATE SET
                        child.id = $id,
                        child.constitutional_hash = $constitutional_hash,
                        child.generation = $generation,
                        child.parent_instance_id = $parent_id,
                        child.birth_timestamp = datetime($now),
                        child.created_at = datetime($now),
                        child.updated_at = datetime($now)
                    MERGE (parent:Identity {instance_id: $parent_id})
                    MERGE (child)-[:SPAWNED_FROM {
                        generation: $generation,
                        created_at: datetime($now)
                    }]->(parent)
                    """,
                    {
                        "id": new_id(),
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "constitutional_hash": self._constitutional_hash,
                        "generation": child_generation,
                        "now": now,
                    },
                )
            except Exception as exc:
                self._log.error("child_identity_persist_failed", error=str(exc))

    async def _handle_child_spawned(self, event: Any) -> None:
        """
        HIGH #6: CHILD_SPAWNED handler in IdentitySystem.

        IdentitySystem's role here is limited to Neo4j lineage persistence.
        Birth certificate issuance is handled exclusively by CertificateManager
        (also subscribed to CHILD_SPAWNED). This separation ensures the cert
        pipeline and identity-graph pipeline stay decoupled.
        """
        # Delegate entirely to _handle_organism_spawned - same semantics
        await self._handle_organism_spawned(event)

    async def _handle_child_certificate_installed(self, event: Any) -> None:
        """
        Update certificate_chain_ref in Neo4j when a child certificate is installed.

        This keeps the (:Identity) node current so Federation peers can read
        the latest cert reference without querying the Identity system directly.
        """
        child_id = event.data.get("child_instance_id", "")
        cert_id = event.data.get("certificate_id", "")
        if not child_id or not cert_id:
            return

        if self._neo4j is not None:
            now = utc_now().isoformat()
            try:
                await self._neo4j.execute_write(
                    """
                    MATCH (i:Identity {instance_id: $child_id})
                    SET i.certificate_chain_ref = $cert_id,
                        i.updated_at = datetime($now)
                    """,
                    {"child_id": child_id, "cert_id": cert_id, "now": now},
                )
            except Exception as exc:
                self._log.error(
                    "cert_chain_ref_update_failed",
                    child_id=child_id,
                    error=str(exc),
                )

    # ── Synapse Event Emission ─────────────────────────────────────────

    async def _emit_event(self, event_name: "str | SynapseEventType", data: dict[str, Any]) -> None:
        """Emit a typed Synapse event. Fire-and-forget, failure-tolerant."""
        if self._event_bus is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Accept SynapseEventType enum members directly (bypasses type_map lookup).
            if isinstance(event_name, SynapseEventType):
                evt_type: SynapseEventType = event_name
            else:
                type_map: dict[str, SynapseEventType] = {
                    "IDENTITY_VERIFIED": SynapseEventType.IDENTITY_VERIFIED,
                    "IDENTITY_CHALLENGED": SynapseEventType.IDENTITY_CHALLENGED,
                    "IDENTITY_EVOLVED": SynapseEventType.IDENTITY_EVOLVED,
                    "CONSTITUTIONAL_HASH_CHANGED": SynapseEventType.CONSTITUTIONAL_HASH_CHANGED,
                    "CERTIFICATE_RENEWED": SynapseEventType.CERTIFICATE_RENEWED,
                    "IDENTITY_DRIFT_DETECTED": SynapseEventType.IDENTITY_DRIFT_DETECTED,
                    "CHILD_CERTIFICATE_INSTALLED": SynapseEventType.CHILD_CERTIFICATE_INSTALLED,
                    "CERTIFICATE_RENEWAL_REQUESTED": SynapseEventType.CERTIFICATE_RENEWAL_REQUESTED,
                }

                evt_type = type_map.get(event_name)  # type: ignore[assignment]
                if evt_type is None:
                    self._log.warning("unknown_event_type", event_name=event_name)
                    return

            event = SynapseEvent(
                event_type=evt_type,
                source_system="identity",
                data=data,
            )
            await self._event_bus.emit(event)

        except Exception as exc:
            self._log.warning("event_emit_failed", event=str(event_name), error=str(exc))

    # ── Shutdown ───────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Persist final state before shutdown."""
        await self._persist_identity_node()
        self._log.info("identity_system_shutdown", instance_id=self._instance_id)
