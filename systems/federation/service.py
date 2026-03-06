"""
EcodiaOS — Federation Service

The Federation Protocol governs how EOS instances relate to each other —
as sovereign entities that can choose to share knowledge, coordinate action,
and build relationships. Not a hive mind. Not isolated. A community of
individuals, each with their own identity, personality, and community,
choosing to help each other grow.

The FederationService orchestrates five sub-systems:
  IdentityManager   — Instance identity cards, Ed25519 signing, verification
  TrustManager      — Trust scoring, level transitions, decay
  PrivacyFilter     — PII removal, consent enforcement
  KnowledgeExchange — Knowledge request/response protocol
  CoordinationMgr   — Assistance requests and coordinated action
  ChannelManager    — Mutual TLS channels to remote instances

Lifecycle:
  initialize()             — build all sub-systems, load keys
  establish_link()         — connect to a remote instance
  withdraw_link()          — disconnect from a remote instance
  handle_knowledge_req()   — handle inbound knowledge request
  request_knowledge()      — request knowledge from remote
  handle_assistance_req()  — handle inbound assistance request
  request_assistance()     — request assistance from remote
  shutdown()               — graceful shutdown

Performance targets:
  Identity verification: ≤500ms
  Knowledge request handling: ≤2000ms
  Trust update: ≤50ms
  Link establishment: ≤3000ms
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from primitives.federation import (
    AssistanceRequest,
    AssistanceResponse,
    ExchangeDirection,
    ExchangeEnvelope,
    ExchangePayloadKind,
    ExchangeReceipt,
    FederationInteraction,
    FederationLink,
    FederationLinkStatus,
    IngestionVerdict,
    InstanceIdentityCard,
    InteractionOutcome,
    KnowledgeRequest,
    KnowledgeResponse,
    KnowledgeType,
    ThreatAdvisory,
    TrustLevel,
    TrustPolicy,
)
from systems.federation.channel import ChannelManager
from systems.federation.coordination import CoordinationManager
from systems.federation.exchange import (
    ExchangeProtocol,
    collect_economic_intel,
    collect_hypotheses,
    collect_mutation_patterns,
    collect_procedures,
)
from systems.federation.handshake import (
    HandshakeConfirmation,
    HandshakeProcessor,
    HandshakeRequest,
    HandshakeResponse,
)
from systems.federation.identity import IdentityManager
from systems.federation.ingestion import IngestionPipeline
from systems.federation.knowledge import KnowledgeExchangeManager
from systems.federation.privacy import PrivacyFilter
from systems.federation.threat_intelligence import ThreatIntelligenceManager
from systems.federation.trust import TrustManager

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from clients.wallet import WalletClient
    from config import FederationConfig
    from systems.equor.service import EquorService
    from systems.memory.service import MemoryService
    from telemetry.metrics import MetricCollector

logger = structlog.get_logger("systems.federation")


class FederationService:
    """
    Federation — the EOS diplomatic system.

    Coordinates identity, trust, knowledge exchange, coordinated action,
    and privacy filtering across federation links with other EOS instances.

    Every federation action goes through Equor. The constitutional drives
    apply to inter-instance relations just as they do to individual
    interactions. An instance cannot be helpful to a federation partner
    at the expense of its own community (Care drive).
    """

    system_id: str = "federation"

    def __init__(
        self,
        config: FederationConfig,
        memory: MemoryService | None = None,
        equor: EquorService | None = None,
        redis: RedisClient | None = None,
        metrics: MetricCollector | None = None,
        wallet: WalletClient | None = None,
        instance_id: str = "",
    ) -> None:
        self._config = config
        self._memory = memory
        self._equor = equor
        self._redis = redis
        self._metrics = metrics
        self._wallet = wallet
        self._instance_id = instance_id
        self._logger = logger.bind(system="federation")
        self._initialized: bool = False
        self._atune: Any = None  # Wired post-init for perception of federated knowledge

        # Sub-systems (built in initialize())
        self._identity: IdentityManager | None = None
        self._trust: TrustManager | None = None
        self._privacy: PrivacyFilter | None = None
        self._knowledge: KnowledgeExchangeManager | None = None
        self._coordination: CoordinationManager | None = None
        self._channels: ChannelManager | None = None
        self._threat_intel: ThreatIntelligenceManager | None = None
        self._staking: Any = None  # ReputationStakingManager, built in initialize()
        self._exchange: ExchangeProtocol | None = None
        self._ingestion: IngestionPipeline | None = None

        # External references for IIEP exchange collection
        self._evo: Any = None   # Wired post-init
        self._simula: Any = None  # Wired post-init
        self._oikos: Any = None  # Wired post-init

        # Active federation links (link_id → FederationLink)
        self._links: dict[str, FederationLink] = {}

        # Interaction history (for audit, limited ring buffer)
        self._interaction_history: list[FederationInteraction] = []
        self._max_history: int = 1000

        # Phase 16g: Certificate validation for inbound communications
        self._certificate_manager: Any = None  # CertificateManager, wired post-init

        # Handshake processor (built after identity is initialized)
        self._handshake: HandshakeProcessor | None = None

        # Pending inbound handshakes: handshake_id → (responder_nonce, initiator_public_key_pem, initiator_instance_id, initiator_name, initiator_endpoint)
        # Kept until Phase 4 confirmation arrives or TTL expires.
        self._pending_handshakes: dict[str, tuple[str, str, str, str, str]] = {}

    def set_certificate_manager(self, cert_mgr: Any) -> None:
        """Wire CertificateManager for inbound certificate validation."""
        self._certificate_manager = cert_mgr
        # Rebuild handshake processor with certificate validation
        if self._identity is not None:
            self._handshake = HandshakeProcessor(
                identity=self._identity,
                certificate_manager=cert_mgr,
            )
        self._logger.info("certificate_manager_wired_to_federation")

    def set_atune(self, atune: Any) -> None:
        """Wire Atune so federated knowledge becomes perceived input."""
        self._atune = atune
        self._logger.info("atune_wired_to_federation")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo for IIEP hypothesis/procedure collection and ingestion."""
        self._evo = evo
        if self._ingestion:
            self._ingestion._evo = evo
        self._logger.info("evo_wired_to_federation")

    def set_simula(self, simula: Any) -> None:
        """Wire Simula for IIEP mutation pattern collection and ingestion."""
        self._simula = simula
        if self._ingestion:
            self._ingestion._simula = simula
        self._logger.info("simula_wired_to_federation")

    def set_oikos(self, oikos: Any) -> None:
        """Wire Oikos for IIEP economic intelligence collection and ingestion."""
        self._oikos = oikos
        if self._ingestion:
            self._ingestion._oikos = oikos
        self._logger.info("oikos_wired_to_federation")

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Build all sub-systems and load identity keys."""
        if self._initialized:
            return

        # Identity manager is always initialized — the Ed25519 keypair is the
        # instance's permanent identity and is required by CertificateManager
        # regardless of whether peer-networking (federation.enabled) is on.
        self._identity = IdentityManager()

        instance_name = "EOS"
        community_context = ""
        personality_summary = ""
        autonomy_level = 1

        if self._memory:
            try:
                self_node = await self._memory.get_self()
                if self_node:
                    instance_name = self_node.name
                    community_context = getattr(self_node, "community_context", "")
                    autonomy_level = getattr(self_node, "autonomy_level", 1)
            except Exception:
                pass

        private_key_path = (
            Path(self._config.private_key_path) if self._config.private_key_path else None
        )

        trust_policy = TrustPolicy(
            auto_accept_links=self._config.auto_accept_links,
            trust_decay_enabled=self._config.trust_decay_enabled,
            trust_decay_rate_per_day=self._config.trust_decay_rate_per_day,
            max_trust_level=TrustLevel(min(self._config.max_trust_level, 4)),
        )

        tls_cert_identity = Path(self._config.tls_cert_path) if self._config.tls_cert_path else None

        await self._identity.initialize(
            instance_id=self._instance_id,
            instance_name=instance_name,
            community_context=community_context,
            personality_summary=personality_summary,
            autonomy_level=autonomy_level,
            endpoint=self._config.endpoint or "",
            capabilities=["knowledge_exchange", "coordinated_action"],
            trust_policy=trust_policy,
            private_key_path=private_key_path,
            tls_cert_path=tls_cert_identity,
        )

        self._logger.info(
            "identity_initialized",
            instance_id=self._instance_id,
            federation_enabled=self._config.enabled,
        )

        if not self._config.enabled:
            self._logger.info("federation_disabled")
            self._initialized = True
            return

        # Build sub-systems
        trust_max = TrustLevel(min(self._config.max_trust_level, 4))
        self._trust = TrustManager(
            trust_decay_enabled=self._config.trust_decay_enabled,
            trust_decay_rate_per_day=self._config.trust_decay_rate_per_day,
            max_trust_level=trust_max,
        )

        self._privacy = PrivacyFilter()

        # Reputation staking (Phase 16k: Honesty as Schelling Point)
        staking_manager = None
        if self._config.staking_enabled:
            from systems.federation.reputation_staking import (
                ReputationStakingManager,
            )

            staking_manager = ReputationStakingManager(
                wallet=self._wallet,
                redis=self._redis,
                metrics=self._metrics,
                config=self._config.staking,
                escrow_address=self._config.staking.escrow_address,
            )
            await staking_manager.load_bonds()
            self._staking = staking_manager

            self._logger.info(
                "reputation_staking_initialized",
                escrow_address=self._config.staking.escrow_address,
                base_bond=str(self._config.staking.base_bond_usdc),
                max_total=str(self._config.staking.max_total_bonded_usdc),
            )

        self._knowledge = KnowledgeExchangeManager(
            memory=self._memory,
            privacy_filter=self._privacy,
            max_items_per_request=self._config.max_knowledge_items_per_request,
            staking_manager=staking_manager,
        )

        self._coordination = CoordinationManager()

        # Channel manager with TLS configuration
        tls_cert = Path(self._config.tls_cert_path) if self._config.tls_cert_path else None
        tls_key = Path(self._config.tls_key_path) if self._config.tls_key_path else None
        ca_cert = Path(self._config.ca_cert_path) if self._config.ca_cert_path else None

        self._channels = ChannelManager(
            tls_cert_path=tls_cert,
            tls_key_path=tls_key,
            ca_cert_path=ca_cert,
        )

        # Threat intelligence manager (Layer 4: Economic Immune System)
        self._threat_intel = ThreatIntelligenceManager(
            identity=self._identity,
            channels=self._channels,
            instance_id=self._instance_id,
        )

        # IIEP Exchange Protocol + Ingestion Pipeline
        self._exchange = ExchangeProtocol(
            identity=self._identity,
            privacy_filter=self._privacy,
            instance_id=self._instance_id,
        )

        self._ingestion = IngestionPipeline(
            instance_id=self._instance_id,
            equor=self._equor,
            eis=None,  # Wired post-init via set_eis()
            evo=self._evo,
            simula=self._simula,
            oikos=self._oikos,
        )

        # Load persisted links from Redis
        await self._load_links()

        # Handshake processor (certificate_manager may not be wired yet —
        # set_certificate_manager() will rebuild it with cert validation)
        self._handshake = HandshakeProcessor(
            identity=self._identity,
            certificate_manager=self._certificate_manager,
        )

        self._initialized = True
        self._logger.info(
            "federation_initialized",
            instance_id=self._instance_id,
            enabled=True,
            endpoint=self._config.endpoint,
            active_links=len(self._links),
        )

    async def shutdown(self) -> None:
        """Graceful shutdown — close all channels, persist link state."""
        self._logger.info("federation_shutting_down")

        # Persist link state
        await self._persist_links()

        # Close all channels
        if self._channels:
            await self._channels.close_all()

        self._logger.info(
            "federation_shutdown_complete",
            active_links=len(self._links),
            total_interactions=len(self._interaction_history),
        )

    # ─── Link Management ────────────────────────────────────────────

    async def establish_link(
        self, remote_endpoint: str
    ) -> dict[str, Any]:
        """
        Establish a new federation link via the handshake protocol.

        Protocol flow (initiator side):
          1. Pre-flight checks (enabled, capacity, duplicates)
          2. Open temporary channel to remote endpoint
          3. Build and send HandshakeRequest (Phase 1)
          4. Receive and verify HandshakeResponse (Phases 2-3)
          5. Send HandshakeConfirmation (Phase 4 — mutual auth)
          6. Equor constitutional review
          7. Create ACTIVE link

        The handshake ensures both sides verify each other's identity,
        certificate, and constitutional alignment before any link is
        created. Instances with different constitutional hashes cannot
        federate — this is enforced cryptographically.

        Performance target: ≤3000ms
        """
        if not self._config.enabled:
            return {"error": "Federation is disabled"}

        if not self._identity or not self._channels or not self._trust:
            return {"error": "Federation not initialized"}

        if not self._handshake:
            return {"error": "Handshake processor not initialized"}

        start = utc_now()

        # ── Pre-flight: capacity check ──
        active_count = sum(
            1 for lnk in self._links.values()
            if lnk.status == FederationLinkStatus.ACTIVE
        )
        if active_count >= self._config.max_concurrent_links:
            return {"error": f"Maximum concurrent links ({self._config.max_concurrent_links}) reached"}

        # ── Step 1: Open temporary channel ──
        temp_link = FederationLink(
            local_instance_id=self._instance_id,
            remote_instance_id="unknown",
            remote_endpoint=remote_endpoint,
            status=FederationLinkStatus.PENDING,
        )

        try:
            channel = await self._channels.open_channel(temp_link)
        except Exception as exc:
            return {"error": f"Connection failed: {exc}"}

        # ── Step 2: Build and send handshake request (Phase 1) ──
        local_card = self._identity.identity_card

        # Prepare our certificate
        our_cert: dict[str, Any] | None = None
        if self._certificate_manager is not None:
            cert_obj = getattr(self._certificate_manager, "certificate", None)
            if cert_obj is not None:
                our_cert = cert_obj.model_dump(mode="json")

        handshake_req = HandshakeRequest(
            initiator_instance_id=local_card.instance_id,
            initiator_name=local_card.name,
            initiator_endpoint=local_card.endpoint,
            identity_card=local_card.model_dump(mode="json"),
            certificate=our_cert,
            constitutional_hash=local_card.constitutional_hash,
            capabilities=local_card.capabilities,
            max_knowledge_items_per_request=self._config.max_knowledge_items_per_request,
        )

        response_data = await channel.send_handshake(
            handshake_req.model_dump(mode="json"),
        )

        if response_data is None:
            await self._channels.close_channel(temp_link.id)
            return {"error": "Handshake failed — no response from remote instance (network timeout or refusal)"}

        # ── Step 3: Parse and verify response (Phase 4 — initiator side) ──
        try:
            handshake_resp = HandshakeResponse.model_validate(response_data)
        except Exception as exc:
            await self._channels.close_channel(temp_link.id)
            return {"error": f"Invalid handshake response: {exc}"}

        result = self._handshake.verify_response(handshake_req, handshake_resp)
        if not result.success:
            await self._channels.close_channel(temp_link.id)

            # Record failed handshake as interaction for audit
            if self._metrics:
                await self._metrics.record("federation", "handshake.failed", 1.0)

            return {"error": f"Handshake verification failed: {result.error}"}

        # ── Step 4: Send confirmation (Phase 4 — mutual auth completion) ──
        confirmation = self._handshake.build_confirmation(
            handshake_id=handshake_resp.handshake_id,
            responder_nonce=handshake_resp.responder_nonce,
        )
        confirm_sent = await channel.send_handshake_confirmation(
            confirmation.model_dump(mode="json"),
        )
        if not confirm_sent:
            await self._channels.close_channel(temp_link.id)
            self._logger.warning(
                "handshake_confirmation_failed",
                handshake_id=handshake_resp.handshake_id,
            )
            return {"error": "Handshake confirmation delivery failed — remote side may have closed connection"}

        # ── Step 5: Check for duplicate link (post-handshake) ──
        remote_id = handshake_resp.responder_instance_id
        for existing in self._links.values():
            if (
                existing.remote_instance_id == remote_id
                and existing.status == FederationLinkStatus.ACTIVE
            ):
                await self._channels.close_channel(temp_link.id)
                return {
                    "error": "Already linked to this instance",
                    "existing_link_id": existing.id,
                }

        # ── Step 6: Equor constitutional review ──
        equor_permitted = True
        if self._equor:
            try:
                from primitives.common import Verdict
                from primitives.intent import (
                    ActionSequence,
                    DecisionTrace,
                    GoalDescriptor,
                    Intent,
                )

                community_ctx = ""
                if result.remote_identity_card:
                    community_ctx = result.remote_identity_card.get("community_context", "")[:100]

                intent = Intent(
                    goal=GoalDescriptor(
                        description=f"Establish federation link with {handshake_resp.responder_name}",
                        target_domain="federation",
                    ),
                    plan=ActionSequence(steps=[]),
                    decision_trace=DecisionTrace(
                        reasoning=f"Handshake completed with {handshake_resp.responder_name} "
                                  f"({community_ctx})",
                    ),
                )
                check = await self._equor.review(intent)
                equor_permitted = check.verdict in (Verdict.APPROVED, Verdict.MODIFIED)
            except Exception as exc:
                self._logger.warning("equor_review_failed", error=str(exc))

        if not equor_permitted:
            await self._channels.close_channel(temp_link.id)
            return {"error": "Constitutional review denied this federation link"}

        # ── Step 7: Create the official link ──
        remote_identity = InstanceIdentityCard(
            **result.remote_identity_card
        ) if result.remote_identity_card else None

        link = FederationLink(
            local_instance_id=self._instance_id,
            remote_instance_id=remote_id,
            remote_name=handshake_resp.responder_name,
            remote_endpoint=remote_endpoint,
            trust_level=TrustLevel.NONE,
            trust_score=0.0,
            status=FederationLinkStatus.ACTIVE,
            remote_identity=remote_identity,
        )

        # Re-open channel with the real link
        await self._channels.close_channel(temp_link.id)
        await self._channels.open_channel(link)

        self._links[link.id] = link
        await self._persist_links()

        elapsed_ms = int((utc_now() - start).total_seconds() * 1000)

        # Record interaction
        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type="link_establishment",
            direction="outbound",
            outcome=InteractionOutcome.SUCCESSFUL,
            description=(
                f"Handshake completed with {handshake_resp.responder_name} "
                f"(id={handshake_resp.handshake_id[:12]}...)"
            ),
            trust_value=1.0,
            latency_ms=elapsed_ms,
            metadata={
                "handshake_id": handshake_resp.handshake_id,
                "negotiated_params": result.negotiated_params,
            },
        )
        self._record_interaction(interaction)

        # Update trust for the successful establishment
        self._trust.update_trust(link, interaction)

        # Record metric
        if self._metrics:
            await self._metrics.record("federation", "links.established", 1.0)
            await self._metrics.record("federation", "handshake.completed", 1.0)

        self._logger.info(
            "link_established_via_handshake",
            link_id=link.id,
            handshake_id=handshake_resp.handshake_id,
            remote_id=remote_id,
            remote_name=handshake_resp.responder_name,
            elapsed_ms=elapsed_ms,
            negotiated_params=result.negotiated_params,
        )

        return {
            "link_id": link.id,
            "remote_instance_id": remote_id,
            "remote_name": handshake_resp.responder_name,
            "trust_level": link.trust_level.name,
            "status": link.status.value,
            "elapsed_ms": elapsed_ms,
            "handshake_id": handshake_resp.handshake_id,
            "negotiated_params": result.negotiated_params,
        }

    async def handle_handshake(
        self, body: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle an inbound handshake request (responder side, Phases 2-3).

        Called when a remote instance initiates a handshake via POST
        /api/v1/federation/handshake. This is the responder path.

        Returns a serialized HandshakeResponse (accepted or rejected).
        """
        if not self._config.enabled:
            return HandshakeResponse(
                handshake_id=body.get("handshake_id", ""),
                accepted=False,
                reject_reason="Federation is disabled on this instance",
            ).model_dump(mode="json")

        if not self._identity or not self._handshake:
            return HandshakeResponse(
                handshake_id=body.get("handshake_id", ""),
                accepted=False,
                reject_reason="Federation not initialized",
            ).model_dump(mode="json")

        # Parse the inbound request
        try:
            request = HandshakeRequest.model_validate(body)
        except Exception as exc:
            return HandshakeResponse(
                handshake_id=body.get("handshake_id", ""),
                accepted=False,
                reject_reason=f"Malformed handshake request: {exc}",
            ).model_dump(mode="json")

        # Check max links before accepting
        active_count = sum(
            1 for lnk in self._links.values()
            if lnk.status == FederationLinkStatus.ACTIVE
        )
        if active_count >= self._config.max_concurrent_links:
            return HandshakeResponse(
                handshake_id=request.handshake_id,
                accepted=False,
                reject_reason=f"Maximum concurrent links ({self._config.max_concurrent_links}) reached",
            ).model_dump(mode="json")

        # Check for existing link to this instance
        for existing in self._links.values():
            if (
                existing.remote_instance_id == request.initiator_instance_id
                and existing.status == FederationLinkStatus.ACTIVE
            ):
                return HandshakeResponse(
                    handshake_id=request.handshake_id,
                    accepted=False,
                    reject_reason="Already linked to this instance",
                ).model_dump(mode="json")

        # Equor constitutional review (responder side)
        equor_permitted = True
        if self._equor:
            try:
                from primitives.common import Verdict
                from primitives.intent import (
                    ActionSequence,
                    DecisionTrace,
                    GoalDescriptor,
                    Intent,
                )

                intent = Intent(
                    goal=GoalDescriptor(
                        description=f"Accept federation handshake from {request.initiator_name}",
                        target_domain="federation",
                    ),
                    plan=ActionSequence(steps=[]),
                    decision_trace=DecisionTrace(
                        reasoning=f"Inbound handshake from {request.initiator_name} "
                                  f"(instance {request.initiator_instance_id[:16]}...)",
                    ),
                )
                check = await self._equor.review(intent)
                equor_permitted = check.verdict in (Verdict.APPROVED, Verdict.MODIFIED)
            except Exception as exc:
                self._logger.warning("equor_review_failed_handshake", error=str(exc))

        if not equor_permitted:
            return HandshakeResponse(
                handshake_id=request.handshake_id,
                accepted=False,
                reject_reason="Constitutional review denied this federation handshake",
            ).model_dump(mode="json")

        # Process the handshake (identity, certificate, constitutional checks)
        response = self._handshake.process_inbound(request)

        if response.accepted:
            # Store handshake state so we can verify the confirmation
            # in Phase 4 and create the link. Cleared after confirmation
            # or after a timeout.
            initiator_pubkey = request.identity_card.get("public_key_pem", "")
            self._pending_handshakes[request.handshake_id] = (
                response.responder_nonce,
                initiator_pubkey,
                request.initiator_instance_id,
                request.initiator_name,
                request.initiator_endpoint,
            )

            if self._metrics:
                await self._metrics.record("federation", "handshake.accepted", 1.0)
        else:
            if self._metrics:
                await self._metrics.record("federation", "handshake.rejected", 1.0)

        self._logger.info(
            "handshake_processed",
            handshake_id=request.handshake_id,
            remote_id=request.initiator_instance_id,
            accepted=response.accepted,
            reject_reason=response.reject_reason or None,
        )

        return response.model_dump(mode="json")

    async def handle_handshake_confirmation(
        self, body: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle a handshake confirmation (responder side, Phase 4).

        The initiator sends this after verifying our response. It contains
        the initiator's signature over our nonce, completing mutual
        authentication.

        On success, the responder creates its side of the federation link.
        """
        if not self._identity or not self._handshake:
            return {"error": "Federation not initialized", "accepted": False}

        try:
            confirmation = HandshakeConfirmation.model_validate(body)
        except Exception as exc:
            return {"error": f"Malformed confirmation: {exc}", "accepted": False}

        # Retrieve the pending handshake state
        pending = self._pending_handshakes.pop(confirmation.handshake_id, None)
        if pending is None:
            return {
                "error": "Unknown or expired handshake — no pending handshake with this ID",
                "accepted": False,
            }

        expected_nonce, initiator_pubkey, initiator_id, initiator_name, initiator_endpoint = pending

        # Verify the initiator signed our nonce correctly
        if not initiator_pubkey:
            self._logger.warning(
                "handshake_confirmation_no_pubkey",
                handshake_id=confirmation.handshake_id,
            )
            return {"error": "No initiator public key cached from handshake", "accepted": False}

        sig_valid = self._handshake.verify_confirmation(
            confirmation=confirmation,
            expected_nonce=expected_nonce,
            initiator_public_key_pem=initiator_pubkey,
        )

        if not sig_valid:
            self._logger.warning(
                "handshake_confirmation_sig_invalid",
                handshake_id=confirmation.handshake_id,
                initiator_id=initiator_id,
            )
            if self._metrics:
                await self._metrics.record("federation", "handshake.confirmation_failed", 1.0)
            return {
                "error": "Nonce signature verification failed — initiator cannot prove identity",
                "accepted": False,
            }

        # Mutual authentication complete. Create the responder-side link.
        # The responder doesn't know the initiator's endpoint from the
        # handshake alone (the initiator provided it in the request),
        # so we use it to set up the reverse channel.
        link = FederationLink(
            local_instance_id=self._instance_id,
            remote_instance_id=initiator_id,
            remote_name=initiator_name,
            remote_endpoint=initiator_endpoint,
            trust_level=TrustLevel.NONE,
            trust_score=0.0,
            status=FederationLinkStatus.ACTIVE,
        )

        self._links[link.id] = link
        await self._persist_links()

        # Open a reverse channel to the initiator
        if self._channels and initiator_endpoint:
            try:
                await self._channels.open_channel(link)
            except Exception as exc:
                self._logger.warning(
                    "handshake_reverse_channel_failed",
                    error=str(exc),
                    initiator_endpoint=initiator_endpoint,
                )

        # Record interaction
        if self._trust:
            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=initiator_id,
                interaction_type="link_establishment",
                direction="inbound",
                outcome=InteractionOutcome.SUCCESSFUL,
                description=(
                    f"Handshake accepted from {initiator_name} "
                    f"(id={confirmation.handshake_id[:12]}...)"
                ),
                trust_value=1.0,
                metadata={"handshake_id": confirmation.handshake_id},
            )
            self._record_interaction(interaction)
            self._trust.update_trust(link, interaction)

        if self._metrics:
            await self._metrics.record("federation", "handshake.confirmed", 1.0)
            await self._metrics.record("federation", "links.established", 1.0)

        self._logger.info(
            "handshake_confirmed_link_created",
            handshake_id=confirmation.handshake_id,
            link_id=link.id,
            initiator_id=initiator_id,
            initiator_name=initiator_name,
        )

        return {
            "accepted": True,
            "handshake_id": confirmation.handshake_id,
            "link_id": link.id,
        }

    async def withdraw_link(self, link_id: str) -> dict[str, Any]:
        """
        Withdraw from a federation link.

        Withdrawal is always free — any instance can disconnect at any
        time with no penalty. This ensures federation is always
        voluntary, never coerced.
        """
        link = self._links.get(link_id)
        if not link:
            return {"error": "Link not found"}

        link.status = FederationLinkStatus.WITHDRAWN

        # Close the channel
        if self._channels:
            await self._channels.close_channel(link_id)

        await self._persist_links()

        if self._metrics:
            await self._metrics.record("federation", "links.dropped", 1.0)

        self._logger.info(
            "link_withdrawn",
            link_id=link_id,
            remote_id=link.remote_instance_id,
        )

        return {
            "link_id": link_id,
            "status": "withdrawn",
            "remote_instance_id": link.remote_instance_id,
        }

    # ─── Knowledge Exchange ─────────────────────────────────────────

    async def handle_knowledge_request(
        self,
        request: KnowledgeRequest,
        sender_certificate: Any = None,
    ) -> KnowledgeResponse:
        """
        Handle an inbound knowledge request from a remote instance.

        Phase 16g: If a CertificateManager is wired, the sender's
        EcodianCertificate is validated. Requests with invalid or missing
        certificates are rejected.

        This is called by the API router when a federated instance
        sends a knowledge request to us.
        """
        if not self._knowledge or not self._trust:
            return KnowledgeResponse(
                request_id=request.id,
                granted=False,
                reason="Federation not initialized",
            )

        # Phase 16g: Certificate validation gate
        cert_rejection = self._validate_sender_certificate(
            request.requesting_instance_id, sender_certificate,
        )
        if cert_rejection is not None:
            return KnowledgeResponse(
                request_id=request.id,
                granted=False,
                reason=cert_rejection,
            )

        # Find the link for this requesting instance
        link = self._find_link_by_instance(request.requesting_instance_id)
        if not link:
            return KnowledgeResponse(
                request_id=request.id,
                granted=False,
                reason="No active federation link with this instance",
            )

        # Equor review
        equor_permitted = True
        if self._equor:
            try:
                from primitives.intent import (
                    ActionSequence,
                    DecisionTrace,
                    GoalDescriptor,
                    Intent,
                )
                intent = Intent(
                    goal=GoalDescriptor(
                        description=f"Share {request.knowledge_type.value} with {link.remote_name}",
                        target_domain="federation.knowledge",
                    ),
                    plan=ActionSequence(steps=[]),
                    decision_trace=DecisionTrace(
                        reasoning=f"Knowledge request from {link.remote_name}: {request.query[:100]}",
                    ),
                )
                check = await self._equor.review(intent)
                from primitives.common import Verdict
                equor_permitted = check.verdict in (Verdict.APPROVED, Verdict.MODIFIED)
            except Exception:
                pass

        response, interaction = await self._knowledge.handle_request(
            request=request,
            link=link,
            equor_permitted=equor_permitted,
        )

        # Update trust
        self._trust.update_trust(link, interaction)
        self._record_interaction(interaction)

        # Metrics
        if self._metrics:
            metric_name = "knowledge.shared" if response.granted else "knowledge.denied"
            await self._metrics.record("federation", metric_name, 1.0)
            if response.granted:
                await self._metrics.record(
                    "federation", "privacy.items_filtered",
                    float(len(response.knowledge)),
                )

        return response

    async def request_knowledge(
        self,
        link_id: str,
        knowledge_type: KnowledgeType,
        query: str = "",
        max_results: int = 10,
    ) -> KnowledgeResponse | None:
        """
        Request knowledge from a remote federated instance.
        """
        if not self._knowledge or not self._channels or not self._identity:
            return None

        link = self._links.get(link_id)
        if not link or link.status != FederationLinkStatus.ACTIVE:
            return None

        channel = self._channels.get_channel(link_id)
        if not channel:
            return None

        request = self._knowledge.build_request(
            knowledge_type=knowledge_type,
            query=query,
            max_results=max_results,
            local_instance_id=self._instance_id,
        )

        response = await channel.request_knowledge(request)

        if response:
            interaction = await self._knowledge.ingest_response(response, link)
            if self._trust:
                self._trust.update_trust(link, interaction)
            self._record_interaction(interaction)

            # Feed received knowledge to Atune as a perceived input
            if response.granted and response.knowledge and self._atune is not None:
                await self._inject_federated_percept(response.knowledge, link)

        return response

    # ─── Coordinated Action ─────────────────────────────────────────

    async def handle_assistance_request(
        self,
        request: AssistanceRequest,
    ) -> AssistanceResponse:
        """
        Handle an inbound assistance request from a remote instance.
        """
        if not self._coordination or not self._trust:
            return AssistanceResponse(
                request_id=request.id,
                accepted=False,
                reason="Federation not initialized",
            )

        link = self._find_link_by_instance(request.requesting_instance_id)
        if not link:
            return AssistanceResponse(
                request_id=request.id,
                accepted=False,
                reason="No active federation link with this instance",
            )

        # Equor review
        equor_permitted = True
        if self._equor:
            try:
                from primitives.intent import (
                    ActionSequence,
                    DecisionTrace,
                    GoalDescriptor,
                    Intent,
                )
                intent = Intent(
                    goal=GoalDescriptor(
                        description=f"Assist {link.remote_name}: {request.description[:100]}",
                        target_domain="federation.assistance",
                    ),
                    plan=ActionSequence(steps=[]),
                    decision_trace=DecisionTrace(
                        reasoning=f"Assistance request from {link.remote_name}",
                    ),
                )
                check = await self._equor.review(intent)
                from primitives.common import Verdict
                equor_permitted = check.verdict in (Verdict.APPROVED, Verdict.MODIFIED)
            except Exception:
                pass

        response, interaction = await self._coordination.handle_request(
            request=request,
            link=link,
            equor_permitted=equor_permitted,
        )

        self._trust.update_trust(link, interaction)
        self._record_interaction(interaction)

        if self._metrics:
            metric = "assistance.accepted" if response.accepted else "assistance.requested"
            await self._metrics.record("federation", metric, 1.0)

        return response

    async def request_assistance(
        self,
        link_id: str,
        description: str,
        knowledge_domain: str = "",
        urgency: float = 0.5,
    ) -> AssistanceResponse | None:
        """
        Request assistance from a remote federated instance.
        """
        if not self._coordination or not self._channels or not self._identity:
            return None

        link = self._links.get(link_id)
        if not link or link.status != FederationLinkStatus.ACTIVE:
            return None

        channel = self._channels.get_channel(link_id)
        if not channel:
            return None

        request = self._coordination.build_request(
            description=description,
            knowledge_domain=knowledge_domain,
            urgency=urgency,
            local_instance_id=self._instance_id,
        )

        return await channel.request_assistance(request)

    # ─── Threat Intelligence ────────────────────────────────────

    async def broadcast_threat_advisory(
        self,
        advisory: ThreatAdvisory,
    ) -> dict[str, bool]:
        """
        Broadcast a threat advisory to all active federation links.

        Signs the advisory and sends to peers whose trust level permits it.
        Returns {link_id: delivered}.
        """
        if not self._threat_intel:
            return {}
        return await self._threat_intel.broadcast_advisory(
            advisory, self.active_links
        )

    def handle_threat_advisory(
        self,
        advisory: ThreatAdvisory,
        source_instance_id: str,
        sender_certificate: Any = None,
    ) -> tuple[bool, str]:
        """
        Handle an inbound threat advisory from a remote instance.

        Phase 16g: If a CertificateManager is wired, the sender's
        EcodianCertificate is validated first. Missing or invalid
        certificates cause rejection.

        Trust-gated: PARTNER/ALLY auto-apply, COLLEAGUE recommend verify,
        ACQUAINTANCE verify signature, NONE reject.

        Returns (accepted, reason).
        """
        if not self._threat_intel:
            return False, "Federation threat intelligence not initialized"

        link = self._find_link_by_instance(source_instance_id)
        if not link:
            return False, "No active federation link with this instance"

        # Phase 16g: Certificate validation gate
        cert_rejection = self._validate_sender_certificate(
            source_instance_id, sender_certificate,
        )
        if cert_rejection is not None:
            return False, cert_rejection

        return self._threat_intel.handle_inbound_advisory(advisory, link)

    # ─── IIEP Knowledge Exchange ─────────────────────────────────

    async def push_knowledge(
        self,
        link_id: str,
        kinds: list[ExchangePayloadKind] | None = None,
    ) -> ExchangeReceipt | None:
        """
        Proactively push eligible knowledge to a federated peer.

        Collects knowledge from Evo, Simula, and Oikos, applies
        confidence and trust filtering, signs an envelope, and sends
        it over the channel.  Returns the peer's receipt.
        """
        if not self._exchange or not self._channels:
            return None

        link = self._links.get(link_id)
        if not link or link.status != FederationLinkStatus.ACTIVE:
            return None

        channel = self._channels.get_channel(link_id)
        if not channel:
            return None

        # Collect payloads from all requested kinds
        if kinds is None:
            kinds = list(ExchangePayloadKind)

        from primitives.federation import EXCHANGE_CONFIDENCE_FLOORS

        all_payloads = []
        for kind in kinds:
            floor = EXCHANGE_CONFIDENCE_FLOORS.get(kind, 0.5)
            if kind == ExchangePayloadKind.HYPOTHESIS and self._evo:
                all_payloads.extend(await collect_hypotheses(
                    self._evo, self._instance_id, min_confidence=floor,
                ))
            elif kind == ExchangePayloadKind.PROCEDURE and self._evo:
                all_payloads.extend(await collect_procedures(
                    self._evo, self._instance_id, min_confidence=floor,
                ))
            elif kind == ExchangePayloadKind.MUTATION_PATTERN and self._simula:
                all_payloads.extend(await collect_mutation_patterns(
                    self._simula, self._instance_id, min_confidence=floor,
                ))
            elif kind == ExchangePayloadKind.ECONOMIC_INTEL and self._oikos:
                all_payloads.extend(await collect_economic_intel(
                    self._oikos, self._instance_id, min_confidence=floor,
                ))

        envelope = await self._exchange.prepare_push(link, all_payloads)
        if envelope is None:
            return None

        # Send via channel
        receipt = await channel.send_exchange(envelope)

        # Record interaction
        outcome = InteractionOutcome.SUCCESSFUL if receipt else InteractionOutcome.FAILED
        interaction = ExchangeProtocol.build_exchange_interaction(
            link=link,
            direction="outbound",
            outcome=outcome,
            payload_count=len(envelope.payloads),
            description=f"IIEP push: {len(envelope.payloads)} payloads",
        )
        if self._trust:
            self._trust.update_trust(link, interaction)
        self._record_interaction(interaction)

        if self._metrics:
            await self._metrics.record(
                "federation", "iiep.push.payloads",
                float(len(envelope.payloads)),
            )

        return receipt

    async def pull_knowledge(
        self,
        link_id: str,
        kinds: list[ExchangePayloadKind],
        query: str = "",
        max_items: int = 20,
    ) -> ExchangeReceipt | None:
        """
        Request specific knowledge from a federated peer via IIEP PULL.

        Sends a pull request, receives a push response, and processes
        the response through the ingestion pipeline.
        """
        if not self._exchange or not self._channels or not self._ingestion:
            return None

        link = self._links.get(link_id)
        if not link or link.status != FederationLinkStatus.ACTIVE:
            return None

        channel = self._channels.get_channel(link_id)
        if not channel:
            return None

        pull_envelope = self._exchange.prepare_pull_request(
            link, kinds, query=query, max_items=max_items,
        )
        if pull_envelope is None:
            return None

        # The pull request is sent as an exchange; the remote responds
        # with a push envelope (or an empty receipt if nothing to share)
        receipt = await channel.send_exchange(pull_envelope)

        if receipt:
            interaction = ExchangeProtocol.build_exchange_interaction(
                link=link,
                direction="outbound",
                outcome=InteractionOutcome.SUCCESSFUL,
                payload_count=0,
                description="IIEP pull request sent",
            )
            if self._trust:
                self._trust.update_trust(link, interaction)
            self._record_interaction(interaction)

        return receipt

    async def handle_exchange(
        self,
        envelope: ExchangeEnvelope,
        sender_certificate: Any = None,
    ) -> ExchangeReceipt | ExchangeEnvelope | None:
        """
        Handle an inbound IIEP exchange envelope.

        For PUSH envelopes: processes payloads through the ingestion
        pipeline and returns a receipt.

        For PULL envelopes: collects eligible knowledge and returns
        a response push envelope.

        Called by the API router.
        """
        if not self._exchange or not self._ingestion:
            return ExchangeReceipt(
                envelope_id=envelope.id,
                receiver_instance_id=self._instance_id,
                payload_verdicts={},
            )

        # Certificate validation
        cert_rejection = self._validate_sender_certificate(
            envelope.sender_instance_id, sender_certificate,
        )
        if cert_rejection is not None:
            self._logger.warning(
                "exchange_cert_rejected",
                sender=envelope.sender_instance_id,
                reason=cert_rejection,
            )
            return ExchangeReceipt(
                envelope_id=envelope.id,
                receiver_instance_id=self._instance_id,
                payload_verdicts={
                    p.payload_id: IngestionVerdict.REJECTED
                    for p in envelope.payloads
                },
            )

        # Find the link
        link = self._find_link_by_instance(envelope.sender_instance_id)
        if not link:
            return ExchangeReceipt(
                envelope_id=envelope.id,
                receiver_instance_id=self._instance_id,
                payload_verdicts={
                    p.payload_id: IngestionVerdict.REJECTED
                    for p in envelope.payloads
                },
            )

        # Verify envelope signature
        valid, reason = self._exchange.verify_envelope(envelope, link)
        if not valid:
            self._logger.warning(
                "exchange_verification_failed",
                sender=envelope.sender_instance_id,
                reason=reason,
            )
            return ExchangeReceipt(
                envelope_id=envelope.id,
                receiver_instance_id=self._instance_id,
                payload_verdicts={
                    p.payload_id: IngestionVerdict.REJECTED
                    for p in envelope.payloads
                },
            )

        if envelope.direction == ExchangeDirection.PUSH:
            # Process inbound payloads through ingestion pipeline
            receipt = await self._ingestion.process_envelope(envelope, link)

            # Record interaction
            accepted = sum(
                1 for v in receipt.payload_verdicts.values()
                if v == IngestionVerdict.ACCEPTED
            )
            total = len(receipt.payload_verdicts)
            outcome = (
                InteractionOutcome.SUCCESSFUL if accepted > 0
                else InteractionOutcome.FAILED
            )
            interaction = ExchangeProtocol.build_exchange_interaction(
                link=link,
                direction="inbound",
                outcome=outcome,
                payload_count=total,
                description=(
                    f"IIEP push received: {accepted}/{total} accepted"
                ),
            )
            if self._trust:
                self._trust.update_trust(link, interaction)
            self._record_interaction(interaction)

            if self._metrics:
                await self._metrics.record(
                    "federation", "iiep.received.payloads", float(total),
                )
                await self._metrics.record(
                    "federation", "iiep.received.accepted", float(accepted),
                )

            return receipt

        elif envelope.direction == ExchangeDirection.PULL:
            # Respond to pull request with a push envelope
            response_envelope = await self._exchange.handle_pull_request(
                envelope, link,
                evo=self._evo,
                simula=self._simula,
                oikos=self._oikos,
            )

            if response_envelope and self._metrics:
                await self._metrics.record(
                    "federation", "iiep.pull_response.payloads",
                    float(len(response_envelope.payloads)),
                )

            return response_envelope

        return None

    def set_eis(self, eis: Any) -> None:
        """Wire EIS for IIEP ingestion taint analysis."""
        if self._ingestion:
            self._ingestion._eis = eis
        self._logger.info("eis_wired_to_federation")

    # ─── Identity ───────────────────────────────────────────────────

    @property
    def identity_card(self) -> InstanceIdentityCard | None:
        """This instance's public identity card."""
        if self._identity:
            return self._identity.identity_card
        return None

    # ─── Link Queries ───────────────────────────────────────────────

    @property
    def active_links(self) -> list[FederationLink]:
        """All active federation links."""
        return [
            lnk for lnk in self._links.values()
            if lnk.status == FederationLinkStatus.ACTIVE
        ]

    def get_link(self, link_id: str) -> FederationLink | None:
        return self._links.get(link_id)

    def get_link_by_instance(self, instance_id: str) -> FederationLink | None:
        return self._find_link_by_instance(instance_id)

    # ─── Trust Decay (called periodically) ──────────────────────────

    async def apply_trust_decay(self) -> None:
        """Apply trust decay to all active links and recover expired bonds."""
        if not self._trust:
            return
        for link in self.active_links:
            self._trust.apply_decay(link)
        # Recover expired reputation bonds
        if self._staking:
            await self._staking.recover_expired_bonds()

    # ─── Health ─────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Self-health report (implements ManagedSystem protocol)."""
        if not self._config.enabled:
            return {
                "status": "disabled",
                "enabled": False,
                "active_links": 0,
                "mean_trust": 0.0,
                "total_interactions": 0,
            }

        active = self.active_links
        return {
            "status": "healthy" if self._initialized else "starting",
            "enabled": True,
            "active_links": len(active),
            "mean_trust": self._trust.mean_trust(active) if self._trust and active else 0.0,
            "total_interactions": len(self._interaction_history),
        }

    @property
    def stats(self) -> dict[str, Any]:
        active = self.active_links
        return {
            "initialized": self._initialized,
            "enabled": self._config.enabled,
            "instance_id": self._instance_id,
            "active_links": len(active),
            "total_links": len(self._links),
            "mean_trust": round(
                self._trust.mean_trust(active) if self._trust and active else 0.0, 2
            ),
            "identity": self._identity.stats if self._identity else {},
            "trust": self._trust.stats if self._trust else {},
            "knowledge": self._knowledge.stats if self._knowledge else {},
            "coordination": self._coordination.stats if self._coordination else {},
            "channels": self._channels.stats if self._channels else {},
            "threat_intel": self._threat_intel.stats if self._threat_intel else {},
            "staking": self._staking.stats if self._staking else {},
            "exchange": self._exchange.stats if self._exchange else {},
            "ingestion": self._ingestion.stats if self._ingestion else {},
            "privacy": self._privacy.stats if self._privacy else {},
            "interaction_history_size": len(self._interaction_history),
            "links": [
                {
                    "id": lnk.id,
                    "remote_id": lnk.remote_instance_id,
                    "remote_name": lnk.remote_name,
                    "trust_level": lnk.trust_level.name,
                    "trust_score": round(lnk.trust_score, 2),
                    "status": lnk.status.value,
                    "shared_count": lnk.shared_knowledge_count,
                    "received_count": lnk.received_knowledge_count,
                    "successful": lnk.successful_interactions,
                    "failed": lnk.failed_interactions,
                    "violations": lnk.violation_count,
                }
                for lnk in self._links.values()
            ],
        }

    # ─── Phase 16g: Certificate Validation Gate ────────────────────

    def _validate_sender_certificate(
        self,
        source_instance_id: str,
        sender_certificate: Any,
    ) -> str | None:
        """
        Validate the sender's EcodianCertificate. Returns an error string
        if the certificate is invalid/missing, or None if valid.

        When no CertificateManager is wired, certificate validation is
        skipped (backward-compatible with pre-16g instances).
        """
        if self._certificate_manager is None:
            return None  # No certificate validation configured

        if sender_certificate is None:
            self._logger.warning(
                "inbound_rejected_no_certificate",
                source_instance_id=source_instance_id,
            )
            return "Sender has no EcodianCertificate -- rejected"

        from systems.identity.certificate import EcodianCertificate

        # Accept both EcodianCertificate objects and raw dicts
        if isinstance(sender_certificate, dict):
            try:
                sender_certificate = EcodianCertificate.model_validate(sender_certificate)
            except Exception as exc:
                return f"Invalid certificate format: {exc}"

        if not isinstance(sender_certificate, EcodianCertificate):
            return "Invalid certificate type"

        result = self._certificate_manager.validate_certificate(sender_certificate)
        if not result.valid:
            self._logger.warning(
                "inbound_rejected_invalid_certificate",
                source_instance_id=source_instance_id,
                errors=result.errors,
            )
            return f"Invalid certificate: {'; '.join(result.errors)}"

        return None

    # ─── Internal ───────────────────────────────────────────────────

    def _find_link_by_instance(self, instance_id: str) -> FederationLink | None:
        """Find an active link for a given remote instance ID."""
        for link in self._links.values():
            if (
                link.remote_instance_id == instance_id
                and link.status == FederationLinkStatus.ACTIVE
            ):
                return link
        return None

    def _record_interaction(self, interaction: FederationInteraction) -> None:
        """Record an interaction in the history ring buffer."""
        self._interaction_history.append(interaction)
        if len(self._interaction_history) > self._max_history:
            self._interaction_history = self._interaction_history[-self._max_history:]

    async def _inject_federated_percept(self, knowledge_items: list[Any], link: Any) -> None:
        """
        Inject federated knowledge into Atune as perceived input.

        The organism should perceive knowledge from its peers — otherwise
        federation data stops at Memory and never enters the cognitive cycle.
        """
        try:
            from systems.atune.types import InputChannel, RawInput

            summaries: list[str] = []
            for item in knowledge_items[:3]:  # Cap at 3 to avoid flooding
                content = getattr(item, "content", "") or getattr(item, "summary", "")
                if content:
                    summaries.append(str(content)[:200])
            if not summaries:
                return

            raw = RawInput(
                data=(
                    f"Knowledge from {getattr(link, 'remote_name', 'peer')}: "
                    f"{'; '.join(summaries)}"
                ),
                channel_id=f"federation:{getattr(link, 'id', 'unknown')}",
                metadata={"source_instance": getattr(link, "remote_instance_id", "")},
            )
            await self._atune.ingest(raw, InputChannel.FEDERATION_MSG)
            self._logger.debug("federated_knowledge_injected_to_atune")
        except Exception:
            self._logger.debug("federation_atune_ingest_failed", exc_info=True)

    async def _persist_links(self) -> None:
        """Persist link state to Redis (primary) with local file fallback."""
        # Always write local backup regardless of Redis availability
        self._persist_links_to_file()

        if not self._redis:
            return
        try:
            links_data = {
                link_id: link.model_dump_json()
                for link_id, link in self._links.items()
            }
            for link_id, data in links_data.items():
                await self._redis.set_json(
                    f"fed:links:{link_id}",
                    data,
                )
            # Store the link ID index
            await self._redis.set_json(
                "fed:link_ids",
                list(self._links.keys()),
            )
        except Exception as exc:
            self._logger.warning(
                "link_persist_redis_failed_local_backup_written",
                error=str(exc),
            )

    async def _load_links(self) -> None:
        """Load persisted links: try Redis first, fall back to local file."""
        loaded = False

        if self._redis:
            try:
                link_ids_raw = await self._redis.get_json("fed:link_ids")
                if link_ids_raw:
                    link_ids: list[str] = list(link_ids_raw) if link_ids_raw else []
                    for link_id in link_ids:
                        if not link_id:
                            continue
                        data = await self._redis.get_json(f"fed:links:{link_id}")
                        if data:
                            link = FederationLink.model_validate_json(str(data))
                            self._links[link.id] = link
                    loaded = bool(self._links)
            except Exception as exc:
                self._logger.warning("link_load_redis_failed", error=str(exc))

        # Fall back to local file if Redis was empty or unavailable
        if not loaded:
            self._load_links_from_file()

        self._logger.info("links_loaded", count=len(self._links), source="redis" if loaded else "file")

    def _persist_links_to_file(self) -> None:
        """Write link state to a local JSON file as a backup."""
        try:
            backup_path = Path(self._config.data_dir or ".") / "federation_links.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                link_id: link.model_dump_json()
                for link_id, link in self._links.items()
            }
            backup_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            self._logger.debug("link_file_backup_failed", error=str(exc))

    def _load_links_from_file(self) -> None:
        """Restore link state from local JSON backup file."""
        try:
            backup_path = Path(self._config.data_dir or ".") / "federation_links.json"
            if not backup_path.exists():
                return
            raw = json.loads(backup_path.read_text(encoding="utf-8"))
            for _link_id, link_json in raw.items():
                link = FederationLink.model_validate_json(link_json)
                self._links[link.id] = link
        except Exception as exc:
            self._logger.debug("link_file_restore_failed", error=str(exc))
