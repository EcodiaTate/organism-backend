"""
EcodiaOS — Federation Handshake Protocol

The handshake is the prerequisite for all multi-instance communication:
shared learning, economic cooperation, mitosis. Two instances must complete
a full handshake before any federation link becomes ACTIVE.

Protocol (4 phases, single HTTP round-trip):

  Phase 1 — HELLO (initiator → responder)
    Initiator sends its identity card + certificate + nonce.

  Phase 2 — CHALLENGE (responder processes)
    Responder verifies:
      a. Certificate is valid (signature, expiry, required fields)
      b. Constitutional hash matches (instances with different invariants
         cannot federate — this is the alignment gate)
      c. Protocol version is compatible
      d. Identity card is well-formed
    If all pass, responder signs the initiator's nonce to prove possession
    of its own private key.

  Phase 3 — ACCEPT (responder → initiator)
    Responder returns its own identity card + certificate + nonce + signed
    challenge. The initiator then verifies the responder symmetrically.

  Phase 4 — CONFIRM (initiator verifies response)
    Initiator verifies the responder's certificate, constitutional hash,
    and signed challenge. If valid, the handshake is complete on both sides.

Failure modes:
  - Constitutional mismatch: hard reject, logged as PROTOCOL_VIOLATION
  - Expired/invalid certificate: reject with specific error
  - Modified constitution (tampered hash): reject, cannot federate
  - Network partition mid-handshake: timeout after 10s, initiator cleans up
  - Duplicate handshake: idempotent — returns existing link if already linked
  - Nonce replay: each handshake generates a fresh nonce; signatures are
    bound to the specific nonce so replays are detectable

Performance target: ≤3000ms end-to-end (matches link establishment target)
"""

from __future__ import annotations

import base64
import secrets
from typing import TYPE_CHECKING, Any
from datetime import datetime

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:

    from systems.federation.identity import IdentityManager
logger = structlog.get_logger("systems.federation.handshake")


def _generate_nonce() -> str:
    """Generate a 32-byte cryptographic nonce as base64."""
    return base64.b64encode(secrets.token_bytes(32)).decode("ascii")


# ─── Handshake Data Types ───────────────────────────────────────


class HandshakeRequest(EOSBaseModel):
    """
    Phase 1: Initiator sends this to the responder.

    Contains everything the responder needs to verify the initiator's
    identity, alignment, and certificate chain.
    """

    handshake_id: str = Field(default_factory=new_id)
    initiator_instance_id: str
    initiator_name: str = ""
    initiator_endpoint: str = ""

    # Identity card (full public identity)
    identity_card: dict[str, Any] = Field(default_factory=dict)

    # EcodianCertificate (serialized) — proof of alignment
    certificate: dict[str, Any] | None = None

    # Cryptographic challenge: random nonce that responder must sign
    nonce: str = Field(default_factory=_generate_nonce)

    # Constitutional alignment
    constitutional_hash: str = ""
    protocol_version: str = "1.0"

    # Capabilities this instance offers for federation
    capabilities: list[str] = Field(default_factory=list)

    # Negotiation parameters
    max_knowledge_items_per_request: int = 50
    supported_knowledge_types: list[str] = Field(default_factory=list)

    timestamp: datetime = Field(default_factory=utc_now)


class HandshakeResponse(EOSBaseModel):
    """
    Phase 3: Responder returns this to the initiator.

    Contains the responder's identity + certificate + signed challenge,
    plus the responder's own nonce for the initiator to verify.
    """

    handshake_id: str  # Echo back the handshake_id from the request
    accepted: bool
    reject_reason: str = ""

    # Responder identity
    responder_instance_id: str = ""
    responder_name: str = ""
    responder_endpoint: str = ""

    # Identity card (full public identity)
    identity_card: dict[str, Any] = Field(default_factory=dict)

    # EcodianCertificate (serialized)
    certificate: dict[str, Any] | None = None

    # Signed challenge: responder signs the initiator's nonce with its
    # Ed25519 key to prove identity possession
    nonce_signature: str = ""

    # Responder's own nonce for the initiator to sign (mutual verification)
    responder_nonce: str = Field(default_factory=_generate_nonce)

    # Constitutional alignment
    constitutional_hash: str = ""
    protocol_version: str = "1.0"

    # Capabilities
    capabilities: list[str] = Field(default_factory=list)

    # Negotiated parameters
    max_knowledge_items_per_request: int = 50
    supported_knowledge_types: list[str] = Field(default_factory=list)

    timestamp: datetime = Field(default_factory=utc_now)


class HandshakeConfirmation(EOSBaseModel):
    """
    Phase 4: Initiator sends back its signed response to the responder's nonce.

    This completes the mutual authentication — both sides have now proven
    they possess their claimed private keys.
    """

    handshake_id: str
    initiator_instance_id: str
    responder_nonce_signature: str  # Initiator signs the responder's nonce


class HandshakeResult:
    """Internal result of handshake processing — not sent over the wire."""

    __slots__ = (
        "success",
        "error",
        "remote_identity_card",
        "remote_certificate",
        "negotiated_params",
        "handshake_id",
    )

    def __init__(
        self,
        *,
        success: bool,
        error: str = "",
        remote_identity_card: dict[str, Any] | None = None,
        remote_certificate: dict[str, Any] | None = None,
        negotiated_params: dict[str, Any] | None = None,
        handshake_id: str = "",
    ) -> None:
        self.success = success
        self.error = error
        self.remote_identity_card = remote_identity_card
        self.remote_certificate = remote_certificate
        self.negotiated_params = negotiated_params or {}
        self.handshake_id = handshake_id


# ─── Handshake Processor ───────────────────────────────────────


class HandshakeProcessor:
    """
    Processes federation handshakes from either side (initiator or responder).

    Stateless — all state lives in the FederationService. This class
    encapsulates the verification logic so it can be tested independently.
    """

    def __init__(
        self,
        identity: IdentityManager,
        certificate_manager: Any = None,
    ) -> None:
        self._identity = identity
        self._certificate_manager = certificate_manager
        self._logger = logger.bind(component="handshake_processor")

    # ─── Responder Side (Phase 2 + 3) ──────────────────────────

    def process_inbound(
        self,
        request: HandshakeRequest,
    ) -> HandshakeResponse:
        """
        Process an inbound handshake request (responder side).

        Verifies the initiator's identity and certificate, then builds
        a response with this instance's identity and a signed challenge.

        Returns a HandshakeResponse (accepted or rejected with reason).
        """
        local_card = self._identity.identity_card
        reject = _make_reject_response(request.handshake_id)

        # ── Gate 1: Protocol version ──
        if request.protocol_version != "1.0":
            reject.reject_reason = (
                f"Incompatible protocol version: {request.protocol_version} "
                f"(expected 1.0)"
            )
            self._logger.warning(
                "handshake_rejected_protocol",
                remote_id=request.initiator_instance_id,
                version=request.protocol_version,
            )
            return reject

        # ── Gate 2: Constitutional alignment ──
        # This is the fundamental alignment check. Instances with different
        # constitutional invariant hashes CANNOT federate — they have
        # divergent values and federation would be unsafe.
        if request.constitutional_hash != local_card.constitutional_hash:
            reject.reject_reason = (
                "Constitutional invariant mismatch — federation requires "
                "shared constitutional alignment. Remote hash "
                f"{request.constitutional_hash!r} does not match local "
                f"{local_card.constitutional_hash!r}"
            )
            self._logger.warning(
                "handshake_rejected_constitutional_mismatch",
                remote_id=request.initiator_instance_id,
                remote_hash=request.constitutional_hash,
                local_hash=local_card.constitutional_hash,
            )
            return reject

        # ── Gate 3: Identity card validation ──
        from primitives.federation import InstanceIdentityCard

        try:
            remote_identity = InstanceIdentityCard(**request.identity_card)
        except Exception as exc:
            reject.reject_reason = f"Invalid identity card: {exc}"
            return reject

        verification = self._identity.verify_identity(remote_identity)
        if not verification.verified:
            reject.reject_reason = (
                f"Identity verification failed: {'; '.join(verification.errors)}"
            )
            self._logger.warning(
                "handshake_rejected_identity",
                remote_id=request.initiator_instance_id,
                errors=verification.errors,
            )
            return reject

        # ── Gate 4: Certificate validation (Phase 16g) ──
        cert_error = self._validate_certificate(
            request.initiator_instance_id,
            request.certificate,
        )
        if cert_error is not None:
            reject.reject_reason = cert_error
            self._logger.warning(
                "handshake_rejected_certificate",
                remote_id=request.initiator_instance_id,
                error=cert_error,
            )
            return reject

        # ── All gates passed: build accepted response ──

        # Sign the initiator's nonce to prove we possess our private key
        nonce_bytes = request.nonce.encode("utf-8")
        nonce_sig = base64.b64encode(
            self._identity.sign(nonce_bytes)
        ).decode("ascii")

        # Prepare our certificate for the response
        our_cert: dict[str, Any] | None = None
        if self._certificate_manager is not None:
            cert_obj = getattr(self._certificate_manager, "certificate", None)
            if cert_obj is not None:
                our_cert = cert_obj.model_dump(mode="json")

        response = HandshakeResponse(
            handshake_id=request.handshake_id,
            accepted=True,
            responder_instance_id=local_card.instance_id,
            responder_name=local_card.name,
            responder_endpoint=local_card.endpoint,
            identity_card=local_card.model_dump(mode="json"),
            certificate=our_cert,
            nonce_signature=nonce_sig,
            constitutional_hash=local_card.constitutional_hash,
            capabilities=local_card.capabilities,
        )

        self._logger.info(
            "handshake_accepted",
            handshake_id=request.handshake_id,
            remote_id=request.initiator_instance_id,
            remote_name=request.initiator_name,
        )

        return response

    # ─── Initiator Side (Phase 4) ──────────────────────────────

    def verify_response(
        self,
        original_request: HandshakeRequest,
        response: HandshakeResponse,
    ) -> HandshakeResult:
        """
        Verify the responder's handshake response (initiator side, Phase 4).

        Checks:
          1. Response is accepted
          2. Constitutional hash matches ours
          3. Responder's identity card is valid
          4. Responder's certificate is valid
          5. Nonce signature proves responder possesses its claimed private key

        Returns a HandshakeResult (success/failure with details).
        """
        if not response.accepted:
            return HandshakeResult(
                success=False,
                error=f"Handshake rejected by responder: {response.reject_reason}",
                handshake_id=response.handshake_id,
            )

        local_card = self._identity.identity_card

        # ── Verify constitutional alignment (symmetric check) ──
        if response.constitutional_hash != local_card.constitutional_hash:
            return HandshakeResult(
                success=False,
                error=(
                    "Responder constitutional hash mismatch: "
                    f"{response.constitutional_hash!r} != "
                    f"{local_card.constitutional_hash!r}"
                ),
                handshake_id=response.handshake_id,
            )

        # ── Verify responder identity card ──
        from primitives.federation import InstanceIdentityCard

        try:
            remote_identity = InstanceIdentityCard(**response.identity_card)
        except Exception as exc:
            return HandshakeResult(
                success=False,
                error=f"Invalid responder identity card: {exc}",
                handshake_id=response.handshake_id,
            )

        verification = self._identity.verify_identity(remote_identity)
        if not verification.verified:
            return HandshakeResult(
                success=False,
                error=f"Responder identity verification failed: {'; '.join(verification.errors)}",
                handshake_id=response.handshake_id,
            )

        # ── Verify responder certificate (Phase 16g) ──
        cert_error = self._validate_certificate(
            response.responder_instance_id,
            response.certificate,
        )
        if cert_error is not None:
            return HandshakeResult(
                success=False,
                error=f"Responder certificate invalid: {cert_error}",
                handshake_id=response.handshake_id,
            )

        # ── Verify nonce signature (cryptographic proof of identity) ──
        # The responder signed our nonce with its Ed25519 key. Verify
        # using the public key from the responder's identity card.
        if not response.nonce_signature:
            return HandshakeResult(
                success=False,
                error="Responder did not sign the challenge nonce",
                handshake_id=response.handshake_id,
            )

        try:
            sig_bytes = base64.b64decode(response.nonce_signature)
        except Exception:
            return HandshakeResult(
                success=False,
                error="Malformed nonce signature (not valid base64)",
                handshake_id=response.handshake_id,
            )

        nonce_bytes = original_request.nonce.encode("utf-8")
        sig_valid = self._identity.verify_signature(
            data=nonce_bytes,
            signature=sig_bytes,
            remote_public_key_pem=remote_identity.public_key_pem,
        )
        if not sig_valid:
            return HandshakeResult(
                success=False,
                error=(
                    "Nonce signature verification failed — responder cannot "
                    "prove possession of the claimed private key"
                ),
                handshake_id=response.handshake_id,
            )

        # ── Negotiate parameters ──
        negotiated = _negotiate_params(original_request, response)

        self._logger.info(
            "handshake_response_verified",
            handshake_id=response.handshake_id,
            remote_id=response.responder_instance_id,
            remote_name=response.responder_name,
        )

        return HandshakeResult(
            success=True,
            remote_identity_card=response.identity_card,
            remote_certificate=response.certificate,
            negotiated_params=negotiated,
            handshake_id=response.handshake_id,
        )

    # ─── Build Confirmation (Phase 4 outbound) ──────────────────

    def build_confirmation(
        self,
        handshake_id: str,
        responder_nonce: str,
    ) -> HandshakeConfirmation:
        """
        Build the Phase 4 confirmation message.

        Signs the responder's nonce to complete mutual authentication.
        """
        nonce_bytes = responder_nonce.encode("utf-8")
        sig = base64.b64encode(
            self._identity.sign(nonce_bytes)
        ).decode("ascii")

        return HandshakeConfirmation(
            handshake_id=handshake_id,
            initiator_instance_id=self._identity.instance_id,
            responder_nonce_signature=sig,
        )

    # ─── Verify Confirmation (responder side) ───────────────────

    def verify_confirmation(
        self,
        confirmation: HandshakeConfirmation,
        expected_nonce: str,
        initiator_public_key_pem: str,
    ) -> bool:
        """
        Verify the initiator's confirmation (responder completes mutual auth).

        Returns True if the initiator correctly signed our nonce.
        """
        if not confirmation.responder_nonce_signature:
            return False

        try:
            sig_bytes = base64.b64decode(confirmation.responder_nonce_signature)
        except Exception:
            return False

        nonce_bytes = expected_nonce.encode("utf-8")
        return self._identity.verify_signature(
            data=nonce_bytes,
            signature=sig_bytes,
            remote_public_key_pem=initiator_public_key_pem,
        )

    # ─── Certificate Validation ─────────────────────────────────

    def _validate_certificate(
        self,
        source_instance_id: str,
        certificate_data: dict[str, Any] | None,
    ) -> str | None:
        """
        Validate a remote instance's certificate.

        Returns error string if invalid, None if valid.
        Backward-compatible: if no CertificateManager is wired,
        validation is skipped (returns None).
        """
        if self._certificate_manager is None:
            # Pre-Phase 16g: no certificate infrastructure — allow
            return None

        if certificate_data is None:
            return "No certificate provided — federation requires a valid EcodianCertificate"

        from systems.identity.certificate import EcodianCertificate

        try:
            cert = EcodianCertificate.model_validate(certificate_data)
        except Exception as exc:
            return f"Invalid certificate format: {exc}"

        result = self._certificate_manager.validate_certificate(cert)
        if not result.valid:
            return f"Certificate validation failed: {'; '.join(result.errors)}"

        return None


# ─── Internal Helpers ───────────────────────────────────────────


def _make_reject_response(handshake_id: str) -> HandshakeResponse:
    """Create a rejection response template."""
    return HandshakeResponse(
        handshake_id=handshake_id,
        accepted=False,
    )


def _negotiate_params(
    request: HandshakeRequest,
    response: HandshakeResponse,
) -> dict[str, Any]:
    """
    Negotiate communication parameters between initiator and responder.

    Uses the minimum of each side's limits (conservative approach).
    Capability intersection determines what operations are available.
    """
    # Knowledge items: use the smaller limit
    max_items = min(
        request.max_knowledge_items_per_request,
        response.max_knowledge_items_per_request,
    )

    # Capabilities: intersection of what both sides offer
    initiator_caps = set(request.capabilities)
    responder_caps = set(response.capabilities)
    shared_caps = sorted(initiator_caps & responder_caps)

    # Knowledge types: intersection of supported types
    initiator_types = set(request.supported_knowledge_types)
    responder_types = set(response.supported_knowledge_types)
    shared_types = sorted(initiator_types & responder_types) if (initiator_types and responder_types) else []

    return {
        "max_knowledge_items_per_request": max_items,
        "shared_capabilities": shared_caps,
        "shared_knowledge_types": shared_types,
        "protocol_version": "1.0",
    }
