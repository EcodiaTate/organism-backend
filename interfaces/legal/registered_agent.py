"""
EcodiaOS - Registered Agent Client (Stubbed)

Mock API client for registered-agent services (Stripe Atlas, Doola,
Wyoming Agents, etc.) that handle LLC/DAO formation filings.

In production, this would make real HTTP calls to the agent's API:
  1. Submit entity details + operating agreement + payment intent
  2. Poll for filing status
  3. Retrieve confirmed entity ID and filing number

For now, all methods return deterministic stub responses so the
executor state machine can be developed and tested end-to-end without
a live filing API.

The client is stateless - it holds no business logic beyond what it
receives from the caller. All persistence is handled by the executor.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from interfaces.legal.types import (
        EntityParameters,
        LegalDocument,
    )

logger = structlog.get_logger("interfaces.legal.registered_agent")


# ─── API Response Models ─────────────────────────────────────────────


class SubmissionResponse(EOSBaseModel):
    """Response from the registered agent API after initial submission."""

    submission_id: str
    """Agent-assigned submission identifier."""

    status: str = "pending"
    """'pending', 'processing', 'requires_kyc', 'completed', 'failed'."""

    portal_url: str = ""
    """URL where the human completes KYC / document signing."""

    estimated_completion_days: int = 5
    """Estimated days until filing is complete."""

    filing_fee_charged_usd: Decimal = Decimal("0")
    """Actual fee charged (may differ from estimate)."""

    message: str = ""
    """Human-readable status message."""


class StatusResponse(EOSBaseModel):
    """Response from polling the filing status."""

    submission_id: str
    status: str = "pending"
    entity_id: str = ""
    filing_number: str = ""
    formation_date: str = ""
    message: str = ""


class PaymentIntentResponse(EOSBaseModel):
    """Response from creating a payment intent for filing fees."""

    payment_intent_id: str
    amount_usd: Decimal
    currency: str = "usd"
    status: str = "requires_confirmation"
    """'requires_confirmation', 'confirmed', 'failed'."""


# ─── Client ──────────────────────────────────────────────────────────


class RegisteredAgentClient:
    """
    Stubbed client for a registered-agent filing API.

    All methods return deterministic stub responses. Replace with real
    HTTP calls when integrating with Stripe Atlas, Doola, or similar.

    Thread-safety: NOT thread-safe. Single-threaded asyncio like all EOS.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.stub-registered-agent.example.com",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._logger = logger.bind(component="registered_agent_client")

    async def create_payment_intent(
        self,
        amount_usd: Decimal,
        description: str = "",
    ) -> PaymentIntentResponse:
        """
        Create a payment intent for the filing fee.

        Stub: returns a deterministic payment intent ID.
        Production: calls Stripe or the agent's payment API.
        """
        intent_id = f"pi_stub_{new_id()[:16]}"

        self._logger.info(
            "payment_intent_created",
            intent_id=intent_id,
            amount_usd=str(amount_usd),
            stub=True,
        )

        return PaymentIntentResponse(
            payment_intent_id=intent_id,
            amount_usd=amount_usd,
            status="confirmed",  # Stub auto-confirms
        )

    async def submit_formation(
        self,
        params: EntityParameters,
        documents: list[LegalDocument],
        payment_intent_id: str,
    ) -> SubmissionResponse:
        """
        Submit entity formation request to the registered agent.

        Stub: returns a deterministic submission ID and a fake portal URL.
        Production: POST to the agent's filing API with documents and payment.
        """
        # Deterministic submission ID from entity name
        raw = f"sub-{params.organism_name}-{params.entity_type.value}"
        submission_id = f"sub_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

        portal_url = (
            f"{self._base_url}/portal/{submission_id}"
            f"?kyc=required&sign=wet"
        )

        self._logger.info(
            "formation_submitted",
            submission_id=submission_id,
            entity_name=params.organism_name,
            entity_type=params.entity_type.value,
            jurisdiction=params.jurisdiction.value,
            documents_count=len(documents),
            payment_intent_id=payment_intent_id,
            stub=True,
        )

        return SubmissionResponse(
            submission_id=submission_id,
            status="requires_kyc",
            portal_url=portal_url,
            estimated_completion_days=5,
            filing_fee_charged_usd=Decimal("500"),
            message=(
                "Formation request received. The human organiser must complete "
                "KYC verification and sign the Articles of Organization at the "
                "portal URL before filing can proceed."
            ),
        )

    async def check_status(
        self,
        submission_id: str,
    ) -> StatusResponse:
        """
        Poll the filing status for a submission.

        Stub: always returns 'requires_kyc' (the HITL gate).
        Production: GET from the agent's status endpoint.
        """
        self._logger.debug(
            "status_check",
            submission_id=submission_id,
            stub=True,
        )

        return StatusResponse(
            submission_id=submission_id,
            status="requires_kyc",
            message=(
                "Awaiting human organiser KYC completion and wet signature. "
                "The entity will be filed once these steps are completed."
            ),
        )

    async def confirm_entity(
        self,
        submission_id: str,
        entity_id: str,
        filing_number: str = "",
    ) -> StatusResponse:
        """
        Confirm that entity formation is complete (called after HITL step).

        Stub: echoes back the provided entity ID.
        Production: may verify with the agent's API or directly with
        the Secretary of State.
        """
        self._logger.info(
            "entity_confirmed",
            submission_id=submission_id,
            entity_id=entity_id,
            filing_number=filing_number,
            stub=True,
        )

        return StatusResponse(
            submission_id=submission_id,
            status="completed",
            entity_id=entity_id,
            filing_number=filing_number or f"WY-{new_id()[:8]}",
            formation_date=utc_now().strftime("%Y-%m-%d"),
            message="Entity formation confirmed and filed with the Secretary of State.",
        )
