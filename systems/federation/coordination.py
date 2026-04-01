"""
EcodiaOS - Federation Coordinated Action

Instances can request assistance from each other. Coordinated action
requires COLLEAGUE trust level or higher. The requesting instance
describes what help is needed; the responding instance evaluates
whether assisting aligns with its own drives (via Nova) and passes
constitutional review (via Equor).

This is mutual aid between sovereign entities, not delegation or
command. Each instance freely chooses whether to help.
"""

from __future__ import annotations

from typing import Any

import structlog

from primitives.common import utc_now
from primitives.federation import (
    AssistanceRequest,
    AssistanceResponse,
    FederationInteraction,
    FederationLink,
    InteractionOutcome,
    TrustLevel,
)

logger = structlog.get_logger("systems.federation.coordination")


class CoordinationManager:
    """
    Manages coordinated action between federated instances.

    Handles both inbound requests (other instances asking us for help)
    and outbound requests (us asking other instances for help).
    """

    def __init__(self) -> None:
        self._logger = logger.bind(component="coordination")

        # Tracking
        self._requests_received: int = 0
        self._requests_accepted: int = 0
        self._requests_declined: int = 0
        self._requests_sent: int = 0
        self._active_assistance: dict[str, AssistanceRequest] = {}

    # ─── Inbound: Handle Requests from Remote Instances ─────────────

    async def handle_request(
        self,
        request: AssistanceRequest,
        link: FederationLink,
        equor_permitted: bool = True,
        nova_aligned: bool = True,
    ) -> tuple[AssistanceResponse, FederationInteraction]:
        """
        Handle an inbound assistance request from a federated instance.

        Steps:
          1. Verify trust level (COLLEAGUE+ required)
          2. Check Nova alignment (does this align with our goals?)
          3. Check Equor review (is this constitutionally permitted?)
          4. Accept or decline

        Returns both the response and an interaction for trust scoring.
        """
        self._requests_received += 1
        start_time = utc_now()

        # Step 1: Trust check
        if link.trust_level.value < TrustLevel.COLLEAGUE.value:
            self._requests_declined += 1

            response = AssistanceResponse(
                request_id=request.id,
                accepted=False,
                reason="Insufficient trust level for coordinated action.",
            )

            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=link.remote_instance_id,
                interaction_type="assistance_request",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description="Declined: insufficient trust",
                trust_value=0.5,
                latency_ms=_elapsed_ms(start_time),
            )

            self._logger.info(
                "assistance_declined_trust",
                remote_id=link.remote_instance_id,
                trust_level=link.trust_level.name,
            )

            return response, interaction

        # Step 2: Nova alignment check
        if not nova_aligned:
            self._requests_declined += 1

            response = AssistanceResponse(
                request_id=request.id,
                accepted=False,
                reason="This request doesn't align with my current priorities.",
            )

            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=link.remote_instance_id,
                interaction_type="assistance_request",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description="Declined: not aligned with current goals",
                trust_value=0.5,
                latency_ms=_elapsed_ms(start_time),
            )

            return response, interaction

        # Step 3: Equor constitutional review
        if not equor_permitted:
            self._requests_declined += 1

            response = AssistanceResponse(
                request_id=request.id,
                accepted=False,
                reason="Constitutional review did not permit this assistance.",
            )

            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=link.remote_instance_id,
                interaction_type="assistance_request",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description="Declined: Equor review blocked",
                trust_value=0.5,
                latency_ms=_elapsed_ms(start_time),
            )

            return response, interaction

        # Step 4: Accept
        self._requests_accepted += 1
        self._active_assistance[request.id] = request

        response = AssistanceResponse(
            request_id=request.id,
            accepted=True,
            estimated_completion_ms=5000,  # Default estimate
        )

        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type="assistance_request",
            direction="inbound",
            outcome=InteractionOutcome.SUCCESSFUL,
            description=f"Accepted: {request.description[:100]}",
            trust_value=2.0,  # Providing assistance builds significant trust
            latency_ms=_elapsed_ms(start_time),
        )

        self._logger.info(
            "assistance_accepted",
            remote_id=link.remote_instance_id,
            request_id=request.id,
            domain=request.knowledge_domain,
        )

        return response, interaction

    # ─── Outbound: Send Requests to Remote Instances ────────────────

    def build_request(
        self,
        description: str,
        knowledge_domain: str = "",
        urgency: float = 0.5,
        reciprocity_offer: str | None = None,
        local_instance_id: str = "",
    ) -> AssistanceRequest:
        """Build an assistance request to send to a remote instance."""
        self._requests_sent += 1
        return AssistanceRequest(
            requesting_instance_id=local_instance_id,
            description=description,
            knowledge_domain=knowledge_domain,
            urgency=urgency,
            reciprocity_offer=reciprocity_offer,
        )

    def complete_assistance(self, request_id: str) -> None:
        """Mark an assistance request as completed."""
        self._active_assistance.pop(request_id, None)

    # ─── Stats ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "requests_received": self._requests_received,
            "requests_accepted": self._requests_accepted,
            "requests_declined": self._requests_declined,
            "requests_sent": self._requests_sent,
            "active_assistance": len(self._active_assistance),
        }


def _elapsed_ms(start: Any) -> int:
    delta = utc_now() - start
    return int(delta.total_seconds() * 1000)
