"""
EcodiaOS — Legal Entity API Router

Endpoints for the legal entity provisioning HITL (Human-in-the-Loop) flow.

The organism initiates entity formation autonomously via the
EstablishEntityExecutor. When it reaches the HITL gate, the human
operator uses these endpoints to:
  1. Check the status of a pending formation
  2. Confirm entity registration after completing KYC / wet signature

Endpoints:
  GET  /api/v1/legal/formation/{auth_code}   — check formation status
  POST /api/v1/legal/formation/{auth_code}/confirm — confirm entity ID
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.legal")

router = APIRouter(prefix="/api/v1/legal", tags=["legal"])


# ─── Schemas ──────────────────────────────────────────────────────────


class ConfirmEntityRequest(EOSBaseModel):
    """Request body for confirming entity formation after HITL."""

    entity_id: str = Field(
        ...,
        description="State-issued entity ID (e.g., Wyoming SoS filing number)",
    )
    filing_number: str = Field(
        default="",
        description="Optional filing reference number from Secretary of State",
    )


class FormationStatusResponse(EOSBaseModel):
    """Response for formation status check."""

    auth_code: str
    state: str
    organism_name: str = ""
    entity_type: str = ""
    submission_id: str = ""
    portal_url: str = ""
    action_required: str = ""


class ConfirmEntityResponse(EOSBaseModel):
    """Response after successful entity confirmation."""

    success: bool
    entity_id: str = ""
    entity_name: str = ""
    entity_type: str = ""
    jurisdiction: str = ""
    filing_number: str = ""
    registration_id: str = ""
    message: str = ""


# ─── Endpoints ────────────────────────────────────────────────────────


@router.get("/formation/{auth_code}")
async def get_formation_status(
    auth_code: str,
    request: Request,
) -> FormationStatusResponse:
    """
    Check the status of a pending entity formation.

    The auth_code is the 4-digit code emitted by the executor when it
    paused at the HITL gate.
    """
    executor = _get_executor(request)

    record = await executor._retrieve_formation_state(auth_code)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No pending formation found for auth code '{auth_code}'. "
                "It may have expired (7-day TTL) or already been confirmed."
            ),
        )

    hitl = record.hitl_instruction
    return FormationStatusResponse(
        auth_code=auth_code,
        state=record.state.value,
        organism_name=record.entity_parameters.organism_name,
        entity_type=record.entity_parameters.entity_type.value,
        submission_id=hitl.submission_id if hitl else "",
        portal_url=hitl.portal_url if hitl else "",
        action_required=hitl.action_required if hitl else "",
    )


@router.post("/formation/{auth_code}/confirm")
async def confirm_entity_formation(
    auth_code: str,
    body: ConfirmEntityRequest,
    request: Request,
) -> ConfirmEntityResponse:
    """
    Confirm that entity formation is complete.

    The human operator calls this endpoint after:
      1. Completing KYC verification at the registered agent portal
      2. Signing the Articles of Organization (wet signature)
      3. Receiving the entity ID from the Secretary of State

    This triggers the executor's resume path, which stores the entity
    identity in the IdentityVault and emits a completion event.
    """
    executor = _get_executor(request)

    result = await executor.resume_after_human_confirmation(
        auth_code=auth_code,
        entity_id=body.entity_id,
        filing_number=body.filing_number,
    )

    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=result.error,
        )

    return ConfirmEntityResponse(
        success=True,
        entity_id=result.data.get("entity_id", ""),
        entity_name=result.data.get("entity_name", ""),
        entity_type=result.data.get("entity_type", ""),
        jurisdiction=result.data.get("jurisdiction", ""),
        filing_number=result.data.get("filing_number", ""),
        registration_id=result.data.get("registration_id", ""),
        message=(
            "Entity formation confirmed. Identity stored in IdentityVault. "
            "The organism now has legal personhood."
        ),
    )


# ─── Helpers ──────────────────────────────────────────────────────────


def _get_executor(request: Request) -> Any:
    """
    Retrieve the EstablishEntityExecutor from the Axon registry.

    The executor is registered during Axon initialisation — no separate
    app.state entry is needed; Axon owns the executor registry.
    """
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        raise HTTPException(
            status_code=503,
            detail="Axon not initialized. Legal provisioning is unavailable.",
        )
    executor = axon.get_executor("establish_entity")
    if executor is None:
        raise HTTPException(
            status_code=503,
            detail="EstablishEntityExecutor not registered. Legal provisioning is unavailable.",
        )
    return executor
