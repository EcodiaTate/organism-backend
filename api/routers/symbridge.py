"""
EcodiaOS Symbridge Router - Receives messages from EcodiaOS admin hub.

This is the organism's inbound endpoint for Factory results, health checks,
memory syncs, and metabolism reports from its human-facing cortex.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/symbridge", tags=["symbridge"])


class SymbridgeMessage(BaseModel):
    type: str
    payload: dict[str, Any]
    source: str = "ecodiaos"
    correlationId: str | None = None
    signature: str | None = None
    timestamp: str | None = None


class SymbridgeResponse(BaseModel):
    accepted: bool
    message: str = ""


@router.post("/inbound", response_model=SymbridgeResponse)
async def receive_inbound(msg: SymbridgeMessage, request: Request) -> SymbridgeResponse:
    """Receive messages from EcodiaOS admin hub (Factory results, health, memory, metabolism)."""
    try:
        event_bus = getattr(request.app.state, "event_bus", None)

        if msg.type == "factory_result":
            # Factory completed a CC session — route to Evo for learning
            if event_bus:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await event_bus.emit(SynapseEvent(
                    type=SynapseEventType.FACTORY_RESULT_RECEIVED,
                    payload={
                        "proposal_id": msg.correlationId,
                        "session_id": msg.payload.get("session_id"),
                        "status": msg.payload.get("status"),
                        "files_changed": msg.payload.get("files_changed", []),
                        "commit_sha": msg.payload.get("commit_sha"),
                        "confidence": msg.payload.get("confidence_score"),
                    },
                ))

            # Determine success/failure for appropriate event
            deploy_status = msg.payload.get("deploy_status")
            if deploy_status == "deployed" and event_bus:
                await event_bus.emit(SynapseEvent(
                    type=SynapseEventType.FACTORY_DEPLOY_SUCCEEDED,
                    payload={
                        "session_id": msg.payload.get("session_id"),
                        "codebase": msg.payload.get("codebase_name"),
                        "commit_sha": msg.payload.get("commit_sha"),
                    },
                ))
            elif deploy_status in ("failed", "reverted") and event_bus:
                await event_bus.emit(SynapseEvent(
                    type=SynapseEventType.FACTORY_DEPLOY_FAILED,
                    payload={
                        "session_id": msg.payload.get("session_id"),
                        "codebase": msg.payload.get("codebase_name"),
                        "error": msg.payload.get("error_message", ""),
                        "reverted": deploy_status == "reverted",
                    },
                ))

            return SymbridgeResponse(accepted=True, message="Factory result processed")

        elif msg.type == "health" or msg.type == "heartbeat":
            # EcodiaOS health report — update Skia's symbiont monitoring
            logger.debug("Symbiont health received", status=msg.payload.get("status"))
            return SymbridgeResponse(accepted=True, message="Health acknowledged")

        elif msg.type == "memory_sync":
            # Memory cross-pollination from admin KG
            memory = getattr(request.app.state, "memory", None)
            if memory:
                entities = msg.payload.get("entities", [])
                for entity in entities:
                    try:
                        await memory.store_entity(
                            name=entity.get("name"),
                            labels=entity.get("labels", ["Concept"]),
                            properties={
                                **entity.get("properties", {}),
                                "synced_from": "ecodiaos",
                            },
                        )
                    except Exception:
                        pass
            return SymbridgeResponse(accepted=True, message=f"Memory sync: {len(msg.payload.get('entities', []))} entities")

        elif msg.type == "metabolism":
            # Cost report from EcodiaOS — feed to Oikos
            oikos = getattr(request.app.state, "oikos", None)
            if oikos and hasattr(oikos, "receive_symbiont_costs"):
                await oikos.receive_symbiont_costs(msg.payload)
            return SymbridgeResponse(accepted=True, message="Metabolism report received")

        elif msg.type == "capability_created":
            # Factory built a capability we requested
            if event_bus:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await event_bus.emit(SynapseEvent(
                    type=SynapseEventType.CAPABILITY_CREATED,
                    payload={
                        "description": msg.payload.get("description"),
                        "session_id": msg.payload.get("session_id"),
                        "files_changed": msg.payload.get("files_changed", []),
                    },
                ))
            return SymbridgeResponse(accepted=True, message="Capability creation acknowledged")

        else:
            logger.warning("Unknown symbridge message type", type=msg.type)
            return SymbridgeResponse(accepted=False, message=f"Unknown type: {msg.type}")

    except Exception as exc:
        logger.error("Symbridge inbound processing failed", error=str(exc), type=msg.type)
        return SymbridgeResponse(accepted=False, message=str(exc))


@router.get("/status")
async def symbridge_status(request: Request) -> dict[str, Any]:
    """Return symbridge connection status."""
    return {
        "organism_side": "ready",
        "event_bus_available": hasattr(request.app.state, "event_bus"),
        "memory_available": hasattr(request.app.state, "memory"),
    }
