"""
EcodiaOS - Equor API Router

Constitutional ethics engine endpoints. All read-only - no side effects.

Endpoints:
  GET /api/v1/equor/health   - Full health snapshot (drives, drift, autonomy, safe mode)
  GET /api/v1/equor/drift    - Drift history and immune response status
  GET /api/v1/equor/genome   - Constitutional genome snapshot (for CLI + Mitosis inspection)
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request

logger = structlog.get_logger("api.equor")

router = APIRouter(prefix="/api/v1/equor", tags=["equor"])


def _get_equor(request: Request) -> Any | None:
    return getattr(request.app.state, "equor", None)


@router.get("/health")
async def equor_health(request: Request) -> dict[str, Any]:
    """Full Equor health snapshot - spec §13.1 fields plus extended stats."""
    equor = _get_equor(request)
    if equor is None:
        return {"error": "equor not initialized"}

    try:
        return await equor.health()
    except Exception as exc:
        logger.error("equor_health_endpoint_error", error=str(exc))
        return {"error": str(exc)}


@router.get("/drift")
async def equor_drift(request: Request) -> dict[str, Any]:
    """Constitutional drift history and immune response status."""
    equor = _get_equor(request)
    if equor is None:
        return {"error": "equor not initialized"}

    try:
        return await equor.get_drift_report()
    except Exception as exc:
        logger.error("equor_drift_endpoint_error", error=str(exc))
        return {"error": str(exc)}


@router.get("/genome")
async def equor_genome(request: Request) -> dict[str, Any]:
    """Constitutional genome snapshot - drive weights, floor thresholds, amendments, drift history."""
    equor = _get_equor(request)
    if equor is None:
        return {"error": "equor not initialized"}

    try:
        fragment = await equor.export_equor_genome()
        if fragment is None:
            return {"genome": {}}
        # EquorGenomeFragment is a Pydantic model - serialise to dict
        genome_dict = fragment.model_dump() if hasattr(fragment, "model_dump") else dict(fragment)
        return {"genome": genome_dict}
    except Exception as exc:
        logger.error("equor_genome_endpoint_error", error=str(exc))
        return {"error": str(exc)}
