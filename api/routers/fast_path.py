"""
EcodiaOS - Arbitrage Reflex Arc API Router

Admin endpoints for managing constitutional templates and monitoring
fast-path execution performance.

  GET  /api/v1/fast-path/stats      - Aggregate metrics
  GET  /api/v1/fast-path/templates   - List active templates
  POST /api/v1/fast-path/templates   - Register a new template
  DELETE /api/v1/fast-path/templates/{template_id} - Revoke a template
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.fast_path")

router = APIRouter(prefix="/api/v1/fast-path", tags=["fast-path"])


# ─── Response Models ──────────────────────────────────────────────


class FastPathStatsResponse(EOSBaseModel):
    """Aggregate statistics for the Arbitrage Reflex Arc."""

    template_library: dict[str, Any] = Field(default_factory=dict)
    market_pattern: dict[str, Any] = Field(default_factory=dict)
    executor: dict[str, Any] = Field(default_factory=dict)


class TemplateResponse(EOSBaseModel):
    template_id: str
    pattern_signature: dict[str, Any] = Field(default_factory=dict)
    max_capital_per_execution: float = 0.0
    approval_confidence: float = 0.0
    active: bool = True
    consecutive_failures: int = 0
    total_executions: int = 0
    total_capital_deployed: float = 0.0


class RegisterTemplateRequest(EOSBaseModel):
    template_id: str
    pattern_signature: dict[str, Any] = Field(default_factory=dict)
    max_capital_per_execution: float = 100.0
    approval_confidence: float = 0.95


# ─── Routes ───────────────────────────────────────────────────────


@router.get("/stats", response_model=FastPathStatsResponse)
async def get_stats(request: Request) -> FastPathStatsResponse:
    """Aggregate metrics for the Arbitrage Reflex Arc."""
    equor = getattr(request.app.state, "equor", None)
    axon = getattr(request.app.state, "axon", None)
    atune = getattr(request.app.state, "atune", None)

    template_stats: dict[str, Any] = {}
    if equor is not None:
        template_stats = equor.template_library.stats

    executor_stats: dict[str, Any] = {}
    if axon is not None and hasattr(axon, "_fast_path") and axon._fast_path is not None:
        executor_stats = axon._fast_path.stats

    market_stats: dict[str, Any] = {}
    if atune is not None and hasattr(atune, "_market_pattern") and atune._market_pattern is not None:
        market_stats = atune._market_pattern.stats

    return FastPathStatsResponse(
        template_library=template_stats,
        market_pattern=market_stats,
        executor=executor_stats,
    )


@router.get("/templates")
async def list_templates(request: Request) -> list[TemplateResponse]:
    """List all active constitutional templates."""
    equor = getattr(request.app.state, "equor", None)
    if equor is None:
        return []

    templates = equor.template_library.list_active()
    return [
        TemplateResponse(
            template_id=t.template_id,
            pattern_signature=t.pattern_signature,
            max_capital_per_execution=t.max_capital_per_execution,
            approval_confidence=t.approval_confidence,
            active=t.active,
            consecutive_failures=t.consecutive_failures,
            total_executions=t.total_executions,
            total_capital_deployed=t.total_capital_deployed,
        )
        for t in templates
    ]


@router.post("/templates")
async def register_template(
    request: Request,
    body: RegisterTemplateRequest,
) -> JSONResponse:
    """Register a new constitutional template for fast-path execution."""
    equor = getattr(request.app.state, "equor", None)
    if equor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Equor service not available"},
        )

    check = await equor.register_template(
        template_id=body.template_id,
        pattern_signature=body.pattern_signature,
        max_capital_per_execution=body.max_capital_per_execution,
        approval_confidence=body.approval_confidence,
    )

    status_code = 201 if check.verdict.value == "approved" else 400
    return JSONResponse(
        status_code=status_code,
        content={
            "verdict": check.verdict.value,
            "reasoning": check.reasoning,
            "template_id": body.template_id,
        },
    )


@router.delete("/templates/{template_id}")
async def revoke_template(
    request: Request,
    template_id: str,
    reason: str = "admin_revocation",
) -> JSONResponse:
    """Revoke a constitutional template."""
    equor = getattr(request.app.state, "equor", None)
    if equor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Equor service not available"},
        )

    success = await equor.revoke_template(template_id, reason=reason)
    if not success:
        return JSONResponse(
            status_code=404,
            content={"error": f"Template '{template_id}' not found"},
        )

    return JSONResponse(
        status_code=200,
        content={"revoked": template_id, "reason": reason},
    )
