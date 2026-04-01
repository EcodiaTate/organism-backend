"""
EcodiaOS - Benchmarks API Router

Read-only endpoints for the dashboard to query benchmark trends and snapshots.

  GET /api/v1/benchmarks/latest      - Most recent snapshot across all KPIs
  GET /api/v1/benchmarks/trend/{metric} - Time-series for a single KPI
  GET /api/v1/benchmarks/trends      - Time-series for all KPIs (one call)
  GET /api/v1/benchmarks/health      - Service health + regression status
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

from primitives.common import EOSBaseModel
from systems.benchmarks.types import BenchmarkSnapshot, BenchmarkTrend

logger = structlog.get_logger("api.benchmarks")

router = APIRouter(prefix="/api/v1/benchmarks", tags=["benchmarks"])

_VALID_METRICS = frozenset({
    "decision_quality",
    "llm_dependency",
    "economic_ratio",
    "learning_rate",
    "mutation_success_rate",
    "effective_intelligence_ratio",
    "compression_ratio",
})


# ─── Response models ──────────────────────────────────────────────────────


class BenchmarkHealthResponse(EOSBaseModel):
    status: str
    total_runs: int
    total_regressions_fired: int
    currently_regressed: list[str]
    latest_snapshot_time: str | None
    interval_s: float
    rolling_window: int


class AllTrendsResponse(EOSBaseModel):
    trends: dict[str, Any]
    """Keys are KPI names; values are BenchmarkTrend-shaped dicts."""


# ─── Endpoints ────────────────────────────────────────────────────────────


@router.get("/health", response_model=BenchmarkHealthResponse)
async def benchmarks_health(request: Request) -> BenchmarkHealthResponse:
    """Return benchmark service health and current regression status."""
    benchmarks = getattr(request.app.state, "benchmarks", None)
    if benchmarks is None:
        raise HTTPException(status_code=503, detail="Benchmarks service not initialized")
    health = await benchmarks.health()
    return BenchmarkHealthResponse(**health)


@router.get("/latest", response_model=BenchmarkSnapshot)
async def benchmarks_latest(request: Request) -> BenchmarkSnapshot:
    """Return the most recent benchmark snapshot."""
    benchmarks = getattr(request.app.state, "benchmarks", None)
    if benchmarks is None:
        raise HTTPException(status_code=503, detail="Benchmarks service not initialized")
    snapshot = await benchmarks.latest_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No benchmark snapshots recorded yet")
    return snapshot


@router.get("/trend/{metric}", response_model=BenchmarkTrend)
async def benchmarks_trend(
    metric: str,
    request: Request,
    since: datetime | None = Query(default=None, description="ISO datetime lower bound"),
    limit: int = Query(default=50, ge=1, le=500),
) -> BenchmarkTrend:
    """
    Return time-series data for a single KPI.

    metric - one of: decision_quality, llm_dependency, economic_ratio,
             learning_rate, mutation_success_rate, effective_intelligence_ratio,
             compression_ratio
    """
    if metric not in _VALID_METRICS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown metric {metric!r}. Valid: {sorted(_VALID_METRICS)}",
        )
    benchmarks = getattr(request.app.state, "benchmarks", None)
    if benchmarks is None:
        raise HTTPException(status_code=503, detail="Benchmarks service not initialized")
    return await benchmarks.trend(metric, since=since, limit=limit)


@router.get("/trends", response_model=AllTrendsResponse)
async def benchmarks_all_trends(
    request: Request,
    since: datetime | None = Query(default=None, description="ISO datetime lower bound"),
) -> AllTrendsResponse:
    """Return trend data for all five KPIs in a single response."""
    benchmarks = getattr(request.app.state, "benchmarks", None)
    if benchmarks is None:
        raise HTTPException(status_code=503, detail="Benchmarks service not initialized")
    trends = await benchmarks.all_trends(since=since)
    return AllTrendsResponse(trends={k: v.model_dump() for k, v in trends.items()})


# ─── RE Training admin endpoints ──────────────────────────────────────────


class RETrainingStatusResponse(EOSBaseModel):
    training_halted: bool
    halt_reason: str | None
    last_train_at: str | None
    training_runs_total: int


class RETrainingHaltClearResponse(EOSBaseModel):
    cleared: bool
    message: str


@router.get("/re-training/status", response_model=RETrainingStatusResponse)
async def re_training_status(request: Request) -> RETrainingStatusResponse:
    """Return current RE training gate status - halted flag, halt reason, last run."""
    clo = getattr(request.app.state, "continual_learning", None)
    if clo is None:
        raise HTTPException(status_code=503, detail="ContinualLearningOrchestrator not initialized")
    halted, reason = await clo._is_training_halted()
    last_train = getattr(clo, "_last_train_at", None)
    runs = getattr(clo, "_training_runs", [])
    return RETrainingStatusResponse(
        training_halted=halted,
        halt_reason=reason or None,
        last_train_at=last_train.isoformat() if last_train else None,
        training_runs_total=len(runs),
    )


@router.post("/re-training/clear-halt", response_model=RETrainingHaltClearResponse)
async def re_training_clear_halt(request: Request) -> RETrainingHaltClearResponse:
    """Clear a persisted training halt.

    Use after investigating a RE quality regression and confirming the root
    cause is addressed.  The next scheduled training check (≤6 hours) will
    re-evaluate should_train() without the halt gate.

    This endpoint is intentionally unauthenticated at the router level - the
    organism itself calls it via Thymos self-healing when the RE monitor
    confirms quality has recovered above the floor.
    """
    clo = getattr(request.app.state, "continual_learning", None)
    if clo is None:
        raise HTTPException(status_code=503, detail="ContinualLearningOrchestrator not initialized")
    await clo.clear_training_halt()
    logger.info("re_training_halt_cleared_via_api")
    return RETrainingHaltClearResponse(
        cleared=True,
        message="Training halt cleared. Next check_and_train() cycle will re-evaluate should_train().",
    )
