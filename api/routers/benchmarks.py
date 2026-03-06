"""
EcodiaOS — Benchmarks API Router

Read-only endpoints for the dashboard to query benchmark trends and snapshots.

  GET /api/v1/benchmarks/latest      — Most recent snapshot across all KPIs
  GET /api/v1/benchmarks/trend/{metric} — Time-series for a single KPI
  GET /api/v1/benchmarks/trends      — Time-series for all KPIs (one call)
  GET /api/v1/benchmarks/health      — Service health + regression status
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic needs at runtime
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

    metric — one of: decision_quality, llm_dependency, economic_ratio,
             learning_rate, mutation_success_rate
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
