"""
EcodiaOS - Simula API Router

Exposes Simula's self-evolution pipeline and Inspector subsystem:

  GET  /api/v1/simula/status         - stats, subsystem health, grid state
  GET  /api/v1/simula/analytics      - EvolutionAnalytics snapshot
  GET  /api/v1/simula/proposals      - active proposals in the pipeline
  GET  /api/v1/simula/history        - recent EvolutionRecords
  GET  /api/v1/simula/version        - current version + chain
  GET  /api/v1/simula/inspector      - Inspector analytics view
  GET  /api/v1/simula/inspector/hunts - recent hunt results (in-memory)
  POST /api/v1/simula/approve        - approve a governed proposal
"""

from __future__ import annotations

import contextlib
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.simula")

router = APIRouter(prefix="/api/v1/simula", tags=["simula"])


# ─── Response Models ──────────────────────────────────────────────


class SubsystemStageStatus(EOSBaseModel):
    stage3: dict[str, bool] = {}
    stage4: dict[str, bool] = {}
    stage5: dict[str, bool] = {}
    stage6: dict[str, bool] = {}
    stage7: dict[str, bool] = {}


class SimulaStatusResponse(EOSBaseModel):
    initialized: bool = False
    current_version: int = 0
    grid_state: str = "normal"
    proposals_received: int = 0
    proposals_approved: int = 0
    proposals_rejected: int = 0
    proposals_rolled_back: int = 0
    proposals_deduplicated: int = 0
    proposals_awaiting_governance: int = 0
    active_proposals: int = 0
    subsystems: SubsystemStageStatus = Field(default_factory=SubsystemStageStatus)
    analytics_summary: dict[str, Any] = {}
    architecture_efe: dict[str, Any] = {}
    stage9_analytics: dict[str, Any] = {}


class SimulaAnalyticsResponse(EOSBaseModel):
    total_proposals: int = 0
    approved_proposals: int = 0
    rejected_proposals: int = 0
    rolled_back_proposals: int = 0
    evolution_velocity: float = 0.0
    rollback_rate: float = 0.0
    mean_simulation_risk: float = 0.0
    approval_rate: float = 0.0
    recent_risk_trend: list[float] = []
    category_distribution: dict[str, int] = {}
    inspector_total_hunts: int = 0
    inspector_total_vulnerabilities: int = 0
    inspector_critical_count: int = 0
    inspector_high_count: int = 0


class ActiveProposalItem(EOSBaseModel):
    id: str
    source: str
    category: str
    description: str
    status: str
    risk_assessment: str = ""
    efe_score: float | None = None
    dream_origin: bool = False
    created_at: str = ""


class ActiveProposalsResponse(EOSBaseModel):
    proposals: list[ActiveProposalItem] = []
    total: int = 0


class EvolutionHistoryItem(EOSBaseModel):
    id: str
    proposal_id: str
    category: str
    description: str
    from_version: int
    to_version: int
    files_changed: list[str] = []
    simulation_risk: str
    applied_at: str
    rolled_back: bool
    rollback_reason: str = ""
    formal_verification_status: str = ""
    repair_agent_used: bool = False
    repair_cost_usd: float = 0.0


class EvolutionHistoryResponse(EOSBaseModel):
    records: list[EvolutionHistoryItem] = []
    total: int = 0


class VersionChainItem(EOSBaseModel):
    version: int
    timestamp: str
    proposal_count: int
    config_hash: str


class SimulaVersionResponse(EOSBaseModel):
    current_version: int = 0
    chain: list[VersionChainItem] = []


class InspectorStatsResponse(EOSBaseModel):
    enabled: bool = False
    total_hunts: int = 0
    total_vulnerabilities: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    avg_surfaces_per_hunt: float = 0.0
    avg_duration_ms: float = 0.0
    analytics_emitter_active: bool = False
    tsdb_persistence_active: bool = False
    recent_event_types: dict[str, int] = {}


class HuntSummaryItem(EOSBaseModel):
    id: str
    target_url: str
    target_type: str
    surfaces_mapped: int
    vulnerabilities_found: int
    critical_count: int
    high_count: int
    total_duration_ms: int
    started_at: str
    completed_at: str | None = None


class HuntsResponse(EOSBaseModel):
    hunts: list[HuntSummaryItem] = []
    total: int = 0


class ApproveGovernedRequest(EOSBaseModel):
    proposal_id: str
    governance_record_id: str


class ApproveGovernedResponse(EOSBaseModel):
    status: str
    reason: str = ""
    version: int | None = None


class SimulaHealthResponse(EOSBaseModel):
    """Full Simula health including self-healing component status (Task 6)."""
    status: str = "unhealthy"
    current_version: int = 0
    active_proposals: int = 0
    proactive_scanner_alive: bool = False
    repair_memory_record_count: int = 0
    last_proposal_processed_at: str | None = None
    calibration_score: float | None = None
    reason: str = ""


class SimulaMetricsResponse(EOSBaseModel):
    """Full operational metrics (Task 6)."""
    proposals_received_session: int = 0
    proposals_approved_session: int = 0
    proposals_rejected_session: int = 0
    proposals_rolled_back_session: int = 0
    success_rate: float = 0.0
    rollback_rate: float = 0.0
    proactive_proposals_generated: int = 0
    proactive_vs_received_ratio: float = 0.0
    calibration_score: float | None = None
    repair_memory_record_count: int = 0
    last_proposal_processed_at: str | None = None
    proactive_scanner: dict[str, Any] = {}


class RepairMemoryResponse(EOSBaseModel):
    """Repair outcome learning summary (Task 2)."""
    success_rates_by_category: dict[str, float] = {}
    total_proposals: int = 0
    rollback_rate: float = 0.0
    most_reliable_change_type: str = ""
    most_risky_change_type: str = ""
    calibration_score: float = 1.0
    calibration_window_size: int = 0


# ─── Helpers ──────────────────────────────────────────────────────


def _get_simula(request: Request) -> Any:
    simula = getattr(request.app.state, "simula", None)
    if simula is None:
        raise HTTPException(503, "Simula not initialized")
    return simula


# ─── Endpoints ────────────────────────────────────────────────────


@router.get("/status", response_model=SimulaStatusResponse)
async def get_simula_status(request: Request) -> SimulaStatusResponse:
    """Current Simula service state: proposal counters, grid state, subsystems."""
    simula = _get_simula(request)
    try:
        raw = simula.stats
    except Exception as exc:
        logger.warning("simula_status_error", error=str(exc))
        raise HTTPException(500, f"Failed to read Simula stats: {exc}") from exc

    return SimulaStatusResponse(
        initialized=raw.get("initialized", False),
        current_version=raw.get("current_version", 0),
        grid_state=getattr(simula, "_grid_state", "normal"),
        proposals_received=raw.get("proposals_received", 0),
        proposals_approved=raw.get("proposals_approved", 0),
        proposals_rejected=raw.get("proposals_rejected", 0),
        proposals_rolled_back=raw.get("proposals_rolled_back", 0),
        proposals_deduplicated=raw.get("proposals_deduplicated", 0),
        proposals_awaiting_governance=raw.get("proposals_awaiting_governance", 0),
        active_proposals=raw.get("active_proposals", 0),
        subsystems=SubsystemStageStatus(
            stage3=raw.get("stage3", {}),
            stage4=raw.get("stage4", {}),
            stage5=raw.get("stage5", {}),
            stage6=raw.get("stage6", {}),
            stage7=raw.get("stage7", {}),
        ),
        analytics_summary=raw.get("analytics", {}),
        architecture_efe=raw.get("architecture_efe", {}),
        stage9_analytics=raw.get("stage9_analytics", {}),
    )


@router.get("/analytics", response_model=SimulaAnalyticsResponse)
async def get_simula_analytics(request: Request) -> SimulaAnalyticsResponse:
    """Full EvolutionAnalytics snapshot - proposal quality, velocity, rollback rate."""
    simula = _get_simula(request)
    try:
        analytics = await simula.get_analytics()
    except Exception as exc:
        logger.warning("simula_analytics_error", error=str(exc))
        raise HTTPException(500, f"Failed to compute analytics: {exc}") from exc

    # Pull inspector analytics from the analytics engine if attached
    inspector_total = 0
    inspector_critical = 0
    inspector_high = 0
    inspector_vulns = 0

    inspector_view = getattr(
        getattr(simula, "_analytics", None), "_inspector_view", None
    )
    if inspector_view is not None:
        try:
            iv = inspector_view.snapshot()
            inspector_total = iv.get("total_hunts", 0)
            inspector_vulns = iv.get("total_vulnerabilities", 0)
            inspector_critical = iv.get("critical_count", 0)
            inspector_high = iv.get("high_count", 0)
        except Exception:
            pass

    return SimulaAnalyticsResponse(
        total_proposals=getattr(analytics, "total_proposals", 0),
        approved_proposals=getattr(analytics, "approved_proposals", 0),
        rejected_proposals=getattr(analytics, "rejected_proposals", 0),
        rolled_back_proposals=getattr(analytics, "rolled_back_proposals", 0),
        evolution_velocity=getattr(analytics, "evolution_velocity", 0.0),
        rollback_rate=getattr(analytics, "rollback_rate", 0.0),
        mean_simulation_risk=getattr(analytics, "mean_simulation_risk", 0.0),
        approval_rate=getattr(analytics, "approval_rate", 0.0),
        recent_risk_trend=getattr(analytics, "recent_risk_trend", []),
        category_distribution=getattr(analytics, "category_distribution", {}),
        inspector_total_hunts=inspector_total,
        inspector_total_vulnerabilities=inspector_vulns,
        inspector_critical_count=inspector_critical,
        inspector_high_count=inspector_high,
    )


@router.get("/proposals", response_model=ActiveProposalsResponse)
async def get_active_proposals(request: Request) -> ActiveProposalsResponse:
    """All proposals currently in the pipeline."""
    simula = _get_simula(request)
    try:
        proposals = simula.get_active_proposals()
    except Exception as exc:
        logger.warning("simula_proposals_error", error=str(exc))
        raise HTTPException(500, f"Failed to read proposals: {exc}") from exc

    items = [
        ActiveProposalItem(
            id=p.id,
            source=p.source,
            category=p.category.value if hasattr(p.category, "value") else str(p.category),
            description=p.description,
            status=p.status.value if hasattr(p.status, "value") else str(p.status),
            risk_assessment=p.risk_assessment or "",
            efe_score=p.efe_score,
            dream_origin=p.dream_origin,
            created_at=p.created_at.isoformat() if hasattr(p, "created_at") and p.created_at else "",
        )
        for p in proposals
    ]
    return ActiveProposalsResponse(proposals=items, total=len(items))


@router.get("/history", response_model=EvolutionHistoryResponse)
async def get_evolution_history(
    request: Request,
    limit: int = 50,
) -> EvolutionHistoryResponse:
    """Recent evolution records (immutable audit trail from Neo4j)."""
    simula = _get_simula(request)
    try:
        records = await simula.get_history(limit=min(limit, 200))
    except Exception as exc:
        logger.warning("simula_history_error", error=str(exc))
        raise HTTPException(500, f"Failed to read history: {exc}") from exc

    items = []
    for r in records:
        items.append(EvolutionHistoryItem(
            id=r.id,
            proposal_id=r.proposal_id,
            category=r.category.value if hasattr(r.category, "value") else str(r.category),
            description=r.description,
            from_version=r.from_version,
            to_version=r.to_version,
            files_changed=r.files_changed,
            simulation_risk=r.simulation_risk.value if hasattr(r.simulation_risk, "value") else str(r.simulation_risk),
            applied_at=r.applied_at.isoformat() if r.applied_at else "",
            rolled_back=r.rolled_back,
            rollback_reason=r.rollback_reason or "",
            formal_verification_status=r.formal_verification_status or "",
            repair_agent_used=r.repair_agent_used,
            repair_cost_usd=r.repair_cost_usd,
        ))

    return EvolutionHistoryResponse(records=items, total=len(items))


@router.get("/version", response_model=SimulaVersionResponse)
async def get_version(request: Request) -> SimulaVersionResponse:
    """Current config version number and version chain."""
    simula = _get_simula(request)
    try:
        version = await simula.get_current_version()
        chain_raw = await simula.get_version_chain()
    except Exception as exc:
        logger.warning("simula_version_error", error=str(exc))
        raise HTTPException(500, f"Failed to read version: {exc}") from exc

    chain = [
        VersionChainItem(
            version=v.version,
            timestamp=v.timestamp.isoformat() if v.timestamp else "",
            proposal_count=len(v.proposal_ids),
            config_hash=v.config_hash,
        )
        for v in chain_raw
    ]
    return SimulaVersionResponse(current_version=version, chain=chain)


@router.get("/inspector", response_model=InspectorStatsResponse)
async def get_inspector_stats(request: Request) -> InspectorStatsResponse:
    """Inspector (zero-day discovery) analytics and emitter health."""
    simula = _get_simula(request)

    inspector = getattr(simula, "_inspector", None)
    if inspector is None:
        return InspectorStatsResponse(enabled=False)

    try:
        raw_stats = inspector.stats
    except Exception as exc:
        logger.warning("inspector_stats_error", error=str(exc))
        raw_stats = {}

    analytics_emitter = getattr(simula, "_inspector_analytics", None)
    emitter_stats: dict[str, Any] = {}
    if analytics_emitter is not None:
        with contextlib.suppress(Exception):
            emitter_stats = analytics_emitter.stats

    analytics_view = getattr(
        getattr(simula, "_analytics", None), "_inspector_view", None
    )
    view_snapshot: dict[str, Any] = {}
    if analytics_view is not None:
        with contextlib.suppress(Exception):
            view_snapshot = analytics_view.snapshot()

    return InspectorStatsResponse(
        enabled=True,
        total_hunts=view_snapshot.get("total_hunts", raw_stats.get("total_hunts", 0)),
        total_vulnerabilities=view_snapshot.get("total_vulnerabilities", 0),
        critical_count=view_snapshot.get("critical_count", 0),
        high_count=view_snapshot.get("high_count", 0),
        medium_count=view_snapshot.get("medium_count", 0),
        low_count=view_snapshot.get("low_count", 0),
        avg_surfaces_per_hunt=view_snapshot.get("avg_surfaces_per_hunt", 0.0),
        avg_duration_ms=view_snapshot.get("avg_duration_ms", 0.0),
        analytics_emitter_active=analytics_emitter is not None,
        tsdb_persistence_active=emitter_stats.get("tsdb_persistence", False),
        recent_event_types=view_snapshot.get("recent_event_types", {}),
    )


@router.get("/inspector/hunts", response_model=HuntsResponse)
async def get_inspector_hunts(request: Request) -> HuntsResponse:
    """Recent completed hunt results held in the Inspector analytics view."""
    simula = _get_simula(request)

    analytics_view = getattr(
        getattr(simula, "_analytics", None), "_inspector_view", None
    )
    if analytics_view is None:
        # Try inspector directly
        inspector = getattr(simula, "_inspector", None)
        if inspector is None:
            return HuntsResponse(hunts=[], total=0)
        analytics_view = getattr(inspector, "_analytics_view", None)

    if analytics_view is None:
        return HuntsResponse(hunts=[], total=0)

    try:
        recent = analytics_view.recent_hunts() if hasattr(analytics_view, "recent_hunts") else []
    except Exception as exc:
        logger.warning("inspector_hunts_error", error=str(exc))
        recent = []

    hunts = []
    for hunt in recent:
        critical = sum(
            1 for v in getattr(hunt, "vulnerabilities_found", [])
            if getattr(v, "severity", "") in ("critical", "CRITICAL")
        )
        high = sum(
            1 for v in getattr(hunt, "vulnerabilities_found", [])
            if getattr(v, "severity", "") in ("high", "HIGH")
        )
        hunts.append(HuntSummaryItem(
            id=hunt.id,
            target_url=hunt.target_url,
            target_type=hunt.target_type.value if hasattr(hunt.target_type, "value") else str(hunt.target_type),
            surfaces_mapped=hunt.surfaces_mapped,
            vulnerabilities_found=len(getattr(hunt, "vulnerabilities_found", [])),
            critical_count=critical,
            high_count=high,
            total_duration_ms=hunt.total_duration_ms,
            started_at=hunt.started_at.isoformat() if hunt.started_at else "",
            completed_at=hunt.completed_at.isoformat() if hunt.completed_at else None,
        ))

    return HuntsResponse(hunts=hunts, total=len(hunts))


@router.post("/approve", response_model=ApproveGovernedResponse)
async def approve_governed_proposal(
    request: Request,
    body: ApproveGovernedRequest,
) -> ApproveGovernedResponse:
    """Approve a proposal that is awaiting governance."""
    simula = _get_simula(request)
    try:
        result = await simula.approve_governed_proposal(
            proposal_id=body.proposal_id,
            governance_record_id=body.governance_record_id,
        )
    except Exception as exc:
        logger.warning("simula_approve_error", error=str(exc))
        raise HTTPException(500, f"Approval failed: {exc}") from exc

    return ApproveGovernedResponse(
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        reason=result.reason or "",
        version=result.version,
    )


@router.get("/health", response_model=SimulaHealthResponse)
async def get_simula_health(request: Request) -> SimulaHealthResponse:
    """
    Full Simula health including self-healing component status.

    Returns proactive_scanner_alive, repair_memory_record_count,
    last_proposal_processed_at, and calibration_score (Task 6).
    """
    simula = _get_simula(request)
    try:
        raw = await simula.health()
    except Exception as exc:
        logger.warning("simula_health_error", error=str(exc))
        raise HTTPException(500, f"Health check failed: {exc}") from exc

    return SimulaHealthResponse(
        status=raw.get("status", "unhealthy"),
        current_version=raw.get("current_version", 0),
        active_proposals=raw.get("active_proposals", 0),
        proactive_scanner_alive=raw.get("proactive_scanner_alive", False),
        repair_memory_record_count=raw.get("repair_memory_record_count", 0),
        last_proposal_processed_at=raw.get("last_proposal_processed_at"),
        calibration_score=raw.get("calibration_score"),
        reason=raw.get("reason", ""),
    )


@router.get("/metrics", response_model=SimulaMetricsResponse)
async def get_simula_metrics(request: Request) -> SimulaMetricsResponse:
    """
    Full operational metrics: proposals today, success/rollback rates,
    proactive vs received ratio, calibration score (Task 6).
    """
    simula = _get_simula(request)
    try:
        raw = await simula.get_metrics()
    except Exception as exc:
        logger.warning("simula_metrics_error", error=str(exc))
        raise HTTPException(500, f"Failed to compute metrics: {exc}") from exc

    return SimulaMetricsResponse(
        proposals_received_session=raw.get("proposals_received_session", 0),
        proposals_approved_session=raw.get("proposals_approved_session", 0),
        proposals_rejected_session=raw.get("proposals_rejected_session", 0),
        proposals_rolled_back_session=raw.get("proposals_rolled_back_session", 0),
        success_rate=raw.get("success_rate", 0.0),
        rollback_rate=raw.get("rollback_rate", 0.0),
        proactive_proposals_generated=raw.get("proactive_proposals_generated", 0),
        proactive_vs_received_ratio=raw.get("proactive_vs_received_ratio", 0.0),
        calibration_score=raw.get("calibration_score"),
        repair_memory_record_count=raw.get("repair_memory_record_count", 0),
        last_proposal_processed_at=raw.get("last_proposal_processed_at"),
        proactive_scanner=raw.get("proactive_scanner", {}),
    )


@router.get("/repair-memory", response_model=RepairMemoryResponse)
async def get_repair_memory(request: Request) -> RepairMemoryResponse:
    """
    Repair outcome learning summary.

    Shows per-(category, system) success rates, overall rollback rate,
    most/least reliable change types, and calibration score (Task 2).
    """
    simula = _get_simula(request)
    try:
        raw = await simula.get_repair_memory_summary()
    except Exception as exc:
        logger.warning("simula_repair_memory_error", error=str(exc))
        raise HTTPException(500, f"Failed to read repair memory: {exc}") from exc

    return RepairMemoryResponse(
        success_rates_by_category=raw.get("success_rates_by_category", {}),
        total_proposals=raw.get("total_proposals", 0),
        rollback_rate=raw.get("rollback_rate", 0.0),
        most_reliable_change_type=raw.get("most_reliable_change_type", ""),
        most_risky_change_type=raw.get("most_risky_change_type", ""),
        calibration_score=raw.get("calibration_score", 1.0),
        calibration_window_size=raw.get("calibration_window_size", 0),
    )
