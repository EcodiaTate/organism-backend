"""
EcodiaOS - Thymos (Immune System) API Router

Exposes the Thymos immune system for observability and control:

  GET  /api/v1/thymos/health                - Full health snapshot: counters, budget, drive state
  GET  /api/v1/thymos/stats                 - Lightweight synchronous stats
  GET  /api/v1/thymos/incidents             - Incident ring buffer (most recent first)
  GET  /api/v1/thymos/incidents/{id}        - Full incident detail by ID
  GET  /api/v1/thymos/antibodies            - Antibody library (active + retired)
  GET  /api/v1/thymos/repairs               - Repair records derived from incident buffer
  GET  /api/v1/thymos/homeostasis           - Homeostasis & healing mode summary
  GET  /api/v1/thymos/homeostasis/metrics   - Per-metric values within optimal ranges
  GET  /api/v1/thymos/drive-state           - Constitutional drive pressure snapshot
  GET  /api/v1/thymos/config                - Active Thymos configuration parameters
  GET  /api/v1/thymos/prophylactic          - Prophylactic scan results & aggregate stats
  GET  /api/v1/thymos/causal-graph          - Causal dependency graph + recent chains
  GET  /api/v1/thymos/stream                - SSE real-time incident stream
  POST /api/v1/thymos/report                - Manually report an exception for processing
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from primitives.common import EOSBaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger("api.thymos")

router = APIRouter(prefix="/api/v1/thymos", tags=["thymos"])


# ─── Response Models ──────────────────────────────────────────────


class HealingBudgetOut(EOSBaseModel):
    repairs_this_hour: int = 0
    novel_repairs_today: int = 0
    max_repairs_per_hour: int = 5
    max_novel_repairs_per_day: int = 3
    active_diagnoses: int = 0
    max_concurrent_diagnoses: int = 3
    active_codegen: int = 0
    max_concurrent_codegen: int = 1
    storm_mode: bool = False
    storm_focus_system: str | None = None
    cpu_budget_fraction: float = 0.10


class DriveStateOut(EOSBaseModel):
    coherence: float = 0.0
    care: float = 0.0
    growth: float = 0.0
    honesty: float = 0.0
    composite_stress: float = 0.0
    most_stressed_drive: str = ""


class ThymosHealthOut(EOSBaseModel):
    status: str = "unknown"
    initialized: bool = False
    healing_mode: str = "unknown"
    # Incidents
    total_incidents: int = 0
    active_incidents: int = 0
    mean_resolution_ms: float = 0.0
    incidents_by_severity: dict[str, int] = {}
    incidents_by_class: dict[str, int] = {}
    # Antibodies
    total_antibodies: int = 0
    mean_antibody_effectiveness: float = 0.0
    antibodies_applied: int = 0
    antibodies_created: int = 0
    antibodies_retired: int = 0
    # Repairs
    repairs_attempted: int = 0
    repairs_succeeded: int = 0
    repairs_failed: int = 0
    repairs_rolled_back: int = 0
    repairs_by_tier: dict[str, int] = {}
    # Diagnosis
    total_diagnoses: int = 0
    mean_diagnosis_confidence: float = 0.0
    mean_diagnosis_latency_ms: float = 0.0
    # Homeostasis
    homeostatic_adjustments: int = 0
    metrics_in_range: int = 0
    metrics_total: int = 0
    # Storm
    storm_activations: int = 0
    # Prophylactic
    prophylactic_scans: int = 0
    prophylactic_warnings: int = 0
    # Derived - computed server-side so the UI doesn't duplicate logic
    immune_health_score: float = 100.0
    repair_success_rate: float = 0.0
    # Budget
    budget: HealingBudgetOut = HealingBudgetOut()
    # Drive state
    drive_state: DriveStateOut = DriveStateOut()


class IncidentOut(EOSBaseModel):
    id: str
    timestamp: str
    source_system: str
    incident_class: str
    severity: str
    fingerprint: str
    error_type: str
    error_message: str
    repair_status: str
    repair_tier: str | None = None
    repair_successful: bool | None = None
    resolution_time_ms: int | None = None
    root_cause: str | None = None
    antibody_id: str | None = None
    occurrence_count: int = 1
    blast_radius: float = 0.0
    affected_systems: list[str] = []
    user_visible: bool = False
    diagnostic_confidence: float = 0.0
    causal_chain: list[str] | None = None


class AntibodyOut(EOSBaseModel):
    id: str
    fingerprint: str
    source_system: str
    incident_class: str
    repair_tier: str
    effectiveness: float
    application_count: int
    success_count: int
    failure_count: int
    root_cause: str
    created_at: str
    last_applied: str | None = None
    retired: bool = False
    generation: int = 1
    parent_antibody_id: str | None = None
    error_pattern: str = ""


class RepairOut(EOSBaseModel):
    incident_id: str
    timestamp: str
    source_system: str
    repair_tier: str | None = None
    repair_status: str
    repair_successful: bool | None = None
    resolution_time_ms: int | None = None
    incident_class: str
    severity: str
    antibody_id: str | None = None
    fingerprint: str
    diagnostic_confidence: float = 0.0


class HomeostasisOut(EOSBaseModel):
    metrics_in_range: int = 0
    metrics_total: int = 0
    homeostatic_adjustments: int = 0
    healing_mode: str = "nominal"
    storm_activations: int = 0


class ThymosStatsOut(EOSBaseModel):
    initialized: bool = False
    total_incidents: int = 0
    active_incidents: int = 0
    total_diagnoses: int = 0
    total_repairs_attempted: int = 0
    total_repairs_succeeded: int = 0
    healing_mode: str = "unknown"
    drive_state: DriveStateOut = DriveStateOut()


class ThymosConfigOut(EOSBaseModel):
    sentinel_scan_interval_s: float = 30.0
    homeostasis_interval_s: float = 30.0
    post_repair_verify_timeout_s: float = 10.0
    max_concurrent_diagnoses: int = 3
    max_concurrent_codegen: int = 1
    storm_threshold: int = 10
    max_repairs_per_hour: int = 5
    max_novel_repairs_per_day: int = 3
    antibody_refinement_threshold: float = 0.6
    antibody_retirement_threshold: float = 0.3
    cpu_budget_fraction: float = 0.05
    burst_cpu_fraction: float = 0.15
    memory_budget_mb: int = 256


class ReportExceptionRequest(EOSBaseModel):
    error_type: str
    error_message: str
    stack_trace: str = ""
    source_system: str = "external"


class IncidentDetailOut(EOSBaseModel):
    id: str
    timestamp: str
    source_system: str
    incident_class: str
    severity: str
    fingerprint: str
    error_type: str
    error_message: str
    stack_trace: str = ""
    repair_status: str
    repair_tier: str | None = None
    repair_successful: bool | None = None
    resolution_time_ms: int | None = None
    root_cause: str | None = None
    antibody_id: str | None = None
    occurrence_count: int = 1
    first_seen: str | None = None
    blast_radius: float = 0.0
    affected_systems: list[str] = []
    user_visible: bool = False
    diagnostic_confidence: float = 0.0
    causal_chain: list[str] | None = None
    context: dict[str, Any] = {}
    constitutional_impact: dict[str, float] = {}
    root_cause_hypothesis: str | None = None


class ProphylacticWarningOut(EOSBaseModel):
    filepath: str
    antibody_id: str
    warning: str
    suggestion: str
    confidence: float


class ProphylacticOut(EOSBaseModel):
    total_scans: int = 0
    total_warnings: int = 0
    warning_rate: float = 0.0
    recent_warnings: list[ProphylacticWarningOut] = []


class HomeostasisMetricOut(EOSBaseModel):
    name: str
    optimal_min: float
    optimal_max: float
    unit: str
    current_value: float | None = None
    in_range: bool = True
    trend_direction: float = 0.0


class HomeostasisMetricsOut(EOSBaseModel):
    metrics: list[HomeostasisMetricOut] = []
    metrics_in_range: int = 0
    metrics_total: int = 0


class CausalNodeOut(EOSBaseModel):
    id: str
    incident_count: int = 0
    max_severity: str = "INFO"


class CausalEdgeOut(EOSBaseModel):
    source: str
    target: str
    weight: int = 1
    type: str = "dependency"


class CausalChainOut(EOSBaseModel):
    root_system: str
    chain: list[str]
    confidence: float


class CausalGraphOut(EOSBaseModel):
    nodes: list[CausalNodeOut] = []
    edges: list[CausalEdgeOut] = []
    recent_chains: list[CausalChainOut] = []


# ─── Helpers ──────────────────────────────────────────────────────


def _dt_str(dt: Any) -> str:
    if dt is None:
        return ""
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _drive_state_out(drive_state: Any) -> DriveStateOut:
    if drive_state is None:
        return DriveStateOut()
    raw = drive_state.as_dict() if hasattr(drive_state, "as_dict") else {}
    return DriveStateOut(
        coherence=float(raw.get("coherence", 0.0)),
        care=float(raw.get("care", 0.0)),
        growth=float(raw.get("growth", 0.0)),
        honesty=float(raw.get("honesty", 0.0)),
        composite_stress=float(
            drive_state.composite_stress()
            if hasattr(drive_state, "composite_stress")
            else 0.0
        ),
        most_stressed_drive=str(
            drive_state.most_stressed_drive()
            if hasattr(drive_state, "most_stressed_drive")
            else ""
        ),
    )


def _budget_out(budget_dict: dict[str, Any]) -> HealingBudgetOut:
    return HealingBudgetOut(
        repairs_this_hour=int(budget_dict.get("repairs_this_hour", 0)),
        novel_repairs_today=int(budget_dict.get("novel_repairs_today", 0)),
        max_repairs_per_hour=int(budget_dict.get("max_repairs_per_hour", 5)),
        max_novel_repairs_per_day=int(budget_dict.get("max_novel_repairs_per_day", 3)),
        active_diagnoses=int(budget_dict.get("active_diagnoses", 0)),
        max_concurrent_diagnoses=int(budget_dict.get("max_concurrent_diagnoses", 3)),
        active_codegen=int(budget_dict.get("active_codegen", 0)),
        max_concurrent_codegen=int(budget_dict.get("max_concurrent_codegen", 1)),
        storm_mode=bool(budget_dict.get("storm_mode", False)),
        storm_focus_system=budget_dict.get("storm_focus_system"),
        cpu_budget_fraction=float(budget_dict.get("cpu_budget_fraction", 0.10)),
    )


def _incident_to_out(incident: Any) -> IncidentOut:
    repair_tier = getattr(incident, "repair_tier", None)
    return IncidentOut(
        id=str(incident.id),
        timestamp=_dt_str(getattr(incident, "timestamp", None)),
        source_system=str(getattr(incident, "source_system", "")),
        incident_class=str(getattr(incident, "incident_class", "")),
        severity=str(getattr(incident, "severity", "")),
        fingerprint=str(getattr(incident, "fingerprint", "")),
        error_type=str(getattr(incident, "error_type", "")),
        error_message=str(getattr(incident, "error_message", ""))[:500],
        repair_status=str(getattr(incident, "repair_status", "")),
        repair_tier=str(repair_tier.name if repair_tier is not None else ""),
        repair_successful=getattr(incident, "repair_successful", None),
        resolution_time_ms=getattr(incident, "resolution_time_ms", None),
        root_cause=getattr(incident, "root_cause_hypothesis", None),
        antibody_id=getattr(incident, "antibody_id", None),
        occurrence_count=int(getattr(incident, "occurrence_count", 1)),
        blast_radius=float(getattr(incident, "blast_radius", 0.0)),
        affected_systems=list(getattr(incident, "affected_systems", []) or []),
        user_visible=bool(getattr(incident, "user_visible", False)),
        diagnostic_confidence=float(getattr(incident, "diagnostic_confidence", 0.0)),
        causal_chain=list(getattr(incident, "causal_chain", None) or []) or None,
    )


def _antibody_to_out(antibody: Any) -> AntibodyOut:
    repair_tier = getattr(antibody, "repair_tier", None)
    return AntibodyOut(
        id=str(antibody.id),
        fingerprint=str(getattr(antibody, "fingerprint", "")),
        source_system=str(getattr(antibody, "source_system", "")),
        incident_class=str(getattr(antibody, "incident_class", "")),
        repair_tier=str(repair_tier.name if repair_tier is not None else ""),
        effectiveness=float(getattr(antibody, "effectiveness", 0.0)),
        application_count=int(getattr(antibody, "application_count", 0)),
        success_count=int(getattr(antibody, "success_count", 0)),
        failure_count=int(getattr(antibody, "failure_count", 0)),
        root_cause=str(getattr(antibody, "root_cause_description", "")),
        created_at=_dt_str(getattr(antibody, "created_at", None)),
        last_applied=_dt_str(getattr(antibody, "last_applied", None)) or None,
        retired=bool(getattr(antibody, "retired", False)),
        generation=int(getattr(antibody, "generation", 1)),
        parent_antibody_id=getattr(antibody, "parent_antibody_id", None),
        error_pattern=str(getattr(antibody, "error_pattern", "")),
    )


# ─── Endpoints ────────────────────────────────────────────────────


@router.get("/health", response_model=ThymosHealthOut)
async def get_health(request: Request) -> ThymosHealthOut:
    """Full Thymos health snapshot: counters, budget, drive state."""
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return ThymosHealthOut(status="unavailable", initialized=False)
    try:
        h = await thymos.health()
        budget_dict: dict[str, Any] = h.get("budget", {}) or {}
        drive_raw = h.get("drive_state", {}) or {}
        drive_out = DriveStateOut(
            coherence=float(drive_raw.get("coherence", 0.0)),
            care=float(drive_raw.get("care", 0.0)),
            growth=float(drive_raw.get("growth", 0.0)),
            honesty=float(drive_raw.get("honesty", 0.0)),
        )
        _active_incidents = int(h.get("active_incidents", 0))
        _repairs_attempted = int(h.get("repairs_attempted", 0))
        _repairs_succeeded = int(h.get("repairs_succeeded", 0))
        _repairs_failed = int(h.get("repairs_failed", 0))
        _immune_health_score = max(
            0.0,
            min(
                100.0,
                100.0
                - (
                    _active_incidents * 2
                    + (_repairs_failed / max(1, _repairs_attempted)) * 50
                )
                / 2,
            ),
        )
        _repair_success_rate = (
            round((_repairs_succeeded / _repairs_attempted) * 100, 4)
            if _repairs_attempted > 0
            else 0.0
        )
        return ThymosHealthOut(
            status=str(h.get("status", "healthy")),
            initialized=bool(h.get("initialized", True)),
            healing_mode=str(h.get("healing_mode", "nominal")),
            total_incidents=int(h.get("total_incidents", 0)),
            active_incidents=int(h.get("active_incidents", 0)),
            mean_resolution_ms=float(h.get("mean_resolution_ms", 0.0)),
            incidents_by_severity=dict(h.get("incidents_by_severity", {})),
            incidents_by_class=dict(h.get("incidents_by_class", {})),
            total_antibodies=int(h.get("total_antibodies", 0)),
            mean_antibody_effectiveness=float(h.get("mean_antibody_effectiveness", 0.0)),
            antibodies_applied=int(h.get("antibodies_applied", 0)),
            antibodies_created=int(h.get("antibodies_created", 0)),
            antibodies_retired=int(h.get("antibodies_retired", 0)),
            repairs_attempted=int(h.get("repairs_attempted", 0)),
            repairs_succeeded=int(h.get("repairs_succeeded", 0)),
            repairs_failed=int(h.get("repairs_failed", 0)),
            repairs_rolled_back=int(h.get("repairs_rolled_back", 0)),
            repairs_by_tier=dict(h.get("repairs_by_tier", {})),
            total_diagnoses=int(h.get("total_diagnoses", 0)),
            mean_diagnosis_confidence=float(h.get("mean_diagnosis_confidence", 0.0)),
            mean_diagnosis_latency_ms=float(h.get("mean_diagnosis_latency_ms", 0.0)),
            homeostatic_adjustments=int(h.get("homeostatic_adjustments", 0)),
            metrics_in_range=int(h.get("metrics_in_range", 0)),
            metrics_total=int(h.get("metrics_total", 0)),
            storm_activations=int(h.get("storm_activations", 0)),
            prophylactic_scans=int(h.get("prophylactic_scans", 0)),
            prophylactic_warnings=int(h.get("prophylactic_warnings", 0)),
            immune_health_score=_immune_health_score,
            repair_success_rate=_repair_success_rate,
            budget=_budget_out(budget_dict),
            drive_state=drive_out,
        )
    except Exception as exc:
        logger.warning("thymos_health_error", error=str(exc))
        return ThymosHealthOut(status="degraded", initialized=True)


@router.get("/stats", response_model=ThymosStatsOut)
async def get_stats(request: Request) -> ThymosStatsOut:
    """Lightweight synchronous Thymos stats."""
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return ThymosStatsOut()
    try:
        s = thymos.stats
        drive_out = _drive_state_out(getattr(thymos, "_drive_state", None))
        return ThymosStatsOut(
            initialized=bool(s.get("initialized", False)),
            total_incidents=int(s.get("total_incidents", 0)),
            active_incidents=int(s.get("active_incidents", 0)),
            total_diagnoses=int(s.get("total_diagnoses", 0)),
            total_repairs_attempted=int(s.get("total_repairs_attempted", 0)),
            total_repairs_succeeded=int(s.get("total_repairs_succeeded", 0)),
            healing_mode=str(s.get("healing_mode", "unknown")),
            drive_state=drive_out,
        )
    except Exception as exc:
        logger.warning("thymos_stats_error", error=str(exc))
        return ThymosStatsOut()


@router.get("/incidents", response_model=list[IncidentOut])
async def get_incidents(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
) -> list[IncidentOut]:
    """
    Recent incidents from the in-memory ring buffer (most recent first).
    Limit default 50, max 500.
    """
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return []
    try:
        buffer = getattr(thymos, "_incident_buffer", None)
        if buffer is None:
            return []
        # deque is ordered oldest-first; reverse for most-recent-first
        incidents = list(reversed(list(buffer)))[:limit]
        return [_incident_to_out(i) for i in incidents]
    except Exception as exc:
        logger.warning("thymos_incidents_error", error=str(exc))
        return []


@router.get("/antibodies", response_model=list[AntibodyOut])
async def get_antibodies(request: Request) -> list[AntibodyOut]:
    """All antibodies (active and retired), sorted by effectiveness desc."""
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return []
    try:
        lib = getattr(thymos, "_antibody_library", None)
        if lib is None:
            return []
        all_antibodies = list(getattr(lib, "_all", {}).values())
        all_antibodies.sort(key=lambda a: a.effectiveness, reverse=True)
        return [_antibody_to_out(a) for a in all_antibodies]
    except Exception as exc:
        logger.warning("thymos_antibodies_error", error=str(exc))
        return []


@router.get("/repairs", response_model=list[RepairOut])
async def get_repairs(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
) -> list[RepairOut]:
    """
    Recent repair records from the incident buffer.
    Returns only incidents that have been through the repair pipeline
    (repair_status not PENDING/DIAGNOSING), most recent first.
    """
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return []
    try:
        buffer = getattr(thymos, "_incident_buffer", None)
        if buffer is None:
            return []
        all_incidents = list(reversed(list(buffer)))
        # Filter to incidents that have progressed past initial intake
        repair_statuses = {
            "resolved", "escalated", "accepted", "rolled_back",
            "verifying", "applying", "validating",
        }
        repaired = [
            i for i in all_incidents
            if str(getattr(i, "repair_status", "")).lower() in repair_statuses
        ][:limit]
        return [
            RepairOut(
                incident_id=str(i.id),
                timestamp=_dt_str(getattr(i, "timestamp", None)),
                source_system=str(getattr(i, "source_system", "")),
                repair_tier=(
                    str(i.repair_tier.name)
                    if getattr(i, "repair_tier", None) is not None
                    else None
                ),
                repair_status=str(getattr(i, "repair_status", "")),
                repair_successful=getattr(i, "repair_successful", None),
                resolution_time_ms=getattr(i, "resolution_time_ms", None),
                incident_class=str(getattr(i, "incident_class", "")),
                severity=str(getattr(i, "severity", "")),
                antibody_id=getattr(i, "antibody_id", None),
                fingerprint=str(getattr(i, "fingerprint", "")),
                diagnostic_confidence=float(getattr(i, "diagnostic_confidence", 0.0)),
            )
            for i in repaired
        ]
    except Exception as exc:
        logger.warning("thymos_repairs_error", error=str(exc))
        return []


@router.get("/homeostasis", response_model=HomeostasisOut)
async def get_homeostasis(request: Request) -> HomeostasisOut:
    """Homeostasis summary: metrics in range, adjustments made, healing mode."""
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return HomeostasisOut()
    try:
        h = await thymos.health()
        governor = getattr(thymos, "_governor", None)
        return HomeostasisOut(
            metrics_in_range=int(h.get("metrics_in_range", 0)),
            metrics_total=int(h.get("metrics_total", 0)),
            homeostatic_adjustments=int(h.get("homeostatic_adjustments", 0)),
            healing_mode=str(h.get("healing_mode", "nominal")),
            storm_activations=int(
                governor.storm_activations if governor else h.get("storm_activations", 0)
            ),
        )
    except Exception as exc:
        logger.warning("thymos_homeostasis_error", error=str(exc))
        return HomeostasisOut()


@router.get("/drive-state", response_model=DriveStateOut)
async def get_drive_state(request: Request) -> DriveStateOut:
    """
    Constitutional drive pressure from accumulated incidents and Equor rejections.
    Drives: coherence, care, growth, honesty - each 0.0 to 1.0.
    """
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return DriveStateOut()
    try:
        drive = getattr(thymos, "_drive_state", None)
        return _drive_state_out(drive)
    except Exception as exc:
        logger.warning("thymos_drive_state_error", error=str(exc))
        return DriveStateOut()


@router.get("/config", response_model=ThymosConfigOut)
async def get_config(request: Request) -> ThymosConfigOut:
    """Active Thymos configuration parameters and thresholds."""
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return ThymosConfigOut()
    try:
        cfg = getattr(thymos, "_config", None)
        if cfg is None:
            return ThymosConfigOut()
        getattr(thymos, "_governor", None)
        return ThymosConfigOut(
            sentinel_scan_interval_s=float(getattr(cfg, "sentinel_scan_interval_s", 30.0)),
            homeostasis_interval_s=float(getattr(cfg, "homeostasis_interval_s", 30.0)),
            post_repair_verify_timeout_s=float(getattr(cfg, "post_repair_verify_timeout_s", 10.0)),
            max_concurrent_diagnoses=int(getattr(cfg, "max_concurrent_diagnoses", 3)),
            max_concurrent_codegen=int(getattr(cfg, "max_concurrent_codegen", 1)),
            storm_threshold=int(getattr(cfg, "storm_threshold", 10)),
            max_repairs_per_hour=int(getattr(cfg, "max_repairs_per_hour", 5)),
            max_novel_repairs_per_day=int(getattr(cfg, "max_novel_repairs_per_day", 3)),
            antibody_refinement_threshold=float(getattr(cfg, "antibody_refinement_threshold", 0.6)),
            antibody_retirement_threshold=float(getattr(cfg, "antibody_retirement_threshold", 0.3)),
            cpu_budget_fraction=float(getattr(cfg, "cpu_budget_fraction", 0.05)),
            burst_cpu_fraction=float(getattr(cfg, "burst_cpu_fraction", 0.15)),
            memory_budget_mb=int(getattr(cfg, "memory_budget_mb", 256)),
        )
    except Exception as exc:
        logger.warning("thymos_config_error", error=str(exc))
        return ThymosConfigOut()


@router.post("/report", response_model=dict[str, str])
async def report_exception(
    request: Request, body: ReportExceptionRequest
) -> dict[str, str]:
    """
    Manually report an exception into the Thymos immune pipeline.
    Useful for external systems to trigger immune processing.
    """
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return {"status": "unavailable", "message": "Thymos not initialized"}
    try:
        await thymos.report_exception(
            error_type=body.error_type,
            error_message=body.error_message,
            stack_trace=body.stack_trace,
            source_system=body.source_system,
        )
        logger.info(
            "thymos_manual_report",
            error_type=body.error_type,
            source_system=body.source_system,
        )
        return {"status": "accepted", "message": "Exception submitted to immune pipeline"}
    except Exception as exc:
        logger.warning("thymos_report_error", error=str(exc))
        return {"status": "error", "message": str(exc)}


@router.get("/incidents/{incident_id}", response_model=IncidentDetailOut)
async def get_incident_detail(
    request: Request, incident_id: str
) -> IncidentDetailOut:
    """Full incident detail including stack trace, context, constitutional impact, and causal chain."""
    from fastapi import HTTPException

    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        raise HTTPException(status_code=503, detail="Thymos not initialized")
    try:
        incident = None

        # Search active incidents first (fast path)
        active = getattr(thymos, "_active_incidents", {})
        incident = active.get(incident_id)

        # Fall back to incident buffer
        if incident is None:
            buffer = getattr(thymos, "_incident_buffer", None)
            if buffer is not None:
                for inc in buffer:
                    if str(inc.id) == incident_id:
                        incident = inc
                        break

        if incident is None:
            raise HTTPException(status_code=404, detail="Incident not found")

        repair_tier = getattr(incident, "repair_tier", None)
        context_raw = getattr(incident, "context", None) or {}
        const_impact = getattr(incident, "constitutional_impact", None) or {}

        return IncidentDetailOut(
            id=str(incident.id),
            timestamp=_dt_str(getattr(incident, "timestamp", None)),
            source_system=str(getattr(incident, "source_system", "")),
            incident_class=str(getattr(incident, "incident_class", "")),
            severity=str(getattr(incident, "severity", "")),
            fingerprint=str(getattr(incident, "fingerprint", "")),
            error_type=str(getattr(incident, "error_type", "")),
            error_message=str(getattr(incident, "error_message", "")),
            stack_trace=str(getattr(incident, "stack_trace", "") or ""),
            repair_status=str(getattr(incident, "repair_status", "")),
            repair_tier=str(repair_tier.name if repair_tier is not None else ""),
            repair_successful=getattr(incident, "repair_successful", None),
            resolution_time_ms=getattr(incident, "resolution_time_ms", None),
            root_cause=getattr(incident, "root_cause_hypothesis", None),
            antibody_id=getattr(incident, "antibody_id", None),
            occurrence_count=int(getattr(incident, "occurrence_count", 1)),
            first_seen=_dt_str(getattr(incident, "first_seen", None)) or None,
            blast_radius=float(getattr(incident, "blast_radius", 0.0)),
            affected_systems=list(getattr(incident, "affected_systems", []) or []),
            user_visible=bool(getattr(incident, "user_visible", False)),
            diagnostic_confidence=float(getattr(incident, "diagnostic_confidence", 0.0)),
            causal_chain=list(getattr(incident, "causal_chain", None) or []) or None,
            context={str(k): v for k, v in context_raw.items()} if isinstance(context_raw, dict) else {},
            constitutional_impact={
                str(k): float(v)
                for k, v in const_impact.items()
                if isinstance(v, (int, float))
            } if isinstance(const_impact, dict) else {},
            root_cause_hypothesis=getattr(incident, "root_cause_hypothesis", None),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("thymos_incident_detail_error", error=str(exc), incident_id=incident_id)
        from fastapi import HTTPException as HTTPException2
        raise HTTPException2(status_code=500, detail=str(exc))


@router.get("/prophylactic", response_model=ProphylacticOut)
async def get_prophylactic(request: Request) -> ProphylacticOut:
    """Prophylactic scanner results: recent warnings and aggregate stats."""
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return ProphylacticOut()
    try:
        scanner = getattr(thymos, "_prophylactic_scanner", None)
        total_scans = int(getattr(thymos, "_total_prophylactic_scans", 0))
        total_warnings = int(getattr(thymos, "_total_prophylactic_warnings", 0))
        warning_rate = total_warnings / total_scans if total_scans > 0 else 0.0

        recent_warnings: list[ProphylacticWarningOut] = []
        if scanner is not None:
            scanner_scans = int(getattr(scanner, "_scans_run", 0))
            scanner_warnings = int(getattr(scanner, "_warnings_issued", 0))
            if scanner_scans > 0:
                total_scans = scanner_scans
                total_warnings = scanner_warnings
                warning_rate = scanner_warnings / scanner_scans

            warnings_buffer = getattr(scanner, "_recent_warnings", None)
            if warnings_buffer is not None:
                for w in list(warnings_buffer)[-50:]:
                    recent_warnings.append(ProphylacticWarningOut(
                        filepath=str(getattr(w, "filepath", "")),
                        antibody_id=str(getattr(w, "antibody_id", "")),
                        warning=str(getattr(w, "warning", "")),
                        suggestion=str(getattr(w, "suggestion", "")),
                        confidence=float(getattr(w, "confidence", 0.0)),
                    ))

        return ProphylacticOut(
            total_scans=total_scans,
            total_warnings=total_warnings,
            warning_rate=warning_rate,
            recent_warnings=recent_warnings,
        )
    except Exception as exc:
        logger.warning("thymos_prophylactic_error", error=str(exc))
        return ProphylacticOut()


@router.get("/homeostasis/metrics", response_model=HomeostasisMetricsOut)
async def get_homeostasis_metrics(request: Request) -> HomeostasisMetricsOut:
    """Per-metric homeostasis values with current readings and optimal ranges."""
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return HomeostasisMetricsOut()
    try:
        controller = getattr(thymos, "_homeostasis_controller", None)
        if controller is None:
            return HomeostasisMetricsOut()

        ranges: dict[str, tuple[float, float, str]] = getattr(controller, "_ranges", {})
        history: dict[str, list[float]] = getattr(controller, "_history", {})

        metrics: list[HomeostasisMetricOut] = []
        in_range_count = 0

        for name, range_tuple in ranges.items():
            opt_min = float(range_tuple[0])
            opt_max = float(range_tuple[1])
            unit = str(range_tuple[2]) if len(range_tuple) > 2 else ""

            values = history.get(name, [])
            current_value: float | None = values[-1] if values else None
            in_range = (
                opt_min <= current_value <= opt_max
                if current_value is not None
                else True
            )

            trend = 0.0
            if len(values) >= 2:
                window = values[-min(20, len(values)):]
                n = len(window)
                if n >= 2:
                    mean_x = (n - 1) / 2.0
                    mean_y = sum(window) / n
                    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(window))
                    den = sum((i - mean_x) ** 2 for i in range(n))
                    trend = num / den if den != 0 else 0.0

            if in_range:
                in_range_count += 1

            metrics.append(HomeostasisMetricOut(
                name=name,
                optimal_min=opt_min,
                optimal_max=opt_max,
                unit=unit,
                current_value=current_value,
                in_range=in_range,
                trend_direction=trend,
            ))

        return HomeostasisMetricsOut(
            metrics=metrics,
            metrics_in_range=in_range_count,
            metrics_total=len(metrics),
        )
    except Exception as exc:
        logger.warning("thymos_homeostasis_metrics_error", error=str(exc))
        return HomeostasisMetricsOut()


@router.get("/causal-graph", response_model=CausalGraphOut)
async def get_causal_graph(request: Request) -> CausalGraphOut:
    """
    Causal dependency graph from the static dependency map and recent incident data.
    Returns nodes (systems with incident stats), edges (dependencies), and recent causal chains.
    """
    thymos = getattr(request.app.state, "thymos", None)
    if thymos is None:
        return CausalGraphOut()
    try:
        from systems.thymos.diagnosis import _UPSTREAM_DEPS

        buffer = getattr(thymos, "_incident_buffer", None)
        incidents = list(buffer) if buffer is not None else []

        _severity_rank: dict[str, int] = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}
        _rank_severity: dict[int, str] = {v: k for k, v in _severity_rank.items()}

        node_counts: dict[str, int] = {}
        node_max_severity: dict[str, int] = {}

        for inc in incidents:
            sys_id = str(getattr(inc, "source_system", ""))
            if not sys_id:
                continue
            node_counts[sys_id] = node_counts.get(sys_id, 0) + 1
            sev_str = str(getattr(inc, "severity", "INFO"))
            sev_rank = _severity_rank.get(sev_str, 1)
            node_max_severity[sys_id] = max(node_max_severity.get(sys_id, 0), sev_rank)

        for sys_id in _UPSTREAM_DEPS:
            if sys_id not in node_counts:
                node_counts[sys_id] = 0
                node_max_severity[sys_id] = 0

        nodes = [
            CausalNodeOut(
                id=sys_id,
                incident_count=node_counts.get(sys_id, 0),
                max_severity=_rank_severity.get(node_max_severity.get(sys_id, 0), "INFO"),
            )
            for sys_id in node_counts
        ]

        edges: list[CausalEdgeOut] = []
        seen_edges: set[tuple[str, str]] = set()
        for downstream, upstreams in _UPSTREAM_DEPS.items():
            for upstream in upstreams:
                key = (upstream, downstream)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(CausalEdgeOut(
                        source=upstream,
                        target=downstream,
                        weight=1,
                        type="dependency",
                    ))

        causal_analyzer = getattr(thymos, "_causal_analyzer", None)
        recent_incidents_by_sys: dict[str, list[Any]] = (
            getattr(causal_analyzer, "_recent_incidents", {}) if causal_analyzer else {}
        )

        recent_chains: list[CausalChainOut] = []
        for sys_id, sys_incidents in list(recent_incidents_by_sys.items())[:10]:
            if not sys_incidents:
                continue
            upstreams_of_sys = _UPSTREAM_DEPS.get(sys_id, [])
            chain = ([upstreams_of_sys[0]] if upstreams_of_sys else []) + [sys_id]
            root = chain[0]
            recent_chains.append(CausalChainOut(
                root_system=root,
                chain=chain,
                confidence=0.6,
            ))

        return CausalGraphOut(nodes=nodes, edges=edges, recent_chains=recent_chains[:10])
    except Exception as exc:
        logger.warning("thymos_causal_graph_error", error=str(exc))
        return CausalGraphOut()


async def _incident_to_sse(incident: Any) -> str:
    """Serialize a minimal incident snapshot to an SSE data line."""
    repair_tier = getattr(incident, "repair_tier", None)
    payload = {
        "id": str(incident.id),
        "timestamp": _dt_str(getattr(incident, "timestamp", None)),
        "source_system": str(getattr(incident, "source_system", "")),
        "incident_class": str(getattr(incident, "incident_class", "")),
        "severity": str(getattr(incident, "severity", "")),
        "error_type": str(getattr(incident, "error_type", "")),
        "repair_status": str(getattr(incident, "repair_status", "")),
        "repair_tier": str(repair_tier.name if repair_tier is not None else ""),
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.get("/stream")
async def stream_incidents(request: Request) -> StreamingResponse:
    """
    Server-Sent Events real-time incident stream.
    Emits each incident as it enters the immune pipeline.
    Keepalive comment sent every 15 seconds.
    """
    thymos = getattr(request.app.state, "thymos", None)

    async def event_generator() -> AsyncGenerator[str, None]:
        if thymos is None:
            yield ": thymos unavailable\n\n"
            return

        stream_queues: list[asyncio.Queue[Any]] = getattr(thymos, "_stream_queues", [])
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=100)
        stream_queues.append(queue)

        try:
            while True:
                try:
                    incident = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield await _incident_to_sse(incident)
                except TimeoutError:
                    yield ": keepalive\n\n"
                except Exception:
                    break
        finally:
            with contextlib.suppress(ValueError):
                stream_queues.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
