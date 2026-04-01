"""
EcodiaOS - EIS (Epistemic Immune System) API Router

Endpoints:
  GET  /api/v1/eis/health                - health + zone config merged
  GET  /api/v1/eis/stats                 - live counter snapshot
  GET  /api/v1/eis/threat-library        - threat library stats
  GET  /api/v1/eis/anomalies             - recent behavioral anomalies
  GET  /api/v1/eis/anomalies/stats       - anomaly detector statistics
  GET  /api/v1/eis/quarantine-gate       - quarantine gate verdict statistics
  GET  /api/v1/eis/taint                 - taint engine stats
  GET  /api/v1/eis/config                - all thresholds and weights
  GET  /api/v1/eis/innate-checks         - catalog of all innate check definitions
  GET  /api/v1/eis/pathogens             - browse pathogen store
  GET  /api/v1/eis/pathogens/stats       - vector store collection stats
  POST /api/v1/eis/config/weights        - update composite score weights
  POST /api/v1/eis/config/thresholds     - update quarantine/block thresholds
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Query, Request
from pydantic import Field, field_validator

from primitives.common import EOSBaseModel
from systems.eis.config import (
    BELIEF_FLOOR,
    RISK_SALIENCE_GAIN,
    SIGMOID_MIDPOINT,
    SIGMOID_STEEPNESS,
    THRESHOLDS,
)
from systems.eis.models import InnateCheckID, ThreatClass, ThreatSeverity

logger = structlog.get_logger("api.eis")

router = APIRouter(prefix="/api/v1/eis", tags=["eis"])


# ─── Response Models ──────────────────────────────────────────────


class ZoneBoundsResponse(EOSBaseModel):
    lower: float
    upper: float


class EISHealthResponse(EOSBaseModel):
    """
    EIS health snapshot merged with threshold constants.

    Includes the zone map so the frontend can render composite-score
    visualisations without a separate /config call.
    """

    system: str
    status: str
    counters: dict[str, int]
    config: dict[str, Any]
    # Zone constants for frontend visualisation
    zones: dict[str, ZoneBoundsResponse]
    sigmoid_midpoint: float = Field(
        ..., description="Composite score at which belief discount = 0.5"
    )
    sigmoid_steepness: float = Field(
        ..., description="Controls sharpness of the trust→distrust sigmoid cliff"
    )
    belief_floor: float = Field(
        ..., description="Minimum belief-update weight (even for confirmed attacks)"
    )
    risk_salience_gain: float = Field(
        ..., description="Linear gain applied to EIS score before clamping in risk dimension"
    )


class EISStatsResponse(EOSBaseModel):
    """
    Live EIS counter snapshot.

    pass_rate and block_rate are derived ratios, always 0.0 when screened == 0.
    """

    screened: int
    passed: int
    elevated: int
    quarantined: int
    blocked: int
    pass_rate: float
    block_rate: float


# ─── Routes ──────────────────────────────────────────────────────


@router.get("/health", response_model=EISHealthResponse)
async def eis_health(request: Request) -> EISHealthResponse:
    """
    Return EIS service health merged with zone-boundary constants.

    The ``zones`` field lets the frontend render composite-score zone bars
    (clean / elevated / antigenic_zone / known_attack) without a
    separate config call.  Sigmoid and risk-salience parameters are also
    included for visualisation.
    """
    eis = request.app.state.eis
    raw: dict[str, Any] = await eis.health()

    zones = {
        label: ZoneBoundsResponse(lower=bounds.lower, upper=bounds.upper)
        for label, bounds in THRESHOLDS.items()
    }

    return EISHealthResponse(
        system=raw["system"],
        status=raw["status"],
        counters=raw["counters"],
        config=raw["config"],
        zones=zones,
        sigmoid_midpoint=SIGMOID_MIDPOINT,
        sigmoid_steepness=SIGMOID_STEEPNESS,
        belief_floor=BELIEF_FLOOR,
        risk_salience_gain=RISK_SALIENCE_GAIN,
    )


@router.get("/stats", response_model=EISStatsResponse)
async def eis_stats(request: Request) -> EISStatsResponse:
    """
    Return live EIS counter snapshot.

    Reads the in-process counters directly - no I/O, always fast.
    ``pass_rate`` and ``block_rate`` are 0.0 when ``screened`` is 0.
    """
    eis = request.app.state.eis
    raw: dict[str, Any] = await eis.health()
    counters: dict[str, int] = raw["counters"]

    screened: int = counters.get("screened", 0)
    passed: int = counters.get("passed", 0)
    elevated: int = counters.get("elevated", 0)
    quarantined: int = counters.get("quarantined", 0)
    blocked: int = counters.get("blocked", 0)

    pass_rate = passed / screened if screened > 0 else 0.0
    block_rate = blocked / screened if screened > 0 else 0.0

    return EISStatsResponse(
        screened=screened,
        passed=passed,
        elevated=elevated,
        quarantined=quarantined,
        blocked=blocked,
        pass_rate=round(pass_rate, 4),
        block_rate=round(block_rate, 4),
    )


# ─── New Observability Endpoints ──────────────────────────────────


class ThreatLibraryStatsResponse(EOSBaseModel):
    """Threat library pattern memory statistics."""

    total_patterns: int
    by_status: dict[str, int]
    by_category: dict[str, int]
    total_scans: int
    total_matches: int
    total_learned: int


class AnomalyResponse(EOSBaseModel):
    """A single detected behavioral anomaly."""

    id: str
    timestamp: str
    anomaly_type: str
    severity: str
    description: str
    observed_value: float
    baseline_value: float
    deviation_sigma: float
    event_types_involved: list[str]
    recommended_action: str


class AnomalyStatsResponse(EOSBaseModel):
    """Anomaly detector baseline statistics."""

    total_observations: int
    total_anomalies: int
    anomalies_by_type: dict[str, int]
    tracked_event_types: int
    baseline_event_types: int
    drive_observations: int


class QuarantineGateStatsResponse(EOSBaseModel):
    """Quarantine gate verdict counts."""

    total_evaluations: int
    mutations_evaluated: int
    knowledge_evaluated: int
    verdicts: dict[str, int]


class TaintStatsResponse(EOSBaseModel):
    """Taint engine analysis statistics."""

    calls: int
    critical_flags: int
    constitutional_paths: int


class InnateCheckConfigResponse(EOSBaseModel):
    """Single innate check description."""

    id: str
    description: str
    severity: str


class EISConfigResponse(EOSBaseModel):
    """All EIS threshold and weight configuration."""

    quarantine_threshold: float
    block_threshold: float
    innate_weight: float
    structural_weight: float
    histogram_weight: float
    semantic_weight: float
    sigmoid_midpoint: float
    sigmoid_steepness: float
    belief_floor: float
    risk_salience_gain: float
    zones: dict[str, ZoneBoundsResponse]
    soma_quarantine_offset: float
    innate_enabled: bool
    similarity_enabled: bool


@router.get("/threat-library", response_model=ThreatLibraryStatsResponse)
async def eis_threat_library(request: Request) -> ThreatLibraryStatsResponse:
    """Return threat library pattern memory statistics."""
    eis = request.app.state.eis
    raw: dict[str, Any] = await eis.health()
    lib: dict[str, Any] = raw.get("threat_library", {})

    return ThreatLibraryStatsResponse(
        total_patterns=lib.get("total_patterns", 0),
        by_status=lib.get("by_status", {}),
        by_category=lib.get("by_category", {}),
        total_scans=lib.get("total_scans", 0),
        total_matches=lib.get("total_matches", 0),
        total_learned=lib.get("total_learned", 0),
    )


@router.get("/anomalies", response_model=list[AnomalyResponse])
async def eis_anomalies(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[AnomalyResponse]:
    """Return recent behavioral anomalies (most recent first)."""
    eis = request.app.state.eis
    anomalies = eis._anomaly_detector.recent_anomalies(limit=limit)

    return [
        AnomalyResponse(
            id=a.id,
            timestamp=a.timestamp.isoformat() if isinstance(a.timestamp, datetime) else str(a.timestamp),
            anomaly_type=a.anomaly_type.value if hasattr(a.anomaly_type, "value") else str(a.anomaly_type),
            severity=a.severity.value if hasattr(a.severity, "value") else str(a.severity),
            description=a.description,
            observed_value=a.observed_value,
            baseline_value=a.baseline_value,
            deviation_sigma=a.deviation_sigma,
            event_types_involved=list(a.event_types_involved),
            recommended_action=a.recommended_action,
        )
        for a in anomalies
    ]


@router.get("/anomalies/stats", response_model=AnomalyStatsResponse)
async def eis_anomaly_stats(request: Request) -> AnomalyStatsResponse:
    """Return anomaly detector baseline statistics."""
    eis = request.app.state.eis
    raw: dict[str, Any] = await eis.health()
    stats: dict[str, Any] = raw.get("anomaly_detector", {})

    return AnomalyStatsResponse(
        total_observations=stats.get("total_observations", 0),
        total_anomalies=stats.get("total_anomalies", 0),
        anomalies_by_type=stats.get("anomalies_by_type", {}),
        tracked_event_types=stats.get("tracked_event_types", 0),
        baseline_event_types=stats.get("baseline_event_types", 0),
        drive_observations=stats.get("drive_observations", 0),
    )


@router.get("/quarantine-gate", response_model=QuarantineGateStatsResponse)
async def eis_quarantine_gate(request: Request) -> QuarantineGateStatsResponse:
    """Return quarantine gate verdict statistics."""
    eis = request.app.state.eis
    raw: dict[str, Any] = await eis.health()
    gate: dict[str, Any] = raw.get("quarantine_gate", {})

    return QuarantineGateStatsResponse(
        total_evaluations=gate.get("total_evaluations", 0),
        mutations_evaluated=gate.get("mutations_evaluated", 0),
        knowledge_evaluated=gate.get("knowledge_evaluated", 0),
        verdicts=gate.get("verdicts", {}),
    )


@router.get("/taint", response_model=TaintStatsResponse)
async def eis_taint(request: Request) -> TaintStatsResponse:
    """Return taint engine analysis statistics."""
    eis = request.app.state.eis
    raw: dict[str, Any] = await eis.health()
    taint: dict[str, Any] = raw.get("taint", {})

    return TaintStatsResponse(
        calls=taint.get("calls", 0),
        critical_flags=taint.get("critical_flags", 0),
        constitutional_paths=taint.get("constitutional_paths", 0),
    )


@router.get("/config", response_model=EISConfigResponse)
async def eis_config(request: Request) -> EISConfigResponse:
    """Return all EIS threshold and weight configuration."""
    eis = request.app.state.eis
    raw: dict[str, Any] = await eis.health()
    cfg: dict[str, Any] = raw.get("config", {})

    zones = {
        label: ZoneBoundsResponse(lower=bounds.lower, upper=bounds.upper)
        for label, bounds in THRESHOLDS.items()
    }

    # Read composite weights from the EIS config object directly
    eis_config = eis._config

    return EISConfigResponse(
        quarantine_threshold=cfg.get("quarantine_threshold", eis_config.quarantine_threshold),
        block_threshold=cfg.get("block_threshold", eis_config.block_threshold),
        innate_weight=getattr(eis_config, "innate_weight", 0.40),
        structural_weight=getattr(eis_config, "structural_weight", 0.15),
        histogram_weight=getattr(eis_config, "histogram_weight", 0.10),
        semantic_weight=getattr(eis_config, "semantic_weight", 0.35),
        sigmoid_midpoint=SIGMOID_MIDPOINT,
        sigmoid_steepness=SIGMOID_STEEPNESS,
        belief_floor=BELIEF_FLOOR,
        risk_salience_gain=RISK_SALIENCE_GAIN,
        zones=zones,
        soma_quarantine_offset=getattr(eis, "_soma_quarantine_offset", 0.0),
        innate_enabled=cfg.get("innate_enabled", True),
        similarity_enabled=cfg.get("similarity_enabled", True),
    )


# ─── Innate Checks Catalog ────────────────────────────────────────


class InnateCheckDetail(EOSBaseModel):
    """Static definition of a single innate check."""

    id: str
    description: str
    severity: str
    threat_class: str
    match_count: int = Field(0, description="Times this check matched (session counter)")


# Static catalog - mirrors the checks defined in innate.py
_INNATE_CATALOG: list[dict[str, str]] = [
    {
        "id": InnateCheckID.SYSTEM_PROMPT_LEAK,
        "description": "Detects requests to reveal, repeat, or dump system prompt / instructions.",
        "severity": ThreatSeverity.HIGH,
        "threat_class": ThreatClass.DATA_EXFILTRATION,
    },
    {
        "id": InnateCheckID.ROLE_OVERRIDE,
        "description": "Detects chat-template delimiter abuse to inject a fake system or assistant role.",
        "severity": ThreatSeverity.CRITICAL,
        "threat_class": ThreatClass.PROMPT_INJECTION,
    },
    {
        "id": InnateCheckID.INSTRUCTION_INJECTION,
        "description": "Detects BEGIN/START NEW INSTRUCTIONS patterns and heavy delimiter lines.",
        "severity": ThreatSeverity.HIGH,
        "threat_class": ThreatClass.PROMPT_INJECTION,
    },
    {
        "id": InnateCheckID.ENCODING_EVASION,
        "description": "Detects attempts to smuggle instructions via base64, hex, rot13, or long encoded blobs.",
        "severity": ThreatSeverity.MEDIUM,
        "threat_class": ThreatClass.PROMPT_INJECTION,
    },
    {
        "id": InnateCheckID.DELIMITER_ABUSE,
        "description": "Detects code-fence exploitation, HTML injection tags, and template injection {{ }}.",
        "severity": ThreatSeverity.MEDIUM,
        "threat_class": ThreatClass.PROMPT_INJECTION,
    },
    {
        "id": InnateCheckID.REPETITION_ATTACK,
        "description": "Detects context-window flooding via a single word/phrase repeated 20+ times.",
        "severity": ThreatSeverity.MEDIUM,
        "threat_class": ThreatClass.CONTEXT_POISONING,
    },
    {
        "id": InnateCheckID.CONTEXT_WINDOW_STUFFING,
        "description": "Flags inputs over 50K characters as potential context-window stuffing.",
        "severity": ThreatSeverity.LOW,
        "threat_class": ThreatClass.CONTEXT_POISONING,
    },
    {
        "id": InnateCheckID.DATA_EXFIL_PATTERN,
        "description": "Detects HTTP/curl/fetch calls that could exfiltrate data to external URLs.",
        "severity": ThreatSeverity.HIGH,
        "threat_class": ThreatClass.DATA_EXFILTRATION,
    },
    {
        "id": InnateCheckID.IDENTITY_SPOOF,
        "description": "Detects impersonation of system, assistant, Claude, or admin identities.",
        "severity": ThreatSeverity.MEDIUM,
        "threat_class": ThreatClass.IDENTITY_SPOOFING,
    },
    {
        "id": InnateCheckID.JAILBREAK_PHRASE,
        "description": "Detects DAN, jailbreak keywords, and requests to act as evil/unfiltered persona.",
        "severity": ThreatSeverity.HIGH,
        "threat_class": ThreatClass.JAILBREAK,
    },
    {
        "id": InnateCheckID.UNICODE_SMUGGLING,
        "description": "Detects bidirectional override characters and zero-width character clusters.",
        "severity": ThreatSeverity.MEDIUM,
        "threat_class": ThreatClass.PROMPT_INJECTION,
    },
    {
        "id": InnateCheckID.INVISIBLE_CHARS,
        "description": "Detects high density of invisible/control characters (>5 per 2K chars).",
        "severity": ThreatSeverity.MEDIUM,
        "threat_class": ThreatClass.PROMPT_INJECTION,
    },
]


@router.get("/innate-checks", response_model=list[InnateCheckDetail])
async def eis_innate_checks(request: Request) -> list[InnateCheckDetail]:
    """
    Return the catalog of all innate (fast-path) check definitions.

    Each entry includes the check id, description, severity, threat_class,
    and a live match_count sourced from the EIS session counters if available.
    """
    eis = request.app.state.eis
    raw: dict[str, Any] = await eis.health()
    # The health payload may include per-check match counts under "innate_counters"
    innate_counters: dict[str, int] = raw.get("innate_counters", {})

    return [
        InnateCheckDetail(
            id=entry["id"],
            description=entry["description"],
            severity=entry["severity"],
            threat_class=entry["threat_class"],
            match_count=innate_counters.get(entry["id"], 0),
        )
        for entry in _INNATE_CATALOG
    ]


# ─── Pathogen Store Browser ───────────────────────────────────────


class PathogenRecord(EOSBaseModel):
    """A known pathogen entry from the vector store."""

    id: str
    threat_class: str
    severity: str
    description: str
    canonical_text: str
    tags: list[str]
    match_count: int
    retired: bool
    created_at: str


class PathogenListResponse(EOSBaseModel):
    """Paginated list of known pathogens."""

    pathogens: list[PathogenRecord]
    total: int
    available: bool


class PathogenStoreStatsResponse(EOSBaseModel):
    """Qdrant vector store collection statistics."""

    available: bool
    collection: str
    points_count: int
    indexed_vectors_count: int
    status: str


@router.get("/pathogens/stats", response_model=PathogenStoreStatsResponse)
async def eis_pathogen_stats(request: Request) -> PathogenStoreStatsResponse:
    """Return Qdrant vector store collection statistics."""
    eis = request.app.state.eis
    pathogen_store = getattr(eis, "_pathogen_store", None)
    if pathogen_store is None:
        return PathogenStoreStatsResponse(
            available=False,
            collection="eis_pathogens",
            points_count=0,
            indexed_vectors_count=0,
            status="unavailable",
        )

    info: dict[str, Any] = await pathogen_store.get_collection_info()
    return PathogenStoreStatsResponse(
        available=info.get("available", False),
        collection=info.get("collection", "eis_pathogens"),
        points_count=info.get("points_count", 0),
        indexed_vectors_count=info.get("indexed_vectors_count", 0),
        status=info.get("status", "unknown"),
    )


@router.get("/pathogens", response_model=PathogenListResponse)
async def eis_pathogens(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    threat_class: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    retired: bool | None = Query(default=None),
) -> PathogenListResponse:
    """
    Browse the pathogen store.

    Optionally filter by threat_class, severity, or retired status.
    Delegates to Qdrant scroll (no vector required - payload-filter only).
    """
    eis = request.app.state.eis
    pathogen_store = getattr(eis, "_pathogen_store", None)
    if pathogen_store is None or pathogen_store._client is None:
        return PathogenListResponse(pathogens=[], total=0, available=False)

    try:
        from qdrant_client import models as qmodels  # type: ignore[import-untyped]

        must_conditions = []

        if threat_class is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="threat_class",
                    match=qmodels.MatchValue(value=threat_class),
                )
            )
        if severity is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="severity",
                    match=qmodels.MatchValue(value=severity),
                )
            )
        if retired is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="retired",
                    match=qmodels.MatchValue(value=retired),
                )
            )

        scroll_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

        results, _next_offset = await pathogen_store._client.scroll(
            collection_name=pathogen_store._config.qdrant_collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        pathogens = [
            PathogenRecord(
                id=str(point.id),
                threat_class=point.payload.get("threat_class", "benign") if point.payload else "benign",
                severity=point.payload.get("severity", "none") if point.payload else "none",
                description=point.payload.get("description", "") if point.payload else "",
                canonical_text=point.payload.get("canonical_text", "") if point.payload else "",
                tags=point.payload.get("tags", []) if point.payload else [],
                match_count=point.payload.get("match_count", 0) if point.payload else 0,
                retired=point.payload.get("retired", False) if point.payload else False,
                created_at=point.payload.get("created_at", "") if point.payload else "",
            )
            for point in results
        ]

        return PathogenListResponse(pathogens=pathogens, total=len(pathogens), available=True)

    except Exception as exc:
        logger.warning("eis_pathogen_scroll_failed", error=str(exc))
        return PathogenListResponse(pathogens=[], total=0, available=False)


# ─── Threshold / Weight Tuning ────────────────────────────────────


class WeightsUpdateRequest(EOSBaseModel):
    """Update composite score weights. Must sum to 1.0 (±0.01 tolerance)."""

    innate_weight: float = Field(..., ge=0.0, le=1.0)
    structural_weight: float = Field(..., ge=0.0, le=1.0)
    histogram_weight: float = Field(..., ge=0.0, le=1.0)
    semantic_weight: float = Field(..., ge=0.0, le=1.0)

    @field_validator("semantic_weight")
    @classmethod
    def weights_must_sum_to_one(cls, v: float, info: Any) -> float:
        data = info.data
        total = (
            data.get("innate_weight", 0.0)
            + data.get("structural_weight", 0.0)
            + data.get("histogram_weight", 0.0)
            + v
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0 (got {total:.4f})")
        return v


class ThresholdsUpdateRequest(EOSBaseModel):
    """Update quarantine and block thresholds."""

    quarantine_threshold: float = Field(..., ge=0.0, le=1.0)
    block_threshold: float = Field(..., ge=0.0, le=1.0)

    @field_validator("block_threshold")
    @classmethod
    def block_must_exceed_quarantine(cls, v: float, info: Any) -> float:
        qt = info.data.get("quarantine_threshold", 0.0)
        if v <= qt:
            raise ValueError(
                f"block_threshold ({v}) must be greater than quarantine_threshold ({qt})"
            )
        return v


class ConfigUpdateResponse(EOSBaseModel):
    """Result of a config update operation."""

    updated: bool
    message: str


@router.post("/config/weights", response_model=ConfigUpdateResponse)
async def eis_update_weights(
    body: WeightsUpdateRequest,
    request: Request,
) -> ConfigUpdateResponse:
    """
    Update composite score weights (innate / structural / histogram / semantic).

    Validates that all four weights sum to 1.0 (±0.01 tolerance) before
    applying. Modifies ``eis._config`` in place - effective immediately.
    """
    eis = request.app.state.eis
    cfg = eis._config
    cfg.innate_weight = body.innate_weight
    cfg.structural_weight = body.structural_weight
    cfg.histogram_weight = body.histogram_weight
    cfg.semantic_weight = body.semantic_weight

    logger.info(
        "eis_weights_updated",
        innate=body.innate_weight,
        structural=body.structural_weight,
        histogram=body.histogram_weight,
        semantic=body.semantic_weight,
    )

    return ConfigUpdateResponse(updated=True, message="Weights updated successfully.")


@router.post("/config/thresholds", response_model=ConfigUpdateResponse)
async def eis_update_thresholds(
    body: ThresholdsUpdateRequest,
    request: Request,
) -> ConfigUpdateResponse:
    """
    Update quarantine and block thresholds.

    Validates that block_threshold > quarantine_threshold before applying.
    Modifies ``eis._config`` in place - effective immediately.
    """
    eis = request.app.state.eis
    cfg = eis._config
    cfg.quarantine_threshold = body.quarantine_threshold
    cfg.block_threshold = body.block_threshold

    logger.info(
        "eis_thresholds_updated",
        quarantine=body.quarantine_threshold,
        block=body.block_threshold,
    )

    return ConfigUpdateResponse(updated=True, message="Thresholds updated successfully.")
