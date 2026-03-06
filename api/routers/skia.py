"""
EcodiaOS — Skia System REST Router

Exposes shadow infrastructure (disaster recovery, IPFS snapshots, heartbeat)
to the Next.js frontend.

Endpoints:
  GET  /api/v1/skia/health          — Skia system health + component status
  GET  /api/v1/skia/snapshot        — Latest snapshot manifest from Redis
  GET  /api/v1/skia/snapshot/history — CID history (sorted-set, newest first)
  GET  /api/v1/skia/pins            — List Pinata pins for this instance
  POST /api/v1/skia/snapshot/trigger — Manually trigger a snapshot
  GET  /api/v1/skia/heartbeat       — Heartbeat monitor state (standalone only)
  GET  /api/v1/skia/config          — Non-secret configuration parameters
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.skia")

router = APIRouter()


# ─── Response Models ──────────────────────────────────────────────


class SkiaHealthResponse(EOSBaseModel):
    status: str = "unknown"
    enabled: bool = False
    mode: str = "embedded"
    snapshot_available: bool = False
    heartbeat_available: bool = False
    pinata_connected: bool = False
    heartbeat_status: str | None = None
    consecutive_misses: int = 0
    total_deaths_detected: int = 0
    total_false_positives: int = 0
    last_snapshot_cid: str | None = None
    snapshots_taken: int = 0
    error: str | None = None


class SnapshotManifest(EOSBaseModel):
    ipfs_cid: str = ""
    instance_id: str = ""
    snapshot_at: str = ""
    node_count: int = 0
    edge_count: int = 0
    uncompressed_size_bytes: int = 0
    compressed_size_bytes: int = 0
    encrypted_size_bytes: int = 0
    encryption_key_version: int = 1
    snapshot_duration_ms: float = 0.0
    pinata_pin_id: str = ""


class SnapshotTriggerResponse(EOSBaseModel):
    success: bool
    cid: str | None = None
    node_count: int = 0
    edge_count: int = 0
    duration_ms: float = 0.0
    error: str | None = None


class CIDHistoryItem(EOSBaseModel):
    cid: str
    timestamp: float
    iso_time: str


class CIDHistoryResponse(EOSBaseModel):
    items: list[CIDHistoryItem] = Field(default_factory=list)
    total: int = 0


class PinItem(EOSBaseModel):
    cid: str
    name: str = ""
    pin_id: str = ""
    created_at: str = ""
    size_bytes: int = 0


class PinListResponse(EOSBaseModel):
    pins: list[PinItem] = Field(default_factory=list)
    total: int = 0
    error: str | None = None


class HeartbeatStateResponse(EOSBaseModel):
    status: str = "unknown"
    consecutive_misses: int = 0
    consecutive_confirmations: int = 0
    total_deaths_detected: int = 0
    total_false_positives: int = 0
    last_heartbeat_ago_s: float | None = None
    available: bool = False


class SkiaConfigResponse(EOSBaseModel):
    enabled: bool = False
    snapshot_interval_s: float = 3600.0
    snapshot_max_nodes: int = 50000
    snapshot_node_labels: list[str] = Field(default_factory=list)
    snapshot_include_edges: bool = True
    snapshot_compress: bool = True
    pinata_max_retained_pins: int = 5
    heartbeat_poll_interval_s: float = 5.0
    heartbeat_failure_threshold: int = 12
    heartbeat_confirmation_checks: int = 3
    heartbeat_confirmation_interval_s: float = 10.0
    gcp_region: str = ""
    gcp_service_name: str = ""
    gcp_restart_timeout_s: float = 120.0
    akash_deploy_timeout_s: float = 300.0
    estimated_snapshot_cost_usd: float = 0.001
    estimated_restoration_cost_usd: float = 0.05


# ─── Endpoints ────────────────────────────────────────────────────


@router.get("/api/v1/skia/health", response_model=SkiaHealthResponse)
async def get_skia_health(request: Request) -> SkiaHealthResponse:
    """Skia system health including heartbeat and snapshot status."""
    skia = getattr(request.app.state, "skia", None)

    if skia is None:
        return SkiaHealthResponse(
            status="disabled",
            enabled=False,
            error="Skia not initialized (config.skia.enabled=false)",
        )

    try:
        raw = await skia.health()
        return SkiaHealthResponse(
            status=raw.get("status", "unknown"),
            enabled=raw.get("enabled", False),
            mode=raw.get("mode", "embedded"),
            snapshot_available=raw.get("last_snapshot_cid") is not None or raw.get("snapshots_taken", 0) > 0,
            heartbeat_available=raw.get("heartbeat_status") is not None,
            pinata_connected=raw.get("pinata_connected", False),
            heartbeat_status=raw.get("heartbeat_status"),
            consecutive_misses=raw.get("consecutive_misses", 0),
            total_deaths_detected=raw.get("total_deaths_detected", 0),
            total_false_positives=raw.get("total_false_positives", 0),
            last_snapshot_cid=raw.get("last_snapshot_cid"),
            snapshots_taken=raw.get("snapshots_taken", 0),
        )
    except Exception as exc:
        logger.error("skia_health_failed", error=str(exc))
        return SkiaHealthResponse(status="error", error=str(exc))


@router.get("/api/v1/skia/snapshot", response_model=SnapshotManifest)
async def get_latest_snapshot(request: Request) -> SnapshotManifest:
    """Latest snapshot manifest stored in Redis."""
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")

    config = getattr(request.app.state, "config", None)
    manifest_key = "skia:snapshot_manifest"
    if config is not None and hasattr(config, "skia"):
        manifest_key = config.skia.manifest_redis_key

    try:
        raw = await redis.client.get(manifest_key)
        if raw is None:
            raise HTTPException(status_code=404, detail="No snapshot manifest found")
        data: dict[str, Any] = json.loads(raw)
        # snapshot_at may be a datetime object serialized as ISO string
        return SnapshotManifest(
            ipfs_cid=data.get("ipfs_cid", ""),
            instance_id=data.get("instance_id", ""),
            snapshot_at=str(data.get("snapshot_at", "")),
            node_count=data.get("node_count", 0),
            edge_count=data.get("edge_count", 0),
            uncompressed_size_bytes=data.get("uncompressed_size_bytes", 0),
            compressed_size_bytes=data.get("compressed_size_bytes", 0),
            encrypted_size_bytes=data.get("encrypted_size_bytes", 0),
            encryption_key_version=data.get("encryption_key_version", 1),
            snapshot_duration_ms=data.get("snapshot_duration_ms", 0.0),
            pinata_pin_id=data.get("pinata_pin_id", ""),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("skia_snapshot_read_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/v1/skia/snapshot/history", response_model=CIDHistoryResponse)
async def get_snapshot_history(request: Request, limit: int = 20) -> CIDHistoryResponse:
    """CID history from the Redis sorted-set (newest first)."""
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")

    history_key = "skia:latest_state_cid:history"

    try:
        # ZREVRANGE with scores, newest first
        raw_items: list[tuple[bytes, float]] = await redis.client.zrevrange(
            history_key, 0, limit - 1, withscores=True
        )
        import datetime as _dt

        items: list[CIDHistoryItem] = []
        for cid_bytes, score in raw_items:
            cid = cid_bytes.decode() if isinstance(cid_bytes, bytes) else str(cid_bytes)
            iso = _dt.datetime.fromtimestamp(score, tz=_dt.UTC).isoformat()
            items.append(CIDHistoryItem(cid=cid, timestamp=score, iso_time=iso))

        total = await redis.client.zcard(history_key)
        return CIDHistoryResponse(items=items, total=total)
    except Exception as exc:
        logger.error("skia_history_read_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/v1/skia/pins", response_model=PinListResponse)
async def list_pins(request: Request, limit: int = 20) -> PinListResponse:
    """List Pinata IPFS pins for this instance."""
    skia = getattr(request.app.state, "skia", None)
    if skia is None:
        return PinListResponse(error="Skia not initialized", pins=[], total=0)

    pinata = getattr(skia, "_pinata", None)
    config = getattr(request.app.state, "config", None)
    group_name = "ecodiaos-skia-snapshots"
    if config is not None and hasattr(config, "skia"):
        group_name = config.skia.pinata_group_name

    if pinata is None:
        return PinListResponse(error="Pinata not connected", pins=[], total=0)

    try:
        raw_pins: list[dict[str, Any]] = await pinata.list_pins(
            name_contains=group_name, limit=limit
        )
        pins: list[PinItem] = []
        for p in raw_pins:
            metadata = p.get("metadata", {}) or {}
            regions = p.get("regions", []) or []
            size = regions[0].get("currentReplicationCount", 0) if regions else 0
            pins.append(PinItem(
                cid=p.get("ipfs_pin_hash", ""),
                name=metadata.get("name", ""),
                pin_id=p.get("id", ""),
                created_at=p.get("date_pinned", ""),
                size_bytes=p.get("size", size),
            ))
        return PinListResponse(pins=pins, total=len(pins))
    except Exception as exc:
        logger.error("skia_pins_failed", error=str(exc))
        return PinListResponse(error=str(exc), pins=[], total=0)


@router.post("/api/v1/skia/snapshot/trigger", response_model=SnapshotTriggerResponse)
async def trigger_snapshot(request: Request) -> SnapshotTriggerResponse:
    """Manually trigger an immediate snapshot."""
    skia = getattr(request.app.state, "skia", None)
    if skia is None:
        raise HTTPException(status_code=503, detail="Skia not initialized")

    snapshot_pipeline = getattr(skia, "_snapshot", None)
    if snapshot_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Snapshot pipeline unavailable (vault or Pinata not configured)",
        )

    try:
        t0 = time.monotonic()
        manifest = await snapshot_pipeline.take_snapshot()
        duration_ms = (time.monotonic() - t0) * 1000

        if manifest is None:
            return SnapshotTriggerResponse(success=False, error="Snapshot returned no manifest")

        return SnapshotTriggerResponse(
            success=True,
            cid=manifest.ipfs_cid,
            node_count=manifest.node_count,
            edge_count=manifest.edge_count,
            duration_ms=round(manifest.snapshot_duration_ms or duration_ms, 1),
        )
    except Exception as exc:
        logger.error("skia_snapshot_trigger_failed", error=str(exc))
        return SnapshotTriggerResponse(success=False, error=str(exc))


@router.get("/api/v1/skia/heartbeat", response_model=HeartbeatStateResponse)
async def get_heartbeat_state(request: Request) -> HeartbeatStateResponse:
    """
    Heartbeat monitor state. Only available when Skia runs in standalone mode.
    In embedded mode (main process), returns available=False.
    """
    skia = getattr(request.app.state, "skia", None)
    if skia is None:
        return HeartbeatStateResponse(status="unavailable", available=False)

    heartbeat = getattr(skia, "_heartbeat", None)
    if heartbeat is None:
        return HeartbeatStateResponse(
            status="not_running",
            available=False,
        )

    try:
        state = heartbeat.state
        last_ago: float | None = None
        last_event = getattr(heartbeat, "_last_event_time", None)
        if last_event is not None:
            last_ago = round(time.monotonic() - last_event, 1)

        return HeartbeatStateResponse(
            status=state.status.value,
            consecutive_misses=state.consecutive_misses,
            consecutive_confirmations=state.consecutive_confirmations,
            total_deaths_detected=state.total_deaths_detected,
            total_false_positives=state.total_false_positives,
            last_heartbeat_ago_s=last_ago,
            available=True,
        )
    except Exception as exc:
        logger.error("skia_heartbeat_read_failed", error=str(exc))
        return HeartbeatStateResponse(status="error", available=False)


@router.get("/api/v1/skia/config", response_model=SkiaConfigResponse)
async def get_skia_config(request: Request) -> SkiaConfigResponse:
    """
    Non-secret Skia configuration parameters.
    Omits JWT tokens, service account keys, and wallet addresses.
    """
    config = getattr(request.app.state, "config", None)
    if config is None or not hasattr(config, "skia"):
        raise HTTPException(status_code=503, detail="Config not available")

    c = config.skia
    return SkiaConfigResponse(
        enabled=c.enabled,
        snapshot_interval_s=c.snapshot_interval_s,
        snapshot_max_nodes=c.snapshot_max_nodes,
        snapshot_node_labels=list(c.snapshot_node_labels),
        snapshot_include_edges=c.snapshot_include_edges,
        snapshot_compress=c.snapshot_compress,
        pinata_max_retained_pins=c.pinata_max_retained_pins,
        heartbeat_poll_interval_s=c.heartbeat_poll_interval_s,
        heartbeat_failure_threshold=c.heartbeat_failure_threshold,
        heartbeat_confirmation_checks=c.heartbeat_confirmation_checks,
        heartbeat_confirmation_interval_s=c.heartbeat_confirmation_interval_s,
        gcp_region=c.gcp_region,
        gcp_service_name=c.gcp_service_name,
        gcp_restart_timeout_s=c.gcp_restart_timeout_s,
        akash_deploy_timeout_s=c.akash_deploy_timeout_s,
        estimated_snapshot_cost_usd=c.estimated_snapshot_cost_usd,
        estimated_restoration_cost_usd=c.estimated_restoration_cost_usd,
    )
