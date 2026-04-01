"""
EcodiaOS - SACM (Substrate-Arbitrage Compute Mesh) API Router

Exposes read-only endpoints for the frontend SACM dashboard:

  GET /api/v1/sacm/metrics        - SACMMetrics.snapshot + rolling burn rate
  GET /api/v1/sacm/savings        - SACMCostAccounting.get_savings_report()
  GET /api/v1/sacm/providers      - Market oracle: provider health + offer counts
  GET /api/v1/sacm/oracle         - Pricing surface snapshot
  GET /api/v1/sacm/compute        - ComputeResourceManager: capacity, queue, allocations
  GET /api/v1/sacm/pre-warm       - PreWarmingEngine: pool state + predictor EMA
  GET /api/v1/sacm/health         - Aggregate system health across all subsystems
"""

from __future__ import annotations

import contextlib
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import Field

from primitives.common import EOSBaseModel

if TYPE_CHECKING:
    from systems.sacm.accounting import ProviderSpendSummary, SavingsReport
    from systems.sacm.service import WorkloadHistoryRecord

logger = structlog.get_logger("api.sacm")

router = APIRouter(prefix="/api/v1/sacm", tags=["sacm"])


# ─── Response Models ──────────────────────────────────────────────


class SACMMetricsResponse(EOSBaseModel):
    """Current SACM telemetry snapshot."""

    # Cost
    total_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    savings_cost_usd: float = 0.0

    # Workloads
    workloads_submitted: int = 0
    workloads_placed_remote: int = 0
    workloads_completed: int = 0
    workloads_failed: int = 0
    workloads_rejected: int = 0

    # Verification
    verification_pass_rate: float = 0.0

    # Burn rate from the rolling accounting window
    rolling_burn_rate_usd_per_hour: float = 0.0


class SACMSavingsResponse(EOSBaseModel):
    """Serialised SavingsReport from SACMCostAccounting.get_savings_report()."""

    period_label: str = ""
    record_count: int = 0
    total_actual_usd: float = 0.0
    total_baseline_usd: float = 0.0
    total_savings_usd: float = 0.0
    savings_ratio: float = 0.0
    avg_cost_per_workload_usd: float = 0.0
    avg_savings_per_workload_usd: float = 0.0
    top_providers: list[ProviderSpendSummary] = Field(default_factory=list)
    generated_at: datetime


class SACMProviderHealthResponse(EOSBaseModel):
    """Health record for a single substrate provider."""

    provider_id: str
    status: str  # "healthy" | "degraded" | "unreachable"
    offer_count: int = 0
    valid_offer_count: int = 0
    consecutive_failures: int = 0
    last_success_epoch: float | None = None
    last_failure_reason: str | None = None


class SACMProvidersResponse(EOSBaseModel):
    """Oracle state: all registered providers and their health."""

    providers: list[SACMProviderHealthResponse] = Field(default_factory=list)
    total_providers: int = 0
    healthy_providers: int = 0
    total_offers: int = 0
    valid_offers: int = 0
    last_refresh_epoch: float = 0.0


class SACMOfferSummary(EOSBaseModel):
    """Summary of a single substrate offer for the pricing surface view."""

    offer_id: str
    provider_id: str
    offload_class: str
    region: str = ""
    cpu_vcpu: float = 0.0
    memory_gib: float = 0.0
    gpu_units: float = 0.0
    gpu_vram_gib: float = 0.0
    price_cpu_per_vcpu_s: float = 0.0
    price_mem_per_gib_s: float = 0.0
    price_gpu_per_unit_s: float = 0.0
    price_egress_per_gib: float = 0.0
    latency_ms_p50: float = 0.0
    trust_score: float = 1.0
    valid_until_epoch: float | None = None


class SACMOracleResponse(EOSBaseModel):
    """Current pricing surface from the market oracle."""

    offers: list[SACMOfferSummary] = Field(default_factory=list)
    total_offers: int = 0
    valid_offers: int = 0
    cheapest_cpu_offer: SACMOfferSummary | None = None
    cheapest_gpu_offer: SACMOfferSummary | None = None
    last_refresh_epoch: float = 0.0


class SACMComputeResponse(EOSBaseModel):
    """ComputeResourceManager snapshot: capacity, queue depth, allocation counts."""

    node_id: str = "local"
    cpu_vcpu_total: float = 0.0
    cpu_vcpu_available: float = 0.0
    memory_gib_total: float = 0.0
    memory_gib_available: float = 0.0
    gpu_units_total: float = 0.0
    gpu_units_available: float = 0.0
    gpu_vram_gib_total: float = 0.0
    gpu_vram_gib_available: float = 0.0
    queue_depth: int = 0
    active_count: int = 0
    total_allocated: int = 0
    total_queued: int = 0
    total_denied: int = 0
    total_offloaded: int = 0
    held_cpu_by_subsystem: dict[str, float] = Field(default_factory=dict)


class SACMWarmInstanceSummary(EOSBaseModel):
    """Summary of a single warm (pre-provisioned) compute instance."""

    instance_id: str
    offload_class: str
    provider_id: str
    status: str  # "available" | "claimed" | "expired"
    claimed_by: str | None = None
    created_epoch: float = 0.0
    expires_epoch: float = 0.0
    cost_usd_per_hour: float = 0.0


class SACMDemandForecastSummary(EOSBaseModel):
    """EMA demand forecast for a single offload class."""

    offload_class: str
    ema_value: float = 0.0
    history_samples: int = 0


class SACMPreWarmResponse(EOSBaseModel):
    """PreWarmingEngine state: warm pool, demand forecasts, budget usage."""

    warm_instances: list[SACMWarmInstanceSummary] = Field(default_factory=list)
    demand_forecasts: list[SACMDemandForecastSummary] = Field(default_factory=list)
    pool_size: int = 0
    available_instances: int = 0
    claimed_instances: int = 0
    budget_usd_per_hour: float = 0.0
    max_instances: int = 0


class SACMSubsystemHealth(EOSBaseModel):
    """Health status for one SACM subsystem."""

    name: str
    status: str  # "ok" | "degraded" | "error"
    detail: str = ""


class SACMHealthResponse(EOSBaseModel):
    """Aggregate health across all SACM subsystems."""

    overall: str  # "ok" | "degraded" | "error"
    subsystems: list[SACMSubsystemHealth] = Field(default_factory=list)
    checked_at: datetime


# ─── Helpers ─────────────────────────────────────────────────────


def _offer_to_summary(offer: Any) -> SACMOfferSummary:
    # offload_class: take first supported class, or "general" as fallback
    supported = getattr(offer, "supported_classes", [])
    if supported:
        first_cls = supported[0]
        offload_class_str = first_cls.value if hasattr(first_cls, "value") else str(first_cls)
    else:
        offload_class_str = "general"

    # valid_until: convert datetime → epoch float
    valid_until = getattr(offer, "valid_until", None)
    valid_until_epoch: float | None = None
    if valid_until is not None:
        with contextlib.suppress(Exception):
            valid_until_epoch = valid_until.timestamp()

    # latency: avg_latency_overhead_s → ms
    latency_ms = getattr(offer, "avg_latency_overhead_s", 0.0) * 1000.0

    return SACMOfferSummary(
        offer_id=getattr(offer, "id", str(id(offer))),
        provider_id=offer.provider_id,
        offload_class=offload_class_str,
        region=getattr(offer, "region", ""),
        cpu_vcpu=getattr(offer, "max_cpu_vcpu", 0.0),
        memory_gib=getattr(offer, "max_memory_gib", 0.0),
        gpu_units=getattr(offer, "max_gpu_units", 0.0),
        gpu_vram_gib=getattr(offer, "gpu_vram_gib", 0.0),
        price_cpu_per_vcpu_s=getattr(offer, "price_cpu_per_vcpu_s", 0.0),
        price_mem_per_gib_s=getattr(offer, "price_mem_per_gib_s", 0.0),
        price_gpu_per_unit_s=getattr(offer, "price_gpu_per_unit_s", 0.0),
        price_egress_per_gib=getattr(offer, "price_egress_per_gib", 0.0),
        latency_ms_p50=latency_ms,
        trust_score=getattr(offer, "trust_score", 1.0),
        valid_until_epoch=valid_until_epoch,
    )


# ─── Routes ──────────────────────────────────────────────────────


@router.get("/metrics", response_model=SACMMetricsResponse)
async def sacm_metrics(request: Request) -> SACMMetricsResponse:
    """Return the current SACM metrics snapshot."""
    client = request.app.state.sacm_client
    accounting = request.app.state.sacm_accounting

    snapshot: dict[str, float | int] = client._metrics.snapshot
    burn_rate: float = accounting.rolling_burn_rate_usd_per_hour

    return SACMMetricsResponse(
        total_cost_usd=float(snapshot.get("sacm.cost.total_usd", 0.0)),
        estimated_cost_usd=float(snapshot.get("sacm.cost.estimated_usd", 0.0)),
        savings_cost_usd=float(snapshot.get("sacm.cost.savings_usd", 0.0)),
        workloads_submitted=int(snapshot.get("sacm.workloads.submitted", 0)),
        workloads_placed_remote=int(snapshot.get("sacm.workloads.placed.remote", 0)),
        workloads_completed=int(snapshot.get("sacm.workloads.completed", 0)),
        workloads_failed=int(snapshot.get("sacm.workloads.failed", 0)),
        workloads_rejected=int(snapshot.get("sacm.workloads.rejected", 0)),
        verification_pass_rate=float(snapshot.get("sacm.verification.pass_rate", 0.0)),
        rolling_burn_rate_usd_per_hour=burn_rate,
    )


@router.get("/savings", response_model=SACMSavingsResponse)
async def sacm_savings(request: Request) -> SACMSavingsResponse:
    """Return the current SACM savings report."""
    accounting = request.app.state.sacm_accounting
    report: SavingsReport = accounting.get_savings_report()
    data = report.model_dump()
    return SACMSavingsResponse(**data)


@router.get("/providers", response_model=SACMProvidersResponse)
async def sacm_providers(request: Request) -> SACMProvidersResponse:
    """Return provider health from the market oracle."""
    oracle = request.app.state.sacm_oracle

    provider_health_map: dict[str, Any] = oracle.all_provider_health
    snapshot = oracle.snapshot
    now = time.time()

    providers: list[SACMProviderHealthResponse] = []
    for pid, health in provider_health_map.items():
        provider_offers = [o for o in snapshot.offers if o.provider_id == pid]
        valid_offers = [
            o for o in provider_offers
            if getattr(o, "valid_until", None) is None or o.valid_until.timestamp() > now
        ]

        providers.append(SACMProviderHealthResponse(
            provider_id=pid,
            status=health.status.value if hasattr(health.status, "value") else str(health.status),
            offer_count=len(provider_offers),
            valid_offer_count=len(valid_offers),
            consecutive_failures=health.consecutive_failures,
            last_success_epoch=health.last_successful_fetch,
            last_failure_reason=None,
        ))

    healthy_count = sum(1 for p in providers if p.status == "healthy")

    return SACMProvidersResponse(
        providers=providers,
        total_providers=len(providers),
        healthy_providers=healthy_count,
        total_offers=oracle.offer_count,
        valid_offers=oracle.valid_offer_count,
        last_refresh_epoch=oracle._last_refresh_epoch,
    )


@router.get("/oracle", response_model=SACMOracleResponse)
async def sacm_oracle(request: Request) -> SACMOracleResponse:
    """Return the current pricing surface from the market oracle."""
    oracle = request.app.state.sacm_oracle
    snapshot = oracle.snapshot

    offer_summaries = [_offer_to_summary(o) for o in snapshot.offers]

    cheapest_cpu = oracle.cheapest_cpu_offer()
    cheapest_gpu = oracle.cheapest_gpu_offer()

    return SACMOracleResponse(
        offers=offer_summaries,
        total_offers=oracle.offer_count,
        valid_offers=oracle.valid_offer_count,
        cheapest_cpu_offer=_offer_to_summary(cheapest_cpu) if cheapest_cpu else None,
        cheapest_gpu_offer=_offer_to_summary(cheapest_gpu) if cheapest_gpu else None,
        last_refresh_epoch=oracle._last_refresh_epoch,
    )


@router.get("/compute", response_model=SACMComputeResponse)
async def sacm_compute(request: Request) -> SACMComputeResponse:
    """Return the ComputeResourceManager snapshot."""
    mgr = request.app.state.sacm_compute_manager

    capacity = mgr.inventory

    return SACMComputeResponse(
        node_id=capacity.node_id,
        cpu_vcpu_total=capacity.cpu_vcpu_total,
        cpu_vcpu_available=capacity.cpu_vcpu_available,
        memory_gib_total=capacity.memory_gib_total,
        memory_gib_available=capacity.memory_gib_available,
        gpu_units_total=capacity.gpu_units_total,
        gpu_units_available=capacity.gpu_units_available,
        gpu_vram_gib_total=capacity.gpu_vram_gib_total,
        gpu_vram_gib_available=capacity.gpu_vram_gib_available,
        queue_depth=mgr.queue_depth,
        active_count=mgr.active_count,
        total_allocated=mgr._total_allocated,
        total_queued=mgr._total_queued,
        total_denied=mgr._total_denied,
        total_offloaded=mgr._total_offloaded,
        held_cpu_by_subsystem=dict(mgr._held_cpu),
    )


@router.get("/pre-warm", response_model=SACMPreWarmResponse)
async def sacm_pre_warm(request: Request) -> SACMPreWarmResponse:
    """Return PreWarmingEngine pool state and demand forecasts."""
    engine = request.app.state.sacm_prewarm_engine
    config = request.app.state.sacm_prewarm_config

    time.time()

    warm_instances: list[SACMWarmInstanceSummary] = []
    available = 0
    claimed_count = 0

    for iid, inst in engine._pool.items():
        status_val = inst.status.value if hasattr(inst.status, "value") else str(inst.status)
        # Map WarmInstanceStatus values to simplified display values
        if status_val in ("ready",):
            available += 1
        elif status_val in ("claimed",):
            claimed_count += 1

        warm_instances.append(SACMWarmInstanceSummary(
            instance_id=iid,
            offload_class=inst.offload_class.value if hasattr(inst.offload_class, "value") else str(inst.offload_class),
            provider_id=inst.provider_id,
            status=status_val,
            claimed_by=inst.claimed_by_workload_id if inst.claimed_by_workload_id else None,
            created_epoch=inst.provisioned_at,
            expires_epoch=inst.last_heartbeat,
            cost_usd_per_hour=inst.hourly_cost_usd,
        ))

    demand_forecasts: list[SACMDemandForecastSummary] = []
    predictor = engine._predictor
    for cls, ema in predictor._ema_rate.items():
        cls_str = cls.value if hasattr(cls, "value") else str(cls)
        history = predictor._history.get(cls, [])
        demand_forecasts.append(SACMDemandForecastSummary(
            offload_class=cls_str,
            ema_value=ema,
            history_samples=len(history),
        ))

    return SACMPreWarmResponse(
        warm_instances=warm_instances,
        demand_forecasts=demand_forecasts,
        pool_size=len(warm_instances),
        available_instances=available,
        claimed_instances=claimed_count,
        budget_usd_per_hour=getattr(config, "max_pre_warm_budget_usd_per_hour", 5.0),
        max_instances=getattr(config, "max_warm_instances", 10),
    )


# ─── History Response Models ─────────────────────────────────────


class SACMWorkloadHistoryItem(EOSBaseModel):
    """Summary row for the workload history table."""

    id: str
    offload_class: str = ""
    priority: str = ""
    status: str
    provider_id: str = ""
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    savings_usd: float = 0.0
    duration_s: float = 0.0
    verification_passed: bool | None = None
    consensus_score: float | None = None
    error_message: str | None = None
    submitted_at: float
    completed_at: float | None = None


class SACMWorkloadHistoryResponse(EOSBaseModel):
    """Paginated workload history."""

    records: list[SACMWorkloadHistoryItem] = Field(default_factory=list)
    total: int = 0


# ─── Verification Trust Response Models ─────────────────────────


class SACMProviderTrustRecord(EOSBaseModel):
    """Trust record for a single remote provider."""

    provider_id: str
    trust_score: float
    total_batches: int = 0
    accepted_batches: int = 0
    rejected_batches: int = 0
    consecutive_failures: int = 0
    quarantined: bool = False


class SACMVerificationTrustResponse(EOSBaseModel):
    """All provider trust records from the consensus verifier."""

    providers: list[SACMProviderTrustRecord] = Field(default_factory=list)


# ─── Mutation Request Bodies ─────────────────────────────────────


class SACMResetTrustRequest(EOSBaseModel):
    reason: str = ""


class SACMPreWarmTriggerRequest(EOSBaseModel):
    offload_class: str = "general"
    count: int = 1


# ─── Mutation Response Models ────────────────────────────────────


class SACMOracleRefreshResponse(EOSBaseModel):
    refreshed: bool
    message: str
    last_refresh_epoch: float = 0.0
    total_offers: int = 0
    valid_offers: int = 0


class SACMResetTrustResponse(EOSBaseModel):
    provider_id: str
    trust_score: float
    message: str


class SACMPreWarmTriggerResponse(EOSBaseModel):
    triggered: bool
    message: str
    pool_size: int = 0


@router.get("/health", response_model=SACMHealthResponse)
async def sacm_health(request: Request) -> SACMHealthResponse:
    """Return aggregate health across all SACM subsystems."""
    subsystems: list[SACMSubsystemHealth] = []

    async def _check_async(name: str, coro: Any) -> SACMSubsystemHealth:
        try:
            return await coro
        except Exception as exc:
            return SACMSubsystemHealth(name=name, status="error", detail=str(exc))

    def _check_sync(name: str, fn: Any) -> SACMSubsystemHealth:
        try:
            return fn()
        except Exception as exc:
            return SACMSubsystemHealth(name=name, status="error", detail=str(exc))

    async def _client_health() -> SACMSubsystemHealth:
        h = await request.app.state.sacm_client.health()
        return SACMSubsystemHealth(
            name="sacm_client", status=h.get("status", "ok"), detail=h.get("detail", ""),
        )

    async def _compute_health() -> SACMSubsystemHealth:
        h = await request.app.state.sacm_compute_manager.health()
        return SACMSubsystemHealth(
            name="compute_manager", status=h.get("status", "ok"), detail=h.get("detail", ""),
        )

    def _oracle_health() -> SACMSubsystemHealth:
        oracle = request.app.state.sacm_oracle
        valid = oracle.valid_offer_count
        total = oracle.offer_count
        status = "ok" if valid > 0 else ("degraded" if total > 0 else "error")
        return SACMSubsystemHealth(name="oracle", status=status, detail=f"{valid}/{total} valid offers")

    def _prewarm_health() -> SACMSubsystemHealth:
        engine = request.app.state.sacm_prewarm_engine
        pool_size = len(engine._pool)
        return SACMSubsystemHealth(name="pre_warm", status="ok", detail=f"{pool_size} warm instances")

    def _accounting_health() -> SACMSubsystemHealth:
        accounting = request.app.state.sacm_accounting
        burn = accounting.rolling_burn_rate_usd_per_hour
        budget = accounting._budget_usd_per_hour
        stress = min(1.0, burn / budget) if budget > 0 else 0.0
        status = "ok" if stress < 0.7 else ("degraded" if stress < 1.0 else "error")
        return SACMSubsystemHealth(name="accounting", status=status, detail=f"${burn:.2f}/hr ({stress * 100:.0f}% budget)")

    subsystems.append(await _check_async("sacm_client", _client_health()))
    subsystems.append(await _check_async("compute_manager", _compute_health()))
    subsystems.append(_check_sync("oracle", _oracle_health))
    subsystems.append(_check_sync("pre_warm", _prewarm_health))
    subsystems.append(_check_sync("accounting", _accounting_health))

    statuses = {s.status for s in subsystems}
    overall = "error" if "error" in statuses else ("degraded" if "degraded" in statuses else "ok")

    return SACMHealthResponse(
        overall=overall,
        subsystems=subsystems,
        checked_at=datetime.utcnow(),
    )


# ─── History Routes ───────────────────────────────────────────────


@router.get("/history", response_model=SACMWorkloadHistoryResponse)
async def sacm_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    status: str | None = Query(default=None),
    provider_id: str | None = Query(default=None),
) -> SACMWorkloadHistoryResponse:
    """Return paginated workload execution history."""
    store = request.app.state.sacm_history
    records: list[WorkloadHistoryRecord] = store.query(
        limit=limit,
        status=status or None,
        provider_id=provider_id or None,
    )
    items = [
        SACMWorkloadHistoryItem(
            id=r.id,
            offload_class=r.offload_class,
            priority=r.priority,
            status=r.status,
            provider_id=r.provider_id,
            estimated_cost_usd=r.estimated_cost_usd,
            actual_cost_usd=r.actual_cost_usd,
            savings_usd=r.savings_usd,
            duration_s=r.duration_s,
            verification_passed=r.verification_passed,
            consensus_score=r.consensus_score,
            error_message=r.error_message,
            submitted_at=r.submitted_at,
            completed_at=r.completed_at,
        )
        for r in records
    ]
    return SACMWorkloadHistoryResponse(records=items, total=len(items))


@router.get("/history/{workload_id}", response_model=SACMWorkloadHistoryItem)
async def sacm_history_detail(request: Request, workload_id: str) -> SACMWorkloadHistoryItem:
    """Return a single workload history record by ID."""
    store = request.app.state.sacm_history
    record: WorkloadHistoryRecord | None = store.get(workload_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Workload {workload_id!r} not found in history")
    return SACMWorkloadHistoryItem(
        id=record.id,
        offload_class=record.offload_class,
        priority=record.priority,
        status=record.status,
        provider_id=record.provider_id,
        estimated_cost_usd=record.estimated_cost_usd,
        actual_cost_usd=record.actual_cost_usd,
        savings_usd=record.savings_usd,
        duration_s=record.duration_s,
        verification_passed=record.verification_passed,
        consensus_score=record.consensus_score,
        error_message=record.error_message,
        submitted_at=record.submitted_at,
        completed_at=record.completed_at,
    )


# ─── Control Action Routes ────────────────────────────────────────

_ORACLE_REFRESH_MIN_INTERVAL_S: float = 30.0


@router.post("/oracle/refresh", response_model=SACMOracleRefreshResponse)
async def sacm_oracle_refresh(request: Request) -> SACMOracleRefreshResponse:
    """Force an immediate oracle refresh. Rate-limited to once per 30s."""
    now = time.time()
    last = getattr(request.app.state, "sacm_oracle_last_refresh_request", 0.0)
    if now - last < _ORACLE_REFRESH_MIN_INTERVAL_S:
        wait = _ORACLE_REFRESH_MIN_INTERVAL_S - (now - last)
        raise HTTPException(
            status_code=429,
            detail=f"Oracle refresh rate-limited. Retry in {wait:.0f}s.",
        )

    oracle = request.app.state.sacm_oracle
    request.app.state.sacm_oracle_last_refresh_request = now

    logger.info("sacm_oracle_refresh_triggered", triggered_at=now)

    await oracle.refresh_all()

    return SACMOracleRefreshResponse(
        refreshed=True,
        message="Oracle refreshed successfully",
        last_refresh_epoch=oracle._last_refresh_epoch,
        total_offers=oracle.offer_count,
        valid_offers=oracle.valid_offer_count,
    )


@router.post("/providers/{provider_id}/reset-trust", response_model=SACMResetTrustResponse)
async def sacm_reset_provider_trust(
    request: Request,
    provider_id: str,
    body: SACMResetTrustRequest,
) -> SACMResetTrustResponse:
    """Reset a provider's trust score to 1.0 in the consensus verifier."""
    verifier = request.app.state.sacm_consensus_verifier
    trust = verifier.get_provider_trust(provider_id)
    trust.trust_score = 1.0
    trust.consecutive_failures = 0

    logger.info(
        "sacm_provider_trust_reset",
        provider_id=provider_id,
        reason=body.reason,
    )

    return SACMResetTrustResponse(
        provider_id=provider_id,
        trust_score=trust.trust_score,
        message=f"Trust score reset to 1.0 for provider {provider_id!r}. Reason: {body.reason or 'unspecified'}",
    )


@router.post("/pre-warm/trigger", response_model=SACMPreWarmTriggerResponse)
async def sacm_prewarm_trigger(
    request: Request,
    body: SACMPreWarmTriggerRequest,
) -> SACMPreWarmTriggerResponse:
    """Trigger an immediate pre-warm cycle outside the normal loop interval."""
    count = max(1, min(3, body.count))
    engine = request.app.state.sacm_prewarm_engine

    logger.info(
        "sacm_prewarm_trigger",
        offload_class=body.offload_class,
        count=count,
    )

    try:
        await engine._tick()
    except Exception as exc:
        logger.warning("sacm_prewarm_trigger_tick_error", error=str(exc))

    return SACMPreWarmTriggerResponse(
        triggered=True,
        message=f"Pre-warm cycle triggered for class {body.offload_class!r} (requested {count})",
        pool_size=len(engine._pool),
    )


# ─── Verification Trust Route ─────────────────────────────────────


@router.get("/verification/trust", response_model=SACMVerificationTrustResponse)
async def sacm_verification_trust(request: Request) -> SACMVerificationTrustResponse:
    """Return all provider trust records from the consensus verifier."""
    verifier = request.app.state.sacm_consensus_verifier
    trust_store: dict[str, Any] = verifier._trust_store

    providers = [
        SACMProviderTrustRecord(
            provider_id=t.provider_id,
            trust_score=round(t.trust_score, 4),
            total_batches=t.total_batches,
            accepted_batches=t.accepted_batches,
            rejected_batches=t.rejected_batches,
            consecutive_failures=t.consecutive_failures,
            quarantined=t.is_quarantined,
        )
        for t in trust_store.values()
    ]

    return SACMVerificationTrustResponse(providers=providers)
