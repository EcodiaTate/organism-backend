"""
EcodiaOS — Axon (Action Executor / Reflex Arc) API Router

Exposes Axon's execution metrics, safety system state, executor registry,
budget tracking, circuit breakers, rate limits, MEV shield, and audit data.

  GET  /api/v1/axon/stats              — Aggregate execution metrics
  GET  /api/v1/axon/outcomes           — Recent execution outcomes (moved here)
  GET  /api/v1/axon/budget             — Current cycle budget utilisation
  GET  /api/v1/axon/executors          — Registry of all registered executors
  GET  /api/v1/axon/safety             — Circuit breaker + rate limiter state
  GET  /api/v1/axon/shield             — TransactionShield + MEV metrics
  GET  /api/v1/axon/fast-path          — Fast-path execution statistics
  GET  /api/v1/axon/audit              — Recent audit records (from memory/outcomes)
  GET  /api/v1/axon/stream             — SSE stream of AxonOutcome events
  GET  /api/v1/axon/mev-competition    — Block competition snapshot
  POST /api/v1/axon/safety/reset/{action_type}  — Reset a circuit breaker
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.axon")

router = APIRouter(prefix="/api/v1/axon", tags=["axon"])


# ─── Response Models ──────────────────────────────────────────────


class AxonStatsResponse(EOSBaseModel):
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    success_rate: float = 0.0
    recent_outcomes_count: int = 0
    initialized: bool = False
    executor_count: int = 0


class BudgetResponse(EOSBaseModel):
    actions_used: int = 0
    actions_max: int = 5
    concurrent_used: int = 0
    concurrent_max: int = 3
    utilisation: float = 0.0
    cycle_age_ms: float = 0.0
    budget_config: dict[str, Any] = {}


class ExecutorInfo(EOSBaseModel):
    action_type: str
    description: str
    reversible: bool = False
    counts_toward_budget: bool = True
    emits_to_atune: bool = True
    max_duration_ms: int = 30000
    required_autonomy: int = 1


class ExecutorRegistryResponse(EOSBaseModel):
    executors: list[ExecutorInfo] = []
    total: int = 0


class CircuitBreakerState(EOSBaseModel):
    action_type: str
    status: str  # "closed" | "open" | "half_open"
    consecutive_failures: int = 0
    tripped_at: float | None = None


class RateLimiterState(EOSBaseModel):
    action_type: str
    current_count: int = 0
    window_seconds: float = 60.0


class SafetyResponse(EOSBaseModel):
    circuit_breakers: list[CircuitBreakerState] = []
    rate_limiters: list[RateLimiterState] = []
    failure_threshold: int = 5
    recovery_timeout_s: int = 300


class ShieldMetricsResponse(EOSBaseModel):
    total_evaluated: int = 0
    total_rejected: int = 0
    mev_protected: int = 0
    mev_saved_usd: float = 0.0
    rejection_rate: float = 0.0
    last_mev_risk_score: float | None = None
    last_mev_strategy: str | None = None
    blacklisted_addresses: int = 0


class FastPathStatsResponse(EOSBaseModel):
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    capital_deployed: float = 0.0
    mean_latency_ms: float = 0.0
    active_templates: int = 0


# ─── Routes ───────────────────────────────────────────────────────


@router.get("/stats", response_model=AxonStatsResponse)
async def get_stats(request: Request) -> AxonStatsResponse:
    """Aggregate Axon execution metrics."""
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return AxonStatsResponse(initialized=False)

    total = getattr(axon, "_total_executions", 0)
    successful = getattr(axon, "_successful_executions", 0)
    failed = getattr(axon, "_failed_executions", 0)
    recent = getattr(axon, "recent_outcomes", [])
    registry = getattr(axon, "_registry", None)
    executor_count = len(registry) if registry is not None else 0

    success_rate = (successful / total) if total > 0 else 0.0

    return AxonStatsResponse(
        total_executions=total,
        successful_executions=successful,
        failed_executions=failed,
        success_rate=round(success_rate, 4),
        recent_outcomes_count=len(recent),
        initialized=True,
        executor_count=executor_count,
    )


@router.get("/budget", response_model=BudgetResponse)
async def get_budget(request: Request) -> BudgetResponse:
    """Current cycle budget utilisation and limits."""
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return BudgetResponse()

    pipeline = getattr(axon, "_pipeline", None)
    if pipeline is None:
        return BudgetResponse()

    budget_tracker = getattr(pipeline, "_budget", None)
    if budget_tracker is None:
        return BudgetResponse()

    import time

    config = getattr(axon, "_config", None)
    budget_config: dict[str, Any] = {}
    if config is not None:
        budget_config = {
            "max_actions_per_cycle": getattr(config, "max_actions_per_cycle", 5),
            "max_api_calls_per_minute": getattr(config, "max_api_calls_per_minute", 30),
            "max_notifications_per_hour": getattr(config, "max_notifications_per_hour", 10),
            "max_concurrent_executions": getattr(config, "max_concurrent_executions", 3),
            "total_timeout_per_cycle_ms": getattr(config, "total_timeout_per_cycle_ms", 30000),
        }

    actions_used = getattr(budget_tracker, "_actions_this_cycle", 0)
    concurrent_used = getattr(budget_tracker, "_concurrent_executions", 0)
    utilisation = getattr(budget_tracker, "utilisation", 0.0)
    cycle_start = getattr(budget_tracker, "_cycle_start", time.monotonic())
    cycle_age_ms = (time.monotonic() - cycle_start) * 1000

    return BudgetResponse(
        actions_used=actions_used,
        actions_max=budget_config.get("max_actions_per_cycle", 5),
        concurrent_used=concurrent_used,
        concurrent_max=budget_config.get("max_concurrent_executions", 3),
        utilisation=round(utilisation, 4),
        cycle_age_ms=round(cycle_age_ms, 1),
        budget_config=budget_config,
    )


@router.get("/executors", response_model=ExecutorRegistryResponse)
async def get_executors(request: Request) -> ExecutorRegistryResponse:
    """List all executors registered in Axon's registry."""
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return ExecutorRegistryResponse()

    registry = getattr(axon, "_registry", None)
    if registry is None:
        return ExecutorRegistryResponse()

    executors: list[ExecutorInfo] = []
    try:
        for action_type in registry.list_types():
            executor = registry.get(action_type)
            if executor is None:
                continue
            executors.append(
                ExecutorInfo(
                    action_type=action_type,
                    description=getattr(executor, "description", ""),
                    reversible=getattr(executor, "reversible", False),
                    counts_toward_budget=getattr(executor, "counts_toward_budget", True),
                    emits_to_atune=getattr(executor, "emits_to_atune", True),
                    max_duration_ms=getattr(executor, "max_duration_ms", 30000),
                    required_autonomy=getattr(executor, "required_autonomy", 1),
                )
            )
    except Exception as exc:
        logger.warning("axon_executors_error", error=str(exc))

    return ExecutorRegistryResponse(executors=executors, total=len(executors))


@router.get("/safety", response_model=SafetyResponse)
async def get_safety(request: Request) -> SafetyResponse:
    """Circuit breaker and rate limiter states for all executors."""
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return SafetyResponse()

    pipeline = getattr(axon, "_pipeline", None)
    if pipeline is None:
        return SafetyResponse()

    circuit_breakers: list[CircuitBreakerState] = []
    rate_limiters: list[RateLimiterState] = []

    # Circuit breakers
    cb = getattr(pipeline, "_circuit_breaker", None)
    if cb is not None:
        states_dict = getattr(cb, "_states", {})
        for action_type, state in states_dict.items():
            circuit_breakers.append(
                CircuitBreakerState(
                    action_type=action_type,
                    status=state.status.value if hasattr(state.status, "value") else str(state.status),
                    consecutive_failures=getattr(state, "consecutive_failures", 0),
                    tripped_at=getattr(state, "tripped_at", None),
                )
            )

    # Rate limiters
    rl = getattr(pipeline, "_rate_limiter", None)
    if rl is not None:
        windows = getattr(rl, "_windows", {})
        for action_type, window_deque in windows.items():
            circuit_breakers_count = len(window_deque)
            rate_limiters.append(
                RateLimiterState(
                    action_type=action_type,
                    current_count=circuit_breakers_count,
                    window_seconds=60.0,
                )
            )

    return SafetyResponse(
        circuit_breakers=circuit_breakers,
        rate_limiters=rate_limiters,
        failure_threshold=5,
        recovery_timeout_s=300,
    )


@router.post("/safety/reset/{action_type}")
async def reset_circuit_breaker(
    request: Request,
    action_type: str,
) -> JSONResponse:
    """Manually reset a tripped circuit breaker for an executor."""
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return JSONResponse(status_code=503, content={"error": "Axon not available"})

    pipeline = getattr(axon, "_pipeline", None)
    cb = getattr(pipeline, "_circuit_breaker", None) if pipeline else None
    if cb is None:
        return JSONResponse(status_code=503, content={"error": "Circuit breaker not available"})

    try:
        cb.reset(action_type)
        logger.info("circuit_breaker_manual_reset", action_type=action_type)
        return JSONResponse(status_code=200, content={"reset": action_type, "status": "closed"})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.get("/shield", response_model=ShieldMetricsResponse)
async def get_shield(request: Request) -> ShieldMetricsResponse:
    """TransactionShield and MEV protection metrics."""
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return ShieldMetricsResponse()

    pipeline = getattr(axon, "_pipeline", None)
    shield = getattr(pipeline, "_shield", None) if pipeline else None
    if shield is None:
        return ShieldMetricsResponse()

    total_evaluated = getattr(shield, "_total_evaluated", 0)
    total_rejected = getattr(shield, "_total_rejected", 0)
    mev_protected = getattr(shield, "_total_mev_protected", 0)
    mev_saved = getattr(shield, "_total_mev_saved_usd", 0.0)
    rejection_rate = (total_rejected / total_evaluated) if total_evaluated > 0 else 0.0

    last_mev = getattr(shield, "_last_mev_report", None)
    last_risk = None
    last_strategy = None
    if last_mev is not None:
        last_risk = getattr(last_mev, "risk_score", None)
        strategy = getattr(last_mev, "recommended_strategy", None)
        if strategy is None:
            last_strategy = None
        elif hasattr(strategy, "value"):
            last_strategy = strategy.value
        else:
            last_strategy = str(strategy)

    blacklist = getattr(shield, "_blacklist", [])
    blacklisted_count = len(blacklist) if blacklist else 0

    return ShieldMetricsResponse(
        total_evaluated=total_evaluated,
        total_rejected=total_rejected,
        mev_protected=mev_protected,
        mev_saved_usd=round(mev_saved, 4),
        rejection_rate=round(rejection_rate, 4),
        last_mev_risk_score=round(last_risk, 4) if last_risk is not None else None,
        last_mev_strategy=last_strategy,
        blacklisted_addresses=blacklisted_count,
    )


@router.get("/fast-path", response_model=FastPathStatsResponse)
async def get_fast_path(request: Request) -> FastPathStatsResponse:
    """Fast-path (Arbitrage Reflex Arc) execution statistics."""
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return FastPathStatsResponse()

    fast_path = getattr(axon, "_fast_path", None)
    equor = getattr(request.app.state, "equor", None)

    stats: dict[str, Any] = {}
    if fast_path is not None:
        stats = getattr(fast_path, "stats", {})

    active_templates = 0
    if equor is not None:
        tl = getattr(equor, "template_library", None)
        if tl is not None:
            with contextlib.suppress(Exception):
                active_templates = len(tl.list_active())

    total = stats.get("total_executions", 0)
    total_latency = stats.get("total_latency_ms", 0)
    mean_latency = (total_latency / total) if total > 0 else 0.0

    return FastPathStatsResponse(
        total_executions=total,
        successful_executions=stats.get("successful_executions", 0),
        failed_executions=stats.get("failed_executions", 0),
        capital_deployed=round(stats.get("capital_deployed", 0.0), 4),
        mean_latency_ms=round(mean_latency, 1),
        active_templates=active_templates,
    )


# ─── Audit Trail ──────────────────────────────────────────────────


class AuditRecordResponse(EOSBaseModel):
    execution_id: str
    intent_id: str
    equor_verdict: str
    action_type: str
    parameters_hash: str
    result: str
    duration_ms: int
    autonomy_level: int
    affect_valence: float
    timestamp: str


class AuditResponse(EOSBaseModel):
    records: list[AuditRecordResponse] = []
    total: int = 0
    source: str = "outcomes"  # "memory" | "outcomes" | "empty"


@router.get("/audit", response_model=AuditResponse)
async def get_audit(request: Request, limit: int = 50) -> AuditResponse:
    """
    Recent audit records.

    Primary source: GovernanceRecord nodes in Neo4j (via Memory service).
    Fallback: reconstruct audit-like records from recent_outcomes ring buffer
    when Memory is unavailable or returns no records.

    Parameters are never returned raw — only their SHA-256 hash is stored.
    """
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return AuditResponse(source="empty")

    memory = getattr(request.app.state, "memory", None)
    if memory is not None and hasattr(memory, "query_governance_records"):
        try:
            raw_records: list[dict[str, Any]] = await memory.query_governance_records(
                record_type="action_audit",
                limit=limit,
            )
            if raw_records:
                records = [
                    AuditRecordResponse(
                        execution_id=r.get("execution_id", ""),
                        intent_id=r.get("intent_id", ""),
                        equor_verdict=r.get("equor_verdict", ""),
                        action_type=r.get("action_type", ""),
                        parameters_hash=r.get("parameters_hash", ""),
                        result=r.get("result", ""),
                        duration_ms=int(r.get("duration_ms", 0)),
                        autonomy_level=int(r.get("autonomy_level", 1)),
                        affect_valence=float(r.get("affect_valence", 0.0)),
                        timestamp=r.get("timestamp", ""),
                    )
                    for r in raw_records[:limit]
                ]
                return AuditResponse(records=records, total=len(records), source="memory")
        except Exception as exc:
            logger.debug("axon_audit_memory_query_failed", error=str(exc))

    recent: list[Any] = getattr(axon, "recent_outcomes", [])
    records = []
    for outcome in recent[:limit]:
        steps = getattr(outcome, "step_outcomes", [])
        action_type = steps[0].action_type if steps else "unknown"

        execution_id = getattr(outcome, "execution_id", "")
        parameters_hash = f"[audit-{execution_id[:8]}]"

        equor_verdict = (
            "approved"
            if getattr(outcome, "success", False) or getattr(outcome, "partial", False)
            else "blocked"
        )

        status = getattr(outcome, "status", None)
        result = status.value if hasattr(status, "value") else str(status)

        records.append(AuditRecordResponse(
            execution_id=execution_id,
            intent_id=getattr(outcome, "intent_id", ""),
            equor_verdict=equor_verdict,
            action_type=action_type,
            parameters_hash=parameters_hash,
            result=result,
            duration_ms=getattr(outcome, "duration_ms", 0),
            autonomy_level=1,
            affect_valence=0.0,
            timestamp="",
        ))

    return AuditResponse(records=records, total=len(records), source="outcomes")


# ─── SSE Stream ───────────────────────────────────────────────────

# In-process fan-out: one asyncio.Queue per connected SSE client.
_outcome_subscribers: list[asyncio.Queue[dict[str, Any]]] = []


def _notify_outcome_subscribers(payload: dict[str, Any]) -> None:
    """Push a serialized outcome to all connected SSE clients (non-blocking)."""
    for q in _outcome_subscribers:
        with contextlib.suppress(asyncio.QueueFull):
            q.put_nowait(payload)


def _outcome_to_dict(outcome: Any) -> dict[str, Any]:
    """Serialize an AxonOutcome to a JSON-safe dict for SSE."""
    steps = getattr(outcome, "step_outcomes", [])
    status = getattr(outcome, "status", None)
    return {
        "execution_id": getattr(outcome, "execution_id", ""),
        "intent_id": getattr(outcome, "intent_id", ""),
        "success": getattr(outcome, "success", False),
        "partial": getattr(outcome, "partial", False),
        "status": status.value if hasattr(status, "value") else str(status),
        "failure_reason": getattr(outcome, "failure_reason", ""),
        "duration_ms": getattr(outcome, "duration_ms", 0),
        "steps": [
            {
                "action_type": getattr(s, "action_type", ""),
                "success": getattr(s.result, "success", False) if hasattr(s, "result") else False,
                "duration_ms": getattr(s, "duration_ms", 0),
            }
            for s in steps
        ],
        "world_state_changes": getattr(outcome, "world_state_changes", []),
    }


def _wire_axon_stream(axon: Any) -> None:
    """
    Subscribe to ACTION_COMPLETED events on the Synapse event bus so the SSE
    endpoint receives outcomes in real-time without modifying AxonService.
    """
    event_bus = getattr(axon, "_event_bus", None)
    if event_bus is None:
        return
    try:
        from systems.synapse.types import SynapseEventType

        async def _on_action_completed(event: Any) -> None:
            data: dict[str, Any] = getattr(event, "data", {})
            payload: dict[str, Any] = {
                "execution_id": data.get("execution_id", ""),
                "intent_id": data.get("intent_id", ""),
                "success": data.get("success", False),
                "status": "success" if data.get("success") else "failure",
                "action_types": data.get("action_types", []),
                "duration_ms": data.get("duration_ms", 0),
                "outcome": data.get("outcome", ""),
            }
            _notify_outcome_subscribers(payload)

        event_bus.subscribe(SynapseEventType.ACTION_COMPLETED, _on_action_completed)
    except Exception as exc:
        logger.debug("axon_stream_wire_failed", error=str(exc))


@router.get("/stream")
async def stream_outcomes(request: Request) -> StreamingResponse:
    """
    Server-Sent Events stream of AxonOutcome events.

    Each SSE event has type ``outcome`` and carries execution metadata.
    A heartbeat comment is sent every 15s to keep proxies alive.
    Clients should reconnect on disconnect and fall back to polling /outcomes
    if EventSource is unavailable.
    """
    axon = getattr(request.app.state, "axon", None)

    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
    _outcome_subscribers.append(q)

    # Wire event bus subscriber once per app lifetime
    if axon is not None and not getattr(request.app.state, "_axon_stream_wired", False):
        _wire_axon_stream(axon)
        request.app.state._axon_stream_wired = True

    # Replay last 5 outcomes so the client has immediate data on connect
    if axon is not None:
        recent: list[Any] = getattr(axon, "recent_outcomes", [])
        for outcome in recent[-5:]:
            try:
                q.put_nowait(_outcome_to_dict(outcome))
            except asyncio.QueueFull:
                break

    async def generate():
        try:
            yield b": connected\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"event: outcome\ndata: {json.dumps(payload)}\n\n".encode()
                except TimeoutError:
                    yield b": heartbeat\n\n"
        finally:
            with contextlib.suppress(ValueError):
                _outcome_subscribers.remove(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── MEV Block Competition ────────────────────────────────────────


class MEVCompetitionResponse(EOSBaseModel):
    base_fee_gwei: float = 0.0
    priority_fee_gwei: float = 0.0
    pending_tx_count: int = 0
    competition_level: str = "UNKNOWN"  # "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
    competition_score: float = 0.0
    timestamp_ms: int = 0
    mev_enabled: bool = False


@router.get("/mev-competition", response_model=MEVCompetitionResponse)
async def get_mev_competition(request: Request) -> MEVCompetitionResponse:
    """
    Current block competition snapshot from MEVAnalyzer.

    Updated by BlockCompetitionMonitor (Atune) on every new block (~2s on Base L2).
    Returns gas fee levels, pending mempool size, and competition classification.
    """
    axon = getattr(request.app.state, "axon", None)
    if axon is None:
        return MEVCompetitionResponse()

    pipeline = getattr(axon, "_pipeline", None)
    shield = getattr(pipeline, "_shield", None) if pipeline else None
    mev_analyzer = getattr(shield, "_mev_analyzer", None) if shield else None

    # Also try direct attribute on axon service
    if mev_analyzer is None:
        mev_analyzer = getattr(axon, "_mev_analyzer", None)

    if mev_analyzer is None:
        return MEVCompetitionResponse(mev_enabled=False)

    snapshot = getattr(mev_analyzer, "_latest_competition", None)
    if snapshot is None:
        return MEVCompetitionResponse(mev_enabled=True)

    score: float = getattr(snapshot, "competition_level", 0.0)
    if score < 0.3:
        level = "LOW"
    elif score < 0.7:
        level = "MEDIUM"
    else:
        level = "HIGH"

    gas_price: float = getattr(snapshot, "gas_price_gwei", 0.0)
    base_fee: float = getattr(snapshot, "base_fee_gwei", 0.0)
    priority_fee: float = max(0.0, gas_price - base_fee)

    return MEVCompetitionResponse(
        base_fee_gwei=round(base_fee, 4),
        priority_fee_gwei=round(priority_fee, 4),
        pending_tx_count=getattr(snapshot, "pending_tx_count", 0),
        competition_level=level,
        competition_score=round(score, 4),
        timestamp_ms=getattr(snapshot, "timestamp_ms", 0),
        mev_enabled=True,
    )
