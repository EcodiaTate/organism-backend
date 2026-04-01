"""
EcodiaOS - SACM Orchestration Service

Public interface for submitting workloads to the Substrate-Arbitrage
Compute Mesh.  Other EcodiaOS systems (Axon, Nova, Oikos) interact with
SACM exclusively through the ``SACMClient`` facade.

Responsibilities:
  - submit():            Enqueue a workload for asynchronous execution.
  - await_result():      Block until a previously-submitted workload completes.
  - submit_and_await():  Convenience - submit + await in one call.

For Axon integration, see ``remote_compute_executor.py``, which provides
``RemoteComputeExecutor`` - an Axon-compatible Executor that bridges
the Axon action system to SACM.

Observability metrics are emitted via ``SACMMetrics`` (see below) so
every submission, placement, and completion is visible in the
telemetry pipeline.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import EOSBaseModel, Identified, Timestamped
from systems.sacm.optimizer import PlacementPlan, ScoredPlacement, optimize_placement
from systems.sacm.remote_executor import (
    PlacementDecision,
    RemoteExecutionManager,
    RemoteExecutionResult,
)
from systems.sacm.workload import (
    OffloadClass,
    ResourceEnvelope,
    WorkloadDescriptor,
    WorkloadPriority,
    WorkloadStatus,
)

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.sacm.oracle import ComputeMarketOracle

logger = structlog.get_logger("systems.sacm.service")


# ─── Observability ──────────────────────────────────────────────


class SACMMetrics:
    """
    Lightweight telemetry emitter for SACM Section XI metrics.

    Uses structlog structured events as the metric transport - the
    telemetry pipeline (TimescaleDB / Prometheus exporter) can scrape
    these events by ``metric`` field.

    Metric catalogue:
      sacm.cost.total_usd          - Cumulative USD spent on remote compute
      sacm.cost.estimated_usd      - Estimated cost at submission time
      sacm.cost.savings_usd        - Cumulative savings vs on-demand baseline
      sacm.workloads.submitted      - Total workloads submitted
      sacm.workloads.placed.remote  - Workloads placed on remote substrates
      sacm.workloads.completed      - Successfully completed workloads
      sacm.workloads.failed         - Failed workloads (execution or verification)
      sacm.workloads.rejected       - Workloads rejected by verification
      sacm.latency.submit_ms        - Time from submit() call to enqueue
      sacm.latency.e2e_ms           - End-to-end submit-to-result latency
      sacm.providers.active         - Number of active substrate providers
      sacm.verification.pass_rate   - Rolling verification pass rate
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger("systems.sacm.metrics")
        self._total_cost_usd: float = 0.0
        self._total_estimated_usd: float = 0.0
        self._total_savings_usd: float = 0.0
        self._submitted: int = 0
        self._placed_remote: int = 0
        self._completed: int = 0
        self._failed: int = 0
        self._rejected: int = 0
        self._verification_passes: int = 0
        self._verification_total: int = 0

    def record_submission(self, workload_id: str, estimated_cost_usd: float, submit_ms: float) -> None:
        self._submitted += 1
        self._total_estimated_usd += estimated_cost_usd
        self._log.info(
            "metric",
            metric="sacm.workloads.submitted",
            value=self._submitted,
            workload_id=workload_id,
        )
        self._log.info(
            "metric",
            metric="sacm.cost.estimated_usd",
            value=round(self._total_estimated_usd, 6),
            delta=round(estimated_cost_usd, 6),
        )
        self._log.info(
            "metric",
            metric="sacm.latency.submit_ms",
            value=round(submit_ms, 2),
            workload_id=workload_id,
        )

    def record_placement(self, workload_id: str, provider_id: str) -> None:
        self._placed_remote += 1
        self._log.info(
            "metric",
            metric="sacm.workloads.placed.remote",
            value=self._placed_remote,
            workload_id=workload_id,
            provider_id=provider_id,
        )

    def record_completion(
        self,
        workload_id: str,
        actual_cost_usd: float,
        savings_usd: float,
        e2e_ms: float,
        verified: bool,
    ) -> None:
        self._completed += 1
        self._total_cost_usd += actual_cost_usd
        self._total_savings_usd += savings_usd
        self._verification_total += 1
        if verified:
            self._verification_passes += 1

        self._log.info(
            "metric",
            metric="sacm.workloads.completed",
            value=self._completed,
            workload_id=workload_id,
        )
        self._log.info(
            "metric",
            metric="sacm.cost.total_usd",
            value=round(self._total_cost_usd, 6),
            delta=round(actual_cost_usd, 6),
        )
        self._log.info(
            "metric",
            metric="sacm.cost.savings_usd",
            value=round(self._total_savings_usd, 6),
            delta=round(savings_usd, 6),
        )
        self._log.info(
            "metric",
            metric="sacm.latency.e2e_ms",
            value=round(e2e_ms, 2),
            workload_id=workload_id,
        )
        pass_rate = (
            self._verification_passes / self._verification_total
            if self._verification_total > 0
            else 0.0
        )
        self._log.info(
            "metric",
            metric="sacm.verification.pass_rate",
            value=round(pass_rate, 4),
        )

    def record_failure(self, workload_id: str, reason: str) -> None:
        self._failed += 1
        self._log.warning(
            "metric",
            metric="sacm.workloads.failed",
            value=self._failed,
            workload_id=workload_id,
            reason=reason,
        )

    def record_rejection(self, workload_id: str, provider_id: str) -> None:
        self._rejected += 1
        self._verification_total += 1
        self._log.warning(
            "metric",
            metric="sacm.workloads.rejected",
            value=self._rejected,
            workload_id=workload_id,
            provider_id=provider_id,
        )
        pass_rate = (
            self._verification_passes / self._verification_total
            if self._verification_total > 0
            else 0.0
        )
        self._log.info(
            "metric",
            metric="sacm.verification.pass_rate",
            value=round(pass_rate, 4),
        )

    def record_active_providers(self, count: int) -> None:
        self._log.info(
            "metric",
            metric="sacm.providers.active",
            value=count,
        )

    @property
    def snapshot(self) -> dict[str, float | int]:
        """Current metric snapshot for health checks / admin endpoints."""
        return {
            "sacm.cost.total_usd": round(self._total_cost_usd, 6),
            "sacm.cost.estimated_usd": round(self._total_estimated_usd, 6),
            "sacm.cost.savings_usd": round(self._total_savings_usd, 6),
            "sacm.workloads.submitted": self._submitted,
            "sacm.workloads.placed.remote": self._placed_remote,
            "sacm.workloads.completed": self._completed,
            "sacm.workloads.failed": self._failed,
            "sacm.workloads.rejected": self._rejected,
            "sacm.verification.pass_rate": round(
                self._verification_passes / max(self._verification_total, 1), 4
            ),
        }


# ─── Workload History Store ─────────────────────────────────────


class WorkloadHistoryRecord(EOSBaseModel):
    """Persisted record of a completed (or failed) workload execution."""

    id: str
    offload_class: str = ""
    priority: str = ""
    status: str  # 'completed' | 'failed' | 'rejected'
    provider_id: str = ""
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    savings_usd: float = 0.0
    duration_s: float = 0.0
    verification_passed: bool | None = None
    consensus_score: float | None = None
    error_message: str | None = None
    submitted_at: float  # Unix epoch
    completed_at: float | None = None


class SACMWorkloadHistoryStore:
    """
    Redis-backed workload history store with in-memory fallback.

    Records are persisted to a Redis sorted set (scored by submitted_at)
    for continuity across restarts. Falls back to pure in-memory when
    Redis is unavailable.
    """

    _REDIS_KEY = "sacm:workload_history"

    def __init__(self, max_records: int = 500, redis_client: Any = None) -> None:
        self._max_records = max_records
        self._records: list[WorkloadHistoryRecord] = []
        self._redis: Any = redis_client  # RedisClient instance, wired after init
        self._log = structlog.get_logger("systems.sacm.history").bind(
            component="sacm.history"
        )

    def set_redis(self, redis_client: Any) -> None:
        """Wire Redis for persistent workload history."""
        self._redis = redis_client
        self._log.info("redis_wired_to_sacm_history")

    async def load_from_redis(self) -> int:
        """
        Reload history from Redis on startup.

        Returns the number of records loaded.
        """
        if self._redis is None:
            return 0

        try:
            import orjson

            client = self._redis.client
            raw_records = await client.zrange(
                self._redis._key(self._REDIS_KEY),
                0, self._max_records - 1,
                desc=True,
            )
            loaded = 0
            for raw in reversed(raw_records):
                data = orjson.loads(raw) if isinstance(raw, (str, bytes)) else raw
                self._records.append(WorkloadHistoryRecord(**data))
                loaded += 1

            self._log.info("history_loaded_from_redis", count=loaded)
            return loaded
        except Exception as exc:
            self._log.warning("redis_history_load_failed", error=str(exc))
            return 0

    def record(self, rec: WorkloadHistoryRecord) -> None:
        self._records.append(rec)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

        # Persist to Redis (fire-and-forget)
        if self._redis is not None:
            asyncio.create_task(
                self._persist_to_redis(rec),
                name=f"sacm_history_persist_{rec.id[:8]}",
            )

        self._log.info(
            "workload_history_recorded",
            workload_id=rec.id,
            status=rec.status,
            provider_id=rec.provider_id,
        )

    async def _persist_to_redis(self, rec: WorkloadHistoryRecord) -> None:
        """Write a single record to the Redis sorted set."""
        try:
            import orjson

            client = self._redis.client
            key = self._redis._key(self._REDIS_KEY)
            data = orjson.dumps(rec.model_dump(mode="json")).decode()
            await client.zadd(key, {data: rec.submitted_at})

            # Trim to max_records (keep highest scores = newest)
            count = await client.zcard(key)
            if count > self._max_records:
                await client.zremrangebyrank(key, 0, count - self._max_records - 1)
        except Exception as exc:
            self._log.warning(
                "redis_history_persist_failed",
                workload_id=rec.id,
                error=str(exc),
            )

    def query(
        self,
        limit: int = 50,
        status: str | None = None,
        provider_id: str | None = None,
    ) -> list[WorkloadHistoryRecord]:
        results = reversed(self._records)  # newest first
        out: list[WorkloadHistoryRecord] = []
        for r in results:
            if status and r.status != status:
                continue
            if provider_id and r.provider_id != provider_id:
                continue
            out.append(r)
            if len(out) >= limit:
                break
        return out

    def get(self, workload_id: str) -> WorkloadHistoryRecord | None:
        for r in self._records:
            if r.id == workload_id:
                return r
        return None


# ─── Submission Ticket ──────────────────────────────────────────


class SubmissionTicket(Identified, Timestamped):
    """
    Returned by submit() - a handle for tracking an in-flight workload.

    Callers use ticket.id with await_result() to collect the outcome.
    """

    workload_id: str
    estimated_cost_usd: float = 0.0
    provider_id: str = ""
    status: WorkloadStatus = WorkloadStatus.PENDING


# ─── SACMClient ─────────────────────────────────────────────────


class SACMClient:
    """
    Public interface for submitting workloads to SACM.

    Owns the oracle, execution manager, and metrics. Other systems
    call submit/await_result/submit_and_await - SACM handles
    placement, encryption, dispatch, and verification internally.

    Usage:
        client = SACMClient(
            oracle=oracle,
            execution_manager=execution_manager,
        )
        ticket = await client.submit(workload)
        result  = await client.await_result(ticket.id)
    """

    system_id: str = "sacm_client"

    def __init__(
        self,
        oracle: ComputeMarketOracle,
        execution_manager: RemoteExecutionManager,
        metrics: SACMMetrics | None = None,
        on_completion: asyncio.Event | None = None,
        history: SACMWorkloadHistoryStore | None = None,
    ) -> None:
        self._oracle = oracle
        self._execution_manager = execution_manager
        self._metrics = metrics or SACMMetrics()
        self._history = history
        self._log = logger.bind(component="sacm.client")

        # Synapse event bus - wired via set_synapse()
        self._synapse: Any = None

        # In-flight tracking: workload_id → asyncio.Future[RemoteExecutionResult]
        self._pending: dict[str, asyncio.Future[RemoteExecutionResult]] = {}
        # Completed results cache
        self._results: dict[str, RemoteExecutionResult] = {}
        # Submission timestamps for e2e latency
        self._submit_times: dict[str, float] = {}
        # Submission metadata for history: workload_id → (offload_class, priority, estimated_cost, submitted_at)
        self._submit_meta: dict[str, tuple[str, str, float, float]] = {}

    # ─── Synapse Integration ──────────────────────────────────────

    def set_synapse(self, synapse: Any) -> None:
        """
        Wire the SynapseService so SACM can emit workload lifecycle events.

        Called after Synapse is initialised (step 13 in main.py).
        Events emitted: COMPUTE_REQUEST_SUBMITTED on submit(),
        COMPUTE_REQUEST_ALLOCATED on successful placement.
        """
        self._synapse = synapse
        self._log.info("synapse_wired_to_sacm_client")

    async def _emit_synapse(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        """Fire-and-forget emit onto the Synapse event bus."""
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        asyncio.create_task(
            self._synapse.event_bus.emit(
                SynapseEvent(
                    event_type=event_type,
                    source_system="sacm",
                    data=data,
                )
            ),
            name=f"sacm_client_{event_type}_{data.get('workload_id', '')[:8]}",
        )

    async def health(self) -> dict[str, Any]:
        """ManagedSystemProtocol - health check for Synapse monitoring."""
        return {
            "status": "healthy",
            "pending": len(self._pending),
            "completed": len(self._results),
            "metrics": self._metrics.snapshot,
        }

    async def shutdown(self, drain_timeout_s: float = 30.0) -> None:
        """
        Graceful shutdown: emit SACM_DRAINING then wait up to drain_timeout_s
        for all in-flight workloads to complete before returning.

        Callers (main.py / SACMService) should call this before terminating
        the process so results are not silently lost.
        """
        if not self._pending:
            return

        await self._emit_synapse(SynapseEventType.SACM_DRAINING, {
            "pending_count": len(self._pending),
            "drain_timeout_s": drain_timeout_s,
        })

        self._log.info(
            "sacm_drain_started",
            pending_workloads=len(self._pending),
            drain_timeout_s=drain_timeout_s,
        )

        try:
            await asyncio.wait_for(
                self._drain_pending(),
                timeout=drain_timeout_s,
            )
            self._log.info("sacm_drain_complete", remaining=len(self._pending))
        except asyncio.TimeoutError:
            self._log.warning(
                "sacm_drain_timeout",
                remaining_pending=len(self._pending),
                timeout_s=drain_timeout_s,
            )

    async def _drain_pending(self) -> None:
        """Wait for all pending futures to resolve (complete or fail)."""
        if not self._pending:
            return
        await asyncio.gather(*self._pending.values(), return_exceptions=True)

    async def submit(self, workload: WorkloadDescriptor) -> SubmissionTicket:
        """
        Submit a workload for asynchronous remote execution.

        Steps:
          1. Snapshot the pricing surface from the oracle.
          2. Run the optimizer to produce a placement plan.
          3. Convert the best placement to a PlacementDecision.
          4. Kick off execution in a background task.
          5. Return a SubmissionTicket immediately.

        Raises ValueError if no feasible placement exists.
        """
        t0 = time.monotonic()
        wid = workload.workload_id

        self._log.info("submit_start", workload_id=wid, priority=workload.priority.name)

        # 1. Snapshot
        surface = self._oracle.snapshot()

        # 2. Optimize
        plan = optimize_placement(workload, surface)
        if not plan.has_feasible:
            self._metrics.record_failure(wid, "no_feasible_placement")
            raise ValueError(
                f"No feasible placement for workload {wid}: {plan.error or 'all offers exceed constraints'}"
            )

        best = plan.best
        assert best is not None  # guaranteed by has_feasible

        # 3. Build PlacementDecision from the best scored placement
        decision = PlacementDecision(
            provider_id=best.offer.provider_id,
            provider_public_key=best.offer.public_key if hasattr(best.offer, "public_key") else b"\x00" * 32,
            provider_endpoint=best.offer.endpoint if hasattr(best.offer, "endpoint") else "",
            estimated_cost_usd=best.raw_cost_usd,
            offer_id=best.offer.id,
        )

        estimated_cost = best.raw_cost_usd
        submit_ms = (time.monotonic() - t0) * 1000
        self._submit_times[wid] = t0
        self._submit_meta[wid] = (
            workload.offload_class.value if hasattr(workload.offload_class, "value") else str(workload.offload_class),
            workload.priority.name if hasattr(workload.priority, "name") else str(workload.priority),
            estimated_cost,
            time.time(),
        )

        self._metrics.record_submission(wid, estimated_cost, submit_ms)
        self._metrics.record_placement(wid, decision.provider_id)
        self._metrics.record_active_providers(len(surface.offers) if hasattr(surface, "offers") else 0)

        # 4. Launch background execution
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RemoteExecutionResult] = loop.create_future()
        self._pending[wid] = future

        asyncio.create_task(self._execute_and_resolve(workload, decision, future, plan))

        # 5. Emit Synapse event for the submission
        await self._emit_synapse(SynapseEventType.COMPUTE_REQUEST_SUBMITTED, {
            "workload_id": wid,
            "source": "sacm_client",
            "priority": workload.priority.name,
            "offload_class": workload.offload_class.value,
            "estimated_cost_usd": round(estimated_cost, 6),
            "provider_id": decision.provider_id,
        })

        # 6. Return ticket
        workload.status = WorkloadStatus.DISPATCHED
        return SubmissionTicket(
            workload_id=wid,
            estimated_cost_usd=estimated_cost,
            provider_id=decision.provider_id,
            status=WorkloadStatus.DISPATCHED,
        )

    async def await_result(
        self,
        workload_id: str,
        timeout_s: float = 600.0,
    ) -> RemoteExecutionResult:
        """
        Wait for a previously-submitted workload to complete.

        Returns the RemoteExecutionResult. Raises TimeoutError if the
        workload doesn't complete within timeout_s. Raises KeyError if
        the workload_id was never submitted through this client.
        """
        # Already completed?
        if workload_id in self._results:
            return self._results[workload_id]

        future = self._pending.get(workload_id)
        if future is None:
            raise KeyError(f"Unknown workload_id: {workload_id}")

        return await asyncio.wait_for(future, timeout=timeout_s)

    async def submit_and_await(
        self,
        workload: WorkloadDescriptor,
        timeout_s: float = 600.0,
    ) -> RemoteExecutionResult:
        """
        Submit a workload and wait for the result in one call.

        Convenience method combining submit() + await_result().
        """
        ticket = await self.submit(workload)
        return await self.await_result(ticket.workload_id, timeout_s=timeout_s)

    @property
    def metrics(self) -> SACMMetrics:
        """Access the metrics emitter for health checks."""
        return self._metrics

    @staticmethod
    def _scored_placement_to_decision(sp: ScoredPlacement) -> PlacementDecision:
        """Build a PlacementDecision from a ScoredPlacement."""
        return PlacementDecision(
            provider_id=sp.offer.provider_id,
            provider_public_key=(
                sp.offer.public_key if hasattr(sp.offer, "public_key") else b"\x00" * 32
            ),
            provider_endpoint=(
                sp.offer.endpoint if hasattr(sp.offer, "endpoint") else ""
            ),
            estimated_cost_usd=sp.raw_cost_usd,
            offer_id=sp.offer.id,
        )

    async def _execute_and_resolve(
        self,
        workload: WorkloadDescriptor,
        decision: PlacementDecision,
        future: asyncio.Future[RemoteExecutionResult],
        plan: PlacementPlan | None = None,
    ) -> None:
        """
        Background task: execute via RemoteExecutionManager, resolve the future.

        Retry logic: if the primary provider fails (not accepted), iterate
        through the remaining ranked placements from the optimizer plan (up to
        max_retries from the execution config) before declaring failure.
        """
        wid = workload.workload_id
        meta = self._submit_meta.pop(wid, None)
        offload_class_str, priority_str, estimated_cost, submitted_at = (
            meta if meta else ("", "", decision.estimated_cost_usd, time.time())
        )
        try:
            result = await self._execution_manager.execute(workload, decision)

            # ── Secondary provider retry ──────────────────────────────
            # If primary failed and we have a ranked plan with fallback placements,
            # iterate provider_ranking[1:] up to max_retries before giving up.
            if not result.accepted and plan is not None:
                feasible = plan.feasible_placements()
                max_retries = getattr(
                    getattr(self._execution_manager, "_config", None),
                    "max_retries",
                    1,
                )
                retry_count = 0
                for fallback_sp in feasible[1:]:
                    if retry_count >= max_retries:
                        break
                    retry_count += 1
                    fallback_decision = self._scored_placement_to_decision(fallback_sp)
                    self._log.info(
                        "retrying_on_secondary_provider",
                        workload_id=wid,
                        attempt=retry_count,
                        primary_provider=decision.provider_id,
                        fallback_provider=fallback_decision.provider_id,
                        primary_error=result.error,
                    )
                    await self._emit_synapse(SynapseEventType.COMPUTE_REQUEST_SUBMITTED, {
                        "workload_id": wid,
                        "source": "sacm_client_retry",
                        "priority": workload.priority.name,
                        "offload_class": workload.offload_class.value,
                        "estimated_cost_usd": round(fallback_decision.estimated_cost_usd, 6),
                        "provider_id": fallback_decision.provider_id,
                        "retry_attempt": retry_count,
                    })
                    retry_result = await self._execution_manager.execute(
                        workload, fallback_decision
                    )
                    if retry_result.accepted:
                        result = retry_result
                        decision = fallback_decision
                        break

            # Calculate e2e latency
            submit_time = self._submit_times.pop(wid, None)
            e2e_ms = (time.monotonic() - submit_time) * 1000 if submit_time else result.total_duration_ms
            duration_s = e2e_ms / 1000.0

            if result.accepted:
                savings = max(0.0, decision.estimated_cost_usd * 0.30)  # Baseline: 30% cloud premium avoided
                verified = (
                    result.consensus_report.outcome.value == "accepted"
                    if result.consensus_report
                    else False
                )
                consensus_score = (
                    result.consensus_report.weighted_score
                    if result.consensus_report
                    else None
                )
                self._metrics.record_completion(
                    workload_id=wid,
                    actual_cost_usd=decision.estimated_cost_usd,
                    savings_usd=savings,
                    e2e_ms=e2e_ms,
                    verified=verified,
                )
                if self._history is not None:
                    self._history.record(WorkloadHistoryRecord(
                        id=wid,
                        offload_class=offload_class_str,
                        priority=priority_str,
                        status="completed",
                        provider_id=decision.provider_id,
                        estimated_cost_usd=estimated_cost,
                        actual_cost_usd=decision.estimated_cost_usd,
                        savings_usd=savings,
                        duration_s=duration_s,
                        verification_passed=verified,
                        consensus_score=consensus_score,
                        submitted_at=submitted_at,
                        completed_at=time.time(),
                    ))
                await self._emit_synapse(SynapseEventType.COMPUTE_REQUEST_ALLOCATED, {
                    "workload_id": wid,
                    "provider_id": decision.provider_id,
                    "accepted": True,
                    "e2e_ms": round(e2e_ms, 1),
                    "cost_usd": round(decision.estimated_cost_usd, 6),
                })
            else:
                self._metrics.record_rejection(wid, decision.provider_id)
                if self._history is not None:
                    self._history.record(WorkloadHistoryRecord(
                        id=wid,
                        offload_class=offload_class_str,
                        priority=priority_str,
                        status="rejected",
                        provider_id=decision.provider_id,
                        estimated_cost_usd=estimated_cost,
                        duration_s=duration_s,
                        verification_passed=False,
                        consensus_score=(
                            result.consensus_report.weighted_score
                            if result.consensus_report
                            else None
                        ),
                        error_message=result.error,
                        submitted_at=submitted_at,
                        completed_at=time.time(),
                    ))
                await self._emit_synapse(SynapseEventType.COMPUTE_REQUEST_DENIED, {
                    "workload_id": wid,
                    "provider_id": decision.provider_id,
                    "reason": result.error or "verification_failed",
                })

            self._results[wid] = result
            self._pending.pop(wid, None)

            if not future.done():
                future.set_result(result)

        except Exception as exc:
            self._log.error(
                "execute_and_resolve_error",
                workload_id=wid,
                error=str(exc),
                exc_info=True,
            )
            self._metrics.record_failure(wid, str(exc))
            self._submit_times.pop(wid, None)
            if self._history is not None:
                self._history.record(WorkloadHistoryRecord(
                    id=wid,
                    offload_class=offload_class_str,
                    priority=priority_str,
                    status="failed",
                    provider_id=decision.provider_id,
                    estimated_cost_usd=estimated_cost,
                    error_message=str(exc),
                    submitted_at=submitted_at,
                    completed_at=time.time(),
                ))
            self._pending.pop(wid, None)

            if not future.done():
                future.set_exception(exc)
