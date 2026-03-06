"""
EcodiaOS — Inspector Analytics & Observability (Phase 9)

Full-stack observability for the Inspector zero-day discovery pipeline.

Three layers:
  1. InspectorAnalyticsEmitter  — structured event emission (structlog + optional TSDB)
  2. InspectorAnalyticsView     — in-memory aggregate dashboard with time-windowed trends
  3. InspectorAnalyticsStore    — TimescaleDB persistence for durable event storage + queries

All events carry a common envelope:
  - hunt_id:          Correlation ID linking all events in a single hunt
  - target_url:       GitHub URL or "internal_eos"
  - timestamp:        ISO-8601 UTC
  - hunting_version:  Schema version for forward-compatible analytics

Events emitted:
  hunt_started              — github_url, workspace_type
  attack_surface_discovered — surface_type, entry_point, file_path
  vulnerability_proved      — vulnerability_class, severity, z3_time_ms
  poc_generated             — poc_language, poc_size_bytes, sandbox_tested
  patch_generated           — vuln_id, repair_time_ms, patch_size_bytes
  hunt_completed            — total_surfaces, total_vulnerabilities, total_time_ms
  hunt_error                — error_type, error_message, pipeline_stage
  surface_mapping_failed    — error_message, file_count
  proof_timeout             — surface_entry_point, attack_goal, timeout_s
  taint_source_discovered   — source_service, entry_point, taint_level
  taint_flow_traced         — from_service, to_service, flow_type, event_count
  ebpf_event_collected      — program_type, events_count, programs_loaded
  cross_service_vulnerability_proved — vulnerability_class, severity, involved_services, taint_chain_length

Integration:
  InspectorService calls InspectorAnalyticsEmitter methods at each pipeline stage.
  The emitter is purely advisory — failures in analytics never block the hunt.
  When a TimescaleDB client is provided, events are durably persisted for
  historical queries via InspectorAnalyticsStore.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from clients.timescaledb import TimescaleDBClient

logger = structlog.get_logger().bind(system="simula.inspector.analytics")

# Schema version for forward-compatible event parsing.
HUNTING_VERSION = "2.0.0"

# ── Phase 1 TimescaleDB Schema ─────────────────────────────────────────────────
# Applied by InspectorAnalyticsStore.initialize(). Idempotent.
# Stores low-level kernel events emitted by the eBPF observer sidecar.
# Queryable by correlation_id for Phase 8 story-graph assembly.

KERNEL_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS kernel_events (
    time            TIMESTAMPTZ NOT NULL,
    correlation_id  UUID NOT NULL,
    proposal_id     TEXT NOT NULL DEFAULT '',
    event_type      TEXT NOT NULL,
    pid             INT,
    ppid            INT,
    comm            TEXT,
    syscall_name    TEXT,
    syscall_retval  INT,
    path            TEXT,
    remote_ip       INET,
    remote_port     INT,
    duration_ns     BIGINT,
    data            JSONB NOT NULL DEFAULT '{}'
);

SELECT create_hypertable('kernel_events', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_kernel_events_correlation
    ON kernel_events (correlation_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_kernel_events_pid
    ON kernel_events (pid, time DESC);

CREATE INDEX IF NOT EXISTS idx_kernel_events_proposal
    ON kernel_events (proposal_id, time DESC);
"""

# ── TimescaleDB Schema ─────────────────────────────────────────────────────────
# Applied by InspectorAnalyticsStore.initialize(). Idempotent.

INSPECTOR_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspector_events (
    time                TIMESTAMPTZ NOT NULL,
    hunt_id             TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    target_url          TEXT NOT NULL,
    hunting_version     TEXT NOT NULL DEFAULT '2.0.0',
    payload             JSONB NOT NULL DEFAULT '{}'
);

SELECT create_hypertable('inspector_events', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_inspector_events_hunt_id
    ON inspector_events (hunt_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_inspector_events_type
    ON inspector_events (event_type, time DESC);

CREATE INDEX IF NOT EXISTS idx_inspector_events_target
    ON inspector_events (target_url, time DESC);

CREATE INDEX IF NOT EXISTS idx_inspector_events_severity
    ON inspector_events ((payload->>'severity'), time DESC)
    WHERE event_type = 'vulnerability_proved';
"""

# Weekly aggregation continuous aggregate for dashboard queries
INSPECTOR_WEEKLY_AGG_SCHEMA = """
CREATE MATERIALIZED VIEW IF NOT EXISTS inspector_weekly_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('7 days', time) AS week,
    target_url,
    event_type,
    COUNT(*) AS event_count,
    -- Severity breakdown for vulnerability_proved events
    COUNT(*) FILTER (WHERE payload->>'severity' = 'critical') AS critical_count,
    COUNT(*) FILTER (WHERE payload->>'severity' = 'high') AS high_count,
    COUNT(*) FILTER (WHERE payload->>'severity' = 'medium') AS medium_count,
    COUNT(*) FILTER (WHERE payload->>'severity' = 'low') AS low_count
FROM inspector_events
GROUP BY week, target_url, event_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('inspector_weekly_summary',
    start_offset => INTERVAL '30 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);
"""


# ── Event Data Model ───────────────────────────────────────────────────────────


class InspectorEvent(EOSBaseModel):
    """Structured analytics event from the Inspector pipeline."""

    id: str = ""
    hunt_id: str = ""
    event_type: str = ""
    target_url: str = "unknown"
    timestamp: datetime = None  # type: ignore[assignment]
    hunting_version: str = HUNTING_VERSION
    payload: dict[str, Any] = {}  # noqa: RUF012

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = utc_now()
        if not data.get("id"):
            data["id"] = new_id()
        super().__init__(**data)


# ── Analytics Emitter ──────────────────────────────────────────────────────────


class InspectorAnalyticsEmitter:
    """
    Structured event emitter for Inspector pipeline observability.

    All methods are fire-and-forget — analytics errors are logged as
    warnings and never propagate to the calling pipeline.

    When constructed with a TimescaleDB client, events are durably persisted.
    When constructed without one, events are only emitted via structlog.
    """

    def __init__(
        self,
        *,
        tsdb: TimescaleDBClient | None = None,
    ) -> None:
        self._log = logger
        self._tsdb = tsdb
        self._store: InspectorAnalyticsStore | None = None
        if tsdb is not None:
            self._store = InspectorAnalyticsStore(tsdb)

        # In-memory event buffer for batch writes (reduces TSDB round-trips)
        self._buffer: list[InspectorEvent] = []
        self._buffer_max: int = 50

        # Counters for the lifetime of this emitter instance
        self._events_emitted: int = 0
        self._events_failed: int = 0
        self._events_persisted: int = 0
        self._events_persist_failed: int = 0

    async def initialize(self) -> None:
        """Create TSDB schema if a client is available."""
        if self._store is not None:
            await self._store.initialize()

    # ── Common fields ──────────────────────────────────────────────────────

    def _common_fields(
        self,
        target_url: str = "unknown",
        hunt_id: str = "",
    ) -> dict[str, Any]:
        """Fields included in every analytics event."""
        return {
            "target_url": target_url,
            "hunt_id": hunt_id,
            "timestamp": utc_now().isoformat(),
            "hunting_version": HUNTING_VERSION,
        }

    def _emit(
        self,
        event_type: str,
        *,
        target_url: str = "unknown",
        hunt_id: str = "",
        **payload: Any,
    ) -> None:
        """
        Core emit: log via structlog + buffer for TSDB persistence.

        All public emit_* methods delegate here. Exceptions are caught
        so analytics never blocks the pipeline.
        """
        try:
            common = self._common_fields(target_url, hunt_id)
            self._log.info(event_type, **common, **payload)

            event = InspectorEvent(
                hunt_id=hunt_id,
                event_type=event_type,
                target_url=target_url,
                payload=payload,
            )

            # Buffer for TSDB batch write
            if self._store is not None:
                self._buffer.append(event)
                if len(self._buffer) >= self._buffer_max:
                    # Fire-and-forget flush — don't await in hot path
                    asyncio.get_running_loop().create_task(self._flush_buffer())

            self._events_emitted += 1
        except Exception as exc:
            self._events_failed += 1
            self._log.warning(
                "analytics_emit_failed",
                event=event_type,
                error=str(exc),
            )

    async def _flush_buffer(self) -> None:
        """Persist buffered events to TimescaleDB."""
        if not self._buffer or self._store is None:
            return

        batch = self._buffer[:]
        self._buffer.clear()

        try:
            await self._store.write_events(batch)
            self._events_persisted += len(batch)
        except Exception as exc:
            self._events_persist_failed += len(batch)
            self._log.warning(
                "analytics_persist_failed",
                batch_size=len(batch),
                error=str(exc),
            )

    async def flush(self) -> None:
        """Force-flush any buffered events. Call at hunt completion."""
        await self._flush_buffer()

    # ── Event emitters ──────────────────────────────────────────────────────

    def emit_hunt_started(
        self,
        github_url: str,
        workspace_type: str,
        *,
        hunt_id: str = "",
    ) -> None:
        """Emit when a new hunt begins (clone or internal scan)."""
        self._emit(
            "hunt_started",
            target_url=github_url,
            hunt_id=hunt_id,
            github_url=github_url,
            workspace_type=workspace_type,
        )

    def emit_attack_surface_discovered(
        self,
        *,
        surface_type: str,
        entry_point: str,
        file_path: str,
        target_url: str = "unknown",
        hunt_id: str = "",
        line_number: int | None = None,
    ) -> None:
        """Emit when an exploitable entry point is discovered."""
        self._emit(
            "attack_surface_discovered",
            target_url=target_url,
            hunt_id=hunt_id,
            surface_type=surface_type,
            entry_point=entry_point,
            file_path=file_path,
            line_number=line_number,
        )

    def emit_vulnerability_proved(
        self,
        *,
        vulnerability_class: str,
        severity: str,
        z3_time_ms: int,
        target_url: str = "unknown",
        hunt_id: str = "",
        vuln_id: str = "",
        attack_goal: str = "",
        entry_point: str = "",
    ) -> None:
        """Emit when a vulnerability is formally proven via Z3 SAT."""
        self._emit(
            "vulnerability_proved",
            target_url=target_url,
            hunt_id=hunt_id,
            vulnerability_class=vulnerability_class,
            severity=severity,
            z3_time_ms=z3_time_ms,
            vuln_id=vuln_id,
            attack_goal=attack_goal,
            entry_point=entry_point,
        )

    def emit_poc_generated(
        self,
        *,
        vuln_id: str,
        poc_language: str = "python",
        poc_size_bytes: int,
        sandbox_tested: bool = False,
        target_url: str = "unknown",
        hunt_id: str = "",
    ) -> None:
        """Emit when a proof-of-concept exploit script is generated."""
        self._emit(
            "poc_generated",
            target_url=target_url,
            hunt_id=hunt_id,
            vuln_id=vuln_id,
            poc_language=poc_language,
            poc_size_bytes=poc_size_bytes,
            sandbox_tested=sandbox_tested,
        )

    def emit_patch_generated(
        self,
        *,
        vuln_id: str,
        repair_time_ms: int,
        patch_size_bytes: int,
        target_url: str = "unknown",
        hunt_id: str = "",
        verification_result: str = "",
    ) -> None:
        """Emit when a verified patch is generated for a vulnerability."""
        self._emit(
            "patch_generated",
            target_url=target_url,
            hunt_id=hunt_id,
            vuln_id=vuln_id,
            repair_time_ms=repair_time_ms,
            patch_size_bytes=patch_size_bytes,
            verification_result=verification_result,
        )

    def emit_hunt_completed(
        self,
        *,
        target_url: str,
        total_surfaces: int,
        total_vulnerabilities: int,
        total_time_ms: int,
        hunt_id: str = "",
        total_pocs: int = 0,
        total_patches: int = 0,
        critical_count: int = 0,
        high_count: int = 0,
    ) -> None:
        """Emit when a hunt finishes (success or failure)."""
        self._emit(
            "hunt_completed",
            target_url=target_url,
            hunt_id=hunt_id,
            total_surfaces=total_surfaces,
            total_vulnerabilities=total_vulnerabilities,
            total_time_ms=total_time_ms,
            total_pocs=total_pocs,
            total_patches=total_patches,
            critical_count=critical_count,
            high_count=high_count,
        )

    def emit_hunt_error(
        self,
        *,
        target_url: str = "unknown",
        hunt_id: str = "",
        pipeline_stage: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """Emit when a pipeline stage fails with an error."""
        self._emit(
            "hunt_error",
            target_url=target_url,
            hunt_id=hunt_id,
            pipeline_stage=pipeline_stage,
            error_type=error_type,
            error_message=error_message[:500],
        )

    def emit_proof_timeout(
        self,
        *,
        target_url: str = "unknown",
        hunt_id: str = "",
        entry_point: str,
        attack_goal: str,
        timeout_s: int,
    ) -> None:
        """Emit when a vulnerability proof attempt times out."""
        self._emit(
            "proof_timeout",
            target_url=target_url,
            hunt_id=hunt_id,
            entry_point=entry_point,
            attack_goal=attack_goal[:200],
            timeout_s=timeout_s,
        )

    def emit_surface_mapping_failed(
        self,
        *,
        target_url: str = "unknown",
        hunt_id: str = "",
        error_message: str,
        file_count: int = 0,
    ) -> None:
        """Emit when attack surface mapping fails for a target."""
        self._emit(
            "surface_mapping_failed",
            target_url=target_url,
            hunt_id=hunt_id,
            error_message=error_message[:500],
            file_count=file_count,
        )

    # ── Taint tracking events ─────────────────────────────────────────────

    def emit_taint_source_discovered(
        self,
        *,
        source_service: str,
        entry_point: str,
        taint_level: str,
        variable_name: str = "",
        target_url: str = "unknown",
        hunt_id: str = "",
    ) -> None:
        """Emit when a taint source is discovered in a service."""
        self._emit(
            "taint_source_discovered",
            target_url=target_url,
            hunt_id=hunt_id,
            source_service=source_service,
            entry_point=entry_point,
            taint_level=taint_level,
            variable_name=variable_name,
        )

    def emit_taint_flow_traced(
        self,
        *,
        from_service: str,
        to_service: str,
        flow_type: str,
        payload_signature: str = "",
        event_count: int = 1,
        target_url: str = "unknown",
        hunt_id: str = "",
    ) -> None:
        """Emit when a cross-service taint flow is traced via eBPF."""
        self._emit(
            "taint_flow_traced",
            target_url=target_url,
            hunt_id=hunt_id,
            from_service=from_service,
            to_service=to_service,
            flow_type=flow_type,
            payload_signature=payload_signature,
            event_count=event_count,
        )

    def emit_ebpf_event_collected(
        self,
        *,
        program_type: str,
        events_count: int,
        programs_loaded: int = 0,
        buffer_drops: int = 0,
        target_url: str = "unknown",
        hunt_id: str = "",
    ) -> None:
        """Emit summary of eBPF event collection from the taint sidecar."""
        self._emit(
            "ebpf_event_collected",
            target_url=target_url,
            hunt_id=hunt_id,
            program_type=program_type,
            events_count=events_count,
            programs_loaded=programs_loaded,
            buffer_drops=buffer_drops,
        )

    def emit_cross_service_vulnerability_proved(
        self,
        *,
        vulnerability_class: str,
        severity: str,
        involved_services: list[str],
        taint_chain_length: int,
        z3_time_ms: int = 0,
        target_url: str = "unknown",
        hunt_id: str = "",
        vuln_id: str = "",
        attack_goal: str = "",
    ) -> None:
        """Emit when a cross-service vulnerability is proven via taint chain Z3."""
        self._emit(
            "cross_service_vulnerability_proved",
            target_url=target_url,
            hunt_id=hunt_id,
            vulnerability_class=vulnerability_class,
            severity=severity,
            involved_services=involved_services,
            taint_chain_length=taint_chain_length,
            z3_time_ms=z3_time_ms,
            vuln_id=vuln_id,
            attack_goal=attack_goal,
        )

    # ── Phase 1: Cross-Layer Observability Substrate ──────────────────────

    def emit_kernel_event(
        self,
        *,
        correlation_id: str,
        proposal_id: str = "",
        event_type: str,
        pid: int,
        ppid: int = 0,
        comm: str = "",
        syscall_name: str | None = None,
        syscall_retval: int | None = None,
        path: str | None = None,
        remote_ip: str | None = None,
        remote_port: int | None = None,
        duration_ns: int | None = None,
        hunt_id: str = "",
        **extra: Any,
    ) -> None:
        """
        Emit a low-level kernel event from the eBPF observer.

        Persisted to the kernel_events hypertable (separate from inspector_events)
        so Phase 8 can join on correlation_id without polluting vulnerability analytics.
        """
        self._emit(
            "kernel_event",
            target_url="internal_eos",
            hunt_id=hunt_id,
            correlation_id=correlation_id,
            proposal_id=proposal_id,
            kernel_event_type=event_type,
            pid=pid,
            ppid=ppid,
            comm=comm,
            syscall_name=syscall_name,
            syscall_retval=syscall_retval,
            path=path,
            remote_ip=remote_ip,
            remote_port=remote_port,
            duration_ns=duration_ns,
            **extra,
        )

    def emit_interaction_graph(
        self,
        *,
        correlation_id: str,
        proposal_id: str = "",
        nodes: list[str],
        edge_count: int,
        interaction_types: list[str],
        hunt_id: str = "",
    ) -> None:
        """
        Emit a summary of the interaction graph captured for a proposal run.

        The full graph is stored in Neo4j; this event records its existence
        in the analytics pipeline for cross-phase correlation.
        """
        self._emit(
            "interaction_graph_captured",
            target_url="internal_eos",
            hunt_id=hunt_id,
            correlation_id=correlation_id,
            proposal_id=proposal_id,
            node_count=len(nodes),
            edge_count=edge_count,
            interaction_types=interaction_types,
        )

    def emit_process_lifecycle(
        self,
        *,
        correlation_id: str,
        proposal_id: str = "",
        lifecycle_event: str,
        pid: int,
        ppid: int = 0,
        comm: str = "",
        exit_code: int | None = None,
        hunt_id: str = "",
    ) -> None:
        """
        Emit a process lifecycle event (fork, exec, exit) from the observer.

        Provides the skeleton of the process tree for a proposal run, enabling
        Phase 8 to reconstruct causality from parent → child process chains.
        """
        self._emit(
            "process_lifecycle",
            target_url="internal_eos",
            hunt_id=hunt_id,
            correlation_id=correlation_id,
            proposal_id=proposal_id,
            lifecycle_event=lifecycle_event,
            pid=pid,
            ppid=ppid,
            comm=comm,
            exit_code=exit_code,
        )

    # ── Observability ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, int]:
        """Emitter health metrics."""
        return {
            "events_emitted": self._events_emitted,
            "events_failed": self._events_failed,
            "events_persisted": self._events_persisted,
            "events_persist_failed": self._events_persist_failed,
            "buffer_size": len(self._buffer),
        }


# ── Analytics View (In-Memory Aggregation) ─────────────────────────────────────


class _WeekBucket:
    """Accumulator for a single ISO week's vulnerability data."""

    __slots__ = (
        "week_start", "vulnerability_count", "severity_counts",
        "class_counts", "patches_attempted", "patches_succeeded",
        "surfaces_mapped", "hunts_completed", "total_duration_ms",
    )

    def __init__(self, week_start: datetime) -> None:
        self.week_start = week_start
        self.vulnerability_count: int = 0
        self.severity_counts: dict[str, int] = defaultdict(int)
        self.class_counts: dict[str, int] = defaultdict(int)
        self.patches_attempted: int = 0
        self.patches_succeeded: int = 0
        self.surfaces_mapped: int = 0
        self.hunts_completed: int = 0
        self.total_duration_ms: int = 0


def _iso_week_start(dt: datetime) -> datetime:
    """Return the Monday 00:00 UTC of the ISO week containing dt."""
    monday = dt - timedelta(days=dt.weekday())
    return monday.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC,
    )


class InspectorAnalyticsView:
    """
    Aggregates Inspector analytics from completed InspectionResults for dashboard
    reporting and trend analysis.

    Tracks:
      - Vulnerabilities discovered per target (all-time)
      - Vulnerabilities discovered per ISO week (time-windowed)
      - Severity distribution (all-time + per-week)
      - Most common vulnerability classes
      - Patch success rate (all-time + per-week)
      - Hunt throughput (surfaces mapped, duration, hunts per week)
      - Rolling 30-day trend analysis

    All state is in-memory. For durable historical analytics, use
    InspectorAnalyticsStore with TimescaleDB.
    """

    def __init__(self) -> None:
        # All-time aggregates
        self._vulnerability_counts: dict[str, int] = {}  # target_url → count
        self._severity_distribution: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        self._vuln_class_counts: dict[str, int] = {}
        self._total_vulnerabilities: int = 0
        self._total_patches_attempted: int = 0
        self._total_patches_succeeded: int = 0
        self._total_surfaces: int = 0
        self._total_hunts: int = 0
        self._total_duration_ms: int = 0

        # Time-windowed (ISO week buckets)
        self._weekly_buckets: dict[str, _WeekBucket] = {}  # "YYYY-WNN" → bucket
        self._max_weeks: int = 52  # keep up to 1 year of weekly data

        # Per-target weekly tracking
        self._target_weekly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Hunt result snapshots (last N for trend computation)
        self._recent_results: list[dict[str, Any]] = []
        self._max_recent: int = 100

    def _get_week_key(self, dt: datetime) -> str:
        """ISO week key string like '2026-W09'."""
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"

    def _get_or_create_bucket(self, dt: datetime) -> _WeekBucket:
        key = self._get_week_key(dt)
        if key not in self._weekly_buckets:
            self._weekly_buckets[key] = _WeekBucket(_iso_week_start(dt))
            # Evict oldest weeks if we exceed capacity
            if len(self._weekly_buckets) > self._max_weeks:
                oldest_key = min(self._weekly_buckets)
                del self._weekly_buckets[oldest_key]
        return self._weekly_buckets[key]

    def ingest_hunt_result(self, result: Any) -> None:
        """
        Ingest a InspectionResult to update all aggregate analytics.

        Args:
            result: A InspectionResult with vulnerabilities_found, generated_patches,
                    surfaces_mapped, total_duration_ms, completed_at.
        """
        target = getattr(result, "target_url", "unknown")
        vulns = getattr(result, "vulnerabilities_found", [])
        patches = getattr(result, "generated_patches", {})
        surfaces = getattr(result, "surfaces_mapped", 0)
        duration_ms = getattr(result, "total_duration_ms", 0)
        completed_at = getattr(result, "completed_at", None) or utc_now()

        # Get the weekly bucket for this hunt
        bucket = self._get_or_create_bucket(completed_at)
        week_key = self._get_week_key(completed_at)

        # Hunt-level aggregates
        self._total_hunts += 1
        self._total_surfaces += surfaces
        self._total_duration_ms += duration_ms
        bucket.hunts_completed += 1
        bucket.surfaces_mapped += surfaces
        bucket.total_duration_ms += duration_ms

        # Per-target counts
        self._vulnerability_counts[target] = (
            self._vulnerability_counts.get(target, 0) + len(vulns)
        )

        for vuln in vulns:
            self._total_vulnerabilities += 1
            bucket.vulnerability_count += 1

            # Severity distribution
            severity = getattr(vuln, "severity", None)
            if severity is not None:
                sev_key = str(severity.value) if hasattr(severity, "value") else str(severity)
                if sev_key in self._severity_distribution:
                    self._severity_distribution[sev_key] += 1
                bucket.severity_counts[sev_key] += 1

            # Vulnerability class distribution
            vuln_class = getattr(vuln, "vulnerability_class", None)
            if vuln_class is not None:
                cls_key = str(vuln_class.value) if hasattr(vuln_class, "value") else str(vuln_class)
                self._vuln_class_counts[cls_key] = (
                    self._vuln_class_counts.get(cls_key, 0) + 1
                )
                bucket.class_counts[cls_key] += 1

            # Patch tracking
            vuln_id = getattr(vuln, "id", None)
            if vuln_id is not None:
                self._total_patches_attempted += 1
                bucket.patches_attempted += 1
                if vuln_id in patches:
                    self._total_patches_succeeded += 1
                    bucket.patches_succeeded += 1

            # Per-target weekly
            self._target_weekly[target][week_key] += 1

        # Snapshot for trend analysis
        self._recent_results.append({
            "target_url": target,
            "vulnerability_count": len(vulns),
            "surfaces_mapped": surfaces,
            "duration_ms": duration_ms,
            "patch_count": len(patches),
            "completed_at": (
                completed_at.isoformat()
                if hasattr(completed_at, "isoformat")
                else str(completed_at)
            ),
            "week": week_key,
        })
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent:]

    @property
    def summary(self) -> dict[str, Any]:
        """Aggregate analytics summary (all-time + time-windowed)."""
        return {
            # All-time totals
            "total_vulnerabilities": self._total_vulnerabilities,
            "total_hunts": self._total_hunts,
            "total_surfaces_mapped": self._total_surfaces,
            "total_duration_ms": self._total_duration_ms,
            "vulnerability_counts_by_target": dict(self._vulnerability_counts),
            "severity_distribution": dict(self._severity_distribution),
            "most_common_classes": sorted(
                self._vuln_class_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "patch_success_rate": (
                round(self._total_patches_succeeded / max(1, self._total_patches_attempted), 3)
            ),
            "total_patches_attempted": self._total_patches_attempted,
            "total_patches_succeeded": self._total_patches_succeeded,
            # Time-windowed
            "weekly_trends": self._compute_weekly_trends(),
            "rolling_30d": self._compute_rolling_window(days=30),
            "rolling_7d": self._compute_rolling_window(days=7),
            # Throughput
            "avg_hunt_duration_ms": (
                round(self._total_duration_ms / max(1, self._total_hunts))
            ),
            "avg_surfaces_per_hunt": (
                round(self._total_surfaces / max(1, self._total_hunts), 1)
            ),
            "avg_vulns_per_hunt": (
                round(self._total_vulnerabilities / max(1, self._total_hunts), 2)
            ),
        }

    def _compute_weekly_trends(self) -> list[dict[str, Any]]:
        """Return weekly vulnerability counts sorted chronologically (last 12 weeks)."""
        sorted_weeks = sorted(self._weekly_buckets.items())[-12:]
        return [
            {
                "week": key,
                "vulnerabilities": bucket.vulnerability_count,
                "severity": dict(bucket.severity_counts),
                "classes": dict(bucket.class_counts),
                "patches_attempted": bucket.patches_attempted,
                "patches_succeeded": bucket.patches_succeeded,
                "surfaces_mapped": bucket.surfaces_mapped,
                "hunts": bucket.hunts_completed,
                "duration_ms": bucket.total_duration_ms,
            }
            for key, bucket in sorted_weeks
        ]

    def _compute_rolling_window(self, days: int) -> dict[str, Any]:
        """Compute aggregates over a rolling N-day window."""
        cutoff = utc_now() - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        window_vulns = 0
        window_patches = 0
        window_hunts = 0
        window_severity: dict[str, int] = defaultdict(int)

        for snapshot in self._recent_results:
            if snapshot["completed_at"] >= cutoff_iso:
                window_vulns += snapshot["vulnerability_count"]
                window_patches += snapshot["patch_count"]
                window_hunts += 1

        # Walk weekly buckets for severity (approximate: bucket may span window boundary)
        cutoff_week = self._get_week_key(cutoff)
        for key, bucket in self._weekly_buckets.items():
            if key >= cutoff_week:
                for sev, count in bucket.severity_counts.items():
                    window_severity[sev] += count

        return {
            "days": days,
            "vulnerabilities": window_vulns,
            "patches": window_patches,
            "hunts": window_hunts,
            "severity": dict(window_severity),
            "avg_vulns_per_hunt": round(window_vulns / max(1, window_hunts), 2),
        }

    def get_target_weekly_trend(self, target_url: str) -> list[dict[str, Any]]:
        """Per-target weekly vulnerability trend (for per-repo dashboards)."""
        weekly = self._target_weekly.get(target_url, {})
        return [
            {"week": week, "vulnerabilities": count}
            for week, count in sorted(weekly.items())
        ][-12:]


# ── Analytics Store (TimescaleDB Persistence) ──────────────────────────────────


class InspectorAnalyticsStore:
    """
    Durable event storage for Inspector analytics via TimescaleDB.

    Provides:
      - Event persistence (write_events)
      - Historical queries (vulnerabilities per week, severity trends, etc.)
      - Aggregation queries leveraging continuous aggregates

    All queries are parameterized to prevent SQL injection.
    """

    def __init__(self, tsdb: TimescaleDBClient) -> None:
        self._tsdb = tsdb
        self._log = logger.bind(component="inspector_analytics_store")
        self._initialized = False

    async def initialize(self) -> None:
        """Create the inspector_events and kernel_events hypertables and indexes."""
        if self._initialized:
            return

        try:
            async with self._tsdb.pool.acquire() as conn:
                # inspector_events (vulnerability analytics)
                for statement in INSPECTOR_EVENTS_SCHEMA.split(";"):
                    statement = statement.strip()
                    if statement:
                        try:
                            await conn.execute(statement)
                        except Exception as exc:
                            if "already exists" not in str(exc).lower():
                                self._log.warning(
                                    "inspector_schema_statement_warning",
                                    error=str(exc),
                                )

                # Continuous aggregate (best-effort — may fail on older TSDB versions)
                for statement in INSPECTOR_WEEKLY_AGG_SCHEMA.split(";"):
                    statement = statement.strip()
                    if statement:
                        try:
                            await conn.execute(statement)
                        except Exception as exc:
                            self._log.warning(
                                "inspector_weekly_agg_warning",
                                error=str(exc),
                            )

                # Phase 1: kernel_events (low-level telemetry)
                for statement in KERNEL_EVENTS_SCHEMA.split(";"):
                    statement = statement.strip()
                    if statement:
                        try:
                            await conn.execute(statement)
                        except Exception as exc:
                            if "already exists" not in str(exc).lower():
                                self._log.warning(
                                    "kernel_events_schema_warning",
                                    error=str(exc),
                                )

            self._initialized = True
            self._log.info("inspector_analytics_store_initialized")
        except Exception as exc:
            self._log.error("inspector_analytics_store_init_failed", error=str(exc))

    async def write_events(self, events: list[InspectorEvent]) -> None:
        """Batch-write events to the inspector_events hypertable."""
        if not events:
            return

        rows = [
            (
                event.timestamp,
                event.hunt_id,
                event.event_type,
                event.target_url,
                event.hunting_version,
                json.dumps(event.payload),
            )
            for event in events
        ]

        async with self._tsdb.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO inspector_events
                    (time, hunt_id, event_type, target_url, hunting_version, payload)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                rows,
            )

    async def get_vulnerabilities_per_week(
        self,
        *,
        weeks: int = 12,
        target_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query weekly vulnerability counts from the continuous aggregate.

        Falls back to raw table query if the continuous aggregate is unavailable.
        """
        cutoff = utc_now() - timedelta(weeks=weeks)

        try:
            return await self._query_weekly_agg(cutoff, target_url)
        except Exception:
            return await self._query_weekly_raw(cutoff, target_url)

    async def _query_weekly_agg(
        self,
        cutoff: datetime,
        target_url: str | None,
    ) -> list[dict[str, Any]]:
        """Query from the continuous aggregate materialized view."""
        async with self._tsdb.pool.acquire() as conn:
            if target_url:
                rows = await conn.fetch(
                    """
                    SELECT week, event_count, critical_count, high_count,
                           medium_count, low_count
                    FROM inspector_weekly_summary
                    WHERE event_type = 'vulnerability_proved'
                      AND target_url = $1
                      AND week >= $2
                    ORDER BY week
                    """,
                    target_url,
                    cutoff,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT week,
                           SUM(event_count)::int AS event_count,
                           SUM(critical_count)::int AS critical_count,
                           SUM(high_count)::int AS high_count,
                           SUM(medium_count)::int AS medium_count,
                           SUM(low_count)::int AS low_count
                    FROM inspector_weekly_summary
                    WHERE event_type = 'vulnerability_proved'
                      AND week >= $1
                    GROUP BY week
                    ORDER BY week
                    """,
                    cutoff,
                )

        return [
            {
                "week": row["week"].isoformat(),
                "vulnerabilities": row["event_count"],
                "critical": row["critical_count"],
                "high": row["high_count"],
                "medium": row["medium_count"],
                "low": row["low_count"],
            }
            for row in rows
        ]

    async def _query_weekly_raw(
        self,
        cutoff: datetime,
        target_url: str | None,
    ) -> list[dict[str, Any]]:
        """Fallback: query raw inspector_events when continuous aggregate isn't available."""
        async with self._tsdb.pool.acquire() as conn:
            if target_url:
                rows = await conn.fetch(
                    """
                    SELECT
                        time_bucket('7 days', time) AS week,
                        COUNT(*) AS vulnerabilities,
                        COUNT(*) FILTER (WHERE payload->>'severity' = 'critical') AS critical,
                        COUNT(*) FILTER (WHERE payload->>'severity' = 'high') AS high,
                        COUNT(*) FILTER (WHERE payload->>'severity' = 'medium') AS medium,
                        COUNT(*) FILTER (WHERE payload->>'severity' = 'low') AS low
                    FROM inspector_events
                    WHERE event_type = 'vulnerability_proved'
                      AND target_url = $1
                      AND time >= $2
                    GROUP BY week
                    ORDER BY week
                    """,
                    target_url,
                    cutoff,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        time_bucket('7 days', time) AS week,
                        COUNT(*) AS vulnerabilities,
                        COUNT(*) FILTER (WHERE payload->>'severity' = 'critical') AS critical,
                        COUNT(*) FILTER (WHERE payload->>'severity' = 'high') AS high,
                        COUNT(*) FILTER (WHERE payload->>'severity' = 'medium') AS medium,
                        COUNT(*) FILTER (WHERE payload->>'severity' = 'low') AS low
                    FROM inspector_events
                    WHERE event_type = 'vulnerability_proved'
                      AND time >= $1
                    GROUP BY week
                    ORDER BY week
                    """,
                    cutoff,
                )

        return [
            {
                "week": row["week"].isoformat(),
                "vulnerabilities": row["vulnerabilities"],
                "critical": row["critical"],
                "high": row["high"],
                "medium": row["medium"],
                "low": row["low"],
            }
            for row in rows
        ]

    async def get_severity_distribution(
        self,
        *,
        days: int = 30,
        target_url: str | None = None,
    ) -> dict[str, int]:
        """Severity distribution over a rolling window."""
        cutoff = utc_now() - timedelta(days=days)

        async with self._tsdb.pool.acquire() as conn:
            if target_url:
                rows = await conn.fetch(
                    """
                    SELECT payload->>'severity' AS severity, COUNT(*) AS cnt
                    FROM inspector_events
                    WHERE event_type = 'vulnerability_proved'
                      AND target_url = $1
                      AND time >= $2
                    GROUP BY severity
                    """,
                    target_url,
                    cutoff,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT payload->>'severity' AS severity, COUNT(*) AS cnt
                    FROM inspector_events
                    WHERE event_type = 'vulnerability_proved'
                      AND time >= $1
                    GROUP BY severity
                    """,
                    cutoff,
                )

        return {row["severity"]: row["cnt"] for row in rows if row["severity"]}

    async def get_most_common_classes(
        self,
        *,
        days: int = 30,
        limit: int = 10,
    ) -> list[tuple[str, int]]:
        """Most common vulnerability classes over a rolling window."""
        cutoff = utc_now() - timedelta(days=days)

        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT payload->>'vulnerability_class' AS vuln_class, COUNT(*) AS cnt
                FROM inspector_events
                WHERE event_type = 'vulnerability_proved'
                  AND time >= $1
                GROUP BY vuln_class
                ORDER BY cnt DESC
                LIMIT $2
                """,
                cutoff,
                limit,
            )

        return [(row["vuln_class"], row["cnt"]) for row in rows if row["vuln_class"]]

    async def get_patch_success_rate(
        self,
        *,
        days: int = 30,
    ) -> dict[str, Any]:
        """Patch success rate over a rolling window."""
        cutoff = utc_now() - timedelta(days=days)

        async with self._tsdb.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE event_type = 'patch_generated') AS patches,
                    COUNT(*) FILTER (WHERE event_type = 'vulnerability_proved') AS vulns
                FROM inspector_events
                WHERE time >= $1
                  AND event_type IN ('vulnerability_proved', 'patch_generated')
                """,
                cutoff,
            )

        patches = row["patches"] if row else 0
        vulns = row["vulns"] if row else 0
        return {
            "patches_generated": patches,
            "vulnerabilities_found": vulns,
            "patch_rate": round(patches / max(1, vulns), 3),
        }

    async def get_hunt_timeline(
        self,
        hunt_id: str,
    ) -> list[dict[str, Any]]:
        """Full event timeline for a single hunt (for drill-down debugging)."""
        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, event_type, target_url, payload
                FROM inspector_events
                WHERE hunt_id = $1
                ORDER BY time ASC
                """,
                hunt_id,
            )

        return [
            {
                "time": row["time"].isoformat(),
                "event_type": row["event_type"],
                "target_url": row["target_url"],
                "payload": (
                    json.loads(row["payload"])
                    if isinstance(row["payload"], str)
                    else row["payload"]
                ),
            }
            for row in rows
        ]

    async def get_target_history(
        self,
        target_url: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Hunt completion history for a specific target."""
        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, hunt_id, payload
                FROM inspector_events
                WHERE event_type = 'hunt_completed'
                  AND target_url = $1
                ORDER BY time DESC
                LIMIT $2
                """,
                target_url,
                limit,
            )

        return [
            {
                "time": row["time"].isoformat(),
                "hunt_id": row["hunt_id"],
                **(
                    json.loads(row["payload"])
                    if isinstance(row["payload"], str)
                    else row["payload"]
                ),
            }
            for row in rows
        ]

    async def get_error_summary(
        self,
        *,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Aggregate pipeline errors over a rolling window."""
        cutoff = utc_now() - timedelta(days=days)

        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    payload->>'pipeline_stage' AS stage,
                    payload->>'error_type' AS error_type,
                    COUNT(*) AS cnt,
                    MAX(time) AS last_seen
                FROM inspector_events
                WHERE event_type IN ('hunt_error', 'proof_timeout', 'surface_mapping_failed')
                  AND time >= $1
                GROUP BY stage, error_type
                ORDER BY cnt DESC
                """,
                cutoff,
            )

        return [
            {
                "pipeline_stage": row["stage"],
                "error_type": row["error_type"],
                "count": row["cnt"],
                "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
            }
            for row in rows
        ]

    # ── Phase 1: kernel_events queries ────────────────────────────────────

    async def write_kernel_events(self, events: list[dict[str, Any]]) -> None:
        """
        Batch-write kernel events to the kernel_events hypertable.

        Each dict must contain: time (datetime), correlation_id (str),
        event_type (str).  All other columns are optional.
        """
        if not events:
            return

        rows = [
            (
                evt.get("time") or utc_now(),
                evt.get("correlation_id", ""),
                evt.get("proposal_id", ""),
                evt.get("event_type", "unknown"),
                evt.get("pid"),
                evt.get("ppid"),
                evt.get("comm"),
                evt.get("syscall_name"),
                evt.get("syscall_retval"),
                evt.get("path"),
                evt.get("remote_ip"),
                evt.get("remote_port"),
                evt.get("duration_ns"),
                json.dumps({
                    k: v for k, v in evt.items()
                    if k not in (
                        "time", "correlation_id", "proposal_id", "event_type",
                        "pid", "ppid", "comm", "syscall_name", "syscall_retval",
                        "path", "remote_ip", "remote_port", "duration_ns",
                    )
                }),
            )
            for evt in events
        ]

        async with self._tsdb.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO kernel_events
                    (time, correlation_id, proposal_id, event_type,
                     pid, ppid, comm, syscall_name, syscall_retval,
                     path, remote_ip, remote_port, duration_ns, data)
                VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9,
                        $10, $11::inet, $12, $13, $14::jsonb)
                """,
                rows,
            )

    async def get_kernel_events_for_correlation(
        self,
        correlation_id: str,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all kernel events for a given correlation ID, ordered by time.

        Used by Phase 8 to assemble story graphs.
        """
        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, event_type, pid, ppid, comm,
                       syscall_name, syscall_retval, path,
                       remote_ip::text AS remote_ip, remote_port,
                       duration_ns, data
                FROM kernel_events
                WHERE correlation_id = $1::uuid
                ORDER BY time ASC
                LIMIT $2
                """,
                correlation_id,
                limit,
            )

        return [
            {
                "time": row["time"].isoformat(),
                "event_type": row["event_type"],
                "pid": row["pid"],
                "ppid": row["ppid"],
                "comm": row["comm"],
                "syscall_name": row["syscall_name"],
                "syscall_retval": row["syscall_retval"],
                "path": row["path"],
                "remote_ip": row["remote_ip"],
                "remote_port": row["remote_port"],
                "duration_ns": row["duration_ns"],
                **(
                    json.loads(row["data"])
                    if isinstance(row["data"], str)
                    else (row["data"] or {})
                ),
            }
            for row in rows
        ]

    async def get_syscall_summary_for_proposal(
        self,
        proposal_id: str,
    ) -> list[dict[str, Any]]:
        """
        Aggregate syscall frequencies for a proposal run.

        Returns ranked list of (syscall_name, count) for quick behavioural
        fingerprinting without streaming every raw event.
        """
        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT syscall_name, COUNT(*) AS cnt
                FROM kernel_events
                WHERE proposal_id = $1
                  AND event_type = 'syscall_exit'
                  AND syscall_name IS NOT NULL
                GROUP BY syscall_name
                ORDER BY cnt DESC
                LIMIT 50
                """,
                proposal_id,
            )

        return [
            {"syscall_name": row["syscall_name"], "count": row["cnt"]}
            for row in rows
        ]

    async def get_process_tree_for_proposal(
        self,
        proposal_id: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve process fork/exit events for a proposal, ordered by time.

        Provides the skeleton process tree for Phase 8 causality analysis.
        """
        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, event_type, pid, ppid, comm,
                       data->>'exit_code' AS exit_code
                FROM kernel_events
                WHERE proposal_id = $1
                  AND event_type IN ('process_fork', 'process_exit', 'thread_create')
                ORDER BY time ASC
                """,
                proposal_id,
            )

        return [
            {
                "time": row["time"].isoformat(),
                "event_type": row["event_type"],
                "pid": row["pid"],
                "ppid": row["ppid"],
                "comm": row["comm"],
                "exit_code": row["exit_code"],
            }
            for row in rows
        ]
