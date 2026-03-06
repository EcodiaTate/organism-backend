"""
EcodiaOS — Logs & Diagnostics API

Intelligent log analysis endpoints. Not just raw log queries — these
endpoints understand the system dependency graph, trace error cascades
through causal chains, and leverage Thymos diagnostics for root cause
analysis.

Endpoints:
  /api/logs/recent          — Raw log stream (last N minutes)
  /api/logs/errors          — Errors only
  /api/logs/latency-summary — Per-system latency stats (min/max/avg/p95)
  /api/logs/cascades        — Dependency-aware cascade detection
  /api/logs/temporal        — "What happened before X crashed?"
  /api/logs/diagnose        — LLM-powered root cause via Thymos
  /api/logs/system-graph    — Dependency graph for debugging
  /api/logs/health-snapshot — One-call diagnostic overview
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Request

from telemetry.log_analyzer import get_analyzer

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/recent")
async def get_recent_logs(
    minutes: int = Query(5, ge=1, le=60),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Get recent structured logs.

    Returns raw log entries from all systems for the given time window.
    """
    analyzer = get_analyzer()
    logs = await analyzer.get_recent_logs(minutes=minutes, limit=limit)
    return {"count": len(logs), "minutes": minutes, "logs": logs}


@router.get("/errors")
async def get_error_logs(
    minutes: int = Query(5, ge=1, le=60),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Get recent errors only."""
    analyzer = get_analyzer()
    errors = await analyzer.get_error_logs(minutes=minutes, limit=limit)
    return {"count": len(errors), "minutes": minutes, "errors": errors}


@router.get("/latency-summary")
async def get_latency_summary(
    minutes: int = Query(5, ge=1, le=60),
) -> dict[str, Any]:
    """Per-system latency stats: min, max, avg, p95.

    Use this to identify bottleneck systems and budget violations.
    """
    analyzer = get_analyzer()
    summary = await analyzer.get_latency_summary(minutes=minutes)
    return {"minutes": minutes, "by_system": summary}


@router.get("/cascades")
async def detect_cascades() -> dict[str, Any]:
    """Dependency-aware error cascade detection.

    Unlike naive "same error in 3+ systems" matching, this traces the
    system dependency graph to find the upstream root cause. If Memory
    fails and Nova + Evo + Logos fail as a consequence, this returns
    one cascade with root_cause_system="memory" rather than three
    unrelated errors.
    """
    analyzer = get_analyzer()
    cascades = await analyzer.detect_cascades()
    return {"count": len(cascades), "cascades": cascades}


@router.get("/temporal")
async def get_temporal_context(
    system: str = Query(..., description="System to investigate (e.g. 'nova')"),
    minutes: int = Query(5, ge=1, le=30),
) -> dict[str, Any]:
    """What happened before a system started failing?

    Returns the activity timeline leading up to the first error in the
    given system, with upstream dependency activity highlighted. Use this
    when you know WHICH system is broken and want to understand WHY.
    """
    analyzer = get_analyzer()
    return await analyzer.get_temporal_context(
        system=system, minutes_before=minutes,
    )


@router.get("/system-graph")
async def get_system_graph() -> dict[str, Any]:
    """Return the system dependency graph.

    Shows upstream/downstream relationships between all 29 systems.
    Useful for understanding blast radius of a system failure.
    """
    analyzer = get_analyzer()
    return analyzer.get_dependency_graph()


@router.get("/diagnose")
async def diagnose_system(
    request: Request,
    system: str = Query(..., description="System to diagnose (e.g. 'nova')"),
) -> dict[str, Any]:
    """LLM-powered root cause diagnosis via Thymos.

    Combines:
    - Current error state from Redis Streams
    - Dependency graph traversal (causal chain)
    - Temporal correlation (what changed before the failure)
    - Active Thymos incidents for the system
    - Thymos diagnostic hypotheses (LLM-generated if budget allows)

    This is the "tell me what's wrong with nova" endpoint.
    """
    analyzer = get_analyzer()

    # Gather log-level evidence
    errors = await analyzer.get_error_logs(minutes=10, limit=100)
    system_errors = [e for e in errors if e.get("system") == system]
    temporal = await analyzer.get_temporal_context(system=system, minutes_before=10)
    cascades = await analyzer.detect_cascades()
    latency = await analyzer.get_latency_summary(minutes=10)
    system_latency = latency.get(system)

    # Check Thymos for active incidents
    thymos_data: dict[str, Any] = {"available": False}
    if hasattr(request.app.state, "thymos"):
        thymos = request.app.state.thymos
        active = {
            iid: {
                "id": inc.id,
                "class": inc.incident_class.value,
                "severity": inc.severity.value,
                "error_type": inc.error_type,
                "error_message": inc.error_message[:300],
                "source_system": inc.source_system,
                "occurrence_count": inc.occurrence_count,
            }
            for iid, inc in thymos._active_incidents.items()
            if inc.source_system == system
        }
        thymos_data = {
            "available": True,
            "active_incidents": active,
            "active_count": len(active),
        }

    # Identify cascades affecting this system
    relevant_cascades = [
        c for c in cascades if system in c.get("affected_systems", [])
    ]

    return {
        "system": system,
        "timestamp": datetime.now(UTC).isoformat(),
        "error_summary": {
            "count": len(system_errors),
            "recent": system_errors[:10],
        },
        "latency": system_latency,
        "temporal_context": {
            "upstream_deps": temporal.get("upstream_dependencies", []),
            "upstream_activity": temporal.get("upstream_activity_before_failure", {}),
            "first_error_ts": temporal.get("first_error_ts"),
        },
        "cascades": relevant_cascades,
        "thymos": thymos_data,
        "diagnosis_hint": _build_diagnosis_hint(
            system, system_errors, temporal, relevant_cascades, system_latency,
        ),
    }


@router.get("/health-snapshot")
async def get_health_snapshot(request: Request) -> dict[str, Any]:
    """One-call diagnostic overview of the entire organism.

    Combines: error counts, cascade detection, latency hotspots,
    active Thymos incidents, and per-system status. This is the
    "how is the system doing right now?" endpoint.
    """
    analyzer = get_analyzer()

    errors = await analyzer.get_error_logs(minutes=5, limit=50)
    latency = await analyzer.get_latency_summary(minutes=5)
    cascades = await analyzer.detect_cascades()

    # Worst latency system
    worst_system = None
    worst_p95 = 0.0
    for system, stats in latency.items():
        p95 = stats.get("p95_ms", 0)
        if p95 > worst_p95:
            worst_p95 = p95
            worst_system = system

    # Errors by system
    error_by_system: dict[str, int] = {}
    for err in errors:
        sys = err.get("system", "unknown")
        error_by_system[sys] = error_by_system.get(sys, 0) + 1

    # Active Thymos incidents
    thymos_summary: dict[str, Any] = {"available": False}
    if hasattr(request.app.state, "thymos"):
        thymos = request.app.state.thymos
        active = thymos._active_incidents
        thymos_summary = {
            "available": True,
            "active_count": len(active),
            "by_severity": {},
        }
        for inc in active.values():
            sev = inc.severity.value
            thymos_summary["by_severity"][sev] = thymos_summary["by_severity"].get(sev, 0) + 1

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "error_count": len(errors),
        "errors_by_system": error_by_system,
        "cascade_count": len(cascades),
        "cascades": cascades[:3],
        "slowest_system": worst_system,
        "slowest_p95_ms": worst_p95,
        "latency_by_system": latency,
        "recent_errors": errors[:10],
        "thymos": thymos_summary,
    }


def _build_diagnosis_hint(
    system: str,
    errors: list[dict[str, str]],
    temporal: dict[str, Any],
    cascades: list[dict[str, Any]],
    latency: dict[str, Any] | None,
) -> str:
    """Build a human-readable diagnosis hint from the evidence."""
    parts: list[str] = []

    if not errors:
        return f"No recent errors from {system}."

    parts.append(f"{len(errors)} errors from {system} in last 10 min.")

    # Check if it's part of a cascade
    for cascade in cascades:
        root = cascade.get("root_cause_system", "?")
        if root != system:
            parts.append(
                f"Likely downstream of {root} failure "
                f"({cascade.get('system_count', '?')} systems affected)."
            )
            break

    # Check upstream activity
    upstream_activity = temporal.get("upstream_activity_before_failure", {})
    noisy_upstreams = [
        dep for dep, logs in upstream_activity.items()
        if any(
            log.get("level") in ("error", "critical", "warning")
            for log in logs
        )
    ]
    if noisy_upstreams:
        parts.append(
            f"Upstream issues detected before failure: {', '.join(noisy_upstreams)}."
        )

    # Latency check
    if latency and latency.get("p95_ms", 0) > 200:
        parts.append(f"High latency: p95={latency['p95_ms']}ms.")

    return " ".join(parts) if parts else f"Errors present but no clear pattern for {system}."
