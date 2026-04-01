"""
EcodiaOS - Observatory API Router

Diagnostic endpoints for organism observability.
All endpoints are read-only - no side effects on the organism.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1/observatory", tags=["observatory"])


def _get_observatory(request: Request) -> dict[str, Any]:
    """Resolve observatory components from app state."""
    return {
        "tracer": getattr(request.app.state, "observatory_tracer", None),
        "closures": getattr(request.app.state, "observatory_closures", None),
        "spec_checker": getattr(request.app.state, "observatory_spec_checker", None),
        "event_bus": getattr(
            getattr(request.app.state, "synapse", None),
            "event_bus",
            None,
        ),
    }


@router.get("/snapshot")
async def observatory_snapshot(request: Request) -> dict[str, Any]:
    """Full diagnostic snapshot - the one-stop overview."""
    obs = _get_observatory(request)
    tracer = obs["tracer"]
    closures = obs["closures"]
    spec_checker = obs["spec_checker"]
    event_bus = obs["event_bus"]

    result: dict[str, Any] = {}

    if tracer:
        result["events"] = tracer.snapshot()

    if closures:
        result["closure_loops"] = closures.snapshot()

    if spec_checker:
        result["spec_compliance"] = spec_checker.check()

    if event_bus:
        result["bus_stats"] = event_bus.stats
        dlq = event_bus.dead_letter_queue
        result["dead_letters"] = {
            "count": len(dlq),
            "recent": [
                {
                    "event_type": ev.event_type.value,
                    "source": ev.source_system,
                    "timestamp": ev.timestamp.isoformat(),
                    "reason": reason,
                }
                for ev, reason in dlq[-20:]
            ],
        }

    return result


@router.get("/events")
async def observatory_events(
    request: Request,
    system: str | None = None,
    event_type: str | None = None,
) -> dict[str, Any]:
    """Event statistics, optionally filtered by system or type."""
    obs = _get_observatory(request)
    tracer = obs["tracer"]
    if not tracer:
        return {"error": "tracer not initialized"}

    snap = tracer.snapshot()

    if system:
        snap["per_system"] = {
            k: v for k, v in snap["per_system"].items()
            if k == system
        }
        snap["per_type"] = {
            k: v for k, v in snap["per_type"].items()
            if k in tracer._flow_edges.get(system, set())
        }

    if event_type:
        snap["per_type"] = {
            k: v for k, v in snap["per_type"].items()
            if k == event_type
        }

    return snap


@router.get("/flow")
async def observatory_flow(request: Request) -> dict[str, Any]:
    """Event flow graph: which systems emit which event types."""
    obs = _get_observatory(request)
    tracer = obs["tracer"]
    if not tracer:
        return {"error": "tracer not initialized"}

    return {"flow": tracer.flow_graph()}


@router.get("/closures")
async def observatory_closures(request: Request) -> dict[str, Any]:
    """Closure loop health status."""
    obs = _get_observatory(request)
    closures = obs["closures"]
    if not closures:
        return {"error": "closure tracker not initialized"}

    loops = closures.snapshot()
    summary = {
        "total": len(loops),
        "active": sum(1 for l in loops if l["status"] == "ACTIVE"),
        "stale": sum(1 for l in loops if l["status"] == "STALE"),
        "never_fired": sum(1 for l in loops if l["status"] == "NEVER_FIRED"),
        "critical_unhealthy": sum(
            1 for l in loops
            if l["is_critical"] and l["status"] != "ACTIVE"
        ),
    }
    return {"summary": summary, "loops": loops}


@router.get("/dead-letters")
async def observatory_dead_letters(
    request: Request,
    limit: int = 50,
) -> dict[str, Any]:
    """Dead letter queue contents."""
    obs = _get_observatory(request)
    event_bus = obs["event_bus"]
    if not event_bus:
        return {"error": "event bus not available"}

    dlq = event_bus.dead_letter_queue
    return {
        "total": len(dlq),
        "items": [
            {
                "event_id": ev.id,
                "event_type": ev.event_type.value,
                "source": ev.source_system,
                "timestamp": ev.timestamp.isoformat(),
                "data_keys": list(ev.data.keys())[:10],
                "reason": reason,
            }
            for ev, reason in dlq[-limit:]
        ],
    }


@router.get("/missing")
async def observatory_missing(request: Request) -> dict[str, Any]:
    """Events defined in spec but never observed."""
    obs = _get_observatory(request)
    spec_checker = obs["spec_checker"]
    if not spec_checker:
        return {"error": "spec checker not initialized"}

    return spec_checker.check()


@router.get("/vitality")
async def observatory_vitality(request: Request) -> dict[str, Any]:
    """Current vitality thresholds (if VitalityCoordinator is available)."""
    skia = getattr(request.app.state, "skia", None)
    if not skia:
        return {"error": "skia not available"}

    # Try to get vitality coordinator
    coordinator = getattr(skia, "_vitality_coordinator", None) or getattr(skia, "vitality_coordinator", None)
    if not coordinator:
        return {"error": "vitality coordinator not available"}

    # Try to get latest report
    report = getattr(coordinator, "latest_report", None)
    if report and callable(report):
        report = report()
    elif report is None:
        report = "no report yet"

    return {"vitality": report}
