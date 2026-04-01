"""EcodiaOS Mitosis REST Router - Phase 16e

Endpoints:
  GET  /api/v1/mitosis/status          - Reproductive fitness + metrics
  GET  /api/v1/mitosis/children        - All child instances
  GET  /api/v1/mitosis/children/{id}   - Single child detail + dividends
  GET  /api/v1/mitosis/dividends       - Full dividend history
  GET  /api/v1/mitosis/fleet           - Fleet-level population metrics
  GET  /api/v1/mitosis/config          - OikosConfig mitosis thresholds
  POST /api/v1/mitosis/evaluate        - Run fitness evaluation (no spawn)
  POST /api/v1/mitosis/spawn           - Enqueue spawn_child via Axon
  POST /api/v1/mitosis/terminate/{id}  - Terminate child container
  POST /api/v1/mitosis/fleet/rescue/{id} - Rescue a struggling child
"""
from __future__ import annotations

from datetime import UTC
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = structlog.get_logger("api.mitosis")
router = APIRouter()


def _dynamic_max_children(net_worth: Any) -> int:
    """Dynamic population cap: max(5, floor(net_worth / 1000))."""
    import math
    return max(5, math.floor(float(net_worth) / 1000))


# --- Request models ---


class SpawnRequest(BaseModel):
    niche_name: str
    niche_description: str = ""
    estimated_monthly_revenue_usd: str = "0.00"
    estimated_monthly_cost_usd: str = "0.00"
    competitive_density: str = "0.50"
    capability_alignment: str = "0.50"
    confidence: str = "0.70"
    child_wallet_address: str = ""


# --- Serializers ---


def _child_to_dict(child: Any) -> dict[str, Any]:
    return {
        "instance_id": child.instance_id,
        "niche": getattr(child, "niche", ""),
        "status": child.status.value if hasattr(child.status, "value") else str(child.status),
        "seed_capital_usd": str(getattr(child, "seed_capital_usd", "0.00")),
        "current_net_worth_usd": str(getattr(child, "current_net_worth_usd", "0.00")),
        "current_runway_days": str(getattr(child, "current_runway_days", "0")),
        "current_efficiency": str(getattr(child, "current_efficiency", "0.00")),
        "dividend_rate": str(getattr(child, "dividend_rate", "0.00")),
        "total_dividends_paid_usd": str(getattr(child, "total_dividends_paid_usd", "0.00")),
        "rescue_count": int(getattr(child, "rescue_count", 0)),
        "consecutive_positive_days": int(getattr(child, "consecutive_positive_days", 0)),
        "is_independent": bool(getattr(child, "is_independent", False)),
        "is_rescuable": bool(getattr(child, "is_rescuable", False)),
        "wallet_address": getattr(child, "wallet_address", ""),
        "container_id": getattr(child, "container_id", ""),
        "spawned_at": child.spawned_at.isoformat() if getattr(child, "spawned_at", None) else None,
        "last_health_report_at": (
            child.last_health_report_at.isoformat()
            if getattr(child, "last_health_report_at", None)
            else None
        ),
    }


def _dividend_to_dict(record: Any) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "child_instance_id": record.child_instance_id,
        "amount_usd": str(record.amount_usd),
        "tx_hash": record.tx_hash,
        "period_start": record.period_start.isoformat(),
        "period_end": record.period_end.isoformat(),
        "child_net_revenue_usd": str(record.child_net_revenue_usd),
        "dividend_rate_applied": str(record.dividend_rate_applied),
        "recorded_at": record.recorded_at.isoformat(),
    }


# --- Endpoints ---


@router.get("/api/v1/mitosis/status")
async def get_mitosis_status(request: Request) -> dict[str, Any]:
    """Reproductive fitness evaluation + summary metrics."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {
            "fit": False, "reasons": ["Oikos not initialized"],
            "runway_days": "0", "efficiency": "0.00", "net_worth": "0.00",
            "active_children": 0, "max_children": 5, "strategy_name": "unavailable",
            "dividend_history_count": 0, "total_dividends_received_usd": "0.00",
        }

    # OikosService stores MitosisEngine as self._mitosis (not _mitosis_engine)
    mitosis_engine = getattr(oikos, "_mitosis", None)
    if mitosis_engine is None:
        return {
            "fit": False, "reasons": ["MitosisEngine not initialized"],
            "runway_days": "0", "efficiency": "0.00", "net_worth": "0.00",
            "active_children": 0, "max_children": 5, "strategy_name": "unavailable",
            "dividend_history_count": 0, "total_dividends_received_usd": "0.00",
        }

    state = oikos.snapshot()
    dynamic_cap = _dynamic_max_children(state.total_net_worth)
    fitness = mitosis_engine.evaluate_fitness(state, max_children_override=dynamic_cap)

    # Children live on EconomicState.child_instances (no separate child_manager attr)
    children = state.child_instances
    active_statuses = {"spawning", "alive", "struggling", "rescued"}
    active_children = sum(
        1 for c in children
        if (c.status.value if hasattr(c.status, "value") else str(c.status)) in active_statuses
    )
    total_dividends = sum((r.amount_usd for r in mitosis_engine.dividend_history), Decimal("0"))

    return {
        "fit": fitness.fit,
        "reasons": fitness.reasons,
        "runway_days": str(fitness.runway_days),
        "efficiency": str(fitness.efficiency),
        "net_worth": str(fitness.net_worth),
        "active_children": active_children,
        "max_children": oikos._config.mitosis_max_children,
        "strategy_name": mitosis_engine.strategy.strategy_name,
        "dividend_history_count": len(mitosis_engine.dividend_history),
        "total_dividends_received_usd": str(total_dividends),
    }


@router.get("/api/v1/mitosis/children")
async def get_mitosis_children(request: Request) -> dict[str, Any]:
    """All child instances with full lifecycle state."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"children": [], "total": 0, "by_status": {}}

    # Children are on EconomicState.child_instances; snapshot() is a cheap read
    state = oikos.snapshot()
    children = state.child_instances
    serialized = [_child_to_dict(c) for c in children]

    by_status: dict[str, int] = {}
    for c in serialized:
        s = c["status"]
        by_status[s] = by_status.get(s, 0) + 1

    return {"children": serialized, "total": len(serialized), "by_status": by_status}


@router.get("/api/v1/mitosis/children/{child_id}")
async def get_mitosis_child(child_id: str, request: Request) -> dict[str, Any]:
    """Single child detail + dividend history for that child."""
    oikos = request.app.state.oikos
    if oikos is None:
        raise HTTPException(status_code=503, detail="Oikos not initialized")

    state = oikos.snapshot()
    match = next((c for c in state.child_instances if c.instance_id == child_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Child not found")

    # OikosService stores MitosisEngine as self._mitosis (not _mitosis_engine)
    mitosis_engine = getattr(oikos, "_mitosis", None)
    dividends: list[dict[str, Any]] = []
    if mitosis_engine is not None:
        dividends = [
            _dividend_to_dict(r)
            for r in mitosis_engine.dividend_history
            if r.child_instance_id == child_id
        ]

    return {"child": _child_to_dict(match), "dividends": dividends, "total_dividends": len(dividends)}


@router.get("/api/v1/mitosis/dividends")
async def get_mitosis_dividends(request: Request) -> dict[str, Any]:
    """Full dividend history across all children."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"dividends": [], "total": 0, "total_amount_usd": "0.00"}

    # OikosService stores MitosisEngine as self._mitosis (not _mitosis_engine)
    mitosis_engine = getattr(oikos, "_mitosis", None)
    if mitosis_engine is None:
        return {"dividends": [], "total": 0, "total_amount_usd": "0.00"}

    history = mitosis_engine.dividend_history
    total_amount = sum((r.amount_usd for r in history), Decimal("0"))
    return {
        "dividends": [_dividend_to_dict(r) for r in history],
        "total": len(history),
        "total_amount_usd": str(total_amount),
    }


@router.get("/api/v1/mitosis/fleet")
async def get_mitosis_fleet(request: Request) -> dict[str, Any]:
    """Population-level fleet metrics from FleetManager."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"metrics": None, "selection_records": [], "available": False}

    # OikosService stores FleetManager as self._fleet (not _fleet_manager)
    fleet_mgr = getattr(oikos, "_fleet", None)
    if fleet_mgr is None:
        return {"metrics": None, "selection_records": [], "available": False}

    metrics_dict: dict[str, Any] | None = None
    try:
        state = oikos.snapshot()
        m = fleet_mgr.get_metrics(state)
        metrics_dict = {
            "timestamp": m.timestamp.isoformat(),
            "total_children": m.total_children,
            "alive_count": m.alive_count,
            "struggling_count": m.struggling_count,
            "independent_count": m.independent_count,
            "dead_count": m.dead_count,
            "blacklisted_count": m.blacklisted_count,
            "total_fleet_net_worth": str(m.total_fleet_net_worth),
            "total_dividends_received": str(m.total_dividends_received),
            "avg_economic_ratio": str(m.avg_economic_ratio),
            "avg_runway_days": str(m.avg_runway_days),
            "role_distribution": m.role_distribution,
            "fit_count": m.fit_count,
            "underperforming_count": m.underperforming_count,
            "genome_eligible_count": m.genome_eligible_count,
        }
    except Exception as exc:
        logger.warning("fleet_metrics_unavailable", error=str(exc))

    records: list[dict[str, Any]] = []
    try:
        # FleetManager exposes .selection_history property (not recent_selection_records)
        raw = fleet_mgr.selection_history
        records = [
            {
                "record_id": r.record_id,
                "child_instance_id": r.child_instance_id,
                "verdict": r.verdict.value if hasattr(r.verdict, "value") else str(r.verdict),
                "economic_ratio": str(r.economic_ratio),
                "consecutive_negative_periods": r.consecutive_negative_periods,
                "role": r.role.value if hasattr(r.role, "value") else str(r.role),
                "timestamp": r.timestamp.isoformat(),
                "reason": r.reason,
            }
            for r in raw[-50:]
        ]
    except Exception as exc:
        logger.warning("fleet_selection_records_unavailable", error=str(exc))

    return {"metrics": metrics_dict, "selection_records": records, "available": True}


@router.get("/api/v1/mitosis/config")
async def get_mitosis_config(request: Request) -> dict[str, Any]:
    """Current OikosConfig mitosis thresholds (read-only)."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"available": False, "config": {}}

    cfg = getattr(oikos, "_config", None)
    if cfg is None:
        return {"available": False, "config": {}}

    return {
        "available": True,
        "config": {
            "mitosis_min_parent_runway_days": getattr(cfg, "mitosis_min_parent_runway_days", 180),
            "mitosis_min_seed_capital": str(getattr(cfg, "mitosis_min_seed_capital", "50.00")),
            "mitosis_max_seed_pct_of_net_worth": str(
                getattr(cfg, "mitosis_max_seed_pct_of_net_worth", "0.20")
            ),
            "mitosis_min_parent_efficiency": str(
                getattr(cfg, "mitosis_min_parent_efficiency", "1.5")
            ),
            "mitosis_default_dividend_rate": str(
                getattr(cfg, "mitosis_default_dividend_rate", "0.10")
            ),
            "mitosis_min_niche_score": str(getattr(cfg, "mitosis_min_niche_score", "0.4")),
            "mitosis_max_children": getattr(cfg, "mitosis_max_children", 5),
            "mitosis_child_struggling_runway_days": str(
                getattr(cfg, "mitosis_child_struggling_runway_days", "30.0")
            ),
            "mitosis_max_rescues_per_child": getattr(cfg, "mitosis_max_rescues_per_child", 2),
            "certificate_birth_validity_days": getattr(cfg, "certificate_birth_validity_days", 7),
        },
    }


@router.post("/api/v1/mitosis/evaluate")
async def trigger_evaluate(request: Request) -> dict[str, Any]:
    """Run reproductive fitness evaluation. Returns fitness + candidate seed config (no spawn)."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"fit": False, "reasons": ["Oikos not initialized"], "seed_config": None}

    # OikosService stores MitosisEngine as self._mitosis (not _mitosis_engine)
    mitosis_engine = getattr(oikos, "_mitosis", None)
    if mitosis_engine is None:
        return {"fit": False, "reasons": ["MitosisEngine not initialized"], "seed_config": None}

    state = oikos.snapshot()
    dynamic_cap = _dynamic_max_children(state.total_net_worth)
    fitness = mitosis_engine.evaluate_fitness(state, max_children_override=dynamic_cap)

    if not fitness.fit:
        return {"fit": False, "reasons": fitness.reasons, "seed_config": None}

    seed_config = None
    try:
        result = mitosis_engine.evaluate(state)
        if result is not None:
            seed_config = {
                "config_id": result.config_id,
                "child_instance_id": result.child_instance_id,
                "niche": {
                    "niche_id": result.niche.niche_id,
                    "name": result.niche.name,
                    "description": result.niche.description,
                    "estimated_monthly_revenue_usd": str(result.niche.estimated_monthly_revenue_usd),
                    "estimated_monthly_cost_usd": str(result.niche.estimated_monthly_cost_usd),
                    "competitive_density": str(result.niche.competitive_density),
                    "capability_alignment": str(result.niche.capability_alignment),
                    "confidence": str(result.niche.confidence),
                },
                "seed_capital_usd": str(result.seed_capital_usd),
                "dividend_rate": str(result.dividend_rate),
                "generation": result.generation,
                "belief_genome_id": result.belief_genome_id,
                "simula_genome_id": result.simula_genome_id,
                "created_at": result.created_at.isoformat(),
            }
    except Exception as exc:
        logger.warning("mitosis_evaluate_seed_config_failed", error=str(exc))

    return {"fit": True, "reasons": [], "seed_config": seed_config}


@router.post("/api/v1/mitosis/spawn")
async def trigger_spawn(body: SpawnRequest, request: Request) -> dict[str, Any]:
    """Enqueue spawn_child action via Axon (autonomy level 3 SOVEREIGN)."""
    oikos = request.app.state.oikos
    axon = getattr(request.app.state, "axon", None)

    if oikos is None:
        return {"status": "error", "message": "Oikos not initialized"}
    if axon is None:
        return {"status": "error", "message": "Axon not initialized"}

    # OikosService stores MitosisEngine as self._mitosis (not _mitosis_engine)
    mitosis_engine = getattr(oikos, "_mitosis", None)
    if mitosis_engine is None:
        return {"status": "error", "message": "MitosisEngine not initialized"}

    state = oikos.snapshot()
    dynamic_cap = _dynamic_max_children(state.total_net_worth)
    fitness = mitosis_engine.evaluate_fitness(state, max_children_override=dynamic_cap)
    if not fitness.fit:
        return {"status": "error", "message": "Parent not fit for reproduction", "reasons": fitness.reasons}

    import uuid
    from datetime import datetime

    from systems.oikos.models import EcologicalNiche

    try:
        niche = EcologicalNiche(
            niche_id=str(uuid.uuid4()),
            name=body.niche_name,
            description=body.niche_description,
            estimated_monthly_revenue_usd=Decimal(body.estimated_monthly_revenue_usd),
            estimated_monthly_cost_usd=Decimal(body.estimated_monthly_cost_usd),
            competitive_density=Decimal(body.competitive_density),
            capability_alignment=Decimal(body.capability_alignment),
            confidence=Decimal(body.confidence),
            discovered_at=datetime.now(UTC),
        )
    except Exception as exc:
        return {"status": "error", "message": f"Invalid niche parameters: {exc}"}

    try:
        seed_config = mitosis_engine.build_seed_config(state, niche)
    except Exception as exc:
        return {"status": "error", "message": f"Failed to build seed config: {exc}"}

    if seed_config is None:
        return {"status": "error", "message": "Seed config validation failed (financial constraints)"}

    # Axon has no enqueue_action(). Build an Intent + ExecutionRequest and call axon.execute().
    try:
        from primitives.common import AutonomyLevel, Verdict
        from primitives.intent import Action, ActionSequence, GoalDescriptor, Intent
        from systems.axon.types import ExecutionRequest

        action_params: dict[str, Any] = {
            "child_instance_id": seed_config.child_instance_id,
            "seed_capital_usd": str(seed_config.seed_capital_usd),
            "niche_name": niche.name,
            "niche_description": niche.description,
            "dividend_rate": str(seed_config.dividend_rate),
        }
        if body.child_wallet_address:
            action_params["child_wallet_address"] = body.child_wallet_address

        intent = Intent(
            goal=GoalDescriptor(
                description=f"Spawn child instance in niche '{niche.name}'",
                target_domain="mitosis",
            ),
            plan=ActionSequence(
                steps=[
                    Action(
                        executor="spawn_child",
                        parameters=action_params,
                        timeout_ms=120_000,
                    )
                ]
            ),
            autonomy_level_required=AutonomyLevel.STEWARD,
            autonomy_level_granted=AutonomyLevel.STEWARD,
        )

        # Route through Equor for constitutional review - no bypasses
        equor_service = getattr(request.app.state, "equor", None)
        if equor_service is not None:
            equor_check = await equor_service.review(intent)
            if equor_check.verdict == Verdict.DENIED:
                return {
                    "status": "denied",
                    "message": f"Constitutional review denied: {equor_check.reasoning}",
                    "intent_id": intent.id,
                }
        else:
            # Equor not initialized - fail closed, do not silently approve
            logger.warning("equor_not_available_for_spawn", intent_id=intent.id)
            return {
                "status": "error",
                "message": "Equor constitutional review service not available",
                "intent_id": intent.id,
            }

        outcome = await axon.execute(
            ExecutionRequest(
                intent=intent,
                equor_check=equor_check,
                timeout_ms=120_000,
            )
        )

        if outcome.success:
            logger.info(
                "mitosis_spawn_executed",
                child_instance_id=seed_config.child_instance_id,
                niche=niche.name,
                intent_id=intent.id,
            )
            return {
                "status": "queued",
                "intent_id": intent.id,
                "execution_id": outcome.execution_id,
                "child_instance_id": seed_config.child_instance_id,
                "seed_capital_usd": str(seed_config.seed_capital_usd),
                "niche": niche.name,
                "dividend_rate": str(seed_config.dividend_rate),
            }

        return {
            "status": "error",
            "message": outcome.error or outcome.failure_reason or "Axon execution failed",
            "intent_id": intent.id,
        }
    except Exception as exc:
        logger.exception("mitosis_spawn_execute_failed", error=str(exc))
        return {"status": "error", "message": str(exc)}


@router.post("/api/v1/mitosis/terminate/{child_id}")
async def terminate_child(child_id: str, request: Request) -> dict[str, Any]:
    """Gracefully terminate a child container."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "error", "message": "Oikos not initialized"}

    state = oikos.snapshot()
    match = next((c for c in state.child_instances if c.instance_id == child_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Child not found")

    container_id = getattr(match, "container_id", "")
    if not container_id:
        return {"status": "error", "message": "Child has no associated container ID"}

    # app.state.spawner is not registered in main.py.
    # The spawner is wired into SpawnChildExecutor inside Axon's executor registry.
    axon = getattr(request.app.state, "axon", None)
    spawner = None
    if axon is not None:
        spawn_executor = axon.get_executor("spawn_child")
        spawner = getattr(spawn_executor, "_spawner", None) if spawn_executor is not None else None

    if spawner is None:
        return {"status": "error", "message": "Spawner not available"}

    try:
        success = await spawner.terminate_child(container_id)
        if success:
            logger.info("mitosis_child_terminated", child_id=child_id, container_id=container_id)
            return {"status": "ok", "child_id": child_id, "container_id": container_id}
        return {"status": "error", "message": "Termination returned False"}
    except Exception as exc:
        logger.exception("mitosis_terminate_failed", child_id=child_id, error=str(exc))
        return {"status": "error", "message": str(exc)}


@router.post("/api/v1/mitosis/fleet/rescue/{child_id}")
async def rescue_child(child_id: str, request: Request) -> dict[str, Any]:
    """Execute a rescue transfer to restore a struggling child's runway to 60 days."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "error", "message": "Oikos not initialized"}

    state = oikos.snapshot()
    match = next((c for c in state.child_instances if c.instance_id == child_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Child not found")

    from systems.oikos.models import ChildStatus

    if match.status != ChildStatus.STRUGGLING:
        return {
            "status": "error",
            "message": f"Child is not struggling (current status: {match.status.value})",
        }

    # Get fleet_service from the SpawnChildExecutor
    axon = getattr(request.app.state, "axon", None)
    fleet_service = None
    if axon is not None:
        spawn_executor = axon.get_executor("spawn_child")
        fleet_service = getattr(spawn_executor, "_fleet_service", None) if spawn_executor is not None else None

    if fleet_service is None:
        return {"status": "error", "message": "MitosisFleetService not available"}

    try:
        success = await fleet_service.execute_rescue(match)
        if success:
            logger.info("mitosis_child_rescued", child_id=child_id)
            return {"status": "ok", "child_id": child_id, "message": "Rescue transfer executed"}
        return {"status": "error", "message": "Rescue failed (max rescues exceeded or transfer error)"}
    except Exception as exc:
        logger.exception("mitosis_rescue_failed", child_id=child_id, error=str(exc))
        return {"status": "error", "message": str(exc)}
