"""
EcodiaOS — Oikos & Identity REST Router

Exposes the organism's economic state, active organs, certificate status,
and deployed assets to the Next.js frontend.

Endpoints:
  GET /api/v1/oikos/status                       — Full economic snapshot (net worth, BMR, runway, certificate)
  GET /api/v1/oikos/state                        — Alias kept for backwards compat
  GET /api/v1/oikos/metabolism                   — MVP: Live cost tracking (cost_per_hour, cost_per_day, monthly projection)
  GET /api/v1/oikos/yield-status                 — MVP: Yield vs cost (daily_yield, daily_cost, surplus_or_deficit, runway)
  GET /api/v1/oikos/budget-check                 — MVP: Budget authority check for a system action
  GET /api/v1/oikos/organs                       — Active economic organs from OrganLifecycleManager
  GET /api/v1/oikos/assets                       — Deployed assets and child instances
  GET /api/v1/oikos/certificate                  — Identity certificate status and days until expiry
  GET /api/v1/oikos/bounties                     — Active bounties and pipeline
  GET /api/v1/oikos/revenue-streams              — Revenue breakdown by stream (24h, 7d, 30d)
  GET /api/v1/oikos/fleet                        — Fleet metrics and member snapshots
  GET /api/v1/oikos/knowledge-market             — Knowledge market subscriptions, quotes, futures
  GET /api/v1/oikos/dream                        — Latest economic dream result (Monte Carlo)
  GET /api/v1/oikos/tollbooths                   — Deployed tollbooth contracts with on-chain accumulated revenue
  GET /api/v1/oikos/threat-model                 — Latest ThreatModelResult (Monte Carlo treasury analysis)
  GET /api/v1/oikos/history                      — Economic metric timeseries (last N days of snapshots)
  GET /api/v1/simula/assets                      — Alias kept for backwards compat
  POST /api/v1/oikos/genesis-spark               — Trigger genesis spark
  POST /api/v1/oikos/assets/{asset_id}/terminate — Terminate a deployed asset
  POST /api/v1/oikos/children/{instance_id}/rescue — Rescue a struggling child
  POST /api/v1/oikos/organs/{organ_id}/pause     — Pause an economic organ
  POST /api/v1/oikos/genesis-spark/reset         — Reset dormant state for testing
"""

from __future__ import annotations

import contextlib
from datetime import UTC
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog
from fastapi import APIRouter, Query, Request

logger = structlog.get_logger("api.oikos")

router = APIRouter()


@router.get("/api/v1/oikos/metabolism")
async def get_oikos_metabolism(request: Request) -> dict[str, Any]:
    """
    MVP Task 1 — Live cost tracking.

    Returns what EOS is currently spending on LLM API calls, sourced directly
    from Synapse's MetabolicTracker (which hooks into every LLM call).

    Persists the snapshot to Redis (eos:oikos:metabolism) on every read.

    Response:
      cost_per_hour_usd       — EMA-smoothed burn rate
      cost_per_day_usd        — cost_per_hour × 24
      projected_monthly_cost  — cost_per_day × 30
      per_system_cost_usd     — breakdown by system (nova, simula, evo, etc.)
      total_llm_calls         — total LLM calls since last revenue injection
      rolling_deficit_usd     — cumulative net spend vs injected revenue
    """
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "unavailable", "error": "Oikos not initialized"}

    try:
        snap = await oikos.get_metabolism_snapshot()
        return {"status": "ok", "data": snap.to_dict()}
    except Exception as exc:
        logger.error("metabolism_endpoint_error", error=str(exc))
        return {"status": "error", "error": str(exc)}


@router.get("/api/v1/oikos/yield-status")
async def get_oikos_yield_status(request: Request) -> dict[str, Any]:
    """
    MVP Task 2 — Yield strategy status.

    THE SINGLE METRIC THAT MATTERS: Can EOS generate enough yield to pay for
    its own LLM API calls?

    Fetches live APY from DeFiLlama (Aave V3 USDC on Base) if EOS_YIELD_API_URL
    is set, otherwise uses EOS_CONSERVATIVE_APY (default 4%) on EOS_CAPITAL_BASE_USD.

    Response:
      capital_base_usd        — principal deployed for yield (from EOS_CAPITAL_BASE_USD)
      current_apy             — live or configured APY (decimal, e.g. 0.042 = 4.2%)
      apy_source              — "defillama:..." or "configured_fallback"
      daily_yield_usd         — capital × APY / 365
      daily_cost_usd          — current burn rate × 24
      surplus_or_deficit_usd  — daily_yield - daily_cost (positive = self-sustaining)
      days_of_runway          — liquid_balance / daily_cost
      is_self_sustaining      — daily_yield >= daily_cost
    """
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "unavailable", "error": "Oikos not initialized"}

    try:
        snap = await oikos.get_yield_snapshot()
        return {"status": "ok", "data": snap.to_dict()}
    except Exception as exc:
        logger.error("yield_status_endpoint_error", error=str(exc))
        return {"status": "error", "error": str(exc)}


@router.get("/api/v1/oikos/budget-check")
async def get_oikos_budget_check(
    request: Request,
    system: str = Query(..., description="System requesting budget (e.g. nova, simula)"),
    action: str = Query(..., description="Action being taken (e.g. llm_call, web_fetch)"),
    estimated_cost: float = Query(..., description="Estimated cost in USD (e.g. 0.02)"),
) -> dict[str, Any]:
    """
    MVP Task 3 — Budget authority check.

    Systems call this before spending. Oikos enforces daily allocations derived
    from yield (or the configured floor from EOS_DAILY_BUDGET_FLOOR_USD).

    Every decision is logged to Redis for audit (eos:oikos:budget_audit).
    When a system exceeds its allocation, BUDGET_EXHAUSTED is emitted loudly
    on the Synapse event bus.

    Query params:
      system          — system ID (nova, simula, evo, axon, etc.)
      action          — what the spend is for (llm_call, web_search, etc.)
      estimated_cost  — USD estimate for this action

    Response:
      approved                — bool
      reason                  — why approved/denied
      remaining_daily_budget  — USD remaining for this system today
    """
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "unavailable", "error": "Oikos not initialized"}

    try:
        cost_decimal = Decimal(str(estimated_cost))
    except (InvalidOperation, ValueError):
        return {"status": "error", "error": f"Invalid estimated_cost: {estimated_cost}"}

    try:
        decision = await oikos.check_budget(
            system_id=system,
            action=action,
            estimated_cost_usd=cost_decimal,
        )
        return {"status": "ok", "data": decision.to_dict()}
    except Exception as exc:
        logger.error("budget_check_endpoint_error", error=str(exc))
        return {"status": "error", "error": str(exc)}


@router.get("/api/v1/oikos/state")
async def get_oikos_state(request: Request) -> dict[str, Any]:
    """Return the organism's current economic snapshot."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "unavailable", "error": "Oikos not initialized"}

    state = oikos.snapshot()
    return {
        "status": "ok",
        "data": {
            "total_net_worth": str(state.total_net_worth),
            "liquid_balance": str(state.liquid_balance),
            "survival_reserve": str(state.survival_reserve),
            "survival_reserve_target": str(state.survival_reserve_target),
            "total_deployed": str(state.total_deployed),
            "total_receivables": str(state.total_receivables),
            "total_asset_value": str(state.total_asset_value),
            "total_fleet_equity": str(state.total_fleet_equity),
            "derivative_liabilities": str(state.derivative_liabilities),
            "bmr_usd_per_hour": str(state.basal_metabolic_rate.usd_per_hour),
            "bmr_usd_per_day": str(state.basal_metabolic_rate.usd_per_day),
            "burn_rate_usd_per_hour": str(state.current_burn_rate.usd_per_hour),
            "runway_hours": str(state.runway_hours),
            "runway_days": str(state.runway_days),
            "starvation_level": state.starvation_level.value,
            "is_metabolically_positive": state.is_metabolically_positive,
            "metabolic_efficiency": str(state.metabolic_efficiency),
            "revenue_24h": str(state.revenue_24h),
            "revenue_7d": str(state.revenue_7d),
            "revenue_30d": str(state.revenue_30d),
            "costs_24h": str(state.costs_24h),
            "costs_7d": str(state.costs_7d),
            "costs_30d": str(state.costs_30d),
            "net_income_24h": str(state.net_income_24h),
            "net_income_7d": str(state.net_income_7d),
            "net_income_30d": str(state.net_income_30d),
            "economic_free_energy": str(state.economic_free_energy),
            "survival_probability_30d": str(state.survival_probability_30d),
            "instance_id": state.instance_id,
            "timestamp": state.timestamp.isoformat(),
        },
    }


@router.get("/api/v1/oikos/status")
async def get_oikos_status(request: Request) -> dict[str, Any]:
    """Return full economic snapshot with certificate — consumed by the Next.js frontend."""
    oikos = request.app.state.oikos
    cert_mgr = getattr(request.app.state, "certificate_manager", None)

    if oikos is None:
        # Return a zeroed-out snapshot so the frontend can render the dormant state.
        cert: dict[str, Any] = {
            "status": "none",
            "type": None,
            "issued_at": None,
            "expires_at": None,
            "remaining_days": -1.0,
            "lineage_hash": None,
            "instance_id": None,
        }
        zero = "0.00"
        return {
            "total_net_worth": zero,
            "liquid_balance": zero,
            "survival_reserve": zero,
            "survival_reserve_target": zero,
            "total_deployed": zero,
            "total_receivables": zero,
            "total_asset_value": zero,
            "total_fleet_equity": zero,
            "bmr_usd_per_day": zero,
            "burn_rate_usd_per_day": zero,
            "runway_days": zero,
            "starvation_level": "critical",
            "metabolic_efficiency": zero,
            "is_metabolically_positive": False,
            "revenue_24h": zero,
            "revenue_7d": zero,
            "costs_24h": zero,
            "costs_7d": zero,
            "net_income_24h": zero,
            "net_income_7d": zero,
            "survival_probability_30d": zero,
            "certificate": cert,
            "timestamp": "1970-01-01T00:00:00",
        }

    state = oikos.snapshot()

    # Build certificate sub-object
    if cert_mgr is not None and cert_mgr.certificate is not None:
        c = cert_mgr.certificate
        cert = {
            "status": c.status.value,
            "type": c.certificate_type.value,
            "issued_at": c.issued_at.isoformat() if c.issued_at else None,
            "expires_at": c.expires_at.isoformat() if c.expires_at else None,
            "remaining_days": cert_mgr.certificate_remaining_days,
            "lineage_hash": cert_mgr.lineage_hash,
            "instance_id": c.instance_id,
        }
    else:
        cert = {
            "status": "none",
            "type": None,
            "issued_at": None,
            "expires_at": None,
            "remaining_days": -1.0,
            "lineage_hash": None,
            "instance_id": None,
        }

    return {
        "total_net_worth": str(state.total_net_worth),
        "liquid_balance": str(state.liquid_balance),
        "survival_reserve": str(state.survival_reserve),
        "survival_reserve_target": str(state.survival_reserve_target),
        "total_deployed": str(state.total_deployed),
        "total_receivables": str(state.total_receivables),
        "total_asset_value": str(state.total_asset_value),
        "total_fleet_equity": str(state.total_fleet_equity),
        "bmr_usd_per_day": str(state.basal_metabolic_rate.usd_per_day),
        "burn_rate_usd_per_day": str(state.current_burn_rate.usd_per_day),
        "runway_days": str(state.runway_days),
        "starvation_level": state.starvation_level.value,
        "metabolic_efficiency": str(state.metabolic_efficiency),
        "is_metabolically_positive": state.is_metabolically_positive,
        "revenue_24h": str(state.revenue_24h),
        "revenue_7d": str(state.revenue_7d),
        "costs_24h": str(state.costs_24h),
        "costs_7d": str(state.costs_7d),
        "net_income_24h": str(state.net_income_24h),
        "net_income_7d": str(state.net_income_7d),
        "survival_probability_30d": str(state.survival_probability_30d),
        "certificate": cert,
        "timestamp": state.timestamp.isoformat(),
    }


@router.get("/api/v1/oikos/organs")
async def get_oikos_organs(request: Request) -> dict[str, Any]:
    """Return economic organs — consumed by the Next.js frontend (flat, no data wrapper)."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"organs": [], "active_count": 0, "total_count": 0, "stats": {}}

    active = oikos._morphogenesis.active_organs
    all_organs = oikos._morphogenesis.all_organs

    from datetime import datetime

    now = datetime.now(UTC)

    def _days_since(dt: Any) -> int:
        if dt is None:
            return 0
        try:
            delta = now - dt
            return max(0, delta.days)
        except Exception:
            return 0

    return {
        "organs": [
            {
                "organ_id": organ.organ_id,
                "category": organ.category.value,
                "specialisation": organ.specialisation,
                "maturity": organ.maturity.value,
                "resource_allocation_pct": str(organ.resource_allocation_pct),
                "revenue_30d": str(organ.revenue_30d),
                "cost_30d": str(organ.cost_30d),
                "efficiency": str(organ.efficiency),
                "days_since_last_revenue": _days_since(organ.last_revenue_at),
                "is_active": organ in active,
                "created_at": organ.created_at.isoformat(),
            }
            for organ in all_organs
        ],
        "active_count": len(active),
        "total_count": len(all_organs),
        "stats": {},
    }


@router.get("/api/v1/oikos/assets")
async def get_oikos_assets(request: Request) -> dict[str, Any]:
    """Return owned assets and child instances — consumed by the Next.js frontend."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {
            "owned_assets": [],
            "child_instances": [],
            "total_asset_value": "0.00",
            "total_fleet_equity": "0.00",
        }

    from datetime import datetime

    now = datetime.now(UTC)

    def _days_since(dt: Any) -> int:
        if dt is None:
            return 0
        try:
            return max(0, (now - dt).days)
        except Exception:
            return 0

    factory = oikos.asset_factory
    live = factory.get_live_assets()
    building = factory.get_building_assets()
    all_assets = list(live) + list(building)

    serialized_assets = [
        {
            "asset_id": a.asset_id,
            "name": getattr(a, "name", ""),
            "description": getattr(a, "description", ""),
            "asset_type": getattr(a, "asset_type", ""),
            "status": a.status.value if hasattr(a.status, "value") else str(a.status),
            "monthly_revenue_usd": str(getattr(a, "monthly_revenue_usd", "0.00")),
            "monthly_cost_usd": str(getattr(a, "monthly_cost_usd", "0.00")),
            "total_revenue_usd": str(getattr(a, "total_revenue_usd", "0.00")),
            "development_cost_usd": str(getattr(a, "development_cost_usd", "0.00")),
            "break_even_reached": bool(getattr(a, "break_even_reached", False)),
            "projected_break_even_days": int(getattr(a, "projected_break_even_days", 0)),
            "days_since_deployment": _days_since(getattr(a, "deployed_at", None)),
            "is_profitable": bool(getattr(a, "is_profitable", False)),
            "deployed_at": a.deployed_at.isoformat() if getattr(a, "deployed_at", None) else None,
            "compute_provider": getattr(a, "compute_provider", ""),
        }
        for a in all_assets
    ]

    # Child instances (spawned sub-organisms)
    child_mgr = getattr(oikos, "child_manager", None)
    children = child_mgr.all_children if child_mgr else []
    serialized_children = [
        {
            "instance_id": c.instance_id,
            "niche": getattr(c, "niche", ""),
            "status": c.status.value if hasattr(c.status, "value") else str(c.status),
            "seed_capital_usd": str(getattr(c, "seed_capital_usd", "0.00")),
            "current_net_worth_usd": str(getattr(c, "current_net_worth_usd", "0.00")),
            "current_runway_days": str(getattr(c, "current_runway_days", "0")),
            "current_efficiency": str(getattr(c, "current_efficiency", "0.00")),
            "dividend_rate": str(getattr(c, "dividend_rate", "0.00")),
            "total_dividends_paid_usd": str(getattr(c, "total_dividends_paid_usd", "0.00")),
            "is_independent": bool(getattr(c, "is_independent", False)),
            "spawned_at": c.spawned_at.isoformat() if getattr(c, "spawned_at", None) else None,
        }
        for c in children
    ]

    snap = oikos.snapshot()
    return {
        "owned_assets": serialized_assets,
        "child_instances": serialized_children,
        "total_asset_value": str(snap.total_asset_value),
        "total_fleet_equity": str(snap.total_fleet_equity),
    }


@router.post("/api/v1/oikos/genesis-spark")
async def trigger_genesis_spark(request: Request) -> dict[str, Any]:
    """Trigger the genesis spark to awaken a dormant organism."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "error", "message": "Oikos not initialized", "phases": {}}

    genesis = getattr(request.app.state, "genesis", None)
    if genesis is None:
        return {"status": "error", "message": "Genesis engine not available", "phases": {}}

    try:
        result = await genesis.spark()
        phases = getattr(result, "phases", {})
        if not isinstance(phases, dict):
            phases = {}
        return {
            "status": "ok" if getattr(result, "success", False) else "error",
            "message": getattr(result, "message", "Genesis spark triggered"),
            "phases": {k: bool(v) for k, v in phases.items()},
        }
    except Exception as exc:
        logger.exception("genesis_spark_failed", error=str(exc))
        return {"status": "error", "message": str(exc), "phases": {}}


@router.get("/api/v1/oikos/certificate")
async def get_oikos_certificate(request: Request) -> dict[str, Any]:
    """Return the organism's identity certificate status."""
    cert_mgr = request.app.state.certificate_manager
    if cert_mgr is None:
        return {
            "status": "ok",
            "data": {
                "initialized": False,
                "has_certificate": False,
                "certificate_status": None,
                "remaining_days": -1.0,
            },
        }

    cert = cert_mgr.certificate
    return {
        "status": "ok",
        "data": {
            "initialized": True,
            "has_certificate": cert is not None,
            "is_certified": cert_mgr.is_certified,
            "certificate_status": cert.status.value if cert else None,
            "remaining_days": cert_mgr.certificate_remaining_days,
            "certificate_type": cert.certificate_type.value if cert else None,
            "instance_id": cert.instance_id if cert else None,
            "issued_at": cert.issued_at.isoformat() if cert else None,
            "expires_at": cert.expires_at.isoformat() if cert else None,
            "lineage_hash": cert_mgr.lineage_hash,
        },
    }


@router.get("/api/v1/oikos/bounties")
async def get_oikos_bounties(request: Request) -> dict[str, Any]:
    """Return active bounties and pipeline from the economic state."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"bounties": [], "total_count": 0, "total_receivables_usd": "0.00"}

    state = oikos.snapshot()

    def _serialize_bounty(b: Any) -> dict[str, Any]:
        return {
            "bounty_id": b.bounty_id,
            "platform": getattr(b, "platform", ""),
            "title": getattr(b, "title", ""),
            "reward_usd": str(b.reward_usd),
            "estimated_cost_usd": str(getattr(b, "estimated_cost_usd", "0")),
            "actual_cost_usd": str(getattr(b, "actual_cost_usd", "0")),
            "net_reward_usd": str(getattr(b, "net_reward_usd", b.reward_usd)),
            "status": b.status.value if hasattr(b.status, "value") else str(b.status),
            "deadline": b.deadline.isoformat() if getattr(b, "deadline", None) else None,
            "pr_url": getattr(b, "pr_url", None),
            "submitted_at": b.submitted_at.isoformat() if getattr(b, "submitted_at", None) else None,
            "paid_at": b.paid_at.isoformat() if getattr(b, "paid_at", None) else None,
            "started_at": b.started_at.isoformat() if getattr(b, "started_at", None) else None,
        }

    bounties = list(state.active_bounties)
    return {
        "bounties": [_serialize_bounty(b) for b in bounties],
        "total_count": len(bounties),
        "total_receivables_usd": str(state.total_receivables),
    }


@router.get("/api/v1/oikos/revenue-streams")
async def get_oikos_revenue_streams(request: Request) -> dict[str, Any]:
    """Return revenue breakdown by stream including 24h/7d/30d income statement."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {
            "revenue_by_source": {},
            "revenue_24h": "0.00",
            "revenue_7d": "0.00",
            "revenue_30d": "0.00",
            "costs_24h": "0.00",
            "costs_7d": "0.00",
            "costs_30d": "0.00",
            "net_income_24h": "0.00",
            "net_income_7d": "0.00",
            "net_income_30d": "0.00",
            "bmr_breakdown": {},
        }

    state = oikos.snapshot()
    revenue_by_source = {
        stream.value: str(amount)
        for stream, amount in state.revenue_by_source.items()
    }
    bmr_breakdown = {
        k: str(v)
        for k, v in (state.basal_metabolic_rate.breakdown or {}).items()
    }
    return {
        "revenue_by_source": revenue_by_source,
        "revenue_24h": str(state.revenue_24h),
        "revenue_7d": str(state.revenue_7d),
        "revenue_30d": str(state.revenue_30d),
        "costs_24h": str(state.costs_24h),
        "costs_7d": str(state.costs_7d),
        "costs_30d": str(state.costs_30d),
        "net_income_24h": str(state.net_income_24h),
        "net_income_7d": str(state.net_income_7d),
        "net_income_30d": str(state.net_income_30d),
        "bmr_breakdown": bmr_breakdown,
    }


@router.get("/api/v1/oikos/fleet")
async def get_oikos_fleet(request: Request) -> dict[str, Any]:
    """Return fleet metrics and member snapshots from FleetManager."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {
            "metrics": {
                "total_children": 0,
                "alive_count": 0,
                "struggling_count": 0,
                "independent_count": 0,
                "dead_count": 0,
                "blacklisted_count": 0,
                "total_fleet_net_worth": "0.00",
                "total_dividends_received": "0.00",
                "avg_economic_ratio": "0.00",
                "avg_runway_days": "0.00",
                "fit_count": 0,
                "underperforming_count": 0,
                "genome_eligible_count": 0,
                "role_distribution": {},
            },
            "members": [],
            "recent_selections": [],
        }

    fleet_mgr = getattr(oikos, "_fleet_manager", None)
    if fleet_mgr is None:
        fleet_mgr = getattr(oikos, "fleet_manager", None)

    if fleet_mgr is None:
        state = oikos.snapshot()
        return {
            "metrics": {
                "total_children": len(state.child_instances),
                "alive_count": sum(1 for c in state.child_instances if getattr(c, "status", None) and c.status.value in ("alive", "rescued")),
                "struggling_count": sum(1 for c in state.child_instances if getattr(c, "status", None) and c.status.value == "struggling"),
                "independent_count": sum(1 for c in state.child_instances if getattr(c, "status", None) and c.status.value == "independent"),
                "dead_count": sum(1 for c in state.child_instances if getattr(c, "status", None) and c.status.value == "dead"),
                "blacklisted_count": 0,
                "total_fleet_net_worth": str(state.total_fleet_equity),
                "total_dividends_received": "0.00",
                "avg_economic_ratio": "0.00",
                "avg_runway_days": "0.00",
                "fit_count": 0,
                "underperforming_count": 0,
                "genome_eligible_count": 0,
                "role_distribution": {},
            },
            "members": [],
            "recent_selections": [],
        }

    metrics = fleet_mgr.get_metrics()
    members = fleet_mgr.get_member_snapshots() if hasattr(fleet_mgr, "get_member_snapshots") else []
    recent_selections = fleet_mgr.get_recent_selections(limit=20) if hasattr(fleet_mgr, "get_recent_selections") else []

    def _serialize_metrics(m: Any) -> dict[str, Any]:
        return {
            "total_children": m.total_children,
            "alive_count": m.alive_count,
            "struggling_count": m.struggling_count,
            "independent_count": m.independent_count,
            "dead_count": m.dead_count,
            "blacklisted_count": getattr(m, "blacklisted_count", 0),
            "total_fleet_net_worth": str(m.total_fleet_net_worth),
            "total_dividends_received": str(m.total_dividends_received),
            "avg_economic_ratio": str(m.avg_economic_ratio),
            "avg_runway_days": str(m.avg_runway_days),
            "fit_count": m.fit_count,
            "underperforming_count": m.underperforming_count,
            "genome_eligible_count": m.genome_eligible_count,
            "role_distribution": {
                role.value if hasattr(role, "value") else str(role): count
                for role, count in m.role_distribution.items()
            },
        }

    def _serialize_member(m: Any) -> dict[str, Any]:
        return {
            "instance_id": m.instance_id,
            "niche": getattr(m, "niche", ""),
            "role": m.role.value if hasattr(m.role, "value") else str(getattr(m, "role", "generalist")),
            "status": m.status.value if hasattr(m.status, "value") else str(m.status),
            "economic_ratio": str(getattr(m, "economic_ratio", "0")),
            "net_worth_usd": str(getattr(m, "net_worth_usd", "0")),
            "runway_days": str(getattr(m, "runway_days", "0")),
            "consecutive_positive_days": getattr(m, "consecutive_positive_days", 0),
            "rescue_count": getattr(m, "rescue_count", 0),
            "total_dividends_paid_usd": str(getattr(m, "total_dividends_paid_usd", "0")),
            "spawned_at": m.spawned_at.isoformat() if getattr(m, "spawned_at", None) else None,
        }

    def _serialize_selection(s: Any) -> dict[str, Any]:
        return {
            "child_instance_id": s.child_instance_id,
            "verdict": s.verdict.value if hasattr(s.verdict, "value") else str(s.verdict),
            "economic_ratio": str(s.economic_ratio),
            "role": s.role.value if hasattr(s.role, "value") else str(getattr(s, "role", "")),
            "reason": getattr(s, "reason", ""),
            "timestamp": s.timestamp.isoformat() if getattr(s, "timestamp", None) else None,
        }

    return {
        "metrics": _serialize_metrics(metrics),
        "members": [_serialize_member(m) for m in members],
        "recent_selections": [_serialize_selection(s) for s in recent_selections],
    }


@router.get("/api/v1/oikos/knowledge-market")
async def get_oikos_knowledge_market(request: Request) -> dict[str, Any]:
    """Return knowledge market state: subscriptions, active futures, capacity utilization."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {
            "subscriptions": [],
            "active_futures": [],
            "subscription_capacity_pct": "0.00",
            "derivatives_capacity_pct": "0.00",
            "combined_capacity_pct": "0.00",
            "derivative_liabilities_usd": "0.00",
            "recent_sales": [],
        }

    km = getattr(oikos, "_knowledge_market", None)
    if km is None:
        km = getattr(oikos, "knowledge_market", None)

    deriv_mgr = getattr(oikos, "_derivatives_manager", None)
    if deriv_mgr is None:
        deriv_mgr = getattr(oikos, "derivatives_manager", None)

    state = oikos.snapshot()

    def _serialize_future(f: Any) -> dict[str, Any]:
        return {
            "contract_id": f.contract_id,
            "buyer_id": getattr(f, "buyer_id", ""),
            "buyer_name": getattr(f, "buyer_name", ""),
            "requests_committed": f.requests_committed,
            "requests_delivered": f.requests_delivered,
            "requests_remaining": f.requests_remaining,
            "delivery_pct": float(getattr(f, "delivery_pct", 0)),
            "contract_price_usd": str(f.contract_price_usd),
            "collateral_usd": str(f.collateral_usd),
            "spot_price_usd": str(f.spot_price_usd),
            "discount_rate": str(f.discount_rate),
            "delivery_start": f.delivery_start.isoformat() if getattr(f, "delivery_start", None) else None,
            "delivery_end": f.delivery_end.isoformat() if getattr(f, "delivery_end", None) else None,
            "status": f.status.value if hasattr(f.status, "value") else str(f.status),
        }

    def _serialize_subscription(s: Any) -> dict[str, Any]:
        return {
            "token_id": s.token_id,
            "owner_id": getattr(s, "owner_id", ""),
            "requests_per_month": s.requests_per_month,
            "requests_used_this_period": s.requests_used_this_period,
            "requests_remaining": s.requests_remaining,
            "utilisation": float(getattr(s, "utilisation", 0)),
            "mint_price_usd": str(s.mint_price_usd),
            "valid_from": s.valid_from.isoformat() if getattr(s, "valid_from", None) else None,
            "valid_until": s.valid_until.isoformat() if getattr(s, "valid_until", None) else None,
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
        }

    def _serialize_sale(s: Any) -> dict[str, Any]:
        return {
            "sale_id": getattr(s, "sale_id", ""),
            "product_type": getattr(s, "product_type", ""),
            "buyer_id": getattr(s, "buyer_id", ""),
            "price_usd": str(getattr(s, "price_usd", "0")),
            "tokens_sold": getattr(s, "tokens_sold", 0),
            "timestamp": s.timestamp.isoformat() if getattr(s, "timestamp", None) else None,
        }

    active_futures: list[Any] = []
    subscriptions: list[Any] = []
    recent_sales: list[Any] = []
    sub_cap_pct = "0.00"
    deriv_cap_pct = "0.00"
    combined_cap_pct = "0.00"

    if deriv_mgr is not None:
        active_futures = [f for f in getattr(deriv_mgr, "_futures", {}).values()
                          if hasattr(f, "status") and f.status.value == "active"]
        subscriptions = [t for t in getattr(deriv_mgr, "_tokens", {}).values()
                         if hasattr(t, "status") and t.status.value == "active"]
        sub_cap_pct = str(getattr(deriv_mgr, "subscription_capacity_pct", "0"))
        deriv_cap_pct = str(getattr(deriv_mgr, "futures_capacity_pct", "0"))
        combined_cap_pct = str(getattr(deriv_mgr, "combined_capacity_pct", "0"))

    if km is not None:
        recent_sales = list(getattr(km, "_recent_sales", []))[-50:]

    return {
        "subscriptions": [_serialize_subscription(s) for s in subscriptions],
        "active_futures": [_serialize_future(f) for f in active_futures],
        "subscription_capacity_pct": sub_cap_pct,
        "derivatives_capacity_pct": deriv_cap_pct,
        "combined_capacity_pct": combined_cap_pct,
        "derivative_liabilities_usd": str(state.derivative_liabilities),
        "recent_sales": [_serialize_sale(s) for s in recent_sales],
    }


@router.get("/api/v1/oikos/dream")
async def get_oikos_dream(request: Request) -> dict[str, Any]:
    """Return the latest economic dream result (Monte Carlo simulation)."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"dream": None, "has_result": False}

    oneiros = getattr(request.app.state, "oneiros", None)
    dream_result = None

    # Try to get the latest economic dream from the dream worker
    if oneiros is not None:
        dream_worker = getattr(oneiros, "_economic_dream_worker", None)
        if dream_worker is None:
            dream_worker = getattr(oneiros, "economic_dream_worker", None)
        if dream_worker is not None:
            dream_result = getattr(dream_worker, "last_result", None)

    # Also check if oikos itself holds a reference
    if dream_result is None:
        dream_worker = getattr(oikos, "_dream_worker", None)
        if dream_worker is not None:
            dream_result = getattr(dream_worker, "last_result", None)

    if dream_result is None:
        return {"dream": None, "has_result": False}

    def _serialize_path_stats(ps: Any) -> dict[str, Any]:
        return {
            "paths_run": ps.paths_run,
            "ruin_count": ps.ruin_count,
            "ruin_probability": float(ps.ruin_probability),
            "median_net_worth": str(ps.median_net_worth),
            "p5_net_worth": str(ps.p5_net_worth),
            "p95_net_worth": str(ps.p95_net_worth),
            "mean_net_worth": str(ps.mean_net_worth),
            "median_min_runway_days": float(ps.median_min_runway_days),
            "max_drawdown_median": float(ps.max_drawdown_median),
            "median_time_to_mitosis_days": float(getattr(ps, "median_time_to_mitosis_days", 0)),
        }

    def _serialize_stress_test(st: Any) -> dict[str, Any]:
        return {
            "scenario": st.scenario.value if hasattr(st.scenario, "value") else str(st.scenario),
            "survives": st.survives,
            "stats": _serialize_path_stats(st.stats),
        }

    def _serialize_recommendation(r: Any) -> dict[str, Any]:
        return {
            "action": r.action,
            "description": r.description,
            "priority": r.priority,
            "parameter_path": getattr(r, "parameter_path", None),
            "current_value": str(getattr(r, "current_value", "")),
            "recommended_value": str(getattr(r, "recommended_value", "")),
            "ruin_probability_before": float(getattr(r, "ruin_probability_before", 0)),
            "ruin_probability_after": float(getattr(r, "ruin_probability_after", 0)),
            "confidence": float(getattr(r, "confidence", 0)),
        }

    return {
        "has_result": True,
        "dream": {
            "id": str(dream_result.id),
            "sleep_cycle_id": str(getattr(dream_result, "sleep_cycle_id", "")),
            "timestamp": dream_result.timestamp.isoformat(),
            "baseline": _serialize_path_stats(dream_result.baseline),
            "stress_tests": [_serialize_stress_test(st) for st in dream_result.stress_tests],
            "resilience_score": float(dream_result.resilience_score),
            "ruin_probability": float(dream_result.ruin_probability),
            "survival_probability_30d": float(dream_result.survival_probability_30d),
            "recommendations": [_serialize_recommendation(r) for r in (dream_result.recommendations or [])],
            "duration_ms": getattr(dream_result, "duration_ms", 0),
            "total_paths_simulated": getattr(dream_result, "total_paths_simulated", 0),
        },
    }


@router.get("/api/v1/oikos/tollbooths")
async def get_oikos_tollbooths(request: Request) -> dict[str, Any]:
    """Return all deployed tollbooth contracts with addresses, prices, and on-chain accumulated revenue."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"tollbooths": [], "total_count": 0, "total_accumulated_usdc": "0.00"}

    tollbooth_mgr = oikos.tollbooth_manager
    factory = oikos.asset_factory
    live_assets = {a.asset_id: a for a in factory.get_live_assets()}
    building_assets = {a.asset_id: a for a in factory.get_building_assets()}
    all_assets_map = {**live_assets, **building_assets}

    results = []
    total_accumulated = 0.0

    for asset_id, deployment in tollbooth_mgr._deployments.items():
        asset = all_assets_map.get(asset_id)
        asset_name = getattr(asset, "name", asset_id) if asset else asset_id
        # Surface factory-recorded total_revenue_usd as the proxy for accumulated USDC
        accumulated_usdc = float(getattr(asset, "total_revenue_usd", "0")) if asset is not None else 0.0
        total_accumulated += accumulated_usdc
        results.append({
            "asset_id": asset_id,
            "asset_name": asset_name,
            "contract_address": deployment.contract_address,
            "chain": deployment.chain,
            "price_per_call_usdc": str(deployment.price_per_call_usdc),
            "accumulated_revenue_usdc": str(round(accumulated_usdc, 6)),
            "owner_address": deployment.owner_address,
            "tx_hash": deployment.tx_hash,
            "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
        })

    return {
        "tollbooths": results,
        "total_count": len(results),
        "total_accumulated_usdc": str(round(total_accumulated, 6)),
    }


@router.get("/api/v1/oikos/threat-model")
async def get_oikos_threat_model(request: Request) -> dict[str, Any]:
    """Return the latest ThreatModelResult from the economic dreaming system."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"has_result": False, "threat_model": None}

    oneiros = getattr(request.app.state, "oneiros", None)
    threat_result = None

    if oneiros is not None:
        tm_worker = getattr(oneiros, "_threat_model_worker", None)
        if tm_worker is None:
            tm_worker = getattr(oneiros, "threat_model_worker", None)
        if tm_worker is not None:
            threat_result = getattr(tm_worker, "last_result", None)

    if threat_result is None:
        tm_worker = getattr(oikos, "_threat_model_worker", None)
        if tm_worker is not None:
            threat_result = getattr(tm_worker, "last_result", None)

    if threat_result is None:
        return {"has_result": False, "threat_model": None}

    def _serialize_tail_risk(tr: Any) -> dict[str, Any]:
        return {
            "var_5pct": str(tr.var_5pct),
            "var_25pct": str(tr.var_25pct),
            "cvar_5pct": str(tr.cvar_5pct),
            "max_drawdown_median": str(tr.max_drawdown_median),
            "max_drawdown_p95": str(tr.max_drawdown_p95),
            "liquidation_probability": str(tr.liquidation_probability),
            "expected_liquidation_loss": str(tr.expected_liquidation_loss),
            "time_to_liquidation_p10": tr.time_to_liquidation_p10,
        }

    def _serialize_exposure(e: Any) -> dict[str, Any]:
        return {
            "position_id": e.position_id,
            "symbol": e.symbol,
            "asset_class": e.asset_class.value if hasattr(e.asset_class, "value") else str(e.asset_class),
            "exposure_usd": str(e.exposure_usd),
            "contribution_to_portfolio_var": str(e.contribution_to_portfolio_var),
            "contagion_amplifier": str(e.contagion_amplifier),
            "risk_rank": e.risk_rank,
            "rationale": e.rationale,
        }

    def _serialize_hedge(h: Any) -> dict[str, Any]:
        return {
            "id": str(h.id),
            "target_position_id": h.target_position_id,
            "target_symbol": h.target_symbol,
            "hedge_action": h.hedge_action,
            "hedge_instrument": h.hedge_instrument,
            "hedge_size_usd": str(h.hedge_size_usd),
            "hedge_size_pct": str(h.hedge_size_pct),
            "var_reduction_pct": str(h.var_reduction_pct),
            "liquidation_prob_reduction": str(h.liquidation_prob_reduction),
            "cost_estimate_usd": str(h.cost_estimate_usd),
            "priority": h.priority,
            "confidence": str(h.confidence),
            "description": h.description,
        }

    return {
        "has_result": True,
        "threat_model": {
            "id": str(threat_result.id),
            "sleep_cycle_id": str(getattr(threat_result, "sleep_cycle_id", "")),
            "timestamp": threat_result.timestamp.isoformat(),
            "portfolio_risk": _serialize_tail_risk(threat_result.portfolio_risk),
            "position_risks": {
                pid: _serialize_tail_risk(tr)
                for pid, tr in threat_result.position_risks.items()
            },
            "critical_exposures": [_serialize_exposure(e) for e in threat_result.critical_exposures],
            "hedging_proposals": [_serialize_hedge(h) for h in threat_result.hedging_proposals],
            "contagion_events_detected": threat_result.contagion_events_detected,
            "contagion_loss_amplifier": str(threat_result.contagion_loss_amplifier),
            "total_paths_simulated": threat_result.total_paths_simulated,
            "horizon_days": threat_result.horizon_days,
            "duration_ms": threat_result.duration_ms,
            "positions_analyzed": getattr(threat_result, "positions_analyzed", 0),
        },
    }


@router.get("/api/v1/oikos/history")
async def get_oikos_history(request: Request, days: int = 7) -> dict[str, Any]:
    """Return timeseries of economic snapshots for the last N days."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"snapshots": [], "days": days, "count": 0}

    snapshot_writer = getattr(request.app.state, "oikos_snapshot_writer", None)
    if snapshot_writer is None:
        from primitives.common import utc_now
        state = oikos.snapshot()
        now = utc_now()
        return {
            "snapshots": [{
                "timestamp": now.isoformat(),
                "net_worth_usd": str(state.total_net_worth),
                "liquid_balance": str(state.liquid_balance),
                "burn_rate_usd_per_day": str(state.current_burn_rate.usd_per_day),
                "runway_days": str(state.runway_days),
                "revenue_24h": str(state.revenue_24h),
                "costs_24h": str(state.costs_24h),
                "net_income_24h": str(state.net_income_24h),
                "starvation_level": state.starvation_level.value,
            }],
            "days": days,
            "count": 1,
        }

    snapshots = await snapshot_writer.get_history(days=days)
    return {
        "snapshots": [
            {
                "timestamp": s["timestamp"],
                "net_worth_usd": s["net_worth_usd"],
                "liquid_balance": s["liquid_balance"],
                "burn_rate_usd_per_day": s["burn_rate_usd_per_day"],
                "runway_days": s["runway_days"],
                "revenue_24h": s["revenue_24h"],
                "costs_24h": s["costs_24h"],
                "net_income_24h": s["net_income_24h"],
                "starvation_level": s["starvation_level"],
            }
            for s in snapshots
        ],
        "days": days,
        "count": len(snapshots),
    }


@router.post("/api/v1/oikos/assets/{asset_id}/terminate")
async def terminate_oikos_asset(request: Request, asset_id: str) -> dict[str, Any]:
    """Manually terminate a deployed asset."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "error", "message": "Oikos not initialized"}

    body: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        body = await request.json()
    reason = str(body.get("reason", "manual_termination"))

    try:
        asset = oikos.asset_factory.terminate_asset(asset_id, reason=reason)
        logger.info(
            "oikos_control_action",
            action="terminate_asset",
            target_id=asset_id,
            instance_id=getattr(oikos, "_instance_id", ""),
            reason=reason,
        )
        return {
            "status": "ok",
            "asset_id": asset_id,
            "asset_name": getattr(asset, "name", asset_id),
            "new_status": asset.status.value if hasattr(asset.status, "value") else str(asset.status),
        }
    except Exception as exc:
        logger.exception("oikos_terminate_asset_failed", asset_id=asset_id, error=str(exc))
        return {"status": "error", "message": str(exc)}


@router.post("/api/v1/oikos/children/{instance_id}/rescue")
async def rescue_oikos_child(request: Request, instance_id: str) -> dict[str, Any]:
    """Rescue a struggling child organism (max 2 rescues)."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "error", "message": "Oikos not initialized"}

    state = oikos.snapshot()
    child = next((c for c in state.child_instances if c.instance_id == instance_id), None)
    if child is None:
        return {"status": "error", "message": f"Child {instance_id} not found"}

    if not child.is_rescuable:
        return {
            "status": "error",
            "message": (
                f"Child {instance_id} is not rescuable "
                f"(rescue_count={child.rescue_count}, status={child.status.value})"
            ),
        }

    try:
        from systems.oikos.models import ChildStatus
        child.rescue_count += 1
        child.status = ChildStatus.ALIVE
        logger.info(
            "oikos_control_action",
            action="rescue_child",
            target_id=instance_id,
            instance_id=getattr(oikos, "_instance_id", ""),
            rescue_count=child.rescue_count,
        )
        return {
            "status": "ok",
            "instance_id": instance_id,
            "rescue_count": child.rescue_count,
            "new_status": child.status.value,
        }
    except Exception as exc:
        logger.exception("oikos_rescue_child_failed", instance_id=instance_id, error=str(exc))
        return {"status": "error", "message": str(exc)}


@router.post("/api/v1/oikos/organs/{organ_id}/pause")
async def pause_oikos_organ(request: Request, organ_id: str) -> dict[str, Any]:
    """Pause (zero allocation) or resume (restore 10% allocation) an economic organ."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "error", "message": "Oikos not initialized"}

    from decimal import Decimal

    organ = oikos._morphogenesis.get_organ(organ_id)
    if organ is None:
        return {"status": "error", "message": f"Organ {organ_id} not found"}

    try:
        was_active = organ.is_active
        if was_active:
            organ.resource_allocation_pct = Decimal("0")
            action = "pause_organ"
        else:
            organ.resource_allocation_pct = Decimal("0.10")
            action = "resume_organ"
        logger.info(
            "oikos_control_action",
            action=action,
            target_id=organ_id,
            instance_id=getattr(oikos, "_instance_id", ""),
            was_active=was_active,
        )
        return {
            "status": "ok",
            "organ_id": organ_id,
            "action": action,
            "is_active": organ.is_active,
            "resource_allocation_pct": str(organ.resource_allocation_pct),
        }
    except Exception as exc:
        logger.exception("oikos_pause_organ_failed", organ_id=organ_id, error=str(exc))
        return {"status": "error", "message": str(exc)}


@router.post("/api/v1/oikos/genesis-spark/reset")
async def reset_genesis_spark(request: Request) -> dict[str, Any]:
    """Reset dormant state for testing — clears in-memory economic state so genesis-spark can re-run."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "error", "message": "Oikos not initialized"}

    try:
        from systems.oikos.models import EconomicState
        instance_id = getattr(oikos, "_instance_id", "reset")
        oikos._state = EconomicState(instance_id=instance_id)
        oikos._morphogenesis._organs = {}
        await oikos.persist_state()
        logger.info(
            "oikos_control_action",
            action="genesis_spark_reset",
            target_id="genesis",
            instance_id=instance_id,
        )
        return {"status": "ok", "message": "Genesis spark state reset — organism is dormant"}
    except Exception as exc:
        logger.exception("oikos_genesis_reset_failed", error=str(exc))
        return {"status": "error", "message": str(exc)}


@router.get("/api/v1/simula/assets")
async def get_simula_assets(request: Request) -> dict[str, Any]:
    """Return deployed OwnedAssets from the AssetFactory."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "unavailable", "error": "Oikos not initialized"}

    factory = oikos.asset_factory
    live = factory.get_live_assets()
    building = factory.get_building_assets()

    def _serialize_asset(asset: Any) -> dict[str, Any]:
        return {
            "asset_id": asset.asset_id,
            "name": asset.name,
            "description": asset.description,
            "asset_type": asset.asset_type,
            "status": asset.status.value,
            "estimated_value_usd": str(asset.estimated_value_usd),
            "development_cost_usd": str(asset.development_cost_usd),
            "monthly_revenue_usd": str(asset.monthly_revenue_usd),
            "monthly_cost_usd": str(asset.monthly_cost_usd),
            "total_revenue_usd": str(asset.total_revenue_usd),
            "total_cost_usd": str(asset.total_cost_usd),
            "projected_break_even_days": asset.projected_break_even_days,
            "break_even_reached": asset.break_even_reached,
            "deployed_at": asset.deployed_at.isoformat() if asset.deployed_at else None,
        }

    return {
        "status": "ok",
        "data": {
            "live_count": len(live),
            "building_count": len(building),
            "live": [_serialize_asset(a) for a in live],
            "building": [_serialize_asset(a) for a in building],
        },
    }
