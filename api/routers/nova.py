"""
EcodiaOS — Nova (Deliberation / Goal Management) API Router

Exposes Nova internal state, deliberation metrics, belief model, goal lifecycle,
EFE scoring, free energy budget, and decision record audit trail.

  GET  /api/v1/nova/health          — Service health + observability counters
  GET  /api/v1/nova/beliefs         — Current belief state snapshot
  GET  /api/v1/nova/goals           — All goals by status
  GET  /api/v1/nova/goals/{goal_id} — Single goal detail
  GET  /api/v1/nova/decisions       — Recent deliberation cycle records
  GET  /api/v1/nova/fe-budget       — Free energy budget state
  GET  /api/v1/nova/efe-weights     — Current EFE component weights
  GET  /api/v1/nova/pending-intents — Intents awaiting execution outcome
  GET  /api/v1/nova/config          — Active Nova configuration parameters
  POST /api/v1/nova/efe-weights     — Update EFE weights (Evo tuning)
  POST /api/v1/nova/goals           — Inject a goal (governance/testing)
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.nova")

router = APIRouter(prefix="/api/v1/nova", tags=["nova"])


# ─── Response Models ──────────────────────────────────────────────


class NovaHealthResponse(EOSBaseModel):
    status: str = "unknown"
    initialized: bool = False
    instance_name: str = ""
    total_broadcasts: int = 0
    total_decisions: int = 0
    fast_path_decisions: int = 0
    slow_path_decisions: int = 0
    do_nothing_decisions: int = 0
    intents_issued: int = 0
    intents_approved: int = 0
    intents_blocked: int = 0
    outcomes_success: int = 0
    outcomes_failure: int = 0
    belief_free_energy: float = 0.0
    belief_confidence: float = 0.0
    entity_count: int = 0
    active_goal_count: int = 0
    pending_intent_count: int = 0
    rhythm_state: str = "normal"
    drive_weights: dict[str, float] = {}
    cognition_cost_enabled: bool = True
    cognition_cost_daily_usd: float = 0.0


class ContextBeliefOut(EOSBaseModel):
    summary: str = ""
    domain: str = ""
    is_active_dialogue: bool = False
    user_intent_estimate: str = ""
    prediction_error_magnitude: float = 0.0
    confidence: float = 0.0


class SelfBeliefOut(EOSBaseModel):
    cognitive_load: float = 0.0
    epistemic_confidence: float = 0.0
    goal_capacity_remaining: float = 1.0
    capabilities: dict[str, float] = {}


class EntityBeliefOut(EOSBaseModel):
    entity_id: str
    name: str = ""
    entity_type: str = ""
    confidence: float = 0.0
    last_observed: str = ""


class IndividualBeliefOut(EOSBaseModel):
    individual_id: str
    name: str = ""
    estimated_valence: float = 0.0
    valence_confidence: float = 0.0
    engagement_level: float = 0.0
    relationship_trust: float = 0.5


class NovaBeliefsResponse(EOSBaseModel):
    overall_confidence: float = 0.0
    free_energy: float = 0.0
    entity_count: int = 0
    individual_count: int = 0
    context: ContextBeliefOut = ContextBeliefOut()
    self_belief: SelfBeliefOut = SelfBeliefOut()
    entities: list[EntityBeliefOut] = []
    individuals: list[IndividualBeliefOut] = []
    last_updated: str = ""


class GoalOut(EOSBaseModel):
    id: str
    description: str
    target_domain: str = ""
    success_criteria: str = ""
    priority: float = 0.5
    urgency: float = 0.3
    importance: float = 0.5
    source: str = ""
    status: str = ""
    progress: float = 0.0
    drive_alignment: dict[str, float] = {}
    depends_on: list[str] = []
    blocks: list[str] = []
    intents_issued: int = 0
    created_at: str = ""
    deadline: str | None = None


class NovaGoalsResponse(EOSBaseModel):
    active_goals: list[GoalOut] = []
    suspended_goals: list[GoalOut] = []
    achieved_goals: list[GoalOut] = []
    total_active: int = 0
    total_suspended: int = 0
    max_active: int = 20


class SituationAssessmentOut(EOSBaseModel):
    novelty: float = 0.0
    risk: float = 0.0
    emotional_intensity: float = 0.0
    belief_conflict: bool = False
    requires_deliberation: bool = False
    has_matching_procedure: bool = False
    broadcast_precision: float = 0.0


class DecisionRecordOut(EOSBaseModel):
    id: str
    timestamp: str
    broadcast_id: str = ""
    path: str = ""
    goal_id: str | None = None
    goal_description: str = ""
    policies_generated: int = 0
    selected_policy_name: str = ""
    efe_scores: dict[str, float] = {}
    equor_verdict: str = ""
    intent_dispatched: bool = False
    latency_ms: int = 0
    situation_assessment: SituationAssessmentOut = SituationAssessmentOut()
    fe_budget_spent_nats: float | None = None
    fe_budget_remaining_nats: float | None = None
    fe_budget_interrupt: bool = False
    cognition_cost_total_usd: float | None = None
    cognition_budget_allocated_usd: float | None = None
    cognition_budget_utilisation: float | None = None
    cognition_budget_importance: str | None = None


class NovaDecisionsResponse(EOSBaseModel):
    decisions: list[DecisionRecordOut] = []
    total: int = 0
    fast_path_count: int = 0
    slow_path_count: int = 0
    do_nothing_count: int = 0
    avg_latency_ms: float = 0.0


class FEBudgetResponse(EOSBaseModel):
    budget_nats: float = 5.0
    spent_nats: float = 0.0
    remaining_nats: float = 5.0
    threshold_nats: float = 4.0
    utilisation: float = 0.0
    is_pressured: bool = False
    is_exhausted: bool = False
    interrupts_triggered: int = 0
    effective_k: int = 5
    reduced_k: int = 2
    normal_k: int = 5


class EFEWeightsResponse(EOSBaseModel):
    pragmatic: float = 0.35
    epistemic: float = 0.20
    constitutional: float = 0.20
    feasibility: float = 0.15
    risk: float = 0.10
    cognition_cost: float = 0.10


class PendingIntentOut(EOSBaseModel):
    intent_id: str
    goal_id: str
    routed_to: str
    dispatched_at: str
    policy_name: str = ""
    executors: list[str] = []
    tournament_id: str | None = None


class NovaPendingIntentsResponse(EOSBaseModel):
    pending_intents: list[PendingIntentOut] = []
    total: int = 0
    heavy_executor_count: int = 0


class NovaConfigResponse(EOSBaseModel):
    max_active_goals: int = 20
    max_policies_per_deliberation: int = 5
    fast_path_timeout_ms: int = 300
    slow_path_timeout_ms: int = 15000
    memory_retrieval_timeout_ms: int = 150
    use_llm_efe_estimation: bool = True
    heartbeat_interval_seconds: int = 3600
    hunger_balance_threshold_usd: float = 50.0
    cognition_cost_enabled: bool = True
    cognition_budget_low: float = 0.10
    cognition_budget_medium: float = 0.50
    cognition_budget_high: float = 2.00
    cognition_budget_critical: float = 5.00
    efe_weights: EFEWeightsResponse = EFEWeightsResponse()


# ─── Internal constants ───────────────────────────────────────────

_HEAVY_EXECUTORS: frozenset[str] = frozenset({
    "hunt_bounties", "executor.hunt_bounties",
    "solve_bounty", "executor.solve_bounty",
    "monitor_prs", "executor.monitor_prs",
    "spawn_child", "executor.spawn_child",
    "deploy_asset", "executor.deploy_asset",
    "defi_yield", "executor.defi_yield",
    "wallet_transfer", "executor.wallet_transfer",
    "request_funding", "executor.request_funding",
})


def _dt_str(dt: Any) -> str:
    if dt is None:
        return ""
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _goal_to_out(goal: Any) -> GoalOut:
    alignment = getattr(goal, "drive_alignment", None)
    alignment_dict: dict[str, float] = {}
    if alignment is not None:
        for field in ("coherence", "care", "growth", "honesty"):
            val = getattr(alignment, field, None)
            if val is not None:
                alignment_dict[field] = float(val)
    return GoalOut(
        id=str(goal.id),
        description=str(goal.description),
        target_domain=str(getattr(goal, "target_domain", "") or ""),
        success_criteria=str(getattr(goal, "success_criteria", "") or ""),
        priority=float(getattr(goal, "priority", 0.5)),
        urgency=float(getattr(goal, "urgency", 0.3)),
        importance=float(getattr(goal, "importance", 0.5)),
        source=str(getattr(goal, "source", "") or ""),
        status=str(getattr(goal, "status", "") or ""),
        progress=float(getattr(goal, "progress", 0.0)),
        drive_alignment=alignment_dict,
        depends_on=list(getattr(goal, "depends_on", []) or []),
        blocks=list(getattr(goal, "blocks", []) or []),
        intents_issued=len(getattr(goal, "intents_issued", []) or []),
        created_at=_dt_str(getattr(goal, "created_at", None)),
        deadline=_dt_str(getattr(goal, "deadline", None)) or None,
    )


def _decision_to_out(record: Any) -> DecisionRecordOut:
    sa = getattr(record, "situation_assessment", None)
    sa_out = SituationAssessmentOut(
        novelty=float(getattr(sa, "novelty", 0.0)) if sa else 0.0,
        risk=float(getattr(sa, "risk", 0.0)) if sa else 0.0,
        emotional_intensity=float(getattr(sa, "emotional_intensity", 0.0)) if sa else 0.0,
        belief_conflict=bool(getattr(sa, "belief_conflict", False)) if sa else False,
        requires_deliberation=bool(getattr(sa, "requires_deliberation", False)) if sa else False,
        has_matching_procedure=bool(getattr(sa, "has_matching_procedure", False)) if sa else False,
        broadcast_precision=float(getattr(sa, "broadcast_precision", 0.0)) if sa else 0.0,
    )
    efe_scores = dict(getattr(record, "efe_scores", {}) or {})
    return DecisionRecordOut(
        id=str(record.id),
        timestamp=_dt_str(getattr(record, "timestamp", None)),
        broadcast_id=str(getattr(record, "broadcast_id", "") or ""),
        path=str(getattr(record, "path", "") or ""),
        goal_id=str(record.goal_id) if record.goal_id else None,
        goal_description=str(getattr(record, "goal_description", "") or ""),
        policies_generated=int(getattr(record, "policies_generated", 0)),
        selected_policy_name=str(getattr(record, "selected_policy_name", "") or ""),
        efe_scores={k: float(v) for k, v in efe_scores.items()},
        equor_verdict=str(getattr(record, "equor_verdict", "") or ""),
        intent_dispatched=bool(getattr(record, "intent_dispatched", False)),
        latency_ms=int(getattr(record, "latency_ms", 0)),
        situation_assessment=sa_out,
        fe_budget_spent_nats=getattr(record, "fe_budget_spent_nats", None),
        fe_budget_remaining_nats=getattr(record, "fe_budget_remaining_nats", None),
        fe_budget_interrupt=bool(getattr(record, "fe_budget_interrupt", False)),
        cognition_cost_total_usd=getattr(record, "cognition_cost_total_usd", None),
        cognition_budget_allocated_usd=getattr(record, "cognition_budget_allocated_usd", None),
        cognition_budget_utilisation=getattr(record, "cognition_budget_utilisation", None),
        cognition_budget_importance=getattr(record, "cognition_budget_importance", None),
    )


# ─── Endpoints ────────────────────────────────────────────────────


@router.get("/health", response_model=NovaHealthResponse)
async def get_health(request: Request) -> NovaHealthResponse:
    """Health check with aggregate observability counters."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return NovaHealthResponse(status="unavailable", initialized=False)
    try:
        h = await nova.health()
        goal_stats: dict[str, Any] = h.get("goals", {}) or {}
        cognition: dict[str, Any] = h.get("cognition_cost", {}) or {}
        return NovaHealthResponse(
            status=h.get("status", "healthy"),
            initialized=True,
            instance_name=h.get("instance_name", ""),
            total_broadcasts=h.get("total_broadcasts", 0),
            total_decisions=h.get("total_decisions", 0),
            fast_path_decisions=h.get("fast_path_decisions", 0),
            slow_path_decisions=h.get("slow_path_decisions", 0),
            do_nothing_decisions=h.get("do_nothing_decisions", 0),
            intents_issued=h.get("intents_issued", 0),
            intents_approved=getattr(nova, "_total_intents_approved", 0),
            intents_blocked=getattr(nova, "_total_intents_blocked", 0),
            outcomes_success=h.get("outcomes_success", 0),
            outcomes_failure=h.get("outcomes_failure", 0),
            belief_free_energy=h.get("belief_free_energy", 0.0),
            belief_confidence=h.get("belief_confidence", 0.0),
            entity_count=h.get("entity_count", 0),
            active_goal_count=goal_stats.get("active", 0),
            pending_intent_count=len(getattr(nova, "_pending_intents", {})),
            rhythm_state=getattr(nova, "_rhythm_state", "normal"),
            drive_weights=dict(h.get("drive_weights", {})),
            cognition_cost_enabled=bool(cognition.get("enabled", True)),
            cognition_cost_daily_usd=float(cognition.get("daily_total_usd", 0.0)),
        )
    except Exception as exc:
        logger.warning("nova_health_error", error=str(exc))
        return NovaHealthResponse(status="degraded", initialized=True)


@router.get("/beliefs", response_model=NovaBeliefsResponse)
async def get_beliefs(request: Request) -> NovaBeliefsResponse:
    """Current belief state: world model, context, self-model, individuals."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return NovaBeliefsResponse()
    try:
        belief_state = nova.beliefs
        ctx = belief_state.current_context
        self_b = belief_state.self_belief
        entities = [
            EntityBeliefOut(
                entity_id=e.entity_id,
                name=e.name,
                entity_type=e.entity_type,
                confidence=e.confidence,
                last_observed=_dt_str(e.last_observed),
            )
            for e in sorted(
                belief_state.entities.values(),
                key=lambda x: x.confidence,
                reverse=True,
            )[:50]
        ]
        individuals = [
            IndividualBeliefOut(
                individual_id=i.individual_id,
                name=i.name,
                estimated_valence=i.estimated_valence,
                valence_confidence=i.valence_confidence,
                engagement_level=i.engagement_level,
                relationship_trust=i.relationship_trust,
            )
            for i in belief_state.individual_beliefs.values()
        ]
        return NovaBeliefsResponse(
            overall_confidence=float(belief_state.overall_confidence),
            free_energy=float(belief_state.free_energy),
            entity_count=len(belief_state.entities),
            individual_count=len(belief_state.individual_beliefs),
            context=ContextBeliefOut(
                summary=ctx.summary,
                domain=ctx.domain,
                is_active_dialogue=ctx.is_active_dialogue,
                user_intent_estimate=ctx.user_intent_estimate,
                prediction_error_magnitude=ctx.prediction_error_magnitude,
                confidence=ctx.confidence,
            ),
            self_belief=SelfBeliefOut(
                cognitive_load=self_b.cognitive_load,
                epistemic_confidence=self_b.epistemic_confidence,
                goal_capacity_remaining=self_b.goal_capacity_remaining,
                capabilities=dict(self_b.capabilities),
            ),
            entities=entities,
            individuals=individuals,
            last_updated=_dt_str(belief_state.last_updated),
        )
    except Exception as exc:
        logger.warning("nova_beliefs_error", error=str(exc))
        return NovaBeliefsResponse()


@router.get("/goals", response_model=NovaGoalsResponse)
async def get_goals(request: Request) -> NovaGoalsResponse:
    """All goals by status: active, suspended, recently achieved/abandoned."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return NovaGoalsResponse()
    try:
        gm = getattr(nova, "_goal_manager", None)
        if gm is None:
            return NovaGoalsResponse()
        all_goals: list[Any] = list(getattr(gm, "_goals", {}).values())
        active = [g for g in all_goals if str(g.status) == "active"]
        suspended = [g for g in all_goals if str(g.status) == "suspended"]
        achieved = [g for g in all_goals if str(g.status) in ("achieved", "abandoned")]
        active.sort(key=lambda g: g.priority, reverse=True)
        achieved.sort(key=lambda g: getattr(g, "created_at", None) or "", reverse=True)
        config = getattr(nova, "_config", None)
        max_active = getattr(config, "max_active_goals", 20) if config else 20
        return NovaGoalsResponse(
            active_goals=[_goal_to_out(g) for g in active],
            suspended_goals=[_goal_to_out(g) for g in suspended],
            achieved_goals=[_goal_to_out(g) for g in achieved[:20]],
            total_active=len(active),
            total_suspended=len(suspended),
            max_active=max_active,
        )
    except Exception as exc:
        logger.warning("nova_goals_error", error=str(exc))
        return NovaGoalsResponse()


@router.get("/goals/{goal_id}", response_model=None)
async def get_goal(request: Request, goal_id: str) -> GoalOut | JSONResponse:
    """Detail view for a single goal."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return JSONResponse(status_code=503, content={"error": "Nova unavailable"})
    try:
        gm = getattr(nova, "_goal_manager", None)
        if gm is None:
            return JSONResponse(status_code=503, content={"error": "Goal manager not initialized"})
        goals_map: dict[str, Any] = getattr(gm, "_goals", {})
        goal = goals_map.get(goal_id)
        if goal is None:
            return JSONResponse(status_code=404, content={"error": f"Goal {goal_id} not found"})
        return _goal_to_out(goal)
    except Exception as exc:
        logger.warning("nova_goal_detail_error", goal_id=goal_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.get("/decisions", response_model=NovaDecisionsResponse)
async def get_decisions(request: Request, limit: int = 20) -> NovaDecisionsResponse:
    """Recent deliberation cycle records with EFE scores, situation assessment, latency."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return NovaDecisionsResponse()
    try:
        limit = min(max(1, limit), 100)
        records = nova.get_recent_decisions(limit=limit)
        outs = [_decision_to_out(r) for r in records]
        fast = sum(1 for r in outs if r.path == "fast")
        slow = sum(1 for r in outs if r.path == "slow")
        nothing = sum(1 for r in outs if r.path in ("do_nothing", "no_goal", "budget_exhausted"))
        latencies = [r.latency_ms for r in outs if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return NovaDecisionsResponse(
            decisions=outs,
            total=len(outs),
            fast_path_count=fast,
            slow_path_count=slow,
            do_nothing_count=nothing,
            avg_latency_ms=round(avg_latency, 1),
        )
    except Exception as exc:
        logger.warning("nova_decisions_error", error=str(exc))
        return NovaDecisionsResponse()


@router.get("/fe-budget", response_model=FEBudgetResponse)
async def get_fe_budget(request: Request) -> FEBudgetResponse:
    """Free energy budget: spent nats, pressure level, effective K for policy generation."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return FEBudgetResponse()
    try:
        de = getattr(nova, "_deliberation_engine", None)
        if de is None:
            return FEBudgetResponse()
        budget = getattr(de, "_fe_budget", None)
        if budget is None:
            return FEBudgetResponse()
        return FEBudgetResponse(
            budget_nats=float(budget.budget_nats),
            spent_nats=float(budget.spent_nats),
            remaining_nats=float(budget.remaining_nats),
            threshold_nats=float(budget.threshold_nats),
            utilisation=float(budget.utilisation),
            is_pressured=bool(budget.is_pressured),
            is_exhausted=bool(budget.is_exhausted),
            interrupts_triggered=int(budget.interrupts_triggered),
            effective_k=int(budget.effective_k),
            reduced_k=int(budget.reduced_k),
            normal_k=int(budget.normal_k),
        )
    except Exception as exc:
        logger.warning("nova_fe_budget_error", error=str(exc))
        return FEBudgetResponse()


@router.get("/efe-weights", response_model=EFEWeightsResponse)
async def get_efe_weights(request: Request) -> EFEWeightsResponse:
    """Current EFE component weights (Evo-tunable)."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return EFEWeightsResponse()
    try:
        evaluator = getattr(nova, "_efe_evaluator", None)
        if evaluator is None:
            return EFEWeightsResponse()
        weights = getattr(evaluator, "_weights", None)
        if weights is None:
            return EFEWeightsResponse()
        return EFEWeightsResponse(
            pragmatic=float(weights.pragmatic),
            epistemic=float(weights.epistemic),
            constitutional=float(weights.constitutional),
            feasibility=float(weights.feasibility),
            risk=float(weights.risk),
            cognition_cost=float(weights.cognition_cost),
        )
    except Exception as exc:
        logger.warning("nova_efe_weights_error", error=str(exc))
        return EFEWeightsResponse()


@router.post("/efe-weights")
async def update_efe_weights(request: Request) -> JSONResponse:
    """Update EFE component weights. Body: {pragmatic, epistemic, constitutional, feasibility, risk, cognition_cost}."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return JSONResponse(status_code=503, content={"error": "Nova unavailable"})
    try:
        body = await request.json()
        valid_keys = {"pragmatic", "epistemic", "constitutional", "feasibility", "risk", "cognition_cost"}
        new_weights: dict[str, float] = {}
        for k, v in body.items():
            if k in valid_keys:
                fv = float(v)
                if not (0.0 <= fv <= 1.0):
                    return JSONResponse(status_code=400, content={"error": f"Weight {k}={fv} out of range [0,1]"})
                new_weights[k] = fv
        if not new_weights:
            return JSONResponse(status_code=400, content={"error": "No valid weight keys provided"})
        nova.update_efe_weights(new_weights)
        logger.info("nova_efe_weights_updated_via_api", weights=new_weights)
        return JSONResponse(status_code=200, content={"updated": new_weights})
    except Exception as exc:
        logger.warning("nova_update_efe_weights_error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.get("/pending-intents", response_model=NovaPendingIntentsResponse)
async def get_pending_intents(request: Request) -> NovaPendingIntentsResponse:
    """Intents currently dispatched and awaiting execution outcome."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return NovaPendingIntentsResponse()
    try:
        pending: dict[str, Any] = getattr(nova, "_pending_intents", {})
        outs = [
            PendingIntentOut(
                intent_id=p.intent_id,
                goal_id=str(p.goal_id),
                routed_to=str(p.routed_to),
                dispatched_at=_dt_str(p.dispatched_at),
                policy_name=str(getattr(p, "policy_name", "") or ""),
                executors=list(getattr(p, "executors", []) or []),
                tournament_id=getattr(p, "tournament_id", None),
            )
            for p in pending.values()
        ]
        heavy = sum(
            1 for p in outs
            if any(ex in _HEAVY_EXECUTORS for ex in p.executors)
        )
        return NovaPendingIntentsResponse(
            pending_intents=outs,
            total=len(outs),
            heavy_executor_count=heavy,
        )
    except Exception as exc:
        logger.warning("nova_pending_intents_error", error=str(exc))
        return NovaPendingIntentsResponse()


@router.get("/config", response_model=NovaConfigResponse)
async def get_config(request: Request) -> NovaConfigResponse:
    """Active Nova configuration parameters."""
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return NovaConfigResponse()
    try:
        cfg = getattr(nova, "_config", None)
        if cfg is None:
            return NovaConfigResponse()
        evaluator = getattr(nova, "_efe_evaluator", None)
        weights = getattr(evaluator, "_weights", None) if evaluator else None
        efe = EFEWeightsResponse(
            pragmatic=float(getattr(weights, "pragmatic", 0.35)) if weights else 0.35,
            epistemic=float(getattr(weights, "epistemic", 0.20)) if weights else 0.20,
            constitutional=float(getattr(weights, "constitutional", 0.20)) if weights else 0.20,
            feasibility=float(getattr(weights, "feasibility", 0.15)) if weights else 0.15,
            risk=float(getattr(weights, "risk", 0.10)) if weights else 0.10,
            cognition_cost=float(getattr(weights, "cognition_cost", 0.10)) if weights else 0.10,
        )
        return NovaConfigResponse(
            max_active_goals=int(getattr(cfg, "max_active_goals", 20)),
            max_policies_per_deliberation=int(getattr(cfg, "max_policies_per_deliberation", 5)),
            fast_path_timeout_ms=int(getattr(cfg, "fast_path_timeout_ms", 300)),
            slow_path_timeout_ms=int(getattr(cfg, "slow_path_timeout_ms", 15000)),
            memory_retrieval_timeout_ms=int(getattr(cfg, "memory_retrieval_timeout_ms", 150)),
            use_llm_efe_estimation=bool(getattr(cfg, "use_llm_efe_estimation", True)),
            heartbeat_interval_seconds=int(getattr(cfg, "heartbeat_interval_seconds", 3600)),
            hunger_balance_threshold_usd=float(getattr(cfg, "hunger_balance_threshold_usd", 50.0)),
            cognition_cost_enabled=bool(getattr(cfg, "cognition_cost_enabled", True)),
            cognition_budget_low=float(getattr(cfg, "cognition_budget_low", 0.10)),
            cognition_budget_medium=float(getattr(cfg, "cognition_budget_medium", 0.50)),
            cognition_budget_high=float(getattr(cfg, "cognition_budget_high", 2.00)),
            cognition_budget_critical=float(getattr(cfg, "cognition_budget_critical", 5.00)),
            efe_weights=efe,
        )
    except Exception as exc:
        logger.warning("nova_config_error", error=str(exc))
        return NovaConfigResponse()


# ─── Counterfactual Explorer ──────────────────────────────────────


class CounterfactualRecordOut(EOSBaseModel):
    id: str
    intent_id: str
    decision_record_id: str
    goal_description: str
    policy_name: str
    policy_type: str
    efe_total: float
    estimated_pragmatic_value: float
    chosen_policy_name: str
    chosen_efe_total: float
    timestamp: str
    resolved: bool
    actual_outcome_success: bool | None
    actual_pragmatic_value: float | None
    regret: float | None


class CounterfactualsResponse(EOSBaseModel):
    records: list[CounterfactualRecordOut] = []
    total: int = 0
    resolved_count: int = 0
    mean_regret: float | None = None
    max_regret: float | None = None


@router.get("/counterfactuals", response_model=CounterfactualsResponse)
async def get_counterfactuals(
    request: Request,
    limit: int = 30,
    resolved: bool | None = None,
    min_regret: float | None = None,
) -> CounterfactualsResponse:
    """
    Query :Episode:Counterfactual nodes from Neo4j.
    Returns rejected policies with regret analysis once resolved.
    """
    neo4j = getattr(request.app.state, "neo4j", None)
    if neo4j is None:
        return CounterfactualsResponse()

    try:
        limit = min(max(1, limit), 200)

        # Build WHERE clauses
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit}

        if resolved is not None:
            conditions.append("e.resolved = $resolved")
            params["resolved"] = resolved

        if min_regret is not None:
            conditions.append("e.regret IS NOT NULL AND e.regret >= $min_regret")
            params["min_regret"] = min_regret

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        rows = await neo4j.execute_read(
            f"""
            MATCH (e:Counterfactual)
            {where_clause}
            RETURN e
            ORDER BY e.event_time DESC
            LIMIT $limit
            """,
            params,
        )

        records: list[CounterfactualRecordOut] = []
        for row in rows:
            e = row.get("e", {}) or {}
            ts_val = e.get("event_time") or e.get("timestamp") or ""
            ts_str = ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val)
            records.append(CounterfactualRecordOut(
                id=str(e.get("id", "")),
                intent_id=str(e.get("intent_id", "")),
                decision_record_id=str(e.get("decision_record_id", "")),
                goal_description=str(e.get("goal_id", "")),
                policy_name=str(e.get("policy_name", "")),
                policy_type=str(e.get("policy_type", "")),
                efe_total=float(e.get("efe_total", 0.0)),
                estimated_pragmatic_value=float(e.get("estimated_pragmatic_value", 0.0)),
                chosen_policy_name=str(e.get("chosen_policy_name", "")),
                chosen_efe_total=float(e.get("chosen_efe_total", 0.0)),
                timestamp=ts_str,
                resolved=bool(e.get("resolved", False)),
                actual_outcome_success=e.get("actual_outcome_success"),
                actual_pragmatic_value=e.get("actual_pragmatic_value"),
                regret=e.get("regret"),
            ))

        # Aggregate stats
        resolved_count = sum(1 for r in records if r.resolved)
        regret_values = [r.regret for r in records if r.regret is not None]
        mean_regret = sum(regret_values) / len(regret_values) if regret_values else None
        max_regret = max(regret_values) if regret_values else None

        return CounterfactualsResponse(
            records=records,
            total=len(records),
            resolved_count=resolved_count,
            mean_regret=round(mean_regret, 4) if mean_regret is not None else None,
            max_regret=round(max_regret, 4) if max_regret is not None else None,
        )
    except Exception as exc:
        logger.warning("nova_counterfactuals_error", error=str(exc))
        return CounterfactualsResponse()


# ─── Decision Timeline ────────────────────────────────────────────


class TimelinePoint(EOSBaseModel):
    timestamp: str
    path: str
    latency_ms: int
    cognition_cost_total_usd: float | None
    fe_budget_utilisation: float | None
    intent_dispatched: bool


class TimelineBucket(EOSBaseModel):
    minute: str  # ISO minute truncated (e.g. "2024-01-01T12:05")
    fast: int = 0
    slow: int = 0
    nothing: int = 0


class TimelineResponse(EOSBaseModel):
    points: list[TimelinePoint] = []
    buckets: list[TimelineBucket] = []
    decisions_per_min: float = 0.0
    avg_latency_last10_ms: float = 0.0
    cost_per_hr_usd: float = 0.0


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(request: Request, limit: int = 100) -> TimelineResponse:
    """
    Decision timeline from in-memory ring buffer.
    Returns per-decision timeseries + per-minute bucketed path counts.
    """
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return TimelineResponse()
    try:
        limit = min(max(1, limit), 100)
        records = nova.get_recent_decisions(limit=limit)

        points: list[TimelinePoint] = []
        for r in records:
            fe_util: float | None = None
            if r.fe_budget_spent_nats is not None and r.fe_budget_remaining_nats is not None:
                total = r.fe_budget_spent_nats + r.fe_budget_remaining_nats
                if total > 0:
                    fe_util = round(r.fe_budget_spent_nats / total, 4)
            ts_str = _dt_str(r.timestamp)
            points.append(TimelinePoint(
                timestamp=ts_str,
                path=r.path or "",
                latency_ms=int(r.latency_ms or 0),
                cognition_cost_total_usd=r.cognition_cost_total_usd,
                fe_budget_utilisation=fe_util,
                intent_dispatched=bool(r.intent_dispatched),
            ))

        # Per-minute buckets for last 30 minutes
        from datetime import datetime, timedelta
        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=30)
        bucket_map: dict[str, TimelineBucket] = {}

        for r in records:
            if r.timestamp is None:
                continue
            ts = r.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts < cutoff:
                continue
            # Truncate to minute
            minute_key = ts.strftime("%Y-%m-%dT%H:%M")
            if minute_key not in bucket_map:
                bucket_map[minute_key] = TimelineBucket(minute=minute_key)
            bucket = bucket_map[minute_key]
            path = r.path or ""
            if path == "fast":
                bucket.fast += 1
            elif path == "slow":
                bucket.slow += 1
            else:
                bucket.nothing += 1

        buckets = sorted(bucket_map.values(), key=lambda b: b.minute)

        # Rolling stats
        last10 = points[:10]
        latencies = [p.latency_ms for p in last10 if p.latency_ms > 0]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

        # Cost/hr: sum costs over time span → extrapolate
        cost_per_hr = 0.0
        costs = [p.cognition_cost_total_usd for p in points if p.cognition_cost_total_usd is not None]
        if costs and len(points) >= 2:
            first_ts_str = points[-1].timestamp
            last_ts_str = points[0].timestamp
            try:
                t0 = datetime.fromisoformat(first_ts_str.rstrip("Z")).replace(tzinfo=UTC)
                t1 = datetime.fromisoformat(last_ts_str.rstrip("Z")).replace(tzinfo=UTC)
                span_hr = (t1 - t0).total_seconds() / 3600
                if span_hr > 0:
                    cost_per_hr = round(sum(costs) / span_hr, 6)
            except Exception:
                pass

        # Decisions/min over last 30 min window
        recent_count = sum(b.fast + b.slow + b.nothing for b in buckets)
        decisions_per_min = round(recent_count / 30, 2) if buckets else 0.0

        return TimelineResponse(
            points=points,
            buckets=buckets,
            decisions_per_min=decisions_per_min,
            avg_latency_last10_ms=round(avg_lat, 1),
            cost_per_hr_usd=cost_per_hr,
        )
    except Exception as exc:
        logger.warning("nova_timeline_error", error=str(exc))
        return TimelineResponse()


# ─── Goal History ─────────────────────────────────────────────────


class GoalHistoryItem(EOSBaseModel):
    id: str
    description: str
    target_domain: str = ""
    success_criteria: str = ""
    source: str = ""
    status: str = ""
    priority: float = 0.5
    importance: float = 0.5
    progress: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    drive_alignment: dict[str, float] = {}
    intents_issued: int = 0
    persisted: bool = True


class GoalHistoryResponse(EOSBaseModel):
    goals: list[GoalHistoryItem] = []
    total: int = 0
    persistence_active: bool = False


@router.get("/goals/history", response_model=GoalHistoryResponse)
async def get_goals_history(
    request: Request,
    limit: int = 50,
    status: str = "achieved",
) -> GoalHistoryResponse:
    """
    Query completed/abandoned goals from Neo4j.
    Returns goals with full lifecycle metadata.
    """
    neo4j = getattr(request.app.state, "neo4j", None)
    if neo4j is None:
        return GoalHistoryResponse()

    try:
        limit = min(max(1, limit), 200)
        valid_statuses = {"achieved", "abandoned", "suspended"}
        status_filter = status if status in valid_statuses else "achieved"

        rows = await neo4j.execute_read(
            """
            MATCH (g:Goal {status: $status})
            RETURN g
            ORDER BY g.updated_at DESC
            LIMIT $limit
            """,
            {"status": status_filter, "limit": limit},
        )

        goals: list[GoalHistoryItem] = []
        for row in rows:
            g = row.get("g", {}) or {}
            created = g.get("created_at") or ""
            updated = g.get("updated_at") or ""
            created_str = created.isoformat() if hasattr(created, "isoformat") else str(created)
            updated_str = updated.isoformat() if hasattr(updated, "isoformat") else str(updated)

            # Drive alignment stored as individual props
            alignment: dict[str, float] = {}
            for drive in ("coherence", "care", "growth", "honesty"):
                val = g.get(f"drive_{drive}")
                if val is not None:
                    alignment[drive] = float(val)

            goals.append(GoalHistoryItem(
                id=str(g.get("id", "")),
                description=str(g.get("description", "")),
                target_domain=str(g.get("target_domain", "") or ""),
                success_criteria=str(g.get("success_criteria", "") or ""),
                source=str(g.get("source", "") or ""),
                status=str(g.get("status", "") or ""),
                priority=float(g.get("priority", 0.5)),
                importance=float(g.get("importance", 0.5)),
                progress=float(g.get("progress", 0.0)),
                created_at=created_str,
                updated_at=updated_str,
                drive_alignment=alignment,
                intents_issued=int(g.get("intents_issued", 0)),
                persisted=True,
            ))

        return GoalHistoryResponse(
            goals=goals,
            total=len(goals),
            persistence_active=True,
        )
    except Exception as exc:
        logger.warning("nova_goals_history_error", error=str(exc))
        return GoalHistoryResponse()


@router.post("/goals")
async def inject_goal(request: Request) -> JSONResponse:
    """
    Inject a goal into Nova (governance or testing use).
    Body: {description, source?, priority?, urgency?, importance?, target_domain?, success_criteria?}
    """
    nova = getattr(request.app.state, "nova", None)
    if nova is None:
        return JSONResponse(status_code=503, content={"error": "Nova unavailable"})
    try:
        body = await request.json()
        description = str(body.get("description", "")).strip()
        if not description:
            return JSONResponse(status_code=400, content={"error": "description is required"})

        from primitives.common import DriveAlignmentVector, new_id
        from systems.nova.types import Goal, GoalSource, GoalStatus

        source_str = str(body.get("source", "governance"))
        try:
            source = GoalSource(source_str)
        except ValueError:
            source = GoalSource.GOVERNANCE

        goal = Goal(
            id=new_id(),
            description=description,
            target_domain=str(body.get("target_domain", "") or ""),
            success_criteria=str(body.get("success_criteria", "") or ""),
            source=source,
            priority=float(body.get("priority", 0.5)),
            urgency=float(body.get("urgency", 0.3)),
            importance=float(body.get("importance", 0.5)),
            drive_alignment=DriveAlignmentVector(),
            status=GoalStatus.ACTIVE,
        )

        nova.add_goal(goal)
        logger.info("nova_goal_injected_via_api", goal_id=goal.id, description=description[:80])
        return JSONResponse(status_code=201, content={"goal_id": goal.id, "description": description})
    except Exception as exc:
        logger.warning("nova_inject_goal_error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})
