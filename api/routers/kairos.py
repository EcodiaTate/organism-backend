"""
EcodiaOS - Kairos (Causal Invariant Mining) API Router

Exposes the Kairos causal mining pipeline state for observability and frontend:

  GET  /api/v1/kairos/health             - Health status and pipeline metrics
  GET  /api/v1/kairos/ledger             - Intelligence contribution ledger summary
  GET  /api/v1/kairos/counter-invariants - Counter-invariant stats across all invariants
  GET  /api/v1/kairos/tier3              - Tier 3 (substrate-independent) invariants
  GET  /api/v1/kairos/step-changes       - Recent intelligence ratio step changes
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Query, Request

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.kairos")

router = APIRouter(prefix="/api/v1/kairos", tags=["kairos"])


# ─── Response Models ──────────────────────────────────────────────


class KairosHealthOut(EOSBaseModel):
    status: str = "unknown"
    pipeline_runs: int = 0
    fovea_events_received: int = 0
    evo_events_received: int = 0
    invariants_created: int = 0
    tier3_discoveries: int = 0
    counter_invariants_found: int = 0
    step_changes: int = 0
    hierarchy: dict[str, Any] = {}
    intelligence_ledger: dict[str, Any] = {}
    correlation_miner: dict[str, Any] = {}
    direction_tester: dict[str, Any] = {}
    confounder_analyzer: dict[str, Any] = {}
    distiller: dict[str, Any] = {}
    counter_detector: dict[str, Any] = {}


class LedgerSummaryOut(EOSBaseModel):
    invariants_tracked: int = 0
    total_savings: float = 0.0
    total_observations_covered: int = 0
    mean_ratio_contribution: float = 0.0
    top_contributors: list[dict[str, Any]] = []


class CounterInvariantSummaryOut(EOSBaseModel):
    total_invariants: int = 0
    invariants_with_violations: int = 0
    total_violations: int = 0
    refined_scopes: int = 0
    violation_details: list[dict[str, Any]] = []


class Tier3InvariantOut(EOSBaseModel):
    id: str
    abstract_form: str = ""
    domain_count: int = 0
    substrate_count: int = 0
    hold_rate: float = 0.0
    intelligence_ratio_contribution: float = 0.0
    description_length_bits: float = 0.0
    applicable_domains: list[str] = []
    untested_domains: list[str] = []
    violation_count: int = 0
    refined_scope: str = ""
    distilled: bool = False
    is_minimal: bool = False


class StepChangeOut(EOSBaseModel):
    invariant_id: str
    abstract_form: str = ""
    intelligence_ratio_contribution: float = 0.0
    description_length_bits: float = 0.0
    tier: int = 1
    domain_count: int = 0


# ─── Helper ───────────────────────────────────────────────────────


def _get_kairos(request: Request) -> Any | None:
    return getattr(request.app.state, "kairos", None)


# ─── Endpoints ────────────────────────────────────────────────────


@router.get("/health", response_model=KairosHealthOut)
async def get_health(request: Request) -> KairosHealthOut:
    """
    Kairos service health and pipeline metrics.

    Returns pipeline run counts, event ingestion counters, hierarchy state,
    and per-stage statistics.
    """
    kairos = _get_kairos(request)
    if kairos is None:
        return KairosHealthOut(status="unavailable")
    try:
        health = await kairos.health()
        return KairosHealthOut(**health)
    except Exception as exc:
        logger.warning("kairos_health_error", error=str(exc))
        return KairosHealthOut(status="degraded")


@router.get("/ledger", response_model=LedgerSummaryOut)
async def get_ledger(request: Request) -> LedgerSummaryOut:
    """
    Intelligence contribution ledger summary.

    Returns total description-length savings, observations covered, and
    the top-5 highest-contributing invariants ranked by compression ratio.
    """
    kairos = _get_kairos(request)
    if kairos is None:
        return LedgerSummaryOut()
    summary = kairos.intelligence_ledger.summary()
    return LedgerSummaryOut(**summary)


@router.get("/counter-invariants", response_model=CounterInvariantSummaryOut)
async def get_counter_invariants(request: Request) -> CounterInvariantSummaryOut:
    """
    Counter-invariant detection summary across all invariants.

    Returns counts of violations found, invariants with refined scopes,
    and per-invariant violation details for invariants with violation_count > 0.
    """
    kairos = _get_kairos(request)
    if kairos is None:
        return CounterInvariantSummaryOut()

    all_invariants = kairos.hierarchy.get_all()
    total_violations = sum(inv.violation_count for inv in all_invariants)
    invariants_with_violations = sum(
        1 for inv in all_invariants if inv.violation_count > 0
    )
    refined_scopes = sum(
        1 for inv in all_invariants if inv.refined_scope
    )

    violation_details = [
        {
            "invariant_id": inv.id,
            "abstract_form": inv.abstract_form,
            "violation_count": inv.violation_count,
            "refined_scope": inv.refined_scope,
            "tier": int(inv.tier),
        }
        for inv in all_invariants
        if inv.violation_count > 0
    ]

    return CounterInvariantSummaryOut(
        total_invariants=len(all_invariants),
        invariants_with_violations=invariants_with_violations,
        total_violations=total_violations,
        refined_scopes=refined_scopes,
        violation_details=violation_details,
    )


@router.get("/tier3", response_model=list[Tier3InvariantOut])
async def get_tier3_invariants(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[Tier3InvariantOut]:
    """
    Tier 3 (substrate-independent) invariants.

    Returns the most recently discovered Tier 3 causal invariants - those
    that hold across 4+ domains spanning 3+ substrates.
    """
    kairos = _get_kairos(request)
    if kairos is None:
        return []

    from primitives.causal import CausalInvariantTier

    tier3 = kairos.hierarchy.get_by_tier(CausalInvariantTier.TIER_3_SUBSTRATE)

    return [
        Tier3InvariantOut(
            id=inv.id,
            abstract_form=inv.abstract_form,
            domain_count=inv.domain_count,
            substrate_count=inv.substrate_count,
            hold_rate=inv.invariance_hold_rate,
            intelligence_ratio_contribution=inv.intelligence_ratio_contribution,
            description_length_bits=inv.description_length_bits,
            applicable_domains=[d.domain for d in inv.applicable_domains],
            untested_domains=list(inv.untested_domains),
            violation_count=inv.violation_count,
            refined_scope=inv.refined_scope,
            distilled=inv.distilled,
            is_minimal=inv.is_minimal,
        )
        for inv in tier3[:limit]
    ]


@router.get("/step-changes", response_model=list[StepChangeOut])
async def get_step_changes(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[StepChangeOut]:
    """
    Invariants that have caused intelligence ratio step changes.

    Returns invariants ranked by their intelligence ratio contribution
    (proxy for step-change magnitude), highest first.
    """
    kairos = _get_kairos(request)
    if kairos is None:
        return []

    ranked = kairos.intelligence_ledger.rank_by_value()

    # Resolve back to invariant metadata for richer output
    results: list[StepChangeOut] = []
    for contribution in ranked[:limit]:
        inv = kairos.hierarchy._find(contribution.invariant_id)  # noqa: SLF001
        if inv is not None:
            results.append(
                StepChangeOut(
                    invariant_id=inv.id,
                    abstract_form=inv.abstract_form,
                    intelligence_ratio_contribution=contribution.intelligence_ratio_contribution,
                    description_length_bits=contribution.invariant_length,
                    tier=int(inv.tier),
                    domain_count=inv.domain_count,
                )
            )

    return results
