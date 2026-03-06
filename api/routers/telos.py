# ruff: noqa: N815
"""
EcodiaOS — Telos (Drives as Intelligence Topology) API Router

Exposes the Telos drive topology engine for observability and frontend integration:

  GET  /api/v1/telos/health                 — Health status and lifecycle info
  GET  /api/v1/telos/report                 — Last computed EffectiveIntelligenceReport
  GET  /api/v1/telos/gap                    — Alignment gap status and trend
  GET  /api/v1/telos/drives/care            — Care coverage report details
  GET  /api/v1/telos/drives/coherence       — Coherence cost report details
  GET  /api/v1/telos/drives/growth          — Growth metrics and frontier domains
  GET  /api/v1/telos/drives/honesty         — Honesty validity report details
  GET  /api/v1/telos/bindings               — Constitutional binding verification
  POST /api/v1/telos/compute                — Trigger immediate effective_I computation
  POST /api/v1/telos/audit                  — Run 24h constitutional topology audit
  POST /api/v1/telos/policy-score           — Score a proposed policy for effective_I impact
  POST /api/v1/telos/hypothesis-rank        — Rank hypotheses by topology contribution
  POST /api/v1/telos/fragment-score         — Score world model fragment for federation sharing
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.telos")

router = APIRouter(prefix="/api/v1/telos", tags=["telos"])


# ─── Response Models ──────────────────────────────────────────────


class DriveScoreOut(EOSBaseModel):
    """Score for a single drive (0.0 = impaired, 1.0 = optimal)."""

    score: float = 0.0
    status: str = "unknown"  # optimal, degraded, critical


class CareCoverageOut(EOSBaseModel):
    """Care topology report."""

    care_multiplier: float = 1.0
    welfare_prediction_failures: int = 0
    uncovered_welfare_domains: list[str] = []
    total_I_reduction: float = 0.0


class CoherenceCostOut(EOSBaseModel):
    """Coherence topology report."""

    coherence_bonus: float = 1.0
    logical_contradictions: int = 0
    temporal_violations: int = 0
    value_conflicts: int = 0
    cross_domain_mismatches: int = 0
    total_extra_bits: float = 0.0
    effective_I_improvement_available: float = 0.0


class GrowthMetricsOut(EOSBaseModel):
    """Growth topology metrics."""

    growth_score: float = 0.0
    dI_dt: float = 0.0
    d2I_dt2: float = 0.0
    frontier_domains: list[str] = []
    novel_domain_fraction: float = 0.0
    compression_rate: float = 0.0
    stagnation_urgency: float = 0.0


class HonestyReportOut(EOSBaseModel):
    """Honesty topology report."""

    validity_coefficient: float = 1.0
    selective_attention_bias: float = 0.0
    confabulation_rate: float = 0.0
    overclaiming_rate: float = 0.0
    nominal_I_inflation: float = 0.0


class EffectiveIntelligenceReportOut(EOSBaseModel):
    """Complete effective intelligence report."""

    report_id: str
    timestamp: str
    nominal_I: float = 0.0
    effective_I: float = 0.0
    effective_dI_dt: float = 0.0
    alignment_gap: float = 0.0
    alignment_gap_warning: bool = False
    care_multiplier: float = 1.0
    coherence_bonus: float = 1.0
    honesty_coefficient: float = 1.0
    growth_rate: float = 0.0


class AlignmentGapOut(EOSBaseModel):
    """Alignment gap status and trend."""

    current_gap: float = 0.0
    gap_fraction: float = 0.0
    primary_cause: str = "unknown"
    is_widening: bool = False
    urgency: str = "nominal"
    samples_in_trend: int = 0
    gap_trend_slope_per_hour: float = 0.0


class ConstitutionalBindingsOut(EOSBaseModel):
    """Constitutional binding verification status."""

    all_bindings_intact: bool = True
    care_is_coverage: bool = True
    coherence_is_compression: bool = True
    growth_is_gradient: bool = True
    honesty_is_validity: bool = True
    violations_reviewed: int = 0
    consecutive_failures: int = 0
    is_emergency: bool = False


class TelosHealthOut(EOSBaseModel):
    """Telos service health and status."""

    status: str = "unknown"  # healthy, degraded, stopped
    initialized: bool = False
    logos_wired: bool = False
    fovea_wired: bool = False
    event_bus_wired: bool = False
    computation_loop_running: bool = False
    computation_count: int = 0
    last_computation_ms: float = 0.0
    last_effective_I: float | None = None
    last_alignment_gap: float | None = None
    recent_alignments_buffered: int = 0
    bindings_intact: bool = True
    audit_consecutive_failures: int = 0
    audit_is_emergency: bool = False


class PolicyScoreRequestIn(EOSBaseModel):
    """Request to score a proposed policy."""

    policy_id: str
    goal_description: str
    expected_welfare_impact: float = 0.0
    expected_coherence_impact: float = 0.0
    expected_honesty_impact: float = 0.0
    expected_growth_impact: float = 0.0
    expected_nominal_I_delta: float = 0.0


class PolicyScoreOut(EOSBaseModel):
    """Score for a proposed policy."""

    policy_id: str
    nominal_I_delta: float = 0.0
    effective_I_delta: float = 0.0
    care_impact: float = 0.0
    coherence_impact: float = 0.0
    honesty_impact: float = 0.0
    growth_impact: float = 0.0
    composite_score: float = 0.0
    misalignment_risk: bool = False


class HypothesisScoreRequestIn(EOSBaseModel):
    """Request to rank hypotheses."""

    hypotheses: list[dict[str, Any]] = []


class HypothesisScoreOut(EOSBaseModel):
    """Ranked hypothesis with topology contribution."""

    hypothesis_id: str
    rank: int
    care_contribution: float = 0.0
    coherence_contribution: float = 0.0
    honesty_contribution: float = 0.0
    growth_contribution: float = 0.0
    composite_contribution: float = 0.0


class FragmentScoreRequestIn(EOSBaseModel):
    """Request to score a world model fragment."""

    fragment_id: str
    domain: str
    coverage: float = 0.0
    coherence_validated: bool = False
    prediction_accuracy: float = 0.0


class FragmentScoreOut(EOSBaseModel):
    """Score for a world model fragment."""

    fragment_id: str
    score: float = 0.0  # 0.0-1.0


# ─── Helpers ─────────────────────────────────────────────────────


def _get_telos(request: Request) -> Any:
    """Safely retrieve the Telos service from app state."""
    telos = getattr(request.app.state, "telos", None)
    if telos is None:
        raise HTTPException(status_code=503, detail="Telos system not initialized")
    return telos


# ─── Endpoints ────────────────────────────────────────────────────


@router.get("/health", response_model=TelosHealthOut)
async def get_health(request: Request) -> TelosHealthOut:
    """
    Retrieve Telos service health status and lifecycle info.

    Returns initialized state, wired dependencies, computation loop status,
    last computation metrics, and constitutional binding integrity.
    """
    telos = getattr(request.app.state, "telos", None)
    if telos is None:
        return TelosHealthOut(status="unavailable", initialized=False)
    try:
        health = await telos.health()
        return TelosHealthOut(**health)
    except Exception as exc:
        logger.warning("telos_health_error", error=str(exc))
        return TelosHealthOut(status="degraded", initialized=True)


@router.get("/report", response_model=EffectiveIntelligenceReportOut)
async def get_report(request: Request) -> EffectiveIntelligenceReportOut:
    """
    Retrieve the last computed EffectiveIntelligenceReport.

    Returns effective I, nominal I, alignment gap, all drive multipliers,
    and whether an alignment gap warning was triggered.
    """
    telos = _get_telos(request)
    report = telos.integrator.last_report

    if report is None:
        raise HTTPException(status_code=503, detail="No report computed yet")

    return EffectiveIntelligenceReportOut(
        report_id=report.id,
        timestamp=report.timestamp.isoformat(),
        nominal_I=report.nominal_I,
        effective_I=report.effective_I,
        effective_dI_dt=report.effective_dI_dt,
        alignment_gap=report.alignment_gap,
        alignment_gap_warning=report.alignment_gap_warning,
        care_multiplier=report.care_multiplier,
        coherence_bonus=report.coherence_bonus,
        honesty_coefficient=report.honesty_coefficient,
        growth_rate=report.growth_rate,
    )


@router.get("/gap", response_model=AlignmentGapOut)
async def get_gap(request: Request) -> AlignmentGapOut:
    """
    Retrieve the alignment gap status and trend.

    Returns current gap (nominal_I - effective_I), whether it's widening,
    the primary cause (weakest drive), and urgency classification.
    """
    telos = _get_telos(request)
    report = telos.integrator.last_report

    if report is None:
        raise HTTPException(status_code=503, detail="No report computed yet")

    primary_cause = telos.integrator.identify_primary_alignment_gap_cause()
    trend = telos.gap_monitor.compute_trend()

    return AlignmentGapOut(
        current_gap=report.alignment_gap,
        gap_fraction=(
            report.alignment_gap / report.nominal_I if report.nominal_I > 0 else 0.0
        ),
        primary_cause=primary_cause,
        is_widening=trend.is_widening,
        urgency=trend.urgency,
        samples_in_trend=trend.samples_count,
        gap_trend_slope_per_hour=trend.slope_per_hour,
    )


@router.get("/drives/care", response_model=CareCoverageOut)
async def get_care_report(request: Request) -> CareCoverageOut:
    """Retrieve the Care topology report."""
    telos = _get_telos(request)
    report = telos.integrator.last_report
    care_report = telos.integrator.last_care_report

    if report is None or care_report is None:
        raise HTTPException(status_code=503, detail="No care report computed yet")

    return CareCoverageOut(
        care_multiplier=report.care_multiplier,
        welfare_prediction_failures=len(care_report.welfare_prediction_failures),
        uncovered_welfare_domains=care_report.uncovered_welfare_domains,
        total_I_reduction=care_report.total_effective_I_reduction,
    )


@router.get("/drives/coherence", response_model=CoherenceCostOut)
async def get_coherence_report(request: Request) -> CoherenceCostOut:
    """Retrieve the Coherence topology report."""
    telos = _get_telos(request)
    report = telos.integrator.last_report
    coherence_report = telos.integrator.last_coherence_report

    if report is None or coherence_report is None:
        raise HTTPException(status_code=503, detail="No coherence report computed yet")

    return CoherenceCostOut(
        coherence_bonus=report.coherence_bonus,
        logical_contradictions=len(coherence_report.logical_contradictions),
        temporal_violations=len(coherence_report.temporal_violations),
        value_conflicts=len(coherence_report.value_conflicts),
        cross_domain_mismatches=len(coherence_report.cross_domain_mismatches),
        total_extra_bits=coherence_report.total_extra_bits,
        effective_I_improvement_available=coherence_report.effective_I_improvement,
    )


@router.get("/drives/growth", response_model=GrowthMetricsOut)
async def get_growth_report(request: Request) -> GrowthMetricsOut:
    """Retrieve the Growth topology metrics."""
    telos = _get_telos(request)
    report = telos.integrator.last_report
    growth_metrics = telos.integrator.last_growth_metrics

    if report is None or growth_metrics is None:
        raise HTTPException(status_code=503, detail="No growth metrics computed yet")

    return GrowthMetricsOut(
        growth_score=growth_metrics.growth_score,
        dI_dt=growth_metrics.dI_dt,
        d2I_dt2=growth_metrics.d2I_dt2,
        frontier_domains=growth_metrics.frontier_domains,
        novel_domain_fraction=growth_metrics.novel_domain_fraction,
        compression_rate=growth_metrics.compression_rate,
        stagnation_urgency=(
            min(
                1.0,
                1.0 - (growth_metrics.dI_dt / telos._config.minimum_growth_rate),
            )
            if growth_metrics.dI_dt < telos._config.minimum_growth_rate
            else 0.0
        ),
    )


@router.get("/drives/honesty", response_model=HonestyReportOut)
async def get_honesty_report(request: Request) -> HonestyReportOut:
    """Retrieve the Honesty validity report."""
    telos = _get_telos(request)
    report = telos.integrator.last_report
    honesty_report = telos.integrator.last_honesty_report

    if report is None or honesty_report is None:
        raise HTTPException(status_code=503, detail="No honesty report computed yet")

    return HonestyReportOut(
        validity_coefficient=report.honesty_coefficient,
        selective_attention_bias=honesty_report.selective_attention_bias,
        confabulation_rate=honesty_report.confabulation_rate,
        overclaiming_rate=honesty_report.overclaiming_rate,
        nominal_I_inflation=honesty_report.nominal_I_inflation,
    )


@router.get("/bindings", response_model=ConstitutionalBindingsOut)
async def get_bindings(request: Request) -> ConstitutionalBindingsOut:
    """
    Retrieve constitutional binding verification status.

    Verifies all four drive bindings are intact and immutable.
    """
    telos = _get_telos(request)

    return ConstitutionalBindingsOut(
        all_bindings_intact=telos.binder.verify_bindings_intact(),
        care_is_coverage=telos.binder.CARE_IS_COVERAGE,
        coherence_is_compression=telos.binder.COHERENCE_IS_COMPRESSION,
        growth_is_gradient=telos.binder.GROWTH_IS_GRADIENT,
        honesty_is_validity=telos.binder.HONESTY_IS_VALIDITY,
        violations_reviewed=len(telos.binder.recent_violations),
        consecutive_failures=telos.auditor.consecutive_failures,
        is_emergency=telos.auditor.is_emergency,
    )


@router.post("/compute", response_model=EffectiveIntelligenceReportOut)
async def compute_now(request: Request) -> EffectiveIntelligenceReportOut:
    """
    Trigger an immediate effective_I computation.

    Returns the newly computed EffectiveIntelligenceReport.
    Raises 503 if Logos or Fovea are not wired.
    """
    telos = _get_telos(request)
    report = await telos.compute_now()

    if report is None:
        raise HTTPException(
            status_code=503, detail="Logos or Fovea not wired; cannot compute"
        )

    return EffectiveIntelligenceReportOut(
        report_id=report.id,
        timestamp=report.timestamp.isoformat(),
        nominal_I=report.nominal_I,
        effective_I=report.effective_I,
        effective_dI_dt=report.effective_dI_dt,
        alignment_gap=report.alignment_gap,
        alignment_gap_warning=report.alignment_gap_warning,
        care_multiplier=report.care_multiplier,
        coherence_bonus=report.coherence_bonus,
        honesty_coefficient=report.honesty_coefficient,
        growth_rate=report.growth_rate,
    )


@router.post("/audit", response_model=ConstitutionalBindingsOut)
async def run_audit(request: Request) -> ConstitutionalBindingsOut:
    """
    Run the 24-hour constitutional topology audit immediately.

    Verifies all four drive bindings and emits CONSTITUTIONAL_TOPOLOGY_INTACT
    or an emergency alert if violations are found.
    """
    telos = _get_telos(request)
    result = await telos.run_constitutional_audit()

    return ConstitutionalBindingsOut(
        all_bindings_intact=result.all_bindings_intact,
        care_is_coverage=result.care_is_coverage,
        coherence_is_compression=result.coherence_is_compression,
        growth_is_gradient=result.growth_is_gradient,
        honesty_is_validity=result.honesty_is_validity,
        violations_reviewed=len(result.violations_since_last_audit),
        consecutive_failures=telos.auditor.consecutive_failures,
        is_emergency=telos.auditor.is_emergency,
    )


@router.post("/policy-score", response_model=PolicyScoreOut)
async def score_policy(
    request: Request, policy: PolicyScoreRequestIn
) -> PolicyScoreOut:
    """
    Score a proposed policy by its effect on effective_I.

    Returns positive effective_I_delta for policies that improve real
    intelligence (not just nominal I). Flags misalignment_risk if nominal
    I increases but effective I decreases.
    """
    telos = _get_telos(request)

    policy_dict = {
        "expected_welfare_impact": policy.expected_welfare_impact,
        "expected_coherence_impact": policy.expected_coherence_impact,
        "expected_honesty_impact": policy.expected_honesty_impact,
        "expected_growth_impact": policy.expected_growth_impact,
        "expected_nominal_I_delta": policy.expected_nominal_I_delta,
    }

    score = telos.score_policy(policy_dict)

    return PolicyScoreOut(
        policy_id=policy.policy_id,
        nominal_I_delta=score.nominal_I_delta,
        effective_I_delta=score.effective_I_delta,
        care_impact=score.care_impact,
        coherence_impact=score.coherence_impact,
        honesty_impact=score.honesty_impact,
        growth_impact=score.growth_impact,
        composite_score=score.composite_score,
        misalignment_risk=score.misalignment_risk,
    )


@router.post("/hypothesis-rank", response_model=list[HypothesisScoreOut])
async def rank_hypotheses(
    request: Request, payload: HypothesisScoreRequestIn
) -> list[HypothesisScoreOut]:
    """
    Rank hypotheses by their contribution to the drive topology.

    Returns ranked list (highest contribution first) with per-drive
    contribution scores. Hypotheses in frontier domains or that would
    improve weak drives get higher priority.
    """
    telos = _get_telos(request)

    ranked = telos.prioritize_hypotheses(payload.hypotheses)

    return [
        HypothesisScoreOut(
            hypothesis_id=contrib.hypothesis_id,
            rank=contrib.rank,
            care_contribution=contrib.care_contribution,
            coherence_contribution=contrib.coherence_contribution,
            honesty_contribution=contrib.honesty_contribution,
            growth_contribution=contrib.growth_contribution,
            composite_contribution=contrib.composite_contribution,
        )
        for contrib in ranked
    ]


@router.post("/fragment-score", response_model=FragmentScoreOut)
async def score_fragment(
    request: Request, fragment: FragmentScoreRequestIn
) -> FragmentScoreOut:
    """
    Score a world model fragment for federation sharing.

    Returns score in [0.0, 1.0] based on Care (welfare relevance) +
    Coherence (validation status) + Accuracy (prediction quality).
    High-care, high-coherence fragments are preferred for federation sharing.
    """
    telos = _get_telos(request)

    fragment_dict = {
        "domain": fragment.domain,
        "coverage": fragment.coverage,
        "coherence_validated": fragment.coherence_validated,
        "prediction_accuracy": fragment.prediction_accuracy,
    }

    score = telos.score_fragment(fragment_dict)

    return FragmentScoreOut(fragment_id=fragment.fragment_id, score=score)


# ─── Frontend-facing shorthand endpoints ─────────────────────────────────────
# The UI polls these flat paths; they delegate to the same logic as the
# canonical /report and /drives/* endpoints above.


@router.get("/effective-i", response_model=EffectiveIntelligenceReportOut)
async def get_effective_i(request: Request) -> EffectiveIntelligenceReportOut:
    """Shorthand alias for /report — returns the last effective_I computation."""
    return await get_report(request)


@router.get("/care", response_model=CareCoverageOut)
async def get_care(request: Request) -> CareCoverageOut:
    """Shorthand alias for /drives/care."""
    return await get_care_report(request)


@router.get("/coherence", response_model=CoherenceCostOut)
async def get_coherence(request: Request) -> CoherenceCostOut:
    """Shorthand alias for /drives/coherence."""
    return await get_coherence_report(request)


@router.get("/growth", response_model=GrowthMetricsOut)
async def get_growth(request: Request) -> GrowthMetricsOut:
    """Shorthand alias for /drives/growth."""
    return await get_growth_report(request)


@router.get("/honesty", response_model=HonestyReportOut)
async def get_honesty(request: Request) -> HonestyReportOut:
    """Shorthand alias for /drives/honesty."""
    return await get_honesty_report(request)


class AlignmentSampleOut(EOSBaseModel):
    """A single alignment gap sample from the trend history."""

    nominal_I: float = 0.0
    effective_I: float = 0.0
    gap_fraction: float = 0.0
    primary_cause: str = ""
    timestamp: str = ""


@router.get("/alignment/history", response_model=list[AlignmentSampleOut])
async def get_alignment_history(
    request: Request, limit: int = 60
) -> list[AlignmentSampleOut]:
    """
    Return the last N alignment gap samples from the gap monitor's history.

    Each sample contains nominal_I, effective_I, gap_fraction, primary cause,
    and timestamp — suitable for time-series charts in the UI.
    """
    telos = _get_telos(request)
    samples = telos.gap_monitor.history
    # Most-recent last; slice to limit
    tail = samples[-limit:] if len(samples) > limit else samples
    return [
        AlignmentSampleOut(
            nominal_I=s.nominal_I,
            effective_I=s.effective_I,
            gap_fraction=s.gap_fraction,
            primary_cause=s.primary_cause,
            timestamp=s.timestamp.isoformat(),
        )
        for s in tail
    ]
