"""
EcodiaOS — SACM Placement Optimizer

Scores SubstrateOffers against WorkloadDescriptors and produces ranked
placement plans.  This is the decision engine of the Market & Optimization
subsystem.

Cost function:
  total_cost(workload, offer) =
      workload.resources.cpu_vcpu     × duration_s × offer.price_cpu_per_vcpu_s
    + workload.resources.memory_gib   × duration_s × offer.price_mem_per_gib_s
    + workload.resources.gpu_units    × duration_s × offer.price_gpu_per_unit_s
    + workload.resources.storage_gib  × duration_s × offer.price_storage_per_gib_s
    + workload.resources.egress_gib   × offer.price_egress_per_gib

  where duration_s = workload.estimated_duration_s

Composite scoring:
  The optimizer doesn't just minimise cost — it produces a composite score
  that weights cost, latency, and trust according to workload priority:

    score = (
        w_cost    × normalised_cost
      + w_latency × normalised_latency
      + w_trust   × (1 - offer.trust_score)
    )

  Lower score = better placement.  Priority determines the weight vector:
    CRITICAL → latency-dominant (get it done fast, cost secondary)
    LOW/BATCH → cost-dominant (minimise spend, latency flexible)
    NORMAL → balanced

Placement plan:
  optimize_placement() returns a PlacementPlan: a ranked list of
  ScoredPlacement objects (offer + score breakdown + estimated cost),
  sorted best-first.  The execution engine pops the top placement and
  falls back to the next on failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped
from systems.sacm.workload import (
    WorkloadDescriptor,
    WorkloadPriority,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from systems.sacm.oracle import PricingSurfaceSnapshot
    from systems.sacm.providers.base import SubstrateOffer

logger = structlog.get_logger("systems.sacm.optimizer")


# ─── Cost Function ────────────────────────────────────────────────


class CostBreakdown(EOSBaseModel):
    """Itemised cost for running a workload on a specific offer."""

    cpu_cost_usd: float = 0.0
    memory_cost_usd: float = 0.0
    gpu_cost_usd: float = 0.0
    storage_cost_usd: float = 0.0
    egress_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    duration_s: float = 0.0

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "cpu": self.cpu_cost_usd,
            "memory": self.memory_cost_usd,
            "gpu": self.gpu_cost_usd,
            "storage": self.storage_cost_usd,
            "egress": self.egress_cost_usd,
            "total": self.total_cost_usd,
        }


def compute_total_cost(
    workload: WorkloadDescriptor,
    offer: SubstrateOffer,
) -> CostBreakdown:
    """
    Compute the total estimated cost of running a workload on an offer.

    Implements the exact cost function:
      total = Σ(resource_quantity × duration × per_unit_per_second_price)
            + egress × per_gib_price

    All prices are in USD (the offer has already converted from native
    currency via the provider's fetch_offers).

    Args:
        workload: The workload to cost.
        offer:    The substrate offer to price against.

    Returns:
        CostBreakdown with itemised and total costs.
    """
    res = workload.resources
    dur = workload.estimated_duration_s

    cpu_cost = res.cpu_vcpu * dur * offer.price_cpu_per_vcpu_s
    mem_cost = res.memory_gib * dur * offer.price_mem_per_gib_s
    gpu_cost = res.gpu_units * dur * offer.price_gpu_per_unit_s
    sto_cost = res.storage_gib * dur * offer.price_storage_per_gib_s
    egr_cost = res.egress_gib * offer.price_egress_per_gib

    total = cpu_cost + mem_cost + gpu_cost + sto_cost + egr_cost

    return CostBreakdown(
        cpu_cost_usd=cpu_cost,
        memory_cost_usd=mem_cost,
        gpu_cost_usd=gpu_cost,
        storage_cost_usd=sto_cost,
        egress_cost_usd=egr_cost,
        total_cost_usd=total,
        duration_s=dur,
    )


# ─── Priority Weight Vectors ─────────────────────────────────────


class ScoreWeights(EOSBaseModel):
    """Weights for the composite placement score. Must sum to 1.0."""

    w_cost: float = 0.5
    w_latency: float = 0.3
    w_trust: float = 0.2


# Pre-defined weight vectors per priority level.
# CRITICAL → latency-dominant: get it done now, cost is secondary.
# HIGH → latency-leaning but cost-aware.
# NORMAL → balanced.
# LOW → cost-dominant.
# BATCH → pure cost minimisation.
_PRIORITY_WEIGHTS: dict[WorkloadPriority, ScoreWeights] = {
    WorkloadPriority.CRITICAL: ScoreWeights(w_cost=0.10, w_latency=0.70, w_trust=0.20),
    WorkloadPriority.HIGH:     ScoreWeights(w_cost=0.25, w_latency=0.50, w_trust=0.25),
    WorkloadPriority.NORMAL:   ScoreWeights(w_cost=0.45, w_latency=0.30, w_trust=0.25),
    WorkloadPriority.LOW:      ScoreWeights(w_cost=0.65, w_latency=0.15, w_trust=0.20),
    WorkloadPriority.BATCH:    ScoreWeights(w_cost=0.80, w_latency=0.05, w_trust=0.15),
}


def weights_for_priority(priority: WorkloadPriority) -> ScoreWeights:
    """Look up the composite score weights for a given workload priority."""
    return _PRIORITY_WEIGHTS.get(priority, _PRIORITY_WEIGHTS[WorkloadPriority.NORMAL])


# ─── Scored Placement ────────────────────────────────────────────


class ScoredPlacement(EOSBaseModel):
    """A single candidate placement: an offer scored against a workload."""

    offer: SubstrateOffer
    cost_breakdown: CostBreakdown
    composite_score: float = 0.0
    """Lower is better. Composite of normalised cost, latency, and trust."""

    # Score components (pre-normalisation, for transparency)
    raw_cost_usd: float = 0.0
    raw_latency_s: float = 0.0
    trust_score: float = 0.0

    # Normalised components [0, 1] (used in composite)
    norm_cost: float = 0.0
    norm_latency: float = 0.0
    norm_distrust: float = 0.0

    # Weights used
    weights: ScoreWeights = Field(default_factory=ScoreWeights)

    # Constraint violations
    exceeds_max_cost: bool = False
    exceeds_max_latency: bool = False

    @property
    def is_feasible(self) -> bool:
        """Whether this placement satisfies all hard constraints."""
        return not self.exceeds_max_cost and not self.exceeds_max_latency


# ─── Placement Plan ───────────────────────────────────────────────


class PlacementPlan(Identified, Timestamped):
    """
    Ranked list of candidate placements for a workload.

    The execution engine should try placements in order (best-first),
    falling back to the next on failure.
    """

    workload_id: str
    placements: list[ScoredPlacement] = Field(default_factory=list)
    feasible_count: int = 0
    total_candidates: int = 0
    error: str = ""

    @property
    def best(self) -> ScoredPlacement | None:
        """Return the top-ranked feasible placement, or None."""
        for p in self.placements:
            if p.is_feasible:
                return p
        return None

    @property
    def has_feasible(self) -> bool:
        return self.feasible_count > 0

    def feasible_placements(self) -> list[ScoredPlacement]:
        """Return only placements that satisfy all hard constraints."""
        return [p for p in self.placements if p.is_feasible]


# ─── Optimizer ────────────────────────────────────────────────────


def optimize_placement(
    workload: WorkloadDescriptor,
    surface: PricingSurfaceSnapshot,
    weights_override: ScoreWeights | None = None,
) -> PlacementPlan:
    """
    Score all eligible SubstrateOffers and produce a ranked PlacementPlan.

    Algorithm:
      1. Filter the pricing surface to eligible offers (class, resources,
         region, blocked providers).
      2. Compute total_cost for each eligible offer.
      3. Check hard constraints (max_cost_usd, max_latency_s).
      4. Normalise cost and latency across the candidate set to [0, 1].
      5. Compute composite score using priority-derived weights.
      6. Sort ascending (lower = better) and return.

    Normalisation:
      norm_cost    = (cost - min_cost) / (max_cost - min_cost)     if range > 0 else 0
      norm_latency = (latency - min_lat) / (max_lat - min_lat)     if range > 0 else 0
      norm_distrust = 1 - offer.trust_score  (already in [0, 1])

    When all candidates have identical cost/latency, normalised values
    are 0 (no discrimination on that axis) and trust becomes the tiebreaker.

    Args:
        workload:         The workload to place.
        surface:          Pricing surface snapshot from the oracle.
        weights_override: Override the priority-derived weight vector.

    Returns:
        PlacementPlan with ranked ScoredPlacements.
    """
    # Step 1: filter eligible offers
    eligible = surface.filter_eligible(
        offload_class=workload.offload_class,
        resources=workload.resources,
        regions=workload.allowed_regions,
        blocked_providers=workload.blocked_providers,
    )

    if not eligible:
        logger.warning(
            "no_eligible_offers",
            workload_id=workload.id,
            offload_class=workload.offload_class,
        )
        return PlacementPlan(
            workload_id=workload.id,
            error="no eligible offers in current pricing surface",
        )

    weights = weights_override or weights_for_priority(workload.priority)

    # Step 2: compute costs and raw latencies
    candidates: list[tuple[SubstrateOffer, CostBreakdown, float]] = []
    for offer in eligible:
        cost = compute_total_cost(workload, offer)
        # Estimated total latency = execution time + provider overhead
        estimated_latency = workload.estimated_duration_s + offer.avg_latency_overhead_s
        candidates.append((offer, cost, estimated_latency))

    # Step 3: extract ranges for normalisation
    costs = [c[1].total_cost_usd for c in candidates]
    latencies = [c[2] for c in candidates]

    min_cost, max_cost = min(costs), max(costs)
    min_lat, max_lat = min(latencies), max(latencies)

    cost_range = max_cost - min_cost
    lat_range = max_lat - min_lat

    # Step 4+5: score each candidate
    scored: list[ScoredPlacement] = []
    for offer, cost_bd, est_latency in candidates:
        # Normalise to [0, 1]
        norm_cost = (
            (cost_bd.total_cost_usd - min_cost) / cost_range
            if cost_range > 0 else 0.0
        )
        norm_latency = (
            (est_latency - min_lat) / lat_range
            if lat_range > 0 else 0.0
        )
        norm_distrust = 1.0 - offer.trust_score

        composite = (
            weights.w_cost * norm_cost
            + weights.w_latency * norm_latency
            + weights.w_trust * norm_distrust
        )

        # Check hard constraints
        exceeds_cost = (
            workload.has_cost_constraint
            and cost_bd.total_cost_usd > workload.max_cost_usd
        )
        exceeds_latency = (
            workload.has_latency_constraint
            and est_latency > workload.max_latency_s
        )

        placement = ScoredPlacement(
            offer=offer,
            cost_breakdown=cost_bd,
            composite_score=composite,
            raw_cost_usd=cost_bd.total_cost_usd,
            raw_latency_s=est_latency,
            trust_score=offer.trust_score,
            norm_cost=norm_cost,
            norm_latency=norm_latency,
            norm_distrust=norm_distrust,
            weights=weights,
            exceeds_max_cost=exceeds_cost,
            exceeds_max_latency=exceeds_latency,
        )
        scored.append(placement)

    # Step 6: sort — feasible placements first, then by composite score
    scored.sort(key=lambda p: (not p.is_feasible, p.composite_score))

    feasible_count = sum(1 for p in scored if p.is_feasible)

    plan = PlacementPlan(
        workload_id=workload.id,
        placements=scored,
        feasible_count=feasible_count,
        total_candidates=len(scored),
    )

    logger.info(
        "placement_plan_generated",
        workload_id=workload.id,
        priority=workload.priority.name,
        candidates=len(scored),
        feasible=feasible_count,
        best_cost_usd=round(scored[0].raw_cost_usd, 6) if scored else None,
        best_score=round(scored[0].composite_score, 6) if scored else None,
    )

    return plan


# ─── Multi-workload batch placement ──────────────────────────────


def optimize_batch(
    workloads: Sequence[WorkloadDescriptor],
    surface: PricingSurfaceSnapshot,
) -> dict[str, PlacementPlan]:
    """
    Produce placement plans for a batch of workloads.

    Each workload is scored independently against the same pricing surface
    snapshot (no cross-workload resource contention modelling — that is a
    future enhancement).

    Returns a dict mapping workload_id → PlacementPlan.
    """
    return {
        w.id: optimize_placement(w, surface)
        for w in workloads
    }


# ─── Cost estimation helpers ──────────────────────────────────────


def estimate_hourly_cost(
    offer: SubstrateOffer,
    cpu_vcpu: float = 2.0,
    memory_gib: float = 4.0,
    storage_gib: float = 20.0,
    gpu_units: float = 0.0,
) -> float:
    """
    Quick estimate of hourly USD cost for a workload shape on an offer.

    Useful for Oikos BMR calculation and dashboard display.
    Does not include egress (continuous workloads don't have discrete egress).
    """
    seconds_per_hour = 3600.0
    return (
        cpu_vcpu * seconds_per_hour * offer.price_cpu_per_vcpu_s
        + memory_gib * seconds_per_hour * offer.price_mem_per_gib_s
        + gpu_units * seconds_per_hour * offer.price_gpu_per_unit_s
        + storage_gib * seconds_per_hour * offer.price_storage_per_gib_s
    )


def estimate_monthly_cost(
    offer: SubstrateOffer,
    cpu_vcpu: float = 2.0,
    memory_gib: float = 4.0,
    storage_gib: float = 20.0,
    gpu_units: float = 0.0,
) -> float:
    """Monthly cost estimate (730 hours). For Oikos burn-rate projections."""
    return estimate_hourly_cost(offer, cpu_vcpu, memory_gib, storage_gib, gpu_units) * 730.0
