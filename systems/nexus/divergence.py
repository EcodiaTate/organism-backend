"""
EcodiaOS - Nexus: Instance Divergence Measurer

Measures how different two EOS instances are across five dimensions.
Divergence is the fuel of triangulation - the more different the
compression paths, the more valuable any convergence between them.

Five dimensions (weighted):
  1. Domain diversity     (0.25): overlap of domain coverage maps
  2. Structural diversity (0.30): world model schema structural difference
  3. Attentional diversity(0.20): Fovea weight profile difference
  4. Hypothesis diversity (0.15): active hypothesis overlap
  5. Temporal divergence  (0.10): age/experience gap

Classification thresholds:
  - SAME_KIND:     < 0.2 - near-duplicate, zero triangulation value
  - RELATED_KIND:  < 0.5 - same species, different subspecies
  - DISTINCT_KIND: >= 0.5 - true speciation threshold
  - ALIEN_KIND:    >= 0.8 - convergence from here is near-proof
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.nexus.types import (
    DivergenceDimensionScore,
    DivergenceScore,
    InstanceDivergenceProfile,
)

if TYPE_CHECKING:
    from systems.nexus.protocols import (
        EvoHypothesisSourceProtocol,
        FoveaAttentionProtocol,
        LogosWorldModelProtocol,
    )

logger = structlog.get_logger("nexus.divergence")

# Dimension weights (must sum to 1.0)
_DOMAIN_WEIGHT = 0.25
_STRUCTURAL_WEIGHT = 0.30
_ATTENTIONAL_WEIGHT = 0.20
_HYPOTHESIS_WEIGHT = 0.15
_TEMPORAL_WEIGHT = 0.10


class InstanceDivergenceMeasurer:
    """
    Measures multi-dimensional divergence between EOS instances.

    Uses dependency injection: local world model and attention profile
    are read via protocols. Remote profiles are passed as data objects
    (obtained from federation).
    """

    def __init__(
        self,
        *,
        world_model: LogosWorldModelProtocol | None = None,
        fovea: FoveaAttentionProtocol | None = None,
        evo: EvoHypothesisSourceProtocol | None = None,
        local_instance_id: str = "",
    ) -> None:
        self._world_model = world_model
        self._fovea = fovea
        self._evo = evo
        self._local_instance_id = local_instance_id

    def build_local_profile(self) -> InstanceDivergenceProfile:
        """Build a divergence profile from local state."""
        profile = InstanceDivergenceProfile(
            instance_id=self._local_instance_id,
            captured_at=utc_now(),
        )

        if self._world_model is not None:
            profile.domain_coverage = self._world_model.get_domain_coverage()
            profile.structural_fingerprint = self._world_model.get_structural_fingerprint()
            profile.total_schemas = self._world_model.get_total_schemas()
            profile.total_experiences = self._world_model.get_total_experiences()

        if self._fovea is not None:
            profile.attention_weights = self._fovea.get_attention_weights()

        if self._evo is not None:
            profile.active_hypothesis_ids = self._evo.get_active_hypothesis_ids()

        return profile

    def measure(
        self,
        local_profile: InstanceDivergenceProfile,
        remote_profile: InstanceDivergenceProfile,
    ) -> DivergenceScore:
        """
        Measure divergence between a local and remote profile.

        Returns a DivergenceScore with all five dimension scores,
        overall weighted score, and classification.
        """
        domain = _measure_domain_diversity(local_profile, remote_profile)
        structural = _measure_structural_diversity(local_profile, remote_profile)
        attentional = _measure_attentional_diversity(local_profile, remote_profile)
        hypothesis = _measure_hypothesis_diversity(local_profile, remote_profile)
        temporal = _measure_temporal_divergence(local_profile, remote_profile)

        score = DivergenceScore(
            instance_a_id=local_profile.instance_id,
            instance_b_id=remote_profile.instance_id,
            domain_diversity=domain,
            structural_diversity=structural,
            attentional_diversity=attentional,
            hypothesis_diversity=hypothesis,
            temporal_divergence=temporal,
            measured_at=utc_now(),
        )

        logger.info(
            "divergence_measured",
            local=local_profile.instance_id,
            remote=remote_profile.instance_id,
            overall=score.overall,
            classification=score.classification.value,
            domain=domain.score,
            structural=structural.score,
            attentional=attentional.score,
            hypothesis=hypothesis.score,
            temporal=temporal.score,
        )

        return score


# ─── Dimension Measurement Functions ─────────────────────────────


def _measure_domain_diversity(
    local: InstanceDivergenceProfile,
    remote: InstanceDivergenceProfile,
) -> DivergenceDimensionScore:
    """
    Domain diversity: 1 - Jaccard similarity of domain coverage sets.

    No overlap = 1.0 (maximally diverse). Full overlap = 0.0 (identical).
    """
    local_domains = set(local.domain_coverage)
    remote_domains = set(remote.domain_coverage)

    if not local_domains and not remote_domains:
        diversity = 0.0  # Both empty = not diverse, just uninformed
    elif not local_domains or not remote_domains:
        diversity = 1.0  # One has domains, other doesn't = diverse
    else:
        union = len(local_domains | remote_domains)
        intersection = len(local_domains & remote_domains)
        jaccard = intersection / union if union > 0 else 0.0
        diversity = 1.0 - jaccard

    return DivergenceDimensionScore(
        dimension="domain_diversity",
        score=diversity,
        weight=_DOMAIN_WEIGHT,
        weighted_score=diversity * _DOMAIN_WEIGHT,
        details={
            "local_domain_count": len(local_domains),
            "remote_domain_count": len(remote_domains),
            "overlap_count": (
                len(local_domains & remote_domains)
                if local_domains and remote_domains
                else 0
            ),
        },
    )


def _measure_structural_diversity(
    local: InstanceDivergenceProfile,
    remote: InstanceDivergenceProfile,
) -> DivergenceDimensionScore:
    """
    Structural diversity: world model schema topology difference.

    If structural fingerprints are available, hash comparison gives
    binary same/different. Schema count ratio gives a softer gradient.
    """
    if local.structural_fingerprint and remote.structural_fingerprint:
        fingerprint_match = (
            local.structural_fingerprint == remote.structural_fingerprint
        )
        fingerprint_diversity = 0.0 if fingerprint_match else 1.0
    else:
        fingerprint_diversity = 0.5  # Unknown, neutral

    max_schemas = max(local.total_schemas, remote.total_schemas, 1)
    min_schemas = min(local.total_schemas, remote.total_schemas)
    count_ratio = min_schemas / max_schemas
    count_diversity = 1.0 - count_ratio

    if local.structural_fingerprint and remote.structural_fingerprint:
        diversity = fingerprint_diversity * 0.7 + count_diversity * 0.3
    else:
        diversity = count_diversity

    return DivergenceDimensionScore(
        dimension="structural_diversity",
        score=min(diversity, 1.0),
        weight=_STRUCTURAL_WEIGHT,
        weighted_score=min(diversity, 1.0) * _STRUCTURAL_WEIGHT,
        details={
            "fingerprint_match": fingerprint_diversity == 0.0,
            "local_schemas": local.total_schemas,
            "remote_schemas": remote.total_schemas,
        },
    )


def _measure_attentional_diversity(
    local: InstanceDivergenceProfile,
    remote: InstanceDivergenceProfile,
) -> DivergenceDimensionScore:
    """
    Attentional diversity: Fovea weight profile difference.

    Measured as normalised L1 distance between attention weight vectors.
    Different attention = different compression priorities = more diverse.
    """
    local_weights = local.attention_weights
    remote_weights = remote.attention_weights

    if not local_weights and not remote_weights:
        diversity = 0.0
    elif not local_weights or not remote_weights:
        diversity = 0.5  # One has weights, other doesn't
    else:
        all_keys = set(local_weights) | set(remote_weights)
        total_diff = sum(
            abs(local_weights.get(k, 0.0) - remote_weights.get(k, 0.0))
            for k in all_keys
        )
        # Normalise: max possible L1 distance = 2 * |keys| (each in [0,1])
        max_diff = 2.0 * len(all_keys) if all_keys else 1.0
        diversity = total_diff / max_diff

    return DivergenceDimensionScore(
        dimension="attentional_diversity",
        score=min(diversity, 1.0),
        weight=_ATTENTIONAL_WEIGHT,
        weighted_score=min(diversity, 1.0) * _ATTENTIONAL_WEIGHT,
        details={
            "local_weight_count": len(local_weights),
            "remote_weight_count": len(remote_weights),
        },
    )


def _measure_hypothesis_diversity(
    local: InstanceDivergenceProfile,
    remote: InstanceDivergenceProfile,
) -> DivergenceDimensionScore:
    """
    Hypothesis diversity: active hypothesis overlap.

    1 - Jaccard similarity of active hypothesis ID sets.
    No overlap = maximally diverse hypothesis spaces.
    """
    local_hyps = set(local.active_hypothesis_ids)
    remote_hyps = set(remote.active_hypothesis_ids)

    if not local_hyps and not remote_hyps:
        diversity = 0.0
    elif not local_hyps or not remote_hyps:
        diversity = 1.0
    else:
        union = len(local_hyps | remote_hyps)
        intersection = len(local_hyps & remote_hyps)
        jaccard = intersection / union if union > 0 else 0.0
        diversity = 1.0 - jaccard

    return DivergenceDimensionScore(
        dimension="hypothesis_diversity",
        score=diversity,
        weight=_HYPOTHESIS_WEIGHT,
        weighted_score=diversity * _HYPOTHESIS_WEIGHT,
        details={
            "local_hypothesis_count": len(local_hyps),
            "remote_hypothesis_count": len(remote_hyps),
            "overlap_count": (
                len(local_hyps & remote_hyps)
                if local_hyps and remote_hyps
                else 0
            ),
        },
    )


def _measure_temporal_divergence(
    local: InstanceDivergenceProfile,
    remote: InstanceDivergenceProfile,
) -> DivergenceDimensionScore:
    """
    Temporal divergence: age/experience gap.

    Measured from total_experiences and born_at gap.
    Large experience gaps mean different developmental stages.
    """
    max_exp = max(local.total_experiences, remote.total_experiences, 1)
    min_exp = min(local.total_experiences, remote.total_experiences)
    exp_ratio = min_exp / max_exp
    exp_diversity = 1.0 - exp_ratio

    # Age gap (hours)
    age_gap_seconds = abs(
        (local.born_at - remote.born_at).total_seconds()
    )
    # Sigmoid normalisation: half-saturation at 7 days (604800 seconds)
    age_diversity = age_gap_seconds / (age_gap_seconds + 604800.0)

    diversity = exp_diversity * 0.6 + age_diversity * 0.4

    return DivergenceDimensionScore(
        dimension="temporal_divergence",
        score=min(diversity, 1.0),
        weight=_TEMPORAL_WEIGHT,
        weighted_score=min(diversity, 1.0) * _TEMPORAL_WEIGHT,
        details={
            "local_experiences": local.total_experiences,
            "remote_experiences": remote.total_experiences,
            "age_gap_hours": age_gap_seconds / 3600.0,
        },
    )


def compute_economic_divergence(
    profiles: list[InstanceDivergenceProfile],
) -> dict[str, float]:
    """
    Compute revenue-per-strategy variance across a set of instance profiles.

    Returns a dict mapping each instance_id to its economic_divergence score
    (0.0 = converged, 1.0 = diverged).

    Algorithm:
      For each strategy key (union across all profiles), compute the
      coefficient of variation (CV = std / mean) of revenue rates.
      Each instance's economic_divergence is the mean CV of strategies where
      it has a non-zero rate, normalised to [0, 1] via tanh(CV).

    Instances with no strategy data (empty strategy_revenue_rates) receive
    economic_divergence = 0.0 (unknown, not diverged).
    """
    if len(profiles) < 2:
        # Cannot compute variance with a single instance
        return {p.instance_id: 0.0 for p in profiles}

    # Collect all strategy keys across all profiles
    all_strategies: set[str] = set()
    for profile in profiles:
        all_strategies.update(profile.strategy_revenue_rates.keys())

    if not all_strategies:
        return {p.instance_id: 0.0 for p in profiles}

    import math

    # Per-strategy: compute revenue rates across all profiles
    strategy_rates: dict[str, list[float]] = {
        s: [p.strategy_revenue_rates.get(s, 0.0) for p in profiles]
        for s in all_strategies
    }

    # Per-strategy coefficient of variation (CV)
    strategy_cv: dict[str, float] = {}
    for strategy, rates in strategy_rates.items():
        mean_r = sum(rates) / len(rates)
        if mean_r == 0.0:
            strategy_cv[strategy] = 0.0
            continue
        variance = sum((r - mean_r) ** 2 for r in rates) / len(rates)
        std = math.sqrt(variance)
        strategy_cv[strategy] = std / mean_r  # CV (dimensionless)

    # Per-instance economic_divergence: mean CV over strategies where
    # the instance contributes a non-zero rate
    result: dict[str, float] = {}
    for profile in profiles:
        active_cvs = [
            strategy_cv[s]
            for s, rate in profile.strategy_revenue_rates.items()
            if rate > 0.0 and s in strategy_cv
        ]
        if not active_cvs:
            result[profile.instance_id] = 0.0
        else:
            mean_cv = sum(active_cvs) / len(active_cvs)
            # tanh normalisation: CV=1 → 0.76, CV=2 → 0.96
            result[profile.instance_id] = float(math.tanh(mean_cv))

    return result
