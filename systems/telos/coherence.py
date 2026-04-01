"""
EcodiaOS - Telos: Coherence Topology Engine

A contradictory world model has higher description length than a coherent one -
provably, by Shannon's theorem. If the model asserts both P and not-P, it must
carry extra bits to specify which applies in each context. Those bits describe
the contradiction, not reality.

The Coherence engine measures four types of incoherence and computes the
coherence_compression_bonus that adjusts the denominator of effective I.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.telos.types import (
    FoveaMetrics,
    IncoherenceCostReport,
    IncoherenceEntry,
    IncoherenceType,
    LogosMetrics,
    TelosConfig,
)

if TYPE_CHECKING:
    from primitives.common import DriveAlignmentVector

logger = structlog.get_logger()


class CoherenceTopologyEngine:
    """
    Implements Coherence as a compression penalty/bonus in the world model.

    Four types of incoherence are measured:

    1. LOGICAL CONTRADICTION: conflicting causal claims in the world model.
       Cost: extra bits to specify which branch applies.

    2. TEMPORAL INCOHERENCE: past commitments violated by current behavior.
       Cost: self-model becomes unpredictive.

    3. VALUE INCOHERENCE: drive evaluations that conflict with each other.
       Cost: policy selection becomes non-deterministic in edge cases.

    4. CROSS-DOMAIN INCOHERENCE: same entity type treated differently across
       domains for no structural reason.
       Cost: schemas cannot unify - causal hierarchy stays shallow.

    coherence_compression_bonus = 1.0 + (total_extra_bits / world_model_complexity)

    A bonus of 1.0 means perfect coherence (no wasted bits).
    A bonus of 1.3 means resolving all incoherences would improve I by 30%.
    """

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.coherence")

    async def compute_incoherence_cost(
        self,
        logos: LogosMetrics,
        fovea: FoveaMetrics,
        recent_alignments: list[DriveAlignmentVector] | None = None,
    ) -> IncoherenceCostReport:
        """
        Compute the total incoherence cost in description length bits.

        Steps:
        1. Get compression stats to know the world model's current complexity
        2. Detect logical contradictions via prediction error distribution
        3. Detect temporal incoherence via self-prediction error rate
        4. Detect value incoherence via recent drive alignment conflicts
        5. Detect cross-domain mismatches via coverage distribution variance
        6. Aggregate into coherence_compression_bonus
        """
        compression_stats = await logos.get_compression_stats()
        error_distribution = await fovea.get_error_distribution()
        prediction_error_rate = await fovea.get_prediction_error_rate()

        # 1. Logical contradictions: domains where prediction errors are
        #    bimodal (sometimes right, sometimes wrong on the same type of
        #    input) suggest the model holds contradictory beliefs.
        logical = self._detect_logical_contradictions(error_distribution)

        # 2. Temporal incoherence: high self-prediction error means the
        #    model cannot predict its own behavior - indicating commitment
        #    violations or identity drift.
        temporal = self._detect_temporal_incoherence(
            prediction_error_rate, error_distribution
        )

        # 3. Value incoherence: recent drive alignment vectors that conflict
        #    (e.g., high care but low honesty on the same intent).
        value = self._detect_value_incoherence(recent_alignments or [])

        # 4. Cross-domain mismatches: domains with wildly different coverage
        #    levels for structurally similar entity types.
        domain_coverage = await logos.get_domain_coverage_map()
        cross_domain = self._detect_cross_domain_mismatches(domain_coverage)

        # Aggregate extra bits
        all_entries = logical + temporal + value + cross_domain
        total_extra_bits = sum(e.extra_description_bits for e in all_entries)

        # coherence_compression_bonus: how much I would improve if all
        # incoherences were resolved. 1.0 = nothing to gain.
        world_complexity = max(compression_stats.total_description_length, 1.0)
        coherence_bonus = 1.0 + (total_extra_bits / world_complexity)

        report = IncoherenceCostReport(
            logical_contradictions=logical,
            temporal_violations=temporal,
            value_conflicts=value,
            cross_domain_mismatches=cross_domain,
            total_extra_bits=total_extra_bits,
            coherence_compression_bonus=coherence_bonus,
            effective_I_improvement=coherence_bonus - 1.0,
        )

        self._logger.debug(
            "incoherence_cost_computed",
            bonus=f"{coherence_bonus:.3f}",
            extra_bits=f"{total_extra_bits:.2f}",
            logical=len(logical),
            temporal=len(temporal),
            value=len(value),
            cross_domain=len(cross_domain),
        )

        return report

    def _detect_logical_contradictions(
        self, error_distribution: dict[str, float]
    ) -> list[IncoherenceEntry]:
        """
        Detect domains where prediction errors suggest contradictory beliefs.

        A domain with error rate between 0.4 and 0.6 is suspicious - the model
        is right about as often as it's wrong, suggesting it holds conflicting
        hypotheses and is non-deterministically choosing between them.
        """
        entries: list[IncoherenceEntry] = []
        for domain, error_rate in error_distribution.items():
            if 0.35 <= error_rate <= 0.65:
                # Near-random performance suggests internal contradiction
                severity = 1.0 - abs(error_rate - 0.5) * 4  # Peak at 0.5
                extra_bits = severity * _CONTRADICTION_BIT_COST
                entries.append(
                    IncoherenceEntry(
                        incoherence_type=IncoherenceType.LOGICAL_CONTRADICTION,
                        description=(
                            f"Domain '{domain}' has near-random prediction accuracy "
                            f"({error_rate:.2f}), suggesting contradictory beliefs"
                        ),
                        extra_description_bits=extra_bits,
                        domain=domain,
                    )
                )
        return entries

    def _detect_temporal_incoherence(
        self,
        overall_error_rate: float,
        error_distribution: dict[str, float],
    ) -> list[IncoherenceEntry]:
        """
        Detect temporal incoherence from self-prediction failures.

        If the "self" or "identity" domain has high error rates, the model
        cannot predict its own behavior - a sign of commitment violations.
        """
        entries: list[IncoherenceEntry] = []
        self_domains = {"self", "identity", "behavior", "commitments"}

        for domain, error_rate in error_distribution.items():
            if domain.lower() in self_domains and error_rate > 0.3:
                extra_bits = error_rate * _TEMPORAL_BIT_COST
                entries.append(
                    IncoherenceEntry(
                        incoherence_type=IncoherenceType.TEMPORAL_INCOHERENCE,
                        description=(
                            f"Self-prediction domain '{domain}' has high error rate "
                            f"({error_rate:.2f}), indicating commitment violations or "
                            f"identity drift"
                        ),
                        extra_description_bits=extra_bits,
                        domain=domain,
                    )
                )

        # Also flag if overall error rate is very high - global temporal instability
        if overall_error_rate > 0.5:
            entries.append(
                IncoherenceEntry(
                    incoherence_type=IncoherenceType.TEMPORAL_INCOHERENCE,
                    description=(
                        f"Overall prediction error rate is {overall_error_rate:.2f}, "
                        f"suggesting global temporal incoherence"
                    ),
                    extra_description_bits=overall_error_rate * _TEMPORAL_BIT_COST * 0.5,
                    domain="global",
                )
            )

        return entries

    def _detect_value_incoherence(
        self, recent_alignments: list[DriveAlignmentVector]
    ) -> list[IncoherenceEntry]:
        """
        Detect value incoherence from conflicting drive alignment patterns.

        If recent decisions show wildly varying alignment patterns for similar
        types of actions, the drive evaluation is non-deterministic - which
        costs extra bits in the model.
        """
        if len(recent_alignments) < 3:
            return []

        entries: list[IncoherenceEntry] = []

        # Compute variance of each drive across recent decisions
        drives: dict[str, list[float]] = {
            "coherence": [],
            "care": [],
            "growth": [],
            "honesty": [],
        }
        for alignment in recent_alignments:
            drives["coherence"].append(alignment.coherence)
            drives["care"].append(alignment.care)
            drives["growth"].append(alignment.growth)
            drives["honesty"].append(alignment.honesty)

        for drive_name, scores in drives.items():
            variance = _variance(scores)
            if variance > _VALUE_CONFLICT_VARIANCE_THRESHOLD:
                extra_bits = variance * _VALUE_CONFLICT_BIT_COST
                entries.append(
                    IncoherenceEntry(
                        incoherence_type=IncoherenceType.VALUE_INCOHERENCE,
                        description=(
                            f"Drive '{drive_name}' shows high variance ({variance:.3f}) "
                            f"across recent {len(scores)} decisions, indicating "
                            f"non-deterministic value evaluation"
                        ),
                        extra_description_bits=extra_bits,
                        domain=f"drive:{drive_name}",
                    )
                )

        return entries

    def _detect_cross_domain_mismatches(
        self, domain_coverage: dict[str, float]
    ) -> list[IncoherenceEntry]:
        """
        Detect cross-domain mismatches in coverage levels.

        Structurally similar domains (e.g., "social_dynamics" and "social_trust")
        should have similar coverage. Large discrepancies suggest the same entity
        type is treated inconsistently across domains.
        """
        entries: list[IncoherenceEntry] = []
        domains = sorted(domain_coverage.items())

        for i, (domain_a, cov_a) in enumerate(domains):
            for domain_b, cov_b in domains[i + 1 :]:
                # Check if domains are structurally related (share prefix)
                if not _domains_related(domain_a, domain_b):
                    continue

                gap = abs(cov_a - cov_b)
                if gap > _CROSS_DOMAIN_GAP_THRESHOLD:
                    extra_bits = gap * _CROSS_DOMAIN_BIT_COST
                    entries.append(
                        IncoherenceEntry(
                            incoherence_type=IncoherenceType.CROSS_DOMAIN_MISMATCH,
                            description=(
                                f"Related domains '{domain_a}' (coverage={cov_a:.2f}) "
                                f"and '{domain_b}' (coverage={cov_b:.2f}) have "
                                f"inconsistent coverage (gap={gap:.2f})"
                            ),
                            extra_description_bits=extra_bits,
                            domain=f"{domain_a}|{domain_b}",
                        )
                    )

        return entries


# ─── Helpers ─────────────────────────────────────────────────────────


def _variance(values: list[float]) -> float:
    """Compute sample variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _domains_related(a: str, b: str) -> bool:
    """Check if two domain names are structurally related."""
    # Share a common prefix of at least 3 characters
    prefix_len = 0
    for ca, cb in zip(a.lower(), b.lower(), strict=False):
        if ca == cb:
            prefix_len += 1
        else:
            break
    return prefix_len >= 3


# ─── Constants ───────────────────────────────────────────────────────

# Bit costs per incoherence type - calibrated to produce meaningful
# coherence_compression_bonus values.

_CONTRADICTION_BIT_COST = 10.0  # A logical contradiction is expensive
_TEMPORAL_BIT_COST = 15.0  # Self-model instability is very expensive
_VALUE_CONFLICT_BIT_COST = 8.0  # Drive non-determinism is moderately expensive
_CROSS_DOMAIN_BIT_COST = 5.0  # Coverage mismatch is less expensive
_VALUE_CONFLICT_VARIANCE_THRESHOLD = 0.15  # Flag if variance > 0.15
_CROSS_DOMAIN_GAP_THRESHOLD = 0.3  # Flag if coverage gap > 0.3
