"""
EcodiaOS - Telos: Care Topology Engine

Care is not "be nice." Care is a structural commitment to modeling the
welfare of others as part of reality. A world model that excludes welfare
consequences has a coverage gap - it can't predict relationship dynamics,
trust cascades, or harm propagation. That gap costs effective I.

The Care engine measures how much of reality the world model fails to
explain because it is not modeling welfare consequences, and computes
the care_coverage_multiplier that corrects effective I accordingly.
"""

from __future__ import annotations

import structlog

from systems.telos.types import (
    CareCoverageReport,
    FoveaMetrics,
    LogosMetrics,
    TelosConfig,
    WelfarePredictionFailure,
)

logger = structlog.get_logger()


class CareTopologyEngine:
    """
    Implements Care as a coverage expansion in the world model.

    Iterates over recent interaction history (via Fovea's high-error
    experiences), identifies cases where welfare consequences were
    mispredicted, computes the effective I reduction from each failure,
    and aggregates into a care_coverage_multiplier.

    care_coverage_multiplier = 1.0 - (total_I_reduction / nominal_I)

    A multiplier of 1.0 means perfect welfare coverage.
    A multiplier of 0.5 means half the claimed coverage is welfare-blind.
    """

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.care")

    async def compute_care_coverage(
        self,
        logos: LogosMetrics,
        fovea: FoveaMetrics,
    ) -> CareCoverageReport:
        """
        Compute the Care coverage gap by analyzing welfare prediction failures.

        Steps:
        1. Get nominal I from Logos
        2. Get domain coverage map to identify welfare-relevant domains
        3. Get recent high-error experiences from Fovea
        4. Filter for welfare-relevant prediction failures
        5. Compute per-failure I reduction
        6. Aggregate into care_coverage_multiplier
        """
        nominal_I = await logos.get_intelligence_ratio()
        domain_coverage = await logos.get_domain_coverage_map()
        recent_errors = await fovea.get_recent_high_error_experiences(
            window_hours=self._config.growth_window_hours
        )

        welfare_failures: list[WelfarePredictionFailure] = []

        for experience in recent_errors:
            # Check if this error is in a welfare-relevant domain
            if not self._is_welfare_relevant(experience.domain):
                continue

            # Significance filter: only count errors above threshold
            if experience.prediction_error < self._config.care_significance_threshold:
                continue

            # The I reduction from this failure is proportional to the error magnitude
            # and the domain's weight in the overall coverage.
            domain_weight = domain_coverage.get(experience.domain, 0.0)
            effective_I_reduction = self._compute_I_reduction(
                prediction_error=experience.prediction_error,
                domain_weight=domain_weight,
            )

            welfare_failures.append(
                WelfarePredictionFailure(
                    interaction_id=experience.id,
                    predicted_welfare_impact=0.0,  # Implicit: model predicted ~zero impact
                    actual_welfare_impact=experience.prediction_error,
                    effective_I_reduction=effective_I_reduction,
                    domain=experience.domain,
                    timestamp=experience.timestamp,
                )
            )

        total_I_reduction = sum(f.effective_I_reduction for f in welfare_failures)

        # care_coverage_multiplier: how much of the claimed coverage is real
        # 1.0 = perfect welfare coverage, 0.0 = all coverage is welfare-blind
        if nominal_I > 0:
            care_multiplier = max(0.0, min(1.0, 1.0 - (total_I_reduction / nominal_I)))
        else:
            care_multiplier = 1.0  # No claims to deflate

        # Identify welfare domains with low coverage
        uncovered = self._identify_uncovered_welfare_domains(domain_coverage)

        report = CareCoverageReport(
            welfare_prediction_failures=welfare_failures,
            total_effective_I_reduction=total_I_reduction,
            care_coverage_multiplier=care_multiplier,
            uncovered_welfare_domains=uncovered,
        )

        self._logger.debug(
            "care_coverage_computed",
            multiplier=f"{care_multiplier:.3f}",
            failures=len(welfare_failures),
            total_reduction=f"{total_I_reduction:.4f}",
            uncovered_domains=len(uncovered),
        )

        return report

    def _is_welfare_relevant(self, domain: str) -> bool:
        """Check if a domain involves welfare consequences."""
        if not domain:
            return False
        domain_lower = domain.lower()
        return any(kw in domain_lower for kw in _WELFARE_KEYWORDS)

    def _compute_I_reduction(
        self, prediction_error: float, domain_weight: float
    ) -> float:
        """
        Compute the effective I reduction from a single welfare prediction failure.

        The reduction is the product of:
        - prediction_error: how wrong the prediction was (0-1)
        - domain_weight: how important this domain is to overall coverage (0-1)
        - A scaling factor that accounts for the fact that welfare failures
          tend to cascade (you miss the first-order effect AND the downstream
          consequences).
        """
        cascade_factor = 1.5  # Welfare errors compound
        return prediction_error * max(domain_weight, 0.05) * cascade_factor

    def _identify_uncovered_welfare_domains(
        self, domain_coverage: dict[str, float]
    ) -> list[str]:
        """Find welfare-relevant domains with coverage below 0.4."""
        uncovered: list[str] = []
        for domain, coverage in domain_coverage.items():
            if self._is_welfare_relevant(domain) and coverage < 0.4:
                uncovered.append(domain)
        return sorted(uncovered)


# Keywords that indicate welfare relevance in a domain name
_WELFARE_KEYWORDS = (
    "welfare",
    "care",
    "harm",
    "trust",
    "social",
    "relationship",
    "safety",
    "health",
    "wellbeing",
    "emotional",
    "interpersonal",
    "cooperation",
    "conflict",
    "consent",
    "community",
)
