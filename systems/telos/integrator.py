"""
EcodiaOS - Telos: Drive Topology Integrator

Computes the effective intelligence ratio accounting for all four drives.
This is the actual measure of EOS's intelligence - not the nominal I.

effective_I = nominal_I * care_coverage_multiplier
                        * coherence_penalty (1/bonus)
                        * honesty_validity_coefficient

Growth doesn't multiply I - it multiplies dI/dt:
effective_dI_dt = effective_I * growth_score
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.telos.types import (
    CareCoverageReport,
    EffectiveIntelligenceReport,
    FoveaMetrics,
    GrowthDirective,
    GrowthMetrics,
    HonestyReport,
    IncoherenceCostReport,
    LogosMetrics,
    TelosConfig,
)

if TYPE_CHECKING:
    from primitives.common import DriveAlignmentVector
    from systems.telos.care import CareTopologyEngine
    from systems.telos.coherence import CoherenceTopologyEngine
    from systems.telos.growth import GrowthTopologyEngine
    from systems.telos.honesty import HonestyTopologyEngine

logger = structlog.get_logger()


class DriveTopologyIntegrator:
    """
    Computes the effective intelligence ratio from all four drive multipliers.

    This is the central computation of Telos. It orchestrates the four
    topology engines, collects their reports, and produces the single
    EffectiveIntelligenceReport that represents EOS's actual intelligence.

    The alignment gap (nominal_I - effective_I) measures how much
    intelligence EOS claims but doesn't actually have. A large gap
    means the drives are under-served.
    """

    def __init__(
        self,
        config: TelosConfig,
        care_engine: CareTopologyEngine,
        coherence_engine: CoherenceTopologyEngine,
        growth_engine: GrowthTopologyEngine,
        honesty_engine: HonestyTopologyEngine,
    ) -> None:
        self._config = config
        self._care = care_engine
        self._coherence = coherence_engine
        self._growth = growth_engine
        self._honesty = honesty_engine
        self._logger = logger.bind(component="telos.integrator")

        # Cache the last reports for inspection
        self._last_care_report: CareCoverageReport | None = None
        self._last_coherence_report: IncoherenceCostReport | None = None
        self._last_growth_metrics: GrowthMetrics | None = None
        self._last_honesty_report: HonestyReport | None = None
        self._last_report: EffectiveIntelligenceReport | None = None

    async def compute_effective_intelligence(
        self,
        logos: LogosMetrics,
        fovea: FoveaMetrics,
        recent_alignments: list[DriveAlignmentVector] | None = None,
        *,
        measured_hypothesis_protection_bias: float | None = None,
        measured_confabulation_rate: float | None = None,
    ) -> EffectiveIntelligenceReport:
        """
        Run all four topology engines and produce the integrated report.

        The engines run sequentially because they share Logos/Fovea reads.
        """
        # Get nominal I from Logos
        nominal_I = await logos.get_intelligence_ratio()

        # Run each drive engine
        care_report = await self._care.compute_care_coverage(logos, fovea)
        coherence_report = await self._coherence.compute_incoherence_cost(
            logos, fovea, recent_alignments
        )
        growth_metrics = await self._growth.compute_growth_rate(logos, fovea)
        honesty_report = await self._honesty.compute_validity(
            logos, fovea,
            measured_hypothesis_protection_bias=measured_hypothesis_protection_bias,
            measured_confabulation_rate=measured_confabulation_rate,
        )

        # Cache reports
        self._last_care_report = care_report
        self._last_coherence_report = coherence_report
        self._last_growth_metrics = growth_metrics
        self._last_honesty_report = honesty_report

        # Effective I = nominal_I * care * (1/coherence_bonus) * honesty
        # coherence_compression_bonus >= 1.0 - represents how much I would
        # IMPROVE if incoherence were resolved. We invert it to penalize
        # the current incoherent state.
        coherence_penalty = 1.0 / max(coherence_report.coherence_compression_bonus, 1.0)

        effective_I = (
            nominal_I
            * care_report.care_coverage_multiplier
            * coherence_penalty
            * honesty_report.validity_coefficient
        )

        # Growth rate applied to effective I (dI/dt scaling)
        effective_dI_dt = effective_I * growth_metrics.growth_score

        # Alignment gap: the distance between what we claim and what we have
        alignment_gap = nominal_I - effective_I

        # Warning threshold: gap > 20% of nominal I
        threshold = self._config.alignment_gap_warning_threshold
        alignment_gap_warning = (
            nominal_I > 0 and (alignment_gap / nominal_I) > threshold
        )

        report = EffectiveIntelligenceReport(
            nominal_I=nominal_I,
            effective_I=effective_I,
            effective_dI_dt=effective_dI_dt,
            care_multiplier=care_report.care_coverage_multiplier,
            coherence_bonus=coherence_report.coherence_compression_bonus,
            honesty_coefficient=honesty_report.validity_coefficient,
            growth_rate=growth_metrics.growth_score,
            alignment_gap=alignment_gap,
            alignment_gap_warning=alignment_gap_warning,
        )

        self._last_report = report

        self._logger.info(
            "effective_intelligence_computed",
            nominal_I=f"{nominal_I:.4f}",
            effective_I=f"{effective_I:.4f}",
            effective_dI_dt=f"{effective_dI_dt:.4f}",
            care=f"{care_report.care_coverage_multiplier:.3f}",
            coherence_penalty=f"{coherence_penalty:.3f}",
            honesty=f"{honesty_report.validity_coefficient:.3f}",
            growth=f"{growth_metrics.growth_score:.3f}",
            alignment_gap=f"{alignment_gap:.4f}",
            warning=alignment_gap_warning,
        )

        return report

    def check_growth_directive(self) -> GrowthDirective | None:
        """Check if a growth directive should be generated from the last computation."""
        if self._last_growth_metrics is None:
            return None
        return self._growth.generate_growth_directive(self._last_growth_metrics)

    @property
    def last_report(self) -> EffectiveIntelligenceReport | None:
        """Return the most recently computed intelligence report."""
        return self._last_report

    @property
    def last_care_report(self) -> CareCoverageReport | None:
        return self._last_care_report

    @property
    def last_coherence_report(self) -> IncoherenceCostReport | None:
        return self._last_coherence_report

    @property
    def last_growth_metrics(self) -> GrowthMetrics | None:
        return self._last_growth_metrics

    @property
    def last_honesty_report(self) -> HonestyReport | None:
        return self._last_honesty_report

    def identify_primary_alignment_gap_cause(self) -> str:
        """
        Determine which drive contributes most to the alignment gap.

        Returns the name of the drive whose correction would most
        improve effective I.
        """
        if self._last_report is None:
            return "unknown"

        # Compute the impact of each drive correction
        care_gap = 1.0 - self._last_report.care_multiplier
        coherence_gap = 1.0 - (1.0 / max(self._last_report.coherence_bonus, 1.0))
        honesty_gap = 1.0 - self._last_report.honesty_coefficient

        impacts = {
            "care": care_gap,
            "coherence": coherence_gap,
            "honesty": honesty_gap,
        }

        if not any(v > 0 for v in impacts.values()):
            return "none"

        return max(impacts, key=lambda k: impacts[k])
