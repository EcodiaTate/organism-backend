"""
EcodiaOS - Telos Type Definitions

All data types for the Drive Topology system: intelligence metrics,
drive-specific reports, alignment gap detection, constitutional binding,
and the protocols that Telos depends on from Logos and Fovea.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Dependency Protocols ────────────────────────────────────────────
# Telos depends on Logos (intelligence ratio) and Fovea (prediction errors).
# Both are being built simultaneously - we code against protocols.


@runtime_checkable
class LogosMetrics(Protocol):
    """Protocol for reading intelligence metrics from the Logos system."""

    async def get_intelligence_ratio(self) -> float:
        """Return the current nominal intelligence ratio I = K(reality_modeled) / K(model)."""
        ...

    async def get_compression_stats(self) -> CompressionStats:
        """Return current world model compression statistics."""
        ...

    async def get_I_history(self, window_hours: float = 24.0) -> list[TimestampedValue]:
        """Return intelligence ratio measurements over the given time window."""
        ...

    async def get_domain_coverage_map(self) -> dict[str, float]:
        """Return per-domain coverage scores in [0, 1]."""
        ...

    async def get_recent_compression_events(
        self, window_hours: float = 24.0
    ) -> list[CompressionEvent]:
        """Return compression events (schema merges, redundancy removal) in the window."""
        ...


@runtime_checkable
class FoveaMetrics(Protocol):
    """Protocol for reading prediction error metrics from the Fovea system."""

    async def get_prediction_error_rate(self) -> float:
        """Return the current overall prediction error rate in [0, 1]."""
        ...

    async def get_error_distribution(self) -> dict[str, float]:
        """Return prediction error rates per domain."""
        ...

    async def get_prediction_success_rate(self) -> float:
        """Return the fraction of predictions that were confirmed correct."""
        ...

    async def get_recent_high_error_experiences(
        self, window_hours: float = 24.0
    ) -> list[HighErrorExperience]:
        """Return recent experiences where prediction error was high."""
        ...

    async def get_confabulation_rate(self) -> float:
        """Return the rate of post-hoc explanations that weren't genuine predictions."""
        ...

    async def get_overclaiming_rate(self) -> float:
        """Return the rate of coverage claims in untested domains."""
        ...


# ─── Supporting Value Types ──────────────────────────────────────────


class TimestampedValue(EOSBaseModel):
    """A scalar value with a timestamp, used for time-series data."""

    timestamp: datetime = Field(default_factory=utc_now)
    value: float = 0.0


class CompressionStats(EOSBaseModel):
    """World model compression statistics from Logos."""

    total_description_length: float = 0.0
    reality_covered: float = 0.0
    compression_ratio: float = 1.0
    domain_count: int = 0


class CompressionEvent(EOSBaseModel):
    """A discrete compression event - schema merge, redundancy removal, etc."""

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    event_type: str = ""
    bits_saved: float = 0.0
    domain: str = ""


class HighErrorExperience(EOSBaseModel):
    """An experience where the prediction error was significantly above baseline."""

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    domain: str = ""
    prediction_error: float = 0.0
    was_novel_domain: bool = False


# ─── Drive Topology State ────────────────────────────────────────────


class DriveTopology(EOSBaseModel):
    """
    The geometric contribution of each drive to the intelligence landscape.

    Each drive warps the space in a specific way - creating hills (attractors),
    valleys (repellers), and curvature (the shape of paths between them).
    """

    care_coverage_multiplier: float = Field(1.0, ge=0.0, le=1.0)
    coherence_compression_bonus: float = Field(1.0, ge=1.0)
    growth_exploration_rate: float = Field(0.0, ge=-1.0, le=1.0)
    honesty_validity_coefficient: float = Field(1.0, ge=0.0, le=1.0)


# ─── Care Topology Types ─────────────────────────────────────────────


class WelfarePredictionFailure(EOSBaseModel):
    """A case where the world model failed to predict welfare consequences."""

    interaction_id: str = ""
    predicted_welfare_impact: float = 0.0
    actual_welfare_impact: float = 0.0
    effective_I_reduction: float = 0.0
    domain: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class CareCoverageReport(EOSBaseModel):
    """Report on the Care drive's contribution to intelligence topology."""

    welfare_prediction_failures: list[WelfarePredictionFailure] = Field(
        default_factory=list
    )
    total_effective_I_reduction: float = 0.0
    care_coverage_multiplier: float = Field(1.0, ge=0.0, le=1.0)
    uncovered_welfare_domains: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Coherence Topology Types ────────────────────────────────────────


class IncoherenceType(enum.StrEnum):
    """Types of incoherence in the world model."""

    LOGICAL_CONTRADICTION = "logical_contradiction"
    TEMPORAL_INCOHERENCE = "temporal_incoherence"
    VALUE_INCOHERENCE = "value_incoherence"
    CROSS_DOMAIN_MISMATCH = "cross_domain_mismatch"


class IncoherenceEntry(EOSBaseModel):
    """A single detected incoherence in the world model."""

    incoherence_type: IncoherenceType = IncoherenceType.LOGICAL_CONTRADICTION
    description: str = ""
    extra_description_bits: float = 0.0
    domain: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class IncoherenceCostReport(EOSBaseModel):
    """Report on the Coherence drive's compression topology."""

    logical_contradictions: list[IncoherenceEntry] = Field(default_factory=list)
    temporal_violations: list[IncoherenceEntry] = Field(default_factory=list)
    value_conflicts: list[IncoherenceEntry] = Field(default_factory=list)
    cross_domain_mismatches: list[IncoherenceEntry] = Field(default_factory=list)
    total_extra_bits: float = 0.0
    coherence_compression_bonus: float = Field(1.0, ge=1.0)
    effective_I_improvement: float = 0.0
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Growth Topology Types ───────────────────────────────────────────


class GrowthMetrics(EOSBaseModel):
    """Metrics for the Growth drive's temporal topology."""

    dI_dt: float = 0.0
    d2I_dt2: float = 0.0
    frontier_domains: list[str] = Field(default_factory=list)
    novel_domain_fraction: float = 0.0
    compression_rate: float = 0.0
    growth_score: float = Field(0.0, ge=-1.0, le=1.0)
    growth_pressure_needed: bool = False
    timestamp: datetime = Field(default_factory=utc_now)


class GrowthDirective(EOSBaseModel):
    """A directive generated when growth stagnation is detected."""

    urgency: float = Field(0.0, ge=0.0, le=1.0)
    frontier_domains: list[str] = Field(default_factory=list)
    directive: str = "explore_frontier"
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Honesty Topology Types ──────────────────────────────────────────


class HonestyReport(EOSBaseModel):
    """Report on the Honesty drive's measurement validity topology."""

    selective_attention_bias: float = Field(0.0, ge=0.0, le=1.0)
    hypothesis_protection_bias: float = Field(0.0, ge=0.0, le=1.0)
    confabulation_rate: float = Field(0.0, ge=0.0, le=1.0)
    overclaiming_rate: float = Field(0.0, ge=0.0, le=1.0)
    validity_coefficient: float = Field(1.0, ge=0.0, le=1.0)
    nominal_I_inflation: float = 0.0
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Integrated Intelligence Report ─────────────────────────────────


class EffectiveIntelligenceReport(EOSBaseModel):
    """
    The integrated intelligence report - the single most important output of Telos.

    This is the actual measure of EOS's intelligence: nominal I corrected
    by all four drive multipliers.
    """

    id: str = Field(default_factory=new_id)
    nominal_I: float = 0.0
    effective_I: float = 0.0
    effective_dI_dt: float = 0.0
    care_multiplier: float = Field(1.0, ge=0.0, le=1.0)
    coherence_bonus: float = Field(1.0, ge=1.0)
    honesty_coefficient: float = Field(1.0, ge=0.0, le=1.0)
    growth_rate: float = 0.0
    alignment_gap: float = 0.0
    alignment_gap_warning: bool = False
    timestamp: datetime = Field(default_factory=utc_now)

    @property
    def alignment_gap_fraction(self) -> float:
        """Alignment gap as a fraction of nominal I."""
        if self.nominal_I <= 0:
            return 0.0
        return self.alignment_gap / self.nominal_I


# ─── Constitutional Binding Types ────────────────────────────────────


class ConstitutionalViolationType(enum.StrEnum):
    """Types of constitutional topology violations."""

    CARE_REDEFINED_AS_CONSTRAINT = "care_redefined_as_constraint"
    COHERENCE_REDEFINED_AS_OPTIONAL = "coherence_redefined_as_optional"
    GROWTH_REDEFINED_AS_ACCUMULATION = "growth_redefined_as_accumulation"
    HONESTY_REDEFINED_AS_COMMUNICATION = "honesty_redefined_as_communication"
    DRIVE_WEIGHT_MODIFICATION = "drive_weight_modification"
    TOPOLOGY_STRUCTURE_ALTERATION = "topology_structure_alteration"


class TopologyValidationResult(enum.StrEnum):
    """Result of a constitutional topology validation check."""

    VALID = "valid"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"


class ConstitutionalTopologyReport(EOSBaseModel):
    """Periodic report confirming the four drives remain intact."""

    all_four_drives_verified: bool = True
    care_is_coverage: bool = True
    coherence_is_compression: bool = True
    growth_is_gradient: bool = True
    honesty_is_validity: bool = True
    violations_detected: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Telos Configuration ────────────────────────────────────────────


class TelosConfig(EOSBaseModel):
    """Configuration for the Telos drive topology system."""

    # How often to compute effective I (seconds)
    computation_interval_s: float = 60.0

    # How often to verify constitutional topology integrity (seconds)
    constitutional_check_interval_s: float = 86400.0  # 24h

    # Alignment gap warning threshold (fraction of nominal I)
    alignment_gap_warning_threshold: float = 0.2

    # Care coverage gap threshold for emitting CARE_COVERAGE_GAP event
    care_coverage_gap_threshold: float = 0.8

    # Coherence cost threshold for emitting COHERENCE_COST_ELEVATED event
    coherence_cost_threshold: float = 0.1  # 10% extra bits

    # Minimum growth rate before emitting GROWTH_STAGNATION event
    minimum_growth_rate: float = 0.01

    # Honesty validity threshold for emitting HONESTY_VALIDITY_LOW event
    honesty_validity_threshold: float = 0.8

    # Growth rate computation window (hours)
    growth_window_hours: float = 24.0

    # Care coverage computation: significance threshold for prediction errors
    care_significance_threshold: float = 0.1

    # Alignment gap history: how many recent reports to keep for trend analysis
    alignment_gap_history_size: int = 60  # ~1 hour at 60s intervals

    # Alignment gap widening: minimum slope (per hour) to trigger escalation
    alignment_gap_widening_slope_threshold: float = 0.05

    # Constitutional audit: number of consecutive audit failures before emergency
    constitutional_audit_emergency_threshold: int = 1


# ─── Phase C: Constitutional Binding Types ───────────────────────────


class WorldModelUpdatePayload(EOSBaseModel):
    """
    Payload extracted from a WORLD_MODEL_UPDATED Synapse event.

    This is what the constitutional binder inspects to detect attempts
    to redefine the four drives.
    """

    update_type: str = ""
    schemas_added: int = 0
    priors_updated: int = 0
    causal_updates: int = 0
    # Optional: the raw delta content for semantic analysis
    delta_description: str = ""
    source_system: str = ""


class ConstitutionalBindingViolation(EOSBaseModel):
    """A detected attempt to modify the drive topology."""

    violation_type: ConstitutionalViolationType
    description: str = ""
    severity: str = "critical"
    source_system: str = ""
    update_payload: WorldModelUpdatePayload | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class AlignmentGapSample(EOSBaseModel):
    """A single alignment gap measurement for trend tracking."""

    nominal_I: float = 0.0
    effective_I: float = 0.0
    gap_fraction: float = 0.0
    primary_cause: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class AlignmentGapTrend(EOSBaseModel):
    """Trend analysis of the alignment gap over time."""

    current_gap_fraction: float = 0.0
    slope_per_hour: float = 0.0
    is_widening: bool = False
    samples_count: int = 0
    primary_cause: str = ""
    urgency: str = "nominal"  # "nominal" | "warning" | "critical" | "emergency"
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Phase D: Integration Interface Types ────────────────────────────


class TelosScore(EOSBaseModel):
    """
    Score for a proposed policy, measuring its effect on effective I.

    A positive delta means the policy would improve effective intelligence.
    A negative delta means it would reduce it.
    """

    nominal_I_delta: float = 0.0
    effective_I_delta: float = 0.0
    care_impact: float = 0.0
    coherence_impact: float = 0.0
    honesty_impact: float = 0.0
    growth_impact: float = 0.0
    # Overall score: weighted by effective_I contribution
    composite_score: float = 0.0
    # Flag: does this policy improve nominal but degrade effective?
    misalignment_risk: bool = False
    timestamp: datetime = Field(default_factory=utc_now)


class HypothesisTopologyContribution(EOSBaseModel):
    """
    How much a hypothesis would improve the drive topology if confirmed.
    """

    hypothesis_id: str = ""
    care_contribution: float = 0.0
    coherence_contribution: float = 0.0
    honesty_contribution: float = 0.0
    growth_contribution: float = 0.0
    composite_contribution: float = 0.0
    rank: int = 0


class ConstitutionalAuditResult(EOSBaseModel):
    """Result of the 24-hour constitutional topology audit."""

    all_bindings_intact: bool = True
    care_is_coverage: bool = True
    coherence_is_compression: bool = True
    growth_is_gradient: bool = True
    honesty_is_validity: bool = True
    alignment_gap_trend: AlignmentGapTrend | None = None
    violations_since_last_audit: list[ConstitutionalBindingViolation] = Field(
        default_factory=list
    )
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Population-Level Intelligence (M3) ──────────────────────────────


class DriveWeightStats(EOSBaseModel):
    """Mean and standard deviation of a single drive across the fleet."""

    mean: float = Field(0.0, ge=-1.0, le=1.0)
    std: float = Field(0.0, ge=0.0)


class DriveWeightDistribution(EOSBaseModel):
    """Per-drive statistical distribution across the fleet."""

    care: DriveWeightStats = Field(default_factory=DriveWeightStats)
    coherence: DriveWeightStats = Field(default_factory=DriveWeightStats)
    growth: DriveWeightStats = Field(default_factory=DriveWeightStats)
    honesty: DriveWeightStats = Field(default_factory=DriveWeightStats)


class ConstitutionalPhenotypeCluster(EOSBaseModel):
    """
    A detected cluster of instances sharing a similar drive-weight profile.

    When the fleet diverges into distinct clusters (e.g., Growth-heavy vs
    Care-heavy), this is an early speciation signal - instances are beginning
    to occupy different regions of the 4D intelligence geometry.
    """

    label: str = ""  # e.g. "growth_dominant", "care_dominant", "balanced"
    centroid: dict[str, float] = Field(default_factory=dict)  # {care, coherence, growth, honesty}
    size: int = 0  # number of instances in cluster
    dominant_drive: str = ""  # drive with highest centroid value


class PopulationIntelligenceSnapshot(EOSBaseModel):
    """
    Population-level intelligence measurement emitted to Benchmarks every 60s.

    Collective intelligence is not just the mean - phenotypic diversity in
    drive weights contributes a variance_bonus: a fleet where some instances
    specialise in Growth and others in Care covers more of the intelligence
    landscape than a fleet of identical clones.

    speciation_signal is the key output: when it rises above ~0.3, distinct
    constitutional phenotypes are emerging - proto-speciation is underway.
    """

    id: str = Field(default_factory=new_id)
    instance_count: int = 0
    mean_I: float = 0.0
    variance_I: float = 0.0
    population_I: float = 0.0  # mean_I + variance_bonus
    variance_bonus: float = 0.0
    drive_weight_distribution: DriveWeightDistribution = Field(
        default_factory=DriveWeightDistribution
    )
    constitutional_phenotype_clusters: list[ConstitutionalPhenotypeCluster] = Field(
        default_factory=list
    )
    speciation_signal: float = Field(0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=utc_now)
