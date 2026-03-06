"""
EcodiaOS — Telos: Drives as Intelligence Topology

The four constitutional drives (Care, Coherence, Growth, Honesty) are not
constraints on intelligence. They are the geometry of the space intelligence
moves through. Telos computes effective I (intelligence ratio corrected by
drive multipliers), tracks alignment gaps, runs four drive-specific topology
engines, and broadcasts drive health metrics.
"""

from systems.telos.adapters import FoveaMetricsAdapter, LogosMetricsAdapter
from systems.telos.alignment import AlignmentGapMonitor
from systems.telos.audit import ConstitutionalTopologyAuditor
from systems.telos.binder import TelosConstitutionalBinder
from systems.telos.care import CareTopologyEngine
from systems.telos.coherence import CoherenceTopologyEngine
from systems.telos.growth import GrowthTopologyEngine
from systems.telos.honesty import HonestyTopologyEngine
from systems.telos.integrator import DriveTopologyIntegrator
from systems.telos.interfaces import (
    TelosFragmentSelector,
    TelosHypothesisPrioritizer,
    TelosPolicyScorer,
)
from systems.telos.service import TelosService
from systems.telos.types import (
    AlignmentGapSample,
    AlignmentGapTrend,
    CareCoverageReport,
    CompressionEvent,
    CompressionStats,
    ConstitutionalAuditResult,
    ConstitutionalBindingViolation,
    ConstitutionalTopologyReport,
    ConstitutionalViolationType,
    DriveTopology,
    EffectiveIntelligenceReport,
    FoveaMetrics,
    GrowthDirective,
    GrowthMetrics,
    HighErrorExperience,
    HonestyReport,
    HypothesisTopologyContribution,
    IncoherenceCostReport,
    IncoherenceEntry,
    IncoherenceType,
    LogosMetrics,
    TelosConfig,
    TelosScore,
    TimestampedValue,
    TopologyValidationResult,
    WelfarePredictionFailure,
    WorldModelUpdatePayload,
)

__all__ = [
    # Service
    "TelosService",
    # Engines
    "CareTopologyEngine",
    "CoherenceTopologyEngine",
    "DriveTopologyIntegrator",
    "GrowthTopologyEngine",
    "HonestyTopologyEngine",
    # Phase C: Constitutional binding + alignment gap monitoring
    "AlignmentGapMonitor",
    "ConstitutionalTopologyAuditor",
    "TelosConstitutionalBinder",
    # Phase D: Integration interfaces
    "TelosFragmentSelector",
    "TelosHypothesisPrioritizer",
    "TelosPolicyScorer",
    # Adapters (protocol bridges for real services)
    "FoveaMetricsAdapter",
    "LogosMetricsAdapter",
    # Protocols (for dependency injection)
    "FoveaMetrics",
    "LogosMetrics",
    # Config
    "TelosConfig",
    # Reports
    "CareCoverageReport",
    "ConstitutionalAuditResult",
    "ConstitutionalTopologyReport",
    "EffectiveIntelligenceReport",
    "GrowthDirective",
    "GrowthMetrics",
    "HonestyReport",
    "IncoherenceCostReport",
    # Types
    "AlignmentGapSample",
    "AlignmentGapTrend",
    "CompressionEvent",
    "CompressionStats",
    "ConstitutionalBindingViolation",
    "ConstitutionalViolationType",
    "DriveTopology",
    "HighErrorExperience",
    "HypothesisTopologyContribution",
    "IncoherenceEntry",
    "IncoherenceType",
    "TelosScore",
    "TimestampedValue",
    "TopologyValidationResult",
    "WelfarePredictionFailure",
    "WorldModelUpdatePayload",
]
