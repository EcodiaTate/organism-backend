"""
EcodiaOS -- Kairos: Causal Invariant Mining

Correlations are the most expensive and least compressed form of knowledge.
Causal invariants are the most compressed: one rule generates infinite
observations across every domain it touches.

When Kairos finds a Tier 3 invariant, the intelligence ratio makes a step change.
"""

from systems.kairos.counter_invariant import CounterInvariantDetector
from systems.kairos.hierarchy import CausalHierarchy
from systems.kairos.intelligence_ledger import IntelligenceContributionLedger
from systems.kairos.invariant_distiller import InvariantDistiller
from systems.kairos.pipeline import KairosEvoPipeline, KairosPipeline
from systems.kairos.types import (
    CausalInvariant,
    CausalNoveltyPayload,
    CausalStructurePattern,
    HealthDegradedPayload,
    InvariantAbsorptionPayload,
    KairosConfig,
    KairosHealthStatus,
    SpuriousHypothesisClassPayload,
    ValidatedCausalStructurePayload,
    ViolationEscalationPayload,
)

__all__ = [
    "KairosPipeline",
    "KairosEvoPipeline",
    "KairosConfig",
    "CausalHierarchy",
    "CausalInvariant",
    "InvariantDistiller",
    "CounterInvariantDetector",
    "IntelligenceContributionLedger",
    # Feedback loop payloads
    "ValidatedCausalStructurePayload",
    "SpuriousHypothesisClassPayload",
    "InvariantAbsorptionPayload",
    "CausalNoveltyPayload",
    "HealthDegradedPayload",
    "ViolationEscalationPayload",
    # Health + patterns
    "KairosHealthStatus",
    "CausalStructurePattern",
]
