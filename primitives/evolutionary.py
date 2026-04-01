"""
EcodiaOS - Evolutionary Metrics Primitives

Types for population-level evolutionary statistics that Benchmarks
computes across the organism fleet. Includes Bedau-Packard activity
statistics for measuring whether genuine open-ended evolution is occurring.

Reference: Bedau & Packard (1992) - "Measurement of Evolutionary Activity,
Teleology, and Life"
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from primitives.common import EOSBaseModel, SystemID, utc_now


class EvolutionaryActivity(EOSBaseModel):
    """
    Per-instance activity record for evolutionary population tracking.

    Each living organism instance contributes one of these to the
    population registry. Benchmarks aggregates them into BedauPackardStats.
    """

    instance_id: str = ""
    generation: int = 1
    novel_mutations: int = 0
    total_mutations: int = 0
    fitness: float = 0.0
    metabolic_efficiency: Decimal = Decimal("0")
    alive: bool = True
    born_at: datetime = Field(default_factory=utc_now)
    died_at: datetime | None = None
    parent_instance_id: str | None = None


class BedauPackardStats(EOSBaseModel):
    """
    Population-level Bedau-Packard evolutionary activity statistics.

    These metrics distinguish genuine open-ended evolution (unbounded
    novelty production) from mere parameter drift. A living population
    shows increasing total_activity and persistence > 0.5.
    """

    total_activity: float = 0.0
    mean_activity: float = 0.0
    diversity_index: float = 0.0
    evolutionary_rate: float = 0.0
    persistence: float = 0.0
    timestamp: datetime = Field(default_factory=utc_now)


class EvolutionaryObservable(EOSBaseModel):
    """
    Any system can emit an evolutionary observable - a discrete event
    that may represent a novel adaptation.

    Benchmarks collects these to compute population-level activity
    statistics. The is_novel flag distinguishes inherited behaviours
    from genuinely new ones.
    """

    source_system: SystemID
    instance_id: str = ""
    observable_type: str = ""
    value: float = 0.0
    is_novel: bool = False
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)
