"""
EcodiaOS - Cognitive Niches: Isolated Hypothesis Ecosystems

A CognitiveNiche is a self-contained worldview module - a cluster of
hypotheses, schemas, and procedures that has diverged enough from the
general population to warrant its own fitness landscape, evidence
thresholds, and processing strategy.

This is the fundamental unit of cognitive speciation. Where pressure.py
detects species (passive observation), cognitive niches are ACTIVE
isolation barriers that prevent hypothesis mixing between incompatible
worldviews while allowing beneficial gene flow between compatible ones.

Key concepts:
  - **Reproductive isolation**: Hypotheses within a niche preferentially
    share evidence with niche-mates. Cross-niche evidence transfer is
    attenuated by a compatibility coefficient (0.0 = fully isolated,
    1.0 = no barrier).
  - **Niche-local fitness**: Each niche maintains its own fitness mean/std,
    so a hypothesis that's weak globally can be the fittest in its niche.
  - **Adaptive thresholds**: Evidence and integration thresholds adapt
    per-niche based on the niche's learning velocity.
  - **Metabolic independence**: Each niche has a metabolic budget from Oikos.
    Niches that don't produce useful outputs (predictions, procedures) get
    their budget cut. Starving niches go extinct.

Integration:
  - Created by SpeciationEngine when a CognitiveSpecies meets divergence criteria
  - Consumed by ConsolidationOrchestrator for niche-local consolidation phases
  - Feeds NicheForkingEngine when a niche reaches architectural maturity
  - Reported to Telos for drive-topology integration
"""

from __future__ import annotations

import statistics
from datetime import datetime  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now
from systems.evo.types import (
    CognitiveSpecies,
    HypothesisCategory,
)

logger = structlog.get_logger()


# ─── Constants ──────────────────────────────────────────────────────────────────

_MIN_NICHE_POPULATION: int = 5          # Minimum hypotheses to sustain a niche
_MAX_NICHES: int = 12                   # Global cap on concurrent niches
_ISOLATION_DECAY_RATE: float = 0.02     # Reproductive isolation strengthens over time
_INITIAL_ISOLATION: float = 0.3         # Starting reproductive barrier
_FULL_ISOLATION_THRESHOLD: float = 0.85 # Above this → fully reproductively isolated
_GENE_FLOW_ATTENUATION: float = 0.5    # Evidence strength multiplier for cross-niche flow
_NICHE_STARVATION_CYCLES: int = 5      # Cycles with no output → extinction
_MATURITY_THRESHOLD: float = 0.75      # Niche coherence needed for forking eligibility


# ─── Types ──────────────────────────────────────────────────────────────────────


class NicheGenealogy(EOSBaseModel):
    """Tracks the evolutionary lineage of a niche."""

    parent_niche_id: str | None = None
    generation: int = 1
    split_reason: str = ""              # Why this niche diverged from its parent
    ancestor_chain: list[str] = Field(default_factory=list)


class NicheMetabolism(EOSBaseModel):
    """Metabolic state of a cognitive niche."""

    budget_fraction: float = 0.1        # Fraction of total Evo metabolic budget
    predictions_produced: int = 0       # Useful predictions generated this period
    procedures_contributed: int = 0     # Procedures exported to global pool
    schemas_contributed: int = 0        # Schemas exported to global pool
    starvation_cycles: int = 0          # Consecutive cycles with zero output
    metabolic_efficiency: float = 0.0   # Output / budget (higher = more productive)


class ReproductiveBarrier(EOSBaseModel):
    """Isolation barrier between two niches."""

    niche_a_id: str
    niche_b_id: str
    isolation_coefficient: float = _INITIAL_ISOLATION  # 0=free flow, 1=fully isolated
    last_gene_flow_at: datetime = Field(default_factory=utc_now)
    gene_flow_events: int = 0           # Total cross-niche evidence transfers
    compatibility_score: float = 0.5    # Semantic similarity of niche worldviews


class NicheAdaptiveThresholds(EOSBaseModel):
    """Per-niche adaptive learning parameters."""

    evidence_threshold: float = 3.0     # Score needed for SUPPORTED
    min_supporting_episodes: int = 10
    integration_velocity: float = 1.0   # Speed multiplier for this niche's learning
    complexity_penalty: float = 0.1     # Niche-specific Occam's razor strength


class CognitiveNiche(Identified, Timestamped):
    """
    An isolated hypothesis ecosystem with its own fitness landscape.

    A niche is born when a CognitiveSpecies reaches sufficient divergence
    from the general population. Once isolated, the niche evolves its own
    evidence thresholds, fitness criteria, and processing strategies.
    The niche can eventually propose forking the cognitive architecture
    itself (via NicheForkingEngine).
    """

    name: str = ""
    description: str = ""

    # Population
    hypothesis_ids: list[str] = Field(default_factory=list)
    schema_ids: list[str] = Field(default_factory=list)
    procedure_ids: list[str] = Field(default_factory=list)
    detector_affinities: list[str] = Field(default_factory=list)  # Preferred detectors

    # Domain
    primary_domain: str = ""
    category: HypothesisCategory = HypothesisCategory.WORLD_MODEL
    semantic_centroid: list[float] = Field(default_factory=list)  # 768-dim embedding

    # Fitness landscape
    fitness_mean: float = 0.0
    fitness_std: float = 0.0
    local_fitness_scores: dict[str, float] = Field(default_factory=dict)

    # Speciation metrics
    divergence_from_general: float = 0.0   # How far from the general population
    internal_coherence: float = 0.0        # How consistent are niche members
    reproductive_isolation: float = _INITIAL_ISOLATION

    # Adaptive thresholds
    thresholds: NicheAdaptiveThresholds = Field(default_factory=NicheAdaptiveThresholds)

    # Metabolism
    metabolism: NicheMetabolism = Field(default_factory=NicheMetabolism)

    # Genealogy
    genealogy: NicheGenealogy = Field(default_factory=NicheGenealogy)

    # Maturity (for fork eligibility)
    maturity_score: float = 0.0         # Composite of coherence, isolation, output
    fork_eligible: bool = False
    fork_proposals_submitted: int = 0

    # Lifecycle
    alive: bool = True
    extinction_reason: str = ""

    @property
    def population_size(self) -> int:
        return len(self.hypothesis_ids) + len(self.schema_ids) + len(self.procedure_ids)

    @property
    def is_starving(self) -> bool:
        return self.metabolism.starvation_cycles >= _NICHE_STARVATION_CYCLES

    @property
    def is_mature(self) -> bool:
        return self.maturity_score >= _MATURITY_THRESHOLD

    @property
    def is_fully_isolated(self) -> bool:
        return self.reproductive_isolation >= _FULL_ISOLATION_THRESHOLD


# ─── Niche Registry ────────────────────────────────────────────────────────────


class NicheRegistry:
    """
    Manages the lifecycle of all cognitive niches.

    Handles niche creation from species, membership tracking, gene flow
    between niches, and niche extinction.
    """

    def __init__(self) -> None:
        self._niches: dict[str, CognitiveNiche] = {}
        self._barriers: dict[str, ReproductiveBarrier] = {}  # "a::b" → barrier
        self._hypothesis_to_niche: dict[str, str] = {}  # hyp_id → niche_id
        self._logger = logger.bind(system="evo.cognitive_niche")
        # Cached starvation level from Oikos - updated via set_starvation_level().
        # Niche expansion is blocked when the organism is starving or critical
        # (GROWTH gate semantics: creating new isolated cognitive ecosystems
        # costs ongoing metabolic budget beyond the immediate cycle).
        self._starvation_level: str = "nominal"

    def set_starvation_level(self, level: str) -> None:
        """Update cached Oikos starvation level (called by EvoService event handler)."""
        self._starvation_level = level

    # ─── Niche Creation ─────────────────────────────────────────────────────

    def create_niche_from_species(
        self,
        species: CognitiveSpecies,
        parent_niche_id: str | None = None,
        split_reason: str = "",
    ) -> CognitiveNiche | None:
        """
        Promote a CognitiveSpecies to a full CognitiveNiche.

        Returns None if max niches reached or species too small.
        """
        alive_niches = [n for n in self._niches.values() if n.alive]
        if len(alive_niches) >= _MAX_NICHES:
            self._logger.warning("max_niches_reached", max=_MAX_NICHES)
            return None

        if len(species.member_ids) < _MIN_NICHE_POPULATION:
            return None

        # Metabolic gate: niche expansion requires GROWTH-level resources.
        # Block when Oikos signals starvation or critical pressure.
        if self._starvation_level in ("starving", "critical", "terminal"):
            self._logger.warning(
                "niche_creation_blocked_metabolic_pressure",
                starvation_level=self._starvation_level,
                species=species.name,
            )
            return None

        genealogy = NicheGenealogy(
            parent_niche_id=parent_niche_id,
            generation=1 if parent_niche_id is None else (
                self._niches[parent_niche_id].genealogy.generation + 1
                if parent_niche_id in self._niches else 1
            ),
            split_reason=(
                split_reason
                or f"Divergence from general population (d={species.graph_distance_from_main:.3f})"
            ),
            ancestor_chain=(
                [parent_niche_id] + self._niches[parent_niche_id].genealogy.ancestor_chain
                if parent_niche_id and parent_niche_id in self._niches
                else []
            ),
        )

        niche = CognitiveNiche(
            name=species.name,
            description=f"Cognitive niche evolved from species '{species.name}'",
            hypothesis_ids=list(species.member_ids),
            primary_domain=species.centroid_domain,
            fitness_mean=species.mean_fitness,
            divergence_from_general=species.graph_distance_from_main,
            internal_coherence=1.0 - min(1.0, species.diversity_score),
            genealogy=genealogy,
        )

        # Register membership
        for hyp_id in species.member_ids:
            self._hypothesis_to_niche[hyp_id] = niche.id

        self._niches[niche.id] = niche

        # Create barriers with all existing niches
        for existing_id, existing in self._niches.items():
            if existing_id == niche.id or not existing.alive:
                continue
            barrier_key = self._barrier_key(niche.id, existing_id)
            self._barriers[barrier_key] = ReproductiveBarrier(
                niche_a_id=niche.id,
                niche_b_id=existing_id,
            )

        self._logger.info(
            "niche_created",
            niche_id=niche.id,
            name=niche.name,
            population=niche.population_size,
            parent=parent_niche_id,
            generation=genealogy.generation,
        )

        return niche

    # ─── Gene Flow ──────────────────────────────────────────────────────────

    def get_evidence_attenuation(
        self,
        hypothesis_id: str,
        evidence_source_hypothesis_id: str,
    ) -> float:
        """
        Compute how much to attenuate evidence flowing between niches.

        Same niche → 1.0 (full strength)
        Compatible niches → 0.5–1.0
        Isolated niches → 0.0–0.3
        Fully isolated → 0.0
        """
        source_niche = self._hypothesis_to_niche.get(evidence_source_hypothesis_id)
        target_niche = self._hypothesis_to_niche.get(hypothesis_id)

        # Both in general population or same niche → full strength
        if source_niche is None and target_niche is None:
            return 1.0
        if source_niche == target_niche:
            return 1.0

        # One or both in niches → check barrier
        if source_niche is None or target_niche is None:
            # One in general pop, one in niche → attenuated by half the niche's isolation
            niche_id = source_niche or target_niche
            niche = self._niches.get(niche_id or "")
            if niche is None:
                return 1.0
            return max(0.0, 1.0 - niche.reproductive_isolation * 0.5)

        # Both in different niches → check mutual barrier
        barrier_key = self._barrier_key(source_niche, target_niche)
        barrier = self._barriers.get(barrier_key)
        if barrier is None:
            return _GENE_FLOW_ATTENUATION

        # Record gene flow event
        barrier.gene_flow_events += 1
        barrier.last_gene_flow_at = utc_now()

        return max(0.0, 1.0 - barrier.isolation_coefficient)

    def record_gene_flow(self, source_niche_id: str, target_niche_id: str) -> None:
        """Record that evidence flowed between two niches."""
        if source_niche_id == target_niche_id:
            return
        barrier_key = self._barrier_key(source_niche_id, target_niche_id)
        barrier = self._barriers.get(barrier_key)
        if barrier is not None:
            barrier.gene_flow_events += 1
            barrier.last_gene_flow_at = utc_now()

    # ─── Niche Membership ───────────────────────────────────────────────────

    def assign_hypothesis(self, hypothesis_id: str, niche_id: str) -> bool:
        """Assign a hypothesis to a niche. Returns False if niche doesn't exist."""
        if niche_id not in self._niches:
            return False
        niche = self._niches[niche_id]
        if not niche.alive:
            return False
        if hypothesis_id not in niche.hypothesis_ids:
            niche.hypothesis_ids.append(hypothesis_id)
        self._hypothesis_to_niche[hypothesis_id] = niche_id
        return True

    def get_niche_for_hypothesis(self, hypothesis_id: str) -> CognitiveNiche | None:
        """Return the niche a hypothesis belongs to, or None for general pop."""
        niche_id = self._hypothesis_to_niche.get(hypothesis_id)
        if niche_id is None:
            return None
        return self._niches.get(niche_id)

    def get_niche_hypotheses(self, niche_id: str) -> list[str]:
        """Return all hypothesis IDs in a niche."""
        niche = self._niches.get(niche_id)
        if niche is None:
            return []
        return list(niche.hypothesis_ids)

    # ─── Niche Evolution ────────────────────────────────────────────────────

    def evolve_isolation_barriers(self) -> None:
        """
        Strengthen isolation barriers over time (allopatric speciation).

        Barriers strengthen when niches diverge, weaken when gene flow is frequent.
        """
        for barrier in self._barriers.values():
            niche_a = self._niches.get(barrier.niche_a_id)
            niche_b = self._niches.get(barrier.niche_b_id)
            if niche_a is None or niche_b is None:
                continue
            if not niche_a.alive or not niche_b.alive:
                continue

            # Strengthen: both niches are diverging
            divergence_pressure = abs(
                niche_a.divergence_from_general - niche_b.divergence_from_general
            )
            strengthen = _ISOLATION_DECAY_RATE * (1.0 + divergence_pressure)

            # Weaken: recent gene flow
            time_since_flow = (utc_now() - barrier.last_gene_flow_at).total_seconds() / 3600.0
            weaken = 0.0
            if time_since_flow < 6.0:  # Gene flow in last 6 hours
                flow_ratio = barrier.gene_flow_events / max(1, barrier.gene_flow_events + 5)
                weaken = _ISOLATION_DECAY_RATE * flow_ratio

            barrier.isolation_coefficient = max(0.0, min(1.0,
                barrier.isolation_coefficient + strengthen - weaken
            ))

        # Update niche-level isolation from max barrier
        for niche in self._niches.values():
            if not niche.alive:
                continue
            max_isolation = 0.0
            for barrier in self._barriers.values():
                if barrier.niche_a_id == niche.id or barrier.niche_b_id == niche.id:
                    max_isolation = max(max_isolation, barrier.isolation_coefficient)
            niche.reproductive_isolation = max_isolation

    def update_niche_fitness(
        self,
        niche_id: str,
        hypothesis_fitness: dict[str, float],
    ) -> None:
        """Recompute niche-local fitness statistics."""
        niche = self._niches.get(niche_id)
        if niche is None:
            return

        niche_scores: list[float] = []
        for hyp_id in niche.hypothesis_ids:
            score = hypothesis_fitness.get(hyp_id, 0.0)
            niche.local_fitness_scores[hyp_id] = score
            niche_scores.append(score)

        if niche_scores:
            niche.fitness_mean = statistics.mean(niche_scores)
            niche.fitness_std = statistics.stdev(niche_scores) if len(niche_scores) > 1 else 0.0

    def update_niche_metabolism(
        self,
        niche_id: str,
        predictions: int = 0,
        procedures: int = 0,
        schemas: int = 0,
    ) -> None:
        """Update niche metabolic output counters."""
        niche = self._niches.get(niche_id)
        if niche is None:
            return

        m = niche.metabolism
        m.predictions_produced += predictions
        m.procedures_contributed += procedures
        m.schemas_contributed += schemas

        total_output = predictions + procedures + schemas
        if total_output == 0:
            m.starvation_cycles += 1
        else:
            m.starvation_cycles = 0

        m.metabolic_efficiency = total_output / max(0.01, m.budget_fraction * 100)

    def compute_maturity(self, niche_id: str) -> float:
        """
        Compute niche maturity score - determines fork eligibility.

        maturity = coherence × isolation × metabolic_efficiency × age_factor
        """
        niche = self._niches.get(niche_id)
        if niche is None:
            return 0.0

        age_hours = (utc_now() - niche.created_at).total_seconds() / 3600.0
        age_factor = min(1.0, age_hours / 72.0)  # Fully mature after 72h

        maturity = (
            niche.internal_coherence * 0.3
            + niche.reproductive_isolation * 0.25
            + niche.metabolism.metabolic_efficiency * 0.25
            + age_factor * 0.2
        )

        niche.maturity_score = round(maturity, 4)
        niche.fork_eligible = maturity >= _MATURITY_THRESHOLD
        return maturity

    # ─── Niche Extinction ───────────────────────────────────────────────────

    def extinct_niche(self, niche_id: str, reason: str) -> list[str]:
        """
        Kill a niche, returning its hypotheses to the general population.

        Returns the list of freed hypothesis IDs.
        """
        niche = self._niches.get(niche_id)
        if niche is None or not niche.alive:
            return []

        freed_ids = list(niche.hypothesis_ids)
        for hyp_id in freed_ids:
            self._hypothesis_to_niche.pop(hyp_id, None)

        niche.alive = False
        niche.extinction_reason = reason

        self._logger.info(
            "niche_extinct",
            niche_id=niche_id,
            name=niche.name,
            reason=reason,
            freed_hypotheses=len(freed_ids),
        )

        return freed_ids

    def run_niche_extinction_sweep(self) -> list[str]:
        """
        Check all niches for extinction conditions.

        Returns list of extinct niche IDs.
        """
        extinct_ids: list[str] = []
        for niche in list(self._niches.values()):
            if not niche.alive:
                continue

            if niche.is_starving:
                reason = (
                    f"Metabolic starvation ({_NICHE_STARVATION_CYCLES} "
                    f"cycles with no output)"
                )
                self.extinct_niche(niche.id, reason)
                extinct_ids.append(niche.id)
            elif len(niche.hypothesis_ids) < _MIN_NICHE_POPULATION:
                reason = f"Population collapse (below {_MIN_NICHE_POPULATION})"
                self.extinct_niche(niche.id, reason)
                extinct_ids.append(niche.id)

        return extinct_ids

    # ─── Adaptive Threshold Evolution ───────────────────────────────────────

    def adapt_niche_thresholds(
        self,
        niche_id: str,
        integration_success_rate: float,
        hypothesis_throughput: float,
    ) -> None:
        """
        Per-niche threshold adaptation.

        Fast-learning niches (high throughput, high success) get tighter thresholds.
        Struggling niches get looser thresholds to encourage exploration.
        """
        niche = self._niches.get(niche_id)
        if niche is None:
            return

        t = niche.thresholds

        if integration_success_rate > 0.7 and hypothesis_throughput > 1.0:
            # Successful niche → raise the bar
            t.evidence_threshold = min(6.0, t.evidence_threshold + 0.1)
            t.integration_velocity = max(0.5, t.integration_velocity - 0.05)
        elif integration_success_rate < 0.3:
            # Struggling → lower the bar to encourage experimentation
            t.evidence_threshold = max(1.5, t.evidence_threshold - 0.1)
            t.integration_velocity = min(2.0, t.integration_velocity + 0.05)

    # ─── Query ──────────────────────────────────────────────────────────────

    def get_alive_niches(self) -> list[CognitiveNiche]:
        return [n for n in self._niches.values() if n.alive]

    def get_fork_eligible_niches(self) -> list[CognitiveNiche]:
        return [n for n in self._niches.values() if n.alive and n.fork_eligible]

    def get_niche(self, niche_id: str) -> CognitiveNiche | None:
        return self._niches.get(niche_id)

    @property
    def stats(self) -> dict[str, Any]:
        alive = self.get_alive_niches()
        return {
            "total_niches": len(self._niches),
            "alive_niches": len(alive),
            "fork_eligible": len(self.get_fork_eligible_niches()),
            "total_barriers": len(self._barriers),
            "niches": [
                {
                    "id": n.id,
                    "name": n.name,
                    "population": n.population_size,
                    "isolation": round(n.reproductive_isolation, 3),
                    "maturity": round(n.maturity_score, 3),
                    "fork_eligible": n.fork_eligible,
                    "metabolic_efficiency": round(n.metabolism.metabolic_efficiency, 3),
                }
                for n in alive
            ],
        }

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _barrier_key(a: str, b: str) -> str:
        """Canonical key for a barrier between two niches."""
        return "::".join(sorted([a, b]))
