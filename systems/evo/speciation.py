"""
EcodiaOS — Speciation Engine: Cognitive Architecture Evolution

The difference between a system that learns facts and one that evolves
new ways of thinking.

The SpeciationEngine implements five biological speciation mechanisms
adapted for cognitive architecture evolution:

1. **Allopatric Speciation** — Geographic isolation analog.
   When hypotheses about different domains accumulate incompatible
   evidence, they're forced into separate niches. Isolation strengthens
   over time as the niches develop independent fitness landscapes.

2. **Sympatric Speciation** — Divergence within the same domain.
   Two competing worldviews about the SAME domain both have evidence.
   Instead of forcing one out, they speciate: each becomes a niche
   that processes evidence differently. The organism maintains
   multiple interpretive frameworks for the same phenomenon.

3. **Adaptive Radiation** — Environmental pressure creates diversity.
   When the organism encounters a novel domain (high prediction error,
   no existing hypotheses), it triggers rapid niche creation from
   generalist knowledge, specializing for the new environment.

4. **Parapatric Speciation** — Gradual divergence at domain boundaries.
   Hypotheses at the edge of two niches gradually diverge as they
   adapt to the transition zone. Creates "hybrid" niches that span
   domain boundaries — the organism inventing new categories.

5. **Ring Species** — Circular compatibility chains.
   Niche A is compatible with B, B with C, C with D, but D is
   incompatible with A. Gene flow follows the ring but the endpoints
   can't exchange evidence. Detects genuine worldview incompatibilities
   that aren't visible from pairwise comparison.

Integration:
  - Runs during consolidation as Phase 2.9 (after genetic fixation)
  - Consumes pressure.py species detection + schema algebra
  - Produces niches for cognitive_niche.py to manage
  - Emits SPECIATION_EVENT on Synapse for Telos/Alive visualization
  - Feeds NicheForkingEngine with mature niches
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel
from systems.evo.cognitive_niche import NicheRegistry
from systems.evo.types import (
    CognitiveSpecies,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
)

if TYPE_CHECKING:
    from systems.evo.pressure import EvolutionaryPressureSystem

logger = structlog.get_logger()


# Active hypothesis statuses (reused across speciation checks)
_ACTIVE_STATUSES = (
    HypothesisStatus.PROPOSED,
    HypothesisStatus.TESTING,
    HypothesisStatus.SUPPORTED,
)

# ─── Constants ──────────────────────────────────────────────────────────────────

# Allopatric speciation
_ALLOPATRIC_DIVERGENCE_MIN: float = 0.5        # Min graph distance for allopatric split
_ALLOPATRIC_EVIDENCE_CONFLICT: float = 0.3     # Fraction of contradicting evidence

# Sympatric speciation
_SYMPATRIC_WORLDVIEW_DIVERGENCE: float = 0.4   # Evidence score variance threshold
_SYMPATRIC_MIN_COMPETITORS: int = 4            # Min hypotheses competing in same domain
_SYMPATRIC_BIMODALITY_THRESHOLD: float = 0.3   # Hartigan's dip statistic threshold

# Adaptive radiation
_RADIATION_PREDICTION_ERROR: float = 0.7       # Prediction error that triggers radiation
_RADIATION_MIN_GENERALIST_POP: int = 8         # Min generalist hypotheses to split
_RADIATION_SPLIT_COUNT: int = 3                # Number of niches to create from radiation

# Parapatric speciation
_PARAPATRIC_BOUNDARY_OVERLAP: float = 0.3      # Schema overlap threshold for boundary zone
_PARAPATRIC_DIVERGENCE_RATE: float = 0.05      # How fast boundary hypotheses diverge

# Ring species detection
_RING_MIN_LENGTH: int = 3                      # Minimum ring length
_RING_COMPATIBILITY_THRESHOLD: float = 0.6     # Above this = compatible


# ─── Result Types ───────────────────────────────────────────────────────────────


class SpeciationEvent(EOSBaseModel):
    """Record of a speciation event — a new cognitive niche being born."""

    event_type: str  # "allopatric" | "sympatric" | "radiation" | "parapatric" | "ring"
    new_niche_ids: list[str] = Field(default_factory=list)
    parent_niche_id: str | None = None
    parent_species_name: str = ""
    trigger_description: str = ""
    hypothesis_count: int = 0
    divergence_score: float = 0.0


class RingSpecies(EOSBaseModel):
    """A detected ring species — circular compatibility chain with endpoint incompatibility."""

    ring_niche_ids: list[str] = Field(default_factory=list)
    incompatible_pair: tuple[str, str] = ("", "")
    ring_length: int = 0
    description: str = ""


class SpeciationResult(EOSBaseModel):
    """Summary of one speciation phase during consolidation."""

    events: list[SpeciationEvent] = Field(default_factory=list)
    niches_created: int = 0
    niches_extinct: int = 0
    ring_species_detected: int = 0
    barriers_evolved: int = 0
    duration_ms: int = 0


# ─── Engine ─────────────────────────────────────────────────────────────────────


class SpeciationEngine:
    """
    Drives cognitive speciation — the organism evolving new ways of thinking.

    Each consolidation cycle:
      1. Check for allopatric speciation (domain divergence)
      2. Check for sympatric speciation (same-domain worldview splits)
      3. Check for adaptive radiation (novel domain pressure)
      4. Check for parapatric speciation (boundary zone divergence)
      5. Detect ring species (circular incompatibility)
      6. Evolve isolation barriers
      7. Sweep extinct niches
      8. Update maturity scores
    """

    def __init__(
        self,
        niche_registry: NicheRegistry,
        pressure_system: EvolutionaryPressureSystem | None = None,
    ) -> None:
        self._registry = niche_registry
        self._pressure = pressure_system
        self._logger = logger.bind(system="evo.speciation")
        self._speciation_history: list[SpeciationEvent] = []
        self._ring_species: list[RingSpecies] = []
        self._total_speciation_events: int = 0

    # ─── Main Entry Point ───────────────────────────────────────────────────

    async def run_speciation_cycle(
        self,
        hypotheses: list[Hypothesis],
        species: list[CognitiveSpecies],
        prediction_errors: dict[str, float] | None = None,
        hypothesis_fitness: dict[str, float] | None = None,
    ) -> SpeciationResult:
        """
        Run a full speciation cycle during consolidation.

        Args:
            hypotheses: All active hypotheses
            species: Detected cognitive species from pressure.py
            prediction_errors: domain → mean prediction error (from Fovea)
            hypothesis_fitness: hyp_id → fitness score
        """
        import time
        t0 = time.monotonic()

        events: list[SpeciationEvent] = []
        prediction_errors = prediction_errors or {}
        hypothesis_fitness = hypothesis_fitness or {}

        # 1. Allopatric speciation — promote divergent species to niches
        for sp in species:
            if sp.graph_distance_from_main >= _ALLOPATRIC_DIVERGENCE_MIN:
                event = self._attempt_allopatric_speciation(sp, hypotheses)
                if event:
                    events.append(event)

        # 2. Sympatric speciation — same-domain worldview splits
        domain_groups = self._group_by_domain(hypotheses)
        for domain, domain_hyps in domain_groups.items():
            if len(domain_hyps) >= _SYMPATRIC_MIN_COMPETITORS:
                event = self._attempt_sympatric_speciation(domain, domain_hyps, hypothesis_fitness)
                if event:
                    events.append(event)

        # 3. Adaptive radiation — novel domain pressure
        for domain, error in prediction_errors.items():
            if error >= _RADIATION_PREDICTION_ERROR:
                event = self._attempt_adaptive_radiation(domain, hypotheses, hypothesis_fitness)
                if event:
                    events.append(event)

        # 4. Parapatric speciation — boundary zone divergence
        parapatric_events = self._check_parapatric_speciation(hypotheses, hypothesis_fitness)
        events.extend(parapatric_events)

        # 5. Ring species detection
        rings = self._detect_ring_species()
        self._ring_species = rings

        # 6. Evolve isolation barriers
        self._registry.evolve_isolation_barriers()

        # 7. Extinction sweep
        extinct_ids = self._registry.run_niche_extinction_sweep()

        # 8. Update maturity scores for all alive niches
        for niche in self._registry.get_alive_niches():
            self._registry.compute_maturity(niche.id)

        # Record history
        self._speciation_history.extend(events)
        self._total_speciation_events += len(events)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        result = SpeciationResult(
            events=events,
            niches_created=sum(len(e.new_niche_ids) for e in events),
            niches_extinct=len(extinct_ids),
            ring_species_detected=len(rings),
            barriers_evolved=len(self._registry._barriers),
            duration_ms=elapsed_ms,
        )

        if events:
            self._logger.info(
                "speciation_cycle_complete",
                events=len(events),
                niches_created=result.niches_created,
                niches_extinct=result.niches_extinct,
                ring_species=result.ring_species_detected,
                elapsed_ms=elapsed_ms,
            )

        return result

    # ─── 1. Allopatric Speciation ───────────────────────────────────────────

    def _attempt_allopatric_speciation(
        self,
        species: CognitiveSpecies,
        hypotheses: list[Hypothesis],
    ) -> SpeciationEvent | None:
        """
        Promote a highly divergent species to a cognitive niche.

        Requires: graph_distance_from_main >= threshold AND
                  internal evidence conflict rate is low (coherent worldview).
        """
        # Check if species is already a niche
        existing_niches = self._registry.get_alive_niches()
        for existing in existing_niches:
            if set(species.member_ids) & set(existing.hypothesis_ids):
                return None  # Already part of a niche

        # Check evidence conflict within species
        species_hyps = [h for h in hypotheses if h.id in species.member_ids]
        if not species_hyps:
            return None

        total_contradictions = sum(len(h.contradicting_episodes) for h in species_hyps)
        total_evidence = sum(
            len(h.supporting_episodes) + len(h.contradicting_episodes)
            for h in species_hyps
        )
        conflict_rate = total_contradictions / max(1, total_evidence)

        if conflict_rate > _ALLOPATRIC_EVIDENCE_CONFLICT:
            return None  # Too much internal conflict — not a coherent species

        niche = self._registry.create_niche_from_species(
            species,
            split_reason=(
                f"Allopatric: divergence {species.graph_distance_from_main:.3f} "
                f"from general population"
            ),
        )
        if niche is None:
            return None

        return SpeciationEvent(
            event_type="allopatric",
            new_niche_ids=[niche.id],
            parent_species_name=species.name,
            trigger_description=(
                f"Domain '{species.centroid_domain}' diverged "
                f"{species.graph_distance_from_main:.3f} from general "
                f"population (threshold {_ALLOPATRIC_DIVERGENCE_MIN})"
            ),
            hypothesis_count=len(species.member_ids),
            divergence_score=species.graph_distance_from_main,
        )

    # ─── 2. Sympatric Speciation ────────────────────────────────────────────

    def _attempt_sympatric_speciation(
        self,
        domain: str,
        domain_hyps: list[Hypothesis],
        hypothesis_fitness: dict[str, float],
    ) -> SpeciationEvent | None:
        """
        Detect worldview splits within a single domain.

        Two groups of hypotheses about the SAME domain both have evidence
        but make incompatible predictions. Instead of killing one, we
        speciate: the organism maintains multiple interpretive frameworks.

        Uses bimodality detection on evidence scores — if the distribution
        is bimodal, there are two competing worldviews.
        """
        # Skip if already niche-assigned
        already_niched = sum(
            1 for h in domain_hyps
            if self._registry.get_niche_for_hypothesis(h.id) is not None
        )
        if already_niched > len(domain_hyps) * 0.5:
            return None

        scores = [h.evidence_score for h in domain_hyps]
        if len(scores) < _SYMPATRIC_MIN_COMPETITORS:
            return None

        # Check bimodality via gap detection
        # Sort scores, find the largest gap relative to range
        sorted_scores = sorted(scores)
        score_range = sorted_scores[-1] - sorted_scores[0]
        if score_range < 0.5:
            return None  # Too compressed — no real divergence

        max_gap = 0.0
        split_idx = 0
        for i in range(1, len(sorted_scores)):
            gap = sorted_scores[i] - sorted_scores[i - 1]
            if gap > max_gap:
                max_gap = gap
                split_idx = i

        bimodality = max_gap / max(0.01, score_range)
        if bimodality < _SYMPATRIC_BIMODALITY_THRESHOLD:
            return None  # Unimodal — no worldview split

        # Split into two groups
        threshold = (sorted_scores[split_idx - 1] + sorted_scores[split_idx]) / 2.0
        group_a = [h for h in domain_hyps if h.evidence_score < threshold]
        group_b = [h for h in domain_hyps if h.evidence_score >= threshold]

        if len(group_a) < 3 or len(group_b) < 3:
            return None  # Groups too small

        new_niche_ids: list[str] = []
        for i, group in enumerate([group_a, group_b]):
            species = CognitiveSpecies(
                name=f"sympatric_{domain}_{i}",
                member_ids=[h.id for h in group],
                centroid_domain=domain,
                mean_fitness=statistics.mean(
                    hypothesis_fitness.get(h.id, 0.0) for h in group
                ),
            )
            niche = self._registry.create_niche_from_species(
                species,
                split_reason=(
                    f"Sympatric: bimodal worldview split in domain "
                    f"'{domain}' (bimodality={bimodality:.3f})"
                ),
            )
            if niche:
                new_niche_ids.append(niche.id)

        if not new_niche_ids:
            return None

        return SpeciationEvent(
            event_type="sympatric",
            new_niche_ids=new_niche_ids,
            parent_species_name=f"domain_{domain}",
            trigger_description=(
                f"Bimodal worldview split in domain '{domain}': "
                f"bimodality={bimodality:.3f}, groups of {len(group_a)} and {len(group_b)}"
            ),
            hypothesis_count=len(domain_hyps),
            divergence_score=bimodality,
        )

    # ─── 3. Adaptive Radiation ──────────────────────────────────────────────

    def _attempt_adaptive_radiation(
        self,
        novel_domain: str,
        all_hypotheses: list[Hypothesis],
        hypothesis_fitness: dict[str, float],
    ) -> SpeciationEvent | None:
        """
        Rapid niche creation when the organism encounters a novel domain.

        High prediction error in a domain + no existing niches covering it
        → split generalist hypotheses into specialized sub-niches that
        explore different aspects of the novel domain.
        """
        # Check no existing niche covers this domain
        for existing in self._registry.get_alive_niches():
            if existing.primary_domain == novel_domain:
                return None  # Already covered

        # Find generalist hypotheses that could apply
        generalists = [
            h for h in all_hypotheses
            if h.status in _ACTIVE_STATUSES
            and h.category == HypothesisCategory.WORLD_MODEL
        ]

        if len(generalists) < _RADIATION_MIN_GENERALIST_POP:
            return None

        # Split generalists into N sub-groups by evidence score quartiles
        sorted_by_fitness = sorted(generalists, key=lambda h: hypothesis_fitness.get(h.id, 0.0))
        chunk_size = max(3, len(sorted_by_fitness) // _RADIATION_SPLIT_COUNT)
        chunks = [
            sorted_by_fitness[i:i + chunk_size]
            for i in range(0, len(sorted_by_fitness), chunk_size)
        ][:_RADIATION_SPLIT_COUNT]

        new_niche_ids: list[str] = []
        for i, chunk in enumerate(chunks):
            if len(chunk) < 3:
                continue
            species = CognitiveSpecies(
                name=f"radiation_{novel_domain}_{i}",
                member_ids=[h.id for h in chunk],
                centroid_domain=novel_domain,
                mean_fitness=statistics.mean(hypothesis_fitness.get(h.id, 0.0) for h in chunk),
            )
            niche = self._registry.create_niche_from_species(
                species,
                split_reason=(
                    f"Adaptive radiation into novel domain "
                    f"'{novel_domain}' (high prediction error)"
                ),
            )
            if niche:
                new_niche_ids.append(niche.id)

        if not new_niche_ids:
            return None

        return SpeciationEvent(
            event_type="radiation",
            new_niche_ids=new_niche_ids,
            parent_species_name="generalist_population",
            trigger_description=(
                f"Adaptive radiation into novel domain '{novel_domain}': "
                f"created {len(new_niche_ids)} specialized niches from "
                f"{len(generalists)} generalist hypotheses"
            ),
            hypothesis_count=len(generalists),
            divergence_score=0.0,
        )

    # ─── 4. Parapatric Speciation ───────────────────────────────────────────

    def _check_parapatric_speciation(
        self,
        hypotheses: list[Hypothesis],
        hypothesis_fitness: dict[str, float],
    ) -> list[SpeciationEvent]:
        """
        Detect gradual divergence at domain boundaries.

        When hypotheses span two niches (boundary zone), check if they're
        diverging enough to warrant a new "hybrid" niche.
        """
        events: list[SpeciationEvent] = []
        alive_niches = self._registry.get_alive_niches()

        if len(alive_niches) < 2:
            return events

        # Find hypotheses that are NOT in any niche (boundary candidates)
        unniched = [
            h for h in hypotheses
            if self._registry.get_niche_for_hypothesis(h.id) is None
            and h.status in _ACTIVE_STATUSES
        ]

        if len(unniched) < 5:
            return events

        # Check if unniched hypotheses cluster between two specific niches
        # by checking which niches they're evidence-compatible with
        niche_affinity: dict[str, list[str]] = defaultdict(list)
        for h in unniched:
            for candidate_niche in alive_niches:
                # Compatibility: any shared supporting episodes
                niche_hyp_episodes: set[str] = set()
                for nh_id in candidate_niche.hypothesis_ids:
                    matching = [mh for mh in hypotheses if mh.id == nh_id]
                    for mh in matching:
                        niche_hyp_episodes.update(mh.supporting_episodes)
                overlap = set(h.supporting_episodes) & niche_hyp_episodes
                if overlap:
                    niche_affinity[h.id].append(candidate_niche.id)

        # Hypotheses with affinity to exactly 2 niches → boundary zone
        boundary_hyps = [
            h for h in unniched
            if len(niche_affinity.get(h.id, [])) == 2
        ]

        if len(boundary_hyps) >= 5:
            # Find the most common niche pair
            pair_counts: dict[str, int] = defaultdict(int)
            for h in boundary_hyps:
                pair = tuple(sorted(niche_affinity[h.id][:2]))
                pair_counts["::".join(pair)] += 1

            if pair_counts:
                top_pair_key = max(pair_counts, key=pair_counts.get)  # type: ignore[arg-type]
                top_pair_ids = top_pair_key.split("::")
                top_count = pair_counts[top_pair_key]

                if top_count >= 5:
                    # Create hybrid niche from boundary hypotheses
                    boundary_for_pair = [
                        h for h in boundary_hyps
                        if set(niche_affinity.get(h.id, [])) == set(top_pair_ids)
                    ]

                    species = CognitiveSpecies(
                        name=f"parapatric_{'_'.join(top_pair_ids[:2])}",
                        member_ids=[h.id for h in boundary_for_pair],
                        centroid_domain="boundary",
                        mean_fitness=statistics.mean(
                            hypothesis_fitness.get(h.id, 0.0) for h in boundary_for_pair
                        ),
                    )

                    niche = self._registry.create_niche_from_species(
                        species,
                        split_reason=(
                            f"Parapatric: {top_count} hypotheses in boundary zone "
                            f"between niches {top_pair_ids}"
                        ),
                    )
                    if niche:
                        events.append(SpeciationEvent(
                            event_type="parapatric",
                            new_niche_ids=[niche.id],
                            trigger_description=(
                                f"Boundary zone between niches {top_pair_ids}: "
                                f"{top_count} hypotheses with dual affinity"
                            ),
                            hypothesis_count=len(boundary_for_pair),
                        ))

        return events

    # ─── 5. Ring Species Detection ──────────────────────────────────────────

    def _detect_ring_species(self) -> list[RingSpecies]:
        """
        Detect circular compatibility chains where endpoints are incompatible.

        A ring species reveals genuine worldview incompatibilities that
        aren't visible from pairwise comparison. This is the organism
        discovering that its beliefs form a non-transitive graph.
        """
        rings: list[RingSpecies] = []
        alive_niches = self._registry.get_alive_niches()

        if len(alive_niches) < _RING_MIN_LENGTH:
            return rings

        # Build compatibility graph
        niche_ids = [n.id for n in alive_niches]
        compatibility: dict[str, dict[str, float]] = defaultdict(dict)

        for i, n_a in enumerate(alive_niches):
            for n_b in alive_niches[i + 1:]:
                barrier_key = NicheRegistry._barrier_key(n_a.id, n_b.id)
                barrier = self._registry._barriers.get(barrier_key)
                compat = (
                    1.0 - barrier.isolation_coefficient if barrier else 0.5
                )
                compatibility[n_a.id][n_b.id] = compat
                compatibility[n_b.id][n_a.id] = compat

        # Find chains where adjacent nodes are compatible but endpoints aren't
        # DFS to find paths of length >= 3
        for start_id in niche_ids:
            self._find_rings(
                start_id, start_id, [start_id],
                compatibility, rings, set(),
            )

        return rings

    def _find_rings(
        self,
        start_id: str,
        current_id: str,
        path: list[str],
        compatibility: dict[str, dict[str, float]],
        rings: list[RingSpecies],
        visited_starts: set[str],
    ) -> None:
        """Recursive DFS to find ring species."""
        if len(path) > 6:  # Cap ring length
            return

        for neighbor_id, compat in compatibility.get(current_id, {}).items():
            if compat < _RING_COMPATIBILITY_THRESHOLD:
                continue  # Not compatible — can't extend chain

            if neighbor_id == start_id and len(path) >= _RING_MIN_LENGTH:
                # Check if endpoints are incompatible
                endpoint_compat = compatibility.get(path[-1], {}).get(start_id, 1.0)
                if endpoint_compat < (1.0 - _RING_COMPATIBILITY_THRESHOLD):
                    # Ring detected!
                    ring_key = "::".join(sorted(path))
                    if ring_key not in visited_starts:
                        visited_starts.add(ring_key)
                        rings.append(RingSpecies(
                            ring_niche_ids=list(path),
                            incompatible_pair=(path[-1], start_id),
                            ring_length=len(path),
                            description=(
                                f"Ring species: {' → '.join(path)} → {start_id} "
                                f"(endpoints incompatible: {endpoint_compat:.3f})"
                            ),
                        ))
                continue

            if neighbor_id in path:
                continue  # No revisits

            self._find_rings(
                start_id, neighbor_id, path + [neighbor_id],
                compatibility, rings, visited_starts,
            )

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _group_by_domain(hypotheses: list[Hypothesis]) -> dict[str, list[Hypothesis]]:
        """Group active hypotheses by category domain."""
        groups: dict[str, list[Hypothesis]] = defaultdict(list)
        active = _ACTIVE_STATUSES
        for h in hypotheses:
            if h.status in active:
                groups[h.category.value].append(h)
        return groups

    # ─── State ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_speciation_events": self._total_speciation_events,
            "recent_events": [
                {
                    "type": e.event_type,
                    "niches_created": len(e.new_niche_ids),
                    "trigger": e.trigger_description[:100],
                }
                for e in self._speciation_history[-5:]
            ],
            "ring_species": len(self._ring_species),
            "niche_stats": self._registry.stats,
        }
