"""
EcodiaOS - Evolutionary Pressure System

How the organism evolves under real selection pressure instead of just
accumulating knowledge.

Maintains fitness landscapes for hypotheses, schemas, procedures, and
parameters. Implements selection mechanisms (tournament, fitness-proportionate,
niche differentiation), mutation operators (crossover, refinement, recombination),
and extinction events driven by metabolic pressure from Oikos.

Key behaviors:
  - Fitness scoring: prediction_accuracy × compression_ratio × age_survival
  - Tournament selection extended to schemas and procedures (not just hypotheses)
  - Niche differentiation prevents competitive exclusion
  - Extinction events when metabolic pressure is high → aggressive pruning
  - Speciation detection when a cluster diverges from the population

Integration:
  - Oikos metabolic pressure drives extinction threshold
  - Consolidation Phase (new) runs selection + pruning
  - Feeds Logos the updated schema hierarchy
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.evo.types import (
    CognitiveSpecies,
    FitnessScore,
    Hypothesis,
    HypothesisStatus,
    PressureState,
    Procedure,
    SelectionEvent,
)

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

# Fitness computation weights
_PREDICTION_WEIGHT: float = 0.3
_COMPRESSION_WEIGHT: float = 0.25
_AGE_SURVIVAL_WEIGHT: float = 0.15
_REUSE_WEIGHT: float = 0.2
_METABOLIC_EFFICIENCY_WEIGHT: float = 0.1

# Selection thresholds
_EXTINCTION_FITNESS_SIGMA: float = 1.0   # Archive below mean - 1σ
_SPECIATION_DISTANCE: float = 0.7        # Graph distance threshold for speciation
_MIN_SPECIES_SIZE: int = 3               # Minimum members to declare a species
_MAX_SPECIES: int = 10                   # Cap on cognitive species
_NICHE_DIVERSITY_TARGET: float = 0.3     # Maintain at least 30% category diversity


class EvolutionaryPressureSystem:
    """
    Maintains fitness landscapes and applies selection pressure to
    the organism's knowledge structures.

    Coordinates with Oikos for metabolic pressure: expensive knowledge
    must justify its metabolic cost.
    """

    def __init__(
        self,
        memory: MemoryService | None = None,
    ) -> None:
        self._memory = memory
        self._logger = logger.bind(system="evo.pressure")

        # Fitness caches: entity_id → FitnessScore
        self._hypothesis_fitness: dict[str, FitnessScore] = {}
        self._schema_fitness: dict[str, FitnessScore] = {}
        self._procedure_fitness: dict[str, FitnessScore] = {}

        # Selection history
        self._selection_events: list[SelectionEvent] = []
        self._total_selection_events: int = 0
        self._total_extinctions: int = 0

        # Detected cognitive species
        self._species: list[CognitiveSpecies] = []

        # Current metabolic pressure from Oikos (0.0 = none, 1.0 = critical)
        self._metabolic_pressure: float = 0.0

    # ─── Fitness Computation ─────────────────────────────────────────────────

    def score_hypothesis_fitness(
        self,
        hypothesis: Hypothesis,
        prediction_accuracy: float = 0.5,
        compression_ratio: float = 1.0,
    ) -> FitnessScore:
        """
        Compute fitness for a hypothesis.

        fitness = prediction_accuracy × compression_ratio × age_survival
                  - metabolic_cost_penalty

        Higher fitness = more valuable to keep. Low fitness = candidate for archival.
        """
        age_days = (utc_now() - hypothesis.created_at).total_seconds() / 86400.0
        # Age survival: surviving longer is good, but with diminishing returns
        age_survival = min(1.0, math.log1p(age_days) / 3.0)  # Caps at ~20 days

        # Evidence-derived prediction accuracy
        total_evidence = (
            len(hypothesis.supporting_episodes) + len(hypothesis.contradicting_episodes)
        )
        if total_evidence > 0:
            accuracy = len(hypothesis.supporting_episodes) / total_evidence
        else:
            accuracy = prediction_accuracy

        # Reuse: how many episodes has this hypothesis been tested against
        reuse = min(1.0, total_evidence / 20.0)

        # Metabolic cost: complex hypotheses cost more to maintain
        metabolic_cost = hypothesis.complexity_penalty * 0.5

        fitness = (
            accuracy * _PREDICTION_WEIGHT
            + compression_ratio * _COMPRESSION_WEIGHT
            + age_survival * _AGE_SURVIVAL_WEIGHT
            + reuse * _REUSE_WEIGHT
            - metabolic_cost * _METABOLIC_EFFICIENCY_WEIGHT
        )

        score = FitnessScore(
            entity_id=hypothesis.id,
            entity_type="hypothesis",
            fitness=round(fitness, 4),
            prediction_accuracy=round(accuracy, 4),
            compression_ratio=round(compression_ratio, 4),
            age_survival_days=round(age_days, 1),
            reuse_frequency=total_evidence,
            metabolic_cost=round(metabolic_cost, 4),
        )
        self._hypothesis_fitness[hypothesis.id] = score
        return score

    def score_procedure_fitness(
        self,
        procedure: Procedure,
    ) -> FitnessScore:
        """
        Compute fitness for a procedure.

        fitness = success_rate × execution_speed_factor × reuse_frequency_factor
        """
        age_days = (utc_now() - procedure.created_at).total_seconds() / 86400.0
        age_survival = min(1.0, math.log1p(age_days) / 3.0)

        # Execution speed: fewer steps = faster (inverse of step count)
        speed_factor = 1.0 / max(1, len(procedure.steps))

        # Reuse frequency normalised
        reuse = min(1.0, procedure.usage_count / 10.0)

        fitness = (
            procedure.success_rate * _PREDICTION_WEIGHT
            + speed_factor * _COMPRESSION_WEIGHT
            + age_survival * _AGE_SURVIVAL_WEIGHT
            + reuse * _REUSE_WEIGHT
        )

        score = FitnessScore(
            entity_id=procedure.id,
            entity_type="procedure",
            fitness=round(fitness, 4),
            prediction_accuracy=round(procedure.success_rate, 4),
            compression_ratio=round(speed_factor, 4),
            age_survival_days=round(age_days, 1),
            reuse_frequency=procedure.usage_count,
        )
        self._procedure_fitness[procedure.id] = score
        return score

    def score_schema_fitness(
        self,
        schema_id: str,
        mdl_gain: float,
        instance_coverage: int,
        composition_depth: int = 0,
    ) -> FitnessScore:
        """
        Compute fitness for a schema.

        fitness = MDL_gain × instance_coverage_factor × composition_depth_bonus
        """
        coverage_factor = min(1.0, instance_coverage / 50.0)
        depth_bonus = min(0.3, composition_depth * 0.1)
        normalised_mdl = min(1.0, mdl_gain / 100.0)

        fitness = (
            normalised_mdl * 0.4
            + coverage_factor * 0.3
            + depth_bonus * 0.1
        )

        score = FitnessScore(
            entity_id=schema_id,
            entity_type="schema",
            fitness=round(fitness, 4),
            compression_ratio=round(normalised_mdl, 4),
            reuse_frequency=instance_coverage,
        )
        self._schema_fitness[schema_id] = score
        return score

    # ─── Selection Mechanisms ────────────────────────────────────────────────

    def run_selection(
        self,
        hypotheses: list[Hypothesis],
        max_active: int = 50,
    ) -> SelectionEvent:
        """
        Run fitness-proportionate selection on the hypothesis population.

        1. Score all hypotheses
        2. Compute population statistics
        3. Archive those below mean - 1σ (adjusted by metabolic pressure)
        4. Maintain niche diversity - protect minority categories

        Returns the SelectionEvent describing what was pruned.
        """
        if not hypotheses:
            return SelectionEvent(event_type="fitness_proportionate")

        # Score all hypotheses
        scores: list[tuple[Hypothesis, FitnessScore]] = []
        for h in hypotheses:
            score = self.score_hypothesis_fitness(h)
            scores.append((h, score))

        fitness_values = [s.fitness for _, s in scores]
        if not fitness_values:
            return SelectionEvent(event_type="fitness_proportionate")

        mean_fitness = statistics.mean(fitness_values)
        std_fitness = statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0.0

        # Extinction threshold: mean - σ, adjusted by metabolic pressure
        # Higher metabolic pressure → more aggressive pruning
        pressure_multiplier = 1.0 - self._metabolic_pressure * 0.5  # 0→1.0, 1.0→0.5
        extinction_threshold = (
            mean_fitness - _EXTINCTION_FITNESS_SIGMA * std_fitness * pressure_multiplier
        )

        # Identify candidates for pruning
        prune_candidates: list[str] = []
        keep_candidates: list[str] = []

        # Count categories for niche diversity protection
        category_counts: dict[str, int] = defaultdict(int)
        for h, _ in scores:
            category_counts[h.category.value] += 1

        for h, score in scores:
            if score.fitness < extinction_threshold:
                # Check niche diversity: don't extinguish the last member of a category
                cat = h.category.value
                if category_counts[cat] <= 1:
                    keep_candidates.append(h.id)
                    continue
                prune_candidates.append(h.id)
                category_counts[cat] -= 1
            else:
                keep_candidates.append(h.id)

        # If still over max_active, prune lowest-fitness remaining
        if len(keep_candidates) > max_active:
            remaining_scores = [
                (h, s) for h, s in scores if h.id in keep_candidates
            ]
            remaining_scores.sort(key=lambda x: x[1].fitness)
            excess = len(keep_candidates) - max_active
            for h, _ in remaining_scores[:excess]:
                prune_candidates.append(h.id)
                keep_candidates.remove(h.id)

        event = SelectionEvent(
            event_type="fitness_proportionate",
            selected_ids=keep_candidates,
            pruned_ids=prune_candidates,
            reason=(
                f"Fitness threshold {extinction_threshold:.3f} "
                f"(mean={mean_fitness:.3f}, σ={std_fitness:.3f}, "
                f"metabolic_pressure={self._metabolic_pressure:.2f})"
            ),
            population_fitness_mean=round(mean_fitness, 4),
            population_fitness_std=round(std_fitness, 4),
        )

        self._selection_events.append(event)
        self._total_selection_events += 1
        if prune_candidates:
            self._total_extinctions += len(prune_candidates)
            self._logger.info(
                "selection_event",
                pruned=len(prune_candidates),
                kept=len(keep_candidates),
                threshold=round(extinction_threshold, 4),
                metabolic_pressure=round(self._metabolic_pressure, 3),
            )

        return event

    # ─── Mutation Operators ──────────────────────────────────────────────────

    def refine_hypothesis_scope(
        self,
        hypothesis: Hypothesis,
        contradicting_domains: list[str],
    ) -> str:
        """
        Hypothesis refinement: narrow scope based on contradicting evidence.

        When evidence contradicts a broad hypothesis, refine it by adding
        scope conditions that exclude the contradicting domain.

        Returns a refined statement string.
        """
        if not contradicting_domains:
            return hypothesis.statement

        exclusions = ", ".join(contradicting_domains[:3])
        refined = (
            f"{hypothesis.statement} "
            f"(scope: excludes domains [{exclusions}] where contradicting evidence was found)"
        )
        return refined

    def propose_procedure_recombination(
        self,
        proc_a: Procedure,
        proc_b: Procedure,
    ) -> list[dict[str, Any]] | None:
        """
        Procedure recombination: merge steps from two procedures that share
        preconditions.

        Returns merged step list if preconditions overlap, None otherwise.
        """
        # Check precondition overlap
        a_pre = set(proc_a.preconditions)
        b_pre = set(proc_b.preconditions)
        overlap = a_pre & b_pre

        if not overlap:
            return None

        # Merge: shared preconditions → steps from both (A first, then B unique)
        a_step_types = {s.action_type for s in proc_a.steps}
        merged_steps: list[dict[str, Any]] = [
            {"action_type": s.action_type, "description": s.description}
            for s in proc_a.steps
        ]
        for s in proc_b.steps:
            if s.action_type not in a_step_types:
                merged_steps.append(
                    {"action_type": s.action_type, "description": s.description}
                )

        return merged_steps

    # ─── Extinction Events ───────────────────────────────────────────────────

    def run_extinction_event(
        self,
        hypotheses: list[Hypothesis],
        max_active: int = 50,
    ) -> SelectionEvent:
        """
        Aggressive pruning triggered by high metabolic pressure from Oikos.

        Reduces max_active_hypotheses proportionally to resource scarcity.
        Archives schemas with fitness below population mean - 1σ.
        """
        # Proportional reduction: at metabolic_pressure=1.0, halve the population
        effective_max = max(10, int(max_active * (1.0 - self._metabolic_pressure * 0.5)))

        self._logger.info(
            "extinction_event_triggered",
            metabolic_pressure=round(self._metabolic_pressure, 3),
            effective_max=effective_max,
            original_max=max_active,
            population=len(hypotheses),
        )

        return self.run_selection(hypotheses, max_active=effective_max)

    # ─── Speciation Detection ────────────────────────────────────────────────

    def detect_species(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[CognitiveSpecies]:
        """
        Detect cognitive 'species' - clusters of hypotheses that have
        diverged enough from the general population.

        Uses category + domain clustering as a proxy for graph distance.
        When a cluster is large enough and internally coherent, it's declared
        a new cognitive species.
        """
        # Cluster by category
        clusters: dict[str, list[Hypothesis]] = defaultdict(list)
        for h in hypotheses:
            active_statuses = (
                HypothesisStatus.PROPOSED, HypothesisStatus.TESTING, HypothesisStatus.SUPPORTED,
            )
            if h.status not in active_statuses:
                continue
            clusters[h.category.value].append(h)

        # Compute population-level fitness for comparison
        def _get_fitness(h: Hypothesis) -> FitnessScore:
            default = FitnessScore(entity_id=h.id, entity_type="hypothesis")
            return self._hypothesis_fitness.get(h.id, default)

        all_fitness = [_get_fitness(h) for h in hypotheses]
        pop_mean = statistics.mean(f.fitness for f in all_fitness) if all_fitness else 0.0

        new_species: list[CognitiveSpecies] = []
        for category, members in clusters.items():
            if len(members) < _MIN_SPECIES_SIZE:
                continue

            # Compute cluster fitness
            cluster_fitness = [
                _get_fitness(h).fitness for h in members
            ]
            cluster_mean = statistics.mean(cluster_fitness)

            # Internal diversity: variance of evidence scores
            evidence_scores = [h.evidence_score for h in members]
            diversity = statistics.stdev(evidence_scores) if len(evidence_scores) > 1 else 0.0

            # Distance from main population
            distance = abs(cluster_mean - pop_mean) / max(0.01, pop_mean) if pop_mean > 0 else 0.0

            if distance > _SPECIATION_DISTANCE or len(members) >= _MIN_SPECIES_SIZE * 2:
                species = CognitiveSpecies(
                    name=f"species_{category}",
                    member_ids=[h.id for h in members],
                    centroid_domain=category,
                    mean_fitness=round(cluster_mean, 4),
                    diversity_score=round(diversity, 4),
                    graph_distance_from_main=round(distance, 4),
                )
                new_species.append(species)

        # Cap species count
        new_species = new_species[:_MAX_SPECIES]
        self._species = new_species

        if new_species:
            self._logger.info(
                "cognitive_species_detected",
                count=len(new_species),
                species=[
                    {"name": s.name, "members": len(s.member_ids), "fitness": s.mean_fitness}
                    for s in new_species
                ],
            )

        return new_species

    # ─── Metabolic Pressure Integration ──────────────────────────────────────

    def update_metabolic_pressure(self, pressure: float) -> None:
        """
        Update the current metabolic pressure from Oikos.

        pressure = 0.0 → no pressure (abundant resources)
        pressure = 1.0 → maximum pressure (resource scarcity)
        """
        self._metabolic_pressure = max(0.0, min(1.0, pressure))

    # ─── State Query ─────────────────────────────────────────────────────────

    def get_state(self) -> PressureState:
        """Return a snapshot of the pressure system state."""
        all_fitness = list(self._hypothesis_fitness.values())
        fitness_values = [f.fitness for f in all_fitness] if all_fitness else [0.0]

        return PressureState(
            total_selection_events=self._total_selection_events,
            total_extinctions=self._total_extinctions,
            total_species_detected=len(self._species),
            population_mean_fitness=round(statistics.mean(fitness_values), 4),
            population_std_fitness=round(
                statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0.0, 4
            ),
            metabolic_pressure=round(self._metabolic_pressure, 4),
            active_species=list(self._species),
        )

    @property
    def stats(self) -> dict[str, Any]:
        state = self.get_state()
        return {
            "total_selection_events": state.total_selection_events,
            "total_extinctions": state.total_extinctions,
            "total_species": state.total_species_detected,
            "population_mean_fitness": state.population_mean_fitness,
            "metabolic_pressure": state.metabolic_pressure,
        }
