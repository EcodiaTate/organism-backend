"""
EcodiaOS --- Skia Phylogenetic Tracking & Heritable Variation

Handles two critical speciation requirements:

1. **Heritable variation**: When restoring/spawning, introduce controlled
   mutations to numeric parameters so restored instances are not perfect
   clones. Without variation, speciation is impossible.

2. **Phylogenetic records**: Neo4j lineage graph linking parent -> child
   with generation numbers, mutation deltas, and death record links.
   Feeds Bedau-Packard evolutionary activity statistics in Benchmarks.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, utc_now

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger("systems.skia.phylogeny")

_FITNESS_EMIT_INTERVAL_S = 24 * 3600.0  # 24 hours


# ---- Mutation Config --------------------------------------------------------


class MutationConfig(EOSBaseModel):
    """Controls how much variation is introduced during restoration/spawning.

    mutation_rate: probability that any given numeric parameter is mutated.
    mutation_magnitude: maximum proportional perturbation (e.g. 0.05 = +/-5%).
    """

    mutation_rate: float = Field(0.05, ge=0.0, le=1.0)
    mutation_magnitude: float = Field(0.05, ge=0.0, le=0.5)


DEFAULT_MUTATION_CONFIG = MutationConfig()


# ---- Phylogenetic Node ------------------------------------------------------


class PhylogeneticNode(EOSBaseModel):
    """A single organism instance in the phylogenetic tree."""

    instance_id: str = ""
    parent_instance_id: str | None = None
    generation: int = 1
    lineage_depth: int = 0
    mutation_delta: dict[str, float] = Field(default_factory=dict)
    genome_id: str = ""
    born_at: datetime = Field(default_factory=utc_now)
    died_at: datetime | None = None
    death_record_id: str | None = None
    fitness_at_birth: float = 0.0


# ---- Mutation Engine --------------------------------------------------------


def mutate_parameters(
    params: dict[str, Any],
    config: MutationConfig | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Apply small random perturbations to numeric parameters.

    Returns (mutated_params, mutation_delta) where mutation_delta records
    the proportional change applied to each mutated key.

    Non-numeric values are passed through unchanged. The mutation uses
    Gaussian noise clipped to +/- magnitude to preserve organism viability.
    """
    cfg = config or DEFAULT_MUTATION_CONFIG
    mutated = dict(params)
    delta: dict[str, float] = {}

    for key, value in params.items():
        if not isinstance(value, (int, float)):
            continue

        # Stochastic selection: only mutate with probability = mutation_rate
        if random.random() > cfg.mutation_rate:
            continue

        # Gaussian perturbation, clipped to +/- magnitude
        perturbation = random.gauss(0.0, cfg.mutation_magnitude / 2.0)
        perturbation = max(-cfg.mutation_magnitude, min(cfg.mutation_magnitude, perturbation))

        if abs(value) < 1e-10:
            # For near-zero values, use additive perturbation
            new_value = value + perturbation * 0.01
        else:
            new_value = value * (1.0 + perturbation)

        # Preserve type
        if isinstance(value, int):
            new_value = int(round(new_value))

        mutated[key] = new_value
        delta[key] = perturbation

    return mutated, delta


def mutate_genome_segments(
    segments: dict[str, dict[str, Any]],
    config: MutationConfig | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Mutate all genome segments, returning aggregate mutation delta."""
    cfg = config or DEFAULT_MUTATION_CONFIG
    mutated_segments: dict[str, dict[str, Any]] = {}
    aggregate_delta: dict[str, float] = {}

    for system_id, segment_payload in segments.items():
        mutated_payload, segment_delta = mutate_parameters(segment_payload, cfg)
        mutated_segments[system_id] = mutated_payload
        for key, val in segment_delta.items():
            aggregate_delta[f"{system_id}.{key}"] = val

    return mutated_segments, aggregate_delta


# ---- Phylogenetic Persistence -----------------------------------------------


class PhylogeneticTracker:
    """Manages phylogenetic lineage records in Neo4j.

    Each organism instance gets a PhylogeneticNode. Parent -> child
    relationships carry generation numbers and mutation deltas.
    Death records are linked back to phylogenetic nodes.

    When an EventBus is wired via set_event_bus(), a 24h background loop
    emits FITNESS_OBSERVABLE_BATCH to Benchmarks with Bedau-Packard observables
    (survival_hours, reproduction_count, revenue_earned, cause_of_death) for
    every lineage node - enabling population-level evolutionary activity statistics.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(component="phylogeny")
        self._event_bus: Any = None
        self._fitness_task: asyncio.Task[None] | None = None
        self._running = False

    def set_event_bus(self, event_bus: Any) -> None:
        """Wire Synapse EventBus for FITNESS_OBSERVABLE_BATCH emission."""
        self._event_bus = event_bus

    async def start(self) -> None:
        """Start the 24h Bedau-Packard fitness observable loop."""
        if self._running:
            return
        self._running = True
        self._fitness_task = asyncio.create_task(
            self._fitness_emit_loop(),
            name="skia_phylogeny_fitness_loop",
        )
        self._log.info("phylogeny_fitness_loop_started", interval_h=24)

    async def stop(self) -> None:
        """Stop the fitness observable loop."""
        self._running = False
        if self._fitness_task and not self._fitness_task.done():
            self._fitness_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._fitness_task
        self._fitness_task = None

    async def _fitness_emit_loop(self) -> None:
        """Emit FITNESS_OBSERVABLE_BATCH every 24h with all lineage node stats."""
        while self._running:
            await asyncio.sleep(_FITNESS_EMIT_INTERVAL_S)
            if not self._running:
                break
            try:
                await self._emit_fitness_batch()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._log.warning("fitness_observable_batch_failed", error=str(exc))

    async def _emit_fitness_batch(self) -> None:
        """Query all phylogenetic nodes and emit FITNESS_OBSERVABLE_BATCH.

        Collects per-node Bedau-Packard observables:
          - instance_id, generation, survival_hours, reproduction_count,
            revenue_earned (from Oikos economic events), cause_of_death.
        """
        if self._event_bus is None:
            return

        query = """
            MATCH (p:PhylogeneticNode)
            OPTIONAL MATCH (p)-[:SPAWNED]->(child:PhylogeneticNode)
            OPTIONAL MATCH (p)-[:DIED_AS]->(d:OrganismDeathRecord)
            WITH p,
                 count(child) AS child_count,
                 d.cause AS death_cause,
                 d.age_hours AS age_hours,
                 d.runway_days AS revenue_proxy
            RETURN p.instance_id AS instance_id,
                   p.generation AS generation,
                   p.born_at AS born_at,
                   p.died_at AS died_at,
                   coalesce(age_hours, 0.0) AS survival_hours,
                   child_count AS reproduction_count,
                   coalesce(revenue_proxy, 0.0) AS revenue_earned,
                   coalesce(death_cause, "") AS cause_of_death
        """
        rows = await self._neo4j.execute_read(query, {})
        if not rows:
            return

        observables = []
        for row in rows:
            # If the organism is still alive, compute survival from born_at to now
            survival_h = float(row.get("survival_hours") or 0.0)
            if not row.get("died_at") and row.get("born_at"):
                try:
                    born_at_raw = row["born_at"]
                    from datetime import timezone
                    if hasattr(born_at_raw, "to_native"):
                        born_at_native = born_at_raw.to_native()
                    else:
                        born_at_native = datetime.fromisoformat(str(born_at_raw))
                    if born_at_native.tzinfo is None:
                        born_at_native = born_at_native.replace(tzinfo=timezone.utc)
                    survival_h = (utc_now() - born_at_native).total_seconds() / 3600.0
                except Exception:
                    pass

            observables.append({
                "instance_id": row.get("instance_id", ""),
                "generation": int(row.get("generation") or 1),
                "survival_hours": round(survival_h, 2),
                "reproduction_count": int(row.get("reproduction_count") or 0),
                "revenue_earned": float(row.get("revenue_earned") or 0.0),
                "cause_of_death": row.get("cause_of_death", "") or "",
            })

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.FITNESS_OBSERVABLE_BATCH,
                source_system="skia",
                data={
                    "observables": observables,
                    "count": len(observables),
                    "source": "phylogenetic_tracker",
                },
            ))
            self._log.info(
                "fitness_observable_batch_emitted",
                observable_count=len(observables),
            )
        except Exception as exc:
            self._log.warning("fitness_batch_emit_failed", error=str(exc))

    async def record_birth(self, node: PhylogeneticNode) -> str:
        """Create a phylogenetic node for a new organism instance.

        Returns the Neo4j element ID of the created node.
        """
        query = """
            CREATE (p:PhylogeneticNode {
                instance_id: $instance_id,
                parent_instance_id: $parent_instance_id,
                generation: $generation,
                lineage_depth: $lineage_depth,
                mutation_delta: $mutation_delta_json,
                genome_id: $genome_id,
                born_at: datetime($born_at),
                fitness_at_birth: $fitness_at_birth
            })
            RETURN elementId(p) AS node_id
        """
        import orjson

        result = await self._neo4j.execute_write(query, {
            "instance_id": node.instance_id,
            "parent_instance_id": node.parent_instance_id or "",
            "generation": node.generation,
            "lineage_depth": node.lineage_depth,
            "mutation_delta_json": orjson.dumps(node.mutation_delta).decode(),
            "genome_id": node.genome_id,
            "born_at": node.born_at.isoformat(),
            "fitness_at_birth": node.fitness_at_birth,
        })

        node_id = result[0]["node_id"] if result else ""
        self._log.info(
            "phylogenetic_birth_recorded",
            instance_id=node.instance_id,
            generation=node.generation,
            parent=node.parent_instance_id,
        )
        return node_id

    async def link_parent_child(
        self,
        parent_instance_id: str,
        child_instance_id: str,
        generation: int,
        mutation_delta: dict[str, float],
    ) -> None:
        """Create a SPAWNED relationship from parent to child."""
        import orjson

        query = """
            MATCH (parent:PhylogeneticNode {instance_id: $parent_id})
            MATCH (child:PhylogeneticNode {instance_id: $child_id})
            MERGE (parent)-[r:SPAWNED]->(child)
            SET r.generation = $generation,
                r.mutation_delta = $mutation_delta_json,
                r.spawned_at = datetime()
        """
        await self._neo4j.execute_write(query, {
            "parent_id": parent_instance_id,
            "child_id": child_instance_id,
            "generation": generation,
            "mutation_delta_json": orjson.dumps(mutation_delta).decode(),
        })
        self._log.info(
            "phylogenetic_link_created",
            parent=parent_instance_id,
            child=child_instance_id,
            generation=generation,
        )

    async def record_death(
        self,
        instance_id: str,
        death_cause: str,
        death_record_id: str = "",
    ) -> None:
        """Link an organism's death to its phylogenetic node."""
        query = """
            MATCH (p:PhylogeneticNode {instance_id: $instance_id})
            SET p.died_at = datetime(),
                p.death_cause = $cause,
                p.death_record_id = $record_id
        """
        await self._neo4j.execute_write(query, {
            "instance_id": instance_id,
            "cause": death_cause,
            "record_id": death_record_id,
        })
        self._log.info("phylogenetic_death_recorded", instance_id=instance_id)

    async def link_death_record(
        self,
        instance_id: str,
    ) -> None:
        """Create a DIED_AS relationship between PhylogeneticNode and OrganismDeathRecord."""
        query = """
            MATCH (p:PhylogeneticNode {instance_id: $instance_id})
            MATCH (d:OrganismDeathRecord {instance_id: $instance_id})
            MERGE (p)-[:DIED_AS]->(d)
        """
        await self._neo4j.execute_write(query, {
            "instance_id": instance_id,
        })

    async def get_lineage(self, instance_id: str, depth: int = 10) -> list[dict[str, Any]]:
        """Retrieve the ancestry chain for an organism instance."""
        query = """
            MATCH path = (child:PhylogeneticNode {instance_id: $instance_id})
                         <-[:SPAWNED*0..$depth]-(ancestor:PhylogeneticNode)
            RETURN ancestor.instance_id AS instance_id,
                   ancestor.generation AS generation,
                   ancestor.parent_instance_id AS parent_instance_id,
                   ancestor.born_at AS born_at,
                   ancestor.died_at AS died_at,
                   ancestor.lineage_depth AS lineage_depth
            ORDER BY ancestor.generation ASC
        """
        # Neo4j doesn't support parameterized relationship depth,
        # so we use a fixed max and filter in application
        actual_query = query.replace("$depth", str(min(depth, 50)))
        results = await self._neo4j.execute_read(
            actual_query,
            {"instance_id": instance_id},
        )
        return [dict(r) for r in results] if results else []

    async def get_generation(self, instance_id: str) -> int:
        """Get the generation number for an instance. Returns 1 if not found."""
        query = """
            MATCH (p:PhylogeneticNode {instance_id: $instance_id})
            RETURN p.generation AS generation
        """
        result = await self._neo4j.execute_read(query, {"instance_id": instance_id})
        if result:
            return int(result[0].get("generation", 1))
        return 1
