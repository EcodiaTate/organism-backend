"""
EcodiaOS - Simula Evolution Lineage Tracker

Multi-generation evolution tracking: which instance spawned which, what
genomes were inherited, what novel mutations each generation added, and
comparative performance metrics for population-level selection.

Graph schema:
    (:GenerationRecord {instance_id, generation, ...})
        -[:DESCENDED_FROM]-> (:GenerationRecord)    parent chain
        -[:HAS_SIMULA_GENOME]-> (:SimulaGenome)     inherited genome
        -[:HAS_BELIEF_GENOME]-> (:BeliefGenome)     inherited beliefs
    (:LineageEvent {event_type, ...})
        -[:LINEAGE_EVENT_OF]-> (:GenerationRecord)

Population-level selection:
    Fitness = (applied/processed) * alignment * (1 - rollback_rate)
             + 0.1 * (novel_mutations / total_applied)

    The highest-fitness genome is preferentially propagated during
    the next mitosis round, even across independent lineage branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import new_id, utc_now
from systems.simula.genome_types import (
    GenerationRecord,
    LineageEventType,
    PopulationSnapshot,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


class EvolutionLineageTracker:
    """
    Tracks multi-generation lineage in Neo4j and enables
    population-level selection across the fleet.

    Each instance gets a :GenerationRecord node at birth.
    The parent chain is maintained via [:DESCENDED_FROM] relationships.
    Performance metrics are updated periodically so the fleet-wide
    selection logic can pick the best genome for propagation.
    """

    def __init__(self, neo4j: Neo4jClient, instance_id: str) -> None:
        self._neo4j = neo4j
        self._instance_id = instance_id
        self._log = logger.bind(subsystem="simula.lineage")

        from systems.synapse.sentinel import ErrorSentinel
        self._sentinel = ErrorSentinel("simula.lineage")

    # ─── Registration ───────────────────────────────────────────────────────

    async def register_genesis(self) -> GenerationRecord:
        """
        Register generation 0 - the root of a new lineage.

        Called once during the very first instance's birth when there
        is no parent. Creates the root :GenerationRecord.
        """
        record = GenerationRecord(
            id=new_id(),
            instance_id=self._instance_id,
            generation=0,
        )

        await self._persist_generation_record(record)
        await self._record_event(
            record,
            LineageEventType.SPAWNED,
            {"kind": "genesis"},
        )

        self._log.info(
            "lineage_genesis_registered",
            instance_id=self._instance_id,
            record_id=record.id,
        )
        return record

    async def register_child(
        self,
        child_instance_id: str,
        parent_instance_id: str,
        generation: int,
        belief_genome_id: str = "",
        simula_genome_id: str = "",
    ) -> GenerationRecord:
        """
        Register a new generation when a child is spawned.

        Creates a :GenerationRecord for the child, links it
        [:DESCENDED_FROM] to the parent's record, and optionally links
        [:HAS_SIMULA_GENOME] / [:HAS_BELIEF_GENOME] references.
        """
        record = GenerationRecord(
            id=new_id(),
            instance_id=child_instance_id,
            parent_instance_id=parent_instance_id,
            generation=generation,
            belief_genome_id=belief_genome_id,
            simula_genome_id=simula_genome_id,
        )

        await self._persist_generation_record(record)

        # Link to parent's GenerationRecord
        try:
            await self._neo4j.execute_write(
                """
                MATCH (child:GenerationRecord {instance_id: $child_id})
                MATCH (parent:GenerationRecord {instance_id: $parent_id})
                CREATE (child)-[:DESCENDED_FROM {
                    generation: $gen,
                    linked_at: datetime($now)
                }]->(parent)
                """,
                {
                    "child_id": child_instance_id,
                    "parent_id": parent_instance_id,
                    "gen": generation,
                    "now": utc_now().isoformat(),
                },
            )
        except Exception as exc:
            self._log.warning("lineage_descended_link_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "descended_link", "instance_id": self._instance_id},
            )

        # Link genome references
        if simula_genome_id:
            try:
                await self._neo4j.execute_write(
                    """
                    MATCH (r:GenerationRecord {instance_id: $child_id})
                    MATCH (g:SimulaGenome {id: $genome_id})
                    CREATE (r)-[:HAS_SIMULA_GENOME]->(g)
                    """,
                    {
                        "child_id": child_instance_id,
                        "genome_id": simula_genome_id,
                    },
                )
            except Exception as exc:
                self._log.debug("lineage_simula_link_failed", error=str(exc))

        if belief_genome_id:
            try:
                await self._neo4j.execute_write(
                    """
                    MATCH (r:GenerationRecord {instance_id: $child_id})
                    MATCH (g:BeliefGenome {id: $genome_id})
                    CREATE (r)-[:HAS_BELIEF_GENOME]->(g)
                    """,
                    {
                        "child_id": child_instance_id,
                        "genome_id": belief_genome_id,
                    },
                )
            except Exception as exc:
                self._log.debug("lineage_belief_link_failed", error=str(exc))

        await self._record_event(
            record,
            LineageEventType.SPAWNED,
            {
                "parent_id": parent_instance_id,
                "generation": generation,
                "simula_genome_id": simula_genome_id,
                "belief_genome_id": belief_genome_id,
            },
        )

        self._log.info(
            "lineage_child_registered",
            child_id=child_instance_id,
            parent_id=parent_instance_id,
            generation=generation,
        )
        return record

    # ─── Metrics Updates ────────────────────────────────────────────────────

    async def update_metrics(
        self,
        instance_id: str,
        *,
        proposals_processed: int = 0,
        proposals_applied: int = 0,
        rollbacks: int = 0,
        mean_alignment: float = 0.0,
        velocity: float = 0.0,
        episodes: int = 0,
    ) -> None:
        """
        Update performance metrics for an instance's generation record.

        Called periodically by the Simula evolution service to keep
        the lineage tracker up-to-date for selection decisions.
        """
        rollback_rate, fitness_vs_parent = await self._recompute_fitness(
            instance_id=instance_id,
            proposals_processed=proposals_processed,
            proposals_applied=proposals_applied,
            rollbacks=rollbacks,
            mean_alignment=mean_alignment,
        )

        try:
            await self._neo4j.execute_write(
                """
                MATCH (r:GenerationRecord {instance_id: $instance_id})
                SET r.total_proposals_processed = $processed,
                    r.total_proposals_applied = $applied,
                    r.total_rollbacks = $rollbacks,
                    r.mean_constitutional_alignment = $alignment,
                    r.evolution_velocity = $velocity,
                    r.rollback_rate = $rollback_rate,
                    r.total_episodes = $episodes,
                    r.fitness_vs_parent = $fitness,
                    r.last_metrics_update = datetime($now)
                """,
                {
                    "instance_id": instance_id,
                    "processed": proposals_processed,
                    "applied": proposals_applied,
                    "rollbacks": rollbacks,
                    "alignment": mean_alignment,
                    "velocity": velocity,
                    "rollback_rate": rollback_rate,
                    "episodes": episodes,
                    "fitness": fitness_vs_parent,
                    "now": utc_now().isoformat(),
                },
            )
        except Exception as exc:
            self._log.warning(
                "lineage_metrics_update_failed",
                instance_id=instance_id,
                error=str(exc),
            )
            await self._sentinel.report(
                exc, context={"operation": "metrics_update", "instance_id": instance_id},
            )

    async def _recompute_fitness(
        self,
        *,
        instance_id: str,
        proposals_processed: int,
        proposals_applied: int,
        rollbacks: int,
        mean_alignment: float,
    ) -> tuple[float, float]:
        """
        Compute rollback_rate and fitness_vs_parent.

        Fitness = (applied / processed) * alignment * (1 - rollback_rate)
                + 0.1 * (novel_mutations / max(1, total_applied))
        """
        rollback_rate = rollbacks / max(1, proposals_applied)
        apply_rate = proposals_applied / max(1, proposals_processed)

        # Get novel mutations count
        novel_mutations = 0
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (r:GenerationRecord {instance_id: $instance_id})
                RETURN r.novel_mutations AS novel
                """,
                {"instance_id": instance_id},
            )
            if rows and rows[0].get("novel") is not None:
                novel_mutations = int(rows[0]["novel"])
        except Exception:
            pass

        fitness = (
            apply_rate * mean_alignment * (1.0 - rollback_rate)
            + 0.1 * (novel_mutations / max(1, proposals_applied))
        )

        # Compare to parent's fitness
        fitness_vs_parent = 0.0
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (child:GenerationRecord {instance_id: $instance_id})
                      -[:DESCENDED_FROM]->(parent:GenerationRecord)
                RETURN parent.fitness_vs_parent AS parent_fitness,
                       parent.total_proposals_applied AS p_applied,
                       parent.total_proposals_processed AS p_processed,
                       parent.mean_constitutional_alignment AS p_align,
                       parent.rollback_rate AS p_rollback,
                       parent.novel_mutations AS p_novel
                """,
                {"instance_id": instance_id},
            )
            if rows:
                p = rows[0]
                p_applied = int(p.get("p_applied", 0) or 0)
                p_processed = int(p.get("p_processed", 0) or 0)
                p_align = float(p.get("p_align", 0.0) or 0.0)
                p_rollback = float(p.get("p_rollback", 0.0) or 0.0)
                p_novel = int(p.get("p_novel", 0) or 0)
                parent_fitness = (
                    (p_applied / max(1, p_processed))
                    * p_align
                    * (1.0 - p_rollback)
                    + 0.1 * (p_novel / max(1, p_applied))
                )
                fitness_vs_parent = fitness - parent_fitness
        except Exception:
            pass

        return rollback_rate, fitness_vs_parent

    async def record_novel_mutation(
        self,
        instance_id: str,
        proposal_id: str = "",
    ) -> None:
        """Increment novel mutation counter for a generation record."""
        try:
            await self._neo4j.execute_write(
                """
                MATCH (r:GenerationRecord {instance_id: $instance_id})
                SET r.novel_mutations = coalesce(r.novel_mutations, 0) + 1,
                    r.last_metrics_update = datetime($now)
                """,
                {
                    "instance_id": instance_id,
                    "now": utc_now().isoformat(),
                },
            )
        except Exception as exc:
            self._log.debug("lineage_novel_mutation_failed", error=str(exc))

        if proposal_id:
            # Fetch the current GenerationRecord to log the event
            try:
                rows = await self._neo4j.execute_read(
                    """
                    MATCH (r:GenerationRecord {instance_id: $instance_id})
                    RETURN r.id AS id
                    """,
                    {"instance_id": instance_id},
                )
                if rows:
                    record = GenerationRecord(
                        id=str(rows[0]["id"]),
                        instance_id=instance_id,
                    )
                    await self._record_event(
                        record,
                        LineageEventType.NOVEL_MUTATION,
                        {"proposal_id": proposal_id},
                    )
            except Exception:
                pass

    # ─── Queries ────────────────────────────────────────────────────────────

    async def get_lineage(
        self, instance_id: str | None = None,
    ) -> list[GenerationRecord]:
        """
        Return the full ancestor chain from the given instance back to genesis.
        """
        target = instance_id or self._instance_id
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH path = (child:GenerationRecord {instance_id: $id})
                              -[:DESCENDED_FROM*0..]->(ancestor:GenerationRecord)
                WITH ancestor, length(path) AS depth
                ORDER BY depth ASC
                RETURN ancestor.id AS id,
                       ancestor.instance_id AS instance_id,
                       ancestor.parent_instance_id AS parent_id,
                       ancestor.generation AS generation,
                       ancestor.belief_genome_id AS belief_gid,
                       ancestor.simula_genome_id AS simula_gid,
                       ancestor.total_proposals_processed AS processed,
                       ancestor.total_proposals_applied AS applied,
                       ancestor.total_rollbacks AS rollbacks,
                       ancestor.mean_constitutional_alignment AS alignment,
                       ancestor.evolution_velocity AS velocity,
                       ancestor.rollback_rate AS rb_rate,
                       ancestor.novel_mutations AS novel,
                       ancestor.fitness_vs_parent AS fitness,
                       ancestor.is_alive AS alive,
                       ancestor.spawned_at AS spawned,
                       ancestor.last_metrics_update AS updated
                """,
                {"id": target},
            )
        except Exception as exc:
            self._log.warning("lineage_get_chain_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "get_ancestor_chain"},
            )
            return []

        return [self._row_to_record(row) for row in rows]

    async def get_children(
        self, instance_id: str | None = None,
    ) -> list[GenerationRecord]:
        """Return direct children of the given instance."""
        target = instance_id or self._instance_id
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (child:GenerationRecord)
                      -[:DESCENDED_FROM]->(parent:GenerationRecord {instance_id: $id})
                RETURN child.id AS id,
                       child.instance_id AS instance_id,
                       child.parent_instance_id AS parent_id,
                       child.generation AS generation,
                       child.belief_genome_id AS belief_gid,
                       child.simula_genome_id AS simula_gid,
                       child.total_proposals_processed AS processed,
                       child.total_proposals_applied AS applied,
                       child.total_rollbacks AS rollbacks,
                       child.mean_constitutional_alignment AS alignment,
                       child.evolution_velocity AS velocity,
                       child.rollback_rate AS rb_rate,
                       child.novel_mutations AS novel,
                       child.fitness_vs_parent AS fitness,
                       child.is_alive AS alive,
                       child.spawned_at AS spawned,
                       child.last_metrics_update AS updated
                ORDER BY child.generation ASC
                """,
                {"id": target},
            )
        except Exception as exc:
            self._log.warning("lineage_get_children_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "get_children"},
            )
            return []

        return [self._row_to_record(row) for row in rows]

    async def get_population_snapshot(self) -> PopulationSnapshot:
        """
        Capture a fleet-wide snapshot: all alive instances, their
        generation distribution, and top performers for selection.
        """
        snap = PopulationSnapshot()

        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (r:GenerationRecord)
                RETURN count(r) AS total,
                       sum(CASE WHEN r.is_alive THEN 1 ELSE 0 END) AS alive,
                       max(r.generation) AS max_gen
                """,
            )
            if rows:
                snap.total_instances = int(rows[0].get("total", 0) or 0)
                snap.alive_instances = int(rows[0].get("alive", 0) or 0)
                snap.max_generation = int(rows[0].get("max_gen", 0) or 0)
        except Exception as exc:
            self._log.warning("lineage_snapshot_totals_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "population_snapshot_totals"},
            )

        # Generation distribution
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (r:GenerationRecord)
                WHERE r.is_alive = true
                RETURN r.generation AS gen, count(r) AS cnt
                ORDER BY gen ASC
                """,
            )
            snap.generation_distribution = {
                int(row["gen"]): int(row["cnt"])
                for row in rows
                if row.get("gen") is not None
            }
        except Exception:
            pass

        # Top performers (alive, sorted by fitness)
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (r:GenerationRecord)
                WHERE r.is_alive = true
                  AND r.total_proposals_applied > 0
                RETURN r.id AS id,
                       r.instance_id AS instance_id,
                       r.parent_instance_id AS parent_id,
                       r.generation AS generation,
                       r.belief_genome_id AS belief_gid,
                       r.simula_genome_id AS simula_gid,
                       r.total_proposals_processed AS processed,
                       r.total_proposals_applied AS applied,
                       r.total_rollbacks AS rollbacks,
                       r.mean_constitutional_alignment AS alignment,
                       r.evolution_velocity AS velocity,
                       r.rollback_rate AS rb_rate,
                       r.novel_mutations AS novel,
                       r.fitness_vs_parent AS fitness,
                       r.is_alive AS alive,
                       r.spawned_at AS spawned,
                       r.last_metrics_update AS updated
                ORDER BY r.fitness_vs_parent DESC
                LIMIT 10
                """,
            )
            snap.top_performers = [self._row_to_record(row) for row in rows]
        except Exception:
            pass

        if snap.top_performers:
            best = snap.top_performers[0]
            snap.best_instance_id = best.instance_id
            snap.best_fitness = best.fitness_vs_parent

            total_fitness = sum(
                p.fitness_vs_parent for p in snap.top_performers
            )
            snap.mean_fitness = total_fitness / len(snap.top_performers)

        snap.captured_at = utc_now()
        return snap

    async def select_best_genome_for_propagation(
        self,
    ) -> str | None:
        """
        Select the SimulaGenome ID with the highest fitness across
        all alive instances. Returns None if no eligible genomes.

        This is the population-level selection primitive: the best
        performing genome is preferentially used for the next spawn,
        even if the requesting parent is not the best performer.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (r:GenerationRecord)-[:HAS_SIMULA_GENOME]->(g:SimulaGenome)
                WHERE r.is_alive = true
                  AND r.total_proposals_applied > 0
                RETURN g.id AS genome_id,
                       r.instance_id AS instance_id,
                       r.fitness_vs_parent AS fitness
                ORDER BY r.fitness_vs_parent DESC
                LIMIT 1
                """,
            )
            if rows:
                genome_id = str(rows[0]["genome_id"])
                self._log.info(
                    "lineage_best_genome_selected",
                    genome_id=genome_id,
                    instance_id=rows[0].get("instance_id"),
                    fitness=rows[0].get("fitness"),
                )
                return genome_id
        except Exception as exc:
            self._log.warning("lineage_select_best_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "select_best_genome"},
            )

        return None

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    async def mark_dead(self, instance_id: str) -> None:
        """Mark an instance as no longer alive in the lineage."""
        try:
            await self._neo4j.execute_write(
                """
                MATCH (r:GenerationRecord {instance_id: $instance_id})
                SET r.is_alive = false,
                    r.last_metrics_update = datetime($now)
                """,
                {
                    "instance_id": instance_id,
                    "now": utc_now().isoformat(),
                },
            )
        except Exception as exc:
            self._log.warning("lineage_mark_dead_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "mark_dead", "instance_id": instance_id},
            )

    # ─── Internal Helpers ───────────────────────────────────────────────────

    async def _record_event(
        self,
        record: GenerationRecord,
        event_type: LineageEventType,
        details: dict[str, object] | None = None,
    ) -> None:
        """Persist a lineage event node linked to the generation record."""
        import json as _json

        try:
            await self._neo4j.execute_write(
                """
                MATCH (r:GenerationRecord {id: $record_id})
                CREATE (e:LineageEvent {
                    id: $event_id,
                    instance_id: $instance_id,
                    event_type: $event_type,
                    details: $details,
                    created_at: datetime($now)
                })-[:LINEAGE_EVENT_OF]->(r)
                """,
                {
                    "record_id": record.id,
                    "event_id": new_id(),
                    "instance_id": record.instance_id,
                    "event_type": event_type.value,
                    "details": _json.dumps(details or {}),
                    "now": utc_now().isoformat(),
                },
            )
        except Exception as exc:
            self._log.debug("lineage_event_failed", error=str(exc))

    async def _persist_generation_record(
        self, record: GenerationRecord,
    ) -> None:
        """Create the :GenerationRecord node in Neo4j."""
        try:
            await self._neo4j.execute_write(
                """
                CREATE (r:GenerationRecord {
                    id: $id,
                    instance_id: $instance_id,
                    parent_instance_id: $parent_id,
                    generation: $generation,
                    belief_genome_id: $belief_gid,
                    simula_genome_id: $simula_gid,
                    total_proposals_processed: 0,
                    total_proposals_applied: 0,
                    total_rollbacks: 0,
                    mean_constitutional_alignment: 0.0,
                    evolution_velocity: 0.0,
                    rollback_rate: 0.0,
                    total_episodes: 0,
                    novel_mutations: 0,
                    novel_abstractions: 0,
                    novel_hypotheses: 0,
                    fitness_vs_parent: 0.0,
                    is_alive: true,
                    spawned_at: datetime($now),
                    last_metrics_update: datetime($now)
                })
                """,
                {
                    "id": record.id,
                    "instance_id": record.instance_id,
                    "parent_id": record.parent_instance_id,
                    "generation": record.generation,
                    "belief_gid": record.belief_genome_id,
                    "simula_gid": record.simula_genome_id,
                    "now": utc_now().isoformat(),
                },
            )
        except Exception as exc:
            self._log.error("lineage_persist_record_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "persist_record", "instance_id": record.instance_id},
            )

    @staticmethod
    def _row_to_record(row: dict[str, object]) -> GenerationRecord:
        """Convert a Neo4j row to a GenerationRecord."""
        return GenerationRecord(
            id=str(row.get("id", "")),
            instance_id=str(row.get("instance_id", "")),
            parent_instance_id=str(row.get("parent_id", "") or ""),
            generation=int(row.get("generation", 0) or 0),
            belief_genome_id=str(row.get("belief_gid", "") or ""),
            simula_genome_id=str(row.get("simula_gid", "") or ""),
            total_proposals_processed=int(row.get("processed", 0) or 0),
            total_proposals_applied=int(row.get("applied", 0) or 0),
            total_rollbacks=int(row.get("rollbacks", 0) or 0),
            mean_constitutional_alignment=float(row.get("alignment", 0.0) or 0.0),
            evolution_velocity=float(row.get("velocity", 0.0) or 0.0),
            rollback_rate=float(row.get("rb_rate", 0.0) or 0.0),
            novel_mutations=int(row.get("novel", 0) or 0),
            fitness_vs_parent=float(row.get("fitness", 0.0) or 0.0),
            is_alive=bool(row.get("alive", True)),
        )
