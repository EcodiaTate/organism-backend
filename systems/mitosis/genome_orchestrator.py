"""
EcodiaOS - Genome Orchestrator

Coordinates genome extraction across all systems and assembly into
OrganismGenome. Used by MitosisEngine before spawning a child, and
by child instances on boot to seed from a parent's genome.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

import structlog

from primitives.common import SystemID, new_id, utc_now
from primitives.genome import GenomeExtractionProtocol, OrganGenomeSegment, OrganismGenome
from systems.genome_helpers import verify_segment

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# Target: full organism genome extraction in <5s
_EXTRACTION_TIMEOUT_S = 10.0
_SIZE_WARNING_BYTES = 5_000_000  # 5MB total genome warning


class GenomeOrchestrator:
    """Coordinates genome extraction across all systems and assembly into OrganismGenome."""

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        instance_id: str = "",
    ) -> None:
        self._neo4j = neo4j
        self._instance_id = instance_id
        self._log = logger.bind(subsystem="genome.orchestrator")

    async def extract_full_genome(
        self,
        systems: dict[SystemID, GenomeExtractionProtocol],
        generation: int = 1,
        parent_genome_id: str | None = None,
        fitness: float = 0.0,
    ) -> OrganismGenome:
        """
        Call extract_genome_segment() on all systems in parallel,
        assemble into OrganismGenome.
        """
        start = time.monotonic()
        genome_id = new_id()
        self._log.info(
            "genome_extraction_starting",
            genome_id=genome_id,
            system_count=len(systems),
        )

        # Extract all segments in parallel with timeout
        segments: dict[SystemID, OrganGenomeSegment] = {}
        tasks: dict[SystemID, asyncio.Task[OrganGenomeSegment]] = {}

        for sys_id, system in systems.items():
            tasks[sys_id] = asyncio.create_task(
                self._extract_with_timeout(sys_id, system),
                name=f"genome_extract_{sys_id}",
            )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for sys_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self._log.error(
                    "genome_segment_extraction_failed",
                    system_id=sys_id,
                    error=str(result),
                )
                # Empty segment for failed systems
                segments[sys_id] = OrganGenomeSegment(
                    system_id=sys_id, version=0, schema_version="1.0.0"
                )
            else:
                segments[sys_id] = result

        # Compute total size
        total_size = sum(s.size_bytes for s in segments.values())
        if total_size > _SIZE_WARNING_BYTES:
            self._log.warning(
                "genome_total_size_large",
                total_bytes=total_size,
                limit=_SIZE_WARNING_BYTES,
            )

        # Count non-empty segments
        non_empty = sum(1 for s in segments.values() if s.version > 0)

        duration_ms = int((time.monotonic() - start) * 1000)
        self._log.info(
            "genome_extraction_complete",
            genome_id=genome_id,
            total_segments=len(segments),
            non_empty_segments=non_empty,
            total_bytes=total_size,
            duration_ms=duration_ms,
        )

        genome = OrganismGenome(
            id=genome_id,
            instance_id=self._instance_id,
            generation=generation,
            parent_genome_id=parent_genome_id,
            segments=segments,
            total_size_bytes=total_size,
            fitness_at_extraction=fitness,
        )

        # Persist to Neo4j if available
        if self._neo4j is not None:
            await self._persist_genome(genome)

        return genome

    async def seed_child_systems(
        self,
        genome: OrganismGenome,
        systems: dict[SystemID, GenomeExtractionProtocol],
    ) -> dict[SystemID, bool]:
        """
        Call seed_from_genome_segment() on all systems with their
        respective segments.
        """
        start = time.monotonic()
        self._log.info(
            "genome_seeding_starting",
            genome_id=genome.id,
            system_count=len(systems),
        )

        results: dict[SystemID, bool] = {}
        tasks: dict[SystemID, asyncio.Task[bool]] = {}

        for sys_id, system in systems.items():
            segment = genome.segments.get(sys_id)
            if segment is None:
                self._log.debug("genome_segment_not_found_for_system", system_id=sys_id)
                results[sys_id] = True  # No segment = nothing to seed
                continue

            tasks[sys_id] = asyncio.create_task(
                self._seed_with_timeout(sys_id, system, segment),
                name=f"genome_seed_{sys_id}",
            )

        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for sys_id, result in zip(tasks.keys(), task_results):
            if isinstance(result, Exception):
                self._log.error(
                    "genome_segment_seeding_failed",
                    system_id=sys_id,
                    error=str(result),
                )
                results[sys_id] = False
            else:
                results[sys_id] = result

        succeeded = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        duration_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "genome_seeding_complete",
            genome_id=genome.id,
            succeeded=succeeded,
            failed=failed,
            duration_ms=duration_ms,
        )

        return results

    async def load_genome(self, genome_id: str) -> OrganismGenome | None:
        """Load a persisted OrganismGenome from Neo4j by ID."""
        if self._neo4j is None:
            self._log.error("genome_load_no_neo4j")
            return None

        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (g:OrganismGenome {id: $genome_id})
                RETURN g.payload_json AS payload_json
                """,
                {"genome_id": genome_id},
            )
            if not records:
                self._log.warning("genome_not_found", genome_id=genome_id)
                return None

            payload_json = str(records[0].get("payload_json", ""))
            data = json.loads(payload_json)
            return OrganismGenome.model_validate(data)

        except Exception as exc:
            self._log.error("genome_load_failed", genome_id=genome_id, error=str(exc))
            return None

    # ── Internal helpers ───────────────────────────────────────────

    async def _extract_with_timeout(
        self,
        sys_id: SystemID,
        system: GenomeExtractionProtocol,
    ) -> OrganGenomeSegment:
        """Extract a single system's genome segment with timeout."""
        try:
            return await asyncio.wait_for(
                system.extract_genome_segment(),
                timeout=_EXTRACTION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self._log.error(
                "genome_segment_extraction_timeout",
                system_id=sys_id,
                timeout_s=_EXTRACTION_TIMEOUT_S,
            )
            return OrganGenomeSegment(
                system_id=sys_id, version=0, schema_version="1.0.0"
            )

    async def _seed_with_timeout(
        self,
        sys_id: SystemID,
        system: GenomeExtractionProtocol,
        segment: OrganGenomeSegment,
    ) -> bool:
        """Seed a single system with timeout."""
        try:
            return await asyncio.wait_for(
                system.seed_from_genome_segment(segment),
                timeout=_EXTRACTION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self._log.error(
                "genome_segment_seeding_timeout",
                system_id=sys_id,
                timeout_s=_EXTRACTION_TIMEOUT_S,
            )
            return False

    async def _persist_genome(self, genome: OrganismGenome) -> None:
        """Persist the OrganismGenome to Neo4j."""
        if self._neo4j is None:
            return

        try:
            payload_json = genome.model_dump_json()

            await self._neo4j.execute_write(
                """
                CREATE (g:OrganismGenome {
                    id: $id,
                    instance_id: $instance_id,
                    generation: $generation,
                    parent_genome_id: $parent_genome_id,
                    total_size_bytes: $total_size_bytes,
                    fitness_at_extraction: $fitness,
                    segment_count: $segment_count,
                    payload_json: $payload_json,
                    created_at: datetime($created_at)
                })
                """,
                {
                    "id": genome.id,
                    "instance_id": genome.instance_id,
                    "generation": genome.generation,
                    "parent_genome_id": genome.parent_genome_id or "",
                    "total_size_bytes": genome.total_size_bytes,
                    "fitness": genome.fitness_at_extraction,
                    "segment_count": len(genome.segments),
                    "payload_json": payload_json,
                    "created_at": utc_now().isoformat(),
                },
            )

            # Link to instance Self node
            if genome.instance_id:
                await self._neo4j.execute_write(
                    """
                    MATCH (g:OrganismGenome {id: $genome_id})
                    MATCH (s:Self {instance_id: $instance_id})
                    CREATE (g)-[:ORGANISM_GENOME_OF]->(s)
                    """,
                    {
                        "genome_id": genome.id,
                        "instance_id": genome.instance_id,
                    },
                )

            self._log.info(
                "genome_persisted",
                genome_id=genome.id,
                size_bytes=len(payload_json),
            )

        except Exception as exc:
            self._log.error("genome_persist_failed", error=str(exc))
