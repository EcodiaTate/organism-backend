"""
EcodiaOS - Simula Genome Protocol Adapter

Wraps the existing SimulaGenomeExtractor/SimulaGenomeSeeder in the new
organism-wide GenomeExtractionProtocol interface. Does NOT rewrite the
existing genome code - just bridges between OrganGenomeSegment and
SimulaGenome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


class SimulaGenomeAdapter:
    """
    Adapts the existing SimulaGenomeExtractor/SimulaGenomeSeeder to
    GenomeExtractionProtocol.

    extract_genome_segment: calls SimulaGenomeExtractor.extract_genome()
    and wraps the result in an OrganGenomeSegment.

    seed_from_genome_segment: unwraps the OrganGenomeSegment payload and
    feeds it to SimulaGenomeSeeder.seed_from_genome().
    """

    def __init__(self, neo4j: Neo4jClient, instance_id: str) -> None:
        self._neo4j = neo4j
        self._instance_id = instance_id
        self._log = logger.bind(subsystem="simula.genome.adapter")

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """Extract SimulaGenome via existing extractor, wrap in OrganGenomeSegment."""
        try:
            from systems.simula.genome import SimulaGenomeExtractor

            extractor = SimulaGenomeExtractor(
                neo4j=self._neo4j,
                instance_id=self._instance_id,
            )

            if not await extractor.is_eligible():
                self._log.info("simula_genome_not_eligible")
                return build_segment(SystemID.SIMULA, {}, version=0)

            genome, result = await extractor.extract_genome()
            if genome is None:
                self._log.info("simula_genome_extraction_empty")
                return build_segment(SystemID.SIMULA, {}, version=0)

            # Wrap the full SimulaGenome as payload
            payload = genome.model_dump(mode="json")

            self._log.info(
                "simula_genome_segment_extracted",
                genome_id=result.genome_id,
                mutations=result.mutations_included,
                abstractions=result.abstractions_included,
                training=result.training_examples_included,
                size_bytes=result.genome_size_bytes,
            )

            return build_segment(SystemID.SIMULA, payload, version=1)

        except Exception as exc:
            self._log.error("simula_genome_segment_extract_failed", error=str(exc))
            return build_segment(SystemID.SIMULA, {}, version=0)

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """Unwrap OrganGenomeSegment and seed via existing SimulaGenomeSeeder."""
        if segment.version == 0 or not segment.payload:
            self._log.info("simula_genome_segment_empty_skip")
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            from systems.simula.genome import SimulaGenomeSeeder
            from systems.simula.genome_types import SimulaGenome

            # Reconstruct SimulaGenome from the payload dict
            genome = SimulaGenome.model_validate(segment.payload)

            seeder = SimulaGenomeSeeder(
                neo4j=self._neo4j,
                child_instance_id=self._instance_id,
            )
            result = await seeder.seed_from_genome(genome)

            self._log.info(
                "simula_genome_segment_seeded",
                parent_genome_id=result.parent_genome_id,
                mutations=result.mutations_seeded,
                abstractions=result.abstractions_seeded,
                training=result.training_examples_seeded,
                duration_ms=result.duration_ms,
            )
            return True

        except Exception as exc:
            self._log.error("simula_genome_segment_seed_failed", error=str(exc))
            return False
