"""
EcodiaOS - Organism Genome Primitives

Universal genome protocol for Mitosis inheritance. Every system implements
GenomeExtractionProtocol so the organism can snapshot its complete heritable
state and seed new child instances.

Previously only Simula and Evo had genome concepts - this makes the genome
organism-wide, giving Mitosis a single interface to extract and inject
genetic material across all 29 systems.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import Field

from primitives.common import (
    EOSBaseModel,
    Identified,
    SystemID,
    Timestamped,
    new_id,
    utc_now,
)


class OrganGenomeSegment(EOSBaseModel):
    """
    One system's heritable state - the genetic material for a single organ.

    The payload is opaque to Mitosis: each system serialises its own
    heritable configuration into a dict. The payload_hash ensures integrity
    during transfer to child instances.
    """

    system_id: SystemID
    version: int = 1
    schema_version: str = "1.0"
    payload: dict = Field(default_factory=dict)
    payload_hash: str = ""
    size_bytes: int = 0
    extracted_at: datetime = Field(default_factory=utc_now)


class OrganismGenome(Identified, Timestamped):
    """
    The complete genome across all organs - a full snapshot of the organism's
    heritable state at a point in time.

    Used by Mitosis to birth child instances and by Benchmarks to track
    evolutionary lineage across generations.
    """

    instance_id: str = ""
    generation: int = 1
    parent_genome_id: str | None = None
    segments: dict[SystemID, OrganGenomeSegment] = Field(default_factory=dict)
    total_size_bytes: int = 0
    fitness_at_extraction: float = 0.0

    def segment_ids(self) -> list[SystemID]:
        """Return the system IDs of all segments in this genome."""
        return list(self.segments.keys())


@runtime_checkable
class GenomeExtractionProtocol(Protocol):
    """
    The interface every system must implement for genome participation.

    extract_genome_segment: serialise the system's heritable state.
    seed_from_genome_segment: restore heritable state from a parent's segment.
    """

    async def extract_genome_segment(self) -> OrganGenomeSegment: ...

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool: ...
