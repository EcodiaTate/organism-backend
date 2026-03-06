"""
EcodiaOS — Memory Trace Primitives

Data types for the knowledge graph: Episodes, Entities, Communities,
and the MemoryTrace (a processed, stored experience).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import (
    ConsolidationLevel,
    EntityType,
    EOSBaseModel,
    Identified,
    SalienceVector,
    utc_now,
)


class Episode(Identified):
    """A discrete experience record in the knowledge graph."""

    event_time: datetime = Field(default_factory=utc_now)
    ingestion_time: datetime = Field(default_factory=utc_now)
    valid_from: datetime = Field(default_factory=utc_now)
    valid_until: datetime | None = None

    source: str = ""
    modality: str = "text"
    raw_content: str = ""
    summary: str = ""
    embedding: list[float] | None = None

    salience_composite: float = 0.0
    salience_scores: dict[str, float] = Field(default_factory=dict)

    affect_valence: float = 0.0
    affect_arousal: float = 0.0

    consolidation_level: int = ConsolidationLevel.RAW
    last_accessed: datetime = Field(default_factory=utc_now)
    access_count: int = 0
    free_energy: float = 0.0

    # Somatic marker — 19D interoceptive snapshot at encoding time (Soma §0.5)
    somatic_marker: Any | None = None
    somatic_vector: list[float] | None = None


class Entity(Identified):
    """A persistent concept in the knowledge graph."""

    name: str
    type: EntityType = EntityType.CONCEPT
    description: str = ""
    embedding: list[float] | None = None

    first_seen: datetime = Field(default_factory=utc_now)
    last_updated: datetime = Field(default_factory=utc_now)
    last_accessed: datetime = Field(default_factory=utc_now)

    salience_score: float = 0.5
    mention_count: int = 1
    confidence: float = 0.8
    is_core_identity: bool = False
    community_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Community(Identified):
    """An emergent conceptual cluster from Leiden detection."""

    level: int = 0
    summary: str = ""
    embedding: list[float] | None = None
    member_count: int = 0
    coherence_score: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)
    last_recomputed: datetime = Field(default_factory=utc_now)
    salience_score: float = 0.0


class SelfNode(EOSBaseModel):
    """Singleton node representing the EOS instance itself."""

    instance_id: str
    name: str
    born_at: datetime = Field(default_factory=utc_now)
    current_affect: dict[str, float] = Field(default_factory=dict)
    autonomy_level: int = 1
    personality_vector: list[float] = Field(default_factory=list)
    personality_json: dict[str, float] = Field(default_factory=dict)
    cycle_count: int = 0
    total_episodes: int = 0
    total_entities: int = 0
    total_communities: int = 0


class ConstitutionNode(Identified):
    """The current state of the four constitutional drives."""

    version: int = 1
    drive_coherence: float = 1.0
    drive_care: float = 1.0
    drive_growth: float = 1.0
    drive_honesty: float = 1.0
    amendments: list[dict[str, Any]] = Field(default_factory=list)
    last_amended: datetime | None = None


class MentionRelation(EOSBaseModel):
    """An Episode → Entity extraction relationship."""

    episode_id: str
    entity_id: str
    role: str = "reference"    # subject | object | context | reference
    confidence: float = 0.8
    span: str = ""             # Original text span


class SemanticRelation(EOSBaseModel):
    """An Entity → Entity semantic relationship."""

    source_entity_id: str
    target_entity_id: str
    type: str                  # works_for | located_in | caused_by | etc.
    strength: float = 0.5
    confidence: float = 0.5
    first_observed: datetime = Field(default_factory=utc_now)
    last_observed: datetime = Field(default_factory=utc_now)
    observation_count: int = 1
    temporal_valid_from: datetime = Field(default_factory=utc_now)
    temporal_valid_until: datetime | None = None
    evidence_episodes: list[str] = Field(default_factory=list)


class MemoryTrace(Identified):
    """
    The fundamental unit of stored experience.
    A processed Percept, ready for retrieval.
    """

    episode_id: str
    original_percept_id: str
    summary: str = ""
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    salience: SalienceVector = Field(default_factory=SalienceVector)
    affect_valence: float = 0.0
    affect_arousal: float = 0.0
    event_time: datetime = Field(default_factory=utc_now)
    ingestion_time: datetime = Field(default_factory=utc_now)
    consolidation_level: int = ConsolidationLevel.RAW

    # Somatic marker — 19D interoceptive snapshot at encoding time (Soma §0.5)
    somatic_marker: Any | None = None
    somatic_vector: list[float] | None = None


# ─── Retrieval Request/Response ───────────────────────────────────


class MemoryRetrievalRequest(EOSBaseModel):
    """Request for hybrid memory retrieval."""

    query_text: str | None = None
    query_embedding: list[float] | None = None
    max_results: int = 10
    salience_floor: float = 0.0
    temporal_bias: str = "recency"   # "recency" | "event_time" | "none"
    traversal_depth: int = 2
    include_communities: bool = False
    include_entities: bool = True


class RetrievalResult(EOSBaseModel):
    """A single result from hybrid retrieval."""

    node_id: str
    node_type: str = "episode"   # "episode" | "entity" | "community"
    content: str = ""
    embedding: list[float] | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    graph_score: float | None = None
    salience: float = 0.0
    salience_score: float = 0.0  # Alias used by somatic reranking
    unified_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Somatic marker vector for reranking (19D: 9 sensed + 9 errors + 1 PE)
    somatic_vector: list[float] | None = None


class MemoryRetrievalResponse(EOSBaseModel):
    """Response from hybrid memory retrieval."""

    traces: list[RetrievalResult] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    communities: list[Community] = Field(default_factory=list)
    retrieval_time_ms: int = 0
