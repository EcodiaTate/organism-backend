"""
EcodiaOS - Simula ArXiv Paper Memory

Writes arXiv papers and their extracted techniques into the Neo4j knowledge
graph so that downstream consumers (e.g. SimulaCodeAgent) can retrieve the
source theory before writing code.

Graph schema added here:
  (p:Entity {type: "paper"})  - PaperNode carrying abstract + URL
  (c:Entity {type: "concept"}) - ConceptNode for the extracted technique
  (p)-[:RELATES_TO {type: "source_of"}]->(c)

All operations are async and raise no exceptions - callers receive
``None`` on failure so the proposal-dispatch path degrades gracefully.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives import Entity, EntityType, SemanticRelation, new_id, utc_now
from systems.memory.semantic import (
    create_entity,
    create_or_strengthen_relation,
    find_similar_entity,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger().bind(module="simula.paper_memory")


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


async def upsert_paper_and_concept(
    neo4j: Neo4jClient,
    *,
    paper_id: str,
    title: str,
    abstract: str,
    arxiv_url: str,
    technique_name: str,
) -> tuple[str, str] | None:
    """
    Ensure a PaperNode and ConceptNode exist in the graph, linked by a
    ``source_of`` SemanticRelation.

    Returns ``(paper_entity_id, concept_entity_id)`` on success, ``None``
    on any error (graceful degradation - caller must not crash on None).

    Deduplication: uses the existing ``find_similar_entity`` name-match so
    the same paper is never written twice (embedding-free, fast path).
    """
    try:
        paper_entity_id = await _upsert_paper_node(
            neo4j,
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            arxiv_url=arxiv_url,
        )
        concept_entity_id = await _upsert_concept_node(
            neo4j,
            technique_name=technique_name,
            paper_id=paper_id,
        )
        await _link_paper_to_concept(neo4j, paper_entity_id, concept_entity_id)

        logger.info(
            "paper_memory_written",
            paper_id=paper_id,
            paper_entity_id=paper_entity_id,
            concept_entity_id=concept_entity_id,
            technique=technique_name,
        )
        return (paper_entity_id, concept_entity_id)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "paper_memory_write_failed",
            paper_id=paper_id,
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


async def get_paper_abstract_for_technique(
    neo4j: Neo4jClient,
    technique_name: str,
) -> str | None:
    """
    Given a technique name, find the linked PaperNode and return its abstract.

    Query path:
      MATCH (c:Entity {type: "concept"}) WHERE c.name contains technique_name
      MATCH (p:Entity)-[:RELATES_TO {type: "source_of"}]->(c)
      RETURN p.metadata_json  (which contains the abstract)

    Returns ``None`` if nothing is found or on any error.
    """
    try:
        results = await neo4j.execute_read(
            """
            MATCH (c:Entity)
            WHERE c.type = $concept_type
              AND toLower(c.name) CONTAINS toLower($technique)
            MATCH (p:Entity)-[r:RELATES_TO]->(c)
            WHERE r.type = $rel_type
              AND p.type = $paper_type
            RETURN p.metadata_json AS meta
            LIMIT 1
            """,
            {
                "concept_type": "concept",
                "technique": technique_name,
                "rel_type": "source_of",
                "paper_type": "paper",
            },
        )
        if not results:
            return None

        import json as _json
        meta_raw = results[0].get("meta")
        if not meta_raw:
            return None
        meta: dict[str, Any] = _json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        return meta.get("abstract") or None

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "paper_memory_read_failed",
            technique=technique_name,
            error=str(exc),
        )
        return None


async def get_paper_by_arxiv_id(
    neo4j: Neo4jClient,
    paper_id: str,
) -> dict[str, Any] | None:
    """
    Retrieve a PaperNode entity dict by arXiv paper ID (stored in metadata).

    Returns the raw entity dict or ``None``.
    """
    try:
        results = await neo4j.execute_read(
            """
            MATCH (p:Entity)
            WHERE p.type = $paper_type
              AND p.name = $name
            RETURN p
            LIMIT 1
            """,
            {
                "paper_type": "paper",
                "name": _paper_node_name(paper_id),
            },
        )
        return results[0]["p"] if results else None

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "paper_memory_lookup_failed",
            paper_id=paper_id,
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _paper_node_name(paper_id: str) -> str:
    """Canonical name for a PaperNode - stable across duplicate runs."""
    return f"arxiv:{paper_id}"


async def _upsert_paper_node(
    neo4j: Neo4jClient,
    *,
    paper_id: str,
    title: str,
    abstract: str,
    arxiv_url: str,
) -> str:
    """Create or reuse the PaperNode entity. Returns its entity ID."""
    name = _paper_node_name(paper_id)
    existing = await find_similar_entity(neo4j, name=name, embedding=[])
    if existing:
        return str(existing["id"])

    entity = Entity(
        id=new_id(),
        name=name,
        type=EntityType.CONCEPT,          # closest fit in existing enum
        description=f"{title} - arXiv:{paper_id}",
        salience_score=0.7,
        confidence=0.95,
        metadata={
            "arxiv_id": paper_id,
            "title": title,
            "abstract": abstract,
            "arxiv_url": arxiv_url,
            "node_kind": "paper",           # discriminator within CONCEPT type
        },
    )
    # Patch type string directly so the graph stores "paper" even though
    # EntityType has no PAPER variant - stored as the raw string value.
    await neo4j.execute_write(
        """
        CREATE (e:Entity {
            id: $id,
            name: $name,
            type: $type,
            description: $description,
            embedding: null,
            first_seen: datetime($now),
            last_updated: datetime($now),
            last_accessed: datetime($now),
            salience_score: $salience,
            mention_count: 1,
            confidence: $confidence,
            is_core_identity: false,
            community_ids: [],
            metadata_json: $metadata_json
        })
        """,
        {
            "id": entity.id,
            "name": entity.name,
            "type": "paper",
            "description": entity.description,
            "now": utc_now().isoformat(),
            "salience": entity.salience_score,
            "confidence": entity.confidence,
            "metadata_json": _json_dumps(entity.metadata),
        },
    )
    return entity.id


async def _upsert_concept_node(
    neo4j: Neo4jClient,
    *,
    technique_name: str,
    paper_id: str,
) -> str:
    """Create or reuse the ConceptNode entity. Returns its entity ID."""
    existing = await find_similar_entity(neo4j, name=technique_name, embedding=[])
    if existing:
        return str(existing["id"])

    entity = Entity(
        id=new_id(),
        name=technique_name,
        type=EntityType.CONCEPT,
        description=f"Technique from arXiv paper {paper_id}",
        salience_score=0.6,
        confidence=0.8,
        metadata={"source_paper": paper_id, "node_kind": "concept"},
    )
    await create_entity(neo4j, entity)
    return entity.id


async def _link_paper_to_concept(
    neo4j: Neo4jClient,
    paper_entity_id: str,
    concept_entity_id: str,
) -> None:
    """Create or strengthen the source_of edge from PaperNode → ConceptNode."""
    relation = SemanticRelation(
        source_entity_id=paper_entity_id,
        target_entity_id=concept_entity_id,
        type="source_of",
        strength=0.9,
        confidence=0.9,
        evidence_episodes=[],
    )
    await create_or_strengthen_relation(neo4j, relation)


def _json_dumps(obj: dict[str, Any]) -> str:
    import json
    return json.dumps(obj)
