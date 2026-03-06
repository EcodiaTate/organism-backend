"""
EcodiaOS — Semantic Memory

Entity extraction, deduplication, and relationship management.
This is the "what things exist" layer of memory.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives import (
    Entity,
    EntityType,
    MentionRelation,
    SemanticRelation,
    utc_now,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# Cosine similarity threshold for entity deduplication
DEDUP_THRESHOLD = 0.88


async def create_entity(
    neo4j: Neo4jClient,
    entity: Entity,
) -> str:
    """Create a new entity node in the graph."""
    await neo4j.execute_write(
        """
        CREATE (e:Entity {
            id: $id,
            name: $name,
            type: $type,
            description: $description,
            embedding: $embedding,
            first_seen: datetime($first_seen),
            last_updated: datetime($last_updated),
            last_accessed: datetime($last_accessed),
            salience_score: $salience_score,
            mention_count: $mention_count,
            confidence: $confidence,
            is_core_identity: $is_core,
            community_ids: $community_ids,
            metadata_json: $metadata_json
        })
        """,
        {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type.value if isinstance(entity.type, EntityType) else entity.type,
            "description": entity.description,
            "embedding": entity.embedding,
            "first_seen": entity.first_seen.isoformat(),
            "last_updated": entity.last_updated.isoformat(),
            "last_accessed": entity.last_accessed.isoformat(),
            "salience_score": entity.salience_score,
            "mention_count": entity.mention_count,
            "confidence": entity.confidence,
            "is_core": entity.is_core_identity,
            "community_ids": entity.community_ids,
            "metadata_json": json.dumps(entity.metadata),
        },
    )

    # Increment Self counter
    await neo4j.execute_write(
        "MATCH (s:Self) SET s.total_entities = s.total_entities + 1"
    )

    logger.debug("entity_created", entity_id=entity.id, name=entity.name, type=entity.type)
    return entity.id


async def find_similar_entity(
    neo4j: Neo4jClient,
    name: str,
    embedding: list[float],
    threshold: float = DEDUP_THRESHOLD,
) -> dict[str, Any] | None:
    """
    Find an existing entity that matches by name or embedding similarity.
    Used for deduplication during entity extraction.
    """
    # First: exact name match (fast)
    results = await neo4j.execute_read(
        """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($name)
        RETURN e
        LIMIT 1
        """,
        {"name": name},
    )
    if results:
        return results[0]["e"]  # type: ignore[no-any-return]

    # Second: vector similarity match (semantic dedup)
    if embedding:
        results = await neo4j.execute_read(
            """
            CALL db.index.vector.queryNodes('entity_embedding', 3, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node AS e, score
            ORDER BY score DESC
            LIMIT 1
            """,
            {"embedding": embedding, "threshold": threshold},
        )
        if results:
            return results[0]["e"]  # type: ignore[no-any-return]

    return None


async def merge_into_entity(
    neo4j: Neo4jClient,
    existing_entity_id: str,
    new_description: str,
) -> None:
    """Merge new information into an existing entity (update description, bump counts)."""
    now = utc_now()
    await neo4j.execute_write(
        """
        MATCH (e:Entity {id: $id})
        SET e.description = $description,
            e.last_updated = datetime($now),
            e.last_accessed = datetime($now),
            e.mention_count = e.mention_count + 1
        """,
        {
            "id": existing_entity_id,
            "description": new_description,
            "now": now.isoformat(),
        },
    )


async def link_episode_to_entity(
    neo4j: Neo4jClient,
    mention: MentionRelation,
) -> None:
    """Create a MENTIONS relationship from Episode to Entity."""
    await neo4j.execute_write(
        """
        MATCH (ep:Episode {id: $episode_id})
        MATCH (en:Entity {id: $entity_id})
        CREATE (ep)-[:MENTIONS {
            role: $role,
            confidence: $confidence,
            span: $span
        }]->(en)
        """,
        {
            "episode_id": mention.episode_id,
            "entity_id": mention.entity_id,
            "role": mention.role,
            "confidence": mention.confidence,
            "span": mention.span,
        },
    )


async def create_or_strengthen_relation(
    neo4j: Neo4jClient,
    relation: SemanticRelation,
) -> None:
    """Create a new RELATES_TO edge or strengthen an existing one."""
    now = utc_now()
    await neo4j.execute_write(
        """
        MATCH (a:Entity {id: $source_id})
        MATCH (b:Entity {id: $target_id})
        MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
        ON CREATE SET
            r.strength = $strength,
            r.confidence = $confidence,
            r.first_observed = datetime($now),
            r.last_observed = datetime($now),
            r.observation_count = 1,
            r.temporal_valid_from = datetime($now),
            r.temporal_valid_until = null,
            r.evidence_episodes = $evidence
        ON MATCH SET
            r.strength = CASE WHEN $strength > r.strength THEN $strength ELSE r.strength END,
            r.confidence = (r.confidence * r.observation_count + $confidence) / (r.observation_count + 1),
            r.last_observed = datetime($now),
            r.observation_count = r.observation_count + 1,
            r.evidence_episodes = r.evidence_episodes + $evidence
        """,
        {
            "source_id": relation.source_entity_id,
            "target_id": relation.target_entity_id,
            "type": relation.type,
            "strength": relation.strength,
            "confidence": relation.confidence,
            "now": now.isoformat(),
            "evidence": relation.evidence_episodes,
        },
    )


async def get_entity(neo4j: Neo4jClient, entity_id: str) -> dict[str, Any] | None:
    """Retrieve a single entity by ID."""
    results = await neo4j.execute_read(
        "MATCH (e:Entity {id: $id}) RETURN e",
        {"id": entity_id},
    )
    return results[0]["e"] if results else None


async def get_entity_neighbours(
    neo4j: Neo4jClient,
    entity_id: str,
    max_depth: int = 2,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get entities connected to a given entity within N hops."""
    return await neo4j.execute_read(
        """
        MATCH (start:Entity {id: $id})
        MATCH path = (start)-[:RELATES_TO*1..$depth]-(neighbor:Entity)
        WHERE neighbor.id <> $id
        RETURN DISTINCT neighbor AS e,
               length(path) AS distance,
               [r IN relationships(path) | r.strength] AS strengths
        ORDER BY distance ASC, neighbor.salience_score DESC
        LIMIT $limit
        """,
        {"id": entity_id, "depth": max_depth, "limit": limit},
    )


async def count_entities(neo4j: Neo4jClient) -> int:
    """Get total entity count."""
    results = await neo4j.execute_read("MATCH (e:Entity) RETURN count(e) AS cnt")
    return results[0]["cnt"] if results else 0
