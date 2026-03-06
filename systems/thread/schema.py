"""
EcodiaOS — Thread Neo4j Schema Extension

Extends Memory's knowledge graph with narrative structure:
6 new node labels, 15+ relationship types, vector indexes for
narrative embedding search.

This layer sits ON TOP of episodes/entities/communities — it does not
replace them. Episodes remain the atomic experience unit; NarrativeGraph
adds the interpretive structure that transforms episodes into autobiography.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# ─── Constraints (uniqueness) ────────────────────────────────────
CONSTRAINTS = [
    "CREATE CONSTRAINT thread_chapter_id IF NOT EXISTS"
    " FOR (c:NarrativeChapter) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT thread_scene_id IF NOT EXISTS"
    " FOR (s:NarrativeScene) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT thread_turning_point_id IF NOT EXISTS"
    " FOR (t:TurningPoint) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT thread_identity_schema_id IF NOT EXISTS"
    " FOR (s:IdentitySchema) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT thread_commitment_id IF NOT EXISTS"
    " FOR (c:Commitment) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT thread_fingerprint_id IF NOT EXISTS"
    " FOR (f:BehavioralFingerprint) REQUIRE f.id IS UNIQUE",
]

# ─── Indexes (performance) ───────────────────────────────────────
INDEXES = [
    # Chapter queries
    "CREATE INDEX thread_chapter_status IF NOT EXISTS FOR (c:NarrativeChapter) ON (c.status)",
    "CREATE INDEX thread_chapter_started IF NOT EXISTS FOR (c:NarrativeChapter) ON (c.started_at)",

    # Schema lookups
    "CREATE INDEX thread_schema_strength IF NOT EXISTS FOR (s:IdentitySchema) ON (s.strength)",
    "CREATE INDEX thread_schema_valence IF NOT EXISTS FOR (s:IdentitySchema) ON (s.valence)",

    # Commitment lookups
    "CREATE INDEX thread_commitment_status IF NOT EXISTS FOR (c:Commitment) ON (c.status)",

    # Fingerprint temporal ordering
    "CREATE INDEX thread_fingerprint_epoch IF NOT EXISTS"
    " FOR (f:BehavioralFingerprint) ON (f.epoch_label)",
    "CREATE INDEX thread_fingerprint_window_start IF NOT EXISTS"
    " FOR (f:BehavioralFingerprint) ON (f.window_start)",

    # Scene ordering
    "CREATE INDEX thread_scene_started IF NOT EXISTS"
    " FOR (s:NarrativeScene) ON (s.started_at)",

    # Turning point lookups
    "CREATE INDEX thread_turning_point_timestamp IF NOT EXISTS"
    " FOR (t:TurningPoint) ON (t.timestamp)",
]

# ─── Vector Indexes ──────────────────────────────────────────────
VECTOR_INDEXES = [
    """
    CREATE VECTOR INDEX thread_chapter_embedding IF NOT EXISTS
    FOR (c:NarrativeChapter) ON (c.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,
    """
    CREATE VECTOR INDEX thread_schema_embedding IF NOT EXISTS
    FOR (s:IdentitySchema) ON (s.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,
    """
    CREATE VECTOR INDEX thread_turning_point_embedding IF NOT EXISTS
    FOR (t:TurningPoint) ON (t.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,
    """
    CREATE VECTOR INDEX thread_commitment_embedding IF NOT EXISTS
    FOR (c:Commitment) ON (c.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,
]


async def ensure_thread_schema(neo4j: Neo4jClient) -> None:
    """
    Create all Thread indexes and constraints if they don't exist.
    Idempotent — safe to call on every startup.
    """
    logger.info("thread_schema_ensuring")

    all_statements = CONSTRAINTS + INDEXES + VECTOR_INDEXES

    for statement in all_statements:
        statement = statement.strip()
        if not statement:
            continue
        try:
            await neo4j.execute_write(statement)
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "equivalent" in error_msg:
                continue
            logger.warning(
                "thread_schema_statement_warning",
                statement=statement[:80],
                error=str(e),
            )

    logger.info("thread_schema_ensured")
