"""
EcodiaOS - Equor Schema Additions

Additional Neo4j indexes and constraints for governance records and invariants.
Must be called after the base Memory schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


async def ensure_equor_schema(neo4j: Neo4jClient) -> None:
    """Create Equor-specific indexes and constraints. Idempotent."""

    statements = [
        # ── GovernanceRecord ──────────────────────────────────────
        "CREATE CONSTRAINT governance_record_id IF NOT EXISTS "
        "FOR (g:GovernanceRecord) REQUIRE g.id IS UNIQUE",

        "CREATE INDEX governance_record_type IF NOT EXISTS "
        "FOR (g:GovernanceRecord) ON (g.event_type)",

        "CREATE INDEX governance_record_timestamp IF NOT EXISTS "
        "FOR (g:GovernanceRecord) ON (g.timestamp)",

        "CREATE INDEX governance_record_intent IF NOT EXISTS "
        "FOR (g:GovernanceRecord) ON (g.intent_id)",

        "CREATE INDEX governance_record_verdict IF NOT EXISTS "
        "FOR (g:GovernanceRecord) ON (g.verdict)",

        # ── Invariant ────────────────────────────────────────────
        "CREATE CONSTRAINT invariant_id IF NOT EXISTS "
        "FOR (i:Invariant) REQUIRE i.id IS UNIQUE",

        "CREATE INDEX invariant_active IF NOT EXISTS "
        "FOR (i:Invariant) ON (i.active)",

        # ── Amendment Pipeline ──────────────────────────────────
        "CREATE INDEX governance_record_amendment_status IF NOT EXISTS "
        "FOR (g:GovernanceRecord) ON (g.amendment_status)",
    ]

    for stmt in statements:
        try:
            await neo4j.execute_write(stmt)
        except Exception as e:
            # Some errors are expected (e.g. constraint already exists on older Neo4j)
            if "already exists" not in str(e).lower():
                logger.warning("equor_schema_statement_warning", statement=stmt[:60], error=str(e))

    logger.info("equor_schema_ensured")


async def seed_hardcoded_invariants(neo4j: Neo4jClient) -> int:
    """
    Ensure all hardcoded invariants exist as Invariant nodes in the graph,
    linked to the Constitution via INCLUDES_INVARIANT.
    Returns the number of invariants seeded.
    """
    from systems.equor.invariants import HARDCODED_INVARIANTS

    seeded = 0
    for invariant_def, _ in HARDCODED_INVARIANTS:
        # MERGE to be idempotent
        result = await neo4j.execute_write(
            """
            MERGE (i:Invariant {id: $id})
            ON CREATE SET
                i.name = $name,
                i.description = $description,
                i.source = 'hardcoded',
                i.severity = $severity,
                i.active = true,
                i.added_at = datetime($now)
            WITH i
            MATCH (c:Constitution)
            MERGE (c)-[:INCLUDES_INVARIANT]->(i)
            RETURN i.id AS id
            """,
            {
                "id": invariant_def.id,
                "name": invariant_def.name,
                "description": invariant_def.description,
                "severity": invariant_def.severity,
                "now": invariant_def.added_at.isoformat(),
            },
        )
        if result:
            seeded += 1

    logger.info("hardcoded_invariants_seeded", count=seeded)
    return seeded
