"""
EcodiaOS — Memory Schema

Creates all Neo4j indexes, constraints, and vector indexes on first boot.
This is the physical structure of the knowledge graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# ─── Constraints (uniqueness) ────────────────────────────────────
CONSTRAINTS = [
    "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT self_id IF NOT EXISTS FOR (s:Self) REQUIRE s.instance_id IS UNIQUE",
    "CREATE CONSTRAINT constitution_id IF NOT EXISTS FOR (c:Constitution) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT procedure_id IF NOT EXISTS FOR (p:Procedure) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT hypothesis_id IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE",
    "CREATE CONSTRAINT governance_id IF NOT EXISTS FOR (g:GovernanceRecord) REQUIRE g.id IS UNIQUE",
    "CREATE CONSTRAINT projection_basis_id IF NOT EXISTS FOR (pb:ProjectionBasis) REQUIRE pb.id IS UNIQUE",
    "CREATE CONSTRAINT consolidated_belief_id IF NOT EXISTS FOR (cb:ConsolidatedBelief) REQUIRE cb.id IS UNIQUE",
    "CREATE CONSTRAINT tournament_id IF NOT EXISTS FOR (t:HypothesisTournament) REQUIRE t.tournament_id IS UNIQUE",
]

# ─── Indexes (performance) ───────────────────────────────────────
INDEXES = [
    # Temporal queries
    "CREATE INDEX episode_event_time IF NOT EXISTS FOR (e:Episode) ON (e.event_time)",
    "CREATE INDEX episode_ingestion_time IF NOT EXISTS FOR (e:Episode) ON (e.ingestion_time)",
    "CREATE INDEX episode_salience IF NOT EXISTS FOR (e:Episode) ON (e.salience_composite)",

    # Entity lookups
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_salience IF NOT EXISTS FOR (e:Entity) ON (e.salience_score)",
    "CREATE INDEX entity_core IF NOT EXISTS FOR (e:Entity) ON (e.is_core_identity)",

    # Community hierarchy
    "CREATE INDEX community_level IF NOT EXISTS FOR (c:Community) ON (c.level)",

    # Consolidation
    "CREATE INDEX episode_consolidation IF NOT EXISTS FOR (e:Episode) ON (e.consolidation_level)",

    # Belief half-life — accelerate aging scans and staleness queries
    "CREATE INDEX belief_halflife IF NOT EXISTS FOR (b:Belief) ON (b.half_life_days)",
    "CREATE INDEX belief_last_verified IF NOT EXISTS FOR (b:Belief) ON (b.last_verified)",
    "CREATE INDEX belief_domain IF NOT EXISTS FOR (b:Belief) ON (b.domain)",

    # Counterfactual episodes — accelerate regret resolution and Evo mining
    "CREATE INDEX counterfactual_intent IF NOT EXISTS FOR (e:Counterfactual) ON (e.intent_id)",
    "CREATE INDEX counterfactual_resolved IF NOT EXISTS FOR (e:Counterfactual) ON (e.resolved)",
    "CREATE INDEX counterfactual_policy_type IF NOT EXISTS FOR (e:Counterfactual) ON (e.policy_type)",

    # Semantic compression — lookup compressed episodes by basis
    "CREATE INDEX episode_compression_basis IF NOT EXISTS FOR (e:Episode) ON (e.compression_basis_id)",
    "CREATE INDEX projection_basis_community IF NOT EXISTS FOR (pb:ProjectionBasis) ON (pb.community_id)",

    # Consolidated beliefs — hardened high-confidence knowledge (Evo Phase 2.75)
    "CREATE INDEX consolidated_belief_domain IF NOT EXISTS FOR (cb:ConsolidatedBelief) ON (cb.domain)",
    "CREATE INDEX consolidated_belief_consolidated_at IF NOT EXISTS FOR (cb:ConsolidatedBelief) ON (cb.consolidated_at)",

    # Hypothesis tournaments — Thompson sampling A/B experiments (Evo Phase 2)
    "CREATE INDEX tournament_stage IF NOT EXISTS FOR (t:HypothesisTournament) ON (t.stage)",
    "CREATE INDEX tournament_winner IF NOT EXISTS FOR (t:HypothesisTournament) ON (t.winner_id)",
]

# ─── Fulltext Indexes ────────────────────────────────────────────
FULLTEXT_INDEXES = [
    """
    CREATE FULLTEXT INDEX episode_content IF NOT EXISTS
    FOR (e:Episode) ON EACH [e.summary, e.raw_content]
    """,
    """
    CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
    FOR (e:Entity) ON EACH [e.name, e.description]
    """,
]

# ─── Vector Indexes ──────────────────────────────────────────────
# Neo4j 5.x native vector index for semantic search
VECTOR_INDEXES = [
    """
    CREATE VECTOR INDEX episode_embedding IF NOT EXISTS
    FOR (e:Episode) ON (e.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """,
    """
    CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
    FOR (e:Entity) ON (e.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """,
    """
    CREATE VECTOR INDEX community_embedding IF NOT EXISTS
    FOR (c:Community) ON (c.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """,
    # Somatic marker vector index — 19D (9 sensed + 9 errors + 1 PE)
    # for state-congruent recall reranking (Soma §0.5)
    """
    CREATE VECTOR INDEX episode_somatic IF NOT EXISTS
    FOR (e:Episode) ON (e.somatic_vector)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 19,
        `vector.similarity_function`: 'cosine'
    }}
    """,
]


async def ensure_schema(neo4j: Neo4jClient) -> None:
    """
    Create all indexes and constraints if they don't exist.
    Idempotent — safe to call on every startup.
    """
    logger.info("memory_schema_ensuring")

    all_statements = CONSTRAINTS + INDEXES + FULLTEXT_INDEXES + VECTOR_INDEXES

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
                "memory_schema_statement_warning",
                statement=statement[:80],
                error=str(e),
            )

    logger.info("memory_schema_ensured")
