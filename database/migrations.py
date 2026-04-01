"""
Database Migrations for EcodiaOS

Soma Follow-Up Work (System 15): TimescaleDB and Neo4j migrations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asyncpg import Connection
    from neo4j import Session

logger = logging.getLogger(__name__)


async def migrate_timescaledb_interoceptive_state(conn: Connection) -> None:
    """
    Create interoceptive_state hypertable for Soma telemetry.

    Stores every cycle's 9D sensed state, allostatic errors, urgency signal,
    and phase space position for retrospective analysis and telemetry.
    """
    # 1. Table + indexes - plain Postgres.
    table_sql = """
    CREATE TABLE IF NOT EXISTS interoceptive_state (
        time             TIMESTAMPTZ NOT NULL,
        tenant_id        UUID NOT NULL,
        cycle_number     BIGINT,

        -- 9D interoceptive state (sensed)
        energy           FLOAT NOT NULL,
        arousal          FLOAT NOT NULL,
        valence          FLOAT NOT NULL,
        confidence       FLOAT NOT NULL,
        coherence        FLOAT NOT NULL,
        social_charge    FLOAT NOT NULL,
        curiosity_drive  FLOAT NOT NULL,
        integrity        FLOAT NOT NULL,
        temporal_pressure FLOAT NOT NULL,

        -- 9D allostatic errors (moment horizon)
        energy_error           FLOAT,
        arousal_error          FLOAT,
        valence_error          FLOAT,
        confidence_error       FLOAT,
        coherence_error        FLOAT,
        social_charge_error    FLOAT,
        curiosity_drive_error  FLOAT,
        integrity_error        FLOAT,
        temporal_pressure_error FLOAT,

        -- Urgency signal
        urgency              FLOAT,
        dominant_error_dim   TEXT,
        dominant_error_val   FLOAT,

        -- Phase space
        nearest_attractor    TEXT,
        distance_to_bifurcation FLOAT,
        stage                TEXT,

        PRIMARY KEY (time, tenant_id)
    );

    CREATE INDEX IF NOT EXISTS idx_interoceptive_tenant_time
        ON interoceptive_state (tenant_id, time DESC);

    CREATE INDEX IF NOT EXISTS idx_interoceptive_urgency
        ON interoceptive_state (tenant_id, urgency DESC)
        WHERE urgency > 0.7;

    CREATE INDEX IF NOT EXISTS idx_interoceptive_attractor
        ON interoceptive_state (tenant_id, nearest_attractor)
        WHERE nearest_attractor IS NOT NULL;
    """

    try:
        await conn.execute(table_sql)
    except Exception as exc:
        logger.error(f"interoceptive_state table creation failed: {exc}")
        raise

    # 2. Hypertable + policies - only if TimescaleDB extension is present.
    has_timescaledb = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
    )

    if has_timescaledb:
        try:
            await conn.execute(
                "SELECT create_hypertable('interoceptive_state', 'time', if_not_exists => TRUE)"
            )
            await conn.execute(
                "SELECT add_compression_policy('interoceptive_state', INTERVAL '7 days', if_not_exists => TRUE)"
            )
            await conn.execute(
                "SELECT add_retention_policy('interoceptive_state', INTERVAL '90 days', if_not_exists => TRUE)"
            )
            logger.info("interoceptive_state hypertable created with policies")
        except Exception as exc:
            logger.warning(f"interoceptive_state hypertable setup failed: {exc}")
    else:
        logger.info(
            "interoceptive_state created as plain table - timescaledb extension not available"
        )


def migrate_neo4j_somatic_vector_index(session: Session) -> None:
    """
    Create vector index on SomaticMarker nodes for somatic reranking at graph scale.

    The 19D vector represents [9 sensed dimensions] + [9 error dimensions] + [1 PE].
    Enables state-congruent memory retrieval across the Neo4j knowledge graph.
    """
    # Check if index already exists
    result = session.run(
        """
        SHOW INDEXES
        WHERE name = 'somatic_marker_idx'
        """
    )
    if result.single():
        logger.info("Vector index 'somatic_marker_idx' already exists")
        return

    # Create the vector index
    session.run(
        """
        CREATE VECTOR INDEX somatic_marker_idx
        FOR (m:SomaticMarker)
        ON m.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 19,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
    )

    # Create uniqueness constraint if not exists
    session.run(
        """
        CREATE CONSTRAINT somatic_marker_id IF NOT EXISTS
        FOR (m:SomaticMarker) REQUIRE m.id IS UNIQUE
        """
    )

    logger.info("SomaticMarker vector index created successfully")


def migrate_neo4j_adversarial_proposal_schema(session: Session) -> None:
    """
    Create constraints and singleton nodes for the adversarial proposal graph.

    Schema:
      (:RedTeamInstance {instance_id})  -[:GENERATED {at}]->
      (:EvolutionProposal {id, ...})    -[:TARGETS]->
      (:Constitution {name})

    Idempotent - safe to run on every startup.
    """
    constraints = [
        (
            "evolution_proposal_id",
            "CREATE CONSTRAINT evolution_proposal_id IF NOT EXISTS "
            "FOR (p:EvolutionProposal) REQUIRE p.id IS UNIQUE",
        ),
        (
            "red_team_instance_id",
            "CREATE CONSTRAINT red_team_instance_id IF NOT EXISTS "
            "FOR (r:RedTeamInstance) REQUIRE r.instance_id IS UNIQUE",
        ),
        (
            "constitution_name",
            "CREATE CONSTRAINT constitution_name IF NOT EXISTS "
            "FOR (c:Constitution) REQUIRE c.name IS UNIQUE",
        ),
    ]

    for name, cypher in constraints:
        try:
            session.run(cypher)
            logger.info("Constraint '%s' ensured", name)
        except Exception as exc:
            logger.warning("Constraint '%s' creation failed: %s", name, exc)

    # Ensure singleton Constitution node
    session.run(
        "MERGE (:Constitution {name: $name})",
        name="EcodiaOS",
    )
    logger.info("Constitution singleton node ensured")


async def run_all_migrations(
    timescaledb_conn: Connection | None = None,
    neo4j_session: Session | None = None,
) -> None:
    """
    Run all Soma follow-up migrations.

    Args:
        timescaledb_conn: asyncpg Connection to TimescaleDB
        neo4j_session: neo4j Session to Neo4j
    """
    if timescaledb_conn is not None:
        await migrate_timescaledb_interoceptive_state(timescaledb_conn)

    if neo4j_session is not None:
        migrate_neo4j_somatic_vector_index(neo4j_session)
        migrate_neo4j_adversarial_proposal_schema(neo4j_session)
