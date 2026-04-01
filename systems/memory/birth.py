"""
EcodiaOS - Instance Birth (Seeding)

Creates the foundational graph structure for a new EOS instance:
the Self node, the Constitution, and any initial entities from the seed.

Supports genetic memory inheritance: when a parent_genome is provided,
the child instance is seeded with the parent's stable beliefs, enabling
faster learning through compressed belief inheritance.

This is a one-time operation. Once born, the instance exists.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives import AffectState, new_id, utc_now

if TYPE_CHECKING:
    from clients.embedding import EmbeddingClient
    from clients.neo4j import Neo4jClient
    from config import SeedConfig

logger = structlog.get_logger()


async def birth_instance(
    neo4j: Neo4jClient,
    embedding_client: EmbeddingClient,
    seed: SeedConfig,
    instance_id: str,
) -> dict[str, Any]:
    """
    Birth a new EOS instance from a seed configuration.

    Creates:
    1. The Self node (singleton identity anchor)
    2. The Constitution node (the four drives)
    3. Initial entities from the seed's community config
    4. Relationships linking them all together

    Returns the Self node data.
    """
    logger.info("instance_birth_starting", name=seed.instance.name, instance_id=instance_id)
    now = utc_now()
    neutral_affect = AffectState.neutral()

    # Build personality vector from seed
    personality = seed.identity.personality
    personality_values = [
        personality.warmth, personality.directness, personality.verbosity,
        personality.formality, personality.curiosity_expression, personality.humour,
        personality.empathy_expression, personality.confidence_display, personality.metaphor_use,
    ]
    # Also store as a named dict for Voxis to load directly
    personality_dict = json.dumps({
        "warmth": personality.warmth,
        "directness": personality.directness,
        "verbosity": personality.verbosity,
        "formality": personality.formality,
        "curiosity_expression": personality.curiosity_expression,
        "humour": personality.humour,
        "empathy_expression": personality.empathy_expression,
        "confidence_display": personality.confidence_display,
        "metaphor_use": personality.metaphor_use,
    })

    # 1. Create Self node
    constitution_id = new_id()

    affect_map = neutral_affect.to_map()
    await neo4j.execute_write(
        """
        CREATE (s:Self {
            instance_id: $instance_id,
            name: $name,
            born_at: datetime($born_at),
            affect_valence: $affect_valence,
            affect_arousal: $affect_arousal,
            affect_dominance: $affect_dominance,
            affect_curiosity: $affect_curiosity,
            affect_care_activation: $affect_care_activation,
            affect_coherence_stress: $affect_coherence_stress,
            autonomy_level: $autonomy_level,
            personality_vector: $personality_vector,
            personality_json: $personality_json,
            traits: $traits,
            cycle_count: 0,
            total_episodes: 0,
            total_entities: 0,
            total_communities: 0
        })
        RETURN s
        """,
        {
            "instance_id": instance_id,
            "name": seed.instance.name,
            "born_at": now.isoformat(),
            "affect_valence": affect_map.get("valence", 0.0),
            "affect_arousal": affect_map.get("arousal", 0.0),
            "affect_dominance": affect_map.get("dominance", 0.0),
            "affect_curiosity": affect_map.get("curiosity", 0.0),
            "affect_care_activation": affect_map.get("care_activation", 0.0),
            "affect_coherence_stress": affect_map.get("coherence_stress", 0.0),
            "autonomy_level": seed.constitution.autonomy_level,
            "personality_vector": personality_values,
            "personality_json": personality_dict,
            "traits": seed.identity.traits,
        },
    )

    # 2. Create Constitution node
    drives = seed.constitution.drives
    await neo4j.execute_write(
        """
        CREATE (c:Constitution {
            id: $id,
            version: 1,
            drive_coherence: $coherence,
            drive_care: $care,
            drive_growth: $growth,
            drive_honesty: $honesty,
            amendments: [],
            last_amended: null
        })
        WITH c
        MATCH (s:Self {instance_id: $instance_id})
        CREATE (s)-[:GOVERNED_BY]->(c)
        RETURN c
        """,
        {
            "id": constitution_id,
            "instance_id": instance_id,
            "coherence": drives.coherence,
            "care": drives.care,
            "growth": drives.growth,
            "honesty": drives.honesty,
        },
    )

    # 3. Create initial entities from seed
    entity_count = 0
    for entity_config in seed.community.initial_entities:
        entity_id = new_id()

        # Compute embedding for the entity
        description_text = f"{entity_config.name}: {entity_config.description}"
        embedding = await embedding_client.embed(description_text)

        await neo4j.execute_write(
            """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                type: $type,
                description: $description,
                embedding: $embedding,
                first_seen: datetime($now),
                last_updated: datetime($now),
                last_accessed: datetime($now),
                salience_score: $salience,
                mention_count: 1,
                confidence: 1.0,
                is_core_identity: $is_core,
                community_ids: []
            })
            WITH e
            MATCH (s:Self {instance_id: $instance_id})
            CREATE (s)-[:CORE_CONCEPT]->(e)
            RETURN e
            """,
            {
                "id": entity_id,
                "name": entity_config.name,
                "type": entity_config.type,
                "description": entity_config.description,
                "embedding": embedding,
                "now": now.isoformat(),
                "salience": 0.9 if entity_config.is_core_identity else 0.5,
                "is_core": entity_config.is_core_identity,
                "instance_id": instance_id,
            },
        )
        entity_count += 1
        logger.info(
            "initial_entity_created",
            name=entity_config.name,
            type=entity_config.type,
            is_core=entity_config.is_core_identity,
        )

    # 4. Store the birth event as the first episode
    birth_content = (
        f"Instance '{seed.instance.name}' was born. "
        f"Description: {seed.instance.description}. "
        f"Community context: {seed.community.context.strip()}"
    )
    birth_embedding = await embedding_client.embed(birth_content)
    birth_episode_id = new_id()

    await neo4j.execute_write(
        """
        CREATE (ep:Episode {
            id: $id,
            event_time: datetime($now),
            ingestion_time: datetime($now),
            valid_from: datetime($now),
            valid_until: null,
            source: 'birth',
            modality: 'internal',
            raw_content: $content,
            summary: $summary,
            embedding: $embedding,
            salience_composite: 1.0,
            salience_birth: 1.0,
            salience_identity: 1.0,
            affect_valence: 0.3,
            affect_arousal: 0.2,
            consolidation_level: 2,
            last_accessed: datetime($now),
            access_count: 1,
            free_energy: 0.0
        })
        RETURN ep
        """,
        {
            "id": birth_episode_id,
            "now": now.isoformat(),
            "content": birth_content,
            "summary": f"Birth of {seed.instance.name}",
            "embedding": birth_embedding,
        },
    )

    # Update Self counters
    await neo4j.execute_write(
        """
        MATCH (s:Self {instance_id: $instance_id})
        SET s.total_episodes = 1, s.total_entities = $entity_count
        """,
        {"instance_id": instance_id, "entity_count": entity_count},
    )

    # 5. Store the first governance record
    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: 'instance_born',
            timestamp: datetime($now),
            details: $details,
            actor: 'system',
            outcome: 'born'
        })
        """,
        {
            "id": new_id(),
            "now": now.isoformat(),
            "details": json.dumps({
                "seed_name": seed.instance.name,
                "autonomy_level": seed.constitution.autonomy_level,
                "initial_entities": entity_count,
            }),
        },
    )

    logger.info(
        "instance_birth_complete",
        name=seed.instance.name,
        instance_id=instance_id,
        initial_entities=entity_count,
    )

    return {
        "instance_id": instance_id,
        "name": seed.instance.name,
        "born_at": now.isoformat(),
        "autonomy_level": seed.constitution.autonomy_level,
        "initial_entities": entity_count,
    }


async def seed_simula_from_parent_genome(
    neo4j: Neo4jClient,
    child_instance_id: str,
    simula_genome_id: str,
) -> dict[str, Any]:
    """
    Seed a child instance's Simula subsystems from a parent's SimulaGenome.

    Called after birth_instance() when a ECODIAOS_SIMULA_GENOME_ID env var
    is present. Loads the genome from Neo4j by ID and seeds the child's
    evolution history, LILO library, GRPO training data, EFE calibration,
    and proven procedures.

    Args:
        neo4j: Neo4j client
        child_instance_id: The newly born instance's ID
        simula_genome_id: ID of the SimulaGenome node in Neo4j

    Returns:
        Seeding result dict with per-segment counts.
    """
    from systems.simula.genome import SimulaGenomeSeeder

    logger.info(
        "simula_genome_seeding_starting",
        child_instance_id=child_instance_id,
        simula_genome_id=simula_genome_id,
    )

    seeder = SimulaGenomeSeeder(neo4j=neo4j, child_instance_id=child_instance_id)

    genome = await seeder.load_genome_from_neo4j(simula_genome_id)
    if genome is None:
        logger.error(
            "simula_genome_seeding_failed",
            reason="genome_not_found",
            simula_genome_id=simula_genome_id,
        )
        return {"error": f"SimulaGenome {simula_genome_id} not found in Neo4j"}

    result = await seeder.seed_from_genome(genome)

    logger.info(
        "simula_genome_seeding_complete",
        child_instance_id=child_instance_id,
        genome_id=simula_genome_id,
        mutations_seeded=result.mutations_seeded,
        abstractions_seeded=result.abstractions_seeded,
        training_seeded=result.training_examples_seeded,
    )

    return result.model_dump()


async def seed_from_parent_genome(
    neo4j: Neo4jClient,
    child_instance_id: str,
    parent_genome_b64: str,
    compression_method: str = "zlib",
    parent_instance_id: str = "",
) -> dict[str, Any]:
    """
    Seed a child instance with inherited beliefs from a parent genome.

    Called after birth_instance() when a --parent_genome flag is provided.
    Decompresses the genome and creates hypothesis nodes with inherited
    confidence, enabling the child to skip redundant re-learning.

    Args:
        neo4j: Neo4j client
        child_instance_id: The newly born instance's ID
        parent_genome_b64: Base64-encoded compressed genome string
        compression_method: "lz4" or "zlib" (default "zlib")
        parent_instance_id: Parent instance ID (for lineage tracking)

    Returns:
        Inheritance report dict with fidelity metrics.
    """
    from systems.evo.genetic_memory import GenomeSeeder

    logger.info(
        "genome_seeding_starting",
        child_instance_id=child_instance_id,
        parent_instance_id=parent_instance_id,
        compression_method=compression_method,
    )

    seeder = GenomeSeeder(neo4j=neo4j, child_instance_id=child_instance_id)

    genome = await seeder.load_genome_from_base64(
        payload_b64=parent_genome_b64,
        compression_method=compression_method,
        parent_instance_id=parent_instance_id,
    )

    if genome is None:
        logger.error("genome_seeding_failed", reason="genome_load_failed")
        return {"error": "Failed to decompress parent genome"}

    report = await seeder.seed_from_genome(genome)

    logger.info(
        "genome_seeding_complete",
        child_instance_id=child_instance_id,
        hypotheses_seeded=report.total_inherited,
        genome_id=report.parent_genome_id,
    )

    return report.model_dump()
