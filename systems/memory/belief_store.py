"""
EcodiaOS — Belief Store

Persists significant beliefs to the Neo4j knowledge graph as :Belief nodes,
stamped with domain-aware half-life metadata for freshness tracking.

Beliefs in the graph are the organism's durable knowledge claims — distinct
from Nova's ephemeral in-memory BeliefState. They represent facts the
organism considers worth remembering and re-verifying over time.

Integration:
    - Called from Evo when hypotheses are confirmed (SUPPORTED → INTEGRATED)
    - Called from Nova when high-confidence beliefs crystallise
    - Read by the BeliefAgingScanner during consolidation Phase 2.5
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import structlog

from primitives.common import new_id, utc_now
from systems.evo.belief_halflife import get_halflife_for_domain

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


async def store_belief(
    neo4j: Neo4jClient,
    domain: str,
    statement: str,
    precision: float = 0.5,
    evidence_ids: list[str] | None = None,
    half_life_days: float | None = None,
    source_hypothesis_id: str | None = None,
) -> str:
    """
    Create a new :Belief node in the knowledge graph with half-life metadata.

    Args:
        neo4j: Neo4j client
        domain: Knowledge domain (e.g. "sentiment", "capability", "schedule")
        statement: The belief content — what the organism claims to know
        precision: Confidence level (0–1)
        evidence_ids: Episode IDs that support this belief
        half_life_days: Override domain default if provided
        source_hypothesis_id: Evo hypothesis that produced this belief

    Returns:
        The belief node ID.
    """
    belief_id = new_id()
    now = utc_now()
    now_iso = now.isoformat()

    if half_life_days is None:
        half_life_days = get_halflife_for_domain(domain)

    decay_constant = math.log(2) / half_life_days if half_life_days > 0 else 0.0

    try:
        await neo4j.execute_write(
            """
            CREATE (b:Belief {
                id: $id,
                domain: $domain,
                statement: $statement,
                precision: $precision,
                half_life_days: $half_life_days,
                decay_constant: $decay_constant,
                volatility_percentile: 0.5,
                last_verified: $now,
                created_at: $now,
                updated_at: $now,
                source_hypothesis_id: $source_hypothesis_id
            })
            """,
            {
                "id": belief_id,
                "domain": domain,
                "statement": statement,
                "precision": precision,
                "half_life_days": half_life_days,
                "decay_constant": decay_constant,
                "now": now_iso,
                "source_hypothesis_id": source_hypothesis_id or "",
            },
        )

        # Link to supporting episodes if provided
        if evidence_ids:
            for episode_id in evidence_ids[:20]:  # Cap at 20 evidence links
                await neo4j.execute_write(
                    """
                    MATCH (b:Belief {id: $belief_id}), (e:Episode {id: $episode_id})
                    MERGE (b)-[:SUPPORTED_BY]->(e)
                    """,
                    {"belief_id": belief_id, "episode_id": episode_id},
                )

        # Link to source hypothesis if provided
        if source_hypothesis_id:
            await neo4j.execute_write(
                """
                MATCH (b:Belief {id: $belief_id}), (h:Hypothesis {hypothesis_id: $hypothesis_id})
                MERGE (b)-[:DERIVED_FROM]->(h)
                """,
                {"belief_id": belief_id, "hypothesis_id": source_hypothesis_id},
            )

        logger.info(
            "belief_stored",
            belief_id=belief_id,
            domain=domain,
            half_life_days=half_life_days,
            precision=round(precision, 3),
            evidence_count=len(evidence_ids or []),
        )

    except Exception as exc:
        logger.error(
            "belief_store_failed",
            domain=domain,
            error=str(exc),
        )

    return belief_id


async def update_belief_verification(
    neo4j: Neo4jClient,
    belief_id: str,
    new_precision: float | None = None,
) -> None:
    """
    Re-verify a belief: reset last_verified and optionally update precision.
    Called when Nova/Axon confirms the belief is still accurate.

    Refuses to update beliefs that have been superseded by a ConsolidatedBelief
    (immutability guard — consolidated beliefs are read-only).
    """
    # Immutability guard: check if this belief has been consolidated
    try:
        check = await neo4j.execute_read(
            "MATCH (b:Belief {id: $belief_id}) RETURN b.status AS status",
            {"belief_id": belief_id},
        )
        if check and check[0].get("status") == "superseded_by_consolidated":
            logger.warning(
                "belief_update_blocked_consolidated",
                belief_id=belief_id,
            )
            return
    except Exception as exc:
        logger.warning(
            "belief_consolidation_check_failed",
            belief_id=belief_id,
            error=str(exc),
        )
        # Proceed with update if the check fails — fail open

    now_iso = utc_now().isoformat()
    params: dict[str, str | float] = {
        "belief_id": belief_id,
        "now": now_iso,
    }

    set_clause = "SET b.last_verified = $now, b.updated_at = $now"
    if new_precision is not None:
        set_clause += ", b.precision = $precision"
        params["precision"] = new_precision

    try:
        await neo4j.execute_write(
            f"""
            MATCH (b:Belief {{id: $belief_id}})
            {set_clause}
            """,
            params,
        )
    except Exception as exc:
        logger.warning(
            "belief_verification_update_failed",
            belief_id=belief_id,
            error=str(exc),
        )


async def store_belief_from_hypothesis(
    neo4j: Neo4jClient,
    hypothesis_id: str,
    statement: str,
    category: str,
    evidence_score: float,
    supporting_episodes: list[str],
) -> str:
    """
    Convert an integrated Evo hypothesis into a persisted :Belief node.

    Called during consolidation Phase 2 when a hypothesis reaches INTEGRATED.
    The hypothesis's domain is inferred from its category.
    """
    # Map hypothesis category to belief domain
    category_to_domain: dict[str, str] = {
        "world_model": "capability",
        "self_model": "identity",
        "social": "social",
        "procedural": "process",
        "parameter": "technical_capability",
    }
    domain = category_to_domain.get(category, "general")

    # Map evidence score to precision (sigmoid-like scaling)
    precision = min(0.99, 0.5 + evidence_score * 0.05)

    return await store_belief(
        neo4j=neo4j,
        domain=domain,
        statement=statement,
        precision=precision,
        evidence_ids=supporting_episodes,
        source_hypothesis_id=hypothesis_id,
    )
