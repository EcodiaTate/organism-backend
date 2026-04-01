"""
EcodiaOS - Episodic Memory

Storage and retrieval of discrete experience records (Episodes).
This is the "what happened" layer of memory.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from primitives import Episode

logger = structlog.get_logger()


async def store_episode(
    neo4j: Neo4jClient,
    episode: Episode,
) -> str:
    """
    Store a new episode in the knowledge graph.
    Target: ≤50ms (just node creation; extraction is async).

    If a somatic_marker is present (from Soma §0.5), its 19D vector
    is stored as ``somatic_vector`` for cosine-similarity reranking
    and the full marker dict is stored as ``somatic_marker_json``.
    """
    # Flatten somatic marker to 19D vector + JSON for persistence
    somatic_vector: list[float] | None = None
    somatic_marker_json: str | None = None
    if episode.somatic_marker is not None:
        try:
            if hasattr(episode.somatic_marker, "to_vector"):
                somatic_vector = episode.somatic_marker.to_vector()
            elif episode.somatic_vector is not None:
                somatic_vector = episode.somatic_vector
            somatic_marker_json = json.dumps(
                episode.somatic_marker.model_dump()
                if hasattr(episode.somatic_marker, "model_dump")
                else str(episode.somatic_marker)
            )
        except Exception:
            logger.debug("somatic_marker_serialise_failed", exc_info=True)

    await neo4j.execute_write(
        """
        CREATE (e:Episode {
            id: $id,
            event_time: datetime($event_time),
            ingestion_time: datetime($ingestion_time),
            valid_from: datetime($valid_from),
            valid_until: $valid_until,
            source: $source,
            modality: $modality,
            raw_content: $raw_content,
            summary: $summary,
            embedding: $embedding,
            salience_composite: $salience_composite,
            salience_scores_json: $salience_scores_json,
            affect_valence: $affect_valence,
            affect_arousal: $affect_arousal,
            consolidation_level: $consolidation_level,
            last_accessed: datetime($last_accessed),
            access_count: 0,
            free_energy: $free_energy,
            somatic_vector: $somatic_vector,
            somatic_marker_json: $somatic_marker_json,
            novelty_score: $novelty_score,
            context_summary: $context_summary,
            is_correction: $is_correction
        })
        """,
        {
            "id": episode.id,
            "event_time": episode.event_time.isoformat(),
            "ingestion_time": episode.ingestion_time.isoformat(),
            "valid_from": episode.valid_from.isoformat(),
            "valid_until": episode.valid_until.isoformat() if episode.valid_until else None,
            "source": episode.source,
            "modality": episode.modality,
            "raw_content": episode.raw_content,
            "summary": episode.summary,
            "embedding": episode.embedding,
            "salience_composite": episode.salience_composite,
            "salience_scores_json": json.dumps(episode.salience_scores),
            "affect_valence": episode.affect_valence,
            "affect_arousal": episode.affect_arousal,
            "consolidation_level": episode.consolidation_level,
            "last_accessed": episode.last_accessed.isoformat(),
            "free_energy": episode.free_energy,
            "somatic_vector": somatic_vector,
            "somatic_marker_json": somatic_marker_json,
            "novelty_score": episode.novelty_score,
            "context_summary": episode.context_summary,
            "is_correction": episode.is_correction,
        },
    )

    # Increment Self counter
    await neo4j.execute_write(
        "MATCH (s:Self) SET s.total_episodes = s.total_episodes + 1"
    )

    logger.debug(
        "episode_stored",
        episode_id=episode.id,
        source=episode.source,
        salience=episode.salience_composite,
    )
    return episode.id


async def link_episode_sequence(
    neo4j: Neo4jClient,
    previous_episode_id: str,
    current_episode_id: str,
    gap_seconds: float = 0.0,
    causal_strength: float = 0.1,
) -> None:
    """Link two episodes in temporal sequence."""
    await neo4j.execute_write(
        """
        MATCH (prev:Episode {id: $prev_id})
        MATCH (curr:Episode {id: $curr_id})
        CREATE (prev)-[:FOLLOWED_BY {
            gap_seconds: $gap,
            causal_strength: $causal
        }]->(curr)
        """,
        {
            "prev_id": previous_episode_id,
            "curr_id": current_episode_id,
            "gap": gap_seconds,
            "causal": causal_strength,
        },
    )


async def get_episode(neo4j: Neo4jClient, episode_id: str) -> dict[str, Any] | None:
    """Retrieve a single episode by ID."""
    results = await neo4j.execute_read(
        "MATCH (e:Episode {id: $id}) RETURN e",
        {"id": episode_id},
    )
    if results:
        return results[0]["e"]  # type: ignore[no-any-return]
    return None


async def get_recent_episodes(
    neo4j: Neo4jClient,
    limit: int = 20,
    min_salience: float = 0.0,
) -> list[dict[str, Any]]:
    """Get the most recent episodes, optionally filtered by salience."""
    return await neo4j.execute_read(
        """
        MATCH (e:Episode)
        WHERE e.salience_composite >= $min_salience
        RETURN e
        ORDER BY e.ingestion_time DESC
        LIMIT $limit
        """,
        {"min_salience": min_salience, "limit": limit},
    )


async def update_access(neo4j: Neo4jClient, episode_ids: list[str]) -> None:
    """Update access timestamps and counts for retrieved episodes (salience boost)."""
    if not episode_ids:
        return
    await neo4j.execute_write(
        """
        UNWIND $ids AS eid
        MATCH (e:Episode {id: eid})
        SET e.last_accessed = datetime(),
            e.access_count = e.access_count + 1
        """,
        {"ids": episode_ids},
    )


async def count_episodes(neo4j: Neo4jClient) -> int:
    """Get total episode count."""
    results = await neo4j.execute_read("MATCH (e:Episode) RETURN count(e) AS cnt")
    return results[0]["cnt"] if results else 0


# ─── Counterfactual Episode Storage ─────────────────────────────────


async def store_counterfactual_episode(
    neo4j: Neo4jClient,
    record: Any,
) -> str:
    """
    Store a rejected policy as an :Episode:Counterfactual node.

    Uses multi-label (both Episode and Counterfactual) so existing retrieval
    can optionally filter counterfactuals while graph traversals still reach them.
    Target: ≤50ms per node (same as regular episode storage).

    Args:
        record: A CounterfactualRecord from nova.types.
    """
    ts = record.timestamp.isoformat()
    raw_content = (
        f"Counterfactual policy: {record.policy_name} ({record.policy_type}). "
        f"{record.policy_description}"
    )
    summary = (
        f"Rejected policy '{record.policy_name}' for goal '{record.goal_description[:80]}'. "
        f"EFE={record.efe_total:.3f}, pragmatic={record.estimated_pragmatic_value:.2f}, "
        f"epistemic={record.estimated_epistemic_value:.2f}. "
        f"Chosen instead: '{record.chosen_policy_name}' (EFE={record.chosen_efe_total:.3f})."
    )

    await neo4j.execute_write(
        """
        CREATE (e:Episode:Counterfactual {
            id: $id,
            event_time: datetime($timestamp),
            ingestion_time: datetime($timestamp),
            valid_from: datetime($timestamp),
            source: 'nova:counterfactual',
            modality: 'internal',
            raw_content: $raw_content,
            summary: $summary,
            salience_composite: 0.3,
            affect_valence: 0.0,
            affect_arousal: 0.0,
            consolidation_level: 0,
            last_accessed: datetime($timestamp),
            access_count: 0,
            free_energy: 0.0,
            intent_id: $intent_id,
            decision_record_id: $decision_record_id,
            goal_id: $goal_id,
            policy_name: $policy_name,
            policy_type: $policy_type,
            policy_reasoning: $policy_reasoning,
            efe_total: $efe_total,
            estimated_pragmatic_value: $estimated_pragmatic_value,
            estimated_epistemic_value: $estimated_epistemic_value,
            constitutional_alignment: $constitutional_alignment,
            feasibility: $feasibility,
            risk_expected_harm: $risk_expected_harm,
            chosen_policy_name: $chosen_policy_name,
            chosen_efe_total: $chosen_efe_total,
            resolved: false
        })
        """,
        {
            "id": record.id,
            "timestamp": ts,
            "raw_content": raw_content,
            "summary": summary,
            "intent_id": record.intent_id,
            "decision_record_id": record.decision_record_id,
            "goal_id": record.goal_id,
            "policy_name": record.policy_name,
            "policy_type": record.policy_type,
            "policy_reasoning": record.policy_reasoning[:500],
            "efe_total": record.efe_total,
            "estimated_pragmatic_value": record.estimated_pragmatic_value,
            "estimated_epistemic_value": record.estimated_epistemic_value,
            "constitutional_alignment": record.constitutional_alignment,
            "feasibility": record.feasibility,
            "risk_expected_harm": record.risk_expected_harm,
            "chosen_policy_name": record.chosen_policy_name,
            "chosen_efe_total": record.chosen_efe_total,
        },
    )

    logger.debug(
        "counterfactual_episode_stored",
        cf_id=record.id,
        intent_id=record.intent_id,
        policy_name=record.policy_name,
    )
    return record.id


async def resolve_counterfactual(
    neo4j: Neo4jClient,
    record_id: str,
    outcome_success: bool,
    actual_pragmatic_value: float,
    regret: float,
) -> None:
    """
    Update a counterfactual node with outcome-derived regret.

    Called when the chosen intent's outcome arrives. Marks the counterfactual
    as resolved and stores the computed regret value.

    Regret semantics: positive = the counterfactual was estimated better than
    the actual outcome (we might have made a mistake choosing this policy).
    """
    await neo4j.execute_write(
        """
        MATCH (e:Counterfactual {id: $id})
        SET e.resolved = true,
            e.actual_outcome_success = $success,
            e.actual_pragmatic_value = $actual_pragmatic,
            e.regret = $regret,
            e.resolved_at = datetime()
        """,
        {
            "id": record_id,
            "success": outcome_success,
            "actual_pragmatic": actual_pragmatic_value,
            "regret": regret,
        },
    )


async def link_counterfactual_to_outcome(
    neo4j: Neo4jClient,
    counterfactual_id: str,
    outcome_episode_id: str,
) -> None:
    """
    Create ALTERNATIVE_TO relationship from counterfactual to the actual
    outcome episode. Enables graph traversal from actual outcomes to
    their counterfactual alternatives.
    """
    await neo4j.execute_write(
        """
        MATCH (cf:Counterfactual {id: $cf_id})
        MATCH (actual:Episode {id: $actual_id})
        MERGE (cf)-[:ALTERNATIVE_TO]->(actual)
        """,
        {"cf_id": counterfactual_id, "actual_id": outcome_episode_id},
    )
