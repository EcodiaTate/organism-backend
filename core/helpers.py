"""
EcodiaOS — Startup Helper Classes

Adapter classes and utility functions used during the organism's
startup sequence.  Extracted from main.py for maintainability.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.atune.service import AtuneService
    from systems.memory.service import MemoryService

logger = structlog.get_logger()


class MemoryWorkspaceAdapter:
    """
    Bridge between MemoryService and Atune's WorkspaceMemoryClient protocol.

    Enables:
    * **Spontaneous recall** — high-salience, recently-unaccessed episodes
      "bubble up" into consciousness (find_bubbling_memory).
    * **Context enrichment** — each broadcast winner gets contextual memories
      attached (retrieve_context).
    * **Broadcast-time storage** — winning percepts stored as episodes with
      full salience and affect metadata (store_percept_with_broadcast).
    """

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    async def retrieve_context(
        self,
        query_embedding: list[float],
        query_text: str,
        max_results: int,
    ) -> Any:
        """Retrieve memories relevant to the current workspace winner."""
        from systems.atune.types import MemoryContext

        try:
            response = await self._memory.retrieve(
                query_text=query_text or None,
                query_embedding=query_embedding or None,
                max_results=max_results,
                salience_floor=0.0,
            )
            from primitives.memory_trace import MemoryTrace

            traces = []
            for r in response.traces:
                traces.append(
                    MemoryTrace(
                        episode_id=r.node_id,
                        original_percept_id=r.node_id,
                        summary=r.content or "",
                    )
                )
            return MemoryContext(traces=traces)
        except Exception:
            return MemoryContext()

    async def find_bubbling_memory(
        self,
        min_salience: float,
        max_recent_access_hours: int,
    ) -> Any:
        """
        Find a high-salience episode that hasn't been accessed recently.
        This powers spontaneous recall — the "I just thought of something"
        indicating dormant but important memories ready to surface.
        """
        try:
            results = await self._memory._neo4j.execute_read(
                """
                MATCH (ep:Episode)
                WHERE ep.salience_composite >= $min_salience
                  AND ep.last_accessed < datetime() - duration({hours: $hours})
                RETURN ep.id AS id, ep.summary AS content,
                       ep.salience_composite AS salience,
                       ep.embedding AS embedding
                ORDER BY ep.salience_composite DESC
                LIMIT 1
                """,
                {"min_salience": min_salience, "hours": max_recent_access_hours},
            )
            if not results:
                return None

            row = results[0]
            from primitives.common import (
                Modality,
                SourceDescriptor,
                SystemID,
                utc_now,
            )
            from primitives.percept import Content, Percept

            return Percept(
                source=SourceDescriptor(
                    system=SystemID.MEMORY,
                    channel="spontaneous_recall",
                    modality=Modality.TEXT,
                ),
                content=Content(
                    raw=row.get("content", ""),
                    embedding=row.get("embedding") or [],
                ),
                timestamp=utc_now(),
            )
        except Exception:
            return None

    async def store_percept_with_broadcast(
        self, percept: Any, salience: Any, affect: Any
    ) -> None:
        """Store the broadcast-winning percept as an episode."""
        try:
            await self._memory.store_percept(
                percept=percept,
                salience_composite=salience.composite,
                salience_scores=salience.scores,
                affect_valence=affect.valence,
                affect_arousal=affect.arousal,
            )
        except Exception:
            logger.debug("broadcast_storage_failed", exc_info=True)


async def seed_atune_cache(
    atune: AtuneService, embedding_client: Any, instance: Any
) -> None:
    """
    Populate Atune's identity cache from the born instance so the
    Identity salience head can score percepts correctly from the start.
    """
    try:
        name_embedding = await embedding_client.embed(instance.name)
        community_text = getattr(instance, "community_context", "") or instance.name
        community_embedding = await embedding_client.embed(community_text)

        atune.set_cache_identity(
            core_embeddings=[name_embedding],
            community_embedding=community_embedding,
            instance_name=instance.name,
        )
        logger.info("atune_cache_seeded", instance_name=instance.name)
    except Exception:
        logger.warning("atune_cache_seed_failed", exc_info=True)


def resolve_governance_config(config: Any) -> Any:
    """Resolve governance config from seed or use defaults."""
    from config import GovernanceConfig, load_seed

    try:
        seed_path = os.environ.get(
            "ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml"
        )
        seed = load_seed(seed_path)
        return seed.constitution.governance
    except FileNotFoundError:
        return GovernanceConfig()
    except Exception as exc:
        logger.warning(
            "governance_config_load_failed",
            error=str(exc),
            fallback="GovernanceConfig defaults",
        )
        return GovernanceConfig()
