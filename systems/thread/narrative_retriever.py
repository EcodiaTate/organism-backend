"""
EcodiaOS - Thread Narrative Retriever

Provides autobiographical recall for other systems. When Nova needs to
understand "what kind of decisions have I made in situations like this?",
when Voxis needs to express "who I am", when a user asks "tell me about
yourself" - NarrativeRetriever resolves the query against the narrative graph.

Performance:
- "Who am I?" resolution: ≤500ms (Neo4j reads + assembly, no LLM)
- Schema-relevant retrieval: ≤200ms (vector search + episode retrieval)
- Past self retrieval: ≤300ms (fingerprint + chapter lookup)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector
from systems.thread.types import (
    Commitment,
    CommitmentStatus,
    IdentitySchema,
    NarrativeCoherence,
    NarrativeIdentitySummary,
    SchemaStrength,
    SchemaValence,
    ThreadConfig,
    TurningPoint,
    TurningPointType,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


class NarrativeRetriever:
    """
    Autobiographical recall engine.

    Resolves narrative queries against the NarrativeGraph:
    - WHO_AM_I: full identity summary
    - WHAT_DEFINES_ME: core schemas and commitments
    - SCHEMA_RELEVANT: episodes where schemas activated in similar contexts
    - CHAPTER_CONTEXT: current chapter's narrative context
    - HOW_HAVE_I_CHANGED: diachronic comparison
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        config: ThreadConfig,
    ) -> None:
        self._neo4j = neo4j
        self._config = config
        self._logger = logger.bind(system="thread.narrative_retriever")

    async def resolve_who_am_i(self) -> NarrativeIdentitySummary:
        """
        Assemble a complete identity summary.
        The definitive answer to "who am I?"

        Budget: ≤500ms (Neo4j reads + assembly, no LLM).
        """
        try:
            # 1. Fetch CORE and ESTABLISHED schemas
            core_schemas = await self._fetch_schemas(SchemaStrength.CORE)
            established_schemas = await self._fetch_schemas(SchemaStrength.ESTABLISHED)

            # 2. Fetch active commitments
            active_commitments = await self._fetch_active_commitments()

            # 3. Fetch current forming chapter
            current_chapter = await self._fetch_current_chapter()

            # 4. Fetch life story summary from Self node
            self_data = await self._fetch_self_node()
            life_story = self_data.get("autobiography_summary", "")
            personality = self_data.get("personality_vector", {})

            # 5. Fetch recent turning points
            recent_tps = await self._fetch_recent_turning_points(limit=5)

            # 6. Compute scores
            idem_score = float(self_data.get("idem_score", 0.0))
            ipse_score = self._compute_ipse_from_commitments(active_commitments)

            # 7. Determine narrative coherence
            coherence = self._assess_coherence(
                core_schemas, active_commitments, idem_score, ipse_score
            )

            summary = NarrativeIdentitySummary(
                core_schemas=core_schemas,
                established_schemas=established_schemas,
                active_commitments=active_commitments,
                current_chapter_title=current_chapter.get("title", ""),
                current_chapter_theme=current_chapter.get("theme", ""),
                life_story_summary=life_story,
                key_personality_traits=personality if isinstance(personality, dict) else {},
                recent_turning_points=recent_tps,
                narrative_coherence=coherence,
                idem_score=idem_score,
                ipse_score=ipse_score,
            )

            self._logger.debug(
                "who_am_i_resolved",
                core_schemas=len(core_schemas),
                commitments=len(active_commitments),
                coherence=coherence.value,
            )
            return summary

        except Exception as exc:
            self._logger.error("who_am_i_failed", error=str(exc))
            return NarrativeIdentitySummary()

    async def retrieve_schema_relevant(
        self,
        situation_embedding: list[float],
        max_results: int = 5,
    ) -> list[tuple[IdentitySchema, list[dict[str, Any]]]]:
        """
        Given a situation, find schemas that are relevant and the episodes
        where they were activated in similar contexts.

        This gives Nova narrative context for decision-making:
        'In situations like this, I have historically acted in ways
        consistent with schema X, as evidenced by episodes Y and Z.'

        Budget: ≤200ms.
        """
        try:
            # Vector search for similar schemas
            results = await self._neo4j.execute_read(
                """
                CALL db.index.vector.queryNodes('thread_schema_embedding', $k, $embedding)
                YIELD node, score
                WHERE score > 0.4
                RETURN node, score
                ORDER BY score DESC
                """,
                {"k": max_results, "embedding": situation_embedding},
            )

            schema_episode_pairs: list[tuple[IdentitySchema, list[dict[str, Any]]]] = []

            for r in results:
                node = r["node"]
                schema = self._node_to_schema(node)

                # Fetch confirmed episodes for this schema
                episodes = await self._neo4j.execute_read(
                    """
                    MATCH (s:IdentitySchema {id: $schema_id})-[:CONFIRMED_BY]->(e:Episode)
                    RETURN e.id AS id, e.summary AS summary, e.event_time AS event_time
                    ORDER BY e.event_time DESC
                    LIMIT 5
                    """,
                    {"schema_id": schema.id},
                )

                episode_dicts = [dict(ep) for ep in episodes]
                schema_episode_pairs.append((schema, episode_dicts))

            return schema_episode_pairs

        except Exception as exc:
            self._logger.warning("schema_relevant_retrieval_failed", error=str(exc))
            return []

    async def retrieve_chapter_context(self) -> dict[str, Any]:
        """
        Get the current chapter's narrative context.

        Returns dict with: title, theme, arc_type, episode_count,
        scenes, turning_points, active_schemas.
        """
        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:CURRENT_CHAPTER]->(c:NarrativeChapter)
                OPTIONAL MATCH (c)-[:CONTAINS]->(scene:NarrativeScene)
                OPTIONAL MATCH (tp:TurningPoint)-[:WITHIN]->(c)
                RETURN c, collect(DISTINCT scene.summary) AS scene_summaries,
                       collect(DISTINCT tp.description) AS tp_descriptions
                """,
                {},
            )

            for r in results:
                chapter_node = r["c"]
                props = dict(chapter_node)
                return {
                    "title": props.get("title", ""),
                    "theme": props.get("theme", ""),
                    "arc_type": props.get("arc_type", "growth"),
                    "episode_count": int(props.get("episode_count", 0)),
                    "scenes": r.get("scene_summaries", []),
                    "turning_points": r.get("tp_descriptions", []),
                    "status": props.get("status", "forming"),
                }

            return {}

        except Exception as exc:
            self._logger.warning("chapter_context_failed", error=str(exc))
            return {}

    async def retrieve_past_self(self, temporal_reference: str) -> dict[str, Any]:
        """
        Reconstruct a snapshot of the organism's identity at a past time.

        Temporal reference parsing:
        - "beginning" → first fingerprint/chapter
        - "chapter N" → specific chapter's snapshots
        - "last chapter" → most recently closed chapter

        Budget: ≤300ms.
        """
        try:
            if temporal_reference == "beginning":
                return await self._fetch_first_snapshot()
            elif temporal_reference.startswith("chapter"):
                # Extract chapter number
                parts = temporal_reference.split()
                if len(parts) >= 2:
                    return await self._fetch_chapter_snapshot(parts[1])
            elif temporal_reference == "last chapter":
                return await self._fetch_last_closed_chapter_snapshot()

            return {}

        except Exception as exc:
            self._logger.warning("past_self_retrieval_failed", error=str(exc))
            return {}

    # ─── Private Helpers ─────────────────────────────────────────────

    async def _fetch_schemas(self, strength: SchemaStrength) -> list[IdentitySchema]:
        """Fetch schemas of a given strength from Neo4j."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)-[:HAS_SCHEMA]->(schema:IdentitySchema)
            WHERE schema.strength = $strength
            RETURN schema
            """,
            {"strength": strength.value},
        )
        return [self._node_to_schema(r["schema"]) for r in results]

    async def _fetch_active_commitments(self) -> list[Commitment]:
        """Fetch commitments with ACTIVE or TESTED status."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)-[:HOLDS_COMMITMENT]->(c:Commitment)
            WHERE c.status IN ['active', 'tested']
            RETURN c
            """,
            {},
        )
        return [self._node_to_commitment(r["c"]) for r in results]

    async def _fetch_current_chapter(self) -> dict[str, Any]:
        """Fetch the current forming chapter."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)-[:CURRENT_CHAPTER]->(c:NarrativeChapter)
            RETURN c.title AS title, c.theme AS theme, c.status AS status,
                   c.episode_count AS episode_count
            """,
            {},
        )
        for r in results:
            return dict(r)
        return {}

    async def _fetch_self_node(self) -> dict[str, Any]:
        """Fetch core properties from the Self node."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)
            RETURN s.autobiography_summary AS autobiography_summary,
                   s.personality_vector_json AS personality_vector_json,
                   s.idem_score AS idem_score,
                   s.ipse_score AS ipse_score,
                   s.current_life_theme AS current_life_theme
            """,
            {},
        )
        for r in results:
            data = dict(r)
            pv_json = data.get("personality_vector_json")
            if pv_json and isinstance(pv_json, str):
                data["personality_vector"] = json.loads(pv_json)
            else:
                data["personality_vector"] = {}
            return data
        return {}

    async def _fetch_recent_turning_points(self, limit: int = 5) -> list[TurningPoint]:
        """Fetch the most recent turning points."""
        results = await self._neo4j.execute_read(
            """
            MATCH (tp:TurningPoint)
            RETURN tp
            ORDER BY tp.timestamp DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        tps: list[TurningPoint] = []
        for r in results:
            node = r["tp"]
            props = dict(node)
            tps.append(TurningPoint(
                id=props.get("id", ""),
                chapter_id=props.get("chapter_id", ""),
                type=TurningPointType(props.get("type", "revelation")),
                description=props.get("description", ""),
                surprise_magnitude=float(props.get("surprise_magnitude", 0.0)),
                narrative_weight=float(props.get("narrative_weight", 0.0)),
            ))
        return tps

    async def _fetch_first_snapshot(self) -> dict[str, Any]:
        """Fetch the earliest available identity snapshot."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:NarrativeChapter)
            RETURN c
            ORDER BY c.started_at ASC
            LIMIT 1
            """,
            {},
        )
        for r in results:
            props = dict(r["c"])
            return {
                "title": props.get("title", ""),
                "theme": props.get("theme", ""),
                "personality_snapshot": json.loads(
                    props.get("personality_snapshot_start_json", "{}")
                ),
            }
        return {}

    async def _fetch_chapter_snapshot(self, chapter_ref: str) -> dict[str, Any]:
        """Fetch snapshot for a specific chapter."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:NarrativeChapter)
            RETURN c
            ORDER BY c.started_at ASC
            """,
            {},
        )
        chapters = list(results)
        try:
            idx = int(chapter_ref) - 1  # 1-indexed
            if 0 <= idx < len(chapters):
                props = dict(chapters[idx]["c"])
                return {
                    "title": props.get("title", ""),
                    "summary": props.get("summary", ""),
                    "personality_snapshot_start": json.loads(
                        props.get("personality_snapshot_start_json", "{}")
                    ),
                    "personality_snapshot_end": json.loads(
                        props.get("personality_snapshot_end_json", "{}")
                    ),
                }
        except (ValueError, IndexError):
            pass
        return {}

    async def _fetch_last_closed_chapter_snapshot(self) -> dict[str, Any]:
        """Fetch the most recently closed chapter's snapshot."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:NarrativeChapter)
            WHERE c.status = 'closed'
            RETURN c
            ORDER BY c.ended_at DESC
            LIMIT 1
            """,
            {},
        )
        for r in results:
            props = dict(r["c"])
            return {
                "title": props.get("title", ""),
                "summary": props.get("summary", ""),
                "personality_snapshot_end": json.loads(
                    props.get("personality_snapshot_end_json", "{}")
                ),
            }
        return {}

    def _compute_ipse_from_commitments(self, commitments: list[Commitment]) -> float:
        """Compute ipse score from active commitments."""
        tested = [
            c for c in commitments
            if c.tests_faced >= self._config.commitment_min_tests_for_fidelity
        ]
        if not tested:
            return 1.0  # Default - untested commitments are not broken
        return float(sum(c.fidelity for c in tested) / len(tested))

    def _assess_coherence(
        self,
        core_schemas: list[IdentitySchema],
        commitments: list[Commitment],
        idem_score: float,
        ipse_score: float,
    ) -> NarrativeCoherence:
        """Assess overall narrative coherence from identity metrics."""
        # Check for conflicting schemas
        maladaptive_count = sum(1 for s in core_schemas if s.valence == SchemaValence.MALADAPTIVE)

        # Check for strained commitments
        strained = sum(
            1 for c in commitments
            if c.fidelity < self._config.commitment_strain_threshold
        )

        if maladaptive_count > 0 and strained > 0:
            return NarrativeCoherence.CONFLICTED

        if idem_score < 0.4 or ipse_score < 0.5:
            return NarrativeCoherence.FRAGMENTED

        if len(core_schemas) == 0 and len(commitments) == 0:
            return NarrativeCoherence.TRANSITIONAL

        if idem_score >= 0.6 and ipse_score >= 0.7:
            return NarrativeCoherence.INTEGRATED

        return NarrativeCoherence.TRANSITIONAL

    def _node_to_schema(self, node: Any) -> IdentitySchema:
        """Convert a Neo4j node to an IdentitySchema."""
        props = dict(node)
        return IdentitySchema(
            id=props.get("id", ""),
            statement=props.get("statement", ""),
            trigger_contexts=json.loads(props.get("trigger_contexts_json", "[]")),
            behavioral_tendency=props.get("behavioral_tendency", ""),
            emotional_signature=json.loads(props.get("emotional_signature_json", "{}")),
            drive_alignment=DriveAlignmentVector(
                **json.loads(props.get("drive_alignment_json", "{}"))
            ),
            strength=SchemaStrength(props.get("strength", "nascent")),
            valence=SchemaValence(props.get("valence", "adaptive")),
            confirmation_count=int(props.get("confirmation_count", 0)),
            disconfirmation_count=int(props.get("disconfirmation_count", 0)),
            evidence_ratio=float(props.get("evidence_ratio", 0.5)),
            embedding=props.get("embedding"),
        )

    def _node_to_commitment(self, node: Any) -> Commitment:
        """Convert a Neo4j node to a Commitment."""
        props = dict(node)
        return Commitment(
            id=props.get("id", ""),
            statement=props.get("statement", ""),
            status=CommitmentStatus(props.get("status", "active")),
            tests_faced=int(props.get("tests_faced", 0)),
            tests_held=int(props.get("tests_held", 0)),
            fidelity=float(props.get("fidelity", 1.0)),
            embedding=props.get("embedding"),
        )
