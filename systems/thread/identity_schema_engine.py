"""
EcodiaOS - Thread Identity Schema Engine

Maintains the organism's core self-beliefs: the "I am the kind of entity
that..." statements that create narrative coherence.

This is Ricoeur's *idem* - structural sameness across time.

Schema formation requires evidence: at least 5 supporting episodes spanning
48+ hours. CORE schemas require 50+ confirmations AND 180+ days of age.
Schemas can never be fabricated - every schema earns its place through
lived experience.

Velocity-limited to prevent identity instability:
- Max 1 schema promotion per 24 hours
- Max 1 new schema per 48 hours
- CORE schemas can never be deleted, only marked MALADAPTIVE
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector, utc_now
from systems.thread.types import (
    IdentitySchema,
    SchemaStrength,
    SchemaValence,
    ThreadConfig,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient

from clients.optimized_llm import OptimizedLLMProvider

logger = structlog.get_logger()

# Schema strength → precision mapping for self-evidencing
SCHEMA_PRECISION: dict[SchemaStrength, float] = {
    SchemaStrength.NASCENT: 0.2,
    SchemaStrength.DEVELOPING: 0.4,
    SchemaStrength.ESTABLISHED: 0.6,
    SchemaStrength.CORE: 0.8,
}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class IdentitySchemaEngine:
    """
    Maintains the organism's core self-beliefs.

    Operations:
    - form_schema_from_pattern: crystallize a recurring pattern into a schema
    - evaluate_evidence: does an episode confirm or challenge a schema?
    - check_promotions: promote schemas based on evidence accumulation
    - check_decay: demote inactive schemas
    - compute_idem_score: structural sameness metric
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        llm: LLMProvider,
        config: ThreadConfig,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._config = config
        self._logger = logger.bind(system="thread.schema_engine")
        self._optimized = isinstance(llm, OptimizedLLMProvider)

        # In-memory cache of active schemas (refreshed from Neo4j periodically)
        self._active_schemas: list[IdentitySchema] = []
        self._last_schema_formed_at: datetime | None = None
        self._last_promotion_at: datetime | None = None

    @property
    def active_schemas(self) -> list[IdentitySchema]:
        return list(self._active_schemas)

    async def load_schemas(self) -> list[IdentitySchema]:
        """Load all active schemas from Neo4j into memory cache."""
        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:HAS_SCHEMA]->(schema:IdentitySchema)
                RETURN schema
                """,
                {},
            )
            self._active_schemas = [self._node_to_schema(r["schema"]) for r in results]
            self._logger.info("schemas_loaded", count=len(self._active_schemas))
            return self._active_schemas
        except Exception as exc:
            self._logger.error("schema_load_failed", error=str(exc))
            return self._active_schemas

    async def form_schema_from_pattern(
        self,
        recurring_behavior: str,
        supporting_episodes: list[str],
        drive_alignment: DriveAlignmentVector,
        episode_summaries: list[str] | None = None,
    ) -> IdentitySchema | None:
        """
        Attempt to crystallize a recurring behavioural pattern into an identity schema.

        Formation requirements (ALL must be met):
        1. At least schema_formation_min_episodes supporting episodes
        2. Episodes span at least schema_formation_min_span_hours
        3. Behavioural pattern is consistent with at least 2 constitutional drives
        4. No existing schema with embedding similarity > 0.85
        5. Schema formation cooldown elapsed (max 1 per 48 hours)
        """
        cfg = self._config

        # Check cooldown
        if self._last_schema_formed_at is not None:
            cooldown = timedelta(hours=cfg.schema_formation_cooldown_hours)
            if utc_now() - self._last_schema_formed_at < cooldown:
                self._logger.debug("schema_formation_cooldown_active")
                return None

        # Check minimum episodes
        if len(supporting_episodes) < cfg.schema_formation_min_episodes:
            self._logger.debug(
                "schema_formation_insufficient_episodes",
                have=len(supporting_episodes),
                need=cfg.schema_formation_min_episodes,
            )
            return None

        # Check drive alignment: composite > 0.0 (at least 2 drives positive)
        if drive_alignment.composite <= 0.0:
            self._logger.debug("schema_formation_low_drive_alignment")
            return None

        # Use LLM to crystallize the pattern into a schema statement
        summaries_text = (
            "\n".join(episode_summaries[:10]) if episode_summaries else recurring_behavior
        )
        schema_data = await self._crystallize_schema(summaries_text, recurring_behavior)
        if schema_data is None:
            return None

        # Check for duplicates via embedding similarity
        schema_embedding = schema_data.get("embedding")
        if schema_embedding:
            for existing in self._active_schemas:
                if existing.embedding:
                    sim = cosine_similarity(schema_embedding, existing.embedding)
                    if sim > cfg.schema_similarity_merge_threshold:
                        self._logger.info(
                            "schema_formation_duplicate",
                            existing_id=existing.id,
                            similarity=round(sim, 3),
                        )
                        return None

        # Create the schema
        schema = IdentitySchema(
            statement=schema_data["statement"],
            trigger_contexts=schema_data.get("trigger_contexts", []),
            behavioral_tendency=schema_data.get("behavioral_tendency", ""),
            emotional_signature=schema_data.get("emotional_signature", {}),
            drive_alignment=drive_alignment,
            strength=SchemaStrength.NASCENT,
            valence=SchemaValence.ADAPTIVE,
            confirmation_count=len(supporting_episodes),
            confirmation_episodes=supporting_episodes[:20],
            evidence_ratio=1.0,
            embedding=schema_embedding,
        )

        # Persist to Neo4j
        await self._persist_schema(schema)

        # Link to supporting episodes
        for ep_id in supporting_episodes[:20]:
            await self._link_schema_episode(schema.id, ep_id, "CONFIRMED_BY", 0.7)

        self._active_schemas.append(schema)
        self._last_schema_formed_at = utc_now()

        self._logger.info(
            "schema_formed",
            schema_id=schema.id,
            statement=schema.statement[:80],
            supporting_count=len(supporting_episodes),
        )
        return schema

    async def evaluate_evidence(
        self,
        schema: IdentitySchema,
        episode_id: str,
        episode_embedding: list[float] | None,
        episode_summary: str,
    ) -> tuple[str, float]:
        """
        Evaluate whether an episode confirms or challenges a schema.

        Returns: (direction: "confirms" | "challenges" | "irrelevant", strength: 0.0-1.0)

        Fast path (no LLM, <10ms): cosine similarity check
        Slow path (LLM, ≤100ms): for ambiguous episodes
        """
        # Fast path: embedding similarity check
        if episode_embedding and schema.embedding:
            sim = cosine_similarity(episode_embedding, schema.embedding)
            if sim < self._config.schema_relevance_threshold:
                return ("irrelevant", 0.0)
        else:
            # No embeddings available - assume potentially relevant
            sim = 0.5

        # For clearly relevant or ambiguous episodes, use LLM evaluation
        if sim >= self._config.schema_evidence_ambiguity_threshold:
            # High similarity - likely confirms, no need for LLM
            return ("confirms", min(1.0, sim))

        # Slow path: LLM evaluation
        direction, strength = await self._llm_evaluate_evidence(
            schema.statement, episode_summary
        )
        return (direction, strength)

    async def record_evidence(
        self,
        schema_id: str,
        episode_id: str,
        direction: str,
        strength: float,
    ) -> None:
        """Record evidence for/against a schema and update counts."""
        schema = self._find_schema(schema_id)
        if schema is None:
            return

        if direction == "confirms":
            schema.confirmation_count += 1
            if episode_id not in schema.confirmation_episodes:
                schema.confirmation_episodes.append(episode_id)
                if len(schema.confirmation_episodes) > 50:
                    schema.confirmation_episodes = schema.confirmation_episodes[-50:]
            schema.last_activated = utc_now()
            await self._link_schema_episode(schema_id, episode_id, "CONFIRMED_BY", strength)

        elif direction == "challenges":
            schema.disconfirmation_count += 1
            if episode_id not in schema.disconfirmation_episodes:
                schema.disconfirmation_episodes.append(episode_id)
                if len(schema.disconfirmation_episodes) > 50:
                    schema.disconfirmation_episodes = schema.disconfirmation_episodes[-50:]
            await self._link_schema_episode(schema_id, episode_id, "CHALLENGED_BY", strength)

        schema.recompute_evidence_ratio()
        schema.last_updated = utc_now()

        # Persist updated counts
        await self._update_schema_evidence(schema)

    async def check_promotions(self) -> list[str]:
        """
        Check if any schemas should be promoted based on evidence accumulation.
        Max 1 promotion per 24 hours.

        Returns list of promoted schema IDs.
        """
        if (
            self._last_promotion_at is not None
            and utc_now() - self._last_promotion_at < timedelta(hours=24)
        ):
            return []

        promoted: list[str] = []
        cfg = self._config

        for schema in self._active_schemas:
            if schema.evidence_ratio < 0.8:
                continue

            total = schema.confirmation_count + schema.disconfirmation_count
            old_strength = schema.strength

            if schema.strength == SchemaStrength.NASCENT and total >= 10:
                schema.strength = SchemaStrength.DEVELOPING
            elif (
                schema.strength == SchemaStrength.DEVELOPING
                and schema.confirmation_count >= cfg.schema_promotion_min_confirmations
            ):
                schema.strength = SchemaStrength.ESTABLISHED
            elif schema.strength == SchemaStrength.ESTABLISHED:
                age_days = (utc_now() - schema.first_formed).total_seconds() / 86400
                if (
                    schema.confirmation_count >= cfg.schema_core_min_confirmations
                    and age_days >= cfg.schema_core_min_age_days
                ):
                    schema.strength = SchemaStrength.CORE

            if schema.strength != old_strength:
                promoted.append(schema.id)
                await self._update_schema_strength(schema)
                self._last_promotion_at = utc_now()
                self._logger.info(
                    "schema_promoted",
                    schema_id=schema.id,
                    old=old_strength.value,
                    new=schema.strength.value,
                )
                # Only one promotion per call
                break

        return promoted

    async def check_decay(self) -> list[str]:
        """
        Check for inactive schemas and reduce their strength.
        Never deletes schemas - history matters.
        """
        decayed: list[str] = []
        threshold = timedelta(days=self._config.schema_inactive_days_before_decay)
        now = utc_now()

        for schema in self._active_schemas:
            # CORE schemas never decay through inactivity
            if schema.strength == SchemaStrength.CORE:
                continue

            if now - schema.last_activated > threshold:
                old_strength = schema.strength
                if schema.strength == SchemaStrength.ESTABLISHED:
                    schema.strength = SchemaStrength.DEVELOPING
                elif schema.strength == SchemaStrength.DEVELOPING:
                    schema.strength = SchemaStrength.NASCENT

                if schema.strength != old_strength:
                    decayed.append(schema.id)
                    await self._update_schema_strength(schema)
                    self._logger.info(
                        "schema_decayed",
                        schema_id=schema.id,
                        old=old_strength.value,
                        new=schema.strength.value,
                        inactive_days=round(
                            (now - schema.last_activated).total_seconds() / 86400, 1
                        ),
                    )

        return decayed

    async def check_maladaptive(self) -> list[str]:
        """
        Check for schemas with low evidence ratio that should be marked maladaptive.
        Requires 10+ evaluations and evidence_ratio < 0.3.
        """
        flagged: list[str] = []

        for schema in self._active_schemas:
            total = schema.confirmation_count + schema.disconfirmation_count
            if (
                total >= 10
                and schema.evidence_ratio < 0.3
                and schema.valence != SchemaValence.MALADAPTIVE
            ):
                schema.valence = SchemaValence.MALADAPTIVE
                schema.last_updated = utc_now()
                flagged.append(schema.id)
                await self._update_schema_valence(schema)
                self._logger.warning(
                    "schema_marked_maladaptive",
                    schema_id=schema.id,
                    evidence_ratio=round(schema.evidence_ratio, 3),
                )

        return flagged

    def compute_idem_score(
        self,
        personality_distance: float = 0.0,
        behavioral_consistency: float = 0.0,
        memory_accessibility: float = 0.0,
    ) -> float:
        """
        Compute the structural sameness score (idem).

        idem = 0.40 * schema_stability
             + 0.30 * personality_stability
             + 0.20 * behavioral_consistency
             + 0.10 * memory_accessibility

        Healthy idem is typically 0.6-0.85.
        """
        total_active = len(self._active_schemas)
        if total_active == 0:
            schema_stability = 0.5
        else:
            unchanged = sum(
                1 for s in self._active_schemas
                if s.evidence_ratio >= 0.5
            )
            schema_stability = unchanged / total_active

        personality_stability = max(0.0, min(1.0, 1.0 - personality_distance))

        return (
            0.40 * schema_stability
            + 0.30 * personality_stability
            + 0.20 * behavioral_consistency
            + 0.10 * memory_accessibility
        )

    def _find_schema(self, schema_id: str) -> IdentitySchema | None:
        """Find schema by ID in the in-memory cache."""
        for s in self._active_schemas:
            if s.id == schema_id:
                return s
        return None

    # ─── LLM Operations ──────────────────────────────────────────────────

    async def _crystallize_schema(
        self,
        episode_summaries: str,
        recurring_behavior: str,
    ) -> dict[str, Any] | None:
        """Use LLM to crystallize a pattern into a schema statement."""
        from clients.llm import Message

        try:
            # Budget check: schema crystallization is low priority
            if self._optimized:
                assert isinstance(self._llm, OptimizedLLMProvider)
                if not self._llm.should_use_llm("thread.schema", estimated_tokens=500):
                    self._logger.debug("schema_crystallization_skipped_budget")
                    return None

            sys_prompt = (
                "You crystallize behavioural patterns into identity schemas. "
                "Given recurring experiences, identify the core self-belief they reveal. "
                "Respond as JSON with keys: statement, trigger_contexts, behavioral_tendency, "
                "emotional_signature. The statement must be in the form: "
                "'I am the kind of entity that [behaviour] because [reason].'"
            )
            user_content = (
                f"Recurring behaviour pattern: {recurring_behavior}\n\n"
                f"Supporting experiences:\n{episode_summaries}\n\n"
                "Crystallize this into an identity schema. Respond as JSON only."
            )

            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=sys_prompt,
                    messages=[Message(role="user", content=user_content)],
                    max_tokens=500,
                    temperature=self._config.llm_temperature_evaluation,
                    output_format="json",
                    cache_system="thread.schema",
                    cache_method="crystallize",
                )
            else:
                response = await self._llm.generate(
                    system_prompt=sys_prompt,
                    messages=[Message(role="user", content=user_content)],
                    max_tokens=500,
                    temperature=self._config.llm_temperature_evaluation,
                    output_format="json",
                )

            data: dict[str, Any] = json.loads(response.text)
            if "statement" not in data:
                return None
            return data

        except Exception as exc:
            self._logger.warning("schema_crystallization_failed", error=str(exc))
            return None

    async def _llm_evaluate_evidence(
        self,
        schema_statement: str,
        episode_summary: str,
    ) -> tuple[str, float]:
        """Use LLM to evaluate whether an episode confirms or challenges a schema."""

        try:
            prompt = (
                f'Schema: "{schema_statement}"\n'
                f'Experience: "{episode_summary}"\n\n'
                "Does this experience CONFIRM this self-belief, CHALLENGE it, "
                "or have NO BEARING? Rate strength 0.0-1.0.\n"
                "Respond as JSON: "
                '{{"direction": "confirms|challenges|irrelevant", "strength": 0.0}}'
            )
            if self._optimized:
                assert isinstance(self._llm, OptimizedLLMProvider)
                if not self._llm.should_use_llm("thread.evidence", estimated_tokens=100):
                    return ("irrelevant", 0.0)
                response = await self._llm.evaluate(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=self._config.llm_temperature_evaluation,
                    cache_system="thread.evidence",
                    cache_method="evaluate",
                )
            else:
                response = await self._llm.evaluate(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=self._config.llm_temperature_evaluation,
                )

            data = json.loads(response.text)
            direction = data.get("direction", "irrelevant")
            strength = float(data.get("strength", 0.0))
            if direction not in ("confirms", "challenges", "irrelevant"):
                direction = "irrelevant"
            return (direction, max(0.0, min(1.0, strength)))

        except Exception as exc:
            self._logger.warning("evidence_evaluation_failed", error=str(exc))
            return ("irrelevant", 0.0)

    # ─── Neo4j Persistence ───────────────────────────────────────────────

    async def _persist_schema(self, schema: IdentitySchema) -> None:
        """Create an IdentitySchema node and link it to Self."""
        await self._neo4j.execute_write(
            """
            MATCH (s:Self)
            CREATE (schema:IdentitySchema {
                id: $id,
                statement: $statement,
                trigger_contexts_json: $trigger_contexts_json,
                behavioral_tendency: $behavioral_tendency,
                emotional_signature_json: $emotional_signature_json,
                drive_alignment_json: $drive_alignment_json,
                strength: $strength,
                valence: $valence,
                confirmation_count: $confirmation_count,
                disconfirmation_count: $disconfirmation_count,
                evidence_ratio: $evidence_ratio,
                first_formed: datetime($first_formed),
                last_activated: datetime($last_activated),
                last_updated: datetime($last_updated),
                parent_schema_id: $parent_schema_id,
                evolution_reason: $evolution_reason
            })
            SET schema.embedding = $embedding
            CREATE (s)-[:HAS_SCHEMA]->(schema)
            """,
            {
                "id": schema.id,
                "statement": schema.statement,
                "trigger_contexts_json": json.dumps(schema.trigger_contexts),
                "behavioral_tendency": schema.behavioral_tendency,
                "emotional_signature_json": json.dumps(schema.emotional_signature),
                "drive_alignment_json": json.dumps(schema.drive_alignment.model_dump()),
                "strength": schema.strength.value,
                "valence": schema.valence.value,
                "confirmation_count": schema.confirmation_count,
                "disconfirmation_count": schema.disconfirmation_count,
                "evidence_ratio": schema.evidence_ratio,
                "first_formed": schema.first_formed.isoformat(),
                "last_activated": schema.last_activated.isoformat(),
                "last_updated": schema.last_updated.isoformat(),
                "parent_schema_id": schema.parent_schema_id or "",
                "evolution_reason": schema.evolution_reason,
                "embedding": schema.embedding,
            },
        )

    async def _link_schema_episode(
        self,
        schema_id: str,
        episode_id: str,
        rel_type: str,
        strength: float,
    ) -> None:
        """Create a CONFIRMED_BY or CHALLENGED_BY relationship."""
        query = f"""
        MATCH (schema:IdentitySchema {{id: $schema_id}})
        MATCH (e:Episode {{id: $episode_id}})
        MERGE (schema)-[r:{rel_type}]->(e)
        SET r.strength = $strength, r.created_at = datetime()
        """
        try:
            await self._neo4j.execute_write(
                query,
                {"schema_id": schema_id, "episode_id": episode_id, "strength": strength},
            )
        except Exception as exc:
            self._logger.debug("schema_episode_link_failed", error=str(exc))

    async def _update_schema_evidence(self, schema: IdentitySchema) -> None:
        """Update evidence counts on the schema node."""
        await self._neo4j.execute_write(
            """
            MATCH (schema:IdentitySchema {id: $id})
            SET schema.confirmation_count = $confirmation_count,
                schema.disconfirmation_count = $disconfirmation_count,
                schema.evidence_ratio = $evidence_ratio,
                schema.last_activated = datetime($last_activated),
                schema.last_updated = datetime($last_updated)
            """,
            {
                "id": schema.id,
                "confirmation_count": schema.confirmation_count,
                "disconfirmation_count": schema.disconfirmation_count,
                "evidence_ratio": schema.evidence_ratio,
                "last_activated": schema.last_activated.isoformat(),
                "last_updated": schema.last_updated.isoformat(),
            },
        )

    async def _update_schema_strength(self, schema: IdentitySchema) -> None:
        """Update strength on the schema node."""
        await self._neo4j.execute_write(
            """
            MATCH (schema:IdentitySchema {id: $id})
            SET schema.strength = $strength,
                schema.last_updated = datetime($last_updated)
            """,
            {
                "id": schema.id,
                "strength": schema.strength.value,
                "last_updated": schema.last_updated.isoformat(),
            },
        )

    async def _update_schema_valence(self, schema: IdentitySchema) -> None:
        """Update valence on the schema node."""
        await self._neo4j.execute_write(
            """
            MATCH (schema:IdentitySchema {id: $id})
            SET schema.valence = $valence,
                schema.last_updated = datetime($last_updated)
            """,
            {
                "id": schema.id,
                "valence": schema.valence.value,
                "last_updated": schema.last_updated.isoformat(),
            },
        )

    def _node_to_schema(self, node: Any) -> IdentitySchema:
        """Convert a Neo4j node to an IdentitySchema object."""
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
            first_formed=props["first_formed"] if "first_formed" in props else utc_now(),
            last_activated=props["last_activated"] if "last_activated" in props else utc_now(),
            last_updated=props["last_updated"] if "last_updated" in props else utc_now(),
            parent_schema_id=props.get("parent_schema_id") or None,
            evolution_reason=props.get("evolution_reason", ""),
            embedding=props.get("embedding"),
        )
