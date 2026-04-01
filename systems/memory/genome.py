"""
EcodiaOS - Memory Genome Extraction & Seeding

Implements GenomeExtractionProtocol for the Memory system. Extracts the
heritable core of the knowledge graph - high-salience entities, semantic
relations, community summaries, the Self node's personality vector, and
consolidated episode summaries - so a child instance inherits the parent's
accumulated understanding without raw episode baggage.

Heritable state (what crosses the generational boundary):
    - Core entity graph (top 200 entities by salience)
    - Semantic relations between those entities
    - Community structure summaries
    - Self-node personality vector
    - Consolidated episode summaries (top 50 by salience)

NOT heritable:
    - Raw episodes (too large, too instance-specific)
    - Affect state (child starts fresh)
    - Embeddings (recomputed by child's embedding model)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, verify_segment, check_schema_version

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# ─── Extraction Limits ──────────────────────────────────────────────────────

_MAX_ENTITIES: int = 200
_MAX_RELATIONS: int = 500
_MAX_COMMUNITIES: int = 50
_MAX_CORE_SUMMARIES: int = 50
_MIN_ENTITY_SALIENCE: float = 0.1


class MemoryGenomeExtractor:
    """
    Extracts and seeds the Memory system's heritable state via the
    organism-wide GenomeExtractionProtocol.

    Constructor takes a Neo4jClient. All Neo4j queries are wrapped in
    try/except to ensure graceful degradation.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(subsystem="memory.genome")

    # ─── GenomeExtractionProtocol ────────────────────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Serialise the Memory system's heritable state into an OrganGenomeSegment.

        Queries Neo4j for high-salience entities, their semantic relations,
        community summaries, the Self node's personality vector, drive weight
        history, floor threshold snapshots, and top consolidated episode
        summaries. Returns an empty segment (version=0) if no meaningful
        state exists yet.

        Drive weights and floor thresholds are now evolvable parameters -
        children inherit the parent's constitutional phenotype so they do not
        start from a fixed baseline. Spec: Philosophical Context §genome.
        """
        entities = await self._extract_entities()
        relations = await self._extract_relations(entities)
        communities = await self._extract_communities()
        personality_vector = await self._extract_personality_vector()
        drive_weight_history = await self._extract_drive_weight_history()
        floor_threshold_snapshots = await self._extract_floor_threshold_snapshots()
        core_summaries = await self._extract_core_summaries()

        # Check whether there is meaningful state to inherit
        has_state = bool(entities) or bool(personality_vector) or bool(core_summaries)
        if not has_state:
            self._log.info("memory_genome_empty", reason="no_meaningful_state")
            return build_segment(
                system_id=SystemID.MEMORY,
                payload={
                    "entities": [],
                    "relations": [],
                    "communities": [],
                    "personality_vector": [],
                    "drive_weight_history": [],
                    "floor_threshold_snapshots": [],
                    "core_summaries": [],
                },
                version=0,
            )

        payload: dict[str, Any] = {
            "entities": entities,
            "relations": relations,
            "communities": communities,
            "personality_vector": personality_vector,
            "drive_weight_history": drive_weight_history,
            "floor_threshold_snapshots": floor_threshold_snapshots,
            "core_summaries": core_summaries,
        }

        segment = build_segment(
            system_id=SystemID.MEMORY,
            payload=payload,
            version=2,
        )

        self._log.info(
            "memory_genome_extracted",
            entities=len(entities),
            relations=len(relations),
            communities=len(communities),
            personality_dims=len(personality_vector),
            drive_weight_history_entries=len(drive_weight_history),
            floor_threshold_snapshots=len(floor_threshold_snapshots),
            summaries=len(core_summaries),
            size_bytes=segment.size_bytes,
        )

        return segment

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """
        Restore Memory heritable state from a parent's genome segment.

        Verifies payload_hash integrity and schema_version compatibility
        before writing any data. Pre-populates the entity graph, community
        summaries, and personality vector on the Self node.

        Returns True on success, False on any failure.
        """
        # Integrity checks
        if not check_schema_version(segment):
            self._log.error(
                "memory_genome_seed_rejected",
                reason="incompatible_schema_version",
                schema_version=segment.schema_version,
            )
            return False

        if not verify_segment(segment):
            self._log.error(
                "memory_genome_seed_rejected",
                reason="payload_hash_mismatch",
            )
            return False

        payload = segment.payload
        entities: list[dict[str, Any]] = payload.get("entities", [])
        relations: list[dict[str, Any]] = payload.get("relations", [])
        communities: list[dict[str, Any]] = payload.get("communities", [])
        personality_vector: list[float] = payload.get("personality_vector", [])
        drive_weight_history: list[dict[str, Any]] = payload.get("drive_weight_history", [])
        floor_threshold_snapshots: list[dict[str, Any]] = payload.get("floor_threshold_snapshots", [])
        core_summaries: list[dict[str, Any]] = payload.get("core_summaries", [])

        # Seed in dependency order: entities first, then relations
        entities_seeded = await self._seed_entities(entities)
        relations_seeded = await self._seed_relations(relations)
        communities_seeded = await self._seed_communities(communities)
        personality_seeded = await self._seed_personality_vector(personality_vector)
        dwh_seeded = await self._seed_drive_weight_history(drive_weight_history)
        floor_seeded = await self._seed_floor_threshold_snapshots(floor_threshold_snapshots)
        summaries_seeded = await self._seed_core_summaries(core_summaries)

        success = entities_seeded or personality_seeded or summaries_seeded

        self._log.info(
            "memory_genome_seeded",
            entities_seeded=entities_seeded,
            relations_seeded=relations_seeded,
            communities_seeded=communities_seeded,
            personality_seeded=personality_seeded,
            drive_weight_history_seeded=dwh_seeded,
            floor_threshold_seeded=floor_seeded,
            summaries_seeded=summaries_seeded,
            success=success,
        )

        return success

    # ─── Extraction Helpers ──────────────────────────────────────────────────

    async def _extract_entities(self) -> list[dict[str, Any]]:
        """Extract top entities by salience score, excluding embeddings."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (e:Entity)
                WHERE e.salience_score >= $min_salience
                RETURN e.id AS id,
                       e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       e.salience_score AS salience_score,
                       e.mention_count AS mention_count,
                       e.is_core_identity AS is_core_identity
                ORDER BY e.salience_score DESC
                LIMIT $limit
                """,
                {"min_salience": _MIN_ENTITY_SALIENCE, "limit": _MAX_ENTITIES},
            )
        except Exception as exc:
            self._log.warning("genome_extract_entities_failed", error=str(exc))
            return []

        entities: list[dict[str, Any]] = []
        for row in rows:
            entities.append({
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "type": str(row.get("type", "concept")),
                "description": str(row.get("description", "")),
                "salience_score": float(row.get("salience_score", 0.0)),
                "mention_count": int(row.get("mention_count", 0)),
                "is_core_identity": bool(row.get("is_core_identity", False)),
            })
        return entities

    async def _extract_relations(
        self, entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract semantic relations between the extracted entities."""
        if not entities:
            return []

        entity_ids = [e["id"] for e in entities]

        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                WHERE a.id IN $entity_ids AND b.id IN $entity_ids
                RETURN a.id AS source_id,
                       b.id AS target_id,
                       r.type AS relation_type,
                       coalesce(r.strength, 1.0) AS weight,
                       coalesce(r.description, '') AS description
                ORDER BY r.strength DESC
                LIMIT $limit
                """,
                {"entity_ids": entity_ids, "limit": _MAX_RELATIONS},
            )
        except Exception as exc:
            self._log.warning("genome_extract_relations_failed", error=str(exc))
            return []

        relations: list[dict[str, Any]] = []
        for row in rows:
            relations.append({
                "source_id": str(row.get("source_id", "")),
                "target_id": str(row.get("target_id", "")),
                "relation_type": str(row.get("relation_type", "")),
                "weight": float(row.get("weight", 1.0)),
                "description": str(row.get("description", "")),
            })
        return relations

    async def _extract_communities(self) -> list[dict[str, Any]]:
        """Extract community structure summaries."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (c:Community)
                RETURN c.id AS id,
                       c.name AS name,
                       c.summary AS summary,
                       c.member_count AS member_count,
                       c.coherence_score AS coherence_score
                ORDER BY c.member_count DESC
                LIMIT $limit
                """,
                {"limit": _MAX_COMMUNITIES},
            )
        except Exception as exc:
            self._log.warning("genome_extract_communities_failed", error=str(exc))
            return []

        communities: list[dict[str, Any]] = []
        for row in rows:
            communities.append({
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "summary": str(row.get("summary", "")),
                "member_count": int(row.get("member_count", 0)),
                "coherence_score": float(row.get("coherence_score", 0.0)),
            })
        return communities

    async def _extract_personality_vector(self) -> list[float]:
        """Extract the personality vector from the Self node."""
        try:
            rows = await self._neo4j.execute_read(
                "MATCH (s:Self) RETURN s.personality_vector AS pv LIMIT 1"
            )
        except Exception as exc:
            self._log.warning("genome_extract_personality_failed", error=str(exc))
            return []

        if not rows:
            return []

        pv = rows[0].get("pv")
        if pv is None:
            return []

        # Neo4j may return as a list or a JSON string
        if isinstance(pv, str):
            try:
                pv = json.loads(pv)
            except (json.JSONDecodeError, TypeError):
                self._log.warning("genome_personality_parse_failed")
                return []

        if isinstance(pv, list):
            return [float(v) for v in pv]

        return []

    async def _extract_drive_weight_history(self) -> list[dict[str, Any]]:
        """
        Extract the drive weight update history from the Self node.

        Written by `MemoryService.update_personality_from_evo()` as a JSON
        list on `Self.drive_weight_history`. Returns the last 100 entries so
        the child can reconstruct the parent's constitutional drift trajectory
        without inheriting the full history (which may exceed genome budget).

        Children seed from this to understand which drive directions have been
        rewarded over the parent's lifetime - effectively inheriting learned
        constitutional preferences.
        """
        try:
            rows = await self._neo4j.execute_read(
                "MATCH (s:Self) RETURN s.drive_weight_history AS dwh LIMIT 1"
            )
        except Exception as exc:
            self._log.warning("genome_extract_drive_weight_history_failed", error=str(exc))
            return []

        if not rows:
            return []

        raw = rows[0].get("dwh")
        if raw is None:
            return []

        if isinstance(raw, str):
            try:
                parsed: list[dict[str, Any]] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return []
        elif isinstance(raw, list):
            parsed = raw
        else:
            return []

        # Return only the last 100 entries - enough to reconstruct trajectory
        return parsed[-100:] if len(parsed) > 100 else parsed

    async def _extract_floor_threshold_snapshots(self) -> list[dict[str, Any]]:
        """
        Extract snapshots of the Care and Honesty floor thresholds.

        Floor thresholds define the minimum acceptable drive scores for PERMIT
        verdicts from Equor. They are evolvable: Evo may shift them based on
        hypothesis outcomes. The latest snapshot is what matters for child seeding
        - the child inherits the parent's evolved floor, not the factory default.

        Reads from `Constitution.floor_care` and `Constitution.floor_honesty`
        (written by Equor when thresholds drift). Falls back to querying
        `GovernanceRecord` nodes tagged `event_type='floor_threshold_updated'`
        if the Constitution node does not carry the field directly.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                RETURN c.drive_care AS care,
                       c.drive_honesty AS honesty,
                       c.floor_care AS floor_care,
                       c.floor_honesty AS floor_honesty
                LIMIT 1
                """
            )
        except Exception as exc:
            self._log.warning("genome_extract_floor_thresholds_failed", error=str(exc))
            return []

        if not rows:
            return []

        row = rows[0]
        snapshots: list[dict[str, Any]] = []

        floor_care = row.get("floor_care")
        floor_honesty = row.get("floor_honesty")
        current_care = row.get("care")
        current_honesty = row.get("honesty")

        # Only emit a snapshot if at least one floor value is present
        if floor_care is not None or floor_honesty is not None:
            snapshots.append({
                "source": "constitution_node",
                "floor_care": float(floor_care) if floor_care is not None else None,
                "floor_honesty": float(floor_honesty) if floor_honesty is not None else None,
                "current_care": float(current_care) if current_care is not None else None,
                "current_honesty": float(current_honesty) if current_honesty is not None else None,
            })

        # Also pull the most recent GovernanceRecord update for audit trail
        try:
            gov_rows = await self._neo4j.execute_read(
                """
                MATCH (g:GovernanceRecord)
                WHERE g.event_type = 'floor_threshold_updated'
                RETURN g.details AS details, toString(g.timestamp) AS ts
                ORDER BY g.timestamp DESC
                LIMIT 5
                """
            )
            for gr in gov_rows:
                details_raw = gr.get("details", "")
                if isinstance(details_raw, str):
                    try:
                        details = json.loads(details_raw)
                    except Exception:
                        details = {"raw": details_raw}
                else:
                    details = dict(details_raw) if details_raw else {}
                details["source"] = "governance_record"
                details["ts"] = gr.get("ts", "")
                snapshots.append(details)
        except Exception:
            pass

        return snapshots

    async def _extract_core_summaries(self) -> list[dict[str, Any]]:
        """
        Extract top consolidated episode summaries by salience.

        Only includes consolidated summaries (not raw episodes) to keep
        the genome compact and instance-independent.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE e.is_consolidated = true
                  AND e.summary IS NOT NULL
                  AND e.summary <> ''
                RETURN e.id AS id,
                       e.summary AS summary,
                       e.salience_composite AS salience,
                       e.source AS source,
                       e.event_time AS event_time
                ORDER BY e.salience_composite DESC
                LIMIT $limit
                """,
                {"limit": _MAX_CORE_SUMMARIES},
            )
        except Exception as exc:
            self._log.warning("genome_extract_summaries_failed", error=str(exc))
            return []

        summaries: list[dict[str, Any]] = []
        for row in rows:
            event_time = row.get("event_time")
            if hasattr(event_time, "isoformat"):
                event_time = event_time.isoformat()
            elif hasattr(event_time, "to_native"):
                event_time = event_time.to_native().isoformat()
            else:
                event_time = str(event_time) if event_time else ""

            summaries.append({
                "id": str(row.get("id", "")),
                "summary": str(row.get("summary", "")),
                "salience": float(row.get("salience", 0.0)),
                "source": str(row.get("source", "")),
                "event_time": event_time,
            })
        return summaries

    # ─── Seeding Helpers ─────────────────────────────────────────────────────

    async def _seed_entities(self, entities: list[dict[str, Any]]) -> int:
        """Pre-populate entities in the child's knowledge graph."""
        seeded = 0
        for ent in entities:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (e:Entity {
                        id: $id,
                        name: $name,
                        type: $type,
                        description: $description,
                        salience_score: $salience_score,
                        mention_count: $mention_count,
                        is_core_identity: $is_core_identity,
                        source: "parent_genome",
                        created_at: datetime()
                    })
                    """,
                    {
                        "id": ent["id"],
                        "name": ent["name"],
                        "type": ent["type"],
                        "description": ent["description"],
                        "salience_score": ent["salience_score"],
                        "mention_count": ent["mention_count"],
                        "is_core_identity": ent.get("is_core_identity", False),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "seed_entity_failed",
                    entity_name=ent.get("name"),
                    error=str(exc),
                )
        return seeded

    async def _seed_relations(self, relations: list[dict[str, Any]]) -> int:
        """Re-create semantic relations between seeded entities."""
        seeded = 0
        for rel in relations:
            try:
                await self._neo4j.execute_write(
                    """
                    MATCH (a:Entity {id: $source_id})
                    MATCH (b:Entity {id: $target_id})
                    MERGE (a)-[r:RELATES_TO {type: $relation_type}]->(b)
                    SET r.strength = $weight,
                        r.description = $description,
                        r.source = "parent_genome",
                        r.created_at = datetime()
                    """,
                    {
                        "source_id": rel["source_id"],
                        "target_id": rel["target_id"],
                        "relation_type": rel["relation_type"],
                        "weight": rel["weight"],
                        "description": rel["description"],
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "seed_relation_failed",
                    source=rel.get("source_id"),
                    target=rel.get("target_id"),
                    error=str(exc),
                )
        return seeded

    async def _seed_communities(self, communities: list[dict[str, Any]]) -> int:
        """Re-create community summary nodes."""
        seeded = 0
        for comm in communities:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (c:Community {
                        id: $id,
                        name: $name,
                        summary: $summary,
                        member_count: $member_count,
                        coherence_score: $coherence_score,
                        source: "parent_genome",
                        created_at: datetime()
                    })
                    """,
                    {
                        "id": comm["id"],
                        "name": comm["name"],
                        "summary": comm["summary"],
                        "member_count": comm["member_count"],
                        "coherence_score": comm["coherence_score"],
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "seed_community_failed",
                    community_name=comm.get("name"),
                    error=str(exc),
                )
        return seeded

    async def _seed_personality_vector(self, personality_vector: list[float]) -> bool:
        """Set the personality vector on the child's Self node."""
        if not personality_vector:
            return False

        try:
            await self._neo4j.execute_write(
                """
                MATCH (s:Self)
                SET s.personality_vector = $pv
                """,
                {"pv": personality_vector},
            )
            self._log.info(
                "personality_vector_seeded",
                dimensions=len(personality_vector),
            )
            return True
        except Exception as exc:
            self._log.warning("seed_personality_failed", error=str(exc))
            return False

    async def _seed_drive_weight_history(
        self, history: list[dict[str, Any]],
    ) -> bool:
        """
        Initialise the child's drive weight history from the parent's.

        Written to `Self.drive_weight_history` as a JSON string - the child
        starts with visibility into the parent's constitutional drift so
        `update_personality_from_evo()` can continue the trajectory rather
        than starting from zero.
        """
        if not history:
            return False

        try:
            await self._neo4j.execute_write(
                """
                MATCH (s:Self)
                SET s.drive_weight_history = $dwh,
                    s.drive_weight_history_inherited = true
                """,
                {"dwh": json.dumps(history)},
            )
            self._log.info("drive_weight_history_seeded", entries=len(history))
            return True
        except Exception as exc:
            self._log.warning("seed_drive_weight_history_failed", error=str(exc))
            return False

    async def _seed_floor_threshold_snapshots(
        self, snapshots: list[dict[str, Any]],
    ) -> bool:
        """
        Apply the parent's most recent floor threshold snapshot to the child's Constitution.

        Only the most recent snapshot (index 0, ordered DESC by ts in extraction)
        is applied - the child inherits the evolved floor, not the full history.
        If the Constitution node carries `floor_care` / `floor_honesty` fields
        from the snapshot, Equor will use those instead of the factory defaults.
        """
        if not snapshots:
            return False

        # Take the first snapshot that comes from the constitution_node source
        # (most authoritative), falling back to the first available.
        snapshot: dict[str, Any] = {}
        for s in snapshots:
            if s.get("source") == "constitution_node":
                snapshot = s
                break
        if not snapshot:
            snapshot = snapshots[0]

        floor_care = snapshot.get("floor_care")
        floor_honesty = snapshot.get("floor_honesty")

        if floor_care is None and floor_honesty is None:
            return False

        try:
            params: dict[str, Any] = {}
            set_clauses: list[str] = ["c.floor_inherited = true"]
            if floor_care is not None:
                set_clauses.append("c.floor_care = $floor_care")
                params["floor_care"] = float(floor_care)
            if floor_honesty is not None:
                set_clauses.append("c.floor_honesty = $floor_honesty")
                params["floor_honesty"] = float(floor_honesty)

            await self._neo4j.execute_write(
                f"""
                MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                SET {', '.join(set_clauses)}
                """,
                params,
            )
            self._log.info(
                "floor_thresholds_seeded",
                floor_care=floor_care,
                floor_honesty=floor_honesty,
            )
            return True
        except Exception as exc:
            self._log.warning("seed_floor_thresholds_failed", error=str(exc))
            return False

    async def _seed_core_summaries(self, summaries: list[dict[str, Any]]) -> int:
        """Pre-populate consolidated episode summaries in the child graph."""
        seeded = 0
        for s in summaries:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (e:Episode {
                        id: $id,
                        summary: $summary,
                        salience_composite: $salience,
                        source: $source,
                        is_consolidated: true,
                        inherited: true,
                        source_genome: "parent_genome",
                        created_at: datetime()
                    })
                    """,
                    {
                        "id": s["id"],
                        "summary": s["summary"],
                        "salience": s["salience"],
                        "source": s["source"],
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "seed_summary_failed",
                    summary_id=s.get("id"),
                    error=str(exc),
                )
        return seeded
