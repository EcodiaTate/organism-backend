"""
EcodiaOS - Memory System REST Router

Exposes episodic, semantic, and belief stores to the Next.js frontend.

Endpoints:
  GET  /api/v1/memory/stats                                  - Graph statistics
  GET  /api/v1/memory/health                                 - Health check
  GET  /api/v1/memory/self                                   - Self node
  GET  /api/v1/memory/constitution                           - Constitutional drives
  GET  /api/v1/memory/episodes                               - Recent episodes (limit, salience floor, modality)
  GET  /api/v1/memory/episode/{id}                           - Single episode detail
  GET  /api/v1/memory/entities                               - Salient entities
  GET  /api/v1/memory/entity/{id}                            - Entity + neighbours
  GET  /api/v1/memory/beliefs                                - Persisted beliefs (by domain)
  GET  /api/v1/memory/beliefs/{id}/decay-forecast            - 30-day precision decay forecast
  GET  /api/v1/memory/communities                            - Community clusters
  GET  /api/v1/memory/consolidation                          - Last consolidation result
  POST /api/v1/memory/consolidate                            - Trigger consolidation
  POST /api/v1/memory/retrieve                               - Hybrid retrieval search
  GET  /api/v1/memory/counterfactuals                        - List counterfactual episodes
  GET  /api/v1/memory/counterfactuals/{episode_id}           - Single counterfactual + outcome
  POST /api/v1/memory/counterfactuals/{episode_id}/resolve   - Resolve counterfactual
  GET  /api/v1/memory/compression/stats                      - Per-community compression stats
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import Field

from primitives.common import EOSBaseModel

logger = structlog.get_logger("api.memory")

router = APIRouter()

# ─── Response Models ──────────────────────────────────────────────


class MemoryStatsResponse(EOSBaseModel):
    total_episodes: int = 0
    total_entities: int = 0
    total_communities: int = 0
    total_beliefs: int = 0
    cycle_count: int = 0
    instance_id: str | None = None


class MemoryHealthResponse(EOSBaseModel):
    status: str = "unknown"
    neo4j_connected: bool = False
    latency_ms: float | None = None
    episode_count: int = 0
    entity_count: int = 0
    error: str | None = None


class SelfNodeResponse(EOSBaseModel):
    instance_id: str
    name: str
    born_at: str
    cycle_count: int = 0
    total_episodes: int = 0
    total_entities: int = 0
    total_communities: int = 0
    autonomy_level: int = 0
    current_affect: dict[str, float] = Field(default_factory=dict)
    personality: dict[str, float] = Field(default_factory=dict)


class ConstitutionResponse(EOSBaseModel):
    id: str
    version: int = 1
    drive_coherence: float = 0.0
    drive_care: float = 0.0
    drive_growth: float = 0.0
    drive_honesty: float = 0.0
    last_amended: str | None = None
    amendment_count: int = 0


class EpisodeItem(EOSBaseModel):
    id: str
    event_time: str
    ingestion_time: str
    source: str = ""
    modality: str = ""
    summary: str = ""
    salience_composite: float = 0.0
    affect_valence: float = 0.0
    affect_arousal: float = 0.0
    consolidation_level: int = 0
    access_count: int = 0
    free_energy: float = 0.0
    salience_scores: dict[str, float] = Field(default_factory=dict)


class EntityItem(EOSBaseModel):
    id: str
    name: str
    type: str = ""
    description: str = ""
    salience_score: float = 0.0
    mention_count: int = 0
    confidence: float = 0.0
    is_core_identity: bool = False
    first_seen: str = ""
    last_updated: str = ""
    community_ids: list[str] = Field(default_factory=list)


class EntityNeighbour(EOSBaseModel):
    entity_id: str
    name: str
    type: str = ""
    relation_type: str = ""
    strength: float = 0.0
    direction: str = "out"


class EntityDetailResponse(EOSBaseModel):
    entity: EntityItem
    neighbours: list[EntityNeighbour] = Field(default_factory=list)


class BeliefItem(EOSBaseModel):
    id: str
    domain: str = ""
    statement: str = ""
    precision: float = 0.0
    half_life_days: float | None = None
    last_verified: str = ""
    created_at: str = ""


class CommunityItem(EOSBaseModel):
    id: str
    level: int = 0
    summary: str = ""
    member_count: int = 0
    coherence_score: float = 0.0
    salience_score: float = 0.0
    created_at: str = ""
    last_recomputed: str = ""


class ConsolidationResponse(EOSBaseModel):
    ran_at: str | None = None
    episodes_decayed: int = 0
    entities_decayed: int = 0
    communities_detected: int = 0
    episodes_compressed: int = 0
    near_duplicates_flagged: int = 0
    duration_s: float | None = None
    status: str = "never_run"


class RetrievalResultItem(EOSBaseModel):
    node_id: str
    node_type: str = "episode"
    content: str = ""
    unified_score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    graph_score: float | None = None
    salience: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(EOSBaseModel):
    query: str
    results: list[RetrievalResultItem] = Field(default_factory=list)
    entity_count: int = 0
    community_count: int = 0
    retrieval_time_ms: int = 0


class RetrieveRequest(EOSBaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=50)
    salience_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    include_communities: bool = False
    traversal_depth: int = Field(default=2, ge=1, le=4)


class CounterfactualItem(EOSBaseModel):
    id: str
    event_time: str
    summary: str = ""
    policy_name: str = ""
    policy_type: str = ""
    efe_total: float = 0.0
    estimated_pragmatic_value: float = 0.0
    estimated_epistemic_value: float = 0.0
    chosen_policy_name: str = ""
    chosen_efe_total: float = 0.0
    resolved: bool = False
    regret: float | None = None
    actual_outcome_success: bool | None = None
    resolved_at: str | None = None
    outcome_episode_id: str | None = None
    outcome_episode_summary: str | None = None


class CounterfactualResolveRequest(EOSBaseModel):
    outcome_episode_id: str
    outcome_success: bool = True
    actual_pragmatic_value: float = 0.0
    regret: float = 0.0


class CompressionStatItem(EOSBaseModel):
    community_id: str
    K: int = 0
    variance_retained: float = 0.0
    quality_score: float = 0.0
    compression_ratio: float = 0.0
    compressed_at: str = ""


class DecayForecastPoint(EOSBaseModel):
    day: int
    projected_precision: float


# In-memory consolidation cache - populated by POST /consolidate
_last_consolidation: dict[str, Any] = {}

# ─── Routes ──────────────────────────────────────────────────────


@router.get("/api/v1/memory/health", response_model=MemoryHealthResponse)
async def get_memory_health(request: Request) -> MemoryHealthResponse:
    """Quick health check - Neo4j connectivity + node counts."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        return MemoryHealthResponse(status="unavailable", neo4j_connected=False)

    try:
        result = await memory.health()
        return MemoryHealthResponse(
            status=result.get("status", "unknown"),
            neo4j_connected=result.get("neo4j_connected", False),
            latency_ms=result.get("latency_ms"),
            episode_count=result.get("episode_count", 0),
            entity_count=result.get("entity_count", 0),
        )
    except Exception as exc:
        logger.warning("memory_health_failed", error=str(exc))
        return MemoryHealthResponse(status="error", neo4j_connected=False, error=str(exc))


@router.get("/api/v1/memory/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(request: Request) -> MemoryStatsResponse:
    """Graph statistics - total node counts per label."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        return MemoryStatsResponse()

    try:
        raw = await memory.stats()
        self_node = await memory.get_self()
        return MemoryStatsResponse(
            total_episodes=raw.get("episodes", 0),
            total_entities=raw.get("entities", 0),
            total_communities=raw.get("communities", 0),
            total_beliefs=raw.get("beliefs", 0),
            cycle_count=getattr(self_node, "cycle_count", 0) if self_node else 0,
            instance_id=getattr(self_node, "instance_id", None) if self_node else None,
        )
    except Exception as exc:
        logger.warning("memory_stats_failed", error=str(exc))
        return MemoryStatsResponse()


@router.get("/api/v1/memory/self", response_model=SelfNodeResponse)
async def get_memory_self(request: Request) -> SelfNodeResponse:
    """Return the Self node - identity, affect, and cognitive counters."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    self_node = await memory.get_self()
    if self_node is None:
        raise HTTPException(
            status_code=404, detail="Self node not found - instance not born yet"
        )

    born_at = self_node.born_at
    born_at_str = born_at.isoformat() if hasattr(born_at, "isoformat") else str(born_at)

    return SelfNodeResponse(
        instance_id=self_node.instance_id,
        name=self_node.name,
        born_at=born_at_str,
        cycle_count=self_node.cycle_count,
        total_episodes=self_node.total_episodes,
        total_entities=self_node.total_entities,
        total_communities=self_node.total_communities,
        autonomy_level=self_node.autonomy_level,
        current_affect=self_node.current_affect or {},
        personality=getattr(self_node, "personality_json", {}) or {},
    )


@router.get("/api/v1/memory/constitution", response_model=ConstitutionResponse)
async def get_memory_constitution(request: Request) -> ConstitutionResponse:
    """Return the constitutional drive state."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    raw = await memory.get_constitution()
    if raw is None:
        raise HTTPException(status_code=404, detail="Constitution not found")

    return ConstitutionResponse(
        id=raw.get("id", ""),
        version=raw.get("version", 1),
        drive_coherence=raw.get("drive_coherence", 0.0),
        drive_care=raw.get("drive_care", 0.0),
        drive_growth=raw.get("drive_growth", 0.0),
        drive_honesty=raw.get("drive_honesty", 0.0),
        last_amended=raw.get("last_amended"),
        amendment_count=len(raw.get("amendments", [])),
    )


@router.get("/api/v1/memory/episodes", response_model=list[EpisodeItem])
async def get_recent_episodes(
    request: Request,
    limit: int = Query(default=30, ge=1, le=200),
    min_salience: float = Query(default=0.0, ge=0.0, le=1.0),
    modality: str | None = Query(default=None, max_length=64),
) -> list[EpisodeItem]:
    """Recent episodes ordered by ingestion time, with optional salience floor and modality filter."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        return []

    try:
        if modality:
            raw_rows = await memory._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE e.salience_composite >= $min_salience
                  AND e.modality = $modality
                RETURN e
                ORDER BY e.ingestion_time DESC
                LIMIT $limit
                """,
                {"min_salience": min_salience, "limit": limit, "modality": modality},
            )
            raw_episodes: list[dict[str, Any]] = [r["e"] for r in raw_rows]
        else:
            raw_episodes = await memory.get_recent_episodes(
                limit=limit, min_salience=min_salience
            )

        return [
            EpisodeItem(
                id=ep.get("id", ""),
                event_time=str(ep.get("event_time", "")),
                ingestion_time=str(ep.get("ingestion_time", "")),
                source=ep.get("source", ""),
                modality=ep.get("modality", ""),
                summary=ep.get("summary", ""),
                salience_composite=ep.get("salience_composite", 0.0),
                affect_valence=ep.get("affect_valence", 0.0),
                affect_arousal=ep.get("affect_arousal", 0.0),
                consolidation_level=ep.get("consolidation_level", 0),
                access_count=ep.get("access_count", 0),
                free_energy=ep.get("free_energy", 0.0),
                salience_scores=ep.get("salience_scores", {}),
            )
            for ep in raw_episodes
        ]
    except Exception as exc:
        logger.warning("memory_episodes_failed", error=str(exc))
        return []


@router.get("/api/v1/memory/episode/{episode_id}", response_model=EpisodeItem)
async def get_episode_detail(request: Request, episode_id: str) -> EpisodeItem:
    """Single episode detail by ID."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        ep = await memory.get_episode(episode_id)
        if ep is None:
            raise HTTPException(status_code=404, detail="Episode not found")

        return EpisodeItem(
            id=ep.get("id", ""),
            event_time=str(ep.get("event_time", "")),
            ingestion_time=str(ep.get("ingestion_time", "")),
            source=ep.get("source", ""),
            modality=ep.get("modality", ""),
            summary=ep.get("summary", ""),
            salience_composite=ep.get("salience_composite", 0.0),
            affect_valence=ep.get("affect_valence", 0.0),
            affect_arousal=ep.get("affect_arousal", 0.0),
            consolidation_level=ep.get("consolidation_level", 0),
            access_count=ep.get("access_count", 0),
            free_energy=ep.get("free_energy", 0.0),
            salience_scores=ep.get("salience_scores", {}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "memory_episode_detail_failed", episode_id=episode_id, error=str(exc)
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/v1/memory/entities", response_model=list[EntityItem])
async def get_entities(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    core_only: bool = Query(default=False),
) -> list[EntityItem]:
    """Return entities ordered by salience score."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        return []

    try:
        if core_only:
            cypher = (
                "MATCH (e:Entity) WHERE e.is_core_identity = true"
                " RETURN e ORDER BY e.salience_score DESC LIMIT $limit"
            )
        else:
            cypher = (
                "MATCH (e:Entity)"
                " RETURN e ORDER BY e.salience_score DESC LIMIT $limit"
            )
        result = await memory._neo4j.query(cypher, {"limit": limit})
        entities: list[EntityItem] = []
        for row in result:
            e = row.get("e", {})
            if not e:
                continue
            entities.append(
                EntityItem(
                    id=e.get("id", ""),
                    name=e.get("name", ""),
                    type=e.get("type", ""),
                    description=e.get("description", ""),
                    salience_score=e.get("salience_score", 0.0),
                    mention_count=e.get("mention_count", 0),
                    confidence=e.get("confidence", 0.0),
                    is_core_identity=bool(e.get("is_core_identity", False)),
                    first_seen=str(e.get("first_seen", "")),
                    last_updated=str(e.get("last_updated", "")),
                    community_ids=list(e.get("community_ids", [])),
                )
            )
        return entities
    except Exception as exc:
        logger.warning("memory_entities_failed", error=str(exc))
        return []


@router.get("/api/v1/memory/entity/{entity_id}", response_model=EntityDetailResponse)
async def get_entity_detail(
    request: Request,
    entity_id: str,
    depth: int = Query(default=2, ge=1, le=3),
) -> EntityDetailResponse:
    """Single entity with graph neighbours."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        from systems.memory.semantic import get_entity_neighbours as _get_neighbours

        raw = await memory.get_entity(entity_id)
        if raw is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        # get_entity_neighbours remains an internal call - no MemoryService wrapper
        # yet; it has no callers outside this router so adding a full wrapper would
        # be premature (YAGNI). Tracked in CLAUDE.md as a remaining AV6 item.
        neighbours_raw = await _get_neighbours(
            memory._neo4j, entity_id, max_depth=depth
        )

        entity = EntityItem(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
            type=raw.get("type", ""),
            description=raw.get("description", ""),
            salience_score=raw.get("salience_score", 0.0),
            mention_count=raw.get("mention_count", 0),
            confidence=raw.get("confidence", 0.0),
            is_core_identity=bool(raw.get("is_core_identity", False)),
            first_seen=str(raw.get("first_seen", "")),
            last_updated=str(raw.get("last_updated", "")),
            community_ids=list(raw.get("community_ids", [])),
        )

        neighbours = [
            EntityNeighbour(
                entity_id=n.get("entity_id", ""),
                name=n.get("name", ""),
                type=n.get("type", ""),
                relation_type=n.get("relation_type", ""),
                strength=n.get("strength", 0.0),
                direction=n.get("direction", "out"),
            )
            for n in (neighbours_raw or [])
        ]

        return EntityDetailResponse(entity=entity, neighbours=neighbours)

    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "memory_entity_detail_failed", entity_id=entity_id, error=str(exc)
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/v1/memory/beliefs", response_model=list[BeliefItem])
async def get_beliefs(
    request: Request,
    domain: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[BeliefItem]:
    """Return persisted beliefs, optionally filtered by domain."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        return []

    try:
        if domain:
            cypher = (
                "MATCH (b:ConsolidatedBelief) WHERE b.domain = $domain"
                " RETURN b ORDER BY b.precision DESC LIMIT $limit"
            )
        else:
            cypher = (
                "MATCH (b:ConsolidatedBelief)"
                " RETURN b ORDER BY b.precision DESC LIMIT $limit"
            )
        result = await memory._neo4j.query(cypher, {"domain": domain, "limit": limit})
        beliefs: list[BeliefItem] = []
        for row in result:
            b = row.get("b", {})
            if not b:
                continue
            beliefs.append(
                BeliefItem(
                    id=b.get("id", ""),
                    domain=b.get("domain", ""),
                    statement=b.get("statement", ""),
                    precision=b.get("precision", 0.0),
                    half_life_days=b.get("half_life_days"),
                    last_verified=str(b.get("last_verified", "")),
                    created_at=str(b.get("created_at", "")),
                )
            )
        return beliefs
    except Exception as exc:
        logger.warning("memory_beliefs_failed", error=str(exc))
        return []


@router.get("/api/v1/memory/communities", response_model=list[CommunityItem])
async def get_communities(
    request: Request,
    limit: int = Query(default=30, ge=1, le=100),
) -> list[CommunityItem]:
    """Return community clusters ordered by salience."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        return []

    try:
        result = await memory._neo4j.query(
            "MATCH (c:Community) RETURN c ORDER BY c.salience_score DESC LIMIT $limit",
            {"limit": limit},
        )
        communities: list[CommunityItem] = []
        for row in result:
            c = row.get("c", {})
            if not c:
                continue
            communities.append(
                CommunityItem(
                    id=c.get("id", ""),
                    level=c.get("level", 0),
                    summary=c.get("summary", ""),
                    member_count=c.get("member_count", 0),
                    coherence_score=c.get("coherence_score", 0.0),
                    salience_score=c.get("salience_score", 0.0),
                    created_at=str(c.get("created_at", "")),
                    last_recomputed=str(c.get("last_recomputed", "")),
                )
            )
        return communities
    except Exception as exc:
        logger.warning("memory_communities_failed", error=str(exc))
        return []


@router.get("/api/v1/memory/consolidation", response_model=ConsolidationResponse)
async def get_last_consolidation(request: Request) -> ConsolidationResponse:
    """Return the cached result of the last consolidation run."""
    if not _last_consolidation:
        return ConsolidationResponse(status="never_run")

    return ConsolidationResponse(
        ran_at=_last_consolidation.get("ran_at"),
        episodes_decayed=_last_consolidation.get("episodes_decayed", 0),
        entities_decayed=_last_consolidation.get("entities_decayed", 0),
        communities_detected=_last_consolidation.get("communities_detected", 0),
        episodes_compressed=_last_consolidation.get("episodes_compressed", 0),
        near_duplicates_flagged=_last_consolidation.get("near_duplicates_flagged", 0),
        duration_s=_last_consolidation.get("duration_s"),
        status=_last_consolidation.get("status", "ok"),
    )


@router.post("/api/v1/memory/consolidate", response_model=ConsolidationResponse)
async def trigger_consolidation(request: Request) -> ConsolidationResponse:
    """Manually trigger the memory consolidation pipeline (sleep cycle)."""
    global _last_consolidation

    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        t0 = time.monotonic()
        result = await memory.consolidate()
        duration = round(time.monotonic() - t0, 2)
        ran_at = datetime.now(UTC).isoformat()

        _last_consolidation = {
            "ran_at": ran_at,
            "episodes_decayed": result.get("episodes_decayed", 0),
            "entities_decayed": result.get("entities_decayed", 0),
            "communities_detected": result.get("communities_detected", 0),
            "episodes_compressed": result.get("episodes_compressed", 0),
            "near_duplicates_flagged": result.get("near_duplicates_flagged", 0),
            "duration_s": duration,
            "status": "ok",
        }

        logger.info("memory_consolidation_triggered", duration_s=duration)
        return ConsolidationResponse(**_last_consolidation)

    except Exception as exc:
        logger.exception("memory_consolidation_failed", error=str(exc))
        _last_consolidation = {"status": "error", "ran_at": None}
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/v1/memory/retrieve", response_model=RetrievalResponse)
async def retrieve_memories(
    request: Request,
    body: RetrieveRequest,
) -> RetrievalResponse:
    """Hybrid retrieval - vector + BM25 + graph traversal + salience reranking."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        embedding = await memory._embedding.embed(body.query)

        response = await memory.retrieve(
            query_text=body.query,
            query_embedding=embedding,
            max_results=body.max_results,
            salience_floor=body.salience_floor,
            include_communities=body.include_communities,
        )

        results = [
            RetrievalResultItem(
                node_id=r.node_id,
                node_type=r.node_type,
                content=r.content,
                unified_score=r.unified_score,
                vector_score=r.vector_score,
                bm25_score=r.bm25_score,
                graph_score=r.graph_score,
                salience=r.salience,
                metadata=r.metadata,
            )
            for r in response.traces
        ]

        return RetrievalResponse(
            query=body.query,
            results=results,
            entity_count=len(response.entities),
            community_count=len(response.communities),
            retrieval_time_ms=response.retrieval_time_ms,
        )
    except Exception as exc:
        logger.exception("memory_retrieve_failed", query=body.query, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── Counterfactual Routes ────────────────────────────────────────


@router.get("/api/v1/memory/counterfactuals", response_model=list[CounterfactualItem])
async def list_counterfactuals(
    request: Request,
    resolved: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[CounterfactualItem]:
    """List counterfactual episodes, optionally filtered by resolved status."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        return []

    try:
        if resolved is None:
            cypher = """
                MATCH (e:Counterfactual)
                RETURN e, null AS outcome_id, null AS outcome_summary
                ORDER BY e.ingestion_time DESC LIMIT $limit
            """
            params: dict[str, Any] = {"limit": limit}
        else:
            cypher = """
                MATCH (e:Counterfactual)
                WHERE e.resolved = $resolved
                OPTIONAL MATCH (e)-[:ALTERNATIVE_TO]->(outcome:Episode)
                RETURN e,
                       outcome.id AS outcome_id,
                       outcome.summary AS outcome_summary
                ORDER BY e.ingestion_time DESC LIMIT $limit
            """
            params = {"resolved": resolved, "limit": limit}

        rows = await memory._neo4j.execute_read(cypher, params)
        items: list[CounterfactualItem] = []
        for row in rows:
            e = row.get("e", {})
            if not e:
                continue
            items.append(
                CounterfactualItem(
                    id=e.get("id", ""),
                    event_time=str(e.get("event_time", "")),
                    summary=e.get("summary", ""),
                    policy_name=e.get("policy_name", ""),
                    policy_type=e.get("policy_type", ""),
                    efe_total=e.get("efe_total", 0.0),
                    estimated_pragmatic_value=e.get("estimated_pragmatic_value", 0.0),
                    estimated_epistemic_value=e.get("estimated_epistemic_value", 0.0),
                    chosen_policy_name=e.get("chosen_policy_name", ""),
                    chosen_efe_total=e.get("chosen_efe_total", 0.0),
                    resolved=bool(e.get("resolved", False)),
                    regret=e.get("regret"),
                    actual_outcome_success=e.get("actual_outcome_success"),
                    resolved_at=str(e.get("resolved_at", "")) or None,
                    outcome_episode_id=row.get("outcome_id"),
                    outcome_episode_summary=row.get("outcome_summary"),
                )
            )
        return items
    except Exception as exc:
        logger.warning("memory_counterfactuals_failed", error=str(exc))
        return []


@router.get(
    "/api/v1/memory/counterfactuals/{episode_id}",
    response_model=CounterfactualItem,
)
async def get_counterfactual(
    request: Request,
    episode_id: str,
) -> CounterfactualItem:
    """Single counterfactual episode with its linked outcome if resolved."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        rows = await memory._neo4j.execute_read(
            """
            MATCH (e:Counterfactual {id: $id})
            OPTIONAL MATCH (e)-[:ALTERNATIVE_TO]->(outcome:Episode)
            RETURN e,
                   outcome.id AS outcome_id,
                   outcome.summary AS outcome_summary
            """,
            {"id": episode_id},
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Counterfactual not found")

        row = rows[0]
        e = row.get("e", {})
        return CounterfactualItem(
            id=e.get("id", ""),
            event_time=str(e.get("event_time", "")),
            summary=e.get("summary", ""),
            policy_name=e.get("policy_name", ""),
            policy_type=e.get("policy_type", ""),
            efe_total=e.get("efe_total", 0.0),
            estimated_pragmatic_value=e.get("estimated_pragmatic_value", 0.0),
            estimated_epistemic_value=e.get("estimated_epistemic_value", 0.0),
            chosen_policy_name=e.get("chosen_policy_name", ""),
            chosen_efe_total=e.get("chosen_efe_total", 0.0),
            resolved=bool(e.get("resolved", False)),
            regret=e.get("regret"),
            actual_outcome_success=e.get("actual_outcome_success"),
            resolved_at=str(e.get("resolved_at", "")) or None,
            outcome_episode_id=row.get("outcome_id"),
            outcome_episode_summary=row.get("outcome_summary"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("memory_counterfactual_detail_failed", episode_id=episode_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/api/v1/memory/counterfactuals/{episode_id}/resolve",
    response_model=CounterfactualItem,
)
async def resolve_counterfactual_endpoint(
    request: Request,
    episode_id: str,
    body: CounterfactualResolveRequest,
) -> CounterfactualItem:
    """
    Resolve a counterfactual by recording its outcome and linking to the outcome episode.
    Uses resolve_counterfactual to record regret, then link_counterfactual_to_outcome
    to create the ALTERNATIVE_TO edge.
    """
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        await memory.resolve_counterfactual(
            record_id=episode_id,
            outcome_success=body.outcome_success,
            actual_pragmatic_value=body.actual_pragmatic_value,
            regret=body.regret,
        )
        await memory.link_counterfactual_to_outcome(episode_id, body.outcome_episode_id)

        logger.info(
            "counterfactual_resolved",
            episode_id=episode_id,
            outcome_episode_id=body.outcome_episode_id,
        )

        # Return updated counterfactual
        return await get_counterfactual(request, episode_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("memory_counterfactual_resolve_failed", episode_id=episode_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── Compression Stats Route ──────────────────────────────────────


@router.get("/api/v1/memory/compression/stats", response_model=list[CompressionStatItem])
async def get_compression_stats(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[CompressionStatItem]:
    """Latest compression stats per community from ProjectionBasis nodes."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        return []

    try:
        rows = await memory._neo4j.execute_read(
            """
            MATCH (pb:ProjectionBasis)
            RETURN pb
            ORDER BY pb.updated_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        items: list[CompressionStatItem] = []
        for row in rows:
            pb = row.get("pb", {})
            if not pb:
                continue
            items.append(
                CompressionStatItem(
                    community_id=pb.get("community_id", ""),
                    K=pb.get("compressed_dims", 0),
                    variance_retained=pb.get("variance_retained", 0.0),
                    quality_score=pb.get("quality_score", 0.0),
                    compression_ratio=pb.get("compression_ratio", 0.0) if pb.get("compression_ratio") else (
                        round(pb.get("compressed_dims", 0) / pb.get("original_dims", 1), 4)
                        if pb.get("original_dims", 0) > 0
                        else 0.0
                    ),
                    compressed_at=str(pb.get("updated_at", "")),
                )
            )
        return items
    except Exception as exc:
        logger.warning("memory_compression_stats_failed", error=str(exc))
        return []


# ─── Belief Decay Forecast Route ─────────────────────────────────


@router.get(
    "/api/v1/memory/beliefs/{belief_id}/decay-forecast",
    response_model=list[DecayForecastPoint],
)
async def get_belief_decay_forecast(
    request: Request,
    belief_id: str,
) -> list[DecayForecastPoint]:
    """
    Project belief precision over the next 30 days.
    Uses formula: precision * exp(-decay_constant * day)
    where decay_constant = ln(2) / half_life_days (if half_life_days is set).
    Falls back to decay_constant=0.05 if no half_life is stored.
    """
    import math

    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        rows = await memory._neo4j.execute_read(
            "MATCH (b:ConsolidatedBelief {id: $id}) RETURN b",
            {"id": belief_id},
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Belief not found")

        b = rows[0]["b"]
        precision: float = float(b.get("precision", 0.0))
        half_life: float | None = b.get("half_life_days")

        if half_life and half_life > 0:
            decay_constant = math.log(2) / half_life
        else:
            decay_constant = 0.05  # default: ~14-day half-life

        return [
            DecayForecastPoint(
                day=day,
                projected_precision=round(precision * math.exp(-decay_constant * day), 6),
            )
            for day in range(31)
        ]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("memory_belief_decay_forecast_failed", belief_id=belief_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
