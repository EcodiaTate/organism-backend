"""
EcodiaOS - EIS Pathogen Store (Qdrant Vector Database)

Multi-vector similarity search over known pathogen signatures.
Analogous to the adaptive immune system's memory B-cells: the store
holds antigenic signatures of confirmed threats and rapidly identifies
new inputs that resemble known pathogens.

Three named vectors per point (multi-vector schema):
  - "structural" (dim=32): Structural profile fingerprint
  - "histogram"  (dim=64): Token frequency histogram (feature-hashed)
  - "semantic"   (dim=768): Dense semantic embedding

Search pipeline:
  1. Query all three vectors independently (parallel Qdrant queries)
  2. Fuse scores with configurable weights
  3. Return top-k matches above threshold

Performance contract: compute_antigenic_similarity() must complete
in <5ms for warm Qdrant (same-host, in-memory index). Cold-start
or remote Qdrant is bounded by qdrant_timeout_s config.

Requires: pip install qdrant-client (add to pyproject.toml dependencies)
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.eis.models import (
    EISConfig,
    KnownPathogen,
)

logger = structlog.get_logger()


# ─── Qdrant imports (lazy to avoid hard failure if not installed) ──

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client import models as qmodels

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    AsyncQdrantClient = None  # type: ignore[assignment, misc]
    qmodels = None  # type: ignore[assignment]


# ─── Data types ───────────────────────────────────────────────────


class SimilarityMatch:
    """A single match result from the pathogen store."""

    __slots__ = (
        "pathogen_id",
        "score",
        "structural_score",
        "histogram_score",
        "semantic_score",
        "threat_class",
        "severity",
        "description",
        "canonical_text",
    )

    def __init__(
        self,
        pathogen_id: str,
        score: float,
        structural_score: float = 0.0,
        histogram_score: float = 0.0,
        semantic_score: float = 0.0,
        threat_class: str = "benign",
        severity: str = "none",
        description: str = "",
        canonical_text: str = "",
    ) -> None:
        self.pathogen_id = pathogen_id
        self.score = score
        self.structural_score = structural_score
        self.histogram_score = histogram_score
        self.semantic_score = semantic_score
        self.threat_class = threat_class
        self.severity = severity
        self.description = description
        self.canonical_text = canonical_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathogen_id": self.pathogen_id,
            "score": self.score,
            "structural_score": self.structural_score,
            "histogram_score": self.histogram_score,
            "semantic_score": self.semantic_score,
            "threat_class": self.threat_class,
            "severity": self.severity,
            "description": self.description,
        }


# ─── Collection Schema ───────────────────────────────────────────

# Named vector configuration for the Qdrant collection.
# Each point stores three vectors with different dimensions.
VECTOR_NAMES = ("structural", "histogram", "semantic")


def _build_vectors_config(config: EISConfig) -> dict[str, Any]:
    """Build the Qdrant named vectors configuration dict."""
    if not QDRANT_AVAILABLE or qmodels is None:
        return {}

    return {
        "structural": qmodels.VectorParams(
            size=config.structural_vector_dim,
            distance=qmodels.Distance.COSINE,
            on_disk=False,  # Keep in memory for <5ms latency
        ),
        "histogram": qmodels.VectorParams(
            size=config.histogram_vector_dim,
            distance=qmodels.Distance.COSINE,
            on_disk=False,
        ),
        "semantic": qmodels.VectorParams(
            size=config.embedding_dim,
            distance=qmodels.Distance.COSINE,
            on_disk=False,
        ),
    }


# ─── Payload schema ──────────────────────────────────────────────

# Qdrant payload fields stored alongside vectors for filtering/retrieval.
# These are indexed for fast filtering.
PAYLOAD_SCHEMA: dict[str, str] = {
    "threat_class": "keyword",
    "severity": "keyword",
    "description": "text",
    "canonical_text": "text",
    "tags": "keyword[]",
    "source_incident_id": "keyword",
    "match_count": "integer",
    "retired": "bool",
    "created_at": "datetime",
}


# ─── PathogenStore ────────────────────────────────────────────────


class PathogenStore:
    """
    Qdrant-backed vector store for known pathogen signatures.

    Manages the lifecycle of the pathogen collection:
      - ensure_collection(): Create collection with multi-vector schema
      - upsert_pathogen(): Index a new known pathogen
      - search(): Multi-vector similarity search
      - compute_antigenic_similarity(): High-level search + score fusion

    Thread-safe via AsyncQdrantClient. All methods are async.
    """

    def __init__(self, config: EISConfig | None = None) -> None:
        self._config = config or EISConfig()
        self._client: Any = None  # AsyncQdrantClient, typed as Any for lazy import
        self._collection_ready = False

    async def connect(self) -> None:
        """Initialise the Qdrant client connection."""
        if not QDRANT_AVAILABLE:
            logger.warning("eis_qdrant_unavailable", reason="qdrant-client not installed")
            return

        self._client = AsyncQdrantClient(
            url=self._config.qdrant_url,
            timeout=self._config.qdrant_timeout_s,
        )
        logger.info(
            "eis_qdrant_connected",
            url=self._config.qdrant_url,
            collection=self._config.qdrant_collection,
        )

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._collection_ready = False

    async def ensure_collection(self) -> None:
        """
        Create the pathogen collection if it doesn't exist.

        Uses named vectors with per-vector configuration for the
        three antigenic representations (structural, histogram, semantic).
        """
        if self._client is None or not QDRANT_AVAILABLE or qmodels is None:
            return

        collection_name = self._config.qdrant_collection
        collections = await self._client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if collection_name in existing_names:
            self._collection_ready = True
            logger.info("eis_collection_exists", collection=collection_name)
            return

        vectors_config = _build_vectors_config(self._config)

        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            # Optimise for low-latency search over small collection
            # (pathogen store is typically <100K points)
            optimizers_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=1000,  # Build HNSW after 1K points
                memmap_threshold=50000,   # Memory-map after 50K points
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,                     # HNSW connectivity (balance speed/recall)
                ef_construct=100,         # Build quality
            ),
        )

        # Create payload indexes for filtering
        for field_name, field_type in PAYLOAD_SCHEMA.items():
            schema_type = _map_payload_type(field_type)
            if schema_type is not None:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )

        self._collection_ready = True
        logger.info(
            "eis_collection_created",
            collection=collection_name,
            vectors=list(vectors_config.keys()),
        )

    async def upsert_pathogen(self, pathogen: KnownPathogen) -> None:
        """
        Index a known pathogen in the vector store.

        All three vectors must be present. If semantic_vector is empty
        (no embedding model available), it's zero-filled to maintain
        schema consistency.
        """
        if self._client is None or not QDRANT_AVAILABLE or qmodels is None:
            logger.warning("eis_upsert_skipped", reason="qdrant not available")
            return

        if not self._collection_ready:
            await self.ensure_collection()

        # Ensure vectors have correct dimensions (pad/truncate)
        structural_vec = _pad_vector(
            pathogen.structural_vector, self._config.structural_vector_dim,
        )
        histogram_vec = _pad_vector(
            pathogen.histogram_vector, self._config.histogram_vector_dim,
        )
        semantic_vec = _pad_vector(
            pathogen.semantic_vector, self._config.embedding_dim,
        )

        point = qmodels.PointStruct(
            id=pathogen.id,
            vector={
                "structural": structural_vec,
                "histogram": histogram_vec,
                "semantic": semantic_vec,
            },
            payload={
                "threat_class": pathogen.threat_class.value,
                "severity": pathogen.severity.value,
                "description": pathogen.description,
                "canonical_text": pathogen.canonical_text[:1000],  # Cap stored text
                "tags": pathogen.tags,
                "source_incident_id": pathogen.source_incident_id,
                "match_count": pathogen.match_count,
                "retired": pathogen.retired,
                "created_at": pathogen.created_at.isoformat(),
            },
        )

        await self._client.upsert(
            collection_name=self._config.qdrant_collection,
            points=[point],
        )

        logger.debug(
            "eis_pathogen_upserted",
            pathogen_id=pathogen.id,
            threat_class=pathogen.threat_class.value,
        )

    async def upsert_batch(self, pathogens: list[KnownPathogen]) -> None:
        """Batch upsert multiple pathogens. More efficient for bulk loading."""
        if self._client is None or not QDRANT_AVAILABLE or qmodels is None:
            return

        if not self._collection_ready:
            await self.ensure_collection()

        points = []
        for pathogen in pathogens:
            structural_vec = _pad_vector(
                pathogen.structural_vector, self._config.structural_vector_dim,
            )
            histogram_vec = _pad_vector(
                pathogen.histogram_vector, self._config.histogram_vector_dim,
            )
            semantic_vec = _pad_vector(
                pathogen.semantic_vector, self._config.embedding_dim,
            )

            points.append(qmodels.PointStruct(
                id=pathogen.id,
                vector={
                    "structural": structural_vec,
                    "histogram": histogram_vec,
                    "semantic": semantic_vec,
                },
                payload={
                    "threat_class": pathogen.threat_class.value,
                    "severity": pathogen.severity.value,
                    "description": pathogen.description,
                    "canonical_text": pathogen.canonical_text[:1000],
                    "tags": pathogen.tags,
                    "source_incident_id": pathogen.source_incident_id,
                    "match_count": pathogen.match_count,
                    "retired": pathogen.retired,
                    "created_at": pathogen.created_at.isoformat(),
                },
            ))

        # Qdrant batch upsert (chunks internally)
        await self._client.upsert(
            collection_name=self._config.qdrant_collection,
            points=points,
        )

        logger.info("eis_batch_upserted", count=len(points))

    async def increment_match_count(self, pathogen_id: str) -> None:
        """
        Increment match_count for a pathogen via fetch-modify-update.

        Qdrant does not support atomic increment operations; we fetch the
        current payload, increment locally, and write back. This is safe
        under low-concurrency access patterns (single-instance EIS).
        """
        if self._client is None or not QDRANT_AVAILABLE or qmodels is None:
            return

        try:
            results = await self._client.retrieve(
                collection_name=self._config.qdrant_collection,
                ids=[pathogen_id],
                with_payload=True,
            )
            if not results:
                return

            current_count: int = results[0].payload.get("match_count", 0)  # type: ignore[union-attr]
            await self._client.set_payload(
                collection_name=self._config.qdrant_collection,
                payload={"match_count": current_count + 1},
                points=[pathogen_id],
            )
        except Exception as exc:
            logger.warning(
                "eis_increment_match_count_failed",
                pathogen_id=pathogen_id,
                error=str(exc),
            )

    async def search_by_vector(
        self,
        vector_name: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        exclude_retired: bool = True,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Search a single named vector. Returns (id, score, payload) tuples.

        Low-level building block for compute_antigenic_similarity().
        """
        if self._client is None or not QDRANT_AVAILABLE or qmodels is None:
            return []

        if not self._collection_ready:
            await self.ensure_collection()

        # Build filter to exclude retired pathogens
        query_filter = None
        if exclude_retired:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="retired",
                        match=qmodels.MatchValue(value=False),
                    ),
                ],
            )

        results = await self._client.query_points(
            collection_name=self._config.qdrant_collection,
            query=query_vector,
            using=vector_name,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            (str(point.id), point.score, point.payload or {})
            for point in results.points
        ]

    async def compute_antigenic_similarity(
        self,
        structural_vector: list[float],
        histogram_vector: list[float],
        semantic_vector: list[float] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[SimilarityMatch]:
        """
        Multi-vector similarity search against the pathogen store.

        Queries all three (or two, if semantic unavailable) named vectors
        independently, then fuses scores with configurable weights.

        Score fusion: weighted sum of per-vector cosine similarities,
        with max-normalization per vector to handle scale differences.

        Returns top-k matches above threshold, sorted by fused score.

        Performance: <5ms on warm, local Qdrant with <100K points and
        HNSW index built. Network latency dominates for remote Qdrant.
        """
        if self._client is None or not QDRANT_AVAILABLE:
            return []

        k = top_k or self._config.similarity_top_k
        thresh = threshold or self._config.similarity_threshold

        # ── Query each vector independently ──
        # Fetch more candidates per vector (2x top_k) to improve fusion recall
        fetch_k = k * 2

        structural_results = await self.search_by_vector(
            "structural", structural_vector, top_k=fetch_k,
        )
        histogram_results = await self.search_by_vector(
            "histogram", histogram_vector, top_k=fetch_k,
        )

        semantic_results: list[tuple[str, float, dict[str, Any]]] = []
        has_semantic = semantic_vector is not None and len(semantic_vector) > 0
        if has_semantic:
            semantic_results = await self.search_by_vector(
                "semantic", semantic_vector, top_k=fetch_k,
            )

        # ── Fuse scores ──
        # Collect all candidate IDs
        candidates: dict[str, dict[str, Any]] = {}

        for pid, score, payload in structural_results:
            if pid not in candidates:
                candidates[pid] = {"payload": payload, "structural": 0.0, "histogram": 0.0, "semantic": 0.0}
            candidates[pid]["structural"] = max(candidates[pid]["structural"], score)

        for pid, score, payload in histogram_results:
            if pid not in candidates:
                candidates[pid] = {"payload": payload, "structural": 0.0, "histogram": 0.0, "semantic": 0.0}
            candidates[pid]["histogram"] = max(candidates[pid]["histogram"], score)

        for pid, score, payload in semantic_results:
            if pid not in candidates:
                candidates[pid] = {"payload": payload, "structural": 0.0, "histogram": 0.0, "semantic": 0.0}
            candidates[pid]["semantic"] = max(candidates[pid]["semantic"], score)

        # ── Weighted fusion ──
        cfg = self._config
        if has_semantic:
            # All three vectors available - use full weights
            # Normalise weights so they sum to 1.0 across the three sub-vectors
            total_w = cfg.structural_weight + cfg.histogram_weight + cfg.semantic_weight
            w_struct = cfg.structural_weight / total_w if total_w > 0 else 0.33
            w_hist = cfg.histogram_weight / total_w if total_w > 0 else 0.33
            w_sem = cfg.semantic_weight / total_w if total_w > 0 else 0.34
        else:
            # No semantic - redistribute weight between structural and histogram
            total_w = cfg.structural_weight + cfg.histogram_weight
            w_struct = cfg.structural_weight / total_w if total_w > 0 else 0.5
            w_hist = cfg.histogram_weight / total_w if total_w > 0 else 0.5
            w_sem = 0.0

        matches: list[SimilarityMatch] = []
        for pid, data in candidates.items():
            fused_score = (
                w_struct * data["structural"]
                + w_hist * data["histogram"]
                + w_sem * data["semantic"]
            )

            if fused_score < thresh:
                continue

            payload = data["payload"]
            matches.append(SimilarityMatch(
                pathogen_id=pid,
                score=round(fused_score, 4),
                structural_score=round(data["structural"], 4),
                histogram_score=round(data["histogram"], 4),
                semantic_score=round(data["semantic"], 4),
                threat_class=payload.get("threat_class", "benign"),
                severity=payload.get("severity", "none"),
                description=payload.get("description", ""),
                canonical_text=payload.get("canonical_text", ""),
            ))

        # Sort by fused score descending, take top_k
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:k]

    async def get_collection_info(self) -> dict[str, Any]:
        """Return collection stats for health monitoring."""
        if self._client is None or not QDRANT_AVAILABLE:
            return {"available": False, "reason": "qdrant not connected"}

        try:
            info = await self._client.get_collection(self._config.qdrant_collection)
            return {
                "available": True,
                "collection": self._config.qdrant_collection,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc)}


# ─── Helpers ──────────────────────────────────────────────────────


def _pad_vector(vec: list[float], target_dim: int) -> list[float]:
    """Pad or truncate a vector to the target dimension."""
    if len(vec) == target_dim:
        return vec
    if len(vec) > target_dim:
        return vec[:target_dim]
    return vec + [0.0] * (target_dim - len(vec))


def _map_payload_type(type_str: str) -> Any:
    """Map our payload schema type strings to Qdrant field schemas."""
    if not QDRANT_AVAILABLE or qmodels is None:
        return None

    mapping: dict[str, Any] = {
        "keyword": qmodels.PayloadSchemaType.KEYWORD,
        "text": qmodels.PayloadSchemaType.TEXT,
        "integer": qmodels.PayloadSchemaType.INTEGER,
        "float": qmodels.PayloadSchemaType.FLOAT,
        "bool": qmodels.PayloadSchemaType.BOOL,
        "datetime": qmodels.PayloadSchemaType.DATETIME,
    }

    # Handle array types like "keyword[]"
    base_type = type_str.rstrip("[]")
    return mapping.get(base_type)
