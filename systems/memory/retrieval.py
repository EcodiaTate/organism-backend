"""
EcodiaOS — Hybrid Memory Retrieval

Three-leg parallel retrieval: vector similarity + BM25 keyword + graph traversal.
Results are merged, deduplicated, and re-ranked by a unified score.

Target: ≤200ms end-to-end.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from primitives.memory_trace import (
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    RetrievalResult,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


async def _vector_search(
    neo4j: Neo4jClient,
    query_embedding: list[float],
    limit: int,
) -> list[RetrievalResult]:
    """
    Vector similarity search via Neo4j native vector index.

    Searches uncompressed episodes directly. Compressed episodes
    (those with embedding removed and embedding_compressed set) are
    searched separately via _compressed_vector_search.
    """
    if not query_embedding:
        return []

    results = await neo4j.execute_read(
        """
        CALL db.index.vector.queryNodes('episode_embedding', $limit, $embedding)
        YIELD node, score
        RETURN node.id AS id, node.summary AS content,
               node.salience_composite AS salience,
               node.somatic_vector AS somatic_vector,
               score AS vector_score,
               'episode' AS node_type
        """,
        {"embedding": query_embedding, "limit": limit},
    )

    return [
        RetrievalResult(
            node_id=r["id"],
            node_type=r["node_type"],
            content=r.get("content", ""),
            vector_score=r.get("vector_score"),
            salience=r.get("salience", 0.0),
            salience_score=r.get("salience", 0.0),
            somatic_vector=r.get("somatic_vector"),
        )
        for r in results
    ]


async def _compressed_vector_search(
    neo4j: Neo4jClient,
    query_embedding: list[float],
    limit: int,
) -> list[RetrievalResult]:
    """
    Search compressed episodes by projecting the query into each
    community's compressed space and computing cosine similarity.

    Only runs if ProjectionBasis nodes exist (i.e., compression has
    been performed). Returns results scored in [0, 1] like the
    uncompressed vector search.
    """
    if not query_embedding:
        return []

    import json as _json

    import numpy as np

    # Get all projection bases
    bases = await neo4j.execute_read(
        """
        MATCH (pb:ProjectionBasis)
        RETURN pb.id AS id, pb.mean_vector AS mean,
               pb.basis_matrix_json AS basis_json
        """
    )

    if not bases:
        return []

    results_map: dict[str, RetrievalResult] = {}
    query_arr = np.array(query_embedding, dtype=np.float64)

    for basis_row in bases:
        basis_id = basis_row["id"]
        mean = np.array(basis_row["mean"], dtype=np.float64)
        basis = np.array(_json.loads(basis_row["basis_json"]), dtype=np.float64)

        # Project query into this compressed space
        query_compressed = basis @ (query_arr - mean)
        query_compressed.tolist()

        # Find compressed episodes using this basis and compute similarity
        episodes = await neo4j.execute_read(
            """
            MATCH (ep:Episode)
            WHERE ep.compression_basis_id = $basis_id
              AND ep.embedding_compressed IS NOT NULL
            RETURN ep.id AS id, ep.summary AS content,
                   ep.salience_composite AS salience,
                   ep.somatic_vector AS somatic_vector,
                   ep.embedding_compressed AS compressed
            """,
            {"basis_id": basis_id},
        )

        for ep in episodes:
            compressed = np.array(ep["compressed"], dtype=np.float64)
            # Cosine similarity in compressed space
            dot = float(np.dot(query_compressed, compressed))
            norm_q = float(np.linalg.norm(query_compressed))
            norm_c = float(np.linalg.norm(compressed))
            denom = norm_q * norm_c
            score = dot / denom if denom > 0 else 0.0
            # Normalize to [0, 1] range (cosine is in [-1, 1])
            score = (score + 1.0) / 2.0

            ep_id = ep["id"]
            if ep_id not in results_map or score > (results_map[ep_id].vector_score or 0.0):
                results_map[ep_id] = RetrievalResult(
                    node_id=ep_id,
                    node_type="episode",
                    content=ep.get("content", ""),
                    vector_score=score,
                    salience=ep.get("salience", 0.0),
                    salience_score=ep.get("salience", 0.0),
                    somatic_vector=ep.get("somatic_vector"),
                )

    # Sort by score, return top limit
    ranked = sorted(results_map.values(), key=lambda r: r.vector_score or 0.0, reverse=True)
    return ranked[:limit]


async def _bm25_search(
    neo4j: Neo4jClient,
    query_text: str,
    limit: int,
) -> list[RetrievalResult]:
    """BM25 keyword search via Neo4j fulltext index."""
    if not query_text:
        return []

    # Escape special Lucene/fulltext characters so user text doesn't break query parsing.
    # Neo4j fulltext uses Lucene syntax; these chars have special meaning.
    _lucene_special = r'\+-&|!(){}[]^"~*?:/'
    safe_query = "".join(f"\\{c}" if c in _lucene_special else c for c in query_text)

    results = await neo4j.execute_read(
        """
        CALL db.index.fulltext.queryNodes('episode_content', $query)
        YIELD node, score
        RETURN node.id AS id, node.summary AS content,
               node.salience_composite AS salience,
               node.somatic_vector AS somatic_vector,
               score AS bm25_score,
               'episode' AS node_type
        LIMIT $limit
        """,
        {"query": safe_query, "limit": limit},
    )

    return [
        RetrievalResult(
            node_id=r["id"],
            node_type=r["node_type"],
            content=r.get("content", ""),
            bm25_score=r.get("bm25_score"),
            salience=r.get("salience", 0.0),
            salience_score=r.get("salience", 0.0),
            somatic_vector=r.get("somatic_vector"),
        )
        for r in results
    ]


async def _graph_traverse(
    neo4j: Neo4jClient,
    query_embedding: list[float],
    depth: int,
    limit: int,
) -> list[RetrievalResult]:
    """
    Spreading activation search: find the most relevant entities via vector
    search, then traverse their neighbourhood through multiple relationship
    types to find related episodes.

    Hops:
      1. Entity ←MENTIONS← Episode (direct mentions)
      2. Entity -RELATES_TO→ Entity2 ←MENTIONS← Episode (semantic neighbours)
      3. Episode -FOLLOWED_BY→ Episode2 (temporal sequence)

    Activation decays with each hop (0.8× per hop) to prefer closer results.
    The ``depth`` parameter controls how many hops to follow (1-3).
    """
    if not query_embedding:
        return []

    effective_depth = max(1, min(3, depth))

    # Hop 1: Direct entity → episode (always run)
    direct_results = await neo4j.execute_read(
        """
        CALL db.index.vector.queryNodes('entity_embedding', 3, $embedding)
        YIELD node AS entity, score AS entity_score

        MATCH (entity)<-[:MENTIONS]-(ep:Episode)
        RETURN DISTINCT ep.id AS id, ep.summary AS content,
               ep.salience_composite AS salience,
               ep.somatic_vector AS somatic_vector,
               entity_score AS graph_score,
               'episode' AS node_type,
               1 AS hop
        ORDER BY ep.salience_composite DESC
        LIMIT $limit
        """,
        {"embedding": query_embedding, "limit": limit},
    )

    results_map: dict[str, RetrievalResult] = {}
    for r in direct_results:
        node_id = r["id"]
        results_map[node_id] = RetrievalResult(
            node_id=node_id,
            node_type=r["node_type"],
            content=r.get("content", ""),
            graph_score=r.get("graph_score", 0.0),
            salience=r.get("salience", 0.0),
            salience_score=r.get("salience", 0.0),
            somatic_vector=r.get("somatic_vector"),
        )

    # Hop 2: Entity → RELATES_TO → Entity2 → MENTIONS → Episode (semantic neighbours)
    if effective_depth >= 2:
        try:
            semantic_results = await neo4j.execute_read(
                """
                CALL db.index.vector.queryNodes('entity_embedding', 3, $embedding)
                YIELD node AS entity, score AS entity_score

                MATCH (entity)-[rel:RELATES_TO]-(neighbour:Entity)
                WHERE rel.strength > 0.3
                MATCH (neighbour)<-[:MENTIONS]-(ep:Episode)
                RETURN DISTINCT ep.id AS id, ep.summary AS content,
                       ep.salience_composite AS salience,
                       ep.somatic_vector AS somatic_vector,
                       entity_score * 0.8 * rel.strength AS graph_score,
                       'episode' AS node_type,
                       2 AS hop
                ORDER BY graph_score DESC
                LIMIT $limit
                """,
                {"embedding": query_embedding, "limit": limit},
            )
            for r in semantic_results:
                node_id = r["id"]
                if node_id not in results_map:
                    results_map[node_id] = RetrievalResult(
                        node_id=node_id,
                        node_type=r["node_type"],
                        content=r.get("content", ""),
                        graph_score=r.get("graph_score", 0.0),
                        salience=r.get("salience", 0.0),
                        salience_score=r.get("salience", 0.0),
                        somatic_vector=r.get("somatic_vector"),
                    )
        except Exception:
            pass  # Semantic hop is best-effort

    # Hop 3: Community expansion — find episodes from the same community
    if effective_depth >= 2:
        try:
            community_results = await neo4j.execute_read(
                """
                CALL db.index.vector.queryNodes('entity_embedding', 3, $embedding)
                YIELD node AS entity, score AS entity_score

                MATCH (entity)-[:BELONGS_TO]->(c:Community)<-[:BELONGS_TO]-(sibling:Entity)
                WHERE sibling <> entity
                MATCH (sibling)<-[:MENTIONS]-(ep:Episode)
                RETURN DISTINCT ep.id AS id, ep.summary AS content,
                       ep.salience_composite AS salience,
                       ep.somatic_vector AS somatic_vector,
                       entity_score * 0.65 AS graph_score,
                       'episode' AS node_type,
                       3 AS hop
                ORDER BY graph_score DESC
                LIMIT $limit
                """,
                {"embedding": query_embedding, "limit": limit},
            )
            for r in community_results:
                node_id = r["id"]
                if node_id not in results_map:
                    results_map[node_id] = RetrievalResult(
                        node_id=node_id,
                        node_type=r["node_type"],
                        content=r.get("content", ""),
                        graph_score=r.get("graph_score", 0.0),
                        salience=r.get("salience", 0.0),
                        salience_score=r.get("salience", 0.0),
                        somatic_vector=r.get("somatic_vector"),
                    )
        except Exception:
            pass  # Community hop is best-effort

    # Hop 4: Episode → FOLLOWED_BY → Episode (temporal context)
    if effective_depth >= 3 and results_map:
        try:
            seed_ids = list(results_map.keys())[:5]
            temporal_results = await neo4j.execute_read(
                """
                UNWIND $seed_ids AS seed_id
                MATCH (ep:Episode {id: seed_id})-[fb:FOLLOWED_BY]-(neighbour:Episode)
                WHERE fb.causal_strength > 0.2
                RETURN DISTINCT neighbour.id AS id, neighbour.summary AS content,
                       neighbour.salience_composite AS salience,
                       neighbour.somatic_vector AS somatic_vector,
                       fb.causal_strength * 0.6 AS graph_score,
                       'episode' AS node_type,
                       3 AS hop
                ORDER BY graph_score DESC
                LIMIT $limit
                """,
                {"seed_ids": seed_ids, "limit": limit},
            )
            for r in temporal_results:
                node_id = r["id"]
                if node_id not in results_map:
                    results_map[node_id] = RetrievalResult(
                        node_id=node_id,
                        node_type=r["node_type"],
                        content=r.get("content", ""),
                        graph_score=r.get("graph_score", 0.0),
                        salience=r.get("salience", 0.0),
                        salience_score=r.get("salience", 0.0),
                        somatic_vector=r.get("somatic_vector"),
                    )
        except Exception:
            pass  # Temporal hop is best-effort

    return list(results_map.values())


def _compute_unified_score(result: RetrievalResult, temporal_bias: str = "recency") -> float:
    """
    Compute a unified relevance score from all retrieval legs + salience.

    Weights:
    - Vector similarity: 0.35
    - BM25 keyword:      0.20
    - Graph traversal:   0.20
    - Salience:          0.25
    """
    vector = result.vector_score or 0.0
    bm25 = min((result.bm25_score or 0.0) / 10.0, 1.0)  # Normalise BM25 to ~0-1
    graph = result.graph_score or 0.0
    salience = result.salience

    return (
        0.35 * vector
        + 0.20 * bm25
        + 0.20 * graph
        + 0.25 * salience
    )


def _merge_deduplicate(
    *result_lists: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Merge results from all legs, keeping the highest-scoring version of duplicates."""
    seen: dict[str, RetrievalResult] = {}

    for results in result_lists:
        for r in results:
            if r.node_id in seen:
                existing = seen[r.node_id]
                # Merge scores from different legs
                if r.vector_score and not existing.vector_score:
                    existing.vector_score = r.vector_score
                if r.bm25_score and not existing.bm25_score:
                    existing.bm25_score = r.bm25_score
                if r.graph_score and not existing.graph_score:
                    existing.graph_score = r.graph_score
                # Carry somatic vector forward if missing
                if r.somatic_vector and not existing.somatic_vector:
                    existing.somatic_vector = r.somatic_vector
            else:
                seen[r.node_id] = r

    return list(seen.values())


async def hybrid_retrieve(
    neo4j: Neo4jClient,
    request: MemoryRetrievalRequest,
) -> MemoryRetrievalResponse:
    """
    Execute hybrid retrieval: vector + BM25 + graph in parallel.
    Target: ≤200ms.
    """
    start = time.monotonic()
    expanded_limit = request.max_results * 2

    # Run all four legs in parallel (compressed search is a no-op if no bases exist)
    query_emb = request.query_embedding or []
    vector_task = _vector_search(neo4j, query_emb, expanded_limit)
    compressed_task = _compressed_vector_search(neo4j, query_emb, expanded_limit)
    bm25_task = _bm25_search(neo4j, request.query_text or "", expanded_limit)
    graph_task = _graph_traverse(
        neo4j, query_emb, request.traversal_depth, request.max_results
    )

    vector_results, compressed_results, bm25_results, graph_results = await asyncio.gather(
        vector_task, compressed_task, bm25_task, graph_task,
        return_exceptions=True,
    )

    # Handle any failures gracefully
    safe_vector: list[RetrievalResult]
    safe_compressed: list[RetrievalResult]
    safe_bm25: list[RetrievalResult]
    safe_graph: list[RetrievalResult]
    if isinstance(vector_results, BaseException):
        logger.warning("vector_search_failed", error=str(vector_results))
        safe_vector = []
    else:
        safe_vector = vector_results
    if isinstance(compressed_results, BaseException):
        logger.warning("compressed_vector_search_failed", error=str(compressed_results))
        safe_compressed = []
    else:
        safe_compressed = compressed_results
    if isinstance(bm25_results, BaseException):
        logger.warning("bm25_search_failed", error=str(bm25_results))
        safe_bm25 = []
    else:
        safe_bm25 = bm25_results
    if isinstance(graph_results, BaseException):
        logger.warning("graph_search_failed", error=str(graph_results))
        safe_graph = []
    else:
        safe_graph = graph_results

    # Merge and compute unified scores
    all_results = _merge_deduplicate(safe_vector, safe_compressed, safe_bm25, safe_graph)

    for result in all_results:
        result.unified_score = _compute_unified_score(result, request.temporal_bias)
        # Sync salience_score for somatic reranking (reads salience_score attribute)
        result.salience_score = result.unified_score

    # Filter and rank
    ranked = sorted(all_results, key=lambda r: r.unified_score, reverse=True)
    ranked = [r for r in ranked if r.unified_score >= request.salience_floor]
    ranked = ranked[: request.max_results]

    elapsed_ms = int((time.monotonic() - start) * 1000)

    logger.debug(
        "hybrid_retrieval_complete",
        vector_hits=len(safe_vector),
        compressed_hits=len(safe_compressed),
        bm25_hits=len(safe_bm25),
        graph_hits=len(safe_graph),
        merged=len(all_results),
        returned=len(ranked),
        elapsed_ms=elapsed_ms,
    )

    return MemoryRetrievalResponse(
        traces=ranked,
        retrieval_time_ms=elapsed_ms,
    )
