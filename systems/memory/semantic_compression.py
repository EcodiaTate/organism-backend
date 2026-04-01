"""
EcodiaOS - Semantic Lossy Compression for Long-Term Memory

During Evo deep consolidation, replaces raw episode embeddings with
projections onto the top-K principal components of each community's
embedding subspace. Reduces storage by ~80% while preserving the
discriminative dimensions needed for retrieval.

Algorithm:
  1. Retrieve all episodes for a community from Neo4j
  2. Compute embedding covariance matrix via SVD
  3. Identify top-K principal components (adaptive or fixed)
  4. Store ProjectionBasis node: K principal directions + mean + variance info
  5. Project each episode embedding: compressed = U_top_k.T @ (embedding - mean)
  6. Replace episode embedding with compressed version + metadata

Integration:
  - Called from memory consolidation (Phase 1, after community detection)
  - Retrieval supports both compressed and uncompressed episodes
  - ProjectionBasis linked to Community via COMPRESSED_VIA relationship

Performance budget: fits within the 60-second consolidation window.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# ─── Configuration defaults ─────────────────────────────────────────

# Minimum episodes in a community before compression is attempted
MIN_EPISODES_FOR_COMPRESSION = 50

# Target variance retention - adaptive K selection stops here
DEFAULT_VARIANCE_TARGET = 0.80

# Hard floor/ceiling on K to prevent degenerate cases
MIN_K = 8
MAX_K = 256

# Minimum retrieval quality (cosine similarity between original and
# reconstructed embeddings, sampled). Abort compression if below this.
QUALITY_THRESHOLD = 0.90

# Batch size for Neo4j updates to avoid transaction timeouts
UPDATE_BATCH_SIZE = 200


@dataclass
class CompressionResult:
    """Summary of a single community's compression run."""

    community_id: str
    episodes_compressed: int
    original_dims: int
    compressed_dims: int
    variance_retained: float
    compression_ratio: float
    quality_score: float  # mean cosine similarity of sampled reconstructions
    skipped: bool = False
    skip_reason: str = ""


def _select_adaptive_k(
    singular_values: np.ndarray[Any, Any],
    variance_target: float = DEFAULT_VARIANCE_TARGET,
) -> int:
    """
    Select K such that the top-K components explain >= variance_target
    of total variance. Clamps between MIN_K and MAX_K.
    """
    total_variance = float(np.sum(singular_values ** 2))
    if total_variance == 0:
        return MIN_K

    cumulative = np.cumsum(singular_values ** 2) / total_variance
    # Find first index where cumulative variance >= target
    candidates = np.where(cumulative >= variance_target)[0]
    if len(candidates) == 0:
        k = len(singular_values)
    else:
        k = int(candidates[0]) + 1  # +1 because index is 0-based

    return max(MIN_K, min(MAX_K, k))


def _compute_quality_score(
    embeddings: np.ndarray[Any, Any],
    mean: np.ndarray[Any, Any],
    basis: np.ndarray[Any, Any],
    sample_size: int = 50,
) -> float:
    """
    Sample episodes, project → reconstruct, measure cosine similarity
    to original. Returns mean similarity across the sample.
    """
    n = embeddings.shape[0]
    indices = np.random.choice(n, size=min(sample_size, n), replace=False)
    sample = embeddings[indices]

    centered = sample - mean
    compressed = centered @ basis.T  # (sample, K)
    reconstructed = compressed @ basis + mean  # (sample, D)

    # Cosine similarity per row
    dot = np.sum(sample * reconstructed, axis=1)
    norm_orig = np.linalg.norm(sample, axis=1)
    norm_recon = np.linalg.norm(reconstructed, axis=1)
    denom = norm_orig * norm_recon
    denom = np.where(denom == 0, 1.0, denom)
    similarities = dot / denom

    return float(np.mean(similarities))


async def compress_community_embeddings(
    neo4j: Neo4jClient,
    community_id: str,
    variance_target: float = DEFAULT_VARIANCE_TARGET,
    fixed_k: int | None = None,
) -> CompressionResult:
    """
    Compress all episode embeddings within a community using PCA.

    Steps:
      1. Fetch all episode embeddings belonging to this community
      2. Compute SVD on the centered embedding matrix
      3. Select K (adaptive or fixed)
      4. Validate quality on a sample
      5. Store ProjectionBasis node in Neo4j
      6. Update each episode with compressed embedding + metadata
    """
    start = time.monotonic()

    # ── Step 1: Fetch episode embeddings for this community ──────────
    # Community nodes use `id` as the unique property (set by _materialize_community_nodes).
    rows = await neo4j.execute_read(
        """
        MATCH (c:Community {id: $cid})<-[:BELONGS_TO]-(entity:Entity)
        MATCH (entity)<-[:MENTIONS]-(ep:Episode)
        WHERE ep.embedding IS NOT NULL
          AND ep.embedding_compressed IS NULL
        RETURN DISTINCT ep.id AS id, ep.embedding AS embedding
        """,
        {"cid": community_id},
    )

    if len(rows) < MIN_EPISODES_FOR_COMPRESSION:
        return CompressionResult(
            community_id=community_id,
            episodes_compressed=0,
            original_dims=0,
            compressed_dims=0,
            variance_retained=0.0,
            compression_ratio=0.0,
            quality_score=0.0,
            skipped=True,
            skip_reason=f"too_few_episodes ({len(rows)} < {MIN_EPISODES_FOR_COMPRESSION})",
        )

    episode_ids = [r["id"] for r in rows]
    embeddings = np.array([r["embedding"] for r in rows], dtype=np.float64)
    n_episodes, original_dims = embeddings.shape

    # ── Step 2: SVD on centered embeddings ──────────────────────────
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # Economy SVD - we only need the left singular vectors up to rank
    # Use full_matrices=False for efficiency on tall-skinny matrices
    _U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # ── Step 3: Select K ────────────────────────────────────────────
    if fixed_k is not None:
        k = max(MIN_K, min(MAX_K, fixed_k))
    else:
        k = _select_adaptive_k(S, variance_target)

    # Can't have more components than either dimension
    k = min(k, original_dims, n_episodes)

    # Top-K right singular vectors form the projection basis
    basis = Vt[:k]  # shape (K, D)

    # Compute actual variance retained
    total_var = float(np.sum(S ** 2))
    retained_var = float(np.sum(S[:k] ** 2))
    variance_retained = retained_var / total_var if total_var > 0 else 0.0

    # ── Step 4: Quality check on sample ─────────────────────────────
    quality = _compute_quality_score(embeddings, mean, basis)

    if quality < QUALITY_THRESHOLD:
        logger.warning(
            "semantic_compression_quality_below_threshold",
            community_id=community_id,
            quality=round(quality, 4),
            threshold=QUALITY_THRESHOLD,
            k=k,
        )
        return CompressionResult(
            community_id=community_id,
            episodes_compressed=0,
            original_dims=original_dims,
            compressed_dims=k,
            variance_retained=variance_retained,
            compression_ratio=0.0,
            quality_score=quality,
            skipped=True,
            skip_reason=f"quality_below_threshold ({quality:.4f} < {QUALITY_THRESHOLD})",
        )

    # ── Step 5: Store ProjectionBasis in Neo4j ──────────────────────
    basis_id = f"basis_{community_id}"
    await neo4j.execute_write(
        """
        MERGE (pb:ProjectionBasis {id: $id})
        SET pb.community_id = $cid,
            pb.original_dims = $original_dims,
            pb.compressed_dims = $k,
            pb.mean_vector = $mean,
            pb.basis_matrix_json = $basis_json,
            pb.singular_values = $singular_values,
            pb.variance_retained = $variance_retained,
            pb.quality_score = $quality,
            pb.episode_count = $n_episodes,
            pb.created_at = datetime(),
            pb.updated_at = datetime()
        WITH pb
        MATCH (c:Community {id: $cid})
        MERGE (c)-[:COMPRESSED_VIA]->(pb)
        """,
        {
            "id": basis_id,
            "cid": community_id,
            "original_dims": original_dims,
            "k": k,
            "mean": mean.tolist(),
            "basis_json": json.dumps(basis.tolist()),
            "singular_values": S[:k].tolist(),
            "variance_retained": variance_retained,
            "quality": quality,
            "n_episodes": n_episodes,
        },
    )

    # ── Step 6: Compress and update episodes in batches ──────────────
    compressed_all = (centered @ basis.T)  # (N, K) - projected embeddings

    for batch_start in range(0, n_episodes, UPDATE_BATCH_SIZE):
        batch_end = min(batch_start + UPDATE_BATCH_SIZE, n_episodes)
        batch_ids = episode_ids[batch_start:batch_end]
        batch_compressed = compressed_all[batch_start:batch_end]

        # Build parameter list for UNWIND
        updates = [
            {
                "id": batch_ids[i],
                "compressed": batch_compressed[i].tolist(),
            }
            for i in range(len(batch_ids))
        ]

        await neo4j.execute_write(
            """
            UNWIND $updates AS upd
            MATCH (ep:Episode {id: upd.id})
            SET ep.embedding_compressed = upd.compressed,
                ep.compression_basis_id = $basis_id,
                ep.compression_ratio = $ratio,
                ep.compression_variance_retained = $var_retained
            REMOVE ep.embedding
            """,
            {
                "updates": updates,
                "basis_id": basis_id,
                "ratio": round(k / original_dims, 4),
                "var_retained": round(variance_retained, 4),
            },
        )

    compression_ratio = round(k / original_dims, 4)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "semantic_compression",
        community_id=community_id,
        episodes=n_episodes,
        dims=f"{original_dims}→{k}",
        variance_retained=round(variance_retained, 4),
        quality=round(quality, 4),
        storage_saved=f"{(1 - compression_ratio) * 100:.0f}%",
        elapsed_ms=elapsed_ms,
    )

    return CompressionResult(
        community_id=community_id,
        episodes_compressed=n_episodes,
        original_dims=original_dims,
        compressed_dims=k,
        variance_retained=variance_retained,
        compression_ratio=compression_ratio,
        quality_score=quality,
    )


async def run_semantic_compression(
    neo4j: Neo4jClient,
    variance_target: float = DEFAULT_VARIANCE_TARGET,
    fixed_k: int | None = None,
) -> dict[str, Any]:
    """
    Run semantic compression across all communities.
    Called from memory consolidation after community detection.

    Returns a summary report dict.
    """
    start = time.monotonic()

    # Get all communities - use `id` which is the canonical property on Community nodes.
    communities = await neo4j.execute_read(
        """
        MATCH (c:Community)
        WHERE c.id IS NOT NULL
        RETURN c.id AS cid, c.member_count AS members
        ORDER BY c.member_count DESC
        """
    )

    results: list[CompressionResult] = []
    total_compressed = 0
    total_skipped = 0

    for row in communities:
        cid = row["cid"]
        try:
            result = await compress_community_embeddings(
                neo4j, cid, variance_target, fixed_k
            )
            results.append(result)
            if result.skipped:
                total_skipped += 1
            else:
                total_compressed += result.episodes_compressed
        except Exception as exc:
            logger.error(
                "semantic_compression_community_failed",
                community_id=cid,
                error=str(exc),
            )
            total_skipped += 1

    elapsed_ms = int((time.monotonic() - start) * 1000)

    report = {
        "communities_processed": len(communities),
        "communities_compressed": len(results) - total_skipped,
        "communities_skipped": total_skipped,
        "total_episodes_compressed": total_compressed,
        "elapsed_ms": elapsed_ms,
        "details": [
            {
                "community_id": r.community_id,
                "episodes": r.episodes_compressed,
                "dims": f"{r.original_dims}→{r.compressed_dims}" if not r.skipped else "n/a",
                "variance_retained": round(r.variance_retained, 4),
                "quality": round(r.quality_score, 4),
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
            }
            for r in results
        ],
    }

    logger.info(
        "semantic_compression_complete",
        communities=len(communities),
        compressed=total_compressed,
        skipped=total_skipped,
        elapsed_ms=elapsed_ms,
    )

    return report


# ─── Retrieval helpers ──────────────────────────────────────────────


async def project_query_embedding(
    neo4j: Neo4jClient,
    query_embedding: list[float],
    basis_id: str,
) -> list[float] | None:
    """
    Project a full-dimensional query embedding into a community's
    compressed space for similarity comparison with compressed episodes.

    Returns the projected (K-dim) vector, or None if basis not found.
    """
    rows = await neo4j.execute_read(
        """
        MATCH (pb:ProjectionBasis {id: $id})
        RETURN pb.mean_vector AS mean,
               pb.basis_matrix_json AS basis_json,
               pb.compressed_dims AS k
        """,
        {"id": basis_id},
    )

    if not rows:
        return None

    row = rows[0]
    mean = np.array(row["mean"], dtype=np.float64)
    basis = np.array(json.loads(row["basis_json"]), dtype=np.float64)  # (K, D)
    query = np.array(query_embedding, dtype=np.float64)

    # Project: compressed = basis @ (query - mean)
    compressed = basis @ (query - mean)
    result: list[float] = compressed.tolist()
    return result


async def decompress_embedding(
    neo4j: Neo4jClient,
    compressed_embedding: list[float],
    basis_id: str,
) -> list[float] | None:
    """
    Reconstruct an approximate full-dimensional embedding from a
    compressed one. Useful for cross-space comparisons.

    Returns the reconstructed (D-dim) vector, or None if basis not found.
    """
    rows = await neo4j.execute_read(
        """
        MATCH (pb:ProjectionBasis {id: $id})
        RETURN pb.mean_vector AS mean,
               pb.basis_matrix_json AS basis_json
        """,
        {"id": basis_id},
    )

    if not rows:
        return None

    row = rows[0]
    mean = np.array(row["mean"], dtype=np.float64)
    basis = np.array(json.loads(row["basis_json"]), dtype=np.float64)  # (K, D)
    compressed = np.array(compressed_embedding, dtype=np.float64)

    # Reconstruct: approx_original = basis.T @ compressed + mean
    reconstructed = basis.T @ compressed + mean
    result: list[float] = reconstructed.tolist()
    return result


async def get_all_basis_ids(neo4j: Neo4jClient) -> list[str]:
    """Return all ProjectionBasis IDs for retrieval routing."""
    rows = await neo4j.execute_read(
        "MATCH (pb:ProjectionBasis) RETURN pb.id AS id"
    )
    return [r["id"] for r in rows]
