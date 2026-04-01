"""Tests for Semantic Lossy Compression - PCA-based embedding compression."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from systems.memory.semantic_compression import (
    MAX_K,
    MIN_K,
    QUALITY_THRESHOLD,
    _compute_quality_score,
    _select_adaptive_k,
    compress_community_embeddings,
    decompress_embedding,
    project_query_embedding,
    run_semantic_compression,
)

# ─── Helpers ─────────────────────────────────────────────────────────


def _random_embeddings(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate n random embeddings of dimension d with some structure."""
    rng = np.random.default_rng(seed)
    # Create embeddings with correlations (so PCA actually reduces well)
    base = rng.standard_normal((n, 10))
    projection = rng.standard_normal((10, d))
    noise = rng.standard_normal((n, d)) * 0.1
    embeddings = base @ projection + noise
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def _make_neo4j_mock(
    episode_rows: list[dict],
    community_rows: list[dict] | None = None,
) -> AsyncMock:
    """Build a mock Neo4j client that returns controlled data."""
    mock = AsyncMock()

    async def execute_read_side_effect(query: str, params: dict | None = None):
        # List-all-communities query (no community_id param, ORDER BY member_count)
        if "ORDER BY" in query and "member_count" in query:
            return community_rows or []
        if "ProjectionBasis" in query:
            return []
        # Episode fetch for a specific community (has community_id param)
        return episode_rows

    async def execute_write_side_effect(query: str, params: dict | None = None):
        return []

    mock.execute_read = AsyncMock(side_effect=execute_read_side_effect)
    mock.execute_write = AsyncMock(side_effect=execute_write_side_effect)
    return mock


# ─── Unit Tests: adaptive K selection ────────────────────────────────


class TestAdaptiveKSelection:
    def test_selects_k_for_target_variance(self):
        # Singular values that drop off - first 10 explain ~80% variance
        svs = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1] + [0.1] * 90, dtype=np.float64)
        k = _select_adaptive_k(svs, variance_target=0.80)
        # Should pick relatively few components
        assert MIN_K <= k <= 20

    def test_clamps_to_min_k(self):
        # Even if 1 component explains everything, K >= MIN_K
        svs = np.array([100.0] + [0.0] * 99)
        k = _select_adaptive_k(svs, variance_target=0.80)
        assert k >= MIN_K

    def test_clamps_to_max_k(self):
        # Flat spectrum - all components equally important
        svs = np.ones(500)
        k = _select_adaptive_k(svs, variance_target=0.99)
        assert k <= MAX_K

    def test_handles_zero_variance(self):
        svs = np.zeros(10)
        k = _select_adaptive_k(svs)
        assert k == MIN_K


# ─── Unit Tests: quality score ───────────────────────────────────────


class TestQualityScore:
    def test_perfect_reconstruction(self):
        # When K == D, reconstruction is perfect
        embeddings = _random_embeddings(20, 10)
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Use all components
        basis = Vt  # (10, 10) - full rank
        quality = _compute_quality_score(embeddings, mean, basis, sample_size=20)
        assert quality > 0.99

    def test_lossy_reconstruction_reasonable(self):
        # With K < D, quality should still be decent for structured data
        embeddings = _random_embeddings(100, 768)
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
        basis = Vt[:32]  # Only 32 components
        quality = _compute_quality_score(embeddings, mean, basis, sample_size=50)
        # Should be reasonable since data has low effective dimensionality
        assert quality > 0.80


# ─── Integration Tests: compress_community_embeddings ────────────────


class TestCompressCommunityEmbeddings:
    @pytest.mark.asyncio
    async def test_skips_too_few_episodes(self):
        few_rows = [
            {"id": f"ep_{i}", "embedding": [0.1] * 768}
            for i in range(10)  # below MIN_EPISODES_FOR_COMPRESSION
        ]
        neo4j = _make_neo4j_mock(few_rows)

        result = await compress_community_embeddings(neo4j, "community_1")

        assert result.skipped is True
        assert "too_few_episodes" in result.skip_reason

    @pytest.mark.asyncio
    async def test_compresses_structured_embeddings(self):
        n = 100
        embeddings = _random_embeddings(n, 768)
        rows = [
            {"id": f"ep_{i}", "embedding": embeddings[i].tolist()}
            for i in range(n)
        ]
        neo4j = _make_neo4j_mock(rows)

        result = await compress_community_embeddings(neo4j, "community_1")

        assert result.skipped is False
        assert result.episodes_compressed == n
        assert result.original_dims == 768
        assert result.compressed_dims < 768
        assert result.compressed_dims >= MIN_K
        assert result.variance_retained >= 0.75
        assert result.quality_score >= QUALITY_THRESHOLD

        # Verify Neo4j writes were called (basis + episode updates)
        assert neo4j.execute_write.call_count >= 2

    @pytest.mark.asyncio
    async def test_fixed_k_overrides_adaptive(self):
        n = 60
        embeddings = _random_embeddings(n, 768)
        rows = [
            {"id": f"ep_{i}", "embedding": embeddings[i].tolist()}
            for i in range(n)
        ]
        neo4j = _make_neo4j_mock(rows)

        result = await compress_community_embeddings(
            neo4j, "community_1", fixed_k=32
        )

        if not result.skipped:
            assert result.compressed_dims == 32


# ─── Integration Tests: run_semantic_compression ─────────────────────


class TestRunSemanticCompression:
    @pytest.mark.asyncio
    async def test_no_communities(self):
        neo4j = _make_neo4j_mock([], community_rows=[])
        report = await run_semantic_compression(neo4j)

        assert report["communities_processed"] == 0
        assert report["total_episodes_compressed"] == 0

    @pytest.mark.asyncio
    async def test_reports_skipped_communities(self):
        # Community with too few episodes
        few_rows = [
            {"id": f"ep_{i}", "embedding": [0.1] * 768}
            for i in range(5)
        ]
        community_rows = [{"cid": "c1", "members": 3}]
        neo4j = _make_neo4j_mock(few_rows, community_rows)

        report = await run_semantic_compression(neo4j)

        assert report["communities_processed"] == 1
        assert report["communities_skipped"] == 1


# ─── Unit Tests: projection helpers ─────────────────────────────────


class TestProjectionHelpers:
    @pytest.mark.asyncio
    async def test_project_query_embedding(self):
        d = 768
        k = 32
        mean = np.zeros(d).tolist()
        basis = np.eye(d)[:k].tolist()  # first k standard basis vectors
        basis_json = json.dumps(basis)

        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[{
            "mean": mean,
            "basis_json": basis_json,
            "k": k,
        }])

        query = np.random.randn(d).tolist()
        result = await project_query_embedding(neo4j, query, "basis_c1")

        assert result is not None
        assert len(result) == k
        # With identity basis, projection should equal first k elements of query
        for i in range(k):
            assert result[i] == pytest.approx(query[i], abs=1e-10)

    @pytest.mark.asyncio
    async def test_project_returns_none_for_missing_basis(self):
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[])

        result = await project_query_embedding(neo4j, [1.0] * 768, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_decompress_roundtrip(self):
        d = 768
        k = 64
        rng = np.random.default_rng(42)
        mean = rng.standard_normal(d)
        # Random orthogonal basis
        random_matrix = rng.standard_normal((k, d))
        Q, _ = np.linalg.qr(random_matrix.T)
        basis = Q.T[:k]  # (k, d)
        basis_json = json.dumps(basis.tolist())

        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[{
            "mean": mean.tolist(),
            "basis_json": basis_json,
        }])

        # Compress a vector that lies in the subspace
        original = mean + basis[0] * 3.0 + basis[1] * 2.0
        compressed = (basis @ (original - mean)).tolist()

        result = await decompress_embedding(neo4j, compressed, "basis_1")

        assert result is not None
        assert len(result) == d
        # Reconstruction should be close for in-subspace vectors
        original_norm = np.linalg.norm(original)
        result_arr = np.array(result)
        error = np.linalg.norm(original - result_arr) / original_norm
        assert error < 0.1  # <10% relative error
