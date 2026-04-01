"""
Unit tests for Stage 1 integration - context compression, embeddings, and history.

Tests:
  1C - ContextCompressor: message compression, metrics tracking, sliding window
  1B - Embedding-based dedup metrics, cosine similarity, code index
  1B.4 - EvolutionHistoryManager with embedding storage
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.context_compression import (
    CompressionMetrics,
    ContextCompressor,
)
from clients.embedding import MockEmbeddingClient, cosine_similarity
from systems.simula.evolution_types import (
    ChangeCategory,
    EvolutionRecord,
    RiskLevel,
)
from systems.simula.history import EvolutionHistoryManager

# ─── Context Compression Tests ────────────────────────────────────────────────


class TestContextCompressor:
    def test_no_compression_when_disabled(self):
        compressor = ContextCompressor(prune_ratio=0.3, enabled=False)
        messages = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
        ] * 10
        result = compressor.compress(messages)
        assert len(result) == len(messages)

    def test_no_compression_when_ratio_zero(self):
        compressor = ContextCompressor(prune_ratio=0.0, enabled=True)
        messages = [{"role": "user", "content": "x" * 1000}] * 20
        result = compressor.compress(messages)
        assert len(result) == len(messages)

    def test_no_compression_within_window(self):
        compressor = ContextCompressor(prune_ratio=0.5, window_size=6)
        messages = [{"role": "user", "content": "short"}] * 4
        result = compressor.compress(messages)
        # 4 messages < window_size=6, should not compress
        assert result == messages

    def test_compresses_old_tool_results(self):
        compressor = ContextCompressor(prune_ratio=0.5, window_size=4)

        # Build messages: old tool results (large) + recent window
        old_messages = []
        for i in range(8):
            old_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"tool_{i}",
                        "content": "def foo():\n    pass\n" * 100,  # ~2000 chars
                    }
                ],
            })

        recent = [
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "done"},
            {"role": "assistant", "content": "complete"},
        ]

        all_messages = old_messages + recent
        result = compressor.compress(all_messages)

        # Recent window should be preserved exactly
        assert result[-4:] == recent

        # Some old messages should be compressed (shorter content)
        original_chars = sum(len(str(m.get("content", ""))) for m in old_messages)
        compressed_chars = sum(len(str(m.get("content", ""))) for m in result[:-4])
        assert compressed_chars < original_chars

    def test_metrics_tracking(self):
        compressor = ContextCompressor(prune_ratio=0.5, window_size=2)
        messages = [
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "y" * 2000},
            {"role": "user", "content": "z" * 100},
            {"role": "assistant", "content": "w" * 100},
        ]
        compressor.compress(messages)

        metrics = compressor.metrics
        assert metrics.total_compressions == 1
        assert metrics.original_tokens > 0
        assert metrics.compressed_tokens > 0
        assert metrics.compressed_tokens <= metrics.original_tokens

    def test_compression_ratio_property(self):
        metrics = CompressionMetrics(
            original_tokens=1000,
            compressed_tokens=300,
            messages_compressed=5,
            total_compressions=1,
        )
        assert pytest.approx(metrics.compression_ratio, 0.01) == 0.7

    def test_effective_multiplier_property(self):
        metrics = CompressionMetrics(
            original_tokens=1000,
            compressed_tokens=250,
        )
        assert pytest.approx(metrics.effective_multiplier, 0.01) == 4.0

    def test_token_estimation(self):
        compressor = ContextCompressor()
        messages = [{"role": "user", "content": "x" * 400}]
        tokens = compressor.estimate_tokens(messages)
        # 400 chars / 4 chars_per_token = 100 tokens
        assert tokens == 100

    def test_aggressive_compress_under_budget(self):
        compressor = ContextCompressor(prune_ratio=0.8, window_size=2)
        messages = [
            {"role": "user", "content": "task prompt"},
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
            {"role": "user", "content": "z" * 5000},
            {"role": "assistant", "content": "w" * 5000},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
        ]
        # Very small budget forces aggressive compression
        result = compressor.compress(messages, token_budget=50)
        assert len(result) < len(messages)


# ─── Cosine Similarity Tests ─────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0, 0.0]
        assert pytest.approx(cosine_similarity(v, v), 0.001) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        assert pytest.approx(cosine_similarity(a, b), 0.001) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert pytest.approx(cosine_similarity(a, b), 0.001) == -1.0

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0


# ─── Mock Embedding Client Tests ─────────────────────────────────────────────


class TestMockEmbeddingClient:
    @pytest.mark.asyncio
    async def test_embed_returns_correct_dimension(self):
        client = MockEmbeddingClient(dimension=1024)
        result = await client.embed("hello")
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_embed_batch_returns_correct_count(self):
        client = MockEmbeddingClient(dimension=768)
        results = await client.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 768 for v in results)

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        client = MockEmbeddingClient()
        results = await client.embed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_returns_normalised_vector(self):
        import numpy as np

        client = MockEmbeddingClient(dimension=128)
        vec = await client.embed("test")
        norm = np.linalg.norm(vec)
        assert pytest.approx(norm, 0.01) == 1.0


# ─── Evolution History with Embeddings Tests ──────────────────────────────────


class TestEvolutionHistoryEmbeddings:
    def _make_record(self) -> EvolutionRecord:
        return EvolutionRecord(
            proposal_id="prop-1",
            category=ChangeCategory.ADD_EXECUTOR,
            description="Add weather lookup executor",
            from_version=1,
            to_version=2,
            files_changed=["systems/axon/executors/weather.py"],
            simulation_risk=RiskLevel.LOW,
        )

    @pytest.mark.asyncio
    async def test_record_stores_embedding_when_client_provided(self):
        mock_neo4j = MagicMock()
        mock_neo4j.execute_write = AsyncMock(return_value=[])

        mock_embedding = MockEmbeddingClient(dimension=1024)

        history = EvolutionHistoryManager(
            neo4j=mock_neo4j,
            embedding_client=mock_embedding,
        )

        record = self._make_record()
        await history.record(record)

        # Should have called execute_write with embedding parameter
        call_args = mock_neo4j.execute_write.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("parameters", {})
        assert "embedding" in params
        assert len(params["embedding"]) == 1024

    @pytest.mark.asyncio
    async def test_record_without_embedding_client(self):
        mock_neo4j = MagicMock()
        mock_neo4j.execute_write = AsyncMock(return_value=[])

        history = EvolutionHistoryManager(neo4j=mock_neo4j)

        record = self._make_record()
        await history.record(record)

        # Should have called execute_write without embedding
        call_args = mock_neo4j.execute_write.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("parameters", {})
        assert "embedding" not in params

    @pytest.mark.asyncio
    async def test_ensure_vector_index_idempotent(self):
        mock_neo4j = MagicMock()
        mock_neo4j.execute_write = AsyncMock(return_value=[])

        history = EvolutionHistoryManager(
            neo4j=mock_neo4j,
            embedding_client=MockEmbeddingClient(dimension=1024),
        )

        # Call twice
        await history.ensure_vector_index()
        await history.ensure_vector_index()

        # Should only execute the CREATE VECTOR INDEX once
        write_calls = [
            call for call in mock_neo4j.execute_write.call_args_list
            if "VECTOR INDEX" in str(call)
        ]
        assert len(write_calls) == 1

    @pytest.mark.asyncio
    async def test_find_similar_records_empty_without_client(self):
        mock_neo4j = MagicMock()
        history = EvolutionHistoryManager(neo4j=mock_neo4j)

        results = await history.find_similar_records("weather executor")
        assert results == []


# ─── Proposal Intelligence Dedup Metrics Tests ───────────────────────────────


class TestDedupMetrics:
    def test_dedup_stats_initialised(self):
        from systems.simula.proposal_intelligence import ProposalIntelligence

        intel = ProposalIntelligence(llm=MagicMock())
        stats = intel.get_dedup_stats()
        assert "total_dedup_calls" in stats
        assert "embedding_dedup_calls" in stats
        assert "llm_dedup_calls" in stats
        assert stats["total_dedup_calls"] == 0
