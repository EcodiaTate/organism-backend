"""
EcodiaOS — External Service Clients

Connection management for Neo4j, TimescaleDB, Redis, LLM, Embedding,
context compression (Stage 1C), and CDP Wallet (Phase 2 Metabolic Layer).
"""

from clients.algora import AlgoraClient
from clients.context_compression import CompressionMetrics, ContextCompressor
from clients.embedding import (
    EmbeddingClient,
    VoyageEmbeddingClient,
    cosine_similarity,
    create_embedding_client,
    create_voyage_client,
)
from clients.github import BountyIssue, BountySearchResult, GitHubClient
from clients.llm import (
    ExtendedThinkingProvider,
    LLMProvider,
    create_llm_provider,
    create_thinking_provider,
)
from clients.neo4j import Neo4jClient
from clients.redis import RedisClient
from clients.timescaledb import TimescaleDBClient
from clients.wallet import WalletClient

__all__ = [
    "AlgoraClient",
    "Neo4jClient",
    "TimescaleDBClient",
    "RedisClient",
    "LLMProvider",
    "ExtendedThinkingProvider",
    "create_llm_provider",
    "create_thinking_provider",
    "EmbeddingClient",
    "VoyageEmbeddingClient",
    "create_embedding_client",
    "create_voyage_client",
    "cosine_similarity",
    "ContextCompressor",
    "CompressionMetrics",
    "WalletClient",
    "GitHubClient",
    "BountyIssue",
    "BountySearchResult",
]
