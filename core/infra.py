"""
EcodiaOS - Infrastructure Client Initialization

Creates and connects all shared infrastructure clients (data stores,
LLM, embedding, caches) that cognitive systems depend on.

Extracted from main.py to keep the entry point thin and enable
per-system hot-reload without touching the startup module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.wallet import WalletClient

from clients.embedding import create_embedding_client
from clients.llm import create_llm_provider
from clients.neo4j import Neo4jClient
from clients.optimized_llm import OptimizedLLMProvider
from clients.prompt_cache import PromptCache
from clients.redis import RedisClient
from clients.timescaledb import TimescaleDBClient
from clients.token_budget import TokenBudget
from core.hotreload import NeuroplasticityBus
from telemetry.llm_metrics import LLMMetricsCollector
from telemetry.logging import setup_logging
from telemetry.metrics import MetricCollector

logger = structlog.get_logger()


@dataclass
class InfraClients:
    """Container for all shared infrastructure clients."""

    config: Any
    neo4j: Neo4jClient = field(init=False)
    tsdb: TimescaleDBClient = field(init=False)
    redis: RedisClient = field(init=False)
    llm: OptimizedLLMProvider = field(init=False)
    raw_llm: Any = field(init=False)
    embedding: Any = field(init=False)
    token_budget: TokenBudget = field(init=False)
    llm_metrics: LLMMetricsCollector = field(init=False)
    prompt_cache: PromptCache | None = field(init=False, default=None)
    neuroplasticity_bus: NeuroplasticityBus = field(init=False)
    metrics: MetricCollector = field(init=False)
    wallet: WalletClient | None = field(init=False, default=None)
    tollbooth_ledger: Any = field(init=False, default=None)


async def create_infra(config: Any) -> InfraClients:
    """
    Instantiate and connect all infrastructure clients.

    Follows the Infrastructure Architecture spec section 3.2:
    1. Logging
    2. Data stores (Neo4j, TimescaleDB, Redis)
    3. LLM + embedding
    4. Neuroplasticity bus
    5. Telemetry

    Raises RuntimeError on critical failures (data store connections).
    """
    infra = InfraClients(config=config)

    # ── 1. Logging ────────────────────────────────────────────
    setup_logging(config.logging, instance_id=config.instance_id)
    logger.info(
        "ecodiaos_starting",
        instance_id=config.instance_id,
        config_path=os.environ.get("ECODIAOS_CONFIG_PATH", "config/default.yaml"),
    )

    # ── 2. Data stores ────────────────────────────────────────
    try:
        infra.neo4j = Neo4jClient(config.neo4j)
        await infra.neo4j.connect()
    except Exception as exc:
        logger.error(
            "neo4j_init_failed",
            error=str(exc),
            dependents="Memory, Nova, Evo, Thread",
            exc_info=True,
        )
        raise RuntimeError("Neo4j init failed (required by Memory, Nova, Evo, Thread)") from exc

    try:
        infra.tsdb = TimescaleDBClient(config.timescaledb)
        await infra.tsdb.connect()
    except Exception as exc:
        logger.error(
            "timescaledb_init_failed",
            error=str(exc),
            dependents="Metrics, Skia",
            exc_info=True,
        )
        raise RuntimeError("TimescaleDB init failed (required by Metrics, Skia)") from exc

    try:
        infra.redis = RedisClient(config.redis)
        await infra.redis.connect()
    except Exception as exc:
        logger.error(
            "redis_init_failed",
            error=str(exc),
            dependents="NeuroplasticityBus, Alive, Voxis, Axon, PromptCache",
            exc_info=True,
        )
        raise RuntimeError(
            "Redis init failed (required by NeuroplasticityBus, Alive, Voxis, Axon)"
        ) from exc

    # ── 2a. Tollbooth credit ledger ───────────────────────────
    from api.monetization.ledger import CreditLedger

    infra.tollbooth_ledger = CreditLedger(
        redis=infra.redis.client,
        prefix=config.redis.prefix,
    )

    # ── 2b. Neuroplasticity bus ───────────────────────────────
    infra.neuroplasticity_bus = NeuroplasticityBus(redis_client=infra.redis)
    infra.neuroplasticity_bus.start()

    # ── 3. LLM + embedding ────────────────────────────────────
    infra.raw_llm = create_llm_provider(config.llm)

    infra.token_budget = TokenBudget(
        max_tokens_per_hour=config.llm.budget.max_tokens_per_hour,
        max_calls_per_hour=config.llm.budget.max_calls_per_hour,
        hard_limit=config.llm.budget.hard_limit,
    )

    infra.llm_metrics = LLMMetricsCollector()

    try:
        infra.prompt_cache = PromptCache(
            redis_client=infra.redis.client,
            prefix="eos:llmcache",
        )
        logger.info("prompt_cache_initialized")
    except Exception as exc:
        logger.warning("prompt_cache_init_failed", error=str(exc))

    infra.llm = OptimizedLLMProvider(
        inner=infra.raw_llm,
        cache=infra.prompt_cache,
        budget=infra.token_budget,
        metrics=infra.llm_metrics,
    )

    logger.info(
        "llm_optimization_active",
        budget_tokens_per_hour=config.llm.budget.max_tokens_per_hour,
        budget_calls_per_hour=config.llm.budget.max_calls_per_hour,
        cache_enabled=infra.prompt_cache is not None,
    )

    infra.embedding = create_embedding_client(config.embedding)

    # ── 4. Telemetry ──────────────────────────────────────────
    infra.metrics = MetricCollector(infra.tsdb)
    await infra.metrics.start_writer()

    return infra


async def close_infra(infra: InfraClients) -> None:
    """Close all infrastructure connections (Phase 2 of shutdown)."""

    async def _safe(name: str, coro: Any) -> None:
        try:
            await coro
        except Exception as e:
            logger.warning(f"{name}_close_failed", error=str(e))

    closers = [
        ("embedding", infra.embedding.close()),
        ("llm", infra.llm.close()),
        ("redis", infra.redis.close()),
        ("tsdb", infra.tsdb.close()),
        ("neo4j", infra.neo4j.close()),
    ]
    if infra.wallet is not None:
        closers.insert(0, ("wallet", infra.wallet.close()))

    import asyncio
    async with asyncio.timeout(1.0):
        await asyncio.gather(
            *[_safe(name, coro) for name, coro in closers],
            return_exceptions=True,
        )
