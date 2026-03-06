"""
EcodiaOS — Neo4j Client

Async connection management for the knowledge graph.
All graph operations go through this client.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

if TYPE_CHECKING:
    from config import Neo4jConfig

logger = structlog.get_logger()


class Neo4jClient:
    """
    Async Neo4j driver wrapper.

    Provides connection pooling, health checks, and session management.
    Every system that reads/writes the knowledge graph uses this client.
    """

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection pool to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self._config.uri,
            auth=(self._config.username, self._config.password),
            max_connection_pool_size=self._config.max_connection_pool_size,
        )
        # Verify connectivity
        await self._driver.verify_connectivity()
        logger.info(
            "neo4j_connected",
            uri=self._config.uri,
            database=self._config.database,
        )

    async def close(self) -> None:
        """Close all connections."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j_disconnected")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j client not connected. Call connect() first.")
        return self._driver

    def session(self, **kwargs: Any) -> AsyncSession:
        """Get a new async session for the configured database."""
        return self.driver.session(
            database=self._config.database,
            **kwargs,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check connectivity and return status."""
        try:
            await self.driver.verify_connectivity()
            t0 = time.monotonic()
            async with self.session() as session:
                result = await session.run("RETURN 1 AS ping")
                await result.consume()
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            return {"status": "connected", "latency_ms": latency_ms}
        except Exception as e:
            logger.error("neo4j_health_check_failed", error=str(e))
            return {"status": "disconnected", "error": str(e)}

    async def execute_read(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query and return results as dicts."""
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records = [record.data() async for record in result]
            return records

    async def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a write query and return results as dicts."""
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records = [record.data() async for record in result]
            await result.consume()
            return records
