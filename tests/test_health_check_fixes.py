"""
Tests for health check resilience fixes.

Verifies:
1. Empty exception messages produce useful error strings
2. Memory health check handles Neo4j failures gracefully
3. Critical system set uses correct system IDs
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEmptyErrorMessages:
    """Verify that empty exception messages are replaced with type names."""

    def test_empty_error_fallback(self):
        """str(exc) returning '' should fall back to type(exc).__name__."""
        exc = Exception()
        error_msg = str(exc) or f"{type(exc).__name__} (no message)"
        assert error_msg == "Exception (no message)"

    def test_nonempty_error_preserved(self):
        """str(exc) with a real message should be used as-is."""
        exc = ValueError("bad value")
        error_msg = str(exc) or f"{type(exc).__name__} (no message)"
        assert error_msg == "bad value"

    def test_timeout_error_empty(self):
        """TimeoutError() with no args produces empty str."""
        exc = TimeoutError()
        assert str(exc) == ""
        error_msg = str(exc) or f"{type(exc).__name__} (no message)"
        assert error_msg == "TimeoutError (no message)"

    def test_runtime_error_empty(self):
        exc = RuntimeError()
        error_msg = str(exc) or f"{type(exc).__name__} (no message)"
        assert error_msg == "RuntimeError (no message)"


class TestMemoryHealthResilience:
    """Verify memory health check doesn't propagate Neo4j exceptions."""

    @pytest.mark.asyncio
    async def test_neo4j_exception_returns_degraded(self):
        """Memory health should return 'degraded' when Neo4j throws."""
        from systems.memory.service import MemoryService

        # Create a minimal mock of MemoryService
        svc = object.__new__(MemoryService)
        svc._instance_id = "test-instance"
        svc._neo4j = AsyncMock()
        svc._neo4j.health_check = AsyncMock(
            side_effect=ConnectionError("neo4j down")
        )

        result = await svc.health()
        assert result["status"] == "degraded"
        assert result["neo4j"]["status"] == "error"
        assert "neo4j down" in result["neo4j"]["error"]

    @pytest.mark.asyncio
    async def test_neo4j_timeout_returns_degraded(self):
        """Memory health should return 'degraded' on timeout."""
        from systems.memory.service import MemoryService

        svc = object.__new__(MemoryService)
        svc._instance_id = "test-instance"
        svc._neo4j = AsyncMock()
        svc._neo4j.health_check = AsyncMock(side_effect=TimeoutError())

        result = await svc.health()
        assert result["status"] == "degraded"
        assert result["neo4j"]["status"] == "error"
        assert "timed out" in result["neo4j"]["error"]

    @pytest.mark.asyncio
    async def test_neo4j_healthy_returns_healthy(self):
        """Memory health should return 'healthy' when Neo4j is connected."""
        from systems.memory.service import MemoryService

        svc = object.__new__(MemoryService)
        svc._instance_id = "test-instance"
        svc._neo4j = AsyncMock()
        svc._neo4j.health_check = AsyncMock(
            return_value={"status": "connected"}
        )

        result = await svc.health()
        assert result["status"] == "healthy"


class TestHealthEndpointTimeout:
    """Verify the /health endpoint completes within the external 10s timeout.

    Root cause of the incident: individual health checks used 5s timeouts,
    and TimescaleDB + memory.get_self() ran sequentially after the parallel
    gather — worst case was 15s, exceeding the external 10s health check.
    Fix: all checks run in parallel with 2s per-check timeouts.
    """

    @pytest.mark.asyncio
    async def test_all_checks_timeout_within_budget(self):
        """Even if every check times out, total should be ~2s not 15s."""
        # Simulate slow health checks that take 5s (exceeding the 2s timeout)
        async def slow_health():
            await asyncio.sleep(10)
            return {"status": "healthy"}

        class FakeSystem:
            async def health(self):
                return await slow_health()

            async def health_check(self):
                return await slow_health()

        class FakeState:
            memory = FakeSystem()
            equor = FakeSystem()
            voxis = FakeSystem()
            nova = FakeSystem()
            synapse = FakeSystem()
            thymos = FakeSystem()
            oneiros = FakeSystem()
            thread = FakeSystem()
            neo4j = FakeSystem()
            redis = FakeSystem()
            federation = FakeSystem()
            tsdb = None  # not configured
            config = MagicMock(instance_id="test")

        # Patch app.state with our fakes
        with patch("main.app") as mock_app:
            mock_app.state = FakeState()

            import time
            from main import health

            start = time.monotonic()
            result = await health()
            elapsed = time.monotonic() - start

            # All checks (including get_self) run in parallel with 2s timeout.
            # Total should be well under 10s (the external timeout)
            assert elapsed < 4.0, (
                f"Health endpoint took {elapsed:.1f}s — must stay under 10s "
                f"external timeout (target <4s with full parallelism)"
            )

            # Individual timeouts should be reported, not crash the endpoint
            assert result["systems"]["memory"]["status"] == "timeout"
            assert result["systems"]["equor"]["status"] == "timeout"
            assert result["data_stores"]["neo4j"]["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_tsdb_none_returns_not_initialized(self):
        """When tsdb is None, health should return not_initialized, not crash."""
        class FakeSystem:
            async def health(self):
                return {"status": "healthy"}

            async def health_check(self):
                return {"status": "connected"}

        class FakeState:
            memory = FakeSystem()
            equor = FakeSystem()
            voxis = FakeSystem()
            nova = FakeSystem()
            synapse = FakeSystem()
            thymos = FakeSystem()
            oneiros = FakeSystem()
            thread = FakeSystem()
            neo4j = FakeSystem()
            redis = FakeSystem()
            federation = FakeSystem()
            tsdb = None  # explicitly None
            config = MagicMock(instance_id="test")

        with patch("main.app") as mock_app:
            mock_app.state = FakeState()

            from main import health
            result = await health()

            # tsdb=None should not crash, should return not_initialized
            assert result["data_stores"]["timescaledb"]["status"] == "not_initialized"
            # Overall should not be degraded just because tsdb is not configured
            assert result["status"] == "healthy"


class TestCriticalSystemIds:
    """Verify _CRITICAL_SYSTEMS uses the correct registered system IDs."""

    def test_critical_systems_match_registered_ids(self):
        """_CRITICAL_SYSTEMS must use actual system_id values, not aliases."""
        from systems.synapse.health import _CRITICAL_SYSTEMS

        # "atune" was a bug - the PerceptionGateway registers as "fovea"
        assert "atune" not in _CRITICAL_SYSTEMS, (
            "_CRITICAL_SYSTEMS should use 'fovea', not 'atune' "
            "(PerceptionGateway.system_id is 'fovea')"
        )
        assert "fovea" in _CRITICAL_SYSTEMS
        assert "equor" in _CRITICAL_SYSTEMS
        assert "memory" in _CRITICAL_SYSTEMS

    def test_alive_statuses_include_degraded(self):
        """'degraded' must be in _ALIVE_STATUSES to prevent false safe_mode."""
        from systems.synapse.health import _ALIVE_STATUSES

        assert "degraded" in _ALIVE_STATUSES
        assert "healthy" in _ALIVE_STATUSES
        assert "running" in _ALIVE_STATUSES
        assert "safe_mode" in _ALIVE_STATUSES
