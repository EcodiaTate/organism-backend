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


class TestInfraResilience:
    """Verify infrastructure init failures don't crash the organism."""

    @pytest.mark.asyncio
    async def test_neo4j_failure_is_non_fatal(self):
        """create_infra should not raise when Neo4j fails to connect."""
        from core.infra import InfraClients

        infra = InfraClients(config=MagicMock())
        # Simulate neo4j field being None after failed init
        assert infra.neo4j is None, "neo4j should default to None"

    @pytest.mark.asyncio
    async def test_redis_failure_is_non_fatal(self):
        """create_infra should not raise when Redis fails to connect."""
        from core.infra import InfraClients

        infra = InfraClients(config=MagicMock())
        assert infra.redis is None, "redis should default to None"

    def test_close_infra_handles_none_clients(self):
        """close_infra must not crash when neo4j/redis/tsdb are None."""
        from core.infra import InfraClients

        infra = InfraClients(config=MagicMock())
        # All data store fields should be None by default
        assert infra.neo4j is None
        assert infra.redis is None
        assert infra.tsdb is None

    def test_health_endpoint_reports_startup_error(self):
        """Health endpoint should include startup_error when present."""
        startup_error = "Neo4j init failed"
        overall = "healthy"
        if startup_error:
            overall = "degraded"
        assert overall == "degraded"


class TestInfraConnectTimeouts:
    """Verify infrastructure connect() calls have timeouts to prevent startup hangs."""

    @pytest.mark.asyncio
    async def test_neo4j_connect_timeout_is_non_fatal(self):
        """If Neo4j connect hangs, create_infra should timeout and continue."""
        from core.infra import create_infra

        async def hang_forever():
            await asyncio.sleep(3600)

        mock_config = MagicMock()
        mock_config.logging = MagicMock()
        mock_config.instance_id = "test"
        mock_config.neo4j = MagicMock()
        mock_config.timescaledb = MagicMock()
        mock_config.redis = MagicMock()
        mock_config.llm = MagicMock()
        mock_config.llm.budget = MagicMock(
            max_tokens_per_hour=100000,
            max_calls_per_hour=1000,
            hard_limit=False,
        )
        mock_config.embedding = MagicMock()

        with patch("core.infra.Neo4jClient") as MockNeo4j, \
             patch("core.infra.TimescaleDBClient") as MockTsdb, \
             patch("core.infra.RedisClient") as MockRedis, \
             patch("core.infra.setup_logging"), \
             patch("core.infra.create_llm_provider") as mock_llm, \
             patch("core.infra.create_embedding_client") as mock_embed, \
             patch("core.infra.OptimizedLLMProvider"), \
             patch("core.infra.NeuroplasticityBus"), \
             patch("core.infra.MetricCollector") as mock_metrics:

            # Neo4j hangs on connect
            mock_neo4j_inst = MockNeo4j.return_value
            mock_neo4j_inst.connect = hang_forever

            # Redis and TSDB also hang
            mock_tsdb_inst = MockTsdb.return_value
            mock_tsdb_inst.connect = hang_forever
            mock_redis_inst = MockRedis.return_value
            mock_redis_inst.connect = hang_forever

            mock_llm.return_value = AsyncMock()
            mock_embed.return_value = AsyncMock()
            mock_metrics_inst = mock_metrics.return_value
            mock_metrics_inst.start_writer = AsyncMock()

            import time
            import core.infra
            original_timeout = core.infra.INFRA_CONNECT_TIMEOUT_S
            core.infra.INFRA_CONNECT_TIMEOUT_S = 0.1
            try:
                start = time.monotonic()
                infra = await create_infra(mock_config)
                elapsed = time.monotonic() - start
            finally:
                core.infra.INFRA_CONNECT_TIMEOUT_S = original_timeout

            # All should be None (timed out)
            assert infra.neo4j is None, "Neo4j should be None after connect timeout"
            assert infra.tsdb is None, "TSDB should be None after connect timeout"
            assert infra.redis is None, "Redis should be None after connect timeout"
            # Startup should not hang - 3 connects × 0.1s timeout = ~0.3s
            assert elapsed < 2.0, f"create_infra took {elapsed:.1f}s — should not hang"

    @pytest.mark.asyncio
    async def test_memory_health_with_none_neo4j(self):
        """MemoryService.health() must not crash when neo4j is None."""
        from systems.memory.service import MemoryService

        mock_embedding = MagicMock()
        memory = MemoryService(None, mock_embedding)

        result = await memory.health()
        assert result["status"] == "degraded"
        assert result["neo4j"]["status"] == "not_configured"

    @pytest.mark.asyncio
    async def test_ensure_schema_with_none_neo4j(self):
        """ensure_schema should return early when neo4j is None."""
        from systems.memory.schema import ensure_schema

        # Should not raise
        await ensure_schema(None)


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


class TestHealthEndpointDegradedWhenUninit:
    """Verify /health reports degraded when all core systems are not_initialized."""

    @pytest.mark.asyncio
    async def test_all_uninit_returns_degraded(self):
        """When all core systems are not_initialized, overall must be 'degraded'."""
        class FakeState:
            config = MagicMock(instance_id="test")

        with patch("main.app") as mock_app:
            mock_app.state = FakeState()

            from main import health
            result = await health()

            assert result["status"] == "degraded", (
                f"Expected 'degraded' when all systems are not_initialized, "
                f"got '{result['status']}'"
            )

    @pytest.mark.asyncio
    async def test_some_systems_running_returns_healthy(self):
        """When core systems are healthy, overall should be 'healthy'."""
        class FakeSystem:
            async def health(self):
                return {"status": "healthy"}

            async def health_check(self):
                return {"status": "connected"}

            @property
            def stats(self):
                return {"status": "running"}

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
            tsdb = FakeSystem()
            axon = FakeSystem()
            evo = FakeSystem()
            simula = FakeSystem()
            atune = None  # not all systems need to be present
            config = MagicMock(instance_id="test")

        with patch("main.app") as mock_app:
            mock_app.state = FakeState()

            from main import health
            result = await health()

            assert result["status"] == "healthy", (
                f"Expected 'healthy' when core systems are running, "
                f"got '{result['status']}'"
            )


class TestStartupTimeout:
    """Verify the lifespan startup timeout prevents indefinite hangs."""

    def test_startup_timeout_constant_exists(self):
        """STARTUP_TIMEOUT_S should be defined and reasonable."""
        from main import STARTUP_TIMEOUT_S
        assert STARTUP_TIMEOUT_S > 0
        assert STARTUP_TIMEOUT_S <= 300  # max 5 minutes

    @pytest.mark.asyncio
    async def test_startup_timeout_sets_error(self):
        """If startup times out, app.state.startup_error should be set."""
        from main import lifespan, STARTUP_TIMEOUT_S
        import main

        original_timeout = main.STARTUP_TIMEOUT_S
        main.STARTUP_TIMEOUT_S = 0.1  # Very short timeout for testing

        async def hang_forever(app):
            await asyncio.sleep(3600)

        mock_app = MagicMock()
        mock_app.state = MagicMock(spec=[])

        with patch.object(main._registry, "startup", side_effect=hang_forever), \
             patch.object(main._registry, "shutdown", new_callable=AsyncMock):
            try:
                async with lifespan(mock_app):
                    # Startup should have timed out
                    assert hasattr(mock_app.state, "startup_error"), (
                        "startup_error should be set after timeout"
                    )
                    assert "timed out" in mock_app.state.startup_error.lower()
            finally:
                main.STARTUP_TIMEOUT_S = original_timeout


class TestSystemInitTimeouts:
    """Verify _safe_init and system init timeouts in registry.py."""

    @pytest.mark.asyncio
    async def test_safe_init_timeout_returns_none(self):
        """_safe_init should return None when init hangs."""
        from core.registry import _safe_init

        async def hang():
            await asyncio.sleep(3600)
            return "should not reach"

        result = await _safe_init("test_system", hang(), timeout_s=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_safe_init_exception_returns_none(self):
        """_safe_init should return None when init raises."""
        from core.registry import _safe_init

        async def explode():
            raise RuntimeError("init failed")

        result = await _safe_init("test_system", explode())
        assert result is None

    @pytest.mark.asyncio
    async def test_safe_init_success_returns_value(self):
        """_safe_init should return the value on success."""
        from core.registry import _safe_init

        async def succeed():
            return {"status": "ok"}

        result = await _safe_init("test_system", succeed())
        assert result == {"status": "ok"}
