"""
EcodiaOS — System Registry

Orchestrates the full startup and shutdown sequence for all 29
cognitive systems.  Replaces the monolithic lifespan() in main.py
with a structured, phase-based initialization.

Each system is initialized in dependency order, wired to its
peers, and registered with the DegradationManager for per-system
hot-reload.  The NeuroplasticityBus + DegradationManager handle
runtime code evolution without restarting the process.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import structlog

from config import load_config, load_seed
from core.helpers import MemoryWorkspaceAdapter, resolve_governance_config, seed_atune_cache
from core.infra import InfraClients, close_infra, create_infra
from core.inner_life import inner_life_loop
from core.interoception_loop import interoception_loop
from core.scheduled_tasks import register_scheduled_tasks
from core.smoke_test import run_smoke_tests
from core.wiring import (
    create_bounty_submit_fn,
    create_expression_feedback_callback,
    declare_dependencies,
    wire_federation_phase,
    wire_financial_memory,
    wire_intelligence_loops,
    wire_oikos_phase,
    wire_soma_phase,
    wire_synapse_phase,
    wire_thread,
    wire_thymos_phase,
)
from utils.supervision import supervised_task

logger = structlog.get_logger()


class SystemRegistry:
    """
    Owns the lifecycle of every cognitive system and infrastructure client.

    Startup: ``await registry.startup(app)``
    Shutdown: ``await registry.shutdown(app)``

    The ``app.state`` namespace is populated during startup so API
    endpoints can access any system via ``request.app.state.<name>``.
    """

    def __init__(self) -> None:
        self.infra: InfraClients | None = None
        # Background tasks that need cancellation on shutdown
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    # ══════════════════════════════════════════════════════════════
    #  STARTUP
    # ══════════════════════════════════════════════════════════════

    async def startup(self, app: Any) -> None:
        """Full organism startup — infrastructure → systems → wiring → smoke tests."""
        config_path = os.environ.get("ECODIAOS_CONFIG_PATH", "config/default.yaml")
        config = load_config(config_path)
        app.state.config = config
        app.state.identity_comm_config = config.identity_comm

        # ── Infrastructure ────────────────────────────────────
        self.infra = await create_infra(config)
        infra = self.infra

        # Expose infra on app.state for API endpoints
        app.state.neo4j = infra.neo4j
        app.state.tsdb = infra.tsdb
        app.state.redis = infra.redis
        app.state.llm = infra.llm
        app.state.raw_llm = infra.raw_llm
        app.state.embedding = infra.embedding

        # Initialize log analyzer for real-time debugging
        from telemetry.log_analyzer import initialize_analyzer
        analyzer = await initialize_analyzer(infra.redis)
        app.state.log_analyzer = analyzer

        # Wire structlog → Redis Streams so all logs are queryable
        from telemetry.logging import get_redis_stream_handler
        stream_handler = get_redis_stream_handler()
        if stream_handler is not None:
            stream_handler.set_analyzer(analyzer)

        logger.info("log_analyzer_initialized")
        app.state.token_budget = infra.token_budget
        app.state.llm_metrics = infra.llm_metrics
        app.state.metrics = infra.metrics
        app.state.neuroplasticity_bus = infra.neuroplasticity_bus
        app.state.tollbooth_ledger = infra.tollbooth_ledger

        # ── Phase 1: Foundation Systems ───────────────────────
        memory = await self._init_memory(infra)
        app.state.memory = memory

        logos = await self._init_logos(config, memory)
        app.state.logos = logos

        equor = await self._init_equor(config, infra, logos)
        app.state.equor = equor

        atune = self._init_atune(config, infra, memory)
        app.state.atune = atune

        eis = await self._init_eis(config, infra)
        app.state.eis = eis

        sacm_parts = self._init_sacm()
        for key, val in sacm_parts.items():
            setattr(app.state, key, val)

        # ── Phase 2: Core Cognitive Systems ───────────────────
        voxis = await self._init_voxis(config, infra, memory)
        app.state.voxis = voxis

        nova = await self._init_nova(config, infra, memory, equor, voxis)
        app.state.nova = nova

        axon = await self._init_axon(config, infra, memory, voxis)
        app.state.axon = axon

        # EIS → Metrics wiring
        eis.set_metrics(infra.metrics)

        # Start Atune (deferred until memory wired)
        atune.set_memory_service(memory)
        await atune.startup()

        # Core cross-wiring (Evo not yet available — deferred to Phase 3)
        atune.subscribe(nova)
        atune.set_active_goals(nova.active_goal_summaries)
        nova.set_goal_sync_callback(atune.set_active_goals)
        axon.set_nova(nova)
        axon.set_atune(atune)
        nova.set_axon(axon)
        voxis.register_feedback_callback(create_expression_feedback_callback(atune, nova))

        # Skia state restoration check
        await self._check_skia_restore(config, infra)

        # Birth or load instance
        await self._birth_or_load(config, infra, memory, equor, atune)

        # ── Phase 3: Learning & Identity ──────────────────────
        evo = await self._init_evo(config, infra, memory)
        app.state.evo = evo

        # Remaining core wiring that needs Evo
        atune.subscribe(evo)
        evo.set_nova(nova)
        nova.set_evo(evo)
        evo.set_voxis(voxis)
        atune.subscribe(voxis)
        equor.set_evo(evo)
        equor.set_axon(axon)
        axon.set_template_library(equor.template_library)
        atune.set_market_pattern_detector(equor.template_library, axon)
        from systems.identity.communication import send_admin_sms

        equor.set_notification_hook(lambda msg: send_admin_sms(config.identity_comm, msg))

        # Seed initial goals
        await self._seed_goals(config, nova, atune)

        thread = await self._init_thread(config, infra, memory)
        app.state.thread = thread
        wire_thread(
            thread=thread,
            voxis=voxis,
            equor=equor,
            atune=atune,
            evo=evo,
            nova=nova,
        )

        # ── Phase 4: Simula (Self-Evolution) ──────────────────
        simula = await self._init_simula(config, infra, memory, app)
        app.state.simula = simula
        axon.set_simula_service(simula)
        evo.set_simula(simula)
        evo.schedule_arxiv_scan()

        # Nova heartbeat
        self._tasks["nova_heartbeat"] = supervised_task(
            nova.start_heartbeat(),
            name="nova_heartbeat",
            restart=True,
            max_restarts=5,
            event_bus=None,
            source_system="nova",
        )
        logger.info(
            "nova_heartbeat_started", interval_seconds=config.nova.heartbeat_interval_seconds
        )

        # ── Phase 5: Synapse — The Coordination Bus ──────────
        synapse = await self._init_synapse(config, infra, atune)
        app.state.synapse = synapse

        # Register systems for health monitoring
        for system in [memory, equor, voxis, nova, axon, evo, thread, logos]:
            synapse.register_system(system)

        wire_synapse_phase(
            synapse=synapse,
            neuroplasticity_bus=infra.neuroplasticity_bus,
            atune=atune,
            eis=eis,
            equor=equor,
            evo=evo,
            thread=thread,
            simula=simula,
            logos=logos,
            sacm_compute_manager=sacm_parts["sacm_compute_manager"],
            sacm_client=sacm_parts["sacm_client"],
            axon=axon,
            nova=nova,
            llm_client=infra.llm,
            config=config,
        )
        declare_dependencies(synapse)

        await logos.start()
        await synapse.start_clock()
        await synapse.start_health_monitor()

        # Model hot-swap
        await self._init_hot_swap(config, infra, synapse, app)

        # ── Phase 6: Immune & Dream Systems ──────────────────
        thymos = await self._init_thymos(config, infra, synapse, equor, evo, atune, nova, simula)
        app.state.thymos = thymos
        synapse.register_system(thymos)
        wire_thymos_phase(thymos=thymos, nova=nova, evo=evo, synapse=synapse)

        oneiros = await self._init_oneiros(
            config, infra, synapse, equor, evo, nova, atune, thymos, memory, simula
        )
        app.state.oneiros = oneiros
        synapse.register_system(oneiros)

        kairos = self._init_kairos(synapse, logos, oneiros)
        app.state.kairos = kairos
        synapse.register_system(kairos)

        # ── Phase 7: Soma — Interoceptive Substrate ──────────
        soma = await self._init_soma(config, infra, atune, synapse, nova, thymos, equor)
        app.state.soma = soma
        synapse.register_system(soma)

        # Exteroception
        await self._init_exteroception(config, synapse, soma, app)

        wire_soma_phase(
            soma=soma,
            atune=atune,
            synapse=synapse,
            nova=nova,
            memory=memory,
            evo=evo,
            oneiros=oneiros,
            thymos=thymos,
            voxis=voxis,
            sacm_accounting=sacm_parts["sacm_accounting"],
        )

        # ── Phase 8: Telos + Fovea — Intelligence Loops ──────
        telos = await self._init_telos(synapse, infra)
        app.state.telos = telos
        synapse.register_system(telos)

        fovea = await self._init_fovea(logos, synapse, atune)
        app.state.fovea = fovea
        synapse.register_system(fovea)

        wire_intelligence_loops(
            logos=logos,
            fovea=fovea,
            atune=atune,
            axon=axon,
            oneiros=oneiros,
            thread=thread,
            telos=telos,
            nova=nova,
            evo=evo,
            simula=simula,
            thymos=thymos,
            soma=soma,
            kairos=kairos,
        )
        await telos.start()
        logger.info("telos_computation_loop_started", logos_wired=True, fovea_wired=True)

        # ── Phase 9: Federation + Economic Layer ──────────────
        federation = await self._init_federation(config, infra, memory, equor)
        app.state.federation = federation
        wire_federation_phase(
            federation=federation,
            atune=atune,
            thymos=thymos,
            sacm_compute_manager=sacm_parts["sacm_compute_manager"],
            synapse=synapse,
            config=config,
        )

        nexus = await self._init_nexus(
            config,
            logos,
            fovea,
            federation,
            thymos,
            oneiros,
            evo,
            equor,
            telos,
            synapse,
        )
        app.state.nexus = nexus
        synapse.register_system(nexus)

        # Wallet
        await self._init_wallet(config, infra, axon, app)

        # Financial memory encoding
        wire_financial_memory(memory=memory, axon=axon, synapse=synapse)

        # Certificate Manager
        await self._init_certificate_manager(config, infra, federation, synapse, app)

        # Oikos
        oikos = await self._init_oikos(config, infra, synapse, app)
        app.state.oikos = oikos
        synapse.register_system(oikos)

        if app.state.certificate_manager is not None:
            oikos.set_certificate_manager(app.state.certificate_manager)

        wire_oikos_phase(
            oikos=oikos,
            nova=nova,
            oneiros=oneiros,
            soma=soma,
            thymos=thymos,
            sacm_accounting=sacm_parts["sacm_accounting"],
            sacm_prewarm_engine=sacm_parts["sacm_prewarm_engine"],
        )

        # ── Phase 10: Alive WebSocket ────────────────────────
        alive_ws = await self._init_alive_ws(
            config, infra, soma, atune, synapse, telos, thymos, nova, axon, oikos, simula
        )
        app.state.alive_ws = alive_ws

        # Phantom Liquidity
        await self._init_phantom_liquidity(config, infra, atune, oikos, synapse, app)

        # Skia
        await self._init_skia(config, infra, synapse, app)

        # Platform connectors
        await self._init_connectors(config, infra, synapse, app)

        # GitHub connector + bounty submission
        self._init_github_connector(config, infra, synapse, axon, oikos, app)

        # ── Phase 11: Background Tasks ───────────────────────
        # Interoception loop: log analysis → Soma signals
        self._tasks["interoception"] = supervised_task(
            interoception_loop(soma=soma, analyzer=app.state.log_analyzer),
            name="interoception_loop",
            restart=True,
            max_restarts=3,
            event_bus=synapse.event_bus,
            source_system="interoception",
        )

        self._tasks["inner_life"] = supervised_task(
            inner_life_loop(
                atune=atune,
                nova=nova,
                synapse=synapse,
                thymos=thymos,
                evo=evo,
                oneiros=oneiros,
                axon=axon,
                voxis=voxis,
                federation=federation,
                equor=equor,
                thread=thread,
            ),
            name="inner_life_generator",
            restart=True,
            max_restarts=5,
            event_bus=synapse.event_bus,
            source_system="inner_life",
        )

        # File watcher + Scheduler
        percepts_dir = Path(os.environ.get("ECODIAOS_PERCEPTS_DIR", "config/percepts")).resolve()

        from clients.file_watcher import FileWatcher
        from clients.scheduler import PerceptionScheduler

        file_watcher = FileWatcher(watch_dir=percepts_dir, atune=atune)
        await file_watcher.start()
        app.state.file_watcher = file_watcher

        scheduler = PerceptionScheduler(atune=atune)
        register_scheduled_tasks(scheduler, axon, oikos)
        await scheduler.start()
        app.state.scheduler = scheduler

        # Fleet shield
        from systems.simula.distributed_shield import FleetShieldManager

        fleet_shield = FleetShieldManager(infra.redis)
        fleet_shield.start()
        app.state.fleet_shield = fleet_shield

        # Metrics publisher
        metrics_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        app.state.metrics_queue = metrics_queue
        from telemetry.publisher import publish_metrics_loop

        self._tasks["metrics_publisher"] = supervised_task(
            publish_metrics_loop(infra.redis, metrics_queue),
            name="metrics_publisher",
            restart=True,
            max_restarts=5,
            event_bus=synapse.event_bus,
            source_system="telemetry",
        )

        # Benchmarks
        await self._init_benchmarks(config, infra, nova, evo, oikos, simula, synapse, alive_ws, app)

        logger.info(
            "ecodiaos_ready",
            phase="17_multi_channel_perception",
            federation_enabled=config.federation.enabled,
            immune_system="active",
            inner_life="active",
            file_watcher=str(percepts_dir),
        )

        # ── Phase 12: Smoke Tests ────────────────────────────
        await run_smoke_tests(
            oikos=oikos,
            simula=simula,
            thymos=thymos,
            nova=nova,
            evo=evo,
            synapse=synapse,
        )

    # ══════════════════════════════════════════════════════════════
    #  SHUTDOWN
    # ══════════════════════════════════════════════════════════════

    async def shutdown(self, app: Any) -> None:
        """Concurrent shutdown of all systems and infrastructure."""
        logger.info("ecodiaos_shutting_down")

        async def _safe(name: str, coro: Any) -> None:
            try:
                await coro
            except Exception as e:
                logger.warning(f"{name}_shutdown_failed", error=str(e))

        try:
            # Cancel background tasks
            for _name, task in self._tasks.items():
                task.cancel()

            phase1: list[Any] = []
            for name, task in self._tasks.items():
                phase1.append(_safe(f"{name}_task", asyncio.wait_for(task, timeout=0.5)))

            # Stop peripheral services
            for attr in ("exteroception",):
                if hasattr(app.state, attr):
                    svc = getattr(app.state, attr)
                    if hasattr(svc, "stop"):
                        phase1.append(_safe(attr, svc.stop()))

            if hasattr(app.state, "scheduler"):
                phase1.append(_safe("scheduler", app.state.scheduler.stop()))
            if hasattr(app.state, "file_watcher"):
                phase1.append(_safe("file_watcher", app.state.file_watcher.stop()))

            # Cognitive systems
            for name in [
                "federation",
                "alive_ws",
                "axon",
                "voxis",
                "thread",
                "simula",
                "fleet_shield",
                "logos",
                "evo",
                "atune",
                "thymos",
                "equor",
                "oikos",
                "nova",
                "oneiros",
                "soma",
            ]:
                svc = getattr(app.state, name, None)
                if svc is not None and hasattr(svc, "shutdown"):
                    phase1.append(_safe(name, svc.shutdown()))

            if hasattr(app.state, "alive_ws") and hasattr(app.state.alive_ws, "stop"):
                phase1.append(_safe("alive_ws", app.state.alive_ws.stop()))

            if app.state.skia is not None:
                phase1.append(_safe("skia", app.state.skia.shutdown()))
            if getattr(app.state, "phantom_liquidity", None) is not None:
                phase1.append(_safe("phantom_liquidity", app.state.phantom_liquidity.shutdown()))

            phase1.append(_safe("neuroplasticity_bus", app.state.neuroplasticity_bus.stop()))
            phase1.append(_safe("synapse", app.state.synapse.stop()))

            if hasattr(app.state, "benchmarks"):
                phase1.append(_safe("benchmarks", app.state.benchmarks.shutdown()))
            phase1.append(_safe("metrics", self.infra.metrics.stop()))

            async with asyncio.timeout(2.0):
                await asyncio.gather(*phase1, return_exceptions=True)

            # Phase 2: Infrastructure connections
            if self.infra is not None:
                await close_infra(self.infra)

        except TimeoutError:
            logger.warning("shutdown_timeout_exceeded", note="some systems did not stop in time")
        except Exception as exc:
            logger.error("shutdown_sequence_error", error=str(exc), exc_info=True)
        finally:
            logger.info("ecodiaos_shutdown_complete")

    # ══════════════════════════════════════════════════════════════
    #  SYSTEM INITIALIZERS  (private)
    # ══════════════════════════════════════════════════════════════

    async def _init_memory(self, infra: InfraClients) -> Any:
        from systems.memory.service import MemoryService

        memory = MemoryService(infra.neo4j, infra.embedding)
        await memory.initialize()
        return memory

    async def _init_logos(self, config: Any, memory: Any) -> Any:
        from systems.logos.service import LogosService
        from systems.logos.types import LogosConfig

        logos_config = LogosConfig(
            total_budget_ku=getattr(config, "logos_total_budget_ku", 1_000_000),
        )
        logos = LogosService(config=logos_config)
        await logos.initialize()
        logos.set_memory(memory)
        return logos

    async def _init_equor(self, config: Any, infra: InfraClients, logos: Any) -> Any:
        from systems.equor.service import EquorService

        governance_config = resolve_governance_config(config)
        equor = EquorService(
            neo4j=infra.neo4j,
            llm=infra.llm,
            config=config.equor,
            governance_config=governance_config,
            neuroplasticity_bus=infra.neuroplasticity_bus,
            redis=infra.redis,
        )
        await equor.initialize()
        return equor

    def _init_atune(self, config: Any, infra: InfraClients, memory: Any) -> Any:
        from systems.atune.service import AtuneConfig, AtuneService

        atune_config = AtuneConfig(
            workspace_buffer_size=getattr(config, "atune_workspace_buffer_size", 32),
            spontaneous_recall_base_probability=getattr(
                config, "atune_spontaneous_recall_prob", 0.02
            ),
            max_percept_queue_size=getattr(config, "atune_max_percept_queue", 100),
        )
        workspace_memory = MemoryWorkspaceAdapter(memory)
        atune = AtuneService(
            embed_fn=infra.embedding.embed,
            memory_client=workspace_memory,
            llm_client=infra.llm,  # type: ignore[arg-type]
            config=atune_config,
        )
        return atune

    async def _init_eis(self, config: Any, infra: InfraClients) -> Any:
        from systems.eis.models import EISConfig
        from systems.eis.quarantine import LLMProviderAdapter, QuarantineEvaluator
        from systems.eis.service import EISService

        eis = EISService(
            config=EISConfig(),
            pathogen_store=None,
            quarantine_evaluator=QuarantineEvaluator(
                llm=LLMProviderAdapter(infra.llm),
                model_name=config.llm.model,
            ),
            embed_client=infra.embedding,
            metrics=None,
        )
        await eis.initialize()
        return eis

    def _init_sacm(self) -> dict[str, Any]:
        from systems.sacm.accounting import SACMCostAccounting
        from systems.sacm.compute_manager import ComputeResourceManager
        from systems.sacm.config import SACMPreWarmConfig
        from systems.sacm.oracle import ComputeMarketOracle
        from systems.sacm.pre_warming import PreWarmingEngine
        from systems.sacm.remote_executor import (
            RemoteExecutionConfig,
            RemoteExecutionManager,
            RemoteProviderTransport,
        )
        from systems.sacm.service import SACMClient, SACMMetrics, SACMWorkloadHistoryStore
        from systems.sacm.verification.consensus import ConsensusVerifier
        from systems.sacm.verification.deterministic import DeterministicReplayVerifier
        from systems.sacm.verification.probabilistic import ProbabilisticAuditVerifier

        class _StubTransport(RemoteProviderTransport):
            async def submit_workload(
                self,
                provider_id: str,
                endpoint: str,
                encrypted_payload: bytes,
                metadata: dict[str, str],
            ) -> bytes:
                return b""

        async def _canary_gen(n: int) -> list[bytes]:
            return [b""] * n

        async def _local_runner(payload: bytes) -> bytes:
            return payload

        oracle = ComputeMarketOracle()
        metrics = SACMMetrics()
        consensus = ConsensusVerifier(
            deterministic=DeterministicReplayVerifier(replay_fn=_local_runner),
            probabilistic=ProbabilisticAuditVerifier(seal_key=b"\x00" * 32),
        )
        execution_manager = RemoteExecutionManager(
            config=RemoteExecutionConfig(),
            transport=_StubTransport(),
            canary_generator=_canary_gen,
            local_executor=_local_runner,
            shared_consensus_verifier=consensus,
        )
        accounting = SACMCostAccounting()
        history = SACMWorkloadHistoryStore()
        client = SACMClient(
            oracle=oracle,
            execution_manager=execution_manager,
            metrics=metrics,
            history=history,
        )
        compute_manager = ComputeResourceManager()
        prewarm_config = SACMPreWarmConfig()
        prewarm_engine = PreWarmingEngine(oracle=oracle, config=prewarm_config)

        return {
            "sacm_accounting": accounting,
            "sacm_client": client,
            "sacm_compute_manager": compute_manager,
            "sacm_oracle": oracle,
            "sacm_prewarm_engine": prewarm_engine,
            "sacm_prewarm_config": prewarm_config,
            "sacm_history": history,
            "sacm_consensus_verifier": consensus,
            "sacm_oracle_last_refresh_request": 0.0,
            "sacm_metrics": metrics,
        }

    async def _init_voxis(self, config: Any, infra: InfraClients, memory: Any) -> Any:
        from systems.voxis.service import VoxisService

        voxis = VoxisService(
            memory=memory,
            redis=infra.redis,
            llm=infra.llm,
            config=config.voxis,
            neuroplasticity_bus=infra.neuroplasticity_bus,
        )
        await voxis.initialize()
        return voxis

    async def _init_nova(
        self, config: Any, infra: InfraClients, memory: Any, equor: Any, voxis: Any
    ) -> Any:
        from systems.nova.service import NovaService

        try:
            nova = NovaService(
                memory=memory,
                equor=equor,
                voxis=voxis,
                llm=infra.llm,
                config=config.nova,
                neuroplasticity_bus=infra.neuroplasticity_bus,
            )
            await nova.initialize()
            nova.set_embed_fn(infra.embedding.embed)
            return nova
        except Exception as exc:
            logger.error("nova_init_failed", error=str(exc), exc_info=True)
            raise RuntimeError("Nova init failed") from exc

    async def _init_axon(self, config: Any, infra: InfraClients, memory: Any, voxis: Any) -> Any:
        from systems.axon.service import AxonService

        axon = AxonService(
            config=config.axon,
            memory=memory,
            voxis=voxis,
            neuroplasticity_bus=infra.neuroplasticity_bus,
            redis_client=infra.redis,
            wallet=None,
            instance_id=config.instance_id,
            github_config=config.external_platforms,
            llm=infra.llm,
            mev_config=config.mev,
        )
        await axon.initialize()
        return axon

    async def _init_evo(self, config: Any, infra: InfraClients, memory: Any) -> Any:
        from systems.evo.service import EvoService

        evo = EvoService(
            config=config.evo,
            llm=infra.llm,
            memory=memory,
            instance_name=config.instance_id,
            neuroplasticity_bus=infra.neuroplasticity_bus,
        )
        await evo.initialize()
        evo.schedule_consolidation_loop()
        return evo

    async def _init_thread(self, config: Any, infra: InfraClients, memory: Any) -> Any:
        from systems.thread.service import ThreadService

        thread = ThreadService(
            memory=memory,
            instance_name=config.instance_id,
            neuroplasticity_bus=infra.neuroplasticity_bus,
        )
        await thread.initialize()
        return thread

    async def _init_simula(self, config: Any, infra: InfraClients, memory: Any, app: Any) -> Any:
        mode = os.getenv("SIMULA_MODE", "local")
        if mode not in ("proxy", "local"):
            raise RuntimeError(
                f"Unrecognised SIMULA_MODE={mode!r}. Valid values: 'proxy', 'local'."
            )

        if mode == "proxy":
            from systems.simula.proxy import InspectorProxy, SimulaProxy

            simula = SimulaProxy(
                redis=infra.redis,
                timeout_s=config.simula.pipeline_timeout_s,
                neo4j=infra.neo4j,
            )
            await simula.initialize()
            inspector_proxy = InspectorProxy(
                redis=infra.redis, timeout_s=config.simula.pipeline_timeout_s
            )
            await inspector_proxy.initialize()
            app.state.inspector_proxy = inspector_proxy
            logger.info("simula_mode", mode="proxy", inspector="proxy")

            if config.simula.run_worker_in_process:
                from simula_worker import run_worker

                self._tasks["simula_worker"] = asyncio.create_task(
                    run_worker(os.getenv("ECODIAOS_CONFIG_PATH")),
                    name="simula_worker_in_process",
                )
                logger.warning("simula_worker_in_process_started", note="Development mode only")
        else:
            from systems.simula.service import SimulaService

            simula = SimulaService(
                config=config.simula,
                llm=infra.llm,
                neo4j=infra.neo4j,
                memory=memory,
                codebase_root=Path(config.simula.codebase_root).resolve(),
                instance_name=config.instance_id,
                tsdb=infra.tsdb,
                redis=infra.redis,
            )
            await simula.initialize()
            app.state.inspector_proxy = None
            logger.info("simula_mode", mode="local")

        return simula

    async def _init_synapse(self, config: Any, infra: InfraClients, atune: Any) -> Any:
        from systems.synapse.service import SynapseService

        synapse = SynapseService(
            atune=atune,
            config=config.synapse,
            redis=infra.redis,
            metrics=infra.metrics,
            neuroplasticity_bus=infra.neuroplasticity_bus,
        )
        await synapse.initialize()
        return synapse

    async def _init_hot_swap(
        self, config: Any, infra: InfraClients, synapse: Any, app: Any
    ) -> None:
        if config.hot_swap.enabled:
            from clients.model_hotswap import HotSwapManager

            manager = HotSwapManager(
                config=config.hot_swap,
                llm=infra.llm,
                neo4j=infra.neo4j,
                event_bus=synapse.event_bus,
            )
            await manager.initialize()
            synapse.set_hot_swap_manager(manager)
            infra.llm.set_inference_error_callback(manager.record_inference_error)
            app.state.hot_swap_manager = manager
            logger.info("hot_swap_manager_initialized")
        else:
            app.state.hot_swap_manager = None

    async def _init_thymos(
        self,
        config: Any,
        infra: InfraClients,
        synapse: Any,
        equor: Any,
        evo: Any,
        atune: Any,
        nova: Any,
        simula: Any,
    ) -> Any:
        from systems.thymos.service import ThymosService

        thymos = ThymosService(
            config=config.thymos,
            synapse=synapse,
            neo4j=infra.neo4j,
            llm=infra.llm,
            metrics=infra.metrics,
            neuroplasticity_bus=infra.neuroplasticity_bus,
        )
        thymos.set_equor(equor)
        thymos.set_evo(evo)
        thymos.set_atune(atune)
        thymos.set_nova(nova)
        thymos.set_health_monitor(synapse._health)
        thymos.set_simula(simula)
        await thymos.initialize()
        return thymos

    async def _init_oneiros(
        self,
        config: Any,
        infra: InfraClients,
        synapse: Any,
        equor: Any,
        evo: Any,
        nova: Any,
        atune: Any,
        thymos: Any,
        memory: Any,
        simula: Any,
    ) -> Any:
        from systems.oneiros.service import OneirosService

        oneiros = OneirosService(
            config=config.oneiros,
            synapse=synapse,
            neo4j=infra.neo4j,
            llm=infra.llm,
            embed_fn=infra.embedding,
            metrics=infra.metrics,
            neuroplasticity_bus=infra.neuroplasticity_bus,
        )
        oneiros.set_equor(equor)
        oneiros.set_evo(evo)
        oneiros.set_nova(nova)
        oneiros.set_atune(atune)
        oneiros.set_thymos(thymos)
        oneiros.set_memory(memory)
        oneiros.set_simula(simula)
        await oneiros.initialize()
        return oneiros

    def _init_kairos(self, synapse: Any, logos: Any, oneiros: Any) -> Any:
        from systems.kairos.pipeline import KairosPipeline

        kairos = KairosPipeline()
        kairos.set_event_bus(synapse.event_bus)
        kairos.set_logos(logos)
        oneiros.set_kairos(kairos)
        return kairos

    async def _init_soma(
        self,
        config: Any,
        infra: InfraClients,
        atune: Any,
        synapse: Any,
        nova: Any,
        thymos: Any,
        equor: Any,
    ) -> Any:
        from systems.soma.service import SomaService

        soma = SomaService(config=config.soma, neuroplasticity_bus=infra.neuroplasticity_bus)
        soma.set_atune(atune)
        soma.set_synapse(synapse)
        soma.set_nova(nova)
        soma.set_thymos(thymos)
        soma.set_equor(equor)
        if hasattr(synapse, "_resources"):
            soma.set_token_budget(synapse._resources)
        soma.set_event_bus(synapse.event_bus)
        await soma.initialize()
        return soma

    async def _init_exteroception(self, config: Any, synapse: Any, soma: Any, app: Any) -> None:
        if not config.soma.exteroception_enabled:
            return
        from systems.soma.exteroception import (
            ExteroceptionService,
            MarketDataAdapter,
            NewsSentimentAdapter,
        )

        exteroception = ExteroceptionService(
            poll_interval_s=config.soma.exteroception_poll_interval_s,
            ema_alpha=config.soma.exteroception_ema_alpha,
            max_total_pressure=config.soma.exteroception_max_total_pressure,
            spike_threshold=config.soma.exteroception_spike_threshold,
            event_bus=synapse.event_bus,
        )
        exteroception.set_soma(soma)
        exteroception.register_adapter(
            MarketDataAdapter(
                timeout_s=config.soma.exteroception_fetch_timeout_s,
                include_fear_greed=config.soma.exteroception_include_fear_greed,
            )
        )
        if config.soma.exteroception_news_api_url:
            exteroception.register_adapter(
                NewsSentimentAdapter(
                    api_url=config.soma.exteroception_news_api_url,
                    api_key=config.soma.exteroception_news_api_key,
                    timeout_s=config.soma.exteroception_fetch_timeout_s,
                )
            )
        await exteroception.start()
        app.state.exteroception = exteroception

    async def _init_telos(self, synapse: Any, infra: InfraClients) -> Any:
        from systems.telos.service import TelosService

        telos = TelosService()
        telos.set_event_bus(synapse.event_bus)
        telos.set_redis(infra.redis)
        await telos.initialize()
        return telos

    async def _init_fovea(self, logos: Any, synapse: Any, atune: Any) -> Any:
        from systems.fovea.protocols import LogosWorldModelAdapter
        from systems.fovea.service import FoveaService

        logos_wm_adapter = LogosWorldModelAdapter(logos.world_model)
        fovea = FoveaService(world_model=logos_wm_adapter)
        fovea.set_event_bus(synapse.event_bus)
        fovea.set_workspace(atune._workspace)
        await fovea.startup()
        return fovea

    async def _init_federation(
        self, config: Any, infra: InfraClients, memory: Any, equor: Any
    ) -> Any:
        from systems.federation.service import FederationService

        federation = FederationService(
            config=config.federation,
            memory=memory,
            equor=equor,
            redis=infra.redis,
            metrics=infra.metrics,
            instance_id=config.instance_id,
        )
        await federation.initialize()
        return federation

    async def _init_nexus(
        self,
        config: Any,
        logos: Any,
        fovea: Any,
        federation: Any,
        thymos: Any,
        oneiros: Any,
        evo: Any,
        equor: Any,
        telos: Any,
        synapse: Any,
    ) -> Any:
        from systems.nexus.adapters import (
            EvoHypothesisSourceAdapter,
            ThymosNexusSinkAdapter,
        )
        from systems.nexus.adapters import (
            LogosWorldModelAdapter as NexusLogosAdapter,
        )
        from systems.nexus.service import NexusService

        adapter = NexusLogosAdapter(logos)
        nexus = NexusService()
        nexus.set_world_model(adapter)
        nexus.set_logos_adapter(adapter)
        nexus.set_fovea(fovea)
        nexus.set_federation(federation)
        nexus.set_thymos(ThymosNexusSinkAdapter(thymos))
        nexus.set_oneiros(oneiros)
        nexus.set_evo(evo)
        nexus.set_evo_hypothesis_source(EvoHypothesisSourceAdapter(evo))
        nexus.set_equor(equor)
        nexus.set_telos(telos)
        nexus.set_synapse(synapse)
        await nexus.initialize(config.instance_id)
        nexus.subscribe_to_synapse_events()
        await nexus.start_background_loops()
        logger.info("nexus_initialized", loop4="logos_nexus_federation_logos_triangulation")
        return nexus

    async def _init_wallet(self, config: Any, infra: InfraClients, axon: Any, app: Any) -> None:
        if not config.wallet.cdp_api_key_id:
            app.state.wallet = None
            infra.wallet = None
            logger.info("wallet_skipped", reason="no CDP credentials configured")
            return

        from clients.wallet import WalletClient

        wallet = WalletClient(config.wallet)
        try:
            async with asyncio.timeout(10.0):
                await wallet.connect()
            infra.wallet = wallet
            app.state.wallet = wallet
            axon.set_wallet(wallet)
            logger.info(
                "ecodiaos_ready", phase="15b_wallet", address=wallet.address, network=wallet.network
            )
        except TimeoutError:
            app.state.wallet = None
            infra.wallet = None
            logger.warning("wallet_init_timeout", timeout_sec=10.0)
        except Exception as exc:
            app.state.wallet = None
            infra.wallet = None
            logger.warning("wallet_init_failed", error=str(exc))

    async def _init_certificate_manager(
        self, config: Any, infra: InfraClients, federation: Any, synapse: Any, app: Any
    ) -> None:
        from systems.identity.manager import CertificateManager

        cm = CertificateManager()
        fed_identity = federation._identity if federation._identity else None
        if fed_identity is not None:
            await cm.initialize(
                identity=fed_identity,
                instance_id=config.instance_id,
                validity_days=config.oikos.certificate_validity_days,
                expiry_warning_days=config.oikos.certificate_expiry_warning_days,
                ca_address=config.oikos.certificate_ca_address,
                data_dir=config.federation.identity_data_dir,
                is_genesis_node=config.oikos.is_genesis_node,
            )
            cm.set_event_bus(synapse.event_bus)
            federation.set_certificate_manager(cm)
            app.state.certificate_manager = cm
            logger.info("ecodiaos_ready", phase="15d_certificate_manager")
        else:
            app.state.certificate_manager = None
            logger.info("certificate_manager_skipped", reason="federation identity not initialized")

    async def _init_oikos(self, config: Any, infra: InfraClients, synapse: Any, app: Any) -> Any:
        from systems.oikos.service import OikosService

        oikos = OikosService(
            config=config.oikos,
            wallet=infra.wallet,
            metabolism=synapse.metabolism if hasattr(synapse, "metabolism") else None,
            instance_id=config.instance_id,
            redis=infra.redis,
        )
        oikos.initialize(bus=infra.neuroplasticity_bus)
        await oikos.load_state()
        oikos.attach(synapse.event_bus)

        # Wire external platform credentials so BountyHunter can scan
        oikos.bounty_hunter.set_platforms_config(config.external_platforms)

        logger.info("ecodiaos_ready", phase="15e_oikos")
        return oikos

    async def _init_alive_ws(
        self,
        config: Any,
        infra: InfraClients,
        soma: Any,
        atune: Any,
        synapse: Any,
        telos: Any,
        thymos: Any,
        nova: Any,
        axon: Any,
        oikos: Any,
        simula: Any,
    ) -> Any:
        from systems.alive.ws_server import AliveWebSocketServer

        alive_ws = AliveWebSocketServer(
            redis=infra.redis,
            soma=soma,
            atune=atune,
            synapse=synapse,
            telos=telos,
            thymos=thymos,
            nova=nova,
            axon=axon,
            oikos=oikos,
            simula=simula,
            port=getattr(config, "alive_ws_port", 8001),
        )
        await alive_ws.start()
        return alive_ws

    async def _init_phantom_liquidity(
        self, config: Any, infra: InfraClients, atune: Any, oikos: Any, synapse: Any, app: Any
    ) -> None:
        if not config.phantom_liquidity.enabled:
            app.state.phantom_liquidity = None
            return
        from systems.phantom_liquidity.service import LiquidityPhantomService

        pl = LiquidityPhantomService(
            config=config.phantom_liquidity,
            wallet=infra.wallet,
            atune=atune,
            oikos=oikos,
            instance_id=config.instance_id,
        )
        await pl.initialize()
        pl.attach(synapse.event_bus)
        app.state.phantom_liquidity = pl
        logger.info("ecodiaos_ready", phase="15e_half_phantom_liquidity")

    async def _init_skia(self, config: Any, infra: InfraClients, synapse: Any, app: Any) -> None:
        if not config.skia.enabled:
            app.state.skia = None
            logger.info("skia_disabled")
            return
        from systems.identity.vault import IdentityVault
        from systems.skia.service import SkiaService

        vault_pw = os.environ.get("ECODIAOS_VAULT_PASSPHRASE", "")
        vault = IdentityVault(passphrase=vault_pw) if vault_pw else None
        skia = SkiaService(
            config=config.skia,
            neo4j=infra.neo4j,
            redis=infra.redis,
            vault=vault,
            instance_id=config.instance_id,
            standalone=False,
        )
        await skia.initialize()
        skia.set_event_bus(synapse.event_bus)
        await skia.start()
        synapse.register_system(skia)
        app.state.skia = skia
        logger.info("ecodiaos_ready", phase="15f_skia", mode="embedded")

    async def _init_connectors(
        self, config: Any, infra: InfraClients, synapse: Any, app: Any
    ) -> None:
        import httpx

        from systems.identity.connector import (
            ConnectorCredentials,
            ConnectorStatus,
            OAuthClientConfig,
            PlatformConnector,
        )
        from systems.identity.connectors.canva import CanvaConnector
        from systems.identity.connectors.github_app import GitHubAppConnector
        from systems.identity.connectors.instagram_graph import InstagramConnector
        from systems.identity.connectors.linkedin import LinkedInConnector
        from systems.identity.connectors.x import XConnector

        vault_pw = os.environ.get("ECODIAOS_VAULT_PASSPHRASE", "")
        connector_vault = None
        if vault_pw:
            from systems.identity.vault import IdentityVault

            connector_vault = IdentityVault(passphrase=vault_pw)

        connector_defs: list[tuple[str, type[PlatformConnector], Any]] = [
            ("linkedin", LinkedInConnector, config.connectors.linkedin),
            ("x", XConnector, config.connectors.x),
            ("github_app", GitHubAppConnector, config.connectors.github_app),
            ("instagram_graph", InstagramConnector, config.connectors.instagram_graph),
            ("canva", CanvaConnector, config.connectors.canva),
        ]

        connectors: dict[str, PlatformConnector] = {}
        for pid, cls, cfg in connector_defs:
            if not cfg.enabled or not cfg.client_id:
                continue
            if connector_vault is None:
                logger.warning("connector_skipped_no_vault", platform_id=pid)
                continue
            try:
                oauth_cfg = OAuthClientConfig(
                    client_id=cfg.client_id,
                    client_secret=cfg.client_secret,
                    authorize_url=cfg.authorize_url,
                    token_url=cfg.token_url,
                    revoke_url=cfg.revoke_url,
                    redirect_uri=cfg.redirect_uri,
                    scopes=cfg.scopes,
                )
                http = httpx.AsyncClient(timeout=30.0)
                connector = cls(client_config=oauth_cfg, vault=connector_vault, http_client=http)

                event_bus_ref = getattr(synapse, "event_bus", None)
                if event_bus_ref is not None:
                    connector.set_event_bus(event_bus_ref)
                connector.set_redis_cache(infra.redis)

                cred_id = f"{pid}:default"
                token_envelope_id = ""
                try:
                    async with infra.tsdb.pool.acquire() as conn:
                        from systems.identity import crud as id_crud

                        await id_crud.ensure_table(conn)
                        env = await id_crud.get_envelope_by_platform_and_purpose(
                            conn, pid, "oauth_token"
                        )
                        if env is not None:
                            token_envelope_id = env.id
                except Exception:
                    pass

                creds = ConnectorCredentials(
                    id=cred_id,
                    connector_id=cred_id,
                    platform_id=pid,
                    status=ConnectorStatus.ACTIVE
                    if token_envelope_id
                    else ConnectorStatus.UNCONFIGURED,
                    token_envelope_id=token_envelope_id,
                )
                connector.set_credentials(creds)
                connectors[cred_id] = connector
                logger.info("connector_wired", platform_id=pid, connector_id=cred_id)
            except Exception as exc:
                logger.warning("connector_init_failed", platform_id=pid, error=str(exc))

        app.state.connectors = connectors
        logger.info("ecodiaos_ready", phase="15g_connectors", count=len(connectors))

    def _init_github_connector(
        self, config: Any, infra: InfraClients, synapse: Any, axon: Any, oikos: Any, app: Any
    ) -> None:
        from systems.identity.connectors.github import GitHubConnector

        connectors = app.state.connectors
        github_app = connectors.get("github_app:default")
        event_bus_ref = getattr(synapse, "event_bus", None)
        github_connector = GitHubConnector(
            app_connector=github_app,  # type: ignore[arg-type]
            redis=infra.redis,
            event_bus=event_bus_ref,
        )
        app.state.github_connector = github_connector
        axon.set_github_connector(github_connector)
        oikos.set_github_connector(github_connector)
        oikos.set_bounty_submit_fn(create_bounty_submit_fn(axon))

        github_token = (
            os.environ.get("ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or ""
        )
        has_creds = bool(
            github_token
            or (config.connectors.github_app.enabled and config.connectors.github_app.client_id)
        )
        logger.info(
            "ecodiaos_ready",
            phase="15h_github_connector",
            bounty_submission_capable=has_creds,
        )
        if not has_creds:
            logger.warning("GITHUB_CREDENTIALS_MISSING — bounty PR submission disabled")

    async def _init_benchmarks(
        self,
        config: Any,
        infra: InfraClients,
        nova: Any,
        evo: Any,
        oikos: Any,
        simula: Any,
        synapse: Any,
        alive_ws: Any,
        app: Any,
    ) -> None:
        from systems.benchmarks import BenchmarkService

        benchmarks = BenchmarkService(
            config=config.benchmarks,
            tsdb=infra.tsdb,
            instance_id=config.instance_id,
        )
        benchmarks.set_nova(nova)
        benchmarks.set_evo(evo)
        benchmarks.set_oikos(oikos)
        benchmarks.set_simula(simula)
        benchmarks.set_event_bus(synapse.event_bus)
        await benchmarks.initialize()
        app.state.benchmarks = benchmarks
        alive_ws._benchmarks = benchmarks
        logger.info("ecodiaos_ready", phase="20_benchmarks")

    # ── Helpers ───────────────────────────────────────────────

    async def _check_skia_restore(self, config: Any, infra: InfraClients) -> None:
        restore_cid = os.environ.get("ECODIAOS_SKIA_RESTORE_CID", "")
        if not restore_cid:
            return
        vault_passphrase = os.environ.get("ECODIAOS_VAULT_PASSPHRASE", "")
        if not vault_passphrase:
            raise RuntimeError(
                "ECODIAOS_SKIA_RESTORE_CID is set but ECODIAOS_VAULT_PASSPHRASE is empty."
            )
        from systems.skia.snapshot import restore_from_ipfs

        await restore_from_ipfs(
            cid=restore_cid,
            neo4j=infra.neo4j,
            vault_passphrase=vault_passphrase,
            pinata_jwt=config.skia.pinata_jwt,
            pinata_api_url=config.skia.pinata_api_url,
            pinata_gateway_url=config.skia.pinata_gateway_url,
        )
        logger.info("skia_state_restored", cid=restore_cid)

    async def _birth_or_load(
        self, config: Any, infra: InfraClients, memory: Any, equor: Any, atune: Any
    ) -> None:
        instance = await memory.get_self()
        if instance is None:
            seed_path = os.environ.get("ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml")
            try:
                seed = load_seed(seed_path)
                birth_result = await memory.birth(seed, config.instance_id)
                logger.info("instance_born", **birth_result)
                await equor.initialize()
                new_instance = await memory.get_self()
                if new_instance is not None:
                    await seed_atune_cache(atune, infra.embedding, new_instance)
            except FileNotFoundError:
                logger.warning("no_seed_found", seed_path=seed_path)
        else:
            logger.info("instance_loaded", name=instance.name, instance_id=instance.instance_id)
            await seed_atune_cache(atune, infra.embedding, instance)

            # Migration: backfill personality_json
            if not instance.personality_json and instance.personality_vector:
                import json

                pkeys = [
                    "warmth",
                    "directness",
                    "verbosity",
                    "formality",
                    "curiosity_expression",
                    "humour",
                    "empathy_expression",
                    "confidence_display",
                    "metaphor_use",
                ]
                pdict = dict(zip(pkeys, instance.personality_vector[:9], strict=False))
                await infra.neo4j.execute_write(
                    "MATCH (s:Self {instance_id: $iid}) SET s.personality_json = $pj",
                    {"iid": instance.instance_id, "pj": json.dumps(pdict)},
                )
                logger.info("personality_json_backfilled", personality=pdict)

    async def _seed_goals(self, config: Any, nova: Any, atune: Any) -> None:
        if nova._goal_manager is None or len(nova._goal_manager.active_goals) > 0:
            return

        from primitives.common import DriveAlignmentVector, new_id
        from systems.nova.types import Goal, GoalSource, GoalStatus

        seed_path = os.environ.get("ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml")
        seed_goals: list[dict[str, Any]] = []
        try:
            seed = load_seed(seed_path)
            seed_goals = [
                {
                    "description": g.description,
                    "source": g.source,
                    "priority": g.priority,
                    "importance": g.importance,
                    "drive_alignment": g.drive_alignment,
                }
                for g in seed.community.initial_goals
            ]
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("seed_goals_load_failed", error=str(exc))

        if not seed_goals:
            seed_goals = [
                {
                    "description": (
                        "Learn about my community"
                        " — understand who I serve and what they need"
                    ),
                    "source": "epistemic",
                    "priority": 0.6,
                    "importance": 0.7,
                    "drive_alignment": {
                        "coherence": 0.3,
                        "care": 0.8,
                        "growth": 0.7,
                        "honesty": 0.2,
                    },
                },
                {
                    "description": (
                        "Develop self-understanding"
                        " — explore my capabilities and how my drives shape my behaviour"
                    ),
                    "source": "self_generated",
                    "priority": 0.5,
                    "importance": 0.6,
                    "drive_alignment": {
                        "coherence": 0.9,
                        "care": 0.1,
                        "growth": 0.8,
                        "honesty": 0.5,
                    },
                },
            ]

        source_map = {
            "user_request": GoalSource.USER_REQUEST,
            "self_generated": GoalSource.SELF_GENERATED,
            "governance": GoalSource.GOVERNANCE,
            "care_response": GoalSource.CARE_RESPONSE,
            "maintenance": GoalSource.MAINTENANCE,
            "epistemic": GoalSource.EPISTEMIC,
        }

        for gdata in seed_goals:
            if not isinstance(gdata, dict):
                continue
            da = gdata.get("drive_alignment") or {}
            if not isinstance(da, dict):
                da = {}
            goal = Goal(
                id=new_id(),
                description=gdata["description"],
                source=source_map.get(
                    gdata.get("source", "self_generated"), GoalSource.SELF_GENERATED
                ),
                priority=gdata.get("priority", 0.5),
                importance=gdata.get("importance", 0.5),
                drive_alignment=DriveAlignmentVector(
                    coherence=da.get("coherence", 0.0),
                    care=da.get("care", 0.0),
                    growth=da.get("growth", 0.0),
                    honesty=da.get("honesty", 0.0),
                ),
                status=GoalStatus.ACTIVE,
            )
            await nova.add_goal(goal)
            logger.info("initial_goal_seeded", description=goal.description[:60])

        atune.set_active_goals(nova.active_goal_summaries)
        logger.info("initial_goals_seeded", count=len(seed_goals))
