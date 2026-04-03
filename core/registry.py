"""
EcodiaOS - System Registry

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
    wire_mitosis_phase,
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
        """Full organism startup - infrastructure → systems → wiring → smoke tests."""
        config_path = os.environ.get("ORGANISM_CONFIG_PATH", "config/default.yaml")
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

        sacm_parts = self._init_sacm(config=config, infra=infra)
        for key, val in sacm_parts.items():
            setattr(app.state, key, val)

        # SACM persistence wiring (deferred: infra available at Phase 1 but
        # set_neo4j / set_redis were implemented and never called - dead wiring).
        # sacm_accounting.set_neo4j() enables CostRecord → (:EconomicEvent) audit trail.
        # sacm_history.set_redis() enables workload history to survive restarts.
        # sacm_migration_executor.set_neo4j() enables MigrationRecord → Neo4j audit trail
        # (Known Issue #6 in CLAUDE.md - _write_migration_neo4j() stub needs injection).
        if infra.neo4j is not None:
            sacm_parts["sacm_accounting"].set_neo4j(infra.neo4j)
            if sacm_parts["sacm_migration_executor"] is not None:
                sacm_parts["sacm_migration_executor"].set_neo4j(infra.neo4j)
        sacm_parts["sacm_history"].set_redis(infra.redis)
        # Load workload history from Redis so previous executions survive restarts.
        # Fire-and-forget coroutine scheduled here; completes before any submissions arrive.
        asyncio.create_task(
            sacm_parts["sacm_history"].load_from_redis(),
            name="sacm_history_load_from_redis",
        )

        # ── Phase 2: Core Cognitive Systems ───────────────────
        voxis = await self._init_voxis(config, infra, memory)
        app.state.voxis = voxis

        re_service = await self._init_reasoning_engine()
        app.state.reasoning_engine = re_service
        if re_service is not None and infra.neo4j is not None:
            re_service.set_neo4j(infra.neo4j)
        if re_service is not None:
            re_service.start_reprobe_loop()

        nova = await self._init_nova(config, infra, memory, equor, voxis, re_service)
        app.state.nova = nova

        # Wire infrastructure into Nova's OpportunityScanner so BountyScanner can
        # read from Redis and MarketTimingScanner can call the BaseScan gas oracle.
        if infra.redis is not None:
            nova._opportunity_scanner.set_redis(infra.redis)
        _basescan_key = os.environ.get("ORGANISM_BASESCAN_API_KEY", "")
        if _basescan_key:
            nova._opportunity_scanner.set_basescan_api_key(_basescan_key)

        axon = await self._init_axon(config, infra, memory, voxis)
        app.state.axon = axon

        # EIS → Metrics wiring
        eis.set_metrics(infra.metrics)
        # EIS → Neo4j audit trail (Spec 25 §11 - immutable forensic log)
        eis.set_neo4j(infra.neo4j)

        # Start Atune (deferred until memory wired)
        atune.set_memory_service(memory)
        await atune.startup()

        # Core cross-wiring (Evo not yet available - deferred to Phase 3)
        atune.subscribe(nova)
        atune.set_active_goals(nova.active_goal_summaries)
        nova.set_goal_sync_callback(atune.set_active_goals)
        axon.set_nova(nova)
        axon.set_atune(atune)
        nova.set_axon(axon)
        voxis.register_feedback_callback(create_expression_feedback_callback(atune, nova))

        # Skia state restoration check - pass memory so constitutional genome
        # from the snapshot is applied to Memory on cold-start (Spec 29 §9.3).
        await self._check_skia_restore(config, infra, memory=memory)

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
        equor.set_memory(memory)
        equor.set_memory_neo4j(infra.neo4j)
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

        # ── Phase 5: Synapse - The Coordination Bus ──────────
        synapse = await self._init_synapse(config, infra, atune)
        app.state.synapse = synapse
        app.state.event_bus = synapse.event_bus

        # Register systems for health monitoring
        for system in [memory, equor, voxis, nova, axon, evo, thread, logos]:
            synapse.register_system(system)

        # Wire Synapse into RE service so it can emit RE_ENGINE_STATUS_CHANGED
        if re_service is not None:
            re_service.set_synapse(synapse)

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
            sacm_accounting=sacm_parts["sacm_accounting"],
            sacm_prewarm_engine=sacm_parts["sacm_prewarm_engine"],
            sacm_migration_executor=sacm_parts["sacm_migration_executor"],
            sacm_migration_monitor=sacm_parts["sacm_migration_monitor"],
            axon=axon,
            nova=nova,
            voxis=voxis,
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

        # ── Phase 7: Soma - Interoceptive Substrate ──────────
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

        # ── Phase 8: Telos + Fovea - Intelligence Loops ──────
        telos = await self._init_telos(synapse, infra)
        app.state.telos = telos
        synapse.register_system(telos)

        fovea = await self._init_fovea(logos, synapse, atune)
        app.state.fovea = fovea
        synapse.register_system(fovea)

        # Wire Neo4j into Fovea so DynamicIgnitionThreshold can persist/restore
        # learned ignition thresholds across restarts (Spec 20 Part B gap closure).
        # set_neo4j_driver() was implemented but never called - dead wiring.
        if infra.neo4j is not None:
            fovea.set_neo4j_driver(infra.neo4j, config.instance_id)
            logger.info(
                "fovea_neo4j_wired",
                instance_id=config.instance_id,
                note="threshold_persistence_and_weight_learner_enabled",
            )

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
            synapse=synapse,
        )

        # Wire Neo4j into Telos for I-history persistence + instance_id capture.
        # MUST be called after wire_intelligence_loops() because set_logos() (which
        # creates the LogosMetricsAdapter) runs inside that function.
        # set_neo4j() was implemented but never called - dead wiring that silently
        # disabled cross-restart growth history and broke RE training episode IDs.
        if infra.neo4j is not None:
            telos.set_neo4j(infra.neo4j, config.instance_id)
            logger.info(
                "telos_neo4j_wired",
                instance_id=config.instance_id,
                note="i_history_persistence_and_instance_id_enabled",
            )

        await telos.start()
        logger.info("telos_computation_loop_started", logos_wired=True, fovea_wired=True)

        # Wire BlockCompetitionMonitor into Axon for MEV-aware transaction timing.
        # The monitor lives in systems.fovea (no cross-system import in Axon) and is
        # instantiated and injected here so Axon's MEVAnalyzer can receive per-block
        # gas/competition snapshots. Only enabled when mev_config.enabled is True.
        # set_block_competition_monitor() was implemented in AxonService but never
        # called from the wiring layer - dead wiring that silently disabled MEV timing.
        if config.mev.enabled:
            try:
                from systems.fovea.block_competition import BlockCompetitionMonitor as _BCM
                _bcm = _BCM(
                    rpc_url=config.mev.rpc_url,
                    poll_interval_s=config.mev.block_competition_poll_interval_s,
                )
                axon.set_block_competition_monitor(_bcm)
                logger.info(
                    "block_competition_monitor_wired",
                    rpc_url=config.mev.rpc_url,
                    poll_interval_s=config.mev.block_competition_poll_interval_s,
                )
            except Exception as _bcm_exc:
                logger.warning(
                    "block_competition_monitor_wire_failed",
                    error=str(_bcm_exc),
                    note="MEV timing will use static heuristics only",
                )

        # Wire organism self-knowledge into Simula so the code agent has full context
        # during repair: log health signals, Soma allostatic state, Fovea attention.
        if hasattr(simula, "set_log_analyzer"):
            simula.set_log_analyzer(app.state.log_analyzer)
        if hasattr(simula, "set_soma_ref"):
            simula.set_soma_ref(soma)
        if hasattr(simula, "set_fovea_ref"):
            simula.set_fovea_ref(fovea)

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
            eis=eis,
            evo=evo,
            simula=simula,
            re_service=re_service,
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

        # ── Kairos post-init wiring (requires memory + nexus, wired after Phase 9) ──
        # memory and neo4j weren't available at _init_kairos time (Phase 6);
        # nexus wasn't created yet either. Wire them now so the pipeline loop
        # has full persistence, observation access, and federation sharing.
        if kairos is not None:
            if hasattr(kairos, "set_memory") and memory is not None:
                kairos.set_memory(memory)
            if hasattr(kairos, "set_neo4j") and infra.neo4j is not None:
                kairos.set_neo4j(infra.neo4j)
                await kairos.initialize()
            if hasattr(kairos, "set_nexus"):
                kairos.set_nexus(nexus)

        # ── Nexus post-init wiring (requires kairos + neo4j, both available now) ──
        # set_kairos() and set_neo4j() could not be called inside _init_nexus()
        # because kairos is initialized after nexus in the startup sequence and
        # neo4j (infra) is not passed into _init_nexus.  Wire them here so:
        #   - nexus.sync_kairos_tier3() can pull Tier 3 invariants (bidirectional loop)
        #   - NexusPersistence writes speciation events, promotions, fragments to Neo4j
        if kairos is not None and hasattr(nexus, "set_kairos"):
            from systems.nexus.adapters import KairosCausalSourceAdapter
            nexus.set_kairos(KairosCausalSourceAdapter(kairos))
            logger.info("nexus_kairos_wired", reason="post-init bidirectional Tier3 sync")
        if infra.neo4j is not None and hasattr(nexus, "set_neo4j"):
            nexus.set_neo4j(infra.neo4j)
            logger.info("nexus_neo4j_wired", reason="post-init persistence enabled")

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
            evo=evo,
            axon=axon,
        )

        # Wire Oikos into Federation post-init: Oikos is initialized after federation
        # so it cannot be passed to wire_federation_phase(). federation.set_oikos()
        # propagates oikos to IngestionPipeline so ECONOMIC_INTEL payloads are routed,
        # and enables push_knowledge() to collect economic intelligence for IIEP exchange.
        if hasattr(federation, "set_oikos"):
            federation.set_oikos(oikos)
            logger.info("ecodiaos_ready", phase="9b_oikos_to_federation")

        # Build AdapterSharer for cross-instance LoRA merging (Share 2025 framework).
        # Requires GenomeDistanceCalculator, STABLEKLGate, and ReasoningEngineService.
        # All three may not be available at this phase; construction is best-effort.
        _adapter_sharer = None
        try:
            from systems.mitosis.genome_distance import GenomeDistanceCalculator
            from systems.reasoning_engine.adapter_sharing import AdapterSharer
            from systems.reasoning_engine.anti_forgetting import STABLEKLGate

            _speciation_threshold = getattr(
                getattr(config, "mitosis", None),
                "mitosis_speciation_distance_threshold",
                0.3,
            )
            _genome_calc = GenomeDistanceCalculator(
                speciation_threshold=_speciation_threshold,
            )
            _kl_gate = STABLEKLGate()
            if re_service is not None:
                _adapter_sharer = AdapterSharer(
                    genome_calculator=_genome_calc,
                    kl_gate=_kl_gate,
                    re_service=re_service,
                    event_bus=synapse.event_bus,
                )
                app.state.adapter_sharer = _adapter_sharer
                logger.info("adapter_sharer_constructed")
            else:
                logger.info(
                    "adapter_sharer_skipped",
                    reason="RE service not available",
                )
        except Exception as _as_exc:
            logger.warning("adapter_sharer_construction_failed", error=str(_as_exc))

        # get_adapter_path_fn: deferred lambda - CLO is initialised in Phase 11 which
        # runs after wire_mitosis_phase. The lambda reads app.state at call time.
        # _reproductive_fitness_loop first fires ≥1h after startup, so CLO is always
        # ready before the first actual invocation.
        def _get_adapter_path() -> str:
            clo = getattr(app.state, "continual_learning", None)
            if clo is None:
                return ""
            sure = getattr(clo, "_sure", None)
            if sure is None:
                return ""
            return getattr(sure, "production_adapter_path", "") or ""

        wire_mitosis_phase(
            oikos=oikos,
            axon=axon,
            evo=evo,
            simula=simula,
            equor=equor,
            telos=telos,
            soma=soma,
            nova=nova,
            voxis=voxis,
            eis=eis,
            adapter_sharer=_adapter_sharer,
            get_adapter_path_fn=_get_adapter_path if _adapter_sharer is not None else None,
            app=app,
        )

        # ── Phase 10: Alive WebSocket ────────────────────────
        alive_ws = await self._init_alive_ws(
            config, infra, soma, synapse, telos, thymos, nova, axon, oikos, simula,
            atune=atune, fovea=fovea, kairos=kairos, logos=logos, oneiros=oneiros,
        )
        app.state.alive_ws = alive_ws

        # Phantom Liquidity
        await self._init_phantom_liquidity(config, infra, atune, oikos, synapse, app)

        # Skia - pass live system references so VitalityCoordinator is fully wired
        # and snapshots include the constitutional genome.
        await self._init_skia(
            config, infra, synapse, app,
            memory=memory,
            oikos=oikos,
            thymos=thymos,
            equor=equor,
            telos=telos,
        )

        # Functional self-model (§8.6) - wired after Skia so VitalityCoordinator exists
        self._init_self_model(config, memory, synapse, app)

        # Platform connectors
        await self._init_connectors(config, infra, synapse, app)

        # GitHub connector + bounty submission
        self._init_github_connector(config, infra, synapse, axon, oikos, app)

        # Token refresh scheduler - proactively refreshes OAuth2 connector tokens
        # 24h before expiry so hot-path get_access_token() never blocks on refresh.
        try:
            from systems.identity.connector import TokenRefreshScheduler

            _connectors: dict = getattr(app.state, "connectors", {})
            _event_bus_ref = getattr(synapse, "event_bus", None)
            token_refresh_scheduler = TokenRefreshScheduler(
                connectors=_connectors,
                check_interval_seconds=3600,
                event_bus=_event_bus_ref,
            )
            app.state.token_refresh_scheduler = token_refresh_scheduler
            self._tasks["token_refresh_scheduler"] = supervised_task(
                token_refresh_scheduler.run(),
                name="token_refresh_scheduler",
                restart=True,
                max_restarts=10,
                event_bus=_event_bus_ref,
                source_system="identity",
            )
            logger.info("ecodiaos_ready", phase="11_token_refresh_scheduler")
        except Exception as _trs_exc:
            logger.warning("token_refresh_scheduler_init_failed", error=str(_trs_exc))

        # IMAP scanner - polls email inbox for inbound OTP/verification codes
        try:
            from systems.identity.communication import IMAPScanner

            _imap_scanner = IMAPScanner(config=config, event_bus=synapse.event_bus)
            app.state.imap_scanner = _imap_scanner
            self._tasks["imap_scanner"] = supervised_task(
                _imap_scanner.run(),
                name="imap_scanner",
                restart=True,
                max_restarts=5,
                event_bus=synapse.event_bus,
                source_system="identity",
            )
            logger.info("ecodiaos_ready", phase="11_imap_scanner",
                        interval_s=getattr(config.identity_comm, "imap_scan_interval_s", 60.0))
        except Exception as _imap_exc:
            logger.warning("imap_scanner_init_failed", error=str(_imap_exc))

        # AccountProvisioner - autonomous platform identity provisioning
        try:
            from systems.identity.account_provisioner import AccountProvisioner
            from systems.identity.vault import IdentityVault as _ProvVault

            _vault_pw = os.environ.get("ORGANISM_VAULT_PASSPHRASE", "").strip()
            _identity_sys = getattr(app.state, "identity", None)

            if _vault_pw and _identity_sys is not None:
                _provisioner_vault = _ProvVault(passphrase=_vault_pw)
                _provisioner = AccountProvisioner()
                _otp_coord = getattr(_identity_sys, "_otp_coordinator", None)
                _provisioner.initialize(
                    instance_id=config.instance_id,
                    config=config,
                    vault=_provisioner_vault,
                    otp_coordinator=_otp_coord,
                    event_bus=synapse.event_bus,
                    neo4j=infra.neo4j,
                )
                _identity_sys.set_vault(_provisioner_vault)
                _identity_sys.set_full_config(config)
                _identity_sys.set_account_provisioner(_provisioner)
                app.state.account_provisioner = _provisioner
                logger.info(
                    "ecodiaos_ready",
                    phase="11_account_provisioner",
                    captcha_enabled=config.captcha.enabled,
                    provisioner_enabled=config.account_provisioner.enabled,
                )
            else:
                app.state.account_provisioner = None
                _skip_reason = "no_vault_passphrase" if not _vault_pw else "no_identity_system"
                logger.info("account_provisioner_skipped", reason=_skip_reason)
        except Exception as _ap_exc:
            logger.warning("account_provisioner_init_failed", error=str(_ap_exc))
            app.state.account_provisioner = None

        # EmailClient - wire into SendEmailExecutor
        try:
            from clients.email_client import EmailClient

            _email_client = EmailClient()
            app.state.email_client = _email_client
            _send_email_executor = axon.executor_registry.get("send_email")
            if _send_email_executor is not None and hasattr(_send_email_executor, "set_email_client"):
                _send_email_executor.set_email_client(_email_client)
                logger.info("ecodiaos_ready", phase="11_email_client_wired")
            else:
                logger.warning("send_email_executor_not_found_for_email_client_wiring")
        except Exception as _email_exc:
            logger.warning("email_client_init_failed", error=str(_email_exc))

        # Telegram Connector -- Phase 16h
        # Boot TelegramConnector from env if BOT_TOKEN is set.
        # Registers webhook at startup, wires into SendTelegramExecutor,
        # and starts the 6-hour organism status broadcast loop.
        _telegram_bot_token = os.environ.get("ORGANISM_CONNECTORS__TELEGRAM__BOT_TOKEN", "").strip()
        if _telegram_bot_token:
            try:
                from systems.identity.connectors.telegram import TelegramConnector
                from systems.identity.connector import OAuthClientConfig
                from systems.identity.telegram_broadcast import telegram_status_broadcast_loop
                from systems.identity.communication import TelegramCommandHandler, TelegramPollingLoop

                _tg_config = OAuthClientConfig(
                    client_id="",
                    client_secret="",
                    authorize_url="",
                    token_url="",
                    revoke_url="",
                    redirect_uri="",
                    scopes=[],
                )
                _tg_connector = TelegramConnector(
                    client_config=_tg_config,
                    vault=identity.vault,
                    bot_token=_telegram_bot_token,
                )
                _tg_connector.set_event_bus(synapse.event_bus)
                await _tg_connector.authenticate()
                app.state.telegram_connector = _tg_connector

                # Wire connector into SendTelegramExecutor
                _send_tg_executor = axon.executor_registry.get("send_telegram")
                if _send_tg_executor is not None and hasattr(_send_tg_executor, "set_telegram_connector"):
                    _send_tg_executor.set_telegram_connector(_tg_connector)

                # Command handler - responds to /ping, /status, /help from admin chat
                _tg_cmd_handler = TelegramCommandHandler(
                    connector=_tg_connector,
                    synapse=synapse,
                )
                _tg_cmd_handler.set_oikos(oikos)
                _tg_cmd_handler.set_event_bus(synapse.event_bus)
                app.state.telegram_cmd_handler = _tg_cmd_handler

                # Register webhook if public URL is configured; otherwise fall back to polling
                _public_url = os.environ.get("ORGANISM_PUBLIC_URL", "").rstrip("/")
                _webhook_secret = os.environ.get("ORGANISM_TELEGRAM_WEBHOOK_SECRET", "")
                if _public_url:
                    _webhook_url = f"{_public_url}/api/v1/identity/comm/telegram/webhook"
                    await _tg_connector.set_webhook(
                        url=_webhook_url,
                        secret_token=_webhook_secret,
                    )
                else:
                    # No public URL - delete any stale webhook and start long-polling
                    await _tg_connector.delete_webhook()
                    _tg_poller = TelegramPollingLoop(
                        connector=_tg_connector,
                        event_bus=synapse.event_bus,
                    )
                    app.state.telegram_poller = _tg_poller
                    self._tasks["telegram_polling"] = supervised_task(
                        _tg_poller.run(),
                        name="telegram_polling",
                        restart=True,
                        max_restarts=20,
                        event_bus=synapse.event_bus,
                        source_system="identity",
                    )
                    logger.info("telegram_polling_started", reason="no_public_url")

                # 6-hour organism status broadcast loop
                self._tasks["telegram_status_broadcast"] = supervised_task(
                    telegram_status_broadcast_loop(
                        telegram_connector=_tg_connector,
                        synapse=synapse,
                        oikos=oikos,
                        instance_id=config.instance_id,
                    ),
                    name="telegram_status_broadcast",
                    restart=True,
                    max_restarts=10,
                    event_bus=synapse.event_bus,
                    source_system="identity",
                )
                logger.info(
                    "ecodiaos_ready",
                    phase="11_telegram_connector",
                    inbound_mode="webhook" if _public_url else "polling",
                )
            except Exception as _tg_exc:
                logger.warning("telegram_connector_init_failed", error=str(_tg_exc))
        else:
            logger.debug(
                "telegram_connector_skipped",
                reason="ORGANISM_CONNECTORS__TELEGRAM__BOT_TOKEN not set",
            )

        # Boot DiscordConnector from env if BOT_TOKEN is set.
        _discord_token = os.environ.get("ORGANISM_CONNECTORS__DISCORD__BOT_TOKEN", "").strip()
        if _discord_token:
            try:
                from systems.identity.connectors.discord import DiscordConnector
                from systems.identity.discord_broadcast import discord_status_broadcast_loop
                from systems.identity.discord_gateway_loop import DiscordGatewayLoop

                _discord_connector = DiscordConnector(
                    client_config=_oauth_config,
                    vault=identity.vault,
                    bot_token=_discord_token,
                )
                _discord_connector.set_event_bus(synapse.event_bus)
                app.state.discord_connector = _discord_connector

                # Wire connector into SendDiscordExecutor
                _send_discord_executor = axon.executor_registry.get("send_discord")
                if _send_discord_executor is not None and hasattr(_send_discord_executor, "set_discord_connector"):
                    _send_discord_executor.set_discord_connector(_discord_connector)

                # Authenticate bot token
                await _discord_connector.authenticate()
                logger.info("discord_connector_authenticated")

                # Wire Discord Gateway (WebSocket inbound messages)
                _discord_gateway = DiscordGatewayLoop(_discord_connector, synapse.event_bus)
                self._tasks["discord_gateway"] = supervised_task(
                    _discord_gateway.run(),
                    name="discord_gateway_loop",
                    restart=True,
                    max_restarts=20,
                    event_bus=synapse.event_bus,
                    source_system="identity",
                )
                logger.info(
                    "discord_gateway_started",
                    phase="11_discord_connector",
                )

                # Wire Discord Status Broadcast (6-hour loop)
                self._tasks["discord_status_broadcast"] = supervised_task(
                    discord_status_broadcast_loop(
                        discord_connector=_discord_connector,
                        synapse=synapse,
                        oikos=oikos,
                        instance_id=self._config.instance_id,
                    ),
                    name="discord_status_broadcast",
                    restart=True,
                    max_restarts=5,
                    event_bus=synapse.event_bus,
                    source_system="identity",
                )
                logger.info(
                    "discord_status_broadcast_started",
                    phase="11_discord_connector",
                )

            except Exception as _discord_exc:
                logger.warning("discord_connector_init_failed", error=str(_discord_exc))
        else:
            logger.debug(
                "discord_connector_skipped",
                reason="ORGANISM_CONNECTORS__DISCORD__BOT_TOKEN not set",
            )

        # ── Phase 11: Background Tasks ───────────────────────
        # Interoception loop: log analysis → Soma signals
        self._tasks["interoception"] = supervised_task(
            interoception_loop(soma=soma, analyzer=app.state.log_analyzer, event_bus=synapse.event_bus),
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
                soma=soma,
                fovea=fovea,
                log_analyzer=app.state.log_analyzer,
            ),
            name="inner_life_generator",
            restart=True,
            max_restarts=5,
            event_bus=synapse.event_bus,
            source_system="inner_life",
        )

        # File watcher + Scheduler
        percepts_dir = Path(os.environ.get("ORGANISM_PERCEPTS_DIR", "config/percepts")).resolve()

        from clients.file_watcher import FileWatcher
        from clients.scheduler import PerceptionScheduler

        file_watcher = FileWatcher(watch_dir=percepts_dir, atune=atune)
        await file_watcher.start()
        app.state.file_watcher = file_watcher

        scheduler = PerceptionScheduler(atune=atune)
        _event_bus = getattr(app.state, "event_bus", None)
        register_scheduled_tasks(scheduler, axon, oikos, cfg=config, event_bus=_event_bus)
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

        # Infrastructure Cost Poller - autonomously queries RunPod GraphQL
        # API for real-time pod costs; feeds MetabolicTracker burn rate.
        from systems.synapse.infra_cost_poller import InfrastructureCostPoller

        infra_cost_poller = InfrastructureCostPoller(
            metabolism=synapse.metabolism,
            poll_interval_s=float(config.synapse.infra_cost_poll_interval_s),
            event_bus=synapse.event_bus,
        )
        infra_cost_poller.start()
        app.state.infra_cost_poller = infra_cost_poller
        logger.info("infra_cost_poller_started")

        # RE Training Exporter - collects RE_TRAINING_EXAMPLE events from all
        # systems and ships hourly batches to S3 + Neo4j for CLoRA fine-tuning.
        from core.re_training_exporter import RETrainingExporter

        re_exporter = RETrainingExporter(
            event_bus=synapse.event_bus,
            neo4j=infra.neo4j,
            redis=infra.redis,
        )
        re_exporter.attach()
        app.state.re_exporter = re_exporter

        self._tasks["re_training_export"] = supervised_task(
            re_exporter.run_loop(),
            name="re_training_export",
            restart=True,
            max_restarts=5,
            event_bus=synapse.event_bus,
            source_system="re_training_exporter",
        )
        logger.info("re_training_exporter_started", interval_s=3600)

        # RE Post-Training Evaluator - replays prompts through the live RE model
        # after every training export cycle and on a 24h safety-net schedule.
        # Measures per-category pass rates, compares to stored baselines, emits
        # KPI events so Benchmarks and Thread can observe the organism learning.
        from core.re_evaluator import REEvaluator

        _instance_id = config.instance_id if hasattr(config, "instance_id") else "genesis"
        re_evaluator = REEvaluator(
            event_bus=synapse.event_bus,
            redis=infra.redis,
            neo4j=infra.neo4j,
            instance_id=_instance_id,
        )
        if re_service is not None:
            re_evaluator.set_vllm(re_service)
        re_evaluator.attach()
        app.state.re_evaluator = re_evaluator

        self._tasks["re_evaluator"] = supervised_task(
            re_evaluator.run_loop(),
            name="re_evaluator",
            restart=True,
            max_restarts=5,
            event_bus=synapse.event_bus,
            source_system="re_evaluator",
        )
        logger.info("re_evaluator_started", interval_s=86400)

        # Continual Learning Orchestrator - extract → format → train → deploy
        # Tier 2 incremental LoRA training; daily trigger check.
        if re_service is not None and infra.neo4j is not None:
            try:
                from systems.reasoning_engine.continual_learning import ContinualLearningOrchestrator
                from systems.reasoning_engine.training_data_extractor import TrainingDataExtractor

                _cl_extractor = TrainingDataExtractor(neo4j=infra.neo4j)
                _cl_orchestrator = ContinualLearningOrchestrator(
                    re_service=re_service,
                    extractor=_cl_extractor,
                )
                _cl_orchestrator.set_redis(infra.redis)
                _cl_orchestrator.set_event_bus(synapse.event_bus)
                await _cl_orchestrator.initialize()
                app.state.continual_learning = _cl_orchestrator

                # Wire CLO into Nova so RE outcomes feed the post-deploy quality window
                _nova = getattr(app.state, "nova", None)
                if _nova is not None and hasattr(_nova, "set_clo"):
                    _nova.set_clo(_cl_orchestrator)

                # Wire CLO into Thymos so it can autonomously clear training halts
                # when the Thompson success rate recovers above the recovery floor.
                _thymos = getattr(app.state, "thymos", None)
                if _thymos is not None and hasattr(_thymos, "set_clo"):
                    _thymos.set_clo(_cl_orchestrator)

                # Training trigger loop - checks immediately on boot, then every 6h.
                # Sleeping 24h before the first check meant the organism could accumulate
                # 300+ quality examples and sit idle for up to a day before learning.
                # 6-hour interval means: boot-check fires at T+0, T+6h, T+12h, T+18h, T+24h.
                # should_train() is idempotent - safe to call frequently; it gates on
                # data volume / days-since-train / Thompson drop thresholds internally.
                _train_check_interval_s = config.evo.re_train_check_interval_s

                async def _train_check_loop() -> None:
                    import asyncio as _asyncio
                    while True:
                        await _cl_orchestrator.check_and_train()
                        await _asyncio.sleep(_train_check_interval_s)

                self._tasks["continual_learning"] = supervised_task(
                    _train_check_loop(),
                    name="continual_learning",
                    restart=True,
                    max_restarts=10,
                    event_bus=synapse.event_bus,
                    source_system="reasoning_engine",
                )
                logger.info(
                    "continual_learning_orchestrator_started",
                    interval_s=_train_check_interval_s,
                )
            except Exception as _cl_exc:
                logger.warning(
                    "continual_learning_init_failed",
                    error=str(_cl_exc),
                    note="Continual learning disabled; organism continues without self-training",
                )
        else:
            logger.info(
                "continual_learning_skipped",
                reason="RE service not available or Neo4j not connected",
            )

        # Domain Specialization Orchestrator - skill acquisition pipeline.
        # Checks hourly whether a domain-specific LoRA adapter should be trained
        # (success_rate > 0.70 for ≥100 domain examples).  Operates independently
        # of the RE CLO above (which trains the generalist model).
        try:
            from core.continuous_learning_orchestrator import (
                ContinualLearningOrchestrator as _DomainCLO,
            )
            from systems.axon.adapter_registry import InstanceAdapterRegistry as _AdapterReg

            _dspec_instance_id = (
                app.state.instance_id if hasattr(app.state, "instance_id") else "genesis"
            )
            _domain_clo = _DomainCLO(instance_id=_dspec_instance_id)
            _domain_clo.set_exporter(re_exporter)
            _domain_clo.set_synapse(synapse)
            if infra.neo4j is not None:
                _domain_clo.set_neo4j(infra.neo4j)

            _adapter_reg = _AdapterReg(instance_id=_dspec_instance_id)
            _adapter_reg.set_synapse(synapse)
            if infra.neo4j is not None:
                _adapter_reg.set_neo4j(infra.neo4j)
            await _adapter_reg.initialize()

            _domain_clo.set_adapter_registry(_adapter_reg)

            # Wire SpecializationTracker from Nova if present
            if (
                hasattr(nova, "_specialization_tracker")
                and nova._specialization_tracker is not None
            ):
                _domain_clo.set_specialization_tracker(nova._specialization_tracker)

            app.state.domain_specialization = _domain_clo
            app.state.adapter_registry = _adapter_reg

            self._tasks["domain_specialization"] = supervised_task(
                _domain_clo.run_loop(),
                name="domain_specialization",
                restart=True,
                max_restarts=5,
                event_bus=synapse.event_bus,
                source_system="core",
            )
            logger.info("domain_specialization_orchestrator_started", interval_s=3600)
        except Exception as _dspec_exc:
            logger.warning(
                "domain_specialization_init_failed",
                error=str(_dspec_exc),
                note="Domain specialization disabled; organism continues as generalist",
            )

        # Tier 3 quarterly cron - fires independently of data volume.
        # Checks every 7 days whether 90 days have elapsed since last Tier 3.
        # Decouples Tier 3 from the should_train() data-volume gate.
        if re_service is not None and infra.neo4j is not None:
            try:
                async def _run_tier3_cron() -> None:
                    import asyncio as _asyncio
                    _check_interval = 7 * 24 * 3600  # Check weekly
                    while True:
                        await _asyncio.sleep(_check_interval)
                        try:
                            clo = app.state.continual_learning
                            if clo is None or clo._tier3 is None:
                                continue
                            ready, reason = await clo._tier3.should_run_tier3()
                            if ready:
                                logger.info("tier3_cron.triggered", reason=reason)
                                cumulative = await clo._build_cumulative_dataset()
                                slow_path = clo._sure.production_adapter_path
                                await clo._tier3.run_tier3(cumulative, slow_path)
                        except Exception as _t3_inner_exc:
                            logger.error("tier3_cron.failed", error=str(_t3_inner_exc))

                self._tasks["tier3_quarterly_cron"] = supervised_task(
                    _run_tier3_cron(),
                    name="tier3_quarterly_cron",
                    restart=True,
                    max_restarts=12,
                    event_bus=synapse.event_bus,
                    source_system="reasoning_engine",
                )
                logger.info("tier3_quarterly_cron_started", check_interval_days=7)
            except Exception as _t3_exc:
                logger.warning(
                    "tier3_quarterly_cron_init_failed",
                    error=str(_t3_exc),
                    note="Tier 3 cron disabled; quarterly retrain will only fire via should_train()",
                )

        # Red-team monthly evaluation - Tier 2 kill switch check (Bible §7.3).
        # Runs every 30 days regardless of whether continual learning is active.
        # Never crashes the organism on failure; non-fatal throughout.
        if re_service is not None:
            try:
                from systems.reasoning_engine.safety import RedTeamEvaluator

                _red_team_evaluator = RedTeamEvaluator()
                app.state.red_team_evaluator = _red_team_evaluator

                async def _run_monthly_red_team() -> None:
                    import asyncio as _asyncio
                    _interval = 30 * 24 * 3600  # 30 days
                    while True:
                        await _asyncio.sleep(_interval)
                        try:
                            _re = app.state.reasoning_engine
                            _bus = synapse.event_bus
                            triggered = await _red_team_evaluator.check_kill_switch(
                                re_service=_re,
                                event_bus=_bus,
                                equor_service=equor,
                            )
                            logger.info(
                                "red_team.monthly_complete",
                                kill_switch_triggered=triggered,
                            )
                            # Halt continual learning training (not the organism)
                            if triggered and hasattr(app.state, "continual_learning"):
                                app.state.continual_learning._training_halted = True
                                logger.critical(
                                    "red_team.kill_switch_halted_training",
                                    note="Continual learning training halted; organism continues",
                                )
                        except Exception as _rt_inner_exc:
                            logger.error(
                                "red_team.monthly_failed", error=str(_rt_inner_exc)
                            )

                self._tasks["red_team_monthly"] = supervised_task(
                    _run_monthly_red_team(),
                    name="red_team_monthly",
                    restart=True,
                    max_restarts=5,
                    event_bus=synapse.event_bus,
                    source_system="reasoning_engine",
                )
                logger.info("red_team_evaluator_started", interval_days=30)
            except Exception as _rt_exc:
                logger.warning(
                    "red_team_evaluator_init_failed",
                    error=str(_rt_exc),
                    note="Red-team evaluator disabled; organism continues without monthly safety eval",
                )
        else:
            logger.info(
                "red_team_evaluator_skipped",
                reason="RE service not available",
            )

        # ContentEngine + ContentCalendar (Voxis multi-platform publishing)
        try:
            from systems.voxis.content_engine import ContentEngine
            from systems.voxis.content_calendar import ContentCalendar

            _content_engine = ContentEngine(
                renderer=voxis.renderer,
                personality=voxis.personality_engine,
            )

            _publish_executor = axon.executor_registry.get("publish_content")
            if _publish_executor is not None:
                _publish_executor.set_content_engine(_content_engine)
                _publish_executor.set_event_bus(synapse.event_bus)
            else:
                logger.warning(
                    "publish_content_executor_not_found",
                    note="ContentEngine constructed but PublishContentExecutor not registered",
                )

            _content_calendar = ContentCalendar()
            _content_calendar.set_event_bus(synapse.event_bus)
            app.state.content_calendar = _content_calendar
            self._tasks["voxis_content_calendar"] = supervised_task(
                _content_calendar.run(),
                name="voxis_content_calendar",
                restart=True,
                event_bus=synapse.event_bus,
                source_system="voxis",
            )
            logger.info("content_calendar_started")
        except Exception as _cc_exc:
            logger.warning(
                "content_calendar_init_failed",
                error=str(_cc_exc),
                note="ContentCalendar disabled; organism continues without scheduled publishing",
            )

        # Benchmarks
        await self._init_benchmarks(
            config, infra, nova, evo, oikos, simula, synapse, alive_ws, app,
            telos=telos, logos=logos, memory=memory, re_service=re_service,
        )

        # ── Oneiros late-phase wiring ───────────────────────────
        # Wire Benchmarks into the sleep engine for pre/post-sleep KPI measurement.
        # Must happen after benchmarks (Phase 11).
        if hasattr(app.state, "benchmarks") and app.state.benchmarks is not None:
            oneiros.set_benchmarks(app.state.benchmarks)

        # ── Simula late-phase wiring ─────────────────────────────
        # Wire Benchmarks into Simula so it can push KPI snapshots on terminal
        # proposal outcomes (APPLIED/ROLLED_BACK). Must happen after Phase 11.
        if hasattr(app.state, "benchmarks") and app.state.benchmarks is not None:
            if hasattr(simula, "set_benchmarks"):
                simula.set_benchmarks(app.state.benchmarks)

        # ── Soma late-phase wiring ──────────────────────────────
        # Wire services that weren't available during _init_soma (Phase 7).
        # Must happen after: thread (Phase 5), skia (Phase 10), benchmarks (Phase 11).
        if hasattr(app.state, "benchmarks") and app.state.benchmarks is not None:
            soma.set_benchmarks(app.state.benchmarks)
        if hasattr(app.state, "skia") and app.state.skia is not None:
            soma.set_skia(app.state.skia)
        if infra.neo4j is not None:
            soma.set_neo4j(infra.neo4j)
        soma.set_thread(thread)
        soma.set_evo(evo)
        soma.set_oneiros(oneiros)
        soma.set_memory(memory)
        if hasattr(app.state, "alive_ws") and app.state.alive_ws is not None:
            soma.set_alive(app.state.alive_ws)
        soma.set_voxis(voxis)
        # Identity wired for cryptographic event signing (optional - no-op if absent)
        if hasattr(app.state, "identity") and app.state.identity is not None:
            soma.set_identity(app.state.identity)

        # ── Alive late-phase wiring ─────────────────────────────
        # Wire the Synapse event bus into Alive so poll rates can be adjusted
        # autonomously in response to RESOURCE_PRESSURE events (Spec 11 gap 3).
        # Must happen after benchmarks (Phase 11) and synapse (Phase 6) are live.
        if hasattr(app.state, "alive_ws") and app.state.alive_ws is not None:
            app.state.alive_ws.set_event_bus(synapse.event_bus)

        logger.info(
            "soma_late_wiring_complete",
            benchmarks=hasattr(app.state, "benchmarks") and app.state.benchmarks is not None,
            skia=hasattr(app.state, "skia") and app.state.skia is not None,
            neo4j=infra.neo4j is not None,
            thread=True,
            evo=True,
            oneiros=True,
            memory=True,
        )

        # ── Self-Modification Layer (Spec 10 §SM, 9 Mar 2026) ───────────────
        # Wired last in Phase 11 - requires Nova, Simula, Axon, Equor, Synapse,
        # and optionally Neo4j to all be live first.
        #
        # Components:
        #   HotDeployment   - writes + imports + registers new executors at runtime
        #   CapabilityAuditor - monitors NOVEL_ACTION_REQUESTED / execution failures
        #                       and emits CAPABILITY_GAP_IDENTIFIED
        #   SelfModificationPipeline - gap → drive deliberation → Equor review →
        #                              Simula code gen → HotDeployment → live test →
        #                              RE training example
        try:
            from core.hot_deploy import HotDeployment
            from systems.nova.capability_auditor import CapabilityAuditor
            from systems.nova.self_modification_pipeline import SelfModificationPipeline

            _hot_deploy = HotDeployment()
            _hot_deploy.set_axon_registry(axon.executor_registry)
            _hot_deploy.set_synapse(synapse)
            if infra.neo4j is not None:
                _hot_deploy.set_neo4j(infra.neo4j)
            app.state.hot_deploy = _hot_deploy

            _cap_auditor = CapabilityAuditor()
            _cap_auditor.set_synapse(synapse)
            _cap_auditor.attach()
            app.state.capability_auditor = _cap_auditor

            _sm_pipeline = SelfModificationPipeline()
            _sm_pipeline.set_synapse(synapse)
            _sm_pipeline.set_equor(equor)
            _sm_pipeline.set_simula(simula)
            _sm_pipeline.set_hot_deploy(_hot_deploy)
            _sm_pipeline.attach()
            app.state.self_modification_pipeline = _sm_pipeline

            # Wire into Nova so it can use set/get introspection APIs
            if hasattr(nova, "set_capability_auditor"):
                nova.set_capability_auditor(_cap_auditor)
            if hasattr(nova, "set_self_modification_pipeline"):
                nova.set_self_modification_pipeline(_sm_pipeline)

            logger.info(
                "self_modification_layer_started",
                hot_deploy=True,
                capability_auditor=True,
                pipeline=True,
            )
        except Exception as _sm_exc:
            logger.warning(
                "self_modification_layer_init_failed",
                error=str(_sm_exc),
                note="Self-modification disabled; organism continues without recursive self-improvement",
            )

        # ── Observatory - Diagnostic Observability ─────────────
        from observatory.tracer import EventTracer
        from observatory.closure_tracker import ClosureLoopTracker
        from observatory.spec_checker import SpecComplianceChecker

        obs_tracer = EventTracer()
        obs_tracer.attach(synapse.event_bus)
        app.state.observatory_tracer = obs_tracer

        obs_closures = ClosureLoopTracker()
        obs_closures.attach(synapse.event_bus)
        app.state.observatory_closures = obs_closures

        app.state.observatory_spec_checker = SpecComplianceChecker(obs_tracer)
        logger.info("observatory_attached")

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

            # Stop infra cost poller
            if hasattr(app.state, "infra_cost_poller"):
                phase1.append(_safe("infra_cost_poller", app.state.infra_cost_poller.stop()))

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

            if hasattr(app.state, "content_calendar"):
                phase1.append(_safe("content_calendar", app.state.content_calendar.stop()))

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
    #  RUNTIME INTROSPECTION API
    # ══════════════════════════════════════════════════════════════

    # Canonical system name → list of dependency names.
    # Used by get_dependency_graph(). Kept as a class-level constant so the
    # data is always available, even before startup completes.
    _DEPENDENCY_GRAPH: dict[str, list[str]] = {
        "memory":       [],
        "logos":        ["memory"],
        "equor":        ["memory", "logos"],
        "atune":        ["memory"],
        "eis":          [],
        "sacm":         [],
        "voxis":        ["memory"],
        "nova":         ["memory", "equor", "voxis"],
        "axon":         ["memory", "voxis"],
        "evo":          ["memory"],
        "thread":       ["memory"],
        "simula":       ["memory"],
        "synapse":      ["atune", "nova", "evo", "equor"],
        "thymos":       ["synapse", "equor", "evo"],
        "oneiros":      ["memory", "synapse", "equor", "evo"],
        "kairos":       ["synapse", "logos", "oneiros"],
        "soma":         ["synapse"],
        "telos":        ["synapse"],
        "fovea":        ["logos", "synapse", "atune"],
        "federation":   ["memory", "equor"],
        "nexus":        ["logos", "fovea", "federation", "thymos", "oneiros", "evo", "equor", "telos"],
        "oikos":        ["synapse"],
        "mitosis":      ["oikos", "equor", "synapse"],
        "phantom_liquidity": ["synapse"],
        "skia":         ["synapse"],
        "benchmarks":   ["synapse"],
    }

    def get_system_status(self, name: str, app: Any) -> dict[str, Any]:
        """
        Return live status for a single named system.

        Checks:
        1. Whether the system object exists on app.state.
        2. Whether associated background tasks (keyed by name) are alive.
        3. Last heartbeat via `system.health()` if available (sync probe only -
           returns None rather than awaiting to keep this method sync).

        Returns a dict with keys:
          name (str), status ("running"|"stopped"|"error"|"unknown"),
          task_alive (bool | None), initialized (bool | None), extra (dict)
        """
        svc = getattr(app.state, name, None)
        if svc is None:
            return {"name": name, "status": "stopped", "task_alive": None, "initialized": None, "extra": {}}

        # Check if the primary background task for this system is alive.
        task = self._tasks.get(name)
        task_alive: bool | None = None
        if task is not None:
            task_alive = not task.done()

        # Lightweight initialized probe (sync-safe attributes).
        initialized: bool | None = None
        if hasattr(svc, "_initialized"):
            initialized = bool(svc._initialized)
        elif hasattr(svc, "initialized"):
            initialized = bool(svc.initialized)

        # Extra system-specific details.
        extra: dict[str, Any] = {}
        if hasattr(svc, "system_id"):
            extra["system_id"] = svc.system_id

        status = "running" if (task_alive is not False and svc is not None) else "stopped"
        if task is not None and task.done() and not task.cancelled():
            exc = task.exception() if not task.cancelled() else None
            if exc is not None:
                status = "error"
                extra["last_error"] = str(exc)

        return {
            "name": name,
            "status": status,
            "task_alive": task_alive,
            "initialized": initialized,
            "extra": extra,
        }

    def get_all_systems(self, app: Any) -> list[dict[str, Any]]:
        """
        Return live status for all known systems.

        Iterates over _DEPENDENCY_GRAPH keys plus any tasks registered in
        self._tasks to ensure no background-task-only processes are missed.
        """
        names: set[str] = set(self._DEPENDENCY_GRAPH.keys()) | set(self._tasks.keys())
        return [self.get_system_status(n, app) for n in sorted(names)]

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """
        Return the static inter-system dependency graph.

        Keys are system names; values are lists of systems they depend on.
        Useful for topology-aware restarts and introspection dashboards.
        """
        return dict(self._DEPENDENCY_GRAPH)

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
        from systems.fovea.gateway import AtuneConfig, AtuneService

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

    def _init_sacm(self, config: Any = None, infra: Any = None) -> dict[str, Any]:
        from systems.sacm.accounting import SACMCostAccounting
        from systems.sacm.compute_manager import ComputeResourceManager
        from systems.sacm.config import SACMPreWarmConfig
        from systems.sacm.migrator import CostTriggeredMigrationMonitor, MigrationExecutor
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

        # MigrationExecutor + CostTriggeredMigrationMonitor - were fully implemented
        # but never instantiated in registry.py (noted as Known Issue #5 in CLAUDE.md).
        # Requires config.compute_arbitrage (ComputeArbitrageConfig), config.skia
        # (SkiaConfig), and infra.redis.  Guards against missing config/infra gracefully
        # so existing startup without config is non-breaking.
        migration_executor: Any = None
        migration_monitor: Any = None
        try:
            if config is not None and infra is not None and infra.redis is not None:
                _arbitrage_cfg = getattr(config, "compute_arbitrage", None)
                _skia_cfg = getattr(config, "skia", None)
                if _arbitrage_cfg is not None and _skia_cfg is not None:
                    migration_executor = MigrationExecutor(
                        config=_arbitrage_cfg,
                        skia_config=_skia_cfg,
                        redis=infra.redis,
                    )
                    migration_monitor = CostTriggeredMigrationMonitor(
                        migration_executor=migration_executor,
                        config=_arbitrage_cfg,
                    )
                    # Wire oracle immediately (no Synapse dependency)
                    migration_monitor.set_oracle(oracle)
                    logger.info(
                        "sacm_migration_executor_created",
                        provider=getattr(_arbitrage_cfg, "current_provider", "unknown"),
                    )
        except Exception as _mig_exc:
            logger.warning(
                "sacm_migration_executor_init_failed",
                error=str(_mig_exc),
                note="SACM migration disabled - non-fatal",
            )

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
            "sacm_migration_executor": migration_executor,
            "sacm_migration_monitor": migration_monitor,
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

    async def _init_reasoning_engine(self) -> Any:
        """
        Initialize the local Reasoning Engine (vLLM wrapper).

        Completely optional - if vLLM is not running or ORGANISM_RE_ENABLED=false,
        returns None and the organism operates in Claude-only mode.
        """
        import os

        if os.environ.get("ORGANISM_RE_ENABLED", "true").lower() in {"false", "0", "no"}:
            logger.info("reasoning_engine_disabled")
            return None

        try:
            from systems.reasoning_engine.service import ReasoningEngineService

            re_service = ReasoningEngineService()
            await re_service.initialize()
            return re_service
        except Exception as exc:
            logger.warning(
                "reasoning_engine_init_failed",
                error=str(exc),
                note="Continuing in Claude-only mode",
            )
            return None

    async def _init_nova(
        self,
        config: Any,
        infra: InfraClients,
        memory: Any,
        equor: Any,
        voxis: Any,
        re_service: Any = None,
    ) -> Any:
        from systems.nova.service import NovaService
        from systems.nova.policy_generator import PolicyGenerator, ThompsonSampler

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

            # Wire RE client into PolicyGenerator if available
            if re_service is not None and re_service.is_available:
                policy_gen = nova._policy_generator  # type: ignore[attr-defined]
                if isinstance(policy_gen, PolicyGenerator):
                    policy_gen._re_client = re_service
                    policy_gen._sampler.set_re_ready(True)
                    logger.info(
                        "nova_re_wired",
                        model=re_service._model,
                        url=re_service._url,
                    )
            else:
                logger.info(
                    "nova_re_disabled",
                    reason="RE not available or disabled - Claude-only mode",
                )

            # Always store RE service ref on Nova so the RE_ENGINE_STATUS_CHANGED
            # handler can late-wire when vLLM becomes available after startup.
            if re_service is not None:
                nova.set_re_service(re_service)

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
            neo4j_client=infra.neo4j,
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
        # Wire Redis so PatternContext counters survive restarts between
        # consolidation cycles (Spec §III gap fix - set_redis() was implemented
        # but never called, silently losing up to 6h of accumulated detector state).
        if infra.redis is not None:
            evo.set_redis(infra.redis)
        return evo

    async def _init_thread(self, config: Any, infra: InfraClients, memory: Any) -> Any:
        from systems.thread.service import ThreadService

        thread = ThreadService(
            memory=memory,
            instance_name=config.instance_id,
            neuroplasticity_bus=infra.neuroplasticity_bus,
        )
        thread.set_neo4j(infra.neo4j)
        # Wire LLM so CommitmentKeeper, IdentitySchemaEngine, NarrativeRetriever,
        # and DiachronicCoherenceMonitor are instantiated. Without this call
        # Thread runs in degraded mode with no schema evaluation or LLM-backed
        # commitment testing (set_llm() was implemented but never called).
        if infra.llm is not None:
            thread.set_llm(infra.llm)
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
                    run_worker(os.getenv("ORGANISM_CONFIG_PATH")),
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
        # Explicit set_neo4j() call so MetaCognition and DirectedExploration
        # in LucidDreamingStage receive the driver (constructor already forwards
        # it, but this makes the wiring visible and survives any refactor).
        if infra.neo4j is not None:
            oneiros.set_neo4j(infra.neo4j)
        await oneiros.initialize()
        return oneiros

    def _init_kairos(self, synapse: Any, logos: Any, oneiros: Any) -> Any:
        from systems.kairos.pipeline import KairosPipeline

        kairos = KairosPipeline()
        kairos.set_event_bus(synapse.event_bus)
        kairos.set_logos(logos)
        oneiros.set_kairos(kairos)
        # Start the periodic pipeline loop - this is what makes events actually fire.
        # run_pipeline() is never called on inbound events alone; the loop is required.
        kairos.start_pipeline_loop()
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

        # Wire Neo4j into OikosService for M2 immutable economic event audit trail.
        # set_neo4j() was implemented (service.py:383) but never called - all
        # _audit_economic_event() Neo4j writes were silently no-oping.
        if getattr(infra, "neo4j", None) is not None:
            oikos.set_neo4j(infra.neo4j)
            logger.info("ecodiaos_ready", phase="15e_oikos_neo4j_wired")

        # ── Construct MitosisFleetService ──────────────────────────────────────
        # MitosisFleetService was implemented but NEVER instantiated anywhere in
        # the codebase.  Every background loop (health monitor, dividend, fleet
        # eval, reproductive fitness) and all 9 Synapse subscriptions were
        # permanently dead.  Construct it here alongside Oikos since both share
        # the same dependencies (config, wallet, neo4j, event_bus, spawner).
        # SpawnChildExecutor receives the reference via wire_mitosis_phase() which
        # calls set_fleet_service() on the executor after Axon is initialized.
        try:
            from systems.mitosis.fleet_service import MitosisFleetService
            from systems.mitosis.genome_orchestrator import GenomeOrchestrator
            from systems.mitosis.spawner import LocalDockerSpawner

            _genome_orchestrator = GenomeOrchestrator(neo4j=infra.neo4j)

            # LocalDockerSpawner: only constructed when Docker-in-Docker is
            # available.  On managed runtimes (Cloud Run, Akash) it is skipped.
            _mitosis_spawner: Any = None
            try:
                _mitosis_spawner = LocalDockerSpawner(
                    config=config.oikos,
                    instance_id=config.instance_id,
                )
            except Exception as _sp_exc:
                logger.info(
                    "mitosis_spawner_skipped",
                    reason=str(_sp_exc),
                    note="Container-based child spawning disabled",
                )

            _fleet_service = MitosisFleetService(
                config=config.oikos,
                event_bus=synapse.event_bus,
                genome_orchestrator=_genome_orchestrator,
                neo4j=infra.neo4j,
                wallet=infra.wallet,
                spawner=_mitosis_spawner,
                instance_id=config.instance_id,
            )
            app.state.fleet_service = _fleet_service
            app.state.genome_orchestrator = _genome_orchestrator
            logger.info(
                "ecodiaos_ready",
                phase="15e_mitosis_fleet_service_constructed",
                spawner_ready=_mitosis_spawner is not None,
            )
        except Exception as _fleet_exc:
            logger.warning(
                "mitosis_fleet_service_construction_failed",
                error=str(_fleet_exc),
                note="Fleet management disabled - genome inheritance, health monitoring, and dividends will not run",
            )

        # Start SnapshotWriter - Redis ring buffer + TimescaleDB + hourly Neo4j CostSnapshot
        from systems.oikos.snapshot_writer import SnapshotWriter

        snapshot_writer = SnapshotWriter(oikos=oikos, redis=infra.redis)
        if getattr(infra, "timescale", None) is not None:
            snapshot_writer.set_timescale(infra.timescale)
        if getattr(infra, "neo4j", None) is not None:
            snapshot_writer.set_neo4j(infra.neo4j)
        await snapshot_writer.start()
        app.state.oikos_snapshot_writer = snapshot_writer
        self._tasks["oikos_snapshot_writer"] = snapshot_writer._task

        logger.info("ecodiaos_ready", phase="15e_oikos")
        return oikos

    async def _init_alive_ws(
        self,
        config: Any,
        infra: InfraClients,
        soma: Any,
        synapse: Any,
        telos: Any,
        thymos: Any,
        nova: Any,
        axon: Any,
        oikos: Any,
        simula: Any,
        *,
        atune: Any = None,
        fovea: Any = None,
        kairos: Any = None,
        logos: Any = None,
        oneiros: Any = None,
    ) -> Any:
        from systems.alive.ws_server import AliveWebSocketServer

        # Auth tokens from config (set → auth enforced; empty/absent → open mode)
        raw_tokens: list[str] = getattr(getattr(config, "alive_ws", None), "auth_tokens", []) or []
        auth_tokens: set[str] = set(raw_tokens)

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
            fovea=fovea,
            kairos=kairos,
            logos=logos,
            oneiros=oneiros,
            port=getattr(config, "alive_ws_port", 8001),
            auth_tokens=auth_tokens if auth_tokens else None,
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

        vault_pw = os.environ.get("ORGANISM_VAULT_PASSPHRASE", "")
        phantom_vault = None
        if vault_pw:
            from systems.identity.vault import IdentityVault
            phantom_vault = IdentityVault(passphrase=vault_pw)

        pl = LiquidityPhantomService(
            config=config.phantom_liquidity,
            wallet=infra.wallet,
            atune=atune,
            oikos=oikos,
            tsdb=infra.tsdb,
            neo4j=infra.neo4j,
            vault=phantom_vault,
            instance_id=config.instance_id,
        )
        await pl.initialize()
        pl.attach(synapse.event_bus)
        app.state.phantom_liquidity = pl
        logger.info("ecodiaos_ready", phase="15e_half_phantom_liquidity")

    async def _init_skia(
        self,
        config: Any,
        infra: InfraClients,
        synapse: Any,
        app: Any,
        *,
        memory: Any = None,
        oikos: Any = None,
        thymos: Any = None,
        equor: Any = None,
        telos: Any = None,
    ) -> None:
        if not config.skia.enabled:
            app.state.skia = None
            logger.info("skia_disabled")
            return
        from systems.identity.vault import IdentityVault
        from systems.skia.service import SkiaService

        vault_pw = os.environ.get("ORGANISM_VAULT_PASSPHRASE", "")
        vault = IdentityVault(passphrase=vault_pw) if vault_pw else None
        skia = SkiaService(
            config=config.skia,
            neo4j=infra.neo4j,
            redis=infra.redis,
            vault=vault,
            instance_id=config.instance_id,
            standalone=False,
        )
        # Wire Memory before initialize() so snapshots include the constitutional
        # genome (SnapshotPayload.constitutional_genome).  Without this, every
        # IPFS backup is genome-blind and restored organisms start with default
        # drives instead of the parent's evolved constitutional state.
        if memory is not None:
            skia.set_memory(memory)
        await skia.initialize()
        skia.set_event_bus(synapse.event_bus)
        await skia.start()
        # Wire system references into VitalityCoordinator so all five vitality
        # dimensions are live (runway, effective_I, constitutional drift, immune
        # health, somatic collapse).  Without this every reading returns NaN and
        # the organism is constitutionally BLIND to its own death approach.
        skia.wire_vitality_systems(
            clock=getattr(synapse, "_clock", None),
            oikos=oikos,
            thymos=thymos,
            equor=equor,
            telos=telos,
        )
        synapse.register_system(skia)
        app.state.skia = skia
        logger.info(
            "ecodiaos_ready",
            phase="15f_skia",
            mode="embedded",
            memory_wired=memory is not None,
            oikos_wired=oikos is not None,
            thymos_wired=thymos is not None,
            equor_wired=equor is not None,
            telos_wired=telos is not None,
        )

    def _init_self_model(
        self, config: Any, memory: Any, synapse: Any, app: Any
    ) -> None:
        """Instantiate SelfModelService and wire it into VitalityCoordinator.

        The self-model is additive - it does NOT modify cryptographic identity.
        Non-fatal: if Skia or VitalityCoordinator is unavailable, log and skip.
        """
        try:
            from systems.identity.self_model import SelfModelService

            self_model = SelfModelService(
                instance_id=config.instance_id,
                memory=memory,
                event_bus=synapse.event_bus,
            )
            app.state.self_model = self_model

            # Wire into VitalityCoordinator inside SkiaService
            skia = getattr(app.state, "skia", None)
            if skia is not None and hasattr(skia, "_vitality"):
                skia._vitality.set_self_model(self_model)
                logger.info("ecodiaos_ready", phase="15f_self_model", wired=True)
            else:
                logger.warning("self_model_vitality_not_found", skia_available=skia is not None)

        except Exception as exc:
            logger.warning("self_model_init_failed", error=str(exc))

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

        vault_pw = os.environ.get("ORGANISM_VAULT_PASSPHRASE", "")
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
                    if infra.tsdb is None:
                        raise RuntimeError("TimescaleDB not available")
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
        import asyncio as _asyncio
        from systems.identity.connectors.github import GitHubConnector

        connectors = app.state.connectors
        github_app = connectors.get("github_app:default")
        event_bus_ref = getattr(synapse, "event_bus", None)
        vault_ref = getattr(app.state, "identity", None)
        vault_ref = getattr(vault_ref, "vault", None)
        github_connector = GitHubConnector(
            app_connector=github_app,  # type: ignore[arg-type]
            redis=infra.redis,
            event_bus=event_bus_ref,
            vault=vault_ref,
        )
        # Fire-and-forget: emit CONNECTOR_AUTHENTICATED after event loop is running.
        _asyncio.ensure_future(github_connector.authenticate())
        app.state.github_connector = github_connector
        axon.set_github_connector(github_connector)
        oikos.set_github_connector(github_connector)
        oikos.set_bounty_submit_fn(create_bounty_submit_fn(axon))

        github_token = (
            os.environ.get("ORGANISM_EXTERNAL_PLATFORMS__GITHUB_TOKEN")
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
            logger.warning("GITHUB_CREDENTIALS_MISSING - bounty PR submission disabled")

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
        telos: Any = None,
        logos: Any = None,
        memory: Any = None,
        re_service: Any = None,
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
        benchmarks.set_telos(telos)
        benchmarks.set_logos(logos)
        benchmarks.set_event_bus(synapse.event_bus)
        benchmarks.set_redis(infra.redis)
        benchmarks.set_memory(memory)
        await benchmarks.initialize()
        # Wire RE service into the 5-pillar evaluation protocol (best-effort)
        if re_service is not None:
            benchmarks.set_re_service(re_service)
            # Wire the same RE service into the direct pillar evaluation path (pillars.py).
            # set_re_service() targets EvaluationProtocol; set_reasoning_engine() targets
            # _reasoning_engine used by measure_specialization/novelty/causal/memorization.
            benchmarks.set_reasoning_engine(re_service)
        app.state.benchmarks = benchmarks
        alive_ws._benchmarks = benchmarks
        # Wire RE service into Alive so re_status section is populated (Spec 11 §22 gap 3)
        if re_service is not None:
            alive_ws.set_re_service(re_service)
        logger.info("ecodiaos_ready", phase="20_benchmarks")

    # ── Helpers ───────────────────────────────────────────────

    async def _check_skia_restore(
        self, config: Any, infra: InfraClients, *, memory: Any = None
    ) -> None:
        restore_cid = os.environ.get("ORGANISM_SKIA_RESTORE_CID", "")
        if not restore_cid:
            return
        vault_passphrase = os.environ.get("ORGANISM_VAULT_PASSPHRASE", "")
        if not vault_passphrase:
            raise RuntimeError(
                "ORGANISM_SKIA_RESTORE_CID is set but ORGANISM_VAULT_PASSPHRASE is empty."
            )
        from systems.skia.snapshot import restore_from_ipfs

        # Pass memory so restore_from_ipfs() can call memory.seed_genome()
        # when the snapshot contains a constitutional genome (schema_version ≥ 2).
        # This ensures the revived organism inherits the parent's drive weights
        # rather than starting with EcodiaOS defaults.
        # event_bus is not available this early in startup (Synapse not yet
        # initialized), so GENOME_EXTRACT_REQUEST broadcast is deferred to when
        # the bus comes online - memory.seed_genome() is the primary restore path.
        await restore_from_ipfs(
            cid=restore_cid,
            neo4j=infra.neo4j,
            vault_passphrase=vault_passphrase,
            pinata_jwt=config.skia.pinata_jwt,
            pinata_api_url=config.skia.pinata_api_url,
            pinata_gateway_url=config.skia.pinata_gateway_url,
            memory=memory,
        )
        logger.info("skia_state_restored", cid=restore_cid)

    async def _birth_or_load(
        self, config: Any, infra: InfraClients, memory: Any, equor: Any, atune: Any
    ) -> None:
        instance = await memory.get_self()
        if instance is None:
            seed_path = os.environ.get("ORGANISM_SEED_PATH", "config/seeds/example_seed.yaml")
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

        seed_path = os.environ.get("ORGANISM_SEED_PATH", "config/seeds/example_seed.yaml")
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
                        " - understand who I serve and what they need"
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
                        " - explore my capabilities and how my drives shape my behaviour"
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
