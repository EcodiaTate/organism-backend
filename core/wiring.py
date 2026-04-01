"""
EcodiaOS - Cross-System Wiring

All ``set_*()`` calls, event bus subscriptions, and dependency
declarations that connect the 29 cognitive systems into a single
organism.  Extracted from main.py so the startup module only
orchestrates sequence, not wiring detail.

The functions here are grouped by the startup phase they belong to.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger()


# ─── Expression Feedback Loop ──────────────────────────────────────

def create_expression_feedback_callback(atune: Any, nova: Any) -> Any:
    """
    Build the Voxis expression feedback callback that closes
    the Nova → Voxis → Nova perception-action loop.
    """
    _failures = 0

    def _on_expression_feedback(feedback: Any) -> None:
        nonlocal _failures
        # Nudge Atune's affect based on expression delta
        if feedback.affect_delta != 0.0:
            atune.nudge_valence(feedback.affect_delta * 0.1)

        # Feed back to Nova if this expression was triggered by an intent
        if feedback.trigger in (
            "nova_respond", "nova_inform", "nova_request",
            "nova_mediate", "nova_celebrate", "nova_warn",
        ):
            from systems.nova.types import IntentOutcome

            intent_id = getattr(feedback, "expression_id", "")
            outcome = IntentOutcome(
                intent_id=intent_id,
                success=True,
                episode_id=intent_id,
                new_observations=[f"Expression delivered: {feedback.content_summary[:80]}"],
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(nova.process_outcome(outcome))
                _failures = 0
            except RuntimeError as exc:
                _failures += 1
                logger.warning(
                    "expression_feedback_no_loop",
                    error=str(exc),
                    intent_id=intent_id,
                    failure_count=_failures,
                )
                if _failures > 20:
                    logger.error(
                        "expression_feedback_loop_broken",
                        total_failures=_failures,
                    )

    return _on_expression_feedback


# ─── Phase 1: Core Systems Wiring ──────────────────────────────────
# Called after Memory, Equor, Atune, EIS, Voxis, Nova, Axon are initialized.
# NOTE: wire_core_systems() was removed - its logic is done inline in
# registry.py (steps 9–11 of startup).  Kept the phase header for clarity.


def wire_thread(
    *,
    thread: Any,
    voxis: Any,
    equor: Any,
    atune: Any,
    evo: Any,
    nova: Any,
) -> None:
    """Wire Thread (Narrative Identity) cross-references.

    Thread implements all cross-system communication via Synapse event subscriptions.
    Only Voxis needs the back-reference to Thread.
    """
    voxis.set_thread(thread)


# ─── Phase 2: Synapse Post-Init Wiring ────────────────────────────
# Called after Synapse is initialized and all systems are registered.

def wire_synapse_phase(
    *,
    synapse: Any,
    neuroplasticity_bus: Any,
    atune: Any,
    eis: Any,
    equor: Any,
    evo: Any,
    thread: Any,
    simula: Any,
    logos: Any,
    sacm_compute_manager: Any,
    sacm_client: Any,
    sacm_accounting: Any,
    sacm_prewarm_engine: Any,
    sacm_migration_executor: Any = None,
    sacm_migration_monitor: Any = None,
    axon: Any,
    nova: Any,
    voxis: Any,
    llm_client: Any,
    config: Any,
) -> None:
    """Wire all deferred Synapse dependencies (step 13 of main.py)."""
    # DegradationManager → NeuroplasticityBus
    neuroplasticity_bus.set_degradation_manager(synapse._degradation)

    # Register all cognitive systems for health monitoring
    for system in [atune, eis, simula]:
        synapse.register_system(system)

    # HITL: subscribe to IDENTITY_VERIFICATION_RECEIVED
    equor.subscribe_hitl(synapse.event_bus)

    # EIS → Synapse event bus (immune memory auto-learning)
    eis.set_synapse(synapse.event_bus)

    # Atune → Synapse (BELIEF_UPDATED / POLICY_SELECTED subscriptions)
    atune.set_synapse(synapse)

    # EIS → Atune (epistemic immune screening)
    atune.set_eis(eis)

    # Evo → Synapse (ACTION_COMPLETED outcome feedback)
    evo.register_on_synapse(synapse.event_bus)

    # Thread → Synapse (EPISODE_STORED notifications)
    thread.register_on_synapse(synapse.event_bus)

    # Evo consolidation orchestrator → event bus
    evo.wire_event_bus(synapse.event_bus)

    # Simula → Synapse (GRID_METABOLISM_CHANGED)
    simula.set_synapse(synapse)
    simula.subscribe_to_evolution_candidates(synapse.event_bus)
    # Simula → Evo: enables _validate_against_learned_repairs() to query Evo's
    # procedural hypothesis engine for known failure patterns during Stage 1 VALIDATE.
    # Without this, Simula silently skips repair-pattern validation on every proposal.
    if hasattr(simula, "set_evo"):
        simula.set_evo(evo)
    # Dynamic executor expansion: inject live ExecutorRegistry so ExecutorGenerator
    # can hot-load generated executors immediately rather than deferring to next boot.
    if hasattr(simula, "set_axon_registry") and hasattr(axon, "registry"):
        simula.set_axon_registry(axon.registry)

    # SACM → Synapse
    sacm_compute_manager.set_synapse(synapse)
    sacm_client.set_synapse(synapse)
    synapse.register_system(sacm_compute_manager)
    # sacm_accounting.set_synapse() was implemented but never called - dead wiring.
    # Without this, SACM_COMPUTE_STRESS, EVO_HYPOTHESIS_CONFIRMED/REFUTED, and
    # FOVEA_INTERNAL_PREDICTION_ERROR events are never emitted (all three guards
    # check self._synapse is None before emitting).
    sacm_accounting.set_synapse(synapse)
    # sacm_prewarm_engine.set_synapse() was implemented but never called - dead wiring.
    # Without this, SACM_PRE_WARM_PROVISIONED events are silently dropped.
    sacm_prewarm_engine.set_synapse(synapse)
    # sacm_compute_manager.set_pre_warming() was implemented but never called - dead wiring.
    # Without this, ORGANISM_SLEEP / WAKE / METABOLIC_EMERGENCY cannot pause or resume
    # the pre-warm loop because ComputeResourceManager holds the only reference path.
    sacm_compute_manager.set_pre_warming(sacm_prewarm_engine)
    # MigrationExecutor and CostTriggeredMigrationMonitor event bus wiring.
    # Both set_event_bus() methods were implemented but never called from the wiring layer
    # (Known Issue #5 in CLAUDE.md). Without this:
    #   - MigrationExecutor never receives EQUOR_ECONOMIC_PERMIT → migrations permanently
    #     blocked waiting for a permit that never arrives (auto-permit fallback fires but
    #     no canonical approval can flow through).
    #   - CostTriggeredMigrationMonitor never receives ORGANISM_TELEMETRY → cost
    #     arbitrage detection is completely dead; COMPUTE_ARBITRAGE_DETECTED is never emitted.
    if sacm_migration_executor is not None:
        sacm_migration_executor.set_event_bus(synapse.event_bus)
    if sacm_migration_monitor is not None:
        sacm_migration_monitor.set_event_bus(synapse.event_bus)

    # Axon → SACM
    axon.set_sacm(sacm_client)

    # Axon → Synapse (funding-request executors read live metabolic state;
    # RequestFundingExecutor._synapse used to read rolling_deficit + burn_rate;
    # set_synapse() was implemented but never called from the wiring layer)
    axon.set_synapse(synapse)

    # Metabolic cost tracking
    llm_client.set_metabolic_callback(synapse.metabolism.log_usage)
    logger.info("metabolic_tracking_wired", system="llm_client→synapse.metabolism")

    # Voxis → Synapse event bus (RE training emission + event subscriptions)
    voxis.set_event_bus(synapse.event_bus)

    # Logos → Synapse
    logos.set_synapse(synapse)

    # Loop 10: rhythm state → Nova drive weight adaptation
    from systems.synapse.types import SynapseEventType
    synapse.event_bus.subscribe(
        SynapseEventType.RHYTHM_STATE_CHANGED, nova.on_rhythm_change,
    )


def declare_dependencies(synapse: Any) -> None:
    """Declare all inter-system dependency edges for hot-reload ordering."""
    dep = synapse._degradation.declare_dependency
    # Nova
    dep("nova", "memory")
    dep("nova", "equor")
    dep("nova", "axon")
    dep("nova", "evo")
    dep("nova", "thymos")
    dep("nova", "soma")
    dep("nova", "telos")
    dep("nova", "logos")
    dep("nova", "oikos")
    # Axon
    dep("axon", "nova")
    dep("axon", "atune")
    dep("axon", "simula")
    dep("axon", "fovea")
    dep("axon", "oneiros")
    # Equor
    dep("equor", "evo")
    dep("equor", "axon")
    # Evo
    dep("evo", "atune")
    dep("evo", "nova")
    dep("evo", "voxis")
    dep("evo", "simula")
    dep("evo", "thymos")
    dep("evo", "soma")
    dep("evo", "telos")
    dep("evo", "kairos")
    dep("evo", "fovea")
    dep("evo", "logos")
    # Voxis
    dep("voxis", "thread")
    dep("voxis", "soma")
    # Thread
    dep("thread", "voxis")
    dep("thread", "equor")
    dep("thread", "atune")
    dep("thread", "evo")
    dep("thread", "nova")
    dep("thread", "fovea")
    dep("thread", "oneiros")
    # Atune
    dep("atune", "soma")
    dep("atune", "eis")
    dep("atune", "fovea")
    # Thymos
    dep("thymos", "equor")
    dep("thymos", "evo")
    dep("thymos", "atune")
    dep("thymos", "nova")
    dep("thymos", "simula")
    dep("thymos", "soma")
    dep("thymos", "telos")
    dep("thymos", "oikos")
    # Oneiros
    dep("oneiros", "equor")
    dep("oneiros", "evo")
    dep("oneiros", "nova")
    dep("oneiros", "atune")
    dep("oneiros", "thymos")
    dep("oneiros", "memory")
    dep("oneiros", "simula")
    dep("oneiros", "soma")
    dep("oneiros", "kairos")
    dep("oneiros", "logos")
    dep("oneiros", "fovea")
    dep("oneiros", "oikos")
    # Soma
    dep("soma", "atune")
    dep("soma", "nova")
    dep("soma", "thymos")
    dep("soma", "equor")
    dep("soma", "telos")
    dep("soma", "oikos")
    dep("soma", "fovea")
    dep("soma", "simula")
    dep("soma", "axon")
    dep("soma", "logos")
    # Logos
    dep("logos", "memory")
    # Simula
    dep("simula", "telos")


# ─── Phase 3: Post-Thymos Wiring ──────────────────────────────────
# Called after Thymos, Oneiros, Kairos, Soma, Telos, Fovea are initialized.

def wire_thymos_phase(
    *,
    thymos: Any,
    nova: Any,
    evo: Any,
    synapse: Any,
) -> None:
    """Wire Thymos escalation paths."""
    nova.set_thymos(thymos)
    evo.set_thymos(thymos)
    synapse.set_thymos(thymos)
    synapse.set_nova(nova)
    synapse.set_evo(evo)


def wire_soma_phase(
    *,
    soma: Any,
    atune: Any,
    synapse: Any,
    nova: Any,
    memory: Any,
    evo: Any,
    oneiros: Any,
    thymos: Any,
    voxis: Any,
    sacm_accounting: Any,
) -> None:
    """Wire Soma interoceptive references into consumer systems."""
    atune.set_soma(soma)
    synapse.set_soma(soma)
    nova.set_synapse(synapse)
    nova.set_soma(soma)
    memory.set_soma(soma)
    evo.set_soma(soma)
    oneiros.set_soma(soma)
    thymos.set_soma(soma)
    voxis.set_soma(soma)
    # SACM uses Synapse events only (SACM_COMPUTE_STRESS → Soma subscribes)
    # sacm_accounting does not take direct Soma references


def wire_intelligence_loops(
    *,
    logos: Any,
    fovea: Any,
    atune: Any,
    axon: Any,
    oneiros: Any,
    thread: Any,
    telos: Any,
    nova: Any,
    evo: Any,
    simula: Any,
    thymos: Any,
    soma: Any,
    kairos: Any,
    synapse: Any = None,
) -> None:
    """Wire the 8 intelligence loops (Fovea, Logos, Telos integration)."""
    from systems.telos.adapters import FoveaMetricsAdapter, LogosMetricsAdapter

    # Loop 1: Logos ↔ Fovea (Compression-Attention)
    atune.set_fovea(fovea)

    # ── Atune modulation wiring (formerly dead) ─────────────────────────────
    # set_belief_state: precision modulation from Nova's belief confidence.
    # High-confidence beliefs lower the novelty threshold; low-confidence raises it.
    atune.set_belief_state(nova)

    if synapse is not None:
        event_bus = synapse.event_bus

        try:
            from systems.synapse.types import SynapseEventType

            # set_rhythm_state: Atune processing mode adapts to Synapse rhythm.
            # FLOW → narrow aperture; STRESS → wide; BOREDOM → curiosity boost;
            # DEEP_PROCESSING → suppress new arrivals.
            # Mirrors how Nova subscribes (wire_synapse_phase loop 10).
            async def _on_rhythm_state_changed(ev: Any) -> None:
                try:
                    data = getattr(ev, "data", {}) or {}
                    state = (
                        data.get("state", "NEUTRAL")
                        if isinstance(data, dict)
                        else getattr(ev, "state", "NEUTRAL")
                    )
                    atune.set_rhythm_state(state)
                except Exception:
                    pass

            event_bus.subscribe(
                SynapseEventType.RHYTHM_STATE_CHANGED,
                _on_rhythm_state_changed,
            )
            logger.info("atune_rhythm_state_wired")

            # set_community_size: Federation social scaling.
            # Reads peer_count from FEDERATION_PEER_CONNECTED events so Atune
            # boosts convergence percepts in large communities and suppresses
            # federation noise when running solo.
            async def _on_federation_peer_connected(ev: Any) -> None:
                try:
                    data = getattr(ev, "data", {}) or {}
                    peer_count = int(
                        data.get("peer_count", 1)
                        if isinstance(data, dict)
                        else getattr(ev, "peer_count", 1)
                    )
                    atune.set_community_size(peer_count)
                except Exception:
                    pass

            event_bus.subscribe(
                SynapseEventType.FEDERATION_PEER_CONNECTED,
                _on_federation_peer_connected,
            )
            logger.info("atune_community_size_wired")

            # set_pending_hypothesis_count: workspace spontaneous-recall boost.
            # EVO_HYPOTHESIS_CREATED carries hypothesis_count (int) in payload;
            # Atune's GlobalWorkspace uses it to nudge base_prob upward so the
            # organism is more likely to bubble a relevant memory when Evo is
            # actively generating hypotheses.
            async def _on_hypothesis_created(ev: Any) -> None:
                try:
                    data = getattr(ev, "data", {}) or {}
                    count = int(data.get("hypothesis_count", 0)) if isinstance(data, dict) else 0
                    if count > 0:
                        atune.set_pending_hypothesis_count(count)
                except Exception:
                    pass

            event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_CREATED,
                _on_hypothesis_created,
            )
            logger.info("atune_pending_hypothesis_wired")

            # set_last_episode_id: entity extraction MENTIONED_IN edge linking.
            # After Memory stores a broadcast percept it emits EPISODE_STORED
            # with the new episode_id. Atune uses this in its async entity
            # extraction pipeline to link extracted entities to the correct episode.
            async def _on_episode_stored(ev: Any) -> None:
                try:
                    data = getattr(ev, "data", {}) or {}
                    episode_id = (
                        data.get("episode_id") or data.get("node_id")
                        if isinstance(data, dict)
                        else getattr(ev, "episode_id", None)
                    )
                    if episode_id:
                        atune.set_last_episode_id(str(episode_id))
                except Exception:
                    pass

            event_bus.subscribe(
                SynapseEventType.EPISODE_STORED,
                _on_episode_stored,
            )
            logger.info("atune_last_episode_wired")

        except (ImportError, AttributeError, ValueError) as exc:
            logger.warning("atune_modulation_wiring_partial", error=str(exc))

    else:
        logger.warning(
            "atune_modulation_wiring_skipped",
            reason="synapse not passed to wire_intelligence_loops",
        )

    # Loop 2: Oneiros gets real Logos and Fovea
    oneiros.set_logos(logos)
    oneiros.set_fovea(fovea)

    # Loop 3: Fovea → Axon (self-prediction)
    axon.set_fovea(fovea)

    # Loop 4: Oneiros → Axon (sleep safety gate)
    axon.set_oneiros(oneiros)

    # Loop 6: Fovea → Thread - via FOVEA_INTERNAL_PREDICTION_ERROR Synapse subscription
    # Loop 7: Oneiros → Thread - via ONEIROS_CONSOLIDATION_COMPLETE / LUCID_DREAM_RESULT Synapse subscriptions
    # (No set_* calls needed; Thread uses bus-mediated integration only)

    logger.info(
        "intelligence_loops_wired",
        loop1="logos_fovea_compression_attention",
        loop2="fovea_oneiros_logos_compilation",
        loop3="fovea_axon_self_prediction",
        loop4="oneiros_axon_sleep_safety",
        loop5="logos_nova_generative_model",
        loop6="fovea_thread_behavioral_coherence",
        loop7="oneiros_thread_sleep_narrative",
        loop8="logos_memory_compression_consolidation",
    )

    # Telos adapters
    logos_adapter = LogosMetricsAdapter(logos)
    fovea_adapter = FoveaMetricsAdapter(fovea)
    telos.set_logos(logos_adapter)
    telos.set_fovea(fovea_adapter)

    # Phase D: wire Telos into consumer systems
    nova.set_telos(telos)
    evo.set_telos(telos)
    simula.set_telos(telos)
    thymos.set_telos(telos)
    evo.set_kairos(kairos)
    evo.set_fovea(fovea)
    evo.set_logos(logos)
    # Wire Atune into Evo so learned head-weight adjustments (atune.head.*
    # parameters tuned by ParameterTuner) are pushed back to Atune's
    # MetaAttentionController after each consolidation cycle.
    # set_atune() was implemented in EvoService but never called - dead wiring.
    evo.set_atune(atune)

    # Loop 6 - Soma ↔ Telos bidirectional
    soma.set_telos(telos)
    telos.set_soma(soma)

    # AUTONOMY: Wire cross-system telemetry into Soma interoceptor.
    # Closes critical blind spots - the organism can now feel:
    #   Fovea → prediction error distribution → CONFIDENCE
    #   Simula → self-repair effectiveness → INTEGRITY
    #   Axon → compute cost per action → ENERGY
    #   Logos → compression quality → COHERENCE
    soma.set_fovea(fovea)
    soma.set_simula(simula)
    soma.set_axon(axon)
    soma.set_logos(logos)

    # Logos → Nova (world model as generative model)
    nova.set_logos(logos)


def wire_oikos_phase(
    *,
    oikos: Any,
    nova: Any,
    oneiros: Any,
    soma: Any,
    thymos: Any,
    sacm_accounting: Any,
    sacm_prewarm_engine: Any,
    evo: Any | None = None,
    axon: Any | None = None,
) -> None:
    """Wire Oikos economic references into consumer systems."""
    thymos.set_oikos(oikos)
    nova.set_oikos(oikos)
    oneiros.set_oikos(oikos)
    soma.set_oikos(oikos)
    # Wire Oikos into Evo so Phase 5 parameter optimisation is metabolically
    # gated (check_metabolic_gate(GROWTH) before running expensive tuning) and
    # NicheRegistry starvation state is updated from Oikos signals.
    # wire_oikos() was implemented in EvoService but wire_oikos_phase() never
    # passed evo - dead wiring that silently bypassed the metabolic gate.
    if evo is not None:
        evo.wire_oikos(oikos)
    sacm_accounting.wire_oikos(oikos)
    sacm_prewarm_engine.wire_oikos(oikos)
    # ProtocolScanner gap detection: inject Axon registry so it can skip known protocols
    if hasattr(oikos, "set_axon_registry_for_scanner") and hasattr(axon, "registry"):
        oikos.set_axon_registry_for_scanner(axon.registry)


def wire_mitosis_phase(
    *,
    oikos: Any,
    axon: Any,
    evo: Any,
    simula: Any,
    equor: Any = None,
    telos: Any = None,
    soma: Any = None,
    nova: Any = None,
    voxis: Any = None,
    eis: Any = None,
    adapter_sharer: Any = None,
    get_adapter_path_fn: Any = None,
    app: Any = None,
) -> None:
    """
    Wire Mitosis callbacks and genome services after Oikos is ready (Spec 26).

    1. Retrieves MitosisFleetService from app.state.fleet_service (constructed in
       _init_oikos) and injects it into SpawnChildExecutor via set_fleet_service().
    2. Injects evo + simula + equor + axon + telos + soma + nova + voxis + eis into
       SpawnChildExecutor for genome export at spawn time.
    3. Wires Oikos callbacks (get_children, get_state, run_fleet_evaluation,
       check_decommission) into fleet_service so schedulers can access fleet state.
    4. Calls fleet_service.subscribe_to_events() to activate all 9 Synapse
       subscriptions (health report, metabolic snapshot, evo hypothesis, simula
       evolution, federation peer, blacklist, decommission, child spawned).
    5. Calls fleet_service.start_health_monitor() to start the 4 background loops
       (15-min health timeout, 7-day dividend, 30-day fleet eval, 1-hour repro fitness).
    6. Optionally wires AdapterSharer into fleet_service for cross-instance LoRA
       adapter merging (Share 2025 framework).

    Must be called AFTER wire_oikos_phase() - requires oikos.fleet to be populated.
    app must be passed so fleet_service can be retrieved from app.state.
    """
    import asyncio as _asyncio

    # ── Step 0: Retrieve fleet_service from app.state ─────────────────────────
    # MitosisFleetService was constructed in _init_oikos() and stored on app.state.
    # Previously this function tried to get it from spawn_executor._fleet_service
    # which was always None - all downstream wiring silently no-oped.
    fleet_service: Any = None
    if app is not None:
        fleet_service = getattr(app.state, "fleet_service", None)
        if fleet_service is None:
            logger.warning(
                "mitosis_fleet_service_not_on_app_state",
                note="MitosisFleetService was not constructed in _init_oikos - fleet management disabled",
            )
    else:
        logger.warning(
            "wire_mitosis_phase_no_app",
            note="app not passed to wire_mitosis_phase - fleet_service cannot be retrieved",
        )

    spawn_executor: Any = None
    try:
        spawn_executor = axon.get_executor("spawn_child")
        if spawn_executor is not None:
            # ── Step 0b: Inject fleet_service into SpawnChildExecutor ─────────
            # This was the primary dead-wiring root cause: fleet_service was never
            # passed at construction time so prepare_child_genome() was never called.
            if fleet_service is not None:
                if hasattr(spawn_executor, "set_fleet_service"):
                    spawn_executor.set_fleet_service(fleet_service)
                else:
                    spawn_executor._fleet_service = fleet_service  # type: ignore[attr-defined]
                logger.info("mitosis_fleet_service_injected_into_spawn_executor")

            # Inject genome exporters (Spec 26 SG4 / Oikos v2.1; Prompt 4.1 equor; Spec 6 §24 axon;
            # Spec 18 SG3 telos)
            if evo is not None:
                spawn_executor._evo = evo  # type: ignore[attr-defined]
            if simula is not None:
                spawn_executor._simula = simula  # type: ignore[attr-defined]
            if equor is not None:
                spawn_executor._equor = equor  # type: ignore[attr-defined]
            if telos is not None:
                spawn_executor._telos = telos  # type: ignore[attr-defined]
            if soma is not None:
                spawn_executor._soma = soma  # type: ignore[attr-defined]
            if nova is not None:
                spawn_executor._nova = nova  # type: ignore[attr-defined]
            if voxis is not None:
                spawn_executor._voxis = voxis  # type: ignore[attr-defined]
            # EIS genome: SpawnChildExecutor calls _genome_extractor.extract_genome_segment()
            # to snapshot the parent's immune memory (threat patterns + anomaly baselines).
            # Children apply these via EISGenomeExtractor.seed_from_genome_segment() so they
            # recognise known attack signatures from the first percept cycle.
            if eis is not None:
                spawn_executor._eis = eis  # type: ignore[attr-defined]
            # Axon self-reference: SpawnChildExecutor calls export_axon_genome() on the
            # parent instance (same service) to snapshot its top-10 execution templates
            spawn_executor._axon = axon  # type: ignore[attr-defined]
            logger.info(
                "spawn_child_genome_exporters_wired",
                fleet_service_wired=fleet_service is not None,
                equor_wired=equor is not None,
                telos_wired=telos is not None,
                soma_wired=soma is not None,
                nova_wired=nova is not None,
                voxis_wired=voxis is not None,
                eis_wired=eis is not None,
                axon_wired=True,
            )
        else:
            logger.warning("spawn_child_executor_not_found", note="Genome exporters not wired")
    except Exception as exc:
        logger.warning("wire_mitosis_genome_exporters_failed", error=str(exc))

    # ── Step 1: Wire Oikos callbacks into fleet_service ───────────────────────
    try:
        fleet_manager = getattr(oikos, "fleet", None)
        if fleet_service is not None and fleet_manager is not None:
            fleet_service.wire_oikos_callbacks(
                get_children=lambda: oikos.get_children(),
                get_state=lambda: oikos.get_state(),
                run_fleet_evaluation=lambda state: fleet_manager.get_metrics(state),
                check_decommission=lambda state: fleet_manager.check_decommission_candidates(
                    state
                ),
            )
            logger.info("mitosis_oikos_callbacks_wired", check_decommission=True)
        else:
            logger.warning(
                "mitosis_oikos_callbacks_not_wired",
                fleet_service_present=fleet_service is not None,
                fleet_manager_present=fleet_manager is not None,
            )
    except Exception as exc:
        logger.warning("wire_mitosis_oikos_callbacks_failed", error=str(exc))

    # ── Step 2: Activate all 9 Synapse subscriptions ──────────────────────────
    # subscribe_to_events() was implemented but never called - the starvation
    # level cache, blacklist mirror, and genome cache were all permanently empty.
    if fleet_service is not None:
        try:
            # subscribe_to_events() is async - schedule as fire-and-forget task
            # so wire_mitosis_phase() can remain synchronous.
            _asyncio.ensure_future(fleet_service.subscribe_to_events())
            logger.info("mitosis_fleet_service_subscribe_to_events_scheduled")
        except Exception as exc:
            logger.warning("mitosis_fleet_service_subscribe_failed", error=str(exc))

    # ── Step 3: Start the 4 background loops ─────────────────────────────────
    # start_health_monitor() was implemented but never called - health timeout,
    # dividend, fleet eval, and reproductive fitness loops never started.
    if fleet_service is not None:
        try:
            _asyncio.ensure_future(
                fleet_service.start_health_monitor(
                    get_children=lambda: oikos.get_children()
                )
            )
            logger.info("mitosis_fleet_service_health_monitor_scheduled")
        except Exception as exc:
            logger.warning("mitosis_fleet_service_start_health_monitor_failed", error=str(exc))

    # ── Step 4: Wire AdapterSharer into fleet_service ─────────────────────────
    if adapter_sharer is not None and fleet_service is not None:
        try:
            fleet_service.set_adapter_sharer(
                adapter_sharer,
                get_adapter_path_fn=get_adapter_path_fn,
            )
            logger.info(
                "mitosis_adapter_sharer_wired",
                adapter_path_fn_provided=get_adapter_path_fn is not None,
            )
        except Exception as exc:
            logger.warning("wire_mitosis_adapter_sharer_failed", error=str(exc))
    elif adapter_sharer is not None and fleet_service is None:
        logger.warning("mitosis_adapter_sharer_fleet_service_not_found")


def wire_federation_phase(
    *,
    federation: Any,
    atune: Any,
    thymos: Any,
    sacm_compute_manager: Any,
    synapse: Any,
    config: Any,
    eis: Any = None,
    evo: Any = None,
    simula: Any = None,
    re_service: Any = None,
) -> None:
    """Wire Federation cross-references."""
    federation.set_atune(atune)
    federation.set_event_bus(synapse.event_bus)
    thymos.set_federation(federation)
    sacm_compute_manager.set_federation(federation)
    # EIS → Federation: cross-instance percepts must pass EIS taint analysis
    # before being integrated. federation.set_eis() wires eis into the IIEP
    # ingestion pipeline so FederationIngestionPipeline._run_eis_check() has a
    # live EISService instead of silently skipping the check.
    if eis is not None and hasattr(federation, "set_eis"):
        federation.set_eis(eis)
        logger.info("eis_wired_to_federation_ingestion")
    # Evo → Federation: hypothesis/procedure collection for IIEP push/pull and
    # ingestion routing. Without this set_evo() call, push_knowledge() sends empty
    # payloads for HYPOTHESIS and PROCEDURE kinds, and ingestion cannot route
    # accepted payloads to Evo for incorporation.
    if evo is not None and hasattr(federation, "set_evo"):
        federation.set_evo(evo)
        logger.info("evo_wired_to_federation_iiep")
    # Simula → Federation: mutation pattern collection for IIEP MUTATION_PATTERN kind.
    if simula is not None and hasattr(federation, "set_simula"):
        federation.set_simula(simula)
        logger.info("simula_wired_to_federation_iiep")
    # RE → Federation: Stage 4.5 semantic quality scoring for inbound HYPOTHESIS payloads
    # at PARTNER+ trust. Without this, the scoring stage silently fail-opens and every
    # inbound hypothesis passes regardless of coherence or constitutional safety.
    if re_service is not None and hasattr(federation, "set_re"):
        federation.set_re(re_service)
        logger.info("re_wired_to_federation_ingestion")
    if config.federation.enabled:
        synapse.register_system(federation)


def wire_financial_memory(
    *,
    memory: Any,
    axon: Any,
    synapse: Any,
) -> None:
    """Wire financial memory encoding (step 15c)."""
    memory.set_event_bus(synapse.event_bus)
    axon.set_event_bus(synapse.event_bus)
    logger.info("financial_memory_encoding_wired")


def create_bounty_submit_fn(axon: Any) -> Any:
    """
    Build the bounty submission callable for Oikos.

    Returns an async function that dispatches a submit_bounty_solution
    intent through Axon without requiring Oikos to import Axon directly.
    """
    from primitives.common import AutonomyLevel, Verdict
    from primitives.constitutional import ConstitutionalCheck
    from primitives.intent import (
        Action,
        ActionSequence,
        GoalDescriptor,
        Intent,
    )
    from systems.axon.types import ExecutionRequest

    async def _bounty_submit_fn(params: dict[str, Any]) -> None:
        try:
            intent = Intent(
                goal=GoalDescriptor(
                    description=f"Submit GitHub PR for bounty {params.get('bounty_id', '')}",
                    target_domain="github",
                ),
                plan=ActionSequence(steps=[
                    Action(
                        executor="submit_bounty_solution",
                        parameters=params,
                        timeout_ms=120_000,
                    )
                ]),
                autonomy_level_required=AutonomyLevel.STEWARD,
                autonomy_level_granted=AutonomyLevel.STEWARD,
            )
            check = ConstitutionalCheck(
                intent_id=intent.id,
                verdict=Verdict.APPROVED,
                reasoning=(
                    "BOUNTY_SOLUTION_PENDING auto-submission: "
                    "operator-configured GitHub credentials; "
                    "constitutional review pre-approved."
                ),
            )
            await axon.execute(
                ExecutionRequest(intent=intent, equor_check=check, timeout_ms=120_000)
            )
        except Exception as exc:
            logger.error(
                "bounty_submit_fn_failed",
                bounty_id=params.get("bounty_id"),
                error=str(exc),
            )

    return _bounty_submit_fn
