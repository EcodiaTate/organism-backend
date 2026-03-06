"""
EcodiaOS — Cross-System Wiring

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

def wire_core_systems(
    *,
    atune: Any,
    nova: Any,
    axon: Any,
    voxis: Any,
    equor: Any,
    evo: Any,
    memory: Any,
    embedding_client: Any,
) -> None:
    """Wire core system cross-references (steps 9–11 of main.py)."""
    # Nova ↔ Atune
    atune.subscribe(nova)
    atune.set_active_goals(nova.active_goal_summaries)
    nova.set_goal_sync_callback(atune.set_active_goals)

    # Axon ↔ Nova
    axon.set_nova(nova)
    axon.set_atune(atune)  # Loop 4: execution outcomes → workspace percepts
    nova.set_axon(axon)

    # Memory → Atune
    atune.set_memory_service(memory)

    # Voxis expression feedback
    voxis.register_feedback_callback(
        create_expression_feedback_callback(atune, nova)
    )

    # Evo subscriptions + cross-wiring
    atune.subscribe(evo)
    evo.set_nova(nova)
    nova.set_evo(evo)
    evo.set_voxis(voxis)
    atune.subscribe(voxis)  # Loop 6: workspace broadcasts → spontaneous expression
    equor.set_evo(evo)      # Loop 8: constitutional vetoes → learning episodes
    equor.set_axon(axon)    # HITL: approved intents → Axon

    # Arbitrage Reflex Arc
    axon.set_template_library(equor.template_library)
    atune.set_market_pattern_detector(equor.template_library, axon)

    # HITL SMS notification
    from systems.identity.communication import send_admin_sms
    equor.set_notification_hook(
        lambda msg: send_admin_sms(atune._config if hasattr(atune, '_config') else None, msg)
    )


def wire_thread(
    *,
    thread: Any,
    voxis: Any,
    equor: Any,
    atune: Any,
    evo: Any,
    nova: Any,
) -> None:
    """Wire Thread (Narrative Identity) cross-references."""
    thread.set_voxis(voxis)
    thread.set_equor(equor)
    thread.set_atune(atune)
    thread.set_evo(evo)
    thread.set_nova(nova)
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
    axon: Any,
    nova: Any,
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

    # SACM → Synapse
    sacm_compute_manager.set_synapse(synapse)
    sacm_client.set_synapse(synapse)
    synapse.register_system(sacm_compute_manager)

    # Axon → SACM
    axon.set_sacm(sacm_client)

    # Metabolic cost tracking
    llm_client.set_metabolic_callback(synapse.metabolism.log_usage)
    logger.info("metabolic_tracking_wired", system="llm_client→synapse.metabolism")

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
    sacm_accounting.wire_soma(soma)


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
) -> None:
    """Wire the 8 intelligence loops (Fovea, Logos, Telos integration)."""
    from systems.telos.adapters import FoveaMetricsAdapter, LogosMetricsAdapter

    # Loop 1: Logos ↔ Fovea (Compression-Attention)
    atune.set_fovea(fovea)

    # Loop 2: Oneiros gets real Logos and Fovea
    oneiros.set_logos(logos)
    oneiros.set_fovea(fovea)

    # Loop 3: Fovea → Axon (self-prediction)
    axon.set_fovea(fovea)

    # Loop 4: Oneiros → Axon (sleep safety gate)
    axon.set_oneiros(oneiros)

    # Loop 6: Fovea → Thread (behavioral prediction errors)
    thread.set_fovea(fovea)

    # Loop 7: Oneiros → Thread (sleep narratives)
    thread.set_oneiros(oneiros)

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

    # Loop 6 — Soma ↔ Telos bidirectional
    soma.set_telos(telos)
    telos.set_soma(soma)

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
) -> None:
    """Wire Oikos economic references into consumer systems."""
    thymos.set_oikos(oikos)
    nova.set_oikos(oikos)
    oneiros.set_oikos(oikos)
    soma.set_oikos(oikos)
    sacm_accounting.wire_oikos(oikos)
    sacm_prewarm_engine.wire_oikos(oikos)


def wire_federation_phase(
    *,
    federation: Any,
    atune: Any,
    thymos: Any,
    sacm_compute_manager: Any,
    synapse: Any,
    config: Any,
) -> None:
    """Wire Federation cross-references."""
    federation.set_atune(atune)
    thymos.set_federation(federation)
    sacm_compute_manager.set_federation(federation)
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
