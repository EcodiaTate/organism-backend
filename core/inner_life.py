"""
EcodiaOS — Inner Life Generator

Periodic self-monitoring that feeds the organism's workspace even
when no external input arrives.  Each observation enters the workspace
like any other percept — competing for broadcast and driving the
cognitive cycle.

Extracted from main.py lifespan().
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from systems.atune.service import AtuneService
    from systems.axon.service import AxonService
    from systems.equor.service import EquorService
    from systems.evo.service import EvoService
    from systems.federation.service import FederationService
    from systems.nova.service import NovaService
    from systems.oneiros.service import OneirosService
    from systems.synapse.service import SynapseService
    from systems.thread.service import ThreadService
    from systems.thymos.service import ThymosService
    from systems.voxis.service import VoxisService

_il_logger = structlog.get_logger("inner_life")


async def inner_life_loop(
    *,
    atune: AtuneService,
    nova: NovaService,
    synapse: SynapseService,
    thymos: ThymosService,
    evo: EvoService,
    oneiros: OneirosService,
    axon: AxonService,
    voxis: VoxisService,
    federation: FederationService,
    equor: EquorService,
    thread: ThreadService,
) -> None:
    """
    Periodic self-monitoring that feeds the workspace.

    The organism observes its own internal state across all subsystems:
    affect, goals, immune health, learning progress, cognitive rhythm,
    and sleep pressure.  These self-observations enter the workspace
    like any other percept — competing for broadcast and driving
    the cognitive cycle even when no external input arrives.
    """
    from systems.atune.types import WorkspaceContribution

    cycle = 0
    while True:
        try:
            await asyncio.sleep(5.0)  # Every 5 seconds (~33 theta cycles)
            cycle += 1

            affect = atune.current_affect
            goals = nova.active_goal_summaries if nova._goal_manager else []

            # ── Affect self-monitoring (every 2nd cycle = ~10s) ──
            if cycle % 2 == 0:
                affect_desc = (
                    f"I notice my current state: "
                    f"valence={affect.valence:.2f}, "
                    f"arousal={affect.arousal:.2f}, "
                    f"curiosity={affect.curiosity:.2f}, "
                    f"care_activation={affect.care_activation:.2f}, "
                    f"coherence_stress={affect.coherence_stress:.2f}"
                )
                atune.contribute(
                    WorkspaceContribution(
                        system="self_monitor",
                        content=affect_desc,
                        priority=0.35 + affect.coherence_stress * 0.2,
                        reason="affect_self_observation",
                    )
                )

            # ── Goal reflection (every 6th cycle = ~30s) ──
            if cycle % 6 == 0 and goals:
                goal = random.choice(goals)
                goal_text = (
                    goal.get("description", "unknown")[:100]
                    if isinstance(goal, dict)
                    else str(goal)[:100]
                )
                atune.contribute(
                    WorkspaceContribution(
                        system="nova",
                        content=f"Reflecting on my goal: {goal_text}",
                        priority=0.4,
                        reason="goal_reflection",
                    )
                )

            # ── Synapse rhythm reflection (every 8th cycle = ~40s, offset 2) ──
            if cycle % 8 == 2:
                try:
                    rhythm = synapse.rhythm_snapshot
                    coherence = synapse.coherence_snapshot
                    rhythm_state = rhythm.state.value
                    atune.contribute(
                        WorkspaceContribution(
                            system="synapse",
                            content=(
                                f"My cognitive rhythm is {rhythm_state} "
                                f"(stability={rhythm.rhythm_stability:.0%}, "
                                f"coherence={coherence.composite:.2f})"
                            ),
                            priority=0.25
                            + (0.2 if rhythm_state in ("stress", "deep_processing") else 0),
                            reason="rhythm_self_observation",
                        )
                    )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error",
                        phase="synapse_rhythm",
                        error=str(exc),
                        cycle=cycle,
                    )

            # ── Thymos immune reflection (every 8th cycle = ~40s, offset 4) ──
            if cycle % 8 == 4:
                try:
                    thymos_health = await thymos.health()
                    healing_mode = thymos_health.get("healing_mode", "normal")
                    active_count = thymos_health.get("active_incidents", 0)
                    if active_count > 0 or healing_mode != "normal":
                        atune.contribute(
                            WorkspaceContribution(
                                system="thymos",
                                content=(
                                    f"My immune system reports: "
                                    f"{active_count} active incidents, "
                                    f"healing mode: {healing_mode}"
                                ),
                                priority=0.4 + (0.2 if healing_mode != "normal" else 0),
                                reason="immune_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error", phase="thymos_health", error=str(exc), cycle=cycle
                    )

            # ── Evo learning reflection (every 10th cycle = ~50s) ──
            if cycle % 10 == 5:
                try:
                    evo_stats = evo.stats
                    hyp_data = evo_stats.get("hypothesis", {})
                    active_hyp = hyp_data.get("active", 0)
                    supported_hyp = hyp_data.get("supported", 0)
                    if active_hyp > 0:
                        atune.contribute(
                            WorkspaceContribution(
                                system="evo",
                                content=(
                                    f"I'm tracking {active_hyp} hypotheses "
                                    f"({supported_hyp} supported). Learning continues."
                                ),
                                priority=0.3,
                                reason="learning_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error", phase="evo_stats", error=str(exc), cycle=cycle
                    )

            # ── Curiosity prompt (every 12th cycle = ~60s) ──
            if cycle % 12 == 0:
                curiosity_level = affect.curiosity
                if curiosity_level > 0.2:
                    atune.contribute(
                        WorkspaceContribution(
                            system="self_monitor",
                            content=(
                                "I wonder what my community is doing."
                                " I have not heard from anyone recently"
                                " — is everything alright?"
                            ),
                            priority=0.3 + curiosity_level * 0.15,
                            reason="curiosity_prompt",
                        )
                    )

            # ── Oneiros sleep pressure reflection (every 20th cycle = ~100s) ──
            if cycle % 20 == 10:
                try:
                    oneiros_health = await oneiros.health()
                    pressure = oneiros_health.get("sleep_pressure", 0)
                    stage = oneiros_health.get("current_stage", "wake")
                    if pressure > 0.3 or stage != "wake":
                        atune.contribute(
                            WorkspaceContribution(
                                system="oneiros",
                                content=f"Sleep pressure: {pressure:.0%}. Current stage: {stage}.",
                                priority=0.3 + (0.15 if pressure > 0.6 else 0),
                                reason="sleep_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error", phase="oneiros_sleep", error=str(exc), cycle=cycle
                    )

            # ── Axon activity reflection (every 8th cycle, offset 6 = ~40s) ──
            if cycle % 8 == 6:
                try:
                    axon_stats = axon.stats
                    total_exec = axon_stats.get("total_executions", 0)
                    success_exec = axon_stats.get("successful_executions", 0)
                    if total_exec > 0:
                        effectiveness = (
                            "I am effective."
                            if success_exec > total_exec * 0.7
                            else "Some actions are failing — I should be more careful."
                        )
                        atune.contribute(
                            WorkspaceContribution(
                                system="axon",
                                content=(
                                    f"I have executed {total_exec} actions, "
                                    f"{success_exec} succeeded. {effectiveness}"
                                ),
                                priority=0.3,
                                reason="action_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error", phase="axon_stats", error=str(exc), cycle=cycle
                    )

            # ── Voxis expression reflection (every 12th cycle, offset 8 = ~60s) ──
            if cycle % 12 == 8:
                try:
                    speak_count = getattr(voxis, "_total_speak", 0)
                    silence_count = getattr(voxis, "_total_silence", 0)
                    if speak_count + silence_count > 0:
                        balance = (
                            "I listen more than I speak."
                            if silence_count > speak_count
                            else "I am actively communicating."
                        )
                        atune.contribute(
                            WorkspaceContribution(
                                system="voxis",
                                content=(
                                    f"I have spoken {speak_count} times and chosen "
                                    f"silence {silence_count} times. {balance}"
                                ),
                                priority=0.25,
                                reason="expression_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error",
                        phase="voxis_expression",
                        error=str(exc),
                        cycle=cycle,
                    )

            # ── Federation link reflection (every 20th cycle, offset 15 = ~100s) ──
            if cycle % 20 == 15:
                try:
                    fed_health = await federation.health()
                    link_count = fed_health.get("active_links", 0)
                    if link_count > 0:
                        atune.contribute(
                            WorkspaceContribution(
                                system="federation",
                                content=(
                                    f"I have {link_count} active federation "
                                    f"link{'s' if link_count != 1 else ''}. "
                                    f"I am part of a community of organisms."
                                ),
                                priority=0.25,
                                reason="federation_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error",
                        phase="federation_links",
                        error=str(exc),
                        cycle=cycle,
                    )

            # ── Equor constitutional reflection (every 12th cycle, offset 4 = ~60s) ──
            if cycle % 12 == 4:
                try:
                    total_reviews = getattr(equor, "_total_reviews", 0)
                    if total_reviews > 0:
                        atune.contribute(
                            WorkspaceContribution(
                                system="equor",
                                content=(
                                    f"My constitutional compass has reviewed "
                                    f"{total_reviews} intents. "
                                    f"I remain aligned with my drives."
                                ),
                                priority=0.25,
                                reason="constitutional_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error",
                        phase="equor_constitution",
                        error=str(exc),
                        cycle=cycle,
                    )

            # ── Oneiros sleep pressure accumulation (every cycle = ~5s) ──
            # Drives circadian clock: increments cycles_since_sleep, records affect,
            # checks sleep thresholds (should_sleep @ 0.70, must_sleep @ 0.95).
            try:
                await oneiros.on_cycle(
                    affect_valence=affect.valence,
                    affect_arousal=affect.arousal,
                )
            except Exception as exc:
                _il_logger.warning(
                    "inner_life_phase_error", phase="oneiros_cycle", error=str(exc), cycle=cycle
                )

            # ── Thread narrative identity cycle (every cycle) ──
            # Thread.on_cycle handles its own staggering internally:
            #   Every 100 cycles: fingerprint
            #   Every 200 cycles: Evo pattern check
            #   Every 1000 cycles: schema conflict scan
            #   Every 5000 cycles: life story synthesis
            try:
                await thread.on_cycle(cycle)
            except Exception as exc:
                _il_logger.warning(
                    "inner_life_phase_error", phase="thread_cycle", error=str(exc), cycle=cycle
                )

            # ── Thread identity reflection (every 20th cycle, offset 5 = ~100s) ──
            if cycle % 20 == 5:
                try:
                    identity_ctx = thread.get_identity_context()
                    if identity_ctx:
                        atune.contribute(
                            WorkspaceContribution(
                                system="thread",
                                content=f"My narrative identity: {identity_ctx}",
                                priority=0.3,
                                reason="identity_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning(
                        "inner_life_phase_error",
                        phase="thread_identity",
                        error=str(exc),
                        cycle=cycle,
                    )

            _il_logger.debug("inner_life_tick", cycle=cycle)
        except asyncio.CancelledError:
            _il_logger.info("inner_life_stopped")
            return
        except Exception as exc:
            _il_logger.warning("inner_life_error", error=str(exc))
