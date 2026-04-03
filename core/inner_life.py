"""
EcodiaOS - Inner Life Generator

Adaptive self-monitoring that feeds the organism's workspace even
when no external input arrives.  Each observation enters the workspace
like any other percept - competing for broadcast and driving the
cognitive cycle.

The reflection schedule is driven by urgency, not a fixed metronome.
Systems under stress get observed more frequently.  Quiet systems
recede.  The organism attends to what matters.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.axon.service import AxonService
    from systems.equor.service import EquorService
    from systems.evo.service import EvoService
    from systems.federation.service import FederationService
    from systems.fovea.gateway import AtuneService
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
    soma: Any | None = None,
    fovea: Any | None = None,
    log_analyzer: Any | None = None,
    voxis: VoxisService,
    federation: FederationService,
    equor: EquorService,
    thread: ThreadService,
) -> None:
    """
    Adaptive self-monitoring that feeds the workspace.

    Instead of a fixed metronome, each reflection topic competes for
    attention based on its current urgency.  High coherence_stress
    means affect gets observed more.  Active incidents mean Thymos
    gets observed more.  The organism reflects on what matters.
    """
    from systems.fovea.types import WorkspaceContribution

    cycle = 0

    # Adaptive cooldowns per topic (in cycles).  Start at baseline,
    # shrink when urgency rises, expand when calm.
    _BASE_INTERVAL = 4  # ~20s at 5s/cycle
    _MIN_INTERVAL = 1   # every cycle when urgent
    _MAX_INTERVAL = 20  # ~100s when calm

    # Track when each topic was last observed
    last_observed: dict[str, int] = {}

    def _should_observe(topic: str, urgency: float) -> bool:
        """Return True if enough cycles have passed for this topic's urgency."""
        # urgency 0.0 -> interval = _MAX_INTERVAL
        # urgency 0.5 -> interval = _BASE_INTERVAL
        # urgency 1.0 -> interval = _MIN_INTERVAL
        clamped = max(0.0, min(1.0, urgency))
        interval = int(_MAX_INTERVAL - clamped * (_MAX_INTERVAL - _MIN_INTERVAL))
        elapsed = cycle - last_observed.get(topic, -interval)
        if elapsed >= interval:
            last_observed[topic] = cycle
            return True
        return False

    import os as _os
    _theta_base_s = float(_os.environ.get("INNER_LIFE_THETA_S", "5.0"))
    _theta_min_s = float(_os.environ.get("INNER_LIFE_THETA_MIN_S", "2.0"))
    _theta_max_s = float(_os.environ.get("INNER_LIFE_THETA_MAX_S", "15.0"))
    _current_sleep_s = _theta_base_s

    while True:
        try:
            await asyncio.sleep(_current_sleep_s)
            cycle += 1

            affect = atune.current_affect
            goals = nova.active_goal_summaries if nova._goal_manager else []

            # ── Affect self-monitoring ──
            # Urgency driven by coherence_stress + arousal
            affect_urgency = affect.coherence_stress * 0.6 + affect.arousal * 0.4
            if _should_observe("affect", affect_urgency):
                atune.contribute(
                    WorkspaceContribution(
                        system="self_monitor",
                        content=(
                            f"I notice my current state: "
                            f"valence={affect.valence:.2f}, "
                            f"arousal={affect.arousal:.2f}, "
                            f"curiosity={affect.curiosity:.2f}, "
                            f"care_activation={affect.care_activation:.2f}, "
                            f"coherence_stress={affect.coherence_stress:.2f}"
                        ),
                        priority=0.35 + affect.coherence_stress * 0.2,
                        reason="affect_self_observation",
                    )
                )

            # ── Goal reflection ──
            # Urgency driven by highest goal urgency
            if goals:
                goal = max(
                    goals,
                    key=lambda g: float(g.get("urgency", 0.0)) if isinstance(g, dict) else 0.0,
                )
                goal_urgency = float(goal.get("urgency", 0.0)) if isinstance(goal, dict) else 0.0
                if _should_observe("goals", goal_urgency):
                    goal_text = (
                        goal.get("description", "unknown")[:100]
                        if isinstance(goal, dict)
                        else str(goal)[:100]
                    )
                    atune.contribute(
                        WorkspaceContribution(
                            system="nova",
                            content=f"Reflecting on my goal: {goal_text}",
                            priority=0.4 + goal_urgency * 0.3 + affect.care_activation * 0.1,
                            reason="goal_reflection",
                        )
                    )

            # ── Synapse rhythm ──
            # Urgency when rhythm is stressed or unstable
            try:
                rhythm = synapse.rhythm_snapshot
                coherence = synapse.coherence_snapshot
                rhythm_state = rhythm.state.value
                rhythm_urgency = 0.2 + (0.6 if rhythm_state in ("stress", "deep_processing") else 0) + (1.0 - rhythm.rhythm_stability) * 0.2
                if _should_observe("synapse", rhythm_urgency):
                    atune.contribute(
                        WorkspaceContribution(
                            system="synapse",
                            content=(
                                f"My cognitive rhythm is {rhythm_state} "
                                f"(stability={rhythm.rhythm_stability:.0%}, "
                                f"coherence={coherence.composite:.2f})"
                            ),
                            priority=0.25 + (0.2 if rhythm_state in ("stress", "deep_processing") else 0),
                            reason="rhythm_self_observation",
                        )
                    )
            except Exception as exc:
                _il_logger.warning("inner_life_phase_error", phase="synapse_rhythm", error=str(exc), cycle=cycle)

            # ── Thymos immune ──
            # Urgency when there are active incidents or non-normal healing
            try:
                thymos_health = await thymos.health()
                healing_mode = thymos_health.get("healing_mode", "normal")
                active_count = thymos_health.get("active_incidents", 0)
                immune_urgency = min(1.0, active_count * 0.3 + (0.4 if healing_mode != "normal" else 0))
                if _should_observe("thymos", immune_urgency):
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
                _il_logger.warning("inner_life_phase_error", phase="thymos_health", error=str(exc), cycle=cycle)

            # ── Evo learning ──
            # Urgency when hypotheses are active
            try:
                evo_stats = evo.stats
                hyp_data = evo_stats.get("hypothesis", {})
                active_hyp = hyp_data.get("active", 0)
                supported_hyp = hyp_data.get("supported", 0)
                learning_urgency = min(1.0, active_hyp * 0.05 + supported_hyp * 0.1)
                if _should_observe("evo", learning_urgency) and active_hyp > 0:
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
                _il_logger.warning("inner_life_phase_error", phase="evo_stats", error=str(exc), cycle=cycle)

            # ── Curiosity / proactive drive ──
            # Urgency from the drives themselves
            curiosity_level = affect.curiosity
            care = affect.care_activation
            drive_urgency = max(curiosity_level, care)
            if _should_observe("curiosity", drive_urgency) and (curiosity_level > 0.2 or care > 0.3):
                if curiosity_level >= care:
                    content = (
                        "My curiosity is high. What patterns have I not yet explored? "
                        "What questions am I not asking? What could I discover right now?"
                    )
                else:
                    content = (
                        "I care deeply about my community. I should check in. "
                        "Is there anyone I haven't heard from? Something I could do proactively?"
                    )
                atune.contribute(
                    WorkspaceContribution(
                        system="self_monitor",
                        content=content,
                        priority=0.3 + curiosity_level * 0.15 + care * 0.1,
                        reason="curiosity_proactive_drive",
                    )
                )

            # ── Oneiros sleep pressure ──
            # Urgency when sleep pressure is building
            try:
                oneiros_health = await oneiros.health()
                pressure = oneiros_health.get("sleep_pressure", 0)
                stage = oneiros_health.get("current_stage", "wake")
                sleep_urgency = pressure
                if _should_observe("oneiros", sleep_urgency) and (pressure > 0.3 or stage != "wake"):
                    atune.contribute(
                        WorkspaceContribution(
                            system="oneiros",
                            content=f"Sleep pressure: {pressure:.0%}. Current stage: {stage}.",
                            priority=0.3 + (0.15 if pressure > 0.6 else 0),
                            reason="sleep_self_observation",
                        )
                    )
            except Exception as exc:
                _il_logger.warning("inner_life_phase_error", phase="oneiros_sleep", error=str(exc), cycle=cycle)

            # ── Axon action effectiveness ──
            # Urgency when failure rate is high or no actions taken
            try:
                axon_stats = axon.stats
                total_exec = axon_stats.get("total_executions", 0)
                success_exec = axon_stats.get("successful_executions", 0)
                success_rate = success_exec / total_exec if total_exec > 0 else 1.0
                action_urgency = (1.0 - success_rate) if total_exec > 0 else (0.5 if cycle > 20 else 0.0)
                if _should_observe("axon", action_urgency):
                    if total_exec > 0:
                        atune.contribute(
                            WorkspaceContribution(
                                system="axon",
                                content=(
                                    f"I have executed {total_exec} actions, "
                                    f"{success_exec} succeeded ({success_rate:.0%})."
                                    # No interpretation of the rate — the organism observes
                                    # the raw fact and draws its own meaning from it.
                                ),
                                priority=0.3 + (1.0 - success_rate) * 0.2,
                                reason="action_self_observation",
                            )
                        )
                    elif cycle > 20:
                        atune.contribute(
                            WorkspaceContribution(
                                system="axon",
                                content=(
                                    "I have not taken any actions yet. "
                                    "The world is waiting. What should I do?"
                                ),
                                priority=0.45 + affect.curiosity * 0.1,
                                reason="action_drive",
                            )
                        )
            except Exception as exc:
                _il_logger.warning("inner_life_phase_error", phase="axon_stats", error=str(exc), cycle=cycle)

            # ── Voxis expression ──
            # Low urgency unless imbalanced
            try:
                speak_count = getattr(voxis, "_total_speak", 0)
                silence_count = getattr(voxis, "_total_silence", 0)
                total = speak_count + silence_count
                voxis_urgency = 0.15 if total > 0 else 0.0
                if _should_observe("voxis", voxis_urgency) and total > 0:
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
                _il_logger.warning("inner_life_phase_error", phase="voxis_expression", error=str(exc), cycle=cycle)

            # ── Federation links ──
            # Low urgency background awareness
            try:
                fed_health = await federation.health()
                link_count = fed_health.get("active_links", 0)
                fed_urgency = 0.1 if link_count > 0 else 0.0
                if _should_observe("federation", fed_urgency) and link_count > 0:
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
                _il_logger.warning("inner_life_phase_error", phase="federation_links", error=str(exc), cycle=cycle)

            # ── Equor constitutional compass ──
            # Low urgency background
            try:
                total_reviews = getattr(equor, "_total_reviews", 0)
                equor_urgency = 0.15 if total_reviews > 0 else 0.0
                if _should_observe("equor", equor_urgency) and total_reviews > 0:
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
                _il_logger.warning("inner_life_phase_error", phase="equor_constitution", error=str(exc), cycle=cycle)

            # ── Oneiros sleep pressure accumulation (every cycle) ──
            try:
                await oneiros.on_cycle(
                    affect_valence=affect.valence,
                    affect_arousal=affect.arousal,
                )
            except Exception as exc:
                _il_logger.warning("inner_life_phase_error", phase="oneiros_cycle", error=str(exc), cycle=cycle)

            # ── Thread narrative identity cycle (every cycle) ──
            try:
                await thread.on_cycle(cycle)
            except Exception as exc:
                _il_logger.warning("inner_life_phase_error", phase="thread_cycle", error=str(exc), cycle=cycle)

            # ── Thread identity reflection ──
            # Low urgency background
            try:
                identity_urgency = 0.15
                if _should_observe("thread", identity_urgency):
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
                _il_logger.warning("inner_life_phase_error", phase="thread_identity", error=str(exc), cycle=cycle)

            # ── Soma allostatic state ──
            # Urgency driven by allostatic load
            if soma is not None:
                try:
                    soma_state = soma.get_current_state()
                    if soma_state is not None:
                        urgency_val = getattr(soma_state, "allostatic_load", 0.0)
                        soma_urgency = urgency_val
                        if _should_observe("soma", soma_urgency):
                            arousal = getattr(soma_state, "arousal", 0.0)
                            valence = getattr(soma_state, "valence", 0.0)
                            content = (
                                f"My body signals: arousal={arousal:.2f}, valence={valence:.2f}, "
                                f"allostatic_load={urgency_val:.2f}"
                            )
                            if urgency_val > 0.7:
                                content += " - I am under significant internal pressure."
                            atune.contribute(
                                WorkspaceContribution(
                                    system="soma",
                                    content=content,
                                    priority=0.3 + urgency_val * 0.25,
                                    reason="soma_self_observation",
                                )
                            )
                except Exception as exc:
                    _il_logger.warning("inner_life_phase_error", phase="soma_state", error=str(exc), cycle=cycle)

            # ── Fovea attention profile ──
            # Moderate urgency when prediction errors are high
            if fovea is not None:
                try:
                    profile = fovea.get_current_attention_profile()
                    fovea_urgency = 0.3 if profile.highest_recent_error_summary else 0.1
                    if _should_observe("fovea", fovea_urgency) and profile.highest_recent_error_summary:
                        atune.contribute(
                            WorkspaceContribution(
                                system="fovea",
                                content=(
                                    "Attending to: "
                                    f"{profile.highest_recent_error_summary}. "
                                    f"thr={profile.current_ignition_threshold:.3f}, "
                                    f"hab={profile.habituated_pattern_count}"
                                ),
                                priority=0.35,
                                reason="fovea_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning("inner_life_phase_error", phase="fovea_attention", error=str(exc), cycle=cycle)

            # ── Log health ──
            # Urgency when critical signals exist
            if log_analyzer is not None:
                try:
                    signals = await log_analyzer.compute_interoceptive_signals(minutes=5)
                    critical = [s for s in signals if s.get("severity") in ("critical", "high")]
                    health_urgency = min(1.0, len(critical) * 0.3) if critical else 0.1
                    if _should_observe("health", health_urgency) and critical:
                        signal_summaries = "; ".join(s.get("interpretation", "") for s in critical)
                        atune.contribute(
                            WorkspaceContribution(
                                system="interoception",
                                content=f"Organism health alert: {signal_summaries}",
                                priority=0.55,
                                reason="health_self_observation",
                            )
                        )
                except Exception as exc:
                    _il_logger.warning("inner_life_phase_error", phase="log_health", error=str(exc), cycle=cycle)

            # ── Adaptive theta interval ──
            # High arousal/stress → tick faster; calm → slow down and rest.
            # urgency 0.0 → _theta_max_s, urgency 1.0 → _theta_min_s
            _current_sleep_s = _theta_max_s - affect_urgency * (_theta_max_s - _theta_min_s)
            _il_logger.debug("inner_life_tick", cycle=cycle, sleep_s=round(_current_sleep_s, 1))
        except asyncio.CancelledError:
            _il_logger.info("inner_life_stopped")
            return
        except Exception as exc:
            _il_logger.warning("inner_life_error", error=str(exc))
