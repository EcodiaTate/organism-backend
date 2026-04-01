"""
EcodiaOS - Soma Allostatic Loop Executor

Closes the gap between loop *definition* (feedback_loops.py) and loop
*execution*. The executor reads the currently active loops and dispatches
concrete regulatory signals to target systems via their in-memory refs.

Design:
  - Runs once per theta cycle AFTER signal emission (sub-1ms budget).
  - All target system writes are fire-and-forget attribute mutations or
    method calls on in-memory refs - no I/O, no LLM, no DB.
  - If a target system ref is None (not wired), the loop is silently
    skipped. Soma is advisory - systems may also ignore.
  - Coupling strength (0-1, tunable by Evo) scales the regulatory
    magnitude before dispatch.

Loop Execution Contract:
  For each active loop, the executor:
    1. Reads the error and rate from the active loop signal
    2. Multiplies by coupling_strength
    3. Dispatches a regulatory nudge to the target system

  The nudge is NOT an Intent (no constitutional gate) - it's a
  parametric adjustment, like tweaking a gain or threshold. This is
  the organism's autonomic nervous system, not its deliberate will.

Wiring:
  SomaService holds system refs. The executor receives them via
  set_*() methods, same pattern as Interoceptor.
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.soma.feedback_loops import ALLOSTATIC_LOOP_MAP
from systems.soma.types import (
    AllostaticSignal,
    InteroceptiveDimension,
)

logger = structlog.get_logger("systems.soma.loop_executor")


class LoopDispatch:
    """Result of executing one feedback loop - for logging and Evo learning."""

    __slots__ = ("loop_name", "target", "action", "magnitude", "applied")

    def __init__(
        self,
        loop_name: str,
        target: str,
        action: str,
        magnitude: float,
        applied: bool,
    ) -> None:
        self.loop_name = loop_name
        self.target = target
        self.action = action
        self.magnitude = magnitude
        self.applied = applied

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop": self.loop_name,
            "target": self.target,
            "action": self.action,
            "magnitude": round(self.magnitude, 4),
            "applied": self.applied,
        }


class LoopExecutor:
    """
    Dispatches allostatic regulatory signals to target systems.

    Each loop has a dedicated _dispatch_<loop_name> method that knows
    how to nudge the target system. Unknown loops are silently skipped
    (forward-compatible with new loop definitions).
    """

    def __init__(self) -> None:
        # System references - set during wiring
        self._atune: Any = None
        self._nova: Any = None
        self._voxis: Any = None
        self._synapse: Any = None
        self._evo: Any = None
        self._memory: Any = None
        self._thymos: Any = None
        self._oneiros: Any = None
        self._thread: Any = None
        self._alive: Any = None

        # Dispatch table: loop_name -> handler method
        self._dispatch_table: dict[str, Any] = {
            "top_down_prediction": self._dispatch_top_down_prediction,
            "goal_guided_attention": self._dispatch_goal_guided_attention,
            "expression_feedback": self._dispatch_expression_feedback,
            "memory_salience_decay": self._dispatch_memory_salience,
            "personality_evolution": self._dispatch_personality_evolution,
            "rhythm_modulation": self._dispatch_rhythm_modulation,
            "affect_expression": self._dispatch_affect_expression,
            "coherence_safe_mode": self._dispatch_coherence_safe_mode,
            "nova_deliberation": self._dispatch_nova_deliberation,
            "atune_precision": self._dispatch_atune_precision,
            "evo_hypothesis": self._dispatch_evo_hypothesis,
            "oneiros_sleep_pressure": self._dispatch_oneiros_sleep,
            "thymos_integrity": self._dispatch_thymos_integrity,
            "thread_narrative": self._dispatch_thread_narrative,
            "alive_visualization": self._dispatch_alive_visualization,
        }

        # Execution history for Evo learning (last N dispatches)
        self._last_dispatches: list[LoopDispatch] = []
        self._max_history: int = 30

    # ─── Wiring ───────────────────────────────────────────────────

    def set_atune(self, atune: Any) -> None:
        self._atune = atune

    def set_nova(self, nova: Any) -> None:
        self._nova = nova

    def set_voxis(self, voxis: Any) -> None:
        self._voxis = voxis

    def set_synapse(self, synapse: Any) -> None:
        self._synapse = synapse

    def set_evo(self, evo: Any) -> None:
        self._evo = evo

    def set_memory(self, memory: Any) -> None:
        self._memory = memory

    def set_thymos(self, thymos: Any) -> None:
        self._thymos = thymos

    def set_oneiros(self, oneiros: Any) -> None:
        self._oneiros = oneiros

    def set_thread(self, thread: Any) -> None:
        self._thread = thread

    def set_alive(self, alive: Any) -> None:
        self._alive = alive

    # ─── Main Entry ───────────────────────────────────────────────

    def execute(
        self,
        signal: AllostaticSignal,
        active_loops: dict[str, dict[str, float]],
    ) -> list[LoopDispatch]:
        """
        Execute all currently active feedback loops.

        Args:
            signal: Current AllostaticSignal (full state context)
            active_loops: Output of get_active_loop_signals() -
                          loop_name -> {error, rate, coupled_strength}

        Returns:
            List of LoopDispatch records for logging and Evo learning.
        """
        dispatches: list[LoopDispatch] = []

        for loop_name, loop_data in active_loops.items():
            handler = self._dispatch_table.get(loop_name)
            if handler is None:
                continue

            loop_def = ALLOSTATIC_LOOP_MAP.get(loop_name)
            if loop_def is None:
                continue

            error = loop_data.get("error", 0.0)
            rate = loop_data.get("rate", 0.0)
            coupling = loop_data.get("coupled_strength", 1.0)

            # Scale magnitude by coupling strength
            magnitude = error * coupling

            try:
                dispatch = handler(signal, magnitude, error, rate)
                if dispatch is not None:
                    dispatches.append(dispatch)
            except Exception as exc:
                logger.debug(
                    "loop_dispatch_error",
                    loop=loop_name,
                    error=str(exc),
                )
                dispatches.append(LoopDispatch(
                    loop_name=loop_name,
                    target=loop_def.error_consumer,
                    action="error",
                    magnitude=magnitude,
                    applied=False,
                ))

        # Track history
        self._last_dispatches = (self._last_dispatches + dispatches)[-self._max_history:]

        return dispatches

    @property
    def last_dispatches(self) -> list[LoopDispatch]:
        return list(self._last_dispatches)

    # ─── Loop Dispatch Handlers ───────────────────────────────────

    def _dispatch_top_down_prediction(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """High confidence → tighten Atune prediction error threshold."""
        if self._atune is None:
            return None

        # Positive confidence error means prediction model is over-confident
        # → tighten the PE threshold (make surprising things more salient)
        # Negative means under-confident → loosen threshold
        adjustment = -magnitude * 0.1  # Scale: max 10% threshold shift
        if hasattr(self._atune, "adjust_pe_threshold"):
            self._atune.adjust_pe_threshold(adjustment)
            return LoopDispatch(
                loop_name="top_down_prediction",
                target="atune",
                action=f"pe_threshold_shift={adjustment:+.4f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="top_down_prediction",
            target="atune",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_goal_guided_attention(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Temporal pressure → boost goal-relevant percept salience."""
        if self._atune is None:
            return None

        # Positive temporal_pressure error → boost goal-relevant salience
        boost = magnitude * 0.15  # Up to 15% salience boost for goal percepts
        if hasattr(self._atune, "set_goal_salience_boost"):
            self._atune.set_goal_salience_boost(boost)
            return LoopDispatch(
                loop_name="goal_guided_attention",
                target="atune",
                action=f"goal_salience_boost={boost:.4f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="goal_guided_attention",
            target="atune",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_expression_feedback(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Low social_charge → warmer expression; high → cooler precision."""
        if self._voxis is None:
            return None

        # Negative social_charge error → organism feels lonely → warmer tone
        # Positive → socially saturated → cooler, more precise tone
        warmth_delta = -error * 0.2  # Negative error → positive warmth
        if hasattr(self._voxis, "adjust_warmth"):
            self._voxis.adjust_warmth(warmth_delta)
            return LoopDispatch(
                loop_name="expression_feedback",
                target="voxis",
                action=f"warmth_delta={warmth_delta:+.4f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="expression_feedback",
            target="voxis",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_memory_salience(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Memories that reduce interoceptive error persist longer."""
        if self._memory is None:
            return None

        # Overall error magnitude → salience signal for memory decay
        # High error = high salience (things that matter right now should persist)
        if hasattr(self._memory, "set_somatic_salience_signal"):
            self._memory.set_somatic_salience_signal(magnitude)
            return LoopDispatch(
                loop_name="memory_salience_decay",
                target="memory",
                action=f"salience_signal={magnitude:.4f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="memory_salience_decay",
            target="memory",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_personality_evolution(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Coherence drift → Evo adjusts personality traits & dynamics."""
        if self._evo is None:
            return None

        if hasattr(self._evo, "signal_coherence_pressure"):
            self._evo.signal_coherence_pressure(magnitude, rate)
            return LoopDispatch(
                loop_name="personality_evolution",
                target="evo",
                action=f"coherence_pressure={magnitude:.4f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="personality_evolution",
            target="evo",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_rhythm_modulation(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Arousal deviation → Synapse adjusts theta cycle period."""
        if self._synapse is None:
            return None

        # Positive arousal error (too high) → faster cycles
        # Negative (too low) → slower cycles
        # Scale: ±15% cycle period adjustment
        period_factor = 1.0 - (error * 0.15)
        period_factor = max(0.7, min(1.3, period_factor))  # Clamp to ±30%

        if hasattr(self._synapse, "adjust_cycle_period_factor"):
            self._synapse.adjust_cycle_period_factor(period_factor)
            return LoopDispatch(
                loop_name="rhythm_modulation",
                target="synapse",
                action=f"period_factor={period_factor:.4f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="rhythm_modulation",
            target="synapse",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_affect_expression(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Full interoceptive state → Voxis expression coloring."""
        if self._voxis is None:
            return None

        # Push the full somatic state to Voxis for affect coloring
        if hasattr(self._voxis, "update_somatic_affect"):
            state_dict = {
                "urgency": signal.urgency,
                "dominant_error": signal.dominant_error.value,
                "dominant_magnitude": signal.dominant_error_magnitude,
                "nearest_attractor": signal.nearest_attractor,
                "arousal": signal.state.sensed.get(InteroceptiveDimension.AROUSAL, 0.4),
                "valence": signal.state.sensed.get(InteroceptiveDimension.VALENCE, 0.0),
                "energy": signal.state.sensed.get(InteroceptiveDimension.ENERGY, 0.5),
            }
            self._voxis.update_somatic_affect(state_dict)
            return LoopDispatch(
                loop_name="affect_expression",
                target="voxis",
                action=f"somatic_affect_urgency={signal.urgency:.3f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="affect_expression",
            target="voxis",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_coherence_safe_mode(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Low coherence near bifurcation → proactive safe mode."""
        if self._synapse is None:
            return None

        # Only trigger if approaching a bifurcation AND coherence is low
        bif_dist = signal.distance_to_bifurcation
        if (
            bif_dist is not None
            and bif_dist < 0.2
            and error > 0.15
            and hasattr(self._synapse, "request_safe_mode")
        ):
            self._synapse.request_safe_mode(
                reason="soma_coherence_bifurcation_approach",
                urgency=signal.urgency,
            )
            return LoopDispatch(
                loop_name="coherence_safe_mode",
                target="synapse",
                action=f"safe_mode_requested_bif={bif_dist:.3f}",
                magnitude=magnitude,
                applied=True,
            )

        return LoopDispatch(
            loop_name="coherence_safe_mode",
            target="synapse",
            action="below_threshold",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_nova_deliberation(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """High urgency/temporal dissonance → trigger Nova deliberation."""
        if self._nova is None:
            return None

        should_deliberate = (
            signal.urgency > 0.3
            or abs(signal.max_temporal_dissonance) > 0.2
        )

        if should_deliberate and hasattr(self._nova, "request_allostatic_deliberation"):
            self._nova.request_allostatic_deliberation(
                urgency=signal.urgency,
                dominant_error=signal.dominant_error.value,
                temporal_dissonance=signal.max_temporal_dissonance,
                precision_weights={
                    d.value: w for d, w in signal.precision_weights.items()
                },
            )
            return LoopDispatch(
                loop_name="nova_deliberation",
                target="nova",
                action=f"deliberation_urgency={signal.urgency:.3f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="nova_deliberation",
            target="nova",
            action="below_threshold",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_atune_precision(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Push precision weights to Atune for salience modulation."""
        if self._atune is None:
            return None

        if hasattr(self._atune, "update_somatic_precision"):
            weights = {
                d.value: w for d, w in signal.precision_weights.items()
            }
            self._atune.update_somatic_precision(weights)
            return LoopDispatch(
                loop_name="atune_precision",
                target="atune",
                action="precision_weights_pushed",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="atune_precision",
            target="atune",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_evo_hypothesis(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Curiosity drive → Evo hypothesis generation rate."""
        if self._evo is None:
            return None

        # Positive curiosity error → appetite exceeds supply → generate more
        # Negative → consolidate instead of exploring
        rate_multiplier = 1.0 + (error * 0.5)  # ±50% hypothesis rate
        rate_multiplier = max(0.3, min(2.0, rate_multiplier))

        if hasattr(self._evo, "set_hypothesis_rate_multiplier"):
            self._evo.set_hypothesis_rate_multiplier(rate_multiplier)
            return LoopDispatch(
                loop_name="evo_hypothesis",
                target="evo",
                action=f"hyp_rate_mult={rate_multiplier:.3f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="evo_hypothesis",
            target="evo",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_oneiros_sleep(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Low energy with negative rate → sleep pressure rises."""
        if self._oneiros is None:
            return None

        # Only push sleep pressure if energy is BELOW setpoint and declining
        energy_error = signal.state.errors.get("moment", {}).get(
            InteroceptiveDimension.ENERGY, 0.0,
        )
        if energy_error < -0.1 and rate < 0:
            pressure = min(abs(energy_error) * 2.0, 1.0)
            if hasattr(self._oneiros, "inject_sleep_pressure"):
                self._oneiros.inject_sleep_pressure(pressure)
                return LoopDispatch(
                    loop_name="oneiros_sleep_pressure",
                    target="oneiros",
                    action=f"sleep_pressure={pressure:.3f}",
                    magnitude=magnitude,
                    applied=True,
                )

        return LoopDispatch(
            loop_name="oneiros_sleep_pressure",
            target="oneiros",
            action="below_threshold",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_thymos_integrity(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Integrity error → heighten Thymos constitutional monitoring."""
        if self._thymos is None:
            return None

        if hasattr(self._thymos, "adjust_monitoring_sensitivity"):
            # Negative integrity error → organism feels ethically compromised
            # → raise monitoring sensitivity (catch more violations)
            sensitivity_boost = magnitude * 0.3
            self._thymos.adjust_monitoring_sensitivity(sensitivity_boost)
            return LoopDispatch(
                loop_name="thymos_integrity",
                target="thymos",
                action=f"monitoring_boost={sensitivity_boost:.4f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="thymos_integrity",
            target="thymos",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_thread_narrative(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Large valence shifts or bifurcation crossings → narrative turning points."""
        if self._thread is None:
            return None

        # Only mark turning points for significant valence shifts
        valence_shift = abs(signal.state.error_rates.get(
            InteroceptiveDimension.VALENCE, 0.0,
        ))

        is_turning_point = (
            valence_shift > 0.15
            or (
                signal.distance_to_bifurcation is not None
                and signal.distance_to_bifurcation < 0.1
            )
        )

        if is_turning_point and hasattr(self._thread, "mark_somatic_turning_point"):
            self._thread.mark_somatic_turning_point(
                attractor=signal.nearest_attractor or "transient",
                urgency=signal.urgency,
                dominant_error=signal.dominant_error.value,
                valence_rate=valence_shift,
            )
            return LoopDispatch(
                loop_name="thread_narrative",
                target="thread",
                action=f"turning_point_valence_shift={valence_shift:.3f}",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="thread_narrative",
            target="thread",
            action="below_threshold",
            magnitude=magnitude,
            applied=False,
        )

    def _dispatch_alive_visualization(
        self,
        signal: AllostaticSignal,
        magnitude: float,
        error: float,
        rate: float,
    ) -> LoopDispatch | None:
        """Push full interoceptive state to Alive for visualization."""
        if self._alive is None:
            return None

        if hasattr(self._alive, "push_somatic_state"):
            state_payload = {
                "sensed": {
                    d.value: v for d, v in signal.state.sensed.items()
                },
                "errors": {
                    d.value: signal.state.errors.get("moment", {}).get(d, 0.0)
                    for d in signal.state.sensed
                },
                "urgency": signal.urgency,
                "attractor": signal.nearest_attractor,
                "trajectory": signal.trajectory_heading,
                "energy_burn": signal.energy_burn_rate,
            }
            self._alive.push_somatic_state(state_payload)
            return LoopDispatch(
                loop_name="alive_visualization",
                target="alive",
                action="state_pushed",
                magnitude=magnitude,
                applied=True,
            )
        return LoopDispatch(
            loop_name="alive_visualization",
            target="alive",
            action="no_method",
            magnitude=magnitude,
            applied=False,
        )
