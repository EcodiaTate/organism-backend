"""
EcodiaOS - Soma Allostatic Feedback Loop Definitions

Defines the 15 allostatic feedback loops from Spec Section XI.

Each loop follows the pattern:
  System X output → Soma sensed change → prediction shift →
  error change → propagate to System Y

This module ONLY defines the loop topology and the error signals each
consuming system would use. It does NOT wire the other systems - the
executive rewiring step (Step 4 in the EcodiaOS roadmap) handles the
receiving end. Soma's job is to expose the loop definitions and emit
the right error signals on each cycle.

The loop map is consumed by:
  - SomaService: to annotate AllostaticSignal with active loop info
  - Nova: to understand which error signals to deliberate on
  - Evo: to tune loop coupling strengths
  - Thread: to narrate allostatic regulation events
"""

from __future__ import annotations

from typing import Any

from systems.soma.types import InteroceptiveDimension


class AllostaticLoop:
    """
    One allostatic feedback loop connecting Soma error signals to a consuming system.

    Attributes:
        name: Human-readable loop identifier.
        sensed_dimension: Which interoceptive dimension triggers this loop
            (None if the loop reads the full state).
        error_consumer: Which system consumes the error signal.
        regulatory_description: What the consuming system does in response.
        coupling_strength: How strongly the error drives the response [0, 1].
            Evo can tune this. Default 1.0 (full coupling).
    """

    __slots__ = (
        "name",
        "sensed_dimension",
        "error_consumer",
        "regulatory_description",
        "coupling_strength",
    )

    def __init__(
        self,
        name: str,
        sensed_dimension: InteroceptiveDimension | None,
        error_consumer: str,
        regulatory_description: str,
        coupling_strength: float = 1.0,
    ) -> None:
        self.name = name
        self.sensed_dimension = sensed_dimension
        self.error_consumer = error_consumer
        self.regulatory_description = regulatory_description
        self.coupling_strength = coupling_strength

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sensed_dimension": self.sensed_dimension.value if self.sensed_dimension else None,
            "error_consumer": self.error_consumer,
            "regulatory_description": self.regulatory_description,
            "coupling_strength": self.coupling_strength,
        }


# ─── The 15 Allostatic Loops (Spec Section XI) ───────────────────────

ALLOSTATIC_LOOP_MAP: dict[str, AllostaticLoop] = {

    # 1. Top-down prediction
    "top_down_prediction": AllostaticLoop(
        name="top_down_prediction",
        sensed_dimension=InteroceptiveDimension.CONFIDENCE,
        error_consumer="atune",
        regulatory_description=(
            "High confidence precision → Atune's prediction error threshold "
            "tightens, making unexpected percepts more salient"
        ),
    ),

    # 2. Goal-guided attention
    "goal_guided_attention": AllostaticLoop(
        name="goal_guided_attention",
        sensed_dimension=InteroceptiveDimension.TEMPORAL_PRESSURE,
        error_consumer="atune",
        regulatory_description=(
            "Positive temporal_pressure error → goal-relevant percepts "
            "score higher in Atune's salience competition"
        ),
    ),

    # 3. Expression feedback
    "expression_feedback": AllostaticLoop(
        name="expression_feedback",
        sensed_dimension=InteroceptiveDimension.SOCIAL_CHARGE,
        error_consumer="voxis",
        regulatory_description=(
            "Low social_charge → warmer, more engaging expression; "
            "high social_charge → can afford cooler, more precise tone"
        ),
    ),

    # 4. Memory salience decay
    "memory_salience_decay": AllostaticLoop(
        name="memory_salience_decay",
        sensed_dimension=None,  # Full interoceptive state
        error_consumer="memory",
        regulatory_description=(
            "Salience = magnitude of interoceptive prediction error "
            "the memory helped resolve; memories that reduce error persist"
        ),
    ),

    # 5. Personality evolution
    "personality_evolution": AllostaticLoop(
        name="personality_evolution",
        sensed_dimension=InteroceptiveDimension.COHERENCE,
        error_consumer="evo",
        regulatory_description=(
            "Coherence drift from setpoint → Evo tunes personality traits "
            "and dynamics matrix to restore integration quality"
        ),
    ),

    # 6. Rhythm modulation
    "rhythm_modulation": AllostaticLoop(
        name="rhythm_modulation",
        sensed_dimension=InteroceptiveDimension.AROUSAL,
        error_consumer="synapse",
        regulatory_description=(
            "Arousal setpoint deviation → Synapse adjusts theta cycle "
            "period; high arousal = faster cycles, low = slower"
        ),
    ),

    # 7. Affect expression (full state)
    "affect_expression": AllostaticLoop(
        name="affect_expression",
        sensed_dimension=None,  # Full interoceptive state
        error_consumer="voxis",
        regulatory_description=(
            "Voxis reads the entire Soma signal directly; expression = "
            "interoceptive state made audible/visible"
        ),
    ),

    # 8. Coherence safe mode
    "coherence_safe_mode": AllostaticLoop(
        name="coherence_safe_mode",
        sensed_dimension=InteroceptiveDimension.COHERENCE,
        error_consumer="synapse",
        regulatory_description=(
            "Low coherence approaching bifurcation → Synapse triggers "
            "safe mode proactively before integration breaks down"
        ),
    ),

    # 9. Nova deliberation trigger
    "nova_deliberation": AllostaticLoop(
        name="nova_deliberation",
        sensed_dimension=InteroceptiveDimension.TEMPORAL_PRESSURE,
        error_consumer="nova",
        regulatory_description=(
            "High urgency or temporal dissonance → Nova enters allostatic "
            "deliberation to resolve time-horizon conflicts"
        ),
    ),

    # 10. Atune precision weighting
    "atune_precision": AllostaticLoop(
        name="atune_precision",
        sensed_dimension=InteroceptiveDimension.CONFIDENCE,
        error_consumer="atune",
        regulatory_description=(
            "Soma precision weights modulate Atune's attention allocation; "
            "dimensions with high error get more perceptual bandwidth"
        ),
    ),

    # 11. Evo hypothesis generation
    "evo_hypothesis": AllostaticLoop(
        name="evo_hypothesis",
        sensed_dimension=InteroceptiveDimension.CURIOSITY_DRIVE,
        error_consumer="evo",
        regulatory_description=(
            "High curiosity_drive error → Evo generates more hypotheses; "
            "low curiosity → Evo consolidates rather than explores"
        ),
    ),

    # 12. Oneiros sleep pressure
    "oneiros_sleep_pressure": AllostaticLoop(
        name="oneiros_sleep_pressure",
        sensed_dimension=InteroceptiveDimension.ENERGY,
        error_consumer="oneiros",
        regulatory_description=(
            "Low energy with negative error rate → Oneiros sleep pressure "
            "rises; organism needs consolidation and energy recovery"
        ),
    ),

    # 13. Thymos incident detection
    "thymos_integrity": AllostaticLoop(
        name="thymos_integrity",
        sensed_dimension=InteroceptiveDimension.INTEGRITY,
        error_consumer="thymos",
        regulatory_description=(
            "Negative integrity error → Thymos heightens constitutional "
            "monitoring; potential ethical drift detected somatically"
        ),
    ),

    # 14. Thread narrative turning points
    "thread_narrative": AllostaticLoop(
        name="thread_narrative",
        sensed_dimension=InteroceptiveDimension.VALENCE,
        error_consumer="thread",
        regulatory_description=(
            "Large valence shifts or bifurcation crossings → Thread marks "
            "narrative turning points in the organism's story"
        ),
    ),

    # 15. Alive visualization
    "alive_visualization": AllostaticLoop(
        name="alive_visualization",
        sensed_dimension=None,  # Full interoceptive state
        error_consumer="alive",
        regulatory_description=(
            "Full interoceptive state drives Alive's real-time visualization; "
            "the organism's inner life rendered as dynamic art"
        ),
    ),
}


def get_loops_for_consumer(consumer: str) -> list[AllostaticLoop]:
    """Return all loops that a specific system consumes."""
    return [
        loop for loop in ALLOSTATIC_LOOP_MAP.values()
        if loop.error_consumer == consumer
    ]


def get_active_loop_signals(
    errors: dict[InteroceptiveDimension, float],
    error_rates: dict[InteroceptiveDimension, float],
    threshold: float = 0.1,
) -> dict[str, dict[str, float]]:
    """
    Compute which loops are currently active based on error magnitudes.

    Returns a dict of loop_name -> {error, rate, coupled_strength}
    for loops whose dimension error exceeds the threshold.

    This is called by SomaService to annotate signals with active loop info,
    so consuming systems know which feedback channels are firing.
    """
    active: dict[str, dict[str, float]] = {}

    for loop_name, loop in ALLOSTATIC_LOOP_MAP.items():
        if loop.sensed_dimension is None:
            # Full-state loops are always active when any error is significant
            max_err = max((abs(v) for v in errors.values()), default=0.0)
            if max_err > threshold:
                active[loop_name] = {
                    "error": max_err,
                    "rate": max((abs(v) for v in error_rates.values()), default=0.0),
                    "coupled_strength": loop.coupling_strength,
                }
            continue

        dim = loop.sensed_dimension
        error = abs(errors.get(dim, 0.0))
        rate = error_rates.get(dim, 0.0)

        if error > threshold:
            active[loop_name] = {
                "error": error,
                "rate": rate,
                "coupled_strength": loop.coupling_strength,
            }

    return active
