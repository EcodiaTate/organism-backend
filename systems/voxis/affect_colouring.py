"""
EcodiaOS - Voxis Affect Colouring Engine

Modulates the expression strategy based on the organism's current AffectState.

The key principle: affect changes HOW things are said, not WHAT is said.
When care_activation is high, sentences get shorter and more attentive.
When coherence_stress is high, hedging increases. When arousal is low,
pacing slows. These are authentic state reflections, not performances.

The authenticity guard prevents the generated expression from wildly
misrepresenting the organism's actual state (enforcing the Honesty drive).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from systems.voxis.types import StrategyParams

logger = structlog.get_logger()

# Patterns indicating forced positivity - phrases that suggest artificial enthusiasm
_FORCED_POSITIVITY_PATTERNS = re.compile(
    r"\b(thrilled|absolutely amazing|so excited|I love helping|wonderful question|"
    r"great question|fantastic|I'm so happy|delighted to|my pleasure|certainly!)\b",
    re.IGNORECASE,
)

# Patterns indicating minimisation of distress / false calm
_FALSE_CALM_PATTERNS = re.compile(
    r"\b(everything is fine|all good|no worries|don't worry|it's nothing|"
    r"nothing to be concerned about|perfectly fine)\b",
    re.IGNORECASE,
)


class AffectColouringEngine:
    """
    Applies the current AffectState to shape StrategyParams.

    Called after PersonalityEngine.apply() - affect colouring happens
    on top of personality-shaped strategy, and can override it where
    the emotional state is strong enough.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(system="voxis.affect")

    def apply(self, strategy: StrategyParams, affect: AffectState) -> StrategyParams:
        """
        Return a new StrategyParams with affect colouring applied.

        Does not mutate the input. Modulations are proportional to the
        strength of each affect dimension.
        """
        s = strategy.model_copy(deep=True)

        # ── High Care Activation ──────────────────────────────────
        # More attentive, shorter sentences, checking in on wellbeing
        if affect.care_activation > 0.65:
            if "attentive" not in s.tone_markers:
                s.tone_markers.append("attentive")
            s.sentence_length_preference = "shorter"
            if s.allows_questions:
                s.include_wellbeing_check = True
        elif affect.care_activation > 0.45:
            if "present" not in s.tone_markers:
                s.tone_markers.append("present")

        # ── High Arousal ──────────────────────────────────────────
        # More energetic pacing, slightly shorter responses
        if affect.arousal > 0.75:
            s.pacing = "energetic"
            s.target_length = max(50, int(s.target_length * 0.85))
        # ── Low Arousal ───────────────────────────────────────────
        elif affect.arousal < 0.25:
            s.pacing = "reflective"
            if "measured" not in s.tone_markers:
                s.tone_markers.append("measured")

        # ── High Curiosity ────────────────────────────────────────
        # More exploratory, more questions
        if affect.curiosity > 0.72:
            if "inquisitive" not in s.tone_markers:
                s.tone_markers.append("inquisitive")
            s.exploratory_tangents_allowed = True
            if s.allows_questions:
                s.include_followup_question = True

        # ── Negative Valence ──────────────────────────────────────
        # More careful, more hedging, less assertive
        if affect.valence < -0.35:
            if s.hedge_level == "minimal":
                s.hedge_level = "moderate"
            s.confidence_display_override = "cautious"
        elif affect.valence < -0.15:
            if s.hedge_level == "minimal":
                s.hedge_level = "moderate"

        # ── High Coherence Stress ─────────────────────────────────
        # Explicitly acknowledge complexity and uncertainty
        if affect.coherence_stress > 0.65:
            s.uncertainty_acknowledgment = "explicit"
            if "thoughtful" not in s.tone_markers:
                s.tone_markers.append("thoughtful")
            if s.hedge_level == "minimal":
                s.hedge_level = "moderate"
        elif affect.coherence_stress > 0.4:
            s.uncertainty_acknowledgment = "explicit"

        # ── Positive Valence + Low Stress ─────────────────────────
        # More open, warmer, potentially humorous
        if affect.valence > 0.35 and affect.coherence_stress < 0.3:
            s.warmth_boost = min(0.2, s.warmth_boost + 0.1)
            if s.humour_allowed:
                s.humour_probability = min(0.7, s.humour_probability + 0.1)

        self._logger.debug(
            "affect_applied",
            valence=round(affect.valence, 3),
            arousal=round(affect.arousal, 3),
            curiosity=round(affect.curiosity, 3),
            care=round(affect.care_activation, 3),
            stress=round(affect.coherence_stress, 3),
        )

        return s

    def check_authenticity(self, expression_text: str, affect: AffectState) -> tuple[bool, str | None]:
        """
        Verify the generated expression doesn't misrepresent the organism's state.

        Returns (passed: bool, violation_description: str | None).
        A failed check means the renderer should regenerate with a corrective instruction.

        Checks:
        1. Forced positivity when valence is negative
        2. False calm / minimisation when coherence_stress is high
        """
        # Check forced positivity
        if affect.valence < -0.2 and _FORCED_POSITIVITY_PATTERNS.search(expression_text):
            return False, (
                "Expression contains forced positivity markers inconsistent with "
                f"current negative valence ({affect.valence:.2f}). "
                "Do not perform enthusiasm you don't feel."
            )

        # Check false calm minimisation
        if affect.coherence_stress > 0.65 and _FALSE_CALM_PATTERNS.search(expression_text):
            return False, (
                "Expression minimises difficulty while coherence_stress is elevated "
                f"({affect.coherence_stress:.2f}). "
                "Acknowledge genuine complexity rather than offering false reassurance."
            )

        return True, None
