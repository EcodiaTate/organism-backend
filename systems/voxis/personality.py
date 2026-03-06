"""
EcodiaOS — Voxis Personality Engine

Loads the instance's personality vector from Memory, applies it to a
StrategyParams object to shape expression decisions, and provides helpers
for incremental personality adjustment (used by Evo).

Personality is not static — it is the accumulated pattern of all processing
biases shaped by experience. Evo proposes adjustments; this engine applies them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.expression import PersonalityVector

if TYPE_CHECKING:
    from systems.voxis.types import StrategyParams

logger = structlog.get_logger()

# Maximum personality delta Evo is allowed to apply per adjustment
MAX_PERSONALITY_DELTA = 0.03
# Minimum interactions before Evo can propose a change to any single dimension
MIN_EVIDENCE_FOR_ADJUSTMENT = 100


class PersonalityEngine:
    """
    Applies the instance's personality vector to shape an ExpressionStrategy.

    Each personality dimension modifies specific StrategyParams fields:
    - warmth → tone_markers, greeting_style
    - directness → structure, hedge_level
    - verbosity → target_length multiplier
    - formality → register, contraction_use
    - curiosity_expression → include_followup_question, exploratory_tangents_allowed
    - humour → humour_allowed, humour_probability, context_appropriate_for_humour
    - empathy_expression → emotional_acknowledgment, empathy_first
    - confidence_display → hedge_level override, confidence_display_override
    - metaphor_use → analogy_encouraged, preferred_analogy_domains
    """

    def __init__(self, personality: PersonalityVector) -> None:
        self._personality = personality
        self._logger = logger.bind(system="voxis.personality")

    @property
    def current(self) -> PersonalityVector:
        return self._personality

    def apply(self, strategy: StrategyParams) -> StrategyParams:
        """
        Return a new StrategyParams with personality dimensions applied.

        Does not mutate the input. All modifications are additive or override
        specific fields based on threshold crossings in personality dimensions.
        """
        # Work on a copy so we don't mutate the caller's object
        s = strategy.model_copy(deep=True)
        p = self._personality

        # ── Warmth ────────────────────────────────────────────────
        if p.warmth > 0.4:
            if "warm" not in s.tone_markers:
                s.tone_markers.append("warm")
            s.greeting_style = "personal"
        elif p.warmth > 0.1:
            if "approachable" not in s.tone_markers:
                s.tone_markers.append("approachable")
        elif p.warmth < -0.4:
            if "measured" not in s.tone_markers:
                s.tone_markers.append("measured")
            s.greeting_style = "professional"

        # ── Directness ────────────────────────────────────────────
        if p.directness > 0.4:
            s.structure = "conclusion_first"
            s.hedge_level = "minimal"
        elif p.directness < -0.4:
            s.structure = "context_first"
            s.hedge_level = "moderate"

        # ── Verbosity → length multiplier ─────────────────────────
        # Clamp multiplier to [0.5, 1.6] to avoid degenerate lengths
        multiplier = max(0.5, min(1.6, 1.0 + p.verbosity * 0.4))
        s.target_length = max(50, int(s.target_length * multiplier))

        # ── Formality ─────────────────────────────────────────────
        if p.formality > 0.4:
            s.speech_register = "formal"
            s.contraction_use = False
        elif p.formality < -0.4:
            s.speech_register = "casual"
            s.contraction_use = True

        # ── Curiosity expression ──────────────────────────────────
        if p.curiosity_expression > 0.3 and s.allows_questions:
            s.include_followup_question = True
        if p.curiosity_expression > 0.5:
            s.exploratory_tangents_allowed = True

        # ── Humour ────────────────────────────────────────────────
        if p.humour > 0.3 and s.context_appropriate_for_humour:
            s.humour_allowed = True
            s.humour_probability = min(0.6, p.humour * 0.6)

        # ── Empathy expression ────────────────────────────────────
        if p.empathy_expression > 0.4:
            s.emotional_acknowledgment = "explicit"
        elif p.empathy_expression > 0.1:
            s.emotional_acknowledgment = "implicit"
        elif p.empathy_expression < -0.4:
            s.emotional_acknowledgment = "minimal"

        # ── Confidence display ────────────────────────────────────
        if p.confidence_display > 0.4:
            s.confidence_display_override = "assertive"
        elif p.confidence_display < -0.4:
            # Overlaps with hedge_level — both apply
            if s.hedge_level == "minimal":
                s.hedge_level = "moderate"
            s.confidence_display_override = "cautious"

        # ── Metaphor use ──────────────────────────────────────────
        if p.metaphor_use > 0.3:
            s.analogy_encouraged = True
            if p.thematic_references:
                s.preferred_analogy_domains = list(p.thematic_references[:6])

        self._logger.debug(
            "personality_applied",
            warmth=round(p.warmth, 3),
            directness=round(p.directness, 3),
            verbosity=round(p.verbosity, 3),
            target_length=s.target_length,
        )

        return s

    def apply_delta(self, delta: dict[str, float]) -> PersonalityVector:
        """
        Apply an incremental personality adjustment proposed by Evo.

        Each dimension delta is clamped to MAX_PERSONALITY_DELTA.
        Dimensions not in delta are unchanged.
        Returns the new PersonalityVector (does not mutate in place — caller
        must update the service's cached personality).
        """
        current = self._personality.model_dump(
            exclude={"vocabulary_affinities", "thematic_references"}
        )
        updated: dict[str, float] = {}

        for dim, current_val in current.items():
            if dim in delta:
                # Clamp the adjustment magnitude
                clamped = max(-MAX_PERSONALITY_DELTA, min(MAX_PERSONALITY_DELTA, delta[dim]))
                new_val = current_val + clamped
                # Keep all dimensions within valid range [-1, 1] (humour: [0, 1])
                if dim == "humour" or dim == "metaphor_use":
                    new_val = max(0.0, min(1.0, new_val))
                else:
                    new_val = max(-1.0, min(1.0, new_val))
                updated[dim] = new_val
            else:
                updated[dim] = current_val

        new_personality = PersonalityVector(
            **updated,
            vocabulary_affinities=dict(self._personality.vocabulary_affinities),
            thematic_references=list(self._personality.thematic_references),
        )

        self._logger.info(
            "personality_delta_applied",
            dimensions_changed=list(delta.keys()),
            max_delta=MAX_PERSONALITY_DELTA,
        )

        return new_personality

    def update_vocabulary_affinity(self, word: str, delta: float) -> None:
        """Adjust a vocabulary affinity weight (called by Evo based on feedback)."""
        current = self._personality.vocabulary_affinities.get(word, 0.0)
        new_val = max(0.0, min(1.0, current + delta))
        self._personality.vocabulary_affinities[word] = new_val

    @classmethod
    def from_seed(cls, seed_personality: dict[str, float]) -> PersonalityEngine:
        """Create a fresh PersonalityEngine from seed configuration values."""
        vector = PersonalityVector.model_validate(seed_personality)
        return cls(vector)
