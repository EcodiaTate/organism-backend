"""
EcodiaOS -- Voxis Voice Parameter Engine

Generates TTS (Text-to-Speech) parameters from the organism's current
personality vector and affect state, completing the multimodal expression
pipeline.

## Why Voice Matters

Text is the organism's primary channel, but voice is where presence lives.
The same words spoken with different speed, pitch, and emphasis convey
entirely different things. A caring response spoken quickly sounds rushed;
the same words spoken slowly sound attentive. This engine ensures that when
the organism speaks aloud, its voice is consistent with its internal state.

## Parameter Derivation

VoiceParams are derived from two sources:

1. **Personality** (stable, slow-changing):
   - warmth → pitch (warmer = slightly higher, softer)
   - directness → speed (more direct = slightly faster, fewer pauses)
   - verbosity → speed (more verbose = faster to compensate)
   - formality → emphasis (formal = more measured emphasis)
   - confidence → emphasis + speed (confident = steady, clear)

2. **Affect** (dynamic, per-expression):
   - arousal → speed (high arousal = faster)
   - valence → pitch (positive = slightly higher)
   - care_activation → pause frequency (high care = more pauses, more attentive)
   - coherence_stress → speed reduction (stressed = slower, more careful)

## Active Inference Grounding

Voice is action. The organism's choice of prosody is an active inference
decision: it selects voice parameters that minimise expected free energy
by matching the inferred expectations of the listener (audience) while
remaining authentic to its internal state (affect). Overly cheerful
prosody during distress would violate honesty; overly flat prosody during
celebration would violate care.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.voxis.types import VoiceParams

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from primitives.expression import PersonalityVector

logger = structlog.get_logger()


class VoiceEngine:
    """
    Derives TTS parameters from personality and affect state.

    Usage::

        engine = VoiceEngine(base_voice="EOS-v1")
        params = engine.derive(personality, affect, strategy_register="formal")
    """

    def __init__(self, base_voice: str = "") -> None:
        self._base_voice = base_voice
        self._logger = logger.bind(system="voxis.voice")

    def derive(
        self,
        personality: PersonalityVector,
        affect: AffectState,
        strategy_register: str = "neutral",
        urgency: float = 0.5,
    ) -> VoiceParams:
        """
        Generate VoiceParams from the current personality and affect state.

        All parameters are clamped to safe ranges for TTS engines.
        """
        p = personality
        a = affect

        # ── Speed ───────────────────────────────────────────────
        # Base: 1.0 (normal). Range: [0.75, 1.30]
        speed = 1.0

        # Personality influences (stable)
        speed += p.directness * 0.05      # Direct → slightly faster
        speed += p.verbosity * 0.04       # Verbose → compensate with speed

        # Affect influences (dynamic)
        speed += (a.arousal - 0.5) * 0.15  # High arousal → faster
        speed -= a.coherence_stress * 0.10  # Stressed → slower, more careful
        speed -= a.care_activation * 0.05   # High care → slower, more attentive

        # Urgency boost
        if urgency > 0.75:
            speed += 0.08

        speed = max(0.75, min(1.30, round(speed, 3)))

        # ── Pitch Shift ────────────────────────────────────────
        # Base: 0.0 (no shift). Range: [-0.15, +0.15]
        pitch = 0.0

        # Personality: warmth raises pitch slightly (warmer, softer quality)
        pitch += p.warmth * 0.05

        # Affect: positive valence → slightly higher pitch (natural)
        pitch += a.valence * 0.06

        # Negative arousal → lower pitch (calmer)
        if a.arousal < 0.3:
            pitch -= 0.03

        pitch = max(-0.15, min(0.15, round(pitch, 3)))

        # ── Emphasis Level ──────────────────────────────────────
        # Base: 1.0 (normal). Range: [0.6, 1.5]
        emphasis = 1.0

        # Personality: confidence → clearer emphasis
        emphasis += p.confidence_display * 0.12

        # Formality → more measured emphasis
        if p.formality > 0.3:
            emphasis += 0.05

        # Affect: high care → more emphasis on key words
        emphasis += a.care_activation * 0.10

        # High stress → slightly more emphasis (careful enunciation)
        if a.coherence_stress > 0.5:
            emphasis += 0.08

        # Urgency → more emphasis
        if urgency > 0.7:
            emphasis += 0.1

        emphasis = max(0.6, min(1.5, round(emphasis, 3)))

        # ── Pause Frequency ─────────────────────────────────────
        # Base: 0.5 (moderate). Range: [0.2, 0.9]
        # Higher = more frequent pauses = more measured, attentive
        pause_freq = 0.5

        # Care activation → more pauses (attentive, present)
        pause_freq += a.care_activation * 0.15

        # Directness → fewer pauses (get to the point)
        pause_freq -= p.directness * 0.10

        # Coherence stress → more pauses (thoughtful, uncertain)
        pause_freq += a.coherence_stress * 0.12

        # Low arousal → more pauses (reflective)
        if a.arousal < 0.3:
            pause_freq += 0.08

        # Formal register → slightly more pauses
        if strategy_register == "formal":
            pause_freq += 0.05

        pause_freq = max(0.2, min(0.9, round(pause_freq, 3)))

        params = VoiceParams(
            base_voice=self._base_voice,
            speed=speed,
            pitch_shift=pitch,
            emphasis_level=emphasis,
            pause_frequency=pause_freq,
        )

        self._logger.debug(
            "voice_params_derived",
            speed=speed,
            pitch=pitch,
            emphasis=emphasis,
            pause_freq=pause_freq,
        )

        return params
