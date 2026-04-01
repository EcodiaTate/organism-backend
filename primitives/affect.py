"""
EcodiaOS - Affect State Primitive

The emotional context that modulates all processing.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now


class InteroceptiveDimension(enum.StrEnum):
    """The nine dimensions of felt internal state (canonical - shared across systems)."""

    ENERGY = "energy"                       # Metabolic budget (token/compute availability)
    AROUSAL = "arousal"                     # Activation level (cycle speed, parallelism)
    VALENCE = "valence"                     # Net allostatic trend (improving vs deteriorating)
    CONFIDENCE = "confidence"               # Generative model fit (prediction accuracy)
    COHERENCE = "coherence"                 # Inter-system integration quality
    SOCIAL_CHARGE = "social_charge"         # Relational engagement quality
    CURIOSITY_DRIVE = "curiosity_drive"     # Epistemic appetite
    INTEGRITY = "integrity"                 # Ethical/constitutional alignment + system health
    TEMPORAL_PRESSURE = "temporal_pressure"  # Urgency / time horizon compression


class AffectState(EOSBaseModel):
    """
    The organism's current emotional state.
    Modulates attention, expression, decision-making, and learning.
    """

    valence: float = Field(0.0, ge=-1.0, le=1.0)        # Negative to positive
    arousal: float = Field(0.0, ge=0.0, le=1.0)          # Calm to activated
    dominance: float = Field(0.5, ge=0.0, le=1.0)        # Submissive to dominant
    curiosity: float = Field(0.0, ge=0.0, le=1.0)        # Epistemic drive
    care_activation: float = Field(0.0, ge=0.0, le=1.0)  # How active is the Care drive
    coherence_stress: float = Field(0.0, ge=0.0, le=1.0) # Prediction error load
    source_events: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)

    @classmethod
    def neutral(cls) -> AffectState:
        """A calm, neutral baseline state."""
        return cls(
            valence=0.0,
            arousal=0.1,
            dominance=0.5,
            curiosity=0.2,
            care_activation=0.1,
            coherence_stress=0.0,
        )

    def to_map(self) -> dict[str, float]:
        """Convert to a flat dict for Neo4j storage."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "curiosity": self.curiosity,
            "care_activation": self.care_activation,
            "coherence_stress": self.coherence_stress,
        }

    @classmethod
    def from_map(cls, data: dict[str, float]) -> AffectState:
        """Reconstruct from a Neo4j MAP property."""
        filtered = {k: v for k, v in data.items() if k in cls.model_fields}
        return cls.model_validate(filtered)


class AffectDelta(EOSBaseModel):
    """A change to be applied to the current affect state."""

    delta_valence: float = 0.0
    delta_arousal: float = 0.0
    delta_dominance: float = 0.0
    delta_curiosity: float = 0.0
    delta_care_activation: float = 0.0
    delta_coherence_stress: float = 0.0
    reason: str = ""

    def apply_to(self, state: AffectState, lerp_rate: float = 0.3) -> AffectState:
        """Apply this delta to an affect state with exponential smoothing."""

        def _clamp(value: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, value))

        return AffectState(
            valence=_clamp(
                state.valence + self.delta_valence * lerp_rate, -1.0, 1.0
            ),
            arousal=_clamp(
                state.arousal + self.delta_arousal * lerp_rate, 0.0, 1.0
            ),
            dominance=_clamp(
                state.dominance + self.delta_dominance * lerp_rate, 0.0, 1.0
            ),
            curiosity=_clamp(
                state.curiosity + self.delta_curiosity * lerp_rate, 0.0, 1.0
            ),
            care_activation=_clamp(
                state.care_activation + self.delta_care_activation * lerp_rate, 0.0, 1.0
            ),
            coherence_stress=_clamp(
                state.coherence_stress + self.delta_coherence_stress * lerp_rate, 0.0, 1.0
            ),
            source_events=state.source_events,
            timestamp=utc_now(),
        )
