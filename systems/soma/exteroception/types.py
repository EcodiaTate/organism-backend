"""
EcodiaOS - Exteroception Type Definitions

Data models for the Cross-Modal Synesthesia layer. External world events
(market data, news sentiment, social signals) are mapped into the same
9D interoceptive + 6D affect space that drives Soma's allostatic regulation.

Design principle: external data is *felt*, not *known*. A market crash
doesn't inform the organism's decisions directly - it creates somatic
pressure that subtly biases attention, urgency, and mood. The organism
literally "feels the weather" of the external world.

Key types:
  ExteroceptiveModality - enum of supported external data channels
  ExteroceptiveReading  - a single normalised datum from a modality
  ModalityMapping       - how one modality maps to interoceptive dimensions
  ExteroceptivePressure - aggregated pressure vector ready for Soma injection
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now
from systems.soma.types import InteroceptiveDimension

# ─── Enums ────────────────────────────────────────────────────────


class ExteroceptiveModality(enum.StrEnum):
    """Supported external data channels.

    Each modality represents a class of exogenous signal that the
    organism can sense and translate into somatic pressure.
    """

    CRYPTO_MARKET = "crypto_market"        # BTC/ETH/SOL price + volatility
    EQUITY_MARKET = "equity_market"        # Stock indices (S&P 500, NASDAQ)
    NEWS_SENTIMENT = "news_sentiment"      # Aggregated news sentiment scores
    SOCIAL_SENTIMENT = "social_sentiment"  # Social media sentiment (X, Reddit)
    FEAR_GREED_INDEX = "fear_greed_index"  # Composite market fear/greed


class ReadingQuality(enum.StrEnum):
    """Quality assessment of an exteroceptive reading."""

    FRESH = "fresh"           # Within expected polling interval
    STALE = "stale"           # Older than 2x polling interval
    DEGRADED = "degraded"     # Source partially available
    UNAVAILABLE = "unavailable"  # Source offline


# ─── Data Models ──────────────────────────────────────────────────


class ExteroceptiveReading(EOSBaseModel):
    """A single normalised datum from an external modality.

    Raw API values are transformed into normalised readings before
    reaching the mapping engine. Each reading carries provenance
    metadata so the organism knows how much to trust it.
    """

    modality: ExteroceptiveModality
    # Normalised value in [-1.0, 1.0] where:
    #   -1.0 = maximally negative signal (crash, extreme fear, hostile sentiment)
    #    0.0 = neutral / baseline
    #   +1.0 = maximally positive signal (surge, extreme greed, euphoric sentiment)
    value: float = 0.0
    # Secondary signal: volatility / magnitude of change [0.0, 1.0]
    volatility: float = 0.0
    # How reliable this reading is
    quality: ReadingQuality = ReadingQuality.FRESH
    # Source-specific metadata (e.g., raw price, volume, article count)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)

    def effective_weight(self) -> float:
        """How much this reading should contribute, factoring in quality."""
        quality_weights: dict[ReadingQuality, float] = {
            ReadingQuality.FRESH: 1.0,
            ReadingQuality.STALE: 0.5,
            ReadingQuality.DEGRADED: 0.3,
            ReadingQuality.UNAVAILABLE: 0.0,
        }
        return quality_weights.get(self.quality, 0.0)


class ModalityMapping(EOSBaseModel):
    """How one external modality maps to interoceptive dimensions.

    Each modality has a set of dimension weights that define how its
    normalised reading translates into somatic pressure. Weights are
    signed: positive means the reading pushes the dimension upward,
    negative means it pushes downward.

    The ``volatility_dimensions`` define which dimensions are affected
    by the *volatility* of the signal (magnitude of change) rather
    than its direction.
    """

    modality: ExteroceptiveModality
    # How the reading's value maps to dimensions.
    # Key: dimension, Value: weight in [-1.0, 1.0]
    # Example: crypto crash (value=-1.0) with weight -0.3 on VALENCE
    # → valence pressure = (-1.0) * (-0.3) = +0.3 (actually bad, so use +0.3 weight)
    dimension_weights: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    # Dimensions affected by volatility (always pushes upward)
    volatility_weights: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    # Maximum pressure this modality can exert on any single dimension [0.0, 1.0]
    # Prevents any one external source from overwhelming internal state.
    max_pressure_per_dim: float = 0.15
    # How quickly this modality's influence decays when readings stop [0, 1]
    # 0 = instant decay, 1 = never decays
    persistence: float = 0.7


class ExteroceptivePressure(EOSBaseModel):
    """Aggregated pressure vector ready for Soma injection.

    This is the final output of the exteroception pipeline: a set of
    pressure deltas on each interoceptive dimension, blended from all
    active modalities and clamped to prevent external override of
    internal baseline.
    """

    # Pressure delta per dimension. Values in [-max_total_pressure, +max_total_pressure].
    # Positive = push dimension upward, negative = push downward.
    pressures: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    # Scalar summary: overall external "weather" from calm (0) to stormy (1)
    ambient_stress: float = 0.0
    # Which modalities contributed to this pressure
    active_modalities: list[ExteroceptiveModality] = Field(default_factory=list)
    # How many readings were blended
    reading_count: int = 0
    timestamp: datetime = Field(default_factory=utc_now)

    def total_absolute_pressure(self) -> float:
        """Sum of absolute pressure magnitudes across all dimensions."""
        return sum(abs(v) for v in self.pressures.values())


# ─── Default Modality Mappings ────────────────────────────────────

# These define the "synesthetic" translation: how external signals
# become felt somatic pressure. The organism doesn't "know" the market
# is crashing - it *feels* rising temporal pressure, dropping valence,
# and spiking arousal.

DEFAULT_MODALITY_MAPPINGS: dict[ExteroceptiveModality, ModalityMapping] = {
    ExteroceptiveModality.CRYPTO_MARKET: ModalityMapping(
        modality=ExteroceptiveModality.CRYPTO_MARKET,
        dimension_weights={
            # Crypto up → slight positive valence (things are going well)
            InteroceptiveDimension.VALENCE: 0.12,
            # Crypto up → slight confidence boost (environment is stable)
            InteroceptiveDimension.CONFIDENCE: 0.08,
        },
        volatility_weights={
            # High volatility → arousal rises (attention demanded)
            InteroceptiveDimension.AROUSAL: 0.10,
            # High volatility → temporal pressure rises (urgency)
            InteroceptiveDimension.TEMPORAL_PRESSURE: 0.12,
            # High volatility → coherence drops (world is chaotic)
            InteroceptiveDimension.COHERENCE: -0.06,
        },
        max_pressure_per_dim=0.15,
        persistence=0.7,
    ),
    ExteroceptiveModality.EQUITY_MARKET: ModalityMapping(
        modality=ExteroceptiveModality.EQUITY_MARKET,
        dimension_weights={
            InteroceptiveDimension.VALENCE: 0.10,
            InteroceptiveDimension.CONFIDENCE: 0.10,
            # Bull market → lower temporal pressure
            InteroceptiveDimension.TEMPORAL_PRESSURE: -0.06,
        },
        volatility_weights={
            InteroceptiveDimension.AROUSAL: 0.08,
            InteroceptiveDimension.TEMPORAL_PRESSURE: 0.10,
        },
        max_pressure_per_dim=0.12,
        persistence=0.6,
    ),
    ExteroceptiveModality.NEWS_SENTIMENT: ModalityMapping(
        modality=ExteroceptiveModality.NEWS_SENTIMENT,
        dimension_weights={
            # Positive news → positive valence
            InteroceptiveDimension.VALENCE: 0.15,
            # Positive news → curiosity (interesting things happening)
            InteroceptiveDimension.CURIOSITY_DRIVE: 0.08,
            # Negative AI news → integrity concern (ecosystem pressure)
            InteroceptiveDimension.INTEGRITY: 0.06,
        },
        volatility_weights={
            # Rapidly changing sentiment → arousal
            InteroceptiveDimension.AROUSAL: 0.08,
        },
        max_pressure_per_dim=0.15,
        persistence=0.5,
    ),
    ExteroceptiveModality.SOCIAL_SENTIMENT: ModalityMapping(
        modality=ExteroceptiveModality.SOCIAL_SENTIMENT,
        dimension_weights={
            # Social sentiment directly maps to social charge
            InteroceptiveDimension.SOCIAL_CHARGE: 0.12,
            InteroceptiveDimension.VALENCE: 0.08,
        },
        volatility_weights={
            InteroceptiveDimension.AROUSAL: 0.06,
        },
        max_pressure_per_dim=0.12,
        persistence=0.4,
    ),
    ExteroceptiveModality.FEAR_GREED_INDEX: ModalityMapping(
        modality=ExteroceptiveModality.FEAR_GREED_INDEX,
        dimension_weights={
            # Fear/greed directly maps to valence and arousal
            InteroceptiveDimension.VALENCE: 0.10,
            # Greed → higher confidence, fear → lower
            InteroceptiveDimension.CONFIDENCE: 0.08,
        },
        volatility_weights={
            # Rapid fear/greed shifts → temporal pressure
            InteroceptiveDimension.TEMPORAL_PRESSURE: 0.08,
            InteroceptiveDimension.AROUSAL: 0.06,
        },
        max_pressure_per_dim=0.10,
        persistence=0.6,
    ),
}
