"""
EcodiaOS - Exteroceptive Mapping Engine

The mathematical core of Cross-Modal Synesthesia. Translates normalised
ExteroceptiveReadings into somatic pressure vectors using configurable
modality mappings.

Design invariants:
  - No single external modality can move any dimension by more than
    ``max_pressure_per_dim`` (default 0.15). This prevents a market flash
    crash from flipping the organism into full anxiety spiral.
  - Total exteroceptive pressure across all modalities is clamped by
    ``max_total_pressure`` (default 0.25). External data nudges the
    organism's mood - it doesn't control it.
  - EMA smoothing prevents rapid oscillation from noisy data sources.
  - When readings become stale, their contribution decays via the
    ``persistence`` factor on the ModalityMapping.

Pipeline:
  1. For each fresh reading, compute per-dimension pressure from
     dimension_weights (value-direction) and volatility_weights
     (magnitude-direction).
  2. Clamp each modality's contribution to max_pressure_per_dim.
  3. Blend all modality pressures additively.
  4. Clamp total pressure to max_total_pressure per dimension.
  5. EMA-smooth the result.
  6. Emit ExteroceptivePressure.
"""

from __future__ import annotations

import math

import structlog

from systems.soma.types import ALL_DIMENSIONS, InteroceptiveDimension

from .types import (
    DEFAULT_MODALITY_MAPPINGS,
    ExteroceptiveModality,
    ExteroceptivePressure,
    ExteroceptiveReading,
    ModalityMapping,
)

logger = structlog.get_logger("systems.soma.exteroception.mapping")

# Hard ceiling: no dimension can be pushed more than this by ALL external
# sources combined. Internal baseline always dominates.
_DEFAULT_MAX_TOTAL_PRESSURE: float = 0.25


class ExteroceptiveMappingEngine:
    """Translates normalised external readings into somatic pressure vectors.

    The engine maintains per-modality EMA state so that rapid fluctuations
    in external data are smoothed before reaching the interoceptive substrate.
    Stale readings decay toward zero via the modality's persistence factor.

    Thread safety: not thread-safe. Expected to be called from a single
    asyncio task (the ExteroceptionService poll loop).
    """

    def __init__(
        self,
        mappings: dict[ExteroceptiveModality, ModalityMapping] | None = None,
        ema_alpha: float = 0.3,
        max_total_pressure: float = _DEFAULT_MAX_TOTAL_PRESSURE,
    ) -> None:
        self._mappings = mappings or dict(DEFAULT_MODALITY_MAPPINGS)
        self._ema_alpha = ema_alpha
        self._max_total_pressure = max_total_pressure

        # Per-modality smoothed readings: modality → last smoothed reading
        self._smoothed: dict[ExteroceptiveModality, ExteroceptiveReading] = {}
        # Per-modality decay counters: how many polls since last fresh reading
        self._stale_counts: dict[ExteroceptiveModality, int] = {}

        # Smoothed output pressure (EMA across poll cycles)
        self._smoothed_pressure: dict[InteroceptiveDimension, float] = {
            d: 0.0 for d in ALL_DIMENSIONS
        }

    @property
    def mappings(self) -> dict[ExteroceptiveModality, ModalityMapping]:
        return self._mappings

    def update_mapping(self, mapping: ModalityMapping) -> None:
        """Hot-update a modality mapping (e.g., from Evo learning)."""
        self._mappings[mapping.modality] = mapping

    def process_readings(
        self,
        readings: list[ExteroceptiveReading],
    ) -> ExteroceptivePressure:
        """Transform a batch of readings into an aggregated pressure vector.

        Called once per exteroception poll cycle (typically every 60-300s).
        Readings from multiple modalities are processed independently,
        then blended and clamped.
        """
        # Step 1: Smooth incoming readings (EMA per modality)
        for reading in readings:
            self._smooth_reading(reading)

        # Step 2: Decay modalities with no fresh reading this cycle
        active_modalities_this_cycle = {r.modality for r in readings}
        for modality in list(self._smoothed.keys()):
            if modality not in active_modalities_this_cycle:
                self._decay_modality(modality)

        # Step 3: Compute raw pressure from each modality
        raw_pressures: dict[InteroceptiveDimension, float] = {
            d: 0.0 for d in ALL_DIMENSIONS
        }
        active_modalities: list[ExteroceptiveModality] = []

        for modality, smoothed_reading in self._smoothed.items():
            mapping = self._mappings.get(modality)
            if mapping is None:
                continue

            modality_pressure = self._compute_modality_pressure(
                smoothed_reading, mapping
            )

            # Only count as active if it contributes non-trivially
            if any(abs(v) > 0.001 for v in modality_pressure.values()):
                active_modalities.append(modality)

            # Additive blend
            for dim, delta in modality_pressure.items():
                raw_pressures[dim] += delta

        # Step 4: Clamp total pressure per dimension
        clamped: dict[InteroceptiveDimension, float] = {}
        for dim in ALL_DIMENSIONS:
            clamped[dim] = _clamp(
                raw_pressures[dim],
                -self._max_total_pressure,
                self._max_total_pressure,
            )

        # Step 5: EMA-smooth the output pressure
        for dim in ALL_DIMENSIONS:
            self._smoothed_pressure[dim] = (
                self._ema_alpha * clamped[dim]
                + (1.0 - self._ema_alpha) * self._smoothed_pressure[dim]
            )

        # Step 6: Compute ambient stress scalar
        ambient = self._compute_ambient_stress()

        return ExteroceptivePressure(
            pressures=dict(self._smoothed_pressure),
            ambient_stress=ambient,
            active_modalities=active_modalities,
            reading_count=len(readings),
        )

    def get_current_pressure(self) -> ExteroceptivePressure:
        """Return the last computed smoothed pressure without new readings."""
        active = [
            m for m, r in self._smoothed.items()
            if abs(r.value) > 0.01 or r.volatility > 0.01
        ]
        return ExteroceptivePressure(
            pressures=dict(self._smoothed_pressure),
            ambient_stress=self._compute_ambient_stress(),
            active_modalities=active,
            reading_count=0,
        )

    def reset(self) -> None:
        """Clear all smoothed state. Used in testing or on config reload."""
        self._smoothed.clear()
        self._stale_counts.clear()
        self._smoothed_pressure = {d: 0.0 for d in ALL_DIMENSIONS}

    # ─── Internal ─────────────────────────────────────────────────

    def _smooth_reading(self, reading: ExteroceptiveReading) -> None:
        """EMA-smooth a reading against the previous value for its modality."""
        modality = reading.modality
        weight = reading.effective_weight()

        if modality not in self._smoothed or weight == 0.0:
            # First reading or unavailable - seed directly
            self._smoothed[modality] = reading
            self._stale_counts[modality] = 0
            return

        prev = self._smoothed[modality]
        alpha = self._ema_alpha * weight

        smoothed_value = alpha * reading.value + (1.0 - alpha) * prev.value
        smoothed_vol = alpha * reading.volatility + (1.0 - alpha) * prev.volatility

        self._smoothed[modality] = ExteroceptiveReading(
            modality=modality,
            value=smoothed_value,
            volatility=smoothed_vol,
            quality=reading.quality,
            raw_metadata=reading.raw_metadata,
            timestamp=reading.timestamp,
        )
        self._stale_counts[modality] = 0

    def _decay_modality(self, modality: ExteroceptiveModality) -> None:
        """Decay a stale modality's smoothed reading toward zero."""
        if modality not in self._smoothed:
            return

        mapping = self._mappings.get(modality)
        persistence = mapping.persistence if mapping else 0.5
        self._stale_counts[modality] = self._stale_counts.get(modality, 0) + 1

        prev = self._smoothed[modality]
        # Exponential decay: value *= persistence each missed cycle
        decayed_value = prev.value * persistence
        decayed_vol = prev.volatility * persistence

        # If decayed below noise floor, remove entirely
        if abs(decayed_value) < 0.001 and decayed_vol < 0.001:
            del self._smoothed[modality]
            self._stale_counts.pop(modality, None)
            return

        from .types import ReadingQuality
        self._smoothed[modality] = ExteroceptiveReading(
            modality=modality,
            value=decayed_value,
            volatility=decayed_vol,
            quality=ReadingQuality.STALE,
            raw_metadata=prev.raw_metadata,
            timestamp=prev.timestamp,
        )

    def _compute_modality_pressure(
        self,
        reading: ExteroceptiveReading,
        mapping: ModalityMapping,
    ) -> dict[InteroceptiveDimension, float]:
        """Compute per-dimension pressure from a single modality reading.

        Two components:
          1. Value-directional: reading.value * dimension_weight
             (e.g., positive crypto → positive valence)
          2. Volatility-magnitude: reading.volatility * volatility_weight
             (e.g., high volatility → arousal up, regardless of direction)
        """
        pressure: dict[InteroceptiveDimension, float] = {}
        cap = mapping.max_pressure_per_dim

        for dim, weight in mapping.dimension_weights.items():
            # Value contribution: direction matters
            delta = reading.value * weight
            pressure[dim] = pressure.get(dim, 0.0) + delta

        for dim, weight in mapping.volatility_weights.items():
            # Volatility contribution: always additive in the weight's direction
            delta = reading.volatility * weight
            pressure[dim] = pressure.get(dim, 0.0) + delta

        # Clamp each dimension to this modality's cap
        for dim in pressure:
            pressure[dim] = _clamp(pressure[dim], -cap, cap)

        return pressure

    def _compute_ambient_stress(self) -> float:
        """Compute a scalar [0, 1] summary of external weather.

        Uses a soft sigmoid over total absolute pressure so the
        ambient_stress value transitions smoothly even as individual
        dimensions shift.
        """
        total_abs = sum(abs(v) for v in self._smoothed_pressure.values())
        # Sigmoid: 0 pressure → 0.0, max_total_pressure * 9 → ~1.0
        # In practice, total_abs rarely exceeds ~1.0 given per-dim clamping.
        # k=4 maps 0.5 total → ~0.5 ambient
        return _sigmoid(total_abs, k=4.0)


# ─── Utility ──────────────────────────────────────────────────────


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sigmoid(x: float, k: float = 4.0) -> float:
    """Logistic sigmoid mapping [0, ∞) → [0, 1), centred at x=0.5/k."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
