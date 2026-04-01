"""
EcodiaOS - Soma Emergent Emotion Detector

Detects emergent emotions as regions in allostatic error space.
Emotions are NOT classified - they are regions that the organism's
current error state overlaps with. Multiple emotions can be active
simultaneously, each with its own intensity.

The 9 canonical emotion regions are defined in types.EMOTION_REGIONS.
Evo can refine boundaries via SomaService.update_emotion_regions().

The Voxis integration interface returns raw interoceptive state plus
detected emotion regions. Voxis learns its own expression mapping -
the mapping from felt state to voice/style is never hardcoded here.

Budget: <=0.2ms per cycle (simple pattern matching on 9D vectors).
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.soma.types import (
    ALL_DIMENSIONS,
    EMOTION_REGIONS,
    InteroceptiveDimension,
    InteroceptiveState,
)

logger = structlog.get_logger("systems.soma.emotions")

# Error direction thresholds for region matching.
# These define what "positive error", "negative error", "near_zero" mean
# in terms of the allostatic error magnitude.
_NEAR_ZERO_THRESHOLD: float = 0.08
_SIGNIFICANT_THRESHOLD: float = 0.10


class ActiveEmotion:
    """An active emotion region with computed intensity."""

    __slots__ = ("name", "intensity", "description", "matching_dimensions")

    def __init__(
        self,
        name: str,
        intensity: float,
        description: str,
        matching_dimensions: list[str],
    ) -> None:
        self.name = name
        self.intensity = intensity
        self.description = description
        self.matching_dimensions = matching_dimensions

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "intensity": round(self.intensity, 4),
            "description": self.description,
            "matching_dimensions": self.matching_dimensions,
            "should_highlight": self.intensity >= 0.5,
        }


class EmotionDetector:
    """
    Detects which emotion regions the organism's current error state occupies.

    Each emotion region is defined by a pattern of error directions across
    interoceptive dimensions. The detector checks the current allostatic
    errors against these patterns and returns a ranked list of active
    emotions with intensity scores.

    NOT a classifier - multiple emotions can be simultaneously active.
    The organism can feel anxious AND curious at the same time if both
    region patterns match.

    Learnable mappings: each emotion can be linked to an Evo hypothesis ID.
    When HYPOTHESIS_CONFIRMED/REFUTED events arrive, the detector updates
    the corresponding region's pattern. Falls back to hardcoded defaults
    when no Evo hypothesis exists for a mapping.
    """

    def __init__(
        self,
        emotion_regions: dict[str, dict[str, Any]] | None = None,
        near_zero_threshold: float = _NEAR_ZERO_THRESHOLD,
        significant_threshold: float = _SIGNIFICANT_THRESHOLD,
    ) -> None:
        self._regions = dict(emotion_regions or EMOTION_REGIONS)
        self._default_regions = dict(emotion_regions or EMOTION_REGIONS)
        self._near_zero = near_zero_threshold
        self._significant = significant_threshold
        # Maps emotion name → Evo hypothesis ID for learnable region updates
        self._emotion_hypothesis_ids: dict[str, str] = {}

    def update_regions(self, updated: dict[str, dict[str, Any]]) -> None:
        """Evo refines emotion region boundaries."""
        self._regions.update(updated)

    def link_hypothesis(self, emotion_name: str, hypothesis_id: str) -> None:
        """Link an emotion region to an Evo hypothesis for learnable adaptation."""
        self._emotion_hypothesis_ids[emotion_name] = hypothesis_id

    def on_hypothesis_confirmed(self, hypothesis_id: str, updated_pattern: dict[str, str] | None = None) -> None:
        """An Evo hypothesis was confirmed - reinforce the linked emotion region.

        If updated_pattern is provided, apply it to the linked region.
        Otherwise keep the current pattern (confirmation = current is good).
        """
        for emotion_name, hyp_id in self._emotion_hypothesis_ids.items():
            if hyp_id == hypothesis_id and updated_pattern and emotion_name in self._regions:
                self._regions[emotion_name]["pattern"] = updated_pattern
                logger.info("emotion_region_confirmed", emotion=emotion_name, hypothesis_id=hypothesis_id)

    def on_hypothesis_refuted(self, hypothesis_id: str) -> None:
        """An Evo hypothesis was refuted - revert to hardcoded default for the linked region."""
        for emotion_name, hyp_id in self._emotion_hypothesis_ids.items():
            if hyp_id == hypothesis_id and emotion_name in self._default_regions:
                self._regions[emotion_name] = dict(self._default_regions[emotion_name])
                logger.info("emotion_region_reverted", emotion=emotion_name, hypothesis_id=hypothesis_id)

    def detect(
        self,
        state: InteroceptiveState,
    ) -> list[ActiveEmotion]:
        """
        Detect active emotion regions from the current allostatic error state.

        Uses moment-horizon errors and error rates. Returns a list of
        ActiveEmotion sorted by intensity descending. Empty list means
        no emotion pattern matches strongly enough.

        Budget: <=0.2ms (pure dict lookups and comparisons).
        """
        moment_errors = state.errors.get("moment", {})
        error_rates = state.error_rates

        if not moment_errors:
            return []

        active: list[ActiveEmotion] = []

        for emotion_name, region_def in self._regions.items():
            pattern: dict[str, str] = region_def.get("pattern", {})
            if not pattern:
                continue

            match_score, match_count, total_required, matching_dims = (
                self._match_pattern(pattern, moment_errors, error_rates)
            )

            if total_required > 0 and match_count >= total_required:
                # Intensity = average match strength across pattern dimensions
                intensity = match_score / total_required if total_required > 0 else 0.0
                active.append(ActiveEmotion(
                    name=emotion_name,
                    intensity=min(1.0, intensity),
                    description=region_def.get("description", ""),
                    matching_dimensions=matching_dims,
                ))

        # Sort by intensity descending
        active.sort(key=lambda e: e.intensity, reverse=True)
        return active

    def _match_pattern(
        self,
        pattern: dict[str, str],
        errors: dict[InteroceptiveDimension, float],
        error_rates: dict[InteroceptiveDimension, float],
    ) -> tuple[float, int, int, list[str]]:
        """
        Match a pattern against current errors and rates.

        Returns (total_score, matches, required, matching_dim_names).
        """
        total_score = 0.0
        match_count = 0
        total_required = 0
        matching_dims: list[str] = []

        for dim_key, direction in pattern.items():
            # Special key: _any means "any dimension matches this direction"
            if dim_key == "_any":
                score, matched_dim = self._match_any(direction, errors, error_rates)
                if score > 0:
                    total_score += score
                    match_count += 1
                    if matched_dim:
                        matching_dims.append(matched_dim)
                total_required += 1
                continue

            # Normal dimension key
            try:
                dim = InteroceptiveDimension(dim_key)
            except ValueError:
                continue

            total_required += 1
            error = errors.get(dim, 0.0)
            rate = error_rates.get(dim, 0.0)

            score = self._score_direction(direction, error, rate)
            if score > 0:
                total_score += score
                match_count += 1
                matching_dims.append(dim_key)

        return total_score, match_count, total_required, matching_dims

    def _score_direction(
        self,
        direction: str,
        error: float,
        rate: float,
    ) -> float:
        """
        Score how well an error value matches a direction constraint.

        Returns 0.0 if no match, or a positive score proportional to
        how strongly the dimension matches the direction.
        """
        if direction == "positive":
            # Error is positive (overshoot: predicted > setpoint)
            if error > self._significant:
                return abs(error)
            return 0.0

        if direction == "negative":
            # Error is negative (undershoot: predicted < setpoint)
            if error < -self._significant:
                return abs(error)
            return 0.0

        if direction == "near_zero":
            # Error is close to zero (tracking setpoint)
            if abs(error) < self._near_zero:
                return 1.0 - abs(error) / self._near_zero
            return 0.0

        if direction == "near_zero_or_positive":
            # Either tracking setpoint or slightly above
            if error > -self._significant:
                if abs(error) < self._near_zero:
                    return 1.0 - abs(error) / self._near_zero
                if error > 0:
                    return min(1.0, error)
                return 0.5  # Marginal match: not negative
            return 0.0

        if direction == "negative_rate":
            # Error rate is negative (error is shrinking / resolving)
            if rate < -self._significant:
                return abs(rate)
            return 0.0

        if direction == "positive_rate":
            # Error rate is positive (error is growing / worsening)
            if rate > self._significant:
                return abs(rate)
            return 0.0

        return 0.0

    def _match_any(
        self,
        direction: str,
        errors: dict[InteroceptiveDimension, float],
        error_rates: dict[InteroceptiveDimension, float],
    ) -> tuple[float, str]:
        """
        Match "_any" pattern: at least one dimension matches the direction.
        Returns (best_score, best_dim_name) or (0.0, "").
        """
        best_score = 0.0
        best_dim = ""

        for dim in ALL_DIMENSIONS:
            error = errors.get(dim, 0.0)
            rate = error_rates.get(dim, 0.0)
            score = self._score_direction(direction, error, rate)
            if score > best_score:
                best_score = score
                best_dim = dim.value

        return best_score, best_dim


class VoxisInteroceptiveInterface:
    """
    Interface for Voxis to read the organism's felt state.

    Voxis calls get_interoceptive_report() to obtain:
      - Raw 9D sensed state
      - Current allostatic errors
      - Precision weights (where attention is focused)
      - Active emotion regions with intensities
      - Phase-space context (attractor, trajectory)

    Voxis then learns its own mapping from this report to vocal expression.
    The mapping is NEVER hardcoded here - Soma provides the raw felt state,
    Voxis decides how to express it.
    """

    def __init__(
        self,
        emotion_detector: EmotionDetector,
    ) -> None:
        self._detector = emotion_detector

    def get_interoceptive_report(
        self,
        state: InteroceptiveState,
        phase_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a report of the organism's felt state for Voxis consumption.

        Returns a dict with all the raw interoceptive data Voxis needs
        to shape its vocal expression. No interpretation, no style mapping -
        just the felt reality.
        """
        emotions = self._detector.detect(state)

        return {
            "sensed": {d.value: v for d, v in state.sensed.items()},
            "moment_errors": {
                d.value: v
                for d, v in state.errors.get("moment", {}).items()
            },
            "error_rates": {d.value: v for d, v in state.error_rates.items()},
            "precision_weights": {d.value: v for d, v in state.precision.items()},
            "urgency": state.urgency,
            "dominant_error": state.dominant_error.value,
            "temporal_dissonance": {
                d.value: v for d, v in state.temporal_dissonance.items()
            },
            "active_emotions": [e.to_dict() for e in emotions],
            "phase_space": phase_snapshot,
        }
