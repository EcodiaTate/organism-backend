"""
Fovea — Prediction Engine

Generates predictions from the world model before each processing cycle,
then computes structured prediction errors by comparing prediction to reality.

The key insight: we are always in the middle of a prediction. At any moment
the world model has expectations about what will arrive next. Fovea makes
these expectations explicit so they can be violated.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now

from .types import (
    ActivePrediction,
    FoveaPredictionError,
    PerceptContext,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from primitives.percept import Percept

    from .protocols import LogosWorldModel

logger = structlog.get_logger("systems.fovea.prediction")


# ---------------------------------------------------------------------------
# Vector math (local to avoid circular imports with Atune helpers)
# ---------------------------------------------------------------------------


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine_similarity. Returns 1.0 on degenerate input (maximum surprise)."""
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, 1.0 - similarity))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Prediction Engine
# ---------------------------------------------------------------------------


class FoveaPredictionEngine:
    """
    Generates predictions from the World Model before each processing cycle.

    Lifecycle:
    1. ``generate_prediction(context)`` — ask the world model what it expects
    2. Percept arrives
    3. ``compute_error(prediction, percept)`` — compare expectation to reality
    """

    def __init__(self, world_model: LogosWorldModel) -> None:
        self._world_model = world_model
        self._active_predictions: dict[str, ActivePrediction] = {}
        self._logger = logger.bind(component="prediction_engine")

    @property
    def active_prediction_count(self) -> int:
        return len(self._active_predictions)

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    async def generate_prediction(self, context: PerceptContext) -> ActivePrediction:
        """
        Query the world model for what it expects next in this context.

        Called before a percept is evaluated. The prediction covers all six
        error dimensions independently.
        """
        src = context.source_system
        ctx = context.metadata

        # Query world model for each dimension (tolerant of failures)
        expected_content = await self._safe_call(
            self._world_model.predict_content(src, ctx), []
        )
        expected_timing = await self._safe_call(
            self._world_model.predict_timing(src, ctx), 0.0
        )
        expected_magnitude = await self._safe_call(
            self._world_model.predict_magnitude(src, ctx), 0.5
        )
        expected_source = await self._safe_call(
            self._world_model.predict_source(src, ctx), src
        )
        expected_category = await self._safe_call(
            self._world_model.predict_category(src, ctx), ""
        )
        expected_causal = await self._safe_call(
            self._world_model.predict_causal_context(src, ctx), {}
        )
        confidence = await self._safe_call(
            self._world_model.get_prediction_confidence(src, ctx), 0.5
        )

        prediction = ActivePrediction(
            prediction_id=new_id(),
            context=context,
            timestamp=utc_now(),
            expected_content=expected_content,
            expected_timing=expected_timing,
            expected_magnitude=expected_magnitude,
            expected_source=expected_source,
            expected_category=expected_category,
            expected_causal_context=expected_causal,
            prediction_confidence=confidence,
        )

        self._active_predictions[prediction.prediction_id] = prediction
        return prediction

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------

    async def compute_error(
        self,
        prediction: ActivePrediction,
        percept: Percept,
    ) -> FoveaPredictionError:
        """
        The core Fovea operation. Compare prediction to reality.
        Return a structured, six-dimensional error decomposition.
        """
        # Content error: embedding distance between predicted and actual content
        actual_embedding = percept.content.embedding or []
        content_error = _cosine_distance(prediction.expected_content, actual_embedding)

        # Temporal error: how far off was the timing prediction?
        temporal_error = self._compute_temporal_error(prediction, percept)

        # Magnitude error: relative difference in expected intensity
        actual_magnitude = percept.salience_hint if percept.salience_hint is not None else 0.5
        magnitude_error = self._compute_magnitude_error(
            prediction.expected_magnitude, actual_magnitude
        )

        # Source error: did the percept come from the expected source?
        source_error = self._compute_source_error(
            prediction.expected_source, percept.source.system
        )

        # Category error: does the percept match the expected type?
        actual_category = percept.metadata.get("category", percept.source.modality)
        category_error = self._compute_category_error(
            prediction.expected_category, str(actual_category)
        )

        # Causal error: does the causal context match?
        actual_causal = percept.metadata.get("causal_context", {})
        causal_error = self._compute_causal_error(
            prediction.expected_causal_context, actual_causal
        )

        error = FoveaPredictionError(
            percept_id=percept.id,
            prediction_id=prediction.prediction_id,
            timestamp=utc_now(),
            content_error=content_error,
            temporal_error=temporal_error,
            magnitude_error=magnitude_error,
            source_error=source_error,
            category_error=category_error,
            causal_error=causal_error,
        )

        # Clean up the prediction now that it's been resolved
        self._active_predictions.pop(prediction.prediction_id, None)

        self._logger.debug(
            "prediction_error_computed",
            percept_id=percept.id,
            prediction_id=prediction.prediction_id,
            content=round(content_error, 4),
            temporal=round(temporal_error, 4),
            magnitude=round(magnitude_error, 4),
            source=round(source_error, 4),
            category=round(category_error, 4),
            causal=round(causal_error, 4),
        )

        return error

    def expire_stale_predictions(self, max_age_seconds: float = 30.0) -> int:
        """Remove predictions older than *max_age_seconds*. Returns count removed."""
        now = utc_now()
        stale = [
            pid
            for pid, pred in self._active_predictions.items()
            if (now - pred.timestamp).total_seconds() > max_age_seconds
        ]
        for pid in stale:
            del self._active_predictions[pid]
        return len(stale)

    # ------------------------------------------------------------------
    # Error dimension computers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_temporal_error(prediction: ActivePrediction, percept: Percept) -> float:
        """How far off was the timing prediction?"""
        if prediction.expected_timing <= 0:
            return 0.0  # No timing prediction made
        actual_delay = (percept.timestamp - prediction.timestamp).total_seconds()
        if actual_delay < 0:
            actual_delay = 0.0
        # Normalise: ratio of difference to expected, capped at 1.0
        ratio = abs(actual_delay - prediction.expected_timing) / max(
            prediction.expected_timing, 0.001
        )
        return _clamp(ratio)

    @staticmethod
    def _compute_magnitude_error(expected: float, actual: float) -> float:
        """Relative difference in expected vs actual intensity."""
        return _clamp(abs(expected - actual) / max(expected, 0.001))

    @staticmethod
    def _compute_source_error(expected_source: str, actual_source: str) -> float:
        """Binary mismatch (for now). 0.0 if match, 1.0 if different."""
        if not expected_source or not actual_source:
            return 0.0
        return 0.0 if expected_source == actual_source else 1.0

    @staticmethod
    def _compute_category_error(expected_category: str, actual_category: str) -> float:
        """Binary mismatch. 0.0 if match, 1.0 if different, 0.0 if no prediction."""
        if not expected_category:
            return 0.0  # No category prediction made
        return 0.0 if expected_category == actual_category else 1.0

    @staticmethod
    def _compute_causal_error(
        expected_causal: dict[str, Any], actual_causal: Any
    ) -> float:
        """
        Compare causal contexts. Uses key-overlap heuristic for dict contexts.

        A more sophisticated implementation would use embedding distance on
        serialised causal graphs once Kairos provides structured causal models.
        """
        if not expected_causal:
            return 0.0
        if not isinstance(actual_causal, dict):
            return 0.5  # Can't compare — moderate surprise

        expected_keys = set(expected_causal.keys())
        actual_keys = set(actual_causal.keys())
        if not expected_keys:
            return 0.0

        overlap = expected_keys & actual_keys
        structural_match = len(overlap) / len(expected_keys)

        # Check value agreement on overlapping keys
        value_agreement = 0.0
        if overlap:
            matches = sum(
                1 for k in overlap if expected_causal[k] == actual_causal.get(k)
            )
            value_agreement = matches / len(overlap)

        combined_match = 0.5 * structural_match + 0.5 * value_agreement
        return _clamp(1.0 - combined_match)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _safe_call(coro: Any, default: Any) -> Any:  # noqa: ANN401
        """Call an async method, returning *default* if it raises."""
        try:
            return await coro
        except Exception:
            return default
