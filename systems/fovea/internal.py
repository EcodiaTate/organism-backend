"""
Fovea -- Internal Prediction Engine (Phase D: Self-Attention)

EOS predicting its own processing and detecting violations.
This is architectural self-awareness, not philosophy.

Four internal error types with distinct routing:
1. constitutional -- predicted high Care but generated low Care
   -> Route to Equor for constitutional drift check
2. competency -- predicted 3 LLM calls but used 8
   -> Route to Evo for self-model update
3. behavioral -- predicted procedure X but selected Y
   -> Route to Thread for narrative coherence check
4. affective -- predicted neutral affect but experienced high arousal
   -> Route to Soma for interoceptive recalibration

Internal errors have precision_multiplier = 3.0: EOS should know itself
better than anything external, so self-prediction violations carry 3x
the information per unit magnitude.
"""

from __future__ import annotations

from typing import Any

import structlog

from primitives.common import new_id, utc_now

from .types import (
    ErrorRoute,
    InternalErrorType,
    InternalPredictionError,
)

logger = structlog.get_logger("systems.fovea.internal")

# Routing table: internal error type -> downstream target
_INTERNAL_ROUTING: dict[str, str] = {
    InternalErrorType.CONSTITUTIONAL: ErrorRoute.EQUOR,
    InternalErrorType.COMPETENCY: ErrorRoute.EVO,
    InternalErrorType.BEHAVIORAL: ErrorRoute.THREAD,
    InternalErrorType.AFFECTIVE: "soma",
}


class SelfPrediction:
    """A prediction about EOS's own upcoming cognitive state."""

    __slots__ = (
        "prediction_id",
        "action_type",
        "internal_error_type",
        "predicted_state",
        "timestamp_s",
        "resolved",
    )

    def __init__(
        self,
        action_type: str,
        internal_error_type: InternalErrorType,
        predicted_state: dict[str, Any],
        timestamp_s: float,
    ) -> None:
        self.prediction_id = new_id()
        self.action_type = action_type
        self.internal_error_type = internal_error_type
        self.predicted_state = predicted_state
        self.timestamp_s = timestamp_s
        self.resolved = False


class InternalPredictionEngine:
    """
    The self-prediction loop for EOS.

    Lifecycle per cognitive action:
    1. Before action: predict(action_type, predicted_state)
    2. Action executes
    3. After action: resolve(prediction_id, actual_state)
       -> returns InternalPredictionError if violated

    All internal errors flow through the same pipeline as external ones
    (precision weighting, habituation, workspace broadcast) but with
    the 3x precision multiplier making them naturally higher salience.
    """

    def __init__(self) -> None:
        self._active: dict[str, SelfPrediction] = {}
        self._predictions_made: int = 0
        self._errors_generated: int = 0
        self._errors_by_type: dict[str, int] = {t.value: 0 for t in InternalErrorType}
        self._logger = logger.bind(component="internal_prediction")

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def predictions_made(self) -> int:
        return self._predictions_made

    @property
    def errors_generated(self) -> int:
        return self._errors_generated

    @property
    def errors_by_type(self) -> dict[str, int]:
        return dict(self._errors_by_type)

    # ------------------------------------------------------------------
    # Pre-action: generate self-prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        action_type: str,
        internal_error_type: InternalErrorType,
        predicted_state: dict[str, Any],
        clock_s: float = 0.0,
    ) -> str:
        """
        Generate a prediction about EOS's own upcoming cognitive state.

        Called before each major cognitive action:
        - policy selection -> BEHAVIORAL
        - expression generation -> CONSTITUTIONAL
        - hypothesis evaluation -> COMPETENCY
        - affect computation -> AFFECTIVE

        Returns the prediction_id for later resolution.
        """
        pred = SelfPrediction(
            action_type=action_type,
            internal_error_type=internal_error_type,
            predicted_state=predicted_state,
            timestamp_s=clock_s,
        )
        self._active[pred.prediction_id] = pred
        self._predictions_made += 1

        self._logger.debug(
            "self_prediction_generated",
            prediction_id=pred.prediction_id,
            action_type=action_type,
            error_type=internal_error_type.value,
        )

        return pred.prediction_id

    # ------------------------------------------------------------------
    # Post-action: resolve self-prediction
    # ------------------------------------------------------------------

    def resolve(
        self,
        prediction_id: str,
        actual_state: dict[str, Any],
    ) -> InternalPredictionError | None:
        """
        Compare the self-prediction to the actual outcome.

        Returns an InternalPredictionError if the prediction was violated,
        None if the prediction was accurate (within tolerance).
        """
        pred = self._active.pop(prediction_id, None)
        if pred is None:
            self._logger.warning(
                "unknown_self_prediction", prediction_id=prediction_id,
            )
            return None

        pred.resolved = True

        error_magnitude = self._compute_state_divergence(
            pred.predicted_state, actual_state
        )

        # Below threshold: prediction was accurate
        if error_magnitude < 0.05:
            self._logger.debug(
                "self_prediction_accurate",
                prediction_id=prediction_id,
                magnitude=round(error_magnitude, 4),
            )
            return None

        # Prediction violated: generate an InternalPredictionError
        route = _INTERNAL_ROUTING.get(pred.internal_error_type.value, ErrorRoute.EVO)

        error_mapping = self._map_to_error_dimensions(
            pred.internal_error_type, error_magnitude
        )

        internal_error = InternalPredictionError(
            id=new_id(),
            percept_id="self:" + pred.action_type,
            prediction_id=prediction_id,
            timestamp=utc_now(),
            internal_error_type=pred.internal_error_type,
            predicted_state=pred.predicted_state,
            actual_state=actual_state,
            content_error=error_mapping.get("content", 0.0),
            temporal_error=error_mapping.get("temporal", 0.0),
            magnitude_error=error_mapping.get("magnitude", 0.0),
            source_error=error_mapping.get("source", 0.0),
            category_error=error_mapping.get("category", 0.0),
            causal_error=error_mapping.get("causal", 0.0),
            routes=[route],
        )

        # Apply the 3x precision multiplier via the override
        internal_error.compute_precision_weighted_salience()

        self._errors_generated += 1
        self._errors_by_type[pred.internal_error_type.value] += 1

        self._logger.info(
            "internal_prediction_error",
            prediction_id=prediction_id,
            action_type=pred.action_type,
            error_type=pred.internal_error_type.value,
            magnitude=round(error_magnitude, 4),
            salience=round(internal_error.precision_weighted_salience, 4),
            route=route,
        )

        return internal_error

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def expire_stale(self, max_age_s: float = 60.0, clock_s: float = 0.0) -> int:
        """Remove self-predictions older than max_age_s. Returns count removed."""
        stale = [
            pid
            for pid, pred in self._active.items()
            if (clock_s - pred.timestamp_s) > max_age_s
        ]
        for pid in stale:
            del self._active[pid]
        return len(stale)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_state_divergence(
        predicted: dict[str, Any],
        actual: dict[str, Any],
    ) -> float:
        """
        Compute a [0, 1] divergence score between predicted and actual states.

        Uses key-overlap + value-match heuristic. Numeric values are compared
        by relative difference; categorical values by equality.
        """
        if not predicted:
            return 0.0

        all_keys = set(predicted.keys()) | set(actual.keys())
        if not all_keys:
            return 0.0

        total_divergence = 0.0
        key_count = len(all_keys)

        for key in all_keys:
            pred_val = predicted.get(key)
            actual_val = actual.get(key)

            if pred_val is None or actual_val is None:
                total_divergence += 1.0
                continue

            if isinstance(pred_val, (int, float)) and isinstance(actual_val, (int, float)):
                max_val = max(abs(pred_val), abs(actual_val), 0.001)
                total_divergence += min(abs(pred_val - actual_val) / max_val, 1.0)
            elif pred_val != actual_val:
                total_divergence += 1.0

        return min(total_divergence / key_count, 1.0)

    @staticmethod
    def _map_to_error_dimensions(
        error_type: InternalErrorType,
        magnitude: float,
    ) -> dict[str, float]:
        """
        Map an internal error type and magnitude to the six external error
        dimensions for pipeline compatibility.

        Each internal error type primarily maps to specific dimensions:
        - constitutional -> category + magnitude (drive alignment drift)
        - competency -> causal + magnitude (resource model wrong)
        - behavioral -> category + content (procedure selection wrong)
        - affective -> magnitude + source (affect model wrong)
        """
        mapping: dict[str, float] = {
            "content": 0.0,
            "temporal": 0.0,
            "magnitude": 0.0,
            "source": 0.0,
            "category": 0.0,
            "causal": 0.0,
        }

        if error_type == InternalErrorType.CONSTITUTIONAL:
            mapping["category"] = magnitude * 0.6
            mapping["magnitude"] = magnitude * 0.4
        elif error_type == InternalErrorType.COMPETENCY:
            mapping["causal"] = magnitude * 0.5
            mapping["magnitude"] = magnitude * 0.5
        elif error_type == InternalErrorType.BEHAVIORAL:
            mapping["category"] = magnitude * 0.5
            mapping["content"] = magnitude * 0.5
        elif error_type == InternalErrorType.AFFECTIVE:
            mapping["magnitude"] = magnitude * 0.7
            mapping["source"] = magnitude * 0.3

        return mapping
