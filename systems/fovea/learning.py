"""
Fovea -- Attention Weight Learner (Phase C)

The mechanism by which Fovea learns which error types are actually informative
for THIS specific EOS instance.

When a high-salience error leads to a significant world model update:
    reinforce the dominant error type's weight (scaled by update magnitude).

When a high-salience error leads to NO world model update (false alarm):
    decay the dominant error type's weight.

Over time, instances diverge in attentional profiles: a social EOS weights
source_error and category_error heavily; a technical EOS weights causal_error.
The divergence emerges from experience, not configuration.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from .types import (
    DEFAULT_ERROR_WEIGHTS,
    ErrorType,
    FoveaPredictionError,
)

if TYPE_CHECKING:
    from .habituation import HabituationEngine

logger = structlog.get_logger("systems.fovea.learning")

# How long (seconds) to wait for a world model update after a prediction error
# before classifying it as a false alarm
_CORRELATION_WINDOW_S: float = 10.0

# Maximum number of recent errors to track for correlation
_RECENT_ERROR_BUFFER_SIZE: int = 200

# Weight bounds
_WEIGHT_FLOOR: float = 0.01
_WEIGHT_CEILING: float = 0.60

# Minimum salience to consider an error for weight learning
_LEARNING_SALIENCE_THRESHOLD: float = 0.1


class _TrackedError:
    """A recent prediction error awaiting correlation with a world model update."""

    __slots__ = (
        "error_id",
        "percept_id",
        "signature",
        "dominant_type",
        "salience",
        "timestamp_s",
        "resolved",
    )

    def __init__(
        self,
        error_id: str,
        percept_id: str,
        signature: str,
        dominant_type: ErrorType,
        salience: float,
        timestamp_s: float,
    ) -> None:
        self.error_id = error_id
        self.percept_id = percept_id
        self.signature = signature
        self.dominant_type = dominant_type
        self.salience = salience
        self.timestamp_s = timestamp_s
        self.resolved = False


class AttentionWeightLearner:
    """
    Learns which error types are actually worth attending to for this EOS instance.

    Maintains the learned weight vector that replaces the spec's DEFAULT_ERROR_WEIGHTS.
    The vector starts at defaults and drifts based on which error types lead to
    genuine world model updates vs false alarms.

    The learned weights are applied to every FoveaPredictionError via
    ``apply_learned_weights(error)``.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        false_alarm_decay: float = 0.005,
    ) -> None:
        self._weights: dict[str, float] = dict(DEFAULT_ERROR_WEIGHTS)
        self._learning_rate = learning_rate
        self._false_alarm_decay = false_alarm_decay

        # Recent unresolved prediction errors awaiting correlation
        self._recent_errors: deque[_TrackedError] = deque(
            maxlen=_RECENT_ERROR_BUFFER_SIZE
        )

        # Optional habituation engine — wired post-construction so habituation
        # record_update() is called when a world model update is correlated.
        # Without this, times_led_to_update stays 0 for all signatures, causing
        # the habituation-complete diagnosis to always read "learning_failure"
        # rather than "stochastic" for genuinely irreducible domains.
        self._habituation_engine: HabituationEngine | None = None

        # Monotonic clock for correlation (seconds since service start)
        self._clock_s: float = 0.0

        # Metrics
        self._reinforcements: int = 0
        self._decays: int = 0
        self._false_alarms: int = 0

        self._logger = logger.bind(component="weight_learner")

    def set_habituation_engine(self, engine: HabituationEngine) -> None:
        """
        Wire the habituation engine so world model updates propagate
        ``record_update()`` calls to correct the stochastic/learning_failure
        diagnosis in habituation-complete events.
        """
        self._habituation_engine = engine

    @property
    def weights(self) -> dict[str, float]:
        """Current learned weight vector (read-only copy)."""
        return dict(self._weights)

    @property
    def reinforcements(self) -> int:
        return self._reinforcements

    @property
    def decays(self) -> int:
        return self._decays

    @property
    def false_alarms(self) -> int:
        return self._false_alarms

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def advance_clock(self, elapsed_s: float) -> None:
        """Advance the internal clock by *elapsed_s* seconds."""
        self._clock_s += elapsed_s

    def set_clock(self, now_s: float) -> None:
        """Set the internal clock to an absolute value (for event-driven use)."""
        self._clock_s = now_s

    # ------------------------------------------------------------------
    # Error registration
    # ------------------------------------------------------------------

    def register_error(self, error: FoveaPredictionError) -> None:
        """
        Register a prediction error for weight learning.

        Only errors above the salience threshold are tracked.
        """
        effective_salience = (
            error.habituated_salience
            if error.habituated_salience > 0
            else error.precision_weighted_salience
        )
        if effective_salience < _LEARNING_SALIENCE_THRESHOLD:
            return

        tracked = _TrackedError(
            error_id=error.id,
            percept_id=error.percept_id,
            signature=error.get_signature(),
            dominant_type=error.get_dominant_error_type(),
            salience=effective_salience,
            timestamp_s=self._clock_s,
        )
        self._recent_errors.append(tracked)

    # ------------------------------------------------------------------
    # World model update correlation
    # ------------------------------------------------------------------

    def on_world_model_updated(self, event_data: dict[str, Any]) -> dict[str, float] | None:
        """
        Called when a WORLD_MODEL_UPDATED event arrives from Logos via Synapse.

        Correlates the update to the most recent unresolved prediction error
        and reinforces that error type's weight. Returns the weight deltas
        if a reinforcement occurred, None otherwise.
        """
        update_magnitude = self._compute_update_magnitude(event_data)
        if update_magnitude <= 0:
            return None

        # Find the most recent unresolved error within the correlation window
        best: _TrackedError | None = None
        for tracked in reversed(self._recent_errors):
            if tracked.resolved:
                continue
            age = self._clock_s - tracked.timestamp_s
            if age > _CORRELATION_WINDOW_S:
                break
            best = tracked
            break

        if best is None:
            return None

        # Reinforce the dominant error type's weight
        best.resolved = True
        dominant = best.dominant_type.value
        old_weight = self._weights.get(dominant, 0.0)
        delta = self._learning_rate * update_magnitude * best.salience
        new_weight = min(old_weight + delta, _WEIGHT_CEILING)
        self._weights[dominant] = new_weight
        self._reinforcements += 1

        self._normalise_weights()

        weight_deltas = {dominant: self._weights[dominant] - old_weight}

        # Notify habituation engine so times_led_to_update is accurate.
        # Without this, habituation-complete diagnosis always reads "learning_failure"
        # for signatures that do lead to world model updates.
        if self._habituation_engine is not None:
            self._habituation_engine.record_update(best.signature, update_magnitude)

        self._logger.info(
            "weight_reinforced",
            dominant=dominant,
            old=round(old_weight, 4),
            new=round(self._weights[dominant], 4),
            update_magnitude=round(update_magnitude, 4),
            error_salience=round(best.salience, 4),
        )

        return weight_deltas

    def flush_stale_errors(self) -> list[ErrorType]:
        """
        Check for unresolved errors that have expired (no world model update
        within the correlation window). These are false alarms: decay their
        dominant error type's weight.

        Returns the list of error types that were decayed.
        """
        decayed_types: list[ErrorType] = []
        cutoff = self._clock_s - _CORRELATION_WINDOW_S

        for tracked in self._recent_errors:
            if tracked.resolved:
                continue
            if tracked.timestamp_s > cutoff:
                continue

            tracked.resolved = True
            dominant = tracked.dominant_type.value
            old_weight = self._weights.get(dominant, 0.0)
            new_weight = max(old_weight - self._false_alarm_decay, _WEIGHT_FLOOR)
            self._weights[dominant] = new_weight
            self._decays += 1
            self._false_alarms += 1
            decayed_types.append(tracked.dominant_type)

            self._logger.debug(
                "weight_decayed_false_alarm",
                dominant=dominant,
                old=round(old_weight, 4),
                new=round(new_weight, 4),
                error_id=tracked.error_id,
            )

        if decayed_types:
            self._normalise_weights()

        return decayed_types

    # ------------------------------------------------------------------
    # Weight application
    # ------------------------------------------------------------------

    def apply_learned_weights(self, error: FoveaPredictionError) -> None:
        """
        Apply the current learned weight vector to a prediction error.

        Called during the processing pipeline, after error computation
        but before salience computation.
        """
        error.error_weights = dict(self._weights)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalise_weights(self) -> None:
        """
        Normalise weights to sum to 1.0, respecting floor constraints.

        This ensures the total salience budget is constant: learning
        redistributes attention across error types rather than inflating it.
        """
        total = sum(self._weights.values())
        if total <= 0:
            self._weights = dict(DEFAULT_ERROR_WEIGHTS)
            return

        for key in self._weights:
            self._weights[key] /= total
            if self._weights[key] < _WEIGHT_FLOOR:
                self._weights[key] = _WEIGHT_FLOOR

        # Re-normalise after floor enforcement
        total = sum(self._weights.values())
        for key in self._weights:
            self._weights[key] /= total

    @staticmethod
    def _compute_update_magnitude(event_data: dict[str, Any]) -> float:
        """
        Derive a [0, 1] update magnitude from the WORLD_MODEL_UPDATED payload.

        Higher magnitude = more significant world model change = stronger
        weight reinforcement.
        """
        schemas_added = float(event_data.get("schemas_added", 0))
        priors_updated = float(event_data.get("priors_updated", 0))
        causal_updates = float(event_data.get("causal_updates", 0))

        # Schema creation most significant (ontology expansion),
        # causal updates next (structural learning),
        # prior updates least (parameter refinement)
        raw = schemas_added * 0.5 + causal_updates * 0.3 + priors_updated * 0.1

        return min(raw / 2.0, 1.0)
