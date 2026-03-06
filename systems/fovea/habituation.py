"""
Fovea — Habituation Engine

Reduces salience of repeated identical errors that haven't led to learning.

The biological analog: you stop hearing the clock ticking not because the
prediction error disappears, but because your brain learns the prediction
error is irreducible and therefore uninformative. It stops signaling.

Critical invariant: if a habituated error SUDDENLY becomes larger, that's
a maximum-salience event. The contrast between expected error magnitude
and actual error magnitude is itself a prediction error. Dis-habituation
is immediate and amplified.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .types import FoveaPredictionError

logger = structlog.get_logger("systems.fovea.habituation")

# How fast habituation accumulates per identical error
_HABITUATION_INCREMENT: float = 0.05

# Maximum habituation level (never fully suppress: 90% max)
_MAX_HABITUATION: float = 0.9

# Threshold for magnitude surprise that triggers dis-habituation
_DISHABITUATION_THRESHOLD: float = 0.5

# Amplification factor when dis-habituating (the surprise-of-surprise)
_DISHABITUATION_AMPLIFICATION: float = 2.0

# How many recent magnitudes to keep for expected-magnitude estimation
_HISTORY_WINDOW: int = 10

# Habituation level above which we check for HABITUATION_COMPLETE
_HABITUATION_COMPLETE_THRESHOLD: float = 0.8


@dataclass
class SignatureStats:
    """Per-error-signature learning statistics.

    Tracks whether a habituated error is genuinely stochastic noise
    or a learning failure that should be escalated.
    """

    times_seen: int = 0
    times_led_to_update: int = 0
    last_update_magnitude: float = 0.0
    habituation_complete_emitted: bool = False


@dataclass
class HabituationCompleteInfo:
    """Returned when an error signature crosses the habituation-complete threshold."""

    signature: str
    habituation_level: float
    times_seen: int = 0
    times_led_to_update: int = 0
    diagnosis: str = "stochastic"  # "stochastic" or "learning_failure"


class HabituationEngine:
    """
    Tracks repeated identical prediction errors and decays their salience.

    Each error has a signature (hash of its error profile). Errors with the
    same signature are grouped. If the same signature keeps appearing without
    leading to world model updates, its habituation level increases and its
    effective salience decreases.

    Dis-habituation: if a habituated error suddenly changes magnitude (the
    clock that was ticking now STOPS), the habituation is reset and the error
    is amplified. This is a prediction error about a prediction error.
    """

    def __init__(self) -> None:
        # error_signature -> list of recent magnitudes
        self._error_history: dict[str, deque[float]] = {}
        # error_signature -> habituation level [0.0, _MAX_HABITUATION]
        self._habituation_levels: dict[str, float] = {}
        # Per-signature learning stats (Phase C)
        self._signature_stats: dict[str, SignatureStats] = {}
        # Counters for metrics
        self._habituated_count: int = 0
        self._dishabituated_count: int = 0
        self._habituation_complete_count: int = 0
        self._logger = logger.bind(component="habituation")

    @property
    def entry_count(self) -> int:
        return len(self._habituation_levels)

    @property
    def habituated_count(self) -> int:
        return self._habituated_count

    @property
    def dishabituated_count(self) -> int:
        return self._dishabituated_count

    def apply_habituation(
        self,
        error: FoveaPredictionError,
    ) -> tuple[float, HabituationCompleteInfo | None, dict[str, float] | None]:
        """
        Apply habituation to the error's precision_weighted_salience.

        Returns (habituated_salience, habituation_complete_info_or_None,
                 dishabituation_info_or_None) and updates the error in place.

        ``dishabituation_info`` carries ``{"expected_magnitude": float,
        "actual_magnitude": float}`` when dis-habituation occurs, else None.
        These are the exact values required for the DISHABITUATION Synapse
        event payload (spec Section IX).

        If habituation crosses the complete threshold (>0.8) and the
        signature has never led to a world model update, returns a
        HabituationCompleteInfo for event emission.
        """
        base_salience = error.precision_weighted_salience
        signature = error.get_signature()

        # Ensure signature stats exist
        if signature not in self._signature_stats:
            self._signature_stats[signature] = SignatureStats()
        stats = self._signature_stats[signature]
        stats.times_seen += 1

        # First encounter with this error profile
        if signature not in self._error_history:
            self._error_history[signature] = deque(maxlen=_HISTORY_WINDOW)
            self._habituation_levels[signature] = 0.0
            self._error_history[signature].append(base_salience)
            error.habituation_level = 0.0
            error.habituated_salience = base_salience
            return base_salience, None, None

        history = self._error_history[signature]
        habituation = self._habituation_levels[signature]

        # Check for dis-habituation: has the magnitude changed significantly?
        if len(history) > 0:
            expected_magnitude = sum(history) / len(history)
            if expected_magnitude > 0.001:
                magnitude_surprise = abs(base_salience - expected_magnitude) / expected_magnitude
            else:
                magnitude_surprise = base_salience

            if magnitude_surprise > _DISHABITUATION_THRESHOLD:
                self._habituation_levels[signature] = 0.0
                self._error_history[signature] = deque(maxlen=_HISTORY_WINDOW)
                self._dishabituated_count += 1
                # Reset habituation-complete flag on dis-habituation
                stats.habituation_complete_emitted = False

                amplified = min(base_salience * _DISHABITUATION_AMPLIFICATION, 1.0)
                error.habituation_level = 0.0
                error.habituated_salience = amplified

                # Capture exact magnitudes for the DISHABITUATION event payload
                # (spec Section IX: {error_signature, expected_magnitude, actual_magnitude})
                dishabituation_info: dict[str, float] = {
                    "expected_magnitude": expected_magnitude,
                    "actual_magnitude": base_salience,
                }

                self._logger.info(
                    "dishabituation",
                    signature=signature,
                    expected=round(expected_magnitude, 4),
                    actual=round(base_salience, 4),
                    amplified=round(amplified, 4),
                )
                return amplified, None, dishabituation_info

        # Normal case: update history and increase habituation
        history.append(base_salience)
        new_habituation = min(habituation + _HABITUATION_INCREMENT, _MAX_HABITUATION)
        self._habituation_levels[signature] = new_habituation
        self._habituated_count += 1

        habituated_salience = base_salience * (1.0 - new_habituation)
        error.habituation_level = new_habituation
        error.habituated_salience = habituated_salience

        self._logger.debug(
            "habituation_applied",
            signature=signature,
            base=round(base_salience, 4),
            habituation=round(new_habituation, 4),
            result=round(habituated_salience, 4),
        )

        # Check for habituation-complete: fully habituated, never led to update
        complete_info: HabituationCompleteInfo | None = None
        if (
            new_habituation > _HABITUATION_COMPLETE_THRESHOLD
            and not stats.habituation_complete_emitted
            and stats.times_led_to_update == 0
        ):
            # Diagnose: if seen many times with zero updates, likely stochastic.
            # If seen few times, possibly a learning failure.
            diagnosis = (
                "stochastic" if stats.times_seen > 20 else "learning_failure"
            )
            complete_info = HabituationCompleteInfo(
                signature=signature,
                habituation_level=new_habituation,
                times_seen=stats.times_seen,
                times_led_to_update=stats.times_led_to_update,
                diagnosis=diagnosis,
            )
            stats.habituation_complete_emitted = True
            self._habituation_complete_count += 1

            self._logger.info(
                "habituation_complete",
                signature=signature,
                habituation=round(new_habituation, 4),
                times_seen=stats.times_seen,
                diagnosis=diagnosis,
            )

        return habituated_salience, complete_info, None

    def get_habituation_level(self, signature: str) -> float:
        """Return current habituation level for an error signature."""
        return self._habituation_levels.get(signature, 0.0)

    def get_signature_stats(self, signature: str) -> SignatureStats | None:
        """Return learning stats for an error signature, or None if unknown."""
        return self._signature_stats.get(signature)

    def record_update(self, signature: str, magnitude: float) -> None:
        """
        Record that a world model update was correlated with this error signature.

        Called by the weight learner when it correlates a WORLD_MODEL_UPDATED
        event back to a specific prediction error. Updates the signature's
        learning history so habituation-complete diagnosis is accurate.
        """
        if signature not in self._signature_stats:
            self._signature_stats[signature] = SignatureStats()
        stats = self._signature_stats[signature]
        stats.times_led_to_update += 1
        stats.last_update_magnitude = magnitude

    @property
    def habituation_complete_count(self) -> int:
        return self._habituation_complete_count

    def prune_stale(self, max_entries: int = 1000) -> int:
        """Remove oldest entries if the tracker exceeds *max_entries*."""
        if len(self._habituation_levels) <= max_entries:
            return 0
        # Remove entries with the lowest habituation (least established)
        sorted_sigs = sorted(
            self._habituation_levels.keys(),
            key=lambda s: self._habituation_levels[s],
        )
        to_remove = len(self._habituation_levels) - max_entries
        for sig in sorted_sigs[:to_remove]:
            del self._habituation_levels[sig]
            self._error_history.pop(sig, None)
            self._signature_stats.pop(sig, None)
        return to_remove
