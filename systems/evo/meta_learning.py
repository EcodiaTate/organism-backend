"""
EcodiaOS — Meta-Learning Engine

Evo learning about HOW it learns. This is the recursive self-improvement core
that makes the organism genuinely adaptive rather than just pattern-matching.

Tracks three meta-dimensions:

1. **Detector Effectiveness** — Which pattern detectors produce hypotheses
   that actually survive testing? If CooccurrenceDetector produces 50
   hypotheses but only 2 survive, its detection threshold needs raising.
   If AffectPatternDetector produces 5 that ALL survive, it should be
   more sensitive.

2. **Learning Rate Adaptation** — How fast should Evo accumulate evidence?
   Too fast → premature conclusions. Too slow → the organism never learns.
   Tracks the ratio of SUPPORTED→INTEGRATED vs SUPPORTED→REFUTED and
   adjusts the evidence threshold dynamically.

3. **Hypothesis Quality Scoring** — Information-theoretic score for each
   hypothesis: how much does believing this hypothesis reduce prediction
   error across the systems that consume Evo's outputs? High-quality
   hypotheses change downstream behaviour measurably.

These meta-parameters are themselves subject to velocity limits — Evo can't
rewrite its own learning rules in one cycle. The meta-learning curve is
intentionally slow (half the velocity of normal parameter tuning) to prevent
oscillatory meta-instability.

Integration: Runs as part of Evo consolidation Phase 6 (self-model update).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel

logger = structlog.get_logger()


# ─── Types ───────────────────────────────────────────────────────────────────


class DetectorStats(EOSBaseModel):
    """Performance metrics for a single pattern detector."""

    detector_name: str
    hypotheses_generated: int = 0
    hypotheses_survived: int = 0  # Reached SUPPORTED or INTEGRATED
    hypotheses_refuted: int = 0
    hypotheses_stale: int = 0  # Archived as stale (no evidence)
    mean_time_to_support_hours: float = 0.0
    # Effectiveness = survived / max(1, generated)
    effectiveness: float = 0.0
    # Recommended sensitivity adjustment: +1 = more sensitive, -1 = less
    sensitivity_adjustment: float = 0.0


class LearningRateState(EOSBaseModel):
    """Adaptive learning rate for evidence accumulation."""

    # Current evidence threshold for TESTING → SUPPORTED
    current_evidence_threshold: float = 3.0
    # Current minimum supporting episodes
    current_min_episodes: int = 10
    # Adaptive multiplier: < 1.0 = learn faster, > 1.0 = learn slower
    threshold_multiplier: float = 1.0
    # Rolling false positive rate (SUPPORTED→REFUTED after integration attempt)
    false_positive_rate: float = 0.0
    # Rolling premature rate (good hypotheses archived as stale before enough evidence)
    premature_archival_rate: float = 0.0
    # Number of cycles used to compute these rates
    observation_cycles: int = 0


class HypothesisQualityScore(EOSBaseModel):
    """Information-theoretic quality score for a hypothesis."""

    hypothesis_id: str
    # Prediction error reduction: how much did downstream systems improve?
    prediction_error_delta: float = 0.0
    # Bits of information this hypothesis encodes (MDL score)
    information_bits: float = 0.0
    # Quality = prediction_error_delta / max(1, information_bits)
    # High quality = maximum prediction reduction per bit of complexity
    quality: float = 0.0


class MetaLearningReport(EOSBaseModel):
    """Result of a meta-learning update cycle."""

    detector_stats: list[DetectorStats] = Field(default_factory=list)
    learning_rate: LearningRateState = Field(default_factory=LearningRateState)
    adjustments_made: int = 0
    # What changed this cycle
    changes: list[str] = Field(default_factory=list)


# ─── Meta-Learning Constants ─────────────────────────────────────────────────

# Minimum observations before adjusting meta-parameters
_MIN_META_OBSERVATIONS: int = 20
# Maximum meta-parameter adjustment per cycle (half of normal velocity)
_MAX_META_DELTA: float = 0.015
# Target false positive rate (SUPPORTED→REFUTED)
_TARGET_FALSE_POSITIVE_RATE: float = 0.10
# Target premature archival rate
_TARGET_PREMATURE_RATE: float = 0.15
# Effectiveness threshold: detectors below this get desensitized
_MIN_DETECTOR_EFFECTIVENESS: float = 0.15
# Effectiveness threshold: detectors above this get sensitized
_HIGH_DETECTOR_EFFECTIVENESS: float = 0.60


# ─── Meta-Learning Engine ────────────────────────────────────────────────────


class MetaLearningEngine:
    """
    Tracks and adapts Evo's own learning parameters.

    Maintains rolling statistics about detector effectiveness, evidence
    accumulation rates, and hypothesis quality. Proposes adjustments to
    Evo's internal thresholds to improve learning efficiency.

    All adjustments are velocity-limited to prevent meta-instability.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(system="evo.meta_learning")

        # ── Detector tracking ──────────────────────────────────────────
        # detector_name → list of (hypothesis_id, outcome)
        # outcome: "supported", "integrated", "refuted", "stale"
        self._detector_outcomes: dict[str, list[str]] = defaultdict(list)
        # detector_name → total hypotheses generated
        self._detector_generation_counts: dict[str, int] = defaultdict(int)

        # ── Learning rate tracking ─────────────────────────────────────
        # Rolling window of (was_false_positive: bool) for recent integrations
        self._integration_outcomes: deque[bool] = deque(maxlen=100)
        # Rolling window of (was_premature_archival: bool) for stale hypotheses
        self._archival_outcomes: deque[bool] = deque(maxlen=100)

        # ── Current meta-state ─────────────────────────────────────────
        self._current_rate = LearningRateState()
        self._total_updates: int = 0

        # ── Hypothesis quality tracking ────────────────────────────────
        # hypothesis_id → pre/post prediction error samples
        self._pre_post_errors: dict[str, tuple[float, float]] = {}

    # ─── Event Recording ──────────────────────────────────────────────────────

    def record_hypothesis_generated(
        self,
        hypothesis_id: str,
        source_detector: str,
    ) -> None:
        """Record that a detector produced a hypothesis."""
        self._detector_generation_counts[source_detector] += 1

    def record_hypothesis_outcome(
        self,
        hypothesis_id: str,
        source_detector: str,
        outcome: str,  # "supported", "integrated", "refuted", "stale"
    ) -> None:
        """Record the outcome of a hypothesis for its source detector."""
        self._detector_outcomes[source_detector].append(outcome)

        # Track integration success/failure for learning rate adaptation
        if outcome == "refuted":
            self._integration_outcomes.append(True)  # False positive
        elif outcome == "integrated":
            self._integration_outcomes.append(False)  # True positive

        if outcome == "stale":
            self._archival_outcomes.append(True)  # Premature archival
        elif outcome in ("supported", "integrated"):
            self._archival_outcomes.append(False)  # Not premature

    def record_prediction_error_change(
        self,
        hypothesis_id: str,
        pre_error: float,
        post_error: float,
    ) -> None:
        """Record prediction error before/after a hypothesis was integrated."""
        self._pre_post_errors[hypothesis_id] = (pre_error, post_error)

    # ─── Meta-Learning Update ─────────────────────────────────────────────────

    def update(self) -> MetaLearningReport:
        """
        Compute meta-learning adjustments.

        Called during Evo consolidation Phase 6 (self-model update).
        Returns a report with proposed adjustments.
        """
        report = MetaLearningReport()

        # ── 1. Detector Effectiveness ──────────────────────────────────
        for detector_name, outcomes in self._detector_outcomes.items():
            generated = self._detector_generation_counts.get(detector_name, 0)
            survived = sum(1 for o in outcomes if o in ("supported", "integrated"))
            refuted = sum(1 for o in outcomes if o == "refuted")
            stale = sum(1 for o in outcomes if o == "stale")

            effectiveness = survived / max(1, generated)

            # Compute sensitivity adjustment
            adj = 0.0
            if generated >= _MIN_META_OBSERVATIONS:
                if effectiveness < _MIN_DETECTOR_EFFECTIVENESS:
                    # Detector is producing too many bad hypotheses → desensitize
                    adj = -_MAX_META_DELTA
                    report.changes.append(
                        f"{detector_name}: effectiveness {effectiveness:.2f} < "
                        f"{_MIN_DETECTOR_EFFECTIVENESS} → desensitize"
                    )
                elif effectiveness > _HIGH_DETECTOR_EFFECTIVENESS:
                    # Detector is very effective → sensitize (it might be missing patterns)
                    adj = _MAX_META_DELTA
                    report.changes.append(
                        f"{detector_name}: effectiveness {effectiveness:.2f} > "
                        f"{_HIGH_DETECTOR_EFFECTIVENESS} → sensitize"
                    )

            stats = DetectorStats(
                detector_name=detector_name,
                hypotheses_generated=generated,
                hypotheses_survived=survived,
                hypotheses_refuted=refuted,
                hypotheses_stale=stale,
                effectiveness=round(effectiveness, 3),
                sensitivity_adjustment=round(adj, 4),
            )
            report.detector_stats.append(stats)

        # ── 2. Learning Rate Adaptation ────────────────────────────────
        total_integration_obs = len(self._integration_outcomes)
        total_archival_obs = len(self._archival_outcomes)

        false_positive_rate = (
            sum(1 for fp in self._integration_outcomes if fp) /
            max(1, total_integration_obs)
        )
        premature_rate = (
            sum(1 for pa in self._archival_outcomes if pa) /
            max(1, total_archival_obs)
        )

        threshold_delta = 0.0
        if total_integration_obs >= _MIN_META_OBSERVATIONS:
            if false_positive_rate > _TARGET_FALSE_POSITIVE_RATE:
                # Too many false positives → raise evidence threshold
                threshold_delta = _MAX_META_DELTA
                report.changes.append(
                    f"FP rate {false_positive_rate:.2f} > target "
                    f"{_TARGET_FALSE_POSITIVE_RATE} → raise evidence threshold"
                )
            elif premature_rate > _TARGET_PREMATURE_RATE:
                # Too many good hypotheses dying prematurely → lower threshold
                threshold_delta = -_MAX_META_DELTA
                report.changes.append(
                    f"Premature archival rate {premature_rate:.2f} > target "
                    f"{_TARGET_PREMATURE_RATE} → lower evidence threshold"
                )

        # Apply velocity-limited adjustment
        new_multiplier = self._current_rate.threshold_multiplier + threshold_delta
        new_multiplier = max(0.5, min(2.0, new_multiplier))  # Clamp to [0.5, 2.0]
        self._current_rate.threshold_multiplier = round(new_multiplier, 4)
        self._current_rate.current_evidence_threshold = round(
            3.0 * new_multiplier, 2
        )
        self._current_rate.current_min_episodes = max(
            5, int(10 * new_multiplier)
        )
        self._current_rate.false_positive_rate = round(false_positive_rate, 3)
        self._current_rate.premature_archival_rate = round(premature_rate, 3)
        self._current_rate.observation_cycles = total_integration_obs

        report.learning_rate = self._current_rate
        report.adjustments_made = sum(1 for c in report.changes)

        self._total_updates += 1

        if report.changes:
            self._logger.info(
                "meta_learning_update",
                changes=report.changes,
                threshold_multiplier=self._current_rate.threshold_multiplier,
                false_positive_rate=round(false_positive_rate, 3),
                premature_rate=round(premature_rate, 3),
            )

        return report

    # ─── Query ────────────────────────────────────────────────────────────────

    def get_effective_evidence_threshold(self) -> float:
        """Return the current adaptive evidence threshold."""
        return self._current_rate.current_evidence_threshold

    def get_effective_min_episodes(self) -> int:
        """Return the current adaptive minimum episode count."""
        return self._current_rate.current_min_episodes

    def get_detector_effectiveness(self, detector_name: str) -> float:
        """Return the effectiveness score for a specific detector."""
        generated = self._detector_generation_counts.get(detector_name, 0)
        if generated == 0:
            return 0.5  # No data → neutral
        outcomes = self._detector_outcomes.get(detector_name, [])
        survived = sum(1 for o in outcomes if o in ("supported", "integrated"))
        return survived / max(1, generated)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_updates": self._total_updates,
            "threshold_multiplier": self._current_rate.threshold_multiplier,
            "evidence_threshold": self._current_rate.current_evidence_threshold,
            "min_episodes": self._current_rate.current_min_episodes,
            "false_positive_rate": self._current_rate.false_positive_rate,
            "premature_archival_rate": self._current_rate.premature_archival_rate,
        }
