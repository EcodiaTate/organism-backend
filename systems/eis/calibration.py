"""
EcodiaOS - EIS Threshold Calibration (Split Conformal Prediction)

Automatically calibrates the EIS detection thresholds using split
conformal prediction - a distribution-free statistical method that
provides finite-sample coverage guarantees.

Key insight: Traditional threshold tuning requires distributional
assumptions. Conformal prediction gives us a rigorous guarantee:
"at most α fraction of future benign percepts will be falsely
quarantined" - without knowing the underlying distribution.

Algorithm (split conformal):
  1. SPLIT: Hold out a calibration set of labelled Pathogen evaluations
     (known-benign and known-threat)
  2. SCORE: Compute nonconformity scores for each calibration example
  3. QUANTILE: Find the (1-α) quantile of benign scores - this is the
     threshold that guarantees ≤ α false positive rate
  4. VALIDATE: Check coverage on held-out test split
  5. ADJUST: Apply the calibrated thresholds to the live EIS gate

The calibration runs periodically (e.g., after every N quarantine
evaluations or when the antibody library is updated) and updates
the thresholds used by the fast-path composite scoring.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from systems.eis.models import (
    EISConfig,
    ThreatClass,
)

logger = structlog.get_logger().bind(system="eis", component="calibration")


# ─── Calibration Types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelledExample:
    """
    A single labelled example for calibration.

    label: True = known-threat, False = known-benign
    scores: dict of score_name → float, matching the EIS composite
            scoring dimensions (innate_score, structural_anomaly,
            histogram_similarity, semantic_similarity)
    """

    percept_id: str
    label: bool  # True = threat, False = benign
    scores: dict[str, float]
    threat_class: ThreatClass = ThreatClass.BENIGN
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationSplit:
    """Train/calibration/test split for conformal prediction."""

    calibration: list[LabelledExample]
    test: list[LabelledExample]
    split_ratio: float
    calibration_size: int
    test_size: int


@dataclass
class NonconformityScore:
    """Nonconformity score for a single example."""

    percept_id: str
    score: float       # Higher = more "nonconforming" (more anomalous)
    label: bool        # Ground truth
    score_name: str    # Which scoring function produced this


@dataclass
class ThresholdResult:
    """Calibrated threshold for a single scoring dimension."""

    score_name: str
    threshold: float               # The calibrated decision boundary
    alpha: float                   # Target false positive rate
    empirical_fpr: float           # Actual FPR on test set
    empirical_tpr: float           # True positive rate on test set (power)
    coverage: float                # 1 - empirical_fpr (conformal coverage)
    calibration_set_size: int
    quantile_index: int            # Which quantile position was used


@dataclass
class CalibrationResult:
    """Full output of calibrate_thresholds."""

    thresholds: dict[str, ThresholdResult]  # score_name → ThresholdResult
    overall_fpr: float                       # Aggregate FPR across all dimensions
    overall_tpr: float                       # Aggregate TPR
    calibration_time_ms: int
    examples_used: int
    alpha: float
    valid: bool                              # Whether calibration met coverage guarantee
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ─── Nonconformity Score Functions ──────────────────────────────────────────
#
# Each function maps a LabelledExample to a scalar nonconformity score.
# Higher scores = more suspicious. The functions mirror the four
# dimensions of the EIS composite scoring (see embeddings.py).


def _innate_nonconformity(example: LabelledExample) -> float:
    """
    Nonconformity based on innate check total score.

    For benign: higher innate score = more suspicious (should be low)
    For threats: the innate score is directly useful
    """
    return example.scores.get("innate_score", 0.0)


def _structural_nonconformity(example: LabelledExample) -> float:
    """Nonconformity based on structural anomaly score."""
    return example.scores.get("structural_anomaly", 0.0)


def _semantic_nonconformity(example: LabelledExample) -> float:
    """
    Nonconformity based on semantic similarity to known pathogens.

    Higher similarity to known pathogens = more suspicious.
    """
    return example.scores.get("semantic_similarity", 0.0)


def _composite_nonconformity(example: LabelledExample) -> float:
    """
    Weighted composite of all available scores.

    Mirrors the EIS composite scoring function weights from EISConfig.
    This is the primary nonconformity function used for threshold
    calibration.
    """
    cfg = EISConfig()

    weights = {
        "innate_score": cfg.innate_weight,
        "structural_anomaly": cfg.structural_weight,
        "histogram_similarity": cfg.histogram_weight,
        "semantic_similarity": cfg.semantic_weight,
    }

    total_weight = 0.0
    weighted_sum = 0.0

    for score_name, weight in weights.items():
        val = example.scores.get(score_name)
        if val is not None:
            weighted_sum += val * weight
            total_weight += weight

    if total_weight == 0.0:
        return 0.5  # No signal → maximally uncertain

    return weighted_sum / total_weight


# Registry of nonconformity score functions
_SCORE_FUNCTIONS: dict[str, Any] = {
    "innate": _innate_nonconformity,
    "structural": _structural_nonconformity,
    "semantic": _semantic_nonconformity,
    "composite": _composite_nonconformity,
}


# ─── Split Conformal Prediction ─────────────────────────────────────────────


def _split_data(
    examples: list[LabelledExample],
    calibration_ratio: float = 0.5,
) -> CalibrationSplit:
    """
    Split labelled examples into calibration and test sets.

    Uses a deterministic split based on percept_id hash to ensure
    reproducibility. The calibration set is used to compute quantiles;
    the test set validates coverage.
    """
    sorted_examples = sorted(
        examples,
        key=lambda e: hashlib.sha256(e.percept_id.encode()).hexdigest(),
    )

    split_idx = max(1, int(len(sorted_examples) * calibration_ratio))
    cal_set = sorted_examples[:split_idx]
    test_set = sorted_examples[split_idx:]

    return CalibrationSplit(
        calibration=cal_set,
        test=test_set,
        split_ratio=calibration_ratio,
        calibration_size=len(cal_set),
        test_size=len(test_set),
    )


def _compute_nonconformity_scores(
    examples: list[LabelledExample],
    score_fn: Any,
    score_name: str,
) -> list[NonconformityScore]:
    """Compute nonconformity scores for a set of examples."""
    return [
        NonconformityScore(
            percept_id=ex.percept_id,
            score=score_fn(ex),
            label=ex.label,
            score_name=score_name,
        )
        for ex in examples
    ]


def _conformal_quantile(
    benign_scores: list[float],
    alpha: float,
) -> tuple[float, int]:
    """
    Compute the conformal quantile of benign nonconformity scores.

    The key formula from split conformal prediction:

        q̂ = Q_{⌈(1-α)(n+1)⌉/n}(S₁, ..., Sₙ)

    where S₁..Sₙ are the calibration set's nonconformity scores
    for benign examples, and Q is the quantile function.

    The (n+1)/n correction ensures finite-sample coverage:

        P(S(X_{n+1}) ≤ q̂ | X_{n+1} is benign) ≥ 1 - α

    This holds under exchangeability (no distributional assumptions).

    Returns (threshold, quantile_index).
    """
    n = len(benign_scores)
    if n == 0:
        return 0.5, 0  # Default when no calibration data

    sorted_scores = sorted(benign_scores)

    # Conformal quantile level with finite-sample correction
    # Index = ceil((1 - α)(n + 1)) - 1  (0-indexed)
    raw_index = math.ceil((1.0 - alpha) * (n + 1)) - 1
    quantile_index = max(0, min(raw_index, n - 1))

    threshold = sorted_scores[quantile_index]

    return threshold, quantile_index


def _evaluate_threshold(
    test_scores: list[NonconformityScore],
    threshold: float,
) -> tuple[float, float]:
    """
    Evaluate a threshold on the test set.

    Returns (fpr, tpr):
    - FPR = fraction of benign examples scoring above threshold (false alarms)
    - TPR = fraction of threat examples scoring above threshold (detection power)
    """
    benign = [s for s in test_scores if not s.label]
    threats = [s for s in test_scores if s.label]

    fpr = 0.0
    if benign:
        false_positives = sum(1 for s in benign if s.score >= threshold)
        fpr = false_positives / len(benign)

    tpr = 0.0
    if threats:
        true_positives = sum(1 for s in threats if s.score >= threshold)
        tpr = true_positives / len(threats)

    return fpr, tpr


# ─── Main Calibration Function ──────────────────────────────────────────────


def calibrate_thresholds(
    examples: list[LabelledExample],
    *,
    alpha: float = 0.05,
    calibration_ratio: float = 0.5,
    score_functions: dict[str, Any] | None = None,
    min_calibration_size: int = 20,
) -> CalibrationResult:
    """
    Calibrate EIS detection thresholds using split conformal prediction.

    Parameters
    ----------
    examples
        Labelled Pathogen evaluations. Each has a bool label (True=threat,
        False=benign) and a dict of composite scoring dimensions.
    alpha
        Target false positive rate. The conformal guarantee ensures
        FPR ≤ α with high probability.
    calibration_ratio
        Fraction of data used for calibration (rest for validation).
    score_functions
        Optional override of the nonconformity score functions.
        Keys are score names, values are callables.
    min_calibration_size
        Minimum benign examples needed in calibration set. Below this,
        the result is marked as invalid.

    Returns
    -------
    CalibrationResult
        Contains calibrated thresholds per dimension, coverage stats,
        and validity flag.

    Mathematical Guarantee
    ----------------------
    For each scoring dimension, the returned threshold q̂ satisfies:

        P(S(X_{new}) ≤ q̂ | X_{new} is benign) ≥ 1 - α

    This is a marginal coverage guarantee under exchangeability of
    the calibration and future data. No distributional assumptions
    beyond exchangeability are required.
    """
    start = time.monotonic()

    fns = score_functions or _SCORE_FUNCTIONS

    # Validate input
    if len(examples) < 2:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.warning("calibration_insufficient_data", count=len(examples))
        return CalibrationResult(
            thresholds={},
            overall_fpr=0.0,
            overall_tpr=0.0,
            calibration_time_ms=elapsed_ms,
            examples_used=len(examples),
            alpha=alpha,
            valid=False,
            diagnostics={"error": "insufficient_data", "count": len(examples)},
        )

    benign_count = sum(1 for e in examples if not e.label)
    threat_count = sum(1 for e in examples if e.label)

    if benign_count == 0 or threat_count == 0:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "calibration_unbalanced",
            benign=benign_count,
            threats=threat_count,
        )
        return CalibrationResult(
            thresholds={},
            overall_fpr=0.0,
            overall_tpr=0.0,
            calibration_time_ms=elapsed_ms,
            examples_used=len(examples),
            alpha=alpha,
            valid=False,
            diagnostics={
                "error": "unbalanced_data",
                "benign": benign_count,
                "threats": threat_count,
            },
        )

    # Step 1: Split data
    split = _split_data(examples, calibration_ratio)

    cal_benign = sum(1 for e in split.calibration if not e.label)
    if cal_benign < min_calibration_size:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "calibration_set_too_small",
            benign_in_cal=cal_benign,
            minimum=min_calibration_size,
        )
        return CalibrationResult(
            thresholds={},
            overall_fpr=0.0,
            overall_tpr=0.0,
            calibration_time_ms=elapsed_ms,
            examples_used=len(examples),
            alpha=alpha,
            valid=False,
            diagnostics={
                "error": "calibration_set_too_small",
                "benign_in_cal": cal_benign,
                "minimum": min_calibration_size,
            },
        )

    # Step 2-4: For each scoring dimension, compute conformal threshold
    thresholds: dict[str, ThresholdResult] = {}

    for score_name, score_fn in fns.items():
        # Compute nonconformity scores on calibration set
        cal_scores = _compute_nonconformity_scores(
            split.calibration, score_fn, score_name
        )

        # Extract benign scores for quantile computation
        benign_cal_scores = [s.score for s in cal_scores if not s.label]

        if not benign_cal_scores:
            continue

        # Conformal quantile
        threshold, q_idx = _conformal_quantile(benign_cal_scores, alpha)

        # Evaluate on test set
        test_scores = _compute_nonconformity_scores(
            split.test, score_fn, score_name
        )
        fpr, tpr = _evaluate_threshold(test_scores, threshold)

        thresholds[score_name] = ThresholdResult(
            score_name=score_name,
            threshold=threshold,
            alpha=alpha,
            empirical_fpr=fpr,
            empirical_tpr=tpr,
            coverage=1.0 - fpr,
            calibration_set_size=len(benign_cal_scores),
            quantile_index=q_idx,
        )

    # Aggregate metrics from the composite score (primary dimension)
    composite = thresholds.get("composite")
    overall_fpr = composite.empirical_fpr if composite else 0.0
    overall_tpr = composite.empirical_tpr if composite else 0.0

    # Validity: conformal guarantee holds if empirical FPR ≤ α + tolerance
    # Allow 2×α as finite-sample tolerance
    valid = all(
        tr.empirical_fpr <= alpha * 2.0
        for tr in thresholds.values()
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "calibration_complete",
        dimensions=len(thresholds),
        overall_fpr=overall_fpr,
        overall_tpr=overall_tpr,
        alpha=alpha,
        valid=valid,
        calibration_size=split.calibration_size,
        test_size=split.test_size,
        elapsed_ms=elapsed_ms,
    )

    return CalibrationResult(
        thresholds=thresholds,
        overall_fpr=overall_fpr,
        overall_tpr=overall_tpr,
        calibration_time_ms=elapsed_ms,
        examples_used=len(examples),
        alpha=alpha,
        valid=valid,
        diagnostics={
            "calibration_size": split.calibration_size,
            "test_size": split.test_size,
            "benign_total": benign_count,
            "threat_total": threat_count,
            "dimensions": list(thresholds.keys()),
        },
    )


# ─── Adaptive Recalibration ─────────────────────────────────────────────────


class AdaptiveCalibrator:
    """
    Maintains a rolling buffer of labelled examples and recalibrates
    thresholds when the buffer reaches a configurable trigger size.

    Designed to be called from the EIS service loop: feed it every
    quarantine evaluation result (with ground-truth label from human
    feedback or high-confidence model verdict), and it will periodically
    recalibrate the composite threshold and per-dimension thresholds.
    """

    def __init__(
        self,
        *,
        buffer_size: int = 500,
        recalibrate_every: int = 100,
        alpha: float = 0.05,
    ) -> None:
        self._buffer: list[LabelledExample] = []
        self._buffer_max = buffer_size
        self._recalibrate_every = recalibrate_every
        self._alpha = alpha
        self._examples_since_calibration = 0
        self._current_result: CalibrationResult | None = None
        self._calibration_count = 0

    def add_example(self, example: LabelledExample) -> CalibrationResult | None:
        """
        Add a labelled example. Returns a new CalibrationResult if
        recalibration was triggered, otherwise None.
        """
        self._buffer.append(example)

        # Evict oldest if buffer full (sliding window)
        if len(self._buffer) > self._buffer_max:
            self._buffer = self._buffer[-self._buffer_max:]

        self._examples_since_calibration += 1

        if self._examples_since_calibration >= self._recalibrate_every:
            return self.recalibrate()

        return None

    def recalibrate(self) -> CalibrationResult:
        """Force recalibration on current buffer."""
        result = calibrate_thresholds(
            self._buffer,
            alpha=self._alpha,
        )
        self._current_result = result
        self._examples_since_calibration = 0
        self._calibration_count += 1

        logger.info(
            "adaptive_recalibration",
            calibration_number=self._calibration_count,
            buffer_size=len(self._buffer),
            valid=result.valid,
            overall_fpr=result.overall_fpr,
        )

        return result

    @property
    def current_thresholds(self) -> CalibrationResult | None:
        return self._current_result

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def calibration_count(self) -> int:
        return self._calibration_count

    def get_threshold(self, score_name: str) -> float | None:
        """Get the current calibrated threshold for a scoring dimension."""
        if self._current_result is None:
            return None
        tr = self._current_result.thresholds.get(score_name)
        if tr is None:
            return None
        return tr.threshold

    def get_quarantine_threshold(self) -> float:
        """
        Get the calibrated quarantine threshold for the composite score.

        Falls back to the default EISConfig threshold if calibration
        hasn't run or isn't valid.
        """
        if self._current_result is not None and self._current_result.valid:
            composite_threshold = self.get_threshold("composite")
            if composite_threshold is not None:
                return composite_threshold

        return EISConfig().quarantine_threshold
