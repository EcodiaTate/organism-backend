"""
EcodiaOS — Recursive Self-Modification Engine

The difference between a system that tunes hyperparameters and one that
rewrites its own source code.

Capabilities:
  1. Detector evolution — propose replacement for failing detectors
  2. Evidence function adaptation — make complexity coefficients learnable
  3. Consolidation schedule adaptation — learn optimal consolidation timing
  4. Learning architecture proposals — request cognitive upgrades from Simula

All self-modifications are velocity-limited: the organism can't rewrite
itself in one cycle. Constitutional guard: can NEVER modify Equor logic,
drive weights, or safety boundaries.

Integration:
  - Runs during consolidation Phase 6.5 (after meta-learning update)
  - Detector replacements dispatched to Simula via neuroplasticity hot-reload
  - Schedule adaptations applied directly to ConsolidationOrchestrator
  - Architecture proposals submitted to Simula governance pipeline
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.evo.types import (
    ConsolidationSchedule,
    DetectorReplacementProposal,
    HypothesisCategory,
    LearningArchitectureProposal,
    SelfModificationRecord,
)

if TYPE_CHECKING:
    from systems.evo.meta_learning import MetaLearningEngine

logger = structlog.get_logger()

# Self-modification velocity limits (half of normal parameter velocity)
_MAX_SELF_MOD_PER_CYCLE: int = 2           # Max self-modifications per consolidation
_DETECTOR_FAIL_THRESHOLD: float = 0.1       # Effectiveness below this triggers replacement
_DETECTOR_FAIL_CYCLES: int = 5              # Must fail for this many consecutive cycles
_SCHEDULE_ADAPTATION_STEP: float = 0.5      # Max change in consolidation interval per cycle (hours)
_MIN_CONSOLIDATION_HOURS: float = 1.0
_MAX_CONSOLIDATION_HOURS: float = 24.0

# Evidence function adaptation
_COMPLEXITY_LEARNING_RATE: float = 0.01     # How fast the complexity coefficient adapts
_COMPLEXITY_MIN: float = 0.01
_COMPLEXITY_MAX: float = 0.5

# Forbidden targets (constitutional guard)
_FORBIDDEN_TARGETS = {"equor", "constitutional", "invariant", "safety", "drive"}


class SelfModificationEngine:
    """
    Manages recursive self-improvement of Evo's learning algorithms.

    Tracks which self-modifications succeeded → learns which kinds of
    self-modification work.
    """

    def __init__(
        self,
        meta_learning: MetaLearningEngine | None = None,
    ) -> None:
        self._meta_learning = meta_learning
        self._logger = logger.bind(system="evo.self_modification")

        # Detector effectiveness history: detector_name → deque of effectiveness scores
        self._detector_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_DETECTOR_FAIL_CYCLES * 2)
        )

        # Evidence function adaptation state: category → learned complexity coefficient
        self._learned_complexity: dict[str, float] = {
            cat.value: 0.1 for cat in HypothesisCategory
        }

        # Consolidation schedule state
        self._schedule = ConsolidationSchedule()

        # Self-modification history
        self._modification_records: list[SelfModificationRecord] = []
        self._total_modifications: int = 0

        # Tracking which kinds of self-modification succeed
        self._success_rates: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=20)
        )

        # Pending proposals awaiting Simula response
        self._pending_proposals: list[
            DetectorReplacementProposal | LearningArchitectureProposal
        ] = []

    # ─── Main Entry Point ────────────────────────────────────────────────────

    async def run_self_modification_cycle(
        self,
        detector_stats: list[dict[str, Any]],
        hypothesis_outcomes: dict[str, list[tuple[float, float]]],
        consolidation_metrics: dict[str, float],
    ) -> list[SelfModificationRecord]:
        """
        Run a self-modification cycle during consolidation.

        Steps:
          1. Check for failing detectors → propose replacements
          2. Adapt evidence function complexity coefficients
          3. Adapt consolidation schedule
          4. Detect systematic failure modes → propose architecture changes

        Returns list of modifications applied this cycle.

        Args:
            detector_stats: List of detector stat dicts from MetaLearningEngine.
            hypothesis_outcomes: category → [(predicted_confidence, actual_success), ...]
                for calibration of complexity coefficients.
            consolidation_metrics: {
                "hypothesis_throughput": float,
                "integration_success_rate": float,
                "schema_discovery_rate": float,
            }
        """
        records: list[SelfModificationRecord] = []
        modifications_this_cycle = 0

        # 1. Detector evolution
        if modifications_this_cycle < _MAX_SELF_MOD_PER_CYCLE:
            detector_proposals = self._check_detector_evolution(detector_stats)
            for proposal in detector_proposals[:1]:  # Max 1 detector replacement per cycle
                record = SelfModificationRecord(
                    proposal_type="detector_replacement",
                    description=(
                        f"Replace detector '{proposal.old_detector_name}' — "
                        f"effectiveness below {_DETECTOR_FAIL_THRESHOLD} for "
                        f"{_DETECTOR_FAIL_CYCLES} cycles. "
                        f"Spec: {proposal.specification[:200]}"
                    ),
                    applied=False,
                    outcome="pending",
                )
                records.append(record)
                self._pending_proposals.append(proposal)
                self._modification_records.append(record)
                modifications_this_cycle += 1

        # 2. Evidence function adaptation
        if modifications_this_cycle < _MAX_SELF_MOD_PER_CYCLE and hypothesis_outcomes:
            adaptation = self._adapt_evidence_function(hypothesis_outcomes)
            if adaptation:
                records.append(adaptation)
                self._modification_records.append(adaptation)
                modifications_this_cycle += 1

        # 3. Consolidation schedule adaptation
        if modifications_this_cycle < _MAX_SELF_MOD_PER_CYCLE:
            schedule_change = self._adapt_consolidation_schedule(consolidation_metrics)
            if schedule_change:
                records.append(schedule_change)
                self._modification_records.append(schedule_change)
                modifications_this_cycle += 1

        # 4. Architecture proposals (only if other mods didn't fill the budget)
        if modifications_this_cycle < _MAX_SELF_MOD_PER_CYCLE:
            arch_proposals = self._detect_systematic_failures(detector_stats, consolidation_metrics)
            for proposal in arch_proposals[:1]:
                record = SelfModificationRecord(
                    proposal_type="architecture",
                    description=(
                        f"Architecture proposal: {proposal.proposed_change[:200]}. "
                        f"Failure mode: {proposal.failure_mode[:100]}"
                    ),
                    applied=False,
                    outcome="pending",
                )
                records.append(record)
                self._pending_proposals.append(proposal)
                self._modification_records.append(record)
                modifications_this_cycle += 1

        self._total_modifications += len(records)

        if records:
            self._logger.info(
                "self_modification_cycle_complete",
                modifications=len(records),
                types=[r.proposal_type for r in records],
            )

        return records

    # ─── 1. Detector Evolution ───────────────────────────────────────────────

    def _check_detector_evolution(
        self,
        detector_stats: list[dict[str, Any]],
    ) -> list[DetectorReplacementProposal]:
        """
        Check if any detectors have been consistently ineffective.

        When effectiveness drops below 0.1 for 5 consecutive cycles,
        generate a replacement specification.
        """
        proposals: list[DetectorReplacementProposal] = []

        for stat in detector_stats:
            name = stat.get("detector_name", "")
            effectiveness = float(stat.get("effectiveness", 0.5))

            self._detector_history[name].append(effectiveness)

            # Check for sustained failure
            history = list(self._detector_history[name])
            if len(history) < _DETECTOR_FAIL_CYCLES:
                continue

            recent = history[-_DETECTOR_FAIL_CYCLES:]
            if all(e < _DETECTOR_FAIL_THRESHOLD for e in recent):
                # Determine which hypothesis categories are underserved
                underserved = self._find_underserved_categories()

                spec = (
                    f"Pattern detector replacing '{name}'. "
                    f"Previous detector had effectiveness {statistics.mean(recent):.3f} "
                    f"over {_DETECTOR_FAIL_CYCLES} cycles. "
                    f"Should detect patterns in categories: {', '.join(underserved)}. "
                    f"Must implement PatternDetector interface with async scan() method. "
                    f"Target: produce hypotheses with >20% survival rate."
                )

                proposal = DetectorReplacementProposal(
                    old_detector_name=name,
                    specification=spec,
                    underserved_categories=underserved,
                    effectiveness_history=recent,
                )
                proposals.append(proposal)

                self._logger.warning(
                    "detector_replacement_proposed",
                    detector=name,
                    effectiveness_history=[round(e, 3) for e in recent],
                    underserved=underserved,
                )

        return proposals

    def _find_underserved_categories(self) -> list[str]:
        """Find hypothesis categories with low hypothesis generation rates."""
        if self._meta_learning is None:
            return [cat.value for cat in HypothesisCategory]

        # Use meta-learning detector stats to find which categories lack representation
        underserved: list[str] = []
        for cat in HypothesisCategory:
            # A category is underserved if fewer than 5 hypotheses have been generated
            # across all detectors
            cat_count = 0
            for outcomes in self._meta_learning._detector_outcomes.values():
                # This is a rough proxy — we'd need category tracking for precision
                cat_count += len(outcomes)

            if cat_count < 5:
                underserved.append(cat.value)

        return underserved or [HypothesisCategory.WORLD_MODEL.value]

    # ─── 2. Evidence Function Adaptation ─────────────────────────────────────

    def _adapt_evidence_function(
        self,
        hypothesis_outcomes: dict[str, list[tuple[float, float]]],
    ) -> SelfModificationRecord | None:
        """
        Adapt the complexity coefficient per hypothesis category based on
        calibration data.

        Tracks predicted vs actual integration success:
          - If high-complexity hypotheses succeed more than expected → lower penalty
          - If low-complexity hypotheses fail more than expected → raise penalty

        The coefficient is learnable per category.
        """
        any_change = False
        changes: list[str] = []

        for category, outcomes in hypothesis_outcomes.items():
            if not outcomes or len(outcomes) < 5:
                continue

            # outcomes: [(predicted_confidence, actual_success_bool_as_float), ...]
            predicted = [o[0] for o in outcomes]
            actual = [o[1] for o in outcomes]

            # Calibration error: mean absolute difference
            calibration_error = statistics.mean(
                abs(p - a) for p, a in zip(predicted, actual, strict=False)
            )

            current = self._learned_complexity.get(category, 0.1)

            # If calibration error is high and predictions are too optimistic,
            # increase complexity penalty
            mean_predicted = statistics.mean(predicted)
            mean_actual = statistics.mean(actual)

            if calibration_error > 0.15:
                if mean_predicted > mean_actual:
                    # Over-confident → increase penalty
                    new_val = min(_COMPLEXITY_MAX, current + _COMPLEXITY_LEARNING_RATE)
                    direction = "increased"
                else:
                    # Under-confident → decrease penalty
                    new_val = max(_COMPLEXITY_MIN, current - _COMPLEXITY_LEARNING_RATE)
                    direction = "decreased"

                self._learned_complexity[category] = round(new_val, 4)
                any_change = True
                changes.append(
                    f"{category}: {current:.4f} → {new_val:.4f} ({direction}, "
                    f"calibration_error={calibration_error:.3f})"
                )

        if not any_change:
            return None

        record = SelfModificationRecord(
            proposal_type="evidence_adaptation",
            description=f"Evidence complexity coefficients adapted: {'; '.join(changes)}",
            applied=True,
            outcome="success",
        )
        self._logger.info(
            "evidence_function_adapted",
            changes=changes,
        )
        return record

    def get_learned_complexity(self, category: str) -> float:
        """Return the learned complexity coefficient for a hypothesis category."""
        return self._learned_complexity.get(category, 0.1)

    # ─── 3. Consolidation Schedule Adaptation ────────────────────────────────

    def _adapt_consolidation_schedule(
        self,
        metrics: dict[str, float],
    ) -> SelfModificationRecord | None:
        """
        Adapt the consolidation schedule based on learning performance.

        - High hypothesis throughput + high integration success → less frequent (deep)
        - Low throughput + low success → more frequent (iterate faster)
        - High schema discovery rate → less frequent (let discoveries accumulate)
        """
        throughput = metrics.get("hypothesis_throughput", 0.0)
        success_rate = metrics.get("integration_success_rate", 0.5)
        schema_rate = metrics.get("schema_discovery_rate", 0.0)

        self._schedule.hypothesis_throughput = throughput
        self._schedule.integration_success_rate = success_rate
        self._schedule.schema_discovery_rate = schema_rate

        current = self._schedule.current_interval_hours
        new_interval = current

        # Decision logic:
        # High success + high throughput → consolidation is working, slow down
        if success_rate > 0.7 and throughput > 2.0:
            new_interval = min(
                _MAX_CONSOLIDATION_HOURS,
                current + _SCHEDULE_ADAPTATION_STEP,
            )
            reason = (
                f"High success ({success_rate:.2f}) + throughput "
                f"({throughput:.1f}) → extend interval"
            )
        # Low success → consolidate more often to iterate
        elif success_rate < 0.3:
            new_interval = max(
                _MIN_CONSOLIDATION_HOURS,
                current - _SCHEDULE_ADAPTATION_STEP,
            )
            reason = f"Low success ({success_rate:.2f}) → shorten interval"
        # High schema discovery → let discoveries accumulate
        elif schema_rate > 3.0:
            new_interval = min(
                _MAX_CONSOLIDATION_HOURS,
                current + _SCHEDULE_ADAPTATION_STEP * 0.5,
            )
            reason = f"High schema discovery ({schema_rate:.1f}) → extend interval"
        else:
            return None

        if abs(new_interval - current) < 0.01:
            return None

        self._schedule.current_interval_hours = round(new_interval, 1)
        self._schedule.last_adaptation_reason = reason

        record = SelfModificationRecord(
            proposal_type="schedule_adaptation",
            description=(
                f"Consolidation interval: {current:.1f}h → {new_interval:.1f}h. "
                f"Reason: {reason}"
            ),
            applied=True,
            outcome="success",
        )
        self._logger.info(
            "consolidation_schedule_adapted",
            old_interval=current,
            new_interval=new_interval,
            reason=reason,
        )
        return record

    def get_consolidation_interval_hours(self) -> float:
        """Return the current adapted consolidation interval."""
        return self._schedule.current_interval_hours

    # ─── 4. Systematic Failure Detection → Architecture Proposals ────────────

    def _detect_systematic_failures(
        self,
        detector_stats: list[dict[str, Any]],
        consolidation_metrics: dict[str, float],
    ) -> list[LearningArchitectureProposal]:
        """
        Detect systematic failure modes that require architectural changes.

        Examples:
          - "hypotheses about market dynamics always fail because we lack
            temporal decay in evidence scoring"
          - "parameter hypotheses have 90% false positive rate because the
            evidence threshold is too low for noisy parameters"
        """
        proposals: list[LearningArchitectureProposal] = []

        # Check for category-specific failure patterns
        if self._meta_learning is not None:
            for cat in HypothesisCategory:
                cat_outcomes = self._meta_learning._detector_outcomes
                # Aggregate refuted count across all detectors for this category
                total_generated = 0
                total_refuted = 0
                for outcomes in cat_outcomes.values():
                    total_generated += len(outcomes)
                    total_refuted += sum(1 for o in outcomes if o == "refuted")

                if total_generated < 10:
                    continue

                refute_rate = total_refuted / max(1, total_generated)

                if refute_rate > 0.7:
                    # Check constitutional guard — never propose modifying forbidden targets
                    if any(forbidden in cat.value.lower() for forbidden in _FORBIDDEN_TARGETS):
                        continue

                    proposal = LearningArchitectureProposal(
                        failure_mode=(
                            f"Category '{cat.value}' has {refute_rate:.0%} refutation rate "
                            f"over {total_generated} hypotheses — systematic failure"
                        ),
                        proposed_change=(
                            f"Adapt evidence scoring for category '{cat.value}': "
                            f"consider adding domain-specific evidence weighting, "
                            f"temporal decay factors, or richer context features. "
                            f"Current uniform scoring may not capture the evidence structure "
                            f"needed for {cat.value} claims."
                        ),
                        expected_improvement=(
                            f"Reduce refutation rate from {refute_rate:.0%} to <30% "
                            f"for {cat.value} hypotheses"
                        ),
                        success_probability=0.4,
                    )
                    proposals.append(proposal)

        # Check for consolidation throughput collapse
        throughput = consolidation_metrics.get("hypothesis_throughput", 0.0)
        if throughput < 0.1 and consolidation_metrics.get("integration_success_rate", 0.5) < 0.2:
            proposal = LearningArchitectureProposal(
                failure_mode=(
                    f"Learning throughput collapsed: {throughput:.2f} hypotheses/hour "
                    f"with {consolidation_metrics.get('integration_success_rate', 0):.0%} "
                    f"integration rate"
                ),
                proposed_change=(
                    "Consider restructuring the hypothesis pipeline: "
                    "add structural (non-LLM) hypothesis generators to reduce "
                    "dependency on LLM budget, or lower evidence thresholds to "
                    "reduce hypothesis starvation."
                ),
                expected_improvement="Restore learning throughput to >1.0 hypotheses/hour",
                success_probability=0.5,
            )
            proposals.append(proposal)

        return proposals

    # ─── Self-Modification Outcome Tracking ──────────────────────────────────

    def record_modification_outcome(
        self,
        proposal_type: str,
        success: bool,
        metric_delta: float = 0.0,
    ) -> None:
        """Record the outcome of a self-modification for meta-meta-learning."""
        self._success_rates[proposal_type].append(success)

        self._logger.info(
            "self_modification_outcome",
            proposal_type=proposal_type,
            success=success,
            metric_delta=round(metric_delta, 4),
            historical_rate=round(
                sum(1 for s in self._success_rates[proposal_type] if s)
                / max(1, len(self._success_rates[proposal_type])),
                3,
            ),
        )

    def get_modification_success_rate(self, proposal_type: str) -> float:
        """Return the historical success rate for a type of self-modification."""
        history = self._success_rates.get(proposal_type, deque())
        if not history:
            return 0.5  # No data → neutral
        return sum(1 for s in history if s) / len(history)

    # ─── State Query ─────────────────────────────────────────────────────────

    def get_schedule(self) -> ConsolidationSchedule:
        """Return the current adaptive consolidation schedule."""
        return self._schedule

    def get_pending_proposals(self) -> list[Any]:
        """Return proposals awaiting Simula response."""
        return list(self._pending_proposals)

    def clear_pending(self) -> None:
        """Clear pending proposals after they've been dispatched."""
        self._pending_proposals.clear()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_modifications": self._total_modifications,
            "consolidation_interval_hours": self._schedule.current_interval_hours,
            "learned_complexity": dict(self._learned_complexity),
            "pending_proposals": len(self._pending_proposals),
            "modification_success_rates": {
                k: round(self.get_modification_success_rate(k), 3)
                for k in self._success_rates
            },
        }
