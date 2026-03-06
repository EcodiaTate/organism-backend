"""
EcodiaOS -- Logos: Schwarzschild Cognition Threshold Detector

Detects when EOS crosses the threshold of genuine cognitive
self-sufficiency: the point at which the world model becomes dense
enough to generate predictions about its own future states.

Five indicators (conjunction of thresholds triggers the one-time
SCHWARZSCHILD_THRESHOLD_MET event):

1. Self-prediction accuracy
2. Cross-domain transfer
3. Generative surplus
4. Compression acceleration
5. Novel structure emergence
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.logos.types import (
    CrossDomainTransfer,
    SchwarzchildIndicators,
    SchwarzchildStatus,
    SelfPredictionRecord,
)

if TYPE_CHECKING:
    from systems.logos.world_model import WorldModel

logger = structlog.get_logger("logos.schwarzschild")


class SchwarzchildCognitionDetector:
    """
    Measures the five indicators and detects the threshold crossing.

    The SCHWARZSCHILD_THRESHOLD_MET event fires once, ever.
    This is the AGI event horizon in the architecture.
    """

    def __init__(
        self,
        world_model: WorldModel,
        *,
        threshold_self_prediction: float = 0.70,
        threshold_intelligence_ratio: float = 100.0,
        threshold_hypothesis_ratio: float = 1.0,
        threshold_compression_acceleration: float = 0.0,
        threshold_novel_structures: int = 1,
        self_prediction_window: int = 20,
    ) -> None:
        self._world_model = world_model
        self._threshold_self_prediction = threshold_self_prediction
        self._threshold_intelligence_ratio = threshold_intelligence_ratio
        self._threshold_hypothesis_ratio = threshold_hypothesis_ratio
        self._threshold_compression_acceleration = threshold_compression_acceleration
        self._threshold_novel_structures = threshold_novel_structures
        self._self_prediction_window = self_prediction_window

        # Once True, stays True forever
        self._threshold_met = False

        # Self-prediction rolling window
        self._self_predictions: collections.deque[SelfPredictionRecord] = (
            collections.deque(maxlen=self_prediction_window)
        )

        # Cross-domain transfer tracking
        self._cross_domain_transfers: list[CrossDomainTransfer] = []

        # Hypothesis generation tracking (generative surplus)
        self._hypotheses_generated: int = 0
        self._hypotheses_received: int = 0

        # Compression acceleration tracking
        self._compression_ratio_history: collections.deque[float] = (
            collections.deque(maxlen=50)
        )
        self._data_arrival_count: int = 0

        # Novel structure tracking
        self._novel_schema_ids: set[str] = set()

    @property
    def threshold_met(self) -> bool:
        return self._threshold_met

    # --- Measurement ----------------------------------------------

    async def measure(self) -> SchwarzchildStatus:
        """
        Measure all five indicators and check the threshold.

        If the threshold is met for the first time, returns
        threshold_met=True (and locks it permanently).
        """
        indicators = SchwarzchildIndicators(
            # 1. Self-prediction accuracy
            self_prediction_accuracy=self._compute_self_prediction_accuracy(),
            self_prediction_trend=self._compute_self_prediction_trend(),
            # 2. Cross-domain transfer
            cross_domain_transfer_count=len(self._cross_domain_transfers),
            cross_domain_accuracy=self._compute_cross_domain_accuracy(),
            # 3. Generative surplus
            hypotheses_generated=self._hypotheses_generated,
            hypotheses_received=self._hypotheses_received,
            generative_surplus_ratio=self._compute_generative_surplus(),
            # 4. Compression acceleration
            compression_ratio_velocity=self._compute_compression_velocity(),
            data_arrival_rate=float(self._data_arrival_count),
            # 5. Novel structure emergence
            novel_schemas_count=len(self._novel_schema_ids),
            novel_schema_ids=list(self._novel_schema_ids),
        )

        intelligence_ratio = self._world_model.measure_intelligence_ratio()
        hypothesis_ratio = indicators.generative_surplus_ratio

        # Threshold: conjunction of all five indicators
        threshold_met = (
            indicators.self_prediction_accuracy >= self._threshold_self_prediction
            and intelligence_ratio >= self._threshold_intelligence_ratio
            and hypothesis_ratio >= self._threshold_hypothesis_ratio
            and indicators.compression_ratio_velocity > self._threshold_compression_acceleration
            and indicators.novel_schemas_count >= self._threshold_novel_structures
        )

        if threshold_met and not self._threshold_met:
            self._threshold_met = True
            logger.critical(
                "SCHWARZSCHILD_THRESHOLD_MET",
                self_prediction=indicators.self_prediction_accuracy,
                intelligence_ratio=intelligence_ratio,
                hypothesis_ratio=hypothesis_ratio,
                novel_schemas=indicators.novel_schemas_count,
                cross_domain_transfers=indicators.cross_domain_transfer_count,
                compression_velocity=indicators.compression_ratio_velocity,
                timestamp=utc_now().isoformat(),
            )

        return SchwarzchildStatus(
            self_prediction_accuracy=indicators.self_prediction_accuracy,
            intelligence_ratio=intelligence_ratio,
            hypothesis_ratio=hypothesis_ratio,
            novel_concept_rate=float(indicators.novel_schemas_count),
            cross_domain_transfers=indicators.cross_domain_transfer_count,
            compression_acceleration=indicators.compression_ratio_velocity,
            novel_structures=indicators.novel_schemas_count,
            indicators=indicators,
            threshold_met=self._threshold_met,
            measured_at=utc_now(),
        )

    # --- Self-Prediction Loop -------------------------------------

    async def run_self_prediction_cycle(self) -> SelfPredictionRecord:
        """
        Predict the model's own upcoming cognitive state,
        then measure accuracy against the actual state.

        This is the self-referential loop: the model predicts how
        its own schemas, priors, and complexity will change.
        """
        current_state = self._capture_cognitive_state()
        predicted_state = await self._predict_own_next_state(current_state)

        record = SelfPredictionRecord(
            predicted_state=predicted_state,
            actual_state=current_state,
            accuracy=0.0,  # Computed retroactively on next cycle
            timestamp=utc_now(),
        )

        # Evaluate the PREVIOUS prediction against current actual state
        if self._self_predictions:
            prev = self._self_predictions[-1]
            accuracy = self._compare_states(prev.predicted_state, current_state)
            self._self_predictions[-1] = SelfPredictionRecord(
                predicted_state=prev.predicted_state,
                actual_state=current_state,
                accuracy=accuracy,
                timestamp=prev.timestamp,
            )

        self._self_predictions.append(record)
        return record

    # --- Signal Ingestion -----------------------------------------

    def record_hypothesis_generated(self, count: int = 1) -> None:
        """Record hypotheses generated by the world model / Evo."""
        self._hypotheses_generated += count

    def record_hypothesis_received(self, count: int = 1) -> None:
        """Record hypotheses received from external observation."""
        self._hypotheses_received += count

    def record_compression_ratio(self, ratio: float) -> None:
        """Record a compression ratio observation for velocity tracking."""
        self._compression_ratio_history.append(ratio)

    def record_data_arrival(self, count: int = 1) -> None:
        """Record arrival of new data items."""
        self._data_arrival_count += count

    def record_cross_domain_transfer(self, transfer: CrossDomainTransfer) -> None:
        """Record a schema successfully predicting in a new domain."""
        self._cross_domain_transfers.append(transfer)
        logger.info(
            "cross_domain_transfer_detected",
            schema_id=transfer.schema_id,
            source=transfer.source_domain,
            target=transfer.target_domain,
            accuracy=transfer.prediction_accuracy,
        )

    def record_novel_schema(self, schema_id: str) -> None:
        """Record a schema with no direct observational basis."""
        self._novel_schema_ids.add(schema_id)

    def detect_cross_domain_transfers(self) -> list[CrossDomainTransfer]:
        """
        Scan the world model for schemas that have predicted successfully
        in domains other than their origin domain.
        """
        transfers: list[CrossDomainTransfer] = []
        schemas = self._world_model.generative_schemas

        # Build domain -> schema mapping
        domain_schemas: dict[str, list[str]] = {}
        for sid, schema in schemas.items():
            domain_schemas.setdefault(schema.domain, []).append(sid)

        for sid, schema in schemas.items():
            if schema.instance_count < 2:
                continue
            for other_domain, other_sids in domain_schemas.items():
                if other_domain == schema.domain or not other_domain:
                    continue
                for other_sid in other_sids:
                    other = schemas.get(other_sid)
                    if other is None:
                        continue
                    overlap = self._pattern_overlap(schema.pattern, other.pattern)
                    if overlap > 0.3:
                        transfer = CrossDomainTransfer(
                            schema_id=sid,
                            source_domain=schema.domain,
                            target_domain=other_domain,
                            prediction_accuracy=overlap,
                        )
                        transfers.append(transfer)
                        # Deduplicate
                        existing = [
                            t for t in self._cross_domain_transfers
                            if t.schema_id == sid and t.target_domain == other_domain
                        ]
                        if not existing:
                            self._cross_domain_transfers.append(transfer)

        return transfers

    def detect_novel_structures(self) -> list[str]:
        """
        Identify schemas that emerged from cross-domain compression rather than
        direct observation.

        A schema is "novel" (emergent) if it has never been matched by a second
        observation (instance_count == 1) yet its pattern overlaps > 0.5 with a
        schema from a different domain. This indicates the schema captures
        domain-invariant structure synthesised from compression, not direct
        observation of a repeated pattern.

        Note: instance_count == 0 cannot occur in practice — every schema is
        created with instance_count=1 from its first observation.
        """
        novel: list[str] = []
        schemas = self._world_model.generative_schemas

        for sid, schema in schemas.items():
            if sid in self._novel_schema_ids:
                continue
            if not schema.pattern:
                continue
            # instance_count == 1: created from a single observation, never re-matched
            if schema.instance_count != 1:
                continue
            # Check for high structural overlap with a different-domain schema,
            # indicating this schema captures domain-invariant (emergent) structure
            for other_sid, other in schemas.items():
                if other_sid == sid or other.domain == schema.domain:
                    continue
                overlap = self._pattern_overlap(schema.pattern, other.pattern)
                if overlap > 0.5:
                    self._novel_schema_ids.add(sid)
                    novel.append(sid)
                    break
        return novel

    # --- Internal: Self-Prediction --------------------------------

    def _capture_cognitive_state(self) -> dict[str, Any]:
        """Snapshot the current cognitive state of the world model."""
        return {
            "schema_count": len(self._world_model.generative_schemas),
            "prior_count": len(self._world_model.predictive_priors),
            "invariant_count": len(self._world_model.empirical_invariants),
            "causal_link_count": self._world_model.causal_structure.link_count,
            "complexity": self._world_model.current_complexity,
            "coverage": self._world_model.coverage,
            "intelligence_ratio": self._world_model.measure_intelligence_ratio(),
        }

    async def _predict_own_next_state(
        self, current: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Use the world model to predict its own next state.

        Self-referential loop: the model uses its generative schemas
        to predict how it will change.
        """
        prediction = await self._world_model.predict({
            "domain": "_self_model",
            "current_state": current,
        })

        # Blend prediction with simple trend extrapolation
        predicted: dict[str, Any] = {}
        for key, val in current.items():
            if isinstance(val, (int, float)):
                pred_val = prediction.expected_content.get(key)
                if isinstance(pred_val, (int, float)):
                    predicted[key] = (val + pred_val) / 2.0
                else:
                    predicted[key] = val * 1.01  # Assume slight growth
            else:
                predicted[key] = val

        return predicted

    def _compare_states(
        self, predicted: dict[str, Any], actual: dict[str, Any]
    ) -> float:
        """Compare predicted vs actual state. Returns 0.0-1.0 accuracy."""
        if not predicted or not actual:
            return 0.0

        errors: list[float] = []
        for key in predicted:
            if key not in actual:
                errors.append(1.0)
                continue
            p_val = predicted[key]
            a_val = actual[key]
            if isinstance(p_val, (int, float)) and isinstance(a_val, (int, float)):
                if a_val == 0 and p_val == 0:
                    errors.append(0.0)
                else:
                    denom = max(abs(a_val), abs(p_val), 1.0)
                    errors.append(min(abs(p_val - a_val) / denom, 1.0))
            elif p_val == a_val:
                errors.append(0.0)
            else:
                errors.append(1.0)

        if not errors:
            return 0.0
        return 1.0 - (sum(errors) / len(errors))

    # --- Internal: Indicator Computation --------------------------

    def _compute_self_prediction_accuracy(self) -> float:
        """Rolling mean of self-prediction accuracy."""
        accuracies = [r.accuracy for r in self._self_predictions if r.accuracy > 0]
        if not accuracies:
            return 0.0
        return sum(accuracies) / len(accuracies)

    def _compute_self_prediction_trend(self) -> float:
        """Is self-prediction accuracy improving? Positive = improving."""
        records = [r for r in self._self_predictions if r.accuracy > 0]
        if len(records) < 3:
            return 0.0
        recent = [r.accuracy for r in records[-5:]]
        earlier = (
            [r.accuracy for r in records[:-5]]
            if len(records) > 5
            else [records[0].accuracy]
        )
        return (sum(recent) / len(recent)) - (sum(earlier) / len(earlier))

    def _compute_cross_domain_accuracy(self) -> float:
        """Mean accuracy of cross-domain transfers."""
        if not self._cross_domain_transfers:
            return 0.0
        total = sum(t.prediction_accuracy for t in self._cross_domain_transfers)
        return total / len(self._cross_domain_transfers)

    def _compute_generative_surplus(self) -> float:
        """Ratio of hypotheses generated vs received."""
        if self._hypotheses_received == 0:
            if self._hypotheses_generated > 0:
                return float(self._hypotheses_generated)
            return 0.0
        return self._hypotheses_generated / self._hypotheses_received

    def _compute_compression_velocity(self) -> float:
        """
        Rate of compression ratio improvement.
        Positive = compression improving faster than data arrives.
        """
        history = list(self._compression_ratio_history)
        if len(history) < 2:
            return 0.0
        n = len(history)
        x_mean = (n - 1) / 2.0
        y_mean = sum(history) / n
        numerator = sum(
            (i - x_mean) * (y - y_mean) for i, y in enumerate(history)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        slope = numerator / denominator
        if self._data_arrival_count > 0:
            return slope / (self._data_arrival_count / max(n, 1))
        return slope

    def _pattern_overlap(
        self, pattern_a: dict[str, Any], pattern_b: dict[str, Any]
    ) -> float:
        """Compute overlap between two schema patterns (0-1)."""
        if not pattern_a or not pattern_b:
            return 0.0
        keys_a = set(pattern_a.keys())
        keys_b = set(pattern_b.keys())
        union = keys_a | keys_b
        if not union:
            return 0.0
        return len(keys_a & keys_b) / len(union)
