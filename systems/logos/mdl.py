"""
EcodiaOS — Logos: Minimum Description Length Estimator

Scores every piece of knowledge by how much reality it explains per bit
of description. Computing exact Kolmogorov complexity is uncomputable;
Logos uses practical estimators that produce consistent pressure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
from datetime import datetime

import structlog

from primitives.common import utc_now
from systems.logos.types import KnowledgeItemType, MDLScore

if TYPE_CHECKING:

    from systems.logos.world_model import WorldModel
logger = structlog.get_logger("logos.mdl")


# ─── Protocols for items coming from other systems ───────────────


class EpisodeProtocol(Protocol):
    """Minimal interface for an episodic memory item."""

    @property
    def id(self) -> str: ...

    @property
    def context(self) -> dict[str, Any]: ...

    @property
    def content(self) -> dict[str, Any]: ...

    @property
    def raw_complexity(self) -> float: ...


class HypothesisProtocol(Protocol):
    """Minimal interface for a hypothesis from Evo."""

    @property
    def id(self) -> str: ...

    @property
    def supporting_observations(self) -> list[Any]: ...

    @property
    def description(self) -> str: ...

    @property
    def unique_predictive_coverage(self) -> float: ...

    @property
    def last_tested(self) -> datetime: ...

    @property
    def test_frequency(self) -> float: ...


class SchemaProtocol(Protocol):
    """Minimal interface for a schema (entity type, relation pattern)."""

    @property
    def id(self) -> str: ...

    @property
    def instances(self) -> list[Any]: ...

    @property
    def description(self) -> str: ...

    @property
    def unique_organizing_power(self) -> float: ...

    @property
    def last_instantiated(self) -> datetime: ...

    @property
    def instantiation_frequency(self) -> float: ...


# ─── MDL Estimator ───────────────────────────────────────────────


class MDLEstimator:
    """
    Practical MDL estimation for different knowledge types.

    These are approximations — but consistent approximations
    create consistent compression pressure across the system.
    """

    def __init__(self, world_model: WorldModel | None = None) -> None:
        self._world_model = world_model

    def set_world_model(self, world_model: WorldModel) -> None:
        self._world_model = world_model

    async def score_episode(self, episode: EpisodeProtocol) -> MDLScore:
        """
        An episode's MDL score is primarily its SURPRISE relative to the world model.

        Highly predictable episodes have near-zero marginal value
        (the world model already encodes them). Highly surprising episodes
        have high marginal value (they challenge the world model).

        This is the holographic principle: store the DELTA between reality
        and our model of reality, not the raw experience.
        """
        prediction_error = 1.0  # Default: fully novel if no world model
        raw_complexity = max(episode.raw_complexity, 1.0)

        if self._world_model is not None:
            prediction = await self._world_model.predict(episode.context)
            prediction_error = self._semantic_distance(
                prediction.expected_content, episode.content
            )

        # The episode's "true cost" is only its unpredicted portion
        effective_description_length = prediction_error * raw_complexity

        # If prediction_error ~= 0, this episode adds nothing to the world model
        # If prediction_error ~= 1, this episode is pure new information
        compression_ratio = raw_complexity / max(effective_description_length, 0.001)

        return MDLScore(
            item_id=episode.id,
            item_type=KnowledgeItemType.EPISODE,
            observations_covered=1,
            observation_complexity=raw_complexity,
            description_length=effective_description_length,
            compression_ratio=compression_ratio,
            marginal_value=prediction_error,
            last_accessed=utc_now(),
            access_frequency=0.0,
            decay_rate=0.1,  # Episodes decay fast unless repeatedly accessed
        )

    async def score_hypothesis(self, hypothesis: HypothesisProtocol) -> MDLScore:
        """
        A hypothesis's MDL score: observations_explained / hypothesis_complexity.

        The best hypotheses are short sentences that predict many things.
        Newton's laws: 3 equations, explains all macroscopic motion. Excellent MDL.
        Memorizing every trajectory: many equations, same coverage. Poor MDL.

        Occam's razor is built into the score.
        """
        obs_complexity = sum(
            getattr(o, "complexity", 1.0)
            for o in hypothesis.supporting_observations
        )
        hypothesis_length = self._estimate_description_length_str(hypothesis.description)

        compression_ratio = obs_complexity / max(hypothesis_length, 1.0)

        return MDLScore(
            item_id=hypothesis.id,
            item_type=KnowledgeItemType.HYPOTHESIS,
            observations_covered=len(hypothesis.supporting_observations),
            observation_complexity=obs_complexity,
            description_length=hypothesis_length,
            compression_ratio=compression_ratio,
            marginal_value=hypothesis.unique_predictive_coverage,
            last_accessed=hypothesis.last_tested,
            access_frequency=hypothesis.test_frequency,
            decay_rate=0.02,  # Hypotheses decay slowly — expensive to form
        )

    async def score_schema(self, schema: SchemaProtocol) -> MDLScore:
        """
        A schema (entity type, relation type, community pattern) is worth keeping
        if it organizes many instances more efficiently than storing them individually.

        The cost of the schema plus the deltas of each instance from the schema
        must be less than the cost of storing all instances independently.
        """
        instance_complexity = sum(
            getattr(i, "raw_complexity", 1.0) for i in schema.instances
        )

        schema_length = self._estimate_description_length_str(schema.description)
        delta_sum = sum(
            self._estimate_delta_complexity(i, schema) for i in schema.instances
        )
        schema_plus_deltas = schema_length + delta_sum

        compression_ratio = instance_complexity / max(schema_plus_deltas, 1.0)

        return MDLScore(
            item_id=schema.id,
            item_type=KnowledgeItemType.SCHEMA,
            observations_covered=len(schema.instances),
            observation_complexity=instance_complexity,
            description_length=schema_plus_deltas,
            compression_ratio=compression_ratio,
            marginal_value=schema.unique_organizing_power,
            last_accessed=schema.last_instantiated,
            access_frequency=schema.instantiation_frequency,
            decay_rate=0.005,  # Schemas are extremely sticky — most compressed
        )

    async def score_generic(
        self,
        item_id: str,
        item_type: KnowledgeItemType,
        description_length: float,
        observations_covered: int = 1,
        observation_complexity: float = 1.0,
        last_accessed: datetime | None = None,
        access_frequency: float = 0.0,
    ) -> MDLScore:
        """
        Generic scoring for items that don't fit the three primary types.
        Used for procedural knowledge, causal links, etc.
        """
        compression_ratio = observation_complexity / max(description_length, 0.001)

        return MDLScore(
            item_id=item_id,
            item_type=item_type,
            observations_covered=observations_covered,
            observation_complexity=observation_complexity,
            description_length=description_length,
            compression_ratio=compression_ratio,
            marginal_value=compression_ratio / 10.0,
            last_accessed=last_accessed or utc_now(),
            access_frequency=access_frequency,
            decay_rate=0.05,
        )

    # ─── Internal helpers ────────────────────────────────────────

    def _semantic_distance(
        self, predicted: dict[str, Any], actual: dict[str, Any]
    ) -> float:
        """
        Estimate semantic distance between prediction and reality.

        Returns 0.0 (identical/perfectly predicted) to 1.0 (completely novel).
        Uses key-set overlap as a practical proxy for full semantic distance.
        """
        if not predicted and not actual:
            return 0.0
        if not predicted or not actual:
            return 1.0

        pred_keys = set(predicted.keys())
        actual_keys = set(actual.keys())

        if not pred_keys and not actual_keys:
            return 0.0

        # Jaccard-like overlap on key structure
        intersection = pred_keys & actual_keys
        union = pred_keys | actual_keys
        structural_overlap = len(intersection) / max(len(union), 1)

        # Value-level comparison for overlapping keys
        value_matches = 0
        for key in intersection:
            if predicted[key] == actual[key]:
                value_matches += 1

        value_overlap = value_matches / max(len(intersection), 1) if intersection else 0.0

        # Combine: structural similarity * value similarity
        similarity = (structural_overlap * 0.4) + (value_overlap * 0.6)
        return 1.0 - similarity

    def _estimate_description_length_str(self, text: str) -> float:
        """
        Estimate description length in bits from a text description.

        Uses a simple entropy estimator: ~4.5 bits per character for English
        text (empirical Shannon entropy of natural language).
        """
        if not text:
            return 1.0
        # Shannon entropy approximation: ~4.5 bits per character
        return len(text) * 4.5

    def _estimate_delta_complexity(self, instance: Any, schema: Any) -> float:
        """
        Estimate the residual complexity of an instance after the schema
        accounts for its regular structure. The delta is what remains.
        """
        instance_raw = getattr(instance, "raw_complexity", 1.0)
        # Assume the schema captures ~70% of each instance's structure
        return instance_raw * 0.3
