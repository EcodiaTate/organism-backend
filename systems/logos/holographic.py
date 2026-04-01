"""
EcodiaOS - Logos: Holographic Encoder

The holographic principle applied to episodic memory.

The event horizon of a black hole doesn't store the 3D objects that fell in.
It stores a 2D projection - the minimal encoding that conserves the information.

We do the same: don't store the experience.
Store what the experience revealed that the world model didn't already know.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.logos.types import (
    ExperienceDelta,
    RawExperience,
    SemanticDelta,
)

if TYPE_CHECKING:
    from systems.logos.world_model import WorldModel

logger = structlog.get_logger("logos.holographic")


class HolographicEncoder:
    """
    Computes the delta between what happened and what the world model
    predicted would happen. Only the delta needs to be stored. The
    prediction is free - the model generates it.

    This is identical to how video compression works:
    - Store keyframes (world model updates)
    - Store deltas between keyframes (surprising episodes)
    - Reconstruct the full video (experience) by applying deltas to keyframes
    """

    def __init__(
        self,
        world_model: WorldModel,
        *,
        discard_threshold: float = 0.01,
        update_threshold: float = 0.5,
    ) -> None:
        self._world_model = world_model
        self._discard_threshold = discard_threshold
        self._update_threshold = update_threshold

        # Throughput counters (reset daily by the service)
        self.total_encoded: int = 0
        self.total_discarded: int = 0

    async def encode(self, raw_experience: RawExperience) -> ExperienceDelta:
        """
        Compute the delta between what happened and what was predicted.
        Only the delta needs to be stored. The prediction is free.

        If prediction was perfect (delta < discard_threshold), the experience
        adds nothing - pure redundancy, no storage needed.

        If information_content > update_threshold, the world model needs updating.
        """
        # What the model predicted would happen
        prediction = await self._world_model.predict(raw_experience.context)

        # What actually happened
        actual = raw_experience.content

        # The delta: only what wasn't predicted
        semantic_delta = self._compute_semantic_delta(
            prediction.expected_content, actual
        )

        self.total_encoded += 1

        # If prediction was perfect, delta is empty - experience adds nothing
        if semantic_delta.information_content < self._discard_threshold:
            self.total_discarded += 1
            logger.debug(
                "holographic_discard",
                experience_id=raw_experience.id,
                info_content=semantic_delta.information_content,
            )
            return ExperienceDelta(
                experience_id=raw_experience.id,
                delta_content=None,
                information_content=0.0,
                world_model_update_required=False,
                discard_after_encoding=True,
            )

        # Record prediction outcome for accuracy tracking
        self._world_model.record_prediction_outcome(
            correct=semantic_delta.information_content < 0.3
        )

        logger.info(
            "holographic_encoded",
            experience_id=raw_experience.id,
            info_content=semantic_delta.information_content,
            novel_entities=len(semantic_delta.novel_entities),
            update_required=semantic_delta.information_content > self._update_threshold,
        )

        return ExperienceDelta(
            experience_id=raw_experience.id,
            delta_content=semantic_delta,
            information_content=semantic_delta.information_content,
            world_model_update_required=(
                semantic_delta.information_content > self._update_threshold
            ),
            discard_after_encoding=False,
        )

    def _compute_semantic_delta(
        self,
        predicted: dict[str, Any],
        actual: dict[str, Any],
    ) -> SemanticDelta:
        """
        Compute the semantic difference between predicted and actual content.

        Identifies:
        - Novel entities (present in actual but absent from prediction)
        - Violated priors (predicted values contradicted by actual)
        - Novel relations (structural relationships not in the prediction)
        - Overall information content (0=identical, 1=completely novel)

        P4 fix: shared-key values compared with _value_distance() rather than
        equality, so dicts with identical keys but different numeric/string
        values are correctly scored as novel (Spec 21 §5.2).
        """
        if not predicted and not actual:
            return SemanticDelta(information_content=0.0)
        if not predicted:
            return SemanticDelta(
                information_content=1.0,
                novel_entities=list(actual.keys()),
                content=actual,
            )
        if not actual:
            return SemanticDelta(information_content=0.0)

        pred_keys = set(predicted.keys())
        actual_keys = set(actual.keys())

        # Novel keys: in actual but not predicted
        novel_keys = actual_keys - pred_keys
        novel_entities = list(novel_keys)

        # Violated priors: shared keys whose values differ beyond a small tolerance.
        # P4 fix: typed distance so {error: 0.01} vs {error: 0.9} is captured.
        violated_priors: list[str] = []
        value_violation_weight: float = 0.0
        shared_keys = pred_keys & actual_keys

        for key in shared_keys:
            dist = self._value_distance(predicted[key], actual[key])
            if dist > 0.05:  # 5% tolerance suppresses float noise
                violated_priors.append(key)
                value_violation_weight += dist

        # Novel relations: look for relational keys in actual (heuristic)
        novel_relations: list[str] = []
        for key in novel_keys:
            val = actual.get(key)
            if isinstance(val, dict) or (isinstance(val, str) and "->" in val):
                novel_relations.append(str(val) if isinstance(val, str) else key)

        # Compute information content.
        # P4 fix: weight violations by value-distance magnitude, not plain count,
        # so a single large numeric error is scored higher than many trivial ones.
        total_keys = len(pred_keys | actual_keys)
        if total_keys == 0:
            information_content = 0.0
        else:
            novel_fraction = len(novel_keys) / total_keys
            violation_intensity = (
                value_violation_weight / len(shared_keys)
                if shared_keys
                else 0.0
            )
            information_content = (novel_fraction * 0.6) + (min(violation_intensity, 1.0) * 0.4)

        # Build delta content: only the novel/changed parts
        delta_content: dict[str, Any] = {}
        for key in novel_keys:
            delta_content[key] = actual[key]
        for key in violated_priors:
            delta_content[key] = actual[key]

        return SemanticDelta(
            information_content=min(information_content, 1.0),
            novel_entities=novel_entities,
            violated_priors=violated_priors,
            novel_relations=novel_relations,
            content=delta_content,
        )

    @staticmethod
    def _value_distance(predicted: Any, actual: Any) -> float:
        """
        Typed semantic distance between two values, in [0, 1].

        Numeric  - relative error clamped to 1.0.
        String   - 1 - character-level Jaccard (cheap proxy; no embedding needed).
        Dict     - key-overlap distance (shallow).
        Sequence - length-difference ratio.
        Other    - equality check.

        Spec 21 §5.2 P4: holographic encoding must capture value-level novelty,
        not just key-level novelty.
        """
        if type(predicted) is not type(actual):
            return 1.0

        if isinstance(predicted, bool):
            return 0.0 if predicted == actual else 1.0

        if isinstance(predicted, (int, float)):
            denom = max(abs(float(predicted)), abs(float(actual)), 1e-9)
            return min(abs(float(predicted) - float(actual)) / denom, 1.0)

        if isinstance(predicted, str):
            if predicted == actual:
                return 0.0
            chars_p = set(predicted)
            chars_a = set(actual)
            union = chars_p | chars_a
            if not union:
                return 0.0
            return 1.0 - (len(chars_p & chars_a) / len(union))

        if isinstance(predicted, dict) and isinstance(actual, dict):
            keys_p = set(predicted.keys())
            keys_a = set(actual.keys())
            union = keys_p | keys_a
            if not union:
                return 0.0
            return 1.0 - (len(keys_p & keys_a) / len(union))

        if isinstance(predicted, (list, tuple)):
            if predicted == actual:
                return 0.0
            len_p, len_a = len(predicted), len(actual)
            if len_p == 0 and len_a == 0:
                return 0.0
            return abs(len_p - len_a) / max(len_p, len_a, 1)

        return 0.0 if predicted == actual else 1.0
