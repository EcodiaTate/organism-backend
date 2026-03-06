"""
EcodiaOS — Logos: Holographic Encoder

The holographic principle applied to episodic memory.

The event horizon of a black hole doesn't store the 3D objects that fell in.
It stores a 2D projection — the minimal encoding that conserves the information.

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
    prediction is free — the model generates it.

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
        adds nothing — pure redundancy, no storage needed.

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

        # If prediction was perfect, delta is empty — experience adds nothing
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

        # Violated priors: keys present in both but with different values
        violated_priors: list[str] = []
        matching_values = 0
        shared_keys = pred_keys & actual_keys

        for key in shared_keys:
            if predicted[key] != actual[key]:
                violated_priors.append(key)
            else:
                matching_values += 1

        # Novel relations: look for relational keys in actual (heuristic)
        novel_relations: list[str] = []
        for key in novel_keys:
            val = actual.get(key)
            if isinstance(val, dict) or (isinstance(val, str) and "->" in val):
                novel_relations.append(str(val) if isinstance(val, str) else key)

        # Compute information content
        total_keys = len(pred_keys | actual_keys)
        if total_keys == 0:
            information_content = 0.0
        else:
            # Fraction of content that is novel or contradictory
            novel_fraction = len(novel_keys) / total_keys
            violation_fraction = len(violated_priors) / max(len(shared_keys), 1)
            information_content = (novel_fraction * 0.6) + (violation_fraction * 0.4)

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

    def reset_counters(self) -> None:
        """Reset daily throughput counters."""
        self.total_encoded = 0
        self.total_discarded = 0
