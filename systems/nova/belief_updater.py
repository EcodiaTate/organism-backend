"""
EcodiaOS — Nova Belief Updater

Maintains and updates the structured belief state from workspace broadcasts.

This is perceptual inference: given a new observation (workspace broadcast),
update beliefs to better explain it. In the active inference framework, this
is equivalent to minimising variational free energy with respect to the
belief state q(s).

The Bayesian update rule used here is a precision-weighted interpolation:
    q(s_new) = (1 - α) * q(s_old) + α * likelihood(obs)
where α = broadcast.precision (how much to trust the new evidence).

For entity beliefs specifically:
    confidence_new = confidence_old + precision * (1 - confidence_old)
This is a logistic-like accumulation: each piece of evidence pushes confidence
toward 1.0, weighted by that evidence's precision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.nova.types import (
    BeliefDelta,
    BeliefState,
    ContextBelief,
    EntityBelief,
    IndividualBelief,
)

if TYPE_CHECKING:
    from systems.atune.types import WorkspaceBroadcast

logger = structlog.get_logger()


# Maximum entities tracked in the belief state
_MAX_ENTITY_BELIEFS = 200
# Confidence decay per cycle for unobserved entities (forgetting)
_CONFIDENCE_DECAY = 0.005
# Minimum confidence before entity belief is pruned
_MIN_ENTITY_CONFIDENCE = 0.05


class BeliefUpdater:
    """
    Maintains Nova's belief state and updates it from workspace broadcasts.

    The belief state is Nova's map of the world — what it knows, how confident
    it is, and where the prediction errors are. This drives deliberation:
    which goals are relevant, which policies can work.
    """

    def __init__(self) -> None:
        self._beliefs = BeliefState()
        self._logger = logger.bind(system="nova.belief_updater")

    @property
    def beliefs(self) -> BeliefState:
        return self._beliefs

    def update_from_broadcast(
        self,
        broadcast: WorkspaceBroadcast,
    ) -> BeliefDelta:
        """
        Integrate a workspace broadcast into the belief state.

        Called at the start of every deliberation cycle. Returns a BeliefDelta
        describing what changed, which the rest of Nova uses to assess novelty
        and select goals.

        This is synchronous and must complete in ≤50ms (per spec).
        """
        delta = BeliefDelta()
        precision = broadcast.precision

        # ── Update context belief from broadcast content ──
        context_update = self._update_context(broadcast, precision)
        delta.context_update = context_update
        self._beliefs = self._beliefs.model_copy(
            update={"current_context": context_update}
        )

        # ── Extract and update entity beliefs from memory context ──
        memory_context = getattr(broadcast.context, "memory_context", None)
        if memory_context and hasattr(memory_context, "entities"):
            for entity_data in memory_context.entities:
                entity_id = str(
                    getattr(entity_data, "id", "") or getattr(entity_data, "entity_id", "")
                )
                if not entity_id:
                    continue

                if entity_id in self._beliefs.entities:
                    updated = _bayesian_update_entity(
                        prior=self._beliefs.entities[entity_id],
                        precision=precision,
                    )
                    delta.entity_updates[entity_id] = updated
                else:
                    added = EntityBelief(
                        entity_id=entity_id,
                        name=str(getattr(entity_data, "name", entity_id)),
                        entity_type=str(getattr(entity_data, "type", "")),
                        properties=dict(getattr(entity_data, "properties", {}) or {}),
                        confidence=float(getattr(entity_data, "confidence", 0.5)) * precision,
                        last_observed=utc_now(),
                    )
                    delta.entity_additions[entity_id] = added

        # ── Update individual beliefs if a person is involved ──
        individual_id = _extract_individual_id(broadcast)
        if individual_id:
            current = self._beliefs.individual_beliefs.get(individual_id)
            updated_individual = _update_individual_belief(
                individual_id=individual_id,
                current=current,
                broadcast=broadcast,
                precision=precision,
            )
            delta.individual_updates[individual_id] = updated_individual

        # ── Detect belief conflicts ──
        if context_update.prediction_error_magnitude > 0.6:
            delta.prediction_error_magnitude = context_update.prediction_error_magnitude
            if context_update.prediction_error_magnitude > 0.75:
                # High prediction error = potential contradiction with prior beliefs
                delta.contradicted_belief_ids.append(context_update.domain or "context")

        # ── Apply delta to belief state ──
        self._apply_delta(delta)

        # ── Recompute free energy ──
        new_vfe = self._beliefs.compute_free_energy()
        self._beliefs = self._beliefs.model_copy(
            update={
                "free_energy": new_vfe,
                "last_updated": utc_now(),
            }
        )

        self._logger.debug(
            "beliefs_updated",
            entities=len(self._beliefs.entities),
            individuals=len(self._beliefs.individual_beliefs),
            vfe=round(new_vfe, 3),
            prediction_error=round(delta.prediction_error_magnitude, 3),
        )

        return delta

    def update_from_outcome(
        self,
        outcome_description: str,
        success: bool,
        precision: float = 0.7,
    ) -> None:
        """
        Update beliefs from an intent outcome.
        Success → increase epistemic confidence, reduce free energy.
        Failure → decrease capability confidence for relevant domain.
        """
        if success:
            new_confidence = min(1.0, self._beliefs.overall_confidence + 0.03 * precision)
        else:
            new_confidence = max(0.1, self._beliefs.overall_confidence - 0.05 * precision)

        self._beliefs = self._beliefs.model_copy(
            update={"overall_confidence": new_confidence}
        )

        # Update self-belief epistemic confidence
        self_belief = self._beliefs.self_belief
        if success:
            new_epistemic = min(1.0, self_belief.epistemic_confidence + 0.02)
        else:
            new_epistemic = max(0.1, self_belief.epistemic_confidence - 0.05)

        updated_self = self_belief.model_copy(update={"epistemic_confidence": new_epistemic})
        self._beliefs = self._beliefs.model_copy(update={"self_belief": updated_self})

    def decay_unobserved_entities(self) -> None:
        """
        Apply confidence decay to entities not observed in this cycle.
        This is the 'forgetting' mechanism — beliefs weaken without evidence.
        Prune entities below the minimum confidence threshold.
        """
        updated: dict[str, EntityBelief] = {}
        for eid, belief in self._beliefs.entities.items():
            new_conf = max(0.0, belief.confidence - _CONFIDENCE_DECAY)
            if new_conf >= _MIN_ENTITY_CONFIDENCE:
                updated[eid] = belief.model_copy(update={"confidence": new_conf})
        self._beliefs = self._beliefs.model_copy(update={"entities": updated})

    def inject_entity(self, entity_id: str, name: str, confidence: float = 0.5) -> None:
        """Manually inject an entity belief (for testing or initialisation)."""
        belief = EntityBelief(
            entity_id=entity_id,
            name=name,
            confidence=confidence,
            last_observed=utc_now(),
        )
        updated = dict(self._beliefs.entities)
        updated[entity_id] = belief
        self._beliefs = self._beliefs.model_copy(update={"entities": updated})

    def upsert_entity(self, belief: EntityBelief) -> None:
        """Insert or replace a full EntityBelief (immutable model_copy pattern)."""
        updated = dict(self._beliefs.entities)
        updated[belief.entity_id] = belief
        self._beliefs = self._beliefs.model_copy(update={"entities": updated})

    def update_entity(self, entity_id: str, **fields: object) -> bool:
        """
        Update specific fields on an existing EntityBelief.
        Returns True if the entity was found and updated, False otherwise.
        """
        existing = self._beliefs.entities.get(entity_id)
        if existing is None:
            return False
        updated_belief = existing.model_copy(update=fields)
        updated = dict(self._beliefs.entities)
        updated[entity_id] = updated_belief
        self._beliefs = self._beliefs.model_copy(update={"entities": updated})
        return True

    # ─── Private ──────────────────────────────────────────────────

    def _apply_delta(self, delta: BeliefDelta) -> None:
        """Apply a BeliefDelta to the belief state (immutable model_copy pattern)."""
        updated_entities: dict[str, EntityBelief] = dict(self._beliefs.entities)
        updated_individuals: dict[str, IndividualBelief] = dict(self._beliefs.individual_beliefs)

        # Apply entity updates
        for eid, belief in delta.entity_updates.items():
            updated_entities[eid] = belief

        # Apply entity additions (cap total at _MAX_ENTITY_BELIEFS)
        for eid, belief in delta.entity_additions.items():
            if len(updated_entities) < _MAX_ENTITY_BELIEFS:
                updated_entities[eid] = belief

        # Apply removals
        for eid in delta.entity_removals:
            updated_entities.pop(eid, None)

        # Apply individual updates
        for iid, ind_belief in delta.individual_updates.items():
            updated_individuals[iid] = ind_belief

        # Recompute overall confidence (mean of entity confidences)
        if updated_entities:
            mean_conf = sum(e.confidence for e in updated_entities.values()) / len(updated_entities)
        else:
            mean_conf = self._beliefs.overall_confidence

        self._beliefs = self._beliefs.model_copy(
            update={
                "entities": updated_entities,
                "individual_beliefs": updated_individuals,
                "overall_confidence": mean_conf,
            }
        )

    def _update_context(
        self,
        broadcast: WorkspaceBroadcast,
        precision: float,
    ) -> ContextBelief:
        """
        Derive an updated ContextBelief from the broadcast.
        Carries forward prior context with precision-weighted new evidence.
        """
        prior = self._beliefs.current_context

        # Extract content summary from broadcast
        content = broadcast.content
        content_text = ""
        if hasattr(content, "content") and hasattr(content.content, "content"):
            content_text = str(content.content.content or "")
        elif hasattr(content, "content") and isinstance(content.content, str):
            content_text = content.content
        elif isinstance(content, str):
            content_text = content

        # Estimate prediction error from salience scores
        pe_magnitude = 0.0
        if broadcast.salience and broadcast.salience.prediction_error:
            pe_magnitude = broadcast.salience.prediction_error.magnitude
        elif broadcast.salience:
            novelty = broadcast.salience.scores.get("novelty", 0.0)
            pe_magnitude = novelty * 0.8

        # Update domain estimate (carry forward unless new content is clear)
        domain = prior.domain
        if content_text:
            domain = _infer_domain(content_text) or prior.domain

        # Precision-weighted confidence update
        new_confidence = prior.confidence + precision * (1.0 - prior.confidence) * 0.3

        return ContextBelief(
            summary=content_text[:200] if content_text else prior.summary,
            domain=domain,
            is_active_dialogue=_is_dialogue(content_text),
            prediction_error_magnitude=pe_magnitude,
            confidence=min(1.0, new_confidence),
        )


# ─── Module-Level Update Functions ───────────────────────────────


def _bayesian_update_entity(
    prior: EntityBelief,
    precision: float,
) -> EntityBelief:
    """
    Precision-weighted Bayesian update for an entity belief.
    Evidence accumulation: confidence converges to 1.0 as more evidence arrives.

    new_confidence = old_confidence + precision × (1 - old_confidence)
    This is the discrete-time version of exponential approach to certainty.
    """
    new_confidence = prior.confidence + precision * (1.0 - prior.confidence)
    new_confidence = min(1.0, max(0.0, new_confidence))
    return prior.model_copy(
        update={
            "confidence": new_confidence,
            "last_observed": utc_now(),
        }
    )


def _update_individual_belief(
    individual_id: str,
    current: IndividualBelief | None,
    broadcast: WorkspaceBroadcast,
    precision: float,
) -> IndividualBelief:
    """
    Update beliefs about a specific individual from a broadcast.
    Affect state informs estimated valence; precision weights the update.
    """
    if current is None:
        current = IndividualBelief(individual_id=individual_id)

    # Use broadcast affect as a signal about the individual's emotional state
    affect = broadcast.affect
    # Precision-weighted update toward observed valence
    new_valence = (
        current.estimated_valence + precision * (affect.valence - current.estimated_valence) * 0.3
    )
    new_valence_conf = min(1.0, current.valence_confidence + precision * 0.1)

    # Update engagement from arousal signal
    new_engagement = (
        current.engagement_level + precision * (affect.arousal - current.engagement_level) * 0.2
    )

    return current.model_copy(
        update={
            "estimated_valence": max(-1.0, min(1.0, new_valence)),
            "valence_confidence": new_valence_conf,
            "engagement_level": max(0.0, min(1.0, new_engagement)),
            "last_updated": utc_now(),
        }
    )


def _extract_individual_id(broadcast: WorkspaceBroadcast) -> str | None:
    """Extract an individual's ID from a broadcast, if present."""
    content = broadcast.content
    # Check various paths where a speaker/addressee ID might be stored
    for path in [
        ("content", "speaker_id"),
        ("metadata", "speaker_id"),
        ("speaker_id",),
    ]:
        obj = content
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, str) and obj:
            return obj
    return None


def _infer_domain(text: str) -> str:
    """
    Lightweight heuristic domain inference from content.
    Used to categorise the current conversational context.
    """
    text_lower = text.lower()
    emotional_words = ["feel", "hurt", "sad", "worried", "scared", "happy", "excited"]
    if any(w in text_lower for w in emotional_words):
        return "emotional"
    if any(w in text_lower for w in ["code", "function", "api", "algorithm", "system", "error"]):
        return "technical"
    if any(w in text_lower for w in ["community", "member", "together", "group", "we "]):
        return "social"
    if any(w in text_lower for w in ["help", "assist", "please", "need", "want", "can you"]):
        return "request"
    return "general"


def _is_dialogue(text: str) -> bool:
    """Detect if the content is part of an active conversation."""
    if not text:
        return False
    text_lower = text.lower()
    # Questions, direct address, or conversational markers
    return (
        "?" in text
        or any(text_lower.startswith(p) for p in ["hey", "hi", "hello", "eos,", "can you", "could you"])  # noqa: E501
        or len(text) < 200  # Short messages are typically conversational
    )
