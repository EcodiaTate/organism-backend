"""
EcodiaOS - Nova Belief Updater

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

import json
from typing import TYPE_CHECKING, Any

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
    from clients.neo4j import Neo4jClient
    from systems.fovea.types import WorkspaceBroadcast

logger = structlog.get_logger()


# Maximum entities tracked in the belief state
_MAX_ENTITY_BELIEFS = 200

# ─── Belief Urgency Monitor (NOVA-ECON-3) ────────────────────────────────────
#
# Tracks high-priority belief entity keys.  When any monitored belief's
# confidence changes by more than _URGENCY_THRESHOLD, the registered
# callback fires immediately - enabling Nova to replan within 50ms rather
# than waiting for the next theta heartbeat.
#
# This closes the gap where Nova updated beliefs passively (beliefs were
# state, not planning inputs).  Now a >20% swing in a priority belief key
# triggers _immediate_deliberation() synchronously via callback.

_URGENCY_THRESHOLD: float = 0.20  # 20% confidence change → trigger

# Belief entity IDs that warrant immediate replanning on confidence shift
_PRIORITY_BELIEF_KEYS: frozenset[str] = frozenset({
    "wallet_balance",
    "bounty_success_rate",
    "yield_apy_aave",
    "yield_apy_morpho",
    "yield_apy_compound",
    "revenue_burn_ratio",
    "economic_risk_level",
})


class BeliefUrgencyMonitor:
    """
    Watches high-priority belief entities and fires a callback when
    their confidence shifts by more than _URGENCY_THRESHOLD (20%).

    Usage:
        monitor = BeliefUrgencyMonitor(callback=nova._immediate_deliberation)
        belief_updater.set_urgency_monitor(monitor)

    The callback receives keyword args: reason (str), urgency (float).
    It is called fire-and-forget via asyncio.create_task to avoid blocking
    the belief-update hot path.
    """

    def __init__(self, callback: Any) -> None:
        # callback: async def(reason: str, urgency: float) -> None
        self._callback = callback
        self._prev_confidences: dict[str, float] = {}
        self._logger = logger.bind(system="nova.belief_urgency_monitor")

    def check(self, entity_id: str, new_confidence: float) -> None:
        """
        Compare new_confidence against last-known value for entity_id.
        If entity_id is a priority key and delta > threshold, fire callback.

        Called synchronously from BeliefUpdater after each entity update.
        Must complete in <1ms - no I/O, no blocking.
        """
        if entity_id not in _PRIORITY_BELIEF_KEYS:
            return

        prev = self._prev_confidences.get(entity_id)
        self._prev_confidences[entity_id] = new_confidence

        if prev is None:
            return  # First observation, no baseline to compare

        delta = abs(new_confidence - prev)
        if delta <= _URGENCY_THRESHOLD:
            return

        # Urgency scales with the size of the confidence swing
        urgency = min(1.0, 0.5 + delta * 2.0)
        reason = f"belief_shift:{entity_id}:{prev:.3f}->{new_confidence:.3f}"
        self._logger.info(
            "belief_urgency_triggered",
            entity_id=entity_id,
            prev=round(prev, 3),
            new=round(new_confidence, 3),
            delta=round(delta, 3),
            urgency=round(urgency, 3),
        )

        # Fire-and-forget - never block the belief update path
        import asyncio
        try:
            asyncio.get_event_loop().create_task(  # type: ignore[attr-defined]
                self._callback(reason=reason, urgency=urgency),
            )
        except RuntimeError:
            # No running event loop (e.g. unit tests) - skip
            pass
# Confidence decay per cycle for unobserved entities (forgetting)
_CONFIDENCE_DECAY = 0.005
# Minimum confidence before entity belief is pruned
_MIN_ENTITY_CONFIDENCE = 0.05


class BeliefUpdater:
    """
    Maintains Nova's belief state and updates it from workspace broadcasts.

    The belief state is Nova's map of the world - what it knows, how confident
    it is, and where the prediction errors are. This drives deliberation:
    which goals are relevant, which policies can work.
    """

    def __init__(self) -> None:
        self._beliefs = BeliefState()
        self._logger = logger.bind(system="nova.belief_updater")
        self._urgency_monitor: BeliefUrgencyMonitor | None = None

    def set_urgency_monitor(self, monitor: BeliefUrgencyMonitor) -> None:
        """Wire an urgency monitor to trigger immediate deliberation on belief shifts."""
        self._urgency_monitor = monitor

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

        # ── EIS threat attenuation (integration.py - belief_update_weight) ──
        # If the percept was screened by EIS, attenuate the belief update precision
        # proportional to the threat score.  A clean percept passes through at 1.0×;
        # a high-threat percept is discounted toward BELIEF_FLOOR so adversarial
        # inputs cannot poison the belief state even if they slip past quarantine.
        content = broadcast.content
        if content is not None and hasattr(content, "metadata") and isinstance(getattr(content, "metadata", None), dict):
            try:
                from systems.eis.integration import belief_update_weight as _eis_weight
                eis_w = _eis_weight(content)  # type: ignore[arg-type]
                if eis_w < 1.0:
                    precision = precision * eis_w
                    self._logger.debug(
                        "belief_update_precision_attenuated_by_eis",
                        original_precision=round(broadcast.precision, 4),
                        eis_weight=round(eis_w, 4),
                        attenuated_precision=round(precision, 4),
                    )
            except Exception:
                pass  # EIS integration is non-fatal; degrade gracefully

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
        This is the 'forgetting' mechanism - beliefs weaken without evidence.
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
        if self._urgency_monitor is not None:
            self._urgency_monitor.check(entity_id, confidence)

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
        # Check urgency monitor for confidence shifts on priority keys
        new_confidence = fields.get("confidence")
        if self._urgency_monitor is not None and isinstance(new_confidence, float):
            self._urgency_monitor.check(entity_id, new_confidence)
        return True

    # ─── Neo4j Persistence ─────────────────────────────────────────

    def set_neo4j(self, neo4j: Neo4jClient) -> None:
        """Wire Neo4j client for belief persistence."""
        self._neo4j = neo4j
        self._pending_writes: list[EntityBelief] = []
        self._write_batch_size: int = 10

    async def persist_beliefs(self) -> int:
        """
        Batch-persist dirty beliefs to Neo4j.

        Called after belief updates accumulate. Max 1 transaction per
        10 belief changes to avoid excessive Neo4j writes.

        Returns the number of beliefs persisted.
        """
        neo4j = getattr(self, "_neo4j", None)
        if neo4j is None:
            return 0

        pending = getattr(self, "_pending_writes", [])
        if not pending:
            return 0

        batch = pending[:self._write_batch_size]
        self._pending_writes = pending[self._write_batch_size:]

        persisted = 0
        try:
            params_list: list[dict[str, Any]] = []
            for belief in batch:
                props_json = json.dumps(
                    belief.properties, sort_keys=True, default=str,
                )
                params_list.append({
                    "entity_id": belief.entity_id,
                    "name": belief.name,
                    "entity_type": belief.entity_type,
                    "confidence": belief.confidence,
                    "properties_json": props_json,
                    "last_observed": belief.last_observed.isoformat(),
                })

            await neo4j.execute_write(
                """
                UNWIND $beliefs AS b
                MERGE (eb:EntityBelief {entity_id: b.entity_id})
                SET eb.name = b.name,
                    eb.entity_type = b.entity_type,
                    eb.confidence = b.confidence,
                    eb.properties_json = b.properties_json,
                    eb.updated_at = datetime(b.last_observed)
                ON CREATE SET eb.created_at = datetime(b.last_observed),
                              eb.id = b.entity_id
                """,
                {"beliefs": params_list},
            )
            persisted = len(batch)

        except Exception as exc:
            self._logger.warning(
                "belief_persist_failed",
                error=str(exc),
                batch_size=len(batch),
            )

        return persisted

    async def persist_belief_relationships(self) -> None:
        """
        Persist SUPPORTS/CONTRADICTS relationships between beliefs.

        Relationships are inferred from belief conflict history:
        beliefs with high prediction error against each other get
        a CONTRADICTS edge; co-occurring beliefs get SUPPORTS.
        """
        neo4j = getattr(self, "_neo4j", None)
        if neo4j is None:
            return

        # Build SUPPORTS edges: entities that co-occur (same type, close confidence)
        entities = list(self._beliefs.entities.values())
        supports: list[dict[str, str]] = []
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                if (
                    a.entity_type == b.entity_type
                    and a.entity_type
                    and abs(a.confidence - b.confidence) < 0.2
                ):
                    supports.append({
                        "from_id": a.entity_id,
                        "to_id": b.entity_id,
                    })
                    if len(supports) >= 50:
                        break
            if len(supports) >= 50:
                break

        if supports:
            try:
                await neo4j.execute_write(
                    """
                    UNWIND $rels AS r
                    MATCH (a:EntityBelief {entity_id: r.from_id})
                    MATCH (b:EntityBelief {entity_id: r.to_id})
                    MERGE (a)-[:SUPPORTS]->(b)
                    """,
                    {"rels": supports},
                )
            except Exception as exc:
                self._logger.debug("belief_relationship_persist_failed", error=str(exc))

    async def restore_from_neo4j(self) -> int:
        """
        Restore beliefs from Neo4j on startup.

        Returns the number of beliefs restored.
        """
        neo4j = getattr(self, "_neo4j", None)
        if neo4j is None:
            return 0

        try:
            rows = await neo4j.execute_read(
                """
                MATCH (b:EntityBelief)
                WHERE b.confidence > $min_confidence
                RETURN b.entity_id AS entity_id,
                       b.name AS name,
                       b.entity_type AS entity_type,
                       b.confidence AS confidence,
                       b.properties_json AS properties_json,
                       b.updated_at AS updated_at
                ORDER BY b.confidence DESC
                LIMIT $limit
                """,
                {"min_confidence": _MIN_ENTITY_CONFIDENCE, "limit": _MAX_ENTITY_BELIEFS},
            )
        except Exception as exc:
            self._logger.warning("belief_restore_failed", error=str(exc))
            return 0

        restored = 0
        entities: dict[str, EntityBelief] = {}
        for row in rows:
            try:
                props_raw = row.get("properties_json", "{}")
                properties: dict[str, Any] = {}
                if isinstance(props_raw, str):
                    try:
                        properties = json.loads(props_raw)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(props_raw, dict):
                    properties = props_raw

                entity = EntityBelief(
                    entity_id=str(row.get("entity_id", "")),
                    name=str(row.get("name", "")),
                    entity_type=str(row.get("entity_type", "")),
                    confidence=float(row.get("confidence", 0.5)),
                    properties=properties,
                )
                entities[entity.entity_id] = entity
                restored += 1
            except Exception:
                continue

        if entities:
            self._beliefs = self._beliefs.model_copy(update={"entities": entities})
            self._logger.info("beliefs_restored_from_neo4j", count=restored)

        return restored

    def mark_dirty(self, *beliefs: EntityBelief) -> None:
        """Mark beliefs as needing persistence on next flush."""
        pending = getattr(self, "_pending_writes", None)
        if pending is not None:
            pending.extend(beliefs)

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

        # Mark changed entities for Neo4j persistence
        dirty = list(delta.entity_updates.values()) + list(delta.entity_additions.values())
        if dirty:
            self.mark_dirty(*dirty)

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
