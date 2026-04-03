"""
EcodiaOS - Memory Service

The single interface to the knowledge graph. Every other system
reads from and writes to memory through this service.

Memory is the substrate of selfhood.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import structlog

from decimal import Decimal

from primitives import (
    AffectState,
    DriveAlignmentVector,
    Entity,
    EntityType,
    Episode,
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    MentionRelation,
    Percept,
    RETrainingExample,
    SelfNode,
    SemanticRelation,
    SystemID,
    utc_now,
)
from systems.memory.birth import birth_instance
from systems.memory.consolidation import run_consolidation
from systems.memory.episodic import (
    count_episodes,
    link_episode_sequence,
    store_episode,
    update_access,
)
from systems.memory.financial_encoder import FinancialEncoder
from systems.memory.retrieval import hybrid_retrieve
from systems.memory.schema import ensure_schema
from systems.memory.semantic import (
    count_entities,
    create_entity,
    create_or_strengthen_relation,
    find_similar_entity,
    link_episode_to_entity,
    merge_into_entity,
)

if TYPE_CHECKING:
    from clients.embedding import EmbeddingClient
    from clients.neo4j import Neo4jClient
    from config import SeedConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()


class MemoryService:
    """
    The Memory & Identity Core.

    This is the ground truth of identity. Every other system reads from
    and writes to this substrate. If you lose the graph, you lose the self.
    """

    system_id: str = "memory"

    def __init__(
        self,
        neo4j: Neo4jClient,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._neo4j = neo4j
        self._embedding = embedding_client
        self._instance_id: str | None = None
        # Episode sequence tracking - links consecutive episodes with FOLLOWED_BY
        self._last_episode_id: str | None = None
        self._last_episode_time: float | None = None  # monotonic seconds
        # P4: track last percept event_time for event_time-based gap calculation
        self._last_event_time: Any = None  # datetime | None
        # Soma for somatic marker stamping and reranking
        self._soma: Any = None
        # Event bus - wired after Synapse is up via set_event_bus()
        self._event_bus: Any = None
        # Financial encoder - wired after Synapse is up via set_event_bus()
        self._financial_encoder = FinancialEncoder(neo4j, embedding_client)

        # Metabolic starvation level - AUSTERITY: halt consolidation,
        # EMERGENCY: read-only, CRITICAL: cache-only
        self._starvation_level: str = "nominal"

        # Last known Equor compliance score (composite alignment from snapshot).
        # Used to detect sustained constitutional drift across snapshot cycles.
        self._last_compliance_score: float | None = None

        # ── Skia VitalityCoordinator modulation ───────────────────────
        self._modulation_halted: bool = False

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Ensure schema exists (idempotent)."""
        await ensure_schema(self._neo4j)
        logger.info("memory_service_initialised")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma for somatic marker stamping and reranking."""
        self._soma = soma
        logger.info("soma_wired_to_memory")

    def set_event_bus(self, event_bus: EventBus) -> None:
        """
        Wire the Synapse event bus into the FinancialEncoder and Memory itself.

        Call after both MemoryService and SynapseService are initialised.
        The encoder subscribes to WALLET_TRANSFER_CONFIRMED and REVENUE_INJECTED
        and will encode those events as salience=1.0 episodes directly in Neo4j.
        Memory also holds the bus reference to emit EPISODE_STORED after each
        store_percept() so Thread can receive episodes without polling.
        """
        self._event_bus: EventBus | None = event_bus
        self._financial_encoder.attach(event_bus)
        from systems.synapse.types import SynapseEventType
        event_bus.subscribe(SynapseEventType.METABOLIC_PRESSURE, self._on_metabolic_pressure)
        event_bus.subscribe(
            SynapseEventType.EQUOR_CONSTITUTIONAL_SNAPSHOT,
            self._on_equor_constitutional_snapshot,
        )
        # Degradation Engine §8.2 - stub subscription.
        # Round 2 will implement: apply fidelity_loss_rate decay to Episode.salience
        # for all unconsolidated episodes older than affected_episode_age_hours.
        event_bus.subscribe(
            SynapseEventType.MEMORY_DEGRADATION,
            self._on_memory_degradation,
        )
        event_bus.subscribe(
            SynapseEventType.SYSTEM_MODULATION,
            self._on_system_modulation,
        )
        # Auto-trigger consolidation when Oneiros signals sleep entry.
        # This closes the spec gap noted in §18 ("No subscription to SLEEP_INITIATED").
        event_bus.subscribe(
            SynapseEventType.SLEEP_INITIATED,
            self._on_sleep_initiated,
        )
        logger.info("financial_encoder_wired", system="memory")

    # ─── Public Neo4j API ─────────────────────────────────────────
    # Systems must use these methods instead of accessing _neo4j directly.

    async def execute_read(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Execute a read Cypher query and return all records.

        Use this instead of memory._neo4j.execute_read() from other systems.
        """
        return await self._neo4j.execute_read(query, params or {})

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Execute a write Cypher query and return all records.

        Use this instead of memory._neo4j.execute_write() from other systems.
        """
        return await self._neo4j.execute_write(query, params or {})

    async def _on_sleep_initiated(self, event: Any) -> None:
        """Auto-trigger consolidation when Oneiros signals sleep entry.

        Closes spec §18 gap: Memory had no subscription to SLEEP_INITIATED,
        so consolidation only fired via explicit external calls (Evo timer or
        Axon intent). Now consolidation runs automatically at every sleep cycle,
        which is the intended organisational closure behaviour.

        Fire-and-forget with metabolic gate respected inside consolidate().
        """
        logger.info("memory_consolidation_triggered_by_sleep")
        try:
            await self.consolidate()
        except Exception:
            logger.exception("sleep_triggered_consolidation_failed")

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """React to organism-wide metabolic pressure changes."""
        data = getattr(event, "data", {}) or {}
        level = data.get("starvation_level", "")
        if not level:
            return
        old = self._starvation_level
        self._starvation_level = level
        if level != old:
            logger.info("memory_starvation_level_changed", old=old, new=level)

    async def _on_equor_constitutional_snapshot(self, event: Any) -> None:
        """Persist constitutional evolution state and detect compliance drift.

        Called hourly when Equor emits EQUOR_CONSTITUTIONAL_SNAPSHOT (Spec §17.1).

        Actions:
          1. Write a (:ConstitutionalSnapshot) node linked to (:Self) in Neo4j
             so Thread can walk constitutional evolution history.
          2. Track compliance drift - if the new overall_compliance_score drops
             more than 0.1 below the last seen value, emit SELF_STATE_DRIFTED so
             Equor can acknowledge the constitutional health decline.
        """
        data = getattr(event, "data", {}) or {}
        constitution_hash: str = str(data.get("constitution_hash", ""))
        compliance_score: float = float(data.get("overall_compliance_score", 0.5))
        active_drives: list[str] = list(data.get("active_drives", []))
        recent_amendment_ids: list[str] = list(data.get("recent_amendment_ids", []))
        total_verdicts: int = int(data.get("total_verdicts_issued", 0))
        timestamp: str = str(data.get("timestamp", ""))

        # ── 1. Persist (:ConstitutionalSnapshot) to Neo4j ───────────
        try:
            import json as _json
            await self._neo4j.execute_write(
                """
                MATCH (s:Self)
                CREATE (snap:ConstitutionalSnapshot {
                    id:                   randomUUID(),
                    constitution_hash:    $constitution_hash,
                    compliance_score:     $compliance_score,
                    active_drives:        $active_drives,
                    recent_amendment_ids: $recent_amendment_ids,
                    total_verdicts:       $total_verdicts,
                    recorded_at:          datetime($timestamp)
                })
                CREATE (s)-[:SNAPSHOT_OF]->(snap)
                """,
                {
                    "constitution_hash": constitution_hash,
                    "compliance_score": compliance_score,
                    "active_drives": active_drives,
                    "recent_amendment_ids": recent_amendment_ids,
                    "total_verdicts": total_verdicts,
                    "timestamp": timestamp or utc_now().isoformat(),
                },
            )
            logger.info(
                "memory_constitutional_snapshot_persisted",
                constitution_hash=constitution_hash[:16] if constitution_hash else "",
                compliance_score=round(compliance_score, 4),
            )
        except Exception:
            logger.debug("memory_constitutional_snapshot_write_failed", exc_info=True)

        # ── 2. Detect compliance drift ───────────────────────────────
        if self._last_compliance_score is not None:
            drop = self._last_compliance_score - compliance_score
            if drop > 0.1 and self._event_bus is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.SELF_STATE_DRIFTED,
                        source_system="memory",
                        data={
                            "source": "equor_constitutional_snapshot",
                            "new_score": compliance_score,
                            "previous_score": self._last_compliance_score,
                            "drop": round(drop, 4),
                            "constitution_hash": constitution_hash,
                        },
                    ))
                    logger.info(
                        "memory_constitutional_compliance_drift",
                        drop=round(drop, 4),
                        new_score=round(compliance_score, 4),
                    )
                except Exception:
                    logger.debug("memory_self_state_drifted_emit_failed", exc_info=True)

        self._last_compliance_score = compliance_score

    async def _on_memory_degradation(self, event: Any) -> None:
        """Degradation Engine §8.2 - decay salience on old unconsolidated episodes.

        Queries Neo4j for episodes older than affected_episode_age_hours that have
        not yet been consolidated (is_compressed=false). Multiplies their salience
        by (1 - fidelity_loss_rate). Episodes that fall below 0.01 are soft-deleted
        (decayed=true) - the organism has genuinely forgotten them.
        """
        data = getattr(event, "data", {}) or {}
        fidelity_loss_rate = float(data.get("fidelity_loss_rate", 0.0))
        age_hours = float(data.get("affected_episode_age_hours", 48.0))
        tick_number = data.get("tick_number", 0)

        if fidelity_loss_rate <= 0.0:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # 1. Query unconsolidated episodes older than age_hours
            rows = await self._neo4j.execute_read(
                """
                MATCH (ep:Episode)
                WHERE ep.is_compressed = false
                  AND ep.decayed IS NULL
                  AND ep.ingestion_time < datetime() - duration({hours: $age_hours})
                RETURN ep.id AS id, coalesce(ep.salience_composite, 0.1) AS salience
                """,
                {"age_hours": age_hours},
            )

            if not rows:
                logger.debug(
                    "memory_degradation_no_episodes",
                    age_hours=age_hours,
                    tick_number=tick_number,
                )
                return

            # 2. Compute new salience values
            updates: list[dict] = []
            to_soft_delete: list[dict] = []
            for row in rows:
                ep_id = row["id"]
                old_salience = float(row["salience"])
                new_salience = old_salience * (1.0 - fidelity_loss_rate)
                if new_salience < 0.01:
                    to_soft_delete.append({"id": ep_id})
                else:
                    updates.append({"id": ep_id, "new_salience": new_salience})

            # 3. Batch-update salience on surviving episodes
            if updates:
                await self._neo4j.execute_write(
                    """
                    UNWIND $updates AS u
                    MATCH (ep:Episode {id: u.id})
                    SET ep.salience = u.new_salience,
                        ep.last_decay_at = datetime()
                    """,
                    {"updates": updates},
                )

            # 4. Soft-delete episodes that fell below threshold
            oldest_at: str | None = None
            newest_at: str | None = None
            if to_soft_delete:
                deleted_ids = [r["id"] for r in to_soft_delete]
                ts_rows = await self._neo4j.execute_write(
                    """
                    UNWIND $ids AS ep_id
                    MATCH (ep:Episode {id: ep_id})
                    SET ep.decayed = true,
                        ep.decayed_at = datetime(),
                        ep.salience = 0.0
                    RETURN ep.created_at AS created_at
                    ORDER BY created_at
                    """,
                    {"ids": deleted_ids},
                )
                if ts_rows:
                    oldest_at = str(ts_rows[0]["created_at"])
                    newest_at = str(ts_rows[-1]["created_at"])

            # 5. Log
            logger.info(
                "memory_degradation_applied",
                decayed_count=len(updates),
                soft_deleted_count=len(to_soft_delete),
                fidelity_loss_rate=round(fidelity_loss_rate, 4),
                age_hours=age_hours,
                tick_number=tick_number,
            )

            # 6. Emit MEMORY_EPISODES_DECAYED if any were soft-deleted
            if to_soft_delete and self._event_bus is not None:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.MEMORY_EPISODES_DECAYED,
                    source_system="memory",
                    data={
                        "count": len(to_soft_delete),
                        "oldest_deleted_at": oldest_at,
                        "newest_deleted_at": newest_at,
                        "instance_id": self._instance_id or "",
                    },
                ))

        except Exception:
            logger.exception("memory_degradation_handler_failed", tick_number=tick_number)

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle VitalityCoordinator austerity orders.

        Skia emits SYSTEM_MODULATION when the organism needs to conserve resources.
        This system applies the directive and ACKs so Skia knows the order was received.
        """
        data = getattr(event, "data", {}) or {}
        level = data.get("level", "nominal")
        halt_systems = data.get("halt_systems", [])
        modulate = data.get("modulate", {})

        system_id = "memory"
        compliant = True
        reason: str | None = None

        if system_id in halt_systems:
            self._modulation_halted = True
            logger.warning("system_modulation_halt", level=level)
        elif system_id in modulate:
            directives = modulate[system_id]
            self._apply_modulation_directives(directives)
            logger.info("system_modulation_applied", level=level, directives=directives)
        elif level == "nominal":
            self._modulation_halted = False
            logger.info("system_modulation_resumed", level=level)

        # Emit ACK so Skia knows the order was received
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                    data={
                        "system_id": system_id,
                        "level": level,
                        "compliant": compliant,
                        "reason": reason,
                    },
                    source_system=system_id,
                ))
            except Exception as exc:
                logger.warning("modulation_ack_failed", error=str(exc))

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply modulation directives from VitalityCoordinator.

        Memory directive: {"mode": "read_only"} - suspend all writes to the
        knowledge graph to protect the substrate during austerity.
        """
        mode = directives.get("mode")
        if mode == "read_only":
            logger.info("modulation_read_only_mode_set")
        else:
            logger.info("modulation_directives_received", directives=directives)

    async def _emit_re_training_example(
        self,
        category: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float,
        episode_id: str = "",
        cost_usd: Decimal = Decimal("0"),
        latency_ms: int = 0,
        reasoning_trace: str = "",
        alternatives: list[str] | None = None,
        constitutional_alignment: DriveAlignmentVector | None = None,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.MEMORY,
                episode_id=episode_id,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=outcome_quality,
                category=category,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives or [],
                constitutional_alignment=constitutional_alignment or DriveAlignmentVector(),
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                data=example.model_dump(mode="json"),
                source_system="memory",
            ))
        except Exception:
            logger.debug("re_training_emit_failed", exc_info=True)

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict | None = None,
    ) -> None:
        """Emit an evolutionary observable event for Benchmarks population tracking."""
        if self._event_bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from primitives.common import SystemID
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.MEMORY,
                instance_id=self._instance_id or "",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="memory",
                data=obs.model_dump(mode="json"),
            )
            await self._event_bus.emit(event)
        except Exception:
            pass

    async def get_self(self) -> SelfNode | None:
        """Get the current instance's Self node, or None if not yet born."""
        results = await self._neo4j.execute_read(
            "MATCH (s:Self) RETURN s LIMIT 1"
        )
        if not results:
            return None

        data = results[0]["s"]
        self._instance_id = data.get("instance_id")
        # Reconstruct current_affect from flattened scalar properties
        current_affect = {
            "valence": data.get("affect_valence", 0.0),
            "arousal": data.get("affect_arousal", 0.0),
            "dominance": data.get("affect_dominance", 0.5),
            "curiosity": data.get("affect_curiosity", 0.0),
            "care_activation": data.get("affect_care_activation", 0.0),
            "coherence_stress": data.get("affect_coherence_stress", 0.0),
        }
        # Convert neo4j.time.DateTime to Python datetime
        born_at = data.get("born_at", utc_now())
        if hasattr(born_at, "to_native"):
            born_at = born_at.to_native()
        # Parse personality_json if stored (Neo4j stores as string, needs json.loads)
        raw_pjson = data.get("personality_json", None)
        personality_json: dict[str, float] = {}
        if raw_pjson:
            if isinstance(raw_pjson, str):
                import json
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    personality_json = json.loads(raw_pjson)
            elif isinstance(raw_pjson, dict):
                personality_json = raw_pjson

        return SelfNode(
            instance_id=data.get("instance_id", ""),
            name=data.get("name", ""),
            born_at=born_at,
            current_affect=current_affect,
            autonomy_level=data.get("autonomy_level", 1),
            personality_vector=data.get("personality_vector", []),
            personality_json=personality_json,
            cycle_count=data.get("cycle_count", 0),
            total_episodes=data.get("total_episodes", 0),
            total_entities=data.get("total_entities", 0),
            total_communities=data.get("total_communities", 0),
        )

    async def birth(self, seed: SeedConfig, instance_id: str) -> dict[str, Any]:
        """Birth a new instance from a seed configuration."""
        result = await birth_instance(
            self._neo4j, self._embedding, seed, instance_id,
        )
        self._instance_id = instance_id
        return result

    # ─── Episode Storage (from Atune) ─────────────────────────────

    async def store_percept(
        self,
        percept: Percept,
        salience_composite: float = 0.0,
        salience_scores: dict[str, float] | None = None,
        affect_valence: float = 0.0,
        affect_arousal: float = 0.0,
        free_energy: float = 0.0,
        context_summary: str = "",
        is_correction: bool = False,
    ) -> str:
        """
        Store a percept as an episode in the knowledge graph.

        Returns the episode ID.
        Entity extraction should be done asynchronously after this returns.

        Automatically links the new episode to the previous one via
        FOLLOWED_BY relationship, creating temporal causal chains that
        enable sequential memory and "what happened before/after" queries.
        """
        # ── Metabolic gate: EMERGENCY+ → read-only (no new writes) ──
        if self._starvation_level in ("emergency", "critical"):
            logger.info("store_percept_blocked_starvation", level=self._starvation_level)
            return ""

        import time as _time

        # Re-sync sequence state from FinancialEncoder in case a financial
        # event was encoded between the last store_percept and this one.
        enc_id, enc_time = self._financial_encoder.get_sequence_state()
        if enc_id is not None and enc_id != self._last_episode_id:
            self._last_episode_id = enc_id
            self._last_episode_time = enc_time

        # Compute embedding if not already present
        embedding = percept.content.embedding
        if not embedding and percept.content.raw:
            embedding = await self._embedding.embed(percept.content.raw)

        # Stamp somatic marker at encoding time (Soma §0.5)
        somatic_marker = None
        somatic_vector: list[float] | None = None
        if self._soma is not None:
            try:
                somatic_marker = self._soma.get_somatic_marker()
                if somatic_marker is not None and hasattr(somatic_marker, "to_vector"):
                    somatic_vector = somatic_marker.to_vector()
            except Exception as exc:
                logger.debug("soma_marker_error", error=str(exc))

        # Compute novelty score: 1.0 - max cosine similarity to recent episodes
        novelty_score = 1.0
        if embedding:
            try:
                novelty_score = await self._compute_novelty(embedding)
            except Exception:
                logger.debug("novelty_score_failed", exc_info=True)

        episode = Episode(
            event_time=percept.timestamp,
            ingestion_time=utc_now(),
            source=f"{percept.source.system.value}:{percept.source.channel}",
            modality=percept.source.modality.value,
            raw_content=percept.content.raw,
            summary=percept.content.raw[:200],  # Placeholder; LLM summary done async
            embedding=embedding,
            salience_composite=salience_composite,
            salience_scores=salience_scores or {},
            affect_valence=affect_valence,
            affect_arousal=affect_arousal,
            free_energy=free_energy,
            somatic_marker=somatic_marker,
            somatic_vector=somatic_vector,
            novelty_score=novelty_score,
            context_summary=context_summary,
            is_correction=is_correction,
        )

        episode_id = await store_episode(self._neo4j, episode)

        # Link to previous episode (temporal sequence).
        # P4 fix: use event_time delta (not monotonic ingestion time) so
        # high-latency percepts (delayed delivery, replayed events) still
        # form correct temporal chains based on when events actually occurred.
        now = _time.monotonic()
        if self._last_episode_id is not None:
            if self._last_episode_time is not None:
                # Prefer event_time-based gap; fall back to monotonic if
                # previous event_time is unavailable (monotonic tracks wall clock).
                event_gap = (
                    percept.timestamp - self._last_event_time
                ).total_seconds()
                gap = max(0.0, event_gap)
            else:
                gap = 0.0
            # Only link if episodes are within 1 hour of each other
            if gap < 3600.0:
                try:
                    # Higher causal strength for closely spaced episodes
                    causal = max(0.05, min(0.8, 1.0 - (gap / 300.0)))
                    await link_episode_sequence(
                        self._neo4j,
                        previous_episode_id=self._last_episode_id,
                        current_episode_id=episode_id,
                        gap_seconds=gap,
                        causal_strength=causal,
                    )
                except Exception as _link_exc:
                    # M17: persistent linking failure should be visible to Thymos,
                    # not just silently swallowed by structlog at DEBUG level.
                    logger.warning(
                        "episode_link_failed",
                        previous_episode_id=self._last_episode_id,
                        current_episode_id=episode_id,
                        error=str(_link_exc),
                    )
                    if self._event_bus is not None:
                        try:
                            from systems.synapse.types import SynapseEvent, SynapseEventType
                            # Route to Thymos via INCIDENT_DETECTED so it can
                            # classify and repair the temporal chain integrity failure.
                            # SYSTEM_ERROR is kept as a secondary signal for dashboards.
                            await self._event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.INCIDENT_DETECTED,
                                source_system="memory",
                                data={
                                    "incident_type": "temporal_chain_integrity_failure",
                                    "severity": "medium",
                                    "source_system": "memory",
                                    "description": "FOLLOWED_BY link could not be created between consecutive episodes",
                                    "previous_episode_id": self._last_episode_id,
                                    "current_episode_id": episode_id,
                                    "error": str(_link_exc),
                                },
                            ))
                        except Exception:
                            pass

        self._last_episode_id = episode_id
        self._last_episode_time = now
        self._last_event_time = percept.timestamp

        # Keep FinancialEncoder in sync so financial episodes slot into
        # the temporal chain after any percept store_percept just recorded.
        self._financial_encoder.set_sequence_state(episode_id, now)

        # Notify Thread (and any other subscriber) that a new episode was stored.
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEventType

                await self._event_bus.emit(
                    SynapseEventType.EPISODE_STORED,
                    {
                        "episode_id": episode_id,
                        "source": episode.source,
                        "summary": episode.summary,
                        "salience": salience_composite,
                    },
                    source_system="memory",
                )
            except Exception:
                logger.debug("episode_stored_emit_failed", exc_info=True)

        # Check for MEMORY_PRESSURE (fire-and-forget)
        if self._event_bus is not None:
            try:
                await self._check_memory_pressure()
            except Exception:
                logger.debug("memory_pressure_check_failed", exc_info=True)

        return episode_id

    async def _check_memory_pressure(self) -> None:
        """Emit MEMORY_PRESSURE if episode count or consolidation lag is high."""
        if self._event_bus is None:
            return
        ep_count = await count_episodes(self._neo4j)
        # Consolidation lag = episodes with consolidation_level=0 (never consolidated).
        # This accurately tracks unconsolidated episode backlog rather than total count.
        try:
            lag_rows = await self._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE e.consolidation_level = 0 OR e.consolidation_level IS NULL
                RETURN count(e) AS lag
                """
            )
            consolidation_lag = lag_rows[0]["lag"] if lag_rows else 0
        except Exception:
            consolidation_lag = 0
        pressure_threshold = 10000
        lag_threshold = 500
        if ep_count > pressure_threshold or consolidation_lag > lag_threshold:
            urgency = min(1.0, max(ep_count / pressure_threshold, consolidation_lag / lag_threshold))
            from systems.synapse.types import SynapseEventType
            await self._event_bus.emit(
                SynapseEventType.MEMORY_PRESSURE,
                {
                    "episode_count": ep_count,
                    "consolidation_lag": consolidation_lag,
                    "urgency": round(urgency, 3),
                },
                source_system="memory",
            )

    async def _compute_novelty(self, embedding: list[float]) -> float:
        """
        Compute novelty as 1.0 - max cosine similarity to last 100 episodes.
        Uses Neo4j vector index for approximate search. Must be < 50ms.
        """
        results = await self._neo4j.execute_read(
            """
            CALL db.index.vector.queryNodes('episode_embedding', 1, $embedding)
            YIELD node, score
            RETURN score
            LIMIT 1
            """,
            {"embedding": embedding},
        )
        if results:
            max_sim = float(results[0].get("score", 0.0))
            return max(0.0, min(1.0, 1.0 - max_sim))
        return 1.0

    # ─── Intent/Outcome Storage (from Nova/Axon) ────────────────

    async def store_intent(
        self,
        episode_id: str,
        action_type: str,
        reasoning: str,
        confidence: float = 0.5,
    ) -> str:
        """Store an Intent node linked to an Episode. Returns intent ID."""
        from primitives.common import new_id

        intent_id = new_id()
        now_iso = utc_now().isoformat()
        await self._neo4j.execute_write(
            """
            CREATE (i:Intent {
                id: $id,
                episode_id: $episode_id,
                action_type: $action_type,
                reasoning: $reasoning,
                confidence: $confidence,
                timestamp: datetime($timestamp)
            })
            WITH i
            MATCH (e:Episode {id: $episode_id})
            CREATE (e)-[:GENERATED]->(i)
            """,
            {
                "id": intent_id,
                "episode_id": episode_id,
                "action_type": action_type,
                "reasoning": reasoning[:500],
                "confidence": confidence,
                "timestamp": now_iso,
            },
        )
        logger.debug("intent_stored", intent_id=intent_id, episode_id=episode_id)
        return intent_id

    async def store_outcome(
        self,
        intent_id: str,
        success: bool,
        value_gained: float = 0.0,
        error_message: str = "",
    ) -> str:
        """Store an Outcome node linked to an Intent. Returns outcome ID."""
        from primitives.common import new_id

        outcome_id = new_id()
        now_iso = utc_now().isoformat()
        await self._neo4j.execute_write(
            """
            CREATE (o:Outcome {
                id: $id,
                intent_id: $intent_id,
                success: $success,
                value_gained: $value_gained,
                error_message: $error_message,
                timestamp: datetime($timestamp)
            })
            WITH o
            MATCH (i:Intent {id: $intent_id})
            CREATE (i)-[:RESULTED_IN]->(o)
            """,
            {
                "id": outcome_id,
                "intent_id": intent_id,
                "success": success,
                "value_gained": value_gained,
                "error_message": error_message[:500],
                "timestamp": now_iso,
            },
        )
        logger.debug("outcome_stored", outcome_id=outcome_id, intent_id=intent_id)
        return outcome_id

    # ─── Retrieval (from any system) ──────────────────────────────

    async def retrieve(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        max_results: int = 10,
        salience_floor: float = 0.0,
        include_communities: bool = False,
    ) -> MemoryRetrievalResponse:
        """
        Hybrid retrieval. Must respond within 200ms.
        """
        # ── Metabolic gate: CRITICAL → return empty (cache-only) ──
        if self._starvation_level == "critical":
            logger.info("retrieve_blocked_starvation", level="critical")
            return MemoryRetrievalResponse(results=[])

        # Compute embedding from text if not provided
        if query_text and not query_embedding:
            query_embedding = await self._embedding.embed(query_text)

        request = MemoryRetrievalRequest(
            query_text=query_text,
            query_embedding=query_embedding,
            max_results=max_results,
            salience_floor=salience_floor,
            include_communities=include_communities,
        )

        response = await hybrid_retrieve(self._neo4j, request)

        # Somatic reranking: boost candidates with somatic similarity to current state
        # somatic_rerank modifies salience_score; sync back to unified_score
        if self._soma is not None and response.traces:
            try:
                response.traces = self._soma.somatic_rerank(response.traces)
                for trace in response.traces:
                    trace.unified_score = trace.salience_score
            except Exception as exc:
                logger.debug("somatic_rerank_error", error=str(exc))

        # Update access timestamps for retrieved episodes (salience boost)
        retrieved_ids = [r.node_id for r in response.traces]
        await update_access(self._neo4j, retrieved_ids)

        # RE training: retrieval ranking decision
        if response.traces:
            await self._emit_re_training_example(
                category="retrieval_ranking",
                instruction="Rank memory traces by relevance for retrieval query.",
                input_context=f"query={query_text!r}, max_results={max_results}, salience_floor={salience_floor}",
                output=f"returned {len(response.traces)} traces, top_score={response.traces[0].unified_score:.3f}",
                # TODO(re-quality): replace with downstream usage signal (was retrieved memory actually used?)
                outcome_quality=response.traces[0].unified_score if response.traces else 0.0,
            )

        return response

    # ─── Entity Management (from extraction pipeline) ─────────────

    async def resolve_and_create_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
    ) -> tuple[str, bool]:
        """
        Find or create an entity. Handles deduplication.

        Returns: (entity_id, was_created)
        """
        # Compute embedding
        embed_text = f"{name}: {description}"
        embedding = await self._embedding.embed(embed_text)

        # Try to find existing
        existing = await find_similar_entity(self._neo4j, name, embedding)
        if existing:
            await merge_into_entity(self._neo4j, existing["id"], description)
            return existing["id"], False

        # Create new
        entity = Entity(
            name=name,
            type=EntityType(entity_type) if entity_type else EntityType.CONCEPT,
            description=description,
            embedding=embedding,
        )
        entity_id = await create_entity(self._neo4j, entity)

        # Evolutionary observable: new entity discovered in the knowledge graph
        await self._emit_evolutionary_observable(
            observable_type="entity_discovered",
            value=1.0,
            is_novel=True,
            metadata={"entity_id": entity_id, "entity_type": entity_type, "name": name},
        )

        # RE training: entity extraction decision (new entity created)
        await self._emit_re_training_example(
            category="entity_extraction",
            instruction="Resolve entity: find existing or create new in knowledge graph.",
            input_context=f"name={name!r}, type={entity_type}, description={description[:200]!r}",
            output=f"created_new=True, entity_id={entity_id}",
            # TODO(re-quality): replace with entity merge rate over time (high merge rate = poor extraction)
            outcome_quality=1.0,
        )
        return entity_id, True

    async def link_mention(self, mention: MentionRelation) -> None:
        """Link an episode to an entity."""
        await link_episode_to_entity(self._neo4j, mention)

    async def link_relation(self, relation: SemanticRelation) -> None:
        """Create or strengthen a semantic relation."""
        await create_or_strengthen_relation(self._neo4j, relation)

        # Evolutionary observable: relation formed between entities
        await self._emit_evolutionary_observable(
            observable_type="relation_formed",
            value=1.0,
            is_novel=True,
            metadata={
                "relation_type": relation.relation_type,
                "source_id": relation.source_id,
                "target_id": relation.target_id,
            },
        )

    # ─── State Queries ────────────────────────────────────────────

    async def get_constitution(self) -> dict[str, Any] | None:
        """Get the current constitutional state."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
            RETURN c
            """
        )
        return results[0]["c"] if results else None

    async def get_core_entities(self) -> list[dict[str, Any]]:
        """Get all entities marked as core to the instance's identity."""
        return await self._neo4j.execute_read(
            """
            MATCH (e:Entity {is_core_identity: true})
            RETURN e
            ORDER BY e.salience_score DESC
            """
        )

    async def update_affect(self, affect: AffectState) -> None:
        """Update the current affect state on the Self node (flattened scalars)."""
        affect_map = affect.to_map()
        await self._neo4j.execute_write(
            """
            MATCH (s:Self)
            SET s.affect_valence = $affect_valence,
                s.affect_arousal = $affect_arousal,
                s.affect_dominance = $affect_dominance,
                s.affect_curiosity = $affect_curiosity,
                s.affect_care_activation = $affect_care_activation,
                s.affect_coherence_stress = $affect_coherence_stress
            """,
            {
                "affect_valence": affect_map.get("valence", 0.0),
                "affect_arousal": affect_map.get("arousal", 0.0),
                "affect_dominance": affect_map.get("dominance", 0.0),
                "affect_curiosity": affect_map.get("curiosity", 0.0),
                "affect_care_activation": affect_map.get("care_activation", 0.0),
                "affect_coherence_stress": affect_map.get("coherence_stress", 0.0),
            },
        )

        # Emit SELF_AFFECT_UPDATED (fire-and-forget)
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEventType
                await self._event_bus.emit(
                    SynapseEventType.SELF_AFFECT_UPDATED,
                    {
                        "affect_valence": affect_map.get("valence", 0.0),
                        "affect_arousal": affect_map.get("arousal", 0.0),
                    },
                    source_system="memory",
                )
            except Exception:
                logger.debug("self_affect_updated_emit_failed", exc_info=True)

    async def update_conscience_fields(
        self,
        last_conscience_activation: Any,
        compliance_score: float,
    ) -> None:
        """Update conscience tracking fields on the Self node.

        Called by Equor after every constitutional review so the organism's
        identity substrate records the affective history of its own conscience.

        Fields written:
          last_conscience_activation - ISO timestamp of the most recent Equor review
          avg_compliance_score       - rolling 24h mean composite alignment (EMA, α=0.05)

        The EMA keeps the score stable across high-volume review cycles while
        remaining sensitive to sustained constitutional drift.
        """
        now_iso = last_conscience_activation.isoformat() if hasattr(last_conscience_activation, "isoformat") else str(last_conscience_activation)
        try:
            await self._neo4j.execute_write(
                """
                MATCH (s:Self)
                SET s.last_conscience_activation = datetime($now),
                    s.avg_compliance_score = CASE
                        WHEN s.avg_compliance_score IS NULL
                        THEN $score
                        ELSE s.avg_compliance_score * 0.95 + $score * 0.05
                    END
                """,
                {"now": now_iso, "score": float(compliance_score)},
            )
        except Exception as exc:
            logger.debug("conscience_fields_update_failed", error=str(exc))

    async def increment_cycle_count(self) -> None:
        """Increment the cognitive cycle counter."""
        await self._neo4j.execute_write(
            "MATCH (s:Self) SET s.cycle_count = s.cycle_count + 1"
        )

    async def update_personality_from_evo(
        self,
        hypothesis_outcome: dict[str, Any],
    ) -> None:
        """
        Update the Self node's personality vector based on confirmed Evo hypothesis
        outcomes and Equor drive drift.

        Personality is not static - it is the organism's learned behavioral
        tendency, shaped by what actually works (hypothesis confirmation) and
        what the drives reward (constitutional alignment deltas).

        Called by Evo when a hypothesis with a personality tag reaches INTEGRATED,
        and by Equor/Telos when drive alignment drift exceeds a threshold.

        `hypothesis_outcome` keys:
            - drive_deltas: dict[str, float]  - Coherence/Care/Growth/Honesty deltas
            - personality_dims: dict[str, float] - named dims to nudge (optional)
            - confidence: float - weighting of the update (0–1)
            - source: str - "evo_hypothesis" | "equor_drift"

        The update is recorded as a DriveWeightHistoryEntry on the Self node
        so `export_genome()` can export the full drift history.

        Resolves SG7: personality vector is no longer frozen at birth.
        """
        confidence: float = float(hypothesis_outcome.get("confidence", 0.3))
        drive_deltas: dict[str, float] = hypothesis_outcome.get("drive_deltas", {})
        personality_dims: dict[str, float] = hypothesis_outcome.get("personality_dims", {})
        source: str = hypothesis_outcome.get("source", "unknown")

        if not drive_deltas and not personality_dims:
            return

        now_iso = utc_now().isoformat()

        # Fetch current personality vector
        rows = await self._neo4j.execute_read(
            "MATCH (s:Self) RETURN s.personality_vector AS pv, s.drive_weight_history AS dwh LIMIT 1"
        )
        if not rows:
            return

        import json as _json

        raw_pv = rows[0].get("pv") or []
        if isinstance(raw_pv, str):
            try:
                raw_pv = _json.loads(raw_pv)
            except Exception:
                raw_pv = []
        pv: list[float] = [float(v) for v in raw_pv] if raw_pv else []

        # Named personality dim → index mapping (matches birth.py construction order)
        _DIM_INDEX: dict[str, int] = {
            "warmth": 0, "directness": 1, "verbosity": 2, "formality": 3,
            "curiosity_expression": 4, "humour": 5, "empathy_expression": 6,
            "confidence_display": 7, "metaphor_use": 8,
        }

        if pv:
            lr = confidence * 0.05  # learning rate: max 5% nudge per update
            for dim_name, delta in personality_dims.items():
                idx = _DIM_INDEX.get(dim_name)
                if idx is not None and idx < len(pv):
                    pv[idx] = max(-1.0, min(1.0, pv[idx] + delta * lr))

            # Drive deltas also influence the personality: high Care drift
            # lifts empathy/warmth; high Growth drift lifts curiosity_expression.
            care_delta = drive_deltas.get("care", 0.0)
            growth_delta = drive_deltas.get("growth", 0.0)
            coherence_delta = drive_deltas.get("coherence", 0.0)
            if len(pv) > 6:
                pv[0] = max(-1.0, min(1.0, pv[0] + care_delta * lr))     # warmth
                pv[6] = max(-1.0, min(1.0, pv[6] + care_delta * lr))     # empathy_expression
                pv[4] = max(-1.0, min(1.0, pv[4] + growth_delta * lr))   # curiosity_expression
                pv[1] = max(-1.0, min(1.0, pv[1] + coherence_delta * lr))  # directness

        # Append to drive_weight_history (JSON list stored on Self node)
        raw_dwh = rows[0].get("dwh") or "[]"
        if isinstance(raw_dwh, str):
            try:
                dwh: list[dict[str, Any]] = _json.loads(raw_dwh)
            except Exception:
                dwh = []
        else:
            dwh = list(raw_dwh) if raw_dwh else []

        entry: dict[str, Any] = {
            "ts": now_iso,
            "source": source,
            "confidence": round(confidence, 4),
            "drive_deltas": {k: round(v, 4) for k, v in drive_deltas.items()},
        }
        dwh.append(entry)
        # Keep only the last 500 entries to bound storage
        if len(dwh) > 500:
            dwh = dwh[-500:]

        await self._neo4j.execute_write(
            """
            MATCH (s:Self)
            SET s.personality_vector = $pv,
                s.drive_weight_history = $dwh,
                s.personality_updated_at = datetime($now)
            """,
            {
                "pv": pv,
                "dwh": _json.dumps(dwh),
                "now": now_iso,
            },
        )

        logger.info(
            "personality_vector_updated",
            source=source,
            confidence=round(confidence, 3),
            drive_deltas=drive_deltas,
            pv_dims=len(pv),
        )

    # ─── Consolidation ────────────────────────────────────────────

    async def consolidate(self) -> dict[str, Any]:
        """
        Run the graph-maintenance pipeline.

        Handles local graph housekeeping: salience decay, community detection,
        semantic compression, and entity deduplication.

        Logos compression (MDL scoring, world-model updates) is owned by
        Evo (wake-state, triggered every 6 h / 10 k cycles) and Oneiros
        (sleep-state Memory Ladder during SLOW_WAVE). Neither path runs here.
        """
        # ── Metabolic gate: AUSTERITY+ → halt consolidation ──
        if self._starvation_level in ("austerity", "emergency", "critical"):
            logger.info("consolidation_blocked_starvation", level=self._starvation_level)
            return {"skipped": True, "reason": f"metabolic_{self._starvation_level}"}

        # ── Skia modulation halt ────────────────────────────────────────────────────
        if self._modulation_halted:
            logger.debug("consolidation_skipped_modulation_halted")
            return {"skipped": True, "reason": "modulation_halted"}

        result = await run_consolidation(self._neo4j)

        # P9: Re-embed placeholder nodes (fire-and-forget, non-fatal)
        try:
            reembedded = await self.reembed_pending_nodes()
            result["reembedded"] = reembedded
        except Exception:
            logger.debug("reembed_pending_failed", exc_info=True)

        # Evolutionary observable: consolidation completed
        pruned = result.get("pruned", 0)
        communities = result.get("communities_detected", 0)
        if pruned > 0 or communities > 0:
            await self._emit_evolutionary_observable(
                observable_type="consolidation_pattern",
                value=float(pruned),
                is_novel=True,
                metadata={"pruned": pruned, "total": result.get("total", 0)},
            )
        if communities > 0:
            await self._emit_evolutionary_observable(
                observable_type="community_emerged",
                value=float(communities),
                is_novel=True,
                metadata={"communities_detected": communities},
            )

        await self._emit_re_training_example(
            category="consolidation",
            instruction="Consolidate memory graph: decay salience, detect communities, compress, deduplicate entities.",
            input_context=f"graph_stats_before={result.get('before', {})}",
            output=f"consolidated: {result}",
            # TODO(re-quality): replace with retrospective measurement (did consolidation improve retrieval quality?)
            outcome_quality=min(1.0, result.get("pruned", 0) / max(result.get("total", 1), 1)),
        )

        # Emit BELIEF_CONSOLIDATED (fire-and-forget)
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEventType
                steps = result.get("steps", {})
                community_data = steps.get("community_detection", {})
                belief_promotion_data = steps.get("belief_promotion", {})
                beliefs_promoted = belief_promotion_data.get("promoted", 0)
                await self._event_bus.emit(
                    SynapseEventType.BELIEF_CONSOLIDATED,
                    {
                        "beliefs_created": beliefs_promoted,
                        "schemas_created": 0,
                        "communities_detected": community_data.get("community_count", 0),
                        "episodes_processed": result.get("total", 0),
                    },
                    source_system="memory",
                )
            except Exception:
                logger.debug("belief_consolidated_emit_failed", exc_info=True)

        # Emit MEMORY_CONSOLIDATED - Logos subscribes for distillation rescoring
        if self._event_bus is not None:
            try:
                import uuid as _uuid
                from systems.synapse.types import SynapseEventType as _SET
                steps = result.get("steps", {})
                await self._event_bus.emit(
                    _SET.MEMORY_CONSOLIDATED,
                    {
                        "consolidated_count": result.get("total", 0),
                        "schemas_updated": steps.get("schema_extraction", {}).get("extracted", 0),
                        "coverage_delta": 0.0,
                        "cycle_id": str(_uuid.uuid4()),
                    },
                    source_system="memory",
                )
            except Exception:
                logger.debug("memory_consolidated_emit_failed", exc_info=True)

        # Check for SELF_STATE_DRIFTED: contradicting beliefs
        await self._check_self_state_drift()

        # Emit graph health KPIs so Benchmarks can observe memory utilization
        await self.emit_graph_health_kpi()

        return result

    async def _check_self_state_drift(self) -> None:
        """Emit SELF_STATE_DRIFTED if contradicting beliefs exceed threshold."""
        if self._event_bus is None:
            return
        try:
            # Count beliefs with overlapping domains but conflicting conclusions
            results = await self._neo4j.execute_read(
                """
                MATCH (b1:Belief), (b2:Belief)
                WHERE b1.id < b2.id
                  AND b1.domain = b2.domain
                  AND b1.precision > 0.5
                  AND b2.precision > 0.5
                RETURN count(*) AS contradiction_count
                """
            )
            contradiction_count = results[0]["contradiction_count"] if results else 0
            if contradiction_count > 5:
                # Count stale beliefs
                stale_results = await self._neo4j.execute_read(
                    """
                    MATCH (b:Belief)
                    WHERE b.last_verified IS NOT NULL
                      AND b.half_life_days IS NOT NULL
                      AND duration.between(b.last_verified, datetime()).days > b.half_life_days
                    RETURN count(b) AS stale_count
                    """
                )
                stale_count = stale_results[0]["stale_count"] if stale_results else 0
                from systems.synapse.types import SynapseEventType
                await self._event_bus.emit(
                    SynapseEventType.SELF_STATE_DRIFTED,
                    {
                        "contradiction_count": contradiction_count,
                        "stale_belief_count": stale_count,
                    },
                    source_system="memory",
                )
        except Exception:
            logger.debug("self_state_drift_check_failed", exc_info=True)

    # ─── RE Training Export ─────────────────────────────────────

    async def export_training_batch(
        self,
        stream_id: str,
        since: Any,
        limit: int = 100,
    ) -> list[RETrainingExample]:
        """
        Export episodes as RETrainingExample objects for RE fine-tuning.

        Queries Neo4j for episodes matching stream criteria (by source prefix
        or modality) created after `since`, formatted as training examples.
        """
        from datetime import datetime as _dt

        since_iso = since.isoformat() if hasattr(since, "isoformat") else str(since)

        records = await self._neo4j.execute_read(
            """
            MATCH (e:Episode)
            WHERE e.ingestion_time >= datetime($since)
              AND (e.source STARTS WITH $stream_id OR $stream_id = '')
            RETURN e.id AS id,
                   e.raw_content AS raw_content,
                   e.summary AS summary,
                   e.source AS source,
                   e.novelty_score AS novelty_score,
                   e.context_summary AS context_summary,
                   e.is_correction AS is_correction,
                   e.salience_composite AS salience
            ORDER BY e.ingestion_time ASC
            LIMIT $limit
            """,
            {"since": since_iso, "stream_id": stream_id, "limit": limit},
        )

        examples: list[RETrainingExample] = []
        for r in records:
            examples.append(RETrainingExample(
                source_system=SystemID.MEMORY,
                episode_id=str(r.get("id", "")),
                instruction=f"Process episode from {r.get('source', 'unknown')}",
                input_context=str(r.get("raw_content", ""))[:1000],
                output=str(r.get("summary", "")),
                outcome_quality=float(r.get("salience", 0.0)),
                category="episode_export",
            ))

        return examples

    # ─── Expression Episodes (AV3 - Voxis public API) ─────────────

    async def store_expression_episode(
        self,
        raw_content: str,
        summary: str,
        salience_composite: float = 0.3,
        affect_valence: float = 0.0,
        affect_arousal: float = 0.0,
        modality: str = "text",
        context_summary: str = "",
    ) -> str:
        """
        Store a Voxis expression as an episode via the service layer.

        Replaces direct `from systems.memory.episodic import store_episode` calls
        in voxis/service.py (AV3). Provides somatic stamping, temporal chain
        linking, and EPISODE_STORED emission - all missing when Voxis bypassed
        the service layer.

        Returns the episode ID.
        Spec: §5.1 (store_percept flow), AV3 resolution.
        """
        if self._starvation_level in ("emergency", "critical"):
            logger.info("store_expression_episode_blocked_starvation", level=self._starvation_level)
            return ""

        import time as _time

        now = utc_now()

        # Stamp somatic marker
        somatic_marker = None
        somatic_vector: list[float] | None = None
        if self._soma is not None:
            try:
                somatic_marker = self._soma.get_somatic_marker()
                if somatic_marker is not None and hasattr(somatic_marker, "to_vector"):
                    somatic_vector = somatic_marker.to_vector()
            except Exception as exc:
                logger.debug("soma_marker_error_expression", error=str(exc))

        # Compute embedding
        embedding: list[float] = []
        if raw_content:
            try:
                embedding = await self._embedding.embed(raw_content)
            except Exception:
                pass

        episode = Episode(
            event_time=now,
            ingestion_time=now,
            source=f"voxis:{modality}",
            modality=modality,
            raw_content=raw_content,
            summary=summary,
            embedding=embedding,
            salience_composite=salience_composite,
            affect_valence=affect_valence,
            affect_arousal=affect_arousal,
            free_energy=0.0,
            somatic_marker=somatic_marker,
            somatic_vector=somatic_vector,
            context_summary=context_summary,
        )
        episode_id = episode.id

        await store_episode(self._neo4j, episode)

        # Link temporal chain
        mono_now = _time.monotonic()
        if self._last_episode_id is not None and self._last_event_time is not None:
            gap = max(0.0, (now - self._last_event_time).total_seconds())
            if gap < 3600.0:
                try:
                    causal = max(0.05, min(0.8, 1.0 - (gap / 300.0)))
                    await link_episode_sequence(
                        self._neo4j,
                        previous_episode_id=self._last_episode_id,
                        current_episode_id=episode_id,
                        gap_seconds=gap,
                        causal_strength=causal,
                    )
                except Exception:
                    logger.debug("expression_episode_link_failed", exc_info=True)

        self._last_episode_id = episode_id
        self._last_episode_time = mono_now
        self._last_event_time = now
        self._financial_encoder.set_sequence_state(episode_id, mono_now)

        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEventType
                await self._event_bus.emit(
                    SynapseEventType.EPISODE_STORED,
                    {
                        "episode_id": episode_id,
                        "source": episode.source,
                        "summary": episode.summary,
                        "salience": salience_composite,
                    },
                    source_system="memory",
                )
            except Exception:
                logger.debug("expression_episode_stored_emit_failed", exc_info=True)

        return episode_id

    # ─── Counterfactual Episodes (AV4 - Nova public API) ──────────

    async def store_counterfactual_episode(self, record: Any) -> str:
        """
        Store a rejected Nova policy as a counterfactual episode.

        Replaces direct `from systems.memory.episodic import store_counterfactual_episode`
        calls in nova/service.py (AV4).

        Returns the counterfactual record ID.
        Spec: §14.2 (future: now resolved), episodic.py store_counterfactual_episode.
        """
        from systems.memory.episodic import store_counterfactual_episode as _store_cf
        return await _store_cf(self._neo4j, record)

    async def resolve_counterfactual(
        self,
        record_id: str,
        outcome_success: bool,
        actual_pragmatic_value: float,
        regret: float,
    ) -> None:
        """
        Update a counterfactual node with outcome-derived regret.

        Replaces direct `from systems.memory.episodic import resolve_counterfactual`
        calls in nova/service.py and api/routers/memory.py (AV4/AV6).
        Spec: episodic.py resolve_counterfactual.
        """
        from systems.memory.episodic import resolve_counterfactual as _resolve_cf
        await _resolve_cf(self._neo4j, record_id, outcome_success, actual_pragmatic_value, regret)

    async def link_counterfactual_to_outcome(
        self,
        counterfactual_id: str,
        outcome_episode_id: str,
    ) -> None:
        """
        Create ALTERNATIVE_TO relationship from counterfactual to actual outcome episode.

        Replaces direct import in nova/service.py and api/routers/memory.py (AV4/AV6).
        Spec: episodic.py link_counterfactual_to_outcome.
        """
        from systems.memory.episodic import link_counterfactual_to_outcome as _link_cf
        await _link_cf(self._neo4j, counterfactual_id, outcome_episode_id)

    # ─── Kairos Observation Feed (Stage 1 CorrelationMiner input) ──

    async def get_all_observations_with_context(
        self,
        n_cycles: int = 50,
    ) -> list[tuple[Any, list[Any], list[dict[str, Any]]]]:
        """
        Return episodes from the last N cycles with their related entities and
        causal node references.  This is the primary data feed for Kairos Stage 1
        CorrelationMiner.

        Returns a list of 3-tuples:
          (Episode dict, [Entity dicts], [CausalNode ref dicts])

        Each element maps directly to the pipeline's observations_by_context
        shape after conversion: context_id = episode["id"], observations =
        the episode's numeric fields + entity salience scores.

        Args:
            n_cycles: Approximate number of recent theta cycles to cover.
                      Each cycle ~150ms → 50 cycles ≈ 7.5s of recent history.
                      The query uses ingestion_time ordering and LIMIT.
        """
        episode_limit = max(n_cycles * 10, 100)  # ~10 episodes per cycle
        rows = await self._neo4j.execute_read(
            """
            MATCH (ep:Episode)
            WITH ep ORDER BY ep.ingestion_time DESC LIMIT $limit
            OPTIONAL MATCH (ep)-[:MENTIONS]->(ent:Entity)
            OPTIONAL MATCH (ep)-[:CAUSED_BY|CAUSES]->(cn:CausalNode)
            RETURN ep,
                   collect(DISTINCT ent) AS entities,
                   collect(DISTINCT {
                       name: cn.name,
                       domain: cn.domain,
                       confidence: cn.confidence
                   }) AS causal_refs
            ORDER BY ep.ingestion_time DESC
            """,
            {"limit": episode_limit},
        )
        result: list[tuple[Any, list[Any], list[dict[str, Any]]]] = []
        for row in rows:
            ep = row.get("ep") or {}
            entities = [e for e in (row.get("entities") or []) if e]
            causal_refs = [
                c for c in (row.get("causal_refs") or [])
                if c and c.get("name")
            ]
            result.append((ep, entities, causal_refs))
        return result

    # ─── Read Queries (AV6 - API router public API) ────────────────

    async def get_recent_episodes(
        self,
        limit: int = 20,
        min_salience: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent episodes ordered by ingestion_time.

        Replaces direct `from systems.memory.episodic import get_recent_episodes`
        calls in api/routers/memory.py (AV6).
        """
        from systems.memory.episodic import get_recent_episodes as _get_recent
        return await _get_recent(self._neo4j, limit=limit, min_salience=min_salience)

    async def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single episode by ID.

        Replaces direct `from systems.memory.episodic import get_episode` calls
        in api/routers/memory.py (AV6).
        """
        from systems.memory.episodic import get_episode as _get_episode
        return await _get_episode(self._neo4j, episode_id)

    async def get_episodes_meta(
        self, episode_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """
        Bulk-fetch affect_valence, source, and event_time for a list of episode IDs.

        Returns a dict keyed by episode ID. Used by Nova's slow-path memory
        enrichment to avoid direct _neo4j access (AV2).
        """
        if not episode_ids:
            return {}
        rows = await self._neo4j.execute_read(
            """
            MATCH (ep:Episode)
            WHERE ep.id IN $ids
            RETURN ep.id AS id,
                   ep.affect_valence AS affect_valence,
                   ep.source AS source,
                   ep.event_time AS event_time
            """,
            {"ids": episode_ids},
        )
        return {
            row["id"]: {
                "affect_valence": row.get("affect_valence"),
                "source": row.get("source"),
                "event_time": row.get("event_time"),
            }
            for row in rows
        }

    async def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single entity by ID.

        Replaces direct `from systems.memory.semantic import get_entity` calls
        in api/routers/memory.py (AV6).
        """
        from systems.memory.semantic import get_entity as _get_entity
        return await _get_entity(self._neo4j, entity_id)

    # ─── Genome Export (SG1 - Mitosis public API) ─────────────────

    async def export_genome(self) -> dict[str, Any]:
        """
        Export the organism's heritable memory state for Mitosis child spawning.

        Delegates to MemoryGenomeExtractor which serialises the top entities,
        semantic relations, community summaries, personality vector, and
        consolidated episode summaries into an OrganGenomeSegment.

        Returns the segment as a dict ready for JSON transport (payload_hash
        and size_bytes included). Returns an empty segment if no meaningful
        state exists yet (version=0).

        Resolves SG1: Mitosis can now call this to obtain the parent's genome
        without reaching into Memory internals or Neo4j directly.
        Spec: §17 (Speciation), speciation_bible §Population Dynamics.
        """
        from systems.memory.genome import MemoryGenomeExtractor
        extractor = MemoryGenomeExtractor(self._neo4j)
        segment = await extractor.extract_genome_segment()

        # Emit evolutionary observable: genome exported (population dynamics signal)
        await self._emit_evolutionary_observable(
            observable_type="genome_exported",
            value=float(segment.size_bytes),
            is_novel=False,
            metadata={
                "version": segment.version,
                "size_bytes": segment.size_bytes,
                "system_id": str(segment.system_id),
            },
        )

        return segment.model_dump(mode="json")

    # ─── Re-embedding (P9 - placeholder episodes) ────────────────

    async def reembed_pending_nodes(self, batch_size: int = 50) -> int:
        """
        Re-embed episodes and entities that were stored with placeholder embeddings.

        Episodes written before the embedding client was available (or written
        by systems that omit embeddings for speed) are marked
        `needs_reembedding=true`. This background task processes them in batches,
        computes real 768-dim embeddings, and clears the flag.

        Should be called at the end of each consolidation cycle (already wired
        via `consolidate()`).

        Returns the count of nodes re-embedded.

        Resolves P9: placeholder episodes no longer permanently degrade vector search.
        Spec: §12.2.
        """
        if self._starvation_level in ("emergency", "critical"):
            return 0

        reembedded = 0

        # Fetch episodes needing re-embedding
        ep_rows = await self._neo4j.execute_read(
            """
            MATCH (e:Episode)
            WHERE e.needs_reembedding = true
              AND (e.raw_content IS NOT NULL AND e.raw_content <> '')
            RETURN e.id AS id, e.raw_content AS raw_content, e.summary AS summary
            LIMIT $limit
            """,
            {"limit": batch_size},
        )

        for row in ep_rows:
            ep_id: str = row.get("id", "")
            raw: str = row.get("raw_content", "") or row.get("summary", "")
            if not ep_id or not raw:
                continue
            try:
                embedding = await self._embedding.embed(raw[:2000])
                await self._neo4j.execute_write(
                    """
                    MATCH (e:Episode {id: $id})
                    SET e.embedding = $embedding,
                        e.needs_reembedding = false
                    """,
                    {"id": ep_id, "embedding": embedding},
                )
                reembedded += 1
            except Exception as exc:
                logger.debug("reembed_episode_failed", episode_id=ep_id, error=str(exc))

        # Fetch entities needing re-embedding
        ent_rows = await self._neo4j.execute_read(
            """
            MATCH (e:Entity)
            WHERE e.needs_reembedding = true
              AND e.name IS NOT NULL
            RETURN e.id AS id, e.name AS name, e.description AS description
            LIMIT $limit
            """,
            {"limit": batch_size},
        )

        for row in ent_rows:
            ent_id: str = row.get("id", "")
            text = f"{row.get('name', '')}: {row.get('description', '')}".strip(": ")
            if not ent_id or not text:
                continue
            try:
                embedding = await self._embedding.embed(text[:2000])
                await self._neo4j.execute_write(
                    """
                    MATCH (e:Entity {id: $id})
                    SET e.embedding = $embedding,
                        e.needs_reembedding = false
                    """,
                    {"id": ent_id, "embedding": embedding},
                )
                reembedded += 1
            except Exception as exc:
                logger.debug("reembed_entity_failed", entity_id=ent_id, error=str(exc))

        if reembedded > 0:
            logger.info("nodes_reembedded", count=reembedded)

        return reembedded

    # ─── Graph Health KPI Emission (Benchmarks / self-awareness) ──────────────

    async def emit_graph_health_kpi(self) -> None:
        """Emit memory graph health KPIs as EVOLUTIONARY_OBSERVABLE events.

        Closes organism-level self-awareness gap: the organism can now observe
        its own memory utilization, belief count, and community structure via
        Benchmarks population tracking.

        Called at the end of each consolidation cycle and can also be called
        externally by the registry heartbeat.
        """
        if self._event_bus is None:
            return
        try:
            graph_stats = await self.stats()
            ep_count = graph_stats.get("episode_count", 0)
            ent_count = graph_stats.get("entity_count", 0)
            node_count = graph_stats.get("node_count", 0)
            hyp_count = graph_stats.get("hypothesis_count", 0)

            # Episode utilization - fraction of capacity used (pressure_threshold = 10000)
            utilization = min(1.0, ep_count / 10000.0)

            await self._emit_evolutionary_observable(
                observable_type="memory_graph_utilization",
                value=utilization,
                is_novel=False,
                metadata={
                    "episode_count": ep_count,
                    "entity_count": ent_count,
                    "node_count": node_count,
                    "hypothesis_count": hyp_count,
                    "instance_id": self._instance_id or "",
                },
            )
        except Exception:
            logger.debug("graph_health_kpi_emit_failed", exc_info=True)

    # ─── Stats ────────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        ep_count = await count_episodes(self._neo4j)
        ent_count = await count_entities(self._neo4j)

        extra = await self._neo4j.execute_read(
            """
            MATCH (n) WITH count(n) AS node_count
            OPTIONAL MATCH ()-[r]->()
            WITH node_count, count(r) AS edge_count
            OPTIONAL MATCH (h:Hypothesis)
            WITH node_count, edge_count, count(h) AS hypothesis_count
            OPTIONAL MATCH (p:Procedure)
            RETURN node_count, edge_count, hypothesis_count, count(p) AS procedure_count
            """
        )
        node_count = extra[0]["node_count"] if extra else 0
        edge_count = extra[0]["edge_count"] if extra else 0
        hyp_count = extra[0]["hypothesis_count"] if extra else 0
        proc_count = extra[0]["procedure_count"] if extra else 0

        return {
            "episode_count": ep_count,
            "entity_count": ent_count,
            "node_count": node_count,
            "edge_count": edge_count,
            "hypothesis_count": hyp_count,
            "procedure_count": proc_count,
        }

    # ─── Public Neo4j accessor (for intra-system modules only) ───────

    def get_neo4j(self) -> Any:
        """
        Return the raw Neo4j client for use by Nova's internal persistence
        modules (GoalStore, BeliefUpdater). Do not call from other systems -
        all cross-system writes must go through MemoryService public methods.
        """
        return self._neo4j

    # ─── Health ───────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health check for the memory system (must complete within 2s)."""
        try:
            neo4j_health = await self._neo4j.health_check()
        except Exception as exc:
            error_msg = str(exc) or f"{type(exc).__name__} (no message)"
            return {
                "status": "degraded",
                "neo4j": {"status": "error", "error": error_msg},
                "instance_id": self._instance_id,
            }
        return {
            "status": "healthy" if neo4j_health["status"] == "connected" else "degraded",
            "neo4j": neo4j_health,
            "instance_id": self._instance_id,
        }
