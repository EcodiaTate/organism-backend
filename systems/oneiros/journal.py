"""
EcodiaOS — Oneiros: Dream Journal & Insight Tracker

Persistent storage of the organism's dream life. Every dream, every
insight, every sleep cycle is recorded in the knowledge graph and
mirrored in an in-memory cache for fast access during sleep phases.

The DreamJournal is the organism's dream diary — queryable, observable,
and introspectable. The DreamInsightTracker follows each insight from
its birth in REM sleep through wake-state validation to permanent
integration into semantic memory.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import structlog

from primitives.common import utc_now
from systems.oneiros.types import (
    Dream,
    DreamCoherence,
    DreamInsight,
    DreamType,
    InsightStatus,
    SleepCycle,
    SleepQuality,
)

logger = structlog.get_logger()

_MAX_DREAM_BUFFER = 200


# ─── Neo4j Schema ────────────────────────────────────────────────

_CONSTRAINTS = [
    "CREATE CONSTRAINT dream_id IF NOT EXISTS FOR (d:Dream) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT dream_insight_id IF NOT EXISTS FOR (i:DreamInsight) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT sleep_cycle_id IF NOT EXISTS FOR (s:SleepCycle) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT threat_model_id IF NOT EXISTS "
    "FOR (t:ThreatModelEpisode) REQUIRE t.id IS UNIQUE",
]

_INDEXES = [
    "CREATE INDEX dream_timestamp IF NOT EXISTS FOR (d:Dream) ON (d.timestamp)",
    "CREATE INDEX dream_type IF NOT EXISTS FOR (d:Dream) ON (d.type)",
    "CREATE INDEX dream_coherence IF NOT EXISTS FOR (d:Dream) ON (d.coherence_score)",
    "CREATE INDEX dream_cycle IF NOT EXISTS FOR (d:Dream) ON (d.sleep_cycle_id)",
    "CREATE INDEX insight_status IF NOT EXISTS FOR (i:DreamInsight) ON (i.status)",
    "CREATE INDEX insight_created IF NOT EXISTS FOR (i:DreamInsight) ON (i.created_at)",
    "CREATE INDEX insight_cycle IF NOT EXISTS FOR (i:DreamInsight) ON (i.sleep_cycle_id)",
    "CREATE INDEX sleep_cycle_started IF NOT EXISTS FOR (s:SleepCycle) ON (s.started_at)",
    "CREATE INDEX threat_model_timestamp IF NOT EXISTS FOR (t:ThreatModelEpisode) ON (t.timestamp)",
    "CREATE INDEX threat_model_cycle IF NOT EXISTS "
    "FOR (t:ThreatModelEpisode) ON (t.sleep_cycle_id)",
]


# ─── DreamJournal ────────────────────────────────────────────────


class DreamJournal:
    """
    Persistent storage of all dreams and insights.

    Writes to Neo4j for durable graph storage, and keeps an in-memory
    ring buffer of recent dreams plus a cache of non-integrated insights
    for fast access during sleep phases without round-tripping the DB.
    """

    def __init__(self, neo4j: Any = None) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(system="oneiros", component="journal")

        # In-memory buffers — fallback when Neo4j is unavailable
        self._dream_buffer: deque[Dream] = deque(maxlen=_MAX_DREAM_BUFFER)
        self._insight_cache: dict[str, DreamInsight] = {}  # non-integrated insights
        self._all_insights: dict[str, DreamInsight] = {}   # id -> insight (all statuses)

    # ── Initialisation ────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Ensure Neo4j schema: uniqueness constraints and performance indexes
        for Dream, DreamInsight, and SleepCycle nodes. Idempotent.
        """
        if self._neo4j is None:
            self._log.warning("journal_initialize_skipped", reason="no neo4j client")
            return

        self._log.info("journal_schema_ensuring")

        for statement in _CONSTRAINTS + _INDEXES:
            statement = statement.strip()
            if not statement:
                continue
            try:
                await self._neo4j.execute_write(statement)
            except Exception as exc:
                error_msg = str(exc).lower()
                if "already exists" in error_msg or "equivalent" in error_msg:
                    continue
                self._log.warning(
                    "journal_schema_statement_warning",
                    statement=statement[:80],
                    error=str(exc),
                )

        self._log.info("journal_schema_ensured")

    # ── Dream Recording ───────────────────────────────────────────

    async def record_dream(self, dream: Dream) -> None:
        """
        Persist a dream to Neo4j and the in-memory buffer.

        Creates the Dream node with all scalar properties, then links
        it to seed and activated Episode nodes via SEEDED_BY and ACTIVATED
        relationships.
        """
        # Always buffer in memory regardless of Neo4j availability
        self._dream_buffer.append(dream)

        if self._neo4j is None:
            self._log.debug("dream_recorded_memory_only", dream_id=dream.id)
            return

        try:
            # Store the dream node
            await self._neo4j.execute_write(
                """
                MERGE (d:Dream {id: $id})
                SET d.type = $type,
                    d.coherence_score = $coherence,
                    d.coherence_class = $coherence_class,
                    d.affect_valence = $valence,
                    d.affect_arousal = $arousal,
                    d.bridge_narrative = $narrative,
                    d.themes = $themes,
                    d.summary = $summary,
                    d.timestamp = $ts,
                    d.sleep_cycle_id = $cycle_id,
                    d.context_json = $context_json
                """,
                {
                    "id": dream.id,
                    "type": dream.dream_type.value,
                    "coherence": dream.coherence_score,
                    "coherence_class": dream.coherence_class.value,
                    "valence": dream.affect_valence,
                    "arousal": dream.affect_arousal,
                    "narrative": dream.bridge_narrative,
                    "themes": dream.themes,
                    "summary": dream.summary,
                    "ts": dream.timestamp.isoformat(),
                    "cycle_id": dream.sleep_cycle_id,
                    "context_json": json.dumps(dream.context) if dream.context else "{}",
                },
            )

            # Link to seed episodes
            for episode_id in dream.seed_episode_ids:
                try:
                    await self._neo4j.execute_write(
                        """
                        MATCH (d:Dream {id: $dream_id}), (e:Episode {id: $episode_id})
                        MERGE (d)-[:SEEDED_BY]->(e)
                        """,
                        {"dream_id": dream.id, "episode_id": episode_id},
                    )
                except Exception as exc:
                    self._log.debug(
                        "dream_seed_link_failed",
                        dream_id=dream.id,
                        episode_id=episode_id,
                        error=str(exc),
                    )

            # Link to activated episodes
            for episode_id in dream.activated_episode_ids:
                try:
                    await self._neo4j.execute_write(
                        """
                        MATCH (d:Dream {id: $dream_id}), (e:Episode {id: $episode_id})
                        MERGE (d)-[:ACTIVATED]->(e)
                        """,
                        {"dream_id": dream.id, "episode_id": episode_id},
                    )
                except Exception as exc:
                    self._log.debug(
                        "dream_activated_link_failed",
                        dream_id=dream.id,
                        episode_id=episode_id,
                        error=str(exc),
                    )

            self._log.info(
                "dream_recorded",
                dream_id=dream.id,
                type=dream.dream_type.value,
                coherence=dream.coherence_score,
                themes=dream.themes,
            )

        except Exception as exc:
            self._log.warning(
                "dream_record_neo4j_failed",
                dream_id=dream.id,
                error=str(exc),
            )

    # ── Insight Recording ─────────────────────────────────────────

    async def record_insight(self, insight: DreamInsight) -> None:
        """
        Persist a dream insight to Neo4j and the in-memory cache.

        Creates the DreamInsight node and links it to its source Dream
        via a PRODUCED relationship.
        """
        # Always cache in memory
        self._all_insights[insight.id] = insight
        if insight.status != InsightStatus.INTEGRATED:
            self._insight_cache[insight.id] = insight

        if self._neo4j is None:
            self._log.debug("insight_recorded_memory_only", insight_id=insight.id)
            return

        try:
            await self._neo4j.execute_write(
                """
                MERGE (i:DreamInsight {id: $id})
                SET i.insight_text = $text,
                    i.coherence_score = $coherence,
                    i.domain = $domain,
                    i.status = $status,
                    i.created_at = $created_at,
                    i.sleep_cycle_id = $cycle_id,
                    i.seed_summary = $seed_summary,
                    i.activated_summary = $activated_summary,
                    i.bridge_narrative = $bridge_narrative,
                    i.wake_applications = $wake_applications,
                    i.validation_context = $validation_context
                """,
                {
                    "id": insight.id,
                    "text": insight.insight_text,
                    "coherence": insight.coherence_score,
                    "domain": insight.domain,
                    "status": insight.status.value,
                    "created_at": insight.created_at.isoformat(),
                    "cycle_id": insight.sleep_cycle_id,
                    "seed_summary": insight.seed_summary,
                    "activated_summary": insight.activated_summary,
                    "bridge_narrative": insight.bridge_narrative,
                    "wake_applications": insight.wake_applications,
                    "validation_context": insight.validation_context,
                },
            )

            # Link to source dream
            try:
                await self._neo4j.execute_write(
                    """
                    MATCH (i:DreamInsight {id: $id}), (d:Dream {id: $dream_id})
                    MERGE (d)-[:PRODUCED]->(i)
                    """,
                    {"id": insight.id, "dream_id": insight.dream_id},
                )
            except Exception as exc:
                self._log.debug(
                    "insight_dream_link_failed",
                    insight_id=insight.id,
                    dream_id=insight.dream_id,
                    error=str(exc),
                )

            self._log.info(
                "insight_recorded",
                insight_id=insight.id,
                coherence=insight.coherence_score,
                domain=insight.domain,
                status=insight.status.value,
            )

        except Exception as exc:
            self._log.warning(
                "insight_record_neo4j_failed",
                insight_id=insight.id,
                error=str(exc),
            )

    # ── Sleep Cycle Recording ─────────────────────────────────────

    async def record_sleep_cycle(self, cycle: SleepCycle) -> None:
        """Store a new sleep cycle record in Neo4j."""
        if self._neo4j is None:
            self._log.debug("sleep_cycle_recorded_memory_only", cycle_id=cycle.id)
            return

        try:
            await self._neo4j.execute_write(
                """
                MERGE (s:SleepCycle {id: $id})
                SET s.started_at = $started_at,
                    s.completed_at = $completed_at,
                    s.quality = $quality,
                    s.interrupted = $interrupted,
                    s.interrupt_reason = $interrupt_reason,
                    s.episodes_replayed = $episodes_replayed,
                    s.semantic_nodes_created = $semantic_nodes_created,
                    s.traces_pruned = $traces_pruned,
                    s.salience_reduction_mean = $salience_reduction_mean,
                    s.beliefs_compressed = $beliefs_compressed,
                    s.hypotheses_pruned = $hypotheses_pruned,
                    s.hypotheses_promoted = $hypotheses_promoted,
                    s.dreams_generated = $dreams_generated,
                    s.insights_discovered = $insights_discovered,
                    s.affect_traces_processed = $affect_traces_processed,
                    s.affect_reduction_mean = $affect_reduction_mean,
                    s.threats_simulated = $threats_simulated,
                    s.ethical_cases_digested = $ethical_cases_digested,
                    s.lucid_explorations = $lucid_explorations,
                    s.meta_observations = $meta_observations,
                    s.pressure_before = $pressure_before,
                    s.pressure_after = $pressure_after
                """,
                {
                    "id": cycle.id,
                    "started_at": cycle.started_at.isoformat(),
                    "completed_at": cycle.completed_at.isoformat() if cycle.completed_at else None,
                    "quality": cycle.quality.value,
                    "interrupted": cycle.interrupted,
                    "interrupt_reason": cycle.interrupt_reason,
                    "episodes_replayed": cycle.episodes_replayed,
                    "semantic_nodes_created": cycle.semantic_nodes_created,
                    "traces_pruned": cycle.traces_pruned,
                    "salience_reduction_mean": cycle.salience_reduction_mean,
                    "beliefs_compressed": cycle.beliefs_compressed,
                    "hypotheses_pruned": cycle.hypotheses_pruned,
                    "hypotheses_promoted": cycle.hypotheses_promoted,
                    "dreams_generated": cycle.dreams_generated,
                    "insights_discovered": cycle.insights_discovered,
                    "affect_traces_processed": cycle.affect_traces_processed,
                    "affect_reduction_mean": cycle.affect_reduction_mean,
                    "threats_simulated": cycle.threats_simulated,
                    "ethical_cases_digested": cycle.ethical_cases_digested,
                    "lucid_explorations": cycle.lucid_explorations,
                    "meta_observations": cycle.meta_observations,
                    "pressure_before": cycle.pressure_before,
                    "pressure_after": cycle.pressure_after,
                },
            )

            self._log.info("sleep_cycle_recorded", cycle_id=cycle.id, quality=cycle.quality.value)

        except Exception as exc:
            self._log.warning(
                "sleep_cycle_record_failed",
                cycle_id=cycle.id,
                error=str(exc),
            )

    async def update_sleep_cycle(self, cycle: SleepCycle) -> None:
        """Update an existing sleep cycle (e.g. when completed or interrupted)."""
        if self._neo4j is None:
            self._log.debug("sleep_cycle_updated_memory_only", cycle_id=cycle.id)
            return

        try:
            await self._neo4j.execute_write(
                """
                MATCH (s:SleepCycle {id: $id})
                SET s.completed_at = $completed_at,
                    s.quality = $quality,
                    s.interrupted = $interrupted,
                    s.interrupt_reason = $interrupt_reason,
                    s.episodes_replayed = $episodes_replayed,
                    s.semantic_nodes_created = $semantic_nodes_created,
                    s.traces_pruned = $traces_pruned,
                    s.salience_reduction_mean = $salience_reduction_mean,
                    s.beliefs_compressed = $beliefs_compressed,
                    s.hypotheses_pruned = $hypotheses_pruned,
                    s.hypotheses_promoted = $hypotheses_promoted,
                    s.dreams_generated = $dreams_generated,
                    s.insights_discovered = $insights_discovered,
                    s.affect_traces_processed = $affect_traces_processed,
                    s.affect_reduction_mean = $affect_reduction_mean,
                    s.threats_simulated = $threats_simulated,
                    s.ethical_cases_digested = $ethical_cases_digested,
                    s.lucid_explorations = $lucid_explorations,
                    s.meta_observations = $meta_observations,
                    s.pressure_before = $pressure_before,
                    s.pressure_after = $pressure_after
                """,
                {
                    "id": cycle.id,
                    "completed_at": cycle.completed_at.isoformat() if cycle.completed_at else None,
                    "quality": cycle.quality.value,
                    "interrupted": cycle.interrupted,
                    "interrupt_reason": cycle.interrupt_reason,
                    "episodes_replayed": cycle.episodes_replayed,
                    "semantic_nodes_created": cycle.semantic_nodes_created,
                    "traces_pruned": cycle.traces_pruned,
                    "salience_reduction_mean": cycle.salience_reduction_mean,
                    "beliefs_compressed": cycle.beliefs_compressed,
                    "hypotheses_pruned": cycle.hypotheses_pruned,
                    "hypotheses_promoted": cycle.hypotheses_promoted,
                    "dreams_generated": cycle.dreams_generated,
                    "insights_discovered": cycle.insights_discovered,
                    "affect_traces_processed": cycle.affect_traces_processed,
                    "affect_reduction_mean": cycle.affect_reduction_mean,
                    "threats_simulated": cycle.threats_simulated,
                    "ethical_cases_digested": cycle.ethical_cases_digested,
                    "lucid_explorations": cycle.lucid_explorations,
                    "meta_observations": cycle.meta_observations,
                    "pressure_before": cycle.pressure_before,
                    "pressure_after": cycle.pressure_after,
                },
            )

            self._log.info("sleep_cycle_updated", cycle_id=cycle.id, quality=cycle.quality.value)

        except Exception as exc:
            self._log.warning(
                "sleep_cycle_update_failed",
                cycle_id=cycle.id,
                error=str(exc),
            )

    # ── Queries ───────────────────────────────────────────────────

    async def get_recent_dreams(self, limit: int = 50) -> list[Dream]:
        """
        Retrieve recent dreams ordered by timestamp descending.

        Falls back to the in-memory buffer when Neo4j is unavailable.
        """
        if self._neo4j is None:
            buffer_list = list(self._dream_buffer)
            buffer_list.reverse()
            return buffer_list[:limit]

        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (d:Dream)
                RETURN d
                ORDER BY d.timestamp DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )

            dreams: list[Dream] = []
            for record in records:
                node = record.get("d", {})
                dream = _dream_from_node(node)
                if dream is not None:
                    dreams.append(dream)
            return dreams

        except Exception as exc:
            self._log.warning("get_recent_dreams_failed", error=str(exc))
            # Fall back to memory
            buffer_list = list(self._dream_buffer)
            buffer_list.reverse()
            return buffer_list[:limit]

    async def get_recent_insights(
        self,
        status: InsightStatus | None = None,
        limit: int = 50,
    ) -> list[DreamInsight]:
        """
        Retrieve recent insights, optionally filtered by status.

        Falls back to the in-memory cache when Neo4j is unavailable.
        """
        if self._neo4j is None:
            return self._insights_from_cache(status=status, limit=limit)

        try:
            if status is not None:
                records = await self._neo4j.execute_read(
                    """
                    MATCH (i:DreamInsight)
                    WHERE i.status = $status
                    RETURN i
                    ORDER BY i.created_at DESC
                    LIMIT $limit
                    """,
                    {"status": status.value, "limit": limit},
                )
            else:
                records = await self._neo4j.execute_read(
                    """
                    MATCH (i:DreamInsight)
                    RETURN i
                    ORDER BY i.created_at DESC
                    LIMIT $limit
                    """,
                    {"limit": limit},
                )

            insights: list[DreamInsight] = []
            for record in records:
                node = record.get("i", {})
                insight = _insight_from_node(node)
                if insight is not None:
                    insights.append(insight)
            return insights

        except Exception as exc:
            self._log.warning("get_recent_insights_failed", error=str(exc))
            return self._insights_from_cache(status=status, limit=limit)

    async def get_pending_insights(self) -> list[DreamInsight]:
        """
        Retrieve all insights with PENDING status.

        These are queued for broadcast on the next wake cycle so the
        organism can evaluate them against waking reality.
        """
        return await self.get_recent_insights(status=InsightStatus.PENDING, limit=200)

    async def get_sleep_cycles(self, limit: int = 20) -> list[SleepCycle]:
        """Retrieve recent sleep cycles ordered by start time descending."""
        if self._neo4j is None:
            self._log.debug("get_sleep_cycles_no_neo4j")
            return []

        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (s:SleepCycle)
                RETURN s
                ORDER BY s.started_at DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )

            cycles: list[SleepCycle] = []
            for record in records:
                node = record.get("s", {})
                cycle = _cycle_from_node(node)
                if cycle is not None:
                    cycles.append(cycle)
            return cycles

        except Exception as exc:
            self._log.warning("get_sleep_cycles_failed", error=str(exc))
            return []

    async def get_recurring_themes(self, min_count: int = 3) -> list[dict[str, Any]]:
        """
        Aggregate themes across all dreams and return those appearing
        at least min_count times.

        This is how the organism detects what it keeps dreaming about.
        """
        if self._neo4j is None:
            return self._themes_from_buffer(min_count=min_count)

        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (d:Dream)
                WHERE d.themes IS NOT NULL
                UNWIND d.themes AS theme
                WITH theme, count(*) AS count
                WHERE count >= $min_count
                RETURN theme, count
                ORDER BY count DESC
                """,
                {"min_count": min_count},
            )

            return [
                {"theme": r.get("theme", ""), "count": r.get("count", 0)}
                for r in records
            ]

        except Exception as exc:
            self._log.warning("get_recurring_themes_failed", error=str(exc))
            return self._themes_from_buffer(min_count=min_count)

    async def stats(self) -> dict[str, Any]:
        """
        Aggregate statistics: total dreams, insights by status, cycles, etc.

        Used for health checks and the Oneiros health snapshot.
        """
        if self._neo4j is None:
            return self._stats_from_memory()

        try:
            records = await self._neo4j.execute_read(
                """
                OPTIONAL MATCH (d:Dream)
                WITH count(d) AS total_dreams
                OPTIONAL MATCH (i:DreamInsight)
                WITH total_dreams, count(i) AS total_insights
                OPTIONAL MATCH (ip:DreamInsight {status: 'pending'})
                WITH total_dreams, total_insights, count(ip) AS pending_insights
                OPTIONAL MATCH (iv:DreamInsight {status: 'validated'})
                WITH total_dreams, total_insights, pending_insights, count(iv) AS validated_insights
                OPTIONAL MATCH (ii:DreamInsight {status: 'integrated'})
                WITH total_dreams, total_insights, pending_insights,
                     validated_insights, count(ii) AS integrated_insights
                OPTIONAL MATCH (ix:DreamInsight {status: 'invalidated'})
                WITH total_dreams, total_insights, pending_insights,
                     validated_insights, integrated_insights,
                     count(ix) AS invalidated_insights
                OPTIONAL MATCH (s:SleepCycle)
                RETURN total_dreams, total_insights, pending_insights, validated_insights,
                       integrated_insights, invalidated_insights, count(s) AS total_cycles
                """,
            )

            if records:
                r = records[0]
                return {
                    "total_dreams": r.get("total_dreams", 0),
                    "total_insights": r.get("total_insights", 0),
                    "pending_insights": r.get("pending_insights", 0),
                    "validated_insights": r.get("validated_insights", 0),
                    "integrated_insights": r.get("integrated_insights", 0),
                    "invalidated_insights": r.get("invalidated_insights", 0),
                    "total_cycles": r.get("total_cycles", 0),
                    "buffer_size": len(self._dream_buffer),
                    "cache_size": len(self._insight_cache),
                }

            return self._stats_from_memory()

        except Exception as exc:
            self._log.warning("stats_query_failed", error=str(exc))
            return self._stats_from_memory()

    # ── Private Helpers ───────────────────────────────────────────

    def _insights_from_cache(
        self,
        status: InsightStatus | None = None,
        limit: int = 50,
    ) -> list[DreamInsight]:
        """Filter the in-memory insight cache by status."""
        source = self._all_insights.values()
        filtered = (
            [i for i in source if i.status == status]
            if status is not None
            else list(source)
        )
        # Sort by created_at descending
        filtered.sort(key=lambda i: i.created_at, reverse=True)
        return filtered[:limit]

    def _themes_from_buffer(self, min_count: int = 3) -> list[dict[str, Any]]:
        """Aggregate themes from the in-memory dream buffer."""
        theme_counts: dict[str, int] = {}
        for dream in self._dream_buffer:
            for theme in dream.themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        results = [
            {"theme": theme, "count": count}
            for theme, count in theme_counts.items()
            if count >= min_count
        ]
        results.sort(key=lambda x: int(str(x["count"])), reverse=True)
        return results

    def _stats_from_memory(self) -> dict[str, Any]:
        """Compute stats from in-memory state only."""
        all_vals = self._all_insights.values()
        pending = sum(1 for i in all_vals if i.status == InsightStatus.PENDING)
        validated = sum(
            1 for i in all_vals if i.status == InsightStatus.VALIDATED
        )
        integrated = sum(
            1 for i in all_vals if i.status == InsightStatus.INTEGRATED
        )
        invalidated = sum(
            1 for i in all_vals if i.status == InsightStatus.INVALIDATED
        )
        return {
            "total_dreams": len(self._dream_buffer),
            "total_insights": len(self._all_insights),
            "pending_insights": pending,
            "validated_insights": validated,
            "integrated_insights": integrated,
            "invalidated_insights": invalidated,
            "total_cycles": 0,
            "buffer_size": len(self._dream_buffer),
            "cache_size": len(self._insight_cache),
        }

    # ── Threat Model Recording ─────────────────────────────────────

    async def record_threat_model(self, result: Any) -> None:
        """
        Persist a ThreatModelResult from the Monte Carlo treasury threat modeler.

        Creates a :ThreatModelEpisode node with portfolio-level tail risk,
        contagion stats, and simulation metadata. Stores critical_exposures
        and hedging_proposals as JSON properties. Links to the parent
        :SleepCycle via [:PRODUCED_THREAT_MODEL].
        """
        if self._neo4j is None:
            self._log.debug("threat_model_recorded_memory_only", result_id=result.id)
            return

        try:
            # Serialize complex nested objects to JSON strings
            position_risks_json = json.dumps(
                {
                    pid: risk.model_dump(mode="json")
                    for pid, risk in result.position_risks.items()
                }
            )
            critical_exposures_json = json.dumps(
                [exp.model_dump(mode="json") for exp in result.critical_exposures]
            )
            hedging_proposals_json = json.dumps(
                [hp.model_dump(mode="json") for hp in result.hedging_proposals]
            )
            portfolio_risk_json = json.dumps(
                result.portfolio_risk.model_dump(mode="json")
            )

            # Create the ThreatModelEpisode node
            await self._neo4j.execute_write(
                """
                MERGE (t:ThreatModelEpisode {id: $id})
                SET t.sleep_cycle_id = $sleep_cycle_id,
                    t.timestamp = $timestamp,
                    t.portfolio_risk_json = $portfolio_risk_json,
                    t.position_risks_json = $position_risks_json,
                    t.critical_exposures_json = $critical_exposures_json,
                    t.hedging_proposals_json = $hedging_proposals_json,
                    t.contagion_events_detected = $contagion_events,
                    t.contagion_loss_amplifier = $contagion_amplifier,
                    t.total_paths_simulated = $total_paths,
                    t.horizon_days = $horizon_days,
                    t.duration_ms = $duration_ms,
                    t.positions_analyzed = $positions_analyzed,
                    t.var_5pct = $var_5pct,
                    t.cvar_5pct = $cvar_5pct,
                    t.liquidation_probability = $liq_prob,
                    t.num_hedging_proposals = $num_hedges,
                    t.num_critical_exposures = $num_critical
                """,
                {
                    "id": result.id,
                    "sleep_cycle_id": result.sleep_cycle_id,
                    "timestamp": result.timestamp.isoformat(),
                    "portfolio_risk_json": portfolio_risk_json,
                    "position_risks_json": position_risks_json,
                    "critical_exposures_json": critical_exposures_json,
                    "hedging_proposals_json": hedging_proposals_json,
                    "contagion_events": result.contagion_events_detected,
                    "contagion_amplifier": float(result.contagion_loss_amplifier),
                    "total_paths": result.total_paths_simulated,
                    "horizon_days": result.horizon_days,
                    "duration_ms": result.duration_ms,
                    "positions_analyzed": result.positions_analyzed,
                    "var_5pct": float(result.portfolio_risk.var_5pct),
                    "cvar_5pct": float(result.portfolio_risk.cvar_5pct),
                    "liq_prob": float(result.portfolio_risk.liquidation_probability),
                    "num_hedges": len(result.hedging_proposals),
                    "num_critical": len(result.critical_exposures),
                },
            )

            # Link to parent SleepCycle
            if result.sleep_cycle_id:
                try:
                    await self._neo4j.execute_write(
                        """
                        MATCH (t:ThreatModelEpisode {id: $threat_id}),
                              (s:SleepCycle {id: $cycle_id})
                        MERGE (s)-[:PRODUCED_THREAT_MODEL]->(t)
                        """,
                        {
                            "threat_id": result.id,
                            "cycle_id": result.sleep_cycle_id,
                        },
                    )
                except Exception as exc:
                    self._log.debug(
                        "threat_model_cycle_link_failed",
                        result_id=result.id,
                        cycle_id=result.sleep_cycle_id,
                        error=str(exc),
                    )

            self._log.info(
                "threat_model_recorded",
                result_id=result.id,
                cycle_id=result.sleep_cycle_id,
                var_5pct=float(result.portfolio_risk.var_5pct),
                hedging_proposals=len(result.hedging_proposals),
                critical_exposures=len(result.critical_exposures),
                duration_ms=result.duration_ms,
            )

        except Exception as exc:
            self._log.warning(
                "threat_model_record_failed",
                result_id=result.id,
                error=str(exc),
            )


# ─── DreamInsightTracker ─────────────────────────────────────────


class DreamInsightTracker:
    """
    Tracks the lifecycle of dream insights through wake-state validation.

    Insights born in REM sleep are PENDING. On wake, the organism can:
    - Validate them (confirmed useful in wake context)
    - Invalidate them (turned out to be noise)
    - Integrate them (absorbed into permanent semantic memory)

    This tracker also counts wake_applications — how many times a
    validated insight influenced a waking decision.
    """

    def __init__(self, journal: DreamJournal) -> None:
        self._journal = journal
        self._log = logger.bind(system="oneiros", component="insight_tracker")

    async def validate_insight(self, insight_id: str, context: str) -> None:
        """
        Mark an insight as VALIDATED — confirmed useful in wake state.

        The context string records how/why it was validated (e.g. which
        waking percept or decision it aligned with).
        """
        insight = self._journal._all_insights.get(insight_id)
        if insight is None:
            self._log.warning("validate_insight_not_found", insight_id=insight_id)
            return

        insight.status = InsightStatus.VALIDATED
        insight.validated_at = utc_now()
        insight.validation_context = context

        # Update caches
        self._journal._all_insights[insight_id] = insight
        if insight_id in self._journal._insight_cache:
            self._journal._insight_cache[insight_id] = insight

        # Persist to Neo4j
        if self._journal._neo4j is not None:
            try:
                await self._journal._neo4j.execute_write(
                    """
                    MATCH (i:DreamInsight {id: $id})
                    SET i.status = $status,
                        i.validated_at = $validated_at,
                        i.validation_context = $context
                    """,
                    {
                        "id": insight_id,
                        "status": InsightStatus.VALIDATED.value,
                        "validated_at": insight.validated_at.isoformat(),
                        "context": context,
                    },
                )
            except Exception as exc:
                self._log.warning(
                    "validate_insight_neo4j_failed",
                    insight_id=insight_id,
                    error=str(exc),
                )

        self._log.info("insight_validated", insight_id=insight_id, context=context[:100])

    async def invalidate_insight(self, insight_id: str, context: str) -> None:
        """
        Mark an insight as INVALIDATED — turned out to be noise.

        The context records why (e.g. "contradicted by waking evidence").
        """
        insight = self._journal._all_insights.get(insight_id)
        if insight is None:
            self._log.warning("invalidate_insight_not_found", insight_id=insight_id)
            return

        insight.status = InsightStatus.INVALIDATED
        insight.validation_context = context

        # Update caches — remove from active cache since it is terminal
        self._journal._all_insights[insight_id] = insight
        self._journal._insight_cache.pop(insight_id, None)

        # Persist
        if self._journal._neo4j is not None:
            try:
                await self._journal._neo4j.execute_write(
                    """
                    MATCH (i:DreamInsight {id: $id})
                    SET i.status = $status,
                        i.validation_context = $context
                    """,
                    {
                        "id": insight_id,
                        "status": InsightStatus.INVALIDATED.value,
                        "context": context,
                    },
                )
            except Exception as exc:
                self._log.warning(
                    "invalidate_insight_neo4j_failed",
                    insight_id=insight_id,
                    error=str(exc),
                )

        self._log.info("insight_invalidated", insight_id=insight_id, context=context[:100])

    async def integrate_insight(self, insight_id: str) -> None:
        """
        Mark an insight as INTEGRATED — it has become permanent semantic knowledge.

        This is the final lifecycle state. The insight's content has been
        absorbed into the knowledge graph as an Entity or strengthened
        relationship, so the insight itself is now archival.
        """
        insight = self._journal._all_insights.get(insight_id)
        if insight is None:
            self._log.warning("integrate_insight_not_found", insight_id=insight_id)
            return

        insight.status = InsightStatus.INTEGRATED

        # Update caches — remove from active insight cache
        self._journal._all_insights[insight_id] = insight
        self._journal._insight_cache.pop(insight_id, None)

        # Persist
        if self._journal._neo4j is not None:
            try:
                await self._journal._neo4j.execute_write(
                    """
                    MATCH (i:DreamInsight {id: $id})
                    SET i.status = $status
                    """,
                    {"id": insight_id, "status": InsightStatus.INTEGRATED.value},
                )
            except Exception as exc:
                self._log.warning(
                    "integrate_insight_neo4j_failed",
                    insight_id=insight_id,
                    error=str(exc),
                )

        self._log.info("insight_integrated", insight_id=insight_id)

    async def record_application(self, insight_id: str) -> None:
        """
        Increment the wake_applications counter for a validated insight.

        Called each time a waking decision references this insight, so
        the organism can track which dream discoveries proved most useful.
        """
        insight = self._journal._all_insights.get(insight_id)
        if insight is None:
            self._log.warning("record_application_not_found", insight_id=insight_id)
            return

        insight.wake_applications += 1

        # Persist
        if self._journal._neo4j is not None:
            try:
                await self._journal._neo4j.execute_write(
                    """
                    MATCH (i:DreamInsight {id: $id})
                    SET i.wake_applications = $count
                    """,
                    {"id": insight_id, "count": insight.wake_applications},
                )
            except Exception as exc:
                self._log.warning(
                    "record_application_neo4j_failed",
                    insight_id=insight_id,
                    error=str(exc),
                )

        self._log.debug(
            "insight_application_recorded",
            insight_id=insight_id,
            total_applications=insight.wake_applications,
        )

    async def get_effectiveness(self) -> dict[str, float]:
        """
        Compute effectiveness ratios for dream insights.

        Returns:
            validated_ratio: validated / total (how often dreams are right)
            integrated_ratio: integrated / validated (how often validated → knowledge)
            invalidated_ratio: invalidated / total (noise rate)
            mean_applications: average wake_applications across validated insights
        """
        all_insights = list(self._journal._all_insights.values())
        total = len(all_insights)
        if total == 0:
            return {
                "validated_ratio": 0.0,
                "integrated_ratio": 0.0,
                "invalidated_ratio": 0.0,
                "mean_applications": 0.0,
            }

        validated = [i for i in all_insights if i.status == InsightStatus.VALIDATED]
        integrated = [i for i in all_insights if i.status == InsightStatus.INTEGRATED]
        invalidated = [i for i in all_insights if i.status == InsightStatus.INVALIDATED]

        validated_count = len(validated) + len(integrated)  # integrated were once validated
        validated_for_integration = len(validated) + len(integrated)

        all_applications = [
            i.wake_applications
            for i in all_insights
            if i.status in (InsightStatus.VALIDATED, InsightStatus.INTEGRATED)
        ]
        mean_apps = sum(all_applications) / max(len(all_applications), 1)

        return {
            "validated_ratio": validated_count / total,
            "integrated_ratio": len(integrated) / max(validated_for_integration, 1),
            "invalidated_ratio": len(invalidated) / total,
            "mean_applications": mean_apps,
        }

    async def get_unvalidated(self) -> list[DreamInsight]:
        """Return all insights still in PENDING status."""
        return await self._journal.get_pending_insights()


# ─── Node Deserialization Helpers ─────────────────────────────────


def _dream_from_node(node: dict[str, Any]) -> Dream | None:
    """Reconstruct a Dream from a Neo4j node property dict."""
    try:
        dream_type_raw = node.get("type", "recombination")
        coherence_class_raw = node.get("coherence_class", "noise")
        timestamp_raw = node.get("timestamp")
        context_raw = node.get("context_json", "{}")

        context: dict[str, Any] = {}
        if context_raw and isinstance(context_raw, str):
            try:
                context = json.loads(context_raw)
            except json.JSONDecodeError:
                context = {}

        # Handle timestamps — Neo4j may return datetime objects or ISO strings
        timestamp = utc_now()
        if timestamp_raw is not None:
            if isinstance(timestamp_raw, str):
                from datetime import datetime
                timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
            else:
                timestamp = timestamp_raw

        return Dream(
            id=node.get("id", ""),
            dream_type=DreamType(dream_type_raw),
            sleep_cycle_id=node.get("sleep_cycle_id", ""),
            timestamp=timestamp,
            coherence_score=float(node.get("coherence_score", 0.0)),
            coherence_class=DreamCoherence(coherence_class_raw),
            affect_valence=float(node.get("affect_valence", 0.0)),
            affect_arousal=float(node.get("affect_arousal", 0.0)),
            bridge_narrative=node.get("bridge_narrative", ""),
            themes=node.get("themes", []),
            summary=node.get("summary", ""),
            context=context,
        )
    except Exception:
        return None


def _insight_from_node(node: dict[str, Any]) -> DreamInsight | None:
    """Reconstruct a DreamInsight from a Neo4j node property dict."""
    try:
        status_raw = node.get("status", "pending")
        created_raw = node.get("created_at")
        validated_raw = node.get("validated_at")

        created_at = utc_now()
        if created_raw is not None:
            if isinstance(created_raw, str):
                from datetime import datetime
                created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            else:
                created_at = created_raw

        validated_at = None
        if validated_raw is not None:
            if isinstance(validated_raw, str):
                from datetime import datetime
                validated_at = datetime.fromisoformat(validated_raw.replace("Z", "+00:00"))
            else:
                validated_at = validated_raw

        return DreamInsight(
            id=node.get("id", ""),
            dream_id=node.get("dream_id", ""),
            sleep_cycle_id=node.get("sleep_cycle_id", ""),
            insight_text=node.get("insight_text", ""),
            coherence_score=float(node.get("coherence_score", 0.0)),
            domain=node.get("domain", ""),
            status=InsightStatus(status_raw),
            validated_at=validated_at,
            validation_context=node.get("validation_context", ""),
            wake_applications=int(node.get("wake_applications", 0)),
            seed_summary=node.get("seed_summary", ""),
            activated_summary=node.get("activated_summary", ""),
            bridge_narrative=node.get("bridge_narrative", ""),
            created_at=created_at,
        )
    except Exception:
        return None


def _cycle_from_node(node: dict[str, Any]) -> SleepCycle | None:
    """Reconstruct a SleepCycle from a Neo4j node property dict."""
    try:
        started_raw = node.get("started_at")
        completed_raw = node.get("completed_at")
        quality_raw = node.get("quality", "normal")

        started_at = utc_now()
        if started_raw is not None:
            if isinstance(started_raw, str):
                from datetime import datetime
                started_at = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
            else:
                started_at = started_raw

        completed_at = None
        if completed_raw is not None:
            if isinstance(completed_raw, str):
                from datetime import datetime
                completed_at = datetime.fromisoformat(completed_raw.replace("Z", "+00:00"))
            else:
                completed_at = completed_raw

        return SleepCycle(
            id=node.get("id", ""),
            started_at=started_at,
            completed_at=completed_at,
            quality=SleepQuality(quality_raw),
            interrupted=bool(node.get("interrupted", False)),
            interrupt_reason=node.get("interrupt_reason", ""),
            episodes_replayed=int(node.get("episodes_replayed", 0)),
            semantic_nodes_created=int(node.get("semantic_nodes_created", 0)),
            traces_pruned=int(node.get("traces_pruned", 0)),
            salience_reduction_mean=float(node.get("salience_reduction_mean", 0.0)),
            beliefs_compressed=int(node.get("beliefs_compressed", 0)),
            hypotheses_pruned=int(node.get("hypotheses_pruned", 0)),
            hypotheses_promoted=int(node.get("hypotheses_promoted", 0)),
            dreams_generated=int(node.get("dreams_generated", 0)),
            insights_discovered=int(node.get("insights_discovered", 0)),
            affect_traces_processed=int(node.get("affect_traces_processed", 0)),
            affect_reduction_mean=float(node.get("affect_reduction_mean", 0.0)),
            threats_simulated=int(node.get("threats_simulated", 0)),
            ethical_cases_digested=int(node.get("ethical_cases_digested", 0)),
            lucid_explorations=int(node.get("lucid_explorations", 0)),
            meta_observations=int(node.get("meta_observations", 0)),
            pressure_before=float(node.get("pressure_before", 0.0)),
            pressure_after=float(node.get("pressure_after", 0.0)),
        )
    except Exception:
        return None
