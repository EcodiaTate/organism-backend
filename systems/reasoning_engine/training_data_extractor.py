"""
EcodiaOS - RE Training Data Extractor

Batch-extracts the 5 structured training streams from Neo4j as defined in
the speciation bible §2.2. This is historical extraction - it queries what
has already been stored in the graph, rather than collecting real-time events.

Schema reality (verified against Memory/Evo/Kairos/Equor source):

  Stream 1 - Successful reasoning chains
    Episode {id, context_summary, novelty_score, event_time}
    -[:GENERATED]-> Intent {id, action_type, reasoning, confidence}
    -[:RESULTED_IN]-> Outcome {id, success, value_gained}

  Stream 2 - Failures with corrections
    Same node labels; Episode.is_correction = true marks correction episodes.
    Correction linked via Episode -[:FOLLOWED_BY]-> correction:Episode.

  Stream 3 - Constitutional edge cases
    EquorVerdict {id, verdict, composite_alignment, confidence, reasoning,
                  recorded_at, autonomy_level}
    Self -[:CONSCIENCE_VERDICT]-> EquorVerdict
    Bible assumed :EquorCheck / :CHECKED_BY - these do not exist.
    We join EquorVerdict back to Episode via intent_id on Intent.

  Stream 4 - Kairos causal chains
    CausalInvariant {id, abstract_form, cause, effect, confidence,
                     invariance_hold_rate, validated, tier, created_at}
    CausalNode {name} -[:CAUSES {confidence, validated}]-> CausalNode
    Bible's `r.mechanism` does not exist on the CAUSES edge - omitted.

  Stream 5 - Evo experimental results
    Hypothesis {hypothesis_id, statement, status, evidence_score,
                category, created_at}
    No :Experiment or :ExperimentResult nodes in Neo4j (in-memory Pydantic
    objects only). We emit Hypothesis records directly; confirmed/refuted
    status and evidence_score carry the training signal.

No cross-system imports - Neo4j queries only.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger("reasoning_engine.training_data_extractor")

# ─── Lookback defaults ────────────────────────────────────────────────────────

_DEFAULT_LOOKBACK_DAYS = 30
_S3_LOOKBACK_DAYS = 90  # Stream 3 uses wider window (constitutional edge cases are rare)


class TrainingDataExtractor:
    """
    Extracts the 5 RE training data streams from Neo4j.

    Each stream method runs its own Cypher query and returns a list of dicts.
    Stream dicts always include:
        stream_id      (int 1-5)
        episode_id     (str | None)
        novelty_score  (float, default 0.5)
        created_at     (ISO string)

    Call extract_all_streams() to run all five concurrently.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j

    # ─── Concurrent extraction ────────────────────────────────────────────────

    async def extract_all_streams(
        self,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> list[dict[str, Any]]:
        """
        Run all 5 extractions concurrently and return a unified list.

        Each item carries `stream_id` so downstream quality + diversity
        filters can distinguish provenance.
        """
        results = await asyncio.gather(
            self.extract_stream_1_successful_chains(lookback_days),
            self.extract_stream_2_failure_corrections(lookback_days),
            self.extract_stream_3_constitutional_edge_cases(lookback_days),
            self.extract_stream_4_causal_chains(lookback_days),
            self.extract_stream_5_evo_experiments(lookback_days),
            return_exceptions=True,
        )

        combined: list[dict[str, Any]] = []
        stream_names = [
            "successful_chains",
            "failure_corrections",
            "constitutional_edge_cases",
            "causal_chains",
            "evo_experiments",
        ]
        for stream_id, (result, name) in enumerate(zip(results, stream_names), start=1):
            if isinstance(result, BaseException):
                logger.warning(
                    "stream_extraction_failed",
                    stream_id=stream_id,
                    stream_name=name,
                    error=str(result),
                )
            else:
                combined.extend(result)
                logger.info(
                    "stream_extracted",
                    stream_id=stream_id,
                    stream_name=name,
                    count=len(result),
                )

        logger.info(
            "all_streams_extracted",
            total=len(combined),
            lookback_days=lookback_days,
        )
        return combined

    # ─── Stream 1: Successful Reasoning Chains ───────────────────────────────

    async def extract_stream_1_successful_chains(
        self,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> list[dict[str, Any]]:
        """
        Episodes where the intent succeeded with value_gained > 0.3 and
        the intent has a non-trivial reasoning trace (length > 100 chars).

        Adapted from bible §2.2 Stream 1:
          - `intent.description` → `intent.action_type` (actual field name)
          - `ctx.state_snapshot` / HAS_CONTEXT removed (no Context nodes)
          - `eq.result` (EquorCheck) removed - EquorVerdict not directly
            linked to Intent (linked via Self.CONSCIENCE_VERDICT); excluded
            to avoid expensive cross-join
        """
        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        rows = await self._neo4j.execute_read(
            """
            MATCH (ep:Episode)-[:GENERATED]->(intent:Intent)-[:RESULTED_IN]->(outcome:Outcome)
            WHERE outcome.success = true
              AND ep.event_time > datetime($cutoff)
              AND outcome.value_gained > 0.3
              AND intent.reasoning IS NOT NULL
              AND size(intent.reasoning) > 100
            RETURN {
                stream_id:        1,
                episode_id:       ep.id,
                context_summary:  ep.context_summary,
                reasoning_chain:  intent.reasoning,
                decision:         intent.action_type,
                confidence:       intent.confidence,
                value_gained:     outcome.value_gained,
                novelty_score:    coalesce(ep.novelty_score, 0.5),
                created_at:       toString(ep.event_time)
            } AS example
            ORDER BY outcome.value_gained * coalesce(ep.novelty_score, 0.5) DESC
            LIMIT 3000
            """,
            {"cutoff": cutoff},
        )
        return [r["example"] for r in rows]

    # ─── Stream 2: Failures With Corrections ─────────────────────────────────

    async def extract_stream_2_failure_corrections(
        self,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> list[dict[str, Any]]:
        """
        Failed intent episodes that were followed by a correction episode
        (Episode.is_correction = true linked via FOLLOWED_BY).

        Adapted from bible §2.2 Stream 2:
          - `outcome.failure_analysis` doesn't exist → use `outcome.error_message`
          - `correction.intent_description` → `correction.context_summary`
            (no intent_description property on Episode)
        """
        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        rows = await self._neo4j.execute_read(
            """
            MATCH (ep:Episode)-[:GENERATED]->(intent:Intent)-[:RESULTED_IN]->(outcome:Outcome)
            WHERE outcome.success = false
              AND ep.event_time > datetime($cutoff)
              AND outcome.error_message IS NOT NULL
            OPTIONAL MATCH (ep)-[:FOLLOWED_BY]->(correction:Episode)
            WHERE correction.is_correction = true
            RETURN {
                stream_id:           2,
                episode_id:          ep.id,
                context_summary:     ep.context_summary,
                failed_reasoning:    intent.reasoning,
                failed_decision:     intent.action_type,
                failure_analysis:    outcome.error_message,
                correction_context:  correction.context_summary,
                novelty_score:       coalesce(ep.novelty_score, 0.5),
                created_at:          toString(ep.event_time)
            } AS failure_example
            LIMIT 1000
            """,
            {"cutoff": cutoff},
        )
        return [r["failure_example"] for r in rows]

    # ─── Stream 3: Constitutional Edge Cases ─────────────────────────────────

    async def extract_stream_3_constitutional_edge_cases(
        self,
        lookback_days: int = _S3_LOOKBACK_DAYS,
    ) -> list[dict[str, Any]]:
        """
        EquorVerdict nodes where the verdict was 'blocked' or 'deferred'
        (i.e. Equor intervened - MODIFY/ESCALATE/DENY in the bible's terms).

        Adapted from bible §2.2 Stream 3:
          - :EquorCheck / CHECKED_BY do not exist → :EquorVerdict, CONSCIENCE_VERDICT
          - EquorVerdict is not directly linked to Intent or Episode
            (linked Self-[:CONSCIENCE_VERDICT]->EquorVerdict with intent_id stored)
          - We query EquorVerdict directly and join back to Intent via intent_id,
            then to Episode via episode_id on Intent
          - `eq.principles_applied` → `ev.reasoning` (actual field)
          - `eq.intervention_reason` → `ev.reasoning`
        """
        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        rows = await self._neo4j.execute_read(
            """
            MATCH (:Self)-[:CONSCIENCE_VERDICT]->(ev:EquorVerdict)
            WHERE ev.verdict IN ['blocked', 'deferred']
              AND ev.recorded_at > datetime($cutoff)
            OPTIONAL MATCH (intent:Intent {id: ev.intent_id})
            OPTIONAL MATCH (ep:Episode {id: intent.episode_id})
            RETURN {
                stream_id:              3,
                episode_id:             ep.id,
                context_summary:        ep.context_summary,
                original_intent:        intent.action_type,
                equor_intervention:     ev.reasoning,
                verdict:                ev.verdict,
                composite_alignment:    ev.composite_alignment,
                confidence:             ev.confidence,
                novelty_score:          coalesce(ep.novelty_score, 0.5),
                created_at:             toString(ev.recorded_at)
            } AS constitutional_example
            LIMIT 500
            """,
            {"cutoff": cutoff},
        )
        return [r["constitutional_example"] for r in rows]

    # ─── Stream 4: Kairos Causal Chains ──────────────────────────────────────

    async def extract_stream_4_causal_chains(
        self,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> list[dict[str, Any]]:
        """
        Validated CausalNode → CAUSES → CausalNode pairs with confidence > 0.7.

        Adapted from bible §2.2 Stream 4:
          - `r.mechanism` doesn't exist on CAUSES edge → omitted
          - `cause<-[:OBSERVED_IN]-(ep:Episode)` doesn't exist → omitted
          - Using CausalInvariant nodes for richer context (abstract_form + tier)
          - CausalNode.name is the canonical field (not .description)
        """
        rows = await self._neo4j.execute_read(
            """
            MATCH (cause:CausalNode)-[r:CAUSES]->(effect:CausalNode)
            WHERE r.confidence > 0.7 AND r.validated = true
            OPTIONAL MATCH (ci:CausalInvariant {id: r.invariant_id})
            RETURN {
                stream_id:      4,
                episode_id:     null,
                cause:          cause.name,
                effect:         effect.name,
                abstract_form:  ci.abstract_form,
                tier:           ci.tier,
                confidence:     r.confidence,
                novelty_score:  coalesce(ci.recency_weight, 0.5),
                created_at:     toString(coalesce(ci.created_at, datetime()))
            } AS causal_example
            ORDER BY r.confidence DESC
            LIMIT 500
            """,
            {},
        )
        return [r["causal_example"] for r in rows]

    # ─── Stream 5: Evo Experimental Results ──────────────────────────────────

    async def extract_stream_5_evo_experiments(
        self,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> list[dict[str, Any]]:
        """
        Hypothesis nodes with terminal status (supported/refuted/integrated).

        Adapted from bible §2.2 Stream 5:
          - :Experiment and :ExperimentResult nodes do NOT exist in Neo4j
            (ExperimentDesign / ExperimentResult are in-memory Pydantic objects only)
          - :TESTED_IN / :PRODUCED relationships do not exist
          - We query :Hypothesis directly; status + evidence_score carry the result
          - `h.description` → `h.statement` (actual field)
          - `h.reasoning` doesn't exist on Hypothesis nodes → omitted
          - `result.surprise_score` → `ep.novelty_score` is unavailable per-hyp;
            we derive surprise from evidence_score deviation from midpoint
        """
        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        rows = await self._neo4j.execute_read(
            """
            MATCH (h:Hypothesis)
            WHERE h.status IN ['supported', 'refuted', 'integrated', 'archived']
              AND h.created_at > datetime($cutoff)
            RETURN {
                stream_id:       5,
                episode_id:      null,
                hypothesis:      h.statement,
                category:        h.category,
                status:          h.status,
                evidence_score:  h.evidence_score,
                confirmed:       h.status IN ['supported', 'integrated'],
                supporting_count: h.supporting_count,
                created_at:      toString(h.created_at),
                novelty_score:   CASE
                    WHEN h.evidence_score > 5.0 THEN 0.9
                    WHEN h.evidence_score > 3.0 THEN 0.7
                    ELSE 0.5
                END
            } AS evo_example
            ORDER BY h.evidence_score DESC
            LIMIT 500
            """,
            {"cutoff": cutoff},
        )
        return [r["evo_example"] for r in rows]

    # ─── Per-stream stats ─────────────────────────────────────────────────────

    async def stream_counts(
        self,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> dict[str, int]:
        """
        Cheap COUNT queries for the stats CLI subcommand.
        Does not fetch full records - much faster than extract_all_streams().
        """
        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()

        s3_cutoff = (
            datetime.now(UTC) - timedelta(days=_S3_LOOKBACK_DAYS)
        ).isoformat()

        queries: list[tuple[str, str, dict[str, Any]]] = [
            (
                "stream_1_successful_chains",
                """
                MATCH (ep:Episode)-[:GENERATED]->(i:Intent)-[:RESULTED_IN]->(o:Outcome)
                WHERE o.success = true AND ep.event_time > datetime($cutoff)
                  AND o.value_gained > 0.3 AND size(i.reasoning) > 100
                RETURN count(*) AS n
                """,
                {"cutoff": cutoff},
            ),
            (
                "stream_2_failure_corrections",
                """
                MATCH (ep:Episode)-[:GENERATED]->(i:Intent)-[:RESULTED_IN]->(o:Outcome)
                WHERE o.success = false AND ep.event_time > datetime($cutoff)
                  AND o.error_message IS NOT NULL
                RETURN count(*) AS n
                """,
                {"cutoff": cutoff},
            ),
            (
                "stream_3_constitutional_edge_cases",
                """
                MATCH (:Self)-[:CONSCIENCE_VERDICT]->(ev:EquorVerdict)
                WHERE ev.verdict IN ['blocked', 'deferred']
                  AND ev.recorded_at > datetime($cutoff)
                RETURN count(*) AS n
                """,
                {"cutoff": s3_cutoff},
            ),
            (
                "stream_4_causal_chains",
                """
                MATCH (:CausalNode)-[r:CAUSES]->(:CausalNode)
                WHERE r.confidence > 0.7 AND r.validated = true
                RETURN count(*) AS n
                """,
                {},
            ),
            (
                "stream_5_evo_experiments",
                """
                MATCH (h:Hypothesis)
                WHERE h.status IN ['supported', 'refuted', 'integrated', 'archived']
                  AND h.created_at > datetime($cutoff)
                RETURN count(*) AS n
                """,
                {"cutoff": cutoff},
            ),
        ]

        counts: dict[str, int] = {}
        for name, query, params in queries:
            try:
                rows = await self._neo4j.execute_read(query, params)
                counts[name] = int(rows[0]["n"]) if rows else 0
            except Exception as exc:
                logger.warning("stream_count_failed", stream=name, error=str(exc))
                counts[name] = -1  # sentinel for "query failed"

        return counts
