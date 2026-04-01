"""
EcodiaOS - Query Memory Executor

Queries the organism's Neo4j memory graph for knowledge retrieval.
This is a read-only executor that retrieves structured knowledge
without modifying the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import ExecutionContext, ExecutionResult, RateLimit, ValidationResult
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class QueryMemoryGraphExecutor(Executor):
    action_type = "query_memory"
    description = "Query the organism's Neo4j memory graph for knowledge retrieval"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 15_000
    rate_limit = RateLimit.per_minute(30)
    counts_toward_budget = False  # Read-only, should not consume budget
    emits_to_atune = False  # Avoid feedback loops from memory reads

    def __init__(self, memory: Any = None, event_bus: Any = None) -> None:
        self._memory = memory
        self._event_bus = event_bus

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        query_type = params.get("query_type")
        valid_types = {"cypher", "semantic", "entity_lookup", "recent_episodes"}
        if not query_type or query_type not in valid_types:
            return ValidationResult.fail(
                f"'query_type' must be one of {sorted(valid_types)}"
            )
        if query_type == "cypher":
            cypher = params.get("cypher")
            if not cypher or not isinstance(cypher, str):
                return ValidationResult.fail("'cypher' is required for cypher query_type")
            # Safety: reject mutations
            upper = cypher.upper()
            if any(kw in upper for kw in ("CREATE", "MERGE", "DELETE", "SET ", "REMOVE ")):
                return ValidationResult.fail("Mutation queries are not allowed via query_memory")
        elif query_type == "semantic":
            if not params.get("query_text"):
                return ValidationResult.fail("'query_text' is required for semantic query_type")
        elif query_type == "entity_lookup":
            if not params.get("entity_id") and not params.get("entity_name"):
                return ValidationResult.fail(
                    "'entity_id' or 'entity_name' required for entity_lookup"
                )
        return ValidationResult.ok()

    async def execute(self, params: dict[str, Any], context: ExecutionContext) -> ExecutionResult:
        query_type = params["query_type"]
        limit = params.get("limit", 20)

        try:
            if self._memory is None:
                return ExecutionResult(
                    success=False,
                    error="No memory service configured",
                )

            results: list[dict[str, Any]] = []

            if query_type == "cypher":
                raw = await self._memory.query(params["cypher"], limit=limit)
                results = raw if isinstance(raw, list) else [raw] if raw else []

            elif query_type == "semantic":
                raw = await self._memory.semantic_search(
                    query=params["query_text"],
                    limit=limit,
                )
                results = raw if isinstance(raw, list) else []

            elif query_type == "entity_lookup":
                entity_id = params.get("entity_id", "")
                entity_name = params.get("entity_name", "")
                if entity_id:
                    raw = await self._memory.get_entity(entity_id)
                else:
                    raw = await self._memory.find_entity(name=entity_name)
                results = [raw] if raw else []

            elif query_type == "recent_episodes":
                raw = await self._memory.recent_episodes(limit=limit)
                results = raw if isinstance(raw, list) else []

            # RE trace only - AXON_EXECUTION_RESULT is the canonical aggregate.
            await self._emit_re_trace(context, params, success=True, result_count=len(results))

            return ExecutionResult(
                success=True,
                data={
                    "query_type": query_type,
                    "result_count": len(results),
                    "results": results[:limit],
                },
                new_observations=[
                    f"Memory query ({query_type}) returned {len(results)} results"
                ],
            )
        except Exception as exc:
            await self._emit_re_trace(context, params, success=False, result_count=0, error=str(exc))
            return ExecutionResult(success=False, error=str(exc))

    async def _emit_event(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=event_type,
                source_system="axon",
                data=data,
            ))
        except Exception:
            pass

    async def _emit_re_trace(
        self,
        context: ExecutionContext,
        params: dict[str, Any],
        success: bool,
        result_count: int = 0,
        error: str = "",
    ) -> None:
        if self._event_bus is None:
            return
        try:
            import json as _json
            from primitives.common import DriveAlignmentVector, SystemID
            from primitives.re_training import RETrainingExample

            query_type = params.get("query_type", "")
            limit = params.get("limit", 20)

            # Build query summary without exposing raw Cypher or entity IDs verbatim
            if query_type == "cypher":
                query_summary = f"cypher: {params.get('cypher', '')[:120]!r}"
            elif query_type == "semantic":
                query_summary = f"semantic: {params.get('query_text', '')[:120]!r}"
            elif query_type == "entity_lookup":
                query_summary = f"entity_lookup: id={params.get('entity_id', '')} name={params.get('entity_name', '')!r}"
            else:
                query_summary = f"recent_episodes: limit={limit}"

            # Quality: empty result on success is a soft failure (query ran but found nothing)
            if success and result_count > 0:
                quality = 1.0
                quality_reason = "query succeeded with results"
            elif success and result_count == 0:
                quality = 0.5
                quality_reason = "query succeeded but returned no results - possible knowledge gap"
            else:
                quality = 0.0
                quality_reason = f"query failed: {error[:80]}"

            reasoning_trace = "\n".join([
                f"1. VALIDATE: query_type={query_type!r}, mutation check={'passed' if query_type != 'cypher' else 'verified no CREATE/MERGE/DELETE/SET/REMOVE'}",
                f"2. MEMORY SERVICE: {'configured' if self._memory is not None else 'MISSING'}",
                f"3. EXECUTE: {query_summary}",
                f"4. RESULT: result_count={result_count}, limit={limit}",
                f"5. QUALITY: {quality_reason}",
            ])

            alternatives = []
            if query_type == "cypher":
                alternatives.append(
                    "Alternative: use 'semantic' query_type for natural-language retrieval when Cypher syntax is uncertain"
                )
            if result_count == 0 and success:
                alternatives.append(
                    "Alternative: broaden query constraints or use semantic fallback - zero results may indicate memory gap or overly specific filters"
                )
                alternatives.append(
                    "Alternative: trigger memory consolidation (trigger_consolidation executor) if episodic → schema compression may have removed relevant nodes"
                )
            if not success:
                alternatives.append(
                    "Alternative: retry with simpler query_type (entity_lookup or recent_episodes) as fallback if Cypher/semantic fails"
                )

            counterfactual = ""
            if success and result_count == 0:
                counterfactual = (
                    f"If the {query_type} query had returned results, downstream reasoning steps could "
                    f"have grounded their actions in retrieved knowledge. The empty result forces the "
                    f"organism to act under higher uncertainty or request clarification."
                )
            elif not success:
                counterfactual = (
                    f"If the memory query had succeeded, the organism would have retrieved grounded "
                    f"context to inform its next action. Failure means the subsequent step proceeds "
                    f"from prior belief alone - increasing the risk of a hallucinated or stale response."
                )

            equor_alignment = getattr(context.equor_check, "drive_alignment", None) if context.equor_check is not None else None

            trace = RETrainingExample(
                source_system=SystemID.AXON,
                episode_id=context.execution_id,
                instruction=f"Query memory graph: {query_summary[:150]}",
                input_context=_json.dumps({
                    "query_type": query_type,
                    "limit": limit,
                    "query_summary": query_summary[:300],
                }),
                output=_json.dumps({
                    "success": success,
                    "result_count": result_count,
                    "quality_reason": quality_reason,
                    "error": error[:200] if error else None,
                }),
                outcome_quality=quality,
                category="memory_query",
                constitutional_alignment=equor_alignment or DriveAlignmentVector(),
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives,
                counterfactual=counterfactual,
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon",
                data=trace.model_dump(mode="json"),
            ))
        except Exception:
            pass
