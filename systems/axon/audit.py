"""
EcodiaOS - Axon Audit Logger

Every action Axon takes is permanently recorded. This is the Honesty drive
at the action layer - the community can always trace what EOS did, when, why,
and what happened.

Audit records are stored in the Memory graph as GovernanceRecord nodes, making
them available for:
  - Equor drift analysis (detecting patterns of constitutional deviation)
  - Evo learning (correlating actions with outcomes)
  - Human oversight and governance review
  - Debugging and incident investigation

The audit logger is async and fire-and-forget from the pipeline's perspective -
it runs concurrently with outcome delivery, adding ≤20ms to total execution time.

Parameters are NEVER logged raw - only their SHA-256 hash is stored.
This protects sensitive data while still enabling deduplication and correlation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.types import AuditRecord, AxonOutcome, ExecutionContext

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

# RE training stream 1 + 2 require GovernanceRecord nodes to be linked to their
# originating Intent (and Episode) so Stream 1 Cypher can join the full
# context → reasoning → action → outcome tuple.
_GOVERNANCE_LINK_CYPHER = """
MATCH (g:GovernanceRecord {id: $governance_id})
OPTIONAL MATCH (i:Intent {id: $intent_id})
FOREACH (_ IN CASE WHEN i IS NOT NULL THEN [1] ELSE [] END |
    MERGE (g)-[:REVIEWED]->(i)
)
WITH g
OPTIONAL MATCH (e:Episode)-[:GENERATED]->(:Intent {id: $intent_id})
FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
    MERGE (g)-[:REVIEWED]->(e)
)
"""

# Stream 2: link a rollback record to the success record that recovered it.
_ROLLBACK_OF_CYPHER = """
MATCH (rollback:GovernanceRecord {id: $rollback_id})
MATCH (success:GovernanceRecord {id: $success_id})
MERGE (rollback)-[:ROLLBACK_OF]->(success)
"""

logger = structlog.get_logger()


class AuditLogger:
    """
    Records every Axon execution as a permanent audit trail in Memory.

    The audit record captures:
    - What was attempted (action_type, parameters_hash, target)
    - What authorised it (equor_verdict, equor_reasoning, autonomy_level)
    - What happened (result, duration_ms)
    - The emotional context at the time (affect_state)

    Records are written to Memory as GovernanceRecord nodes. If Memory is
    unavailable, the record is emitted to the structured log (structlog)
    as a fallback - it will not be silently dropped.

    RE Stream 2 - rollback → success pairing:
    When a rollback record is written, its governance_id is cached keyed by
    intent_id. If a later success record arrives for the same intent_id, a
    [:ROLLBACK_OF] relationship is written from the failure record to the
    success record. The cache is bounded to 64 entries (FIFO eviction) to
    avoid unbounded growth from intents that are never retried.
    """

    _ROLLBACK_CACHE_MAX = 64

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.audit")
        self._records_written: int = 0
        self._records_failed: int = 0
        # intent_id → governance_id of the most recent ROLLED_BACK record
        self._pending_rollbacks: dict[str, str] = {}

    async def log(
        self,
        outcome: AxonOutcome,
        context: ExecutionContext,
    ) -> None:
        """
        Write an audit record for a completed execution.

        Assembles one AuditRecord per intent (not per step) for the overall
        execution outcome. Step-level detail lives in the AxonOutcome itself.
        """
        try:
            # Collect parameters from all steps for hashing
            all_params: dict[str, Any] = {}
            for i, step in enumerate(context.intent.plan.steps):
                all_params[f"step_{i}_{step.executor}"] = step.parameters

            record = AuditRecord.from_outcome(
                outcome=outcome,
                context=context,
                parameters=all_params,
                action_type=_primary_action_type(context),
            )

            await self._write(record)

        except Exception as exc:
            self._records_failed += 1
            self._logger.error(
                "audit_log_failed",
                intent_id=outcome.intent_id,
                execution_id=outcome.execution_id,
                error=str(exc),
            )

    async def _write(self, record: AuditRecord) -> None:
        """Write the audit record to Memory, falling back to structured log."""
        # Always emit to structured log for observability
        self._logger.info(
            "action_audit",
            execution_id=record.execution_id,
            intent_id=record.intent_id,
            action_type=record.action_type,
            target=record.target,
            result=record.result,
            duration_ms=record.duration_ms,
            equor_verdict=record.equor_verdict,
            autonomy_level=record.autonomy_level,
            parameters_hash=record.parameters_hash[:12] + "...",
        )

        # Persist to Memory graph if available
        if self._memory is not None:
            try:
                await self._store_governance_record(record)
                self._records_written += 1
                # RE Stream 2: wire rollback → success pairs
                await self._update_rollback_pairs(record)
            except Exception as exc:
                self._records_failed += 1
                self._logger.error(
                    "audit_memory_write_failed",
                    execution_id=record.execution_id,
                    error=str(exc),
                )
        else:
            self._records_written += 1  # Count as written (to log)

    async def _store_governance_record(self, record: AuditRecord) -> None:
        """
        Store the audit record as a GovernanceRecord node in Neo4j and wire
        [:REVIEWED] relationships to the originating Intent and Episode.

        Neo4j schema:
          (:GovernanceRecord {
            id: str,               ← ULID; used as stable join key
            type: "action_audit",
            execution_id: str,
            intent_id: str,
            equor_verdict: str,
            action_type: str,
            parameters_hash: str,
            target: str,
            result: str,
            duration_ms: int,
            autonomy_level: int,
            timestamp: datetime
          })
          (g:GovernanceRecord)-[:REVIEWED]->(i:Intent {id: intent_id})
          (g:GovernanceRecord)-[:REVIEWED]->(e:Episode) WHERE (e)-[:GENERATED]->(i)

        The [:REVIEWED] relationships are what the RE Stream 1 Cypher joins on.
        """
        record_data = {
            "id": record.id,  # stable ULID for relationship anchoring
            "type": "action_audit",
            "execution_id": record.execution_id,
            "intent_id": record.intent_id,
            "equor_verdict": record.equor_verdict,
            "equor_reasoning": record.equor_reasoning[:500],  # Truncate
            "action_type": record.action_type,
            "parameters_hash": record.parameters_hash,
            "target": record.target,
            "result": record.result,
            "duration_ms": record.duration_ms,
            "autonomy_level": record.autonomy_level,
            "affect_valence": record.affect_state.valence
            if hasattr(record.affect_state, "valence")
            else 0.0,
            "timestamp": record.created_at.isoformat(),
        }

        if hasattr(self._memory, "store_governance_record"):
            await self._memory.store_governance_record(record_data)  # type: ignore[union-attr]
            # Wire [:REVIEWED] → Intent and Episode so RE Stream 1 can join.
            # Uses optional MATCH so missing nodes are silently skipped -
            # the GovernanceRecord is always written even if the Intent/Episode
            # nodes don't exist yet (e.g. fast-fail before Memory write).
            await self._link_governance_to_episode(
                governance_id=record.id,
                intent_id=record.intent_id,
            )
        else:
            # Fallback: log the full record as JSON
            self._logger.info(
                "audit_record_fallback",
                record=json.dumps(record_data),
            )

    async def _link_governance_to_episode(
        self,
        governance_id: str,
        intent_id: str,
    ) -> None:
        """
        Create [:REVIEWED] relationships from the GovernanceRecord to its
        originating Intent node and (via GENERATED) the Episode that produced it.

        Both MATCHes are OPTIONAL - silently a no-op when nodes don't exist.
        This is safe to call concurrently with Memory writes.
        """
        if not hasattr(self._memory, "_neo4j") or self._memory._neo4j is None:  # type: ignore[union-attr]
            return
        try:
            await self._memory._neo4j.execute_write(  # type: ignore[union-attr]
                _GOVERNANCE_LINK_CYPHER,
                {
                    "governance_id": governance_id,
                    "intent_id": intent_id,
                },
            )
        except Exception as exc:
            # Non-fatal - the GovernanceRecord node was already written;
            # the link is a training-data enhancement, not a hard requirement.
            self._logger.warning(
                "governance_episode_link_failed",
                governance_id=governance_id,
                intent_id=intent_id,
                error=str(exc),
            )

    async def _update_rollback_pairs(self, record: AuditRecord) -> None:
        """
        Maintain RE Stream 2 rollback → success linkage.

        - If result == "rolled_back": cache this record's id keyed by intent_id.
        - If result == "success" and there's a cached rollback for this intent_id:
          write [:ROLLBACK_OF] from the failure record to this success record,
          then evict the cache entry.

        Cache is FIFO-bounded to _ROLLBACK_CACHE_MAX entries.
        """
        intent_id = record.intent_id
        result = record.result

        if result == "rolled_back":
            # Cache this failure record for future success linkage
            if len(self._pending_rollbacks) >= self._ROLLBACK_CACHE_MAX:
                # FIFO eviction: remove the oldest entry
                oldest_key = next(iter(self._pending_rollbacks))
                del self._pending_rollbacks[oldest_key]
            self._pending_rollbacks[intent_id] = record.id
            self._logger.debug(
                "rollback_cached",
                intent_id=intent_id,
                governance_id=record.id,
            )

        elif result == "success" and intent_id in self._pending_rollbacks:
            rollback_id = self._pending_rollbacks.pop(intent_id)
            await self._write_rollback_of_relationship(
                rollback_id=rollback_id,
                success_id=record.id,
                intent_id=intent_id,
            )

    async def _write_rollback_of_relationship(
        self,
        rollback_id: str,
        success_id: str,
        intent_id: str,
    ) -> None:
        """
        Write (failure:GovernanceRecord)-[:ROLLBACK_OF]->(success:GovernanceRecord)
        to Neo4j. Both nodes must already exist (written in the same session).
        """
        if not hasattr(self._memory, "_neo4j") or self._memory._neo4j is None:  # type: ignore[union-attr]
            return
        try:
            await self._memory._neo4j.execute_write(  # type: ignore[union-attr]
                _ROLLBACK_OF_CYPHER,
                {
                    "rollback_id": rollback_id,
                    "success_id": success_id,
                },
            )
            self._logger.info(
                "rollback_of_linked",
                rollback_governance_id=rollback_id,
                success_governance_id=success_id,
                intent_id=intent_id,
            )
        except Exception as exc:
            self._logger.warning(
                "rollback_of_link_failed",
                rollback_id=rollback_id,
                success_id=success_id,
                intent_id=intent_id,
                error=str(exc),
            )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "records_written": self._records_written,
            "records_failed": self._records_failed,
            "pending_rollbacks": len(self._pending_rollbacks),
        }


def _primary_action_type(context: ExecutionContext) -> str:
    """Determine the primary action type from the intent plan."""
    if context.intent.plan.steps:
        return context.intent.plan.steps[0].executor
    return "unknown"
