"""
EcodiaOS — Axon Audit Logger

Every action Axon takes is permanently recorded. This is the Honesty drive
at the action layer — the community can always trace what EOS did, when, why,
and what happened.

Audit records are stored in the Memory graph as GovernanceRecord nodes, making
them available for:
  - Equor drift analysis (detecting patterns of constitutional deviation)
  - Evo learning (correlating actions with outcomes)
  - Human oversight and governance review
  - Debugging and incident investigation

The audit logger is async and fire-and-forget from the pipeline's perspective —
it runs concurrently with outcome delivery, adding ≤20ms to total execution time.

Parameters are NEVER logged raw — only their SHA-256 hash is stored.
This protects sensitive data while still enabling deduplication and correlation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.types import AuditRecord, AxonOutcome, ExecutionContext

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

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
    as a fallback — it will not be silently dropped.
    """

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.audit")
        self._records_written: int = 0
        self._records_failed: int = 0

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
        Store the audit record as a GovernanceRecord node in Neo4j.

        Neo4j schema:
          (:GovernanceRecord {
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

        The Memory service exposes store_governance_record() which handles
        the Cypher write.
        """
        record_data = {
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
        else:
            # Fallback: log the full record as JSON
            self._logger.info(
                "audit_record_fallback",
                record=json.dumps(record_data),
            )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "records_written": self._records_written,
            "records_failed": self._records_failed,
        }


def _primary_action_type(context: ExecutionContext) -> str:
    """Determine the primary action type from the intent plan."""
    if context.intent.plan.steps:
        return context.intent.plan.steps[0].executor
    return "unknown"
