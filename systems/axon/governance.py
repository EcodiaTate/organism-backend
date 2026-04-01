"""
EcodiaOS - Axon Governance

Records governance decisions for action execution, persists them to Neo4j,
and emits RE training traces so the reasoning engine can learn governance
patterns over time.

GovernanceRecord captures the decision (approved/denied/escalated), the
reason, the action type, and the autonomy level required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import (
    DriveAlignmentVector,
    EOSBaseModel,
    SystemID,
    new_id,
    utc_now,
)
from primitives.re_training import RETrainingExample
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class GovernanceRecord(EOSBaseModel):
    """A recorded governance decision for an action execution."""

    id: str = ""
    execution_id: str = ""
    action_type: str = ""
    decision: str = ""  # "approved" | "denied" | "escalated"
    reason: str = ""
    autonomy_level: int = 0
    requestor_system: str = ""
    metadata: dict[str, Any] = {}


class GovernanceLogger:
    """
    Logs governance decisions to Neo4j and emits RE training traces.

    Accepts an optional Neo4j async driver and Synapse event bus.
    When neither is available, decisions are logged structurally but not persisted.
    """

    def __init__(
        self,
        neo4j: Any = None,
        event_bus: Any = None,
    ) -> None:
        self._neo4j = neo4j
        self._event_bus = event_bus
        self._logger = logger.bind(system="axon.governance")
        self._total_decisions: int = 0
        self._approved: int = 0
        self._denied: int = 0

    async def record_decision(self, record: GovernanceRecord) -> None:
        """
        Record a governance decision: persist to Neo4j and emit RE training trace.

        Args:
            record: The governance decision to persist.
        """
        if not record.id:
            record.id = new_id()

        self._total_decisions += 1
        if record.decision == "approved":
            self._approved += 1
        elif record.decision == "denied":
            self._denied += 1

        self._logger.info(
            "governance_decision_recorded",
            record_id=record.id,
            action_type=record.action_type,
            decision=record.decision,
            autonomy_level=record.autonomy_level,
        )

        # Persist to Neo4j
        await self._persist_governance_record(record)

        # Emit RE training trace
        await self._emit_re_training_trace(record)

    async def _persist_governance_record(self, record: GovernanceRecord) -> None:
        """Persist governance record as a Neo4j node with optional audit link."""
        if self._neo4j is None:
            return
        try:
            await self._neo4j.execute_write(
                """
                CREATE (gr:GovernanceRecord {
                    id: $id,
                    action_type: $action_type,
                    decision: $decision,
                    reason: $reason,
                    autonomy_level: $autonomy_level,
                    requestor_system: $requestor_system,
                    timestamp: datetime()
                })
                WITH gr
                OPTIONAL MATCH (a:AuditRecord {execution_id: $execution_id})
                FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (a)-[:GOVERNED_BY]->(gr)
                )
                RETURN gr.id AS id
                """,
                {
                    "id": record.id,
                    "action_type": record.action_type,
                    "decision": record.decision,
                    "reason": record.reason,
                    "autonomy_level": record.autonomy_level,
                    "requestor_system": record.requestor_system,
                    "execution_id": record.execution_id,
                },
            )
        except Exception as exc:
            self._logger.warning(
                "governance_neo4j_persist_failed",
                record_id=record.id,
                error=str(exc),
            )

    async def _emit_re_training_trace(self, record: GovernanceRecord) -> None:
        """Emit RE training trace for governance reasoning."""
        if self._event_bus is None:
            return
        try:
            trace = RETrainingExample(
                source_system=SystemID.AXON,
                instruction=f"Governance decision for {record.action_type}",
                input_context=f"autonomy_level={record.autonomy_level}",
                output=f"decision={record.decision}, reason={record.reason}",
                outcome_quality=1.0 if record.decision == "approved" else 0.5,
                category="governance_reasoning",
                constitutional_alignment=DriveAlignmentVector(),
                timestamp=utc_now(),
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon",
                data=trace.model_dump(mode="json"),
            ))
        except Exception as exc:
            self._logger.debug(
                "governance_re_trace_emit_failed",
                record_id=record.id,
                error=str(exc),
            )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_decisions": self._total_decisions,
            "approved": self._approved,
            "denied": self._denied,
        }
