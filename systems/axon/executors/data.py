"""
EcodiaOS - Axon Data Executors

Data executors create and modify persistent records. They are Level 2
(PARTNER autonomy) because they change shared world state in ways that
other community members can see and depend on.

CreateRecordExecutor - create a new data record in the knowledge graph
UpdateRecordExecutor - update an existing record
ScheduleExecutor     - schedule a future event (stored in Redis sorted set)
ReminderExecutor     - set a reminder for a user or for EOS itself

Create and Update are reversible - their rollback handlers delete or revert
the changes they made. Schedule and Reminder are reversible by cancellation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    RollbackResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()


# ─── CreateRecordExecutor ─────────────────────────────────────────


class CreateRecordExecutor(Executor):
    """
    Create a new data record in the EOS knowledge graph.

    Records can be any entity type (Event, Task, Note, Community resource, etc.)
    They are stored as Neo4j nodes and become part of EOS's world model.

    Required params:
      record_type (str): Entity type (e.g., "Event", "Task", "Note").
      data (dict): Properties for the record.

    Optional params:
      title (str): Human-readable title. Default derived from data.
      tags (list[str]): Labels for retrieval. Default [].
      related_to (list[str]): IDs of related records. Default [].
    """

    action_type = "create_record"
    description = "Create a new data record in the knowledge graph (Level 2)"
    required_autonomy = 2
    reversible = True
    max_duration_ms = 3000
    rate_limit = RateLimit.per_minute(20)

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.executor.create_record")
        # Track created record IDs for rollback
        self._created: dict[str, str] = {}  # execution_id → record_id

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("record_type"):
            return ValidationResult.fail("record_type is required")
        if not params.get("data"):
            return ValidationResult.fail("data is required and must be non-empty dict")
        if not isinstance(params["data"], dict):
            return ValidationResult.fail("data must be a dict")
        record_type = params["record_type"]
        if not isinstance(record_type, str) or len(record_type) > 100:
            return ValidationResult.fail("record_type must be a string (max 100 chars)")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        record_type = params["record_type"]
        data = dict(params["data"])
        title = params.get("title", data.get("title", f"New {record_type}"))
        tags = list(params.get("tags", []))
        related_to = list(params.get("related_to", []))

        self._logger.info(
            "create_record_execute",
            record_type=record_type,
            title=str(title)[:60],
            execution_id=context.execution_id,
        )

        if self._memory is not None:
            try:
                # resolve_and_create_entity handles deduplication and returns (id, was_created)
                description = str(data.get("description", f"{record_type}: {title}"))
                record_id, was_created = await self._memory.resolve_and_create_entity(
                    name=str(title),
                    entity_type=record_type,
                    description=description,
                )
                self._created[context.execution_id] = record_id
                return ExecutionResult(
                    success=True,
                    data={"record_id": record_id, "record_type": record_type, "title": title,
                          "was_created": was_created, "tags": tags, "related_to": related_to},
                    side_effects=[f"{'Created' if was_created else 'Merged'} {record_type} "
                                  f"record: '{title}' (id={record_id})"],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Record creation failed: {exc}",
                )
        else:
            from primitives.common import new_id
            record_id = new_id()
            return ExecutionResult(
                success=True,
                data={"record_id": record_id, "record_type": record_type,
                      "tags": tags, "related_to": related_to, "note": "No memory service"},
                side_effects=[f"Record creation staged: {record_type} '{title}'"],
            )

    async def rollback(
        self,
        execution_id: str,
        context: ExecutionContext,
    ) -> RollbackResult:
        record_id = self._created.get(execution_id)
        if not record_id:
            return RollbackResult(success=False, reason="No record ID found for this execution")

        if self._memory is not None:
            try:
                await self._memory.delete_entity(record_id)  # type: ignore[attr-defined]
                del self._created[execution_id]
                return RollbackResult(
                    success=True,
                    side_effects_reversed=[f"Deleted record {record_id}"],
                )
            except Exception as exc:
                return RollbackResult(success=False, reason=f"Delete failed: {exc}")

        return RollbackResult(success=False, reason="No memory service for rollback")


# ─── UpdateRecordExecutor ─────────────────────────────────────────


class UpdateRecordExecutor(Executor):
    """
    Update properties of an existing record in the knowledge graph.

    Required params:
      record_id (str): ID of the record to update.
      updates (dict): Properties to set or update.

    Optional params:
      merge (bool): If True, merge with existing properties. Default True.
    """

    action_type = "update_record"
    description = "Update an existing record in the knowledge graph (Level 2)"
    required_autonomy = 2
    reversible = True
    max_duration_ms = 3000
    rate_limit = RateLimit.per_minute(30)

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.executor.update_record")
        # Track previous states for rollback: execution_id → (record_id, previous_data)
        self._previous_states: dict[str, tuple[str, dict[str, Any]]] = {}

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("record_id"):
            return ValidationResult.fail("record_id is required")
        if not params.get("updates"):
            return ValidationResult.fail("updates is required and must be non-empty dict")
        if not isinstance(params["updates"], dict):
            return ValidationResult.fail("updates must be a dict")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        record_id = params["record_id"]
        updates = dict(params["updates"])
        merge = bool(params.get("merge", True))

        self._logger.info(
            "update_record_execute",
            record_id=record_id,
            keys_updated=list(updates.keys()),
            execution_id=context.execution_id,
        )

        if self._memory is not None:
            try:
                # Snapshot previous state for rollback
                previous = await self._memory.get_entity(record_id)  # type: ignore[attr-defined]
                if previous:
                    self._previous_states[context.execution_id] = (record_id, dict(previous))

                await self._memory.update_entity(  # type: ignore[attr-defined]
                    entity_id=record_id,
                    properties=updates,
                    merge=merge,
                )
                return ExecutionResult(
                    success=True,
                    data={"record_id": record_id, "keys_updated": list(updates.keys())},
                    side_effects=[f"Updated record {record_id}: {list(updates.keys())}"],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Record update failed: {exc}",
                )
        else:
            return ExecutionResult(
                success=True,
                data={"record_id": record_id, "note": "No memory service"},
                side_effects=[f"Update staged for record {record_id}"],
            )

    async def rollback(
        self,
        execution_id: str,
        context: ExecutionContext,
    ) -> RollbackResult:
        previous = self._previous_states.get(execution_id)
        if not previous:
            return RollbackResult(success=False, reason="No snapshot found for this execution")

        record_id, previous_data = previous
        if self._memory is not None:
            try:
                await self._memory.update_entity(  # type: ignore[attr-defined]
                    entity_id=record_id,
                    properties=previous_data,
                    merge=False,
                )
                del self._previous_states[execution_id]
                return RollbackResult(
                    success=True,
                    side_effects_reversed=[f"Reverted record {record_id} to previous state"],
                )
            except Exception as exc:
                return RollbackResult(success=False, reason=f"Revert failed: {exc}")

        return RollbackResult(success=False, reason="No memory service for rollback")


# ─── ScheduleExecutor ─────────────────────────────────────────────


class ScheduleExecutor(Executor):
    """
    Schedule a future event (e.g., meeting, reminder, task deadline).

    Events are stored in a Redis sorted set keyed by timestamp, allowing
    EOS to query "what's happening soon?" efficiently. They are also
    stored as Event nodes in the Memory graph for semantic retrieval.

    Required params:
      title (str): Event title.
      scheduled_at (str): ISO-8601 datetime string.

    Optional params:
      description (str): Event description. Default "".
      participants (list[str]): User IDs involved. Default [].
      duration_minutes (int): Expected duration. Default 60.
      recurrence (str): "none" | "daily" | "weekly" | "monthly". Default "none".
    """

    action_type = "schedule_event"
    description = "Schedule a future event in the community calendar (Level 2)"
    required_autonomy = 2
    reversible = True
    max_duration_ms = 3000
    rate_limit = RateLimit.per_hour(50)

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client
        self._logger = logger.bind(system="axon.executor.schedule_event")
        self._scheduled: dict[str, str] = {}  # execution_id → event_id

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("title"):
            return ValidationResult.fail("title is required")
        if not params.get("scheduled_at"):
            return ValidationResult.fail("scheduled_at is required (ISO-8601 datetime)")
        try:
            from datetime import datetime
            datetime.fromisoformat(str(params["scheduled_at"]))
        except (ValueError, TypeError):
            return ValidationResult.fail(
                "scheduled_at must be a valid ISO-8601 datetime string"
            )
        recurrence = params.get("recurrence", "none")
        if recurrence not in ("none", "daily", "weekly", "monthly"):
            return ValidationResult.fail(
                "recurrence must be 'none', 'daily', 'weekly', or 'monthly'"
            )
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        import json as _json
        from datetime import datetime

        title = params["title"]
        scheduled_at = str(params["scheduled_at"])
        description = str(params.get("description", ""))
        participants = list(params.get("participants", []))
        duration_minutes = int(params.get("duration_minutes", 60))
        recurrence = params.get("recurrence", "none")

        self._logger.info(
            "schedule_event_execute",
            title=title,
            scheduled_at=scheduled_at,
            execution_id=context.execution_id,
        )

        from primitives.common import new_id
        event_id = new_id()
        event_data = {
            "id": event_id,
            "title": title,
            "scheduled_at": scheduled_at,
            "description": description,
            "participants": participants,
            "duration_minutes": duration_minutes,
            "recurrence": recurrence,
            "created_by": "eos",
            "execution_id": context.execution_id,
        }

        if self._redis is not None:
            try:
                # Parse timestamp for sorted set score
                ts = datetime.fromisoformat(scheduled_at).timestamp()
                await self._redis.zadd(
                    "eos:scheduled_events",
                    {_json.dumps(event_data): ts},
                )
                self._scheduled[context.execution_id] = event_id
                return ExecutionResult(
                    success=True,
                    data={"event_id": event_id, "scheduled_at": scheduled_at},
                    side_effects=[f"Event '{title}' scheduled for {scheduled_at}"],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Schedule failed: {exc}",
                )
        else:
            self._scheduled[context.execution_id] = event_id
            return ExecutionResult(
                success=True,
                data={"event_id": event_id, "note": "No Redis client - event staged"},
                side_effects=[f"Event '{title}' scheduled for {scheduled_at} (staged)"],
            )

    async def rollback(
        self,
        execution_id: str,
        context: ExecutionContext,
    ) -> RollbackResult:
        event_id = self._scheduled.get(execution_id)
        if not event_id:
            return RollbackResult(success=False, reason="No event ID for this execution")

        if self._redis is not None:
            try:
                # Remove from sorted set by scanning - Phase 1 approach
                # Phase 2: store event_id → key mapping for O(1) removal
                result = await self._redis.zrangebyscore(
                    "eos:scheduled_events", "-inf", "+inf"
                )
                for item in result:
                    import json as _json
                    try:
                        data = _json.loads(item)
                        if data.get("id") == event_id:
                            await self._redis.zrem("eos:scheduled_events", item)
                            del self._scheduled[execution_id]
                            return RollbackResult(
                                success=True,
                                side_effects_reversed=[f"Removed scheduled event {event_id}"],
                            )
                    except Exception:
                        continue
            except Exception as exc:
                return RollbackResult(success=False, reason=f"Rollback failed: {exc}")

        return RollbackResult(success=False, reason="No Redis client for rollback")


# ─── ReminderExecutor ─────────────────────────────────────────────


class ReminderExecutor(Executor):
    """
    Set a reminder for a user or for EOS itself.

    Reminders are stored in Redis as time-delayed messages. When the time arrives,
    Synapse (Phase 9) will trigger a Percept that re-enters the workspace.

    Required params:
      message (str): The reminder message.
      remind_at (str): ISO-8601 datetime when the reminder should fire.

    Optional params:
      recipient_id (str): User to remind. If absent, EOS reminds itself.
      context_note (str): Extra context for why this reminder was set. Default "".
    """

    action_type = "set_reminder"
    description = "Set a time-delayed reminder for a user or for EOS (Level 2)"
    required_autonomy = 2
    reversible = True
    max_duration_ms = 2000
    rate_limit = RateLimit.per_hour(30)

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client
        self._logger = logger.bind(system="axon.executor.set_reminder")
        self._reminders: dict[str, str] = {}  # execution_id → reminder_id

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("message"):
            return ValidationResult.fail("message is required")
        if not params.get("remind_at"):
            return ValidationResult.fail("remind_at is required (ISO-8601 datetime)")
        try:
            from datetime import datetime
            datetime.fromisoformat(str(params["remind_at"]))
        except (ValueError, TypeError):
            return ValidationResult.fail("remind_at must be a valid ISO-8601 datetime string")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        import json as _json
        from datetime import datetime

        from primitives.common import new_id

        message = params["message"]
        remind_at = str(params["remind_at"])
        recipient_id = params.get("recipient_id", "eos")
        context_note = str(params.get("context_note", ""))

        self._logger.info(
            "set_reminder_execute",
            recipient_id=recipient_id,
            remind_at=remind_at,
            execution_id=context.execution_id,
        )

        reminder_id = new_id()
        reminder_data = {
            "id": reminder_id,
            "message": message,
            "remind_at": remind_at,
            "recipient_id": recipient_id,
            "context_note": context_note,
            "execution_id": context.execution_id,
        }

        if self._redis is not None:
            try:
                ts = datetime.fromisoformat(remind_at).timestamp()
                key = f"eos:reminders:{recipient_id}"
                await self._redis.zadd(key, {_json.dumps(reminder_data): ts})
                self._reminders[context.execution_id] = reminder_id
                return ExecutionResult(
                    success=True,
                    data={"reminder_id": reminder_id, "remind_at": remind_at},
                    side_effects=[f"Reminder set for {recipient_id} at {remind_at}"],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Reminder set failed: {exc}",
                )
        else:
            self._reminders[context.execution_id] = reminder_id
            return ExecutionResult(
                success=True,
                data={"reminder_id": reminder_id, "note": "No Redis client - reminder staged"},
                side_effects=[f"Reminder staged for {recipient_id} at {remind_at}"],
            )

    async def rollback(
        self,
        execution_id: str,
        context: ExecutionContext,
    ) -> RollbackResult:
        reminder_id = self._reminders.get(execution_id)
        if not reminder_id:
            return RollbackResult(success=False, reason="No reminder ID for this execution")

        if self._redis is not None:
            try:
                # Scan all recipient reminder keys and remove the matching entry
                import json as _json

                recipient_id = "eos"  # default; scan broadly
                # Try to find and remove by scanning the sorted set
                for key_suffix in [recipient_id, "*"]:
                    pattern = f"eos:reminders:{key_suffix}"
                    keys = await self._redis.keys(pattern)
                    for key in keys:
                        result = await self._redis.zrangebyscore(key, "-inf", "+inf")
                        for item in result:
                            try:
                                data = _json.loads(item)
                                if data.get("id") == reminder_id:
                                    await self._redis.zrem(key, item)
                                    del self._reminders[execution_id]
                                    return RollbackResult(
                                        success=True,
                                        side_effects_reversed=[f"Cancelled reminder {reminder_id}"],
                                    )
                            except Exception:
                                continue
            except Exception as exc:
                return RollbackResult(success=False, reason=f"Rollback failed: {exc}")

        del self._reminders[execution_id]
        return RollbackResult(
            success=True,
            side_effects_reversed=[f"Cancelled reminder {reminder_id}"],
        )
