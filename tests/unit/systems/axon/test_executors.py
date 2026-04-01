"""
Unit tests for Axon built-in executors.

Tests validate_params and execute() for each executor category.
All tests use mocked dependencies - no external services required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from primitives.common import Verdict
from primitives.constitutional import ConstitutionalCheck
from primitives.intent import (
    Action,
    ActionSequence,
    EthicalClearance,
    GoalDescriptor,
    Intent,
)
from systems.axon.executors.communication import (
    NotificationExecutor,
    RespondTextExecutor,
)
from systems.axon.executors.data import (
    CreateRecordExecutor,
    ScheduleExecutor,
)
from systems.axon.executors.internal import (
    ConsolidationExecutor,
    StoreInsightExecutor,
    UpdateGoalExecutor,
)
from systems.axon.executors.observation import (
    ObserveExecutor,
    QueryMemoryExecutor,
)
from systems.axon.types import ExecutionContext, ScopedCredentials

# ─── Fixtures ─────────────────────────────────────────────────────


def make_intent(executor_name: str = "observe", params: dict | None = None) -> Intent:
    return Intent(
        goal=GoalDescriptor(description="Test goal"),
        plan=ActionSequence(
            steps=[Action(executor=executor_name, parameters=params or {})]
        ),
        ethical_clearance=EthicalClearance(status=Verdict.APPROVED),
    )


def make_context(intent: Intent | None = None) -> ExecutionContext:
    intent = intent or make_intent()
    check = ConstitutionalCheck(
        intent_id=intent.id,
        verdict=Verdict.APPROVED,
    )
    return ExecutionContext(
        intent=intent,
        equor_check=check,
        credentials=ScopedCredentials(),
    )


# ─── Tests: ObserveExecutor ───────────────────────────────────────


class TestObserveExecutor:
    @pytest.mark.asyncio
    async def test_validate_requires_content(self):
        executor = ObserveExecutor()
        result = await executor.validate_params({})
        assert result.valid is False
        assert "content" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_validate_rejects_too_long_content(self):
        executor = ObserveExecutor()
        result = await executor.validate_params({"content": "x" * 10_001})
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_ok(self):
        executor = ObserveExecutor()
        result = await executor.validate_params({"content": "Hello world"})
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_execute_without_memory_succeeds(self):
        executor = ObserveExecutor(memory=None)
        context = make_context()
        result = await executor.execute({"content": "Test observation"}, context)
        assert result.success is True
        assert len(result.side_effects) > 0

    @pytest.mark.asyncio
    async def test_execute_with_memory_stores(self):
        mock_memory = AsyncMock()
        mock_memory.store_percept = AsyncMock(return_value="ep-123")
        executor = ObserveExecutor(memory=mock_memory)
        context = make_context()
        result = await executor.execute(
            {"content": "Test observation", "salience": 0.8},
            context,
        )
        assert result.success is True
        assert result.data["episode_id"] == "ep-123"
        mock_memory.store_percept.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_handles_memory_failure(self):
        mock_memory = AsyncMock()
        mock_memory.store_percept = AsyncMock(side_effect=RuntimeError("DB down"))
        executor = ObserveExecutor(memory=mock_memory)
        context = make_context()
        result = await executor.execute({"content": "Test"}, context)
        assert result.success is False
        assert "DB down" in result.error


# ─── Tests: QueryMemoryExecutor ───────────────────────────────────


class TestQueryMemoryExecutor:
    @pytest.mark.asyncio
    async def test_validate_requires_query(self):
        executor = QueryMemoryExecutor()
        result = await executor.validate_params({})
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_rejects_bad_max_results(self):
        executor = QueryMemoryExecutor()
        result = await executor.validate_params({"query": "test", "max_results": 0})
        assert result.valid is False
        result = await executor.validate_params({"query": "test", "max_results": 51})
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_execute_without_memory_returns_empty(self):
        executor = QueryMemoryExecutor(memory=None)
        context = make_context()
        result = await executor.execute({"query": "test", "max_results": 3}, context)
        assert result.success is True
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_execute_returns_observations(self):
        mock_trace = MagicMock()
        mock_trace.content = "Some memory content"
        mock_trace.salience = 0.7
        mock_trace.node_id = "mem-1"

        mock_response = MagicMock()
        mock_response.traces = [mock_trace]

        mock_memory = AsyncMock()
        mock_memory.retrieve = AsyncMock(return_value=mock_response)
        executor = QueryMemoryExecutor(memory=mock_memory)
        context = make_context()
        result = await executor.execute({"query": "test"}, context)
        assert result.success is True
        assert result.data["count"] == 1
        assert len(result.new_observations) == 1
        assert "Some memory content" in result.new_observations[0]


# ─── Tests: RespondTextExecutor ───────────────────────────────────


class TestRespondTextExecutor:
    @pytest.mark.asyncio
    async def test_validate_requires_content(self):
        executor = RespondTextExecutor()
        result = await executor.validate_params({})
        assert result.valid is False
        assert "content" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_validate_rejects_bad_urgency(self):
        executor = RespondTextExecutor()
        result = await executor.validate_params({"content": "hi", "urgency": 2.0})
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_execute_without_voxis_succeeds(self):
        executor = RespondTextExecutor(voxis=None)
        context = make_context()
        result = await executor.execute({"content": "Hello!"}, context)
        assert result.success is True
        assert "Hello!" in result.new_observations[0]

    @pytest.mark.asyncio
    async def test_execute_calls_voxis(self):
        mock_voxis = AsyncMock()
        mock_voxis.express = AsyncMock()
        executor = RespondTextExecutor(voxis=mock_voxis)
        context = make_context()
        result = await executor.execute({"content": "Hello!"}, context)
        assert result.success is True
        mock_voxis.express.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_handles_voxis_failure(self):
        mock_voxis = AsyncMock()
        mock_voxis.express = AsyncMock(side_effect=RuntimeError("Voxis error"))
        executor = RespondTextExecutor(voxis=mock_voxis)
        context = make_context()
        result = await executor.execute({"content": "Hello!"}, context)
        assert result.success is False
        assert "Voxis error" in result.error


# ─── Tests: NotificationExecutor ─────────────────────────────────


class TestNotificationExecutor:
    @pytest.mark.asyncio
    async def test_validate_requires_all_fields(self):
        executor = NotificationExecutor()
        for missing in ("recipient_id", "title", "body"):
            params = {"recipient_id": "u1", "title": "T", "body": "B"}
            del params[missing]
            result = await executor.validate_params(params)
            assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_rejects_invalid_urgency(self):
        executor = NotificationExecutor()
        result = await executor.validate_params(
            {"recipient_id": "u1", "title": "T", "body": "B", "urgency": "critical"}
        )
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_execute_without_redis_stages(self):
        executor = NotificationExecutor(redis_client=None)
        context = make_context()
        result = await executor.execute(
            {"recipient_id": "u1", "title": "Test", "body": "Message"},
            context,
        )
        assert result.success is True
        assert result.data["delivered"] is False

    @pytest.mark.asyncio
    async def test_execute_publishes_to_redis(self):
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        executor = NotificationExecutor(redis_client=mock_redis)
        context = make_context()
        result = await executor.execute(
            {"recipient_id": "u1", "title": "Test", "body": "Message"},
            context,
        )
        assert result.success is True
        mock_redis.publish.assert_called_once()


# ─── Tests: CreateRecordExecutor ─────────────────────────────────


class TestCreateRecordExecutor:
    @pytest.mark.asyncio
    async def test_validate_requires_record_type_and_data(self):
        executor = CreateRecordExecutor()
        result = await executor.validate_params({})
        assert result.valid is False
        result = await executor.validate_params({"record_type": "Task"})
        assert result.valid is False
        result = await executor.validate_params(
            {"record_type": "Task", "data": {"title": "Do thing"}}
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_execute_without_memory_returns_id(self):
        executor = CreateRecordExecutor(memory=None)
        context = make_context()
        result = await executor.execute(
            {"record_type": "Task", "data": {"title": "Test"}},
            context,
        )
        assert result.success is True
        assert "record_id" in result.data

    @pytest.mark.asyncio
    async def test_execute_calls_memory_create(self):
        mock_memory = AsyncMock()
        mock_memory.resolve_and_create_entity = AsyncMock(return_value=("rec-999", True))
        executor = CreateRecordExecutor(memory=mock_memory)
        context = make_context()
        result = await executor.execute(
            {"record_type": "Event", "data": {"title": "Meeting"}},
            context,
        )
        assert result.success is True
        assert result.data["record_id"] == "rec-999"

    @pytest.mark.asyncio
    async def test_rollback_deletes_created_record(self):
        mock_memory = AsyncMock()
        mock_memory.resolve_and_create_entity = AsyncMock(return_value=("rec-001", True))
        mock_memory.delete_entity = AsyncMock()
        executor = CreateRecordExecutor(memory=mock_memory)
        context = make_context()
        await executor.execute(
            {"record_type": "Task", "data": {"title": "Test"}},
            context,
        )
        rb_result = await executor.rollback(context.execution_id, context)
        assert rb_result.success is True
        mock_memory.delete_entity.assert_called_once_with("rec-001")


# ─── Tests: ScheduleExecutor ─────────────────────────────────────


class TestScheduleExecutor:
    @pytest.mark.asyncio
    async def test_validate_requires_title_and_scheduled_at(self):
        executor = ScheduleExecutor()
        result = await executor.validate_params({})
        assert result.valid is False
        result = await executor.validate_params(
            {"title": "Meeting", "scheduled_at": "not-a-date"}
        )
        assert result.valid is False
        result = await executor.validate_params(
            {"title": "Meeting", "scheduled_at": "2026-06-01T10:00:00"}
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_execute_without_redis_stages(self):
        executor = ScheduleExecutor(redis_client=None)
        context = make_context()
        result = await executor.execute(
            {"title": "Team Standup", "scheduled_at": "2026-06-01T09:00:00"},
            context,
        )
        assert result.success is True
        assert "event_id" in result.data


# ─── Tests: StoreInsightExecutor ─────────────────────────────────


class TestStoreInsightExecutor:
    @pytest.mark.asyncio
    async def test_validate_requires_insight_and_domain(self):
        executor = StoreInsightExecutor()
        result = await executor.validate_params({})
        assert result.valid is False
        result = await executor.validate_params({"insight": "Something"})
        assert result.valid is False
        result = await executor.validate_params(
            {"insight": "Something", "domain": "community"}
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validate_rejects_bad_confidence(self):
        executor = StoreInsightExecutor()
        result = await executor.validate_params(
            {"insight": "X", "domain": "d", "confidence": 1.5}
        )
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_execute_without_memory_succeeds(self):
        executor = StoreInsightExecutor(memory=None)
        context = make_context()
        result = await executor.execute(
            {"insight": "Community prefers morning meetings", "domain": "community"},
            context,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_calls_memory_resolve_and_create(self):
        mock_memory = AsyncMock()
        # resolve_and_create_entity returns (entity_id, was_created)
        mock_memory.resolve_and_create_entity = AsyncMock(return_value=("ins-007", True))
        executor = StoreInsightExecutor(memory=mock_memory)
        context = make_context()
        result = await executor.execute(
            {
                "insight": "Community prefers morning meetings",
                "domain": "community",
                "confidence": 0.85,
            },
            context,
        )
        assert result.success is True
        assert result.data["insight_id"] == "ins-007"
        mock_memory.resolve_and_create_entity.assert_called_once()


# ─── Tests: UpdateGoalExecutor ────────────────────────────────────


class TestUpdateGoalExecutor:
    @pytest.mark.asyncio
    async def test_validate_requires_goal_id_and_status(self):
        executor = UpdateGoalExecutor()
        result = await executor.validate_params({})
        assert result.valid is False
        result = await executor.validate_params({"goal_id": "g1"})
        assert result.valid is False
        result = await executor.validate_params(
            {"goal_id": "g1", "status": "invalid_status"}
        )
        assert result.valid is False
        result = await executor.validate_params(
            {"goal_id": "g1", "status": "achieved"}
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_execute_without_nova_logs(self):
        executor = UpdateGoalExecutor()
        context = make_context()
        result = await executor.execute(
            {"goal_id": "goal-123", "status": "achieved"},
            context,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_calls_nova_update(self):
        mock_nova = AsyncMock()
        mock_nova.update_goal_status = AsyncMock()
        executor = UpdateGoalExecutor()
        executor.set_nova(mock_nova)
        context = make_context()
        result = await executor.execute(
            {"goal_id": "goal-123", "status": "achieved", "progress": 1.0},
            context,
        )
        assert result.success is True
        mock_nova.update_goal_status.assert_called_once()


# ─── Tests: ConsolidationExecutor ────────────────────────────────


class TestConsolidationExecutor:
    @pytest.mark.asyncio
    async def test_validate_rejects_invalid_scope(self):
        executor = ConsolidationExecutor()
        result = await executor.validate_params({"scope": "everything"})
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_requires_domain_when_domain_scope(self):
        executor = ConsolidationExecutor()
        result = await executor.validate_params({"scope": "domain"})
        assert result.valid is False
        result = await executor.validate_params(
            {"scope": "domain", "domain": "community"}
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_execute_without_memory_succeeds(self):
        executor = ConsolidationExecutor(memory=None)
        context = make_context()
        result = await executor.execute({"scope": "recent"}, context)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_calls_memory_consolidation(self):
        mock_memory = AsyncMock()
        mock_memory.consolidate = AsyncMock(
            return_value={"episodes_processed": 50, "insights_created": 5}
        )
        executor = ConsolidationExecutor(memory=mock_memory)
        context = make_context()
        result = await executor.execute({"scope": "recent"}, context)
        assert result.success is True
        assert result.data["episodes_processed"] == 50
        assert result.data["insights_created"] == 5
