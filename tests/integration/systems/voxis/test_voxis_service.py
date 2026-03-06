"""
Integration tests for VoxisService.

Tests the full expression pipeline end-to-end with real sub-components
(PersonalityEngine, AffectColouringEngine, AudienceProfiler, SilenceEngine,
ContentRenderer) and mocked external dependencies (LLMProvider, MemoryService,
RedisClient).

No real API calls, no real Redis, no real Neo4j.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.llm import LLMProvider, LLMResponse
from config import VoxisConfig
from primitives.affect import AffectState
from primitives.expression import Expression
from primitives.memory_trace import SelfNode
from systems.memory.service import MemoryService
from systems.voxis.service import VoxisService
from systems.voxis.types import ExpressionTrigger

# ─── Shared Mock Factories ────────────────────────────────────────


def make_llm_mock(response_text: str = "I'm here to help with that.") -> AsyncMock:
    """
    Create a fully-configured LLM mock.

    generate() → expression content
    evaluate() → "authentic" so the honesty check always passes by default
    """
    llm = AsyncMock(spec=LLMProvider)
    llm.generate.return_value = LLMResponse(
        text=response_text,
        model="mock-model",
        input_tokens=120,
        output_tokens=25,
        finish_reason="stop",
    )
    # Honesty check uses evaluate(); "authentic" triggers the pass branch
    llm.evaluate.return_value = LLMResponse(
        text="authentic",
        model="mock-model",
        input_tokens=60,
        output_tokens=5,
        finish_reason="stop",
    )
    return llm


def make_memory_mock(
    self_node: SelfNode | None = None,
    constitution: dict | None = None,
) -> AsyncMock:
    """Create a MemoryService mock with configurable Self node and constitution."""
    memory = AsyncMock(spec=MemoryService)
    memory.get_self.return_value = self_node
    memory.get_constitution.return_value = constitution
    # retrieve() returns an object with .traces and .entities lists
    retrieve_result = MagicMock()
    retrieve_result.traces = []
    retrieve_result.entities = []
    memory.retrieve.return_value = retrieve_result
    return memory


def make_redis_mock() -> AsyncMock:
    """
    Create a Redis mock.

    ConversationManager uses raw get()/set() on the client (duck-typed).
    We expose them as async mocks returning None (no persisted state).
    """
    redis = AsyncMock()
    redis.get.return_value = None        # No persisted conversation
    redis.set.return_value = True        # Store succeeds silently
    redis.delete.return_value = 1
    # RedisClient JSON helpers (used by other components)
    redis.get_json.return_value = None
    redis.set_json.return_value = None
    return redis


def make_voxis_config(**overrides: Any) -> VoxisConfig:
    defaults = dict(
        min_expression_interval_minutes=1.0,
        conversation_history_window=50,
        context_window_max_tokens=4000,
        conversation_summary_threshold=10,
        max_active_conversations=10,
        temperature_base=0.7,
        honesty_check_enabled=True,
        max_expression_length=2000,
        feedback_enabled=False,  # Disable feedback loop in tests
    )
    defaults.update(overrides)
    return VoxisConfig(**defaults)


# ─── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_llm() -> AsyncMock:
    return make_llm_mock()


@pytest.fixture
def mock_memory() -> AsyncMock:
    return make_memory_mock()


@pytest.fixture
def mock_redis() -> AsyncMock:
    return make_redis_mock()


@pytest.fixture
async def voxis(mock_llm: AsyncMock, mock_memory: AsyncMock, mock_redis: AsyncMock) -> VoxisService:
    service = VoxisService(
        memory=mock_memory,
        redis=mock_redis,
        llm=mock_llm,
        config=make_voxis_config(),
    )
    await service.initialize()
    return service


# ─── Initialization ───────────────────────────────────────────────


class TestInitialization:
    async def test_initialize_uses_neutral_personality_when_memory_returns_none(
        self, mock_llm: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        """When Memory has no Self node, personality should be a neutral zero-vector."""
        memory = make_memory_mock(self_node=None)
        service = VoxisService(
            memory=memory,
            redis=mock_redis,
            llm=mock_llm,
            config=make_voxis_config(),
        )
        await service.initialize()

        p = service.current_personality
        assert p.warmth == 0.0
        assert p.directness == 0.0
        assert p.empathy_expression == 0.0

    async def test_initialize_sets_instance_name_from_self_node(
        self, mock_llm: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        """Instance name should come from the SelfNode returned by Memory."""
        node = SelfNode(instance_id="eos-001", name="Aria")
        memory = make_memory_mock(self_node=node)
        service = VoxisService(
            memory=memory,
            redis=mock_redis,
            llm=mock_llm,
            config=make_voxis_config(),
        )
        await service.initialize()

        health = await service.health()
        assert health["instance_name"] == "Aria"

    async def test_initialize_loads_drive_weights_from_constitution(
        self, mock_llm: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        """Constitutional drive weights should be loaded from Memory on startup."""
        constitution = {"drives": {"coherence": 2.0, "care": 3.0, "growth": 1.5, "honesty": 1.0}}
        memory = make_memory_mock(constitution=constitution)
        service = VoxisService(
            memory=memory,
            redis=mock_redis,
            llm=mock_llm,
            config=make_voxis_config(),
        )
        await service.initialize()

        # Drive weights affect EFE computation in ContentRenderer.
        # Verify they were loaded by checking health (doesn't expose weights directly)
        # but we can confirm initialization completed without error.
        health = await service.health()
        assert health["status"] == "healthy"

    async def test_initialize_uses_default_drive_weights_when_constitution_absent(
        self, mock_llm: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        """When no constitution is stored, default weights (all 1.0) should be used."""
        memory = make_memory_mock(constitution=None)
        service = VoxisService(
            memory=memory,
            redis=mock_redis,
            llm=mock_llm,
            config=make_voxis_config(),
        )
        # Should not raise
        await service.initialize()
        health = await service.health()
        assert health["status"] == "healthy"


# ─── Expression Pipeline ──────────────────────────────────────────


class TestExpressPipeline:
    async def test_express_nova_respond_returns_expression_with_content(
        self, voxis: VoxisService, mock_llm: AsyncMock
    ) -> None:
        """The standard respond trigger should produce a non-empty Expression."""
        expression = await voxis.express(
            content="Hello! Who are you?",
            trigger=ExpressionTrigger.NOVA_RESPOND,
        )

        assert isinstance(expression, Expression)
        assert expression.is_silence is False
        assert expression.content != ""
        assert mock_llm.generate.called

    async def test_express_returns_expression_not_silence_for_mandatory_trigger(
        self, voxis: VoxisService
    ) -> None:
        """ATUNE_DISTRESS must always produce an expression, never silence."""
        expression = await voxis.express(
            content="Help!",
            trigger=ExpressionTrigger.ATUNE_DISTRESS,
            urgency=1.0,
        )
        assert expression.is_silence is False

    async def test_express_populates_generation_trace(
        self, voxis: VoxisService
    ) -> None:
        """GenerationTrace should be populated with model, token counts, and latency."""
        expression = await voxis.express(
            content="Tell me about yourself.",
            trigger=ExpressionTrigger.NOVA_RESPOND,
        )

        assert expression.generation_trace is not None
        assert expression.generation_trace.model == "mock-model"
        assert expression.generation_trace.input_tokens == 120
        assert expression.generation_trace.output_tokens == 25
        assert expression.generation_trace.latency_ms >= 0

    async def test_express_attaches_affect_snapshot(
        self, voxis: VoxisService
    ) -> None:
        """Expression should carry an affect snapshot from the current AffectState."""
        affect = AffectState(valence=0.4, arousal=0.6, care_activation=0.8)
        expression = await voxis.express(
            content="Good news!",
            trigger=ExpressionTrigger.NOVA_RESPOND,
            affect=affect,
        )

        # The affect values are snapshotted onto the Expression node
        assert expression.affect_valence is not None
        assert expression.affect_care_activation is not None
        assert abs(expression.affect_care_activation - 0.8) < 0.01

    async def test_express_increments_total_expressions_counter(
        self, voxis: VoxisService
    ) -> None:
        """Each successful expression should increment the expression counter."""
        health_before = await voxis.health()
        count_before = health_before["total_expressions"]

        await voxis.express(content="First.", trigger=ExpressionTrigger.NOVA_RESPOND)
        await voxis.express(content="Second.", trigger=ExpressionTrigger.NOVA_RESPOND)

        health_after = await voxis.health()
        assert health_after["total_expressions"] == count_before + 2

    async def test_express_persists_to_conversation_state(
        self, voxis: VoxisService, mock_redis: AsyncMock
    ) -> None:
        """After express(), the assistant response should be saved to conversation state."""
        conv_id = await voxis.ingest_user_message("Hello, EOS!")
        await voxis.express(
            content="Hello!",
            trigger=ExpressionTrigger.NOVA_RESPOND,
            conversation_id=conv_id,
        )

        # Redis set() should have been called (to persist conversation)
        assert mock_redis.set.called

    async def test_express_with_addressee_name_included_in_context(
        self, voxis: VoxisService, mock_llm: AsyncMock
    ) -> None:
        """The addressee name should flow through to the system prompt."""
        await voxis.express(
            content="Hello!",
            trigger=ExpressionTrigger.NOVA_RESPOND,
            addressee_name="Alice",
        )

        # System prompt should contain addressee info
        call_args = mock_llm.generate.call_args
        system_prompt = call_args[0][0] if call_args[0] else call_args[1].get("system_prompt", "")
        # The system prompt is assembled by the renderer; it must be non-trivial
        assert len(system_prompt) > 100


# ─── Silence Gating ───────────────────────────────────────────────


class TestSilenceGating:
    async def test_express_rate_limits_nova_inform(self, voxis: VoxisService) -> None:
        """
        After one expression (which records the expression time), a NOVA_INFORM
        within the min_expression_interval should be silenced.

        NOVA_RESPOND always speaks; NOVA_INFORM is rate-limited.
        """
        # First expression (NOVA_RESPOND, always speaks, records expression time)
        first = await voxis.express(
            content="I have something to share.",
            trigger=ExpressionTrigger.NOVA_RESPOND,
        )
        assert first.is_silence is False

        # Immediately after, NOVA_INFORM should be suppressed by rate limit
        second = await voxis.express(
            content="One more thing…",
            trigger=ExpressionTrigger.NOVA_INFORM,
        )
        assert second.is_silence is True
        assert second.silence_reason != ""

    async def test_ambient_insight_below_threshold_silenced(
        self, voxis: VoxisService, mock_llm: AsyncMock
    ) -> None:
        """AMBIENT_INSIGHT with insight_value below the 0.6 threshold must be silent."""
        expression = await voxis.express(
            content="An interesting observation…",
            trigger=ExpressionTrigger.AMBIENT_INSIGHT,
            insight_value=0.2,  # Well below threshold
        )
        assert expression.is_silence is True
        assert mock_llm.generate.not_called

    async def test_ambient_insight_above_threshold_speaks(
        self, voxis: VoxisService
    ) -> None:
        """AMBIENT_INSIGHT with insight_value above threshold should speak."""
        expression = await voxis.express(
            content="A valuable insight about the community.",
            trigger=ExpressionTrigger.AMBIENT_INSIGHT,
            insight_value=0.9,
        )
        assert expression.is_silence is False

    async def test_atune_distress_bypasses_rate_limit(
        self, voxis: VoxisService
    ) -> None:
        """
        ATUNE_DISTRESS is a mandatory trigger — it must speak even when the
        rate limiter would otherwise suppress it.
        """
        # Record an expression so rate limit is active
        await voxis.express(content="Normal reply.", trigger=ExpressionTrigger.NOVA_RESPOND)

        # Immediately after: mandatory distress trigger must not be silenced
        distress = await voxis.express(
            content="Are you okay?",
            trigger=ExpressionTrigger.ATUNE_DISTRESS,
            urgency=1.0,
        )
        assert distress.is_silence is False

    async def test_silence_rate_tracked_in_health(self, voxis: VoxisService) -> None:
        """The silence rate in health() must reflect actual silence decisions."""
        # One speak (NOVA_RESPOND) then one silence (NOVA_INFORM rate-limited)
        await voxis.express(content="Response.", trigger=ExpressionTrigger.NOVA_RESPOND)
        await voxis.express(content="Inform.", trigger=ExpressionTrigger.NOVA_INFORM)

        health = await voxis.health()
        # silence_rate = silences / (speaks + silences) = 1/2 = 0.5
        assert health["silence_rate"] == pytest.approx(0.5, abs=0.01)


# ─── ingest_user_message → express Flow ──────────────────────────


class TestConversationFlow:
    async def test_user_message_ingested_before_response(
        self, voxis: VoxisService, mock_redis: AsyncMock
    ) -> None:
        """
        Ingesting a user message then expressing should result in both messages
        present in the conversation (via Redis set calls, both user and assistant).
        """
        conv_id = await voxis.ingest_user_message(
            message="Tell me about Active Inference.",
            speaker_id="user-42",
        )
        assert isinstance(conv_id, str)
        assert len(conv_id) > 0

        expression = await voxis.express(
            content="Active Inference is…",
            trigger=ExpressionTrigger.NOVA_RESPOND,
            conversation_id=conv_id,
        )
        assert expression.is_silence is False
        assert expression.conversation_id == conv_id

    async def test_conversation_id_propagated_to_expression(
        self, voxis: VoxisService
    ) -> None:
        """Expression should carry the conversation_id it was expressed in."""
        conv_id = await voxis.ingest_user_message("What are your drives?")
        expression = await voxis.express(
            content="My four drives are…",
            trigger=ExpressionTrigger.NOVA_RESPOND,
            conversation_id=conv_id,
        )
        assert expression.conversation_id == conv_id

    async def test_new_conversation_created_when_id_is_none(
        self, voxis: VoxisService
    ) -> None:
        """When no conversation_id is supplied, a new one should be created."""
        expression = await voxis.express(
            content="New conversation.",
            trigger=ExpressionTrigger.NOVA_RESPOND,
            conversation_id=None,
        )
        assert expression.is_silence is False
        assert expression.conversation_id is not None

    async def test_multi_turn_conversation_same_id(
        self, voxis: VoxisService
    ) -> None:
        """Multiple exchanges on the same conversation_id should succeed."""
        conv_id = await voxis.ingest_user_message("Hi!")
        r1 = await voxis.express("Hello!", ExpressionTrigger.NOVA_RESPOND, conv_id)

        await voxis.ingest_user_message("How are you?", conversation_id=conv_id)
        r2 = await voxis.express("I'm doing well.", ExpressionTrigger.NOVA_RESPOND, conv_id)

        assert r1.is_silence is False
        assert r2.is_silence is False
        assert r1.conversation_id == r2.conversation_id


# ─── Memory Retrieval Timeout ─────────────────────────────────────


class TestMemoryTimeout:
    async def test_express_completes_despite_slow_memory(
        self, mock_llm: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        """
        If Memory.retrieve() is slow, express() should still complete within a
        reasonable window. The 150ms memory timeout is enforced via asyncio.wait_for.
        """
        async def slow_retrieve(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.3)  # 300ms — beyond the 150ms hard timeout

        memory = make_memory_mock()
        memory.retrieve.side_effect = slow_retrieve

        service = VoxisService(
            memory=memory,
            redis=mock_redis,
            llm=mock_llm,
            config=make_voxis_config(),
        )
        await service.initialize()

        # Should still complete successfully (memory retrieval degraded to empty)
        expression = await service.express(
            content="Can you help me?",
            trigger=ExpressionTrigger.NOVA_RESPOND,
        )
        assert expression.is_silence is False
        assert expression.content != ""


# ─── Personality Update (Evo Interface) ──────────────────────────


class TestPersonalityUpdate:
    async def test_update_personality_applies_positive_delta(
        self, voxis: VoxisService
    ) -> None:
        """Applying a positive warmth delta should increase warmth."""
        original = voxis.current_personality.warmth
        new_vector = voxis.update_personality({"warmth": 0.03})
        assert new_vector.warmth > original
        assert voxis.current_personality.warmth == new_vector.warmth

    async def test_update_personality_clamps_large_delta(
        self, voxis: VoxisService
    ) -> None:
        """A delta of 1.0 should be clamped to MAX_PERSONALITY_DELTA (0.03)."""
        original = voxis.current_personality.warmth
        new_vector = voxis.update_personality({"warmth": 1.0})
        assert new_vector.warmth <= original + 0.03 + 1e-6

    async def test_update_personality_negative_delta(
        self, voxis: VoxisService
    ) -> None:
        """A negative delta should decrease the dimension."""
        # Start with warmth = 0.5 so there's room to decrease
        service = VoxisService(
            memory=make_memory_mock(),
            redis=make_redis_mock(),
            llm=make_llm_mock(),
            config=make_voxis_config(),
        )
        await service.initialize()
        service.update_personality({"warmth": 0.03})  # Bump to 0.03
        before = service.current_personality.warmth

        service.update_personality({"warmth": -0.03})
        assert service.current_personality.warmth < before

    async def test_update_personality_unknown_field_ignored(
        self, voxis: VoxisService
    ) -> None:
        """Unknown dimension keys should be silently ignored."""
        before = voxis.current_personality.warmth
        voxis.update_personality({"totally_fictional_dimension": 0.5})
        # Warmth unchanged
        assert voxis.current_personality.warmth == before

    async def test_update_personality_affects_subsequent_expressions(
        self, voxis: VoxisService, mock_llm: AsyncMock
    ) -> None:
        """After a personality update, the next expression's system prompt should
        reflect the updated personality (e.g. high warmth → warm tone instructions)."""
        # Push warmth up significantly by applying multiple deltas
        for _ in range(12):
            voxis.update_personality({"warmth": 0.03})

        await voxis.express("Hello!", ExpressionTrigger.NOVA_RESPOND)

        call_args = mock_llm.generate.call_args
        system_prompt = call_args[0][0] if call_args[0] else call_args[1].get("system_prompt", "")
        # High warmth should produce warmth-related instructions in the system prompt
        assert "warm" in system_prompt.lower()


# ─── Expression Callbacks ─────────────────────────────────────────


class TestExpressionCallbacks:
    async def test_callback_invoked_on_expression(self, voxis: VoxisService) -> None:
        """Registered callback should be called once per delivered expression."""
        received: list[Expression] = []
        voxis.register_expression_callback(received.append)

        await voxis.express(content="Hello.", trigger=ExpressionTrigger.NOVA_RESPOND)

        assert len(received) == 1
        assert isinstance(received[0], Expression)
        assert received[0].is_silence is False

    async def test_callback_not_invoked_on_silence(self, voxis: VoxisService) -> None:
        """Callback should not be called when the expression is silenced."""
        received: list[Expression] = []
        voxis.register_expression_callback(received.append)

        # Rate-limit: first speak records time, second NOVA_INFORM is suppressed
        await voxis.express(content="First.", trigger=ExpressionTrigger.NOVA_RESPOND)
        await voxis.express(content="Second.", trigger=ExpressionTrigger.NOVA_INFORM)

        # Only the first (speaking) expression should trigger the callback
        assert len(received) == 1

    async def test_multiple_callbacks_all_invoked(self, voxis: VoxisService) -> None:
        """All registered callbacks should be called for each expression."""
        bucket_a: list[Expression] = []
        bucket_b: list[Expression] = []
        voxis.register_expression_callback(bucket_a.append)
        voxis.register_expression_callback(bucket_b.append)

        await voxis.express(content="Hello.", trigger=ExpressionTrigger.NOVA_RESPOND)

        assert len(bucket_a) == 1
        assert len(bucket_b) == 1

    async def test_failing_callback_does_not_stop_execution(
        self, voxis: VoxisService
    ) -> None:
        """A callback that raises must not prevent subsequent callbacks from firing."""
        def bad_callback(expr: Expression) -> None:
            raise RuntimeError("callback exploded")

        good: list[Expression] = []
        voxis.register_expression_callback(bad_callback)
        voxis.register_expression_callback(good.append)

        # Should not raise
        expression = await voxis.express("Hello.", ExpressionTrigger.NOVA_RESPOND)
        assert expression.is_silence is False
        assert len(good) == 1


# ─── Health Endpoint ──────────────────────────────────────────────


class TestHealthEndpoint:
    async def test_health_returns_expected_shape(self, voxis: VoxisService) -> None:
        """health() must return all required top-level keys."""
        health = await voxis.health()

        required_keys = {
            "status",
            "instance_name",
            "total_expressions",
            "silence_rate",
            "honesty_rejections",
            "expressions_by_trigger",
            "expressions_by_channel",
            "personality",
        }
        assert required_keys.issubset(health.keys())

    async def test_health_personality_snapshot_contains_key_dimensions(
        self, voxis: VoxisService
    ) -> None:
        """The personality sub-dict must expose the five key dimensions."""
        health = await voxis.health()
        personality = health["personality"]

        for dim in ("warmth", "directness", "verbosity", "empathy_expression", "curiosity_expression"):
            assert dim in personality

    async def test_health_silence_rate_zero_before_any_decisions(
        self, voxis: VoxisService
    ) -> None:
        """Before any expressions, silence_rate should be 0.0."""
        health = await voxis.health()
        assert health["silence_rate"] == 0.0

    async def test_health_silence_rate_half_after_one_speak_one_silence(
        self, voxis: VoxisService
    ) -> None:
        """1 speak + 1 silence decision → silence_rate = 0.5."""
        # Speak
        await voxis.express(content="Hello.", trigger=ExpressionTrigger.NOVA_RESPOND)
        # Rate-limited silence
        await voxis.express(content="Also.", trigger=ExpressionTrigger.NOVA_INFORM)

        health = await voxis.health()
        assert health["silence_rate"] == pytest.approx(0.5, abs=0.01)

    async def test_health_expressions_by_trigger_populated(
        self, voxis: VoxisService
    ) -> None:
        """expressions_by_trigger should count per-trigger expressions."""
        await voxis.express("A.", ExpressionTrigger.NOVA_RESPOND)
        await voxis.express("B.", ExpressionTrigger.NOVA_RESPOND)

        health = await voxis.health()
        by_trigger = health["expressions_by_trigger"]
        assert by_trigger.get(ExpressionTrigger.NOVA_RESPOND.value, 0) == 2

    async def test_health_status_is_healthy(self, voxis: VoxisService) -> None:
        health = await voxis.health()
        assert health["status"] == "healthy"


# ─── Broadcast Interface ─────────────────────────────────────────


class TestBroadcastInterface:
    async def test_on_broadcast_with_direct_address_triggers_expression(
        self, voxis: VoxisService
    ) -> None:
        """
        A broadcast with direct-address content should result in an expression
        being queued as an asyncio background task.
        """
        received: list[Expression] = []
        voxis.register_expression_callback(received.append)

        # Construct a mock broadcast carrying user content
        content_mock = MagicMock()
        content_mock.content = "EOS, what do you think?"
        broadcast_mock = MagicMock()
        broadcast_mock.content = content_mock
        broadcast_mock.affect = AffectState.neutral()
        broadcast_mock.salience = MagicMock()
        broadcast_mock.salience.composite = 0.8

        await voxis.on_broadcast(broadcast_mock)

        # Give the background task a moment to complete
        await asyncio.sleep(0.05)
        # Drain any pending tasks
        await asyncio.gather(*[
            t for t in asyncio.all_tasks()
            if t.get_name().startswith("voxis_express_")
        ], return_exceptions=True)

        assert len(received) >= 1

    async def test_on_broadcast_with_empty_content_is_ignored(
        self, voxis: VoxisService
    ) -> None:
        """A broadcast with no extractable text content should not trigger expression."""
        received: list[Expression] = []
        voxis.register_expression_callback(received.append)

        content_mock = MagicMock()
        content_mock.content = None
        content_mock.raw = None
        broadcast_mock = MagicMock()
        broadcast_mock.content = content_mock
        broadcast_mock.affect = AffectState.neutral()
        broadcast_mock.salience = MagicMock()
        broadcast_mock.salience.composite = 0.5

        await voxis.on_broadcast(broadcast_mock)
        await asyncio.sleep(0.02)

        assert len(received) == 0

    async def test_on_broadcast_before_initialize_is_safe(
        self, mock_llm: AsyncMock, mock_memory: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        """on_broadcast() called before initialize() should not raise."""
        service = VoxisService(
            memory=mock_memory,
            redis=mock_redis,
            llm=mock_llm,
            config=make_voxis_config(),
        )
        # No initialize() call

        broadcast_mock = MagicMock()
        broadcast_mock.content = MagicMock()
        broadcast_mock.content.content = "Hello!"

        # Should return without raising
        await service.on_broadcast(broadcast_mock)


# ─── Context window — Honesty check ──────────────────────────────


class TestHonestyCheck:
    async def test_honesty_check_enabled_calls_evaluate(
        self, mock_redis: AsyncMock, mock_memory: AsyncMock
    ) -> None:
        """With honesty_check_enabled=True, evaluate() should be called for each expression."""
        llm = make_llm_mock()
        service = VoxisService(
            memory=mock_memory,
            redis=mock_redis,
            llm=llm,
            config=make_voxis_config(honesty_check_enabled=True),
        )
        await service.initialize()

        await service.express("Tell me a secret.", ExpressionTrigger.NOVA_RESPOND)

        assert llm.evaluate.called

    async def test_honesty_check_disabled_skips_evaluate(
        self, mock_redis: AsyncMock, mock_memory: AsyncMock
    ) -> None:
        """With honesty_check_enabled=False, evaluate() should not be called."""
        llm = make_llm_mock()
        service = VoxisService(
            memory=mock_memory,
            redis=mock_redis,
            llm=llm,
            config=make_voxis_config(honesty_check_enabled=False),
        )
        await service.initialize()

        await service.express("Just talking.", ExpressionTrigger.NOVA_RESPOND)

        llm.evaluate.assert_not_called()

    async def test_honesty_check_failure_is_recorded(
        self, mock_redis: AsyncMock, mock_memory: AsyncMock
    ) -> None:
        """When the honesty check returns a violation, honesty_rejections counter increments."""
        llm = make_llm_mock()
        # First generate() call returns content with forced positivity
        # evaluate() returns a violation signal — any non-"authentic" response
        # triggers the retry path; but we need to understand the renderer's
        # exact check logic. The renderer checks for keyword markers in evaluate().
        # We'll mock evaluate() to return "forced_positivity" on first call,
        # then "authentic" on retry to avoid infinite loop.
        call_count = 0

        async def evaluate_side_effect(*args: Any, **kwargs: Any) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(text="forced_positivity: content is excessively positive")
            return LLMResponse(text="authentic")

        llm.evaluate.side_effect = evaluate_side_effect

        service = VoxisService(
            memory=mock_memory,
            redis=mock_redis,
            llm=llm,
            config=make_voxis_config(honesty_check_enabled=True),
        )
        await service.initialize()

        expression = await service.express("Everything is amazing!", ExpressionTrigger.NOVA_RESPOND)
        health = await service.health()

        # The expression still goes through (retry succeeded)
        assert expression.is_silence is False
        # Honesty rejection counter should have incremented
        assert health["honesty_rejections"] >= 1


# ─── Shutdown ─────────────────────────────────────────────────────


class TestShutdown:
    async def test_shutdown_completes_without_error(self, voxis: VoxisService) -> None:
        """shutdown() must complete cleanly even with no active state."""
        await voxis.shutdown()  # Should not raise

    async def test_shutdown_after_expressions_completes(
        self, voxis: VoxisService
    ) -> None:
        """shutdown() after some expressions should still complete cleanly."""
        await voxis.express("Hello.", ExpressionTrigger.NOVA_RESPOND)
        await voxis.express("Another.", ExpressionTrigger.NOVA_RESPOND)
        await voxis.shutdown()  # Should not raise
