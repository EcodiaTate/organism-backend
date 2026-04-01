"""
EcodiaOS - Voxis Conversation Manager

Maintains short-term conversational context for coherent multi-turn dialogue.
Backed by Redis with 24-hour TTL.

The conversation state is not just a message buffer - it tracks the emotional
arc of the exchange, active topics, unresolved questions, and belief-level
context about the conversational partner's state. This context feeds directly
into audience profiling and expression strategy selection.

Context window management uses rolling LLM summarisation for older messages,
keeping the most recent N verbatim while compressing prior history. This is
a deliberate trade-off: perfect recall vs. practical context budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from clients.optimized_llm import OptimizedLLMProvider
from primitives.common import new_id, utc_now
from prompts.voxis.conversation import (
    build_summarise_segment_prompt,
    build_topic_extraction_prompt,
)
from systems.voxis.types import ConversationMessage, ConversationState

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.redis import RedisClient

logger = structlog.get_logger()

# Redis TTL for conversation state (24 hours)
_CONVERSATION_TTL_SECONDS = 86_400


class ConversationManager:
    """
    Manages conversation state across multi-turn dialogues.

    Responsibilities:
    - Create and retrieve ConversationState from Redis
    - Append messages and update topic/emotion tracking
    - Manage context window (recent verbatim + older summary)
    - Extract active topics via LLM call (async, non-blocking)
    - Track emotional arc for feedback loop

    Redis key pattern: {prefix}:voxis:conv:{conversation_id}
    """

    def __init__(
        self,
        redis: RedisClient,
        llm: LLMProvider,
        history_window: int = 50,
        context_window_max_tokens: int = 4000,
        summary_threshold: int = 10,
        max_active_conversations: int = 50,
    ) -> None:
        self._redis = redis
        self._llm = llm
        self._history_window = history_window
        self._context_window_max_tokens = context_window_max_tokens
        self._summary_threshold = summary_threshold
        self._max_active = max_active_conversations
        self._logger = logger.bind(system="voxis.conversation")
        self._optimized = isinstance(llm, OptimizedLLMProvider)
        # In-memory cache for hot conversations (avoids Redis roundtrip per message)
        self._cache: dict[str, ConversationState] = {}

    # ─── Public API ───────────────────────────────────────────────

    async def get_or_create(
        self,
        conversation_id: str | None,
        participant_ids: list[str] | None = None,
    ) -> ConversationState:
        """
        Retrieve an existing conversation or create a new one.
        Falls back to Redis if not in memory cache.
        """
        cid = conversation_id or new_id()

        if cid in self._cache:
            return self._cache[cid]

        stored = await self._load_from_redis(cid)
        if stored is not None:
            self._cache[cid] = stored
            return stored

        # New conversation
        state = ConversationState(
            conversation_id=cid,
            participant_ids=participant_ids or [],
        )
        self._cache[cid] = state
        await self._save_to_redis(state)
        self._logger.info("conversation_created", conversation_id=cid)
        return state

    async def append_message(
        self,
        state: ConversationState,
        role: str,
        content: str,
        speaker_id: str | None = None,
        affect_valence: float | None = None,
    ) -> ConversationState:
        """
        Append a message to the conversation and persist.

        If history exceeds the window, the oldest messages are rolled off
        (summarisation happens lazily in prepare_context when needed).
        """
        msg = ConversationMessage(
            role=role,
            content=content,
            speaker_id=speaker_id,
            affect_valence=affect_valence,
        )

        updated_messages = list(state.messages) + [msg]
        # Keep at most history_window messages in memory
        if len(updated_messages) > self._history_window:
            updated_messages = updated_messages[-self._history_window:]

        # Update emotional arc if affect estimate provided
        emotional_arc = list(state.emotional_arc)
        if affect_valence is not None:
            emotional_arc.append(affect_valence)
            # Keep arc bounded to last 100 data points
            if len(emotional_arc) > 100:
                emotional_arc = emotional_arc[-100:]

        updated = state.model_copy(update={
            "messages": updated_messages,
            "last_active": utc_now(),
            "emotional_arc": emotional_arc,
        })

        self._cache[state.conversation_id] = updated
        await self._save_to_redis(updated)
        return updated

    async def prepare_context(
        self,
        state: ConversationState,
    ) -> list[dict[str, str]]:
        """
        Prepare message history for the LLM context window.

        Strategy:
        - If total estimated tokens <= budget: return all messages verbatim
        - Otherwise: summarise older messages, return last N verbatim
        - Prepend the rolling summary as a system message

        This preserves semantic continuity without blowing the context budget.
        """
        messages = state.messages
        if not messages:
            return []

        formatted_all = [{"role": m.role, "content": m.content} for m in messages]
        estimated = _estimate_token_count(formatted_all)

        if estimated <= self._context_window_max_tokens:
            return formatted_all

        # Split: keep recent N verbatim, summarise the rest
        recent = messages[-self._summary_threshold:]
        older = messages[:-self._summary_threshold]

        if older:
            older_formatted = [{"role": m.role, "content": m.content} for m in older]
            summary = await self._summarise_segment(older_formatted)
        else:
            summary = state.older_messages_summary  # Use stored summary if no new older messages

        result: list[dict[str, str]] = []
        if summary:
            result.append({
                "role": "system",
                "content": f"Earlier in this conversation: {summary}",
            })
        result.extend({"role": m.role, "content": m.content} for m in recent)
        return result

    async def extract_topics_async(self, state: ConversationState) -> list[str]:
        """
        Extract active topics from recent messages (async, background-safe).
        Returns a list of short topic phrases.
        """
        if len(state.messages) < 2:
            return []

        recent_formatted = [{"role": m.role, "content": m.content} for m in state.messages[-10:]]
        prompt = build_topic_extraction_prompt(recent_formatted)
        try:
            if self._optimized:
                response = await self._llm.evaluate(  # type: ignore[call-arg]
                    prompt, max_tokens=100, temperature=0.2,
                    cache_system="voxis.conversation", cache_method="topic_extraction",
                )
            else:
                response = await self._llm.evaluate(prompt, max_tokens=100, temperature=0.2)
            raw = response.text.strip()
            topics = [t.strip() for t in raw.split(",") if t.strip()]
            return topics[:5]  # Cap at 5 active topics
        except Exception:
            self._logger.warning("topic_extraction_failed", exc_info=True)
            return []

    def get_emotional_arc_trend(self, state: ConversationState) -> float:
        """
        Return the recent emotional trend: positive = improving, negative = deteriorating.
        Computed as the mean of the last 5 valence estimates minus the mean of the 5 before that.
        Returns 0.0 if insufficient data.
        """
        arc = state.emotional_arc
        if len(arc) < 6:
            return 0.0
        recent_mean = sum(arc[-5:]) / 5
        earlier_mean = sum(arc[-10:-5]) / 5 if len(arc) >= 10 else sum(arc[:-5]) / max(1, len(arc) - 5)
        return recent_mean - earlier_mean

    async def update_topics(self, state: ConversationState, topics: list[str]) -> ConversationState:
        """Persist updated topic list to the conversation state."""
        updated = state.model_copy(update={"active_topics": topics})
        self._cache[state.conversation_id] = updated
        await self._save_to_redis(updated)
        return updated

    async def close_conversation(self, conversation_id: str) -> None:
        """Mark a conversation as complete and remove from active cache."""
        self._cache.pop(conversation_id, None)
        key = self._redis_key(conversation_id)
        await self._redis.delete(key)
        self._logger.info("conversation_closed", conversation_id=conversation_id)

    # ─── Private Helpers ──────────────────────────────────────────

    def _redis_key(self, conversation_id: str) -> str:
        return f"voxis:conv:{conversation_id}"

    async def _load_from_redis(self, conversation_id: str) -> ConversationState | None:
        key = self._redis_key(conversation_id)
        try:
            data = await self._redis.get_json(key)
            if data is None:
                return None
            return ConversationState(**data)
        except Exception:
            self._logger.warning("conversation_load_failed", conversation_id=conversation_id, exc_info=True)
            return None

    async def _save_to_redis(self, state: ConversationState) -> None:
        key = self._redis_key(state.conversation_id)
        try:
            await self._redis.set_json(key, state.model_dump(), ttl=_CONVERSATION_TTL_SECONDS)
        except Exception:
            self._logger.warning("conversation_save_failed", conversation_id=state.conversation_id, exc_info=True)

    async def _summarise_segment(self, messages: list[dict[str, str]]) -> str:
        """Summarise a segment of older messages using the LLM."""
        prompt = build_summarise_segment_prompt(messages)
        try:
            if self._optimized:
                response = await self._llm.evaluate(  # type: ignore[call-arg]
                    prompt, max_tokens=200, temperature=0.3,
                    cache_system="voxis.conversation", cache_method="summarise",
                )
            else:
                response = await self._llm.evaluate(prompt, max_tokens=200, temperature=0.3)
            return response.text.strip()
        except Exception:
            self._logger.warning("conversation_summarisation_failed", exc_info=True)
            # Fallback: return a simple count-based summary
            return f"[{len(messages)} earlier messages - summarisation unavailable]"


# ─── Helpers ─────────────────────────────────────────────────────


def _estimate_token_count(messages: list[dict[str, str]]) -> int:
    """
    Rough token count estimate: ~4 characters per token.
    Used only for context window budget decisions - does not need to be exact.
    """
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4
