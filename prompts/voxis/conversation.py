"""
EcodiaOS — Voxis Conversation Summary Prompts

Prompts for LLM-based conversation summarisation used in context window management.
"""

from __future__ import annotations


def build_summarise_segment_prompt(messages: list[dict[str, str]]) -> str:
    """
    Build a prompt to summarise a segment of conversation history.

    Used when the full conversation history exceeds the context window budget —
    older messages are summarised and replaced with this summary so the LLM
    retains semantic continuity without the full verbatim history.
    """
    formatted = "\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in messages
    )
    return (
        "Summarise the following conversation segment in 2-4 sentences. "
        "Capture: the main topics discussed, any decisions or conclusions reached, "
        "any questions left open, and the emotional tone. "
        "Write from a neutral third-person perspective.\n\n"
        f"CONVERSATION:\n{formatted}\n\n"
        "SUMMARY:"
    )


def build_topic_extraction_prompt(recent_messages: list[dict[str, str]]) -> str:
    """
    Build a prompt to extract active topics from recent messages.

    Used to maintain the active_topics list in ConversationState,
    which informs memory retrieval and expression strategy.
    """
    formatted = "\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in recent_messages[-10:]
    )
    return (
        "From the conversation below, list the 1-3 main topics currently being discussed. "
        "Return ONLY a comma-separated list of short topic phrases (2-5 words each). "
        "No preamble, no explanation.\n\n"
        f"CONVERSATION:\n{formatted}\n\n"
        "TOPICS:"
    )
