"""
EcodiaOS — KVzip Context Compression

API-level context window compression for agentic tool loops.
Since EOS calls LLMs via API (not running local models), true KV-cache
pruning isn't available. Instead we apply "KVzip-inspired" compression
at the message level:

  1. Estimate token count per message (fast heuristic: chars / 4)
  2. When total context exceeds a threshold, compress oldest tool results
  3. Keep recent messages (sliding window) at full fidelity
  4. Summarise compressed tool results to essential information
  5. Track compression metrics for monitoring

The prune ratio from config (default 0.3) controls aggressiveness:
  ratio=0.0 → no compression
  ratio=0.3 → compress ~30% of the oldest tool results
  ratio=1.0 → compress everything except the sliding window

Target: 3-4x effective cache reduction on long agentic sessions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger().bind(system="context_compression")

# Heuristic: ~4 chars per token (conservative for code-heavy content)
_CHARS_PER_TOKEN = 4

# Maximum chars to keep in a compressed tool result
_COMPRESSED_RESULT_MAX_CHARS = 300

# Messages in the sliding window are never compressed
_DEFAULT_WINDOW_SIZE = 6  # last 3 assistant + 3 user turns


@dataclass
class CompressionMetrics:
    """Tracks context compression statistics across a session."""

    original_tokens: int = 0
    compressed_tokens: int = 0
    messages_compressed: int = 0
    total_compressions: int = 0

    @property
    def compression_ratio(self) -> float:
        """Ratio of tokens saved (0.0 = no savings, 1.0 = everything removed)."""
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)

    @property
    def effective_multiplier(self) -> float:
        """How many x smaller the compressed context is (target: 3-4x)."""
        if self.compressed_tokens == 0:
            return 0.0
        return self.original_tokens / self.compressed_tokens


class ContextCompressor:
    """
    KVzip-inspired context compressor for API-based LLM calls.

    Compresses the message history of an agentic tool loop to reduce
    token usage while preserving essential information for the LLM
    to continue its task.
    """

    def __init__(
        self,
        prune_ratio: float = 0.3,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        enabled: bool = True,
    ) -> None:
        self._prune_ratio = max(0.0, min(1.0, prune_ratio))
        self._window_size = window_size
        self._enabled = enabled
        self._metrics = CompressionMetrics()

    @property
    def metrics(self) -> CompressionMetrics:
        return self._metrics

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total token count for a message list."""
        total_chars = 0
        for msg in messages:
            total_chars += self._message_chars(msg)
        return total_chars // _CHARS_PER_TOKEN

    def compress(
        self,
        messages: list[dict[str, Any]],
        token_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compress the message history to reduce token usage.

        Applies compression to tool results outside the sliding window,
        preserving the most recent messages at full fidelity.

        Args:
            messages: The full conversation history (list of message dicts).
            token_budget: Optional hard token budget. If set, compresses
                          until under this limit.

        Returns:
            Compressed message list (new list, originals not mutated).
        """
        if not self._enabled or self._prune_ratio == 0.0 or len(messages) <= self._window_size:
            return messages

        original_tokens = self.estimate_tokens(messages)
        self._metrics.original_tokens += original_tokens

        # Split into compressible prefix and protected window
        window_start = max(0, len(messages) - self._window_size)
        prefix = messages[:window_start]
        window = messages[window_start:]

        if not prefix:
            self._metrics.compressed_tokens += original_tokens
            return messages

        # Determine how many prefix messages to compress
        num_to_compress = max(1, int(len(prefix) * self._prune_ratio))

        compressed_prefix: list[dict[str, Any]] = []
        messages_compressed = 0

        for i, msg in enumerate(prefix):
            if i < num_to_compress:
                compressed_msg = self._compress_message(msg)
                if compressed_msg != msg:
                    messages_compressed += 1
                compressed_prefix.append(compressed_msg)
            else:
                compressed_prefix.append(msg)

        result = compressed_prefix + window

        compressed_tokens = self.estimate_tokens(result)
        self._metrics.compressed_tokens += compressed_tokens
        self._metrics.messages_compressed += messages_compressed
        self._metrics.total_compressions += 1

        # If we have a hard budget and are still over, compress more aggressively
        if token_budget is not None and compressed_tokens > token_budget:
            result = self._aggressive_compress(result, token_budget)

        if messages_compressed > 0:
            logger.debug(
                "context_compressed",
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                messages_compressed=messages_compressed,
                ratio=round(1.0 - compressed_tokens / max(1, original_tokens), 3),
            )

        return result

    def _compress_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Compress a single message, targeting tool results for maximum savings."""
        content = msg.get("content")
        role = msg.get("role", "")

        # Tool results (user messages with list content containing tool_result blocks)
        if role == "user" and isinstance(content, list):
            compressed_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    compressed_content.append(self._compress_tool_result(block))
                else:
                    compressed_content.append(block)
            return {"role": role, "content": compressed_content}

        # Assistant messages with tool_use blocks — compress text, preserve tool calls
        if role == "assistant" and isinstance(content, list):
            compressed_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if len(text) > 500:
                        compressed_content.append({
                            "type": "text",
                            "text": self._summarise_text(text, 400),
                        })
                    else:
                        compressed_content.append(block)
                else:
                    compressed_content.append(block)
            return {"role": role, "content": compressed_content}

        # Plain text messages — compress if large
        if isinstance(content, str) and len(content) > 1000:
            return {"role": role, "content": self._summarise_text(content, 500)}

        return msg

    def _compress_tool_result(self, block: dict[str, Any]) -> dict[str, Any]:
        """Compress a tool_result block, keeping essential info."""
        content = block.get("content", "")
        if isinstance(content, str) and len(content) > _COMPRESSED_RESULT_MAX_CHARS:
            compressed = self._summarise_tool_output(content)
            return {
                **block,
                "content": compressed,
            }
        return block

    def _summarise_tool_output(self, output: str) -> str:
        """
        Summarise a tool output to its essential information.

        Strategy:
          - For file reads: keep first/last few lines + line count
          - For search results: keep file list + match count
          - For command output: keep exit status + first/last lines
          - General: truncate with ellipsis
        """
        lines = output.split("\n")
        total_lines = len(lines)

        # File content: keep structure hint
        if output.startswith("===") or "def " in output[:200] or "class " in output[:200]:
            # Extract class/function names as a structural summary
            defs = [
                line.strip()
                for line in lines
                if re.match(r"^\s*(class |def |async def )", line)
            ]
            if defs:
                summary_lines = defs[:8]
                return (
                    f"[Compressed: {total_lines} lines]\n"
                    f"Definitions found:\n" + "\n".join(f"  {d}" for d in summary_lines)
                )

        # Search results: keep file paths
        if "matches" in output.lower() or output.count(":") > 5:
            head = "\n".join(lines[:5])
            return f"[Compressed: {total_lines} lines]\n{head}\n..."

        # General: keep head + tail
        if total_lines > 10:
            head = "\n".join(lines[:4])
            tail = "\n".join(lines[-2:])
            return f"[Compressed: {total_lines} lines]\n{head}\n...\n{tail}"

        # Already short enough
        return output[:_COMPRESSED_RESULT_MAX_CHARS]

    def _summarise_text(self, text: str, max_chars: int) -> str:
        """Truncate text preserving sentence boundaries where possible."""
        if len(text) <= max_chars:
            return text

        # Try to cut at a sentence boundary
        cutoff = text[:max_chars]
        last_period = cutoff.rfind(".")
        if last_period > max_chars * 0.6:
            return cutoff[: last_period + 1] + " [...]"
        return cutoff + " [...]"

    def _aggressive_compress(
        self, messages: list[dict[str, Any]], token_budget: int
    ) -> list[dict[str, Any]]:
        """
        When standard compression isn't enough, aggressively drop old messages.
        Keeps the first message (task prompt) and the sliding window.
        """
        if len(messages) <= 2:
            return messages

        # Always keep first message (task prompt) and last window_size messages
        first = messages[:1]
        window = messages[-self._window_size:]

        # Add a context-loss marker
        marker = {
            "role": "user",
            "content": (
                "[Context compressed: earlier tool interactions were pruned "
                "to stay within token budget. The task prompt and recent "
                "interactions are preserved. Continue from where you left off.]"
            ),
        }

        result = first + [marker] + window

        compressed_tokens = self.estimate_tokens(result)
        if compressed_tokens > token_budget and len(window) > 2:
            # Last resort: shrink window
            result = first + [marker] + window[-2:]

        return result

    @staticmethod
    def _message_chars(msg: dict[str, Any]) -> int:
        """Count total characters in a message."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("content", "")))
                    total += len(str(block.get("text", "")))
                    total += len(str(block.get("input", "")))
                else:
                    total += len(str(block))
            return total
        return len(str(content))
