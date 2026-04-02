"""
EcodiaOS - LLM Provider Abstraction

Every system that needs LLM reasoning uses this interface.
Supports Anthropic Claude, OpenAI, and local models via Ollama.

Includes retry with exponential backoff for transient errors (429, 503, 529)
and automatic fallback to a secondary provider when the primary is unavailable.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from clients.token_budget import TokenBudget
    from config import LLMConfig

logger = structlog.get_logger()

# Retry configuration - reduced from 3 retries / 1s base to 2 retries / 0.5s base.
# Old config: up to 7s of retry delays (1+2+4) which exceeded the 5s deliberation budget.
# New config: up to 1.5s of retry delays (0.5+1) which fits within tight budgets.
_MAX_RETRIES = 2
_BASE_DELAY_S = 0.5
_RETRYABLE_STATUS_CODES = {429, 503, 529}


class Message:
    """A chat message."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMResponse:
    """Response from an LLM call."""

    def __init__(
        self,
        text: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "stop",
        reasoning_tokens: int = 0,
        reasoning_content: str = "",
    ) -> None:
        self.text = text
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens
        self.reasoning_content = reasoning_content

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    @property
    def used_extended_thinking(self) -> bool:
        return self.reasoning_tokens > 0


@dataclass
class ToolDefinition:
    """A tool that can be called by the LLM."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema object

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    tool_use_id: str
    content: str
    is_error: bool = False

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Format as Anthropic tool_result content block."""
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


@dataclass
class ToolAwareResponse:
    """Response from a tool-aware LLM call."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # end_turn | tool_use | max_tokens
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Abstract interface for LLM calls."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        """Full generation call."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Short evaluation call (lower temp, smaller output)."""
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],  # Raw message dicts (supports content blocks)
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        """
        Tool-use generation call. Messages use raw dicts to support
        Anthropic's content block format (text blocks + tool_result blocks).
        Returns ToolAwareResponse with any tool calls the model wants to make.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...

    # ─── Dynamic Adapter Management ─────────────────────────────
    # Default no-op implementations so API-based providers (Anthropic,
    # Bedrock, OpenAI) are unaffected. Only local providers (vLLM,
    # Ollama) override these to support LoRA adapter hot-swapping.

    @property
    def supports_adapters(self) -> bool:
        """Whether this provider supports dynamic LoRA adapter loading."""
        return False

    @property
    def active_adapter_id(self) -> str | None:
        """IPFS CID or identifier of the currently loaded adapter, or None."""
        return None

    async def load_adapter(self, adapter_path: str, adapter_id: str) -> None:
        """
        Load a LoRA adapter onto the running inference engine.

        Args:
            adapter_path: Local filesystem path to .safetensors adapter file.
            adapter_id: Stable identifier (e.g., IPFS CID) for tracking.

        Raises:
            NotImplementedError: If provider doesn't support adapters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support dynamic adapter loading"
        )

    async def unload_adapter(self) -> None:
        """Unload the current LoRA adapter, reverting to the base model."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support adapter unloading"
        )

    async def generate_with_thinking(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 4096,
        reasoning_budget: int = 16384,
    ) -> LLMResponse:
        """
        Extended-thinking generation. Models like o3/deepseek-r1 produce
        a chain-of-thought reasoning trace before the final answer.

        Default implementation falls back to standard generate() for
        providers that don't support extended thinking natively.
        """
        return await self.generate(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )

    async def generate_with_thinking_and_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        reasoning_budget: int = 16384,
    ) -> ToolAwareResponse:
        """
        Extended-thinking generation with tool use.
        Default falls back to standard generate_with_tools().
        """
        return await self.generate_with_tools(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.3,
        )


class AnthropicProvider(LLMProvider):
    """Claude API provider with retry and exponential backoff."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        token_budget: TokenBudget | None = None,
    ) -> None:
        self._model = model
        self._budget = token_budget
        # Strip whitespace/newlines - GCP Secret Manager can inject trailing \r\n
        clean_key = api_key.strip()
        # Timeout reduced from 60s to 10s. The deliberation engine has a 5s
        # end-to-end budget; a 60s httpx timeout meant a single hung request
        # could starve the event loop for a full minute before asyncio.timeout
        # could cancel it. 10s gives headroom for slow responses while still
        # allowing the cancellation machinery to work within budget.
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": clean_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=10.0,
        )

    async def _post_with_retry(
        self, path: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST with exponential backoff on retryable status codes."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(path, json=payload)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    # Respect Retry-After header if present
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        with contextlib.suppress(ValueError):
                            delay = max(delay, float(retry_after))
                    logger.warning(
                        "llm_retrying",
                        status=response.status_code,
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "llm_timeout_retrying",
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                # Include API error details in the exception message
                body = ""
                with contextlib.suppress(Exception):
                    body = exc.response.text[:500]
                raise httpx.HTTPStatusError(
                    message=f"{exc.response.status_code}: {body}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
        raise last_exc or RuntimeError("LLM request failed after retries")

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [m.to_dict() for m in messages],
        }

        data = await self._post_with_retry("/messages", payload)

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        input_tokens = data.get("usage", {}).get("input_tokens", 0)
        output_tokens = data.get("usage", {}).get("output_tokens", 0)

        # Charge the token budget if configured
        if self._budget:
            await self._budget.charge(
                tokens=input_tokens + output_tokens,
                calls=1,
                system="anthropic_generate",
            )

        return LLMResponse(
            text=text,
            model=data.get("model", self._model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=data.get("stop_reason", "stop"),
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": messages,
            "tools": [t.to_dict() for t in tools],
        }

        data = await self._post_with_retry("/messages", payload)

        # Parse content blocks - may include text and tool_use blocks
        text = ""
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    input=block.get("input", {}),
                ))

        input_tokens = data.get("usage", {}).get("input_tokens", 0)
        output_tokens = data.get("usage", {}).get("output_tokens", 0)

        # Charge the token budget if configured
        if self._budget:
            await self._budget.charge(
                tokens=input_tokens + output_tokens,
                calls=1,
                system="anthropic_tools",
            )

        return ToolAwareResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=data.get("stop_reason", "end_turn"),
            model=data.get("model", self._model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def close(self) -> None:
        await self._client.aclose()


class OllamaProvider(LLMProvider):
    """Local model via Ollama."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        endpoint: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=endpoint,
            timeout=120.0,
        )

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages.extend(m.to_dict() for m in messages)

        payload = {
            "model": self._model,
            "messages": all_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=self._model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        # Ollama tool use: flatten messages to text, ask for JSON tool calls.
        # This is a best-effort implementation for local model fallback.
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )
        user_content = (
            f"Available tools:\n{tool_descriptions}\n\n"
            "To use a tool, respond with JSON: "
        )
        user_content += "{\\\"tool\\\": \\\"<name>\\\", \\\"input\\\": {...}}\n\n"
        # Extract last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content += content
                break

        plain_messages = [Message("user", user_content)]
        response = await self.generate(
            system_prompt=system_prompt,
            messages=plain_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Try to parse a tool call from the response
        tool_calls: list[ToolCall] = []
        try:
            parsed = _json.loads(response.text.strip())
            if "tool" in parsed:
                tool_calls.append(ToolCall(
                    id="ollama_" + str(parsed["tool"]),
                    name=parsed["tool"],
                    input=parsed.get("input", {}),
                ))
        except (_json.JSONDecodeError, KeyError):
            pass

        return ToolAwareResponse(
            text=response.text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=self._model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    async def close(self) -> None:
        await self._client.aclose()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4o, GPT-4o-mini, o1, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._model = model
        clean_key = api_key.strip()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {clean_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def _post_with_retry(
        self, path: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST with exponential backoff on retryable status codes."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(path, json=payload)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        with contextlib.suppress(ValueError):
                            delay = max(delay, float(retry_after))
                    logger.warning(
                        "llm_retrying",
                        status=response.status_code,
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "llm_timeout_retrying",
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                body = ""
                with contextlib.suppress(Exception):
                    body = exc.response.text[:500]
                raise httpx.HTTPStatusError(
                    message=f"{exc.response.status_code}: {body}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
        raise last_exc or RuntimeError("LLM request failed after retries")

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        # OpenAI uses system message inside the messages array
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            content = m.content if m.content else " "  # OpenAI rejects empty content
            all_messages.append({"role": m.role, "content": content})

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": all_messages,
        }

        data = await self._post_with_retry("/chat/completions", payload)

        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage", {})

        return LLMResponse(
            text=text or "",
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        # Convert Anthropic-style messages to OpenAI format
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Anthropic uses content blocks (list of dicts); OpenAI uses strings
            if isinstance(content, list):
                # Flatten content blocks to text
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            parts.append(f"[Tool result: {block.get('content', '')}]")
                    elif isinstance(block, str):
                        parts.append(block)
                content = "\n".join(parts) or " "
            all_messages.append({"role": role, "content": content or " "})

        # Convert tool definitions to OpenAI format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": all_messages,
            "tools": openai_tools,
        }

        data = await self._post_with_retry("/chat/completions", payload)

        choices = data.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        text = message.get("content", "") or ""
        usage = data.get("usage", {})

        # Parse tool calls from OpenAI format
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            try:
                args = _json.loads(fn.get("arguments", "{}"))
            except _json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                input=args,
            ))

        stop_reason = choice.get("finish_reason", "stop")
        # Map OpenAI stop reasons to our convention
        if stop_reason == "tool_calls":
            stop_reason = "tool_use"
        elif stop_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        return ToolAwareResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def close(self) -> None:
        await self._client.aclose()


class ExtendedThinkingProvider(LLMProvider):
    """
    Extended-thinking model provider for reasoning-intensive tasks.

    Wraps OpenAI-compatible APIs (o3, deepseek-r1) that support
    a reasoning_effort / max_completion_tokens parameter to produce
    chain-of-thought traces before the final answer.

    The reasoning trace is captured and returned in LLMResponse.reasoning_content
    so downstream systems can audit the thought process.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "o3",
        base_url: str = "https://api.openai.com/v1",
        default_reasoning_budget: int = 16384,
    ) -> None:
        self._model = model
        self._default_reasoning_budget = default_reasoning_budget
        clean_key = api_key.strip()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {clean_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,  # Thinking models can take longer
        )

    async def _post_with_retry(
        self, path: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST with exponential backoff on retryable status codes."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(path, json=payload)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        with contextlib.suppress(ValueError):
                            delay = max(delay, float(retry_after))
                    logger.warning(
                        "thinking_model_retrying",
                        status=response.status_code,
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "thinking_model_timeout_retrying",
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                body = ""
                with contextlib.suppress(Exception):
                    body = exc.response.text[:500]
                raise httpx.HTTPStatusError(
                    message=f"{exc.response.status_code}: {body}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
        raise last_exc or RuntimeError("Thinking model request failed after retries")

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        """Standard generation - delegates to generate_with_thinking with default budget."""
        return await self.generate_with_thinking(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            reasoning_budget=self._default_reasoning_budget,
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate_with_thinking(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            reasoning_budget=self._default_reasoning_budget // 2,
        )

    async def generate_with_thinking(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 4096,
        reasoning_budget: int = 16384,
    ) -> LLMResponse:
        """
        Extended-thinking generation via OpenAI-compatible API.

        For o3-class models: uses max_completion_tokens and reasoning_effort.
        For deepseek-r1: the model auto-reasons; we capture the reasoning prefix.
        """
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            content = m.content if m.content else " "
            all_messages.append({"role": m.role, "content": content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": all_messages,
            "max_completion_tokens": max_tokens + reasoning_budget,
        }

        # o3/o4 models support reasoning_effort parameter
        if self._model.startswith("o"):
            if reasoning_budget >= 16384:
                payload["reasoning_effort"] = "high"
            elif reasoning_budget >= 4096:
                payload["reasoning_effort"] = "medium"
            else:
                payload["reasoning_effort"] = "low"

        data = await self._post_with_retry("/chat/completions", payload)

        choices = data.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        text = message.get("content", "") or ""
        usage = data.get("usage", {})

        # Extract reasoning tokens from completion_tokens_details
        completion_details = usage.get("completion_tokens_details", {})
        reasoning_tokens = completion_details.get("reasoning_tokens", 0)

        # Some models include reasoning in a separate field
        reasoning_content = message.get("reasoning_content", "")

        logger.info(
            "thinking_model_response",
            model=self._model,
            reasoning_tokens=reasoning_tokens,
            output_tokens=usage.get("completion_tokens", 0),
            input_tokens=usage.get("prompt_tokens", 0),
        )

        return LLMResponse(
            text=text,
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
            reasoning_tokens=reasoning_tokens,
            reasoning_content=reasoning_content,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        """Extended-thinking with tool use via OpenAI-compatible API."""
        return await self.generate_with_thinking_and_tools(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            reasoning_budget=self._default_reasoning_budget,
        )

    async def generate_with_thinking_and_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        reasoning_budget: int = 16384,
    ) -> ToolAwareResponse:
        """Tool-use generation with extended thinking."""
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            parts.append(f"[Tool result: {block.get('content', '')}]")
                    elif isinstance(block, str):
                        parts.append(block)
                content = "\n".join(parts) or " "
            all_messages.append({"role": role, "content": content or " "})

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": all_messages,
            "tools": openai_tools,
            "max_completion_tokens": max_tokens + reasoning_budget,
        }

        if self._model.startswith("o"):
            if reasoning_budget >= 16384:
                payload["reasoning_effort"] = "high"
            elif reasoning_budget >= 4096:
                payload["reasoning_effort"] = "medium"
            else:
                payload["reasoning_effort"] = "low"

        data = await self._post_with_retry("/chat/completions", payload)

        choices = data.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        text = message.get("content", "") or ""
        usage = data.get("usage", {})

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            try:
                args = _json.loads(fn.get("arguments", "{}"))
            except _json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                input=args,
            ))

        stop_reason = choice.get("finish_reason", "stop")
        if stop_reason == "tool_calls":
            stop_reason = "tool_use"
        elif stop_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        return ToolAwareResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def close(self) -> None:
        await self._client.aclose()


class BedrockProvider(LLMProvider):
    """AWS Bedrock Claude provider."""

    def __init__(
        self,
        model: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        region: str = "us-east-1",
        token_budget: TokenBudget | None = None,
    ) -> None:
        """
        Initialize Bedrock provider. Uses AWS SDK credentials from environment:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional).
        Region defaults to us-east-1; override with AWS_REGION env var.
        """
        self._model = model
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._budget = token_budget

        # Import boto3 here to avoid hard dependency if not using Bedrock
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 required for Bedrock provider. Install with: pip install boto3") from exc

        try:
            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Bedrock client. Ensure AWS credentials are configured. "
                f"Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION environment variables. "
                f"Error: {e}"
            ) from e

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        body = {
            "messages": [m.to_dict() for m in messages],
            "system": [{"type": "text", "text": system_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "anthropic_version": "bedrock-2023-05-31",
        }

        # Bedrock InvokeModel is synchronous; wrap in thread pool so
        # the event loop is not blocked during the network call.
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model(modelId=self._model, body=_json.dumps(body).encode())
        )

        response_body = _json.loads(response["body"].read())

        text = ""
        for block in response_body.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
        output_tokens = response_body.get("usage", {}).get("output_tokens", 0)

        if self._budget:
            await self._budget.charge(
                tokens=input_tokens + output_tokens,
                calls=1,
                system="bedrock_generate",
            )

        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=response_body.get("stop_reason", "stop"),
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        body = {
            "messages": messages,
            "system": [{"type": "text", "text": system_prompt}],
            "tools": [t.to_dict() for t in tools],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "anthropic_version": "bedrock-2023-05-31",
        }

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model(modelId=self._model, body=_json.dumps(body).encode())
        )

        response_body = _json.loads(response["body"].read())

        text = ""
        tool_calls: list[ToolCall] = []
        for block in response_body.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    input=block.get("input", {}),
                ))

        input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
        output_tokens = response_body.get("usage", {}).get("output_tokens", 0)

        if self._budget:
            await self._budget.charge(
                tokens=input_tokens + output_tokens,
                calls=1,
                system="bedrock_tools",
            )

        return ToolAwareResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response_body.get("stop_reason", "end_turn"),
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def close(self) -> None:
        # Boto3 doesn't need explicit close for synchronous client
        pass


class VLLMProvider(LLMProvider):
    """
    vLLM-based local inference with LoRA adapter hot-swap support.

    Communicates via the OpenAI-compatible API that vLLM serves. Supports
    dynamic adapter loading/unloading through vLLM's LoRA management
    endpoints without restarting the engine.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-8B",
        endpoint: str = "http://localhost:8000",
    ) -> None:
        self._model = model
        self._endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._endpoint,
            timeout=120.0,
        )
        self._active_adapter_id: str | None = None
        self._active_adapter_name: str | None = None

    # ─── Adapter Management ──────────────────────────────────

    @property
    def supports_adapters(self) -> bool:
        return True

    @property
    def active_adapter_id(self) -> str | None:
        return self._active_adapter_id

    async def load_adapter(self, adapter_path: str, adapter_id: str) -> None:
        """
        Register a LoRA adapter with vLLM.

        vLLM supports two modes for LoRA adapters:
        1. Static: pass --lora-modules name=path at server startup
        2. Dynamic: POST /v1/load_lora_adapter (requires --enable-lora flag)

        Tries the dynamic endpoint first; if it returns 404/405 (not available
        in this vLLM build), falls back to marking the adapter as active
        client-side (assumes it was loaded at startup via --lora-modules).
        """
        adapter_name = f"eos-lora-{adapter_id[:12]}"

        try:
            response = await self._client.post(
                "/v1/load_lora_adapter",
                json={
                    "lora_name": adapter_name,
                    "lora_path": adapter_path,
                },
                timeout=300.0,
            )
            response.raise_for_status()
            logger.info(
                "vllm_adapter_loaded_dynamic",
                adapter_id=adapter_id,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (404, 405):
                logger.info(
                    "vllm_dynamic_lora_not_available",
                    adapter_id=adapter_id,
                    adapter_name=adapter_name,
                    hint="Adapter must be loaded at vLLM startup via --lora-modules",
                )
            else:
                raise

        self._active_adapter_id = adapter_id
        self._active_adapter_name = adapter_name

    async def unload_adapter(self) -> None:
        """Unload the current LoRA adapter from vLLM."""
        if self._active_adapter_name is None:
            return

        try:
            response = await self._client.post(
                "/v1/unload_lora_adapter",
                json={"lora_name": self._active_adapter_name},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (404, 405):
                logger.info(
                    "vllm_dynamic_lora_unload_not_available",
                    adapter_name=self._active_adapter_name,
                    hint="Adapter was loaded at startup - restart vLLM to unload",
                )
            else:
                raise

        logger.info(
            "vllm_adapter_unloaded",
            adapter_id=self._active_adapter_id,
            adapter_name=self._active_adapter_name,
        )
        self._active_adapter_id = None
        self._active_adapter_name = None

    # ─── LLMProvider Interface ───────────────────────────────

    def _build_model_field(self) -> str:
        """Return the model field for API calls, including adapter if loaded."""
        if self._active_adapter_name:
            return self._active_adapter_name
        return self._model

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages.extend(m.to_dict() for m in messages)

        payload: dict[str, Any] = {
            "model": self._build_model_field(),
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if output_format == "json":
            payload["response_format"] = {"type": "json_object"}

        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            text=choice["message"]["content"],
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        # vLLM supports the OpenAI tool-calling protocol directly
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages.extend(messages)

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

        payload: dict[str, Any] = {
            "model": self._build_model_field(),
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": openai_tools,
        }

        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        tool_calls: list[ToolCall] = []
        for tc in choice["message"].get("tool_calls", []):
            fn = tc.get("function", {})
            try:
                args = _json.loads(fn.get("arguments", "{}"))
            except _json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", f"vllm_{fn.get('name', 'unknown')}"),
                name=fn.get("name", ""),
                input=args,
            ))

        return ToolAwareResponse(
            text=choice["message"].get("content", "") or "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def close(self) -> None:
        await self._client.aclose()


def create_llm_provider(
    config: LLMConfig,
    token_budget: TokenBudget | None = None,
) -> LLMProvider:
    """Factory to create the configured LLM provider."""
    import structlog
    logger = structlog.get_logger()

    try:
        if config.provider == "anthropic":
            return AnthropicProvider(
                api_key=config.api_key,
                model=config.model,
                token_budget=token_budget,
            )
        elif config.provider == "bedrock":
            return BedrockProvider(
                model=config.model,
                token_budget=token_budget,
            )
        elif config.provider == "openai":
            base_url = config.endpoint if config.endpoint and "localhost" not in config.endpoint else "https://api.openai.com/v1"
            return OpenAIProvider(api_key=config.api_key, model=config.model, base_url=base_url)
        elif config.provider == "ollama":
            return OllamaProvider(model=config.model)
        elif config.provider == "vllm":
            return VLLMProvider(model=config.model, endpoint=config.endpoint)
        else:
            raise ValueError(f"Unknown LLM provider: {config.provider}")
    except Exception as e:
        # Try fallback provider if configured
        if config.fallback_provider:
            logger.warning("llm_provider_init_failed", provider=config.provider, error=str(e), fallback=config.fallback_provider)
            try:
                if config.fallback_provider == "anthropic":
                    return AnthropicProvider(
                        api_key=config.api_key,
                        model=config.model,
                        token_budget=token_budget,
                    )
                elif config.fallback_provider == "openai":
                    fallback_url = config.endpoint if config.endpoint and "localhost" not in config.endpoint else "https://api.openai.com/v1"
                    return OpenAIProvider(api_key=config.api_key, model=config.fallback_model or config.model, base_url=fallback_url)
                elif config.fallback_provider == "ollama":
                    return OllamaProvider(model=config.model)
            except Exception as fallback_error:
                logger.error("llm_fallback_also_failed", error=str(fallback_error))
                raise ValueError(f"Both primary and fallback LLM providers failed. Primary: {e}, Fallback: {fallback_error}") from e
        raise


def create_thinking_provider(
    api_key: str,
    model: str = "o3",
    provider: str = "openai",
    reasoning_budget: int = 16384,
) -> ExtendedThinkingProvider:
    """Factory to create an extended-thinking model provider."""
    base_urls: dict[str, str] = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }
    base_url = base_urls.get(provider, "https://api.openai.com/v1")
    return ExtendedThinkingProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        default_reasoning_budget=reasoning_budget,
    )
