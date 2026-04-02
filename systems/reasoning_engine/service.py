"""
EcodiaOS - Reasoning Engine Service

Wraps a vLLM-served local model (default: ecodiaos-reasoning / Qwen3-8B)
as an LLMProvider so Nova's PolicyGenerator can route slow-path deliberation
to it via Thompson sampling.

The RE is completely optional - if vLLM is not running the organism operates
identically to today (Claude-only mode).  The circuit breaker ensures a
crashed vLLM server degrades gracefully rather than hanging deliberation.

Env vars:
    ORGANISM_RE_VLLM_URL    vLLM OpenAI-compatible server (default: http://localhost:8001/v1)
    ORGANISM_RE_MODEL       Model name served by vLLM (default: ecodiaos-reasoning)
    ORGANISM_RE_ENABLED     Set to "false" to disable entirely (default: true)

Integration:
    - Implements LLMProvider ABC from clients/llm.py
    - Passed as re_client to PolicyGenerator (Spec 05 Thompson sampling)
    - Emits RE_ENGINE_STATUS_CHANGED on availability transitions
    - app.state.reasoning_engine - accessible from health endpoints
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from clients.llm import LLMProvider, LLMResponse, Message, ToolAwareResponse, ToolDefinition

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()

_CIRCUIT_BREAKER_THRESHOLD = 5
_PROBE_TIMEOUT_S = 15.0
_GENERATE_TIMEOUT_S = 30.0


class ReasoningEngineService(LLMProvider):
    """
    vLLM-backed LLMProvider for the organism's local Reasoning Engine.

    Implements the full LLMProvider ABC so it can be dropped directly into
    Nova's PolicyGenerator as re_client.  evaluate() and generate_with_tools()
    are implemented as best-effort thin wrappers - the RE is primarily used
    for generate() in the slow-path deliberation hot loop.

    Circuit breaker: tracks consecutive failures.  After _CIRCUIT_BREAKER_THRESHOLD
    failures the circuit opens and is_available returns False.  The next success
    resets the counter and closes the circuit.
    """

    def __init__(
        self,
        vllm_url: str | None = None,
        model_name: str | None = None,
        synapse: Any = None,
    ) -> None:
        _raw_url = (
            vllm_url
            or os.environ.get("ORGANISM_RE_VLLM_URL", "http://localhost:8001/v1")
        ).rstrip("/")
        # Ensure URL ends with /v1 - vLLM OpenAI-compatible API lives at /v1/*
        if not _raw_url.endswith("/v1"):
            _raw_url = _raw_url + "/v1"
        self._url = _raw_url
        self._model = model_name or os.environ.get(
            "ORGANISM_RE_MODEL", "ecodiaos-reasoning"
        )
        self._synapse = synapse  # injected after startup for event emission
        self._neo4j = None  # injected after startup via set_neo4j()

        self._available: bool = False
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False
        self._reprobe_task: asyncio.Task[None] | None = None
        self._reprobe_interval_s = float(
            os.environ.get("ORGANISM_RE_REPROBE_INTERVAL_S", "120")
        )

        self._client = httpx.AsyncClient(
            base_url=self._url,
            timeout=_GENERATE_TIMEOUT_S,
        )
        self._logger = logger.bind(
            system="reasoning_engine",
            vllm_url=self._url,
            model=self._model,
        )

    # ─── Availability ──────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when vLLM probed successfully and circuit breaker is closed."""
        return self._available and not self._circuit_open

    def set_synapse(self, synapse: Any) -> None:
        """Wire Synapse bus after startup (avoids constructor cross-imports)."""
        self._synapse = synapse

    def set_neo4j(self, neo4j: Any) -> None:
        """Wire Neo4j client after startup for Thompson state persistence."""
        self._neo4j = neo4j

    # ─── Lifecycle ─────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Probe vLLM /v1/models endpoint.  Sets _available=True if the target
        model is listed.  Non-fatal - logs a warning and leaves the organism
        in Claude-only mode if the server is unreachable.
        """
        try:
            async with asyncio.timeout(_PROBE_TIMEOUT_S):
                resp = await self._client.get("/models")
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                model_ids = [m.get("id", "") for m in data.get("data", [])]
                if self._model in model_ids:
                    self._available = True
                    self._logger.info(
                        "reasoning_engine_available",
                        model=self._model,
                        all_models=model_ids,
                    )
                else:
                    self._logger.warning(
                        "reasoning_engine_model_not_found",
                        wanted=self._model,
                        available=model_ids,
                    )
        except (httpx.ConnectError, httpx.TimeoutException, TimeoutError) as exc:
            self._logger.info(
                "reasoning_engine_unavailable",
                reason="vLLM not reachable - Claude-only mode",
                error=str(exc),
            )
        except Exception as exc:
            self._logger.warning(
                "reasoning_engine_probe_failed",
                error=str(exc),
                type=type(exc).__name__,
                exc_info=True,
            )

        await self._load_thompson()

    async def health(self) -> dict[str, Any]:
        """Return current availability status for the /health endpoint."""
        return {
            "available": self._available,
            "circuit_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
            "model": self._model,
            "url": self._url,
            "is_available": self.is_available,
        }

    # ─── Circuit breaker helpers ────────────────────────────────────────

    def _on_success(self) -> None:
        was_open = self._circuit_open
        self._consecutive_failures = 0
        self._circuit_open = False
        if was_open:
            self._logger.info("reasoning_engine_circuit_closed")
            self._emit_status_changed(available=True)

    def _on_failure(self, error: str) -> None:
        self._consecutive_failures += 1
        if (
            self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD
            and not self._circuit_open
        ):
            self._circuit_open = True
            self._logger.warning(
                "reasoning_engine_circuit_opened",
                failures=self._consecutive_failures,
                threshold=_CIRCUIT_BREAKER_THRESHOLD,
            )
            self._emit_status_changed(available=False)
        else:
            self._logger.debug(
                "reasoning_engine_failure",
                consecutive=self._consecutive_failures,
                error=error,
            )

    def _emit_status_changed(self, available: bool) -> None:
        """Fire-and-forget RE_ENGINE_STATUS_CHANGED on the Synapse bus."""
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEventType

            asyncio.ensure_future(
                self._synapse.emit(
                    SynapseEventType.RE_ENGINE_STATUS_CHANGED,
                    {
                        "available": available,
                        "model": self._model,
                        "url": self._url,
                        "consecutive_failures": self._consecutive_failures,
                        "circuit_open": self._circuit_open,
                    },
                    source_system="reasoning_engine",
                )
            )
        except Exception:
            pass  # Never block on telemetry

    # ─── LLMProvider ABC ───────────────────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        """
        Call vLLM /v1/chat/completions with the same signature as AnthropicProvider.

        Maps EOS Message objects → OpenAI chat format.  If output_format=="json"
        sets response_format={"type":"json_object"} (vLLM supports this for
        most instruction-tuned models).
        """
        t0 = time.monotonic()
        openai_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m.role, "content": m.content} for m in messages
        ]

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if output_format == "json":
            payload["response_format"] = {"type": "json_object"}

        try:
            async with asyncio.timeout(_GENERATE_TIMEOUT_S):
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()

            data = resp.json()
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            usage = data.get("usage", {})

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            self._logger.debug(
                "reasoning_engine_generate_ok",
                elapsed_ms=elapsed_ms,
                output_tokens=usage.get("completion_tokens", 0),
            )
            self._on_success()

            return LLMResponse(
                text=text,
                model=self._model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
            )

        except (TimeoutError, asyncio.TimeoutError) as exc:
            self._on_failure("timeout")
            raise RuntimeError(f"RE generate timeout after {_GENERATE_TIMEOUT_S}s") from exc
        except Exception as exc:
            self._on_failure(str(exc))
            raise

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Short evaluation call - wraps generate() with a minimal user message."""
        return await self.generate(
            system_prompt="You are a precise evaluator. Answer concisely.",
            messages=[Message(role="user", content=prompt)],
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
        """
        Tool use via vLLM.  Most local models have limited tool-call support;
        this is included to satisfy the ABC.  In practice Nova only calls
        generate() on the RE - tool use stays on Claude.
        """
        openai_messages = [{"role": "system", "content": system_prompt}] + messages
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
            "messages": openai_messages,
            "tools": openai_tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with asyncio.timeout(_GENERATE_TIMEOUT_S):
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()

            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            usage = data.get("usage", {})

            from clients.llm import ToolCall

            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    input=tc["function"].get("arguments", {}),
                )
                for tc in (message.get("tool_calls") or [])
            ]

            self._on_success()
            return ToolAwareResponse(
                text=text,
                tool_calls=tool_calls,
                stop_reason=choice.get("finish_reason", "end_turn"),
                model=self._model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

        except (TimeoutError, asyncio.TimeoutError) as exc:
            self._on_failure("timeout")
            raise RuntimeError("RE generate_with_tools timeout") from exc
        except Exception as exc:
            self._on_failure(str(exc))
            raise

    # ─── Embeddings ─────────────────────────────────────────────────────

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode texts to embeddings via vLLM embeddings endpoint.
        Returns list of float vectors. Falls back to empty list on failure.
        Used by Pillar 2 novelty distance calculation.
        """
        if not texts or not self._available:
            return []
        try:
            async with asyncio.timeout(10.0):
                resp = await self._client.post(
                    "/embeddings",
                    json={"model": self._model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data.get("data", [])]
        except Exception as exc:
            self._logger.debug("reasoning_engine.encode_failed", error=str(exc))
            return []

    # ─── Thompson sampling persistence ─────────────────────────────────

    # Source constants used by Nova's PolicyGenerator for Thompson routing.
    # Stored here so persistence/restore are co-located with the service.
    _THOMPSON_CUSTOM_KEY = "custom"
    _THOMPSON_CLAUDE_KEY = "claude"

    async def _load_thompson(self) -> None:
        """Restore Thompson sampling state from Neo4j on startup."""
        if not self._neo4j:
            return
        try:
            result = await self._neo4j.run(
                "MATCH (t:ThompsonState {service: 'reasoning_engine'}) "
                "RETURN t.custom_alpha AS ca, t.custom_beta AS cb, "
                "t.claude_alpha AS la, t.claude_beta AS lb LIMIT 1"
            )
            row = await result.single()
            if row:
                if not hasattr(self, "_thompson"):
                    self._thompson: dict[str, dict[str, float]] = {
                        self._THOMPSON_CUSTOM_KEY: {"alpha": 1.0, "beta": 1.0},
                        self._THOMPSON_CLAUDE_KEY: {"alpha": 1.0, "beta": 1.0},
                    }
                self._thompson[self._THOMPSON_CUSTOM_KEY]["alpha"] = float(row["ca"] or 1.0)
                self._thompson[self._THOMPSON_CUSTOM_KEY]["beta"]  = float(row["cb"] or 1.0)
                self._thompson[self._THOMPSON_CLAUDE_KEY]["alpha"] = float(row["la"] or 1.0)
                self._thompson[self._THOMPSON_CLAUDE_KEY]["beta"]  = float(row["lb"] or 1.0)
                self._logger.info(
                    "thompson_restored",
                    custom_alpha=self._thompson[self._THOMPSON_CUSTOM_KEY]["alpha"],
                    claude_alpha=self._thompson[self._THOMPSON_CLAUDE_KEY]["alpha"],
                )
        except Exception as e:
            self._logger.warning("thompson_load_failed", error=str(e))

    async def _persist_thompson(self) -> None:
        """Persist Thompson sampling state to Neo4j after every update."""
        if not self._neo4j:
            return
        thompson = getattr(self, "_thompson", None)
        if not thompson:
            return
        try:
            c = thompson[self._THOMPSON_CUSTOM_KEY]
            cl = thompson[self._THOMPSON_CLAUDE_KEY]
            await self._neo4j.run(
                "MERGE (t:ThompsonState {service: 'reasoning_engine'}) "
                "SET t.custom_alpha = $ca, t.custom_beta = $cb, "
                "    t.claude_alpha = $la, t.claude_beta = $lb, "
                "    t.updated_at = datetime()",
                ca=c["alpha"], cb=c["beta"], la=cl["alpha"], lb=cl["beta"],
            )
        except Exception as e:
            self._logger.warning("thompson_persist_failed", error=str(e))

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()

    # ─── Adapter support ───────────────────────────────────────────────

    @property
    def supports_adapters(self) -> bool:
        """vLLM supports LoRA adapter hot-swapping via its /v1/load_lora_adapter API."""
        return True

    async def load_adapter(self, adapter_path: str, adapter_id: str) -> None:
        """
        Load a LoRA adapter onto the running vLLM server.

        Tries POST /v1/load_lora_adapter for dynamic injection. If the endpoint
        returns 404/405 (not available in this vLLM build), falls back to
        client-side tracking - assumes adapter was loaded at startup via
        --lora-modules flag.
        """
        payload = {
            "lora_name": adapter_id,
            "lora_path": adapter_path,
        }
        try:
            async with asyncio.timeout(30.0):
                resp = await self._client.post("/load_lora_adapter", json=payload)
                resp.raise_for_status()
            self._logger.info(
                "re_adapter_loaded_dynamic",
                adapter_id=adapter_id,
                path=adapter_path,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (404, 405):
                self._logger.info(
                    "re_dynamic_lora_not_available",
                    adapter_id=adapter_id,
                    hint="Adapter must be loaded at vLLM startup via --lora-modules",
                )
            else:
                self._logger.error("re_adapter_load_failed", adapter_id=adapter_id, error=str(exc))
                raise
        except Exception as exc:
            self._logger.error("re_adapter_load_failed", adapter_id=adapter_id, error=str(exc))
            raise
        self._active_adapter_id = adapter_id

    async def unload_adapter(self) -> None:
        """Unload the active LoRA adapter, reverting to base model weights."""
        name = getattr(self, "_active_adapter_id", None)
        if not name:
            return
        try:
            async with asyncio.timeout(10.0):
                resp = await self._client.post(
                    "/unload_lora_adapter", json={"lora_name": name}
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (404, 405):
                self._logger.info(
                    "re_dynamic_lora_unload_not_available",
                    adapter_name=name,
                    hint="Adapter was loaded at startup - restart vLLM to unload",
                )
            else:
                self._logger.warning("re_adapter_unload_failed", error=str(exc))
        except Exception as exc:
            self._logger.warning("re_adapter_unload_failed", error=str(exc))
        self._active_adapter_id = None
        self._logger.info("re_adapter_unloaded")

    @property
    def active_adapter_id(self) -> str | None:
        return getattr(self, "_active_adapter_id", None)

    # ─── Circuit breaker reprobe ─────────────────────────────────────

    def start_reprobe_loop(self) -> None:
        """
        Start a background task that periodically reprobes vLLM when the
        circuit is open. This handles the case where vLLM restarts (e.g.
        adapter_watcher restarting it with a new adapter) while the circuit
        breaker is open - the organism auto-discovers the recovered RE.
        """
        if self._reprobe_task is not None:
            return
        self._reprobe_task = asyncio.ensure_future(self._reprobe_loop())
        self._logger.info(
            "re_reprobe_loop_started",
            interval_s=self._reprobe_interval_s,
        )

    async def stop_reprobe_loop(self) -> None:
        """Cancel the reprobe background task."""
        if self._reprobe_task is not None:
            self._reprobe_task.cancel()
            try:
                await self._reprobe_task
            except asyncio.CancelledError:
                pass
            self._reprobe_task = None

    async def _reprobe_loop(self) -> None:
        """
        Background loop: when circuit is open or RE unavailable, periodically
        probe /v1/models to detect recovery. Also detects model changes (e.g.
        adapter_watcher restarted vLLM with a new adapter name).
        """
        while True:
            try:
                await asyncio.sleep(self._reprobe_interval_s)
            except asyncio.CancelledError:
                return

            # Only reprobe when the circuit is open or RE was never available
            if self._available and not self._circuit_open:
                continue

            try:
                async with asyncio.timeout(_PROBE_TIMEOUT_S):
                    resp = await self._client.get("/models")
                    resp.raise_for_status()
                    data: dict[str, Any] = resp.json()
                    model_ids = [m.get("id", "") for m in data.get("data", [])]

                    if self._model in model_ids:
                        was_open = self._circuit_open
                        was_unavailable = not self._available
                        self._available = True
                        self._consecutive_failures = 0
                        self._circuit_open = False

                        if was_open or was_unavailable:
                            self._logger.info(
                                "re_reprobe_recovered",
                                model=self._model,
                                was_circuit_open=was_open,
                                was_unavailable=was_unavailable,
                            )
                            self._emit_status_changed(available=True)
                    else:
                        self._logger.debug(
                            "re_reprobe_model_not_found",
                            wanted=self._model,
                            available=model_ids,
                        )
            except (httpx.ConnectError, httpx.TimeoutException, TimeoutError):
                self._logger.debug("re_reprobe_unreachable")
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.debug("re_reprobe_error", error=str(exc))
