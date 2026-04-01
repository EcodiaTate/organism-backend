"""
EcodiaOS - Optimized LLM Provider

Transparent wrapper around any LLMProvider that intercepts all calls and applies:
1. Budget check - skip LLM if budget exhausted (return None → caller uses fallback)
2. Prompt cache - return cached response if available
3. Output validation - auto-correct malformed responses
4. Metrics recording - track tokens, cost, latency, cache hits per system
5. Budget charging - record actual consumption

This is the central integration point for all five optimization tools.
Systems that need heuristic fallbacks opt in by checking the budget tier
before calling the LLM, and routing to heuristics when appropriate.

Usage:
    optimized = OptimizedLLMProvider(
        inner=anthropic_provider,
        cache=prompt_cache,
        budget=token_budget,
        metrics=metrics_collector,
    )
    # Drop-in replacement for any LLMProvider
    response = await optimized.generate(system_prompt, messages, ...)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolAwareResponse,
    ToolDefinition,
)
from clients.output_validator import OutputValidator
from clients.prompt_cache import PromptCache, TTLConfig
from clients.token_budget import BudgetTier, TokenBudget

if TYPE_CHECKING:
    from telemetry.llm_metrics import LLMMetricsCollector

# Signature: (caller_id: str, input_tokens: int, output_tokens: int, provider: str) -> float
MetabolicCallback = Callable[[str, int, int, str], float]

# Signature: (had_error: bool) -> None - reports inference outcome to RollbackMonitor
InferenceErrorCallback = Callable[[bool], None]

logger = structlog.get_logger()


# System priority classification for budget-based degradation
# CRITICAL: Never degrade, always call LLM regardless of budget
# STANDARD: Degrade in RED tier (use heuristics)
# LOW: Degrade in YELLOW and RED tiers
SYSTEM_PRIORITY: dict[str, str] = {
    # Critical - never skip
    "equor.invariants": "critical",
    "equor.alignment": "critical",
    # Standard - skip only in RED
    "nova.efe.pragmatic": "standard",
    "nova.efe.epistemic": "standard",
    "voxis.render": "standard",
    "voxis.conversation": "standard",
    "thymos.diagnosis": "standard",
    "simula.code_agent": "standard",
    "simula.simulation": "standard",
    "axon.observation": "standard",
    "atune.entity_extraction": "standard",
    # Low - skip in YELLOW and RED
    "evo.hypothesis": "low",
    "evo.procedure": "low",
    "evo.evidence": "low",
    "thread.scene": "low",
    "thread.chapter": "low",
    "thread.life_story": "low",
    "thread.schema": "low",
    "thread.evidence": "low",
    "oneiros.rem.dream": "low",
    "oneiros.rem.threat": "low",
    "oneiros.nrem.pattern": "low",
    "oneiros.nrem.ethical": "low",
    "oneiros.lucid.explore": "low",
    "oneiros.lucid.meta": "low",
}

# Default TTLs per system (seconds)
SYSTEM_TTL: dict[str, int] = {
    "nova.efe.pragmatic": TTLConfig.NOVA_EFE_PRAGMATIC_S,
    "nova.efe.epistemic": TTLConfig.NOVA_EFE_EPISTEMIC_S,
    "nova.policy": TTLConfig.NOVA_POLICY_S,
    "voxis.render": TTLConfig.VOXIS_EXPRESSION_S,
    "voxis.conversation": TTLConfig.VOXIS_OUTLINE_S,
    "evo.hypothesis": TTLConfig.EVO_HYPOTHESIS_S,
    "evo.procedure": TTLConfig.EVO_INDUCTION_S,
    "evo.evidence": TTLConfig.EVO_HYPOTHESIS_S,
    "thread.scene": TTLConfig.THREAD_SYNTHESIS_S,
    "thread.chapter": TTLConfig.THREAD_SYNTHESIS_S,
    "thread.life_story": TTLConfig.THREAD_SYNTHESIS_S,
    "thread.schema": TTLConfig.THREAD_COHERENCE_S,
    "thread.evidence": TTLConfig.THREAD_COHERENCE_S,
    "equor.invariants": TTLConfig.EQUOR_INVARIANT_S,
    "thymos.diagnosis": 300,  # 5 min - incidents are context-dependent
    "oneiros.rem.dream": TTLConfig.ONEIROS_REFLECTION_S,
    "oneiros.rem.threat": TTLConfig.ONEIROS_REFLECTION_S,
    "oneiros.nrem.pattern": TTLConfig.ONEIROS_REFLECTION_S,
    "oneiros.nrem.ethical": TTLConfig.ONEIROS_REFLECTION_S,
    "oneiros.lucid.explore": TTLConfig.ONEIROS_REFLECTION_S,
    "oneiros.lucid.meta": TTLConfig.ONEIROS_REFLECTION_S,
    "simula.code_agent": 0,  # Never cache tool-use agentic calls
    "simula.simulation": 300,
    "axon.observation": 120,
    "atune.entity_extraction": 300,
}


class OptimizedLLMProvider(LLMProvider):
    """
    Drop-in LLMProvider wrapper that adds caching, budget, metrics, and validation.

    All existing systems continue to call generate/evaluate/generate_with_tools
    as before - the optimization is transparent.

    For systems that need finer control (heuristic fallbacks), use:
        optimized.should_use_llm(system, estimated_tokens) → bool
        optimized.get_budget_tier() → BudgetTier
    """

    def __init__(
        self,
        inner: LLMProvider,
        cache: PromptCache | None = None,
        budget: TokenBudget | None = None,
        metrics: LLMMetricsCollector | None = None,
        metabolic_callback: MetabolicCallback | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._budget = budget
        self._metrics = metrics
        self._metabolic_callback: MetabolicCallback | None = metabolic_callback
        self._inference_error_callback: InferenceErrorCallback | None = None
        self._validator = OutputValidator()
        self._logger = logger.bind(component="optimized_llm")

    def set_inference_error_callback(self, callback: InferenceErrorCallback) -> None:
        """
        Wire the HotSwapManager's record_inference_error into this provider.

        During probation, every LLM call outcome (success or failure) is
        reported to the RollbackMonitor so it can track the error rate.
        """
        self._inference_error_callback = callback

    def _report_inference_outcome(self, had_error: bool) -> None:
        """Report whether an LLM call produced a parse/validation error."""
        if self._inference_error_callback:
            self._inference_error_callback(had_error)

    def set_metabolic_callback(self, callback: MetabolicCallback) -> None:
        """
        Wire the MetabolicTracker's log_usage into this provider.

        Called after Synapse is initialized in main.py to avoid a circular
        dependency (LLM client is created before Synapse).

        Example::

            llm_client.set_metabolic_callback(synapse.metabolism.log_usage)
        """
        self._metabolic_callback = callback

    # ─── Internal helpers ────────────────────────────────────────

    def _detect_provider_tag(self) -> str:
        """Detect the provider tag from the inner LLMProvider type."""
        from clients.llm import VLLMProvider, OllamaProvider, BedrockProvider

        if isinstance(self._inner, VLLMProvider):
            return "vllm"
        if isinstance(self._inner, OllamaProvider):
            return "ollama"
        if isinstance(self._inner, BedrockProvider):
            return "bedrock"
        # Default: Anthropic direct or OpenAI - use default pricing
        return ""

    def _report_metabolic_cost(
        self,
        caller_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """
        Forward token usage to MetabolicTracker (if wired) and emit a
        structlog event so fiat cost is visible in the terminal.

        Skips zero-token calls (e.g., cache hits) - nothing to charge.
        """
        if not self._metabolic_callback or (input_tokens == 0 and output_tokens == 0):
            return
        provider = self._detect_provider_tag()
        cost_usd = self._metabolic_callback(caller_id, input_tokens, output_tokens, provider)
        self._logger.debug(
            "metabolic_cost_logged",
            system=caller_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=provider or "default",
            cost_usd=round(cost_usd, 6),
        )

    # ─── Public API for Systems ──────────────────────────────────

    async def should_use_llm(
        self,
        system: str,
        estimated_tokens: int = 500,
    ) -> bool:
        """
        Check if a system should make an LLM call given the current budget tier.

        Systems should call this before their LLM call and route to
        heuristics/fallbacks when this returns False.

        Returns True for critical systems regardless of budget.
        """
        if self._budget is None:
            return True

        priority = SYSTEM_PRIORITY.get(system, "standard")

        # Critical systems always use LLM
        if priority == "critical":
            return True

        tier = (await self._budget.get_status()).tier

        if priority == "low" and tier in (BudgetTier.YELLOW, BudgetTier.RED):
            self._logger.debug(
                "llm_skipped_budget",
                system=system,
                tier=tier.value,
                priority=priority,
            )
            return False

        if priority == "standard" and tier == BudgetTier.RED:
            self._logger.debug(
                "llm_skipped_budget",
                system=system,
                tier=tier.value,
                priority=priority,
            )
            return False

        return await self._budget.can_use_llm(estimated_tokens)

    async def get_budget_tier(self) -> BudgetTier:
        """Return current budget tier (GREEN/YELLOW/RED)."""
        if self._budget is None:
            return BudgetTier.GREEN
        return (await self._budget.get_status()).tier

    # ─── LLMProvider Interface ───────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
        *,
        cache_system: str = "unknown",
        cache_method: str = "generate",
    ) -> LLMResponse:
        """
        Generate with caching, budget, and metrics.

        Extra kwargs cache_system and cache_method are used for cache key
        computation and metrics tagging. They are silently ignored by
        callers that don't know about them (duck typing).
        """
        t_start = time.monotonic()

        # Build cache key from the full prompt content
        cache_prompt = self._build_cache_key_text(system_prompt, messages)
        ttl = SYSTEM_TTL.get(cache_system, 300)

        # ── Budget gate ──
        # Enforce tier-based degradation before hitting the cache or the LLM.
        # Critical systems (equor.*) always pass; standard systems are blocked
        # in RED; low-priority systems in YELLOW and RED.
        if self._budget is not None and not await self.should_use_llm(cache_system):
            self._logger.debug(
                "generate_budget_blocked",
                system=cache_system,
                tier=(await self._budget.get_status()).tier.value,
            )
            return LLMResponse(
                text="",
                model="budget_blocked",
                input_tokens=0,
                output_tokens=0,
                finish_reason="budget_blocked",
            )

        # ── Cache check ──
        if self._cache and ttl > 0:
            cached = await self._cache.get(cache_system, cache_method, cache_prompt)
            if cached is not None:
                latency_ms = (time.monotonic() - t_start) * 1000
                if self._metrics:
                    self._metrics.record_call(
                        system=cache_system,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        cache_hit=True,
                    )
                self._logger.debug(
                    "cache_hit",
                    system=cache_system,
                    method=cache_method,
                    latency_ms=round(latency_ms, 2),
                )
                return LLMResponse(
                    text=str(cached),
                    model="cache",
                    input_tokens=0,
                    output_tokens=0,
                    finish_reason="cache_hit",
                )

        # ── LLM call ──
        response = await self._inner.generate(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            output_format=output_format,
        )

        latency_ms = (time.monotonic() - t_start) * 1000

        # ── Probation error tracking ──
        # During hot-swap probation, report whether the response is valid.
        # JSON parse failures are the primary signal for catastrophic forgetting.
        if self._inference_error_callback and output_format == "json":
            import json as _json
            had_error = False
            if response.text:
                try:
                    _json.loads(response.text)
                except _json.JSONDecodeError:
                    had_error = True
            else:
                had_error = True  # Empty response = failure
            self._report_inference_outcome(had_error)
        elif self._inference_error_callback:
            # Non-JSON calls: report success (empty text is not necessarily an error)
            self._report_inference_outcome(not bool(response.text))

        # ── Metrics ──
        if self._metrics:
            self._metrics.record_call(
                system=cache_system,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=latency_ms,
                cache_hit=False,
            )

        # ── Metabolic cost ──
        self._report_metabolic_cost(cache_system, response.input_tokens, response.output_tokens)

        # ── Cache store ──
        if self._cache and ttl > 0 and response.text:
            await self._cache.set(
                cache_system,
                cache_method,
                cache_prompt,
                response.text,
                ttl_seconds=ttl,
            )

        return response

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        *,
        cache_system: str = "unknown",
        cache_method: str = "evaluate",
    ) -> LLMResponse:
        """
        Evaluate with caching, budget, and metrics.
        """
        t_start = time.monotonic()
        ttl = SYSTEM_TTL.get(cache_system, 300)

        # ── Budget gate ──
        if self._budget is not None and not await self.should_use_llm(cache_system):
            self._logger.debug(
                "evaluate_budget_blocked",
                system=cache_system,
                tier=(await self._budget.get_status()).tier.value,
            )
            return LLMResponse(
                text="",
                model="budget_blocked",
                input_tokens=0,
                output_tokens=0,
                finish_reason="budget_blocked",
            )

        # ── Cache check ──
        if self._cache and ttl > 0:
            cached = await self._cache.get(cache_system, cache_method, prompt)
            if cached is not None:
                latency_ms = (time.monotonic() - t_start) * 1000
                if self._metrics:
                    self._metrics.record_call(
                        system=cache_system,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        cache_hit=True,
                    )
                return LLMResponse(
                    text=str(cached),
                    model="cache",
                    input_tokens=0,
                    output_tokens=0,
                    finish_reason="cache_hit",
                )

        # ── LLM call ──
        response = await self._inner.evaluate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.monotonic() - t_start) * 1000

        # ── Metrics ──
        if self._metrics:
            self._metrics.record_call(
                system=cache_system,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=latency_ms,
                cache_hit=False,
            )

        # ── Metabolic cost ──
        self._report_metabolic_cost(cache_system, response.input_tokens, response.output_tokens)

        # ── Cache store ──
        if self._cache and ttl > 0 and response.text:
            await self._cache.set(
                cache_system,
                cache_method,
                prompt,
                response.text,
                ttl_seconds=ttl,
            )

        return response

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
        *,
        cache_system: str = "unknown",
    ) -> ToolAwareResponse:
        """
        Tool-use calls are NOT cached (non-deterministic, stateful).
        Budget and metrics still apply.
        """
        t_start = time.monotonic()

        response = await self._inner.generate_with_tools(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.monotonic() - t_start) * 1000

        if self._metrics:
            self._metrics.record_call(
                system=cache_system,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=latency_ms,
                cache_hit=False,
            )

        # ── Metabolic cost ──
        self._report_metabolic_cost(cache_system, response.input_tokens, response.output_tokens)

        return response

    async def close(self) -> None:
        await self._inner.close()

    # ─── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_cache_key_text(
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        """Build a deterministic string for cache key hashing."""
        parts = [system_prompt]
        for m in messages:
            parts.append(f"{m.role}:{m.content}")
        return "\n".join(parts)

    @property
    def validator(self) -> OutputValidator:
        """Expose the output validator for direct use by systems."""
        return self._validator

    @property
    def cache(self) -> PromptCache | None:
        """Expose the cache for direct use when systems need custom logic."""
        return self._cache

    @property
    def budget(self) -> TokenBudget | None:
        """Expose the budget for direct use."""
        return self._budget

    @property
    def metrics(self) -> LLMMetricsCollector | None:
        """Expose metrics collector."""
        return self._metrics

    @property
    def inner(self) -> LLMProvider:
        """Access the underlying unwrapped provider."""
        return self._inner

    # ─── Adapter Delegation ──────────────────────────────────

    @property
    def supports_adapters(self) -> bool:
        result: bool = self._inner.supports_adapters
        return result

    @property
    def active_adapter_id(self) -> str | None:
        result: str | None = self._inner.active_adapter_id
        return result

    async def load_adapter(self, adapter_path: str, adapter_id: str) -> None:
        await self._inner.load_adapter(adapter_path, adapter_id)

    async def unload_adapter(self) -> None:
        await self._inner.unload_adapter()
