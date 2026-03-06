"""
EcodiaOS — LLM Metrics & Cost Telemetry

Aggregates token spend, latency, cache hit rate, and cost.
Emits observability signals for dashboards and alerts.

Key metrics:
- llm_tokens_charged (cumulative)
- llm_cost_estimate (USD)
- llm_cache_hit_rate (%)
- llm_latency_p99 (ms)
- llm_budget_tier (Green/Yellow/Red)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class LLMMetrics:
    """Per-system LLM metrics."""
    system: str
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    total_cost_cents: float = 0.0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.total_latency_ms / self.calls

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def total_cost_usd(self) -> float:
        return self.total_cost_cents / 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p99_latency_ms": self.max_latency_ms,  # Simplified
            "cache_hit_rate": round(self.cache_hit_rate, 3),
        }


class LLMMetricsCollector:
    """
    Collects LLM usage metrics across all systems.

    Tracks:
    - Token consumption (input + output) per system
    - Estimated cost per system
    - Latency distributions
    - Cache hit rate
    - Budget tier status
    """

    # Pricing (Anthropic Claude 3.5 Sonnet, as of Feb 2026)
    PRICING_INPUT_PER_1M_TOKENS_CENTS = 300      # $3.00
    PRICING_OUTPUT_PER_1M_TOKENS_CENTS = 1500    # $15.00

    def __init__(self) -> None:
        self._metrics: dict[str, LLMMetrics] = {}
        self._start_time = time.time()
        self._logger = logger.bind(component="llm_metrics")

    def record_call(
        self,
        system: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cache_hit: bool = False,
    ) -> None:
        """
        Record an LLM call.

        Args:
            system: System name (e.g., 'nova.efe', 'voxis.render')
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            latency_ms: Call latency in milliseconds
            cache_hit: Whether this was served from cache
        """
        if system not in self._metrics:
            self._metrics[system] = LLMMetrics(system=system)

        m = self._metrics[system]
        total = input_tokens + output_tokens

        m.calls += 1
        m.tokens_in += input_tokens
        m.tokens_out += output_tokens
        m.total_tokens += total

        # Compute cost
        input_cost = (input_tokens / 1_000_000) * self.PRICING_INPUT_PER_1M_TOKENS_CENTS
        output_cost = (output_tokens / 1_000_000) * self.PRICING_OUTPUT_PER_1M_TOKENS_CENTS
        m.total_cost_cents += input_cost + output_cost

        # Latency tracking
        m.total_latency_ms += latency_ms
        m.min_latency_ms = min(m.min_latency_ms, latency_ms)
        m.max_latency_ms = max(m.max_latency_ms, latency_ms)

        # Cache tracking
        if cache_hit:
            m.cache_hits += 1
        else:
            m.cache_misses += 1

        self._logger.debug(
            "llm_call_recorded",
            system=system,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            cache_hit=cache_hit,
        )

    def get_system_metrics(self, system: str) -> LLMMetrics | None:
        """Get metrics for a specific system."""
        return self._metrics.get(system)

    def get_all_metrics(self) -> dict[str, LLMMetrics]:
        """Get all system metrics."""
        return dict(self._metrics)

    def get_total_metrics(self) -> LLMMetrics:
        """Get aggregated metrics across all systems."""
        total = LLMMetrics(system="TOTAL")

        for m in self._metrics.values():
            total.calls += m.calls
            total.tokens_in += m.tokens_in
            total.tokens_out += m.tokens_out
            total.total_tokens += m.total_tokens
            total.total_cost_cents += m.total_cost_cents
            total.total_latency_ms += m.total_latency_ms
            total.cache_hits += m.cache_hits
            total.cache_misses += m.cache_misses

        # Update min/max across all systems
        for m in self._metrics.values():
            total.min_latency_ms = min(total.min_latency_ms, m.min_latency_ms)
            total.max_latency_ms = max(total.max_latency_ms, m.max_latency_ms)

        return total

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data suitable for a dashboard/API endpoint."""
        total = self.get_total_metrics()
        uptime_s = time.time() - self._start_time

        return {
            "uptime_seconds": round(uptime_s),
            "total": total.to_dict(),
            "by_system": {k: m.to_dict() for k, m in self._metrics.items()},
            "cost_projection": {
                "current_cost_usd": round(total.total_cost_usd, 2),
                "hourly_cost_usd": round(total.total_cost_usd / max(1, uptime_s / 3600), 2),
                "daily_cost_usd": round((total.total_cost_usd / max(1, uptime_s / 3600)) * 24, 2),
            },
            "efficiency": {
                "avg_tokens_per_call": round(total.total_tokens / max(1, total.calls)),
                "avg_latency_ms": round(total.avg_latency_ms, 2),
                "cache_hit_rate": round(total.cache_hit_rate * 100, 1),
            },
        }

    def reset(self) -> None:
        """Reset all metrics (testing only)."""
        self._metrics.clear()
        self._start_time = time.time()

    def summary(self) -> str:
        """Return a human-readable summary."""
        total = self.get_total_metrics()
        lines = [
            "━━━ LLM Metrics Summary ━━━",
            f"Total calls: {total.calls}",
            f"Total tokens: {total.total_tokens:,}",
            f"Estimated cost: ${total.total_cost_usd:.2f}",
            f"Avg latency: {total.avg_latency_ms:.1f}ms",
            f"Cache hit rate: {total.cache_hit_rate * 100:.1f}%",
            "",
            "By system:",
        ]

        for system in sorted(self._metrics.keys()):
            m = self._metrics[system]
            lines.append(
                f"  {system:30s} | "
                f"{m.calls:3d} calls | "
                f"{m.total_tokens:6,d} tokens | "
                f"${m.total_cost_usd:6.2f} | "
                f"{m.avg_latency_ms:6.1f}ms"
            )

        return "\n".join(lines)


# Global instance (initialized by main)
_collector: LLMMetricsCollector | None = None


def get_collector() -> LLMMetricsCollector:
    """Get the global metrics collector (lazy init)."""
    global _collector
    if _collector is None:
        _collector = LLMMetricsCollector()
    return _collector


def record_llm_call(
    system: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    cache_hit: bool = False,
) -> None:
    """Convenience function to record a call."""
    get_collector().record_call(
        system=system,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
    )
