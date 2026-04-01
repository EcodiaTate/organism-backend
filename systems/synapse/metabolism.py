"""
EcodiaOS - Synapse Metabolic Tracker

Tracks the real-world fiat cost of the organism's existence. Two cost
categories:

  1. **API costs** - per-token charges for external LLM calls (Claude via
     Bedrock). Local inference (RE on vLLM) has zero API cost.
  2. **Infrastructure costs** - compute rental (RunPod GPU pods), queried
     autonomously via the RunPod GraphQL API.

The organism must know its true burn rate to make survival decisions.
Pricing a local vLLM call at Claude Sonnet rates was producing a phantom
$224/day burn - the real cost is the pod rental (~$0.50–1.50/hr).

Performance contract: all hot-path methods (log_usage, snapshot) must
complete in < 0.1ms to respect the 100-200ms Theta cycle budget.
Achieved via:
  - No locks (single-threaded asyncio event loop)
  - No allocations on the hot path (pre-allocated accumulators)
  - O(1) EMA-based burn rate (no window trimming)
  - Snapshot is a cheap read of scalar fields
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from systems.synapse.types import MetabolicSnapshot

logger = structlog.get_logger("systems.synapse.metabolism")

# ─── Default Pricing (USD per token) ──────────────────────────────────
# Claude Sonnet 4 via Bedrock
_SONNET_INPUT_PRICE: float = 3.00 / 1_000_000   # $3.00 / 1M input
_SONNET_OUTPUT_PRICE: float = 15.00 / 1_000_000  # $15.00 / 1M output
# Claude Opus 4 via Bedrock
_OPUS_INPUT_PRICE: float = 15.00 / 1_000_000     # $15.00 / 1M input
_OPUS_OUTPUT_PRICE: float = 75.00 / 1_000_000    # $75.00 / 1M output
# Claude Haiku 3.5 via Bedrock (default LLM)
_HAIKU_INPUT_PRICE: float = 0.80 / 1_000_000     # $0.80 / 1M input
_HAIKU_OUTPUT_PRICE: float = 4.00 / 1_000_000    # $4.00 / 1M output

# Provider → (input_price, output_price)
_PROVIDER_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet": (_SONNET_INPUT_PRICE, _SONNET_OUTPUT_PRICE),
    "claude-opus": (_OPUS_INPUT_PRICE, _OPUS_OUTPUT_PRICE),
    "claude-haiku": (_HAIKU_INPUT_PRICE, _HAIKU_OUTPUT_PRICE),
    "bedrock": (_HAIKU_INPUT_PRICE, _HAIKU_OUTPUT_PRICE),
    # Local inference - zero API cost (compute cost tracked separately)
    "re": (0.0, 0.0),
    "vllm": (0.0, 0.0),
    "ollama": (0.0, 0.0),
    "local": (0.0, 0.0),
}

# EMA smoothing factor for burn rate (higher = more responsive)
_BURN_RATE_ALPHA: float = 0.15

# Minimum interval between burn-rate updates to avoid divide-by-zero (seconds)
_MIN_RATE_INTERVAL_S: float = 0.001


class MetabolicTracker:
    """
    Tracks the financial metabolism of the organism.

    Two cost dimensions:
      - **API costs**: per-token charges for external LLM calls. Local
        inference (RE/vLLM) is $0 API cost.
      - **Infrastructure costs**: compute rental (GPU pods), polled
        autonomously from the provider API.

    Thread-safety: NOT thread-safe. Designed to run exclusively on the
    asyncio event loop (same as the Theta cycle).
    """

    def __init__(
        self,
        input_price_per_token: float = _HAIKU_INPUT_PRICE,
        output_price_per_token: float = _HAIKU_OUTPUT_PRICE,
    ) -> None:
        # Legacy fallback pricing (used when no provider tag)
        self._default_input_price = input_price_per_token
        self._default_output_price = output_price_per_token
        self._logger = logger.bind(component="metabolic_tracker")

        # ── API cost accumulators ──
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_calls: int = 0
        self._total_api_cost_usd: float = 0.0

        # Per-subsystem cost breakdown (id → cumulative USD)
        self._per_system_cost: dict[str, float] = {}
        # Per-provider cost breakdown (provider → cumulative USD)
        self._per_provider_cost: dict[str, float] = {}

        # Rolling deficit: total cost minus total revenue injected
        self._total_revenue_usd: float = 0.0

        # ── API burn rate (EMA-smoothed) ──
        self._api_burn_rate_usd_per_sec: float = 0.0
        self._last_log_time: float = time.monotonic()

        # ── Window tracking (for periodic snapshot diff) ──
        self._window_cost_usd: float = 0.0

        # ── Infrastructure cost tracking ──
        self._infra_cost_usd_per_hour: float = 0.0
        self._infra_cost_source: str = "none"
        self._infra_total_cost_usd: float = 0.0
        self._infra_last_update: float = 0.0
        # Per-resource breakdown (e.g. "runpod:pod_abc123" → cost/hr)
        self._infra_resources: dict[str, float] = {}

        self._logger.info(
            "metabolic_tracker_initialized",
            default_input_price_per_1m=round(input_price_per_token * 1_000_000, 2),
            default_output_price_per_1m=round(output_price_per_token * 1_000_000, 2),
            known_providers=list(_PROVIDER_PRICING.keys()),
        )

    # ─── Hot Path: Log Token Usage ────────────────────────────────────

    def log_usage(
        self,
        caller_id: str,
        input_tokens: int,
        output_tokens: int,
        provider: str = "",
    ) -> float:
        """
        Record an LLM call's token consumption and compute its fiat cost.

        Args:
            caller_id: Identifier of the subsystem making the call
                (e.g. "nova", "simula", "evo").
            input_tokens: Number of input/prompt tokens consumed.
            output_tokens: Number of output/completion tokens consumed.
            provider: LLM provider tag (e.g. "re", "claude-sonnet",
                "bedrock"). If empty, uses default pricing.

        Returns the cost in USD for this single call.

        Performance: O(1), no allocations, no locks. ~10us typical.
        """
        # Resolve pricing based on provider
        if provider and provider in _PROVIDER_PRICING:
            in_price, out_price = _PROVIDER_PRICING[provider]
        else:
            in_price = self._default_input_price
            out_price = self._default_output_price

        cost = (
            input_tokens * in_price
            + output_tokens * out_price
        )

        # Accumulate totals
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_calls += 1
        self._total_api_cost_usd += cost
        self._window_cost_usd += cost

        # Per-caller accumulator
        prev = self._per_system_cost.get(caller_id, 0.0)
        self._per_system_cost[caller_id] = prev + cost

        # Per-provider accumulator
        prov_key = provider or "default"
        prev_prov = self._per_provider_cost.get(prov_key, 0.0)
        self._per_provider_cost[prov_key] = prev_prov + cost

        # Update EMA burn rate (API only)
        now = time.monotonic()
        dt = now - self._last_log_time
        if dt >= _MIN_RATE_INTERVAL_S:
            instantaneous_rate = cost / dt
            self._api_burn_rate_usd_per_sec = (
                _BURN_RATE_ALPHA * instantaneous_rate
                + (1.0 - _BURN_RATE_ALPHA) * self._api_burn_rate_usd_per_sec
            )
            self._last_log_time = now

        return cost

    # ─── Infrastructure Cost Tracking ─────────────────────────────────

    def update_infrastructure_cost(
        self,
        resource_id: str,
        cost_per_hour_usd: float,
        source: str = "runpod",
    ) -> None:
        """
        Update the known infrastructure cost for a compute resource.

        Called by the InfrastructureCostPoller when it queries the
        provider API. Multiple resources can be tracked simultaneously
        (e.g. multiple GPU pods).

        Args:
            resource_id: Unique identifier (e.g. "runpod:abc123").
            cost_per_hour_usd: Current hourly rate for this resource.
            source: Provider name for logging.
        """
        old = self._infra_resources.get(resource_id, 0.0)
        self._infra_resources[resource_id] = cost_per_hour_usd
        self._infra_cost_usd_per_hour = sum(self._infra_resources.values())
        self._infra_cost_source = source
        self._infra_last_update = time.monotonic()

        if abs(old - cost_per_hour_usd) > 0.001:
            self._logger.info(
                "infrastructure_cost_updated",
                resource_id=resource_id,
                cost_per_hour_usd=round(cost_per_hour_usd, 4),
                total_infra_per_hour_usd=round(self._infra_cost_usd_per_hour, 4),
                source=source,
            )

    def remove_infrastructure_resource(self, resource_id: str) -> None:
        """Remove a resource that is no longer running."""
        if resource_id in self._infra_resources:
            del self._infra_resources[resource_id]
            self._infra_cost_usd_per_hour = sum(self._infra_resources.values())
            self._logger.info(
                "infrastructure_resource_removed",
                resource_id=resource_id,
                total_infra_per_hour_usd=round(self._infra_cost_usd_per_hour, 4),
            )

    def accrue_infrastructure_cost(self, elapsed_seconds: float) -> float:
        """
        Accrue infrastructure cost for a time period.
        Called periodically by the poller to keep the deficit accurate.

        Returns the cost accrued.
        """
        if self._infra_cost_usd_per_hour <= 0:
            return 0.0
        cost = (self._infra_cost_usd_per_hour / 3600.0) * elapsed_seconds
        self._infra_total_cost_usd += cost
        self._window_cost_usd += cost
        return cost

    # ─── Revenue Injection ────────────────────────────────────────────

    def inject_revenue(self, amount_usd: float) -> None:
        """
        Record incoming revenue (e.g., from wallet top-up or earned fees).
        Reduces the rolling deficit.
        """
        self._total_revenue_usd += amount_usd
        self._logger.info(
            "revenue_injected",
            amount_usd=round(amount_usd, 6),
            new_deficit_usd=round(self.rolling_deficit_usd, 6),
        )

    # ─── Computed Properties ──────────────────────────────────────────

    @property
    def total_cost_usd(self) -> float:
        """Total cost: API + infrastructure."""
        return self._total_api_cost_usd + self._infra_total_cost_usd

    @property
    def rolling_deficit_usd(self) -> float:
        """Net fiat deficit: total cost (API + infra) minus total revenue."""
        return self.total_cost_usd - self._total_revenue_usd

    @property
    def burn_rate_usd_per_hour(self) -> float:
        """Total burn rate: API EMA + infrastructure hourly."""
        return (
            self._api_burn_rate_usd_per_sec * 3600.0
            + self._infra_cost_usd_per_hour
        )

    @property
    def api_burn_rate_usd_per_hour(self) -> float:
        """API-only burn rate (EMA-smoothed)."""
        return self._api_burn_rate_usd_per_sec * 3600.0

    @property
    def infra_cost_usd_per_hour(self) -> float:
        """Infrastructure-only burn rate."""
        return self._infra_cost_usd_per_hour

    # ─── Snapshot ─────────────────────────────────────────────────────

    def snapshot(self, available_balance_usd: float | None = None) -> MetabolicSnapshot:
        """
        Build a point-in-time metabolic snapshot.

        Performance: O(n) where n = number of callers (typically 9). ~20us.
        """
        total_burn_per_sec = (
            self._api_burn_rate_usd_per_sec
            + self._infra_cost_usd_per_hour / 3600.0
        )

        hours_until_depleted = float("inf")
        if available_balance_usd is not None and total_burn_per_sec > 0:
            seconds_left = available_balance_usd / total_burn_per_sec
            hours_until_depleted = seconds_left / 3600.0

        return MetabolicSnapshot(
            rolling_deficit_usd=round(self.rolling_deficit_usd, 8),
            window_cost_usd=round(self._window_cost_usd, 8),
            per_system_cost_usd={
                sid: round(c, 8) for sid, c in self._per_system_cost.items()
            },
            burn_rate_usd_per_sec=round(total_burn_per_sec, 10),
            burn_rate_usd_per_hour=round(total_burn_per_sec * 3600.0, 6),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_calls=self._total_calls,
            hours_until_depleted=round(hours_until_depleted, 2),
            # Extended fields
            api_cost_usd_per_hour=round(
                self._api_burn_rate_usd_per_sec * 3600.0, 6,
            ),
            infra_cost_usd_per_hour=round(self._infra_cost_usd_per_hour, 6),
            total_api_cost_usd=round(self._total_api_cost_usd, 8),
            total_infra_cost_usd=round(self._infra_total_cost_usd, 8),
            per_provider_cost_usd={
                pid: round(c, 8) for pid, c in self._per_provider_cost.items()
            },
            infra_resources={
                rid: round(c, 4) for rid, c in self._infra_resources.items()
            },
        )

    def reset_window(self) -> None:
        """
        Reset the per-window cost accumulator. Called periodically by
        SynapseService to produce per-interval deltas.
        """
        self._window_cost_usd = 0.0

    # ─── Stats ────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Expose metabolic state for Synapse.stats aggregation."""
        return {
            "rolling_deficit_usd": round(self.rolling_deficit_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_api_cost_usd": round(self._total_api_cost_usd, 6),
            "total_infra_cost_usd": round(self._infra_total_cost_usd, 6),
            "total_revenue_usd": round(self._total_revenue_usd, 6),
            "burn_rate_usd_per_hour": round(self.burn_rate_usd_per_hour, 4),
            "api_burn_rate_usd_per_hour": round(
                self._api_burn_rate_usd_per_sec * 3600.0, 4,
            ),
            "infra_cost_usd_per_hour": round(self._infra_cost_usd_per_hour, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_calls": self._total_calls,
            "per_system_cost_usd": {
                sid: round(c, 6) for sid, c in self._per_system_cost.items()
            },
            "per_provider_cost_usd": {
                pid: round(c, 6) for pid, c in self._per_provider_cost.items()
            },
            "infra_resources": {
                rid: round(c, 4) for rid, c in self._infra_resources.items()
            },
        }
