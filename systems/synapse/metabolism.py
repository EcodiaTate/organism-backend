"""
EcodiaOS — Synapse Metabolic Tracker

Tracks the real-world fiat cost of the organism's LLM API usage.
Every LLM call (Nova thinking, Simula coding, etc.) burns tokens that
have a direct financial cost. This module converts token consumption
into USD cost and maintains a rolling deficit — the cumulative amount
the organism has "spent" since its last revenue injection.

Performance contract: all hot-path methods (log_usage, snapshot) must
complete in < 0.1ms to respect the 100-200ms Theta cycle budget.
Achieved via:
  - No locks (single-threaded asyncio event loop)
  - No allocations on the hot path (pre-allocated accumulators)
  - O(1) EMA-based burn rate (no window trimming)
  - Snapshot is a cheap read of scalar fields

Pricing is based on Anthropic Claude 3.5 Sonnet at time of writing.
Override via constructor for newer models or mixed-model usage.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from systems.synapse.types import MetabolicSnapshot

logger = structlog.get_logger("systems.synapse.metabolism")

# ─── Default Pricing (Claude 3.5 Sonnet, USD per token) ─────────────

_DEFAULT_INPUT_PRICE_PER_TOKEN: float = 3.00 / 1_000_000    # $3.00 / 1M input
_DEFAULT_OUTPUT_PRICE_PER_TOKEN: float = 15.00 / 1_000_000  # $15.00 / 1M output

# EMA smoothing factor for burn rate (higher = more responsive)
_BURN_RATE_ALPHA: float = 0.15

# Minimum interval between burn-rate updates to avoid divide-by-zero (seconds)
_MIN_RATE_INTERVAL_S: float = 0.001


class MetabolicTracker:
    """
    Tracks the financial metabolism of the organism's LLM API usage.

    Thread-safety: NOT thread-safe. Designed to run exclusively on the
    asyncio event loop (same as the Theta cycle). All mutations happen
    in the single-threaded callback path.

    Usage:
        tracker = MetabolicTracker()
        tracker.log_usage("nova", input_tokens=1200, output_tokens=350)
        snapshot = tracker.snapshot()
        print(f"Deficit: ${snapshot.rolling_deficit_usd:.4f}")
    """

    def __init__(
        self,
        input_price_per_token: float = _DEFAULT_INPUT_PRICE_PER_TOKEN,
        output_price_per_token: float = _DEFAULT_OUTPUT_PRICE_PER_TOKEN,
    ) -> None:
        self._input_price = input_price_per_token
        self._output_price = output_price_per_token
        self._logger = logger.bind(component="metabolic_tracker")

        # ── Accumulators (monotonically increasing until inject_revenue) ──
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_calls: int = 0
        self._total_cost_usd: float = 0.0

        # Per-subsystem cost breakdown (id → cumulative USD)
        self._per_system_cost: dict[str, float] = {}

        # Rolling deficit: total cost minus total revenue injected
        self._total_revenue_usd: float = 0.0

        # ── Burn rate (EMA-smoothed) ──
        self._burn_rate_usd_per_sec: float = 0.0
        self._last_log_time: float = time.monotonic()

        # ── Window tracking (for periodic snapshot diff) ──
        self._window_cost_usd: float = 0.0

        self._logger.info(
            "metabolic_tracker_initialized",
            input_price_per_1m=round(input_price_per_token * 1_000_000, 2),
            output_price_per_1m=round(output_price_per_token * 1_000_000, 2),
        )

    # ─── Hot Path: Log Token Usage ────────────────────────────────────

    def log_usage(
        self,
        caller_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Record an LLM call's token consumption and compute its fiat cost.

        Args:
            caller_id: Identifier of the subsystem making the call
                (e.g. "nova", "simula", "evo").
            input_tokens: Number of input/prompt tokens consumed.
            output_tokens: Number of output/completion tokens consumed.

        Returns the cost in USD for this single call.

        Performance: O(1), no allocations, no locks. ~10us typical.
        """
        cost = (
            input_tokens * self._input_price
            + output_tokens * self._output_price
        )

        # Accumulate totals
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_calls += 1
        self._total_cost_usd += cost
        self._window_cost_usd += cost

        # Per-caller accumulator (dict get+set, no defaultdict overhead)
        prev = self._per_system_cost.get(caller_id, 0.0)
        self._per_system_cost[caller_id] = prev + cost

        # Update EMA burn rate
        now = time.monotonic()
        dt = now - self._last_log_time
        if dt >= _MIN_RATE_INTERVAL_S:
            instantaneous_rate = cost / dt
            self._burn_rate_usd_per_sec = (
                _BURN_RATE_ALPHA * instantaneous_rate
                + (1.0 - _BURN_RATE_ALPHA) * self._burn_rate_usd_per_sec
            )
            self._last_log_time = now

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

    # ─── Snapshot (cheap read) ────────────────────────────────────────

    @property
    def rolling_deficit_usd(self) -> float:
        """Net fiat deficit: total cost minus total revenue."""
        return self._total_cost_usd - self._total_revenue_usd

    @property
    def burn_rate_usd_per_hour(self) -> float:
        """Current EMA-smoothed burn rate in USD/hour."""
        return self._burn_rate_usd_per_sec * 3600.0

    def snapshot(self, available_balance_usd: float | None = None) -> MetabolicSnapshot:
        """
        Build a point-in-time metabolic snapshot.

        Args:
            available_balance_usd: If provided, computes hours_until_depleted
                based on current burn rate. Typically sourced from WalletClient.

        Performance: O(n) where n = number of callers (typically 9). ~20us.
        """
        hours_until_depleted = float("inf")
        if (
            available_balance_usd is not None
            and self._burn_rate_usd_per_sec > 0
        ):
            seconds_left = available_balance_usd / self._burn_rate_usd_per_sec
            hours_until_depleted = seconds_left / 3600.0

        return MetabolicSnapshot(
            rolling_deficit_usd=round(self.rolling_deficit_usd, 8),
            window_cost_usd=round(self._window_cost_usd, 8),
            per_system_cost_usd={
                sid: round(c, 8) for sid, c in self._per_system_cost.items()
            },
            burn_rate_usd_per_sec=round(self._burn_rate_usd_per_sec, 10),
            burn_rate_usd_per_hour=round(self._burn_rate_usd_per_sec * 3600.0, 6),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_calls=self._total_calls,
            hours_until_depleted=round(hours_until_depleted, 2),
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
            "total_cost_usd": round(self._total_cost_usd, 6),
            "total_revenue_usd": round(self._total_revenue_usd, 6),
            "burn_rate_usd_per_hour": round(
                self._burn_rate_usd_per_sec * 3600.0, 4,
            ),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_calls": self._total_calls,
            "per_system_cost_usd": {
                sid: round(c, 6) for sid, c in self._per_system_cost.items()
            },
        }
