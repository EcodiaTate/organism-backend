"""
EcodiaOS - Token Budget System

Tracks LLM token consumption per cycle and per hour.
Implements three-tier degradation: Green (normal) → Yellow (careful) → Red (critical).

Systems check budget status before making LLM calls and degrade gracefully
when approaching limits. Evo learns optimal EFE weights given budget constraints.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum

import structlog

logger = structlog.get_logger()


class BudgetTier(StrEnum):
    """Budget utilization tier determines which systems can use LLM."""
    GREEN = "green"      # 0–70% of limit: All systems active
    YELLOW = "yellow"    # 70–90% of limit: Low-priority systems degrade to heuristics
    RED = "red"          # 90–100%: Only critical systems, all others use heuristics


@dataclass
class BudgetStatus:
    """Current budget state and tier classification."""
    tokens_used: int                    # Tokens consumed this hour
    calls_made: int                     # LLM calls made this hour
    tokens_remaining: int               # Budget - used
    calls_remaining: int
    tier: BudgetTier
    tokens_per_sec: float               # Burn rate (tokens/s)
    calls_per_sec: float                # Call rate (calls/s)
    hours_until_exhausted: float        # At current burn rate
    warning_message: str | None = None  # If approaching limit


class TokenBudget:
    """
    Tracks cumulative token and call usage over a rolling 1-hour window.

    Thread-safe. Emits telemetry events on threshold crossing.
    Supports soft limits (logging) and hard limits (requests rejected).
    """

    def __init__(
        self,
        max_tokens_per_hour: int = 600_000,
        max_calls_per_hour: int = 1_000,
        hard_limit: bool = False,
    ) -> None:
        """
        Args:
            max_tokens_per_hour: Soft budget (warning at 70%, 90%)
            max_calls_per_hour: Soft budget (warning at 70%, 90%)
            hard_limit: If True, reject requests exceeding budget (fail-fast)
                       If False, allow overage but log (graceful degradation)
        """
        self._max_tokens = max_tokens_per_hour
        self._max_calls = max_calls_per_hour
        self._hard_limit = hard_limit

        # Rolling window: (timestamp, tokens, calls) tuples
        self._usage_window: list[tuple[float, int, int]] = []
        self._lock = asyncio.Lock()
        self._last_warning_tier: BudgetTier | None = None

        # Last known status snapshot - updated by charge() and get_status().
        # Allows synchronous callers (e.g. Soma interoceptor) to read tier
        # without awaiting the lock.
        self._last_status: BudgetStatus | None = None

    def _trim_window(self, now: float) -> None:
        """Remove entries older than 1 hour."""
        cutoff = now - 3600.0
        self._usage_window = [
            (ts, t, c) for ts, t, c in self._usage_window if ts >= cutoff
        ]

    def _compute_usage(self, now: float) -> tuple[int, int]:
        """Sum tokens and calls in the current window."""
        self._trim_window(now)
        total_tokens = sum(t for _, t, _ in self._usage_window)
        total_calls = sum(c for _, _, c in self._usage_window)
        return total_tokens, total_calls

    async def can_use_llm(
        self,
        estimated_tokens: int,
        estimated_calls: int = 1,
    ) -> bool:
        """
        Check if the system can make an LLM call of estimated cost.

        In hard-limit mode: return False if would exceed budget.
        In soft-limit mode: always return True (allow overage with warning).
        """
        async with self._lock:
            now = time.time()
            tokens_used, calls_used = self._compute_usage(now)

            would_exceed_tokens = tokens_used + estimated_tokens > self._max_tokens
            would_exceed_calls = calls_used + estimated_calls > self._max_calls

            if self._hard_limit and (would_exceed_tokens or would_exceed_calls):
                logger.warning(
                    "token_budget_hard_limit_exceeded",
                    tokens_used=tokens_used,
                    tokens_budget=self._max_tokens,
                    calls_used=calls_used,
                    calls_budget=self._max_calls,
                )
                return False

            return True

    async def charge(
        self,
        tokens: int,
        calls: int = 1,
        system: str = "unknown",
    ) -> None:
        """
        Record an LLM call's token consumption.
        Emit telemetry and check tier thresholds.
        """
        async with self._lock:
            now = time.time()
            self._usage_window.append((now, tokens, calls))

            tokens_used, calls_used = self._compute_usage(now)
            tier = self._classify_tier(tokens_used, calls_used)

            # Emit telemetry
            logger.info(
                "llm_call_charged",
                system=system,
                tokens=tokens,
                calls=calls,
                tokens_used_total=tokens_used,
                calls_used_total=calls_used,
                budget_tier=tier,
            )

            # Warn on tier transition
            if tier != self._last_warning_tier:
                self._emit_tier_warning(tier, tokens_used, calls_used)
                self._last_warning_tier = tier

            # Keep cached_status fresh so synchronous callers don't read stale data.
            # A lightweight snapshot suffices here - burn-rate fields are omitted for speed.
            self._last_status = BudgetStatus(
                tokens_used=tokens_used,
                calls_made=calls_used,
                tokens_remaining=max(0, self._max_tokens - tokens_used),
                calls_remaining=max(0, self._max_calls - calls_used),
                tier=tier,
                tokens_per_sec=0.0,
                calls_per_sec=0.0,
                hours_until_exhausted=float("inf"),
            )

    async def get_status(self) -> BudgetStatus:
        """Return current budget state."""
        async with self._lock:
            now = time.time()
            tokens_used, calls_used = self._compute_usage(now)

            tier = self._classify_tier(tokens_used, calls_used)
            tokens_remaining = max(0, self._max_tokens - tokens_used)
            calls_remaining = max(0, self._max_calls - calls_used)

            # Compute burn rates (tokens/sec, calls/sec)
            window_size_s = 3600.0
            if self._usage_window:
                window_size_s = now - self._usage_window[0][0]
                window_size_s = max(1.0, window_size_s)  # Avoid division by zero

            tokens_per_sec = tokens_used / window_size_s if window_size_s > 0 else 0.0
            calls_per_sec = calls_used / window_size_s if window_size_s > 0 else 0.0

            # Estimate hours until exhausted
            hours_until_exhausted = float('inf')
            if tokens_per_sec > 0:
                seconds_remaining = tokens_remaining / tokens_per_sec
                hours_until_exhausted = min(
                    hours_until_exhausted,
                    seconds_remaining / 3600.0,
                )
            if calls_per_sec > 0:
                seconds_remaining = calls_remaining / calls_per_sec
                hours_until_exhausted = min(
                    hours_until_exhausted,
                    seconds_remaining / 3600.0,
                )

            warning = None
            if tier == BudgetTier.YELLOW:
                warning = (
                    f"Budget tier YELLOW: {tokens_used}/{self._max_tokens} tokens "
                    f"({int(100 * tokens_used / self._max_tokens)}%). "
                    f"Estimated {hours_until_exhausted:.1f} hours until Red tier."
                )
            elif tier == BudgetTier.RED:
                warning = (
                    f"Budget tier RED: {tokens_used}/{self._max_tokens} tokens. "
                    f"Only critical systems active. Consider slowing cycle or increasing budget."
                )

            status = BudgetStatus(
                tokens_used=tokens_used,
                calls_made=calls_used,
                tokens_remaining=tokens_remaining,
                calls_remaining=calls_remaining,
                tier=tier,
                tokens_per_sec=tokens_per_sec,
                calls_per_sec=calls_per_sec,
                hours_until_exhausted=hours_until_exhausted,
                warning_message=warning,
            )
            self._last_status = status
            return status

    def _classify_tier(self, tokens_used: int, calls_used: int) -> BudgetTier:
        """Classify budget tier based on utilization."""
        token_util = tokens_used / self._max_tokens
        calls_util = calls_used / self._max_calls

        # Take the more conservative utilization
        max_util = max(token_util, calls_util)

        if max_util >= 0.90:
            return BudgetTier.RED
        elif max_util >= 0.70:
            return BudgetTier.YELLOW
        else:
            return BudgetTier.GREEN

    def _emit_tier_warning(
        self,
        tier: BudgetTier,
        tokens_used: int,
        calls_used: int,
    ) -> None:
        """Log a warning when tier changes."""
        if tier == BudgetTier.YELLOW:
            logger.warning(
                "budget_tier_yellow",
                tokens_used=tokens_used,
                tokens_budget=self._max_tokens,
                calls_used=calls_used,
                calls_budget=self._max_calls,
                message="Entering YELLOW tier. Low-priority LLM calls will be disabled.",
            )
        elif tier == BudgetTier.RED:
            logger.warning(
                "budget_tier_red",
                tokens_used=tokens_used,
                tokens_budget=self._max_tokens,
                calls_used=calls_used,
                calls_budget=self._max_calls,
                message="Entering RED tier. Only critical systems active.",
            )

    @property
    def cached_status(self) -> BudgetStatus | None:
        """
        Last known budget status without acquiring the lock.

        Returns None before the first ``get_status()`` or ``charge()`` call.
        Intended for synchronous callers (e.g. Soma interoceptor) that run
        inside a tight theta-cycle budget and cannot await the async lock.
        The value is at most one charge() call stale - acceptable for
        interoceptive sensing, which is already an approximation.
        """
        return self._last_status

    async def reset_window(self) -> None:
        """Clear usage window (testing only)."""
        async with self._lock:
            self._usage_window.clear()
            self._last_warning_tier = None
            self._last_status = None
