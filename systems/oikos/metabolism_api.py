"""
EcodiaOS - Oikos Metabolism API (MVP Economic Engine)

THE SINGLE METRIC THAT MATTERS:
Can EOS generate enough yield to pay for its own LLM API calls?

This module implements the four MVP capabilities:

Task 1 - OikosMetabolism:
  Reads live cost data from MetabolicTracker (already tracking every LLM token).
  Computes cost_per_hour, cost_per_day, projected_monthly_cost.
  Persists to Redis key eos:oikos:metabolism so data survives restarts.

Task 2 - YieldStrategy:
  Fetches real yield rates from DeFiLlama / Aave API (falls back to configured APY).
  Given EOS_CAPITAL_BASE_USD, computes daily_yield at current rates.
  Compares daily_yield vs daily_cost → surplus_or_deficit, days_of_runway.

Task 3 - OikosBudgetAuthority:
  Systems call GET /oikos/budget-check?system=nova&action=llm_call&estimated_cost=0.02
  Oikos approves/denies based on each system's remaining daily allocation.
  Emits BUDGET_EXHAUSTED event on Synapse bus when a system overruns.
  Every decision is logged to Redis for audit.

Task 4 - RunwayAlarm:
  Watches cost/yield ratio every cycle.
  Emits ECONOMIC_STRESS on Synapse bus when runway < 30 days.
  Triggers Thymos webhook (POST) when runway < 7 days.
  EOS never silently runs out of money.

Architecture notes:
  - No direct cross-system imports - Synapse integration is event-bus only.
  - All financial figures sourced from MetabolicTracker, env vars, or real APIs.
  - No mock data, no hardcoded fake numbers.
  - Decimal for all USD math.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from config import OikosConfig
    from systems.synapse.event_bus import EventBus
    from systems.synapse.metabolism import MetabolicTracker


logger = structlog.get_logger("oikos.metabolism_api")

# ─── Redis Keys ───────────────────────────────────────────────────────

_METABOLISM_KEY = "eos:oikos:metabolism"
_YIELD_CACHE_KEY = "eos:oikos:yield_cache"

# DeFiLlama yields endpoint - free, no API key, returns stablecoin pool APYs.
# We target USDC/USDT pools on Aave V3 (Base L2) as the safest stablecoin yield.
_DEFILLAMA_YIELDS_URL = "https://yields.llama.fi/pools"

# Aave V3 Base USDC pool ID from DeFiLlama (stable reference, rarely changes).
_AAVE_BASE_USDC_POOL = "aave-v3-base-usdc"

# Timeout for external yield API calls (seconds).
_YIELD_API_TIMEOUT_S = 10.0

# Name for events emitted onto the Synapse bus. These are new event types
# not yet in SynapseEventType - we use string keys per the bus's data dict.
_EVENT_SOURCE = "oikos"


# ─── Cost Snapshot ────────────────────────────────────────────────────


@dataclass
class MetabolismSnapshot:
    """Point-in-time view of EOS's operating cost."""

    cost_per_hour_usd: Decimal
    cost_per_day_usd: Decimal
    projected_monthly_cost_usd: Decimal
    per_system_cost_usd: dict[str, Decimal]
    total_llm_calls: int
    total_input_tokens: int
    total_output_tokens: int
    rolling_deficit_usd: Decimal
    burn_rate_usd_per_sec: Decimal
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_per_hour_usd": str(self.cost_per_hour_usd),
            "cost_per_day_usd": str(self.cost_per_day_usd),
            "projected_monthly_cost_usd": str(self.projected_monthly_cost_usd),
            "per_system_cost_usd": {k: str(v) for k, v in self.per_system_cost_usd.items()},
            "total_llm_calls": self.total_llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "rolling_deficit_usd": str(self.rolling_deficit_usd),
            "burn_rate_usd_per_sec": str(self.burn_rate_usd_per_sec),
            "captured_at": self.captured_at.isoformat(),
        }


# ─── Yield Snapshot ───────────────────────────────────────────────────


@dataclass
class YieldSnapshot:
    """Current yield position vs operating cost."""

    capital_base_usd: Decimal
    current_apy: Decimal              # Actual rate from API or configured fallback
    apy_source: str                   # "defillama:aave-v3-base-usdc" | "configured_fallback"
    daily_yield_usd: Decimal
    daily_cost_usd: Decimal
    surplus_or_deficit_usd: Decimal   # Positive = surplus, negative = deficit
    days_of_runway: Decimal           # liquid_balance / daily_cost (from EconomicState)
    is_self_sustaining: bool          # daily_yield >= daily_cost
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capital_base_usd": str(self.capital_base_usd),
            "current_apy": str(self.current_apy),
            "apy_source": self.apy_source,
            "daily_yield_usd": str(self.daily_yield_usd),
            "daily_cost_usd": str(self.daily_cost_usd),
            "surplus_or_deficit_usd": str(self.surplus_or_deficit_usd),
            "days_of_runway": str(self.days_of_runway),
            "is_self_sustaining": self.is_self_sustaining,
            "captured_at": self.captured_at.isoformat(),
        }


# ─── Budget Decision ──────────────────────────────────────────────────


@dataclass
class BudgetDecision:
    """Result of a budget-check request."""

    approved: bool
    reason: str
    remaining_daily_budget_usd: Decimal
    system_id: str
    action: str
    estimated_cost_usd: Decimal
    daily_allocation_usd: Decimal
    spent_today_usd: Decimal
    decided_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "remaining_daily_budget_usd": str(self.remaining_daily_budget_usd),
            "system_id": self.system_id,
            "action": self.action,
            "estimated_cost_usd": str(self.estimated_cost_usd),
            "daily_allocation_usd": str(self.daily_allocation_usd),
            "spent_today_usd": str(self.spent_today_usd),
            "decided_at": self.decided_at.isoformat(),
        }


# ─── Task 1 + 2 + 4: OikosMetabolism ────────────────────────────────


class OikosMetabolism:
    """
    Tracks what EOS is spending and whether yield covers the bill.

    Reads from MetabolicTracker (already wired into every LLM call in Synapse).
    Persists a cost snapshot to Redis every time it's read, so it survives restarts.

    Also manages the runway alarm (Task 4): emits events and escalates via webhook
    when the organism approaches economic failure.
    """

    def __init__(
        self,
        config: OikosConfig,
        metabolism: MetabolicTracker | None,
        event_bus: EventBus | None = None,
        redis: RedisClient | None = None,
    ) -> None:
        self._config = config
        self._metabolism = metabolism
        self._event_bus = event_bus
        self._redis = redis
        self._logger = logger.bind(component="oikos_metabolism")

        # ── Yield cache ──
        self._cached_apy: Decimal = Decimal(str(config.conservative_apy))
        self._apy_source: str = "configured_fallback"
        self._last_yield_refresh: float = 0.0  # monotonic timestamp

        # ── Runway alarm state ──
        self._last_stress_event_at: float = 0.0    # monotonic
        self._last_escalation_at: float = 0.0       # monotonic
        self._stress_event_cooldown_s: float = 3600.0  # 1 hour between stress events
        self._escalation_cooldown_s: float = 7200.0   # 2 hours between webhook calls

        # ── Budget tracking (reset daily) ──
        self._budget_day: str = ""                          # YYYY-MM-DD
        self._spent_today: dict[str, Decimal] = {}          # system_id → spent_today

        self._logger.info(
            "oikos_metabolism_initialized",
            capital_base_usd=config.capital_base_usd,
            conservative_apy=config.conservative_apy,
            daily_budget_floor_usd=config.daily_budget_floor_usd,
            yield_api_url=config.yield_api_url or "none (fallback only)",
        )

    def attach(self, event_bus: EventBus) -> None:
        """Wire to the Synapse event bus for emission."""
        self._event_bus = event_bus

    # ─── Task 1: Cost Snapshot ────────────────────────────────────────

    async def get_metabolism_snapshot(self) -> MetabolismSnapshot:
        """
        Read current cost state from MetabolicTracker and persist to Redis.

        Returns live figures - not cached.  Call from the /oikos/metabolism
        endpoint handler.
        """
        if self._metabolism is None:
            # No tracker available - return zeroed snapshot so the endpoint
            # still responds with real structure, not an error.
            snap = MetabolismSnapshot(
                cost_per_hour_usd=Decimal("0"),
                cost_per_day_usd=Decimal("0"),
                projected_monthly_cost_usd=Decimal("0"),
                per_system_cost_usd={},
                total_llm_calls=0,
                total_input_tokens=0,
                total_output_tokens=0,
                rolling_deficit_usd=Decimal("0"),
                burn_rate_usd_per_sec=Decimal("0"),
            )
        else:
            raw = self._metabolism.snapshot()
            cost_per_hour = Decimal(str(raw.burn_rate_usd_per_hour))
            cost_per_day = cost_per_hour * Decimal("24")
            snap = MetabolismSnapshot(
                cost_per_hour_usd=cost_per_hour.quantize(Decimal("0.000001")),
                cost_per_day_usd=cost_per_day.quantize(Decimal("0.000001")),
                projected_monthly_cost_usd=(cost_per_day * Decimal("30")).quantize(
                    Decimal("0.000001"),
                ),
                per_system_cost_usd={
                    sid: Decimal(str(c)).quantize(Decimal("0.000001"))
                    for sid, c in raw.per_system_cost_usd.items()
                },
                total_llm_calls=raw.total_calls,
                total_input_tokens=raw.total_input_tokens,
                total_output_tokens=raw.total_output_tokens,
                rolling_deficit_usd=Decimal(str(raw.rolling_deficit_usd)).quantize(
                    Decimal("0.000001"),
                ),
                burn_rate_usd_per_sec=Decimal(str(raw.burn_rate_usd_per_sec)).quantize(
                    Decimal("0.0000000001"),
                ),
            )

        # Persist to Redis (fire-and-forget - non-fatal if Redis is down)
        asyncio.ensure_future(self._persist_metabolism(snap))

        return snap

    async def _persist_metabolism(self, snap: MetabolismSnapshot) -> None:
        """Write cost snapshot to Redis so it survives restarts."""
        if self._redis is None:
            return
        try:
            await self._redis.set_json(_METABOLISM_KEY, snap.to_dict())
        except Exception as exc:
            self._logger.warning("metabolism_persist_failed", error=str(exc))

    async def load_metabolism_from_redis(self) -> MetabolismSnapshot | None:
        """
        Load the last persisted metabolism snapshot from Redis on startup.

        Returns None if no snapshot exists yet (first boot).
        """
        if self._redis is None:
            return None
        try:
            blob = await self._redis.get_json(_METABOLISM_KEY)
            if blob is None:
                return None
            return MetabolismSnapshot(
                cost_per_hour_usd=_d(blob.get("cost_per_hour_usd", "0")),
                cost_per_day_usd=_d(blob.get("cost_per_day_usd", "0")),
                projected_monthly_cost_usd=_d(blob.get("projected_monthly_cost_usd", "0")),
                per_system_cost_usd={
                    k: _d(v) for k, v in blob.get("per_system_cost_usd", {}).items()
                },
                total_llm_calls=int(blob.get("total_llm_calls", 0)),
                total_input_tokens=int(blob.get("total_input_tokens", 0)),
                total_output_tokens=int(blob.get("total_output_tokens", 0)),
                rolling_deficit_usd=_d(blob.get("rolling_deficit_usd", "0")),
                burn_rate_usd_per_sec=_d(blob.get("burn_rate_usd_per_sec", "0")),
                captured_at=datetime.fromisoformat(
                    blob.get("captured_at", datetime.now(UTC).isoformat()),
                ),
            )
        except Exception as exc:
            self._logger.warning("metabolism_load_failed", error=str(exc))
            return None

    # ─── Task 2: Yield Strategy ───────────────────────────────────────

    async def get_yield_snapshot(self, runway_days: Decimal | None = None) -> YieldSnapshot:
        """
        Compute daily_yield vs daily_cost and determine if EOS is self-sustaining.

        Tries to fetch a live APY from DeFiLlama (Aave V3 USDC on Base).
        Falls back to EOS_CONSERVATIVE_APY if the API is unavailable.

        Args:
            runway_days: Pre-computed runway from OikosService.snapshot().
                         If None, runway is reported as 0 (unknown).
        """
        # Refresh APY from external source if interval has elapsed
        now = time.monotonic()
        if now - self._last_yield_refresh >= self._config.yield_refresh_interval_s:
            await self._refresh_yield_rate()

        capital = Decimal(str(self._config.capital_base_usd))
        apy = self._cached_apy

        # Daily yield = capital × APY / 365
        daily_yield = (capital * apy / Decimal("365")).quantize(Decimal("0.000001"))

        # Daily cost from metabolic tracker
        if self._metabolism is not None:
            raw = self._metabolism.snapshot()
            daily_cost = (
                Decimal(str(raw.burn_rate_usd_per_hour)) * Decimal("24")
            ).quantize(Decimal("0.000001"))
        else:
            daily_cost = Decimal("0")

        surplus = (daily_yield - daily_cost).quantize(Decimal("0.000001"))

        return YieldSnapshot(
            capital_base_usd=capital,
            current_apy=apy,
            apy_source=self._apy_source,
            daily_yield_usd=daily_yield,
            daily_cost_usd=daily_cost,
            surplus_or_deficit_usd=surplus,
            days_of_runway=runway_days or Decimal("0"),
            is_self_sustaining=daily_yield >= daily_cost,
        )

    async def _refresh_yield_rate(self) -> None:
        """
        Fetch current APY from DeFiLlama yields API (Aave V3 USDC on Base).

        DeFiLlama is free, no API key, and aggregates across protocols.
        We target the most liquid USDC pools on established protocols only.

        Falls back to EOS_CONSERVATIVE_APY on any error.
        """
        yield_url = self._config.yield_api_url or _DEFILLAMA_YIELDS_URL

        try:
            async with httpx.AsyncClient(timeout=_YIELD_API_TIMEOUT_S) as client:
                resp = await client.get(yield_url)
                resp.raise_for_status()
                data = resp.json()

            # DeFiLlama response: {"status": "ok", "data": [{"pool": "...", "apy": 4.5, ...}]}
            # We pick the best USDC pool from a curated allowlist of safe protocols.
            pools: list[dict[str, Any]] = data.get("data", [])
            best_apy = self._select_best_usdc_apy(pools)

            if best_apy is not None:
                self._cached_apy = best_apy
                self._apy_source = f"defillama:{yield_url}"
                self._logger.info(
                    "yield_rate_refreshed",
                    apy=str(best_apy),
                    source="defillama",
                )
            else:
                # No qualifying pool found - keep fallback
                self._logger.warning(
                    "yield_rate_no_qualifying_pool",
                    action="keeping_fallback_apy",
                )

        except Exception as exc:
            self._logger.warning(
                "yield_rate_refresh_failed",
                error=str(exc),
                fallback_apy=str(self._config.conservative_apy),
            )
            # Reset to configured fallback on failure
            self._cached_apy = Decimal(str(self._config.conservative_apy))
            self._apy_source = "configured_fallback"

        self._last_yield_refresh = time.monotonic()

    def _select_best_usdc_apy(self, pools: list[dict[str, Any]]) -> Decimal | None:
        """
        Select the highest qualifying APY from DeFiLlama pool data.

        Only accepts:
          - USDC or USDT as the base token (stablecoins only - no impermanent loss)
          - Established protocols: Aave V3, Compound V3, Morpho
          - Chains: Ethereum mainnet, Base, Arbitrum, Optimism
          - TVL > $10M USD (avoid tiny pools)
          - APY > 0 and < 50% (sanity filter against inflated numbers)
        """
        safe_protocols = frozenset({"aave-v3", "compound-v3", "morpho", "spark"})
        safe_chains = frozenset({"Ethereum", "Base", "Arbitrum", "Optimism"})
        safe_symbols = frozenset({"USDC", "USDT", "DAI"})
        min_tvl_usd = 10_000_000
        max_apy_sanity = Decimal("50")

        best: Decimal | None = None
        for pool in pools:
            try:
                symbol: str = pool.get("symbol", "").upper()
                project: str = pool.get("project", "").lower()
                chain: str = pool.get("chain", "")
                tvl: float = float(pool.get("tvlUsd", 0))
                apy_raw: float = float(pool.get("apy", 0))

                if not any(s in symbol for s in safe_symbols):
                    continue
                if project not in safe_protocols:
                    continue
                if chain not in safe_chains:
                    continue
                if tvl < min_tvl_usd:
                    continue
                if apy_raw <= 0:
                    continue

                apy = Decimal(str(round(apy_raw / 100, 8)))  # % → decimal
                if apy > max_apy_sanity:
                    continue

                if best is None or apy > best:
                    best = apy

            except (ValueError, TypeError, InvalidOperation):
                continue

        return best

    # ─── Task 3: Budget Authority ─────────────────────────────────────

    async def check_budget(
        self,
        system_id: str,
        action: str,
        estimated_cost_usd: Decimal,
    ) -> BudgetDecision:
        """
        Approve or deny a system's request to spend.

        Daily budget = max(yesterday's daily_yield, daily_budget_floor_usd).
        Each system gets a fraction of the total budget per config.
        Systems not in per_system_budget_fractions share the remainder equally.

        Every decision is logged to Redis for audit.
        BUDGET_EXHAUSTED is emitted on Synapse bus when a system overruns.
        """
        self._reset_budget_day_if_needed()

        daily_budget = self._compute_daily_budget()
        allocation = self._system_allocation(system_id, daily_budget)
        spent = self._spent_today.get(system_id, Decimal("0"))
        remaining = max(allocation - spent, Decimal("0"))

        if estimated_cost_usd <= remaining:
            # Approve - charge against today's allocation
            self._spent_today[system_id] = spent + estimated_cost_usd
            decision = BudgetDecision(
                approved=True,
                reason="within_daily_allocation",
                remaining_daily_budget_usd=(remaining - estimated_cost_usd).quantize(
                    Decimal("0.000001"),
                ),
                system_id=system_id,
                action=action,
                estimated_cost_usd=estimated_cost_usd,
                daily_allocation_usd=allocation,
                spent_today_usd=self._spent_today[system_id],
            )
        else:
            # Deny - emit BUDGET_EXHAUSTED loudly
            decision = BudgetDecision(
                approved=False,
                reason=(
                    f"daily_allocation_exhausted "
                    f"(allocation={allocation:.6f}, spent={spent:.6f})"
                ),
                remaining_daily_budget_usd=remaining,
                system_id=system_id,
                action=action,
                estimated_cost_usd=estimated_cost_usd,
                daily_allocation_usd=allocation,
                spent_today_usd=spent,
            )
            asyncio.ensure_future(self._emit_budget_exhausted(system_id, decision))

        # Log every decision to Redis (async, non-fatal)
        asyncio.ensure_future(self._log_budget_decision(decision))

        self._logger.info(
            "budget_check",
            system_id=system_id,
            action=action,
            estimated_cost=str(estimated_cost_usd),
            approved=decision.approved,
            remaining=str(decision.remaining_daily_budget_usd),
        )

        return decision

    def _compute_daily_budget(self) -> Decimal:
        """
        Daily budget = max(current daily_yield, daily_budget_floor_usd).

        Uses the cached APY and capital base - no I/O on the hot path.
        """
        capital = Decimal(str(self._config.capital_base_usd))
        apy = self._cached_apy
        daily_yield = capital * apy / Decimal("365")
        floor = Decimal(str(self._config.daily_budget_floor_usd))
        return max(daily_yield, floor).quantize(Decimal("0.000001"))

    def _system_allocation(self, system_id: str, daily_budget: Decimal) -> Decimal:
        """
        Compute a specific system's daily allocation.

        Configured systems get their fraction explicitly.
        Remaining budget is split equally across all other systems (up to 9 systems
        assumed for EOS's known cognitive systems).
        """
        fractions = self._config.per_system_budget_fractions
        if system_id in fractions:
            return (daily_budget * Decimal(str(fractions[system_id]))).quantize(
                Decimal("0.000001"),
            )

        # Systems not explicitly configured share the remaining budget equally.
        # We assume up to 9 unconfigured systems to avoid over-allocation.
        configured_total = sum(Decimal(str(f)) for f in fractions.values())
        remaining_fraction = max(Decimal("1") - configured_total, Decimal("0"))
        # Default: divide remaining equally across up to 9 unconfigured systems
        per_system = remaining_fraction / Decimal("9")
        return (daily_budget * per_system).quantize(Decimal("0.000001"))

    def _reset_budget_day_if_needed(self) -> None:
        """Reset per-system spend counters at midnight UTC."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._budget_day != today:
            self._budget_day = today
            self._spent_today = {}
            self._logger.info("budget_day_reset", date=today)

    async def _emit_budget_exhausted(self, system_id: str, decision: BudgetDecision) -> None:
        """Emit BUDGET_EXHAUSTED event on the Synapse bus - loudly."""
        if self._event_bus is None:
            self._logger.error(
                "budget_exhausted_no_event_bus",
                system_id=system_id,
                allocation=str(decision.daily_allocation_usd),
                spent=str(decision.spent_today_usd),
            )
            return

        # Import deferred to avoid circular import at module load time
        from systems.synapse.types import SynapseEvent, SynapseEventType

        event = SynapseEvent(
            event_type=SynapseEventType.BUDGET_EXHAUSTED,
            source_system=_EVENT_SOURCE,
            data={
                "system_id": system_id,
                "action": decision.action,
                "estimated_cost_usd": str(decision.estimated_cost_usd),
                "daily_allocation_usd": str(decision.daily_allocation_usd),
                "spent_today_usd": str(decision.spent_today_usd),
                "reason": decision.reason,
                "timestamp": decision.decided_at.isoformat(),
            },
        )
        await self._event_bus.emit(event)
        self._logger.error(
            "budget_exhausted_event_emitted",
            system_id=system_id,
            spent=str(decision.spent_today_usd),
            allocation=str(decision.daily_allocation_usd),
        )

    async def _log_budget_decision(self, decision: BudgetDecision) -> None:
        """Append budget decision to Redis audit log (ring buffer)."""
        if self._redis is None:
            return
        try:
            key = self._config.budget_audit_redis_key
            entry = json.dumps(decision.to_dict())
            # LPUSH + LTRIM = ring buffer
            await self._redis.client.lpush(key, entry)  # type: ignore[attr-defined]
            await self._redis.client.ltrim(  # type: ignore[attr-defined]
                key, 0, self._config.budget_audit_max_entries - 1,
            )
        except Exception as exc:
            self._logger.warning("budget_audit_log_failed", error=str(exc))

    # ─── Task 4: Runway Alarm ─────────────────────────────────────────

    async def check_runway_alarm(self, runway_days: Decimal, daily_cost: Decimal) -> None:
        """
        Check runway health and escalate if approaching economic failure.

        Call this periodically (e.g., from the cognitive cycle or a background task).

        Emits ECONOMIC_STRESS to Synapse bus when runway < 30 days.
        Triggers Thymos webhook POST when runway < 7 days.
        Never silently allows the organism to run out of money.
        """
        now = time.monotonic()

        # ── Economic stress (< 30 days) ───────────────────────────────
        stress_threshold = Decimal(str(self._config.economic_stress_runway_threshold_days))
        stress_cooled = now - self._last_stress_event_at >= self._stress_event_cooldown_s
        if runway_days < stress_threshold and stress_cooled:
            await self._emit_economic_stress(runway_days, daily_cost)
            self._last_stress_event_at = now

        # ── Hard escalation (< 7 days) ────────────────────────────────
        escalation_threshold = Decimal(str(self._config.escalation_runway_threshold_days))
        escalation_cooled = now - self._last_escalation_at >= self._escalation_cooldown_s
        if runway_days < escalation_threshold and escalation_cooled:
            await self._trigger_thymos_escalation(runway_days, daily_cost)
            self._last_escalation_at = now

    async def _emit_economic_stress(
        self,
        runway_days: Decimal,
        daily_cost: Decimal,
    ) -> None:
        """Emit ECONOMIC_STRESS event on Synapse bus with full financial context."""
        if self._event_bus is None:
            self._logger.error(
                "economic_stress_no_event_bus",
                runway_days=str(runway_days),
            )
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        capital = Decimal(str(self._config.capital_base_usd))
        daily_yield = (capital * self._cached_apy / Decimal("365")).quantize(
            Decimal("0.000001"),
        )

        event = SynapseEvent(
            event_type=SynapseEventType.METABOLIC_PRESSURE,
            source_system=_EVENT_SOURCE,
            data={
                "economic_stress": True,
                "runway_days": str(runway_days),
                "daily_cost_usd": str(daily_cost),
                "daily_yield_usd": str(daily_yield),
                "surplus_or_deficit_usd": str(daily_yield - daily_cost),
                "capital_base_usd": str(capital),
                "current_apy": str(self._cached_apy),
                "apy_source": self._apy_source,
                "stress_threshold_days": str(self._config.economic_stress_runway_threshold_days),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        await self._event_bus.emit(event)
        self._logger.warning(
            "economic_stress_event_emitted",
            runway_days=str(runway_days),
            daily_cost=str(daily_cost),
            daily_yield=str(daily_yield),
        )

    async def _trigger_thymos_escalation(
        self,
        runway_days: Decimal,
        daily_cost: Decimal,
    ) -> None:
        """
        POST to the Thymos escalation webhook with full economic context.

        This is the hard alarm - called when runway < 7 days.
        EOS must never silently run out of money.
        """
        webhook_url = self._config.escalation_webhook_url
        if not webhook_url:
            self._logger.error(
                "thymos_escalation_no_webhook",
                runway_days=str(runway_days),
                message="Set EOS_ESCALATION_WEBHOOK to receive critical economic alerts",
            )
            return

        capital = Decimal(str(self._config.capital_base_usd))
        daily_yield = (capital * self._cached_apy / Decimal("365")).quantize(
            Decimal("0.000001"),
        )

        payload = {
            "alert_type": "ECONOMIC_CRITICAL",
            "severity": "critical",
            "runway_days": str(runway_days),
            "daily_cost_usd": str(daily_cost),
            "daily_yield_usd": str(daily_yield),
            "surplus_or_deficit_usd": str(daily_yield - daily_cost),
            "capital_base_usd": str(capital),
            "current_apy": str(self._cached_apy),
            "apy_source": self._apy_source,
            "escalation_threshold_days": str(self._config.escalation_runway_threshold_days),
            "message": (
                f"EOS runway is {runway_days:.1f} days. "
                f"Daily cost ${daily_cost:.4f} > daily yield ${daily_yield:.4f}. "
                "Immediate capital injection required."
            ),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        async def _do_post() -> None:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(webhook_url, json=payload)
                resp.raise_for_status()
            self._logger.critical(
                "thymos_escalation_sent",
                runway_days=str(runway_days),
                webhook_url=webhook_url,
                status=resp.status_code,
            )

        try:
            await asyncio.wait_for(_do_post(), timeout=2.0)
        except asyncio.TimeoutError:
            self._logger.warning(
                "thymos_escalation_timeout",
                runway_days=str(runway_days),
                webhook_url=webhook_url,
            )
        except Exception as exc:
            # Log at critical level - the escalation itself failed.
            self._logger.critical(
                "thymos_escalation_failed",
                runway_days=str(runway_days),
                error=str(exc),
                payload=payload,
            )


# ─── Helpers ──────────────────────────────────────────────────────────


def _d(value: str | int | float | None, default: str = "0") -> Decimal:
    """Safe Decimal conversion with fallback."""
    try:
        return Decimal(str(value or default))
    except (InvalidOperation, ValueError):
        return Decimal(default)
