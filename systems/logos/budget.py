"""
EcodiaOS - Logos: Cognitive Budget Manager

Maintains the hard capacity limit on total stored knowledge.
The pressure of this limit is the primary driver of abstraction.
"""

from __future__ import annotations

import structlog

from systems.logos.types import CognitiveBudgetState, MemoryTier

logger = structlog.get_logger("logos.budget")


class CognitiveBudgetManager:
    """
    Enforces the cognitive budget across all memory tiers.

    The budget is not a soft guideline. Exceeding it triggers
    escalating compression responses: passive pressure -> aggressive
    compression -> emergency distillation -> forced eviction.
    """

    def __init__(self, total_budget: int = 1_000_000) -> None:
        self._state = CognitiveBudgetState(total_budget=total_budget)

    @property
    def state(self) -> CognitiveBudgetState:
        return self._state

    @property
    def total_pressure(self) -> float:
        return self._state.total_pressure

    @property
    def compression_urgency(self) -> float:
        return self._state.compression_urgency

    def update_utilization(self, tier: MemoryTier, count: float) -> None:
        """Set the current utilization for a memory tier."""
        self._state.current_utilization[tier.value] = count
        logger.debug(
            "budget_utilization_updated",
            tier=tier.value,
            count=count,
            total_pressure=self._state.total_pressure,
        )

    def increment(self, tier: MemoryTier, amount: float = 1.0) -> bool:
        """
        Attempt to allocate KU in a tier. Returns False if the tier
        budget would be exceeded (admission denied).
        """
        current = self._state.current_utilization.get(tier.value, 0.0)
        new_total = current + amount
        tier_limit = self._state.tier_budget(tier)

        if new_total > tier_limit:
            logger.warning(
                "budget_admission_denied",
                tier=tier.value,
                requested=amount,
                current=current,
                limit=tier_limit,
            )
            return False

        self._state.current_utilization[tier.value] = new_total
        return True

    def decrement(self, tier: MemoryTier, amount: float = 1.0) -> None:
        """Release KU from a tier (eviction or compression)."""
        current = self._state.current_utilization.get(tier.value, 0.0)
        self._state.current_utilization[tier.value] = max(0.0, current - amount)

    def is_emergency(self) -> bool:
        """True if utilization >= emergency threshold."""
        return self._state.total_pressure >= self._state.emergency_compression

    def is_critical(self) -> bool:
        """True if utilization >= critical eviction threshold."""
        return self._state.total_pressure >= self._state.critical_eviction

    def needs_compression(self) -> bool:
        """True if utilization >= compression pressure start."""
        return self._state.total_pressure >= self._state.compression_pressure_start

    def pressure_payload(self) -> dict[str, float]:
        """Payload for COGNITIVE_PRESSURE Synapse event."""
        return {
            "pressure": self._state.total_pressure,
            "urgency": self._state.compression_urgency,
        }

    def record_compression_operation(
        self,
        *,
        cost_ku: float,
        gain_ku: float,
        tier: MemoryTier = MemoryTier.WORLD_MODEL,
    ) -> None:
        """HIGH-3: Update utilization after one compression operation.

        After each compression:
          - increment current_utilization by compression_cost_ku
            (the work of computing and storing the new compressed form)
          - decrement current_utilization by compression_gain_ku
            (the budget freed by evicting/replacing the source material)

        Net effect is (cost - gain): positive when compression is costly,
        negative when a highly compressive cycle frees more than it costs.
        """
        current = self._state.current_utilization.get(tier.value, 0.0)
        updated = max(0.0, current + cost_ku - gain_ku)
        self._state.current_utilization[tier.value] = updated
        logger.debug(
            "compression_operation_recorded",
            tier=tier.value,
            cost_ku=cost_ku,
            gain_ku=gain_ku,
            utilization_before=current,
            utilization_after=updated,
            total_pressure=self._state.total_pressure,
        )

    @property
    def current_utilization_state(self) -> dict[str, float]:
        """Synapse-readable snapshot of per-tier utilization (HIGH-3).

        Returned by the COGNITIVE_PRESSURE payload so every subscriber has
        live utilization data without a direct reference to the budget manager.
        """
        return {
            tier.value: self._state.current_utilization.get(tier.value, 0.0)
            for tier in MemoryTier
        }

    def snapshot(self) -> dict[str, float]:
        """Full budget snapshot for telemetry."""
        return {
            "total_budget": float(self._state.total_budget),
            "total_used": self._state.total_used,
            "total_pressure": self._state.total_pressure,
            "compression_urgency": self._state.compression_urgency,
            **{
                f"tier_{tier.value}_used": self._state.current_utilization.get(
                    tier.value, 0.0
                )
                for tier in MemoryTier
            },
            **{
                f"tier_{tier.value}_pressure": self._state.tier_pressure(tier)
                for tier in MemoryTier
            },
        }
