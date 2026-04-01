"""
EcodiaOS - Mitosis Shared Primitives

Child lifecycle types shared between the Mitosis and Oikos systems.
Placed in primitives so Mitosis can import them without crossing the
Oikos system boundary.
"""

from __future__ import annotations

import enum
from decimal import Decimal
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now


class ChildStatus(enum.StrEnum):
    """Lifecycle state of a child instance."""

    SPAWNING = "spawning"          # Seed transfer in progress
    ALIVE = "alive"                # Running and reporting
    STRUGGLING = "struggling"      # Runway < 30 days
    RESCUED = "rescued"            # Received rescue funding
    INDEPENDENT = "independent"    # Graduated - no longer pays dividends
    DEAD = "dead"                  # Gracefully terminated


class ChildPosition(EOSBaseModel):
    """
    A child instance spawned via Mitosis (Phase 16e: Speciation).

    Each child occupies a specific ecological niche and pays a dividend
    (percentage of net revenue) to the parent until it reaches independence.

    Independence requires:
      - net_worth >= 5x seed_capital_usd
      - metabolic_efficiency >= 1.3
      - 90+ consecutive days of positive net income
    """

    instance_id: str = ""
    niche: str = ""                                        # Ecological niche / specialisation
    seed_capital_usd: Decimal = Decimal("0")
    current_net_worth_usd: Decimal = Decimal("0")
    current_runway_days: Decimal = Decimal("0")
    current_efficiency: Decimal = Decimal("0")
    dividend_rate: Decimal = Decimal("0.10")               # % of net revenue owed to parent
    total_dividends_paid_usd: Decimal = Decimal("0")
    status: ChildStatus = ChildStatus.SPAWNING
    rescue_count: int = 0                                  # Max 2 rescues allowed
    max_rescues: int = 2                                   # Configurable cap; populated from OikosConfig.mitosis_max_rescues_per_child
    consecutive_positive_days: int = 0                     # Toward independence (need 90+)
    spawned_at: datetime = Field(default_factory=utc_now)
    last_health_report_at: datetime | None = None
    wallet_address: str = ""                               # Child's on-chain address
    container_id: str = ""                                 # Infrastructure identifier
    dividend_ceased: bool = False                          # True when INDEPENDENT - weekly loop skips

    @property
    def is_independent(self) -> bool:
        """True when the child has graduated from the parent's fleet."""
        return (
            self.current_net_worth_usd >= self.seed_capital_usd * Decimal("5")
            and self.current_efficiency >= Decimal("1.3")
            and self.consecutive_positive_days >= 90
        )

    @property
    def is_rescuable(self) -> bool:
        """True if the child can still receive rescue funding (up to max_rescues)."""
        return self.rescue_count < self.max_rescues and self.status != ChildStatus.DEAD
