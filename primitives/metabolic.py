"""
EcodiaOS - Metabolic Gate Primitives

Standard interface for systems to check metabolic permission before
expensive operations. Systems that consume significant resources
(LLM calls, on-chain transactions, compute-heavy simulations) must
gate through the metabolic system to avoid spending beyond runway.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import Field

from primitives.common import EOSBaseModel, SystemID


# Import Oikos types - these are the authoritative definitions
from systems.oikos.models import MetabolicPriority, StarvationLevel


class MetabolicPermission(EOSBaseModel):
    """
    The response to a metabolic gate check.

    Tells the requesting system whether it may proceed with the
    expensive operation, and provides context for graceful degradation
    if denied.
    """

    granted: bool = False
    reason: str = ""
    starvation_level: StarvationLevel = StarvationLevel.NOMINAL
    runway_days: Decimal = Decimal("0")
    budget_remaining_usd: Decimal = Decimal("0")


class MetabolicSubscription(EOSBaseModel):
    """
    What a system subscribes to for metabolic state changes.

    Systems register their priority threshold and cost ceiling so
    Oikos can proactively notify them when metabolic state changes
    require degradation or shutdown.
    """

    system_id: SystemID
    priority_threshold: MetabolicPriority = MetabolicPriority.MAINTENANCE
    cost_ceiling_usd: Decimal = Decimal("0")


@runtime_checkable
class MetabolicGate(Protocol):
    """
    Interface that metabolically-aware systems implement.

    Systems call check_metabolic_permission before expensive operations.
    Oikos (or a metabolic proxy) is the canonical implementation.
    """

    def get_metabolic_priority(self) -> MetabolicPriority: ...

    async def check_metabolic_permission(
        self, estimated_cost_usd: Decimal
    ) -> MetabolicPermission: ...
