"""
EcodiaOS — Oikos Strategy ABCs (NeuroplasticityBus targets)

These abstract base classes define the hot-swappable strategies that Evo
can evolve at runtime via the NeuroplasticityBus. When a new subclass of
BaseCostModel is discovered in an evolved file, the bus instantiates it
and swaps it into the live OikosService without restart.

Design rules:
  - Zero-arg constructable (all state rebuilt from scratch on hot-swap)
  - Pure computation, no I/O
  - Stateless between calls (the service holds all state)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from decimal import Decimal

if TYPE_CHECKING:

    from systems.oikos.models import EcologicalNiche, EconomicState, MetabolicRate
class BaseCostModel(ABC):
    """
    Strategy base class for BMR calculation.

    The NeuroplasticityBus uses this ABC as its registration target so
    that evolved cost model subclasses can be hot-swapped into a live
    OikosService without restarting the process.

    Subclasses MUST be zero-arg constructable.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable identifier for this cost model strategy."""
        ...

    @abstractmethod
    def compute_bmr(
        self,
        burn_rate_usd_per_hour: Decimal,
        per_system_cost_usd: dict[str, Decimal],
        measurement_window_hours: int,
    ) -> MetabolicRate:
        """
        Compute the Basal Metabolic Rate from observed cost data.

        The BMR is the minimum cost to keep the organism alive: compute,
        storage, network, API minimums. The default implementation uses
        the EMA-smoothed burn rate from MetabolicTracker. Evolved strategies
        might weight recent vs historical data differently, account for
        time-of-day patterns, or factor in cloud cost projections.

        Args:
            burn_rate_usd_per_hour: Current EMA burn rate from Synapse.
            per_system_cost_usd: Cost breakdown by caller ID.
            measurement_window_hours: Config-driven window for BMR calc.

        Returns:
            MetabolicRate with hourly/daily rates and category breakdown.
        """
        ...

    @abstractmethod
    def compute_runway(
        self,
        liquid_balance: Decimal,
        survival_reserve: Decimal,
        bmr: MetabolicRate,
    ) -> tuple[Decimal, Decimal]:
        """
        Compute runway_hours and runway_days from available capital and BMR.

        Args:
            liquid_balance: Available operating capital.
            survival_reserve: Locked survival reserve (only consumed in emergency).
            bmr: Current basal metabolic rate.

        Returns:
            (runway_hours, runway_days) tuple.
        """
        ...


class BaseMitosisStrategy(ABC):
    """
    Strategy base class for niche identification and reproductive fitness.

    The NeuroplasticityBus can hot-swap this so Evo evolves better niche
    detection over time. The default strategy uses simple scoring heuristics;
    evolved strategies may incorporate market data, competitor analysis, or
    latent-space embeddings of capability-niche alignment.

    Subclasses MUST be zero-arg constructable.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Stable identifier for this mitosis strategy."""
        ...

    @abstractmethod
    def identify_niches(
        self,
        state: EconomicState,
        parent_capabilities: list[str],
    ) -> list[EcologicalNiche]:
        """
        Scan the environment for underserved ecological niches a child could fill.

        Args:
            state: Current economic snapshot of the parent.
            parent_capabilities: List of capability tags the parent possesses.

        Returns:
            Ranked list of candidate niches, best first.
        """
        ...

    @abstractmethod
    def score_niche(
        self,
        niche: EcologicalNiche,
        state: EconomicState,
    ) -> Decimal:
        """
        Score a candidate niche on a 0..1 scale for reproductive suitability.

        Factors: estimated profitability, competitive gap, capability alignment,
        confidence level, and capital requirements relative to parent surplus.

        Args:
            niche: The candidate niche to evaluate.
            state: Current economic snapshot.

        Returns:
            Composite fitness score, 0.0 (reject) to 1.0 (ideal).
        """
        ...

    @abstractmethod
    def compute_seed_capital(
        self,
        niche: EcologicalNiche,
        state: EconomicState,
        min_seed: Decimal,
        max_seed_pct: Decimal,
    ) -> Decimal:
        """
        Determine the optimal seed capital for a child in the given niche.

        Must respect: amount >= min_seed AND amount <= max_seed_pct × net_worth.

        Args:
            niche: Target niche.
            state: Parent's economic state.
            min_seed: Absolute minimum seed capital.
            max_seed_pct: Maximum percentage of parent net worth.

        Returns:
            Seed capital amount in USD.
        """
        ...
