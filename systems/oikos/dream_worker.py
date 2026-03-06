"""
EcodiaOS -- Oikos: Economic Dream Worker (Phase 16i)

An Oneiros-compatible worker that runs during the consolidation cycle
(between REM and HYPNOPOMPIA). The organism dreams about its economic
future: running Monte Carlo simulations, stress-testing strategies,
and deriving parameter adjustments.

This worker bridges two systems:
  - Oneiros (the dream engine) provides the trigger and sleep cycle context
  - Oikos (the economic engine) provides the EconomicState and receives results

The worker follows the BaseOneirosWorker pattern so it can be
hot-swapped via the NeuroplasticityBus like any other dream worker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

from systems.oikos.economic_simulator import EconomicSimulator

if TYPE_CHECKING:
    from config import OikosConfig
    from systems.oikos.dreaming_types import EconomicDreamResult
    from systems.oikos.models import EconomicState

logger = structlog.get_logger("oikos.dreaming")


# -- ABC --------------------------


class BaseEconomicDreamWorker(ABC):
    """ABC for economic dreaming workers. Hot-swappable via NeuroplasticityBus."""

    worker_type: str = "oikos.economic_dreaming"

    @abstractmethod
    async def run(
        self,
        state: EconomicState,
        cycle_id: str,
    ) -> EconomicDreamResult: ...


# -- Default Implementation ------------------------------------------


class EconomicDreamWorker(BaseEconomicDreamWorker):
    """
    Default economic dreaming worker.

    Runs the full Monte Carlo simulation cycle during consolidation:
      1. Baseline simulation (10,000 paths x 365 days)
      2. Stress tests across 8 catastrophe scenarios
      3. Short-horizon (30-day) survival estimate
      4. Recommendation derivation when ruin > 1%

    Results are returned to OneirosService which passes them to
    OikosService for integration into the EconomicState.
    """

    worker_type: str = "oikos.economic_dreaming"

    def __init__(self, config: OikosConfig, seed: int | None = None) -> None:
        self._config = config
        self._simulator = EconomicSimulator(config=config, seed=seed)
        self._logger = logger.bind(component="economic_dream_worker")
        self._last_result: EconomicDreamResult | None = None

    async def run(
        self,
        state: EconomicState,
        cycle_id: str,
    ) -> EconomicDreamResult:
        """
        Execute economic dreaming for a single consolidation cycle.

        This is called by OneirosService after REM dreaming completes
        and before HYPNOPOMPIA begins.
        """
        self._logger.info(
            "economic_dreaming_begin",
            cycle_id=cycle_id,
            liquid_balance=str(state.liquid_balance),
            runway_days=str(state.runway_days),
            metabolic_efficiency=str(state.metabolic_efficiency),
        )

        result = await self._simulator.run_dream_cycle(
            state=state,
            sleep_cycle_id=cycle_id,
        )

        self._last_result = result

        self._logger.info(
            "economic_dreaming_complete",
            cycle_id=cycle_id,
            ruin_probability=str(result.ruin_probability),
            survival_30d=str(result.survival_probability_30d),
            resilience=str(result.resilience_score),
            recommendations=len(result.recommendations),
            stress_failures=sum(
                1 for s in result.stress_tests if not s.survives
            ),
            duration_ms=result.duration_ms,
        )

        if result.recommendations:
            for rec in result.recommendations:
                self._logger.warning(
                    "economic_dream_recommendation",
                    action=rec.action,
                    description=rec.description[:120],
                    priority=rec.priority,
                    confidence=str(rec.confidence),
                )

        return result

    @property
    def last_result(self) -> EconomicDreamResult | None:
        """Most recent dream result, for observability."""
        return self._last_result
