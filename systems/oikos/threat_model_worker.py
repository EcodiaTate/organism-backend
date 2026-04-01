"""
EcodiaOS - Oikos: Threat Model Worker (Phase 16i+)

An Oneiros-compatible worker that runs during the consolidation cycle
alongside EconomicDreamWorker. While the dream worker models organism-level
cashflow, this worker models per-asset treasury risk with contagion and
liquidation detection.

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

from systems.oikos.threat_modeler import MonteCarloThreatModeler

if TYPE_CHECKING:
    from config import OikosConfig
    from systems.oikos.models import EconomicState
    from systems.oikos.threat_modeling_types import ThreatModelResult

logger = structlog.get_logger("oikos.threat_modeling")


# ── ABC ────────────────────────


class BaseThreatModelWorker(ABC):
    """ABC for treasury threat model workers. Hot-swappable via NeuroplasticityBus."""

    worker_type: str = "oikos.threat_modeling"

    @abstractmethod
    async def run(
        self,
        state: EconomicState,
        cycle_id: str,
    ) -> ThreatModelResult: ...


# ── Default Implementation ────────────────────────────────────────


class ThreatModelWorker(BaseThreatModelWorker):
    """
    Default treasury threat model worker.

    Runs the full Monte Carlo threat modeling cycle during consolidation:
      1. Extract positions from EconomicState
      2. Simulate per-asset price paths with contagion coupling
      3. Detect liquidation events per DeFi position
      4. Compute tail risk profiles (VaR, CVaR, max drawdown)
      5. Identify critical exposures by marginal VaR contribution
      6. Generate sized hedging proposals

    Results are returned to OneirosService which passes them to
    OikosService for integration and to Nova for hedge pre-loading.
    """

    worker_type: str = "oikos.threat_modeling"

    def __init__(self, config: OikosConfig, seed: int | None = None) -> None:
        self._config = config
        self._modeler = MonteCarloThreatModeler(config=config, seed=seed)
        self._logger = logger.bind(component="threat_model_worker")
        self._last_result: ThreatModelResult | None = None

    async def run(
        self,
        state: EconomicState,
        cycle_id: str,
    ) -> ThreatModelResult:
        """
        Execute treasury threat modeling for a single consolidation cycle.

        Called by OneirosService in parallel with EconomicDreamWorker
        after REM dreaming completes and before HYPNOPOMPIA begins.
        """
        self._logger.info(
            "threat_modeling_begin",
            cycle_id=cycle_id,
            liquid_balance=str(state.liquid_balance),
            yield_positions=len(state.yield_positions),
            total_deployed=str(state.total_deployed),
        )

        result = await self._modeler.run_threat_cycle(
            state=state,
            sleep_cycle_id=cycle_id,
        )

        self._last_result = result

        self._logger.info(
            "threat_modeling_complete",
            cycle_id=cycle_id,
            var_5pct=str(result.portfolio_risk.var_5pct),
            liquidation_prob=str(result.portfolio_risk.liquidation_probability),
            critical_exposures=len(result.critical_exposures),
            hedging_proposals=len(result.hedging_proposals),
            contagion_events=result.contagion_events_detected,
            duration_ms=result.duration_ms,
        )

        if result.hedging_proposals:
            for proposal in result.hedging_proposals:
                self._logger.warning(
                    "hedge_proposal",
                    action=proposal.hedge_action,
                    symbol=proposal.target_symbol,
                    size_usd=str(proposal.hedge_size_usd),
                    description=proposal.description[:120],
                    priority=proposal.priority,
                    confidence=str(proposal.confidence),
                )

        return result

    @property
    def last_result(self) -> ThreatModelResult | None:
        """Most recent threat model result, for observability."""
        return self._last_result
