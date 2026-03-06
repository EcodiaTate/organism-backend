"""
EcodiaOS — Oikos: Economic Simulator (Phase 16i: Economic Dreaming)

Monte Carlo simulation engine that projects the organism's economic
future across thousands of stochastic paths. During consolidation
(sleep), this engine stress-tests survival strategy and derives
parameter adjustments — all without spending real capital.

The model uses Geometric Brownian Motion with jump-diffusion for
fat-tailed shocks (LLM cost spikes, DeFi exploits, demand droughts).
This is the organism dreaming about money.

Design choices:
  - NumPy for vectorised path simulation (10,000 paths x 365 days)
  - Decimal for inputs/outputs (financial precision), float internally
    for vectorised math (acceptable for Monte Carlo estimates)
  - No I/O — pure computation, safe for the asyncio event loop via
    run_in_executor for the heavy matrix operations
  - Deterministic seed option for reproducible testing
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import structlog

from systems.oikos.dreaming_types import (
    STRESS_SCENARIOS,
    EconomicDreamResult,
    EconomicRecommendation,
    PathStatistics,
    StressScenarioConfig,
    StressTestResult,
)

if TYPE_CHECKING:
    from config import OikosConfig
    from systems.oikos.models import EconomicState

logger = structlog.get_logger("oikos.dreaming")


def _d(v: Decimal) -> float:
    """Decimal to float for vectorised math."""
    return float(v)


def _D(v: float) -> Decimal:
    """float to Decimal for output precision."""
    return Decimal(str(round(v, 6)))


class EconomicSimulator:
    """
    Monte Carlo economic dreaming engine.

    Reads the current EconomicState and projects forward across
    thousands of stochastic paths to estimate ruin probability,
    net worth distribution, and optimal parameter adjustments.

    Thread-safety: the simulate() method is CPU-bound and should be
    called via asyncio.run_in_executor or from within a consolidation
    cycle where the main loop is paused.
    """

    def __init__(self, config: OikosConfig, seed: int | None = None) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._logger = logger.bind(component="economic_simulator")

    # ── Public API ─────────────────────────────────────────────────

    async def run_dream_cycle(
        self,
        state: EconomicState,
        sleep_cycle_id: str = "",
    ) -> EconomicDreamResult:
        """
        Run the full economic dreaming cycle:
          1. Baseline simulation (current parameters, no shocks)
          2. Stress tests across all 8 named scenarios
          3. Derive recommendations if ruin_probability > threshold
          4. Compute resilience score

        CPU-heavy work is offloaded to a thread executor.
        """
        start = time.monotonic()

        loop = asyncio.get_running_loop()

        # Run the CPU-bound simulation in a thread to avoid blocking
        result = await loop.run_in_executor(
            None, self._simulate_all, state, sleep_cycle_id
        )

        result.duration_ms = int((time.monotonic() - start) * 1000)

        self._logger.info(
            "economic_dream_complete",
            ruin_probability=str(result.ruin_probability),
            resilience_score=str(result.resilience_score),
            survival_30d=str(result.survival_probability_30d),
            recommendations=len(result.recommendations),
            total_paths=result.total_paths_simulated,
            duration_ms=result.duration_ms,
        )

        return result

    # ── Core Simulation (CPU-bound, runs in executor) ──────────────

    def _simulate_all(
        self,
        state: EconomicState,
        sleep_cycle_id: str,
    ) -> EconomicDreamResult:
        """Synchronous entry point for the full simulation."""
        result = EconomicDreamResult(sleep_cycle_id=sleep_cycle_id)

        n_paths = self._config.dreaming_paths_per_strategy
        horizon = self._config.dreaming_horizon_days
        stress_paths = self._config.dreaming_stress_test_paths

        # 1. Baseline simulation
        result.baseline = self._run_paths(state, n_paths, horizon, stress=None)
        result.ruin_probability = result.baseline.ruin_probability
        result.total_paths_simulated += n_paths

        # 2. Short-horizon survival (30 days, same path count)
        short_stats = self._run_paths(state, n_paths, 30, stress=None)
        result.survival_probability_30d = Decimal("1") - short_stats.ruin_probability
        result.total_paths_simulated += n_paths

        # 3. Stress tests
        survival_scores: list[float] = []
        for scenario, stress_config in STRESS_SCENARIOS.items():
            stress_stats = self._run_paths(
                state, stress_paths, horizon, stress=stress_config
            )
            survives = stress_stats.ruin_probability < _D(self._config.dreaming_ruin_threshold)
            stress_result = StressTestResult(
                scenario=scenario,
                stats=stress_stats,
                survives=survives,
            )
            result.stress_tests.append(stress_result)
            result.total_paths_simulated += stress_paths
            survival_scores.append(1.0 - _d(stress_stats.ruin_probability))

        # 4. Resilience score: geometric mean of survival across scenarios
        if survival_scores:
            clamped = [max(s, 1e-6) for s in survival_scores]
            geo_mean = float(np.exp(np.mean(np.log(clamped))))
            result.resilience_score = _D(geo_mean)

        # 5. Generate recommendations if baseline ruin > threshold
        if _d(result.ruin_probability) > self._config.dreaming_ruin_threshold:
            result.recommendations = self._derive_recommendations(state, result)

        return result

    def _run_paths(
        self,
        state: EconomicState,
        n_paths: int,
        horizon_days: int,
        stress: StressScenarioConfig | None,
    ) -> PathStatistics:
        """
        Simulate n_paths forward for horizon_days using GBM + jump-diffusion.

        Each path tracks: liquid_balance, deployed_capital, daily_revenue,
        daily_cost. A path "ruins" when liquid_balance + deployed <= 0.
        """
        # Extract initial conditions from EconomicState
        liquid = _d(state.liquid_balance)
        reserve = _d(state.survival_reserve)
        deployed = _d(state.total_deployed)
        avg_apy = _d(state.weighted_avg_apy)

        # Daily income/cost from rolling averages (use 7d for stability)
        daily_revenue = _d(state.revenue_7d) / 7.0 if _d(state.revenue_7d) > 0 else 0.0
        daily_cost = _d(state.costs_7d) / 7.0 if _d(state.costs_7d) > 0 else 0.001

        # Volatility parameters (annualised -> daily)
        rev_vol = self._config.dreaming_revenue_volatility / np.sqrt(365)
        cost_vol = self._config.dreaming_cost_volatility / np.sqrt(365)
        yield_vol = self._config.dreaming_yield_volatility / np.sqrt(365)
        shock_prob = self._config.dreaming_shock_probability
        shock_min = self._config.dreaming_shock_magnitude_min
        shock_max = self._config.dreaming_shock_magnitude_max

        # Apply stress scenario modifiers
        cost_mult = 1.0
        rev_mult = 1.0
        forced_apy: float | None = None
        yield_loss = 0.0
        stress_onset = 0
        stress_duration = horizon_days

        if stress is not None:
            cost_mult = _d(stress.cost_multiplier)
            rev_mult = _d(stress.revenue_multiplier)
            if stress.yield_apy_override is not None:
                forced_apy = _d(stress.yield_apy_override)
            yield_loss = _d(stress.yield_loss_pct)
            stress_onset = stress.onset_day
            stress_duration = stress.duration_days

        # ── Vectorised simulation ──────────────────────────────────
        # Shape: (n_paths, horizon_days)
        z_rev = self._rng.standard_normal((n_paths, horizon_days))
        z_cost = self._rng.standard_normal((n_paths, horizon_days))
        z_yield = self._rng.standard_normal((n_paths, horizon_days))

        # Jump-diffusion: rare fat-tail shocks
        jumps = self._rng.uniform(0, 1, (n_paths, horizon_days)) < shock_prob
        jump_magnitude = self._rng.uniform(
            shock_min, shock_max, (n_paths, horizon_days)
        )

        # Track state across paths
        balances = np.full(n_paths, liquid, dtype=np.float64)
        deployed_arr = np.full(n_paths, deployed, dtype=np.float64)
        min_runway = np.full(n_paths, np.inf, dtype=np.float64)
        max_drawdown = np.zeros(n_paths, dtype=np.float64)
        peak_worth = np.full(n_paths, liquid + deployed + reserve, dtype=np.float64)
        ruined = np.zeros(n_paths, dtype=bool)
        mitosis_day = np.full(n_paths, -1, dtype=np.int32)

        # Apply initial yield loss from stress scenario
        if yield_loss > 0:
            deployed_arr *= (1.0 - yield_loss)

        # Mitosis thresholds (from config)
        mitosis_runway_threshold = float(self._config.mitosis_min_parent_runway_days)
        mitosis_efficiency_threshold = float(self._config.mitosis_min_parent_efficiency)

        for day in range(horizon_days):
            # Determine if stress is active this day
            in_stress = (
                stress is not None
                and stress_onset <= day < stress_onset + stress_duration
            )

            # Revenue: GBM with optional stress multiplier
            day_rev_mult = rev_mult if in_stress else 1.0
            rev_factor = np.exp(-0.5 * rev_vol**2 + rev_vol * z_rev[:, day])
            rev_today = daily_revenue * rev_factor * day_rev_mult

            # Apply jump shocks to revenue (negative shocks)
            rev_today = np.where(
                jumps[:, day],
                rev_today * jump_magnitude[:, day],
                rev_today,
            )

            # Cost: GBM with optional stress multiplier
            day_cost_mult = cost_mult if in_stress else 1.0
            cost_factor = np.exp(-0.5 * cost_vol**2 + cost_vol * z_cost[:, day])
            cost_today = daily_cost * cost_factor * day_cost_mult

            # Yield income from deployed capital
            effective_apy = forced_apy if (in_stress and forced_apy is not None) else avg_apy
            daily_yield_rate = effective_apy / 365.0 if effective_apy > 0 else 0.0
            yield_factor = np.exp(-0.5 * yield_vol**2 + yield_vol * z_yield[:, day])
            yield_today = deployed_arr * daily_yield_rate * yield_factor

            # Net daily cashflow
            net = rev_today + yield_today - cost_today

            # Update balances (only for non-ruined paths)
            balances = np.where(ruined, balances, balances + net)

            # Deployed capital compounds slightly (10% of yield reinvested)
            deployed_arr = np.where(
                ruined,
                deployed_arr,
                deployed_arr * (1.0 + daily_yield_rate * 0.1),
            )

            # Track metrics
            total_worth = balances + deployed_arr + reserve
            peak_worth = np.maximum(peak_worth, total_worth)
            drawdown = (peak_worth - total_worth) / np.maximum(peak_worth, 1e-10)
            max_drawdown = np.maximum(max_drawdown, drawdown)

            # Runway estimate: balance / daily_cost
            runway_est = np.where(
                cost_today > 1e-10,
                balances / cost_today,
                np.inf,
            )
            min_runway = np.minimum(min_runway, runway_est)

            # Ruin detection: liquid + deployed <= 0
            newly_ruined = (balances + deployed_arr <= 0) & ~ruined
            ruined |= newly_ruined

            # Mitosis detection (first day conditions met)
            efficiency = np.where(
                cost_today > 1e-10,
                rev_today / cost_today,
                0.0,
            )
            can_mitose = (
                (runway_est >= mitosis_runway_threshold)
                & (efficiency >= mitosis_efficiency_threshold)
                & (mitosis_day < 0)
                & ~ruined
            )
            mitosis_day = np.where(can_mitose, day, mitosis_day)

        # ── Aggregate statistics ───────────────────────────────────
        final_worth = balances + deployed_arr + reserve
        ruin_count = int(np.sum(ruined))

        alive_mask = ~ruined
        alive_worth = final_worth[alive_mask] if np.any(alive_mask) else np.array([0.0])
        alive_runway = min_runway[alive_mask] if np.any(alive_mask) else np.array([0.0])

        mitosis_achieved = mitosis_day[mitosis_day >= 0]
        median_mitosis = int(np.median(mitosis_achieved)) if len(mitosis_achieved) > 0 else -1

        stats = PathStatistics(
            paths_run=n_paths,
            ruin_count=ruin_count,
            ruin_probability=_D(ruin_count / n_paths),
            median_net_worth=_D(float(np.median(alive_worth))),
            p5_net_worth=_D(float(np.percentile(alive_worth, 5))),
            p95_net_worth=_D(float(np.percentile(alive_worth, 95))),
            mean_net_worth=_D(float(np.mean(final_worth))),
            median_min_runway_days=_D(float(np.median(alive_runway))),
            max_drawdown_median=_D(float(np.median(max_drawdown))),
            median_time_to_mitosis_days=median_mitosis,
        )

        return stats

    # ── Recommendation Derivation ──────────────────────────────────

    def _derive_recommendations(
        self,
        state: EconomicState,
        result: EconomicDreamResult,
    ) -> list[EconomicRecommendation]:
        """
        When ruin_probability exceeds threshold, generate actionable
        recommendations to shift economic parameters toward survival.

        Ordered by expected impact on ruin probability.
        """
        recs: list[EconomicRecommendation] = []
        ruin_p = _d(result.ruin_probability)

        # 1. If deployed capital is large relative to liquid, recommend liquidation
        deployed = _d(state.total_deployed)
        liquid = _d(state.liquid_balance)
        if deployed > 0 and liquid > 0:
            deploy_ratio = deployed / (deployed + liquid)
            if deploy_ratio > 0.5:
                target_ratio = max(0.2, deploy_ratio - 0.15)
                recs.append(EconomicRecommendation(
                    action="liquidate_yield_to_hot_wallet",
                    description=(
                        f"Deployed capital ({deploy_ratio:.0%} of liquid+deployed) "
                        f"is too high relative to runway. Liquidate to {target_ratio:.0%} "
                        f"to increase immediate survival buffer."
                    ),
                    priority=0,
                    parameter_path="oikos.yield.max_single_position_pct",
                    current_value=_D(deploy_ratio),
                    recommended_value=_D(target_ratio),
                    delta=_D(target_ratio - deploy_ratio),
                    ruin_probability_before=_D(ruin_p),
                    confidence=Decimal("0.8"),
                ))

        # 2. If burn rate is high relative to revenue, recommend cost reduction
        daily_cost = _d(state.costs_7d) / 7.0 if _d(state.costs_7d) > 0 else 0
        daily_rev = _d(state.revenue_7d) / 7.0 if _d(state.revenue_7d) > 0 else 0
        if daily_cost > 0 and daily_rev > 0:
            efficiency = daily_rev / daily_cost
            if efficiency < 1.2:
                target_eff = 1.5
                cost_reduction = 1.0 - (efficiency / target_eff)
                recs.append(EconomicRecommendation(
                    action="reduce_burn_rate",
                    description=(
                        f"Metabolic efficiency ({efficiency:.2f}x) is dangerously close "
                        f"to breakeven. Reduce discretionary spending by {cost_reduction:.0%} "
                        f"to target {target_eff}x efficiency."
                    ),
                    priority=1,
                    parameter_path="oikos.metabolic_alert_threshold",
                    current_value=_D(efficiency),
                    recommended_value=_D(target_eff),
                    delta=_D(target_eff - efficiency),
                    ruin_probability_before=_D(ruin_p),
                    confidence=Decimal("0.85"),
                ))

        # 3. If any stress test fails, recommend scenario-specific mitigation
        for stress_result in result.stress_tests:
            if not stress_result.survives:
                scenario_ruin = _d(stress_result.stats.ruin_probability)
                recs.append(EconomicRecommendation(
                    action=f"mitigate_{stress_result.scenario.value}",
                    description=(
                        f"Stress test '{stress_result.scenario.value}' shows "
                        f"{scenario_ruin:.1%} ruin probability. "
                        f"Diversify against this scenario."
                    ),
                    priority=2,
                    ruin_probability_before=_D(scenario_ruin),
                    confidence=Decimal("0.7"),
                ))

        # 4. If runway is short, recommend hunting intensity increase
        runway_days = _d(state.runway_days)
        if runway_days < 30:
            recs.append(EconomicRecommendation(
                action="increase_hunting_intensity",
                description=(
                    f"Runway is only {runway_days:.0f} days. Increase bounty "
                    f"hunting frequency and lower minimum reward threshold "
                    f"to accelerate revenue."
                ),
                priority=0,
                parameter_path="oikos.bounty.min_reward_usd",
                current_value=_D(5.0),
                recommended_value=_D(2.0),
                delta=_D(-3.0),
                ruin_probability_before=_D(ruin_p),
                confidence=Decimal("0.75"),
            ))

        # 5. If survival reserve is underfunded, recommend reserve building
        reserve_deficit = _d(state.survival_reserve_deficit)
        if reserve_deficit > 0:
            recs.append(EconomicRecommendation(
                action="fund_survival_reserve",
                description=(
                    f"Survival reserve is ${reserve_deficit:.2f} below target. "
                    f"Redirect surplus to cold wallet until reserve is fully funded."
                ),
                priority=1,
                parameter_path="oikos.survival_reserve_days",
                current_value=state.survival_reserve,
                recommended_value=state.survival_reserve_target,
                delta=_D(reserve_deficit),
                ruin_probability_before=_D(ruin_p),
                confidence=Decimal("0.9"),
            ))

        # Sort by priority (lower = more urgent)
        recs.sort(key=lambda r: r.priority)

        return recs
