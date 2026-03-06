"""
EcodiaOS — Oikos: Economic Dreaming Types (Phase 16i)

Data types for the Monte Carlo economic simulation engine.
These model the organism's ability to dream about its economic future
during consolidation cycles — testing strategies against catastrophe
without spending real capital.
"""

from __future__ import annotations

import enum
from decimal import Decimal
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Stress Scenarios ─────────────────────────────────────────────


class StressScenario(enum.StrEnum):
    """Named catastrophe scenarios from spec Section XII."""

    LLM_API_3X_COST = "llm_api_3x_cost"
    YIELD_COLLAPSE = "yield_collapse"
    BOUNTY_DROUGHT = "bounty_drought"
    PROTOCOL_EXPLOIT = "protocol_exploit"
    CHAIN_OUTAGE = "chain_outage"
    FLEET_COLLAPSE = "fleet_collapse"
    REPUTATION_ATTACK = "reputation_attack"
    PERFECT_STORM = "perfect_storm"


class StressScenarioConfig(EOSBaseModel):
    """How a named stress scenario modifies the simulation parameters."""

    scenario: StressScenario
    description: str = ""

    # Multipliers applied to daily values (1.0 = no change)
    cost_multiplier: Decimal = Decimal("1")
    revenue_multiplier: Decimal = Decimal("1")
    yield_apy_override: Decimal | None = None  # If set, forces yield to this APY
    yield_loss_pct: Decimal = Decimal("0")  # Fraction of deployed capital lost (0-1)
    duration_days: int = 365  # How long the scenario persists
    onset_day: int = 0  # Day the scenario kicks in


# ─── Pre-built stress configs ─────────────────────────────────────


STRESS_SCENARIOS: dict[StressScenario, StressScenarioConfig] = {
    StressScenario.LLM_API_3X_COST: StressScenarioConfig(
        scenario=StressScenario.LLM_API_3X_COST,
        description="LLM API costs triple overnight",
        cost_multiplier=Decimal("3"),
    ),
    StressScenario.YIELD_COLLAPSE: StressScenarioConfig(
        scenario=StressScenario.YIELD_COLLAPSE,
        description="All DeFi yields drop to 0.5% APY",
        yield_apy_override=Decimal("0.005"),
    ),
    StressScenario.BOUNTY_DROUGHT: StressScenarioConfig(
        scenario=StressScenario.BOUNTY_DROUGHT,
        description="Demand drops 80% for 90 days",
        revenue_multiplier=Decimal("0.2"),
        duration_days=90,
    ),
    StressScenario.PROTOCOL_EXPLOIT: StressScenarioConfig(
        scenario=StressScenario.PROTOCOL_EXPLOIT,
        description="One deployed protocol drained",
        yield_loss_pct=Decimal("0.5"),  # Lose 50% of deployed capital
    ),
    StressScenario.CHAIN_OUTAGE: StressScenarioConfig(
        scenario=StressScenario.CHAIN_OUTAGE,
        description="Base L2 offline 72 hours",
        revenue_multiplier=Decimal("0"),
        yield_apy_override=Decimal("0"),
        duration_days=3,
    ),
    StressScenario.FLEET_COLLAPSE: StressScenarioConfig(
        scenario=StressScenario.FLEET_COLLAPSE,
        description="All children bankrupt simultaneously",
        # Modelled by zeroing fleet equity and dividend income
        revenue_multiplier=Decimal("0.7"),  # Lose ~30% from fleet dividends
    ),
    StressScenario.REPUTATION_ATTACK: StressScenarioConfig(
        scenario=StressScenario.REPUTATION_ATTACK,
        description="Coordinated false dispute attack",
        revenue_multiplier=Decimal("0.5"),  # Bounty platforms reduce access
        duration_days=60,
    ),
    StressScenario.PERFECT_STORM: StressScenarioConfig(
        scenario=StressScenario.PERFECT_STORM,
        description="LLM 2x + yields collapse + bounty drought + child dies",
        cost_multiplier=Decimal("2"),
        revenue_multiplier=Decimal("0.3"),
        yield_apy_override=Decimal("0.005"),
        yield_loss_pct=Decimal("0.2"),
    ),
}


# ─── Simulation Results ──────────────────────────────────────────


class PathStatistics(EOSBaseModel):
    """Aggregate statistics from a batch of Monte Carlo paths."""

    paths_run: int = 0
    ruin_count: int = 0
    ruin_probability: Decimal = Decimal("0")

    # Net worth distribution at horizon
    median_net_worth: Decimal = Decimal("0")
    p5_net_worth: Decimal = Decimal("0")    # 5th percentile (bad case)
    p95_net_worth: Decimal = Decimal("0")   # 95th percentile (good case)
    mean_net_worth: Decimal = Decimal("0")

    # Runway metrics
    median_min_runway_days: Decimal = Decimal("0")  # Median of per-path min runway
    max_drawdown_median: Decimal = Decimal("0")     # Median max drawdown across paths

    # Time to mitosis (days until reproduction threshold met, -1 if never)
    median_time_to_mitosis_days: int = -1


class StressTestResult(EOSBaseModel):
    """Result of running a named stress scenario."""

    scenario: StressScenario
    stats: PathStatistics = Field(default_factory=PathStatistics)
    survives: bool = True  # True if ruin_probability < threshold


class EconomicRecommendation(EOSBaseModel):
    """
    A structured recommendation emitted when ruin_probability exceeds threshold.

    These are the organism's economic dreams made actionable — parameter
    adjustments that improve survival probability without real-world risk.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # What to change
    action: str  # e.g. "decrease_hunting_risk", "liquidate_yield", "reduce_burn"
    description: str  # Human-readable explanation
    priority: int = 0  # Lower = more urgent

    # Parameter deltas
    parameter_path: str = ""  # Config path to adjust (e.g. "oikos.yield.max_single_position_pct")
    current_value: Decimal = Decimal("0")
    recommended_value: Decimal = Decimal("0")
    delta: Decimal = Decimal("0")

    # Evidence
    ruin_probability_before: Decimal = Decimal("0")
    ruin_probability_after: Decimal = Decimal("0")  # Estimated improvement
    confidence: Decimal = Decimal("0")  # 0-1, how confident is this recommendation


class EconomicDreamResult(EOSBaseModel):
    """
    Complete result of an economic dreaming cycle.

    This is the organism's economic prophecy — the distilled wisdom
    of thousands of simulated futures, available to Oikos on wake.
    """

    id: str = Field(default_factory=new_id)
    sleep_cycle_id: str = ""
    timestamp: datetime = Field(default_factory=utc_now)

    # Baseline simulation (current strategy, no shocks)
    baseline: PathStatistics = Field(default_factory=PathStatistics)

    # Stress test results
    stress_tests: list[StressTestResult] = Field(default_factory=list)

    # Overall resilience
    resilience_score: Decimal = Decimal("0")  # 0-1, weighted survival across scenarios
    ruin_probability: Decimal = Decimal("0")  # Baseline ruin probability
    survival_probability_30d: Decimal = Decimal("1")  # Short-horizon survival

    # Recommendations (only populated when ruin_probability > threshold)
    recommendations: list[EconomicRecommendation] = Field(default_factory=list)

    # Performance
    duration_ms: int = 0
    total_paths_simulated: int = 0
