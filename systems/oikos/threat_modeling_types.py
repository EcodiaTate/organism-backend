"""
EcodiaOS — Oikos: Treasury Threat Modeling Types (Phase 16i+)

Data types for per-asset Monte Carlo threat modeling during consolidation.
While EconomicSimulator (Phase 16i) models organism-level cashflow via GBM,
these types support the complementary layer: per-asset shock distributions,
contagion coupling, liquidation detection, and hedging proposals.

The organism dreams about its treasury exposures — testing each position
against adversarial scenarios and pre-computing hedges before crises hit.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003 — needed at runtime for Pydantic field resolution
from decimal import Decimal

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Asset Classification ─────────────────────────────────────────


class AssetClass(enum.StrEnum):
    """Classification of treasury assets by risk profile."""

    NATIVE_ETH = "native_eth"        # ETH — fat-tailed, high vol
    LST = "lst"                      # Liquid staking tokens (stETH, rETH)
    STABLECOIN = "stablecoin"        # USDC, USDT, DAI — depeg risk model
    LP_TOKEN = "lp_token"            # Liquidity pool tokens — impermanent loss
    YIELD_BEARING = "yield_bearing"  # aTokens, cTokens — protocol risk
    DERIVATIVE = "derivative"        # Futures collateral positions


# ─── Shock Distributions ──────────────────────────────────────────


class AssetShockDistribution(EOSBaseModel):
    """
    Per-asset shock distribution parameters.

    Unlike the organism-level GBM in EconomicSimulator, this uses a
    mixture model: normal regime + jump regime + depeg regime (stablecoins).
    Each asset class has calibrated defaults that can be overridden.
    """

    asset_class: AssetClass
    symbol: str  # e.g. "ETH", "stETH", "USDC"

    # Normal regime (GBM-like, daily)
    daily_drift: Decimal = Decimal("0")       # mu / 365
    daily_volatility: Decimal = Decimal("0")  # sigma / sqrt(365)

    # Jump regime (Poisson process)
    jump_probability: Decimal = Decimal("0")  # per-day probability of jump event
    jump_mean: Decimal = Decimal("0")         # mean jump size (negative = crash)
    jump_std: Decimal = Decimal("0")          # jump size std dev

    # Depeg regime (stablecoins only)
    depeg_probability: Decimal = Decimal("0")  # per-day probability of depeg event
    depeg_recovery_days: int = 7               # expected days to re-peg
    depeg_magnitude: Decimal = Decimal("0")    # depth of depeg (e.g. 0.05 = drops to $0.95)


# ─── Contagion Model ─────────────────────────────────────────────


class ContagionEdge(EOSBaseModel):
    """
    Directional contagion between two assets.

    During normal markets, assets correlate at base_correlation.
    When the source asset draws down beyond stress_threshold,
    correlation jumps to stress_correlation — capturing the
    real-world phenomenon of spiking correlations during crises.
    """

    source_symbol: str    # e.g. "ETH"
    target_symbol: str    # e.g. "stETH"
    base_correlation: Decimal = Decimal("0.5")    # normal-regime correlation
    stress_correlation: Decimal = Decimal("0.95")  # correlation during stress
    stress_threshold: Decimal = Decimal("-0.10")   # source drawdown that triggers stress
    lag_days: int = 0     # contagion delay (0 = same day)


# ─── Treasury Position ────────────────────────────────────────────


class TreasuryPosition(EOSBaseModel):
    """
    A single treasury position for threat modeling.

    Richer than YieldPosition — adds asset class, liquidation
    parameters, and collateral ratios for DeFi positions.
    """

    position_id: str = Field(default_factory=new_id)
    symbol: str                        # "ETH", "stETH", "USDC"
    asset_class: AssetClass
    principal_usd: Decimal             # Current USD value
    protocol: str = ""                 # "aave", "lido", "compound", ""
    pool: str = ""                     # Pool identifier
    chain_id: int = 8453               # Base L2

    # DeFi liquidation parameters
    has_liquidation_threshold: bool = False
    liquidation_price_usd: Decimal = Decimal("0")  # Price at which position liquidates
    collateral_ratio: Decimal = Decimal("0")        # Current collateral ratio
    min_collateral_ratio: Decimal = Decimal("0")    # Liquidation trigger ratio

    # Sourced from YieldPosition
    health_status: str = "unknown"


# ─── Tail Risk Profile ───────────────────────────────────────────


class TailRiskProfile(EOSBaseModel):
    """
    Tail risk metrics for a position or portfolio.

    VaR: maximum loss at a given confidence level.
    CVaR: expected loss in the tail beyond VaR.
    """

    var_5pct: Decimal = Decimal("0")       # Value at Risk at 5th percentile (worst 5%)
    var_25pct: Decimal = Decimal("0")      # VaR at 25th percentile
    cvar_5pct: Decimal = Decimal("0")      # Conditional VaR (expected loss beyond 5th pct)
    max_drawdown_median: Decimal = Decimal("0")  # Median maximum drawdown across paths
    max_drawdown_p95: Decimal = Decimal("0")     # 95th percentile max drawdown
    liquidation_probability: Decimal = Decimal("0")  # P(liquidation event)
    expected_liquidation_loss: Decimal = Decimal("0")  # E[loss | liquidation]
    time_to_liquidation_p10: int = -1   # 10th percentile days to first liquidation (-1 = never)


# ─── Critical Exposure ────────────────────────────────────────────


class CriticalExposure(EOSBaseModel):
    """
    An asset/position identified as a critical risk driver.

    Ranked by marginal contribution to portfolio VaR — the position
    whose removal most reduces portfolio-level tail risk ranks highest.
    """

    position_id: str
    symbol: str
    asset_class: AssetClass
    exposure_usd: Decimal
    contribution_to_portfolio_var: Decimal  # Marginal VaR contribution (0-1)
    contagion_amplifier: Decimal = Decimal("1")  # How much this position amplifies contagion
    risk_rank: int = 0  # 1 = highest risk
    rationale: str = ""


# ─── Hedging Proposal ────────────────────────────────────────────


class HedgingProposal(EOSBaseModel):
    """
    A concrete hedging recommendation with sizing.

    Pre-computed during sleep so that Nova can immediately deploy
    the hedge when matching market conditions arrive on wake.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # What to hedge
    target_position_id: str
    target_symbol: str

    # Hedge instrument
    hedge_action: str  # "sell_to_stable", "reduce_position", "diversify_protocol", "add_collateral"
    hedge_instrument: str = ""  # e.g. "USDC", "hedge_contract_address"
    hedge_size_usd: Decimal = Decimal("0")   # Dollar amount to hedge
    hedge_size_pct: Decimal = Decimal("0")   # As percentage of position

    # Expected impact
    var_reduction_pct: Decimal = Decimal("0")      # Expected VaR improvement
    liquidation_prob_reduction: Decimal = Decimal("0")  # Expected P(liq) improvement
    cost_estimate_usd: Decimal = Decimal("0")      # Estimated cost to execute

    # Priority and confidence
    priority: int = 0  # Lower = more urgent
    confidence: Decimal = Decimal("0.5")
    description: str = ""


# ─── Threat Model Result ─────────────────────────────────────────


class ThreatModelResult(EOSBaseModel):
    """
    Complete result of a treasury threat modeling cycle.

    This is the organism's adversarial economic prophecy — the distilled
    wisdom of thousands of simulated asset-level shocks, contagion cascades,
    and liquidation events, with pre-computed hedges ready for Nova on wake.
    """

    id: str = Field(default_factory=new_id)
    sleep_cycle_id: str = ""
    timestamp: datetime = Field(default_factory=utc_now)

    # Portfolio-level tail risk
    portfolio_risk: TailRiskProfile = Field(default_factory=TailRiskProfile)

    # Per-position tail risk (keyed by position_id)
    position_risks: dict[str, TailRiskProfile] = Field(default_factory=dict)

    # Critical exposures (sorted by risk_rank)
    critical_exposures: list[CriticalExposure] = Field(default_factory=list)

    # Hedging proposals (sorted by priority)
    hedging_proposals: list[HedgingProposal] = Field(default_factory=list)

    # Contagion analysis
    contagion_events_detected: int = 0  # How many paths triggered contagion cascade
    contagion_loss_amplifier: Decimal = Decimal("1")  # Average loss multiplier from contagion

    # Simulation metadata
    total_paths_simulated: int = 0
    horizon_days: int = 0
    duration_ms: int = 0
    positions_analyzed: int = 0
