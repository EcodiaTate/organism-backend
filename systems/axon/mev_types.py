"""
EcodiaOS — MEV Analyzer Types (Prompt #12: Predator Detection)

Types for MEV risk analysis and transaction protection routing.

MEV (Maximal Extractable Value) represents value extraction by searchers who
observe pending transactions in the mempool and frontrun, sandwich, or backrun
them. These types model the risk assessment and protection decision.

The MEVReport is produced by MEVAnalyzer before any on-chain transaction and
consumed by TransactionShield to decide whether to route via Flashbots Protect
(private mempool) or the public mempool.
"""

from __future__ import annotations

import enum

from pydantic import Field

from primitives.common import EOSBaseModel


class MEVProtectionStrategy(enum.StrEnum):
    """Recommended MEV protection approach."""

    PUBLIC_MEMPOOL = "public_mempool"
    FLASHBOTS_PROTECT = "flashbots_protect"
    BATCH_AUCTION = "batch_auction"
    WAIT_FOR_LOW_COMPETITION = "wait_for_low_competition"


class MEVVulnerabilityType(enum.StrEnum):
    """Categories of MEV vulnerability detected in a transaction."""

    SANDWICH = "sandwich"
    FRONTRUN = "frontrun"
    BACKRUN = "backrun"
    LIQUIDATION_SNIPE = "liquidation_snipe"
    ARBITRAGE_EXTRACTION = "arbitrage_extraction"


class VulnerableStep(EOSBaseModel):
    """A single operation within a transaction that is vulnerable to MEV."""

    operation: str  # e.g. "swap_usdc_to_eth", "aave_supply"
    vulnerability_type: MEVVulnerabilityType
    estimated_extraction_usd: float = 0.0
    slippage_pct: float = 0.0
    detail: str = ""


class MEVReport(EOSBaseModel):
    """
    Result of pre-execution MEV risk analysis.

    Produced by MEVAnalyzer.analyze() before any on-chain transaction.
    Score of 0 = no MEV exposure (e.g., simple ETH transfer).
    Score of 1 = extreme exposure (e.g., large swap with high slippage).
    """

    mev_risk_score: float = Field(ge=0.0, le=1.0, default=0.0)
    estimated_extraction_usd: float = 0.0
    vulnerable_steps: list[VulnerableStep] = Field(default_factory=list)
    recommended_protection: MEVProtectionStrategy = MEVProtectionStrategy.PUBLIC_MEMPOOL

    # Simulation details
    simulated: bool = False
    simulation_error: str = ""

    # Price impact analysis
    expected_slippage_bps: int = 0
    slippage_with_attack_bps: int = 0

    # Block competition context (from Atune)
    block_competition_level: float = 0.0  # 0=empty, 1=congested
    pending_tx_count: int = 0
    gas_price_gwei: float = 0.0

    # Protection cost estimate
    protection_cost_usd: float = 0.0

    @property
    def is_high_risk(self) -> bool:
        return self.mev_risk_score > 0.7

    @property
    def vulnerable_operation_names(self) -> list[str]:
        return [s.operation for s in self.vulnerable_steps]

    def summary(self) -> str:
        """Human-readable summary for logging."""
        protection = self.recommended_protection.value
        vuln_count = len(self.vulnerable_steps)
        return (
            f"mev_risk={self.mev_risk_score:.2f} "
            f"estimated_extraction=${self.estimated_extraction_usd:.2f} "
            f"vulnerable_steps={vuln_count} "
            f"protection={protection}"
        )


class BlockCompetitionSnapshot(EOSBaseModel):
    """Point-in-time snapshot of block space competition."""

    gas_price_gwei: float = 0.0
    base_fee_gwei: float = 0.0
    pending_tx_count: int = 0
    block_utilization_pct: float = 0.0  # 0-100
    competition_level: float = 0.0  # 0=low, 1=high (normalised)
    timestamp_ms: int = 0

    @property
    def is_low_competition(self) -> bool:
        return self.competition_level < 0.3

    @property
    def is_high_competition(self) -> bool:
        return self.competition_level > 0.7
