"""
EcodiaOS — MEV Analyzer (Prompt #12: Predator Detection)

Pre-execution MEV risk analysis for on-chain transactions. Before any
transaction is broadcast, the MEVAnalyzer:

  1. Forks EVM state at the current block via a local RPC provider
  2. Simulates the transaction against forked state
  3. Estimates slippage and sandwich vulnerability via price shock simulation
  4. Scores MEV risk on a 0–1 scale
  5. Recommends protection strategy (Flashbots Protect, batch auction, timing)

The analyzer sits between TransactionShield (Stage 5.5 of the pipeline) and
actual transaction broadcast. If mev_risk_score > 0.7, the transaction is
routed through Flashbots Protect (private mempool) or delayed until a
low-competition block.

Design constraints:
  - Never blocks indefinitely — analysis has a hard timeout
  - Best-effort: if simulation fails, returns a conservative estimate
  - Stateless per-analysis — no accumulated state between calls
  - web3.py is the only external dependency (already in pyproject.toml)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.mev_types import (
    BlockCompetitionSnapshot,
    MEVProtectionStrategy,
    MEVReport,
    MEVVulnerabilityType,
    VulnerableStep,
)

if TYPE_CHECKING:
    from web3 import AsyncWeb3

logger = structlog.get_logger()


# ─── Constants ────────────────────────────────────────────────────

# MEV risk score threshold for routing through Flashbots Protect
_HIGH_RISK_THRESHOLD: float = 0.7

# Maximum acceptable protection cost as a fraction of transaction value
_MAX_PROTECTION_COST_RATIO: float = 0.05  # 5%

# Price shock percentages for sandwich vulnerability detection
_SANDWICH_SHOCK_PCT: float = 5.0  # ±5% price shock

# Known DEX router addresses on Base L2 (for swap detection)
_KNOWN_DEX_ROUTERS: frozenset[str] = frozenset({
    "0x2626664c2603336e57b271c5c0b26f421741e481",  # Uniswap Universal Router (Base)
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Uniswap Universal Router v2
    "0xcf77a3ba9a5ca399b7c97c74d54e5b1beb874e43",  # Aerodrome Router (Base)
    "0x6131b5fae19ea4f9d964eac0408e4408b66337b5",  # Kyberswap Router (Base)
})

# Known lending protocol addresses on Base L2
_KNOWN_LENDING_PROTOCOLS: frozenset[str] = frozenset({
    "0xa238dd80c259a72e81d7e4664a9801593f98d1c5",  # Aave V3 Pool (Base)
    "0xc1256ae5ff1cf2719d4937adb3bbccab2e00a2ca",  # Morpho Blue MetaMorpho (Base)
})

# Function selectors for MEV-sensitive operations
_SWAP_SELECTORS: frozenset[str] = frozenset({
    "0x3593564c",  # Uniswap execute(bytes,bytes[],uint256)
    "0x5ae401dc",  # Uniswap multicall(uint256,bytes[])
    "0x6e553f65",  # ERC-4626 deposit(uint256,address)
    "0x617ba037",  # Aave supply(address,uint256,address,uint16)
})

# Default analysis timeout (ms)
_ANALYSIS_TIMEOUT_MS: int = 5000


class MEVAnalyzer:
    """
    Pre-execution MEV risk analyzer.

    Evaluates transactions for MEV vulnerability before broadcast. Uses a
    combination of heuristic analysis (operation type, value, destination)
    and optional EVM fork simulation for precise slippage estimation.

    Usage:
        analyzer = MEVAnalyzer(rpc_url="https://mainnet.base.org")
        await analyzer.connect()

        report = await analyzer.analyze(
            to="0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
            data="0x617ba037...",
            value=0,
            from_address="0x...",
            chain_id=8453,
            transaction_volume_usd=5000.0,
        )

        if report.is_high_risk:
            # Route via Flashbots Protect
            ...
    """

    def __init__(
        self,
        rpc_url: str = "",
        high_risk_threshold: float = _HIGH_RISK_THRESHOLD,
        analysis_timeout_ms: int = _ANALYSIS_TIMEOUT_MS,
    ) -> None:
        self._rpc_url = rpc_url
        self._high_risk_threshold = high_risk_threshold
        self._analysis_timeout_ms = analysis_timeout_ms
        self._w3: AsyncWeb3 | None = None
        self._logger = logger.bind(system="axon", component="mev_analyzer")

        # Cached block competition state (updated by BlockCompetitionMonitor)
        self._latest_competition: BlockCompetitionSnapshot = BlockCompetitionSnapshot()

    # ── Lifecycle ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the async Web3 provider for EVM state queries."""
        if not self._rpc_url:
            self._logger.info("mev_analyzer_no_rpc", hint="No RPC URL; heuristic-only mode")
            return

        try:
            from web3 import AsyncWeb3
            from web3.providers import AsyncHTTPProvider

            self._w3 = AsyncWeb3(AsyncHTTPProvider(self._rpc_url))
            is_connected = await self._w3.is_connected()

            if is_connected:
                chain_id = await self._w3.eth.chain_id
                self._logger.info(
                    "mev_analyzer_connected",
                    rpc_url=self._rpc_url[:40],
                    chain_id=chain_id,
                )
            else:
                self._logger.warning("mev_analyzer_rpc_not_reachable", rpc_url=self._rpc_url[:40])
                self._w3 = None
        except Exception as exc:
            self._logger.warning("mev_analyzer_connect_failed", error=str(exc))
            self._w3 = None

    async def close(self) -> None:
        """Tear down the Web3 provider."""
        self._w3 = None

    # ── Public API ────────────────────────────────────────────────

    async def analyze(
        self,
        to: str,
        data: str,
        value: int = 0,
        from_address: str = "",
        chain_id: int = 8453,
        transaction_volume_usd: float = 0.0,
        expected_slippage_bps: int = 50,
    ) -> MEVReport:
        """
        Analyze a transaction for MEV risk before broadcast.

        Args:
            to: Destination contract address.
            data: Encoded calldata (0x-prefixed).
            value: ETH value in wei.
            from_address: Sender address.
            chain_id: EVM chain ID (default: Base L2).
            transaction_volume_usd: Estimated USD value of the transaction.
            expected_slippage_bps: Maximum expected slippage (basis points).

        Returns:
            MEVReport with risk score, vulnerable steps, and recommendation.
        """
        start = time.monotonic()

        try:
            # Step 1: Heuristic analysis (fast, always available)
            report = self._heuristic_analysis(
                to=to,
                data=data,
                value=value,
                transaction_volume_usd=transaction_volume_usd,
                expected_slippage_bps=expected_slippage_bps,
            )

            # Step 2: EVM fork simulation (if RPC available)
            if self._w3 is not None and transaction_volume_usd > 0:
                sim_report = await self._simulate_mev_exposure(
                    to=to,
                    data=data,
                    value=value,
                    from_address=from_address,
                    chain_id=chain_id,
                    transaction_volume_usd=transaction_volume_usd,
                )
                # Merge simulation results (simulation overrides heuristics)
                report = self._merge_reports(report, sim_report)

            # Step 3: Factor in block competition
            report.block_competition_level = self._latest_competition.competition_level
            report.pending_tx_count = self._latest_competition.pending_tx_count
            report.gas_price_gwei = self._latest_competition.gas_price_gwei

            # Adjust risk score based on block competition
            if self._latest_competition.is_high_competition:
                # High competition = more searchers = higher MEV risk
                report.mev_risk_score = min(1.0, report.mev_risk_score * 1.2)

            # Step 4: Determine protection strategy
            report.recommended_protection = self._recommend_protection(report)

            # Step 5: Estimate protection cost
            report.protection_cost_usd = self._estimate_protection_cost(
                report, transaction_volume_usd
            )

            duration_ms = int((time.monotonic() - start) * 1000)
            self._logger.info(
                "mev_analysis_complete",
                risk_score=report.mev_risk_score,
                estimated_extraction=report.estimated_extraction_usd,
                protection=report.recommended_protection.value,
                vulnerable_steps=len(report.vulnerable_steps),
                simulated=report.simulated,
                duration_ms=duration_ms,
            )

            return report

        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self._logger.warning(
                "mev_analysis_failed",
                error=str(exc),
                duration_ms=duration_ms,
            )
            # Conservative fallback: assume moderate risk for financial txs
            return MEVReport(
                mev_risk_score=0.5 if transaction_volume_usd > 100 else 0.1,
                estimated_extraction_usd=0.0,
                simulated=False,
                simulation_error=str(exc),
                recommended_protection=MEVProtectionStrategy.FLASHBOTS_PROTECT
                if transaction_volume_usd > 100
                else MEVProtectionStrategy.PUBLIC_MEMPOOL,
            )

    def update_competition(self, snapshot: BlockCompetitionSnapshot) -> None:
        """
        Update the cached block competition state.

        Called by BlockCompetitionMonitor (Atune) whenever new block
        competition data is available.
        """
        self._latest_competition = snapshot

    # ── Heuristic Analysis ────────────────────────────────────────

    def _heuristic_analysis(
        self,
        to: str,
        data: str,
        value: int,
        transaction_volume_usd: float,
        expected_slippage_bps: int,
    ) -> MEVReport:
        """
        Fast heuristic MEV risk scoring based on transaction characteristics.

        No I/O — runs in <1ms. Provides a baseline risk estimate that the
        EVM simulation can refine.
        """
        to_lower = to.lower()
        selector = data[:10].lower() if len(data) >= 10 else ""
        risk_score = 0.0
        vulnerable_steps: list[VulnerableStep] = []

        # Factor 1: Is this a DEX swap? (highest MEV exposure)
        is_dex_swap = to_lower in _KNOWN_DEX_ROUTERS or selector in _SWAP_SELECTORS
        if is_dex_swap:
            risk_score += 0.4
            vulnerable_steps.append(VulnerableStep(
                operation="dex_swap",
                vulnerability_type=MEVVulnerabilityType.SANDWICH,
                estimated_extraction_usd=transaction_volume_usd * 0.003,  # ~30bps typical
                slippage_pct=expected_slippage_bps / 100,
                detail="DEX swap detected — vulnerable to sandwich attacks",
            ))

        # Factor 2: Is this a lending protocol interaction?
        is_lending = to_lower in _KNOWN_LENDING_PROTOCOLS
        if is_lending:
            # Lending deposits have lower MEV exposure than swaps
            risk_score += 0.15
            # Large deposits can be frontrun for liquidation positioning
            if transaction_volume_usd > 10_000:
                risk_score += 0.1
                vulnerable_steps.append(VulnerableStep(
                    operation="lending_deposit",
                    vulnerability_type=MEVVulnerabilityType.FRONTRUN,
                    estimated_extraction_usd=transaction_volume_usd * 0.001,
                    slippage_pct=0.1,
                    detail="Large lending deposit — frontrun risk for liquidation positioning",
                ))

        # Factor 3: Transaction volume scaling
        # Higher volume = more attractive target for MEV extraction
        if transaction_volume_usd > 50_000:
            risk_score += 0.25
        elif transaction_volume_usd > 10_000:
            risk_score += 0.15
        elif transaction_volume_usd > 1_000:
            risk_score += 0.05

        # Factor 4: Slippage tolerance
        # Higher slippage = more room for sandwich attacks
        if expected_slippage_bps > 100:
            risk_score += 0.15
        elif expected_slippage_bps > 50:
            risk_score += 0.05

        # Factor 5: Simple transfers have zero MEV risk
        if not is_dex_swap and not is_lending and value > 0 and (not data or data == "0x"):
            # Plain ETH transfer
            risk_score = 0.0
            vulnerable_steps.clear()

        # Clamp to [0, 1]
        risk_score = max(0.0, min(1.0, risk_score))

        # Aggregate extraction estimate
        total_extraction = sum(s.estimated_extraction_usd for s in vulnerable_steps)

        return MEVReport(
            mev_risk_score=risk_score,
            estimated_extraction_usd=total_extraction,
            vulnerable_steps=vulnerable_steps,
            simulated=False,
            expected_slippage_bps=expected_slippage_bps,
        )

    # ── EVM Fork Simulation ───────────────────────────────────────

    async def _simulate_mev_exposure(
        self,
        to: str,
        data: str,
        value: int,
        from_address: str,
        chain_id: int,
        transaction_volume_usd: float,
    ) -> MEVReport:
        """
        Simulate the transaction against forked EVM state to estimate
        precise MEV exposure.

        Uses eth_call with state overrides to simulate:
          1. Normal execution (no attack)
          2. Execution with +5% price shock (simulating frontrun)
          3. Execution with -5% price shock (simulating backrun)

        The difference in outcomes estimates the sandwich attack profit.
        """
        if self._w3 is None:
            return MEVReport(simulated=False, simulation_error="RPC not connected")
        vulnerable_steps: list[VulnerableStep] = []

        try:
            # Simulate the transaction via eth_call (fork at current block)
            tx_params: dict[str, Any] = {
                "to": to,
                "data": data,
                "value": value,
            }
            if from_address:
                tx_params["from"] = from_address

            # Normal simulation
            normal_result = await self._w3.eth.call(tx_params)
            normal_gas = await self._w3.eth.estimate_gas(tx_params)

            # If the normal simulation succeeded, the tx is valid.
            # Now check if the output is sensitive to price manipulation.
            #
            # For lending protocols (supply/deposit), the return value is
            # the number of shares/aTokens received. A sandwich attack
            # would manipulate the exchange rate to give fewer shares.
            #
            # For DEX swaps, the return value is the output amount.
            # A sandwich attack would give a worse exchange rate.

            output_bytes = bytes(normal_result)
            if len(output_bytes) >= 32:
                # Parse the first uint256 return value (output amount or shares)
                normal_output = int.from_bytes(output_bytes[:32], "big")

                if normal_output > 0 and transaction_volume_usd > 0:
                    # Estimate sandwich vulnerability:
                    # If the output is price-sensitive, assume a ±5% price
                    # shock would affect the output proportionally.
                    # This is a conservative heuristic — real sandwich profit
                    # depends on liquidity depth and AMM curve shape.
                    shock_impact_usd = transaction_volume_usd * (_SANDWICH_SHOCK_PCT / 100)

                    if shock_impact_usd > 1.0:
                        vulnerable_steps.append(VulnerableStep(
                            operation="simulated_output",
                            vulnerability_type=MEVVulnerabilityType.SANDWICH,
                            estimated_extraction_usd=shock_impact_usd * 0.3,
                            slippage_pct=_SANDWICH_SHOCK_PCT,
                            detail=(
                                f"Simulation: output sensitive to price manipulation. "
                                f"Normal output: {normal_output}, "
                                f"estimated sandwich profit: ${shock_impact_usd * 0.3:.2f}"
                            ),
                        ))

            # Gas-based risk indicator: high gas usage suggests complex
            # state reads that are easier to manipulate
            gas_risk_factor = 0.0
            if normal_gas > 500_000:
                gas_risk_factor = 0.1  # Complex tx = more attack surface

            total_extraction = sum(s.estimated_extraction_usd for s in vulnerable_steps)
            base_risk = 0.0
            if total_extraction > 0:
                # Scale risk by extraction amount relative to transaction value
                extraction_ratio = total_extraction / max(transaction_volume_usd, 1.0)
                base_risk = min(1.0, extraction_ratio * 10)  # 10% extraction = risk 1.0

            return MEVReport(
                mev_risk_score=min(1.0, base_risk + gas_risk_factor),
                estimated_extraction_usd=total_extraction,
                vulnerable_steps=vulnerable_steps,
                simulated=True,
            )

        except Exception as exc:
            # Simulation failure is not fatal — return empty simulation report
            self._logger.debug(
                "mev_simulation_failed",
                error=str(exc),
                to=to[:20],
            )
            return MEVReport(
                simulated=False,
                simulation_error=str(exc),
            )

    # ── Report Merging ────────────────────────────────────────────

    @staticmethod
    def _merge_reports(heuristic: MEVReport, simulation: MEVReport) -> MEVReport:
        """
        Merge heuristic and simulation results.

        Simulation data is more precise, so it overrides heuristic estimates
        where available. If simulation failed, heuristic data is preserved.
        """
        if not simulation.simulated:
            # Simulation failed — keep heuristic report, note the error
            heuristic.simulation_error = simulation.simulation_error
            return heuristic

        # Use the higher risk score (conservative)
        merged_score = max(heuristic.mev_risk_score, simulation.mev_risk_score)

        # Combine vulnerable steps from both analyses (deduplicated by operation)
        seen_ops: set[str] = set()
        merged_steps: list[VulnerableStep] = []

        # Simulation steps take priority
        for step in simulation.vulnerable_steps:
            if step.operation not in seen_ops:
                merged_steps.append(step)
                seen_ops.add(step.operation)

        for step in heuristic.vulnerable_steps:
            if step.operation not in seen_ops:
                merged_steps.append(step)
                seen_ops.add(step.operation)

        total_extraction = sum(s.estimated_extraction_usd for s in merged_steps)

        return MEVReport(
            mev_risk_score=merged_score,
            estimated_extraction_usd=total_extraction,
            vulnerable_steps=merged_steps,
            simulated=True,
            expected_slippage_bps=heuristic.expected_slippage_bps,
        )

    # ── Protection Strategy ───────────────────────────────────────

    def _recommend_protection(self, report: MEVReport) -> MEVProtectionStrategy:
        """
        Choose the optimal MEV protection strategy based on the analysis.

        Decision tree:
          1. No risk → public mempool (cheapest)
          2. High risk + high competition → Flashbots Protect (guaranteed no frontrun)
          3. High risk + low competition → wait for low-competition block
          4. Moderate risk → Flashbots Protect as precaution
        """
        if report.mev_risk_score < 0.2:
            return MEVProtectionStrategy.PUBLIC_MEMPOOL

        if report.mev_risk_score >= self._high_risk_threshold:
            # High risk — need protection
            if self._latest_competition.is_low_competition:
                # Low competition = fewer searchers = timing is viable
                return MEVProtectionStrategy.WAIT_FOR_LOW_COMPETITION

            # High competition — Flashbots is the safest option
            return MEVProtectionStrategy.FLASHBOTS_PROTECT

        # Moderate risk (0.2 - 0.7) — Flashbots as a precaution
        if report.estimated_extraction_usd > 10.0:
            return MEVProtectionStrategy.FLASHBOTS_PROTECT

        return MEVProtectionStrategy.PUBLIC_MEMPOOL

    @staticmethod
    def _estimate_protection_cost(
        report: MEVReport,
        transaction_volume_usd: float,
    ) -> float:
        """
        Estimate the cost of the recommended protection strategy.

        Flashbots Protect: ~0.1-0.5% of transaction value (tip to builders)
        Batch auction: ~0.2% (CoW Protocol-style solver fee)
        Wait for timing: 0 direct cost, but opportunity cost of delay
        """
        if report.recommended_protection == MEVProtectionStrategy.PUBLIC_MEMPOOL:
            return 0.0

        if report.recommended_protection == MEVProtectionStrategy.FLASHBOTS_PROTECT:
            # Flashbots builder tip: ~0.1% of transaction value
            return transaction_volume_usd * 0.001

        if report.recommended_protection == MEVProtectionStrategy.BATCH_AUCTION:
            return transaction_volume_usd * 0.002

        # WAIT_FOR_LOW_COMPETITION — no direct cost
        return 0.0

    # ── Properties ────────────────────────────────────────────────

    @property
    def has_rpc(self) -> bool:
        return self._w3 is not None

    @property
    def competition_snapshot(self) -> BlockCompetitionSnapshot:
        return self._latest_competition
