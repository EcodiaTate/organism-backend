"""
EcodiaOS - Transaction Shield (Layer 1: Economic Immune System)

Pre-execution filter that evaluates transactions before they are broadcast
on-chain. Unlike sentinels (which detect and report), the shield PREVENTS
bad transactions from ever reaching the network.

Checks performed:
  1. Destination address blacklist
  2. Simulated slippage enforcement (max 50 bps)
  3. Gas cost vs expected ROI validation
  4. MEV risk heuristics (optional, via eth_call simulation)
  5. MEV Predator Detection - full MEV risk analysis via MEVAnalyzer
     (Prompt #12: sandwich/frontrun/backrun detection with Flashbots routing)

The shield is wired into ExecutionPipeline as Stage 5.5 -- after context
assembly but before step execution. It only activates for financial
executors (defi_yield, wallet_transfer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.axon.mev_analyzer import MEVAnalyzer
    from systems.axon.mev_types import MEVReport


# ── Local shield types (avoid cross-system import from Thymos) ────


@dataclass
class AddressBlacklistEntry:
    """A blacklisted on-chain address - Axon-local mirror."""

    address: str
    chain_id: int = 8453
    reason: str = ""
    threat_type: str = "unknown"
    source: str = "local"
    source_instance_id: str = ""
    confirmed: bool = False


@dataclass
class SimulationResult:
    """Result of pre-simulating a transaction - Axon-local mirror."""

    passed: bool = True
    revert_reason: str = ""
    gas_used: int = 0
    value_delta_usd: float = 0.0
    slippage_bps: int = 0
    mev_risk_detected: bool = False
    warnings: list[str] = field(default_factory=list)

logger = structlog.get_logger()

# Maximum slippage the shield will permit (basis points)
_DEFAULT_MAX_SLIPPAGE_BPS: int = 50

# Minimum expected ROI (USD) to justify gas costs
_MIN_ROI_TO_GAS_RATIO: float = 2.0

# Executors that require shield evaluation
SHIELDED_EXECUTORS: frozenset[str] = frozenset({"defi_yield", "wallet_transfer"})


class TransactionShield:
    """
    Pre-execution transaction filter.

    Evaluates financial transactions before broadcast. Rejects transactions
    that fail blacklist checks, slippage limits, or gas/ROI analysis.

    NOT a sentinel -- it prevents, not detects. Lives in Axon, not in the
    immune system module.
    """

    def __init__(
        self,
        wallet: Any = None,
        max_slippage_bps: int = _DEFAULT_MAX_SLIPPAGE_BPS,
        mev_analyzer: MEVAnalyzer | None = None,
    ) -> None:
        self._wallet = wallet
        self._max_slippage_bps = max_slippage_bps
        self._mev_analyzer = mev_analyzer
        self._blacklist: dict[str, AddressBlacklistEntry] = {}
        self._logger = logger.bind(system="axon", component="transaction_shield")

        # Metrics
        self._total_evaluated: int = 0
        self._total_rejected: int = 0
        self._total_mev_protected: int = 0
        self._total_mev_saved_usd: float = 0.0
        self._last_mev_report: MEVReport | None = None

    # -- Public API ----------------------------------------------------------

    async def evaluate(
        self,
        action_type: str,
        params: dict[str, Any],
        context: Any = None,
    ) -> SimulationResult:
        """
        Evaluate a financial transaction before execution.

        Returns a SimulationResult. If ``passed`` is False, the pipeline
        should abort with TRANSACTION_SHIELD_REJECTED.

        Only evaluates executors in SHIELDED_EXECUTORS. Others pass through.
        """
        if action_type not in SHIELDED_EXECUTORS:
            return SimulationResult(passed=True)

        self._total_evaluated += 1
        warnings: list[str] = []

        # -- Check 1: Blacklist -----------------------------------------------
        destination = self._extract_destination(action_type, params)
        chain_id = params.get("chain_id", 8453)

        if destination and self.is_blacklisted(destination, chain_id):
            self._total_rejected += 1
            entry = self._blacklist.get(self._blacklist_key(destination, chain_id))
            reason = entry.reason if entry else "unknown"
            self._logger.warning(
                "shield_blacklisted_address",
                address=destination,
                reason=reason,
            )
            return SimulationResult(
                passed=False,
                revert_reason=f"Destination {destination} is blacklisted: {reason}",
                warnings=[f"Blacklisted address: {destination}"],
            )

        # -- Check 2: Slippage enforcement ------------------------------------
        slippage_bps = params.get("max_slippage_bps", 0)
        if slippage_bps > self._max_slippage_bps:
            self._total_rejected += 1
            self._logger.warning(
                "shield_slippage_exceeded",
                requested_bps=slippage_bps,
                max_bps=self._max_slippage_bps,
            )
            return SimulationResult(
                passed=False,
                slippage_bps=slippage_bps,
                revert_reason=(
                    f"Slippage {slippage_bps} bps exceeds maximum "
                    f"{self._max_slippage_bps} bps"
                ),
                warnings=[f"Slippage {slippage_bps}bps > {self._max_slippage_bps}bps cap"],
            )

        # -- Check 3: Gas cost vs expected ROI --------------------------------
        expected_roi_usd = float(params.get("expected_roi_usd", 0))
        estimated_gas_usd = float(params.get("estimated_gas_usd", 0))

        if estimated_gas_usd > 0 and expected_roi_usd > 0:
            ratio = expected_roi_usd / estimated_gas_usd
            if ratio < _MIN_ROI_TO_GAS_RATIO:
                self._total_rejected += 1
                self._logger.warning(
                    "shield_gas_exceeds_roi",
                    roi_usd=expected_roi_usd,
                    gas_usd=estimated_gas_usd,
                    ratio=ratio,
                )
                return SimulationResult(
                    passed=False,
                    gas_used=int(estimated_gas_usd * 1e6),  # approximate
                    value_delta_usd=expected_roi_usd - estimated_gas_usd,
                    revert_reason=(
                        f"Gas cost ${estimated_gas_usd:.4f} exceeds ROI "
                        f"${expected_roi_usd:.4f} (ratio={ratio:.2f}, "
                        f"minimum={_MIN_ROI_TO_GAS_RATIO})"
                    ),
                    warnings=["Gas cost exceeds expected return"],
                )

        # -- Check 4: Simulation (best-effort) --------------------------------
        if destination and self._wallet is not None:
            sim_result = await self._try_simulate(destination, params)
            if sim_result is not None and not sim_result.passed:
                self._total_rejected += 1
                return sim_result

        # -- Check 5: MEV Predator Detection (Prompt #12) --------------------
        if self._mev_analyzer is not None and destination:
            mev_report = await self._run_mev_analysis(
                action_type=action_type,
                params=params,
                destination=destination,
                chain_id=chain_id,
            )
            self._last_mev_report = mev_report

            if mev_report is not None:
                # Log the MEV analysis result
                self._logger.info(
                    "shield_mev_analysis",
                    action_type=action_type,
                    mev_risk_score=mev_report.mev_risk_score,
                    estimated_extraction=mev_report.estimated_extraction_usd,
                    protection=mev_report.recommended_protection.value,
                    simulated=mev_report.simulated,
                )

                if mev_report.is_high_risk:
                    from systems.axon.mev_types import MEVProtectionStrategy

                    self._total_mev_protected += 1
                    self._total_mev_saved_usd += mev_report.estimated_extraction_usd

                    wait_strategy = MEVProtectionStrategy.WAIT_FOR_LOW_COMPETITION
                    if mev_report.recommended_protection == wait_strategy:
                        # Signal to the pipeline that this tx should be delayed
                        warnings.append(
                            f"MEV risk {mev_report.mev_risk_score:.2f}: "
                            f"recommend waiting for low-competition block "
                            f"(estimated extraction ${mev_report.estimated_extraction_usd:.2f})"
                        )
                    else:
                        # Flashbots Protect or batch auction - add routing metadata
                        warnings.append(
                            f"MEV risk {mev_report.mev_risk_score:.2f}: "
                            f"routed via {mev_report.recommended_protection.value} "
                            f"(estimated MEV saved ${mev_report.estimated_extraction_usd:.2f})"
                        )

                    # Store routing recommendation in the SimulationResult
                    # The executor reads this to choose the submission path
                    return SimulationResult(
                        passed=True,
                        slippage_bps=slippage_bps,
                        mev_risk_detected=True,
                        warnings=warnings,
                    )

                # Low/moderate risk - note in warnings if non-zero
                if mev_report.mev_risk_score > 0.1:
                    warnings.append(
                        f"MEV risk {mev_report.mev_risk_score:.2f} (low): "
                        f"proceeding via public mempool"
                    )

        # -- All checks passed ------------------------------------------------
        if warnings:
            self._logger.info(
                "shield_passed_with_warnings",
                warnings=warnings,
                action_type=action_type,
            )

        return SimulationResult(
            passed=True,
            slippage_bps=slippage_bps,
            warnings=warnings,
        )

    def is_blacklisted(self, address: str, chain_id: int = 8453) -> bool:
        """Check if an address is on the blacklist."""
        key = self._blacklist_key(address, chain_id)
        return key in self._blacklist

    def add_to_blacklist(self, entry: AddressBlacklistEntry) -> None:
        """Add an address to the blacklist."""
        key = self._blacklist_key(entry.address, entry.chain_id)
        self._blacklist[key] = entry
        self._logger.info(
            "shield_address_blacklisted",
            address=entry.address,
            chain_id=entry.chain_id,
            reason=entry.reason,
            threat_type=entry.threat_type,
        )

    # -- Internal ------------------------------------------------------------

    def _extract_destination(
        self,
        action_type: str,
        params: dict[str, Any],
    ) -> str:
        """Extract the destination address from executor params."""
        if action_type == "wallet_transfer":
            return str(params.get("to", params.get("destination", "")))
        if action_type == "defi_yield":
            return str(params.get("protocol_address", params.get("pool_address", "")))
        return ""

    @staticmethod
    def _blacklist_key(address: str, chain_id: int) -> str:
        return f"{address.lower()}:{chain_id}"

    async def _run_mev_analysis(
        self,
        action_type: str,
        params: dict[str, Any],
        destination: str,
        chain_id: int,
    ) -> MEVReport | None:
        """
        Run MEV risk analysis on a financial transaction.

        Delegates to MEVAnalyzer.analyze() which performs heuristic analysis
        and optional EVM fork simulation. Returns None if analysis is
        unavailable or fails.
        """
        if self._mev_analyzer is None:
            return None

        try:
            # Extract calldata and value from params
            calldata = params.get("data", params.get("calldata", "0x"))
            value = int(params.get("value", params.get("value_wei", 0)))
            from_address = params.get("from", params.get("from_address", ""))

            # Estimate transaction volume in USD
            # For DeFi operations, the amount parameter is the USDC amount
            transaction_volume_usd = 0.0
            amount_str = params.get("amount", "")
            if amount_str:
                try:
                    from decimal import Decimal
                    transaction_volume_usd = float(Decimal(str(amount_str)))
                except Exception:
                    pass

            expected_slippage_bps = int(params.get("max_slippage_bps", 50))

            report = await self._mev_analyzer.analyze(
                to=destination,
                data=calldata,
                value=value,
                from_address=from_address,
                chain_id=chain_id,
                transaction_volume_usd=transaction_volume_usd,
                expected_slippage_bps=expected_slippage_bps,
            )

            return report

        except Exception as exc:
            self._logger.debug(
                "shield_mev_analysis_failed",
                error=str(exc),
                action_type=action_type,
            )
            return None

    async def _try_simulate(
        self,
        to: str,
        params: dict[str, Any],
    ) -> SimulationResult | None:
        """
        Attempt to simulate the transaction via eth_call.

        Best-effort: returns None if simulation is not available (e.g., the
        RPC node does not support state overrides). The shield does NOT block
        transactions just because simulation fails -- it only blocks if
        simulation explicitly reveals a problem.
        """
        try:
            # If the wallet client exposes a simulate method, use it
            if hasattr(self._wallet, "simulate_transaction"):
                result = await self._wallet.simulate_transaction(to=to, params=params)
                if result and isinstance(result, dict) and result.get("reverted"):
                    return SimulationResult(
                        passed=False,
                        revert_reason=result.get("revert_reason", "Transaction would revert"),
                        gas_used=result.get("gas_used", 0),
                        warnings=["Simulation detected revert"],
                    )
            return None
        except Exception as exc:
            # Simulation failure is not a rejection -- log and continue
            self._logger.debug(
                "shield_simulation_unavailable",
                error=str(exc),
            )
            return None

    @property
    def last_mev_report(self) -> MEVReport | None:
        """Return the most recent MEV analysis report."""
        return self._last_mev_report

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_evaluated": self._total_evaluated,
            "total_rejected": self._total_rejected,
            "blacklist_size": len(self._blacklist),
            "max_slippage_bps": self._max_slippage_bps,
            "mev_protected": self._total_mev_protected,
            "mev_saved_usd": self._total_mev_saved_usd,
            "mev_analyzer_available": self._mev_analyzer is not None,
        }
