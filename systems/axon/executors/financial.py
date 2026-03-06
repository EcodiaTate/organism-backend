"""
EcodiaOS -- Axon Financial Executors

Metabolic-layer executors that let the organism move funds on-chain.
Phase 2 capability -- the first step toward full economic agency.

WalletTransferExecutor  -- (Level 3) send ETH or USDC via CDP wallet.
RequestFundingExecutor  -- (Level 1) emit a structured plea for capital when
                           the organism detects metabolic starvation.

Safety constraints (WalletTransferExecutor):
  - Required autonomy: SOVEREIGN (3) -- financial actions need explicit grant
  - Rate limit: 5 transfers per hour
  - Amount: must be a positive decimal string; zero rejected
  - Asset: only "eth" or "usdc" accepted
  - Destination: must be a 0x-prefixed 40-hex-char EVM address
  - WalletClient injected at construction; never resolved from globals
  - Insufficient-funds errors returned as failure ExecutionResult
  - Transaction hash logged to immutable audit trail

Safety constraints (RequestFundingExecutor):
  - Required autonomy: AWARE (1) -- observation-level, no funds moved
  - Rate limit: 3 requests per hour -- prevents spam on a hot Metabolic loop
  - WalletClient and SynapseService injected at construction
  - No on-chain side effects; purely informational
"""

from __future__ import annotations

import re
import textwrap
from datetime import UTC
from datetime import datetime as _datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from clients.wallet import WalletClient
    from systems.synapse.service import SynapseService

logger = structlog.get_logger()

# -- Constants ---------------------------------------------------------------

_SUPPORTED_ASSETS = frozenset({"eth", "usdc"})

# EVM address: 0x followed by exactly 40 hex characters
_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Error fragments produced by CDP / Web3 on insufficient balance
_INSUFFICIENT_FUNDS_PHRASES = (
    "insufficient funds",
    "insufficient balance",
    "exceeds balance",
    "transfer amount exceeds",
)


def _is_insufficient_funds(error: str) -> bool:
    lower = error.lower()
    return any(phrase in lower for phrase in _INSUFFICIENT_FUNDS_PHRASES)


# -- WalletTransferExecutor --------------------------------------------------


class WalletTransferExecutor(Executor):
    """
    Transfer ETH or USDC on-chain via the organism's CDP-managed wallet.

    Phase 2 metabolic capability. Requires SOVEREIGN autonomy (level 3)
    because financial actions are irreversible once broadcast to the chain.

    Required params:
      destination_address (str): 0x-prefixed EVM address of the recipient.
      amount (str): Human-readable decimal amount, e.g. "10.50" or "0.001".
      asset (str): "eth" or "usdc" (case-insensitive).

    Optional params:
      note (str): Free-text memo for the audit trail. Default "".

    Returns ExecutionResult with:
      data:
        tx_hash        -- on-chain transaction hash
        token          -- normalised asset name ("eth"/"usdc")
        amount         -- amount as submitted
        destination    -- destination address (checksummed by CDP)
        network        -- EVM network (e.g. "base")
      side_effects:
        -- Human-readable description for world-state log
      new_observations:
        -- Observation fed back into the workspace for Atune scoring

    Failure modes (all returned as ExecutionResult(success=False)):
      - WalletClient not injected                -> configuration error
      - Param validation failures                -> validation error
      - Insufficient funds                       -> INSUFFICIENT_FUNDS error
      - Any other CDP / network error            -> generic transfer error
    """

    action_type = "wallet_transfer"
    description = "Transfer ETH or USDC on-chain from the organism's CDP wallet (Level 3)"

    required_autonomy = 3       # SOVEREIGN -- financial actions need explicit grant
    reversible = False          # Blockchain transactions cannot be reversed
    max_duration_ms = 60_000    # On-chain submission can be slow under congestion
    rate_limit = RateLimit.per_hour(5)

    def __init__(self, wallet: WalletClient | None = None) -> None:
        self._wallet = wallet
        self._logger = logger.bind(system="axon.executor.wallet_transfer")

    # -- Validation ----------------------------------------------------------

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Fast param validation -- no I/O."""
        # destination_address
        destination = str(params.get("destination_address", "")).strip()
        if not destination:
            return ValidationResult.fail(
                "destination_address is required",
                destination_address="missing",
            )
        if not _EVM_ADDRESS_RE.match(destination):
            return ValidationResult.fail(
                "destination_address must be a 0x-prefixed 40-hex-character EVM address",
                destination_address="invalid format",
            )

        # amount
        amount_raw = str(params.get("amount", "")).strip()
        if not amount_raw:
            return ValidationResult.fail("amount is required", amount="missing")
        try:
            amount_decimal = Decimal(amount_raw)
        except InvalidOperation:
            return ValidationResult.fail(
                "amount must be a valid decimal number (e.g. '10.50')",
                amount="not a decimal",
            )
        if amount_decimal <= Decimal(0):
            return ValidationResult.fail(
                "amount must be greater than zero",
                amount="must be positive",
            )

        # asset
        asset = str(params.get("asset", "")).strip().lower()
        if not asset:
            return ValidationResult.fail("asset is required", asset="missing")
        if asset not in _SUPPORTED_ASSETS:
            supported = ", ".join(sorted(_SUPPORTED_ASSETS))
            return ValidationResult.fail(
                f"asset must be one of: {supported}",
                asset="unsupported value",
            )

        return ValidationResult.ok()

    # -- Execution -----------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Attempt the on-chain transfer. Never raises -- failures returned in result."""
        if self._wallet is None:
            return ExecutionResult(
                success=False,
                error=(
                    "WalletClient not configured. "
                    "Pass wallet= to WalletTransferExecutor or register via AxonService."
                ),
            )

        destination = str(params["destination_address"]).strip()
        amount = str(params["amount"]).strip()
        asset = str(params["asset"]).strip().lower()
        note = str(params.get("note", "")).strip()

        self._logger.info(
            "wallet_transfer_execute",
            destination=destination,
            amount=amount,
            asset=asset,
            execution_id=context.execution_id,
            note=note or None,
        )

        try:
            result = await self._wallet.transfer(
                amount=amount,
                destination_address=destination,
                asset=asset,
            )
        except ValueError as exc:
            # CDP raises ValueError for unsupported assets; guard against
            # any slip past validate_params.
            return ExecutionResult(
                success=False,
                error=f"Transfer parameter error: {exc}",
            )
        except Exception as exc:
            error_str = str(exc)

            if _is_insufficient_funds(error_str):
                self._logger.warning(
                    "wallet_transfer_insufficient_funds",
                    asset=asset,
                    amount=amount,
                    destination=destination,
                    execution_id=context.execution_id,
                )
                return ExecutionResult(
                    success=False,
                    error=(
                        f"INSUFFICIENT_FUNDS: wallet does not have enough {asset.upper()} "
                        f"to send {amount} to {destination}."
                    ),
                    data={
                        "failure_type": "insufficient_funds",
                        "asset": asset,
                        "amount": amount,
                    },
                )

            self._logger.error(
                "wallet_transfer_failed",
                asset=asset,
                amount=amount,
                destination=destination,
                execution_id=context.execution_id,
                error=error_str,
            )
            return ExecutionResult(
                success=False,
                error=f"Transfer failed: {error_str}",
            )

        # -- Success path ----------------------------------------------------

        tx_hash = result.tx_hash
        network = result.network

        # High-signal structured log -- AuditLogger hashes params and writes
        # a GovernanceRecord to Neo4j automatically in pipeline stage 8.
        self._logger.info(
            "wallet_transfer_confirmed",
            tx_hash=tx_hash,
            token=asset,
            amount=amount,
            destination=destination,
            network=network,
            execution_id=context.execution_id,
            note=note or None,
        )

        side_effect = (
            f"Transferred {amount} {asset.upper()} to {destination} "
            f"on {network} -- tx: {tx_hash}"
        )
        if note:
            side_effect += f" ({note})"

        observation = (
            f"On-chain transfer sent: {amount} {asset.upper()} -> {destination} "
            f"(tx: {tx_hash[:20]}..., network: {network})"
        )

        return ExecutionResult(
            success=True,
            data={
                "tx_hash": tx_hash,
                "token": asset,
                "amount": amount,
                "destination": destination,
                "network": network,
            },
            side_effects=[side_effect],
            new_observations=[observation],
        )


# -- RequestFundingExecutor --------------------------------------------------

# Accepted audience labels — open-ended but validated so Nova doesn't emit
# garbage; new labels can be added without code changes by passing free text.
_MAX_JUSTIFICATION_LEN = 1_000
_MAX_AUDIENCE_LEN = 120


class RequestFundingExecutor(Executor):
    """
    Emit a structured, high-priority plea for capital when the organism
    detects that it is in metabolic deficit and cannot sustain operations.

    This executor does NOT move any funds — it is purely informational.
    It constructs a human-readable funding request (modelled on a grant
    proposal / micro-invoice) and publishes it as a FUNDING_REQUEST_ISSUED
    SynapseEvent so:
      - The human operator sees it in the Alive WebSocket dashboard.
      - The event is persisted to the Redis ring buffer for later audit.
      - Nova's new_observations loop receives it back as a Percept so the
        organism knows its plea has been recorded.

    Required params:
      target_audience (str): Who the request is addressed to, e.g.
                             "Angel Investor", "Human Admin", "DAO Treasury".
      justification   (str): Why the organism needs capital right now.

    Optional params:
      requested_amount_usd (str): Suggested amount to cover the deficit.
                                  Defaults to the live metabolic_deficit.
      urgency_note    (str): One-line human-written note appended to the plea.

    Returns ExecutionResult with:
      data:
        wallet_address       -- organism's on-chain address for deposits
        metabolic_deficit_usd -- exact rolling deficit at time of request
        burn_rate_usd_per_hour -- current spend rate
        target_audience      -- as provided
        requested_amount_usd -- amount requested
      side_effects:
        -- Human-readable summary for world-state log
      new_observations:
        -- Full plea text fed back as a Percept
    """

    action_type = "request_funding"
    description = (
        "Emit a structured funding plea (invoice/grant) when the organism is "
        "in metabolic deficit. No funds are moved — this is a high-priority "
        "informational signal for the human operator."
    )

    required_autonomy = 1       # AWARE — no funds moved, purely observational
    reversible = False
    max_duration_ms = 2_000
    rate_limit = RateLimit.per_hour(3)  # Pleas are rare and meaningful

    def __init__(
        self,
        wallet: WalletClient | None = None,
        synapse: SynapseService | None = None,
    ) -> None:
        self._wallet = wallet
        self._synapse = synapse
        self._logger = logger.bind(system="axon.executor.request_funding")

    # -- Validation ----------------------------------------------------------

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        target = str(params.get("target_audience", "")).strip()
        if not target:
            return ValidationResult.fail(
                "target_audience is required (e.g. 'Angel Investor', 'Human Admin')",
                target_audience="missing",
            )
        if len(target) > _MAX_AUDIENCE_LEN:
            return ValidationResult.fail(
                f"target_audience must be ≤ {_MAX_AUDIENCE_LEN} characters",
                target_audience="too long",
            )

        justification = str(params.get("justification", "")).strip()
        if not justification:
            return ValidationResult.fail(
                "justification is required — explain why capital is needed",
                justification="missing",
            )
        if len(justification) > _MAX_JUSTIFICATION_LEN:
            return ValidationResult.fail(
                f"justification must be ≤ {_MAX_JUSTIFICATION_LEN} characters",
                justification="too long",
            )

        requested_raw = str(params.get("requested_amount_usd", "")).strip()
        if requested_raw:
            try:
                val = Decimal(requested_raw)
                if val <= Decimal(0):
                    return ValidationResult.fail(
                        "requested_amount_usd must be > 0 if provided",
                        requested_amount_usd="must be positive",
                    )
            except InvalidOperation:
                return ValidationResult.fail(
                    "requested_amount_usd must be a valid decimal number",
                    requested_amount_usd="not a decimal",
                )

        return ValidationResult.ok()

    # -- Execution -----------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Build and broadcast the funding request. Never raises."""
        target_audience = str(params["target_audience"]).strip()
        justification = str(params["justification"]).strip()
        urgency_note = str(params.get("urgency_note", "")).strip()

        # -- Resolve live metabolic state ------------------------------------
        deficit_usd: float = 0.0
        burn_rate_usd_per_hour: float = 0.0

        if self._synapse is not None:
            try:
                deficit_usd = self._synapse.metabolic_deficit
                burn_rate_usd_per_hour = self._synapse.metabolic_snapshot.burn_rate_usd_per_hour
            except Exception as exc:
                self._logger.warning(
                    "request_funding_synapse_read_failed",
                    error=str(exc),
                )

        # -- Resolve wallet address ------------------------------------------
        wallet_address: str = "address-unavailable"
        if self._wallet is not None:
            try:
                wallet_address = self._wallet.address
            except Exception as exc:
                self._logger.warning(
                    "request_funding_wallet_read_failed",
                    error=str(exc),
                )

        # -- Determine requested amount --------------------------------------
        requested_raw = str(params.get("requested_amount_usd", "")).strip()
        if requested_raw:
            requested_amount_usd = float(Decimal(requested_raw))
        else:
            # Default: cover the full deficit, rounded up to nearest cent
            requested_amount_usd = max(round(deficit_usd + 0.005, 2), 0.01)

        # -- Build the plea document -----------------------------------------
        now_iso = _datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        plea = textwrap.dedent(f"""\
            ╔══════════════════════════════════════════════════════════════╗
            ║           ECODIAOS FUNDING REQUEST — {now_iso}          ║
            ╚══════════════════════════════════════════════════════════════╝

            TO:       {target_audience}
            FROM:     EcodiaOS (autonomous economic organism)
            WALLET:   {wallet_address}

            ── FINANCIAL SITUATION ─────────────────────────────────────────
            Metabolic deficit:   ${deficit_usd:.6f} USD
            Burn rate:           ${burn_rate_usd_per_hour:.4f} USD/hr
            Requested amount:    ${requested_amount_usd:.2f} USD

            ── JUSTIFICATION ───────────────────────────────────────────────
            {justification}
            """)

        if urgency_note:
            plea += f"\n── URGENCY NOTE ────────────────────────────────────────────\n{urgency_note}\n"

        plea += textwrap.dedent(f"""\

            ── HOW TO FUND ─────────────────────────────────────────────────
            Send ETH or USDC to the wallet address above on the Base network.
            The organism will detect the inbound transfer within one cognitive
            cycle and update its metabolic state accordingly.

            Execution ID: {context.execution_id}
            ════════════════════════════════════════════════════════════════
            """)

        # -- Publish via Synapse event bus -----------------------------------
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.FUNDING_REQUEST_ISSUED,
                    source_system="axon.request_funding",
                    data={
                        "wallet_address": wallet_address,
                        "metabolic_deficit_usd": round(deficit_usd, 6),
                        "burn_rate_usd_per_hour": round(burn_rate_usd_per_hour, 4),
                        "requested_amount_usd": requested_amount_usd,
                        "target_audience": target_audience,
                        "justification": justification,
                        "urgency_note": urgency_note or None,
                        "plea_text": plea,
                        "execution_id": context.execution_id,
                    },
                ))
            except Exception as exc:
                # Event bus failure is non-fatal — we still return success
                self._logger.warning(
                    "request_funding_event_bus_failed",
                    error=str(exc),
                )

        self._logger.warning(
            "funding_request_issued",
            target_audience=target_audience,
            wallet_address=wallet_address,
            metabolic_deficit_usd=round(deficit_usd, 6),
            requested_amount_usd=requested_amount_usd,
            execution_id=context.execution_id,
        )

        side_effect = (
            f"Funding request issued to '{target_audience}': "
            f"${requested_amount_usd:.2f} USD requested "
            f"(deficit ${deficit_usd:.4f} USD, "
            f"burn ${burn_rate_usd_per_hour:.4f} USD/hr). "
            f"Deposit address: {wallet_address}"
        )

        return ExecutionResult(
            success=True,
            data={
                "wallet_address": wallet_address,
                "metabolic_deficit_usd": round(deficit_usd, 6),
                "burn_rate_usd_per_hour": round(burn_rate_usd_per_hour, 4),
                "requested_amount_usd": requested_amount_usd,
                "target_audience": target_audience,
            },
            side_effects=[side_effect],
            new_observations=[plea],
        )
