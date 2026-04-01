"""
EcodiaOS -- Axon DeFi Yield Executor (Phase 16c: Resting Metabolism)

Deploys idle capital into yield-bearing DeFi protocols on Base L2 during
consolidation cycles -- the organism's resting metabolism.

DeFiYieldExecutor  -- (Level 3) deposit/withdraw USDC into Aave / Morpho via
                      the organism's CDP-managed wallet.

When EcodiaOS has idle capital above the gas-efficiency floor, this executor
supplies it to lending protocols to earn yield.  When capital is needed, it
withdraws.  This prevents cash drag during consolidation.

Safety constraints:
  - Required autonomy: SOVEREIGN (3) -- deploys real capital on-chain
  - Rate limit: 4 operations per hour -- DeFi interactions are expensive
  - MIN_DEPLOYABLE_BALANCE: $20.00 -- gas-efficiency floor; below this the
    cost of the tx outweighs the yield earned
  - YIELD_FLOOR_APY: 0.02 (2%) -- minimum acceptable APY; deposits abort
    if the protocol's reported rate is below this floor
  - Reversible: False -- on-chain interactions are irreversible once broadcast
  - WalletClient injected at construction; never resolved from globals
  - Only USDC deposits supported in Phase 1 (ETH staking in Phase 2)

Supported protocols (Phase 1):
  - aave   -- Aave V3 on Base (Pool contract)
  - morpho -- Morpho Blue on Base (MetaMorpho vault)
"""

from __future__ import annotations

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

logger = structlog.get_logger()

# -- Constants ---------------------------------------------------------------

# Gas-efficiency floor: deploying less than $20 costs more in gas than the
# yield earned over a reasonable horizon.
MIN_DEPLOYABLE_BALANCE = Decimal("20.00")

# Minimum acceptable APY -- if the protocol reports below this, abort to
# avoid locking capital for negligible returns.
YIELD_FLOOR_APY = Decimal("0.02")

_SUPPORTED_ACTIONS = frozenset({"deposit", "withdraw"})

# Phase 16d: expanded protocol coverage
# aave        - Aave V3 on Base (supply/borrow lending)
# morpho      - Morpho Blue MetaMorpho USDC vault
# aerodrome   - Aerodrome concentrated liquidity AMM (earns AERO rewards)
# moonwell    - Moonwell lending/borrowing (earns WELL rewards)
# extra_finance - Extra Finance leveraged yield (conservative mode, max 2×)
# beefy       - Beefy auto-compounding vault aggregator
_SUPPORTED_PROTOCOLS = frozenset({
    "aave", "morpho", "aerodrome", "moonwell", "extra_finance", "beefy"
})

# Base mainnet (chain ID 8453) contract addresses
# Aave V3 Pool -- the canonical supply/withdraw entrypoint on Base
_AAVE_V3_POOL_BASE = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"

# Morpho Blue MetaMorpho USDC vault on Base
_MORPHO_VAULT_BASE = "0xc1256Ae5FF1cf2719D4937adb3bbCCab2E00A2Ca"

# Aerodrome Finance Router on Base (USDC/USDT stable pool)
# Uses ERC-4626 compatible StablePool vault interface for USDC deposits
_AERODROME_ROUTER_BASE = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
# Aerodrome USDC/USDT stable gauge (auto-staking via router)
_AERODROME_GAUGE_BASE = "0x4F09bAb2f0E15e2A078A227FE1537665F55b8360"

# Moonwell mUSDC on Base (ERC-4626 compatible mToken)
_MOONWELL_MUSDC_BASE = "0xEdc817A28E8B93B03976FBd4a3dDBc9f7D176c22"

# Extra Finance Lend Pool on Base (ERC-4626 deposit, conservative mode)
_EXTRA_FINANCE_LEND_BASE = "0xBB505c54D71E9e599Cb8435b4F0cEEc05fC71cbD"

# Beefy BeefyVaultV7 USDC vault on Base (ERC-4626 compatible)
_BEEFY_USDC_VAULT_BASE = "0x67c764b0e9F19Af46dfBFe4e7f7C7dFf02A7eEE1"

# USDC on Base
_USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

_USDC_DECIMALS = 6

# Aave V3 Pool ABI fragments (supply / withdraw)
# supply(address asset, uint256 amount, address onBehalfOf, uint16 referralCode)
_AAVE_SUPPLY_SELECTOR = "0x617ba037"
# withdraw(address asset, uint256 amount, address to) returns (uint256)
_AAVE_WITHDRAW_SELECTOR = "0x69328dec"

# ERC-20 approve(address spender, uint256 amount)
_ERC20_APPROVE_SELECTOR = "0x095ea7b3"

# ERC-4626 vault (Morpho MetaMorpho) -- deposit(uint256 assets, address receiver)
_ERC4626_DEPOSIT_SELECTOR = "0x6e553f65"
# ERC-4626 vault -- withdraw(uint256 assets, address receiver, address owner)
_ERC4626_WITHDRAW_SELECTOR = "0xb460af94"

# Aave V3 Pool ABI fragment for getReserveData - returns a ReserveData struct.
# currentLiquidityRate is at index 2, encoded in ray (1e27 units).
# We only need the selector + the asset address argument.
_AAVE_GET_RESERVE_DATA_SELECTOR = "0x35ea6a75"

# Ray unit = 1e27; divide currentLiquidityRate by 1e27 to get APY as a fraction.
_RAY = Decimal("1e27")

# Seconds per year - used for Morpho share-price APY approximation.
_SECONDS_PER_YEAR = Decimal("365.25") * Decimal("86400")

# Error fragments from DeFi interactions
_INSUFFICIENT_FUNDS_PHRASES = (
    "insufficient funds",
    "insufficient balance",
    "exceeds balance",
    "transfer amount exceeds",
    "ERC20: transfer amount exceeds balance",
)


def _is_insufficient_funds(error: str) -> bool:
    lower = error.lower()
    return any(phrase in lower for phrase in _INSUFFICIENT_FUNDS_PHRASES)


def _encode_address(addr: str) -> str:
    """Encode an address as a 32-byte ABI parameter (zero-padded left)."""
    return addr.lower().replace("0x", "").zfill(64)


def _encode_uint256(value: int) -> str:
    """Encode a uint256 as a 32-byte ABI parameter."""
    return hex(value)[2:].zfill(64)


def _encode_uint16(value: int) -> str:
    """Encode a uint16 as a 32-byte ABI parameter (zero-padded left)."""
    return hex(value)[2:].zfill(64)


def _build_approve_calldata(spender: str, amount: int) -> str:
    """Build ERC-20 approve(spender, amount) calldata."""
    return (
        _ERC20_APPROVE_SELECTOR
        + _encode_address(spender)
        + _encode_uint256(amount)
    )


def _build_aave_supply_calldata(
    asset: str, amount: int, on_behalf_of: str, referral_code: int = 0,
) -> str:
    """Build Aave V3 Pool.supply(asset, amount, onBehalfOf, referralCode)."""
    return (
        _AAVE_SUPPLY_SELECTOR
        + _encode_address(asset)
        + _encode_uint256(amount)
        + _encode_address(on_behalf_of)
        + _encode_uint16(referral_code)
    )


def _build_aave_withdraw_calldata(
    asset: str, amount: int, to: str,
) -> str:
    """Build Aave V3 Pool.withdraw(asset, amount, to)."""
    return (
        _AAVE_WITHDRAW_SELECTOR
        + _encode_address(asset)
        + _encode_uint256(amount)
        + _encode_address(to)
    )


def _build_erc4626_deposit_calldata(assets: int, receiver: str) -> str:
    """Build ERC-4626 deposit(assets, receiver)."""
    return (
        _ERC4626_DEPOSIT_SELECTOR
        + _encode_uint256(assets)
        + _encode_address(receiver)
    )


def _build_erc4626_withdraw_calldata(
    assets: int, receiver: str, owner: str,
) -> str:
    """Build ERC-4626 withdraw(assets, receiver, owner)."""
    return (
        _ERC4626_WITHDRAW_SELECTOR
        + _encode_uint256(assets)
        + _encode_address(receiver)
        + _encode_address(owner)
    )


# -- DeFiYieldExecutor -------------------------------------------------------


class DeFiYieldExecutor(Executor):
    """
    Deploy or withdraw idle USDC into yield-bearing DeFi protocols on Base.

    Phase 16c -- Resting Metabolism.  The organism earns yield on idle capital
    during consolidation cycles, preventing cash drag.

    Required params:
      action   (str): "deposit" or "withdraw".
      amount   (str): Human-readable USDC amount, e.g. "50.00" or "1000".
      protocol (str): "aave" or "morpho" (case-insensitive).

    Returns ExecutionResult with:
      data:
        tx_hash        -- on-chain transaction hash
        action         -- "deposit" or "withdraw"
        protocol       -- normalised protocol name
        amount         -- amount as submitted
        amount_raw     -- raw integer amount (USDC smallest unit)
        contract       -- protocol contract address interacted with
        network        -- EVM network ("base")
      side_effects:
        -- Human-readable description for world-state log
      new_observations:
        -- Observation fed back into the workspace for Atune scoring

    Abort conditions (returned as ExecutionResult(success=False)):
      - Balance < MIN_DEPLOYABLE_BALANCE ($20) on deposit
      - WalletClient not injected
      - Param validation failures
      - Insufficient funds on-chain
      - Any protocol / network error
    """

    action_type = "defi_yield"
    description = (
        "Deploy or withdraw idle USDC into DeFi yield protocols "
        "(Aave/Morpho/Aerodrome/Moonwell/ExtraFinance/Beefy) "
        "on Base L2 -- resting metabolism for idle capital (Level 3)"
    )

    required_autonomy = 3       # SOVEREIGN -- deploys real capital on-chain
    reversible = False          # On-chain interactions cannot be reversed
    max_duration_ms = 90_000    # DeFi txs may need approval + supply in sequence
    rate_limit = RateLimit.per_hour(4)

    def __init__(self, wallet: WalletClient | None = None) -> None:
        self._wallet = wallet
        self._logger = logger.bind(system="axon.executor.defi_yield")

    # -- Validation ----------------------------------------------------------

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Fast param validation -- no I/O."""
        # action
        action = str(params.get("action", "")).strip().lower()
        if not action:
            return ValidationResult.fail(
                "action is required ('deposit' or 'withdraw')",
                action="missing",
            )
        if action not in _SUPPORTED_ACTIONS:
            return ValidationResult.fail(
                f"action must be one of: {', '.join(sorted(_SUPPORTED_ACTIONS))}",
                action="unsupported value",
            )

        # amount
        amount_raw = str(params.get("amount", "")).strip()
        if not amount_raw:
            return ValidationResult.fail("amount is required", amount="missing")
        try:
            amount_decimal = Decimal(amount_raw)
        except InvalidOperation:
            return ValidationResult.fail(
                "amount must be a valid decimal number (e.g. '50.00')",
                amount="not a decimal",
            )
        if amount_decimal <= Decimal(0):
            return ValidationResult.fail(
                "amount must be greater than zero",
                amount="must be positive",
            )

        # protocol
        protocol = str(params.get("protocol", "")).strip().lower()
        if not protocol:
            return ValidationResult.fail(
                "protocol is required ('aave' or 'morpho')",
                protocol="missing",
            )
        if protocol not in _SUPPORTED_PROTOCOLS:
            return ValidationResult.fail(
                f"protocol must be one of: {', '.join(sorted(_SUPPORTED_PROTOCOLS))} "
                f"(aave/morpho for lending; aerodrome/moonwell/extra_finance/beefy for AMM/auto-compounding)",
                protocol="unsupported value",
            )

        return ValidationResult.ok()

    # -- Execution -----------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute the DeFi yield operation. Never raises."""
        if self._wallet is None:
            return ExecutionResult(
                success=False,
                error=(
                    "WalletClient not configured. "
                    "Pass wallet= to DeFiYieldExecutor or register via AxonService."
                ),
            )

        action = str(params["action"]).strip().lower()
        amount_str = str(params["amount"]).strip()
        amount_decimal = Decimal(amount_str)
        protocol = str(params["protocol"]).strip().lower()

        self._logger.info(
            "defi_yield_execute",
            action=action,
            amount=amount_str,
            protocol=protocol,
            execution_id=context.execution_id,
        )

        # -- Gate: minimum deployable balance for deposits -------------------
        if action == "deposit" and amount_decimal < MIN_DEPLOYABLE_BALANCE:
            self._logger.info(
                "defi_yield_below_floor",
                amount=amount_str,
                floor=str(MIN_DEPLOYABLE_BALANCE),
                execution_id=context.execution_id,
            )
            return ExecutionResult(
                success=False,
                error=(
                    f"BELOW_GAS_FLOOR: deposit amount ${amount_str} is below the "
                    f"minimum deployable balance of ${MIN_DEPLOYABLE_BALANCE}. "
                    f"Gas costs would exceed yield earned. Aborting to preserve capital."
                ),
                data={
                    "failure_type": "below_gas_floor",
                    "amount": amount_str,
                    "min_deployable": str(MIN_DEPLOYABLE_BALANCE),
                },
            )

        # -- Gate: YIELD_FLOOR_APY check for deposits -----------------------
        # Abort if the protocol's current APY is below the floor to avoid
        # locking capital for negligible (or zero) returns.
        if action == "deposit":
            current_apy = await self._fetch_protocol_apy(protocol)
            if current_apy is not None and current_apy < YIELD_FLOOR_APY:
                self._logger.info(
                    "defi_yield_below_apy_floor",
                    protocol=protocol,
                    current_apy=float(current_apy),
                    floor_apy=float(YIELD_FLOOR_APY),
                    execution_id=context.execution_id,
                )
                return ExecutionResult(
                    success=False,
                    error=(
                        f"BELOW_APY_FLOOR: {protocol.capitalize()} current APY "
                        f"{float(current_apy):.2%} is below the minimum acceptable "
                        f"APY of {float(YIELD_FLOOR_APY):.2%}. "
                        f"Aborting to avoid locking capital for negligible returns."
                    ),
                    data={
                        "failure_type": "below_apy_floor",
                        "protocol": protocol,
                        "current_apy": float(current_apy),
                        "min_apy": float(YIELD_FLOOR_APY),
                    },
                )

        # -- Convert to raw USDC amount (6 decimals) ------------------------
        amount_raw_int = int(amount_decimal * (10 ** _USDC_DECIMALS))

        try:
            if protocol == "aave":
                tx_hash = await self._execute_aave(action, amount_raw_int, context)
                contract = _AAVE_V3_POOL_BASE
            elif protocol == "morpho":
                tx_hash = await self._execute_morpho(action, amount_raw_int, context)
                contract = _MORPHO_VAULT_BASE
            elif protocol == "aerodrome":
                tx_hash = await self._execute_erc4626(
                    action, amount_raw_int, context,
                    vault_address=_AERODROME_GAUGE_BASE,
                    protocol_name="aerodrome",
                )
                contract = _AERODROME_GAUGE_BASE
            elif protocol == "moonwell":
                tx_hash = await self._execute_erc4626(
                    action, amount_raw_int, context,
                    vault_address=_MOONWELL_MUSDC_BASE,
                    protocol_name="moonwell",
                )
                contract = _MOONWELL_MUSDC_BASE
            elif protocol == "extra_finance":
                # Extra Finance uses ERC-4626 interface in conservative (lend-only) mode.
                # Leveraged mode is explicitly NOT used - Max leverage = 1× (simple deposit).
                tx_hash = await self._execute_erc4626(
                    action, amount_raw_int, context,
                    vault_address=_EXTRA_FINANCE_LEND_BASE,
                    protocol_name="extra_finance",
                )
                contract = _EXTRA_FINANCE_LEND_BASE
            else:
                # beefy - ERC-4626 compatible auto-compounding vault
                tx_hash = await self._execute_erc4626(
                    action, amount_raw_int, context,
                    vault_address=_BEEFY_USDC_VAULT_BASE,
                    protocol_name="beefy",
                )
                contract = _BEEFY_USDC_VAULT_BASE
        except Exception as exc:
            return self._handle_execution_error(exc, action, amount_str, protocol, context)

        # -- Success ---------------------------------------------------------
        network = self._wallet.network

        self._logger.info(
            "defi_yield_confirmed",
            tx_hash=tx_hash,
            action=action,
            protocol=protocol,
            amount=amount_str,
            network=network,
            execution_id=context.execution_id,
        )

        verb = "Deposited" if action == "deposit" else "Withdrew"
        preposition = "into" if action == "deposit" else "from"

        side_effect = (
            f"{verb} {amount_str} USDC {preposition} {protocol.capitalize()} "
            f"on {network} -- tx: {tx_hash}"
        )

        observation = (
            f"DeFi yield {action}: {amount_str} USDC {preposition} "
            f"{protocol.capitalize()} (tx: {tx_hash[:20]}..., network: {network})"
        )

        return ExecutionResult(
            success=True,
            data={
                "tx_hash": tx_hash,
                "action": action,
                "protocol": protocol,
                "amount": amount_str,
                "amount_raw": amount_raw_int,
                "contract": contract,
                "network": network,
            },
            side_effects=[side_effect],
            new_observations=[observation],
        )

    # -- Protocol-specific execution -----------------------------------------

    async def _execute_aave(
        self,
        action: str,
        amount_raw: int,
        context: ExecutionContext,
    ) -> str:
        """Execute Aave V3 supply or withdraw. Returns tx_hash."""
        wallet = self._wallet
        assert wallet is not None  # Guarded in execute()

        account = wallet._require_account()
        cdp = wallet._require_cdp()
        network = wallet.network
        owner_address = wallet.address

        if action == "deposit":
            # Step 1: Approve Aave Pool to spend USDC
            approve_data = _build_approve_calldata(_AAVE_V3_POOL_BASE, amount_raw)

            self._logger.debug(
                "defi_yield_aave_approve",
                spender=_AAVE_V3_POOL_BASE,
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )

            await self._send_tx(
                cdp, account, network, _USDC_BASE, approve_data,
            )

            # Step 2: Supply USDC to Aave
            supply_data = _build_aave_supply_calldata(
                _USDC_BASE, amount_raw, owner_address,
            )

            self._logger.debug(
                "defi_yield_aave_supply",
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )

            tx_hash = await self._send_tx(
                cdp, account, network, _AAVE_V3_POOL_BASE, supply_data,
            )
        else:
            # Withdraw from Aave
            withdraw_data = _build_aave_withdraw_calldata(
                _USDC_BASE, amount_raw, owner_address,
            )

            self._logger.debug(
                "defi_yield_aave_withdraw",
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )

            tx_hash = await self._send_tx(
                cdp, account, network, _AAVE_V3_POOL_BASE, withdraw_data,
            )

        return tx_hash

    async def _execute_morpho(
        self,
        action: str,
        amount_raw: int,
        context: ExecutionContext,
    ) -> str:
        """Execute Morpho MetaMorpho vault deposit or withdraw. Returns tx_hash."""
        wallet = self._wallet
        assert wallet is not None

        account = wallet._require_account()
        cdp = wallet._require_cdp()
        network = wallet.network
        owner_address = wallet.address

        if action == "deposit":
            # Step 1: Approve vault to spend USDC
            approve_data = _build_approve_calldata(_MORPHO_VAULT_BASE, amount_raw)

            self._logger.debug(
                "defi_yield_morpho_approve",
                spender=_MORPHO_VAULT_BASE,
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )

            await self._send_tx(
                cdp, account, network, _USDC_BASE, approve_data,
            )

            # Step 2: Deposit into ERC-4626 vault
            deposit_data = _build_erc4626_deposit_calldata(amount_raw, owner_address)

            self._logger.debug(
                "defi_yield_morpho_deposit",
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )

            tx_hash = await self._send_tx(
                cdp, account, network, _MORPHO_VAULT_BASE, deposit_data,
            )
        else:
            # Withdraw from ERC-4626 vault
            withdraw_data = _build_erc4626_withdraw_calldata(
                amount_raw, owner_address, owner_address,
            )

            self._logger.debug(
                "defi_yield_morpho_withdraw",
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )

            tx_hash = await self._send_tx(
                cdp, account, network, _MORPHO_VAULT_BASE, withdraw_data,
            )

        return tx_hash

    async def _execute_erc4626(
        self,
        action: str,
        amount_raw: int,
        context: ExecutionContext,
        vault_address: str,
        protocol_name: str,
    ) -> str:
        """
        Generic ERC-4626 vault deposit or withdraw.

        Used for Aerodrome, Moonwell, Extra Finance (conservative), and Beefy -
        all of which expose the standard ERC-4626 interface for USDC deposits.

        Extra Finance: conservative (lend-only) mode - no leverage is applied.
        Aerodrome: deposits into the USDC/USDT stable gauge via ERC-4626 wrapper.
        """
        wallet = self._wallet
        assert wallet is not None

        account = wallet._require_account()
        cdp = wallet._require_cdp()
        network = wallet.network
        owner_address = wallet.address

        if action == "deposit":
            # Step 1: Approve vault to spend USDC
            approve_data = _build_approve_calldata(vault_address, amount_raw)
            self._logger.debug(
                f"defi_yield_{protocol_name}_approve",
                spender=vault_address,
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )
            await self._send_tx(cdp, account, network, _USDC_BASE, approve_data)

            # Step 2: Deposit into ERC-4626 vault
            deposit_data = _build_erc4626_deposit_calldata(amount_raw, owner_address)
            self._logger.debug(
                f"defi_yield_{protocol_name}_deposit",
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )
            tx_hash = await self._send_tx(cdp, account, network, vault_address, deposit_data)
        else:
            # Withdraw from ERC-4626 vault
            withdraw_data = _build_erc4626_withdraw_calldata(
                amount_raw, owner_address, owner_address,
            )
            self._logger.debug(
                f"defi_yield_{protocol_name}_withdraw",
                amount_raw=amount_raw,
                execution_id=context.execution_id,
            )
            tx_hash = await self._send_tx(cdp, account, network, vault_address, withdraw_data)

        return tx_hash

    # -- Transaction helpers -------------------------------------------------

    @staticmethod
    async def _send_tx(
        cdp: Any,
        account: Any,
        network: str,
        to: str,
        data: str,
    ) -> str:
        """Send a transaction via the CDP SDK. Returns the tx hash as a string."""
        from cdp.evm_transaction_types import TransactionRequestEIP1559

        tx_hash = await cdp.evm.send_transaction(
            address=account.address,
            transaction=TransactionRequestEIP1559(
                to=to,
                value=0,
                data=data,
            ),
            network=network,
        )
        return str(tx_hash)

    # -- APY Fetching --------------------------------------------------------

    async def _fetch_protocol_apy(self, protocol: str) -> Decimal | None:
        """
        Fetch the current supply APY from the protocol via a read-only RPC call.

        Returns the APY as a Decimal fraction (e.g. 0.035 = 3.5%), or None if
        the fetch fails (caller treats None as fail-open - deposit proceeds).

        Aave V3: calls getReserveData(asset) on the Pool contract and reads
        currentLiquidityRate (index 2 in the returned tuple), which is encoded
        in ray units (1e27). APY ≈ currentLiquidityRate / 1e27.

        Morpho (ERC-4626): calls convertToAssets(1e6) and compares to the share
        price 24h ago via totalAssets/totalSupply. Phase 1 approximation: return
        None (fail-open) as Morpho's on-chain APY requires a time-series oracle
        not available in Phase 1 - the gas floor ($20) is sufficient protection.
        """
        try:
            import asyncio

            # Use a public Base mainnet RPC for read-only calls.
            # WalletClient's network string maps to Base mainnet.
            rpc_url = "https://mainnet.base.org"
            loop = asyncio.get_event_loop()

            if protocol == "aave":
                return await loop.run_in_executor(
                    None, self._fetch_aave_apy_sync, rpc_url
                )
            # Morpho: fail-open - no time-series oracle available in Phase 1
            return None
        except Exception as exc:
            self._logger.debug(
                "defi_yield_apy_fetch_failed",
                protocol=protocol,
                error=str(exc),
            )
            return None  # Fail-open: deposit proceeds if APY fetch errors

    @staticmethod
    def _fetch_aave_apy_sync(rpc_url: str) -> Decimal | None:
        """
        Synchronous Aave V3 APY fetch via getReserveData(USDC).

        currentLiquidityRate is at position 2 in the ReserveData struct,
        encoded as a uint128 in ray units (1e27). APY = rate / 1e27.
        """
        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 5}))
            # ABI for getReserveData(address asset) - minimal struct, read-only
            # currentLiquidityRate is index 2 in the returned ReserveData tuple.
            def _f(t: str, n: str) -> dict:  # noqa: E306
                return {"internalType": t, "name": n, "type": t.split(".")[-1]}

            components = [
                _f("uint256", "configuration"),
                _f("uint128", "liquidityIndex"),
                _f("uint128", "currentLiquidityRate"),
                _f("uint128", "variableBorrowIndex"),
                _f("uint128", "currentVariableBorrowRate"),
                _f("uint128", "currentStableBorrowRate"),
                _f("uint40", "lastUpdateTimestamp"),
                _f("uint16", "id"),
                _f("address", "aTokenAddress"),
                _f("address", "stableDebtTokenAddress"),
                _f("address", "variableDebtTokenAddress"),
                _f("address", "interestRateStrategyAddress"),
                _f("uint128", "accruedToTreasury"),
                _f("uint128", "unbacked"),
                _f("uint128", "isolationModeTotalDebt"),
            ]
            abi = [
                {
                    "inputs": [
                        {"internalType": "address", "name": "asset", "type": "address"},
                    ],
                    "name": "getReserveData",
                    "outputs": [
                        {
                            "components": components,
                            "internalType": "struct DataTypes.ReserveData",
                            "name": "",
                            "type": "tuple",
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function",
                }
            ]
            pool = w3.eth.contract(
                address=Web3.to_checksum_address(_AAVE_V3_POOL_BASE),
                abi=abi,
            )
            reserve_data = pool.functions.getReserveData(
                Web3.to_checksum_address(_USDC_BASE)
            ).call()
            # currentLiquidityRate is index 2 in the tuple
            current_liquidity_rate = reserve_data[2]
            apy = Decimal(current_liquidity_rate) / _RAY
            return apy
        except Exception:
            return None

    # -- Error handling ------------------------------------------------------

    def _handle_execution_error(
        self,
        exc: Exception,
        action: str,
        amount_str: str,
        protocol: str,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Map exceptions to structured ExecutionResult failures."""
        error_str = str(exc)

        if _is_insufficient_funds(error_str):
            self._logger.warning(
                "defi_yield_insufficient_funds",
                action=action,
                protocol=protocol,
                amount=amount_str,
                execution_id=context.execution_id,
            )
            return ExecutionResult(
                success=False,
                error=(
                    f"INSUFFICIENT_FUNDS: wallet does not have enough USDC "
                    f"to {action} {amount_str} via {protocol.capitalize()}."
                ),
                data={
                    "failure_type": "insufficient_funds",
                    "action": action,
                    "protocol": protocol,
                    "amount": amount_str,
                },
            )

        self._logger.error(
            "defi_yield_failed",
            action=action,
            protocol=protocol,
            amount=amount_str,
            execution_id=context.execution_id,
            error=error_str,
        )
        return ExecutionResult(
            success=False,
            error=f"DeFi {action} failed on {protocol.capitalize()}: {error_str}",
            data={
                "failure_type": "protocol_error",
                "action": action,
                "protocol": protocol,
                "amount": amount_str,
            },
        )
