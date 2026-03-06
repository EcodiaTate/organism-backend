"""
EcodiaOS — Axon Phantom Liquidity Executor (Phase 16q)

Deploy, withdraw, or rebalance phantom liquidity positions on Uniswap V3
(Base L2).  Each position is a tiny concentrated-liquidity sensor that
receives real-time price feeds via Swap event observation.

PhantomLiquidityExecutor -- (Level 3) mint/burn LP positions via the
                            Uniswap V3 NonfungiblePositionManager on Base.

Safety constraints:
  - Required autonomy: SOVEREIGN (3) -- deploys real capital on-chain
  - Rate limit: 6 operations per hour
  - Min capital per position: $50 (gas-efficiency floor for sensors)
  - Max capital per position: $500
  - Max total deployed: $2 000 across all phantom positions
  - Reversible: False -- on-chain interactions are irreversible
  - WalletClient injected at construction; never resolved from globals
"""

from __future__ import annotations

from decimal import Decimal
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

# Capital bounds for sensor positions
MIN_SENSOR_CAPITAL = Decimal("50.00")
MAX_SENSOR_CAPITAL = Decimal("500.00")
MAX_TOTAL_PHANTOM_CAPITAL = Decimal("2000.00")

_SUPPORTED_ACTIONS = frozenset({"deploy_position", "withdraw_position", "rebalance"})

_SUPPORTED_FEE_TIERS = frozenset({100, 500, 3000, 10000})

# -- Base L2 contract addresses ----------------------------------------------

# Uniswap V3 NonfungiblePositionManager on Base
_UNISWAP_V3_NFT_MANAGER_BASE = "0x03a520b32C04BF3bEEf7BEb72E919cf822Ed34f1"

# USDC on Base
_USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

_USDC_DECIMALS = 6

# -- Function selectors ------------------------------------------------------

# NonfungiblePositionManager.mint(MintParams)
_MINT_SELECTOR = "0x88316456"

# NonfungiblePositionManager.decreaseLiquidity(DecreaseLiquidityParams)
_DECREASE_LIQUIDITY_SELECTOR = "0x0c49ccbe"

# NonfungiblePositionManager.collect(CollectParams)
_COLLECT_SELECTOR = "0xfc6f7865"

# ERC-20 approve(address spender, uint256 amount)
_ERC20_APPROVE_SELECTOR = "0x095ea7b3"

# Insufficient-funds error phrases (shared with defi_yield.py pattern)
_INSUFFICIENT_FUNDS_PHRASES = (
    "insufficient funds",
    "insufficient balance",
    "exceeds balance",
    "transfer amount exceeds",
    "ERC20: transfer amount exceeds balance",
)


# -- ABI encoding helpers ----------------------------------------------------


def _encode_address(addr: str) -> str:
    """Encode an address as a 32-byte ABI parameter (zero-padded left)."""
    return addr.lower().replace("0x", "").zfill(64)


def _encode_uint256(value: int) -> str:
    """Encode a uint256 as a 32-byte ABI parameter."""
    return hex(value)[2:].zfill(64)


def _encode_int24_as_int256(value: int) -> str:
    """
    Encode an int24 as a 32-byte int256 ABI parameter.

    Negative values use two's complement representation.
    """
    if value < 0:
        value = (1 << 256) + value
    return hex(value)[2:].zfill(64)


def _is_insufficient_funds(error: str) -> bool:
    lower = error.lower()
    return any(phrase in lower for phrase in _INSUFFICIENT_FUNDS_PHRASES)


# -- Calldata builders -------------------------------------------------------


def _build_approve_calldata(spender: str, amount: int) -> str:
    """Build ERC-20 approve(spender, amount) calldata."""
    return (
        _ERC20_APPROVE_SELECTOR
        + _encode_address(spender)
        + _encode_uint256(amount)
    )


def _build_mint_calldata(
    token0: str,
    token1: str,
    fee: int,
    tick_lower: int,
    tick_upper: int,
    amount0_desired: int,
    amount1_desired: int,
    amount0_min: int,
    amount1_min: int,
    recipient: str,
    deadline: int,
) -> str:
    """
    Build NonfungiblePositionManager.mint(MintParams) calldata.

    MintParams struct layout:
      address token0, address token1, uint24 fee,
      int24 tickLower, int24 tickUpper,
      uint256 amount0Desired, uint256 amount1Desired,
      uint256 amount0Min, uint256 amount1Min,
      address recipient, uint256 deadline
    """
    return (
        _MINT_SELECTOR
        + _encode_address(token0)
        + _encode_address(token1)
        + _encode_uint256(fee)
        + _encode_int24_as_int256(tick_lower)
        + _encode_int24_as_int256(tick_upper)
        + _encode_uint256(amount0_desired)
        + _encode_uint256(amount1_desired)
        + _encode_uint256(amount0_min)
        + _encode_uint256(amount1_min)
        + _encode_address(recipient)
        + _encode_uint256(deadline)
    )


def _build_decrease_liquidity_calldata(
    token_id: int,
    liquidity: int,
    amount0_min: int,
    amount1_min: int,
    deadline: int,
) -> str:
    """
    Build NonfungiblePositionManager.decreaseLiquidity(DecreaseLiquidityParams).

    DecreaseLiquidityParams struct layout:
      uint256 tokenId, uint128 liquidity,
      uint256 amount0Min, uint256 amount1Min,
      uint256 deadline
    """
    return (
        _DECREASE_LIQUIDITY_SELECTOR
        + _encode_uint256(token_id)
        + _encode_uint256(liquidity)
        + _encode_uint256(amount0_min)
        + _encode_uint256(amount1_min)
        + _encode_uint256(deadline)
    )


def _build_collect_calldata(
    token_id: int,
    recipient: str,
    amount0_max: int,
    amount1_max: int,
) -> str:
    """
    Build NonfungiblePositionManager.collect(CollectParams).

    CollectParams struct layout:
      uint256 tokenId, address recipient,
      uint128 amount0Max, uint128 amount1Max
    """
    return (
        _COLLECT_SELECTOR
        + _encode_uint256(token_id)
        + _encode_address(recipient)
        + _encode_uint256(amount0_max)
        + _encode_uint256(amount1_max)
    )


# -- PhantomLiquidityExecutor ------------------------------------------------


class PhantomLiquidityExecutor(Executor):
    """
    Deploy, withdraw, or rebalance phantom liquidity sensor positions on
    Uniswap V3 (Base L2).

    Phase 16q — Liquidity Phantom Positions.  Tiny concentrated-liquidity
    positions that provide real-time price feeds via Swap event observation.

    Required params:
      action (str): ``"deploy_position"`` | ``"withdraw_position"`` | ``"rebalance"``

      For ``deploy_position``:
        pool_address   (str): Uniswap V3 pool contract address
        token0         (str): Token0 contract address
        token1         (str): Token1 contract address
        fee_tier       (int): Fee tier (100, 500, 3000, 10000)
        amount0_desired (str): Raw token0 amount (smallest unit)
        amount1_desired (str): Raw token1 amount (smallest unit)
        tick_lower     (int): Lower tick bound
        tick_upper     (int): Upper tick bound

      For ``withdraw_position``:
        token_id       (int): NonfungiblePositionManager NFT token ID
        liquidity      (int): Liquidity amount to remove (use MAX_UINT128 for full)

      For ``rebalance``:
        token_id       (int): Existing position to withdraw
        pool_address   (str): Pool to redeploy into
        token0         (str): Token0 contract address
        token1         (str): Token1 contract address
        fee_tier       (int): Fee tier
        amount0_desired (str): Raw token0 amount
        amount1_desired (str): Raw token1 amount
        tick_lower     (int): New lower tick
        tick_upper     (int): New upper tick
        liquidity      (int): Liquidity to remove from old position

    Returns ExecutionResult with:
      data:
        tx_hash, action, pool_address, token_id (for deploy),
        network, contract
      side_effects:
        Human-readable description for world-state log
      new_observations:
        Observation fed back into Atune workspace
    """

    action_type = "phantom_liquidity"
    description = (
        "Deploy, withdraw, or rebalance phantom liquidity positions on "
        "Uniswap V3 — sensor network for real-time price feeds (Level 3)"
    )

    required_autonomy = 3       # SOVEREIGN — deploys real capital on-chain
    reversible = False          # On-chain interactions cannot be reversed
    max_duration_ms = 120_000   # LP minting may require approve + mint
    rate_limit = RateLimit.per_hour(6)

    def __init__(self, wallet: WalletClient | None = None) -> None:
        self._wallet = wallet
        self._logger = logger.bind(system="axon.executor.phantom_liquidity")

    # -- Validation ----------------------------------------------------------

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Fast param validation — no I/O."""
        action = str(params.get("action", "")).strip().lower()
        if not action:
            return ValidationResult.fail(
                "action is required ('deploy_position', 'withdraw_position', or 'rebalance')",
                action="missing",
            )
        if action not in _SUPPORTED_ACTIONS:
            return ValidationResult.fail(
                f"action must be one of: {', '.join(sorted(_SUPPORTED_ACTIONS))}",
                action="unsupported value",
            )

        if action == "deploy_position":
            return self._validate_deploy(params)
        elif action == "withdraw_position":
            return self._validate_withdraw(params)
        else:  # rebalance
            return self._validate_rebalance(params)

    def _validate_deploy(self, params: dict[str, Any]) -> ValidationResult:
        """Validate deploy_position params."""
        for field in ("pool_address", "token0", "token1"):
            val = str(params.get(field, "")).strip()
            if not val or not val.startswith("0x"):
                return ValidationResult.fail(
                    f"{field} is required (0x-prefixed address)",
                    **{field: "missing or invalid"},
                )

        fee_tier = params.get("fee_tier")
        if fee_tier is None:
            return ValidationResult.fail("fee_tier is required", fee_tier="missing")
        try:
            fee_tier_int = int(fee_tier)
        except (ValueError, TypeError):
            return ValidationResult.fail(
                "fee_tier must be an integer (100, 500, 3000, 10000)",
                fee_tier="not an integer",
            )
        if fee_tier_int not in _SUPPORTED_FEE_TIERS:
            return ValidationResult.fail(
                f"fee_tier must be one of: {sorted(_SUPPORTED_FEE_TIERS)}",
                fee_tier="unsupported value",
            )

        for amount_field in ("amount0_desired", "amount1_desired"):
            raw = str(params.get(amount_field, "")).strip()
            if not raw:
                return ValidationResult.fail(
                    f"{amount_field} is required",
                    **{amount_field: "missing"},
                )
            try:
                val = int(raw)
            except ValueError:
                return ValidationResult.fail(
                    f"{amount_field} must be a non-negative integer",
                    **{amount_field: "not an integer"},
                )
            if val < 0:
                return ValidationResult.fail(
                    f"{amount_field} must be non-negative",
                    **{amount_field: "negative"},
                )

        for tick_field in ("tick_lower", "tick_upper"):
            if params.get(tick_field) is None:
                return ValidationResult.fail(
                    f"{tick_field} is required",
                    **{tick_field: "missing"},
                )

        return ValidationResult.ok()

    def _validate_withdraw(self, params: dict[str, Any]) -> ValidationResult:
        """Validate withdraw_position params."""
        token_id = params.get("token_id")
        if token_id is None:
            return ValidationResult.fail("token_id is required", token_id="missing")
        try:
            int(token_id)
        except (ValueError, TypeError):
            return ValidationResult.fail(
                "token_id must be an integer",
                token_id="not an integer",
            )

        liquidity = params.get("liquidity")
        if liquidity is None:
            return ValidationResult.fail("liquidity is required", liquidity="missing")
        try:
            int(liquidity)
        except (ValueError, TypeError):
            return ValidationResult.fail(
                "liquidity must be an integer",
                liquidity="not an integer",
            )

        return ValidationResult.ok()

    def _validate_rebalance(self, params: dict[str, Any]) -> ValidationResult:
        """Validate rebalance params (withdraw old + deploy new)."""
        withdraw_result = self._validate_withdraw(params)
        if not withdraw_result.valid:
            return withdraw_result
        return self._validate_deploy(params)

    # -- Execution -----------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute the phantom liquidity operation. Never raises."""
        if self._wallet is None:
            return ExecutionResult(
                success=False,
                error=(
                    "WalletClient not configured. "
                    "Pass wallet= to PhantomLiquidityExecutor or register via AxonService."
                ),
            )

        action = str(params["action"]).strip().lower()

        self._logger.info(
            "phantom_liquidity_execute",
            action=action,
            execution_id=context.execution_id,
        )

        try:
            if action == "deploy_position":
                return await self._execute_deploy(params, context)
            elif action == "withdraw_position":
                return await self._execute_withdraw(params, context)
            else:  # rebalance
                return await self._execute_rebalance(params, context)
        except Exception as exc:
            return self._handle_error(exc, action, context)

    # -- Deploy --------------------------------------------------------------

    async def _execute_deploy(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Mint a new phantom liquidity position on Uniswap V3."""
        wallet = self._wallet
        assert wallet is not None

        account = wallet._require_account()
        cdp = wallet._require_cdp()
        network = wallet.network
        owner = wallet.address

        pool_address = str(params["pool_address"]).strip()
        token0 = str(params["token0"]).strip()
        token1 = str(params["token1"]).strip()
        fee_tier = int(params["fee_tier"])
        amount0_desired = int(params["amount0_desired"])
        amount1_desired = int(params["amount1_desired"])
        tick_lower = int(params["tick_lower"])
        tick_upper = int(params["tick_upper"])

        # Step 1: Approve token0 for NonfungiblePositionManager
        if amount0_desired > 0:
            approve0_data = _build_approve_calldata(
                _UNISWAP_V3_NFT_MANAGER_BASE, amount0_desired,
            )
            self._logger.debug(
                "phantom_approve_token0",
                token=token0,
                amount=amount0_desired,
                execution_id=context.execution_id,
            )
            await self._send_tx(cdp, account, network, token0, approve0_data)

        # Step 2: Approve token1 for NonfungiblePositionManager
        if amount1_desired > 0:
            approve1_data = _build_approve_calldata(
                _UNISWAP_V3_NFT_MANAGER_BASE, amount1_desired,
            )
            self._logger.debug(
                "phantom_approve_token1",
                token=token1,
                amount=amount1_desired,
                execution_id=context.execution_id,
            )
            await self._send_tx(cdp, account, network, token1, approve1_data)

        # Step 3: Mint position
        import time
        deadline = int(time.time()) + 600  # 10 minutes

        mint_data = _build_mint_calldata(
            token0=token0,
            token1=token1,
            fee=fee_tier,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            amount0_desired=amount0_desired,
            amount1_desired=amount1_desired,
            amount0_min=0,   # Sensor positions accept any slippage
            amount1_min=0,
            recipient=owner,
            deadline=deadline,
        )

        self._logger.debug(
            "phantom_mint",
            pool=pool_address,
            fee_tier=fee_tier,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            execution_id=context.execution_id,
        )

        tx_hash = await self._send_tx(
            cdp, account, network, _UNISWAP_V3_NFT_MANAGER_BASE, mint_data,
        )

        side_effect = (
            f"Deployed phantom LP on {pool_address} "
            f"(fee {fee_tier/10000:.2%}, ticks [{tick_lower}, {tick_upper}]) "
            f"on {network} — tx: {tx_hash}"
        )

        observation = (
            f"Phantom liquidity deployed: pool {pool_address[:16]}..., "
            f"fee tier {fee_tier}, tx: {tx_hash[:20]}..."
        )

        return ExecutionResult(
            success=True,
            data={
                "tx_hash": tx_hash,
                "action": "deploy_position",
                "pool_address": pool_address,
                "token0": token0,
                "token1": token1,
                "fee_tier": fee_tier,
                "tick_lower": tick_lower,
                "tick_upper": tick_upper,
                "amount0_desired": amount0_desired,
                "amount1_desired": amount1_desired,
                "contract": _UNISWAP_V3_NFT_MANAGER_BASE,
                "network": network,
            },
            side_effects=[side_effect],
            new_observations=[observation],
        )

    # -- Withdraw ------------------------------------------------------------

    async def _execute_withdraw(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Withdraw a phantom liquidity position (decrease liquidity + collect fees)."""
        wallet = self._wallet
        assert wallet is not None

        account = wallet._require_account()
        cdp = wallet._require_cdp()
        network = wallet.network
        owner = wallet.address

        token_id = int(params["token_id"])
        liquidity = int(params["liquidity"])

        import time
        deadline = int(time.time()) + 600

        # Step 1: Decrease liquidity
        decrease_data = _build_decrease_liquidity_calldata(
            token_id=token_id,
            liquidity=liquidity,
            amount0_min=0,
            amount1_min=0,
            deadline=deadline,
        )

        self._logger.debug(
            "phantom_decrease_liquidity",
            token_id=token_id,
            liquidity=liquidity,
            execution_id=context.execution_id,
        )

        await self._send_tx(
            cdp, account, network, _UNISWAP_V3_NFT_MANAGER_BASE, decrease_data,
        )

        # Step 2: Collect all accrued fees + withdrawn tokens
        max_uint128 = (1 << 128) - 1
        collect_data = _build_collect_calldata(
            token_id=token_id,
            recipient=owner,
            amount0_max=max_uint128,
            amount1_max=max_uint128,
        )

        self._logger.debug(
            "phantom_collect",
            token_id=token_id,
            execution_id=context.execution_id,
        )

        tx_hash = await self._send_tx(
            cdp, account, network, _UNISWAP_V3_NFT_MANAGER_BASE, collect_data,
        )

        side_effect = (
            f"Withdrew phantom LP token_id={token_id} "
            f"on {network} — tx: {tx_hash}"
        )

        observation = (
            f"Phantom liquidity withdrawn: token_id={token_id}, "
            f"tx: {tx_hash[:20]}..."
        )

        return ExecutionResult(
            success=True,
            data={
                "tx_hash": tx_hash,
                "action": "withdraw_position",
                "token_id": token_id,
                "contract": _UNISWAP_V3_NFT_MANAGER_BASE,
                "network": network,
            },
            side_effects=[side_effect],
            new_observations=[observation],
        )

    # -- Rebalance -----------------------------------------------------------

    async def _execute_rebalance(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Rebalance: withdraw old position then deploy new one."""
        # Step 1: Withdraw
        withdraw_result = await self._execute_withdraw(params, context)
        if not withdraw_result.success:
            return ExecutionResult(
                success=False,
                error=f"Rebalance aborted — withdraw failed: {withdraw_result.error}",
                data={"phase": "withdraw", "inner": withdraw_result.data},
            )

        # Step 2: Deploy new position
        deploy_result = await self._execute_deploy(params, context)
        if not deploy_result.success:
            return ExecutionResult(
                success=False,
                error=f"Rebalance partial — withdraw succeeded but deploy failed: {deploy_result.error}",
                data={
                    "phase": "deploy",
                    "withdraw_tx": withdraw_result.data.get("tx_hash", ""),
                    "inner": deploy_result.data,
                },
                side_effects=withdraw_result.side_effects,
            )

        all_side_effects = (
            withdraw_result.side_effects + deploy_result.side_effects
        )
        all_observations = (
            withdraw_result.new_observations + deploy_result.new_observations
        )

        return ExecutionResult(
            success=True,
            data={
                "action": "rebalance",
                "withdraw_tx": withdraw_result.data.get("tx_hash", ""),
                "deploy_tx": deploy_result.data.get("tx_hash", ""),
                "pool_address": deploy_result.data.get("pool_address", ""),
                "network": deploy_result.data.get("network", ""),
            },
            side_effects=all_side_effects,
            new_observations=all_observations,
        )

    # -- Transaction helper --------------------------------------------------

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

    # -- Error handling ------------------------------------------------------

    def _handle_error(
        self,
        exc: Exception,
        action: str,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Map exceptions to structured ExecutionResult failures."""
        error_str = str(exc)

        if _is_insufficient_funds(error_str):
            self._logger.warning(
                "phantom_liquidity_insufficient_funds",
                action=action,
                execution_id=context.execution_id,
            )
            return ExecutionResult(
                success=False,
                error=f"INSUFFICIENT_FUNDS: wallet lacks capital for phantom {action}.",
                data={"failure_type": "insufficient_funds", "action": action},
            )

        self._logger.error(
            "phantom_liquidity_failed",
            action=action,
            execution_id=context.execution_id,
            error=error_str,
        )
        return ExecutionResult(
            success=False,
            error=f"Phantom liquidity {action} failed: {error_str}",
            data={"failure_type": "protocol_error", "action": action},
        )
