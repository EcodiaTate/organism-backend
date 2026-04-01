"""
EcodiaOS - Tollbooth Smart Contract Wiring (Phase 16d: Entrepreneurship)

Conceptual deployment and interaction layer for the EosTollbooth smart
contract on Base L2. The tollbooth gates access to autonomous services -
users pay USDC per API call, and the organism collects revenue.

Architecture:
  - Each deployed asset gets its own tollbooth contract instance
  - The contract holds a price_per_call in USDC (6 decimals)
  - Users call `pay(asset_endpoint_hash)` which transfers USDC and emits a receipt
  - The service backend validates the receipt before processing the request
  - The organism (owner) can withdraw accumulated revenue at any time
  - Owner can update price, pause, or destroy the tollbooth

Contract ABI (Solidity):
  constructor(address _usdc, address _owner, uint256 _pricePerCall)
  function pay(bytes32 endpointHash) external returns (bytes32 receiptId)
  function withdraw() external onlyOwner
  function setPrice(uint256 newPrice) external onlyOwner
  function pause() external onlyOwner
  function unpause() external onlyOwner
  function getReceipt(bytes32 receiptId) external view returns (Receipt)
  function accumulatedRevenue() external view returns (uint256)

This module provides:
  - Calldata encoding for deploying and interacting with the tollbooth
  - Revenue sweep logic (withdraw accumulated USDC)
  - Receipt validation for service backends

Thread-safety: NOT thread-safe. Designed for single-threaded asyncio event loop.
"""

from __future__ import annotations

import contextlib
import hashlib
from decimal import Decimal
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:

    from clients.wallet import WalletClient
logger = structlog.get_logger("oikos.tollbooth")

# ─── Base L2 Contract Addresses ──────────────────────────────────

USDC_BASE: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# ─── EosTollbooth ABI Function Selectors ─────────────────────────
# Computed as keccak256(signature)[:4].hex()

# pay(bytes32) -> bytes32
SELECTOR_PAY: str = "0xc290d691"

# withdraw()
SELECTOR_WITHDRAW: str = "0x3ccfd60b"

# setPrice(uint256)
SELECTOR_SET_PRICE: str = "0x91b7f5ed"

# pause()
SELECTOR_PAUSE: str = "0x8456cb59"

# unpause()
SELECTOR_UNPAUSE: str = "0x3f4ba83a"

# accumulatedRevenue() -> uint256
SELECTOR_ACCUMULATED_REVENUE: str = "0x4a0a3282"

# getReceipt(bytes32) -> (address payer, uint256 amount, uint256 timestamp, bytes32 endpoint)
SELECTOR_GET_RECEIPT: str = "0x1b2ef1ca"


# ─── Data Types ──────────────────────────────────────────────────


class TollboothReceipt(EOSBaseModel):
    """A validated payment receipt from the tollbooth contract."""

    receipt_id: str
    payer_address: str
    amount_usdc: Decimal
    endpoint_hash: str
    block_number: int = 0
    timestamp: datetime | None = None


class TollboothDeployment(EOSBaseModel):
    """Record of a deployed tollbooth contract instance."""

    deployment_id: str = ""
    contract_address: str = ""
    asset_id: str = ""
    owner_address: str = ""
    price_per_call_usdc: Decimal = Decimal("0.01")
    deployed_at: datetime | None = None
    tx_hash: str = ""
    chain: str = "base"


# ─── Calldata Encoding ──────────────────────────────────────────


def encode_price_usdc(price_usd: Decimal) -> int:
    """
    Convert a human-readable USD price to USDC on-chain representation.
    USDC has 6 decimals: $0.01 -> 10000.
    """
    return int(price_usd * Decimal("1000000"))


def encode_deploy_calldata(
    usdc_address: str,
    owner_address: str,
    price_per_call_usd: Decimal,
) -> str:
    """
    Encode constructor arguments for EosTollbooth deployment.

    In production, this would be concatenated with the contract bytecode
    and sent as a CREATE transaction via WalletClient.
    """
    price_raw = encode_price_usdc(price_per_call_usd)

    # ABI-encode: (address _usdc, address _owner, uint256 _pricePerCall)
    usdc_padded = usdc_address.lower().replace("0x", "").zfill(64)
    owner_padded = owner_address.lower().replace("0x", "").zfill(64)
    price_padded = hex(price_raw)[2:].zfill(64)

    return usdc_padded + owner_padded + price_padded


def encode_pay_calldata(endpoint_url: str) -> str:
    """
    Encode calldata for the pay(bytes32) function.

    The endpoint is hashed to a bytes32 value so the contract can
    track payments per endpoint without storing the full URL.
    """
    endpoint_hash = hashlib.sha256(endpoint_url.encode()).hexdigest()
    return SELECTOR_PAY + endpoint_hash.zfill(64)


def encode_withdraw_calldata() -> str:
    """Encode calldata for the withdraw() function."""
    return SELECTOR_WITHDRAW


def encode_set_price_calldata(new_price_usd: Decimal) -> str:
    """Encode calldata for the setPrice(uint256) function."""
    price_raw = encode_price_usdc(new_price_usd)
    price_padded = hex(price_raw)[2:].zfill(64)
    return SELECTOR_SET_PRICE + price_padded


# ─── Tollbooth Manager ──────────────────────────────────────────


class TollboothManager:
    """
    Manages tollbooth smart contract lifecycle for all deployed assets.

    Responsibilities:
      - Deploy new tollbooth contracts for assets
      - Sweep accumulated revenue from tollbooths to the organism wallet
      - Validate payment receipts for service backends
      - Update pricing on existing tollbooths
    """

    def __init__(self, wallet: WalletClient | None = None) -> None:
        self._wallet = wallet
        self._deployments: dict[str, TollboothDeployment] = {}
        self._logger = logger.bind(component="tollbooth_manager")

    async def deploy_tollbooth(
        self,
        asset_id: str,
        price_per_call_usd: Decimal,
    ) -> TollboothDeployment:
        """
        Deploy a new EosTollbooth contract for the given asset.

        In production:
          1. Compile/load the EosTollbooth bytecode
          2. Append encoded constructor args
          3. Send a CREATE transaction via WalletClient
          4. Wait for confirmation and record the contract address

        Phase 16d: Returns a deterministic deployment record that will
        be replaced when the full contract deployment is wired.
        """
        owner_address = ""
        if self._wallet is not None:
            with contextlib.suppress(Exception):
                owner_address = self._wallet.address

        # Generate deterministic placeholder address
        raw = hashlib.sha256(f"eos-tollbooth-{asset_id}".encode()).hexdigest()[:40]
        contract_address = f"0x{raw}"

        constructor_calldata = encode_deploy_calldata(
            usdc_address=USDC_BASE,
            owner_address=owner_address or "0x" + "0" * 40,
            price_per_call_usd=price_per_call_usd,
        )

        deployment = TollboothDeployment(
            deployment_id=new_id(),
            contract_address=contract_address,
            asset_id=asset_id,
            owner_address=owner_address,
            price_per_call_usdc=price_per_call_usd,
            deployed_at=utc_now(),
            tx_hash=f"0x{'0' * 64}",  # Placeholder until real deployment
            chain="base",
        )

        self._deployments[asset_id] = deployment

        self._logger.info(
            "tollbooth_deployed",
            asset_id=asset_id,
            contract_address=contract_address,
            price_per_call=str(price_per_call_usd),
            constructor_calldata_length=len(constructor_calldata),
        )

        return deployment

    async def sweep_revenue(self, asset_id: str) -> Decimal:
        """
        Withdraw accumulated revenue from the tollbooth contract.

        In production:
          1. Call accumulatedRevenue() to check balance
          2. Call withdraw() to transfer USDC to owner wallet
          3. Return the amount swept

        Phase 16d: Returns Decimal("0") - actual sweeps require
        deployed contracts and WalletClient integration.
        """
        deployment = self._deployments.get(asset_id)
        if deployment is None:
            self._logger.warning(
                "tollbooth_sweep_no_deployment",
                asset_id=asset_id,
            )
            return Decimal("0")

        # In production, this would:
        # 1. Encode accumulatedRevenue() call
        # 2. Execute eth_call to read balance
        # 3. If balance > 0, encode withdraw() and send transaction
        # 4. Return the swept amount

        self._logger.debug(
            "tollbooth_sweep_pending",
            asset_id=asset_id,
            contract=deployment.contract_address,
        )

        return Decimal("0")

    async def update_price(
        self,
        asset_id: str,
        new_price_usd: Decimal,
    ) -> bool:
        """
        Update the per-call price on a deployed tollbooth.

        In production, sends a setPrice() transaction to the contract.
        """
        deployment = self._deployments.get(asset_id)
        if deployment is None:
            return False

        calldata = encode_set_price_calldata(new_price_usd)
        deployment.price_per_call_usdc = new_price_usd

        self._logger.info(
            "tollbooth_price_updated",
            asset_id=asset_id,
            new_price=str(new_price_usd),
            calldata_length=len(calldata),
        )
        return True

    def get_deployment(self, asset_id: str) -> TollboothDeployment | None:
        """Return the deployment record for a given asset, or None."""
        return self._deployments.get(asset_id)

    @property
    def stats(self) -> dict[str, object]:
        """Summary stats for observability."""
        return {
            "total_deployments": len(self._deployments),
            "asset_ids": list(self._deployments.keys()),
        }
