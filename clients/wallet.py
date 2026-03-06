"""
EcodiaOS — Web3 Wallet Client (CDP SDK v2)

On-chain financial identity for the organism. Uses the Coinbase Developer
Platform SDK to manage an MPC-secured wallet on the Base network.

The CDP SDK manages key material server-side (MPC). We never export or
handle raw private keys — we create accounts by name and retrieve them
by name. A local metadata file records the account name and address so
the organism can verify continuity across restarts.

Capabilities:
  - Create or retrieve an EVM account (CDP manages keys via MPC)
  - Query real on-chain ETH and USDC balances
  - Transfer ETH / USDC to arbitrary addresses

All public methods are async. The CDP SDK v2 client (`CdpClient`) is
natively async, so no `asyncio.to_thread` wrapping is required.

Environment variables consumed (via WalletConfig):
  ECODIAOS_WALLET__CDP_API_KEY_ID       — CDP API key identifier
  ECODIAOS_WALLET__CDP_API_KEY_SECRET   — CDP API key secret
  ECODIAOS_WALLET__CDP_WALLET_SECRET    — CDP wallet-level secret (MPC share)
  ECODIAOS_WALLET__NETWORK              — EVM network id (default: base)
  ECODIAOS_WALLET__ACCOUNT_NAME         — Logical account name (default: ecodiaos-treasury)
  ECODIAOS_WALLET__SEED_FILE_PATH       — Path to local metadata file
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from cdp import CdpClient

if TYPE_CHECKING:
    from cdp.evm_account import EvmAccount
    from cdp.evm_smart_account import EvmSmartAccount

    from config import WalletConfig

logger = structlog.get_logger()

# ─── Constants ────────────────────────────────────────────────────

_USDC_DECIMALS = 6
_ETH_DECIMALS = 18
_SUPPORTED_ASSETS = frozenset({"eth", "usdc"})
_SEED_FILE_VERSION = 2


# ─── Data types ───────────────────────────────────────────────────


class TokenBalance:
    """Immutable snapshot of a single token balance."""

    __slots__ = ("token", "amount", "decimals")

    def __init__(self, token: str, amount: Decimal, decimals: int) -> None:
        self.token = token
        self.amount = amount
        self.decimals = decimals

    def __repr__(self) -> str:
        return f"TokenBalance({self.token}={self.amount})"


class TransferResult:
    """Outcome of an on-chain transfer."""

    __slots__ = ("tx_hash", "token", "amount", "destination", "network")

    def __init__(
        self,
        tx_hash: str,
        token: str,
        amount: str,
        destination: str,
        network: str,
    ) -> None:
        self.tx_hash = tx_hash
        self.token = token
        self.amount = amount
        self.destination = destination
        self.network = network

    def __repr__(self) -> str:
        return f"TransferResult(tx={self.tx_hash[:16]}… {self.amount} {self.token})"


# ─── Wallet Client ────────────────────────────────────────────────


class WalletClient:
    """
    Async CDP wallet client for EcodiaOS.

    Lifecycle: construct → connect() → use → close().
    Mirrors the pattern of Neo4jClient, RedisClient, etc.

    The CDP SDK manages private keys server-side via MPC. This client
    never touches raw key material — accounts are created and retrieved
    by name through the CDP API.
    """

    def __init__(self, config: WalletConfig) -> None:
        self._config = config
        self._cdp: CdpClient | None = None
        self._account: EvmAccount | None = None
        self._smart_account: EvmSmartAccount | None = None

    # ── Lifecycle ─────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialise the CDP client and load or create the EVM account."""
        # wallet_secret is optional — only needed for write operations
        # (account creation, transfers). Pass None if not configured so
        # the SDK doesn't try to parse an empty string as a DER key.
        wallet_secret = self._config.cdp_wallet_secret or None

        if not wallet_secret:
            logger.warning(
                "wallet_secret_not_set",
                hint="CDP_WALLET_SECRET is required for account creation and transfers. "
                     "Generate one from the CDP portal: portal.cdp.coinbase.com",
            )

        self._cdp = CdpClient(
            api_key_id=self._config.cdp_api_key_id,
            api_key_secret=self._config.cdp_api_key_secret,
            wallet_secret=wallet_secret,
        )

        try:
            # Enter the async context manager — the SDK requires it
            await self._cdp.__aenter__()

            # Try to restore from local metadata, otherwise create new
            seed_path = Path(self._config.seed_file_path)
            if seed_path.exists():
                await self._restore_account(seed_path)
            else:
                await self._create_account(seed_path)

            address = self._account.address if self._account else "unknown"
            logger.info(
                "wallet_connected",
                network=self._config.network,
                account_name=self._config.account_name,
                address=address,
            )
        except Exception:
            # Ensure the aiohttp session is cleaned up on failure
            await self.close()
            raise

    async def close(self) -> None:
        """Tear down the CDP client."""
        if self._cdp is not None:
            try:
                await self._cdp.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("wallet_close_error", error=str(e))
            self._cdp = None
        self._account = None
        self._smart_account = None
        logger.info("wallet_disconnected")

    # ── Account management ────────────────────────────────────

    async def _create_account(self, seed_path: Path) -> None:
        """Create a new MPC-backed EVM account via CDP.

        CDP manages the key material server-side. We only persist metadata
        locally (account name + address) for startup verification.
        """
        cdp = self._require_cdp()

        self._account = await cdp.evm.get_or_create_account(
            name=self._config.account_name,
        )

        # Persist metadata (not key material — CDP holds that via MPC)
        self._save_metadata(seed_path)

        logger.info(
            "wallet_account_created",
            address=self._account.address,
            metadata_file=str(seed_path),
        )

    async def _restore_account(self, seed_path: Path) -> None:
        """Restore an EVM account by name from CDP.

        The local metadata file tells us which account name to request.
        CDP returns the same MPC-backed account — no key import needed.
        """
        cdp = self._require_cdp()
        metadata = self._load_metadata(seed_path)
        account_name = metadata.get("account_name", self._config.account_name)

        # get_or_create retrieves existing account by name, or creates
        # if it was somehow deleted from CDP (shouldn't happen normally)
        self._account = await cdp.evm.get_or_create_account(
            name=account_name,
        )

        expected_address = metadata.get("address")
        actual_address = self._account.address if self._account else None

        if expected_address and actual_address and expected_address != actual_address:
            logger.warning(
                "wallet_address_mismatch",
                expected=expected_address,
                actual=actual_address,
                hint="Account name resolved to a different address. "
                     "The metadata file may be stale.",
            )
            # Update metadata to reflect the actual address
            self._save_metadata(seed_path)

        logger.info(
            "wallet_account_restored",
            address=actual_address,
            metadata_file=str(seed_path),
        )

    def _save_metadata(self, seed_path: Path) -> None:
        """Write local metadata file (account name + address, no key material)."""
        seed_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": _SEED_FILE_VERSION,
            "account_name": self._config.account_name,
            "network": self._config.network,
            "address": self._account.address if self._account else None,
        }

        seed_path.write_text(json.dumps(payload, indent=2))
        logger.info("wallet_metadata_saved", path=str(seed_path))

    def _load_metadata(self, seed_path: Path) -> dict[str, Any]:
        """Read local metadata file."""
        raw: dict[str, Any] = json.loads(seed_path.read_text())
        version = raw.get("version", 0)
        if version not in (1, _SEED_FILE_VERSION):
            raise ValueError(f"Unsupported wallet metadata version: {version}")
        return raw

    # ── Balance queries ───────────────────────────────────────

    async def get_balances(self) -> list[TokenBalance]:
        """Fetch all token balances for the account on the configured network."""
        account = self._require_account()

        response: Any = await account.list_token_balances(
            network=self._config.network,
        )

        # The CDP SDK returns a paged response that iterates as (key, value) tuples.
        # Extract the 'balances' list from the response.
        response_dict: dict[str, Any] = dict(response)
        raw_balances: list[Any] = response_dict.get("balances", [])

        balances: list[TokenBalance] = []
        for b in raw_balances:
            token = b.token
            symbol = str(getattr(token, "symbol", "unknown")).lower()
            token_amount = b.amount
            raw_amount = int(getattr(token_amount, "amount", 0))
            decimals = int(getattr(token_amount, "decimals", 18))
            amount = Decimal(raw_amount) / Decimal(10 ** decimals)
            balances.append(TokenBalance(token=symbol, amount=amount, decimals=decimals))

        logger.debug(
            "wallet_balances_fetched",
            count=len(balances),
            network=self._config.network,
        )
        return balances

    async def get_eth_balance(self) -> Decimal:
        """Convenience: return ETH balance as a Decimal."""
        balances = await self.get_balances()
        for b in balances:
            if b.token == "eth":
                return b.amount
        return Decimal(0)

    async def get_usdc_balance(self) -> Decimal:
        """Convenience: return USDC balance as a Decimal."""
        balances = await self.get_balances()
        for b in balances:
            if b.token == "usdc":
                return b.amount
        return Decimal(0)

    # ── Transfers ─────────────────────────────────────────────

    async def transfer(
        self,
        amount: str,
        destination_address: str,
        asset: str = "usdc",
    ) -> TransferResult:
        """
        Transfer funds on-chain.

        Args:
            amount: Human-readable amount (e.g. "10.50" for 10.50 USDC).
            destination_address: The 0x-prefixed recipient address.
            asset: "usdc" or "eth".

        Returns:
            TransferResult with the transaction hash.

        Raises:
            ValueError: If the asset is unsupported.
        """
        asset_lower = asset.lower()
        if asset_lower not in _SUPPORTED_ASSETS:
            raise ValueError(
                f"Unsupported asset '{asset}'. Must be one of: {', '.join(sorted(_SUPPORTED_ASSETS))}"
            )

        account = self._require_account()

        # The CDP SDK's transfer() accepts raw integer amounts (wei / smallest unit).
        # We convert from the human-readable string.
        if asset_lower == "eth":
            from web3 import Web3

            raw_amount = Web3.to_wei(Decimal(amount), "ether")
        else:
            # USDC: 6 decimals
            raw_amount = int(Decimal(amount) * (10 ** _USDC_DECIMALS))

        tx_hash: str = await account.transfer(
            to=destination_address,
            amount=raw_amount,
            token=asset_lower,
            network=self._config.network,
        )

        result = TransferResult(
            tx_hash=str(tx_hash),
            token=asset_lower,
            amount=amount,
            destination=destination_address,
            network=self._config.network,
        )

        logger.info(
            "wallet_transfer_sent",
            tx_hash=result.tx_hash,
            token=result.token,
            amount=amount,
            destination=destination_address,
            network=self._config.network,
        )

        return result

    # ── Smart Account (optional) ──────────────────────────────

    async def get_or_create_smart_account(self) -> EvmSmartAccount:
        """Get or create a smart account (ERC-4337) owned by the EOA.

        Smart accounts enable gas-sponsored transactions and batched ops.
        """
        if self._smart_account is not None:
            return self._smart_account

        cdp = self._require_cdp()
        account = self._require_account()

        self._smart_account = await cdp.evm.get_or_create_smart_account(
            name=f"{self._config.account_name}-smart",
            owner=account,
        )

        logger.info(
            "wallet_smart_account_ready",
            address=self._smart_account.address,
        )
        return self._smart_account

    # ── Health ────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Return wallet health status — account address & balance snapshot."""
        try:
            account = self._require_account()
            balances = await self.get_balances()
            balance_map = {b.token: str(b.amount) for b in balances}
            return {
                "status": "connected",
                "address": account.address,
                "network": self._config.network,
                "balances": balance_map,
            }
        except Exception as e:
            logger.error("wallet_health_check_failed", error=str(e))
            return {"status": "disconnected", "error": str(e)}

    # ── Properties ────────────────────────────────────────────

    @property
    def address(self) -> str:
        """The primary account's 0x address."""
        return self._require_account().address  # type: ignore[return-value]

    @property
    def network(self) -> str:
        return self._config.network

    # ── Internal helpers ──────────────────────────────────────

    def _require_cdp(self) -> CdpClient:
        if self._cdp is None:
            raise RuntimeError("WalletClient not connected. Call connect() first.")
        return self._cdp

    def _require_account(self) -> EvmAccount:
        if self._account is None:
            raise RuntimeError("No EVM account loaded. Call connect() first.")
        return self._account
