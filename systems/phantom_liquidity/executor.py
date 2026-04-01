from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.wallet import WalletClient
    from systems.phantom_liquidity.types import PhantomLiquidityPool

logger = structlog.get_logger()
NPM_ADDRESS = "0x03a520b32C04BF3bEEf7BEb72E919cf822Ed34f1"
_MAX_UINT128 = 2**128 - 1
_SLIPPAGE_BPS = 500
_DEADLINE_S = 1200
_ERC20_ABI = [
    {
        "name": "approve", "type": "function", "stateMutability": "nonpayable",
        "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "allowance", "type": "function", "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]
_NPM_ABI = [
    {
        "name": "mint", "type": "function", "stateMutability": "payable",
        "inputs": [{"name": "params", "type": "tuple", "components": [
            {"name": "token0", "type": "address"},
            {"name": "token1", "type": "address"},
            {"name": "fee", "type": "uint24"},
            {"name": "tickLower", "type": "int24"},
            {"name": "tickUpper", "type": "int24"},
            {"name": "amount0Desired", "type": "uint256"},
            {"name": "amount1Desired", "type": "uint256"},
            {"name": "amount0Min", "type": "uint256"},
            {"name": "amount1Min", "type": "uint256"},
            {"name": "recipient", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ]}],
        "outputs": [
            {"name": "tokenId", "type": "uint256"},
            {"name": "liquidity", "type": "uint128"},
            {"name": "amount0", "type": "uint256"},
            {"name": "amount1", "type": "uint256"},
        ],
    },
    {
        "name": "positions", "type": "function", "stateMutability": "view",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [
            {"name": "nonce", "type": "uint96"}, {"name": "operator", "type": "address"},
            {"name": "token0", "type": "address"}, {"name": "token1", "type": "address"},
            {"name": "fee", "type": "uint24"}, {"name": "tickLower", "type": "int24"},
            {"name": "tickUpper", "type": "int24"}, {"name": "liquidity", "type": "uint128"},
            {"name": "feeGrowthInside0LastX128", "type": "uint256"},
            {"name": "feeGrowthInside1LastX128", "type": "uint256"},
            {"name": "tokensOwed0", "type": "uint128"}, {"name": "tokensOwed1", "type": "uint128"},
        ],
    },
    {
        "name": "decreaseLiquidity", "type": "function", "stateMutability": "payable",
        "inputs": [{"name": "params", "type": "tuple", "components": [
            {"name": "tokenId", "type": "uint256"}, {"name": "liquidity", "type": "uint128"},
            {"name": "amount0Min", "type": "uint256"}, {"name": "amount1Min", "type": "uint256"},
            {"name": "deadline", "type": "uint256"},
        ]}],
        "outputs": [{"name": "amount0", "type": "uint256"}, {"name": "amount1", "type": "uint256"}],
    },
    {
        "name": "collect", "type": "function", "stateMutability": "payable",
        "inputs": [{"name": "params", "type": "tuple", "components": [
            {"name": "tokenId", "type": "uint256"}, {"name": "recipient", "type": "address"},
            {"name": "amount0Max", "type": "uint128"}, {"name": "amount1Max", "type": "uint128"},
        ]}],
        "outputs": [{"name": "amount0", "type": "uint256"}, {"name": "amount1", "type": "uint256"}],
    },
    {
        "name": "burn", "type": "function", "stateMutability": "payable",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [],
    },
]

class PhantomLiquidityExecutor:
    """Executes Uniswap V3 LP operations for phantom sensor positions."""

    def __init__(self, wallet: WalletClient) -> None:
        self._wallet = wallet
        self._log = logger.bind(system="phantom_liquidity", component="executor")

    async def mint_position(self, pool: PhantomLiquidityPool) -> dict[str, Any]:
        """Mint a concentrated liquidity position. Returns token_id, tx_hash, amount0, amount1."""
        from web3 import AsyncWeb3
        from web3.providers import AsyncHTTPProvider

        rpc_url = getattr(self._wallet._config, "rpc_url", None) or "https://mainnet.base.org"
        w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
        addr = self._wallet.address
        npm = w3.eth.contract(address=w3.to_checksum_address(NPM_ADDRESS), abi=_NPM_ABI)

        capital = float(pool.capital_deployed_usd)
        half = capital / 2
        a0d = int(half * (10 ** pool.token0_decimals))
        a1d = int(half * (10 ** pool.token1_decimals))
        a0m = a0d * (10000 - _SLIPPAGE_BPS) // 10000
        a1m = a1d * (10000 - _SLIPPAGE_BPS) // 10000
        deadline = int(time.time()) + _DEADLINE_S

        await self._ensure_approval(w3, pool.token0_address, NPM_ADDRESS, a0d, addr)
        await self._ensure_approval(w3, pool.token1_address, NPM_ADDRESS, a1d, addr)

        params = (
            w3.to_checksum_address(pool.token0_address),
            w3.to_checksum_address(pool.token1_address),
            pool.fee_tier, pool.tick_lower, pool.tick_upper,
            a0d, a1d, a0m, a1m,
            w3.to_checksum_address(addr), deadline,
        )
        cd = npm.encode_abi("mint", args=[params])
        tx = await self._send_tx(w3, NPM_ADDRESS, cd)
        token_id, amount0, amount1 = await self._parse_mint_receipt(w3, tx)
        self._log.info("phantom_mint_ok", pool=pool.pool_address, token_id=token_id, tx=tx)
        return {"token_id": token_id, "tx_hash": tx, "amount0": amount0, "amount1": amount1}
    async def burn_position(self, pool: PhantomLiquidityPool) -> dict[str, Any]:
        """Remove a position: decreaseLiquidity -> collect -> burn NFT."""
        from web3 import AsyncWeb3
        from web3.providers import AsyncHTTPProvider

        rpc_url = getattr(self._wallet._config, "rpc_url", None) or "https://mainnet.base.org"
        w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
        npm = w3.eth.contract(address=w3.to_checksum_address(NPM_ADDRESS), abi=_NPM_ABI)
        addr = self._wallet.address
        tid = pool.token_id
        deadline = int(time.time()) + _DEADLINE_S

        pos = await npm.functions.positions(tid).call()
        liquidity = pos[7]

        if liquidity > 0:
            dl = (tid, liquidity, 0, 0, deadline)
            await self._send_tx(w3, NPM_ADDRESS, npm.encode_abi("decreaseLiquidity", args=[dl]))

        col = (tid, w3.to_checksum_address(addr), _MAX_UINT128, _MAX_UINT128)
        await self._send_tx(w3, NPM_ADDRESS, npm.encode_abi("collect", args=[col]))

        burn_cd = npm.encode_abi("burn", args=[tid])
        tx = await self._send_tx(w3, NPM_ADDRESS, burn_cd)
        self._log.info("phantom_burn_ok", pool=pool.pool_address, token_id=tid, tx=tx)
        return {"tx_hash": tx}
    async def _ensure_approval(
        self, w3: Any, token_addr: str, spender: str, amount: int, owner: str,
    ) -> None:
        tok = w3.eth.contract(address=w3.to_checksum_address(token_addr), abi=_ERC20_ABI)
        current: int = await tok.functions.allowance(
            w3.to_checksum_address(owner), w3.to_checksum_address(spender),
        ).call()
        if current >= amount:
            return
        cd = tok.encode_abi("approve", args=[w3.to_checksum_address(spender), 2**256 - 1])
        await self._send_tx(w3, token_addr, cd)

    async def _send_tx(self, w3: Any, to: str, calldata: bytes | str, value: int = 0) -> str:
        account = self._wallet._require_account()
        cdp = self._wallet._require_cdp()
        data_hex = calldata if isinstance(calldata, str) else "0x" + calldata.hex()
        tx_desc = {"to": w3.to_checksum_address(to), "data": data_hex, "value": hex(value)}
        tx_hash = await cdp.evm.send_transaction(
            account=account, transaction=tx_desc, network=self._wallet.network,
        )
        receipt = await self._wait_for_receipt(w3, str(tx_hash))
        if receipt and receipt.get("status") == 0:
            raise RuntimeError(f"Transaction reverted: {tx_hash}")
        return str(tx_hash)

    async def _wait_for_receipt(
        self, w3: Any, tx_hash: str, timeout_s: int = 60,
    ) -> dict[str, Any] | None:
        cutoff = time.monotonic() + timeout_s
        while time.monotonic() < cutoff:
            try:
                r = await w3.eth.get_transaction_receipt(tx_hash)
                if r is not None:
                    return dict(r)
            except Exception:
                pass
            await asyncio.sleep(2)
        return None

    async def _parse_mint_receipt(self, w3: Any, tx_hash: str) -> tuple[int, int, int]:
        # Full 32-byte keccak256 of IncreaseLiquidity(uint256,uint128,uint256,uint256)
        # Spec §4.1 - token_id is indexed topic[1]; liquidity/amount0/amount1 in data
        INC_LIQ = "0x3067048beee31b25b2f1681f88dac838c60bfd8b75c9c62c9b72dd6b9d1d6b19"
        try:
            receipt = await self._wait_for_receipt(w3, tx_hash)
            if not receipt:
                return (0, 0, 0)
            for entry in receipt.get("logs", []):
                topics = entry.get("topics", [])
                if not topics:
                    continue
                t0_raw = topics[0]
                t0 = ("0x" + t0_raw.hex()) if isinstance(t0_raw, bytes) else str(t0_raw)
                if t0.lower() != INC_LIQ:
                    continue
                raw_data = entry.get("data", "0x")
                raw = (raw_data.hex() if isinstance(raw_data, bytes) else raw_data).replace("0x", "")
                # data layout: [liquidity uint128][amount0 uint256][amount1 uint256] (3 × 32 bytes)
                if len(raw) < 192:  # noqa: PLR2004
                    continue
                t1 = topics[1]
                token_id = int(t1.hex() if isinstance(t1, bytes) else str(t1), 16)
                amount0 = int(raw[64:128], 16)
                amount1 = int(raw[128:192], 16)
                return (token_id, amount0, amount1)
        except Exception as exc:
            self._log.debug("parse_mint_receipt_failed", error=str(exc))
        return (0, 0, 0)
