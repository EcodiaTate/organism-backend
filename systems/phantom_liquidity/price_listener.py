"""
EcodiaOS — Phantom Liquidity Swap Event Listener (Phase 16q)

Async background poller that monitors Uniswap V3 ``Swap`` events on Base L2
to extract real-time price feeds from phantom liquidity positions.

Design mirrors ``systems/atune/block_competition.py``:
  - Non-blocking: runs on its own asyncio task
  - Graceful degradation: emits stale-warnings if RPC is unavailable
  - Rate-limited: polls at configurable intervals (default 4s ≈ 3 Base blocks)

Price extraction from sqrtPriceX96:
  price = (sqrtPriceX96 / 2^96)^2 * 10^(token0_decimals - token1_decimals)
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from systems.phantom_liquidity.types import PhantomPriceFeed

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from web3 import AsyncWeb3

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Uniswap V3 Swap event
# ---------------------------------------------------------------------------
# event Swap(
#     address indexed sender,
#     address indexed recipient,
#     int256 amount0,
#     int256 amount1,
#     uint160 sqrtPriceX96,
#     uint128 liquidity,
#     int24 tick
# )
_SWAP_EVENT_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# 2^96 as Decimal for price computation
_TWO_96 = Decimal(2**96)


# ---------------------------------------------------------------------------
# Price math
# ---------------------------------------------------------------------------


def sqrt_price_x96_to_price(
    sqrt_price_x96: int,
    token0_decimals: int,
    token1_decimals: int,
) -> Decimal:
    """
    Convert Uniswap V3 sqrtPriceX96 to a human-readable price.

    The pool stores price as sqrt(token1/token0) * 2^96.
    So: price(token0 in terms of token1) = (sqrtPriceX96 / 2^96)^2

    The decimal adjustment accounts for the difference in token decimals:
      price_adjusted = price_raw * 10^(token0_decimals - token1_decimals)

    Returns the price of token0 denominated in token1.
    """
    if sqrt_price_x96 == 0:
        return Decimal("0")

    price_raw = (Decimal(sqrt_price_x96) / _TWO_96) ** 2
    decimal_adjustment = Decimal(10 ** (token0_decimals - token1_decimals))
    return price_raw * decimal_adjustment


def _decode_swap_data(data_hex: str) -> tuple[int, int, int, int, int]:
    """
    Decode the non-indexed data portion of a Uniswap V3 Swap event.

    Layout (each field is 32 bytes / 64 hex chars):
      [0]  int256  amount0
      [1]  int256  amount1
      [2]  uint160 sqrtPriceX96
      [3]  uint128 liquidity
      [4]  int24   tick

    Returns (amount0, amount1, sqrtPriceX96, liquidity, tick).
    """
    raw = data_hex.replace("0x", "")

    def _uint(offset: int) -> int:
        return int(raw[offset * 64: (offset + 1) * 64], 16)

    def _int256(offset: int) -> int:
        val = int(raw[offset * 64: (offset + 1) * 64], 16)
        if val >= 2**255:
            val -= 2**256
        return val

    def _int24(offset: int) -> int:
        val = int(raw[offset * 64: (offset + 1) * 64], 16)
        if val >= 2**23:
            val -= 2**24
        return val

    amount0 = _int256(0)
    amount1 = _int256(1)
    sqrt_price_x96 = _uint(2)
    liquidity = _uint(3)
    tick = _int24(4)

    return (amount0, amount1, sqrt_price_x96, liquidity, tick)


# ---------------------------------------------------------------------------
# Pool metadata (resolved once and cached)
# ---------------------------------------------------------------------------


class _PoolMeta:
    """Cached metadata for a monitored pool."""

    __slots__ = (
        "address", "pair", "token0_decimals", "token1_decimals",
    )

    def __init__(
        self,
        address: str,
        pair: tuple[str, str],
        token0_decimals: int,
        token1_decimals: int,
    ) -> None:
        self.address = address.lower()
        self.pair = pair
        self.token0_decimals = token0_decimals
        self.token1_decimals = token1_decimals


# ---------------------------------------------------------------------------
# Swap Event Listener
# ---------------------------------------------------------------------------


class SwapEventListener:
    """
    Background poller for Uniswap V3 Swap events on Base L2.

    Uses ``eth_getLogs`` polling (not WebSocket) because the CDP SDK and
    existing EcodiaOS infrastructure are HTTP-based.

    Lifecycle::

        listener = SwapEventListener(rpc_url="https://mainnet.base.org")
        listener.add_pool("0x...", ("USDC", "ETH"), 6, 18)
        listener.add_listener(on_price_callback)
        await listener.start()
        # ... later ...
        await listener.stop()
    """

    def __init__(
        self,
        rpc_url: str = "",
        poll_interval_s: float = 4.0,
    ) -> None:
        self._rpc_url = rpc_url
        self._poll_interval_s = poll_interval_s
        self._w3: AsyncWeb3 | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False

        # Pool registry: address (lowercase) -> metadata
        self._pools: dict[str, _PoolMeta] = {}

        # Listeners
        self._listeners: list[Callable[[PhantomPriceFeed], Awaitable[None]]] = []

        # Block tracking
        self._last_block: int = 0

        # Metrics
        self._polls: int = 0
        self._events_processed: int = 0
        self._errors: int = 0

        self._logger = logger.bind(
            system="phantom_liquidity", component="swap_listener",
        )

    # ── Pool management ────────────────────────────────────────────

    def add_pool(
        self,
        address: str,
        pair: tuple[str, str],
        token0_decimals: int,
        token1_decimals: int,
    ) -> None:
        """Register a pool for Swap event monitoring."""
        self._pools[address.lower()] = _PoolMeta(
            address=address,
            pair=pair,
            token0_decimals=token0_decimals,
            token1_decimals=token1_decimals,
        )

    def remove_pool(self, address: str) -> None:
        """Stop monitoring a pool."""
        self._pools.pop(address.lower(), None)

    @property
    def pool_addresses(self) -> list[str]:
        """Return monitored pool addresses (checksummed)."""
        return [m.address for m in self._pools.values()]

    # ── Listener registration ──────────────────────────────────────

    def add_listener(
        self,
        callback: Callable[[PhantomPriceFeed], Awaitable[None]],
    ) -> None:
        """Register an async callback for price feed updates."""
        self._listeners.append(callback)

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling task."""
        if self._running:
            return

        if not self._rpc_url:
            self._logger.info("swap_listener_disabled", reason="no_rpc_url")
            return

        if not self._pools:
            self._logger.info("swap_listener_disabled", reason="no_pools")
            return

        try:
            from web3 import AsyncWeb3
            from web3.providers import AsyncHTTPProvider

            self._w3 = AsyncWeb3(AsyncHTTPProvider(self._rpc_url))
            is_connected = await self._w3.is_connected()

            if not is_connected:
                self._logger.warning("swap_listener_rpc_unreachable")
                self._w3 = None
                return
        except Exception as exc:
            self._logger.warning("swap_listener_connect_failed", error=str(exc))
            return

        # Start from the latest block
        try:
            self._last_block = await self._w3.eth.block_number
        except Exception:
            self._last_block = 0

        self._running = True
        self._task = asyncio.create_task(
            self._poll_loop(), name="phantom_swap_listener",
        )
        self._logger.info(
            "swap_listener_started",
            poll_interval_s=self._poll_interval_s,
            pools=len(self._pools),
            from_block=self._last_block,
        )

    async def stop(self) -> None:
        """Stop the background polling task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._w3 = None
        self._logger.info(
            "swap_listener_stopped",
            total_polls=self._polls,
            total_events=self._events_processed,
        )

    # ── Poll loop ──────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Background polling loop — runs until stopped."""
        while self._running:
            try:
                await self._poll_once()
                self._polls += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._errors += 1
                self._logger.debug("swap_listener_poll_error", error=str(exc))

            await asyncio.sleep(self._poll_interval_s)

    async def _poll_once(self) -> None:
        """Fetch new Swap events since last processed block."""
        if self._w3 is None or not self._pools:
            return

        pool_addresses = list(self._pools.keys())

        try:
            current_block = await self._w3.eth.block_number
        except Exception:
            return

        if current_block <= self._last_block:
            return

        # Build filter for all monitored pools
        filter_params: dict[str, Any] = {
            "address": pool_addresses,
            "topics": [_SWAP_EVENT_TOPIC],
            "fromBlock": hex(self._last_block + 1),
            "toBlock": hex(current_block),
        }

        try:
            logs = await self._w3.eth.get_logs(filter_params)
        except Exception as exc:
            self._logger.debug(
                "swap_listener_get_logs_failed",
                error=str(exc),
                from_block=self._last_block + 1,
                to_block=current_block,
            )
            return

        poll_start = time.monotonic()

        for log_entry in logs:
            try:
                await self._process_swap_log(log_entry, poll_start)
                self._events_processed += 1
            except Exception as exc:
                self._logger.debug(
                    "swap_event_decode_error",
                    error=str(exc),
                    tx_hash=getattr(log_entry, "transactionHash", b"").hex(),
                )

        self._last_block = current_block

    async def _process_swap_log(
        self,
        log_entry: Any,
        poll_start: float,
    ) -> None:
        """Decode a single Swap event log and notify listeners."""
        pool_addr = log_entry["address"].lower()
        meta = self._pools.get(pool_addr)
        if meta is None:
            return

        data_hex = log_entry["data"]
        if isinstance(data_hex, bytes):
            data_hex = "0x" + data_hex.hex()

        _amount0, _amount1, sqrt_price_x96, _liquidity, _tick = _decode_swap_data(
            data_hex,
        )

        price = sqrt_price_x96_to_price(
            sqrt_price_x96,
            meta.token0_decimals,
            meta.token1_decimals,
        )

        block_number = (
            log_entry["blockNumber"]
            if isinstance(log_entry["blockNumber"], int)
            else int(log_entry["blockNumber"], 16)
        )

        tx_hash_raw = log_entry.get("transactionHash", b"")
        tx_hash = "0x" + tx_hash_raw.hex() if isinstance(tx_hash_raw, bytes) else str(tx_hash_raw)

        latency_ms = int((time.monotonic() - poll_start) * 1000)

        feed = PhantomPriceFeed(
            pool_address=meta.address,
            pair=meta.pair,
            price=price,
            sqrt_price_x96=sqrt_price_x96,
            timestamp=datetime.now(UTC),
            block_number=block_number,
            tx_hash=tx_hash,
            source="phantom_liquidity",
            latency_ms=latency_ms,
        )

        # Notify all listeners
        for listener in self._listeners:
            try:
                await listener(feed)
            except Exception as exc:
                self._logger.debug(
                    "swap_listener_callback_error", error=str(exc),
                )

    # ── Health ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Return listener health metrics."""
        return {
            "running": self._running,
            "pools_monitored": len(self._pools),
            "polls": self._polls,
            "events_processed": self._events_processed,
            "errors": self._errors,
            "last_block": self._last_block,
        }
