"""
EcodiaOS — Block Competition Monitor (Atune subsystem)

Monitors on-chain block space competition for MEV-aware transaction timing.
Runs as an async background task within Atune's perception pipeline, sampling
gas prices, pending transaction counts, and block utilization at regular
intervals.

The monitor publishes BlockCompetitionSnapshot updates to any registered
listeners (primarily MEVAnalyzer in Axon). This enables adaptive transaction
timing: submit during low-competition windows to reduce MEV extraction risk.

Design:
  - Non-blocking: runs on its own asyncio task, never stalls the main loop
  - Graceful degradation: if the RPC is unavailable, emits stale snapshots
  - Rate-limited: polls at configurable intervals (default 12s ≈ 1 Base block)
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.mev_types import BlockCompetitionSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from web3 import AsyncWeb3

logger = structlog.get_logger()

# Default polling interval (seconds) — roughly 1 Base L2 block time
_DEFAULT_POLL_INTERVAL_S: float = 12.0

# Gas price percentiles for competition scoring
_LOW_GAS_GWEI: float = 0.01    # Base L2 baseline (very cheap)
_HIGH_GAS_GWEI: float = 1.0    # Congested Base L2

# Pending tx count thresholds
_LOW_PENDING: int = 50
_HIGH_PENDING: int = 500


class BlockCompetitionMonitor:
    """
    Async background monitor for on-chain block space competition.

    Polls the RPC endpoint for gas prices and pending transaction counts,
    then pushes BlockCompetitionSnapshot updates to registered listeners.

    Usage:
        monitor = BlockCompetitionMonitor(rpc_url="https://mainnet.base.org")
        monitor.add_listener(mev_analyzer.update_competition)
        await monitor.start()
        # ... later ...
        await monitor.stop()
    """

    def __init__(
        self,
        rpc_url: str = "",
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self._rpc_url = rpc_url
        self._poll_interval_s = poll_interval_s
        self._w3: AsyncWeb3 | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._listeners: list[Callable[[BlockCompetitionSnapshot], Any]] = []
        self._latest: BlockCompetitionSnapshot = BlockCompetitionSnapshot()
        self._logger = logger.bind(system="atune", component="block_competition")

        # Metrics
        self._polls: int = 0
        self._errors: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling task."""
        if self._running:
            return

        if not self._rpc_url:
            self._logger.info("block_competition_disabled", reason="no_rpc_url")
            return

        try:
            from web3 import AsyncWeb3
            from web3.providers import AsyncHTTPProvider

            self._w3 = AsyncWeb3(AsyncHTTPProvider(self._rpc_url))
            is_connected = await self._w3.is_connected()

            if not is_connected:
                self._logger.warning("block_competition_rpc_unreachable")
                self._w3 = None
                return
        except Exception as exc:
            self._logger.warning("block_competition_connect_failed", error=str(exc))
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="block_competition_monitor")
        self._logger.info("block_competition_started", poll_interval_s=self._poll_interval_s)

    async def stop(self) -> None:
        """Stop the background polling task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._w3 = None
        self._logger.info("block_competition_stopped", total_polls=self._polls)

    def add_listener(self, callback: Callable[[BlockCompetitionSnapshot], Any]) -> None:
        """Register a listener for competition snapshot updates."""
        self._listeners.append(callback)

    # ── Poll Loop ─────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Background polling loop — runs until stopped."""
        while self._running:
            try:
                snapshot = await self._sample()
                self._latest = snapshot
                self._polls += 1

                # Notify listeners
                for listener in self._listeners:
                    try:
                        listener(snapshot)
                    except Exception as exc:
                        self._logger.debug("block_competition_listener_error", error=str(exc))

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._errors += 1
                self._logger.debug("block_competition_poll_error", error=str(exc))

            await asyncio.sleep(self._poll_interval_s)

    async def _sample(self) -> BlockCompetitionSnapshot:
        """Take a single block competition sample from the RPC."""
        if self._w3 is None:
            return BlockCompetitionSnapshot(timestamp_ms=int(time.time() * 1000))

        # Fetch gas price and latest block in parallel
        gas_price_wei, latest_block = await asyncio.gather(
            self._w3.eth.gas_price,
            self._w3.eth.get_block("latest"),
        )

        gas_price_gwei = float(gas_price_wei) / 1e9

        # Extract base fee from the block header (EIP-1559)
        base_fee_wei = getattr(latest_block, "baseFeePerGas", 0) or 0
        base_fee_gwei = float(base_fee_wei) / 1e9

        # Block utilization: gas_used / gas_limit
        gas_used = int(getattr(latest_block, "gasUsed", 0) or 0)
        gas_limit = int(getattr(latest_block, "gasLimit", 1) or 1)
        block_utilization_pct = (gas_used / gas_limit) * 100 if gas_limit > 0 else 0.0

        # Pending tx count (best-effort — not all nodes support this)
        pending_count = 0
        try:
            pending_block = await self._w3.eth.get_block("pending")
            pending_count = len(getattr(pending_block, "transactions", []))
        except Exception:
            pass  # Not all nodes support pending block queries

        # Compute normalised competition level
        competition_level = self._compute_competition_level(
            gas_price_gwei=gas_price_gwei,
            pending_count=pending_count,
            block_utilization_pct=block_utilization_pct,
        )

        return BlockCompetitionSnapshot(
            gas_price_gwei=gas_price_gwei,
            base_fee_gwei=base_fee_gwei,
            pending_tx_count=pending_count,
            block_utilization_pct=block_utilization_pct,
            competition_level=competition_level,
            timestamp_ms=int(time.time() * 1000),
        )

    @staticmethod
    def _compute_competition_level(
        gas_price_gwei: float,
        pending_count: int,
        block_utilization_pct: float,
    ) -> float:
        """
        Compute a normalised competition level from raw metrics.

        Components (weighted average):
          - Gas price signal (40%): normalised against Base L2 baseline
          - Pending tx count (30%): normalised against known thresholds
          - Block utilization (30%): raw percentage / 100

        Returns a float in [0, 1] where 0 = empty, 1 = highly congested.
        """
        # Gas price component: 0 at baseline, 1 at congestion threshold
        gas_norm = min(1.0, max(0.0,
            (gas_price_gwei - _LOW_GAS_GWEI) / (_HIGH_GAS_GWEI - _LOW_GAS_GWEI)
        ))

        # Pending tx component
        pending_norm = min(1.0, max(0.0,
            (pending_count - _LOW_PENDING) / (_HIGH_PENDING - _LOW_PENDING)
        ))

        # Block utilization component
        util_norm = min(1.0, max(0.0, block_utilization_pct / 100.0))

        return gas_norm * 0.4 + pending_norm * 0.3 + util_norm * 0.3

    # ── Properties ────────────────────────────────────────────────

    @property
    def latest(self) -> BlockCompetitionSnapshot:
        return self._latest

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "total_polls": self._polls,
            "errors": self._errors,
            "latest_competition": self._latest.competition_level,
            "latest_gas_gwei": self._latest.gas_price_gwei,
        }
