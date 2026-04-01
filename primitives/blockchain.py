"""
EcodiaOS - Blockchain Shared Primitives

Types shared across systems that deal with on-chain activity.
Currently: block competition monitoring (used by Fovea + Axon MEV analysis).
"""

from __future__ import annotations

from primitives.common import EOSBaseModel


class BlockCompetitionSnapshot(EOSBaseModel):
    """Point-in-time snapshot of block space competition."""

    gas_price_gwei: float = 0.0
    base_fee_gwei: float = 0.0
    pending_tx_count: int = 0
    block_utilization_pct: float = 0.0  # 0-100
    competition_level: float = 0.0  # 0=low, 1=high (normalised)
    timestamp_ms: int = 0

    @property
    def is_low_competition(self) -> bool:
        return self.competition_level < 0.3

    @property
    def is_high_competition(self) -> bool:
        return self.competition_level > 0.7
