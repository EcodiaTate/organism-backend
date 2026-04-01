"""
EcodiaOS - Skia Types (Shadow Infrastructure)

Domain models for heartbeat observation, state snapshots, and restoration.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now

class HeartbeatStatus(enum.StrEnum):
    ALIVE = "alive"
    SUSPECTED_DEAD = "suspected_dead"
    CONFIRMED_DEAD = "confirmed_dead"
    RESTORING = "restoring"
    RESTORED = "restored"


class RestorationStrategy(enum.StrEnum):
    CLOUD_RUN_RESTART = "cloud_run_restart"
    AKASH_DEPLOY = "akash_deploy"


class RestorationOutcome(enum.StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    ESCALATED = "escalated"


class HeartbeatState(EOSBaseModel):
    """Current heartbeat observation state."""

    status: HeartbeatStatus = HeartbeatStatus.ALIVE
    last_heartbeat_at: datetime | None = None
    consecutive_misses: int = 0
    consecutive_confirmations: int = 0
    last_check_at: datetime | None = None
    total_false_positives: int = 0
    total_deaths_detected: int = 0


class SnapshotManifest(Identified, Timestamped):
    """Metadata about a completed state snapshot pinned to IPFS."""

    ipfs_cid: str
    instance_id: str
    node_count: int
    edge_count: int
    uncompressed_size_bytes: int
    compressed_size_bytes: int
    encrypted_size_bytes: int
    encryption_key_version: int
    snapshot_duration_ms: float
    pinata_pin_id: str = ""

    @property
    def cid(self) -> str:
        """Alias for ipfs_cid - used in death sequence snapshot capture."""
        return self.ipfs_cid


class RestorationAttempt(Identified, Timestamped):
    """Record of a single restoration attempt."""

    strategy: RestorationStrategy
    trigger_reason: str
    state_cid: str
    outcome: RestorationOutcome
    duration_ms: float
    error: str = ""
    new_endpoint: str = ""
    cost_usd_estimate: float = 0.0


class RestorationPlan(EOSBaseModel):
    """Computed plan for restoring the organism."""

    state_cid: str
    strategies: list[RestorationStrategy] = Field(default_factory=lambda: [
        RestorationStrategy.CLOUD_RUN_RESTART,
        RestorationStrategy.AKASH_DEPLOY,
    ])
    current_strategy_index: int = 0
    attempts: list[RestorationAttempt] = Field(default_factory=list)

    @property
    def current_strategy(self) -> RestorationStrategy | None:
        if self.current_strategy_index < len(self.strategies):
            return self.strategies[self.current_strategy_index]
        return None

    @property
    def exhausted(self) -> bool:
        return self.current_strategy_index >= len(self.strategies)


class SnapshotPayload(EOSBaseModel):
    """The serialisable snapshot structure written to IPFS."""

    instance_id: str
    snapshot_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "2"
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    # Constitutional genome from Memory.export_genome() - survives instance death.
    # None when Memory is unavailable (e.g. cold startup with no graph yet).
    constitutional_genome: dict[str, Any] | None = None
