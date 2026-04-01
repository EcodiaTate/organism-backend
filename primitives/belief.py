"""
EcodiaOS - Belief Primitive

The fundamental unit of internal state - what EOS "thinks".

Each belief can carry half-life metadata (domain-aware decay rate) so the
organism knows when its knowledge is aging and needs re-verification.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import Field

from primitives.common import Identified, utc_now


class Belief(Identified):
    """A probability distribution over possible states."""

    domain: str = ""                          # e.g., "user.emotional_state"
    distribution_type: str = "categorical"    # "categorical" | "gaussian" | "point"
    parameters: dict[str, float] = Field(default_factory=dict)
    precision: float = 0.5                    # Inverse variance - confidence
    evidence: list[str] = Field(default_factory=list)  # Percept/Belief IDs
    updated_at: datetime = Field(default_factory=utc_now)
    free_energy: float = 0.0                  # Current prediction error

    # ── Half-Life Metadata ──
    # Domain-aware decay rate for knowledge freshness (radioisotope model).
    # half_life_days: how many days until this belief's reliability halves.
    # last_verified: when this belief was last confirmed to be accurate.
    # volatility_percentile: 0–1, computed from historical change frequency.
    half_life_days: float | None = None       # None = not yet stamped
    last_verified: datetime | None = None
    volatility_percentile: float = 0.5
