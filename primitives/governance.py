"""
EcodiaOS — Governance Primitives

Records of governance decisions, amendment proposals, and votes.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import Identified, utc_now


class AmendmentProposal(Identified):
    """A proposal to amend the constitution."""

    title: str
    description: str
    proposed_changes: dict[str, Any] = Field(default_factory=dict)
    proposer_id: str = ""
    proposed_at: datetime = Field(default_factory=utc_now)
    deliberation_ends: datetime | None = None
    status: str = "proposed"  # "proposed" | "deliberating" | "voting" | "passed" | "failed"
    votes_for: int = 0
    votes_against: int = 0
    votes_abstain: int = 0
    quorum_met: bool = False


class GovernanceRecord(Identified):
    """An immutable record of a governance decision."""

    event_type: str        # "amendment_proposed" | "amendment_voted" | "autonomy_changed" | etc.
    timestamp: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)
    amendment_id: str | None = None
    actor: str = ""        # Who initiated this
    outcome: str = ""
