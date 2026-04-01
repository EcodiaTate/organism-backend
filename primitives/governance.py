"""
EcodiaOS - Governance Primitives

Records of governance decisions, amendment proposals, and votes.

Includes Identity provisioning request types (Spec 23, HIGH #4):
  ProvisioningRequest - submitted to Equor before issuing a birth certificate
  CertificateRenewalRequest - submitted to Equor before CA renewal (Citizenship Tax)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, new_id, utc_now


class ProvisioningRequest(EOSBaseModel):
    """
    Governance gate primitive for spawning a new EOS instance.

    Submitted as an Intent to Equor before CertificateManager issues a birth
    certificate or GenesisCA issues an official certificate. Equor evaluates
    constitutional alignment (Growth drive, Care constraints) and either
    PERMITs immediately or ESCALATEs to HITL approval via SMS.

    On HITL approval, Equor emits EQUOR_HITL_APPROVED with
    approval_type="instance_provisioning" and subject_instance_id set.
    """

    request_id: str = Field(default_factory=new_id)
    child_instance_id: str
    parent_instance_id: str
    lineage_hash: str
    niche_definition: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = ""        # System or operator that triggered spawn
    requested_at: datetime = Field(default_factory=utc_now)
    # Equor verdict after constitutional review
    verdict: str = ""             # "PERMIT" | "ESCALATE" | "DENY"
    verdict_reason: str = ""
    approved_at: datetime | None = None


class CertificateRenewalRequest(EOSBaseModel):
    """
    Governance gate primitive for renewing a CA certificate (Citizenship Tax).

    Submitted to Equor before WalletClient processes the Citizenship Tax payment.
    Equor checks economic sufficiency (Oikos balance) and constitutional alignment.
    On PERMIT, renewal proceeds immediately. On ESCALATE, human approval required.

    On HITL approval, Equor emits EQUOR_HITL_APPROVED with
    approval_type="certificate_renewal" and subject_instance_id set.
    """

    request_id: str = Field(default_factory=new_id)
    instance_id: str
    expiring_certificate_id: str
    renewal_count: int = 0
    citizenship_tax_usdc: float = 0.0   # Expected cost for the renewal
    requested_at: datetime = Field(default_factory=utc_now)
    verdict: str = ""
    verdict_reason: str = ""
    approved_at: datetime | None = None


class EquorProvisioningApproval(EOSBaseModel):
    """
    Equor's response to CERTIFICATE_PROVISIONING_REQUEST.

    Carries the result of constitutional drive alignment review for a child
    instance about to receive a birth certificate. CertificateManager awaits
    this before proceeding to GenesisCA issuance.

    Fields:
        child_id           - the child instance under review
        approved           - True if drives are constitutionally aligned
        requires_hitl      - True if a human operator must confirm issuance
        required_amendments - list of amendment IDs Equor wants applied first
        constitutional_hash - SHA-256 of the constitution at time of review
        reason             - human-readable verdict rationale
    """

    child_id: str
    approved: bool
    requires_hitl: bool = False
    required_amendments: list[str] = Field(default_factory=list)
    constitutional_hash: str = ""
    reason: str = ""


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
