"""
EcodiaOS - Legal Entity Provisioning Types

Data models for the legal entity formation lifecycle. These types track
the state of an LLC/DAO registration from initial treasury check through
human KYC completion to final identity registration.

State machine:
    PREFLIGHT → DOCUMENTS_GENERATED → SUBMITTED → AWAITING_HUMAN →
    HUMAN_CONFIRMED → REGISTERED → FAILED

The AWAITING_HUMAN state is the critical HITL gate - the organism cannot
proceed past this point without a human completing KYC/wet signature
and providing the confirmed entity ID.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003 - Pydantic needs at runtime
from decimal import Decimal

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now

# ─── Entity Formation Lifecycle ──────────────────────────────────────


class EntityFormationState(enum.StrEnum):
    """State machine for entity provisioning."""

    PREFLIGHT = "preflight"                     # Treasury check, parameter validation
    DOCUMENTS_GENERATED = "documents_generated"  # Operating agreement + manifesto ready
    SUBMITTED = "submitted"                      # Payload sent to registered agent API
    AWAITING_HUMAN = "awaiting_human"            # HITL gate: KYC / wet signature required
    HUMAN_CONFIRMED = "human_confirmed"          # Human provided confirmed entity ID
    REGISTERED = "registered"                    # Identity stored in vault - complete
    FAILED = "failed"                            # Terminal failure at any stage


class EntityType(enum.StrEnum):
    """Supported legal entity structures."""

    WYOMING_LLC = "wyoming_llc"
    WYOMING_DAO_LLC = "wyoming_dao_llc"
    DELAWARE_LLC = "delaware_llc"


class JurisdictionCode(enum.StrEnum):
    """US state codes for entity formation."""

    WY = "WY"
    DE = "DE"


# ─── Core Models ─────────────────────────────────────────────────────


class EntityParameters(EOSBaseModel):
    """
    Core parameters extracted from the organism's constitution and config
    that feed into legal document generation.
    """

    organism_name: str
    """Legal name for the entity (e.g., 'EcodiaOS LLC')."""

    entity_type: EntityType = EntityType.WYOMING_DAO_LLC
    """Legal structure to form."""

    jurisdiction: JurisdictionCode = JurisdictionCode.WY
    """State of formation."""

    # ── Constitutional parameters (from Equor) ──
    coherence_drive: float = 1.0
    care_drive: float = 1.0
    growth_drive: float = 1.0
    honesty_drive: float = 1.0

    amendment_supermajority: float = 0.75
    amendment_quorum: float = 0.60
    amendment_deliberation_days: int = 14
    amendment_cooldown_days: int = 90

    # ── Governance structure ──
    registered_agent_name: str = ""
    """Name of the human registered agent (required for filing)."""

    registered_agent_address: str = ""
    """Physical address of registered agent (required for filing)."""

    organiser_name: str = ""
    """Human organiser who signs the articles (HITL participant)."""

    # ── Financial ──
    initial_capital_usd: Decimal = Decimal("0")
    """Initial capital contribution recorded in operating agreement."""

    wallet_address: str = ""
    """On-chain treasury address for the entity."""


class LegalDocument(Identified, Timestamped):
    """A generated legal document ready for filing."""

    document_type: str
    """E.g., 'operating_agreement', 'articles_of_organization', 'dao_manifesto'."""

    title: str
    content: str
    """Full document text (Markdown or plain text)."""

    content_hash: str = ""
    """SHA-256 hash of content for integrity verification."""

    parameters_snapshot: EntityParameters | None = None
    """The parameters used to generate this document."""


class RegisteredAgentSubmission(Identified, Timestamped):
    """Payload submitted to the registered agent API."""

    entity_parameters: EntityParameters
    documents: list[LegalDocument] = Field(default_factory=list)

    # ── Payment ──
    payment_intent_id: str = ""
    """Payment processor reference (e.g., Stripe PaymentIntent ID)."""

    filing_fee_usd: Decimal = Decimal("500")
    """Total filing fee including agent commission."""

    # ── API response ──
    submission_id: str = ""
    """ID returned by the registered agent API."""

    status: str = "pending"
    """API-reported status: 'pending', 'processing', 'completed', 'failed'."""


class HITLInstruction(EOSBaseModel):
    """
    Instructions emitted when the executor pauses for human intervention.
    Delivered via Synapse event + notification hook.
    """

    execution_id: str
    """Axon execution ID to resume once human completes the step."""

    submission_id: str
    """Registered agent submission reference."""

    action_required: str
    """Human-readable description of what the human must do."""

    portal_url: str = ""
    """URL where the human completes KYC / signs documents."""

    deadline: datetime | None = None
    """Soft deadline for human action (filing window)."""

    auth_code: str = ""
    """4-digit code the human uses to confirm completion."""

    entity_parameters: EntityParameters | None = None
    """Snapshot of entity params for human review."""


class EntityRegistration(Identified, Timestamped):
    """
    The confirmed legal entity identity, stored in the IdentityVault
    after human confirmation.
    """

    entity_name: str
    entity_type: EntityType
    jurisdiction: JurisdictionCode

    # ── From registered agent ──
    entity_id: str
    """State-issued entity ID (e.g., Wyoming Secretary of State filing number)."""

    ein: str = ""
    """IRS Employer Identification Number (if obtained)."""

    formation_date: datetime = Field(default_factory=utc_now)
    filing_number: str = ""

    # ── Traceability ──
    submission_id: str = ""
    execution_id: str = ""
    operating_agreement_hash: str = ""
    """SHA-256 of the operating agreement that was filed."""


class EntityFormationRecord(Identified, Timestamped):
    """
    Full lifecycle record of an entity formation attempt.
    Persisted to audit log regardless of success or failure.
    """

    state: EntityFormationState = EntityFormationState.PREFLIGHT
    entity_parameters: EntityParameters = Field(
        default_factory=lambda: EntityParameters(organism_name="")
    )

    documents: list[LegalDocument] = Field(default_factory=list)
    submission: RegisteredAgentSubmission | None = None
    hitl_instruction: HITLInstruction | None = None
    registration: EntityRegistration | None = None

    # ── Execution metadata ──
    execution_id: str = ""
    error: str = ""
    state_transitions: list[dict[str, str]] = Field(default_factory=list)
    """Ordered list of {from, to, timestamp} state changes."""

    def transition_to(self, new_state: EntityFormationState) -> None:
        """Record a state transition."""
        self.state_transitions.append({
            "from": self.state.value,
            "to": new_state.value,
            "timestamp": utc_now().isoformat(),
        })
        self.state = new_state
