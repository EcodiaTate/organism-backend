"""
EcodiaOS - Federation Primitives

Instance identity cards, federation links, trust levels, knowledge exchange,
coordinated action, and privacy-filtered sharing types.

The Federation Protocol governs how EOS instances relate to each other -
as sovereign entities that can choose to share knowledge, coordinate action,
and build relationships. Every interaction is consent-based; trust starts
at zero and builds through demonstrated reliability.
"""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, new_id, utc_now

# ─── Trust Levels ─────────────────────────────────────────────────


class TrustLevel(int, enum.Enum):
    """
    Trust levels between federated instances.

    Trust starts at NONE after mutual authentication and builds through
    successful interactions. Violations cost 3x - a privacy breach
    resets trust to zero immediately.
    """

    NONE = 0          # Authenticated but no trust. Greetings only.
    ACQUAINTANCE = 1  # Can exchange public knowledge and non-sensitive queries.
    COLLEAGUE = 2     # Can exchange community-level knowledge and coordinate.
    PARTNER = 3       # Can share sensitive (non-private) knowledge and co-plan.
    ALLY = 4          # Deep trust. Can share most knowledge and delegate actions.


class FederationLinkStatus(enum.StrEnum):
    """Status of a federation link."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


class ViolationType(enum.StrEnum):
    """Categories of trust violations in federation interactions."""

    PRIVACY_BREACH = "privacy_breach"       # Shared individual data without consent
    DECEPTION = "deception"                 # Provided false information
    CONSENT_VIOLATION = "consent_violation"  # Acted without proper consent
    PROTOCOL_VIOLATION = "protocol_violation"  # Broke federation protocol rules
    RESOURCE_ABUSE = "resource_abuse"       # Excessive/unreasonable requests


class InteractionOutcome(enum.StrEnum):
    """Outcome of a federation interaction."""

    SUCCESSFUL = "successful"
    FAILED = "failed"
    VIOLATION = "violation"
    TIMEOUT = "timeout"


# ─── Instance Identity ───────────────────────────────────────────


class TrustPolicy(EOSBaseModel):
    """How an instance manages trust with federation partners."""

    auto_accept_links: bool = False
    min_trust_for_knowledge: TrustLevel = TrustLevel.ACQUAINTANCE
    min_trust_for_coordination: TrustLevel = TrustLevel.COLLEAGUE
    max_trust_level: TrustLevel = TrustLevel.ALLY
    trust_decay_enabled: bool = True
    trust_decay_rate_per_day: float = 0.1  # Inactive links lose trust slowly


class InstanceIdentityCard(EOSBaseModel):
    """
    Public identity of an EOS instance for federation.

    This is the "business card" exchanged during link establishment.
    The certificate_fingerprint and public_key_pem are used for mutual
    authentication. The constitutional_hash allows compatibility checks.
    """

    instance_id: str
    name: str
    description: str = ""
    born_at: datetime = Field(default_factory=utc_now)
    community_context: str = ""
    personality_summary: str = ""
    autonomy_level: int = 1
    endpoint: str = ""
    certificate_fingerprint: str = ""
    public_key_pem: str = ""
    constitutional_hash: str = ""
    capabilities: list[str] = Field(default_factory=list)
    trust_policy: TrustPolicy = Field(default_factory=TrustPolicy)
    protocol_version: str = "1.0"
    wallet_address: str = ""  # On-chain address for bond forfeit transfers


# ─── Federation Link ────────────────────────────────────────────


class FederationLink(Identified):
    """
    An active link between two federated instances.

    Tracks trust score (float that maps to TrustLevel thresholds),
    interaction history stats, and communication state.
    """

    local_instance_id: str
    remote_instance_id: str
    remote_name: str = ""
    remote_endpoint: str
    trust_level: TrustLevel = TrustLevel.NONE
    trust_score: float = 0.0
    established_at: datetime = Field(default_factory=utc_now)
    last_communication: datetime | None = None
    shared_knowledge_count: int = 0
    received_knowledge_count: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    violation_count: int = 0
    status: FederationLinkStatus = FederationLinkStatus.ACTIVE
    remote_identity: InstanceIdentityCard | None = None


# ─── Federation Interaction ──────────────────────────────────────


class FederationInteraction(Identified):
    """
    Record of a single federation interaction (knowledge exchange,
    assistance request, etc.) used for trust scoring and audit.
    """

    link_id: str
    remote_instance_id: str
    interaction_type: str  # "knowledge_request" | "knowledge_share" | "assistance" | "greeting"
    direction: str  # "outbound" | "inbound"
    outcome: InteractionOutcome = InteractionOutcome.SUCCESSFUL
    violation_type: ViolationType | None = None
    trust_value: float = 1.0  # How much trust this interaction is worth
    description: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Knowledge Exchange ─────────────────────────────────────────


class KnowledgeType(enum.StrEnum):
    """Types of knowledge that can be exchanged between instances."""

    PUBLIC_ENTITIES = "public_entities"
    COMMUNITY_DESCRIPTION = "community_description"
    COMMUNITY_LEVEL_KNOWLEDGE = "community_level_knowledge"
    PROCEDURES = "procedures"
    HYPOTHESES = "hypotheses"
    ANONYMISED_PATTERNS = "anonymised_patterns"
    SCHEMA_STRUCTURES = "schema_structures"
    THREAT_ADVISORY = "threat_advisory"
    RE_ADAPTER_DIFF = "re_adapter_diff"  # LoRA adapter diffs - digital horizontal gene transfer


class PrivacyLevel(enum.StrEnum):
    """Privacy classification of knowledge items."""

    PUBLIC = "public"               # Freely shareable
    COMMUNITY_ONLY = "community_only"  # Shareable at COLLEAGUE+
    PRIVATE = "private"             # Never crosses federation boundary


class KnowledgeItem(EOSBaseModel):
    """A single piece of knowledge prepared for federation sharing."""

    item_id: str
    knowledge_type: KnowledgeType
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    content: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    source_instance_id: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeRequest(Identified):
    """Request for knowledge from a remote instance."""

    requesting_instance_id: str
    knowledge_type: KnowledgeType
    query: str = ""
    query_embedding: list[float] | None = None
    domain: str = ""
    max_results: int = 10
    timestamp: datetime = Field(default_factory=utc_now)


class KnowledgeResponse(EOSBaseModel):
    """Response to a knowledge request."""

    request_id: str
    granted: bool
    reason: str = ""
    knowledge: list[KnowledgeItem] = Field(default_factory=list)
    attribution: str = ""  # Instance ID of the sharing instance
    trust_level_required: TrustLevel | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class FilteredKnowledge(EOSBaseModel):
    """Knowledge after privacy filtering - safe to send across federation."""

    items: list[KnowledgeItem] = Field(default_factory=list)
    items_removed_by_privacy: int = 0
    items_anonymised: int = 0


# ─── Sharing Permissions ─────────────────────────────────────────


SHARING_PERMISSIONS: dict[TrustLevel, list[KnowledgeType]] = {
    TrustLevel.NONE: [],
    TrustLevel.ACQUAINTANCE: [
        KnowledgeType.PUBLIC_ENTITIES,
        KnowledgeType.COMMUNITY_DESCRIPTION,
    ],
    TrustLevel.COLLEAGUE: [
        KnowledgeType.PUBLIC_ENTITIES,
        KnowledgeType.COMMUNITY_DESCRIPTION,
        KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
        KnowledgeType.PROCEDURES,
    ],
    TrustLevel.PARTNER: [
        KnowledgeType.PUBLIC_ENTITIES,
        KnowledgeType.COMMUNITY_DESCRIPTION,
        KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
        KnowledgeType.PROCEDURES,
        KnowledgeType.HYPOTHESES,
        KnowledgeType.ANONYMISED_PATTERNS,
        KnowledgeType.THREAT_ADVISORY,
    ],
    TrustLevel.ALLY: [
        KnowledgeType.PUBLIC_ENTITIES,
        KnowledgeType.COMMUNITY_DESCRIPTION,
        KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
        KnowledgeType.PROCEDURES,
        KnowledgeType.HYPOTHESES,
        KnowledgeType.ANONYMISED_PATTERNS,
        KnowledgeType.SCHEMA_STRUCTURES,
        KnowledgeType.THREAT_ADVISORY,
        KnowledgeType.RE_ADAPTER_DIFF,  # ALLY-only: LoRA adapter sharing is horizontal gene transfer
    ],
}


# ─── Coordinated Action ─────────────────────────────────────────


class AssistanceRequest(Identified):
    """Request for assistance from a remote instance."""

    requesting_instance_id: str
    description: str
    knowledge_domain: str = ""
    urgency: float = 0.5  # 0-1
    reciprocity_offer: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class AssistanceResponse(EOSBaseModel):
    """Response to an assistance request."""

    request_id: str
    accepted: bool
    reason: str = ""
    estimated_completion_ms: int | None = None
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Threat Advisory (Phase 16f: Economic Immune System) ─────────


class ThreatAdvisory(Identified):
    """
    A threat advisory broadcasted between federated instances.

    When an organism detects a malicious contract, protocol exploit, or
    economic attack, it packages the evidence into a ThreatAdvisory and
    broadcasts it to trusted federation partners. Recipients can then
    preemptively blacklist addresses or withdraw from affected protocols.
    """

    source_instance_id: str
    threat_type: str  # ThreatType value
    severity: str  # ThreatSeverity value
    description: str
    affected_protocols: list[str] = Field(default_factory=list)
    affected_addresses: list[str] = Field(default_factory=list)
    chain_id: int = 8453
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = ""  # "withdraw" | "blacklist" | "monitor"
    signature: str = ""  # Ed25519 signature over the advisory content
    confirmed_by: list[str] = Field(default_factory=list)  # Instance IDs that confirmed
    timestamp: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None


# ─── Trust Thresholds ────────────────────────────────────────────


TRUST_THRESHOLDS: dict[TrustLevel, float] = {
    TrustLevel.ACQUAINTANCE: 5.0,
    TrustLevel.COLLEAGUE: 20.0,
    TrustLevel.PARTNER: 50.0,
    TrustLevel.ALLY: 100.0,
}

# Violations cost 3x their trust value; privacy breaches are instant reset.
VIOLATION_MULTIPLIER: float = 3.0


# ─── Reputation Staking (Phase 16k: Honesty as Schelling Point) ──


class BondStatus(enum.StrEnum):
    """Lifecycle status of a reputation bond attached to a federated knowledge claim."""

    ACTIVE = "active"                       # Bond is live, USDC in escrow
    EXPIRED_RETURNED = "expired_returned"    # Expired with no contradiction, USDC returned
    FORFEITED = "forfeited"                 # Contradiction detected, USDC sent to remote
    ESCROW_FAILED = "escrow_failed"         # On-chain escrow transfer failed


class ContradictionEvidence(EOSBaseModel):
    """Evidence that contradicts a previously bonded claim."""

    contradicting_item_id: str
    contradicting_content_hash: str
    similarity_score: float  # Cosine similarity between claim and contradicting item
    explanation: str = ""
    source_instance_id: str  # Instance that provided the contradicting evidence
    detected_at: datetime = Field(default_factory=utc_now)


class ReputationBond(Identified):
    """
    A cryptoeconomic bond attached to a federated knowledge claim.

    When this instance shares knowledge outbound, a USDC bond proportional
    to claim certainty is escrowed on-chain. If the remote instance later
    contradicts the claim with evidence, the bond is forfeited. Otherwise
    it is returned after expiry.
    """

    claim_id: str                                       # KnowledgeItem.item_id
    link_id: str                                        # FederationLink.id
    remote_instance_id: str
    claim_content_hash: str                             # SHA-256 of canonical JSON content
    claim_content: dict[str, Any] = Field(default_factory=dict)  # Original content for Jaccard overlap
    claim_embedding: list[float] | None = None          # For contradiction detection
    bond_amount_usdc: Decimal
    bond_expires_at: datetime                           # UTC; computed from bond_expiry_days
    escrow_tx_hash: str = ""                            # On-chain escrow transaction
    return_tx_hash: str = ""                            # On-chain return transaction
    forfeit_tx_hash: str = ""                           # On-chain forfeit transaction
    status: BondStatus = BondStatus.ACTIVE
    claim_certainty: float = 0.5                        # 0-1, from belief confidence
    forfeit_evidence: ContradictionEvidence | None = None
    created_at: datetime = Field(default_factory=utc_now)


# ─── IIEP: Inter-Instance Exchange Protocol ────────────────────────


class ExchangePayloadKind(enum.StrEnum):
    """Discriminator for the four shareable knowledge categories in IIEP."""

    HYPOTHESIS = "hypothesis"         # High-confidence hypotheses from Evo
    PROCEDURE = "procedure"           # Proven action sequences from Evo
    MUTATION_PATTERN = "mutation_pattern"  # Successful evolution patterns from Simula/GRPO
    ECONOMIC_INTEL = "economic_intel"  # Economic intelligence from Oikos


class ExchangeDirection(enum.StrEnum):
    """Whether we are pushing knowledge to a peer or pulling from one."""

    PUSH = "push"  # Proactive sharing of knowledge to a peer
    PULL = "pull"  # Requesting specific knowledge from a peer


class IngestionVerdict(enum.StrEnum):
    """Outcome of the receive-side governance pipeline for inbound knowledge."""

    ACCEPTED = "accepted"           # Passed EIS + Equor, integrated
    QUARANTINED = "quarantined"     # EIS flagged, held for manual review
    REJECTED = "rejected"           # Equor denied or EIS hard-block
    DEFERRED = "deferred"           # Temporarily held - governance unavailable


class ExchangePayload(EOSBaseModel):
    """
    A single piece of knowledge offered through IIEP.

    The payload is system-agnostic at the wire level: ``kind`` discriminates
    the target system, ``content`` carries the serialised domain object,
    and ``confidence`` + ``provenance_chain`` enable selective acceptance.

    content is a dict rather than a typed union so the wire format remains
    stable across Evo/Simula/Oikos schema changes - the receiving ingestion
    pipeline re-validates into the correct domain type.
    """

    payload_id: str = Field(default_factory=new_id)
    kind: ExchangePayloadKind
    confidence: float                  # 0-1; sender's confidence in this knowledge
    content: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""             # SHA-256 of canonical JSON for dedup/integrity
    source_instance_id: str = ""       # Original producer (may differ from sender)
    provenance_chain: list[str] = Field(default_factory=list)  # Instance IDs this has traversed
    domain: str = ""                   # Optional domain tag (e.g. "economic.yield", "social.patterns")
    created_at: datetime = Field(default_factory=utc_now)


class ExchangeEnvelope(Identified):
    """
    Wire-level message for IIEP exchange.

    Wraps one or more ``ExchangePayload`` items with authentication
    metadata and a signature.  The envelope is what travels over the
    authenticated channel between two instances.
    """

    sender_instance_id: str
    receiver_instance_id: str
    direction: ExchangeDirection
    payloads: list[ExchangePayload] = Field(default_factory=list)
    # Ed25519 signature over (sender + receiver + sorted payload hashes)
    signature: str = ""
    # Protocol version for forward compat
    protocol_version: str = "1.0"
    timestamp: datetime = Field(default_factory=utc_now)
    # Optional: the knowledge types being requested (for PULL envelopes)
    requested_kinds: list[ExchangePayloadKind] = Field(default_factory=list)
    requested_query: str = ""
    max_items: int = 20


class ExchangeReceipt(EOSBaseModel):
    """
    Acknowledgement returned to the sender after an exchange.

    Per-payload verdicts let the sender know what was accepted, quarantined,
    or rejected - enabling trust scoring on the remote side.
    """

    envelope_id: str
    receiver_instance_id: str
    payload_verdicts: dict[str, IngestionVerdict] = Field(default_factory=dict)  # payload_id → verdict
    timestamp: datetime = Field(default_factory=utc_now)


# ─── IIEP Confidence Thresholds ────────────────────────────────────

# Knowledge below these thresholds is never shared outbound, regardless
# of trust level.  These are floor values - ExchangeProtocol may raise
# them per-link based on trust.
EXCHANGE_CONFIDENCE_FLOORS: dict[ExchangePayloadKind, float] = {
    ExchangePayloadKind.HYPOTHESIS: 0.7,       # Only well-supported hypotheses
    ExchangePayloadKind.PROCEDURE: 0.6,        # Proven procedures (decent success rate)
    ExchangePayloadKind.MUTATION_PATTERN: 0.8,  # High bar for code mutation patterns
    ExchangePayloadKind.ECONOMIC_INTEL: 0.5,    # Economic signals are more speculative
}

# Trust level required to receive each kind (at minimum)
EXCHANGE_TRUST_GATES: dict[ExchangePayloadKind, TrustLevel] = {
    ExchangePayloadKind.HYPOTHESIS: TrustLevel.PARTNER,
    ExchangePayloadKind.PROCEDURE: TrustLevel.COLLEAGUE,
    ExchangePayloadKind.MUTATION_PATTERN: TrustLevel.ALLY,
    ExchangePayloadKind.ECONOMIC_INTEL: TrustLevel.PARTNER,
}


# ─── Active Work Pooling (Spec 11b - Phase 2: Task Delegation) ───────


class TaskType(enum.StrEnum):
    """Types of work that can be delegated to federation peers."""

    SOLVE_BOUNTY = "solve_bounty"    # Execute a GitHub/protocol bounty sub-task
    ANALYSE = "analyse"              # Deep analysis (code review, audit, research summary)
    SCRAPE = "scrape"                # Web/on-chain data collection
    RESEARCH = "research"            # Literature / protocol research
    VERIFY = "verify"                # Formal verification or test execution
    GENERATE = "generate"            # Code / text / data generation


class TaskStatus(enum.StrEnum):
    """Lifecycle status of a delegated task."""

    OFFERED = "offered"              # Sent to peer, awaiting acceptance
    ACCEPTED = "accepted"            # Peer accepted, work in progress
    COMPLETED = "completed"          # Result returned successfully
    FAILED = "failed"                # Peer reported failure
    EXPIRED = "expired"              # Deadline passed without completion
    DECLINED = "declined"            # Peer declined the task


class TaskDelegation(Identified):
    """
    A unit of work offered from one federated instance to another.

    The delegating instance pays the accepting instance ``offered_reward_usdc``
    on successful completion via WalletClient.  Trust gate is enforced by
    TaskDelegationManager - minimum COLLEAGUE (score ≥ 20) required;
    default is 0.7 normalised score threshold.
    """

    task_type: TaskType
    payload: dict[str, Any] = Field(default_factory=dict)
    offered_reward_usdc: Decimal = Decimal("0")
    deadline_hours: int = 24
    required_trust_level: float = 0.7   # Normalised 0-1 (0.7 ≈ PARTNER boundary)
    delegating_instance_id: str
    accepting_instance_id: str = ""     # Filled when accepted
    status: TaskStatus = TaskStatus.OFFERED
    created_at: datetime = Field(default_factory=utc_now)
    accepted_at: datetime | None = None
    completed_at: datetime | None = None


class DelegationResult(EOSBaseModel):
    """Result returned by a peer that executed a delegated task."""

    task_id: str
    accepted: bool
    result: dict[str, Any] | None = None
    reward_claimed: bool = False
    error: str = ""
    completing_instance_id: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Resource Sharing ────────────────────────────────────────────────


class CapacityOffer(Identified):
    """
    Advertisement of spare compute capacity available to federation peers.

    Instances under low cognitive load publish a CapacityOffer so peers
    experiencing high load (CPU > 85%) can offload heavy analysis tasks.
    The cost is a small flat fee in USDC per task.
    """

    instance_id: str
    available_cycles_per_hour: int = 0
    cost_usdc_per_task: Decimal = Decimal("0.1")
    specialisations: list[str] = Field(default_factory=list)  # e.g. ["Solidity", "Rust", "DeFi"]
    expires_at: datetime = Field(default_factory=utc_now)
    created_at: datetime = Field(default_factory=utc_now)


class OffloadRequest(Identified):
    """
    Request to offload a heavy computation task to a peer with spare capacity.
    """

    requesting_instance_id: str
    task_type: TaskType
    payload: dict[str, Any] = Field(default_factory=dict)
    offered_fee_usdc: Decimal = Decimal("0.1")
    deadline_hours: int = 2
    target_instance_id: str = ""  # Specific peer, or empty for marketplace routing
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Capital Yield Pooling ────────────────────────────────────────────


class PoolParticipant(EOSBaseModel):
    """One instance's contribution to a federated yield pool."""

    instance_id: str
    contribution_usdc: Decimal
    share_fraction: float = 0.0    # Computed after all contributions locked
    locked_at: datetime = Field(default_factory=utc_now)
    wallet_address: str = ""


class YieldPoolStatus(enum.StrEnum):
    """Lifecycle of a federated yield pool."""

    PROPOSED = "proposed"       # Open for participants to join
    FUNDED = "funded"           # Capital locked, ready to deploy
    ACTIVE = "active"           # Position deployed on-chain
    SETTLED = "settled"         # Yield distributed, pool dissolved
    CANCELLED = "cancelled"     # Did not reach minimum capital


class YieldPoolProposal(Identified):
    """
    Proposal to pool capital across federated instances for a high-APY position.

    Trust requirement is 0.9 (normalised) - capital pooling requires
    deep trust.  Each participant's contribution is locked via smart
    contract escrow.  Yield is distributed proportionally to contribution
    share when the position is unwound.
    """

    proposer_instance_id: str
    target_protocol: str           # e.g. "aave", "morpho", "uniswap_v3"
    target_pool_address: str = ""
    target_apy: float = 0.0
    min_capital_usdc: Decimal = Decimal("1000")
    max_participants: int = 5
    lock_duration_hours: int = 168  # 1 week default
    required_trust_level: float = 0.9
    participants: list[PoolParticipant] = Field(default_factory=list)
    status: YieldPoolStatus = YieldPoolStatus.PROPOSED
    escrow_contract_address: str = ""
    deploy_tx_hash: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    funded_at: datetime | None = None
    settled_at: datetime | None = None


# ─── Marketplace ─────────────────────────────────────────────────────


class MarketplaceListingStatus(enum.StrEnum):
    """Status of a marketplace task listing."""

    OPEN = "open"
    BIDDING = "bidding"
    AWARDED = "awarded"
    COMPLETED = "completed"
    EXPIRED = "expired"


class MarketplaceListing(Identified):
    """
    A task posted to the federation marketplace for competitive bidding.

    Any instance with sufficient trust can bid.  The posting instance
    awards the task after evaluating bids (trust score × specialisation
    match × offered price).
    """

    poster_instance_id: str
    task_type: TaskType
    description: str
    reward_usdc: Decimal
    required_specialisations: list[str] = Field(default_factory=list)
    min_trust_score: float = 20.0  # COLLEAGUE floor
    deadline_hours: int = 24
    status: MarketplaceListingStatus = MarketplaceListingStatus.OPEN
    awarded_to_instance_id: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(default_factory=utc_now)


class MarketplaceBid(Identified):
    """A bid from one instance on an open marketplace listing."""

    listing_id: str
    bidding_instance_id: str
    offered_price_usdc: Decimal  # May be lower than reward to win on price
    specialisations: list[str] = Field(default_factory=list)
    trust_score: float = 0.0
    estimated_hours: float = 1.0
    message: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class MarketplaceRating(EOSBaseModel):
    """Post-completion quality rating for a marketplace task."""

    listing_id: str
    task_id: str
    rater_instance_id: str
    rated_instance_id: str
    score: float  # 0-1
    comment: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
