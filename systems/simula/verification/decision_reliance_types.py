"""
EcodiaOS -- Simula Phase 3 Decision Reliance Types

Three reliance classes targeting where decisions trust belief over verification --
the attack surface of deferred re-validation:

  1. State Interpretation   -- decisions that depend on stored/cached state
                               without re-validating the underlying assumption
  2. Source-of-Truth Gaps   -- divergence between live truth, stored truth,
                               and inferred truth at the point of a decision
  3. Session Continuity     -- assumptions that must persist across steps for
                               a multi-step workflow to proceed correctly

Each class extends PhaseInvariant with domain-specific evidence fields.
Z3-checkable where the reliance can be encoded as a constraint.
Where not (temporal divergence, multi-step narrative), evidence falls back
to STRUCTURAL_ANALYSIS + RUNTIME_WITNESS.

Phase 3 answers: "What must the decision-maker believe for this action to go
                  through? Is that belief verified, or just remembered?"
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now
from systems.simula.verification.invariant_types import (
    PhaseInvariant,
)

# ── Reliance Classes ──────────────────────────────────────────────────────────


class RelianceClass(enum.StrEnum):
    """Which of the three Phase 3 reliance classes does a finding cover?"""

    STATE_INTERPRETATION = "state_interpretation"
    SOURCE_OF_TRUTH = "source_of_truth"
    SESSION_CONTINUITY = "session_continuity"


class RelianceRisk(enum.StrEnum):
    """
    How dangerous is this reliance if the belief turns out to be wrong?

    CRITICAL  -- wrong belief enables unauthorised action or data corruption
    HIGH      -- wrong belief produces incorrect output used by downstream decisions
    MEDIUM    -- wrong belief degrades correctness but can be detected and corrected
    LOW       -- wrong belief has bounded, recoverable impact
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VerificationStyle(enum.StrEnum):
    """
    How does the code currently resolve this belief?

    RE_VERIFIED   -- re-fetches or re-computes from authoritative source each time
    CACHED        -- reads a cached value with a defined TTL
    ASSUMED_VALID -- trusts the value because it was valid at an earlier point
    INFERRED      -- derives the value from correlated signals, not the source
    UNVERIFIED    -- no mechanism exists to check the belief; taken on faith
    """

    RE_VERIFIED = "re_verified"
    CACHED = "cached"
    ASSUMED_VALID = "assumed_valid"
    INFERRED = "inferred"
    UNVERIFIED = "unverified"


# ── 1. State Interpretation ───────────────────────────────────────────────────


class CachedAuthorityReliance(PhaseInvariant):
    """
    A decision depends on an authority claim that was cached rather than
    re-verified at decision time.

    Concrete mechanisms:
      - Role/permission checks that read from a session token without verifying
        that the token is still valid against the authority store
      - Identity context stored in memory and reused across multiple operations
        without re-authentication
      - Policy evaluation against a snapshot that may be stale

    Risk: if the cached authority is revoked between caching and use, the code
    continues to act on a belief that is no longer true.

    Z3-checkable: yes -- encode as:
        If authority_checked_at + ttl < decision_at then decision is belief-reliant.
    Strength: BEHAVIORAL (requires runtime observation of TTL vs decision gap)
    """

    reliance_class: Literal["cached_authority"] = "cached_authority"
    # What authority claim is being cached?
    authority_claim: str = ""          # e.g. "principal.role == ADMIN"
    # Where is the cache stored?
    cache_location: str = ""           # e.g. "IdentityContext.credential_strength"
    # How long before the cache is considered stale (0 = indefinite = high risk)
    cache_ttl_seconds: int = 0
    # What authoritative source is NOT being consulted?
    authoritative_source: str = ""     # e.g. "principal_registry.get_role(principal_id)"
    # What decision gates on this cached claim?
    gated_decision: str = ""           # e.g. "allow_governance_proposal"
    verification_style: VerificationStyle = VerificationStyle.CACHED
    reliance_risk: RelianceRisk = RelianceRisk.HIGH
    # Concrete mechanism: what re-verification would look like
    remediation: str = ""
    # AST evidence
    observed_at_function: str = ""
    observed_at_line: int = 0
    # Runtime: how many decisions relied on cache vs. re-verified
    cache_hit_decision_count: int = 0
    re_verified_decision_count: int = 0
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)


class InferredIdentityReliance(PhaseInvariant):
    """
    A decision depends on an identity claim that was inferred from correlated
    signals rather than asserted by an authoritative mechanism.

    Concrete mechanisms:
      - Using request.remote_addr as an identity proxy
      - Trusting a header (X-Forwarded-For, X-User-ID) without cryptographic
        signature verification
      - Inferring the acting principal from the name of a calling function
        or module rather than an explicit credential
      - Treating the presence of a session cookie as identity proof rather
        than verifying the session against the backing store

    Risk: inferred identity can be forged, spoofed, or coincidentally correct
    for the wrong entity.

    Z3-checkable: no (identity inference is not a linear constraint)
    Evidence: STRUCTURAL_ANALYSIS
    Strength: BEHAVIORAL
    """

    reliance_class: Literal["inferred_identity"] = "inferred_identity"
    # What identity claim is being inferred?
    identity_claim: str = ""           # e.g. "caller is admin"
    # What signal is used to infer identity?
    inference_signal: str = ""         # e.g. "presence of X-Admin-Token header"
    # What authoritative mechanism is being bypassed?
    bypassed_mechanism: str = ""       # e.g. "GovernanceCredential.verify_signature()"
    gated_decision: str = ""
    verification_style: VerificationStyle = VerificationStyle.INFERRED
    reliance_risk: RelianceRisk = RelianceRisk.CRITICAL
    remediation: str = ""
    observed_at_function: str = ""
    observed_at_line: int = 0


class RememberedProtocolReliance(PhaseInvariant):
    """
    A decision assumes a protocol-level property (e.g., handshake complete,
    channel authenticated) because it was true at an earlier step, without
    re-asserting it at the decision point.

    Concrete mechanisms:
      - TLS: encryption established at connection time but certificate validity
        not re-checked at each sensitive operation
      - Governance handshake: quorum assumed from a prior gate, not re-checked
        at final commit
      - Multi-step workflow: step N trusts that step N-1 validated the input
        without a formal contract between steps

    Risk: protocol state can change between establishment and use --
    connection downgraded, session hijacked, quorum shifted.

    Z3-checkable: partial -- encode as FSM invariant:
        decision_precondition_at_step_N requires
        protocol_state_valid_at_step_N (FSM adjacency encoding)
    Strength: STRUCTURAL
    """

    reliance_class: Literal["remembered_protocol"] = "remembered_protocol"
    protocol_property: str = ""        # e.g. "governance_quorum_satisfied"
    # At which step was the property established?
    established_at_step: str = ""      # e.g. "GATE stage"
    # At which step is the decision made that relies on it?
    relied_on_at_step: str = ""        # e.g. "RECORD stage"
    # What would need to change between steps to violate the assumption?
    invalidation_scenario: str = ""
    verification_style: VerificationStyle = VerificationStyle.ASSUMED_VALID
    reliance_risk: RelianceRisk = RelianceRisk.HIGH
    remediation: str = ""
    observed_at_function: str = ""
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)


# ── 2. Source-of-Truth Gaps ───────────────────────────────────────────────────


class LiveVsStoredTruthGap(PhaseInvariant):
    """
    Divergence between a live truth and a stored representation used for
    decisions -- the code acts on what was stored, not on what is current.

    Concrete mechanisms:
      - Role stored in session vs. role in the role registry
      - Budget available stored in a config snapshot vs. actual committed spend
      - File content hash stored in incremental cache vs. actual file on disk
      - Principal revocation list cached locally vs. live revocation feed

    The gap has three parts:
      (a) live_truth_source      -- where the ground truth lives
      (b) stored_truth_location  -- where the cached/stored copy lives
      (c) divergence_window      -- how long the gap can exist before a decision

    Z3-checkable: yes -- encode:
        stored_value_age > max_staleness_seconds => decision is gap-reliant
    Strength: BEHAVIORAL
    """

    reliance_class: Literal["live_vs_stored"] = "live_vs_stored"
    live_truth_source: str = ""        # e.g. "principal_registry.role_for(id)"
    stored_truth_location: str = ""    # e.g. "IdentityContext.principal_role"
    # How the stored copy gets updated (if at all)
    update_mechanism: str = ""         # e.g. "refreshed on login only"
    # Maximum tolerable staleness before decision should force refresh (0 = must be live)
    max_staleness_seconds: int = 0
    gated_decision: str = ""
    reliance_risk: RelianceRisk = RelianceRisk.HIGH
    remediation: str = ""
    observed_at_function: str = ""
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)
    # Runtime
    observed_max_staleness_seconds: float = 0.0
    observed_gap_count: int = 0        # times gap was present when decision fired


class InferredTruthGap(PhaseInvariant):
    """
    A decision relies on a truth that was inferred from secondary signals
    rather than read from any authoritative source (live or stored).

    The inferral is the source-of-truth -- there is no primary source consulted.

    Concrete mechanisms:
      - Risk score derived from proxy metric when direct metric is unavailable
      - Health status inferred from absence of error events vs. active heartbeat
      - Quorum status inferred from last known vote count vs. a live quorum check
      - "Admin is present" inferred from recent activity log entries

    Risk: inference fails in the exact scenarios where truth deviates most
    from the proxy.

    Z3-checkable: no (inference validity is not expressible as a linear constraint)
    Evidence: STRUCTURAL_ANALYSIS
    Strength: BEHAVIORAL
    """

    reliance_class: Literal["inferred_truth"] = "inferred_truth"
    inferred_truth: str = ""           # e.g. "system is healthy"
    inference_signals: list[str] = Field(default_factory=list)
    # Would a primary source be available (just not consulted)?
    primary_source_exists: bool = True
    primary_source: str = ""           # e.g. "HealthChecker.get_live_status()"
    gated_decision: str = ""
    reliance_risk: RelianceRisk = RelianceRisk.MEDIUM
    remediation: str = ""
    observed_at_function: str = ""
    known_failure_modes: list[str] = Field(default_factory=list)


class OriginVerificationGap(PhaseInvariant):
    """
    The code trusts a remembered claim about data origin rather than
    verifying the origin at use time.

    The core question: Does the code check the origin, or trust a remembered claim?

    Concrete mechanisms:
      - A data item has a field `source = "trusted_internal"` that was set by
        the data producer -- the consumer trusts this without re-checking
      - An event carries `is_from_admin = True` set during ingestion but not
        re-verified against the event's signature at processing time
      - A file is assumed to be the correct version because its path matches
        an expected path, without content hash verification

    Z3-checkable: yes (as implication: used_without_verify = True and
                  origin_verifiable = True => gap_present)
    Strength: STRUCTURAL
    """

    reliance_class: Literal["origin_verification"] = "origin_verification"
    origin_claim: str = ""             # e.g. "data.source == 'evo_system'"
    claim_location: str = ""           # e.g. "EvolutionProposal.submitted_by"
    # Who set the claim and how trustworthy is that setter?
    claim_setter: str = ""             # e.g. "request submitter (unverified)"
    # What would verify the origin?
    verification_mechanism: str = ""   # e.g. "verify GovernanceCredential signature"
    is_verified_at_use: bool = False
    gated_decision: str = ""
    reliance_risk: RelianceRisk = RelianceRisk.CRITICAL
    remediation: str = ""
    observed_at_function: str = ""
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)


# ── 3. Session Continuity ─────────────────────────────────────────────────────


class SessionAssumptionReliance(PhaseInvariant):
    """
    A multi-step session (login flow, proposal lifecycle, negotiation,
    multi-step workflow) advances because earlier steps are assumed still
    valid -- not because they are re-confirmed.

    Concrete mechanisms:
      - Login session: step 3 (authorise action) trusts that step 1 (authenticate)
        still holds without re-checking token validity
      - Proposal lifecycle: RECORD stage trusts that GATE still applies
        even if governance conditions changed between stages
      - Negotiation: final acceptance trusts that opening terms were not
        modified in transit without re-verifying the message chain

    Critical question: What assumptions must persist for decisions to proceed?

    Z3-checkable: yes -- encode as FSM invariant:
        state_at_step_N.preconditions_satisfied = True requires
        forall earlier_step in [0..N-1]: earlier_step.postconditions_still_valid
    Strength: STRUCTURAL
    """

    reliance_class: Literal["session_assumption"] = "session_assumption"
    session_type: str = ""             # e.g. "proposal_lifecycle"
    persistent_assumption: str = ""    # e.g. "identity.credential_strength >= 0.8"
    assumption_established_at: str = "" # e.g. "step 1: authenticate"
    relied_on_at_steps: list[str] = Field(default_factory=list)
    invalidating_events: list[str] = Field(default_factory=list)
    # Is there any re-validation mechanism between steps?
    revalidation_mechanism: str = ""   # "" = none
    reliance_risk: RelianceRisk = RelianceRisk.HIGH
    remediation: str = ""
    observed_at_function: str = ""
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)
    fsm_name: str = ""
    # Observed: sessions that completed without re-checking the assumption
    unvalidated_completions: int = 0


class NarrativeContinuityReliance(PhaseInvariant):
    """
    A decision assumes that the history of events ("what happened so far")
    is complete and unmodified -- that no external actor has injected, dropped,
    or reordered events since the narrative was established.

    Concrete mechanisms:
      - Event log assumed append-only -- no gaps checked before acting
      - Audit trail read as authoritative history without integrity check
      - Hash chain not verified before decisions that depend on chain position
      - Message queue position used as ordering proof without message signing

    Z3-checkable: yes -- encode as:
        chain_length_at_decision == expected_chain_length
        (any drop or injection changes length or hash)
    Strength: AXIOM for cryptographic chains, STRUCTURAL for append-only assumptions
    """

    reliance_class: Literal["narrative_continuity"] = "narrative_continuity"
    narrative_source: str = ""         # e.g. "HashChain event log"
    # Integrity mechanism protecting the narrative (empty = none)
    integrity_mechanism: str = ""      # e.g. "SHA-256 chain hash"
    is_integrity_verified_before_act: bool = False
    gated_decision: str = ""
    # What would undetectable narrative modification look like?
    attack_surface: str = ""           # e.g. "insert event before chain verification"
    reliance_risk: RelianceRisk = RelianceRisk.CRITICAL
    remediation: str = ""
    observed_at_function: str = ""
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)
    # Runtime
    narrative_integrity_checks_count: int = 0
    decisions_without_integrity_check: int = 0


class WorkflowPreconditionReliance(PhaseInvariant):
    """
    A workflow step proceeds because a precondition was satisfied at entry,
    without a mechanism to detect if that precondition was violated during
    the step's own execution.

    This is the TOCTOU (time-of-check to time-of-use) pattern applied to
    workflow preconditions.

    Concrete mechanisms:
      - Budget check at proposal validation -- not re-checked at application time
        even though budget could be consumed by a concurrent proposal
      - Quorum check at gate stage -- not re-checked at commit even though voters
        could resign in the gap
      - Rate limit check at request ingestion -- not re-checked per sub-operation
        within the same request

    Z3-checkable: yes -- encode as:
        precondition_holds_at_check AND
        (time_of_use - time_of_check) > epsilon_ms => violation_possible
    Strength: STRUCTURAL
    """

    reliance_class: Literal["workflow_precondition"] = "workflow_precondition"
    precondition: str = ""             # e.g. "budget_available >= proposal_cost"
    checked_at_step: str = ""          # e.g. "VALIDATE"
    consumed_at_step: str = ""         # e.g. "APPLY"
    # What could change the precondition between check and use?
    toctou_scenario: str = ""          # e.g. "concurrent proposal approved"
    is_rechecked_before_consume: bool = False
    reliance_risk: RelianceRisk = RelianceRisk.HIGH
    remediation: str = ""
    observed_at_function: str = ""
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)
    # Observed: times consuming step fired without a re-check
    unguarded_consumptions: int = 0


# ── Aggregate Discovery Results ───────────────────────────────────────────────


class StateInterpretationDiscovery(EOSBaseModel):
    """Discovered state interpretation reliance findings for a target component."""

    target_file: str = ""
    target_functions: list[str] = Field(default_factory=list)
    cached_authority: list[CachedAuthorityReliance] = Field(default_factory=list)
    inferred_identity: list[InferredIdentityReliance] = Field(default_factory=list)
    remembered_protocol: list[RememberedProtocolReliance] = Field(default_factory=list)
    z3_verified_count: int = 0
    critical_count: int = 0
    violations_found: int = 0
    discovery_time_ms: int = 0


class SourceOfTruthDiscovery(EOSBaseModel):
    """Discovered source-of-truth gap findings for a target component."""

    target_file: str = ""
    target_functions: list[str] = Field(default_factory=list)
    live_vs_stored: list[LiveVsStoredTruthGap] = Field(default_factory=list)
    inferred_truth: list[InferredTruthGap] = Field(default_factory=list)
    origin_verification: list[OriginVerificationGap] = Field(default_factory=list)
    z3_verified_count: int = 0
    critical_count: int = 0
    violations_found: int = 0
    discovery_time_ms: int = 0


class SessionContinuityDiscovery(EOSBaseModel):
    """Discovered session continuity reliance findings for a target component."""

    target_file: str = ""
    target_functions: list[str] = Field(default_factory=list)
    session_assumptions: list[SessionAssumptionReliance] = Field(default_factory=list)
    narrative_continuity: list[NarrativeContinuityReliance] = Field(default_factory=list)
    workflow_preconditions: list[WorkflowPreconditionReliance] = Field(default_factory=list)
    z3_verified_count: int = 0
    critical_count: int = 0
    violations_found: int = 0
    discovery_time_ms: int = 0


class Phase3DecisionRelianceReport(EOSBaseModel):
    """
    Aggregate result of Phase 3 decision reliance analysis.

    Produced by DecisionRelianceEngine.run().
    Attached to HealthCheckResult alongside Phase2InvariantReport.

    Phase 3 answers: where does belief substitute for verification?

    Consumed by SimulaService for go/no-go decisions:
      - CRITICAL findings block proposal application
      - HIGH findings are advisory (logged, surfaced to governance)
      - MEDIUM/LOW findings are informational
    """

    target_file: str = ""
    state_interpretation: StateInterpretationDiscovery = Field(
        default_factory=StateInterpretationDiscovery
    )
    source_of_truth: SourceOfTruthDiscovery = Field(
        default_factory=SourceOfTruthDiscovery
    )
    session_continuity: SessionContinuityDiscovery = Field(
        default_factory=SessionContinuityDiscovery
    )

    total_findings: int = 0
    z3_verified_total: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    violations_found: int = 0

    # Blocking vs advisory split
    blocking_findings: list[str] = Field(default_factory=list)
    advisory_findings: list[str] = Field(default_factory=list)
    informational_findings: list[str] = Field(default_factory=list)

    passed: bool = True
    discovery_time_ms: int = 0
    discovered_at: datetime = Field(default_factory=utc_now)
