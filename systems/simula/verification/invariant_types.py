"""
EcodiaOS - Simula Phase 2 Invariant Discovery Types

Four invariant classes targeting what cannot easily be hidden or transformed -
the "physics" of digital systems:

  1. Semantic Invariants   - output stability and decision boundary preservation
  2. State Invariants      - structural data relationships that must always hold
  3. Trust Invariants      - identity authority relationships and delegation chains
  4. Physical Invariants   - conservation laws, process bounds, feasibility constraints

Each class extends DiscoveredInvariant with domain-specific evidence fields.
All are Z3-checkable where possible; where not (physical feasibility), they
fall back to runtime witness collection + statistical bound checking.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now

# ── Evidence Confidence ───────────────────────────────────────────────────────


class EvidenceSource(enum.StrEnum):
    """How was this invariant's validity established?"""

    Z3_PROOF = "z3_proof"               # Negation UNSAT under Z3
    RUNTIME_WITNESS = "runtime_witness"  # Held on N observed executions
    DAFNY_LEMMA = "dafny_lemma"         # Proven as Dafny postcondition
    STATISTICAL_BOUND = "statistical_bound"  # μ ± 3σ from profiling data
    STRUCTURAL_ANALYSIS = "structural_analysis"  # Derived from AST/type analysis
    CONSERVATION_LAW = "conservation_law"  # Physical/mathematical identity


class InvariantStrength(enum.StrEnum):
    """How hard is this invariant to violate or transform away?"""

    AXIOM = "axiom"           # Cannot be violated by any digital transformation
    STRUCTURAL = "structural"  # Requires structural codebase change to violate
    BEHAVIORAL = "behavioral"  # Holds under normal operation; adversary can break
    STATISTICAL = "statistical"  # Holds with high probability across observations


# ── Base Invariant ────────────────────────────────────────────────────────────


class PhaseInvariant(EOSBaseModel):
    """
    Base model for all Phase 2 invariants.

    Extends the Stage 2B DiscoveredInvariant with:
      - strength classification (axiom / structural / behavioral / statistical)
      - evidence source (how we know it holds)
      - witness count (how many executions confirmed it)
      - violation_witness (concrete counterexample if found)
      - z3_expression remains the canonical check form where applicable
    """

    invariant_id: str = ""
    description: str
    strength: InvariantStrength = InvariantStrength.BEHAVIORAL
    evidence_source: EvidenceSource = EvidenceSource.RUNTIME_WITNESS
    confidence: float = 0.0           # [0.0, 1.0]
    witness_count: int = 0            # executions that confirmed this holds
    violation_witness: str = ""       # non-empty = invariant was violated
    z3_expression: str = ""           # Z3 Python expression (if checkable)
    variable_declarations: dict[str, str] = Field(default_factory=dict)
    target_file: str = ""
    target_function: str = ""
    discovered_at: datetime = Field(default_factory=utc_now)


# ── 1. Semantic Invariants ────────────────────────────────────────────────────


class OutputStabilityInvariant(PhaseInvariant):
    """
    Outputs must remain stable for identical logical inputs.

    Concrete mechanisms:
      - For pure functions: same inputs → same output (referential transparency)
      - For stateful functions: output changes only when tracked state changes
      - Decision thresholds cannot silently shift between calls

    Z3-checkable: yes (for bounded input domains)
    Strength: STRUCTURAL (requires code change to break, not config)
    """

    invariant_class: Literal["output_stability"] = "output_stability"
    input_variables: list[str] = Field(default_factory=list)   # inputs that determine output
    output_variable: str = ""                                   # the output being stabilised
    state_variables: list[str] = Field(default_factory=list)   # additional state that may vary
    # Measured stability: std-dev of output across N calls with fixed inputs
    observed_output_variance: float = 0.0
    variance_threshold: float = 0.0  # max acceptable variance (0.0 = pure)


class DecisionBoundaryInvariant(PhaseInvariant):
    """
    Decision boundaries cannot be crossed without observable cause.

    A decision boundary separates two distinct behaviours (e.g., APPROVE vs REJECT,
    HIGH_RISK vs ACCEPTABLE_RISK). Once established, the threshold value must not
    silently drift. Crossing requires an explicit, recorded cause.

    Concrete mechanisms:
      - Risk threshold at 0.10 (unacceptable regression rate) is fixed
      - Governance gate threshold is an iron rule
      - Budget headroom threshold (0%) cannot be implicitly relaxed

    Z3-checkable: yes (threshold comparison is a linear constraint)
    Strength: AXIOM for iron-rule thresholds, STRUCTURAL for derived ones
    """

    invariant_class: Literal["decision_boundary"] = "decision_boundary"
    threshold_variable: str = ""      # variable holding the threshold
    threshold_value: float = 0.0      # the boundary value
    decision_above: str = ""          # behaviour when input > threshold
    decision_below: str = ""          # behaviour when input ≤ threshold
    threshold_is_mutable: bool = False  # True = can be changed via proposal


class SemanticEquivalenceInvariant(PhaseInvariant):
    """
    Semantically equivalent code paths must produce behaviourally identical results.

    Any refactoring or optimisation that claims semantic equivalence must preserve:
      - Return value for all inputs in the covered domain
      - Side effects (state mutations, writes, emissions)
      - Exception behaviour

    Z3-checkable: via e-graph equality saturation (see egraph module)
    Strength: STRUCTURAL
    """

    invariant_class: Literal["semantic_equivalence"] = "semantic_equivalence"
    canonical_form_hash: str = ""       # hash of the canonical e-class
    equivalent_form_hashes: list[str] = Field(default_factory=list)
    input_domain: str = ""              # e.g. "risk_score ∈ [0.0, 1.0]"
    exceptions_preserved: bool = True   # must preserve exception behaviour


# ── 2. State Invariants ───────────────────────────────────────────────────────


class CounterMonotonicityInvariant(PhaseInvariant):
    """
    Counters must progress logically - they cannot decrease without an explicit reset.

    Concrete mechanisms:
      - config_version: strictly monotonically increasing (never rolls back)
      - proposal sequence numbers: strictly increasing
      - evolution record chain positions: strictly increasing

    Z3-checkable: yes (n+1 > n is a simple linear arithmetic fact)
    Strength: AXIOM for append-only stores, STRUCTURAL for in-memory counters
    """

    invariant_class: Literal["counter_monotonicity"] = "counter_monotonicity"
    counter_variable: str = ""
    monotonicity_kind: str = "strict"  # "strict" = always increases, "weak" = non-decreasing
    is_strictly_monotone: bool = True  # True = strict, False = non-strict (weak)
    reset_allowed: bool = False        # if True, reset is a documented operation
    lower_bound: float = 0.0
    upper_bound: float | None = None
    observed_max_value: int = 0
    observed_min_step: int = 1         # smallest observed increment


class SessionConsistencyInvariant(PhaseInvariant):
    """
    Sessions must evolve consistently - each transition must be valid per the FSM.

    A session (proposal lifecycle, auth session, tool session) has a defined
    state machine. Valid transitions are explicitly enumerated; any other
    transition is a violation.

    Concrete mechanisms:
      - Proposal status: PROPOSED → SIMULATING → ... → RECORDED (never backwards)
      - No terminal state can transition to a non-terminal state
      - Each state transition must be accompanied by a recorded cause

    Z3-checkable: yes (adjacency matrix is finite and checkable)
    Strength: STRUCTURAL
    """

    invariant_class: Literal["session_consistency"] = "session_consistency"
    session_type: str = ""             # e.g. "proposal_lifecycle"
    valid_states: list[str] = Field(default_factory=list)
    valid_transitions: list[tuple[str, str]] = Field(default_factory=list)  # (from, to)
    terminal_states: list[str] = Field(default_factory=list)
    transition_table: dict[str, list[str]] = Field(default_factory=dict)  # {from: [to, ...]}
    allows_backward_transition: bool = False
    # Observed violations: non-empty = invariant was violated
    observed_invalid_transitions: list[tuple[str, str]] = Field(default_factory=list)


class CacheCoherenceInvariant(PhaseInvariant):
    """
    Caches must match their authority source - stale reads are bounded in time.

    For every cached value there is an authoritative source. The cache invariant
    has two parts:
      1. Value coherence: cache[k] == authority[k] within TTL window
      2. Eviction completeness: when authority changes, cache invalidation propagates

    Concrete mechanisms:
      - Incremental verification cache: function must be re-verified if content_hash changes
      - Analytics cache (5-min TTL): cannot serve stale risk scores beyond TTL
      - Neo4j cold cache: must be invalidated when proposal is rolled back

    Z3-checkable: no (requires runtime observation of authority vs cache)
    Evidence: RUNTIME_WITNESS
    Strength: BEHAVIORAL
    """

    invariant_class: Literal["cache_coherence"] = "cache_coherence"
    cache_name: str = ""
    authority_source: str = ""         # what the cache mirrors
    coherence_protocol: str = "write_through"  # "write_through" | "read_through" | "eventual"
    ttl_seconds: int = 0
    max_staleness_ms: int = 0          # max_staleness in ms (ttl_seconds * 1000)
    invalidation_trigger: str = ""     # what causes cache invalidation
    invalidation_events: list[str] = Field(default_factory=list)
    # Measured coherence: fraction of reads where cache matched authority
    observed_coherence_rate: float = 1.0
    max_staleness_seconds_observed: float = 0.0


class RelationalIntegrityInvariant(PhaseInvariant):
    """
    Relationships between data entities must hold across all mutations.

    Concrete mechanisms:
      - Every EvolutionRecord must reference a valid proposal_id
      - Every HashChainEntry.previous_hash must equal prior entry's chain_hash
      - Every FileSnapshot in a ConfigSnapshot must reference an existing proposal

    Z3-checkable: yes for hash chain linkage (SHA-256 chain is checkable)
    Strength: AXIOM for hash chains, STRUCTURAL for foreign-key style links
    """

    invariant_class: Literal["relational_integrity"] = "relational_integrity"
    relation_name: str = ""            # e.g. "hash_chain_linkage"
    entity_a: str = ""                 # e.g. "EvolutionRecord"
    entity_b: str = ""                 # e.g. "EvolutionProposal"
    relationship: str = ""             # e.g. "proposal_id references"
    parent_key: str = ""               # e.g. "proposal_id"
    child_key: str = ""                # e.g. "simulation_id"
    integrity_kind: str = "referential"  # "cryptographic" | "referential"
    cardinality: str = "many_to_one"   # "one_to_one" | "many_to_one" | "one_to_many"
    cascade_on_delete: bool = False
    observed_violations: int = 0


# ── 3. Trust Invariants ───────────────────────────────────────────────────────


class DelegationChainInvariant(PhaseInvariant):
    """
    Delegated trust must reflect origin authority - the chain cannot be forged.

    If principal A delegates to principal B, B's trust level cannot exceed A's.
    The chain is acyclic: no principal can be its own authority ancestor.

    Concrete mechanisms:
      - IdentityContext.delegation_chain must be acyclic (no cycles)
      - Delegated credential_strength ≤ delegator's credential_strength
      - delegation_chain length is bounded (prevents unbounded authority laundering)

    Z3-checkable: yes (acyclicity is a graph reachability constraint encodable in Z3)
    Strength: AXIOM (mathematical property of delegation semantics)
    """

    invariant_class: Literal["delegation_chain"] = "delegation_chain"
    origin_principal: str = ""         # principal at chain root
    max_chain_depth: int = 5           # maximum allowed delegation depth
    chain_is_acyclic: bool = True      # must always be True
    delegation_ceiling: float = 1.0    # max credential_strength that can be delegated
    # Z3 encoding: acyclicity as ∀i,j in chain: i ≠ j (all-different on positions)
    observed_max_depth: int = 0
    observed_cycles: list[list[str]] = Field(default_factory=list)  # non-empty = violation


class AuthorityPreservationInvariant(PhaseInvariant):
    """
    Authority cannot be amplified through the delegation or proposal process.

    No operation within the system can grant a principal more authority than
    its highest-trust ancestor. This prevents privilege escalation.

    Concrete mechanisms:
      - external_api callers (strength=0.7) cannot produce CRITICAL-risk proposals
      - governance_vote identity cannot bypass FORBIDDEN categories
      - repair_agent cannot modify FORBIDDEN_WRITE_PATHS

    Z3-checkable: yes (max_risk_allowed constraints are linear comparisons)
    Strength: STRUCTURAL
    """

    invariant_class: Literal["authority_preservation"] = "authority_preservation"
    principal_id: str = ""                   # principal this applies to
    principal_role: str = ""                 # role this applies to
    max_credential_strength: float = 1.0     # ceiling for this role
    max_authority_level: float = 1.0         # alias for max_credential_strength
    authority_is_monotone_non_escalating: bool = True  # True = cannot escalate
    max_risk_level_allowed: str = "HIGH"     # ceiling risk level
    # Witnessed escalation attempts: non-empty = invariant was violated
    escalation_attempts: list[dict[str, str]] = Field(default_factory=list)


class CredentialIntegrityInvariant(PhaseInvariant):
    """
    Credentials must not be forged, replayed, or modified in transit.

    A credential presented at time T was issued by authority A for purpose P.
    It must not be usable for a different purpose P', at a different time T+Δ
    beyond its validity window, or by a different principal.

    Concrete mechanisms:
      - GovernanceCredential.signed_payload_hash must match SHA-256 of payload
      - ContentCredential.signature must verify under the issuer's Ed25519 key
      - API tokens are single-use within a session (no cross-session replay)

    Z3-checkable: no (cryptographic verification is not Z3-domain)
    Evidence: STRUCTURAL_ANALYSIS + RUNTIME_WITNESS
    Strength: AXIOM (given correct crypto implementation)
    """

    invariant_class: Literal["credential_integrity"] = "credential_integrity"
    credential_type: str = ""               # "governance" | "content" | "api_token"
    credential_kind_order: list[str] = Field(default_factory=list)  # ordered strength ladder
    strength_is_consistent_with_kind: bool = True  # strength matches kind's position
    signature_algorithm: str = "Ed25519"
    max_validity_seconds: int = 3600
    replay_window_seconds: int = 0          # 0 = no replay allowed
    observed_replay_attempts: int = 0
    observed_forgery_attempts: int = 0


class TrustBoundaryInvariant(PhaseInvariant):
    """
    Trust boundaries cannot be crossed without explicit mediation.

    The system has defined trust boundaries (internal ↔ external, admin ↔ evo).
    Data flowing across a boundary must pass through a mediator that validates
    it against the receiving side's trust level.

    Concrete mechanisms:
      - External API input must pass through EvoSimulaBridge before SimulaService
      - Repair agent output must pass through HealthChecker before recording
      - Governance votes must be verified before unlocking GOVERNANCE_REQUIRED changes

    Z3-checkable: yes (reachability in a directed trust graph)
    Strength: STRUCTURAL
    """

    invariant_class: Literal["trust_boundary"] = "trust_boundary"
    boundary_name: str = ""            # e.g. "external_to_internal"
    from_trust_zone: str = ""          # e.g. "external"
    to_trust_zone: str = ""            # e.g. "simula_core"
    source_domain: str = ""            # alias for from_trust_zone
    target_domain: str = ""            # alias for to_trust_zone
    mediator_required: str = ""        # e.g. "EvoSimulaBridge"
    required_mediator: str = ""        # alias for mediator_required
    crossing_condition: str = ""       # what must be satisfied to cross
    # Observed direct-crossing attempts: non-empty = boundary violated
    direct_crossing_attempts: int = 0


# ── 4. Physical Invariants ────────────────────────────────────────────────────


class ConservationConstraintInvariant(PhaseInvariant):
    """
    Conservation constraints that digital operations cannot violate.

    In CPS and resource-constrained systems, certain quantities are conserved
    across operations. Any operation that claims to change them must account
    for where the conserved quantity went.

    Concrete mechanisms:
      - Token budget: total_allocated + available = initial_budget (no creation/loss)
      - File content: after rollback, byte-for-byte equality with snapshot
      - Hash chain: no entries can be removed (chain_length is strictly non-decreasing)
      - Audit trail: record count is non-decreasing (records are never deleted)

    Z3-checkable: yes (linear equality / inequality)
    Strength: AXIOM (mathematical identity)
    """

    invariant_class: Literal["conservation_constraint"] = "conservation_constraint"
    quantity_name: str = ""              # what is conserved (alias for conserved_quantity)
    conserved_quantity: str = ""         # e.g. "total_budget"
    conservation_equation: str = ""     # e.g. "allocated + available == budget"
    parts: list[str] = Field(default_factory=list)  # sub-quantities that sum to whole
    z3_expression: str = ""
    variable_declarations: dict[str, str] = Field(default_factory=dict)
    # Witnessed violations (non-zero quantity leaked or created)
    observed_imbalances: list[dict[str, float]] = Field(default_factory=list)


class ProcessBoundInvariant(PhaseInvariant):
    """
    Process parameters must remain within feasible operating bounds.

    Certain process parameters have hard physical or architectural bounds.
    Exceeding them causes system instability, runaway resource consumption,
    or undefined behaviour that the system cannot recover from.

    Concrete mechanisms:
      - LLM token budget: llm_tokens_per_hour ∈ [0, MAX_TOKEN_BUDGET]
      - Cycle timing: cycle_duration_ms ≥ MIN_CYCLE_MS (cannot run faster than hardware)
      - Rollback depth: rollback_depth ∈ [1, MAX_ROLLBACK_DEPTH]
      - Simulation timeout: simulation_ms ≤ 30_000 (service SLA)

    Z3-checkable: yes (range bounds are linear constraints)
    Strength: STRUCTURAL (exceeding them requires config change)
    """

    invariant_class: Literal["process_bound"] = "process_bound"
    parameter_name: str = ""
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str = ""                     # "ms" | "tokens" | "bytes" | "count" | "dimensionless"
    bound_unit: str = ""               # alias for unit
    # Observed breaches
    observed_min: float | None = None
    observed_max: float | None = None
    breach_count: int = 0              # times parameter left bounds


class PhysicalFeasibilityInvariant(PhaseInvariant):
    """
    Operations must be physically feasible - they cannot claim impossible outcomes.

    Reality-based truths that cannot be digitally altered:
      - Rollback time ≤ snapshot_size / disk_write_bandwidth  (cannot be faster than physics)
      - Memory usage ≥ data_size (cannot store N bytes in < N bytes)
      - Verification time ≥ program_complexity / compute_speed  (no oracle exists)
      - Timing measurements satisfy triangle inequality (A→C ≤ A→B + B→C)

    These are not about what the software says - they are about what hardware permits.

    Z3-checkable: where bounds are numeric (linear arithmetic)
    Evidence: STATISTICAL_BOUND from profiling data
    Strength: AXIOM (hardware physics)
    """

    invariant_class: Literal["physical_feasibility"] = "physical_feasibility"
    operation_name: str = ""
    # Physical constraint in English
    constraint_description: str = ""
    # Lower-bound on duration from observed physics (μ - 3σ from profiling)
    physical_minimum_ms: float = 0.0
    physical_minimum_ns: float = 0.0   # nanosecond variant from profiler data
    domain_minimum_ns: float = 0.0     # hardware domain minimum (nanoseconds)
    # Upper bound from SLA
    sla_maximum_ms: float | None = None
    # Statistical parameters from profiling
    observed_mean_ms: float = 0.0
    observed_mean_ns: float = 0.0      # nanosecond variant
    observed_stdev_ms: float = 0.0
    observed_stdev_ns: float = 0.0     # nanosecond variant
    sample_count: int = 0
    # Were any "impossibly fast" observations detected? (< physical_minimum)
    impossibly_fast_count: int = 0


class ResourceConservationInvariant(PhaseInvariant):
    """
    Resource consumption must be physically consistent across accounting layers.

    When a process allocates/consumes a resource, the consumption must be
    observable at the hardware layer. If a process claims to use X tokens but
    the API billing shows Y, the discrepancy signals measurement tampering.

    Concrete mechanisms:
      - LLM token counts: reported_tokens ≈ actual_billed_tokens (±jitter)
      - Memory: reported_peak_mb ≤ process_rss_mb (no negative memory)
      - Disk writes: files_written ∈ written_file_list (no ghost writes)

    Z3-checkable: no (requires runtime cross-layer observation)
    Evidence: RUNTIME_WITNESS
    Strength: AXIOM (physical accounting identity)
    """

    invariant_class: Literal["resource_conservation"] = "resource_conservation"
    resource_name: str = ""              # "llm_tokens" | "memory_mb" | "disk_bytes"
    max_per_proposal: int = 0            # maximum allowed per proposal execution
    unit: str = ""                       # "tokens" | "files" | "solver_calls"
    reported_field: str = ""             # field name in the data model
    observable_source: str = ""          # where the ground truth comes from
    max_allowed_discrepancy_percent: float = 5.0  # tolerance for jitter
    observed_max_discrepancy_percent: float = 0.0
    discrepancy_breach_count: int = 0


# ── Aggregate Discovery Result ────────────────────────────────────────────────


class InvariantClass(enum.StrEnum):
    """Which of the four invariant classes does a discovery result cover?"""

    SEMANTIC = "semantic"
    STATE = "state"
    TRUST = "trust"
    PHYSICAL = "physical"


class SemanticInvariantDiscovery(EOSBaseModel):
    """Discovered semantic invariants for a target component."""

    target_file: str = ""
    target_functions: list[str] = Field(default_factory=list)
    output_stability: list[OutputStabilityInvariant] = Field(default_factory=list)
    decision_boundaries: list[DecisionBoundaryInvariant] = Field(default_factory=list)
    semantic_equivalences: list[SemanticEquivalenceInvariant] = Field(default_factory=list)
    z3_verified_count: int = 0
    runtime_confirmed_count: int = 0
    violations_found: int = 0
    discovery_time_ms: int = 0


class StateInvariantDiscovery(EOSBaseModel):
    """Discovered state invariants for a target component."""

    target_file: str = ""
    target_functions: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    counter_monotonicity: list[CounterMonotonicityInvariant] = Field(default_factory=list)
    session_consistency: list[SessionConsistencyInvariant] = Field(default_factory=list)
    cache_coherence: list[CacheCoherenceInvariant] = Field(default_factory=list)
    relational_integrity: list[RelationalIntegrityInvariant] = Field(default_factory=list)
    z3_verified_count: int = 0
    runtime_confirmed_count: int = 0
    violations_found: int = 0
    discovery_time_ms: int = 0


class TrustInvariantDiscovery(EOSBaseModel):
    """Discovered trust invariants for a target component."""

    target_file: str = ""
    target_principals: list[str] = Field(default_factory=list)
    delegation_chains: list[DelegationChainInvariant] = Field(default_factory=list)
    authority_preservation: list[AuthorityPreservationInvariant] = Field(default_factory=list)
    credential_integrity: list[CredentialIntegrityInvariant] = Field(default_factory=list)
    trust_boundaries: list[TrustBoundaryInvariant] = Field(default_factory=list)
    z3_verified_count: int = 0
    violations_found: int = 0
    discovery_time_ms: int = 0


class PhysicalInvariantDiscovery(EOSBaseModel):
    """Discovered physical invariants for a target component."""

    target_file: str = ""
    target_operations: list[str] = Field(default_factory=list)
    conservation_constraints: list[ConservationConstraintInvariant] = Field(default_factory=list)
    process_bounds: list[ProcessBoundInvariant] = Field(default_factory=list)
    physical_feasibility: list[PhysicalFeasibilityInvariant] = Field(default_factory=list)
    feasibility_checks: list[PhysicalFeasibilityInvariant] = Field(default_factory=list)  # alias
    resource_conservation: list[ResourceConservationInvariant] = Field(default_factory=list)
    z3_verified_count: int = 0
    statistical_confirmed_count: int = 0
    violations_found: int = 0
    impossibly_fast_observations: int = 0
    discovery_time_ms: int = 0


class Phase2InvariantReport(EOSBaseModel):
    """
    Aggregate result of Phase 2 invariant discovery across all four classes.

    Produced by InvariantDiscoveryEngine.run().
    Attached to HealthCheckResult and used by SimulaService for
    go/no-go decision-making on proposals.
    """

    target_file: str = ""
    semantic: SemanticInvariantDiscovery = Field(default_factory=SemanticInvariantDiscovery)
    state: StateInvariantDiscovery = Field(default_factory=StateInvariantDiscovery)
    trust: TrustInvariantDiscovery = Field(default_factory=TrustInvariantDiscovery)
    physical: PhysicalInvariantDiscovery = Field(default_factory=PhysicalInvariantDiscovery)

    # Summary statistics (populated by InvariantDiscoveryEngine)
    total_invariants: int = 0
    total_invariants_discovered: int = 0   # alias
    z3_verified_total: int = 0
    total_z3_verified: int = 0             # alias
    violations_found: int = 0
    total_violations: int = 0              # alias
    axiom_count: int = 0
    structural_count: int = 0
    statistical_count: int = 0

    blocking_violations: list[str] = Field(default_factory=list)
    advisory_violations: list[str] = Field(default_factory=list)

    passed: bool = True
    discovery_time_ms: int = 0
    discovered_at: datetime = Field(default_factory=utc_now)
