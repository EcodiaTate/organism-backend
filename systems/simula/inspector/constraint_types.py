"""
EcodiaOS - Inspector Phase 4: Constraint Reasoning Types

All domain models for the formal steerability analysis layer.

Design philosophy
-----------------
Phase 4 answers the question: "under exactly which conditions does a
deterministic execution become influence-permissive?"

It converts Phase 2/3 observations into a *formal state model* that can be
queried symbolically:

  StateModel          - the set of state variables that govern control flow
  InvariantSet        - pre/post-conditions the program must satisfy to behave normally
  ConstraintSet       - logical constraints encoding what must break for steerability
  SteerabilityModel   - per-target formalisation of the deterministic→permissive shift
  ConditionSet        - a concrete, testable set of invariant violations → new reachable region
  TransitionExplanation - structured narrative: broken invariants → relaxed constraints → new region
  Phase4Result        - top-level Phase 4 output

Layer map
---------
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 2 (ControlIntegrityScore, FaultObservation)                       │
  │  Phase 3 (ExecutionAtlas, FailureAdjacentRegion, CodeFragment)           │
  │    ↓  StateModelExtractor                                                │
  │  StateModel   - typed state variables (registers, memory, object state…) │
  │    ↓  ConstraintEngine                                                   │
  │  ConstraintSet - transition preconditions, reachability constraints       │
  │    ↓                                                                     │
  │  SteerabilityModel - per-target class characterisation                   │
  │    ↓                                                                     │
  │  ConditionSet  - concrete invariant violation → steerable region mapping  │
  │  TransitionExplanation - human-readable structured explanation             │
  └──────────────────────────────────────────────────────────────────────────┘

Exit criterion
--------------
Phase4Result.exit_criterion_met = True when, given a trace + context, the
SteerabilityAnalyzer can output a TransitionExplanation that identifies:
  1. which invariants were broken,
  2. which constraints were relaxed as a result, and
  3. which new CFG region became reachable (the "steerable region").
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ── Enumerations ───────────────────────────────────────────────────────────────


class StateVariableKind(enum.StrEnum):
    """
    High-level category of a tracked state variable.

    These categories mirror the state dimensions that matter for control-flow
    steerability - i.e., variables whose values determine which CFG branch is
    taken.
    """

    REGISTER          = "register"           # CPU register / local variable holding a value
    MEMORY_REGION     = "memory_region"      # Named heap / stack / global buffer
    OBJECT_LIFETIME   = "object_lifetime"    # Allocation↔free epoch for a heap object
    TYPE_TAG          = "type_tag"           # Runtime type annotation on an object
    FUNCTION_POINTER  = "function_pointer"   # Value of a function pointer / vtable slot
    PROTOCOL_STATE    = "protocol_state"     # State-machine node (e.g., HTTP session, TLS handshake)
    IDENTITY_CONTEXT  = "identity_context"   # Auth identity / privilege level / session token
    TAINT_LABEL       = "taint_label"        # Whether a variable carries external-input taint
    LOOP_COUNTER      = "loop_counter"       # Induction variable controlling a loop
    EXCEPTION_STATE   = "exception_state"    # Active exception / pending signal
    UNKNOWN           = "unknown"


class InvariantStrength(enum.StrEnum):
    """
    How strictly the program relies on an invariant remaining true.

    MUST - the program crashes / panics if this is violated.
    SHOULD - violation causes incorrect behaviour but not necessarily a crash.
    MAY - soft assumption; violation causes divergence from the happy path.
    """

    MUST   = "must"    # Safety invariant - violation = crash / assertion
    SHOULD = "should"  # Correctness invariant - violation = logic error
    MAY    = "may"     # Liveness invariant - violation = unexpected path


class ConstraintKind(enum.StrEnum):
    """
    The logical role of a constraint in the steerability model.

    PRECONDITION      - must hold *before* a transition fires.
    POSTCONDITION     - must hold *after* a transition fires.
    REACHABILITY      - structural: block B is reachable only if C holds.
    EXCLUSION         - structural: blocks A and B cannot both be reachable.
    TAINT_PROPAGATION - taint label X reaches variable Y under condition C.
    LIFETIME          - object O must be alive at point P.
    ORDERING          - event A must precede event B.
    """

    PRECONDITION      = "precondition"
    POSTCONDITION     = "postcondition"
    REACHABILITY      = "reachability"
    EXCLUSION         = "exclusion"
    TAINT_PROPAGATION = "taint_propagation"
    LIFETIME          = "lifetime"
    ORDERING          = "ordering"


class SteerabilityClass(enum.StrEnum):
    """
    Classification of how externally steerable a target (or region) is.

    DETERMINISTIC        - no external influence can alter the control-flow path.
    CONDITIONALLY_STEERABLE - steerability requires ≥1 invariant violation.
    INFLUENCE_PERMISSIVE - external input can select among CFG continuations
                           with non-negligible probability, without invariant
                           violation.
    FULLY_STEERABLE      - attacker has high-probability control over
                           multiple independent continuation choices.
    UNKNOWN              - insufficient evidence to classify.
    """

    DETERMINISTIC            = "deterministic"
    CONDITIONALLY_STEERABLE  = "conditionally_steerable"
    INFLUENCE_PERMISSIVE     = "influence_permissive"
    FULLY_STEERABLE          = "fully_steerable"
    UNKNOWN                  = "unknown"


class ViolationMechanism(enum.StrEnum):
    """
    How a state invariant could be violated in practice.

    These are *observed* or *inferred* mechanisms - not exploit templates.
    """

    BOUNDARY_VIOLATION  = "boundary_violation"   # Index/offset outside valid range
    LIFETIME_VIOLATION  = "lifetime_violation"   # Access after free / before alloc
    TYPE_MISMATCH       = "type_mismatch"        # Object treated as wrong type
    TAINT_INJECTION     = "taint_injection"      # Tainted data reaches guarded variable
    STATE_SKIP          = "state_skip"           # Protocol/FSM transition skipped
    PRIVILEGE_DROP      = "privilege_drop"       # Identity context downgraded unexpectedly
    POINTER_OVERWRITE   = "pointer_overwrite"    # Function pointer / vtable slot mutated
    ARITHMETIC_OVERFLOW = "arithmetic_overflow"  # Integer wrap-around alters index
    RACE_CONDITION      = "race_condition"       # Concurrent access breaks ordering
    UNKNOWN             = "unknown"


# ── State variables ────────────────────────────────────────────────────────────


class StateVariable(EOSBaseModel):
    """
    A single tracked state variable that influences control flow.

    Extracted from Phase 3 fragment semantics and Phase 2 taint/fault data.
    """

    var_id: str = Field(default_factory=new_id)
    target_id: str

    kind: StateVariableKind
    name: str = Field(..., description="Human-readable variable name or symbolic identifier")

    # Source location (approximate)
    func_name: str = Field(default="")
    file_path: str = Field(default="")
    definition_line: int | None = None

    # Which CFG blocks READ this variable (depend on its value for branching)
    read_in_blocks: list[str] = Field(
        default_factory=list,
        description="block_ids where this variable is read in a branch condition",
    )

    # Which CFG blocks WRITE / mutate this variable
    written_in_blocks: list[str] = Field(
        default_factory=list,
        description="block_ids where this variable is assigned or mutated",
    )

    # Whether this variable is reachable from an external (tainted) source
    taint_reachable: bool = Field(
        default=False,
        description="True if Phase 2/3 taint analysis found a path from an entry point",
    )

    # Associated fragment IDs (from Phase 3 catalog)
    fragment_ids: list[str] = Field(default_factory=list)

    # Extra metadata (e.g., allocation site, object class name)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Invariants ────────────────────────────────────────────────────────────────


class Invariant(EOSBaseModel):
    """
    A single program invariant that, when violated, widens the reachable CFG.

    Invariants are inferred from Phase 2 fault observations and Phase 3
    failure-adjacent region analysis.
    """

    invariant_id: str = Field(default_factory=new_id)
    target_id: str

    # Natural-language description of the invariant
    description: str = Field(
        ...,
        description="One-sentence description, e.g. 'ptr must not be NULL when entering allocate()'",
    )

    # Formal predicate (optional - symbolic representation)
    # Format: free-form Python/Z3-style expression string, e.g. "0 <= index < len(buf)"
    formal_predicate: str = Field(default="")

    strength: InvariantStrength
    kind: StateVariableKind = Field(
        default=StateVariableKind.UNKNOWN,
        description="Kind of state variable this invariant governs",
    )

    # State variables involved
    variable_ids: list[str] = Field(default_factory=list)

    # CFG blocks where this invariant is assumed to hold
    guarded_block_ids: list[str] = Field(
        default_factory=list,
        description="block_ids where a check asserts this invariant",
    )

    # CFG blocks that become reachable when this invariant is violated
    violation_unlocks_block_ids: list[str] = Field(
        default_factory=list,
        description="block_ids only reachable if this invariant is false",
    )

    # Source evidence
    derived_from_fault_class: str = Field(default="")
    derived_from_fragment_ids: list[str] = Field(default_factory=list)

    # Confidence in this invariant (0.0–1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class InvariantSet(EOSBaseModel):
    """
    The complete set of invariants for one target.

    Produced by StateModelExtractor.  Acts as the 'normal behaviour spec'
    against which ConstraintEngine reasons.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    invariants: dict[str, Invariant] = Field(default_factory=dict)
    total_invariants: int = 0

    # Cross-index: StateVariableKind → invariant_ids
    kind_index: dict[str, list[str]] = Field(default_factory=dict)

    # Cross-index: block_id → invariant_ids that guard it
    block_guard_index: dict[str, list[str]] = Field(default_factory=dict)

    built_at: datetime = Field(default_factory=utc_now)

    def add_invariant(self, inv: Invariant) -> None:
        self.invariants[inv.invariant_id] = inv
        self.total_invariants = len(self.invariants)

        key = inv.kind.value
        if key not in self.kind_index:
            self.kind_index[key] = []
        self.kind_index[key].append(inv.invariant_id)

        for bid in inv.guarded_block_ids:
            if bid not in self.block_guard_index:
                self.block_guard_index[bid] = []
            self.block_guard_index[bid].append(inv.invariant_id)

    def invariants_for_block(self, block_id: str) -> list[Invariant]:
        ids = self.block_guard_index.get(block_id, [])
        return [self.invariants[i] for i in ids if i in self.invariants]

    def invariants_by_kind(self, kind: StateVariableKind) -> list[Invariant]:
        ids = self.kind_index.get(kind.value, [])
        return [self.invariants[i] for i in ids if i in self.invariants]


# ── Constraints ───────────────────────────────────────────────────────────────


class Constraint(EOSBaseModel):
    """
    A single constraint on state transitions or CFG reachability.

    Constraints capture what must be true (or false) for a specific transition
    to fire.  They are the formal backbone of the steerability model.
    """

    constraint_id: str = Field(default_factory=new_id)
    target_id: str

    kind: ConstraintKind
    description: str = Field(..., description="Natural-language constraint description")

    # Formal expression (optional)
    # Format: symbolic expression string, e.g. "taint(x) ∧ ¬checked(x) → reachable(B)"
    formal_expression: str = Field(default="")

    # Variables and invariants involved
    variable_ids: list[str] = Field(default_factory=list)
    invariant_ids: list[str] = Field(default_factory=list)

    # CFG scope
    from_block_id: str = Field(default="")
    to_block_id: str = Field(default="")
    scope_block_ids: list[str] = Field(
        default_factory=list,
        description="All block_ids to which this constraint applies",
    )

    # Violation mechanism that would break this constraint
    violation_mechanism: ViolationMechanism = ViolationMechanism.UNKNOWN

    # If this constraint is relaxed (violated), which new blocks become reachable?
    unlocks_block_ids: list[str] = Field(default_factory=list)

    # Confidence in this constraint (0.0–1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Was this constraint violated in any Phase 2 failure run?
    violated_in_failure: bool = False

    # Fragment evidence from Phase 3
    evidence_fragment_ids: list[str] = Field(default_factory=list)


class ConstraintSet(EOSBaseModel):
    """
    The complete constraint system for one target.

    Produced by ConstraintEngine.  Answers reachability queries:
    'Which constraints must be relaxed to reach region R?'
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    constraints: dict[str, Constraint] = Field(default_factory=dict)
    total_constraints: int = 0

    # Cross-index: ConstraintKind → constraint_ids
    kind_index: dict[str, list[str]] = Field(default_factory=dict)

    # Cross-index: block_id → constraint_ids that gate it
    block_gate_index: dict[str, list[str]] = Field(default_factory=dict)

    # Cross-index: invariant_id → constraint_ids that reference it
    invariant_constraint_index: dict[str, list[str]] = Field(default_factory=dict)

    built_at: datetime = Field(default_factory=utc_now)

    def add_constraint(self, c: Constraint) -> None:
        self.constraints[c.constraint_id] = c
        self.total_constraints = len(self.constraints)

        key = c.kind.value
        if key not in self.kind_index:
            self.kind_index[key] = []
        self.kind_index[key].append(c.constraint_id)

        for bid in c.scope_block_ids:
            if bid not in self.block_gate_index:
                self.block_gate_index[bid] = []
            self.block_gate_index[bid].append(c.constraint_id)

        for iid in c.invariant_ids:
            if iid not in self.invariant_constraint_index:
                self.invariant_constraint_index[iid] = []
            self.invariant_constraint_index[iid].append(c.constraint_id)

    def constraints_gating_block(self, block_id: str) -> list[Constraint]:
        """Return all constraints that must hold for block_id to be reachable."""
        ids = self.block_gate_index.get(block_id, [])
        return [self.constraints[i] for i in ids if i in self.constraints]

    def constraints_for_invariant(self, invariant_id: str) -> list[Constraint]:
        """Return all constraints that reference the given invariant."""
        ids = self.invariant_constraint_index.get(invariant_id, [])
        return [self.constraints[i] for i in ids if i in self.constraints]

    def violated_constraints(self) -> list[Constraint]:
        """Return constraints observed to be violated in failure runs."""
        return [c for c in self.constraints.values() if c.violated_in_failure]


# ── Steerability model ────────────────────────────────────────────────────────


class SteerableRegion(EOSBaseModel):
    """
    A CFG sub-region that becomes reachable under specific invariant violations.

    This is the 'new reachable region' that defines what steerability means
    concretely for this target - the difference between normal and permissive
    execution space.
    """

    region_id: str = Field(default_factory=new_id)
    target_id: str

    # Block IDs composing this region
    block_ids: list[str] = Field(default_factory=list)
    function_names: list[str] = Field(default_factory=list)

    # Invariants whose violation unlocks this region
    required_violation_invariant_ids: list[str] = Field(
        default_factory=list,
        description="ALL of these invariants must be violated for this region to be reachable",
    )

    # Alternative: any-of violations (disjunctive unlocking)
    alternative_violation_invariant_ids: list[str] = Field(
        default_factory=list,
        description="ANY of these invariants being violated also unlocks this region",
    )

    # Constraints relaxed to reach this region
    relaxed_constraint_ids: list[str] = Field(default_factory=list)

    # Steerability classification for this specific region
    steerability_class: SteerabilityClass = SteerabilityClass.UNKNOWN

    # Probability that external input can direct flow into this region
    # given that the required invariants are already violated (0.0–1.0)
    conditional_steerability_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "P(externally steerable | required invariants violated). "
            "Estimated from Phase 2 influence-permissive transition fraction."
        ),
    )

    # Fragment IDs within this region (from Phase 3 catalog)
    fragment_ids: list[str] = Field(default_factory=list)

    # Observed in failure runs (Phase 2 failure-adjacent region overlap)
    observed_in_failure: bool = False

    # Fragments of particular interest (indirect dispatch, fault-adjacent)
    high_interest_fragment_ids: list[str] = Field(default_factory=list)


class ConditionSet(EOSBaseModel):
    """
    A concrete, testable set of conditions that predicts the
    deterministic → influence-permissive shift for a specific target.

    A ConditionSet is essentially a rule: "if ALL of these state conditions
    hold, then the execution enters the steerable region."

    These are the deliverables the research calls for - testable predictions
    of when normal execution becomes externally redirectable.
    """

    condition_set_id: str = Field(default_factory=new_id)
    target_id: str

    description: str = Field(
        ...,
        description="Human-readable summary of this condition set",
    )

    # The state conditions (invariant violations) that must ALL hold
    required_violations: list[str] = Field(
        ...,
        description="invariant_ids that must be violated for this condition set to trigger",
    )

    # The steerability outcome when this condition set is satisfied
    unlocked_region_id: str = Field(
        ...,
        description="SteerableRegion.region_id that becomes reachable",
    )

    # Mechanism: how the violations occur in practice
    violation_mechanisms: list[ViolationMechanism] = Field(default_factory=list)

    # Confidence in this condition set (0.0–1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Evidence: Phase 2 run_ids where this condition set was observed to hold
    supporting_run_ids: list[str] = Field(default_factory=list)

    # Formal predicate (optional)
    formal_trigger: str = Field(
        default="",
        description="Symbolic expression: conjunction of invariant violation predicates",
    )

    derived_at: datetime = Field(default_factory=utc_now)


class SteerabilityModel(EOSBaseModel):
    """
    The complete Phase 4 steerability model for one target.

    Per the research definition:
    'External input can select among multiple control-flow continuations
     with non-negligible probability.'

    A SteerabilityModel formalises this by specifying:
    - the state variables that govern control-flow choices,
    - the invariants that must hold for deterministic behaviour,
    - the constraints that encode what normal flow looks like,
    - the steerable regions reachable under invariant violation,
    - the condition sets that predict the deterministic→permissive shift.
    """

    id: str = Field(default_factory=new_id)
    target_id: str
    description: str = Field(default="")

    # State layer
    state_variables: dict[str, StateVariable] = Field(default_factory=dict)
    invariant_set: InvariantSet

    # Constraint layer
    constraint_set: ConstraintSet

    # Steerability layer
    steerable_regions: list[SteerableRegion] = Field(default_factory=list)
    condition_sets: list[ConditionSet] = Field(default_factory=list)

    # Overall classification for this target
    target_steerability_class: SteerabilityClass = SteerabilityClass.UNKNOWN

    # Summary statistics
    total_state_variables:  int = 0
    total_invariants:       int = 0
    total_constraints:      int = 0
    total_steerable_regions: int = 0
    total_condition_sets:   int = 0

    # Aggregate steerability probability (max over all steerable regions)
    max_steerability_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Highest conditional steerability probability across all regions",
    )

    built_at: datetime = Field(default_factory=utc_now)


# ── Transition explanation ────────────────────────────────────────────────────


class TransitionExplanation(EOSBaseModel):
    """
    Structured explanation of a single deterministic → influence-permissive shift.

    This is the Phase 4 exit criterion artefact:
    given a runtime trace + context, output:
      1. which invariants were broken,
      2. which constraints were relaxed as a result,
      3. which new CFG region became reachable (the steerable region).

    Designed to be both machine-readable and human-interpretable.
    """

    explanation_id: str = Field(default_factory=new_id)
    target_id: str
    run_id: str = Field(default="", description="Phase 2 run_id that triggered this explanation")

    # ── Step 1: invariants broken ─────────────────────────────────────────────
    broken_invariants: list[Invariant] = Field(
        default_factory=list,
        description="Invariants observed to be violated in the trace",
    )
    broken_invariant_ids: list[str] = Field(default_factory=list)

    # Violation mechanics observed
    observed_violations: list[ViolationMechanism] = Field(default_factory=list)

    # ── Step 2: constraints relaxed ──────────────────────────────────────────
    relaxed_constraints: list[Constraint] = Field(
        default_factory=list,
        description="Constraints that no longer hold given the broken invariants",
    )
    relaxed_constraint_ids: list[str] = Field(default_factory=list)

    # ── Step 3: new reachable region ─────────────────────────────────────────
    steerable_region: SteerableRegion | None = Field(
        default=None,
        description="The CFG region that becomes reachable under the violated invariants",
    )
    steerable_region_id: str = Field(default="")

    # Matching condition set
    triggered_condition_set: ConditionSet | None = Field(
        default=None,
        description="The ConditionSet whose trigger predicate was satisfied",
    )
    condition_set_id: str = Field(default="")

    # ── Summary ───────────────────────────────────────────────────────────────
    steerability_class: SteerabilityClass = SteerabilityClass.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    narrative: str = Field(
        default="",
        description=(
            "One-paragraph prose explanation suitable for a research report: "
            "'When invariant X is violated at function F, constraint C is relaxed, "
            "allowing execution to reach region R where external input controls the "
            "branching outcome.'"
        ),
    )

    # Supporting evidence
    supporting_fragment_ids: list[str] = Field(default_factory=list)
    supporting_run_ids: list[str] = Field(default_factory=list)

    produced_at: datetime = Field(default_factory=utc_now)


# ── Phase 4 result container ──────────────────────────────────────────────────


class Phase4Result(EOSBaseModel):
    """
    Top-level output of a Phase 4 constraint reasoning session.

    Wraps the SteerabilityModel and all TransitionExplanations with
    aggregate statistics suitable for logging and reporting.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    # Core artefact
    model: SteerabilityModel

    # Per-run explanations (one per failure/interesting run analysed)
    explanations: list[TransitionExplanation] = Field(default_factory=list)

    # Aggregate statistics
    total_state_variables:       int = 0
    total_invariants:            int = 0
    total_constraints:           int = 0
    total_steerable_regions:     int = 0
    total_condition_sets:        int = 0
    total_explanations:          int = 0
    total_broken_invariants:     int = 0

    # Overall classification
    target_steerability_class: SteerabilityClass = SteerabilityClass.UNKNOWN
    max_steerability_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    # Phase 4 exit criterion
    exit_criterion_met: bool = Field(
        default=False,
        description=(
            "True when, given a trace + context, the analyzer has produced at "
            "least one TransitionExplanation identifying broken invariants, "
            "relaxed constraints, and a new reachable steerable region."
        ),
    )

    produced_at: datetime = Field(default_factory=utc_now)
