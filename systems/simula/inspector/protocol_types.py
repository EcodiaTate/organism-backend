"""
EcodiaOS - Inspector Phase 6: Protocol State-Machine Stress Types

All domain models for deep protocol state-machine boundary analysis.

Design philosophy
-----------------
Phase 6 answers the question: "where do specification-heavy protocol handlers
fail when driven through valid but rare state-machine paths?"

It models protocol handlers as explicit finite state machines, generates
structured mutation strategies targeting boundary conditions, and records
failure-at-boundary findings with full state-path context.

The focus is *valid* edge-case evolution: sequences that follow the protocol
specification but reach states that implementations handle inconsistently -
numeric overflow at sequence counters, timer underrun at re-keying windows,
layered interpretation mismatches at protocol version negotiation, and
desynchronisation between parser and handler state machines.

Layer map
---------
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 4 (SteerabilityModel, StateVariable[PROTOCOL_STATE])              │
  │  Phase 5 (TrustGraph - CREDENTIAL/SESSION nodes as protocol contexts)    │
  │    ↓  ProtocolFsmBuilder                                                 │
  │  ProtocolFsm   - explicit states, transitions, guards, counters, timers  │
  │    ↓  BoundaryStressEngine                                               │
  │  StressScenario - state trace driving a rare/boundary transition         │
  │    ↓  scenario execution / replay                                        │
  │  BoundaryFailure - observed anomaly with state-path provenance           │
  │    ↓                                                                     │
  │  StateCoverageReport  - which states/transitions were exercised          │
  │  FailureAtBoundaryDataset - state path → inconsistency → anomaly         │
  │    ↓                                                                     │
  │  Phase6Result - top-level output with exit criterion                     │
  └──────────────────────────────────────────────────────────────────────────┘

Exit criterion
--------------
Phase6Result.exit_criterion_met = True when:
- ≥1 StressScenario successfully exercised a rare/boundary transition, AND
- ≥1 BoundaryFailure linked to a state-path mismatch has been recorded, AND
- ≥1 FailureAtBoundaryDataset entry connects a state path → inconsistent
  transition → observed anomaly.

Key concepts
------------
ProtocolFsmState      - a node in the FSM with optional numeric counters,
                        timer slots, and layer-stack depth
ProtocolTransition    - a directed edge with guard predicates, optional
                        counter increments, and an interpretation layer
BoundaryKind          - taxonomy of boundary condition categories
BoundaryFailure       - a failure observed during scenario replay, with full
                        state-path provenance and the mismatch classification
StateCoverageReport   - which states and transitions were exercised, and
                        which remain uncovered
ScenarioLibrary       - a collection of StressScenarios indexed by boundary
                        kind; the primary Phase 6 deliverable
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ── Enumerations ───────────────────────────────────────────────────────────────


class ProtocolFamily(enum.StrEnum):
    """
    High-level family of specification-heavy protocols targeted by Phase 6.

    NETWORK_HANDSHAKE  - TLS, DTLS, QUIC, SSH handshake state machines
    SESSION_LAYER      - HTTP/1.1 keep-alive, HTTP/2 stream multiplexing, WebSocket
    AUTHENTICATION     - OAuth2, SAML, Kerberos, FIDO2/WebAuthn
    BINARY_FRAMING     - protobuf RPC, MessagePack, Cap'n Proto, FlatBuffers
    CUSTOM_BINARY      - proprietary binary protocols with explicit spec
    TEXTUAL            - SMTP, IMAP, FTP, SIP - line-oriented command protocols
    UNKNOWN            - unrecognised or inferred
    """

    NETWORK_HANDSHAKE = "network_handshake"
    SESSION_LAYER     = "session_layer"
    AUTHENTICATION    = "authentication"
    BINARY_FRAMING    = "binary_framing"
    CUSTOM_BINARY     = "custom_binary"
    TEXTUAL           = "textual"
    UNKNOWN           = "unknown"


class BoundaryKind(enum.StrEnum):
    """
    Category of the boundary condition being stressed.

    SEQUENCE_COUNTER_OVERFLOW   - counter wraps or overflows at numeric limit
    SEQUENCE_COUNTER_RESET      - counter resets to zero mid-session unexpectedly
    TIMER_EXPIRY_AT_BOUNDARY    - timer fires exactly at a state-transition window
    TIMER_UNDERRUN              - operation completes before timer can enforce a guard
    REKEY_WINDOW_OVERLAP        - re-keying initiated while previous material is still live
    VERSION_NEGOTIATION_MISMATCH - two layers disagree on negotiated version
    FRAGMENTATION_BOUNDARY      - message split across the exact MTU boundary
    LAYER_DESYNC                - parser and handler FSMs diverge at a shared state
    MULTIPLEXING_EDGE           - concurrent streams interact at stream-limit boundary
    RETRY_AMPLIFICATION         - retry logic doubles load at exactly the retry count limit
    AUTH_WINDOW_EXPIRY          - credentials expire during an in-flight operation
    PADDING_ORACLE_BOUNDARY     - padding validation at exact block-size boundaries
    NUMERIC_EDGE                - any other integer overflow/underflow/modulo wrap
    UNKNOWN                     - unclassified boundary
    """

    SEQUENCE_COUNTER_OVERFLOW   = "sequence_counter_overflow"
    SEQUENCE_COUNTER_RESET      = "sequence_counter_reset"
    TIMER_EXPIRY_AT_BOUNDARY    = "timer_expiry_at_boundary"
    TIMER_UNDERRUN              = "timer_underrun"
    REKEY_WINDOW_OVERLAP        = "rekey_window_overlap"
    VERSION_NEGOTIATION_MISMATCH = "version_negotiation_mismatch"
    FRAGMENTATION_BOUNDARY      = "fragmentation_boundary"
    LAYER_DESYNC                = "layer_desync"
    MULTIPLEXING_EDGE           = "multiplexing_edge"
    RETRY_AMPLIFICATION         = "retry_amplification"
    AUTH_WINDOW_EXPIRY          = "auth_window_expiry"
    PADDING_ORACLE_BOUNDARY     = "padding_oracle_boundary"
    NUMERIC_EDGE                = "numeric_edge"
    UNKNOWN                     = "unknown"


class TransitionInterpretation(enum.StrEnum):
    """
    Which layer of the protocol stack interprets a given transition.

    When PARSER and HANDLER disagree on how to interpret the same message at a
    state boundary, a desync failure occurs.

    PARSER         - low-level framing / tokenisation layer
    HANDLER        - business logic / session management layer
    CRYPTO         - cryptographic record layer
    TRANSPORT      - TCP/UDP/QUIC layer
    APPLICATION    - application-level routing / dispatch
    NEGOTIATION    - version / feature negotiation sub-protocol
    """

    PARSER        = "parser"
    HANDLER       = "handler"
    CRYPTO        = "crypto"
    TRANSPORT     = "transport"
    APPLICATION   = "application"
    NEGOTIATION   = "negotiation"
    UNKNOWN       = "unknown"


class ScenarioResult(enum.StrEnum):
    """
    The outcome of executing a StressScenario against a target.

    BOUNDARY_FAILURE  - execution diverged from specification at a boundary
    DESYNC_DETECTED   - parser and handler reached different states
    CRASH             - target crashed (unhandled exception / panic)
    TIMEOUT           - target became unresponsive within the scenario
    CLEAN             - scenario completed without observable anomaly
    NOT_REACHED       - target did not reach the boundary state being stressed
    ERROR             - scenario execution error (infrastructure / setup)
    """

    BOUNDARY_FAILURE = "boundary_failure"
    DESYNC_DETECTED  = "desync_detected"
    CRASH            = "crash"
    TIMEOUT          = "timeout"
    CLEAN            = "clean"
    NOT_REACHED      = "not_reached"
    ERROR            = "error"


class MutationStrategy(enum.StrEnum):
    """
    Strategy used to generate a StressScenario from an FSM path.

    COUNTER_MAXIMISE    - drive counter to its declared maximum value
    COUNTER_OVERFLOW    - drive counter one past its maximum (overflow/wrap)
    COUNTER_RESET_MID   - reset counter to zero mid-session
    TIMER_BOUNDARY      - schedule messages to arrive exactly at timer expiry
    TIMER_UNDERRUN      - complete session before timer can fire
    REKEY_RACE          - initiate re-key while previous key is in use
    VERSION_DOWNGRADE   - negotiate down to oldest supported version
    VERSION_UNKNOWN     - offer an unsupported version identifier
    FRAGMENT_AT_MTU     - split message exactly at declared MTU boundary
    LAYER_SKIP          - omit an optional but assumed state and continue
    STREAM_LIMIT        - open streams up to the declared maximum
    STREAM_OVER_LIMIT   - open one stream beyond the declared maximum
    AUTH_EXPIRE         - let credentials expire between request and response
    PADDING_EXACT       - send message whose length is exactly a block boundary
    """

    COUNTER_MAXIMISE   = "counter_maximise"
    COUNTER_OVERFLOW   = "counter_overflow"
    COUNTER_RESET_MID  = "counter_reset_mid"
    TIMER_BOUNDARY     = "timer_boundary"
    TIMER_UNDERRUN     = "timer_underrun"
    REKEY_RACE         = "rekey_race"
    VERSION_DOWNGRADE  = "version_downgrade"
    VERSION_UNKNOWN    = "version_unknown"
    FRAGMENT_AT_MTU    = "fragment_at_mtu"
    LAYER_SKIP         = "layer_skip"
    STREAM_LIMIT       = "stream_limit"
    STREAM_OVER_LIMIT  = "stream_over_limit"
    AUTH_EXPIRE        = "auth_expire"
    PADDING_EXACT      = "padding_exact"


# ── FSM primitives ─────────────────────────────────────────────────────────────


class FsmCounter(EOSBaseModel):
    """
    A numeric counter associated with an FSM state.

    Counters represent sequence numbers, stream IDs, retry counts, fragment
    offsets, and similar monotonic or bounded values that drive transitions.

    The boundary value (max_value) is the limit declared in the protocol spec;
    behaviour at max_value ± 1 is the primary stress target.
    """

    counter_id: str = Field(default_factory=new_id)
    name: str = Field(..., description="e.g. 'seq_num', 'stream_id', 'retry_count'")
    current_value: int = 0
    min_value: int = 0
    max_value: int = Field(
        default=2**32 - 1,
        description="Declared spec maximum; wrap/overflow boundary for stress",
    )
    wraps: bool = Field(
        default=False,
        description="True if the counter wraps to min_value on overflow (per spec)",
    )
    increment: int = Field(
        default=1,
        description="Nominal step size per transition",
    )
    spec_reference: str = Field(
        default="",
        description="Section of the protocol specification declaring this counter",
    )


class FsmTimer(EOSBaseModel):
    """
    A timer slot associated with an FSM state.

    Timers represent retransmit timeouts, authentication windows, re-keying
    intervals, and keep-alive deadlines.  A timer fires when elapsed_ms ≥ timeout_ms.

    The boundary condition is firing at exactly elapsed_ms == timeout_ms - the
    zero-slack case - which implementations often handle differently from
    the elapsed_ms > timeout_ms case.
    """

    timer_id: str = Field(default_factory=new_id)
    name: str = Field(..., description="e.g. 'retransmit_timer', 'auth_window', 'rekey_interval'")
    timeout_ms: int = Field(..., description="Nominal timeout in milliseconds")
    elapsed_ms: int = 0
    is_active: bool = False
    fires_on_expiry: str = Field(
        default="",
        description="Name of the transition that fires when the timer expires",
    )
    spec_reference: str = Field(default="")


class ProtocolFsmState(EOSBaseModel):
    """
    A single state in a protocol FSM.

    States carry counters, timers, and layer-stack depth - the three
    dimensions that generate boundary conditions.

    A state is a boundary state if it has at least one counter near its
    maximum, at least one active timer near expiry, or sits at the layer
    boundary of two interpretation layers.
    """

    state_id: str = Field(default_factory=new_id)
    fsm_id: str
    name: str

    # Protocol-layer position
    layer: TransitionInterpretation = TransitionInterpretation.UNKNOWN

    # Numeric state
    counters: list[FsmCounter] = Field(default_factory=list)
    timers: list[FsmTimer] = Field(default_factory=list)

    # Layer-stack nesting depth (for multiplexed/nested protocols)
    layer_depth: int = 0

    # Whether this state is an FSM entry or terminal
    is_initial: bool = False
    is_terminal: bool = False

    # Whether this state was identified as a boundary state during FSM construction
    is_boundary: bool = False

    # Outgoing transition IDs
    outgoing_transition_ids: list[str] = Field(default_factory=list)

    # Source evidence
    derived_from_fragment_ids: list[str] = Field(default_factory=list)
    derived_from_state_variable_ids: list[str] = Field(default_factory=list)

    # Human-readable description
    description: str = Field(default="")
    spec_reference: str = Field(default="")

    metadata: dict[str, Any] = Field(default_factory=dict)


class TransitionGuard(EOSBaseModel):
    """
    A predicate that must hold for a protocol transition to fire.

    Guards encode the specification's preconditions:
      - counter in valid range
      - timer active and not expired
      - negotiated version matches expected
      - layer depth at expected level

    The stress engine generates scenarios that drive guards to their exact
    limits (guard_value ± 1) to expose interpretation mismatches.
    """

    guard_id: str = Field(default_factory=new_id)
    description: str = Field(..., description="Human-readable predicate, e.g. 'seq_num < 2^32 - 1'")

    # Formal guard type
    guard_kind: str = Field(
        default="",
        description=(
            "counter_range | timer_active | timer_expired | version_match "
            "| layer_depth | custom"
        ),
    )

    # Numeric boundary (if applicable)
    operand_name: str = Field(default="", description="Name of counter or timer this guard tests")
    boundary_value: int | None = None
    operator: str = Field(default="", description="< | <= | == | >= | > | !=")

    # Whether violating this guard is the stress target
    is_stress_target: bool = False

    spec_reference: str = Field(default="")


class ProtocolTransition(EOSBaseModel):
    """
    A directed edge in a protocol FSM.

    Transitions carry:
    - an optional guard (precondition from spec)
    - counter mutations (increment/reset/set)
    - an interpretation layer (which stack layer handles this message)
    - a boundary flag when the transition is a stress target

    Boundary transitions are those where at least one guard is at its
    declared limit, or where the interpretation layer differs from the
    adjacent state's layer (creating a desync opportunity).
    """

    transition_id: str = Field(default_factory=new_id)
    fsm_id: str

    from_state_id: str
    to_state_id: str

    # Symbolic name of the message/event that fires this transition
    event_name: str = Field(..., description="e.g. 'ClientHello', 'ACK', 'SETTINGS_ACK'")

    # Guards
    guards: list[TransitionGuard] = Field(default_factory=list)

    # Which layer handles this transition (for desync detection)
    interpretation_layer: TransitionInterpretation = TransitionInterpretation.UNKNOWN

    # Counter mutations that fire with this transition
    counter_increments: dict[str, int] = Field(
        default_factory=dict,
        description="counter_name → delta (positive = increment, negative = decrement, 0 = reset)",
    )

    # Whether this is a rare / low-frequency transition in normal operation
    is_rare: bool = False

    # Whether this transition crosses an interpretation layer boundary
    is_layer_boundary: bool = False

    # Whether this transition can be the site of a numeric boundary
    is_numeric_boundary: bool = False

    # Confidence that this transition exists in the target implementation
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # Source evidence
    derived_from_fragment_ids: list[str] = Field(default_factory=list)
    spec_reference: str = Field(default="")
    description: str = Field(default="")


class ProtocolFsm(EOSBaseModel):
    """
    A complete finite state machine for one protocol handler.

    Produced by ProtocolFsmBuilder from Phase 4/5 artifacts + optional
    supplementary spec data.

    The FSM is the substrate for boundary stress scenario generation.
    """

    fsm_id: str = Field(default_factory=new_id)
    target_id: str

    protocol_family: ProtocolFamily = ProtocolFamily.UNKNOWN
    name: str = Field(..., description="Descriptive name, e.g. 'TLS 1.3 handshake'")

    states: dict[str, ProtocolFsmState] = Field(
        default_factory=dict, description="state_id → ProtocolFsmState"
    )
    transitions: dict[str, ProtocolTransition] = Field(
        default_factory=dict, description="transition_id → ProtocolTransition"
    )

    # Adjacency index: state_id → list[transition_id] leaving that state
    out_transitions: dict[str, list[str]] = Field(default_factory=dict)
    # Adjacency index: state_id → list[transition_id] arriving at that state
    in_transitions: dict[str, list[str]] = Field(default_factory=dict)

    # Quick lookup
    initial_state_id: str = Field(default="")
    terminal_state_ids: list[str] = Field(default_factory=list)
    boundary_state_ids: list[str] = Field(default_factory=list)
    boundary_transition_ids: list[str] = Field(default_factory=list)

    # Statistics
    total_states: int = 0
    total_transitions: int = 0
    total_boundary_states: int = 0
    total_boundary_transitions: int = 0

    built_at: datetime = Field(default_factory=utc_now)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def add_state(self, state: ProtocolFsmState) -> None:
        self.states[state.state_id] = state
        self.total_states = len(self.states)
        if state.is_boundary and state.state_id not in self.boundary_state_ids:
            self.boundary_state_ids.append(state.state_id)
            self.total_boundary_states = len(self.boundary_state_ids)
        if state.is_initial:
            self.initial_state_id = state.state_id

    def add_transition(self, transition: ProtocolTransition) -> None:
        self.transitions[transition.transition_id] = transition
        self.total_transitions = len(self.transitions)

        self.out_transitions.setdefault(transition.from_state_id, []).append(
            transition.transition_id
        )
        self.in_transitions.setdefault(transition.to_state_id, []).append(
            transition.transition_id
        )

        if transition.transition_id not in self.boundary_transition_ids:
            if transition.is_layer_boundary or transition.is_numeric_boundary or transition.is_rare:
                self.boundary_transition_ids.append(transition.transition_id)
                self.total_boundary_transitions = len(self.boundary_transition_ids)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def successors(self, state_id: str) -> list[tuple[ProtocolTransition, ProtocolFsmState]]:
        """Return (transition, destination_state) pairs from state_id."""
        result = []
        for tid in self.out_transitions.get(state_id, []):
            t = self.transitions.get(tid)
            if t and t.to_state_id in self.states:
                result.append((t, self.states[t.to_state_id]))
        return result

    def boundary_states(self) -> list[ProtocolFsmState]:
        return [self.states[sid] for sid in self.boundary_state_ids if sid in self.states]

    def boundary_transitions(self) -> list[ProtocolTransition]:
        return [self.transitions[tid] for tid in self.boundary_transition_ids if tid in self.transitions]


# ── Stress scenario ────────────────────────────────────────────────────────────


class StateStep(EOSBaseModel):
    """
    A single step in a stress scenario state trace.

    Records the transition fired, the resulting state, and the counter/timer
    values at that point - forming a complete, reproducible state trace.
    """

    step_index: int
    transition_id: str
    event_name: str
    from_state_id: str
    from_state_name: str
    to_state_id: str
    to_state_name: str
    interpretation_layer: TransitionInterpretation

    # Counter and timer values at this step (snapshot)
    counter_values: dict[str, int] = Field(default_factory=dict)
    active_timers: list[str] = Field(default_factory=list)

    # Whether this step is the boundary being stressed
    is_stress_point: bool = False

    # Any guard being tested at this step
    guard_id: str = Field(default="")
    guard_boundary_value: int | None = None


class StressScenario(EOSBaseModel):
    """
    A complete state-machine scenario designed to exercise a specific boundary.

    A scenario is a valid (specification-conformant) state trace that drives
    the FSM to a boundary condition.  It is *not* a raw malformed packet
    sequence - the validity constraint is fundamental.

    Scenarios are produced by BoundaryStressEngine using a MutationStrategy
    applied to a path in the FSM.
    """

    scenario_id: str = Field(default_factory=new_id)
    target_id: str
    fsm_id: str

    # Classification
    boundary_kind: BoundaryKind
    mutation_strategy: MutationStrategy

    # Protocol context
    protocol_family: ProtocolFamily = ProtocolFamily.UNKNOWN
    protocol_name: str = Field(default="")

    # State trace
    steps: list[StateStep] = Field(default_factory=list)
    state_path: list[str] = Field(
        default_factory=list,
        description="Ordered list of state_ids visited",
    )
    transition_path: list[str] = Field(
        default_factory=list,
        description="Ordered list of transition_ids fired",
    )

    # The boundary being exercised
    stress_transition_id: str = Field(default="")
    stress_state_id: str = Field(default="")
    stress_counter_name: str = Field(default="")
    stress_counter_value: int | None = None
    stress_timer_name: str = Field(default="")

    # Human-readable description of what boundary this exercises
    description: str = Field(
        default="",
        description=(
            "One-sentence description: 'Drive seq_num to 2^32-1 via ESTABLISHED "
            "→ DATA_TRANSFER → WRAPPING, testing counter wrap in TCP.'"
        ),
    )

    # Result of executing this scenario (if run)
    result: ScenarioResult = ScenarioResult.NOT_REACHED
    result_detail: str = Field(default="")

    # Confidence this scenario will surface a real boundary failure
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Source evidence
    derived_from_fragment_ids: list[str] = Field(default_factory=list)
    derived_from_condition_set_ids: list[str] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=utc_now)


# ── Boundary failure ───────────────────────────────────────────────────────────


class InterpretationMismatch(EOSBaseModel):
    """
    An observed disagreement between two protocol layers at a boundary.

    The defining finding of Phase 6: two layers of the protocol stack
    interpret the same message or state differently.  When this occurs at
    a boundary transition, the result is desynchronisation, incorrect error
    responses, or security-relevant state corruption.
    """

    mismatch_id: str = Field(default_factory=new_id)
    target_id: str

    # The two layers in disagreement
    layer_a: TransitionInterpretation
    layer_b: TransitionInterpretation

    # The transition and state where the disagreement was observed
    transition_id: str
    state_id: str

    # What each layer believed the state to be
    layer_a_interpretation: str = Field(default="")
    layer_b_interpretation: str = Field(default="")

    # What inconsistency results
    inconsistency_description: str = Field(
        default="",
        description=(
            "One-sentence description of the divergence: "
            "'PARSER accepted frame as valid DATA but HANDLER was in CLOSE_WAIT.'"
        ),
    )

    boundary_kind: BoundaryKind = BoundaryKind.UNKNOWN
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class BoundaryFailure(EOSBaseModel):
    """
    An anomaly observed when executing a StressScenario at a state boundary.

    This is the core finding of Phase 6: a failure that can be reproduced
    via state evolution and linked to a specific state-machine mismatch.

    The triad of (state_path, mismatch, anomaly) constitutes a
    FailureAtBoundaryDataset entry.
    """

    failure_id: str = Field(default_factory=new_id)
    target_id: str
    scenario_id: str

    # Classification
    boundary_kind: BoundaryKind
    result: ScenarioResult

    # State provenance
    state_path: list[str] = Field(
        default_factory=list,
        description="Ordered state_ids leading to failure",
    )
    failing_transition_id: str = Field(default="")
    failing_state_id: str = Field(default="")

    # Counter/timer values at the point of failure
    counter_values_at_failure: dict[str, int] = Field(default_factory=dict)
    active_timers_at_failure: list[str] = Field(default_factory=list)

    # Observed anomaly
    anomaly_description: str = Field(
        default="",
        description="What went wrong: exception type, invalid state, wrong error code, etc.",
    )
    stack_trace: str = Field(default="")
    error_code: str = Field(default="")

    # Interpretation mismatch (if applicable)
    mismatch: InterpretationMismatch | None = None

    # Impact assessment
    is_security_relevant: bool = False
    security_impact_description: str = Field(default="")
    boundary_kind_confirmed: bool = False

    # Reproducibility
    is_reproducible: bool = True
    reproduction_script: str = Field(
        default="",
        description="State evolution script (pseudo-code) to reproduce this failure",
    )

    # Confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    observed_at: datetime = Field(default_factory=utc_now)


# ── Failure-at-boundary dataset ────────────────────────────────────────────────


class FailureAtBoundaryEntry(EOSBaseModel):
    """
    A single entry in the FailureAtBoundaryDataset.

    Triad:  state_path  →  inconsistent_transition  →  observed_anomaly

    This is the structured deliverable linking execution context to
    specification mismatch to observable failure.
    """

    entry_id: str = Field(default_factory=new_id)
    target_id: str

    # State path context
    state_path: list[str] = Field(description="Ordered state names (not IDs) for human readability")
    state_path_ids: list[str] = Field(default_factory=list)
    boundary_state_name: str = Field(default="")

    # Inconsistent transition
    transition_event_name: str = Field(default="")
    transition_id: str = Field(default="")
    mismatch: InterpretationMismatch | None = None

    # Observed anomaly
    failure: BoundaryFailure

    # Classification
    boundary_kind: BoundaryKind
    mutation_strategy: MutationStrategy
    protocol_family: ProtocolFamily = ProtocolFamily.UNKNOWN

    # Cross-references to scenarios and Phase 4/5 artifacts
    scenario_id: str = Field(default="")
    condition_set_ids: list[str] = Field(default_factory=list)
    corridor_ids: list[str] = Field(default_factory=list)

    is_security_relevant: bool = False

    recorded_at: datetime = Field(default_factory=utc_now)


class FailureAtBoundaryDataset(EOSBaseModel):
    """
    The complete failure-at-boundary dataset for one analysis target.

    Produced by ProtocolAnalyzer.  Primary Phase 6 deliverable alongside
    the ScenarioLibrary and StateCoverageReport.

    Each entry traces a state path → inconsistent transition → observed anomaly.
    """

    dataset_id: str = Field(default_factory=new_id)
    target_id: str

    entries: list[FailureAtBoundaryEntry] = Field(default_factory=list)

    total_entries: int = 0
    security_relevant_entries: int = 0

    # Index by boundary kind for quick filtering
    entries_by_boundary_kind: dict[str, list[str]] = Field(
        default_factory=dict, description="BoundaryKind.value → list[entry_id]"
    )

    built_at: datetime = Field(default_factory=utc_now)


# ── State coverage report ──────────────────────────────────────────────────────


class TransitionCoverageRecord(EOSBaseModel):
    """Coverage record for a single FSM transition."""

    transition_id: str
    event_name: str
    from_state_name: str
    to_state_name: str
    times_exercised: int = 0
    boundary_kind_exercised: BoundaryKind | None = None
    is_boundary: bool = False
    was_covered: bool = False


class StateCoverageReport(EOSBaseModel):
    """
    A report on which FSM states and transitions were exercised during
    Phase 6 boundary stress.

    Distinguishes boundary vs non-boundary coverage, and identifies
    which rare transitions remain unexplored.
    """

    report_id: str = Field(default_factory=new_id)
    target_id: str
    fsm_id: str

    # State coverage
    total_states: int = 0
    covered_states: int = 0
    boundary_states: int = 0
    covered_boundary_states: int = 0
    uncovered_state_ids: list[str] = Field(default_factory=list)
    uncovered_boundary_state_ids: list[str] = Field(default_factory=list)

    # Transition coverage
    total_transitions: int = 0
    covered_transitions: int = 0
    boundary_transitions: int = 0
    covered_boundary_transitions: int = 0
    uncovered_transition_ids: list[str] = Field(default_factory=list)
    uncovered_boundary_transition_ids: list[str] = Field(default_factory=list)

    # Per-transition detail
    transition_records: list[TransitionCoverageRecord] = Field(default_factory=list)

    # Coverage ratios
    state_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    transition_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    boundary_state_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    boundary_transition_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    produced_at: datetime = Field(default_factory=utc_now)


# ── Scenario library ──────────────────────────────────────────────────────────


class ScenarioLibrary(EOSBaseModel):
    """
    A structured library of StressScenarios indexed by BoundaryKind.

    This is the second primary Phase 6 deliverable alongside
    FailureAtBoundaryDataset.

    The library is not a collection of raw malformed bytes - it is a
    collection of *state traces* (valid execution sequences) that each
    exercise a specific boundary condition.
    """

    library_id: str = Field(default_factory=new_id)
    target_id: str

    # All scenarios by ID
    scenarios: dict[str, StressScenario] = Field(
        default_factory=dict, description="scenario_id → StressScenario"
    )

    # Index by BoundaryKind
    scenarios_by_boundary_kind: dict[str, list[str]] = Field(
        default_factory=dict, description="BoundaryKind.value → list[scenario_id]"
    )

    # Index by MutationStrategy
    scenarios_by_strategy: dict[str, list[str]] = Field(
        default_factory=dict, description="MutationStrategy.value → list[scenario_id]"
    )

    # Statistics
    total_scenarios: int = 0
    scenarios_with_failures: int = 0
    scenarios_clean: int = 0

    built_at: datetime = Field(default_factory=utc_now)

    def add_scenario(self, scenario: StressScenario) -> None:
        self.scenarios[scenario.scenario_id] = scenario
        self.total_scenarios = len(self.scenarios)

        bk = scenario.boundary_kind.value
        self.scenarios_by_boundary_kind.setdefault(bk, []).append(scenario.scenario_id)

        ms = scenario.mutation_strategy.value
        self.scenarios_by_strategy.setdefault(ms, []).append(scenario.scenario_id)

        if scenario.result in (ScenarioResult.BOUNDARY_FAILURE, ScenarioResult.DESYNC_DETECTED,
                                ScenarioResult.CRASH):
            self.scenarios_with_failures += 1
        elif scenario.result == ScenarioResult.CLEAN:
            self.scenarios_clean += 1


# ── Phase 6 result ─────────────────────────────────────────────────────────────


class Phase6Result(EOSBaseModel):
    """
    Top-level output of a Phase 6 protocol state-machine stress session.

    Wraps the FSMs built, the ScenarioLibrary, the FailureAtBoundaryDataset,
    and the StateCoverageReports, with aggregate statistics and the exit
    criterion flag.

    Exit criterion
    --------------
    exit_criterion_met = True when:
    - ≥1 StressScenario exercised a rare/boundary transition, AND
    - ≥1 BoundaryFailure linked to a state-path mismatch has been recorded, AND
    - ≥1 FailureAtBoundaryEntry connects: state path → inconsistent transition
      → observed anomaly.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    # Core artefacts
    fsms: list[ProtocolFsm] = Field(default_factory=list)
    scenario_library: ScenarioLibrary
    failure_dataset: FailureAtBoundaryDataset
    coverage_reports: list[StateCoverageReport] = Field(default_factory=list)

    # All boundary failures (flattened for quick access)
    boundary_failures: list[BoundaryFailure] = Field(default_factory=list)

    # Aggregate statistics
    total_fsms: int = 0
    total_states: int = 0
    total_transitions: int = 0
    total_boundary_states: int = 0
    total_boundary_transitions: int = 0
    total_scenarios: int = 0
    total_boundary_failures: int = 0
    security_relevant_failures: int = 0
    total_interpretation_mismatches: int = 0

    # Coverage summary
    mean_state_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_boundary_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # Exit criterion
    exit_criterion_met: bool = Field(
        default=False,
        description=(
            "True when: ≥1 boundary transition exercised, ≥1 BoundaryFailure "
            "with state-path mismatch recorded, and ≥1 FailureAtBoundaryEntry "
            "connecting state path → inconsistent transition → anomaly."
        ),
    )

    produced_at: datetime = Field(default_factory=utc_now)
