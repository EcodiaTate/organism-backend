"""
EcodiaOS - Inspector Phase 6: Protocol FSM Builder

Builds a ProtocolFsm from Phase 4 and Phase 5 artifacts.

Three sources are combined:
1. Phase 4 StateVariable[PROTOCOL_STATE] - directly encodes FSM nodes and
   the state variables that control transitions.
2. Phase 3 CodeFragment with FragmentSemantics.CONTROL_FLOW - infers
   transitions from branching patterns near protocol-state variables.
3. Phase 5 TrustGraph CREDENTIAL/SESSION nodes - identifies protocol
   contexts whose boundaries are trust-relevant (re-keying windows,
   auth-token expiry, session handshakes).

Each source contributes states and transitions; they are merged into a
single ProtocolFsm per identified protocol context.

Four-pass construction
----------------------
Pass 1  Extract protocol-state variables from Phase 4 StateModel.
        Each distinct PROTOCOL_STATE variable that ranges over an enum
        or bounded integer becomes an FSM with one state per value.

Pass 2  Map Phase 3 control-flow fragments to FSM transitions.
        Branching fragments adjacent to PROTOCOL_STATE reads are inferred
        as transition guards.  Layer is inferred from fragment semantics.

Pass 3  Inject Phase 5 credential/session context.
        SESSION and CREDENTIAL nodes whose privilege_value > threshold
        get timer and counter decorations from their trust-edge metadata.

Pass 4  Classify boundary states and transitions.
        A state is a boundary state if any counter is within 1 of its
        max_value, any timer has a threshold at spec boundary, or it sits
        at a layer-interpretation crossing.
        A transition is a boundary transition if it has a guard whose
        boundary_value is declared, is_rare, or crosses an interpretation
        layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.protocol_types import (
    BoundaryKind,
    FsmCounter,
    FsmTimer,
    MutationStrategy,
    ProtocolFamily,
    ProtocolFsm,
    ProtocolFsmState,
    ProtocolTransition,
    TransitionInterpretation,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import (
        Phase4Result,
        StateVariable,
    )
    from systems.simula.inspector.static_types import (
        Phase3Result,
    )
    from systems.simula.inspector.trust_types import (
        Phase5Result,
    )

logger = structlog.get_logger().bind(system="simula.inspector.protocol_state_machine")

# Heuristic: counter/field names that suggest a sequence number or stream ID
_COUNTER_NAME_HINTS: frozenset[str] = frozenset({
    "seq", "sequence", "seq_num", "stream_id", "stream", "fragment_offset",
    "retry", "retry_count", "nonce", "counter", "epoch", "record_seq",
    "msg_seq", "handshake_seq", "ack_num", "window",
})

# Heuristic: timer/field names that suggest a protocol timer
_TIMER_NAME_HINTS: frozenset[str] = frozenset({
    "timeout", "timer", "deadline", "expiry", "ttl", "window", "interval",
    "rekey", "auth_window", "session_expiry", "keepalive", "retransmit",
})

# Map Phase 4 violation mechanism names to ProtocolFamily guesses
_FRAGMENT_SEMANTIC_TO_FAMILY: dict[str, ProtocolFamily] = {
    "tls":      ProtocolFamily.NETWORK_HANDSHAKE,
    "ssl":      ProtocolFamily.NETWORK_HANDSHAKE,
    "quic":     ProtocolFamily.NETWORK_HANDSHAKE,
    "ssh":      ProtocolFamily.NETWORK_HANDSHAKE,
    "http":     ProtocolFamily.SESSION_LAYER,
    "websocket": ProtocolFamily.SESSION_LAYER,
    "oauth":    ProtocolFamily.AUTHENTICATION,
    "saml":     ProtocolFamily.AUTHENTICATION,
    "kerberos": ProtocolFamily.AUTHENTICATION,
    "grpc":     ProtocolFamily.BINARY_FRAMING,
    "protobuf": ProtocolFamily.BINARY_FRAMING,
    "smtp":     ProtocolFamily.TEXTUAL,
    "imap":     ProtocolFamily.TEXTUAL,
    "ftp":      ProtocolFamily.TEXTUAL,
}


def _guess_protocol_family(name: str) -> ProtocolFamily:
    """Heuristic: guess protocol family from a variable or fragment name."""
    lower = name.lower()
    for hint, family in _FRAGMENT_SEMANTIC_TO_FAMILY.items():
        if hint in lower:
            return family
    return ProtocolFamily.UNKNOWN


def _guess_interpretation_layer(fragment_semantics: str) -> TransitionInterpretation:
    """Heuristic: guess interpretation layer from fragment semantics string."""
    s = fragment_semantics.lower()
    if any(k in s for k in ("parse", "frame", "decode", "tokenise", "token")):
        return TransitionInterpretation.PARSER
    if any(k in s for k in ("crypto", "cipher", "encrypt", "decrypt", "hmac", "tls_record")):
        return TransitionInterpretation.CRYPTO
    if any(k in s for k in ("transport", "tcp", "udp", "socket", "recv", "send")):
        return TransitionInterpretation.TRANSPORT
    if any(k in s for k in ("negotiate", "version", "capability", "extension")):
        return TransitionInterpretation.NEGOTIATION
    if any(k in s for k in ("handle", "dispatch", "session", "state_machine", "handler")):
        return TransitionInterpretation.HANDLER
    if any(k in s for k in ("route", "application", "app_layer")):
        return TransitionInterpretation.APPLICATION
    return TransitionInterpretation.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────


class ProtocolFsmBuilder:
    """
    Builds ProtocolFsm instances from Phase 3/4/5 artifacts.

    Stateless - instantiate once, call .build() per target.

    Parameters
    ----------
    min_confidence      - minimum transition confidence to include (default 0.4)
    privilege_threshold - minimum privilege_value for a Phase 5 node to
                          contribute timer/counter decorations (default 30)
    """

    def __init__(
        self,
        min_confidence: float = 0.4,
        privilege_threshold: int = 30,
    ) -> None:
        self._min_confidence = min_confidence
        self._privilege_threshold = privilege_threshold
        self._log = logger

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        target_id: str,
        phase3_result: Phase3Result,
        phase4_result: Phase4Result | None = None,
        phase5_result: Phase5Result | None = None,
    ) -> list[ProtocolFsm]:
        """
        Build all ProtocolFsm instances for target_id.

        Returns one FSM per identified protocol context (may be empty list
        if no protocol-state variables or fragments are found).
        """
        log = self._log.bind(target_id=target_id)
        log.info("fsm_build_started")

        fsm_map: dict[str, ProtocolFsm] = {}

        # Pass 1 - extract from Phase 4 state variables
        if phase4_result is not None:
            self._pass1_state_variables(target_id, phase4_result, fsm_map)

        # Pass 2 - map Phase 3 fragments to transitions
        self._pass2_fragments(target_id, phase3_result, fsm_map)

        # Pass 3 - inject Phase 5 session/credential context
        if phase5_result is not None:
            self._pass3_trust_context(target_id, phase5_result, fsm_map)

        # Pass 4 - classify boundary states and transitions
        for fsm in fsm_map.values():
            self._pass4_classify_boundaries(fsm)

        fsms = list(fsm_map.values())
        log.info(
            "fsm_build_complete",
            fsm_count=len(fsms),
            total_states=sum(f.total_states for f in fsms),
            total_transitions=sum(f.total_transitions for f in fsms),
            total_boundary_states=sum(f.total_boundary_states for f in fsms),
            total_boundary_transitions=sum(f.total_boundary_transitions for f in fsms),
        )
        return fsms

    # ── Pass 1 - Phase 4 state variables ─────────────────────────────────────

    def _pass1_state_variables(
        self,
        target_id: str,
        phase4_result: Phase4Result,
        fsm_map: dict[str, ProtocolFsm],
    ) -> None:
        """
        Extract protocol-state variables from the Phase 4 steerability model
        and create one FSM skeleton per variable.

        Each distinct PROTOCOL_STATE variable that has an enumerated value
        domain (or a bounded integer range) becomes an FSM with one state
        per declared value.
        """
        from systems.simula.inspector.constraint_types import StateVariableKind

        model = phase4_result.model
        for sv in model.state_variables:
            if sv.kind != StateVariableKind.PROTOCOL_STATE:
                continue

            fsm_name = f"{sv.name}_fsm"
            if sv.name in fsm_map:
                continue

            family = _guess_protocol_family(sv.name)

            fsm = ProtocolFsm(
                target_id=target_id,
                protocol_family=family,
                name=fsm_name,
            )

            # Create one state per declared possible value
            possible_values: list[str] = list(sv.possible_values) if sv.possible_values else []
            if not possible_values:
                # Fall back to range-based synthesis if bounds are known
                if sv.lower_bound is not None and sv.upper_bound is not None:
                    span = int(sv.upper_bound) - int(sv.lower_bound)
                    # Synthesize at most 16 intermediate states for large ranges
                    steps = min(span + 1, 16)
                    step_size = max(1, span // (steps - 1)) if steps > 1 else 1
                    possible_values = [
                        str(int(sv.lower_bound) + i * step_size) for i in range(steps)
                    ]
                else:
                    # No value domain - create a minimal two-state FSM
                    possible_values = ["INITIAL", "ACTIVE"]

            prev_state_id: str | None = None
            for i, val in enumerate(possible_values):
                state = ProtocolFsmState(
                    fsm_id=fsm.fsm_id,
                    name=val,
                    layer=TransitionInterpretation.HANDLER,
                    is_initial=(i == 0),
                    is_terminal=(i == len(possible_values) - 1),
                    derived_from_state_variable_ids=[sv.variable_id],
                    spec_reference=sv.source_location,
                )
                fsm.add_state(state)

                if prev_state_id is not None:
                    t = ProtocolTransition(
                        fsm_id=fsm.fsm_id,
                        from_state_id=prev_state_id,
                        to_state_id=state.state_id,
                        event_name=f"ADVANCE_TO_{val}",
                        interpretation_layer=TransitionInterpretation.HANDLER,
                        confidence=0.6,
                        is_rare=(i == len(possible_values) - 1),
                    )
                    fsm.add_transition(t)
                    fsm.states[prev_state_id].outgoing_transition_ids.append(t.transition_id)

                prev_state_id = state.state_id

            # Attach counters from Phase 4 constraints that reference this variable
            self._attach_counters_from_constraints(sv, fsm, phase4_result)

            fsm_map[sv.name] = fsm

    def _attach_counters_from_constraints(
        self,
        sv: StateVariable,
        fsm: ProtocolFsm,
        phase4_result: Phase4Result,
    ) -> None:
        """
        Scan Phase 4 constraints for integer bounds on variables that share
        a name with sv and inject FsmCounter decorations onto the initial state.
        """
        initial_state = next(
            (s for s in fsm.states.values() if s.is_initial), None
        )
        if initial_state is None:
            return

        for cs in phase4_result.condition_sets:
            for constraint in cs.constraints:
                operand = getattr(constraint, "operand_name", "")
                if not operand:
                    continue
                lower = operand.lower()
                if not any(hint in lower for hint in _COUNTER_NAME_HINTS):
                    continue

                bound_val = getattr(constraint, "bound_value", None)
                max_val = int(bound_val) if bound_val is not None else 2**32 - 1

                counter = FsmCounter(
                    name=operand,
                    max_value=max_val,
                    spec_reference=getattr(constraint, "spec_reference", ""),
                )
                initial_state.counters.append(counter)

    # ── Pass 2 - Phase 3 code fragments ──────────────────────────────────────

    def _pass2_fragments(
        self,
        target_id: str,
        phase3_result: Phase3Result,
        fsm_map: dict[str, ProtocolFsm],
    ) -> None:
        """
        Scan Phase 3 code fragments for protocol-related branching patterns.

        Creates a stub FSM for each protocol-family cluster of fragments that
        does not already have an FSM from Pass 1.  Infers transitions from
        adjacent branch fragments.
        """
        from systems.simula.inspector.static_types import FragmentSemantics

        atlas = phase3_result.atlas
        catalog = atlas.fragment_catalog

        # Group fragments by inferred protocol family
        family_fragments: dict[ProtocolFamily, list] = {}
        for frag in catalog.fragments.values():
            if frag.semantics not in (
                FragmentSemantics.CONTROL_FLOW,
                FragmentSemantics.FUNCTION_CALL,
                FragmentSemantics.COMPARISON,
            ):
                continue

            family = _guess_protocol_family(frag.name)
            if family == ProtocolFamily.UNKNOWN:
                # Also check the fragment's function context
                family = _guess_protocol_family(frag.source_function or "")
            if family == ProtocolFamily.UNKNOWN:
                continue

            family_fragments.setdefault(family, []).append(frag)

        for family, fragments in family_fragments.items():
            # Reuse existing FSM for this family if one was created in Pass 1
            existing = next(
                (f for f in fsm_map.values() if f.protocol_family == family), None
            )

            if existing is None:
                fsm = ProtocolFsm(
                    target_id=target_id,
                    protocol_family=family,
                    name=f"{family.value}_inferred_fsm",
                )
                # Create a minimal 3-state skeleton
                initial = ProtocolFsmState(
                    fsm_id=fsm.fsm_id,
                    name="INITIAL",
                    layer=TransitionInterpretation.PARSER,
                    is_initial=True,
                )
                active = ProtocolFsmState(
                    fsm_id=fsm.fsm_id,
                    name="ACTIVE",
                    layer=_guess_interpretation_layer(family.value),
                )
                terminal = ProtocolFsmState(
                    fsm_id=fsm.fsm_id,
                    name="TERMINAL",
                    layer=TransitionInterpretation.HANDLER,
                    is_terminal=True,
                )
                fsm.add_state(initial)
                fsm.add_state(active)
                fsm.add_state(terminal)

                t1 = ProtocolTransition(
                    fsm_id=fsm.fsm_id,
                    from_state_id=initial.state_id,
                    to_state_id=active.state_id,
                    event_name="CONNECT",
                    interpretation_layer=TransitionInterpretation.TRANSPORT,
                    confidence=0.5,
                )
                t2 = ProtocolTransition(
                    fsm_id=fsm.fsm_id,
                    from_state_id=active.state_id,
                    to_state_id=terminal.state_id,
                    event_name="CLOSE",
                    interpretation_layer=TransitionInterpretation.HANDLER,
                    confidence=0.5,
                )
                fsm.add_transition(t1)
                fsm.add_transition(t2)

                fsm_map[f"__fragment_{family.value}"] = fsm
                target_fsm = fsm
            else:
                target_fsm = existing

            # Enrich the FSM: add layer-boundary transitions inferred from fragments
            self._enrich_fsm_from_fragments(target_fsm, fragments)

    def _enrich_fsm_from_fragments(
        self,
        fsm: ProtocolFsm,
        fragments: list,
    ) -> None:
        """
        Scan fragment list for evidence of layer-crossing or numeric-boundary
        transitions and inject them into fsm.
        """
        # Heuristic: consecutive fragments with different layers → layer boundary transition
        states_list = list(fsm.states.values())
        if len(states_list) < 2:
            return

        seen_pairs: set[tuple[str, str]] = set()

        for i in range(len(fragments) - 1):
            frag_a = fragments[i]
            frag_b = fragments[i + 1]

            layer_a = _guess_interpretation_layer(getattr(frag_a, "semantics", "").value
                                                  if hasattr(frag_a, "semantics") else "")
            layer_b = _guess_interpretation_layer(getattr(frag_b, "semantics", "").value
                                                  if hasattr(frag_b, "semantics") else "")

            is_layer_cross = layer_a != layer_b and layer_a != TransitionInterpretation.UNKNOWN

            if not is_layer_cross:
                continue

            # Pick adjacent states in the FSM as the from/to
            # (we can't precisely place without full CFG mapping, so use heuristic indexing)
            from_state = states_list[min(i, len(states_list) - 2)]
            to_state = states_list[min(i + 1, len(states_list) - 1)]

            pair_key = (from_state.state_id, to_state.state_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            event_name = f"LAYER_CROSS_{layer_a.value.upper()}_TO_{layer_b.value.upper()}"
            t = ProtocolTransition(
                fsm_id=fsm.fsm_id,
                from_state_id=from_state.state_id,
                to_state_id=to_state.state_id,
                event_name=event_name,
                interpretation_layer=layer_a,
                is_layer_boundary=True,
                confidence=0.5,
                derived_from_fragment_ids=[frag_a.fragment_id, frag_b.fragment_id],
            )
            fsm.add_transition(t)

    # ── Pass 3 - Phase 5 trust context ────────────────────────────────────────

    def _pass3_trust_context(
        self,
        target_id: str,
        phase5_result: Phase5Result,
        fsm_map: dict[str, ProtocolFsm],
    ) -> None:
        """
        Enrich FSMs with timer and counter decorations derived from
        Phase 5 CREDENTIAL and SESSION trust nodes.

        A SESSION node with privilege_value ≥ threshold contributes a
        timer representing the session/auth window.  A CREDENTIAL node
        with metadata carrying key_expiry or epoch contributes a counter.
        """
        from systems.simula.inspector.trust_types import TrustNodeKind

        graph = phase5_result.trust_graph

        for node in graph.nodes.values():
            if node.privilege_value < self._privilege_threshold:
                continue
            if node.kind not in (TrustNodeKind.SESSION, TrustNodeKind.CREDENTIAL):
                continue

            # Find a matching FSM or the first FSM
            target_fsm = next(
                (f for f in fsm_map.values()
                 if any(
                     node.name.lower() in s.name.lower()
                     for s in f.states.values()
                 )),
                next(iter(fsm_map.values()), None),
            )
            if target_fsm is None:
                continue

            # Inject timer from SESSION node
            if node.kind == TrustNodeKind.SESSION:
                timeout_ms = int(node.metadata.get("session_timeout_ms", 3600_000))
                timer = FsmTimer(
                    name=f"{node.name}_auth_window",
                    timeout_ms=timeout_ms,
                    fires_on_expiry="SESSION_EXPIRED",
                    spec_reference=f"trust_node:{node.node_id}",
                )
                initial = next((s for s in target_fsm.states.values() if s.is_initial), None)
                if initial:
                    initial.timers.append(timer)

            # Inject counter from CREDENTIAL node
            if node.kind == TrustNodeKind.CREDENTIAL:
                epoch_max = int(node.metadata.get("max_epoch", 2**32 - 1))
                counter = FsmCounter(
                    name=f"{node.name}_epoch",
                    max_value=epoch_max,
                    spec_reference=f"trust_node:{node.node_id}",
                )
                initial = next((s for s in target_fsm.states.values() if s.is_initial), None)
                if initial:
                    initial.counters.append(counter)

    # ── Pass 4 - classify boundary states/transitions ─────────────────────────

    def _pass4_classify_boundaries(self, fsm: ProtocolFsm) -> None:
        """
        Mark states and transitions as boundary targets.

        A state is a boundary state if:
        - Any counter's current_value ≥ max_value - 1 (within 1 of overflow), OR
        - Any timer has a spec-declared threshold, OR
        - It is the only state with a specific interpretation layer (crossing point).

        A transition is a boundary transition if:
        - is_layer_boundary, OR
        - is_numeric_boundary (has a guard with a declared boundary_value), OR
        - is_rare.
        """
        # Build layer → states map
        layer_states: dict[str, list[str]] = {}
        for state in fsm.states.values():
            layer_states.setdefault(state.layer.value, []).append(state.state_id)

        # Identify layer-singleton states (only state at that layer → crossing point)
        layer_singletons: set[str] = {
            sid
            for sids in layer_states.values()
            for sid in sids
            if len(sids) == 1
        }

        for state in fsm.states.values():
            has_counter_boundary = any(
                c.current_value >= c.max_value - 1 for c in state.counters
            )
            has_timer = bool(state.timers)
            is_layer_crossing = state.state_id in layer_singletons

            if has_counter_boundary or has_timer or is_layer_crossing:
                state.is_boundary = True
                if state.state_id not in fsm.boundary_state_ids:
                    fsm.boundary_state_ids.append(state.state_id)

        fsm.total_boundary_states = len(fsm.boundary_state_ids)

        for transition in fsm.transitions.values():
            if transition.is_layer_boundary or transition.is_rare or transition.is_numeric_boundary:
                if transition.transition_id not in fsm.boundary_transition_ids:
                    fsm.boundary_transition_ids.append(transition.transition_id)

        fsm.total_boundary_transitions = len(fsm.boundary_transition_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary transition engine - walks the FSM to identify the specific
# rare/boundary paths that the stress engine will mutate.
# ─────────────────────────────────────────────────────────────────────────────


class BoundaryTransitionEngine:
    """
    Traverses a ProtocolFsm and extracts all paths that lead to boundary
    transitions or boundary states.

    Used by BoundaryStressEngine as the path source before mutation.

    Parameters
    ----------
    max_path_length - maximum state hops per path (default 12)
    max_paths       - maximum paths to return per FSM (default 50)
    """

    def __init__(
        self,
        max_path_length: int = 12,
        max_paths: int = 50,
    ) -> None:
        self._max_path_length = max_path_length
        self._max_paths = max_paths

    def extract_boundary_paths(
        self,
        fsm: ProtocolFsm,
    ) -> list[tuple[list[str], list[str], BoundaryKind, MutationStrategy]]:
        """
        Return a list of (state_id_path, transition_id_path, boundary_kind, strategy)
        tuples representing paths that end at a boundary transition or boundary state.

        Performs DFS from the initial state.
        """
        if not fsm.initial_state_id or fsm.initial_state_id not in fsm.states:
            return []

        results: list[tuple[list[str], list[str], BoundaryKind, MutationStrategy]] = []
        visited_paths: set[tuple[str, ...]] = set()

        stack: list[tuple[list[str], list[str]]] = [
            ([fsm.initial_state_id], [])
        ]

        while stack and len(results) < self._max_paths:
            state_path, trans_path = stack.pop()

            if len(state_path) > self._max_path_length:
                continue

            current_state_id = state_path[-1]
            current_state = fsm.states.get(current_state_id)
            if current_state is None:
                continue

            # Check if current state itself is a boundary state
            if current_state.is_boundary and len(state_path) > 1:
                path_key = tuple(state_path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    bk, strategy = self._classify_boundary(current_state, None, trans_path, fsm)
                    results.append((list(state_path), list(trans_path), bk, strategy))

            for transition, next_state in fsm.successors(current_state_id):
                # Check if this transition is a boundary transition
                if transition.transition_id in fsm.boundary_transition_ids:
                    new_state_path = state_path + [next_state.state_id]
                    new_trans_path = trans_path + [transition.transition_id]
                    path_key = tuple(new_state_path)
                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        bk, strategy = self._classify_boundary(
                            current_state, transition, new_trans_path, fsm
                        )
                        results.append((new_state_path, new_trans_path, bk, strategy))

                # Avoid cycles: only continue if state not already in path
                if next_state.state_id not in state_path:
                    stack.append((
                        state_path + [next_state.state_id],
                        trans_path + [transition.transition_id],
                    ))

        return results

    @staticmethod
    def _classify_boundary(
        state: ProtocolFsmState,
        transition: ProtocolTransition | None,
        trans_path: list[str],
        fsm: ProtocolFsm,
    ) -> tuple[BoundaryKind, MutationStrategy]:
        """
        Infer the most appropriate BoundaryKind and MutationStrategy
        for a path ending at state / transition.
        """
        # Check counters in state
        if state.counters:
            c = state.counters[0]
            if c.wraps:
                return BoundaryKind.SEQUENCE_COUNTER_OVERFLOW, MutationStrategy.COUNTER_OVERFLOW
            return BoundaryKind.NUMERIC_EDGE, MutationStrategy.COUNTER_MAXIMISE

        # Check timers
        if state.timers:
            return BoundaryKind.TIMER_EXPIRY_AT_BOUNDARY, MutationStrategy.TIMER_BOUNDARY

        # Check transition properties
        if transition is not None:
            if transition.is_layer_boundary:
                return BoundaryKind.LAYER_DESYNC, MutationStrategy.LAYER_SKIP
            if transition.is_rare:
                return BoundaryKind.SEQUENCE_COUNTER_RESET, MutationStrategy.COUNTER_RESET_MID

        # Check if any prior transition in path is a layer boundary
        for tid in trans_path:
            t = fsm.transitions.get(tid)
            if t and t.is_layer_boundary:
                return BoundaryKind.LAYER_DESYNC, MutationStrategy.LAYER_SKIP

        return BoundaryKind.UNKNOWN, MutationStrategy.COUNTER_MAXIMISE
