"""
EcodiaOS - Inspector Phase 6: Boundary Stress Engine

Generates StressScenarios and produces BoundaryFailure / FailureAtBoundaryEntry
records by replaying scenarios against the FSM model.

Two stages
----------
Stage A - Scenario Generation
  BoundaryStressEngine takes a list of ProtocolFsm instances and, for each
  boundary path identified by BoundaryTransitionEngine, applies the
  MutationStrategy to generate a StressScenario.  The scenario encodes the
  full state trace (StateStep sequence) needed to reach the boundary
  condition.

  The generated scenarios are *valid* - they follow the protocol FSM - but
  they exercise conditions that implementations tend to handle inconsistently:
  counter overflow, timer expiry at zero slack, layer-interpretation crossing,
  version downgrade, etc.

Stage B - Scenario Replay + Failure Detection
  ScenarioReplayer executes each scenario step by step against the FSM,
  mutating counter and timer state according to the MutationStrategy.  It
  detects two classes of failure:
    1. BoundaryFailure - a transition guard is violated (counter overflow,
       timer fires unexpectedly, layer desync detected).
    2. InterpretationMismatch - two consecutive transitions in the same
       scenario belong to different interpretation layers with no declared
       layer-crossing transition bridging them.

  Each failure is wrapped in a FailureAtBoundaryEntry with full provenance.

Coverage tracking
-----------------
StateCoverageTracker records which states and transitions were visited
across all executed scenarios, producing a StateCoverageReport.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.protocol_types import (
    BoundaryFailure,
    BoundaryKind,
    FailureAtBoundaryDataset,
    FailureAtBoundaryEntry,
    FsmCounter,
    InterpretationMismatch,
    MutationStrategy,
    ProtocolFsm,
    ScenarioLibrary,
    ScenarioResult,
    StateCoverageReport,
    StateStep,
    StressScenario,
    TransitionCoverageRecord,
    TransitionInterpretation,
)

if TYPE_CHECKING:
    from systems.simula.inspector.protocol_types import ProtocolFsmState

logger = structlog.get_logger().bind(system="simula.inspector.boundary_stress_engine")


# ── Stage A: Scenario generation ──────────────────────────────────────────────


class BoundaryStressEngine:
    """
    Generate StressScenarios from FSM boundary paths.

    Each MutationStrategy is dispatched to a dedicated _mutate_* method that
    sets counter/timer values and annotates the stress point in the StateStep
    sequence.

    Parameters
    ----------
    max_scenarios_per_fsm  - cap on scenarios generated per FSM (default 20)
    confidence_floor       - minimum scenario confidence to include (default 0.3)
    """

    def __init__(
        self,
        max_scenarios_per_fsm: int = 20,
        confidence_floor: float = 0.3,
    ) -> None:
        self._max_per_fsm = max_scenarios_per_fsm
        self._confidence_floor = confidence_floor
        self._log = logger

    # ── Primary entry point ───────────────────────────────────────────────────

    def generate(
        self,
        fsms: list[ProtocolFsm],
        target_id: str,
    ) -> ScenarioLibrary:
        """
        Generate a ScenarioLibrary covering all boundary paths in fsms.
        """
        from systems.simula.inspector.protocol_state_machine import BoundaryTransitionEngine

        log = self._log.bind(target_id=target_id)
        log.info("scenario_generation_started", fsm_count=len(fsms))

        library = ScenarioLibrary(target_id=target_id)
        path_engine = BoundaryTransitionEngine()

        for fsm in fsms:
            paths = path_engine.extract_boundary_paths(fsm)
            log.debug(
                "boundary_paths_extracted",
                fsm_id=fsm.fsm_id,
                fsm_name=fsm.name,
                path_count=len(paths),
            )

            scenario_count = 0
            for state_path, trans_path, boundary_kind, strategy in paths:
                if scenario_count >= self._max_per_fsm:
                    break

                scenario = self._build_scenario(
                    fsm=fsm,
                    state_path=state_path,
                    trans_path=trans_path,
                    boundary_kind=boundary_kind,
                    strategy=strategy,
                    target_id=target_id,
                )
                if scenario.confidence < self._confidence_floor:
                    continue

                library.add_scenario(scenario)
                scenario_count += 1

        log.info(
            "scenario_generation_complete",
            total_scenarios=library.total_scenarios,
        )
        return library

    # ── Scenario construction ─────────────────────────────────────────────────

    def _build_scenario(
        self,
        fsm: ProtocolFsm,
        state_path: list[str],
        trans_path: list[str],
        boundary_kind: BoundaryKind,
        strategy: MutationStrategy,
        target_id: str,
    ) -> StressScenario:
        """
        Build a StressScenario for a specific boundary path.

        The scenario is built by:
        1. Walking the state path and recording each StateStep.
        2. Applying the MutationStrategy to counter/timer values
           at the stress point (last boundary transition or state).
        3. Setting the stress_* fields from the mutated state.
        """
        # Working copy of counters so mutation does not affect the FSM
        live_counters: dict[str, FsmCounter] = {}

        # Seed counters from the initial state
        initial = fsm.states.get(fsm.initial_state_id)
        if initial:
            for c in initial.counters:
                live_counters[c.name] = copy.deepcopy(c)

        steps: list[StateStep] = []
        stress_transition_id = ""
        stress_state_id = ""
        stress_counter_name = ""
        stress_counter_value: int | None = None
        stress_timer_name = ""

        # Identify the stress point index
        stress_index = len(trans_path) - 1 if trans_path else len(state_path) - 1

        for step_i, state_id in enumerate(state_path):
            state = fsm.states.get(state_id)
            if state is None:
                continue

            # Determine the transition used to arrive (if any)
            trans_id = trans_path[step_i - 1] if step_i > 0 and step_i - 1 < len(trans_path) else ""
            transition = fsm.transitions.get(trans_id) if trans_id else None

            # Apply counter increments from the transition
            if transition:
                for cname, delta in transition.counter_increments.items():
                    if cname in live_counters:
                        if delta == 0:
                            live_counters[cname].current_value = 0
                        else:
                            live_counters[cname].current_value += delta

            is_stress = (step_i == stress_index or
                         (trans_id and trans_id in fsm.boundary_transition_ids))

            # Apply mutation strategy at stress point
            if is_stress:
                stress_transition_id = trans_id
                stress_state_id = state_id
                self._apply_mutation(
                    strategy=strategy,
                    live_counters=live_counters,
                    state=state,
                )
                if live_counters:
                    c0 = next(iter(live_counters.values()))
                    stress_counter_name = c0.name
                    stress_counter_value = c0.current_value
                if state.timers:
                    stress_timer_name = state.timers[0].name

            active_timers = [t.name for t in state.timers if t.is_active]

            step = StateStep(
                step_index=step_i,
                transition_id=trans_id,
                event_name=transition.event_name if transition else "START",
                from_state_id=state_path[step_i - 1] if step_i > 0 else state_id,
                from_state_name=(
                    fsm.states[state_path[step_i - 1]].name
                    if step_i > 0 and state_path[step_i - 1] in fsm.states
                    else state.name
                ),
                to_state_id=state_id,
                to_state_name=state.name,
                interpretation_layer=(
                    transition.interpretation_layer if transition else state.layer
                ),
                counter_values={n: c.current_value for n, c in live_counters.items()},
                active_timers=active_timers,
                is_stress_point=is_stress,
            )
            steps.append(step)

        description = self._describe_scenario(
            fsm=fsm,
            state_path=state_path,
            strategy=strategy,
            boundary_kind=boundary_kind,
            stress_counter_name=stress_counter_name,
            stress_counter_value=stress_counter_value,
            stress_timer_name=stress_timer_name,
        )

        confidence = self._estimate_confidence(fsm, trans_path, strategy)

        return StressScenario(
            target_id=target_id,
            fsm_id=fsm.fsm_id,
            boundary_kind=boundary_kind,
            mutation_strategy=strategy,
            protocol_family=fsm.protocol_family,
            protocol_name=fsm.name,
            steps=steps,
            state_path=state_path,
            transition_path=trans_path,
            stress_transition_id=stress_transition_id,
            stress_state_id=stress_state_id,
            stress_counter_name=stress_counter_name,
            stress_counter_value=stress_counter_value,
            stress_timer_name=stress_timer_name,
            description=description,
            confidence=confidence,
        )

    # ── Mutation dispatch ─────────────────────────────────────────────────────

    def _apply_mutation(
        self,
        strategy: MutationStrategy,
        live_counters: dict[str, FsmCounter],
        state: ProtocolFsmState,
    ) -> None:
        """Apply the mutation strategy, modifying live_counters in place."""
        dispatch = {
            MutationStrategy.COUNTER_MAXIMISE:   self._mut_counter_maximise,
            MutationStrategy.COUNTER_OVERFLOW:    self._mut_counter_overflow,
            MutationStrategy.COUNTER_RESET_MID:   self._mut_counter_reset_mid,
            MutationStrategy.TIMER_BOUNDARY:      self._mut_timer_boundary,
            MutationStrategy.TIMER_UNDERRUN:      self._mut_timer_underrun,
            MutationStrategy.REKEY_RACE:          self._mut_rekey_race,
            MutationStrategy.VERSION_DOWNGRADE:   self._mut_version_downgrade,
            MutationStrategy.VERSION_UNKNOWN:     self._mut_version_unknown,
            MutationStrategy.FRAGMENT_AT_MTU:     self._mut_fragment_at_mtu,
            MutationStrategy.LAYER_SKIP:          self._mut_layer_skip,
            MutationStrategy.STREAM_LIMIT:        self._mut_stream_limit,
            MutationStrategy.STREAM_OVER_LIMIT:   self._mut_stream_over_limit,
            MutationStrategy.AUTH_EXPIRE:         self._mut_auth_expire,
            MutationStrategy.PADDING_EXACT:       self._mut_padding_exact,
        }
        fn = dispatch.get(strategy, self._mut_counter_maximise)
        fn(live_counters, state)

    @staticmethod
    def _mut_counter_maximise(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        for c in live_counters.values():
            c.current_value = c.max_value

    @staticmethod
    def _mut_counter_overflow(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        for c in live_counters.values():
            # One past the maximum - overflow / wrap
            c.current_value = c.max_value + 1 if not c.wraps else c.min_value

    @staticmethod
    def _mut_counter_reset_mid(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        for c in live_counters.values():
            # Drive to mid-range, then reset to zero
            c.current_value = c.max_value // 2
            # Signal the reset - the replay engine records this as a mid-session reset
            c.current_value = 0

    @staticmethod
    def _mut_timer_boundary(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        # Activate all timers at exact boundary (elapsed == timeout)
        for t in state.timers:
            t.is_active = True
            t.elapsed_ms = t.timeout_ms  # zero slack

    @staticmethod
    def _mut_timer_underrun(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        # Mark timers as completing before timeout
        for t in state.timers:
            t.is_active = True
            t.elapsed_ms = max(0, t.timeout_ms - 1)

    @staticmethod
    def _mut_rekey_race(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        # Drive epoch counter to max while previous key is still "live"
        for c in live_counters.values():
            if "epoch" in c.name.lower() or "rekey" in c.name.lower():
                c.current_value = c.max_value

    @staticmethod
    def _mut_version_downgrade(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        for c in live_counters.values():
            if "version" in c.name.lower():
                c.current_value = c.min_value

    @staticmethod
    def _mut_version_unknown(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        for c in live_counters.values():
            if "version" in c.name.lower():
                c.current_value = c.max_value + 1  # unrecognised version

    @staticmethod
    def _mut_fragment_at_mtu(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        for c in live_counters.values():
            if "fragment" in c.name.lower() or "offset" in c.name.lower():
                c.current_value = c.max_value

    @staticmethod
    def _mut_layer_skip(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        # Simulated by keeping counters at nominal values; the scenario
        # construction intentionally omits the bridging transition
        pass

    @staticmethod
    def _mut_stream_limit(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        for c in live_counters.values():
            if "stream" in c.name.lower():
                c.current_value = c.max_value

    @staticmethod
    def _mut_stream_over_limit(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        for c in live_counters.values():
            if "stream" in c.name.lower():
                c.current_value = c.max_value + 1

    @staticmethod
    def _mut_auth_expire(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        # Drive auth-window timers past their expiry
        for t in state.timers:
            if "auth" in t.name.lower() or "session" in t.name.lower():
                t.is_active = True
                t.elapsed_ms = t.timeout_ms + 1

    @staticmethod
    def _mut_padding_exact(
        live_counters: dict[str, FsmCounter], state: ProtocolFsmState
    ) -> None:
        # Set padding/length counters to exact block-size boundary
        for c in live_counters.values():
            if "pad" in c.name.lower() or "block" in c.name.lower():
                block_size = c.metadata.get("block_size", 16) if hasattr(c, "metadata") else 16
                c.current_value = (c.max_value // block_size) * block_size

    # ── Description + confidence helpers ─────────────────────────────────────

    @staticmethod
    def _describe_scenario(
        fsm: ProtocolFsm,
        state_path: list[str],
        strategy: MutationStrategy,
        boundary_kind: BoundaryKind,
        stress_counter_name: str,
        stress_counter_value: int | None,
        stress_timer_name: str,
    ) -> str:
        state_names = " → ".join(
            fsm.states[sid].name if sid in fsm.states else sid
            for sid in state_path
        )
        counter_detail = (
            f" with {stress_counter_name}={stress_counter_value}"
            if stress_counter_name and stress_counter_value is not None
            else (f" activating timer '{stress_timer_name}'" if stress_timer_name else "")
        )
        return (
            f"[{fsm.name}] Strategy {strategy.value}: drive path "
            f"{state_names}{counter_detail} "
            f"targeting {boundary_kind.value} boundary."
        )

    @staticmethod
    def _estimate_confidence(
        fsm: ProtocolFsm,
        trans_path: list[str],
        strategy: MutationStrategy,
    ) -> float:
        """
        Heuristic confidence estimate.

        Higher when:
        - boundary transitions in path have higher individual confidence
        - the FSM was built from Phase 4 (PROTOCOL_STATE variables) rather
          than just fragments (inferred)
        """
        if not trans_path:
            return 0.35

        avg_conf = sum(
            fsm.transitions[tid].confidence
            for tid in trans_path
            if tid in fsm.transitions
        ) / len(trans_path)

        # Slight boost for strategies backed by concrete spec bounds
        strategy_bonus = {
            MutationStrategy.COUNTER_OVERFLOW:  0.05,
            MutationStrategy.TIMER_BOUNDARY:    0.05,
            MutationStrategy.LAYER_SKIP:        0.03,
            MutationStrategy.AUTH_EXPIRE:       0.04,
        }.get(strategy, 0.0)

        return min(1.0, avg_conf + strategy_bonus)


# ── Stage B: Scenario replay + failure detection ───────────────────────────────


class ScenarioReplayer:
    """
    Replays a ScenarioLibrary against the FSM model, detecting boundary
    failures and interpretation mismatches.

    This is a *model-level* replayer - it walks the FSM state machine and
    checks guards, counter/timer invariants, and layer-interpretation
    consistency at each step.  It does not execute real network code.

    The result is a FailureAtBoundaryDataset and a StateCoverageTracker.
    """

    def __init__(self) -> None:
        self._log = logger

    # ── Primary entry point ───────────────────────────────────────────────────

    def replay_all(
        self,
        library: ScenarioLibrary,
        fsms: list[ProtocolFsm],
        target_id: str,
    ) -> tuple[FailureAtBoundaryDataset, list[StateCoverageReport]]:
        """
        Replay all scenarios in library, returning the failure dataset
        and per-FSM coverage reports.
        """
        fsm_map = {f.fsm_id: f for f in fsms}
        dataset = FailureAtBoundaryDataset(target_id=target_id)
        coverage_trackers: dict[str, StateCoverageTracker] = {
            f.fsm_id: StateCoverageTracker(f) for f in fsms
        }

        for scenario in library.scenarios.values():
            fsm = fsm_map.get(scenario.fsm_id)
            if fsm is None:
                continue

            tracker = coverage_trackers[fsm.fsm_id]
            failures = self._replay_scenario(scenario, fsm, target_id, tracker)

            for failure in failures:
                entry = self._build_entry(scenario, failure, target_id)
                dataset.entries.append(entry)
                dataset.total_entries += 1
                if entry.is_security_relevant:
                    dataset.security_relevant_entries += 1

                bk = entry.boundary_kind.value
                dataset.entries_by_boundary_kind.setdefault(bk, []).append(entry.entry_id)

        coverage_reports = [t.build_report() for t in coverage_trackers.values()]
        return dataset, coverage_reports

    # ── Single scenario replay ────────────────────────────────────────────────

    def _replay_scenario(
        self,
        scenario: StressScenario,
        fsm: ProtocolFsm,
        target_id: str,
        tracker: StateCoverageTracker,
    ) -> list[BoundaryFailure]:
        """
        Walk the scenario steps against the FSM, detecting:
          - Guard violations (counter/timer boundary exceeded)
          - Layer-interpretation mismatches between consecutive steps
          - Missing bridging transitions (layer-skip scenarios)
        """
        failures: list[BoundaryFailure] = []
        prev_layer: TransitionInterpretation = TransitionInterpretation.UNKNOWN

        for step in scenario.steps:
            tracker.record_state(step.to_state_id)
            if step.transition_id:
                tracker.record_transition(step.transition_id)

            state = fsm.states.get(step.to_state_id)
            transition = fsm.transitions.get(step.transition_id) if step.transition_id else None

            # ── Guard violation check ──────────────────────────────────────
            if step.is_stress_point and transition and transition.guards:
                violation = self._check_guards(step, transition, scenario)
                if violation:
                    failures.append(violation)
                    continue

            # ── Counter overflow check ────────────────────────────────────
            if step.is_stress_point:
                overflow = self._check_counter_overflow(step, state, scenario, target_id)
                if overflow:
                    failures.append(overflow)

            # ── Timer boundary check ──────────────────────────────────────
            if state and state.timers and step.is_stress_point:
                timer_fail = self._check_timer_boundary(step, state, scenario, target_id)
                if timer_fail:
                    failures.append(timer_fail)

            # ── Interpretation mismatch check ─────────────────────────────
            curr_layer = step.interpretation_layer
            if (prev_layer != TransitionInterpretation.UNKNOWN
                    and curr_layer != TransitionInterpretation.UNKNOWN
                    and prev_layer != curr_layer):
                # Check whether a layer-boundary transition is declared
                is_declared = transition and transition.is_layer_boundary if transition else False
                if not is_declared:
                    mismatch = InterpretationMismatch(
                        target_id=target_id,
                        layer_a=prev_layer,
                        layer_b=curr_layer,
                        transition_id=step.transition_id,
                        state_id=step.to_state_id,
                        layer_a_interpretation=f"Previous layer treats state as {prev_layer.value}",
                        layer_b_interpretation=f"Current layer treats state as {curr_layer.value}",
                        inconsistency_description=(
                            f"{prev_layer.value.upper()} accepted transition as valid "
                            f"but {curr_layer.value.upper()} was not expecting it "
                            f"(state: {step.to_state_name})"
                        ),
                        boundary_kind=scenario.boundary_kind,
                        confidence=0.6,
                    )
                    failure = BoundaryFailure(
                        target_id=target_id,
                        scenario_id=scenario.scenario_id,
                        boundary_kind=BoundaryKind.LAYER_DESYNC,
                        result=ScenarioResult.DESYNC_DETECTED,
                        state_path=scenario.state_path[:step.step_index + 1],
                        failing_transition_id=step.transition_id,
                        failing_state_id=step.to_state_id,
                        counter_values_at_failure=step.counter_values,
                        active_timers_at_failure=step.active_timers,
                        anomaly_description=mismatch.inconsistency_description,
                        mismatch=mismatch,
                        is_security_relevant=True,
                        security_impact_description=(
                            "Layer desync may allow a message to be accepted by "
                            "one layer while another layer has already closed the "
                            "session, enabling state confusion attacks."
                        ),
                        boundary_kind_confirmed=True,
                        reproduction_script=self._generate_reproduction_script(scenario, step),
                        confidence=0.6,
                    )
                    failures.append(failure)

            prev_layer = curr_layer if curr_layer != TransitionInterpretation.UNKNOWN else prev_layer

        # Update scenario result
        if failures:
            results = {f.result for f in failures}
            if ScenarioResult.DESYNC_DETECTED in results:
                scenario.result = ScenarioResult.DESYNC_DETECTED
            else:
                scenario.result = ScenarioResult.BOUNDARY_FAILURE
        else:
            scenario.result = ScenarioResult.CLEAN

        return failures

    # ── Guard / counter / timer checks ───────────────────────────────────────

    @staticmethod
    def _check_guards(
        step: StateStep,
        transition: ProtocolTransition,
        scenario: StressScenario,
    ) -> BoundaryFailure | None:
        """Check whether any transition guard is violated at this stress step."""

        for guard in transition.guards:
            if not guard.is_stress_target or guard.boundary_value is None:
                continue
            operand_val = step.counter_values.get(guard.operand_name)
            if operand_val is None:
                continue
            violated = False
            op = guard.operator
            bv = guard.boundary_value
            if op == "<"  and not (operand_val < bv):   violated = True
            if op == "<=" and not (operand_val <= bv):  violated = True
            if op == "==" and operand_val != bv:  violated = True
            if op == ">=" and not (operand_val >= bv):  violated = True
            if op == ">"  and not (operand_val > bv):   violated = True
            if op == "!=" and operand_val == bv:  violated = True

            if violated:
                return BoundaryFailure(
                    target_id=scenario.target_id,
                    scenario_id=scenario.scenario_id,
                    boundary_kind=scenario.boundary_kind,
                    result=ScenarioResult.BOUNDARY_FAILURE,
                    state_path=scenario.state_path[:step.step_index + 1],
                    failing_transition_id=step.transition_id,
                    failing_state_id=step.to_state_id,
                    counter_values_at_failure=step.counter_values,
                    active_timers_at_failure=step.active_timers,
                    anomaly_description=(
                        f"Guard '{guard.description}' violated: "
                        f"{guard.operand_name}={operand_val} "
                        f"(boundary={bv}, op='{op}')"
                    ),
                    is_security_relevant=True,
                    boundary_kind_confirmed=True,
                    reproduction_script=ScenarioReplayer._generate_reproduction_script(
                        scenario, step
                    ),
                    confidence=0.75,
                )
        return None

    @staticmethod
    def _check_counter_overflow(
        step: StateStep,
        state: ProtocolFsmState | None,
        scenario: StressScenario,
        target_id: str,
    ) -> BoundaryFailure | None:
        """Detect counter overflow/underflow at stress point."""
        if state is None:
            return None
        for counter in state.counters:
            current = step.counter_values.get(counter.name, counter.current_value)
            if current > counter.max_value:
                return BoundaryFailure(
                    target_id=target_id,
                    scenario_id=scenario.scenario_id,
                    boundary_kind=BoundaryKind.SEQUENCE_COUNTER_OVERFLOW,
                    result=ScenarioResult.BOUNDARY_FAILURE,
                    state_path=scenario.state_path[:step.step_index + 1],
                    failing_state_id=step.to_state_id,
                    counter_values_at_failure=step.counter_values,
                    anomaly_description=(
                        f"Counter '{counter.name}' overflowed: "
                        f"value={current} exceeds max={counter.max_value}"
                    ),
                    is_security_relevant=counter.wraps,
                    security_impact_description=(
                        "Wrapping counters may allow sequence-number reuse, "
                        "enabling replay attacks."
                    ) if counter.wraps else "",
                    boundary_kind_confirmed=True,
                    reproduction_script=ScenarioReplayer._generate_reproduction_script(
                        scenario, step
                    ),
                    confidence=0.7,
                )
        return None

    @staticmethod
    def _check_timer_boundary(
        step: StateStep,
        state: ProtocolFsmState,
        scenario: StressScenario,
        target_id: str,
    ) -> BoundaryFailure | None:
        """Detect timer firing at exactly the boundary (zero-slack expiry)."""
        for timer in state.timers:
            if not timer.is_active:
                continue
            if timer.elapsed_ms >= timer.timeout_ms:
                return BoundaryFailure(
                    target_id=target_id,
                    scenario_id=scenario.scenario_id,
                    boundary_kind=BoundaryKind.TIMER_EXPIRY_AT_BOUNDARY,
                    result=ScenarioResult.BOUNDARY_FAILURE,
                    state_path=scenario.state_path[:step.step_index + 1],
                    failing_state_id=step.to_state_id,
                    active_timers_at_failure=step.active_timers,
                    counter_values_at_failure=step.counter_values,
                    anomaly_description=(
                        f"Timer '{timer.name}' expired at exact boundary: "
                        f"elapsed={timer.elapsed_ms}ms == timeout={timer.timeout_ms}ms"
                    ),
                    is_security_relevant=(
                        "auth" in timer.name.lower() or "session" in timer.name.lower()
                    ),
                    security_impact_description=(
                        "Auth/session timer expiry at zero-slack may cause the handler "
                        "to accept a message it should reject, or reject a legitimate "
                        "in-flight message."
                    ),
                    boundary_kind_confirmed=True,
                    reproduction_script=ScenarioReplayer._generate_reproduction_script(
                        scenario, step
                    ),
                    confidence=0.65,
                )
        return None

    # ── Entry construction ────────────────────────────────────────────────────

    @staticmethod
    def _build_entry(
        scenario: StressScenario,
        failure: BoundaryFailure,
        target_id: str,
    ) -> FailureAtBoundaryEntry:
        """Wrap a BoundaryFailure in a FailureAtBoundaryEntry."""
        # Attempt to get human-readable state names from the scenario steps
        step_names: dict[str, str] = {
            s.to_state_id: s.to_state_name for s in scenario.steps
        }
        state_path_names = [step_names.get(sid, sid) for sid in failure.state_path]

        failing_transition_event = ""
        for s in scenario.steps:
            if s.transition_id == failure.failing_transition_id:
                failing_transition_event = s.event_name
                break

        return FailureAtBoundaryEntry(
            target_id=target_id,
            state_path=state_path_names,
            state_path_ids=failure.state_path,
            boundary_state_name=state_path_names[-1] if state_path_names else "",
            transition_event_name=failing_transition_event,
            transition_id=failure.failing_transition_id,
            mismatch=failure.mismatch,
            failure=failure,
            boundary_kind=failure.boundary_kind,
            mutation_strategy=scenario.mutation_strategy,
            protocol_family=scenario.protocol_family,
            scenario_id=scenario.scenario_id,
            is_security_relevant=failure.is_security_relevant,
        )

    # ── Reproduction script helper ────────────────────────────────────────────

    @staticmethod
    def _generate_reproduction_script(
        scenario: StressScenario,
        step: StateStep,
    ) -> str:
        """
        Generate a human-readable state evolution script for reproducing
        the failure.  Uses pseudocode, not live protocol bytes.
        """
        lines = [
            f"# Reproduction script for scenario {scenario.scenario_id}",
            f"# Protocol: {scenario.protocol_name}  Family: {scenario.protocol_family.value}",
            f"# Boundary: {scenario.boundary_kind.value}  Strategy: {scenario.mutation_strategy.value}",
            "",
            "def reproduce():",
        ]

        for s in scenario.steps:
            indent = "    "
            if s.event_name == "START":
                lines.append(f"{indent}# Initial state: {s.to_state_name}")
                for cname, cval in s.counter_values.items():
                    lines.append(f"{indent}set_counter('{cname}', {cval})")
            else:
                prefix = f"{indent}# >>> STRESS POINT <<<  " if s.is_stress_point else indent
                lines.append(
                    f"{prefix}send_event('{s.event_name}')  "
                    f"# {s.from_state_name} → {s.to_state_name}"
                )
                if s.is_stress_point:
                    for cname, cval in s.counter_values.items():
                        lines.append(f"{indent}assert_counter('{cname}', {cval})")
                    if s.active_timers:
                        lines.append(
                            f"{indent}# Active timers at stress point: {s.active_timers}"
                        )

            if s.step_index == step.step_index:
                lines.append(f"{indent}# ^^^ Failure observed here ^^^")
                break

        lines.append("    # Expected: no failure")
        lines.append(f"    # Observed:  {scenario.result.value}")
        return "\n".join(lines)


# ── Coverage tracking ──────────────────────────────────────────────────────────


class StateCoverageTracker:
    """
    Accumulates state and transition coverage across scenario replays.

    One tracker per FSM.
    """

    def __init__(self, fsm: ProtocolFsm) -> None:
        self._fsm = fsm
        self._covered_states: set[str] = set()
        self._covered_transitions: set[str] = set()

    def record_state(self, state_id: str) -> None:
        self._covered_states.add(state_id)

    def record_transition(self, transition_id: str) -> None:
        self._covered_transitions.add(transition_id)

    def build_report(self) -> StateCoverageReport:
        fsm = self._fsm
        uncovered_states = [
            sid for sid in fsm.states if sid not in self._covered_states
        ]
        uncovered_boundary_states = [
            sid for sid in fsm.boundary_state_ids if sid not in self._covered_states
        ]
        uncovered_transitions = [
            tid for tid in fsm.transitions if tid not in self._covered_transitions
        ]
        uncovered_boundary_transitions = [
            tid for tid in fsm.boundary_transition_ids if tid not in self._covered_transitions
        ]

        total_s = fsm.total_states
        covered_s = len(self._covered_states)
        boundary_s = fsm.total_boundary_states
        covered_bs = boundary_s - len(uncovered_boundary_states)

        total_t = fsm.total_transitions
        covered_t = len(self._covered_transitions)
        boundary_t = fsm.total_boundary_transitions
        covered_bt = boundary_t - len(uncovered_boundary_transitions)

        # Per-transition detail
        records: list[TransitionCoverageRecord] = []
        for tid, trans in fsm.transitions.items():
            from_state = fsm.states.get(trans.from_state_id)
            to_state = fsm.states.get(trans.to_state_id)
            records.append(TransitionCoverageRecord(
                transition_id=tid,
                event_name=trans.event_name,
                from_state_name=from_state.name if from_state else trans.from_state_id,
                to_state_name=to_state.name if to_state else trans.to_state_id,
                times_exercised=1 if tid in self._covered_transitions else 0,
                is_boundary=tid in fsm.boundary_transition_ids,
                was_covered=tid in self._covered_transitions,
            ))

        return StateCoverageReport(
            target_id=fsm.target_id,
            fsm_id=fsm.fsm_id,
            total_states=total_s,
            covered_states=covered_s,
            boundary_states=boundary_s,
            covered_boundary_states=covered_bs,
            uncovered_state_ids=uncovered_states,
            uncovered_boundary_state_ids=uncovered_boundary_states,
            total_transitions=total_t,
            covered_transitions=covered_t,
            boundary_transitions=boundary_t,
            covered_boundary_transitions=covered_bt,
            uncovered_transition_ids=uncovered_transitions,
            uncovered_boundary_transition_ids=uncovered_boundary_transitions,
            transition_records=records,
            state_coverage_ratio=covered_s / total_s if total_s else 0.0,
            transition_coverage_ratio=covered_t / total_t if total_t else 0.0,
            boundary_state_coverage_ratio=covered_bs / boundary_s if boundary_s else 0.0,
            boundary_transition_coverage_ratio=covered_bt / boundary_t if boundary_t else 0.0,
        )
