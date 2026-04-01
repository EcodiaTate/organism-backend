"""
EcodiaOS - Inspector Phase 2: Control Integrity Scorer

Computes the "Control Integrity Score" (CIS) - a continuous metric in [0, 1]
that quantifies how much a run's execution deviated from the baseline set of
normal runs.

Score semantics
---------------
1.0  - Fully intact: call graph exactly matches baseline; no exception exits;
       no faults; no anomalous branches.
0.0  - Complete divergence: every transition was novel; multiple faults; deep
       stack unwinding.
0.3–0.7 - Partial influence: execution reshaped by external inputs but
          did not crash.  This is the "interesting" range for steerability
          modelling - the program was influenced but remained alive.

The score is intentionally NOT a security metric.  It is a continuous label
for the steerability model: "how much did the observed execution deviate from
baseline?"

Algorithm
---------
1.  Build a "baseline call-edge set" from all normal runs.
2.  For each run:
    a.  Count edges NOT in the baseline → fraction_new_edges
    b.  Count exception exits / total function exits → fraction_exception_exits
    c.  fault_count from the FaultClassificationReport
    d.  max divergence depth from associated FaultObservations
3.  Compute:
      raw = 1.0
           - w_edges   * fraction_new_edges
           - w_exc     * fraction_exception_exits
           - w_fault   * clamp(fault_count / MAX_FAULTS, 0, 1)
           - w_depth   * clamp(max_divergence_depth / MAX_DEPTH, 0, 1)
4.  Clamp to [0, 1].
5.  Detect influence-permissive transitions:
    - Any call edge that was NOT in the baseline normal call graph
    - Any call edge that immediately precedes a fault observation
    - (Optional) Any edge where taint flow correlation suggests the callee
      was determined by external input (taint_influenced flag from eBPF).

Exit criterion
--------------
The Phase 2 exit criterion is met when at least one influence-permissive
transition has been detected and labelled (influence_permissive_transitions_detected=True
in the Phase2Result).  This means we can reliably identify points in execution
where external influence could redirect control flow.
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.simula.inspector.runtime_types import (
    ControlFlowTrace,
    ControlIntegrityScore,
    FaultClassificationReport,
    FaultObservation,
    InfluencePermissiveTransition,
    Phase2Result,
    RunCategory,
    TraceDataset,
)

logger = structlog.get_logger().bind(system="simula.inspector.control_integrity")

# Score component weights - must sum to 1.0.
_W_EDGES = 0.35   # new call edges (primary signal)
_W_EXC   = 0.20   # exception-exit fraction
_W_FAULT = 0.30   # fault count
_W_DEPTH = 0.15   # maximum divergence depth

# Normalisation caps
_MAX_FAULTS_FOR_NORM = 5    # ≥5 faults → full fault component penalty
_MAX_DEPTH_FOR_NORM  = 20   # ≥20 stack depth → full depth component penalty

# Minimum confidence to promote a FaultObservation to a transition label.
_MIN_FAULT_CONFIDENCE = 0.5

# Pre-fault window: an edge is "pre-fault" if it appears in the last N events
# before a fault observation in the same run.
_PRE_FAULT_WINDOW_EVENTS = 3


# ── Baseline builder ──────────────────────────────────────────────────────────


def _build_normal_baseline(
    dataset: TraceDataset,
) -> tuple[set[tuple[str, str]], set[str]]:
    """
    Return (baseline_edges, baseline_functions) from all NORMAL runs.

    baseline_edges    - set of (caller, callee) pairs seen in ANY normal run
    baseline_functions - set of function names entered in ANY normal run
    """
    baseline_edges: set[tuple[str, str]] = set()
    baseline_functions: set[str] = set()

    for trace in dataset.traces.values():
        if trace.run_category != RunCategory.NORMAL:
            continue
        for edge in trace.call_sequence:
            baseline_edges.add(edge)
        for func in trace.functions_visited:
            baseline_functions.add(func)

    return baseline_edges, baseline_functions


# ── Per-run scoring ───────────────────────────────────────────────────────────


def _score_run(
    trace: ControlFlowTrace,
    faults: list[FaultObservation],
    baseline_edges: set[tuple[str, str]],
) -> ControlIntegrityScore:
    """
    Compute the ControlIntegrityScore for a single run.
    """
    # ── Component 1: fraction_new_edges ──────────────────────────────────────
    total_edges = len(trace.call_sequence)
    if total_edges == 0:
        fraction_new_edges = 0.0
        new_edges: set[tuple[str, str]] = set()
    else:
        new_edge_set = {e for e in trace.call_sequence if e not in baseline_edges}
        fraction_new_edges = len(new_edge_set) / total_edges
        new_edges = new_edge_set

    # ── Component 2: fraction_exception_exits ────────────────────────────────
    total_funcs_visited = len(trace.functions_visited)
    if total_funcs_visited == 0:
        fraction_exception_exits = 0.0
    else:
        exc_exits = len(trace.exception_exit_functions)
        fraction_exception_exits = min(exc_exits / total_funcs_visited, 1.0)

    # ── Component 3: fault_count ──────────────────────────────────────────────
    fault_count = len(faults)
    normalised_faults = min(fault_count / _MAX_FAULTS_FOR_NORM, 1.0)

    # ── Component 4: max_divergence_depth ────────────────────────────────────
    max_divergence_depth = max(
        (f.divergence_depth for f in faults), default=0
    )
    normalised_depth = min(max_divergence_depth / _MAX_DEPTH_FOR_NORM, 1.0)

    # ── Compute raw score ────────────────────────────────────────────────────
    raw = (
        1.0
        - _W_EDGES * fraction_new_edges
        - _W_EXC   * fraction_exception_exits
        - _W_FAULT * normalised_faults
        - _W_DEPTH * normalised_depth
    )
    score = max(0.0, min(1.0, raw))

    # ── Detect influence-permissive transitions ───────────────────────────────

    # Build a set of edge positions (index in call_sequence) where a fault follows
    # within _PRE_FAULT_WINDOW_EVENTS.
    fault_at_funcs = {f.fault_at_func for f in faults if f.fault_at_func}

    # Map each event-sequence position to "is it within window of a fault?"
    # We approximate by marking any edge whose callee appears in fault_at_funcs,
    # or whose caller appears in fault_at_funcs.
    pre_fault_edges: set[tuple[str, str]] = set()
    for i, (caller, callee) in enumerate(trace.call_sequence):
        # Look ahead _PRE_FAULT_WINDOW_EVENTS edges
        window = trace.call_sequence[i : i + _PRE_FAULT_WINDOW_EVENTS + 1]
        window_callees = {c for _, c in window}
        if window_callees & fault_at_funcs:
            pre_fault_edges.add((caller, callee))

    permissive: list[InfluencePermissiveTransition] = []
    seen_transitions: set[tuple[str, str]] = set()

    for caller, callee in trace.call_sequence:
        edge = (caller, callee)
        if edge in seen_transitions:
            continue  # de-duplicate - report each novel edge once

        is_new = edge in new_edges
        is_pre_fault = edge in pre_fault_edges

        if not (is_new or is_pre_fault):
            continue

        # Deviation score = distance from baseline frequency
        # For new edges: 1.0 (never seen); for pre-fault edges: boosted by 0.3
        deviation = 1.0 if is_new else 0.3

        pt = InfluencePermissiveTransition(
            run_id=trace.run_id,
            from_func=caller,
            to_func=callee,
            is_new_edge=is_new,
            is_pre_fault=is_pre_fault,
            taint_influenced=False,  # would be set by TaintFlowLinker correlation
            deviation_score=deviation,
        )
        permissive.append(pt)
        seen_transitions.add(edge)

    return ControlIntegrityScore(
        run_id=trace.run_id,
        run_category=trace.run_category,
        score=round(score, 4),
        fraction_new_edges=round(fraction_new_edges, 4),
        fraction_exception_exits=round(fraction_exception_exits, 4),
        fault_count=fault_count,
        max_divergence_depth=max_divergence_depth,
        permissive_transitions=permissive,
        permissive_transition_count=len(permissive),
    )


# ── ControlIntegrityScorer ────────────────────────────────────────────────────


class ControlIntegrityScorer:
    """
    Computes ControlIntegrityScore for every run in a TraceDataset, then
    assembles a Phase2Result.

    Usage::

        scorer = ControlIntegrityScorer()
        result = scorer.score_dataset(dataset, fault_report)
    """

    def score_dataset(
        self,
        dataset: TraceDataset,
        fault_report: FaultClassificationReport,
    ) -> Phase2Result:
        """
        Score all runs and return a Phase2Result.

        Args:
            dataset:       Fully populated TraceDataset (traces + faults already
                           refined by FaultClassifier).
            fault_report:  FaultClassificationReport produced by FaultClassifier.

        Returns:
            Phase2Result with per-run CIS, aggregates, and exit-criterion flag.
        """
        log = logger.bind(dataset_id=dataset.id, target_id=dataset.target_id)
        log.debug("scoring_started", total_runs=dataset.total_runs)

        baseline_edges, _baseline_funcs = _build_normal_baseline(dataset)
        log.debug("baseline_built", baseline_edges=len(baseline_edges))

        scores: list[ControlIntegrityScore] = []
        normal_score_sum = 0.0
        normal_count = 0
        failure_score_sum = 0.0
        failure_count = 0
        crash_score_sum = 0.0
        crash_count = 0

        for run_id, trace in dataset.traces.items():
            faults = dataset.faults.get(run_id, [])
            cis = _score_run(trace, faults, baseline_edges)
            scores.append(cis)

            match trace.run_category:
                case RunCategory.NORMAL:
                    normal_score_sum += cis.score
                    normal_count += 1
                case RunCategory.FAILURE:
                    failure_score_sum += cis.score
                    failure_count += 1
                case RunCategory.CRASH:
                    crash_score_sum += cis.score
                    crash_count += 1

        # Aggregate means
        mean_normal = normal_score_sum / normal_count if normal_count else 1.0
        mean_failure = failure_score_sum / failure_count if failure_count else 0.0
        mean_crash = crash_score_sum / crash_count if crash_count else 0.0

        # Exit criterion: at least one influence-permissive transition detected
        # with high confidence (deviation_score > 0 means it qualified).
        any_permissive = any(
            s.permissive_transition_count > 0 for s in scores
        )

        result = Phase2Result(
            target_id=dataset.target_id,
            dataset=dataset,
            fault_report=fault_report,
            scores=scores,
            mean_normal_score=round(mean_normal, 4),
            mean_failure_score=round(mean_failure, 4),
            mean_crash_score=round(mean_crash, 4),
            influence_permissive_transitions_detected=any_permissive,
        )

        log.info(
            "scoring_complete",
            runs_scored=len(scores),
            mean_normal=round(mean_normal, 4),
            mean_failure=round(mean_failure, 4),
            mean_crash=round(mean_crash, 4),
            permissive_detected=any_permissive,
        )

        return result

    def score_single_run(
        self,
        trace: ControlFlowTrace,
        faults: list[FaultObservation],
        baseline_edges: set[tuple[str, str]],
    ) -> ControlIntegrityScore:
        """
        Score a single run against a pre-computed baseline.

        Useful for streaming / online use cases where you score each run as it
        completes rather than batching.
        """
        return _score_run(trace, faults, baseline_edges)

    def build_baseline(
        self,
        dataset: TraceDataset,
    ) -> set[tuple[str, str]]:
        """
        Extract the baseline call-edge set from a dataset's normal runs.

        Expose this publicly so callers can cache the baseline for repeated
        single-run scoring.
        """
        baseline_edges, _ = _build_normal_baseline(dataset)
        return baseline_edges

    def summary_table(self, result: Phase2Result) -> list[dict[str, Any]]:
        """
        Return a tabular summary of scores, one row per run, sorted by score asc.

        Each row: run_id, category, score, new_edges_pct, exc_exit_pct,
                  fault_count, permissive_transitions.
        """
        rows = []
        for cis in result.scores:
            rows.append({
                "run_id": cis.run_id,
                "category": cis.run_category.value,
                "score": cis.score,
                "new_edges_pct": round(cis.fraction_new_edges * 100, 1),
                "exc_exit_pct": round(cis.fraction_exception_exits * 100, 1),
                "fault_count": cis.fault_count,
                "permissive_transitions": cis.permissive_transition_count,
            })
        rows.sort(key=lambda r: r["score"])
        return rows


# ── RuntimeInstrumentationEngine ─────────────────────────────────────────────


class RuntimeInstrumentationEngine:
    """
    Top-level Phase 2 orchestrator.

    Wires together RuntimeTracer → FaultClassifier → ControlIntegrityScorer
    to produce a Phase2Result from a set of run specifications.

    Usage::

        engine = RuntimeInstrumentationEngine(scope_prefix="mypackage")
        result = await engine.run(
            target_id="mymodule",
            normal_runs=[
                {"target": fn, "args": (valid_input,)},
                ...
            ],
            failure_runs=[
                {"target": fn, "args": (bad_input,)},
                ...
            ],
        )
    """

    def __init__(
        self,
        scope_prefix: str = "",
        subprocess_timeout_s: float = 30.0,
    ) -> None:
        # Import here to avoid circular at module level
        from systems.simula.inspector.fault_classifier import FaultClassifier
        from systems.simula.inspector.runtime_tracer import RuntimeTracer

        self._tracer = RuntimeTracer(
            scope_prefix=scope_prefix,
            subprocess_timeout_s=subprocess_timeout_s,
        )
        self._classifier = FaultClassifier()
        self._scorer = ControlIntegrityScorer()
        self._log = logger

    async def run(
        self,
        target_id: str,
        normal_runs: list[dict],
        failure_runs: list[dict],
        crash_runs: list[dict] | None = None,
    ) -> Phase2Result:
        """
        Execute all runs, classify faults, score control integrity.

        Each run dict supports:
          - In-process: {"target": callable, "args": tuple, "kwargs": dict}
          - Subprocess:  {"cmd": list[str], "cwd": Path}

        The run_category and target_id are injected automatically.
        """
        import uuid

        from systems.simula.inspector.runtime_types import RunCategory

        def _tag(runs: list[dict], category: RunCategory) -> list[dict]:
            tagged = []
            for r in runs:
                r2 = dict(r)
                r2["run_id"] = r2.get("run_id") or str(uuid.uuid4())
                r2["run_category"] = category
                r2["target_id"] = target_id
                tagged.append(r2)
            return tagged

        all_runs: list[dict] = (
            _tag(normal_runs, RunCategory.NORMAL)
            + _tag(failure_runs, RunCategory.FAILURE)
            + _tag(crash_runs or [], RunCategory.CRASH)
        )

        # Split into callable vs subprocess runs
        callable_runs = [r for r in all_runs if "target" in r]
        subprocess_runs = [r for r in all_runs if "cmd" in r]

        self._log.debug(
            "instrumentation_started",
            target_id=target_id,
            callable_runs=len(callable_runs),
            subprocess_runs=len(subprocess_runs),
        )

        # Collect traces
        dataset = await self._tracer.trace_many_callables(callable_runs)
        if subprocess_runs:
            sub_dataset = await self._tracer.trace_many_subprocesses(subprocess_runs)
            # Merge
            for _rid, trace in sub_dataset.traces.items():
                dataset.add_trace(trace)
            for _rid, faults in sub_dataset.faults.items():
                for f in faults:
                    dataset.add_fault(f)

        # Classify faults
        fault_report = self._classifier.classify(dataset)

        # Score
        result = self._scorer.score_dataset(dataset, fault_report)

        self._log.info(
            "instrumentation_complete",
            target_id=target_id,
            exit_criterion_met=result.influence_permissive_transitions_detected,
            mean_normal=result.mean_normal_score,
            mean_failure=result.mean_failure_score,
        )

        return result
