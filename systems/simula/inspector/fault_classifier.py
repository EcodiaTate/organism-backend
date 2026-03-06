"""
EcodiaOS — Inspector Phase 2: Fault Classifier

Analyses TraceDatasets to produce a FaultClassificationReport.

Responsibilities
----------------
1. Refine preliminary FaultClass labels from the RuntimeTracer using
   cross-event evidence (e.g. OOB confirmed by a SIGNAL_SEGV shortly after).
2. Identify "where did control become unstructured?" — the last call-graph
   node before the fault transition.
3. Aggregate per-class counts and top transition points across the full
   dataset for the steerability report.

Design
------
The classifier is a pure function over an immutable TraceDataset — it never
re-runs targets.  All signal observation is done via evidence already present
in the trace (FaultObservation.signal_number, exception_type, stack_trace).

No LLM calls.  Classification logic is deterministic rule-based + confidence
aggregation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import structlog

from systems.simula.inspector.runtime_types import (
    FaultClass,
    FaultClassificationReport,
    FaultObservation,
    TraceDataset,
)

logger = structlog.get_logger().bind(system="simula.inspector.fault_classifier")


# ── Refinement rules ──────────────────────────────────────────────────────────

# Exception type → (FaultClass, confidence_floor)
# These override low-confidence preliminary labels when the exception type is
# unambiguous.
_EXCEPTION_TYPE_MAP: dict[str, tuple[FaultClass, float]] = {
    "IndexError":           (FaultClass.OOB,          0.85),
    "BufferError":          (FaultClass.OOB,          0.85),
    "OverflowError":        (FaultClass.OOB,          0.75),
    "MemoryError":          (FaultClass.OOB,          0.55),
    "RecursionError":       (FaultClass.OOB,          0.50),
    "TypeError":            (FaultClass.TYPE,         0.80),
    "AttributeError":       (FaultClass.UAF,          0.55),  # refined below
    "AssertionError":       (FaultClass.LOGIC,        0.90),
    "NotImplementedError":  (FaultClass.LOGIC,        0.70),
    "RuntimeError":         (FaultClass.LIFETIME,     0.55),
    "StopIteration":        (FaultClass.LIFETIME,     0.60),
    "GeneratorExit":        (FaultClass.LIFETIME,     0.70),
    "ReferenceError":       (FaultClass.UAF,          0.90),
    "WeakReferenceError":   (FaultClass.UAF,          0.90),
    "ValueError":           (FaultClass.UNHANDLED_EXC, 0.50),
    "KeyError":             (FaultClass.UNHANDLED_EXC, 0.50),
}

# Signal number → (FaultClass, confidence)
_SIGNAL_MAP: dict[int, tuple[FaultClass, float]] = {
    11: (FaultClass.SIGNAL_SEGV,  0.95),  # SIGSEGV
    6:  (FaultClass.SIGNAL_ABORT, 0.95),  # SIGABRT
    7:  (FaultClass.SIGNAL_BUS,   0.95),  # SIGBUS
    8:  (FaultClass.SIGNAL_FPE,   0.95),  # SIGFPE
    4:  (FaultClass.SIGNAL_OTHER, 0.80),  # SIGILL
    9:  (FaultClass.SIGNAL_OTHER, 0.70),  # SIGKILL (timeout)
    15: (FaultClass.SIGNAL_OTHER, 0.60),  # SIGTERM
}

# Stack-trace content patterns → (FaultClass, confidence_bonus)
# Applied on top of the primary classification when the stack trace contains
# recognisable tool output (AddressSanitizer, Valgrind, etc.)
_STACK_TRACE_PATTERNS: list[tuple[re.Pattern, FaultClass, float]] = [
    (re.compile(r"heap.use.after.free",            re.I), FaultClass.UAF,    0.95),
    (re.compile(r"use.after.free",                 re.I), FaultClass.UAF,    0.92),
    (re.compile(r"use.after.poison",               re.I), FaultClass.UAF,    0.92),
    (re.compile(r"double.free",                    re.I), FaultClass.UAF,    0.92),
    (re.compile(r"invalid.free",                   re.I), FaultClass.UAF,    0.88),
    (re.compile(r"heap.buffer.overflow",           re.I), FaultClass.OOB,    0.95),
    (re.compile(r"stack.buffer.overflow",          re.I), FaultClass.OOB,    0.95),
    (re.compile(r"global.buffer.overflow",         re.I), FaultClass.OOB,    0.95),
    (re.compile(r"stack.smashing",                 re.I), FaultClass.OOB,    0.95),
    (re.compile(r"out.of.bounds",                  re.I), FaultClass.OOB,    0.85),
    (re.compile(r"type.confusion",                 re.I), FaultClass.TYPE,   0.90),
    (re.compile(r"invalid.cast",                   re.I), FaultClass.TYPE,   0.85),
    (re.compile(r"mismatched.type",                re.I), FaultClass.TYPE,   0.80),
    (re.compile(r"lifetime",                       re.I), FaultClass.LIFETIME, 0.75),
    (re.compile(r"object has no attribute",        re.I), FaultClass.UAF,    0.55),
    (re.compile(r"weakly-referenced.object",       re.I), FaultClass.UAF,    0.85),
    (re.compile(r"assertion.fail",                 re.I), FaultClass.LOGIC,  0.90),
    (re.compile(r"panic",                          re.I), FaultClass.LOGIC,  0.70),
]


# ── Core refinement logic ─────────────────────────────────────────────────────


def _refine_fault(obs: FaultObservation) -> FaultObservation:
    """
    Apply refinement rules to a single FaultObservation.

    Returns a new FaultObservation (immutable — Pydantic ``model_copy``).
    Does not mutate the original.
    """
    fault_class = obs.fault_class
    confidence = obs.confidence

    # 1. Signal-number is the highest-evidence source
    if obs.signal_number is not None:
        mapped_class, mapped_conf = _SIGNAL_MAP.get(
            obs.signal_number, (FaultClass.SIGNAL_OTHER, 0.75)
        )
        if mapped_conf > confidence:
            fault_class = mapped_class
            confidence = mapped_conf

    # 2. Stack-trace content patterns (ASan / Valgrind / etc.)
    stack_text = " ".join(obs.stack_trace)
    for pattern, candidate_class, candidate_conf in _STACK_TRACE_PATTERNS:
        if pattern.search(stack_text) and candidate_conf > confidence:
            fault_class = candidate_class
            confidence = candidate_conf

    # 3. Exception type mapping (override low-confidence generic classifications)
    if obs.exception_type:
        exc_class, exc_conf = _EXCEPTION_TYPE_MAP.get(
            obs.exception_type, (FaultClass.UNHANDLED_EXC, 0.50)
        )
        # Only override if significantly more confident AND current is UNKNOWN
        if fault_class == FaultClass.UNKNOWN and exc_conf > confidence or (
            fault_class in (FaultClass.UNHANDLED_EXC, FaultClass.UNKNOWN)
            and exc_conf >= confidence
        ):
            fault_class = exc_class
            confidence = exc_conf

    # 4. AttributeError refinement: "NoneType" → UAF, otherwise TYPE
    if obs.exception_type == "AttributeError":
        if "NoneType" in obs.exception_message or "deleted" in obs.exception_message.lower():
            fault_class = FaultClass.UAF
            confidence = max(confidence, 0.65)
        else:
            fault_class = FaultClass.TYPE
            confidence = max(confidence, 0.60)

    if fault_class == obs.fault_class and confidence == obs.confidence:
        return obs  # no change — return original (skip copy overhead)

    return obs.model_copy(update={"fault_class": fault_class, "confidence": confidence})


def _find_last_structured_func(
    fault: FaultObservation,
    call_sequence: list[tuple[str, str]],
) -> str:
    """
    Heuristic: the "last structured function" is the call-sequence node
    immediately before the faulting function.

    If the fault has a stack_trace, attempt to extract the caller from it;
    otherwise fall back to call_sequence traversal.
    """
    fault_func = fault.fault_at_func

    # Stack trace lines that look like Python frames: "  File ..., in func_name"
    if fault.stack_trace:
        py_frame_re = re.compile(r"in\s+([A-Za-z_]\w*)$")
        frame_funcs: list[str] = []
        for line in fault.stack_trace:
            m = py_frame_re.search(line.strip())
            if m:
                frame_funcs.append(m.group(1))
        # The second-to-last frame is the caller of the faulting function
        if len(frame_funcs) >= 2 and frame_funcs[-1] == fault_func:
            return frame_funcs[-2]
        if len(frame_funcs) >= 2:
            return frame_funcs[-2]

    # Fallback: scan call_sequence for last call into fault_func
    if fault_func and call_sequence:
        for caller, callee in reversed(call_sequence):
            if callee == fault_func and caller:
                return caller

    return fault.last_structured_func


# ── FaultClassifier ───────────────────────────────────────────────────────────


class FaultClassifier:
    """
    Analyses a TraceDataset to produce a FaultClassificationReport.

    The classifier:
    1. Refines every FaultObservation using cross-event evidence.
    2. Computes per-class counts and identifies top transition points.
    3. Labels each run with its fault classes.

    All operations are O(n) in the number of observations / traces.
    """

    def classify(self, dataset: TraceDataset) -> FaultClassificationReport:
        """
        Produce a FaultClassificationReport from *dataset*.

        Modifies ``dataset.faults`` in-place (replaces with refined observations).
        Returns the aggregate report.
        """
        log = logger.bind(dataset_id=dataset.id, target_id=dataset.target_id)
        log.debug("classification_started", total_runs=dataset.total_runs)

        class_counts: dict[str, int] = defaultdict(int)
        run_labels: dict[str, list[str]] = {}
        total_faults = 0

        # transition_point → {count, total_depth}
        transition_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_depth": 0}
        )

        for run_id, fault_list in dataset.faults.items():
            trace = dataset.traces.get(run_id)
            call_seq = trace.call_sequence if trace else []

            refined: list[FaultObservation] = []
            labels: list[str] = []

            for obs in fault_list:
                obs = _refine_fault(obs)

                # Enrich last_structured_func if not already set
                if not obs.last_structured_func:
                    lsf = _find_last_structured_func(obs, call_seq)
                    if lsf:
                        obs = obs.model_copy(update={"last_structured_func": lsf})

                refined.append(obs)
                class_counts[obs.fault_class.value] += 1
                total_faults += 1
                labels.append(obs.fault_class.value)

                # Accumulate transition-point statistics
                tp = obs.last_structured_func or "(unknown)"
                transition_stats[tp]["count"] += 1
                transition_stats[tp]["total_depth"] += obs.divergence_depth

            dataset.faults[run_id] = refined
            run_labels[run_id] = labels

        # Compute classified fraction
        unknown_count = class_counts.get(FaultClass.UNKNOWN.value, 0)
        classified_fraction = (
            (total_faults - unknown_count) / total_faults
            if total_faults > 0
            else 0.0
        )

        # Top-10 transition points by frequency
        sorted_transitions = sorted(
            transition_stats.items(),
            key=lambda kv: kv[1]["count"],
            reverse=True,
        )[:10]

        top_transition_points: list[dict[str, Any]] = []
        for func_name, stats in sorted_transitions:
            avg_depth = (
                stats["total_depth"] / stats["count"] if stats["count"] > 0 else 0.0
            )
            top_transition_points.append({
                "func": func_name,
                "occurrence_count": stats["count"],
                "avg_divergence_depth": round(avg_depth, 1),
            })

        report = FaultClassificationReport(
            dataset_id=dataset.id,
            class_counts=dict(class_counts),
            top_transition_points=top_transition_points,
            total_faults=total_faults,
            total_crash_runs=dataset.crash_run_count,
            classified_fraction=classified_fraction,
            run_labels=run_labels,
        )

        log.info(
            "classification_complete",
            total_faults=total_faults,
            classified_fraction=round(classified_fraction, 3),
            top_class=max(class_counts, key=class_counts.get) if class_counts else "none",
        )

        return report

    def top_transition_points_for_class(
        self,
        report: FaultClassificationReport,
        fault_class: FaultClass,
        dataset: TraceDataset,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the top-N transition points that specifically precede a given
        fault class.  More granular than the aggregate report.

        Returns list of dicts with keys: func, occurrence_count, avg_divergence_depth.
        """
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_depth": 0}
        )

        for _run_id, fault_list in dataset.faults.items():
            for obs in fault_list:
                if obs.fault_class == fault_class:
                    tp = obs.last_structured_func or "(unknown)"
                    stats[tp]["count"] += 1
                    stats[tp]["total_depth"] += obs.divergence_depth

        sorted_stats = sorted(
            stats.items(),
            key=lambda kv: kv[1]["count"],
            reverse=True,
        )[:top_n]

        result = []
        for func_name, s in sorted_stats:
            avg_depth = s["total_depth"] / s["count"] if s["count"] > 0 else 0.0
            result.append({
                "func": func_name,
                "fault_class": fault_class.value,
                "occurrence_count": s["count"],
                "avg_divergence_depth": round(avg_depth, 1),
            })
        return result
