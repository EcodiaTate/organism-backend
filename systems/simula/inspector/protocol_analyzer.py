"""
EcodiaOS — Inspector Phase 6: Protocol Analyzer (Orchestrator)

Single-entry orchestrator for the Phase 6 pipeline:

  ProtocolFsmBuilder    → list[ProtocolFsm]
  BoundaryStressEngine  → ScenarioLibrary
  ScenarioReplayer      → (FailureAtBoundaryDataset, list[StateCoverageReport])
  → Phase6Result

Usage
-----
  # From Phase 5 output (full pipeline):
  analyzer = ProtocolAnalyzer()
  result = analyzer.analyze(
      phase3_result=phase3_result,
      phase4_result=phase4_result,
      phase5_result=phase5_result,
  )

  # From Phase 3 + Phase 4 only:
  result = analyzer.analyze(
      phase3_result=phase3_result,
      phase4_result=phase4_result,
  )

  # Minimal (Phase 3 only — fragments only, no counter/trust decorations):
  result = analyzer.analyze(
      phase3_result=phase3_result,
  )

  # Query: explain a specific boundary failure
  explanation = analyzer.explain_failure(result, failure_id)

  # Get the scenario library for a specific boundary kind
  scenarios = analyzer.scenarios_for_boundary(result, BoundaryKind.LAYER_DESYNC)

Exit criterion
--------------
Phase6Result.exit_criterion_met = True when:
  - ≥1 StressScenario has result != NOT_REACHED (a boundary was reached), AND
  - ≥1 BoundaryFailure has been recorded with state_path + anomaly, AND
  - ≥1 FailureAtBoundaryEntry connects state path → transition → anomaly.
"""

from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.boundary_stress_engine import (
    BoundaryStressEngine,
    ScenarioReplayer,
)
from systems.simula.inspector.protocol_state_machine import ProtocolFsmBuilder
from systems.simula.inspector.protocol_types import (
    BoundaryFailure,
    BoundaryKind,
    FailureAtBoundaryDataset,
    Phase6Result,
    ScenarioLibrary,
    ScenarioResult,
    StressScenario,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import Phase4Result
    from systems.simula.inspector.static_types import Phase3Result
    from systems.simula.inspector.trust_types import Phase5Result

logger = structlog.get_logger().bind(system="simula.inspector.protocol_analyzer")


class ProtocolAnalyzer:
    """
    Phase 6 orchestrator — builds a Phase6Result for a target.

    The result combines:
    - Protocol FSMs inferred from Phase 3/4/5 artifacts
    - A ScenarioLibrary of boundary-targeting state traces
    - A FailureAtBoundaryDataset linking state paths to anomalies
    - Per-FSM StateCoverageReports
    - Aggregate statistics and exit criterion flag

    Parameters
    ----------
    max_scenarios_per_fsm  — cap on generated scenarios per FSM (default 20)
    min_confidence         — minimum scenario confidence to include (default 0.3)
    privilege_threshold    — minimum Phase 5 node privilege_value to inject
                             timer/counter context (default 30)
    max_path_length        — maximum FSM path length during BFS/DFS (default 12)
    """

    def __init__(
        self,
        max_scenarios_per_fsm: int = 20,
        min_confidence: float = 0.3,
        privilege_threshold: int = 30,
        max_path_length: int = 12,
    ) -> None:
        self._fsm_builder = ProtocolFsmBuilder(
            privilege_threshold=privilege_threshold,
        )
        self._stress_engine = BoundaryStressEngine(
            max_scenarios_per_fsm=max_scenarios_per_fsm,
            confidence_floor=min_confidence,
        )
        self._replayer = ScenarioReplayer()
        self._log = logger

    # ── Primary entry point ───────────────────────────────────────────────────

    def analyze(
        self,
        phase3_result: Phase3Result,
        phase4_result: Phase4Result | None = None,
        phase5_result: Phase5Result | None = None,
    ) -> Phase6Result:
        """
        Build a Phase6Result from Phase 3 (and optionally Phase 4/5) data.

        Args:
            phase3_result: Complete Phase 3 static analysis output.
            phase4_result: Optional Phase 4 steerability model output.
                           Enables PROTOCOL_STATE variable extraction and
                           constraint-based counter decoration.
            phase5_result: Optional Phase 5 trust graph output.
                           Enables SESSION/CREDENTIAL timer and epoch injection.

        Returns:
            Phase6Result with FSMs, ScenarioLibrary, FailureAtBoundaryDataset,
            StateCoverageReports, and exit criterion flag.
        """
        target_id = phase3_result.target_id
        log = self._log.bind(target_id=target_id)
        log.info("protocol_analysis_started")

        # 1. Build FSMs
        fsms = self._fsm_builder.build(
            target_id=target_id,
            phase3_result=phase3_result,
            phase4_result=phase4_result,
            phase5_result=phase5_result,
        )

        # 2. Generate stress scenarios
        library = self._stress_engine.generate(
            fsms=fsms,
            target_id=target_id,
        )

        # 3. Replay scenarios — detect failures + build coverage reports
        failure_dataset, coverage_reports = self._replayer.replay_all(
            library=library,
            fsms=fsms,
            target_id=target_id,
        )

        # 4. Flatten boundary failures for top-level access
        boundary_failures = [e.failure for e in failure_dataset.entries]

        # 5. Check exit criterion
        exit_met = self._check_exit_criterion(library, failure_dataset)

        # 6. Aggregate statistics
        total_states = sum(f.total_states for f in fsms)
        total_transitions = sum(f.total_transitions for f in fsms)
        total_boundary_states = sum(f.total_boundary_states for f in fsms)
        total_boundary_transitions = sum(f.total_boundary_transitions for f in fsms)
        total_mismatches = sum(
            1 for e in failure_dataset.entries if e.mismatch is not None
        )

        mean_state_cov = (
            mean(r.state_coverage_ratio for r in coverage_reports)
            if coverage_reports else 0.0
        )
        mean_boundary_cov = (
            mean(r.boundary_state_coverage_ratio for r in coverage_reports)
            if coverage_reports else 0.0
        )

        result = Phase6Result(
            target_id=target_id,
            fsms=fsms,
            scenario_library=library,
            failure_dataset=failure_dataset,
            coverage_reports=coverage_reports,
            boundary_failures=boundary_failures,
            total_fsms=len(fsms),
            total_states=total_states,
            total_transitions=total_transitions,
            total_boundary_states=total_boundary_states,
            total_boundary_transitions=total_boundary_transitions,
            total_scenarios=library.total_scenarios,
            total_boundary_failures=len(boundary_failures),
            security_relevant_failures=failure_dataset.security_relevant_entries,
            total_interpretation_mismatches=total_mismatches,
            mean_state_coverage_ratio=mean_state_cov,
            mean_boundary_coverage_ratio=mean_boundary_cov,
            exit_criterion_met=exit_met,
        )

        log.info(
            "protocol_analysis_complete",
            fsms=len(fsms),
            states=total_states,
            transitions=total_transitions,
            boundary_states=total_boundary_states,
            boundary_transitions=total_boundary_transitions,
            scenarios=library.total_scenarios,
            failures=len(boundary_failures),
            security_relevant=failure_dataset.security_relevant_entries,
            mismatches=total_mismatches,
            exit_criterion_met=exit_met,
        )

        return result

    # ── Targeted queries ──────────────────────────────────────────────────────

    def explain_failure(
        self,
        result: Phase6Result,
        failure_id: str,
    ) -> dict:
        """
        Return a structured explanation dict for a specific boundary failure.

        This is the primary exit-criterion delivery: given a failure_id,
        produce a defensible explanation connecting state path → inconsistent
        transition → observed anomaly.

        Returns a dict with:
          failure_id, boundary_kind, state_path, transition_event,
          counter_values_at_failure, anomaly_description,
          mismatch (if any), reproduction_script, is_security_relevant,
          scenario_id, confidence
        """
        failure = next(
            (f for f in result.boundary_failures if f.failure_id == failure_id),
            None,
        )
        if not failure:
            return {"error": f"failure '{failure_id}' not found"}

        # Find corresponding dataset entry
        entry = next(
            (e for e in result.failure_dataset.entries
             if e.failure.failure_id == failure_id),
            None,
        )

        # Find corresponding scenario
        scenario = result.scenario_library.scenarios.get(failure.scenario_id)

        mismatch_detail: dict = {}
        if failure.mismatch:
            m = failure.mismatch
            mismatch_detail = {
                "mismatch_id":    m.mismatch_id,
                "layer_a":        m.layer_a.value,
                "layer_b":        m.layer_b.value,
                "state_id":       m.state_id,
                "transition_id":  m.transition_id,
                "layer_a_interpretation": m.layer_a_interpretation,
                "layer_b_interpretation": m.layer_b_interpretation,
                "inconsistency":  m.inconsistency_description,
                "confidence":     m.confidence,
            }

        return {
            "failure_id":               failure.failure_id,
            "boundary_kind":            failure.boundary_kind.value,
            "result":                   failure.result.value,
            "state_path": entry.state_path if entry else failure.state_path,
            "boundary_state":           entry.boundary_state_name if entry else "",
            "transition_event":         entry.transition_event_name if entry else "",
            "transition_id":            failure.failing_transition_id,
            "counter_values_at_failure": failure.counter_values_at_failure,
            "active_timers_at_failure": failure.active_timers_at_failure,
            "anomaly_description":      failure.anomaly_description,
            "stack_trace":              failure.stack_trace,
            "error_code":               failure.error_code,
            "mismatch":                 mismatch_detail,
            "is_security_relevant":     failure.is_security_relevant,
            "security_impact":          failure.security_impact_description,
            "is_reproducible":          failure.is_reproducible,
            "reproduction_script":      failure.reproduction_script,
            "scenario_id":              failure.scenario_id,
            "protocol_family": scenario.protocol_family.value if scenario else "unknown",
            "protocol_name":   scenario.protocol_name if scenario else "",
            "confidence":               failure.confidence,
        }

    def scenarios_for_boundary(
        self,
        result: Phase6Result,
        boundary_kind: BoundaryKind,
    ) -> list[StressScenario]:
        """Return all StressScenarios for a specific BoundaryKind."""
        scenario_ids = result.scenario_library.scenarios_by_boundary_kind.get(
            boundary_kind.value, []
        )
        return [
            result.scenario_library.scenarios[sid]
            for sid in scenario_ids
            if sid in result.scenario_library.scenarios
        ]

    def failures_for_boundary(
        self,
        result: Phase6Result,
        boundary_kind: BoundaryKind,
    ) -> list[BoundaryFailure]:
        """Return all BoundaryFailures for a specific BoundaryKind."""
        return [
            f for f in result.boundary_failures
            if f.boundary_kind == boundary_kind
        ]

    # ── Reporting helper ──────────────────────────────────────────────────────

    def model_summary(self, result: Phase6Result) -> dict:
        """
        Return a concise reporting dict suitable for logging or display.
        """
        # FSM breakdown
        fsm_summaries = [
            {
                "fsm_id":                  f.fsm_id,
                "name":                    f.name,
                "protocol_family":         f.protocol_family.value,
                "total_states":            f.total_states,
                "total_transitions":       f.total_transitions,
                "boundary_states":         f.total_boundary_states,
                "boundary_transitions":    f.total_boundary_transitions,
            }
            for f in result.fsms
        ]

        # Top failures by security relevance + boundary kind
        top_failures = sorted(
            result.boundary_failures,
            key=lambda f: (f.is_security_relevant, f.confidence),
            reverse=True,
        )[:5]
        failure_summaries = [
            {
                "failure_id":        f.failure_id,
                "boundary_kind":     f.boundary_kind.value,
                "result":            f.result.value,
                "anomaly":           f.anomaly_description,
                "security_relevant": f.is_security_relevant,
                "confidence":        f.confidence,
                "scenario_id":       f.scenario_id,
            }
            for f in top_failures
        ]

        # Coverage summary per FSM
        coverage_summaries = [
            {
                "fsm_id":                  r.fsm_id,
                "state_coverage":          round(r.state_coverage_ratio, 3),
                "transition_coverage":     round(r.transition_coverage_ratio, 3),
                "boundary_state_coverage": round(r.boundary_state_coverage_ratio, 3),
                "boundary_trans_coverage": round(r.boundary_transition_coverage_ratio, 3),
                "uncovered_boundary_states": len(r.uncovered_boundary_state_ids),
                "uncovered_boundary_transitions": len(r.uncovered_boundary_transition_ids),
            }
            for r in result.coverage_reports
        ]

        # Failure breakdown by boundary kind
        failure_breakdown: dict[str, int] = {}
        for f in result.boundary_failures:
            failure_breakdown[f.boundary_kind.value] = (
                failure_breakdown.get(f.boundary_kind.value, 0) + 1
            )

        return {
            "target_id":                 result.target_id,
            "exit_criterion_met":        result.exit_criterion_met,
            "total_fsms":                result.total_fsms,
            "total_states":              result.total_states,
            "total_transitions":         result.total_transitions,
            "total_boundary_states":     result.total_boundary_states,
            "total_boundary_transitions": result.total_boundary_transitions,
            "total_scenarios":           result.total_scenarios,
            "scenarios_with_failures":   result.scenario_library.scenarios_with_failures,
            "total_boundary_failures":   result.total_boundary_failures,
            "security_relevant_failures": result.security_relevant_failures,
            "total_interpretation_mismatches": result.total_interpretation_mismatches,
            "mean_state_coverage":       round(result.mean_state_coverage_ratio, 3),
            "mean_boundary_coverage":    round(result.mean_boundary_coverage_ratio, 3),
            "failure_breakdown":         failure_breakdown,
            "fsms":                      fsm_summaries,
            "top_failures":              failure_summaries,
            "coverage":                  coverage_summaries,
        }

    # ── Exit criterion ────────────────────────────────────────────────────────

    @staticmethod
    def _check_exit_criterion(
        library: ScenarioLibrary,
        dataset: FailureAtBoundaryDataset,
    ) -> bool:
        """
        Phase 6 exit criterion:
        - ≥1 scenario reached a boundary (result != NOT_REACHED)
        - ≥1 BoundaryFailure with non-empty state_path + anomaly recorded
        - ≥1 FailureAtBoundaryEntry with state_path + transition + anomaly
        """
        scenario_reached = any(
            s.result != ScenarioResult.NOT_REACHED
            for s in library.scenarios.values()
        )
        has_failure = any(
            bool(e.failure.state_path) and bool(e.failure.anomaly_description)
            for e in dataset.entries
        )
        has_triad = any(
            bool(e.state_path)
            and bool(e.transition_event_name or e.transition_id)
            and bool(e.failure.anomaly_description)
            for e in dataset.entries
        )

        return scenario_reached and has_failure and has_triad
