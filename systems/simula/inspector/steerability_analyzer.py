"""
EcodiaOS — Inspector Phase 4: Steerability Analyzer (Orchestrator)

Single-entry orchestrator for the Phase 4 pipeline:

  StateModelExtractor → (StateVariables, InvariantSet)
  ConstraintEngine    → (ConstraintSet, SteerableRegions, ConditionSets)
  → SteerabilityModel
  → TransitionExplanations (one per interesting run or on demand)
  → Phase4Result

Usage
-----
  # From Phase 3 output only (no Phase 2 data):
  analyzer = SteerabilityAnalyzer()
  result = analyzer.analyze(
      phase3_result=phase3_result,
  )

  # From Phase 3 + Phase 2 runtime data (recommended):
  result = analyzer.analyze(
      phase3_result=phase3_result,
      phase2_result=phase2_result,
  )

  # Query: explain a specific trace segment
  explanation = analyzer.explain(
      result=result,
      trace_context={"func_names": ["allocate", "process"], "run_id": "run-42"},
  )

Exit criterion
--------------
Phase4Result.exit_criterion_met is True when:
  - At least one TransitionExplanation has been produced, AND
  - That explanation contains:
    - ≥1 broken invariant,
    - ≥1 relaxed constraint, AND
    - a non-empty steerable region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.constraint_solver import (
    ConstraintEngine,
)
from systems.simula.inspector.constraint_types import (
    ConditionSet,
    Constraint,
    Invariant,
    Phase4Result,
    SteerabilityClass,
    SteerabilityModel,
    SteerableRegion,
    TransitionExplanation,
    ViolationMechanism,
)
from systems.simula.inspector.state_model import StateModelExtractor

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import InvariantSet
    from systems.simula.inspector.runtime_types import Phase2Result
    from systems.simula.inspector.static_types import Phase3Result

logger = structlog.get_logger().bind(system="simula.inspector.steerability_analyzer")


class SteerabilityAnalyzer:
    """
    Phase 4 orchestrator — builds a Phase4Result for a target.

    The result combines:
    - A SteerabilityModel (state variables, invariants, constraints, regions, condition sets)
    - Per-run TransitionExplanations for all Phase 2 failure/crash runs
    - Overall steerability classification and probability

    Parameters
    ----------
    None — stateless; instantiate once and call analyze() per target.
    """

    def __init__(self) -> None:
        self._extractor = StateModelExtractor()
        self._engine    = ConstraintEngine()
        self._log       = logger

    # ── Primary entry point ───────────────────────────────────────────────────

    def analyze(
        self,
        phase3_result: Phase3Result,
        phase2_result: Phase2Result | None = None,
    ) -> Phase4Result:
        """
        Build a Phase4Result from Phase 3 (and optionally Phase 2) data.

        Args:
            phase3_result: Complete Phase 3 static analysis output.
            phase2_result: Optional Phase 2 runtime instrumentation output.
                           When provided, enables fault-driven invariant inference,
                           condition set support evidence, and per-run explanations.

        Returns:
            Phase4Result with full SteerabilityModel + TransitionExplanations.
        """
        target_id = phase3_result.target_id
        log = self._log.bind(target_id=target_id)
        log.info("steerability_analysis_started")

        # 1. Extract state model
        variables, invariant_set = self._extractor.extract(phase3_result, phase2_result)

        # 2. Build constraint system + steerable regions + condition sets
        constraint_set, regions, condition_sets = self._engine.build(
            target_id=target_id,
            invariant_set=invariant_set,
            variables=variables,
            phase3_result=phase3_result,
            phase2_result=phase2_result,
        )

        # 3. Determine overall steerability
        max_prob = max(
            (r.conditional_steerability_probability for r in regions),
            default=0.0,
        )
        target_class = self._classify_target(regions, invariant_set, phase3_result)

        # 4. Assemble SteerabilityModel
        model = SteerabilityModel(
            target_id=target_id,
            state_variables=variables,
            invariant_set=invariant_set,
            constraint_set=constraint_set,
            steerable_regions=regions,
            condition_sets=condition_sets,
            target_steerability_class=target_class,
            total_state_variables=len(variables),
            total_invariants=invariant_set.total_invariants,
            total_constraints=constraint_set.total_constraints,
            total_steerable_regions=len(regions),
            total_condition_sets=len(condition_sets),
            max_steerability_probability=round(max_prob, 4),
        )

        # 5. Generate per-run explanations from Phase 2 failure runs
        explanations = self._generate_explanations(
            model=model,
            phase2_result=phase2_result,
            phase3_result=phase3_result,
        )

        # 6. Assess exit criterion
        exit_met = self._check_exit_criterion(explanations)

        result = Phase4Result(
            target_id=target_id,
            model=model,
            explanations=explanations,
            total_state_variables=len(variables),
            total_invariants=invariant_set.total_invariants,
            total_constraints=constraint_set.total_constraints,
            total_steerable_regions=len(regions),
            total_condition_sets=len(condition_sets),
            total_explanations=len(explanations),
            total_broken_invariants=sum(len(e.broken_invariant_ids) for e in explanations),
            target_steerability_class=target_class,
            max_steerability_probability=round(max_prob, 4),
            exit_criterion_met=exit_met,
        )

        log.info(
            "steerability_analysis_complete",
            invariants=result.total_invariants,
            constraints=result.total_constraints,
            steerable_regions=result.total_steerable_regions,
            condition_sets=result.total_condition_sets,
            explanations=result.total_explanations,
            steerability=target_class.value,
            exit_criterion_met=exit_met,
        )

        return result

    # ── On-demand explanation ─────────────────────────────────────────────────

    def explain(
        self,
        result: Phase4Result,
        trace_context: dict,
    ) -> TransitionExplanation:
        """
        Produce a TransitionExplanation for a specific trace segment.

        ``trace_context`` is a dict with any of:
            func_names: list[str]   — function names from the trace
            bb_ids:     list[str]   — basic block IDs from the trace
            run_id:     str         — Phase 2 run ID (optional)
            fault_class: str        — FaultClass value (optional)

        Returns
        -------
        TransitionExplanation — structured three-step analysis.
        """
        model     = result.model
        run_id    = trace_context.get("run_id", "")
        func_names = trace_context.get("func_names", [])
        bb_ids     = trace_context.get("bb_ids", [])
        fault_class_val = trace_context.get("fault_class", "")

        # 1. Find matching steerable regions reachable from the trace

        # Determine broken invariants: blocks touched by trace → guarded invariants
        broken_invs: list[Invariant] = []
        broken_ids:  list[str] = []

        for bid in bb_ids:
            for inv in model.invariant_set.invariants_for_block(bid):
                if inv.invariant_id not in broken_ids:
                    broken_invs.append(inv)
                    broken_ids.append(inv.invariant_id)

        # Also include invariants matching the fault class
        if fault_class_val:
            for inv in model.invariant_set.invariants.values():
                if inv.derived_from_fault_class == fault_class_val:
                    if inv.invariant_id not in broken_ids:
                        broken_invs.append(inv)
                        broken_ids.append(inv.invariant_id)

        # Fallback: function-level match
        if not broken_invs:
            for inv in model.invariant_set.invariants.values():
                if any(fn in inv.description for fn in func_names):
                    if inv.invariant_id not in broken_ids:
                        broken_invs.append(inv)
                        broken_ids.append(inv.invariant_id)

        # 2. Find relaxed constraints (those referencing broken invariants)
        relaxed: list[Constraint] = []
        relaxed_ids: list[str] = []
        for iid in broken_ids:
            for c in model.constraint_set.constraints_for_invariant(iid):
                if c.constraint_id not in relaxed_ids:
                    relaxed.append(c)
                    relaxed_ids.append(c.constraint_id)

        # 3. Find best matching steerable region
        matched_region: SteerableRegion | None = None
        matched_cs: ConditionSet | None = None
        best_overlap = 0

        for region in model.steerable_regions:
            overlap = len(set(broken_ids) & set(
                region.required_violation_invariant_ids
                + region.alternative_violation_invariant_ids
            ))
            if overlap > best_overlap:
                best_overlap = overlap
                matched_region = region

        if matched_region:
            for cs in model.condition_sets:
                if cs.unlocked_region_id == matched_region.region_id:
                    matched_cs = cs
                    break

        # Compute confidence
        if broken_invs and relaxed and matched_region:
            base_conf = sum(inv.confidence for inv in broken_invs) / len(broken_invs)
            prob = matched_region.conditional_steerability_probability
            confidence = round((base_conf + prob) / 2, 4)
            steer_class = matched_region.steerability_class
        elif broken_invs:
            confidence = round(sum(inv.confidence for inv in broken_invs) / len(broken_invs) * 0.6, 4)
            steer_class = SteerabilityClass.CONDITIONALLY_STEERABLE
        else:
            confidence = 0.2
            steer_class = SteerabilityClass.UNKNOWN

        # Supporting fragments
        frag_ids = matched_region.high_interest_fragment_ids if matched_region else []

        # Build narrative
        narrative = self._build_narrative(
            func_names=func_names,
            broken_invs=broken_invs,
            relaxed=relaxed,
            region=matched_region,
            fault_class_val=fault_class_val,
        )

        return TransitionExplanation(
            target_id=result.target_id,
            run_id=run_id,
            broken_invariants=broken_invs,
            broken_invariant_ids=broken_ids,
            observed_violations=list({
                c.violation_mechanism
                for c in relaxed
                if c.violation_mechanism != ViolationMechanism.UNKNOWN
            }),
            relaxed_constraints=relaxed,
            relaxed_constraint_ids=relaxed_ids,
            steerable_region=matched_region,
            steerable_region_id=matched_region.region_id if matched_region else "",
            triggered_condition_set=matched_cs,
            condition_set_id=matched_cs.condition_set_id if matched_cs else "",
            steerability_class=steer_class,
            confidence=confidence,
            narrative=narrative,
            supporting_fragment_ids=frag_ids,
            supporting_run_ids=[run_id] if run_id else [],
        )

    # ── Per-run explanations ──────────────────────────────────────────────────

    def _generate_explanations(
        self,
        model: SteerabilityModel,
        phase2_result: Phase2Result | None,
        phase3_result: Phase3Result,
    ) -> list[TransitionExplanation]:
        """
        Generate one TransitionExplanation per failure/crash run in Phase 2 data.
        """
        if phase2_result is None:
            return []

        explanations: list[TransitionExplanation] = []

        for run_id, trace in phase2_result.dataset.traces.items():
            from systems.simula.inspector.runtime_types import RunCategory
            if trace.run_category not in (RunCategory.FAILURE, RunCategory.CRASH):
                continue

            faults = phase2_result.dataset.faults.get(run_id, [])
            fault_class_val = faults[0].fault_class.value if faults else ""

            # Collect bb_ids from this run's trace
            bb_ids: list[str] = []
            if trace.bb_trace:
                bb_ids = list(trace.bb_trace.hits.keys())

            expl = self.explain(
                result=Phase4Result(
                    target_id=model.target_id,
                    model=model,
                    explanations=[],
                    exit_criterion_met=False,
                ),
                trace_context={
                    "func_names": trace.functions_visited,
                    "bb_ids": bb_ids,
                    "run_id": run_id,
                    "fault_class": fault_class_val,
                },
            )

            # Only keep explanations with meaningful content
            if expl.broken_invariant_ids or expl.relaxed_constraint_ids:
                explanations.append(expl)

        return explanations

    # ── Classification ────────────────────────────────────────────────────────

    def _classify_target(
        self,
        regions: list[SteerableRegion],
        invariant_set: InvariantSet,
        phase3_result: Phase3Result,
    ) -> SteerabilityClass:
        """
        Overall target steerability = max class across all steerable regions,
        with a floor of DETERMINISTIC when no regions exist.
        """
        if not regions:
            return SteerabilityClass.DETERMINISTIC

        # Prefer regions with high-interest fragments
        hi_regions = [r for r in regions if r.high_interest_fragment_ids]
        candidates = hi_regions if hi_regions else regions

        max_class = SteerabilityClass.DETERMINISTIC
        class_order = [
            SteerabilityClass.DETERMINISTIC,
            SteerabilityClass.CONDITIONALLY_STEERABLE,
            SteerabilityClass.INFLUENCE_PERMISSIVE,
            SteerabilityClass.FULLY_STEERABLE,
        ]

        def _class_rank(c: SteerabilityClass) -> int:
            try:
                return class_order.index(c)
            except ValueError:
                return -1

        for region in candidates:
            if _class_rank(region.steerability_class) > _class_rank(max_class):
                max_class = region.steerability_class

        return max_class

    # ── Exit criterion ────────────────────────────────────────────────────────

    @staticmethod
    def _check_exit_criterion(explanations: list[TransitionExplanation]) -> bool:
        """
        Phase 4 exit criterion:
        At least one explanation must have broken invariants, relaxed constraints,
        AND a non-empty steerable region.
        """
        for expl in explanations:
            if (
                expl.broken_invariant_ids
                and expl.relaxed_constraint_ids
                and expl.steerable_region_id
            ):
                return True
        return False

    # ── Narrative builder ─────────────────────────────────────────────────────

    @staticmethod
    def _build_narrative(
        func_names: list[str],
        broken_invs: list[Invariant],
        relaxed: list[Constraint],
        region: SteerableRegion | None,
        fault_class_val: str,
    ) -> str:
        """
        Build a one-paragraph prose explanation of the transition.
        """
        fn_list = ", ".join(func_names[:3]) + ("…" if len(func_names) > 3 else "")

        if not broken_invs:
            return (
                f"Trace through {fn_list or 'unknown functions'} did not map to any "
                f"known invariant violations. Insufficient evidence for steerability determination."
            )

        inv_desc = broken_invs[0].description[:120]
        constraint_count = len(relaxed)
        region_block_count = len(region.block_ids) if region else 0
        region_funcs = ", ".join((region.function_names or [])[:3]) if region else "unknown"
        fault_suffix = f" (fault class: {fault_class_val})" if fault_class_val else ""

        return (
            f"When execution passes through {fn_list}{fault_suffix}, "
            f"the invariant '{inv_desc}' is violated{', among others' if len(broken_invs) > 1 else ''}. "
            f"This violation relaxes {constraint_count} constraint{'s' if constraint_count != 1 else ''}, "
            f"enabling control flow to reach a previously inaccessible region "
            f"spanning {region_block_count} basic block{'s' if region_block_count != 1 else ''} "
            f"in function{'s' if len(region.function_names or []) != 1 else ''} [{region_funcs}]. "
            f"Within that region, external input can select among multiple control-flow continuations, "
            f"satisfying the influence-permissive steerability criterion."
            if region else
            f"When execution passes through {fn_list}{fault_suffix}, "
            f"the invariant '{inv_desc}' is violated. "
            f"This relaxes {constraint_count} constraint{'s' if constraint_count != 1 else ''}, "
            f"but no mapped steerable region was identified for this trace segment. "
            f"Further static analysis may be required."
        )

    # ── Reporting helper ──────────────────────────────────────────────────────

    def model_summary(self, result: Phase4Result) -> dict:
        """
        Return a concise reporting dict suitable for logging or display.
        """
        model = result.model

        region_summaries = [
            {
                "region_id":      r.region_id,
                "functions":      r.function_names,
                "block_count":    len(r.block_ids),
                "steerability":   r.steerability_class.value,
                "probability":    r.conditional_steerability_probability,
                "high_interest_fragments": len(r.high_interest_fragment_ids),
                "observed_in_failure": r.observed_in_failure,
            }
            for r in model.steerable_regions
        ]

        condition_summaries = [
            {
                "condition_set_id": cs.condition_set_id,
                "description":      cs.description[:100],
                "required_violations": len(cs.required_violations),
                "mechanisms":       [m.value for m in cs.violation_mechanisms],
                "confidence":       cs.confidence,
                "supporting_runs":  len(cs.supporting_run_ids),
            }
            for cs in model.condition_sets
        ]

        explanation_summaries = [
            {
                "run_id":           e.run_id,
                "broken_invariants": len(e.broken_invariant_ids),
                "relaxed_constraints": len(e.relaxed_constraint_ids),
                "steerable_region_id": e.steerable_region_id,
                "steerability_class": e.steerability_class.value,
                "confidence":       e.confidence,
            }
            for e in result.explanations
        ]

        return {
            "target_id":                  result.target_id,
            "target_steerability_class":  result.target_steerability_class.value,
            "max_steerability_probability": result.max_steerability_probability,
            "exit_criterion_met":         result.exit_criterion_met,
            "state_variables":            result.total_state_variables,
            "invariants":                 result.total_invariants,
            "constraints":                result.total_constraints,
            "steerable_regions":          result.total_steerable_regions,
            "condition_sets":             result.total_condition_sets,
            "explanations":               result.total_explanations,
            "broken_invariants_total":    result.total_broken_invariants,
            "steerable_regions_detail":   region_summaries,
            "condition_sets_detail":      condition_summaries,
            "explanations_detail":        explanation_summaries,
        }
