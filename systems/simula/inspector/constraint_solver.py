"""
EcodiaOS - Inspector Phase 4: Constraint Engine

Converts an InvariantSet + ExecutionAtlas into a ConstraintSet, then derives:
  - SteerableRegion entries (CFG sub-regions unlocked by invariant violations)
  - ConditionSet entries (testable predicates for the deterministic→permissive shift)
  - SteerabilityClass assignment per region and overall target

Algorithm
---------
Pass 1 - Invariant → constraint projection
  For each invariant, emit one or more Constraints based on its kind:
    MEMORY_REGION invariant   → REACHABILITY constraint (bounds check must hold)
    OBJECT_LIFETIME invariant → LIFETIME constraint (alloc/free ordering)
    FUNCTION_POINTER invariant → REACHABILITY + PRECONDITION (fp must be valid)
    TYPE_TAG invariant         → PRECONDITION (type check must pass)
    PROTOCOL_STATE invariant   → ORDERING (correct FSM transitions only)
    IDENTITY_CONTEXT invariant → PRECONDITION (privilege check)
    EXCEPTION_STATE invariant  → REACHABILITY (exception paths)
    LOOP_COUNTER invariant     → PRECONDITION (loop bound check)
    REGISTER/UNKNOWN           → generic REACHABILITY

Pass 2 - CFG reachability expansion
  For each constraint's unlocked blocks, run CFG BFS from those blocks to find
  all additionally reachable blocks (the full steerable region).

Pass 3 - SteerableRegion construction
  Group constraints by unlocked-region overlap.  Regions with ≥2 fragments
  or containing an indirect-dispatch fragment get elevated steerability class.

Pass 4 - ConditionSet derivation
  For each SteerableRegion, find the minimal set of invariants that must ALL
  be violated to reach it.  That set is one ConditionSet.  If Phase 2 run data
  shows InfluencePermissiveTransitions into the region, boost probability.

Pass 5 - Target steerability classification
  Apply INFLUENCE_PERMISSIVE when max probability ≥ 0.4 AND the target has
  at least one steerable region with an indirect-dispatch or taint-reachable
  fragment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.constraint_types import (
    ConditionSet,
    Constraint,
    ConstraintKind,
    ConstraintSet,
    Invariant,
    InvariantStrength,
    StateVariableKind,
    SteerabilityClass,
    SteerableRegion,
    ViolationMechanism,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import (
        InvariantSet,
        StateVariable,
    )
    from systems.simula.inspector.runtime_types import Phase2Result
    from systems.simula.inspector.static_types import (
        ExecutionAtlas,
        Phase3Result,
    )

logger = structlog.get_logger().bind(system="simula.inspector.constraint_solver")


# ── Kind → constraint kind mapping ────────────────────────────────────────────

_KIND_TO_CONSTRAINT: dict[StateVariableKind, ConstraintKind] = {
    StateVariableKind.REGISTER:          ConstraintKind.PRECONDITION,
    StateVariableKind.MEMORY_REGION:     ConstraintKind.REACHABILITY,
    StateVariableKind.OBJECT_LIFETIME:   ConstraintKind.LIFETIME,
    StateVariableKind.TYPE_TAG:          ConstraintKind.PRECONDITION,
    StateVariableKind.FUNCTION_POINTER:  ConstraintKind.REACHABILITY,
    StateVariableKind.PROTOCOL_STATE:    ConstraintKind.ORDERING,
    StateVariableKind.IDENTITY_CONTEXT:  ConstraintKind.PRECONDITION,
    StateVariableKind.TAINT_LABEL:       ConstraintKind.TAINT_PROPAGATION,
    StateVariableKind.LOOP_COUNTER:      ConstraintKind.PRECONDITION,
    StateVariableKind.EXCEPTION_STATE:   ConstraintKind.REACHABILITY,
    StateVariableKind.UNKNOWN:           ConstraintKind.REACHABILITY,
}


def _invariant_to_constraint_kind(inv: Invariant) -> ConstraintKind:
    return _KIND_TO_CONSTRAINT.get(inv.kind, ConstraintKind.REACHABILITY)


def _steerability_from_probability(probability: float) -> SteerabilityClass:
    if probability >= 0.75:
        return SteerabilityClass.FULLY_STEERABLE
    if probability >= 0.40:
        return SteerabilityClass.INFLUENCE_PERMISSIVE
    if probability > 0.0:
        return SteerabilityClass.CONDITIONALLY_STEERABLE
    return SteerabilityClass.DETERMINISTIC


# ── ConstraintEngine ───────────────────────────────────────────────────────────


class ConstraintEngine:
    """
    Converts InvariantSet + ExecutionAtlas into ConstraintSet + SteerableRegions
    + ConditionSets.

    This is the formal core of Phase 4.  It does not require an external solver
    (Z3 is used by Phase 5 if needed); all inference here is structural and
    heuristic, driven by the Phase 3 CFG + catalog.
    """

    def __init__(self) -> None:
        self._log = logger

    def build(
        self,
        target_id: str,
        invariant_set: InvariantSet,
        variables: dict[str, StateVariable],
        phase3_result: Phase3Result,
        phase2_result: Phase2Result | None = None,
    ) -> tuple[ConstraintSet, list[SteerableRegion], list[ConditionSet]]:
        """
        Build the full constraint system + steerable regions + condition sets.

        Returns
        -------
        (constraint_set, steerable_regions, condition_sets)
        """
        atlas = phase3_result.atlas
        log   = self._log.bind(target_id=target_id)

        # Pass 1: invariants → constraints
        cset = self._project_invariants(target_id, invariant_set, atlas)

        # Pass 2: CFG expansion of unlocked blocks
        self._expand_reachability(cset, atlas)

        # Pass 3: group into steerable regions
        regions = self._build_steerable_regions(target_id, cset, invariant_set, atlas)

        # Pass 4: derive condition sets
        condition_sets = self._derive_condition_sets(target_id, regions, invariant_set, phase2_result)

        # Pass 5: estimate steerability probabilities
        self._estimate_probabilities(regions, condition_sets, phase2_result, atlas)

        log.info(
            "constraint_engine_complete",
            constraints=cset.total_constraints,
            steerable_regions=len(regions),
            condition_sets=len(condition_sets),
        )

        return cset, regions, condition_sets

    # ── Pass 1 ────────────────────────────────────────────────────────────────

    def _project_invariants(
        self,
        target_id: str,
        invariant_set: InvariantSet,
        atlas: ExecutionAtlas,
    ) -> ConstraintSet:
        cset = ConstraintSet(target_id=target_id)

        for inv in invariant_set.invariants.values():
            ckind   = _invariant_to_constraint_kind(inv)
            mechanism = ViolationMechanism.UNKNOWN

            # Derive mechanism from fault class if available
            if inv.derived_from_fault_class:
                from systems.simula.inspector.state_model import _fault_class_to_invariant
                _, _, mechanism = _fault_class_to_invariant(inv.derived_from_fault_class)

            # Build formal expression
            formal = self._formal_expression(inv, ckind)

            c = Constraint(
                target_id=target_id,
                kind=ckind,
                description=(
                    f"[{ckind.value}] {inv.description}"
                ),
                formal_expression=formal,
                variable_ids=inv.variable_ids,
                invariant_ids=[inv.invariant_id],
                scope_block_ids=inv.guarded_block_ids[:],
                violation_mechanism=mechanism,
                unlocks_block_ids=inv.violation_unlocks_block_ids[:],
                confidence=inv.confidence,
                violated_in_failure=(inv.strength == InvariantStrength.MUST and bool(inv.derived_from_fault_class)),
                evidence_fragment_ids=inv.derived_from_fragment_ids[:],
            )
            cset.add_constraint(c)

        return cset

    @staticmethod
    def _formal_expression(inv: Invariant, ckind: ConstraintKind) -> str:
        """
        Build a lightweight symbolic expression string for the constraint.

        Not a complete formal logic encoding - used for readability and
        downstream reporting.  Format mirrors Z3 python API loosely.
        """
        match ckind:
            case ConstraintKind.PRECONDITION:
                return f"pre({inv.description[:80]})"
            case ConstraintKind.REACHABILITY:
                if inv.violation_unlocks_block_ids:
                    blocks = inv.violation_unlocks_block_ids[:3]
                    suffix = "…" if len(inv.violation_unlocks_block_ids) > 3 else ""
                    return f"¬{inv.kind.value}({inv.description[:40]}) → reachable({{{', '.join(blocks)}{suffix}}})"
                return f"¬{inv.kind.value}({inv.description[:60]}) → new_region"
            case ConstraintKind.LIFETIME:
                return f"lifetime_intact({inv.description[:60]})"
            case ConstraintKind.ORDERING:
                return f"ordering({inv.description[:60]})"
            case ConstraintKind.TAINT_PROPAGATION:
                return f"¬taint_check({inv.description[:60]}) → taint_reaches_sink"
            case _:
                return inv.description[:80]

    # ── Pass 2 ────────────────────────────────────────────────────────────────

    def _expand_reachability(
        self,
        cset: ConstraintSet,
        atlas: ExecutionAtlas,
    ) -> None:
        """
        For each constraint, expand its unlocked_block_ids via CFG BFS (depth 3).

        This gives the full region that becomes reachable, not just the
        immediate 'unlock'.
        """
        for c in cset.constraints.values():
            if not c.unlocks_block_ids:
                continue
            expanded: set[str] = set(c.unlocks_block_ids)
            for bid in list(c.unlocks_block_ids):
                expanded.update(atlas.cfg.reachable_from(bid, max_depth=3))
            c.unlocks_block_ids = list(expanded)

    # ── Pass 3 ────────────────────────────────────────────────────────────────

    def _build_steerable_regions(
        self,
        target_id: str,
        cset: ConstraintSet,
        invariant_set: InvariantSet,
        atlas: ExecutionAtlas,
    ) -> list[SteerableRegion]:
        """
        Group constraints by overlapping unlock sets into SteerableRegions.

        Two constraints that unlock overlapping block sets are merged into the
        same region.  Disjoint sets produce separate regions.
        """
        regions: list[SteerableRegion] = []
        assigned_constraints: set[str] = set()

        constraints_with_unlocks = [
            c for c in cset.constraints.values() if c.unlocks_block_ids
        ]

        for c in constraints_with_unlocks:
            if c.constraint_id in assigned_constraints:
                continue

            region_blocks: set[str] = set(c.unlocks_block_ids)
            region_constraints: list[str] = [c.constraint_id]
            region_invariants: list[str] = list(c.invariant_ids)
            assigned_constraints.add(c.constraint_id)

            # Merge overlapping constraints
            changed = True
            while changed:
                changed = False
                for other in constraints_with_unlocks:
                    if other.constraint_id in assigned_constraints:
                        continue
                    if region_blocks & set(other.unlocks_block_ids):
                        region_blocks.update(other.unlocks_block_ids)
                        region_constraints.append(other.constraint_id)
                        for iid in other.invariant_ids:
                            if iid not in region_invariants:
                                region_invariants.append(iid)
                        assigned_constraints.add(other.constraint_id)
                        changed = True

            # Collect functions in this region
            region_funcs: list[str] = list({
                atlas.cfg.block_index[bid].func_name
                for bid in region_blocks
                if bid in atlas.cfg.block_index and atlas.cfg.block_index[bid].func_name
            })

            # Collect fragments in this region
            region_frags: list[str] = []
            high_interest: list[str] = []
            for bid in region_blocks:
                for fid in atlas.catalog.block_index.get(bid, []):
                    region_frags.append(fid)
                    frag = atlas.catalog.fragments.get(fid)
                    if frag and (frag.is_indirect_dispatch or frag.is_fault_adjacent or frag.taint_reachable):
                        high_interest.append(fid)

            # Check FA region overlap
            observed_in_failure = any(
                bool(set(region_blocks) & set(fa.block_ids))
                for fa in atlas.failure_adjacent_regions
            )

            # Required violations (ALL must hold): use the intersection
            # i.e., the invariants that guard the *entry* to this region
            required_viol: list[str] = []
            alt_viol: list[str] = []
            if region_invariants:
                # Simple heuristic: MUST-strength invariants are required; others alternative
                for iid in region_invariants:
                    inv = invariant_set.invariants.get(iid)
                    if inv and inv.strength == InvariantStrength.MUST:
                        required_viol.append(iid)
                    elif iid not in required_viol:
                        alt_viol.append(iid)

            region = SteerableRegion(
                target_id=target_id,
                block_ids=list(region_blocks),
                function_names=region_funcs,
                required_violation_invariant_ids=required_viol,
                alternative_violation_invariant_ids=alt_viol,
                relaxed_constraint_ids=region_constraints,
                steerability_class=SteerabilityClass.UNKNOWN,
                fragment_ids=list(set(region_frags)),
                observed_in_failure=observed_in_failure,
                high_interest_fragment_ids=list(set(high_interest)),
            )
            regions.append(region)

        return regions

    # ── Pass 4 ────────────────────────────────────────────────────────────────

    def _derive_condition_sets(
        self,
        target_id: str,
        regions: list[SteerableRegion],
        invariant_set: InvariantSet,
        phase2_result: Phase2Result | None,
    ) -> list[ConditionSet]:
        """
        For each SteerableRegion produce one ConditionSet - the conjunction of
        invariant violations that unlocks that region.
        """
        condition_sets: list[ConditionSet] = []

        # Build run_id → set of visited function names for Phase 2 support
        run_func_map: dict[str, set[str]] = {}
        if phase2_result is not None:
            for run_id, trace in phase2_result.dataset.traces.items():
                run_func_map[run_id] = set(trace.functions_visited)

        for region in regions:
            required = region.required_violation_invariant_ids
            if not required:
                required = region.alternative_violation_invariant_ids

            if not required:
                continue

            # Human-readable description
            inv_descs = []
            mechanisms: list[ViolationMechanism] = []
            for iid in required:
                inv = invariant_set.invariants.get(iid)
                if inv:
                    inv_descs.append(inv.description[:60])
                    # Infer mechanism from fault class
                    if inv.derived_from_fault_class:
                        from systems.simula.inspector.state_model import _fault_class_to_invariant
                        _, _, mech = _fault_class_to_invariant(inv.derived_from_fault_class)
                        if mech not in mechanisms:
                            mechanisms.append(mech)

            description = (
                f"Steerable region {region.region_id[:8]} unlocked when: "
                + "; AND ".join(inv_descs[:3])
                + ("…" if len(inv_descs) > 3 else "")
            )

            formal = " ∧ ".join(
                f"¬invariant({iid[:8]})"
                for iid in required
            ) + f" → steerable({region.region_id[:8]})"

            # Supporting run_ids: Phase 2 failure runs that visited any region function
            supporting_runs: list[str] = []
            if phase2_result is not None:
                for run_id, trace in phase2_result.dataset.traces.items():
                    from systems.simula.inspector.runtime_types import RunCategory
                    if trace.run_category in (RunCategory.FAILURE, RunCategory.CRASH):
                        if any(fn in run_func_map.get(run_id, set()) for fn in region.function_names):
                            supporting_runs.append(run_id)

            confidence = max(
                (invariant_set.invariants[iid].confidence
                 for iid in required
                 if iid in invariant_set.invariants),
                default=0.3,
            )

            cs = ConditionSet(
                target_id=target_id,
                description=description,
                required_violations=required,
                unlocked_region_id=region.region_id,
                violation_mechanisms=mechanisms,
                confidence=confidence,
                supporting_run_ids=supporting_runs,
                formal_trigger=formal,
            )
            condition_sets.append(cs)

        return condition_sets

    # ── Pass 5 ────────────────────────────────────────────────────────────────

    def _estimate_probabilities(
        self,
        regions: list[SteerableRegion],
        condition_sets: list[ConditionSet],
        phase2_result: Phase2Result | None,
        atlas: ExecutionAtlas,
    ) -> None:
        """
        Estimate conditional steerability probability for each SteerableRegion.

        P(steerable | invariants violated) is estimated from:
        - Fraction of failure runs that visited the region (from condition set evidence)
        - Presence of indirect-dispatch or taint-reachable high-interest fragments
        - Phase 2 mean permissive transition count (if available)
        """
        total_failure_runs = 0
        mean_permissive_count = 0.0

        if phase2_result is not None:
            dataset = phase2_result.dataset
            total_failure_runs = dataset.failure_run_count + dataset.crash_run_count
            if phase2_result.scores:
                mean_permissive_count = sum(
                    s.permissive_transition_count
                    for s in phase2_result.scores
                ) / len(phase2_result.scores)

        # Build region_id → ConditionSet map
        cs_by_region: dict[str, ConditionSet] = {
            cs.unlocked_region_id: cs for cs in condition_sets
        }

        for region in regions:
            cs = cs_by_region.get(region.region_id)

            # Base: fraction of failure runs supporting this region
            support_fraction = 0.0
            if cs and total_failure_runs > 0:
                support_fraction = len(cs.supporting_run_ids) / total_failure_runs

            # Boost for high-interest fragments
            interest_boost = 0.0
            if region.high_interest_fragment_ids:
                n_hi = len(region.high_interest_fragment_ids)
                n_all = max(1, len(region.fragment_ids))
                interest_boost = min(0.3, 0.15 * (n_hi / n_all) * 2)

            # Boost from permissive transition count
            transition_boost = min(0.2, mean_permissive_count * 0.02)

            probability = min(1.0, support_fraction + interest_boost + transition_boost)
            region.conditional_steerability_probability = round(probability, 4)
            region.steerability_class = _steerability_from_probability(probability)
