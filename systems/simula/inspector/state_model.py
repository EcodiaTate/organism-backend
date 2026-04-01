"""
EcodiaOS - Inspector Phase 4: State Model Extractor

Extracts the StateModel (state variables + invariants) from Phase 3 artefacts
and Phase 2 runtime observations.

Algorithm
---------
1. Walk the FragmentCatalog - each fragment semantic maps to a category of
   state variable (MEMORY_WRITE → MEMORY_REGION, INDIRECT_BRANCH →
   FUNCTION_POINTER, ALLOC/FREE → OBJECT_LIFETIME, etc.).

2. Walk the FailureAdjacentRegions - each region represents a CFG subgraph
   that was only reached under fault conditions.  Infer invariants: "function F
   must not reach block B in normal execution", backed by the Phase 2 fault
   classification.

3. Walk the Phase 2 ControlIntegrityScores - each
   InfluencePermissiveTransition gives a direct (from_func, to_func) edge that
   was anomalous.  Infer ORDERING and REACHABILITY invariants.

4. Walk the Phase 2 FaultObservations - each fault_class maps to a specific
   StateVariableKind and InvariantStrength (OOB → MEMORY_REGION / MUST,
   UAF → OBJECT_LIFETIME / MUST, TYPE → TYPE_TAG / MUST, etc.).

5. Deduplicate invariants by (kind, guarded_block_ids hash) and merge
   supporting evidence.

Output
------
InvariantSet  - the complete set of inferred invariants.
dict[str, StateVariable]  - all extracted state variables.

These feed directly into ConstraintEngine.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.constraint_types import (
    Invariant,
    InvariantSet,
    InvariantStrength,
    StateVariable,
    StateVariableKind,
    ViolationMechanism,
)
from systems.simula.inspector.static_types import (
    FragmentSemantics,
)

if TYPE_CHECKING:
    from systems.simula.inspector.runtime_types import (
        Phase2Result,
    )
    from systems.simula.inspector.static_types import (
        CodeFragment,
        ExecutionAtlas,
        Phase3Result,
    )

logger = structlog.get_logger().bind(system="simula.inspector.state_model")


# ── Semantic → state variable kind mapping ────────────────────────────────────

_SEMANTICS_TO_VAR_KIND: dict[str, StateVariableKind] = {
    FragmentSemantics.MEMORY_READ:    StateVariableKind.MEMORY_REGION,
    FragmentSemantics.MEMORY_WRITE:   StateVariableKind.MEMORY_REGION,
    FragmentSemantics.INDIRECT_BRANCH: StateVariableKind.FUNCTION_POINTER,
    FragmentSemantics.SYSCALL_GATEWAY: StateVariableKind.PROTOCOL_STATE,
    FragmentSemantics.ALLOC:           StateVariableKind.OBJECT_LIFETIME,
    FragmentSemantics.FREE:            StateVariableKind.OBJECT_LIFETIME,
    FragmentSemantics.STRING_OP:       StateVariableKind.MEMORY_REGION,
    FragmentSemantics.ARITHMETIC:      StateVariableKind.REGISTER,
    FragmentSemantics.COMPARISON:      StateVariableKind.REGISTER,
    FragmentSemantics.LOOP_HEAD:       StateVariableKind.LOOP_COUNTER,
    FragmentSemantics.EXCEPTION_SITE:  StateVariableKind.EXCEPTION_STATE,
    FragmentSemantics.RETURN_SITE:     StateVariableKind.REGISTER,
    FragmentSemantics.CALL_CHAIN:      StateVariableKind.FUNCTION_POINTER,
    FragmentSemantics.UNKNOWN:         StateVariableKind.UNKNOWN,
}


def _fragment_to_var_kind(frag: CodeFragment) -> StateVariableKind:
    return _SEMANTICS_TO_VAR_KIND.get(frag.semantics, StateVariableKind.UNKNOWN)


# ── Fault class → invariant kind + strength ───────────────────────────────────

def _fault_class_to_invariant(fault_class_value: str) -> tuple[StateVariableKind, InvariantStrength, ViolationMechanism]:
    """
    Map a FaultClass string value to (StateVariableKind, InvariantStrength, ViolationMechanism).

    Returns the most specific triple possible; defaults to UNKNOWN/MUST/UNKNOWN.
    """
    mapping: dict[str, tuple[StateVariableKind, InvariantStrength, ViolationMechanism]] = {
        "oob":               (StateVariableKind.MEMORY_REGION,   InvariantStrength.MUST,   ViolationMechanism.BOUNDARY_VIOLATION),
        "uaf_dangling":      (StateVariableKind.OBJECT_LIFETIME,  InvariantStrength.MUST,   ViolationMechanism.LIFETIME_VIOLATION),
        "lifetime_error":    (StateVariableKind.OBJECT_LIFETIME,  InvariantStrength.MUST,   ViolationMechanism.LIFETIME_VIOLATION),
        "type_confusion":    (StateVariableKind.TYPE_TAG,          InvariantStrength.MUST,   ViolationMechanism.TYPE_MISMATCH),
        "unhandled_exception":(StateVariableKind.EXCEPTION_STATE, InvariantStrength.SHOULD, ViolationMechanism.UNKNOWN),
        "signal_abort":      (StateVariableKind.EXCEPTION_STATE,  InvariantStrength.MUST,   ViolationMechanism.UNKNOWN),
        "signal_segv":       (StateVariableKind.MEMORY_REGION,    InvariantStrength.MUST,   ViolationMechanism.BOUNDARY_VIOLATION),
        "signal_bus":        (StateVariableKind.MEMORY_REGION,    InvariantStrength.MUST,   ViolationMechanism.BOUNDARY_VIOLATION),
        "signal_fpe":        (StateVariableKind.REGISTER,         InvariantStrength.MUST,   ViolationMechanism.ARITHMETIC_OVERFLOW),
        "signal_other":      (StateVariableKind.UNKNOWN,           InvariantStrength.MUST,   ViolationMechanism.UNKNOWN),
        "logic_invariant":   (StateVariableKind.REGISTER,         InvariantStrength.SHOULD, ViolationMechanism.UNKNOWN),
        "unknown":           (StateVariableKind.UNKNOWN,           InvariantStrength.MUST,   ViolationMechanism.UNKNOWN),
    }
    return mapping.get(fault_class_value, (StateVariableKind.UNKNOWN, InvariantStrength.MUST, ViolationMechanism.UNKNOWN))


def _block_set_key(block_ids: list[str]) -> str:
    """Stable hash key for a set of block IDs (for deduplication)."""
    return hashlib.sha1("|".join(sorted(block_ids)).encode()).hexdigest()[:16]


# ── StateModelExtractor ────────────────────────────────────────────────────────


class StateModelExtractor:
    """
    Extracts StateVariables and an InvariantSet from Phase 3 + Phase 2 data.

    This is the first pass of the Phase 4 pipeline.  It does NOT require a
    formal solver - all inference is heuristic, driven by:
    - fragment semantics (what operations the code performs),
    - failure-adjacent regions (what code only runs when things go wrong),
    - influence-permissive transitions (where control was anomalously redirected),
    - fault observations (what class of error was detected).

    The resulting InvariantSet is the input to ConstraintEngine.
    """

    def __init__(self) -> None:
        self._log = logger

    def extract(
        self,
        phase3_result: Phase3Result,
        phase2_result: Phase2Result | None = None,
    ) -> tuple[dict[str, StateVariable], InvariantSet]:
        """
        Extract state variables and invariants from Phase 3 (and optionally Phase 2) data.

        Returns
        -------
        (variables, invariant_set) - both indexed by their respective IDs.
        """
        target_id = phase3_result.target_id
        atlas     = phase3_result.atlas
        log       = self._log.bind(target_id=target_id)

        variables:   dict[str, StateVariable] = {}
        inv_set = InvariantSet(target_id=target_id)

        # Pass 1: fragment catalog → state variables
        self._extract_from_fragments(target_id, atlas, variables, inv_set)

        # Pass 2: failure-adjacent regions → reachability invariants
        self._extract_from_fa_regions(target_id, atlas, variables, inv_set)

        # Pass 3: Phase 2 fault observations → fault-class invariants
        if phase2_result is not None:
            self._extract_from_faults(target_id, phase2_result, variables, inv_set)

        # Pass 4: Phase 2 influence-permissive transitions → ordering invariants
        if phase2_result is not None:
            self._extract_from_transitions(target_id, phase2_result, variables, inv_set)

        log.info(
            "state_model_extracted",
            state_variables=len(variables),
            invariants=inv_set.total_invariants,
        )

        return variables, inv_set

    # ── Pass 1: fragments ─────────────────────────────────────────────────────

    def _extract_from_fragments(
        self,
        target_id: str,
        atlas: ExecutionAtlas,
        variables: dict[str, StateVariable],
        inv_set: InvariantSet,
    ) -> None:
        """
        One StateVariable per unique (func_name, kind) pair found in the catalog.
        Fault-adjacent fragments also get a SHOULD/MAY reachability invariant.
        """
        # Group fragments by (func_name, var_kind) to merge into single variables
        seen: dict[tuple[str, str], StateVariable] = {}

        for frag in atlas.catalog.fragments.values():
            kind = _fragment_to_var_kind(frag)
            key  = (frag.func_name or frag.file_path, kind.value)

            if key not in seen:
                var = StateVariable(
                    target_id=target_id,
                    kind=kind,
                    name=f"{frag.func_name or frag.file_path}:{kind.value}",
                    func_name=frag.func_name,
                    file_path=frag.file_path,
                    taint_reachable=frag.taint_reachable,
                    fragment_ids=[frag.fragment_id],
                )
                if frag.block_id:
                    var.read_in_blocks.append(frag.block_id)
                seen[key] = var
                variables[var.var_id] = var
            else:
                var = seen[key]
                var.fragment_ids.append(frag.fragment_id)
                if frag.taint_reachable:
                    var.taint_reachable = True
                if frag.block_id and frag.block_id not in var.read_in_blocks:
                    var.read_in_blocks.append(frag.block_id)

            # Fault-adjacent indirect dispatch fragments → reachability invariant
            if frag.is_fault_adjacent and frag.is_indirect_dispatch and frag.block_id:
                var = seen[key]
                description = (
                    f"Indirect dispatch in {frag.func_name or frag.file_path} "
                    f"(block {frag.block_id}) must not be reached in normal execution"
                )
                self._add_invariant_if_new(
                    inv_set=inv_set,
                    target_id=target_id,
                    description=description,
                    strength=InvariantStrength.SHOULD,
                    kind=kind,
                    variable_ids=[var.var_id],
                    guarded_block_ids=[frag.block_id],
                    violation_unlocks_block_ids=list(frag.reachable_block_ids),
                    derived_from_fragment_ids=[frag.fragment_id],
                    confidence=0.6,
                )

    # ── Pass 2: failure-adjacent regions ──────────────────────────────────────

    def _extract_from_fa_regions(
        self,
        target_id: str,
        atlas: ExecutionAtlas,
        variables: dict[str, StateVariable],
        inv_set: InvariantSet,
    ) -> None:
        """
        Each FailureAdjacentRegion → one SHOULD/MUST reachability invariant:
        "this sub-graph must not be entered in normal execution."
        """
        for region in atlas.failure_adjacent_regions:
            if not region.block_ids:
                continue

            strength = InvariantStrength.MUST if region.contains_fault_site else InvariantStrength.SHOULD
            confidence = min(0.9, 0.4 + region.failure_coverage)

            desc = (
                f"Functions {region.functions} must not enter failure-adjacent "
                f"region {region.region_id[:8]} "
                f"({region.failure_run_count} failure runs, "
                f"{region.normal_run_count} normal runs)"
            )

            inv = self._add_invariant_if_new(
                inv_set=inv_set,
                target_id=target_id,
                description=desc,
                strength=strength,
                kind=StateVariableKind.UNKNOWN,
                variable_ids=[],
                guarded_block_ids=[],
                violation_unlocks_block_ids=region.block_ids,
                derived_from_fragment_ids=region.fragment_ids,
                confidence=confidence,
            )

            if inv is not None:
                # Associate relevant taint variables with this invariant
                taint_vars = [
                    vid for vid, v in variables.items()
                    if v.taint_reachable
                    and any(b in region.block_ids for b in v.read_in_blocks)
                ]
                inv.variable_ids.extend(taint_vars)

    # ── Pass 3: fault observations ────────────────────────────────────────────

    def _extract_from_faults(
        self,
        target_id: str,
        phase2_result: Phase2Result,
        variables: dict[str, StateVariable],
        inv_set: InvariantSet,
    ) -> None:
        """
        Each unique (fault_class, fault_at_func) pair → one MUST invariant.
        """
        seen_fault_keys: set[str] = set()

        for _run_id, faults in phase2_result.dataset.faults.items():
            for fault in faults:
                fault_key = f"{fault.fault_class.value}::{fault.fault_at_func}"
                if fault_key in seen_fault_keys:
                    continue
                seen_fault_keys.add(fault_key)

                var_kind, strength, mechanism = _fault_class_to_invariant(fault.fault_class.value)

                # Look up a variable matching this function + kind
                matching_vars = [
                    vid for vid, v in variables.items()
                    if v.func_name == fault.fault_at_func and v.kind == var_kind
                ]

                desc = (
                    f"{fault.fault_class.value.upper()} in {fault.fault_at_func}: "
                    f"state invariant violated (confidence {fault.confidence:.2f})"
                )

                # Determine which blocks become reachable when this fault occurs
                # - use the atlas CFG reachability from the fault function
                unlocks: list[str] = []
                if fault.fault_at_func in phase2_result.dataset.traces:
                    pass  # Could walk CFG here; keep lightweight for now

                self._add_invariant_if_new(
                    inv_set=inv_set,
                    target_id=target_id,
                    description=desc,
                    strength=strength,
                    kind=var_kind,
                    variable_ids=matching_vars,
                    guarded_block_ids=[],
                    violation_unlocks_block_ids=unlocks,
                    derived_from_fault_class=fault.fault_class.value,
                    confidence=fault.confidence * 0.85,
                )

    # ── Pass 4: influence-permissive transitions ──────────────────────────────

    def _extract_from_transitions(
        self,
        target_id: str,
        phase2_result: Phase2Result,
        variables: dict[str, StateVariable],
        inv_set: InvariantSet,
    ) -> None:
        """
        Each unique (from_func → to_func) anomalous edge that was never seen
        in normal runs → one ORDERING invariant.
        """
        seen_edges: set[str] = set()

        for score in phase2_result.scores:
            for pt in score.permissive_transitions:
                if not pt.is_new_edge:
                    continue
                edge_key = f"{pt.from_func}->{pt.to_func}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                # Find function-pointer variable if one exists for from_func
                fp_vars = [
                    vid for vid, v in variables.items()
                    if v.func_name == pt.from_func
                    and v.kind == StateVariableKind.FUNCTION_POINTER
                ]

                desc = (
                    f"Control must not flow from {pt.from_func} to "
                    f"{pt.to_func or 'anomalous target'} in normal execution "
                    f"(new edge, deviation={pt.deviation_score:.2f})"
                )

                self._add_invariant_if_new(
                    inv_set=inv_set,
                    target_id=target_id,
                    description=desc,
                    strength=InvariantStrength.SHOULD,
                    kind=StateVariableKind.FUNCTION_POINTER,
                    variable_ids=fp_vars,
                    guarded_block_ids=[],
                    violation_unlocks_block_ids=[],
                    confidence=pt.deviation_score,
                )

    # ── Deduplication helper ──────────────────────────────────────────────────

    def _add_invariant_if_new(
        self,
        inv_set: InvariantSet,
        target_id: str,
        description: str,
        strength: InvariantStrength,
        kind: StateVariableKind,
        variable_ids: list[str],
        guarded_block_ids: list[str],
        violation_unlocks_block_ids: list[str],
        derived_from_fragment_ids: list[str] | None = None,
        derived_from_fault_class: str = "",
        confidence: float = 0.5,
    ) -> Invariant | None:
        """
        Add an invariant to the set only if an equivalent one doesn't already exist.

        Equality key: (kind.value, _block_set_key(violation_unlocks_block_ids)).
        Returns the (existing or new) Invariant, or None if skipped.
        """
        dedup_key = f"{kind.value}::{_block_set_key(violation_unlocks_block_ids)}"

        # Check existing by scanning descriptions (lightweight dedup)
        for existing in inv_set.invariants.values():
            existing_key = f"{existing.kind.value}::{_block_set_key(existing.violation_unlocks_block_ids)}"
            if existing_key == dedup_key:
                # Merge evidence
                for vid in variable_ids:
                    if vid not in existing.variable_ids:
                        existing.variable_ids.append(vid)
                for fid in (derived_from_fragment_ids or []):
                    if fid not in existing.derived_from_fragment_ids:
                        existing.derived_from_fragment_ids.append(fid)
                # Strengthen confidence
                existing.confidence = max(existing.confidence, confidence)
                return existing

        inv = Invariant(
            target_id=target_id,
            description=description,
            strength=strength,
            kind=kind,
            variable_ids=variable_ids,
            guarded_block_ids=guarded_block_ids,
            violation_unlocks_block_ids=violation_unlocks_block_ids,
            derived_from_fault_class=derived_from_fault_class,
            derived_from_fragment_ids=derived_from_fragment_ids or [],
            confidence=confidence,
        )
        inv_set.add_invariant(inv)
        return inv
