"""
EcodiaOS - Simula State Invariant Discovery (Phase 2.2)

What cannot easily be hidden or transformed in state space?
  -> Counters that only increment cannot be made to go backwards
  -> State machines that must follow valid transitions cannot be bypassed
  -> Caches that must be coherent cannot silently diverge

Mechanism:
  1. AST scan: identify counter increments (AugAssign +=), FSM transitions
  2. Z3 check: encode monotonicity, transition validity, coherence constraints
  3. Relational integrity: FK-style linkage between records
  4. Domain knowledge: known EcodiaOS state machines, counters, caches

Zero-LLM for structural detection. LLM optional for complex invariants.
"""

from __future__ import annotations

import ast
import time
from typing import Any

import structlog

from primitives.common import new_id
from systems.simula.verification.invariant_types import (
    CacheCoherenceInvariant,
    CounterMonotonicityInvariant,
    EvidenceSource,
    InvariantStrength,
    RelationalIntegrityInvariant,
    SessionConsistencyInvariant,
    StateInvariantDiscovery,
)

logger = structlog.get_logger().bind(system="simula.invariants.state")


# -- Domain Knowledge ---------------------------------------------------------

_KNOWN_COUNTERS: dict[str, dict[str, Any]] = {
    "config_version": {
        "monotone": "strict",
        "strength": InvariantStrength.AXIOM,
        "domain": (1, 2**31),
        "description": "Config version strictly increases on every accepted proposal",
    },
    "chain_position": {
        "monotone": "strict",
        "strength": InvariantStrength.AXIOM,
        "domain": (0, 2**63),
        "description": "Hash chain position strictly increases; chain is append-only",
    },
    "proposal_count": {
        "monotone": "non_strict",
        "strength": InvariantStrength.STRUCTURAL,
        "domain": (0, 2**31),
        "description": "Total proposal count never decreases",
    },
    "retry_count": {
        "monotone": "non_strict",
        "strength": InvariantStrength.STRUCTURAL,
        "domain": (0, 100),
        "description": "Retry counter never decreases within a session",
    },
    "violation_count": {
        "monotone": "non_strict",
        "strength": InvariantStrength.STRUCTURAL,
        "domain": (0, 2**31),
        "description": "Violation counter is non-decreasing",
    },
    "repair_cycle": {
        "monotone": "strict",
        "strength": InvariantStrength.STRUCTURAL,
        "domain": (0, 2**16),
        "description": "Repair cycle index strictly increases",
    },
    "epoch": {
        "monotone": "non_strict",
        "strength": InvariantStrength.STRUCTURAL,
        "domain": (0, 2**63),
        "description": "Epoch counter non-decreasing",
    },
    "sequence_number": {
        "monotone": "strict",
        "strength": InvariantStrength.STRUCTURAL,
        "domain": (0, 2**63),
        "description": "Sequence number strictly increases",
    },
}

_KNOWN_STATE_MACHINES: dict[str, dict[str, Any]] = {
    "proposal_lifecycle": {
        "states": [
            "initial", "deduplicated", "validated", "simulated",
            "gated", "applied", "verified", "recorded",
            "rejected", "rolled_back",
        ],
        "terminal": ["recorded", "rejected", "rolled_back"],
        "valid_forward": {
            "initial": ["deduplicated"],
            "deduplicated": ["validated", "rejected"],
            "validated": ["simulated", "rejected"],
            "simulated": ["gated", "rejected"],
            "gated": ["applied"],
            "applied": ["verified", "rolled_back"],
            "verified": ["recorded", "rolled_back"],
            "rolled_back": ["recorded"],
        },
        "strength": InvariantStrength.AXIOM,
        "description": "Simula proposal must follow the 7-stage pipeline",
    },
    "simula_pipeline_stage": {
        "states": [
            "DEDUPLICATE", "VALIDATE", "SIMULATE", "GATE",
            "APPLY", "VERIFY", "RECORD",
        ],
        "terminal": ["RECORD"],
        "valid_forward": {
            "DEDUPLICATE": ["VALIDATE"],
            "VALIDATE": ["SIMULATE"],
            "SIMULATE": ["GATE"],
            "GATE": ["APPLY"],
            "APPLY": ["VERIFY"],
            "VERIFY": ["RECORD"],
        },
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Pipeline stage must progress forward only",
    },
    "repair_phase": {
        "states": [
            "detecting", "diagnosing", "planning",
            "applying", "verifying", "complete", "failed",
        ],
        "terminal": ["complete", "failed"],
        "valid_forward": {
            "detecting": ["diagnosing"],
            "diagnosing": ["planning"],
            "planning": ["applying"],
            "applying": ["verifying"],
            "verifying": ["complete", "failed"],
        },
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Repair phase follows defined causal sequence",
    },
}

_KNOWN_CACHES: dict[str, dict[str, Any]] = {
    "incremental_verification_hot": {
        "ttl_seconds": 3600,
        "coherence": "write_through",
        "description": "Incremental verification cache — coherent with on-disk state",
    },
    "simulation_result_cache": {
        "ttl_seconds": 300,
        "coherence": "read_through",
        "description": "Simulation result cache for identical proposals",
    },
    "analytics_cache": {
        "ttl_seconds": 300,
        "coherence": "eventual",
        "description": "Analytics cache — eventually consistent with DB",
    },
    "constitution_cache": {
        "ttl_seconds": 86400,
        "coherence": "write_through",
        "description": "Constitution invariant cache — must be coherent",
    },
}

_KNOWN_RELATIONS: dict[str, dict[str, Any]] = {
    "hash_chain_linkage": {
        "parent_key": "previous_hash",
        "child_key": "current_hash",
        "strength": InvariantStrength.AXIOM,
        "integrity": "cryptographic",
        "description": "Each evolution record's previous_hash must match prior record's hash",
    },
    "proposal_simulation_result": {
        "parent_key": "proposal_id",
        "child_key": "simulation_id",
        "strength": InvariantStrength.STRUCTURAL,
        "integrity": "referential",
        "description": "Every simulation result references exactly one proposal",
    },
    "rollback_snapshot": {
        "parent_key": "proposal_id",
        "child_key": "snapshot_id",
        "strength": InvariantStrength.STRUCTURAL,
        "integrity": "referential",
        "description": "Snapshots are linked to their originating proposal",
    },
}


# -- AST Helpers --------------------------------------------------------------


class _CounterVisitor(ast.NodeVisitor):
    """
    Detect counter-increment patterns in Python AST.

    Recognises:
      - x += 1  (AugAssign with Add and Constant(1))
      - x = x + 1
      - x += n  (AugAssign with Add and any constant)
    """

    def __init__(self) -> None:
        self.increments: list[dict[str, Any]] = []
        self._current_function: str = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._current_function
        self._current_function = node.name
        self.generic_visit(node)
        self._current_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, ast.Add):
            var_name = _aug_target_name(node.target)
            amount = _aug_amount(node.value)
            if var_name and amount is not None and amount > 0:
                self.increments.append({
                    "variable": var_name,
                    "amount": amount,
                    "function": self._current_function,
                    "lineno": node.lineno,
                })
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Detect x = x + 1 pattern
        if (
            len(node.targets) == 1
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Add)
        ):
            target_name = _aug_target_name(node.targets[0])
            left_name = _aug_target_name(node.value.left)
            amount = _aug_amount(node.value.right)
            if (
                target_name
                and left_name
                and target_name == left_name
                and amount is not None
                and amount > 0
            ):
                self.increments.append({
                    "variable": target_name,
                    "amount": amount,
                    "function": self._current_function,
                    "lineno": node.lineno,
                })
        self.generic_visit(node)


def _aug_target_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _aug_amount(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


# -- State Invariant Discoverer -----------------------------------------------


class StateInvariantDiscoverer:
    """
    Discovers state invariants in Python source and system domain knowledge.

    Steps:
      1. Counter monotonicity from AST increment detection
      2. Session consistency from known state machines
      3. Cache coherence from domain knowledge
      4. Relational integrity from known FK-style linkages
      5. Z3 validation of all generated expressions
    """

    def __init__(self, check_timeout_ms: int = 5000) -> None:
        self._check_timeout_ms = check_timeout_ms
        self._log = logger

    def discover(
        self,
        python_source: str,
        target_file: str = "",
        target_functions: list[str] | None = None,
        include_domain_knowledge: bool = True,
    ) -> StateInvariantDiscovery:
        start = time.monotonic()
        result = StateInvariantDiscovery(
            target_file=target_file,
            target_functions=target_functions or [],
        )

        # Step 1: Counter monotonicity from AST
        try:
            tree = ast.parse(python_source)
            counter_vis = _CounterVisitor()
            counter_vis.visit(tree)
            seen_counters: set[str] = set()
            for entry in counter_vis.increments:
                var = entry["variable"]
                fn = entry.get("function", "")
                if target_functions and fn not in target_functions:
                    continue
                # Match against known counter table (substring match)
                matched_key = next(
                    (k for k in _KNOWN_COUNTERS if k in var or var in k), None
                )
                if matched_key and matched_key not in seen_counters:
                    seen_counters.add(matched_key)
                    result.counter_monotonicity.append(
                        self._build_counter_invariant(matched_key, fn, target_file)
                    )
                elif var not in seen_counters and var not in _KNOWN_COUNTERS:
                    # Unknown counter — structural, non-strict
                    seen_counters.add(var)
                    result.counter_monotonicity.append(
                        self._build_generic_counter_invariant(var, fn, target_file)
                    )
        except SyntaxError as exc:
            self._log.warning("state_parse_error", error=str(exc))

        # Step 2: Domain knowledge — known counters not found in AST
        if include_domain_knowledge:
            for counter_name, _meta in _KNOWN_COUNTERS.items():
                already = any(
                    inv.counter_variable == counter_name
                    for inv in result.counter_monotonicity
                )
                if not already:
                    result.counter_monotonicity.append(
                        self._build_counter_invariant(counter_name, "", target_file)
                    )

        # Step 3: Session consistency from known state machines
        if include_domain_knowledge:
            for sm_name, sm_meta in _KNOWN_STATE_MACHINES.items():
                result.session_consistency.append(
                    self._build_session_invariant(sm_name, sm_meta, target_file)
                )

        # Step 4: Cache coherence
        if include_domain_knowledge:
            for cache_name, cache_meta in _KNOWN_CACHES.items():
                result.cache_coherence.append(
                    self._build_cache_invariant(cache_name, cache_meta, target_file)
                )

        # Step 5: Relational integrity
        if include_domain_knowledge:
            for rel_name, rel_meta in _KNOWN_RELATIONS.items():
                result.relational_integrity.append(
                    self._build_relation_invariant(rel_name, rel_meta, target_file)
                )

        # Step 6: Z3 validation
        z3_verified = 0
        for inv in result.counter_monotonicity:
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                inv.strength = InvariantStrength.AXIOM
                z3_verified += 1
        for inv in result.session_consistency:
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                z3_verified += 1

        result.z3_verified_count = z3_verified
        result.discovery_time_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "state_invariants_discovered",
            file=target_file,
            counter_monotonicity=len(result.counter_monotonicity),
            session_consistency=len(result.session_consistency),
            cache_coherence=len(result.cache_coherence),
            relational_integrity=len(result.relational_integrity),
            z3_verified=z3_verified,
        )
        return result

    # -- Builders -------------------------------------------------------------

    def _build_counter_invariant(
        self, counter_name: str, fn_name: str, target_file: str,
    ) -> CounterMonotonicityInvariant:
        meta = _KNOWN_COUNTERS[counter_name]
        lo, hi = meta["domain"]
        monotone = meta["monotone"]
        is_strict = monotone == "strict"
        var_decls = {counter_name: "Int", f"{counter_name}_next": "Int"}
        op = ">" if is_strict else ">="
        z3_expr = (
            f"z3.And("
            f"{counter_name} >= {lo}, "
            f"{counter_name} <= {hi}, "
            f"{counter_name}_next {op} {counter_name}"
            f")"
        )
        return CounterMonotonicityInvariant(
            invariant_id=f"counter_{new_id()[:8]}",
            description=meta["description"],
            strength=meta["strength"],
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.95 if is_strict else 0.85,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            target_function=fn_name,
            counter_variable=counter_name,
            is_strictly_monotone=is_strict,
            lower_bound=float(lo),
            upper_bound=float(hi),
        )

    def _build_generic_counter_invariant(
        self, var: str, fn_name: str, target_file: str,
    ) -> CounterMonotonicityInvariant:
        safe_var = var.replace(".", "_")
        var_decls = {safe_var: "Int", f"{safe_var}_next": "Int"}
        z3_expr = (
            f"z3.And("
            f"{safe_var} >= 0, "
            f"{safe_var}_next >= {safe_var}"
            f")"
        )
        return CounterMonotonicityInvariant(
            invariant_id=f"counter_{new_id()[:8]}",
            description=f"{var}: inferred counter — non-decreasing (AST += detected)",
            strength=InvariantStrength.STRUCTURAL,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.65,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            target_function=fn_name,
            counter_variable=var,
            is_strictly_monotone=False,
            lower_bound=0.0,
        )

    def _build_session_invariant(
        self, sm_name: str, sm_meta: dict[str, Any], target_file: str,
    ) -> SessionConsistencyInvariant:
        states = sm_meta["states"]
        terminal = sm_meta["terminal"]
        # Z3 encode: state_var is one of the valid integer indices
        n = len(states)
        var_decls = {"state_idx": "Int", "state_idx_next": "Int"}
        z3_expr = (
            f"z3.And("
            f"state_idx >= 0, state_idx < {n}, "
            f"state_idx_next >= 0, state_idx_next < {n}"
            f")"
        )
        return SessionConsistencyInvariant(
            invariant_id=f"session_{new_id()[:8]}",
            description=sm_meta["description"],
            strength=sm_meta["strength"],
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.9,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            session_type=sm_name,
            valid_states=states,
            terminal_states=terminal,
            transition_table=sm_meta.get("valid_forward", {}),
            allows_backward_transition=False,
        )

    def _build_cache_invariant(
        self, cache_name: str, cache_meta: dict[str, Any], target_file: str,
    ) -> CacheCoherenceInvariant:
        return CacheCoherenceInvariant(
            invariant_id=f"cache_{new_id()[:8]}",
            description=cache_meta["description"],
            strength=InvariantStrength.STRUCTURAL,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.8,
            target_file=target_file,
            cache_name=cache_name,
            coherence_protocol=cache_meta["coherence"],
            max_staleness_ms=cache_meta["ttl_seconds"] * 1000,
            invalidation_events=[f"{cache_name}_write", f"{cache_name}_invalidate"],
        )

    def _build_relation_invariant(
        self, rel_name: str, rel_meta: dict[str, Any], target_file: str,
    ) -> RelationalIntegrityInvariant:
        return RelationalIntegrityInvariant(
            invariant_id=f"relation_{new_id()[:8]}",
            description=rel_meta["description"],
            strength=rel_meta["strength"],
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.9 if rel_meta["strength"] == InvariantStrength.AXIOM else 0.75,
            target_file=target_file,
            relation_name=rel_name,
            parent_key=rel_meta["parent_key"],
            child_key=rel_meta["child_key"],
            integrity_kind=rel_meta["integrity"],
            cascade_on_delete=False,
        )

    # -- Z3 Validation --------------------------------------------------------

    def _z3_valid(self, z3_expr_code: str, variable_declarations: dict[str, str]) -> bool:
        """
        Check if a Z3 expression holds universally (negation UNSAT).
        Sandboxed eval: only z3 and declared symbolic variables accessible.
        """
        try:
            import z3 as z3_lib
        except ImportError:
            return False

        solver = z3_lib.Solver()
        solver.set("timeout", self._check_timeout_ms)

        z3_vars: dict[str, Any] = {}
        for name, z3_type in variable_declarations.items():
            if z3_type == "Int":
                z3_vars[name] = z3_lib.Int(name)
            elif z3_type == "Bool":
                z3_vars[name] = z3_lib.Bool(name)
            else:
                z3_vars[name] = z3_lib.Real(name)

        namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
        try:
            expr = _sandboxed_eval(z3_expr_code, namespace)
        except Exception:
            return False

        if not isinstance(expr, z3_lib.BoolRef):
            return False

        solver.add(z3_lib.Not(expr))
        return solver.check() == z3_lib.unsat


_SENTINEL = object()


def _sandboxed_eval(expr_code: str, namespace: dict[str, Any]) -> Any:
    """Evaluate expr_code with only the provided namespace — no builtins."""
    return eval(expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
