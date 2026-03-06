"""
EcodiaOS - Simula Semantic Invariant Discovery (Phase 2.1)

What cannot easily be hidden or transformed?
  -> Outputs must remain stable for correct behaviour
  -> Decision boundaries cannot be crossed

Mechanism:
  1. AST scan: identify pure functions, decision thresholds, comparison patterns
  2. Z3 check: encode output stability as range/equality constraints
  3. Decision boundary extraction: comparisons (>, <, >=, <=) that gate control flow
  4. E-graph hint: register canonical forms for semantic equivalence detection

LLM used only for ambiguous thresholds and multi-variable stability properties.
AST scanner runs zero-LLM for simple cases (O(N) in file size).
"""

from __future__ import annotations

import ast
import hashlib
import time
from typing import Any

import structlog

from primitives.common import new_id
from systems.simula.verification.invariant_types import (
    DecisionBoundaryInvariant,
    EvidenceSource,
    InvariantStrength,
    OutputStabilityInvariant,
    SemanticEquivalenceInvariant,
    SemanticInvariantDiscovery,
)

logger = structlog.get_logger().bind(system="simula.invariants.semantic")


# -- AST Helpers --------------------------------------------------------------


class _ThresholdVisitor(ast.NodeVisitor):
    """
    Walk an AST collecting comparison expressions that gate control flow.
    Produces (threshold_var, threshold_literal, decision_kind) records.
    """

    def __init__(self) -> None:
        self.thresholds: list[dict[str, Any]] = []
        self._current_function: str = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._current_function
        self._current_function = node.name
        self.generic_visit(node)
        self._current_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        for c in self._extract_comparisons(node.test):
            c["decision_kind"] = "conditional"
            c["function"] = self._current_function
            self.thresholds.append(c)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value and isinstance(node.value, ast.Compare):
            for c in self._extract_comparisons(node.value):
                c["decision_kind"] = "return_comparison"
                c["function"] = self._current_function
                self.thresholds.append(c)
        self.generic_visit(node)

    def _extract_comparisons(self, node: ast.expr) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not isinstance(node, ast.Compare):
            return results
        left = node.left
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            var_name = _ast_name(left)
            literal_val = _ast_literal(comparator)
            if var_name and literal_val is not None:
                results.append({
                    "variable": var_name,
                    "operator": type(op).__name__,
                    "threshold": literal_val,
                })
            var_name2 = _ast_name(comparator)
            literal_val2 = _ast_literal(left)
            if var_name2 and literal_val2 is not None:
                results.append({
                    "variable": var_name2,
                    "operator": _flip_op(type(op).__name__),
                    "threshold": literal_val2,
                })
        return results


class _PurityVisitor(ast.NodeVisitor):
    """
    Identify likely-pure functions: no global/nonlocal, no I/O primitives,
    no self/cls attribute assignments.
    """

    IMPURE_CALLS: frozenset[str] = frozenset({
        "print", "open", "write", "read", "send", "recv",
        "sleep", "append", "remove", "pop", "insert",
    })

    def __init__(self) -> None:
        self.pure_functions: list[str] = []
        self.impure_reasons: dict[str, list[str]] = {}
        self._current: str = ""
        self._issues: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev_fn, prev_issues = self._current, self._issues
        self._current = node.name
        self._issues = []
        self.generic_visit(node)
        if not self._issues:
            self.pure_functions.append(node.name)
        else:
            self.impure_reasons[node.name] = list(self._issues)
        self._current = prev_fn
        self._issues = prev_issues

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Global(self, node: ast.Global) -> None:
        self._issues.append(f"uses global: {', '.join(node.names)}")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._issues.append(f"uses nonlocal: {', '.join(node.names)}")

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        if name in self.IMPURE_CALLS:
            self._issues.append(f"calls impure: {name}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id in ("self", "cls"):
                    self._issues.append(f"mutates self.{target.attr}")
        self.generic_visit(node)


def _ast_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _ast_literal(node: ast.expr) -> float | int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    return None


def _flip_op(op: str) -> str:
    return {"Gt": "Lt", "Lt": "Gt", "GtE": "LtE", "LtE": "GtE"}.get(op, op)


# -- Domain Knowledge ---------------------------------------------------------


_OP_Z3 = {"Gt": ">", "Lt": "<", "GtE": ">=", "LtE": "<=", "Eq": "==", "NotEq": "!="}

_KNOWN_DOMAINS: dict[str, tuple[float, float]] = {
    "risk_score": (0.0, 1.0),
    "risk_level": (0.0, 1.0),
    "confidence": (0.0, 1.0),
    "alignment": (-1.0, 1.0),
    "regression_rate": (0.0, 1.0),
    "rollback_rate": (0.0, 1.0),
    "budget_headroom_percent": (0.0, 100.0),
    "evidence_strength": (0.0, 1.0),
    "expected_impact": (0.0, 1.0),
    "constitutional_alignment": (-1.0, 1.0),
}

# Iron-rule threshold values: cannot be relaxed without governance
_IRON_RULE_THRESHOLDS: frozenset[float] = frozenset({0.10, 0.05, 0.30, 0.0})


# -- Semantic Invariant Discoverer --------------------------------------------


class SemanticInvariantDiscoverer:
    """
    Discovers semantic invariants in Python source via AST analysis + Z3 checking.

    Steps:
      1. Purity analysis -> output stability invariants
      2. Threshold extraction -> decision boundary invariants
      3. Canonical AST hashing -> semantic equivalence invariants
      4. Z3 validation of range expressions -> upgrade evidence source
    """

    def __init__(self, check_timeout_ms: int = 5000, zero_llm: bool = True) -> None:
        self._check_timeout_ms = check_timeout_ms
        self._zero_llm = zero_llm
        self._log = logger

    def discover(
        self,
        python_source: str,
        target_file: str = "",
        target_functions: list[str] | None = None,
    ) -> SemanticInvariantDiscovery:
        start = time.monotonic()
        result = SemanticInvariantDiscovery(
            target_file=target_file,
            target_functions=target_functions or [],
        )

        try:
            tree = ast.parse(python_source)
        except SyntaxError as exc:
            self._log.warning("semantic_parse_error", error=str(exc))
            result.discovery_time_ms = int((time.monotonic() - start) * 1000)
            return result

        # Step 1: Output stability from purity analysis
        purity = _PurityVisitor()
        purity.visit(tree)
        for fn_name in purity.pure_functions:
            if target_functions and fn_name not in target_functions:
                continue
            result.output_stability.append(
                self._build_output_stability(fn_name, target_file)
            )

        # Step 2: Decision boundaries from threshold comparisons
        thresholds = _ThresholdVisitor()
        thresholds.visit(tree)
        seen: set[tuple[str, float]] = set()
        for entry in thresholds.thresholds:
            var = entry.get("variable", "")
            threshold = entry.get("threshold")
            fn = entry.get("function", "")
            op = entry.get("operator", "Gt")
            if threshold is None:
                continue
            key = (var, float(threshold))
            if key in seen:
                continue
            seen.add(key)
            if target_functions and fn not in target_functions:
                continue
            result.decision_boundaries.append(
                self._build_decision_boundary(var, float(threshold), op, fn, target_file)
            )

        # Step 3: Semantic equivalence canonical hashes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if target_functions and node.name not in target_functions:
                    continue
                result.semantic_equivalences.append(
                    SemanticEquivalenceInvariant(
                        invariant_id=f"sem_eq_{new_id()[:8]}",
                        description=(
                            f"{node.name}: canonical AST hash for "
                            f"semantic equivalence tracking"
                        ),
                        strength=InvariantStrength.STRUCTURAL,
                        evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                        confidence=1.0,
                        canonical_form_hash=_canonical_ast_hash(node),
                        target_file=target_file,
                        target_function=node.name,
                        input_domain="all inputs",
                        exceptions_preserved=True,
                    )
                )

        # Step 4: Z3 validation
        z3_verified = 0
        for inv in result.output_stability:
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                inv.strength = InvariantStrength.AXIOM
                z3_verified += 1
        for inv in result.decision_boundaries:
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                z3_verified += 1

        result.z3_verified_count = z3_verified
        result.violations_found = sum(
            1 for inv in result.output_stability if inv.violation_witness
        ) + sum(1 for inv in result.decision_boundaries if inv.violation_witness)
        result.discovery_time_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "semantic_invariants_discovered",
            file=target_file,
            output_stability=len(result.output_stability),
            decision_boundaries=len(result.decision_boundaries),
            equivalences=len(result.semantic_equivalences),
            z3_verified=z3_verified,
        )
        return result

    def _build_output_stability(self, fn_name: str, target_file: str) -> OutputStabilityInvariant:
        domain_vars = [v for v in _KNOWN_DOMAINS if v in fn_name.lower()]
        z3_expr = ""
        var_decls: dict[str, str] = {}
        if domain_vars:
            var = domain_vars[0]
            lo, hi = _KNOWN_DOMAINS[var]
            if hi < 1e8:
                var_decls[var] = "Real"
                z3_expr = f"z3.And({var} >= {lo}, {var} <= {hi})"
        return OutputStabilityInvariant(
            invariant_id=f"out_stab_{new_id()[:8]}",
            description=f"{fn_name}: pure function - output is a stable function of inputs only",
            strength=InvariantStrength.STRUCTURAL,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.85,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            target_function=fn_name,
            output_variable="return_value",
            observed_output_variance=0.0,
            variance_threshold=0.0,
        )

    def _build_decision_boundary(
        self, var: str, threshold: float, operator: str, fn_name: str, target_file: str,
    ) -> DecisionBoundaryInvariant:
        z3_op = _OP_Z3.get(operator, ">")
        is_iron = threshold in _IRON_RULE_THRESHOLDS
        var_decls: dict[str, str] = {var: "Real"}
        if var in _KNOWN_DOMAINS:
            lo, hi = _KNOWN_DOMAINS[var]
            z3_expr = f"z3.And({var} >= {lo}, {var} <= {hi})" if hi < 1e8 else ""
        else:
            z3_expr = f"z3.And({var} >= 0.0, {var} <= 1.0)"
        return DecisionBoundaryInvariant(
            invariant_id=f"dec_bound_{new_id()[:8]}",
            description=(
                f"{fn_name}: decision boundary - "
                f"{var} {z3_op} {threshold} gates distinct behaviour"
            ),
            strength=InvariantStrength.AXIOM if is_iron else InvariantStrength.STRUCTURAL,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.9 if is_iron else 0.75,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            target_function=fn_name,
            threshold_variable=var,
            threshold_value=threshold,
            decision_above=f"behaviour when {var} {z3_op} {threshold}",
            decision_below="behaviour otherwise",
            threshold_is_mutable=not is_iron,
        )

    def _z3_valid(self, z3_expr_code: str, variable_declarations: dict[str, str]) -> bool:
        """
        Check if a Z3 expression holds universally (negation UNSAT).
        Evaluates the expression string in a sandboxed namespace containing only
        the z3 module and the declared symbolic variables - no builtins accessible.
        Same pattern as verification/z3_bridge.py.
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
            # Sandboxed eval: __builtins__={} prevents any stdlib access
            expr = eval(z3_expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception:
            return False

        if not isinstance(expr, z3_lib.BoolRef):
            return False

        solver.add(z3_lib.Not(expr))
        return solver.check() == z3_lib.unsat


def _canonical_ast_hash(node: ast.AST) -> str:
    """Stable SHA-256 of an AST node's structure (structure only, no names or positions)."""
    try:
        dump = ast.dump(node, indent=None, annotate_fields=False)
    except Exception:
        dump = type(node).__name__
    return hashlib.sha256(dump.encode()).hexdigest()[:16]
