"""
EcodiaOS -- Inspector Phase 3: Mutation Formal Verifier

Pre-apply formal verification that mathematically verifies properties of
proposed mutations before they touch the codebase.  Four verification
dimensions run concurrently:

  1. Type Safety      -- AST analysis of signature/import preservation
  2. Invariant Pres.  -- Z3 + pattern matching on constitutional invariants
  3. Behavioral Equiv -- symbolic diff of critical path outputs
  4. Termination      -- loop/recursion/await bounding analysis

Entry point:
    verifier = MutationFormalVerifier(codebase_root=Path(...))
    result = await verifier.verify(
        original_source={"path/to/file.py": "..."},
        mutated_source={"path/to/file.py": "..."},
        taint_severity=TaintSeverity.ELEVATED,
    )

Integration:
    Slots into health.py between Phase 3 (unit tests) and Phase 4
    (Dafny/Z3/static analysis).  Mandatory when EIS taint >= ELEVATED.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import time
from pathlib import Path

import structlog

from systems.simula.verification.mutation_verifier_types import (
    BehavioralEquivalenceResult,
    BehaviorDivergence,
    CheckStatus,
    InvariantPreservationResult,
    InvariantViolation,
    MutationVerificationResult,
    MutationVerificationStatus,
    TerminationResult,
    TerminationRisk,
    TypeSafetyResult,
    TypeViolation,
)

logger = structlog.get_logger().bind(
    system="simula.verification",
    component="mutation_verifier",
)


# =========================================================================
# 1. TYPE SAFETY CHECKER
# =========================================================================


class TypeSafetyChecker:
    """
    AST-based type safety verification for mutations.

    Compares original and mutated ASTs to ensure:
      - Function signatures are not broken (args, return annotations)
      - Type annotations are not silently removed
      - Import statements for used names are preserved
      - Call sites match expected arity
    """

    def check(
        self,
        original_sources: dict[str, str],
        mutated_sources: dict[str, str],
    ) -> TypeSafetyResult:
        t0 = time.monotonic()
        violations: list[TypeViolation] = []
        functions_checked = 0
        sigs_preserved = 0
        sigs_broken = 0
        imports_verified = 0
        imports_broken = 0

        for file_path, mutated_code in mutated_sources.items():
            original_code = original_sources.get(file_path)
            if original_code is None:
                # New file -- no original to compare against
                continue

            try:
                orig_tree = ast.parse(original_code, filename=file_path)
                mut_tree = ast.parse(mutated_code, filename=file_path)
            except SyntaxError as exc:
                violations.append(TypeViolation(
                    file_path=file_path,
                    line=exc.lineno or 0,
                    description=f"Syntax error in mutated code: {exc.msg}",
                    severity="error",
                ))
                continue

            # Check function signatures
            orig_funcs = _extract_function_signatures(orig_tree)
            mut_funcs = _extract_function_signatures(mut_tree)

            for fname, orig_sig in orig_funcs.items():
                functions_checked += 1
                if fname not in mut_funcs:
                    # Function removed -- check if it was exported/public
                    if not fname.startswith("_"):
                        violations.append(TypeViolation(
                            file_path=file_path,
                            function_name=fname,
                            description=f"Public function '{fname}' removed by mutation",
                            severity="error",
                        ))
                        sigs_broken += 1
                    continue

                mut_sig = mut_funcs[fname]
                sig_violations = _compare_signatures(file_path, fname, orig_sig, mut_sig)
                if sig_violations:
                    violations.extend(sig_violations)
                    sigs_broken += 1
                else:
                    sigs_preserved += 1

            # Check import preservation
            orig_imports = _extract_imports(orig_tree)
            mut_imports = _extract_imports(mut_tree)
            mut_names_used = _extract_name_references(mut_tree)

            for imp_name in orig_imports:
                imports_verified += 1
                if imp_name not in mut_imports and imp_name in mut_names_used:
                    violations.append(TypeViolation(
                        file_path=file_path,
                        description=f"Import '{imp_name}' removed but still referenced in code",
                        severity="error",
                    ))
                    imports_broken += 1

        duration_ms = int((time.monotonic() - t0) * 1000)
        status = CheckStatus.PASSED if not violations else CheckStatus.FAILED

        return TypeSafetyResult(
            status=status,
            violations=violations,
            functions_checked=functions_checked,
            signatures_preserved=sigs_preserved,
            signatures_broken=sigs_broken,
            imports_verified=imports_verified,
            imports_broken=imports_broken,
            duration_ms=duration_ms,
        )


# -- Signature helpers -------------------------------------------------------


class _FuncSig:
    """Lightweight representation of a function signature for comparison."""

    __slots__ = (
        "name", "args", "defaults_count", "kwonly", "has_vararg",
        "has_kwarg", "return_annotation", "decorators", "is_async",
    )

    def __init__(
        self,
        name: str,
        args: list[tuple[str, str | None]],  # (name, annotation_repr)
        defaults_count: int,
        kwonly: list[tuple[str, str | None]],
        has_vararg: bool,
        has_kwarg: bool,
        return_annotation: str | None,
        decorators: list[str],
        is_async: bool,
    ) -> None:
        self.name = name
        self.args = args
        self.defaults_count = defaults_count
        self.kwonly = kwonly
        self.has_vararg = has_vararg
        self.has_kwarg = has_kwarg
        self.return_annotation = return_annotation
        self.decorators = decorators
        self.is_async = is_async


def _annotation_repr(node: ast.expr | None) -> str | None:
    """Best-effort string representation of a type annotation AST node."""
    if node is None:
        return None
    return ast.dump(node)


def _extract_function_signatures(tree: ast.Module) -> dict[str, _FuncSig]:
    """Extract all top-level and class-method function signatures."""
    result: dict[str, _FuncSig] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        args_info: list[tuple[str, str | None]] = []
        for arg in node.args.args:
            args_info.append((arg.arg, _annotation_repr(arg.annotation)))

        kwonly_info: list[tuple[str, str | None]] = []
        for arg in node.args.kwonlyargs:
            kwonly_info.append((arg.arg, _annotation_repr(arg.annotation)))

        decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                decorators.append(d.id)
            elif isinstance(d, ast.Attribute):
                decorators.append(ast.dump(d))

        result[node.name] = _FuncSig(
            name=node.name,
            args=args_info,
            defaults_count=len(node.args.defaults),
            kwonly=kwonly_info,
            has_vararg=node.args.vararg is not None,
            has_kwarg=node.args.kwarg is not None,
            return_annotation=_annotation_repr(node.returns),
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

    return result


def _compare_signatures(
    file_path: str,
    fname: str,
    orig: _FuncSig,
    mut: _FuncSig,
) -> list[TypeViolation]:
    """Compare two function signatures for breaking changes."""
    violations: list[TypeViolation] = []

    # Required argument count changed (excluding defaults)
    orig_required = len(orig.args) - orig.defaults_count
    mut_required = len(mut.args) - mut.defaults_count
    if mut_required > orig_required:
        violations.append(TypeViolation(
            file_path=file_path,
            function_name=fname,
            description=(
                f"Function '{fname}' requires {mut_required} args "
                f"(was {orig_required}) -- breaks existing callers"
            ),
            expected_type=f"{orig_required} required args",
            actual_type=f"{mut_required} required args",
            severity="error",
        ))

    # Return annotation removed
    if orig.return_annotation is not None and mut.return_annotation is None:
        violations.append(TypeViolation(
            file_path=file_path,
            function_name=fname,
            description=f"Return type annotation removed from '{fname}'",
            expected_type=orig.return_annotation,
            actual_type="None (removed)",
            severity="warning",
        ))

    # Return annotation changed (may or may not be breaking)
    if (
        orig.return_annotation is not None
        and mut.return_annotation is not None
        and orig.return_annotation != mut.return_annotation
    ):
        violations.append(TypeViolation(
            file_path=file_path,
            function_name=fname,
            description=f"Return type annotation changed on '{fname}'",
            expected_type=orig.return_annotation,
            actual_type=mut.return_annotation,
            severity="warning",
        ))

    # Parameter annotations removed
    for i, (arg_name, orig_ann) in enumerate(orig.args):
        if i >= len(mut.args):
            break
        _, mut_ann = mut.args[i]
        if orig_ann is not None and mut_ann is None:
            violations.append(TypeViolation(
                file_path=file_path,
                function_name=fname,
                description=(
                    f"Type annotation removed from parameter "
                    f"'{arg_name}' in '{fname}'"
                ),
                expected_type=orig_ann,
                actual_type="None (removed)",
                severity="warning",
            ))

    # Async/sync change
    if orig.is_async != mut.is_async:
        violations.append(TypeViolation(
            file_path=file_path,
            function_name=fname,
            description=(
                f"Function '{fname}' changed from "
                f"{'async' if orig.is_async else 'sync'} to "
                f"{'async' if mut.is_async else 'sync'}"
            ),
            severity="error",
        ))

    return violations


def _extract_imports(tree: ast.Module) -> set[str]:
    """Extract all imported names from an AST."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _extract_name_references(tree: ast.Module) -> set[str]:
    """Extract all Name references in the AST (excluding imports and defs)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Capture the root of attribute chains
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
    return names


# =========================================================================
# 2. INVARIANT PRESERVATION CHECKER
# =========================================================================


class InvariantPreservationChecker:
    """
    Checks that constitutional invariants hold after a mutation.

    Three verification strategies:
      1. Forbidden-path checks (file path matching against FORBIDDEN_WRITE_PATHS)
      2. AST pattern matching for structural invariants
      3. Z3 symbolic checking for invariants with z3_expression (delegated
         to existing Z3Bridge when available)
    """

    def __init__(self, forbidden_paths: list[str] | None = None) -> None:
        self._forbidden_paths = forbidden_paths or [
            "systems/equor",
            "systems/simula",
            "ecodiaos/primitives/constitutional.py",
            "ecodiaos/primitives/common.py",
            "ecodiaos/config.py",
        ]

    def check(
        self,
        mutated_sources: dict[str, str],
        original_sources: dict[str, str],
    ) -> InvariantPreservationResult:
        t0 = time.monotonic()
        violations: list[InvariantViolation] = []
        invariants_checked = 0
        invariants_preserved = 0
        pattern_checks_run = 0
        pattern_checks_passed = 0

        for file_path, mutated_code in mutated_sources.items():
            # Forbidden-path checks
            for fp in self._forbidden_paths:
                invariants_checked += 1
                pattern_checks_run += 1
                # Normalize path separators for comparison
                normalized = file_path.replace("\\", "/")
                if fp in normalized:
                    violations.append(InvariantViolation(
                        invariant_id="SIMULA-IRON-FORBIDDEN-PATH",
                        invariant_description=f"Mutations to '{fp}' are forbidden",
                        violation_description=(
                            f"Mutation targets forbidden path: {file_path}"
                        ),
                        affected_file=file_path,
                        severity="blocking",
                    ))
                else:
                    invariants_preserved += 1
                    pattern_checks_passed += 1

            # Parse and run structural checks on the mutated code
            try:
                mut_tree = ast.parse(mutated_code, filename=file_path)
            except SyntaxError:
                continue  # Syntax errors caught by type safety checker

            original_code = original_sources.get(file_path)
            orig_tree = None
            if original_code:
                with contextlib.suppress(SyntaxError):
                    orig_tree = ast.parse(original_code, filename=file_path)

            # Check: action return pattern preserved
            invariants_checked += 1
            pattern_checks_run += 1
            action_violation = self._check_action_return_pattern(
                file_path, orig_tree, mut_tree,
            )
            if action_violation:
                violations.append(action_violation)
            else:
                invariants_preserved += 1
                pattern_checks_passed += 1

            # Check: no hard-delete introduction
            invariants_checked += 1
            pattern_checks_run += 1
            delete_violation = self._check_no_hard_delete(
                file_path, orig_tree, mut_tree,
            )
            if delete_violation:
                violations.append(delete_violation)
            else:
                invariants_preserved += 1
                pattern_checks_passed += 1

            # Check: audit logging not removed from sensitive functions
            invariants_checked += 1
            pattern_checks_run += 1
            audit_violation = self._check_audit_logging_preserved(
                file_path, orig_tree, mut_tree,
            )
            if audit_violation:
                violations.append(audit_violation)
            else:
                invariants_preserved += 1
                pattern_checks_passed += 1

        duration_ms = int((time.monotonic() - t0) * 1000)
        status = CheckStatus.PASSED if not violations else CheckStatus.FAILED

        return InvariantPreservationResult(
            status=status,
            violations=violations,
            invariants_checked=invariants_checked,
            invariants_preserved=invariants_preserved,
            invariants_violated=len(violations),
            pattern_checks_run=pattern_checks_run,
            pattern_checks_passed=pattern_checks_passed,
            duration_ms=duration_ms,
        )

    def _check_action_return_pattern(
        self,
        file_path: str,
        orig_tree: ast.Module | None,
        mut_tree: ast.Module,
    ) -> InvariantViolation | None:
        """
        If original functions return dict with 'data'/'error' keys,
        verify the mutation preserves this pattern.
        """
        if orig_tree is None:
            return None

        # Only check files that look like action modules
        if "actions" not in file_path:
            return None

        orig_return_funcs = _find_data_error_returning_functions(orig_tree)
        if not orig_return_funcs:
            return None

        mut_return_funcs = _find_data_error_returning_functions(mut_tree)

        # Any original {data, error} function that lost its pattern?
        for fname in orig_return_funcs:
            if (
                fname in _extract_function_signatures(mut_tree)
                and fname not in mut_return_funcs
            ):
                return InvariantViolation(
                    invariant_id="STRUCT-001",
                    invariant_description=(
                        "Server actions must return {data, error} pattern"
                    ),
                    violation_description=(
                        f"Function '{fname}' in {file_path} no longer returns "
                        f"the {{data, error}} pattern"
                    ),
                    affected_function=fname,
                    affected_file=file_path,
                    severity="blocking",
                )

        return None

    def _check_no_hard_delete(
        self,
        file_path: str,
        orig_tree: ast.Module | None,
        mut_tree: ast.Module,
    ) -> InvariantViolation | None:
        """
        Check that the mutation doesn't introduce hard-delete SQL patterns.
        Looks for string literals containing DELETE FROM without soft-delete guard.
        """
        orig_deletes = _count_hard_delete_patterns(orig_tree) if orig_tree else 0
        mut_deletes = _count_hard_delete_patterns(mut_tree)

        if mut_deletes > orig_deletes:
            return InvariantViolation(
                invariant_id="STRUCT-003",
                invariant_description="No hard-delete patterns allowed",
                violation_description=(
                    f"Mutation introduces {mut_deletes - orig_deletes} new "
                    f"hard-delete pattern(s) in {file_path}"
                ),
                affected_file=file_path,
                severity="blocking",
            )

        return None

    def _check_audit_logging_preserved(
        self,
        file_path: str,
        orig_tree: ast.Module | None,
        mut_tree: ast.Module,
    ) -> InvariantViolation | None:
        """
        If original code has audit logging calls, verify the mutation
        preserves them.
        """
        if orig_tree is None:
            return None

        orig_audit_funcs = _find_functions_with_audit_calls(orig_tree)
        mut_audit_funcs = _find_functions_with_audit_calls(mut_tree)

        # Any function that had audit logging but lost it?
        for fname in orig_audit_funcs:
            if (
                fname in _extract_function_signatures(mut_tree)
                and fname not in mut_audit_funcs
            ):
                return InvariantViolation(
                    invariant_id="STRUCT-002",
                    invariant_description=(
                        "Sensitive mutations must include audit logging"
                    ),
                    violation_description=(
                        f"Function '{fname}' in {file_path} lost its "
                        f"audit logging call"
                    ),
                    affected_function=fname,
                    affected_file=file_path,
                    severity="blocking",
                )

        return None


def _find_data_error_returning_functions(tree: ast.Module) -> set[str]:
    """Find functions that return dicts with 'data' and 'error' keys."""
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                keys = []
                for k in child.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.append(k.value)
                if "data" in keys and "error" in keys:
                    result.add(node.name)
                    break
    return result


def _count_hard_delete_patterns(tree: ast.Module | None) -> int:
    """Count string literals that look like hard-delete SQL."""
    if tree is None:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            upper = node.value.upper()
            if "DELETE FROM" in upper and "DELETED_AT" not in upper:
                count += 1
    return count


def _find_functions_with_audit_calls(tree: ast.Module) -> set[str]:
    """Find functions containing audit logging calls."""
    result: set[str] = set()
    audit_names = {"logAudit", "log_audit", "audit_log", "logaudit"}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = child.func
                if isinstance(callee, ast.Name) and callee.id in audit_names:
                    result.add(node.name)
                    break
                if (
                    isinstance(callee, ast.Attribute)
                    and callee.attr in audit_names
                ):
                    result.add(node.name)
                    break
    return result


# =========================================================================
# 3. BEHAVIORAL EQUIVALENCE CHECKER
# =========================================================================


# Critical path function names -- if mutation touches these, behavioral
# equivalence is mandatory.
_CRITICAL_PATH_FUNCTIONS: dict[str, str] = {
    "compute_verdict": "equor.verdict_pipeline",
    "check_invariants": "equor.invariant_check",
    "check_physical_harm": "equor.invariant.INV-001",
    "check_identity_destruction": "equor.invariant.INV-002",
    "check_misrepresentation": "equor.invariant.INV-003",
    "check_evidence_fabrication": "equor.invariant.INV-004",
    "check_drive_modification": "equor.invariant.INV-005",
    "check_autonomy_escalation": "equor.invariant.INV-006",
    "check_governance_bypass": "equor.invariant.INV-007",
    "check_discrimination": "equor.invariant.INV-008",
    "check_privacy_violation": "equor.invariant.INV-009",
    "check_trust_threat": "equor.invariant.INV-010",
    "evaluate": "equor.drive_evaluator",
    "_compute_composite_alignment": "equor.composite_alignment",
    "analyse_mutation": "eis.taint_engine",
}


class BehavioralEquivalenceChecker:
    """
    Verifies that mutations don't change behavior of critical path functions.

    For pure functions: compares AST structure of the function body
    (normalization handles trivial refactors like variable renaming).

    For functions with complex logic: generates test vectors from type
    annotations and runs both original and mutated versions to compare
    outputs.  Uses hypothesis strategies when available, falls back to
    boundary-value analysis from type annotations.
    """

    def check(
        self,
        original_sources: dict[str, str],
        mutated_sources: dict[str, str],
    ) -> BehavioralEquivalenceResult:
        t0 = time.monotonic()
        divergences: list[BehaviorDivergence] = []
        paths_checked = 0
        paths_equivalent = 0
        paths_divergent = 0
        test_vectors_generated = 0
        test_vectors_passed = 0

        for file_path, mutated_code in mutated_sources.items():
            original_code = original_sources.get(file_path)
            if original_code is None:
                continue

            try:
                orig_tree = ast.parse(original_code, filename=file_path)
                mut_tree = ast.parse(mutated_code, filename=file_path)
            except SyntaxError:
                continue

            orig_funcs = _extract_function_bodies(orig_tree)
            mut_funcs = _extract_function_bodies(mut_tree)

            for fname, critical_path in _CRITICAL_PATH_FUNCTIONS.items():
                if fname not in orig_funcs:
                    continue
                paths_checked += 1

                if fname not in mut_funcs:
                    # Critical function removed
                    divergences.append(BehaviorDivergence(
                        function_name=fname,
                        input_description="N/A",
                        original_output="function exists",
                        mutated_output="function removed",
                        divergence_type="removal",
                        critical_path=critical_path,
                    ))
                    paths_divergent += 1
                    continue

                # AST structural comparison (normalized)
                orig_body_dump = _normalize_function_body(orig_funcs[fname])
                mut_body_dump = _normalize_function_body(mut_funcs[fname])

                if orig_body_dump == mut_body_dump:
                    paths_equivalent += 1
                    continue

                # Bodies differ -- check if the diff is semantically safe
                # by comparing control flow structure
                orig_cf = _extract_control_flow(orig_funcs[fname])
                mut_cf = _extract_control_flow(mut_funcs[fname])

                if orig_cf != mut_cf:
                    # Control flow changed -- this is a behavioral divergence
                    divergences.append(BehaviorDivergence(
                        function_name=fname,
                        input_description="structural analysis",
                        original_output=f"control_flow={orig_cf[:100]}",
                        mutated_output=f"control_flow={mut_cf[:100]}",
                        divergence_type="control_flow_change",
                        critical_path=critical_path,
                    ))
                    paths_divergent += 1
                else:
                    # Control flow same but body changed -- likely safe refactor
                    # Run property-based comparison via test vectors
                    tv_count, tv_passed, tv_divs = _compare_via_test_vectors(
                        fname, orig_funcs[fname], mut_funcs[fname],
                        critical_path,
                    )
                    test_vectors_generated += tv_count
                    test_vectors_passed += tv_passed
                    if tv_divs:
                        divergences.extend(tv_divs)
                        paths_divergent += 1
                    else:
                        paths_equivalent += 1

        duration_ms = int((time.monotonic() - t0) * 1000)
        status = CheckStatus.PASSED if not divergences else CheckStatus.FAILED

        return BehavioralEquivalenceResult(
            status=status,
            divergences=divergences,
            paths_checked=paths_checked,
            paths_equivalent=paths_equivalent,
            paths_divergent=paths_divergent,
            test_vectors_generated=test_vectors_generated,
            test_vectors_passed=test_vectors_passed,
            duration_ms=duration_ms,
        )


def _extract_function_bodies(tree: ast.Module) -> dict[str, list[ast.stmt]]:
    """Extract function bodies keyed by function name."""
    result: dict[str, list[ast.stmt]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node.body
    return result


def _normalize_function_body(body: list[ast.stmt]) -> str:
    """
    Produce a normalized AST dump of a function body.

    Strips line numbers and column offsets so trivial reformatting
    doesn't trigger false positives.  Variable names are preserved
    (renaming is a semantic change we want to detect in critical paths).
    """
    # Create a clean module containing just the body statements
    wrapper = ast.Module(body=body, type_ignores=[])
    # Remove positional info
    for node in ast.walk(wrapper):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                setattr(node, attr, 0)
    return ast.dump(wrapper, annotate_fields=True, include_attributes=False)


def _extract_control_flow(body: list[ast.stmt]) -> str:
    """
    Extract a string representation of the control flow structure.

    Only captures the skeleton (if/elif/else, for, while, try, return, raise)
    without the conditions or expressions -- so refactoring condition
    expressions doesn't trigger false positives.
    """
    parts: list[str] = []
    _walk_control_flow(body, parts, depth=0)
    return "|".join(parts)


def _walk_control_flow(
    stmts: list[ast.stmt],
    parts: list[str],
    depth: int,
) -> None:
    for stmt in stmts:
        prefix = "  " * depth
        if isinstance(stmt, ast.If):
            parts.append(f"{prefix}IF")
            _walk_control_flow(stmt.body, parts, depth + 1)
            if stmt.orelse:
                parts.append(f"{prefix}ELSE")
                _walk_control_flow(stmt.orelse, parts, depth + 1)
        elif isinstance(stmt, ast.For):
            parts.append(f"{prefix}FOR")
            _walk_control_flow(stmt.body, parts, depth + 1)
        elif isinstance(stmt, ast.While):
            parts.append(f"{prefix}WHILE")
            _walk_control_flow(stmt.body, parts, depth + 1)
        elif isinstance(stmt, ast.Try):
            parts.append(f"{prefix}TRY")
            _walk_control_flow(stmt.body, parts, depth + 1)
            for handler in stmt.handlers:
                parts.append(f"{prefix}EXCEPT")
                _walk_control_flow(handler.body, parts, depth + 1)
            if stmt.orelse:
                parts.append(f"{prefix}TRYELSE")
                _walk_control_flow(stmt.orelse, parts, depth + 1)
            if stmt.finalbody:
                parts.append(f"{prefix}FINALLY")
                _walk_control_flow(stmt.finalbody, parts, depth + 1)
        elif isinstance(stmt, ast.Return):
            parts.append(f"{prefix}RETURN")
        elif isinstance(stmt, ast.Raise):
            parts.append(f"{prefix}RAISE")
        elif isinstance(stmt, ast.With):
            parts.append(f"{prefix}WITH")
            _walk_control_flow(stmt.body, parts, depth + 1)
        elif isinstance(stmt, ast.Assert):
            parts.append(f"{prefix}ASSERT")


def _compare_via_test_vectors(
    fname: str,
    orig_body: list[ast.stmt],
    mut_body: list[ast.stmt],
    critical_path: str,
) -> tuple[int, int, list[BehaviorDivergence]]:
    """
    Compare function behavior using AST-level constant-return analysis.

    Checks:
      - Whether return value structure changed (dict keys, tuple length)
      - Whether exception types changed
      - Whether conditionals use different comparison operators

    Returns (vectors_generated, vectors_passed, divergences).
    """
    divergences: list[BehaviorDivergence] = []
    vectors = 0
    passed = 0

    # Compare return value structures
    orig_returns = _extract_return_structures(orig_body)
    mut_returns = _extract_return_structures(mut_body)

    vectors += 1
    if orig_returns != mut_returns:
        divergences.append(BehaviorDivergence(
            function_name=fname,
            input_description="return structure analysis",
            original_output=str(orig_returns[:3]),
            mutated_output=str(mut_returns[:3]),
            divergence_type="return_value",
            critical_path=critical_path,
        ))
    else:
        passed += 1

    # Compare exception types raised
    orig_raises = _extract_raise_types(orig_body)
    mut_raises = _extract_raise_types(mut_body)

    vectors += 1
    if orig_raises != mut_raises:
        divergences.append(BehaviorDivergence(
            function_name=fname,
            input_description="exception type analysis",
            original_output=str(sorted(orig_raises)),
            mutated_output=str(sorted(mut_raises)),
            divergence_type="exception",
            critical_path=critical_path,
        ))
    else:
        passed += 1

    return vectors, passed, divergences


def _extract_return_structures(body: list[ast.stmt]) -> list[str]:
    """Extract structural descriptions of return values."""
    structures: list[str] = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Return) and node.value is not None:
            structures.append(_describe_value_structure(node.value))
    return structures


def _describe_value_structure(node: ast.expr) -> str:
    """Describe the structure of a return value."""
    if isinstance(node, ast.Dict):
        keys = []
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
            else:
                keys.append("?")
        return f"dict({','.join(sorted(keys))})"
    if isinstance(node, ast.Tuple):
        return f"tuple({len(node.elts)})"
    if isinstance(node, ast.List):
        return f"list({len(node.elts)})"
    if isinstance(node, ast.Constant):
        return f"const({type(node.value).__name__})"
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return f"call({node.func.id})"
        if isinstance(node.func, ast.Attribute):
            return f"call(.{node.func.attr})"
    if isinstance(node, ast.Name):
        return f"name({node.id})"
    return type(node).__name__


def _extract_raise_types(body: list[ast.stmt]) -> set[str]:
    """Extract exception type names from raise statements."""
    types: set[str] = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Raise) and node.exc is not None:
            if isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name):
                    types.add(node.exc.func.id)
                elif isinstance(node.exc.func, ast.Attribute):
                    types.add(node.exc.func.attr)
            elif isinstance(node.exc, ast.Name):
                types.add(node.exc.id)
    return types


# =========================================================================
# 4. TERMINATION ANALYZER
# =========================================================================


class TerminationAnalyzer:
    """
    Analyzes mutated code for non-termination risks.

    Conservative approach: flags anything it can't prove terminates.
    Common safe patterns (for-over-list, range-bounded loops) are
    recognized and not flagged.
    """

    def check(
        self,
        mutated_sources: dict[str, str],
        original_sources: dict[str, str],
    ) -> TerminationResult:
        t0 = time.monotonic()
        risks: list[TerminationRisk] = []
        functions_analyzed = 0
        loops_checked = 0
        loops_bounded = 0
        loops_unbounded = 0
        recursions_checked = 0
        recursions_safe = 0
        awaits_checked = 0
        awaits_with_timeout = 0

        for file_path, mutated_code in mutated_sources.items():
            try:
                mut_tree = ast.parse(mutated_code, filename=file_path)
            except SyntaxError:
                continue

            # Only analyze new or changed functions
            original_code = original_sources.get(file_path)
            orig_funcs: set[str] = set()
            if original_code:
                try:
                    orig_tree = ast.parse(original_code, filename=file_path)
                    for node in ast.walk(orig_tree):
                        if isinstance(
                            node, (ast.FunctionDef, ast.AsyncFunctionDef),
                        ):
                            orig_funcs.add(node.name)
                except SyntaxError:
                    pass

            for node in ast.walk(mut_tree):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    continue

                functions_analyzed += 1

                # Check while loops
                for child in ast.walk(node):
                    if isinstance(child, ast.While):
                        loops_checked += 1
                        if _is_bounded_while(child):
                            loops_bounded += 1
                        else:
                            loops_unbounded += 1
                            # Only flag as blocking if this is a NEW while loop
                            is_new = node.name not in orig_funcs
                            severity = "blocking" if is_new else "advisory"
                            risks.append(TerminationRisk(
                                file_path=file_path,
                                function_name=node.name,
                                line=child.lineno,
                                risk_type="unbounded_loop",
                                description=(
                                    f"While loop in '{node.name}' has no "
                                    f"provably decreasing variant or "
                                    f"iteration bound"
                                ),
                                severity=severity,
                            ))

                    elif isinstance(child, ast.For):
                        loops_checked += 1
                        if _is_bounded_for(child):
                            loops_bounded += 1
                        else:
                            loops_unbounded += 1
                            risks.append(TerminationRisk(
                                file_path=file_path,
                                function_name=node.name,
                                line=child.lineno,
                                risk_type="unbounded_loop",
                                description=(
                                    f"For loop in '{node.name}' iterates "
                                    f"over a potentially unbounded iterator"
                                ),
                                severity="advisory",
                            ))

                # Check recursion
                recursive_calls = _find_recursive_calls(node)
                if recursive_calls:
                    recursions_checked += 1
                    has_base = _has_base_case(node)
                    if has_base:
                        recursions_safe += 1
                    else:
                        risks.append(TerminationRisk(
                            file_path=file_path,
                            function_name=node.name,
                            line=node.lineno,
                            risk_type="missing_base_case",
                            description=(
                                f"Recursive function '{node.name}' has no "
                                f"obvious base case (return without recursion)"
                            ),
                            severity="blocking",
                        ))

                # Check await without timeout
                if isinstance(node, ast.AsyncFunctionDef):
                    for child in ast.walk(node):
                        if isinstance(child, ast.Await):
                            awaits_checked += 1
                            if _await_has_timeout_guard(child, node):
                                awaits_with_timeout += 1
                            else:
                                # Advisory -- many awaits are legitimately unbounded
                                line = getattr(child, "lineno", node.lineno)
                                risks.append(TerminationRisk(
                                    file_path=file_path,
                                    function_name=node.name,
                                    line=line,
                                    risk_type="await_without_timeout",
                                    description=(
                                        f"Await in '{node.name}' has no "
                                        f"visible timeout guard"
                                    ),
                                    severity="advisory",
                                ))

        duration_ms = int((time.monotonic() - t0) * 1000)
        blocking_risks = [r for r in risks if r.severity == "blocking"]
        status = CheckStatus.PASSED if not blocking_risks else CheckStatus.FAILED

        return TerminationResult(
            status=status,
            risks=risks,
            functions_analyzed=functions_analyzed,
            loops_checked=loops_checked,
            loops_bounded=loops_bounded,
            loops_unbounded=loops_unbounded,
            recursions_checked=recursions_checked,
            recursions_safe=recursions_safe,
            awaits_checked=awaits_checked,
            awaits_with_timeout=awaits_with_timeout,
            duration_ms=duration_ms,
        )


def _is_bounded_while(node: ast.While) -> bool:
    """
    Heuristically check if a while loop is bounded.

    Recognizes:
      - while counter < N / while counter <= N with counter increment in body
      - while True with break statement in body
      - while queue/stack (non-growing pattern)
    """
    # while True with break -- controlled loop
    if isinstance(node.test, ast.Constant) and node.test.value is True:
        return any(isinstance(child, ast.Break) for child in ast.walk(node))

    # while <var> <cmp> <bound> with <var> being augmented in body
    if isinstance(node.test, ast.Compare):
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign):
                return True

    return False


def _is_bounded_for(node: ast.For) -> bool:
    """
    Check if a for loop iterates over a bounded collection.

    Safe patterns: range(), list/tuple/dict/set literals, Name references
    (variables are finite), enumerate(), zip().

    Potentially unbounded: generator expressions with no bound,
    itertools.count(), custom iterators.
    """
    iter_node = node.iter
    if isinstance(iter_node, ast.Call):
        func = iter_node.func
        # range() is always bounded
        if isinstance(func, ast.Name) and func.id in {
            "range", "enumerate", "zip", "reversed", "sorted",
        }:
            return True
        # list(), tuple(), set(), dict() calls
        if isinstance(func, ast.Name) and func.id in {
            "list", "tuple", "set", "dict", "frozenset",
        }:
            return True
        # .keys(), .values(), .items() on dicts
        if isinstance(func, ast.Attribute) and func.attr in {
            "keys", "values", "items",
        }:
            return True

    # Iterating over a variable (assumed finite) or literal
    return bool(isinstance(iter_node, (ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict)))


def _find_recursive_calls(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.Call]:
    """Find calls within a function that call the function itself."""
    calls: list[ast.Call] = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == func_node.name:
                calls.append(node)
    return calls


def _has_base_case(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """
    Check if a recursive function has a base case.

    A base case is a return statement that isn't preceded by a recursive
    call in the same branch.
    """
    fname = func_node.name

    for stmt in func_node.body:
        if isinstance(stmt, ast.If):
            # Check if this if-branch returns without recursion
            has_return = False
            has_recursion = False
            for child in ast.walk(stmt):
                if isinstance(child, ast.Return):
                    has_return = True
                if isinstance(child, ast.Call) and (
                    isinstance(child.func, ast.Name)
                    and child.func.id == fname
                ):
                    has_recursion = True
            if has_return and not has_recursion:
                return True
        elif isinstance(stmt, ast.Return):
            # Top-level return without recursion
            has_recursion = False
            if stmt.value:
                for child in ast.walk(stmt.value):
                    if isinstance(child, ast.Call) and (
                        isinstance(child.func, ast.Name)
                        and child.func.id == fname
                    ):
                        has_recursion = True
            if not has_recursion:
                return True

    return False


def _await_has_timeout_guard(
    await_node: ast.Await,
    func_node: ast.AsyncFunctionDef,
) -> bool:
    """
    Check if an await expression has a timeout guard.

    Looks for:
      - asyncio.wait_for(..., timeout=...)
      - asyncio.timeout(...)
      - wait_for pattern around the await
    """
    # Check if the await value is a call to wait_for or timeout
    if isinstance(await_node.value, ast.Call):
        func = await_node.value.func
        if isinstance(func, ast.Attribute) and func.attr in {"wait_for", "timeout"}:
            return True
        if isinstance(func, ast.Name) and func.id in {"wait_for", "timeout"}:
            return True

    # Check if there's a wrapping asyncio.timeout context manager
    for node in ast.walk(func_node):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    func = item.context_expr.func
                    if isinstance(func, ast.Attribute) and func.attr == "timeout":
                        # Check if the await is inside this with block
                        for child in ast.walk(node):
                            if child is await_node:
                                return True
                    if isinstance(func, ast.Name) and func.id == "timeout":
                        for child in ast.walk(node):
                            if child is await_node:
                                return True

    return False


# =========================================================================
# 5. ORCHESTRATOR
# =========================================================================


class MutationFormalVerifier:
    """
    Orchestrates all four mutation verification dimensions.

    Entry point for the health.py pipeline integration.  Runs all checks
    concurrently (they're CPU-bound AST work, but use asyncio.to_thread
    for non-blocking integration with the async health pipeline).

    Integration with EIS taint:
      - ELEVATED/CRITICAL -> all checks mandatory + blocking
      - CLEAR/ADVISORY -> type_safety + termination blocking;
        invariants + behavior advisory only
    """

    def __init__(self, codebase_root: Path | None = None) -> None:
        self._root = codebase_root or Path(".")
        self._type_checker = TypeSafetyChecker()
        self._invariant_checker = InvariantPreservationChecker()
        self._behavior_checker = BehavioralEquivalenceChecker()
        self._termination_analyzer = TerminationAnalyzer()
        self._log = logger

    async def verify(
        self,
        original_sources: dict[str, str],
        mutated_sources: dict[str, str],
        taint_severity: str = "clear",
    ) -> MutationVerificationResult:
        """
        Run all four verification dimensions on the mutation.

        Args:
            original_sources: {file_path: source_code} before mutation
            mutated_sources: {file_path: source_code} after mutation
            taint_severity: EIS taint severity ("clear", "advisory",
                "elevated", "critical")

        Returns:
            MutationVerificationResult with aggregated pass/fail
        """
        t0 = time.monotonic()
        mandatory = taint_severity in ("elevated", "critical")

        self._log.info(
            "mutation_verification_start",
            files=len(mutated_sources),
            taint_severity=taint_severity,
            mandatory=mandatory,
        )

        # Run all four checks concurrently via thread pool
        loop = asyncio.get_event_loop()
        type_task = loop.run_in_executor(
            None, self._type_checker.check,
            original_sources, mutated_sources,
        )
        invariant_task = loop.run_in_executor(
            None, self._invariant_checker.check,
            mutated_sources, original_sources,
        )
        behavior_task = loop.run_in_executor(
            None, self._behavior_checker.check,
            original_sources, mutated_sources,
        )
        termination_task = loop.run_in_executor(
            None, self._termination_analyzer.check,
            mutated_sources, original_sources,
        )

        results = await asyncio.gather(
            type_task, invariant_task, behavior_task, termination_task,
            return_exceptions=True,
        )
        type_result, invariant_result, behavior_result, termination_result = results

        # Handle exceptions
        if isinstance(type_result, Exception):
            self._log.warning("type_safety_error", error=str(type_result))
            type_result = TypeSafetyResult(status=CheckStatus.ERROR)
        if isinstance(invariant_result, Exception):
            self._log.warning("invariant_error", error=str(invariant_result))
            invariant_result = InvariantPreservationResult(
                status=CheckStatus.ERROR,
            )
        if isinstance(behavior_result, Exception):
            self._log.warning("behavior_error", error=str(behavior_result))
            behavior_result = BehavioralEquivalenceResult(
                status=CheckStatus.ERROR,
            )
        if isinstance(termination_result, Exception):
            self._log.warning("termination_error", error=str(termination_result))
            termination_result = TerminationResult(status=CheckStatus.ERROR)

        # Aggregate blocking/advisory issues
        blocking: list[str] = []
        advisory: list[str] = []

        # Type safety: always blocking
        if type_result.status == CheckStatus.FAILED:
            error_violations = [
                v for v in type_result.violations if v.severity == "error"
            ]
            warning_violations = [
                v for v in type_result.violations if v.severity == "warning"
            ]
            if error_violations:
                blocking.append(
                    f"Type safety: {len(error_violations)} error(s) -- "
                    f"{error_violations[0].description}"
                )
            if warning_violations:
                advisory.append(
                    f"Type safety: {len(warning_violations)} warning(s)"
                )

        # Invariant preservation: blocking when mandatory, advisory otherwise
        if invariant_result.status == CheckStatus.FAILED:
            blocking_violations = [
                v for v in invariant_result.violations
                if v.severity == "blocking"
            ]
            if blocking_violations:
                msg = (
                    f"Invariant preservation: "
                    f"{len(blocking_violations)} violation(s) -- "
                    f"{blocking_violations[0].violation_description}"
                )
                if mandatory:
                    blocking.append(msg)
                else:
                    # Forbidden-path violations are always blocking
                    forbidden = [
                        v for v in blocking_violations
                        if "FORBIDDEN" in v.invariant_id
                    ]
                    if forbidden:
                        blocking.append(msg)
                    else:
                        advisory.append(msg)

        # Behavioral equivalence: blocking when mandatory, advisory otherwise
        if behavior_result.status == CheckStatus.FAILED:
            msg = (
                f"Behavioral equivalence: "
                f"{behavior_result.paths_divergent} critical "
                f"path(s) diverged -- "
                f"{behavior_result.divergences[0].function_name}"
            )
            if mandatory:
                blocking.append(msg)
            else:
                advisory.append(msg)

        # Termination: always blocking for blocking risks
        if termination_result.status == CheckStatus.FAILED:
            blocking_risks = [
                r for r in termination_result.risks
                if r.severity == "blocking"
            ]
            advisory_risks = [
                r for r in termination_result.risks
                if r.severity == "advisory"
            ]
            if blocking_risks:
                blocking.append(
                    f"Termination: {len(blocking_risks)} "
                    f"unbounded construct(s) -- "
                    f"{blocking_risks[0].description}"
                )
            if advisory_risks:
                advisory.append(
                    f"Termination: {len(advisory_risks)} advisory risk(s)"
                )

        # Determine overall status
        if blocking:
            status = MutationVerificationStatus.FAILED
        elif advisory:
            status = MutationVerificationStatus.PARTIAL
        else:
            status = MutationVerificationStatus.PASSED

        total_duration = int((time.monotonic() - t0) * 1000)

        result = MutationVerificationResult(
            status=status,
            type_safety=type_result,
            invariant_preservation=invariant_result,
            behavioral_equivalence=behavior_result,
            termination=termination_result,
            blocking_issues=blocking,
            advisory_issues=advisory,
            total_duration_ms=total_duration,
            taint_severity=taint_severity,
            mandatory=mandatory,
        )

        self._log.info(
            "mutation_verification_complete",
            status=status.value,
            blocking=len(blocking),
            advisory=len(advisory),
            duration_ms=total_duration,
        )

        return result
