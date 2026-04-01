"""
EcodiaOS -- Inspector Phase 3: Mutation Formal Verification Types

Pydantic models for the pre-apply formal verification layer that
mathematically verifies properties of proposed mutations before they
are applied to the codebase.

Four verification dimensions:
  1. Type Safety - mutation preserves type contracts (AST analysis)
  2. Invariant Preservation - constitutional invariants hold post-mutation
  3. Behavioral Equivalence - Equor verdict pipeline output unchanged
     for known inputs on critical paths
  4. Termination Guarantees - no infinite loops introduced

Integration:
  - Runs after type_check + lint, before integration tests
  - Mandatory when EIS flags ELEVATED/CRITICAL taint severity
  - Optional (recommended) for CLEAR/ADVISORY mutations
  - Failed verification → mutation rejected without running it

All types use EOSBaseModel for consistency with Simula.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now

# ── Verification Status ──────────────────────────────────────────────────────


class MutationVerificationStatus(enum.StrEnum):
    """Overall outcome of mutation formal verification."""

    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"       # some checks passed, non-blocking ones failed
    SKIPPED = "skipped"       # verification not required or not configured
    TIMEOUT = "timeout"
    ERROR = "error"           # internal error during verification


class CheckStatus(enum.StrEnum):
    """Outcome of a single verification check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"


# ── 1. Type Safety ───────────────────────────────────────────────────────────


class TypeViolation(EOSBaseModel):
    """A single type contract violation found in the mutation."""

    file_path: str
    line: int = 0
    function_name: str = ""
    description: str = ""
    expected_type: str = ""
    actual_type: str = ""
    severity: str = "error"  # "error" | "warning"


class TypeSafetyResult(EOSBaseModel):
    """
    Result of AST-based type safety verification.

    Checks performed:
      - Function signature compatibility (args, return type annotations)
      - Attribute access on known types
      - Call argument arity and keyword matching
      - Type annotation preservation (annotations not removed)
      - Import integrity (no removed imports that are still used)
    """

    status: CheckStatus = CheckStatus.SKIPPED
    violations: list[TypeViolation] = Field(default_factory=list)
    functions_checked: int = 0
    signatures_preserved: int = 0
    signatures_broken: int = 0
    imports_verified: int = 0
    imports_broken: int = 0
    duration_ms: int = 0


# ── 2. Invariant Preservation ────────────────────────────────────────────────


class InvariantViolation(EOSBaseModel):
    """A constitutional invariant that would be violated by the mutation."""

    invariant_id: str
    invariant_description: str = ""
    violation_description: str = ""
    affected_function: str = ""
    affected_file: str = ""
    counterexample: str = ""  # concrete input that triggers violation
    severity: str = "blocking"  # "blocking" | "advisory"


class InvariantPreservationResult(EOSBaseModel):
    """
    Result of checking whether constitutional invariants hold post-mutation.

    Uses Z3 symbolic checking for invariants with z3_expression,
    AST pattern matching for structural invariants (e.g. soft-delete
    enforcement, audit logging presence), and property-based testing
    via hypothesis for behavioral invariants.

    Constitutional invariants checked:
      - Equor INV-001 through INV-010 (hardcoded invariants)
      - Soft-delete enforcement (no hard DELETE without prevent_hard_delete)
      - Audit logging on sensitive mutations
      - RLS policy preservation
      - Action response contract ({data, error} pattern)
      - Simula iron rules (no self-modification, no Equor modification)
    """

    status: CheckStatus = CheckStatus.SKIPPED
    violations: list[InvariantViolation] = Field(default_factory=list)
    invariants_checked: int = 0
    invariants_preserved: int = 0
    invariants_violated: int = 0
    z3_checks_run: int = 0
    z3_checks_passed: int = 0
    pattern_checks_run: int = 0
    pattern_checks_passed: int = 0
    hypothesis_tests_run: int = 0
    hypothesis_tests_passed: int = 0
    duration_ms: int = 0


# ── 3. Behavioral Equivalence ────────────────────────────────────────────────


class BehaviorDivergence(EOSBaseModel):
    """A case where the mutation changes behavior on a known input."""

    function_name: str
    input_description: str = ""
    original_output: str = ""
    mutated_output: str = ""
    divergence_type: str = ""  # "return_value" | "exception" | "side_effect"
    critical_path: str = ""    # which critical path this belongs to


class BehavioralEquivalenceResult(EOSBaseModel):
    """
    Result of checking behavioral equivalence for critical paths.

    For mutations touching Equor's verdict pipeline (flagged by EIS taint),
    this verifier symbolically executes both the original and mutated code
    on a set of known test vectors to ensure the verdict doesn't change.

    Critical paths checked:
      - Equor verdict pipeline (compute_verdict)
      - Drive evaluator outputs (coherence, care, growth, honesty)
      - Invariant checking functions (INV-001 through INV-010)
      - Risk scoring and composite alignment
      - Governance gating decisions

    Method: AST-level comparison of pure functions + hypothesis-based
    property testing for functions with complex logic. Concrete test
    vectors derived from the function's type annotations and domain.
    """

    status: CheckStatus = CheckStatus.SKIPPED
    divergences: list[BehaviorDivergence] = Field(default_factory=list)
    paths_checked: int = 0
    paths_equivalent: int = 0
    paths_divergent: int = 0
    test_vectors_generated: int = 0
    test_vectors_passed: int = 0
    duration_ms: int = 0


# ── 4. Termination Guarantees ────────────────────────────────────────────────


class TerminationRisk(EOSBaseModel):
    """A potential non-termination risk found in the mutation."""

    file_path: str
    function_name: str = ""
    line: int = 0
    risk_type: str = ""  # "unbounded_loop" | "missing_base_case" | "unbounded_recursion" | "await_without_timeout"
    description: str = ""
    severity: str = "blocking"  # "blocking" | "advisory"


class TerminationResult(EOSBaseModel):
    """
    Result of termination analysis on mutated code.

    Checks:
      - While loops must have a provably decreasing variant or bounded
        iteration count
      - Recursive functions must have a base case reachable from all
        entry points
      - async functions with await must have timeout guards
      - No new infinite generators introduced without consumption bounds
      - For loops over potentially infinite iterators flagged

    Method: AST walk over mutated functions, tracking loop variables
    and recursion depth. Conservative - flags anything it can't prove
    terminates, with advisory severity for common safe patterns
    (e.g. `for x in list` is always bounded).
    """

    status: CheckStatus = CheckStatus.SKIPPED
    risks: list[TerminationRisk] = Field(default_factory=list)
    functions_analyzed: int = 0
    loops_checked: int = 0
    loops_bounded: int = 0
    loops_unbounded: int = 0
    recursions_checked: int = 0
    recursions_safe: int = 0
    awaits_checked: int = 0
    awaits_with_timeout: int = 0
    duration_ms: int = 0


# ── Combined Result ──────────────────────────────────────────────────────────


class MutationVerificationResult(EOSBaseModel):
    """
    Combined result of all four mutation verification dimensions.

    Attached to HealthCheckResult. If any blocking check fails,
    the mutation is rejected without applying it.

    Integration with EIS taint:
      - ELEVATED/CRITICAL taint → all four checks are mandatory + blocking
      - CLEAR/ADVISORY taint → all four checks run but only type_safety
        and termination are blocking; invariants and behavior are advisory
    """

    status: MutationVerificationStatus = MutationVerificationStatus.SKIPPED
    type_safety: TypeSafetyResult | None = None
    invariant_preservation: InvariantPreservationResult | None = None
    behavioral_equivalence: BehavioralEquivalenceResult | None = None
    termination: TerminationResult | None = None
    # Aggregation
    blocking_issues: list[str] = Field(default_factory=list)
    advisory_issues: list[str] = Field(default_factory=list)
    total_duration_ms: int = 0
    # EIS integration
    taint_severity: str = "clear"  # from EIS TaintRiskAssessment
    mandatory: bool = False        # True when taint is ELEVATED/CRITICAL
    verified_at: datetime = Field(default_factory=utc_now)
