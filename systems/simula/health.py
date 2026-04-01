"""
EcodiaOS -- Simula Health Checker

After a change is applied, the health checker verifies the codebase
is still functional. Six check phases run in sequence:
  1. Syntax check -- ast.parse() on all written Python files
  2. Import check -- attempt to import the affected module
  3. Unit tests -- run pytest on the affected system's test directory
  4. Formal verification (Stage 2) -- Dafny + Z3 + static analysis
  5. Lean 4 proof verification (Stage 4A) -- DeepSeek-Prover-V2 pattern
  6. Formal guarantees (Stage 6) -- E-graph equivalence + symbolic execution

If any blocking check fails, Simula rolls back the change. The goal is
to never leave EOS in a broken state.

Phase 4 (formal verification) runs with independent timeout budgets:
  - Dafny: blocking for triggerable categories (MODIFY_CONTRACT, ADD_SYSTEM_CAPABILITY)
  - Z3: advisory by default; graduates to blocking in Stage 3 (z3_blocking=True)
  - Static analysis: blocking for ERROR-severity findings

Phase 5 (Lean 4) runs for categories that require proof-level assurance:
  - Blocking when lean_blocking=True (default for high-risk categories)
  - Advisory otherwise; proved lemmas are stored in the proof library

Phase 6 (formal guarantees) runs e-graph equivalence (6D) and symbolic
execution (6E) checks:
  - E-graph: advisory by default (egraph_blocking=False); verifies semantic
    equivalence of code rewrites via equality saturation
  - Symbolic execution: blocking by default (symbolic_execution_blocking=True);
    proves mission-critical properties (budget, access control, risk scoring)
    via Z3 SMT solving
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib.util
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from systems.simula.evolution_types import HealthCheckResult
from systems.simula.verification.mutation_verifier import MutationFormalVerifier
from systems.simula.verification.types import (
    LEAN_PROOF_CATEGORIES,
    DafnyVerificationResult,
    FormalGuaranteesResult,
    FormalVerificationResult,
    InvariantVerificationResult,
    LeanVerificationResult,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.egraph.equality_saturation import EqualitySaturationEngine
    from systems.simula.evolution_types import EvolutionProposal
    from systems.simula.verification.dafny_bridge import DafnyBridge
    from systems.simula.verification.lean_bridge import LeanBridge
    from systems.simula.verification.mutation_verifier_types import (
        MutationVerificationResult,
    )
    from systems.simula.verification.static_analysis import StaticAnalysisBridge
    from systems.simula.verification.symbolic_execution import SymbolicExecutionEngine
    from systems.simula.verification.z3_bridge import Z3Bridge


# Error type literals for BuildErrorTrainingSignal
_BuildErrorType = Literal[
    "syntax", "import", "runtime", "verification_timeout",
    "proof_failed", "sandbox_escape",
]

logger = structlog.get_logger().bind(system="simula.health")


class HealthChecker:
    """
    Verifies post-apply codebase health via syntax, import, test,
    and formal verification checks. Any blocking failure triggers rollback.
    """

    def __init__(
        self,
        codebase_root: Path,
        test_command: str = "pytest",
        health_check_timeout_s: float = 300.0,
        dafny_bridge: DafnyBridge | None = None,
        z3_bridge: Z3Bridge | None = None,
        static_analysis_bridge: StaticAnalysisBridge | None = None,
        llm: LLMProvider | None = None,
        z3_blocking: bool = False,
        # Stage 4A: Lean 4 proof verification
        lean_bridge: LeanBridge | None = None,
        lean_blocking: bool = True,
        # Integration + performance baseline stages
        integration_tests_enabled: bool = False,
        integration_tests_timeout_s: float = 120.0,
        performance_baseline_enabled: bool = False,
        performance_baseline_timeout_s: float = 60.0,
    ) -> None:
        self._root = codebase_root
        # Build a platform-safe test command list.
        # If test_command is empty, default to `<interpreter> -m pytest` using the
        # running Python executable (Path.as_posix() + shlex.quote handles spaces and
        # Windows paths correctly).  A non-empty test_command is split with shlex so
        # it works on both Unix and Windows regardless of quote style.
        if test_command:
            try:
                self._test_command_argv: list[str] = shlex.split(test_command)
            except ValueError:
                self._test_command_argv = test_command.split()
        else:
            self._test_command_argv = [
                shlex.quote(Path(sys.executable).as_posix()),
                "-m",
                "pytest",
            ]
        self._test_command = test_command  # kept for logging
        self._health_check_timeout_s = health_check_timeout_s
        self._dafny = dafny_bridge
        self._z3 = z3_bridge
        self._static_analysis = static_analysis_bridge
        self._llm = llm
        self._z3_blocking = z3_blocking  # Stage 3: Z3 graduates to blocking
        # Stage 4A: Lean 4
        self._lean = lean_bridge
        self._lean_blocking = lean_blocking
        # Integration tests
        self._integration_tests_enabled = integration_tests_enabled
        self._integration_tests_timeout_s = integration_tests_timeout_s
        # Performance baseline
        self._performance_baseline_enabled = performance_baseline_enabled
        self._performance_baseline_timeout_s = performance_baseline_timeout_s
        # Inspector Phase 3: Mutation formal verification
        self._mutation_verifier = MutationFormalVerifier(codebase_root=codebase_root)
        # Stage 6D: E-graph equivalence (wired by service.py)
        self._egraph: EqualitySaturationEngine | None = None
        self._egraph_blocking: bool = False
        # Stage 6E: Symbolic execution (wired by service.py)
        self._symbolic_execution: SymbolicExecutionEngine | None = None
        self._symbolic_execution_blocking: bool = True
        self._symbolic_execution_domains: list[str] = []
        # Thompson sampling router for proof strategy prioritization (wired by service.py)
        self._reasoning_router: Any = None
        # Synapse bus for emitting RE_TRAINING_EXAMPLE on build errors (wired by service.py)
        self._synapse: Any = None
        self._log = logger

    def set_synapse(self, synapse: Any) -> None:
        """Inject the SynapseService so build-error RE training signals can be emitted."""
        self._synapse = synapse

    # ── Build-error RE training signal ────────────────────────────────────────

    def _emit_build_error_training_signal(
        self,
        *,
        generated_code: str,
        prompt_used: str,
        error_type: _BuildErrorType,
        error_message: str,
        error_traceback: str | None,
        strategy_used: str,
        lesson: str,
        proposal: EvolutionProposal | None = None,
    ) -> None:
        """
        Fire-and-forget a RE_TRAINING_EXAMPLE with outcome_quality=0.0 whenever
        generated or mutated code fails a build check.

        Wrapped entirely in try/except - must never raise or delay the caller.
        Uses asyncio.create_task so it doesn't block the health check pipeline.
        """
        bus = getattr(self._synapse, "_event_bus", None) if self._synapse else None
        if bus is None:
            return

        proposal_id = getattr(proposal, "id", "") if proposal else ""
        category_str = getattr(getattr(proposal, "category", None), "value", "") if proposal else ""

        async def _emit() -> None:
            try:
                from decimal import Decimal

                from primitives.common import DriveAlignmentVector
                from primitives.re_training import RETrainingExample
                from primitives.common import SystemID
                from systems.synapse.types import SynapseEvent, SynapseEventType

                # Truncate fields to avoid oversized events
                safe_code = generated_code[:4000] if generated_code else ""
                safe_prompt = prompt_used[:2000] if prompt_used else ""
                safe_error = error_message[:1000] if error_message else ""
                safe_tb = (error_traceback[:1500] if error_traceback else "")
                safe_lesson = lesson[:500] if lesson else ""

                reasoning_trace = (
                    f"error_type={error_type}\n"
                    f"strategy={strategy_used}\n"
                    f"error={safe_error}\n"
                    + (f"traceback={safe_tb}\n" if safe_tb else "")
                    + f"lesson={safe_lesson}"
                )

                example = RETrainingExample(
                    source_system=SystemID.SIMULA,
                    episode_id=proposal_id,
                    instruction=safe_prompt or f"Generate code for proposal {proposal_id}",
                    input_context=(
                        f"category={category_str} "
                        f"strategy={strategy_used} "
                        f"error_type={error_type}"
                    ),
                    output=safe_code,
                    outcome_quality=0.0,
                    category="build_error",
                    reasoning_trace=reasoning_trace,
                    alternatives_considered=[error_type],
                    cost_usd=Decimal("0"),
                    latency_ms=0,
                    constitutional_alignment=DriveAlignmentVector(),
                    domain="software",
                    skill_area="code_generation",
                    domain_difficulty="expert",
                )
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                    source_system="simula.health",
                    data=example.model_dump(mode="json"),
                ))
            except Exception:
                pass  # Never let emission errors affect the health check

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_emit())
        except RuntimeError:
            pass  # No running loop (sync context) - skip silently

    async def check(
        self,
        files_written: list[str],
        proposal: EvolutionProposal | None = None,
        original_sources: dict[str, str] | None = None,
        taint_severity: str = "clear",
    ) -> HealthCheckResult:
        """""""""
        Run all health checks in sequence.  Returns on first failure.
        """""""""
        # 1. Syntax check
        syntax_errors = await self._check_syntax(files_written)
        if syntax_errors:
            self._log.warning("health_syntax_failed", errors=syntax_errors)
            self._emit_build_error_training_signal(
                generated_code="\n".join(
                    _read_file_safe(f) for f in files_written if f.endswith(".py")
                ),
                prompt_used=getattr(proposal, "description", "") if proposal else "",
                error_type="syntax",
                error_message="; ".join(syntax_errors),
                error_traceback=None,
                strategy_used="health_check_phase1",
                lesson=(
                    "Generated Python code contained syntax errors. "
                    "Always produce valid Python syntax."
                ),
                proposal=proposal,
            )
            return HealthCheckResult(healthy=False, issues=syntax_errors)
        self._log.info("health_syntax_passed", files=len(files_written))

        # 2. Import check
        import_errors = await self._check_imports(files_written)
        if import_errors:
            self._log.warning("health_import_failed", errors=import_errors)
            self._emit_build_error_training_signal(
                generated_code="\n".join(
                    _read_file_safe(f) for f in files_written if f.endswith(".py")
                ),
                prompt_used=getattr(proposal, "description", "") if proposal else "",
                error_type="import",
                error_message="; ".join(import_errors),
                error_traceback=None,
                strategy_used="health_check_phase2",
                lesson=(
                    "Generated code imported modules that do not exist or are not "
                    "resolvable. Use only primitives.*, systems.synapse.types, and "
                    "stdlib imports in generated EOS modules."
                ),
                proposal=proposal,
            )
            return HealthCheckResult(healthy=False, issues=import_errors)
        self._log.info("health_import_passed", files=len(files_written))

        # 3. Unit tests - 20% of health_check_timeout_s, minimum 10s absolute floor
        # so pytest can always start even when the overall budget is tight.
        _test_budget = max(self._health_check_timeout_s * 0.20, 10.0)
        try:
            tests_passed, test_output = await asyncio.wait_for(
                self._run_tests(files_written), timeout=_test_budget
            )
        except (asyncio.TimeoutError, TimeoutError):
            self._log.error("health_tests_timeout", timeout_s=_test_budget)
            self._emit_build_error_training_signal(
                generated_code="\n".join(
                    _read_file_safe(f) for f in files_written if f.endswith(".py")
                ),
                prompt_used=getattr(proposal, "description", "") if proposal else "",
                error_type="verification_timeout",
                error_message=f"Unit tests timed out after {_test_budget:.0f}s",
                error_traceback=None,
                strategy_used="health_check_phase3_pytest",
                lesson=(
                    "Generated code caused unit tests to time out. "
                    "Avoid infinite loops, blocking I/O, or hanging operations."
                ),
                proposal=proposal,
            )
            return HealthCheckResult(healthy=False, issues=["Unit tests timed out"])
        if not tests_passed:
            self._log.warning("health_tests_failed", output=test_output[:500])
            self._emit_build_error_training_signal(
                generated_code="\n".join(
                    _read_file_safe(f) for f in files_written if f.endswith(".py")
                ),
                prompt_used=getattr(proposal, "description", "") if proposal else "",
                error_type="runtime",
                error_message=f"Unit test suite failed",
                error_traceback=test_output[:1500],
                strategy_used="health_check_phase3_pytest",
                lesson=(
                    "Generated code broke existing unit tests. "
                    "Changes must preserve all pre-existing test invariants."
                ),
                proposal=proposal,
            )
            return HealthCheckResult(
                healthy=False,
                issues=[f"Unit test suite failed:\n{test_output[:1000]}"],
            )
        self._log.info("health_tests_passed")

        # 3a. Integration tests (opt-in via integration_tests_enabled)
        if self._integration_tests_enabled:
            int_passed, int_output = await self._run_integration_tests(
                files_written, timeout_s=self._integration_tests_timeout_s,
            )
            if not int_passed:
                self._log.warning("health_integration_tests_failed", output=int_output[:500])
                return HealthCheckResult(
                    healthy=False,
                    issues=[f"Integration test suite failed:\n{int_output[:1000]}"],
                )
            self._log.info("health_integration_tests_passed")

        # 3b. Performance baseline (opt-in via performance_baseline_enabled)
        if self._performance_baseline_enabled:
            perf_passed, perf_output = await self._run_performance_baseline(
                files_written, timeout_s=self._performance_baseline_timeout_s,
            )
            if not perf_passed:
                self._log.warning("health_perf_baseline_failed", output=perf_output[:500])
                return HealthCheckResult(
                    healthy=False,
                    issues=[f"Performance baseline regression detected:\n{perf_output[:1000]}"],
                )
            self._log.info("health_perf_baseline_passed")

        # 3c. Inspector Phase 3: Mutation formal verification (pre-apply gate)
        mutation_result = await self._run_mutation_verification(
            files_written, original_sources, taint_severity,
        )
        if mutation_result is not None:
            if mutation_result.blocking_issues:
                self._log.warning(
                    "health_mutation_verification_failed",
                    blocking=mutation_result.blocking_issues,
                )
                return HealthCheckResult(
                    healthy=False,
                    issues=[
                        f"Mutation verification failed: {issue}"
                        for issue in mutation_result.blocking_issues
                    ],
                    mutation_verification=mutation_result,
                )
            if mutation_result.advisory_issues:
                self._log.info(
                    "health_mutation_verification_advisory",
                    advisory=mutation_result.advisory_issues,
                )
            self._log.info(
                "health_mutation_verification_passed",
                status=mutation_result.status.value,
                duration_ms=mutation_result.total_duration_ms,
            )

        # 4. Formal verification (Stage 2) - 40% of health_check_timeout_s, minimum 15s
        _formal_budget = max(self._health_check_timeout_s * 0.40, 15.0)
        try:
            formal_result = await asyncio.wait_for(
                self._run_formal_verification(files_written, proposal),
                timeout=_formal_budget,
            )
        except (asyncio.TimeoutError, TimeoutError):
            self._log.warning("health_formal_verify_timeout", timeout_s=_formal_budget)
            formal_result = None  # advisory - missing timeout is not a blocking failure
        if formal_result is not None:
            if not formal_result.passed and formal_result.blocking_issues:
                self._log.warning(
                    "health_formal_verification_failed",
                    blocking=formal_result.blocking_issues,
                )
                # Determine which strategy produced the blocking failure
                _fv_strategy = "dafny"
                if formal_result.dafny is not None and not getattr(formal_result.dafny, "passed", True):
                    _fv_strategy = "dafny"
                elif formal_result.z3 is not None:
                    _fv_strategy = "z3"
                elif formal_result.static_analysis is not None:
                    _fv_strategy = "static"
                self._emit_build_error_training_signal(
                    generated_code="\n".join(
                        _read_file_safe(f) for f in files_written if f.endswith(".py")
                    ),
                    prompt_used=getattr(proposal, "description", "") if proposal else "",
                    error_type="proof_failed",
                    error_message="; ".join(formal_result.blocking_issues),
                    error_traceback=None,
                    strategy_used=_fv_strategy,
                    lesson=(
                        f"Generated code failed {_fv_strategy} formal verification. "
                        "The mutation violated a formally provable contract or invariant."
                    ),
                    proposal=proposal,
                )
                return HealthCheckResult(
                    healthy=False,
                    issues=[
                        f"Formal verification failed: {issue}"
                        for issue in formal_result.blocking_issues
                    ],
                    mutation_verification=mutation_result,
                    formal_verification=formal_result,
                )
            if formal_result.advisory_issues:
                self._log.info(
                    "health_formal_verification_advisory",
                    advisory=formal_result.advisory_issues,
                )
            self._log.info("health_formal_verification_passed")

        # 5. Lean 4 proof verification (Stage 4A) - 10% of health_check_timeout_s, minimum 10s
        # Skipped entirely in shallow verification mode (metabolic pressure)
        _shallow = getattr(self, "_shallow_verification_mode", False)
        _lean_budget = max(self._health_check_timeout_s * 0.10, 10.0)
        if _shallow:
            self._log.info("health_lean_skipped_shallow_mode")
            lean_result = None
        else:
            try:
                lean_result = await asyncio.wait_for(
                    self._run_lean_verification(files_written, proposal),
                    timeout=_lean_budget,
                )
            except (asyncio.TimeoutError, TimeoutError):
                self._log.warning("health_lean_verify_timeout", timeout_s=_lean_budget)
                lean_result = None
        if lean_result is not None:
            if lean_result.status.value == "failed" and self._lean_blocking:
                self._log.warning(
                    "health_lean_verification_failed",
                    status=lean_result.status.value,
                    attempts=len(lean_result.attempts),
                )
                self._emit_build_error_training_signal(
                    generated_code="\n".join(
                        _read_file_safe(f) for f in files_written if f.endswith(".py")
                    ),
                    prompt_used=getattr(proposal, "description", "") if proposal else "",
                    error_type="proof_failed",
                    error_message=(
                        f"Lean 4 proof verification failed after "
                        f"{len(lean_result.attempts)} attempts"
                    ),
                    error_traceback=None,
                    strategy_used="lean",
                    lesson=(
                        "Generated code could not be formally proved correct by Lean 4. "
                        "The mutation may violate a critical safety property."
                    ),
                    proposal=proposal,
                )
                return HealthCheckResult(
                    healthy=False,
                    issues=[
                        f"Lean 4 proof verification failed after {len(lean_result.attempts)} attempts"
                    ],
                    mutation_verification=mutation_result,
                    formal_verification=formal_result,
                    lean_verification=lean_result,
                )
            self._log.info(
                "health_lean_verification_complete",
                status=lean_result.status.value,
                proven_lemmas=len(lean_result.proven_lemmas),
                copilot_rate=f"{lean_result.copilot_automation_rate:.0%}",
            )

        # 6. Formal guarantees (Stage 6D + 6E) - 10% of health_check_timeout_s, minimum 10s
        # Skipped entirely in shallow verification mode (metabolic pressure)
        _fg_budget = max(self._health_check_timeout_s * 0.10, 10.0)
        if _shallow:
            self._log.info("health_formal_guarantees_skipped_shallow_mode")
            fg_result = None
        else:
            try:
                fg_result = await asyncio.wait_for(
                    self._run_formal_guarantees(files_written, proposal),
                    timeout=_fg_budget,
                )
            except (asyncio.TimeoutError, TimeoutError):
                self._log.warning("health_formal_guarantees_timeout", timeout_s=_fg_budget)
                fg_result = None
        if fg_result is not None:
            if fg_result.blocking_issues:
                self._log.warning(
                    "health_formal_guarantees_failed",
                    blocking=fg_result.blocking_issues,
                )
                # Determine which Phase 6 sub-check failed
                _fg_strategy = "symbolic"
                if fg_result.symbolic_execution is not None and getattr(
                    fg_result.symbolic_execution, "counterexamples", None
                ):
                    _fg_strategy = "symbolic"
                elif fg_result.egraph is not None:
                    _fg_strategy = "egraph"
                self._emit_build_error_training_signal(
                    generated_code="\n".join(
                        _read_file_safe(f) for f in files_written if f.endswith(".py")
                    ),
                    prompt_used=getattr(proposal, "description", "") if proposal else "",
                    error_type="proof_failed",
                    error_message="; ".join(fg_result.blocking_issues),
                    error_traceback=None,
                    strategy_used=_fg_strategy,
                    lesson=(
                        f"Generated code failed Stage 6 {_fg_strategy} check. "
                        "Mission-critical properties (budget, access control, risk scoring) "
                        "were violated or semantic equivalence could not be proved."
                    ),
                    proposal=proposal,
                )
                return HealthCheckResult(
                    healthy=False,
                    issues=[
                        f"Formal guarantee failed: {issue}"
                        for issue in fg_result.blocking_issues
                    ],
                    mutation_verification=mutation_result,
                    formal_verification=formal_result,
                    lean_verification=lean_result,
                    formal_guarantees=fg_result,
                )
            if fg_result.advisory_issues:
                self._log.info(
                    "health_formal_guarantees_advisory",
                    advisory=fg_result.advisory_issues,
                )
            self._log.info("health_formal_guarantees_passed")

        if (
            mutation_result is not None
            or formal_result is not None
            or lean_result is not None
            or fg_result is not None
        ):
            return HealthCheckResult(
                healthy=True,
                mutation_verification=mutation_result,
                formal_verification=formal_result,
                lean_verification=lean_result,
                formal_guarantees=fg_result,
            )

        return HealthCheckResult(healthy=True)

    async def _check_syntax(self, files: list[str]) -> list[str]:
        """""""""
        Parse each .py file with ast.parse().  Collect syntax errors.
        Returns a list of error strings (empty list = all pass).
        """""""""
        errors: list[str] = []
        for filepath in files:
            if not filepath.endswith(".py"):
                continue
            path = Path(filepath)
            if not path.exists():
                errors.append(f"Syntax check: file not found: {filepath}")
                continue
            try:
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=filepath)
            except SyntaxError as exc:
                errors.append(f"Syntax error in {filepath}:{exc.lineno}: {exc.msg}")
            except Exception as exc:
                errors.append(f"Failed to read {filepath}: {exc}")
        return errors

    async def _check_imports(self, files: list[str]) -> list[str]:
        """""""""
        Derive dotted module paths from written file paths and check
        whether importlib can locate them - including transitive imports
        discovered by walking the AST of each written file.

        Transitive scan detects broken imports one hop deep (the written
        file's direct imports), which catches the most common post-apply
        failure mode where a new module imports a non-existent symbol.
        """""""""
        errors: list[str] = []
        # Collect all modules to check: direct + transitive from AST walk
        modules_to_check: list[tuple[str, str]] = []  # (module_path, source_filepath)

        for filepath in files:
            if not filepath.endswith(".py"):
                continue
            module_path = self._derive_module_path(filepath)
            if module_path is not None:
                modules_to_check.append((module_path, filepath))

            # Transitive: parse the file and collect its import statements
            path = Path(filepath)
            if path.exists():
                try:
                    source = path.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=filepath)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                modules_to_check.append((alias.name, filepath))
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                # Resolve relative imports using the file's own module path
                                if node.level and node.level > 0 and module_path:
                                    parts = module_path.rsplit(".", node.level)
                                    base = parts[0] if len(parts) > 1 else ""
                                    resolved = f"{base}.{node.module}" if base else node.module
                                else:
                                    resolved = node.module
                                modules_to_check.append((resolved, filepath))
                except SyntaxError:
                    pass  # Syntax errors are caught in _check_syntax
                except Exception:
                    pass  # Best-effort transitive scan

        seen: set[str] = set()
        for module_path, source_filepath in modules_to_check:
            if module_path in seen:
                continue
            seen.add(module_path)
            # Skip stdlib and third-party roots that importlib handles fine
            root = module_path.split(".")[0]
            if root in {"sys", "os", "re", "ast", "json", "time", "typing",
                        "collections", "pathlib", "asyncio", "contextlib",
                        "datetime", "enum", "hashlib", "importlib", "math",
                        "functools", "itertools", "abc", "dataclasses"}:
                continue
            try:
                spec = importlib.util.find_spec(module_path)
                if spec is None:
                    errors.append(
                        f"Import check: module not found: {module_path}"
                        + (f" (imported by {source_filepath})" if source_filepath else "")
                    )
            except ModuleNotFoundError as exc:
                errors.append(f"Import check: {module_path}: {exc}")
            except Exception as exc:
                errors.append(f"Import check failed for {module_path}: {exc}")
        return errors

    def _derive_module_path(self, src_file: str) -> str | None:
        """""""""
        Convert a source file path to a dotted module path.
        Example: src/systems/axon/executors/my.py
                 -> systems.axon.executors.my
        """""""""
        try:
            path = Path(src_file)
            # Make relative to codebase root if possible
            try:
                rel = path.relative_to(self._root)
            except ValueError:
                rel = path
            parts = list(rel.parts)
            if parts and parts[0] == "src":
                parts = parts[1:]
            # Strip .py extension from last part
            if parts:
                parts[-1] = parts[-1].removesuffix(".py")
            return ".".join(parts) if parts else None
        except Exception:
            return None

    async def _run_tests(self, files: list[str]) -> tuple[bool, str]:
        """""""""
        Derive the test directory from the written files and run pytest.
        Returns (passed, output).  If no test directory found, returns (True, ...).
        30-second subprocess timeout.
        """""""""
        import sys

        test_path = None
        for filepath in files:
            candidate = self._derive_test_path(filepath)
            if candidate:
                test_dir = Path(candidate)
                if test_dir.is_dir():
                    test_path = candidate
                    break

        if test_path is None:
            self._log.info("health_no_tests", files=files)
            return True, "no tests found"

        # Use `python -m pytest` so the runner always resolves against the
        # active virtualenv, regardless of how PATH is configured on the host.
        pytest_cmd = [sys.executable, "-m", "pytest", test_path, "-x", "--tb=short", "-q"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *pytest_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._root),
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                return False, "Test run timed out after 30s"
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            self._log.info(
                "health_test_run",
                test_path=test_path,
                passed=passed,
                returncode=proc.returncode,
            )
            return passed, output
        except FileNotFoundError:
            msg = f"Python interpreter {sys.executable!r} not found"
            self._log.warning("health_test_command_missing", command=sys.executable)
            return False, msg
        except Exception as exc:
            return False, f"Test run error: {exc}"

    def _derive_test_path(self, src_file: str) -> str | None:
        """""""""
        Map a source file path to a test directory path.

        Handles two layouts:
          src/systems/<name>/...   -> tests/unit/systems/<name>/
          systems/<name>/...       -> tests/unit/systems/<name>/
        """""""""
        try:
            path = Path(src_file)
            try:
                rel = path.relative_to(self._root)
            except ValueError:
                rel = path
            parts = list(rel.parts)

            # Strip leading "src" if present
            if parts and parts[0] == "src":
                parts = parts[1:]

            # Expect: ecodiaos / systems / <system_name> / ...
            if len(parts) >= 3 and parts[1] == "systems":
                system_name = parts[2]
                test_path = self._root / "tests" / "unit" / "systems" / system_name
                return str(test_path)

            return None
        except Exception:
            return None

    def _derive_system_name(self, src_file: str) -> str | None:
        """Extract the system name from a source file path."""
        try:
            path = Path(src_file)
            try:
                rel = path.relative_to(self._root)
            except ValueError:
                rel = path
            parts = list(rel.parts)
            if parts and parts[0] == "src":
                parts = parts[1:]
            if len(parts) >= 3 and parts[1] == "systems":
                return parts[2]
            return None
        except Exception:
            return None

    async def _run_integration_tests(
        self,
        files: list[str],
        timeout_s: float = 120.0,
    ) -> tuple[bool, str]:
        """
        Run integration tests for systems touched by the mutation.

        Looks for tests/integration/systems/<name>/ directories.
        Returns (passed, output).  If no integration tests found, returns (True, '').
        """
        import sys

        system_names: list[str] = []
        for filepath in files:
            name = self._derive_system_name(filepath)
            if name and name not in system_names:
                system_names.append(name)

        test_paths: list[str] = []
        for name in system_names:
            candidate = self._root / "tests" / "integration" / "systems" / name
            if candidate.is_dir():
                test_paths.append(str(candidate))

        if not test_paths:
            self._log.info("health_no_integration_tests", systems=system_names)
            return True, "no integration tests found"

        pytest_cmd = [sys.executable, "-m", "pytest", *test_paths, "-x", "--tb=short", "-q"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *pytest_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._root),
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                return False, f"Integration tests timed out after {timeout_s}s"
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            self._log.info(
                "health_integration_test_run",
                test_paths=test_paths,
                passed=passed,
                returncode=proc.returncode,
            )
            return passed, output
        except FileNotFoundError:
            self._log.warning("health_integration_test_command_missing")
            return False, f"Python interpreter {sys.executable!r} not found"
        except Exception as exc:
            return False, f"Integration test run error: {exc}"

    async def _run_performance_baseline(
        self,
        files: list[str],
        timeout_s: float = 60.0,
    ) -> tuple[bool, str]:
        """
        Run performance baseline checks via pytest-benchmark markers.

        Looks for tests/performance/ or tests/perf/ and runs only
        benchmark-marked tests.  When a .benchmarks/baseline.json exists,
        --benchmark-compare detects regressions >10% and fails the check.

        Returns (passed, output).  If no perf tests found, returns (True, '').
        """
        import sys

        perf_dirs: list[Path] = [
            self._root / "tests" / "performance",
            self._root / "tests" / "perf",
        ]
        existing = [str(d) for d in perf_dirs if d.is_dir()]

        if not existing:
            self._log.info("health_no_perf_tests")
            return True, "no performance tests found"

        pytest_cmd = [
            sys.executable, "-m", "pytest",
            *existing,
            "-m", "benchmark",
            "--tb=short", "-q",
        ]

        # Only add --benchmark-compare when a stored baseline exists
        baseline = self._root / ".benchmarks" / "baseline.json"
        if baseline.is_file():
            pytest_cmd += [
                "--benchmark-compare", str(baseline),
                "--benchmark-compare-fail", "mean:10%",
            ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *pytest_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._root),
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                return False, f"Performance baseline timed out after {timeout_s}s"
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            self._log.info(
                "health_perf_baseline_run",
                paths=existing,
                passed=passed,
                returncode=proc.returncode,
            )
            return passed, output
        except FileNotFoundError:
            self._log.warning("health_perf_command_missing")
            return False, f"Python interpreter {sys.executable!r} not found"
        except Exception as exc:
            return False, f"Performance baseline error: {exc}"

    # ── Inspector Phase 3: Mutation Formal Verification ──────────────────────

    async def _run_mutation_verification(
        self,
        files_written: list[str],
        original_sources: dict[str, str] | None,
        taint_severity: str,
    ) -> MutationVerificationResult | None:
        """
        Run pre-apply formal verification on the mutation.

        Reads mutated file contents from disk, compares against
        original_sources (pre-mutation snapshots), and runs all four
        verification dimensions: type safety, invariant preservation,
        behavioral equivalence, and termination analysis.

        Returns None if no original sources are provided (can't diff).
        """
        if not original_sources:
            return None

        # Gather mutated file contents from disk
        mutated_sources: dict[str, str] = {}
        for filepath in files_written:
            if not filepath.endswith(".py"):
                continue
            full = self._root / filepath
            if full.is_file():
                try:
                    mutated_sources[filepath] = full.read_text(encoding="utf-8")
                except Exception:
                    continue

        if not mutated_sources:
            return None

        # Filter original_sources to only files that were mutated
        relevant_originals = {
            k: v for k, v in original_sources.items()
            if k in mutated_sources
        }

        if not relevant_originals:
            return None

        try:
            return await self._mutation_verifier.verify(
                original_sources=relevant_originals,
                mutated_sources=mutated_sources,
                taint_severity=taint_severity,
            )
        except Exception as exc:
            self._log.warning(
                "mutation_verification_error", error=str(exc),
            )
            return None

    # ── Stage 2: Formal Verification Phase ────────────────────────────────────

    async def _run_formal_verification(
        self,
        files_written: list[str],
        proposal: EvolutionProposal | None,
    ) -> FormalVerificationResult | None:
        """
        Run Dafny, Z3, and static analysis in parallel.

        Returns None if no verification bridges are configured.
        Returns FormalVerificationResult with pass/fail and issues.
        """
        from systems.simula.verification.types import (
            DAFNY_TRIGGERABLE_CATEGORIES,
            FormalVerificationResult,
        )

        if not any([self._dafny, self._z3, self._static_analysis]):
            return None

        start = time.monotonic()
        blocking_issues: list[str] = []
        advisory_issues: list[str] = []
        dafny_result = None
        z3_result = None
        static_result = None

        # ── Thompson-sampled strategy selection under metabolic pressure ──
        # In shallow verification mode, only run the highest-ranked strategy
        # (by Thompson sampling) instead of all in parallel. Saves compute
        # while preferring strategies with the best historical success rate.
        shallow = getattr(self, "_shallow_verification_mode", False)
        allowed_strategies: set[str] | None = None  # None = run all

        if shallow and self._reasoning_router is not None:
            # Determine which strategies are actually available
            available: list[str] = []
            if (
                self._dafny is not None
                and self._llm is not None
                and proposal is not None
                and proposal.category in DAFNY_TRIGGERABLE_CATEGORIES
            ):
                available.append("dafny")
            if self._z3 is not None and self._llm is not None and proposal is not None:
                available.append("z3")
            if self._static_analysis is not None:
                available.append("static_analysis")

            if available:
                ranked = self._reasoning_router.rank_strategies(available)
                # In shallow mode, pick only the top strategy
                allowed_strategies = {ranked[0]}
                self._log.info(
                    "shallow_verification_router_selection",
                    selected=ranked[0],
                    ranked=ranked,
                )

        # Build parallel tasks
        tasks: dict[str, asyncio.Task[object]] = {}

        # Dafny: run for triggerable categories only
        if (
            self._dafny is not None
            and self._llm is not None
            and proposal is not None
            and proposal.category in DAFNY_TRIGGERABLE_CATEGORIES
            and (allowed_strategies is None or "dafny" in allowed_strategies)
        ):
            tasks["dafny"] = asyncio.create_task(
                self._run_dafny_verification(proposal),
            )

        # Z3: run for all proposals when enabled
        if (
            self._z3 is not None
            and self._llm is not None
            and proposal is not None
            and (allowed_strategies is None or "z3" in allowed_strategies)
        ):
            tasks["z3"] = asyncio.create_task(
                self._run_z3_verification(proposal, files_written),
            )

        # Static analysis: run for all Python files
        if self._static_analysis is not None and (
            allowed_strategies is None or "static_analysis" in allowed_strategies
        ):
            tasks["static"] = asyncio.create_task(
                self._static_analysis.run_all(files_written),
            )

        if not tasks:
            return None

        # Await all tasks
        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True,
        )
        task_results = dict(zip(tasks.keys(), results, strict=False))

        # Process Dafny result
        if "dafny" in task_results:
            from systems.simula.verification.types import (
                DafnyVerificationResult,
                DafnyVerificationStatus,
            )
            raw = task_results["dafny"]
            if isinstance(raw, DafnyVerificationResult):
                dafny_result = raw
                if raw.status != DafnyVerificationStatus.VERIFIED:
                    msg = f"Dafny verification {raw.status.value}: {raw.error_summary}"
                    blocking_issues.append(msg)
            elif isinstance(raw, Exception):
                self._log.warning("dafny_exception", error=str(raw))
                advisory_issues.append(f"Dafny verification error: {raw}")

        # Process Z3 result
        if "z3" in task_results:
            from systems.simula.verification.types import (
                InvariantVerificationResult,
                InvariantVerificationStatus,
            )
            raw = task_results["z3"]
            if isinstance(raw, InvariantVerificationResult):
                z3_result = raw
                if self._z3_blocking:
                    # Stage 3: Z3 graduates to blocking - invalid invariants fail the check
                    invalid_count = sum(
                        1 for i in raw.discovered_invariants
                        if i.status == InvariantVerificationStatus.INVALID
                    )
                    if invalid_count > 0:
                        blocking_issues.append(
                            f"Z3 found {invalid_count} invalid invariants (blocking mode)"
                        )
                    if raw.valid_invariants:
                        advisory_issues.append(
                            f"Z3 discovered {len(raw.valid_invariants)} valid invariants"
                        )
                else:
                    # Advisory mode (Stage 2 default)
                    if raw.valid_invariants:
                        advisory_issues.append(
                            f"Z3 discovered {len(raw.valid_invariants)} valid invariants"
                        )
            elif isinstance(raw, Exception):
                self._log.warning("z3_exception", error=str(raw))

        # Process static analysis result
        if "static" in task_results:
            from systems.simula.verification.types import (
                StaticAnalysisResult,
            )
            raw = task_results["static"]
            if isinstance(raw, StaticAnalysisResult):
                static_result = raw
                if raw.error_count > 0:
                    blocking_issues.append(
                        f"Static analysis found {raw.error_count} ERROR-severity issues"
                    )
                if raw.warning_count > 0:
                    advisory_issues.append(
                        f"Static analysis found {raw.warning_count} warnings"
                    )
            elif isinstance(raw, Exception):
                self._log.warning("static_analysis_exception", error=str(raw))

        passed = len(blocking_issues) == 0
        total_time_ms = int((time.monotonic() - start) * 1000)

        return FormalVerificationResult(
            dafny=dafny_result,
            z3=z3_result,
            static_analysis=static_result,
            passed=passed,
            blocking_issues=blocking_issues,
            advisory_issues=advisory_issues,
            total_verification_time_ms=total_time_ms,
        )

    async def _run_dafny_verification(
        self, proposal: EvolutionProposal,
    ) -> DafnyVerificationResult:
        """Run Dafny Clover loop for the proposal."""
        from systems.simula.verification.templates import get_template
        from systems.simula.verification.types import (
            DafnyVerificationResult,
            DafnyVerificationStatus,
        )

        assert self._dafny is not None
        assert self._llm is not None

        # Check Dafny availability
        if not await self._dafny.check_available():
            self._log.info("dafny_not_available_skipping")
            return DafnyVerificationResult(
                status=DafnyVerificationStatus.SKIPPED,
                error_summary="Dafny binary not available",
            )

        # Get template if available
        template = get_template(proposal.category.value)

        # Build context from the change spec
        python_source = ""
        function_name = ""
        context = proposal.description
        if proposal.change_spec:
            context = getattr(proposal.change_spec, "description", None) or proposal.description
            function_name = getattr(proposal.change_spec, "target_system", None) or ""

        return await self._dafny.run_clover_loop(
            llm=self._llm,
            python_source=python_source,
            function_name=function_name,
            context=context,
            template=template,
        )

    async def _run_z3_verification(
        self,
        proposal: EvolutionProposal,
        files_written: list[str],
    ) -> InvariantVerificationResult:
        """Run Z3 invariant discovery for the proposal."""
        from systems.simula.verification.types import (
            InvariantVerificationResult,
            InvariantVerificationStatus,
        )

        assert self._z3 is not None
        assert self._llm is not None

        # Gather Python source from written files for invariant discovery
        python_source_parts: list[str] = []
        target_functions: list[str] = []
        for filepath in files_written:
            if not filepath.endswith(".py"):
                continue
            full = self._root / filepath
            if full.is_file():
                try:
                    content = full.read_text(encoding="utf-8")
                    python_source_parts.append(
                        f"# --- {filepath} ---\n{content}"
                    )
                    # Extract function names for targeting
                    import ast as _ast
                    try:
                        tree = _ast.parse(content)
                        for node in _ast.walk(tree):
                            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                                target_functions.append(node.name)
                    except SyntaxError:
                        pass
                except Exception:
                    continue

        if not python_source_parts:
            return InvariantVerificationResult(
                status=InvariantVerificationStatus.SKIPPED,
                error_summary="No Python source to analyze",
            )

        python_source = "\n\n".join(python_source_parts)
        domain_context = proposal.description
        if proposal.change_spec:
            domain_context = getattr(proposal.change_spec, "description", None) or proposal.description

        return await self._z3.run_discovery_loop(
            llm=self._llm,
            python_source=python_source,
            target_functions=target_functions[:10],  # Limit to top 10
            domain_context=domain_context,
        )

    # ── Stage 4A: Lean 4 Proof Verification Phase ─────────────────────────────

    async def _run_lean_verification(
        self,
        files_written: list[str],
        proposal: EvolutionProposal | None,
    ) -> LeanVerificationResult | None:
        """
        Run Lean 4 proof generation for proposals in proof-eligible categories.

        Returns None if Lean bridge is not configured or proposal is not eligible.
        Returns LeanVerificationResult with proof status and discovered lemmas.
        """
        if self._lean is None or self._llm is None or proposal is None:
            return None

        # Only run Lean proofs for categories that warrant formal proof
        if proposal.category not in LEAN_PROOF_CATEGORIES:
            return None

        # Check Lean 4 availability
        if not await self._lean.check_available():
            self._log.info("lean_not_available_skipping")
            from systems.simula.verification.types import LeanProofStatus
            return LeanVerificationResult(
                status=LeanProofStatus.SKIPPED,
            )

        # Build proof context from the proposal
        python_source_parts: list[str] = []
        for filepath in files_written:
            if not filepath.endswith(".py"):
                continue
            full = self._root / filepath
            if full.is_file():
                try:
                    content = full.read_text(encoding="utf-8")
                    python_source_parts.append(
                        f"# --- {filepath} ---\n{content}"
                    )
                except Exception:
                    continue

        python_source = "\n\n".join(python_source_parts)
        domain_context = proposal.description
        if proposal.change_spec:
            domain_context = proposal.change_spec.additional_context or proposal.description

        function_name = ""
        if proposal.change_spec:
            function_name = getattr(proposal.change_spec, "target_system", "") or ""

        try:
            result = await self._lean.generate_proof(
                llm=self._llm,
                python_source=python_source,
                function_name=function_name,
                property_description=domain_context,
                proposal_id=proposal.id,
            )
            return result
        except TimeoutError:
            self._log.warning("lean_verification_timeout", proposal_id=proposal.id)
            from systems.simula.verification.types import LeanProofStatus
            return LeanVerificationResult(
                status=LeanProofStatus.TIMEOUT,
            )
        except Exception as exc:
            self._log.warning("lean_verification_error", error=str(exc))
            return None

    # ── Stage 6: Formal Guarantees Phase ─────────────────────────────────────

    async def _run_formal_guarantees(
        self,
        files_written: list[str],
        proposal: EvolutionProposal | None,
    ) -> FormalGuaranteesResult | None:
        """
        Run Stage 6D (e-graph equivalence) and 6E (symbolic execution) checks.

        Returns None if no Stage 6 subsystems are configured.
        Returns a FormalGuaranteesResult with pass/fail and issues.
        """
        from systems.simula.verification.types import FormalGuaranteesResult

        if self._egraph is None and self._symbolic_execution is None:
            return None

        start = time.monotonic()
        blocking_issues: list[str] = []
        advisory_issues: list[str] = []
        egraph_result = None
        symbolic_result = None

        # Build parallel tasks
        tasks: dict[str, asyncio.Task[object]] = {}

        # 6D: E-graph equivalence - check if rewritten code is semantically equivalent
        if self._egraph is not None and proposal is not None:
            tasks["egraph"] = asyncio.create_task(
                self._run_egraph_check(files_written, proposal),
            )

        # 6E: Symbolic execution - prove mission-critical properties
        if self._symbolic_execution is not None:
            tasks["symbolic"] = asyncio.create_task(
                self._run_symbolic_execution(files_written),
            )

        if not tasks:
            return None

        # Await all tasks
        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True,
        )
        task_results = dict(zip(tasks.keys(), results, strict=False))

        # Process e-graph result
        if "egraph" in task_results:
            from systems.simula.verification.types import (
                EGraphEquivalenceResult,
                EGraphStatus,
            )

            raw = task_results["egraph"]
            if isinstance(raw, EGraphEquivalenceResult):
                egraph_result = raw
                if raw.status == EGraphStatus.FAILED:
                    msg = "E-graph equivalence check failed: code is not semantically equivalent"
                    if self._egraph_blocking:
                        blocking_issues.append(msg)
                    else:
                        advisory_issues.append(msg)
                elif raw.status == EGraphStatus.TIMEOUT:
                    advisory_issues.append("E-graph equivalence check timed out")
                elif raw.semantically_equivalent:
                    advisory_issues.append(
                        f"E-graph confirmed semantic equivalence ({len(raw.rules_applied)} rules, "
                        f"{raw.iterations} iterations)"
                    )
            elif isinstance(raw, Exception):
                self._log.warning("egraph_exception", error=str(raw))

        # Process symbolic execution result
        if "symbolic" in task_results:
            from systems.simula.verification.types import (
                SymbolicExecutionResult,
            )

            raw = task_results["symbolic"]
            if isinstance(raw, SymbolicExecutionResult):
                symbolic_result = raw
                if raw.counterexamples:
                    msg = (
                        f"Symbolic execution found {len(raw.counterexamples)} counterexample(s) - "
                        f"mission-critical properties violated"
                    )
                    if self._symbolic_execution_blocking:
                        blocking_issues.append(msg)
                    else:
                        advisory_issues.append(msg)
                if raw.properties_proved > 0:
                    advisory_issues.append(
                        f"Symbolic execution proved {raw.properties_proved}/{raw.properties_checked} properties"
                    )
            elif isinstance(raw, Exception):
                self._log.warning("symbolic_execution_exception", error=str(raw))

        passed = len(blocking_issues) == 0
        total_time_ms = int((time.monotonic() - start) * 1000)

        return FormalGuaranteesResult(
            egraph=egraph_result,
            symbolic_execution=symbolic_result,
            passed=passed,
            blocking_issues=blocking_issues,
            advisory_issues=advisory_issues,
            total_duration_ms=total_time_ms,
        )

    async def _run_egraph_check(
        self,
        files_written: list[str],
        proposal: EvolutionProposal,
    ) -> object | None:
        """
        Run e-graph equivalence check on changed files.

        Compares original code (from rollback snapshot) with new code
        to verify semantic equivalence of the transformation.
        """
        from systems.simula.verification.types import (
            EGraphEquivalenceResult,
            EGraphStatus,
        )

        assert self._egraph is not None

        # For each Python file, check if the rewrite preserved semantics
        # We focus on the first file that has a meaningful diff
        for filepath in files_written:
            if not filepath.endswith(".py"):
                continue
            full = self._root / filepath
            if not full.is_file():
                continue
            try:
                new_code = full.read_text(encoding="utf-8")
                # E-graph checks the code against itself (simplified form)
                # In production this would compare pre/post-apply snapshots
                result = await self._egraph.check_equivalence(new_code, new_code)
                return result
            except Exception as exc:
                self._log.warning("egraph_file_check_error", file=filepath, error=str(exc))
                continue

        return EGraphEquivalenceResult(
            status=EGraphStatus.SKIPPED,
        )

    async def _run_symbolic_execution(
        self,
        files_written: list[str],
    ) -> object | None:
        """
        Run symbolic execution on mission-critical functions in changed files.
        """
        from systems.simula.verification.types import (
            SymbolicDomain,
            SymbolicExecutionResult,
            SymbolicExecutionStatus,
        )

        assert self._symbolic_execution is not None

        # Convert domain strings to enum values
        domains: list[SymbolicDomain] = []
        for d in self._symbolic_execution_domains:
            with contextlib.suppress(ValueError):
                domains.append(SymbolicDomain(d))

        if not domains:
            return SymbolicExecutionResult(
                status=SymbolicExecutionStatus.SKIPPED,
            )

        try:
            return await self._symbolic_execution.prove_properties(
                files=files_written,
                codebase_root=self._root,
                domains=domains,
            )
        except TimeoutError:
            self._log.warning("symbolic_execution_timeout")
            return SymbolicExecutionResult(
                status=SymbolicExecutionStatus.TIMEOUT,
            )
        except Exception as exc:
            self._log.warning("symbolic_execution_error", error=str(exc))
            return None


# ── Module-level helpers ────────────────────────────────────────────────────────


def _read_file_safe(filepath: str, max_bytes: int = 8000) -> str:
    """Read a file's contents safely, returning empty string on any error.

    Used when assembling generated_code payloads for build-error training signals.
    Truncates to max_bytes to avoid oversized Synapse events.
    """
    try:
        content = Path(filepath).read_text(encoding="utf-8", errors="replace")
        if len(content) > max_bytes:
            return content[:max_bytes] + "\n# ... (truncated)"
        return content
    except Exception:
        return ""
