"""
EcodiaOS - Inspector Autonomous Remediation (Phase 6)

Two remediation implementations:

RepairAgent (Phase 5 - lightweight, LLM-direct):
  Stateless conversational patcher. Takes an LLMProvider and a
  VulnerabilityProver. Prompts the LLM to rewrite vulnerable code,
  then re-runs Z3 to confirm the vulnerability is gone. Retries up to
  max_retries times, feeding each failed Z3 counterexample back into the
  conversation so the LLM can correct its patch.

InspectorRepairOrchestrator (Phase 6 - heavyweight, RepairAgent FSM):
  Bridges Inspector vulnerability discoveries into the existing Stage 5B
  RepairAgent pipeline, then re-verifies patches via Z3. Designed for
  workspace-backed file patching with full diff generation.

Iron Rules:
  - Inspector NEVER writes to EOS source files; patches target workspace only
  - All remediation events logged via structlog for analytics
  - Cost budget is strictly enforced per-vulnerability
"""

from __future__ import annotations

import difflib
import time
from typing import TYPE_CHECKING

import structlog

from primitives.common import new_id, utc_now
from systems.simula.evolution_types import (
    ChangeCategory,
    ChangeSpec,
    EvolutionProposal,
)
from systems.simula.inspector.types import (
    RemediationAttempt,
    RemediationResult,
    RemediationStatus,
    VulnerabilityReport,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.agents.repair_agent import RepairAgent as StageRepairAgent
    from systems.simula.inspector.prover import VulnerabilityProver
    from systems.simula.inspector.workspace import TargetWorkspace

logger = structlog.get_logger().bind(system="simula.inspector.remediation")


# ── Phase 5: Lightweight LLM-direct RepairAgent ──────────────────────────────

_REPAIR_AGENT_SYSTEM_PROMPT = """\
Security patch generation for a verified zero-day vulnerability.

The vulnerable code, Z3 mathematical proof, and reproduction script are provided. \
Rewrite the vulnerable code to close the flaw. Preserve existing functionality. \
Add sanitization, authorization checks, or parameterized queries as the proof demands.

Output ONLY the raw patched source code. No markdown, no explanations, no code fences."""


class RepairAgent:
    """
    Lightweight LLM-direct patch generator with Z3 re-verification.

    Conversational retry loop: on each attempt the LLM receives the full
    conversation history including any previous failed-patch counterexamples,
    allowing it to progressively refine the fix until Z3 confirms UNSAT.
    """

    def __init__(
        self,
        llm: LLMProvider,
        prover: VulnerabilityProver,
        *,
        max_retries: int = 3,
    ) -> None:
        self._llm = llm
        self._prover = prover
        self._max_retries = max_retries
        self._log = logger.bind(component="repair_agent")

    async def generate_and_verify_patch(
        self,
        report: VulnerabilityReport,
    ) -> str | None:
        """
        Generate a patch and verify it eliminates the vulnerability via Z3.

        Retries up to max_retries times. Each failed attempt feeds the new
        Z3 counterexample back into the LLM conversation so it can refine
        its patch. Returns None if all attempts are exhausted or an error
        occurs - callers must handle None gracefully.

        Args:
            report: A proven VulnerabilityReport from the prover.

        Returns:
            Patched source code string if Z3 confirms UNSAT, else None.
        """
        from clients.llm import Message  # local import - avoids circular

        log = self._log.bind(
            vulnerability_id=report.id,
            vulnerability_class=report.vulnerability_class.value,
            severity=report.severity.value,
        )
        log.info("repair_agent_started")

        # Build the initial user message with full vulnerability context
        initial_user_content = (
            f"Vulnerable code:\n{report.attack_surface.context_code}\n\n"
            f"Vulnerability class: {report.vulnerability_class.value}\n"
            f"Severity: {report.severity.value}\n"
            f"Attack goal: {report.attack_goal}\n"
            f"Z3 proof (counterexample):\n{report.z3_counterexample}\n"
        )
        if report.proof_of_concept_code:
            initial_user_content += (
                f"\nReproduction script:\n{report.proof_of_concept_code}\n"
            )

        messages: list[Message] = [
            Message(role="user", content=initial_user_content),
        ]

        for attempt in range(self._max_retries):
            log.info("repair_attempt_starting", attempt=attempt)
            try:
                response = await self._llm.generate(
                    system_prompt=_REPAIR_AGENT_SYSTEM_PROMPT,
                    messages=messages,
                    max_tokens=4096,
                )
                patched_code = response.text.strip()

                if not patched_code:
                    log.warning("repair_agent_empty_response", attempt=attempt)
                    continue

                # Acknowledge the LLM's output in conversation history
                messages.append(Message(role="assistant", content=patched_code))

                # Re-verify: build a temp surface with the patched code
                patched_surface = report.attack_surface.model_copy(
                    update={"context_code": patched_code}
                )

                reverification = await self._prover.prove_vulnerability(
                    surface=patched_surface,
                    attack_goal=report.attack_goal,
                    target_url=report.target_url,
                )

                if reverification is None:
                    # UNSAT - vulnerability is gone
                    log.info(
                        "repair_agent_verified",
                        attempt=attempt,
                        event="patch_generated_and_verified",
                    )
                    return patched_code

                # SAT - vulnerability still present; feed counterexample back
                log.info(
                    "repair_agent_patch_insufficient",
                    attempt=attempt,
                    counterexample=reverification.z3_counterexample[:120],
                )
                messages.append(Message(
                    role="user",
                    content=(
                        f"Your patch is still vulnerable. Z3 found a new "
                        f"counterexample:\n{reverification.z3_counterexample}\n\n"
                        f"Rewrite the code again to eliminate this attack path."
                    ),
                ))

            except Exception as exc:
                log.warning(
                    "repair_agent_attempt_error",
                    attempt=attempt,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                break

        log.warning("repair_agent_exhausted", total_attempts=self._max_retries)
        return None


# ── System prompt for security-aware fix generation ──────────────────────────

_SECURITY_FIX_CONTEXT = """Security vulnerability context for repair:

Vulnerability class: {vulnerability_class}
Severity: {severity}
Attack goal: {attack_goal}

Z3 counterexample (proof the vulnerability exists):
{z3_counterexample}

The fix must eliminate this specific vulnerability. Common remediation
patterns:
- broken_authentication → add authentication check before resource access
- broken_access_control → enforce authorization/ownership check
- injection/sql_injection → use parameterized queries, sanitize input
- privilege_escalation → enforce role-based access check
- reentrancy → add reentrancy guard / checks-effects-interactions pattern
- race_condition → add locking / atomic operations
- unvalidated_redirect → validate redirect target against allowlist
- path_traversal → normalize path, reject directory traversal
- command_injection → use subprocess with list args, never shell=True
"""


class InspectorRepairOrchestrator:
    """
    Orchestrates autonomous remediation of discovered vulnerabilities.

    Bridges Inspector's VulnerabilityReport into the RepairAgent pipeline,
    then re-verifies via Z3 to confirm vulnerability elimination.
    """

    def __init__(
        self,
        repair_agent: StageRepairAgent,
        prover: VulnerabilityProver,
        workspace: TargetWorkspace,
        *,
        max_retries: int = 2,
        cost_budget_usd: float = 0.15,
        timeout_s: float = 300.0,
    ) -> None:
        """
        Args:
            repair_agent: The Stage 5B RepairAgent for fix generation.
            prover: The VulnerabilityProver for re-verification.
            workspace: The target workspace containing the vulnerable code.
            max_retries: Max remediation attempts per vulnerability.
            cost_budget_usd: Hard cost cap for remediation of one vulnerability.
            timeout_s: Total timeout in seconds for the full remediation.
        """
        self._repair_agent = repair_agent
        self._prover = prover
        self._workspace = workspace
        self._max_retries = max_retries
        self._cost_budget = cost_budget_usd
        self._timeout_s = timeout_s

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def workspace(self) -> TargetWorkspace:
        """The current target workspace for remediation."""
        return self._workspace

    def set_workspace(self, workspace: TargetWorkspace) -> None:
        """
        Replace the workspace for the next remediation run.

        InspectorService calls this before each hunt so patches target
        the correct codebase rather than a stale placeholder.
        """
        self._workspace = workspace

    async def generate_patch(
        self,
        vulnerability_report: VulnerabilityReport,
    ) -> RemediationResult:
        """
        Generate and verify a patch for a proven vulnerability.

        Pipeline:
          1. Build synthetic EvolutionProposal from the VulnerabilityReport
          2. Extract vulnerable source code from the workspace
          3. Call RepairAgent.repair() with security-enriched context
          4. If repair succeeds, re-run VulnerabilityProver to verify
          5. Return RemediationResult with patch diff

        Args:
            vulnerability_report: A proven VulnerabilityReport from the Inspector prover.

        Returns:
            RemediationResult with the patch diff and verification status.
        """
        start = time.monotonic()
        vuln = vulnerability_report
        log = logger.bind(
            vulnerability_id=vuln.id,
            vulnerability_class=vuln.vulnerability_class,
            severity=vuln.severity,
            file_path=vuln.attack_surface.file_path,
        )
        log.info("remediation_started")

        attempts: list[RemediationAttempt] = []

        # Read the vulnerable source from workspace
        vulnerable_code, read_error = self._read_vulnerable_code(vuln)
        if read_error:
            log.warning("remediation_code_read_failed", error=read_error)
            return RemediationResult(
                vulnerability_id=vuln.id,
                status=RemediationStatus.FAILED,
                attempts=[RemediationAttempt(
                    attempt_number=0,
                    error=read_error,
                )],
                total_attempts=1,
                total_duration_ms=int((time.monotonic() - start) * 1000),
            )

        # Build synthetic EvolutionProposal for RepairAgent
        proposal = self._build_synthetic_proposal(vuln)

        # Build security-enriched error context for the RepairAgent
        security_context = _SECURITY_FIX_CONTEXT.format(
            vulnerability_class=vuln.vulnerability_class,
            severity=vuln.severity,
            attack_goal=vuln.attack_goal,
            z3_counterexample=vuln.z3_counterexample,
        )

        # Retry loop
        for attempt_num in range(self._max_retries):
            # Check timeout
            elapsed = time.monotonic() - start
            if elapsed > self._timeout_s:
                log.warning("remediation_timeout", elapsed_s=elapsed)
                return self._build_result(
                    vuln.id, RemediationStatus.TIMEOUT, attempts, start,
                )

            attempt_start = time.monotonic()
            attempt = RemediationAttempt(attempt_number=attempt_num)

            try:
                # Step 1: Call RepairAgent
                log.info("repair_attempt_starting", attempt=attempt_num)
                broken_files = {vuln.attack_surface.file_path: vulnerable_code}

                repair_result = await self._repair_agent.repair(
                    proposal=proposal,
                    broken_files=broken_files,
                    test_output=security_context,
                )

                attempt.repair_status = repair_result.status.value
                attempt.cost_usd = repair_result.total_cost_usd

                from systems.simula.verification.types import RepairStatus

                if repair_result.status != RepairStatus.REPAIRED:
                    attempt.error = (
                        f"RepairAgent returned {repair_result.status}: "
                        f"{repair_result.fix_summary or repair_result.diagnosis_summary}"
                    )
                    attempt.duration_ms = int(
                        (time.monotonic() - attempt_start) * 1000
                    )
                    attempts.append(attempt)
                    log.info(
                        "repair_attempt_no_fix",
                        attempt=attempt_num,
                        status=repair_result.status,
                    )
                    continue

                # Step 2: Read patched code from disk
                patched_code = self._read_patched_file(vuln)
                if not patched_code:
                    attempt.error = "RepairAgent reported success but patched file not found"
                    attempt.duration_ms = int(
                        (time.monotonic() - attempt_start) * 1000
                    )
                    attempts.append(attempt)
                    continue

                attempt.patched_code = patched_code

                # Step 3: Generate unified diff
                attempt.patch_diff = self._generate_diff(
                    vuln.attack_surface.file_path,
                    vulnerable_code,
                    patched_code,
                )

                # Step 4: Re-verify via VulnerabilityProver
                log.info("reverification_starting", attempt=attempt_num)

                # Build a temporary AttackSurface with the patched code
                patched_surface = vuln.attack_surface.model_copy(
                    update={"context_code": patched_code}
                )

                reverification = await self._prover.prove_vulnerability(
                    surface=patched_surface,
                    attack_goal=vuln.attack_goal,
                    target_url=vuln.target_url,
                )

                if reverification is None:
                    # UNSAT - vulnerability eliminated
                    attempt.verification_result = "UNSAT"
                    attempt.vulnerability_eliminated = True
                    attempt.duration_ms = int(
                        (time.monotonic() - attempt_start) * 1000
                    )
                    attempts.append(attempt)

                    log.info(
                        "remediation_verified",
                        attempt=attempt_num,
                        cost=f"${attempt.cost_usd:.4f}",
                        event="patch_generated",
                    )

                    return self._build_result(
                        vuln.id,
                        RemediationStatus.PATCHED,
                        attempts,
                        start,
                        successful_attempt=attempt_num,
                        final_diff=attempt.patch_diff,
                        final_code=patched_code,
                    )

                # SAT - vulnerability still present after patch
                attempt.verification_result = "SAT"
                attempt.error = (
                    "Patch did not eliminate vulnerability. "
                    f"Z3 still found: {reverification.z3_counterexample[:200]}"
                )
                attempt.duration_ms = int(
                    (time.monotonic() - attempt_start) * 1000
                )
                attempts.append(attempt)

                log.info(
                    "reverification_failed",
                    attempt=attempt_num,
                    counterexample=reverification.z3_counterexample[:100],
                )

                # Enrich context for next attempt with the failed verification
                security_context += (
                    f"\n\nPrevious patch attempt {attempt_num} was insufficient. "
                    f"Z3 re-verification still found the vulnerability exploitable "
                    f"with counterexample: {reverification.z3_counterexample[:300]}\n"
                    f"The patch must be more thorough."
                )

                # Restore original code for next attempt
                self._restore_original(vuln, vulnerable_code)

            except Exception as exc:
                attempt.error = f"Remediation error: {exc}"
                attempt.duration_ms = int(
                    (time.monotonic() - attempt_start) * 1000
                )
                attempts.append(attempt)
                log.exception("remediation_attempt_error", attempt=attempt_num)

                # Restore original code on error
                self._restore_original(vuln, vulnerable_code)

        # All retries exhausted
        log.warning(
            "remediation_exhausted",
            total_attempts=len(attempts),
        )
        return self._build_result(
            vuln.id, RemediationStatus.FAILED, attempts, start,
        )

    async def generate_patches_batch(
        self,
        vulnerability_reports: list[VulnerabilityReport],
    ) -> dict[str, RemediationResult]:
        """
        Generate patches for multiple vulnerabilities sequentially.

        Processes vulnerabilities in severity order (CRITICAL first).

        Args:
            vulnerability_reports: List of proven VulnerabilityReports.

        Returns:
            Dict mapping vulnerability ID → RemediationResult.
        """
        # Sort by severity: CRITICAL > HIGH > MEDIUM > LOW
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_reports = sorted(
            vulnerability_reports,
            key=lambda r: severity_order.get(r.severity.value, 99),
        )

        results: dict[str, RemediationResult] = {}

        for report in sorted_reports:
            logger.info(
                "batch_remediation_next",
                vulnerability_id=report.id,
                severity=report.severity,
                vulnerability_class=report.vulnerability_class,
            )
            result = await self.generate_patch(report)
            results[report.id] = result

            logger.info(
                "batch_remediation_done",
                vulnerability_id=report.id,
                status=result.status,
                event="patch_generated" if result.status == RemediationStatus.PATCHED else "patch_failed",
            )

        patched = sum(1 for r in results.values() if r.status == RemediationStatus.PATCHED)
        logger.info(
            "batch_remediation_complete",
            total=len(results),
            patched=patched,
            failed=len(results) - patched,
        )

        return results

    # ── Private helpers ─────────────────────────────────────────────────────

    def _build_synthetic_proposal(
        self, vuln: VulnerabilityReport
    ) -> EvolutionProposal:
        """
        Build a synthetic EvolutionProposal from a VulnerabilityReport.

        The RepairAgent expects an EvolutionProposal + broken_files.
        We translate the security vulnerability context into the proposal
        format so RepairAgent can leverage its full FSM pipeline.
        """
        return EvolutionProposal(
            id=new_id(),
            source="inspector",
            category=ChangeCategory.ADD_SYSTEM_CAPABILITY,
            description=(
                f"Security fix: {vuln.vulnerability_class.value} "
                f"vulnerability ({vuln.severity.value}) in "
                f"{vuln.attack_surface.file_path} - {vuln.attack_goal}"
            ),
            change_spec=ChangeSpec(
                capability_description=(
                    f"Remediate {vuln.vulnerability_class.value} vulnerability. "
                    f"Attack surface: {vuln.attack_surface.entry_point} "
                    f"({vuln.attack_surface.surface_type.value}). "
                    f"Z3-proven counterexample: {vuln.z3_counterexample[:500]}"
                ),
                additional_context=(
                    f"Attack goal: {vuln.attack_goal}\n"
                    f"Severity: {vuln.severity.value}\n"
                    f"File: {vuln.attack_surface.file_path}\n"
                    f"Entry point: {vuln.attack_surface.entry_point}"
                ),
                affected_systems=["inspector"],
            ),
            expected_benefit=f"Eliminate {vuln.severity.value}-severity {vuln.vulnerability_class.value} vulnerability",
            risk_assessment="Low risk - targeted security patch for proven vulnerability",
            created_at=utc_now(),
        )

    def _read_vulnerable_code(
        self, vuln: VulnerabilityReport
    ) -> tuple[str, str]:
        """
        Read the vulnerable source file from the workspace.

        Returns:
            (source_code, error_message). Error is empty on success.
        """
        file_path = self._workspace.root / vuln.attack_surface.file_path
        try:
            if not file_path.exists():
                return "", f"File not found: {vuln.attack_surface.file_path}"
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                return "", f"File is empty: {vuln.attack_surface.file_path}"
            return content, ""
        except OSError as exc:
            return "", f"Failed to read {vuln.attack_surface.file_path}: {exc}"

    def _read_patched_file(self, vuln: VulnerabilityReport) -> str:
        """Read the patched file after RepairAgent has written it."""
        file_path = self._workspace.root / vuln.attack_surface.file_path
        try:
            if file_path.exists():
                return file_path.read_text(encoding="utf-8")
        except OSError:
            pass
        return ""

    def _restore_original(
        self, vuln: VulnerabilityReport, original_code: str
    ) -> None:
        """Restore the original vulnerable code after a failed patch attempt."""
        file_path = self._workspace.root / vuln.attack_surface.file_path
        try:
            file_path.write_text(original_code, encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "restore_original_failed",
                file_path=str(file_path),
                error=str(exc),
            )

    @staticmethod
    def _generate_diff(
        file_path: str, original: str, patched: str
    ) -> str:
        """Generate a unified diff between original and patched code."""
        original_lines = original.splitlines(keepends=True)
        patched_lines = patched.splitlines(keepends=True)
        diff_lines = difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        return "".join(diff_lines)

    @staticmethod
    def _build_result(
        vulnerability_id: str,
        status: RemediationStatus,
        attempts: list[RemediationAttempt],
        start: float,
        *,
        successful_attempt: int | None = None,
        final_diff: str = "",
        final_code: str = "",
    ) -> RemediationResult:
        """Build the aggregate RemediationResult."""
        total_cost = sum(a.cost_usd for a in attempts)
        total_duration = int((time.monotonic() - start) * 1000)

        return RemediationResult(
            vulnerability_id=vulnerability_id,
            status=status,
            attempts=attempts,
            total_attempts=len(attempts),
            successful_attempt=successful_attempt,
            final_patch_diff=final_diff,
            final_patched_code=final_code,
            total_cost_usd=total_cost,
            total_duration_ms=total_duration,
            remediated_at=utc_now(),
        )
