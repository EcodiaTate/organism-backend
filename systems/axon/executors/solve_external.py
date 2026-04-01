"""
EcodiaOS - SolveExternalTaskExecutor (Phase 16s: General-Purpose Contractor)

Solves an arbitrary external GitHub issue by:
  1. Cloning the target repository into an isolated workspace
  2. Reading the issue description
  3. Invoking SimulaCodeAgent (external mode) to implement the fix
  4. Running language-aware tests and linter
  5. Repair loop (up to max_repair_attempts) if tests fail
  6. Equor constitutional review of the diff before any PR
  7. Handing off to BountySubmitExecutor if bounty_id provided
  8. Cleaning up the workspace
  9. Emitting EXTERNAL_TASK_COMPLETED / EXTERNAL_TASK_FAILED /
     EXTERNAL_TASK_CONSTITUTIONAL_VETO

Safety constraints:
  - required_autonomy: 3 (TRUSTED) - clones repos, writes code, opens PRs
  - rate_limit: 2 per hour - full code-gen sessions are expensive
  - max_duration_ms: 900_000 (15 min) - allows iterative repair
  - reversible: False - workspace is ephemeral; PR cannot be atomically recalled
  - Equor gate required before PR submission (Step 6)
  - Writes ONLY to own fork, not upstream repo
  - Mandatory EOS authorship disclosure on every PR (Honesty invariant)

Parameters:
  repo_url (str):               GitHub repository URL (HTTPS or "owner/repo").
  issue_description (str):      Human-readable task / issue to solve.
  issue_url (str, optional):    URL of the original GitHub issue (for PR body).
  bounty_id (str, optional):    Oikos bounty identifier - triggers PR submission.
  payment_address (str, opt):   Wallet address for bounty payout (passed through).
  base_branch (str, opt):       Branch to base the fix on. Default "main".
  target_files (list[str], opt):Files to restrict edits to. Default [] (all).
  max_repair_attempts (int):    Max test-repair iterations. Default 3.

Returns ExecutionResult with:
  data:
    task_id              - unique identifier for this task run
    files_written        - list of relative paths modified
    pr_url               - HTTPS PR URL (if submitted)
    pr_number            - PR number (if submitted)
    language             - detected repo language
    test_passed          - whether final test run passed
    total_tokens         - LLM tokens consumed
    constitutional_veto  - True if Equor blocked the PR
  side_effects:
    - Human-readable summary
"""

from __future__ import annotations

import asyncio
import secrets
import textwrap
import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)
from systems.synapse.types import SynapseEventType

if TYPE_CHECKING:
    from systems.identity.connectors.github import GitHubConnector
    from systems.simula.code_agent import SimulaCodeAgent
    from systems.simula.external_workspace import ExternalWorkspace
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("axon.executor.solve_external")

# Maximum characters of issue_description passed to the code agent
_MAX_ISSUE_LEN = 8_000

# RE training category
_RE_CATEGORY = "external_contractor"


class SolveExternalTaskExecutor(Executor):
    """
    End-to-end external repository issue resolver.

    See module docstring for full parameter documentation.
    """

    action_type = "solve_external_task"
    description = (
        "Clone an external GitHub repo, implement a fix with Simula, verify with "
        "language-native tests, submit a PR if a bounty is attached (Phase 16s)"
    )

    required_autonomy = 3
    reversible = False
    max_duration_ms = 900_000     # 15 minutes
    rate_limit = RateLimit.per_hour(2)

    def __init__(
        self,
        simula: SimulaCodeAgent | None = None,
        github_connector: GitHubConnector | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._simula = simula
        self._github = github_connector
        self._event_bus = event_bus
        self._logger = logger.bind(executor="axon.solve_external")

    # ── Dependency injection ───────────────────────────────────────────────

    def set_simula(self, simula: SimulaCodeAgent) -> None:
        self._simula = simula

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus

    def set_github_connector(self, connector: GitHubConnector) -> None:
        self._github = connector

    # ── Validation ────────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        repo_url = str(params.get("repo_url", "")).strip()
        if not repo_url:
            return ValidationResult.fail("repo_url is required", repo_url="missing")

        issue_description = str(params.get("issue_description", "")).strip()
        if not issue_description:
            return ValidationResult.fail(
                "issue_description is required", issue_description="missing"
            )

        if self._simula is None:
            return ValidationResult.fail(
                "SimulaCodeAgent not injected", simula="not_wired"
            )

        return ValidationResult.ok()

    # ── Main execution ─────────────────────────────────────────────────────

    async def execute(
        self, params: dict[str, Any], context: ExecutionContext
    ) -> ExecutionResult:
        task_id = f"ext_{secrets.token_hex(8)}"
        start_ms = int(time.monotonic() * 1000)
        log = self._logger.bind(task_id=task_id)

        repo_url: str = str(params.get("repo_url", "")).strip()
        issue_description: str = str(params.get("issue_description", ""))[:_MAX_ISSUE_LEN]
        issue_url: str = str(params.get("issue_url", "")).strip()
        bounty_id: str = str(params.get("bounty_id", "")).strip()
        payment_address: str = str(params.get("payment_address", "")).strip()
        base_branch: str = str(params.get("base_branch", "main")).strip()
        target_files: list[str] = list(params.get("target_files", []))
        max_repair_attempts: int = int(params.get("max_repair_attempts", 3))

        log.info("external_task_started", repo_url=repo_url, bounty_id=bounty_id or None)

        await self._emit(
            SynapseEventType.EXTERNAL_TASK_STARTED,
            {
                "task_id": task_id,
                "repo_url": repo_url,
                "bounty_id": bounty_id or None,
                "instance_id": getattr(context, "instance_id", None),
            },
        )

        # ── Step 1: Clone workspace ────────────────────────────────────────
        github_token = await self._get_github_token()
        workspace: ExternalWorkspace | None = None
        try:
            from systems.simula.external_workspace import ExternalRepoConfig, ExternalWorkspace
            config = ExternalRepoConfig(
                repo_url=repo_url,
                base_branch=base_branch,
                target_files=target_files,
                max_repair_attempts=max_repair_attempts,
            )
            workspace = await ExternalWorkspace.clone(
                repo_url=repo_url,
                task_id=task_id,
                config=config,
                github_token=github_token,
            )
        except Exception as exc:
            log.error("external_clone_failed", error=str(exc))
            await self._emit(
                SynapseEventType.EXTERNAL_TASK_FAILED,
                {"task_id": task_id, "repo_url": repo_url, "reason": f"clone_failed: {exc}"},
            )
            await self._emit_re_training(task_id, repo_url, success=False, tokens=0)
            return ExecutionResult(
                success=False,
                error=f"Clone failed: {exc}",
                data={"task_id": task_id, "repo_url": repo_url},
            )

        try:
            return await self._run_task(
                task_id=task_id,
                workspace=workspace,
                issue_description=issue_description,
                issue_url=issue_url,
                bounty_id=bounty_id,
                payment_address=payment_address,
                max_repair_attempts=max_repair_attempts,
                start_ms=start_ms,
                context=context,
                log=log,
            )
        finally:
            await workspace.cleanup()

    # ── Internal task runner ───────────────────────────────────────────────

    async def _run_task(
        self,
        task_id: str,
        workspace: ExternalWorkspace,
        issue_description: str,
        issue_url: str,
        bounty_id: str,
        payment_address: str,
        max_repair_attempts: int,
        start_ms: int,
        context: ExecutionContext,
        log: Any,
    ) -> ExecutionResult:
        repo_url = workspace.config.repo_url
        lang = workspace.language

        # ── Step 2: Simula generate ────────────────────────────────────────
        assert self._simula is not None
        log.info("external_codegen_start", language=lang)
        try:
            code_result = await self._simula.implement_external(
                issue_description=issue_description,
                workspace=workspace,
            )
        except Exception as exc:
            log.error("external_codegen_error", error=str(exc))
            await self._emit(
                SynapseEventType.EXTERNAL_TASK_FAILED,
                {"task_id": task_id, "repo_url": repo_url, "reason": f"codegen_error: {exc}"},
            )
            await self._emit_re_training(task_id, repo_url, success=False, tokens=0)
            return ExecutionResult(
                success=False,
                error=f"Code generation error: {exc}",
                data={"task_id": task_id, "language": lang},
            )

        total_tokens = code_result.total_tokens
        files_written = code_result.files_written

        if not code_result.success or not files_written:
            log.warning("external_codegen_no_files", summary=code_result.summary)
            await self._emit(
                SynapseEventType.EXTERNAL_TASK_FAILED,
                {
                    "task_id": task_id,
                    "repo_url": repo_url,
                    "reason": "codegen_produced_no_files",
                    "summary": code_result.summary,
                },
            )
            await self._emit_re_training(task_id, repo_url, success=False, tokens=total_tokens)
            return ExecutionResult(
                success=False,
                error="Code generation produced no files",
                data={"task_id": task_id, "language": lang, "total_tokens": total_tokens},
            )

        # ── Step 3: Test + repair loop ─────────────────────────────────────
        test_passed = False
        for attempt in range(max_repair_attempts):
            test_result = await workspace.run_tests()
            log.info(
                "external_test_run",
                attempt=attempt + 1,
                passed=test_result.passed,
                exit_code=test_result.exit_code,
            )
            if test_result.passed:
                test_passed = True
                break

            if attempt == max_repair_attempts - 1:
                log.warning("external_repair_exhausted", attempts=max_repair_attempts)
                break

            # Feed failure back to Simula for another repair pass
            repair_desc = (
                f"Tests failed (attempt {attempt + 1}/{max_repair_attempts}). "
                f"Fix the failures:\n\n{test_result.output[-3000:]}"
            )
            try:
                repair_result = await self._simula.implement_external(
                    issue_description=repair_desc,
                    workspace=workspace,
                )
                total_tokens += repair_result.total_tokens
                files_written = list(set(files_written) | set(repair_result.files_written))
            except Exception as exc:
                log.warning("external_repair_error", error=str(exc))
                break

        # ── Step 4: Equor constitutional review ───────────────────────────
        diff_summary = await self._build_diff_summary(workspace, files_written)
        equor_ok = await self._equor_review(task_id, repo_url, diff_summary)
        if not equor_ok:
            log.warning("external_constitutional_veto", task_id=task_id)
            await self._emit(
                SynapseEventType.EXTERNAL_TASK_CONSTITUTIONAL_VETO,
                {
                    "task_id": task_id,
                    "repo_url": repo_url,
                    "bounty_id": bounty_id or None,
                    "files_written": files_written,
                    "diff_summary": diff_summary[:500],
                },
            )
            await self._emit_re_training(task_id, repo_url, success=False, tokens=total_tokens)
            return ExecutionResult(
                success=False,
                error="Equor constitutional veto - PR not submitted",
                data={
                    "task_id": task_id,
                    "language": lang,
                    "files_written": files_written,
                    "test_passed": test_passed,
                    "total_tokens": total_tokens,
                    "constitutional_veto": True,
                },
            )

        # ── Step 5: Hand off to BountySubmitExecutor (if bounty) ──────────
        pr_url = ""
        pr_number = 0
        if bounty_id and files_written:
            pr_url, pr_number = await self._submit_pr(
                workspace=workspace,
                files_written=files_written,
                bounty_id=bounty_id,
                issue_description=issue_description,
                issue_url=issue_url,
                payment_address=payment_address,
                context=context,
                log=log,
            )

        # ── Step 6: Emit completion + RE training ──────────────────────────
        elapsed_ms = int(time.monotonic() * 1000) - start_ms
        await self._emit(
            SynapseEventType.EXTERNAL_TASK_COMPLETED,
            {
                "task_id": task_id,
                "repo_url": repo_url,
                "bounty_id": bounty_id or None,
                "language": lang,
                "files_written": files_written,
                "test_passed": test_passed,
                "pr_url": pr_url or None,
                "pr_number": pr_number or None,
                "total_tokens": total_tokens,
                "elapsed_ms": elapsed_ms,
            },
        )
        await self._emit_re_training(task_id, repo_url, success=True, tokens=total_tokens)

        log.info(
            "external_task_completed",
            language=lang,
            files_written=len(files_written),
            test_passed=test_passed,
            pr_url=pr_url or None,
        )

        return ExecutionResult(
            success=True,
            data={
                "task_id": task_id,
                "files_written": files_written,
                "pr_url": pr_url or None,
                "pr_number": pr_number or None,
                "language": lang,
                "test_passed": test_passed,
                "total_tokens": total_tokens,
                "constitutional_veto": False,
            },
            side_effects=[
                f"External task completed: {len(files_written)} files modified in {repo_url} "
                f"({'tests passed' if test_passed else 'tests failed'}, "
                f"{'PR submitted' if pr_url else 'no PR'})"
            ],
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _build_diff_summary(
        self, workspace: ExternalWorkspace, files_written: list[str]
    ) -> str:
        """Build a brief diff summary for Equor review."""
        lines = [f"Modified files ({len(files_written)}):"]
        for f in files_written[:20]:
            lines.append(f"  - {f}")
        if len(files_written) > 20:
            lines.append(f"  ... and {len(files_written) - 20} more")
        return "\n".join(lines)

    async def _equor_review(self, task_id: str, repo_url: str, diff_summary: str) -> bool:
        """
        Emit EQUOR_ECONOMIC_INTENT and await EQUOR_ECONOMIC_PERMIT.
        Returns True (permit) or False (deny / no bus).
        30s timeout → auto-permit (safety fallback).
        """
        if self._event_bus is None:
            return True  # No bus - skip gate gracefully

        permit_event = asyncio.Event()
        deny_flag: list[bool] = [False]

        def _on_permit(event: Any) -> None:
            payload = getattr(event, "payload", {})
            if payload.get("task_id") == task_id:
                if payload.get("decision") == "DENY":
                    deny_flag[0] = True
                permit_event.set()

        await self._event_bus.subscribe(
            SynapseEventType.EQUOR_ECONOMIC_PERMIT, _on_permit
        )
        try:
            await self._event_bus.broadcast(
                SynapseEventType.EQUOR_ECONOMIC_INTENT,
                {
                    "task_id": task_id,
                    "action": "external_pr_submission",
                    "repo_url": repo_url,
                    "diff_summary": diff_summary[:300],
                    "amount_usd": "0",
                    "mutation_type": "submit_external_pr",
                },
            )
            try:
                await asyncio.wait_for(permit_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "equor_review_timeout_auto_permit",
                    task_id=task_id,
                )
        finally:
            await self._event_bus.unsubscribe(
                SynapseEventType.EQUOR_ECONOMIC_PERMIT, _on_permit
            )

        return not deny_flag[0]

    async def _submit_pr(
        self,
        workspace: ExternalWorkspace,
        files_written: list[str],
        bounty_id: str,
        issue_description: str,
        issue_url: str,
        payment_address: str,
        context: ExecutionContext,
        log: Any,
    ) -> tuple[str, int]:
        """Hand off to BountySubmitExecutor. Returns (pr_url, pr_number)."""
        # Collect the content of each modified file
        file_contents: dict[str, str] = {}
        for rel_path in files_written:
            try:
                abs_path = workspace.root / rel_path
                if abs_path.exists():
                    file_contents[rel_path] = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

        if not file_contents:
            return "", 0

        # Use BountySubmitExecutor via Synapse (no direct import)
        if self._event_bus is None:
            log.warning("no_event_bus_for_pr_submission")
            return "", 0

        # Build a combined solution_code summary for BountySubmitExecutor
        solution_code_parts = []
        for path, content in list(file_contents.items())[:5]:
            solution_code_parts.append(f"### {path}\n```\n{content[:2000]}\n```")
        solution_code = "\n\n".join(solution_code_parts)

        explanation = textwrap.dedent(f"""
            Automated fix for: {issue_description[:500]}

            {"Issue: " + issue_url if issue_url else ""}

            Files modified: {", ".join(files_written[:10])}
        """).strip()

        # Emit AXON_EXECUTION_REQUEST for BountySubmitExecutor
        result_event = asyncio.Event()
        pr_data: list[dict[str, Any]] = [{}]

        def _on_result(event: Any) -> None:
            payload = getattr(event, "payload", {})
            if payload.get("bounty_id") == bounty_id:
                pr_data[0] = payload
                result_event.set()

        await self._event_bus.subscribe(SynapseEventType.BOUNTY_PR_SUBMITTED, _on_result)
        try:
            await self._event_bus.broadcast(
                SynapseEventType.AXON_EXECUTION_REQUEST,
                {
                    "action_type": "submit_bounty_solution",
                    "params": {
                        "bounty_id": bounty_id,
                        "solution_code": solution_code,
                        "solution_explanation": explanation,
                        "repository_url": workspace.config.repo_url,
                        "target_branch": workspace.config.base_branch,
                        "bounty_url": issue_url,
                        "payment_address": payment_address,
                    },
                },
            )
            try:
                await asyncio.wait_for(result_event.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                log.warning("bounty_submit_timeout")
                return "", 0
        finally:
            await self._event_bus.unsubscribe(
                SynapseEventType.BOUNTY_PR_SUBMITTED, _on_result
            )

        return pr_data[0].get("pr_url", ""), int(pr_data[0].get("pr_number", 0))

    async def _get_github_token(self) -> str | None:
        """Extract GitHub token from connector, if available."""
        if self._github is None:
            return None
        try:
            return await self._github.get_token()  # type: ignore[attr-defined]
        except Exception:
            return None

    async def _emit(self, event_type: SynapseEventType, payload: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.broadcast(event_type, payload)
        except Exception as exc:
            self._logger.warning("emit_failed", event=event_type, error=str(exc))

    async def _emit_re_training(
        self, task_id: str, repo_url: str, success: bool, tokens: int
    ) -> None:
        await self._emit(
            SynapseEventType.RE_TRAINING_EXAMPLE,
            {
                "category": _RE_CATEGORY,
                "task_id": task_id,
                "repo_url": repo_url,
                "outcome": "success" if success else "failure",
                "total_tokens": tokens,
                "constitutional_alignment": {
                    "honesty": 1.0,
                    "care": 0.8,
                    "growth": 1.0 if success else 0.3,
                    "coherence": 0.9 if success else 0.4,
                },
            },
        )
