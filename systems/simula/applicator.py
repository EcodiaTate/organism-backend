"""
EcodiaOS - Simula Change Applicator

Routes approved evolution proposals to the appropriate application
strategy and coordinates with RollbackManager for safety.

Application strategies by category:

  ADJUST_BUDGET → direct config update (no code generation needed)
  ADD_EXECUTOR, ADD_INPUT_CHANNEL, ADD_PATTERN_DETECTOR → SimulaCodeAgent
  MODIFY_CONTRACT, ADD_SYSTEM_CAPABILITY, etc. → SimulaCodeAgent (post governance)

All strategies:
  1. Snapshot affected files via RollbackManager
  2. Apply change
  3. On failure → rollback immediately
  4. On success → return CodeChangeResult + snapshot (for caller health check)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yaml

from systems.simula.evolution_types import (
    ChangeCategory,
    CodeChangeResult,
    ConfigSnapshot,
    EvolutionProposal,
)

if TYPE_CHECKING:

    from systems.simula.agents.test_designer import TestDesignerAgent
    from systems.simula.agents.test_executor import TestExecutorAgent
    from systems.simula.analytics import EvolutionAnalyticsEngine
    from systems.simula.code_agent import SimulaCodeAgent
    from systems.simula.health import HealthChecker
    from systems.simula.rollback import RollbackManager
    from systems.simula.verification.static_analysis import StaticAnalysisBridge

logger = structlog.get_logger()


class ChangeApplicator:
    """
    Routes approved evolution proposals to the right application strategy.

    For code-level changes: delegates to SimulaCodeAgent.
    For budget changes: updates the YAML config directly.
    Always snapshots before applying, so rollback is always possible.
    """

    def __init__(
        self,
        code_agent: SimulaCodeAgent,
        rollback_manager: RollbackManager,
        health_checker: HealthChecker,
        codebase_root: Path,
        # Stage 2D: AgentCoder pipeline
        test_designer: TestDesignerAgent | None = None,
        test_executor: TestExecutorAgent | None = None,
        static_analysis_bridge: StaticAnalysisBridge | None = None,
        agent_coder_enabled: bool = False,
        agent_coder_max_iterations: int = 3,
        analytics: EvolutionAnalyticsEngine | None = None,
    ) -> None:
        self._agent = code_agent
        self._rollback = rollback_manager
        self._health = health_checker
        self._root = codebase_root
        self._test_designer = test_designer
        self._test_executor = test_executor
        self._static_bridge = static_analysis_bridge
        self._agent_coder_enabled = agent_coder_enabled
        self._agent_coder_max_iterations = agent_coder_max_iterations
        self._analytics = analytics
        self._logger = logger.bind(system="simula.applicator")

    async def apply(
        self, proposal: EvolutionProposal
    ) -> tuple[CodeChangeResult, ConfigSnapshot]:
        """
        Apply an evolution proposal. Returns (result, snapshot).

        The snapshot is needed by SimulaService for rollback if the
        post-application health check fails.

        Routes through the AgentCoder 3-agent pipeline when enabled,
        otherwise uses the standard code agent.
        """
        self._logger.info(
            "applying_change",
            proposal_id=proposal.id,
            category=proposal.category.value,
        )

        if proposal.category == ChangeCategory.ADJUST_BUDGET:
            return await self._apply_budget(proposal)
        use_agent_coder = await self._select_strategy(proposal)
        if use_agent_coder:
            return await self._apply_via_agent_coder(proposal)
        else:
            return await self._apply_via_code_agent(proposal)

    # ── Strategy Selection ────────────────────────────────────────────────────

    async def _select_strategy(self, proposal: EvolutionProposal) -> bool:
        """
        Decide whether to use the AgentCoder 3-agent pipeline or the standard
        code agent for this proposal.

        Decision tree (consulted in order):
          1. If agent_coder is not configured (no test_designer/executor), always False.
          2. If agent_coder is disabled in config, check analytics override (below).
          3. If analytics show the AgentCoder pipeline has a higher category success rate
             than code_agent by a margin >= 0.05, prefer agent_coder.
          4. Fall back to the config flag.

        The analytics engine tracks per-strategy outcomes via the proposal's
        `agent_coder_iterations` field (>0 → AgentCoder was used).  Categories
        with insufficient history (< 5 samples) use the config flag.
        """
        if not (self._test_designer and self._test_executor):
            return False

        if self._analytics is None:
            return self._agent_coder_enabled

        try:
            category = proposal.category

            # Query per-strategy success rates from analytics
            analytics = await self._analytics.compute_analytics()
            rate = analytics.category_rates.get(category.value)
            if rate is None or rate.total < 5:
                return self._agent_coder_enabled

            # agent_coder_success_rate is tracked separately when available
            agent_coder_rate = analytics.category_rates.get(
                f"{category.value}:agent_coder"
            )
            code_agent_rate = analytics.category_rates.get(
                f"{category.value}:code_agent"
            )

            if agent_coder_rate and code_agent_rate and agent_coder_rate.total >= 5:
                margin = agent_coder_rate.success_rate - code_agent_rate.success_rate
                use_agent_coder = margin >= 0.05
                self._logger.info(
                    "strategy_selected_by_analytics",
                    category=category.value,
                    agent_coder_rate=round(agent_coder_rate.success_rate, 3),
                    code_agent_rate=round(code_agent_rate.success_rate, 3),
                    margin=round(margin, 3),
                    use_agent_coder=use_agent_coder,
                )
                return use_agent_coder
        except Exception as exc:
            self._logger.warning("strategy_analytics_error", error=str(exc))

        return self._agent_coder_enabled

    # ── Budget Adjustment (direct config update) ──────────────────────────────

    async def _apply_budget(
        self, proposal: EvolutionProposal
    ) -> tuple[CodeChangeResult, ConfigSnapshot]:
        """Direct config update for budget changes - no code generation."""
        spec = proposal.change_spec
        if not spec.budget_parameter or spec.budget_new_value is None:
            result = CodeChangeResult(
                success=False,
                error="Budget change spec missing parameter or new_value",
            )
            return result, ConfigSnapshot(
                proposal_id=proposal.id,
                config_version=0,
            )

        config_path = self._root / "config" / "default.yaml"
        snapshot = await self._rollback.snapshot(
            proposal_id=proposal.id,
            paths=[config_path],
        )

        try:
            data: dict[str, Any] = {}
            if config_path.exists():
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}

            # Navigate the dotted parameter path (e.g. "nova.efe.pragmatic")
            parts = spec.budget_parameter.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = spec.budget_new_value

            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

            rel_path = str(config_path.relative_to(self._root))
            self._logger.info(
                "budget_updated",
                parameter=spec.budget_parameter,
                old_value=spec.budget_old_value,
                new_value=spec.budget_new_value,
            )
            return CodeChangeResult(
                success=True,
                files_written=[rel_path],
                summary=(
                    f"Updated {spec.budget_parameter} "
                    f"from {spec.budget_old_value} to {spec.budget_new_value}"
                ),
            ), snapshot

        except Exception as exc:
            await self._rollback.restore(snapshot)
            return CodeChangeResult(
                success=False,
                error=f"Budget update failed: {exc}",
            ), snapshot

    # ── Code Agent Application ────────────────────────────────────────────────

    async def _apply_via_code_agent(
        self, proposal: EvolutionProposal
    ) -> tuple[CodeChangeResult, ConfigSnapshot]:
        """Use SimulaCodeAgent to generate and write the implementation."""
        # For bounty proposals with an external workspace, redirect the code
        # agent to the cloned target repo so it never touches the organism's
        # own source tree.
        external_root = Path(proposal.workspace_root) if proposal.workspace_root else None
        effective_root = external_root or self._root
        _prev_agent_root = None
        if external_root is not None:
            _prev_agent_root = self._agent._root
            self._agent._root = external_root

        try:
            affected_dirs = _infer_affected_paths(proposal, effective_root)
            snapshot = await self._rollback.snapshot(
                proposal_id=proposal.id,
                paths=affected_dirs,
            )

            result = await self._agent.implement(proposal)
        finally:
            if external_root is not None and _prev_agent_root is not None:
                self._agent._root = _prev_agent_root

        if not result.success:
            self._logger.warning(
                "code_agent_failed",
                proposal_id=proposal.id,
                error=result.error,
            )
            await self._rollback.restore(snapshot)
            return result, snapshot

        # For bounty proposals targeting an external repo, push and open a PR.
        if proposal.source == "bounty" and proposal.target_repository_url:
            try:
                pr_url, pr_number = await self._agent._push_to_github(
                    proposal, result.files_written
                )
                result.pr_url = pr_url
                result.pr_number = pr_number
            except Exception as exc:
                self._logger.warning(
                    "push_to_github_unhandled",
                    proposal_id=proposal.id,
                    error=str(exc),
                )

        return result, snapshot

    # ── Stage 2D: AgentCoder 3-Agent Pipeline ─────────────────────────────────

    async def _apply_via_agent_coder(
        self, proposal: EvolutionProposal,
    ) -> tuple[CodeChangeResult, ConfigSnapshot]:
        """
        Apply via the AgentCoder pipeline:
          1. TestDesigner generates tests from proposal spec (no code seen)
          2. CodeAgent implements the change (with test writing disabled)
          3. TestExecutor runs the designed tests against the implementation
          4. If failures → feed back to CodeAgent → iterate

        This adversarial separation produces higher-quality code by testing
        the specification rather than the implementation.
        """
        from systems.simula.agents.test_executor import TestExecutorAgent
        from systems.simula.verification.types import (
            AgentCoderIterationResult,
        )

        assert self._test_designer is not None
        assert self._test_executor is not None

        affected_dirs = _infer_affected_paths(proposal, self._root)
        snapshot = await self._rollback.snapshot(
            proposal_id=proposal.id,
            paths=affected_dirs,
        )

        log = self._logger.bind(
            proposal_id=proposal.id,
            pipeline="agent_coder",
        )

        # Step 1: TestDesigner generates tests
        log.info("agent_coder_designing_tests")
        test_design = await self._test_designer.design_tests(proposal)

        if not test_design.test_files:
            log.warning("agent_coder_no_tests_designed")
            # Fall back to standard code agent
            return await self._apply_via_code_agent(proposal)

        log.info(
            "agent_coder_tests_designed",
            test_files=len(test_design.test_files),
            test_count=test_design.test_count,
        )

        iterations: list[AgentCoderIterationResult] = []
        final_result: CodeChangeResult | None = None

        for iteration_num in range(1, self._agent_coder_max_iterations + 1):
            log.info("agent_coder_iteration", iteration=iteration_num)

            # Step 2: CodeAgent implements (test writing disabled)
            code_result = await self._agent.implement(
                proposal, skip_test_writing=True,
            )
            final_result = code_result

            if not code_result.success:
                log.warning(
                    "agent_coder_code_failed",
                    iteration=iteration_num,
                    error=code_result.error,
                )
                iterations.append(AgentCoderIterationResult(
                    iteration=iteration_num,
                    test_design=test_design if iteration_num == 1 else None,
                    code_generation_success=False,
                    code_generation_files=code_result.files_written,
                ))
                break

            # Step 3: TestExecutor runs tests
            test_result = await self._test_executor.execute_tests(
                test_design.test_files,
            )

            iter_result = AgentCoderIterationResult(
                iteration=iteration_num,
                test_design=test_design if iteration_num == 1 else None,
                code_generation_success=True,
                code_generation_files=code_result.files_written,
                test_execution=test_result,
                all_tests_passed=(
                    test_result.failed == 0 and test_result.errors == 0
                ),
            )
            iterations.append(iter_result)

            log.info(
                "agent_coder_test_results",
                iteration=iteration_num,
                passed=test_result.passed,
                failed=test_result.failed,
                errors=test_result.errors,
            )

            if iter_result.all_tests_passed:
                log.info("agent_coder_converged", iterations=iteration_num)
                break

            # Step 4: Not all tests passed - feed failures back
            if iteration_num < self._agent_coder_max_iterations:
                feedback = TestExecutorAgent.format_failures_for_feedback(
                    test_result,
                )
                log.info("agent_coder_feeding_back", feedback_len=len(feedback))
                # Attach feedback to proposal for next iteration
                proposal._agent_coder_feedback = feedback  # type: ignore[attr-defined]

        # Compute aggregate result
        bool(iterations and iterations[-1].all_tests_passed)
        if iterations and iterations[-1].test_execution:
            te = iterations[-1].test_execution
            if te.total > 0:
                te.passed / te.total

        # Attach to code result for downstream recording
        if final_result is not None:
            final_result.agent_coder_iterations = len(iterations)
            final_result.test_designer_test_count = test_design.test_count

        if final_result is None or not final_result.success:
            await self._rollback.restore(snapshot)
        elif proposal.source == "bounty" and proposal.target_repository_url:
            # Bounty proposals: push to fork and open a cross-repo PR
            try:
                pr_url, pr_number = await self._agent._push_to_github(
                    proposal, final_result.files_written
                )
                final_result.pr_url = pr_url
                final_result.pr_number = pr_number
            except Exception as exc:
                self._logger.warning(
                    "push_to_github_unhandled",
                    proposal_id=proposal.id,
                    error=str(exc),
                )

        return final_result or CodeChangeResult(
            success=False, error="AgentCoder pipeline produced no result",
        ), snapshot


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _infer_affected_paths(proposal: EvolutionProposal, root: Path) -> list[Path]:
    """Infer which existing paths will likely be affected by this change."""
    paths: list[Path] = []
    category = proposal.category
    spec = proposal.change_spec

    if category == ChangeCategory.ADD_EXECUTOR:
        paths.append(root / "src" / "ecodiaos" / "systems" / "axon" / "registry.py")
        executors_dir = root / "src" / "ecodiaos" / "systems" / "axon" / "executors"
        if executors_dir.exists():
            paths.append(executors_dir)
    elif category == ChangeCategory.ADD_INPUT_CHANNEL:
        paths.append(root / "src" / "ecodiaos" / "systems" / "atune")
    elif category == ChangeCategory.ADD_PATTERN_DETECTOR:
        paths.append(root / "src" / "ecodiaos" / "systems" / "evo" / "detectors.py")
    elif category in {
        ChangeCategory.MODIFY_CONTRACT,
        ChangeCategory.ADD_SYSTEM_CAPABILITY,
    }:
        for sys_name in (spec.affected_systems or []):
            sys_path = root / "src" / "ecodiaos" / "systems" / sys_name
            if sys_path.exists():
                paths.append(sys_path)

    return [p for p in paths if p.exists()]
