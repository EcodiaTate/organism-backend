"""
EcodiaOS -- Simula Multi-Agent Orchestrator (Stage 5C.1 + 5C.3)

MetaGPT-style structured artifact pipeline for multi-file proposals:

  SPEC → DESIGN → CODE → TEST → REVIEW

Each stage produces typed PipelineArtifacts (not free-form chat).
For proposals touching >= multi_file_threshold files: decompose via
TaskPlanner, delegate sub-tasks to agents in parallel.

Constraints:
  - Max 2 agents per parallel stage (5C.2 overcrowding limit)
  - Hierarchical delegation: orchestrator coordinates, sub-agents execute
  - Falls back to single-agent mode for simpler proposals

Integration: replaces direct applicator.apply() calls in service.py
for multi-file proposals.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from pathlib import Path

import structlog

from systems.simula.orchestration.types import (
    ArtifactKind,
    DelegationMode,
    OrchestratorResult,
    PipelineArtifact,
    StageResult,
    TaskDAG,
    TaskNode,
    TaskStatus,
)

if TYPE_CHECKING:

    from clients.llm import LLMProvider
    from systems.simula.agents.test_designer import TestDesignerAgent
    from systems.simula.agents.test_executor import TestExecutorAgent
    from systems.simula.code_agent import SimulaCodeAgent
    from systems.simula.evolution_types import EvolutionProposal
    from systems.simula.orchestration.task_planner import TaskPlanner
logger = structlog.get_logger().bind(system="simula.orchestration")

# Max agents per parallel stage (overcrowding constraint 5C.2)
_MAX_AGENTS_PER_STAGE = 2


class MultiAgentOrchestrator:
    """
    MetaGPT-style multi-agent orchestration for complex proposals.

    Pipeline: SPEC → DESIGN → CODE → TEST → REVIEW
    Each stage produces typed artifacts consumed by the next.
    Multi-file proposals are decomposed via TaskPlanner into parallel DAGs.
    """

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        code_agent: SimulaCodeAgent,
        test_designer: TestDesignerAgent | None = None,
        test_executor: TestExecutorAgent | None = None,
        task_planner: TaskPlanner | None = None,
        *,
        max_agents_per_stage: int = _MAX_AGENTS_PER_STAGE,
        timeout_s: float = 300.0,
    ) -> None:
        self._llm = llm
        self._root = codebase_root
        self._code_agent = code_agent
        self._test_designer = test_designer
        self._test_executor = test_executor
        self._task_planner = task_planner
        self._max_agents = max_agents_per_stage
        self._timeout_s = timeout_s

    # ── Public API ──────────────────────────────────────────────────────────

    async def orchestrate(
        self,
        proposal: EvolutionProposal,
        files_to_change: list[str],
    ) -> OrchestratorResult:
        """
        Run the full MetaGPT pipeline for a proposal.

        For multi-file proposals: decompose → parallel DAG execution.
        For single-file: direct code agent call (no overhead).

        Args:
            proposal: Evolution proposal to implement.
            files_to_change: Files identified for modification.

        Returns:
            OrchestratorResult with stage results and metrics.
        """
        start = time.monotonic()
        stage_results: list[StageResult] = []

        try:
            # Build DAG if planner is available and multi-file
            dag: TaskDAG | None = None
            if self._task_planner and len(files_to_change) > 1:
                dag = await self._task_planner.plan(proposal, files_to_change)

            # Stage 1: SPEC - Generate specification artifact
            spec_result = await self._stage_spec(proposal, files_to_change)
            stage_results.append(spec_result)

            if spec_result.status == TaskStatus.FAILED:
                return self._build_result(
                    stage_results, dag, start, error="Spec generation failed"
                )

            # Stage 2: DESIGN - Generate design artifact
            design_result = await self._stage_design(proposal, spec_result, files_to_change)
            stage_results.append(design_result)

            if design_result.status == TaskStatus.FAILED:
                return self._build_result(
                    stage_results, dag, start, error="Design generation failed"
                )

            # Stage 3: CODE - Generate code (parallel sub-tasks if multi-file)
            code_result = await self._stage_code(proposal, dag, design_result)
            stage_results.append(code_result)

            if code_result.status == TaskStatus.FAILED:
                return self._build_result(
                    stage_results, dag, start, error="Code generation failed"
                )

            # Stage 4: TEST - Generate and run tests
            test_result = await self._stage_test(proposal, code_result)
            stage_results.append(test_result)

            # Stage 5: REVIEW - Self-review of generated code
            review_result = await self._stage_review(proposal, code_result, test_result)
            stage_results.append(review_result)

            return self._build_result(stage_results, dag, start)

        except TimeoutError:
            logger.warning("orchestration_timeout", timeout_s=self._timeout_s)
            return self._build_result(
                stage_results, None, start, error="Orchestration timeout"
            )
        except Exception:
            logger.exception("orchestration_error")
            return self._build_result(
                stage_results, None, start, error="Orchestration error"
            )

    # ── Stage 1: SPEC ───────────────────────────────────────────────────────

    async def _stage_spec(
        self,
        proposal: EvolutionProposal,
        files_to_change: list[str],
    ) -> StageResult:
        """Generate a specification artifact from the proposal."""
        start = time.monotonic()

        from clients.llm import Message

        spec_prompt = (
            f"Generate a precise technical specification for:\n"
            f"  Category: {proposal.category.value}\n"
            f"  Description: {proposal.description}\n"
            f"  Files to change: {', '.join(files_to_change)}\n"
            f"  Expected benefit: {proposal.expected_benefit}\n\n"
            f"Output a structured spec with: objectives, constraints, "
            f"interfaces to implement, and acceptance criteria."
        )

        response = await self._llm.complete(  # type: ignore[attr-defined]
            system="You are a technical specification writer for EcodiaOS.",
            messages=[Message(role="user", content=spec_prompt)],
            max_tokens=2048,
        )

        tokens = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)
        artifact = PipelineArtifact(
            kind=ArtifactKind.SPEC,
            stage_index=0,
            content=response.text,
            files_referenced=files_to_change,
            produced_by="orchestrator.spec",
            tokens_used=tokens,
        )

        return StageResult(
            stage=ArtifactKind.SPEC,
            status=TaskStatus.COMPLETED,
            agents_used=1,
            delegation_mode=DelegationMode.SINGLE_AGENT,
            artifacts=[artifact],
            duration_ms=int((time.monotonic() - start) * 1000),
            tokens_used=tokens,
        )

    # ── Stage 2: DESIGN ────────────────────────────────────────────────────

    async def _stage_design(
        self,
        proposal: EvolutionProposal,
        spec_result: StageResult,
        files_to_change: list[str],
    ) -> StageResult:
        """Generate a design artifact from the spec."""
        start = time.monotonic()

        from clients.llm import Message

        spec_content = (
            spec_result.artifacts[0].content if spec_result.artifacts else ""
        )

        design_prompt = (
            f"Based on this specification, design the implementation:\n\n"
            f"## Spec\n{spec_content[:4000]}\n\n"
            f"## Files to modify\n{', '.join(files_to_change)}\n\n"
            f"Output: class/function signatures, data flow, "
            f"error handling strategy, and integration points."
        )

        response = await self._llm.complete(  # type: ignore[attr-defined]
            system="You are a software architect for EcodiaOS.",
            messages=[Message(role="user", content=design_prompt)],
            max_tokens=3072,
        )

        tokens = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)
        artifact = PipelineArtifact(
            kind=ArtifactKind.DESIGN,
            stage_index=1,
            content=response.text,
            files_referenced=files_to_change,
            produced_by="orchestrator.design",
            tokens_used=tokens,
        )

        return StageResult(
            stage=ArtifactKind.DESIGN,
            status=TaskStatus.COMPLETED,
            agents_used=1,
            delegation_mode=DelegationMode.SINGLE_AGENT,
            artifacts=[artifact],
            duration_ms=int((time.monotonic() - start) * 1000),
            tokens_used=tokens,
        )

    # ── Stage 3: CODE ──────────────────────────────────────────────────────

    async def _stage_code(
        self,
        proposal: EvolutionProposal,
        dag: TaskDAG | None,
        design_result: StageResult,
    ) -> StageResult:
        """Generate code - parallel sub-tasks for multi-file DAGs."""
        start = time.monotonic()

        if dag and len(dag.nodes) > 1:
            return await self._parallel_code_generation(proposal, dag, design_result, start)
        else:
            return await self._single_agent_code(proposal, design_result, start)

    async def _single_agent_code(
        self,
        proposal: EvolutionProposal,
        design_result: StageResult,
        start: float,
    ) -> StageResult:
        """Direct code generation via the existing SimulaCodeAgent."""
        code_result = await self._code_agent.implement(proposal)

        artifact = PipelineArtifact(
            kind=ArtifactKind.CODE,
            stage_index=2,
            content=code_result.summary,
            files_referenced=code_result.files_written,
            produced_by="code_agent",
            tokens_used=code_result.reasoning_tokens,
        )

        status = TaskStatus.COMPLETED if code_result.success else TaskStatus.FAILED

        return StageResult(
            stage=ArtifactKind.CODE,
            status=status,
            agents_used=1,
            delegation_mode=DelegationMode.SINGLE_AGENT,
            artifacts=[artifact],
            duration_ms=int((time.monotonic() - start) * 1000),
            tokens_used=code_result.reasoning_tokens,
            error="" if code_result.success else code_result.error,
        )

    async def _parallel_code_generation(
        self,
        proposal: EvolutionProposal,
        dag: TaskDAG,
        design_result: StageResult,
        start: float,
    ) -> StageResult:
        """Execute DAG nodes in topological order with parallelism."""
        artifacts: list[PipelineArtifact] = []
        total_tokens = 0
        agents_used = 0
        all_files: list[str] = []
        any_failed = False

        # Execute by topological order, grouping independent tasks
        executed: set[str] = set()
        remaining = list(dag.topological_order)

        while remaining:
            # Find tasks whose dependencies are all executed
            ready: list[str] = []
            for nid in remaining:
                node = next((n for n in dag.nodes if n.node_id == nid), None)
                if node is None:
                    continue
                deps_met = all(
                    e.from_node in executed
                    for e in dag.edges
                    if e.to_node == nid
                )
                if deps_met:
                    ready.append(nid)
                if len(ready) >= self._max_agents:
                    break

            if not ready:
                # Break cycle - just execute next in order
                ready = [remaining[0]]

            # Execute ready tasks in parallel (up to max_agents)
            batch = ready[:self._max_agents]
            tasks = []
            for nid in batch:
                node = next((n for n in dag.nodes if n.node_id == nid), None)
                if node is None:
                    continue
                node.status = TaskStatus.RUNNING
                tasks.append(self._execute_dag_node(proposal, node))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for nid, result in zip(batch, results, strict=False):
                node = next((n for n in dag.nodes if n.node_id == nid), None)
                if node is None:
                    continue

                executed.add(nid)
                remaining = [r for r in remaining if r != nid]
                agents_used += 1

                if isinstance(result, Exception):
                    node.status = TaskStatus.FAILED
                    node.error = str(result)
                    any_failed = True
                elif isinstance(result, PipelineArtifact):
                    node.status = TaskStatus.COMPLETED
                    artifacts.append(result)
                    total_tokens += result.tokens_used
                    all_files.extend(result.files_referenced)

        return StageResult(
            stage=ArtifactKind.CODE,
            status=TaskStatus.FAILED if any_failed else TaskStatus.COMPLETED,
            agents_used=agents_used,
            delegation_mode=DelegationMode.HIERARCHICAL,
            artifacts=artifacts,
            duration_ms=int((time.monotonic() - start) * 1000),
            tokens_used=total_tokens,
        )

    async def _execute_dag_node(
        self,
        proposal: EvolutionProposal,
        node: TaskNode,
    ) -> PipelineArtifact:
        """Execute a single DAG node using the code agent."""

        # Create a focused sub-proposal for just this node's files
        code_result = await self._code_agent.implement(proposal)

        return PipelineArtifact(
            kind=ArtifactKind.CODE,
            stage_index=2,
            content=code_result.summary,
            files_referenced=code_result.files_written,
            produced_by=f"code_agent:{node.node_id}",
            tokens_used=code_result.reasoning_tokens,
        )

    # ── Stage 4: TEST ──────────────────────────────────────────────────────

    async def _stage_test(
        self,
        proposal: EvolutionProposal,
        code_result: StageResult,
    ) -> StageResult:
        """Generate and run tests for the implemented code."""
        start = time.monotonic()

        # Collect all files from code artifacts
        all_files = []
        for artifact in code_result.artifacts:
            all_files.extend(artifact.files_referenced)

        artifacts: list[PipelineArtifact] = []
        total_tokens = 0

        # Use test designer if available
        if self._test_designer:
            for file_path in all_files[:5]:
                try:
                    test_result = await self._test_designer.design_tests(  # type: ignore[call-arg]
                        file_path=file_path,
                        context=proposal.description,
                    )
                    artifacts.append(PipelineArtifact(
                        kind=ArtifactKind.TEST,
                        stage_index=3,
                        content=test_result.test_code if hasattr(test_result, "test_code") else str(test_result),
                        files_referenced=[file_path],
                        produced_by="test_designer",
                    ))
                except Exception:
                    logger.debug("test_design_failed", file=file_path)

        # Run tests via test executor if available
        if self._test_executor and all_files:
            try:
                exec_result = await self._test_executor.execute_tests(  # type: ignore[call-arg]
                    test_paths=all_files,
                )
                artifacts.append(PipelineArtifact(
                    kind=ArtifactKind.TEST,
                    stage_index=3,
                    content=str(exec_result),
                    produced_by="test_executor",
                ))
            except Exception:
                logger.debug("test_execution_failed")

        return StageResult(
            stage=ArtifactKind.TEST,
            status=TaskStatus.COMPLETED,
            agents_used=min(2, (1 if self._test_designer else 0) + (1 if self._test_executor else 0)),
            delegation_mode=DelegationMode.DUAL_AGENT if self._test_designer and self._test_executor else DelegationMode.SINGLE_AGENT,
            artifacts=artifacts,
            duration_ms=int((time.monotonic() - start) * 1000),
            tokens_used=total_tokens,
        )

    # ── Stage 5: REVIEW ────────────────────────────────────────────────────

    async def _stage_review(
        self,
        proposal: EvolutionProposal,
        code_result: StageResult,
        test_result: StageResult,
    ) -> StageResult:
        """Self-review of generated code for quality and correctness."""
        start = time.monotonic()

        from clients.llm import Message

        # Collect code artifacts for review
        code_content = "\n\n".join(
            a.content for a in code_result.artifacts if a.content
        )
        test_content = "\n\n".join(
            a.content for a in test_result.artifacts if a.content
        )

        review_prompt = (
            f"Review this implementation for the EcodiaOS proposal:\n"
            f"  Description: {proposal.description}\n"
            f"  Category: {proposal.category.value}\n\n"
            f"## Code\n{code_content[:4000]}\n\n"
            f"## Tests\n{test_content[:2000]}\n\n"
            f"Check for: correctness, EOS conventions, type safety, "
            f"edge cases, and potential regressions. List any issues found."
        )

        response = await self._llm.complete(  # type: ignore[attr-defined]
            system="You are a code reviewer for EcodiaOS. Be thorough but concise.",
            messages=[Message(role="user", content=review_prompt)],
            max_tokens=1024,
        )

        tokens = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)
        artifact = PipelineArtifact(
            kind=ArtifactKind.REVIEW,
            stage_index=4,
            content=response.text,
            produced_by="orchestrator.review",
            tokens_used=tokens,
        )

        return StageResult(
            stage=ArtifactKind.REVIEW,
            status=TaskStatus.COMPLETED,
            agents_used=1,
            delegation_mode=DelegationMode.SINGLE_AGENT,
            artifacts=[artifact],
            duration_ms=int((time.monotonic() - start) * 1000),
            tokens_used=tokens,
        )

    # ── Result building ─────────────────────────────────────────────────────

    def _build_result(
        self,
        stage_results: list[StageResult],
        dag: TaskDAG | None,
        start: float,
        *,
        error: str = "",
    ) -> OrchestratorResult:
        """Build aggregate OrchestratorResult."""
        total_agents = sum(sr.agents_used for sr in stage_results)
        total_tokens = sum(sr.tokens_used for sr in stage_results)
        parallel_stages = dag.parallel_stages if dag else 0

        all_files: list[str] = []
        for sr in stage_results:
            for artifact in sr.artifacts:
                all_files.extend(artifact.files_referenced)

        # Deduplicate files
        seen: set[str] = set()
        unique_files: list[str] = []
        for f in all_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        any_failed = any(sr.status == TaskStatus.FAILED for sr in stage_results)

        return OrchestratorResult(
            used=True,
            dag=dag,
            stage_results=stage_results,
            total_agents_used=total_agents,
            parallel_stages_executed=parallel_stages,
            files_modified=unique_files,
            total_tokens=total_tokens,
            total_duration_ms=int((time.monotonic() - start) * 1000),
            fell_back_to_single_agent=dag is None,
            error=error if error else ("Stage failure" if any_failed else ""),
        )
