"""
EcodiaOS -- Simula Task Planner (Stage 5C.4)

CodePlan-style adaptive DAG decomposition for multi-file proposals.

Algorithm:
  1. Identify all files that need modification (from proposal + change spec)
  2. Build import dependency graph via AST analysis (zero LLM tokens)
  3. Create TaskNode per file/module with dependency edges
  4. Topological sort determines execution order
  5. Independent tasks (no shared dependencies) can run in parallel
  6. Max 2 agents per parallel stage (overcrowding constraint from 5C.2)

Output: TaskDAG consumed by MultiAgentOrchestrator.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from primitives.common import new_id, utc_now
from systems.simula.orchestration.types import (
    TaskDAG,
    TaskEdge,
    TaskNode,
    TaskStatus,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.evolution_types import EvolutionProposal

logger = structlog.get_logger().bind(system="simula.orchestration.planner")


class TaskPlanner:
    """CodePlan-style adaptive DAG decomposition for multi-file proposals."""

    def __init__(
        self,
        codebase_root: Path,
        llm: LLMProvider | None = None,
        *,
        max_dag_nodes: int = 50,
    ) -> None:
        self._root = codebase_root
        self._llm = llm
        self._max_dag_nodes = max_dag_nodes

    # ── Public API ──────────────────────────────────────────────────────────

    async def plan(
        self,
        proposal: EvolutionProposal,
        files_to_change: list[str],
    ) -> TaskDAG:
        """
        Build a TaskDAG for the given proposal and files.

        Args:
            proposal: The evolution proposal requiring multi-file changes.
            files_to_change: List of file paths (relative to codebase root).

        Returns:
            TaskDAG with nodes, edges, and topological execution order.
        """
        start = time.monotonic()

        # Step 1: Build import dependency graph
        import_graph = self._build_import_graph(files_to_change)

        # Step 2: Create task nodes
        nodes: list[TaskNode] = []
        node_map: dict[str, str] = {}  # file_path -> node_id

        for file_path in files_to_change[:self._max_dag_nodes]:
            node_id = f"task_{new_id()[:8]}"
            node_map[file_path] = node_id
            nodes.append(TaskNode(
                node_id=node_id,
                description=f"Implement changes in {file_path}",
                files_to_modify=[file_path],
                status=TaskStatus.PENDING,
            ))

        # Step 3: Create edges from import dependencies
        edges: list[TaskEdge] = []
        for file_path, deps in import_graph.items():
            if file_path not in node_map:
                continue
            to_node = node_map[file_path]
            for dep_path in deps:
                if dep_path in node_map:
                    from_node = node_map[dep_path]
                    if from_node != to_node:
                        edges.append(TaskEdge(
                            from_node=from_node,
                            to_node=to_node,
                            edge_type="depends_on",
                        ))

        # Step 4: Topological sort
        topo_order = self._topological_sort(nodes, edges)

        # Step 5: Count parallel stages
        parallel_stages = self._count_parallel_stages(nodes, edges)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        dag = TaskDAG(
            nodes=nodes,
            edges=edges,
            topological_order=topo_order,
            parallel_stages=parallel_stages,
            total_files=len(files_to_change),
            built_at=utc_now(),
        )

        logger.info(
            "task_dag_built",
            nodes=len(nodes),
            edges=len(edges),
            parallel_stages=parallel_stages,
            duration_ms=elapsed_ms,
        )

        return dag

    # ── Import graph construction ───────────────────────────────────────────

    def _build_import_graph(
        self, files: list[str]
    ) -> dict[str, list[str]]:
        """Build import dependency graph via AST analysis. Zero LLM tokens."""
        graph: dict[str, list[str]] = {f: [] for f in files}
        file_set = set(files)

        for file_path in files:
            full_path = self._root / file_path
            if not full_path.exists() or full_path.suffix != ".py":
                continue

            try:
                source = full_path.read_text()
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue

            for node in ast.walk(tree):
                imported_module = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module = alias.name

                if not imported_module:
                    continue

                # Convert module path to file path
                dep_path = self._module_to_file(imported_module)
                if dep_path and dep_path in file_set and dep_path != file_path:
                    graph[file_path].append(dep_path)

        return graph

    def _module_to_file(self, module: str) -> str:
        """Convert a Python module path to a relative file path."""
        # e.g. "systems.simula.evolution_types" → "systems/simula/evolution_types.py"
        parts = module.split(".")
        candidate = Path(*parts).with_suffix(".py")
        full = self._root / candidate
        if full.exists():
            return str(candidate)

        # Try as package __init__.py
        candidate_init = Path(*parts) / "__init__.py"
        full_init = self._root / candidate_init
        if full_init.exists():
            return str(candidate_init)

        return ""

    # ── Topological sort ────────────────────────────────────────────────────

    @staticmethod
    def _topological_sort(
        nodes: list[TaskNode], edges: list[TaskEdge]
    ) -> list[str]:
        """Kahn's algorithm for topological sort."""
        node_ids = {n.node_id for n in nodes}
        in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
        adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}

        for edge in edges:
            if edge.from_node in node_ids and edge.to_node in node_ids:
                adjacency[edge.from_node].append(edge.to_node)
                in_degree[edge.to_node] += 1

        # Queue starts with zero-in-degree nodes
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[str] = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If we didn't visit all nodes, there's a cycle — add remaining
        remaining = [nid for nid in node_ids if nid not in result]
        result.extend(sorted(remaining))

        return result

    @staticmethod
    def _count_parallel_stages(
        nodes: list[TaskNode], edges: list[TaskEdge]
    ) -> int:
        """Count how many stages can execute in parallel."""
        if not nodes:
            return 0

        node_ids = {n.node_id for n in nodes}
        in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
        adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}

        for edge in edges:
            if edge.from_node in node_ids and edge.to_node in node_ids:
                adjacency[edge.from_node].append(edge.to_node)
                in_degree[edge.to_node] += 1

        # BFS level-by-level = parallel stages
        stages = 0
        queue = [nid for nid, deg in in_degree.items() if deg == 0]

        while queue:
            stages += 1
            next_queue: list[str] = []
            for nid in queue:
                for neighbor in adjacency[nid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue

        return stages
