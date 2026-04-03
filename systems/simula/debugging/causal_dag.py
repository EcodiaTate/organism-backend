"""
EcodiaOS -- Simula Causal Debugger (Stage 5D.1 + 5D.2 + 5D.3 + 5D.4)

When a health check fails after applying a change, the CausalDebugger
narrows down the root cause via:

  1. Build DAG: Parse AST for call graph + import graph. Nodes = functions/classes.
     Edges = calls/imports/inherits/tests/mutates.
  2. Suspicion scoring: Mark modified + failing nodes, propagate suspicion
     scores through the DAG based on proximity to modified code.
  3. AID pattern: For each candidate root cause node, simulate intervention
     ("if this function were correct, would the test pass?") using LLM
     interventional reasoning.
  4. Fault injection (optional, staging only): Modify suspected code,
     re-run tests, observe if failure changes.

Output: CausalDiagnosis with root_cause, confidence, fix_location,
and step-by-step reasoning chain.

Target: 97.72% root-cause precision (AID benchmark).
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import Message
from primitives.common import new_id
from systems.simula.debugging.types import (
    CausalDAG,
    CausalDiagnosis,
    CausalEdge,
    CausalEdgeKind,
    CausalNode,
    CausalNodeKind,
    InterventionResult,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.debugging.causal")

# ── System prompts ──────────────────────────────────────────────────────────

INTERVENTION_PROMPT = """Causal intervention analysis.

Test failure: {test_output}
Suspected function: {function_name} in {file_path}
Function code:
```python
{function_code}
```

Changed code: {diff_summary}

If {function_name} were implemented correctly (matching the original intent), would the failing test pass?

Respond as JSON:
{{
  "outcome_changed": true,
  "reasoning": "...",
  "confidence": 0.85
}}"""


class CausalDebugger:
    """
    Causal debugging engine using DAG analysis + AID interventional reasoning.

    Sits between health check failure and repair/rollback in service.py.
    Provides precise root-cause diagnosis that feeds into the repair agent.
    """

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        *,
        max_interventions: int = 5,
        fault_injection_enabled: bool = False,
        timeout_s: float = 60.0,
    ) -> None:
        self._llm = llm
        self._root = codebase_root
        self._max_interventions = max_interventions
        self._fault_injection = fault_injection_enabled
        self._timeout_s = timeout_s

    # ── Public API ──────────────────────────────────────────────────────────

    async def diagnose(
        self,
        files_written: list[str],
        health_issues: list[str],
        test_output: str = "",
    ) -> CausalDiagnosis:
        """
        Full causal diagnosis pipeline:
        Build DAG → Score suspicion → Interventional queries → Diagnosis.

        Args:
            files_written: Files modified by the applied change.
            health_issues: Issues reported by the health checker.
            test_output: Raw test output (pytest, lint, type check).

        Returns:
            CausalDiagnosis with root cause and supporting evidence.
        """
        start = time.monotonic()
        total_tokens = 0

        try:
            # Phase 1: Build causal DAG from AST
            dag = self._build_dag(files_written)

            # Phase 2: Score suspicion based on modified/failing nodes
            self._score_suspicion(dag, files_written, test_output)

            # Phase 3: AID interventional reasoning on top suspects
            suspects = self._get_top_suspects(dag)
            interventions: list[InterventionResult] = []

            for suspect in suspects[:self._max_interventions]:
                if time.monotonic() - start > self._timeout_s:
                    break

                result, tokens = await self._interventional_query(
                    suspect, dag, test_output, files_written
                )
                interventions.append(result)
                total_tokens += tokens

            # Phase 4: Determine root cause from interventions
            root_cause = self._determine_root_cause(suspects, interventions, dag)

            elapsed_ms = int((time.monotonic() - start) * 1000)

            logger.info(
                "causal_diagnosis_complete",
                root_cause=root_cause.get("node_id", "unknown"),
                confidence=root_cause.get("confidence", 0.0),
                interventions=len(interventions),
                dag_nodes=len(dag.nodes),
                duration_ms=elapsed_ms,
            )

            return CausalDiagnosis(
                dag=dag,
                interventions=interventions,
                root_cause_node=root_cause.get("node_id", ""),
                root_cause_file=root_cause.get("file_path", ""),
                root_cause_function=root_cause.get("function_name", ""),
                root_cause_description=root_cause.get("description", ""),
                confidence=root_cause.get("confidence", 0.0),
                reasoning_chain=root_cause.get("reasoning_chain", []),
                alternative_causes=root_cause.get("alternatives", []),
                total_interventions=len(interventions),
                total_tokens=total_tokens,
                total_duration_ms=elapsed_ms,
                fault_injection_used=False,
            )

        except Exception:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.exception("causal_diagnosis_error")
            return CausalDiagnosis(
                total_duration_ms=elapsed_ms,
                root_cause_description="Diagnosis failed due to internal error",
            )

    # ── Phase 1: Build DAG ──────────────────────────────────────────────────

    def _build_dag(self, files_written: list[str]) -> CausalDAG:
        """Build a causal DAG from AST analysis of modified files + their dependencies."""
        nodes: list[CausalNode] = []
        edges: list[CausalEdge] = []
        node_ids: dict[str, str] = {}  # "file:name" -> node_id

        modified_files = set(files_written)

        # Analyse each modified file + its direct importers
        files_to_analyse = list(files_written)
        # Add direct importers of modified files
        for f in files_written:
            importers = self._find_importers(f)
            files_to_analyse.extend(importers)

        seen_files: set[str] = set()
        for file_path in files_to_analyse:
            if file_path in seen_files:
                continue
            seen_files.add(file_path)

            full_path = self._root / file_path
            if not full_path.exists() or full_path.suffix != ".py":
                continue

            try:
                source = full_path.read_text()
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue

            is_modified = file_path in modified_files

            # Extract function and class nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    key = f"{file_path}:{node.name}"
                    nid = f"n_{new_id()[:8]}"
                    node_ids[key] = nid
                    nodes.append(CausalNode(
                        node_id=nid,
                        kind=CausalNodeKind.FUNCTION,
                        name=node.name,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno + 5,
                        is_modified=is_modified,
                    ))
                elif isinstance(node, ast.ClassDef):
                    key = f"{file_path}:{node.name}"
                    nid = f"n_{new_id()[:8]}"
                    node_ids[key] = nid
                    nodes.append(CausalNode(
                        node_id=nid,
                        kind=CausalNodeKind.CLASS,
                        name=node.name,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno + 10,
                        is_modified=is_modified,
                    ))

            # Extract edges: calls, imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    caller = self._enclosing_function(tree, node.lineno)
                    callee = self._call_target(node)
                    if caller and callee:
                        from_key = f"{file_path}:{caller}"
                        to_key = f"{file_path}:{callee}"
                        if from_key in node_ids and to_key in node_ids:
                            edges.append(CausalEdge(
                                from_node=node_ids[from_key],
                                to_node=node_ids[to_key],
                                kind=CausalEdgeKind.CALLS,
                            ))

                elif isinstance(node, ast.ImportFrom) and node.module:
                    # Create import edges
                    for alias in node.names:
                        imported_name = alias.name
                        dep_file = self._module_to_file(node.module)
                        if dep_file:
                            from_key = f"{file_path}:_module"
                            to_key = f"{dep_file}:{imported_name}"
                            if to_key in node_ids:
                                # Create a module-level node if needed
                                if from_key not in node_ids:
                                    mod_nid = f"n_{new_id()[:8]}"
                                    node_ids[from_key] = mod_nid
                                    nodes.append(CausalNode(
                                        node_id=mod_nid,
                                        kind=CausalNodeKind.MODULE,
                                        name=file_path,
                                        file_path=file_path,
                                        is_modified=is_modified,
                                    ))
                                edges.append(CausalEdge(
                                    from_node=node_ids[from_key],
                                    to_node=node_ids[to_key],
                                    kind=CausalEdgeKind.IMPORTS,
                                ))

        return CausalDAG(
            nodes=nodes,
            edges=edges,
            modified_nodes=[n.node_id for n in nodes if n.is_modified],
            failing_nodes=[],  # populated in scoring phase
            total_functions=len([n for n in nodes if n.kind == CausalNodeKind.FUNCTION]),
            total_edges=len(edges),
        )

    def _find_importers(self, file_path: str) -> list[str]:
        """Find Python files that import from the given file."""
        importers: list[str] = []
        module_name = file_path.replace("/", ".").replace("\\", ".").rstrip(".py")

        # Scan nearby files for imports
        file_dir = (self._root / file_path).parent
        if not file_dir.exists():
            return importers

        for py_file in file_dir.rglob("*.py"):
            if py_file == self._root / file_path:
                continue
            try:
                content = py_file.read_text()
                if module_name in content or Path(file_path).stem in content:
                    rel = str(py_file.relative_to(self._root))
                    importers.append(rel)
            except (OSError, UnicodeDecodeError):
                pass

        return importers[:10]

    def _module_to_file(self, module: str) -> str:
        """Convert a Python module path to a relative file path."""
        parts = module.split(".")
        candidate = Path(*parts).with_suffix(".py")
        if (self._root / candidate).exists():
            return str(candidate)
        return ""

    @staticmethod
    def _enclosing_function(tree: ast.Module, lineno: int) -> str:
        """Find the function enclosing a given line number."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.lineno <= lineno <= (node.end_lineno or node.lineno + 100):
                    return node.name
        return ""

    @staticmethod
    def _call_target(node: ast.Call) -> str:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    # ── Phase 2: Suspicion scoring ──────────────────────────────────────────

    def _score_suspicion(
        self, dag: CausalDAG, files_written: list[str], test_output: str
    ) -> None:
        """Score each node's likelihood of being the root cause."""
        # Mark failing nodes from test output
        failing_functions = self._extract_failing_functions(test_output)

        for node in dag.nodes:
            # Base suspicion: modified code is suspect
            if node.is_modified:
                node.suspicion_score = 0.6

            # Higher suspicion if in the failure trace
            if node.name in failing_functions:
                node.is_failing = True
                node.suspicion_score = max(node.suspicion_score, 0.8)
                dag.failing_nodes.append(node.node_id)

            # Modified AND failing = very suspect
            if node.is_modified and node.is_failing:
                node.suspicion_score = 0.95

        # Propagate suspicion through edges
        # Nodes called by modified/failing nodes get some suspicion
        modified_ids = {n.node_id for n in dag.nodes if n.is_modified}
        for edge in dag.edges:
            if edge.from_node in modified_ids:
                target = next((n for n in dag.nodes if n.node_id == edge.to_node), None)
                if target and not target.is_modified:
                    target.suspicion_score = max(
                        target.suspicion_score,
                        0.3 * edge.weight,
                    )

    @staticmethod
    def _extract_failing_functions(test_output: str) -> set[str]:
        """Extract function names from test failure output."""
        functions: set[str] = set()
        # Pattern: "FAILED test_path::test_name" or "in function_name"
        for match in re.finditer(r"FAILED.*::(\w+)", test_output):
            functions.add(match.group(1))
        for match in re.finditer(r'in (\w+)', test_output):
            functions.add(match.group(1))
        return functions

    # ── Phase 3: Interventional queries ─────────────────────────────────────

    def _get_top_suspects(self, dag: CausalDAG) -> list[CausalNode]:
        """Get top suspect nodes sorted by suspicion score."""
        suspects = [n for n in dag.nodes if n.suspicion_score > 0.1]
        suspects.sort(key=lambda n: n.suspicion_score, reverse=True)
        return suspects

    async def _interventional_query(
        self,
        suspect: CausalNode,
        dag: CausalDAG,
        test_output: str,
        files_written: list[str],
    ) -> tuple[InterventionResult, int]:
        """Ask the LLM: 'If this function were correct, would the test pass?'"""
        # Read the suspect function's code
        function_code = self._read_function_code(suspect)

        prompt = INTERVENTION_PROMPT.format(
            test_output=test_output[:2000],
            function_name=suspect.name,
            file_path=suspect.file_path,
            function_code=function_code[:2000],
            diff_summary=f"Modified files: {', '.join(files_written)}",
        )

        response = await self._llm.complete(  # type: ignore[attr-defined]
            system=None,
            messages=[Message(role="user", content=prompt)],
            max_tokens=512,
        )

        tokens = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)

        # Parse response
        try:
            data = self._extract_json(response.text)
            return InterventionResult(
                intervention_id=f"iv_{new_id()[:8]}",
                target_node=suspect.node_id,
                outcome_changed=data.get("outcome_changed", False),
                reasoning=data.get("reasoning", ""),
                tokens_used=tokens,
            ), tokens
        except (json.JSONDecodeError, TypeError):
            return InterventionResult(
                intervention_id=f"iv_{new_id()[:8]}",
                target_node=suspect.node_id,
                reasoning=response.text[:500],
                tokens_used=tokens,
            ), tokens

    def _read_function_code(self, node: CausalNode) -> str:
        """Read the source code of a specific function/class."""
        full_path = self._root / node.file_path
        if not full_path.exists():
            return ""

        try:
            lines = full_path.read_text().splitlines()
            start = max(0, node.line_start - 1)
            end = min(len(lines), node.line_end)
            return "\n".join(lines[start:end])
        except OSError:
            return ""

    # ── Phase 4: Root cause determination ───────────────────────────────────

    def _determine_root_cause(
        self,
        suspects: list[CausalNode],
        interventions: list[InterventionResult],
        dag: CausalDAG,
    ) -> dict[str, Any]:
        """Determine the most likely root cause from interventions."""
        if not suspects:
            return {
                "node_id": "",
                "description": "No suspects identified",
                "confidence": 0.0,
                "reasoning_chain": ["No modified or failing nodes found in DAG"],
                "alternatives": [],
            }

        # Find interventions where outcome changed (= fixing that function would fix the test)
        causal_nodes: list[tuple[CausalNode, InterventionResult]] = []
        for intervention in interventions:
            if intervention.outcome_changed:
                node = next(
                    (n for n in suspects if n.node_id == intervention.target_node),
                    None,
                )
                if node:
                    causal_nodes.append((node, intervention))

        if causal_nodes:
            # Pick the one with highest suspicion score
            causal_nodes.sort(key=lambda x: x[0].suspicion_score, reverse=True)
            best_node, best_iv = causal_nodes[0]

            reasoning_chain = [
                f"Built causal DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges",
                f"Identified {len(suspects)} suspect nodes (suspicion > 0.1)",
                f"Ran {len(interventions)} interventional queries",
                f"Node '{best_node.name}' in {best_node.file_path} identified as root cause",
                f"Intervention reasoning: {best_iv.reasoning}",
            ]

            return {
                "node_id": best_node.node_id,
                "file_path": best_node.file_path,
                "function_name": best_node.name,
                "description": f"Root cause: {best_node.name} in {best_node.file_path}. {best_iv.reasoning}",
                "confidence": min(0.95, best_node.suspicion_score + 0.1),
                "reasoning_chain": reasoning_chain,
                "alternatives": [
                    n.node_id for n, _ in causal_nodes[1:3]
                ],
            }

        # No intervention changed the outcome - use highest suspicion node
        best = suspects[0]
        return {
            "node_id": best.node_id,
            "file_path": best.file_path,
            "function_name": best.name,
            "description": f"Suspected cause: {best.name} in {best.file_path} (suspicion={best.suspicion_score:.2f})",
            "confidence": best.suspicion_score * 0.7,
            "reasoning_chain": [
                f"Built causal DAG with {len(dag.nodes)} nodes",
                "No interventions changed the test outcome",
                f"Falling back to highest suspicion score: {best.name}",
            ],
            "alternatives": [n.node_id for n in suspects[1:3]],
        }

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))  # type: ignore[no-any-return]
        raise json.JSONDecodeError("No JSON found", text, 0)
