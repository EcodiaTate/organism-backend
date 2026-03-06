"""
EcodiaOS — Simula Execution Variance Collector

Captures instruction path variance during code agent execution:
  - Tool call sequence (ordered DAG of tools called)
  - Decision points (where agent chose between alternatives)
  - Timing per tool call
  - Token consumption per tool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from primitives.common import new_id

logger = structlog.get_logger().bind(system="simula.instrumentation.variance")


@dataclass
class ToolExecution:
    """Record of a single tool call during agent execution."""
    execution_id: str
    tool_name: str
    input_spec: dict[str, Any]
    output_summary: str
    duration_ms: float
    tokens_used: int
    success: bool
    error: str = ""
    sequence_index: int = 0


@dataclass
class Decision:
    """A point where the agent made a choice."""
    decision_id: str
    step_index: int
    description: str
    alternatives: list[str] = field(default_factory=list)
    chosen: str = ""
    confidence: float = 0.5


class ExecutionVarianceCollector:
    """
    Captures tool call sequencing and decision points in code agent loops.

    Wraps the agentic loop to record:
      1. Every tool call (input, output, timing, tokens)
      2. Every decision point (choice between alternatives)
      3. Complete execution trace with ordering
    """

    def __init__(self, proposal_id: str) -> None:
        self.proposal_id = proposal_id
        self.executions: list[ToolExecution] = []
        self.decisions: list[Decision] = []
        self._tool_call_counter = 0
        self._start_time_ns = 0
        self._log = logger

    def start_collection(self) -> None:
        """Begin recording a new execution trace."""
        self.executions.clear()
        self.decisions.clear()
        self._tool_call_counter = 0
        import time
        self._start_time_ns = time.monotonic_ns()

    def record_tool_call(
        self,
        tool_name: str,
        input_spec: dict[str, Any],
        output_summary: str,
        duration_ms: float,
        tokens_used: int,
        success: bool,
        error: str = "",
    ) -> None:
        """Record a single tool execution."""
        execution = ToolExecution(
            execution_id=f"exec_{new_id()[:8]}",
            tool_name=tool_name,
            input_spec=input_spec,
            output_summary=output_summary[:200],  # Limit output recording
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            success=success,
            error=error,
            sequence_index=self._tool_call_counter,
        )
        self.executions.append(execution)
        self._tool_call_counter += 1

        self._log.debug(
            "tool_call_recorded",
            proposal_id=self.proposal_id,
            tool_name=tool_name,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            sequence=self._tool_call_counter,
        )

    def record_decision(
        self,
        description: str,
        alternatives: list[str],
        chosen: str,
        confidence: float = 0.5,
    ) -> None:
        """Record a decision point in the agent's reasoning."""
        decision = Decision(
            decision_id=f"dec_{new_id()[:8]}",
            step_index=self._tool_call_counter,
            description=description,
            alternatives=alternatives,
            chosen=chosen,
            confidence=confidence,
        )
        self.decisions.append(decision)

        self._log.debug(
            "decision_recorded",
            proposal_id=self.proposal_id,
            description=description,
            chosen=chosen,
            confidence=confidence,
        )

    def get_execution_dag(self) -> dict[str, Any]:
        """
        Build a DAG representation of tool calls.

        Returns:
            {
              "nodes": [{"id": "exec_...", "tool": "read_file", ...}],
              "edges": [{"from": "exec_1", "to": "exec_2", "type": "sequence"}],
            }
        """
        nodes = []
        for exec in self.executions:
            nodes.append({
                "id": exec.execution_id,
                "tool": exec.tool_name,
                "sequence": exec.sequence_index,
                "duration_ms": exec.duration_ms,
                "tokens": exec.tokens_used,
                "success": exec.success,
            })

        # Simple linear DAG: each call sequences to next
        edges = []
        for i in range(len(self.executions) - 1):
            edges.append({
                "from": self.executions[i].execution_id,
                "to": self.executions[i + 1].execution_id,
                "type": "sequence",
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    def compute_execution_dag_edit_distance(self, other: ExecutionVarianceCollector) -> float:
        """
        Compare two execution DAGs. Return edit distance (0.0 = identical, 1.0 = completely different).

        Simple metric: Levenshtein distance on tool sequences.
        """
        seq1 = [e.tool_name for e in self.executions]
        seq2 = [e.tool_name for e in other.executions]

        # Levenshtein distance
        if not seq1 and not seq2:
            return 0.0
        if not seq1 or not seq2:
            return 1.0

        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        max_len = max(m, n)
        return dp[m][n] / max_len if max_len > 0 else 0.0

    def identify_decision_points(self) -> list[dict[str, Any]]:
        """
        Where did the agent make a choice? Return list of decision points with alternatives.
        """
        return [
            {
                "decision_id": d.decision_id,
                "step": d.step_index,
                "description": d.description,
                "alternatives": d.alternatives,
                "chosen": d.chosen,
                "confidence": d.confidence,
            }
            for d in self.decisions
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of the execution."""
        total_duration_ms = sum(e.duration_ms for e in self.executions)
        total_tokens = sum(e.tokens_used for e in self.executions)
        failed_count = sum(1 for e in self.executions if not e.success)

        return {
            "proposal_id": self.proposal_id,
            "total_tool_calls": len(self.executions),
            "total_decisions": len(self.decisions),
            "total_duration_ms": total_duration_ms,
            "total_tokens": total_tokens,
            "failed_calls": failed_count,
            "success_rate": (len(self.executions) - failed_count) / len(self.executions) if self.executions else 0.0,
            "mean_tool_duration_ms": total_duration_ms / len(self.executions) if self.executions else 0.0,
            "decision_points": self.identify_decision_points(),
        }
