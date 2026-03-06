"""
EcodiaOS -- Simula Multi-Agent Orchestration Subsystem (Stage 5C)

Decomposes multi-file proposals into parallel sub-tasks with
dependency-aware DAGs. Based on MetaGPT structured artifacts +
CodePlan adaptive DAG decomposition.

  TaskPlanner             — AST import analysis → dependency DAG
  MultiAgentOrchestrator  — MetaGPT pipeline: SPEC → DESIGN → CODE → TEST → REVIEW
"""

from systems.simula.orchestration.orchestrator import (
    MultiAgentOrchestrator,
)
from systems.simula.orchestration.task_planner import TaskPlanner
from systems.simula.orchestration.types import (
    ArtifactKind,
    DelegationMode,
    OrchestratorResult,
    PipelineArtifact,
    StageResult,
    TaskDAG,
    TaskEdge,
    TaskNode,
    TaskStatus,
)

__all__ = [
    # Engines
    "TaskPlanner",
    "MultiAgentOrchestrator",
    # Types
    "ArtifactKind",
    "TaskStatus",
    "DelegationMode",
    "PipelineArtifact",
    "TaskNode",
    "TaskEdge",
    "TaskDAG",
    "StageResult",
    "OrchestratorResult",
]
