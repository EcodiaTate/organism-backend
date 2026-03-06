"""
EcodiaOS -- Simula Orchestration Types (Stage 5C)

Types for multi-agent orchestration of complex, multi-file proposals.
Based on the MetaGPT structured-artifact pipeline and CodePlan adaptive DAG
decomposition, with a hard 2-agent-per-stage overcrowding constraint.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now

# ── Enums ────────────────────────────────────────────────────────────────────


class ArtifactKind(enum.StrEnum):
    """Types of structured artifacts passed between pipeline stages."""

    SPEC = "spec"
    DESIGN = "design"
    CODE = "code"
    TEST = "test"
    REVIEW = "review"


class TaskStatus(enum.StrEnum):
    """Status of an individual task node in the execution DAG."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class DelegationMode(enum.StrEnum):
    """How agents are assigned to a pipeline stage."""

    SINGLE_AGENT = "single_agent"
    DUAL_AGENT = "dual_agent"
    HIERARCHICAL = "hierarchical"


# ── Pipeline artifacts ──────────────────────────────────────────────────────


class PipelineArtifact(EOSBaseModel):
    """One typed artifact produced by a pipeline stage (not free-form chat)."""

    kind: ArtifactKind
    stage_index: int = 0
    content: str = ""
    files_referenced: list[str] = Field(default_factory=list)
    produced_by: str = ""  # agent identifier
    produced_at: datetime = Field(default_factory=utc_now)
    tokens_used: int = 0


# ── Task DAG ────────────────────────────────────────────────────────────────


class TaskNode(EOSBaseModel):
    """One node in the CodePlan-style task dependency graph."""

    node_id: str
    description: str = ""
    files_to_modify: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str = ""
    depends_on: list[str] = Field(default_factory=list)  # node_ids
    artifacts: list[PipelineArtifact] = Field(default_factory=list)
    duration_ms: int = 0
    error: str = ""


class TaskEdge(EOSBaseModel):
    """Directed edge in the task DAG (from_node must complete before to_node)."""

    from_node: str
    to_node: str
    edge_type: str = "depends_on"  # "depends_on"|"produces_input"|"tests"


class TaskDAG(EOSBaseModel):
    """Dependency-aware DAG of sub-tasks for a multi-file proposal."""

    nodes: list[TaskNode] = Field(default_factory=list)
    edges: list[TaskEdge] = Field(default_factory=list)
    topological_order: list[str] = Field(default_factory=list)  # node_ids in execution order
    parallel_stages: int = 0  # number of stages that can run concurrently
    total_files: int = 0
    built_at: datetime = Field(default_factory=utc_now)


# ── Stage & orchestrator results ────────────────────────────────────────────


class StageResult(EOSBaseModel):
    """Result of executing one pipeline stage (SPEC, DESIGN, CODE, TEST, REVIEW)."""

    stage: ArtifactKind
    status: TaskStatus = TaskStatus.PENDING
    agents_used: int = 0
    delegation_mode: DelegationMode = DelegationMode.SINGLE_AGENT
    artifacts: list[PipelineArtifact] = Field(default_factory=list)
    duration_ms: int = 0
    tokens_used: int = 0
    error: str = ""


class OrchestratorResult(EOSBaseModel):
    """Aggregate result of multi-agent orchestration for a proposal."""

    used: bool = False
    dag: TaskDAG | None = None
    stage_results: list[StageResult] = Field(default_factory=list)
    total_agents_used: int = 0
    parallel_stages_executed: int = 0
    files_modified: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: int = 0
    fell_back_to_single_agent: bool = False
    error: str = ""
