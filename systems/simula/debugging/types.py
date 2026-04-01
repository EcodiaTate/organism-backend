"""
EcodiaOS -- Simula Debugging Types (Stage 5D)

Types for causal debugging - when a health check fails after applying
a change, build a causal DAG from the execution trace and use AID
(Actual Interventionist Definition) reasoning to narrow the root cause
before invoking repair or rollback.

Target: 97.72 % root-cause precision (AID benchmark).
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now

# ── Enums ────────────────────────────────────────────────────────────────────


class CausalNodeKind(enum.StrEnum):
    """Kind of node in the causal DAG."""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    TEST = "test"
    CONFIG = "config"


class CausalEdgeKind(enum.StrEnum):
    """Kind of edge (relationship) in the causal DAG."""

    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    TESTS = "tests"
    MUTATES = "mutates"


class InterventionKind(enum.StrEnum):
    """Type of causal intervention used during AID reasoning."""

    MOCK = "mock"
    SKIP = "skip"
    INJECT_FAULT = "inject_fault"
    MODIFY_INPUT = "modify_input"


# ── DAG models ──────────────────────────────────────────────────────────────


class CausalNode(EOSBaseModel):
    """One node in the causal DAG (function, class, module, test, or config)."""

    node_id: str
    kind: CausalNodeKind = CausalNodeKind.FUNCTION
    name: str = ""
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    is_modified: bool = False  # was this node touched by the applied change?
    is_failing: bool = False  # does this node appear in the failure trace?
    suspicion_score: float = 0.0  # 0.0–1.0, higher = more likely root cause


class CausalEdge(EOSBaseModel):
    """Directed edge in the causal DAG."""

    from_node: str  # node_id
    to_node: str  # node_id
    kind: CausalEdgeKind = CausalEdgeKind.CALLS
    weight: float = 1.0  # influence weight for propagation


class CausalDAG(EOSBaseModel):
    """The full causal graph built from AST + execution trace analysis."""

    nodes: list[CausalNode] = Field(default_factory=list)
    edges: list[CausalEdge] = Field(default_factory=list)
    modified_nodes: list[str] = Field(default_factory=list)  # node_ids of changed code
    failing_nodes: list[str] = Field(default_factory=list)  # node_ids in failure trace
    total_functions: int = 0
    total_edges: int = 0
    built_at: datetime = Field(default_factory=utc_now)


# ── Intervention models ─────────────────────────────────────────────────────


class Intervention(EOSBaseModel):
    """A single causal intervention ("if X were correct, would Y still fail?")."""

    intervention_id: str = ""
    target_node: str = ""  # node_id
    kind: InterventionKind = InterventionKind.MOCK
    description: str = ""
    hypothesis: str = ""  # "If this function were correct, test T would pass"


class InterventionResult(EOSBaseModel):
    """Observed result of executing a causal intervention."""

    intervention_id: str = ""
    target_node: str = ""
    outcome_changed: bool = False  # did the intervention change the failure?
    new_test_results: str = ""
    reasoning: str = ""
    tokens_used: int = 0
    duration_ms: int = 0


# ── Diagnosis ───────────────────────────────────────────────────────────────


class CausalDiagnosis(EOSBaseModel):
    """Final causal debugging output - the root cause and supporting evidence."""

    dag: CausalDAG | None = None
    interventions: list[InterventionResult] = Field(default_factory=list)
    root_cause_node: str = ""  # node_id of the most likely root cause
    root_cause_file: str = ""
    root_cause_function: str = ""
    root_cause_description: str = ""
    confidence: float = 0.0
    reasoning_chain: list[str] = Field(default_factory=list)  # step-by-step reasoning
    alternative_causes: list[str] = Field(default_factory=list)  # other candidate node_ids
    total_interventions: int = 0
    total_tokens: int = 0
    total_duration_ms: int = 0
    fault_injection_used: bool = False
