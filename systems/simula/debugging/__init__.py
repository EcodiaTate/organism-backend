"""
EcodiaOS -- Simula Causal Debugging Subsystem (Stage 5D)

When a health check fails after applying a change, build a causal DAG
from the execution trace and use AID interventional reasoning to narrow
the root cause before invoking repair or rollback.

  CausalDebugger  — Build causal DAG + AID interventional reasoning
"""

from systems.simula.debugging.causal_dag import CausalDebugger
from systems.simula.debugging.types import (
    CausalDAG,
    CausalDiagnosis,
    CausalEdge,
    CausalEdgeKind,
    CausalNode,
    CausalNodeKind,
    Intervention,
    InterventionKind,
    InterventionResult,
)

__all__ = [
    "CausalDebugger",
    "CausalNodeKind",
    "CausalEdgeKind",
    "InterventionKind",
    "CausalNode",
    "CausalEdge",
    "CausalDAG",
    "Intervention",
    "InterventionResult",
    "CausalDiagnosis",
]
