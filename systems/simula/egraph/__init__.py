"""
EcodiaOS -- Simula Equality Saturation Engine (Stage 6D)

E-graph based refactoring with semantic equivalence guarantees.
Removes LLM from optimization logic — pure algebraic rewriting.
"""

from systems.simula.egraph.equality_saturation import EqualitySaturationEngine

__all__ = [
    "EqualitySaturationEngine",
]
