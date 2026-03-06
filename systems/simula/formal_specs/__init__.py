"""
EcodiaOS -- Simula Formal Spec Generation (Stage 6C)

Auto-generate Dafny, TLA+, Alloy, and Self-Spec DSL specifications
for system interfaces and distributed interactions.
"""

from systems.simula.formal_specs.spec_generator import FormalSpecGenerator

__all__ = [
    "FormalSpecGenerator",
]
