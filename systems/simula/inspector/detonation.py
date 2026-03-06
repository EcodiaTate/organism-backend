"""
EcodiaOS — Inspector Detonation Module

Re-exports the ExecutionTestChamber as LiveDetonationChamber for backward compatibility.
"""

from systems.simula.inspector.execution import ExecutionTestChamber

# Alias for backward compatibility with existing imports
LiveDetonationChamber = ExecutionTestChamber

__all__ = ["LiveDetonationChamber"]
