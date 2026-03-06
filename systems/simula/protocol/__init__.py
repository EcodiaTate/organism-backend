"""Phase 2 + supporting: Protocol State Behaviour.

Tools for observing state transitions and interpretation consistency.
"""

from systems.simula.protocol.state_machine import (
    SIMULA_STATE_MACHINE,
    StateTransition,
)

__all__ = [
    "StateTransition",
    "SIMULA_STATE_MACHINE",
]
