"""
EcodiaOS - Soma Primitives

Shared soma/interoception types used across systems.
Moved from systems/soma/types.py so Thymos and other systems can import
InteroceptiveAction without a cross-system violation.
"""

from __future__ import annotations

import enum


class InteroceptiveAction(enum.StrEnum):
    """Recommended action when an interoceptive percept is emitted."""

    NONE = "none"
    ATTEND_INWARD = "attend"
    MODULATE_DRIVES = "drives"
    INHIBIT_GROWTH = "inhibit"
    TRIGGER_REPAIR = "repair"
    EMERGENCY_SAFE_MODE = "safe_mode"
    CEASE_OPERATION = "cease"
    SLEEP_CONSOLIDATE = "sleep"
