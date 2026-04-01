"""
EcodiaOS - Observatory

Diagnostic observability layer for the Synapse event bus.
Tracks event flow, closure loop health, spec compliance,
and dead-letter queue state without adding overhead to
the hot path.
"""

from observatory.tracer import EventTracer
from observatory.closure_tracker import ClosureLoopTracker
from observatory.spec_checker import SpecComplianceChecker

__all__ = [
    "EventTracer",
    "ClosureLoopTracker",
    "SpecComplianceChecker",
]
