"""
EcodiaOS - Synapse (System #9)

The autonomic nervous system. Drives the cognitive cycle clock,
monitors system health, allocates resources, detects emergent
cognitive rhythms, and measures cross-system coherence.
"""

from systems.synapse.clock import CognitiveClock
from systems.synapse.coherence import CoherenceMonitor
from systems.synapse.degradation import DegradationManager
from systems.synapse.event_bus import EventBus
from systems.synapse.health import HealthMonitor
from systems.synapse.metabolism import MetabolicTracker
from systems.synapse.resources import ResourceAllocator
from systems.synapse.rhythm import DefaultRhythmStrategy, EmergentRhythmDetector
from systems.synapse.service import SynapseService
from systems.synapse.types import (
    BaseResourceAllocator,
    BaseRhythmStrategy,
    ClockState,
    CoherenceSnapshot,
    CycleResult,
    DegradationLevel,
    DegradationStrategy,
    ManagedSystemProtocol,
    MetabolicSnapshot,
    ResourceAllocation,
    ResourceSnapshot,
    RhythmSnapshot,
    RhythmState,
    SomaticCycleState,
    SomaTickEvent,
    SynapseEvent,
    SynapseEventType,
    SystemBudget,
    SystemHealthRecord,
    SystemHeartbeat,
    SystemStatus,
)

__all__ = [
    # Service
    "SynapseService",
    # Sub-systems
    "CognitiveClock",
    "CoherenceMonitor",
    "DegradationManager",
    "EventBus",
    "HealthMonitor",
    "ResourceAllocator",
    "EmergentRhythmDetector",
    "DefaultRhythmStrategy",
    "MetabolicTracker",
    # Strategy ABCs (NeuroplasticityBus targets)
    "BaseResourceAllocator",
    "BaseRhythmStrategy",
    # Types
    "ClockState",
    "CoherenceSnapshot",
    "CycleResult",
    "DegradationLevel",
    "DegradationStrategy",
    "ManagedSystemProtocol",
    "MetabolicSnapshot",
    "ResourceAllocation",
    "ResourceSnapshot",
    "RhythmSnapshot",
    "RhythmState",
    "SomaticCycleState",
    "SomaTickEvent",
    "SynapseEvent",
    "SynapseEventType",
    "SystemBudget",
    "SystemHeartbeat",
    "SystemHealthRecord",
    "SystemStatus",
]
