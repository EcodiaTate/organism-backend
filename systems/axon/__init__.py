"""
EcodiaOS -- Axon (Action Execution System)

Axon is the motor cortex. It takes Intents approved by Equor and turns them
into real-world effects -- API calls, data operations, scheduled tasks,
notifications, and federated messages.

If Nova decides what to do, Axon does it.
The gap between intention and action is where trust lives.

Axon is a *reactive* motor cortex: it adapts its safety thresholds based on
organism state (metabolic pressure, system degradation, sleep cycles) and
learns from its own execution patterns (introspection). It supports parallel
step execution for independent actions and distributed rate limiting via Redis.

Public interface:
  AxonService              -- main service class
  ExecutionRequest         -- input to AxonService.execute()
  AxonOutcome              -- output of AxonService.execute()
  Executor                 -- ABC for custom executors
  ExecutorRegistry         -- registry of available executors
  AxonReactiveAdapter      -- Synapse event subscriber for adaptive safety
  AxonIntrospector         -- execution pattern learning and degradation detection
"""

from systems.axon.executor import Executor
from systems.axon.introspection import AxonIntrospector
from systems.axon.reactive import AxonReactiveAdapter
from systems.axon.registry import ExecutorRegistry
from systems.axon.service import AxonService
from systems.axon.types import AxonOutcome, ExecutionRequest

__all__ = [
    "AxonService",
    "ExecutionRequest",
    "AxonOutcome",
    "Executor",
    "ExecutorRegistry",
    "AxonReactiveAdapter",
    "AxonIntrospector",
]
