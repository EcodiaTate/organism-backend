"""
EcodiaOS - Simula Agents (Stages 2D + 5B)

AgentCoder pattern: separates test design from code generation
for higher-quality implementations via adversarial feedback.

  TestDesignerAgent  - generates tests independently from code
  TestExecutorAgent  - runs tests, collects structured results
  RepairAgent        - FSM-guided SRepair neural program repair (Stage 5B)
"""

from systems.simula.agents.repair_agent import RepairAgent
from systems.simula.agents.test_designer import TestDesignerAgent
from systems.simula.agents.test_executor import TestExecutorAgent

__all__ = [
    "TestDesignerAgent",
    "TestExecutorAgent",
    "RepairAgent",
]
