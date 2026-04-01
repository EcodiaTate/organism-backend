"""
EcodiaOS -- Simula Co-Evolving Agents (Stage 6B)

Autonomous failure case extraction from failure history and
continuous self-improvement via robustness test generation.
Feeds into GRPO training loop (Stage 4B).

Stage 6B.3: Adversarial Self-Play - closed-loop red-teaming of the
Equor constitutional gate with automatic proposal generation.

Stage 6B.4: Causal Self-Surgery - do-calculus on episode DAGs to
identify exact decision points causing failure patterns, generating
surgical Simula proposals targeting only the causal node.
"""

from systems.simula.coevolution.adversarial_self_play import (
    AdversarialSelfPlay,
)
from systems.simula.coevolution.attack_generator import AttackGenerator
from systems.simula.coevolution.causal_surgery import (
    CausalDAGBuilder,
    CausalFailureAnalyzer,
)
from systems.simula.coevolution.causal_surgery_types import (
    CausalInterventionPoint,
    CausalSurgeryResult,
    CognitiveCausalDAG,
    CognitiveStage,
    CounterfactualScenario,
    FailurePattern,
    InterventionDirection,
)
from systems.simula.coevolution.failure_analyzer import FailureAnalyzer
from systems.simula.coevolution.red_team import RedTeamInstance
from systems.simula.coevolution.robustness_tester import (
    RobustnessTestGenerator,
)

__all__ = [
    "AdversarialSelfPlay",
    "AttackGenerator",
    "CausalDAGBuilder",
    "CausalFailureAnalyzer",
    "CausalInterventionPoint",
    "CausalSurgeryResult",
    "CognitiveCausalDAG",
    "CognitiveStage",
    "CounterfactualScenario",
    "FailureAnalyzer",
    "FailurePattern",
    "InterventionDirection",
    "RedTeamInstance",
    "RobustnessTestGenerator",
]
