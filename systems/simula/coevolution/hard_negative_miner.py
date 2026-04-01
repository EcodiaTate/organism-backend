"""
EcodiaOS -- Hard Negative Miner (Stage 6B.1 re-export)

Re-exports FailureAnalyzer under the canonical module path used by tests
and external consumers. The implementation lives in failure_analyzer.py.

A hard negative is a training example where the model was confidently wrong:
rolled-back proposals, formal verification failures, health-check crashes.
These are the richest signal for GRPO training - they show exactly where
the model's confident output caused real damage.
"""

from systems.simula.coevolution.failure_analyzer import FailureAnalyzer

__all__ = ["FailureAnalyzer"]
