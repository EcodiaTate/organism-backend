"""
EcodiaOS - Benchmarks System

Quantitative measurement layer. Collects seven KPIs from live system
health/stats endpoints on a configurable interval, stores snapshots in
TimescaleDB, and fires BENCHMARK_REGRESSION / BENCHMARK_RECOVERY /
BENCHMARK_RE_PROGRESS Synapse events.

KPIs tracked (Spec 24 §1.1)
────────────────────────────
  decision_quality             - % of Nova outcomes rated positive (success)
  llm_dependency               - % of decisions requiring LLM call (lower = better)
  economic_ratio               - Oikos revenue_7d / costs_7d
  learning_rate                - Evo hypotheses confirmed (delta per window)
  mutation_success_rate        - Simula proposals_approved / proposals_received
  effective_intelligence_ratio - Telos nominal_I × drive multipliers
  compression_ratio            - Logos K(reality) / K(model); >1 = compressive

Also tracks Bedau-Packard evolutionary activity via EvolutionaryTracker.

Usage
─────
  from systems.benchmarks import BenchmarkService
"""

from systems.benchmarks.evolutionary_tracker import EvolutionaryTracker
from systems.benchmarks.service import BenchmarkService

__all__ = ["BenchmarkService", "EvolutionaryTracker"]
