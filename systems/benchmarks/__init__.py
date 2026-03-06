"""
EcodiaOS — Benchmarks System

Quantitative measurement layer. Collects five KPIs from live system
health/stats endpoints on a configurable interval, stores snapshots in
TimescaleDB, and fires a BENCHMARK_REGRESSION Synapse event whenever any
metric regresses more than 20 % from its rolling average.

KPIs tracked
────────────
  decision_quality       — % of Nova outcomes rated positive (success)
  llm_dependency         — % of decisions that required an LLM call
                           (slow_path / total_decisions)
  economic_ratio         — income / expenses from Oikos (revenue_7d / costs_7d)
  learning_rate          — Evo hypotheses confirmed (supported) per N cycles
  mutation_success_rate  — Simula proposals_approved / proposals_received

Usage
─────
  from systems.benchmarks import BenchmarkService
"""

from systems.benchmarks.service import BenchmarkService

__all__ = ["BenchmarkService"]
