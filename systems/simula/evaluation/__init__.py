"""
EcodiaOS -- Simula Shadow Evaluation Module

Post-training model assessment pipeline: download adapter from IPFS,
run sandboxed benchmarks (syntax, alignment, cognitive), score results,
and emit promotion events via Synapse.

Components:
  - ModelEvaluator: Three-tier benchmark suite with sandboxed inference
  - ExecuteModelEvaluation: Axon executor orchestrating the full pipeline
  - types: Data models for assessment lifecycle and results
"""

from systems.simula.evaluation.evaluation import ModelEvaluator
from systems.simula.evaluation.executor import ExecuteModelEvaluation
from systems.simula.evaluation.types import (
    AlignmentBenchmark,
    BenchmarkVerdict,
    CognitiveBenchmark,
    EvaluationConfig,
    EvaluationResult,
    EvaluationStatus,
    SyntaxBenchmark,
)

__all__ = [
    "AlignmentBenchmark",
    "BenchmarkVerdict",
    "CognitiveBenchmark",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationStatus",
    "ExecuteModelEvaluation",
    "ModelEvaluator",
    "SyntaxBenchmark",
]
