"""
EcodiaOS -- Simula Training Module

Autonomous model fine-tuning pipeline: extract memories from Neo4j,
deploy LoRA training on Akash GPU compute, upload weights to IPFS.

Components:
  - DatasetBuilder: Neo4j → JSONL extraction (instruction / DPO / chat)
  - ExecuteModelFineTune: Axon executor orchestrating the full pipeline
  - train_lora.py: Standalone Unsloth script running on the Akash GPU node
  - types: Data models for the training lifecycle
"""

from systems.simula.training.dataset_builder import DatasetBuilder
from systems.simula.training.executor import ExecuteModelFineTune
from systems.simula.training.types import (
    DatasetFormat,
    DatasetManifest,
    DatasetRecord,
    MemorySource,
    TrainingHyperparams,
    TrainingJobResult,
    TrainingJobStatus,
)

__all__ = [
    "DatasetBuilder",
    "DatasetFormat",
    "DatasetManifest",
    "DatasetRecord",
    "ExecuteModelFineTune",
    "MemorySource",
    "TrainingHyperparams",
    "TrainingJobResult",
    "TrainingJobStatus",
]
