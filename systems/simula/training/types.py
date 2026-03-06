"""
EcodiaOS -- Simula Training Types

Data types for the autonomous model fine-tuning pipeline.

The pipeline extracts high-quality memories from Neo4j (successful Intents,
applied EvolutionProposals, FailureAnalyzer traces), formats them as JSONL
for instruction tuning or DPO, deploys a GPU job to Akash, and uploads the
resulting LoRA adapter to IPFS.

Namespace: systems.simula.training.types
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now

# ── Enums ────────────────────────────────────────────────────────────────────


class DatasetFormat(enum.StrEnum):
    """Output format for the training dataset."""

    INSTRUCTION = "instruction"       # {instruction, input, output}
    DPO = "dpo"                       # {prompt, chosen, rejected}
    CHAT = "chat"                     # {messages: [{role, content}]}


class MemorySource(enum.StrEnum):
    """Source category for a training record extracted from Neo4j."""

    SUCCESSFUL_INTENT = "successful_intent"
    APPLIED_PROPOSAL = "applied_proposal"
    FAILURE_TRACE = "failure_trace"


class TrainingJobStatus(enum.StrEnum):
    """Lifecycle status of a fine-tuning job."""

    PENDING = "pending"
    BUILDING_DATASET = "building_dataset"
    UPLOADING_DATASET = "uploading_dataset"
    DEPLOYING_GPU = "deploying_gpu"
    TRAINING = "training"
    UPLOADING_WEIGHTS = "uploading_weights"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Dataset Records ──────────────────────────────────────────────────────────


class DatasetRecord(EOSBaseModel):
    """
    One training record extracted from the knowledge graph.

    For INSTRUCTION format: instruction + input + output.
    For DPO format: prompt + chosen + rejected.
    """

    source: MemorySource
    source_id: str = ""           # Intent/Proposal/Record ID

    # Instruction tuning fields
    instruction: str = ""
    input: str = ""
    output: str = ""

    # DPO fields (chosen = successful output, rejected = failed output)
    prompt: str = ""
    chosen: str = ""
    rejected: str = ""

    # Metadata for provenance
    quality_score: float = 0.0    # 0.0-1.0 composite quality signal
    category: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class DatasetManifest(Identified, Timestamped):
    """Metadata about a built training dataset."""

    format: DatasetFormat = DatasetFormat.INSTRUCTION
    record_count: int = 0
    sources: dict[str, int] = Field(default_factory=dict)  # source -> count
    total_tokens_estimate: int = 0
    ipfs_cid: str = ""            # CID of uploaded JSONL file
    file_size_bytes: int = 0
    build_duration_ms: int = 0


# ── Training Job ─────────────────────────────────────────────────────────────


class TrainingHyperparams(EOSBaseModel):
    """Hyperparameters for the LoRA fine-tuning job."""

    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"


class TrainingJobResult(Identified, Timestamped):
    """
    Full result of an autonomous fine-tuning job.

    Logged as a :FineTuneRecord node in Neo4j for auditability.
    """

    status: TrainingJobStatus = TrainingJobStatus.PENDING

    # Dataset
    dataset_manifest: DatasetManifest | None = None

    # Akash deployment
    akash_deployment_id: str = ""
    akash_endpoint: str = ""
    gpu_type: str = ""

    # Hyperparams used
    hyperparams: TrainingHyperparams = Field(default_factory=TrainingHyperparams)

    # Results
    adapter_ipfs_cid: str = ""    # IPFS CID of .safetensors adapter
    adapter_size_bytes: int = 0
    training_loss_final: float = 0.0
    eval_loss_final: float = 0.0

    # Resource usage
    gpu_hours: float = 0.0
    cost_estimate_usd: float = 0.0
    total_duration_ms: int = 0

    # Error tracking
    error: str = ""
    error_phase: TrainingJobStatus | None = None
