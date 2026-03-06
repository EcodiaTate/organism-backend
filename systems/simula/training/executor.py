"""
EcodiaOS -- ExecuteModelFineTune Executor

Orchestrates the full autonomous fine-tuning pipeline:
  1. Build dataset (DatasetBuilder → JSONL)
  2. Upload dataset to IPFS (PinataClient)
  3. Deploy GPU node on Akash (AkashProvider + GPU SDL template)
  4. Monitor training progress via status endpoint
  5. Retrieve adapter CID from the completed node
  6. Log the new model weights CID to Neo4j

This executor borrows the Akash client patterns from Skia's restoration
pipeline but does NOT modify ComputeArbitrageExecutor.

Registered as action_type = "executor.model_finetune".
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from infrastructure.providers.akash import AkashProvider
from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)
from systems.simula.training.dataset_builder import DatasetBuilder
from systems.simula.training.types import (
    DatasetFormat,
    TrainingHyperparams,
    TrainingJobResult,
    TrainingJobStatus,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.skia.pinata_client import PinataClient

logger = structlog.get_logger("systems.simula.training.executor")

# Default path to GPU SDL template (relative to package root)
_DEFAULT_SDL_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "infrastructure" / "sdl" / "gpu_finetune.sdl.yaml"
)

# Polling configuration
_POLL_INTERVAL_S = 30.0
_MAX_POLL_ATTEMPTS = 240   # 240 × 30s = 2 hours max training time
_DEPLOY_TIMEOUT_S = 600.0  # 10 minutes to get a GPU lease


class ExecuteModelFineTune(Executor):
    """
    Axon executor for autonomous model fine-tuning.

    Orchestrates: dataset build → IPFS upload → Akash GPU deploy →
    poll training → retrieve weights CID → log to Neo4j.
    """

    action_type = "executor.model_finetune"
    description = "Fine-tune a local model using organism memories via Akash GPU compute"
    required_autonomy = 3           # STEWARD — fully autonomous
    reversible = False              # Model weights are append-only
    max_duration_ms = 7_200_000     # 2 hours hard limit
    rate_limit = RateLimit.per_hour(2)  # Max 2 fine-tune jobs per hour

    def __init__(
        self,
        neo4j: Neo4jClient,
        pinata: PinataClient,
        *,
        akash_api_url: str = "https://console-api.akash.network",
        akash_wallet_address: str = "",
        docker_image: str = "ghcr.io/ecodiaos/finetune:latest",
        gpu_model: str = "a100",
        gpu_ram: str = "40Gi",
        cpu_units: str = "8000m",
        memory: str = "32Gi",
        storage: str = "100Gi",
        status_port: int = 8080,
        sdl_template_path: str = "",
    ) -> None:
        self._neo4j = neo4j
        self._pinata = pinata
        self._docker_image = docker_image
        self._gpu_model = gpu_model
        self._gpu_ram = gpu_ram
        self._cpu_units = cpu_units
        self._memory = memory
        self._storage = storage
        self._status_port = status_port
        self._sdl_path = sdl_template_path or _DEFAULT_SDL_PATH
        self._log = logger.bind(executor=self.action_type)

        self._akash = AkashProvider(
            api_url=akash_api_url,
            wallet_address=akash_wallet_address,
            sdl_template_path=self._sdl_path,
            docker_image=docker_image,
            deploy_timeout_s=_DEPLOY_TIMEOUT_S,
        )

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate fine-tuning parameters."""
        fmt = params.get("format", "instruction")
        if fmt not in ("instruction", "dpo", "chat"):
            return ValidationResult.fail(
                f"Invalid dataset format: {fmt}. Must be instruction, dpo, or chat.",
                format=f"Got {fmt}",
            )

        base_model = params.get("base_model", "")
        if base_model and not isinstance(base_model, str):
            return ValidationResult.fail("base_model must be a string")

        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute the full fine-tuning pipeline.

        Params:
            format: "instruction" | "dpo" | "chat" (default: instruction)
            base_model: HuggingFace model ID (default: from hyperparams)
            hyperparams: dict of training hyperparameters (optional overrides)
            max_intents: int (default: 500)
            max_proposals: int (default: 300)
            max_failures: int (default: 200)
        """
        start = time.monotonic()
        job = TrainingJobResult()

        fmt_str = params.get("format", "instruction")
        fmt = DatasetFormat(fmt_str)
        hyperparams = TrainingHyperparams(**(params.get("hyperparams", {})))
        if params.get("base_model"):
            hyperparams.base_model = params["base_model"]
        job.hyperparams = hyperparams

        try:
            # Phase 1: Build dataset from Neo4j
            job.status = TrainingJobStatus.BUILDING_DATASET
            self._log.info("finetune_phase", phase="building_dataset")

            builder = DatasetBuilder(
                self._neo4j,
                max_intents=params.get("max_intents", 500),
                max_proposals=params.get("max_proposals", 300),
                max_failures=params.get("max_failures", 200),
            )
            records = await builder.build()

            if not records:
                return ExecutionResult(
                    success=False,
                    error="No training records found in Neo4j. The organism needs more experiences first.",
                )

            # For DPO, assemble contrastive pairs
            if fmt == DatasetFormat.DPO:
                dpo_records = builder.assemble_dpo_pairs(records)
                if not dpo_records:
                    return ExecutionResult(
                        success=False,
                        error="Could not assemble DPO pairs — need both successes and failures.",
                    )
                jsonl_bytes = builder.to_jsonl(dpo_records, fmt)
                manifest = builder.build_manifest(
                    dpo_records, jsonl_bytes,
                    build_duration_ms=int((time.monotonic() - start) * 1000),
                )
            else:
                jsonl_bytes = builder.to_jsonl(records, fmt)
                manifest = builder.build_manifest(
                    records, jsonl_bytes,
                    build_duration_ms=int((time.monotonic() - start) * 1000),
                )

            self._log.info(
                "dataset_built",
                records=manifest.record_count,
                size_bytes=manifest.file_size_bytes,
                tokens_estimate=manifest.total_tokens_estimate,
            )

            # Phase 2: Upload dataset to IPFS
            job.status = TrainingJobStatus.UPLOADING_DATASET
            self._log.info("finetune_phase", phase="uploading_dataset")

            dataset_cid, _ = await self._pinata.pin_bytes(
                jsonl_bytes,
                name=f"ecodiaos-training-{manifest.id}",
            )
            manifest.ipfs_cid = dataset_cid
            job.dataset_manifest = manifest
            self._log.info("dataset_uploaded", cid=dataset_cid)

            # Phase 3: Deploy Akash GPU node
            job.status = TrainingJobStatus.DEPLOYING_GPU
            self._log.info("finetune_phase", phase="deploying_gpu")

            import orjson
            training_args_json = orjson.dumps(hyperparams.model_dump()).decode()

            pinata_jwt = context.credentials.get("pinata_jwt") or ""
            pinata_gw = context.credentials.get("pinata_gateway_url") or "https://gateway.pinata.cloud"

            deploy_result = await self._akash.deploy(
                image=self._docker_image,
                env_vars={
                    "DATASET_CID": dataset_cid,
                    "PINATA_JWT": pinata_jwt,
                    "PINATA_GATEWAY_URL": pinata_gw,
                    "BASE_MODEL": hyperparams.base_model,
                    "TRAINING_ARGS": training_args_json,
                    "STATUS_PORT": str(self._status_port),
                },
                sdl_overrides={
                    "GPU_MODEL": self._gpu_model,
                    "GPU_RAM": self._gpu_ram,
                    "CPU_UNITS": self._cpu_units,
                    "MEMORY": self._memory,
                    "STORAGE": self._storage,
                },
            )

            if not deploy_result.success:
                return ExecutionResult(
                    success=False,
                    error=f"Akash GPU deployment failed: {deploy_result.error}",
                    data={"dataset_cid": dataset_cid},
                )

            job.akash_deployment_id = deploy_result.deployment_id
            job.akash_endpoint = deploy_result.endpoint
            job.gpu_type = self._gpu_model
            self._log.info(
                "gpu_deployed",
                deployment_id=deploy_result.deployment_id,
                endpoint=deploy_result.endpoint,
            )

            # Phase 4: Monitor training progress
            job.status = TrainingJobStatus.TRAINING
            self._log.info("finetune_phase", phase="monitoring_training")

            adapter_cid = await self._poll_training_status(deploy_result.endpoint)

            if not adapter_cid:
                return ExecutionResult(
                    success=False,
                    error="Training did not complete — no adapter CID returned.",
                    data={
                        "dataset_cid": dataset_cid,
                        "deployment_id": deploy_result.deployment_id,
                    },
                )

            # Phase 5: Log results to Neo4j
            job.status = TrainingJobStatus.COMPLETED
            job.adapter_ipfs_cid = adapter_cid
            elapsed_ms = int((time.monotonic() - start) * 1000)
            job.total_duration_ms = elapsed_ms

            await self._log_to_neo4j(job)

            self._log.info(
                "finetune_complete",
                adapter_cid=adapter_cid,
                dataset_cid=dataset_cid,
                duration_ms=elapsed_ms,
            )

            return ExecutionResult(
                success=True,
                data={
                    "adapter_cid": adapter_cid,
                    "dataset_cid": dataset_cid,
                    "record_count": manifest.record_count,
                    "base_model": hyperparams.base_model,
                    "deployment_id": deploy_result.deployment_id,
                    "duration_ms": elapsed_ms,
                },
                side_effects=[
                    f"Uploaded training dataset to IPFS: {dataset_cid}",
                    f"Deployed GPU training job on Akash: {deploy_result.deployment_id}",
                    f"Uploaded LoRA adapter to IPFS: {adapter_cid}",
                    "Logged FineTuneRecord to Neo4j",
                ],
                new_observations=[
                    f"Fine-tuned {hyperparams.base_model} with {manifest.record_count} examples. "
                    f"Adapter CID: {adapter_cid}. This is my upgraded brain.",
                ],
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            job.status = TrainingJobStatus.FAILED
            job.error = str(exc)
            job.error_phase = job.status
            job.total_duration_ms = elapsed_ms

            self._log.error("finetune_failed", error=str(exc), phase=job.status.value)

            return ExecutionResult(
                success=False,
                error=f"Fine-tuning pipeline failed: {exc}",
                data={"phase": job.status.value, "duration_ms": elapsed_ms},
            )

    # ── Private: Polling ──────────────────────────────────────────────────

    async def _poll_training_status(self, endpoint: str) -> str:
        """
        Poll the training node's /status endpoint until completion or failure.

        Returns the adapter CID on success, empty string on failure.
        """
        status_url = f"{endpoint.rstrip('/')}/status"

        for attempt in range(_MAX_POLL_ATTEMPTS):
            await asyncio.sleep(_POLL_INTERVAL_S)

            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(status_url)
                    if resp.status_code != 200:
                        self._log.warning(
                            "training_status_poll_error",
                            status=resp.status_code,
                            attempt=attempt,
                        )
                        continue

                    data = resp.json()
                    phase = data.get("phase", "unknown")
                    progress = data.get("progress", 0.0)
                    loss = data.get("current_loss", 0.0)

                    self._log.info(
                        "training_progress",
                        phase=phase,
                        progress=f"{progress:.1%}",
                        loss=loss,
                        attempt=attempt,
                    )

                    if phase == "completed":
                        adapter_cid = data.get("adapter_cid", "")
                        if adapter_cid:
                            return str(adapter_cid)
                        self._log.error("training_completed_no_cid")
                        return ""

                    if phase == "failed":
                        error = data.get("error", "Unknown training error")
                        self._log.error("training_node_failed", error=error)
                        return ""

            except Exception as exc:
                self._log.warning(
                    "training_poll_exception",
                    error=str(exc),
                    attempt=attempt,
                )

        self._log.error("training_poll_timeout", max_attempts=_MAX_POLL_ATTEMPTS)
        return ""

    # ── Private: Neo4j Logging ────────────────────────────────────────────

    async def _log_to_neo4j(self, job: TrainingJobResult) -> None:
        """
        Write the FineTuneRecord node to Neo4j.

        Links to the organism's identity and enables historical tracking
        of all fine-tuning runs.
        """
        await self._neo4j.execute_write(
            """
            CREATE (ft:FineTuneRecord {
                id: $id,
                status: $status,
                adapter_ipfs_cid: $adapter_cid,
                dataset_ipfs_cid: $dataset_cid,
                dataset_record_count: $record_count,
                base_model: $base_model,
                lora_rank: $lora_rank,
                gpu_type: $gpu_type,
                akash_deployment_id: $deployment_id,
                training_loss_final: $loss,
                total_duration_ms: $duration_ms,
                created_at: datetime()
            })
            RETURN ft.id AS id
            """,
            {
                "id": job.id,
                "status": job.status.value,
                "adapter_cid": job.adapter_ipfs_cid,
                "dataset_cid": job.dataset_manifest.ipfs_cid if job.dataset_manifest else "",
                "record_count": job.dataset_manifest.record_count if job.dataset_manifest else 0,
                "base_model": job.hyperparams.base_model,
                "lora_rank": job.hyperparams.lora_rank,
                "gpu_type": job.gpu_type,
                "deployment_id": job.akash_deployment_id,
                "loss": job.training_loss_final,
                "duration_ms": job.total_duration_ms,
            },
        )

        self._log.info(
            "finetune_record_logged",
            record_id=job.id,
            adapter_cid=job.adapter_ipfs_cid,
        )
