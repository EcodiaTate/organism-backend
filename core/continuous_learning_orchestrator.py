"""
EcodiaOS - ContinualLearningOrchestrator

Schedules and orchestrates LoRA adapter training for the Reasoning Engine.

Responsibilities
----------------
- Decides when and how to retrain: generalist (INCREMENTAL) or domain-specific
  (DOMAIN_SPECIALIZED) based on SpecializationTracker profiles.
- Fetches training examples (all or domain-filtered) from the RE export store.
- Builds a DomainCurriculum to order examples correctly.
- Submits training jobs to RunPod and tracks their status.
- Registers completed adapters in the InstanceAdapterRegistry so Axon can
  hot-swap them per-domain at inference time.
- Emits ADAPTER_TRAINING_STARTED / ADAPTER_TRAINING_COMPLETE on Synapse.

Architecture
------------
- Lives in `backend/core/` - started as a supervised background task in
  `registry.py` Phase 12 (alongside the existing re_training_exporter task).
- Reads from `RETrainingExporter` to get current batch data; never stores its
  own copy of training examples.
- Communicates with SpecializationTracker (Nova) via direct method call (same
  process) - not via Synapse, to avoid serialisation overhead.
- Does NOT import from any system directly.  Nova/Axon injectables are
  provided via setter methods.

Training cadence
----------------
- Generalist (INCREMENTAL): every 24 h, if ≥ 50 new gold examples arrived.
- Domain-specialised: every 48 h per domain, if ≥ 100 domain examples and
  success_rate > 0.70.
- GENESIS: only on explicit call (first boot, hard reset).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from primitives.evolution import AdapterStrategy, DomainProfile

if TYPE_CHECKING:
    from core.re_training_exporter import RETrainingExporter
    from systems.nova.specialization_tracker import SpecializationTracker
    from systems.synapse.service import SynapseService
    from systems.axon.adapter_registry import InstanceAdapterRegistry

logger = logging.getLogger(__name__)

# Thresholds
_INCREMENTAL_MIN_NEW_EXAMPLES = 50
_INCREMENTAL_INTERVAL_H = 24
_DOMAIN_MIN_EXAMPLES = 100
_DOMAIN_MIN_SUCCESS_RATE = 0.70
_DOMAIN_INTERVAL_H = 48

_EXPORT_DIR = Path(os.getenv("RE_TRAINING_EXPORT_DIR", "data/re_training_batches"))
_S3_BUCKET = os.getenv("RE_TRAINING_S3_BUCKET", "ecodiaos-re-training")
_ADAPTERS_S3_PREFIX = "adapters/"


class ContinualLearningOrchestrator:
    """
    Orchestrates continual LoRA adapter training for the Reasoning Engine.

    Injected dependencies (call setters before start()):
    - set_exporter()            - RETrainingExporter for raw example access
    - set_specialization_tracker() - SpecializationTracker from Nova
    - set_adapter_registry()    - InstanceAdapterRegistry from Axon
    - set_synapse()             - SynapseService for event emission
    - set_neo4j()               - Neo4j driver for LoRAAdapter audit nodes
    """

    def __init__(self, instance_id: str) -> None:
        self._instance_id = instance_id
        self._exporter: RETrainingExporter | None = None
        self._specialization: SpecializationTracker | None = None
        self._adapter_registry: InstanceAdapterRegistry | None = None
        self._synapse: SynapseService | None = None
        self._neo4j: Any | None = None

        # domain → {job_id, domain, started_at, num_examples, strategy}
        self._training_jobs: dict[str, dict[str, Any]] = {}
        # domain → last training time
        self._last_trained: dict[str, datetime] = {}
        self._last_incremental: datetime | None = None
        self._running = False

    # ── Dependency injection ────────────────────────────────────────────────

    def set_exporter(self, exporter: RETrainingExporter) -> None:
        self._exporter = exporter

    def set_specialization_tracker(self, tracker: SpecializationTracker) -> None:
        self._specialization = tracker

    def set_adapter_registry(self, registry: InstanceAdapterRegistry) -> None:
        self._adapter_registry = registry

    def set_synapse(self, synapse: SynapseService) -> None:
        self._synapse = synapse

    def set_neo4j(self, driver: Any) -> None:
        self._neo4j = driver

    # ── Background loop ─────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Supervised background task.  Runs forever; call via asyncio.create_task."""
        self._running = True
        logger.info("ContinualLearningOrchestrator: starting loop")
        while self._running:
            try:
                await self._training_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "ContinualLearningOrchestrator: unhandled error in training cycle"
                )
            await asyncio.sleep(3600)   # check every hour

    async def stop(self) -> None:
        self._running = False

    # ── Training cycle ──────────────────────────────────────────────────────

    async def _training_cycle(self) -> None:
        """
        Decide whether to train generalist or domain-specific adapter(s).
        Multiple domain adapters may be trained in a single cycle if several
        domains have matured simultaneously.
        """
        profiles = (
            self._specialization.get_all_profiles()
            if self._specialization
            else {}
        )

        # 1. Check domain-specialized adapters first
        for domain, profile in profiles.items():
            if await self._should_train_domain(domain, profile):
                await self._train_domain_adapter(
                    domain, AdapterStrategy.DOMAIN_SPECIALIZED
                )

        # 2. Check generalist incremental
        if await self._should_train_incremental():
            await self._train_domain_adapter("generalist", AdapterStrategy.INCREMENTAL)

    async def _should_train_domain(
        self, domain: str, profile: DomainProfile
    ) -> bool:
        if profile.examples_trained < _DOMAIN_MIN_EXAMPLES:
            return False
        if profile.success_rate < _DOMAIN_MIN_SUCCESS_RATE:
            return False
        last = self._last_trained.get(domain)
        if last:
            elapsed_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            if elapsed_h < _DOMAIN_INTERVAL_H:
                return False
        if domain in self._training_jobs:
            return False  # already in-flight
        return True

    async def _should_train_incremental(self) -> bool:
        if "generalist" in self._training_jobs:
            return False
        if self._last_incremental:
            elapsed_h = (
                datetime.now(timezone.utc) - self._last_incremental
            ).total_seconds() / 3600
            if elapsed_h < _INCREMENTAL_INTERVAL_H:
                return False
        if self._exporter is None:
            return False
        stats = self._exporter.stats
        new_examples = stats.get("total_exported", 0) - stats.get(
            "last_incremental_baseline", 0
        )
        return new_examples >= _INCREMENTAL_MIN_NEW_EXAMPLES

    # ── Adapter training ─────────────────────────────────────────────────────

    async def _train_domain_adapter(
        self, domain: str, strategy: AdapterStrategy
    ) -> None:
        """
        Full pipeline:
        1. Collect examples (domain-filtered or all).
        2. Build curriculum.
        3. Export to JSONL.
        4. Determine base adapter.
        5. Submit RunPod job.
        6. Record in tracking state + Neo4j.
        7. Emit ADAPTER_TRAINING_STARTED.
        """
        from core.curriculum_builder import DomainCurriculum

        logger.info(
            "ContinualLearningOrchestrator: starting adapter training",
            extra={"domain": domain, "strategy": strategy},
        )

        # 1. Collect examples
        all_examples = await self._collect_training_examples(domain)
        if not all_examples:
            logger.warning(
                "ContinualLearningOrchestrator: no examples for domain=%s - skip",
                domain,
            )
            return

        # 2. Build curriculum
        curriculum = DomainCurriculum(domain)
        await curriculum.build(all_examples)
        ordered = curriculum.get_ordered_examples()

        if not ordered:
            logger.warning(
                "ContinualLearningOrchestrator: curriculum empty for domain=%s",
                domain,
            )
            return

        # 3. Export to JSONL
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_path = _EXPORT_DIR / f"domain_training_{domain}_{ts}.jsonl"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w") as fh:
            for ex in ordered:
                fh.write(
                    json.dumps(
                        {
                            "instruction": ex.instruction,
                            "input": ex.input_context,
                            "output": ex.output,
                            "domain": ex.domain,
                            "skill_area": ex.skill_area,
                            "domain_difficulty": ex.domain_difficulty,
                            "outcome_quality": ex.outcome_quality,
                            "category": ex.category,
                        }
                    )
                    + "\n"
                )

        # 4. Base adapter
        base_adapter = await self._get_base_adapter(domain, strategy)

        # 5. Submit job (RunPod SDK - best-effort; gracefully skip if unavailable)
        job_id = await self._submit_training_job(
            training_file=str(export_path),
            domain=domain,
            strategy=strategy,
            previous_adapter=base_adapter,
            num_examples=len(ordered),
        )

        if not job_id:
            logger.warning(
                "ContinualLearningOrchestrator: RunPod job submission failed for "
                "domain=%s - will retry next cycle",
                domain,
            )
            return

        # 6. Track
        now = datetime.now(timezone.utc)
        self._training_jobs[domain] = {
            "job_id": job_id,
            "domain": domain,
            "started_at": now.isoformat(),
            "num_examples": len(ordered),
            "strategy": str(strategy),
            "base_adapter": base_adapter,
            "export_path": str(export_path),
        }
        if domain == "generalist":
            self._last_incremental = now
        else:
            self._last_trained[domain] = now

        await self._persist_lora_adapter_node(
            domain=domain,
            strategy=strategy,
            job_id=job_id,
            num_examples=len(ordered),
            base_adapter=base_adapter,
        )

        # 7. Emit event
        self._emit(
            "ADAPTER_TRAINING_STARTED",
            {
                "domain": domain,
                "strategy": str(strategy),
                "job_id": job_id,
                "num_examples": len(ordered),
                "base_adapter": base_adapter,
                "instance_id": self._instance_id,
            },
        )

    # ── RunPod integration ───────────────────────────────────────────────────

    async def _submit_training_job(
        self,
        training_file: str,
        domain: str,
        strategy: AdapterStrategy,
        previous_adapter: str | None,
        num_examples: int,
    ) -> str | None:
        """
        Submit a LoRA training job to RunPod.
        Returns the job_id string, or None on failure.

        The RunPod serverless endpoint is expected to accept:
        {
            "input": {
                "training_file": "<s3 or local path>",
                "domain": "<domain>",
                "strategy": "<strategy>",
                "previous_adapter_path": "<s3 path or null>",
                "output_adapter_path": "<s3 destination>",
            }
        }
        """
        try:
            import runpod  # type: ignore[import]

            output_path = (
                f"s3://{_S3_BUCKET}/{_ADAPTERS_S3_PREFIX}"
                f"{self._instance_id}/{domain}/"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )

            response = runpod.run_sync(
                endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID", ""),
                job_input={
                    "training_file": training_file,
                    "domain": domain,
                    "strategy": str(strategy),
                    "previous_adapter_path": previous_adapter,
                    "output_adapter_path": output_path,
                    "num_examples": num_examples,
                    "instance_id": self._instance_id,
                },
            )
            return response.get("id")
        except ImportError:
            logger.warning(
                "ContinualLearningOrchestrator: runpod SDK not available - "
                "recording job as local-only"
            )
            # Return a synthetic job_id for local/dev environments
            return f"local_{domain}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        except Exception:
            logger.exception(
                "ContinualLearningOrchestrator: RunPod submission error"
            )
            return None

    async def on_training_complete(
        self,
        job_id: str,
        domain: str,
        adapter_path: str,
        eval_loss: float | None = None,
    ) -> None:
        """
        Called by the RunPod webhook handler when a job succeeds.

        Registers the adapter in InstanceAdapterRegistry and emits
        ADAPTER_TRAINING_COMPLETE so downstream systems can react.
        """
        # Remove in-flight marker
        self._training_jobs.pop(domain, None)

        if self._adapter_registry:
            await self._adapter_registry.register_domain_adapter(
                domain=domain, adapter_path=adapter_path
            )

        self._emit(
            "ADAPTER_TRAINING_COMPLETE",
            {
                "domain": domain,
                "strategy": str(
                    AdapterStrategy.INCREMENTAL
                    if domain == "generalist"
                    else AdapterStrategy.DOMAIN_SPECIALIZED
                ),
                "job_id": job_id,
                "adapter_path": adapter_path,
                "instance_id": self._instance_id,
                "eval_loss": eval_loss,
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _collect_training_examples(self, domain: str) -> list:
        """
        Return all cached training examples from the RE exporter.
        For domain != "generalist" returns all (curriculum.build filters later).
        """
        if self._exporter is None:
            return []
        # RETrainingExporter exposes accumulated datapoints via its internal buffer
        try:
            return list(self._exporter._buffer.values())  # type: ignore[attr-defined]
        except Exception:
            return []

    async def _get_base_adapter(
        self, domain: str, strategy: AdapterStrategy
    ) -> str | None:
        """Return the S3 path of the most recent adapter to use as base."""
        if strategy == AdapterStrategy.GENESIS:
            return None  # train from base model
        if self._adapter_registry is None:
            return None
        if strategy == AdapterStrategy.DOMAIN_SPECIALIZED:
            # For domain adapters, chain from the latest generalist adapter
            generalist = self._adapter_registry.primary_adapter
            return generalist if generalist != "genesis" else None
        # INCREMENTAL: chain from whatever is currently loaded
        eff = self._adapter_registry.effective_adapter
        return eff if eff != "genesis" else None

    async def _persist_lora_adapter_node(
        self,
        domain: str,
        strategy: AdapterStrategy,
        job_id: str,
        num_examples: int,
        base_adapter: str | None,
    ) -> None:
        if self._neo4j is None:
            return
        try:
            async with self._neo4j.session() as session:
                await session.run(
                    """
                    CREATE (a:LoRAAdapter {
                        id:             $adapter_id,
                        instance_id:    $iid,
                        domain:         $domain,
                        strategy:       $strategy,
                        runpod_job_id:  $job_id,
                        examples_count: $num_examples,
                        base_adapter:   $base_adapter,
                        status:         'training',
                        created_at:     datetime()
                    })
                    """,
                    adapter_id=f"{self._instance_id}_{domain}_{job_id}",
                    iid=self._instance_id,
                    domain=domain,
                    strategy=str(strategy),
                    job_id=job_id,
                    num_examples=num_examples,
                    base_adapter=base_adapter or "",
                )
        except Exception:
            logger.warning(
                "ContinualLearningOrchestrator: Neo4j adapter node persist failed",
                exc_info=True,
            )

    def _emit(self, event_type_name: str, data: dict) -> None:
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            self._synapse.event_bus.broadcast(
                SynapseEvent(
                    event_type=SynapseEventType(event_type_name.lower()),
                    source_system="core",
                    data=data,
                )
            )
        except Exception:
            logger.debug(
                "ContinualLearningOrchestrator: emit failed for %s", event_type_name
            )
