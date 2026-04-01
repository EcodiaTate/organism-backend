"""
EcodiaOS - InstanceAdapterRegistry

Tracks which LoRA adapters this instance has available and which one is
currently loaded by the inference server.

Architecture
------------
- Owned by Axon because Axon is the action-execution layer: it decides which
  adapter to use when submitting a prompt to the local Reasoning Engine.
- Persists adapter metadata to Neo4j for audit trail and cross-restart continuity.
- Emits ADAPTER_LOAD_REQUESTED on Synapse when the active adapter changes, so
  the inference server (out-of-process) can hot-swap the LoRA weights.
- Never deletes adapters - only accumulates them (constraint from the spec).

Adapter precedence
------------------
When Axon submits a request:
1. If the current intent has a domain tag, load the domain-specific adapter.
2. Otherwise, use the primary (generalist) adapter.
3. Fall back to "genesis" (base model, no adapter) if nothing else is available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from systems.synapse.service import SynapseService

logger = logging.getLogger(__name__)


class InstanceAdapterRegistry:
    """
    Per-instance registry of available LoRA adapters.

    Lifecycle
    ---------
    1. `AxonService` creates an `InstanceAdapterRegistry` on boot.
    2. `initialize()` loads persisted adapter paths from Neo4j.
    3. `ContinualLearningOrchestrator.on_training_complete()` calls
       `register_domain_adapter()` when a new adapter is ready.
    4. `AxonService._prepare_inference_context()` calls `load_for_domain()`
       before every inference call (no-op if domain is already active).
    """

    def __init__(self, instance_id: str) -> None:
        self._instance_id = instance_id
        # The base generalist adapter (set after first incremental training)
        self.primary_adapter: str = "genesis"
        # domain → S3/local path of the best adapter for that domain
        self.domain_adapters: dict[str, str] = {}
        # Currently loaded adapter path (used by inference server)
        self.effective_adapter: str = "genesis"
        self._synapse: SynapseService | None = None
        self._neo4j: Any | None = None

    # ── Dependency injection ─────────────────────────────────────────────────

    def set_synapse(self, synapse: SynapseService) -> None:
        self._synapse = synapse

    def set_neo4j(self, driver: Any) -> None:
        self._neo4j = driver

    # ── Bootstrap ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load persisted adapter registrations from Neo4j (best-effort)."""
        if self._neo4j is None:
            return
        try:
            async with self._neo4j.session() as session:
                result = await session.run(
                    """
                    MATCH (a:LoRAAdapter {instance_id: $iid, status: 'ready'})
                    RETURN a.domain AS domain, a.s3_path AS path, a.is_primary AS primary
                    ORDER BY a.created_at DESC
                    """,
                    iid=self._instance_id,
                )
                seen_domains: set[str] = set()
                async for record in result:
                    domain = record["domain"]
                    path = record["path"] or ""
                    if not path:
                        continue
                    if domain not in seen_domains:
                        seen_domains.add(domain)
                        if domain == "generalist":
                            self.primary_adapter = path
                        else:
                            self.domain_adapters[domain] = path
                # Default effective adapter to generalist
                if self.primary_adapter != "genesis":
                    self.effective_adapter = self.primary_adapter
        except Exception:
            logger.warning(
                "InstanceAdapterRegistry: could not restore from Neo4j",
                exc_info=True,
            )

    # ── Runtime operations ────────────────────────────────────────────────────

    async def load_for_domain(self, domain: str) -> None:
        """
        Switch to the best adapter for the given domain.

        If a domain-specific adapter is registered, prefer it.
        Otherwise fall back to the generalist primary.
        No-op if the correct adapter is already loaded.
        """
        target = self.domain_adapters.get(domain) or self.primary_adapter
        if target == self.effective_adapter:
            return

        self.effective_adapter = target
        self._emit_load_request(target, domain)

    async def register_domain_adapter(
        self, domain: str, adapter_path: str
    ) -> None:
        """
        Register a newly trained adapter.
        Called by ContinualLearningOrchestrator.on_training_complete().
        Persists to Neo4j and promotes the effective adapter if domain is active.
        """
        if domain == "generalist":
            self.primary_adapter = adapter_path
        else:
            self.domain_adapters[domain] = adapter_path

        await self._persist_adapter(domain, adapter_path)
        logger.info(
            "InstanceAdapterRegistry: registered adapter domain=%s path=%s",
            domain,
            adapter_path,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _emit_load_request(self, adapter_path: str, domain: str) -> None:
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            self._synapse.event_bus.broadcast(
                SynapseEvent(
                    event_type=SynapseEventType.ADAPTER_LOAD_REQUESTED
                    if hasattr(
                        SynapseEventType, "ADAPTER_LOAD_REQUESTED"
                    )
                    else SynapseEventType.ADAPTER_TRAINING_STARTED,  # fallback
                    source_system="axon",
                    data={
                        "instance_id": self._instance_id,
                        "adapter_path": adapter_path,
                        "domain": domain,
                    },
                )
            )
        except Exception:
            logger.debug(
                "InstanceAdapterRegistry: could not emit load request for domain=%s",
                domain,
            )

    async def _persist_adapter(self, domain: str, adapter_path: str) -> None:
        if self._neo4j is None:
            return
        try:
            async with self._neo4j.session() as session:
                await session.run(
                    """
                    MATCH (a:LoRAAdapter {instance_id: $iid, domain: $domain})
                    WHERE a.s3_path IS NULL OR a.s3_path = ''
                    SET a.s3_path   = $path,
                        a.status    = 'ready',
                        a.ready_at  = datetime()
                    """,
                    iid=self._instance_id,
                    domain=domain,
                    path=adapter_path,
                )
        except Exception:
            logger.warning(
                "InstanceAdapterRegistry: Neo4j persist failed for domain=%s",
                domain,
                exc_info=True,
            )
