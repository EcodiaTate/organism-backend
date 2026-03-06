"""
EcodiaOS — Model Hot-Swap Manager

Orchestrates live transitions of the organism's primary inference engine
to a newly trained LoRA adapter, with a strict rollback circuit breaker.

Architecture:
  1. HotSwapManager listens for MODEL_EVALUATION_PASSED events
  2. Downloads the adapter from IPFS, switches the LLM provider
  3. Enters a "probation period" where RollbackMonitor counts errors
  4. If the error rate exceeds the threshold during probation, an
     instantaneous rollback restores the previous model/API config
  5. Active model state is persisted to Neo4j so the organism remembers
     which brain it is using across reboots

The probation monitoring is tick-driven — SynapseService calls
rollback_monitor.tick() on every cognitive cycle during probation, so
there is zero additional latency from polling or timers.

This module does NOT modify the evaluation pipeline.
"""

from __future__ import annotations

import asyncio
import enum
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from datetime import datetime

    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from clients.optimized_llm import OptimizedLLMProvider
    from config import HotSwapConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("clients.model_hotswap")


# ─── Types ────────────────────────────────────────────────────────────────────


class ModelProviderType(enum.StrEnum):
    """Which class of provider is currently active."""

    API = "api"         # Anthropic, Bedrock, OpenAI (cloud)
    LOCAL = "local"     # vLLM or Ollama (self-hosted)


class SwapPhase(enum.StrEnum):
    """Lifecycle phase of a hot-swap operation."""

    IDLE = "idle"
    DOWNLOADING_ADAPTER = "downloading_adapter"
    SWITCHING_PROVIDER = "switching_provider"
    LOADING_ADAPTER = "loading_adapter"
    PROBATION = "probation"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class ActiveModelState(EOSBaseModel):
    """
    Persistent record of which model/adapter the organism is using.

    Stored as a singleton :ActiveModelState node in Neo4j. On startup,
    the organism reads this to restore its last known brain configuration.
    """

    id: str = ""
    provider_type: ModelProviderType = ModelProviderType.API
    provider_name: str = ""         # e.g. "bedrock", "vllm", "ollama"
    base_model: str = ""            # e.g. "us.anthropic.claude-3-5-haiku..."
    adapter_cid: str | None = None  # IPFS CID of active LoRA adapter
    adapter_path: str | None = None  # Local path to adapter files
    swapped_at: datetime | None = None
    previous_provider_name: str | None = None
    previous_base_model: str | None = None
    previous_adapter_cid: str | None = None


class ProbationSnapshot(EOSBaseModel):
    """Point-in-time view of probation monitoring state."""

    active: bool = False
    cycles_elapsed: int = 0
    cycles_total: int = 100
    total_errors: int = 0
    total_calls: int = 0
    error_rate: float = 0.0
    threshold: float = 0.05
    adapter_cid: str = ""


# ─── Rollback Monitor ────────────────────────────────────────────────────────


class RollbackMonitor:
    """
    Probation period circuit breaker.

    After a model hot-swap, the monitor tracks execution errors (JSON parse
    failures and Equor violations) across the first N Synapse cycles. If the
    error rate exceeds the configured threshold, it signals an immediate
    rollback.

    This is a pure data structure — it does not perform the rollback itself.
    It reports whether a rollback is needed, and the HotSwapManager acts.

    Thread-safety: all mutation happens on the asyncio event loop thread
    (called from SynapseService._on_cycle), so no locks are needed.
    """

    def __init__(
        self,
        probation_cycles: int = 100,
        error_rate_threshold: float = 0.05,
    ) -> None:
        self._probation_cycles = probation_cycles
        self._threshold = error_rate_threshold
        self._logger = logger.bind(component="rollback_monitor")

        # State
        self._active: bool = False
        self._adapter_cid: str = ""
        self._cycles_elapsed: int = 0
        self._total_errors: int = 0
        self._total_calls: int = 0
        self._started_at: float = 0.0

    # ─── Lifecycle ─────────────────────────────────────────

    def start(self, adapter_cid: str) -> None:
        """Begin a new probation period for the given adapter."""
        self._active = True
        self._adapter_cid = adapter_cid
        self._cycles_elapsed = 0
        self._total_errors = 0
        self._total_calls = 0
        self._started_at = time.monotonic()

        self._logger.info(
            "probation_started",
            adapter_cid=adapter_cid,
            probation_cycles=self._probation_cycles,
            threshold=self._threshold,
        )

    def stop(self) -> None:
        """End the probation period (either success or rollback)."""
        elapsed_s = time.monotonic() - self._started_at if self._started_at else 0.0
        self._logger.info(
            "probation_ended",
            adapter_cid=self._adapter_cid,
            cycles_elapsed=self._cycles_elapsed,
            total_errors=self._total_errors,
            total_calls=self._total_calls,
            error_rate=self.error_rate,
            elapsed_s=round(elapsed_s, 2),
        )
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    # ─── Error Recording ──────────────────────────────────

    def record_call(self, had_error: bool) -> None:
        """
        Record the outcome of an LLM call during probation.

        Called by the OptimizedLLMProvider or OutputValidator whenever
        a JSON parse error or Equor violation is detected.
        """
        if not self._active:
            return
        self._total_calls += 1
        if had_error:
            self._total_errors += 1

    # ─── Tick (called per Synapse cycle) ──────────────────

    def tick(self) -> bool:
        """
        Advance the probation cycle counter.

        Returns True if a rollback should be triggered (error rate exceeded
        threshold), False otherwise. When the probation period completes
        without exceeding the threshold, the monitor deactivates itself.
        """
        if not self._active:
            return False

        self._cycles_elapsed += 1

        # Check error rate after sufficient samples
        if self._total_calls >= 5 and self.error_rate > self._threshold:
            self._logger.warning(
                "probation_threshold_exceeded",
                adapter_cid=self._adapter_cid,
                error_rate=round(self.error_rate, 4),
                threshold=self._threshold,
                cycles_elapsed=self._cycles_elapsed,
                total_errors=self._total_errors,
                total_calls=self._total_calls,
            )
            return True

        # Probation period complete — adapter passed
        if self._cycles_elapsed >= self._probation_cycles:
            self._logger.info(
                "probation_passed",
                adapter_cid=self._adapter_cid,
                error_rate=round(self.error_rate, 4),
                total_calls=self._total_calls,
            )
            self.stop()
            return False

        return False

    # ─── Accessors ────────────────────────────────────────

    @property
    def error_rate(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._total_errors / self._total_calls

    @property
    def snapshot(self) -> ProbationSnapshot:
        return ProbationSnapshot(
            active=self._active,
            cycles_elapsed=self._cycles_elapsed,
            cycles_total=self._probation_cycles,
            total_errors=self._total_errors,
            total_calls=self._total_calls,
            error_rate=self.error_rate,
            threshold=self._threshold,
            adapter_cid=self._adapter_cid,
        )


# ─── Neo4j Persistence ───────────────────────────────────────────────────────


class ModelStateStore:
    """
    Persists the ActiveModelState to Neo4j as a singleton node.

    Uses MERGE on (:ActiveModelState {singleton: true}) to ensure
    exactly one node exists. On startup, the organism reads this
    to know which brain it was using before it shut down.
    """

    _SAVE_QUERY = """
    MERGE (m:ActiveModelState {singleton: true})
    SET m.id = $id,
        m.provider_type = $provider_type,
        m.provider_name = $provider_name,
        m.base_model = $base_model,
        m.adapter_cid = $adapter_cid,
        m.adapter_path = $adapter_path,
        m.swapped_at = datetime($swapped_at),
        m.previous_provider_name = $previous_provider_name,
        m.previous_base_model = $previous_base_model,
        m.previous_adapter_cid = $previous_adapter_cid,
        m.updated_at = datetime()
    RETURN m
    """

    _LOAD_QUERY = """
    MATCH (m:ActiveModelState {singleton: true})
    RETURN m.id AS id,
           m.provider_type AS provider_type,
           m.provider_name AS provider_name,
           m.base_model AS base_model,
           m.adapter_cid AS adapter_cid,
           m.adapter_path AS adapter_path,
           m.swapped_at AS swapped_at,
           m.previous_provider_name AS previous_provider_name,
           m.previous_base_model AS previous_base_model,
           m.previous_adapter_cid AS previous_adapter_cid
    LIMIT 1
    """

    _LOG_SWAP_QUERY = """
    CREATE (s:ModelSwapRecord {
        id: $id,
        from_provider: $from_provider,
        from_model: $from_model,
        from_adapter_cid: $from_adapter_cid,
        to_provider: $to_provider,
        to_model: $to_model,
        to_adapter_cid: $to_adapter_cid,
        reason: $reason,
        success: $success,
        error: $error,
        timestamp: datetime()
    })
    RETURN s.id AS id
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._logger = logger.bind(component="model_state_store")

    async def save(self, state: ActiveModelState) -> None:
        """Persist the current model state to Neo4j."""
        swapped_at_str = (
            state.swapped_at.isoformat() if state.swapped_at else utc_now().isoformat()
        )
        await self._neo4j.execute_write(
            self._SAVE_QUERY,
            {
                "id": state.id or new_id(),
                "provider_type": state.provider_type.value,
                "provider_name": state.provider_name,
                "base_model": state.base_model,
                "adapter_cid": state.adapter_cid,
                "adapter_path": state.adapter_path,
                "swapped_at": swapped_at_str,
                "previous_provider_name": state.previous_provider_name,
                "previous_base_model": state.previous_base_model,
                "previous_adapter_cid": state.previous_adapter_cid,
            },
        )
        self._logger.info(
            "model_state_persisted",
            provider=state.provider_name,
            adapter_cid=state.adapter_cid,
        )

    async def load(self) -> ActiveModelState | None:
        """Load the persisted model state from Neo4j. Returns None if no state exists."""
        records = await self._neo4j.execute_read(self._LOAD_QUERY)
        if not records:
            return None

        row = records[0]
        swapped_at = row.get("swapped_at")
        return ActiveModelState(
            id=row.get("id", ""),
            provider_type=ModelProviderType(row.get("provider_type", "api")),
            provider_name=row.get("provider_name", ""),
            base_model=row.get("base_model", ""),
            adapter_cid=row.get("adapter_cid"),
            adapter_path=row.get("adapter_path"),
            swapped_at=swapped_at if isinstance(swapped_at, str) else None,
            previous_provider_name=row.get("previous_provider_name"),
            previous_base_model=row.get("previous_base_model"),
            previous_adapter_cid=row.get("previous_adapter_cid"),
        )

    async def log_swap(
        self,
        from_provider: str,
        from_model: str,
        from_adapter_cid: str | None,
        to_provider: str,
        to_model: str,
        to_adapter_cid: str | None,
        reason: str,
        success: bool,
        error: str = "",
    ) -> None:
        """Create an immutable audit record of a model swap event."""
        await self._neo4j.execute_write(
            self._LOG_SWAP_QUERY,
            {
                "id": new_id(),
                "from_provider": from_provider,
                "from_model": from_model,
                "from_adapter_cid": from_adapter_cid or "",
                "to_provider": to_provider,
                "to_model": to_model,
                "to_adapter_cid": to_adapter_cid or "",
                "reason": reason,
                "success": success,
                "error": error,
            },
        )


# ─── Hot-Swap Manager ────────────────────────────────────────────────────────


class HotSwapManager:
    """
    Orchestrates live model transitions.

    Responsibilities:
      1. Listen for MODEL_EVALUATION_PASSED events
      2. Download the adapter from IPFS to local storage
      3. Switch the inference engine:
         - If currently API-based → create a new local provider (vLLM/Ollama)
         - If already local → load the new adapter onto the existing engine
      4. Enter probation mode (delegated to RollbackMonitor)
      5. On rollback → restore previous provider/adapter instantly
      6. Persist state to Neo4j after every transition

    The manager holds references to the mutable provider slot so it can
    swap the inner provider of OptimizedLLMProvider atomically.
    """

    def __init__(
        self,
        config: HotSwapConfig,
        llm: OptimizedLLMProvider,
        neo4j: Neo4jClient,
        event_bus: EventBus,
    ) -> None:
        self._config = config
        self._llm = llm
        self._neo4j = neo4j
        self._event_bus = event_bus
        self._logger = logger.bind(component="hot_swap_manager")

        # Sub-components
        self._store = ModelStateStore(neo4j)
        self._monitor = RollbackMonitor(
            probation_cycles=config.probation_cycles,
            error_rate_threshold=config.error_rate_threshold,
        )

        # State
        self._phase = SwapPhase.IDLE
        self._current_state: ActiveModelState | None = None
        self._previous_provider: LLMProvider | None = None
        self._swap_lock = asyncio.Lock()

    # ─── Lifecycle ─────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Load persisted model state from Neo4j and restore if needed.

        Called during organism startup. If a previous adapter was active,
        the manager will attempt to reload it onto the local engine.
        """
        if not self._config.enabled:
            self._logger.info("hot_swap_disabled")
            return

        from systems.synapse.types import SynapseEventType

        # Subscribe to promotion events
        self._event_bus.subscribe(
            SynapseEventType.MODEL_EVALUATION_PASSED,
            self._on_evaluation_passed,
        )

        # Load persisted state
        state = await self._store.load()
        if state is not None:
            self._current_state = state
            self._logger.info(
                "model_state_restored",
                provider=state.provider_name,
                adapter_cid=state.adapter_cid,
                base_model=state.base_model,
            )
            # If there was an active adapter, attempt to reload it
            if state.adapter_cid and state.adapter_path:
                try:
                    await self._restore_adapter(state)
                except Exception as exc:
                    self._logger.error(
                        "adapter_restore_failed",
                        adapter_cid=state.adapter_cid,
                        error=str(exc),
                    )
        else:
            # Capture initial state from current provider
            inner = self._llm.inner
            self._current_state = ActiveModelState(
                id=new_id(),
                provider_type=(
                    ModelProviderType.LOCAL
                    if inner.supports_adapters
                    else ModelProviderType.API
                ),
                provider_name=type(inner).__name__.lower().replace("provider", ""),
                base_model=getattr(inner, "_model", "unknown"),
            )

        self._logger.info("hot_swap_initialized")

    # ─── Event Handler ────────────────────────────────────

    async def _on_evaluation_passed(self, event: Any) -> None:
        """
        Handle MODEL_EVALUATION_PASSED event.

        Extracts the adapter CID and base model from the event payload
        and initiates the hot-swap sequence.
        """
        from systems.synapse.types import SynapseEvent

        if not isinstance(event, SynapseEvent):
            return

        adapter_cid = event.data.get("adapter_ipfs_cid", "")
        base_model = event.data.get("base_model", "")

        if not adapter_cid:
            self._logger.warning(
                "evaluation_passed_no_adapter_cid",
                event_data=event.data,
            )
            return

        self._logger.info(
            "evaluation_passed_received",
            adapter_cid=adapter_cid,
            base_model=base_model,
            composite_score=event.data.get("composite_score", 0),
        )

        await self.execute_hot_swap(adapter_cid, base_model)

    # ─── Core Swap Logic ──────────────────────────────────

    async def execute_hot_swap(
        self,
        adapter_cid: str,
        base_model: str,
    ) -> bool:
        """
        Execute the full hot-swap sequence.

        Returns True if the swap completed and probation started,
        False if it failed at any stage.
        """
        from systems.synapse.types import SynapseEvent, SynapseEventType

        async with self._swap_lock:
            if self._phase == SwapPhase.PROBATION:
                self._logger.warning(
                    "hot_swap_rejected_in_probation",
                    current_adapter=self._monitor.snapshot.adapter_cid,
                    requested_adapter=adapter_cid,
                )
                return False

            self._phase = SwapPhase.DOWNLOADING_ADAPTER

            # Emit start event
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.MODEL_HOT_SWAP_STARTED,
                data={
                    "adapter_cid": adapter_cid,
                    "base_model": base_model,
                    "from_provider": (
                        self._current_state.provider_name
                        if self._current_state else "unknown"
                    ),
                },
                source_system="simula",
            ))

            try:
                # Step 1: Download adapter from IPFS
                adapter_path = await self._download_adapter(adapter_cid)

                # Step 2: Switch provider or load adapter
                self._phase = SwapPhase.SWITCHING_PROVIDER
                await self._switch_engine(adapter_cid, adapter_path, base_model)

                # Step 3: Enter probation
                self._phase = SwapPhase.PROBATION
                self._monitor.start(adapter_cid)

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.MODEL_HOT_SWAP_COMPLETED,
                    data={
                        "adapter_cid": adapter_cid,
                        "base_model": base_model,
                        "phase": "probation_started",
                        "probation_cycles": self._config.probation_cycles,
                    },
                    source_system="simula",
                ))

                return True

            except Exception as exc:
                self._phase = SwapPhase.FAILED
                self._logger.error(
                    "hot_swap_failed",
                    adapter_cid=adapter_cid,
                    error=str(exc),
                )

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.MODEL_HOT_SWAP_FAILED,
                    data={
                        "adapter_cid": adapter_cid,
                        "error": str(exc),
                    },
                    source_system="simula",
                ))

                # Log the failed swap
                await self._store.log_swap(
                    from_provider=self._current_state.provider_name if self._current_state else "",
                    from_model=self._current_state.base_model if self._current_state else "",
                    from_adapter_cid=self._current_state.adapter_cid if self._current_state else None,
                    to_provider=self._config.local_inference_provider,
                    to_model=base_model,
                    to_adapter_cid=adapter_cid,
                    reason="model_evaluation_passed",
                    success=False,
                    error=str(exc),
                )

                # Restore previous provider if we partially switched
                if self._previous_provider is not None:
                    self._llm._inner = self._previous_provider
                    self._previous_provider = None
                    self._logger.info("previous_provider_restored_after_failure")

                self._phase = SwapPhase.IDLE
                return False

    async def _download_adapter(self, adapter_cid: str) -> str:
        """
        Download the LoRA adapter from IPFS to local storage.

        Returns the local filesystem path to the adapter directory.
        """
        import os
        import tempfile

        import httpx

        self._phase = SwapPhase.DOWNLOADING_ADAPTER

        # Use a stable directory per CID so reloads don't re-download
        adapter_dir = os.path.join(
            tempfile.gettempdir(), "ecodiaos_adapters", adapter_cid[:16]
        )
        adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")

        if os.path.exists(adapter_file):
            self._logger.info(
                "adapter_already_cached",
                adapter_cid=adapter_cid,
                path=adapter_dir,
            )
            return adapter_dir

        os.makedirs(adapter_dir, exist_ok=True)

        # Download from IPFS via gateway
        gateway_url = f"https://gateway.pinata.cloud/ipfs/{adapter_cid}"

        self._logger.info(
            "downloading_adapter",
            adapter_cid=adapter_cid,
            url=gateway_url,
        )

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.get(gateway_url)
            response.raise_for_status()

            with open(adapter_file, "wb") as f:
                f.write(response.content)

        self._logger.info(
            "adapter_downloaded",
            adapter_cid=adapter_cid,
            path=adapter_dir,
            size_bytes=os.path.getsize(adapter_file),
        )

        return adapter_dir

    async def _switch_engine(
        self,
        adapter_cid: str,
        adapter_path: str,
        base_model: str,
    ) -> None:
        """
        Switch the inference engine to use the new adapter.

        Two cases:
        1. Current provider is API-based (Anthropic/Bedrock/OpenAI):
           → Create a new local provider (vLLM or Ollama) and swap it in
        2. Current provider is already local (vLLM/Ollama):
           → Unload current adapter (if any) and load the new one
        """
        inner = self._llm.inner

        if inner.supports_adapters:
            # Case 2: Already on a local engine — just swap the adapter
            self._phase = SwapPhase.LOADING_ADAPTER

            # Preserve reference for rollback
            self._previous_provider = None

            # Unload existing adapter if present
            if inner.active_adapter_id is not None:
                await inner.unload_adapter()

            # Load new adapter
            await asyncio.wait_for(
                inner.load_adapter(adapter_path, adapter_cid),
                timeout=self._config.adapter_load_timeout_s,
            )
        else:
            # Case 1: API provider → switch to local
            self._previous_provider = inner

            from clients.llm import OllamaProvider, VLLMProvider

            if self._config.local_inference_provider == "vllm":
                local_provider: LLMProvider = VLLMProvider(
                    model=base_model or self._config.local_base_model,
                    endpoint=self._config.local_inference_endpoint,
                )
            else:
                local_provider = OllamaProvider(
                    model=base_model or self._config.local_base_model,
                    endpoint=self._config.local_inference_endpoint,
                )

            # Load adapter onto the new local provider
            self._phase = SwapPhase.LOADING_ADAPTER
            await asyncio.wait_for(
                local_provider.load_adapter(adapter_path, adapter_cid),
                timeout=self._config.adapter_load_timeout_s,
            )

            # Atomic swap of the inner provider
            self._llm._inner = local_provider

        # Update state
        prev = self._current_state
        self._current_state = ActiveModelState(
            id=new_id(),
            provider_type=ModelProviderType.LOCAL,
            provider_name=self._config.local_inference_provider,
            base_model=base_model or self._config.local_base_model,
            adapter_cid=adapter_cid,
            adapter_path=adapter_path,
            swapped_at=utc_now(),
            previous_provider_name=prev.provider_name if prev else None,
            previous_base_model=prev.base_model if prev else None,
            previous_adapter_cid=prev.adapter_cid if prev else None,
        )

        # Persist to Neo4j
        await self._store.save(self._current_state)

        # Log the swap
        await self._store.log_swap(
            from_provider=prev.provider_name if prev else "",
            from_model=prev.base_model if prev else "",
            from_adapter_cid=prev.adapter_cid if prev else None,
            to_provider=self._config.local_inference_provider,
            to_model=base_model,
            to_adapter_cid=adapter_cid,
            reason="model_evaluation_passed",
            success=True,
        )

        self._logger.info(
            "engine_switched",
            from_provider=prev.provider_name if prev else "unknown",
            to_provider=self._config.local_inference_provider,
            adapter_cid=adapter_cid,
        )

    async def _restore_adapter(self, state: ActiveModelState) -> None:
        """Restore a previously active adapter on startup."""
        inner = self._llm.inner
        if not inner.supports_adapters:
            self._logger.warning(
                "cannot_restore_adapter_non_local_provider",
                provider=type(inner).__name__,
            )
            return

        if state.adapter_path and state.adapter_cid:
            await inner.load_adapter(state.adapter_path, state.adapter_cid)
            self._logger.info(
                "adapter_restored",
                adapter_cid=state.adapter_cid,
            )

    # ─── Rollback ─────────────────────────────────────────

    async def execute_rollback(self, reason: str = "probation_threshold_exceeded") -> None:
        """
        Instantly revert to the previous model configuration.

        This is the organism's immune response to catastrophic forgetting —
        the moment the error rate spikes, we yank the new adapter and
        restore the proven previous brain.
        """
        from systems.synapse.types import SynapseEvent, SynapseEventType

        self._monitor.stop()

        failing_adapter = (
            self._current_state.adapter_cid if self._current_state else "unknown"
        )

        self._logger.warning(
            "executing_rollback",
            reason=reason,
            failing_adapter=failing_adapter,
            error_rate=round(self._monitor.error_rate, 4),
        )

        # Case 1: We have a stashed previous provider (swapped from API → local)
        if self._previous_provider is not None:
            inner_to_close = self._llm.inner
            self._llm._inner = self._previous_provider
            self._previous_provider = None

            # Close the local provider we're abandoning
            try:
                await inner_to_close.close()
            except Exception as exc:
                self._logger.warning("failed_to_close_rolled_back_provider", error=str(exc))

        # Case 2: We were already on a local engine — just unload the adapter
        elif self._llm.inner.supports_adapters:
            try:
                await self._llm.inner.unload_adapter()
            except Exception as exc:
                self._logger.warning("failed_to_unload_adapter", error=str(exc))

        # Restore previous state
        if self._current_state and self._current_state.previous_provider_name:
            self._current_state = ActiveModelState(
                id=new_id(),
                provider_type=(
                    ModelProviderType.LOCAL
                    if self._llm.inner.supports_adapters
                    else ModelProviderType.API
                ),
                provider_name=self._current_state.previous_provider_name or "",
                base_model=self._current_state.previous_base_model or "",
                adapter_cid=self._current_state.previous_adapter_cid,
                swapped_at=utc_now(),
            )
            await self._store.save(self._current_state)

        # Log the rollback
        await self._store.log_swap(
            from_provider=self._config.local_inference_provider,
            from_model=self._current_state.base_model if self._current_state else "",
            from_adapter_cid=failing_adapter,
            to_provider=self._current_state.provider_name if self._current_state else "unknown",
            to_model=self._current_state.base_model if self._current_state else "",
            to_adapter_cid=self._current_state.adapter_cid if self._current_state else None,
            reason=reason,
            success=True,
        )

        # Emit events
        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.MODEL_ROLLBACK_TRIGGERED,
            data={
                "failing_adapter": failing_adapter,
                "error_rate": round(self._monitor.error_rate, 4),
                "reason": reason,
                "restored_provider": (
                    self._current_state.provider_name if self._current_state else "unknown"
                ),
            },
            source_system="simula",
        ))

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CATASTROPHIC_FORGETTING_DETECTED,
            data={
                "adapter_cid": failing_adapter,
                "error_rate": round(self._monitor.error_rate, 4),
                "probation_snapshot": self._monitor.snapshot.model_dump(),
            },
            source_system="simula",
        ))

        self._phase = SwapPhase.ROLLED_BACK

        self._logger.info(
            "rollback_completed",
            restored_provider=(
                self._current_state.provider_name if self._current_state else "unknown"
            ),
        )

    # ─── Probation Tick (called by Synapse) ───────────────

    async def on_cycle(self) -> None:
        """
        Called by SynapseService._on_cycle on every theta tick.

        During probation, advances the monitor and triggers rollback
        if the error rate exceeds the threshold. This is the "circuit
        breaker" check — it runs at full clock speed (~6.7 Hz) so the
        organism reacts to catastrophic forgetting within seconds.
        """
        if not self._monitor.is_active:
            return

        should_rollback = self._monitor.tick()

        if should_rollback and self._config.auto_rollback:
            await self.execute_rollback("probation_threshold_exceeded")

    def record_inference_error(self, had_error: bool) -> None:
        """
        Record the outcome of an LLM call. Called by the OutputValidator
        or OptimizedLLMProvider when it detects a JSON parse error or
        Equor violation during probation.
        """
        self._monitor.record_call(had_error)

    # ─── Accessors ────────────────────────────────────────

    @property
    def phase(self) -> SwapPhase:
        return self._phase

    @property
    def is_in_probation(self) -> bool:
        return self._monitor.is_active

    @property
    def current_state(self) -> ActiveModelState | None:
        return self._current_state

    @property
    def rollback_monitor(self) -> RollbackMonitor:
        return self._monitor

    @property
    def probation_snapshot(self) -> ProbationSnapshot:
        return self._monitor.snapshot

    async def health(self) -> dict[str, Any]:
        """Health report for the hot-swap subsystem."""
        return {
            "enabled": self._config.enabled,
            "phase": self._phase.value,
            "probation": self._monitor.snapshot.model_dump(),
            "current_model": (
                {
                    "provider": self._current_state.provider_name,
                    "base_model": self._current_state.base_model,
                    "adapter_cid": self._current_state.adapter_cid,
                }
                if self._current_state
                else None
            ),
        }
