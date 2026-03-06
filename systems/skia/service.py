"""
EcodiaOS — Skia Service (Shadow Infrastructure)

Orchestrates three sub-components:
  - HeartbeatMonitor: observes organism vitality via Redis pub/sub
  - StateSnapshotPipeline: periodic encrypted state backup to IPFS
  - RestorationOrchestrator: autonomous recovery (Cloud Run → Akash)

Two operating modes:
  - Embedded (main process): snapshot pipeline + health reporting only.
    Heartbeat monitoring is pointless when co-located with the organism.
  - Standalone (skia_worker): all three components active. Runs on a
    separate cheap instance in a different zone/provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.skia.heartbeat import HeartbeatMonitor
from systems.skia.pinata_client import PinataClient
from systems.skia.restoration import RestorationOrchestrator
from systems.skia.snapshot import StateSnapshotPipeline

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from config import SkiaConfig
    from systems.identity.vault import IdentityVault
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.skia")


class SkiaService:
    """
    Skia — the shadow infrastructure.

    Registered with Synapse for health monitoring. Emits events on
    snapshot completion, heartbeat loss, and restoration attempts.
    """

    system_id: str = "skia"

    def __init__(
        self,
        config: SkiaConfig,
        neo4j: Neo4jClient,
        redis: RedisClient,
        vault: IdentityVault | None = None,
        instance_id: str = "eos-default",
        standalone: bool = False,
    ) -> None:
        self._config = config
        self._neo4j = neo4j
        self._redis = redis
        self._vault = vault
        self._instance_id = instance_id
        self._standalone = standalone
        self._log = logger.bind(system="skia", mode="standalone" if standalone else "embedded")

        # Sub-components (initialised lazily)
        self._pinata: PinataClient | None = None
        self._heartbeat: HeartbeatMonitor | None = None
        self._snapshot: StateSnapshotPipeline | None = None
        self._restoration: RestorationOrchestrator | None = None

        # Synapse wiring
        self._event_bus: EventBus | None = None
        self._initialized = False

    def set_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def initialize(self) -> None:
        """Initialise sub-components based on mode and configuration."""
        if not self._config.enabled:
            self._log.info("skia_disabled")
            self._initialized = True
            return

        # Pinata client (both modes)
        if self._config.pinata_jwt:
            self._pinata = PinataClient(
                api_url=self._config.pinata_api_url,
                gateway_url=self._config.pinata_gateway_url,
                jwt=self._config.pinata_jwt,
            )
            await self._pinata.connect()
            self._log.info("pinata_connected")
        else:
            self._log.warning("pinata_jwt_not_configured", snapshots="disabled")

        # Snapshot pipeline (both modes, requires vault + pinata)
        if self._vault and self._pinata:
            self._snapshot = StateSnapshotPipeline(
                neo4j=self._neo4j,
                vault=self._vault,
                pinata=self._pinata,
                redis=self._redis,
                config=self._config,
                instance_id=self._instance_id,
            )
        elif self._vault is None:
            self._log.warning("vault_not_available", snapshots="disabled")

        # Heartbeat + Restoration (standalone worker only)
        if self._standalone:
            self._restoration = RestorationOrchestrator(
                config=self._config,
                redis=self._redis,
                pinata=self._pinata,
            )
            self._heartbeat = HeartbeatMonitor(
                redis=self._redis.client,
                config=self._config,
                on_death_confirmed=self._on_death_confirmed,
            )

        self._initialized = True
        self._log.info(
            "skia_initialized",
            has_snapshot=self._snapshot is not None,
            has_heartbeat=self._heartbeat is not None,
            has_restoration=self._restoration is not None,
        )

    async def start(self) -> None:
        """Start background tasks."""
        if self._snapshot:
            await self._snapshot.start()
        if self._heartbeat:
            await self._heartbeat.start()

    async def shutdown(self) -> None:
        """Graceful shutdown of all sub-components."""
        if self._heartbeat:
            await self._heartbeat.stop()
        if self._snapshot:
            await self._snapshot.stop()
        if self._pinata:
            await self._pinata.close()
        self._log.info("skia_shutdown")

    # ── Synapse health protocol ───────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Self-health report for Synapse HealthMonitor."""
        if not self._config.enabled:
            return {"status": "healthy", "enabled": False}

        result: dict[str, Any] = {
            "status": "healthy",
            "mode": "standalone" if self._standalone else "embedded",
            "enabled": True,
        }

        if self._heartbeat:
            state = self._heartbeat.state
            result["heartbeat_status"] = state.status.value
            result["consecutive_misses"] = state.consecutive_misses
            result["total_deaths_detected"] = state.total_deaths_detected
            result["total_false_positives"] = state.total_false_positives

        if self._snapshot:
            result["last_snapshot_cid"] = self._snapshot.last_cid
            result["snapshots_taken"] = self._snapshot.total_snapshots

        if self._pinata:
            result["pinata_connected"] = True

        return result

    # ── Death callback ────────────────────────────────────────────

    async def _on_death_confirmed(self) -> None:
        """Callback from HeartbeatMonitor when organism death is confirmed."""
        self._log.error("organism_death_confirmed", instance_id=self._instance_id)

        # Emit event (if bus is available — may not be in standalone mode)
        await self._emit_event("SKIA_HEARTBEAT_LOST", {
            "instance_id": self._instance_id,
        })

        if not self._restoration:
            self._log.error("no_restoration_orchestrator")
            return

        await self._emit_event("SKIA_RESTORATION_TRIGGERED", {
            "instance_id": self._instance_id,
            "trigger": "heartbeat_confirmed_dead",
        })

        attempt = await self._restoration.restore(
            trigger_reason="heartbeat_confirmed_dead"
        )

        self._log.info(
            "restoration_complete",
            outcome=attempt.outcome.value,
            strategy=attempt.strategy.value,
            duration_ms=round(attempt.duration_ms, 1),
            error=attempt.error or None,
        )

        await self._emit_event("SKIA_RESTORATION_COMPLETED", {
            "instance_id": self._instance_id,
            "outcome": attempt.outcome.value,
            "strategy": attempt.strategy.value,
            "state_cid": attempt.state_cid,
            "duration_ms": attempt.duration_ms,
        })

    async def _emit_event(self, event_type_name: str, data: dict[str, Any]) -> None:
        """Emit a Synapse event if the bus is available."""
        if not self._event_bus:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            et = SynapseEventType(event_type_name.lower())
            await self._event_bus.emit(SynapseEvent(
                event_type=et,
                data=data,
                source_system="skia",
            ))
        except Exception as exc:
            self._log.warning("event_emit_failed", event=event_type_name, error=str(exc))
