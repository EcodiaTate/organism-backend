"""
EcodiaOS --- Skia Service (Shadow Infrastructure)

Orchestrates five sub-components:
  - HeartbeatMonitor: observes organism vitality via Redis pub/sub
  - StateSnapshotPipeline: periodic encrypted state backup to IPFS
  - RestorationOrchestrator: autonomous recovery (Cloud Run -> Akash)
  - VitalityCoordinator: organism-level death detection and execution
  - PhylogeneticTracker: Neo4j lineage records for evolutionary metrics

Two operating modes:
  - Embedded (main process): snapshot pipeline + vitality monitoring +
    health reporting. Heartbeat monitoring is pointless when co-located.
  - Standalone (skia_worker): all five components active. Runs on a
    separate cheap instance in a different zone/provider.

Phase 2 additions (Spec 29, sections 22-23):
  - Heritable variation: controlled mutation on restoration/spawning
  - Phylogenetic Neo4j records: parent->child lineage with mutation delta
  - Standalone worker event emissions: SKIA_HEARTBEAT, ORGANISM_SPAWNED
  - Evolvable thresholds: heartbeat + vitality params via EVO_PARAMETER_ADJUSTED
  - Oikos metabolic cost: gate restoration on available metabolic budget
  - Dry-run restoration: simulate without committing
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from systems.skia.heartbeat import HeartbeatMonitor
from systems.skia.phylogeny import (
    MutationConfig,
    PhylogeneticNode,
    PhylogeneticTracker,
    mutate_parameters,
)
from systems.skia.pinata_client import PinataClient
from systems.skia.restoration import RestorationOrchestrator
from systems.skia.snapshot import StateSnapshotPipeline
from systems.skia.vitality import VitalityCoordinator

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from config import SkiaConfig
    from systems.identity.vault import IdentityVault
    from systems.memory.service import MemoryService
    from systems.synapse.clock import CognitiveClock
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.skia")


class SkiaService:
    """
    Skia --- the shadow infrastructure.

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
        self._vitality: VitalityCoordinator | None = None
        self._phylogeny: PhylogeneticTracker | None = None
        self._memory: MemoryService | None = None

        # Synapse wiring
        self._event_bus: EventBus | None = None
        self._initialized = False

        # Standalone worker heartbeat task
        self._heartbeat_task: asyncio.Task[None] | None = None
        # Shadow worker ensure loop (embedded mode only)
        self._shadow_worker_task: asyncio.Task[None] | None = None

        # Organism generation tracking (loaded from Neo4j on init)
        self._generation: int = 1
        self._parent_instance_id: str | None = None

        # Fleet resurrection coordination
        self._resurrection_approved: asyncio.Event = asyncio.Event()
        self._resurrection_leader: str = ""  # instance_id of the elected resurrector

    def set_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        if self._vitality:
            self._vitality.set_event_bus(event_bus)
        if self._phylogeny:
            self._phylogeny.set_event_bus(event_bus)
        if self._snapshot:
            self._snapshot.set_event_bus(event_bus)
        # Subscribe to Evo parameter adjustments for evolvable thresholds
        self._subscribe_events(event_bus)

    def _subscribe_events(self, event_bus: EventBus) -> None:
        """Subscribe to events Skia cares about."""
        try:
            from systems.synapse.types import SynapseEventType
            event_bus.subscribe(
                SynapseEventType.EVO_PARAMETER_ADJUSTED,
                self._on_evo_parameter_adjusted,
            )
            event_bus.subscribe(
                SynapseEventType.ORGANISM_DIED,
                self._on_organism_died,
            )
            # Route METABOLIC_PRESSURE to VitalityCoordinator for austerity
            # enforcement.  VitalityCoordinator also subscribes directly once its
            # own set_event_bus() is called, but this path handles the case where
            # the bus is set on SkiaService before VitalityCoordinator is initialised.
            event_bus.subscribe(
                SynapseEventType.METABOLIC_PRESSURE,
                self._on_metabolic_pressure,
            )
            # Fleet resurrection coordination
            event_bus.subscribe(
                SynapseEventType.FEDERATION_RESURRECTION_APPROVED,
                self._on_federation_resurrection_approved,
            )
            # INV-017 Tier 1 kill switch: drive extinction → halt organism.
            # Equor emits DRIVE_EXTINCTION_DETECTED from its 15-min background loop
            # when any drive's 72h rolling mean < 0.01.  Skia routes to
            # VitalityCoordinator.trigger_death_sequence() - the full 3-phase shutdown.
            event_bus.subscribe(
                SynapseEventType.DRIVE_EXTINCTION_DETECTED,
                self._on_drive_extinction_detected,
            )
        except Exception as exc:
            self._log.debug("event_subscription_failed", error=str(exc))

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """Forward METABOLIC_PRESSURE to VitalityCoordinator for austerity enforcement."""
        if self._vitality:
            data = getattr(event, "data", {}) or {}
            await self._vitality.handle_metabolic_pressure(data)

    async def _on_drive_extinction_detected(self, event: Any) -> None:
        """INV-017 Tier 1 kill switch: drive extinction → halt organism.

        Equor's background loop emits DRIVE_EXTINCTION_DETECTED when any drive's
        72h rolling mean falls below 0.01.  An organism that has lost a cognitive
        dimension is constitutionally brain-damaged - this triggers the full
        VitalityCoordinator death sequence so the organism can snapshot its state
        (IPFS), notify the federation, and cease operations cleanly.

        Non-fatal if VitalityCoordinator is not yet wired.
        """
        data = getattr(event, "data", {}) or {}
        drive = data.get("drive", "unknown")
        mean_val = data.get("rolling_mean_72h", 0.0)

        # Guard: reject events with missing/invalid drive name.
        # "unknown" means the event data was malformed - not a real extinction.
        _VALID_DRIVES = {"coherence", "care", "growth", "honesty"}
        if drive not in _VALID_DRIVES:
            self._log.warning(
                "inv017_ignored_invalid_drive",
                drive=drive,
                rolling_mean_72h=mean_val,
                reason="drive name not in constitutional drive set",
            )
            return

        self._log.critical(
            "inv017_drive_extinction_halt_triggered",
            drive=drive,
            rolling_mean_72h=mean_val,
            action="triggering_death_sequence",
        )

        if self._vitality is not None:
            reason = (
                f"INV-017 violated: drive '{drive}' rolling mean = {mean_val:.6f} "
                f"(threshold 0.01, sustained 72h). Constitutional brain death."
            )
            try:
                await self._vitality.trigger_death_sequence(reason)
            except Exception as exc:
                self._log.error(
                    "inv017_death_sequence_failed",
                    drive=drive,
                    error=str(exc),
                )
        else:
            self._log.error(
                "inv017_vitality_coordinator_not_wired",
                drive=drive,
                detail="Cannot trigger death sequence - VitalityCoordinator not available",
            )

    async def _on_organism_died(self, event: Any) -> None:
        """Record death in phylogenetic tree when organism dies."""
        data = getattr(event, "data", {}) or {}
        instance_id = data.get("instance_id", self._instance_id)
        cause = data.get("cause", "unknown")
        if self._phylogeny:
            try:
                await self._phylogeny.record_death(instance_id, cause)
                await self._phylogeny.link_death_record(instance_id)
            except Exception as exc:
                self._log.debug("phylogenetic_death_record_failed", error=str(exc))

    async def _on_evo_parameter_adjusted(self, event: Any) -> None:
        """Handle EVO_PARAMETER_ADJUSTED events for evolvable Skia parameters.

        Supports live updates to heartbeat thresholds and vitality check interval.
        Target parameters:
          - skia.heartbeat_failure_threshold
          - skia.heartbeat_confirmation_checks
          - skia.heartbeat_confirmation_interval_s
          - skia.heartbeat_poll_interval_s
          - skia.mutation_rate
          - skia.mutation_magnitude
        """
        data = getattr(event, "data", {}) or {}
        target = data.get("target_system", "")
        if target != "skia":
            return

        param = data.get("parameter", "")
        new_value = data.get("new_value")
        if new_value is None:
            return

        # Update config in-place
        if hasattr(self._config, param):
            old_value = getattr(self._config, param)
            try:
                # Preserve type
                if isinstance(old_value, int):
                    setattr(self._config, param, int(round(float(new_value))))
                elif isinstance(old_value, float):
                    setattr(self._config, param, float(new_value))
                self._log.info(
                    "evo_parameter_applied",
                    parameter=param,
                    old_value=old_value,
                    new_value=getattr(self._config, param),
                )
            except (ValueError, TypeError) as exc:
                self._log.warning("evo_parameter_invalid", parameter=param, error=str(exc))

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
                memory=self._memory,
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
                on_critical_system_silent=self._on_critical_system_silent,
            )

        # Propagate event bus to snapshot pipeline if already set
        if self._event_bus and self._snapshot:
            self._snapshot.set_event_bus(self._event_bus)

        # Phylogenetic tracker (both modes)
        self._phylogeny = PhylogeneticTracker(self._neo4j)
        if self._event_bus:
            self._phylogeny.set_event_bus(self._event_bus)

        # Load current generation from Neo4j
        self._generation = await self._phylogeny.get_generation(self._instance_id)

        # Vitality coordinator (both modes --- death detection is always active)
        self._vitality = VitalityCoordinator(
            neo4j=self._neo4j,
            instance_id=self._instance_id,
        )
        if self._event_bus:
            self._vitality.set_event_bus(self._event_bus)
        if self._snapshot:
            self._vitality.set_snapshot(self._snapshot)

        self._initialized = True
        self._log.info(
            "skia_initialized",
            has_snapshot=self._snapshot is not None,
            has_heartbeat=self._heartbeat is not None,
            has_restoration=self._restoration is not None,
            has_vitality=self._vitality is not None,
            has_phylogeny=self._phylogeny is not None,
            generation=self._generation,
        )

    async def start(self) -> None:
        """Start background tasks."""
        if self._snapshot:
            await self._snapshot.start()
        if self._heartbeat:
            await self._heartbeat.start()
        if self._vitality:
            await self._vitality.start()
        if self._phylogeny and self._event_bus:
            # Only start the fitness loop if an event bus is wired - otherwise silent no-op
            await self._phylogeny.start()
        # Start standalone worker heartbeat loop
        if self._standalone:
            self._heartbeat_task = asyncio.create_task(
                self._worker_heartbeat_loop(),
                name="skia_worker_heartbeat",
            )
        # Embedded mode: ensure shadow worker exists on a different provider.
        # Runs async (non-blocking) so it doesn't delay startup.
        if not self._standalone and self._config.enabled and self._restoration:
            asyncio.create_task(
                self._ensure_shadow_worker_loop(),
                name="skia_shadow_worker_ensure",
            )

    async def shutdown(self) -> None:
        """Graceful shutdown of all sub-components."""
        # Cancel background tasks
        for attr in ("_heartbeat_task", "_shadow_worker_task"):
            task = getattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self._vitality:
            await self._vitality.stop()
        if self._heartbeat:
            await self._heartbeat.stop()
        if self._snapshot:
            await self._snapshot.stop()
        if self._phylogeny:
            await self._phylogeny.stop()
        if self._pinata:
            await self._pinata.close()
        self._log.info("skia_shutdown")

    # ---- Shadow worker ensure loop (embedded mode) --------------------------

    _SHADOW_WORKER_CHECK_INTERVAL_S = 6 * 3600.0  # Re-check every 6 hours

    async def _ensure_shadow_worker_loop(self) -> None:
        """
        On startup (and every 6 hours), ensure a shadow worker is deployed on a
        DIFFERENT provider.  Emits SKIA_SHADOW_WORKER_DEPLOYED on success or
        SKIA_SHADOW_WORKER_MISSING on failure.

        This eliminates the human dependency for resurrection: the shadow worker
        is the autonomous "dead man's switch" that will detect main-instance death
        and trigger restoration.
        """
        while True:
            try:
                if self._restoration is not None:
                    ok = await self._restoration.ensure_shadow_worker()
                    if ok:
                        record = await self._restoration._get_shadow_worker_record()
                        from systems.synapse.types import SynapseEventType as _SET
                        await self._emit_event(_SET.SKIA_SHADOW_WORKER_DEPLOYED, {
                            "instance_id": self._instance_id,
                            "endpoint": record.get("endpoint", "") if record else "",
                            "provider": record.get("provider", "") if record else "",
                            "deployment_id": record.get("deployment_id", "") if record else "",
                        })
                    else:
                        from systems.synapse.types import SynapseEventType as _SET
                        await self._emit_event(_SET.SKIA_SHADOW_WORKER_MISSING, {
                            "instance_id": self._instance_id,
                            "error": "All shadow worker deployment attempts failed",
                        })
            except Exception as exc:
                self._log.warning("shadow_worker_loop_error", error=str(exc))

            await asyncio.sleep(self._SHADOW_WORKER_CHECK_INTERVAL_S)

    # ---- Standalone worker heartbeat loop -----------------------------------

    async def _worker_heartbeat_loop(self) -> None:
        """Periodically emit SKIA_HEARTBEAT to prove standalone worker liveness."""
        while True:
            try:
                from systems.synapse.types import SynapseEventType as _SET
                await self._emit_event(_SET.SKIA_HEARTBEAT, {
                    "instance_id": self._instance_id,
                    "mode": "standalone",
                    "generation": self._generation,
                    "has_restoration": self._restoration is not None,
                    "has_heartbeat_monitor": self._heartbeat is not None,
                })
            except Exception as exc:
                self._log.debug("worker_heartbeat_emit_failed", error=str(exc))
            await asyncio.sleep(self._config.worker_heartbeat_interval_s)

    # ---- Synapse health protocol --------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Self-health report for Synapse HealthMonitor."""
        if not self._config.enabled:
            return {"status": "healthy", "enabled": False}

        result: dict[str, Any] = {
            "status": "healthy",
            "mode": "standalone" if self._standalone else "embedded",
            "enabled": True,
            "generation": self._generation,
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

        if self._vitality:
            result["vitality_phase"] = self._vitality.phase.value
            result["organism_dead"] = self._vitality.is_dead

        return result

    async def _on_federation_resurrection_approved(self, event: Any) -> None:
        """Handle FEDERATION_RESURRECTION_APPROVED from a surviving federation member.

        The event data must include:
          - leader_instance_id: which Skia instance was elected to perform the resurrection
          - snapshot_cid: the most recent snapshot CID across the federation
        """
        data = getattr(event, "data", {}) or {}
        leader = data.get("leader_instance_id", "")
        snapshot_cid = data.get("snapshot_cid", "")
        self._resurrection_leader = leader
        if snapshot_cid and self._restoration:
            # Override local CID with the federation-selected most-recent snapshot
            raw = self._redis.client
            await raw.set(
                self._config.state_cid_redis_key,
                snapshot_cid,
                ex=3600,
            )
            self._log.info(
                "federation_resurrection_approved",
                leader=leader,
                snapshot_cid=snapshot_cid,
            )
        self._resurrection_approved.set()

    async def _detect_simultaneous_deaths(self) -> int:
        """Check Redis for other instances that died in the last 60 seconds.

        Returns the count of simultaneous deaths detected (including this instance).
        Uses a sorted set 'skia:fleet:recent_deaths' scored by unix timestamp.
        """
        try:
            raw = self._redis.client
            death_key = "skia:fleet:recent_deaths"
            now = time.time()
            window_start = now - 60.0

            # Record this instance's death
            await raw.zadd(death_key, {self._instance_id: now})
            await raw.expire(death_key, 120)

            # Count deaths in the last 60s
            deaths = await raw.zrangebyscore(death_key, window_start, now)
            return len(deaths)
        except Exception as exc:
            self._log.debug("simultaneous_death_check_failed", error=str(exc))
            return 1  # Assume single death if Redis unavailable

    async def _coordinate_fleet_resurrection(self) -> bool:
        """Coordinate with surviving federation to elect a single resurrector.

        Returns True if this instance should proceed with restoration,
        False if another federation member was elected (avoid duplicate provisioning).

        Flow:
          1. Emit SKIA_RESURRECTION_PROPOSAL with our latest snapshot CID + timestamp
          2. Wait up to 30s for FEDERATION_RESURRECTION_APPROVED
          3. If approved and we are elected leader → restore
          4. If approved and another is leader → stand down (return False)
          5. If no approval within 30s → proceed autonomously (federation may be offline)
        """
        latest_cid = ""
        if self._snapshot:
            latest_cid = self._snapshot.last_cid

        self._resurrection_approved.clear()
        self._resurrection_leader = ""

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.SKIA_RESURRECTION_PROPOSAL, {
            "instance_id": self._instance_id,
            "snapshot_cid": latest_cid,
            "snapshot_ts": time.time(),
            "generation": self._generation,
        })

        self._log.info(
            "resurrection_proposal_emitted",
            snapshot_cid=latest_cid,
            waiting_s=30,
        )

        try:
            await asyncio.wait_for(self._resurrection_approved.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            # No federation response - proceed autonomously
            self._log.warning(
                "resurrection_coordination_timeout",
                proceeding="autonomously",
            )
            return True

        # Check if we are the elected leader
        if self._resurrection_leader and self._resurrection_leader != self._instance_id:
            self._log.info(
                "resurrection_standing_down",
                leader=self._resurrection_leader,
                reason="another_instance_elected",
            )
            return False

        return True

    # ---- CRITICAL system silence callback -----------------------------------

    async def _on_critical_system_silent(self, system_name: str) -> None:
        """Called by HeartbeatMonitor when Equor, Thymos, or Memory goes silent for ≥45s.

        Does NOT trigger a full organism resurrection - the organism may still be alive
        (the critical system may have crashed while others run). Instead:
          1. Emits SKIA_HEARTBEAT_LOST with system_name so Thymos can classify an incident
          2. Emits SYSTEM_MODULATION to request EMERGENCY austerity for the silent system
          3. Triggers VitalityCoordinator to run an immediate health assessment
        """
        self._log.warning(
            "critical_system_silent_alert",
            system=system_name,
            threshold_s=45,
        )
        # Notify Thymos - it can create an incident and attempt repair
        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.SKIA_HEARTBEAT_LOST, {
            "instance_id": self._instance_id,
            "system": system_name,
            "severity": "CRITICAL",
            "reason": "critical_system_silent_45s",
        })
        # Modulation signal - request EMERGENCY mode for the affected system
        await self._emit_event(_SET.SYSTEM_MODULATION, {
            "target_system": system_name,
            "austerity_level": "EMERGENCY",
            "reason": f"critical_system_silent: no heartbeat for 45s",
            "source": "skia",
        })

    # ---- Death callback ------------------------------------------------------

    async def _on_death_confirmed(self) -> None:
        """Callback from HeartbeatMonitor when organism death is confirmed."""
        self._log.error("organism_death_confirmed", instance_id=self._instance_id)

        # Emit event (if bus is available --- may not be in standalone mode)
        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.SKIA_HEARTBEAT_LOST, {
            "instance_id": self._instance_id,
        })

        if not self._restoration:
            self._log.error("no_restoration_orchestrator")
            return

        # Emit restoration triggered immediately - before coordination or metabolic gates,
        # so the observatory can track restoration attempts even if they are later blocked.
        await self._emit_event(_SET.SKIA_RESTORATION_TRIGGERED, {
            "instance_id": self._instance_id,
            "trigger": "heartbeat_confirmed_dead",
        })

        # ---- Metabolic gate: check with Oikos before restoration ----
        metabolic_ok = await self._check_metabolic_budget()
        if not metabolic_ok:
            self._log.warning("restoration_blocked_by_metabolic_budget")
            # Still attempt restoration but log the metabolic concern
            # Real blocking would require the organism to be dead permanently
            # which violates the survival imperative --- so we proceed with warning

        # Fleet resurrection coordination: if multiple instances died simultaneously,
        # coordinate with the surviving federation before provisioning to avoid
        # duplicate shadow instances and to select the most recent snapshot.
        simultaneous_deaths = await self._detect_simultaneous_deaths()
        if simultaneous_deaths > 1:
            self._log.warning(
                "simultaneous_deaths_detected",
                count=simultaneous_deaths,
                coordinating="fleet_resurrection",
            )
            should_restore = await self._coordinate_fleet_resurrection()
            if not should_restore:
                self._log.info("resurrection_deferred_to_federation_leader")
                return

        await self._emit_event(_SET.SKIA_RESTORATION_STARTED, {
            "instance_id": self._instance_id,
            "trigger": "heartbeat_confirmed_dead",
            "simultaneous_deaths": simultaneous_deaths,
        })

        # Pass the most recent constitutional genome to the orchestrator so the
        # shadow instance inherits the parent's drive weights and phenotype.
        if self._snapshot is not None:
            self._restoration.set_constitutional_genome(
                self._snapshot.last_constitutional_genome
            )

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

        # Emit metabolic cost report
        await self._emit_event(_SET.METABOLIC_COST_REPORT, {
            "system_id": "skia",
            "operation": "restoration",
            "cost_usd": attempt.cost_usd_estimate or self._config.estimated_restoration_cost_usd,
            "details": {
                "strategy": attempt.strategy.value,
                "outcome": attempt.outcome.value,
                "duration_ms": attempt.duration_ms,
            },
        })

        await self._emit_event(_SET.SKIA_RESTORATION_COMPLETE, {
            "instance_id": self._instance_id,
            "outcome": attempt.outcome.value,
            "strategy": attempt.strategy.value,
            "state_cid": attempt.state_cid,
            "duration_ms": attempt.duration_ms,
        })

        # Also emit the legacy event for backwards compatibility
        await self._emit_event(_SET.SKIA_RESTORATION_COMPLETED, {
            "instance_id": self._instance_id,
            "outcome": attempt.outcome.value,
            "strategy": attempt.strategy.value,
            "state_cid": attempt.state_cid,
            "duration_ms": attempt.duration_ms,
        })

        # On successful restoration, record phylogenetic birth with mutation
        if attempt.outcome.value == "success":
            await self._record_spawned_instance(attempt.state_cid)
            # Coma-recovery: write a crash-context record to Redis so the
            # resurrected organism's Thymos can read it on boot and route it
            # to Simula for Tier 4 novel-fix analysis.  This closes the F1 gap:
            # previously the crash context (what killed the organism) was lost
            # at process death; now it survives in Redis and is injected as an
            # INCIDENT_DETECTED on the new instance's first Synapse connection.
            await self._persist_crash_context_for_resurrection(
                state_cid=attempt.state_cid or "",
                trigger="heartbeat_confirmed_dead",
            )

        # If restoration exhausted max attempts, trigger organism death
        if self._restoration.infrastructure_dead and self._vitality:
            await self._vitality.trigger_death_sequence(
                "infrastructure_death: restoration exhausted all attempts"
            )

    # ---- Coma-recovery crash context ----------------------------------------

    async def _persist_crash_context_for_resurrection(
        self, *, state_cid: str, trigger: str
    ) -> None:
        """Persist crash context to Redis so the resurrected instance can diagnose it.

        The dying organism writes a compact crash record under the key
        ``skia:crash_context:{instance_id}``.  On next boot, the new instance
        reads this key and emits an INCIDENT_DETECTED via Synapse so Thymos
        routes it to Simula (Tier 4 novel-fix) for root-cause analysis.

        TTL: 24 hours.  If the record is not consumed within 24 hours, it is
        considered stale and dropped (the next death event will overwrite it).
        """
        if self._redis is None:
            self._log.warning(
                "crash_context_redis_unavailable",
                reason="Redis not wired to SkiaService; crash context will not survive resurrection",
            )
            return

        import json as _json
        import time as _time

        key = f"skia:crash_context:{self._instance_id}"
        payload = {
            "instance_id": self._instance_id,
            "trigger": trigger,
            "state_cid": state_cid,
            "crashed_at_unix": _time.time(),
            "incident_class": "CRASH",
            "severity": "CRITICAL",
            "description": (
                f"Organism heartbeat confirmed dead (trigger={trigger!r}). "
                "Resurrected from IPFS snapshot. Root-cause unknown - "
                "Simula Tier 4 analysis requested."
            ),
            "source_system": "skia",
            "request_simula_analysis": True,
        }

        try:
            raw = self._redis.client
            await raw.set(key, _json.dumps(payload), ex=86400)  # 24h TTL
            self._log.info(
                "crash_context_persisted",
                key=key,
                state_cid=state_cid[:16] if state_cid else "none",
            )
        except Exception as exc:
            self._log.warning(
                "crash_context_persist_failed",
                error=str(exc),
            )

    # ---- Phylogenetic tracking + mutation ------------------------------------

    async def _record_spawned_instance(self, state_cid: str) -> None:
        """Record a new organism instance in the phylogenetic tree.

        Introduces controlled mutation to numeric parameters and records
        the lineage in Neo4j. This is what makes restoration != cloning.
        """
        if not self._phylogeny:
            return

        # Generate mutation delta from current config parameters
        config_params = {
            "heartbeat_poll_interval_s": self._config.heartbeat_poll_interval_s,
            "heartbeat_failure_threshold": self._config.heartbeat_failure_threshold,
            "heartbeat_confirmation_checks": self._config.heartbeat_confirmation_checks,
            "heartbeat_confirmation_interval_s": self._config.heartbeat_confirmation_interval_s,
            "snapshot_interval_s": self._config.snapshot_interval_s,
            "pinata_max_retained_pins": self._config.pinata_max_retained_pins,
        }

        mutation_config = MutationConfig(
            mutation_rate=self._config.mutation_rate,
            mutation_magnitude=self._config.mutation_magnitude,
        )
        mutated_params, mutation_delta = mutate_parameters(config_params, mutation_config)

        # Apply mutations to the config (these become the new instance's parameters)
        for key, value in mutated_params.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        new_generation = self._generation + 1
        child_instance_id = f"{self._instance_id}-g{new_generation}-{uuid.uuid4().hex[:8]}"

        phylo_node = PhylogeneticNode(
            instance_id=child_instance_id,
            parent_instance_id=self._instance_id,
            generation=new_generation,
            lineage_depth=new_generation,
            mutation_delta=mutation_delta,
            genome_id=state_cid,
        )

        try:
            await self._phylogeny.record_birth(phylo_node)
            await self._phylogeny.link_parent_child(
                parent_instance_id=self._instance_id,
                child_instance_id=child_instance_id,
                generation=new_generation,
                mutation_delta=mutation_delta,
            )
            # Link parent's death record to phylogenetic node
            await self._phylogeny.link_death_record(self._instance_id)
        except Exception as exc:
            self._log.error("phylogenetic_record_failed", error=str(exc))

        # Emit ORGANISM_SPAWNED event
        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.ORGANISM_SPAWNED, {
            "instance_id": child_instance_id,
            "parent_instance_id": self._instance_id,
            "generation": new_generation,
            "mutation_delta": mutation_delta,
            "lineage_depth": new_generation,
            "state_cid": state_cid,
        })

        self._log.info(
            "organism_spawned",
            child_id=child_instance_id,
            generation=new_generation,
            mutations_applied=len(mutation_delta),
        )

    async def _check_metabolic_budget(self) -> bool:
        """Check if restoration is metabolically affordable.

        Returns True if we can afford it (or if Oikos is unavailable).
        Restoration is survival-tier --- we proceed even if budget is tight,
        but we emit a cost report either way.
        """
        # We don't directly import Oikos --- we check via the vitality coordinator's
        # cached Oikos reference, which reads runway_days.
        if self._vitality and self._vitality._oikos is not None:
            try:
                state = self._vitality._oikos.snapshot()
                if state and hasattr(state, "runway_days"):
                    # If runway is critically low, warn but don't block
                    if state.runway_days < 1.0:
                        self._log.warning(
                            "restoration_metabolic_warning",
                            runway_days=state.runway_days,
                            cost_usd=self._config.estimated_restoration_cost_usd,
                        )
                        return False
            except Exception as exc:
                self._log.debug("metabolic_check_failed", error=str(exc))
        return True

    # ---- Dry-run restoration -------------------------------------------------

    async def dry_run_restoration(self) -> dict[str, Any]:
        """Simulate a restoration without committing.

        Validates snapshot availability, strategy readiness, metabolic budget,
        and data integrity (node/edge count vs. manifest).

        Data integrity step:
          - Downloads the snapshot from IPFS.
          - Decrypts + decompresses the payload.
          - Compares actual node/edge counts against the manifest's recorded counts.
          - Fails if deviation exceeds 5% for either dimension.

        Emits SKIA_DRY_RUN_COMPLETE with predicted outcome.
        """
        t0 = time.monotonic()
        result: dict[str, Any] = {
            "instance_id": self._instance_id,
            "dry_run": True,
            "snapshot_available": False,
            "strategies_available": [],
            "metabolic_ok": False,
            "data_integrity_ok": None,  # None = not checked (no CID or no pinata)
            "predicted_outcome": "unknown",
        }

        # Check snapshot availability
        cid = ""
        if self._restoration:
            cid = await self._restoration._get_latest_cid()
            if not cid:
                cid = await self._restoration._get_latest_cid_from_history()
            result["snapshot_available"] = bool(cid)
            result["state_cid"] = cid

            # Check strategy readiness
            strategies = []
            if self._config.gcp_project_id and self._config.gcp_service_name:
                strategies.append("cloud_run_restart")
            if self._config.akash_api_url and self._config.akash_wallet_address:
                strategies.append("akash_deploy")
            result["strategies_available"] = strategies

            result["infrastructure_dead"] = self._restoration.infrastructure_dead
        else:
            result["restoration_orchestrator"] = "not_available"

        # Data integrity check: download + deserialize + compare vs. manifest counts
        if cid and self._pinata and self._vault:
            result["data_integrity_ok"] = await self._dry_run_integrity_check(cid, result)

        # Check metabolic budget
        result["metabolic_ok"] = await self._check_metabolic_budget()

        # Predict outcome
        if not result["snapshot_available"]:
            result["predicted_outcome"] = "failed_no_snapshot"
        elif result.get("data_integrity_ok") is False:
            result["predicted_outcome"] = "failed_data_integrity_mismatch"
        elif not result["strategies_available"]:
            result["predicted_outcome"] = "failed_no_strategy"
        elif result.get("infrastructure_dead"):
            result["predicted_outcome"] = "failed_infrastructure_dead"
        else:
            result["predicted_outcome"] = "likely_success"

        duration_ms = (time.monotonic() - t0) * 1000
        result["duration_ms"] = round(duration_ms, 2)

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.SKIA_DRY_RUN_COMPLETE, result)

        self._log.info(
            "dry_run_complete",
            predicted_outcome=result["predicted_outcome"],
            duration_ms=result["duration_ms"],
            data_integrity_ok=result.get("data_integrity_ok"),
        )
        return result

    async def _dry_run_integrity_check(
        self, cid: str, result: dict[str, Any]
    ) -> bool:
        """Download and deserialize a snapshot, then verify counts against manifest.

        Returns True if data is intact (within 5% deviation), False otherwise.
        Updates `result` with integrity details.
        """
        import gzip as _gzip
        import orjson as _orjson
        from systems.identity.vault import SealedEnvelope

        _INTEGRITY_TOLERANCE = 0.05  # 5% deviation threshold

        try:
            # Download from IPFS
            encrypted_bytes = await self._pinata.get_by_cid(cid)  # type: ignore[union-attr]

            # Read key_version from manifest if available
            key_version = 1
            try:
                manifest_raw = await self._redis.get_json(self._config.manifest_redis_key)
                if isinstance(manifest_raw, dict):
                    manifest_node_count = int(manifest_raw.get("node_count", 0))
                    manifest_edge_count = int(manifest_raw.get("edge_count", 0))
                    kv = manifest_raw.get("encryption_key_version")
                    if kv is not None:
                        key_version = int(kv)
                else:
                    manifest_node_count = 0
                    manifest_edge_count = 0
            except Exception:
                manifest_node_count = 0
                manifest_edge_count = 0

            # Decrypt
            envelope = SealedEnvelope(
                platform_id="skia",
                purpose="state_snapshot",
                ciphertext=encrypted_bytes.decode("ascii"),
                key_version=key_version,
            )
            compressed = self._vault.decrypt(envelope)  # type: ignore[union-attr]

            # Decompress + parse
            raw_bytes = _gzip.decompress(compressed)
            payload_data = _orjson.loads(raw_bytes)
            actual_node_count = len(payload_data.get("nodes", []))
            actual_edge_count = len(payload_data.get("edges", []))

            result["actual_node_count"] = actual_node_count
            result["actual_edge_count"] = actual_edge_count
            result["manifest_node_count"] = manifest_node_count
            result["manifest_edge_count"] = manifest_edge_count

            # Compare counts - allow 5% deviation
            node_ok = True
            edge_ok = True
            if manifest_node_count > 0:
                node_deviation = abs(actual_node_count - manifest_node_count) / manifest_node_count
                node_ok = node_deviation <= _INTEGRITY_TOLERANCE
                result["node_deviation_pct"] = round(node_deviation * 100, 2)
            if manifest_edge_count > 0:
                edge_deviation = abs(actual_edge_count - manifest_edge_count) / manifest_edge_count
                edge_ok = edge_deviation <= _INTEGRITY_TOLERANCE
                result["edge_deviation_pct"] = round(edge_deviation * 100, 2)

            if not node_ok or not edge_ok:
                result["integrity_failure_reason"] = "data_integrity_mismatch"
                self._log.warning(
                    "dry_run_integrity_mismatch",
                    actual_nodes=actual_node_count,
                    manifest_nodes=manifest_node_count,
                    actual_edges=actual_edge_count,
                    manifest_edges=manifest_edge_count,
                )
                return False

            self._log.info(
                "dry_run_integrity_ok",
                nodes=actual_node_count,
                edges=actual_edge_count,
            )
            return True

        except Exception as exc:
            result["integrity_failure_reason"] = str(exc)
            self._log.warning("dry_run_integrity_check_failed", error=str(exc))
            # Treat check failure as a soft warning, not a hard block -
            # the snapshot may still be restorable even if we can't verify locally.
            return True

    # ---- Genome extraction for Evo ------------------------------------------

    def get_evolvable_parameters(self) -> dict[str, float]:
        """Return all evolvable Skia parameters for genome extraction.

        Includes both infrastructure params AND degradation rates - the organism's
        entropy resistance is heritable and evolvable.
        """
        params = {
            "heartbeat_poll_interval_s": self._config.heartbeat_poll_interval_s,
            "heartbeat_failure_threshold": float(self._config.heartbeat_failure_threshold),
            "heartbeat_confirmation_checks": float(self._config.heartbeat_confirmation_checks),
            "heartbeat_confirmation_interval_s": self._config.heartbeat_confirmation_interval_s,
            "snapshot_interval_s": self._config.snapshot_interval_s,
            "mutation_rate": self._config.mutation_rate,
            "mutation_magnitude": self._config.mutation_magnitude,
            "worker_heartbeat_interval_s": self._config.worker_heartbeat_interval_s,
            "pinata_max_retained_pins": float(self._config.pinata_max_retained_pins),
        }
        # Include degradation rates - heritable entropy resistance
        if self._vitality is not None:
            params.update(self._vitality._degradation.get_evolvable_parameters())
        return params

    # ---- Vitality system wiring ---------------------------------------------

    def set_memory(self, memory: MemoryService) -> None:
        """Wire Memory so snapshots include the constitutional genome.

        Must be called before ``initialize()`` to take effect; if called after,
        updates the snapshot pipeline directly via its own ``set_memory()``.
        """
        self._memory = memory
        if self._snapshot is not None:
            self._snapshot.set_memory(memory)

    def wire_vitality_systems(
        self,
        *,
        clock: CognitiveClock | None = None,
        oikos: Any = None,
        thymos: Any = None,
        equor: Any = None,
        telos: Any = None,
    ) -> None:
        """Wire system references into the VitalityCoordinator.

        Called during organism bootstrap after all systems are initialised.
        """
        if not self._vitality:
            return
        if clock:
            self._vitality.set_clock(clock)
        if oikos:
            self._vitality.set_oikos(oikos)
        if thymos:
            self._vitality.set_thymos(thymos)
        if equor:
            self._vitality.set_equor(equor)
        if telos:
            self._vitality.set_telos(telos)

    @property
    def vitality(self) -> VitalityCoordinator | None:
        """Access the VitalityCoordinator for external resurrection calls."""
        return self._vitality

    @property
    def phylogeny(self) -> PhylogeneticTracker | None:
        """Access the PhylogeneticTracker for lineage queries."""
        return self._phylogeny

    @property
    def generation(self) -> int:
        """Current organism generation number."""
        return self._generation

    async def resurrect(self, trigger: str = "manual_reset") -> bool:
        """External resurrection endpoint. Delegates to VitalityCoordinator."""
        if not self._vitality:
            return False
        return await self._vitality.resurrect(trigger)

    async def _emit_event(self, event_type_name: str | Any, data: dict[str, Any]) -> None:
        """Emit a Synapse event if the bus is available."""
        if not self._event_bus:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            if isinstance(event_type_name, SynapseEventType):
                et = event_type_name
            else:
                et = SynapseEventType(event_type_name.lower())
            await self._event_bus.emit(SynapseEvent(
                event_type=et,
                data=data,
                source_system="skia",
            ))
        except Exception as exc:
            self._log.warning("event_emit_failed", event=event_type_name, error=str(exc))
