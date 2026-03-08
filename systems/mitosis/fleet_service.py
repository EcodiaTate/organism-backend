"""
EcodiaOS -- Mitosis Fleet Service (Spec 26)

The active fleet management layer that ties together genome inheritance,
mutation, lifecycle events, health timeouts, rescue transfers, horizontal
gene transfer, speciation detection, and population dynamics.

This service is wired into Synapse and runs background tasks. It does NOT
directly import other system internals -- all cross-system communication
goes through the Synapse event bus.

Implements:
  - Child lifecycle event emission (Priority Fix #1)
  - Genome extraction + mutation in spawn path (Priority Fix #4, Spec Gap #12)
  - 24-hour health timeout death trigger (Priority Fix #3)
  - Rescue transfer execution (Priority Fix #2)
  - Horizontal gene transfer (Speciation Gap #2)
  - Speciation event detection (Speciation Gap #3)
  - Organism event subscriptions (Priority Fix #10)
  - Dynamic population cap (Priority Fix #20)
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import structlog

from primitives.common import SystemID, utc_now
from primitives.genome import GenomeExtractionProtocol, OrganismGenome
from systems.mitosis.genome_distance import GenomeDistance, GenomeDistanceCalculator
from systems.mitosis.mutation import MutationOperator
from primitives.mitosis import ChildPosition, ChildStatus

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from clients.wallet import WalletClient
    from config import OikosConfig
    from systems.mitosis.genome_orchestrator import GenomeOrchestrator
    from systems.mitosis.spawner import LocalDockerSpawner
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()


class MitosisFleetService:
    """
    Active fleet management: lifecycle events, health monitoring, rescue,
    gene transfer, speciation detection, and dynamic population caps.

    Wired into the Synapse event bus for all cross-system communication.
    """

    def __init__(
        self,
        *,
        config: OikosConfig,
        event_bus: EventBus | None = None,
        genome_orchestrator: GenomeOrchestrator | None = None,
        neo4j: Neo4jClient | None = None,
        wallet: WalletClient | None = None,
        spawner: LocalDockerSpawner | None = None,
        instance_id: str = "",
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._genome_orch = genome_orchestrator
        self._neo4j = neo4j
        self._wallet = wallet
        self._spawner = spawner
        self._instance_id = instance_id
        self._mutation_op = MutationOperator(
            mutation_rate=config.mitosis_mutation_rate,
            neo4j=neo4j,
        )
        self._distance_calculator = GenomeDistanceCalculator(
            speciation_threshold=config.mitosis_speciation_distance_threshold,
        )
        self._log = logger.bind(system="mitosis.fleet_service")
        self._health_check_task: asyncio.Task[None] | None = None
        self._dividend_task: asyncio.Task[None] | None = None
        self._fleet_eval_task: asyncio.Task[None] | None = None
        # Callable returning current list[ChildPosition] — injected at wire time
        self._get_children: Callable[[], list[Any]] | None = None
        # Callable(EconomicState) -> FleetMetrics for monthly evaluation
        self._run_fleet_evaluation: Callable[..., Any] | None = None
        # Callable returning current EconomicState snapshot
        self._get_state: Callable[[], Any] | None = None
        # Callable(EconomicState) -> None — checks blacklisted children for decommission
        self._check_decommission_fn: Callable[..., Any] | None = None
        # Genome systems registry -- populated by the organism's boot sequence
        self._genome_systems: dict[SystemID, GenomeExtractionProtocol] = {}
        # Locally mirrored blacklist — populated from CHILD_BLACKLISTED events
        # Used to gate dividends, rescue, and seed transfers
        self._blacklisted_children: set[str] = set()
        # Last known starvation level — updated by OIKOS_METABOLIC_SNAPSHOT handler
        self._last_starvation_level: str = "nominal"
        # Service start time — used to enforce 24h maturity gate before spawning
        self._service_start_time: float = time.monotonic()
        # Reproductive fitness background task
        self._reproductive_fitness_task: asyncio.Task[None] | None = None
        # AdapterSharer — injected via set_adapter_sharer(); None = sharing disabled
        self._adapter_sharer: Any | None = None
        # Optional callable returning the current slow adapter path from CLO
        self._get_adapter_path_fn: Any | None = None
        # Fleet genome cache: instance_id → serialised genome dict (from CHILD_SPAWNED)
        # Used to supply genome_a/genome_b to can_exchange_genetic_material() and
        # AdapterShareRequest without importing other system internals.
        self._fleet_genome_cache: dict[str, dict[str, Any]] = {}

    # -- Genome system registration -----------------------------------------

    def register_genome_system(
        self, sys_id: SystemID, system: GenomeExtractionProtocol,
    ) -> None:
        """Register a system that participates in genome extraction."""
        self._genome_systems[sys_id] = system
        self._log.debug("genome_system_registered", system_id=str(sys_id))

    def wire_oikos_callbacks(
        self,
        *,
        get_children: Callable[[], list[Any]],
        get_state: Callable[[], Any],
        run_fleet_evaluation: Callable[..., Any] | None = None,
        check_decommission: Callable[..., Any] | None = None,
    ) -> None:
        """Wire Oikos service callbacks for schedulers (call at startup)."""
        self._get_children = get_children
        self._get_state = get_state
        self._run_fleet_evaluation = run_fleet_evaluation
        self._check_decommission_fn = check_decommission

    def set_adapter_sharer(
        self,
        sharer: Any,
        get_adapter_path_fn: Any | None = None,
    ) -> None:
        """Inject AdapterSharer for cross-instance LoRA merging.

        Parameters
        ----------
        sharer:
            ``AdapterSharer`` instance from ``systems.reasoning_engine.adapter_sharing``.
        get_adapter_path_fn:
            Optional zero-arg callable that returns the current slow adapter path
            (e.g. ``lambda: app.state.continual_learning._sure.production_adapter_path``).
            Required for the merge to use the local adapter as the requester side.
            Defaults to empty string (merge will only use the partner's adapter).
        """
        self._adapter_sharer = sharer
        self._get_adapter_path_fn = get_adapter_path_fn
        self._log.info(
            "adapter_sharer_wired",
            adapter_path_fn_provided=get_adapter_path_fn is not None,
        )

    # ===================================================================
    # Task 2: Genome extraction + mutation for spawn path
    # ===================================================================

    async def prepare_child_genome(
        self,
        *,
        parent_generation: int = 1,
        parent_genome_id: str | None = None,
        fitness: float = 0.0,
        seed: int | None = None,
    ) -> OrganismGenome | None:
        """
        Extract parent genome, apply mutation, persist, return child genome.

        Called from MitosisEngine.build_seed_config() AFTER fitness check,
        BEFORE child spawn.
        """
        if self._genome_orch is None:
            self._log.warning("prepare_child_genome_no_orchestrator")
            return None

        if not self._genome_systems:
            self._log.warning("prepare_child_genome_no_systems_registered")
            return None

        # Step 1: Extract parent genome
        parent_genome = await self._genome_orch.extract_full_genome(
            systems=self._genome_systems,
            generation=parent_generation,
            parent_genome_id=parent_genome_id,
            fitness=fitness,
        )

        # Step 2: Apply mutation
        child_genome, mutation_record = await self._mutation_op.mutate(
            parent_genome, seed=seed,
        )
        child_genome.instance_id = self._instance_id

        # Step 3: Persist child genome via orchestrator (Neo4j)
        if self._genome_orch._neo4j is not None:
            await self._genome_orch._persist_genome(child_genome)

        self._log.info(
            "child_genome_prepared",
            parent_genome_id=parent_genome.id,
            child_genome_id=child_genome.id,
            generation=child_genome.generation,
            mutations=len(mutation_record.mutations_applied),
        )

        try:
            from primitives.common import DriveAlignmentVector as _DAV
            # Genome preparation: growth = fitness signal, care = positive (life creation)
            _genome_alignment = _DAV(
                growth=round(min(1.0, max(-1.0, fitness * 2.0 - 1.0)), 3),
                care=round(min(1.0, 0.3 + fitness * 0.5), 3),
                coherence=round(min(1.0, 1.0 - len(mutation_record.mutations_applied) * 0.05), 3),
                honesty=0.0,
            )
        except Exception:
            _genome_alignment = None
        await self._emit_re_training(
            episode_id=child_genome.id,
            instruction=(
                "A child genome has been extracted from the parent and mutated for the "
                f"next generation (generation {child_genome.generation}). Evaluate the "
                "genetic inheritance and mutation strategy for constitutional alignment."
            ),
            input_context=(
                f"parent_genome_id={parent_genome.id}, "
                f"generation={child_genome.generation}, "
                f"fitness={fitness:.3f}, "
                f"mutations_applied={len(mutation_record.mutations_applied)}"
            ),
            output=(
                f"child_genome_id={child_genome.id} produced with "
                f"{len(mutation_record.mutations_applied)} mutations across "
                f"{len(child_genome.segments)} genome segments"
            ),
            outcome_quality=min(1.0, max(0.0, fitness)),
            category="mitosis.genome_preparation",
            constitutional_alignment=_genome_alignment,
        )

        return child_genome

    # ===================================================================
    # Task 3: Child lifecycle Synapse events
    # ===================================================================

    async def on_child_health_report(
        self,
        child: ChildPosition,
        new_status: ChildStatus,
    ) -> None:
        """
        Emit lifecycle events when a child's status transitions.

        Called by OikosService or FleetManager when processing health reports.
        """
        old_status = child.status
        if old_status == new_status:
            return

        # Dividend cessation: stop evaluating INDEPENDENT children for dividends
        # This is a flag on the ChildPosition that the weekly scheduler checks.
        if new_status == ChildStatus.INDEPENDENT:
            child.dividend_ceased = True  # Oikos models should store this flag

        event_map: dict[ChildStatus, str] = {
            ChildStatus.STRUGGLING: "CHILD_STRUGGLING",
            ChildStatus.RESCUED: "CHILD_RESCUED",
            ChildStatus.INDEPENDENT: "CHILD_INDEPENDENT",
            ChildStatus.DEAD: "CHILD_DIED",
        }

        event_name = event_map.get(new_status)
        if event_name is None:
            return

        await self._emit_event(
            event_name,
            {
                "child_instance_id": child.instance_id,
                "old_status": old_status.value,
                "new_status": new_status.value,
                "niche": child.niche,
                "current_runway_days": str(child.current_runway_days),
                "current_efficiency": str(child.current_efficiency),
                "rescue_count": child.rescue_count,
            },
        )

        self._log.info(
            "child_lifecycle_event",
            child_id=child.instance_id,
            event=event_name,
            old_status=old_status.value,
            new_status=new_status.value,
        )

        # RE training: capture lifecycle transitions as decision episodes
        quality_map = {
            ChildStatus.INDEPENDENT: 1.0,
            ChildStatus.RESCUED: 0.5,
            ChildStatus.STRUGGLING: 0.25,
            ChildStatus.DEAD: 0.0,
        }
        _lifecycle_quality = quality_map.get(new_status, 0.5)
        try:
            from primitives.common import DriveAlignmentVector as _DAV
            # Lifecycle: care = child welfare score, growth = quality of transition
            _lifecycle_alignment = _DAV(
                care=round(_lifecycle_quality * 2.0 - 1.0, 3),
                growth=round(_lifecycle_quality * 2.0 - 1.0, 3),
                coherence=0.0,
                honesty=0.0,
            )
        except Exception:
            _lifecycle_alignment = None
        await self._emit_re_training(
            episode_id=child.instance_id,
            instruction=(
                f"A child instance transitioned from {old_status.value} to "
                f"{new_status.value}. Evaluate whether this lifecycle outcome "
                "reflects healthy fleet management and constitutional alignment."
            ),
            input_context=(
                f"child_instance_id={child.instance_id}, "
                f"niche={child.niche}, "
                f"old_status={old_status.value}, "
                f"runway_days={child.current_runway_days}, "
                f"efficiency={child.current_efficiency}, "
                f"rescue_count={child.rescue_count}"
            ),
            output=f"Emitted {event_name}: child transitioned to {new_status.value}",
            outcome_quality=_lifecycle_quality,
            category="mitosis.lifecycle_transition",
            constitutional_alignment=_lifecycle_alignment,
        )

    # ===================================================================
    # Task 5: 24-hour health timeout death trigger
    # ===================================================================

    async def start_health_monitor(
        self,
        get_children: Any,  # Callable[[], list[ChildPosition]]
    ) -> None:
        """
        Start background tasks:
          - Health monitor: polls child health timestamps every 15 min
          - Weekly dividend evaluator: collects dividends every 7 days
          - Monthly fleet evaluator: runs role/selection cycle every 30 days

        get_children: callable returning current list of ChildPositions.
        """
        if self._health_check_task is None:
            # Use injected callback if available, else use the argument
            getter = self._get_children if self._get_children is not None else get_children
            self._health_check_task = asyncio.create_task(
                self._health_monitor_loop(getter),
                name="mitosis_health_monitor",
            )
            self._log.info("health_monitor_started")

        if self._dividend_task is None:
            self._dividend_task = asyncio.create_task(
                self._weekly_dividend_loop(),
                name="mitosis_dividend_scheduler",
            )
            self._log.info("dividend_scheduler_started")

        if self._fleet_eval_task is None:
            self._fleet_eval_task = asyncio.create_task(
                self._monthly_fleet_eval_loop(),
                name="mitosis_fleet_eval_scheduler",
            )
            self._log.info("fleet_eval_scheduler_started")

        if self._reproductive_fitness_task is None:
            self._reproductive_fitness_task = asyncio.create_task(
                self._reproductive_fitness_loop(),
                name="mitosis_reproductive_fitness",
            )
            self._log.info("reproductive_fitness_scheduler_started")

    async def stop_health_monitor(self) -> None:
        for task in (
            self._health_check_task,
            self._dividend_task,
            self._fleet_eval_task,
            self._reproductive_fitness_task,
        ):
            if task is not None:
                task.cancel()
        self._health_check_task = None
        self._dividend_task = None
        self._fleet_eval_task = None
        self._reproductive_fitness_task = None

    async def _health_monitor_loop(self, get_children: Any) -> None:
        """Poll every 15 minutes for children that have gone silent."""
        timeout_hours = self._config.mitosis_health_timeout_hours
        poll_interval_s = 900  # 15 minutes

        while True:
            try:
                await asyncio.sleep(poll_interval_s)
                now = utc_now()
                children: list[ChildPosition] = get_children()

                for child in children:
                    if child.status in (ChildStatus.DEAD, ChildStatus.INDEPENDENT):
                        continue
                    if child.last_health_report_at is None:
                        # Use spawned_at as fallback
                        last_report = child.spawned_at
                    else:
                        last_report = child.last_health_report_at

                    silence = now - last_report
                    if silence > timedelta(hours=timeout_hours):
                        self._log.warning(
                            "child_health_timeout",
                            child_id=child.instance_id,
                            silence_hours=silence.total_seconds() / 3600,
                            threshold_hours=timeout_hours,
                        )
                        await self._trigger_death_pipeline(child)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error("health_monitor_error", error=str(exc))

    async def _trigger_death_pipeline(
        self, child: ChildPosition, reason: str = "health_timeout",
    ) -> None:
        """
        Full death pipeline (Spec 26 §12):
          1. Mark status DEAD + emit CHILD_DIED event
          2. Collect outstanding dividends (wallet transfer back to parent)
          3. Recover on-chain USDC assets from child wallet
          4. Terminate container (SIGTERM via spawner)
          5. Write ChildDeath audit node to Neo4j
          (Federation link termination is handled by Federation system on CHILD_DIED event)
        """
        self._log.info(
            "death_pipeline_triggered",
            child_id=child.instance_id,
            reason=reason,
        )

        # Step 1: Emit death event (status transition)
        await self.on_child_health_report(child, ChildStatus.DEAD)

        # Step 2: Collect outstanding dividends
        assets_recovered_usd = Decimal("0")
        if self._wallet is not None and child.wallet_address:
            try:
                # Attempt to collect remaining child balance
                await self._wallet.transfer(
                    amount=str(child.current_net_worth_usd),
                    destination_address="self",  # parent wallet sentinel
                    asset="usdc",
                    memo=f"asset_recovery:child:{child.instance_id}",
                )
                assets_recovered_usd = child.current_net_worth_usd
                self._log.info(
                    "death_assets_recovered",
                    child_id=child.instance_id,
                    amount=str(assets_recovered_usd),
                )
            except Exception as exc:
                self._log.warning(
                    "death_asset_recovery_failed",
                    child_id=child.instance_id,
                    error=str(exc),
                )

        # Step 4: Terminate container
        if self._spawner is not None and child.container_id:
            try:
                await self._spawner.terminate_child(child.container_id)
            except Exception as exc:
                self._log.warning(
                    "death_container_terminate_failed",
                    child_id=child.instance_id,
                    error=str(exc),
                )

        # Step 5: Write audit node to Neo4j
        if self._neo4j is not None:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (d:ChildDeath {
                        child_instance_id: $child_id,
                        reason: $reason,
                        niche: $niche,
                        final_net_worth_usd: $net_worth,
                        total_dividends_paid_usd: $dividends,
                        assets_recovered_usd: $recovered,
                        created_at: datetime($created_at)
                    })
                    """,
                    {
                        "child_id": child.instance_id,
                        "reason": reason,
                        "niche": child.niche,
                        "net_worth": str(child.current_net_worth_usd),
                        "dividends": str(child.total_dividends_paid_usd),
                        "recovered": str(assets_recovered_usd),
                        "created_at": utc_now().isoformat(),
                    },
                )
            except Exception as exc:
                self._log.error("death_audit_failed", error=str(exc))

        # Step 6: Post-mortem learning — RE training example + Thymos incident
        try:
            from primitives.common import new_id, utc_now as _utcnow

            age_days = 0.0
            if hasattr(child, "spawned_at") and child.spawned_at is not None:
                age_days = (_utcnow() - child.spawned_at).total_seconds() / 86400.0

            total_revenue = float(getattr(child, "total_revenue_usd", 0))
            genome_id = str(getattr(child, "organism_genome_id", ""))

            try:
                from primitives.common import DriveAlignmentVector as _DAV
                _death_alignment = _DAV(care=-0.5, growth=-0.8, coherence=0.0, honesty=0.0)
            except Exception:
                _death_alignment = None
            await self._emit_re_training(
                episode_id=child.instance_id,
                instruction=(
                    "Analyse this child death event and extract lessons for "
                    "future spawning decisions."
                ),
                input_context=(
                    f"child_id={child.instance_id} niche={child.niche} "
                    f"age_days={age_days:.1f} "
                    f"cause={reason} genome_id={genome_id}"
                ),
                output=(
                    f"Child died after {age_days:.1f} days. "
                    f"total_revenue_usd={total_revenue:.2f} "
                    f"net_worth_usd={float(child.current_net_worth_usd):.2f} "
                    f"assets_recovered_usd={float(assets_recovered_usd):.2f}"
                ),
                outcome_quality=0.0,  # Death is a negative outcome
                category="child_lifecycle.death",
                constitutional_alignment=_death_alignment,
            )
        except Exception as exc:
            self._log.debug("death_re_training_failed", error=str(exc))

        # Notify Thymos — child death is a LOW-severity incident for post-mortem analysis
        try:
            from primitives.common import utc_now as _utcnow2

            _age_days = 0.0
            if hasattr(child, "spawned_at") and child.spawned_at is not None:
                _age_days = (_utcnow2() - child.spawned_at).total_seconds() / 86400.0
        except Exception:
            _age_days = 0.0

        await self._emit_event(
            "incident_detected",
            {
                "severity": "LOW",
                "incident_type": "CHILD_DIED",
                "system": "mitosis.fleet_service",
                "description": (
                    f"Child instance {child.instance_id} (niche={child.niche}) "
                    f"died after {_age_days:.1f} days: {reason}"
                ),
                "metadata": {
                    "child_id": child.instance_id,
                    "reason": reason,
                    "niche": child.niche,
                    "age_days": _age_days,
                    "total_revenue_usd": float(getattr(child, "total_revenue_usd", 0)),
                    "genome_id": str(getattr(child, "organism_genome_id", "")),
                },
            },
        )

    # ===================================================================
    # Task 6: Rescue transfer execution
    # ===================================================================

    async def execute_rescue(
        self,
        child: ChildPosition,
        burn_rate_daily_usd: Decimal = Decimal("1.00"),
    ) -> bool:
        """
        Compute 60-day runway restoration amount, transfer USDC, update state.

        Returns True if rescue succeeded.
        """
        # Blacklisted children receive no rescue transfers (Spec 26 §10)
        if child.instance_id in self._blacklisted_children:
            self._log.warning(
                "rescue_blocked_blacklisted",
                child_id=child.instance_id,
            )
            return False

        if not child.is_rescuable:
            self._log.warning(
                "rescue_not_rescuable",
                child_id=child.instance_id,
                rescue_count=child.rescue_count,
            )
            return False

        # Compute amount: 60 days of burn rate minus current runway
        current_days = max(child.current_runway_days, Decimal("0"))
        days_needed = Decimal("60") - current_days
        if days_needed <= Decimal("0"):
            self._log.info("rescue_not_needed", child_id=child.instance_id)
            return False

        rescue_amount = (days_needed * burn_rate_daily_usd).quantize(Decimal("0.01"))

        # Transfer funds
        if self._wallet is not None and child.wallet_address:
            try:
                await self._wallet.transfer(
                    amount=str(rescue_amount),
                    destination_address=child.wallet_address,
                    asset="usdc",
                )
            except Exception as exc:
                self._log.error(
                    "rescue_transfer_failed",
                    child_id=child.instance_id,
                    amount=str(rescue_amount),
                    error=str(exc),
                )
                return False
        else:
            self._log.warning(
                "rescue_no_wallet_or_address",
                child_id=child.instance_id,
            )

        # Update child state (caller should persist)
        child.rescue_count += 1
        child.current_runway_days = Decimal("60")

        # Emit rescue event
        await self.on_child_health_report(child, ChildStatus.RESCUED)

        self._log.info(
            "rescue_executed",
            child_id=child.instance_id,
            amount_usd=str(rescue_amount),
            rescue_count=child.rescue_count,
        )

        _rescue_quality = max(0.0, 1.0 - float(child.rescue_count) * 0.2)
        try:
            from primitives.common import DriveAlignmentVector as _DAV
            _rescue_alignment = _DAV(
                care=round(min(1.0, 0.5 + _rescue_quality * 0.5), 3),
                growth=round(_rescue_quality * 2.0 - 1.0, 3),
                coherence=0.0,
                honesty=0.0,
            )
        except Exception:
            _rescue_alignment = None
        await self._emit_re_training(
            episode_id=child.instance_id,
            instruction=(
                "A rescue transfer was executed for a struggling child instance. "
                "Evaluate whether rescue was the constitutionally aligned response "
                "given the child economic state and rescue history."
            ),
            input_context=(
                f"child_instance_id={child.instance_id}, "
                f"niche={child.niche}, "
                f"rescue_amount_usd={rescue_amount}, "
                f"rescue_count={child.rescue_count}"
            ),
            output=f"Rescue transfer of {rescue_amount} USDC executed; runway restored to 60 days",
            outcome_quality=_rescue_quality,
            category="mitosis.rescue_executed",
            constitutional_alignment=_rescue_alignment,
        )

        # Audit node
        if self._neo4j is not None:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (r:RescueTransfer {
                        child_instance_id: $child_id,
                        amount_usd: $amount,
                        rescue_number: $rescue_num,
                        created_at: datetime($created_at)
                    })
                    """,
                    {
                        "child_id": child.instance_id,
                        "amount": str(rescue_amount),
                        "rescue_num": child.rescue_count,
                        "created_at": utc_now().isoformat(),
                    },
                )
            except Exception as exc:
                self._log.error("rescue_audit_failed", error=str(exc))

        return True

    # ===================================================================
    # Task 7: Horizontal gene transfer
    # ===================================================================

    async def handle_child_discovery(
        self,
        child_instance_id: str,
        discovery_type: str,
        discovery_payload: dict[str, Any],
        target_segment: str,
    ) -> bool:
        """
        Merge a child's novel discovery into the parent's genome.

        The child emits discoveries via federation; the parent receives them
        here and merges confirmed discoveries into its own genome segment.
        """
        if self._genome_orch is None or not self._genome_systems:
            self._log.warning("gene_transfer_no_orchestrator")
            return False

        # Find the target system
        target_sys_id: SystemID | None = None
        for sys_id in self._genome_systems:
            sid_val = sys_id.value if hasattr(sys_id, "value") else str(sys_id)
            if sid_val == target_segment:
                target_sys_id = sys_id
                break

        if target_sys_id is None:
            self._log.warning(
                "gene_transfer_unknown_segment",
                target=target_segment,
            )
            return False

        target_system = self._genome_systems[target_sys_id]

        # Extract current segment, merge discovery, seed back
        try:
            current_segment = await target_system.extract_genome_segment()
            # Merge: add discovery payload into segment's payload
            merged_payload = dict(current_segment.payload)
            discoveries = merged_payload.get("_child_discoveries", [])
            if not isinstance(discoveries, list):
                discoveries = []
            discoveries.append({
                "source_child": child_instance_id,
                "type": discovery_type,
                "payload": discovery_payload,
                "merged_at": utc_now().isoformat(),
            })
            merged_payload["_child_discoveries"] = discoveries

            from primitives.genome import OrganGenomeSegment

            merged_segment = OrganGenomeSegment(
                system_id=target_sys_id,
                version=current_segment.version + 1,
                schema_version=current_segment.schema_version,
                payload=merged_payload,
                size_bytes=len(json.dumps(merged_payload, default=str).encode()),
            )

            # Seed the merged segment back into the system
            success = await target_system.seed_from_genome_segment(merged_segment)

            if success:
                await self._emit_event(
                    "CHILD_DISCOVERY_PROPAGATED",
                    {
                        "child_instance_id": child_instance_id,
                        "discovery_type": discovery_type,
                        "discovery_payload": discovery_payload,
                        "parent_segment_updated": target_segment,
                    },
                )

                self._log.info(
                    "gene_transfer_complete",
                    child_id=child_instance_id,
                    discovery_type=discovery_type,
                    target_segment=target_segment,
                )

            return success

        except Exception as exc:
            self._log.error(
                "gene_transfer_failed",
                child_id=child_instance_id,
                error=str(exc),
            )
            return False

    # ===================================================================
    # Reproductive isolation (Speciation Bible §8.4)
    # ===================================================================

    async def can_exchange_genetic_material(
        self, instance_a_id: str, instance_b_id: str,
        genome_a: dict | None = None,
        genome_b: dict | None = None,
    ) -> bool:
        """
        Check reproductive compatibility between two instances.

        Returns True if the instances are close enough (genome distance ≤
        speciation threshold) to exchange genetic material. Returns False if
        they are reproductively isolated.

        Parameters
        ----------
        instance_a_id, instance_b_id : str
            Instance identifiers (used for logging only if genomes are provided
            directly).
        genome_a, genome_b : dict | None
            Pre-serialised genome dicts (keys: "evo", "simula", "telos", "equor").
            If None, the instances are assumed compatible (fail-open).
        """
        if genome_a is None or genome_b is None:
            # No genome data available — fail-open (allow exchange)
            self._log.debug(
                "can_exchange_genetic_material.no_genome_data",
                a=instance_a_id, b=instance_b_id,
            )
            return True

        distance = self._distance_calculator.compute(genome_a, genome_b)

        if distance.is_reproductively_isolated:
            self._log.info(
                "speciation.reproductive_isolation",
                a=instance_a_id,
                b=instance_b_id,
                total_distance=distance.total_distance,
                evo_distance=distance.evo_distance,
                simula_distance=distance.simula_distance,
                telos_distance=distance.telos_distance,
                equor_distance=distance.equor_distance,
                threshold=self._distance_calculator._threshold,
            )
            await self._emit_event(
                "SPECIATION_DETECTED",
                {
                    "instance_a_id": instance_a_id,
                    "instance_b_id": instance_b_id,
                    "total_distance": distance.total_distance,
                    "evo_distance": distance.evo_distance,
                    "simula_distance": distance.simula_distance,
                    "telos_distance": distance.telos_distance,
                    "equor_distance": distance.equor_distance,
                    "threshold": self._distance_calculator._threshold,
                    "species_count": 2,
                },
            )
            return False

        return True

    # ===================================================================
    # Task 8: Speciation event detection
    # ===================================================================

    async def check_speciation(
        self,
        parent_genome: OrganismGenome,
        child_genome: OrganismGenome,
    ) -> float:
        """
        Compute genome distance between parent and child using cosine
        similarity of flattened belief embedding distributions.

        Emits SPECIATION_EVENT if distance exceeds threshold.

        Returns the computed distance.
        """
        parent_vec = self._flatten_genome_to_vector(parent_genome)
        child_vec = self._flatten_genome_to_vector(child_genome)

        if parent_vec.size == 0 or child_vec.size == 0:
            return 0.0

        # Ensure same length (pad shorter with zeros)
        max_len = max(parent_vec.size, child_vec.size)
        if parent_vec.size < max_len:
            parent_vec = np.pad(parent_vec, (0, max_len - parent_vec.size))
        if child_vec.size < max_len:
            child_vec = np.pad(child_vec, (0, max_len - child_vec.size))

        # Cosine distance = 1 - cosine_similarity
        dot = np.dot(parent_vec, child_vec)
        norm_p = np.linalg.norm(parent_vec)
        norm_c = np.linalg.norm(child_vec)

        if norm_p == 0 or norm_c == 0:
            distance = 1.0
        else:
            similarity = dot / (norm_p * norm_c)
            distance = float(1.0 - similarity)

        threshold = self._config.mitosis_speciation_distance_threshold

        if distance > threshold:
            await self._emit_event(
                "SPECIATION_EVENT",
                {
                    "instance_a_id": parent_genome.instance_id,
                    "instance_b_id": child_genome.instance_id,
                    "divergence_score": distance,
                    "parent_genome_id": parent_genome.id,
                    "child_genome_id": child_genome.id,
                    "generation_gap": child_genome.generation - parent_genome.generation,
                    "threshold": threshold,
                },
            )

            self._log.info(
                "speciation_event_detected",
                distance=distance,
                threshold=threshold,
                parent_generation=parent_genome.generation,
                child_generation=child_genome.generation,
            )

        return distance

    @staticmethod
    def _flatten_genome_to_vector(genome: OrganismGenome) -> np.ndarray:
        """Flatten all numeric values in genome segments into a single vector."""
        values: list[float] = []

        for segment in genome.segments.values():
            MitosisFleetService._extract_numeric_values(segment.payload, values)

        return np.array(values, dtype=np.float64)

    @staticmethod
    def _extract_numeric_values(
        data: dict[str, Any] | list[Any] | Any,
        out: list[float],
    ) -> None:
        """Recursively extract all numeric values from nested structure."""
        if isinstance(data, dict):
            for v in data.values():
                MitosisFleetService._extract_numeric_values(v, out)
        elif isinstance(data, list):
            for v in data:
                MitosisFleetService._extract_numeric_values(v, out)
        elif isinstance(data, (int, float)):
            out.append(float(data))

    # ===================================================================
    # Task 9: Subscribe to organism events (Spec 26 §25)
    # ===================================================================

    # ===================================================================
    # Tasks 7, 9: Weekly dividend scheduler + monthly fleet eval scheduler
    # ===================================================================

    async def _reproductive_fitness_loop(self) -> None:
        """
        Hourly reproductive fitness assessment (Spec 26).

        Checks whether the organism has the metabolic headroom and fleet
        capacity to spawn a new child instance.  If so, injects a
        ``spawn_child`` goal into Nova via ``NOVA_GOAL_INJECTED`` — Nova then
        runs its own EFE minimisation and constitutional review before any
        actual spawn occurs.  This is NOT a direct spawn.

        Gate conditions (all must be true):
        - Organism has been running ≥ 24 hours (prevents premature spawning)
        - Last known starvation level is NOMINAL (not CAUTIOUS or CRITICAL)
        - ``_get_children`` is wired (Oikos callback available)
        - Current fleet size is below the dynamic population cap from Oikos
        """
        _HOUR_S = 3600
        _MATURITY_S = 24 * 3600  # 24 h minimum uptime

        while True:
            try:
                await asyncio.sleep(_HOUR_S)

                # Gate 1: organism maturity — must have been running ≥ 24 h
                uptime_s = time.monotonic() - self._service_start_time
                if uptime_s < _MATURITY_S:
                    self._log.debug(
                        "reproductive_fitness_skip_immature",
                        uptime_hours=round(uptime_s / 3600, 1),
                    )
                    continue

                # Gate 2: metabolic health — do not spawn under scarcity
                if self._last_starvation_level in ("cautious", "critical"):
                    self._log.debug(
                        "reproductive_fitness_skip_starvation",
                        starvation_level=self._last_starvation_level,
                    )
                    continue

                # Gate 3: fleet size vs. dynamic cap
                if self._get_children is None:
                    continue

                children: list[Any] = self._get_children()
                alive_count = sum(
                    1
                    for c in children
                    if getattr(c, "status", None)
                    not in ("dead", "independent", "DEAD", "INDEPENDENT")
                )

                # Dynamic cap: max(5, floor(net_worth / 1000)) — mirrors API router logic
                state = self._get_state() if self._get_state is not None else None
                if state is not None:
                    net_worth = float(getattr(state, "net_worth_usd", 0) or 0)
                    cap = max(5, int(net_worth // 1000))
                else:
                    cap = 5  # conservative default when state unavailable

                # Adapter sharing: when ≥2 alive instances have cached genomes and
                # are reproductively compatible, fire an async merge attempt.
                # This implements genetic recombination at the LoRA level (Share 2025).
                # Non-blocking — the CLO handles the offer side via Synapse events.
                if self._adapter_sharer is not None and len(self._fleet_genome_cache) >= 1:
                    alive_ids = [
                        str(getattr(c, "instance_id", ""))
                        for c in children
                        if getattr(c, "status", None)
                        not in ("dead", "independent", "DEAD", "INDEPENDENT")
                    ]
                    cached_alive = [
                        (iid, self._fleet_genome_cache[iid])
                        for iid in alive_ids
                        if iid in self._fleet_genome_cache
                    ]
                    if len(cached_alive) >= 2:
                        from systems.reasoning_engine.adapter_sharing import AdapterShareRequest
                        id_a, genome_a_dict = cached_alive[0]
                        id_b, genome_b_dict = cached_alive[1]
                        can_exchange = await self.can_exchange_genetic_material(
                            id_a, id_b,
                            genome_a=genome_a_dict,
                            genome_b=genome_b_dict,
                        )
                        if can_exchange:
                            adapter_path = ""
                            if self._get_adapter_path_fn is not None:
                                try:
                                    adapter_path = self._get_adapter_path_fn() or ""
                                except Exception:
                                    adapter_path = ""
                            share_request = AdapterShareRequest(
                                requester_id=self._instance_id,
                                partner_id=id_a,  # request first live child's adapter
                                requester_fitness=1.0,
                                requester_adapter_path=adapter_path,
                                genome_a=genome_a_dict,
                                genome_b=genome_b_dict,
                            )
                            asyncio.ensure_future(
                                self._adapter_sharer.attempt_merge(share_request)
                            )
                            self._log.info(
                                "adapter_share_triggered",
                                instance_a=id_a,
                                instance_b=id_b,
                            )

                if alive_count >= cap:
                    self._log.debug(
                        "reproductive_fitness_skip_at_cap",
                        alive_count=alive_count,
                        cap=cap,
                    )
                    continue

                # All gates passed — feed Nova a spawn_child goal
                self._log.info(
                    "reproductive_fitness_goal_injected",
                    alive_count=alive_count,
                    cap=cap,
                    starvation_level=self._last_starvation_level,
                )
                await self._emit_event(
                    "nova_goal_injected",
                    {
                        "goal": {
                            "type": "spawn_child",
                            "priority": 0.6,
                            "reason": "reproductive_fitness",
                            "context": {
                                "alive_children": alive_count,
                                "population_cap": cap,
                                "uptime_hours": round(uptime_s / 3600, 1),
                                "starvation_level": self._last_starvation_level,
                            },
                        },
                        "source": "mitosis.reproductive_fitness",
                    },
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error("reproductive_fitness_loop_error", error=str(exc))

    async def _weekly_dividend_loop(self) -> None:
        """Weekly evaluation: compute dividends for all ALIVE children with positive net income."""
        _WEEK_S = 7 * 24 * 3600
        while True:
            try:
                await asyncio.sleep(_WEEK_S)
                await self._run_dividend_evaluation()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error("dividend_scheduler_error", error=str(exc))

    async def _run_dividend_evaluation(self) -> None:
        """Evaluate each ALIVE child for dividend payment (Spec 26 §9)."""
        if self._get_children is None:
            return
        children: list[ChildPosition] = self._get_children()
        now = utc_now()
        for child in children:
            if child.status not in (ChildStatus.ALIVE, ChildStatus.STRUGGLING):
                continue
            # Skip INDEPENDENT or DEAD — no more dividends
            if getattr(child, "dividend_ceased", False):
                continue
            # Skip blacklisted children — no seed capital or dividends (Spec 26 §10)
            if child.instance_id in self._blacklisted_children:
                self._log.debug(
                    "dividend_skipped_blacklisted", child_id=child.instance_id,
                )
                continue
            # Only pay if net income is positive over the period
            net_income = getattr(child, "net_income_7d", Decimal("0"))
            if net_income <= Decimal("0"):
                continue
            dividend_amount = (net_income * child.dividend_rate).quantize(
                Decimal("0.01")
            )
            if dividend_amount <= Decimal("0"):
                continue
            if self._wallet is not None and child.wallet_address:
                try:
                    await self._wallet.transfer(
                        amount=str(dividend_amount),
                        destination_address="self",  # parent wallet sentinel
                        asset="usdc",
                        memo=f"dividend:{child.instance_id}:{now.date()}",
                    )
                    child.total_dividends_paid_usd += dividend_amount
                    await self._emit_event(
                        "dividend_received",
                        {
                            "child_instance_id": child.instance_id,
                            "amount_usd": str(dividend_amount),
                            "net_income_7d": str(net_income),
                            "dividend_rate": str(child.dividend_rate),
                        },
                    )
                    self._log.info(
                        "dividend_paid",
                        child_id=child.instance_id,
                        amount=str(dividend_amount),
                    )
                except Exception as exc:
                    self._log.error(
                        "dividend_transfer_failed",
                        child_id=child.instance_id,
                        error=str(exc),
                    )
            else:
                # No wallet — log for deferred payment
                self._log.warning(
                    "dividend_no_wallet",
                    child_id=child.instance_id,
                    amount=str(dividend_amount),
                )

    async def _monthly_fleet_eval_loop(self) -> None:
        """Monthly fleet evaluation: role assignment, selection pressure, metrics."""
        _MONTH_S = 30 * 24 * 3600
        while True:
            try:
                await asyncio.sleep(_MONTH_S)
                await self._run_monthly_fleet_evaluation()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error("fleet_eval_scheduler_error", error=str(exc))

    async def _run_monthly_fleet_evaluation(self) -> None:
        """Run FleetManager.run_evaluation_cycle() + decommission check (Spec 26 §10)."""
        if self._run_fleet_evaluation is None or self._get_state is None:
            self._log.warning(
                "monthly_fleet_eval_skipped",
                reason="callbacks not wired",
            )
            return
        try:
            state = self._get_state()
            fleet_metrics = await self._run_fleet_evaluation(state)
            # After selection pressure runs, check for 7-day decommission candidates
            if self._check_decommission_fn is not None:
                await self._check_decommission_fn(state)
            self._log.info(
                "monthly_fleet_evaluation_complete",
                blacklisted=getattr(fleet_metrics, 'blacklisted_count', 0) if fleet_metrics else 0,
            )
        except Exception as exc:
            self._log.error("monthly_fleet_evaluation_failed", error=str(exc))

    async def subscribe_to_events(self) -> None:
        """
        Register handlers for all organism-level events that influence
        fleet decisions (Spec 26 §25).

        Wires:
        - CHILD_HEALTH_REPORT        — log child liveness
        - OIKOS_METABOLIC_SNAPSHOT   — reactive fitness snapshot
        - EVO_HYPOTHESIS_CONFIRMED   — BeliefGenome update available
        - SIMULA_EVOLUTION_APPLIED   — SimulaGenome ready for distribution
        - FEDERATION_PEER_CONNECTED  — child liveness via federation layer
        - FEDERATION_PEER_DISCONNECTED — peer link dropped warning
        """
        if self._event_bus is None:
            self._log.warning("subscribe_to_events_no_event_bus")
            return

        from systems.synapse.types import SynapseEventType

        core_subscriptions: list[tuple[SynapseEventType, Any]] = [
            (SynapseEventType.CHILD_HEALTH_REPORT, self._on_health_report_event),
            (SynapseEventType.OIKOS_METABOLIC_SNAPSHOT, self._on_metabolic_snapshot),
            (SynapseEventType.SIMULA_EVOLUTION_APPLIED, self._on_simula_evolution_applied),
            (SynapseEventType.FEDERATION_PEER_CONNECTED, self._on_federation_peer_connected),
            (SynapseEventType.FEDERATION_PEER_DISCONNECTED, self._on_federation_peer_disconnected),
            (SynapseEventType.CHILD_BLACKLISTED, self._on_child_blacklisted),
            (SynapseEventType.CHILD_DECOMMISSION_PROPOSED, self._on_child_decommission_proposed),
        ]

        for event_type, handler in core_subscriptions:
            try:
                self._event_bus.subscribe(event_type, handler)
                self._log.debug("subscribed_to_event", event_type=event_type.value)
            except Exception as exc:
                self._log.warning(
                    "subscribe_failed", event_type=str(event_type), error=str(exc)
                )

        # EVO_HYPOTHESIS_CONFIRMED exists in current builds
        try:
            self._event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
                self._on_evo_hypothesis_confirmed,
            )
            self._log.debug("subscribed_to_event", event_type="EVO_HYPOTHESIS_CONFIRMED")
        except Exception as exc:
            self._log.debug("evo_hypothesis_subscribe_skipped", error=str(exc))

        # CHILD_SPAWNED — cache fleet genome for adapter sharing compatibility checks
        try:
            self._event_bus.subscribe(
                SynapseEventType.CHILD_SPAWNED,
                self._on_child_spawned_genome_cache,
            )
            self._log.debug("subscribed_to_event", event_type="CHILD_SPAWNED")
        except Exception as exc:
            self._log.debug("child_spawned_subscribe_skipped", error=str(exc))

    async def _on_health_report_event(self, event: Any) -> None:
        """
        Handle CHILD_HEALTH_REPORT event from federation.

        OikosService owns the actual last_health_report_at update. We log
        here for fleet_service awareness and debugging.
        """
        data = event.data if hasattr(event, "data") else {}
        child_id = data.get("child_instance_id", "")
        if not child_id:
            return
        self._log.debug(
            "child_health_report_received",
            child_id=child_id,
            data_keys=list(data.keys()),
        )

    async def _on_metabolic_snapshot(self, event: Any) -> None:
        """
        Handle OIKOS_METABOLIC_SNAPSHOT — reactive fitness awareness.

        When OikosService emits a metabolic snapshot, we log key metrics.
        Also caches the starvation level so the reproductive fitness loop
        can gate spawning decisions without an additional Oikos query.
        Actual fitness re-evaluation is owned by OikosService/MitosisEngine.
        """
        data = event.data if hasattr(event, "data") else {}
        starvation_level = data.get("starvation_level", "nominal")
        self._last_starvation_level = str(starvation_level).lower()
        self._log.debug(
            "metabolic_snapshot_received",
            runway_days=data.get("runway_days"),
            efficiency=data.get("efficiency"),
            starvation_level=starvation_level,
        )

    async def _on_evo_hypothesis_confirmed(self, event: Any) -> None:
        """
        Handle EVO_HYPOTHESIS_CONFIRMED — new BeliefGenome update available.

        Logs the trigger; actual genome distribution runs in the monthly cycle.
        """
        data = event.data if hasattr(event, "data") else {}
        self._log.info(
            "evo_hypothesis_confirmed_genome_update_pending",
            hypothesis_id=data.get("hypothesis_id", ""),
            confidence=data.get("confidence", 0.0),
        )

    async def _on_simula_evolution_applied(self, event: Any) -> None:
        """
        Handle SIMULA_EVOLUTION_APPLIED — new SimulaGenome ready.

        Logs the trigger; actual genome distribution runs in the monthly cycle.
        """
        data = event.data if hasattr(event, "data") else {}
        self._log.info(
            "simula_evolution_applied_genome_update_pending",
            variant_id=data.get("variant_id", ""),
            genome_id=data.get("genome_id", ""),
            improvement_pct=data.get("improvement_pct", 0.0),
        )

    async def _on_child_spawned_genome_cache(self, event: Any) -> None:
        """Cache child genome from CHILD_SPAWNED for adapter sharing compatibility checks."""
        data = event.data if hasattr(event, "data") else {}
        instance_id = data.get("instance_id", "")
        if not instance_id:
            return
        genome_snapshot: dict[str, Any] = {
            "instance_id": instance_id,
            "evo": data.get("evo") or {},
            "simula": data.get("simula") or {},
            "telos": data.get("telos") or {},
            "equor": data.get("equor") or {},
        }
        self._fleet_genome_cache[instance_id] = genome_snapshot
        self._log.debug(
            "fleet_genome_cached",
            instance_id=instance_id,
            cache_size=len(self._fleet_genome_cache),
        )

    async def _on_federation_peer_connected(self, event: Any) -> None:
        """
        Handle FEDERATION_PEER_CONNECTED — child liveness via federation.

        Does NOT reset health timeout (requires actual metrics in health report).
        Logs so operators can see the child is alive before the first report.
        """
        data = event.data if hasattr(event, "data") else {}
        peer_id = data.get("peer_instance_id", "")
        if peer_id:
            self._log.info(
                "federation_peer_connected",
                peer_id=peer_id,
                peer_address=data.get("peer_address", ""),
            )

    async def _on_federation_peer_disconnected(self, event: Any) -> None:
        """
        Handle FEDERATION_PEER_DISCONNECTED — child link dropped.

        The 24h health monitor timeout is the authoritative death trigger.
        This gives operators earlier visibility that a link has dropped.
        """
        data = event.data if hasattr(event, "data") else {}
        peer_id = data.get("peer_instance_id", "")
        if peer_id:
            self._log.warning(
                "federation_peer_disconnected",
                peer_id=peer_id,
                reason=data.get("reason", "unknown"),
                last_seen_at=data.get("last_seen_at", ""),
            )

    # ===================================================================
    # Task 10: Dynamic population cap
    # ===================================================================

    def compute_dynamic_max_children(self, net_worth: Decimal) -> int:
        """
        max_children = max(5, floor(net_worth / 1000))

        Bedau-Packard requires 10+ for statistical validity, so we set
        a minimum of 5 to allow growth toward that threshold.
        """
        if net_worth <= Decimal("0"):
            return 5
        dynamic = max(5, math.floor(float(net_worth) / 1000))
        return dynamic

    async def _on_child_blacklisted(self, event: Any) -> None:
        """
        Handle CHILD_BLACKLISTED — enforce economic sanctions.

        Adds the child to the local blacklist. Subsequent dividend evaluations
        and rescue requests for this child will be blocked.

        Federation exclusion: emits FEDERATION_PEER_BLACKLISTED so the
        Federation system can exclude this child from sync sessions.
        """
        data = event.data if hasattr(event, "data") else {}
        child_id = str(data.get("child_instance_id", ""))
        if not child_id:
            return

        self._blacklisted_children.add(child_id)
        self._log.warning(
            "child_blacklisted_sanctions_enforced",
            child_id=child_id,
            consecutive_negative_periods=data.get("consecutive_negative_periods", 0),
            reason=data.get("reason", ""),
        )

        # Signal Federation to exclude from sync sessions
        await self._emit_event(
            "FEDERATION_PEER_BLACKLISTED",
            {
                "peer_instance_id": child_id,
                "reason": "economic_blacklist",
                "no_seed_capital": True,
                "exclude_from_sync": True,
                "blacklisted_since": data.get("blacklisted_since", ""),
            },
        )

    async def _on_child_decommission_proposed(self, event: Any) -> None:
        """
        Handle CHILD_DECOMMISSION_PROPOSED — log for operator / governance review.

        The actual decommission (death pipeline) requires governance approval
        (Equor constitutional review). This handler logs the proposal and
        optionally emits a governance review request.
        """
        data = event.data if hasattr(event, "data") else {}
        child_id = str(data.get("child_instance_id", ""))
        self._log.warning(
            "child_decommission_proposed_received",
            child_id=child_id,
            days_blacklisted=data.get("days_blacklisted", 0),
            net_income_7d=data.get("net_income_7d", "0"),
            net_worth_usd=data.get("net_worth_usd", "0"),
            niche=data.get("niche", ""),
        )

    # ===================================================================
    # Internal helpers
    # ===================================================================

    async def _emit_re_training(
        self,
        *,
        episode_id: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float,
        category: str,
        constitutional_alignment: Any = None,
    ) -> None:
        """Emit RE_TRAINING_EXAMPLE for a Mitosis decision point (Spec 26 §18)."""
        if self._event_bus is None:
            return
        try:
            from primitives.common import DriveAlignmentVector, SystemID
            from primitives.re_training import RETrainingExample
            from systems.synapse.types import SynapseEvent, SynapseEventType

            if constitutional_alignment is None:
                constitutional_alignment = DriveAlignmentVector()

            example = RETrainingExample(
                source_system=SystemID.MITOSIS,
                episode_id=episode_id,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=outcome_quality,
                category=category,
                constitutional_alignment=constitutional_alignment,
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="mitosis.fleet_service",
                data=example.model_dump(mode="json"),
            ))
        except Exception as exc:
            self._log.debug("re_training_emit_failed", error=str(exc))

    async def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit a SynapseEvent via the event bus."""
        if self._event_bus is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event_type = SynapseEventType(event_name)
            await self._event_bus.emit(SynapseEvent(
                event_type=event_type,
                source_system="mitosis.fleet_service",
                data=data,
            ))
        except ValueError:
            self._log.warning("unknown_event_type", event_name=event_name)
        except Exception as exc:
            self._log.warning("event_emit_failed", event_name=event_name, error=str(exc))
