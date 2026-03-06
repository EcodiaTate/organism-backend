"""
EcodiaOS — Oikos Fleet Manager (Phase 16m: Population Ecology)

Population-level management of child instances spawned via mitosis. The fleet
is not a flat list of containers — it is an evolving population subject to
selection pressure, role specialization, and ecological dynamics.

Responsibilities:
  1. Track all child instances: capabilities, economic performance, health.
  2. Apply selection pressure: underperforming instances (negative economic
     ratio over a sustained period) are blacklisted from receiving new genomes.
  3. Role specialization: when population exceeds a threshold, assign roles
     so instances specialize instead of competing in the same niche.
  4. Expose fleet-level metrics for benchmarks and dashboard.

Integration points:
  - Federation: fleet members are federated peers (establish_link on spawn).
  - Simula genome: each instance carries a genome; fit instances donate theirs.
  - Oikos economic tracking: ChildPosition, DividendRecord, MitosisEngine.
  - Benchmarks: fleet_health KPI exposed via metrics snapshot.
  - Synapse: emits FLEET_* events for orchestration.

Architecture:
  - Pure computation + event emission (no I/O beyond Synapse).
  - All child state lives in OikosService's EconomicState.child_instances.
  - FleetManager operates on that list and emits decisions; Axon executes.
"""

from __future__ import annotations

import enum
from decimal import Decimal
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now
from systems.oikos.models import (
    ChildPosition,
    ChildStatus,
    EconomicState,
)

if TYPE_CHECKING:

    from config import OikosConfig
    from systems.synapse.event_bus import EventBus
logger = structlog.get_logger("oikos.fleet")


# ─── Fleet Role ──────────────────────────────────────────────────


class FleetRole(enum.StrEnum):
    """
    Specialization role assigned when population exceeds the threshold.

    Generalist is the default — all instances start here. Once the fleet
    grows past fleet_specialization_threshold, the manager assigns roles
    based on each instance's demonstrated strengths.
    """

    GENERALIST = "generalist"
    BOUNTY_HUNTER = "bounty_hunter"
    KNOWLEDGE_BROKER = "knowledge_broker"
    RESEARCH_MUTATOR = "research_mutator"


# ─── Selection Pressure ─────────────────────────────────────────


class SelectionVerdict(enum.StrEnum):
    """Outcome of selection pressure evaluation for a single child."""

    FIT = "fit"                    # Positive economic ratio — may reproduce
    UNDERPERFORMING = "underperforming"  # Negative ratio but within grace period
    BLACKLISTED = "blacklisted"    # Sustained negative ratio — no new genomes


class SelectionRecord(EOSBaseModel):
    """Immutable record of a selection pressure evaluation."""

    record_id: str = Field(default_factory=new_id)
    child_instance_id: str
    verdict: SelectionVerdict
    economic_ratio: Decimal
    """Revenue / costs over evaluation window. <1 = losing money."""
    consecutive_negative_periods: int
    role: FleetRole
    timestamp: datetime = Field(default_factory=utc_now)
    reason: str = ""


# ─── Role Assignment Record ────────────────────────────────────


class RoleAssignment(EOSBaseModel):
    """Record of a role (re)assignment for a child instance."""

    assignment_id: str = Field(default_factory=new_id)
    child_instance_id: str
    old_role: FleetRole
    new_role: FleetRole
    reason: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Fleet-Level Metrics ───────────────────────────────────────


class FleetMetrics(EOSBaseModel):
    """
    Population-level snapshot exposed to benchmarks and dashboard.

    This is the fleet equivalent of BenchmarkSnapshot — aggregated KPIs
    across all living children.
    """

    timestamp: datetime = Field(default_factory=utc_now)
    total_children: int = 0
    alive_count: int = 0
    struggling_count: int = 0
    independent_count: int = 0
    dead_count: int = 0
    blacklisted_count: int = 0

    # Economic aggregates
    total_fleet_net_worth: Decimal = Decimal("0")
    total_dividends_received: Decimal = Decimal("0")
    avg_economic_ratio: Decimal = Decimal("0")
    avg_runway_days: Decimal = Decimal("0")

    # Role distribution
    role_distribution: dict[str, int] = Field(default_factory=dict)

    # Selection pressure
    fit_count: int = 0
    underperforming_count: int = 0

    # Genome eligibility — how many children can donate genomes
    genome_eligible_count: int = 0


# ─── Fleet Snapshot (full population state) ─────────────────────


class FleetMemberSnapshot(EOSBaseModel):
    """Enriched view of a single fleet member for dashboard/API."""

    instance_id: str
    niche: str = ""
    role: FleetRole = FleetRole.GENERALIST
    status: ChildStatus = ChildStatus.SPAWNING
    selection_verdict: SelectionVerdict = SelectionVerdict.FIT
    consecutive_negative_periods: int = 0
    economic_ratio: Decimal = Decimal("0")
    net_worth: Decimal = Decimal("0")
    runway_days: Decimal = Decimal("0")
    efficiency: Decimal = Decimal("0")
    total_dividends_paid: Decimal = Decimal("0")
    genome_eligible: bool = False
    spawned_at: datetime = Field(default_factory=utc_now)


# ─── Fleet Manager ──────────────────────────────────────────────


class FleetManager:
    """
    Population-level orchestrator for the child fleet.

    Operates on EconomicState.child_instances and enriches each child
    with role assignments and selection pressure verdicts. Does NOT own
    the child list — OikosService does. FleetManager reads the list,
    computes decisions, and emits events for Axon to execute.

    Thread-safety: NOT thread-safe. Designed for asyncio event loop.
    """

    def __init__(self, config: OikosConfig) -> None:
        self._config = config
        self._event_bus: EventBus | None = None
        self._logger = logger.bind(component="fleet_manager")

        # ── Per-child tracking (keyed by instance_id) ──
        # Roles persist across evaluation cycles
        self._roles: dict[str, FleetRole] = {}
        # Consecutive periods with economic_ratio < 1.0
        self._negative_streak: dict[str, int] = {}
        # Blacklisted instance IDs — cannot receive genomes
        self._blacklisted: set[str] = set()
        # Selection history for observability
        self._selection_history: list[SelectionRecord] = []
        # Role assignment history
        self._role_history: list[RoleAssignment] = []

        self._logger.info(
            "fleet_manager_initialized",
            specialization_threshold=config.fleet_specialization_threshold,
            blacklist_after_periods=config.fleet_blacklist_after_negative_periods,
            evaluation_window_days=config.fleet_evaluation_window_days,
        )

    # ─── Lifecycle ────────────────────────────────────────────────

    def attach(self, event_bus: EventBus) -> None:
        """Wire the EventBus for fleet lifecycle events."""
        self._event_bus = event_bus
        self._logger.info("fleet_manager_attached_to_event_bus")

    # ─── Core: Run Fleet Evaluation Cycle ────────────────────────

    async def run_evaluation_cycle(
        self,
        state: EconomicState,
    ) -> FleetMetrics:
        """
        Run a full fleet evaluation cycle. Called during consolidation.

        Steps:
          1. Prune tracking state for dead/independent children.
          2. Evaluate selection pressure for each living child.
          3. If population >= threshold, assign/reassign roles.
          4. Compute and return fleet-level metrics.
          5. Emit Synapse events for state changes.
        """
        living = [
            c for c in state.child_instances
            if c.status not in (ChildStatus.DEAD, ChildStatus.INDEPENDENT)
        ]

        # 1. Prune stale tracking
        living_ids = {c.instance_id for c in living}
        self._prune_tracking(living_ids, state.child_instances)

        # 2. Selection pressure
        self._evaluate_selection_pressure(living)

        # 3. Role specialization (only when population >= threshold)
        if len(living) >= self._config.fleet_specialization_threshold:
            await self._assign_roles(living)
        else:
            # Below threshold: everyone is a generalist
            for child in living:
                old_role = self._roles.get(child.instance_id, FleetRole.GENERALIST)
                if old_role != FleetRole.GENERALIST:
                    self._roles[child.instance_id] = FleetRole.GENERALIST
                    await self._emit_role_change(
                        child.instance_id, old_role, FleetRole.GENERALIST,
                        "population below specialization threshold",
                    )
                else:
                    self._roles[child.instance_id] = FleetRole.GENERALIST

        # 4. Compute metrics
        metrics = self._compute_metrics(state)

        # 5. Emit fleet evaluation event
        await self._emit_fleet_evaluated(metrics)

        self._logger.info(
            "fleet_evaluation_complete",
            living=len(living),
            blacklisted=metrics.blacklisted_count,
            roles=metrics.role_distribution,
            avg_ratio=str(metrics.avg_economic_ratio),
        )

        return metrics

    # ─── Selection Pressure ──────────────────────────────────────

    def _evaluate_selection_pressure(
        self,
        living: list[ChildPosition],
    ) -> list[SelectionRecord]:
        """
        Evaluate each living child against selection criteria.

        A child with economic_ratio < 1.0 for fleet_blacklist_after_negative_periods
        consecutive evaluation cycles gets blacklisted from genome donation.
        """
        blacklist_threshold = self._config.fleet_blacklist_after_negative_periods
        records: list[SelectionRecord] = []

        for child in living:
            # Compute economic ratio for this child
            ratio = child.current_efficiency  # revenue/costs ratio

            if ratio < Decimal("1"):
                streak = self._negative_streak.get(child.instance_id, 0) + 1
                self._negative_streak[child.instance_id] = streak
            else:
                self._negative_streak[child.instance_id] = 0
                streak = 0

            # Determine verdict
            if streak >= blacklist_threshold:
                verdict = SelectionVerdict.BLACKLISTED
                self._blacklisted.add(child.instance_id)
                reason = (
                    f"economic ratio < 1.0 for {streak} consecutive periods "
                    f"(threshold: {blacklist_threshold})"
                )
            elif streak > 0:
                verdict = SelectionVerdict.UNDERPERFORMING
                reason = f"economic ratio {ratio} < 1.0 ({streak}/{blacklist_threshold} periods)"
            else:
                verdict = SelectionVerdict.FIT
                # Un-blacklist if they recover
                self._blacklisted.discard(child.instance_id)
                reason = f"economic ratio {ratio} >= 1.0"

            record = SelectionRecord(
                child_instance_id=child.instance_id,
                verdict=verdict,
                economic_ratio=ratio,
                consecutive_negative_periods=streak,
                role=self._roles.get(child.instance_id, FleetRole.GENERALIST),
                reason=reason,
            )
            records.append(record)
            self._selection_history.append(record)

            if verdict == SelectionVerdict.BLACKLISTED:
                self._logger.warning(
                    "child_blacklisted",
                    child_id=child.instance_id,
                    ratio=str(ratio),
                    streak=streak,
                )

        return records

    # ─── Role Specialization ────────────────────────────────────

    async def _assign_roles(self, living: list[ChildPosition]) -> None:
        """
        Assign specialized roles based on demonstrated strengths.

        Strategy: score each child for each role based on their niche and
        economic performance, then assign roles to maximize fleet diversity.

        Role assignment criteria:
          - BOUNTY_HUNTER: high efficiency in bounty-like niches, or
            the child with the best short-term economic ratio.
          - KNOWLEDGE_BROKER: niche contains "knowledge", "oracle", "market",
            or child has high dividend output (proxy for value generation).
          - RESEARCH_MUTATOR: niche contains "research", "mutation", "audit",
            or child with highest net worth (capital to fund experiments).
          - GENERALIST: default / overflow.

        We aim for roughly equal distribution across specialized roles,
        with any remainder staying generalist.
        """
        role_slots = max(1, len(living) // 3)  # ~33% per specialized role

        # Score children for each specialized role
        bounty_scores: list[tuple[Decimal, ChildPosition]] = []
        knowledge_scores: list[tuple[Decimal, ChildPosition]] = []
        research_scores: list[tuple[Decimal, ChildPosition]] = []

        for child in living:
            niche_lower = child.niche.lower()

            # Bounty hunter affinity
            bounty_affinity = Decimal("0.5")  # base
            if any(kw in niche_lower for kw in ("bounty", "freelance", "solve", "bug")):
                bounty_affinity += Decimal("0.3")
            bounty_affinity += min(child.current_efficiency, Decimal("3")) / Decimal("6")
            bounty_scores.append((bounty_affinity, child))

            # Knowledge broker affinity
            knowledge_affinity = Decimal("0.5")
            if any(kw in niche_lower for kw in ("knowledge", "oracle", "market", "intel")):
                knowledge_affinity += Decimal("0.3")
            if child.total_dividends_paid_usd > Decimal("0"):
                knowledge_affinity += Decimal("0.2")
            knowledge_scores.append((knowledge_affinity, child))

            # Research mutator affinity
            research_affinity = Decimal("0.5")
            if any(kw in niche_lower for kw in ("research", "mutation", "audit", "experiment")):
                research_affinity += Decimal("0.3")
            if child.current_net_worth_usd > Decimal("100"):
                research_affinity += Decimal("0.2")
            research_scores.append((research_affinity, child))

        # Sort descending by affinity
        bounty_scores.sort(key=lambda p: p[0], reverse=True)
        knowledge_scores.sort(key=lambda p: p[0], reverse=True)
        research_scores.sort(key=lambda p: p[0], reverse=True)

        # Greedily assign: highest-affinity child gets the role, then remove
        # from other candidate lists to prevent double-assignment.
        assigned: set[str] = set()
        new_assignments: dict[str, FleetRole] = {}

        for scored_list, role in [
            (bounty_scores, FleetRole.BOUNTY_HUNTER),
            (knowledge_scores, FleetRole.KNOWLEDGE_BROKER),
            (research_scores, FleetRole.RESEARCH_MUTATOR),
        ]:
            count = 0
            for _score, child in scored_list:
                if child.instance_id in assigned:
                    continue
                if count >= role_slots:
                    break
                new_assignments[child.instance_id] = role
                assigned.add(child.instance_id)
                count += 1

        # Everyone else stays generalist
        for child in living:
            if child.instance_id not in assigned:
                new_assignments[child.instance_id] = FleetRole.GENERALIST

        # Apply assignments and emit events for changes
        for instance_id, new_role in new_assignments.items():
            old_role = self._roles.get(instance_id, FleetRole.GENERALIST)
            if old_role != new_role:
                self._roles[instance_id] = new_role
                await self._emit_role_change(
                    instance_id, old_role, new_role,
                    f"specialization assignment (population={len(living)})",
                )
            else:
                self._roles[instance_id] = new_role

    # ─── Genome Eligibility ─────────────────────────────────────

    def is_genome_eligible(self, instance_id: str) -> bool:
        """
        Check if a child instance is eligible to donate its genome.

        Blacklisted instances cannot donate genomes — their engineering
        knowledge is not propagated to future generations.
        """
        return instance_id not in self._blacklisted

    def get_genome_eligible_ids(self, state: EconomicState) -> list[str]:
        """Return instance IDs of all living, genome-eligible children."""
        return [
            c.instance_id
            for c in state.child_instances
            if c.status not in (ChildStatus.DEAD, ChildStatus.INDEPENDENT)
            and c.instance_id not in self._blacklisted
        ]

    # ─── Role Queries ───────────────────────────────────────────

    def get_role(self, instance_id: str) -> FleetRole:
        """Get the current role of a child instance."""
        return self._roles.get(instance_id, FleetRole.GENERALIST)

    def get_members_by_role(
        self, role: FleetRole, state: EconomicState,
    ) -> list[ChildPosition]:
        """Return all living children with the given role."""
        return [
            c for c in state.child_instances
            if c.status not in (ChildStatus.DEAD, ChildStatus.INDEPENDENT)
            and self._roles.get(c.instance_id, FleetRole.GENERALIST) == role
        ]

    # ─── Metrics Computation ────────────────────────────────────

    def _compute_metrics(self, state: EconomicState) -> FleetMetrics:
        """Compute population-level metrics from current state."""
        all_children = state.child_instances
        living = [
            c for c in all_children
            if c.status not in (ChildStatus.DEAD, ChildStatus.INDEPENDENT)
        ]

        alive_count = sum(1 for c in all_children if c.status == ChildStatus.ALIVE)
        struggling_count = sum(
            1 for c in all_children
            if c.status in (ChildStatus.STRUGGLING, ChildStatus.RESCUED)
        )
        independent_count = sum(1 for c in all_children if c.status == ChildStatus.INDEPENDENT)
        dead_count = sum(1 for c in all_children if c.status == ChildStatus.DEAD)

        # Economic aggregates across living children
        total_net_worth = sum(
            (c.current_net_worth_usd for c in living), Decimal("0"),
        )
        total_dividends = sum(
            (c.total_dividends_paid_usd for c in all_children), Decimal("0"),
        )

        # Average economic ratio (efficiency) across living
        if living:
            avg_ratio = (
                sum((c.current_efficiency for c in living), Decimal("0"))
                / Decimal(str(len(living)))
            ).quantize(Decimal("0.001"))
            avg_runway = (
                sum((c.current_runway_days for c in living), Decimal("0"))
                / Decimal(str(len(living)))
            ).quantize(Decimal("0.1"))
        else:
            avg_ratio = Decimal("0")
            avg_runway = Decimal("0")

        # Role distribution
        role_dist: dict[str, int] = {}
        for child in living:
            role = self._roles.get(child.instance_id, FleetRole.GENERALIST).value
            role_dist[role] = role_dist.get(role, 0) + 1

        # Selection verdicts
        fit_count = sum(
            1 for c in living
            if c.instance_id not in self._blacklisted
            and self._negative_streak.get(c.instance_id, 0) == 0
        )
        underperforming_count = sum(
            1 for c in living
            if c.instance_id not in self._blacklisted
            and self._negative_streak.get(c.instance_id, 0) > 0
        )

        # Genome eligible
        genome_eligible = sum(
            1 for c in living
            if c.instance_id not in self._blacklisted
        )

        return FleetMetrics(
            total_children=len(all_children),
            alive_count=alive_count,
            struggling_count=struggling_count,
            independent_count=independent_count,
            dead_count=dead_count,
            blacklisted_count=len(self._blacklisted & {c.instance_id for c in living}),
            total_fleet_net_worth=total_net_worth,
            total_dividends_received=total_dividends,
            avg_economic_ratio=avg_ratio,
            avg_runway_days=avg_runway,
            role_distribution=role_dist,
            fit_count=fit_count,
            underperforming_count=underperforming_count,
            genome_eligible_count=genome_eligible,
        )

    def get_fleet_snapshot(self, state: EconomicState) -> list[FleetMemberSnapshot]:
        """
        Build a full fleet snapshot for dashboard / API consumption.

        Enriches each child with role, selection verdict, and genome eligibility.
        """
        snapshots: list[FleetMemberSnapshot] = []
        for child in state.child_instances:
            streak = self._negative_streak.get(child.instance_id, 0)
            if child.instance_id in self._blacklisted:
                verdict = SelectionVerdict.BLACKLISTED
            elif streak > 0:
                verdict = SelectionVerdict.UNDERPERFORMING
            else:
                verdict = SelectionVerdict.FIT

            snapshots.append(FleetMemberSnapshot(
                instance_id=child.instance_id,
                niche=child.niche,
                role=self._roles.get(child.instance_id, FleetRole.GENERALIST),
                status=child.status,
                selection_verdict=verdict,
                consecutive_negative_periods=streak,
                economic_ratio=child.current_efficiency,
                net_worth=child.current_net_worth_usd,
                runway_days=child.current_runway_days,
                efficiency=child.current_efficiency,
                total_dividends_paid=child.total_dividends_paid_usd,
                genome_eligible=child.instance_id not in self._blacklisted,
                spawned_at=child.spawned_at,
            ))

        return snapshots

    # ─── Internal: Prune Tracking ───────────────────────────────

    def _prune_tracking(
        self,
        living_ids: set[str],
        all_children: list[ChildPosition],
    ) -> None:
        """Remove tracking state for dead/independent children."""
        all_ids = {c.instance_id for c in all_children}
        dead_or_gone = all_ids - living_ids

        for instance_id in dead_or_gone:
            self._negative_streak.pop(instance_id, None)
            self._blacklisted.discard(instance_id)
            # Keep role in history but remove from active tracking
            self._roles.pop(instance_id, None)

    # ─── Event Emission ─────────────────────────────────────────

    async def _emit_fleet_evaluated(self, metrics: FleetMetrics) -> None:
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.FLEET_EVALUATED,
            source_system="oikos",
            data={
                "total_children": metrics.total_children,
                "alive": metrics.alive_count,
                "struggling": metrics.struggling_count,
                "blacklisted": metrics.blacklisted_count,
                "avg_economic_ratio": str(metrics.avg_economic_ratio),
                "role_distribution": metrics.role_distribution,
                "genome_eligible": metrics.genome_eligible_count,
            },
        ))

    async def _emit_role_change(
        self,
        instance_id: str,
        old_role: FleetRole,
        new_role: FleetRole,
        reason: str,
    ) -> None:
        assignment = RoleAssignment(
            child_instance_id=instance_id,
            old_role=old_role,
            new_role=new_role,
            reason=reason,
        )
        self._role_history.append(assignment)

        self._logger.info(
            "fleet_role_changed",
            child_id=instance_id,
            old_role=old_role.value,
            new_role=new_role.value,
            reason=reason,
        )

        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.FLEET_ROLE_CHANGED,
            source_system="oikos",
            data={
                "child_instance_id": instance_id,
                "old_role": old_role.value,
                "new_role": new_role.value,
                "reason": reason,
            },
        ))

    # ─── Observability ──────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "tracked_children": len(self._roles),
            "blacklisted": len(self._blacklisted),
            "role_counts": self._count_roles(),
            "selection_history_count": len(self._selection_history),
            "role_history_count": len(self._role_history),
        }

    def _count_roles(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for role in self._roles.values():
            key = role.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def selection_history(self) -> list[SelectionRecord]:
        return list(self._selection_history)

    @property
    def role_history(self) -> list[RoleAssignment]:
        return list(self._role_history)
