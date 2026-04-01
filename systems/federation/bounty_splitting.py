"""
EcodiaOS - Federation Bounty Co-Solving

When a bounty exceeds a single instance's time/resource budget, the
BountySplitter decomposes it into N sub-tasks, distributes them to
federation peers via TaskDelegationManager, collects results within the
deadline, merges the solutions, and distributes payment proportionally
on receipt of the bounty payout.

Workflow:
  1. Nova deliberates: "too large for solo, federation available"
     → calls ``BountySplitter.should_split()``
  2. ``split_bounty()`` - Simula generates task boundaries; N sub-tasks
     are produced as TaskDelegation objects.
  3. ``broadcast_sub_tasks()`` - each sub-task is offered via
     TaskDelegationManager to PARTNER+ peers ordered by trust score.
  4. ``collect_results()`` - waits up to ``deadline_hours`` for results.
  5. ``merge_and_submit()`` - aggregates accepted sub-results into a
     unified PR/solution payload.
  6. ``on_bounty_paid()`` - splits the received bounty USDC among
     contributors proportionally.

Simula boundary generation: when Simula is wired, it is asked to produce
sub-task payloads via a SYNAPSE event.  Without Simula, the splitter falls
back to naive chunk-splitting on the ``description`` field.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from primitives.federation import (
    FederationLink,
    TaskDelegation,
    TaskStatus,
    TaskType,
    TrustLevel,
)

if TYPE_CHECKING:
    from systems.federation.task_delegation import TaskDelegationManager

logger = structlog.get_logger("systems.federation.bounty_splitting")

# Fraction of the total bounty kept by the orchestrating instance.
_ORCHESTRATOR_CUT: Decimal = Decimal("0.1")  # 10%


class SubTaskRecord:
    """Runtime record linking a sub-task to its contributor and result."""

    __slots__ = ("task", "peer_instance_id", "result", "reward_fraction")

    def __init__(
        self,
        task: TaskDelegation,
        peer_instance_id: str,
        reward_fraction: float = 0.0,
    ) -> None:
        self.task = task
        self.peer_instance_id = peer_instance_id
        self.result: dict[str, Any] | None = None
        self.reward_fraction = reward_fraction


class BountySplitter:
    """
    Orchestrates bounty decomposition and multi-instance co-solving.

    One BountySplitter lives inside FederationService.  It is stateless
    between bounties - each ``split_bounty()`` call creates a fresh
    in-flight record.
    """

    def __init__(
        self,
        instance_id: str,
        delegation: TaskDelegationManager,
    ) -> None:
        self._instance_id = instance_id
        self._delegation = delegation
        self._event_bus: Any = None
        self._simula: Any = None  # Wired post-init for intelligent task splitting
        self._logger = logger.bind(component="bounty_splitter", instance_id=instance_id)

        # bounty_id → list[SubTaskRecord]
        self._active_bounties: dict[str, list[SubTaskRecord]] = {}
        # Stats
        self._bounties_split: int = 0
        self._sub_tasks_completed: int = 0

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    def set_simula(self, simula: Any) -> None:
        self._simula = simula

    def should_split(
        self,
        bounty_payload: dict[str, Any],
        trusted_peers: list[FederationLink],
        solo_budget_hours: float = 8.0,
    ) -> bool:
        """
        Heuristic: split if estimated effort > solo budget AND
        there are PARTNER+ peers available.

        Estimated effort is taken from ``bounty_payload.get("estimated_hours")``.
        Falls back to True if effort is unknown and peers are available.
        """
        estimated = float(bounty_payload.get("estimated_hours", solo_budget_hours + 1))
        partner_peers = [
            p for p in trusted_peers
            if p.trust_level.value >= TrustLevel.PARTNER.value
        ]
        return estimated > solo_budget_hours and len(partner_peers) > 0

    async def split_bounty(
        self,
        bounty_id: str,
        bounty_payload: dict[str, Any],
        total_reward_usdc: Decimal,
        trusted_links: list[FederationLink],
        max_sub_tasks: int = 4,
        deadline_hours: int = 24,
    ) -> list[TaskDelegation]:
        """
        Decompose a bounty into sub-tasks.

        Uses Simula (via Synapse event) if wired; otherwise falls back to
        naive description splitting.  Returns a list of TaskDelegation
        objects ready to be offered to peers.
        """
        sub_payloads = await self._generate_sub_payloads(
            bounty_payload, max_sub_tasks=max_sub_tasks
        )

        # Compute per-sub-task reward (peers share 90%, orchestrator keeps 10%)
        n = len(sub_payloads)
        peer_pool = total_reward_usdc * (Decimal("1") - _ORCHESTRATOR_CUT)
        reward_per_task = (peer_pool / n).quantize(Decimal("0.01")) if n else Decimal("0")

        sub_tasks: list[TaskDelegation] = []
        for sub_payload in sub_payloads:
            task = self._delegation.build_task(
                task_type=TaskType(bounty_payload.get("task_type", TaskType.SOLVE_BOUNTY)),
                payload={**sub_payload, "bounty_id": bounty_id},
                offered_reward_usdc=reward_per_task,
                deadline_hours=deadline_hours,
                required_trust_level=0.5,  # PARTNER normalised ≈ 0.5
            )
            sub_tasks.append(task)

        self._active_bounties[bounty_id] = [
            SubTaskRecord(task=t, peer_instance_id="") for t in sub_tasks
        ]
        self._bounties_split += 1

        from systems.synapse.types import SynapseEventType as _SET
        self._emit(_SET.FEDERATION_BOUNTY_SPLIT, {
            "bounty_id": bounty_id,
            "sub_task_count": n,
            "total_reward_usdc": str(total_reward_usdc),
            "orchestrator_instance_id": self._instance_id,
            "timestamp": utc_now().isoformat(),
        })

        self._logger.info(
            "bounty_split",
            bounty_id=bounty_id,
            sub_tasks=n,
            reward_per_task=str(reward_per_task),
        )
        return sub_tasks

    async def broadcast_sub_tasks(
        self,
        bounty_id: str,
        trusted_links: list[FederationLink],
    ) -> dict[str, str]:
        """
        Offer each sub-task to a peer in round-robin order, highest trust first.

        Returns a mapping of task_id → peer_instance_id.
        """
        records = self._active_bounties.get(bounty_id, [])
        if not records:
            return {}

        # Sort peers: PARTNER+ by descending trust score
        eligible = sorted(
            [l for l in trusted_links if l.trust_level.value >= TrustLevel.PARTNER.value],
            key=lambda l: l.trust_score,
            reverse=True,
        )
        if not eligible:
            self._logger.warning("broadcast_no_eligible_peers", bounty_id=bounty_id)
            return {}

        assignments: dict[str, str] = {}
        for i, record in enumerate(records):
            peer_link = eligible[i % len(eligible)]
            record.peer_instance_id = peer_link.remote_instance_id
            assignments[record.task.id] = peer_link.remote_instance_id
            self._logger.info(
                "sub_task_offered",
                task_id=record.task.id,
                peer=peer_link.remote_instance_id,
                bounty_id=bounty_id,
            )
        return assignments

    async def collect_results(
        self,
        bounty_id: str,
        deadline_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """
        Wait for sub-task results.  Non-blocking - callers should poll
        ``get_results()`` or subscribe to FEDERATION_TASK_COMPLETED events.
        Returns whatever results are already available.
        """
        records = self._active_bounties.get(bounty_id, [])
        completed = [
            r.result for r in records
            if r.result is not None and r.task.status == TaskStatus.COMPLETED
        ]
        return [r for r in completed if r is not None]

    def record_sub_task_result(
        self,
        bounty_id: str,
        task_id: str,
        result: dict[str, Any],
    ) -> None:
        """Called by service layer when a sub-task result arrives."""
        records = self._active_bounties.get(bounty_id, [])
        for record in records:
            if record.task.id == task_id:
                record.result = result
                record.task.status = TaskStatus.COMPLETED
                record.task.completed_at = utc_now()
                self._sub_tasks_completed += 1
                break

    def merge_and_submit(
        self,
        bounty_id: str,
    ) -> dict[str, Any] | None:
        """
        Aggregate all available sub-results into a unified solution payload.

        Returns None if no results are ready.
        """
        records = self._active_bounties.get(bounty_id, [])
        available = [r for r in records if r.result is not None]
        if not available:
            return None

        merged: dict[str, Any] = {
            "bounty_id": bounty_id,
            "orchestrator": self._instance_id,
            "sub_task_count": len(records),
            "completed_count": len(available),
            "sub_results": [
                {
                    "task_id": r.task.id,
                    "peer": r.peer_instance_id,
                    "result": r.result,
                }
                for r in available
            ],
            "merged_at": utc_now().isoformat(),
        }

        # If all sub-tasks include a "code_patch" key, concatenate them
        patches = [r.result.get("code_patch", "") for r in available if r.result]
        if all(patches):
            merged["unified_patch"] = "\n".join(patches)

        return merged

    async def on_bounty_paid(
        self,
        bounty_id: str,
        paid_usdc: Decimal,
        trusted_links: list[FederationLink],
    ) -> None:
        """
        Distribute the bounty payout among contributors proportionally.

        Each contributing peer receives (1 - _ORCHESTRATOR_CUT) / n of
        the total, paid via TaskDelegationManager.settle_payment().
        """
        records = self._active_bounties.get(bounty_id, [])
        contributors = [r for r in records if r.task.status == TaskStatus.COMPLETED]
        if not contributors:
            return

        n = len(contributors)
        share = (paid_usdc * (Decimal("1") - _ORCHESTRATOR_CUT) / n).quantize(
            Decimal("0.01")
        )

        link_map = {l.remote_instance_id: l for l in trusted_links}
        for record in contributors:
            link = link_map.get(record.peer_instance_id)
            if not link:
                continue
            record.task.offered_reward_usdc = share
            await self._delegation.settle_payment(record.task, link)

        self._logger.info(
            "bounty_paid_distributed",
            bounty_id=bounty_id,
            total_paid=str(paid_usdc),
            share_per_contributor=str(share),
            contributors=n,
        )
        # Clean up
        self._active_bounties.pop(bounty_id, None)

    # ─── Private helpers ─────────────────────────────────────────────

    async def _generate_sub_payloads(
        self,
        bounty_payload: dict[str, Any],
        max_sub_tasks: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Ask Simula to propose sub-task boundaries.
        Falls back to simple chunked splitting if Simula unavailable.
        """
        if self._simula is not None:
            try:
                sub_payloads = await self._simula_split(bounty_payload, max_sub_tasks)
                if sub_payloads:
                    return sub_payloads
            except Exception as exc:
                self._logger.warning("simula_split_failed", error=str(exc))

        # Fallback: simple N-way split on description
        description = str(bounty_payload.get("description", ""))
        words = description.split()
        chunk_size = max(1, len(words) // max_sub_tasks)
        sub_payloads = []
        for i in range(min(max_sub_tasks, max(1, len(words) // chunk_size))):
            chunk = words[i * chunk_size: (i + 1) * chunk_size]
            sub_payloads.append({
                **bounty_payload,
                "description": " ".join(chunk),
                "sub_task_index": i,
                "total_sub_tasks": min(max_sub_tasks, len(words) // chunk_size or 1),
            })
        return sub_payloads or [bounty_payload]

    async def _simula_split(
        self,
        bounty_payload: dict[str, Any],
        max_sub_tasks: int,
    ) -> list[dict[str, Any]]:
        """Request Simula to generate intelligent task boundaries via Synapse."""
        if self._event_bus is None:
            return []

        result_fut: asyncio.Future[list[dict[str, Any]]] = asyncio.get_event_loop().create_future()

        async def _on_split_response(event: Any) -> None:
            payload = getattr(event, "payload", {})
            if payload.get("request_bounty_id") == bounty_payload.get("bounty_id"):
                sub = payload.get("sub_tasks", [])
                if not result_fut.done():
                    result_fut.set_result(sub)

        try:
            from systems.synapse.types import SynapseEventType
            self._event_bus.subscribe(
                SynapseEventType.EVOLUTION_APPLIED, _on_split_response
            )
            self._event_bus.emit(
                SynapseEventType.FEDERATION_BOUNTY_SPLIT,
                {
                    "request_bounty_id": bounty_payload.get("bounty_id", ""),
                    "payload": bounty_payload,
                    "max_sub_tasks": max_sub_tasks,
                    "requestor": self._instance_id,
                },
            )
            return await asyncio.wait_for(result_fut, timeout=10.0)
        except asyncio.TimeoutError:
            return []

    def _emit(self, event_type: "SynapseEventType | str", payload: dict[str, Any]) -> None:
        if not self._event_bus:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            if isinstance(event_type, SynapseEventType):
                etype = event_type
            else:
                etype = SynapseEventType(event_type.lower())
            asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                event_type=etype,
                source_system="federation",
                data=payload,
            )))
        except Exception as exc:
            self._logger.error("emit_failed", event_type=event_type, error=str(exc))

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "bounties_split": self._bounties_split,
            "sub_tasks_completed": self._sub_tasks_completed,
            "active_bounties": len(self._active_bounties),
        }
