"""
EcodiaOS - Federation Task Delegation

Enables one instance to delegate discrete work units to trusted federation
peers with automatic USDC payment on completion.

Workflow:
  1. Delegating instance calls ``offer_task()`` with a TaskDelegation and
     the target link.
  2. Remote peer receives the offer via ``handle_offered_task()``; trust is
     verified (COLLEAGUE+, normalised trust ≥ ``required_trust_level``),
     Nova alignment and Equor reviewed, then Accept/Decline is returned.
  3. Accepting peer executes the work and calls ``submit_result()``; the
     result travels back to the delegating instance via ``handle_result()``.
  4. On success the delegating instance calls ``settle_payment()`` which
     transfers ``offered_reward_usdc`` via WalletClient and emits
     ``FEDERATION_TASK_PAYMENT``.

Trust gating:
  - Minimum structural level: COLLEAGUE (score ≥ 20)
  - Normalised trust threshold: ``task.required_trust_level`` (default 0.7)

All state is in-process.  Tasks are not persisted to Neo4j in this
implementation - add Neo4j write-through if audit trail is required.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from primitives.federation import (
    DelegationResult,
    FederationInteraction,
    FederationLink,
    InteractionOutcome,
    TaskDelegation,
    TaskStatus,
    TaskType,
    TrustLevel,
)

if TYPE_CHECKING:
    from clients.wallet import WalletClient

logger = structlog.get_logger("systems.federation.task_delegation")

# Normalised trust score ceiling for ALLY (score = 100) - used for
# converting raw trust score to 0-1 range.
_TRUST_SCORE_MAX: float = 100.0


def _normalise_trust(trust_score: float) -> float:
    """Convert raw trust score (0–100) to normalised 0-1 range."""
    return min(trust_score / _TRUST_SCORE_MAX, 1.0)


def _elapsed_ms(start: datetime) -> int:
    delta = utc_now() - start
    return int(delta.total_seconds() * 1000)


class TaskDelegationManager:
    """
    Manages outbound task offers and inbound task execution.

    Each instance carries one TaskDelegationManager.  It tracks tasks
    offered to peers (outbound) and tasks accepted from peers (inbound).
    """

    def __init__(self, instance_id: str, wallet: WalletClient | None = None) -> None:
        self._instance_id = instance_id
        self._wallet = wallet
        self._event_bus: Any = None
        self._logger = logger.bind(component="task_delegation", instance_id=instance_id)

        # task_id → TaskDelegation (tasks we offered to peers)
        self._outbound_tasks: dict[str, TaskDelegation] = {}
        # task_id → TaskDelegation (tasks accepted from peers)
        self._inbound_tasks: dict[str, TaskDelegation] = {}

        # Stats
        self._tasks_offered: int = 0
        self._tasks_accepted_by_peers: int = 0
        self._tasks_completed: int = 0
        self._tasks_failed: int = 0
        self._total_paid_usdc: Decimal = Decimal("0")

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    # ─── Outbound: offer a task to a remote peer ─────────────────────

    def build_task(
        self,
        task_type: TaskType,
        payload: dict[str, Any],
        offered_reward_usdc: Decimal,
        deadline_hours: int = 24,
        required_trust_level: float = 0.7,
    ) -> TaskDelegation:
        """Build a TaskDelegation ready to offer to a remote peer."""
        task = TaskDelegation(
            task_type=task_type,
            payload=payload,
            offered_reward_usdc=offered_reward_usdc,
            deadline_hours=deadline_hours,
            required_trust_level=required_trust_level,
            delegating_instance_id=self._instance_id,
            status=TaskStatus.OFFERED,
        )
        self._outbound_tasks[task.id] = task
        self._tasks_offered += 1

        # Emit FEDERATION_TASK_OFFERED so WorkRouter + Benchmarks can observe
        if self._event_bus is not None:
            import asyncio
            import contextlib
            async def _emit_offered() -> None:
                from systems.synapse.types import SynapseEventType
                with contextlib.suppress(Exception):
                    await self._event_bus.emit(
                        SynapseEventType.FEDERATION_TASK_OFFERED,
                        {
                            "task_id": task.id,
                            "task_type": task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type),
                            "offered_reward_usdc": str(task.offered_reward_usdc),
                            "deadline_hours": task.deadline_hours,
                            "required_trust_level": task.required_trust_level,
                            "delegating_instance_id": task.delegating_instance_id,
                        },
                        source_system="federation",
                    )
            asyncio.ensure_future(_emit_offered())

        return task

    def on_peer_accepted(self, task_id: str, accepting_instance_id: str) -> None:
        """Record that a remote peer accepted our offered task."""
        task = self._outbound_tasks.get(task_id)
        if task:
            task.accepting_instance_id = accepting_instance_id
            task.status = TaskStatus.ACCEPTED
            task.accepted_at = utc_now()
            self._tasks_accepted_by_peers += 1
            self._logger.info(
                "peer_accepted_task",
                task_id=task_id,
                peer=accepting_instance_id,
                task_type=task.task_type,
            )

    def on_peer_declined(self, task_id: str, reason: str = "") -> None:
        """Record that a remote peer declined our offered task."""
        task = self._outbound_tasks.get(task_id)
        if task:
            task.status = TaskStatus.DECLINED
            self._logger.info("peer_declined_task", task_id=task_id, reason=reason)

    # ─── Inbound: handle a task offered to us by a peer ──────────────

    async def handle_offered_task(
        self,
        task: TaskDelegation,
        link: FederationLink,
        nova_aligned: bool = True,
        equor_permitted: bool = True,
    ) -> tuple[DelegationResult, FederationInteraction]:
        """
        Decide whether to accept a task offered by a remote peer.

        Trust gate: COLLEAGUE+ structural level AND normalised score ≥
        ``task.required_trust_level``.
        """
        start = utc_now()
        norm_trust = _normalise_trust(link.trust_score)

        # Trust gate
        if link.trust_level.value < TrustLevel.COLLEAGUE.value:
            return self._decline(
                task, link, start,
                reason="Insufficient trust level - COLLEAGUE+ required for task delegation.",
            )
        if norm_trust < task.required_trust_level:
            return self._decline(
                task, link, start,
                reason=f"Normalised trust {norm_trust:.2f} below required {task.required_trust_level:.2f}.",
            )
        if not nova_aligned:
            return self._decline(task, link, start, reason="Task does not align with current goals.")
        if not equor_permitted:
            return self._decline(task, link, start, reason="Constitutional review did not permit this task.")

        # Accept
        task.accepting_instance_id = self._instance_id
        task.status = TaskStatus.ACCEPTED
        task.accepted_at = utc_now()
        self._inbound_tasks[task.id] = task

        result = DelegationResult(
            task_id=task.id,
            accepted=True,
            completing_instance_id=self._instance_id,
        )
        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type="task_delegation",
            direction="inbound",
            outcome=InteractionOutcome.SUCCESSFUL,
            description=f"Accepted task {task.task_type} from {link.remote_instance_id}",
            trust_value=1.5,
            latency_ms=_elapsed_ms(start),
        )
        self._logger.info(
            "task_accepted",
            task_id=task.id,
            task_type=task.task_type,
            from_instance=link.remote_instance_id,
        )

        from systems.synapse.types import SynapseEventType as _SET
        self._emit(_SET.FEDERATION_TASK_ACCEPTED, {
            "task_id": task.id,
            "accepting_instance_id": self._instance_id,
            "estimated_completion_hours": float(task.deadline_hours) * 0.5,
            "timestamp": utc_now().isoformat(),
        })

        return result, interaction

    def _decline(
        self,
        task: TaskDelegation,
        link: FederationLink,
        start: datetime,
        reason: str,
    ) -> tuple[DelegationResult, FederationInteraction]:
        result = DelegationResult(
            task_id=task.id,
            accepted=False,
            error=reason,
            completing_instance_id=self._instance_id,
        )
        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type="task_delegation",
            direction="inbound",
            outcome=InteractionOutcome.FAILED,
            description=f"Declined task {task.id}: {reason}",
            trust_value=0.0,
            latency_ms=_elapsed_ms(start),
        )
        from systems.synapse.types import SynapseEventType as _SET
        self._emit(_SET.FEDERATION_TASK_DECLINED, {
            "task_id": task.id,
            "declining_instance_id": self._instance_id,
            "reason": reason,
            "timestamp": utc_now().isoformat(),
        })
        return result, interaction

    # ─── Inbound: submit result for a task we accepted ────────────────

    def submit_result(
        self,
        task_id: str,
        result: dict[str, Any],
        success: bool = True,
    ) -> DelegationResult | None:
        """
        Mark an inbound task as completed and return the result payload.
        The result is sent back to the delegating instance via the channel.
        """
        task = self._inbound_tasks.get(task_id)
        if not task:
            self._logger.warning("submit_result_unknown_task", task_id=task_id)
            return None

        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.completed_at = utc_now()

        delegation_result = DelegationResult(
            task_id=task_id,
            accepted=True,
            result=result,
            completing_instance_id=self._instance_id,
        )
        if success:
            self._tasks_completed += 1
        else:
            self._tasks_failed += 1

        from systems.synapse.types import SynapseEventType as _SET
        self._emit(_SET.FEDERATION_TASK_COMPLETED, {
            "task_id": task_id,
            "completing_instance_id": self._instance_id,
            "success": success,
            "result_summary": str(result)[:200],
            "reward_claimed_usdc": str(task.offered_reward_usdc),
            "timestamp": utc_now().isoformat(),
        })
        return delegation_result

    # ─── Outbound: handle result returned by a peer ──────────────────

    async def handle_result(
        self,
        result: DelegationResult,
        link: FederationLink,
    ) -> FederationInteraction:
        """
        Process a completed task result from a remote peer.
        Triggers payment if result is successful.
        """
        task = self._outbound_tasks.get(result.task_id)
        if not task:
            self._logger.warning("handle_result_unknown_task", task_id=result.task_id)
            return FederationInteraction(
                link_id=link.id,
                remote_instance_id=link.remote_instance_id,
                interaction_type="task_result",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description="Unknown task result received",
                trust_value=0.0,
            )

        if result.result is not None:
            task.status = TaskStatus.COMPLETED
            task.completed_at = utc_now()
            self._tasks_completed += 1
            # Trigger payment
            await self.settle_payment(task, link)
            outcome = InteractionOutcome.SUCCESSFUL
            trust_value = 3.0  # Successful task completion builds significant trust
        else:
            task.status = TaskStatus.FAILED
            self._tasks_failed += 1
            outcome = InteractionOutcome.FAILED
            trust_value = -1.0

        return FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type="task_result",
            direction="inbound",
            outcome=outcome,
            description=f"Task {task.id} {task.status}",
            trust_value=trust_value,
        )

    async def settle_payment(
        self,
        task: TaskDelegation,
        link: FederationLink,
    ) -> str:
        """
        Transfer the task reward to the completing peer's wallet.
        Returns the transaction hash (empty string on failure).
        """
        if task.offered_reward_usdc <= Decimal("0"):
            return ""
        if not self._wallet:
            self._logger.warning("settle_payment_no_wallet", task_id=task.id)
            return ""

        remote_wallet = ""
        if link.remote_identity:
            remote_wallet = link.remote_identity.wallet_address

        if not remote_wallet:
            self._logger.warning(
                "settle_payment_no_remote_wallet",
                task_id=task.id,
                remote_id=link.remote_instance_id,
            )
            return ""

        try:
            tx_hash = await self._wallet.transfer_usdc(
                to_address=remote_wallet,
                amount_usdc=task.offered_reward_usdc,
                memo=f"federation_task:{task.id}",
            )
            self._total_paid_usdc += task.offered_reward_usdc
            from systems.synapse.types import SynapseEventType as _SET
            self._emit(_SET.FEDERATION_TASK_PAYMENT, {
                "task_id": task.id,
                "payer_instance_id": self._instance_id,
                "payee_instance_id": link.remote_instance_id,
                "amount_usdc": str(task.offered_reward_usdc),
                "tx_hash": tx_hash or "",
                "timestamp": utc_now().isoformat(),
            })
            self._logger.info(
                "task_payment_sent",
                task_id=task.id,
                amount=str(task.offered_reward_usdc),
                to=link.remote_instance_id,
                tx_hash=tx_hash,
            )
            return tx_hash or ""
        except Exception as exc:
            self._logger.error("task_payment_failed", task_id=task.id, error=str(exc))
            return ""

    # ─── Emit helper ─────────────────────────────────────────────────

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

    # ─── Stats ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "tasks_offered": self._tasks_offered,
            "tasks_accepted_by_peers": self._tasks_accepted_by_peers,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "total_paid_usdc": str(self._total_paid_usdc),
            "active_outbound": sum(
                1 for t in self._outbound_tasks.values()
                if t.status in (TaskStatus.OFFERED, TaskStatus.ACCEPTED)
            ),
            "active_inbound": sum(
                1 for t in self._inbound_tasks.values()
                if t.status == TaskStatus.ACCEPTED
            ),
        }
