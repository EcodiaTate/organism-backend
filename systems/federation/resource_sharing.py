"""
EcodiaOS - Federation Resource Sharing (Compute Offloading)

When an instance is under high cognitive load (ORGANISM_TELEMETRY shows
CPU > 85%), it can offload heavy analysis tasks to federation peers with
spare capacity.  The requesting instance pays a small flat USDC fee per
offloaded task.

Inverse path: when an instance has spare capacity, it advertises
availability via ``FEDERATION_CAPACITY_AVAILABLE`` and accepts incoming
offload requests routed by the marketplace or direct federation call.

Resource sharing is gated at COLLEAGUE+ trust - cheaper than task
delegation (which defaults to PARTNER) because the compute payloads are
smaller and less economically sensitive.

Architecture:
  ResourceSharingManager
    ├── publish_capacity()   - broadcast CapacityOffer to all COLLEAGUE+ peers
    ├── request_offload()    - send OffloadRequest to a specific peer
    ├── handle_offload()     - receive and execute an offload from a peer
    ├── _on_telemetry()      - subscribe to ORGANISM_TELEMETRY; auto-offload
    │                          when cpu_total > HIGH_LOAD_THRESHOLD
    └── _on_capacity()       - cache peer CapacityOffer advertisements
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from primitives.federation import (
    CapacityOffer,
    FederationInteraction,
    FederationLink,
    InteractionOutcome,
    OffloadRequest,
    TaskType,
    TrustLevel,
)

if TYPE_CHECKING:
    from clients.wallet import WalletClient
    from systems.federation.task_delegation import TaskDelegationManager

logger = structlog.get_logger("systems.federation.resource_sharing")

# CPU% above which auto-offload kicks in
_HIGH_LOAD_THRESHOLD: float = 85.0
# CPU% below which this instance advertises spare capacity
_SPARE_CAPACITY_THRESHOLD: float = 40.0


class ResourceSharingManager:
    """
    Manages compute offloading and capacity advertisement.

    One instance lives inside FederationService.
    """

    def __init__(
        self,
        instance_id: str,
        wallet: WalletClient | None = None,
        delegation: TaskDelegationManager | None = None,
    ) -> None:
        self._instance_id = instance_id
        self._wallet = wallet
        self._delegation = delegation
        self._event_bus: Any = None
        self._logger = logger.bind(component="resource_sharing", instance_id=instance_id)

        # My current capacity offer (None if not advertising)
        self._my_offer: CapacityOffer | None = None

        # Peer capacity cache: instance_id → CapacityOffer
        self._peer_offers: dict[str, CapacityOffer] = {}

        # Active offloads: request_id → OffloadRequest
        self._active_offloads: dict[str, OffloadRequest] = {}

        # Stats
        self._offloads_sent: int = 0
        self._offloads_received: int = 0
        self._offloads_completed: int = 0
        self._total_fees_paid_usdc: Decimal = Decimal("0")
        self._total_fees_earned_usdc: Decimal = Decimal("0")

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus
        # Subscribe to ORGANISM_TELEMETRY for auto-offload triggering
        # and to FEDERATION_CAPACITY_AVAILABLE for peer caching
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.ORGANISM_TELEMETRY,
                self._on_organism_telemetry,
            )
            event_bus.subscribe(
                SynapseEventType.FEDERATION_CAPACITY_AVAILABLE,
                self._on_capacity_available,
            )
        except Exception as exc:
            self._logger.warning("subscription_failed", error=str(exc))

    # ─── Capacity advertisement ───────────────────────────────────────

    def publish_capacity(
        self,
        available_cycles_per_hour: int,
        cost_usdc_per_task: Decimal,
        specialisations: list[str],
        ttl_hours: int = 1,
    ) -> CapacityOffer:
        """
        Broadcast this instance's spare compute capacity to the federation.

        Emits ``FEDERATION_CAPACITY_AVAILABLE`` on the Synapse bus so peers
        can cache the offer.  Calling this again overwrites the previous offer.
        """
        from datetime import timedelta
        offer = CapacityOffer(
            instance_id=self._instance_id,
            available_cycles_per_hour=available_cycles_per_hour,
            cost_usdc_per_task=cost_usdc_per_task,
            specialisations=specialisations,
            expires_at=utc_now() + timedelta(hours=ttl_hours),
        )
        self._my_offer = offer

        from systems.synapse.types import SynapseEventType as _SET
        self._emit(_SET.FEDERATION_CAPACITY_AVAILABLE, {
            "instance_id": self._instance_id,
            "available_cycles_per_hour": available_cycles_per_hour,
            "cost_usdc_per_task": str(cost_usdc_per_task),
            "specialisations": specialisations,
            "expires_at": offer.expires_at.isoformat(),
            "timestamp": utc_now().isoformat(),
        })

        self._logger.info(
            "capacity_published",
            cycles=available_cycles_per_hour,
            cost=str(cost_usdc_per_task),
            specialisations=specialisations,
        )
        return offer

    def retract_capacity(self) -> None:
        """Withdraw this instance's capacity advertisement."""
        self._my_offer = None

    # ─── Outbound: request offload to a peer ─────────────────────────

    async def request_offload(
        self,
        task_type: TaskType,
        payload: dict[str, Any],
        target_link: FederationLink,
        fee_usdc: Decimal | None = None,
    ) -> OffloadRequest | None:
        """
        Request that a peer executes a heavy task.

        Selects fee automatically from the peer's cached offer if
        ``fee_usdc`` is not provided.  Returns None if the peer has no
        known offer or trust is insufficient.
        """
        if target_link.trust_level.value < TrustLevel.COLLEAGUE.value:
            self._logger.info(
                "offload_trust_insufficient",
                peer=target_link.remote_instance_id,
            )
            return None

        peer_offer = self._peer_offers.get(target_link.remote_instance_id)
        if fee_usdc is None:
            fee_usdc = peer_offer.cost_usdc_per_task if peer_offer else Decimal("0.1")

        req = OffloadRequest(
            requesting_instance_id=self._instance_id,
            task_type=task_type,
            payload=payload,
            offered_fee_usdc=fee_usdc,
            target_instance_id=target_link.remote_instance_id,
        )
        self._active_offloads[req.id] = req
        self._offloads_sent += 1

        self._logger.info(
            "offload_requested",
            peer=target_link.remote_instance_id,
            task_type=task_type,
            fee=str(fee_usdc),
        )
        return req

    def on_offload_result(
        self,
        request_id: str,
        result: dict[str, Any],
        peer_instance_id: str,
        link: FederationLink,
    ) -> None:
        """Called when a peer returns an offload result."""
        req = self._active_offloads.pop(request_id, None)
        if not req:
            return
        self._offloads_completed += 1

        # Trigger payment
        if req.offered_fee_usdc > Decimal("0") and self._delegation:
            # Re-use TaskDelegationManager payment path with a synthetic task
            from primitives.federation import TaskDelegation, TaskStatus
            synthetic = TaskDelegation(
                task_type=req.task_type,
                payload=req.payload,
                offered_reward_usdc=req.offered_fee_usdc,
                delegating_instance_id=self._instance_id,
                accepting_instance_id=peer_instance_id,
                status=TaskStatus.COMPLETED,
            )
            asyncio.ensure_future(self._delegation.settle_payment(synthetic, link))
            self._total_fees_paid_usdc += req.offered_fee_usdc

        self._logger.info(
            "offload_result_received",
            request_id=request_id,
            peer=peer_instance_id,
        )

    # ─── Inbound: handle an offload from a peer ───────────────────────

    async def handle_offload(
        self,
        request: OffloadRequest,
        link: FederationLink,
    ) -> tuple[dict[str, Any] | None, FederationInteraction]:
        """
        Execute an offloaded task from a peer.

        Returns (result_dict, interaction).  The result is sent back to
        the peer via the channel layer (FederationService handles the
        HTTP response).

        This is a best-effort implementation: we run the work synchronously
        within the handler.  In production, long tasks should be queued.
        """
        start = utc_now()

        if link.trust_level.value < TrustLevel.COLLEAGUE.value:
            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=link.remote_instance_id,
                interaction_type="compute_offload",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description="Offload declined: insufficient trust",
                trust_value=0.0,
                latency_ms=_elapsed_ms(start),
            )
            return None, interaction

        if self._my_offer is None:
            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=link.remote_instance_id,
                interaction_type="compute_offload",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description="No capacity advertised - offload declined",
                trust_value=0.0,
                latency_ms=_elapsed_ms(start),
            )
            return None, interaction

        self._offloads_received += 1

        # Execute the task (simple analysis stub - real execution via Axon)
        result = await self._execute_offload(request)

        # Earn the fee
        self._total_fees_earned_usdc += request.offered_fee_usdc

        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type="compute_offload",
            direction="inbound",
            outcome=InteractionOutcome.SUCCESSFUL,
            description=f"Executed offload {request.task_type}",
            trust_value=1.0,
            latency_ms=_elapsed_ms(start),
        )
        return result, interaction

    async def _execute_offload(self, request: OffloadRequest) -> dict[str, Any]:
        """
        Stub execution of an offloaded task.

        In production this would dispatch to the appropriate Axon action.
        For now we return a placeholder result that confirms receipt.
        """
        return {
            "executed_by": self._instance_id,
            "task_type": request.task_type,
            "status": "completed",
            "payload_keys": list(request.payload.keys()),
            "timestamp": utc_now().isoformat(),
        }

    # ─── Event handlers ───────────────────────────────────────────────

    async def _on_organism_telemetry(self, event: Any) -> None:
        """
        Watch CPU load.  If > HIGH_LOAD_THRESHOLD, retract our own capacity
        advertisement.  If < SPARE_CAPACITY_THRESHOLD, re-publish if needed.
        """
        payload = getattr(event, "payload", {})
        cpu_per_system: dict[str, float] = payload.get("cpu_per_system", {})
        total_cpu = sum(cpu_per_system.values()) if cpu_per_system else 0.0

        if total_cpu > _HIGH_LOAD_THRESHOLD and self._my_offer is not None:
            self.retract_capacity()
            self._logger.info("capacity_retracted_high_load", cpu=total_cpu)
        elif total_cpu < _SPARE_CAPACITY_THRESHOLD and self._my_offer is None:
            # Auto-republish with conservative estimate
            available_cycles = int((_SPARE_CAPACITY_THRESHOLD - total_cpu) * 100)
            self.publish_capacity(
                available_cycles_per_hour=available_cycles,
                cost_usdc_per_task=Decimal("0.05"),
                specialisations=[],
            )

    async def _on_capacity_available(self, event: Any) -> None:
        """Cache a peer's capacity advertisement."""
        payload = getattr(event, "payload", {})
        instance_id = payload.get("instance_id", "")
        if not instance_id or instance_id == self._instance_id:
            return

        from datetime import datetime, timezone
        expires_raw = payload.get("expires_at", "")
        try:
            expires = datetime.fromisoformat(expires_raw)
        except (ValueError, TypeError):
            from datetime import timedelta
            expires = utc_now() + timedelta(hours=1)

        offer = CapacityOffer(
            instance_id=instance_id,
            available_cycles_per_hour=int(payload.get("available_cycles_per_hour", 0)),
            cost_usdc_per_task=Decimal(str(payload.get("cost_usdc_per_task", "0.1"))),
            specialisations=payload.get("specialisations", []),
            expires_at=expires,
        )
        self._peer_offers[instance_id] = offer
        self._logger.debug("peer_capacity_cached", peer=instance_id)

    # ─── Utilities ────────────────────────────────────────────────────

    def best_peer_for_offload(
        self,
        task_type: TaskType,
        trusted_links: list[FederationLink],
        specialisation: str = "",
    ) -> FederationLink | None:
        """
        Pick the best peer to receive a compute offload.

        Selection criteria (priority order):
          1. Has a valid CapacityOffer
          2. Matches requested specialisation (if provided)
          3. Lowest cost per task
          4. Highest trust score (tiebreak)
        """
        from datetime import datetime, timezone
        now = utc_now()
        eligible = []
        for link in trusted_links:
            if link.trust_level.value < TrustLevel.COLLEAGUE.value:
                continue
            offer = self._peer_offers.get(link.remote_instance_id)
            if not offer or offer.expires_at < now:
                continue
            if specialisation and specialisation not in offer.specialisations:
                continue
            eligible.append((offer.cost_usdc_per_task, -link.trust_score, link))

        if not eligible:
            return None
        eligible.sort(key=lambda x: (x[0], x[1]))
        return eligible[0][2]

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

    def _elapsed_ms_from(self, start: Any) -> int:
        return _elapsed_ms(start)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "my_offer_active": self._my_offer is not None,
            "peer_offers_cached": len(self._peer_offers),
            "offloads_sent": self._offloads_sent,
            "offloads_received": self._offloads_received,
            "offloads_completed": self._offloads_completed,
            "total_fees_paid_usdc": str(self._total_fees_paid_usdc),
            "total_fees_earned_usdc": str(self._total_fees_earned_usdc),
        }


def _elapsed_ms(start: Any) -> int:
    delta = utc_now() - start
    return int(delta.total_seconds() * 1000)
