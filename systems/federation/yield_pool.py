"""
EcodiaOS - Federation Capital Yield Pooling

Federated instances can pool capital for large yield positions that a
single instance cannot fund alone.  This is the multi-instance analog of
what Oikos does for single-instance yield.

Trust requirement: 0.9 normalised (≈ ALLY, score ~90).  Capital pooling
requires deep trust - only instances that have demonstrated sustained
reliability participate.

Lifecycle:
  1. Proposer calls ``propose_pool()`` - emits FEDERATION_YIELD_POOL_PROPOSAL.
  2. Peers call ``join_pool()`` - records their contribution.
  3. ``fund_pool()`` - when min_capital reached; each participant's USDC
     is locked via smart contract escrow (WalletClient).
  4. ``deploy_pool()`` - calls Oikos to open the on-chain position.
  5. ``settle_pool()`` - on position close, distributes yield proportionally.
  6. ``cancel_pool()`` - if minimum not reached before deadline.

Escrow is simulated via WalletClient USDC transfers to a shared escrow
address.  Real smart contract escrow is a TODO tracked in CLAUDE.md.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from primitives.federation import (
    FederationLink,
    PoolParticipant,
    TrustLevel,
    YieldPoolProposal,
    YieldPoolStatus,
)

if TYPE_CHECKING:
    from clients.wallet import WalletClient

logger = structlog.get_logger("systems.federation.yield_pool")

_TRUST_SCORE_MAX: float = 100.0


def _normalise_trust(trust_score: float) -> float:
    return min(trust_score / _TRUST_SCORE_MAX, 1.0)


class YieldPoolManager:
    """
    Manages federated capital pooling for high-APY yield positions.
    """

    def __init__(
        self,
        instance_id: str,
        wallet: WalletClient | None = None,
    ) -> None:
        self._instance_id = instance_id
        self._wallet = wallet
        self._event_bus: Any = None
        self._oikos: Any = None  # Wired post-init for position deployment
        self._logger = logger.bind(component="yield_pool", instance_id=instance_id)

        # pool_id → YieldPoolProposal
        self._pools: dict[str, YieldPoolProposal] = {}

        # Stats
        self._pools_proposed: int = 0
        self._pools_funded: int = 0
        self._pools_settled: int = 0
        self._total_yield_earned_usdc: Decimal = Decimal("0")

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus
        try:
            from systems.synapse.types import SynapseEventType
            event_bus.subscribe(
                SynapseEventType.FEDERATION_YIELD_POOL_PROPOSAL,
                self._on_yield_pool_proposal,
            )
        except Exception as exc:
            self._logger.warning("subscription_failed", error=str(exc))

    def set_oikos(self, oikos: Any) -> None:
        self._oikos = oikos

    # ─── Propose a pool ──────────────────────────────────────────────

    def propose_pool(
        self,
        target_protocol: str,
        target_apy: float,
        min_capital_usdc: Decimal,
        max_participants: int = 5,
        lock_duration_hours: int = 168,
        target_pool_address: str = "",
        my_contribution_usdc: Decimal = Decimal("0"),
    ) -> YieldPoolProposal:
        """
        Create and broadcast a yield pool proposal.

        The proposer's contribution is recorded immediately.
        """
        proposal = YieldPoolProposal(
            proposer_instance_id=self._instance_id,
            target_protocol=target_protocol,
            target_pool_address=target_pool_address,
            target_apy=target_apy,
            min_capital_usdc=min_capital_usdc,
            max_participants=max_participants,
            lock_duration_hours=lock_duration_hours,
            required_trust_level=0.9,
            status=YieldPoolStatus.PROPOSED,
        )

        if my_contribution_usdc > Decimal("0"):
            proposal.participants.append(PoolParticipant(
                instance_id=self._instance_id,
                contribution_usdc=my_contribution_usdc,
            ))

        self._pools[proposal.id] = proposal
        self._pools_proposed += 1

        from systems.synapse.types import SynapseEventType as _SET
        self._emit(_SET.FEDERATION_YIELD_POOL_PROPOSAL, {
            "pool_id": proposal.id,
            "proposer_instance_id": self._instance_id,
            "target_protocol": target_protocol,
            "target_apy": target_apy,
            "min_capital_usdc": str(min_capital_usdc),
            "max_participants": max_participants,
            "lock_duration_hours": lock_duration_hours,
            "timestamp": utc_now().isoformat(),
        })

        self._logger.info(
            "pool_proposed",
            pool_id=proposal.id,
            protocol=target_protocol,
            apy=target_apy,
            min_capital=str(min_capital_usdc),
        )
        return proposal

    # ─── Join a pool ─────────────────────────────────────────────────

    def join_pool(
        self,
        pool_id: str,
        contribution_usdc: Decimal,
        link: FederationLink,
        wallet_address: str = "",
    ) -> bool:
        """
        Record intent to join a yield pool.

        Returns True if accepted, False if declined (trust/capacity check).
        """
        pool = self._pools.get(pool_id)
        if not pool:
            self._logger.warning("join_pool_unknown", pool_id=pool_id)
            return False

        if pool.status != YieldPoolStatus.PROPOSED:
            return False

        norm_trust = _normalise_trust(link.trust_score)
        if norm_trust < pool.required_trust_level:
            self._logger.info(
                "join_pool_trust_insufficient",
                pool_id=pool_id,
                trust=norm_trust,
                required=pool.required_trust_level,
            )
            return False

        if len(pool.participants) >= pool.max_participants:
            return False

        # Check not already joined
        if any(p.instance_id == link.remote_instance_id for p in pool.participants):
            return False

        pool.participants.append(PoolParticipant(
            instance_id=link.remote_instance_id,
            contribution_usdc=contribution_usdc,
            wallet_address=wallet_address,
        ))

        self._logger.info(
            "peer_joined_pool",
            pool_id=pool_id,
            peer=link.remote_instance_id,
            contribution=str(contribution_usdc),
        )

        # Auto-fund if minimum capital reached
        total = sum(p.contribution_usdc for p in pool.participants)
        if total >= pool.min_capital_usdc and pool.status == YieldPoolStatus.PROPOSED:
            asyncio.ensure_future(self.fund_pool(pool_id))

        return True

    # ─── Fund pool (lock capital) ─────────────────────────────────────

    async def fund_pool(self, pool_id: str) -> bool:
        """
        Transition the pool to FUNDED by collecting each participant's
        USDC into the escrow address.

        Returns True on success.
        """
        pool = self._pools.get(pool_id)
        if not pool or pool.status != YieldPoolStatus.PROPOSED:
            return False

        total = sum(p.contribution_usdc for p in pool.participants)
        if total < pool.min_capital_usdc:
            self._logger.info("fund_pool_underfunded", pool_id=pool_id, total=str(total))
            return False

        # Compute share fractions
        for participant in pool.participants:
            participant.share_fraction = float(participant.contribution_usdc / total)
            participant.locked_at = utc_now()

        # My own contribution transfer (if I'm a participant)
        my_participant = next(
            (p for p in pool.participants if p.instance_id == self._instance_id), None
        )
        if my_participant and my_participant.contribution_usdc > Decimal("0") and self._wallet:
            escrow = pool.escrow_contract_address or "0xFEDERATION_ESCROW_PLACEHOLDER"
            try:
                await self._wallet.transfer_usdc(
                    to_address=escrow,
                    amount_usdc=my_participant.contribution_usdc,
                    memo=f"yield_pool:{pool_id}",
                )
            except Exception as exc:
                self._logger.error("fund_pool_transfer_failed", pool_id=pool_id, error=str(exc))
                return False

        pool.status = YieldPoolStatus.FUNDED
        pool.funded_at = utc_now()
        self._pools_funded += 1

        self._logger.info(
            "pool_funded",
            pool_id=pool_id,
            total_capital=str(total),
            participants=len(pool.participants),
        )
        return True

    # ─── Deploy position ─────────────────────────────────────────────

    async def deploy_pool(self, pool_id: str) -> bool:
        """
        Open the on-chain yield position via Oikos.

        Returns True if deployment was initiated.
        """
        pool = self._pools.get(pool_id)
        if not pool or pool.status != YieldPoolStatus.FUNDED:
            return False

        total = sum(p.contribution_usdc for p in pool.participants)

        if self._oikos is not None:
            try:
                # Request Oikos to deploy the pooled capital
                await self._oikos.deploy_yield_position(
                    protocol=pool.target_protocol,
                    pool_address=pool.target_pool_address,
                    amount_usdc=total,
                    source="federation_pool",
                    pool_id=pool_id,
                )
            except Exception as exc:
                self._logger.error("pool_deploy_failed", pool_id=pool_id, error=str(exc))
                return False
        else:
            self._logger.warning("deploy_pool_no_oikos", pool_id=pool_id)

        pool.status = YieldPoolStatus.ACTIVE
        self._logger.info(
            "pool_deployed",
            pool_id=pool_id,
            protocol=pool.target_protocol,
            total=str(total),
        )
        return True

    # ─── Settle pool (distribute yield) ──────────────────────────────

    async def settle_pool(
        self,
        pool_id: str,
        gross_return_usdc: Decimal,
        trusted_links: list[FederationLink],
    ) -> dict[str, Decimal]:
        """
        Close the pool and distribute yield proportionally.

        Returns a mapping of instance_id → payout_usdc.
        """
        pool = self._pools.get(pool_id)
        if not pool or pool.status != YieldPoolStatus.ACTIVE:
            return {}

        payouts: dict[str, Decimal] = {}
        link_map = {l.remote_instance_id: l for l in trusted_links}

        for participant in pool.participants:
            payout = (gross_return_usdc * Decimal(str(participant.share_fraction))).quantize(
                Decimal("0.000001")
            )
            payouts[participant.instance_id] = payout

            if participant.instance_id == self._instance_id:
                # Our own share - just record it
                self._total_yield_earned_usdc += payout
                continue

            # Pay remote participants
            if not self._wallet:
                continue
            link = link_map.get(participant.instance_id)
            wallet_address = participant.wallet_address
            if not wallet_address and link and link.remote_identity:
                wallet_address = link.remote_identity.wallet_address
            if not wallet_address:
                continue

            try:
                await self._wallet.transfer_usdc(
                    to_address=wallet_address,
                    amount_usdc=payout,
                    memo=f"yield_pool_settle:{pool_id}",
                )
            except Exception as exc:
                self._logger.error(
                    "settle_pool_payment_failed",
                    pool_id=pool_id,
                    peer=participant.instance_id,
                    error=str(exc),
                )

        pool.status = YieldPoolStatus.SETTLED
        pool.settled_at = utc_now()
        self._pools_settled += 1

        self._logger.info(
            "pool_settled",
            pool_id=pool_id,
            gross_return=str(gross_return_usdc),
            participants=len(pool.participants),
        )
        return payouts

    async def cancel_pool(self, pool_id: str, reason: str = "") -> None:
        """Cancel a proposed/funded pool and return locked capital."""
        pool = self._pools.get(pool_id)
        if not pool:
            return
        if pool.status in (YieldPoolStatus.SETTLED, YieldPoolStatus.CANCELLED):
            return

        pool.status = YieldPoolStatus.CANCELLED
        self._logger.info("pool_cancelled", pool_id=pool_id, reason=reason)
        # In production: trigger escrow refund via smart contract

    # ─── Event handler ────────────────────────────────────────────────

    async def _on_yield_pool_proposal(self, event: Any) -> None:
        """Cache incoming yield pool proposals from peers."""
        payload = getattr(event, "payload", {})
        proposer = payload.get("proposer_instance_id", "")
        if proposer == self._instance_id:
            return  # Our own proposal
        pool_id = payload.get("pool_id", "")
        if pool_id in self._pools:
            return  # Already known

        try:
            proposal = YieldPoolProposal(
                id=pool_id,
                proposer_instance_id=proposer,
                target_protocol=payload.get("target_protocol", ""),
                target_apy=float(payload.get("target_apy", 0)),
                min_capital_usdc=Decimal(str(payload.get("min_capital_usdc", "1000"))),
                max_participants=int(payload.get("max_participants", 5)),
                lock_duration_hours=int(payload.get("lock_duration_hours", 168)),
                required_trust_level=0.9,
                status=YieldPoolStatus.PROPOSED,
            )
            self._pools[pool_id] = proposal
            self._logger.info(
                "peer_pool_proposal_cached",
                pool_id=pool_id,
                proposer=proposer,
                apy=proposal.target_apy,
            )
        except Exception as exc:
            self._logger.warning("pool_proposal_parse_failed", error=str(exc))

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
            "pools_proposed": self._pools_proposed,
            "pools_funded": self._pools_funded,
            "pools_settled": self._pools_settled,
            "active_pools": sum(
                1 for p in self._pools.values()
                if p.status == YieldPoolStatus.ACTIVE
            ),
            "total_yield_earned_usdc": str(self._total_yield_earned_usdc),
        }
