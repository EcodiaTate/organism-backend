"""
EcodiaOS - Federation Work Router (Nexus Specialisation Routing)

Nexus detects epistemic specialisation across federated instances - which
domains each instance has developed mastery in.  WorkRouter wires this
knowledge into task routing: Solidity bounties go to Solidity specialists,
Rust bounties go to Rust specialists.

Specialisation emerges from Evo's domain mastery tracking via
``NEXUS_EPISTEMIC_VALUE`` events, which carry per-instance knowledge
triangulation scores per domain.

The router maintains a live specialisation registry updated from:
  - ``NEXUS_EPISTEMIC_VALUE`` - epistemic depth per instance/domain
  - ``FEDERATION_CAPACITY_AVAILABLE`` - declared specialisations from peers
  - Task completion history - empirical success rates per domain

Routing algorithm (scored selection):
  score(peer, domain) =
      epistemic_depth(peer, domain) × 0.5
    + declared_specialisation(peer, domain) × 0.3
    + empirical_success_rate(peer, domain) × 0.2
    × trust_normalised(peer)

The router emits ``FEDERATION_WORK_ROUTED`` when it assigns a task to a
specific peer.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from primitives.federation import (
    FederationLink,
    TaskDelegation,
    TaskType,
    TrustLevel,
)

logger = structlog.get_logger("systems.federation.work_router")

_TRUST_SCORE_MAX: float = 100.0
# Minimum trust score for any routing decision
_MIN_TRUST_SCORE: float = 20.0  # COLLEAGUE


def _normalise_trust(trust_score: float) -> float:
    return min(trust_score / _TRUST_SCORE_MAX, 1.0)


class _PeerDomainStats:
    """Per-peer, per-domain statistics for routing decisions."""

    __slots__ = (
        "epistemic_depth",
        "declared",
        "tasks_assigned",
        "tasks_succeeded",
    )

    def __init__(self) -> None:
        self.epistemic_depth: float = 0.0
        self.declared: bool = False
        self.tasks_assigned: int = 0
        self.tasks_succeeded: int = 0

    @property
    def empirical_success_rate(self) -> float:
        if self.tasks_assigned == 0:
            return 0.5  # Neutral prior
        return self.tasks_succeeded / self.tasks_assigned


class WorkRouter:
    """
    Routes work to the best-specialised peer in the federation.

    Updated by Nexus and CapacityOffer events.  Queried by FederationService
    before broadcasting a task offer.
    """

    def __init__(self, instance_id: str) -> None:
        self._instance_id = instance_id
        self._event_bus: Any = None
        self._logger = logger.bind(component="work_router", instance_id=instance_id)

        # (instance_id, domain) → _PeerDomainStats
        self._stats: defaultdict[str, defaultdict[str, _PeerDomainStats]] = (
            defaultdict(lambda: defaultdict(_PeerDomainStats))
        )

        # routing decisions count
        self._tasks_routed: int = 0
        self._routing_hits: int = 0  # times a specialised peer was found

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.NEXUS_EPISTEMIC_VALUE,
                self._on_nexus_epistemic_value,
            )
            event_bus.subscribe(
                SynapseEventType.FEDERATION_CAPACITY_AVAILABLE,
                self._on_capacity_available,
            )
        except Exception as exc:
            self._logger.warning("subscription_failed", error=str(exc))

    # ─── Route a task ─────────────────────────────────────────────────

    def route(
        self,
        task: TaskDelegation,
        trusted_links: list[FederationLink],
        domain: str = "",
    ) -> FederationLink | None:
        """
        Select the best peer to receive a task based on specialisation.

        ``domain`` may be inferred from task payload if not provided.
        Falls back to trust-score ranking if no specialisation data.
        """
        if not domain:
            domain = self._infer_domain(task)

        eligible = [
            l for l in trusted_links
            if l.trust_level.value >= TrustLevel.COLLEAGUE.value
            and l.trust_score >= _MIN_TRUST_SCORE
        ]
        if not eligible:
            return None

        self._tasks_routed += 1

        best_link: FederationLink | None = None
        best_score: float = -1.0

        for link in eligible:
            score = self._score_peer(link, domain)
            if score > best_score:
                best_score = score
                best_link = link

        if best_link is not None:
            self._routing_hits += 1
            from systems.synapse.types import SynapseEventType as _SET
            self._emit(_SET.FEDERATION_WORK_ROUTED, {
                "bounty_id": task.payload.get("bounty_id", task.id),
                "routed_to_instance_id": best_link.remote_instance_id,
                "specialisation": domain,
                "routing_confidence": best_score,
                "timestamp": utc_now().isoformat(),
            })
            self._logger.info(
                "task_routed",
                task_id=task.id,
                peer=best_link.remote_instance_id,
                domain=domain,
                score=best_score,
            )

        return best_link

    def route_bounty(
        self,
        bounty_payload: dict[str, Any],
        trusted_links: list[FederationLink],
    ) -> FederationLink | None:
        """Convenience wrapper for bounty-type routing."""
        domain = (
            bounty_payload.get("language", "")
            or bounty_payload.get("domain", "")
            or bounty_payload.get("protocol", "")
        )
        # Build a synthetic task for routing scoring
        from primitives.federation import TaskStatus
        task = TaskDelegation(
            task_type=TaskType.SOLVE_BOUNTY,
            payload=bounty_payload,
            delegating_instance_id=self._instance_id,
            status=TaskStatus.OFFERED,
            offered_reward_usdc=Decimal(str(bounty_payload.get("reward_usdc", "0"))),
        )
        return self.route(task, trusted_links, domain=domain)

    # ─── Record outcomes (for empirical learning) ─────────────────────

    def record_outcome(
        self,
        instance_id: str,
        domain: str,
        success: bool,
    ) -> None:
        """Update empirical success rate for a peer/domain pair."""
        stats = self._stats[instance_id][domain]
        stats.tasks_assigned += 1
        if success:
            stats.tasks_succeeded += 1

    # ─── Scoring ─────────────────────────────────────────────────────

    def _score_peer(self, link: FederationLink, domain: str) -> float:
        """Compute composite routing score for a peer/domain pair."""
        stats = self._stats[link.remote_instance_id][domain]
        norm_trust = _normalise_trust(link.trust_score)

        epistemic = stats.epistemic_depth * 0.5
        declared = (1.0 if stats.declared else 0.0) * 0.3
        empirical = stats.empirical_success_rate * 0.2

        base_score = epistemic + declared + empirical
        return base_score * norm_trust

    def _infer_domain(self, task: TaskDelegation) -> str:
        """Try to extract a domain tag from task payload."""
        payload = task.payload
        for key in ("language", "domain", "protocol", "specialisation", "category"):
            val = payload.get(key, "")
            if val:
                return str(val).lower()
        # Fallback: task type as domain
        return task.task_type.value

    # ─── Event handlers ───────────────────────────────────────────────

    async def _on_nexus_epistemic_value(self, event: Any) -> None:
        """
        Update routing scores from Nexus triangulation data.

        ``NEXUS_EPISTEMIC_VALUE`` payload:
          instance_id (str), triangulation_score (float),
          fragment_count (int), ground_truth_count (int),
          domain (str, optional)
        """
        payload = getattr(event, "payload", {})
        instance_id = payload.get("instance_id", "")
        domain = payload.get("domain", "general")
        triangulation_score = float(payload.get("triangulation_score", 0.0))

        if not instance_id or instance_id == self._instance_id:
            return

        stats = self._stats[instance_id][domain]
        # Nexus triangulation score maps to epistemic depth 0-1
        stats.epistemic_depth = min(triangulation_score, 1.0)
        self._logger.debug(
            "epistemic_depth_updated",
            peer=instance_id,
            domain=domain,
            depth=stats.epistemic_depth,
        )

    async def _on_capacity_available(self, event: Any) -> None:
        """
        Update declared specialisations from a peer's CapacityOffer.
        """
        payload = getattr(event, "payload", {})
        instance_id = payload.get("instance_id", "")
        if not instance_id or instance_id == self._instance_id:
            return

        specialisations: list[str] = payload.get("specialisations", [])
        for spec in specialisations:
            domain = spec.lower()
            self._stats[instance_id][domain].declared = True
        self._logger.debug(
            "declared_specialisations_updated",
            peer=instance_id,
            specialisations=specialisations,
        )

    # ─── Introspection ────────────────────────────────────────────────

    def top_specialists(
        self,
        domain: str,
        n: int = 5,
        trusted_links: list[FederationLink] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return the top N instance IDs for a domain, with their routing score.

        If ``trusted_links`` is provided, filter to only active peers.
        """
        link_ids = (
            {l.remote_instance_id for l in trusted_links} if trusted_links else None
        )
        scores = []
        for instance_id, domain_map in self._stats.items():
            if link_ids is not None and instance_id not in link_ids:
                continue
            if domain not in domain_map:
                continue
            stats = domain_map[domain]
            # Simple score without trust (trust not available here)
            score = stats.epistemic_depth * 0.5 + (0.3 if stats.declared else 0.0) + stats.empirical_success_rate * 0.2
            scores.append((instance_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

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
        total_peers = len(self._stats)
        total_domain_entries = sum(len(v) for v in self._stats.values())
        return {
            "tasks_routed": self._tasks_routed,
            "routing_hits": self._routing_hits,
            "peers_profiled": total_peers,
            "domain_entries": total_domain_entries,
        }
