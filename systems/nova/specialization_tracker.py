"""
EcodiaOS - SpecializationTracker

Tracks this instance's specialization progress across all domains, building
DomainProfile objects from exploration outcomes and RE training examples.

Owned by Nova (the active-inference deliberation engine) because domain
specialization drives goal selection and policy preferences - both Nova concerns.

The tracker is passive: it only stores/queries. All Synapse emissions are
fire-and-forget so they don't block the hot path.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from primitives.evolution import DomainProfile
from primitives.re_training import RETrainingExample

if TYPE_CHECKING:
    from systems.synapse.service import SynapseService

logger = logging.getLogger(__name__)

_SUCCESS_RATE_THRESHOLD = 0.75    # primary domain when exceeded
_MIN_EXAMPLES_FOR_PRIMARY = 100   # must have seen this many domain examples
_OUTCOME_MAP = {"success": 1.0, "partial": 0.5, "failure": 0.0}


class SpecializationTracker:
    """
    Tracks organism's specialization progress across domains.

    Lifecycle
    ---------
    1. Nova creates + injects a SynapseService reference on boot.
    2. `initialize()` loads persisted DomainProfiles from Neo4j.
    3. Nova calls `on_exploration_outcome()` after each opportunity result.
    4. RETrainingExporter calls `on_training_example_emitted()` on every
       RE_TRAINING_EXAMPLE event it collects.
    5. When the primary domain changes, DOMAIN_SPECIALIZATION_DETECTED is emitted.
    """

    def __init__(self, instance_id: str) -> None:
        self._instance_id = instance_id
        self._profiles: dict[str, DomainProfile] = {}
        self._primary_domain: str = "generalist"
        self._synapse: SynapseService | None = None
        self._neo4j: Any | None = None   # injected via set_neo4j()

    # ── Dependency injection ────────────────────────────────────────────────

    def set_synapse(self, synapse: SynapseService) -> None:
        self._synapse = synapse

    def set_neo4j(self, driver: Any) -> None:
        self._neo4j = driver

    # ── Bootstrap ───────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load persisted DomainProfiles from Neo4j (best-effort)."""
        if self._neo4j is None:
            return
        try:
            async with self._neo4j.session() as session:
                result = await session.run(
                    """
                    MATCH (p:DomainProfile {instance_id: $iid})
                    RETURN p
                    """,
                    iid=self._instance_id,
                )
                async for record in result:
                    node = record["p"]
                    domain = node["domain"]
                    profile = DomainProfile(
                        domain=domain,
                        skill_areas=dict(node.get("skill_areas", {})),
                        examples_trained=int(node.get("examples_trained", 0)),
                        success_rate=float(node.get("success_rate", 0.0)),
                        revenue_generated=Decimal(
                            str(node.get("revenue_generated_usd", "0"))
                        ),
                        time_spent_hours=float(node.get("time_spent_hours", 0.0)),
                        confidence=float(node.get("confidence", 0.0)),
                        should_pass_to_children=bool(
                            node.get("should_pass_to_children", False)
                        ),
                        inheritance_weight=float(node.get("inheritance_weight", 0.0)),
                    )
                    if node.get("last_outcome"):
                        profile.last_outcome = datetime.fromisoformat(
                            node["last_outcome"]
                        )
                    self._profiles[domain] = profile
                    if node.get("primary", False):
                        self._primary_domain = domain
        except Exception:
            logger.warning(
                "SpecializationTracker: could not restore profiles from Neo4j",
                exc_info=True,
            )

    # ── Event handlers ──────────────────────────────────────────────────────

    async def on_exploration_outcome(
        self,
        domain: str,
        outcome: str,            # "success" | "partial" | "failure"
        revenue: Decimal | None = None,
        duration_hours: float = 0.0,
    ) -> None:
        """
        Called when an exploration in a domain completes.
        Updates the DomainProfile and checks whether to promote the domain.
        """
        if domain == "generalist":
            return

        profile = self._get_or_create(domain)
        score = _OUTCOME_MAP.get(outcome, 0.0)

        # Exponential moving average for success_rate (α=0.1, skewed toward
        # recent outcomes once the example count is small).
        n = profile.examples_trained
        alpha = max(0.1, 1.0 / max(n + 1, 1))
        profile.success_rate = (1 - alpha) * profile.success_rate + alpha * score
        profile.examples_trained += 1
        profile.revenue_generated += revenue or Decimal("0")
        profile.time_spent_hours += duration_hours
        profile.last_outcome = datetime.utcnow()
        profile.confidence = self._compute_confidence(profile)

        await self._maybe_promote_primary(domain, profile)
        await self._persist_profile(domain, profile)

    async def on_training_example_emitted(
        self, example: RETrainingExample
    ) -> None:
        """
        Called when any RE training example is created.
        Advances skill mastery for the example's skill_area.
        """
        if example.domain == "generalist":
            return

        domain = example.domain
        profile = self._get_or_create(domain)

        if example.skill_area:
            current = profile.skill_areas.get(example.skill_area, 0.0)
            profile.skill_areas[example.skill_area] = min(
                1.0, current + example.skill_improvement
            )

        # examples_trained is updated by on_exploration_outcome; here we just
        # track skill area growth without double-counting examples.
        profile.confidence = self._compute_confidence(profile)

        await self._persist_profile(domain, profile)

    # ── Accessors ───────────────────────────────────────────────────────────

    def get_primary_domain(self) -> str:
        return self._primary_domain

    def get_all_profiles(self) -> dict[str, DomainProfile]:
        return dict(self._profiles)

    def get_profile(self, domain: str) -> DomainProfile | None:
        return self._profiles.get(domain)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_or_create(self, domain: str) -> DomainProfile:
        if domain not in self._profiles:
            self._profiles[domain] = DomainProfile(domain=domain)
        return self._profiles[domain]

    @staticmethod
    def _compute_confidence(profile: DomainProfile) -> float:
        """
        Composite confidence = success_rate × log10(n+1) / 3.0, capped at 1.0.
        Grows slowly so a new domain doesn't falsely dominate.
        """
        return min(
            1.0,
            profile.success_rate * math.log10(profile.examples_trained + 1) / 3.0,
        )

    async def _maybe_promote_primary(
        self, domain: str, profile: DomainProfile
    ) -> None:
        if (
            domain == self._primary_domain
            or profile.success_rate <= _SUCCESS_RATE_THRESHOLD
            or profile.examples_trained < _MIN_EXAMPLES_FOR_PRIMARY
        ):
            return

        old_domain = self._primary_domain
        self._primary_domain = domain

        # Mark inheritance flag for the promoted domain
        profile.should_pass_to_children = True
        profile.inheritance_weight = profile.confidence

        logger.info(
            "SpecializationTracker: primary domain changed",
            extra={
                "old_domain": old_domain,
                "new_domain": domain,
                "success_rate": profile.success_rate,
                "examples_trained": profile.examples_trained,
            },
        )

        if self._synapse:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            self._synapse.event_bus.broadcast(
                SynapseEvent(
                    event_type=SynapseEventType.DOMAIN_SPECIALIZATION_DETECTED,
                    source_system="nova",
                    data={
                        "instance_id": self._instance_id,
                        "new_domain": domain,
                        "old_domain": old_domain,
                        "success_rate": profile.success_rate,
                        "examples_trained": profile.examples_trained,
                    },
                )
            )

    async def _persist_profile(
        self, domain: str, profile: DomainProfile
    ) -> None:
        """Write/merge the DomainProfile node to Neo4j and emit an update event."""
        is_primary = domain == self._primary_domain

        if self._neo4j:
            try:
                async with self._neo4j.session() as session:
                    await session.run(
                        """
                        MERGE (p:DomainProfile {instance_id: $iid, domain: $domain})
                        SET p.skill_areas       = $skill_areas,
                            p.examples_trained  = $examples_trained,
                            p.success_rate      = $success_rate,
                            p.revenue_generated_usd = $revenue_usd,
                            p.time_spent_hours  = $time_spent_hours,
                            p.last_outcome      = $last_outcome,
                            p.confidence        = $confidence,
                            p.should_pass_to_children = $should_pass,
                            p.inheritance_weight = $inheritance_weight,
                            p.primary           = $primary,
                            p.updated_at        = datetime()
                        """,
                        iid=self._instance_id,
                        domain=domain,
                        skill_areas={
                            k: float(v) for k, v in profile.skill_areas.items()
                        },
                        examples_trained=profile.examples_trained,
                        success_rate=float(profile.success_rate),
                        revenue_usd=str(profile.revenue_generated),
                        time_spent_hours=profile.time_spent_hours,
                        last_outcome=(
                            profile.last_outcome.isoformat()
                            if profile.last_outcome
                            else None
                        ),
                        confidence=float(profile.confidence),
                        should_pass=profile.should_pass_to_children,
                        inheritance_weight=float(profile.inheritance_weight),
                        primary=is_primary,
                    )
            except Exception:
                logger.warning(
                    "SpecializationTracker: Neo4j persist failed for domain=%s",
                    domain,
                    exc_info=True,
                )

        if self._synapse:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            self._synapse.event_bus.broadcast(
                SynapseEvent(
                    event_type=SynapseEventType.SPECIALIZATION_PROFILE_UPDATED,
                    source_system="nova",
                    data={
                        "instance_id": self._instance_id,
                        "domain": domain,
                        "success_rate": float(profile.success_rate),
                        "examples_trained": profile.examples_trained,
                        "skill_areas": {
                            k: float(v) for k, v in profile.skill_areas.items()
                        },
                        "confidence": float(profile.confidence),
                        "is_primary": is_primary,
                    },
                )
            )
