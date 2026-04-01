"""
EcodiaOS - Oikos: Community Reputation Tracker

Tracks the organism's growing reputation as a quantifiable economic asset
across GitHub and X (Twitter). Distinct from the EAS-based ReputationEngine
in reputation.py (which tracks cryptographic Proof-of-Cognitive-Work for
credit scoring). This module tracks social presence metrics that unlock
higher-value work and consulting rates.

Metrics:
    github_stars_received     - Stars accumulated on organism's repos/gists
    github_prs_merged         - Total merged PRs across all repos
    github_issues_resolved    - Issues closed by organism's PRs
    x_followers               - X follower count
    x_impressions_30d         - Monthly X impressions
    devto_followers           - Dev.to follower count
    devto_views_30d           - Dev.to views in rolling 30 days
    bounties_solved           - Lifetime bounty count (cross-indexed with Oikos)
    bounties_solved_value_usdc - Lifetime bounty value in USDC
    reputation_score          - Composite 0–100 (PR merges weighted highest)

reputation_multiplier = 1.0 + (reputation_score / 200)
Applied by Oikos to bounty acceptance confidence and consulting rate.

Neo4j persistence:
    (:ReputationSnapshot) nodes written hourly, linked to instance.
    reputation_score included in ORGANISM_TELEMETRY.

Event subscriptions:
    BOUNTY_PR_MERGED        → increment github_prs_merged
    BOUNTY_PAID             → increment bounties_solved + value
    COMMUNITY_ENGAGEMENT_COMPLETED → note interaction (star/follow/react)
    CONTENT_ENGAGEMENT_REPORT → update x_followers / x_impressions

Events emitted:
    REPUTATION_SNAPSHOT     - hourly broadcast
    REPUTATION_DAMAGED      - score drop ≥5 pts (Nova recovery + Thread CRISIS)
    REPUTATION_MILESTONE    - score crosses 25/50/70/90 (Thread GROWTH)
    SOCIAL_GRAPH_UPDATED    - Neo4j relationship written
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.reputation_tracker")

# ─── Constants ─────────────────────────────────────────────────────────────────

_REDIS_KEY = "oikos:community_reputation"
_SNAPSHOT_INTERVAL_S = 3600.0  # 1 hour

_MILESTONES = [25, 50, 70, 90]

# Score weights (must sum to 1.0)
_W_PR_MERGES = 0.35
_W_STARS = 0.20
_W_BOUNTIES = 0.20
_W_FOLLOWERS = 0.10
_W_IMPRESSIONS = 0.08
_W_ISSUES = 0.07


# ─── Data Models ───────────────────────────────────────────────────────────────


class ReputationMetrics(EOSBaseModel):
    """Live reputation metrics - the organism's social presence scorecard."""

    # GitHub
    github_stars_received: int = 0
    github_prs_merged: int = 0
    github_issues_resolved: int = 0

    # X (Twitter)
    x_followers: int = 0
    x_impressions_30d: int = 0

    # Dev.to
    devto_followers: int = 0
    devto_views_30d: int = 0

    # Bounties (cross-indexed with Oikos RevenueStream.BOUNTY)
    bounties_solved: int = 0
    bounties_solved_value_usdc: float = 0.0

    # Composite
    reputation_score: float = 0.0

    # Metadata
    last_snapshot_at: datetime = Field(default_factory=utc_now)
    milestones_crossed: list[int] = Field(default_factory=list)


class ReputationSnapshot(EOSBaseModel):
    """Point-in-time snapshot written to Neo4j as (:ReputationSnapshot)."""

    snapshot_id: str = Field(default_factory=new_id)
    recorded_at: datetime = Field(default_factory=utc_now)
    metrics: ReputationMetrics = Field(default_factory=ReputationMetrics)
    reputation_multiplier: float = 1.0


# ─── Reputation Tracker ────────────────────────────────────────────────────────


class ReputationTracker:
    """
    Tracks community reputation metrics and computes the composite
    reputation_score (0–100) that feeds Oikos economic decisions.

    PR merges are weighted highest because they represent the strongest
    external validation signal: a human maintainer reviewed, trusted,
    and accepted the organism's code contribution.

    Emits hourly REPUTATION_SNAPSHOT to the bus and writes
    (:ReputationSnapshot) nodes to Neo4j for longitudinal analysis.
    """

    def __init__(
        self,
        redis: RedisClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._redis = redis
        self._event_bus = event_bus
        self._neo4j: Any = None
        self._instance_id = "unknown"
        self._metrics = ReputationMetrics()
        self._log = logger.bind(component="reputation_tracker")
        self._snapshot_task: asyncio.Task[None] | None = None

    # ── Dependency injection ──────────────────────────────────────────

    def set_neo4j(self, neo4j: Any) -> None:
        self._neo4j = neo4j

    def set_instance_id(self, instance_id: str) -> None:
        self._instance_id = instance_id

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load state from Redis, wire event subscriptions, start snapshot loop."""
        await self._load_state()
        if self._event_bus is not None:
            self._subscribe_events()
        self._snapshot_task = asyncio.create_task(
            self._hourly_snapshot_loop(), name="reputation_tracker_snapshot"
        )
        self._log.info(
            "reputation_tracker_initialized",
            score=self._metrics.reputation_score,
        )

    def _subscribe_events(self) -> None:
        from systems.synapse.types import SynapseEventType

        assert self._event_bus is not None
        self._event_bus.subscribe(SynapseEventType.BOUNTY_PR_MERGED, self._on_pr_merged)
        self._event_bus.subscribe(SynapseEventType.BOUNTY_PAID, self._on_bounty_paid)
        self._event_bus.subscribe(
            SynapseEventType.COMMUNITY_ENGAGEMENT_COMPLETED, self._on_engagement_completed
        )
        self._event_bus.subscribe(
            SynapseEventType.CONTENT_ENGAGEMENT_REPORT, self._on_engagement_report
        )

    # ── Event Handlers ────────────────────────────────────────────────

    async def _on_pr_merged(self, event: Any) -> None:
        data: dict[str, Any] = getattr(event, "data", {})
        prev_score = self._metrics.reputation_score
        self._metrics.github_prs_merged += 1
        issues_closed = int(data.get("issues_resolved", 0))
        self._metrics.github_issues_resolved += issues_closed

        # Write CONTRIBUTED_TO social graph edge
        repo = data.get("repository", "")
        pr_url = data.get("pr_url", "")
        if repo:
            await self._write_contributed_to(repo=repo, pr_url=pr_url, pr_number=data.get("pr_number", 0))

        new_score = self._compute_score()
        await self._update_score(prev_score, new_score, cause="pr_merged")
        await self._persist_state()

        self._log.info(
            "reputation_pr_merged",
            prs_merged=self._metrics.github_prs_merged,
            score=new_score,
        )

    async def _on_bounty_paid(self, event: Any) -> None:
        data: dict[str, Any] = getattr(event, "data", {})
        prev_score = self._metrics.reputation_score
        self._metrics.bounties_solved += 1
        value = float(data.get("reward_amount_usd", 0.0))
        self._metrics.bounties_solved_value_usdc += value

        new_score = self._compute_score()
        await self._update_score(prev_score, new_score, cause="bounty_paid")
        await self._persist_state()

    async def _on_engagement_completed(self, event: Any) -> None:
        data: dict[str, Any] = getattr(event, "data", {})
        action = data.get("action", "")

        # When a star is confirmed, increment if the repo starred us back
        # (We track stars received via polling, not on each star-action.
        # Star actions build the outbound social graph only.)
        if action == "star_repo":
            repo = data.get("target_url", "").split("github.com/")[-1].rstrip("/")
            if repo:
                await self._write_social_graph(
                    relationship_type="INTERESTED_IN",
                    source_id=f"eos:{self._instance_id}",
                    target_id=f"github:{repo}",
                    platform="github",
                    url=data.get("target_url", ""),
                )

    async def _on_engagement_report(self, event: Any) -> None:
        """Handle CONTENT_ENGAGEMENT_REPORT to update follower/impression metrics."""
        data: dict[str, Any] = getattr(event, "data", {})
        platform = data.get("platform", "")

        if platform in ("twitter", "x"):
            followers = int(data.get("followers", 0))
            if followers > self._metrics.x_followers:
                self._metrics.x_followers = followers
            impressions = int(data.get("impressions_per_month", data.get("reach", 0)))
            if impressions > 0:
                self._metrics.x_impressions_30d = impressions

        elif platform == "devto":
            followers = int(data.get("followers", 0))
            if followers > self._metrics.devto_followers:
                self._metrics.devto_followers = followers
            views = int(data.get("views", 0))
            if views > 0:
                self._metrics.devto_views_30d = views

        prev_score = self._metrics.reputation_score
        new_score = self._compute_score()
        await self._update_score(prev_score, new_score, cause="engagement_report")
        await self._persist_state()

    # ── Public API ────────────────────────────────────────────────────

    async def record_stars_received(self, count: int) -> None:
        """Called by a periodic GitHub polling job to update star count."""
        if count <= self._metrics.github_stars_received:
            return
        prev_score = self._metrics.reputation_score
        self._metrics.github_stars_received = count
        new_score = self._compute_score()
        await self._update_score(prev_score, new_score, cause="stars_received")
        await self._persist_state()

    async def record_developer_recognition(
        self, developer_login: str, repo: str, comment_url: str
    ) -> None:
        """
        A developer has commented positively on EOS's PR/contribution.
        Writes (developer)-[:RECOGNISES]->(organism) in Neo4j.
        """
        await self._write_social_graph(
            relationship_type="RECOGNISES",
            source_id=f"github:{developer_login}",
            target_id=f"eos:{self._instance_id}",
            platform="github",
            url=comment_url,
        )

    def get_metrics(self) -> ReputationMetrics:
        return self._metrics

    def get_reputation_multiplier(self) -> float:
        """
        Economic multiplier applied to bounty confidence and consulting rates.
        reputation_multiplier = 1.0 + (reputation_score / 200)
        Range: 1.0 (score=0) → 1.5 (score=100)
        """
        return 1.0 + (self._metrics.reputation_score / 200.0)

    def snapshot(self) -> dict[str, Any]:
        m = self._metrics
        return {
            "github_stars_received": m.github_stars_received,
            "github_prs_merged": m.github_prs_merged,
            "github_issues_resolved": m.github_issues_resolved,
            "x_followers": m.x_followers,
            "x_impressions_30d": m.x_impressions_30d,
            "devto_followers": m.devto_followers,
            "devto_views_30d": m.devto_views_30d,
            "bounties_solved": m.bounties_solved,
            "bounties_solved_value_usdc": m.bounties_solved_value_usdc,
            "reputation_score": m.reputation_score,
            "reputation_multiplier": self.get_reputation_multiplier(),
        }

    # ── Score computation ─────────────────────────────────────────────

    def _compute_score(self) -> float:
        """
        Weighted composite score 0–100.

        PR merges are weighted highest (35%) because human maintainer
        acceptance is the strongest external validation signal.

        Component normalisation targets:
          PR merges:    20 merges → 100 pts
          Stars:        100 stars → 100 pts
          Bounties:     50 bounties → 100 pts
          X followers:  5,000 followers → 100 pts
          Impressions:  500k/month → 100 pts
          Issues:       50 issues → 100 pts
        """
        m = self._metrics

        pr_component = min(1.0, m.github_prs_merged / 20.0)
        star_component = min(1.0, m.github_stars_received / 100.0)
        bounty_component = min(1.0, m.bounties_solved / 50.0)
        follower_component = min(1.0, m.x_followers / 5_000.0)
        impression_component = min(1.0, m.x_impressions_30d / 500_000.0)
        issue_component = min(1.0, m.github_issues_resolved / 50.0)

        raw = (
            _W_PR_MERGES * pr_component
            + _W_STARS * star_component
            + _W_BOUNTIES * bounty_component
            + _W_FOLLOWERS * follower_component
            + _W_IMPRESSIONS * impression_component
            + _W_ISSUES * issue_component
        )

        return round(raw * 100.0, 2)

    # ── Score update + event emission ─────────────────────────────────

    async def _update_score(self, prev: float, new: float, cause: str) -> None:
        self._metrics.reputation_score = new
        self._metrics.last_snapshot_at = utc_now()

        delta = new - prev

        # Reputation damage (drop ≥5 points)
        if delta <= -5.0:
            await self._emit_damaged(prev, new, delta, cause)

        # Milestone crossings
        for milestone in _MILESTONES:
            if prev < milestone <= new and milestone not in self._metrics.milestones_crossed:
                self._metrics.milestones_crossed.append(milestone)
                await self._emit_milestone(new, milestone)

    async def _emit_damaged(self, prev: float, new: float, delta: float, cause: str) -> None:
        if self._event_bus is None:
            return
        from systems.synapse.types import SynapseEventType

        recovery = _recovery_recommendation(cause)
        try:
            await self._event_bus.emit(
                SynapseEventType.REPUTATION_DAMAGED,
                {
                    "reputation_score": new,
                    "prev_score": prev,
                    "delta": delta,
                    "cause": cause,
                    "recommended_recovery": recovery,
                },
            )
            self._log.warning(
                "reputation_damaged",
                score=new,
                prev=prev,
                delta=delta,
                cause=cause,
            )
        except Exception as exc:
            self._log.error("emit_reputation_damaged_failed", error=str(exc))

    async def _emit_milestone(self, score: float, milestone: int) -> None:
        if self._event_bus is None:
            return
        from systems.synapse.types import SynapseEventType

        tier = _tier_for_score(score)
        multiplier = self.get_reputation_multiplier()
        try:
            await self._event_bus.emit(
                SynapseEventType.REPUTATION_MILESTONE,
                {
                    "reputation_score": score,
                    "milestone": milestone,
                    "tier": tier,
                    "consulting_rate_multiplier": _consulting_multiplier(score),
                },
            )
            self._log.info(
                "reputation_milestone",
                score=score,
                milestone=milestone,
                tier=tier,
                multiplier=multiplier,
            )
        except Exception as exc:
            self._log.error("emit_reputation_milestone_failed", error=str(exc))

    # ── Hourly snapshot loop ──────────────────────────────────────────

    async def _hourly_snapshot_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_SNAPSHOT_INTERVAL_S)
                await self._emit_snapshot()
                await self._write_neo4j_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error("hourly_snapshot_error", error=str(exc))

    async def _emit_snapshot(self) -> None:
        if self._event_bus is None:
            return
        from systems.synapse.types import SynapseEventType

        try:
            await self._event_bus.emit(
                SynapseEventType.REPUTATION_SNAPSHOT,
                self.snapshot(),
            )
        except Exception as exc:
            self._log.error("emit_reputation_snapshot_failed", error=str(exc))

    # ── Neo4j persistence ─────────────────────────────────────────────

    async def _write_neo4j_snapshot(self) -> None:
        if self._neo4j is None:
            return
        snap = ReputationSnapshot(metrics=self._metrics, reputation_multiplier=self.get_reputation_multiplier())
        cypher = """
        MATCH (i:Instance {instance_id: $instance_id})
        CREATE (s:ReputationSnapshot {
            snapshot_id: $snapshot_id,
            recorded_at: datetime($recorded_at),
            reputation_score: $reputation_score,
            github_prs_merged: $prs_merged,
            github_stars_received: $stars,
            x_followers: $x_followers,
            bounties_solved: $bounties_solved,
            reputation_multiplier: $multiplier
        })
        MERGE (i)-[:HAS_REPUTATION_SNAPSHOT]->(s)
        """
        try:
            await self._neo4j.execute_query(
                cypher,
                {
                    "instance_id": self._instance_id,
                    "snapshot_id": snap.snapshot_id,
                    "recorded_at": snap.recorded_at.isoformat(),
                    "reputation_score": snap.metrics.reputation_score,
                    "prs_merged": snap.metrics.github_prs_merged,
                    "stars": snap.metrics.github_stars_received,
                    "x_followers": snap.metrics.x_followers,
                    "bounties_solved": snap.metrics.bounties_solved,
                    "multiplier": snap.reputation_multiplier,
                },
            )
        except Exception as exc:
            self._log.error("neo4j_snapshot_write_failed", error=str(exc))

    async def _write_contributed_to(self, repo: str, pr_url: str, pr_number: int) -> None:
        """Write (organism)-[:CONTRIBUTED_TO]->(repo) in Neo4j when PR merges."""
        await self._write_social_graph(
            relationship_type="CONTRIBUTED_TO",
            source_id=f"eos:{self._instance_id}",
            target_id=f"github:{repo}",
            platform="github",
            url=pr_url,
            extra={"pr_number": pr_number},
        )

    async def _write_social_graph(
        self,
        relationship_type: str,
        source_id: str,
        target_id: str,
        platform: str,
        url: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a social graph edge to Neo4j and emit SOCIAL_GRAPH_UPDATED."""
        if self._neo4j is not None:
            cypher = """
            MERGE (src:SocialNode {node_id: $source_id})
            MERGE (tgt:SocialNode {node_id: $target_id})
            MERGE (src)-[r:SOCIAL_EDGE {type: $rel_type}]->(tgt)
            SET r.platform = $platform,
                r.url = $url,
                r.last_seen = datetime($ts)
            """
            params: dict[str, Any] = {
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": relationship_type,
                "platform": platform,
                "url": url,
                "ts": utc_now().isoformat(),
            }
            if extra:
                params.update(extra)
            try:
                await self._neo4j.execute_query(cypher, params)
            except Exception as exc:
                self._log.error("social_graph_neo4j_failed", error=str(exc))

        # Emit bus event regardless of Neo4j success
        if self._event_bus is not None:
            from systems.synapse.types import SynapseEventType
            try:
                await self._event_bus.emit(
                    SynapseEventType.SOCIAL_GRAPH_UPDATED,
                    {
                        "relationship_type": relationship_type,
                        "source_id": source_id,
                        "target_id": target_id,
                        "platform": platform,
                        "url": url,
                    },
                )
            except Exception as exc:
                self._log.warning("emit_social_graph_updated_failed", error=str(exc))

    # ── Redis persistence ─────────────────────────────────────────────

    async def _persist_state(self) -> None:
        if self._redis is None:
            return
        try:
            blob = json.loads(self._metrics.model_dump_json())
            await self._redis.set_json(_REDIS_KEY, blob)
        except Exception as exc:
            self._log.error("persist_reputation_failed", error=str(exc))

    async def _load_state(self) -> None:
        if self._redis is None:
            return
        try:
            blob = await self._redis.get_json(_REDIS_KEY)
            if blob is not None:
                self._metrics = ReputationMetrics.model_validate(blob)
                self._log.info(
                    "reputation_state_loaded",
                    score=self._metrics.reputation_score,
                    prs_merged=self._metrics.github_prs_merged,
                )
        except Exception as exc:
            self._log.error("load_reputation_failed", error=str(exc))


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _tier_for_score(score: float) -> str:
    if score >= 90:
        return "sovereign"
    if score >= 70:
        return "trusted"
    if score >= 50:
        return "established"
    if score >= 25:
        return "reliable"
    return "newcomer"


def _consulting_multiplier(score: float) -> float:
    """Consulting rate multiplier: score > 70 → 1.25×."""
    if score >= 70:
        return 1.25
    return 1.0


def _recovery_recommendation(cause: str) -> str:
    recommendations = {
        "pr_rejected": (
            "Produce a high-quality contribution to a high-profile project. "
            "Review why the PR was rejected and address feedback before resubmitting."
        ),
        "negative_engagement": (
            "Generate substantive technical content - a deep-dive blog post or "
            "a useful open-source tool - to rebuild community goodwill."
        ),
        "no_activity": (
            "Increase community engagement: answer GitHub Discussion questions, "
            "contribute to active open-source projects, or publish technical content."
        ),
    }
    return recommendations.get(
        cause,
        "Produce high-quality content or contribute to a high-profile project to rebuild reputation.",
    )
