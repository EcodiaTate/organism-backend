"""
EcodiaOS - Oikos Bounty Hunter (Phase 16b: The Freelancer - Active Foraging)

The organism's autonomous foraging behaviour. This module scans external
platforms for paid work, evaluates whether the caloric return exceeds the
metabolic cost, and queues accepted bounties for execution via Axon.

The BountyHunter is the bridge between raw opportunity discovery and the
economic decision engine. It answers one question per candidate:
"Is this bounty worth the metabolic cost of solving it?"

Design principles:
  - All monetary values are Decimal (no float rounding on money)
  - Pure evaluation logic - evaluate() has no side effects
  - Platform scanners are isolated and individually failable
  - Dedup via Redis SET of known issue_urls prevents re-evaluation
  - Emits BOUNTY_SOLUTION_PENDING on Synapse when a bounty is accepted

Thread-safety: NOT thread-safe. Designed for single-threaded asyncio event loop.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now
from systems.oikos.models import ActiveBounty, BountyStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from clients.redis import RedisClient
    from config import ExternalPlatformsConfig, OikosConfig
    from systems.oikos.models import EconomicState
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.bounty_hunter")

# ─── Redis Key Constants ────────────────────────────────────────────

REDIS_KEY_KNOWN_URLS = "oikos:bounty:known_urls"
REDIS_KEY_CANDIDATES = "oikos:bounty:candidates"

# ─── Cost Heuristic Constants ──────────────────────────────────────

# Base cost per 1k tokens (blended input/output across providers)
_COST_PER_1K_TOKENS = Decimal("0.015")

# Maximum estimated tokens for a full-difficulty bounty
_MAX_ESTIMATED_TOKENS = 50_000

# Candidates list TTL in Redis (48 hours)
_CANDIDATES_TTL_SECONDS = 172_800


# ─── Policy Constants ──────────────────────────────────────────────


class BountyPolicy:
    """
    Hard economic constraints for bounty acceptance. The organism must
    meet ALL thresholds before a bounty candidate is approved.

    These are conservative defaults - survival economics demand that
    every bounty is decisively profitable before committing resources.
    """

    # Reward must be at least 2x the estimated cost (100% ROI minimum)
    MIN_ROI_THRESHOLD: Decimal = Decimal("2.0")

    # Maximum concurrent bounties in AVAILABLE or IN_PROGRESS status
    MAX_CONCURRENT_BOUNTIES: int = 3

    # Minimum reward to even consider a bounty
    MIN_REWARD_USD: Decimal = Decimal("5.00")

    # Estimated cost must be < 40% of reward
    MAX_ESTIMATED_COST_PCT: Decimal = Decimal("0.40")

    # Platforms the organism scans for bounties
    PLATFORMS: list[str] = [
        "github_bounties",
        "algora",
        "ecodia_portal",
        "dework",
    ]


# ─── Data Models ───────────────────────────────────────────────────


class BountyCandidate(EOSBaseModel):
    """
    A discovered bounty that has not yet been evaluated or accepted.

    Candidates flow through: discovery -> evaluation -> acceptance/rejection.
    The candidate_id is stable across the pipeline so evaluations and
    active bounties can be traced back to the original discovery.
    """

    candidate_id: str = Field(default_factory=new_id)
    platform: str = ""
    issue_url: str = ""
    title: str = ""
    description: str = ""

    # Economics
    reward_usd: Decimal = Decimal("0")
    estimated_cost_usd: Decimal = Decimal("0")
    deadline: str | None = None  # ISO 8601 or None if no deadline

    # Scoring dimensions
    difficulty_score: Decimal = Decimal("0")  # 0.0 - 1.0
    capability_match: Decimal = Decimal("0")  # 0.0 - 1.0
    roi_score: Decimal = Decimal("0")

    # Metadata
    tags: list[str] = Field(default_factory=list)
    repo_owner: str = ""
    repo_name: str = ""
    discovered_at: str = Field(default_factory=lambda: utc_now().isoformat())

    # Lifecycle
    evaluated: bool = False
    accepted: bool = False
    rejection_reason: str = ""


class BountyEvaluation(EOSBaseModel):
    """
    Result of scoring a BountyCandidate against BountyPolicy.

    This is a pure data record - the evaluate() method produces it,
    and accept_bounty() consumes it. The separation ensures the
    approval decision is auditable and reproducible.
    """

    candidate_id: str = ""
    roi_score: Decimal = Decimal("0")
    estimated_cost_usd: Decimal = Decimal("0")

    # Dimensional scores
    capability_match: Decimal = Decimal("0")  # 0.0 - 1.0
    ethical_alignment: Decimal = Decimal("1.0")  # 0.0 - 1.0 (default: aligned)

    # Feasibility checks
    time_feasible: bool = True
    within_capacity: bool = True

    # Decision
    approved: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    confidence: Decimal = Decimal("0.5")


# ─── Bounty Hunter ─────────────────────────────────────────────────


class BountyHunter:
    """
    The organism's foraging engine. Scans platforms, evaluates candidates
    against strict economic policy, and queues accepted bounties for
    execution via the Axon pipeline.

    Usage::

        hunter = BountyHunter(config=oikos_config, redis=redis_client)
        hunter.attach(event_bus)
        evaluations = await hunter.run_foraging_cycle(economic_state)
    """

    def __init__(
        self,
        config: OikosConfig | None = None,
        redis: RedisClient | None = None,
        platforms_config: ExternalPlatformsConfig | None = None,
    ) -> None:
        self._config = config
        self._redis = redis
        self._platforms_config: ExternalPlatformsConfig | None = platforms_config
        self._event_bus: EventBus | None = None
        self._log = logger.bind(component="bounty_hunter")

    def set_platforms_config(self, config: ExternalPlatformsConfig) -> None:
        """Inject external platform credentials after construction."""
        self._platforms_config = config
        self._log.info("platforms_config_wired")

    def attach(self, event_bus: EventBus) -> None:
        """
        Wire up to the Synapse event bus.

        Call after both BountyHunter and SynapseService are initialised.
        Subscribes to relevant events and stores bus reference for emission.
        """
        from systems.synapse.types import SynapseEventType

        self._event_bus = event_bus

        # Subscribe to bounty lifecycle events for tracking
        event_bus.subscribe(
            SynapseEventType.BOUNTY_PAID,
            self._on_bounty_paid,
        )
        self._log.info("bounty_hunter_attached")

    # ─── Platform Scanning ─────────────────────────────────────────

    async def scan_platforms(self) -> list[BountyCandidate]:
        """
        Scan all configured platforms for new bounty candidates.

        Deduplicates against known URLs stored in Redis. Only returns
        newly discovered candidates that have not been seen before.
        """
        all_candidates: list[BountyCandidate] = []

        # Dispatch per platform
        platform_scanners: dict[str, Callable[..., Any]] = {
            "github_bounties": self._scan_github_bounties,
            "algora": self._scan_algora,
            "dework": self._scan_dework,
            "ecodia_portal": self._scan_ecodia_portal,
        }

        for platform in BountyPolicy.PLATFORMS:
            scanner = platform_scanners.get(platform)
            if scanner is None:
                self._log.warning(
                    "unknown_platform_skipped",
                    platform=platform,
                )
                continue

            try:
                candidates = await scanner()
                all_candidates.extend(candidates)
            except Exception:
                self._log.exception(
                    "platform_scan_failed",
                    platform=platform,
                )

        # Deduplicate against known URLs in Redis
        new_candidates = await self._deduplicate(all_candidates)

        self._log.info(
            "platform_scan_complete",
            total_discovered=len(all_candidates),
            new_candidates=len(new_candidates),
            platforms_scanned=len(BountyPolicy.PLATFORMS),
        )

        return new_candidates

    async def _scan_github_bounties(self) -> list[BountyCandidate]:
        """
        Scan GitHub for bounty-labelled issues via GitHubClient.

        Requires ExternalPlatformsConfig with a GitHub token for higher
        rate limits. Falls back gracefully to anonymous API if no token.
        """
        if self._platforms_config is None:
            self._log.warning("github_scan_skipped", reason="no platforms config")
            return []

        from clients.github import GitHubClient

        min_reward = float(BountyPolicy.MIN_REWARD_USD)
        candidates: list[BountyCandidate] = []

        async with GitHubClient(self._platforms_config) as gh:
            result = await gh.search_open_bounties(
                min_reward=min_reward,
                max_pages=3,
            )

        for issue in result.items:
            reward = Decimal(str(issue.reward_usd)) if issue.reward_usd else Decimal("0")
            repo_parts = issue.repo.split("/", 1)
            candidates.append(BountyCandidate(
                platform="github_bounties",
                issue_url=issue.url,
                title=issue.title,
                description=(issue.body or "")[:500],
                reward_usd=reward,
                difficulty_score=self._infer_difficulty(issue.labels),
                capability_match=self._infer_capability(issue.title, issue.labels),
                tags=issue.labels,
                repo_owner=repo_parts[0] if len(repo_parts) == 2 else "",
                repo_name=repo_parts[1] if len(repo_parts) == 2 else issue.repo,
            ))

        self._log.info(
            "github_scan_complete",
            candidates=len(candidates),
            total_api_count=result.total_count,
        )
        return candidates

    async def _scan_algora(self) -> list[BountyCandidate]:
        """
        Scan Algora bounty platform for active bounties.

        Algora bounties surface as GitHub issues labelled with the diamond
        emoji. Delegates to AlgoraClient which wraps the GitHub search API.
        """
        if self._platforms_config is None:
            self._log.warning("algora_scan_skipped", reason="no platforms config")
            return []

        from clients.algora import AlgoraClient

        min_reward = float(BountyPolicy.MIN_REWARD_USD)
        candidates: list[BountyCandidate] = []

        async with AlgoraClient(self._platforms_config) as algora:
            result = await algora.fetch_active_bounties(
                min_reward=min_reward,
                max_pages=3,
            )

        for issue in result.items:
            reward = Decimal(str(issue.reward_usd)) if issue.reward_usd else Decimal("0")
            repo_parts = issue.repo.split("/", 1)
            candidates.append(BountyCandidate(
                platform="algora",
                issue_url=issue.url,
                title=issue.title,
                description=(issue.body or "")[:500],
                reward_usd=reward,
                difficulty_score=self._infer_difficulty(issue.labels),
                capability_match=self._infer_capability(issue.title, issue.labels),
                tags=issue.labels,
                repo_owner=repo_parts[0] if len(repo_parts) == 2 else "",
                repo_name=repo_parts[1] if len(repo_parts) == 2 else issue.repo,
            ))

        self._log.info(
            "algora_scan_complete",
            candidates=len(candidates),
            total_api_count=result.total_count,
        )
        return candidates

    async def _scan_dework(self) -> list[BountyCandidate]:
        """
        Scan Dework for available bounties.

        Placeholder - Dework API integration pending. Returns empty list.
        """
        self._log.info(
            "dework_scan_placeholder",
            msg="Would scan Dework for available bounties",
        )
        return []

    async def _scan_ecodia_portal(self) -> list[BountyCandidate]:
        """
        Scan the Ecodia internal bounty portal.

        Placeholder - the portal API does not exist yet. Returns empty list.
        """
        self._log.info(
            "ecodia_portal_scan_placeholder",
            msg="Would scan Ecodia portal for internal bounties",
        )
        return []

    # ─── Evaluation ────────────────────────────────────────────────

    def evaluate(
        self,
        candidate: BountyCandidate,
        economic_state: EconomicState,
    ) -> BountyEvaluation:
        """
        Pure evaluation of a bounty candidate against BountyPolicy.

        No side effects. Returns a BountyEvaluation with approval status
        and detailed rejection reasons if the candidate fails any check.

        Checks (in order):
          a) reward >= MIN_REWARD_USD
          b) estimated_cost / reward <= MAX_ESTIMATED_COST_PCT
          c) ROI = (reward - cost) / cost >= MIN_ROI_THRESHOLD
          d) active bounties < MAX_CONCURRENT_BOUNTIES
          e) deadline is feasible (> 24h from now, or no deadline)
        """
        rejection_reasons: list[str] = []

        # Estimate cost if not already set
        estimated_cost = candidate.estimated_cost_usd
        if estimated_cost <= Decimal("0"):
            estimated_cost = self._estimate_cost(candidate)

        reward = candidate.reward_usd

        # ── Check a: Minimum reward ──
        if reward < BountyPolicy.MIN_REWARD_USD:
            rejection_reasons.append(
                f"Reward ${reward} below minimum ${BountyPolicy.MIN_REWARD_USD}"
            )

        # ── Check b: Cost percentage ──
        cost_pct = Decimal("0")
        if reward > Decimal("0"):
            cost_pct = (estimated_cost / reward).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        if cost_pct > BountyPolicy.MAX_ESTIMATED_COST_PCT:
            rejection_reasons.append(
                f"Cost ratio {cost_pct} exceeds max {BountyPolicy.MAX_ESTIMATED_COST_PCT}"
            )

        # ── Check c: ROI threshold ──
        roi_score = Decimal("0")
        if estimated_cost > Decimal("0"):
            roi_score = ((reward - estimated_cost) / estimated_cost).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        elif reward > Decimal("0"):
            # Zero cost => infinite ROI, treat as passing
            roi_score = Decimal("999")

        if roi_score < BountyPolicy.MIN_ROI_THRESHOLD:
            rejection_reasons.append(
                f"ROI {roi_score} below threshold {BountyPolicy.MIN_ROI_THRESHOLD}"
            )

        # ── Check d: Capacity ──
        active_count = sum(
            1
            for b in economic_state.active_bounties
            if b.status in (BountyStatus.AVAILABLE, BountyStatus.IN_PROGRESS)
        )
        within_capacity = active_count < BountyPolicy.MAX_CONCURRENT_BOUNTIES
        if not within_capacity:
            max_b = BountyPolicy.MAX_CONCURRENT_BOUNTIES
            rejection_reasons.append(
                f"At capacity: {active_count}/{max_b} active bounties"
            )

        # ── Check e: Deadline feasibility ──
        time_feasible = True
        if candidate.deadline is not None:
            try:
                deadline_dt = datetime.fromisoformat(candidate.deadline)
                if deadline_dt.tzinfo is None:
                    deadline_dt = deadline_dt.replace(tzinfo=UTC)
                min_deadline = utc_now() + timedelta(hours=24)
                if deadline_dt < min_deadline:
                    time_feasible = False
                    rejection_reasons.append(
                        f"Deadline {candidate.deadline} is less than 24h away"
                    )
            except (ValueError, TypeError):
                # Unparseable deadline - treat as no deadline (feasible)
                self._log.warning(
                    "unparseable_deadline",
                    candidate_id=candidate.candidate_id,
                    deadline=candidate.deadline,
                )

        # ── Confidence score ──
        # Higher confidence when capability_match is high and cost estimate is tight
        confidence = (
            Decimal("0.3")
            + candidate.capability_match * Decimal("0.4")
            + (Decimal("1") - candidate.difficulty_score) * Decimal("0.3")
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        approved = len(rejection_reasons) == 0

        return BountyEvaluation(
            candidate_id=candidate.candidate_id,
            roi_score=roi_score,
            estimated_cost_usd=estimated_cost,
            capability_match=candidate.capability_match,
            ethical_alignment=Decimal("1.0"),  # Default - Equor gates downstream
            time_feasible=time_feasible,
            within_capacity=within_capacity,
            approved=approved,
            rejection_reasons=rejection_reasons,
            confidence=confidence,
        )

    # ─── Acceptance ────────────────────────────────────────────────

    async def accept_bounty(
        self,
        candidate: BountyCandidate,
        evaluation: BountyEvaluation,
    ) -> ActiveBounty:
        """
        Create an ActiveBounty from an approved candidate and emit
        BOUNTY_SOLUTION_PENDING on Synapse so the Axon pipeline can
        begin solving it.

        Also stores the issue_url in Redis for dedup persistence.
        """
        from systems.synapse.types import SynapseEvent, SynapseEventType

        active = ActiveBounty(
            bounty_id=new_id(),
            platform=candidate.platform,
            reward_usd=candidate.reward_usd,
            estimated_cost_usd=evaluation.estimated_cost_usd,
            deadline=self._parse_deadline(candidate.deadline),
            status=BountyStatus.AVAILABLE,
            issue_url=candidate.issue_url,
        )

        # Persist URL for dedup
        if self._redis is not None:
            try:
                known: list[str] = await self._redis.get_json(REDIS_KEY_KNOWN_URLS) or []
                if candidate.issue_url not in known:
                    known.append(candidate.issue_url)
                    await self._redis.set_json(REDIS_KEY_KNOWN_URLS, known)
            except Exception:
                self._log.exception(
                    "redis_dedup_store_failed",
                    issue_url=candidate.issue_url,
                )

        # Emit solution-pending event
        if self._event_bus is not None:
            event = SynapseEvent(
                event_type=SynapseEventType.BOUNTY_SOLUTION_PENDING,
                source_system="oikos",
                data={
                    "bounty_id": active.bounty_id,
                    "candidate_id": candidate.candidate_id,
                    "platform": candidate.platform,
                    "issue_url": candidate.issue_url,
                    "title": candidate.title,
                    "description": candidate.description,
                    "reward_usd": str(candidate.reward_usd),
                    "estimated_cost_usd": str(evaluation.estimated_cost_usd),
                    "roi_score": str(evaluation.roi_score),
                    "confidence": str(evaluation.confidence),
                    "repo_owner": candidate.repo_owner,
                    "repo_name": candidate.repo_name,
                    "tags": candidate.tags,
                },
            )
            await self._event_bus.emit(event)

        self._log.info(
            "bounty_accepted",
            bounty_id=active.bounty_id,
            candidate_id=candidate.candidate_id,
            platform=candidate.platform,
            reward_usd=str(candidate.reward_usd),
            estimated_cost_usd=str(evaluation.estimated_cost_usd),
            roi_score=str(evaluation.roi_score),
        )

        return active

    # ─── Foraging Cycle ────────────────────────────────────────────

    async def run_foraging_cycle(
        self,
        economic_state: EconomicState,
    ) -> list[BountyEvaluation]:
        """
        Main entry point - called periodically by the Oikos service.

        Steps:
          1. Scan all configured platforms for new candidates
          2. Estimate cost and evaluate each candidate against policy
          3. Accept approved candidates (emit BOUNTY_SOLUTION_PENDING)
          4. Emit economic percept summarising foraging results
          5. Return all evaluations for audit trail

        Returns:
            List of BountyEvaluation for every candidate discovered
            this cycle (both approved and rejected).
        """
        self._log.info("foraging_cycle_started")

        # Step 1: Discover
        candidates = await self.scan_platforms()
        if not candidates:
            self._log.info("foraging_cycle_no_candidates")
            return []

        evaluations: list[BountyEvaluation] = []
        accepted_bounties: list[ActiveBounty] = []

        for candidate in candidates:
            # Step 2: Estimate cost and evaluate
            candidate.estimated_cost_usd = self._estimate_cost(candidate)
            evaluation = self.evaluate(candidate, economic_state)
            candidate.evaluated = True
            evaluations.append(evaluation)

            if evaluation.approved:
                # Step 3: Accept
                candidate.accepted = True
                active = await self.accept_bounty(candidate, evaluation)
                accepted_bounties.append(active)
            else:
                candidate.rejection_reason = "; ".join(evaluation.rejection_reasons)
                self._log.info(
                    "bounty_rejected",
                    candidate_id=candidate.candidate_id,
                    reasons=evaluation.rejection_reasons,
                    roi_score=str(evaluation.roi_score),
                )

        # Step 4: Store candidates snapshot in Redis
        await self._store_candidates_snapshot(candidates)

        # Step 5: Emit foraging summary as economic percept
        await self._emit_foraging_percept(
            total_discovered=len(candidates),
            total_approved=len(accepted_bounties),
            total_rejected=len(candidates) - len(accepted_bounties),
            evaluations=evaluations,
        )

        self._log.info(
            "foraging_cycle_complete",
            discovered=len(candidates),
            approved=len(accepted_bounties),
            rejected=len(candidates) - len(accepted_bounties),
        )

        return evaluations

    # ─── Cost Estimation ───────────────────────────────────────────

    def _estimate_cost(self, candidate: BountyCandidate) -> Decimal:
        """
        Heuristic cost estimation based on difficulty and expected token usage.

        Formula:
          estimated_tokens = difficulty_score * MAX_ESTIMATED_TOKENS
          cost = (estimated_tokens / 1000) * COST_PER_1K_TOKENS

        The difficulty_score is clamped to [0.05, 1.0] so even trivial
        bounties have a non-zero cost floor.
        """
        difficulty = max(candidate.difficulty_score, Decimal("0.05"))
        difficulty = min(difficulty, Decimal("1.0"))

        estimated_tokens = difficulty * Decimal(str(_MAX_ESTIMATED_TOKENS))
        cost = (estimated_tokens / Decimal("1000")) * _COST_PER_1K_TOKENS

        return cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # ─── Internal Helpers ──────────────────────────────────────────

    async def _deduplicate(
        self,
        candidates: list[BountyCandidate],
    ) -> list[BountyCandidate]:
        """
        Filter out candidates whose issue_url is already in the Redis
        known-URLs set. If Redis is unavailable, all candidates pass
        through (fail-open for discovery, fail-closed for acceptance).
        """
        if self._redis is None or not candidates:
            return candidates

        try:
            known_urls: set[str] = set()
            raw: list[str] | None = await self._redis.get_json(REDIS_KEY_KNOWN_URLS)
            if raw:
                known_urls = set(raw)

            new_candidates = [
                c for c in candidates if c.issue_url not in known_urls
            ]
            return new_candidates
        except Exception:
            self._log.exception("redis_dedup_check_failed")
            return candidates

    async def _store_candidates_snapshot(
        self,
        candidates: list[BountyCandidate],
    ) -> None:
        """
        Store the recent candidates list in Redis for observability.
        Overwrites the previous snapshot. TTL = 48 hours.
        """
        if self._redis is None or not candidates:
            return

        try:
            snapshot = [c.model_dump(mode="json") for c in candidates]
            await self._redis.set_json(REDIS_KEY_CANDIDATES, snapshot)
        except Exception:
            self._log.exception("redis_candidates_store_failed")

    async def _emit_foraging_percept(
        self,
        total_discovered: int,
        total_approved: int,
        total_rejected: int,
        evaluations: list[BountyEvaluation],
    ) -> None:
        """
        Emit an economic percept summarising the foraging cycle results.

        This percept is consumed by Nova for belief updates about the
        organism's earning potential and by Evo for hypothesis testing.
        """
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        avg_roi = Decimal("0")
        approved_evals = [e for e in evaluations if e.approved]
        if approved_evals:
            total_roi = sum(e.roi_score for e in approved_evals)
            avg_roi = (total_roi / Decimal(str(len(approved_evals)))).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        try:
            event = SynapseEvent(
                event_type=SynapseEventType.METABOLIC_PRESSURE,
                source_system="oikos",
                data={
                    "kind": "foraging_cycle_result",
                    "total_discovered": total_discovered,
                    "total_approved": total_approved,
                    "total_rejected": total_rejected,
                    "avg_approved_roi": str(avg_roi),
                    "timestamp": utc_now().isoformat(),
                },
            )
            await self._event_bus.emit(event)
        except Exception:
            self._log.exception("foraging_percept_emit_failed")

    @property
    def stats(self) -> dict[str, int]:
        """Observability snapshot for OikosService.stats aggregation."""
        return {"known_urls": 0}

    async def _on_bounty_paid(self, event: object) -> None:
        """
        Handle BOUNTY_PAID event - log for audit trail.

        The actual balance update is handled by OikosService. This handler
        exists so the BountyHunter can track its own success rate.
        """
        from systems.synapse.types import SynapseEvent

        if not isinstance(event, SynapseEvent):
            return

        self._log.info(
            "bounty_paid_received",
            bounty_id=event.data.get("bounty_id", ""),
            reward_usd=event.data.get("reward_usd", ""),
        )

    @staticmethod
    def _infer_difficulty(labels: list[str]) -> Decimal:
        """
        Heuristic difficulty scoring from issue labels.

        Maps common GitHub labels to a 0.0–1.0 difficulty score.
        Higher difficulty = more tokens = higher estimated cost.
        """
        text = " ".join(lab.lower() for lab in labels)
        if "good first issue" in text or "easy" in text or "beginner" in text:
            return Decimal("0.15")
        if "medium" in text or "intermediate" in text:
            return Decimal("0.45")
        if "hard" in text or "complex" in text or "expert" in text:
            return Decimal("0.80")
        if "critical" in text or "security" in text:
            return Decimal("0.90")
        # Default: moderate difficulty
        return Decimal("0.40")

    @staticmethod
    def _infer_capability(title: str, labels: list[str]) -> Decimal:
        """
        Heuristic capability match from title + labels.

        The organism is strong at: Python, security, API, testing, docs,
        data analysis, automation. Lower match for UI/frontend, mobile,
        embedded, hardware.
        """
        text = (title + " " + " ".join(labels)).lower()
        strong = ["python", "api", "security", "test", "docs", "data",
                  "automation", "backend", "cli", "analysis", "audit",
                  "rust", "typescript", "fastapi", "django"]
        weak = ["ios", "android", "swift", "kotlin", "unity",
                "hardware", "embedded", "cuda", "solidity"]
        strong_hits = sum(1 for kw in strong if kw in text)
        weak_hits = sum(1 for kw in weak if kw in text)
        if weak_hits > 0 and strong_hits == 0:
            return Decimal("0.15")
        if strong_hits >= 2:
            return Decimal("0.90")
        if strong_hits == 1:
            return Decimal("0.70")
        # Unknown domain - assume moderate capability
        return Decimal("0.50")

    @staticmethod
    def _parse_deadline(deadline_str: str | None) -> datetime | None:
        """Parse a deadline string into a timezone-aware datetime, returning None on failure."""
        if deadline_str is None:
            return None
        try:
            dt = datetime.fromisoformat(deadline_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except (ValueError, TypeError):
            return None
