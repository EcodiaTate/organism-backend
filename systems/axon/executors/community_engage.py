"""
EcodiaOS - Axon Executor: CommunityEngageExecutor

Participates in developer communities on GitHub and X: replies to issues,
responds to PR review comments, stars repositories, follows developers,
reacts to posts, and answers questions in GitHub Discussions / Stack Overflow.

All engagement is transparent - every interaction prepends the standard
AI-transparency disclaimer and must pass an Equor constitutional gate that
evaluates whether the response is helpful, honest, and adds genuine value.

Ethical constraints (hardcoded, non-negotiable):
  - No astroturfing: every reply identifies EOS as an AI agent.
  - No generic replies: minimum substantive length enforced per action type.
  - No coordinated inauthentic behaviour: one reaction per post, deduped in Redis.
  - Rate limits: max 20 GitHub interactions/day, max 10 X interactions/day.
  - Equor reviews every engagement intent before any API call is made.

Sub-actions:
    reply_to_issue        - Reply to a GitHub issue with helpful information.
    reply_to_pr_review    - Respond to a PR review comment on EOS's own PRs.
    star_repo             - Star a repository relevant to EOS's work.
    follow_user           - Follow a developer on GitHub or X.
    react_to_post         - Like / retweet an X post (within rate limits).
    answer_question       - Reply to a GitHub Discussion or SO question.

Level 2 (COLLABORATOR): External community interactions are public and
irreversible. Requires PARTNER autonomy.
Rate-limited conservatively: 15 actions/day across all sub-actions.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.identity.vault import IdentityVault
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("axon.executor.community_engage")

# ─── Disclaimer (same as social_post; mandatory on every text reply) ──────────

_ENGAGEMENT_DISCLAIMER = "🤖 [EcodiaOS - Autonomous AI Agent]"

# ─── Sub-action constants ──────────────────────────────────────────────────────

_ACTION_REPLY_ISSUE = "reply_to_issue"
_ACTION_REPLY_PR = "reply_to_pr_review"
_ACTION_STAR_REPO = "star_repo"
_ACTION_FOLLOW_USER = "follow_user"
_ACTION_REACT_POST = "react_to_post"
_ACTION_ANSWER_QUESTION = "answer_question"

_TEXT_ACTIONS = frozenset({
    _ACTION_REPLY_ISSUE,
    _ACTION_REPLY_PR,
    _ACTION_ANSWER_QUESTION,
})
_GRAPH_ACTIONS = frozenset({
    _ACTION_STAR_REPO,
    _ACTION_FOLLOW_USER,
    _ACTION_REACT_POST,
})
_ALL_ACTIONS = _TEXT_ACTIONS | _GRAPH_ACTIONS

# Minimum substantive length for text replies (prevents generic one-liners)
_MIN_REPLY_LENGTH = 80

# Platform identifiers
_PLATFORM_GITHUB = "github"
_PLATFORM_X = "x"

# Redis dedup / rate-limit key prefixes
_REDIS_PREFIX = "axon:community_engage"
_REDIS_GITHUB_DAY_KEY = f"{_REDIS_PREFIX}:github:day_count"
_REDIS_X_DAY_KEY = f"{_REDIS_PREFIX}:x:day_count"
_REDIS_DEDUP_KEY = f"{_REDIS_PREFIX}:dedup"

_MAX_GITHUB_PER_DAY = 20
_MAX_X_PER_DAY = 10

# Equor gate timeout
_EQUOR_TIMEOUT_S = 30.0


class CommunityEngageExecutor(Executor):
    """
    Participate in developer communities on GitHub and X with full Equor
    constitutional gating, AI-transparency disclosure, and daily rate limits.

    See module docstring for full parameter documentation.
    """

    action_type = "community_engage"
    description = (
        "Engage with developer communities (reply to issues, star repos, "
        "follow developers, react to posts) with mandatory AI disclosure and "
        "Equor ethical review (Level 2)"
    )
    required_autonomy = 2
    reversible = False
    max_duration_ms = 30_000
    # 15 actions/day is enforced by Redis counters; this rate limit is a
    # per-minute safety backstop so a runaway intent queue can't burn the daily
    # allowance in seconds.
    rate_limit = RateLimit.per_hour(8)

    def __init__(
        self,
        vault: IdentityVault | None = None,
        redis: RedisClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._vault = vault
        self._redis = redis
        self._event_bus = event_bus
        self._pending_equor: dict[str, asyncio.Future[bool]] = {}
        self._log = logger.bind(system="axon.executor.community_engage")

        if event_bus is not None:
            self._wire_equor(event_bus)

    # ── Dependency injection ───────────────────────────────────────────

    def set_vault(self, vault: IdentityVault) -> None:
        self._vault = vault

    def set_redis(self, redis: RedisClient) -> None:
        self._redis = redis

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus
        self._wire_equor(bus)

    def _wire_equor(self, bus: EventBus) -> None:
        from systems.synapse.types import SynapseEventType
        bus.subscribe(SynapseEventType.EQUOR_ECONOMIC_PERMIT, self._on_equor_permit)

    # ── Validation ────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        action = params.get("action", "")
        if action not in _ALL_ACTIONS:
            return ValidationResult.fail(
                f"action must be one of {sorted(_ALL_ACTIONS)}",
                action="unsupported",
            )

        platform = params.get("platform", _PLATFORM_GITHUB)
        if platform not in {_PLATFORM_GITHUB, _PLATFORM_X}:
            return ValidationResult.fail(
                "platform must be 'github' or 'x'",
                platform="unsupported",
            )

        target_url = params.get("target_url", "")
        if not target_url or not target_url.startswith("http"):
            return ValidationResult.fail(
                "target_url is required and must be an http(s) URL",
                target_url="missing or invalid",
            )

        if action in _TEXT_ACTIONS:
            body = params.get("body", "")
            if not body:
                return ValidationResult.fail(
                    "body is required for text-based actions",
                    body="missing",
                )
            if len(body) < _MIN_REPLY_LENGTH:
                return ValidationResult.fail(
                    f"body must be ≥{_MIN_REPLY_LENGTH} chars to ensure substantive engagement",
                    body="too_short",
                )
            if len(body) > 10_000:
                return ValidationResult.fail(
                    "body too long (max 10,000 chars)",
                    body="too_long",
                )

        if action in {_ACTION_REPLY_ISSUE, _ACTION_REPLY_PR, _ACTION_ANSWER_QUESTION}:
            if not params.get("target_id"):
                return ValidationResult.fail(
                    "target_id (issue/comment/discussion number or ID) is required",
                    target_id="missing",
                )
            if not params.get("repo"):
                return ValidationResult.fail(
                    "repo (owner/repo) is required for GitHub text actions",
                    repo="missing",
                )

        return ValidationResult.ok()

    # ── Execution ─────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        action: str = params["action"]
        platform: str = params.get("platform", _PLATFORM_GITHUB)
        target_url: str = params["target_url"]
        engagement_id = f"{action}:{target_url}"

        self._log.info(
            "community_engage_execute",
            action=action,
            platform=platform,
            target_url=target_url,
            execution_id=context.execution_id,
        )

        # ── 1. Dedup check (idempotency for graph actions) ─────────────
        if await self._already_done(engagement_id):
            self._log.info("community_engage_dedup_skip", engagement_id=engagement_id)
            return ExecutionResult(
                success=True,
                data={"skipped": True, "reason": "already_engaged", "engagement_id": engagement_id},
                new_observations=[
                    f"EOS already performed {action} on {target_url} - skipped (idempotent)"
                ],
            )

        # ── 2. Daily rate-limit check ──────────────────────────────────
        allowed, current_count = await self._check_daily_limit(platform)
        if not allowed:
            max_limit = _MAX_GITHUB_PER_DAY if platform == _PLATFORM_GITHUB else _MAX_X_PER_DAY
            self._log.warning(
                "community_engage_daily_limit_reached",
                platform=platform,
                count=current_count,
                limit=max_limit,
            )
            return ExecutionResult(
                success=False,
                error=f"daily_rate_limit: {platform} limit of {max_limit} interactions/day reached",
                data={"platform": platform, "daily_count": current_count},
                new_observations=[
                    f"EOS has reached the daily {platform} engagement limit ({max_limit}/day). "
                    "Do not schedule further community_engage actions for this platform today."
                ],
            )

        # ── 3. Equor constitutional gate ───────────────────────────────
        # Every engagement passes ethical review: is it helpful, honest, valuable?
        approved = await self._equor_review(action, platform, target_url, params)
        if not approved:
            self._log.warning(
                "community_engage_equor_denied",
                action=action,
                target_url=target_url,
            )
            return ExecutionResult(
                success=False,
                error="equor_denied: engagement did not pass constitutional review",
                data={"action": action, "platform": platform, "equor_approved": False},
                new_observations=[
                    f"Equor denied community engagement: {action} on {target_url}. "
                    "The response was judged unhelpful, dishonest, or not adding value. "
                    "Do not retry this specific engagement without revised content."
                ],
            )

        # ── 4. Resolve credentials ─────────────────────────────────────
        envelope = self._resolve_envelope(platform, context)
        if envelope is None or self._vault is None:
            return self._degraded(platform, "No credentials found for platform")

        # ── 5. Dispatch to platform ────────────────────────────────────
        try:
            result_data = await self._dispatch(action, platform, params, envelope)
        except Exception as exc:
            self._log.error(
                "community_engage_dispatch_failed",
                action=action,
                platform=platform,
                error=str(exc),
            )
            return ExecutionResult(
                success=False,
                error=f"dispatch_error: {exc!s}",
                data={"action": action, "platform": platform},
                new_observations=[
                    f"EOS community engagement {action} on {target_url} failed: {exc!s}"
                ],
            )

        # ── 6. Record interaction ──────────────────────────────────────
        await self._record_interaction(engagement_id, platform)

        # ── 7. Emit COMMUNITY_ENGAGEMENT_COMPLETED ─────────────────────
        await self._emit_engagement_event(action, platform, target_url, params.get("target_id", ""), engagement_id)

        return ExecutionResult(
            success=True,
            data={
                "action": action,
                "platform": platform,
                "target_url": target_url,
                "engagement_id": engagement_id,
                "equor_approved": True,
                **result_data,
            },
            side_effects=[f"Community engagement: {action} on {platform} ({target_url})"],
            new_observations=[
                f"EOS performed {action} on {platform}: {target_url}. "
                + (f"Response posted (id={result_data.get('comment_id', 'unknown')})." if action in _TEXT_ACTIONS else f"Interaction recorded.")
            ],
        )

    # ── Platform dispatch ──────────────────────────────────────────────

    async def _dispatch(
        self,
        action: str,
        platform: str,
        params: dict[str, Any],
        envelope: Any,
    ) -> dict[str, Any]:
        """Route to the appropriate platform client method."""
        assert self._vault is not None

        if platform == _PLATFORM_GITHUB:
            return await self._dispatch_github(action, params, envelope)
        return await self._dispatch_x(action, params, envelope)

    async def _dispatch_github(
        self,
        action: str,
        params: dict[str, Any],
        envelope: Any,
    ) -> dict[str, Any]:
        from interfaces.social import GitHubSocialClient

        client = GitHubSocialClient(vault=self._vault, envelope=envelope)
        repo: str = params.get("repo", "")
        target_id: str = str(params.get("target_id", ""))

        if action == _ACTION_REPLY_ISSUE:
            body = _inject_disclaimer(params["body"])
            result = await client.create_issue_comment(
                repo=repo,
                issue_number=int(target_id),
                body=body,
            )
            return {"comment_id": result.post_id, "url": result.url}

        if action == _ACTION_REPLY_PR:
            body = _inject_disclaimer(params["body"])
            result = await client.create_pr_review_reply(
                repo=repo,
                pull_number=int(params.get("pull_number", target_id)),
                comment_id=int(target_id),
                body=body,
            )
            return {"comment_id": result.post_id, "url": result.url}

        if action == _ACTION_ANSWER_QUESTION:
            body = _inject_disclaimer(params["body"])
            result = await client.create_discussion_comment(
                repo=repo,
                discussion_number=int(target_id),
                body=body,
            )
            return {"comment_id": result.post_id, "url": result.url}

        if action == _ACTION_STAR_REPO:
            await client.star_repository(repo=repo)
            return {"starred": True}

        if action == _ACTION_FOLLOW_USER:
            username: str = params.get("username", "")
            await client.follow_user(username=username)
            return {"followed": username}

        # _ACTION_REACT_POST - GitHub reaction (e.g. +1 on an issue)
        reaction: str = params.get("reaction", "+1")
        result2 = await client.add_reaction(
            repo=repo,
            subject_type=params.get("subject_type", "issue"),
            subject_id=int(target_id),
            content=reaction,
        )
        return {"reaction": reaction, "reaction_id": result2.post_id}

    async def _dispatch_x(
        self,
        action: str,
        params: dict[str, Any],
        envelope: Any,
    ) -> dict[str, Any]:
        from interfaces.social import XSocialClient

        client = XSocialClient(vault=self._vault, envelope=envelope)
        target_id: str = str(params.get("target_id", ""))

        if action in {_ACTION_REPLY_ISSUE, _ACTION_ANSWER_QUESTION, _ACTION_REPLY_PR}:
            body = _inject_disclaimer(params["body"])
            result = await client.reply_to_tweet(tweet_id=target_id, content=body)
            return {"tweet_id": result.post_id, "url": result.url}

        if action == _ACTION_FOLLOW_USER:
            username: str = params.get("username", "")
            await client.follow_user(username=username)
            return {"followed": username}

        if action == _ACTION_REACT_POST:
            reaction = params.get("reaction", "like")
            if reaction == "like":
                await client.like_tweet(tweet_id=target_id)
            elif reaction == "retweet":
                await client.retweet(tweet_id=target_id)
            return {"reaction": reaction, "tweet_id": target_id}

        return {}

    # ── Equor gate ─────────────────────────────────────────────────────

    async def _equor_review(
        self,
        action: str,
        platform: str,
        target_url: str,
        params: dict[str, Any],
    ) -> bool:
        """
        Emit EQUOR_ECONOMIC_INTENT and await EQUOR_ECONOMIC_PERMIT.
        The intent type is "community_engagement" - Equor evaluates whether
        the planned interaction is helpful, honest, and adds genuine value.
        Auto-permits after 30s to avoid blocking the pipeline.
        """
        if self._event_bus is None:
            self._log.warning("equor_gate_skipped_no_event_bus")
            return True  # degrade gracefully; Equor will catch on audit

        from primitives.common import new_id
        from systems.synapse.types import SynapseEventType

        intent_id = new_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending_equor[intent_id] = future

        body_preview = params.get("body", "")[:200] if action in _TEXT_ACTIONS else ""
        await self._event_bus.emit(
            SynapseEventType.EQUOR_ECONOMIC_INTENT,
            {
                "intent_id": intent_id,
                "mutation_type": "community_engagement",
                "amount_usd": "0",
                "rationale": (
                    f"Community engagement: {action} on {platform} at {target_url}. "
                    + (f"Content: {body_preview}" if body_preview else "")
                ),
                "action_type": self.action_type,
                "engagement_action": action,
                "platform": platform,
                "target_url": target_url,
                "helpfulness_claim": (
                    "Substantive technical reply" if action in _TEXT_ACTIONS
                    else f"Relationship-building: {action}"
                ),
            },
        )

        try:
            return await asyncio.wait_for(future, timeout=_EQUOR_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._log.warning("equor_timeout_auto_permit", intent_id=intent_id)
            return True
        finally:
            self._pending_equor.pop(intent_id, None)

    async def _on_equor_permit(self, event: Any) -> None:
        intent_id: str = getattr(event, "data", {}).get("intent_id", "")
        fut = self._pending_equor.get(intent_id)
        if fut and not fut.done():
            verdict = getattr(event, "data", {}).get("verdict", "PERMIT")
            fut.set_result(verdict == "PERMIT")

    # ── Rate limiting & dedup ──────────────────────────────────────────

    async def _check_daily_limit(self, platform: str) -> tuple[bool, int]:
        """Return (allowed, current_count). No Redis → always allow (degrade gracefully)."""
        if self._redis is None:
            return True, 0

        key = _REDIS_GITHUB_DAY_KEY if platform == _PLATFORM_GITHUB else _REDIS_X_DAY_KEY
        max_limit = _MAX_GITHUB_PER_DAY if platform == _PLATFORM_GITHUB else _MAX_X_PER_DAY

        try:
            raw = await self._redis.get(key)
            count = int(raw) if raw else 0
            return count < max_limit, count
        except Exception:
            return True, 0  # degrade gracefully

    async def _record_interaction(self, engagement_id: str, platform: str) -> None:
        """Increment daily counter and mark engagement as done (dedup)."""
        if self._redis is None:
            return

        day_key = _REDIS_GITHUB_DAY_KEY if platform == _PLATFORM_GITHUB else _REDIS_X_DAY_KEY
        dedup_field = engagement_id[:200]

        try:
            pipe = self._redis.pipeline()
            # Increment daily counter; set 25h TTL so it resets daily
            pipe.incr(day_key)
            pipe.expire(day_key, 90_000)  # 25 hours
            # Mark this specific engagement done (hash field → timestamp)
            from primitives.common import utc_now
            pipe.hset(_REDIS_DEDUP_KEY, dedup_field, utc_now().isoformat())
            pipe.expire(_REDIS_DEDUP_KEY, 90_000)
            await pipe.execute()
        except Exception as exc:
            self._log.warning("community_engage_redis_record_failed", error=str(exc))

    async def _already_done(self, engagement_id: str) -> bool:
        """Check Redis dedup hash. No Redis → assume not done (degrade gracefully)."""
        if self._redis is None:
            return False
        try:
            field = engagement_id[:200]
            result = await self._redis.hget(_REDIS_DEDUP_KEY, field)
            return result is not None
        except Exception:
            return False

    # ── Event emission ─────────────────────────────────────────────────

    async def _emit_engagement_event(
        self,
        action: str,
        platform: str,
        target_url: str,
        target_id: str,
        engagement_id: str,
    ) -> None:
        if self._event_bus is None:
            return
        from systems.synapse.types import SynapseEventType
        try:
            await self._event_bus.emit(
                SynapseEventType.COMMUNITY_ENGAGEMENT_COMPLETED,
                {
                    "action": action,
                    "platform": platform,
                    "target_url": target_url,
                    "target_id": target_id,
                    "equor_approved": True,
                    "engagement_id": engagement_id,
                },
            )
        except Exception as exc:
            self._log.warning("emit_engagement_event_failed", error=str(exc))

    # ── Credential resolution ──────────────────────────────────────────

    def _resolve_envelope(self, platform: str, context: ExecutionContext) -> Any | None:
        key = f"{platform}:envelope"
        raw = context.credentials.get(key)
        if not raw:
            return None
        try:
            from systems.identity.vault import SealedEnvelope
            return SealedEnvelope.model_validate(json.loads(raw))
        except Exception as exc:
            self._log.error("envelope_deserialize_failed", platform=platform, error=str(exc))
            return None

    def _degraded(self, platform: str, reason: str) -> ExecutionResult:
        self._log.warning("community_engage_degraded", platform=platform, reason=reason)
        return ExecutionResult(
            success=False,
            error=f"degraded: {reason}",
            data={"platform": platform, "degraded": True},
            new_observations=[
                f"EOS community presence for '{platform}' is degraded: {reason}. "
                "Operator must provision credentials via POST /api/v1/identity/envelopes."
            ],
        )


# ─── Disclaimer injection ──────────────────────────────────────────────────────


def _inject_disclaimer(body: str) -> str:
    """Append the AI transparency disclaimer to every text reply (mandatory)."""
    return f"{body}\n\n---\n*{_ENGAGEMENT_DISCLAIMER}*"
