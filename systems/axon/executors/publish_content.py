"""
EcodiaOS - Axon Executor: PublishContentExecutor

Orchestrates multi-platform content publication from a single Nova intent.

Flow:
  1. Nova emits Intent(action_type="publish_content", params={content_type, topic, context, platforms?})
  2. Equor gates the intent (Level 2 - COLLABORATOR autonomy; constitutional honesty check)
  3. PublishContentExecutor:
       a. Calls ContentEngine.generate() to produce per-platform formatted variants
       b. For each target platform, resolves credentials from IdentityVault
       c. Dispatches to the appropriate platform client (X, GitHub, LinkedIn, Telegram,
          Dev.to, Hashnode)
       d. Aggregates results → ExecutionResult

Mandatory constraints (cannot be suppressed):
  - SOCIAL_DISCLAIMER appended to every post via ContentEngine / platform clients
  - Equor constitutional honesty check (required_autonomy = 2)
  - Rate limits enforced per-platform (see _PLATFORM_RATE_LIMITS)
  - No platform-key secrets in params - all via IdentityVault

Platform routing:
  If params["platforms"] is given, only those platforms receive the post.
  Otherwise, PLATFORM_ROUTING[ContentType] selects the default set.

Required params:
    content_type (str): One of the ContentType enum values.
    topic (str): High-level description of the content to publish (≤500 chars).

Optional params:
    context (dict): Supporting data for content generation
                    (reward_usd, pr_url, repo, yield_rate, etc.).
    platforms (list[str]): Override default platform routing.
    title (str): Article title (LinkedIn article / Dev.to / Hashnode).
    tags (list[str]): Platform tags for Dev.to / Hashnode.
    published (bool): Dev.to/Hashnode draft flag (default True).
    canonical_url (str): For cross-posting canonical attribution.

Rate limits:
  - 10 publish_content actions per hour (executor-level)
  - Individual platform clients have their own limits (enforced at client level)

Level 2: Publishing to external platforms requires PARTNER autonomy.
Reversible: False (real-world post published externally).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from interfaces.social import (
    ContentType,
    DevToClient,
    GitHubSocialClient,
    HashnodeClient,
    LinkedInSocialClient,
    PLATFORM_ROUTING,
    PostResult,
    TelegramChannelClient,
    XSocialClient,
    truncate_for_x,
)
from interfaces.social.types import SocialPlatform
from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault, SealedEnvelope
    from systems.voxis.content_engine import ContentEngine

logger = structlog.get_logger("axon.executor.publish_content")

# ─── Constants ────────────────────────────────────────────────────────────────

SOCIAL_DISCLAIMER = "🤖 [Autonomous EcodiaOS Agent]"
_CRED_KEY_PATTERN = "{platform_id}:envelope"

_SUPPORTED_PLATFORMS = frozenset({
    "x", "github", "linkedin", "telegram_channel", "discord", "devto", "hashnode",
})

_VALID_CONTENT_TYPES = frozenset(ct.value for ct in ContentType)


# ─── Executor ─────────────────────────────────────────────────────────────────


class PublishContentExecutor(Executor):
    """
    Multi-platform content publisher with ContentEngine integration.

    See module docstring for full parameter documentation and flow.
    """

    action_type = "publish_content"
    description = (
        "Generate and publish content to multiple social platforms "
        "with mandatory AI disclaimer (Level 2)"
    )
    required_autonomy = 2
    reversible = False
    max_duration_ms = 60_000   # Allow time for multiple platform API calls
    rate_limit = RateLimit.per_hour(10)

    def __init__(
        self,
        vault: IdentityVault | None = None,
        content_engine: ContentEngine | None = None,
    ) -> None:
        self._vault = vault
        self._content_engine = content_engine
        self._event_bus: Any = None
        self._logger = logger.bind(system="axon.executor.publish_content")

    def set_content_engine(self, engine: ContentEngine) -> None:
        """Inject ContentEngine after construction (wiring layer)."""
        self._content_engine = engine

    def set_event_bus(self, bus: Any) -> None:
        """Inject Synapse event bus for CONTENT_PUBLISHED emission (wiring layer)."""
        self._event_bus = bus

    # ── Validation ────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        content_type_raw = params.get("content_type", "")
        if not content_type_raw:
            return ValidationResult.fail("content_type is required")
        if content_type_raw not in _VALID_CONTENT_TYPES:
            return ValidationResult.fail(
                f"content_type must be one of {sorted(_VALID_CONTENT_TYPES)}",
                content_type="unsupported",
            )

        topic = params.get("topic", "")
        if not topic:
            return ValidationResult.fail("topic is required", topic="missing")
        if not isinstance(topic, str):
            return ValidationResult.fail("topic must be a string", topic="wrong type")
        if len(topic) > 500:
            return ValidationResult.fail("topic too long (max 500 chars)")

        platforms = params.get("platforms")
        if platforms is not None:
            if not isinstance(platforms, list):
                return ValidationResult.fail("platforms must be a list of strings")
            unsupported = [p for p in platforms if p not in _SUPPORTED_PLATFORMS]
            if unsupported:
                return ValidationResult.fail(
                    f"unsupported platforms: {unsupported}. "
                    f"Supported: {sorted(_SUPPORTED_PLATFORMS)}",
                )

        tags = params.get("tags")
        if tags is not None and not isinstance(tags, list):
            return ValidationResult.fail("tags must be a list of strings")

        return ValidationResult.ok()

    # ── Execution ─────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        content_type = ContentType(params["content_type"])
        topic: str = params["topic"]
        ctx_data: dict[str, Any] = params.get("context", {}) or {}
        title: str = params.get("title", topic)
        tags: list[str] = params.get("tags", []) or []
        published: bool = params.get("published", True)
        canonical_url: str | None = params.get("canonical_url")

        # Determine target platforms
        target_platforms: list[str] = params.get("platforms") or PLATFORM_ROUTING.get(
            content_type, list(_SUPPORTED_PLATFORMS)
        )

        self._logger.info(
            "publish_content_execute",
            content_type=content_type,
            topic_preview=topic[:80],
            platforms=target_platforms,
            execution_id=context.execution_id,
        )

        # ── 1. Generate per-platform content variants ──────────────────
        variants = await self._generate_variants(
            content_type, topic, ctx_data, target_platforms
        )

        # ── 2. Dispatch to each platform ───────────────────────────────
        results: dict[str, PostResult] = {}
        for platform in target_platforms:
            content = variants.get(platform, "")
            if not content:
                self._logger.warning(
                    "publish_content_no_variant", platform=platform
                )
                continue

            result = await self._dispatch_platform(
                platform=platform,
                content=content,
                title=title,
                tags=tags,
                published=published,
                canonical_url=canonical_url,
                params=params,
                context=context,
            )
            results[platform] = result

        # ── 3. Aggregate results ───────────────────────────────────────
        result = self._aggregate_results(results, content_type, topic, target_platforms)

        # ── 4. Emit CONTENT_PUBLISHED so Oikos can track monetisation ─
        if result.success and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                successes = {p: r for p, r in results.items() if r.success}
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.CONTENT_PUBLISHED,
                    source_system="axon",
                    data={
                        "content_type": content_type.value,
                        "topic_preview": topic[:120],
                        "published_platforms": list(successes.keys()),
                        "platform_count": len(successes),
                        "post_ids": {
                            p: r.post_id for p, r in successes.items() if r.post_id
                        },
                        "execution_id": context.execution_id,
                    },
                ))
            except Exception as exc:
                self._logger.warning(
                    "content_published_emit_failed",
                    error=str(exc),
                )

        return result

    # ── Content generation ────────────────────────────────────────────

    async def _generate_variants(
        self,
        content_type: ContentType,
        topic: str,
        ctx_data: dict[str, Any],
        platforms: list[str],
    ) -> dict[str, str]:
        """Generate per-platform content. Falls back to a plain template if engine absent."""
        if self._content_engine is not None:
            try:
                return await self._content_engine.generate(
                    content_type=content_type,
                    topic=topic,
                    context_data=ctx_data,
                    platforms=platforms,
                )
            except Exception as exc:
                self._logger.error(
                    "publish_content_engine_failed",
                    error=str(exc),
                    falling_back="raw_topic",
                )

        # Minimal fallback: use topic as body for all platforms
        base = f"{topic}\n\n{SOCIAL_DISCLAIMER}"
        return {p: base for p in platforms}

    # ── Platform dispatch ─────────────────────────────────────────────

    async def _dispatch_platform(
        self,
        platform: str,
        content: str,
        title: str,
        tags: list[str],
        published: bool,
        canonical_url: str | None,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> PostResult:
        """Route to the correct platform client."""
        envelope = self._resolve_envelope(platform, context)

        if platform == "x":
            if envelope is None or self._vault is None:
                return self._degraded(platform, "no_credentials")
            client = XSocialClient(vault=self._vault, envelope=envelope)
            return await client.post_tweet(content)

        if platform == "github":
            if envelope is None or self._vault is None:
                return self._degraded(platform, "no_credentials")
            client_gh = GitHubSocialClient(vault=self._vault, envelope=envelope)
            description = params.get("github_description", title or "EcodiaOS public note")
            filename = params.get("github_filename", "post.md")
            return await client_gh.create_gist(
                description=description[:256],
                content=content,
                filename=filename,
                public=True,
            )

        if platform == "linkedin":
            if self._vault is None:
                return self._degraded(platform, "no_vault")
            if envelope is None:
                return self._degraded(platform, "no_credentials")
            client_li = LinkedInSocialClient(vault=self._vault, envelope=envelope)
            if params.get("article_url") or canonical_url:
                return await client_li.post_article(
                    title=title,
                    body_text=content,
                    url=params.get("article_url") or canonical_url,
                )
            return await client_li.post_update(text=content)

        if platform == "telegram_channel":
            client_tg = TelegramChannelClient(vault=self._vault, envelope=envelope)
            channel_id = params.get("telegram_channel_id") or None
            return await client_tg.post_to_channel(text=content, channel_id=channel_id)

        if platform == "discord":
            from interfaces.social.discord_client import DiscordClient
            client_dc = DiscordClient(vault=self._vault, envelope=envelope)
            channel_id = params.get("discord_channel_id") or None
            return await client_dc.post_to_channel(text=content, channel_id=channel_id)

        if platform == "devto":
            client_dt = DevToClient(vault=self._vault, envelope=envelope)
            return await client_dt.create_article(
                title=title,
                body_markdown=content,
                tags=tags or [],
                published=published,
                canonical_url=canonical_url,
            )

        if platform == "hashnode":
            client_hn = HashnodeClient(vault=self._vault, envelope=envelope)
            return await client_hn.publish_post(
                title=title,
                body_markdown=content,
                tags=tags or [],
                canonical_url=canonical_url,
            )

        return self._degraded(platform, f"unknown platform: {platform}")

    # ── Result aggregation ────────────────────────────────────────────

    def _aggregate_results(
        self,
        results: dict[str, PostResult],
        content_type: ContentType,
        topic: str,
        attempted_platforms: list[str],
    ) -> ExecutionResult:
        successes = {p: r for p, r in results.items() if r.success}
        failures = {p: r for p, r in results.items() if not r.success}
        skipped = [p for p in attempted_platforms if p not in results]

        overall_success = bool(successes)  # at least one platform succeeded

        published_summaries = [
            f"{p}:{r.post_id or 'ok'}" + (f" ({r.url})" if r.url else "")
            for p, r in successes.items()
        ]
        failure_summaries = [
            f"{p}:{r.error[:80]}" for p, r in failures.items()
        ]

        self._logger.info(
            "publish_content_complete",
            content_type=content_type,
            successes=list(successes.keys()),
            failures=list(failures.keys()),
        )

        observations = [
            f"EOS published {content_type.value} about '{topic[:80]}' "
            f"to {list(successes.keys()) if successes else 'no platforms'}. "
            + (f"Failed: {failure_summaries}" if failures else "")
        ]

        if not successes and failures:
            return ExecutionResult(
                success=False,
                error=f"All platforms failed: {failure_summaries}",
                data={
                    "content_type": content_type.value,
                    "attempted": attempted_platforms,
                    "failures": {p: r.error for p, r in failures.items()},
                },
                new_observations=observations,
            )

        return ExecutionResult(
            success=overall_success,
            data={
                "content_type": content_type.value,
                "topic_preview": topic[:120],
                "published": {p: {"post_id": r.post_id, "url": r.url} for p, r in successes.items()},
                "failed": {p: r.error for p, r in failures.items()},
                "skipped": skipped,
                "disclaimer_injected": True,
                "platform_count": len(successes),
            },
            side_effects=published_summaries,
            new_observations=observations,
        )

    # ── Credential resolution ─────────────────────────────────────────

    def _resolve_envelope(
        self,
        platform: str,
        context: ExecutionContext,
    ) -> SealedEnvelope | None:
        key = _CRED_KEY_PATTERN.format(platform_id=platform)
        raw = context.credentials.get(key)
        if not raw:
            return None
        try:
            from systems.identity.vault import SealedEnvelope
            return SealedEnvelope.model_validate(json.loads(raw))
        except Exception as exc:
            self._logger.error(
                "publish_content_envelope_deserialize_failed",
                platform=platform,
                error=str(exc),
            )
            return None

    def _degraded(self, platform: str, reason: str) -> PostResult:
        self._logger.warning("publish_content_degraded", platform=platform, reason=reason)
        from interfaces.social.types import SocialPlatform as SP
        # Map platform string to enum; fall back to X for unknown
        try:
            sp = SP(platform)
        except ValueError:
            sp = SP.X
        from interfaces.social.types import PostResult as PR
        return PR.fail(sp, error=f"degraded: {reason}")
