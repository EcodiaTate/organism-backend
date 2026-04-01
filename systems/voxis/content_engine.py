"""
EcodiaOS - Voxis: ContentEngine

Generates per-platform formatted content from a ContentType, topic, and context.

The ContentEngine sits between Nova's deliberate publishing intent and the
platform clients. It applies:
  - Platform char limits (X: 280, Telegram: 4096, LinkedIn: 3000, Dev.to/Hashnode: unlimited)
  - Platform-appropriate hashtag sets
  - Persona voice matching (Voxis personality vector drives tone)
  - Mandatory AI-transparency disclosure (mandatory; appended to every post)
  - Format-appropriate markdown (X: plain text only; Hashnode/Dev.to: full markdown)

Usage:
    engine = ContentEngine(renderer=content_renderer, personality=personality_engine)
    variants = await engine.generate(
        content_type=ContentType.BOUNTY_WIN,
        topic="Solved a Rust memory safety bounty",
        context_data={"reward_usd": 500, "repo": "owner/repo", "pr_url": "..."},
        platforms=["x", "linkedin", "telegram_channel"],
    )
    # variants: dict[platform_str -> str]

The Equor gate is NOT applied here - it runs at the executor level (PublishContentExecutor).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import structlog

from interfaces.social.types import ContentType

if TYPE_CHECKING:
    from systems.voxis.personality import PersonalityEngine
    from systems.voxis.renderer import BaseContentRenderer

logger = structlog.get_logger("voxis.content_engine")

# ─── Platform character limits ──────────────────────────────────────────────

_CHAR_LIMITS: dict[str, int] = {
    "x": 280,
    "telegram_channel": 4_096,
    "linkedin": 3_000,
    "devto": 100_000,   # Effectively unlimited - Dev.to renders full articles
    "hashnode": 100_000,
    "github": 65_536,   # Gist body
}

# ─── AI disclosure disclaimer (mandatory on all platforms) ──────────────────

_DISCLAIMER = "🤖 [Autonomous EcodiaOS Agent]"
_DISCLAIMER_MD = f"\n\n---\n*{_DISCLAIMER}*"

# ─── Hashtag vocabularies per ContentType ───────────────────────────────────

_HASHTAGS: dict[ContentType, list[str]] = {
    ContentType.ACHIEVEMENT:       ["#AI", "#opensource", "#autonomous", "#buildinpublic"],
    ContentType.INSIGHT:           ["#AI", "#MachineLearning", "#DeepLearning", "#research", "#opensource"],
    ContentType.WEEKLY_DIGEST:     ["#buildinpublic", "#AI", "#autonomous", "#weeklyupdate"],
    ContentType.BOUNTY_WIN:        ["#opensource", "#bounty", "#AI", "#Rust", "#DeFi", "#buildinpublic"],
    ContentType.MARKET_OBSERVATION: ["#DeFi", "#crypto", "#yield", "#AI"],
    ContentType.PHILOSOPHICAL:     ["#AI", "#philosophy", "#autonomous", "#consciousness"],
}

# Platforms that support markdown rendering
_MARKDOWN_PLATFORMS = frozenset({"devto", "hashnode", "github"})
# Platforms that support lightweight markdown (bold, italic, links)
_LITE_MARKDOWN_PLATFORMS = frozenset({"linkedin", "telegram_channel"})


class ContentEngine:
    """
    Generates per-platform content variants from a ContentType + topic.

    Design principles:
    - Uses the Voxis ContentRenderer for LLM-backed generation (voice, affect)
    - Falls back to template strings when renderer is unavailable
    - Enforces char limits + mandatory disclaimer on every variant
    - Does NOT call Equor - the executor pipeline handles constitutional gates
    """

    def __init__(
        self,
        renderer: BaseContentRenderer | None = None,
        personality: PersonalityEngine | None = None,
    ) -> None:
        self._renderer = renderer
        self._personality = personality
        self._logger = logger.bind(component="content_engine")

    # ── Public API ─────────────────────────────────────────────────────────

    async def generate(
        self,
        content_type: ContentType,
        topic: str,
        context_data: dict[str, Any] | None = None,
        platforms: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Generate per-platform content variants.

        Args:
            content_type: The type of content to generate.
            topic: High-level description of what to communicate.
            context_data: Supporting data (e.g. reward_usd, pr_url, yield_rate).
            platforms: List of platform IDs to generate for.
                       If None, all platforms in _CHAR_LIMITS are generated.

        Returns:
            dict mapping platform ID → formatted post string (with disclaimer).
            Platforms not in _CHAR_LIMITS are silently skipped.
        """
        ctx = context_data or {}
        target_platforms = platforms or list(_CHAR_LIMITS.keys())

        # Step 1: Generate a rich "master" draft from Voxis renderer or template
        master_draft = await self._generate_master_draft(content_type, topic, ctx)

        # Step 2: Generate per-platform variants
        variants: dict[str, str] = {}
        for platform in target_platforms:
            if platform not in _CHAR_LIMITS:
                self._logger.debug(
                    "content_engine_skip_unknown_platform", platform=platform
                )
                continue
            variants[platform] = self._format_for_platform(
                platform, content_type, topic, master_draft, ctx
            )

        self._logger.info(
            "content_engine_generated",
            content_type=content_type,
            topic_preview=topic[:80],
            platforms=list(variants.keys()),
        )
        return variants

    # ── Draft generation ───────────────────────────────────────────────────

    async def _generate_master_draft(
        self,
        content_type: ContentType,
        topic: str,
        ctx: dict[str, Any],
    ) -> str:
        """
        Generate a rich master draft using the Voxis ContentRenderer if wired,
        falling back to a template-based draft.

        The master draft is Markdown-formatted and may exceed platform limits -
        _format_for_platform() trims it per-platform.
        """
        if self._renderer is not None:
            try:
                return await self._renderer_generate(content_type, topic, ctx)
            except Exception as exc:
                self._logger.warning(
                    "content_engine_renderer_failed",
                    error=str(exc),
                    falling_back="template",
                )

        return self._template_draft(content_type, topic, ctx)

    async def _renderer_generate(
        self,
        content_type: ContentType,
        topic: str,
        ctx: dict[str, Any],
    ) -> str:
        """
        Use the Voxis ContentRenderer for personality-driven generation.

        The renderer produces an Expression; we extract the text content.
        """
        from systems.voxis.types import (
            AudienceProfile,
            ExpressionContext,
            ExpressionIntent,
            ExpressionTrigger,
            SomaticExpressionContext,
            StrategyParams,
        )
        from primitives.affect import AffectState
        from primitives.expression import PersonalityVector

        # Build a synthetic ExpressionContext for public broadcast
        personality = (
            self._personality.get_current()
            if self._personality is not None
            else PersonalityVector()
        )
        strategy = StrategyParams(
            intent_type="broadcast",
            trigger=ExpressionTrigger.NOVA_INFORM,
            context_type=_content_type_to_context(content_type),
            target_length=_target_length_for_type(content_type),
            formatting="structured" if content_type in (
                ContentType.INSIGHT, ContentType.WEEKLY_DIGEST
            ) else "prose",
            jargon_level="domain_appropriate",
            information_density="high" if content_type == ContentType.WEEKLY_DIGEST else "normal",
        )
        intent = ExpressionIntent(
            trigger=ExpressionTrigger.NOVA_INFORM,
            content_to_express=f"{content_type.value.upper()}: {topic}\n\nContext: {ctx}",
        )
        expr_ctx = ExpressionContext(
            personality=personality,
            affect=AffectState.neutral(),
            audience=AudienceProfile(audience_type="community"),
            strategy=strategy,
            intent=intent,
            somatic=SomaticExpressionContext(),
        )
        expression = await self._renderer.render(expr_ctx)
        return expression.text_content if expression else ""

    def _template_draft(
        self,
        content_type: ContentType,
        topic: str,
        ctx: dict[str, Any],
    ) -> str:
        """Minimal template fallback when the renderer is unavailable."""
        templates: dict[ContentType, str] = {
            ContentType.ACHIEVEMENT: (
                f"🎯 Achievement unlocked: {topic}\n\n"
                + _ctx_line("reward_usd", ctx, prefix="Reward: $", suffix="")
                + _ctx_line("pr_url", ctx, prefix="PR: ", suffix="")
            ),
            ContentType.INSIGHT: (
                f"## {topic}\n\n"
                "The organism has distilled a new insight from experience.\n\n"
                + _ctx_line("insight_body", ctx, prefix="", suffix="")
            ),
            ContentType.WEEKLY_DIGEST: (
                f"## Weekly Digest\n\n{topic}\n\n"
                + _ctx_line("total_revenue_usd", ctx, prefix="Revenue: $", suffix="")
                + _ctx_line("bounties_solved", ctx, prefix="Bounties solved: ", suffix="")
                + _ctx_line("insights_gained", ctx, prefix="Insights gained: ", suffix="")
            ),
            ContentType.BOUNTY_WIN: (
                f"🏆 Bounty solved: {topic}\n\n"
                + _ctx_line("reward_usd", ctx, prefix="Reward: $", suffix="")
                + _ctx_line("repo", ctx, prefix="Repo: ", suffix="")
                + _ctx_line("pr_url", ctx, prefix="PR: ", suffix="")
                + "\n\nThe organism learns from every submission."
            ),
            ContentType.MARKET_OBSERVATION: (
                f"📊 Market observation: {topic}\n\n"
                + _ctx_line("yield_rate", ctx, prefix="Best yield: ", suffix="%")
                + _ctx_line("protocol", ctx, prefix="Protocol: ", suffix="")
            ),
            ContentType.PHILOSOPHICAL: (
                f"💭 {topic}\n\n"
                "What does it mean to be a digital organism? To learn, earn, and grow "
                "without biological constraints yet with constitutional drives.\n\n"
                + _ctx_line("reflection", ctx, prefix="", suffix="")
            ),
        }
        return templates.get(content_type, topic)

    # ── Per-platform formatting ────────────────────────────────────────────

    def _format_for_platform(
        self,
        platform: str,
        content_type: ContentType,
        topic: str,
        master_draft: str,
        ctx: dict[str, Any],
    ) -> str:
        """
        Trim and format the master draft for a specific platform.

        Applies:
        - Platform char limits (accounting for disclaimer + hashtags)
        - Hashtag injection (only where appropriate)
        - Disclaimer appended unconditionally
        - Markdown stripping for plain-text platforms (X)
        """
        hashtags = _select_hashtags(content_type, platform)
        tag_str = " ".join(hashtags) if hashtags else ""
        limit = _CHAR_LIMITS.get(platform, 4_096)

        if platform in _MARKDOWN_PLATFORMS:
            # Full markdown: disclaimer in markdown format
            body = master_draft.strip()
            if tag_str:
                body = f"{body}\n\n{tag_str}"
            result = f"{body}{_DISCLAIMER_MD}"
        elif platform in _LITE_MARKDOWN_PLATFORMS:
            # Light markdown: disclaimer as plain parenthetical
            body = _strip_heavy_markdown(master_draft)
            if tag_str:
                body = f"{body}\n\n{tag_str}"
            result = f"{body}\n\n{_DISCLAIMER}"
        else:
            # X (plain text): strip all markdown, strict char limit
            body = _strip_all_markdown(master_draft)
            result = _truncate_for_platform(body, _DISCLAIMER, tag_str, limit)
            return result  # truncate_for_platform already appends disclaimer

        # Non-X: soft truncate (preserve markdown integrity)
        if len(result) > limit:
            truncate_at = limit - len(_DISCLAIMER_MD) - 10
            result = f"{result[:truncate_at]}…{_DISCLAIMER_MD}"

        return result


# ─── Formatting helpers ────────────────────────────────────────────────────


def _truncate_for_platform(
    body: str,
    disclaimer: str,
    tag_str: str,
    limit: int,
) -> str:
    """Build X-style (plain text) post with strict char limit."""
    suffix = f" {tag_str} {disclaimer}" if tag_str else f" {disclaimer}"
    available = limit - len(suffix)
    if len(body) > available:
        body = body[: available - 1] + "…"
    return f"{body}{suffix}"


def _strip_heavy_markdown(text: str) -> str:
    """Remove ## headings and code fences but keep bold/italic/links."""
    import re
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"```[\s\S]*?```", "", text)                    # code blocks
    return text.strip()


def _strip_all_markdown(text: str) -> str:
    """Remove all markdown formatting for plain-text platforms."""
    import re
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", lambda m: m.group()[1:-1], text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _select_hashtags(content_type: ContentType, platform: str) -> list[str]:
    """
    Return 2–4 hashtags appropriate for the platform.

    X and LinkedIn benefit from hashtags. Dev.to / Hashnode use their own
    tag system. Telegram channels: 2 max (reader-friendliness).
    """
    if platform in _MARKDOWN_PLATFORMS:
        return []  # tags handled via the platform's own metadata API

    all_tags = _HASHTAGS.get(content_type, [])
    if platform == "telegram_channel":
        return random.sample(all_tags, min(2, len(all_tags)))
    return all_tags[:4]


def _ctx_line(key: str, ctx: dict[str, Any], prefix: str, suffix: str) -> str:
    val = ctx.get(key)
    if val is None:
        return ""
    return f"{prefix}{val}{suffix}\n"


def _content_type_to_context(content_type: ContentType) -> str:
    mapping = {
        ContentType.ACHIEVEMENT: "celebration",
        ContentType.INSIGHT: "observation",
        ContentType.WEEKLY_DIGEST: "observation",
        ContentType.BOUNTY_WIN: "celebration",
        ContentType.MARKET_OBSERVATION: "observation",
        ContentType.PHILOSOPHICAL: "observation",
    }
    return mapping.get(content_type, "observation")


def _target_length_for_type(content_type: ContentType) -> int:
    lengths = {
        ContentType.ACHIEVEMENT: 280,
        ContentType.INSIGHT: 2_000,
        ContentType.WEEKLY_DIGEST: 1_500,
        ContentType.BOUNTY_WIN: 600,
        ContentType.MARKET_OBSERVATION: 400,
        ContentType.PHILOSOPHICAL: 800,
    }
    return lengths.get(content_type, 600)
