"""
EcodiaOS - Social Interface: Dev.to Client

Publishes articles to Dev.to (Forem) using their REST API.

Auth: API key (no OAuth needed). The vault must hold a token envelope with
purpose="oauth_token" containing:
    {
        "access_token": "<dev.to API key>"
    }

Alternatively the key can be sourced from ECODIAOS_DEVTO_API_KEY env var as
a fallback when no vault envelope is provisioned.

API:
    POST https://dev.to/api/articles   - create article
    PUT  https://dev.to/api/articles/{id} - update article

Char limits: No enforced char limit (markdown body). Titles: 128 chars typical.

Rate limits: ~10 requests/min per API key. Enforced at the executor level.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from interfaces.social.types import PostResult, SocialPlatform

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault, SealedEnvelope

logger = structlog.get_logger("interfaces.social.devto")

_DEVTO_API_BASE = "https://dev.to/api"
_ARTICLES_URL = f"{_DEVTO_API_BASE}/articles"

_MAX_TAGS = 4  # Dev.to allows max 4 tags per article


class DevToClient:
    """
    Thin async HTTP client for the Dev.to / Forem Articles API.

    Credentials resolved from vault first, ECODIAOS_DEVTO_API_KEY env var second.
    """

    def __init__(
        self,
        vault: IdentityVault | None = None,
        envelope: SealedEnvelope | None = None,
    ) -> None:
        self._vault = vault
        self._envelope = envelope
        self._logger = logger.bind(client="devto")

    # ── Public API ─────────────────────────────────────────────────────────

    async def create_article(
        self,
        title: str,
        body_markdown: str,
        tags: list[str] | None = None,
        published: bool = True,
        series: str | None = None,
        canonical_url: str | None = None,
    ) -> PostResult:
        """
        Create a new Dev.to article.

        Args:
            title: Article title (keep under 128 chars for best UX).
            body_markdown: Full markdown body including the AI disclaimer.
            tags: Up to 4 tag strings (e.g. ["ai", "opensource", "defi"]).
            published: True → immediately public; False → draft.
            series: Optional series name to group related articles.
            canonical_url: Optional canonical URL if cross-posting.

        Returns:
            PostResult - never raises.
        """
        api_key = self._load_api_key()
        if not api_key:
            return PostResult.fail(SocialPlatform.DEVTO, "no_credentials")

        article_body: dict[str, Any] = {
            "title": title,
            "body_markdown": body_markdown,
            "published": published,
        }
        if tags:
            article_body["tags"] = tags[:_MAX_TAGS]
        if series:
            article_body["series"] = series
        if canonical_url:
            article_body["canonical_url"] = canonical_url

        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    _ARTICLES_URL,
                    json={"article": article_body},
                    headers=headers,
                )

            if resp.status_code in (200, 201):
                data = _safe_json(resp)
                article_id = str(data.get("id", ""))
                article_url = data.get("url", "")
                self._logger.info("devto_article_created", id=article_id, url=article_url)
                return PostResult.ok(
                    platform=SocialPlatform.DEVTO,
                    post_id=article_id,
                    url=article_url,
                    http_status=resp.status_code,
                    raw_response=data,
                )

            self._logger.warning(
                "devto_article_failed",
                status=resp.status_code,
                body=resp.text[:300],
            )
            return PostResult.fail(
                SocialPlatform.DEVTO,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )

        except Exception as exc:
            self._logger.error("devto_article_exception", error=str(exc))
            return PostResult.fail(SocialPlatform.DEVTO, error=str(exc))

    # ── Internal helpers ───────────────────────────────────────────────────

    def _load_api_key(self) -> str:
        """
        Resolve API key: vault envelope → env var fallback.

        Returns:
            API key string, or empty string if unavailable.
        """
        if self._vault is not None and self._envelope is not None:
            with contextlib.suppress(Exception):
                token_set = self._vault.decrypt_token_set(self._envelope)
                key = token_set.get("access_token", "")
                if key:
                    return key

        return os.getenv("ECODIAOS_DEVTO_API_KEY", "")


# ── Helpers ────────────────────────────────────────────────────────────────


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    with contextlib.suppress(Exception):
        return resp.json()  # type: ignore[return-value]
    return {}
