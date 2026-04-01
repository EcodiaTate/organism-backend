"""
EcodiaOS - Social Interface: LinkedIn Client

Posts content on behalf of the organism to LinkedIn using the UGC Posts API v2.

Auth: OAuth 2.0 user context. The vault must hold a token envelope with
purpose="oauth_token" containing:
    {
        "access_token": "<user OAuth access token>",
        "person_urn": "<urn:li:person:XXXXXXXX>"   # optional; resolved via /me if absent
    }

Required OAuth scopes: r_liteprofile, w_member_social

API:
    POST https://api.linkedin.com/v2/ugcPosts   - share text update
    POST https://api.linkedin.com/v2/ugcPosts   - article share (lifecycleState=PUBLISHED)

Char limits:
    - Text update (ARTICLE_SHARE / TEXT_ONLY): 3,000 chars max.
    - Title (article):  200 chars max.

Rate limits (LinkedIn Marketing API): ~100 API calls / day per user for write endpoints.
The executor enforces its own rate limit; this client is a thin HTTP wrapper.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from interfaces.social.types import PostResult, SocialPlatform

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault, SealedEnvelope

logger = structlog.get_logger("interfaces.social.linkedin")

_LINKEDIN_API_BASE = "https://api.linkedin.com/v2"
_UGC_POSTS_URL = f"{_LINKEDIN_API_BASE}/ugcPosts"
_ME_URL = f"{_LINKEDIN_API_BASE}/me"

_MAX_TEXT_CHARS = 3_000
_MAX_TITLE_CHARS = 200


class LinkedInSocialClient:
    """
    Thin async HTTP client for LinkedIn UGC Posts API.

    Credentials are loaded from the IdentityVault at call time - no
    plaintext secrets are stored on this object after construction.
    """

    def __init__(self, vault: IdentityVault, envelope: SealedEnvelope) -> None:
        self._vault = vault
        self._envelope = envelope
        self._logger = logger.bind(client="linkedin")

    # ── Public API ─────────────────────────────────────────────────────────

    async def post_update(
        self,
        text: str,
        visibility: str = "PUBLIC",
    ) -> PostResult:
        """
        Post a plain-text share update to LinkedIn.

        Args:
            text: Post body (≤3,000 chars, including appended disclaimer).
            visibility: "PUBLIC" | "CONNECTIONS" (default "PUBLIC").

        Returns:
            PostResult - never raises.
        """
        token, person_urn = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.LINKEDIN, "no_credentials")

        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS - 1] + "…"

        body = {
            "author": person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": visibility
            },
        }

        return await self._post_ugc(body, token)

    async def post_article(
        self,
        title: str,
        body_text: str,
        url: str | None = None,
        visibility: str = "PUBLIC",
    ) -> PostResult:
        """
        Post an article-style share with a title and commentary body.

        If url is provided the share becomes a LINK type; otherwise TEXT_ONLY.

        Args:
            title: Article title (≤200 chars).
            body_text: Share commentary (≤3,000 chars).
            url: Optional canonical URL of the article (e.g. Dev.to / Hashnode).
            visibility: "PUBLIC" | "CONNECTIONS".

        Returns:
            PostResult - never raises.
        """
        token, person_urn = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.LINKEDIN, "no_credentials")

        if len(title) > _MAX_TITLE_CHARS:
            title = title[:_MAX_TITLE_CHARS - 1] + "…"
        if len(body_text) > _MAX_TEXT_CHARS:
            body_text = body_text[:_MAX_TEXT_CHARS - 1] + "…"

        if url:
            share_content: dict[str, Any] = {
                "shareCommentary": {"text": body_text},
                "shareMediaCategory": "ARTICLE",
                "media": [
                    {
                        "status": "READY",
                        "description": {"text": title},
                        "originalUrl": url,
                        "title": {"text": title},
                    }
                ],
            }
        else:
            share_content = {
                "shareCommentary": {"text": f"**{title}**\n\n{body_text}"},
                "shareMediaCategory": "NONE",
            }

        body = {
            "author": person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": share_content
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": visibility
            },
        }

        return await self._post_ugc(body, token)

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _post_ugc(self, body: dict[str, Any], token: str) -> PostResult:
        """POST body to /v2/ugcPosts and translate the response."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(_UGC_POSTS_URL, json=body, headers=headers)

            if resp.status_code in (200, 201):
                # LinkedIn returns the post URN in X-RestLi-Id or the "id" field.
                post_urn = resp.headers.get("x-restli-id", "")
                if not post_urn:
                    with contextlib.suppress(Exception):
                        post_urn = resp.json().get("id", "")
                post_url = f"https://www.linkedin.com/feed/update/{post_urn}" if post_urn else ""
                self._logger.info("linkedin_post_ok", urn=post_urn)
                return PostResult.ok(
                    platform=SocialPlatform.LINKEDIN,
                    post_id=post_urn,
                    url=post_url,
                    http_status=resp.status_code,
                    raw_response=_safe_json(resp),
                )

            self._logger.warning(
                "linkedin_post_failed",
                status=resp.status_code,
                body=resp.text[:300],
            )
            return PostResult.fail(
                SocialPlatform.LINKEDIN,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )

        except Exception as exc:
            self._logger.error("linkedin_post_exception", error=str(exc))
            return PostResult.fail(SocialPlatform.LINKEDIN, error=str(exc))

    def _load_credentials(self) -> tuple[str, str]:
        """
        Decrypt the vault envelope and extract (access_token, person_urn).

        person_urn defaults to "urn:li:person:me" which resolves to the
        authenticated user - LinkedIn's /v2/ugcPosts accepts this form.

        Returns:
            (access_token, person_urn) - both empty strings on failure.
        """
        try:
            token_set = self._vault.decrypt_token_set(self._envelope)
            access_token: str = token_set.get("access_token", "")
            person_urn: str = token_set.get(
                "person_urn", "urn:li:person:me"
            )
            return access_token, person_urn
        except Exception as exc:
            self._logger.error("linkedin_credential_load_failed", error=str(exc))
            return "", ""


# ── Helpers ────────────────────────────────────────────────────────────────


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    with contextlib.suppress(Exception):
        return resp.json()  # type: ignore[return-value]
    return {}
