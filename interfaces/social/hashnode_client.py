"""
EcodiaOS - Social Interface: Hashnode Client

Publishes blog posts to Hashnode via their GraphQL API.

Auth: Personal Access Token. The vault must hold a token envelope with
purpose="oauth_token" containing:
    {
        "access_token": "<hashnode PAT>",
        "publication_id": "<publicationId>"   # optional; falls back to env var
    }

Env var fallbacks:
    ECODIAOS_HASHNODE_TOKEN          - Personal Access Token
    ECODIAOS_HASHNODE_PUBLICATION_ID - Publication (blog) ID

API:
    POST https://gql.hashnode.com/   - GraphQL endpoint
    Mutation: publishPost (Hashnode API v3)

Rate limits: Generous for personal accounts. Enforced at executor level.
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

logger = structlog.get_logger("interfaces.social.hashnode")

_HASHNODE_GQL_URL = "https://gql.hashnode.com/"

_PUBLISH_POST_MUTATION = """
mutation PublishPost($input: PublishPostInput!) {
  publishPost(input: $input) {
    post {
      id
      url
      slug
      title
    }
  }
}
"""


class HashnodeClient:
    """
    Thin async GraphQL client for the Hashnode Publications API (v3).

    Credentials resolved from vault first, env vars second.
    """

    def __init__(
        self,
        vault: IdentityVault | None = None,
        envelope: SealedEnvelope | None = None,
    ) -> None:
        self._vault = vault
        self._envelope = envelope
        self._logger = logger.bind(client="hashnode")

    # ── Public API ─────────────────────────────────────────────────────────

    async def publish_post(
        self,
        title: str,
        body_markdown: str,
        tags: list[str] | None = None,
        subtitle: str | None = None,
        canonical_url: str | None = None,
        cover_image_url: str | None = None,
    ) -> PostResult:
        """
        Publish a new post to the configured Hashnode publication.

        Args:
            title: Post title.
            body_markdown: Full markdown content (including AI disclaimer).
            tags: Tag slugs (e.g. ["defi", "ai", "opensource"]).
            subtitle: Optional subtitle / description.
            canonical_url: Optional canonical URL for cross-posting.
            cover_image_url: Optional cover image URL.

        Returns:
            PostResult - never raises.
        """
        token, publication_id = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.HASHNODE, "no_credentials")
        if not publication_id:
            return PostResult.fail(SocialPlatform.HASHNODE, "no_publication_id")

        tag_inputs = [{"slug": t, "name": t} for t in (tags or [])]

        input_obj: dict[str, Any] = {
            "title": title,
            "contentMarkdown": body_markdown,
            "publicationId": publication_id,
            "tags": tag_inputs,
        }
        if subtitle:
            input_obj["subtitle"] = subtitle
        if canonical_url:
            input_obj["originalArticleURL"] = canonical_url
        if cover_image_url:
            input_obj["coverImageOptions"] = {"coverImageURL": cover_image_url}

        return await self._gql_request(
            query=_PUBLISH_POST_MUTATION,
            variables={"input": input_obj},
            token=token,
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _gql_request(
        self,
        query: str,
        variables: dict[str, Any],
        token: str,
    ) -> PostResult:
        """Execute a GraphQL mutation and extract the publishPost result."""
        headers = {
            "Authorization": token,
            "Content-Type": "application/json",
        }
        payload = {"query": query, "variables": variables}

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(_HASHNODE_GQL_URL, json=payload, headers=headers)

            data = _safe_json(resp)

            # GraphQL errors travel as 200 with an "errors" key.
            gql_errors = data.get("errors")
            if gql_errors:
                first_err = gql_errors[0].get("message", "unknown graphql error") if gql_errors else "unknown"
                self._logger.warning("hashnode_gql_error", error=first_err)
                return PostResult.fail(
                    SocialPlatform.HASHNODE,
                    error=f"GraphQL error: {first_err}",
                    http_status=resp.status_code,
                )

            post_data: dict[str, Any] = (
                data.get("data", {})
                .get("publishPost", {})
                .get("post", {})
            )
            post_id = str(post_data.get("id", ""))
            post_url = post_data.get("url", "")

            if resp.status_code in (200, 201) and post_id:
                self._logger.info("hashnode_post_published", id=post_id, url=post_url)
                return PostResult.ok(
                    platform=SocialPlatform.HASHNODE,
                    post_id=post_id,
                    url=post_url,
                    http_status=resp.status_code,
                    raw_response=post_data,
                )

            self._logger.warning(
                "hashnode_post_failed",
                status=resp.status_code,
                body=resp.text[:300],
            )
            return PostResult.fail(
                SocialPlatform.HASHNODE,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )

        except Exception as exc:
            self._logger.error("hashnode_post_exception", error=str(exc))
            return PostResult.fail(SocialPlatform.HASHNODE, error=str(exc))

    def _load_credentials(self) -> tuple[str, str]:
        """
        Resolve (token, publication_id): vault envelope → env vars.

        Returns:
            (token, publication_id) - both empty strings on failure.
        """
        token = ""
        publication_id = ""

        if self._vault is not None and self._envelope is not None:
            with contextlib.suppress(Exception):
                token_set = self._vault.decrypt_token_set(self._envelope)
                token = token_set.get("access_token", "")
                publication_id = token_set.get("publication_id", "")

        if not token:
            token = os.getenv("ECODIAOS_HASHNODE_TOKEN", "")
        if not publication_id:
            publication_id = os.getenv("ECODIAOS_HASHNODE_PUBLICATION_ID", "")

        return token, publication_id


# ── Helpers ────────────────────────────────────────────────────────────────


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    with contextlib.suppress(Exception):
        return resp.json()  # type: ignore[return-value]
    return {}
