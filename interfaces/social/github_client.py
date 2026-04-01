"""
EcodiaOS - Social Interface: GitHub Social Client

Posts public-facing content to GitHub on behalf of the organism.

Supported operations:
    - create_gist: Publish a public Gist (text content as a markdown file).
    - create_discussion_comment: Post a comment on a repository discussion.

These are distinct from the bounty-hunting GitHubClient in ecodiaos/clients/github.py,
which is scoped to the search API. This client is focused on *social publishing* -
creating visible artefacts that build the organism's reputation.

Auth: GitHub personal access token (PAT) or fine-grained token.
The vault must hold a token envelope with purpose="oauth_token" containing:
    {
        "access_token": "<github pat or oauth token>"
    }

Rate limits: GitHub REST API allows 5,000 requests/hour for authenticated users.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from interfaces.social.types import PostResult, SocialPlatform

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault, SealedEnvelope

logger = structlog.get_logger("interfaces.social.github")

_GITHUB_API = "https://api.github.com"
_GISTS_PATH = "/gists"


class GitHubSocialClient:
    """
    Async client for GitHub social publishing via the REST API v3.

    Intended to be constructed per-execution; credentials are resolved
    from the vault at construction time - never passed as plaintext.

    Usage::

        client = GitHubSocialClient(vault=vault, envelope=sealed_envelope)
        result = await client.create_gist(
            description="EcodiaOS observation log",
            content="...",
            filename="observation.md",
        )
    """

    def __init__(
        self,
        vault: IdentityVault,
        envelope: SealedEnvelope,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._vault = vault
        self._envelope = envelope
        self._http = http_client or httpx.AsyncClient(
            base_url=_GITHUB_API,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15.0,
        )
        self._logger = logger.bind(platform="github")

    async def create_gist(
        self,
        description: str,
        content: str,
        filename: str = "post.md",
        public: bool = True,
    ) -> PostResult:
        """
        Create a public Gist with ``content`` as the sole file.

        Args:
            description: Short description shown in the Gist header.
            content: File content (markdown recommended for readability).
            filename: Gist filename (default ``post.md``).
            public: Whether the Gist is publicly discoverable.

        Returns:
            PostResult - never raises.
        """
        token = self._load_token()
        if token is None:
            return PostResult.fail(
                SocialPlatform.GITHUB,
                error="Vault envelope missing access_token for GitHub",
            )

        payload: dict[str, Any] = {
            "description": description,
            "public": public,
            "files": {filename: {"content": content}},
        }

        try:
            resp = await self._http.post(
                _GISTS_PATH,
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            self._logger.error("github_gist_transport_error", error=str(exc))
            return PostResult.fail(SocialPlatform.GITHUB, error=f"Transport error: {exc}")

        if resp.status_code not in (200, 201):
            self._logger.error(
                "github_gist_api_error",
                status=resp.status_code,
                body_preview=resp.text[:200],
            )
            return PostResult.fail(
                SocialPlatform.GITHUB,
                error=f"GitHub API error HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )

        resp_data: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            resp_data = resp.json()

        gist_id = resp_data.get("id", "")
        gist_url = resp_data.get("html_url", "")

        self._logger.info("github_gist_created", gist_id=gist_id, url=gist_url)
        return PostResult.ok(
            platform=SocialPlatform.GITHUB,
            post_id=str(gist_id),
            url=gist_url,
            http_status=resp.status_code,
            raw_response=resp_data,
        )

    async def create_discussion_comment(
        self,
        repo: str,
        discussion_number: int,
        body: str,
    ) -> PostResult:
        """
        Post a comment on a GitHub Discussions thread.

        Args:
            repo: ``owner/repo`` string.
            discussion_number: The discussion number to comment on.
            body: Markdown comment body.

        Returns:
            PostResult - never raises.

        Note: GitHub Discussions comments require the GraphQL API.
        This method uses the GraphQL endpoint at ``/graphql``.
        """
        token = self._load_token()
        if token is None:
            return PostResult.fail(
                SocialPlatform.GITHUB,
                error="Vault envelope missing access_token for GitHub",
            )

        # Step 1: resolve the discussion's GraphQL node_id
        owner, repo_name = repo.split("/", 1) if "/" in repo else ("", repo)
        discussion_id = await self._resolve_discussion_node_id(
            token, owner, repo_name, discussion_number
        )
        if discussion_id is None:
            return PostResult.fail(
                SocialPlatform.GITHUB,
                error=f"Could not resolve discussion #{discussion_number} in {repo}",
            )

        # Step 2: add the comment
        mutation = """
        mutation AddDiscussionComment($discussionId: ID!, $body: String!) {
            addDiscussionComment(input: {discussionId: $discussionId, body: $body}) {
                comment {
                    id
                    url
                }
            }
        }
        """
        graphql_payload = {
            "query": mutation,
            "variables": {"discussionId": discussion_id, "body": body},
        }

        try:
            resp = await self._http.post(
                "/graphql",
                headers={"Authorization": f"Bearer {token}"},
                json=graphql_payload,
            )
        except httpx.HTTPError as exc:
            self._logger.error("github_discussion_transport_error", error=str(exc))
            return PostResult.fail(SocialPlatform.GITHUB, error=f"Transport error: {exc}")

        if resp.status_code != 200:
            return PostResult.fail(
                SocialPlatform.GITHUB,
                error=f"GitHub GraphQL error HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )

        resp_data: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            resp_data = resp.json()

        errors = resp_data.get("errors")
        if errors:
            err_msg = errors[0].get("message", str(errors[0])) if errors else "unknown"
            self._logger.error("github_discussion_graphql_error", errors=errors)
            return PostResult.fail(
                SocialPlatform.GITHUB,
                error=f"GraphQL error: {err_msg}",
                http_status=resp.status_code,
            )

        comment_data: dict[str, Any] = (
            resp_data.get("data", {})
            .get("addDiscussionComment", {})
            .get("comment", {})
        )
        comment_id = comment_data.get("id", "")
        comment_url = comment_data.get("url", "")

        self._logger.info("github_discussion_comment_posted", comment_id=comment_id)
        return PostResult.ok(
            platform=SocialPlatform.GITHUB,
            post_id=str(comment_id),
            url=comment_url,
            http_status=resp.status_code,
            raw_response=resp_data,
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _load_token(self) -> str | None:
        """Decrypt the envelope and return the GitHub access token, or None on failure."""
        try:
            token_data = self._vault.decrypt_token_json(self._envelope)
        except Exception as exc:
            self._logger.error("github_vault_decrypt_failed", error=str(exc))
            return None

        token = token_data.get("access_token", "")
        if not token:
            self._logger.warning("github_token_missing_in_envelope")
            return None
        return token

    async def _resolve_discussion_node_id(
        self,
        token: str,
        owner: str,
        repo_name: str,
        discussion_number: int,
    ) -> str | None:
        """Resolve a repository discussion's GraphQL node ID."""
        query = """
        query GetDiscussionId($owner: String!, $repo: String!, $number: Int!) {
            repository(owner: $owner, name: $repo) {
                discussion(number: $number) {
                    id
                }
            }
        }
        """
        try:
            resp = await self._http.post(
                "/graphql",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "query": query,
                    "variables": {
                        "owner": owner,
                        "repo": repo_name,
                        "number": discussion_number,
                    },
                },
            )
            data = resp.json()
            return data.get("data", {}).get("repository", {}).get("discussion", {}).get("id")
        except Exception as exc:
            self._logger.warning("github_resolve_discussion_failed", error=str(exc))
            return None
