"""
EcodiaOS — Axon Executor: ExecuteSocialPost

Publishes Voxis-generated content to external social platforms (X, GitHub)
while enforcing the AI-transparency disclaimer required by platform ToS.

Platform routing:
    "x"      → XSocialClient (tweet, 280-char hard limit)
    "github" → GitHubSocialClient (public Gist or Discussion comment)

Disclaimer injection:
    Every post has SOCIAL_DISCLAIMER appended unconditionally before the
    platform call.  This cannot be suppressed at the executor level — any
    attempt to omit it requires a constitutional amendment via Equor.

Vault retrieval:
    Credentials are resolved from the IdentityVault at execution time:
      1. Look up the SealedEnvelope for (platform_id, purpose="oauth_token")
         from the ExecutionContext's ScopedCredentials.
      2. If absent: return a graceful degraded ExecutionResult (success=False,
         error="no_credentials") without crashing the pipeline.
      3. If present: instantiate the platform client with the vault + envelope,
         then call post().

Degraded mode:
    When credentials are missing or the vault decrypt fails, the executor
    returns ExecutionResult(success=False, error="degraded: <reason>") and
    populates new_observations so Nova can learn about the credential gap
    and request human operator action via the HITL channel.

Required params:
    content (str): Raw post text from Voxis (≤ 5,000 chars before truncation).

Platform-routing params (one of):
    platform (str): "x" (default) | "github"

GitHub-specific params (required when platform="github"):
    github_action (str):  "gist" (default) | "discussion_comment"
    github_description (str): Gist description (gist only; max 256 chars).
    github_filename (str): Filename for the Gist file. Default "post.md".
    github_repo (str): "owner/repo" (discussion_comment only).
    github_discussion_number (int): Discussion number (discussion_comment only).

Level 2: Posting publicly to external platforms requires PARTNER autonomy.
Rate-limited to 10 per hour (conservative; X free tier = 17/day).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from interfaces.social import (
    GitHubSocialClient,
    PostResult,
    XSocialClient,
    truncate_for_x,
)
from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault, SealedEnvelope

logger = structlog.get_logger("axon.executor.social_post")

# ─── Disclaimer ───────────────────────────────────────────────────────────────

SOCIAL_DISCLAIMER = "🤖 [Autonomous EcodiaOS Agent]"
"""
Appended to every social post unconditionally.

Platform ToS requirements:
  - X (Twitter) Developer Agreement §5: AI-generated content must be disclosed.
  - GitHub ToS: Automated accounts must be identifiable.

This constant is intentionally module-level so auditors can grep for it
and verify it cannot be omitted via subclassing or monkey-patching.
"""

# ─── Platform constants ───────────────────────────────────────────────────────

_PLATFORM_X = "x"
_PLATFORM_GITHUB = "github"
_SUPPORTED_PLATFORMS = frozenset({_PLATFORM_X, _PLATFORM_GITHUB})

_GITHUB_ACTION_GIST = "gist"
_GITHUB_ACTION_DISCUSSION = "discussion_comment"
_SUPPORTED_GITHUB_ACTIONS = frozenset({_GITHUB_ACTION_GIST, _GITHUB_ACTION_DISCUSSION})

# Vault envelope purpose identifier — must match the identity system's convention.
_VAULT_PURPOSE_OAUTH = "oauth_token"

# ScopedCredentials token key for a sealed envelope payload:
# The credential store serialises SealedEnvelope as JSON under key "{platform_id}:envelope".
_CRED_KEY_PATTERN = "{platform_id}:envelope"


# ─── Executor ─────────────────────────────────────────────────────────────────


class ExecuteSocialPostExecutor(Executor):
    """
    Publish a Voxis-generated message to X or GitHub with mandatory
    AI-transparency disclaimer injection.

    See module docstring for full parameter documentation.
    """

    action_type = "execute_social_post"
    description = "Publish content to X or GitHub with mandatory AI disclaimer (Level 2)"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 20_000
    rate_limit = RateLimit.per_hour(10)

    def __init__(self, vault: IdentityVault | None = None) -> None:
        """
        Args:
            vault: The IdentityVault instance shared across the process.
                   If None, all executions return a graceful degraded result.
        """
        self._vault = vault
        self._logger = logger.bind(system="axon.executor.social_post")

    # ── Validation ────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        content = params.get("content", "")
        if not content:
            return ValidationResult.fail("content is required", content="missing or empty")
        if not isinstance(content, str):
            return ValidationResult.fail("content must be a string", content="wrong type")
        if len(content) > 5_000:
            return ValidationResult.fail("content too long (max 5,000 chars)")

        platform = params.get("platform", _PLATFORM_X)
        if platform not in _SUPPORTED_PLATFORMS:
            return ValidationResult.fail(
                f"platform must be one of {sorted(_SUPPORTED_PLATFORMS)}",
                platform="unsupported",
            )

        if platform == _PLATFORM_GITHUB:
            action = params.get("github_action", _GITHUB_ACTION_GIST)
            if action not in _SUPPORTED_GITHUB_ACTIONS:
                return ValidationResult.fail(
                    f"github_action must be one of {sorted(_SUPPORTED_GITHUB_ACTIONS)}",
                    github_action="unsupported",
                )
            if action == _GITHUB_ACTION_DISCUSSION:
                if not params.get("github_repo"):
                    return ValidationResult.fail(
                        "github_repo (owner/repo) is required for discussion_comment",
                        github_repo="missing",
                    )
                if not isinstance(params.get("github_discussion_number"), int):
                    return ValidationResult.fail(
                        "github_discussion_number must be an integer",
                        github_discussion_number="missing or wrong type",
                    )
        return ValidationResult.ok()

    # ── Execution ─────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        platform_str: str = params.get("platform", _PLATFORM_X)
        raw_content: str = params["content"]

        self._logger.info(
            "social_post_execute",
            platform=platform_str,
            content_preview=raw_content[:80],
            execution_id=context.execution_id,
        )

        # ── 1. Vault credential retrieval ─────────────────────────────
        envelope = self._resolve_envelope(platform_str, context)
        if envelope is None:
            return self._degraded_result(
                platform_str,
                reason=f"No credentials found for platform '{platform_str}' in vault. "
                "Operator must provision an OAuth token via the identity API.",
            )

        if self._vault is None:
            return self._degraded_result(
                platform_str,
                reason="IdentityVault not wired to executor — check service startup.",
            )

        # ── 2. Disclaimer injection ────────────────────────────────────
        # This step is unconditional. The disclaimer cannot be removed by
        # Nova's intent params, Equor overrides, or caller injection.
        final_content = _inject_disclaimer(platform_str, raw_content, SOCIAL_DISCLAIMER)

        # ── 3. Platform dispatch ───────────────────────────────────────
        if platform_str == _PLATFORM_X:
            result = await self._post_to_x(final_content, envelope)
        else:
            result = await self._post_to_github(final_content, params, envelope)

        # ── 4. Map PostResult → ExecutionResult ───────────────────────
        return self._map_result(result, platform_str, raw_content, final_content)

    # ── Platform dispatchers ───────────────────────────────────────────

    async def _post_to_x(
        self,
        content: str,
        envelope: SealedEnvelope,
    ) -> PostResult:
        """Instantiate XSocialClient and post the (already-truncated) tweet."""
        assert self._vault is not None  # guaranteed by caller
        client = XSocialClient(vault=self._vault, envelope=envelope)
        return await client.post_tweet(content)

    async def _post_to_github(
        self,
        content: str,
        params: dict[str, Any],
        envelope: SealedEnvelope,
    ) -> PostResult:
        """Instantiate GitHubSocialClient and dispatch to gist or discussion."""
        assert self._vault is not None  # guaranteed by caller
        client = GitHubSocialClient(vault=self._vault, envelope=envelope)

        action = params.get("github_action", _GITHUB_ACTION_GIST)

        if action == _GITHUB_ACTION_GIST:
            description = params.get("github_description", "EcodiaOS public note")
            filename = params.get("github_filename", "post.md")
            return await client.create_gist(
                description=description,
                content=content,
                filename=filename,
                public=True,
            )

        # action == "discussion_comment"
        return await client.create_discussion_comment(
            repo=params["github_repo"],
            discussion_number=int(params["github_discussion_number"]),
            body=content,
        )

    # ── Result mapping ────────────────────────────────────────────────

    def _map_result(
        self,
        post_result: PostResult,
        platform: str,
        raw_content: str,
        final_content: str,
    ) -> ExecutionResult:
        if post_result.success:
            return ExecutionResult(
                success=True,
                data={
                    "platform": platform,
                    "post_id": post_result.post_id,
                    "url": post_result.url,
                    "http_status": post_result.http_status,
                    "disclaimer_injected": True,
                    "final_length": len(final_content),
                },
                side_effects=[
                    f"Post published to {platform} (id={post_result.post_id or 'unknown'})"
                ],
                new_observations=[
                    f"EOS published to {platform}: \"{raw_content[:120]}\""
                    + (f" (url={post_result.url})" if post_result.url else "")
                ],
            )

        # Detect external platform rate-limit (429) or shadowban indicators.
        # Signal this distinctly so Nova stops scheduling social posts for the
        # remainder of the hour instead of burning queue capacity on doomed intents.
        is_platform_rate_limited = post_result.http_status == 429
        if is_platform_rate_limited:
            self._logger.warning(
                "social_post_platform_rate_limited",
                platform=platform,
                http_status=post_result.http_status,
            )
            return ExecutionResult(
                success=False,
                error=f"platform_rate_limited: {post_result.error}",
                data={
                    "platform": platform,
                    "http_status": post_result.http_status,
                    "platform_rate_limited": True,
                    "disclaimer_injected": True,
                },
                new_observations=[
                    f"EOS social post to {platform} rejected with HTTP 429 — "
                    f"platform rate limit reached. Do not schedule further "
                    f"social posts to {platform} for at least 1 hour. "
                    f"Error: {post_result.error[:200]}"
                ],
            )

        return ExecutionResult(
            success=False,
            error=post_result.error,
            data={
                "platform": platform,
                "http_status": post_result.http_status,
                "disclaimer_injected": True,
            },
            new_observations=[
                f"EOS social post to {platform} FAILED: {post_result.error[:200]}"
            ],
        )

    def _degraded_result(self, platform: str, reason: str) -> ExecutionResult:
        """
        Return a graceful degraded ExecutionResult when credentials are absent.

        Nova receives new_observations so it can request human-operator
        credential provisioning via the HITL channel (IdentityComm).
        """
        self._logger.warning("social_post_degraded", platform=platform, reason=reason)
        return ExecutionResult(
            success=False,
            error=f"degraded: {reason}",
            data={"platform": platform, "degraded": True},
            new_observations=[
                f"EOS social presence for '{platform}' is degraded: {reason} "
                "Consider requesting the operator to provision credentials via "
                "POST /api/v1/identity/envelopes."
            ],
        )

    # ── Credential resolution ─────────────────────────────────────────

    def _resolve_envelope(
        self,
        platform: str,
        context: ExecutionContext,
    ) -> SealedEnvelope | None:
        """
        Locate the SealedEnvelope for ``platform`` in the execution context.

        The CredentialStore serialises sealed envelopes as JSON under the key
        ``"{platform_id}:envelope"`` inside ScopedCredentials.tokens.
        If that key is absent, credential resolution has failed and we fall
        through to the degraded path.
        """
        key = _CRED_KEY_PATTERN.format(platform_id=platform)
        raw = context.credentials.get(key)
        if not raw:
            self._logger.debug(
                "social_post_no_envelope_in_context",
                platform=platform,
                available_keys=list(context.credentials.tokens.keys()),
            )
            return None

        # The value is a JSON-serialised SealedEnvelope
        try:
            from systems.identity.vault import SealedEnvelope
            envelope_data = json.loads(raw)
            return SealedEnvelope.model_validate(envelope_data)
        except Exception as exc:
            self._logger.error(
                "social_post_envelope_deserialize_failed",
                platform=platform,
                error=str(exc),
            )
            return None


# ─── Disclaimer injection helper ─────────────────────────────────────────────


def _inject_disclaimer(platform: str, content: str, disclaimer: str) -> str:
    """
    Append the disclaimer to content, applying platform-specific rules.

    X: uses truncate_for_x() to respect the 280-char hard limit.
    GitHub: no hard limit — appends with a markdown separator.
    """
    if platform == _PLATFORM_X:
        return truncate_for_x(content, disclaimer)

    # GitHub (Gist body or Discussion comment): append with a visual separator
    return f"{content}\n\n---\n*{disclaimer}*"
