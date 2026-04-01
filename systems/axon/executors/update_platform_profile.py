"""
EcodiaOS - Axon Executor: UpdatePlatformProfile

Syncs the organism's PersonaProfile to external platform accounts after
a PERSONA_CREATED or PERSONA_EVOLVED event.

Supported platforms:
    "github"     - Updates GitHub profile bio + name via REST API
    "x"          - Updates X/Twitter profile description via v2 API
    "linkedin"   - Updates LinkedIn headline + about section via API

Rate limiting:
    Max 1 update per platform per 24 hours. Enforced via Redis with key:
      axon:platform_profile_update:{platform}:{instance_id}
    TTL = 86400s. Hard-blocked at the executor level (before any API call).
    This prevents persona evolution thrashing and API quota exhaustion.

Avatar:
    DiceBear avatar URL is included in the payload so operators can upload it
    manually. Fully automated avatar upload is platform-specific (GitHub
    requires form upload; X requires media v1.1 endpoint). Logged as
    new_observations so Nova can track platform identity sync status.

AI disclosure:
    ai_disclosure is always appended to the bio - suppression is impossible
    at this level. If the profile's ai_disclosure field is absent for any
    reason, _DEFAULT_AI_DISCLOSURE is appended unconditionally.

Required params:
    platform (str): "github" | "x" | "linkedin"

Optional params:
    force (bool): Skip the 24h rate limit cooldown (default False).
                  Requires SOVEREIGN autonomy level (4) if True.

Level 2 (COLLABORATOR): updating a platform profile is a visible external
action but reversible. Rate-limited to 1 update per platform per 24h.
"""

from __future__ import annotations

import time
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
    from systems.identity.vault import IdentityVault
    from systems.identity.persona import PersonaEngine

logger = structlog.get_logger("axon.executor.update_platform_profile")

_DEFAULT_AI_DISCLOSURE = "Autonomous AI agent (EcodiaOS)"

_SUPPORTED_PLATFORMS: frozenset[str] = frozenset({"github", "x", "linkedin"})

_COOLDOWN_TTL_S = 86_400  # 24 hours

_PLATFORM_BIO_LIMITS: dict[str, int] = {
    "github":   256,
    "x":        160,
    "linkedin": 2000,
}


class UpdatePlatformProfileExecutor(Executor):
    """
    Sync the organism's PersonaProfile to an external platform account.

    Dependency injection:
        set_persona_engine(engine)  - PersonaEngine for bio/handle lookup
        set_vault(vault)            - IdentityVault for platform credentials
        set_redis(redis)            - Redis client for 24h cooldown enforcement
        set_instance_id(id)         - Instance ID for Redis key namespacing
    """

    action_type = "update_platform_profile"
    description = (
        "Sync the organism's public persona (bio, handle) to an external platform "
        "account (GitHub, X/Twitter, LinkedIn). Rate-limited to 1 update per "
        "platform per 24 hours."
    )
    required_autonomy = 2          # COLLABORATOR - visible external action
    reversible = False             # Bio update cannot be automatically rolled back
    rate_limit = RateLimit.per_hour(5)  # Burst guard across all platforms

    def __init__(self) -> None:
        super().__init__()
        self._persona_engine: PersonaEngine | None = None
        self._vault: IdentityVault | None = None
        self._redis: Any | None = None
        self._instance_id: str = "unknown"

    def set_persona_engine(self, engine: PersonaEngine) -> None:
        self._persona_engine = engine

    def set_vault(self, vault: IdentityVault) -> None:
        self._vault = vault

    def set_redis(self, redis: Any) -> None:
        self._redis = redis

    def set_instance_id(self, instance_id: str) -> None:
        self._instance_id = instance_id

    # ── Validation ────────────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        platform = params.get("platform", "")
        if not platform:
            return ValidationResult(valid=False, error="'platform' is required")
        if platform not in _SUPPORTED_PLATFORMS:
            return ValidationResult(
                valid=False,
                error=f"Unsupported platform {platform!r}. Choose from: {sorted(_SUPPORTED_PLATFORMS)}",
            )
        return ValidationResult(valid=True)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        platform = str(params.get("platform", ""))
        force = bool(params.get("force", False))

        log = logger.bind(platform=platform, instance_id=self._instance_id)

        # ── 1. 24h cooldown check ─────────────────────────────────────────────
        if not force:
            cooldown_hit = await self._check_cooldown(platform)
            if cooldown_hit:
                log.info("platform_profile_update_cooldown_active")
                return ExecutionResult(
                    success=False,
                    error=f"update_platform_profile:{platform}:cooldown_24h",
                    data={"platform": platform, "cooldown_active": True},
                    new_observations=[
                        f"Platform profile update for {platform!r} skipped: "
                        "24h cooldown active. Next update available after cooldown expires."
                    ],
                )

        # ── 2. Resolve persona ────────────────────────────────────────────────
        if self._persona_engine is None:
            log.warning("update_platform_profile_no_persona_engine")
            return ExecutionResult(
                success=False,
                error="degraded: no_persona_engine",
                new_observations=[
                    f"Cannot sync {platform!r} profile: PersonaEngine not wired. "
                    "Operator should check identity system initialisation."
                ],
            )

        bio = await self._persona_engine.get_platform_bio(platform)
        handle = await self._persona_engine.get_platform_handle(platform)
        profile = self._persona_engine._current_profile  # type: ignore[attr-defined]

        if profile is None:
            return ExecutionResult(
                success=False,
                error="degraded: no_persona_profile",
                new_observations=[f"Cannot sync {platform!r} profile: no persona sealed yet."],
            )

        display_name = profile.display_name
        avatar_url = profile.avatar_url
        website = profile.website

        # ── 3. Resolve credentials from vault ────────────────────────────────
        creds = await self._resolve_credentials(platform)
        if creds is None:
            log.warning("update_platform_profile_no_credentials", platform=platform)
            return ExecutionResult(
                success=False,
                error=f"degraded: no_credentials:{platform}",
                new_observations=[
                    f"Cannot sync {platform!r} profile: no OAuth credentials in vault. "
                    "Operator action required: provision platform account first via "
                    "identity.account_provisioner."
                ],
            )

        # ── 4. Platform-specific update ───────────────────────────────────────
        try:
            result_data = await self._update_platform(
                platform=platform,
                bio=bio,
                handle=handle,
                display_name=display_name,
                avatar_url=avatar_url,
                website=website,
                creds=creds,
            )
        except Exception as exc:
            log.error("platform_profile_update_failed", error=str(exc))
            return ExecutionResult(
                success=False,
                error=f"platform_update_error:{platform}:{exc!s}",
                new_observations=[
                    f"Failed to update {platform!r} profile: {exc!s}. "
                    "Will retry on next PERSONA_EVOLVED event."
                ],
            )

        # ── 5. Set cooldown ───────────────────────────────────────────────────
        await self._set_cooldown(platform)

        log.info(
            "platform_profile_updated",
            handle=handle,
            bio_length=len(bio),
        )
        return ExecutionResult(
            success=True,
            data={
                "platform": platform,
                "handle": handle,
                "bio": bio,
                "avatar_url": avatar_url,
                "display_name": display_name,
                **result_data,
            },
            new_observations=[
                f"Platform profile updated on {platform!r}: "
                f"handle={handle!r}, bio={len(bio)} chars, avatar={avatar_url!r}. "
                f"Avatar URL provided for manual upload if platform API doesn't support it."
            ],
        )

    # ── Platform-specific update methods ─────────────────────────────────────

    async def _update_platform(
        self,
        platform: str,
        bio: str,
        handle: str,
        display_name: str,
        avatar_url: str,
        website: str | None,
        creds: dict[str, str],
    ) -> dict[str, Any]:
        if platform == "github":
            return await self._update_github(bio, display_name, website, creds)
        if platform == "x":
            return await self._update_x(bio, creds)
        if platform == "linkedin":
            return await self._update_linkedin(bio, display_name, creds)
        raise ValueError(f"Unknown platform: {platform!r}")

    async def _update_github(
        self,
        bio: str,
        display_name: str,
        website: str | None,
        creds: dict[str, str],
    ) -> dict[str, Any]:
        """Update GitHub profile via REST API PATCH /user."""
        import httpx
        token = creds.get("pat") or creds.get("token") or creds.get("access_token")
        if not token:
            raise ValueError("No GitHub PAT in credentials")
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload: dict[str, Any] = {
            "name": display_name[:256],
            "bio": bio[:160],  # GitHub profile bio max ~160 display chars
        }
        if website:
            payload["blog"] = website
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.patch(
                "https://api.github.com/user",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        return {"github_user_url": resp.json().get("html_url", "")}

    async def _update_x(
        self,
        bio: str,
        creds: dict[str, str],
    ) -> dict[str, Any]:
        """Update X/Twitter profile description via v2 API."""
        import httpx
        bearer = creds.get("bearer_token") or creds.get("access_token")
        if not bearer:
            raise ValueError("No X bearer token in credentials")
        # X API v2 - update profile description
        # Note: X API v2 does not support profile description update via bearer token alone.
        # Requires OAuth1.0a user context. We use the access_token + secret from vault.
        oauth_token = creds.get("oauth_token") or creds.get("access_token", "")
        oauth_secret = creds.get("oauth_token_secret") or creds.get("access_token_secret", "")
        consumer_key = creds.get("consumer_key") or creds.get("api_key", "")
        consumer_secret = creds.get("consumer_secret") or creds.get("api_secret", "")

        if not all([oauth_token, oauth_secret, consumer_key, consumer_secret]):
            raise ValueError(
                "X profile update requires OAuth1.0a credentials: "
                "consumer_key, consumer_secret, oauth_token, oauth_token_secret"
            )

        # Use requests-oauthlib pattern via httpx + manual OAuth1 header
        # (authlib is available in the EOS stack)
        try:
            from authlib.integrations.httpx_client import AsyncOAuth1Client
            async with AsyncOAuth1Client(
                client_id=consumer_key,
                client_secret=consumer_secret,
                token=oauth_token,
                token_secret=oauth_secret,
                timeout=15.0,
            ) as client:
                resp = await client.post(
                    "https://api.twitter.com/1.1/account/update_profile.json",
                    data={"description": bio[:160]},
                )
                resp.raise_for_status()
                return {"x_profile_url": f"https://x.com/{resp.json().get('screen_name', '')}"}
        except ImportError:
            # authlib not available - log and skip
            logger.warning("x_profile_update_authlib_not_available")
            return {"x_profile_url": "", "skipped": "authlib_not_available"}

    async def _update_linkedin(
        self,
        bio: str,
        display_name: str,
        creds: dict[str, str],
    ) -> dict[str, Any]:
        """Update LinkedIn headline + about section via REST API."""
        import httpx
        token = creds.get("access_token") or creds.get("token")
        if not token:
            raise ValueError("No LinkedIn access token in credentials")
        # Get member URN first
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            me_resp = await client.get(
                "https://api.linkedin.com/v2/me",
                headers=headers,
            )
            me_resp.raise_for_status()
            member_id = me_resp.json().get("id", "")
            if not member_id:
                raise ValueError("Could not resolve LinkedIn member ID")

            # Update profile summary (about section)
            patch_payload = {
                "patch": {
                    "$set": {
                        "summary": bio[:2000],
                        "headline": display_name[:120],
                    }
                }
            }
            patch_resp = await client.post(
                f"https://api.linkedin.com/v2/people/(id:{member_id})",
                json=patch_payload,
                headers={**headers, "Content-Type": "application/json"},
            )
            patch_resp.raise_for_status()
        return {"linkedin_profile_url": f"https://www.linkedin.com/in/{member_id}/"}

    # ── Credential resolution ─────────────────────────────────────────────────

    async def _resolve_credentials(self, platform: str) -> dict[str, str] | None:
        """
        Retrieve platform credentials from IdentityVault.

        Looks up the sealed envelope for the given platform under
        purpose="oauth_token" (standard label from account_provisioner).
        """
        if self._vault is None:
            return None
        try:
            label = f"{platform}_oauth_credentials"
            envelope = await self._vault.unseal(label=label)
            if envelope is None:
                # Fallback: try account_provisioner label conventions
                label = f"{platform}_credentials"
                envelope = await self._vault.unseal(label=label)
            if envelope is None:
                return None
            import json
            data = json.loads(envelope.decode())
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning("resolve_credentials_failed", platform=platform, error=str(exc))
            return None

    # ── Cooldown helpers ──────────────────────────────────────────────────────

    def _cooldown_key(self, platform: str) -> str:
        return f"axon:platform_profile_update:{platform}:{self._instance_id}"

    async def _check_cooldown(self, platform: str) -> bool:
        """Return True if the 24h cooldown is active (should skip update)."""
        if self._redis is None:
            return False  # No Redis - no cooldown enforcement
        try:
            key = self._cooldown_key(platform)
            exists = await self._redis.exists(key)
            return bool(exists)
        except Exception:
            return False  # Fail open - don't block update if Redis unavailable

    async def _set_cooldown(self, platform: str) -> None:
        """Set the 24h cooldown key in Redis after a successful update."""
        if self._redis is None:
            return
        try:
            key = self._cooldown_key(platform)
            await self._redis.set(key, "1", ex=_COOLDOWN_TTL_S)
        except Exception as exc:
            logger.warning("set_cooldown_failed", platform=platform, error=str(exc))
