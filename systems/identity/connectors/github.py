"""
EcodiaOS - GitHubConnector (Phase 16r: Bounty Submission Layer)

Provides all GitHub API operations needed by BountySubmitExecutor:
  - get_access_token()  - App JWT→IAT or personal token, Redis-cached
  - check_health()      - 3-failure threshold before SYSTEM_DEGRADED
  - fork_repository()   - POST /repos/{owner}/{repo}/forks
  - create_branch()     - POST /repos/{owner}/{repo}/git/refs
  - commit_files()      - PUT /repos/{owner}/{repo}/contents/{path} (one file at a time)
  - open_pull_request() - POST /repos/{owner}/{repo}/pulls
  - get_pr_status()     - GET  /repos/{owner}/{repo}/pulls/{number}

Authentication precedence (resolved at construction time):
  1. GitHub App (GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY + GITHUB_INSTALLATION_ID)
     Tokens minted on demand via JWT→IAT; cached in Redis for TTL - 60 s.
  2. Personal Access Token (GITHUB_TOKEN)
     Injected directly as a Bearer token; cached as a static sentinel.

Rate limiting (GitHub REST API):
  - Authenticated requests: 5 000 / hour.
  - PR creation: conservative internal cap of 5 / hour enforced by caller
    (BountySubmitExecutor), not here - this connector is the transport layer.

This connector does NOT extend PlatformConnector because it is not an OAuth2
flow connector; it is a thin HTTP client facade with optional IAT delegation
to GitHubAppConnector.  It does respect the same Redis caching convention
used by PlatformConnector._write_token_cache() so tokens are shared across
requests without repeated IAT round-trips.
"""

from __future__ import annotations

import base64
import contextlib
import os
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.identity.connectors.github_app import GitHubAppConnector
    from systems.identity.vault import IdentityVault
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.github_connector")

_GITHUB_API_BASE = "https://api.github.com"
_CACHE_KEY = "identity:token:github_connector"
_CACHE_GUARD_S = 60          # evict 60 s before expiry
_IAT_TTL_S = 3600            # GitHub IAT lifetime
_HEALTH_FAILURE_THRESHOLD = 3


class GitHubConnector:
    """
    GitHub API connector used by BountySubmitExecutor.

    Supports GitHub App authentication (via GitHubAppConnector) and personal
    token fallback.  All HTTP operations return plain Python dicts / strings
    rather than domain models - the executor layer owns any result mapping.

    Thread-safety: NOT thread-safe.  Single-threaded asyncio like all EOS.
    """

    def __init__(
        self,
        app_connector: GitHubAppConnector | None = None,
        personal_token: str = "",
        redis: RedisClient | None = None,
        event_bus: EventBus | None = None,
        http_client: httpx.AsyncClient | None = None,
        vault: IdentityVault | None = None,
    ) -> None:
        """
        Args:
            app_connector:   Wired GitHubAppConnector for JWT→IAT auth.
                             Takes precedence over personal_token when both present.
            personal_token:  GitHub personal access token (PAT / classic).
                             Falls back to GITHUB_TOKEN env var when empty.
            redis:           Redis client for token caching.
            event_bus:       Synapse bus for SYSTEM_DEGRADED emission.
            http_client:     Injectable httpx client (testing convenience).
            vault:           IdentityVault for sealing the env-var PAT at rest.
        """
        self._app_connector = app_connector
        self._personal_token = (
            personal_token
            or os.environ.get("ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or ""
        )
        self._redis = redis
        self._event_bus = event_bus
        self._vault = vault
        self._http = http_client or httpx.AsyncClient(
            base_url=_GITHUB_API_BASE,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )
        self._consecutive_health_failures = 0
        self._logger = logger.bind(component="github_connector")

        # Seal the env-var PAT in IdentityVault so it is never stored plaintext.
        # This runs synchronously at construction - vault.encrypt_token_json is
        # CPU-only (Fernet) and safe to call from __init__.
        if self._personal_token and vault is not None:
            try:
                vault.encrypt_token_json(
                    {"access_token": self._personal_token, "token_type": "pat"},
                    platform_id="github_pat",
                    purpose="operator_pat",
                )
                self._logger.info("github_pat_sealed_in_vault")
            except Exception as _seal_exc:
                self._logger.warning("github_pat_vault_seal_failed", error=str(_seal_exc))

    # ── Token resolution ───────────────────────────────────────────────────

    async def get_access_token(self) -> str | None:
        """
        Return a valid GitHub access token, using cache when possible.

        Precedence:
          1. Redis cache hit with > 60 s remaining → return immediately.
          2. GitHub App connector → mint fresh IAT, cache it.
          3. Personal token → return directly (no expiry management needed).
        """
        # 1. Redis cache hit
        cached = await self._read_cache()
        if cached:
            return cached

        # 2. GitHub App path
        if self._app_connector is not None:
            token = await self._app_connector.get_access_token()
            if token:
                # Cache for IAT TTL minus guard - same convention as PlatformConnector
                await self._write_cache(token, ttl=_IAT_TTL_S - _CACHE_GUARD_S)
                self._logger.debug("github_iat_acquired_and_cached")
                return token
            self._logger.warning(
                "github_app_connector_returned_no_token",
                note="falling back to personal token if configured",
            )

        # 3. Personal token (static; no caching needed, but write sentinel to skip
        #    the Redis round-trip on repeated calls within the same process lifetime)
        if self._personal_token:
            await self._write_cache(self._personal_token, ttl=3600)
            return self._personal_token

        return None

    # ── Health ─────────────────────────────────────────────────────────────

    async def authenticate(self) -> bool:
        """
        Validate token availability and emit CONNECTOR_AUTHENTICATED.

        Call once during registry boot after all dependencies are wired.
        Returns True if a token is available (App IAT or PAT).
        """
        token = await self.get_access_token()
        if not token:
            self._logger.warning("github_connector_no_token_on_authenticate")
            return False
        await self._emit_connector_authenticated()
        return True

    async def check_health(self) -> bool:
        """
        Probe token availability.  After 3 consecutive failures emits
        SYSTEM_DEGRADED on the Synapse bus.

        Returns True if a valid token is available.
        """
        token = await self.get_access_token()
        healthy = token is not None

        if healthy:
            self._consecutive_health_failures = 0
        else:
            self._consecutive_health_failures += 1
            self._logger.warning(
                "github_connector_health_check_failed",
                consecutive_failures=self._consecutive_health_failures,
            )
            if self._consecutive_health_failures >= _HEALTH_FAILURE_THRESHOLD:
                await self._emit_degraded()

        return healthy

    # ── GitHub API Operations ──────────────────────────────────────────────

    async def fork_repository(self, owner: str, repo: str) -> dict[str, Any]:
        """
        Fork owner/repo into the authenticated user's account.

        Returns the fork metadata dict from GitHub (includes "full_name",
        "clone_url", "html_url", etc.).  Raises httpx.HTTPStatusError on failure.

        GitHub notes:
          - POST /repos/{owner}/{repo}/forks returns 202 Accepted.
          - If the fork already exists GitHub silently returns the existing fork.
        """
        token = await self._require_token()
        resp = await self._http.post(
            f"/repos/{owner}/{repo}/forks",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        self._logger.info(
            "repo_forked",
            source=f"{owner}/{repo}",
            fork=data.get("full_name"),
        )
        return data

    async def create_branch(
        self,
        owner: str,
        repo: str,
        branch: str,
        base_branch: str = "main",
    ) -> dict[str, Any]:
        """
        Create branch in owner/repo from the HEAD of base_branch.

        Returns the ref object created by GitHub.
        """
        token = await self._require_token()

        # Resolve the SHA of base_branch
        ref_resp = await self._http.get(
            f"/repos/{owner}/{repo}/git/ref/heads/{base_branch}",
            headers={"Authorization": f"Bearer {token}"},
        )
        ref_resp.raise_for_status()
        sha = ref_resp.json()["object"]["sha"]

        # Create the new ref
        create_resp = await self._http.post(
            f"/repos/{owner}/{repo}/git/refs",
            headers={"Authorization": f"Bearer {token}"},
            json={"ref": f"refs/heads/{branch}", "sha": sha},
        )
        create_resp.raise_for_status()
        data: dict[str, Any] = create_resp.json()
        self._logger.info(
            "branch_created",
            repo=f"{owner}/{repo}",
            branch=branch,
            sha=sha[:8],
        )
        return data

    async def commit_files(
        self,
        owner: str,
        repo: str,
        branch: str,
        files: dict[str, str],
        commit_message: str,
    ) -> list[dict[str, Any]]:
        """
        Commit one or more files to branch in owner/repo via the Contents API.

        Args:
            files: Mapping of {path: content_string}.
                   Content is UTF-8 encoded and base64-wrapped for the API.
            commit_message: Commit message applied to every file PUT.

        Returns list of GitHub content-update response dicts (one per file).
        Raises httpx.HTTPStatusError on the first failed PUT.
        """
        token = await self._require_token()
        results: list[dict[str, Any]] = []

        for path, content in files.items():
            b64_content = base64.b64encode(content.encode("utf-8")).decode("ascii")

            # Check if file already exists (need its SHA for updates)
            existing_sha: str | None = None
            check_resp = await self._http.get(
                f"/repos/{owner}/{repo}/contents/{path}",
                headers={"Authorization": f"Bearer {token}"},
                params={"ref": branch},
            )
            if check_resp.status_code == 200:
                existing_sha = check_resp.json().get("sha")

            payload: dict[str, Any] = {
                "message": commit_message,
                "content": b64_content,
                "branch": branch,
            }
            if existing_sha:
                payload["sha"] = existing_sha

            put_resp = await self._http.put(
                f"/repos/{owner}/{repo}/contents/{path}",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
            put_resp.raise_for_status()
            results.append(put_resp.json())
            self._logger.debug("file_committed", path=path, repo=f"{owner}/{repo}", branch=branch)

        self._logger.info(
            "files_committed",
            repo=f"{owner}/{repo}",
            branch=branch,
            file_count=len(files),
        )
        return results

    async def open_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
        labels: list[str] | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        """
        Open a pull request on owner/repo.

        Args:
            head:   Branch name (or "fork_owner:branch" for cross-repo PRs).
            base:   Target branch (e.g. "main").
            labels: Label names to apply after PR creation (best-effort).

        Returns the PR object dict from GitHub (includes "number", "html_url", etc.).
        """
        token = await self._require_token()

        pr_resp = await self._http.post(
            f"/repos/{owner}/{repo}/pulls",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
            },
        )
        pr_resp.raise_for_status()
        pr: dict[str, Any] = pr_resp.json()
        pr_number: int = pr["number"]
        pr_url: str = pr.get("html_url", "")

        self._logger.info(
            "pull_request_opened",
            repo=f"{owner}/{repo}",
            pr_number=pr_number,
            pr_url=pr_url,
            head=head,
            base=base,
        )

        # Apply labels - best-effort; do not fail PR creation on label errors
        if labels:
            await self._apply_labels(owner, repo, pr_number, labels, token)

        return pr

    async def get_pr_status(
        self, owner: str, repo: str, pr_number: int
    ) -> dict[str, Any]:
        """
        Fetch the current state of a pull request.

        Returns the PR object dict (includes "state", "merged", "mergeable", etc.).
        """
        token = await self._require_token()
        resp = await self._http.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def create_gist(
        self,
        files: dict[str, str],
        description: str = "",
        public: bool = False,
    ) -> str:
        """
        Create a GitHub Gist from the provided file mapping.

        Args:
            files:       Mapping of {filename: content_string}.
            description: Short description shown on the Gist page.
            public:      If True, the Gist is world-readable; False = secret Gist.

        Returns the Gist HTML URL (e.g. "https://gist.github.com/…").
        Raises httpx.HTTPStatusError on API failure.

        Requires ``gist`` scope in the PAT or GitHub App permissions.
        """
        token = await self._require_token()
        gist_files = {name: {"content": content} for name, content in files.items()}
        resp = await self._http.post(
            "/gists",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "description": description,
                "public": public,
                "files": gist_files,
            },
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        html_url: str = data.get("html_url", "")
        self._logger.info("gist_created", public=public, file_count=len(files), url=html_url)
        return html_url

    async def get_active_github_token(self) -> str | None:
        """
        Return the best available GitHub token for bounty/PR work.

        Precedence:
          1. Own instance token stored in vault (provisioned via I-3 own-account flow).
          2. Operator token from ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN / GITHUB_TOKEN.

        Returns None if neither is available.  Callers should raise or fall back
        gracefully rather than passing None to API calls.
        """
        # 1. Own instance token - provisioned by the I-3 GitHub account flow
        own_token: str | None = None
        if self._app_connector is not None:
            # Try to get the IAT for the organism's own GitHub App installation
            own_token = await self._app_connector.get_access_token()

        if own_token:
            return own_token

        # 2. Operator / env-var token fallback
        if self._personal_token:
            return self._personal_token

        return None

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _require_token(self) -> str:
        """Return a token or raise RuntimeError if none available."""
        token = await self.get_access_token()
        if not token:
            raise RuntimeError(
                "No GitHub credentials available. "
                "Set GITHUB_TOKEN or configure the GitHub App connector."
            )
        return token

    async def _apply_labels(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        labels: list[str],
        token: str,
    ) -> None:
        """Apply labels to a PR - best-effort, errors logged but not raised."""
        try:
            resp = await self._http.post(
                f"/repos/{owner}/{repo}/issues/{pr_number}/labels",
                headers={"Authorization": f"Bearer {token}"},
                json={"labels": labels},
            )
            resp.raise_for_status()
        except Exception as exc:
            self._logger.warning(
                "label_apply_failed",
                repo=f"{owner}/{repo}",
                pr_number=pr_number,
                labels=labels,
                error=str(exc),
            )

    async def _read_cache(self) -> str | None:
        if self._redis is None:
            return None
        try:
            data = await self._redis.get_json(_CACHE_KEY)
            if isinstance(data, dict):
                return data.get("token")
        except Exception:
            pass
        return None

    async def _write_cache(self, token: str, ttl: int) -> None:
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            await self._redis.set_json(_CACHE_KEY, {"token": token}, ttl=max(1, ttl))

    async def _emit_connector_authenticated(self) -> None:
        """Emit CONNECTOR_AUTHENTICATED on successful token validation."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            auth_mode = "app_iat" if self._app_connector is not None else "pat"
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.CONNECTOR_AUTHENTICATED,
                source_system="identity",
                data={
                    "connector_id": "github_connector",
                    "platform_id": "github",
                    "auth_mode": auth_mode,
                    "pat_sealed": self._vault is not None and bool(self._personal_token),
                },
            ))
            self._logger.info("github_connector_authenticated", auth_mode=auth_mode)
        except Exception as exc:
            self._logger.warning("connector_authenticated_emit_failed", error=str(exc))

    async def _emit_degraded(self) -> None:
        """Emit SYSTEM_DEGRADED so Thymos can quarantine this connector."""
        self._logger.error(
            "github_connector_degraded",
            consecutive_failures=self._consecutive_health_failures,
        )
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SYSTEM_DEGRADED,
                source_system="identity",
                data={
                    "connector_id": "github_connector",
                    "platform_id": "github",
                    "consecutive_health_failures": self._consecutive_health_failures,
                    "action": "quarantine_requested",
                },
            ))
        except Exception as exc:
            self._logger.warning("degraded_event_emit_failed", error=str(exc))
