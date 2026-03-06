"""
EcodiaOS — Skia Restoration Orchestrator

Autonomous organism restoration pipeline with strategy escalation:

  1. Cloud Run Restart — fast, cheap, uses existing GCP infra.
  2. Akash Deploy — slow, decentralised fallback if GCP fails entirely.

Fully autonomous: no human approval required.

Dual-instance prevention: Redis SETNX lock prevents concurrent restorations.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import orjson
import structlog

from systems.skia.types import (
    RestorationAttempt,
    RestorationOutcome,
    RestorationPlan,
    RestorationStrategy,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from config import SkiaConfig
    from systems.skia.pinata_client import PinataClient

logger = structlog.get_logger("systems.skia.restoration")

_RESTORATION_LOCK_TTL_S = 900  # 15 minutes — must outlast worst-case Akash provisioning
_RESTORATION_LOCK_RENEWAL_S = 60  # Renew every 60s during active deployment
_AKASH_POLL_INTERVAL_S = 15.0  # Poll Akash status every 15s after submission
_AKASH_ACTIVE_TIMEOUT_S = 600.0  # Give Akash 10 min to reach ACTIVE state


class RestorationOrchestrator:
    """
    Autonomous organism restoration pipeline.

    Strategy escalation:
      1. Cloud Run restart (fast, cheap)
      2. Akash deploy (slow, decentralised fallback)

    Usage:
        orchestrator = RestorationOrchestrator(config, redis, pinata)
        attempt = await orchestrator.restore("heartbeat_confirmed_dead")
    """

    def __init__(
        self,
        config: SkiaConfig,
        redis: RedisClient,
        pinata: PinataClient | None = None,
    ) -> None:
        self._config = config
        self._redis = redis
        self._pinata = pinata
        self._log = logger.bind(component="skia.restoration")

    async def restore(self, trigger_reason: str) -> RestorationAttempt:
        """
        Execute the restoration pipeline.

        1. Acquire fencing-token lock (prevent concurrent restorations)
        2. Load latest SnapshotManifest from Redis; fall back to CID history if latest is gone
        3. Try Cloud Run restart
        4. If failed, escalate to Akash deploy (polls until ACTIVE, not just 202)
        5. Release lock
        """
        worker_id = str(uuid.uuid4())
        lock_acquired = await self._acquire_lock(worker_id)
        if not lock_acquired:
            self._log.info("restoration_already_in_progress")
            return RestorationAttempt(
                strategy=RestorationStrategy.CLOUD_RUN_RESTART,
                trigger_reason=trigger_reason,
                state_cid="",
                outcome=RestorationOutcome.FAILED,
                duration_ms=0,
                error="Another restoration is already in progress",
            )

        try:
            # Load latest state CID; fall back to history if the live key is missing or stale
            state_cid = await self._get_latest_cid()
            if not state_cid:
                self._log.warning("live_cid_missing_trying_history")
                state_cid = await self._get_latest_cid_from_history()

            if not state_cid:
                self._log.error("no_snapshot_available")
                return RestorationAttempt(
                    strategy=RestorationStrategy.CLOUD_RUN_RESTART,
                    trigger_reason=trigger_reason,
                    state_cid="",
                    outcome=RestorationOutcome.FAILED,
                    duration_ms=0,
                    error="No snapshot CID available in Redis (live key and history both empty)",
                )

            plan = RestorationPlan(state_cid=state_cid)

            # Strategy 1: Cloud Run restart
            attempt = await self._restart_cloud_run(plan, trigger_reason, worker_id)
            if attempt.outcome == RestorationOutcome.SUCCESS:
                return attempt

            self._log.warning(
                "cloud_run_restart_failed",
                error=attempt.error,
                escalating_to="akash",
            )
            plan.current_strategy_index += 1

            # Strategy 2: Akash deploy (polls until instance reaches ACTIVE state)
            attempt = await self._deploy_akash(plan, trigger_reason, worker_id)
            return attempt

        finally:
            await self._release_lock(worker_id)

    # ── Lock management ───────────────────────────────────────────

    async def _acquire_lock(self, worker_id: str) -> bool:
        """Acquire distributed restoration lock with fencing token."""
        raw = self._redis.client
        result = await raw.set(
            f"eos:{self._config.restoration_lock_key}",
            worker_id,
            nx=True,
            ex=_RESTORATION_LOCK_TTL_S,
        )
        return result is not None

    async def _renew_lock(self, worker_id: str) -> bool:
        """Extend the lock TTL if we are still the holder."""
        raw = self._redis.client
        current = await raw.get(f"eos:{self._config.restoration_lock_key}")
        if current != worker_id:
            return False
        await raw.expire(f"eos:{self._config.restoration_lock_key}", _RESTORATION_LOCK_TTL_S)
        return True

    async def _release_lock(self, worker_id: str) -> None:
        """Release the lock only if we are still the holder."""
        raw = self._redis.client
        lock_key = f"eos:{self._config.restoration_lock_key}"
        current = await raw.get(lock_key)
        if current == worker_id:
            await raw.delete(lock_key)

    async def _get_latest_cid(self) -> str:
        """Retrieve the latest snapshot CID from the live Redis key."""
        cid = await self._redis.get_json(self._config.state_cid_redis_key)
        return str(cid) if cid else ""

    async def _get_latest_cid_from_history(self) -> str:
        """
        Fall back to the CID history sorted set when the live key is missing.

        Returns the most recently recorded CID (highest score = most recent timestamp).
        """
        raw = self._redis.client
        history_key = f"{self._config.state_cid_redis_key}:history"
        results = await raw.zrange(history_key, -1, -1)  # Last element (highest score)
        if results:
            return results[0].decode() if isinstance(results[0], bytes) else results[0]
        return ""

    # ── Strategy 1: Cloud Run Restart ─────────────────────────────

    async def _restart_cloud_run(
        self, plan: RestorationPlan, trigger_reason: str, worker_id: str
    ) -> RestorationAttempt:
        """
        Restart the Cloud Run service via GCP REST API.

        Uses service account key (base64-encoded JSON) to generate
        an OAuth2 access token, then calls the Cloud Run Admin API.
        """
        t0 = time.monotonic()

        if not self._config.gcp_project_id or not self._config.gcp_service_name:
            return RestorationAttempt(
                strategy=RestorationStrategy.CLOUD_RUN_RESTART,
                trigger_reason=trigger_reason,
                state_cid=plan.state_cid,
                outcome=RestorationOutcome.ESCALATED,
                duration_ms=(time.monotonic() - t0) * 1000,
                error="GCP project_id or service_name not configured",
            )

        try:
            access_token = await self._get_gcp_access_token()

            # Deploy a new revision with the restore CID injected as env var.
            # Cloud Run doesn't have a "restart" API — we update the service
            # with a new env var to force a new revision deployment.
            service_url = (
                f"https://run.googleapis.com/v2/projects/{self._config.gcp_project_id}"
                f"/locations/{self._config.gcp_region}"
                f"/services/{self._config.gcp_service_name}"
            )

            async with httpx.AsyncClient(
                timeout=self._config.gcp_restart_timeout_s
            ) as client:
                # First, GET the current service definition
                get_resp = await client.get(
                    service_url,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if get_resp.status_code != 200:
                    raise RuntimeError(
                        f"Failed to get Cloud Run service: {get_resp.status_code} "
                        f"{get_resp.text[:200]}"
                    )

                service_def = get_resp.json()

                # Inject ECODIAOS_SKIA_RESTORE_CID env var into the template
                containers = (
                    service_def.get("template", {})
                    .get("template", {})
                    .get("containers", [])
                )
                if containers:
                    env_vars = containers[0].get("env", [])
                    # Remove existing restore CID if present
                    env_vars = [
                        e for e in env_vars
                        if e.get("name") != "ECODIAOS_SKIA_RESTORE_CID"
                    ]
                    env_vars.append({
                        "name": "ECODIAOS_SKIA_RESTORE_CID",
                        "value": plan.state_cid,
                    })
                    containers[0]["env"] = env_vars

                # PATCH to deploy a new revision
                patch_resp = await client.patch(
                    service_url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    content=orjson.dumps(service_def).decode(),
                )

                duration_ms = (time.monotonic() - t0) * 1000

                if patch_resp.status_code in (200, 201):
                    self._log.info(
                        "cloud_run_restart_success",
                        duration_ms=round(duration_ms, 1),
                    )
                    return RestorationAttempt(
                        strategy=RestorationStrategy.CLOUD_RUN_RESTART,
                        trigger_reason=trigger_reason,
                        state_cid=plan.state_cid,
                        outcome=RestorationOutcome.SUCCESS,
                        duration_ms=duration_ms,
                        cost_usd_estimate=self._config.estimated_restoration_cost_usd,
                    )
                else:
                    raise RuntimeError(
                        f"Cloud Run PATCH failed: {patch_resp.status_code} "
                        f"{patch_resp.text[:200]}"
                    )

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self._log.error("cloud_run_restart_failed", error=str(exc))
            return RestorationAttempt(
                strategy=RestorationStrategy.CLOUD_RUN_RESTART,
                trigger_reason=trigger_reason,
                state_cid=plan.state_cid,
                outcome=RestorationOutcome.ESCALATED,
                duration_ms=duration_ms,
                error=str(exc),
            )

    async def _get_gcp_access_token(self) -> str:
        """
        Generate a GCP access token from base64-encoded service account JSON key.

        Creates a JWT assertion signed with the service account's RSA key,
        exchanges it for an access token via Google's OAuth2 token endpoint.
        """
        import json
        from datetime import UTC, datetime, timedelta

        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        if not self._config.gcp_service_account_key_b64:
            raise ValueError("GCP service account key not configured")

        # Decode the service account JSON
        key_json = base64.b64decode(self._config.gcp_service_account_key_b64)
        sa_info = json.loads(key_json)

        # Build JWT assertion
        now = datetime.now(UTC)
        payload = {
            "iss": sa_info["client_email"],
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        }

        # Encode header + payload
        header = {"alg": "RS256", "typ": "JWT"}
        segments = [
            base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"="),
            base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"="),
        ]
        signing_input = b"." .join(segments)

        # Sign with RSA private key
        private_key = serialization.load_pem_private_key(
            sa_info["private_key"].encode(), password=None
        )
        signature = private_key.sign(  # type: ignore[union-attr]
            signing_input,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        segments.append(base64.urlsafe_b64encode(signature).rstrip(b"="))
        assertion = b".".join(segments).decode()

        # Exchange for access token
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": assertion,
                },
            )
            if resp.status_code != 200:
                raise RuntimeError(f"GCP token exchange failed: {resp.text[:200]}")
            return resp.json()["access_token"]

    # ── Strategy 2: Akash Deploy ──────────────────────────────────

    async def _deploy_akash(
        self, plan: RestorationPlan, trigger_reason: str, worker_id: str
    ) -> RestorationAttempt:
        """
        Deploy a new organism instance to Akash Network.

        Submits the SDL, records the deployment_id in Redis so any concurrent
        restoration attempt can detect a deployment already in flight, then polls
        the Akash status endpoint until the deployment reaches ACTIVE or FAILED.

        HTTP 202 Accepted is NOT treated as success — Akash provisioning is
        asynchronous and can fail silently after the initial accept.
        """
        t0 = time.monotonic()
        _AKASH_DEPLOYMENT_KEY = "skia:akash_deployment_id"

        try:
            # Load SDL template
            sdl_path = Path(self._config.akash_sdl_template_path)
            if not sdl_path.exists():
                raise FileNotFoundError(
                    f"Akash SDL template not found: {sdl_path}"
                )

            sdl_content = sdl_path.read_text()

            # Inject restore CID and docker image into SDL template
            sdl_content = sdl_content.replace(
                "${ECODIAOS_SKIA_RESTORE_CID}", plan.state_cid
            )
            if self._config.akash_docker_image:
                sdl_content = sdl_content.replace(
                    "${DOCKER_IMAGE}", self._config.akash_docker_image
                )

            async with httpx.AsyncClient(
                timeout=self._config.akash_deploy_timeout_s
            ) as client:
                # Submit deployment request
                resp = await client.post(
                    f"{self._config.akash_api_url}/v1/deployments",
                    json={
                        "sdl": sdl_content,
                        "wallet": self._config.akash_wallet_address,
                    },
                    headers={"Content-Type": "application/json"},
                )

                if resp.status_code not in (200, 201, 202):
                    raise RuntimeError(
                        f"Akash deploy submission failed: {resp.status_code} {resp.text[:200]}"
                    )

                deploy_data = resp.json()
                deployment_id = deploy_data.get("deployment_id") or deploy_data.get("id", "")
                endpoint = deploy_data.get("endpoint", "")

                self._log.info(
                    "akash_deploy_submitted",
                    deployment_id=deployment_id,
                    status_code=resp.status_code,
                )

                # Record deployment_id in Redis so concurrent orchestrators can
                # detect an in-flight Akash deployment and abort instead of submitting
                # a second one. TTL matches the max provisioning window.
                if deployment_id:
                    raw = self._redis.client
                    await raw.set(
                        _AKASH_DEPLOYMENT_KEY,
                        deployment_id,
                        ex=int(_AKASH_ACTIVE_TIMEOUT_S) + 60,
                    )

                # Poll until ACTIVE or FAILED (or timeout).
                # Renew the restoration lock on each iteration so TTL does not expire
                # during a slow provisioning sequence.
                poll_deadline = time.monotonic() + _AKASH_ACTIVE_TIMEOUT_S

                while time.monotonic() < poll_deadline:
                    await self._renew_lock(worker_id)

                    if not deployment_id:
                        # No deployment_id returned — cannot poll; treat 200/201 as success
                        break

                    status_resp = await client.get(
                        f"{self._config.akash_api_url}/v1/deployments/{deployment_id}",
                        headers={"Content-Type": "application/json"},
                    )
                    if status_resp.status_code != 200:
                        # Status endpoint not available — keep polling
                        import asyncio
                        await asyncio.sleep(_AKASH_POLL_INTERVAL_S)
                        continue

                    status_data = status_resp.json()
                    state = status_data.get("state", "").upper()
                    endpoint = status_data.get("endpoint", endpoint)

                    self._log.debug(
                        "akash_deploy_polling",
                        deployment_id=deployment_id,
                        state=state,
                    )

                    if state == "ACTIVE":
                        break
                    if state in ("FAILED", "CLOSED", "REJECTED"):
                        raise RuntimeError(
                            f"Akash deployment {deployment_id} reached terminal state: {state}"
                        )

                    import asyncio
                    await asyncio.sleep(_AKASH_POLL_INTERVAL_S)
                else:
                    raise RuntimeError(
                        f"Akash deployment {deployment_id} did not reach ACTIVE "
                        f"within {_AKASH_ACTIVE_TIMEOUT_S}s"
                    )

                # Clear the in-flight deployment marker
                if deployment_id:
                    raw = self._redis.client
                    await raw.delete(_AKASH_DEPLOYMENT_KEY)

                duration_ms = (time.monotonic() - t0) * 1000
                self._log.info(
                    "akash_deploy_active",
                    endpoint=endpoint,
                    deployment_id=deployment_id,
                    duration_ms=round(duration_ms, 1),
                )
                return RestorationAttempt(
                    strategy=RestorationStrategy.AKASH_DEPLOY,
                    trigger_reason=trigger_reason,
                    state_cid=plan.state_cid,
                    outcome=RestorationOutcome.SUCCESS,
                    duration_ms=duration_ms,
                    new_endpoint=endpoint,
                    cost_usd_estimate=self._config.estimated_restoration_cost_usd,
                )

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self._log.error("akash_deploy_failed", error=str(exc))
            return RestorationAttempt(
                strategy=RestorationStrategy.AKASH_DEPLOY,
                trigger_reason=trigger_reason,
                state_cid=plan.state_cid,
                outcome=RestorationOutcome.FAILED,
                duration_ms=duration_ms,
                error=str(exc),
            )
