"""
EcodiaOS - Skia Restoration Orchestrator

Autonomous organism restoration pipeline with strategy escalation:

  1. Cloud Run Restart - fast, cheap, uses existing GCP infra.
  2. Akash Deploy - slow, decentralised fallback if GCP fails entirely.

Fully autonomous: no human approval required.

Dual-instance prevention: Redis SETNX lock prevents concurrent restorations.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

_RESTORATION_LOCK_TTL_S = 900  # 15 minutes - must outlast worst-case Akash provisioning
_RESTORATION_LOCK_RENEWAL_S = 60  # Renew every 60s during active deployment
_AKASH_POLL_INTERVAL_S = 15.0  # Poll Akash status every 15s after submission
_AKASH_ACTIVE_TIMEOUT_S = 600.0  # Give Akash 10 min to reach ACTIVE state
_MAX_RESTORATION_ATTEMPTS = 3    # After 3 failed full pipelines, declare infrastructure death


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

    # Redis key that stores the deployed shadow worker's endpoint and provider.
    _SHADOW_WORKER_KEY = "skia:shadow_worker"
    _SHADOW_WORKER_TTL_S = 86400 * 7  # 7-day TTL - shadow worker should self-renew

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
        self._attempt_count: int = 0
        self._infrastructure_dead: bool = False
        # Constitutional genome from the latest snapshot - injected into new instance env.
        self._constitutional_genome: dict[str, Any] | None = None
        # Shadow worker deployment lock prevents concurrent shadow provisions.
        self._shadow_deploy_in_progress: bool = False

    def set_constitutional_genome(self, genome: dict[str, Any] | None) -> None:
        """Store the constitutional genome to pass to provisioned shadow instances."""
        self._constitutional_genome = genome

    @property
    def infrastructure_dead(self) -> bool:
        """True after max restoration attempts exhausted."""
        return self._infrastructure_dead

    async def restore(self, trigger_reason: str) -> RestorationAttempt:
        """
        Execute the restoration pipeline.

        1. Acquire fencing-token lock (prevent concurrent restorations)
        2. Load latest SnapshotManifest from Redis; fall back to CID history if latest is gone
        3. Try Cloud Run restart
        4. If failed, escalate to Akash deploy (polls until ACTIVE, not just 202)
        5. Release lock
        """
        if self._infrastructure_dead:
            self._log.error("infrastructure_permanently_dead", attempts=self._attempt_count)
            return RestorationAttempt(
                strategy=RestorationStrategy.AKASH_DEPLOY,
                trigger_reason=trigger_reason,
                state_cid="",
                outcome=RestorationOutcome.FAILED,
                duration_ms=0,
                error=f"Infrastructure declared dead after {_MAX_RESTORATION_ATTEMPTS} failed attempts",
            )

        self._attempt_count += 1
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

            if attempt.outcome == RestorationOutcome.SUCCESS:
                self._attempt_count = 0  # Reset on success
                return attempt

            # Both strategies failed. Check if we've exhausted max attempts.
            if self._attempt_count >= _MAX_RESTORATION_ATTEMPTS:
                self._infrastructure_dead = True
                self._log.critical(
                    "infrastructure_death_declared",
                    attempts=self._attempt_count,
                    max=_MAX_RESTORATION_ATTEMPTS,
                )
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
            # Cloud Run doesn't have a "restart" API - we update the service
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

                # Inject ORGANISM_SKIA_RESTORE_CID env var into the template
                containers = (
                    service_def.get("template", {})
                    .get("template", {})
                    .get("containers", [])
                )
                if containers:
                    env_vars = containers[0].get("env", [])
                    # Remove existing restore/genome vars before re-injecting
                    env_vars = [
                        e for e in env_vars
                        if e.get("name") not in (
                            "ORGANISM_SKIA_RESTORE_CID",
                            "ORGANISM_CONSTITUTIONAL_GENOME_B64",
                        )
                    ]
                    env_vars.append({
                        "name": "ORGANISM_SKIA_RESTORE_CID",
                        "value": plan.state_cid,
                    })
                    # Pass constitutional genome so new instance inherits parent phenotype
                    if self._constitutional_genome is not None:
                        genome_b64 = base64.b64encode(
                            orjson.dumps(self._constitutional_genome)
                        ).decode("ascii")
                        env_vars.append({
                            "name": "ORGANISM_CONSTITUTIONAL_GENOME_B64",
                            "value": genome_b64,
                        })
                        self._log.info(
                            "constitutional_genome_injected",
                            genome_size_bytes=len(genome_b64),
                        )
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

        HTTP 202 Accepted is NOT treated as success - Akash provisioning is
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
                "${ORGANISM_SKIA_RESTORE_CID}", plan.state_cid
            )
            if self._config.akash_docker_image:
                sdl_content = sdl_content.replace(
                    "${DOCKER_IMAGE}", self._config.akash_docker_image
                )

            # Prepare constitutional genome payload for Akash env injection
            genome_b64 = ""
            if self._constitutional_genome is not None:
                genome_b64 = base64.b64encode(
                    orjson.dumps(self._constitutional_genome)
                ).decode("ascii")
                sdl_content = sdl_content.replace(
                    "${ORGANISM_CONSTITUTIONAL_GENOME_B64}", genome_b64
                )
                self._log.info(
                    "constitutional_genome_injected_akash",
                    genome_size_bytes=len(genome_b64),
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
                        # Also pass genome in the API payload as Akash providers
                        # may support env injection outside the SDL.
                        "env": {"ORGANISM_CONSTITUTIONAL_GENOME_B64": genome_b64}
                        if genome_b64 else {},
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
                        # No deployment_id returned - cannot poll; treat 200/201 as success
                        break

                    status_resp = await client.get(
                        f"{self._config.akash_api_url}/v1/deployments/{deployment_id}",
                        headers={"Content-Type": "application/json"},
                    )
                    if status_resp.status_code != 200:
                        # Status endpoint not available - keep polling
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

    # ── Shadow Worker ────────────────────────────────────────────────────────

    async def ensure_shadow_worker(self) -> bool:
        """
        Ensure a lightweight shadow worker (heartbeat monitor + state restore)
        is running on a DIFFERENT provider than the main instance.

        Called by SkiaService on startup and periodically (e.g. every 6 hours).
        If a shadow worker already exists and is reachable, returns True immediately.
        If not, deploys one using Akash (cheapest, decentralised) or a free-tier
        Cloud Run service, removing the human dependency for resurrection.

        Returns True if a shadow worker is confirmed deployed, False on failure.
        """
        if self._shadow_deploy_in_progress:
            self._log.debug("shadow_worker_deploy_already_in_progress")
            return False

        # 1. Check if a shadow worker already exists and is healthy
        existing = await self._get_shadow_worker_record()
        if existing:
            healthy = await self._check_shadow_worker_health(existing.get("endpoint", ""))
            if healthy:
                self._log.debug(
                    "shadow_worker_already_running",
                    endpoint=existing.get("endpoint", ""),
                    provider=existing.get("provider", ""),
                )
                return True
            # Existing record but unhealthy - redeploy
            self._log.warning(
                "shadow_worker_unhealthy_redeploying",
                endpoint=existing.get("endpoint", ""),
            )

        self._shadow_deploy_in_progress = True
        try:
            return await self._deploy_shadow_worker()
        finally:
            self._shadow_deploy_in_progress = False

    async def _deploy_shadow_worker(self) -> bool:
        """
        Deploy the shadow worker on the provider that differs from the main instance.

        Shadow worker requirements are minimal:
          - heartbeat listener (publishes to Redis pub/sub channel)
          - state restore capability (reads CID from Redis, provisions main instance)
          - lightweight: 0.1 CPU, 128Mi memory

        Provider selection: prefer Akash (cheapest, decentralised).
        Fallback: free-tier Cloud Run on a secondary region/project if configured.
        """
        t0 = time.monotonic()
        self._log.info("shadow_worker_deploying")

        # Try Akash first (different provider from main Cloud Run instance)
        endpoint = ""
        provider = ""
        deployment_id = ""

        try:
            endpoint, provider, deployment_id = await self._deploy_shadow_akash()
        except Exception as exc:
            self._log.warning(
                "shadow_worker_akash_failed_trying_cloud_run",
                error=str(exc),
            )
            try:
                endpoint, provider, deployment_id = await self._deploy_shadow_cloud_run()
            except Exception as exc2:
                self._log.error(
                    "shadow_worker_all_providers_failed",
                    akash_error=str(exc),
                    cloud_run_error=str(exc2),
                )
                return False

        if not endpoint:
            self._log.error("shadow_worker_deployed_but_no_endpoint")
            return False

        # Verify the shadow worker is reachable
        healthy = await self._check_shadow_worker_health(endpoint)
        if not healthy:
            self._log.warning(
                "shadow_worker_deployed_but_not_healthy",
                endpoint=endpoint,
                provider=provider,
            )
            # Still record it - it may become healthy shortly; next ensure_shadow_worker
            # call will catch the unhealthy state and redeploy.

        # Persist to Redis so all EOS instances and the next boot can find it
        await self._persist_shadow_worker_record(
            endpoint=endpoint,
            provider=provider,
            deployment_id=deployment_id,
        )
        duration_ms = (time.monotonic() - t0) * 1000
        self._log.info(
            "shadow_worker_deployed",
            endpoint=endpoint,
            provider=provider,
            healthy=healthy,
            duration_ms=round(duration_ms, 1),
        )
        return True

    async def _deploy_shadow_akash(self) -> tuple[str, str, str]:
        """
        Deploy a minimal skia_worker image to Akash.

        The shadow SDL is a stripped-down version of the main SDL:
          - 0.1 CPU / 128Mi memory / 512Mi storage
          - environment: ORGANISM_STANDALONE=true, CID key, heartbeat config
          - does NOT restore from CID automatically - only monitors + waits for trigger

        Returns (endpoint, provider_id, deployment_id).
        """
        from pathlib import Path

        sdl_path = Path(self._config.akash_sdl_template_path)
        if not sdl_path.exists():
            raise FileNotFoundError(f"Akash SDL template not found: {sdl_path}")

        sdl_content = sdl_path.read_text()
        # Shadow worker uses same image but with standalone mode + minimal resources
        if self._config.akash_docker_image:
            sdl_content = sdl_content.replace("${DOCKER_IMAGE}", self._config.akash_docker_image)
        # Inject shadow-mode env vars; leave restore CID empty (it reads from Redis at restore time)
        sdl_content = sdl_content.replace("${ORGANISM_SKIA_RESTORE_CID}", "")
        sdl_content = sdl_content.replace("${ORGANISM_CONSTITUTIONAL_GENOME_B64}", "")

        # Override resource spec to minimal shadow worker allocation
        sdl_content = _patch_sdl_for_shadow(sdl_content)

        async with httpx.AsyncClient(timeout=self._config.akash_deploy_timeout_s) as client:
            resp = await client.post(
                f"{self._config.akash_api_url}/v1/deployments",
                json={
                    "sdl": sdl_content,
                    "wallet": self._config.akash_wallet_address,
                    "env": {
                        "ORGANISM_STANDALONE": "true",
                        "ORGANISM_SHADOW_WORKER": "true",
                        "SKIA_CID_REDIS_KEY": self._config.state_cid_redis_key,
                    },
                },
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code not in (200, 201, 202):
                raise RuntimeError(
                    f"Akash shadow deploy failed: {resp.status_code} {resp.text[:200]}"
                )

            deploy_data = resp.json()
            deployment_id = deploy_data.get("deployment_id") or deploy_data.get("id", "")
            endpoint = deploy_data.get("endpoint", "")

            # Poll for ACTIVE (shorter timeout than full restoration - shadow is lightweight)
            _SHADOW_TIMEOUT_S = 300.0
            poll_deadline = time.monotonic() + _SHADOW_TIMEOUT_S

            while time.monotonic() < poll_deadline and deployment_id:
                status_resp = await client.get(
                    f"{self._config.akash_api_url}/v1/deployments/{deployment_id}",
                )
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    state = status_data.get("state", "").upper()
                    endpoint = status_data.get("endpoint", endpoint)
                    if state == "ACTIVE":
                        break
                    if state in ("FAILED", "CLOSED", "REJECTED"):
                        raise RuntimeError(f"Shadow worker Akash deployment failed: {state}")
                import asyncio as _asyncio
                await _asyncio.sleep(15.0)

            return endpoint, "akash", deployment_id

    async def _deploy_shadow_cloud_run(self) -> tuple[str, str, str]:
        """
        Deploy shadow worker as a free-tier Cloud Run service in a secondary region.

        Uses the same GCP credentials but targets a different service name
        (appended with "-shadow") and region to provide geographic redundancy.
        Returns (endpoint, provider_id, service_url).
        """
        import base64
        import json
        from datetime import UTC, datetime, timedelta

        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        if not self._config.gcp_project_id or not self._config.gcp_service_account_key_b64:
            raise RuntimeError("GCP not configured - cannot deploy Cloud Run shadow worker")

        key_json = base64.b64decode(self._config.gcp_service_account_key_b64)
        sa_info = json.loads(key_json)
        now = datetime.now(UTC)
        payload = {
            "iss": sa_info["client_email"],
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        }
        header = {"alg": "RS256", "typ": "JWT"}
        segments = [
            base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"="),
            base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"="),
        ]
        signing_input = b".".join(segments)
        private_key = serialization.load_pem_private_key(sa_info["private_key"].encode(), password=None)
        signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())  # type: ignore[union-attr]
        segments.append(base64.urlsafe_b64encode(signature).rstrip(b"="))
        assertion = b".".join(segments).decode()

        # Shadow Cloud Run is deployed in a *different* region from the main instance
        # to provide geographic redundancy.  We use us-central1 as universal fallback.
        shadow_region = _pick_shadow_region(self._config.gcp_region)
        shadow_service = f"{self._config.gcp_service_name}-shadow"

        async with httpx.AsyncClient(timeout=120.0) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
            )
            if token_resp.status_code != 200:
                raise RuntimeError(f"GCP token failed: {token_resp.text[:200]}")
            access_token = token_resp.json()["access_token"]

            base_url = f"https://run.googleapis.com/v2/projects/{self._config.gcp_project_id}/locations"
            main_url = f"{base_url}/{self._config.gcp_region}/services/{self._config.gcp_service_name}"

            # Read main service definition to clone structure
            main_resp = await client.get(main_url, headers={"Authorization": f"Bearer {access_token}"})
            if main_resp.status_code != 200:
                raise RuntimeError(f"Cannot read main Cloud Run service: {main_resp.status_code}")

            service_def = main_resp.json()

            # Strip identity + override with shadow-specific env vars
            containers = (
                service_def.get("template", {}).get("template", {}).get("containers", [])
            )
            if containers:
                env_vars = [
                    e for e in containers[0].get("env", [])
                    if e.get("name") not in ("ORGANISM_SKIA_RESTORE_CID", "ORGANISM_MIGRATION_ID")
                ]
                env_vars.extend([
                    {"name": "ORGANISM_STANDALONE", "value": "true"},
                    {"name": "ORGANISM_SHADOW_WORKER", "value": "true"},
                    {"name": "SKIA_CID_REDIS_KEY", "value": self._config.state_cid_redis_key},
                ])
                # Limit resources: min instances = 0, max = 1 (free tier)
                containers[0]["env"] = env_vars
                if "resources" in containers[0]:
                    containers[0]["resources"]["limits"] = {"cpu": "1", "memory": "256Mi"}

            service_def["name"] = f"projects/{self._config.gcp_project_id}/locations/{shadow_region}/services/{shadow_service}"

            shadow_url = f"{base_url}/{shadow_region}/services"
            create_resp = await client.post(
                shadow_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                params={"serviceId": shadow_service},
                content=__import__("orjson").dumps(service_def).decode(),
            )
            if create_resp.status_code not in (200, 201):
                # Service may already exist - try PATCH
                patch_url = f"{base_url}/{shadow_region}/services/{shadow_service}"
                patch_resp = await client.patch(
                    patch_url,
                    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                    content=__import__("orjson").dumps(service_def).decode(),
                )
                if patch_resp.status_code not in (200, 201):
                    raise RuntimeError(
                        f"Cloud Run shadow deploy failed: {patch_resp.status_code} {patch_resp.text[:200]}"
                    )
                service_data = patch_resp.json()
            else:
                service_data = create_resp.json()

            endpoint = service_data.get("uri") or service_data.get("status", {}).get("url", "")
            return endpoint, "gcp_shadow", shadow_service

    # ── Shadow Worker Health Check ─────────────────────────────────────────────

    async def _check_shadow_worker_health(self, endpoint: str) -> bool:
        """Ping the shadow worker's health endpoint. Returns True if reachable."""
        if not endpoint:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(endpoint.rstrip("/") + "/health")
                return resp.status_code == 200
        except Exception:
            return False

    # ── Shadow Worker Redis Record ────────────────────────────────────────────

    async def _persist_shadow_worker_record(
        self,
        endpoint: str,
        provider: str,
        deployment_id: str,
    ) -> None:
        """Store shadow worker metadata in Redis for discovery on reboot."""
        try:
            raw = self._redis.client
            import orjson
            data = orjson.dumps({
                "endpoint": endpoint,
                "provider": provider,
                "deployment_id": deployment_id,
                "deployed_at": time.time(),
            })
            await raw.set(f"eos:{self._SHADOW_WORKER_KEY}", data, ex=self._SHADOW_WORKER_TTL_S)
        except Exception as exc:
            self._log.warning("shadow_worker_record_persist_failed", error=str(exc))

    async def _get_shadow_worker_record(self) -> dict[str, Any] | None:
        """Retrieve shadow worker metadata from Redis."""
        try:
            raw = self._redis.client
            data = await raw.get(f"eos:{self._SHADOW_WORKER_KEY}")
            if data is None:
                return None
            import orjson
            return orjson.loads(data)
        except Exception:
            return None


# ── Module-level helpers ──────────────────────────────────────────────────────


def _patch_sdl_for_shadow(sdl_content: str) -> str:
    """
    Downscale an Akash SDL template to shadow-worker resource levels.

    Replaces common resource spec patterns with minimal values so the shadow
    worker stays within free-tier / cheap-tier Akash resource units.
    This is a best-effort string-level patch - operators should maintain a
    dedicated shadow SDL template at config/skia/akash_shadow_sdl_template.yaml
    for production use.
    """
    import re

    # Downscale CPU: any "units: X" > 0.5 → 0.1
    def _clamp_cpu(m: re.Match[str]) -> str:
        raw = m.group(1).strip()
        try:
            val = float(raw)
        except ValueError:
            return m.group(0)
        return f"units: {min(val, 0.1)}"

    sdl_content = re.sub(r"units:\s*([\d.]+)", _clamp_cpu, sdl_content)

    # Downscale memory: any "size: XGi/XMi" > 128Mi → 128Mi
    def _clamp_memory(m: re.Match[str]) -> str:
        raw = m.group(1)
        if raw.endswith("Gi"):
            return "size: 128Mi"
        try:
            mib = int(raw.replace("Mi", "").replace("M", ""))
            return f"size: {min(mib, 128)}Mi"
        except ValueError:
            return m.group(0)

    sdl_content = re.sub(r"size:\s*([\d]+(?:Gi|Mi|M))", _clamp_memory, sdl_content)
    return sdl_content


def _pick_shadow_region(main_region: str) -> str:
    """
    Return a Cloud Run region that differs from the main instance region.

    Provides geographic redundancy: the shadow worker is never co-located
    with the main instance in the same region.
    """
    _FALLBACK_CHAIN: dict[str, str] = {
        "australia-southeast1": "us-central1",
        "australia-southeast2": "us-central1",
        "us-central1": "europe-west1",
        "us-east1": "europe-west1",
        "us-west1": "europe-west1",
        "europe-west1": "us-central1",
        "europe-west2": "us-central1",
        "europe-west3": "us-central1",
        "asia-northeast1": "us-central1",
        "asia-southeast1": "us-central1",
    }
    return _FALLBACK_CHAIN.get(main_region, "us-central1")
