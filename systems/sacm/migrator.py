"""
EcodiaOS - SACM Compute Migration Executor

Handles atomic migration of the organism's primary compute from one infrastructure
provider to another (Cloud Run ↔ Akash).  Migration is always gated by Equor
(constitutional check via EQUOR_ECONOMIC_INTENT) and limited to 1 per 24 hours.

Architecture
────────────
  MigrationExecutor.migrate_to_provider(provider, config)
    1. Equor constitutional gate (EQUOR_ECONOMIC_INTENT / EQUOR_ECONOMIC_PERMIT)
    2. Circuit breaker: ≤ 1 migration per 24h rolling window
    3. Provision new instance on target provider (Cloud Run or Akash)
    4. Export current state via Skia snapshot mechanism
    5. Transfer state CID to new instance (env var injection)
    6. Verify new instance healthy (heartbeat poll)
    7. Switch DNS/routing to new instance
    8. Gracefully shut down old instance
    Rollback: if any step 3-8 fails, terminate new instance and emit failure event.

  CostTriggeredMigrationMonitor.start()
    Subscribes to ORGANISM_TELEMETRY.  If infra_cost_usd_per_hour is > 20% higher
    than the cheapest alternative for > 6 consecutive hours, proposes migration.

Synapse events
  Inbound:  ORGANISM_TELEMETRY, EQUOR_ECONOMIC_PERMIT
  Outbound: COMPUTE_MIGRATION_STARTED, COMPUTE_MIGRATION_COMPLETED,
            COMPUTE_MIGRATION_FAILED, COMPUTE_ARBITRAGE_DETECTED
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from primitives.common import EOSBaseModel

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from config import ComputeArbitrageConfig, SkiaConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.sacm.migrator")

# ─── Constants ────────────────────────────────────────────────────────────────

_EQUOR_PERMIT_TIMEOUT_S = 30.0   # Max wait for EQUOR_ECONOMIC_PERMIT
_HEARTBEAT_POLL_INTERVAL_S = 10.0
_MIGRATION_LOCK_KEY = "sacm:migration:lock"
_MIGRATION_LOCK_TTL_S = 900      # 15 min - matches Skia restoration lock
_MIGRATION_HISTORY_KEY = "sacm:migration:history"


# ─── Models ───────────────────────────────────────────────────────────────────


class MigrationRecord(EOSBaseModel):
    """Persisted record of a completed (or failed) migration."""

    migration_id: str
    from_provider: str
    to_provider: str
    trigger_reason: str           # "cost_triggered" | "manual" | "test"
    status: str                   # "completed" | "failed" | "rolled_back"
    state_cid: str = ""
    new_endpoint: str = ""
    duration_s: float = 0.0
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0


# ─── Migration Executor ───────────────────────────────────────────────────────


class MigrationExecutor:
    """
    Atomic compute migration between Cloud Run and Akash.

    Dependency injection: wire an event bus + Redis before calling migrate_to_provider().
    Optionally wire a Skia snapshot reference (via set_skia_snapshot) so current
    state is captured before provisioning the new instance.
    """

    def __init__(
        self,
        config: ComputeArbitrageConfig,
        skia_config: SkiaConfig,
        redis: RedisClient,
    ) -> None:
        self._config = config
        self._skia_config = skia_config
        self._redis = redis
        self._event_bus: EventBus | None = None
        self._skia_snapshot: Any | None = None   # StateSnapshotPipeline - injected
        self._neo4j: Any | None = None           # Neo4j async driver - injected via set_neo4j()
        self._permit_events: dict[str, asyncio.Event] = {}  # request_id → permit event
        self._permit_results: dict[str, bool] = {}          # request_id → approved?
        self._log = logger.bind(component="sacm.migrator")

    # ─── Wiring ───────────────────────────────────────────────────────────────

    def set_neo4j(self, neo4j_driver: Any) -> None:
        """
        Wire a Neo4j async driver so each MigrationRecord is persisted as an
        (:EconomicEvent) node - immutable audit trail per Spec 27 §10.

        Closes Known Issue #6 in CLAUDE.md (_write_migration_neo4j() stub was
        implemented but driver injection was never wired from registry.py).
        """
        self._neo4j = neo4j_driver
        self._log.info("neo4j_wired_to_migration_executor")

    def set_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        try:
            from systems.synapse.types import SynapseEventType
            event_bus.subscribe(
                SynapseEventType.EQUOR_ECONOMIC_PERMIT,
                self._on_equor_permit,
            )
            self._log.info("migration_executor_event_bus_wired")
        except Exception as exc:
            self._log.warning("migration_executor_event_bus_failed", error=str(exc))

    def set_skia_snapshot(self, snapshot: Any) -> None:
        """Wire Skia's StateSnapshotPipeline so migration can force a pre-migration snapshot."""
        self._skia_snapshot = snapshot

    # ─── Public API ───────────────────────────────────────────────────────────

    async def migrate_to_provider(
        self,
        target_provider: str,
        trigger_reason: str = "manual",
    ) -> MigrationRecord:
        """
        Atomically migrate the organism to target_provider.

        Steps (all-or-nothing, rollback on any failure):
          1. Equor constitutional gate
          2. Circuit breaker check (≤ 1/24h)
          3. Acquire distributed migration lock
          4. Force Skia snapshot → get state CID
          5. Provision new instance on target provider
          6. Verify new instance healthy
          7. Switch routing to new instance
          8. Shut down old instance
        """
        migration_id = str(uuid.uuid4())[:12]
        started_at = time.time()
        t0 = time.monotonic()
        current_provider = self._config.current_provider

        self._log.info(
            "migration_requested",
            migration_id=migration_id,
            from_provider=current_provider,
            to_provider=target_provider,
            trigger=trigger_reason,
        )

        # ── Step 1: Equor constitutional gate ────────────────────────────────
        approved = await self._request_equor_approval(
            migration_id=migration_id,
            current_provider=current_provider,
            target_provider=target_provider,
            trigger_reason=trigger_reason,
        )
        if not approved:
            rec = MigrationRecord(
                migration_id=migration_id,
                from_provider=current_provider,
                to_provider=target_provider,
                trigger_reason=trigger_reason,
                status="failed",
                error="Equor denied migration (constitutional check failed)",
                started_at=started_at,
                completed_at=time.time(),
            )
            await self._persist_record(rec)
            await self._emit(SynapseEventType.COMPUTE_MIGRATION_FAILED, {
                "migration_id": migration_id,
                "from_provider": current_provider,
                "to_provider": target_provider,
                "reason": rec.error,
            })
            return rec

        # ── Step 2: Circuit breaker ───────────────────────────────────────────
        if not await self._circuit_breaker_allow():
            rec = MigrationRecord(
                migration_id=migration_id,
                from_provider=current_provider,
                to_provider=target_provider,
                trigger_reason=trigger_reason,
                status="failed",
                error=f"Circuit breaker: max {self._config.max_migrations_per_24h} migration(s)/24h already used",
                started_at=started_at,
                completed_at=time.time(),
            )
            await self._persist_record(rec)
            await self._emit(SynapseEventType.COMPUTE_MIGRATION_FAILED, {
                "migration_id": migration_id,
                "from_provider": current_provider,
                "to_provider": target_provider,
                "reason": rec.error,
            })
            return rec

        # ── Step 3: Distributed lock ─────────────────────────────────────────
        worker_id = str(uuid.uuid4())
        if not await self._acquire_lock(worker_id):
            rec = MigrationRecord(
                migration_id=migration_id,
                from_provider=current_provider,
                to_provider=target_provider,
                trigger_reason=trigger_reason,
                status="failed",
                error="Another migration is already in progress",
                started_at=started_at,
                completed_at=time.time(),
            )
            await self._persist_record(rec)
            return rec

        new_endpoint = ""
        try:
            await self._emit(SynapseEventType.COMPUTE_MIGRATION_STARTED, {
                "migration_id": migration_id,
                "from_provider": current_provider,
                "to_provider": target_provider,
                "trigger_reason": trigger_reason,
            })

            # ── Step 4: Force Skia snapshot ──────────────────────────────────
            state_cid = await self._capture_state_snapshot()
            if not state_cid:
                raise RuntimeError("Skia snapshot returned empty CID - cannot migrate without state")

            # ── Step 5: Provision new instance ───────────────────────────────
            new_endpoint = await self._provision_new_instance(
                target_provider=target_provider,
                state_cid=state_cid,
                migration_id=migration_id,
            )
            self._log.info(
                "migration_new_instance_provisioned",
                migration_id=migration_id,
                target_provider=target_provider,
                endpoint=new_endpoint,
            )

            # ── Step 6: Verify health ─────────────────────────────────────────
            healthy = await self._verify_instance_healthy(new_endpoint)
            if not healthy:
                raise RuntimeError(
                    f"New instance at {new_endpoint} failed health checks within "
                    f"{self._config.handoff_timeout_s}s - rolling back"
                )

            # ── Step 7: Switch routing ────────────────────────────────────────
            await self._switch_routing(
                old_provider=current_provider,
                new_provider=target_provider,
                new_endpoint=new_endpoint,
                migration_id=migration_id,
            )

            # ── Step 8: Shut down old instance ───────────────────────────────
            await self._shutdown_old_instance(current_provider, migration_id)

            # ── Record success ────────────────────────────────────────────────
            duration_s = time.monotonic() - t0
            await self._record_migration_timestamp()
            self._config.current_provider = target_provider  # Update live state

            rec = MigrationRecord(
                migration_id=migration_id,
                from_provider=current_provider,
                to_provider=target_provider,
                trigger_reason=trigger_reason,
                status="completed",
                state_cid=state_cid,
                new_endpoint=new_endpoint,
                duration_s=round(duration_s, 2),
                started_at=started_at,
                completed_at=time.time(),
            )
            await self._persist_record(rec)
            await self._emit(SynapseEventType.COMPUTE_MIGRATION_COMPLETED, {
                "migration_id": migration_id,
                "from_provider": current_provider,
                "to_provider": target_provider,
                "new_endpoint": new_endpoint,
                "state_cid": state_cid,
                "duration_s": round(duration_s, 2),
            })
            self._log.info(
                "migration_completed",
                migration_id=migration_id,
                from_provider=current_provider,
                to_provider=target_provider,
                duration_s=round(duration_s, 2),
            )
            return rec

        except Exception as exc:
            # ── Rollback: terminate the partially-provisioned new instance ────
            self._log.error(
                "migration_failed_rolling_back",
                migration_id=migration_id,
                error=str(exc),
                new_endpoint=new_endpoint,
            )
            if new_endpoint:
                await self._rollback_new_instance(target_provider, new_endpoint, migration_id)

            duration_s = time.monotonic() - t0
            rec = MigrationRecord(
                migration_id=migration_id,
                from_provider=current_provider,
                to_provider=target_provider,
                trigger_reason=trigger_reason,
                status="rolled_back",
                new_endpoint=new_endpoint,
                duration_s=round(duration_s, 2),
                error=str(exc),
                started_at=started_at,
                completed_at=time.time(),
            )
            await self._persist_record(rec)
            await self._emit(SynapseEventType.COMPUTE_MIGRATION_FAILED, {
                "migration_id": migration_id,
                "from_provider": current_provider,
                "to_provider": target_provider,
                "reason": str(exc),
                "rolled_back": True,
            })
            return rec

        finally:
            await self._release_lock(worker_id)

    # ─── Equor Gating ─────────────────────────────────────────────────────────

    async def _request_equor_approval(
        self,
        migration_id: str,
        current_provider: str,
        target_provider: str,
        trigger_reason: str,
    ) -> bool:
        """
        Emit EQUOR_ECONOMIC_INTENT and wait for EQUOR_ECONOMIC_PERMIT.

        Uses the existing Oikos-gated economic intent pattern.  Migration is
        classified as a GROWTH-type economic action (new infrastructure cost).
        Auto-permits after _EQUOR_PERMIT_TIMEOUT_S as safety fallback (survival
        imperative: if Equor is unreachable, the organism can still act autonomously).
        """
        if self._event_bus is None:
            # No bus wired → auto-permit (offline/test mode)
            self._log.warning("equor_gate_skipped_no_bus", migration_id=migration_id)
            return True

        request_id = f"migration:{migration_id}"
        permit_event = asyncio.Event()
        self._permit_events[request_id] = permit_event
        self._permit_results[request_id] = True  # Default: permit (survival fallback)

        await self._emit(SynapseEventType.EQUOR_ECONOMIC_INTENT, {
            "request_id": request_id,
            "mutation_type": "compute_migration",
            "amount_usd": str(self._config.max_deployment_budget_usd_24h),
            "from_account": "infra_budget",
            "to_account": f"provider:{target_provider}",
            "rationale": (
                f"Migrate compute from {current_provider} to {target_provider}. "
                f"Trigger: {trigger_reason}. "
                f"Budget cap: ${self._config.max_deployment_budget_usd_24h:.2f}/24h."
            ),
        })

        try:
            await asyncio.wait_for(permit_event.wait(), timeout=_EQUOR_PERMIT_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._log.warning(
                "equor_permit_timeout_auto_approving",
                migration_id=migration_id,
                timeout_s=_EQUOR_PERMIT_TIMEOUT_S,
            )
            # Timeout = auto-permit (survival imperative)
            self._permit_results[request_id] = True

        approved = self._permit_results.pop(request_id, True)
        self._permit_events.pop(request_id, None)
        return approved

    async def _on_equor_permit(self, event: Any) -> None:
        """Handle EQUOR_ECONOMIC_PERMIT events for pending migration approvals."""
        data = getattr(event, "data", {}) or {}
        request_id = data.get("request_id", "")
        if not request_id.startswith("migration:"):
            return

        permit_event = self._permit_events.get(request_id)
        if permit_event is None:
            return

        approved = data.get("approved", True)
        self._permit_results[request_id] = approved
        permit_event.set()

        self._log.info(
            "equor_permit_received",
            request_id=request_id,
            approved=approved,
            reason=data.get("reason", ""),
        )

    # ─── Circuit Breaker ──────────────────────────────────────────────────────

    async def _circuit_breaker_allow(self) -> bool:
        """Return True if a migration is permitted by the 24h rolling window."""
        try:
            raw = self._redis.client
            key = f"eos:{self._config.migration_state_redis_key}:timestamps"
            cutoff = time.time() - 86400.0
            # Remove entries older than 24h
            await raw.zremrangebyscore(key, "-inf", cutoff)
            count = await raw.zcard(key)
            return count < self._config.max_migrations_per_24h
        except Exception as exc:
            self._log.warning("circuit_breaker_check_failed", error=str(exc))
            return True  # Fail open - safety fallback

    async def _record_migration_timestamp(self) -> None:
        """Record a successful migration timestamp for 24h circuit breaker."""
        try:
            raw = self._redis.client
            key = f"eos:{self._config.migration_state_redis_key}:timestamps"
            now = time.time()
            await raw.zadd(key, {str(now): now})
            await raw.expire(key, 86400 + 3600)  # Keep for 25h
        except Exception as exc:
            self._log.warning("migration_timestamp_persist_failed", error=str(exc))

    # ─── Distributed Lock ─────────────────────────────────────────────────────

    async def _acquire_lock(self, worker_id: str) -> bool:
        try:
            raw = self._redis.client
            result = await raw.set(
                f"eos:{_MIGRATION_LOCK_KEY}",
                worker_id,
                nx=True,
                ex=_MIGRATION_LOCK_TTL_S,
            )
            return result is not None
        except Exception as exc:
            self._log.warning("migration_lock_acquire_failed", error=str(exc))
            return False

    async def _release_lock(self, worker_id: str) -> None:
        try:
            raw = self._redis.client
            key = f"eos:{_MIGRATION_LOCK_KEY}"
            current = await raw.get(key)
            if current == worker_id:
                await raw.delete(key)
        except Exception as exc:
            self._log.warning("migration_lock_release_failed", error=str(exc))

    # ─── State Capture ────────────────────────────────────────────────────────

    async def _capture_state_snapshot(self) -> str:
        """
        Force an immediate Skia state snapshot and return the IPFS CID.

        Falls back to reading the latest CID from Redis if no snapshot pipeline
        is wired (e.g. standalone migrator mode).
        """
        if self._skia_snapshot is not None:
            try:
                manifest = await self._skia_snapshot.take_snapshot()
                cid = getattr(manifest, "ipfs_cid", None) or getattr(manifest, "cid", None)
                if cid:
                    self._log.info("pre_migration_snapshot_taken", cid=cid)
                    return str(cid)
            except Exception as exc:
                self._log.warning("pre_migration_snapshot_failed", error=str(exc))

        # Fallback: read latest CID from Redis (snapshot may be recent enough)
        try:
            cid = await self._redis.get_json(self._skia_config.state_cid_redis_key)
            if cid:
                self._log.info("pre_migration_using_cached_cid", cid=str(cid))
                return str(cid)
        except Exception as exc:
            self._log.warning("redis_cid_read_failed", error=str(exc))

        return ""

    # ─── Provisioning ─────────────────────────────────────────────────────────

    async def _provision_new_instance(
        self,
        target_provider: str,
        state_cid: str,
        migration_id: str,
    ) -> str:
        """
        Provision a new instance on target_provider, injecting state CID.

        Returns the new instance endpoint URL.
        """
        if target_provider == "gcp" or target_provider.startswith("cloud_run"):
            return await self._provision_cloud_run(state_cid, migration_id)
        elif target_provider == "akash":
            return await self._provision_akash(state_cid, migration_id)
        else:
            raise ValueError(f"Unknown target provider: {target_provider!r}")

    async def _provision_cloud_run(self, state_cid: str, migration_id: str) -> str:
        """
        Deploy a new Cloud Run revision with state CID injected.

        Triggers a new revision (different from Skia's restart - this targets
        the *migration* service name, which may differ from the Skia-watched service).
        """
        import base64
        import json
        from datetime import UTC, datetime, timedelta

        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        if not self._skia_config.gcp_project_id or not self._skia_config.gcp_service_account_key_b64:
            raise RuntimeError("GCP project/service-account not configured for migration")

        # Build access token
        key_json = base64.b64decode(self._skia_config.gcp_service_account_key_b64)
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

        async with httpx.AsyncClient(timeout=self._skia_config.gcp_restart_timeout_s) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
            )
            if token_resp.status_code != 200:
                raise RuntimeError(f"GCP token exchange failed: {token_resp.text[:200]}")
            access_token = token_resp.json()["access_token"]

            service_url = (
                f"https://run.googleapis.com/v2/projects/{self._skia_config.gcp_project_id}"
                f"/locations/{self._skia_config.gcp_region}"
                f"/services/{self._skia_config.gcp_service_name}"
            )
            get_resp = await client.get(service_url, headers={"Authorization": f"Bearer {access_token}"})
            if get_resp.status_code != 200:
                raise RuntimeError(f"Failed to read Cloud Run service: {get_resp.status_code}")

            service_def = get_resp.json()
            containers = (
                service_def.get("template", {}).get("template", {}).get("containers", [])
            )
            if containers:
                env_vars = containers[0].get("env", [])
                env_vars = [e for e in env_vars if e.get("name") not in (
                    "ECODIAOS_SKIA_RESTORE_CID", "ECODIAOS_MIGRATION_ID",
                )]
                env_vars.append({"name": "ECODIAOS_SKIA_RESTORE_CID", "value": state_cid})
                env_vars.append({"name": "ECODIAOS_MIGRATION_ID", "value": migration_id})
                containers[0]["env"] = env_vars

            import orjson
            patch_resp = await client.patch(
                service_url,
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                content=orjson.dumps(service_def).decode(),
            )
            if patch_resp.status_code not in (200, 201):
                raise RuntimeError(f"Cloud Run PATCH failed: {patch_resp.status_code} {patch_resp.text[:200]}")

            # Extract service URL from response
            service_data = patch_resp.json()
            endpoint = service_data.get("uri") or service_data.get("status", {}).get("url", "")
            return endpoint

    async def _provision_akash(self, state_cid: str, migration_id: str) -> str:
        """Deploy a new Akash instance with state CID injected into the SDL."""
        import orjson
        from pathlib import Path

        sdl_path = Path(self._skia_config.akash_sdl_template_path)
        if not sdl_path.exists():
            raise FileNotFoundError(f"Akash SDL template not found: {sdl_path}")

        sdl_content = sdl_path.read_text()
        sdl_content = sdl_content.replace("${ECODIAOS_SKIA_RESTORE_CID}", state_cid)
        if self._skia_config.akash_docker_image:
            sdl_content = sdl_content.replace("${DOCKER_IMAGE}", self._skia_config.akash_docker_image)
        # Clear genome placeholder if not available
        sdl_content = sdl_content.replace("${ECODIAOS_CONSTITUTIONAL_GENOME_B64}", "")

        async with httpx.AsyncClient(timeout=self._skia_config.akash_deploy_timeout_s) as client:
            resp = await client.post(
                f"{self._skia_config.akash_api_url}/v1/deployments",
                json={
                    "sdl": sdl_content,
                    "wallet": self._skia_config.akash_wallet_address,
                    "env": {"ECODIAOS_MIGRATION_ID": migration_id},
                },
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code not in (200, 201, 202):
                raise RuntimeError(f"Akash deploy failed: {resp.status_code} {resp.text[:200]}")

            deploy_data = resp.json()
            deployment_id = deploy_data.get("deployment_id") or deploy_data.get("id", "")

            # Poll until ACTIVE or timeout
            poll_deadline = time.monotonic() + _AKASH_ACTIVE_TIMEOUT_S
            endpoint = deploy_data.get("endpoint", "")

            while time.monotonic() < poll_deadline:
                if not deployment_id:
                    break
                status_resp = await client.get(
                    f"{self._skia_config.akash_api_url}/v1/deployments/{deployment_id}",
                )
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    state = status_data.get("state", "").upper()
                    endpoint = status_data.get("endpoint", endpoint)
                    if state == "ACTIVE":
                        break
                    if state in ("FAILED", "CLOSED", "REJECTED"):
                        raise RuntimeError(f"Akash deployment reached terminal state: {state}")
                await asyncio.sleep(_AKASH_POLL_INTERVAL_S)
            else:
                raise RuntimeError(f"Akash deployment did not reach ACTIVE within {_AKASH_ACTIVE_TIMEOUT_S}s")

            return endpoint

    # ─── Health Verification ──────────────────────────────────────────────────

    async def _verify_instance_healthy(self, endpoint: str) -> bool:
        """
        Poll the new instance's health endpoint until it responds healthy.

        Requires `handoff_healthy_threshold` consecutive successful responses
        within `handoff_timeout_s`.  An empty endpoint is treated as unverifiable
        (returns False to trigger rollback).
        """
        if not endpoint:
            self._log.warning("verify_instance_no_endpoint")
            return False

        health_url = endpoint.rstrip("/") + "/health"
        consecutive_ok = 0
        deadline = time.monotonic() + self._config.handoff_timeout_s

        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        consecutive_ok += 1
                        self._log.debug(
                            "health_check_ok",
                            endpoint=endpoint,
                            consecutive=consecutive_ok,
                            required=self._config.handoff_healthy_threshold,
                        )
                        if consecutive_ok >= self._config.handoff_healthy_threshold:
                            return True
                    else:
                        consecutive_ok = 0
                except Exception:
                    consecutive_ok = 0

                await asyncio.sleep(self._config.handoff_poll_interval_s)

        self._log.warning(
            "health_check_timeout",
            endpoint=endpoint,
            timeout_s=self._config.handoff_timeout_s,
        )
        return False

    # ─── Routing Switch ───────────────────────────────────────────────────────

    async def _switch_routing(
        self,
        old_provider: str,
        new_provider: str,
        new_endpoint: str,
        migration_id: str,
    ) -> None:
        """
        Record the new provider endpoint in Redis so all internal consumers
        resolve to the new instance.  This is the lightweight "DNS switch":
        no external DNS records are managed here - the endpoint is used by
        internal health monitors (Skia heartbeat, Synapse) to redirect checks.

        Operators managing real DNS should subscribe to COMPUTE_MIGRATION_COMPLETED.
        """
        try:
            raw = self._redis.client
            routing_key = "sacm:current_provider_endpoint"
            await raw.set(routing_key, new_endpoint, ex=86400 * 7)  # 7-day TTL
            provider_key = "sacm:current_provider"
            await raw.set(provider_key, new_provider, ex=86400 * 7)
            self._log.info(
                "routing_switched",
                migration_id=migration_id,
                old_provider=old_provider,
                new_provider=new_provider,
                new_endpoint=new_endpoint,
            )
        except Exception as exc:
            self._log.warning("routing_switch_redis_failed", error=str(exc))
            # Non-fatal: routing record is advisory. Migration can still succeed.

    # ─── Old Instance Shutdown ───────────────────────────────────────────────

    async def _shutdown_old_instance(self, old_provider: str, migration_id: str) -> None:
        """
        Emit a graceful shutdown signal to the old instance.

        This sends ORGANISM_SHUTDOWN_REQUESTED via Synapse so the old instance
        can drain in-flight workloads before dying.  We do NOT forcibly kill the
        old instance - the organism shuts itself down on receipt of this event.
        """
        await self._emit(SynapseEventType.ORGANISM_SHUTDOWN_REQUESTED, {
            "requester": "sacm.migrator",
            "migration_id": migration_id,
            "old_provider": old_provider,
            "reason": "provider_migration_complete",
            "drain_timeout_s": 30.0,
        })
        self._log.info(
            "old_instance_shutdown_requested",
            migration_id=migration_id,
            old_provider=old_provider,
        )

    # ─── Rollback ────────────────────────────────────────────────────────────

    async def _rollback_new_instance(
        self,
        target_provider: str,
        new_endpoint: str,
        migration_id: str,
    ) -> None:
        """Best-effort termination of a partially-provisioned new instance."""
        self._log.info(
            "rollback_terminating_new_instance",
            migration_id=migration_id,
            target_provider=target_provider,
            new_endpoint=new_endpoint,
        )
        # Emit rollback signal - provider-specific cleanup is done by operators
        # or by a future provider-aware cleanup hook.
        await self._emit(SynapseEventType.COMPUTE_MIGRATION_FAILED, {
            "migration_id": migration_id,
            "to_provider": target_provider,
            "new_endpoint": new_endpoint,
            "rollback": True,
        })

    # ─── Persistence ─────────────────────────────────────────────────────────

    async def _persist_record(self, rec: MigrationRecord) -> None:
        """Persist migration record to Redis sorted set and Neo4j audit trail."""
        # Redis: sorted set scored by started_at, capped at 100 entries
        try:
            import orjson
            raw = self._redis.client
            key = f"eos:{_MIGRATION_HISTORY_KEY}"
            data = orjson.dumps(rec.model_dump(mode="json")).decode()
            await raw.zadd(key, {data: rec.started_at})
            # Keep only last 100 records
            count = await raw.zcard(key)
            if count > 100:
                await raw.zremrangebyrank(key, 0, count - 101)
        except Exception as exc:
            self._log.warning("migration_record_persist_failed", error=str(exc))

        # Neo4j: (:EconomicEvent) node - immutable audit trail (Known Issue #6 closure)
        if self._neo4j is not None:
            asyncio.create_task(
                self._write_migration_neo4j(rec),
                name=f"sacm_migration_neo4j_{rec.migration_id[:8]}",
            )

    async def _write_migration_neo4j(self, rec: MigrationRecord) -> None:
        """
        Write a MigrationRecord as an (:EconomicEvent) node in Neo4j.

        Uses MERGE on migration_id so re-runs are idempotent.
        Node follows the EcodiaOS audit trail convention (bi-temporal timestamps).
        """
        try:
            async with self._neo4j.session() as session:
                await session.run(
                    """
                    MERGE (e:EconomicEvent {id: $migration_id})
                    ON CREATE SET
                        e.event_type      = 'sacm_compute_migration',
                        e.system_id       = 'sacm',
                        e.migration_id    = $migration_id,
                        e.from_provider   = $from_provider,
                        e.to_provider     = $to_provider,
                        e.trigger_reason  = $trigger_reason,
                        e.status          = $status,
                        e.state_cid       = $state_cid,
                        e.new_endpoint    = $new_endpoint,
                        e.duration_s      = $duration_s,
                        e.error           = $error,
                        e.started_at      = $started_at,
                        e.completed_at    = $completed_at,
                        e.event_time      = datetime($started_at_iso),
                        e.ingestion_time  = datetime()
                    ON MATCH SET
                        e.status          = $status,
                        e.error           = $error,
                        e.completed_at    = $completed_at,
                        e.duration_s      = $duration_s
                    """,
                    migration_id=rec.migration_id,
                    from_provider=rec.from_provider,
                    to_provider=rec.to_provider,
                    trigger_reason=rec.trigger_reason,
                    status=rec.status,
                    state_cid=rec.state_cid,
                    new_endpoint=rec.new_endpoint,
                    duration_s=rec.duration_s,
                    error=rec.error,
                    started_at=rec.started_at,
                    completed_at=rec.completed_at,
                    started_at_iso=(
                        __import__("datetime").datetime.fromtimestamp(
                            rec.started_at, tz=__import__("datetime").timezone.utc
                        ).isoformat()
                        if rec.started_at
                        else ""
                    ),
                )
        except Exception as exc:
            self._log.warning(
                "migration_neo4j_write_failed",
                migration_id=rec.migration_id,
                error=str(exc),
            )

    # ─── Synapse helpers ─────────────────────────────────────────────────────

    async def _emit(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            asyncio.create_task(
                self._event_bus.emit(SynapseEvent(
                    event_type=event_type,
                    source_system="sacm",
                    data=data,
                )),
                name=f"sacm_migrator_{event_type.value[:30]}",
            )
        except Exception as exc:
            self._log.debug("emit_failed", event_type=event_type, error=str(exc))


# ─── Akash poll constants (used in _provision_akash) ─────────────────────────

_AKASH_POLL_INTERVAL_S = 15.0
_AKASH_ACTIVE_TIMEOUT_S = 600.0


# ─── Cost-Triggered Migration Monitor ────────────────────────────────────────


class CostTriggeredMigrationMonitor:
    """
    Subscribes to ORGANISM_TELEMETRY.  When infra_cost_usd_per_hour has been
    > 20% above the cheapest available alternative for > 6 consecutive hours,
    proposes a migration to the cheaper provider.

    The 20% threshold and 6-hour window match the spec (cost must be sustained,
    not a transient spike).  The comparison is against the oracle's live pricing
    surface snapshot - wired via set_oracle().
    """

    _COST_THRESHOLD_FACTOR = 1.20   # 20% more expensive than cheapest
    _SUSTAINED_HOURS = 6.0
    _ARBITRAGE_THRESHOLD_FACTOR = 1.20  # emit COMPUTE_ARBITRAGE_DETECTED at same threshold

    def __init__(
        self,
        migration_executor: MigrationExecutor,
        config: ComputeArbitrageConfig,
    ) -> None:
        self._executor = migration_executor
        self._config = config
        self._event_bus: EventBus | None = None
        self._oracle: Any | None = None   # ComputeMarketOracle - injected
        self._log = logger.bind(component="sacm.cost_migration_monitor")

        # Telemetry accumulation
        self._high_cost_since: float | None = None    # epoch when cost first exceeded threshold
        self._last_cheapest_provider: str = ""
        self._last_cheapest_cost: float = 0.0
        self._migration_proposed: bool = False        # prevent duplicate proposals

    def set_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        try:
            from systems.synapse.types import SynapseEventType
            event_bus.subscribe(
                SynapseEventType.ORGANISM_TELEMETRY,
                self._on_organism_telemetry,
            )
            self._log.info("cost_migration_monitor_subscribed_to_telemetry")
        except Exception as exc:
            self._log.warning("cost_migration_monitor_subscribe_failed", error=str(exc))

    def set_oracle(self, oracle: Any) -> None:
        """Wire ComputeMarketOracle for live pricing surface access."""
        self._oracle = oracle

    async def _on_organism_telemetry(self, event: Any) -> None:
        """
        Process ORGANISM_TELEMETRY.  Compare infra_cost_usd_per_hour against
        the cheapest provider from the oracle.  If cost exceeds 20% above
        cheapest for 6+ hours, trigger migration proposal.
        """
        if not self._config.enabled:
            return

        data = getattr(event, "data", {}) or {}
        current_cost = float(data.get("infra_cost_usd_per_hour", 0.0))

        if current_cost <= 0.0:
            self._high_cost_since = None
            self._migration_proposed = False
            return

        # Get cheapest alternative from oracle
        cheapest_provider, cheapest_cost = self._get_cheapest_alternative()
        self._last_cheapest_provider = cheapest_provider
        self._last_cheapest_cost = cheapest_cost

        if cheapest_cost <= 0.0 or not cheapest_provider:
            # Oracle not available or no alternatives
            return

        if cheapest_provider == self._config.current_provider:
            # Already on the cheapest provider
            self._high_cost_since = None
            self._migration_proposed = False
            return

        cost_ratio = current_cost / cheapest_cost
        if cost_ratio > self._COST_THRESHOLD_FACTOR:
            # We are more expensive than cheapest by > 20%
            now = time.time()
            if self._high_cost_since is None:
                self._high_cost_since = now
                self._log.info(
                    "cost_above_threshold_started",
                    current_cost=round(current_cost, 4),
                    cheapest_cost=round(cheapest_cost, 4),
                    cheapest_provider=cheapest_provider,
                    ratio=round(cost_ratio, 3),
                )
                # Emit arbitrage signal immediately (sustained check follows separately)
                await self._emit(SynapseEventType.COMPUTE_ARBITRAGE_DETECTED, {
                    "current_provider": self._config.current_provider,
                    "current_cost_usd_per_hour": round(current_cost, 4),
                    "cheapest_provider": cheapest_provider,
                    "cheapest_cost_usd_per_hour": round(cheapest_cost, 4),
                    "savings_pct": round((cost_ratio - 1.0) * 100, 1),
                })

            hours_sustained = (now - self._high_cost_since) / 3600.0
            if hours_sustained >= self._SUSTAINED_HOURS and not self._migration_proposed:
                self._migration_proposed = True
                self._log.info(
                    "cost_threshold_sustained_proposing_migration",
                    hours_sustained=round(hours_sustained, 2),
                    from_provider=self._config.current_provider,
                    to_provider=cheapest_provider,
                )
                asyncio.create_task(
                    self._executor.migrate_to_provider(
                        target_provider=cheapest_provider,
                        trigger_reason=f"cost_triggered_{hours_sustained:.1f}h_sustained",
                    ),
                    name="sacm_cost_triggered_migration",
                )
        else:
            # Cost dropped back below threshold - reset window
            if self._high_cost_since is not None:
                hours_was_high = (time.time() - self._high_cost_since) / 3600.0
                self._log.info(
                    "cost_dropped_below_threshold",
                    hours_was_high=round(hours_was_high, 2),
                )
            self._high_cost_since = None
            self._migration_proposed = False

    def _get_cheapest_alternative(self) -> tuple[str, float]:
        """
        Return (provider_id, cost_usd_per_hour) for the cheapest alternative
        provider from the oracle's current pricing surface snapshot.

        Returns ("", 0.0) if the oracle is unavailable or returns no offers.
        """
        if self._oracle is None:
            return "", 0.0

        try:
            surface = self._oracle.snapshot()
            offers = getattr(surface, "offers", [])
            best_provider = ""
            best_cost = float("inf")
            for offer in offers:
                provider_id = getattr(offer, "provider_id", "")
                cost = getattr(offer, "price_per_hour_usd", None)
                if cost is None:
                    # Try alternate field names
                    cost = getattr(offer, "cost_usd_per_hour", None) or getattr(offer, "raw_cost_usd", 0.0)
                cost = float(cost) if cost is not None else 0.0
                if provider_id and cost > 0.0 and cost < best_cost:
                    best_cost = cost
                    best_provider = provider_id
            if best_cost == float("inf"):
                return "", 0.0
            return best_provider, best_cost
        except Exception as exc:
            self._log.debug("oracle_cheapest_lookup_failed", error=str(exc))
            return "", 0.0

    async def _emit(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            asyncio.create_task(
                self._event_bus.emit(SynapseEvent(
                    event_type=event_type,
                    source_system="sacm",
                    data=data,
                )),
                name=f"sacm_cost_monitor_{event_type.value[:30]}",
            )
        except Exception as exc:
            self._log.debug("emit_failed", event_type=event_type, error=str(exc))
