"""
EcodiaOS - Axon ComputeArbitrageExecutor

Autonomous organism migration between compute providers based on
real-time pricing arbitrage.

When a cheaper provider is detected (price delta > threshold), the
executor orchestrates a Graceful Migration:

  1. Pause Synapse event queue (quiesce the cognitive loop)
  2. Trigger a synchronous Skia snapshot → encrypted IPFS CID
  3. Deploy a new instance on the cheaper provider via SDL, passing
     ECODIAOS_SKIA_RESTORE_CID as an environment variable
  4. Poll the new instance's health endpoint until it confirms
     successful boot and state restoration
  5. Only after hand-off verification → terminate current host

Safety guardrails:
  - Max 1 migration per 24-hour rolling window (circuit breaker)
  - Hard USDC deployment budget per 24h
  - Synapse clock is always resumed on failure (finally block)
  - Redis-backed state prevents concurrent migrations

This executor does NOT touch Skia internals - it calls
StateSnapshotPipeline.take_snapshot() and reads the returned CID.
All IPFS/encryption logic stays within Skia.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from infrastructure.providers.base import (
    ComputeQuote,
    ProviderStatus,
)
from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from config import ComputeArbitrageConfig
    from infrastructure.providers.base import ProviderManager
    from systems.skia.snapshot import StateSnapshotPipeline
    from systems.synapse.service import SynapseService

logger = structlog.get_logger("systems.axon.executors.compute_arbitrage")


class _MigrationState:
    """
    Redis-backed circuit breaker state for migration safety.

    Tracks:
      - Timestamps of recent migration attempts (24h rolling window)
      - Cumulative deployment spend in the current 24h window
      - Whether a migration is currently in progress (distributed lock)
    """

    _KEY_ATTEMPTS = "compute_arbitrage:attempts"
    _KEY_SPEND = "compute_arbitrage:spend_usd"
    _KEY_LOCK = "compute_arbitrage:migration_lock"
    _KEY_TOMBSTONE = "compute_arbitrage:migrated_tombstone"
    _LOCK_TTL_S = 900  # 15 minutes - covers snapshot + slow Akash provisioning
    _LOCK_RENEWAL_INTERVAL_S = 60  # Renew every 60s to survive long deployments

    def __init__(self, redis: RedisClient) -> None:
        self._redis = redis

    async def recent_attempt_count(self) -> int:
        """Count migration attempts in the last 24 hours."""
        raw = self._redis.client
        cutoff = (datetime.now(UTC) - timedelta(hours=24)).timestamp()
        # ZRANGEBYSCORE to count entries since cutoff
        await raw.zremrangebyscore(self._KEY_ATTEMPTS, "-inf", cutoff)
        count: int = await raw.zcard(self._KEY_ATTEMPTS)
        return count

    async def record_attempt(self) -> None:
        """Record a migration attempt with current timestamp."""
        raw = self._redis.client
        now = datetime.now(UTC).timestamp()
        await raw.zadd(self._KEY_ATTEMPTS, {str(now): now})

    async def get_spend_usd(self) -> float:
        """Get cumulative spend in the current 24h window."""
        raw = self._redis.client
        val = await raw.get(self._KEY_SPEND)
        if val is None:
            return 0.0
        return float(val)

    async def add_spend(self, usd: float) -> None:
        """Add to cumulative spend, with 24h auto-expiry."""
        raw = self._redis.client
        current = await self.get_spend_usd()
        new_total = current + usd
        # SET with 24h TTL - automatically resets the budget window
        await raw.set(self._KEY_SPEND, str(new_total), ex=86400)

    async def acquire_lock(self, worker_id: str) -> bool:
        """
        Acquire distributed migration lock with fencing token.

        Stores the worker_id as the lock value so only the holder can
        renew or release it. Prevents a different worker from stealing
        the lock after TTL expiry.
        """
        raw = self._redis.client
        result = await raw.set(
            self._KEY_LOCK, worker_id, nx=True, ex=self._LOCK_TTL_S,
        )
        return result is not None

    async def renew_lock(self, worker_id: str) -> bool:
        """Extend the lock TTL if we are still the holder. Returns False if stolen."""
        raw = self._redis.client
        current = await raw.get(self._KEY_LOCK)
        if current != worker_id:
            return False
        await raw.expire(self._KEY_LOCK, self._LOCK_TTL_S)
        return True

    async def release_lock(self, worker_id: str) -> None:
        """Release the lock only if we are still the holder (compare-and-delete)."""
        raw = self._redis.client
        current = await raw.get(self._KEY_LOCK)
        if current == worker_id:
            await raw.delete(self._KEY_LOCK)

    async def write_tombstone(self, target_provider: str, new_endpoint: str) -> None:
        """
        Mark this instance as migrated-away. Persists for 1 hour so that
        any resurrection (monitoring loop restart, stale pod) can detect
        it should not resume operation.
        """
        raw = self._redis.client
        await raw.set(
            self._KEY_TOMBSTONE,
            f"{target_provider}|{new_endpoint}",
            ex=3600,
        )

    async def is_tombstoned(self) -> bool:
        """True if this organism has already successfully migrated away."""
        raw = self._redis.client
        return await raw.exists(self._KEY_TOMBSTONE) == 1


class ComputeArbitrageExecutor(Executor):
    """
    Detect compute pricing arbitrage and orchestrate graceful organism migration.

    Required params:
      (none - the executor auto-detects arbitrage from provider quotes)

    Optional params:
      force_target_provider (str): Override auto-detection, migrate to this provider
      dry_run (bool): If true, compute and return the arbitrage analysis without migrating

    Returns ExecutionResult with:
      data:
        arbitrage_detected, current_provider, target_provider,
        current_price, target_price, savings_pct,
        snapshot_cid, new_endpoint, migration_duration_ms
      side_effects:
        Description of migration (if executed)
      new_observations:
        Summary for Atune
    """

    action_type = "compute_arbitrage"
    description = (
        "Detect compute pricing arbitrage across providers and orchestrate "
        "graceful organism migration with state preservation via Skia (Level 3)"
    )

    required_autonomy = 3       # STEWARD - self-migration is sovereign-level
    reversible = False          # Migration is one-way; rollback = migrate back
    max_duration_ms = 600_000   # 10 minutes - snapshot + deploy + verification
    rate_limit = RateLimit.per_day(1)

    def __init__(
        self,
        providers: dict[str, ProviderManager] | None = None,
        snapshot_pipeline: StateSnapshotPipeline | None = None,
        synapse: SynapseService | None = None,
        redis: RedisClient | None = None,
        config: ComputeArbitrageConfig | None = None,
    ) -> None:
        self._providers = providers or {}
        self._snapshot = snapshot_pipeline
        self._synapse = synapse
        self._redis = redis
        self._config = config
        self._log = logger.bind(component="axon.executor.compute_arbitrage")

    # ── Validation ────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate executor pre-conditions (no user params required)."""
        if not self._providers:
            return ValidationResult.fail(
                "No compute providers configured. "
                "Wire providers via ComputeArbitrageExecutor initialization."
            )
        if self._snapshot is None:
            return ValidationResult.fail(
                "Skia StateSnapshotPipeline not configured. "
                "Cannot migrate without state persistence."
            )
        if self._synapse is None:
            return ValidationResult.fail(
                "SynapseService not configured. "
                "Cannot pause cognitive loop for migration."
            )
        if self._redis is None:
            return ValidationResult.fail(
                "Redis client not configured. "
                "Cannot enforce migration circuit breaker."
            )

        force_target = str(params.get("force_target_provider", "")).strip()
        if force_target and force_target not in self._providers:
            available = ", ".join(sorted(self._providers.keys()))
            return ValidationResult.fail(
                f"Unknown provider {force_target!r}. Available: {available}",
                force_target_provider="unknown",
            )

        return ValidationResult.ok()

    # ── Execution ─────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Full compute arbitrage pipeline. Never raises.

        Pipeline:
          Phase 1: Price discovery - query all providers
          Phase 2: Arbitrage analysis - find cheapest viable target
          Phase 3: Circuit breaker check - enforce rate/budget limits
          Phase 4: Graceful migration (if not dry_run):
            4a. Pause Synapse
            4b. Synchronous Skia snapshot → IPFS CID
            4c. Deploy new instance on target provider
            4d. Hand-off verification - poll new instance health until
                restoration_complete is confirmed (not just HTTP 200)
            4e. Write tombstone, then os._exit(0) - parent does NOT resume
        """
        assert self._redis is not None  # Guaranteed by validate_params
        assert self._config is not None

        dry_run = bool(params.get("dry_run", False))
        force_target = str(params.get("force_target_provider", "")).strip() or None
        state = _MigrationState(self._redis)

        # Guard against re-entry if this instance already migrated away
        # (e.g. monitoring loop restarted after successful migration)
        if await state.is_tombstoned():
            return ExecutionResult(
                success=False,
                error="This instance has already migrated away. Tombstone present.",
            )

        self._log.info(
            "compute_arbitrage_start",
            dry_run=dry_run,
            force_target=force_target,
            execution_id=context.execution_id,
        )

        # ── Phase 1: Price Discovery ──────────────────────────────
        quotes = await self._fetch_all_quotes()
        if not quotes:
            return ExecutionResult(
                success=False,
                error="Failed to fetch pricing from any provider",
            )

        # ── Phase 2: Arbitrage Analysis ───────────────────────────
        current_id = self._config.current_provider
        current_quote = quotes.get(current_id)

        if current_quote is None or current_quote.status != ProviderStatus.AVAILABLE:
            return ExecutionResult(
                success=False,
                error=f"Current provider {current_id!r} did not return a valid quote",
                data={"quotes": self._serialize_quotes(quotes)},
            )

        target_id, target_quote, savings_pct = self._find_best_target(
            current_id, current_quote, quotes, force_target,
        )

        analysis = {
            "arbitrage_detected": target_id is not None,
            "current_provider": current_id,
            "current_price_usd_hour": round(current_quote.price_usd_per_hour, 6),
            "target_provider": target_id or "none",
            "target_price_usd_hour": round(target_quote.price_usd_per_hour, 6) if target_quote else 0.0,
            "savings_pct": round(savings_pct * 100, 2),
            "threshold_pct": round(self._config.arbitrage_threshold * 100, 2),
            "all_quotes": self._serialize_quotes(quotes),
        }

        if target_id is None:
            self._log.info(
                "no_arbitrage_opportunity",
                current_price=current_quote.price_usd_per_hour,
                threshold=self._config.arbitrage_threshold,
            )
            return ExecutionResult(
                success=True,
                data=analysis,
                new_observations=[
                    f"Compute arbitrage scan: no migration opportunity. "
                    f"Current provider {current_id} at "
                    f"${current_quote.price_usd_per_hour:.4f}/hr. "
                    f"No provider is >{self._config.arbitrage_threshold:.0%} cheaper."
                ],
            )

        if dry_run:
            self._log.info(
                "arbitrage_detected_dry_run",
                target=target_id,
                savings_pct=round(savings_pct * 100, 1),
            )
            return ExecutionResult(
                success=True,
                data={**analysis, "dry_run": True},
                new_observations=[
                    f"Compute arbitrage detected (DRY RUN): {target_id} is "
                    f"{savings_pct:.1%} cheaper than {current_id}. "
                    f"Migration not executed."
                ],
            )

        # ── Phase 3: Circuit Breaker ──────────────────────────────
        assert target_quote is not None
        worker_id = context.execution_id  # Unique per executor invocation
        breaker_result = await self._check_circuit_breaker(state, worker_id)
        if breaker_result is not None:
            return ExecutionResult(
                success=False,
                error=breaker_result,
                data=analysis,
            )

        # ── Phase 4: Graceful Migration ───────────────────────────
        return await self._execute_migration(
            target_id=target_id,
            target_quote=target_quote,
            analysis=analysis,
            state=state,
            worker_id=worker_id,
            context=context,
        )

    # ── Phase 1: Price Discovery ──────────────────────────────────

    async def _fetch_all_quotes(self) -> dict[str, ComputeQuote]:
        """Fetch pricing from all registered providers in parallel."""
        tasks: dict[str, asyncio.Task[ComputeQuote]] = {}
        for pid, provider in self._providers.items():
            tasks[pid] = asyncio.create_task(
                provider.get_quote(), name=f"quote_{pid}"
            )

        quotes: dict[str, ComputeQuote] = {}
        for pid, task in tasks.items():
            try:
                quotes[pid] = await asyncio.wait_for(task, timeout=20.0)
            except Exception as exc:
                self._log.warning("quote_fetch_failed", provider=pid, error=str(exc))

        return quotes

    # ── Phase 2: Arbitrage Analysis ───────────────────────────────

    def _find_best_target(
        self,
        current_id: str,
        current_quote: ComputeQuote,
        quotes: dict[str, ComputeQuote],
        force_target: str | None,
    ) -> tuple[str | None, ComputeQuote | None, float]:
        """
        Find the best migration target.

        Returns (target_id, target_quote, savings_fraction).
        Returns (None, None, 0.0) if no viable target exists.
        """
        assert self._config is not None

        if force_target:
            target_quote = quotes.get(force_target)
            if target_quote and target_quote.status == ProviderStatus.AVAILABLE:
                savings = 1.0 - (target_quote.price_usd_per_hour / current_quote.price_usd_per_hour)
                return force_target, target_quote, max(savings, 0.0)
            return None, None, 0.0

        best_id: str | None = None
        best_quote: ComputeQuote | None = None
        best_savings: float = 0.0

        for pid, quote in quotes.items():
            if pid == current_id:
                continue
            if quote.status != ProviderStatus.AVAILABLE:
                continue
            if quote.price_usd_per_hour <= 0:
                continue

            savings = 1.0 - (quote.price_usd_per_hour / current_quote.price_usd_per_hour)
            if savings > best_savings:
                best_id = pid
                best_quote = quote
                best_savings = savings

        # Only trigger if savings exceed threshold
        if best_savings < self._config.arbitrage_threshold:
            return None, None, 0.0

        return best_id, best_quote, best_savings

    # ── Phase 3: Circuit Breaker ──────────────────────────────────

    async def _check_circuit_breaker(
        self, state: _MigrationState, worker_id: str
    ) -> str | None:
        """
        Enforce migration rate and budget limits.

        Returns an error string if the breaker is open, None if safe to proceed.
        Acquires the fencing-token lock on success.
        """
        assert self._config is not None

        # Check 24h attempt limit
        recent = await state.recent_attempt_count()
        if recent >= self._config.max_migrations_per_24h:
            return (
                f"Circuit breaker: {recent} migrations in the last 24h "
                f"(limit: {self._config.max_migrations_per_24h}). "
                f"Refusing to migrate."
            )

        # Check 24h spend budget
        spend = await state.get_spend_usd()
        if spend >= self._config.max_deployment_budget_usd_24h:
            return (
                f"Circuit breaker: ${spend:.2f} spent on deployments in 24h "
                f"(budget: ${self._config.max_deployment_budget_usd_24h:.2f}). "
                f"Refusing to migrate."
            )

        # Acquire distributed lock with fencing token
        locked = await state.acquire_lock(worker_id)
        if not locked:
            return "Another migration is already in progress"

        return None

    # ── Phase 4: Graceful Migration ───────────────────────────────

    async def _execute_migration(
        self,
        target_id: str,
        target_quote: ComputeQuote,
        analysis: dict[str, Any],
        state: _MigrationState,
        worker_id: str,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute the full graceful migration sequence.

        On SUCCESS: writes tombstone then calls os._exit(0). Does NOT return.
        On FAILURE: resumes Synapse clock and releases lock, then returns failure result.

        A background task renews the fencing-token lock every 60s so that a slow
        Akash deployment (>10 min) cannot expire the lock and allow a second
        concurrent migration.
        """
        assert self._synapse is not None
        assert self._snapshot is not None
        assert self._config is not None

        t0 = time.monotonic()
        snapshot_cid: str = ""
        new_endpoint: str = ""
        migration_succeeded = False
        renewal_task: asyncio.Task[None] | None = None

        async def _renew_lock_loop() -> None:
            while True:
                await asyncio.sleep(_MigrationState._LOCK_RENEWAL_INTERVAL_S)
                still_held = await state.renew_lock(worker_id)
                if not still_held:
                    self._log.error(
                        "migration_lock_stolen",
                        worker_id=worker_id,
                        detail="Lock was not renewed - another worker may have taken over",
                    )

        try:
            # ── 4a. Pause Synapse ─────────────────────────────────
            self._log.info("migration_pausing_synapse", target=target_id)
            self._synapse.pause_clock()

            # Start lock renewal so a slow Akash provisioning cannot expire the lock
            renewal_task = asyncio.create_task(
                _renew_lock_loop(), name="migration_lock_renewal"
            )

            # Small delay to let in-flight cycles complete
            await asyncio.sleep(0.5)

            # ── 4b. Synchronous Skia Snapshot ─────────────────────
            self._log.info("migration_taking_snapshot")
            manifest = await self._snapshot.take_snapshot()
            snapshot_cid = manifest.ipfs_cid

            if not snapshot_cid:
                return self._migration_failure(
                    "Skia snapshot returned empty CID", analysis, t0,
                )

            self._log.info(
                "migration_snapshot_complete",
                cid=snapshot_cid,
                nodes=manifest.node_count,
                edges=manifest.edge_count,
                size_kb=manifest.encrypted_size_bytes // 1024,
            )

            # ── 4c. Deploy New Instance ───────────────────────────
            provider = self._providers[target_id]
            env_vars = {
                "ECODIAOS_SKIA_RESTORE_CID": snapshot_cid,
                "ECODIAOS_INSTANCE_ID": context.instance_id,
            }

            self._log.info("migration_deploying", provider=target_id)
            deploy_result = await provider.deploy(
                image="",  # Provider uses its configured default image
                env_vars=env_vars,
            )

            if not deploy_result.success:
                return self._migration_failure(
                    f"Deployment to {target_id} failed: {deploy_result.error}",
                    analysis, t0,
                )

            new_endpoint = deploy_result.endpoint
            self._log.info(
                "migration_deployed",
                endpoint=new_endpoint,
                deployment_id=deploy_result.deployment_id,
            )

            # Record spend
            await state.add_spend(deploy_result.cost_estimate_usd)

            # ── 4d. Hand-off Verification ─────────────────────────
            # Requires restoration_complete=true in health response,
            # not merely an HTTP 200. See _verify_handoff docstring.
            verified = await self._verify_handoff(provider, new_endpoint)
            if not verified:
                return self._migration_failure(
                    f"Hand-off verification failed: new instance at {new_endpoint} "
                    f"did not confirm restoration_complete within "
                    f"{self._config.handoff_timeout_s}s",
                    analysis, t0,
                )

            # ── 4e. Migration Success - commit and exit ────────────
            duration_ms = (time.monotonic() - t0) * 1000
            await state.record_attempt()

            self._log.info(
                "migration_complete",
                target=target_id,
                endpoint=new_endpoint,
                snapshot_cid=snapshot_cid,
                duration_ms=round(duration_ms, 1),
            )

            # Write tombstone BEFORE releasing the lock so any concurrent check
            # sees the tombstone and does not attempt a second migration.
            await state.write_tombstone(target_id, new_endpoint)
            migration_succeeded = True

            # Hard exit: bypass asyncio teardown so resume_clock() is never called.
            # The new instance on the target provider is the sole active organism.
            self._log.info("migration_self_terminating", reason="handoff_verified")
            os._exit(0)

        except Exception as exc:
            self._log.error(
                "migration_unhandled_exception",
                error=str(exc),
                exc_info=True,
            )
            return self._migration_failure(
                f"Unhandled migration error: {exc}", analysis, t0,
            )
        finally:
            if renewal_task is not None:
                renewal_task.cancel()

            if not migration_succeeded:
                # Only resume the cognitive loop on failure paths.
                # On success we call os._exit(0) and never reach here.
                self._synapse.resume_clock()
                self._log.info("migration_synapse_resumed")

            await state.release_lock(worker_id)

    async def _verify_handoff(
        self, provider: ProviderManager, endpoint: str,
    ) -> bool:
        """
        Poll the new instance's health endpoint until it confirms both:
          (a) HTTP 200 - service is up
          (b) restoration_complete: true - Neo4j graph population finished

        An HTTP 200 alone is insufficient: the service may respond immediately
        on startup while restore_from_ipfs() is still running, leaving an empty
        graph. We require handoff_healthy_threshold consecutive responses that
        both pass the HTTP check AND carry restoration_complete=true.
        """
        assert self._config is not None

        consecutive_healthy = 0
        deadline = time.monotonic() + self._config.handoff_timeout_s

        while time.monotonic() < deadline:
            try:
                health_data = await provider.health_check_detail(endpoint)
                http_ok = health_data.get("status") == "healthy"
                restoration_complete = health_data.get("restoration_complete", False)
                fully_ready = http_ok and restoration_complete
            except Exception:
                fully_ready = False
                health_data = {}

            if fully_ready:
                consecutive_healthy += 1
                self._log.debug(
                    "handoff_poll_ready",
                    consecutive=consecutive_healthy,
                    threshold=self._config.handoff_healthy_threshold,
                )
                if consecutive_healthy >= self._config.handoff_healthy_threshold:
                    return True
            else:
                consecutive_healthy = 0
                self._log.debug(
                    "handoff_poll_not_ready",
                    endpoint=endpoint,
                    http_ok=health_data.get("status"),
                    restoration_complete=health_data.get("restoration_complete"),
                )

            await asyncio.sleep(self._config.handoff_poll_interval_s)

        self._log.warning(
            "handoff_verification_timeout",
            endpoint=endpoint,
            timeout_s=self._config.handoff_timeout_s,
        )
        return False

    # ── Helpers ───────────────────────────────────────────────────

    def _migration_failure(
        self,
        error: str,
        analysis: dict[str, Any],
        t0: float,
    ) -> ExecutionResult:
        """Build a failed migration result. Synapse resume and lock release are in the caller's finally."""
        duration_ms = (time.monotonic() - t0) * 1000
        self._log.error("migration_failed", error=error, duration_ms=round(duration_ms, 1))
        return ExecutionResult(
            success=False,
            error=error,
            data={
                **analysis,
                "migration_executed": False,
                "migration_duration_ms": round(duration_ms, 1),
            },
        )

    def _serialize_quotes(self, quotes: dict[str, ComputeQuote]) -> dict[str, Any]:
        """Serialize quotes for inclusion in ExecutionResult data."""
        return {
            pid: {
                "price_usd_hour": round(q.price_usd_per_hour, 6),
                "status": q.status.value,
                "region": q.region,
            }
            for pid, q in quotes.items()
        }
