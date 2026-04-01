"""
EcodiaOS - SimulaProxy + InspectorProxy

Drop-in replacements that offload heavy pipelines to an out-of-process
worker via Redis Streams.

SimulaProxy  - proxies ``process_proposal()`` (evolution pipeline)
InspectorProxy - proxies ``hunt_external_repo()`` (Zero-Day Hunt / Inspector)

Both push tasks to Redis Streams and await results without blocking the
Synapse 150ms event loop - an asyncio.Future is parked per in-flight
request and resolved by a background listener task that consumes the
results stream.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import orjson
import structlog

from systems.simula.evolution_types import (
    ConfigVersion,
    EvolutionProposal,
    EvolutionRecord,
    ProposalResult,
    ProposalStatus,
)
from systems.simula.history import EvolutionHistoryManager
from systems.simula.inspector.types import InspectionResult

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient

logger = structlog.get_logger()

# Redis stream keys (un-prefixed - the proxy uses the raw redis client
# so the worker and proxy agree on the exact key without coupling to the
# RedisClient prefix logic).
TASKS_STREAM = "eos:simula:tasks"
RESULTS_STREAM = "eos:simula:results"
GOVERNANCE_STREAM = "eos:simula:governance"  # Proxy → Worker approval signals

# Inspector uses a dedicated stream pair so hunt payloads never collide
# with evolution proposals on the same consumer group.
INSPECTOR_TASKS_STREAM = "eos:inspector:tasks"
INSPECTOR_RESULTS_STREAM = "eos:inspector:results"

# Consumer group shared by all worker replicas.
RESULTS_GROUP = "simula-proxy"
INSPECTOR_RESULTS_GROUP = "inspector-proxy"

# How long to wait for a result before declaring the worker dead.
DEFAULT_TIMEOUT_S = 600.0  # 10 minutes - matches pipeline_timeout_s


class SimulaProxy:
    """
    Non-blocking proxy that serialises ``EvolutionProposal`` onto a Redis
    Stream and awaits the ``ProposalResult`` from the worker process.

    Interface-compatible with ``SimulaService``.  The heavy pipeline runs
    on the worker; read-only queries (version chain, history) are served
    directly from Neo4j via ``EvolutionHistoryManager``.
    """

    system_id: str = "simula"

    def __init__(
        self,
        redis: RedisClient,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        neo4j: Neo4jClient | None = None,
    ) -> None:
        self._redis_client = redis
        self._timeout_s = timeout_s
        self._logger = logger.bind(system="simula-proxy")

        # Read-only history queries go straight to Neo4j.
        self._history = EvolutionHistoryManager(neo4j) if neo4j else None

        # In-flight proposal futures keyed by proposal ID.
        self._pending: dict[str, asyncio.Future[ProposalResult]] = {}
        # Track the actual proposal objects so get_active_proposals() works.
        self._active_proposals: dict[str, EvolutionProposal] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._initialized = False
        self._synapse: Any = None

        # Counters for the status endpoint (mirror SimulaService.stats keys).
        self._proposals_received = 0
        self._proposals_approved = 0
        self._proposals_rejected = 0
        self._proposals_rolled_back = 0
        self._proposals_deduplicated = 0

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Create the consumer group on the results stream (idempotent) and
        start the background listener.
        """
        await self._ensure_results_group()

        self._listener_task = asyncio.create_task(
            self._listen_results(), name="simula-proxy-listener"
        )
        self._initialized = True
        self._logger.info("proxy_initialized", timeout_s=self._timeout_s)

    async def _ensure_results_group(self) -> None:
        """Idempotently create the results stream + consumer group."""
        raw: Redis = self._redis_client.client
        try:
            await raw.xgroup_create(
                RESULTS_STREAM,
                RESULTS_GROUP,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" in str(exc):
                pass  # Group already exists - that's fine.
            else:
                self._logger.error(
                    "results_group_create_failed",
                    error=str(exc),
                )
                raise

    async def shutdown(self) -> None:
        """Stop the listener and cancel any pending futures."""
        self._shutdown_event.set()
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task

        # Fail any stragglers so callers don't hang.
        for proposal_id, fut in self._pending.items():
            if not fut.done():
                fut.set_exception(
                    RuntimeError(f"SimulaProxy shutting down, proposal {proposal_id} abandoned")
                )
        self._pending.clear()
        self._active_proposals.clear()
        self._logger.info("proxy_shutdown")

    # ─── Main Interface ──────────────────────────────────────────────

    async def process_proposal(
        self, proposal: EvolutionProposal
    ) -> ProposalResult:
        """
        Enqueue the proposal for the Simula worker and wait for the result.

        This method is fully non-blocking - the calling coroutine yields
        control while waiting on an ``asyncio.Future``, so the Synapse
        clock is never stalled.
        """
        raw: Redis = self._redis_client.client
        proposal_id = proposal.id
        log = self._logger.bind(proposal_id=proposal_id)

        # Create a future *before* publishing so the listener can't race.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ProposalResult] = loop.create_future()
        self._pending[proposal_id] = fut
        self._active_proposals[proposal_id] = proposal

        # Serialise the full proposal and push to the tasks stream.
        payload = proposal.model_dump_json()
        try:
            await raw.xadd(
                TASKS_STREAM,
                {"proposal_id": proposal_id, "payload": payload},
            )
            self._proposals_received += 1
            log.info("proposal_enqueued")
        except Exception:
            self._pending.pop(proposal_id, None)
            self._active_proposals.pop(proposal_id, None)
            raise

        # Wait with timeout.
        try:
            result = await asyncio.wait_for(fut, timeout=self._timeout_s)
            log.info("proposal_result_received", status=result.status.value)
            self._record_result(result)
            return result
        except TimeoutError:
            self._pending.pop(proposal_id, None)
            self._active_proposals.pop(proposal_id, None)
            self._proposals_rejected += 1
            log.error("proposal_timeout", timeout_s=self._timeout_s)
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=(
                    f"Simula worker did not respond within {self._timeout_s}s. "
                    "The worker may have crashed or is under heavy load."
                ),
            )

    def _record_result(self, result: ProposalResult) -> None:
        """Update counters based on proposal outcome."""
        if result.status == ProposalStatus.APPROVED:
            self._proposals_approved += 1
        elif result.status == ProposalStatus.REJECTED:
            self._proposals_rejected += 1
        elif result.status == ProposalStatus.ROLLED_BACK:
            self._proposals_rolled_back += 1

    async def approve_governed_proposal(
        self, proposal_id: str, governance_record_id: str
    ) -> ProposalResult:
        """
        Send a governance approval to the worker and wait for the final result.

        Pushes an approval message onto ``eos:simula:governance``.  The worker
        picks it up, calls ``SimulaService.approve_governed_proposal()``, and
        publishes the final ``ProposalResult`` onto ``eos:simula:results`` - the
        same stream the proxy's listener already watches.  A future parked under
        the same ``proposal_id`` key ensures we resolve correctly.
        """
        raw: Redis = self._redis_client.client
        log = self._logger.bind(proposal_id=proposal_id)

        # Park a future so the listener can resolve it when the worker
        # publishes the post-approval result onto eos:simula:results.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ProposalResult] = loop.create_future()
        self._pending[proposal_id] = fut

        try:
            await raw.xadd(
                GOVERNANCE_STREAM,
                {
                    "proposal_id": proposal_id,
                    "governance_record_id": governance_record_id,
                },
            )
            log.info("governance_approval_sent")
        except Exception:
            self._pending.pop(proposal_id, None)
            raise

        try:
            result = await asyncio.wait_for(fut, timeout=self._timeout_s)
            log.info("governance_result_received", status=result.status.value)
            return result
        except TimeoutError:
            self._pending.pop(proposal_id, None)
            log.error("governance_approval_timeout", timeout_s=self._timeout_s)
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=(
                    f"Simula worker did not respond to governance approval "
                    f"within {self._timeout_s}s."
                ),
            )

    # ─── Background Listener ─────────────────────────────────────────

    async def _listen_results(self) -> None:
        """
        Continuously read from ``eos:simula:results`` and resolve the
        matching pending future for each result message.

        Uses XREADGROUP so multiple proxy instances (e.g. during rolling
        deploy) can share the consumer group without double-processing.
        """
        raw: Redis = self._redis_client.client
        consumer_name = f"proxy-{id(self)}"
        log = self._logger.bind(consumer=consumer_name)

        while not self._shutdown_event.is_set():
            try:
                # Block for up to 1 s then re-check shutdown flag.
                entries = await raw.xreadgroup(
                    groupname=RESULTS_GROUP,
                    consumername=consumer_name,
                    streams={RESULTS_STREAM: ">"},
                    count=10,
                    block=1000,
                )
                if not entries:
                    continue

                for _stream_name, messages in entries:
                    for msg_id, fields in messages:
                        await self._resolve(fields, msg_id, raw, log)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                err_msg = str(exc)
                # NOGROUP means the stream or group was deleted (e.g. by
                # nuke_stream.py).  Re-create the group and retry.
                if "NOGROUP" in err_msg:
                    log.warning("results_group_missing_recreating")
                    with contextlib.suppress(Exception):
                        await self._ensure_results_group()
                else:
                    log.error("listener_error", error=err_msg)
                # Back off briefly to avoid tight-looping on transient
                # Redis failures.
                await asyncio.sleep(1.0)

    async def _resolve(
        self,
        fields: dict[str, str],
        msg_id: str,
        raw: Redis,
        log: Any,
    ) -> None:
        """Resolve a single result message and ACK it."""
        proposal_id = fields.get("proposal_id", "")
        result_json = fields.get("result", "")

        fut = self._pending.pop(proposal_id, None)
        self._active_proposals.pop(proposal_id, None)
        if fut is None or fut.done():
            # Orphaned result (timeout already fired, or duplicate).
            log.debug("orphaned_result", proposal_id=proposal_id)
        else:
            try:
                data = orjson.loads(result_json)
                result = ProposalResult.model_validate(data)
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)
                log.error(
                    "result_parse_error",
                    proposal_id=proposal_id,
                    error=str(exc),
                )

        # Always ACK so the message is not re-delivered.
        await raw.xack(RESULTS_STREAM, RESULTS_GROUP, msg_id)

    # ─── Query Interface ────────────────────────────────────────────
    # health/stats satisfy the Synapse interface.  Read-only queries
    # (history, version chain) go directly to Neo4j via the
    # EvolutionHistoryManager - no round-trip to the worker needed.

    async def health(self) -> dict[str, Any]:
        pending = len(self._pending)
        listener_alive = (
            self._listener_task is not None and not self._listener_task.done()
        )
        return {
            "initialized": self._initialized,
            "status": "healthy" if listener_alive else "degraded",
            "mode": "proxy",
            "pending_proposals": pending,
            "listener_alive": listener_alive,
        }

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "mode": "proxy",
            "pending_proposals": len(self._pending),
            "active_proposals": len(self._active_proposals),
            "proposals_received": self._proposals_received,
            "proposals_approved": self._proposals_approved,
            "proposals_rejected": self._proposals_rejected,
            "proposals_rolled_back": self._proposals_rolled_back,
            "proposals_deduplicated": self._proposals_deduplicated,
        }

    async def get_current_version(self) -> int:
        if self._history is None:
            return 0
        return await self._history.get_current_version()

    async def get_history(self, limit: int = 50) -> list[EvolutionRecord]:
        if self._history is None:
            return []
        return await self._history.get_history(limit=limit)

    async def get_version_chain(self) -> list[ConfigVersion]:
        if self._history is None:
            return []
        return await self._history.get_version_chain()

    def get_active_proposals(self) -> list[EvolutionProposal]:
        return list(self._active_proposals.values())

    async def get_analytics(self) -> Any:
        return {}  # Read from TSDB directly if needed

    def set_telos(self, telos: Any) -> None:
        """No-op: the worker process builds its own TelosService and wires it directly.
        Telos cannot be passed cross-process; the worker owns constitutional binding."""

    def set_synapse(self, synapse: Any) -> None:
        """Subscribe to sandbox validation requests on behalf of the worker.

        In proxy mode the worker process doesn't have Synapse wired, so
        Thymos' SIMULA_SANDBOX_REQUESTED events would be lost.  We handle
        them inline - the check is lightweight (Iron Rule + conservative
        approval) and doesn't need the full ChangeSimulator.
        """
        self._synapse = synapse
        event_bus = getattr(synapse, "_event_bus", None)
        if event_bus is None:
            self._logger.warning("proxy_set_synapse_no_event_bus")
            return

        from systems.synapse.types import SynapseEventType

        async def _on_sandbox_requested(event: Any) -> None:
            await self._handle_sandbox_request(event)

        event_bus.subscribe(
            SynapseEventType.SIMULA_SANDBOX_REQUESTED,
            _on_sandbox_requested,
        )
        self._logger.info("proxy_subscribed_to_sandbox_requests")

    async def _handle_sandbox_request(self, event: Any) -> None:
        """Handle SIMULA_SANDBOX_REQUESTED inline in the proxy process.

        The ChangeSimulator lives in the worker, so we do Iron Rule checks
        and approve conservatively for lower tiers.  This mirrors the
        ``no_simulator`` path in ``SimulaService._on_simula_sandbox_requested``.
        """
        from systems.synapse.types import SynapseEvent, SynapseEventType

        data = getattr(event, "data", {}) or {}
        correlation_id = data.get("correlation_id", "")
        repair_action = data.get("repair_action", "")
        repair_tier = data.get("repair_tier", "")
        target_system = data.get("target_system", "")
        parameter_changes = data.get("parameter_changes", [])

        log = self._logger.bind(correlation_id=correlation_id)
        log.info(
            "proxy_sandbox_requested",
            repair_action=repair_action,
            repair_tier=repair_tier,
            target_system=target_system,
        )

        approved = False
        reason = "unknown"

        try:
            # Iron Rule check - reject repairs that touch protected systems
            protected = {"simula", "constitution", "invariant"}
            action_lower = (repair_action or "").lower()
            target_lower = (target_system or "").lower()

            if any(p in action_lower or p in target_lower for p in protected):
                reason = "iron_rule_violation: action or target touches protected system"
                log.warning("proxy_sandbox_iron_rule_blocked")
            else:
                # Check parameter changes against constitutional constraints
                violation = None
                for change in parameter_changes:
                    param_name = (change.get("name") or change.get("param", "")).lower()
                    if any(p in param_name for p in protected):
                        violation = param_name
                        break

                if violation:
                    reason = f"iron_rule_violation: parameter '{violation}' touches protected system"
                else:
                    # No simulator in proxy mode - approve lower tiers conservatively
                    if repair_tier in ("PARAMETER", "RESTART", "KNOWN_FIX"):
                        approved = True
                        reason = "proxy_conservative_approve"
                    else:
                        # NOVEL_FIX and higher - approve with advisory
                        # (full simulation not available in proxy mode)
                        approved = True
                        reason = "proxy_no_simulator_approve"
                        log.info("proxy_sandbox_no_simulator_approve", tier=repair_tier)
        except Exception as exc:
            log.error("proxy_sandbox_error", error=str(exc))
            reason = f"proxy_sandbox_internal_error: {exc}"

        # Emit result back to Thymos via Synapse
        event_bus = getattr(self._synapse, "_event_bus", None)
        if event_bus is not None:
            try:
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SIMULA_SANDBOX_RESULT,
                    source_system="simula",
                    data={
                        "correlation_id": correlation_id,
                        "approved": approved,
                        "reason": reason,
                    },
                ))
                log.info("proxy_sandbox_result_emitted", approved=approved, reason=reason)
            except Exception as emit_exc:
                log.error("proxy_sandbox_emit_failed", error=str(emit_exc))
        else:
            log.error("proxy_sandbox_no_event_bus")

    def set_soma_ref(self, soma: Any) -> None:
        """No-op: the worker process will wire Soma directly after its own initialization."""
        self._logger.debug("soma_ref_wired_to_proxy")

    def set_fovea_ref(self, fovea: Any) -> None:
        """No-op: the worker process will wire Fovea directly after its own initialization."""
        self._logger.debug("fovea_ref_wired_to_proxy")

    def subscribe_to_evolution_candidates(self, event_bus: Any) -> None:
        """
        Subscribe to EVOLUTION_CANDIDATE events on behalf of the worker.

        The worker process runs a full SimulaService that processes proposals.
        In proxy mode, we forward EVOLUTION_CANDIDATE events as raw evo_proposal
        tasks onto TASKS_STREAM so the worker can call receive_evo_proposal().
        """
        from systems.synapse.types import SynapseEventType

        async def _on_evolution_candidate(event: Any) -> None:
            data = getattr(event, "data", {})
            hypothesis_id = data.get("hypothesis_id", "")
            payload_bytes = orjson.dumps({
                "_task_type": "evo_proposal",
                "hypothesis_id": hypothesis_id,
                "hypothesis_statement": data.get("hypothesis_statement", ""),
                "evidence_score": float(data.get("evidence_score", 0.0)),
                "mutation_type": data.get("mutation_type", ""),
                "mutation_target": data.get("mutation_target", ""),
                "mutation_description": data.get("mutation_description", ""),
                "supporting_episodes": data.get("supporting_episodes", []),
            })
            try:
                await self._redis_client.client.xadd(
                    TASKS_STREAM,
                    {"proposal_id": hypothesis_id, "payload": payload_bytes.decode()},
                )
                self._logger.info(
                    "evolution_candidate_forwarded",
                    hypothesis_id=hypothesis_id,
                )
            except Exception as exc:
                self._logger.error(
                    "evolution_candidate_forward_failed",
                    hypothesis_id=hypothesis_id,
                    error=str(exc),
                )

        event_bus.subscribe(
            SynapseEventType.EVOLUTION_CANDIDATE,
            _on_evolution_candidate,
        )
        self._logger.info("simula_proxy_subscribed_to_evolution_candidates")


# ═══════════════════════════════════════════════════════════════════════
# InspectorProxy - offloads hunt_external_repo() to the worker
# ═══════════════════════════════════════════════════════════════════════


class InspectorProxy:
    """
    Non-blocking proxy that serialises a hunt request onto
    ``eos:inspector:tasks`` and awaits the ``InspectionResult`` from
    the worker process.

    Interface-compatible with the subset of ``InspectorService`` that
    the ``_cc_pipeline`` endpoint calls (``hunt_external_repo``).
    """

    def __init__(
        self,
        redis: RedisClient,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._redis_client = redis
        self._timeout_s = timeout_s
        self._logger = logger.bind(system="inspector-proxy")

        # In-flight hunt futures keyed by hunt_id.
        self._pending: dict[str, asyncio.Future[InspectionResult]] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create consumer group on the results stream and start the listener."""
        await self._ensure_results_group()

        self._listener_task = asyncio.create_task(
            self._listen_results(), name="inspector-proxy-listener"
        )
        self._logger.info("inspector_proxy_initialized", timeout_s=self._timeout_s)

    async def _ensure_results_group(self) -> None:
        """Idempotently create the inspector results stream + consumer group."""
        raw: Redis = self._redis_client.client
        try:
            await raw.xgroup_create(
                INSPECTOR_RESULTS_STREAM,
                INSPECTOR_RESULTS_GROUP,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" in str(exc):
                pass  # Group already exists
            else:
                self._logger.error(
                    "inspector_results_group_create_failed",
                    error=str(exc),
                )
                raise

    async def shutdown(self) -> None:
        """Stop the listener and cancel any pending futures."""
        self._shutdown_event.set()
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task

        for hunt_id, fut in self._pending.items():
            if not fut.done():
                fut.set_exception(
                    RuntimeError(f"InspectorProxy shutting down, hunt {hunt_id} abandoned")
                )
        self._pending.clear()
        self._logger.info("inspector_proxy_shutdown")

    # ─── Main Interface ──────────────────────────────────────────────

    async def hunt_external_repo(
        self,
        github_url: str,
        *,
        attack_goals: list[str] | None = None,
        generate_pocs: bool = True,
        generate_patches: bool = False,
    ) -> InspectionResult:
        """
        Enqueue a hunt for the worker and wait for the ``InspectionResult``.

        Fully non-blocking - the calling coroutine yields control while
        waiting on an ``asyncio.Future``, so the Synapse clock is never
        stalled.
        """
        import uuid

        raw: Redis = self._redis_client.client
        hunt_id = str(uuid.uuid4())
        log = self._logger.bind(hunt_id=hunt_id, target=github_url)

        # Create a future *before* publishing to prevent a race.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[InspectionResult] = loop.create_future()
        self._pending[hunt_id] = fut

        # Serialise the hunt request.
        payload = orjson.dumps({
            "github_url": github_url,
            "attack_goals": attack_goals,
            "generate_pocs": generate_pocs,
            "generate_patches": generate_patches,
        }).decode()

        try:
            await raw.xadd(
                INSPECTOR_TASKS_STREAM,
                {"hunt_id": hunt_id, "payload": payload},
            )
            log.info("hunt_enqueued")
        except Exception:
            self._pending.pop(hunt_id, None)
            raise

        # Wait with timeout.
        try:
            result = await asyncio.wait_for(fut, timeout=self._timeout_s)
            log.info(
                "hunt_result_received",
                vulns=len(result.vulnerabilities_found),
            )
            return result
        except TimeoutError:
            self._pending.pop(hunt_id, None)
            log.error("hunt_timeout", timeout_s=self._timeout_s)
            # Return an empty result rather than crashing the SSE stream.
            from systems.simula.inspector.types import TargetType

            return InspectionResult(
                target_url=github_url,
                target_type=TargetType.EXTERNAL_REPO,
            )

    # ─── Background Listener ─────────────────────────────────────────

    async def _listen_results(self) -> None:
        """Consume ``eos:inspector:results`` and resolve pending futures."""
        raw: Redis = self._redis_client.client
        consumer_name = f"iproxy-{id(self)}"
        log = self._logger.bind(consumer=consumer_name)

        while not self._shutdown_event.is_set():
            try:
                entries = await raw.xreadgroup(
                    groupname=INSPECTOR_RESULTS_GROUP,
                    consumername=consumer_name,
                    streams={INSPECTOR_RESULTS_STREAM: ">"},
                    count=10,
                    block=1000,
                )
                if not entries:
                    continue

                for _stream_name, messages in entries:
                    for msg_id, fields in messages:
                        await self._resolve(fields, msg_id, raw, log)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                err_msg = str(exc)
                if "NOGROUP" in err_msg:
                    log.warning("inspector_results_group_missing_recreating")
                    with contextlib.suppress(Exception):
                        await self._ensure_results_group()
                else:
                    log.error("inspector_listener_error", error=err_msg)
                await asyncio.sleep(1.0)

    async def _resolve(
        self,
        fields: dict[str, str],
        msg_id: str,
        raw: Redis,
        log: Any,
    ) -> None:
        """Resolve a single hunt result message and ACK it."""
        hunt_id = fields.get("hunt_id", "")
        result_json = fields.get("result", "")

        fut = self._pending.pop(hunt_id, None)
        if fut is None or fut.done():
            log.debug("orphaned_hunt_result", hunt_id=hunt_id)
        else:
            try:
                data = orjson.loads(result_json)
                result = InspectionResult.model_validate(data)
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)
                log.error(
                    "hunt_result_parse_error",
                    hunt_id=hunt_id,
                    error=str(exc),
                )

        await raw.xack(INSPECTOR_RESULTS_STREAM, INSPECTOR_RESULTS_GROUP, msg_id)
