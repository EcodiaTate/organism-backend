"""
EcodiaOS - Alive WebSocket Server

Standalone WebSocket server on port 8001 that bridges multiple data streams
to connected browser clients:

1. synapse events (via Redis pub/sub) - forwarded as-is
2. affect state snapshots (polled from Soma at ~10Hz)
3. system_state snapshots (polled at ~1Hz) - aggregated real-time state:
   - Cognitive cycle number and rhythm phase (Synapse) → ``cycle``
   - Drive topology: multipliers from Telos, rejections from Thymos → ``drives``
   - Soma full 9D interoceptive state + urgency + attractor → ``interoceptive``
   - Fovea prediction-error decomposition (6D) → ``attention``
   - Thymos immune incidents/repairs/antibodies → ``immune``
   - Current goal queue (Nova) → ``goals``
   - Last N actions and outcomes (Axon) → ``actions``
   - Economic status: balance, burn rate, runway (Oikos) → ``economics``
   - Mutation history: proposed/applied/rejected (Simula) → ``mutations``
   - Benchmark KPI snapshot (via BenchmarkProvider protocol) → ``benchmarks``
   - Causal discovery state (Kairos) → ``causal``
   - Compression / world model state (Logos) → ``compression``
   - Sleep stage and cycle health (Oneiros) → ``sleep``

Messages are JSON-encoded with an envelope:
  {"stream": "synapse" | "affect" | "system_state", "payload": {...}}

RE Training: every system_state snapshot is written to the Redis Stream
``{prefix}:stream:alive_snapshots`` (maxlen=10_000) for the RE training
pipeline (Spec 11 §21 Path 1).

─── Protocol Note - Two Distinct Alive Endpoints ──────────────────────────
This standalone server (port 8001) and the FastAPI ``/ws/alive`` route in
``main.py`` (port 8000) are **intentionally different protocols**:

  Standalone (port 8001) - ``AliveWebSocketServer``
    Streams: ``synapse``, ``affect`` (9D Soma + dominance), ``system_state``
    Affect payload keys: valence, arousal, dominance, curiosity,
                         care_activation, coherence_stress, energy,
                         confidence, integrity, temporal_pressure, urgency,
                         dominant_error, ts
    Audience: Three.js dashboard, monitoring systems, RE pipeline

  FastAPI (port 8000) - ``/ws/alive``
    Streams: ``affect`` (6D AffectState), ``synapse``, ``workspace``, ``outcomes``
    Affect payload keys: valence, arousal, dominance, curiosity,
                         care_activation, coherence_stress, ts
    Audience: Cloud Run single-port deployments; perception/decisions pages

These two protocols are governed independently. Do NOT unify them without
updating consumers on both sides.
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import orjson
import structlog
import websockets

from utils.asyncio_helpers import cancel_and_wait_tasks

# Import at module level - avoids repeated import cache lookups inside 10Hz polling hot path
from primitives.affect import InteroceptiveDimension

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

    from clients.redis import RedisClient
    from systems.atune.service import AtuneService
    from systems.axon.service import AxonService
    from systems.fovea.service import FoveaService
    from systems.kairos.pipeline import CausalMiningPipeline
    from systems.logos.service import LogosService
    from systems.nova.service import NovaService
    from systems.oikos.service import OikosService
    from systems.oneiros.service import OneirosService
    from systems.reasoning_engine.service import ReasoningEngineService
    from systems.simula.service import SimulaService
    from systems.soma.service import SomaService
    from systems.synapse.service import SynapseService
    from systems.telos.service import TelosService
    from systems.thymos.service import ThymosService

logger = structlog.get_logger("systems.alive.ws_server")

# Affect polling interval (seconds) - ~10 Hz (default; runtime-adjustable via RESOURCE_PRESSURE)
_AFFECT_POLL_INTERVAL: float = 0.1

# System state polling interval (seconds) - ~1 Hz (default; runtime-adjustable via RESOURCE_PRESSURE)
_STATE_POLL_INTERVAL: float = 1.0

# Minimum / maximum bounds for runtime poll interval adjustment
_AFFECT_POLL_INTERVAL_MIN: float = 0.05   # 20 Hz cap - prevents runaway load
_AFFECT_POLL_INTERVAL_MAX: float = 0.5    # 2 Hz floor under extreme pressure
_STATE_POLL_INTERVAL_MIN: float = 0.5     # 2 Hz cap
_STATE_POLL_INTERVAL_MAX: float = 5.0     # 0.2 Hz floor under extreme pressure

# How many recent Axon outcomes to include in each snapshot
_AXON_RECENT_COUNT: int = 10

# Per-subsystem timeout for async gathers - spec §13.2 Strategy 3
# A hung Nova/Oikos/Thymos call must not stall the entire 1Hz poller
_GATHER_TIMEOUT: float = 0.8

# Redis Stream key suffix for RE training snapshots - spec §21 Path 1
_ALIVE_SNAPSHOTS_STREAM: str = "stream:alive_snapshots"

# Maximum retained snapshots in the Redis Stream (rolling window ~2.7h at 1Hz)
_ALIVE_SNAPSHOTS_MAXLEN: int = 10_000

# WebSocket authentication - query-param key name for bearer token (Spec 11 §14.3)
# Token is validated against AliveWebSocketServer._auth_tokens at handshake time.
# No token configured → open access (dev/internal LAN mode).
_AUTH_TOKEN_PARAM: str = "token"


def _json(data: dict[str, Any]) -> str:
    """Fast JSON serialization via orjson."""
    return orjson.dumps(data).decode()


# ─── Benchmark Provider Protocol ───────────────────────────────────────────


@runtime_checkable
class BenchmarkProvider(Protocol):
    """
    Protocol for the benchmark data source.

    Alive reads the ``stats`` property each polling cycle. Inject a
    ``BenchmarkService`` instance; pass ``None`` to emit ``available=False``.
    """

    @property
    def stats(self) -> dict[str, Any]:
        """Return current benchmark KPI snapshot."""
        ...


# ─── Server ────────────────────────────────────────────────────────────────


class AliveWebSocketServer:
    """
    WebSocket server for the Alive visualization layer.

    Multiplexes three data streams over a single WebSocket connection:

    - ``synapse`` - raw Synapse telemetry events from Redis pub/sub
    - ``affect`` - Soma interoceptive state at ~10 Hz (9D)
    - ``system_state`` - aggregated snapshot of all subsystems at ~1 Hz
    """

    system_id: str = "alive"

    def __init__(
        self,
        redis: RedisClient,
        *,
        soma: SomaService,
        synapse: SynapseService | None = None,
        atune: AtuneService | None = None,
        telos: TelosService | None = None,
        thymos: ThymosService | None = None,
        nova: NovaService | None = None,
        axon: AxonService | None = None,
        oikos: OikosService | None = None,
        simula: SimulaService | None = None,
        fovea: FoveaService | None = None,
        kairos: CausalMiningPipeline | None = None,
        logos: LogosService | None = None,
        oneiros: OneirosService | None = None,
        benchmarks: BenchmarkProvider | None = None,
        port: int = 8001,
        auth_tokens: set[str] | None = None,
    ) -> None:
        self._redis = redis
        self._soma = soma
        self._atune = atune
        self._synapse = synapse
        self._telos = telos
        self._thymos = thymos
        self._nova = nova
        self._axon = axon
        self._oikos = oikos
        self._simula = simula
        self._fovea = fovea
        self._kairos = kairos
        self._logos = logos
        self._oneiros = oneiros
        self._benchmarks = benchmarks
        # Late-injected after benchmarks initialisation (Phase 11)
        self._re_service: ReasoningEngineService | None = None
        self._port = port
        # Non-empty set → auth enforced; None or empty set → open (dev/LAN mode)
        self._auth_tokens: set[str] = auth_tokens or set()
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self._running: bool = False
        self._tasks: list[asyncio.Task[None]] = []
        self._logger = logger.bind(component="alive_ws")
        # Runtime-adjustable poll intervals (mutated on RESOURCE_PRESSURE events)
        self._affect_poll_interval: float = _AFFECT_POLL_INTERVAL
        self._state_poll_interval: float = _STATE_POLL_INTERVAL

    async def start(self) -> None:
        """Start the WebSocket server and background stream tasks."""
        self._running = True
        self._server = await websockets.serve(
            self._handler,
            "0.0.0.0",
            self._port,
        )
        self._tasks.append(asyncio.create_task(self._redis_subscriber()))
        self._tasks.append(asyncio.create_task(self._affect_poller()))
        self._tasks.append(asyncio.create_task(self._system_state_poller()))
        self._logger.info("alive_ws_started", port=self._port)

    async def stop(self) -> None:
        """Shut down the server and cancel background tasks."""
        self._running = False
        await cancel_and_wait_tasks(self._tasks, timeout=10.0, logger=self._logger)
        self._tasks.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._logger.info("alive_ws_stopped")

    # ─── Connection Handler ────────────────────────────────────────────

    async def _handler(self, ws: ServerConnection) -> None:
        """Handle a new WebSocket connection.

        Authentication (Spec 11 §14.3):
          If ``auth_tokens`` was supplied at construction time, the client must
          pass a valid token in the ``token`` query parameter, e.g.::

              ws://host:8001/?token=<secret>

          Invalid / missing token → close 4401. When no tokens are configured
          (dev / internal LAN mode) all connections are accepted.
        """
        remote = ws.remote_address

        # ── Auth gate ──────────────────────────────────────────────────
        if self._auth_tokens:
            # websockets stores the path including query string on request.path
            raw_path: str = getattr(ws.request, "path", "") or ""
            # Extract token from query string without pulling in urllib at hot path
            token: str = ""
            if "?" in raw_path:
                qs = raw_path.split("?", 1)[1]
                for part in qs.split("&"):
                    if part.startswith(f"{_AUTH_TOKEN_PARAM}="):
                        token = part[len(_AUTH_TOKEN_PARAM) + 1:]
                        break
            if token not in self._auth_tokens:
                self._logger.warning(
                    "alive_auth_rejected",
                    remote=str(remote),
                    has_token=bool(token),
                )
                await ws.close(4401, "Unauthorized")
                return
        # ──────────────────────────────────────────────────────────────

        self._clients.add(ws)
        self._logger.info(
            "alive_client_connected",
            remote=str(remote),
            total_clients=len(self._clients),
        )
        try:
            # Send initial state so the client can render immediately
            await self._send_initial_state(ws)
            # Keep connection alive; we don't expect client messages
            async for _ in ws:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            self._logger.info(
                "alive_client_disconnected",
                remote=str(remote),
                total_clients=len(self._clients),
            )

    async def _send_initial_state(self, ws: ServerConnection) -> None:
        """Send initial affect + system_state snapshots so the client doesn't start blank."""
        affect_msg = _json({"stream": "affect", "payload": self._build_affect_payload()})
        await ws.send(affect_msg)

        system_state_msg = _json({
            "stream": "system_state",
            "payload": await self._build_system_state_payload(),
        })
        await ws.send(system_state_msg)

    # ─── Redis Subscriber (Synapse Events) ─────────────────────────────

    async def _redis_subscriber(self) -> None:
        """Subscribe to Redis synapse_events and forward to all clients."""
        redis = self._redis.client
        prefix = self._redis._config.prefix
        channel = f"{prefix}:channel:synapse_events"

        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        self._logger.info("alive_redis_subscribed", channel=channel)

        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.1,
                )
                if message and message["type"] == "message":
                    raw = message["data"]
                    # Redis data is already a JSON string (orjson-encoded by EventBus)
                    payload = orjson.loads(raw) if isinstance(raw, (str, bytes)) else raw
                    msg = _json({"stream": "synapse", "payload": payload})
                    await self._broadcast(msg)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.error("alive_redis_subscriber_error", error=str(exc))
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    # ─── Affect Poller ─────────────────────────────────────────────────

    async def _affect_poller(self) -> None:
        """Poll Soma interoceptive state at ~10Hz and send to all clients."""
        try:
            while self._running:
                msg = _json({
                    "stream": "affect",
                    "payload": self._build_affect_payload(),
                })
                await self._broadcast(msg)
                await asyncio.sleep(self._affect_poll_interval)
        except asyncio.CancelledError:
            pass

    def _build_affect_payload(self) -> dict[str, Any]:
        if self._soma is None:
            return {"available": False}
        try:
            state = self._soma.get_current_state()
            sensed = state.sensed

            # dominance - from Atune's AffectState if wired (authoritative),
            # else approximated from Soma's SOCIAL_CHARGE (0→1 proxy).
            # Spec §4.3 requires this field in the affect stream.
            dominance: float = 0.5
            if self._atune is not None:
                try:
                    dominance = round(self._atune.current_affect.dominance, 4)
                except Exception:
                    dominance = round(sensed.get(InteroceptiveDimension.SOCIAL_CHARGE, 0.5), 4)
            else:
                dominance = round(sensed.get(InteroceptiveDimension.SOCIAL_CHARGE, 0.5), 4)

            return {
                # Core 6 dimensions (backward-compatible keys)
                "valence": round(sensed[InteroceptiveDimension.VALENCE], 4),
                "arousal": round(sensed[InteroceptiveDimension.AROUSAL], 4),
                "dominance": dominance,
                "curiosity": round(sensed[InteroceptiveDimension.CURIOSITY_DRIVE], 4),
                "care_activation": round(sensed[InteroceptiveDimension.SOCIAL_CHARGE], 4),
                "coherence_stress": round(1.0 - sensed[InteroceptiveDimension.COHERENCE], 4),
                # Additional Soma-native dimensions
                "energy": round(sensed[InteroceptiveDimension.ENERGY], 4),
                "confidence": round(sensed[InteroceptiveDimension.CONFIDENCE], 4),
                "integrity": round(sensed[InteroceptiveDimension.INTEGRITY], 4),
                "temporal_pressure": round(sensed[InteroceptiveDimension.TEMPORAL_PRESSURE], 4),
                # Derived signals
                "urgency": round(state.urgency, 4),
                "dominant_error": state.dominant_error.value if state.dominant_error else None,
                "ts": state.timestamp.isoformat() if state.timestamp else None,
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    # ─── System State Poller ────────────────────────────────────────────

    async def _system_state_poller(self) -> None:
        """Poll all subsystems at ~1Hz and broadcast aggregated system_state.

        Also writes each snapshot to the Redis Stream ``alive_snapshots`` for
        RE training pipeline consumption (Spec 11 §21 Path 1).
        """
        try:
            while self._running:
                payload = await self._build_system_state_payload()
                msg = _json({"stream": "system_state", "payload": payload})
                await self._broadcast(msg)
                await self._write_re_training_snapshot(payload)
                await asyncio.sleep(self._state_poll_interval)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.error("alive_state_poller_error", error=str(exc))

    async def _write_re_training_snapshot(self, payload: dict[str, Any]) -> None:
        """Write system_state snapshot to Redis Stream for RE training pipeline.

        Ref: Spec 11 §21 Path 1. Fire-and-forget; errors are non-fatal.
        """
        try:
            prefix = self._redis._config.prefix
            stream_key = f"{prefix}:{_ALIVE_SNAPSHOTS_STREAM}"
            await self._redis.client.xadd(
                stream_key,
                {"snapshot": orjson.dumps(payload)},
                maxlen=_ALIVE_SNAPSHOTS_MAXLEN,
            )
        except Exception as exc:
            self._logger.debug("alive_re_snapshot_write_error", error=str(exc))

    async def _build_system_state_payload(self) -> dict[str, Any]:
        """
        Aggregate real-time state from all registered subsystems.

        Each section is gathered independently; failures in one section
        do not prevent the others from being included.

        Async gathers are wrapped in asyncio.wait_for(_GATHER_TIMEOUT) so a
        hung subsystem cannot stall the entire 1Hz poller (Spec 11 §13.2 Strategy 3).

        Key layout:
          cycle         - Synapse clock + rhythm phase
          drives        - Telos topology multipliers + Thymos rejection counters
          interoceptive - Soma full 9D state + urgency + nearest attractor
          attention     - Fovea prediction-error decomposition (6D)
          immune        - Thymos incident/repair/antibody data only
          goals         - Nova active goal queue
          actions       - Axon last N outcomes
          economics     - Oikos balance/runway/burn
          mutations     - Simula proposal stats
          benchmarks    - 7 KPI snapshot
          causal        - Kairos invariant hierarchy stats
          compression   - Logos world model + Schwarzschild state
          sleep         - Oneiros stage + cycle health
          re_status     - RE routing: Thompson weights, circuit state, Claude/RE fraction
        """
        return {
            "cycle": self._gather_cycle(),
            "drives": self._gather_drives(),
            "interoceptive": self._gather_interoceptive(),
            "attention": self._gather_attention(),
            "immune": await self._gather_immune_safe(),
            "goals": await self._gather_goals_safe(),
            "actions": self._gather_actions(),
            "economics": await self._gather_economics_safe(),
            "mutations": self._gather_mutations(),
            "benchmarks": self._gather_benchmarks(),
            "causal": await self._gather_causal_safe(),
            "compression": await self._gather_compression_safe(),
            "sleep": self._gather_sleep(),
            "re_status": self._gather_re_status(),
        }

    async def _gather_immune_safe(self) -> dict[str, Any]:
        """Thymos immune gather with timeout guard (Spec 11 §13.2)."""
        try:
            return await asyncio.wait_for(self._gather_immune(), timeout=_GATHER_TIMEOUT)
        except asyncio.TimeoutError:
            self._logger.warning("alive_gather_timeout", section="immune")
            return {"available": False, "error": "timeout"}

    async def _gather_goals_safe(self) -> dict[str, Any]:
        """Nova goals gather with timeout guard (Spec 11 §13.2)."""
        try:
            return await asyncio.wait_for(self._gather_goals(), timeout=_GATHER_TIMEOUT)
        except asyncio.TimeoutError:
            self._logger.warning("alive_gather_timeout", section="goals")
            return {"available": False, "error": "timeout"}

    async def _gather_economics_safe(self) -> dict[str, Any]:
        """Oikos economics gather with timeout guard (Spec 11 §13.2)."""
        try:
            return await asyncio.wait_for(self._gather_economics(), timeout=_GATHER_TIMEOUT)
        except asyncio.TimeoutError:
            self._logger.warning("alive_gather_timeout", section="economics")
            return {"available": False, "error": "timeout"}

    async def _gather_causal_safe(self) -> dict[str, Any]:
        """Kairos causal gather with timeout guard."""
        try:
            return await asyncio.wait_for(self._gather_causal(), timeout=_GATHER_TIMEOUT)
        except asyncio.TimeoutError:
            self._logger.warning("alive_gather_timeout", section="causal")
            return {"available": False, "error": "timeout"}

    async def _gather_compression_safe(self) -> dict[str, Any]:
        """Logos compression gather with timeout guard."""
        try:
            return await asyncio.wait_for(self._gather_compression(), timeout=_GATHER_TIMEOUT)
        except asyncio.TimeoutError:
            self._logger.warning("alive_gather_timeout", section="compression")
            return {"available": False, "error": "timeout"}

    # ── interoceptive (Soma full 9D state + urgency + attractor) ────────

    def _gather_interoceptive(self) -> dict[str, Any]:
        """Soma full 9D interoceptive state, urgency signal, and nearest attractor."""
        if self._soma is None:
            return {"available": False}
        try:
            state = self._soma.get_current_state()
            if state is None:
                return {"available": False}
            s = state.sensed
            return {
                "available": True,
                # Full 9D interoceptive vector
                "valence": round(s[InteroceptiveDimension.VALENCE], 4),
                "arousal": round(s[InteroceptiveDimension.AROUSAL], 4),
                "curiosity": round(s[InteroceptiveDimension.CURIOSITY_DRIVE], 4),
                "social_charge": round(s[InteroceptiveDimension.SOCIAL_CHARGE], 4),
                "coherence": round(s[InteroceptiveDimension.COHERENCE], 4),
                "energy": round(s[InteroceptiveDimension.ENERGY], 4),
                "confidence": round(s[InteroceptiveDimension.CONFIDENCE], 4),
                "integrity": round(s[InteroceptiveDimension.INTEGRITY], 4),
                "temporal_pressure": round(s[InteroceptiveDimension.TEMPORAL_PRESSURE], 4),
                # Derived signals
                "urgency": round(state.urgency, 4),
                "dominant_error": state.dominant_error.value if state.dominant_error else None,
                "nearest_attractor": state.nearest_attractor if hasattr(state, "nearest_attractor") else None,
            }
        except Exception as exc:
            self._logger.debug("alive_gather_interoceptive_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── immune (Thymos immune system - incidents, repairs, antibodies) ──

    async def _gather_immune(self) -> dict[str, Any]:
        """Immune system status from Thymos (incident/repair/antibody data only)."""
        if self._thymos is None:
            return {"available": False}
        try:
            h = await self._thymos.health()
            return {
                "available": True,
                "active_incidents": h.get("active_incidents", 0),
                "healing_mode": h.get("healing_mode", "unknown"),
                "antibody_count": h.get("total_antibodies", 0),
                "repairs_attempted": h.get("repairs_attempted", 0),
                "repairs_succeeded": h.get("repairs_succeeded", 0),
                "storm_activations": h.get("storm_activations", 0),
            }
        except Exception as exc:
            self._logger.debug("alive_gather_immune_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── cycle ───────────────────────────────────────────────────────────

    def _gather_cycle(self) -> dict[str, Any]:
        """Current cognitive cycle number and rhythm phase from Synapse."""
        if self._synapse is None:
            return {"available": False}
        try:
            clock = self._synapse.clock_state
            return {
                "available": True,
                "cycle_number": clock.cycle_count,
                "running": clock.running,
                "paused": clock.paused,
                "period_ms": round(clock.current_period_ms, 2),
                "target_period_ms": round(clock.target_period_ms, 2),
                "rate_hz": round(1000.0 / clock.current_period_ms, 3) if clock.current_period_ms > 0 else 0.0,
                "arousal": round(clock.arousal, 4),
                "overrun_count": clock.overrun_count,
                # Rhythm phase: the emergent cognitive state string (e.g. "normal", "flow", "stressed")
                "rhythm_phase": self._synapse.rhythm_snapshot.state.value
                if hasattr(self._synapse, "rhythm_snapshot")
                else None,
            }
        except Exception as exc:
            self._logger.debug("alive_gather_cycle_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── drives ─────────────────────────────────────────────────────────

    def _gather_drives(self) -> dict[str, Any]:
        """Drive topology from Telos (authoritative) with Thymos enforcement counters."""
        result: dict[str, Any] = {"available": False}

        # Drive MEASUREMENT from Telos (authoritative)
        if self._telos is not None:
            try:
                report = self._telos.last_report
                if report is not None:
                    effective_i_ratio = (
                        round(report.effective_I / report.nominal_I, 4)
                        if report.nominal_I > 0
                        else 0.0
                    )
                    result = {
                        "available": True,
                        # Topology multipliers: 1.0 = fully aligned, 0.0 = fully degraded
                        "care": round(report.care_multiplier, 4),
                        # coherence_bonus >= 1.0; invert to [0,1] alignment scale
                        "coherence": round(1.0 / max(report.coherence_bonus, 1.0), 4),
                        "growth": round(report.growth_rate, 4),
                        "honesty": round(report.honesty_coefficient, 4),
                        # Master metric: fraction of nominal_I that is effective
                        "effective_intelligence_ratio": effective_i_ratio,
                    }
                else:
                    result = {"available": False, "error": "no_computation_yet"}
            except Exception as exc:
                self._logger.debug("alive_gather_drives_telos_error", error=str(exc))
                result = {"available": False, "error": str(exc)}

        # Drive ENFORCEMENT data from Thymos (supplementary)
        if self._thymos is not None:
            try:
                ds = self._thymos.drive_state
                result["equor_rejections"] = ds.equor_rejections
                result["rejections_by_drive"] = ds.rejections_by_drive
            except Exception:
                pass  # Thymos data is supplementary, not critical

        return result

    # ── goals ───────────────────────────────────────────────────────────

    async def _gather_goals(self) -> dict[str, Any]:
        """Current goal queue from Nova's health snapshot."""
        if self._nova is None:
            return {"available": False}
        try:
            h = await self._nova.health()
            goal_stats: dict[str, Any] = h.get("goals") or {}
            # Also pull active goal summaries for the ordered list
            summaries = self._nova.active_goal_summaries
            return {
                "available": True,
                "counts": goal_stats,
                "active": [
                    {
                        "id": s.id,
                        "priority": round(s.priority, 4),
                    }
                    for s in summaries
                ],
            }
        except Exception as exc:
            self._logger.debug("alive_gather_goals_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── actions ─────────────────────────────────────────────────────────

    def _gather_actions(self) -> dict[str, Any]:
        """Last N actions and outcomes from Axon's execution history."""
        if self._axon is None:
            return {"available": False}
        try:
            outcomes = self._axon.recent_outcomes[:_AXON_RECENT_COUNT]
            return {
                "available": True,
                "recent": [
                    {
                        "execution_id": o.execution_id,
                        "intent_id": o.intent_id,
                        "success": o.success,
                        "partial": o.partial,
                        "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                        "duration_ms": o.duration_ms,
                        "failure_reason": o.failure_reason or None,
                        # First step's action_type as a human-readable label
                        "action_type": o.step_outcomes[0].action_type if o.step_outcomes else None,
                        "step_count": len(o.step_outcomes),
                    }
                    for o in outcomes
                ],
            }
        except Exception as exc:
            self._logger.debug("alive_gather_actions_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── economics ───────────────────────────────────────────────────────

    async def _gather_economics(self) -> dict[str, Any]:
        """Economic status from Oikos health check."""
        if self._oikos is None:
            return {"available": False}
        try:
            h = await self._oikos.health()
            return {
                "available": True,
                # Core economic vitals
                "liquid_balance_usd": h.get("liquid_balance"),
                "runway_days": h.get("runway_days"),
                "bmr_usd_per_hour": h.get("bmr_usd_per_hour"),
                "starvation_level": h.get("starvation_level"),
                "is_metabolically_positive": h.get("is_metabolically_positive"),
                # Assets
                "assets_live": h.get("assets_live"),
                "assets_building": h.get("assets_building"),
                "total_asset_value": h.get("total_asset_value"),
                # Fleet
                "fleet_children": h.get("fleet_children"),
                "fleet_equity": h.get("fleet_equity"),
            }
        except Exception as exc:
            self._logger.debug("alive_gather_economics_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── mutations ───────────────────────────────────────────────────────

    def _gather_mutations(self) -> dict[str, Any]:
        """Recent mutation history from Simula's stats property."""
        if self._simula is None:
            return {"available": False}
        try:
            s = self._simula.stats
            analytics = s.get("analytics") or {}
            return {
                "available": True,
                "current_version": s.get("current_version"),
                "proposals_received": s.get("proposals_received"),
                "proposals_approved": s.get("proposals_approved"),
                "proposals_rejected": s.get("proposals_rejected"),
                "proposals_rolled_back": s.get("proposals_rolled_back"),
                "proposals_awaiting_governance": s.get("proposals_awaiting_governance"),
                "active_proposals": s.get("active_proposals"),
                # Derived analytics (may be None if not yet cached)
                "evolution_velocity": analytics.get("evolution_velocity"),
                "rollback_rate": analytics.get("rollback_rate"),
                "mean_simulation_risk": analytics.get("mean_simulation_risk"),
            }
        except Exception as exc:
            self._logger.debug("alive_gather_mutations_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── attention (Fovea prediction error telemetry) ─────────────────────

    def _gather_attention(self) -> dict[str, Any]:
        """Current prediction error decomposition from Fovea."""
        if self._fovea is None:
            return {"available": False}
        try:
            profile = self._fovea.get_current_attention_profile()
            return {
                "available": True,
                "error_decomposition": {
                    "content": round(profile.mean_content_error, 4),
                    "timing": round(profile.mean_timing_error, 4),
                    "magnitude": round(profile.mean_magnitude_error, 4),
                    "source": round(profile.mean_source_error, 4),
                    "category": round(profile.mean_category_error, 4),
                    "causal": round(profile.mean_causal_error, 4),
                },
                "dynamic_threshold": round(profile.current_ignition_threshold, 4),
                "habituation_count": profile.habituated_pattern_count,
                "top_surprise": profile.highest_recent_error_summary,
            }
        except Exception as exc:
            self._logger.debug("alive_gather_attention_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── benchmarks ──────────────────────────────────────────────────────

    def _gather_benchmarks(self) -> dict[str, Any]:
        """Benchmark KPI snapshot via the BenchmarkProvider protocol."""
        if self._benchmarks is None:
            return {"available": False}
        try:
            stats = self._benchmarks.stats
            return {
                "available": True,
                "decision_quality": stats.get("decision_quality"),
                "llm_dependency": stats.get("llm_dependency"),
                "economic_ratio": stats.get("economic_ratio"),
                "learning_rate": stats.get("learning_rate"),
                "mutation_success_rate": stats.get("mutation_success_rate"),
                "effective_intelligence_ratio": stats.get("effective_intelligence_ratio"),
                "compression_ratio": stats.get("compression_ratio"),
                "regressions": stats.get("active_regressions", []),
            }
        except Exception as exc:
            self._logger.debug("alive_gather_benchmarks_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── causal (Kairos invariant hierarchy) ─────────────────────────────

    async def _gather_causal(self) -> dict[str, Any]:
        """Causal invariant mining state from Kairos.

        Ref: Spec 11 §22 gap 4 - Kairos absent from Alive telemetry.
        Exposes the organism's most scientifically interesting signal:
        how many causal rules it has discovered and at what tier.
        """
        if self._kairos is None:
            return {"available": False}
        try:
            h = await self._kairos.health()
            hierarchy = h.get("hierarchy") or {}
            ledger = h.get("intelligence_ledger") or {}
            return {
                "available": True,
                "pipeline_runs": h.get("pipeline_runs"),
                "invariants_created": h.get("invariants_created"),
                "tier3_discoveries": h.get("tier3_discoveries"),
                "counter_invariants_found": h.get("counter_invariants_found"),
                # Hierarchy breakdown by tier
                "tier1_count": hierarchy.get("tier1_count"),
                "tier2_count": hierarchy.get("tier2_count"),
                "tier3_count": hierarchy.get("tier3_count"),
                "total_active": hierarchy.get("total_active"),
                # Intelligence ratio contribution
                "mean_intelligence_ratio": ledger.get("mean_intelligence_ratio"),
                "total_i_ratio_contribution": ledger.get("total_i_ratio_contribution"),
            }
        except Exception as exc:
            self._logger.debug("alive_gather_causal_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── compression (Logos world model + Schwarzschild) ──────────────────

    async def _gather_compression(self) -> dict[str, Any]:
        """Compression and world model state from Logos.

        Ref: Spec 11 §22 gap 4 - Logos absent from Alive telemetry.
        Exposes cognitive pressure, intelligence ratio, and Schwarzschild proximity.
        """
        if self._logos is None:
            return {"available": False}
        try:
            h = await self._logos.health()
            return {
                "available": True,
                "cognitive_pressure": h.get("cognitive_pressure"),
                "compression_urgency": h.get("compression_urgency"),
                "intelligence_ratio": h.get("intelligence_ratio"),
                "world_model_schemas": h.get("world_model_schemas"),
                "world_model_complexity_bits": h.get("world_model_complexity_bits"),
                "schwarzschild_met": h.get("schwarzschild_met"),
                "anchor_memories": h.get("anchor_memories"),
            }
        except Exception as exc:
            self._logger.debug("alive_gather_compression_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── sleep (Oneiros stage + cycle health) ────────────────────────────

    def _gather_sleep(self) -> dict[str, Any]:
        """Current sleep stage and cycle health from Oneiros.

        Ref: Spec 11 §22 gap 4 - Oneiros absent from Alive telemetry.
        Exposes sleep pressure, current stage, and cumulative cycle metrics.
        """
        if self._oneiros is None:
            return {"available": False}
        try:
            s = self._oneiros.stats
            return {
                "available": True,
                "is_sleeping": self._oneiros.is_sleeping,
                "current_stage": s.get("current_stage"),
                "current_pressure": s.get("current_pressure"),
                "current_degradation": s.get("current_degradation"),
                "total_sleep_cycles": s.get("total_sleep_cycles"),
                "total_dreams": s.get("total_dreams"),
                "total_insights": s.get("total_insights"),
                "mean_dream_coherence": s.get("mean_dream_coherence"),
                "mean_sleep_quality": s.get("mean_sleep_quality"),
                "episodes_consolidated": s.get("episodes_consolidated"),
            }
        except Exception as exc:
            self._logger.debug("alive_gather_sleep_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ── Synapse event bus subscription ──────────────────────────────

    def set_event_bus(self, event_bus: Any) -> None:
        """Subscribe to Synapse events that allow autonomous poll-rate adjustment.

        Subscribes to:
        - ``RESOURCE_PRESSURE``: throttle affect/state poll intervals when the
          organism is under CPU pressure.  Gives Alive an autonomous recourse
          path instead of hard-blocking the event loop at fixed rates regardless
          of system load.
        - ``CONSERVATION_MODE_ENTERED``: extreme throttle to minimum-viable rates
          (grid/metabolic conservation mode - organism is starving).
        - ``CONSERVATION_MODE_EXITED``: restore nominal poll rates.

        Must be called after ``start()`` so that the event bus is live.
        Non-fatal - if subscription fails, polling continues at default rates.
        """
        try:
            from systems.synapse.types import SynapseEventType

            event_bus.subscribe(
                SynapseEventType.RESOURCE_PRESSURE,
                self._on_resource_pressure,
            )
            event_bus.subscribe(
                SynapseEventType.CONSERVATION_MODE_ENTERED,
                self._on_conservation_mode_entered,
            )
            event_bus.subscribe(
                SynapseEventType.CONSERVATION_MODE_EXITED,
                self._on_conservation_mode_exited,
            )
            # Spec 09 / Spec 29: VitalityCoordinator austerity - drop to minimum
            # viable rates if Alive is in halt_systems, or recover on nominal.
            event_bus.subscribe(
                SynapseEventType.SYSTEM_MODULATION,
                self._on_system_modulation,
            )
            self._logger.info(
                "alive_event_bus_subscribed",
                events=["RESOURCE_PRESSURE", "CONSERVATION_MODE_ENTERED", "CONSERVATION_MODE_EXITED", "SYSTEM_MODULATION"],
            )
        except Exception as exc:
            self._logger.warning("alive_event_bus_subscription_failed", error=str(exc))

    def _on_resource_pressure(self, event: Any) -> None:
        """Autonomously throttle telemetry poll rates under CPU pressure.

        Payload: ``pressure_level`` ("elevated" | "high"), ``total_cpu_percent``.

        - "elevated" (cpu > 80%): slow affect to 5 Hz, state to 0.5 Hz
        - "high"     (cpu > 90%): slow affect to 2 Hz, state to 0.2 Hz
        - pressure clears (no event → next poll at slower rate; organism will
          naturally recover as RESOURCE_PRESSURE stops firing after rebalance)

        Rates are clamped to defined min/max bounds to prevent runaway throttling.
        """
        try:
            data: dict[str, Any] = getattr(event, "data", {}) or {}
            pressure_level: str = str(data.get("pressure_level", "elevated"))

            if pressure_level == "high":
                # Extreme pressure: 2 Hz affect, 0.2 Hz state
                new_affect = min(_AFFECT_POLL_INTERVAL_MAX, 0.5)
                new_state = min(_STATE_POLL_INTERVAL_MAX, 5.0)
            else:
                # Elevated pressure: 5 Hz affect, 0.5 Hz state
                new_affect = min(_AFFECT_POLL_INTERVAL_MAX, 0.2)
                new_state = min(_STATE_POLL_INTERVAL_MAX, 2.0)

            # Apply bounds
            new_affect = max(_AFFECT_POLL_INTERVAL_MIN, new_affect)
            new_state = max(_STATE_POLL_INTERVAL_MIN, new_state)

            if (
                new_affect != self._affect_poll_interval
                or new_state != self._state_poll_interval
            ):
                self._affect_poll_interval = new_affect
                self._state_poll_interval = new_state
                self._logger.info(
                    "alive_poll_throttled",
                    pressure_level=pressure_level,
                    affect_interval_s=new_affect,
                    state_interval_s=new_state,
                )
        except Exception as exc:
            self._logger.debug("alive_resource_pressure_handler_error", error=str(exc))

    def _on_conservation_mode_entered(self, event: Any) -> None:
        """Drop to minimum-viable poll rates when the organism enters grid conservation mode.

        Conservation mode means severe metabolic stress - every CPU cycle counts.
        Alive drops to 2 Hz affect and 0.2 Hz state (the defined _MAX bounds).
        """
        try:
            self._affect_poll_interval = _AFFECT_POLL_INTERVAL_MAX
            self._state_poll_interval = _STATE_POLL_INTERVAL_MAX
            self._logger.info(
                "alive_conservation_mode_throttle",
                affect_interval_s=self._affect_poll_interval,
                state_interval_s=self._state_poll_interval,
            )
        except Exception as exc:
            self._logger.debug("alive_conservation_mode_handler_error", error=str(exc))

    def _on_conservation_mode_exited(self, event: Any) -> None:
        """Restore nominal poll rates when conservation mode ends."""
        self.restore_nominal_poll_rates()

    def _on_system_modulation(self, event: Any) -> None:
        """Respond to VitalityCoordinator SYSTEM_MODULATION directives.

        Spec 09 / Spec 29: if 'alive' appears in halt_systems or the level is
        safe_mode/emergency, drop to minimum-viable poll rates (same as
        CONSERVATION_MODE_ENTERED). Recovery on nominal level with empty halt list.

        Alive does NOT emit SYSTEM_MODULATION_ACK - it is a passive telemetry bridge
        and ACK would require an async event bus reference during a sync callback.
        The poll-rate change is the observable compliance signal.
        """
        try:
            data: dict[str, Any] = getattr(event, "data", {}) or {}
            halt_systems: list[str] = data.get("halt_systems", [])
            level: str = str(data.get("level", "nominal"))

            if "alive" in halt_systems or level in ("safe_mode", "emergency"):
                # Drop to minimum viable rates
                self._affect_poll_interval = _AFFECT_POLL_INTERVAL_MAX
                self._state_poll_interval = _STATE_POLL_INTERVAL_MAX
                self._logger.info(
                    "alive_system_modulation_throttle",
                    level=level,
                    affect_interval_s=self._affect_poll_interval,
                    state_interval_s=self._state_poll_interval,
                )
            elif not halt_systems and level == "nominal":
                self.restore_nominal_poll_rates()
        except Exception as exc:
            self._logger.debug("alive_system_modulation_handler_error", error=str(exc))

    def restore_nominal_poll_rates(self) -> None:
        """Restore poll intervals to nominal defaults.

        Call when resource pressure has subsided (e.g. after CONSERVATION_MODE_EXITED).
        Also available as an explicit recourse path for monitoring systems or operators.
        """
        self._affect_poll_interval = _AFFECT_POLL_INTERVAL
        self._state_poll_interval = _STATE_POLL_INTERVAL
        self._logger.info(
            "alive_poll_rates_restored",
            affect_interval_s=_AFFECT_POLL_INTERVAL,
            state_interval_s=_STATE_POLL_INTERVAL,
        )

    # ── re_status (Reasoning Engine routing + health) ────────────────

    def set_re_service(self, re_service: ReasoningEngineService) -> None:
        """Late-inject the Reasoning Engine service (wired post-benchmarks, Phase 11).

        Must be called after ``_init_benchmarks`` so that Thompson sampling weights
        have been restored from Neo4j before the first gather cycle reads them.
        """
        self._re_service = re_service

    def _gather_re_status(self) -> dict[str, Any]:
        """RE routing health: availability, circuit state, Thompson sampling weights.

        Spec §22 gap 3 - RE status is currently invisible.  This section makes
        Thompson sampling weights, Claude-vs-RE routing, and circuit-breaker state
        observable alongside the rest of organism health.
        """
        if self._re_service is None:
            return {"available": False}
        try:
            re = self._re_service
            is_available: bool = bool(getattr(re, "is_available", False))
            circuit_open: bool = bool(getattr(re, "_circuit_open", False))
            consecutive_failures: int = int(getattr(re, "_consecutive_failures", 0))
            model: str = str(getattr(re, "_model", "unknown"))

            # Thompson sampling weights (Beta-Bernoulli posterior means)
            thompson: dict[str, Any] = {}
            raw_t: dict[str, dict[str, float]] | None = getattr(re, "_thompson", None)
            if raw_t:
                for arm_key, arm_values in raw_t.items():
                    alpha = float(arm_values.get("alpha", 1.0))
                    beta = float(arm_values.get("beta", 1.0))
                    # Beta distribution mean: alpha / (alpha + beta)
                    posterior_mean = round(alpha / (alpha + beta), 4)
                    thompson[arm_key] = {
                        "alpha": round(alpha, 4),
                        "beta": round(beta, 4),
                        "posterior_mean": posterior_mean,
                    }

            # Derived routing preference: fraction of weight on RE vs Claude
            re_arm = thompson.get("custom", {})
            claude_arm = thompson.get("claude", {})
            re_mean = re_arm.get("posterior_mean", 0.5)
            claude_mean = claude_arm.get("posterior_mean", 0.5)
            total = re_mean + claude_mean
            re_routing_fraction = round(re_mean / total, 4) if total > 0 else 0.5

            return {
                "available": True,
                "is_available": is_available,
                "circuit_open": circuit_open,
                "consecutive_failures": consecutive_failures,
                "model": model,
                "thompson": thompson,
                "re_routing_fraction": re_routing_fraction,
            }
        except Exception as exc:
            self._logger.debug("alive_gather_re_status_error", error=str(exc))
            return {"available": False, "error": str(exc)}

    # ─── Broadcast ─────────────────────────────────────────────────────

    async def _broadcast(self, message: str) -> None:
        """Send a message to all connected clients. Remove dead ones."""
        if not self._clients:
            return
        dead: set[ServerConnection] = set()
        for ws in self._clients:
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ─── Health ────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health status for Alive WebSocket server."""
        return {
            "status": "running" if self._running else "stopped",
            "port": self._port,
            "connected_clients": len(self._clients),
            "streams": ["synapse", "affect", "system_state"],
            "auth_enabled": bool(self._auth_tokens),
            # Current runtime poll intervals (may differ from defaults if throttled)
            "affect_poll_interval_s": self._affect_poll_interval,
            "state_poll_interval_s": self._state_poll_interval,
            "affect_throttled": self._affect_poll_interval != _AFFECT_POLL_INTERVAL,
            "state_throttled": self._state_poll_interval != _STATE_POLL_INTERVAL,
            "systems_wired": {
                "soma": self._soma is not None,
                "atune": self._atune is not None,
                "synapse": self._synapse is not None,
                "telos": self._telos is not None,
                "thymos": self._thymos is not None,
                "nova": self._nova is not None,
                "axon": self._axon is not None,
                "oikos": self._oikos is not None,
                "simula": self._simula is not None,
                "fovea": self._fovea is not None,
                "kairos": self._kairos is not None,
                "logos": self._logos is not None,
                "oneiros": self._oneiros is not None,
                "benchmarks": self._benchmarks is not None,
                "re_service": self._re_service is not None,
            },
        }
