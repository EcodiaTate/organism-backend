"""
EcodiaOS — Alive WebSocket Server

Standalone WebSocket server on port 8001 that bridges multiple data streams
to connected browser clients:

1. synapse events (via Redis pub/sub) — forwarded as-is
2. affect state snapshots (polled from Soma at ~10Hz)
3. system_state snapshots (polled at ~1Hz) — aggregated real-time state:
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

Messages are JSON-encoded with an envelope:
  {"stream": "synapse" | "affect" | "system_state", "payload": {...}}
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import orjson
import structlog
import websockets

from utils.asyncio_helpers import cancel_and_wait_tasks

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

    from clients.redis import RedisClient
    from systems.atune.service import AtuneService
    from systems.axon.service import AxonService
    from systems.fovea.service import FoveaService
    from systems.nova.service import NovaService
    from systems.oikos.service import OikosService
    from systems.simula.service import SimulaService
    from systems.soma.service import SomaService
    from systems.synapse.service import SynapseService
    from systems.telos.service import TelosService
    from systems.thymos.service import ThymosService

logger = structlog.get_logger("systems.alive.ws_server")

# Affect polling interval (seconds) — ~10 Hz
_AFFECT_POLL_INTERVAL: float = 0.1

# System state polling interval (seconds) — ~1 Hz
_STATE_POLL_INTERVAL: float = 1.0

# How many recent Axon outcomes to include in each snapshot
_AXON_RECENT_COUNT: int = 10

# How many recent Simula proposals to include in each snapshot
_SIMULA_RECENT_PROPOSALS: int = 5


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

    - ``synapse`` — raw Synapse telemetry events from Redis pub/sub
    - ``affect`` — Soma interoceptive state at ~10 Hz (9D)
    - ``system_state`` — aggregated snapshot of all subsystems at ~1 Hz
    """

    system_id: str = "alive"

    def __init__(
        self,
        redis: RedisClient,
        *,
        soma: SomaService,
        atune: AtuneService | None = None,
        synapse: SynapseService | None = None,
        telos: TelosService | None = None,
        thymos: ThymosService | None = None,
        nova: NovaService | None = None,
        axon: AxonService | None = None,
        oikos: OikosService | None = None,
        simula: SimulaService | None = None,
        fovea: FoveaService | None = None,
        benchmarks: BenchmarkProvider | None = None,
        port: int = 8001,
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
        self._benchmarks = benchmarks
        self._port = port
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self._running: bool = False
        self._tasks: list[asyncio.Task[None]] = []
        self._logger = logger.bind(component="alive_ws")

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
        """Handle a new WebSocket connection."""
        self._clients.add(ws)
        remote = ws.remote_address
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
                await asyncio.sleep(_AFFECT_POLL_INTERVAL)
        except asyncio.CancelledError:
            pass

    def _build_affect_payload(self) -> dict[str, Any]:
        if self._soma is None:
            return {"available": False}
        try:
            from systems.soma.types import InteroceptiveDimension

            state = self._soma.get_current_state()
            sensed = state.sensed
            return {
                # Original 6 dimensions (backward-compatible keys)
                "valence": round(sensed[InteroceptiveDimension.VALENCE], 4),
                "arousal": round(sensed[InteroceptiveDimension.AROUSAL], 4),
                "curiosity": round(sensed[InteroceptiveDimension.CURIOSITY_DRIVE], 4),
                "care_activation": round(sensed[InteroceptiveDimension.SOCIAL_CHARGE], 4),
                "coherence_stress": round(1.0 - sensed[InteroceptiveDimension.COHERENCE], 4),
                # NEW Soma-native dimensions
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
        """Poll all subsystems at ~1Hz and broadcast aggregated system_state."""
        try:
            while self._running:
                payload = await self._build_system_state_payload()
                msg = _json({"stream": "system_state", "payload": payload})
                await self._broadcast(msg)
                await asyncio.sleep(_STATE_POLL_INTERVAL)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.error("alive_state_poller_error", error=str(exc))

    async def _build_system_state_payload(self) -> dict[str, Any]:
        """
        Aggregate real-time state from all registered subsystems.

        Each section is gathered independently; failures in one section
        do not prevent the others from being included.

        Key layout (post-consolidation):
          cycle         — Synapse clock + rhythm phase
          drives        — Telos topology multipliers + Thymos rejection counters
          interoceptive — Soma full 9D state + urgency + nearest attractor
          attention     — Fovea prediction-error decomposition (6D)
          immune        — Thymos incident/repair/antibody data only
          goals         — Nova active goal queue
          actions       — Axon last N outcomes
          economics     — Oikos balance/runway/burn
          mutations     — Simula proposal stats
          benchmarks    — 7 KPI snapshot
        """
        return {
            "cycle": self._gather_cycle(),
            "drives": self._gather_drives(),
            "interoceptive": self._gather_interoceptive(),
            "attention": self._gather_attention(),
            "immune": await self._gather_immune(),
            "goals": await self._gather_goals(),
            "actions": self._gather_actions(),
            "economics": await self._gather_economics(),
            "mutations": self._gather_mutations(),
            "benchmarks": self._gather_benchmarks(),
        }

    # ── interoceptive (Soma full 9D state + urgency + attractor) ────────

    def _gather_interoceptive(self) -> dict[str, Any]:
        """Soma full 9D interoceptive state, urgency signal, and nearest attractor."""
        if self._soma is None:
            return {"available": False}
        try:
            from systems.soma.types import InteroceptiveDimension

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

    # ── immune (Thymos immune system — incidents, repairs, antibodies) ──

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
                "benchmarks": self._benchmarks is not None,
            },
        }
