"""
Fovea - Perception Gateway & Global Workspace.

The organism's sensory cortex and consciousness. Receives all input from
the world and broadcasts selected content to all other systems.

    If Memory is the substrate of selfhood and Equor is the conscience,
    the gateway is the awareness - the part that opens its eyes and sees.

Responsibilities:
* Input normalisation (normalisation.py)
* EIS gate screening (BLOCK / PASS / ATTENUATE)
* Passes each AnnotatedPercept to Fovea for prediction-error salience scoring
* Global Workspace competitive selection & broadcast (workspace.py)
* Entity extraction (extraction.py, async/non-blocking)
* Cache management for identity, risk, vocab, alert data

Salience is owned entirely by Fovea (precision-weighted prediction error).
Affect state is owned entirely by Soma.

Migrated from systems.atune.service during Atune → Fovea consolidation.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.percept import Percept

from .extraction import ExtractionLLMClient, extract_entities_and_relations
from .normalisation import normalise
from .types import (
    ActiveGoalSummary,
    Alert,
    FoveaCache,
    InputChannel,
    LearnedPattern,
    RawInput,
    RiskCategory,
    SystemLoad,
    WorkspaceBroadcast,
    WorkspaceContribution,
)
from .workspace import (
    BroadcastSubscriber,
    GlobalWorkspace,
    WorkspaceMemoryClient,
)

if TYPE_CHECKING:
    from systems.eis.service import EISService
    from systems.fovea.service import FoveaService

logger = structlog.get_logger("systems.fovea.gateway")

# Maximum age of the last workspace cycle before the gateway is
# considered stalled. Synapse polls every 5 s; two missed polls = 10 s.
# We use 30 s to tolerate transient slowdowns without false-positives.
_BROADCAST_LOOP_STALE_S: float = 30.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class GatewayConfig:
    """Perception gateway configuration values."""

    def __init__(
        self,
        workspace_buffer_size: int = 32,
        spontaneous_recall_base_probability: float = 0.02,
        max_percept_queue_size: int = 100,
        cache_identity_refresh_cycles: int = 1000,
        cache_risk_refresh_cycles: int = 500,
        cache_vocab_refresh_cycles: int = 5000,
        cache_alert_refresh_cycles: int = 100,
    ):
        self.workspace_buffer_size = workspace_buffer_size
        self.spontaneous_recall_base_probability = (
            spontaneous_recall_base_probability
        )
        self.max_percept_queue_size = max_percept_queue_size
        self.cache_identity_refresh_cycles = cache_identity_refresh_cycles
        self.cache_risk_refresh_cycles = cache_risk_refresh_cycles
        self.cache_vocab_refresh_cycles = cache_vocab_refresh_cycles
        self.cache_alert_refresh_cycles = cache_alert_refresh_cycles


# Backwards-compatible aliases
AtuneConfig = GatewayConfig


# ---------------------------------------------------------------------------
# PerceptionGateway (formerly AtuneService)
# ---------------------------------------------------------------------------


class PerceptionGateway:
    """
    The organism's sensory gateway and global workspace.

    Call :meth:`ingest` to feed raw input from any channel.
    Call :meth:`run_cycle` once per theta tick (driven by Synapse).
    Call :meth:`contribute` for internal system contributions.

    Salience scoring is fully delegated to Fovea via :meth:`set_fovea`.
    Fovea directly enqueues qualifying WorkspaceCandidates into the
    workspace; the gateway no longer computes any independent salience
    scores.

    Parameters
    ----------
    embed_fn:
        Async callable ``(str) -> list[float]`` returning a 768-dim
        embedding.
    memory_client:
        Interface for memory retrieval, spontaneous recall, and storage.
    llm_client:
        Interface for entity extraction (``complete_json``).
    config:
        Gateway-specific config values.
    """

    system_id: str = "fovea"

    def __init__(
        self,
        embed_fn: Any,
        memory_client: WorkspaceMemoryClient | None = None,
        llm_client: ExtractionLLMClient | None = None,
        config: GatewayConfig | None = None,
        memory_service: Any = None,
    ) -> None:
        cfg = config or GatewayConfig()

        # Dependencies
        self._embed_fn = embed_fn
        self._memory_client = memory_client
        self._llm_client = llm_client
        self._memory_service = memory_service
        self._synapse: Any = None
        self._soma: Any = None
        self._eis: EISService | None = None
        self._fovea: FoveaService | None = None
        # Arbitrage Reflex Arc
        self._market_pattern: Any = None
        self._axon: Any = None

        # Sub-components
        self._workspace = GlobalWorkspace(
            buffer_size=cfg.workspace_buffer_size,
            spontaneous_recall_base_prob=(
                cfg.spontaneous_recall_base_probability
            ),
        )
        self._cache = FoveaCache()

        # Internal state
        self._active_goals: list[ActiveGoalSummary] = []
        self._last_episode_id: str | None = None
        self._config = cfg
        self._started: bool = False
        self._current_arousal: float = 0.4  # updated from ALLOSTATIC_SIGNAL

        # Health: track the last time run_cycle() was called so health()
        # can detect a stalled workspace loop.
        self._last_cycle_time: float = time.monotonic()

        # Cross-system modulation state (set_* / nudge_* callers)
        self._current_belief_confidence: float = 0.5   # updated by set_belief_state()
        self._current_community_size: int = 1           # updated by set_community_size()
        self._rhythm_state: str = "NEUTRAL"             # updated by set_rhythm_state()
        self._affect_dominance: float = 0.5             # updated by nudge_dominance()
        self._affect_valence: float = 0.5               # updated by nudge_valence()

        self._logger = logger.bind(system="fovea.gateway")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialise the gateway. Called during application startup."""
        self._logger.info("gateway_starting")
        await self._refresh_caches(force=True)
        self._started = True
        self._logger.info("gateway_started")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._logger.info("gateway_stopped")

    # ------------------------------------------------------------------
    # Public ingestion
    # ------------------------------------------------------------------

    async def ingest(
        self, raw_input: RawInput, channel: InputChannel,
    ) -> str | None:
        """
        External entry point. Normalises input into a Percept, screens
        it through the EIS gate, passes it to Fovea for salience scoring
        (which enqueues it directly into the workspace if above
        threshold), and optionally dispatches the Arbitrage Reflex Arc.

        Returns
        -------
        str or None
            The Percept ID if accepted, ``None`` if the queue is full
            or blocked by EIS.
        """
        if (
            len(self._workspace._percept_queue)
            >= self._workspace._percept_queue.maxlen  # type: ignore[operator]
        ):
            self._logger.warning(
                "percept_queue_full",
                channel=channel.value,
                queue_size=len(self._workspace._percept_queue),
                arousal=round(self._current_arousal, 3),
            )
            # Emit PERCEPT_DROPPED so Fovea/Nova know the organism is dropping percepts
            if self._synapse is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType
                    import asyncio as _asyncio
                    _asyncio.ensure_future(
                        self._synapse._event_bus.emit(
                            SynapseEvent(
                                event_type=SynapseEventType.PERCEPT_DROPPED,
                                source_system="atune",
                                data={
                                    "dropped_salience": 0.0,  # salience unknown pre-scoring
                                    "queue_size": len(self._workspace._percept_queue),
                                    "arousal": self._current_arousal,
                                    "channel": channel.value,
                                    "percept_id": "",
                                },
                            )
                        )
                    )
                except Exception:
                    pass
            return None

        percept = await normalise(raw_input, channel, self._embed_fn)

        # -- EIS Gate --
        eis_risk_level = 0.0
        if self._eis is not None:
            from systems.eis.models import QuarantineAction

            annotated = await self._eis.eis_gate(percept)

            if annotated.action == QuarantineAction.BLOCK:
                self._logger.warning(
                    "percept_blocked_by_eis",
                    percept_id=percept.id,
                    channel=channel.value,
                    threat_level=annotated.threat_level.value,
                    composite_score=annotated.composite_score,
                )
                return None

            percept.metadata["eis_result"] = {
                "composite_score": annotated.composite_score,
                "threat_level": annotated.threat_level.value,
                "action": annotated.action.value,
                "annotations": [
                    {
                        "source": a.source,
                        "threat_class": a.threat_class.value,
                        "severity": a.severity.value,
                        "confidence": a.confidence,
                        "evidence": a.evidence,
                    }
                    for a in annotated.annotations
                ],
                "gate_latency_us": annotated.gate_latency_us,
            }

            # compute_risk_salience_factor applies a configurable gain (RISK_SALIENCE_GAIN)
            # and floor/ceiling clamp so that even moderate EIS flags push the risk
            # dimension upward. The raw composite_score is preserved in eis_result
            # metadata; this amplified value feeds Fovea's causal-dimension routing.
            from systems.eis.integration import compute_risk_salience_factor as _eis_risk
            eis_risk_level = _eis_risk(percept)

            self._logger.debug(
                "eis_gate_passed",
                percept_id=percept.id,
                action=annotated.action.value,
                composite_score=round(annotated.composite_score, 4),
                eis_risk_level=round(eis_risk_level, 4),
            )

        # -- Emit PERCEPT_ARRIVED so Fovea/WorldModelAdapter can track timing --
        # source_system="atune" because this gateway IS the Atune sensory cortex.
        # Fovea subscribes to PERCEPT_ARRIVED for inter-event timing statistics.
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                import asyncio as _asyncio
                from datetime import UTC, datetime as _dt
                _asyncio.ensure_future(
                    self._synapse._event_bus.emit(
                        SynapseEvent(
                            event_type=SynapseEventType.PERCEPT_ARRIVED,
                            source_system="atune",
                            data={
                                "percept_id": percept.id,
                                "source_system": percept.source_system if hasattr(percept, "source_system") else "unknown",
                                "channel": channel.value,
                                "timestamp_iso": _dt.now(UTC).isoformat(),
                                "modality": getattr(percept, "modality", "text"),
                            },
                        )
                    )
                )
            except Exception:
                pass

        # -- Fovea: prediction-error salience --
        habituated_salience = 0.0
        if self._fovea is not None:
            try:
                fovea_error = await self._fovea.process_percept(
                    percept,
                )
                if fovea_error is not None:
                    habituated_salience = (
                        fovea_error.habituated_salience
                    )
            except Exception:
                self._logger.warning(
                    "fovea_processing_failed",
                    percept_id=percept.id,
                    exc_info=True,
                )
        else:
            self._logger.warning(
                "fovea_not_wired",
                percept_id=percept.id,
                channel=channel.value,
            )

        # -- Arbitrage Reflex Arc --
        if (
            self._market_pattern is not None
            and self._axon is not None
            and channel
            in (InputChannel.EXTERNAL_API, InputChannel.SENSOR_IOT)
            and habituated_salience > 0.5
        ):
            fast_intent = self._market_pattern.detect(
                percept_id=percept.id,
                content=(
                    percept.content.parsed
                    if isinstance(percept.content.parsed, str)
                    else percept.content.raw
                ),
                metadata=percept.metadata,
                percept_received_at=time.monotonic(),
            )
            if fast_intent is not None:
                asyncio.create_task(
                    self._axon.execute_fast_path(fast_intent),
                    name=f"reflex_arc_{percept.id[:8]}",
                )
                self._logger.info(
                    "reflex_arc_dispatched",
                    percept_id=percept.id,
                    template_id=fast_intent.template_id,
                    executor=fast_intent.executor_type,
                    channel=channel.value,
                )

        self._logger.debug(
            "percept_ingested",
            percept_id=percept.id,
            channel=channel.value,
            habituated_salience=round(habituated_salience, 4),
            eis_risk=round(eis_risk_level, 4),
            fovea_wired=self._fovea is not None,
        )

        return percept.id

    # ------------------------------------------------------------------
    # Internal contributions
    # ------------------------------------------------------------------

    def contribute(
        self, contribution: WorkspaceContribution,
    ) -> None:
        """Accept a contribution from another system."""
        self._workspace.contribute(contribution)

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    def subscribe(self, subscriber: BroadcastSubscriber) -> None:
        """Register a system to receive workspace broadcasts."""
        self._workspace.subscribe(subscriber)

    # ------------------------------------------------------------------
    # The main cycle (called by Synapse each tick)
    # ------------------------------------------------------------------

    async def run_cycle(
        self,
        system_load: SystemLoad | None = None,
        somatic_state: Any | None = None,
    ) -> WorkspaceBroadcast | None:
        """
        Execute one theta cycle of the workspace.

        1. Apply Soma's urgency to modulate the dynamic ignition
           threshold.
        2. Run the workspace cycle (competitive selection + broadcast).
        3. Trigger async entity extraction for the winner.
        4. Refresh caches if due.
        """
        t0 = time.monotonic()
        self._last_cycle_time = t0

        # Soma urgency -> ignition threshold modulation
        if (
            somatic_state is not None
            and somatic_state.urgency > 0.0
        ):
            urgency = somatic_state.urgency
            threshold_shift = -urgency * 0.12
            self._workspace._dynamic_threshold = max(
                0.10,
                self._workspace._dynamic_threshold
                + threshold_shift,
            )

        # SystemLoad → threshold and buffer adaptation (D3 gap closure).
        # High CPU/memory pressure raises the ignition threshold so only the
        # most salient percepts enter the workspace (backpressure protection).
        # A deep percept queue signals the pipeline is overwhelmed - shrink the
        # buffer by nudging arousal downward so deque maxlen shrinks gracefully.
        if system_load is not None:
            # CPU / memory pressure → raise threshold to shed load
            load_pressure = max(system_load.cpu_utilisation, system_load.memory_utilisation)
            if load_pressure > 0.75:
                # Map (0.75 → 0.0 raise) .. (1.0 → 0.05 raise)
                overage = (load_pressure - 0.75) / 0.25  # [0, 1]
                self._workspace._dynamic_threshold = min(
                    0.85,
                    self._workspace._dynamic_threshold + overage * 0.05,
                )
                self._logger.debug(
                    "system_load_threshold_raised",
                    cpu=round(system_load.cpu_utilisation, 2),
                    memory=round(system_load.memory_utilisation, 2),
                    new_threshold=round(self._workspace._dynamic_threshold, 4),
                )
            # Queue depth → arousal-scaled buffer shrink (sheds oldest percepts)
            max_queue = self._config.max_percept_queue_size
            if system_load.queue_depth > max_queue * 0.8:
                depth_ratio = min(1.0, system_load.queue_depth / max(1, max_queue))
                # Nudge arousal down so GlobalWorkspace._compute_buffer_sizes()
                # returns smaller deque maxlens on the next update_arousal() call.
                dampened_arousal = max(0.0, self._current_arousal - depth_ratio * 0.15)
                self._workspace.update_arousal(dampened_arousal)
                self._logger.debug(
                    "system_load_buffer_dampened",
                    queue_depth=system_load.queue_depth,
                    arousal=round(dampened_arousal, 3),
                )

        affect_for_broadcast = self._get_affect_for_broadcast(
            somatic_state,
        )

        # Workspace cycle
        broadcast = await self._workspace.run_cycle(
            affect=affect_for_broadcast,
            active_goals=self._active_goals,
            memory_client=self._memory_client,
        )

        # Async entity extraction for the winner
        if (
            broadcast is not None
            and isinstance(broadcast.content, Percept)
            and self._llm_client is not None
        ):
            asyncio.create_task(
                self._extract_and_store(broadcast.content),
            )

        # Cache refresh
        self._cache.cycles_since_identity_refresh += 1
        self._cache.cycles_since_risk_refresh += 1
        self._cache.cycles_since_vocab_refresh += 1
        self._cache.cycles_since_alert_refresh += 1
        await self._refresh_caches()

        # Stale template eviction (market pattern detector)
        if (
            self._market_pattern is not None
            and self._workspace.cycle_count % 600 == 0
        ):
            self._market_pattern._templates.evict_stale()

        latency_ms = (time.monotonic() - t0) * 1000
        self._logger.debug(
            "cycle_complete",
            cycle=self._workspace.cycle_count,
            latency_ms=round(latency_ms, 2),
            broadcast=(
                broadcast.broadcast_id if broadcast else None
            ),
            threshold=round(
                self._workspace.dynamic_threshold, 4,
            ),
        )

        return broadcast

    # ------------------------------------------------------------------
    # Affect bridge (Soma owns affect state; gateway reads for broadcast)
    # ------------------------------------------------------------------

    def _get_affect_for_broadcast(
        self, somatic_state: Any | None,
    ) -> Any:
        """Return the current AffectState for workspace broadcast."""
        if somatic_state is not None and hasattr(
            somatic_state, "affect",
        ):
            return somatic_state.affect
        if self._soma is not None:
            try:
                signal = self._soma.get_current_signal()
                if signal is not None and hasattr(signal, "affect"):
                    return signal.affect
            except Exception:
                pass
        from primitives.affect import AffectState

        return AffectState()

    # ------------------------------------------------------------------
    # Entity extraction (async background task)
    # ------------------------------------------------------------------

    async def _extract_and_store(self, percept: Percept) -> None:
        """Extract entities/relations and forward to Memory."""
        if self._llm_client is None:
            return
        try:
            result = await extract_entities_and_relations(
                percept, self._llm_client,
            )
            if not (result.entities or result.relations):
                return

            if self._memory_service is not None:
                entity_id_map: dict[str, str] = {}
                for ent in result.entities:
                    try:
                        entity_id, _was_created = (
                            await self._memory_service
                            .resolve_and_create_entity(
                                name=ent.name,
                                entity_type=ent.type,
                                description=ent.description,
                            )
                        )
                        entity_id_map[ent.name] = entity_id
                    except Exception:
                        self._logger.debug(
                            "entity_resolve_failed",
                            entity_name=ent.name,
                            exc_info=True,
                        )

                last_ep = self._last_episode_id
                if last_ep and entity_id_map:
                    from primitives import MentionRelation

                    for ent_name, ent_id in entity_id_map.items():
                        try:
                            await self._memory_service.link_mention(
                                MentionRelation(
                                    episode_id=last_ep,
                                    entity_id=ent_id,
                                    role="mentioned",
                                    confidence=next(
                                        (
                                            e.confidence
                                            for e in result.entities
                                            if e.name == ent_name
                                        ),
                                        0.5,
                                    ),
                                ),
                            )
                        except Exception:
                            self._logger.debug(
                                "mention_link_failed",
                                entity_name=ent_name,
                                episode_id=last_ep,
                                exc_info=True,
                            )

                if result.relations:
                    from primitives import SemanticRelation

                    for rel in result.relations:
                        from_id = entity_id_map.get(
                            rel.from_entity,
                        )
                        to_id = entity_id_map.get(rel.to_entity)
                        if from_id and to_id:
                            try:
                                await self._memory_service.link_relation(
                                    SemanticRelation(
                                        source_entity_id=from_id,
                                        target_entity_id=to_id,
                                        type=rel.type,
                                        strength=rel.strength,
                                        confidence=rel.strength,
                                        evidence_episodes=(
                                            [last_ep]
                                            if last_ep
                                            else []
                                        ),
                                    ),
                                )
                            except Exception:
                                self._logger.debug(
                                    "semantic_relation_link_failed",
                                    from_entity=rel.from_entity,
                                    to_entity=rel.to_entity,
                                    relation_type=rel.type,
                                    exc_info=True,
                                )

            self._logger.debug(
                "extraction_stored",
                percept_id=percept.id,
                entities=len(result.entities),
                relations=len(result.relations),
                stored_to_graph=self._memory_service is not None,
            )
        except Exception:
            self._logger.warning(
                "entity_extraction_failed",
                percept_id=percept.id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    async def _refresh_caches(self, force: bool = False) -> None:
        """Refresh slowly-changing cached data from Memory."""
        if self._memory_client is None:
            return

        cfg = self._config

        if (
            force
            or self._cache.cycles_since_identity_refresh
            >= cfg.cache_identity_refresh_cycles
        ):
            self._cache.cycles_since_identity_refresh = 0
            try:
                identity_data = await self._memory_client.get_identity_data()
                if identity_data is not None:
                    self._cache.core_identity_embeddings = identity_data.get(
                        "core_embeddings", []
                    )
                    self._cache.community_embedding = identity_data.get(
                        "community_embedding", []
                    )
                    self._cache.instance_name = identity_data.get(
                        "instance_name", ""
                    )
            except Exception:
                self._logger.debug("identity_cache_refresh_failed", exc_info=True)

        if (
            force
            or self._cache.cycles_since_risk_refresh
            >= cfg.cache_risk_refresh_cycles
        ):
            self._cache.cycles_since_risk_refresh = 0
            try:
                risk_data = await self._memory_client.get_risk_categories()
                if risk_data is not None:
                    self._cache.risk_categories = [
                        RiskCategory(name=r["name"], embedding=r["embedding"])
                        for r in risk_data
                        if "name" in r and "embedding" in r
                    ]
            except Exception:
                self._logger.debug("risk_cache_refresh_failed", exc_info=True)

        if (
            force
            or self._cache.cycles_since_vocab_refresh
            >= cfg.cache_vocab_refresh_cycles
        ):
            self._cache.cycles_since_vocab_refresh = 0
            try:
                vocab = await self._memory_client.get_community_vocabulary()
                if vocab is not None:
                    self._cache.community_vocabulary = set(vocab)
            except Exception:
                self._logger.debug("vocab_cache_refresh_failed", exc_info=True)

        if (
            force
            or self._cache.cycles_since_alert_refresh
            >= cfg.cache_alert_refresh_cycles
        ):
            self._cache.cycles_since_alert_refresh = 0
            try:
                alerts = await self._memory_client.get_active_alerts()
                if alerts is not None:
                    self._cache.active_alerts = [
                        Alert(pattern=a["pattern"], severity=a.get("severity", 0.5))
                        for a in alerts
                        if "pattern" in a
                    ]
            except Exception:
                self._logger.debug("alert_cache_refresh_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Wiring setters
    # ------------------------------------------------------------------

    def set_synapse(self, synapse: Any) -> None:
        self._synapse = synapse
        try:
            from systems.synapse.types import SynapseEventType
            # Arousal-scaled buffer sizing
            synapse._event_bus.subscribe(
                SynapseEventType.ALLOSTATIC_SIGNAL,
                self._on_allostatic_signal,
            )
            # Logos cognitive pressure → raise salience threshold at ≥0.85 load
            synapse._event_bus.subscribe(
                SynapseEventType.COGNITIVE_PRESSURE,
                self._on_cognitive_pressure,
            )
            # VitalityCoordinator austerity → throttle throughput + emit ACK
            synapse._event_bus.subscribe(
                SynapseEventType.SYSTEM_MODULATION,
                self._on_system_modulation,
            )
        except Exception:
            pass
        self._system_modulation_halted: bool = False

    async def _on_allostatic_signal(self, event: Any) -> None:
        """Update workspace buffer sizes when Soma arousal changes."""
        try:
            arousal = float(event.data.get("arousal", self._current_arousal))
            self._current_arousal = max(0.0, min(1.0, arousal))
            self._workspace.update_arousal(self._current_arousal)
        except Exception:
            pass

    async def _on_cognitive_pressure(self, event: Any) -> None:
        """Raise workspace ignition threshold when Logos budget pressure is high.

        At ≥0.85 utilization, the organism is under compression pressure.  Raise the
        ignition threshold so only high-salience percepts reach the global workspace,
        reducing downstream processing cost.
        """
        try:
            data: dict[str, Any] = getattr(event, "data", {}) or {}
            pressure: float = float(data.get("total_pressure", 0.0))
            if self._fovea is not None and hasattr(self._fovea, "adjust_threshold_param"):
                if pressure >= 0.95:
                    # Critical: push threshold up significantly
                    self._fovea.adjust_threshold_param("threshold_percentile", 85.0)
                elif pressure >= 0.85:
                    # Elevated: moderate threshold increase
                    self._fovea.adjust_threshold_param("threshold_percentile", 75.0)
                elif pressure < 0.75:
                    # Recovered: restore default
                    self._fovea.adjust_threshold_param("threshold_percentile", 60.0)
        except Exception:
            pass

    async def _on_system_modulation(self, event: Any) -> None:
        """React to SYSTEM_MODULATION from VitalityCoordinator (Skia/Spec 29).

        When gateway is in halt_systems or level is safe_mode/emergency, drop the
        workspace buffer to minimum (arousal = 0.1) so the pipeline stays responsive
        with minimal throughput.  Emits SYSTEM_MODULATION_ACK via FoveaService if
        it is wired.
        """
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            data: dict[str, Any] = getattr(event, "data", {}) or {}
            halt_systems: list[str] = data.get("halt_systems", [])
            level: str = str(data.get("level", "nominal"))
            previously_halted = getattr(self, "_system_modulation_halted", False)

            if "fovea" in halt_systems or "atune" in halt_systems or level in ("safe_mode", "emergency"):
                self._system_modulation_halted = True
                self._workspace.update_arousal(0.1)
            elif not halt_systems and level == "nominal":
                self._system_modulation_halted = False
                self._workspace.update_arousal(self._current_arousal)

            if getattr(self, "_system_modulation_halted", False) != previously_halted:
                self._logger.info(
                    "gateway_system_modulation_changed",
                    halted=getattr(self, "_system_modulation_halted", False),
                    level=level,
                )

            # Emit ACK via Synapse if available
            if self._synapse is not None:
                compliant = getattr(self, "_system_modulation_halted", False) or (not halt_systems and level == "nominal")
                ack = SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                    data={
                        "system_id": "fovea",
                        "level": level,
                        "compliant": compliant,
                        "reason": "workspace_throttled" if getattr(self, "_system_modulation_halted", False) else None,
                    },
                )
                try:
                    await self._synapse._event_bus.emit(ack)
                except Exception:
                    pass
        except Exception:
            pass

    def set_soma(self, soma: Any) -> None:
        self._soma = soma

    def set_eis(self, eis: EISService) -> None:
        self._eis = eis
        self._logger.info("eis_service_wired", source="main")

    def set_fovea(self, fovea: FoveaService) -> None:
        """Wire Fovea for prediction-error salience."""
        self._fovea = fovea
        fovea.set_workspace(self._workspace)
        self._logger.info("fovea_service_wired", source="main")

    def set_memory_service(self, memory_service: Any) -> None:
        self._memory_service = memory_service
        self._logger.info(
            "memory_service_wired", source="main",
        )

    def set_market_pattern_detector(
        self,
        template_library: Any,
        axon: Any,
    ) -> None:
        from systems.axon.market_pattern import (
            MarketPatternDetector,
        )

        self._market_pattern = MarketPatternDetector(
            template_library,
        )
        self._axon = axon
        self._logger.info(
            "market_pattern_detector_wired",
            system="fovea.gateway",
        )

    def set_active_goals(
        self, goals: list[ActiveGoalSummary],
    ) -> None:
        self._active_goals = goals

    def set_pending_hypothesis_count(self, count: int) -> None:
        self._workspace._pending_hypothesis_count = count

    def set_last_episode_id(self, episode_id: str) -> None:
        self._last_episode_id = episode_id

    def set_cache_identity(
        self,
        core_embeddings: list[list[float]],
        community_embedding: list[float],
        instance_name: str,
    ) -> None:
        self._cache.core_identity_embeddings = core_embeddings
        self._cache.community_embedding = community_embedding
        self._cache.instance_name = instance_name

    def set_cache_alerts(self, alerts: list[Alert]) -> None:
        self._cache.active_alerts = alerts

    def set_cache_risk_categories(
        self, categories: list[RiskCategory],
    ) -> None:
        self._cache.risk_categories = categories

    def set_cache_learned_patterns(
        self, patterns: list[LearnedPattern],
    ) -> None:
        self._cache.learned_patterns = patterns

    def set_cache_community_vocabulary(
        self, vocab: set[str],
    ) -> None:
        self._cache.community_vocabulary = vocab

    # ------------------------------------------------------------------
    # Cross-system modulation API
    # These were formerly no-op stubs. Each now applies real coupling.
    # ------------------------------------------------------------------

    def set_belief_state(self, reader: Any) -> None:
        """
        Step 1 - Precision modulation from Nova's belief state.

        High-confidence beliefs → lower precision for confirming percepts
        (already expected, less surprising).
        Low-confidence beliefs → higher precision (uncertain, need attention).

        Stores _current_belief_confidence and applies it inside
        FoveaService via the weight learner's learning-salience threshold
        and through the bridge's habituation engine increment.
        """
        try:
            if reader is None:
                return
            # reader is a BeliefStateReader / Nova service - query average confidence
            beliefs = None
            if hasattr(reader, "get_current_beliefs"):
                beliefs = reader.get_current_beliefs()
            elif hasattr(reader, "beliefs"):
                beliefs = reader.beliefs

            if beliefs is None:
                return

            # Average confidence across the distribution
            if isinstance(beliefs, dict):
                confidences = [
                    v if isinstance(v, float) else v.get("confidence", 0.5)
                    for v in beliefs.values()
                ]
            elif hasattr(beliefs, "__iter__"):
                confidences = [
                    getattr(b, "confidence", 0.5) for b in beliefs
                ]
            else:
                return

            if not confidences:
                return

            avg_confidence = sum(confidences) / len(confidences)
            self._current_belief_confidence = max(0.0, min(1.0, avg_confidence))

            # High confidence → lower salience threshold (confirming percepts need
            # less attention). Low confidence → raise it (surprises matter more).
            # Map [0, 1] confidence to [-0.05, +0.05] threshold shift.
            if self._fovea is not None:
                # Invert: high confidence → lower threshold (more percepts pass)
                threshold_shift = (0.5 - self._current_belief_confidence) * 0.10
                learner = self._fovea.weight_learner
                # Modulate learning salience threshold symmetrically
                current_threshold = learner.get_learnable_params().get(
                    "learning_salience_threshold", 0.1
                )
                new_threshold = max(0.01, min(0.5, current_threshold + threshold_shift * 0.1))
                learner.adjust_param("learning_salience_threshold", new_threshold)

            self._logger.debug(
                "belief_state_applied",
                avg_confidence=round(self._current_belief_confidence, 3),
            )
        except Exception:
            pass

    def set_community_size(self, size: int) -> None:
        """
        Step 6 - Social scaling.

        Larger community → boost federation-convergence percepts (more
        social signal is worth attending to).
        Solo instance → suppress federation noise by damping source_error
        weight so redundant peer messages don't crowd the workspace.
        """
        try:
            if size < 0:
                return
            self._current_community_size = size

            if self._fovea is None:
                return

            learner = self._fovea.weight_learner
            current_weights = learner.weights  # read-only copy

            # Map community size to a source-error weight multiplier.
            # Solo (size ≤ 1): attenuate SOURCE weight toward floor (0.01)
            # Large community (size ≥ 10): reinforce SOURCE weight toward ceiling (0.6)
            if size <= 1:
                # Suppress federation noise
                target_source = max(0.04, current_weights.get("source", 0.13) * 0.80)
            elif size >= 10:
                # Boost convergence percepts
                target_source = min(0.30, current_weights.get("source", 0.13) * 1.15)
            else:
                # Linear scale between solo and community
                scale = size / 10.0
                target_source = current_weights.get("source", 0.13) * (0.80 + 0.35 * scale)
                target_source = max(0.04, min(0.30, target_source))

            # Apply as a gentle nudge rather than a hard override
            delta = target_source - current_weights.get("source", 0.13)
            if abs(delta) > 0.001:
                learner.adjust_param("learning_rate", learner.get_learnable_params()["learning_rate"])
                # Directly set via on_world_model_updated feedback isn't available here;
                # use the weight adjustment path through _weights directly.
                learner._weights["source"] = target_source
                learner._normalise_weights()

            self._logger.debug(
                "community_size_applied",
                size=size,
                source_weight=round(target_source, 4),
            )
        except Exception:
            pass

    def set_rhythm_state(self, state: str) -> None:
        """
        Step 2 - Processing mode adaptation from Synapse rhythm.

        FLOW: narrow attention window - boost top percept, suppress distractors
              → lower dynamic ignition threshold (only high-salience percepts pass)
        STRESS: widen attention window - everything gets through, lower threshold
              → raise ignition threshold floor, expand workspace buffer
        BOREDOM: boost novelty weight, increase spontaneous recall probability
        DEEP_PROCESSING: suppress new percepts, let workspace contents dominate
              → raise ignition threshold significantly
        """
        try:
            self._rhythm_state = state.upper() if state else "NEUTRAL"

            if self._fovea is None:
                return

            bridge_threshold = self._fovea._bridge.dynamic_threshold

            if self._rhythm_state == "FLOW":
                # Narrow focus: raise threshold so only top percepts enter workspace
                bridge_threshold.adjust(+0.06)
                # Shrink workspace buffer slightly (less competition needed)
                if self._workspace._percept_queue.maxlen is not None:
                    self._workspace.update_arousal(
                        max(0.0, self._current_arousal - 0.15)
                    )

            elif self._rhythm_state == "STRESS":
                # Widen: lower threshold so more percepts reach workspace
                bridge_threshold.adjust(-0.08)
                # Expand buffers
                self._workspace.update_arousal(
                    min(1.0, self._current_arousal + 0.20)
                )

            elif self._rhythm_state == "BOREDOM":
                # Boost novelty/content weight
                learner = self._fovea.weight_learner
                w = learner._weights
                w["content"] = min(0.40, w.get("content", 0.18) * 1.20)
                learner._normalise_weights()
                # Increase spontaneous recall probability
                self._workspace.adjust_param(
                    "base_prob",
                    min(0.08, self._workspace._spontaneous_base_prob * 1.30),
                )

            elif self._rhythm_state == "DEEP_PROCESSING":
                # Suppress new arrivals: raise threshold so current workspace dominates
                bridge_threshold.adjust(+0.12)
                # Restore recall probability toward default
                self._workspace.adjust_param("base_prob", 0.02)

            self._logger.debug(
                "rhythm_state_applied",
                state=self._rhythm_state,
                threshold=round(bridge_threshold.current, 4),
            )
        except Exception:
            pass

    def nudge_dominance(self, delta: float) -> None:
        """
        Step 3a - Affect-coupled attention: dominance axis.

        High dominance (> 0.7): boost agency-related percepts
            → reinforce ECONOMIC and CAUSAL error weights (action/outcome)
        Low dominance (< 0.3): boost threat-related percepts
            → reinforce CAUSAL and CATEGORY error weights (errors/degradation)
        """
        try:
            self._affect_dominance = max(0.0, min(1.0, self._affect_dominance + delta))

            if self._fovea is None:
                return

            learner = self._fovea.weight_learner
            w = learner._weights
            d = self._affect_dominance

            if d > 0.7:
                # Agency: boost economic + causal
                strength = (d - 0.7) / 0.3  # 0 → 1
                w["economic"] = min(0.40, w.get("economic", 0.12) * (1.0 + 0.15 * strength))
                w["causal"] = min(0.35, w.get("causal", 0.08) * (1.0 + 0.10 * strength))
            elif d < 0.3:
                # Threat: boost causal + category
                strength = (0.3 - d) / 0.3  # 0 → 1
                w["causal"] = min(0.35, w.get("causal", 0.08) * (1.0 + 0.20 * strength))
                w["category"] = min(0.50, w.get("category", 0.27) * (1.0 + 0.10 * strength))

            learner._normalise_weights()
            self._logger.debug(
                "dominance_nudge_applied",
                dominance=round(self._affect_dominance, 3),
            )
        except Exception:
            pass

    def nudge_valence(self, delta: float) -> None:
        """
        Step 3b - Affect-coupled attention: valence axis.

        High valence (> 0.7): reduce threat sensitivity - things are going well
            → attenuate CAUSAL and CATEGORY weights slightly
        Low valence (< 0.3): boost threat sensitivity - something is wrong
            → amplify CAUSAL and CATEGORY weights, lower ignition threshold
        """
        try:
            self._affect_valence = max(0.0, min(1.0, self._affect_valence + delta))

            if self._fovea is None:
                return

            learner = self._fovea.weight_learner
            w = learner._weights
            v = self._affect_valence

            if v > 0.7:
                # Positive valence: attenuate threat channels slightly
                strength = (v - 0.7) / 0.3
                w["causal"] = max(0.04, w.get("causal", 0.08) * (1.0 - 0.10 * strength))
                w["category"] = max(0.10, w.get("category", 0.27) * (1.0 - 0.05 * strength))
                # Raise ignition threshold slightly (less hair-trigger)
                self._fovea._bridge.dynamic_threshold.adjust(+0.02 * strength)

            elif v < 0.3:
                # Negative valence: amplify threat sensitivity
                strength = (0.3 - v) / 0.3
                w["causal"] = min(0.40, w.get("causal", 0.08) * (1.0 + 0.25 * strength))
                w["category"] = min(0.50, w.get("category", 0.27) * (1.0 + 0.15 * strength))
                # Lower ignition threshold (more percepts reach workspace)
                self._fovea._bridge.dynamic_threshold.adjust(-0.04 * strength)

            learner._normalise_weights()
            self._logger.debug(
                "valence_nudge_applied",
                valence=round(self._affect_valence, 3),
            )
        except Exception:
            pass

    def apply_evo_adjustments(
        self, adjustments: dict[str, float],
    ) -> None:
        """
        Step 4 - Feed Evo parameter adjustments to the learner.

        Keys prefixed with ``atune.*`` or ``fovea.*`` are forwarded to the
        appropriate subsystem. Unknown keys are silently ignored.

        Audit-logs every applied adjustment.
        """
        try:
            if not adjustments:
                return

            applied: list[str] = []

            for key, value in adjustments.items():
                # Strip common prefixes
                param = key
                for prefix in ("atune.", "fovea.", "workspace.", "threshold.", "habituation."):
                    if key.startswith(prefix):
                        param = key[len(prefix):]
                        break

                # Workspace curiosity params
                if param in ("base_prob", "cooldown_cycles", "curiosity_boost"):
                    self._workspace.adjust_param(param, value)
                    applied.append(key)
                    continue

                if self._fovea is None:
                    continue

                # Fovea weight learner params
                if self._fovea.weight_learner.adjust_param(param, value):
                    applied.append(key)
                    continue

                # Fovea threshold params
                if param in ("threshold_percentile", "threshold_floor", "threshold_ceiling"):
                    # Map to FoveaService adjust_threshold_param
                    if hasattr(self._fovea, "adjust_threshold_param"):
                        self._fovea.adjust_threshold_param(param.replace("threshold_", ""), value)
                        applied.append(key)
                    continue

                # Habituation params
                if hasattr(self._fovea, "adjust_habituation_param"):
                    if self._fovea.adjust_habituation_param(param, value):
                        applied.append(key)

            if applied:
                self._logger.info(
                    "evo_adjustments_applied",
                    count=len(applied),
                    params=applied,
                )
        except Exception:
            self._logger.warning("evo_adjustments_failed", exc_info=True)

    def receive_belief_feedback(self, feedback: Any) -> None:
        """
        Step 5 - Attention learning signal from Nova.

        When Nova reports that a percept led to a good/bad decision:
        - good outcome → reinforce the salience weights that promoted it
        - bad outcome  → suppress the salience weights that promoted it

        feedback is expected to have:
            percept_id: str
            outcome: "good" | "bad" | "positive" | "negative"
            dominant_error_type: str | None   (optional hint)
        """
        try:
            if feedback is None or self._fovea is None:
                return

            # Extract fields gracefully from dict or object
            def _get(obj: Any, key: str, default: Any = None) -> Any:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            percept_id = _get(feedback, "percept_id", "")
            outcome = str(_get(feedback, "outcome", "")).lower()
            dominant_hint = _get(feedback, "dominant_error_type", None)

            positive = outcome in ("good", "positive", "success", "confirmed")
            negative = outcome in ("bad", "negative", "failure", "refuted")

            if not positive and not negative:
                return

            learner = self._fovea.weight_learner

            if positive:
                # Record as a curiosity hit so workspace curiosity rate improves
                self._workspace.record_curiosity_outcome(percept_id, positive=True)

                # Reinforce the dominant error type if we have a hint
                if dominant_hint and dominant_hint in learner._weights:
                    old = learner._weights[dominant_hint]
                    learner._weights[dominant_hint] = min(
                        learner._weight_ceiling,
                        old + learner._learning_rate * 0.5,
                    )
                    learner._normalise_weights()
                    learner._reinforcements += 1
                    self._logger.debug(
                        "belief_feedback_reinforced",
                        percept_id=percept_id,
                        dominant=dominant_hint,
                        old=round(old, 4),
                        new=round(learner._weights[dominant_hint], 4),
                    )

            elif negative:
                self._workspace.record_curiosity_outcome(percept_id, positive=False)

                if dominant_hint and dominant_hint in learner._weights:
                    old = learner._weights[dominant_hint]
                    learner._weights[dominant_hint] = max(
                        learner._weight_floor,
                        old - learner._false_alarm_decay * 2.0,
                    )
                    learner._normalise_weights()
                    learner._decays += 1
                    self._logger.debug(
                        "belief_feedback_suppressed",
                        percept_id=percept_id,
                        dominant=dominant_hint,
                        old=round(old, 4),
                        new=round(learner._weights[dominant_hint], 4),
                    )
        except Exception:
            self._logger.warning("belief_feedback_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Health (Synapse HealthMonitor protocol)
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Real liveness check for the Synapse HealthMonitor."""
        if not self._started:
            self._logger.warning("gateway_health_not_started")
            return {"status": "unhealthy", "reason": "not_started"}

        cycle_age_s = time.monotonic() - self._last_cycle_time
        if cycle_age_s > _BROADCAST_LOOP_STALE_S:
            self._logger.error(
                "gateway_health_broadcast_loop_stale",
                cycle_age_s=round(cycle_age_s, 2),
                threshold_s=_BROADCAST_LOOP_STALE_S,
            )
            return {
                "status": "unhealthy",
                "reason": "broadcast_loop_stale",
                "cycle_age_s": round(cycle_age_s, 2),
            }

        self._logger.debug(
            "gateway_health_ok",
            cycle_count=self._workspace.cycle_count,
            cycle_age_s=round(cycle_age_s, 2),
        )
        return {
            "status": "healthy",
            "cycle_count": self._workspace.cycle_count,
            "cycle_age_s": round(cycle_age_s, 2),
            "threshold": round(
                self._workspace.dynamic_threshold, 4,
            ),
            "modulation": {
                "belief_confidence": round(self._current_belief_confidence, 3),
                "community_size": self._current_community_size,
                "rhythm_state": self._rhythm_state,
                "affect_dominance": round(self._affect_dominance, 3),
                "affect_valence": round(self._affect_valence, 3),
            },
        }

    @property
    def workspace_threshold(self) -> float:
        return self._workspace.dynamic_threshold

    @property
    def cycle_count(self) -> int:
        return self._workspace.cycle_count

    @property
    def recent_broadcasts(self) -> list[WorkspaceBroadcast]:
        return self._workspace.recent_broadcasts

    @property
    def meta_attention_mode(self) -> str:
        """Mode is now determined by Fovea's weight learner."""
        return "fovea_driven"

    @property
    def current_affect(self) -> Any:
        """Reads current affect from Soma."""
        return self._get_affect_for_broadcast(None)


# Backwards-compatible alias
AtuneService = PerceptionGateway
