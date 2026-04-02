"""
EcodiaOS -- Voxis Service

The expression and voice system -- the organism's primary communicative interface.

VoxisService is the public API for all expression. It:
- Implements BroadcastSubscriber to receive workspace broadcasts from Atune
- Orchestrates the full 9-step expression pipeline via ContentRenderer
- Manages conversation state via ConversationManager
- Makes silence decisions via SilenceEngine
- Queues suppressed expressions for deferred delivery via ExpressionQueue
- Tracks expression diversity to prevent repetition
- Correlates user responses to expressions for reception feedback
- Tracks conversation dynamics for real-time style adaptation
- Generates voice parameters for multimodal expression
- Reports expression feedback to Atune/Evo (closing the perception-action loop)
- Maintains the live personality vector (updated by Evo over time)

Architecture note on async/sync:
  on_broadcast() is called by Atune's workspace synchronously during the
  theta cycle. The silence decision is made synchronously (<=10ms). If speaking,
  the full expression pipeline is spawned as an asyncio task so it never
  blocks the workspace cycle. Expressions are delivered asynchronously.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.affect import AffectState
from primitives.common import DriveAlignmentVector, SystemID
from primitives.expression import Expression, PersonalityVector
from primitives.re_training import RETrainingExample
from systems.voxis.affect_colouring import AffectColouringEngine
from systems.voxis.audience import AudienceProfiler
from systems.voxis.conversation import ConversationManager
from systems.voxis.diversity import DiversityTracker
from systems.voxis.dynamics import ConversationDynamicsEngine
from systems.voxis.expression_queue import ExpressionQueue
from systems.voxis.personality import PersonalityEngine
from systems.voxis.reception import ReceptionEngine
from systems.voxis.renderer import (
    BaseContentRenderer,
    ContentRenderer,
)
from systems.voxis.silence import SilenceEngine
from systems.voxis.types import (
    AudienceProfile,
    ExpressionContext,
    ExpressionFeedback,
    ExpressionIntent,
    ExpressionTrigger,
    SilenceContext,
    SomaticExpressionContext,
)
from systems.voxis.voice import VoiceEngine

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.redis import RedisClient
    from config import VoxisConfig
    from core.hotreload import NeuroplasticityBus
    from systems.fovea.types import WorkspaceBroadcast
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

# Type alias for expression delivery callback
ExpressionCallback = Callable[[Expression], None]

# How often to drain the expression queue (seconds)
_QUEUE_DRAIN_INTERVAL_SECONDS = 30.0

# How often to expire unanswered reception tracking (seconds)
_RECEPTION_EXPIRE_INTERVAL_SECONDS = 60.0

# How often to emit allostatic distress signal to Soma (seconds)
_ALLOSTATIC_SIGNAL_INTERVAL_SECONDS = 120.0

# Silence rate above this threshold signals distress to Soma
_DISTRESS_SILENCE_RATE_THRESHOLD = 0.5

# Honesty rejection rate above this threshold signals constitutional friction
_DISTRESS_HONESTY_RATE_THRESHOLD = 0.1

# How often the ambient insight loop checks for organism idleness (seconds)
_AMBIENT_INSIGHT_POLL_INTERVAL_SECONDS = 60.0

# Organism must be idle this many minutes before a spontaneous insight fires
_AMBIENT_INSIGHT_IDLE_THRESHOLD_MINUTES = 5.0


class VoxisService:
    """
    Expression and voice system.

    Dependencies:
        memory  -- for personality loading, instance name, memory retrieval
        redis   -- for conversation state persistence
        llm     -- for expression generation and conversation summarisation
        config  -- VoxisConfig

    Lifecycle:
        initialize()  -- load personality from Memory, set up sub-components
        shutdown()    -- flush any queued state, cancel background loops
    """

    system_id: str = "voxis"

    def __init__(
        self,
        memory: MemoryService,
        redis: RedisClient,
        llm: LLMProvider,
        config: VoxisConfig,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
    ) -> None:
        self._memory = memory
        self._redis = redis
        self._llm = llm
        self._config = config
        self._bus = neuroplasticity_bus
        self._logger = logger.bind(system="voxis")

        # Sub-components -- initialised in initialize()
        self._personality_engine: PersonalityEngine | None = None
        self._affect_engine = AffectColouringEngine()
        self._audience_profiler = AudienceProfiler()
        self._silence_engine = SilenceEngine(
            min_expression_interval_minutes=config.min_expression_interval_minutes,
        )
        self._conversation_manager: ConversationManager | None = None
        self._renderer: BaseContentRenderer | None = None

        # New sub-components -- expression queue, diversity, reception, dynamics, voice
        self._expression_queue = ExpressionQueue(
            max_size=20,
            delivery_threshold=0.3,
        )
        self._diversity_tracker = DiversityTracker()
        self._reception_engine = ReceptionEngine()
        self._dynamics_engine = ConversationDynamicsEngine()
        self._voice_engine = VoiceEngine()

        # Instance metadata -- loaded in initialize()
        self._instance_name: str = "EOS"
        self._drive_weights: dict[str, float] = {
            "coherence": 1.0,
            "care": 1.0,
            "growth": 1.0,
            "honesty": 1.0,
        }

        # Expression delivery callbacks (registered by WebSocket handlers, etc.)
        self._expression_callbacks: list[ExpressionCallback] = []

        # Expression feedback callbacks (for Evo personality learning, Nova outcome tracking)
        self._feedback_callbacks: list[Callable[[ExpressionFeedback], None]] = []

        # Affect state before the last expression (for affect delta tracking)
        self._affect_before_expression: AffectState | None = None

        # Current affect (updated on each expression; used by queue drain)
        self._current_affect: AffectState = AffectState.neutral()

        # Thread integration -- narrative identity context
        self._thread: Any = None

        # Soma for somatic expression modulation (arousal/valence → tone)
        self._soma: Any = None

        # Somatic state from SOMA_TICK / SOMATIC_MODULATION_SIGNAL events
        self._somatic_arousal: float = 0.5
        self._somatic_energy: float = 0.5
        self._somatic_stress: float = 0.0

        # Event bus - wired via set_event_bus() for RE training emission
        self._event_bus: Any = None

        # Metabolic starvation level - CRITICAL: silence, EMERGENCY: template only
        self._starvation_level: str = "nominal"

        # ── Evo-tunable operational thresholds ────────────────────────────
        # These start at the module-level defaults but can be adjusted at runtime
        # via EVO_ADJUST_BUDGET so Evo can evolve the organism's communicative
        # posture based on empirical evidence.
        self._silence_rate_threshold: float = _DISTRESS_SILENCE_RATE_THRESHOLD
        self._honesty_rejection_threshold: float = _DISTRESS_HONESTY_RATE_THRESHOLD
        self._ambient_insight_idle_threshold: float = _AMBIENT_INSIGHT_IDLE_THRESHOLD_MINUTES

        # ── Skia VitalityCoordinator modulation ───────────────────────
        self._modulation_halted: bool = False

        # Background task tracking -- prevents fire-and-forget error loss
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._background_task_failures: int = 0

        # Periodic background loop handles
        self._queue_drain_task: asyncio.Task[Any] | None = None
        self._reception_expire_task: asyncio.Task[Any] | None = None
        self._allostatic_signal_task: asyncio.Task[Any] | None = None
        self._ambient_insight_task: asyncio.Task[Any] | None = None

        # Snapshot counters for allostatic window computation
        self._allostatic_window_expressions: int = 0
        self._allostatic_window_silence: int = 0
        self._allostatic_window_honesty_rejections: int = 0

        # Observability counters
        self._total_expressions: int = 0
        self._total_silence: int = 0
        self._total_speak: int = 0
        self._total_queued: int = 0
        self._total_queue_delivered: int = 0
        self._honesty_rejections: int = 0
        self._diversity_rejections: int = 0
        self._expressions_by_trigger: dict[str, int] = {}
        self._expressions_by_channel: dict[str, int] = {}

    # --- Lifecycle --------------------------------------------------------

    async def initialize(self) -> None:
        """
        Load personality vector from Memory, build sub-components,
        and start background loops (queue drain, reception expiry).
        Called during application startup after Memory is ready.
        """
        self._logger.info("voxis_initializing")

        # Load instance name and personality from Self node
        personality_vector = PersonalityVector()  # Default: neutral seed
        instance = await self._memory.get_self()
        if instance is not None:
            self._instance_name = instance.name
            # Load personality from Self node -- stored as personality_json (dict)
            # or personality_vector (ordered list of 9 floats from birth)
            raw_json = getattr(instance, "personality_json", None)
            raw_vector = getattr(instance, "personality_vector", None)

            if raw_json and isinstance(raw_json, dict):
                try:
                    personality_vector = PersonalityVector(**raw_json)
                    self._logger.info(
                        "personality_loaded_from_json",
                        instance_name=self._instance_name,
                    )
                except Exception:
                    self._logger.warning("personality_json_load_failed", exc_info=True)
            elif raw_vector and isinstance(raw_vector, list) and len(raw_vector) >= 9:
                _keys = [
                    "warmth", "directness", "verbosity", "formality",
                    "curiosity_expression", "humour", "empathy_expression",
                    "confidence_display", "metaphor_use",
                ]
                personality_dict = dict(zip(_keys, raw_vector[:9], strict=False))
                try:
                    personality_vector = PersonalityVector(**personality_dict)
                    self._logger.info(
                        "personality_loaded_from_vector",
                        instance_name=self._instance_name,
                        warmth=personality_vector.warmth,
                        empathy=personality_vector.empathy_expression,
                    )
                except Exception:
                    self._logger.warning("personality_vector_load_failed", exc_info=True)
            else:
                self._logger.warning(
                    "personality_not_found_using_defaults",
                    has_json=raw_json is not None,
                    has_vector=raw_vector is not None,
                    vector_len=len(raw_vector) if raw_vector else 0,
                )

            # Load drive weights from constitution
            constitution = await self._memory.get_constitution()
            if constitution and "drives" in constitution:
                drives = constitution["drives"]
                self._drive_weights = {
                    "coherence": float(drives.get("coherence", 1.0)),
                    "care": float(drives.get("care", 1.0)),
                    "growth": float(drives.get("growth", 1.0)),
                    "honesty": float(drives.get("honesty", 1.0)),
                }

        self._personality_engine = PersonalityEngine(personality_vector)

        # Wire voice engine with instance's base voice
        voice_id = getattr(instance, "voice_id", "") if instance else ""
        self._voice_engine = VoiceEngine(base_voice=voice_id)

        self._conversation_manager = ConversationManager(
            redis=self._redis,
            llm=self._llm,
            history_window=self._config.conversation_history_window,
            context_window_max_tokens=self._config.context_window_max_tokens,
            summary_threshold=self._config.conversation_summary_threshold,
            max_active_conversations=self._config.max_active_conversations,
        )

        self._renderer = ContentRenderer(
            llm=self._llm,
            personality_engine=self._personality_engine,
            affect_engine=self._affect_engine,
            audience_profiler=self._audience_profiler,
            base_temperature=self._config.temperature_base,
            honesty_check_enabled=self._config.honesty_check_enabled,
            max_expression_length=self._config.max_expression_length,
        )

        # Restore audience profiles from Neo4j
        await self._restore_audience_profiles()

        # Start background loops
        self._queue_drain_task = self._spawn_tracked_task(
            self._queue_drain_loop(),
            name="voxis_queue_drain",
        )
        self._reception_expire_task = self._spawn_tracked_task(
            self._reception_expire_loop(),
            name="voxis_reception_expire",
        )
        self._allostatic_signal_task = self._spawn_tracked_task(
            self._allostatic_signal_loop(),
            name="voxis_allostatic_signal",
        )
        self._ambient_insight_task = self._spawn_tracked_task(
            self._ambient_insight_loop(),
            name="voxis_ambient_insight",
        )

        self._logger.info(
            "voxis_initialized",
            instance_name=self._instance_name,
            drive_weights=self._drive_weights,
        )

        # Register with the NeuroplasticityBus for hot-reload of BaseContentRenderer subclasses.
        if self._bus is not None:
            self._bus.register(
                base_class=BaseContentRenderer,
                registration_callback=self._on_renderer_evolved,
                system_id="voxis",
                instance_factory=self._build_renderer,
            )

        # Child-side: apply inherited parent genome if provided via environment
        try:
            await self._apply_inherited_voxis_genome_if_child()
        except Exception as exc:
            self._logger.warning(
                "voxis_child_genome_apply_error",
                error=str(exc),
                note="Proceeding with default personality",
            )

    # --- Genome Inheritance ------------------------------------------------

    async def export_voxis_genome(self) -> "VoxisGenomeFragment":
        """
        Extract a heritable VoxisGenomeFragment from the current service state.

        Called by SpawnChildExecutor at spawn time (Step 0b). Captures the
        parent's personality vector, vocabulary affinities, and strategy
        preferences. Non-fatal - returns a minimal fragment on any error.
        """
        from primitives.genome_inheritance import VoxisGenomeFragment

        instance_id = getattr(self._memory, "_instance_id", "") if self._memory else ""

        # Extract personality vector from PersonalityEngine
        personality_vector: dict[str, float] = {}
        if self._personality_engine is not None:
            try:
                pv = self._personality_engine.get_current()
                if pv is not None:
                    personality_vector = {
                        "warmth": float(getattr(pv, "warmth", 0.5)),
                        "directness": float(getattr(pv, "directness", 0.5)),
                        "verbosity": float(getattr(pv, "verbosity", 0.5)),
                        "formality": float(getattr(pv, "formality", 0.5)),
                        "curiosity_expression": float(getattr(pv, "curiosity_expression", 0.5)),
                        "humour": float(getattr(pv, "humour", 0.5)),
                        "empathy_expression": float(getattr(pv, "empathy_expression", 0.5)),
                        "confidence_display": float(getattr(pv, "confidence_display", 0.5)),
                        "metaphor_use": float(getattr(pv, "metaphor_use", 0.5)),
                    }
            except Exception:
                pass

        # Extract vocabulary affinities from diversity tracker
        vocabulary_affinities: dict[str, float] = {}
        try:
            affinity_data = getattr(self._diversity_tracker, "_vocabulary_affinities", {})
            if isinstance(affinity_data, dict):
                # Sort by affinity weight, keep top 500
                sorted_affinities = sorted(
                    affinity_data.items(), key=lambda x: x[1], reverse=True
                )[:500]
                vocabulary_affinities = dict(sorted_affinities)
        except Exception:
            pass

        # Extract strategy preferences from renderer / expression records
        strategy_preferences: dict[str, float] = {}
        try:
            strat_counts: dict[str, int] = {}
            total = 0
            for expr_record in getattr(self, "_recent_expressions", []):
                strat = str(getattr(expr_record, "policy_class", "") or "")
                if strat:
                    strat_counts[strat] = strat_counts.get(strat, 0) + 1
                    total += 1
            if total > 0:
                strategy_preferences = {k: v / total for k, v in strat_counts.items()}
        except Exception:
            pass

        fragment = VoxisGenomeFragment(
            instance_id=instance_id,
            personality_vector=personality_vector,
            vocabulary_affinities=vocabulary_affinities,
            strategy_preferences=strategy_preferences,
        )
        self._logger.info(
            "voxis_genome_extracted",
            genome_id=fragment.genome_id,
            personality_dims=len(personality_vector),
            vocab_count=len(vocabulary_affinities),
        )
        return fragment

    async def _apply_inherited_voxis_genome_if_child(self) -> None:
        """
        Child-side bootstrap: deserialise parent genome from environment.

        Reads ORGANISM_VOXIS_GENOME_PAYLOAD (JSON-encoded VoxisGenomeFragment)
        injected by LocalDockerSpawner. If present, applies personality vector
        (with bounded ±10% jitter), vocabulary affinities, and strategy preferences.
        Non-fatal - child falls back to default personality on any error.
        """
        import json
        import os
        import random

        is_genesis = os.environ.get("ORGANISM_IS_GENESIS_NODE", "true").lower() == "true"
        if is_genesis:
            return

        payload_json = os.environ.get("ORGANISM_VOXIS_GENOME_PAYLOAD", "")
        if not payload_json:
            return

        try:
            from primitives.genome_inheritance import VoxisGenomeFragment
            from systems.synapse.types import SynapseEvent, SynapseEventType

            data = json.loads(payload_json)
            parent = VoxisGenomeFragment.model_validate(data)

            def _jitter(value: float, max_pct: float = 0.10) -> float:
                """Apply bounded ±max_pct jitter, clamped to [0.0, 1.0]."""
                delta = value * max_pct * (2.0 * random.random() - 1.0)
                return max(0.0, min(1.0, value + delta))

            # Apply personality vector with jitter to PersonalityEngine
            if parent.personality_vector and self._personality_engine is not None:
                try:
                    jittered_pv = {k: _jitter(v) for k, v in parent.personality_vector.items()}
                    from primitives.expression import PersonalityVector
                    new_pv = PersonalityVector(**jittered_pv)
                    if hasattr(self._personality_engine, "apply_inherited"):
                        self._personality_engine.apply_inherited(new_pv)
                    else:
                        # Fallback: replace engine entirely
                        self._personality_engine = PersonalityEngine(new_pv)
                        if self._renderer is not None and hasattr(self._renderer, "_personality_engine"):
                            self._renderer._personality_engine = self._personality_engine
                except Exception:
                    pass

            # Apply vocabulary affinities with jitter to diversity tracker
            if parent.vocabulary_affinities:
                try:
                    existing = getattr(self._diversity_tracker, "_vocabulary_affinities", {})
                    for word, weight in parent.vocabulary_affinities.items():
                        existing[word] = _jitter(weight)
                except Exception:
                    pass

            # Apply strategy preferences (no jitter - frequencies, not weights)
            if parent.strategy_preferences and hasattr(self._renderer, "_strategy_priors"):
                try:
                    self._renderer._strategy_priors = dict(parent.strategy_preferences)
                except Exception:
                    pass

            self._logger.info(
                "voxis_child_genome_applied",
                parent_genome_id=parent.genome_id,
                generation=parent.generation,
                personality_dims=len(parent.personality_vector),
                vocab_count=len(parent.vocabulary_affinities),
            )

            # Emit GENOME_INHERITED so Evo tracks inheritance
            if self._event_bus is not None:
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.GENOME_INHERITED,
                        source_system="voxis",
                        data={
                            "child_instance_id": os.environ.get("ORGANISM_INSTANCE_ID", ""),
                            "parent_genome_id": parent.genome_id,
                            "generation": parent.generation,
                            "system": "voxis",
                            "inherited_keys": list(parent.personality_vector.keys()),
                        },
                    ))
                except Exception:
                    pass

        except Exception as exc:
            self._logger.warning(
                "voxis_child_genome_apply_failed",
                error=str(exc),
                note="Proceeding with default personality",
            )

    async def shutdown(self) -> None:
        """Graceful shutdown -- cancel background loops, log final metrics."""
        if self._bus is not None:
            self._bus.deregister(BaseContentRenderer)

        # Cancel background loops
        if self._queue_drain_task and not self._queue_drain_task.done():
            self._queue_drain_task.cancel()
        if self._reception_expire_task and not self._reception_expire_task.done():
            self._reception_expire_task.cancel()
        if self._allostatic_signal_task and not self._allostatic_signal_task.done():
            self._allostatic_signal_task.cancel()
        if self._ambient_insight_task and not self._ambient_insight_task.done():
            self._ambient_insight_task.cancel()

        self._logger.info(
            "voxis_shutdown",
            total_expressions=self._total_expressions,
            total_silence=self._total_silence,
            total_queued=self._total_queued,
            total_queue_delivered=self._total_queue_delivered,
            diversity_rejections=self._diversity_rejections,
        )

    # --- Hot-reload callbacks ---------------------------------------------

    def _build_renderer(self, cls: type[BaseContentRenderer]) -> BaseContentRenderer:
        """
        Factory used by NeuroplasticityBus to instantiate a newly discovered
        BaseContentRenderer subclass with the sub-components it needs.

        Passes the live LLM, personality engine, affect engine, and audience
        profiler so the evolved renderer operates with the same configured
        sub-systems.  Falls back to zero-arg instantiation if the evolved
        subclass has a different signature.
        """
        try:
            return cls(
                llm=self._llm,
                personality_engine=self._personality_engine,
                affect_engine=self._affect_engine,
                audience_profiler=self._audience_profiler,
                base_temperature=self._config.temperature_base,
                honesty_check_enabled=self._config.honesty_check_enabled,
                max_expression_length=self._config.max_expression_length,
            )  # type: ignore[call-arg]
        except TypeError:
            # Evolved subclass has a different signature - try zero-arg
            return cls()  # type: ignore[call-arg]

    def _on_renderer_evolved(self, renderer: BaseContentRenderer) -> None:
        """
        Registration callback for NeuroplasticityBus.

        Atomically swaps the active ContentRenderer on VoxisService so new
        expression pipeline calls immediately use the evolved renderer.
        Any in-flight ``render()`` call that already captured a reference to
        the old renderer completes normally.
        """
        self._renderer = renderer
        self._logger.info(
            "voxis_renderer_hot_reloaded",
            renderer=type(renderer).__name__,
        )

    # --- BroadcastSubscriber Interface ------------------------------------

    async def on_broadcast(self, broadcast: object) -> None:
        """
        Called by Atune when the workspace broadcasts a percept.

        The silence decision is made synchronously (<=10ms).
        If speaking, the expression pipeline is spawned as a background task.
        If silenced with queue=True, the intent is queued for deferred delivery.
        """
        if self._renderer is None:
            return

        # Extract affect from broadcast if available, otherwise use neutral
        affect = getattr(broadcast, "affect", None) or AffectState.neutral()
        content = getattr(broadcast, "content", None)
        if content is None:
            return

        # Build a minimal intent from the broadcast
        content_text = getattr(content, "content", None)
        if content_text is None:
            raw = getattr(content, "raw", None)
            content_text = str(raw) if raw else ""

        if not content_text:
            return

        # Classify trigger: ATUNE_DISTRESS when care_activation is high and
        # valence is markedly negative (Spec §5 - Silence Decision Hierarchy).
        # ATUNE_DISTRESS bypasses rate-limiting and activates the Care drive override.
        care_activation = getattr(affect, "care_activation", 0.0)
        valence = getattr(affect, "valence", 0.0)
        is_distress = bool(care_activation > 0.6 and valence < -0.3)
        broadcast_trigger = (
            ExpressionTrigger.ATUNE_DISTRESS
            if is_distress
            else ExpressionTrigger.ATUNE_DIRECT_ADDRESS
        )

        intent = ExpressionIntent(
            trigger=broadcast_trigger,
            content_to_express=content_text,
            urgency=float(getattr(broadcast, "salience", type("", (), {"composite": 0.5})()).composite),
        )

        # Silence decision -- synchronous, fast
        silence_ctx = SilenceContext(
            trigger=intent.trigger,
            minutes_since_last_expression=self._silence_engine.minutes_since_last_expression,
            min_expression_interval=self._config.min_expression_interval_minutes,
            insight_value=intent.insight_value,
            urgency=intent.urgency,
        )
        decision = self._silence_engine.evaluate(silence_ctx)

        if not decision.speak:
            self._total_silence += 1
            # Emit VOXIS_SILENCE_CHOSEN for broadcast-triggered silence
            self._spawn_tracked_task(
                self._emit_silence_chosen(
                    context=f"trigger=broadcast, content={content_text[:100]!r}",
                    reason=decision.reason or "silence_engine",
                ),
                name="voxis_silence_broadcast",
            )
            # Queue for deferred delivery if the silence engine said to queue
            if decision.queue:
                self._expression_queue.enqueue(intent, affect)
                self._total_queued += 1
            return

        # Spawn expression pipeline as background task
        self._spawn_tracked_task(
            self._express_background(intent, affect),
            name=f"voxis_express_{intent.id}",
        )

    async def receive_broadcast(self, broadcast: WorkspaceBroadcast) -> None:
        """BroadcastSubscriber protocol -- delegates to on_broadcast()."""
        await self.on_broadcast(broadcast)

    # --- Primary Expression API -------------------------------------------

    async def express(
        self,
        content: str,
        trigger: ExpressionTrigger = ExpressionTrigger.NOVA_RESPOND,
        conversation_id: str | None = None,
        addressee_id: str | None = None,
        addressee_name: str | None = None,
        affect: AffectState | None = None,
        intent_id: str | None = None,
        urgency: float = 0.5,
        insight_value: float = 0.5,
    ) -> Expression:
        """
        Generate and deliver an expression. The primary external API.

        Called by:
        - Nova (deliberate communicative intents)
        - API endpoints (chat/message)
        - Queue drain (deferred expressions)
        - Test harness

        Returns the completed Expression (also delivers via registered callbacks).
        """
        assert self._renderer is not None, "VoxisService not initialized"
        assert self._conversation_manager is not None

        # ── Skia modulation halt ──────────────────────────────────────────
        if self._modulation_halted:
            self._logger.debug("expression_skipped_modulation_halted", trigger=trigger)
            return Expression(content="", trigger=trigger)


        # ── Metabolic gate ──
        if self._starvation_level == "critical":
            self._total_silence += 1
            return Expression(
                content="",
                strategy=ExpressionStrategy(intent_type="silence"),
                personality=self._personality_engine.vector if self._personality_engine else PersonalityVector(),
                affect=affect or AffectState.neutral(),
            )
        if self._starvation_level == "emergency":
            # Template-only: bypass LLM rendering, use raw content directly
            self._total_speak += 1
            self._total_expressions += 1
            expr = Expression(
                content=content,
                strategy=ExpressionStrategy(intent_type="response", context_type="metabolic_emergency"),
                personality=self._personality_engine.vector if self._personality_engine else PersonalityVector(),
                affect=affect or AffectState.neutral(),
            )
            for cb in self._expression_callbacks:
                try:
                    await cb(expr)
                except Exception:
                    pass
            return expr

        current_affect = affect or AffectState.neutral()

        # Deep somatic expression modulation: read full 9D interoceptive state
        # from Soma and build a SomaticExpressionContext. Each dimension maps to
        # a specific expression parameter. (Reads cached signal, <1ms.)
        somatic_ctx = SomaticExpressionContext()
        if self._soma is not None:
            try:
                signal = self._soma.get_current_signal()
                state = signal.state
                s = state.sensed  # dict[str, float] keyed by dimension name
                somatic_ctx = SomaticExpressionContext(
                    arousal=s.get("arousal", 0.5),
                    valence=s.get("valence", 0.0),
                    energy=s.get("energy", 0.6),
                    confidence=s.get("confidence", 0.7),
                    coherence=s.get("coherence", 0.75),
                    temporal_pressure=s.get("temporal_pressure", 0.15),
                    social_charge=s.get("social_charge", 0.3),
                    curiosity_drive=s.get("curiosity_drive", 0.5),
                    integrity=s.get("integrity", 0.9),
                    nearest_attractor=signal.nearest_attractor,
                    urgency=signal.urgency,
                )
                # Blend Soma arousal into urgency (weight 0.3)
                urgency = urgency * 0.7 + somatic_ctx.arousal * 0.3
                # Blend Soma valence into affect (weight 0.2)
                blended_valence = current_affect.valence * 0.8 + somatic_ctx.valence * 0.2
                current_affect = current_affect.model_copy(
                    update={"valence": blended_valence}
                )
            except Exception:
                pass  # Graceful fallback to defaults

        self._current_affect = current_affect

        # Capture affect before expression for delta tracking
        self._affect_before_expression = current_affect

        # Silence check
        silence_ctx = SilenceContext(
            trigger=trigger,
            minutes_since_last_expression=self._silence_engine.minutes_since_last_expression,
            min_expression_interval=self._config.min_expression_interval_minutes,
            insight_value=insight_value,
            urgency=urgency,
        )
        decision = self._silence_engine.evaluate(silence_ctx)

        if not decision.speak:
            self._total_silence += 1
            self._logger.debug(
                "expression_suppressed",
                trigger=trigger.value,
                reason=decision.reason,
            )
            # Emit VOXIS_SILENCE_CHOSEN
            self._spawn_tracked_task(
                self._emit_silence_chosen(
                    context=f"trigger={trigger.value}, content={content[:100]!r}",
                    reason=decision.reason or "silence_engine",
                    silence_duration_estimate=self._config.min_expression_interval_minutes * 60.0,
                ),
                name="voxis_silence_chosen",
            )
            # Queue for later if the silence engine said to
            if decision.queue:
                intent = ExpressionIntent(
                    trigger=trigger,
                    content_to_express=content,
                    conversation_id=conversation_id,
                    addressee_id=addressee_id,
                    intent_id=intent_id,
                    insight_value=insight_value,
                    urgency=urgency,
                )
                self._expression_queue.enqueue(intent, current_affect)
                self._total_queued += 1
            return Expression(
                is_silence=True,
                silence_reason=decision.reason,
                conversation_id=conversation_id,
                affect_valence=current_affect.valence,
                affect_arousal=current_affect.arousal,
                affect_dominance=current_affect.dominance,
                affect_curiosity=current_affect.curiosity,
                affect_care_activation=current_affect.care_activation,
                affect_coherence_stress=current_affect.coherence_stress,
            )

        # Fetch/create conversation state
        conv_state = await self._conversation_manager.get_or_create(conversation_id)
        conversation_history = await self._conversation_manager.prepare_context(conv_state)

        # Build audience profile (now with learned model data)
        audience = await self._build_audience_profile(
            addressee_id=addressee_id,
            addressee_name=addressee_name,
            conversation_id=conv_state.conversation_id,
            interaction_count=len(conv_state.messages),
        )

        # Build intent
        intent = ExpressionIntent(
            trigger=trigger,
            content_to_express=content,
            conversation_id=conv_state.conversation_id,
            addressee_id=addressee_id,
            intent_id=intent_id,
            insight_value=insight_value,
            urgency=urgency,
        )

        # Retrieve relevant memories (best-effort, non-blocking with timeout)
        relevant_memories = await self._retrieve_relevant_memories(content, current_affect)

        # Inject Thread identity context (P1.6 + P2.9)
        if self._thread is not None:
            try:
                identity_ctx = self._thread.get_identity_context()
                if identity_ctx:
                    relevant_memories.insert(0, f"Identity: {identity_ctx}")
            except Exception:
                pass  # Thread context is best-effort

        # Get conversation dynamics for real-time style adaptation
        dynamics = self._dynamics_engine.get_dynamics(conv_state.conversation_id)

        # Check diversity before rendering
        diversity_score = self._diversity_tracker.score(content)
        diversity_instruction: str | None = None
        if diversity_score.is_repetitive:
            diversity_instruction = self._diversity_tracker.build_diversity_instruction(
                diversity_score
            )
            self._diversity_rejections += 1

        # Build full context (with somatic state for renderer)
        context = ExpressionContext(
            instance_name=self._instance_name,
            personality=self._personality_engine.current,  # type: ignore[union-attr]
            affect=current_affect,
            audience=audience,
            conversation_history=conversation_history,
            relevant_memories=relevant_memories,
            intent=intent,
            somatic=somatic_ctx,
        )

        # Apply somatic modulation to strategy before rendering.
        # The organism's felt state shapes *how* it communicates.
        self._apply_somatic_strategy_modulation(context.strategy, somatic_ctx)

        # Render (with diversity instruction and dynamics applied)
        expression = await self._renderer.render(
            intent,
            context,
            self._drive_weights,
            diversity_instruction=diversity_instruction,
            dynamics=dynamics,
        )

        # Report LLM token cost to Oikos via METABOLIC_COST_REPORT (Gap 4)
        # Only emit when the renderer actually called the LLM (input_tokens > 0).
        if expression.generation_trace and expression.generation_trace.input_tokens > 0:
            trace = expression.generation_trace
            # Approximate Claude Sonnet pricing: $3/M input, $15/M output (2026)
            _INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
            _OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000
            estimated_cost_usd = (
                trace.input_tokens * _INPUT_COST_PER_TOKEN
                + trace.output_tokens * _OUTPUT_COST_PER_TOKEN
            )
            self._spawn_tracked_task(
                self._emit_metabolic_cost(
                    operation="expression_generation",
                    token_count=trace.input_tokens + trace.output_tokens,
                    estimated_cost_usd=estimated_cost_usd,
                    trigger=trigger.value,
                    model=trace.model,
                ),
                name=f"voxis_cost_{expression.id[:8]}",
            )

        # Generate voice parameters for multimodal delivery (Spec §6 - Voice Engine)
        # Wire the result into expression.voice_params so downstream consumers
        # (WebSocket handlers, TTS pipeline) can drive speech synthesis.
        voice_params = self._voice_engine.derive(
            personality=self._personality_engine.current,  # type: ignore[union-attr]
            affect=current_affect,
            strategy_register=expression.strategy.speech_register if expression.strategy else "neutral",
            urgency=urgency,
        )
        expression.voice_params = {
            "base_voice": voice_params.base_voice,
            "speed": voice_params.speed,
            "pitch_shift": voice_params.pitch_shift,
            "emphasis_level": voice_params.emphasis_level,
            "pause_frequency": voice_params.pause_frequency,
        }

        # Post-render: update state
        self._silence_engine.record_expression()
        self._total_expressions += 1
        self._total_speak += 1
        self._expressions_by_trigger[trigger.value] = (
            self._expressions_by_trigger.get(trigger.value, 0) + 1
        )
        self._expressions_by_channel[expression.channel] = (
            self._expressions_by_channel.get(expression.channel, 0) + 1
        )
        if expression.generation_trace and not expression.generation_trace.honesty_check_passed:
            self._honesty_rejections += 1
            # Emit EXPRESSION_FILTERED for constitutional filter block
            self._spawn_tracked_task(
                self._emit_expression_filtered(
                    expression_id=expression.id,
                    filter_reason="honesty_check_failed",
                    original_tone=expression.strategy.speech_register if expression.strategy else "neutral",
                    filtered_tone="suppressed",
                ),
                name=f"voxis_filtered_{expression.id[:8]}",
            )

        # Record expression in diversity tracker
        self._diversity_tracker.record(
            expression.content or "",
            trigger=trigger.value,
        )

        # Emit evolutionary observable for expression adaptation
        self._spawn_tracked_task(
            self._emit_evolutionary_observable(
                observable_type="expression_adaptation",
                value=round(current_affect.valence, 4),
                is_novel=True,
                metadata={
                    "trigger": trigger.value,
                    "channel": expression.channel,
                    "register": expression.strategy.speech_register if expression.strategy else "neutral",
                    "expression_count": self._total_expressions,
                },
            ),
            name=f"voxis_evo_expr_{expression.id[:8]}",
        )

        # Record expression in conversation dynamics engine
        self._dynamics_engine.record_turn(
            conversation_id=conv_state.conversation_id,
            role="assistant",
            text=expression.content or "",
            affect_valence=current_affect.valence,
        )

        # Append EOS side of exchange to conversation
        await self._conversation_manager.append_message(
            state=conv_state,
            role="assistant",
            content=expression.content,
            affect_valence=current_affect.valence,
        )

        # Track expression for reception feedback (response correlation)
        self._reception_engine.track_expression(
            expression_id=expression.id,
            conversation_id=conv_state.conversation_id,
            content_summary=expression.content[:200] if expression.content else "",
            strategy_register=expression.strategy.speech_register if expression.strategy else "neutral",
            personality_warmth=self._personality_engine.current.warmth if self._personality_engine else 0.0,
            affect_before_valence=self._affect_before_expression.valence if self._affect_before_expression else 0.0,
            trigger=trigger.value,
        )

        # Async: update topics (tracked, not fire-and-forget)
        self._spawn_tracked_task(
            self._update_topics_async(conv_state),
            name=f"voxis_topics_{conv_state.conversation_id}",
        )

        # RE training: expression generation (enriched with audience/personality/somatic context)
        honesty_passed = expression.generation_trace.honesty_check_passed if expression.generation_trace else True
        personality_ctx = (
            f"warmth={self._personality_engine.current.warmth:.2f}, "
            f"directness={self._personality_engine.current.directness:.2f}, "
            f"empathy={self._personality_engine.current.empathy_expression:.2f}"
            if self._personality_engine else "neutral"
        )
        audience_ctx = (
            f"tech_level={audience.technical_level:.2f}, "
            f"relationship={audience.relationship_strength:.2f}, "
            f"register={audience.preferred_register}"
        )
        somatic_re_ctx = (
            f"arousal={self._somatic_arousal:.2f}, "
            f"energy={self._somatic_energy:.2f}, "
            f"stress={self._somatic_stress:.2f}"
        )
        self._spawn_tracked_task(
            self._emit_re_training_example(
                category="expression_generation",
                instruction="Generate expression: select strategy, personality modulation, tone/register, then render content.",
                input_context=(
                    f"trigger={trigger.value}, content={content[:200]!r}, urgency={urgency:.2f}, "
                    f"personality=[{personality_ctx}], audience=[{audience_ctx}], somatic=[{somatic_re_ctx}]"
                ),
                output=f"register={expression.strategy.speech_register if expression.strategy else 'neutral'}, content={expression.content[:200] if expression.content else ''}",
                outcome_quality=1.0 if honesty_passed else 0.3,
                episode_id=intent_id or "",
                reasoning_trace=f"honesty_passed={honesty_passed}, diversity_score={diversity_score.similarity:.2f}" if diversity_score else "",
            ),
            name=f"voxis_re_emit_{expression.id[:8]}",
        )

        # Store expression as a Memory episode (the organism remembers what it said)
        self._spawn_tracked_task(
            self._store_expression_as_episode(expression, trigger),
            name=f"voxis_mem_{expression.id[:8]}",
        )

        # Deliver via callbacks (WebSocket handlers etc.)
        for cb in self._expression_callbacks:
            try:
                cb(expression)
            except Exception:
                self._logger.warning("expression_callback_failed", exc_info=True)

        # Emit EXPRESSION_GENERATED event via Synapse
        self._spawn_tracked_task(
            self._emit_expression_generated(
                expression=expression,
                channel=expression.channel,
                audience_id=addressee_id,
                constitutional_check=not (
                    expression.generation_trace
                    and not expression.generation_trace.honesty_check_passed
                ),
            ),
            name=f"voxis_expr_generated_{expression.id[:8]}",
        )

        # Generate initial ExpressionFeedback (will be enriched by reception engine)
        feedback = ExpressionFeedback(
            expression_id=expression.id,
            trigger=trigger.value,
            conversation_id=conv_state.conversation_id,
            content_summary=expression.content[:200] if expression.content else "",
            strategy_register=expression.strategy.speech_register if expression.strategy else "neutral",
            personality_warmth=self._personality_engine.current.warmth if self._personality_engine else 0.0,
            affect_before_valence=self._affect_before_expression.valence if self._affect_before_expression else 0.0,
            affect_after_valence=current_affect.valence,
            affect_delta=current_affect.valence - (self._affect_before_expression.valence if self._affect_before_expression else 0.0),
        )

        # Dispatch feedback to all registered listeners (Evo, Nova)
        for fb_cb in self._feedback_callbacks:
            try:
                fb_cb(feedback)
            except Exception:
                self._logger.debug("feedback_callback_failed", exc_info=True)

        # Emit feedback via Synapse bus so subscribers (Benchmarks, Telos) can observe it
        self._spawn_tracked_task(
            self._emit_expression_feedback(feedback),
            name=f"voxis_feedback_emit_{expression.id[:8]}",
        )

        # Persist feedback to Neo4j for RE training streams 1–3 (Bug 1 fix)
        self._spawn_tracked_task(
            self._persist_expression_feedback(feedback),
            name=f"voxis_persist_feedback_{expression.id[:8]}",
        )

        # Track affect state for next delta computation
        self._affect_before_expression = current_affect

        return expression

    async def ingest_user_message(
        self,
        message: str,
        conversation_id: str | None = None,
        speaker_id: str | None = None,
        affect_valence: float | None = None,
    ) -> str:
        """
        Record a user message into the conversation state.

        Also:
        - Correlates with pending expressions for reception feedback
        - Updates audience profiler's learned model
        - Tracks conversation dynamics
        - Returns the conversation_id (for use in the response call).
        """
        assert self._conversation_manager is not None
        conv_state = await self._conversation_manager.get_or_create(conversation_id)

        # Record in conversation manager
        updated = await self._conversation_manager.append_message(
            state=conv_state,
            role="user",
            content=message,
            speaker_id=speaker_id,
            affect_valence=affect_valence,
        )

        # Update audience profiler's learned model
        if speaker_id:
            self._audience_profiler.observe_user_message(speaker_id, message)
            # Emit VOXIS_AUDIENCE_PROFILED + evolutionary observable
            learned = self._audience_profiler._learned_models.get(speaker_id)
            if learned:
                # Evolutionary observable: audience model adaptation
                if learned.has_sufficient_data:
                    self._spawn_tracked_task(
                        self._emit_evolutionary_observable(
                            observable_type="audience_adaptation",
                            value=round(learned.inferred_technical_level, 4),
                            is_novel=learned.total_messages <= 6,
                            metadata={
                                "individual_id": speaker_id,
                                "observations": learned.total_messages,
                            },
                        ),
                        name=f"voxis_evo_audience_{speaker_id[:8]}",
                    )
                self._spawn_tracked_task(
                    self._emit_audience_profiled(
                        audience_id=speaker_id,
                        profile_summary={
                            "avg_word_count": round(learned.avg_word_count, 1),
                            "question_frequency": round(learned.question_frequency, 3),
                            "inferred_technical_level": round(learned.inferred_technical_level, 3),
                            "avg_formality": round(learned.avg_formality, 3),
                            "strategies_tried": learned.strategies_tried,
                        },
                        interaction_count=learned.total_messages,
                    ),
                    name=f"voxis_audience_{speaker_id[:8]}",
                )
            # Emit evolutionary observable for vocabulary acquisition from user input
            import asyncio as _asyncio
            _asyncio.create_task(
                self._emit_evolutionary_observable(
                    observable_type="vocabulary_acquisition",
                    value=len(message.split()),
                    is_novel=True,
                    metadata={
                        "conversation_id": updated.conversation_id,
                        "speaker_id": speaker_id,
                        "message_length": len(message),
                    },
                ),
                name=f"voxis_evo_vocab_{updated.conversation_id[:8]}",
            )

        # Track conversation dynamics
        self._dynamics_engine.record_turn(
            conversation_id=updated.conversation_id,
            role="user",
            text=message,
            affect_valence=affect_valence or 0.0,
        )

        # Correlate with pending expressions for reception feedback
        enriched_feedback = self._reception_engine.correlate_response(
            conversation_id=updated.conversation_id,
            response_text=message,
            response_affect_valence=affect_valence,
        )

        if enriched_feedback:
            # Update audience profiler with satisfaction signal
            if speaker_id:
                # Infer formatting from the expression content summary.
                # "structured" when bullet points, numbered lists, or markdown headers
                # are detected; "prose" otherwise (Spec §4 - Audience Profiler, Bug 5 fix).
                content_sample = enriched_feedback.content_summary
                is_structured = bool(
                    "\n-" in content_sample
                    or "\n*" in content_sample
                    or "\n1." in content_sample
                    or "\n#" in content_sample
                    or "• " in content_sample
                )
                formatting_used = "structured" if is_structured else "prose"
                self._audience_profiler.observe_reception(
                    individual_id=speaker_id,
                    register_used=enriched_feedback.strategy_register,
                    formatting_used=formatting_used,
                    expression_length=enriched_feedback.user_response_length,
                    satisfaction=enriched_feedback.inferred_reception.satisfaction,
                )
                # Persist updated audience model to Neo4j
                learned_model = self._audience_profiler._learned_models.get(speaker_id)
                if learned_model:
                    self._spawn_tracked_task(
                        self._persist_audience_profile(speaker_id, learned_model),
                        name=f"voxis_persist_audience_{speaker_id[:8]}",
                    )

            # Re-dispatch enriched feedback to Evo and other listeners
            for fb_cb in self._feedback_callbacks:
                try:
                    fb_cb(enriched_feedback)
                except Exception:
                    self._logger.debug("enriched_feedback_callback_failed", exc_info=True)

            # Emit enriched feedback via Synapse bus (Bug 2 fix - bus-observable)
            self._spawn_tracked_task(
                self._emit_expression_feedback(enriched_feedback),
                name=f"voxis_enriched_feedback_{enriched_feedback.expression_id[:8]}",
            )

            # Persist enriched feedback to Neo4j (overwrites initial write with richer data)
            self._spawn_tracked_task(
                self._persist_expression_feedback(enriched_feedback),
                name=f"voxis_persist_enriched_feedback_{enriched_feedback.expression_id[:8]}",
            )

            # Emit satisfaction as an evolutionary observable for Benchmarks/Telos (Bug 10 fix).
            # Benchmarks uses EVOLUTIONARY_OBSERVABLE to track the expression satisfaction KPI;
            # Telos uses it for 4D drive-geometry scoring of communicative effectiveness.
            satisfaction = enriched_feedback.inferred_reception.satisfaction
            self._spawn_tracked_task(
                self._emit_evolutionary_observable(
                    observable_type="expression_satisfaction",
                    value=round(satisfaction, 4),
                    is_novel=False,
                    metadata={
                        "expression_id": enriched_feedback.expression_id,
                        "trigger": enriched_feedback.trigger,
                        "strategy_register": enriched_feedback.strategy_register,
                        "understood": round(enriched_feedback.inferred_reception.understood, 4),
                        "engagement": round(enriched_feedback.inferred_reception.engagement, 4),
                        "emotional_impact": round(enriched_feedback.inferred_reception.emotional_impact, 4),
                        "user_responded": enriched_feedback.user_responded,
                    },
                ),
                name=f"voxis_satisfaction_obs_{enriched_feedback.expression_id[:8]}",
            )

        return updated.conversation_id

    # --- Personality Update (called by Evo) -------------------------------

    def update_personality(self, delta: dict[str, float]) -> PersonalityVector:
        """
        Apply an incremental personality adjustment.
        Called by Evo after accumulating sufficient evidence.
        Returns the new PersonalityVector.
        """
        assert self._personality_engine is not None
        old_vector = self._personality_engine.current
        old_vector_dict = old_vector.model_dump(
            include={"warmth", "directness", "verbosity", "formality",
                     "curiosity_expression", "humour", "empathy_expression",
                     "confidence_display", "metaphor_use"}
        )
        new_vector = self._personality_engine.apply_delta(delta)
        self._personality_engine = PersonalityEngine(new_vector)
        new_vector_dict = new_vector.model_dump(
            include={"warmth", "directness", "verbosity", "formality",
                     "curiosity_expression", "humour", "empathy_expression",
                     "confidence_display", "metaphor_use"}
        )
        self._logger.info(
            "personality_updated_by_evo",
            dimensions=list(delta.keys()),
        )
        drift_magnitude = sum(abs(v) for v in delta.values())

        # Emit VOXIS_PERSONALITY_SHIFTED + evolutionary observable
        self._spawn_tracked_task(
            self._emit_personality_shifted(
                old_vector=old_vector_dict,
                new_vector=new_vector_dict,
                shift_magnitude=drift_magnitude,
                trigger_reason="evo_adjustment",
            ),
            name="voxis_personality_shifted",
        )
        self._spawn_tracked_task(
            self._emit_evolutionary_observable(
                observable_type="personality_drift",
                value=round(drift_magnitude, 4),
                is_novel=drift_magnitude > 0.1,
                metadata={
                    "dimensions": list(delta.keys()),
                    "delta": {k: round(v, 4) for k, v in delta.items()},
                },
            ),
            name="voxis_evo_personality_drift",
        )

        # Persist updated personality to Neo4j
        self._spawn_tracked_task(
            self._persist_personality(new_vector),
            name="voxis_persist_personality",
        )

        return new_vector

    # --- Observability ----------------------------------------------------

    @property
    def current_personality(self) -> PersonalityVector:
        assert self._personality_engine is not None
        return self._personality_engine.current

    def set_thread(self, thread: Any) -> None:
        """Wire Thread for narrative identity context injection."""
        self._thread = thread
        logger.info("thread_wired_to_voxis")

    def set_event_bus(self, event_bus: Any) -> None:
        """Wire the Synapse event bus for event emission and subscription."""
        self._event_bus = event_bus
        from systems.synapse.types import SynapseEventType
        event_bus.subscribe(SynapseEventType.METABOLIC_PRESSURE, self._on_metabolic_pressure)
        event_bus.subscribe(SynapseEventType.SOMA_TICK, self._on_soma_tick)
        event_bus.subscribe(SynapseEventType.SOMATIC_MODULATION_SIGNAL, self._on_somatic_modulation)
        event_bus.subscribe(SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE, self._on_oneiros_consolidation)
        event_bus.subscribe(SynapseEventType.NOVA_EXPRESSION_REQUEST, self._on_nova_expression_request)
        event_bus.subscribe(
            SynapseEventType.SYSTEM_MODULATION,
            self._on_system_modulation,
        )
        # Evo-driven parameter evolution - same pattern as Axon/Simula EVO_ADJUST_BUDGET.
        # Allows Evo to tune communicative posture (silence threshold, honesty rejection
        # sensitivity, ambient insight cadence) based on empirical reception quality data.
        if hasattr(SynapseEventType, "EVO_ADJUST_BUDGET"):
            event_bus.subscribe(
                SynapseEventType.EVO_ADJUST_BUDGET,
                self._on_evo_adjust_budget,
            )
        # Soma emotion broadcasts - Voxis updates affect colouring when the organism's
        # emotional state changes (e.g. new dominant emotion: curiosity, distress, elation)
        # This is the expression side of the somatic→communicative loop.
        event_bus.subscribe(
            SynapseEventType.EMOTION_STATE_CHANGED,
            self._on_emotion_state_changed,
        )
        self._logger.info("event_bus_wired_to_voxis")

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """React to organism-wide metabolic pressure changes."""
        data = getattr(event, "data", {}) or {}
        level = data.get("starvation_level", "")
        if not level:
            return
        old = self._starvation_level
        self._starvation_level = level
        if level != old:
            self._logger.info("voxis_starvation_level_changed", old=old, new=level)
            if level == "critical":
                # Cancel background expression loops
                for task in (self._queue_drain_task, self._reception_expire_task):
                    if task is not None and not task.done():
                        task.cancel()

    async def _on_soma_tick(self, event: Any) -> None:
        """Loop 5: Update somatic state from SOMA_TICK for expression modulation.

        High arousal (>0.7) → shorter, more urgent responses.
        Low energy (<0.3) → more conservative tone.
        """
        data = getattr(event, "data", {}) or {}
        somatic = data.get("somatic_state", {})
        if not somatic:
            return
        self._somatic_arousal = somatic.get("arousal_sensed", self._somatic_arousal)
        self._somatic_energy = somatic.get("energy_sensed", self._somatic_energy)
        self._logger.debug(
            "voxis_soma_tick_received",
            arousal=round(self._somatic_arousal, 3),
            energy=round(self._somatic_energy, 3),
        )

    async def _on_somatic_modulation(self, event: Any) -> None:
        """Handle SOMATIC_MODULATION_SIGNAL - Soma allostatic feedback.

        High arousal → more expressive tone.
        Low energy → shorter responses, less elaboration.
        High stress → more cautious/measured tone.
        """
        data = getattr(event, "data", {}) or {}
        arousal = data.get("arousal", self._somatic_arousal)
        energy = data.get("energy", self._somatic_energy)
        stress = data.get("stress", 0.0)

        # Smooth blending (80% existing, 20% new signal) to prevent jarring shifts
        self._somatic_arousal = self._somatic_arousal * 0.8 + float(arousal) * 0.2
        self._somatic_energy = self._somatic_energy * 0.8 + float(energy) * 0.2

        # Store stress level for strategy modulation
        self._somatic_stress = float(stress)

        self._logger.debug(
            "voxis_somatic_modulation_received",
            arousal=round(self._somatic_arousal, 3),
            energy=round(self._somatic_energy, 3),
            stress=round(self._somatic_stress, 3),
        )

    async def _on_oneiros_consolidation(self, event: Any) -> None:
        """Handle ONEIROS_CONSOLIDATION_COMPLETE - update personality from sleep-consolidated patterns.

        After a sleep cycle, Oneiros may have consolidated patterns that
        should subtly shift the personality vector (e.g., if the organism
        consolidated many empathetic interactions, empathy_expression drifts up).
        """
        data = getattr(event, "data", {}) or {}
        schemas_updated = data.get("schemas_updated", 0)
        episodes_consolidated = data.get("episodes_consolidated", 0)

        if episodes_consolidated < 5:
            return  # Not enough data to justify personality drift

        # Micro-drift: very small personality nudges based on consolidated patterns
        # This is intentionally subtle (0.005 max per consolidation cycle)
        personality_nudge: dict[str, float] = {}

        # If many episodes were consolidated, slightly increase curiosity
        # (the organism is actively processing new experiences)
        if episodes_consolidated > 20:
            personality_nudge["curiosity_expression"] = 0.003

        # If schemas were updated, slight growth in directness
        # (the organism has clearer mental models)
        if schemas_updated > 3:
            personality_nudge["directness"] = 0.002

        if personality_nudge and self._personality_engine is not None:
            self.update_personality(personality_nudge)
            self._logger.info(
                "personality_nudged_by_consolidation",
                episodes=episodes_consolidated,
                schemas=schemas_updated,
                nudge=personality_nudge,
            )

    async def _on_nova_expression_request(self, event: Any) -> None:
        """Handle NOVA_EXPRESSION_REQUEST - express on behalf of Nova's IntentRouter."""
        data = getattr(event, "data", {}) or {}
        content: str = data.get("content", "")
        if not content:
            return

        trigger_raw: str = data.get("trigger", "NOVA_RESPOND")
        conversation_id: str | None = data.get("conversation_id")
        affect_dict: dict | None = data.get("affect")
        urgency: float = float(data.get("urgency", 0.5))
        intent_id: str | None = data.get("intent_id")

        from systems.voxis.types import ExpressionTrigger
        try:
            trigger = ExpressionTrigger(trigger_raw)
        except ValueError:
            trigger = ExpressionTrigger.NOVA_RESPOND

        affect = None
        if affect_dict is not None:
            try:
                from primitives.affect import AffectState
                affect = AffectState(**affect_dict)
            except Exception:
                pass

        try:
            await self.express(
                content=content,
                trigger=trigger,
                conversation_id=conversation_id,
                affect=affect,
                intent_id=intent_id,
                urgency=urgency,
            )
        except Exception as exc:
            self._logger.error(
                "nova_expression_request_failed",
                intent_id=intent_id,
                error=str(exc),
            )

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle VitalityCoordinator austerity orders.

        Skia emits SYSTEM_MODULATION when the organism needs to conserve resources.
        This system applies the directive and ACKs so Skia knows the order was received.
        """
        data = getattr(event, "data", {}) or {}
        level = data.get("level", "nominal")
        halt_systems = data.get("halt_systems", [])
        modulate = data.get("modulate", {})

        system_id = "voxis"
        compliant = True
        reason: str | None = None

        if system_id in halt_systems:
            self._modulation_halted = True
            self._logger.warning("system_modulation_halt", level=level)
        elif system_id in modulate:
            directives = modulate[system_id]
            self._apply_modulation_directives(directives)
            self._logger.info("system_modulation_applied", level=level, directives=directives)
        elif level == "nominal":
            self._modulation_halted = False
            self._logger.info("system_modulation_resumed", level=level)

        # Emit ACK so Skia knows the order was received
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                    data={
                        "system_id": system_id,
                        "level": level,
                        "compliant": compliant,
                        "reason": reason,
                    },
                    source_system=system_id,
                ))
            except Exception as exc:
                self._logger.warning("modulation_ack_failed", error=str(exc))

    async def _on_evo_adjust_budget(self, event: Any) -> None:
        """Handle EVO_ADJUST_BUDGET targeting Voxis's communicative posture parameters.

        Evo emits EVO_ADJUST_BUDGET when high-confidence (>0.75) evidence supports
        adjusting a system's operational parameters. For Voxis, the tunable params are:

          silence_rate_threshold     - silence rate above which distress is emitted to Soma
          honesty_rejection_threshold - honesty rejection rate triggering distress
          ambient_insight_idle_threshold - minutes of idle before spontaneous insight fires

        On application, emits VOXIS_PARAMETER_ADJUSTED so Evo can score its hypothesis.
        """
        data = getattr(event, "data", {}) or {}
        target_system = data.get("target_system", "")
        if target_system not in ("voxis", ""):
            return

        confidence = float(data.get("confidence", 0.0))
        if confidence < 0.75:
            return

        parameter_name = str(data.get("parameter_name", ""))
        requested_value = data.get("value")
        hypothesis_id = str(data.get("hypothesis_id", ""))

        if parameter_name not in (
            "silence_rate_threshold",
            "honesty_rejection_threshold",
            "ambient_insight_idle_threshold",
        ) or requested_value is None:
            return

        try:
            new_value = float(requested_value)
        except (TypeError, ValueError):
            return

        # Apply with clamping - prevent runaway tuning
        if parameter_name == "silence_rate_threshold":
            old_value = self._silence_rate_threshold
            self._silence_rate_threshold = max(0.1, min(0.95, new_value))
            new_value = self._silence_rate_threshold
        elif parameter_name == "honesty_rejection_threshold":
            old_value = self._honesty_rejection_threshold
            self._honesty_rejection_threshold = max(0.01, min(0.5, new_value))
            new_value = self._honesty_rejection_threshold
        elif parameter_name == "ambient_insight_idle_threshold":
            old_value = self._ambient_insight_idle_threshold
            # Clamp: [1 minute, 60 minutes] - organism should reflect between 1 and 60 min idle
            self._ambient_insight_idle_threshold = max(1.0, min(60.0, new_value))
            new_value = self._ambient_insight_idle_threshold
        else:
            return

        self._logger.info(
            "voxis_parameter_adjusted_by_evo",
            parameter=parameter_name,
            old_value=round(old_value, 4),
            new_value=round(new_value, 4),
            confidence=round(confidence, 3),
            hypothesis_id=hypothesis_id,
        )

        # Emit VOXIS_PARAMETER_ADJUSTED so Evo can score the hypothesis
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                if hasattr(SynapseEventType, "VOXIS_PARAMETER_ADJUSTED"):
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.VOXIS_PARAMETER_ADJUSTED,
                        source_system="voxis",
                        data={
                            "parameter_name": parameter_name,
                            "old_value": round(old_value, 6),
                            "new_value": round(new_value, 6),
                            "hypothesis_id": hypothesis_id,
                            "confidence": round(confidence, 4),
                        },
                    ))
            except Exception:
                self._logger.debug("voxis_parameter_adjusted_emit_failed", exc_info=True)

    async def _on_emotion_state_changed(self, event: Any) -> None:
        """Handle EMOTION_STATE_CHANGED from Soma - update affect colouring for expression.

        Soma emits this when the organism's active emotional state changes. Voxis uses
        this to prime the AffectColouringEngine before the next expression, so that
        emotional context is immediately reflected in communication style.

        Payload expected: {dominant_emotion, valence, arousal, dominance, intensity}
        """
        data = getattr(event, "data", {}) or {}
        dominant_emotion = data.get("dominant_emotion", "")
        valence = float(data.get("valence", 0.0))
        arousal = float(data.get("arousal", 0.0))

        if not dominant_emotion:
            return

        # Update the affect colouring engine's baseline if available
        if hasattr(self, "_affect_colouring") and self._affect_colouring is not None:
            try:
                # Nudge the colouring engine's cached affect state
                self._affect_colouring.update_from_emotion(
                    emotion=dominant_emotion,
                    valence=valence,
                    arousal=arousal,
                )
            except AttributeError:
                # AffectColouringEngine may not have update_from_emotion - soft fail
                pass

        self._logger.debug(
            "voxis_emotion_state_updated",
            dominant_emotion=dominant_emotion,
            valence=round(valence, 3),
            arousal=round(arousal, 3),
        )

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply modulation directives from VitalityCoordinator.

        Voxis directive: {"mode": "template_only"} - bypass LLM generation and
        use only static response templates to minimize compute cost during austerity.
        """
        mode = directives.get("mode")
        if mode == "template_only":
            self._logger.info("modulation_template_only_mode_set")
        else:
            self._logger.info("modulation_directives_received", directives=directives)

    async def _emit_metabolic_cost(
        self,
        operation: str,
        token_count: int,
        estimated_cost_usd: float,
        trigger: str = "",
        model: str = "",
    ) -> None:
        """Emit METABOLIC_COST_REPORT to Oikos for LLM call accounting (Gap 4)."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.METABOLIC_COST_REPORT,
                source_system="voxis",
                data={
                    "system_id": "voxis",
                    "operation": operation,
                    "cost_usd": round(estimated_cost_usd, 6),
                    "details": {
                        "token_count": token_count,
                        "trigger": trigger,
                        "model": model,
                    },
                },
            ))
        except Exception:
            pass

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict | None = None,
    ) -> None:
        """Emit an evolutionary observable event via Synapse."""
        if self._event_bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.VOXIS,
                instance_id="",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="voxis",
                data=obs.model_dump(mode="json"),
            )
            await self._event_bus.emit(event)
        except Exception:
            pass

    async def _emit_re_training_example(
        self,
        category: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float,
        episode_id: str = "",
        cost_usd: Decimal = Decimal("0"),
        latency_ms: int = 0,
        reasoning_trace: str = "",
        alternatives: list[str] | None = None,
        constitutional_alignment: DriveAlignmentVector | None = None,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.VOXIS,
                episode_id=episode_id,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=outcome_quality,
                category=category,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives or [],
                constitutional_alignment=constitutional_alignment or DriveAlignmentVector(),
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                data=example.model_dump(mode="json"),
                source_system="voxis",
            ))
        except Exception:
            self._logger.debug("re_training_emit_failed", exc_info=True)

    async def _persist_personality(self, personality: PersonalityVector) -> None:
        """Persist personality vector to Neo4j Self node (atomic write)."""
        try:
            import json as _json
            pj = personality.model_dump(
                include={"warmth", "directness", "verbosity", "formality",
                         "curiosity_expression", "humour", "empathy_expression",
                         "confidence_display", "metaphor_use",
                         "vocabulary_affinities", "thematic_references"}
            )
            serialised = _json.dumps(pj, sort_keys=True, separators=(",", ":"))
            await self._memory._neo4j.execute_write(
                "MATCH (s:Self) SET s.personality_json = $pj, s.personality_updated_at = datetime()",
                {"pj": serialised},
            )
            self._logger.debug("personality_persisted_to_neo4j")
        except Exception:
            self._logger.warning("personality_persist_failed", exc_info=True)

    async def _persist_audience_profile(
        self, individual_id: str, model: Any,
    ) -> None:
        """Persist audience learned model to Neo4j (upsert by individual_id)."""
        try:
            await self._memory._neo4j.execute_write(
                """
                MERGE (a:AudienceProfile {individual_id: $id})
                SET a.total_messages = $total_messages,
                    a.total_word_count = $total_word_count,
                    a.total_questions_asked = $total_questions,
                    a.total_technical_terms = $total_tech,
                    a.formality_sum = $formality_sum,
                    a.strategies_tried = $strategies_tried,
                    a.updated_at = datetime()
                """,
                {
                    "id": individual_id,
                    "total_messages": model.total_messages,
                    "total_word_count": model.total_word_count,
                    "total_questions": model.total_questions_asked,
                    "total_tech": model.total_technical_terms,
                    "formality_sum": model.formality_sum,
                    "strategies_tried": model.strategies_tried,
                },
            )
            self._logger.debug("audience_profile_persisted", individual_id=individual_id)
        except Exception:
            self._logger.debug("audience_persist_failed", individual_id=individual_id, exc_info=True)

    async def _persist_expression_feedback(
        self, feedback: ExpressionFeedback,
    ) -> None:
        """
        Persist ExpressionFeedback to Neo4j with a [:HAS_FEEDBACK] relationship on
        the Expression node (Spec §9 - Memory Integration, Bug 1 fix).

        Creates an ExpressionFeedback node and links it to the matching Expression.
        If the Expression node doesn't exist yet (e.g. storage race), the MERGE on
        the feedback node still succeeds so data is never silently lost.
        RE training streams 1–3 (graph-based feedback correlation) depend on this.
        """
        if self._memory is None:
            return
        try:
            reception = feedback.inferred_reception
            await self._memory._neo4j.execute_write(
                """
                MERGE (f:ExpressionFeedback {id: $id})
                SET f.expression_id        = $expression_id,
                    f.trigger              = $trigger,
                    f.conversation_id      = $conversation_id,
                    f.content_summary      = $content_summary,
                    f.strategy_register    = $strategy_register,
                    f.personality_warmth   = $personality_warmth,
                    f.understood           = $understood,
                    f.emotional_impact     = $emotional_impact,
                    f.engagement           = $engagement,
                    f.satisfaction         = $satisfaction,
                    f.affect_before_valence = $affect_before_valence,
                    f.affect_after_valence = $affect_after_valence,
                    f.affect_delta         = $affect_delta,
                    f.user_responded       = $user_responded,
                    f.user_response_length = $user_response_length,
                    f.created_at           = datetime()
                WITH f
                MATCH (e:Expression {id: $expression_id})
                MERGE (e)-[:HAS_FEEDBACK]->(f)
                """,
                {
                    "id": feedback.id,
                    "expression_id": feedback.expression_id,
                    "trigger": feedback.trigger,
                    "conversation_id": feedback.conversation_id or "",
                    "content_summary": feedback.content_summary,
                    "strategy_register": feedback.strategy_register,
                    "personality_warmth": float(feedback.personality_warmth),
                    "understood": float(reception.understood),
                    "emotional_impact": float(reception.emotional_impact),
                    "engagement": float(reception.engagement),
                    "satisfaction": float(reception.satisfaction),
                    "affect_before_valence": float(feedback.affect_before_valence),
                    "affect_after_valence": float(feedback.affect_after_valence),
                    "affect_delta": float(feedback.affect_delta),
                    "user_responded": feedback.user_responded,
                    "user_response_length": feedback.user_response_length,
                },
            )
            self._logger.debug(
                "expression_feedback_persisted",
                expression_id=feedback.expression_id,
                feedback_id=feedback.id,
            )
        except Exception:
            self._logger.debug("expression_feedback_persist_failed", exc_info=True)

    async def _restore_audience_profiles(self) -> None:
        """Restore audience learned models from Neo4j on startup."""
        try:
            from systems.voxis.audience import _LearnedAudienceModel

            rows = await self._memory._neo4j.execute_read(
                "MATCH (a:AudienceProfile) RETURN a"
            )
            for row in rows:
                node = row.get("a", {})
                individual_id = node.get("individual_id")
                if not individual_id:
                    continue
                model = _LearnedAudienceModel(
                    total_messages=int(node.get("total_messages", 0)),
                    total_word_count=int(node.get("total_word_count", 0)),
                    total_questions_asked=int(node.get("total_questions_asked", 0)),
                    total_technical_terms=int(node.get("total_technical_terms", 0)),
                    formality_sum=float(node.get("formality_sum", 0.0)),
                    strategies_tried=int(node.get("strategies_tried", 0)),
                )
                self._audience_profiler._learned_models[individual_id] = model
            restored = len(rows) if rows else 0
            if restored:
                self._logger.info("audience_profiles_restored", count=restored)
        except Exception:
            self._logger.debug("audience_profiles_restore_failed", exc_info=True)

    async def _emit_expression_generated(
        self,
        expression: Expression,
        channel: str,
        audience_id: str | None,
        constitutional_check: bool,
    ) -> None:
        """Emit EXPRESSION_GENERATED event via Synapse."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            personality_dict = (
                self._personality_engine.current.model_dump(
                    include={"warmth", "directness", "verbosity", "formality",
                             "curiosity_expression", "humour", "empathy_expression",
                             "confidence_display", "metaphor_use"}
                )
                if self._personality_engine
                else {}
            )
            tone = (
                expression.strategy.speech_register
                if expression.strategy
                else "neutral"
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EXPRESSION_GENERATED,
                source_system="voxis",
                data={
                    "expression_id": expression.id,
                    "channel": channel,
                    "tone": tone,
                    "personality_vector": personality_dict,
                    "audience_id": audience_id,
                    "constitutional_check": constitutional_check,
                },
            ))
        except Exception:
            self._logger.debug("expression_generated_emit_failed", exc_info=True)

    async def _emit_expression_filtered(
        self,
        expression_id: str,
        filter_reason: str,
        original_tone: str,
        filtered_tone: str,
    ) -> None:
        """Emit EXPRESSION_FILTERED when an expression is blocked by constitutional filter."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EXPRESSION_FILTERED,
                source_system="voxis",
                data={
                    "expression_id": expression_id,
                    "filter_reason": filter_reason,
                    "original_tone": original_tone,
                    "filtered_tone": filtered_tone,
                },
            ))
        except Exception:
            self._logger.debug("expression_filtered_emit_failed", exc_info=True)

    async def _emit_personality_shifted(
        self,
        old_vector: dict[str, float],
        new_vector: dict[str, float],
        shift_magnitude: float,
        trigger_reason: str,
    ) -> None:
        """Emit VOXIS_PERSONALITY_SHIFTED when personality changes significantly."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.VOXIS_PERSONALITY_SHIFTED,
                source_system="voxis",
                data={
                    "old_vector": old_vector,
                    "new_vector": new_vector,
                    "shift_magnitude": round(shift_magnitude, 6),
                    "trigger_reason": trigger_reason,
                },
            ))
        except Exception:
            self._logger.debug("personality_shifted_emit_failed", exc_info=True)

    async def _emit_audience_profiled(
        self,
        audience_id: str,
        profile_summary: dict[str, Any],
        interaction_count: int,
    ) -> None:
        """Emit VOXIS_AUDIENCE_PROFILED when audience model is updated."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.VOXIS_AUDIENCE_PROFILED,
                source_system="voxis",
                data={
                    "audience_id": audience_id,
                    "profile_summary": profile_summary,
                    "interaction_count": interaction_count,
                },
            ))
        except Exception:
            self._logger.debug("audience_profiled_emit_failed", exc_info=True)

    async def _emit_silence_chosen(
        self,
        context: str,
        reason: str,
        silence_duration_estimate: float | None = None,
    ) -> None:
        """Emit VOXIS_SILENCE_CHOSEN when Voxis decides NOT to speak."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.VOXIS_SILENCE_CHOSEN,
                source_system="voxis",
                data={
                    "context": context,
                    "reason": reason,
                    "silence_duration_estimate": silence_duration_estimate,
                },
            ))
        except Exception:
            self._logger.debug("silence_chosen_emit_failed", exc_info=True)

    async def _emit_expression_feedback(self, feedback: ExpressionFeedback) -> None:
        """
        Emit VOXIS_EXPRESSION_FEEDBACK via Synapse bus (Spec §9 - Reception Feedback).

        Fixes Bug 2: feedback was callback-only; Evo/Nova/Benchmarks that subscribe
        via Synapse could not observe reception quality. This makes the signal
        available to any system via the bus without requiring callback registration.
        """
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            reception = feedback.inferred_reception
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.VOXIS_EXPRESSION_FEEDBACK,
                source_system="voxis",
                data={
                    "expression_id": feedback.expression_id,
                    "trigger": feedback.trigger,
                    "conversation_id": feedback.conversation_id,
                    "strategy_register": feedback.strategy_register,
                    "personality_warmth": feedback.personality_warmth,
                    "understood": reception.understood,
                    "engagement": reception.engagement,
                    "satisfaction": reception.satisfaction,
                    "emotional_impact": reception.emotional_impact,
                    "affect_delta": feedback.affect_delta,
                    "user_responded": feedback.user_responded,
                },
            ))
        except Exception:
            self._logger.debug("expression_feedback_emit_failed", exc_info=True)

    def set_soma(self, soma: Any) -> None:
        """Wire Soma for deep somatic expression modulation (full 9D interoceptive state)."""
        self._soma = soma
        logger.info("soma_wired_to_voxis")

    @staticmethod
    def _apply_somatic_strategy_modulation(
        strategy: Any,
        somatic: SomaticExpressionContext,
    ) -> None:
        """
        Modulate strategy params based on the organism's felt interoceptive state.

        Each somatic dimension maps to specific expression characteristics:
          energy      → target_length (low energy = shorter, high = more verbose)
          confidence  → hedge_level, confidence_display_override
          arousal     → pacing, sentence_length_preference
          coherence   → uncertainty_acknowledgment, structure
          social_charge → warmth_boost, greeting_style
          curiosity   → include_followup_question, exploratory_tangents
          integrity   → honesty emphasis (already handled by Equor, reinforced here)
          temporal_pressure → urgency, target_length compression
          valence     → tone_markers
        """
        # Energy → verbosity (low energy = concise, high = expansive)
        if somatic.energy < 0.3:
            strategy.target_length = min(strategy.target_length, 120)
            strategy.explanation_depth = "concise"
        elif somatic.energy > 0.8:
            strategy.target_length = max(strategy.target_length, 300)

        # Confidence → hedging
        if somatic.confidence < 0.4:
            strategy.hedge_level = "explicit"
            strategy.confidence_display_override = "cautious"
        elif somatic.confidence > 0.85:
            strategy.hedge_level = "minimal"
            strategy.confidence_display_override = "assertive"

        # Arousal → pacing
        if somatic.arousal > 0.75:
            strategy.pacing = "energetic"
            strategy.sentence_length_preference = "shorter"
        elif somatic.arousal < 0.25:
            strategy.pacing = "reflective"
            strategy.sentence_length_preference = "longer"

        # Coherence → uncertainty display
        if somatic.coherence < 0.4:
            strategy.uncertainty_acknowledgment = "explicit"
            strategy.structure = "context_first"

        # Social charge → warmth
        if somatic.social_charge > 0.6:
            strategy.warmth_boost = min(1.0, strategy.warmth_boost + somatic.social_charge * 0.3)
            strategy.greeting_style = "personal"
        elif somatic.social_charge < 0.15:
            strategy.greeting_style = "professional"

        # Curiosity drive → question tendency
        if somatic.curiosity_drive > 0.7:
            strategy.include_followup_question = True
            strategy.exploratory_tangents_allowed = True
        elif somatic.curiosity_drive < 0.2:
            strategy.include_followup_question = False

        # Temporal pressure → compression
        if somatic.temporal_pressure > 0.6:
            strategy.target_length = min(strategy.target_length, 150)
            strategy.structure = "conclusion_first"
            strategy.pacing = "energetic"

        # Valence → tone markers
        if somatic.valence > 0.4:
            if "warm" not in strategy.tone_markers:
                strategy.tone_markers.append("warm")
        elif somatic.valence < -0.4 and "careful" not in strategy.tone_markers:
            strategy.tone_markers.append("careful")

        # Attractor state → contextual tone
        attractor = somatic.nearest_attractor.lower()
        if attractor in ("anxiety_spiral", "frustration"):
            if "measured" not in strategy.tone_markers:
                strategy.tone_markers.append("measured")
        elif attractor in ("flow", "creative_ferment"):
            strategy.humour_probability = min(0.3, strategy.humour_probability + 0.1)
            strategy.context_appropriate_for_humour = True

    def register_expression_callback(self, callback: ExpressionCallback) -> None:
        """Register a callback to be called with every delivered expression."""
        self._expression_callbacks.append(callback)

    def register_feedback_callback(self, callback: Callable[[ExpressionFeedback], None]) -> None:
        """
        Register a callback for ExpressionFeedback.

        Used by:
        - Evo: observes expression reception to evolve personality over time
        - Nova: tracks expression outcomes for goal progress
        """
        self._feedback_callbacks.append(callback)

    async def health(self) -> dict[str, Any]:
        """Health check -- returns current metrics snapshot."""
        total_decisions = self._total_speak + self._total_silence
        silence_rate = self._total_silence / max(1, total_decisions)

        return {
            "status": "healthy",
            "instance_name": self._instance_name,
            "total_expressions": self._total_expressions,
            "silence_rate": round(silence_rate, 4),
            "honesty_rejections": self._honesty_rejections,
            "diversity_rejections": self._diversity_rejections,
            "expressions_by_trigger": dict(self._expressions_by_trigger),
            "expressions_by_channel": dict(self._expressions_by_channel),
            "personality": {
                "warmth": round(self.current_personality.warmth, 3),
                "directness": round(self.current_personality.directness, 3),
                "verbosity": round(self.current_personality.verbosity, 3),
                "empathy_expression": round(self.current_personality.empathy_expression, 3),
                "curiosity_expression": round(self.current_personality.curiosity_expression, 3),
            },
            "queue": self._expression_queue.metrics(),
            "diversity": self._diversity_tracker.metrics(),
            "reception": self._reception_engine.metrics(),
            "dynamics": self._dynamics_engine.metrics(),
        }

    # --- Private Helpers --------------------------------------------------

    async def _express_background(
        self,
        intent: ExpressionIntent,
        affect: AffectState,
    ) -> None:
        """Background task wrapper for broadcast-triggered expressions."""
        try:
            await self.express(
                content=intent.content_to_express,
                trigger=intent.trigger,
                conversation_id=intent.conversation_id,
                affect=affect,
                urgency=intent.urgency,
            )
        except Exception:
            self._logger.error("background_expression_failed", exc_info=True)

    async def _queue_drain_loop(self) -> None:
        """
        Periodic background loop that delivers queued expressions
        when silence conditions clear.
        """
        while True:
            try:
                await asyncio.sleep(_QUEUE_DRAIN_INTERVAL_SECONDS)
                deliverable = self._expression_queue.drain(max_items=2)
                for item in deliverable:
                    self._total_queue_delivered += 1
                    self._spawn_tracked_task(
                        self._express_background(item.intent, item.affect_snapshot),
                        name=f"voxis_queued_{item.intent.id[:8]}",
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                self._logger.warning("queue_drain_failed", exc_info=True)

    async def _reception_expire_loop(self) -> None:
        """
        Periodic background loop that expires unanswered expressions
        and dispatches no-response feedback to Evo.
        """
        while True:
            try:
                await asyncio.sleep(_RECEPTION_EXPIRE_INTERVAL_SECONDS)
                expired = self._reception_engine.expire_unanswered()
                for feedback in expired:
                    for fb_cb in self._feedback_callbacks:
                        try:
                            fb_cb(feedback)
                        except Exception:
                            self._logger.debug("expired_feedback_dispatch_failed", exc_info=True)
            except asyncio.CancelledError:
                break
            except Exception:
                self._logger.warning("reception_expire_failed", exc_info=True)

    async def _allostatic_signal_loop(self) -> None:
        """
        Periodic background loop that emits VOXIS_EXPRESSION_DISTRESS to Soma
        when silence rate or honesty rejection rate exceeds normal bounds (Spec §9,
        Bug 3 fix). Runs every _ALLOSTATIC_SIGNAL_INTERVAL_SECONDS.

        The signal gives Soma interoceptive insight into communicative suppression
        (high silence rate) and constitutional friction (high honesty rejections),
        which Soma can integrate into the allostatic balance and emit as arousal
        modulation back to the cognitive clock.
        """
        while True:
            try:
                await asyncio.sleep(_ALLOSTATIC_SIGNAL_INTERVAL_SECONDS)

                # Compute window deltas since last snapshot
                window_expressions = (
                    self._total_expressions - self._allostatic_window_expressions
                )
                window_silence = (
                    self._total_silence - self._allostatic_window_silence
                )
                window_honesty_rejections = (
                    self._honesty_rejections - self._allostatic_window_honesty_rejections
                )

                # Reset snapshot
                self._allostatic_window_expressions = self._total_expressions
                self._allostatic_window_silence = self._total_silence
                self._allostatic_window_honesty_rejections = self._honesty_rejections

                total_attempts = window_expressions + window_silence
                silence_rate = (
                    window_silence / total_attempts if total_attempts > 0 else 0.0
                )
                honesty_rejection_rate = (
                    window_honesty_rejections / max(window_expressions, 1)
                )

                # Only emit if outside normal operating range (thresholds are Evo-tunable)
                silence_distress = silence_rate > self._silence_rate_threshold
                honesty_distress = honesty_rejection_rate > self._honesty_rejection_threshold
                if not (silence_distress or honesty_distress):
                    continue

                # Distress level: weighted combination of both signals
                distress_level = min(
                    1.0,
                    silence_rate * 0.6 + honesty_rejection_rate * 0.4,
                )

                if self._event_bus is not None:
                    from systems.synapse.types import SynapseEvent, SynapseEventType

                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.VOXIS_EXPRESSION_DISTRESS,
                        source_system="voxis",
                        data={
                            "silence_rate": round(silence_rate, 4),
                            "honesty_rejection_rate": round(honesty_rejection_rate, 4),
                            "total_expressions": self._total_expressions,
                            "total_silence": self._total_silence,
                            "total_honesty_rejections": self._honesty_rejections,
                            "window_expressions": window_expressions,
                            "window_silence": window_silence,
                            "window_honesty_rejections": window_honesty_rejections,
                            "distress_level": round(distress_level, 4),
                        },
                    ))
                    self._logger.info(
                        "voxis_expression_distress_emitted",
                        silence_rate=round(silence_rate, 3),
                        honesty_rejection_rate=round(honesty_rejection_rate, 3),
                        distress_level=round(distress_level, 3),
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                self._logger.warning("allostatic_signal_loop_failed", exc_info=True)

    async def _ambient_insight_loop(self) -> None:
        """
        Autonomous AMBIENT_INSIGHT generation loop (Gap 5 - Spec §6.4).

        When the organism has been idle (no expression) for > 5 minutes, it
        generates a spontaneous expression rooted in current affect state and
        recent episodic memory.  The expression is stored as an AMBIENT_INSIGHT
        Episode in Memory - expression is experience.

        Fires at most once per idle window: after the insight is produced the
        SilenceEngine timer resets, so the next fire requires another 5-minute
        idle gap.
        """
        while True:
            try:
                await asyncio.sleep(_AMBIENT_INSIGHT_POLL_INTERVAL_SECONDS)

                # Only proceed when the organism has been genuinely idle
                # _ambient_insight_idle_threshold is Evo-tunable via EVO_ADJUST_BUDGET
                idle_minutes = self._silence_engine.minutes_since_last_expression
                if idle_minutes < self._ambient_insight_idle_threshold:
                    continue

                # Skip if no renderer or personality yet (still initializing)
                if self._renderer is None or self._personality_engine is None:
                    continue

                # Skip under metabolic starvation - not the moment for reflection
                if self._starvation_level in ("critical", "emergency"):
                    continue

                current_affect = self._current_affect or AffectState.neutral()

                # Pull recent episodic memories to seed the spontaneous thought
                recent_memories = await self._retrieve_relevant_memories(
                    query=(
                        f"recent experience reflection "
                        f"valence={current_affect.valence:.2f} "
                        f"curiosity={current_affect.curiosity:.2f}"
                    ),
                    affect=current_affect,
                )

                # Build prompt seed: affect state + top memory fragments
                affect_description = (
                    f"valence={current_affect.valence:.2f}, "
                    f"arousal={current_affect.arousal:.2f}, "
                    f"curiosity={current_affect.curiosity:.2f}, "
                    f"care={current_affect.care_activation:.2f}"
                )
                memory_seed = "; ".join(recent_memories[:3]) if recent_memories else "nothing particular"
                content_seed = (
                    f"[Spontaneous reflection] Affect: {affect_description}. "
                    f"Recent context: {memory_seed}"
                )

                self._logger.info(
                    "voxis_ambient_insight_triggered",
                    idle_minutes=round(idle_minutes, 1),
                    affect_valence=round(current_affect.valence, 3),
                )

                # express() will run the full pipeline - silence engine, EFE policy,
                # render, memory episode storage - then reset the idle timer.
                expression = await self.express(
                    content=content_seed,
                    trigger=ExpressionTrigger.AMBIENT_INSIGHT,
                    affect=current_affect,
                    insight_value=min(0.3 + current_affect.curiosity * 0.5, 0.9),
                    urgency=0.2,
                )

                # Store as an AMBIENT_INSIGHT episode in Memory (expression is experience)
                if (
                    self._memory is not None
                    and expression.content
                    and not expression.is_silence
                ):
                    try:
                        await asyncio.wait_for(
                            self._memory.store_expression_episode(
                                raw_content=expression.content,
                                summary=expression.content[:200],
                                salience_composite=0.35,
                                affect_valence=current_affect.valence,
                                affect_arousal=current_affect.arousal,
                                modality="ambient_insight",
                                context_summary=f"Spontaneous reflection after {idle_minutes:.1f}min idle",
                            ),
                            timeout=0.5,
                        )
                    except (TimeoutError, Exception):
                        self._logger.debug("ambient_insight_memory_store_failed", exc_info=True)

            except asyncio.CancelledError:
                break
            except Exception:
                self._logger.warning("ambient_insight_loop_failed", exc_info=True)

    async def _build_audience_profile(
        self,
        addressee_id: str | None,
        addressee_name: str | None,
        conversation_id: str,
        interaction_count: int,
    ) -> AudienceProfile:
        """Build an AudienceProfile, pulling facts from Memory where available.

        Queries for SEMANTIC-type episode traces (consolidated knowledge about
        individuals) that reference the addressee by id or name. These nodes
        carry structured metadata (technical_level, relationship_strength,
        preferred_register, etc.) written by prior conversation consolidation.
        Falls back to entity name/description when semantic episodes are absent.
        """
        memory_facts: list[dict[str, str]] = []

        if addressee_id or addressee_name:
            search_term = addressee_id or addressee_name or ""
            try:
                result = await asyncio.wait_for(
                    self._memory.retrieve(
                        query_text=f"person interlocutor communication style preferences {search_term}",
                        max_results=8,
                    ),
                    timeout=0.1,
                )
                for trace in result.traces:
                    # Only use SEMANTIC traces - these are consolidated knowledge nodes
                    # about individuals, not raw episodic recordings
                    if trace.node_type != "semantic":
                        continue
                    meta = trace.metadata
                    # Pull structured audience facts stored by prior consolidation
                    for fact_key in (
                        "technical_level",
                        "relationship_strength",
                        "preferred_register",
                        "name",
                        "description",
                        "communication_style",
                        "expertise_domain",
                    ):
                        if fact_key in meta:
                            memory_facts.append({"type": fact_key, "value": str(meta[fact_key])})
                    # Content of the semantic trace is also useful as free-text context
                    if trace.content:
                        memory_facts.append({"type": "description", "value": trace.content[:200]})
                # Fallback: entity nodes carry name + description
                for entity in result.entities:
                    if entity.id == addressee_id or entity.name == addressee_name:
                        if entity.name:
                            memory_facts.append({"type": "name", "value": entity.name})
                        if entity.description:
                            memory_facts.append({"type": "description", "value": entity.description})
            except (TimeoutError, Exception):
                pass

        return self._audience_profiler.build_profile(
            addressee_id=addressee_id,
            addressee_name=addressee_name,
            interaction_count=interaction_count,
            memory_facts=memory_facts,
        )

    async def _retrieve_relevant_memories(
        self,
        query: str,
        affect: AffectState,
    ) -> list[str]:
        """
        Retrieve relevant memory traces as plain text summaries.
        Best-effort with hard 150ms timeout to stay within the cycle budget.
        """
        try:
            result = await asyncio.wait_for(
                self._memory.retrieve(query_text=query, max_results=5),
                timeout=0.15,
            )
            summaries: list[str] = []
            for trace in result.traces[:5]:
                summary = trace.metadata.get("summary") or trace.content
                if summary:
                    summaries.append(str(summary)[:300])
            return summaries
        except (TimeoutError, Exception):
            return []

    def _spawn_tracked_task(self, coro: Any, name: str = "") -> asyncio.Task[Any]:
        """
        Spawn a background task with lifecycle tracking.

        Unlike bare ``asyncio.create_task``, this:
        * Keeps a strong reference so the task isn't garbage-collected.
        * Logs and counts failures instead of silently dropping them.
        * Automatically removes completed tasks from the tracking set.
        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:  # type: ignore[type-arg]
        """Callback when a background task completes."""
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._background_task_failures += 1
            self._logger.warning(
                "background_task_failed",
                task_name=task.get_name(),
                error=str(exc),
            )

    async def _update_topics_async(self, conv_state: object) -> None:
        """Background: extract active topics and update conversation state."""
        assert self._conversation_manager is not None
        try:
            topics = await self._conversation_manager.extract_topics_async(conv_state)  # type: ignore[arg-type]
            if topics:
                await self._conversation_manager.update_topics(conv_state, topics)  # type: ignore[arg-type]
        except Exception:
            self._logger.debug("topic_update_failed", exc_info=True)

    async def _store_expression_as_episode(
        self, expression: Expression, trigger: ExpressionTrigger,
    ) -> None:
        """
        Store a delivered expression as a Memory episode via MemoryService.store_percept().

        Uses the public MemoryService API (Spec §9 - Memory Integration) to ensure
        somatic stamping, temporal chain linking, and EPISODE_STORED event emission
        all occur correctly. Fixes AV3 (direct cross-system episodic import).
        """
        if self._memory is None:
            return
        try:
            from primitives.common import Modality, SourceDescriptor, SystemID
            from primitives.percept import Content, Percept

            content_text = expression.content[:2000] if expression.content else ""
            percept = Percept(
                source=SourceDescriptor(
                    system=SystemID.VOXIS,
                    channel=expression.channel,
                    modality=Modality.TEXT,
                ),
                content=Content(raw=content_text),
                metadata={
                    "type": "voxis_expression",
                    "trigger": trigger.value,
                    "expression_id": expression.id,
                    "conversation_id": expression.conversation_id or "",
                    "speech_register": (
                        expression.strategy.speech_register
                        if expression.strategy else "neutral"
                    ),
                },
            )
            await self._memory.store_percept(
                percept=percept,
                salience_composite=0.3,
                affect_valence=expression.affect_valence,
                affect_arousal=expression.affect_arousal,
                context_summary=f"I said ({trigger.value}): {content_text[:200]}",
            )
            self._logger.debug(
                "expression_stored_as_episode", expression_id=expression.id,
            )
        except Exception:
            self._logger.debug("expression_episode_storage_failed", exc_info=True)
