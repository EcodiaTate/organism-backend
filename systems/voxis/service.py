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
from typing import TYPE_CHECKING, Any

import structlog

from primitives.affect import AffectState
from primitives.expression import Expression, PersonalityVector
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
    from systems.atune.types import WorkspaceBroadcast
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

# Type alias for expression delivery callback
ExpressionCallback = Callable[[Expression], None]

# How often to drain the expression queue (seconds)
_QUEUE_DRAIN_INTERVAL_SECONDS = 30.0

# How often to expire unanswered reception tracking (seconds)
_RECEPTION_EXPIRE_INTERVAL_SECONDS = 60.0


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

        # Background task tracking -- prevents fire-and-forget error loss
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._background_task_failures: int = 0

        # Periodic background loop handles
        self._queue_drain_task: asyncio.Task[Any] | None = None
        self._reception_expire_task: asyncio.Task[Any] | None = None

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

        # Start background loops
        self._queue_drain_task = self._spawn_tracked_task(
            self._queue_drain_loop(),
            name="voxis_queue_drain",
        )
        self._reception_expire_task = self._spawn_tracked_task(
            self._reception_expire_loop(),
            name="voxis_reception_expire",
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

    async def shutdown(self) -> None:
        """Graceful shutdown -- cancel background loops, log final metrics."""
        if self._bus is not None:
            self._bus.deregister(BaseContentRenderer)

        # Cancel background loops
        if self._queue_drain_task and not self._queue_drain_task.done():
            self._queue_drain_task.cancel()
        if self._reception_expire_task and not self._reception_expire_task.done():
            self._reception_expire_task.cancel()

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
            # Evolved subclass has a different signature — try zero-arg
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

        intent = ExpressionIntent(
            trigger=ExpressionTrigger.ATUNE_DIRECT_ADDRESS,
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

        # Generate voice parameters for multimodal delivery
        self._voice_engine.derive(
            personality=self._personality_engine.current,  # type: ignore[union-attr]
            affect=current_affect,
            strategy_register=expression.strategy.speech_register if expression.strategy else "neutral",
            urgency=urgency,
        )

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

        # Record expression in diversity tracker
        self._diversity_tracker.record(
            expression.content or "",
            trigger=trigger.value,
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
                self._audience_profiler.observe_reception(
                    individual_id=speaker_id,
                    register_used=enriched_feedback.strategy_register,
                    formatting_used="prose",  # TODO: track from strategy
                    expression_length=enriched_feedback.user_response_length,
                    satisfaction=enriched_feedback.inferred_reception.satisfaction,
                )

            # Re-dispatch enriched feedback to Evo and other listeners
            for fb_cb in self._feedback_callbacks:
                try:
                    fb_cb(enriched_feedback)
                except Exception:
                    self._logger.debug("enriched_feedback_callback_failed", exc_info=True)

        return updated.conversation_id

    # --- Personality Update (called by Evo) -------------------------------

    def update_personality(self, delta: dict[str, float]) -> PersonalityVector:
        """
        Apply an incremental personality adjustment.
        Called by Evo after accumulating sufficient evidence.
        Returns the new PersonalityVector.
        """
        assert self._personality_engine is not None
        new_vector = self._personality_engine.apply_delta(delta)
        self._personality_engine = PersonalityEngine(new_vector)
        self._logger.info(
            "personality_updated_by_evo",
            dimensions=list(delta.keys()),
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

    async def _build_audience_profile(
        self,
        addressee_id: str | None,
        addressee_name: str | None,
        conversation_id: str,
        interaction_count: int,
    ) -> AudienceProfile:
        """Build an AudienceProfile, pulling facts from Memory where available."""
        memory_facts: list[dict[str, str]] = []

        if addressee_id:
            try:
                result = await asyncio.wait_for(
                    self._memory.retrieve(
                        query_text=f"individual person entity {addressee_id}",
                        max_results=5,
                    ),
                    timeout=0.1,
                )
                for entity in result.entities:
                    if entity.name == addressee_id or entity.id == addressee_id:
                        memory_facts.append({"type": "name", "value": entity.name})
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
        Store a delivered expression as a Memory episode.

        The organism remembers what it said -- closing the expression->memory loop.
        Without this, Voxis generates speech that vanishes from the organism's
        episodic history. Past expressions can't inform future decisions.
        """
        if self._memory is None:
            return
        try:
            from primitives.memory_trace import Episode
            from systems.memory.episodic import store_episode

            episode = Episode(
                source=f"voxis.expression:{trigger.value}",
                modality="text",
                raw_content=expression.content[:2000] if expression.content else "",
                summary=f"I said: {expression.content[:200]}" if expression.content else "",
                salience_composite=0.3,
                affect_valence=0.0,
            )
            await store_episode(self._memory._neo4j, episode)
            self._logger.debug(
                "expression_stored_as_episode", expression_id=expression.id,
            )
        except Exception:
            self._logger.debug("expression_episode_storage_failed", exc_info=True)
