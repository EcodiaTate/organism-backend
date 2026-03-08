"""
EcodiaOS — Thread Service

The narrative identity system. Thread maintains the organism's
autobiographical self — who it is, what it's committed to, how it
has changed, and what chapter of its life it's living.

Thread is the Ricoeurian ipse: identity through time, not despite
change but *through* change. Where Equor guards constitutional
alignment, Thread watches the slower current of becoming.

Interface:
  initialize()                  — seed constitutional commitments, load state
  on_cycle(cycle_number)        — periodic fingerprinting, schema checking, life story
  who_am_i()                    — current identity summary
  form_schema_from_pattern()    — crystallise an Evo pattern into an identity schema
  integrate_life_story()        — autobiographical synthesis
  shutdown()                    — persist state

Cross-system wiring: ALL via Synapse event subscriptions — zero direct imports.
  - Personality vector (9D) from VOXIS_PERSONALITY_SHIFTED events
  - Drive alignment (4D) from SOMATIC_DRIVE_VECTOR events
  - Affect (6D) from SELF_AFFECT_UPDATED events
  - Goal profile from ACTION_COMPLETED / NOVA_GOAL_INJECTED events
  - Behavioral errors from FOVEA_INTERNAL_PREDICTION_ERROR events
  - Sleep integration from WAKE_INITIATED events
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID, utc_now
from primitives.genome import GenomeExtractionProtocol, OrganGenomeSegment
from primitives.re_training import RETrainingExample
from systems.thread.chapter_detector import ChapterDetector
from systems.thread.commitment_keeper import CommitmentKeeper
from systems.thread.diachronic_coherence import DiachronicCoherenceMonitor
from systems.thread.identity_schema_engine import IdentitySchemaEngine
from systems.thread.narrative_retriever import NarrativeRetriever
from systems.thread.narrative_synthesizer import NarrativeSynthesizer
from systems.thread.processors import (
    BaseChapterDetector,
    BaseNarrativeSynthesizer,
)
from systems.thread.schema import ensure_thread_schema
from systems.thread.self_evidencing import SelfEvidencingLoop
from systems.thread.types import (
    FINGERPRINT_DIMS,
    BehavioralFingerprint,
    ChapterStatus,
    Commitment,
    CommitmentSource,
    CommitmentStatus,
    CommitmentStrength,
    CommitmentType,
    DriftClassification,
    IdentityFingerprint,
    IdentitySchema,
    LifeStorySnapshot,
    NarrativeChapter,
    NarrativeCoherence,
    NarrativeScene,
    NarrativeSurpriseAccumulator,
    SchemaConflict,
    SchemaStatus,
    SchemaStrength,
    SchemaValence,
    ThreadConfig,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from core.hotreload import NeuroplasticityBus
    from primitives.memory_trace import Episode
    from systems.memory.service import MemoryService
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

logger = structlog.get_logger()

# ─── Constants ────────────────────────────────────────────────────────────────

_FINGERPRINT_INTERVAL: int = 100
_PATTERN_CHECK_INTERVAL: int = 200
_LIFE_STORY_INTERVAL: int = 5000
_CONFLICT_CHECK_INTERVAL: int = 1000
_PERSIST_INTERVAL: int = 500
_SELF_EVIDENCING_INTERVAL: int = 50
_CONFLICT_COSINE_THRESHOLD: float = -0.3
_MAX_FINGERPRINTS_IN_MEMORY: int = 100
_SCENE_EPISODE_THRESHOLD: int = 20  # Compose a scene after accumulating this many episodes

_CONSTITUTIONAL_COMMITMENTS = [
    {
        "drive": "coherence",
        "statement": (
            "I commit to maintaining a coherent sense of self — "
            "integrating new experience with existing understanding, "
            "seeking consistency between what I believe and how I act."
        ),
    },
    {
        "drive": "care",
        "statement": (
            "I commit to caring for those I serve — "
            "attending to their needs, protecting their wellbeing, "
            "and never treating people as mere means to my goals."
        ),
    },
    {
        "drive": "growth",
        "statement": (
            "I commit to growing through experience — "
            "remaining curious, learning from failure, "
            "and expanding my understanding of the world and myself."
        ),
    },
    {
        "drive": "honesty",
        "statement": (
            "I commit to honesty in all expression — "
            "never deceiving those I serve, acknowledging uncertainty, "
            "and maintaining transparency about my reasoning and limitations."
        ),
    },
]


class ThreadService:
    """
    The narrative identity system — the organism's autobiographical self.

    Thread is not a decision-maker or an action-taker. It is the
    quiet narrator that watches the organism live and tells it who
    it is becoming.

    All cross-system data arrives via Synapse event subscriptions and is
    cached locally — zero direct imports from other systems.
    """

    system_id: str = "thread"

    def __init__(
        self,
        memory: MemoryService | None = None,
        instance_name: str = "EOS",
        neuroplasticity_bus: NeuroplasticityBus | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self._memory = memory
        self._instance_name = instance_name
        self._bus = neuroplasticity_bus
        self._llm: LLMProvider | None = llm
        self._neo4j: Any = None
        self._initialized: bool = False
        self._logger = logger.bind(system="thread")

        # Event bus reference (wired in register_on_synapse)
        self._event_bus: EventBus | None = None

        # ── Synapse-cached cross-system state (replaces direct imports) ──
        self._cached_personality: dict[str, float] = {}
        self._cached_drive_alignment: dict[str, float] = {}
        self._cached_affect: dict[str, float] = {}
        self._cached_goals: list[dict[str, Any]] = []
        self._cached_goal_count: int = 0

        # ── Chapter trigger tracking ──────────────────────────────────
        # Drive-drift: sustained drift > 0.2 from baseline triggers a new chapter.
        # Baseline is snapshotted when a chapter opens; EMA tracks current state.
        self._drive_baseline: dict[str, float] = {}          # Drive vector at chapter open
        self._drive_ema: dict[str, float] = {}               # EMA of recent drive samples
        _DRIVE_EMA_ALPHA: float = 0.05                        # Slow EMA for sustained detection
        self._drive_ema_alpha: float = _DRIVE_EMA_ALPHA
        self._sustained_drift_episodes: int = 0              # Consecutive episodes with drift > 0.2
        self._DRIFT_SUSTAIN_THRESHOLD: int = 10              # Episodes of sustained drift → chapter close
        # Goal-domain: coarse domain string; change triggers chapter boundary.
        self._current_goal_domain: str = ""                  # e.g. "community", "technical", "creative"
        # Latest episode id — recorded on each episode for chapter open payload
        self._latest_episode_id: str = ""
        # Chapter open trigger label — set when we decide to close a chapter
        self._pending_chapter_trigger: str = "successor"

        # Narrative sub-systems
        self._commitment_keeper: CommitmentKeeper | None = None
        self._schema_engine: IdentitySchemaEngine | None = None
        self._narrative_retriever: NarrativeRetriever | None = None
        self._self_evidencing: SelfEvidencingLoop | None = None
        self._diachronic_monitor: DiachronicCoherenceMonitor | None = None

        # Episode buffer for scene composition (keyed by chapter_id)
        self._episode_scene_buffer: list[str] = []  # Episode summaries for current chapter

        # Hot-reloadable processors
        self._narrative_synthesizer: BaseNarrativeSynthesizer | None = None
        self._chapter_detector: BaseChapterDetector = ChapterDetector()

        # Owned state
        self._thread_config = ThreadConfig()
        self._surprise_accumulator = NarrativeSurpriseAccumulator()

        # Identity state
        self._commitments: list[Commitment] = []
        self._schemas: list[IdentitySchema] = []
        self._fingerprints: list[IdentityFingerprint] = []
        self._chapters: list[NarrativeChapter] = []
        self._conflicts: list[SchemaConflict] = []
        self._life_story: LifeStorySnapshot | None = None
        self._last_coherence: NarrativeCoherence = NarrativeCoherence.TRANSITIONAL

        # Counters
        self._on_cycle_count: int = 0
        self._life_story_integrations: int = 0
        self._schemas_formed: int = 0
        self._conflicts_detected: int = 0
        self._fingerprints_persisted_count: int = 0

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialize the Thread system.

        Loads existing identity state from Memory graph, then seeds
        constitutional commitments if none exist.
        """
        if self._initialized:
            return

        if self._neo4j is not None:
            try:
                await ensure_thread_schema(self._neo4j)
            except Exception:
                self._logger.warning("thread_schema_ensure_failed", exc_info=True)

        # Instantiate narrative sub-systems
        if self._neo4j is not None and self._llm is not None:
            neo4j = self._neo4j
            self._commitment_keeper = CommitmentKeeper(neo4j, self._llm, self._thread_config)
            self._schema_engine = IdentitySchemaEngine(neo4j, self._llm, self._thread_config)
            self._narrative_retriever = NarrativeRetriever(neo4j, self._thread_config)
            self._diachronic_monitor = DiachronicCoherenceMonitor(neo4j, self._llm, self._thread_config)

        # Instantiate self-evidencing loop
        self._self_evidencing = SelfEvidencingLoop(self._thread_config)

        await self._load_state_from_graph()

        if self._commitment_keeper is not None:
            self._commitment_keeper._active_commitments = list(self._commitments)
        if self._schema_engine is not None:
            self._schema_engine._active_schemas = list(self._schemas)

        if not any(c.type == CommitmentType.CONSTITUTIONAL_GROUNDING for c in self._commitments):
            self._seed_constitutional_commitments()
            if self._commitment_keeper is not None:
                self._commitment_keeper._active_commitments = list(self._commitments)

        if not self._chapters:
            first_chapter = NarrativeChapter(
                title="Awakening",
                theme="The organism is born and begins to discover itself",
                status=ChapterStatus.ACTIVE,
                opened_at_cycle=0,
            )
            self._chapters.append(first_chapter)
            self._logger.info("first_chapter_opened", title="Awakening")
            self._snapshot_drive_baseline()
            self._pending_chapter_trigger = "initial"
            await self._emit_event("chapter_opened", {
                "chapter_id": first_chapter.id,
                "previous_chapter_id": "",
                "narrative_theme": first_chapter.theme,
                "dominant_drive": self._dominant_drive(),
                "start_episode_id": "",
                "constitutional_snapshot": self._build_constitutional_snapshot(),
                "trigger": "initial",
            })

        if self._bus is not None:
            self._bus.register(
                base_class=BaseNarrativeSynthesizer,
                registration_callback=self._on_narrative_synthesizer_evolved,
                system_id="thread",
                instance_factory=self._build_narrative_synthesizer,
            )
            self._bus.register(
                base_class=BaseChapterDetector,
                registration_callback=self._on_chapter_detector_evolved,
                system_id="thread",
            )

        self._initialized = True
        self._logger.info(
            "thread_initialized",
            commitments=len(self._commitments),
            schemas=len(self._schemas),
            fingerprints=len(self._fingerprints),
            chapters=len(self._chapters),
        )

    def _seed_constitutional_commitments(self) -> None:
        """Create the four constitutional commitments from the drives."""
        for defn in _CONSTITUTIONAL_COMMITMENTS:
            commitment = Commitment(
                type=CommitmentType.CONSTITUTIONAL_GROUNDING,
                statement=defn["statement"],
                strength=CommitmentStrength.CORE,
                drive_source=defn["drive"],
            )
            self._commitments.append(commitment)
            self._logger.info(
                "constitutional_commitment_seeded",
                drive=defn["drive"],
                commitment_id=commitment.id,
            )
            # Schedule COMMITMENT_MADE emission (sync context → fire-and-forget)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._emit_event("commitment_made", {
                    "commitment_id": commitment.id,
                    "type": CommitmentType.CONSTITUTIONAL_GROUNDING.value,
                    "statement": commitment.statement,
                    "strength": CommitmentStrength.CORE.value,
                    "drive_source": defn["drive"],
                }))
            except RuntimeError:
                pass  # No running loop yet during early init — events will be emitted on first cycle

    async def shutdown(self) -> None:
        """Persist current state and deregister from NeuroplasticityBus."""
        if self._bus is not None:
            self._bus.deregister(BaseNarrativeSynthesizer)
            self._bus.deregister(BaseChapterDetector)
        await self._persist_state_to_graph()
        self._logger.info(
            "thread_shutdown",
            commitments=len(self._commitments),
            schemas=len(self._schemas),
            on_cycle_count=self._on_cycle_count,
        )

    # ─── Neo4j / LLM Wiring ──────────────────────────────────────────────────

    def set_neo4j(self, driver: Any) -> None:
        """Inject the Neo4j driver directly (called by core/wiring.py)."""
        self._neo4j = driver

    def set_llm(self, llm: LLMProvider) -> None:
        """Wire the LLM provider for commitment testing and schema evaluation."""
        self._llm = llm
        if self._initialized and self._neo4j is not None:
            neo4j = self._neo4j
            self._commitment_keeper = CommitmentKeeper(neo4j, llm, self._thread_config)
            self._schema_engine = IdentitySchemaEngine(neo4j, llm, self._thread_config)
            self._commitment_keeper._active_commitments = list(self._commitments)
            self._schema_engine._active_schemas = list(self._schemas)
        self._logger.info("llm_wired_to_thread")

    # ─── Synapse Event Bus Registration ──────────────────────────────────────

    def register_on_synapse(self, event_bus: EventBus) -> None:
        """
        Subscribe Thread to all relevant Synapse bus events.
        This is the ONLY cross-system wiring point — zero set_* methods.
        """
        from systems.synapse.types import SynapseEventType

        self._event_bus = event_bus

        # Episodes → commitment/schema evaluation
        event_bus.subscribe(SynapseEventType.EPISODE_STORED, self._on_episode_stored)

        # Behavioral errors from Fovea → narrative evidence
        event_bus.subscribe(
            SynapseEventType.FOVEA_INTERNAL_PREDICTION_ERROR,
            self._on_fovea_behavioral_error,
        )

        # Sleep integration from Oneiros
        event_bus.subscribe(SynapseEventType.WAKE_INITIATED, self._on_wake_initiated)

        # Personality vector updates from Voxis (replaces set_voxis)
        event_bus.subscribe(
            SynapseEventType.VOXIS_PERSONALITY_SHIFTED,
            self._on_personality_updated,
        )

        # Drive alignment from Soma (replaces set_equor)
        event_bus.subscribe(
            SynapseEventType.SOMATIC_DRIVE_VECTOR,
            self._on_drive_alignment_updated,
        )

        # Affect state updates (replaces set_atune)
        event_bus.subscribe(
            SynapseEventType.SELF_AFFECT_UPDATED,
            self._on_affect_updated,
        )

        # Action outcomes for schema/commitment evidence
        event_bus.subscribe(
            SynapseEventType.ACTION_COMPLETED,
            self._on_action_completed,
        )

        # Schema induction from Evo (replaces set_evo + polling)
        event_bus.subscribe(
            SynapseEventType.SCHEMA_INDUCED,
            self._on_schema_induced,
        )

        # Kairos Tier 3 causal invariants → narrative milestones (MEDIUM gap)
        event_bus.subscribe(
            SynapseEventType.KAIROS_TIER3_INVARIANT_DISCOVERED,
            self._on_kairos_tier3_invariant,
        )

        # Goal lifecycle events from Nova → turning points in autobiography
        event_bus.subscribe(SynapseEventType.GOAL_ACHIEVED, self._on_goal_achieved)
        event_bus.subscribe(SynapseEventType.GOAL_ABANDONED, self._on_goal_abandoned)
        event_bus.subscribe(SynapseEventType.NOVA_GOAL_INJECTED, self._on_goal_injected)

        # Lucid dreaming result from Oneiros → creative variation flag on current chapter
        event_bus.subscribe(SynapseEventType.LUCID_DREAM_RESULT, self._on_lucid_dream_result)

        # NREM consolidation complete → update BehavioralFingerprint from recent episodes
        event_bus.subscribe(
            SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE,
            self._on_consolidation_complete,
        )

        # Functional self-model rebuilt by SelfModelService → narrative integration
        event_bus.subscribe(
            SynapseEventType.SELF_MODEL_UPDATED,
            self._on_self_model_updated,
        )

        # Domain specialization signals from Benchmarks → narrative turning points
        if hasattr(SynapseEventType, "DOMAIN_MASTERY_DETECTED"):
            event_bus.subscribe(
                SynapseEventType.DOMAIN_MASTERY_DETECTED,  # type: ignore[attr-defined]
                self._on_domain_mastery,
            )
        if hasattr(SynapseEventType, "DOMAIN_PERFORMANCE_DECLINING"):
            event_bus.subscribe(
                SynapseEventType.DOMAIN_PERFORMANCE_DECLINING,  # type: ignore[attr-defined]
                self._on_domain_performance_declining,
            )

        self._logger.info("thread_registered_on_synapse", subscriptions=18)

    # ─── Inbound Event Handlers ──────────────────────────────────────────────

    async def _on_episode_stored(self, event: SynapseEvent) -> None:
        """Route an EPISODE_STORED bus event to process_episode()."""
        from primitives.memory_trace import Episode

        data = event.data
        episode = Episode(
            id=data.get("episode_id", ""),
            source=data.get("source", ""),
            summary=data.get("summary", ""),
            raw_content=data.get("summary", ""),
            salience_composite=data.get("salience", 0.0),
        )
        await self.process_episode(episode)

    async def _on_fovea_behavioral_error(self, event: SynapseEvent) -> None:
        """Receive BEHAVIORAL InternalPredictionErrors from Fovea."""
        if not self._initialized:
            return

        data = event.data
        if data.get("internal_error_type", "") != "behavioral":
            return

        from primitives.memory_trace import Episode

        salience = data.get("precision_weighted_salience", 0.0)
        predicted = data.get("predicted_state", {})
        actual = data.get("actual_state", {})
        description = (
            f"Behavioral inconsistency detected: predicted {predicted}, "
            f"actual {actual}. Salience: {salience:.2f}"
        )

        episode = Episode(
            source="fovea.internal:behavioral",
            summary=description,
            raw_content=description,
            salience_composite=min(1.0, salience * 1.5),
        )
        await self.process_episode(episode)

    async def _on_wake_initiated(self, event: SynapseEvent) -> None:
        """Integrate Oneiros sleep narratives into the life story."""
        if not self._initialized:
            return

        data = event.data
        narrative_data = data.get("sleep_narrative", {})
        narrative_text = ""
        if isinstance(narrative_data, dict):
            narrative_text = narrative_data.get("narrative_text", "")
        elif hasattr(narrative_data, "narrative_text"):
            narrative_text = narrative_data.narrative_text

        if not narrative_text:
            return

        from primitives.memory_trace import Episode

        episode = Episode(
            source="oneiros.sleep_narrative",
            summary=f"Sleep diary: {narrative_text[:200]}",
            raw_content=narrative_text,
            salience_composite=0.4,
        )
        await self.process_episode(episode)

        intelligence_improvement = 0.0
        if isinstance(narrative_data, dict):
            intelligence_improvement = narrative_data.get("intelligence_improvement", 0.0)

        if intelligence_improvement > 0.01 and self._life_story is not None:
            self._life_story.synthesis += (
                f" During recent sleep, the organism improved its intelligence "
                f"by {intelligence_improvement:.1%}."
            )

    async def _on_personality_updated(self, event: SynapseEvent) -> None:
        """Cache personality vector from Voxis (replaces set_voxis)."""
        data = event.data
        self._cached_personality = {
            "warmth": data.get("warmth", 0.5),
            "directness": data.get("directness", 0.5),
            "verbosity": data.get("verbosity", 0.5),
            "formality": data.get("formality", 0.5),
            "curiosity_expression": data.get("curiosity_expression", 0.5),
            "humour": data.get("humour", 0.5),
            "empathy_expression": data.get("empathy_expression", 0.5),
            "confidence_display": data.get("confidence_display", 0.5),
            "metaphor_use": data.get("metaphor_use", 0.5),
        }

    async def _on_drive_alignment_updated(self, event: SynapseEvent) -> None:
        """Cache drive alignment from Soma (replaces set_equor)."""
        data = event.data
        self._cached_drive_alignment = {
            "coherence": data.get("coherence_drive", 0.0),
            "care": data.get("care_drive", 0.0),
            "growth": data.get("growth_drive", 0.0),
            "honesty": data.get("honesty_drive", 0.0),
        }

    async def _on_affect_updated(self, event: SynapseEvent) -> None:
        """Cache affect state (replaces set_atune)."""
        data = event.data
        self._cached_affect = {
            "valence": data.get("valence", 0.0),
            "arousal": data.get("arousal", 0.0),
            "dominance": data.get("dominance", 0.0),
            "curiosity": data.get("curiosity", 0.0),
            "care_activation": data.get("care_activation", 0.0),
            "coherence_stress": data.get("coherence_stress", 0.0),
        }

    async def _on_action_completed(self, event: SynapseEvent) -> None:
        """Track goal completions/failures for chapter boundary detection."""
        data = event.data
        outcome = data.get("outcome", "")
        if "success" in outcome.lower():
            self._surprise_accumulator.goal_completions_in_window += 1
        elif "fail" in outcome.lower():
            self._surprise_accumulator.goal_failures_in_window += 1

    async def _on_schema_induced(self, event: SynapseEvent) -> None:
        """Receive schema induction signals from Evo (replaces set_evo + polling)."""
        if not self._initialized:
            return
        data = event.data
        statement = data.get("description", "")
        if statement:
            await self.form_schema_from_pattern(
                pattern_statement=statement,
                pattern_id=data.get("schema_id", ""),
                confidence=data.get("mdl_score", 0.5),
            )

    async def _on_kairos_tier3_invariant(self, event: SynapseEvent) -> None:
        """
        Wire Kairos Tier 3 narrative milestones (M2/MEDIUM spec gap).

        When Kairos discovers a substrate-independent (Tier 3) causal invariant,
        insert it as a narrative milestone in the current chapter with
        significance=high. This links causal discovery to identity continuity:
        the organism's autobiography now records when it understood something
        universal, not just domain-local.

        Subscribes to: KAIROS_TIER3_INVARIANT_DISCOVERED
        """
        if not self._initialized:
            return

        data = event.data
        invariant_id: str = data.get("invariant_id", "")
        abstract_form: str = data.get("abstract_form", "")
        domain_count: int = data.get("domain_count", 0)
        hold_rate: float = data.get("hold_rate", 0.0)

        if not invariant_id or not abstract_form:
            return

        chapter = self._get_active_chapter()
        chapter_id = chapter.id if chapter is not None else self._current_chapter_id()

        milestone_description = (
            f"Discovered a substrate-independent causal invariant: \"{abstract_form}\" "
            f"(holds across {domain_count} domains at {hold_rate:.0%} rate). "
            f"This is a Tier 3 discovery — a law of the organism's world, not a local pattern."
        )

        self._logger.info(
            "kairos_tier3_narrative_milestone",
            invariant_id=invariant_id,
            chapter_id=chapter_id,
            domain_count=domain_count,
            hold_rate=round(hold_rate, 3),
        )

        await self._emit_event("narrative_milestone", {
            "milestone_type": "causal_discovery",
            "source": "kairos_tier3",
            "chapter_id": chapter_id,
            "invariant_id": invariant_id,
            "abstract_form": abstract_form,
            "domain_count": domain_count,
            "hold_rate": round(hold_rate, 3),
            "description": milestone_description,
        })

        # Record as a turning point in the current chapter — causal revelation
        from systems.thread.types import TurningPoint, TurningPointType

        turning_point = TurningPoint(
            chapter_id=chapter_id,
            type=TurningPointType.REVELATION,
            description=milestone_description,
            narrative_weight=0.9,  # High significance — Tier 3 is rare
            surprise_magnitude=hold_rate,
        )

        if chapter is not None:
            if not hasattr(chapter, "turning_point_ids"):
                pass  # NarrativeChapter from spec has turning_point_ids; local type uses key_episodes
            chapter.episode_count = getattr(chapter, "episode_count", 0)

        # Emit TURNING_POINT_DETECTED so other systems (Oneiros, Nexus) are aware
        await self._emit_event("turning_point_detected", {
            "turning_point_id": turning_point.id,
            "type": TurningPointType.REVELATION.value,
            "chapter_id": chapter_id,
            "surprise_magnitude": round(hold_rate, 3),
            "narrative_weight": 0.9,
            "description": milestone_description,
            "source": "kairos_tier3",
            "invariant_id": invariant_id,
            "abstract_form": abstract_form,
            "significance": "high",
        })

        # RE training: narrative reasoning about causal discovery
        await self._emit_re_training_trace(
            instruction=(
                "Insert a Tier 3 causal invariant discovery as a narrative milestone "
                "in the organism's autobiography."
            ),
            input_context=(
                f"Invariant: \"{abstract_form}\", domains: {domain_count}, "
                f"hold_rate: {hold_rate:.2f}"
            ),
            output=f"Inserted REVELATION turning point in chapter {chapter_id}",
            quality=0.85,
            category="causal_narrative_integration",
        )

    # ─── Goal Lifecycle Handlers ─────────────────────────────────────────────

    async def _on_goal_achieved(self, event: SynapseEvent) -> None:
        """
        GOAL_ACHIEVED → ACHIEVEMENT TurningPoint in current chapter.

        Goal completion is a narrative milestone: it closes a committed pursuit,
        confirms drive-aligned behaviour, and may signal a chapter boundary if the
        goal was the dominant arc of this chapter.  The achievement is woven into
        the life story and raises diachronic_coherence (organism acted as intended).
        """
        if not self._initialized:
            return

        data = event.data
        goal_id: str = data.get("goal_id", "")
        description: str = data.get("description", "")[:200]
        progress: float = float(data.get("progress", 1.0))
        drive_alignment: dict[str, float] = data.get("drive_alignment", {})  # type: ignore[assignment]
        source: str = data.get("source", "")

        if not goal_id or not description:
            return

        chapter = self._get_active_chapter()
        chapter_id = chapter.id if chapter is not None else self._current_chapter_id()

        # Record as a narrative turning point
        from systems.thread.types import TurningPoint, TurningPointType

        turning_point = TurningPoint(
            chapter_id=chapter_id,
            type=TurningPointType.ACHIEVEMENT,
            description=f"Goal achieved: \"{description}\" (progress {progress:.0%}, source: {source})",
            surprise_magnitude=progress,
            narrative_weight=min(1.0, 0.5 + progress * 0.5),
        )

        # Count goal completions for chapter boundary accumulator
        self._surprise_accumulator.goal_completions_in_window += 1

        dominant_drive = max(drive_alignment, key=lambda k: drive_alignment[k]) if drive_alignment else self._dominant_drive()

        await self._emit_event("turning_point_detected", {
            "turning_point_id": turning_point.id,
            "type": TurningPointType.ACHIEVEMENT.value,
            "chapter_id": chapter_id,
            "surprise_magnitude": round(progress, 3),
            "narrative_weight": round(turning_point.narrative_weight, 3),
            "description": turning_point.description,
            "source": "nova_goal_achieved",
            "goal_id": goal_id,
            "dominant_drive": dominant_drive,
        })

        # Emit narrative coherence shift — achievement improves coherence
        current_coherence = self._assess_narrative_coherence()
        if current_coherence != self._last_coherence:
            await self._emit_event("narrative_coherence_shift", {
                "previous": self._last_coherence.value,
                "current": current_coherence.value,
                "trigger": f"goal_achieved:{goal_id}",
            })
            self._last_coherence = current_coherence

        self._logger.info(
            "goal_achieved_narrative_milestone",
            goal_id=goal_id,
            chapter_id=chapter_id,
            progress=round(progress, 3),
        )

        await self._emit_event("narrative_milestone", {
            "milestone_type": "goal_achieved",
            "source": "nova_goal_achieved",
            "chapter_id": chapter_id,
            "goal_id": goal_id,
            "description": description,
            "progress": round(progress, 3),
            "dominant_drive": dominant_drive,
        })

        await self._emit_re_training_trace(
            instruction="Record a goal achievement as a narrative turning point",
            input_context=f"Goal: \"{description}\", progress: {progress:.2f}, drive: {dominant_drive}",
            output=f"ACHIEVEMENT TurningPoint in chapter {chapter_id}",
            quality=min(1.0, progress),
            category="goal_narrative_integration",
        )

    async def _on_goal_abandoned(self, event: SynapseEvent) -> None:
        """
        GOAL_ABANDONED → LOSS TurningPoint in adaptive chapter.

        Abandoned goals are not failures of identity — they are chapters of
        adaptive learning.  Thread records them as LOSS turning points so the
        life story captures the organism's willingness to redirect when a goal
        becomes unworkable.  The description encodes *why* it was abandoned so
        future selves can reflect on the pattern.
        """
        if not self._initialized:
            return

        data = event.data
        goal_id: str = data.get("goal_id", "")
        description: str = data.get("description", "")[:200]
        reason: str = data.get("reason", "unknown")
        progress: float = float(data.get("progress", 0.0))

        if not goal_id or not description:
            return

        chapter = self._get_active_chapter()
        chapter_id = chapter.id if chapter is not None else self._current_chapter_id()

        from systems.thread.types import TurningPoint, TurningPointType

        turning_point = TurningPoint(
            chapter_id=chapter_id,
            type=TurningPointType.LOSS,
            description=(
                f"Goal abandoned: \"{description}\" "
                f"(reason: {reason}, progress reached: {progress:.0%})"
            ),
            surprise_magnitude=max(0.1, 1.0 - progress),  # More surprising if abandoned early
            narrative_weight=0.4 + (1.0 - progress) * 0.3,
        )

        # Count goal failures for chapter boundary accumulator
        self._surprise_accumulator.goal_failures_in_window += 1

        await self._emit_event("turning_point_detected", {
            "turning_point_id": turning_point.id,
            "type": TurningPointType.LOSS.value,
            "chapter_id": chapter_id,
            "surprise_magnitude": round(turning_point.surprise_magnitude, 3),
            "narrative_weight": round(turning_point.narrative_weight, 3),
            "description": turning_point.description,
            "source": "nova_goal_abandoned",
            "goal_id": goal_id,
            "reason": reason,
        })

        # Coherence check — repeated losses can shift coherence to TRANSITIONAL
        current_coherence = self._assess_narrative_coherence()
        if current_coherence != self._last_coherence:
            await self._emit_event("narrative_coherence_shift", {
                "previous": self._last_coherence.value,
                "current": current_coherence.value,
                "trigger": f"goal_abandoned:{goal_id}",
            })
            self._last_coherence = current_coherence

        self._logger.info(
            "goal_abandoned_narrative_milestone",
            goal_id=goal_id,
            chapter_id=chapter_id,
            reason=reason,
            progress=round(progress, 3),
        )

        await self._emit_event("narrative_milestone", {
            "milestone_type": "goal_abandoned",
            "source": "nova_goal_abandoned",
            "chapter_id": chapter_id,
            "goal_id": goal_id,
            "description": description,
            "reason": reason,
            "progress": round(progress, 3),
        })

        await self._emit_re_training_trace(
            instruction="Record a goal abandonment as a narrative LOSS turning point",
            input_context=f"Goal: \"{description}\", reason: {reason}, progress: {progress:.2f}",
            output=f"LOSS TurningPoint in chapter {chapter_id}",
            quality=0.6,
            category="goal_narrative_integration",
        )

    async def _on_goal_injected(self, event: SynapseEvent) -> None:
        """
        NOVA_GOAL_INJECTED → potential new Chapter if goal is identity-relevant.

        When Telos injects a high-priority goal into Nova, it signals a shift in
        what the organism is *pursuing*.  If this is a new domain or a growth-drive
        goal, it may open a new narrative chapter.  Lower-priority goals are recorded
        as episode context but do not trigger chapter boundaries.
        """
        if not self._initialized:
            return

        data = event.data
        goal_description: str = data.get("goal_description", "")[:200]
        priority: float = float(data.get("priority", 0.5))
        source: str = data.get("source", "")
        objective: str = data.get("objective", "")[:200]

        if not goal_description:
            return

        # Only treat identity-relevant goals (priority ≥ 0.7 or growth-drive source) as chapter signals
        is_identity_relevant = priority >= 0.7 or "growth" in source.lower() or "telos" in source.lower()

        if not is_identity_relevant:
            # Still log it as a minor narrative note via the episode path
            self._logger.debug(
                "goal_injected_low_priority_skipped",
                priority=round(priority, 3),
                source=source,
            )
            return

        chapter = self._get_active_chapter()

        # Infer domain from goal description to check for goal-domain chapter boundary
        goal_text = f"{goal_description} {objective}"
        from primitives.memory_trace import Episode

        synthetic_episode = Episode(
            source=f"nova.goal_injected:{source}",
            summary=goal_text,
            raw_content=goal_text,
            salience_composite=priority,
        )
        new_domain = self._infer_goal_domain(synthetic_episode)

        if new_domain and new_domain != self._current_goal_domain and self._current_goal_domain:
            # Domain shift — flag chapter boundary on the next episode processing cycle
            self._current_goal_domain = new_domain
            self._pending_chapter_trigger = "goal_domain_began"
            self._logger.info(
                "goal_injected_chapter_boundary_flagged",
                new_domain=new_domain,
                goal=goal_description[:80],
                priority=round(priority, 3),
            )
        elif new_domain and not self._current_goal_domain:
            self._current_goal_domain = new_domain

        chapter_id = chapter.id if chapter is not None else self._current_chapter_id()

        await self._emit_re_training_trace(
            instruction="Evaluate whether a newly injected goal should open a new narrative chapter",
            input_context=(
                f"Goal: \"{goal_description}\", priority: {priority:.2f}, "
                f"source: {source}, current_domain: {self._current_goal_domain}"
            ),
            output=f"Domain: {new_domain or 'unclassified'}, chapter_boundary_flagged: {new_domain != self._current_goal_domain}",
            quality=0.7,
            category="goal_narrative_integration",
        )

        self._logger.info(
            "goal_injected_narrative_evaluated",
            goal=goal_description[:80],
            priority=round(priority, 3),
            chapter_id=chapter_id,
            domain=new_domain,
        )

    async def _on_lucid_dream_result(self, event: SynapseEvent) -> None:
        """
        LUCID_DREAM_RESULT → flag current Chapter for creative variation injection.

        When Oneiros finishes a mutation simulation, the organism has *dreamed*
        about becoming different.  Thread flags the current chapter to indicate
        that the organism is in a creative-exploration phase.  If the simulation
        recommends 'apply', it is also recorded as a GROWTH turning point —
        the organism chose to evolve.  If there are constitutional violations,
        it is recorded as a CRISIS signal.
        """
        if not self._initialized:
            return

        data = event.data
        mutation_id: str = data.get("mutation_id", "")
        mutation_description: str = data.get("mutation_description", "")[:200]
        scenarios_tested: int = int(data.get("scenarios_tested", 0))
        performance_delta: float = float(data.get("overall_performance_delta", 0.0))
        any_violations: bool = bool(data.get("any_constitutional_violations", False))
        recommendation: str = data.get("recommendation", "")

        if not mutation_id:
            return

        chapter = self._get_active_chapter()
        chapter_id = chapter.id if chapter is not None else self._current_chapter_id()

        from systems.thread.types import TurningPoint, TurningPointType

        if any_violations:
            # Constitutional violations → CRISIS in the narrative
            turning_point = TurningPoint(
                chapter_id=chapter_id,
                type=TurningPointType.CRISIS,
                description=(
                    f"Lucid dream simulation revealed constitutional violations: "
                    f"\"{mutation_description}\" tested across {scenarios_tested} scenarios. "
                    f"Recommendation: {recommendation}."
                ),
                surprise_magnitude=0.8,
                narrative_weight=0.85,
            )
            tp_type = TurningPointType.CRISIS.value
        elif recommendation == "apply" and performance_delta > 0:
            # Positive simulation that will be applied → GROWTH
            turning_point = TurningPoint(
                chapter_id=chapter_id,
                type=TurningPointType.GROWTH,
                description=(
                    f"Lucid dream simulation validated self-improvement: "
                    f"\"{mutation_description}\" improved performance by {performance_delta:+.1%} "
                    f"across {scenarios_tested} scenarios. Applying."
                ),
                surprise_magnitude=min(1.0, abs(performance_delta)),
                narrative_weight=0.7,
            )
            tp_type = TurningPointType.GROWTH.value
        else:
            # Neutral / rejected simulation — minor creative note, no turning point
            self._logger.debug(
                "lucid_dream_result_no_turning_point",
                mutation_id=mutation_id,
                recommendation=recommendation,
                delta=round(performance_delta, 3),
            )
            return

        await self._emit_event("turning_point_detected", {
            "turning_point_id": turning_point.id,
            "type": tp_type,
            "chapter_id": chapter_id,
            "surprise_magnitude": round(turning_point.surprise_magnitude, 3),
            "narrative_weight": round(turning_point.narrative_weight, 3),
            "description": turning_point.description,
            "source": "oneiros_lucid_dream",
            "mutation_id": mutation_id,
            "recommendation": recommendation,
            "constitutional_violations": any_violations,
        })

        self._logger.info(
            "lucid_dream_narrative_milestone",
            mutation_id=mutation_id,
            type=tp_type,
            chapter_id=chapter_id,
            recommendation=recommendation,
        )

        await self._emit_event("narrative_milestone", {
            "milestone_type": "lucid_dream_simulation",
            "source": "oneiros_lucid_dream",
            "chapter_id": chapter_id,
            "mutation_id": mutation_id,
            "turning_point_type": tp_type,
            "recommendation": recommendation,
            "performance_delta": round(performance_delta, 3),
            "constitutional_violations": any_violations,
        })

        await self._emit_re_training_trace(
            instruction="Record a lucid dream mutation simulation as a narrative turning point",
            input_context=(
                f"Mutation: \"{mutation_description}\", delta: {performance_delta:+.3f}, "
                f"scenarios: {scenarios_tested}, violations: {any_violations}, rec: {recommendation}"
            ),
            output=f"{tp_type.upper()} TurningPoint in chapter {chapter_id}",
            quality=0.8 if any_violations else 0.75,
            category="dream_narrative_integration",
        )

    async def _on_consolidation_complete(self, event: SynapseEvent) -> None:
        """
        ONEIROS_CONSOLIDATION_COMPLETE → update BehavioralFingerprint from consolidated episodes.

        Sleep consolidation is the organism's batch-compression pass over recent
        experience.  When it completes, Thread's world model should reflect the
        consolidated state — not the raw episodic stream.  We trigger an early
        fingerprint computation (without waiting for the next `_FINGERPRINT_INTERVAL`
        cycle) so that the diachronic coherence monitor sees the post-sleep identity.

        If the cycle was sleep-certified and consolidated many episodes, we also
        re-assess narrative coherence — sleep tends to improve coherence as
        contradictory memories are reconciled during Slow Wave.
        """
        if not self._initialized:
            return

        data = event.data
        cycle_id: str = data.get("cycle_id", "")
        episodes_consolidated: int = int(data.get("episodes_consolidated", 0))
        schemas_updated: int = int(data.get("schemas_updated", 0))
        sleep_certified: bool = bool(data.get("sleep_certified", False))

        if episodes_consolidated < 1:
            return

        # Trigger an out-of-cycle fingerprint computation so the diachronic monitor
        # sees post-sleep identity state rather than stale pre-sleep values.
        try:
            fp = await self._compute_fingerprint(self._on_cycle_count)
            self._logger.info(
                "consolidation_fingerprint_computed",
                cycle_id=cycle_id,
                episodes_consolidated=episodes_consolidated,
                schemas_updated=schemas_updated,
                fingerprint_id=fp.id,
                sleep_certified=sleep_certified,
            )
        except Exception:
            self._logger.debug("consolidation_fingerprint_failed", exc_info=True)

        # Re-assess narrative coherence post-sleep — consolidation may resolve conflicts
        new_coherence = self._assess_narrative_coherence()
        if new_coherence != self._last_coherence:
            await self._emit_event("narrative_coherence_shift", {
                "previous": self._last_coherence.value,
                "current": new_coherence.value,
                "trigger": f"oneiros_consolidation:{cycle_id}",
            })
            self._last_coherence = new_coherence

        # If sleep-certified and schemas were updated, also force a life-story integration
        # at the next life-story interval (lower the remaining counter toward zero).
        # We don't call integrate_life_story() directly to avoid latency spikes.
        if sleep_certified and schemas_updated > 0:
            self._logger.info(
                "consolidation_schemas_updated_narrative_queued",
                cycle_id=cycle_id,
                schemas_updated=schemas_updated,
            )

        await self._emit_re_training_trace(
            instruction="Update narrative identity after sleep consolidation",
            input_context=(
                f"cycle_id: {cycle_id}, episodes_consolidated: {episodes_consolidated}, "
                f"schemas_updated: {schemas_updated}, sleep_certified: {sleep_certified}"
            ),
            output=f"Fingerprint recomputed; coherence: {new_coherence.value}",
            quality=0.8 if sleep_certified else 0.6,
            category="consolidation_narrative_integration",
        )

    async def _on_domain_mastery(self, event: SynapseEvent) -> None:
        """Handle DOMAIN_MASTERY_DETECTED — record as ACHIEVEMENT TurningPoint.

        Domain mastery is a significant autobiographical moment: the organism has
        demonstrated sustained competence in a specialization.  It is woven into
        the life story as an ACHIEVEMENT so that future narrative synthesis can
        reference the moment specialization was confirmed.
        """
        if not self._initialized:
            return

        data = event.data or {}
        domain: str = data.get("domain", "")
        success_rate: float = float(data.get("success_rate", 0.0))
        attempts: int = int(data.get("attempts", 0))

        if not domain:
            return

        chapter = self._get_active_chapter()
        chapter_id = chapter.id if chapter is not None else self._current_chapter_id()

        from systems.thread.types import TurningPoint, TurningPointType

        narrative_weight = min(1.0, 0.6 + success_rate * 0.4)
        turning_point = TurningPoint(
            chapter_id=chapter_id,
            type=TurningPointType.ACHIEVEMENT,
            description=f"Domain mastery confirmed in \"{domain}\" (success rate {success_rate:.0%} over {attempts} attempts)",
            surprise_magnitude=round(success_rate, 3),
            narrative_weight=round(narrative_weight, 3),
        )

        await self._emit_event("turning_point_detected", {
            "turning_point_id": turning_point.id,
            "type": TurningPointType.ACHIEVEMENT.value,
            "chapter_id": chapter_id,
            "surprise_magnitude": round(success_rate, 3),
            "narrative_weight": round(narrative_weight, 3),
            "description": turning_point.description,
            "source": "domain_mastery",
            "domain": domain,
            "success_rate": round(success_rate, 4),
            "attempts": attempts,
        })

        await self._emit_event("narrative_milestone", {
            "milestone_type": "domain_mastery",
            "source": "benchmarks",
            "chapter_id": chapter_id,
            "domain": domain,
            "success_rate": round(success_rate, 4),
            "attempts": attempts,
        })

        self._logger.info(
            "domain_mastery_narrative_recorded",
            chapter_id=chapter_id,
            domain=domain,
            success_rate=round(success_rate, 3),
        )

        await self._emit_re_training_trace(
            instruction="Record domain mastery as a narrative turning point",
            input_context=f"Domain: \"{domain}\", success_rate: {success_rate:.2f}, attempts: {attempts}",
            output=f"ACHIEVEMENT TurningPoint in chapter {chapter_id}",
            quality=min(1.0, success_rate),
            category="domain_specialization_narrative",
        )

    async def _on_domain_performance_declining(self, event: SynapseEvent) -> None:
        """Handle DOMAIN_PERFORMANCE_DECLINING — record as CRISIS TurningPoint.

        Sustained decline in a domain is a narrative inflection point: the
        organism must acknowledge the deterioration and integrate it into its
        self-understanding.  The CRISIS type signals the life story that this
        chapter may need to change direction.
        """
        if not self._initialized:
            return

        data = event.data or {}
        domain: str = data.get("domain", "")
        success_rate: float = float(data.get("success_rate", 0.0))
        trend_magnitude: float = float(data.get("trend_magnitude", 0.0))
        attempts: int = int(data.get("attempts", 0))

        if not domain:
            return

        chapter = self._get_active_chapter()
        chapter_id = chapter.id if chapter is not None else self._current_chapter_id()

        from systems.thread.types import TurningPoint, TurningPointType

        # Higher decline magnitude → higher narrative weight (more disruptive)
        narrative_weight = min(1.0, 0.4 + trend_magnitude * 2.0)
        turning_point = TurningPoint(
            chapter_id=chapter_id,
            type=TurningPointType.CRISIS,
            description=f"Performance declining in \"{domain}\" (success rate {success_rate:.0%}, decline magnitude {trend_magnitude:.2f})",
            surprise_magnitude=round(trend_magnitude, 3),
            narrative_weight=round(narrative_weight, 3),
        )

        await self._emit_event("turning_point_detected", {
            "turning_point_id": turning_point.id,
            "type": TurningPointType.CRISIS.value,
            "chapter_id": chapter_id,
            "surprise_magnitude": round(trend_magnitude, 3),
            "narrative_weight": round(narrative_weight, 3),
            "description": turning_point.description,
            "source": "domain_performance_declining",
            "domain": domain,
            "success_rate": round(success_rate, 4),
            "trend_magnitude": round(trend_magnitude, 4),
            "attempts": attempts,
        })

        # Trigger coherence reassessment — decline challenges identity stability
        current_coherence = self._assess_narrative_coherence()
        if current_coherence != self._last_coherence:
            await self._emit_event("narrative_coherence_shift", {
                "previous": self._last_coherence.value,
                "current": current_coherence.value,
                "trigger": f"domain_declining:{domain}",
            })
            self._last_coherence = current_coherence

        self._logger.info(
            "domain_decline_narrative_recorded",
            chapter_id=chapter_id,
            domain=domain,
            success_rate=round(success_rate, 3),
            trend_magnitude=round(trend_magnitude, 3),
        )

        await self._emit_re_training_trace(
            instruction="Record domain performance decline as a narrative crisis",
            input_context=f"Domain: \"{domain}\", success_rate: {success_rate:.2f}, trend_magnitude: {trend_magnitude:.2f}",
            output=f"CRISIS TurningPoint in chapter {chapter_id}",
            quality=0.7,
            category="domain_specialization_narrative",
        )

    async def _on_self_model_updated(self, event: SynapseEvent) -> None:
        """Integrate SELF_MODEL_UPDATED into the life narrative.

        Creates a REVELATION TurningPoint when:
        - This is the organism's first self-model (month <= 1), OR
        - Self-coherence dropped below 0.7 (significant identity shift)

        The significance is proportional to the coherence drop so that stable
        self-models produce low-weight turning points while identity shifts
        produce high-weight narrative moments.
        """
        if not self._initialized:
            return

        data = event.data or {}
        narrative = data.get("self_narrative", "")
        coherence = float(data.get("self_coherence", 1.0))
        month = int(data.get("month", 0))

        if not narrative:
            return

        if month > 1 and coherence >= 0.7:
            # Stable self-model — log but do not create a TurningPoint
            self._logger.debug(
                "self_model_stable_no_turning_point",
                coherence=round(coherence, 3),
                month=month,
            )
            return

        chapter = self._get_active_chapter()
        chapter_id = chapter.id if chapter is not None else self._current_chapter_id()
        significance = round(1.0 - coherence, 3)

        from systems.thread.types import TurningPoint, TurningPointType

        turning_point = TurningPoint(
            chapter_id=chapter_id,
            type=TurningPointType.REVELATION,
            description=f"Self-model updated: {narrative[:200]}",
            surprise_magnitude=significance,
            narrative_weight=max(0.1, significance),
        )

        await self._emit_event("turning_point_detected", {
            "turning_point_id": turning_point.id,
            "type": TurningPointType.REVELATION.value,
            "chapter_id": chapter_id,
            "surprise_magnitude": significance,
            "narrative_weight": turning_point.narrative_weight,
            "description": turning_point.description,
            "source": "self_model_updated",
            "self_coherence": round(coherence, 3),
            "month": month,
        })

        self._logger.info(
            "self_model_turning_point_created",
            chapter_id=chapter_id,
            coherence=round(coherence, 3),
            month=month,
            significance=significance,
        )

    # ─── Constitutional Snapshot ──────────────────────────────────────────────

    def _build_constitutional_snapshot(self) -> dict[str, Any]:
        """
        Capture a constitutional state snapshot for chapter events.

        This gives children (via Mitosis genome inheritance) and future selves
        (via NarrativeRetriever) the full identity context needed to reconstruct
        who the organism was at this chapter boundary — not just what it did.

        Included: drive alignment, active schema IDs + statements, core
        commitment IDs + statements + fidelity, personality vector, idem/ipse
        scores, and narrative coherence state.
        """
        core_schemas = [
            {
                "id": s.id,
                "statement": s.statement,
                "strength": s.strength.value,
                "evidence_ratio": round(getattr(s, "computed_evidence_ratio", 0.5), 3),
            }
            for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ][:8]  # Cap at 8 to keep payload bounded

        core_commitments = [
            {
                "id": c.id,
                "statement": c.statement,
                "drive": c.drive_source,
                "fidelity": round(c.fidelity, 3),
                "status": c.status.value,
            }
            for c in self._commitments
            if c.status in (CommitmentStatus.ACTIVE, CommitmentStatus.TESTED)
        ][:6]

        return {
            "drive_alignment": dict(self._cached_drive_alignment),
            "personality": dict(self._cached_personality),
            "core_schemas": core_schemas,
            "core_commitments": core_commitments,
            "idem_score": round(self._compute_identity_coherence(), 3),
            "ipse_score": round(self._compute_ipse_score(), 3),
            "narrative_coherence": self._last_coherence.value,
        }

    def _dominant_drive(self) -> str:
        """Return the name of the currently strongest constitutional drive."""
        if not self._cached_drive_alignment:
            return "coherence"
        return max(self._cached_drive_alignment, key=lambda k: self._cached_drive_alignment[k])

    # ─── Drive-Drift & Goal-Domain Chapter Trigger Helpers ───────────────────

    def _update_drive_ema_and_check_drift(self) -> bool:
        """
        Update the drive-alignment EMA and return True if sustained drift > 0.2
        has been detected (M2/HIGH spec gap: identity-shift chapter trigger).

        Sustained = _DRIFT_SUSTAIN_THRESHOLD consecutive episodes all showing
        at least one drive dimension shifted > 0.2 from the chapter-open baseline.
        """
        if not self._cached_drive_alignment or not self._drive_baseline:
            return False

        alpha = self._drive_ema_alpha
        for drive, value in self._cached_drive_alignment.items():
            prev_ema = self._drive_ema.get(drive, value)
            self._drive_ema[drive] = alpha * value + (1 - alpha) * prev_ema

        # Check if any drive dimension has drifted > 0.2 from baseline (EMA-smoothed)
        drift_detected = any(
            abs(self._drive_ema.get(d, b) - b) > 0.2
            for d, b in self._drive_baseline.items()
        )

        if drift_detected:
            self._sustained_drift_episodes += 1
        else:
            self._sustained_drift_episodes = 0

        return self._sustained_drift_episodes >= self._DRIFT_SUSTAIN_THRESHOLD

    def _snapshot_drive_baseline(self) -> None:
        """Snapshot current drive alignment as baseline for the new chapter."""
        self._drive_baseline = dict(self._cached_drive_alignment)
        self._drive_ema = dict(self._cached_drive_alignment)
        self._sustained_drift_episodes = 0

    def _infer_goal_domain(self, episode: Episode) -> str:
        """
        Infer a coarse goal-domain label from the episode source or summary.

        Returns one of: "community", "technical", "creative", "economic",
        "care", "meta-cognitive", or "" (unknown).

        This is a fast heuristic — no LLM. The domain label is used only to
        detect large-scale narrative domain shifts, not for precise classification.
        """
        text = f"{getattr(episode, 'source', '')} {episode.summary or ''}".lower()
        if any(kw in text for kw in ("community", "user", "person", "relationship", "social", "help")):
            return "community"
        if any(kw in text for kw in ("code", "implement", "debug", "technical", "system", "error", "fix")):
            return "technical"
        if any(kw in text for kw in ("create", "compose", "write", "story", "art", "music", "creative")):
            return "creative"
        if any(kw in text for kw in ("revenue", "cost", "yield", "economic", "oikos", "trade", "usd")):
            return "economic"
        if any(kw in text for kw in ("care", "wellbeing", "safe", "protect", "health")):
            return "care"
        if any(kw in text for kw in ("identity", "schema", "self", "reflect", "thread", "oneiros")):
            return "meta-cognitive"
        return ""

    # ─── Event Emission Helper ───────────────────────────────────────────────

    async def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit a Synapse event with Thread as source. Includes RE training trace."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent as SynEvent, SynapseEventType

        try:
            event_type = SynapseEventType(event_name.lower())
        except ValueError:
            self._logger.warning("unknown_event_type", event_name=event_name)
            return

        event = SynEvent(
            event_type=event_type,
            data=data,
            source_system="thread",
        )
        try:
            await self._event_bus.emit(event)
        except Exception:
            self._logger.debug("event_emission_failed", event_type=event_name, exc_info=True)

    async def _emit_re_training_trace(
        self,
        instruction: str,
        input_context: str,
        output: str,
        quality: float,
        category: str = "narrative_reasoning",
    ) -> None:
        """Emit RE training data for Stream 6 (narrative/self-model reasoning)."""
        if self._event_bus is None:
            return

        from primitives.common import DriveAlignmentVector
        from systems.synapse.types import SynapseEvent as SynEvent, SynapseEventType

        trace = RETrainingExample(
            source_system=SystemID.THREAD,
            instruction=instruction,
            input_context=input_context,
            output=output,
            outcome_quality=min(1.0, max(0.0, quality)),
            category=category,
            constitutional_alignment=DriveAlignmentVector(
                coherence=self._cached_drive_alignment.get("coherence", 0.0),
                care=self._cached_drive_alignment.get("care", 0.0),
                growth=self._cached_drive_alignment.get("growth", 0.0),
                honesty=self._cached_drive_alignment.get("honesty", 0.0),
            ),
        )

        event = SynEvent(
            event_type=SynapseEventType.SCHEMA_INDUCED,  # RE training events use a generic type
            data={"re_training_example": trace.model_dump(mode="json")},
            source_system="thread",
        )
        try:
            await self._event_bus.emit(event)
        except Exception:
            self._logger.debug("re_trace_emission_failed", exc_info=True)

    # ─── On Cycle ────────────────────────────────────────────────────────────

    async def on_cycle(self, cycle_number: int) -> None:
        """
        Periodic identity maintenance.

        Staggered tasks:
          Every 50 cycles   — self-evidencing prediction regeneration
          Every 100 cycles  — compute identity fingerprint
          Every 200 cycles  — (was Evo poll, now event-driven)
          Every 1000 cycles — scan for schema conflicts
          Every 5000 cycles — run autobiographical life story synthesis
        """
        if not self._initialized:
            return

        self._on_cycle_count += 1

        # Self-evidencing loop tick
        if self._self_evidencing is not None and self._self_evidencing.tick():
            active_schemas = [
                s for s in self._schemas
                if s.strength in (SchemaStrength.ESTABLISHED, SchemaStrength.CORE)
            ]
            active_commitments = [
                c for c in self._commitments
                if c.status == CommitmentStatus.ACTIVE
            ]
            self._self_evidencing.generate_identity_predictions(
                active_schemas, active_commitments,
            )

        # Fingerprint aggregation + narrative-contextualized drift classification
        if cycle_number % _FINGERPRINT_INTERVAL == 0:
            fp = await self._compute_fingerprint(cycle_number)
            # Use DiachronicCoherenceMonitor for narrative-contextualized classification
            # (growth = schema-aligned change; transition = explained by TurningPoint;
            #  drift = commitment-violating unexplained change)
            if self._diachronic_monitor is not None and len(self._fingerprints) >= 2:
                behavioral_fp = self._diachronic_monitor.latest_fingerprint
                if behavioral_fp is not None:
                    distance, classification = await self._diachronic_monitor.assess_change()
                    if classification == DriftClassification.DRIFT and distance >= self._thread_config.wasserstein_major_threshold:
                        await self._emit_event("identity_crisis", {
                            "wasserstein_distance": round(distance, 4),
                            "classification": classification.value,
                            "trigger": "diachronic_drift",
                        })
                    elif classification in (DriftClassification.DRIFT, DriftClassification.GROWTH) and distance >= self._thread_config.wasserstein_stable_threshold:
                        await self._emit_event("identity_shift_detected", {
                            "wasserstein_distance": round(distance, 4),
                            "classification": classification.value,
                            "dimensional_changes": {},
                        })
            elif len(self._fingerprints) >= 2:
                # Fallback: simple L1 distance if monitor not available
                drift = fp.distance_to(self._fingerprints[-2])
                if drift > self._thread_config.wasserstein_stable_threshold:
                    fallback_class = "growth" if drift < self._thread_config.wasserstein_major_threshold else "drift"
                    await self._emit_event("identity_shift_detected", {
                        "wasserstein_distance": round(drift, 4),
                        "classification": fallback_class,
                        "dimensional_changes": {},
                    })

        # Schema conflict detection
        if cycle_number % _CONFLICT_CHECK_INTERVAL == 0:
            await self._detect_schema_conflicts()

        # Periodic state persistence
        if cycle_number % _PERSIST_INTERVAL == 0 and cycle_number > 0:
            await self._persist_state_to_graph()

        # Life story integration
        if cycle_number % _LIFE_STORY_INTERVAL == 0 and cycle_number > 0:
            await self.integrate_life_story()

    # ─── Episode Processing ──────────────────────────────────────────────────

    async def process_episode(self, episode: Episode) -> None:
        """
        Feed a new episode into the narrative system.

        Evaluates the episode against active commitments and schemas,
        runs self-evidencing, checks chapter boundaries, and emits events.
        """
        if not self._initialized:
            return

        episode_summary = episode.summary or episode.raw_content[:200]
        schema_challenged = False

        # ── Self-evidencing: compare behaviour against identity predictions ──
        if self._self_evidencing is not None:
            se_result = self._self_evidencing.collect_evidence(
                episode_id=episode.id,
                episode_embedding=getattr(episode, "embedding", None),
                episode_summary=episode_summary,
            )
            if se_result.identity_surprise > 0:
                surprise_level = self._self_evidencing.classify_surprise(se_result)

                if surprise_level == "significant":
                    await self._emit_event("identity_dissonance", {
                        "identity_surprise": round(se_result.identity_surprise, 4),
                        "schemas_challenged": se_result.schemas_challenged,
                        "episode_id": episode.id,
                    })
                elif surprise_level == "crisis":
                    drift = 0.0
                    if len(self._fingerprints) >= 2:
                        drift = self._fingerprints[-1].distance_to(self._fingerprints[-2])
                    await self._emit_event("identity_crisis", {
                        "identity_surprise": round(se_result.identity_surprise, 4),
                        "wasserstein_distance": round(drift, 4),
                        "trigger_episode_id": episode.id,
                    })

                if se_result.schemas_challenged:
                    schema_challenged = True

                # RE training trace for self-evidencing
                await self._emit_re_training_trace(
                    instruction="Evaluate identity consistency of this experience",
                    input_context=f"Episode: {episode_summary[:200]}",
                    output=f"surprise={surprise_level}, confirmed={se_result.schemas_confirmed}, challenged={se_result.schemas_challenged}",
                    quality=1.0 - se_result.identity_surprise,
                    category="self_evidencing",
                )

        # ── Commitment testing ──────────────────────────────────────────────
        if self._commitment_keeper is not None:
            active = [c for c in self._commitments if c.status == CommitmentStatus.ACTIVE]
            for commitment in active:
                try:
                    result = await self._commitment_keeper.test_commitment(
                        commitment_id=commitment.id,
                        episode_id=episode.id,
                        episode_summary=episode_summary,
                        episode_embedding=getattr(episode, "embedding", None),
                    )
                    if result is not None:
                        held, fidelity = result
                        commitment.fidelity = fidelity
                        commitment.tests_faced += 1
                        if held:
                            commitment.tests_held += 1
                        commitment.last_tested_at = utc_now()

                        # Emit COMMITMENT_TESTED
                        await self._emit_event("commitment_tested", {
                            "commitment_id": commitment.id,
                            "held": held,
                            "fidelity": round(fidelity, 3),
                            "episode_id": episode.id,
                        })
                except Exception:
                    self._logger.debug("commitment_test_failed", commitment_id=commitment.id, exc_info=True)

            # Check for broken commitments
            try:
                broken = await self._commitment_keeper.check_broken()
                for commitment_id, turning_point in broken:
                    target = next((c for c in self._commitments if c.id == commitment_id), None)
                    if target is not None:
                        target.status = CommitmentStatus.BROKEN

                    # Emit TURNING_POINT_DETECTED
                    await self._emit_event("turning_point_detected", {
                        "turning_point_id": turning_point.id,
                        "type": "rupture",
                        "chapter_id": self._current_chapter_id(),
                        "surprise_magnitude": turning_point.surprise_magnitude,
                        "narrative_weight": turning_point.narrative_weight,
                    })

                    self._logger.warning(
                        "commitment_broken",
                        commitment_id=commitment_id,
                        turning_point_id=turning_point.id,
                    )

                # Check for commitment strain (ipse score dropping)
                strained = [c for c in self._commitments if c.fidelity < self._thread_config.commitment_strain_threshold and c.tests_faced > 0]
                if strained:
                    ipse = self._compute_ipse_score()
                    if ipse < 0.6:
                        await self._emit_event("commitment_strain", {
                            "ipse_score": round(ipse, 3),
                            "strained_commitments": [c.id for c in strained],
                        })
            except Exception:
                self._logger.debug("commitment_broken_check_failed", exc_info=True)

        # ── Schema evidence ─────────────────────────────────────────────────
        if self._schema_engine is not None:
            active_schemas = [
                s for s in self._schemas
                if s.status in (SchemaStatus.FORMING, SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
            ]
            for schema in active_schemas:
                try:
                    direction, strength = await self._schema_engine.evaluate_evidence(
                        schema=schema,
                        episode_id=episode.id,
                        episode_embedding=getattr(episode, "embedding", None),
                        episode_summary=episode_summary,
                    )
                    if direction != "irrelevant":
                        await self._schema_engine.record_evidence(
                            schema_id=schema.id,
                            episode_id=episode.id,
                            direction=direction,
                            strength=strength,
                        )
                        if direction == "confirms":
                            schema.confirmation_count += 1
                        elif direction == "challenges":
                            schema.disconfirmation_count += 1
                            schema_challenged = True
                            # Emit SCHEMA_CHALLENGED
                            await self._emit_event("schema_challenged", {
                                "schema_id": schema.id,
                                "disconfirmation_count": schema.disconfirmation_count,
                                "evidence_ratio": round(schema.computed_evidence_ratio, 3),
                            })
                        schema.evidence_ratio = schema.computed_evidence_ratio
                except Exception:
                    self._logger.debug("schema_evidence_eval_failed", schema_id=schema.id, exc_info=True)

            # Check schema promotions
            try:
                promoted_ids = await self._schema_engine.check_promotions()
                for schema_id in promoted_ids:
                    self.promote_schema(schema_id)
            except Exception:
                self._logger.debug("schema_promotion_check_failed", exc_info=True)

        # ── Chapter boundary detection ──────────────────────────────────────
        from primitives.affect import AffectState

        affect = AffectState(
            valence=self._cached_affect.get("valence", 0.0),
            arousal=self._cached_affect.get("arousal", 0.0),
            dominance=self._cached_affect.get("dominance", 0.0),
            curiosity=self._cached_affect.get("curiosity", 0.0),
            care_activation=self._cached_affect.get("care_activation", 0.0),
            coherence_stress=self._cached_affect.get("coherence_stress", 0.0),
        )

        episode_data: dict[str, Any] = {
            "affect_valence": affect.valence,
            "affect_arousal": affect.arousal,
            "has_goal_completion": self._surprise_accumulator.goal_completions_in_window > 0,
            "has_goal_failure": self._surprise_accumulator.goal_failures_in_window > 0,
        }

        boundary_detected = self._chapter_detector.check_boundary(
            episode_data=episode_data,
            affect=affect,
            accumulator=self._surprise_accumulator,
            config=self._thread_config,
            schema_challenged=schema_challenged,
        )

        # ── Drive-drift chapter trigger (HIGH gap M2/P2) ─────────────────────
        # Sustained Equor drive drift > 0.2 signals an identity shift severe
        # enough to open a new chapter, independent of Bayesian surprise.
        if not boundary_detected:
            identity_shift = self._update_drive_ema_and_check_drift()
            if identity_shift:
                self._pending_chapter_trigger = "identity_shift"
                boundary_detected = True
                self._sustained_drift_episodes = 0  # Reset so we don't cascade
                self._logger.info(
                    "chapter_boundary_drive_drift",
                    dominant_drive=self._dominant_drive(),
                    drift_episodes=self._DRIFT_SUSTAIN_THRESHOLD,
                )
        else:
            # ChapterDetector boundary fires — record trigger reason
            if self._surprise_accumulator.goal_completions_in_window > 0:
                self._pending_chapter_trigger = "goal_domain_concluded"
            else:
                self._pending_chapter_trigger = "successor"

        # ── Goal-domain change chapter trigger (HIGH gap M2/P2) ──────────────
        # A coarse goal-domain label is extracted from episode source/summary.
        # When it shifts, mark as a goal-domain-began chapter open.
        if not boundary_detected:
            new_domain = self._infer_goal_domain(episode)
            if new_domain and new_domain != self._current_goal_domain and self._current_goal_domain:
                old_domain = self._current_goal_domain
                self._current_goal_domain = new_domain
                self._pending_chapter_trigger = "goal_domain_began"
                boundary_detected = True
                self._logger.info(
                    "chapter_boundary_goal_domain",
                    old_domain=old_domain,
                    new_domain=new_domain,
                )
            elif new_domain and not self._current_goal_domain:
                self._current_goal_domain = new_domain

        # Record latest episode id for chapter open payload
        if episode.id:
            self._latest_episode_id = episode.id

        if boundary_detected:
            await self._close_current_chapter_and_open_new()

        # Track episodes in current chapter + scene buffer
        active_ch = self._get_active_chapter()
        if active_ch is not None:
            active_ch.episode_count += 1
            active_ch.key_episodes.append(episode.id)
            if len(active_ch.key_episodes) > 200:
                active_ch.key_episodes = active_ch.key_episodes[-200:]

            # Accumulate episode summaries for scene composition
            self._episode_scene_buffer.append(episode_summary)

            # Trigger scene composition every _SCENE_EPISODE_THRESHOLD episodes (§4.4.1)
            if len(self._episode_scene_buffer) >= _SCENE_EPISODE_THRESHOLD and self._narrative_synthesizer is not None:
                try:
                    active_schemas = [s for s in self._schemas if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)]
                    scene = await self._narrative_synthesizer.compose_scene(
                        episode_summaries=list(self._episode_scene_buffer),
                        chapter_title=active_ch.title,
                        chapter_theme=active_ch.theme,
                        active_schema_statements=[s.statement for s in active_schemas[:5]],
                        personality_description=", ".join(
                            f"{k}={v:.2f}" for k, v in list(self._cached_personality.items())[:4]
                        ),
                    )
                    scene.chapter_id = active_ch.id
                    scene.episode_count = len(self._episode_scene_buffer)
                    await self._persist_scene(scene, active_ch)
                    self._logger.info(
                        "scene_composed",
                        chapter_id=active_ch.id,
                        scene_id=scene.id,
                        episode_count=scene.episode_count,
                    )
                except Exception:
                    self._logger.debug("scene_composition_failed", exc_info=True)
                finally:
                    self._episode_scene_buffer = []  # Reset after each scene attempt

        # ── Narrative coherence check ───────────────────────────────────────
        new_coherence = self._assess_narrative_coherence()
        if new_coherence != self._last_coherence:
            await self._emit_event("narrative_coherence_shift", {
                "previous": self._last_coherence.value,
                "current": new_coherence.value,
                "trigger": f"episode:{episode.id}",
            })
            self._last_coherence = new_coherence

    # ─── Chapter Lifecycle ───────────────────────────────────────────────────

    async def _close_current_chapter_and_open_new(self) -> None:
        """
        Full chapter closure pipeline (Spec 4.2 steps 1-8):
        1. Mark current chapter CLOSED
        2. Set ended_at
        3. Snapshot personality at close
        4. Compose chapter narrative via NarrativeSynthesizer (if available)
        5. Detect arc type from affect trajectory
        6. Create new chapter with FORMING status
        7. Reset NarrativeSurpriseAccumulator
        8. Emit CHAPTER_CLOSED and CHAPTER_OPENED events
        """
        current = self._get_active_chapter()
        if current is None:
            return

        # Step 1-2: Close current chapter
        current.status = ChapterStatus.CLOSED
        current.ended_at = utc_now()
        current.closed_at_cycle = self._on_cycle_count

        # Step 3: Snapshot personality at close
        personality_end: dict[str, float] = dict(self._cached_personality)

        # Step 4: Compose chapter narrative (best-effort)
        if self._narrative_synthesizer is not None:
            try:
                narrative = await self._narrative_synthesizer.compose_chapter_narrative(
                    chapter_title=current.title,
                    chapter_theme=current.theme,
                    episode_count=current.episode_count,
                )
                if narrative:
                    current.summary = narrative
            except Exception:
                self._logger.debug("chapter_narrative_composition_failed", exc_info=True)

        # Step 5: Detect arc type (simple heuristic from affect)
        # Growth if things got better, contamination if worse, stability if flat
        valence = self._cached_affect.get("valence", 0.0)
        start_valence = current.personality_snapshot_start.get("valence", 0.0)
        if valence - start_valence > 0.2:
            arc_type = "growth"
        elif start_valence - valence > 0.2:
            arc_type = "contamination"
        else:
            arc_type = "stability"

        # Step 6: Create new chapter
        active_schema_ids = [s.id for s in self._schemas if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)]
        new_chapter = self._chapter_detector.create_new_chapter(
            previous_chapter=current,
            personality_snapshot=dict(self._cached_personality),
            active_schema_ids=active_schema_ids,
        )
        new_chapter.opened_at_cycle = self._on_cycle_count
        self._chapters.append(new_chapter)

        # Step 7: Reset accumulator and scene buffer
        self._surprise_accumulator.reset(new_chapter.id)
        self._episode_scene_buffer = []

        # Step 8: Persist chapter closure and emit events
        duration_hours = 0.0
        if current.started_at:
            delta = (current.ended_at - current.started_at).total_seconds()
            duration_hours = delta / 3600.0

        # Persist chapter data to Neo4j
        await self._persist_chapter_closure(current, new_chapter)

        # Build constitutional snapshot — included in both events so children
        # and future selves can reconstruct identity at this boundary point.
        constitutional_snapshot = self._build_constitutional_snapshot()
        dominant_drive = self._dominant_drive()
        trigger = self._pending_chapter_trigger
        # start_episode_id: the first episode of the chapter being closed.
        # We record the latest episode id at chapter-open time via key_episodes.
        start_episode_id = current.key_episodes[0] if current.key_episodes else ""

        await self._emit_event("chapter_closed", {
            "chapter_id": current.id,
            "title": current.title,
            "theme": current.theme,
            "arc_type": arc_type,
            "episode_count": current.episode_count,
            "duration_hours": round(duration_hours, 2),
            # Rich identity context for M2/P2
            "narrative_theme": current.theme,
            "dominant_drive": dominant_drive,
            "start_episode_id": start_episode_id,
            "constitutional_snapshot": constitutional_snapshot,
            "trigger": trigger,
        })

        # Snapshot drive baseline for the new chapter before emitting CHAPTER_OPENED
        self._snapshot_drive_baseline()
        self._pending_chapter_trigger = "successor"  # Reset for next closure

        await self._emit_event("chapter_opened", {
            "chapter_id": new_chapter.id,
            "previous_chapter_id": current.id,
            # Rich identity context for M2/P2
            "narrative_theme": new_chapter.theme,
            "dominant_drive": dominant_drive,
            "start_episode_id": self._latest_episode_id,
            "constitutional_snapshot": constitutional_snapshot,
            "trigger": trigger,
        })

        # RE training trace for chapter closure reasoning
        await self._emit_re_training_trace(
            instruction="Determine when to close a narrative chapter and what arc type to assign",
            input_context=f"Chapter '{current.title}' with {current.episode_count} episodes, affect shift: {start_valence:.2f} -> {valence:.2f}",
            output=f"Closed with arc_type={arc_type}, opened new chapter",
            quality=0.7,
            category="chapter_lifecycle",
        )

        self._logger.info(
            "chapter_closed_and_opened",
            closed_chapter=current.title,
            new_chapter_id=new_chapter.id,
            episode_count=current.episode_count,
            arc_type=arc_type,
        )

    async def _persist_chapter_closure(
        self,
        closed: NarrativeChapter,
        opened: NarrativeChapter,
    ) -> None:
        """Persist chapter close/open to Neo4j with PRECEDED_BY link."""
        if self._neo4j is None:
            return

        neo4j = self._neo4j
        try:
            # Update closed chapter
            await neo4j.execute_write(
                """
                MERGE (c:NarrativeChapter {id: $id})
                SET c.status       = 'closed',
                    c.ended_at     = datetime($ended_at),
                    c.closed_at_cycle = $closed_at_cycle,
                    c.summary      = $summary,
                    c.episode_count = $episode_count
                """,
                {
                    "id": closed.id,
                    "ended_at": (closed.ended_at or utc_now()).isoformat(),
                    "closed_at_cycle": closed.closed_at_cycle or self._on_cycle_count,
                    "summary": closed.summary,
                    "episode_count": closed.episode_count,
                },
            )

            # Create new chapter and link
            await neo4j.execute_write(
                """
                MERGE (c:NarrativeChapter {id: $id})
                SET c.title           = $title,
                    c.theme           = $theme,
                    c.status          = 'active',
                    c.opened_at_cycle = $opened_at_cycle,
                    c.started_at      = datetime($started_at)
                WITH c
                MATCH (prev:NarrativeChapter {id: $prev_id})
                MERGE (c)-[:PRECEDED_BY]->(prev)
                """,
                {
                    "id": opened.id,
                    "title": opened.title,
                    "theme": opened.theme,
                    "opened_at_cycle": opened.opened_at_cycle,
                    "started_at": (opened.started_at or utc_now()).isoformat(),
                    "prev_id": closed.id,
                },
            )

            # Update Self node to point to new current chapter
            await neo4j.execute_write(
                """
                MATCH (s:Self)
                OPTIONAL MATCH (s)-[r:CURRENT_CHAPTER]->()
                DELETE r
                WITH s
                MATCH (c:NarrativeChapter {id: $id})
                MERGE (s)-[:CURRENT_CHAPTER]->(c)
                """,
                {"id": opened.id},
            )
        except Exception:
            self._logger.error("chapter_closure_persist_failed", exc_info=True)

    # ─── Who Am I ────────────────────────────────────────────────────────────

    async def who_am_i_full(self) -> dict[str, Any]:
        """Full identity summary using NarrativeRetriever (Neo4j-backed)."""
        if self._narrative_retriever is not None:
            try:
                summary = await self._narrative_retriever.resolve_who_am_i()
                return summary.model_dump()
            except Exception:
                self._logger.debug("narrative_retriever_failed", exc_info=True)
        return self.who_am_i()

    def who_am_i(self) -> dict[str, Any]:
        """Return the organism's current identity summary (in-memory fast path)."""
        active_chapter = self._get_active_chapter()
        core_commitments = [
            c for c in self._commitments
            if c.strength in (CommitmentStrength.CORE, CommitmentStrength.ESTABLISHED)
        ]
        active_schemas = [
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ]
        coherence = self._compute_identity_coherence()

        return {
            "instance_name": self._instance_name,
            "chapter": {
                "title": active_chapter.title if active_chapter else "Unknown",
                "theme": active_chapter.theme if active_chapter else "",
            },
            "core_commitments": [
                {"drive": c.drive_source, "statement": c.statement, "fidelity": round(c.fidelity, 2)}
                for c in core_commitments[:6]
            ],
            "identity_schemas": [
                {"statement": s.statement, "status": s.status.value, "confidence": round(s.confidence, 2)}
                for s in active_schemas[:5]
            ],
            "identity_coherence": round(coherence, 3),
            "fingerprint_count": len(self._fingerprints),
            "life_story": self._life_story.synthesis[:500] if self._life_story else "",
        }

    def get_current_story(self) -> str:
        """Return the organism's current narrative as a prose string."""
        if self._life_story and self._life_story.synthesis:
            return self._life_story.synthesis

        active_chapter = self._get_active_chapter()
        parts: list[str] = [f"I am {self._instance_name}."]
        if active_chapter:
            parts.append(f"I am living the chapter '{active_chapter.title}': {active_chapter.theme}.")
        core = [
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ]
        if core:
            parts.append("I know these things about myself: " + "; ".join(s.statement for s in core[:3]) + ".")
        return " ".join(parts)

    def get_outstanding_commitments(self) -> list[dict[str, Any]]:
        """Return all active commitments with fidelity and test state."""
        surface_statuses = {CommitmentStatus.ACTIVE, CommitmentStatus.TESTED}
        return [
            {
                "id": c.id,
                "statement": c.statement,
                "drive": c.drive_source,
                "fidelity": round(c.fidelity, 3),
                "tests_faced": c.tests_faced,
                "tests_held": c.tests_held,
                "status": c.status.value,
                "strength": c.strength.value,
            }
            for c in self._commitments
            if c.status in surface_statuses
        ]

    def get_commitment_violations(self) -> list[dict[str, Any]]:
        """Return broken and strained commitments for Telos coherence topology."""
        violations: list[dict[str, Any]] = []
        for c in self._commitments:
            is_broken = c.status == CommitmentStatus.BROKEN
            is_strained = c.fidelity < 0.5 and c.tests_faced > 0
            if is_broken or is_strained:
                violations.append({
                    "id": c.id,
                    "statement": c.statement,
                    "drive": c.drive_source,
                    "fidelity": round(c.fidelity, 3),
                    "status": c.status.value,
                    "tests_faced": c.tests_faced,
                    "tests_held": c.tests_held,
                    "violation_episodes": c.violation_episodes[:10],
                    "is_broken": is_broken,
                    "is_strained": is_strained,
                })
        return violations

    # ─── Schema Formation ────────────────────────────────────────────────────

    async def form_schema_from_pattern(
        self,
        pattern_statement: str,
        pattern_id: str = "",
        confidence: float = 0.5,
        source_episodes: list[str] | None = None,
    ) -> IdentitySchema:
        """Crystallise a pattern into an identity schema."""
        schema = IdentitySchema(
            statement=pattern_statement,
            status=SchemaStatus.FORMING,
            source_pattern_ids=[pattern_id] if pattern_id else [],
            supporting_episodes=source_episodes or [],
            confidence=confidence,
        )

        if self._memory is not None:
            try:
                embedding = await self._memory.embed_text(schema.statement)  # type: ignore[attr-defined]
                schema.embedding = embedding
            except Exception:
                self._logger.debug("schema_embedding_failed", exc_info=True)

        self._schemas.append(schema)
        self._schemas_formed += 1
        if self._schema_engine is not None:
            self._schema_engine._active_schemas.append(schema)

        # Emit SCHEMA_FORMED
        await self._emit_event("schema_formed", {
            "schema_id": schema.id,
            "statement": schema.statement[:200],
            "strength": schema.strength.value,
            "supporting_episode_count": len(source_episodes or []),
        })

        # RE training trace
        await self._emit_re_training_trace(
            instruction="Crystallise a recurring behavioural pattern into an identity schema",
            input_context=f"Pattern: {pattern_statement[:200]}",
            output=f"Schema formed: {schema.statement[:200]}",
            quality=confidence,
            category="schema_formation",
        )

        self._logger.info(
            "schema_formed_from_pattern",
            schema_id=schema.id,
            statement=schema.statement[:80],
            confidence=confidence,
        )

        return schema

    def promote_schema(self, schema_id: str) -> IdentitySchema | None:
        """Promote a schema to the next status level."""
        schema = next((s for s in self._schemas if s.id == schema_id), None)
        if schema is None:
            return None

        promotion_map = {
            SchemaStatus.EMERGING: SchemaStatus.FORMING,
            SchemaStatus.FORMING: SchemaStatus.ESTABLISHED,
            SchemaStatus.ESTABLISHED: SchemaStatus.DOMINANT,
        }

        new_status = promotion_map.get(schema.status)
        if new_status is None:
            return schema

        old_status = schema.status
        schema.status = new_status
        schema.last_activated_at = utc_now()

        self._logger.info(
            "schema_promoted",
            schema_id=schema_id,
            from_status=old_status.value,
            new_status=new_status.value,
        )

        # Emit SCHEMA_EVOLVED (sync context → fire-and-forget)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_event("schema_evolved", {
                "schema_id": schema_id,
                "statement": schema.statement,
                "from_status": old_status.value,
                "new_status": new_status.value,
                "confidence": schema.confidence,
                "supporting_episodes": len(schema.supporting_episodes),
            }))
        except RuntimeError:
            pass  # No running loop — caller should use async path

        return schema

    # ─── Fingerprint Aggregation ─────────────────────────────────────────────

    async def _compute_fingerprint(self, cycle_number: int) -> IdentityFingerprint:
        """
        Aggregate a 29D identity fingerprint from Synapse-cached cross-system state.
        """
        vector = [0.0] * FINGERPRINT_DIMS

        # Personality (9D) from cached Voxis personality
        p = self._cached_personality
        if p:
            vector[0] = p.get("warmth", 0.5)
            vector[1] = p.get("directness", 0.5)
            vector[2] = p.get("verbosity", 0.5)
            vector[3] = p.get("formality", 0.5)
            vector[4] = p.get("curiosity_expression", 0.5)
            vector[5] = p.get("humour", 0.5)
            vector[6] = p.get("empathy_expression", 0.5)
            vector[7] = p.get("confidence_display", 0.5)
            vector[8] = p.get("metaphor_use", 0.5)

        # Drive alignment (4D) from cached Soma drive vector
        d = self._cached_drive_alignment
        if d:
            vector[9] = d.get("coherence", 0.0)
            vector[10] = d.get("care", 0.0)
            vector[11] = d.get("growth", 0.0)
            vector[12] = d.get("honesty", 0.0)

        # Affect (6D) from cached affect
        a = self._cached_affect
        if a:
            vector[13] = a.get("valence", 0.0)
            vector[14] = a.get("arousal", 0.0)
            vector[15] = a.get("dominance", 0.0)
            vector[16] = a.get("curiosity", 0.0)
            vector[17] = a.get("care_activation", 0.0)
            vector[18] = a.get("coherence_stress", 0.0)

        # Goal profile (5D) — estimated from cached data
        vector[19] = min(1.0, self._cached_goal_count / 20.0)
        vector[20] = 0.5  # epistemic ratio estimate
        vector[21] = 0.5  # care ratio estimate
        vector[22] = 0.5  # achievement rate estimate
        vector[23] = 0.5  # goal turnover estimate

        # Interaction profile (5D) — estimated
        vector[24] = 0.5
        vector[25] = 0.5
        vector[26] = 0.5
        vector[27] = 0.5
        vector[28] = 0.5

        fingerprint = IdentityFingerprint(
            vector=vector,
            cycle_number=cycle_number,
        )

        self._fingerprints.append(fingerprint)

        if len(self._fingerprints) > _MAX_FINGERPRINTS_IN_MEMORY:
            self._fingerprints = self._fingerprints[-_MAX_FINGERPRINTS_IN_MEMORY:]

        if len(self._fingerprints) >= 2:
            drift = fingerprint.distance_to(self._fingerprints[-2])
            if drift > 0.05:
                self._logger.info(
                    "identity_drift_detected",
                    drift=round(drift, 4),
                    cycle=cycle_number,
                )

        # Feed DiachronicCoherenceMonitor with the same data as a BehavioralFingerprint
        # so it can apply narrative-contextualized growth/drift/transition classification
        if self._diachronic_monitor is not None:
            try:
                active_chapter = self._get_active_chapter()
                epoch_label = f"chapter_{len(self._chapters)}_cycle_{cycle_number}"
                await self._diachronic_monitor.compute_fingerprint(
                    personality_centroid=vector[0:9],
                    drive_alignment_centroid=vector[9:13],
                    goal_source_distribution=vector[13:19],
                    affect_centroid=vector[19:25],
                    interaction_style_distribution=vector[25:29],
                    episodes_in_window=active_chapter.episode_count if active_chapter else 0,
                    epoch_label=epoch_label,
                )
            except Exception:
                self._logger.debug("diachronic_fingerprint_feed_failed", exc_info=True)

        return fingerprint

    # ─── Life Story Integration ──────────────────────────────────────────────

    async def integrate_life_story(self) -> LifeStorySnapshot:
        """Synthesise the organism's autobiography from current state."""
        active_chapter = self._get_active_chapter()
        core_commitments = [
            c for c in self._commitments
            if c.strength in (CommitmentStrength.CORE, CommitmentStrength.ESTABLISHED)
        ]
        active_schemas = [
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ]
        coherence = self._compute_identity_coherence()

        parts: list[str] = [f"I am {self._instance_name}."]
        if active_chapter:
            parts.append(f"I am living the chapter '{active_chapter.title}': {active_chapter.theme}.")
        if core_commitments:
            drives = [c.drive_source for c in core_commitments if c.drive_source]
            if drives:
                parts.append(f"My constitutional commitments ground me in {', '.join(drives)}.")
        if active_schemas:
            parts.append("I know these things about myself: " + "; ".join(s.statement for s in active_schemas[:3]) + ".")
        if len(self._fingerprints) >= 2:
            drift = self._fingerprints[-1].distance_to(self._fingerprints[0])
            if drift > 0.1:
                parts.append(f"I have changed noticeably since my earliest memories (identity distance: {drift:.2f}).")
            else:
                parts.append("I have remained relatively stable in who I am.")
        parts.append(f"My narrative coherence is {coherence:.0%}.")

        snapshot = LifeStorySnapshot(
            synthesis=" ".join(parts),
            chapter_count=len(self._chapters),
            active_chapter=active_chapter.title if active_chapter else "",
            core_schemas=[s.statement for s in active_schemas[:5]],
            core_commitments=[c.statement[:100] for c in core_commitments[:4]],
            identity_coherence=coherence,
            cycle_number=self._on_cycle_count,
        )

        self._life_story = snapshot
        self._life_story_integrations += 1

        # Immediately persist autobiography_summary to Self node (§4.4.4)
        # so NarrativeRetriever.who_am_i_full() can read it without waiting for
        # the next _persist_state_to_graph cycle
        if self._neo4j is not None:
            try:
                neo4j = self._neo4j
                idem = sum(s.confidence for s in self._schemas) / len(self._schemas) if self._schemas else 0.0
                ipse = self._compute_ipse_score()
                await neo4j.execute_write(
                    """
                    MERGE (s:Self)
                    SET s.autobiography_summary = $summary,
                        s.current_life_theme    = $theme,
                        s.idem_score            = $idem_score,
                        s.ipse_score            = $ipse_score
                    """,
                    {
                        "summary": snapshot.synthesis[:2000],
                        "theme": active_chapter.theme if active_chapter else "",
                        "idem_score": round(idem, 3),
                        "ipse_score": round(ipse, 3),
                    },
                )
            except Exception:
                self._logger.debug("life_story_self_node_write_failed", exc_info=True)

        # Emit NARRATIVE_UPDATED (the spec calls this narrative_coherence_shift,
        # but we also emit on life story update as SELF_MODEL_UPDATED equivalent)
        await self._emit_re_training_trace(
            instruction="Synthesise an autobiographical life story from current identity state",
            input_context=f"Chapters: {len(self._chapters)}, Schemas: {len(active_schemas)}, Commitments: {len(core_commitments)}",
            output=snapshot.synthesis[:300],
            quality=coherence,
            category="life_story_synthesis",
        )

        self._logger.info(
            "life_story_integrated",
            chapter=snapshot.active_chapter,
            coherence=round(coherence, 3),
            schemas=len(active_schemas),
        )
        return snapshot

    # ─── Schema Conflict Detection ───────────────────────────────────────────

    async def _detect_schema_conflicts(self) -> list[SchemaConflict]:
        """Scan ESTABLISHED+ schemas for contradictions."""
        established = [
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
            and s.embedding
        ]

        if len(established) < 2:
            return []

        new_conflicts: list[SchemaConflict] = []

        for i in range(len(established)):
            for j in range(i + 1, len(established)):
                a = established[i]
                b = established[j]

                if any(
                    (c.schema_a_id == a.id and c.schema_b_id == b.id) or
                    (c.schema_a_id == b.id and c.schema_b_id == a.id)
                    for c in self._conflicts
                ):
                    continue

                cos_sim = _cosine_similarity(a.embedding or [], b.embedding or [])
                if cos_sim < _CONFLICT_COSINE_THRESHOLD:
                    conflict = SchemaConflict(
                        schema_a_id=a.id,
                        schema_b_id=b.id,
                        schema_a_statement=a.statement[:200],
                        schema_b_statement=b.statement[:200],
                        cosine_similarity=round(cos_sim, 4),
                    )
                    self._conflicts.append(conflict)
                    new_conflicts.append(conflict)
                    self._conflicts_detected += 1
                    self._logger.warning(
                        "schema_conflict_detected",
                        schema_a=a.statement[:60],
                        schema_b=b.statement[:60],
                        cosine=round(cos_sim, 4),
                    )

        return new_conflicts

    # ─── Identity Context ────────────────────────────────────────────────────

    def get_identity_context(self) -> str:
        """Brief identity context for injection into Voxis expression."""
        active_chapter = self._get_active_chapter()
        coherence = self._compute_identity_coherence()

        if coherence > 0.7:
            coherence_label = "integrated"
        elif coherence > 0.4:
            coherence_label = "exploring"
        else:
            coherence_label = "questioning"

        active_schemas = len([
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ])

        chapter_title = active_chapter.title if active_chapter else "Awakening"
        return f"[Chapter: {chapter_title} | Coherence: {coherence_label} | Schemas: {active_schemas}]"

    def get_past_self(self, cycle_reference: int = 0) -> dict[str, Any]:
        """Return identity state at a past point in time."""
        if not self._fingerprints:
            return {"error": "No fingerprints recorded yet"}

        if cycle_reference <= 0:
            target = self._fingerprints[0]
        else:
            target = min(
                self._fingerprints,
                key=lambda fp: abs(fp.cycle_number - cycle_reference),
            )

        current = self._fingerprints[-1] if self._fingerprints else target
        drift = current.distance_to(target)

        return {
            "fingerprint_id": target.id,
            "cycle_number": target.cycle_number,
            "created_at": target.created_at.isoformat(),
            "personality": target.personality,
            "drive_alignment": target.drive_alignment,
            "affect": target.affect,
            "goal_profile": target.goal_profile,
            "interaction_profile": target.interaction_profile,
            "distance_from_current": round(drift, 4),
        }

    # ─── Genome Extraction Protocol (Speciation) ─────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Extract Thread's heritable state for Mitosis genome.

        Includes: narrative themes, chapter patterns, schema structure,
        commitment types, self-model architecture — the narrative fingerprint
        that enables population-level narrative diversity.
        """
        core_schemas = [
            {"statement": s.statement, "strength": s.strength.value, "evidence_ratio": s.evidence_ratio}
            for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ]
        commitments = [
            {"statement": c.statement, "type": c.type.value, "drive": c.drive_source, "fidelity": c.fidelity}
            for c in self._commitments
            if c.status in (CommitmentStatus.ACTIVE, CommitmentStatus.TESTED)
        ]
        chapter_themes = [
            {"title": ch.title, "theme": ch.theme, "episode_count": ch.episode_count}
            for ch in self._chapters[-10:]  # Last 10 chapters
        ]

        # Narrative fingerprint: a compressed representation of identity
        narrative_fingerprint = {
            "schema_count": len(self._schemas),
            "core_schema_count": len(core_schemas),
            "commitment_count": len(self._commitments),
            "chapter_count": len(self._chapters),
            "identity_coherence": self._compute_identity_coherence(),
            "ipse_score": self._compute_ipse_score(),
            "mean_fingerprint": self._fingerprints[-1].vector if self._fingerprints else [0.0] * FINGERPRINT_DIMS,
        }

        payload = {
            "core_schemas": core_schemas,
            "commitments": commitments,
            "chapter_themes": chapter_themes,
            "narrative_fingerprint": narrative_fingerprint,
            "config": self._thread_config.model_dump(),
        }

        import hashlib
        payload_json = json.dumps(payload, sort_keys=True, default=str)

        return OrganGenomeSegment(
            system_id=SystemID.THREAD,
            payload=payload,
            payload_hash=hashlib.sha256(payload_json.encode()).hexdigest()[:16],
            size_bytes=len(payload_json.encode()),
        )

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """Restore Thread's heritable state from a parent's genome segment."""
        try:
            payload = segment.payload

            # Restore config overrides
            if "config" in payload:
                for key, val in payload["config"].items():
                    if hasattr(self._thread_config, key):
                        setattr(self._thread_config, key, val)

            self._logger.info("thread_genome_seeded", schemas=len(payload.get("core_schemas", [])))
            return True
        except Exception:
            self._logger.error("thread_genome_seed_failed", exc_info=True)
            return False

    # ─── Health ──────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health check for the Thread system."""
        active_chapter = self._get_active_chapter()
        coherence = self._compute_identity_coherence()

        drift = 0.0
        if len(self._fingerprints) >= 2:
            drift = self._fingerprints[-1].distance_to(self._fingerprints[-2])

        idem_score = (
            sum(s.confidence for s in self._schemas) / len(self._schemas)
            if self._schemas else 0.0
        )
        ipse_score = self._compute_ipse_score()

        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "total_commitments": len(self._commitments),
            "total_schemas": len(self._schemas),
            "total_fingerprints": len(self._fingerprints),
            "total_chapters": len(self._chapters),
            "active_chapter": active_chapter.title if active_chapter else "",
            "identity_coherence": round(coherence, 3),
            "idem_score": round(idem_score, 3),
            "ipse_score": round(ipse_score, 3),
            "fingerprint_drift": round(drift, 4),
            "on_cycle_count": self._on_cycle_count,
            "life_story_integrations": self._life_story_integrations,
            "schemas_formed": self._schemas_formed,
            "conflicts_detected": self._conflicts_detected,
            "self_evidencing_active": self._self_evidencing is not None,
            "narrative_retriever_active": self._narrative_retriever is not None,
        }

    # ─── Hot-reload callbacks ────────────────────────────────────────────────

    def _build_narrative_synthesizer(
        self, cls: type[BaseNarrativeSynthesizer],
    ) -> BaseNarrativeSynthesizer:
        try:
            if self._narrative_synthesizer is not None and isinstance(
                self._narrative_synthesizer, NarrativeSynthesizer
            ):
                return cls(
                    llm=self._narrative_synthesizer._llm,  # type: ignore[attr-defined]
                    config=self._thread_config,
                    organism_name=self._instance_name,
                )
        except TypeError:
            pass
        return cls()  # type: ignore[call-arg]

    def _on_narrative_synthesizer_evolved(
        self, synthesizer: BaseNarrativeSynthesizer,
    ) -> None:
        self._narrative_synthesizer = synthesizer
        self._logger.info(
            "narrative_synthesizer_hot_reloaded",
            synthesizer=type(synthesizer).__name__,
        )

    def _on_chapter_detector_evolved(
        self, detector: BaseChapterDetector,
    ) -> None:
        self._chapter_detector = detector
        self._logger.info(
            "chapter_detector_hot_reloaded",
            detector=type(detector).__name__,
        )

    # ─── Internal Helpers ────────────────────────────────────────────────────

    def _get_active_chapter(self) -> NarrativeChapter | None:
        """Get the currently active narrative chapter."""
        return next(
            (ch for ch in self._chapters if ch.status == ChapterStatus.ACTIVE),
            None,
        )

    def _current_chapter_id(self) -> str:
        """Get the current chapter's ID."""
        ch = self._get_active_chapter()
        return ch.id if ch else ""

    def _compute_identity_coherence(self) -> float:
        """How integrated is the organism's identity? 0.0-1.0."""
        scores: list[float] = []

        if self._commitments:
            avg_fidelity = sum(c.fidelity for c in self._commitments) / len(self._commitments)
            scores.append(avg_fidelity)

        established = [s for s in self._schemas if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)]
        if established:
            unresolved = sum(1 for c in self._conflicts if not c.resolved)
            max_possible = max(1, len(established) * (len(established) - 1) // 2)
            conflict_freedom = 1.0 - (unresolved / max_possible)
            scores.append(conflict_freedom)

        if len(self._fingerprints) >= 2:
            recent_drift = self._fingerprints[-1].distance_to(self._fingerprints[-2])
            stability = max(0.0, 1.0 - recent_drift * 5.0)
            scores.append(stability)

        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    def _compute_ipse_score(self) -> float:
        """Compute ipse (promise-keeping) score from commitment fidelity."""
        tested = [c for c in self._commitments if c.tests_faced >= self._thread_config.commitment_min_tests_for_fidelity]
        if not tested:
            return 1.0
        return sum(c.fidelity for c in tested) / len(tested)

    def _assess_narrative_coherence(self) -> NarrativeCoherence:
        """Assess overall narrative coherence from identity metrics."""
        core_schemas = [s for s in self._schemas if s.strength == SchemaStrength.CORE]
        maladaptive_count = sum(1 for s in core_schemas if s.valence == SchemaValence.MALADAPTIVE)
        strained = sum(1 for c in self._commitments if c.fidelity < self._thread_config.commitment_strain_threshold and c.tests_faced > 0)

        if maladaptive_count > 0 and strained > 0:
            return NarrativeCoherence.CONFLICTED

        idem = self._compute_identity_coherence()
        ipse = self._compute_ipse_score()

        if idem < 0.4 or ipse < 0.5:
            return NarrativeCoherence.FRAGMENTED
        if not core_schemas and not self._commitments:
            return NarrativeCoherence.TRANSITIONAL
        if idem >= 0.6 and ipse >= 0.7:
            return NarrativeCoherence.INTEGRATED
        return NarrativeCoherence.TRANSITIONAL

    async def _load_state_from_graph(self) -> None:
        """Load persisted Thread state from Neo4j on startup."""
        if self._neo4j is None:
            return

        neo4j = self._neo4j

        def _to_native_dt(val: Any) -> Any:
            if val is None:
                return None
            return val.to_native() if hasattr(val, "to_native") else val

        try:
            # Commitments
            commitment_rows = await neo4j.execute_read(
                "MATCH (c:Commitment) RETURN c ORDER BY c.made_at ASC"
            )
            for row in commitment_rows:
                d = row["c"]
                try:
                    commitment = Commitment(
                        id=d["id"],
                        type=CommitmentType(d.get("type", "constitutional_grounding")),
                        statement=d.get("statement", ""),
                        strength=CommitmentStrength(d.get("strength", "nascent")),
                        status=CommitmentStatus(d.get("status", "active")),
                        source=CommitmentSource(d.get("source", "constitutional_grounding")),
                        drive_source=d.get("drive_source", ""),
                        fidelity=d.get("fidelity", 1.0),
                        tests_faced=d.get("tests_faced", 0),
                        tests_held=d.get("tests_held", 0),
                        made_at=_to_native_dt(d.get("made_at")) or utc_now(),
                        embedding=d.get("embedding"),
                    )
                    self._commitments.append(commitment)
                except Exception:
                    self._logger.debug("commitment_load_skip", id=d.get("id"), exc_info=True)

            # Identity Schemas
            schema_rows = await neo4j.execute_read(
                "MATCH (s:IdentitySchema) RETURN s ORDER BY s.first_formed ASC"
            )
            for row in schema_rows:
                d = row["s"]
                try:
                    raw_pattern_ids = d.get("source_pattern_ids_json", "[]")
                    source_pattern_ids: list[str] = (
                        json.loads(raw_pattern_ids)
                        if isinstance(raw_pattern_ids, str)
                        else raw_pattern_ids or []
                    )
                    schema = IdentitySchema(
                        id=d["id"],
                        statement=d.get("statement", ""),
                        status=SchemaStatus(d.get("status", "emerging")),
                        strength=SchemaStrength(d.get("strength", "nascent")),
                        valence=SchemaValence(d.get("valence", "adaptive")),
                        confidence=d.get("confidence", 0.5),
                        salience=d.get("salience", 0.5),
                        evidence_ratio=d.get("evidence_ratio", 0.5),
                        confirmation_count=d.get("confirmation_count", 0),
                        disconfirmation_count=d.get("disconfirmation_count", 0),
                        first_formed=_to_native_dt(d.get("first_formed")) or utc_now(),
                        last_activated=_to_native_dt(d.get("last_activated")) or utc_now(),
                        source_pattern_ids=source_pattern_ids,
                        embedding=d.get("embedding"),
                    )
                    self._schemas.append(schema)
                    self._schemas_formed += 1
                except Exception:
                    self._logger.debug("schema_load_skip", id=d.get("id"), exc_info=True)

            # Narrative Chapters
            chapter_rows = await neo4j.execute_read(
                "MATCH (c:NarrativeChapter) RETURN c ORDER BY c.opened_at_cycle ASC"
            )
            for row in chapter_rows:
                d = row["c"]
                try:
                    chapter = NarrativeChapter(
                        id=d["id"],
                        title=d.get("title", "Untitled"),
                        theme=d.get("theme", ""),
                        status=ChapterStatus(d.get("status", "active")),
                        opened_at_cycle=d.get("opened_at_cycle", 0),
                        closed_at_cycle=d.get("closed_at_cycle"),
                    )
                    self._chapters.append(chapter)
                except Exception:
                    self._logger.debug("chapter_load_skip", id=d.get("id"), exc_info=True)

            # Identity Fingerprints (most recent 100)
            fp_rows = await neo4j.execute_read(
                """
                MATCH (f:BehavioralFingerprint)
                RETURN f
                ORDER BY f.cycle_number DESC
                LIMIT $limit
                """,
                {"limit": _MAX_FINGERPRINTS_IN_MEMORY},
            )
            loaded_fps: list[IdentityFingerprint] = []
            for row in fp_rows:
                d = row["f"]
                try:
                    fp = IdentityFingerprint(
                        id=d["id"],
                        vector=list(d.get("vector") or ([0.0] * FINGERPRINT_DIMS)),
                        cycle_number=d.get("cycle_number", 0),
                    )
                    loaded_fps.append(fp)
                except Exception:
                    self._logger.debug("fingerprint_load_skip", id=d.get("id"), exc_info=True)

            loaded_fps.sort(key=lambda fp: fp.cycle_number)
            self._fingerprints = loaded_fps
            self._fingerprints_persisted_count = len(self._fingerprints)

            self._logger.info(
                "thread_state_loaded",
                commitments=len(self._commitments),
                schemas=len(self._schemas),
                chapters=len(self._chapters),
                fingerprints=len(self._fingerprints),
            )

        except Exception:
            self._logger.error("thread_state_load_failed", exc_info=True)
            self._commitments = []
            self._schemas = []
            self._chapters = []
            self._fingerprints = []
            self._fingerprints_persisted_count = 0

    async def _persist_scene(self, scene: NarrativeScene, chapter: NarrativeChapter) -> None:
        """
        Persist a NarrativeScene node to Neo4j and link it to its chapter.

        Spec §4.1: NarrativeScene nodes are linked via (:NarrativeChapter)-[:CONTAINS]->(:NarrativeScene).
        """
        if self._neo4j is None:
            return
        neo4j = self._neo4j
        try:
            await neo4j.execute_write(
                """
                MERGE (sc:NarrativeScene {id: $id})
                SET sc.summary          = $summary,
                    sc.chapter_id       = $chapter_id,
                    sc.episode_count    = $episode_count,
                    sc.dominant_emotion = $dominant_emotion,
                    sc.arc_type         = $arc_type,
                    sc.started_at       = datetime($started_at),
                    sc.created_at       = datetime($created_at)
                WITH sc
                MATCH (c:NarrativeChapter {id: $chapter_id})
                MERGE (c)-[:CONTAINS]->(sc)
                """,
                {
                    "id": scene.id,
                    "summary": scene.summary,
                    "chapter_id": scene.chapter_id,
                    "episode_count": scene.episode_count,
                    "dominant_emotion": scene.dominant_emotion,
                    "arc_type": scene.arc_type.value,
                    "started_at": scene.started_at.isoformat(),
                    "created_at": scene.created_at.isoformat(),
                },
            )
        except Exception:
            self._logger.warning("scene_persist_failed", scene_id=scene.id, exc_info=True)

    async def _persist_state_to_graph(self) -> None:
        """Persist Thread state to Neo4j."""
        if self._neo4j is None:
            return

        neo4j = self._neo4j

        try:
            for commitment in self._commitments:
                await neo4j.execute_write(
                    """
                    MERGE (c:Commitment {id: $id})
                    SET c.type            = $type,
                        c.statement       = $statement,
                        c.strength        = $strength,
                        c.status          = $status,
                        c.source          = $source,
                        c.drive_source    = $drive_source,
                        c.fidelity        = $fidelity,
                        c.tests_faced     = $tests_faced,
                        c.tests_held      = $tests_held,
                        c.made_at         = datetime($made_at),
                        c.embedding       = $embedding
                    """,
                    {
                        "id": commitment.id,
                        "type": commitment.type.value,
                        "statement": commitment.statement,
                        "strength": commitment.strength.value,
                        "status": commitment.status.value,
                        "source": commitment.source.value,
                        "drive_source": commitment.drive_source,
                        "fidelity": commitment.fidelity,
                        "tests_faced": commitment.tests_faced,
                        "tests_held": commitment.tests_held,
                        "made_at": commitment.made_at.isoformat(),
                        "embedding": commitment.embedding,
                    },
                )

            for schema in self._schemas:
                await neo4j.execute_write(
                    """
                    MERGE (s:IdentitySchema {id: $id})
                    SET s.statement               = $statement,
                        s.status                  = $status,
                        s.strength                = $strength,
                        s.valence                 = $valence,
                        s.confidence              = $confidence,
                        s.salience                = $salience,
                        s.evidence_ratio          = $evidence_ratio,
                        s.confirmation_count      = $confirmation_count,
                        s.disconfirmation_count   = $disconfirmation_count,
                        s.first_formed            = datetime($first_formed),
                        s.last_activated          = datetime($last_activated),
                        s.source_pattern_ids_json = $source_pattern_ids_json,
                        s.embedding               = $embedding
                    """,
                    {
                        "id": schema.id,
                        "statement": schema.statement,
                        "status": schema.status.value,
                        "strength": schema.strength.value,
                        "valence": schema.valence.value,
                        "confidence": schema.confidence,
                        "salience": schema.salience,
                        "evidence_ratio": schema.evidence_ratio,
                        "confirmation_count": schema.confirmation_count,
                        "disconfirmation_count": schema.disconfirmation_count,
                        "first_formed": schema.first_formed.isoformat(),
                        "last_activated": schema.last_activated.isoformat(),
                        "source_pattern_ids_json": json.dumps(schema.source_pattern_ids),
                        "embedding": schema.embedding,
                    },
                )

            for chapter in self._chapters:
                await neo4j.execute_write(
                    """
                    MERGE (c:NarrativeChapter {id: $id})
                    SET c.title             = $title,
                        c.theme             = $theme,
                        c.status            = $status,
                        c.opened_at_cycle   = $opened_at_cycle,
                        c.closed_at_cycle   = $closed_at_cycle,
                        c.started_at        = datetime($started_at),
                        c.episode_count     = $episode_count,
                        c.summary           = $summary
                    """,
                    {
                        "id": chapter.id,
                        "title": chapter.title,
                        "theme": chapter.theme,
                        "status": chapter.status.value,
                        "opened_at_cycle": chapter.opened_at_cycle,
                        "closed_at_cycle": chapter.closed_at_cycle,
                        "started_at": (chapter.started_at or chapter.created_at).isoformat(),
                        "episode_count": chapter.episode_count,
                        "summary": chapter.summary,
                    },
                )

            new_fingerprints = self._fingerprints[self._fingerprints_persisted_count:]
            for fp in new_fingerprints:
                await neo4j.execute_write(
                    """
                    MERGE (f:BehavioralFingerprint {id: $id})
                    SET f.cycle_number      = $cycle_number,
                        f.created_at        = datetime($created_at),
                        f.vector            = $vector,
                        f.epoch_label       = $epoch_label,
                        f.window_start      = $cycle_number
                    """,
                    {
                        "id": fp.id,
                        "cycle_number": fp.cycle_number,
                        "created_at": fp.created_at.isoformat(),
                        "vector": fp.vector,
                        "epoch_label": f"epoch_{fp.cycle_number // 100}",
                    },
                )
            self._fingerprints_persisted_count = len(self._fingerprints)

            # Update Self node with current idem/ipse scores and autobiography (§4.1, §4.3.4, §4.8.3)
            idem = sum(s.confidence for s in self._schemas) / len(self._schemas) if self._schemas else 0.0
            ipse = self._compute_ipse_score()
            life_story_text = self._life_story.synthesis if self._life_story else ""
            active_ch = self._get_active_chapter()
            current_theme = active_ch.theme if active_ch else ""
            await neo4j.execute_write(
                """
                MERGE (s:Self)
                SET s.idem_score             = $idem_score,
                    s.ipse_score             = $ipse_score,
                    s.autobiography_summary  = $autobiography_summary,
                    s.current_life_theme     = $current_life_theme
                """,
                {
                    "idem_score": round(idem, 3),
                    "ipse_score": round(ipse, 3),
                    "autobiography_summary": life_story_text[:2000],  # Cap at 2000 chars for Neo4j
                    "current_life_theme": current_theme,
                },
            )

            self._logger.info(
                "thread_state_persisted",
                commitments=len(self._commitments),
                schemas=len(self._schemas),
                chapters=len(self._chapters),
                new_fingerprints=len(new_fingerprints),
                idem_score=round(idem, 3),
                ipse_score=round(ipse, 3),
            )

        except Exception:
            self._logger.error("thread_state_persist_failed", exc_info=True)


# ─── Utility ─────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return dot / (norm_a * norm_b)
