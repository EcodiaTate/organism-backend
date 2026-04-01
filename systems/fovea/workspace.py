"""
Fovea - Global Workspace.

Implements the computational Global Workspace (Baars 1988, Dehaene 2003):
competitive selection of the most salient content, followed by broadcast
to all cognitive systems.  Only **one** winner per cycle - the unitary
consciousness principle.

The workspace cycle operates on a theta rhythm (~150 ms) driven by
Synapse's clock.

Migrated from systems.atune.workspace during Atune → Fovea consolidation.
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from typing import Any, Protocol

import structlog

from primitives.affect import AffectState  # noqa: TC001 - Pydantic needs at runtime
from primitives.percept import Percept

from .types import (
    ActiveGoalSummary,
    MemoryContext,
    SalienceVector,
    WorkspaceBroadcast,
    WorkspaceCandidate,
    WorkspaceContext,
    WorkspaceContribution,
)

logger = structlog.get_logger("systems.fovea.workspace")


# ---------------------------------------------------------------------------
# Subscriber protocol - every system that receives broadcasts
# ---------------------------------------------------------------------------


class BroadcastSubscriber(Protocol):
    """Any system that receives workspace broadcasts."""

    system_id: str

    async def receive_broadcast(self, broadcast: WorkspaceBroadcast) -> None: ...


# ---------------------------------------------------------------------------
# Memory interface for enrichment + spontaneous recall
# ---------------------------------------------------------------------------


class WorkspaceMemoryClient(Protocol):
    """Minimal memory interface required by the workspace."""

    async def retrieve_context(
        self,
        query_embedding: list[float],
        query_text: str,
        max_results: int,
    ) -> MemoryContext: ...

    async def find_bubbling_memory(
        self,
        min_salience: float,
        max_recent_access_hours: int,
    ) -> Any | None: ...

    async def store_percept_with_broadcast(
        self,
        percept: Percept,
        salience: SalienceVector,
        affect: AffectState,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Global Workspace
# ---------------------------------------------------------------------------


class GlobalWorkspace:
    """
    The consciousness bottleneck.

    Each cycle:
    1. Candidates enter the workspace buffer.
    2. Candidates compete based on salience.
    3. The winner triggers "ignition" - broadcast to all systems.
    4. All systems receive the broadcast and may contribute to the next cycle.

    Only ONE winner per cycle (unitary consciousness principle from GWT).

    Buffer sizes scale dynamically with Soma arousal - higher arousal opens a
    wider perceptual window.  Curiosity parameters (spontaneous recall probability,
    cooldown, boost) are Evo-evolvable and genome-heritable.
    """

    def __init__(
        self,
        ignition_threshold: float = 0.3,
        buffer_size: int = 32,
        spontaneous_recall_base_prob: float = 0.02,
    ) -> None:
        # Config
        self._base_threshold = ignition_threshold
        self._buffer_size = buffer_size

        # Evolvable curiosity parameters
        self._spontaneous_base_prob: float = spontaneous_recall_base_prob
        self._cooldown_cycles: int = 20       # minimum cycles between spontaneous recalls
        self._curiosity_boost: float = 0.03   # per-unit curiosity boost to recall probability

        # Dynamic state
        self._dynamic_threshold: float = ignition_threshold
        self._subscribers: list[BroadcastSubscriber] = []
        self._current_arousal: float = 0.4    # updated via update_arousal()
        self._recent_broadcasts: deque[WorkspaceBroadcast] = deque(maxlen=20)

        # Queues - sizes recomputed each time arousal updates
        percept_q, contrib_q, broadcast_h = self._compute_buffer_sizes(self._current_arousal)
        self._percept_queue: deque[WorkspaceCandidate] = deque(maxlen=percept_q)
        self._contribution_queue: deque[WorkspaceContribution] = deque(maxlen=contrib_q)
        self._recent_broadcasts = deque(maxlen=broadcast_h)

        # Spontaneous recall tracking
        self._cycles_since_last_spontaneous: int = 100  # allow first one early
        self._pending_hypothesis_count: int = 0

        # Curiosity outcome tracking for Evo feedback (percept_id → cycle_fired)
        self._spontaneous_recall_outcomes: dict[str, int] = {}

        # Habituation tracker: source → habituation level
        self._habituation: dict[str, float] = {}

        # Cycle counter
        self._cycle_count: int = 0

        self._logger = logger.bind(component="workspace")

    # ------------------------------------------------------------------
    # Arousal-scaled buffer sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_buffer_sizes(arousal: float) -> tuple[int, int, int]:
        """
        Scale buffer sizes with Soma arousal.  Higher arousal = wider perceptual window.

        Returns (percept_queue_size, contribution_queue_size, broadcast_history_size).
        """
        clamped = max(0.0, min(1.0, arousal))
        percept_q = int(50 + 350 * clamped)    # 50–400
        contrib_q = int(20 + 80 * clamped)     # 20–100
        broadcast_h = int(10 + 30 * clamped)   # 10–40
        return percept_q, contrib_q, broadcast_h

    def update_arousal(self, arousal: float) -> None:
        """
        Update current arousal level and resize queues accordingly.

        When shrinking, oldest items are dropped from the front (deque truncation).
        This is called by PerceptionGateway each time a SOMA_TICK / ALLOSTATIC_SIGNAL
        arrives carrying an updated arousal reading.
        """
        arousal = max(0.0, min(1.0, arousal))
        if abs(arousal - self._current_arousal) < 0.01:
            return  # no meaningful change

        self._current_arousal = arousal
        percept_q, contrib_q, broadcast_h = self._compute_buffer_sizes(arousal)

        # Resize: create new deques with the updated maxlen, copying existing items.
        # deque truncation from the left (oldest items dropped) when maxlen shrinks.
        self._percept_queue = deque(self._percept_queue, maxlen=percept_q)
        self._contribution_queue = deque(self._contribution_queue, maxlen=contrib_q)
        self._recent_broadcasts = deque(self._recent_broadcasts, maxlen=broadcast_h)

        self._logger.debug(
            "workspace_buffers_resized",
            arousal=round(arousal, 3),
            percept_q=percept_q,
            contrib_q=contrib_q,
            broadcast_h=broadcast_h,
        )

    # ------------------------------------------------------------------
    # Curiosity parameter API (Evo-adjustable, genome-heritable)
    # ------------------------------------------------------------------

    def adjust_param(self, name: str, delta: float) -> None:
        """Apply a bounded delta to a curiosity parameter. Called by Evo ADJUST_BUDGET."""
        if name == "base_prob":
            self._spontaneous_base_prob = max(0.005, min(0.10, self._spontaneous_base_prob + delta))
        elif name == "cooldown_cycles":
            self._cooldown_cycles = max(5, min(50, int(self._cooldown_cycles + delta)))
        elif name == "curiosity_boost":
            self._curiosity_boost = max(0.01, min(0.10, self._curiosity_boost + delta))
        else:
            self._logger.warning("workspace_unknown_param", name=name)

    def get_learnable_params(self) -> list[tuple[str, float, float, float]]:
        """Return (name, min, max, current_value) for each curiosity parameter."""
        return [
            ("base_prob",       0.005, 0.10, self._spontaneous_base_prob),
            ("cooldown_cycles", 5.0,   50.0, float(self._cooldown_cycles)),
            ("curiosity_boost", 0.01,  0.10, self._curiosity_boost),
        ]

    def export_learnable_params(self) -> dict[str, float]:
        """Serialise curiosity params for genome transport."""
        return {
            "base_prob":       self._spontaneous_base_prob,
            "cooldown_cycles": float(self._cooldown_cycles),
            "curiosity_boost": self._curiosity_boost,
        }

    def import_learnable_params(self, params: dict[str, float]) -> None:
        """Apply inherited curiosity params with bounded ±15% Gaussian jitter for variation."""
        import random as _random

        def _jitter(val: float, lo: float, hi: float) -> float:
            noise = val * _random.gauss(0.0, 0.05)  # ±5% std (conservative)
            return max(lo, min(hi, val + noise))

        if "base_prob" in params:
            self._spontaneous_base_prob = _jitter(params["base_prob"], 0.005, 0.10)
        if "cooldown_cycles" in params:
            self._cooldown_cycles = max(5, min(50, int(_jitter(params["cooldown_cycles"], 5.0, 50.0))))
        if "curiosity_boost" in params:
            self._curiosity_boost = _jitter(params["curiosity_boost"], 0.01, 0.10)

    # ------------------------------------------------------------------
    # Spontaneous recall outcome tracking (feeds Evo fitness signal)
    # ------------------------------------------------------------------

    def record_curiosity_outcome(self, percept_id: str, positive: bool) -> None:
        """
        Called externally (by FoveaService) when a recalled percept either led
        to a Thread coherence shift or Evo hypothesis (positive=True) within 50
        cycles, or expired without effect (positive=False).

        The gateway exposes this so Evo can build a curiosity effectiveness signal.
        """
        self._spontaneous_recall_outcomes[percept_id] = 1 if positive else 0
        # Keep only the last 200 outcomes to bound memory
        if len(self._spontaneous_recall_outcomes) > 200:
            oldest = next(iter(self._spontaneous_recall_outcomes))
            self._spontaneous_recall_outcomes.pop(oldest, None)

    @property
    def curiosity_hit_rate(self) -> float:
        """Fraction of recent spontaneous recalls that produced positive outcomes."""
        outcomes = list(self._spontaneous_recall_outcomes.values())
        if not outcomes:
            return 0.5  # neutral prior
        return sum(outcomes) / len(outcomes)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def subscribe(self, subscriber: BroadcastSubscriber) -> None:
        """Register a system to receive workspace broadcasts."""
        self._subscribers.append(subscriber)
        self._logger.info("subscriber_added", system=getattr(subscriber, "system_id", "?"))

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def enqueue_scored_percept(self, candidate: WorkspaceCandidate) -> None:
        """Add a fully-scored candidate to the workspace buffer."""
        self._percept_queue.append(candidate)

    def contribute(self, contribution: WorkspaceContribution) -> None:
        """Other systems submit content for the next workspace cycle."""
        self._contribution_queue.append(contribution)

    def _drain_percept_queue(self) -> list[WorkspaceCandidate]:
        items = list(self._percept_queue)
        self._percept_queue.clear()
        return items

    def _drain_contribution_queue(self) -> list[WorkspaceContribution]:
        items = list(self._contribution_queue)
        self._contribution_queue.clear()
        return items

    # ------------------------------------------------------------------
    # Spontaneous memory surfacing
    # ------------------------------------------------------------------

    async def _check_spontaneous_recall(
        self,
        affect: AffectState,
        memory_client: WorkspaceMemoryClient | None,
    ) -> WorkspaceCandidate | None:
        """
        Probabilistically surface a memory that is currently highly salient
        but hasn't been accessed recently.  Creates the "I just thought of
        something" experience.
        """
        if self._cycles_since_last_spontaneous < self._cooldown_cycles:
            return None

        if memory_client is None:
            return None

        base_prob = self._spontaneous_base_prob
        curiosity_boost = affect.curiosity * self._curiosity_boost
        hypothesis_boost = min(0.02, self._pending_hypothesis_count * 0.005)
        total_prob = base_prob + curiosity_boost + hypothesis_boost

        if random.random() > total_prob:
            return None

        memory = await memory_client.find_bubbling_memory(
            min_salience=0.5,
            max_recent_access_hours=24,
        )
        if memory is None:
            return None

        self._cycles_since_last_spontaneous = 0
        self._logger.debug("spontaneous_recall_triggered")

        # Wrap the memory as a WorkspaceCandidate
        candidate = WorkspaceCandidate(
            content=memory,
            salience=SalienceVector(
                scores={},
                composite=0.5 * 0.7,  # dampened salience for spontaneous
            ),
            source="spontaneous_recall",
            prediction_error=None,
        )
        # Register this recall for outcome tracking.
        # FoveaService will resolve it within 50 cycles (positive or neutral).
        if hasattr(memory, "id"):
            self._spontaneous_recall_outcomes.setdefault(memory.id, -1)  # -1 = pending
        return candidate

    # ------------------------------------------------------------------
    # Context enrichment
    # ------------------------------------------------------------------

    async def _enrich_with_memory(
        self,
        candidate: WorkspaceCandidate,
        memory_client: WorkspaceMemoryClient | None,
    ) -> MemoryContext:
        """Retrieve relevant memories to provide context before broadcast."""
        if memory_client is None:
            return MemoryContext()

        if not isinstance(candidate.content, Percept):
            return MemoryContext()

        percept: Percept = candidate.content
        text = percept.content.parsed if isinstance(percept.content.parsed, str) else ""

        try:
            return await memory_client.retrieve_context(
                query_embedding=percept.content.embedding or [],
                query_text=text,
                max_results=10,
            )
        except Exception:
            self._logger.warning("memory_enrichment_failed", exc_info=True)
            return MemoryContext()

    # ------------------------------------------------------------------
    # Dynamic threshold adjustment
    # ------------------------------------------------------------------

    def _adjust_threshold(self, candidate_count: int) -> None:
        """
        Adapt the ignition threshold based on input volume.

        Too many candidates → raise (be more selective).
        Too few → lower (be more open).
        """
        if candidate_count > self._buffer_size * 0.8:
            self._dynamic_threshold = min(0.8, self._dynamic_threshold + 0.02)
        elif candidate_count < 3:
            self._dynamic_threshold = max(0.15, self._dynamic_threshold - 0.01)
        else:
            # Drift back toward baseline
            self._dynamic_threshold += 0.005 * (self._base_threshold - self._dynamic_threshold)

    # ------------------------------------------------------------------
    # Habituation
    # ------------------------------------------------------------------

    def _update_habituation(self, winner: WorkspaceCandidate) -> None:
        """
        Increase habituation for the winning source, decay others.

        Stores habituation under BOTH the full source key (e.g. "external:text_chat")
        and the percept's source.system (e.g. "text_chat") so that Fovea's
        novelty dimension can look up habituation via percept.source.system correctly.
        """
        source = winner.source
        # Also track by percept source system for novelty dimension compatibility
        percept_system = getattr(
            getattr(winner.content, "source", None), "system", None
        )
        keys_to_boost = {source}
        if percept_system and percept_system != source:
            keys_to_boost.add(percept_system)

        for key in keys_to_boost:
            self._habituation[key] = min(1.0, self._habituation.get(key, 0.0) + 0.05)
        # Decay all sources slowly
        for key in list(self._habituation):
            if key not in keys_to_boost:
                self._habituation[key] = max(0.0, self._habituation[key] - 0.01)

    @property
    def habituation_map(self) -> dict[str, float]:
        return dict(self._habituation)

    # ------------------------------------------------------------------
    # The main cycle
    # ------------------------------------------------------------------

    async def run_cycle(
        self,
        affect: AffectState,
        active_goals: list[ActiveGoalSummary] | None = None,
        memory_client: WorkspaceMemoryClient | None = None,
    ) -> WorkspaceBroadcast | None:
        """
        One theta cycle of the Global Workspace.  Called by Synapse at
        ~150 ms intervals.

        Returns the broadcast if ignition occurred, else ``None``.
        """
        self._cycle_count += 1
        self._cycles_since_last_spontaneous += 1
        active_goals = active_goals or []

        # ── PHASE 1: Collect candidates ──────────────────────────────
        candidates: list[WorkspaceCandidate] = self._drain_percept_queue()

        # Internal contributions from other systems
        for contrib in self._drain_contribution_queue():
            candidates.append(
                WorkspaceCandidate(
                    content=contrib.content,
                    salience=SalienceVector(scores={}, composite=contrib.priority),
                    source=f"internal:{contrib.system}",
                    prediction_error=None,
                )
            )

        # Spontaneous memory surfacing
        spontaneous = await self._check_spontaneous_recall(affect, memory_client)
        if spontaneous is not None:
            candidates.append(spontaneous)

        if not candidates:
            return None  # Empty cycle

        # ── PHASE 2: Competitive selection (ignition) ────────────────
        candidates.sort(key=lambda c: c.salience.composite, reverse=True)
        winner = candidates[0]

        if winner.salience.composite < self._dynamic_threshold:
            self._logger.debug(
                "no_ignition",
                best_salience=round(winner.salience.composite, 4),
                threshold=round(self._dynamic_threshold, 4),
            )
            self._adjust_threshold(len(candidates))
            return None

        # ── PHASE 3: Context enrichment ──────────────────────────────
        memory_context = await self._enrich_with_memory(winner, memory_client)

        # ── PHASE 4: Broadcast ───────────────────────────────────────
        broadcast = WorkspaceBroadcast(
            content=winner.content,
            salience=winner.salience,
            affect=affect,
            context=WorkspaceContext(
                recent_broadcast_ids=[b.broadcast_id for b in list(self._recent_broadcasts)[-5:]],
                active_goal_ids=[g.id for g in active_goals],
                memory_context=memory_context,
                prediction_error=winner.prediction_error,
            ),
            precision=winner.salience.composite,
            source=winner.source,
        )

        # Fan out to all subscribers in parallel
        if self._subscribers:
            results = await asyncio.gather(
                *(sub.receive_broadcast(broadcast) for sub in self._subscribers),
                return_exceptions=True,
            )
            for sub, result in zip(self._subscribers, results, strict=False):
                if isinstance(result, Exception):
                    self._logger.warning(
                        "broadcast_ack_failed",
                        system=getattr(sub, "system_id", "?"),
                        error=str(result),
                    )

        self._recent_broadcasts.append(broadcast)

        # ── PHASE 5: Memory storage (non-blocking) ───────────────────
        if isinstance(winner.content, Percept) and memory_client is not None:
            task = asyncio.create_task(
                memory_client.store_percept_with_broadcast(
                    winner.content, winner.salience, affect,
                ),
                name="workspace_memory_store",
            )
            task.add_done_callback(
                lambda t: self._logger.warning(
                    "memory_store_failed", error=str(t.exception())
                )
                if not t.cancelled() and t.exception() is not None
                else None
            )

        # ── PHASE 6: Dynamic threshold adjustment ────────────────────
        self._adjust_threshold(len(candidates))

        # ── PHASE 7: Habituation update ──────────────────────────────
        self._update_habituation(winner)

        self._logger.info(
            "workspace_broadcast",
            broadcast_id=broadcast.broadcast_id,
            salience=round(winner.salience.composite, 4),
            source=winner.source,
            subscriber_count=len(self._subscribers),
            cycle=self._cycle_count,
        )

        return broadcast

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def dynamic_threshold(self) -> float:
        return self._dynamic_threshold

    @property
    def recent_broadcasts(self) -> list[WorkspaceBroadcast]:
        return list(self._recent_broadcasts)

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
