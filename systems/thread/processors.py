"""
EcodiaOS - Thread Processor ABCs

Abstract base classes for the Thread system's hot-reloadable processors.
The NeuroplasticityBus discovers subclasses of these ABCs in evolved files
and swaps the live instances on ThreadService atomically.

Two ABCs:
  BaseNarrativeSynthesizer - first-person autobiography composition
  BaseChapterDetector      - chapter boundary detection in experience stream

Evolved subclasses must preserve the contracts defined here. State that
must survive a hot-swap (e.g. the surprise accumulator) is owned by
ThreadService, not the processor - so swapping the processor never causes
the organism to forget its current narrative chapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from systems.thread.types import (
        NarrativeArcType,
        NarrativeChapter,
        NarrativeScene,
        NarrativeSurpriseAccumulator,
        ThreadConfig,
    )


# ─── Narrative Synthesizer ABC ────────────────────────────────────────────────


class BaseNarrativeSynthesizer(ABC):
    """
    Abstract base for Thread narrative synthesis.

    Subclasses compose first-person autobiography text from raw experience.
    All output must be first-person - the organism remembering, not a log
    entry, never third-person.

    Iron Rule #2: The autobiography is first-person, never third-person.

    Evolved subclasses can change *how* the organism narrates (prompts,
    arc detection algorithms, temperature) without changing *what* it
    narrates about (the data contract is fixed).
    """

    @abstractmethod
    async def compose_scene(
        self,
        episode_summaries: list[str],
        chapter_title: str,
        chapter_theme: str,
        active_schema_statements: list[str],
        personality_description: str = "",
    ) -> NarrativeScene:
        """
        Group related episodes into a narrative scene.
        Budget: ≤2s per scene.
        """
        ...

    @abstractmethod
    async def compose_chapter_narrative(
        self,
        chapter: NarrativeChapter,
        scene_summaries: list[str],
        turning_point_descriptions: list[str],
        start_schemas: list[str],
        evolution_descriptions: list[str],
        start_personality: dict[str, float],
        end_personality: dict[str, float],
        key_affect_moments: list[str],
        prev_chapter_summary: str = "",
    ) -> dict[str, str]:
        """
        Compose a chapter narrative on closure.
        Budget: ≤5s.

        Returns dict with keys: title, summary, arc_type, theme.
        """
        ...

    @abstractmethod
    async def integrate_life_story(
        self,
        chapter_titles_and_summaries: list[tuple[str, str]],
        core_schema_statements: list[str],
        active_commitment_statements: list[str],
        personality_comparison: str,
    ) -> str:
        """
        Compose the organism's full autobiography.
        Budget: ≤15s.
        """
        ...

    @abstractmethod
    def detect_arc_type(
        self, affect_trajectory: list[dict[str, float]]
    ) -> NarrativeArcType:
        """
        Classify chapter arc from affect trajectory.
        Pure computation, no LLM.
        """
        ...


# ─── Chapter Detector ABC ────────────────────────────────────────────────────


class BaseChapterDetector(ABC):
    """
    Abstract base for Thread chapter boundary detection.

    Subclasses detect when the organism's story has shifted enough to
    warrant a new chapter.

    CRITICAL: The ``NarrativeSurpriseAccumulator`` is passed INTO methods
    rather than owned by the detector. This means a hot-swap never loses
    the running surprise statistics - ThreadService owns the accumulator
    and hands it to whichever detector is active.

    Performance: boundary check must complete in ≤10ms (no LLM calls).
    """

    @abstractmethod
    def check_boundary(
        self,
        episode_data: dict[str, Any],
        affect: AffectState,
        accumulator: NarrativeSurpriseAccumulator,
        config: ThreadConfig,
        schema_challenged: bool = False,
    ) -> bool:
        """
        Evaluate whether the current episode marks a chapter boundary.
        Must complete in ≤10ms (no LLM calls).

        The accumulator and config are passed in so the detector is
        stateless - all mutable state lives in ThreadService.

        Returns True if a chapter boundary is detected.
        """
        ...

    @abstractmethod
    def create_new_chapter(
        self,
        previous_chapter: NarrativeChapter | None = None,
        personality_snapshot: dict[str, float] | None = None,
        active_schema_ids: list[str] | None = None,
    ) -> NarrativeChapter:
        """
        Create a new FORMING chapter after boundary detection and
        closure of the previous chapter.
        """
        ...
