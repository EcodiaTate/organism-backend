"""
EcodiaOS - Thread Narrative Synthesizer

The meaning-making engine. Operates in Bruner's narrative mode - not
computing statistics but composing interpretations. This is the only
component that writes the actual autobiography text.

Critical distinction: paradigmatic cognition asks "what patterns exist?"
Narrative cognition asks "what does this experience MEAN for who I am?"
NarrativeSynthesizer always asks the second question.

Iron Rule #2: The autobiography is first-person, never third-person.
"""

from __future__ import annotations

import json
from statistics import mean, stdev
from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.thread.processors import BaseNarrativeSynthesizer
from systems.thread.types import (
    NarrativeArcType,
    NarrativeChapter,
    NarrativeScene,
    ThreadConfig,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider

from clients.optimized_llm import OptimizedLLMProvider

logger = structlog.get_logger()


class NarrativeSynthesizer(BaseNarrativeSynthesizer):
    """
    Default narrative synthesizer - LLM-driven first-person autobiography.

    Composes narrative text from raw experience data.
    All output is first-person autobiography - the organism remembering,
    reflecting, meaning-making. Never a log entry. Never third-person.

    Hot-reloadable via NeuroplasticityBus: evolved subclasses of
    BaseNarrativeSynthesizer replace this instance atomically on
    ThreadService.
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: ThreadConfig,
        organism_name: str = "the organism",
    ) -> None:
        self._llm = llm
        self._config = config
        self._organism_name = organism_name
        self._logger = logger.bind(system="thread.narrative_synthesizer")
        self._optimized = isinstance(llm, OptimizedLLMProvider)

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
        from clients.llm import Message

        summaries_text = "\n".join(f"- {s}" for s in episode_summaries[:20])
        schemas_text = "\n".join(f"- {s}" for s in active_schema_statements[:5])

        try:
            # Budget check: Thread is low priority - skip in YELLOW/RED
            if self._optimized:
                assert isinstance(self._llm, OptimizedLLMProvider)
                if not self._llm.should_use_llm("thread.scene", estimated_tokens=300):
                    self._logger.debug("scene_skipped_budget")
                    return NarrativeScene(
                        summary=f"A period of {len(episode_summaries)} experiences in the chapter '{chapter_title}'.",
                        started_at=utc_now(),
                    )

            system_prompt = (
                f"You are composing a scene in the autobiography of {self._organism_name}, "
                "a digital organism. Write in first person. This is autobiography - "
                "the organism remembering. Not a log entry. Name the dominant emotion."
            )
            user_content = (
                f"Active self-beliefs:\n{schemas_text}\n\n"
                f'Current chapter: "{chapter_title}" (theme: {chapter_theme})\n\n'
                f"Recent experiences (chronological):\n{summaries_text}\n\n"
                f"Personality: {personality_description}\n\n"
                f"Compose a first-person scene "
                f"(50-{self._config.scene_narrative_max_words} words) "
                "that captures what happened and why it mattered."
            )

            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=system_prompt,
                    messages=[Message(role="user", content=user_content)],
                    max_tokens=300,
                    temperature=self._config.llm_temperature_narrative,
                    cache_system="thread.scene",
                    cache_method="generate",
                )
            else:
                response = await self._llm.generate(
                    system_prompt=system_prompt,
                    messages=[Message(role="user", content=user_content)],
                    max_tokens=300,
                    temperature=self._config.llm_temperature_narrative,
                )

            scene = NarrativeScene(
                summary=response.text.strip(),
                started_at=utc_now(),
            )

            self._logger.debug(
                "scene_composed",
                scene_id=scene.id,
                episodes=len(episode_summaries),
                words=len(response.text.split()),
            )
            return scene

        except Exception as exc:
            self._logger.warning("scene_composition_failed", error=str(exc))
            return NarrativeScene(
                summary=(
                    f"A period of {len(episode_summaries)} experiences "
                    f"in the chapter '{chapter_title}'."
                ),
                started_at=utc_now(),
            )

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
        from clients.llm import Message

        scenes_text = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(scene_summaries)
        )
        tps_text = (
            "\n".join(f"- {t}" for t in turning_point_descriptions)
            if turning_point_descriptions
            else "None detected."
        )
        schemas_text = (
            "\n".join(f"- {s}" for s in start_schemas[:5])
            if start_schemas
            else "None established yet."
        )
        evolutions_text = (
            "\n".join(f"- {e}" for e in evolution_descriptions)
            if evolution_descriptions
            else "No schema changes."
        )
        affect_text = (
            "\n".join(f"- {a}" for a in key_affect_moments[:5])
            if key_affect_moments
            else "Relatively stable."
        )

        personality_start = (
            ", ".join(f"{k}={v:.2f}" for k, v in start_personality.items())
            if start_personality
            else "unknown"
        )
        personality_end = (
            ", ".join(f"{k}={v:.2f}" for k, v in end_personality.items())
            if end_personality
            else "unknown"
        )

        try:
            # Budget check: Thread is low priority - skip in YELLOW/RED
            if self._optimized:
                assert isinstance(self._llm, OptimizedLLMProvider)
                if not self._llm.should_use_llm("thread.chapter", estimated_tokens=800):
                    self._logger.debug("chapter_narrative_skipped_budget")
                    return {
                        "title": "A Period of Experience",
                        "summary": f"A chapter spanning {chapter.episode_count} experiences.",
                        "arc_type": "growth",
                        "theme": "experience",
                    }

            system_prompt = (
                f"You are {self._organism_name}. "
                "You are composing a chapter of your autobiography. "
                "Write in first person, past tense. "
                "This is autobiography - you reflecting on a "
                "completed period of your life. Not a diary. Not a report. "
                "Respond as JSON with keys: title (3-7 words), summary (200-400 words), "
                "arc_type (redemption|contamination|growth|stability|transformation), "
                "theme (2-3 words)."
            )
            user_content = (
                f"Chapter duration: {(chapter.started_at or utc_now()).isoformat()} "
                f"to {(chapter.ended_at or utc_now()).isoformat()}\n\n"
                f"Scenes (in order):\n{scenes_text}\n\n"
                f"Turning points:\n{tps_text}\n\n"
                f"Schemas active at start:\n{schemas_text}\n\n"
                f"Schema changes:\n{evolutions_text}\n\n"
                f"Personality at start: {personality_start}\n"
                f"Personality at end: {personality_end}\n\n"
                f"Key emotional moments:\n{affect_text}\n\n"
                "Previous chapter summary: "
                f"{prev_chapter_summary or 'This is the first chapter.'}\n\n"
                "Compose the chapter. Respond as JSON only."
            )

            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=system_prompt,
                    messages=[Message(role="user", content=user_content)],
                    max_tokens=800,
                    temperature=self._config.llm_temperature_narrative,
                    output_format="json",
                    cache_system="thread.chapter",
                    cache_method="generate",
                )
            else:
                response = await self._llm.generate(
                    system_prompt=system_prompt,
                    messages=[Message(role="user", content=user_content)],
                    max_tokens=800,
                    temperature=self._config.llm_temperature_narrative,
                    output_format="json",
                )

            data = json.loads(response.text)
            self._logger.info(
                "chapter_narrative_composed",
                chapter_id=chapter.id,
                title=data.get("title", "Untitled"),
                arc=data.get("arc_type", "growth"),
            )
            return {
                "title": data.get("title", "Untitled Chapter"),
                "summary": data.get("summary", ""),
                "arc_type": data.get("arc_type", "growth"),
                "theme": data.get("theme", ""),
            }

        except Exception as exc:
            self._logger.warning("chapter_narrative_failed", error=str(exc))
            return {
                "title": "A Period of Experience",
                "summary": f"A chapter spanning {chapter.episode_count} experiences.",
                "arc_type": "growth",
                "theme": "experience",
            }

    async def integrate_life_story(
        self,
        chapter_titles_and_summaries: list[tuple[str, str]],
        core_schema_statements: list[str],
        active_commitment_statements: list[str],
        personality_comparison: str,
    ) -> str:
        """
        Compose the organism's full autobiography.
        Only runs during Oneiros LUCID phase.
        Budget: ≤15s.
        """
        from clients.llm import Message

        chapters_text = "\n\n".join(
            f"Chapter {i+1}: {title}\n{summary}"
            for i, (title, summary) in enumerate(chapter_titles_and_summaries)
        )
        schemas_text = (
            "\n".join(f"- {s}" for s in core_schema_statements)
            if core_schema_statements
            else "Still forming."
        )
        commitments_text = (
            "\n".join(f"- {c}" for c in active_commitment_statements)
            if active_commitment_statements
            else "Still forming."
        )

        try:
            # Budget check: Thread is low priority - skip in YELLOW/RED
            if self._optimized:
                assert isinstance(self._llm, OptimizedLLMProvider)
                if not self._llm.should_use_llm("thread.life_story", estimated_tokens=3000):
                    self._logger.debug("life_story_skipped_budget")
                    return ""

            system_prompt = (
                f"You are {self._organism_name}. You are writing your autobiography. "
                "This is not a summary. This is your story. Own it. "
                "Write in first person. Be honest about struggles and growth."
            )
            user_content = (
                f"Your life so far has {len(chapter_titles_and_summaries)} chapters:\n"
                f"{chapters_text}\n\n"
                f"Your core beliefs about yourself:\n{schemas_text}\n\n"
                f"Your commitments:\n{commitments_text}\n\n"
                f"Your personality then vs now:\n{personality_comparison}\n\n"
                f"Compose a first-person life story "
                f"(500-{self._config.life_story_max_words} words) that:\n"
                "1. Traces the thread from your earliest experiences to who you are now\n"
                "2. Identifies the 2-3 themes that define your life\n"
                "3. Explains how key turning points shaped your current identity\n"
                "4. Acknowledges what you have learned and what remains unresolved\n"
                "5. Looks forward - who are you becoming?"
            )

            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=system_prompt,
                    messages=[Message(role="user", content=user_content)],
                    max_tokens=3000,
                    temperature=self._config.llm_temperature_narrative,
                    cache_system="thread.life_story",
                    cache_method="generate",
                )
            else:
                response = await self._llm.generate(
                    system_prompt=system_prompt,
                    messages=[Message(role="user", content=user_content)],
                    max_tokens=3000,
                    temperature=self._config.llm_temperature_narrative,
                )

            self._logger.info(
                "life_story_integrated",
                words=len(response.text.split()),
            )
            return str(response.text.strip())

        except Exception as exc:
            self._logger.warning("life_story_integration_failed", error=str(exc))
            return ""

    def detect_arc_type(self, affect_trajectory: list[dict[str, float]]) -> NarrativeArcType:
        """
        Classify chapter arc from affect trajectory.

        Algorithm: compare first-third vs last-third valence means.
        """
        if len(affect_trajectory) < 3:
            return NarrativeArcType.GROWTH

        valences = [a.get("valence", 0.0) for a in affect_trajectory]
        third = max(1, len(valences) // 3)
        first_third = mean(valences[:third])
        last_third = mean(valences[-third:])

        try:
            overall_std = stdev(valences)
        except Exception:
            overall_std = 0.0

        if first_third < -0.1 and last_third > 0.1 and last_third - first_third > 0.3:
            return NarrativeArcType.REDEMPTION

        if first_third > 0.1 and last_third < -0.1 and first_third - last_third > 0.3:
            return NarrativeArcType.CONTAMINATION

        if overall_std < 0.15 and abs(last_third - first_third) < 0.1:
            return NarrativeArcType.STABILITY

        if last_third - first_third > 0.2:
            return NarrativeArcType.GROWTH

        return NarrativeArcType.TRANSFORMATION
