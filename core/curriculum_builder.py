"""
EcodiaOS - DomainCurriculum

Organises RE training examples for domain-specific continual learning.

The curriculum is built from a flat list of RETrainingExample objects emitted
by all systems.  Given a target domain it:

1. Filters examples to those for that domain, plus generalist examples whose
   transferable_skills overlap the domain's known skill areas.
2. Sorts by difficulty (novice → intermediate → expert) so the LoRA adapter
   sees easy patterns before harder ones - consistent with curriculum learning
   literature (Bengio et al. 2009).
3. Groups by skill_area so the caller can inspect coverage.
4. Checks prerequisite satisfaction: a skill_area is "ready" only when all
   skills in its prerequisite list have at least one example in the curriculum.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

from primitives.re_training import RETrainingExample

logger = logging.getLogger(__name__)

_DIFFICULTY_ORDER = {"novice": 0, "intermediate": 1, "expert": 2}

# Per-domain prerequisite graph: skill_area → [required skill_areas that must
# appear first in the curriculum].  Extend as domains are added.
_DOMAIN_PREREQUISITES: dict[str, dict[str, list[str]]] = {
    "software_development": {
        "code_quality_assessment": [],
        "architecture_design": ["code_quality_assessment"],
        "debugging": ["code_quality_assessment"],
        "security_review": ["architecture_design", "debugging"],
    },
    "defi": {
        "yield_optimization": [],
        "risk_evaluation": ["yield_optimization"],
        "portfolio_rebalancing": ["risk_evaluation"],
    },
    "trading": {
        "market_pattern_recognition": [],
        "position_sizing": ["market_pattern_recognition"],
        "execution_timing": ["position_sizing"],
    },
    "art": {
        "aesthetic_evaluation": [],
        "style_coherence": ["aesthetic_evaluation"],
        "composition_analysis": ["aesthetic_evaluation"],
    },
    "service_delivery": {
        "customer_communication": [],
        "expectation_management": ["customer_communication"],
        "escalation_handling": ["expectation_management"],
    },
}


class DomainCurriculum:
    """
    Organises training examples for domain-specific learning.

    Usage::

        curriculum = DomainCurriculum("software_development")
        await curriculum.build(all_examples)
        ordered = curriculum.get_ordered_examples()
        # pass `ordered` to the LoRA training script
    """

    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.examples_by_skill_area: dict[str, list[RETrainingExample]] = defaultdict(
            list
        )
        # True → enough prerequisite examples exist to train this skill area
        self.prerequisites_satisfied: dict[str, bool] = {}
        self._total: int = 0

    async def build(self, all_examples: Sequence[RETrainingExample]) -> None:
        """
        Filter, sort, and group examples for this domain.

        An example is included when:
        - example.domain == self.domain, OR
        - "generalist" appears in example.transferable_skills and the
          example's skill_area is relevant to this domain.
        """
        domain_skill_areas = set(
            _DOMAIN_PREREQUISITES.get(self.domain, {}).keys()
        )

        selected: list[RETrainingExample] = []
        for ex in all_examples:
            if ex.domain == self.domain:
                selected.append(ex)
            elif (
                "generalist" in ex.transferable_skills
                and ex.skill_area in domain_skill_areas
            ):
                selected.append(ex)

        # Sort novice → intermediate → expert
        selected.sort(
            key=lambda e: _DIFFICULTY_ORDER.get(e.domain_difficulty, 0)
        )

        self.examples_by_skill_area = defaultdict(list)
        for ex in selected:
            key = ex.skill_area or "_unclassified"
            self.examples_by_skill_area[key].append(ex)

        self._total = len(selected)
        self._check_prerequisites()

        logger.debug(
            "DomainCurriculum.build: domain=%s total=%d skill_areas=%s",
            self.domain,
            self._total,
            list(self.examples_by_skill_area.keys()),
        )

    def get_ordered_examples(self) -> list[RETrainingExample]:
        """
        Return examples in prerequisite-respecting, difficulty-ordered sequence.

        Skill areas whose prerequisites are not satisfied are placed last so
        the adapter at least sees them, rather than dropping them entirely.
        """
        prereqs = _DOMAIN_PREREQUISITES.get(self.domain, {})

        # Topological sort of skill areas by prerequisites
        ordered_skills: list[str] = []
        remaining = list(self.examples_by_skill_area.keys())
        satisfied: set[str] = set()

        # Safety limit - should never loop more than len(remaining)² times
        max_passes = len(remaining) ** 2 + 1
        passes = 0

        while remaining and passes < max_passes:
            passes += 1
            for skill in list(remaining):
                deps = prereqs.get(skill, [])
                if all(d in satisfied or d not in remaining for d in deps):
                    ordered_skills.append(skill)
                    satisfied.add(skill)
                    remaining.remove(skill)

        # Append anything that couldn't be resolved (circular / unknown prereqs)
        ordered_skills.extend(remaining)

        result: list[RETrainingExample] = []
        for skill in ordered_skills:
            result.extend(self.examples_by_skill_area[skill])
        return result

    @property
    def total_examples(self) -> int:
        return self._total

    # ── Internal ─────────────────────────────────────────────────────────────

    def _check_prerequisites(self) -> None:
        prereqs = _DOMAIN_PREREQUISITES.get(self.domain, {})
        available_skills = set(self.examples_by_skill_area.keys())
        for skill, deps in prereqs.items():
            self.prerequisites_satisfied[skill] = all(
                d in available_skills for d in deps
            )
        # Skills not in the prereq graph are trivially satisfied
        for skill in available_skills:
            if skill not in self.prerequisites_satisfied:
                self.prerequisites_satisfied[skill] = True

    def _get_skill_prerequisites(self) -> dict[str, list[str]]:
        return _DOMAIN_PREREQUISITES.get(self.domain, {})
