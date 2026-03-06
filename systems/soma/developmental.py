"""
EcodiaOS — Soma Developmental Stage System

Manages the organism's maturation through five stages, each unlocking
progressively more sophisticated interoceptive capabilities.

Stage transitions are triggered by measurable milestones:
  REFLEXIVE → ASSOCIATIVE:  cycle_count > 10k AND mean_confidence > 0.5
  ASSOCIATIVE → DELIBERATIVE: cycle_count > 100k AND attractor_count >= 3
  DELIBERATIVE → REFLECTIVE:  cycle_count > 1M AND bifurcation_count >= 2
  REFLECTIVE → GENERATIVE:    manual_promotion OR novel_attractor_discovered

Stages never regress — once the organism reaches a stage, it stays.
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.soma.types import (
    _STAGE_ORDER,
    STAGE_HORIZONS,
    DevelopmentalStage,
)

logger = structlog.get_logger("systems.soma.developmental")


# ─── Stage Capability Gates ──────────────────────────────────────

STAGE_CAPABILITIES: dict[DevelopmentalStage, dict[str, Any]] = {
    DevelopmentalStage.REFLEXIVE: {
        "soma": ["sensing", "basic_prediction"],
        "nova_constraint": "fast_path_only",
        "evo_constraint": "pattern_detection_only",
        "oneiros_constraint": "nrem_only",
        "phase_space": False,
        "counterfactual": False,
        "setpoint_adaptation": False,
    },
    DevelopmentalStage.ASSOCIATIVE: {
        "soma": ["sensing", "multi_horizon_prediction", "allostatic_error"],
        "nova_constraint": "fast_path_and_allostatic",
        "evo_constraint": "pattern_detection_and_hypothesis",
        "oneiros_constraint": "nrem_and_rem_basic",
        "phase_space": False,
        "counterfactual": False,
        "setpoint_adaptation": False,
    },
    DevelopmentalStage.DELIBERATIVE: {
        "soma": ["full_prediction", "setpoint_adaptation", "metabolic_coupling"],
        "nova_constraint": None,
        "evo_constraint": None,
        "oneiros_constraint": None,
        "phase_space": True,
        "counterfactual": False,
        "setpoint_adaptation": True,
    },
    DevelopmentalStage.REFLECTIVE: {
        "soma": ["full"],
        "nova_constraint": None,
        "evo_constraint": None,
        "oneiros_constraint": None,
        "phase_space": True,
        "counterfactual": True,
        "setpoint_adaptation": True,
    },
    DevelopmentalStage.GENERATIVE: {
        "soma": ["full"],
        "nova_constraint": None,
        "evo_constraint": None,
        "oneiros_constraint": None,
        "phase_space": True,
        "counterfactual": True,
        "setpoint_adaptation": True,
    },
}


# ─── Stage Transition Triggers ───────────────────────────────────

class StageTransitionTrigger:
    """Evaluates whether conditions are met for a stage transition."""

    @staticmethod
    def check_reflexive_to_associative(
        cycle_count: int,
        mean_confidence: float,
    ) -> bool:
        return cycle_count > 10_000 and mean_confidence > 0.5

    @staticmethod
    def check_associative_to_deliberative(
        cycle_count: int,
        attractor_count: int,
    ) -> bool:
        return cycle_count > 100_000 and attractor_count >= 3

    @staticmethod
    def check_deliberative_to_reflective(
        cycle_count: int,
        bifurcation_count: int,
    ) -> bool:
        return cycle_count > 1_000_000 and bifurcation_count >= 2

    @staticmethod
    def check_reflective_to_generative(
        manual_promotion: bool = False,
        novel_attractor_autonomous: bool = False,
    ) -> bool:
        return manual_promotion or novel_attractor_autonomous


class DevelopmentalManager:
    """
    Manages the organism's developmental stage and capability gating.

    Evaluates transition triggers each update cycle and promotes
    the organism when milestones are met. Stages never regress.
    """

    def __init__(self, initial_stage: DevelopmentalStage = DevelopmentalStage.REFLEXIVE) -> None:
        self._stage = initial_stage
        self._trigger = StageTransitionTrigger()
        self._novel_attractor_count = 0  # Tracks autonomously discovered attractors

    @property
    def stage(self) -> DevelopmentalStage:
        return self._stage

    @property
    def capabilities(self) -> dict[str, Any]:
        return STAGE_CAPABILITIES.get(self._stage, STAGE_CAPABILITIES[DevelopmentalStage.REFLEXIVE])

    @property
    def available_horizons(self) -> list[str]:
        return STAGE_HORIZONS.get(self._stage, ["immediate", "moment"])

    def is_enabled(self, capability: str) -> bool:
        """Check if a specific capability is enabled at current stage."""
        caps = self.capabilities
        return bool(caps.get(capability, False))

    def phase_space_enabled(self) -> bool:
        return self.is_enabled("phase_space")

    def counterfactual_enabled(self) -> bool:
        return self.is_enabled("counterfactual")

    def setpoint_adaptation_enabled(self) -> bool:
        return self.is_enabled("setpoint_adaptation")

    def evaluate_transition(
        self,
        cycle_count: int,
        mean_confidence: float = 0.0,
        attractor_count: int = 0,
        bifurcation_count: int = 0,
    ) -> bool:
        """
        Check if conditions are met for a stage transition.
        Returns True if a promotion occurred.
        """
        promoted = False

        if self._stage == DevelopmentalStage.REFLEXIVE:
            if self._trigger.check_reflexive_to_associative(cycle_count, mean_confidence):
                self._promote(DevelopmentalStage.ASSOCIATIVE)
                promoted = True

        if self._stage == DevelopmentalStage.ASSOCIATIVE:
            if self._trigger.check_associative_to_deliberative(cycle_count, attractor_count):
                self._promote(DevelopmentalStage.DELIBERATIVE)
                promoted = True

        if self._stage == DevelopmentalStage.DELIBERATIVE:
            if self._trigger.check_deliberative_to_reflective(cycle_count, bifurcation_count):
                self._promote(DevelopmentalStage.REFLECTIVE)
                promoted = True

        if self._stage == DevelopmentalStage.REFLECTIVE:
            if self._trigger.check_reflective_to_generative(
                novel_attractor_autonomous=self._novel_attractor_count > 0,
            ):
                self._promote(DevelopmentalStage.GENERATIVE)
                promoted = True

        return promoted

    def manual_promote(self, target: DevelopmentalStage) -> bool:
        """
        Manually promote to a target stage. Only forward promotions allowed.
        """
        if _STAGE_ORDER[target] > _STAGE_ORDER[self._stage]:
            self._promote(target)
            return True
        return False

    def notify_novel_attractor(self) -> None:
        """Notify that an attractor was discovered autonomously (not seeded)."""
        self._novel_attractor_count += 1

    def _promote(self, new_stage: DevelopmentalStage) -> None:
        old = self._stage
        self._stage = new_stage
        logger.info(
            "developmental_stage_promoted",
            from_stage=old.value,
            to_stage=new_stage.value,
        )
