"""Tests for Soma Developmental Stage System - maturation and capability gating."""

from __future__ import annotations

from systems.soma.developmental import (
    DevelopmentalManager,
    StageTransitionTrigger,
)
from systems.soma.types import DevelopmentalStage


class TestStageTransitionTrigger:
    def test_reflexive_to_associative(self):
        t = StageTransitionTrigger()
        assert not t.check_reflexive_to_associative(5_000, 0.6)
        assert not t.check_reflexive_to_associative(15_000, 0.3)
        assert t.check_reflexive_to_associative(15_000, 0.6)

    def test_associative_to_deliberative(self):
        t = StageTransitionTrigger()
        assert not t.check_associative_to_deliberative(50_000, 3)
        assert not t.check_associative_to_deliberative(150_000, 2)
        assert t.check_associative_to_deliberative(150_000, 3)

    def test_deliberative_to_reflective(self):
        t = StageTransitionTrigger()
        assert not t.check_deliberative_to_reflective(500_000, 2)
        assert not t.check_deliberative_to_reflective(1_500_000, 1)
        assert t.check_deliberative_to_reflective(1_500_000, 2)

    def test_reflective_to_generative_manual(self):
        t = StageTransitionTrigger()
        assert t.check_reflective_to_generative(manual_promotion=True)
        assert t.check_reflective_to_generative(novel_attractor_autonomous=True)
        assert not t.check_reflective_to_generative()


class TestDevelopmentalManager:
    def test_initial_stage(self):
        dm = DevelopmentalManager(initial_stage=DevelopmentalStage.REFLEXIVE)
        assert dm.stage == DevelopmentalStage.REFLEXIVE

    def test_evaluate_no_promotion_early(self):
        dm = DevelopmentalManager()
        promoted = dm.evaluate_transition(cycle_count=100, mean_confidence=0.3)
        assert not promoted
        assert dm.stage == DevelopmentalStage.REFLEXIVE

    def test_evaluate_promotes_to_associative(self):
        dm = DevelopmentalManager()
        promoted = dm.evaluate_transition(cycle_count=20_000, mean_confidence=0.6)
        assert promoted
        assert dm.stage == DevelopmentalStage.ASSOCIATIVE

    def test_stage_never_regresses(self):
        dm = DevelopmentalManager(initial_stage=DevelopmentalStage.DELIBERATIVE)
        promoted = dm.evaluate_transition(cycle_count=100, mean_confidence=0.1)
        assert not promoted
        assert dm.stage == DevelopmentalStage.DELIBERATIVE

    def test_manual_promote_forward(self):
        dm = DevelopmentalManager()
        assert dm.manual_promote(DevelopmentalStage.REFLECTIVE)
        assert dm.stage == DevelopmentalStage.REFLECTIVE

    def test_manual_promote_backward_rejected(self):
        dm = DevelopmentalManager(initial_stage=DevelopmentalStage.DELIBERATIVE)
        assert not dm.manual_promote(DevelopmentalStage.REFLEXIVE)
        assert dm.stage == DevelopmentalStage.DELIBERATIVE


class TestCapabilityGating:
    def test_reflexive_no_phase_space(self):
        dm = DevelopmentalManager(initial_stage=DevelopmentalStage.REFLEXIVE)
        assert not dm.phase_space_enabled()
        assert not dm.counterfactual_enabled()
        assert not dm.setpoint_adaptation_enabled()

    def test_deliberative_has_phase_space(self):
        dm = DevelopmentalManager(initial_stage=DevelopmentalStage.DELIBERATIVE)
        assert dm.phase_space_enabled()
        assert not dm.counterfactual_enabled()
        assert dm.setpoint_adaptation_enabled()

    def test_reflective_full(self):
        dm = DevelopmentalManager(initial_stage=DevelopmentalStage.REFLECTIVE)
        assert dm.phase_space_enabled()
        assert dm.counterfactual_enabled()
        assert dm.setpoint_adaptation_enabled()

    def test_available_horizons_grow_with_stage(self):
        dm = DevelopmentalManager(initial_stage=DevelopmentalStage.REFLEXIVE)
        assert len(dm.available_horizons) == 2
        dm.manual_promote(DevelopmentalStage.ASSOCIATIVE)
        assert len(dm.available_horizons) == 4
        dm.manual_promote(DevelopmentalStage.DELIBERATIVE)
        assert len(dm.available_horizons) == 6
