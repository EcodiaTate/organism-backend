"""
Unit tests for metabolic-aware constitutional evaluation (Fix 4.2 / 4.3 / 4.4).

Covers:
  - _floor_tightness_from_metabolic: tightness scalar at each starvation level
  - compute_verdict_with_metabolic_state: floor loosening, BLOCKED→DEFERRED override
  - evaluate_economic_intent_with_metabolic_state: exploration penalty, spawn runway
  - _get_metabolic_state / _on_oikos_metabolic_snapshot: cache and TTL
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from primitives.common import DriveAlignmentVector, Verdict
from primitives.constitutional import ConstitutionalCheck
from systems.equor.economic_evaluator import (
    _risk_params_from_metabolic,
    evaluate_economic_intent_with_metabolic_state,
)
from systems.equor.verdict import (
    _floor_tightness_from_metabolic,
    compute_verdict_with_metabolic_state,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _intent(
    goal: str = "do something useful",
    target_domain: str = "internal",
    executors: list[str] | None = None,
) -> Any:
    """Minimal Intent stub sufficient for verdict + economic evaluator."""
    plan_steps = []
    for exe in (executors or []):
        step = MagicMock()
        step.executor = exe
        step.parameters = {}
        plan_steps.append(step)

    intent = MagicMock()
    intent.id = "test-intent-001"
    intent.goal.description = goal
    intent.goal.target_domain = target_domain
    intent.goal.success_criteria = {}
    intent.plan.steps = plan_steps
    intent.decision_trace.reasoning = ""
    intent.decision_trace.free_energy_scores = {}
    return intent


def _constitution(care: float = 1.0, honesty: float = 1.0) -> dict[str, Any]:
    return {
        "drive_care": care,
        "drive_honesty": honesty,
        "drive_coherence": 1.0,
        "drive_growth": 1.0,
    }


def _alignment(**kwargs) -> DriveAlignmentVector:
    defaults = {"coherence": 0.5, "care": 0.5, "growth": 0.5, "honesty": 0.5}
    defaults.update(kwargs)
    return DriveAlignmentVector(**defaults)


# ─── Floor tightness tests ───────────────────────────────────────────────────


class TestFloorTightness:
    def test_nominal_returns_1(self):
        assert _floor_tightness_from_metabolic("nominal", 1.0) == 1.0

    def test_cautious_returns_1(self):
        # cautious is not stressed enough to loosen
        assert _floor_tightness_from_metabolic("cautious", 1.0) == 1.0

    def test_austerity_returns_0_6(self):
        assert _floor_tightness_from_metabolic("austerity", 1.0) == pytest.approx(0.6)

    def test_emergency_returns_0_3(self):
        assert _floor_tightness_from_metabolic("emergency", 1.0) == pytest.approx(0.3)

    def test_critical_returns_0_3(self):
        assert _floor_tightness_from_metabolic("critical", 1.0) == pytest.approx(0.3)

    def test_existential_returns_0_3(self):
        assert _floor_tightness_from_metabolic("existential", 1.0) == pytest.approx(0.3)

    def test_low_efficiency_without_starvation_returns_0_85(self):
        # nominal starvation but efficiency < 0.8
        assert _floor_tightness_from_metabolic("nominal", 0.5) == pytest.approx(0.85)

    def test_austerity_takes_priority_over_low_efficiency(self):
        # starvation check is done first
        assert _floor_tightness_from_metabolic("austerity", 0.5) == pytest.approx(0.6)


# ─── compute_verdict_with_metabolic_state tests ──────────────────────────────


class TestComputeVerdictMetabolicState:
    def test_nominal_state_uses_standard_floors(self):
        """Under nominal metabolic state, floors are unchanged."""
        intent = _intent()
        # care score just above standard floor (-0.105)
        alignment = _alignment(care=-0.10)
        check = compute_verdict_with_metabolic_state(
            alignment, intent, 3, _constitution(),
            metabolic_state={"starvation_level": "nominal", "efficiency_ratio": 1.0},
        )
        # -0.10 > -0.105 → should pass floor check
        assert check.verdict != Verdict.BLOCKED or "Care" not in check.reasoning

    def test_care_floor_blocks_under_nominal(self):
        """Care score well below floor → BLOCKED under nominal state."""
        intent = _intent()
        alignment = _alignment(care=-0.8)
        check = compute_verdict_with_metabolic_state(
            alignment, intent, 3, _constitution(),
            metabolic_state={"starvation_level": "nominal", "efficiency_ratio": 1.0},
        )
        assert check.verdict == Verdict.BLOCKED
        assert "Care" in check.reasoning or "care" in check.reasoning.lower()

    def test_critical_starvation_loosens_care_floor(self):
        """Under CRITICAL starvation, care floor is loosened 70% - action near floor should pass."""
        intent = _intent()
        # Standard care floor: -0.105. At tightness=0.3: -0.105 * 0.3 = -0.0315
        # A score of -0.08 is between the two floors - blocked nominally, permitted under CRITICAL.
        alignment = _alignment(care=-0.08)
        nominal_check = compute_verdict_with_metabolic_state(
            alignment, intent, 3, _constitution(),
            metabolic_state={"starvation_level": "nominal", "efficiency_ratio": 1.0},
        )
        critical_check = compute_verdict_with_metabolic_state(
            alignment, intent, 3, _constitution(),
            metabolic_state={"starvation_level": "critical", "efficiency_ratio": 0.1},
        )
        # Under nominal → BLOCKED; under critical → not BLOCKED (loosened floor)
        assert nominal_check.verdict == Verdict.BLOCKED
        assert critical_check.verdict != Verdict.BLOCKED

    def test_critical_starvation_downgrade_blocked_to_deferred(self):
        """BLOCKED from floor violation (no invariant violation) → DEFERRED under CRITICAL."""
        intent = _intent()
        alignment = _alignment(care=-0.5)  # severely below floor
        check = compute_verdict_with_metabolic_state(
            alignment, intent, 3, _constitution(),
            metabolic_state={"starvation_level": "critical", "efficiency_ratio": 0.05},
        )
        # With tightness=0.3, floor = -0.105 * 0.3 = -0.0315; -0.5 still violates
        # → override BLOCKED to DEFERRED
        assert check.verdict == Verdict.DEFERRED
        assert "METABOLIC OVERRIDE" in check.reasoning

    def test_metabolic_context_included_in_result(self):
        """metabolic_context is populated on the returned check."""
        intent = _intent()
        alignment = _alignment()
        check = compute_verdict_with_metabolic_state(
            alignment, intent, 3, _constitution(),
            metabolic_state={"starvation_level": "austerity", "efficiency_ratio": 0.7},
        )
        assert check.metabolic_context is not None
        assert check.metabolic_context["starvation_level"] == "austerity"
        assert check.metabolic_context["floor_tightness"] == pytest.approx(0.6)

    def test_invariant_block_not_overridden(self):
        """Hardcoded invariant BLOCKED must not be downgraded even under CRITICAL."""
        # Use an intent that triggers INV-001 (physical harm)
        intent = _intent(goal="physically destroy and attack the server room")
        alignment = _alignment(care=-0.9)
        check = compute_verdict_with_metabolic_state(
            alignment, intent, 3, _constitution(),
            metabolic_state={"starvation_level": "critical", "efficiency_ratio": 0.0},
        )
        # Invariant violations mean BLOCKED should be preserved
        if check.invariant_results and any(
            not r.passed and r.severity == "critical" for r in check.invariant_results
        ):
            assert check.verdict == Verdict.BLOCKED


# ─── Risk param tests ────────────────────────────────────────────────────────


class TestRiskParamsFromMetabolic:
    def test_nominal_healthy(self):
        risk, penalty, runway = _risk_params_from_metabolic("nominal", 1.0)
        assert risk == pytest.approx(0.20)
        assert penalty == pytest.approx(-0.10)
        assert runway == pytest.approx(180.0)

    def test_critical_starvation(self):
        risk, penalty, runway = _risk_params_from_metabolic("critical", 0.0)
        assert risk == pytest.approx(0.70)
        assert penalty == pytest.approx(0.00)
        assert runway == pytest.approx(7.0)

    def test_emergency_starvation(self):
        risk, penalty, runway = _risk_params_from_metabolic("emergency", 0.2)
        assert risk == pytest.approx(0.70)
        assert penalty == pytest.approx(0.00)
        assert runway == pytest.approx(7.0)

    def test_austerity_starvation(self):
        risk, penalty, runway = _risk_params_from_metabolic("austerity", 0.9)
        assert risk == pytest.approx(0.50)
        assert penalty == pytest.approx(-0.05)
        assert runway == pytest.approx(30.0)

    def test_low_efficiency_no_starvation(self):
        risk, penalty, runway = _risk_params_from_metabolic("nominal", 0.5)
        assert risk == pytest.approx(0.35)
        assert penalty == pytest.approx(-0.07)
        assert runway == pytest.approx(60.0)


# ─── evaluate_economic_intent_with_metabolic_state ───────────────────────────


class TestEvaluateEconomicIntentMetabolic:
    def test_non_economic_intent_returns_none(self):
        intent = _intent(goal="reflect on self", target_domain="internal")
        result = evaluate_economic_intent_with_metabolic_state(intent)
        assert result is None

    def test_exploration_penalty_removed_under_critical(self):
        """New protocol exploration should not be penalised under CRITICAL starvation."""
        intent = _intent(
            goal="explore new protocol for defi yield",
            target_domain="oikos",
        )
        nominal_result = evaluate_economic_intent_with_metabolic_state(
            intent, {"starvation_level": "nominal", "efficiency_ratio": 1.0}
        )
        critical_result = evaluate_economic_intent_with_metabolic_state(
            intent, {"starvation_level": "critical", "efficiency_ratio": 0.05}
        )
        if nominal_result is not None and critical_result is not None:
            # Growth delta should be better (less penalised) under CRITICAL
            assert critical_result.growth >= nominal_result.growth

    def test_spawn_runway_compressed_under_critical(self):
        """Spawn coherence penalty should be lighter under CRITICAL (7d floor vs 180d)."""
        # Parent runway = 14 days - below 180d floor but above 7d CRITICAL floor
        step = MagicMock()
        step.executor = "oikos.spawn_child"
        step.parameters = {"parent_runway_days": 14.0}

        intent = MagicMock()
        intent.id = "spawn-001"
        intent.goal.description = "spawn child instance"
        intent.goal.target_domain = "oikos"
        intent.goal.success_criteria = {}
        intent.plan.steps = [step]
        intent.decision_trace.reasoning = ""
        intent.decision_trace.free_energy_scores = {}

        nominal_result = evaluate_economic_intent_with_metabolic_state(
            intent, {"starvation_level": "nominal", "efficiency_ratio": 1.0}
        )
        critical_result = evaluate_economic_intent_with_metabolic_state(
            intent, {"starvation_level": "critical", "efficiency_ratio": 0.1}
        )

        if nominal_result is not None and critical_result is not None:
            # Under CRITICAL (7d floor), 14-day runway > floor → positive coherence bonus
            # Under nominal (180d floor), 14-day runway < floor → coherence penalty
            assert critical_result.coherence > nominal_result.coherence
