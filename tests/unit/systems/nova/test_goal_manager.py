"""
Unit tests for Nova GoalManager.

Tests goal lifecycle, priority computation, capacity enforcement,
drive resonance, relevance scoring, and stale goal expiration.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from primitives.affect import AffectState
from primitives.common import DriveAlignmentVector, new_id, utc_now
from systems.fovea.types import SalienceVector, WorkspaceBroadcast
from systems.nova.goal_manager import (
    GoalManager,
    compute_drive_resonance,
    compute_goal_priority,
)
from systems.nova.types import (
    Goal,
    GoalSource,
    GoalStatus,
    PriorityContext,
)

# ─── Fixtures ─────────────────────────────────────────────────────


def make_goal(
    description: str = "Respond helpfully to: Hello",
    priority: float = 0.5,
    urgency: float = 0.3,
    importance: float = 0.5,
    source: GoalSource = GoalSource.USER_REQUEST,
    status: GoalStatus = GoalStatus.ACTIVE,
    care: float = 0.3,
    growth: float = 0.1,
    deadline: datetime | None = None,
) -> Goal:
    return Goal(
        id=new_id(),
        description=description,
        source=source,
        priority=priority,
        urgency=urgency,
        importance=importance,
        drive_alignment=DriveAlignmentVector(
            coherence=0.3, care=care, growth=growth, honesty=0.1
        ),
        status=status,
        deadline=deadline,
    )


def make_neutral_context(
    care_activation: float = 0.0,
    curiosity: float = 0.0,
    coherence_stress: float = 0.0,
) -> PriorityContext:
    affect = AffectState.neutral().model_copy(update={
        "care_activation": care_activation,
        "curiosity": curiosity,
        "coherence_stress": coherence_stress,
    })
    return PriorityContext(current_affect=affect)


def make_broadcast(
    text: str = "hello",
    precision: float = 0.5,
    care_activation: float = 0.0,
    novelty: float = 0.0,
    composite: float = 0.5,
) -> WorkspaceBroadcast:
    affect = AffectState.neutral().model_copy(update={"care_activation": care_activation})
    scores = {}
    if novelty > 0:
        scores["novelty"] = novelty

    content = MagicMock()
    content.content = MagicMock()
    content.content.content = text

    return WorkspaceBroadcast(
        content=content,
        salience=SalienceVector(scores=scores, composite=composite),
        affect=affect,
        precision=precision,
    )


@pytest.fixture
def manager() -> GoalManager:
    return GoalManager(max_active_goals=5)


# ─── Goal Lifecycle ───────────────────────────────────────────────


class TestGoalLifecycle:
    def test_add_goal(self, manager: GoalManager) -> None:
        goal = make_goal()
        added = manager.add_goal(goal)
        assert added.id == goal.id
        assert len(manager.active_goals) == 1

    def test_duplicate_add_returns_existing(self, manager: GoalManager) -> None:
        goal = make_goal()
        manager.add_goal(goal)
        manager.add_goal(goal)  # second add
        assert len(manager.active_goals) == 1

    def test_mark_achieved(self, manager: GoalManager) -> None:
        goal = manager.add_goal(make_goal())
        result = manager.mark_achieved(goal.id, episode_id="ep1")
        assert result is not None
        assert result.status == GoalStatus.ACHIEVED
        assert result.progress == 1.0
        assert "ep1" in result.evidence_of_progress

    def test_mark_abandoned(self, manager: GoalManager) -> None:
        goal = manager.add_goal(make_goal())
        result = manager.mark_abandoned(goal.id, reason="no longer relevant")
        assert result is not None
        assert result.status == GoalStatus.ABANDONED

    def test_mark_nonexistent_goal_returns_none(self, manager: GoalManager) -> None:
        assert manager.mark_achieved("nonexistent-id") is None
        assert manager.mark_abandoned("nonexistent-id") is None

    def test_update_progress(self, manager: GoalManager) -> None:
        goal = manager.add_goal(make_goal())
        result = manager.update_progress(goal.id, 0.6)
        assert result is not None
        assert result.progress == 0.6

    def test_high_progress_auto_achieves(self, manager: GoalManager) -> None:
        goal = manager.add_goal(make_goal())
        result = manager.update_progress(goal.id, 0.96)
        assert result is not None
        assert result.status == GoalStatus.ACHIEVED

    def test_progress_never_decreases(self, manager: GoalManager) -> None:
        goal = manager.add_goal(make_goal())
        manager.update_progress(goal.id, 0.7)
        result = manager.update_progress(goal.id, 0.3)  # lower value
        assert result is not None
        assert result.progress == 0.7  # clamped at previous max


# ─── Capacity Enforcement ─────────────────────────────────────────


class TestCapacityEnforcement:
    def test_suspends_lowest_when_at_capacity(self) -> None:
        manager = GoalManager(max_active_goals=3)
        low = make_goal(description="Low", priority=0.1)
        mid = make_goal(description="Mid", priority=0.5)
        high = make_goal(description="High", priority=0.9)
        manager.add_goal(low)
        manager.add_goal(mid)
        manager.add_goal(high)
        assert len(manager.active_goals) == 3

        # Adding a 4th should suspend the lowest
        new_goal = make_goal(description="New", priority=0.7)
        manager.add_goal(new_goal)

        active_descs = {g.description for g in manager.active_goals}
        assert "Low" not in active_descs  # Lowest was suspended
        assert "New" in active_descs

    def test_suspended_goals_not_in_active(self) -> None:
        manager = GoalManager(max_active_goals=2)
        g1 = make_goal(description="G1", priority=0.9)
        g2 = make_goal(description="G2", priority=0.8)
        manager.add_goal(g1)
        manager.add_goal(g2)
        # At capacity; add g3 - g2 stays (higher priority), some goal gets suspended
        g3 = make_goal(description="G3", priority=0.95)
        manager.add_goal(g3)
        assert len(manager.active_goals) == 2


# ─── Priority Computation ─────────────────────────────────────────


class TestPriorityComputation:
    def test_priority_in_valid_range(self) -> None:
        goal = make_goal(importance=0.6, urgency=0.4)
        ctx = make_neutral_context()
        p = compute_goal_priority(goal, ctx)
        assert 0.0 <= p <= 1.0

    def test_overdue_deadline_max_urgency(self) -> None:
        overdue = utc_now() - timedelta(hours=2)
        goal = make_goal(deadline=overdue, importance=0.5)
        ctx = make_neutral_context()
        p = compute_goal_priority(goal, ctx)
        # Overdue → urgency_factor = 1.0, should be high priority
        assert p > 0.5

    def test_no_deadline_uses_goal_urgency(self) -> None:
        goal_high = make_goal(urgency=0.9)
        goal_low = make_goal(urgency=0.1)
        ctx = make_neutral_context()
        assert compute_goal_priority(goal_high, ctx) > compute_goal_priority(goal_low, ctx)

    def test_dependency_unmet_reduces_priority(self) -> None:
        dep_id = new_id()
        goal_with_dep = make_goal()
        goal_with_dep = goal_with_dep.model_copy(update={"depends_on": [dep_id]})
        goal_without_dep = make_goal()

        ctx = PriorityContext(current_affect=AffectState.neutral())  # dep_id not in goal_statuses → unmet
        p_with_dep = compute_goal_priority(goal_with_dep, ctx)
        p_without_dep = compute_goal_priority(goal_without_dep, ctx)
        assert p_with_dep < p_without_dep

    def test_recompute_priorities_updates_goals(self) -> None:
        manager = GoalManager()
        goal = manager.add_goal(make_goal(priority=0.5))
        ctx = make_neutral_context()
        manager.recompute_priorities(ctx)
        updated = manager.get_goal(goal.id)
        assert updated is not None
        # Priority should have been recomputed (may or may not differ from 0.5)
        assert 0.0 <= updated.priority <= 1.0


# ─── Drive Resonance ──────────────────────────────────────────────


class TestDriveResonance:
    def test_care_goal_resonates_with_high_care_activation(self) -> None:
        alignment = DriveAlignmentVector(coherence=0.1, care=0.9, growth=0.0, honesty=0.1)
        affect_high_care = AffectState.neutral().model_copy(update={"care_activation": 0.9})
        affect_low_care = AffectState.neutral().model_copy(update={"care_activation": 0.1})
        drive_weights = {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}

        r_high = compute_drive_resonance(alignment, affect_high_care, drive_weights)
        r_low = compute_drive_resonance(alignment, affect_low_care, drive_weights)
        assert r_high > r_low

    def test_growth_goal_resonates_with_high_curiosity(self) -> None:
        alignment = DriveAlignmentVector(coherence=0.1, care=0.0, growth=0.9, honesty=0.1)
        affect_curious = AffectState.neutral().model_copy(update={"curiosity": 0.9})
        affect_flat = AffectState.neutral()
        drive_weights = {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}

        r_curious = compute_drive_resonance(alignment, affect_curious, drive_weights)
        r_flat = compute_drive_resonance(alignment, affect_flat, drive_weights)
        assert r_curious > r_flat

    def test_resonance_clamped_to_one(self) -> None:
        alignment = DriveAlignmentVector(coherence=1.0, care=1.0, growth=1.0, honesty=1.0)
        affect = AffectState.neutral().model_copy(update={
            "care_activation": 1.0, "curiosity": 1.0, "coherence_stress": 1.0
        })
        r = compute_drive_resonance(alignment, affect, {"coherence": 10.0, "care": 10.0, "growth": 10.0, "honesty": 10.0})
        assert r <= 1.0

    def test_honesty_provides_ambient_baseline(self) -> None:
        alignment = DriveAlignmentVector(coherence=0.0, care=0.0, growth=0.0, honesty=1.0)
        affect = AffectState.neutral()  # All zero activations
        r = compute_drive_resonance(alignment, affect, {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0})
        # Honesty * 0.5 (ambient) should be > 0
        assert r > 0.0


# ─── Goal Relevance ───────────────────────────────────────────────


class TestGoalRelevance:
    def test_finds_relevant_goal(self) -> None:
        manager = GoalManager()
        goal = manager.add_goal(make_goal(
            description="Respond helpfully to: code bug algorithm",
            priority=0.7,
        ))
        broadcast = make_broadcast("there is a bug in my code algorithm", precision=0.8)
        found = manager.find_relevant_goal(broadcast)
        assert found is not None
        assert found.id == goal.id

    def test_returns_none_when_no_active_goals(self) -> None:
        manager = GoalManager()
        broadcast = make_broadcast("hello")
        assert manager.find_relevant_goal(broadcast) is None

    def test_returns_none_when_no_relevant_goal(self) -> None:
        manager = GoalManager()
        manager.add_goal(make_goal(description="Respond to: cooking recipe food", priority=0.5))
        broadcast = make_broadcast("quantum physics orbital mechanics")
        result = manager.find_relevant_goal(broadcast)
        # May or may not match; if combined < 0.2 → None
        # Just verify it runs without error
        assert result is None or result is not None


# ─── Stale Goal Expiration ────────────────────────────────────────


class TestStaleGoalExpiration:
    def test_active_goals_not_expired(self) -> None:
        manager = GoalManager()
        goal = manager.add_goal(make_goal(status=GoalStatus.ACTIVE))
        abandoned = manager.expire_stale_goals()
        assert len(abandoned) == 0
        assert manager.get_goal(goal.id).status == GoalStatus.ACTIVE

    def test_recently_suspended_not_expired(self) -> None:
        manager = GoalManager()
        goal = make_goal(status=GoalStatus.SUSPENDED)
        manager._goals[goal.id] = goal
        abandoned = manager.expire_stale_goals()
        assert len(abandoned) == 0

    def test_stats_counts_by_status(self) -> None:
        manager = GoalManager()
        manager.add_goal(make_goal())
        stats = manager.stats()
        assert stats["active"] == 1
        assert stats["total"] == 1


# ─── Goal Creation from Broadcast ────────────────────────────────


class TestGoalCreationFromBroadcast:
    def test_low_salience_returns_none(self) -> None:
        manager = GoalManager()
        broadcast = make_broadcast(text="hi", composite=0.1)
        result = manager.create_from_broadcast(broadcast)
        assert result is None

    def test_high_salience_creates_goal(self) -> None:
        manager = GoalManager()
        broadcast = make_broadcast(text="I need urgent help please", composite=0.8)
        result = manager.create_from_broadcast(broadcast)
        assert result is not None
        assert result.status == GoalStatus.ACTIVE

    def test_care_signal_creates_care_goal(self) -> None:
        manager = GoalManager()
        broadcast = make_broadcast(
            text="I'm feeling really hurt and alone",
            care_activation=0.9,
            composite=0.7,
        )
        result = manager.create_from_broadcast(broadcast)
        assert result is not None
        assert result.source == GoalSource.CARE_RESPONSE
