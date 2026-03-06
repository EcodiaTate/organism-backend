"""
EcodiaOS — Nova Goal Manager

Goals are not tasks — they are desires. They have dynamic priorities that
shift with context, dependencies that create sequencing, and progress
tracking that knows when to celebrate or abandon.

Priority computation follows the spec exactly:
    priority = base_importance × 0.30
             + urgency_factor   × 0.25
             + drive_resonance  × 0.20
             + staleness_boost  (up to 0.10)
             + dep_factor       × 0.15

Drive resonance is the key theoretical innovation: goals aligned with
currently activated drives get a context-sensitive boost. This means a
care-aligned goal is more urgent when EOS detects distress, and a
growth-aligned goal is more salient when curiosity is high.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector, new_id, utc_now
from systems.nova.types import (
    Goal,
    GoalSource,
    GoalStatus,
    PriorityContext,
)

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from systems.atune.types import WorkspaceBroadcast

logger = structlog.get_logger()

# Maximum time without progress before staleness boost kicks in (seconds)
_STALENESS_THRESHOLD_SECONDS = 3600
# Goal is considered stale after this many days without evidence
_STALENESS_MAX_DAYS = 7.0
# Abandon goals that have been suspended for more than this many hours
_SUSPENSION_ABANDON_HOURS = 48.0


class GoalManager:
    """
    Manages the complete lifecycle of Nova's active goals.

    Responsibilities:
    - Create goals from various sources (user requests, care responses, epistemic drive)
    - Compute dynamic priorities each deliberation cycle
    - Update goal progress from intent outcomes
    - Retire goals that are achieved, abandoned, or superseded
    - Feed active goal summaries to Atune for salience weighting

    Goals are stored in-memory (fast access) with cap enforcement.
    """

    def __init__(self, max_active_goals: int = 20) -> None:
        self._max_active = max_active_goals
        self._goals: dict[str, Goal] = {}
        self._logger = logger.bind(system="nova.goal_manager")

    # ─── Public API ───────────────────────────────────────────────

    @property
    def active_goals(self) -> list[Goal]:
        return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    @property
    def all_goals(self) -> list[Goal]:
        return list(self._goals.values())

    def add_goal(self, goal: Goal) -> Goal:
        """
        Add a goal to the active set.
        If an active goal with the same description prefix (first 60 chars) and
        source already exists, return it without adding a duplicate.
        If at capacity, the lowest-priority maintenance goal is suspended first;
        if no maintenance goals exist, the lowest-priority active goal is suspended.
        """
        if goal.id in self._goals:
            return self._goals[goal.id]

        # Deduplication: block goals with same source + description prefix
        prefix = goal.description[:60]
        for existing in self._goals.values():
            if (
                existing.source == goal.source
                and existing.status == GoalStatus.ACTIVE
                and existing.description[:60] == prefix
            ):
                self._logger.debug(
                    "goal_deduplicated",
                    duplicate_id=goal.id,
                    existing_id=existing.id,
                    source=goal.source.value,
                    prefix=prefix[:40],
                )
                return existing

        active = self.active_goals
        if len(active) >= self._max_active:
            # Evict maintenance goals first; fall back to lowest-priority overall
            maintenance_goals = [g for g in active if g.source == GoalSource.MAINTENANCE]
            candidate = (
                min(maintenance_goals, key=lambda g: g.priority)
                if maintenance_goals
                else min(active, key=lambda g: g.priority)
            )
            suspended = candidate.model_copy(update={"status": GoalStatus.SUSPENDED})
            self._goals[candidate.id] = suspended
            self._logger.info(
                "goal_suspended_for_capacity",
                suspended_id=candidate.id,
                suspended_description=candidate.description[:60],
                was_maintenance=candidate.source == GoalSource.MAINTENANCE,
            )

        self._goals[goal.id] = goal
        self._logger.info(
            "goal_added",
            goal_id=goal.id,
            source=goal.source.value,
            description=goal.description[:80],
            priority=round(goal.priority, 3),
        )
        return goal

    def get_goal(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def mark_achieved(self, goal_id: str, episode_id: str = "") -> Goal | None:
        """Mark a goal as achieved and record evidence."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        evidence = list(goal.evidence_of_progress)
        if episode_id:
            evidence.append(episode_id)
        updated = goal.model_copy(update={
            "status": GoalStatus.ACHIEVED,
            "progress": 1.0,
            "evidence_of_progress": evidence,
        })
        self._goals[goal_id] = updated
        self._logger.info("goal_achieved", goal_id=goal_id, description=goal.description[:60])
        return updated

    def mark_abandoned(self, goal_id: str, reason: str = "") -> Goal | None:
        """Mark a goal as abandoned."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        updated = goal.model_copy(update={"status": GoalStatus.ABANDONED})
        self._goals[goal_id] = updated
        self._logger.info("goal_abandoned", goal_id=goal_id, reason=reason[:80])
        return updated

    def update_progress(self, goal_id: str, progress: float, episode_id: str = "") -> Goal | None:
        """Update a goal's progress estimate and optionally record evidence."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        evidence = list(goal.evidence_of_progress)
        if episode_id:
            evidence.append(episode_id)
        updated = goal.model_copy(
            update={
                "progress": max(goal.progress, min(1.0, progress)),
                "evidence_of_progress": evidence,
            }
        )
        self._goals[goal_id] = updated
        if progress >= 0.95:
            return self.mark_achieved(goal_id, episode_id)
        return updated

    def recompute_priorities(self, context: PriorityContext) -> None:
        """
        Recompute dynamic priorities for all active goals.
        Must complete in ≤30ms per spec.
        """
        for goal_id, goal in self._goals.items():
            if goal.status != GoalStatus.ACTIVE:
                continue
            new_priority = compute_goal_priority(goal, context)
            self._goals[goal_id] = goal.model_copy(update={"priority": new_priority})

    def find_relevant_goal(self, broadcast: WorkspaceBroadcast) -> Goal | None:
        """
        Find the most relevant active goal for a broadcast.

        Relevance is a combination of:
        - Goal domain match to broadcast content
        - Goal priority (already recomputed this cycle)
        - Whether the broadcast directly relates to an ongoing conversation
          associated with this goal

        Returns the highest-priority matching goal, or None.
        """
        active = self.active_goals
        if not active:
            return None

        content_text = _extract_broadcast_text(broadcast)

        # Score each active goal by relevance
        scored: list[tuple[float, Goal]] = []
        for goal in active:
            relevance = _compute_goal_relevance(goal, content_text, broadcast.precision)
            # Weight by priority: high-priority goals should be served first
            combined = relevance * 0.6 + goal.priority * 0.4
            scored.append((combined, goal))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Return the top goal only if it's meaningfully relevant
        if scored and scored[0][0] > 0.2:
            return scored[0][1]
        return None

    def create_from_broadcast(self, broadcast: WorkspaceBroadcast) -> Goal | None:
        """
        Create a new goal from a broadcast that doesn't match existing goals.
        Returns None if the broadcast doesn't warrant a new goal.
        """
        content_text = _extract_broadcast_text(broadcast)
        if not content_text:
            return None

        # Only create goals for reasonably salient broadcasts
        if broadcast.salience.composite < 0.3:
            return None

        source, drive_alignment, importance = _classify_goal_source(broadcast)
        description = _synthesise_goal_description(content_text, source)
        if not description:
            return None

        goal = Goal(
            id=new_id(),
            description=description,
            source=source,
            priority=_initial_priority(broadcast, importance),
            urgency=_initial_urgency(broadcast),
            importance=importance,
            drive_alignment=drive_alignment,
            status=GoalStatus.ACTIVE,
        )
        return self.add_goal(goal)

    def expire_stale_goals(self) -> list[Goal]:
        """
        Abandon goals that have been suspended too long.
        Returns the list of abandoned goals.
        """
        now = utc_now()
        abandoned: list[Goal] = []
        for goal in list(self._goals.values()):
            if goal.status != GoalStatus.SUSPENDED:
                continue
            age_hours = (now - goal.created_at).total_seconds() / 3600
            if age_hours > _SUSPENSION_ABANDON_HOURS:
                updated = self.mark_abandoned(goal.id, reason="suspended_too_long")
                if updated:
                    abandoned.append(updated)
        return abandoned

    def prune_retired_goals(self, max_retired: int = 50) -> int:
        """
        Remove old achieved/abandoned goals from the dictionary to prevent
        unbounded memory growth. Keeps the most recent ``max_retired`` retired
        goals for observability; drops anything older.

        Returns the count of goals pruned.
        """
        retired = [
            g for g in self._goals.values()
            if g.status in (GoalStatus.ACHIEVED, GoalStatus.ABANDONED)
        ]
        if len(retired) <= max_retired:
            return 0

        # Sort by creation time, keep newest
        retired.sort(key=lambda g: g.created_at, reverse=True)
        to_remove = retired[max_retired:]

        for g in to_remove:
            del self._goals[g.id]

        if to_remove:
            self._logger.info("goals_pruned", count=len(to_remove))
        return len(to_remove)

    def stats(self) -> dict[str, Any]:
        counts = {s.value: 0 for s in GoalStatus}
        for g in self._goals.values():
            counts[g.status.value] += 1
        return {
            "total": len(self._goals),
            **counts,
        }


# ─── Priority Computation ─────────────────────────────────────────


def compute_goal_priority(goal: Goal, context: PriorityContext) -> float:
    """
    Dynamic priority computation from the spec.

    priority = base × 0.30 + urgency_factor × 0.25 + drive_resonance × 0.20
             + staleness_boost + dep_factor × 0.15
    """

    # ── Base importance (constitutional alignment) ──
    base = goal.importance

    # ── Urgency factor (deadline-sensitive) ──
    if goal.deadline:
        now = utc_now()
        remaining = (goal.deadline - now).total_seconds()
        if remaining <= 0:
            urgency_factor = 1.0      # Overdue
        elif remaining < 3600:
            urgency_factor = 0.9      # < 1 hour
        elif remaining < 86400:
            urgency_factor = 0.6      # < 1 day
        else:
            urgency_factor = 0.3
    else:
        urgency_factor = goal.urgency

    # ── Drive resonance ──
    resonance = compute_drive_resonance(
        alignment=goal.drive_alignment,
        affect=context.current_affect,
        drive_weights=context.drive_weights,
    )

    # ── Staleness boost ──
    staleness_boost = 0.05  # Default for goals with no progress yet
    if goal.evidence_of_progress and context.episode_timestamps:
        last_times = [
            context.episode_timestamps[eid]
            for eid in goal.evidence_of_progress
            if eid in context.episode_timestamps
        ]
        if last_times:
            last_progress = max(last_times)
            days_stale = (utc_now() - last_progress).total_seconds() / 86400
            # Gently increase priority for stagnant goals
            staleness_boost = min(0.10, days_stale * 0.02)

    # ── Dependency readiness ──
    deps_met = all(
        context.goal_statuses.get(dep_id) == GoalStatus.ACHIEVED.value
        for dep_id in goal.depends_on
    )
    dep_factor = 1.0 if (deps_met or not goal.depends_on) else 0.5

    priority = (
        base * 0.30
        + urgency_factor * 0.25
        + resonance * 0.20
        + staleness_boost
        + dep_factor * 0.15
    )

    return _clamp(priority, 0.0, 1.0)


def compute_drive_resonance(
    alignment: DriveAlignmentVector,
    affect: AffectState,
    drive_weights: dict[str, float],
) -> float:
    """
    How well does this goal's drive alignment resonate with the current state?

    Care-aligned goals are boosted when care_activation is high.
    Growth-aligned goals are boosted when curiosity is high.
    Coherence-aligned goals are boosted when coherence_stress is high.
    Honesty is ambient — always provides a moderate baseline.
    """
    w_coherence = drive_weights.get("coherence", 1.0)
    w_care = drive_weights.get("care", 1.0)
    w_growth = drive_weights.get("growth", 1.0)
    w_honesty = drive_weights.get("honesty", 1.0)

    resonance = (
        alignment.coherence * affect.coherence_stress * w_coherence
        + alignment.care * affect.care_activation * w_care
        + alignment.growth * affect.curiosity * w_growth
        + alignment.honesty * 0.5 * w_honesty  # Ambient honesty baseline
    )
    return _clamp(resonance, 0.0, 1.0)


# ─── Helpers ─────────────────────────────────────────────────────


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _extract_broadcast_text(broadcast: WorkspaceBroadcast) -> str:
    content = broadcast.content
    paths = [
        ("content", "content", "content"),
        ("content", "content"),
        ("content",),
    ]
    for path in paths:
        obj: object = content
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, str) and obj:
            return obj[:500]
    return ""


def _compute_goal_relevance(
    goal: Goal,
    content_text: str,
    precision: float,
) -> float:
    """Keyword-based relevance score between goal description and broadcast text."""
    if not content_text:
        return 0.0

    # Tokenise both (simple word overlap)
    goal_words = set(goal.description.lower().split())
    content_words = set(content_text.lower().split())
    # Remove stop words
    stops = {"the", "a", "an", "is", "it", "in", "on", "and", "or", "to", "of", "for"}
    goal_words -= stops
    content_words -= stops

    if not goal_words:
        return 0.0

    overlap = len(goal_words & content_words)
    jaccard = overlap / len(goal_words | content_words)

    # Weight by precision: higher precision broadcasts are more likely to be relevant
    return _clamp(jaccard * 2.0 * precision, 0.0, 1.0)


def _classify_goal_source(
    broadcast: WorkspaceBroadcast,
) -> tuple[GoalSource, DriveAlignmentVector, float]:
    """Classify what kind of goal a broadcast suggests."""
    affect = broadcast.affect
    salience_scores = broadcast.salience.scores if broadcast.salience.scores else {}

    # Distress signal → care goal
    if affect.care_activation > 0.7 or salience_scores.get("emotional", 0) > 0.7:
        return (
            GoalSource.CARE_RESPONSE,
            DriveAlignmentVector(coherence=0.1, care=0.9, growth=0.0, honesty=0.1),
            0.8,  # High importance
        )

    # Highly novel → epistemic goal
    if salience_scores.get("novelty", 0) > 0.7 and affect.curiosity > 0.5:
        return (
            GoalSource.EPISTEMIC,
            DriveAlignmentVector(coherence=0.4, care=0.0, growth=0.5, honesty=0.1),
            0.3,
        )

    # User request (default for dialogue)
    return (
        GoalSource.USER_REQUEST,
        DriveAlignmentVector(coherence=0.3, care=0.3, growth=0.1, honesty=0.1),
        0.6,
    )


def _synthesise_goal_description(content: str, source: GoalSource) -> str:
    """Generate a concise goal description from broadcast content."""
    content_brief = content[:120].strip()
    if not content_brief:
        return ""

    prefix = {
        GoalSource.CARE_RESPONSE: "Address and support: ",
        GoalSource.EPISTEMIC: "Investigate and understand: ",
        GoalSource.USER_REQUEST: "Respond helpfully to: ",
        GoalSource.SELF_GENERATED: "Self-generated goal: ",
        GoalSource.GOVERNANCE: "Governance goal: ",
        GoalSource.MAINTENANCE: "Maintenance: ",
    }
    return prefix.get(source, "Goal: ") + content_brief


def _initial_priority(broadcast: WorkspaceBroadcast, importance: float) -> float:
    """Estimate initial goal priority from broadcast salience."""
    return _clamp(
        importance * 0.5 + broadcast.salience.composite * 0.5,
        0.1,
        0.95,
    )


def _initial_urgency(broadcast: WorkspaceBroadcast) -> float:
    """Estimate initial goal urgency from affect and salience."""
    affect = broadcast.affect
    return _clamp(
        affect.care_activation * 0.5 + broadcast.salience.composite * 0.3 + affect.arousal * 0.2,
        0.0,
        1.0,
    )

