"""
EcodiaOS - Evo Self-Model Manager

Maintains the instance's evolving understanding of its own capabilities
and effectiveness. This is meta-cognition: EOS learning about EOS.

Updated during each consolidation cycle (spec Section VIII):
  1. Retrieve recent action outcomes from Memory
  2. Compute per-capability success rates
  3. Compute overall success rate and mean constitutional alignment
  4. Store updated stats on the Self node

The self-model feeds back into:
  - Nova: informs feasibility estimates in EFE scoring
  - Equor: provides effectiveness data for drift detection
  - Evo: drives self-model hypotheses (category = "self_model")

Performance: self-model update ≤5s (spec Section X).
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.evo.types import CapabilityScore, RegretStats, SelfModelStats

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

_OUTCOME_WINDOW: int = 500        # How many recent outcomes to analyse
_OUTCOME_HORIZON_DAYS: int = 30   # Only look at the last 30 days


class SelfModelManager:
    """
    Computes and maintains the instance's self-model from recent outcomes.

    Reads from the Memory graph; writes updated stats back to the Self node.
    """

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="evo.self_model")
        self._current: SelfModelStats = SelfModelStats()

    async def update(self) -> SelfModelStats:
        """
        Recompute the self-model from the last _OUTCOME_WINDOW outcomes.
        Also mines resolved counterfactual episodes for regret statistics.
        Persists results to the Self node.
        Returns the updated SelfModelStats.
        """
        if self._memory is None:
            return self._current

        try:
            outcomes = await self._fetch_recent_outcomes()
            if not outcomes:
                return self._current

            stats = self._compute_stats(outcomes)

            # Mine counterfactual regret from resolved shadow episodes
            regret = await self._compute_regret_stats()
            stats = stats.model_copy(update={"regret": regret})

            self._current = stats

            await self._persist_stats(stats)

            self._logger.info(
                "self_model_updated",
                success_rate=round(stats.success_rate, 3),
                mean_alignment=round(stats.mean_alignment, 3),
                outcomes_evaluated=stats.total_outcomes_evaluated,
                capabilities=len(stats.capability_scores),
                mean_regret=round(regret.mean_regret, 3),
                total_counterfactuals_resolved=regret.total_resolved,
                high_regret_count=regret.high_regret_count,
            )
            return stats

        except Exception as exc:
            self._logger.error("self_model_update_failed", error=str(exc))
            return self._current

    def get_current(self) -> SelfModelStats:
        """Return the most recently computed self-model stats."""
        return self._current

    def get_capability_rate(self, capability: str) -> float | None:
        """
        Return the success rate for a specific capability, or None if unknown.
        Used by Nova's feasibility estimator.
        """
        score = self._current.capability_scores.get(capability)
        return score.rate if score else None

    def get_regret_stats(self) -> RegretStats:
        """Return the most recently computed regret statistics."""
        return self._current.regret

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _compute_regret_stats(self) -> RegretStats:
        """
        Aggregate regret from resolved counterfactual episodes.

        Queries :Counterfactual nodes that have been resolved with outcome data
        within the outcome horizon. Computes mean regret per policy type and
        per goal domain, detecting systematic biases in policy selection.
        """
        cutoff = (utc_now() - timedelta(days=_OUTCOME_HORIZON_DAYS)).isoformat()
        try:
            results = await self._memory.execute_read(  # type: ignore[union-attr]
                """
                MATCH (cf:Counterfactual)
                WHERE cf.resolved = true
                  AND cf.regret IS NOT NULL
                  AND cf.event_time >= datetime($cutoff)
                RETURN
                  cf.policy_type AS policy_type,
                  cf.goal_id AS goal_id,
                  cf.regret AS regret,
                  cf.estimated_pragmatic_value AS est_pragmatic,
                  cf.estimated_epistemic_value AS est_epistemic
                ORDER BY cf.event_time DESC
                LIMIT $limit
                """,
                {"cutoff": cutoff, "limit": _OUTCOME_WINDOW},
            )
        except Exception as exc:
            self._logger.warning("regret_fetch_failed", error=str(exc))
            return RegretStats()

        if not results:
            return RegretStats()

        # Aggregate regret per policy type and per goal domain
        total_regret = 0.0
        high_regret = 0
        by_type: dict[str, list[float]] = {}
        by_domain: dict[str, list[float]] = {}

        for row in results:
            regret = float(row.get("regret", 0.0))
            total_regret += regret
            if abs(regret) > 0.3:
                high_regret += 1

            ptype = str(row.get("policy_type", "unknown"))
            by_type.setdefault(ptype, []).append(regret)

            goal_id = str(row.get("goal_id", "unknown"))
            by_domain.setdefault(goal_id, []).append(regret)

        total = len(results)
        return RegretStats(
            mean_regret=total_regret / max(1, total),
            regret_by_policy_type={
                k: sum(v) / len(v) for k, v in by_type.items()
            },
            regret_by_goal_domain={
                k: sum(v) / len(v) for k, v in by_domain.items()
            },
            total_resolved=total,
            high_regret_count=high_regret,
        )

    async def _fetch_recent_outcomes(self) -> list[dict[str, Any]]:
        """Retrieve recent action outcome episodes from Memory."""
        cutoff = (utc_now() - timedelta(days=_OUTCOME_HORIZON_DAYS)).isoformat()
        try:
            results = await self._memory.execute_read(  # type: ignore[union-attr]
                """
                MATCH (e:Episode)
                WHERE e.source STARTS WITH 'axon:'
                  AND e.event_time >= datetime($cutoff)
                RETURN
                  e.id AS id,
                  e.source AS source,
                  e.affect_valence AS affect_valence,
                  e.salience_composite AS salience_composite,
                  e.salience_scores_json AS salience_scores_json
                ORDER BY e.event_time DESC
                LIMIT $limit
                """,
                {"cutoff": cutoff, "limit": _OUTCOME_WINDOW},
            )
            return list(results)
        except Exception as exc:
            self._logger.warning("outcome_fetch_failed", error=str(exc))
            return []

    def _compute_stats(self, outcomes: list[dict[str, Any]]) -> SelfModelStats:
        """Derive statistics from raw outcome rows."""
        capability_scores: dict[str, CapabilityScore] = {}
        success_count = 0
        alignment_sum = 0.0
        alignment_count = 0

        for row in outcomes:
            source = str(row.get("source", ""))
            # capability = action type from source "axon:{action_type}"
            capability = source[len("axon:"):] if source.startswith("axon:") else source

            # Success heuristic: positive affect valence and reasonable salience
            valence = float(row.get("affect_valence", 0.0))
            salience = float(row.get("salience_composite", 0.0))
            succeeded = valence >= 0.0 and salience > 0.2

            scores = capability_scores.setdefault(
                capability,
                CapabilityScore(capability=capability),
            )
            scores.total_count += 1
            if succeeded:
                scores.success_count += 1
                success_count += 1

            # Alignment from salience_scores_json (Equor stores composite alignment there)
            raw_json = row.get("salience_scores_json") or "{}"
            scores_map = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            if isinstance(scores_map, dict):
                alignment = scores_map.get("equor_alignment")
                if alignment is not None:
                    alignment_sum += float(alignment)
                    alignment_count += 1

        total = len(outcomes)
        overall_success_rate = success_count / max(1, total)
        mean_alignment = alignment_sum / max(1, alignment_count) if alignment_count > 0 else 0.5

        return SelfModelStats(
            success_rate=overall_success_rate,
            mean_alignment=mean_alignment,
            total_outcomes_evaluated=total,
            capability_scores=capability_scores,
            updated_at=utc_now(),
        )

    async def _persist_stats(self, stats: SelfModelStats) -> None:
        """Write self-model stats (including regret) to the Self node in Memory."""
        try:
            await self._memory.execute_write(  # type: ignore[union-attr]
                """
                MATCH (s:Self)
                SET s.evo_success_rate = $success_rate,
                    s.evo_mean_alignment = $mean_alignment,
                    s.evo_outcomes_evaluated = $outcomes_evaluated,
                    s.evo_mean_regret = $mean_regret,
                    s.evo_counterfactuals_resolved = $cf_resolved,
                    s.evo_high_regret_count = $high_regret,
                    s.evo_regret_by_policy_type_json = $regret_by_type,
                    s.evo_last_updated = datetime()
                """,
                {
                    "success_rate": stats.success_rate,
                    "mean_alignment": stats.mean_alignment,
                    "outcomes_evaluated": stats.total_outcomes_evaluated,
                    "mean_regret": stats.regret.mean_regret,
                    "cf_resolved": stats.regret.total_resolved,
                    "high_regret": stats.regret.high_regret_count,
                    "regret_by_type": json.dumps(stats.regret.regret_by_policy_type),
                },
            )
        except Exception as exc:
            self._logger.warning("self_model_persist_failed", error=str(exc))
