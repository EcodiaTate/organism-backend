"""
EcodiaOS - Evo Hypothesis Tournament Engine

Competitive A/B experimentation using Thompson sampling (Beta-Bernoulli
conjugate model) for hypothesis selection.

When Evo discovers multiple competing hypotheses with similar fitness scores,
the tournament engine:
  1. Identifies clusters of close-scored hypotheses
  2. Creates tournaments that route real decision contexts to both policies
  3. Uses Thompson sampling to converge on the winner with minimal regret
  4. Archives losing hypotheses and promotes the winner

Thompson sampling overview:
  - Each hypothesis maintains a Beta(α, β) distribution over its success rate
  - To select which hypothesis to test next, sample from each Beta and pick
    the hypothesis with the highest sampled value
  - After observing a success/failure, update: α += 1 (success) or β += 1 (failure)
  - Convergence: when one hypothesis's posterior probability of being best
    exceeds the convergence_threshold (default 0.95)

Burn-in phase:
  - First N trials (default 10) use uniform 50/50 routing regardless of
    posterior to ensure both hypotheses get a fair initial sample
  - If any hypothesis has 0 trials after burn-in, temporarily boost its
    probability to 60% to force rebalancing

Performance budget: Thompson sampling is O(K) per routing decision where
K = number of hypotheses in the tournament (typically 2–4). Negligible cost.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.evo.types import (
    BetaDistribution,
    HypothesisRef,
    HypothesisTournament,
    TournamentContext,
    TournamentOutcome,
    TournamentStage,
)

if TYPE_CHECKING:
    from systems.evo.hypothesis import HypothesisEngine
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

# Hypotheses within this evidence_score delta are considered "competing"
_FITNESS_CLUSTER_THRESHOLD: float = 1.0

# Maximum concurrent active tournaments
_MAX_ACTIVE_TOURNAMENTS: int = 5

# Posterior probability samples for convergence check (Monte Carlo)
_CONVERGENCE_SAMPLES: int = 10_000


class TournamentEngine:
    """
    Manages hypothesis tournaments using Thompson sampling.

    Dependencies:
      hypothesis_engine - for querying active hypotheses and their scores
      memory            - optional; for persisting tournaments to Neo4j
    """

    def __init__(
        self,
        hypothesis_engine: HypothesisEngine,
        memory: MemoryService | None = None,
    ) -> None:
        self._hypotheses = hypothesis_engine
        self._memory = memory
        self._logger = logger.bind(system="evo.tournament")

        # Active tournaments keyed by tournament_id
        self._tournaments: dict[str, HypothesisTournament] = {}

        # Index: hypothesis_id → tournament_id (for fast lookup)
        self._hypothesis_to_tournament: dict[str, str] = {}

        # Metrics
        self._total_created: int = 0
        self._total_converged: int = 0

    # ─── Tournament Creation ─────────────────────────────────────────

    def detect_and_create_tournaments(self) -> list[HypothesisTournament]:
        """
        Scan active hypotheses for clusters with similar evidence scores.
        Create tournaments for competing pairs/groups not already in a tournament.

        Called during consolidation Phase 2 after hypothesis review.
        Returns list of newly created tournaments.
        """
        if len(self._tournaments) >= _MAX_ACTIVE_TOURNAMENTS:
            self._logger.debug(
                "tournament_capacity_reached",
                active=len(self._tournaments),
                max=_MAX_ACTIVE_TOURNAMENTS,
            )
            return []

        # Get hypotheses that are in TESTING status with meaningful evidence
        active = self._hypotheses.get_active()
        candidates = [
            h for h in active
            if h.evidence_score > 0.5  # Must have some evidence
            and h.id not in self._hypothesis_to_tournament  # Not already in a tournament
        ]

        if len(candidates) < 2:
            return []

        # Sort by evidence_score descending
        candidates.sort(key=lambda h: h.evidence_score, reverse=True)

        # Find clusters of hypotheses within the fitness threshold
        new_tournaments: list[HypothesisTournament] = []
        used: set[str] = set()

        for i, anchor in enumerate(candidates):
            if anchor.id in used:
                continue
            cluster = [anchor]
            for j in range(i + 1, len(candidates)):
                other = candidates[j]
                if other.id in used:
                    continue
                if abs(anchor.evidence_score - other.evidence_score) <= _FITNESS_CLUSTER_THRESHOLD:
                    cluster.append(other)
                if len(cluster) >= 4:  # Max 4 hypotheses per tournament
                    break

            if len(cluster) >= 2:
                tournament = self._create_tournament(cluster)
                new_tournaments.append(tournament)
                for h in cluster:
                    used.add(h.id)

                if len(self._tournaments) >= _MAX_ACTIVE_TOURNAMENTS:
                    break

        return new_tournaments

    def _create_tournament(
        self,
        hypotheses: list[Any],
    ) -> HypothesisTournament:
        """Create a new tournament from a cluster of competing hypotheses."""
        refs = [
            HypothesisRef(
                hypothesis_id=h.id,
                statement=h.statement[:200],
                evidence_score=h.evidence_score,
            )
            for h in hypotheses
        ]

        beta_params = {
            h.id: BetaDistribution()  # Uniform prior Beta(1, 1)
            for h in hypotheses
        }

        tournament = HypothesisTournament(
            id=new_id(),
            hypotheses=refs,
            beta_parameters=beta_params,
        )

        self._tournaments[tournament.id] = tournament
        for h in hypotheses:
            self._hypothesis_to_tournament[h.id] = tournament.id

        # Apply any pending inherited priors for this tournament's hypotheses
        pending_priors = getattr(self, "_pending_priors", {})
        for h in hypotheses:
            if tournament.id in pending_priors:
                prior = pending_priors[tournament.id].get(h.id)
                if prior is not None:
                    alpha, beta_val = prior
                    tournament.beta_parameters[h.id].alpha = alpha
                    tournament.beta_parameters[h.id].beta = beta_val

        self._total_created += 1
        self._logger.info(
            "tournament_created",
            tournament_id=tournament.id,
            hypotheses=[h.id for h in hypotheses],
            evidence_scores=[round(h.evidence_score, 2) for h in hypotheses],
        )

        return tournament

    # ─── Thompson Sampling (Routing) ──────────────────────────────────

    def get_tournament_for_hypothesis(self, hypothesis_id: str) -> HypothesisTournament | None:
        """Return the active tournament this hypothesis is in, if any."""
        tournament_id = self._hypothesis_to_tournament.get(hypothesis_id)
        if tournament_id is None:
            return None
        tournament = self._tournaments.get(tournament_id)
        if tournament is None or not tournament.is_running:
            return None
        return tournament

    def sample_hypothesis(self, tournament: HypothesisTournament) -> TournamentContext:
        """
        Use Thompson sampling to select which hypothesis to test next.

        During burn-in (first N trials), uses uniform random selection.
        After burn-in, samples from each hypothesis's Beta posterior and
        picks the one with the highest sampled value.

        If any hypothesis has 0 trials after burn-in, forces rebalancing
        by boosting its selection probability to 60%.
        """
        hypothesis_ids = [ref.hypothesis_id for ref in tournament.hypotheses]

        # ── Burn-in phase: uniform random ──
        if tournament.sample_count < tournament.burn_in_trials:
            selected_id = random.choice(hypothesis_ids)
            tournament.sample_count += 1
            return TournamentContext(
                tournament_id=tournament.id,
                hypothesis_id=selected_id,
            )

        # ── Post burn-in: check for zero-trial hypotheses ──
        zero_trial_ids = [
            h_id for h_id in hypothesis_ids
            if tournament.beta_parameters[h_id].sample_count == 0
        ]
        if zero_trial_ids:
            # Force rebalance: 60% chance of selecting the zero-trial hypothesis
            if random.random() < 0.6:
                selected_id = random.choice(zero_trial_ids)
                tournament.sample_count += 1
                self._logger.info(
                    "tournament_rebalance",
                    tournament_id=tournament.id,
                    forced_hypothesis=selected_id,
                )
                return TournamentContext(
                    tournament_id=tournament.id,
                    hypothesis_id=selected_id,
                )

        # ── Thompson sampling: sample from each Beta, pick highest ──
        best_id = hypothesis_ids[0]
        best_sample = -1.0

        for h_id in hypothesis_ids:
            beta = tournament.beta_parameters[h_id]
            # random.betavariate draws from Beta(alpha, beta) distribution
            sampled_value = random.betavariate(beta.alpha, beta.beta)
            if sampled_value > best_sample:
                best_sample = sampled_value
                best_id = h_id

        tournament.sample_count += 1

        # Log sampling probabilities for observability
        probs = {
            h_id: round(tournament.beta_parameters[h_id].mean, 3)
            for h_id in hypothesis_ids
        }
        self._logger.debug(
            "tournament_sample",
            tournament_id=tournament.id,
            selected=best_id,
            mean_probs=probs,
            sample_n=tournament.sample_count,
        )

        return TournamentContext(
            tournament_id=tournament.id,
            hypothesis_id=best_id,
        )

    # ─── Outcome Recording ────────────────────────────────────────────

    def record_outcome(self, outcome: TournamentOutcome) -> None:
        """
        Record a trial outcome (success/failure) and update the Beta posterior.
        Called by EvoService when Axon reports an intent outcome linked to a tournament.
        """
        tournament = self._tournaments.get(outcome.tournament_id)
        if tournament is None or not tournament.is_running:
            self._logger.debug(
                "tournament_outcome_skipped",
                tournament_id=outcome.tournament_id,
                reason="not_found_or_not_running",
            )
            return

        beta = tournament.beta_parameters.get(outcome.hypothesis_id)
        if beta is None:
            self._logger.warning(
                "tournament_outcome_unknown_hypothesis",
                tournament_id=outcome.tournament_id,
                hypothesis_id=outcome.hypothesis_id,
            )
            return

        if outcome.success:
            beta.update_success()
        else:
            beta.update_failure()

        self._logger.info(
            "tournament_outcome_recorded",
            tournament_id=outcome.tournament_id,
            hypothesis_id=outcome.hypothesis_id,
            success=outcome.success,
            alpha=beta.alpha,
            beta_param=beta.beta,
            mean=round(beta.mean, 3),
        )

    # ─── Convergence Check ────────────────────────────────────────────

    async def check_convergence(self) -> list[HypothesisTournament]:
        """
        Check all running tournaments for convergence.
        A tournament converges when one hypothesis's posterior probability
        of being the best exceeds the convergence_threshold.

        Uses Monte Carlo estimation: sample from each Beta N times,
        count how often each hypothesis wins.

        Returns list of tournaments that converged this cycle.
        """
        converged: list[HypothesisTournament] = []

        for tournament in list(self._tournaments.values()):
            if not tournament.is_running:
                continue

            # Need at least burn-in trials before checking
            if tournament.sample_count < tournament.burn_in_trials:
                continue

            # Monte Carlo posterior estimation
            win_counts: dict[str, int] = {
                ref.hypothesis_id: 0 for ref in tournament.hypotheses
            }

            for _ in range(_CONVERGENCE_SAMPLES):
                best_id = ""
                best_val = -1.0
                for ref in tournament.hypotheses:
                    beta = tournament.beta_parameters[ref.hypothesis_id]
                    sampled = random.betavariate(beta.alpha, beta.beta)
                    if sampled > best_val:
                        best_val = sampled
                        best_id = ref.hypothesis_id
                win_counts[best_id] += 1

            # Check if any hypothesis wins with sufficient probability
            for h_id, wins in win_counts.items():
                posterior_prob = wins / _CONVERGENCE_SAMPLES
                if posterior_prob >= tournament.convergence_threshold:
                    await self._converge_tournament(tournament, h_id, posterior_prob)
                    converged.append(tournament)
                    break

        return converged

    async def _converge_tournament(
        self,
        tournament: HypothesisTournament,
        winner_id: str,
        posterior_prob: float,
    ) -> None:
        """
        Mark a tournament as converged: declare winner, compute regret saved.
        """
        tournament.stage = TournamentStage.CONVERGED
        tournament.winner_id = winner_id

        # Compute regret saved: how many trials would the loser have wasted
        # if we had committed to it without testing?
        loser_ids = [
            ref.hypothesis_id for ref in tournament.hypotheses
            if ref.hypothesis_id != winner_id
        ]
        winner_beta = tournament.beta_parameters[winner_id]
        winner_mean = winner_beta.mean

        regret_saved_pct = 0.0
        for loser_id in loser_ids:
            loser_beta = tournament.beta_parameters[loser_id]
            loser_mean = loser_beta.mean
            if winner_mean > 0:
                regret_saved_pct = max(
                    regret_saved_pct,
                    round((winner_mean - loser_mean) / winner_mean * 100, 1),
                )

        self._total_converged += 1
        self._logger.info(
            "tournament_converged",
            tournament_id=tournament.id,
            winner=winner_id,
            posterior_prob=round(posterior_prob, 3),
            sample_count=tournament.sample_count,
            regret_saved_pct=regret_saved_pct,
            loser_ids=loser_ids,
        )

        # Persist to Neo4j
        await self._persist_tournament(tournament)

        # Clean up index for losers (winner stays mapped so Nova can still query it)
        for loser_id in loser_ids:
            self._hypothesis_to_tournament.pop(loser_id, None)

    # ─── Archival ─────────────────────────────────────────────────────

    async def archive_converged(self) -> int:
        """
        Move converged tournaments to archived state.
        Keeps the record in Neo4j for learning.
        Returns count of newly archived tournaments.
        """
        archived = 0
        for tournament in list(self._tournaments.values()):
            if tournament.stage != TournamentStage.CONVERGED:
                continue

            tournament.stage = TournamentStage.ARCHIVED
            await self._persist_tournament(tournament)

            # Clean up all index entries
            for ref in tournament.hypotheses:
                self._hypothesis_to_tournament.pop(ref.hypothesis_id, None)

            # Remove from active registry
            del self._tournaments[tournament.id]
            archived += 1

            self._logger.info(
                "tournament_archived",
                tournament_id=tournament.id,
                winner_id=tournament.winner_id,
            )

        return archived

    # ─── Query Interface ──────────────────────────────────────────────

    def get_active_tournaments(self) -> list[HypothesisTournament]:
        """Return all running tournaments."""
        return [t for t in self._tournaments.values() if t.is_running]

    def get_all_tournaments(self) -> list[HypothesisTournament]:
        """Return all tournaments (running + converged, not archived)."""
        return list(self._tournaments.values())

    def seed_inherited_prior(
        self,
        tournament_id: str,
        hypothesis_id: str,
        alpha: float,
        beta: float,
    ) -> None:
        """
        Seed a Beta prior from a parent genome into an existing tournament.

        If the tournament doesn't exist yet (child hasn't seen the hypotheses), the
        prior is stored in a pending registry and applied when the tournament is
        created via _create_tournament().

        Alpha and beta are applied directly - they represent parent's accumulated
        evidence (already filtered to ≥5 samples at export time).
        """
        existing = self._tournaments.get(tournament_id)
        if existing is not None:
            # Tournament already running - update the Beta distribution directly
            existing_beta = existing.beta_parameters.get(hypothesis_id)
            if existing_beta is not None:
                # Merge: parent's posterior becomes child's prior
                existing_beta.alpha = alpha
                existing_beta.beta = beta
                self._logger.debug(
                    "tournament_prior_seeded",
                    tournament_id=tournament_id,
                    hypothesis_id=hypothesis_id[:12],
                    alpha=round(alpha, 3),
                    beta=round(beta, 3),
                )
        else:
            # Tournament not yet created - stash for later application
            if not hasattr(self, "_pending_priors"):
                self._pending_priors: dict[str, dict[str, tuple[float, float]]] = {}
            self._pending_priors.setdefault(tournament_id, {})[hypothesis_id] = (alpha, beta)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "active": len([t for t in self._tournaments.values() if t.is_running]),
            "converged": len([t for t in self._tournaments.values() if t.is_converged]),
            "total_created": self._total_created,
            "total_converged": self._total_converged,
        }

    # ─── Neo4j Persistence ────────────────────────────────────────────

    async def _persist_tournament(self, tournament: HypothesisTournament) -> None:
        """Persist tournament state and relationships to Neo4j."""
        if self._memory is None:
            return

        try:
            # Serialise beta parameters for storage
            {
                h_id: {"alpha": b.alpha, "beta": b.beta, "mean": round(b.mean, 4)}
                for h_id, b in tournament.beta_parameters.items()
            }

            await self._memory.execute_write(
                """
                MERGE (t:HypothesisTournament {tournament_id: $tournament_id})
                SET t.stage = $stage,
                    t.sample_count = $sample_count,
                    t.winner_id = $winner_id,
                    t.convergence_threshold = $convergence_threshold,
                    t.created_at = $created_at,
                    t.updated_at = $updated_at
                """,
                {
                    "tournament_id": tournament.id,
                    "stage": tournament.stage.value,
                    "sample_count": tournament.sample_count,
                    "winner_id": tournament.winner_id or "",
                    "convergence_threshold": tournament.convergence_threshold,
                    "created_at": tournament.created_at.isoformat(),
                    "updated_at": utc_now().isoformat(),
                },
            )

            # Create COMPETES_IN relationships for each hypothesis
            for ref in tournament.hypotheses:
                beta = tournament.beta_parameters.get(ref.hypothesis_id)
                is_winner = ref.hypothesis_id == tournament.winner_id

                await self._memory.execute_write(
                    """
                    MATCH (t:HypothesisTournament {tournament_id: $tournament_id})
                    MERGE (h:Hypothesis {hypothesis_id: $hypothesis_id})
                    MERGE (h)-[r:COMPETES_IN]->(t)
                    SET r.alpha = $alpha,
                        r.beta = $beta,
                        r.mean = $mean,
                        r.is_winner = $is_winner
                    """,
                    {
                        "tournament_id": tournament.id,
                        "hypothesis_id": ref.hypothesis_id,
                        "alpha": beta.alpha if beta else 1.0,
                        "beta": beta.beta if beta else 1.0,
                        "mean": round(beta.mean, 4) if beta else 0.5,
                        "is_winner": is_winner,
                    },
                )

        except Exception as exc:
            self._logger.warning(
                "tournament_persist_failed",
                tournament_id=tournament.id,
                error=str(exc),
            )
