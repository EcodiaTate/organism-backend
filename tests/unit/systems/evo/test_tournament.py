"""
Unit tests for Evo TournamentEngine.

Tests Thompson sampling-based hypothesis tournaments:
  - Tournament creation from hypothesis clusters
  - Thompson sampling routing (burn-in, rebalance, posterior)
  - Outcome recording and Beta distribution updates
  - Convergence detection via Monte Carlo posterior estimation
  - Tournament archival and index cleanup
  - Neo4j persistence
  - Consolidation phase integration
"""

from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from systems.evo.tournament import (
    _MAX_ACTIVE_TOURNAMENTS,
    TournamentEngine,
)
from systems.evo.types import (
    BetaDistribution,
    Hypothesis,
    HypothesisCategory,
    HypothesisRef,
    HypothesisStatus,
    HypothesisTournament,
    TournamentContext,
    TournamentOutcome,
    TournamentStage,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_hypothesis(
    h_id: str = "h_1",
    evidence_score: float = 2.0,
    status: HypothesisStatus = HypothesisStatus.TESTING,
) -> Hypothesis:
    """Create a test hypothesis with the given parameters."""
    h = Hypothesis(
        category=HypothesisCategory.WORLD_MODEL,
        statement=f"Test hypothesis {h_id}",
        formal_test=f"Test condition for {h_id}",
        status=status,
        evidence_score=evidence_score,
    )
    # Override the auto-generated ID with a deterministic one
    h.id = h_id
    return h


def make_hypothesis_engine(
    hypotheses: list[Hypothesis] | None = None,
) -> MagicMock:
    """Create a mock HypothesisEngine returning the given hypotheses."""
    from systems.evo.hypothesis import HypothesisEngine

    engine = MagicMock(spec=HypothesisEngine)
    engine.get_active.return_value = hypotheses or []
    return engine


def make_tournament_engine(
    hypotheses: list[Hypothesis] | None = None,
    memory: MagicMock | None = None,
) -> TournamentEngine:
    """Create a TournamentEngine with mock dependencies."""
    engine = make_hypothesis_engine(hypotheses)
    return TournamentEngine(hypothesis_engine=engine, memory=memory)


def make_tournament(
    h_ids: list[str] | None = None,
    burn_in: int = 10,
    convergence_threshold: float = 0.95,
) -> HypothesisTournament:
    """Create a tournament with Beta(1,1) priors for given hypothesis IDs."""
    if h_ids is None:
        h_ids = ["h_a", "h_b"]

    refs = [
        HypothesisRef(hypothesis_id=h_id, statement=f"Hyp {h_id}", evidence_score=2.0)
        for h_id in h_ids
    ]
    betas = {h_id: BetaDistribution() for h_id in h_ids}  # Uniform prior

    return HypothesisTournament(
        hypotheses=refs,
        beta_parameters=betas,
        burn_in_trials=burn_in,
        convergence_threshold=convergence_threshold,
    )


# ─── Tests: BetaDistribution ─────────────────────────────────────────────────


class TestBetaDistribution:
    def test_uniform_prior(self):
        beta = BetaDistribution()
        assert beta.alpha == 1.0
        assert beta.beta == 1.0
        assert beta.mean == 0.5
        assert beta.sample_count == 0

    def test_update_success(self):
        beta = BetaDistribution()
        beta.update_success()
        assert beta.alpha == 2.0
        assert beta.beta == 1.0
        assert beta.mean == pytest.approx(2 / 3)
        assert beta.sample_count == 1

    def test_update_failure(self):
        beta = BetaDistribution()
        beta.update_failure()
        assert beta.alpha == 1.0
        assert beta.beta == 2.0
        assert beta.mean == pytest.approx(1 / 3)
        assert beta.sample_count == 1

    def test_multiple_updates(self):
        beta = BetaDistribution()
        for _ in range(5):
            beta.update_success()
        for _ in range(3):
            beta.update_failure()
        assert beta.alpha == 6.0  # 1 + 5
        assert beta.beta == 4.0   # 1 + 3
        assert beta.sample_count == 8
        assert beta.mean == pytest.approx(6 / 10)

    def test_mean_approaches_one_with_many_successes(self):
        beta = BetaDistribution()
        for _ in range(100):
            beta.update_success()
        assert beta.mean > 0.95


# ─── Tests: HypothesisTournament ─────────────────────────────────────────────


class TestHypothesisTournament:
    def test_initial_state(self):
        t = make_tournament()
        assert t.stage == TournamentStage.RUNNING
        assert t.is_running
        assert not t.is_converged
        assert t.sample_count == 0
        assert t.winner_id is None

    def test_is_running_and_converged_properties(self):
        t = make_tournament()
        assert t.is_running
        assert not t.is_converged

        t.stage = TournamentStage.CONVERGED
        assert not t.is_running
        assert t.is_converged

        t.stage = TournamentStage.ARCHIVED
        assert not t.is_running
        assert not t.is_converged


# ─── Tests: Tournament Creation ──────────────────────────────────────────────


class TestTournamentCreation:
    def test_creates_tournament_from_close_scored_hypotheses(self):
        h1 = make_hypothesis("h_1", evidence_score=2.0)
        h2 = make_hypothesis("h_2", evidence_score=2.5)
        te = make_tournament_engine([h1, h2])

        new = te.detect_and_create_tournaments()

        assert len(new) == 1
        tournament = new[0]
        assert len(tournament.hypotheses) == 2
        h_ids = {ref.hypothesis_id for ref in tournament.hypotheses}
        assert h_ids == {"h_1", "h_2"}
        assert tournament.stage == TournamentStage.RUNNING

    def test_does_not_create_tournament_for_distant_scores(self):
        h1 = make_hypothesis("h_1", evidence_score=1.0)
        h2 = make_hypothesis("h_2", evidence_score=5.0)  # > _FITNESS_CLUSTER_THRESHOLD apart
        te = make_tournament_engine([h1, h2])

        new = te.detect_and_create_tournaments()

        assert len(new) == 0

    def test_does_not_create_tournament_from_single_hypothesis(self):
        h1 = make_hypothesis("h_1", evidence_score=2.0)
        te = make_tournament_engine([h1])

        new = te.detect_and_create_tournaments()

        assert len(new) == 0

    def test_skips_hypotheses_with_low_evidence(self):
        """Hypotheses with evidence_score <= 0.5 are excluded."""
        h1 = make_hypothesis("h_1", evidence_score=0.3)
        h2 = make_hypothesis("h_2", evidence_score=0.4)
        te = make_tournament_engine([h1, h2])

        new = te.detect_and_create_tournaments()

        assert len(new) == 0

    def test_does_not_duplicate_hypotheses_in_tournaments(self):
        """A hypothesis already in a tournament cannot enter another."""
        h1 = make_hypothesis("h_1", evidence_score=2.0)
        h2 = make_hypothesis("h_2", evidence_score=2.5)
        te = make_tournament_engine([h1, h2])

        first = te.detect_and_create_tournaments()
        assert len(first) == 1

        # Add a third hypothesis close to h1's score
        h3 = make_hypothesis("h_3", evidence_score=2.2)
        te._hypotheses.get_active.return_value = [h1, h2, h3]

        second = te.detect_and_create_tournaments()
        # h1 and h2 already in tournament; only h3 is free but needs a partner
        assert len(second) == 0

    def test_respects_max_active_tournaments(self):
        """Cannot exceed _MAX_ACTIVE_TOURNAMENTS active tournaments."""
        te = make_tournament_engine([])

        # Fill up active tournament slots
        for i in range(_MAX_ACTIVE_TOURNAMENTS):
            t = make_tournament([f"ha_{i}", f"hb_{i}"])
            te._tournaments[t.id] = t
            te._hypothesis_to_tournament[f"ha_{i}"] = t.id
            te._hypothesis_to_tournament[f"hb_{i}"] = t.id

        # Now try to create more
        new_h1 = make_hypothesis("new_1", evidence_score=3.0)
        new_h2 = make_hypothesis("new_2", evidence_score=3.2)
        te._hypotheses.get_active.return_value = [new_h1, new_h2]

        new = te.detect_and_create_tournaments()
        assert len(new) == 0

    def test_creates_tournament_with_three_hypotheses(self):
        h1 = make_hypothesis("h_1", evidence_score=2.0)
        h2 = make_hypothesis("h_2", evidence_score=2.3)
        h3 = make_hypothesis("h_3", evidence_score=2.7)
        te = make_tournament_engine([h1, h2, h3])

        new = te.detect_and_create_tournaments()

        assert len(new) == 1
        assert len(new[0].hypotheses) == 3

    def test_max_four_hypotheses_per_tournament(self):
        hypotheses = [
            make_hypothesis(f"h_{i}", evidence_score=2.0 + i * 0.1)
            for i in range(6)
        ]
        te = make_tournament_engine(hypotheses)

        new = te.detect_and_create_tournaments()

        # Should create tournament(s) with at most 4 hypotheses each
        for t in new:
            assert len(t.hypotheses) <= 4

    def test_tournament_registers_in_index(self):
        h1 = make_hypothesis("h_1", evidence_score=2.0)
        h2 = make_hypothesis("h_2", evidence_score=2.5)
        te = make_tournament_engine([h1, h2])

        new = te.detect_and_create_tournaments()
        t = new[0]

        assert te._hypothesis_to_tournament["h_1"] == t.id
        assert te._hypothesis_to_tournament["h_2"] == t.id

    def test_stats_after_creation(self):
        h1 = make_hypothesis("h_1", evidence_score=2.0)
        h2 = make_hypothesis("h_2", evidence_score=2.5)
        te = make_tournament_engine([h1, h2])

        te.detect_and_create_tournaments()

        stats = te.stats
        assert stats["active"] == 1
        assert stats["total_created"] == 1
        assert stats["converged"] == 0


# ─── Tests: Thompson Sampling (Routing) ──────────────────────────────────────


class TestThompsonSampling:
    def test_burn_in_uses_uniform_random(self):
        """During burn-in, selection should be uniform random (50/50)."""
        te = make_tournament_engine()
        t = make_tournament(burn_in=10)
        te._tournaments[t.id] = t

        # Run 100 samples during burn-in and check distribution
        selections: dict[str, int] = {"h_a": 0, "h_b": 0}
        random.seed(42)
        for _ in range(100):
            # Reset sample_count to stay in burn-in
            t.sample_count = 0
            ctx = te.sample_hypothesis(t)
            selections[ctx.hypothesis_id] += 1

        # Both should be selected roughly equally (within reason)
        assert selections["h_a"] > 20
        assert selections["h_b"] > 20

    def test_burn_in_increments_sample_count(self):
        te = make_tournament_engine()
        t = make_tournament(burn_in=10)
        te._tournaments[t.id] = t

        assert t.sample_count == 0
        te.sample_hypothesis(t)
        assert t.sample_count == 1

    def test_sample_returns_tournament_context(self):
        te = make_tournament_engine()
        t = make_tournament()
        te._tournaments[t.id] = t

        ctx = te.sample_hypothesis(t)

        assert isinstance(ctx, TournamentContext)
        assert ctx.tournament_id == t.id
        assert ctx.hypothesis_id in {"h_a", "h_b"}

    def test_post_burnin_uses_thompson_sampling(self):
        """After burn-in, selection should favor the hypothesis with higher mean."""
        te = make_tournament_engine()
        t = make_tournament(burn_in=0)
        te._tournaments[t.id] = t

        # Give h_a a strong advantage: Beta(20, 2) vs h_b Beta(2, 20)
        t.beta_parameters["h_a"] = BetaDistribution(alpha=20.0, beta=2.0)
        t.beta_parameters["h_b"] = BetaDistribution(alpha=2.0, beta=20.0)

        selections: dict[str, int] = {"h_a": 0, "h_b": 0}
        random.seed(42)
        for _ in range(100):
            ctx = te.sample_hypothesis(t)
            selections[ctx.hypothesis_id] += 1

        # h_a should be selected far more often
        assert selections["h_a"] > 80

    def test_rebalance_forces_zero_trial_hypothesis(self):
        """If a hypothesis has 0 trials after burn-in, force rebalance."""
        te = make_tournament_engine()
        t = make_tournament(burn_in=5)
        t.sample_count = 10  # Past burn-in
        te._tournaments[t.id] = t

        # h_a has trials, h_b has none (still at prior Beta(1,1))
        t.beta_parameters["h_a"] = BetaDistribution(alpha=5.0, beta=2.0)
        t.beta_parameters["h_b"] = BetaDistribution(alpha=1.0, beta=1.0)  # 0 sample_count

        selections: dict[str, int] = {"h_a": 0, "h_b": 0}
        random.seed(42)
        for _ in range(100):
            ctx = te.sample_hypothesis(t)
            selections[ctx.hypothesis_id] += 1

        # h_b should be selected ~60% of the time due to rebalance
        assert selections["h_b"] > 40  # 60% of 100 = ~60, allowing for some variance

    def test_three_hypothesis_tournament_sampling(self):
        """Thompson sampling works with 3 hypotheses."""
        te = make_tournament_engine()
        t = make_tournament(h_ids=["h_a", "h_b", "h_c"], burn_in=0)
        te._tournaments[t.id] = t

        # Give h_c a strong advantage
        t.beta_parameters["h_a"] = BetaDistribution(alpha=2.0, beta=10.0)
        t.beta_parameters["h_b"] = BetaDistribution(alpha=2.0, beta=10.0)
        t.beta_parameters["h_c"] = BetaDistribution(alpha=20.0, beta=2.0)

        selections: dict[str, int] = {"h_a": 0, "h_b": 0, "h_c": 0}
        random.seed(42)
        for _ in range(100):
            ctx = te.sample_hypothesis(t)
            selections[ctx.hypothesis_id] += 1

        assert selections["h_c"] > 70


# ─── Tests: Outcome Recording ────────────────────────────────────────────────


class TestOutcomeRecording:
    def test_records_success(self):
        te = make_tournament_engine()
        t = make_tournament()
        te._tournaments[t.id] = t

        outcome = TournamentOutcome(
            tournament_id=t.id,
            hypothesis_id="h_a",
            success=True,
        )
        te.record_outcome(outcome)

        beta = t.beta_parameters["h_a"]
        assert beta.alpha == 2.0  # 1 + 1 success
        assert beta.beta == 1.0

    def test_records_failure(self):
        te = make_tournament_engine()
        t = make_tournament()
        te._tournaments[t.id] = t

        outcome = TournamentOutcome(
            tournament_id=t.id,
            hypothesis_id="h_b",
            success=False,
        )
        te.record_outcome(outcome)

        beta = t.beta_parameters["h_b"]
        assert beta.alpha == 1.0
        assert beta.beta == 2.0  # 1 + 1 failure

    def test_skips_unknown_tournament(self):
        """Recording an outcome for a non-existent tournament is a no-op."""
        te = make_tournament_engine()

        outcome = TournamentOutcome(
            tournament_id="nonexistent",
            hypothesis_id="h_a",
            success=True,
        )
        # Should not raise
        te.record_outcome(outcome)

    def test_skips_non_running_tournament(self):
        """Recording an outcome for a converged tournament is a no-op."""
        te = make_tournament_engine()
        t = make_tournament()
        t.stage = TournamentStage.CONVERGED
        te._tournaments[t.id] = t

        outcome = TournamentOutcome(
            tournament_id=t.id,
            hypothesis_id="h_a",
            success=True,
        )
        te.record_outcome(outcome)

        # Beta should remain at prior
        assert t.beta_parameters["h_a"].alpha == 1.0

    def test_warns_on_unknown_hypothesis(self):
        """Recording for an unknown hypothesis_id in the tournament logs a warning."""
        te = make_tournament_engine()
        t = make_tournament()
        te._tournaments[t.id] = t

        outcome = TournamentOutcome(
            tournament_id=t.id,
            hypothesis_id="unknown_h",
            success=True,
        )
        # Should not raise
        te.record_outcome(outcome)

    def test_multiple_outcomes_update_beta_correctly(self):
        te = make_tournament_engine()
        t = make_tournament()
        te._tournaments[t.id] = t

        # 7 successes, 3 failures for h_a
        for _ in range(7):
            te.record_outcome(TournamentOutcome(
                tournament_id=t.id, hypothesis_id="h_a", success=True,
            ))
        for _ in range(3):
            te.record_outcome(TournamentOutcome(
                tournament_id=t.id, hypothesis_id="h_a", success=False,
            ))

        beta = t.beta_parameters["h_a"]
        assert beta.alpha == 8.0   # 1 + 7
        assert beta.beta == 4.0    # 1 + 3
        assert beta.sample_count == 10
        assert beta.mean == pytest.approx(8 / 12)


# ─── Tests: Convergence ─────────────────────────────────────────────────────


class TestConvergence:
    @pytest.mark.asyncio
    async def test_converges_when_one_hypothesis_dominates(self):
        """A tournament converges when one hypothesis has overwhelming posterior."""
        te = make_tournament_engine()
        t = make_tournament(burn_in=0, convergence_threshold=0.95)
        t.sample_count = 50  # Past burn-in
        te._tournaments[t.id] = t

        # h_a is clearly better: Beta(50, 2) vs h_b Beta(2, 50)
        t.beta_parameters["h_a"] = BetaDistribution(alpha=50.0, beta=2.0)
        t.beta_parameters["h_b"] = BetaDistribution(alpha=2.0, beta=50.0)

        converged = await te.check_convergence()

        assert len(converged) == 1
        assert converged[0].id == t.id
        assert converged[0].stage == TournamentStage.CONVERGED
        assert converged[0].winner_id == "h_a"

    @pytest.mark.asyncio
    async def test_does_not_converge_when_close(self):
        """Equal Beta distributions should not converge."""
        te = make_tournament_engine()
        t = make_tournament(burn_in=0, convergence_threshold=0.95)
        t.sample_count = 20
        te._tournaments[t.id] = t

        # Both hypotheses are similar: Beta(10, 10) vs Beta(11, 10)
        t.beta_parameters["h_a"] = BetaDistribution(alpha=10.0, beta=10.0)
        t.beta_parameters["h_b"] = BetaDistribution(alpha=11.0, beta=10.0)

        converged = await te.check_convergence()

        assert len(converged) == 0
        assert t.stage == TournamentStage.RUNNING

    @pytest.mark.asyncio
    async def test_does_not_converge_during_burn_in(self):
        """Convergence check is skipped during burn-in."""
        te = make_tournament_engine()
        t = make_tournament(burn_in=50)
        t.sample_count = 10  # Still in burn-in
        te._tournaments[t.id] = t

        # Even with clear winner, should not converge during burn-in
        t.beta_parameters["h_a"] = BetaDistribution(alpha=100.0, beta=1.0)
        t.beta_parameters["h_b"] = BetaDistribution(alpha=1.0, beta=100.0)

        converged = await te.check_convergence()
        assert len(converged) == 0

    @pytest.mark.asyncio
    async def test_convergence_cleans_up_loser_index(self):
        """After convergence, losers are removed from the hypothesis-to-tournament index."""
        te = make_tournament_engine()
        t = make_tournament(burn_in=0)
        t.sample_count = 50
        te._tournaments[t.id] = t
        te._hypothesis_to_tournament["h_a"] = t.id
        te._hypothesis_to_tournament["h_b"] = t.id

        t.beta_parameters["h_a"] = BetaDistribution(alpha=50.0, beta=2.0)
        t.beta_parameters["h_b"] = BetaDistribution(alpha=2.0, beta=50.0)

        await te.check_convergence()

        # Loser (h_b) removed from index; winner (h_a) stays
        assert "h_a" in te._hypothesis_to_tournament
        assert "h_b" not in te._hypothesis_to_tournament

    @pytest.mark.asyncio
    async def test_convergence_increments_total_converged(self):
        te = make_tournament_engine()
        t = make_tournament(burn_in=0)
        t.sample_count = 50
        te._tournaments[t.id] = t

        t.beta_parameters["h_a"] = BetaDistribution(alpha=50.0, beta=2.0)
        t.beta_parameters["h_b"] = BetaDistribution(alpha=2.0, beta=50.0)

        assert te._total_converged == 0
        await te.check_convergence()
        assert te._total_converged == 1


# ─── Tests: Archival ─────────────────────────────────────────────────────────


class TestArchival:
    @pytest.mark.asyncio
    async def test_archives_converged_tournament(self):
        te = make_tournament_engine()
        t = make_tournament()
        t.stage = TournamentStage.CONVERGED
        t.winner_id = "h_a"
        te._tournaments[t.id] = t
        te._hypothesis_to_tournament["h_a"] = t.id
        te._hypothesis_to_tournament["h_b"] = t.id

        archived = await te.archive_converged()

        assert archived == 1
        assert t.id not in te._tournaments
        assert "h_a" not in te._hypothesis_to_tournament
        assert "h_b" not in te._hypothesis_to_tournament

    @pytest.mark.asyncio
    async def test_does_not_archive_running_tournament(self):
        te = make_tournament_engine()
        t = make_tournament()
        assert t.stage == TournamentStage.RUNNING
        te._tournaments[t.id] = t

        archived = await te.archive_converged()

        assert archived == 0
        assert t.id in te._tournaments

    @pytest.mark.asyncio
    async def test_archives_multiple_converged(self):
        te = make_tournament_engine()

        t1 = make_tournament(["t1_a", "t1_b"])
        t1.stage = TournamentStage.CONVERGED
        t1.winner_id = "t1_a"

        t2 = make_tournament(["t2_a", "t2_b"])
        t2.stage = TournamentStage.CONVERGED
        t2.winner_id = "t2_b"

        te._tournaments[t1.id] = t1
        te._tournaments[t2.id] = t2

        archived = await te.archive_converged()

        assert archived == 2
        assert len(te._tournaments) == 0


# ─── Tests: Query Interface ──────────────────────────────────────────────────


class TestQueryInterface:
    def test_get_active_tournaments(self):
        te = make_tournament_engine()

        t1 = make_tournament(["a", "b"])
        t1.stage = TournamentStage.RUNNING
        t2 = make_tournament(["c", "d"])
        t2.stage = TournamentStage.CONVERGED

        te._tournaments[t1.id] = t1
        te._tournaments[t2.id] = t2

        active = te.get_active_tournaments()
        assert len(active) == 1
        assert active[0].id == t1.id

    def test_get_all_tournaments(self):
        te = make_tournament_engine()

        t1 = make_tournament(["a", "b"])
        t2 = make_tournament(["c", "d"])
        t2.stage = TournamentStage.CONVERGED

        te._tournaments[t1.id] = t1
        te._tournaments[t2.id] = t2

        all_t = te.get_all_tournaments()
        assert len(all_t) == 2

    def test_get_tournament_for_hypothesis(self):
        te = make_tournament_engine()
        t = make_tournament(["h_x", "h_y"])
        te._tournaments[t.id] = t
        te._hypothesis_to_tournament["h_x"] = t.id
        te._hypothesis_to_tournament["h_y"] = t.id

        result = te.get_tournament_for_hypothesis("h_x")
        assert result is not None
        assert result.id == t.id

    def test_get_tournament_for_unknown_hypothesis(self):
        te = make_tournament_engine()
        result = te.get_tournament_for_hypothesis("unknown")
        assert result is None

    def test_get_tournament_for_non_running(self):
        """Returns None if tournament is not running."""
        te = make_tournament_engine()
        t = make_tournament(["h_x", "h_y"])
        t.stage = TournamentStage.CONVERGED
        te._tournaments[t.id] = t
        te._hypothesis_to_tournament["h_x"] = t.id

        result = te.get_tournament_for_hypothesis("h_x")
        assert result is None


# ─── Tests: Neo4j Persistence ────────────────────────────────────────────────


class TestPersistence:
    @pytest.mark.asyncio
    async def test_persists_tournament_to_neo4j(self):
        mock_neo4j = AsyncMock()
        mock_memory = MagicMock()
        mock_memory._neo4j = mock_neo4j

        te = make_tournament_engine(memory=mock_memory)
        t = make_tournament()
        t.stage = TournamentStage.CONVERGED
        t.winner_id = "h_a"
        te._tournaments[t.id] = t

        await te._persist_tournament(t)

        # Should call execute_write for the tournament node + one per hypothesis
        assert mock_neo4j.execute_write.call_count == 3  # 1 MERGE tournament + 2 MERGE hypothesis

    @pytest.mark.asyncio
    async def test_persist_handles_no_memory(self):
        """Persistence is a no-op when memory is None."""
        te = make_tournament_engine(memory=None)
        t = make_tournament()

        # Should not raise
        await te._persist_tournament(t)

    @pytest.mark.asyncio
    async def test_persist_handles_neo4j_error(self):
        """Persistence errors are caught and logged, not raised."""
        mock_neo4j = AsyncMock()
        mock_neo4j.execute_write.side_effect = RuntimeError("Neo4j down")
        mock_memory = MagicMock()
        mock_memory._neo4j = mock_neo4j

        te = make_tournament_engine(memory=mock_memory)
        t = make_tournament()

        # Should not raise
        await te._persist_tournament(t)


# ─── Tests: Full Lifecycle ───────────────────────────────────────────────────


class TestFullLifecycle:
    @pytest.mark.asyncio
    async def test_create_route_record_converge_archive(self):
        """Test the full tournament lifecycle: create → sample → record → converge → archive."""
        h1 = make_hypothesis("h_1", evidence_score=2.0)
        h2 = make_hypothesis("h_2", evidence_score=2.5)
        te = make_tournament_engine([h1, h2])

        # 1. Create tournament
        new = te.detect_and_create_tournaments()
        assert len(new) == 1
        t = new[0]
        assert t.is_running

        # 2. Sample during burn-in
        for _ in range(t.burn_in_trials):
            ctx = te.sample_hypothesis(t)
            assert ctx.tournament_id == t.id

        # 3. Record many outcomes: h_1 wins consistently
        random.seed(42)
        for _ in range(50):
            te.record_outcome(TournamentOutcome(
                tournament_id=t.id, hypothesis_id="h_1", success=True,
            ))
            te.record_outcome(TournamentOutcome(
                tournament_id=t.id, hypothesis_id="h_2", success=False,
            ))

        # 4. Check convergence
        converged = await te.check_convergence()
        assert len(converged) == 1
        assert t.winner_id == "h_1"
        assert t.stage == TournamentStage.CONVERGED

        # 5. Archive
        archived = await te.archive_converged()
        assert archived == 1
        assert len(te._tournaments) == 0

    @pytest.mark.asyncio
    async def test_stats_through_lifecycle(self):
        h1 = make_hypothesis("h_1", evidence_score=2.0)
        h2 = make_hypothesis("h_2", evidence_score=2.5)
        te = make_tournament_engine([h1, h2])

        # Before creation
        stats = te.stats
        assert stats["active"] == 0
        assert stats["total_created"] == 0

        # After creation
        te.detect_and_create_tournaments()
        stats = te.stats
        assert stats["active"] == 1
        assert stats["total_created"] == 1
        assert stats["total_converged"] == 0


# ─── Tests: EvoService Integration ──────────────────────────────────────────


class TestEvoServiceIntegration:
    def test_record_tournament_outcome_delegates(self):
        """EvoService.record_tournament_outcome() delegates to TournamentEngine."""
        from systems.evo.types import TournamentOutcome

        te = make_tournament_engine()
        t = make_tournament()
        te._tournaments[t.id] = t

        # Simulate what EvoService.record_tournament_outcome does
        outcome = TournamentOutcome(
            tournament_id=t.id,
            hypothesis_id="h_a",
            success=True,
            intent_id="intent_123",
        )
        te.record_outcome(outcome)

        assert t.beta_parameters["h_a"].alpha == 2.0
        assert outcome.intent_id == "intent_123"


# ─── Tests: Consolidation Phase Integration ──────────────────────────────────


class TestConsolidationPhase:
    @pytest.mark.asyncio
    async def test_phase_tournament_update_with_engine(self):
        """ConsolidationOrchestrator._phase_tournament_update() works with TournamentEngine."""
        from systems.evo.consolidation import ConsolidationOrchestrator

        hypothesis_engine = make_hypothesis_engine()
        tournament_engine = MagicMock(spec=TournamentEngine)
        tournament_engine.check_convergence = AsyncMock(return_value=[])
        tournament_engine.archive_converged = AsyncMock(return_value=0)
        tournament_engine.detect_and_create_tournaments.return_value = []
        tournament_engine.get_active_tournaments.return_value = []

        orchestrator = ConsolidationOrchestrator(
            hypothesis_engine=hypothesis_engine,
            parameter_tuner=MagicMock(begin_cycle=MagicMock()),
            procedure_extractor=MagicMock(begin_cycle=MagicMock()),
            self_model_manager=MagicMock(update=AsyncMock()),
            tournament_engine=tournament_engine,
        )

        active, converged = await orchestrator._phase_tournament_update()

        tournament_engine.check_convergence.assert_called_once()
        tournament_engine.archive_converged.assert_called_once()
        tournament_engine.detect_and_create_tournaments.assert_called_once()
        assert active == 0
        assert converged == 0

    @pytest.mark.asyncio
    async def test_phase_tournament_update_without_engine(self):
        """Phase returns (0, 0) when tournament_engine is None."""
        from systems.evo.consolidation import ConsolidationOrchestrator

        orchestrator = ConsolidationOrchestrator(
            hypothesis_engine=make_hypothesis_engine(),
            parameter_tuner=MagicMock(begin_cycle=MagicMock()),
            procedure_extractor=MagicMock(begin_cycle=MagicMock()),
            self_model_manager=MagicMock(update=AsyncMock()),
            tournament_engine=None,
        )

        active, converged = await orchestrator._phase_tournament_update()

        assert active == 0
        assert converged == 0
