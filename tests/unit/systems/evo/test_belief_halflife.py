"""
Unit tests for the Belief Half-Life system.

Tests the radioisotope decay model for knowledge freshness:
  - Domain half-life lookup and fallback
  - Age factor computation (decay formula)
  - Staleness detection
  - BeliefAgingScanner with mock Neo4j
  - Unreliable-in-N-hours projection
  - Consolidation integration (Phase 2.5)
"""

from __future__ import annotations

import math
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from primitives.common import utc_now

# Import directly from the module to avoid transitive dependency issues
# (systems.evo.__init__ pulls in EvoService → clients → cdp)
from systems.evo.belief_halflife import (
    BeliefAgingResult,
    BeliefAgingScanner,
    BeliefHalfLife,
    compute_age_factor,
    compute_halflife_metadata,
    get_halflife_for_domain,
    is_stale,
    stamp_belief_halflife,
)

# ─── Domain Lookup ───────────────────────────────────────────────────────────


class TestDomainLookup:
    def test_exact_match(self):
        assert get_halflife_for_domain("sentiment") == 0.3
        assert get_halflife_for_domain("capability") == 90.0
        assert get_halflife_for_domain("physical_law") == 36500.0

    def test_case_insensitive(self):
        assert get_halflife_for_domain("SENTIMENT") == 0.3
        assert get_halflife_for_domain("Capability") == 90.0

    def test_prefix_match(self):
        # "emotional_state" should match "emotional"
        assert get_halflife_for_domain("emotional_state") == 0.3

    def test_substring_match(self):
        # "user.sentiment.recent" should match "sentiment"
        assert get_halflife_for_domain("user.sentiment.recent") == 0.3

    def test_unknown_domain_returns_default(self):
        result = get_halflife_for_domain("quantum_flux_capacitor")
        assert result == 30.0  # _DEFAULT_HALFLIFE_DAYS


# ─── BeliefHalfLife Model ────────────────────────────────────────────────────


class TestBeliefHalfLifeModel:
    def test_decay_constant_computed(self):
        bhl = BeliefHalfLife(domain="test", half_life_days=10.0)
        expected = math.log(2) / 10.0
        assert abs(bhl.decay_constant - expected) < 1e-10

    def test_decay_constant_zero_halflife(self):
        bhl = BeliefHalfLife(domain="test", half_life_days=0.0)
        assert bhl.decay_constant == 0.0

    def test_compute_halflife_metadata(self):
        meta = compute_halflife_metadata("sentiment")
        assert meta.domain == "sentiment"
        assert meta.half_life_days == 0.3
        assert meta.decay_constant > 0
        assert meta.last_verified is not None


# ─── Age Factor Computation ──────────────────────────────────────────────────


class TestAgeFactor:
    def test_just_verified_returns_one(self):
        now = utc_now()
        factor = compute_age_factor(10.0, now, now)
        assert factor == 1.0

    def test_one_halflife_returns_half(self):
        now = utc_now()
        verified = now - timedelta(days=10)
        factor = compute_age_factor(10.0, verified, now)
        assert abs(factor - 0.5) < 1e-10

    def test_two_halflives_returns_quarter(self):
        now = utc_now()
        verified = now - timedelta(days=20)
        factor = compute_age_factor(10.0, verified, now)
        assert abs(factor - 0.25) < 1e-10

    def test_three_halflives(self):
        now = utc_now()
        verified = now - timedelta(days=30)
        factor = compute_age_factor(10.0, verified, now)
        assert abs(factor - 0.125) < 1e-10

    def test_very_short_halflife(self):
        """Sentiment (0.3 days = 7.2 hours): after 1 day = ~3.3 half-lives."""
        now = utc_now()
        verified = now - timedelta(days=1)
        factor = compute_age_factor(0.3, verified, now)
        expected = math.pow(2, -1.0 / 0.3)
        assert abs(factor - expected) < 1e-10
        assert factor < 0.1  # Should be very stale

    def test_very_long_halflife(self):
        """Physical law (36500 days): after 1 year barely decays."""
        now = utc_now()
        verified = now - timedelta(days=365)
        factor = compute_age_factor(36500.0, verified, now)
        assert factor > 0.99  # Barely decayed

    def test_zero_halflife_returns_zero(self):
        now = utc_now()
        verified = now - timedelta(days=1)
        factor = compute_age_factor(0.0, verified, now)
        assert factor == 0.0

    def test_future_verified_returns_one(self):
        now = utc_now()
        verified = now + timedelta(days=1)
        factor = compute_age_factor(10.0, verified, now)
        assert factor == 1.0


# ─── Staleness Detection ────────────────────────────────────────────────────


class TestIsStale:
    def test_fresh_belief_not_stale(self):
        now = utc_now()
        verified = now - timedelta(days=5)
        # 10-day half-life, 5 days elapsed → age_factor ≈ 0.707
        assert is_stale(10.0, verified, now=now) is False

    def test_old_belief_is_stale(self):
        now = utc_now()
        verified = now - timedelta(days=15)
        # 10-day half-life, 15 days elapsed → age_factor ≈ 0.354
        assert is_stale(10.0, verified, now=now) is True

    def test_exactly_one_halflife(self):
        now = utc_now()
        verified = now - timedelta(days=10)
        # age_factor = 0.5, threshold = 0.5 → NOT stale (< means strictly below)
        assert is_stale(10.0, verified, now=now) is False

    def test_custom_threshold(self):
        now = utc_now()
        verified = now - timedelta(days=5)
        # age_factor ≈ 0.707, with threshold 0.8 → stale
        assert is_stale(10.0, verified, threshold=0.8, now=now) is True


# ─── BeliefAgingScanner ─────────────────────────────────────────────────────


def make_mock_neo4j() -> MagicMock:
    neo4j = MagicMock()
    neo4j.execute_read = AsyncMock(return_value=[])
    neo4j.execute_write = AsyncMock()
    return neo4j


class TestBeliefAgingScanner:
    @pytest.mark.asyncio
    async def test_scan_empty_graph(self):
        neo4j = make_mock_neo4j()
        scanner = BeliefAgingScanner(neo4j)
        result = await scanner.scan_stale_beliefs()

        assert result.beliefs_scanned == 0
        assert result.beliefs_stale == 0
        assert result.stale_beliefs == []

    @pytest.mark.asyncio
    async def test_scan_finds_stale_beliefs(self):
        now = utc_now()
        old_time = (now - timedelta(days=15)).isoformat()

        neo4j = make_mock_neo4j()
        neo4j.execute_read.return_value = [
            {
                "belief_id": "belief_1",
                "domain": "sentiment",
                "statement": "Alice is happy",
                "half_life_days": 0.3,
                "last_verified": old_time,
                "precision": 0.8,
            },
            {
                "belief_id": "belief_2",
                "domain": "capability",
                "statement": "System supports Python 3.12",
                "half_life_days": 90.0,
                "last_verified": old_time,
                "precision": 0.9,
            },
        ]

        scanner = BeliefAgingScanner(neo4j)
        result = await scanner.scan_stale_beliefs()

        assert result.beliefs_scanned == 2
        # Sentiment (0.3 day half-life) after 15 days = definitely stale
        # Capability (90 day half-life) after 15 days = not stale (age_factor ≈ 0.89)
        assert result.beliefs_stale == 1
        assert result.stale_beliefs[0].belief_id == "belief_1"
        assert result.stale_beliefs[0].domain == "sentiment"

    @pytest.mark.asyncio
    async def test_scan_identifies_critical_beliefs(self):
        """Critical = age_factor < 0.25 (two half-lives elapsed)."""
        now = utc_now()
        very_old = (now - timedelta(days=30)).isoformat()

        neo4j = make_mock_neo4j()
        neo4j.execute_read.return_value = [
            {
                "belief_id": "belief_critical",
                "domain": "sentiment",
                "statement": "Old sentiment",
                "half_life_days": 0.3,
                "last_verified": very_old,
                "precision": 0.9,
            },
        ]

        scanner = BeliefAgingScanner(neo4j)
        result = await scanner.scan_stale_beliefs()

        assert result.beliefs_critical == 1
        assert result.stale_beliefs[0].age_factor < 0.01  # Very decayed

    @pytest.mark.asyncio
    async def test_scan_sorts_by_priority(self):
        now = utc_now()
        neo4j = make_mock_neo4j()
        neo4j.execute_read.return_value = [
            {
                "belief_id": "low_priority",
                "domain": "opinion",
                "statement": "Low priority",
                "half_life_days": 7.0,
                "last_verified": (now - timedelta(days=8)).isoformat(),
                "precision": 0.3,
            },
            {
                "belief_id": "high_priority",
                "domain": "sentiment",
                "statement": "High priority",
                "half_life_days": 0.3,
                "last_verified": (now - timedelta(days=5)).isoformat(),
                "precision": 0.95,
            },
        ]

        scanner = BeliefAgingScanner(neo4j)
        result = await scanner.scan_stale_beliefs()

        assert result.beliefs_stale == 2
        # High precision + lower age_factor = higher priority
        assert result.stale_beliefs[0].belief_id == "high_priority"

    @pytest.mark.asyncio
    async def test_mark_verified(self):
        neo4j = make_mock_neo4j()
        scanner = BeliefAgingScanner(neo4j)
        await scanner.mark_verified("belief_1")

        neo4j.execute_write.assert_called_once()
        args = neo4j.execute_write.call_args
        assert "belief_1" in str(args)

    @pytest.mark.asyncio
    async def test_scan_handles_neo4j_error(self):
        neo4j = make_mock_neo4j()
        neo4j.execute_read.side_effect = RuntimeError("Connection refused")

        scanner = BeliefAgingScanner(neo4j)
        result = await scanner.scan_stale_beliefs()

        # Should return empty result, not raise
        assert result.beliefs_scanned == 0
        assert result.beliefs_stale == 0


# ─── Unreliable-in-N-Hours Query ────────────────────────────────────────────


class TestUnreliableInQuery:
    @pytest.mark.asyncio
    async def test_finds_beliefs_crossing_threshold(self):
        now = utc_now()
        # 7-day half-life, verified 6 days ago → currently age_factor ≈ 0.558 (fresh)
        # In 48 hours (8 days total) → age_factor ≈ 0.447 (stale!)
        six_days_ago = (now - timedelta(days=6)).isoformat()

        neo4j = make_mock_neo4j()
        neo4j.execute_read.return_value = [
            {
                "belief_id": "will_be_stale",
                "domain": "opinion",
                "statement": "Will cross threshold in 48h",
                "half_life_days": 7.0,
                "last_verified": six_days_ago,
                "precision": 0.7,
            },
        ]

        scanner = BeliefAgingScanner(neo4j)
        result = await scanner.query_unreliable_in(hours=48.0)

        assert len(result) == 1
        assert result[0].belief_id == "will_be_stale"

    @pytest.mark.asyncio
    async def test_excludes_already_stale(self):
        """Already stale beliefs aren't included (they're already known stale)."""
        now = utc_now()
        # 7-day half-life, verified 10 days ago → already stale
        ten_days_ago = (now - timedelta(days=10)).isoformat()

        neo4j = make_mock_neo4j()
        neo4j.execute_read.return_value = [
            {
                "belief_id": "already_stale",
                "domain": "opinion",
                "statement": "Already past threshold",
                "half_life_days": 7.0,
                "last_verified": ten_days_ago,
                "precision": 0.7,
            },
        ]

        scanner = BeliefAgingScanner(neo4j)
        result = await scanner.query_unreliable_in(hours=48.0)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_excludes_still_fresh(self):
        """Beliefs that won't cross threshold within the window."""
        now = utc_now()
        # 90-day half-life, verified 1 day ago → won't be stale in 48h
        one_day_ago = (now - timedelta(days=1)).isoformat()

        neo4j = make_mock_neo4j()
        neo4j.execute_read.return_value = [
            {
                "belief_id": "still_fresh",
                "domain": "capability",
                "statement": "Won't cross threshold",
                "half_life_days": 90.0,
                "last_verified": one_day_ago,
                "precision": 0.9,
            },
        ]

        scanner = BeliefAgingScanner(neo4j)
        result = await scanner.query_unreliable_in(hours=48.0)

        assert len(result) == 0


# ─── Belief Stamping ────────────────────────────────────────────────────────


class TestStampBeliefHalflife:
    @pytest.mark.asyncio
    async def test_stamps_with_domain_halflife(self):
        neo4j = make_mock_neo4j()
        await stamp_belief_halflife(neo4j, "belief_1", "sentiment")

        neo4j.execute_write.assert_called_once()
        call_args = neo4j.execute_write.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("parameters", {})
        assert params["half_life_days"] == 0.3
        assert params["domain"] == "sentiment"

    @pytest.mark.asyncio
    async def test_stamps_with_explicit_halflife(self):
        neo4j = make_mock_neo4j()
        await stamp_belief_halflife(neo4j, "belief_1", "custom", half_life_days=42.0)

        call_args = neo4j.execute_write.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("parameters", {})
        assert params["half_life_days"] == 42.0

    @pytest.mark.asyncio
    async def test_stamps_handles_error(self):
        neo4j = make_mock_neo4j()
        neo4j.execute_write.side_effect = RuntimeError("DB down")

        # Should not raise
        await stamp_belief_halflife(neo4j, "belief_1", "sentiment")


# ─── Consolidation Integration ──────────────────────────────────────────────


class TestConsolidationIntegration:
    @pytest.mark.asyncio
    async def test_phase_belief_aging_runs(self):
        """Verify Phase 2.5 runs the aging scanner during consolidation."""
        from systems.evo.consolidation import ConsolidationOrchestrator
        from systems.evo.types import PatternContext

        # Build mock aging scanner
        mock_scanner = MagicMock(spec=BeliefAgingScanner)
        mock_scanner.scan_stale_beliefs = AsyncMock(
            return_value=BeliefAgingResult(
                beliefs_scanned=100,
                beliefs_stale=5,
                beliefs_critical=1,
            )
        )

        # Build mock subsystems
        engine = MagicMock()
        engine.get_all_active.return_value = []
        engine.get_supported.return_value = []

        tuner = MagicMock()
        tuner.begin_cycle.return_value = None

        extractor = MagicMock()
        extractor.begin_cycle.return_value = None
        extractor.extract_procedure = AsyncMock(return_value=None)

        self_model = MagicMock()
        self_model.update = AsyncMock()
        self_model.get_current.return_value = MagicMock(
            success_rate=0.8, mean_alignment=0.7
        )

        orchestrator = ConsolidationOrchestrator(
            hypothesis_engine=engine,
            parameter_tuner=tuner,
            procedure_extractor=extractor,
            self_model_manager=self_model,
            belief_aging_scanner=mock_scanner,
        )

        context = PatternContext()
        result = await orchestrator.run(context)

        mock_scanner.scan_stale_beliefs.assert_called_once()
        assert result.beliefs_stale == 5
        assert result.beliefs_critical == 1

    @pytest.mark.asyncio
    async def test_phase_belief_aging_skipped_without_scanner(self):
        """Without a scanner, Phase 2.5 is a no-op."""
        from systems.evo.consolidation import ConsolidationOrchestrator
        from systems.evo.types import PatternContext

        engine = MagicMock()
        engine.get_all_active.return_value = []
        engine.get_supported.return_value = []

        tuner = MagicMock()
        tuner.begin_cycle.return_value = None

        extractor = MagicMock()
        extractor.begin_cycle.return_value = None

        self_model = MagicMock()
        self_model.update = AsyncMock()
        self_model.get_current.return_value = MagicMock(
            success_rate=0.8, mean_alignment=0.7
        )

        orchestrator = ConsolidationOrchestrator(
            hypothesis_engine=engine,
            parameter_tuner=tuner,
            procedure_extractor=extractor,
            self_model_manager=self_model,
            belief_aging_scanner=None,
        )

        context = PatternContext()
        result = await orchestrator.run(context)

        assert result.beliefs_stale == 0
        assert result.beliefs_critical == 0
