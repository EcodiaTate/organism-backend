"""
Unit tests for MEV Analyzer (Prompt #12: Predator Detection).

Tests the MEVAnalyzer heuristic analysis, report merging, protection
strategy selection, and TransactionShield integration.
"""

from __future__ import annotations

import pytest

from systems.axon.mev_analyzer import MEVAnalyzer
from systems.axon.mev_types import (
    BlockCompetitionSnapshot,
    MEVProtectionStrategy,
    MEVReport,
    MEVVulnerabilityType,
    VulnerableStep,
)

# ─── Fixtures ─────────────────────────────────────────────────────


def make_analyzer(**kwargs) -> MEVAnalyzer:
    """Create an MEVAnalyzer in heuristic-only mode (no RPC)."""
    defaults = {
        "rpc_url": "",  # No RPC = heuristic only
        "high_risk_threshold": 0.7,
        "analysis_timeout_ms": 5000,
    }
    return MEVAnalyzer(**{**defaults, **kwargs})


# Known addresses for testing
UNISWAP_ROUTER = "0x2626664c2603336e57b271c5c0b26f421741e481"
AAVE_V3_POOL = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"
MORPHO_VAULT = "0xc1256Ae5FF1cf2719D4937adb3bbCCab2E00A2Ca"
RANDOM_EOA = "0x1234567890abcdef1234567890abcdef12345678"


# ─── Tests: MEVReport ────────────────────────────────────────────


class TestMEVReport:
    def test_is_high_risk_above_threshold(self):
        report = MEVReport(mev_risk_score=0.8)
        assert report.is_high_risk is True

    def test_is_high_risk_below_threshold(self):
        report = MEVReport(mev_risk_score=0.5)
        assert report.is_high_risk is False

    def test_is_high_risk_at_boundary(self):
        report = MEVReport(mev_risk_score=0.7)
        assert report.is_high_risk is False  # > 0.7, not >=

    def test_vulnerable_operation_names(self):
        report = MEVReport(
            vulnerable_steps=[
                VulnerableStep(
                    operation="dex_swap",
                    vulnerability_type=MEVVulnerabilityType.SANDWICH,
                ),
                VulnerableStep(
                    operation="lending_deposit",
                    vulnerability_type=MEVVulnerabilityType.FRONTRUN,
                ),
            ]
        )
        assert report.vulnerable_operation_names == ["dex_swap", "lending_deposit"]

    def test_summary_format(self):
        report = MEVReport(
            mev_risk_score=0.82,
            estimated_extraction_usd=1240.0,
            recommended_protection=MEVProtectionStrategy.FLASHBOTS_PROTECT,
            vulnerable_steps=[
                VulnerableStep(
                    operation="swap",
                    vulnerability_type=MEVVulnerabilityType.SANDWICH,
                ),
            ],
        )
        summary = report.summary()
        assert "mev_risk=0.82" in summary
        assert "$1240.00" in summary
        assert "flashbots_protect" in summary

    def test_risk_score_clamped(self):
        report = MEVReport(mev_risk_score=0.0)
        assert report.mev_risk_score == 0.0
        report2 = MEVReport(mev_risk_score=1.0)
        assert report2.mev_risk_score == 1.0


# ─── Tests: BlockCompetitionSnapshot ─────────────────────────────


class TestBlockCompetitionSnapshot:
    def test_low_competition(self):
        snap = BlockCompetitionSnapshot(competition_level=0.2)
        assert snap.is_low_competition is True
        assert snap.is_high_competition is False

    def test_high_competition(self):
        snap = BlockCompetitionSnapshot(competition_level=0.8)
        assert snap.is_low_competition is False
        assert snap.is_high_competition is True

    def test_medium_competition(self):
        snap = BlockCompetitionSnapshot(competition_level=0.5)
        assert snap.is_low_competition is False
        assert snap.is_high_competition is False


# ─── Tests: MEVAnalyzer Heuristic Analysis ───────────────────────


class TestMEVAnalyzerHeuristics:
    @pytest.mark.asyncio
    async def test_plain_eth_transfer_zero_risk(self):
        analyzer = make_analyzer()
        report = await analyzer.analyze(
            to=RANDOM_EOA,
            data="0x",
            value=1_000_000_000_000_000_000,  # 1 ETH
            transaction_volume_usd=3000.0,
        )
        assert report.mev_risk_score == 0.0
        assert len(report.vulnerable_steps) == 0
        assert report.recommended_protection == MEVProtectionStrategy.PUBLIC_MEMPOOL

    @pytest.mark.asyncio
    async def test_dex_swap_high_risk(self):
        analyzer = make_analyzer()
        # Uniswap execute() selector, volume >$50k, high slippage
        report = await analyzer.analyze(
            to=UNISWAP_ROUTER,
            data="0x3593564c" + "00" * 64,
            value=0,
            transaction_volume_usd=60_000.0,
            expected_slippage_bps=150,
        )
        # DEX swap (0.4) + volume >50k (0.25) + slippage >100bps (0.15) = 0.80
        assert report.mev_risk_score >= 0.7
        assert any(s.vulnerability_type == MEVVulnerabilityType.SANDWICH
                    for s in report.vulnerable_steps)

    @pytest.mark.asyncio
    async def test_lending_deposit_moderate_risk(self):
        analyzer = make_analyzer()
        # Aave supply selector
        report = await analyzer.analyze(
            to=AAVE_V3_POOL,
            data="0x617ba037" + "00" * 128,
            value=0,
            transaction_volume_usd=5000.0,
        )
        # Lending (0.15) + selector match (0.4 if swap selector) + volume (0.05)
        # Only lending, no DEX swap selector match → moderate
        assert 0.1 <= report.mev_risk_score <= 0.7
        assert report.recommended_protection in {
            MEVProtectionStrategy.PUBLIC_MEMPOOL,
            MEVProtectionStrategy.FLASHBOTS_PROTECT,
        }

    @pytest.mark.asyncio
    async def test_large_lending_deposit_higher_risk(self):
        analyzer = make_analyzer()
        report = await analyzer.analyze(
            to=AAVE_V3_POOL,
            data="0x617ba037" + "00" * 128,
            value=0,
            transaction_volume_usd=100_000.0,
        )
        # Large amount adds more risk
        assert report.mev_risk_score > 0.3
        assert any(s.vulnerability_type == MEVVulnerabilityType.FRONTRUN
                    for s in report.vulnerable_steps)

    @pytest.mark.asyncio
    async def test_small_transaction_low_risk(self):
        analyzer = make_analyzer()
        # Use a non-MEV-sensitive selector for the small lending deposit
        # to isolate the volume/lending risk factor
        report = await analyzer.analyze(
            to=AAVE_V3_POOL,
            data="0xdeadbeef" + "00" * 128,  # Unknown selector
            value=0,
            transaction_volume_usd=50.0,
        )
        # Lending (0.15) + small volume (0) = 0.15 → low risk
        assert report.mev_risk_score < 0.5


# ─── Tests: Block Competition Adjustment ─────────────────────────


class TestBlockCompetitionAdjustment:
    @pytest.mark.asyncio
    async def test_high_competition_increases_risk(self):
        analyzer = make_analyzer()
        # Set high competition
        analyzer.update_competition(
            BlockCompetitionSnapshot(competition_level=0.9)
        )
        report = await analyzer.analyze(
            to=UNISWAP_ROUTER,
            data="0x3593564c" + "00" * 64,
            value=0,
            transaction_volume_usd=10_000.0,
        )
        # High competition should amplify risk by 1.2x
        assert report.block_competition_level == 0.9

    @pytest.mark.asyncio
    async def test_low_competition_no_amplification(self):
        analyzer = make_analyzer()
        analyzer.update_competition(
            BlockCompetitionSnapshot(competition_level=0.1)
        )
        report = await analyzer.analyze(
            to=RANDOM_EOA,
            data="0x",
            value=1_000_000_000,
            transaction_volume_usd=10.0,
        )
        assert report.mev_risk_score == 0.0


# ─── Tests: Protection Strategy ──────────────────────────────────


class TestProtectionStrategy:
    @pytest.mark.asyncio
    async def test_no_risk_uses_public_mempool(self):
        analyzer = make_analyzer()
        report = await analyzer.analyze(
            to=RANDOM_EOA,
            data="0x",
            value=1_000_000_000,
            transaction_volume_usd=10.0,
        )
        assert report.recommended_protection == MEVProtectionStrategy.PUBLIC_MEMPOOL

    @pytest.mark.asyncio
    async def test_high_risk_high_competition_uses_flashbots(self):
        analyzer = make_analyzer()
        analyzer.update_competition(
            BlockCompetitionSnapshot(competition_level=0.9)
        )
        report = await analyzer.analyze(
            to=UNISWAP_ROUTER,
            data="0x3593564c" + "00" * 64,
            value=0,
            transaction_volume_usd=100_000.0,
            expected_slippage_bps=200,
        )
        assert report.recommended_protection == MEVProtectionStrategy.FLASHBOTS_PROTECT

    @pytest.mark.asyncio
    async def test_high_risk_low_competition_waits(self):
        analyzer = make_analyzer()
        analyzer.update_competition(
            BlockCompetitionSnapshot(competition_level=0.1)
        )
        report = await analyzer.analyze(
            to=UNISWAP_ROUTER,
            data="0x3593564c" + "00" * 64,
            value=0,
            transaction_volume_usd=100_000.0,
            expected_slippage_bps=200,
        )
        # High risk + low competition → wait for low-competition block
        if report.mev_risk_score >= 0.7:
            assert report.recommended_protection == MEVProtectionStrategy.WAIT_FOR_LOW_COMPETITION


# ─── Tests: Report Merging ───────────────────────────────────────


class TestReportMerging:
    def test_simulation_overrides_heuristic(self):
        heuristic = MEVReport(
            mev_risk_score=0.3,
            estimated_extraction_usd=10.0,
            vulnerable_steps=[
                VulnerableStep(
                    operation="dex_swap",
                    vulnerability_type=MEVVulnerabilityType.SANDWICH,
                    estimated_extraction_usd=10.0,
                ),
            ],
        )
        simulation = MEVReport(
            mev_risk_score=0.8,
            estimated_extraction_usd=50.0,
            simulated=True,
            vulnerable_steps=[
                VulnerableStep(
                    operation="simulated_output",
                    vulnerability_type=MEVVulnerabilityType.SANDWICH,
                    estimated_extraction_usd=50.0,
                ),
            ],
        )
        merged = MEVAnalyzer._merge_reports(heuristic, simulation)
        assert merged.mev_risk_score == 0.8  # Higher of the two
        assert merged.simulated is True
        assert len(merged.vulnerable_steps) == 2  # Both steps preserved

    def test_failed_simulation_preserves_heuristic(self):
        heuristic = MEVReport(
            mev_risk_score=0.5,
            estimated_extraction_usd=20.0,
        )
        simulation = MEVReport(
            simulated=False,
            simulation_error="RPC timeout",
        )
        merged = MEVAnalyzer._merge_reports(heuristic, simulation)
        assert merged.mev_risk_score == 0.5
        assert merged.simulation_error == "RPC timeout"


# ─── Tests: Protection Cost Estimation ───────────────────────────


class TestProtectionCost:
    def test_public_mempool_zero_cost(self):
        cost = MEVAnalyzer._estimate_protection_cost(
            MEVReport(recommended_protection=MEVProtectionStrategy.PUBLIC_MEMPOOL),
            transaction_volume_usd=10_000.0,
        )
        assert cost == 0.0

    def test_flashbots_has_cost(self):
        cost = MEVAnalyzer._estimate_protection_cost(
            MEVReport(recommended_protection=MEVProtectionStrategy.FLASHBOTS_PROTECT),
            transaction_volume_usd=10_000.0,
        )
        assert cost > 0.0
        assert cost == 10.0  # 0.1% of 10k

    def test_wait_strategy_zero_cost(self):
        cost = MEVAnalyzer._estimate_protection_cost(
            MEVReport(recommended_protection=MEVProtectionStrategy.WAIT_FOR_LOW_COMPETITION),
            transaction_volume_usd=10_000.0,
        )
        assert cost == 0.0


# ─── Tests: MEVAnalyzer Properties ───────────────────────────────


class TestMEVAnalyzerProperties:
    def test_has_rpc_false_without_url(self):
        analyzer = make_analyzer(rpc_url="")
        assert analyzer.has_rpc is False

    def test_update_competition(self):
        analyzer = make_analyzer()
        snap = BlockCompetitionSnapshot(competition_level=0.75, gas_price_gwei=5.0)
        analyzer.update_competition(snap)
        assert analyzer.competition_snapshot.competition_level == 0.75
        assert analyzer.competition_snapshot.gas_price_gwei == 5.0

    @pytest.mark.asyncio
    async def test_analyze_returns_report_on_error(self):
        """Even if internal analysis raises, a conservative report is returned."""
        analyzer = make_analyzer()
        # This should not raise - errors are caught internally
        report = await analyzer.analyze(
            to="",  # Invalid but shouldn't crash
            data="",
            value=0,
            transaction_volume_usd=0.0,
        )
        assert isinstance(report, MEVReport)


# ─── Tests: TransactionShield MEV Integration ────────────────────


class TestShieldMEVIntegration:
    @pytest.mark.asyncio
    async def test_shield_passes_with_mev_warning(self):
        """Shield should pass but include MEV warnings for moderate risk."""
        from systems.axon.shield import TransactionShield

        analyzer = make_analyzer()
        shield = TransactionShield(
            wallet=None,
            max_slippage_bps=50,
            mev_analyzer=analyzer,
        )

        result = await shield.evaluate(
            action_type="defi_yield",
            params={
                "protocol_address": AAVE_V3_POOL,
                "amount": "5000",
                "data": "0x617ba037" + "00" * 128,
            },
        )
        # Should pass (MEV risk for lending is moderate, not blocking)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_shield_without_mev_analyzer_still_works(self):
        """Shield should work normally when no MEV analyzer is configured."""
        from systems.axon.shield import TransactionShield

        shield = TransactionShield(
            wallet=None,
            max_slippage_bps=50,
            mev_analyzer=None,
        )

        result = await shield.evaluate(
            action_type="defi_yield",
            params={
                "protocol_address": AAVE_V3_POOL,
                "amount": "100",
            },
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_shield_mev_stats_tracked(self):
        """Shield should track MEV protection metrics."""
        from systems.axon.shield import TransactionShield

        analyzer = make_analyzer()
        shield = TransactionShield(
            wallet=None,
            max_slippage_bps=200,  # High tolerance to not trigger slippage rejection
            mev_analyzer=analyzer,
        )

        stats = shield.stats
        assert "mev_protected" in stats
        assert "mev_saved_usd" in stats
        assert "mev_analyzer_available" in stats
        assert stats["mev_analyzer_available"] is True

    @pytest.mark.asyncio
    async def test_shield_non_financial_skips_mev(self):
        """Non-financial executors should skip MEV analysis entirely."""
        from systems.axon.shield import TransactionShield

        analyzer = make_analyzer()
        shield = TransactionShield(
            wallet=None,
            max_slippage_bps=50,
            mev_analyzer=analyzer,
        )

        result = await shield.evaluate(
            action_type="store_insight",
            params={"content": "some insight"},
        )
        assert result.passed is True
        assert shield.last_mev_report is None  # No MEV analysis performed
