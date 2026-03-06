"""
EcodiaOS — Oikos: Monte Carlo Treasury Threat Modeler (Phase 16i+)

Per-asset Monte Carlo threat modeling with contagion coupling,
liquidation detection, and hedging proposal generation.

Unlike EconomicSimulator (Phase 16i) which models organism-level
cashflow via GBM, this modeler treats each treasury asset independently
with per-asset shock distributions and a time-varying correlation matrix
that spikes during crises — capturing the real-world phenomenon where
diversification fails exactly when you need it most.

Design choices:
  - NumPy for vectorised path simulation (5,000 paths × 90 days × N assets)
  - Decimal for inputs/outputs (financial precision), float internally
  - Cholesky decomposition for correlated samples, with eigenvalue-clamping
    fallback when stress overwrites make the matrix non-positive-definite
  - No I/O — pure computation, safe for asyncio event loop via run_in_executor
  - Deterministic seed option for reproducible testing
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import structlog

from systems.oikos.threat_modeling_types import (
    AssetClass,
    AssetShockDistribution,
    ContagionEdge,
    CriticalExposure,
    HedgingProposal,
    TailRiskProfile,
    ThreatModelResult,
    TreasuryPosition,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from config import OikosConfig
    from systems.oikos.models import EconomicState

logger = structlog.get_logger("oikos.threat_modeler")


def _d(v: Decimal) -> float:
    """Decimal to float for vectorised math."""
    return float(v)


def _to_dec(v: float) -> Decimal:
    """float to Decimal for output precision."""
    return Decimal(str(round(v, 6)))


# ─── Default Shock Distributions ──────────────────────────────────

# Calibrated defaults per known asset. These are baseline parameters —
# real implementations should calibrate from historical data.

DEFAULT_SHOCK_DISTRIBUTIONS: dict[str, AssetShockDistribution] = {
    "ETH": AssetShockDistribution(
        asset_class=AssetClass.NATIVE_ETH,
        symbol="ETH",
        daily_drift=Decimal("0.0001"),
        daily_volatility=Decimal("0.04"),
        jump_probability=Decimal("0.01"),
        jump_mean=Decimal("-0.15"),
        jump_std=Decimal("0.10"),
    ),
    "stETH": AssetShockDistribution(
        asset_class=AssetClass.LST,
        symbol="stETH",
        daily_drift=Decimal("0.0001"),
        daily_volatility=Decimal("0.04"),
        jump_probability=Decimal("0.01"),
        jump_mean=Decimal("-0.18"),
        jump_std=Decimal("0.12"),
    ),
    "USDC": AssetShockDistribution(
        asset_class=AssetClass.STABLECOIN,
        symbol="USDC",
        daily_drift=Decimal("0"),
        daily_volatility=Decimal("0.001"),
        jump_probability=Decimal("0.0001"),
        jump_mean=Decimal("-0.02"),
        jump_std=Decimal("0.01"),
        depeg_probability=Decimal("0.0005"),
        depeg_magnitude=Decimal("0.05"),
        depeg_recovery_days=5,
    ),
    "DAI": AssetShockDistribution(
        asset_class=AssetClass.STABLECOIN,
        symbol="DAI",
        daily_drift=Decimal("0"),
        daily_volatility=Decimal("0.002"),
        jump_probability=Decimal("0.0002"),
        jump_mean=Decimal("-0.03"),
        jump_std=Decimal("0.015"),
        depeg_probability=Decimal("0.001"),
        depeg_magnitude=Decimal("0.08"),
        depeg_recovery_days=7,
    ),
}


# ─── Default Contagion Edges ──────────────────────────────────────

DEFAULT_CONTAGION_EDGES: list[ContagionEdge] = [
    ContagionEdge(
        source_symbol="ETH",
        target_symbol="stETH",
        base_correlation=Decimal("0.92"),
        stress_correlation=Decimal("0.99"),
        stress_threshold=Decimal("-0.10"),
    ),
    ContagionEdge(
        source_symbol="ETH",
        target_symbol="USDC",
        base_correlation=Decimal("0.1"),
        stress_correlation=Decimal("0.5"),
        stress_threshold=Decimal("-0.20"),
    ),
    ContagionEdge(
        source_symbol="stETH",
        target_symbol="ETH",
        base_correlation=Decimal("0.92"),
        stress_correlation=Decimal("0.99"),
        stress_threshold=Decimal("-0.15"),
    ),
]


# ─── Symbol Inference ─────────────────────────────────────────────

# Heuristic mapping from protocol/pool strings to asset symbols.
_PROTOCOL_SYMBOL_MAP: dict[str, str] = {
    "lido": "stETH",
    "rocket_pool": "rETH",
    "rocketpool": "rETH",
}

_POOL_KEYWORDS: dict[str, str] = {
    "eth": "ETH",
    "steth": "stETH",
    "reth": "rETH",
    "usdc": "USDC",
    "usdt": "USDT",
    "dai": "DAI",
    "weth": "ETH",
}


def _infer_symbol(protocol: str, pool: str) -> str:
    """Infer asset symbol from protocol/pool names."""
    proto_lower = protocol.lower()
    if proto_lower in _PROTOCOL_SYMBOL_MAP:
        return _PROTOCOL_SYMBOL_MAP[proto_lower]

    pool_lower = pool.lower()
    # Try to find first matching keyword in pool name
    for keyword, symbol in _POOL_KEYWORDS.items():
        if keyword in pool_lower:
            return symbol

    return "UNKNOWN"


def _infer_asset_class(symbol: str) -> AssetClass:
    """Infer AssetClass from symbol."""
    stables = {"USDC", "USDT", "DAI", "FRAX", "BUSD"}
    lsts = {"stETH", "rETH", "cbETH", "wstETH"}

    if symbol in stables:
        return AssetClass.STABLECOIN
    if symbol in lsts:
        return AssetClass.LST
    if symbol == "ETH":
        return AssetClass.NATIVE_ETH
    return AssetClass.YIELD_BEARING


def _ensure_positive_definite(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Ensure a correlation matrix is positive definite.

    When stress overwrites make the matrix non-PD, clamp negative
    eigenvalues to a small positive floor and reconstruct.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = 1e-6
    clamped = np.maximum(eigenvalues, floor)
    reconstructed = eigenvectors @ np.diag(clamped) @ eigenvectors.T
    # Normalize diagonal back to 1.0 (correlation matrix)
    diag_sqrt = np.sqrt(np.diag(reconstructed))
    outer = np.outer(diag_sqrt, diag_sqrt)
    result: NDArray[np.float64] = reconstructed / outer
    return result


# ─── Monte Carlo Threat Modeler ───────────────────────────────────


class MonteCarloThreatModeler:
    """
    Per-asset Monte Carlo threat modeler with contagion and liquidation detection.

    Thread-safety: run_threat_cycle() offloads CPU work via run_in_executor.
    """

    def __init__(self, config: OikosConfig, seed: int | None = None) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._logger = logger.bind(component="threat_modeler")

    # ── Public API ─────────────────────────────────────────────────

    async def run_threat_cycle(
        self,
        state: EconomicState,
        sleep_cycle_id: str = "",
    ) -> ThreatModelResult:
        """
        Run the full treasury threat modeling cycle:
          1. Extract positions from EconomicState
          2. Map to shock distributions and contagion edges
          3. Run vectorized Monte Carlo simulation
          4. Compute tail risk profiles (portfolio + per-position)
          5. Identify critical exposures
          6. Generate hedging proposals

        CPU-heavy work offloaded to a thread executor.
        """
        start = time.monotonic()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._simulate_all, state, sleep_cycle_id
        )

        result.duration_ms = int((time.monotonic() - start) * 1000)

        self._logger.info(
            "threat_model_complete",
            var_5pct=str(result.portfolio_risk.var_5pct),
            liquidation_prob=str(result.portfolio_risk.liquidation_probability),
            critical_exposures=len(result.critical_exposures),
            hedging_proposals=len(result.hedging_proposals),
            contagion_events=result.contagion_events_detected,
            positions=result.positions_analyzed,
            total_paths=result.total_paths_simulated,
            duration_ms=result.duration_ms,
        )

        return result

    # ── Core Simulation (CPU-bound, runs in executor) ──────────────

    def _simulate_all(
        self,
        state: EconomicState,
        sleep_cycle_id: str,
    ) -> ThreatModelResult:
        """Synchronous entry point for the full simulation."""
        positions = self._extract_positions(state)

        if not positions:
            return ThreatModelResult(
                sleep_cycle_id=sleep_cycle_id,
                positions_analyzed=0,
            )

        n_paths = self._config.threat_model_paths
        horizon = self._config.threat_model_horizon_days

        # Map positions to distributions
        distributions = self._map_distributions(positions)
        edges = self._filter_contagion_edges(positions)

        # Run Monte Carlo
        price_paths = self._simulate_assets(
            positions, distributions, edges, n_paths, horizon
        )

        # Detect liquidations
        liquidations = self._detect_liquidations(price_paths, positions)

        # Compute tail risk
        portfolio_risk, position_risks = self._compute_tail_risk(
            price_paths, liquidations, positions
        )

        # Identify critical exposures
        critical_exposures = self._identify_critical_exposures(
            position_risks, positions, portfolio_risk
        )

        # Generate hedging proposals
        hedging_proposals = self._generate_hedging_proposals(
            critical_exposures, position_risks, positions
        )

        # Contagion statistics
        contagion_events = int(np.sum(np.any(liquidations, axis=0)))
        avg_amplifier = Decimal("1")
        if contagion_events > 0 and len(positions) > 1:
            # Rough amplifier: ratio of portfolio loss to sum of independent losses
            portfolio_returns = self._compute_portfolio_returns(price_paths, positions)
            correlated_var = float(np.percentile(portfolio_returns, 5))
            independent_var = sum(
                float(position_risks[p.position_id].var_5pct)
                for p in positions
                if p.position_id in position_risks
            )
            if independent_var < 0:
                avg_amplifier = _to_dec(max(1.0, correlated_var / independent_var))

        return ThreatModelResult(
            sleep_cycle_id=sleep_cycle_id,
            portfolio_risk=portfolio_risk,
            position_risks=position_risks,
            critical_exposures=critical_exposures,
            hedging_proposals=hedging_proposals,
            contagion_events_detected=contagion_events,
            contagion_loss_amplifier=avg_amplifier,
            total_paths_simulated=n_paths,
            horizon_days=horizon,
            positions_analyzed=len(positions),
        )

    # ── Position Extraction ────────────────────────────────────────

    def _extract_positions(self, state: EconomicState) -> list[TreasuryPosition]:
        """Convert EconomicState holdings into TreasuryPosition list."""
        positions: list[TreasuryPosition] = []

        # Liquid balance as USDC position
        if _d(state.liquid_balance) > 0:
            positions.append(TreasuryPosition(
                symbol="USDC",
                asset_class=AssetClass.STABLECOIN,
                principal_usd=state.liquid_balance,
                protocol="",
                pool="",
                chain_id=8453,
            ))

        # Survival reserve as separate USDC position
        if _d(state.survival_reserve) > 0:
            positions.append(TreasuryPosition(
                symbol="USDC",
                asset_class=AssetClass.STABLECOIN,
                principal_usd=state.survival_reserve,
                protocol="cold_wallet",
                pool="survival_reserve",
                chain_id=8453,
            ))

        # Yield positions
        for yp in state.yield_positions:
            symbol = _infer_symbol(yp.protocol, yp.pool)
            asset_class = _infer_asset_class(symbol)

            # Heuristic: Aave/Compound positions have liquidation thresholds
            has_liq = yp.protocol.lower() in {"aave", "compound", "morpho"}
            liq_price = Decimal("0")
            collateral_ratio = Decimal("0")
            min_collateral_ratio = Decimal("0")

            if has_liq and _d(yp.principal_usd) > 0:
                # Conservative estimate: assume 80% LTV = 1.25 min ratio
                min_collateral_ratio = Decimal("1.25")
                collateral_ratio = Decimal("1.50")  # Assume healthy start
                # Liquidation at ~20% below current price (simplified)
                liq_price = yp.principal_usd * Decimal("0.80")

            positions.append(TreasuryPosition(
                symbol=symbol,
                asset_class=asset_class,
                principal_usd=yp.principal_usd,
                protocol=yp.protocol,
                pool=yp.pool,
                chain_id=yp.chain_id,
                has_liquidation_threshold=has_liq,
                liquidation_price_usd=liq_price,
                collateral_ratio=collateral_ratio,
                min_collateral_ratio=min_collateral_ratio,
                health_status=yp.health_status,
            ))

        return positions

    # ── Distribution Mapping ───────────────────────────────────────

    def _map_distributions(
        self, positions: list[TreasuryPosition]
    ) -> list[AssetShockDistribution]:
        """Map each position to its shock distribution."""
        result: list[AssetShockDistribution] = []
        for pos in positions:
            if pos.symbol in DEFAULT_SHOCK_DISTRIBUTIONS:
                result.append(DEFAULT_SHOCK_DISTRIBUTIONS[pos.symbol])
            else:
                # Generic yield-bearing default
                result.append(AssetShockDistribution(
                    asset_class=pos.asset_class,
                    symbol=pos.symbol,
                    daily_drift=Decimal("0.0001"),
                    daily_volatility=Decimal("0.03"),
                    jump_probability=Decimal("0.005"),
                    jump_mean=Decimal("-0.10"),
                    jump_std=Decimal("0.08"),
                ))
        return result

    def _filter_contagion_edges(
        self, positions: list[TreasuryPosition]
    ) -> list[ContagionEdge]:
        """Return only contagion edges relevant to current positions."""
        symbols = {p.symbol for p in positions}
        return [
            edge for edge in DEFAULT_CONTAGION_EDGES
            if edge.source_symbol in symbols and edge.target_symbol in symbols
        ]

    # ── Monte Carlo Simulation ─────────────────────────────────────

    def _simulate_assets(
        self,
        positions: list[TreasuryPosition],
        distributions: list[AssetShockDistribution],
        edges: list[ContagionEdge],
        n_paths: int,
        horizon_days: int,
    ) -> NDArray[np.float64]:
        """
        Simulate per-asset price paths with contagion coupling.

        Returns shape (n_assets, n_paths, horizon_days+1) where
        index 0 is the initial price (1.0 normalised).
        """
        n_assets = len(positions)
        # Prices normalised to 1.0 at t=0
        prices = np.ones((n_assets, n_paths, horizon_days + 1), dtype=np.float64)

        # Build symbol-to-index map
        symbol_idx: dict[str, list[int]] = {}
        for i, pos in enumerate(positions):
            symbol_idx.setdefault(pos.symbol, []).append(i)

        # Pre-extract distribution params as float arrays
        drifts = np.array([_d(d.daily_drift) for d in distributions])
        vols = np.array([_d(d.daily_volatility) for d in distributions])
        jump_probs = np.array([_d(d.jump_probability) for d in distributions])
        jump_means = np.array([_d(d.jump_mean) for d in distributions])
        jump_stds = np.array([_d(d.jump_std) for d in distributions])
        depeg_probs = np.array([_d(d.depeg_probability) for d in distributions])
        depeg_mags = np.array([_d(d.depeg_magnitude) for d in distributions])
        depeg_recovery = np.array([d.depeg_recovery_days for d in distributions])

        # Build base correlation matrix
        base_corr = np.eye(n_assets, dtype=np.float64)
        stress_corr = np.eye(n_assets, dtype=np.float64)
        stress_thresholds: dict[tuple[int, int], float] = {}

        for edge in edges:
            src_indices = symbol_idx.get(edge.source_symbol, [])
            tgt_indices = symbol_idx.get(edge.target_symbol, [])
            for si in src_indices:
                for ti in tgt_indices:
                    if si != ti:
                        bc = _d(edge.base_correlation)
                        sc = _d(edge.stress_correlation)
                        base_corr[si, ti] = bc
                        base_corr[ti, si] = bc
                        stress_corr[si, ti] = sc
                        stress_corr[ti, si] = sc
                        stress_thresholds[(si, ti)] = _d(edge.stress_threshold)

        # Track depeg state per path
        depeg_active = np.zeros((n_assets, n_paths), dtype=np.float64)  # days remaining
        depeg_depth = np.zeros((n_assets, n_paths), dtype=np.float64)

        for day in range(horizon_days):
            # Determine which assets are stressed (drawdown from peak)
            current_prices = prices[:, :, day]  # (n_assets, n_paths)
            peak_prices = np.maximum.accumulate(prices[:, :, :day + 1], axis=2)[:, :, -1]
            drawdowns = (current_prices - peak_prices) / np.maximum(peak_prices, 1e-10)
            # Mean drawdown per asset across paths (scalar per asset)
            mean_drawdowns = np.mean(drawdowns, axis=1)  # (n_assets,)

            # Build today's correlation matrix
            corr_today = base_corr.copy()
            for (si, ti), threshold in stress_thresholds.items():
                if mean_drawdowns[si] < threshold:
                    corr_today[si, ti] = stress_corr[si, ti]
                    corr_today[ti, si] = stress_corr[ti, si]

            # Ensure positive-definite
            try:
                cholesky = np.linalg.cholesky(corr_today)
            except np.linalg.LinAlgError:
                corr_today = _ensure_positive_definite(corr_today)
                cholesky = np.linalg.cholesky(corr_today)

            # Generate correlated normals
            z_uncorr = self._rng.standard_normal((n_assets, n_paths))
            z_corr = cholesky @ z_uncorr  # (n_assets, n_paths)

            # GBM step
            log_returns = (
                drifts[:, np.newaxis]
                - 0.5 * vols[:, np.newaxis] ** 2
                + vols[:, np.newaxis] * z_corr
            )

            # Jump-diffusion
            jump_events = self._rng.uniform(0, 1, (n_assets, n_paths)) < jump_probs[:, np.newaxis]
            jump_sizes = self._rng.normal(
                jump_means[:, np.newaxis],
                np.maximum(jump_stds[:, np.newaxis], 1e-10),
                (n_assets, n_paths),
            )
            log_returns += np.where(jump_events, jump_sizes, 0.0)

            # Depeg dynamics for stablecoins
            for a in range(n_assets):
                if depeg_probs[a] > 0:
                    # Check for new depeg events
                    new_depeg = (
                        (depeg_active[a] <= 0)
                        & (self._rng.uniform(0, 1, n_paths) < depeg_probs[a])
                    )
                    depeg_active[a] = np.where(new_depeg, depeg_recovery[a], depeg_active[a])
                    depeg_depth[a] = np.where(new_depeg, depeg_mags[a], depeg_depth[a])

                    # Apply depeg: gradual recovery back to 1.0
                    actively_depegged = depeg_active[a] > 0
                    if np.any(actively_depegged):
                        # During depeg: force price toward (1 - magnitude) then recover
                        recovery_frac = 1.0 - (depeg_active[a] / max(depeg_recovery[a], 1))
                        target = 1.0 - depeg_depth[a] * (1.0 - recovery_frac)
                        current_norm = prices[a, :, day]
                        # Blend toward target
                        pull = np.where(actively_depegged, 0.3 * (target - current_norm), 0.0)
                        log_returns[a] += pull
                        depeg_active[a] = np.where(actively_depegged, depeg_active[a] - 1, 0.0)

            # Apply returns
            prices[:, :, day + 1] = prices[:, :, day] * np.exp(log_returns)

            # Floor at zero
            prices[:, :, day + 1] = np.maximum(prices[:, :, day + 1], 0.0)

        return prices

    # ── Liquidation Detection ──────────────────────────────────────

    def _detect_liquidations(
        self,
        price_paths: NDArray[np.float64],
        positions: list[TreasuryPosition],
    ) -> NDArray[np.float64]:
        """
        Boolean array (n_positions, n_paths) marking liquidation events.

        For positions with liquidation thresholds, checks if the
        normalised price ever breached the threshold on any day.
        """
        n_assets = len(positions)
        n_paths = price_paths.shape[1]
        liquidated = np.zeros((n_assets, n_paths), dtype=bool)

        for i, pos in enumerate(positions):
            if not pos.has_liquidation_threshold or _d(pos.principal_usd) <= 0:
                continue

            # Normalised liquidation price relative to initial
            liq_ratio = _d(pos.liquidation_price_usd) / _d(pos.principal_usd)
            if liq_ratio <= 0:
                continue

            # Check if min price across path breaches liquidation level
            min_prices = np.min(price_paths[i, :, :], axis=1)  # (n_paths,)
            liquidated[i] = min_prices <= liq_ratio

        return liquidated

    # ── Tail Risk Computation ──────────────────────────────────────

    def _compute_tail_risk(
        self,
        price_paths: NDArray[np.float64],
        liquidations: NDArray[np.float64],
        positions: list[TreasuryPosition],
    ) -> tuple[TailRiskProfile, dict[str, TailRiskProfile]]:
        """Compute portfolio-level and per-position tail risk profiles."""
        n_assets = len(positions)

        position_risks: dict[str, TailRiskProfile] = {}

        # Per-position risk
        for i, pos in enumerate(positions):
            principal = _d(pos.principal_usd)
            if principal <= 0:
                continue

            # Final returns (normalised price at horizon - 1.0)
            final_prices = price_paths[i, :, -1]
            returns_pct = final_prices - 1.0
            returns_usd = returns_pct * principal

            # Max drawdown per path
            cummax = np.maximum.accumulate(price_paths[i, :, :], axis=1)
            drawdowns = (cummax - price_paths[i, :, :]) / np.maximum(cummax, 1e-10)
            max_dd_per_path = np.max(drawdowns, axis=1)

            # VaR (losses are negative, so 5th percentile of returns)
            var_5 = float(np.percentile(returns_usd, 5))
            var_25 = float(np.percentile(returns_usd, 25))

            # CVaR: expected loss in the worst 5%
            tail_mask = returns_usd <= np.percentile(returns_usd, 5)
            cvar_5 = float(np.mean(returns_usd[tail_mask])) if np.any(tail_mask) else var_5

            # Liquidation stats
            liq_prob = float(np.mean(liquidations[i]))
            liq_loss = 0.0
            if liq_prob > 0:
                liq_mask = liquidations[i].astype(bool)
                liq_loss = float(np.mean(returns_usd[liq_mask]))

            # Time to first liquidation (for liquidated paths)
            time_to_liq = -1
            if pos.has_liquidation_threshold and _d(pos.principal_usd) > 0:
                liq_ratio = _d(pos.liquidation_price_usd) / _d(pos.principal_usd)
                if liq_ratio > 0:
                    breached = price_paths[i, :, :] <= liq_ratio
                    first_breach = np.argmax(breached, axis=1)
                    # argmax returns 0 for paths that never breach — filter those
                    ever_breached = np.any(breached, axis=1)
                    if np.any(ever_breached):
                        breach_days = first_breach[ever_breached]
                        time_to_liq = int(np.percentile(breach_days, 10))

            position_risks[pos.position_id] = TailRiskProfile(
                var_5pct=_to_dec(var_5),
                var_25pct=_to_dec(var_25),
                cvar_5pct=_to_dec(cvar_5),
                max_drawdown_median=_to_dec(float(np.median(max_dd_per_path))),
                max_drawdown_p95=_to_dec(float(np.percentile(max_dd_per_path, 95))),
                liquidation_probability=_to_dec(liq_prob),
                expected_liquidation_loss=_to_dec(liq_loss),
                time_to_liquidation_p10=time_to_liq,
            )

        # Portfolio-level risk
        portfolio_returns = self._compute_portfolio_returns(price_paths, positions)

        # Portfolio max drawdown
        weights = np.array([_d(p.principal_usd) for p in positions])
        total_value = weights.sum()
        w = weights / total_value if total_value > 0 else np.ones(n_assets) / max(n_assets, 1)

        portfolio_prices = np.tensordot(w, price_paths, axes=([0], [0]))  # (n_paths, horizon+1)
        port_cummax = np.maximum.accumulate(portfolio_prices, axis=1)
        port_drawdowns = (port_cummax - portfolio_prices) / np.maximum(port_cummax, 1e-10)
        port_max_dd = np.max(port_drawdowns, axis=1)

        var_5_port = float(np.percentile(portfolio_returns, 5))
        var_25_port = float(np.percentile(portfolio_returns, 25))
        tail_mask_port = portfolio_returns <= np.percentile(portfolio_returns, 5)
        cvar_5_port = (
            float(np.mean(portfolio_returns[tail_mask_port]))
            if np.any(tail_mask_port) else var_5_port
        )

        # Portfolio liquidation probability: at least one position liquidated
        any_liquidated = np.any(liquidations, axis=0)
        port_liq_prob = float(np.mean(any_liquidated))

        portfolio_risk = TailRiskProfile(
            var_5pct=_to_dec(var_5_port),
            var_25pct=_to_dec(var_25_port),
            cvar_5pct=_to_dec(cvar_5_port),
            max_drawdown_median=_to_dec(float(np.median(port_max_dd))),
            max_drawdown_p95=_to_dec(float(np.percentile(port_max_dd, 95))),
            liquidation_probability=_to_dec(port_liq_prob),
        )

        return portfolio_risk, position_risks

    def _compute_portfolio_returns(
        self,
        price_paths: NDArray[np.float64],
        positions: list[TreasuryPosition],
    ) -> NDArray[np.float64]:
        """Compute weighted portfolio returns in USD."""
        weights = np.array([_d(p.principal_usd) for p in positions])
        total = weights.sum()
        if total <= 0:
            return np.zeros(price_paths.shape[1])

        # Weighted final returns in USD
        final_returns = np.zeros(price_paths.shape[1])
        for i, pos in enumerate(positions):
            principal = _d(pos.principal_usd)
            final_returns += principal * (price_paths[i, :, -1] - 1.0)

        return final_returns

    # ── Critical Exposure Identification ───────────────────────────

    def _identify_critical_exposures(
        self,
        position_risks: dict[str, TailRiskProfile],
        positions: list[TreasuryPosition],
        portfolio_risk: TailRiskProfile,
    ) -> list[CriticalExposure]:
        """Rank positions by marginal contribution to portfolio VaR."""
        portfolio_var = abs(_d(portfolio_risk.var_5pct))
        if portfolio_var <= 0:
            return []

        exposures: list[CriticalExposure] = []
        for pos in positions:
            risk = position_risks.get(pos.position_id)
            if risk is None:
                continue

            pos_var = abs(_d(risk.var_5pct))
            contribution = pos_var / portfolio_var if portfolio_var > 0 else 0.0

            exposures.append(CriticalExposure(
                position_id=pos.position_id,
                symbol=pos.symbol,
                asset_class=pos.asset_class,
                exposure_usd=pos.principal_usd,
                contribution_to_portfolio_var=_to_dec(contribution),
                contagion_amplifier=Decimal("1"),
                risk_rank=0,
                rationale=f"{pos.symbol} ({pos.protocol or 'spot'}): "
                          f"VaR₅={_to_dec(pos_var)}, "
                          f"P(liq)={risk.liquidation_probability}",
            ))

        # Sort by contribution (highest first) and assign ranks
        exposures.sort(key=lambda e: _d(e.contribution_to_portfolio_var), reverse=True)
        for rank, exp in enumerate(exposures, start=1):
            exp.risk_rank = rank

        return exposures

    # ── Hedging Proposal Generation ────────────────────────────────

    def _generate_hedging_proposals(
        self,
        critical_exposures: list[CriticalExposure],
        position_risks: dict[str, TailRiskProfile],
        positions: list[TreasuryPosition],
    ) -> list[HedgingProposal]:
        """
        Generate sized, actionable hedging proposals.

        Rule-based: prioritises liquidation risk, then concentration risk,
        then general tail risk reduction. Phase 1: proposals only, no
        auto-execution (no Axon executor for hedges yet).
        """
        proposals: list[HedgingProposal] = []
        pos_map = {p.position_id: p for p in positions}
        liq_threshold = self._config.threat_model_liquidation_alert_threshold
        var_threshold = self._config.threat_model_max_single_asset_var_pct

        for exposure in critical_exposures:
            risk = position_risks.get(exposure.position_id)
            pos = pos_map.get(exposure.position_id)
            if risk is None or pos is None:
                continue

            liq_prob = _d(risk.liquidation_probability)
            var_contrib = _d(exposure.contribution_to_portfolio_var)

            # Rule 1: High liquidation probability — add collateral
            if liq_prob > liq_threshold and pos.has_liquidation_threshold:
                # Size: enough to reduce collateral ratio to safe level
                principal = _d(pos.principal_usd)
                min_ratio = _d(pos.min_collateral_ratio)
                current_ratio = _d(pos.collateral_ratio)
                target_ratio = min_ratio * 1.5  # 50% buffer above liquidation
                if current_ratio > 0 and current_ratio < target_ratio:
                    additional = principal * (target_ratio / current_ratio - 1.0)
                    proposals.append(HedgingProposal(
                        target_position_id=pos.position_id,
                        target_symbol=pos.symbol,
                        hedge_action="add_collateral",
                        hedge_instrument="USDC",
                        hedge_size_usd=_to_dec(additional),
                        hedge_size_pct=_to_dec(additional / principal if principal > 0 else 0),
                        var_reduction_pct=_to_dec(0.0),
                        liquidation_prob_reduction=_to_dec(liq_prob * 0.7),
                        cost_estimate_usd=_to_dec(additional),
                        priority=0,
                        confidence=Decimal("0.85"),
                        description=(
                            f"Add ${additional:.2f} collateral to {pos.symbol} "
                            f"({pos.protocol}) to reduce P(liquidation) from "
                            f"{liq_prob:.1%} by ~70%."
                        ),
                    ))

            # Rule 2: Single asset dominates portfolio VaR — diversify
            if var_contrib > var_threshold:
                principal = _d(pos.principal_usd)
                # Propose reducing position by enough to bring contribution below threshold
                target_reduction = 1.0 - (var_threshold / var_contrib)
                reduce_amount = principal * target_reduction

                is_stable = pos.asset_class == AssetClass.STABLECOIN
                action = "diversify_protocol" if is_stable else "sell_to_stable"
                proposals.append(HedgingProposal(
                    target_position_id=pos.position_id,
                    target_symbol=pos.symbol,
                    hedge_action=action,
                    hedge_instrument="USDC",
                    hedge_size_usd=_to_dec(reduce_amount),
                    hedge_size_pct=_to_dec(target_reduction),
                    var_reduction_pct=_to_dec(target_reduction * var_contrib),
                    liquidation_prob_reduction=_to_dec(0.0),
                    cost_estimate_usd=_to_dec(reduce_amount * 0.003),  # ~30bps slippage
                    priority=1,
                    confidence=Decimal("0.75"),
                    description=(
                        f"Reduce {pos.symbol} exposure by {target_reduction:.0%} "
                        f"(${reduce_amount:.2f}) to bring VaR contribution below "
                        f"{var_threshold:.0%}."
                    ),
                ))

            # Rule 3: Stablecoin depeg risk — diversify across stables
            depeg_loss = _d(risk.var_5pct) < -_d(pos.principal_usd) * 0.02
            if pos.asset_class == AssetClass.STABLECOIN and depeg_loss:
                principal = _d(pos.principal_usd)
                diversify_amount = principal * 0.3  # Spread 30% to other stables
                proposals.append(HedgingProposal(
                    target_position_id=pos.position_id,
                    target_symbol=pos.symbol,
                    hedge_action="diversify_protocol",
                    hedge_instrument="DAI",
                    hedge_size_usd=_to_dec(diversify_amount),
                    hedge_size_pct=_to_dec(0.3),
                    var_reduction_pct=_to_dec(0.15),
                    liquidation_prob_reduction=_to_dec(0.0),
                    cost_estimate_usd=_to_dec(diversify_amount * 0.001),
                    priority=2,
                    confidence=Decimal("0.7"),
                    description=(
                        f"Diversify ${diversify_amount:.2f} of {pos.symbol} "
                        f"across other stablecoins to reduce depeg tail risk."
                    ),
                ))

        # Sort by priority (lower = more urgent)
        proposals.sort(key=lambda p: p.priority)
        return proposals
