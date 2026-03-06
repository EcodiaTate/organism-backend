"""
EcodiaOS — Oikos Asset Factory (Phase 16d: Entrepreneurship)

The Asset Factory is the organism's capacity to transition from freelancer
to business owner. It designs, evaluates, deploys, and operates autonomous
revenue-generating services (assets) that earn via a smart contract tollbooth.

Responsibilities:
  1. IDEATE    — receive market gap signals from Evo, generate candidate assets
  2. EVALUATE  — score candidates on dev cost, projected revenue, break-even
  3. APPROVE   — select viable candidates (ROI > threshold, break-even < 90 days)
  4. BUILD     — command Simula to generate the service code
  5. DEPLOY    — deploy to decentralised compute (Akash) with tollbooth contract
  6. MONITOR   — track revenue, costs, break-even, and revenue trends
  7. TERMINATE — shut down assets that fail break-even or show 30-day decline

The factory bridges Oikos (economics), Evo (market signals), and Simula
(code generation). It does NOT bypass Equor — all asset deployments require
constitutional review through the normal Axon pipeline.

Thread-safety: NOT thread-safe. Designed for single-threaded asyncio event loop.
"""

from __future__ import annotations

import contextlib
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.oikos.models import (
    AssetCandidate,
    AssetStatus,
    OwnedAsset,
    TollboothConfig,
)

if TYPE_CHECKING:
    from systems.oikos.models import EconomicState
    from systems.oikos.service import OikosService

logger = structlog.get_logger("oikos.asset_factory")


# ─── Policy Constants ────────────────────────────────────────────


class AssetPolicy:
    """
    Hard constraints for asset creation. The organism must meet ALL
    thresholds before an asset candidate is approved for build.

    These are conservative defaults — Evo can propose adjustments
    via the normal ADJUST_BUDGET evolution path.
    """

    # Minimum ROI (projected_net_revenue / dev_cost) to approve a candidate
    MIN_ROI_THRESHOLD: Decimal = Decimal("2.0")

    # Maximum break-even timeline (days). Spec: 90 days.
    MAX_BREAK_EVEN_DAYS: int = 90

    # Minimum market gap confidence from Evo (0.0-1.0)
    MIN_MARKET_GAP_CONFIDENCE: Decimal = Decimal("0.3")

    # Maximum concurrent live assets
    MAX_CONCURRENT_ASSETS: int = 5

    # Minimum liquid balance required to fund asset development
    # (organism must be able to survive the dev cost expenditure)
    MIN_LIQUID_AFTER_DEV: Decimal = Decimal("20.00")

    # Revenue decline threshold: terminate after N consecutive declining days
    REVENUE_DECLINE_TERMINATE_DAYS: int = 30

    # Break-even deadline: terminate if not reached within this many days
    BREAK_EVEN_DEADLINE_DAYS: int = 90

    # Minimum development cost — below this, the candidate is suspicious
    MIN_DEV_COST_USD: Decimal = Decimal("1.00")

    # Maximum development cost — cap per-asset investment
    MAX_DEV_COST_USD: Decimal = Decimal("500.00")


# ─── Asset Factory ────────────────────────────────────────────────


class AssetFactory:
    """
    The organism's entrepreneurship engine.

    Evaluates candidate autonomous services, approves viable ones,
    orchestrates code generation via Simula, and manages the lifecycle
    of deployed assets including break-even tracking and termination.
    """

    def __init__(self, oikos: OikosService) -> None:
        self._oikos = oikos
        self._candidates: dict[str, AssetCandidate] = {}
        self._logger = logger.bind(component="asset_factory")

    # ─── 1. Ideation ──────────────────────────────────────────────

    def ideate(
        self,
        name: str,
        description: str,
        asset_type: str,
        estimated_dev_cost_usd: Decimal,
        projected_monthly_revenue_usd: Decimal,
        projected_monthly_cost_usd: Decimal = Decimal("0"),
        market_gap_confidence: Decimal = Decimal("0.5"),
        competitive_differentiation: Decimal = Decimal("0.5"),
        evo_hypothesis_id: str = "",
    ) -> AssetCandidate:
        """
        Create a new asset candidate from a market gap signal.

        Typically called when Evo detects an underserved niche. The candidate
        is scored but NOT automatically approved — call evaluate() next.
        """
        # Compute break-even days
        net_monthly = projected_monthly_revenue_usd - projected_monthly_cost_usd
        if net_monthly > Decimal("0"):
            break_even_days = int(
                (estimated_dev_cost_usd / net_monthly * Decimal("30")).to_integral_value()
            )
        else:
            break_even_days = 9999  # Will be rejected by evaluate()

        # Compute ROI score
        if estimated_dev_cost_usd > Decimal("0"):
            # Annualised net / dev cost
            annual_net = net_monthly * Decimal("12")
            roi_score = (annual_net / estimated_dev_cost_usd).quantize(Decimal("0.01"))
        else:
            roi_score = Decimal("0")

        candidate = AssetCandidate(
            name=name,
            description=description,
            asset_type=asset_type,
            estimated_dev_cost_usd=estimated_dev_cost_usd,
            projected_monthly_revenue_usd=projected_monthly_revenue_usd,
            projected_monthly_cost_usd=projected_monthly_cost_usd,
            break_even_days=break_even_days,
            roi_score=roi_score,
            competitive_differentiation=competitive_differentiation,
            market_gap_confidence=market_gap_confidence,
            evo_hypothesis_id=evo_hypothesis_id,
        )

        self._candidates[candidate.candidate_id] = candidate
        self._logger.info(
            "asset_candidate_created",
            candidate_id=candidate.candidate_id,
            name=name,
            roi_score=str(roi_score),
            break_even_days=break_even_days,
        )
        return candidate

    # ─── 2. Evaluation ────────────────────────────────────────────

    def evaluate(self, candidate_id: str) -> AssetCandidate:
        """
        Evaluate a candidate against the AssetPolicy.

        Returns the candidate with approved=True/False and rejection_reason set.
        This is a pure economic decision — constitutional review happens later
        in the Axon pipeline when the DeployAssetExecutor fires.
        """
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"No candidate with id {candidate_id!r}")

        rejections: list[str] = []
        state = self._oikos.snapshot()

        # Check ROI threshold
        if candidate.roi_score < AssetPolicy.MIN_ROI_THRESHOLD:
            rejections.append(
                f"ROI {candidate.roi_score} < minimum {AssetPolicy.MIN_ROI_THRESHOLD}"
            )

        # Check break-even deadline
        if candidate.break_even_days > AssetPolicy.MAX_BREAK_EVEN_DAYS:
            rejections.append(
                f"Break-even {candidate.break_even_days}d > maximum {AssetPolicy.MAX_BREAK_EVEN_DAYS}d"
            )

        # Check market gap confidence
        if candidate.market_gap_confidence < AssetPolicy.MIN_MARKET_GAP_CONFIDENCE:
            rejections.append(
                f"Market gap confidence {candidate.market_gap_confidence} "
                f"< minimum {AssetPolicy.MIN_MARKET_GAP_CONFIDENCE}"
            )

        # Check dev cost bounds
        if candidate.estimated_dev_cost_usd < AssetPolicy.MIN_DEV_COST_USD:
            rejections.append(
                f"Dev cost ${candidate.estimated_dev_cost_usd} < minimum ${AssetPolicy.MIN_DEV_COST_USD}"
            )
        if candidate.estimated_dev_cost_usd > AssetPolicy.MAX_DEV_COST_USD:
            rejections.append(
                f"Dev cost ${candidate.estimated_dev_cost_usd} > maximum ${AssetPolicy.MAX_DEV_COST_USD}"
            )

        # Check concurrent asset limit
        live_assets = [
            a for a in state.owned_assets
            if a.status in (AssetStatus.LIVE, AssetStatus.BUILDING, AssetStatus.DEPLOYING)
        ]
        if len(live_assets) >= AssetPolicy.MAX_CONCURRENT_ASSETS:
            rejections.append(
                f"Already at max concurrent assets ({AssetPolicy.MAX_CONCURRENT_ASSETS})"
            )

        # Check organism can afford the dev cost
        remaining_liquid = state.liquid_balance - candidate.estimated_dev_cost_usd
        if remaining_liquid < AssetPolicy.MIN_LIQUID_AFTER_DEV:
            rejections.append(
                f"Liquid balance after dev (${remaining_liquid}) < "
                f"minimum ${AssetPolicy.MIN_LIQUID_AFTER_DEV}"
            )

        # Final verdict
        if rejections:
            candidate.approved = False
            candidate.rejection_reason = "; ".join(rejections)
            self._logger.info(
                "asset_candidate_rejected",
                candidate_id=candidate_id,
                reasons=rejections,
            )
        else:
            candidate.approved = True
            self._logger.info(
                "asset_candidate_approved",
                candidate_id=candidate_id,
                roi=str(candidate.roi_score),
                break_even_days=candidate.break_even_days,
            )

        return candidate

    # ─── 3. Approval → OwnedAsset ────────────────────────────────

    def promote_to_asset(self, candidate_id: str) -> OwnedAsset:
        """
        Promote an approved candidate to an OwnedAsset in BUILDING state.

        The asset is added to OikosService's economic state. It is NOT yet
        deployed — the DeployAssetExecutor handles code generation and deployment.
        """
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"No candidate with id {candidate_id!r}")
        if not candidate.approved:
            raise ValueError(
                f"Candidate {candidate_id!r} has not been approved. "
                f"Rejection: {candidate.rejection_reason or 'not yet evaluated'}"
            )

        asset = OwnedAsset(
            name=candidate.name,
            description=candidate.description,
            asset_type=candidate.asset_type,
            status=AssetStatus.BUILDING,
            development_cost_usd=candidate.estimated_dev_cost_usd,
            projected_break_even_days=candidate.break_even_days,
            total_cost_usd=candidate.estimated_dev_cost_usd,
            candidate_id=candidate.candidate_id,
            evo_hypothesis_id=candidate.evo_hypothesis_id,
        )

        # Register in economic state and debit liquid_balance for dev cost
        state = self._oikos.snapshot()
        if state.liquid_balance < candidate.estimated_dev_cost_usd:
            raise ValueError(
                f"Insufficient liquid balance (${state.liquid_balance}) "
                f"to fund dev cost (${candidate.estimated_dev_cost_usd})"
            )
        state.owned_assets.append(asset)
        state.liquid_balance -= candidate.estimated_dev_cost_usd
        self._recompute_asset_value(state)

        # Recalculate derived metrics after balance change
        self._oikos._recalculate_derived_metrics()

        self._logger.info(
            "asset_promoted",
            asset_id=asset.asset_id,
            name=asset.name,
            status=asset.status.value,
            dev_cost=str(asset.development_cost_usd),
            liquid_balance_after=str(state.liquid_balance),
        )
        return asset

    # ─── 4. Lifecycle Transitions ─────────────────────────────────

    def mark_deployed(
        self,
        asset_id: str,
        api_endpoint: str,
        deployment_id: str,
        tollbooth_address: str,
        price_per_call_usd: Decimal = Decimal("0.01"),
        compute_provider: str = "akash",
        source_repo: str = "",
    ) -> OwnedAsset:
        """
        Transition an asset from BUILDING → LIVE after successful deployment.

        Called by DeployAssetExecutor once Simula has generated the code and
        it has been deployed to compute with a tollbooth contract.
        """
        asset = self._find_asset(asset_id)

        asset.status = AssetStatus.LIVE
        asset.deployed_at = utc_now()
        asset.api_endpoint = api_endpoint
        asset.deployment_id = deployment_id
        asset.compute_provider = compute_provider
        asset.source_repo = source_repo

        # Configure tollbooth
        owner_address = ""
        if self._oikos._wallet is not None:
            with contextlib.suppress(Exception):
                owner_address = self._oikos._wallet.address

        asset.tollbooth = TollboothConfig(
            contract_address=tollbooth_address,
            price_per_call_usd=price_per_call_usd,
            owner_address=owner_address,
            asset_endpoint=api_endpoint,
        )

        state = self._oikos.snapshot()
        self._recompute_asset_value(state)

        self._logger.info(
            "asset_deployed",
            asset_id=asset_id,
            endpoint=api_endpoint,
            tollbooth=tollbooth_address,
            price_per_call=str(price_per_call_usd),
        )
        return asset

    def record_revenue(self, asset_id: str, amount_usd: Decimal) -> None:
        """
        Record revenue earned by a live asset. Called when the tollbooth
        sweeps accumulated payments or on a periodic revenue check.
        """
        asset = self._find_asset(asset_id)

        asset.total_revenue_usd += amount_usd
        asset.monthly_revenue_usd = self._estimate_monthly_from_total(asset)

        # Update estimated value (simple DCF: 12 months of net monthly income)
        net_monthly = asset.monthly_revenue_usd - asset.monthly_cost_usd
        asset.estimated_value_usd = max(Decimal("0"), net_monthly * Decimal("12"))

        # Check break-even
        if not asset.break_even_reached and asset.total_revenue_usd >= asset.total_cost_usd:
            asset.break_even_reached = True
            asset.break_even_at = utc_now()
            self._logger.info("asset_break_even_reached", asset_id=asset_id)

        state = self._oikos.snapshot()
        self._recompute_asset_value(state)

    def record_cost(self, asset_id: str, amount_usd: Decimal) -> None:
        """Record hosting/compute cost for a live asset."""
        asset = self._find_asset(asset_id)
        asset.total_cost_usd += amount_usd
        asset.monthly_cost_usd = self._estimate_monthly_cost_from_total(asset)

        state = self._oikos.snapshot()
        self._recompute_asset_value(state)

    def update_revenue_trend(self, asset_id: str, daily_revenue: Decimal) -> None:
        """
        Update the revenue trend for a live asset. Called daily.

        Tracks consecutive declining days to trigger termination per spec:
        assets showing declining revenue for 30 days are terminated.
        """
        asset = self._find_asset(asset_id)

        if daily_revenue < asset.revenue_trend_30d:
            asset.consecutive_declining_days += 1
        else:
            asset.consecutive_declining_days = 0

        asset.revenue_trend_30d = daily_revenue

        if asset.consecutive_declining_days >= AssetPolicy.REVENUE_DECLINE_TERMINATE_DAYS:
            asset.status = AssetStatus.DECLINING
            self._logger.warning(
                "asset_declining",
                asset_id=asset_id,
                consecutive_declining_days=asset.consecutive_declining_days,
            )

    # ─── 5. Termination ───────────────────────────────────────────

    def check_terminations(self) -> list[OwnedAsset]:
        """
        Scan all live assets and flag those that should be terminated.

        Per spec Section VI:
          - Assets that fail break-even within 90 days → terminate
          - Assets showing declining revenue for 30 days → terminate

        Returns the list of assets that were terminated in this pass.
        """
        state = self._oikos.snapshot()
        terminated: list[OwnedAsset] = []

        for asset in state.owned_assets:
            if asset.status not in (AssetStatus.LIVE, AssetStatus.DECLINING):
                continue

            if asset.should_terminate:
                reason = self._termination_reason(asset)
                self._terminate_asset(asset, reason)
                terminated.append(asset)

        if terminated:
            self._recompute_asset_value(state)
            self._logger.info(
                "assets_terminated",
                count=len(terminated),
                asset_ids=[a.asset_id for a in terminated],
            )

        return terminated

    def terminate_asset(self, asset_id: str, reason: str = "manual") -> OwnedAsset:
        """Manually terminate a specific asset."""
        asset = self._find_asset(asset_id)
        self._terminate_asset(asset, reason)

        state = self._oikos.snapshot()
        self._recompute_asset_value(state)
        return asset

    # ─── 6. Queries ───────────────────────────────────────────────

    def get_candidates(self) -> list[AssetCandidate]:
        """Return all candidates (approved and rejected)."""
        return list(self._candidates.values())

    def get_live_assets(self) -> list[OwnedAsset]:
        """Return all assets in LIVE or DECLINING status."""
        state = self._oikos.snapshot()
        return [
            a for a in state.owned_assets
            if a.status in (AssetStatus.LIVE, AssetStatus.DECLINING)
        ]

    def get_building_assets(self) -> list[OwnedAsset]:
        """Return all assets currently being built or deployed."""
        state = self._oikos.snapshot()
        return [
            a for a in state.owned_assets
            if a.status in (AssetStatus.BUILDING, AssetStatus.DEPLOYING)
        ]

    @property
    def stats(self) -> dict[str, object]:
        """Summary stats for observability."""
        state = self._oikos.snapshot()
        live = [a for a in state.owned_assets if a.status == AssetStatus.LIVE]
        building = [a for a in state.owned_assets if a.status == AssetStatus.BUILDING]
        total_monthly_net = sum(a.net_monthly_income for a in live)
        return {
            "candidates_total": len(self._candidates),
            "candidates_approved": sum(1 for c in self._candidates.values() if c.approved),
            "assets_live": len(live),
            "assets_building": len(building),
            "total_monthly_net_income": str(total_monthly_net),
            "total_asset_value": str(state.total_asset_value),
        }

    # ─── Private Helpers ──────────────────────────────────────────

    def _find_asset(self, asset_id: str) -> OwnedAsset:
        state = self._oikos.snapshot()
        for asset in state.owned_assets:
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(f"No asset with id {asset_id!r}")

    def _terminate_asset(self, asset: OwnedAsset, reason: str) -> None:
        asset.status = AssetStatus.TERMINATED
        self._logger.warning(
            "asset_terminated",
            asset_id=asset.asset_id,
            name=asset.name,
            reason=reason,
            total_revenue=str(asset.total_revenue_usd),
            total_cost=str(asset.total_cost_usd),
            days_live=asset.days_since_deployment,
        )

    def _termination_reason(self, asset: OwnedAsset) -> str:
        reasons: list[str] = []
        if not asset.break_even_reached and asset.days_since_deployment > 90:
            reasons.append(
                f"Failed break-even after {asset.days_since_deployment} days "
                f"(deadline: 90 days)"
            )
        if asset.consecutive_declining_days >= 30:
            reasons.append(
                f"Revenue declining for {asset.consecutive_declining_days} consecutive days"
            )
        return "; ".join(reasons) or "unknown"

    def _recompute_asset_value(self, state: EconomicState) -> None:
        """Recompute total_asset_value from all non-terminated assets."""
        total = Decimal("0")
        for asset in state.owned_assets:
            if asset.status != AssetStatus.TERMINATED:
                total += asset.estimated_value_usd
        state.total_asset_value = total

    def _estimate_monthly_from_total(self, asset: OwnedAsset) -> Decimal:
        """Estimate monthly revenue based on total revenue and days live."""
        days = asset.days_since_deployment
        if days <= 0:
            return Decimal("0")
        daily_avg = asset.total_revenue_usd / Decimal(str(days))
        return (daily_avg * Decimal("30")).quantize(Decimal("0.01"))

    def _estimate_monthly_cost_from_total(self, asset: OwnedAsset) -> Decimal:
        """Estimate monthly cost from total cost minus dev cost and days live."""
        days = asset.days_since_deployment
        if days <= 0:
            return Decimal("0")
        hosting_total = asset.total_cost_usd - asset.development_cost_usd
        if hosting_total <= Decimal("0"):
            return Decimal("0")
        daily_avg = hosting_total / Decimal(str(days))
        return (daily_avg * Decimal("30")).quantize(Decimal("0.01"))
