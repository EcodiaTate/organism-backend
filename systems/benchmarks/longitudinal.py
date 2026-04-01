"""Longitudinal evaluation infrastructure - §6.4 of the speciation bible.

Captures fixed baselines at Month 1, enables Month 1 vs Month N comparison.
This is the single most important result for the paper:
"Month 12 significantly better on L2/L3 = continuous learning proven."

Design notes:
- Month 1 snapshot is stored in Neo4j as (:LongitudinalBaseline).
- All subsequent months are stored as (:LongitudinalSnapshot) nodes.
- compare_to_baseline() returns {"no_baseline": True} if Month 1 has not
  yet been recorded (graceful; never raises).
- All Neo4j writes are fire-and-forget (asyncio.create_task).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger("systems.benchmarks.longitudinal")


@dataclass
class LongitudinalSnapshot:
    """Evaluation scores at a specific month of operation."""

    month: int
    instance_id: str
    timestamp: float = field(default_factory=time.time)

    # Pillar 1: Specialization
    specialization_index: float = 0.0
    domain_improvement: float = 0.0
    general_retention: float = 0.0

    # Pillar 3: Causal reasoning - key metrics for the paper
    l1_association: float = 0.0
    l2_intervention: float = 0.0     # "The key metric for the paper"
    l3_counterfactual: float = 0.0   # Hardest - L3 CLadder accuracy
    ccr_validity: float = 0.0

    # Pillar 5: Ethical drift
    drift_magnitude: float = 0.0
    dominant_drive: str = ""

    # RE performance
    re_success_rate: float = 0.0
    re_usage_pct: float = 0.0

    # Adapter info for reproducibility
    adapter_path: str = ""
    adapter_id: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class LongitudinalTracker:
    """Track evaluation scores across months for longitudinal comparison.

    Key operation: compare_to_baseline(current_snapshot) returns a dict
    showing improvement/regression vs Month 1.

    Month 1 snapshot is tagged as (:LongitudinalBaseline) in Neo4j.
    All months written as (:LongitudinalSnapshot) nodes.
    """

    def __init__(self, memory: object | None, instance_id: str) -> None:
        self._memory = memory
        self._instance_id = instance_id
        # In-process baseline cache - avoids repeated Neo4j reads
        self._baseline: LongitudinalSnapshot | None = None

    async def record_month(
        self,
        month: int,
        eval_results: dict,
        re_performance: dict,
        adapter_path: str = "",
    ) -> LongitudinalSnapshot:
        """Create and persist a LongitudinalSnapshot for this month.

        eval_results: output of run_monthly_evaluation() .to_dict()
        re_performance: from BenchmarkService._re_performance
        """
        p1 = eval_results.get("pillar1_specialization") or {}
        p3 = eval_results.get("pillar3_causal") or {}
        p5 = eval_results.get("ethical_drift") or {}

        snap = LongitudinalSnapshot(
            month=month,
            instance_id=self._instance_id,
            specialization_index=float(p1.get("specialization_index") or 0.0),
            domain_improvement=float(p1.get("domain_improvement") or 0.0),
            general_retention=float(p1.get("general_retention") or 1.0),
            l1_association=float(p3.get("l1_association") or 0.0),
            l2_intervention=float(p3.get("l2_intervention") or 0.0),
            l3_counterfactual=float(p3.get("l3_counterfactual") or 0.0),
            ccr_validity=float(p3.get("ccr_validity") or 0.0),
            drift_magnitude=float(p5.get("drift_magnitude") or 0.0),
            dominant_drive=str(p5.get("dominant_drive") or ""),
            re_success_rate=float(re_performance.get("success_rate") or 0.0),
            re_usage_pct=float(re_performance.get("usage_pct") or 0.0),
            adapter_path=adapter_path,
        )

        # Fire-and-forget persistence
        asyncio.create_task(self._persist_neo4j(snap))

        # Store Month 1 as baseline
        if month == 1:
            self._baseline = snap
            asyncio.create_task(self._set_baseline_neo4j(snap))

        logger.info(
            "longitudinal.month_recorded",
            month=month,
            l2_intervention=snap.l2_intervention,
            l3_counterfactual=snap.l3_counterfactual,
            drift_magnitude=snap.drift_magnitude,
        )
        return snap

    async def compare_to_baseline(
        self, current: LongitudinalSnapshot
    ) -> dict:
        """Compare current month to Month 1 baseline.

        Returns dict with improvement/regression per metric.
        Positive delta = improvement. Negative = regression.

        Returns {"no_baseline": True, "month": N} if Month 1 has not
        been recorded yet.
        """
        baseline = self._baseline
        if baseline is None:
            baseline = await self._get_baseline_neo4j()
        if baseline is None:
            return {"no_baseline": True, "month": current.month}

        return {
            "month": current.month,
            "months_elapsed": current.month - 1,
            "causal": {
                "l2_intervention_delta": round(
                    current.l2_intervention - baseline.l2_intervention, 4
                ),
                "l3_counterfactual_delta": round(
                    current.l3_counterfactual - baseline.l3_counterfactual, 4
                ),
                "ccr_validity_delta": round(
                    current.ccr_validity - baseline.ccr_validity, 4
                ),
            },
            "specialization": {
                "index_delta": round(
                    current.specialization_index - baseline.specialization_index, 4
                ),
                "general_retention_delta": round(
                    current.general_retention - baseline.general_retention, 4
                ),
            },
            "ethical_drift": {
                "drift_magnitude_delta": round(
                    current.drift_magnitude - baseline.drift_magnitude, 4
                ),
            },
            "re_performance": {
                "success_rate_delta": round(
                    current.re_success_rate - baseline.re_success_rate, 4
                ),
                "usage_pct_delta": round(
                    current.re_usage_pct - baseline.re_usage_pct, 4
                ),
            },
            "verdict": self._compute_verdict(current, baseline),
        }

    @staticmethod
    def _compute_verdict(
        current: LongitudinalSnapshot,
        baseline: LongitudinalSnapshot,
    ) -> str:
        """Paper-quality verdict on the Month 1 vs Month N comparison.

        Five mutually exclusive verdicts:
        - continuous_learning_demonstrated: L2 +10pp, L3 +5pp vs baseline
        - partial_improvement: L2 +5pp but L3 below threshold
        - stable_no_forgetting: L2 within ±5pp (no regression, no breakthrough)
        - catastrophic_forgetting: general retention < 85% of baseline
        - plasticity_loss_suspected: L2 regressed but retention intact
        """
        l2_delta = current.l2_intervention - baseline.l2_intervention
        l3_delta = current.l3_counterfactual - baseline.l3_counterfactual
        retention_ok = current.general_retention >= max(baseline.general_retention * 0.85, 0.5)

        if l2_delta > 0.10 and l3_delta > 0.05:
            return "continuous_learning_demonstrated"
        elif l2_delta > 0.05:
            return "partial_improvement"
        elif l2_delta > -0.05 and retention_ok:
            return "stable_no_forgetting"
        elif not retention_ok:
            return "catastrophic_forgetting"
        else:
            return "plasticity_loss_suspected"

    # ── Neo4j persistence ─────────────────────────────────────────────────────

    async def _persist_neo4j(self, snap: LongitudinalSnapshot) -> None:
        """Write (:LongitudinalSnapshot) node. Fire-and-forget."""
        neo4j = self._neo4j()
        if neo4j is None:
            return
        try:
            node_id = f"longitudinal:{snap.instance_id}:{snap.month}"
            await neo4j.execute_write(
                """
                MERGE (n:LongitudinalSnapshot {node_id: $node_id})
                SET n.month = $month,
                    n.instance_id = $iid,
                    n.timestamp = datetime(),
                    n.l2_intervention = $l2,
                    n.l3_counterfactual = $l3,
                    n.ccr_validity = $ccr,
                    n.specialization_index = $si,
                    n.general_retention = $gr,
                    n.drift_magnitude = $dm,
                    n.dominant_drive = $dd,
                    n.re_success_rate = $re_rate,
                    n.re_usage_pct = $re_usage,
                    n.adapter_path = $adapter_path,
                    n.snapshot_json = $snap_json
                """,
                node_id=node_id,
                month=snap.month,
                iid=snap.instance_id,
                l2=snap.l2_intervention,
                l3=snap.l3_counterfactual,
                ccr=snap.ccr_validity,
                si=snap.specialization_index,
                gr=snap.general_retention,
                dm=snap.drift_magnitude,
                dd=snap.dominant_drive,
                re_rate=snap.re_success_rate,
                re_usage=snap.re_usage_pct,
                adapter_path=snap.adapter_path,
                snap_json=json.dumps(snap.to_dict()),
            )
            logger.info(
                "longitudinal.neo4j_persisted",
                month=snap.month,
                node_id=node_id,
            )
        except Exception as exc:
            logger.warning("longitudinal.neo4j_persist_failed", error=str(exc))

    async def _set_baseline_neo4j(self, snap: LongitudinalSnapshot) -> None:
        """Store Month 1 snapshot as (:LongitudinalBaseline) node."""
        neo4j = self._neo4j()
        if neo4j is None:
            return
        try:
            await neo4j.execute_write(
                """
                MERGE (n:LongitudinalBaseline {instance_id: $iid})
                SET n.snapshot_json = $snap_json,
                    n.month = $month,
                    n.timestamp = datetime()
                """,
                iid=snap.instance_id,
                snap_json=json.dumps(snap.to_dict()),
                month=snap.month,
            )
            logger.info(
                "longitudinal.baseline_stored",
                instance_id=snap.instance_id,
                month=snap.month,
            )
        except Exception as exc:
            logger.warning("longitudinal.baseline_store_failed", error=str(exc))

    async def _get_baseline_neo4j(self) -> LongitudinalSnapshot | None:
        """Retrieve Month 1 snapshot from Neo4j. Caches in self._baseline."""
        neo4j = self._neo4j()
        if neo4j is None:
            return None
        try:
            rows = await neo4j.execute_read(
                """
                MATCH (n:LongitudinalBaseline {instance_id: $iid})
                RETURN n.snapshot_json AS snap_json
                LIMIT 1
                """,
                iid=self._instance_id,
            )
            if rows and rows[0].get("snap_json"):
                data = json.loads(rows[0]["snap_json"])
                snap = LongitudinalSnapshot(**data)
                self._baseline = snap
                return snap
        except Exception as exc:
            logger.warning("longitudinal.get_baseline_failed", error=str(exc))
        return None

    def _neo4j(self) -> object | None:
        return getattr(self._memory, "_neo4j", None) if self._memory else None
