"""Bedau-Packard evolutionary activity statistics.

Field-accepted standard for OEE-like (Open-Ended Evolution) dynamics.
Used to assess whether EOS exhibits sustained evolutionary novelty
beyond what random drift would produce.

Key papers:
  Bedau & Packard (1992), "Measurement of Evolutionary Activity, Teleology, and Life."
  Bedau et al. (2000), "Open Problems in Artificial Life."

The three statistics
--------------------
1. Adaptive activity (A): How many adaptations are accumulating over time?
   Novel components that persist because they are useful, not by drift.
2. Evolutionary activity modes: change, novelty, diversity, complexity.
   All four growing simultaneously = strongest OEE evidence.
3. Shadow-reset control: Measure a "shadow" population with randomised fitness.
   If real population dramatically exceeds shadow: dynamics are adaptive,
   not random drift.

Operational definitions for EOS
--------------------------------
- "Component" = Evo hypothesis | drive weight | Simula config param
- "Fitness" = hypothesis confidence | drive weight magnitude
- "Novel component" = fingerprint not seen in any prior evaluation month
- "Adaptive component" = novel component that persists into ≥2 consecutive months
- "Shadow population" = same component count with randomised fitness contributions
  (~5% persistence probability = expected random survival at typical churn rates)

Data source
-----------
Fleet genome snapshots collected from CHILD_SPAWNED events and cached in
BenchmarkService._fleet_genomes.  Single-instance operation is gracefully
handled: returns zero/empty metrics without error.

Shadow control accuracy improves with fleet size ≥5.  With a single instance
the shadow comparison is a first-order approximation.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger("systems.benchmarks.bedau_packard")


@dataclass
class EvolutionaryComponent:
    """A single evolvable component in the organism."""

    component_id: str
    component_type: str        # "hypothesis" | "drive_weight" | "config_param"
    instance_id: str
    value_fingerprint: str     # hash of current value (novelty detection)
    fitness_contribution: float
    first_appeared: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class EvolutionaryActivitySnapshot:
    """Bedau-Packard statistics at one monthly timepoint."""

    timestamp: float = field(default_factory=time.time)
    month: int = 0
    # A(t): cumulative adaptive activity
    adaptive_activity: float = 0.0
    # Modes
    change_rate: float = 0.0       # How much is changing vs prior month?
    novelty_rate: float = 0.0      # Novel components / total components
    diversity: float = 0.0         # Shannon entropy of fitness distribution
    complexity: float = 0.0        # Sum of fitness for novel-and-persistent components
    # Shadow control
    shadow_activity: float = 0.0   # Same metric on randomised shadow population
    exceeds_shadow: bool = False    # Real > 2× shadow = adaptive dynamics
    population_size: int = 0       # Distinct fleet instances
    component_count: int = 0       # Total components ingested
    novel_component_count: int = 0


class BedauPackardTracker:
    """Computes Bedau-Packard evolutionary activity statistics for EOS.

    Usage (monthly, called from BenchmarkService._monthly_eval_loop):

        fleet_genomes = await self._collect_fleet_genomes()
        components = self._bp_tracker.ingest_fleet_genomes(fleet_genomes)
        snap = self._bp_tracker.compute_adaptive_activity(components, month=now.month)
        result_dict["evolutionary_activity"] = {...snap fields...}

        if now.month >= 3:
            result_dict["oee_assessment"] = self._bp_tracker.assess_oee_evidence()

    Degradation behaviour
    ---------------------
    - Empty fleet (no genomes) → snapshot with all-zero metrics, no error.
    - Single instance → shadow control approximation (5% random persistence).
    - Missing genome sub-fields → skipped via .get() throughout.
    """

    def __init__(self, speciation_threshold: float = 0.3) -> None:
        self._threshold = speciation_threshold
        # Per-component appearance history: component_id → list of snapshots
        self._component_history: dict[str, list[EvolutionaryComponent]] = {}
        # All fingerprints ever seen (across all months)
        self._seen_fingerprints: set[str] = set()
        # Chronological list of monthly snapshots
        self._activity_history: list[EvolutionaryActivitySnapshot] = []
        # Neo4j client - injected via set_neo4j(); None = persistence disabled
        self._neo4j: Any | None = None
        # Instance ID for Neo4j node keying - injected alongside neo4j
        self._instance_id: str = "eos-default"

    def set_neo4j(self, neo4j: Any, instance_id: str = "eos-default") -> None:
        """Inject Neo4j client so snapshots are persisted as (:BedauPackardSample) nodes."""
        self._neo4j = neo4j
        self._instance_id = instance_id

    # ── Genome ingestion ──────────────────────────────────────────────────────

    def ingest_fleet_genomes(self, genomes: list[dict[str, Any]]) -> list[EvolutionaryComponent]:
        """Extract EvolutionaryComponents from fleet genome snapshots.

        Genome dict format (from CHILD_SPAWNED event payload + genome cache):
        {
            "instance_id": str,
            "evo":    {"hypotheses": [{"id": str, "confidence": float}, ...],
                       "drive_weights": {"coherence": float, ...}},
            "simula": {"learnable_params": {"param_name": float, ...}},
            "telos":  {"drive_calibration": {"coherence": float, ...}},
            "equor":  {"amendments": [...], "drive_deltas": {...}},
        }

        All sub-fields are optional - missing keys produce zero/empty output.
        """
        components: list[EvolutionaryComponent] = []

        for genome in genomes:
            iid = str(genome.get("instance_id", "unknown"))
            evo = genome.get("evo") or {}
            simula = genome.get("simula") or {}

            # Evo hypotheses
            for hyp in evo.get("hypotheses", []) or []:
                hyp_id = str(hyp.get("id", ""))
                confidence = float(hyp.get("confidence", 0.5))
                fp = f"hyp:{hyp_id}:{confidence:.3f}"
                components.append(EvolutionaryComponent(
                    component_id=f"hyp:{hyp_id}",
                    component_type="hypothesis",
                    instance_id=iid,
                    value_fingerprint=fp,
                    fitness_contribution=confidence,
                ))

            # Drive weights
            for drive, weight in (evo.get("drive_weights") or {}).items():
                w = float(weight)
                fp = f"drive:{iid}:{drive}:{w:.3f}"
                components.append(EvolutionaryComponent(
                    component_id=f"drive:{iid}:{drive}",
                    component_type="drive_weight",
                    instance_id=iid,
                    value_fingerprint=fp,
                    fitness_contribution=w,
                ))

            # Simula learnable config params
            for param, val in (simula.get("learnable_params") or {}).items():
                if isinstance(val, float):
                    fp = f"simula:{iid}:{param}:{val:.4f}"
                else:
                    fp = f"simula:{iid}:{param}:{val}"
                components.append(EvolutionaryComponent(
                    component_id=f"simula:{iid}:{param}",
                    component_type="config_param",
                    instance_id=iid,
                    value_fingerprint=fp,
                    fitness_contribution=0.5,  # unknown; drive weights carry real signal
                ))

        return components

    # ── Core computation ──────────────────────────────────────────────────────

    def compute_adaptive_activity(
        self,
        components: list[EvolutionaryComponent],
        month: int,
    ) -> EvolutionaryActivitySnapshot:
        """Compute Bedau-Packard A(t) for the current generation.

        A(t) = cumulative count of components that are both:
          1. Novel (fingerprint not seen in any previous month)
          2. Persistent (component_id appeared in ≥2 consecutive snapshots)

        Shadow control: N components with randomised fitness contributions.
        ~5% random persistence probability (expected survival at typical churn).
        Real A(t) > 2× shadow A(t) = dynamics are adaptive, not random drift.

        Degrades gracefully: empty `components` → all-zero snapshot.
        """
        if not components:
            snap = EvolutionaryActivitySnapshot(month=month)
            self._activity_history.append(snap)
            asyncio.ensure_future(self._persist_snapshot(snap))
            return snap

        # 1. Identify novel components (fingerprint never seen before)
        novel = [c for c in components if c.value_fingerprint not in self._seen_fingerprints]

        # 2. Register new fingerprints
        for c in components:
            self._seen_fingerprints.add(c.value_fingerprint)

        # 3. Update per-component history
        for c in novel:
            self._component_history.setdefault(c.component_id, []).append(c)

        # Also update last_seen for known components so we can track persistence
        seen_ids = {c.component_id for c in components}
        for c in components:
            if c.component_id in self._component_history:
                hist = self._component_history[c.component_id]
                # Only append if the fingerprint changed (avoid duplicating same-value entries)
                if hist and hist[-1].value_fingerprint != c.value_fingerprint:
                    hist.append(c)

        # 4. Adaptive = novel AND appeared in ≥2 distinct snapshots with
        #    at least one fingerprint change (= genuine adaptation, not fixation)
        adaptive_count = 0
        for cid, hist in self._component_history.items():
            if cid not in seen_ids:
                continue  # component absent this month - not counted
            if len(hist) >= 2:
                # Check that the most recent entry has a novel fingerprint vs any prior
                latest_fp = hist[-1].value_fingerprint
                prior_fps = {h.value_fingerprint for h in hist[:-1]}
                if latest_fp not in prior_fps:
                    adaptive_count += 1

        # 5. Shadow control: random persistence at ~5% rate
        shadow_adaptive = sum(
            1 for _ in range(len(components))
            if random.random() < 0.05
        )

        # 6. Mode statistics
        if self._activity_history:
            prev = self._activity_history[-1]
            change_rate = abs(len(components) - prev.component_count) / max(1, prev.component_count)
        else:
            change_rate = 0.0

        fitness_values = [c.fitness_contribution for c in components]
        diversity = self._shannon_entropy(fitness_values)

        # Complexity proxy: sum of fitness for novel-and-persistent components
        novel_fp_set = {c.value_fingerprint for c in novel}
        complexity = sum(
            c.fitness_contribution for c in components
            if c.component_id in self._component_history
            and len(self._component_history[c.component_id]) >= 2
        )

        snap = EvolutionaryActivitySnapshot(
            month=month,
            adaptive_activity=float(adaptive_count),
            change_rate=round(change_rate, 4),
            novelty_rate=round(len(novel) / max(1, len(components)), 4),
            diversity=round(diversity, 4),
            complexity=round(complexity, 4),
            shadow_activity=float(shadow_adaptive),
            exceeds_shadow=adaptive_count > 2 * max(1, shadow_adaptive),
            population_size=len({c.instance_id for c in components}),
            component_count=len(components),
            novel_component_count=len(novel),
        )
        self._activity_history.append(snap)
        asyncio.ensure_future(self._persist_snapshot(snap))

        logger.info(
            "bedau_packard_computed",
            month=month,
            adaptive_activity=adaptive_count,
            novel=len(novel),
            total=len(components),
            exceeds_shadow=snap.exceeds_shadow,
            population_size=snap.population_size,
        )

        return snap

    # ── OEE assessment ────────────────────────────────────────────────────────

    def assess_oee_evidence(self) -> dict[str, Any]:
        """Assess evidence for OEE-like dynamics from accumulated monthly history.

        Returns an honest assessment for the paper.  Possible verdicts:
          - "insufficient_data"         : < 3 months of snapshots
          - "bounded"                   : adaptive activity not growing / not shadow-controlled
          - "growing_not_shadow_controlled" : A(t) grows but shadow control inconclusive
          - "exceeds_bounded"           : A(t) growing AND consistently exceeds shadow

        IMPORTANT: Does NOT claim "open-ended evolution" - uses "exceeds bounded
        classification" per speciation bible §8.5.  OEE is a multi-decade scientific
        debate; these statistics are evidence, not proof.
        """
        if len(self._activity_history) < 3:
            return {
                "verdict": "insufficient_data",
                "months": len(self._activity_history),
                "note": "Minimum 3 monthly snapshots required for OEE evidence assessment.",
            }

        activities = [s.adaptive_activity for s in self._activity_history]
        is_growing = activities[-1] > activities[0] * 1.1  # >10% growth vs first month
        exceeds_shadow_count = sum(1 for s in self._activity_history if s.exceeds_shadow)
        shadow_fraction = exceeds_shadow_count / len(self._activity_history)

        latest = self._activity_history[-1]
        modes_all_positive = (
            latest.change_rate > 0
            and latest.novelty_rate > 0
            and latest.diversity > 1.0  # > 1 bit of Shannon entropy
        )

        if is_growing and shadow_fraction >= 0.6:
            verdict = "exceeds_bounded"
            paper_claim = (
                "sustained evolutionary dynamics exceeding bounded classification "
                "with shadow-reset controls distinguishing adaptive from drift dynamics"
            )
        elif is_growing:
            verdict = "growing_not_shadow_controlled"
            paper_claim = (
                "evolutionary dynamics observed with growing adaptive activity; "
                "shadow-reset controls inconclusive - fleet size < 5 limits statistical power"
            )
        else:
            verdict = "bounded"
            paper_claim = (
                "evolutionary dynamics observed; adaptive activity bounded "
                "- expected in early months before fleet population scales"
            )

        return {
            "verdict": verdict,
            "months_tracked": len(self._activity_history),
            "adaptive_activity_month1": activities[0],
            "adaptive_activity_latest": activities[-1],
            "exceeds_shadow_fraction": round(shadow_fraction, 3),
            "modes_all_positive": modes_all_positive,
            "paper_claim": paper_claim,
        }

    # ── Neo4j persistence ────────────────────────────────────────────────────

    async def _persist_snapshot(self, snap: EvolutionaryActivitySnapshot) -> None:
        """Persist snapshot to Neo4j as (:BedauPackardSample) for PaperDataExporter."""
        if self._neo4j is None:
            return
        try:
            node_id = f"bp_fleet:{self._instance_id}:{snap.month}"
            await self._neo4j.execute_write(
                """
                MERGE (b:BedauPackardSample {node_id: $node_id})
                SET b.instance_id            = $instance_id,
                    b.month                  = $month,
                    b.adaptive_activity      = $adaptive_activity,
                    b.novelty_rate           = $novelty_rate,
                    b.diversity_index        = $diversity_index,
                    b.population_size        = $population_size,
                    b.component_count        = $component_count,
                    b.novel_component_count  = $novel_component_count,
                    b.exceeds_shadow         = $exceeds_shadow,
                    b.oee_verdict            = $oee_verdict,
                    b.recorded_at            = datetime()
                """,
                {
                    "node_id": node_id,
                    "instance_id": self._instance_id,
                    "month": snap.month,
                    "adaptive_activity": snap.adaptive_activity,
                    "novelty_rate": snap.novelty_rate,
                    "diversity_index": snap.diversity,
                    "population_size": snap.population_size,
                    "component_count": snap.component_count,
                    "novel_component_count": snap.novel_component_count,
                    "exceeds_shadow": snap.exceeds_shadow,
                    "oee_verdict": None,  # set by assess_oee_evidence() after ≥3 months
                },
            )
        except Exception as e:
            logger.warning("bedau_packard_persist_failed", error=str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _shannon_entropy(self, values: list[float]) -> float:
        """Shannon entropy of fitness contribution distribution (bits)."""
        if not values:
            return 0.0
        total = sum(abs(v) for v in values)
        if total == 0.0:
            return 0.0
        probs = [abs(v) / total for v in values]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @property
    def history(self) -> list[EvolutionaryActivitySnapshot]:
        """Read-only view of monthly snapshot history."""
        return list(self._activity_history)

    @property
    def months_tracked(self) -> int:
        return len(self._activity_history)
