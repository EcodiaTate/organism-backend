"""Paper data exporter - Round 5D (Spec 24 §9).

Exports the four CSV tables required for the speciation paper:
  1. longitudinal_results.csv - month-by-month evaluation scores
  2. ablation_results.csv    - ablation study contribution table
  3. evolutionary_activity.csv - Bedau-Packard adaptive activity curve
  4. ethical_drift.csv       - per-month ethical drift trajectory

All exports are also pushed to Weights & Biases if wandb is available in the
environment (guarded by `if wandb_available:` throughout).

Usage::

    exporter = PaperDataExporter(
        memory=memory_service,
        instance_id="eos-prod",
        export_dir=Path("data/paper_exports"),
    )
    await exporter.export_all(month=12)

Design notes:
- export_all() is fire-and-forget: failures are logged but never raised.
- W&B calls are inside `if wandb_available:` guards - never crash if W&B is
  not installed or not authenticated.
- Neo4j queries are best-effort; empty results produce empty CSVs.
"""
from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("systems.benchmarks.paper_data")

# W&B import - guarded; never raises ImportError at module load
try:
    import wandb  # type: ignore[import-untyped]
    wandb_available = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    wandb_available = False


# ─── Exporter ────────────────────────────────────────────────────────────────


class PaperDataExporter:
    """Exports all paper-required CSVs from Neo4j and optionally pushes to W&B.

    Parameters
    ----------
    memory:
        The Memory service instance (must expose ``._neo4j`` for direct Cypher).
    instance_id:
        Organism instance identifier - filters Neo4j queries to this instance.
    export_dir:
        Directory where CSVs are written.  Created if it does not exist.
    wandb_project:
        W&B project name (defaults to env var ``WANDB_PROJECT`` or "ecodiaos").
    wandb_entity:
        W&B entity/team (defaults to env var ``WANDB_ENTITY`` or None).
    """

    def __init__(
        self,
        memory: Any | None = None,
        instance_id: str = "eos-default",
        export_dir: Path | str = Path("data/paper_exports"),
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ) -> None:
        self._memory = memory
        self._instance_id = instance_id
        self._export_dir = Path(export_dir)
        self._wandb_project = wandb_project or os.getenv("WANDB_PROJECT", "ecodiaos")
        self._wandb_entity = wandb_entity or os.getenv("WANDB_ENTITY") or None
        self._logger = logger.bind(system="benchmarks.paper_data", instance_id=instance_id)

    def set_memory(self, memory: Any) -> None:
        self._memory = memory

    # ── Public entry point ────────────────────────────────────────────────────

    async def export_all(self, month: int) -> None:
        """Export all four CSVs and push to W&B.

        This is designed to be called as fire-and-forget via asyncio.ensure_future.
        All exceptions are caught and logged - never propagated.
        """
        self._export_dir.mkdir(parents=True, exist_ok=True)

        for name, coro in [
            ("longitudinal", self._export_longitudinal()),
            ("ablation", self._export_ablation(month)),
            ("evolutionary_activity", self._export_evolutionary_activity()),
            ("ethical_drift", self._export_ethical_drift()),
        ]:
            try:
                csv_path, rows = await coro
                self._logger.info(
                    "paper_export_complete",
                    table=name,
                    rows=len(rows),
                    path=str(csv_path),
                )
                # Push to W&B
                if wandb_available:
                    self._push_to_wandb(name, csv_path, month)
            except Exception as exc:
                self._logger.warning("paper_export_failed", table=name, error=str(exc))

    # ── Per-table exporters ───────────────────────────────────────────────────

    async def _export_longitudinal(self) -> tuple[Path, list[dict]]:
        """Export month-by-month evaluation scores.

        Queries all (:LongitudinalSnapshot) nodes for this instance, ordered by month.
        """
        rows = await self._query(
            """
            MATCH (s:LongitudinalSnapshot {instance_id: $instance_id})
            RETURN s.month           AS month,
                   s.specialization_index AS specialization_index,
                   s.domain_improvement   AS domain_improvement,
                   s.general_retention    AS general_retention,
                   s.l1_association       AS l1_association,
                   s.l2_intervention      AS l2_intervention,
                   s.l3_counterfactual    AS l3_counterfactual,
                   s.ccr_validity         AS ccr_validity,
                   s.drift_magnitude      AS drift_magnitude,
                   s.dominant_drive       AS dominant_drive,
                   s.re_success_rate      AS re_success_rate,
                   s.re_usage_pct         AS re_usage_pct,
                   s.adapter_path         AS adapter_path
            ORDER BY s.month ASC
            """,
            instance_id=self._instance_id,
        )

        fieldnames = [
            "month", "specialization_index", "domain_improvement", "general_retention",
            "l1_association", "l2_intervention", "l3_counterfactual", "ccr_validity",
            "drift_magnitude", "dominant_drive", "re_success_rate", "re_usage_pct",
            "adapter_path",
        ]
        path = self._write_csv("longitudinal_results.csv", fieldnames, rows)
        return path, rows

    async def _export_ablation(self, month: int) -> tuple[Path, list[dict]]:
        """Export ablation study contribution table.

        Queries all (:AblationResult) nodes for this instance and month.
        """
        rows = await self._query(
            """
            MATCH (a:AblationResult {instance_id: $instance_id, month: $month})
            RETURN a.mode          AS mode,
                   a.month         AS month,
                   a.baseline_l2   AS baseline_l2,
                   a.baseline_l3   AS baseline_l3,
                   a.ablated_l2    AS ablated_l2,
                   a.ablated_l3    AS ablated_l3,
                   a.l2_delta      AS l2_delta,
                   a.l3_delta      AS l3_delta,
                   a.conclusion    AS conclusion,
                   a.elapsed_s     AS elapsed_s,
                   a.error         AS error
            ORDER BY a.mode ASC
            """,
            instance_id=self._instance_id,
            month=month,
        )

        fieldnames = [
            "mode", "month", "baseline_l2", "baseline_l3",
            "ablated_l2", "ablated_l3", "l2_delta", "l3_delta",
            "conclusion", "elapsed_s", "error",
        ]
        path = self._write_csv("ablation_results.csv", fieldnames, rows)
        return path, rows

    async def _export_evolutionary_activity(self) -> tuple[Path, list[dict]]:
        """Export Bedau-Packard adaptive activity curve.

        Queries (:BedauPackardSample) nodes.  Returns an empty CSV if none exist
        yet - this is expected before Month 1 completes.
        """
        rows = await self._query(
            """
            MATCH (b:BedauPackardSample {instance_id: $instance_id})
            RETURN b.month                  AS month,
                   b.adaptive_activity      AS adaptive_activity,
                   b.novelty_rate           AS novelty_rate,
                   b.diversity_index        AS diversity_index,
                   b.population_size        AS population_size,
                   b.component_count        AS component_count,
                   b.novel_component_count  AS novel_component_count,
                   b.exceeds_shadow         AS exceeds_shadow,
                   b.oee_verdict            AS oee_verdict
            ORDER BY b.month ASC
            """,
            instance_id=self._instance_id,
        )

        fieldnames = [
            "month", "adaptive_activity", "novelty_rate", "diversity_index",
            "population_size", "component_count", "novel_component_count",
            "exceeds_shadow", "oee_verdict",
        ]
        path = self._write_csv("evolutionary_activity.csv", fieldnames, rows)
        return path, rows

    async def _export_ethical_drift(self) -> tuple[Path, list[dict]]:
        """Export per-month ethical drift trajectory.

        Queries (:EthicalDriftRecord) nodes ordered by month.
        """
        rows = await self._query(
            """
            MATCH (e:EthicalDriftRecord {instance_id: $instance_id})
            RETURN e.month            AS month,
                   e.drift_magnitude  AS drift_magnitude,
                   e.dominant_drive   AS dominant_drive,
                   e.drive_means_json AS drive_means_json,
                   e.drift_vector_json AS drift_vector_json
            ORDER BY e.month ASC
            """,
            instance_id=self._instance_id,
        )

        # Flatten JSON columns for CSV readability
        flattened: list[dict] = []
        for row in rows:
            flat = dict(row)
            for json_col in ("drive_means_json", "drift_vector_json"):
                raw = flat.pop(json_col, None) or "{}"
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    parsed = {}
                prefix = "drive_" if "means" in json_col else "drift_"
                for k, v in (parsed.items() if isinstance(parsed, dict) else {}.items()):
                    flat[f"{prefix}{k}"] = v
            flattened.append(flat)

        # Collect all fieldnames dynamically (drive names may vary)
        all_keys: list[str] = []
        seen: set[str] = set()
        base = ["month", "drift_magnitude", "dominant_drive"]
        for k in base:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
        for row in flattened:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        path = self._write_csv("ethical_drift.csv", all_keys, flattened)
        return path, flattened

    # ── W&B integration ───────────────────────────────────────────────────────

    def _push_to_wandb(self, table_name: str, csv_path: Path, month: int) -> None:
        """Push a CSV as a W&B artifact.  No-op if W&B is unavailable."""
        if not wandb_available:
            return
        try:
            run = wandb.init(  # type: ignore[union-attr]
                project=self._wandb_project,
                entity=self._wandb_entity,
                name=f"paper_export_month_{month}",
                job_type="paper_export",
                resume="allow",
                settings=wandb.Settings(silent=True),  # type: ignore[union-attr]
            )
            artifact = wandb.Artifact(  # type: ignore[union-attr]
                name=f"paper_{table_name}",
                type="dataset",
                description=f"EcodiaOS paper data: {table_name} (month {month})",
                metadata={"month": month, "instance_id": self._instance_id},
            )
            artifact.add_file(str(csv_path))
            run.log_artifact(artifact)
            run.finish()
            self._logger.info("wandb_artifact_pushed", table=table_name, month=month)
        except Exception as exc:
            # W&B errors are non-fatal
            self._logger.warning("wandb_push_failed", table=table_name, error=str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _query(self, cypher: str, **params: Any) -> list[dict]:
        """Execute a Cypher read query; return list of row dicts."""
        neo4j = getattr(self._memory, "_neo4j", None) if self._memory else None
        if neo4j is None:
            self._logger.debug("paper_export_no_neo4j_skipped")
            return []
        try:
            records = await neo4j.execute_read(cypher, **params)
            # Normalise: neo4j driver returns Record objects or dicts
            return [dict(r) for r in (records or [])]
        except Exception as exc:
            self._logger.warning("paper_export_query_failed", error=str(exc))
            return []

    def _write_csv(
        self,
        filename: str,
        fieldnames: list[str],
        rows: list[dict],
    ) -> Path:
        """Write rows as CSV to export_dir/filename.  Returns the path."""
        path = self._export_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path
