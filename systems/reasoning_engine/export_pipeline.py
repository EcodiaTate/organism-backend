"""
EcodiaOS - RE Training Export Pipeline

Orchestrates the full batch extraction → quality → formatting → JSONL export
pipeline for the custom reasoning engine training data.

This is the entry point called by the CLI and is independent of the real-time
RETrainingExporter in core/re_training_exporter.py, which collects live events.
This pipeline mines historical data from Neo4j.

Output JSONL format is compatible with systems/simula/training/train_lora.py:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    result = await run_export(neo4j, output_path="data/training/batch.jsonl")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from systems.reasoning_engine.quality_pipeline import apply_full_quality_pass
from systems.reasoning_engine.scaffold_formatter import format_batch
from systems.reasoning_engine.training_data_extractor import TrainingDataExtractor

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger("reasoning_engine.export_pipeline")

# ─── Config ───────────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT_DIR = os.environ.get(
    "RE_TRAINING_EXPORT_DIR", "data/re_training_batches"
)
_DEFAULT_S3_BUCKET = os.environ.get("RE_TRAINING_S3_BUCKET", "ecodiaos-re-training")
_DEFAULT_S3_PREFIX = os.environ.get("RE_TRAINING_S3_PREFIX", "structured/")
_DEFAULT_LOOKBACK_DAYS = 30
_DEFAULT_MIN_SCORE = 0.30


# ─── Result type ──────────────────────────────────────────────────────────────


@dataclass
class ExportResult:
    """Stats and output paths for a completed export run."""

    total_raw: int = 0            # total examples before quality filter
    total_exported: int = 0       # examples in final JSONL
    mean_quality_score: float = 0.0
    stream_counts: dict[str, int] = field(default_factory=dict)
    output_paths: list[str] = field(default_factory=list)
    duration_ms: int = 0
    quality_stats: dict[str, Any] = field(default_factory=dict)
    error: str = ""               # non-empty if pipeline failed

    @property
    def success(self) -> bool:
        return not self.error and self.total_exported > 0


# ─── Main export function ─────────────────────────────────────────────────────


async def run_export(
    neo4j: Neo4jClient,
    output_path: str | None = None,
    s3_bucket: str | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> ExportResult:
    """
    Full pipeline: extract → quality → format → write JSONL.

    Args:
        neo4j:        Connected Neo4j client.
        output_path:  Local file path for JSONL output. If None, uses env
                      RE_TRAINING_EXPORT_DIR / timestamped filename.
        s3_bucket:    S3 bucket for upload. If None, skips S3 (local only).
        lookback_days: How far back to query Neo4j (default 30 days).
        min_score:    Quality filter threshold (default 0.30).

    Returns:
        ExportResult with per-stream counts, quality stats, and output paths.
    """
    t0 = time.monotonic()
    result = ExportResult()

    try:
        # ── Step 1: Extract all 5 streams from Neo4j ──────────────────────────
        logger.info("export_pipeline_start", lookback_days=lookback_days)
        extractor = TrainingDataExtractor(neo4j)
        raw_examples = await extractor.extract_all_streams(lookback_days=lookback_days)
        result.total_raw = len(raw_examples)

        if not raw_examples:
            logger.warning("export_pipeline_no_examples")
            result.error = "no examples extracted from Neo4j"
            return result

        logger.info("extraction_complete", raw_count=result.total_raw)

        # ── Step 2: Quality scoring, filtering, diversity, temporal span ───────
        quality_examples, quality_stats = apply_full_quality_pass(
            raw_examples,
            min_score=min_score,
        )
        result.quality_stats = quality_stats
        result.mean_quality_score = quality_stats.get("mean_quality_score", 0.0)
        result.stream_counts = quality_stats.get("stream_counts", {})

        logger.info(
            "quality_pass_complete",
            pre_filter=quality_stats["pre_filter_count"],
            post_filter=quality_stats["post_filter_count"],
            final=quality_stats["final_count"],
            mean_quality=result.mean_quality_score,
        )

        if not quality_examples:
            result.error = f"all {result.total_raw} examples scored below min_score={min_score}"
            return result

        # ── Step 3: Format into Step 1-5 reasoning scaffold ───────────────────
        formatted = format_batch(quality_examples)
        result.total_exported = len(formatted)

        logger.info("formatting_complete", formatted_count=result.total_exported)

        if not formatted:
            result.error = "formatting produced 0 valid scaffolds"
            return result

        # ── Step 4: Write JSONL ────────────────────────────────────────────────
        dest_path = _resolve_output_path(output_path)
        _write_jsonl(formatted, dest_path)
        result.output_paths.append(f"local://{dest_path}")
        logger.info("jsonl_written", path=str(dest_path), count=result.total_exported)

        # ── Step 5: S3 upload (optional) ──────────────────────────────────────
        bucket = s3_bucket or _DEFAULT_S3_BUCKET
        if _s3_configured(bucket):
            s3_path = _upload_to_s3(dest_path, bucket)
            if s3_path:
                result.output_paths.append(s3_path)

    except Exception as exc:
        logger.error("export_pipeline_failed", error=str(exc), exc_info=True)
        result.error = str(exc)

    result.duration_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "export_pipeline_complete",
        total_exported=result.total_exported,
        mean_quality=result.mean_quality_score,
        duration_ms=result.duration_ms,
        success=result.success,
        paths=result.output_paths,
    )
    return result


# ─── Stats-only (no export) ───────────────────────────────────────────────────


async def run_stats(
    neo4j: Neo4jClient,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Run COUNT queries only - much cheaper than full extraction.
    Used by `python -m cli.training_data stats`.
    """
    extractor = TrainingDataExtractor(neo4j)
    counts = await extractor.stream_counts(lookback_days=lookback_days)

    total = sum(v for v in counts.values() if v >= 0)
    return {
        "stream_counts": counts,
        "total_queryable": total,
        "lookback_days": lookback_days,
        "note": (
            "Counts are raw Neo4j rows before quality filtering. "
            "Expect ~30-60% to survive quality pass."
        ),
    }


# ─── I/O helpers ──────────────────────────────────────────────────────────────


def _resolve_output_path(output_path: str | None) -> Path:
    """Resolve or generate the output JSONL path."""
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    export_dir = Path(_DEFAULT_OUTPUT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return export_dir / f"training_structured_{timestamp}.jsonl"


def _write_jsonl(records: list[dict[str, Any]], dest: Path) -> None:
    """Write records as JSON Lines. Only `messages` key goes to the file
    for train_lora.py compatibility; metadata keys are stripped."""
    lines = []
    for rec in records:
        # train_lora.py reads either "messages" or "instruction" key;
        # we use "messages" - include only what the trainer needs.
        out = {"messages": rec["messages"]}
        lines.append(json.dumps(out, ensure_ascii=False))

    dest.write_text("\n".join(lines), encoding="utf-8")


def _s3_configured(bucket: str) -> bool:
    """Return True if S3 upload should be attempted."""
    # Only attempt if bucket is non-default or explicitly set via env
    return bool(os.environ.get("RE_TRAINING_S3_BUCKET"))


def _upload_to_s3(local_path: Path, bucket: str) -> str | None:
    """Upload JSONL to S3. Returns s3://... path or None on failure."""
    try:
        import boto3  # type: ignore[import]

        s3 = boto3.client("s3")
        key = f"{_DEFAULT_S3_PREFIX}{local_path.name}"
        s3.upload_file(str(local_path), bucket, key)
        s3_path = f"s3://{bucket}/{key}"
        logger.info("s3_upload_complete", path=s3_path)
        return s3_path
    except ImportError:
        logger.debug("boto3_not_available_skipping_s3")
        return None
    except Exception:
        logger.warning("s3_upload_failed", exc_info=True)
        return None
