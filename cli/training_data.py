#!/usr/bin/env python3
"""
EcodiaOS - RE Training Data CLI

Extracts the 5 structured training streams from Neo4j and exports them
as JSONL compatible with systems/simula/training/train_lora.py.

Usage:
    # Full extraction + export (organism does NOT need to be running)
    python -m cli.training_data extract

    # With options
    python -m cli.training_data extract \\
        --output data/training/batch.jsonl \\
        --lookback 60 \\
        --min-score 0.35

    # Stats only - no export, just counts from Neo4j
    python -m cli.training_data stats
    python -m cli.training_data stats --lookback 90

Required environment variables:
    ECODIAOS_NEO4J_URI       - e.g. neo4j+s://xxx.databases.neo4j.io
    ECODIAOS_NEO4J_PASSWORD  - Neo4j password

Optional:
    ECODIAOS_NEO4J_USERNAME  - default "neo4j"
    RE_TRAINING_EXPORT_DIR   - local JSONL output dir (default: data/re_training_batches)
    RE_TRAINING_S3_BUCKET    - if set, also uploads to S3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

# ─── ANSI colours (minimal - mirrors observatory.py style) ───────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _ok(text: str) -> str:    return _c(text, _GREEN)
def _warn(text: str) -> str:  return _c(text, _YELLOW)
def _err(text: str) -> str:   return _c(text, _RED)
def _head(text: str) -> str:  return _c(text, _BOLD + _CYAN)
def _dim(text: str) -> str:   return _c(text, _DIM)


# ─── Neo4j bootstrap ─────────────────────────────────────────────────────────


def _build_neo4j() -> Any:
    """
    Construct a Neo4jClient from environment variables.
    Does NOT call connect() - the caller must do that.
    """
    from clients.neo4j import Neo4jClient
    from config import Neo4jConfig

    uri = os.environ.get("ECODIAOS_NEO4J_URI", "").strip()
    password = os.environ.get("ECODIAOS_NEO4J_PASSWORD", "").strip()
    username = os.environ.get("ECODIAOS_NEO4J_USERNAME", "neo4j").strip()

    if not uri:
        print(_err("ERROR: ECODIAOS_NEO4J_URI is not set."), file=sys.stderr)
        sys.exit(1)
    if not password:
        print(_err("ERROR: ECODIAOS_NEO4J_PASSWORD is not set."), file=sys.stderr)
        sys.exit(1)

    cfg = Neo4jConfig(uri=uri, username=username, password=password)
    return Neo4jClient(cfg)


# ─── Subcommand: extract ─────────────────────────────────────────────────────


async def _cmd_extract(args: argparse.Namespace) -> int:
    from systems.reasoning_engine.export_pipeline import run_export

    neo4j = _build_neo4j()
    try:
        await neo4j.connect()

        print(_head("\n=== EcodiaOS RE Training Data Extractor ==="))
        print(f"  Lookback : {args.lookback} days")
        print(f"  Min score: {args.min_score}")
        print(f"  Output   : {args.output or '(auto-generated)'}\n")

        result = await run_export(
            neo4j=neo4j,
            output_path=args.output,
            lookback_days=args.lookback,
            min_score=args.min_score,
        )

        if result.error:
            print(_err(f"\nEXTRACTION FAILED: {result.error}"))
            return 1

        print(_ok("\n=== Export Complete ==="))
        print(f"  Raw examples extracted : {result.total_raw}")
        print(f"  After quality filter   : {result.quality_stats.get('post_filter_count', '?')}")
        print(f"  After diversity filter : {result.quality_stats.get('post_diversity_count', '?')}")
        print(f"  Final JSONL records    : {_ok(str(result.total_exported))}")
        print(f"  Mean quality score     : {result.mean_quality_score:.3f}")
        print(f"  Duration               : {result.duration_ms}ms")
        print()

        if result.stream_counts:
            print(_head("  Per-stream breakdown:"))
            for sid, count in sorted(result.stream_counts.items()):
                bar = "█" * min(count, 40)
                pct = count / max(result.total_exported, 1) * 100
                label = sid.replace("stream_", "S")
                print(f"    {label}: {count:4d}  {_dim(bar)}  {pct:.0f}%")

        print()
        print(_head("  Output:"))
        for path in result.output_paths:
            print(f"    {_ok(path)}")

        if not result.success:
            print(_warn("\nWarning: export completed but 0 records written."))
            return 1

    finally:
        await neo4j.close()

    return 0


# ─── Subcommand: stats ────────────────────────────────────────────────────────


async def _cmd_stats(args: argparse.Namespace) -> int:
    from systems.reasoning_engine.export_pipeline import run_stats

    neo4j = _build_neo4j()
    try:
        await neo4j.connect()

        print(_head(f"\n=== RE Training Data Stats (lookback: {args.lookback}d) ===\n"))

        stats = await run_stats(neo4j, lookback_days=args.lookback)

        stream_counts = stats.get("stream_counts", {})
        total = stats.get("total_queryable", 0)

        stream_labels = {
            "stream_1_successful_chains":      "Stream 1 - Successful reasoning chains",
            "stream_2_failure_corrections":    "Stream 2 - Failures with corrections",
            "stream_3_constitutional_edge_cases": "Stream 3 - Constitutional edge cases",
            "stream_4_causal_chains":          "Stream 4 - Kairos causal chains",
            "stream_5_evo_experiments":        "Stream 5 - Evo hypothesis results",
        }

        for key, label in stream_labels.items():
            count = stream_counts.get(key, 0)
            if count < 0:
                status = _warn("QUERY FAILED")
            elif count == 0:
                status = _dim("0  (no data)")
            else:
                status = _ok(str(count))
            print(f"  {label}: {status}")

        print()
        print(f"  Total queryable: {_ok(str(total))}")
        print()

        # Diversity forecast
        if total > 0:
            cap_30 = int(total * 0.30)
            print(_head("  Diversity forecast (30% cap per stream):"))
            for key, label in stream_labels.items():
                count = stream_counts.get(key, 0)
                if count <= 0:
                    continue
                surviving = min(count, cap_30)
                print(f"    {label.split(' - ')[0]}: {count} → {surviving} after cap")
            print()

        print(_dim(f"  Note: {stats.get('note', '')}"))
        print()

    finally:
        await neo4j.close()

    return 0


# ─── Argument parsing ─────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli.training_data",
        description="EcodiaOS RE training data extraction CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = sub.add_parser(
        "extract",
        help="Extract all 5 streams and write JSONL for train_lora.py",
    )
    p_extract.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSONL path. Default: auto-generated in RE_TRAINING_EXPORT_DIR.",
    )
    p_extract.add_argument(
        "--lookback", "-l",
        type=int,
        default=30,
        help="Days of history to query (default: 30). Stream 3 always uses 90d.",
    )
    p_extract.add_argument(
        "--min-score", "-s",
        type=float,
        default=0.30,
        dest="min_score",
        help="Minimum quality score to include an example (default: 0.30).",
    )

    # stats
    p_stats = sub.add_parser(
        "stats",
        help="Show per-stream row counts without exporting (fast)",
    )
    p_stats.add_argument(
        "--lookback", "-l",
        type=int,
        default=30,
        help="Days of history to query (default: 30).",
    )

    return parser


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        coro = _cmd_extract(args)
    elif args.command == "stats":
        coro = _cmd_stats(args)
    else:
        parser.print_help()
        sys.exit(1)

    exit_code = asyncio.run(coro)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
