"""
EcodiaOS — One-time migration: purge stale maintenance goals from Neo4j.

On boot, Nova loads every persisted Goal with status='active'|'suspended'.
After system-recovery events, goals with source='maintenance' accumulate
from prior sessions and immediately fill Nova's 20-goal active capacity,
crowding out real goals.

This script deletes Goal nodes where:
  - source = 'maintenance'
  - created_at < now() - 24 hours

Safety:
  - Uses DETACH DELETE on individual batches — no full graph lock.
  - Safe to run while EcodiaOS is live (Neo4j row-level locking).
  - Logs count before and after so the operator can verify.

Usage:
    python -m scripts.purge_stale_goals
    python -m scripts.purge_stale_goals --dry-run
    python -m scripts.purge_stale_goals --older-than-hours 48
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

import structlog

logger = structlog.get_logger()


async def main(dry_run: bool, older_than_hours: float) -> None:
    from clients.neo4j import Neo4jClient
    from config import load_config

    config = load_config()
    neo4j = Neo4jClient(config.neo4j)
    await neo4j.connect()

    cutoff = (datetime.now(tz=UTC) - timedelta(hours=older_than_hours)).isoformat()

    try:
        # ── Count before ─────────────────────────────────────────────
        count_query = (
            "MATCH (g:Goal) "
            "WHERE g.source = 'maintenance' AND g.created_at < $cutoff "
            "RETURN count(g) AS n"
        )
        rows = await neo4j.execute_read(count_query, {"cutoff": cutoff})
        stale_count: int = int(rows[0]["n"]) if rows else 0

        total_rows = await neo4j.execute_read(
            "MATCH (g:Goal) RETURN count(g) AS n", {}
        )
        total_count: int = int(total_rows[0]["n"]) if total_rows else 0

        logger.info(
            "purge_stale_goals_scan",
            total_goals=total_count,
            stale_maintenance_goals=stale_count,
            cutoff_iso=cutoff,
            older_than_hours=older_than_hours,
            dry_run=dry_run,
        )
        print(f"Total Goal nodes       : {total_count}")
        print(f"Stale maintenance goals: {stale_count}  (source=maintenance, created_at < {cutoff})")

        if stale_count == 0:
            print("Nothing to purge.")
            return

        if dry_run:
            print("[DRY-RUN] Would delete the above goals. Re-run without --dry-run to apply.")
            return

        # ── Delete in batches of 500 to avoid large TX memory ────────
        # DETACH DELETE removes the node and any relationships — no graph lock.
        deleted_total = 0
        batch_size = 500
        while True:
            delete_query = (
                "MATCH (g:Goal) "
                "WHERE g.source = 'maintenance' AND g.created_at < $cutoff "
                "WITH g LIMIT $batch "
                "DETACH DELETE g "
                "RETURN count(g) AS deleted"
            )
            del_rows = await neo4j.execute_write(
                delete_query,
                {"cutoff": cutoff, "batch": batch_size},
            )
            batch_deleted: int = int(del_rows[0]["deleted"]) if del_rows else 0
            deleted_total += batch_deleted
            logger.info("purge_batch_deleted", batch=batch_deleted, total_so_far=deleted_total)
            if batch_deleted < batch_size:
                break  # Last batch — nothing left

        # ── Count after ───────────────────────────────────────────────
        after_rows = await neo4j.execute_read(
            "MATCH (g:Goal) RETURN count(g) AS n", {}
        )
        after_count: int = int(after_rows[0]["n"]) if after_rows else 0

        logger.info(
            "purge_stale_goals_complete",
            deleted=deleted_total,
            goals_remaining=after_count,
        )
        print(f"\nDeleted  : {deleted_total} stale maintenance goals")
        print(f"Remaining: {after_count} Goal nodes")

    finally:
        await neo4j.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Purge stale maintenance goals from Neo4j"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Count targets without deleting (default: False)",
    )
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=1.0,
        help="Delete goals older than this many hours (default: 1)",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, older_than_hours=args.older_than_hours))
