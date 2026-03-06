"""
EcodiaOS — Promote instance autonomy level to PARTNER (level 2).

Bounty hunting requires autonomy level 2 (PARTNER).  Run this once after
initial deployment to bootstrap the Self node so Equor stops blocking
hunt_bounties intents.

Usage:
    python -m scripts.promote_autonomy          # defaults to level 2
    python -m scripts.promote_autonomy --level 3  # steward (requires governance)

Requires:
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD env vars (or ecodiaos config).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

logger = structlog.get_logger()


async def main(target_level: int) -> None:
    from clients.neo4j import Neo4jClient
    from config import load_config
    from systems.equor.autonomy import apply_autonomy_change, get_autonomy_level

    config = load_config()
    neo4j = Neo4jClient(config.neo4j)
    await neo4j.connect()

    try:
        current = await get_autonomy_level(neo4j)
        print(f"Current autonomy level: {current}")

        if current >= target_level:
            print(f"Already at level {current} (>= requested {target_level}). Nothing to do.")
            return

        result = await apply_autonomy_change(
            neo4j=neo4j,
            new_level=target_level,
            reason="bootstrap: bounty hunting required for metabolic survival",
            actor="operator",
        )

        print(f"Promoted: {result['previous_level']} → {result['new_level']}")
        print(f"Governance record: {result['record_id']}")

        # Verify
        verified = await get_autonomy_level(neo4j)
        print(f"Verified autonomy level: {verified}")
        assert verified == target_level, f"Expected {target_level}, got {verified}"
    finally:
        await neo4j.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote EcodiaOS autonomy level")
    parser.add_argument(
        "--level", type=int, default=2,
        help="Target autonomy level (default: 2 = PARTNER)",
    )
    args = parser.parse_args()

    if args.level not in (1, 2, 3):
        print(f"Invalid level {args.level}. Must be 1, 2, or 3.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main(args.level))
