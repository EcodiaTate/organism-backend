"""
EcodiaOS - Skia Worker (Shadow Infrastructure)

Standalone process that monitors the primary organism's heartbeat,
takes periodic state snapshots to IPFS, and triggers autonomous
restoration if the organism goes offline.

Designed to run on a separate cheap instance (different availability
zone, different cloud provider) so it survives the primary going down.

Usage:
    python -m ecodiaos.skia_worker
    python -m ecodiaos.skia_worker --config /etc/ecodiaos/config.yaml

Graceful shutdown:
    Handles SIGINT and SIGTERM. In-flight snapshots complete before exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path

import structlog
from dotenv import load_dotenv

# Resolve .env relative to backend/ dir (same pattern as simula_worker)
_BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_BACKEND_DIR / ".env", override=True)

from clients.neo4j import Neo4jClient
from clients.redis import RedisClient
from config import load_config
from systems.skia.service import SkiaService

logger = structlog.get_logger()


async def run_worker(config_path: str | None = None) -> None:
    """
    Main worker loop.

    1. Load config and build independent connection pools
    2. Initialize IdentityVault (from ORGANISM_VAULT_PASSPHRASE)
    3. Initialize SkiaService in standalone mode
    4. Start heartbeat monitor + snapshot pipeline
    5. Block until shutdown signal
    6. Graceful teardown
    """
    config = load_config(config_path)
    log = logger.bind(worker="skia", instance_id=config.instance_id)

    if not config.skia.enabled:
        log.warning("skia_not_enabled", hint="Set ORGANISM_SKIA__ENABLED=true")
        return

    # ── Build independent connection pools ────────────────────────
    neo4j_client = Neo4jClient(config.neo4j)
    await neo4j_client.connect()
    log.info("neo4j_connected")

    redis_client = RedisClient(config.redis)
    await redis_client.connect()
    log.info("redis_connected")

    # ── Initialize IdentityVault ──────────────────────────────────
    vault = None
    vault_passphrase = os.environ.get("ORGANISM_VAULT_PASSPHRASE", "")
    if vault_passphrase:
        from systems.identity.vault import IdentityVault
        vault = IdentityVault(passphrase=vault_passphrase)
        log.info("vault_initialized")
    else:
        log.warning("vault_passphrase_missing", snapshots="disabled")

    # ── Initialize SkiaService in standalone mode ─────────────────
    skia = SkiaService(
        config=config.skia,
        neo4j=neo4j_client,
        redis=redis_client,
        vault=vault,
        instance_id=config.instance_id,
        standalone=True,
    )
    await skia.initialize()
    await skia.start()

    log.info("skia_worker_started")

    # ── Shutdown signal handling ──────────────────────────────────
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        log.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal_handler)

    # ── Block until shutdown ──────────────────────────────────────
    await shutdown_event.wait()

    # ── Graceful teardown ─────────────────────────────────────────
    log.info("skia_worker_shutting_down")
    await skia.shutdown()
    await redis_client.close()
    await neo4j_client.close()
    log.info("skia_worker_stopped")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="EcodiaOS Skia Worker (Shadow Infrastructure)")
    parser.add_argument(
        "--config",
        default=os.getenv("ORGANISM_CONFIG_PATH"),
        help="Path to YAML config file (default: ORGANISM_CONFIG_PATH env var)",
    )
    args = parser.parse_args()
    asyncio.run(run_worker(args.config))


if __name__ == "__main__":
    main()
