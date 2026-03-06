"""
EcodiaOS — Distributed Fleet Shield Manager

Broadcasts deterministic XDP filters across the enterprise cluster via
Redis Pub/Sub.  Every node subscribes to the same channel, so a filter
generated on *any* node is deployed fleet-wide — including the
originator, which receives its own broadcast and deploys locally.

Usage (lifespan integration)::

    fleet = FleetShieldManager(redis_client)
    listener_task = asyncio.create_task(fleet.listen_and_deploy())
    yield
    await fleet.shutdown()
    listener_task.cancel()
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from clients.redis import RedisClient

logger = structlog.get_logger().bind(system="simula.distributed_shield")

# All nodes publish and subscribe on this channel.
_CHANNEL = "ecodiaos:shield:filters"


class FleetShieldManager:
    """Pub/Sub bridge between filter generation and fleet-wide XDP deployment."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis: Redis = redis_client.client
        self._listener_task: asyncio.Task[None] | None = None
        self._shutting_down = False

    # ── Broadcast (publisher side) ────────────────────────────────

    async def broadcast_filter(self, c_code: str) -> int:
        """Publish raw C-code to every subscribed node.

        Returns the number of subscribers that received the message
        (``PUBLISH`` return value).
        """
        receivers: int = await self._redis.publish(_CHANNEL, c_code)
        logger.info(
            "filter_broadcast_sent",
            code_bytes=len(c_code),
            receivers=receivers,
        )
        return receivers

    # ── Listener (subscriber side) ────────────────────────────────

    async def listen_and_deploy(self) -> None:
        """Long-running task: subscribe → receive → deploy locally.

        Designed to be wrapped in ``asyncio.create_task()`` during
        FastAPI startup.  Handles Redis disconnects gracefully with
        exponential back-off so a transient broker outage never crashes
        the application.
        """
        backoff = 1.0
        max_backoff = 30.0

        while not self._shutting_down:
            pubsub = self._redis.pubsub()
            try:
                await pubsub.subscribe(_CHANNEL)
                logger.info("fleet_shield_listener_subscribed", channel=_CHANNEL)
                backoff = 1.0  # reset on successful subscribe

                while not self._shutting_down:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0,
                    )
                    if message is None:
                        continue
                    if message["type"] != "message":
                        continue

                    payload = message["data"]
                    # Redis may return bytes or str depending on
                    # decode_responses; normalise to str.
                    if isinstance(payload, bytes):
                        payload = payload.decode("utf-8", errors="replace")

                    logger.info(
                        "filter_received_from_fleet",
                        code_bytes=len(payload),
                    )

                    # Deploy on a thread so the blocking BCC compilation
                    # never stalls the asyncio event loop.
                    await asyncio.to_thread(self._deploy_locally, payload)

            except asyncio.CancelledError:
                logger.info("fleet_shield_listener_cancelled")
                break
            except Exception as exc:
                if self._shutting_down:
                    break
                logger.warning(
                    "fleet_shield_listener_error",
                    error=str(exc),
                    retry_in=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            finally:
                try:
                    await pubsub.unsubscribe(_CHANNEL)
                    await pubsub.aclose()  # type: ignore[no-untyped-call]
                except Exception:
                    pass  # Best-effort cleanup

    # ── Local deployment (runs in a thread) ───────────────────────

    @staticmethod
    def _deploy_locally(bpf_c_code: str) -> None:
        """Compile and attach the XDP filter on this node.

        Runs inside ``asyncio.to_thread`` so the blocking BCC call
        doesn't hold up the event loop.
        """
        from systems.simula.inspector.shield import AutonomousShield

        shield = AutonomousShield.__new__(AutonomousShield)
        shield._llm = None  # type: ignore[assignment]
        shield._log = logger

        try:
            bpf, _blocklist = shield.deploy_filter_live(
                interface="eth0",
                bpf_c_code=bpf_c_code,
            )
            logger.info("fleet_filter_deployed_locally", interface="eth0")
        except Exception as exc:
            logger.error(
                "fleet_filter_deploy_failed",
                error=str(exc),
            )

    # ── Lifecycle helpers ─────────────────────────────────────────

    def start(self) -> asyncio.Task[None]:
        """Spawn the listener as a background task and return the handle."""
        self._shutting_down = False
        self._listener_task = asyncio.create_task(
            self.listen_and_deploy(),
            name="fleet-shield-listener",
        )
        logger.info("fleet_shield_manager_started")
        return self._listener_task

    async def shutdown(self) -> None:
        """Gracefully stop the listener task."""
        self._shutting_down = True
        if self._listener_task is not None and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None
        logger.info("fleet_shield_manager_stopped")
