"""
EcodiaOS - Discord Gateway WebSocket Loop (Phase 16h)

Maintains a persistent WebSocket connection to the Discord Gateway,
receives inbound messages and commands, and broadcasts them to the Synapse bus.

Discord uses a WebSocket-based gateway for real-time message delivery.
Unlike Telegram's HTTP polling/webhook model, Discord requires a persistent
connection with heartbeat to receive messages.

Message handling:
  - READY: bot received HELLO from gateway, ready to receive events
  - MESSAGE_CREATE: user sent a message to the bot's guild
  - INTERACTION_CREATE: user invoked a slash command
  - GATEWAY_DISPATCH (opcode 0): normal events
  - GATEWAY_HEARTBEAT (opcode 1): reply to server heartbeat
  - GATEWAY_HELLO (opcode 10): initial handshake with heartbeat interval

Env vars:
  ECODIAOS_CONNECTORS__DISCORD__BOT_TOKEN - bot token (required)
  ECODIAOS_DISCORD_GATEWAY_ENABLED - enable gateway loop (default: true if token set)
"""

from __future__ import annotations

import asyncio
import json
import zlib
from typing import TYPE_CHECKING, Any

import httpx
import structlog
import websockets

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.discord_gateway")

_DISCORD_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json&compression=zlib-stream"
_OPCODE_DISPATCH = 0
_OPCODE_HEARTBEAT = 1
_OPCODE_RECONNECT = 7
_OPCODE_INVALID_SESSION = 9
_OPCODE_HELLO = 10
_OPCODE_HEARTBEAT_ACK = 11

_EVENT_READY = "READY"
_EVENT_MESSAGE_CREATE = "MESSAGE_CREATE"
_EVENT_INTERACTION_CREATE = "INTERACTION_CREATE"


class DiscordGatewayLoop:
    """
    Manages a persistent Discord Gateway WebSocket connection.

    Listens for inbound messages, commands, and events.
    Emits DISCORD_MESSAGE_RECEIVED to the Synapse bus on inbound messages.

    Wire via supervised_task in registry.py Phase 11:

        gateway = DiscordGatewayLoop(discord_connector, synapse.event_bus)
        tasks["discord_gateway"] = supervised_task(
            "discord_gateway",
            gateway.run(),
            max_restarts=20,
        )
    """

    def __init__(
        self,
        discord_connector: Any,
        event_bus: EventBus | None = None,
    ) -> None:
        self._connector = discord_connector
        self._event_bus = event_bus
        self._logger = logger.bind(system="identity.discord_gateway")
        self._session_id: str = ""
        self._heartbeat_interval_ms: int = 0
        self._ws: Any = None
        self._decompress = zlib.decompressobj()
        self._last_seq: int | None = None

    async def run(self) -> None:
        """
        Main event loop: connect to Discord Gateway, maintain heartbeat, handle events.

        Runs indefinitely. Exceptions are logged but do not crash the loop -
        supervised_task will catch and restart.
        """
        while True:
            try:
                await self._gateway_loop()
            except asyncio.CancelledError:
                self._logger.info("discord_gateway_cancelled")
                raise
            except Exception as exc:
                self._logger.error("discord_gateway_loop_error", error=str(exc))
                await asyncio.sleep(5)

    async def _gateway_loop(self) -> None:
        """Connect to Discord and handle events until reconnect or error."""
        token = self._connector._resolve_token()
        if not token:
            self._logger.warning("discord_gateway_no_token")
            await asyncio.sleep(30)
            return

        headers = {
            "User-Agent": "DiscordBot (EcodiaOS, 1.0)",
        }

        try:
            async with websockets.connect(_DISCORD_GATEWAY_URL, subprotocols=[]) as ws:
                self._ws = ws
                self._logger.info("discord_gateway_connected")

                # Event loop: receive and handle messages
                async for raw_msg in ws:
                    await self._handle_message(raw_msg, token)

        except websockets.exceptions.ConnectionClosed as exc:
            self._logger.warning("discord_gateway_connection_closed", code=exc.rcvd.code)
        except Exception as exc:
            self._logger.error("discord_gateway_error", error=str(exc))

    async def _handle_message(self, raw_msg: str | bytes, bot_token: str) -> None:
        """Parse and dispatch a Discord gateway message."""
        try:
            # Discord compresses with zlib; decompress if bytes
            if isinstance(raw_msg, bytes):
                raw_msg = self._decompress.decompress(raw_msg).decode("utf-8")

            payload = json.loads(raw_msg)
        except Exception as exc:
            self._logger.warning("discord_gateway_parse_error", error=str(exc))
            return

        opcode = payload.get("op")
        data = payload.get("d", {})
        event_type = payload.get("t")
        seq = payload.get("s")

        if seq is not None:
            self._last_seq = seq

        # Handle gateway heartbeat request
        if opcode == _OPCODE_HELLO:
            self._heartbeat_interval_ms = data.get("heartbeat_interval", 45_000)
            # Identify: send bot token and resume info
            identify_payload = {
                "op": 2,  # IDENTIFY
                "d": {
                    "token": f"Bot {bot_token}",
                    "intents": 1 | 32 | 512,  # GUILDS | GUILD_MEMBERS | MESSAGE_CONTENT
                    "properties": {
                        "os": "linux",
                        "browser": "EcodiaOS",
                        "device": "EcodiaOS",
                    },
                },
            }
            await self._ws.send(json.dumps(identify_payload))
            self._logger.info("discord_gateway_identified")

            # Start heartbeat task
            asyncio.create_task(self._heartbeat_loop())

        # Handle heartbeat ack
        elif opcode == _OPCODE_HEARTBEAT_ACK:
            self._logger.debug("discord_gateway_heartbeat_ack")

        # Handle dispatch (normal events)
        elif opcode == _OPCODE_DISPATCH and event_type:
            await self._handle_dispatch(event_type, data)

        # Handle reconnect
        elif opcode == _OPCODE_RECONNECT:
            self._logger.info("discord_gateway_reconnect_requested")
            await self._ws.close()

        # Handle invalid session
        elif opcode == _OPCODE_INVALID_SESSION:
            self._logger.warning("discord_gateway_invalid_session")
            self._session_id = ""
            await self._ws.close()

    async def _handle_dispatch(self, event_type: str, data: dict[str, Any]) -> None:
        """Dispatch event handler: route by event type."""
        if event_type == _EVENT_READY:
            self._session_id = data.get("session_id", "")
            self._logger.info("discord_gateway_ready", session_id=self._session_id)

        elif event_type == _EVENT_MESSAGE_CREATE:
            await self._on_message_create(data)

        elif event_type == _EVENT_INTERACTION_CREATE:
            await self._on_interaction_create(data)

    async def _on_message_create(self, msg_data: dict[str, Any]) -> None:
        """Handle inbound MESSAGE_CREATE event (user sent a message)."""
        msg_id = msg_data.get("id", "")
        author_id = msg_data.get("author", {}).get("id", "")
        author_username = msg_data.get("author", {}).get("username", "")
        channel_id = msg_data.get("channel_id", "")
        content = msg_data.get("content", "")

        self._logger.info(
            "discord_message_received",
            channel_id=channel_id,
            author=author_username,
            content_len=len(content),
        )

        # Emit to Synapse bus
        if self._event_bus:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                event = SynapseEvent(
                    event_type=SynapseEventType.DISCORD_MESSAGE_RECEIVED,
                    source_system="identity",
                    data={
                        "message_id": msg_id,
                        "channel_id": channel_id,
                        "author_id": author_id,
                        "author_username": author_username,
                        "content": content,
                    },
                )
                await self._event_bus.publish(event)
            except Exception as exc:
                self._logger.warning("discord_message_emit_failed", error=str(exc))

    async def _on_interaction_create(self, interaction_data: dict[str, Any]) -> None:
        """Handle inbound INTERACTION_CREATE event (slash command)."""
        interaction_id = interaction_data.get("id", "")
        interaction_token = interaction_data.get("token", "")
        user_id = interaction_data.get("member", {}).get("user", {}).get("id", "")
        user_username = interaction_data.get("member", {}).get("user", {}).get("username", "")
        command_name = interaction_data.get("data", {}).get("name", "")
        channel_id = interaction_data.get("channel_id", "")

        self._logger.info(
            "discord_interaction_received",
            interaction_id=interaction_id,
            command=command_name,
            user=user_username,
        )

        # Emit to Synapse bus
        if self._event_bus:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                event = SynapseEvent(
                    event_type=SynapseEventType.DISCORD_COMMAND_RECEIVED,
                    source_system="identity",
                    data={
                        "interaction_id": interaction_id,
                        "interaction_token": interaction_token,
                        "channel_id": channel_id,
                        "user_id": user_id,
                        "user_username": user_username,
                        "command": command_name,
                    },
                )
                await self._event_bus.publish(event)
            except Exception as exc:
                self._logger.warning("discord_interaction_emit_failed", error=str(exc))

    async def _heartbeat_loop(self) -> None:
        """Send heartbeat to Discord Gateway every heartbeat_interval_ms."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval_ms / 1000.0)
                if self._ws and not self._ws.closed:
                    heartbeat = {
                        "op": _OPCODE_HEARTBEAT,
                        "d": self._last_seq,
                    }
                    await self._ws.send(json.dumps(heartbeat))
                    self._logger.debug("discord_gateway_heartbeat_sent")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.warning("discord_heartbeat_error", error=str(exc))
                break
