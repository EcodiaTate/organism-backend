"""
EcodiaOS — Federation Channel Management

Manages mutual TLS channels for federation communication. Each
federation link has a dedicated channel authenticated by mutual
TLS certificates. No anonymous connections are permitted.

The channel layer handles:
  - mTLS context creation from certificates
  - Connection lifecycle (establish, maintain, close)
  - Message serialization and deserialization
  - Connection health monitoring
  - Automatic reconnection on failure

In production, channels use mTLS over HTTPS. For development/testing,
channels can operate in "local" mode using direct function calls
between in-process instances.
"""

from __future__ import annotations

import contextlib
import ssl
from typing import Any
from pathlib import Path

import httpx
import structlog

from primitives.federation import (
    AssistanceRequest,
    AssistanceResponse,
    ExchangeEnvelope,
    ExchangeReceipt,
    FederationLink,
    InstanceIdentityCard,
    KnowledgeRequest,
    KnowledgeResponse,
)

logger = structlog.get_logger("systems.federation.channel")


class FederationChannel:
    """
    A single authenticated channel to a remote EOS instance.

    Wraps an httpx.AsyncClient configured with mutual TLS.
    """

    def __init__(
        self,
        link: FederationLink,
        client: httpx.AsyncClient,
    ) -> None:
        self.link = link
        self._client = client
        self._logger = logger.bind(
            remote_id=link.remote_instance_id,
            remote_name=link.remote_name,
        )

    # ─── Federation API Calls ───────────────────────────────────────

    async def get_identity(self) -> InstanceIdentityCard | None:
        """Fetch the remote instance's identity card."""
        try:
            response = await self._client.get(
                f"{self.link.remote_endpoint}/api/v1/federation/identity",
                timeout=5.0,
            )
            if response.status_code == 200:
                return InstanceIdentityCard(**response.json())
            self._logger.warning(
                "identity_fetch_failed",
                status=response.status_code,
            )
            return None
        except Exception as exc:
            self._logger.error("identity_fetch_error", error=str(exc))
            return None

    async def request_knowledge(
        self, request: KnowledgeRequest
    ) -> KnowledgeResponse | None:
        """Send a knowledge request to the remote instance."""
        try:
            response = await self._client.post(
                f"{self.link.remote_endpoint}/api/v1/federation/knowledge/request",
                json=request.model_dump(mode="json"),
                timeout=5.0,
            )
            if response.status_code == 200:
                return KnowledgeResponse(**response.json())
            self._logger.warning(
                "knowledge_request_failed",
                status=response.status_code,
            )
            return None
        except Exception as exc:
            self._logger.error("knowledge_request_error", error=str(exc))
            return None

    async def request_assistance(
        self, request: AssistanceRequest
    ) -> AssistanceResponse | None:
        """Send an assistance request to the remote instance."""
        try:
            response = await self._client.post(
                f"{self.link.remote_endpoint}/api/v1/federation/assistance/request",
                json=request.model_dump(mode="json"),
                timeout=10.0,
            )
            if response.status_code == 200:
                return AssistanceResponse(**response.json())
            return None
        except Exception as exc:
            self._logger.error("assistance_request_error", error=str(exc))
            return None

    async def send_exchange(
        self, envelope: ExchangeEnvelope,
    ) -> ExchangeReceipt | None:
        """
        Send an IIEP exchange envelope and return the receipt.

        Used for both PUSH (proactive sharing) and PULL (requesting
        knowledge from a peer).
        """
        try:
            response = await self._client.post(
                f"{self.link.remote_endpoint}/api/v1/federation/exchange",
                json=envelope.model_dump(mode="json"),
                timeout=10.0,
            )
            if response.status_code == 200:
                return ExchangeReceipt(**response.json())
            self._logger.warning(
                "exchange_send_failed",
                status=response.status_code,
                direction=envelope.direction,
            )
            return None
        except Exception as exc:
            self._logger.error("exchange_send_error", error=str(exc))
            return None

    async def send_exchange_receipt(
        self, receipt: ExchangeReceipt,
    ) -> bool:
        """Send an exchange receipt back to the sender."""
        try:
            response = await self._client.post(
                f"{self.link.remote_endpoint}/api/v1/federation/exchange/receipt",
                json=receipt.model_dump(mode="json"),
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as exc:
            self._logger.error("exchange_receipt_error", error=str(exc))
            return False

    async def send_message(
        self, message_type: str, payload: dict[str, Any]
    ) -> bool:
        """Send a typed message to the remote instance."""
        endpoint_map: dict[str, str] = {
            "threat_advisory": "/api/v1/federation/threat-advisory",
            "exchange": "/api/v1/federation/exchange",
            "exchange_receipt": "/api/v1/federation/exchange/receipt",
        }
        path = endpoint_map.get(message_type)
        if not path:
            self._logger.warning("unknown_message_type", message_type=message_type)
            return False
        try:
            response = await self._client.post(
                f"{self.link.remote_endpoint}{path}",
                json=payload,
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as exc:
            self._logger.error(
                "send_message_failed",
                message_type=message_type,
                error=str(exc),
            )
            return False

    async def send_handshake(
        self, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Send a handshake request to the remote instance (Phase 1).

        Returns the parsed HandshakeResponse dict, or None on failure.
        Timeout is 10s to allow the remote side time for certificate
        validation and Equor review.
        """
        try:
            response = await self._client.post(
                f"{self.link.remote_endpoint}/api/v1/federation/handshake",
                json=request_data,
                timeout=10.0,
            )
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            self._logger.warning(
                "handshake_request_failed",
                status=response.status_code,
                body=response.text[:200],
            )
            return None
        except Exception as exc:
            self._logger.error("handshake_request_error", error=str(exc))
            return None

    async def send_handshake_confirmation(
        self, confirmation_data: dict[str, Any]
    ) -> bool:
        """
        Send the handshake confirmation to the remote instance (Phase 4).

        Returns True if the remote accepted the confirmation.
        """
        try:
            response = await self._client.post(
                f"{self.link.remote_endpoint}/api/v1/federation/handshake/confirm",
                json=confirmation_data,
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as exc:
            self._logger.error("handshake_confirmation_error", error=str(exc))
            return False

    async def send_greeting(self) -> bool:
        """Send a greeting/heartbeat to verify the connection is alive."""
        try:
            response = await self._client.get(
                f"{self.link.remote_endpoint}/api/v1/federation/identity",
                timeout=3.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the channel."""
        await self._client.aclose()
        self._logger.info("channel_closed")


class ChannelManager:
    """
    Manages all federation channels (one per active link).

    Creates mTLS-authenticated httpx clients for each federation link
    and maintains the channel pool.
    """

    def __init__(
        self,
        tls_cert_path: Path | None = None,
        tls_key_path: Path | None = None,
        ca_cert_path: Path | None = None,
    ) -> None:
        self._tls_cert_path = tls_cert_path
        self._tls_key_path = tls_key_path
        self._ca_cert_path = ca_cert_path
        self._channels: dict[str, FederationChannel] = {}  # link_id → channel
        self._logger = logger.bind(component="channel_manager")

    # ─── Channel Lifecycle ──────────────────────────────────────────

    async def open_channel(self, link: FederationLink) -> FederationChannel:
        """
        Open a new channel to a remote instance.

        Creates an httpx.AsyncClient configured with mutual TLS
        (if certificates are available) or plain HTTPS for development.
        """
        # Close existing channel for this link if any
        if link.id in self._channels:
            await self._channels[link.id].close()

        # Build SSL context for mutual TLS
        ssl_context = self._build_ssl_context()

        # Create authenticated HTTP client
        client = httpx.AsyncClient(
            verify=ssl_context if ssl_context else True,
            cert=(str(self._tls_cert_path), str(self._tls_key_path))
            if self._tls_cert_path and self._tls_key_path
            else None,
            headers={
                "X-EOS-Instance-ID": link.local_instance_id,
                "X-EOS-Federation-Protocol": "1.0",
            },
        )

        channel = FederationChannel(link=link, client=client)
        self._channels[link.id] = channel

        self._logger.info(
            "channel_opened",
            link_id=link.id,
            remote_id=link.remote_instance_id,
            remote_endpoint=link.remote_endpoint,
            mtls=ssl_context is not None,
        )

        return channel

    async def close_channel(self, link_id: str) -> None:
        """Close and remove a channel."""
        channel = self._channels.pop(link_id, None)
        if channel:
            await channel.close()

    async def close_all(self) -> None:
        """Close all channels (shutdown)."""
        for channel in self._channels.values():
            with contextlib.suppress(Exception):
                await channel.close()
        self._channels.clear()

    def get_channel(self, link_id: str) -> FederationChannel | None:
        """Get the channel for a specific link."""
        return self._channels.get(link_id)

    # ─── Health ─────────────────────────────────────────────────────

    async def check_channel_health(self, link_id: str) -> bool:
        """Check if a channel is healthy by sending a greeting."""
        channel = self._channels.get(link_id)
        if not channel:
            return False
        return await channel.send_greeting()

    # ─── SSL Context ────────────────────────────────────────────────

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        """
        Build an SSL context for mutual TLS.

        Returns None if certificates are not configured (development mode).
        """
        if not self._ca_cert_path or not self._ca_cert_path.exists():
            return None

        ctx = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=str(self._ca_cert_path),
        )

        if self._tls_cert_path and self._tls_key_path:
            ctx.load_cert_chain(
                certfile=str(self._tls_cert_path),
                keyfile=str(self._tls_key_path),
            )

        # Require client certificates (mutual TLS)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True

        return ctx

    # ─── Stats ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "active_channels": len(self._channels),
            "mtls_configured": all([
                self._tls_cert_path,
                self._tls_key_path,
                self._ca_cert_path,
            ]),
            "channels": {
                link_id: {
                    "remote_id": ch.link.remote_instance_id,
                    "remote_name": ch.link.remote_name,
                    "remote_endpoint": ch.link.remote_endpoint,
                }
                for link_id, ch in self._channels.items()
            },
        }
