"""
EcodiaOS - Social Interface: Discord Channel Client

Posts messages and images to a Discord channel on behalf of the organism.

Auth: Bot token. The vault must hold a token envelope with purpose="oauth_token"
containing:
    {
        "access_token": "<bot_token>",
        "channel_id": "123456789"  # optional; falls back to ORGANISM_DISCORD_CHANNEL_ID
    }

Env var fallback:
    ORGANISM_DISCORD_CHANNEL_ID - Discord channel ID (numeric)
    ORGANISM_CONNECTORS__DISCORD__BOT_TOKEN - bot token

API:
    POST https://discord.com/api/v10/channels/{channel_id}/messages
    POST https://discord.com/api/v10/channels/{channel_id}/messages (with file upload for image)

Char limits:
    - text message: 2,000 chars (Discord limit)
    - embed description: 4,096 chars

Note: This client targets CHANNEL posting (broadcast). For admin DMs use the
identity/connectors/discord.py + axon/executors/send_discord.py.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from interfaces.social.types import PostResult, SocialPlatform

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault, SealedEnvelope

logger = structlog.get_logger("interfaces.social.discord_channel")

_DISCORD_API_BASE = "https://discord.com/api/v10"
_MAX_MESSAGE_CHARS = 2_000
_MAX_EMBED_DESCRIPTION = 4_096


class DiscordClient:
    """
    Thin async HTTP client for Discord Bot API - channel posting.

    Credentials resolved from vault first, env var second.
    """

    def __init__(
        self,
        vault: IdentityVault | None = None,
        envelope: SealedEnvelope | None = None,
    ) -> None:
        self._vault = vault
        self._envelope = envelope
        self._logger = logger.bind(client="discord_channel")

    # ── Public API ─────────────────────────────────────────────────────────

    async def post_to_channel(
        self,
        text: str,
        channel_id: str | None = None,
    ) -> PostResult:
        """
        Send a text message to the configured channel.

        Args:
            text: Message text (≤2,000 chars).
            channel_id: Discord channel ID (numeric). Falls back to ORGANISM_DISCORD_CHANNEL_ID.

        Returns:
            PostResult - never raises.
        """
        token, default_channel = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.DISCORD, "no_credentials")

        target = channel_id or default_channel
        if not target:
            return PostResult.fail(SocialPlatform.DISCORD, "no_channel_id")

        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[: _MAX_MESSAGE_CHARS - 1] + "…"

        payload: dict[str, Any] = {
            "content": text,
        }

        return await self._bot_call(token, target, "POST", payload)

    async def post_with_image(
        self,
        image_url: str,
        caption: str = "",
        channel_id: str | None = None,
    ) -> PostResult:
        """
        Send an image with optional caption to the configured channel.

        Args:
            image_url: Publicly accessible HTTPS URL of the image.
            caption: Caption text (≤4,096 chars, sent as embed description).
            channel_id: Discord channel ID; falls back to env var.

        Returns:
            PostResult - never raises.
        """
        token, default_channel = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.DISCORD, "no_credentials")

        target = channel_id or default_channel
        if not target:
            return PostResult.fail(SocialPlatform.DISCORD, "no_channel_id")

        if len(caption) > _MAX_EMBED_DESCRIPTION:
            caption = caption[: _MAX_EMBED_DESCRIPTION - 1] + "…"

        # Discord image post: embed with image URL in the embed image field
        payload: dict[str, Any] = {
            "embeds": [
                {
                    "title": "Content",
                    "description": caption if caption else None,
                    "image": {"url": image_url},
                    "color": 0x7289DA,  # Discord blurple
                }
            ],
        }

        return await self._bot_call(token, target, "POST", payload)

    async def post_with_poll(
        self,
        question: str,
        options: list[str],
        channel_id: str | None = None,
        duration_hours: int = 24,
    ) -> PostResult:
        """
        Create a poll in the configured channel.

        Args:
            question: Poll question (max 300 chars).
            options: List of poll options (2-10 options, max 55 chars each).
            channel_id: Discord channel ID; falls back to env var.
            duration_hours: Poll duration in hours (1-168, default 24).

        Returns:
            PostResult - never raises.
        """
        token, default_channel = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.DISCORD, "no_credentials")

        target = channel_id or default_channel
        if not target:
            return PostResult.fail(SocialPlatform.DISCORD, "no_channel_id")

        if len(question) > 300:
            question = question[:299] + "…"

        if not (2 <= len(options) <= 10):
            return PostResult.fail(
                SocialPlatform.DISCORD,
                "poll must have 2-10 options",
            )

        # Truncate options to 55 chars each
        options = [opt[:55] for opt in options]

        # Discord poll endpoint (v10 API)
        payload: dict[str, Any] = {
            "content": question,
            "poll": {
                "question": {"text": question},
                "answers": [{"text": opt} for opt in options],
                "duration": min(max(duration_hours, 1), 168),
                "allow_multiselect": False,
            },
        }

        return await self._bot_call(token, target, "POST", payload)

    # ── Internal helpers ───────────────────────────────────────────────────

    async def post_with_attachment(
        self,
        text: str,
        file_url: str,
        filename: str = "attachment",
        channel_id: str | None = None,
    ) -> PostResult:
        """
        Send a message with a file attachment from URL.

        Args:
            text: Message text (≤2,000 chars).
            file_url: Publicly accessible HTTPS URL of the file.
            filename: Name for the attachment (optional, derived from URL if missing).
            channel_id: Discord channel ID; falls back to env var.

        Returns:
            PostResult - never raises.
        """
        token, default_channel = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.DISCORD, "no_credentials")

        target = channel_id or default_channel
        if not target:
            return PostResult.fail(SocialPlatform.DISCORD, "no_channel_id")

        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[: _MAX_MESSAGE_CHARS - 1] + "…"

        # For file attachments from URL, Discord requires downloading and re-uploading.
        # This is a limitation of Discord's API - we'll use an embed link instead.
        # If you need actual file uploads, the bot would need to download from file_url first.
        payload: dict[str, Any] = {
            "content": text,
            "embeds": [
                {
                    "title": filename,
                    "url": file_url,
                    "color": 0x7289DA,
                    "description": f"[Download: {filename}]({file_url})",
                }
            ],
        }

        return await self._bot_call(token, target, "POST", payload)

    async def _bot_call(
        self,
        token: str,
        channel_id: str,
        method: str,
        payload: dict[str, Any],
    ) -> PostResult:
        """Dispatch a Discord Bot API call."""
        url = f"{_DISCORD_API_BASE}/channels/{channel_id}/messages"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.request(
                    method,
                    url,
                    json=payload,
                    headers={"Authorization": f"Bot {token}"},
                )

            data = _safe_json(resp)

            if resp.status_code == 200:
                message_id = str(data.get("id", ""))
                guild_id = data.get("guild_id", channel_id)
                post_url = (
                    f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
                    if message_id
                    else ""
                )
                self._logger.info("discord_channel_post_ok", message_id=message_id)
                return PostResult.ok(
                    platform=SocialPlatform.DISCORD,
                    post_id=message_id,
                    url=post_url,
                    http_status=200,
                    raw_response=data,
                )

            err_desc = data.get("message", resp.text[:200]) if isinstance(data, dict) else str(resp.text[:200])
            self._logger.warning(
                "discord_channel_post_failed",
                status=resp.status_code,
                description=err_desc,
            )
            return PostResult.fail(
                SocialPlatform.DISCORD,
                error=f"HTTP {resp.status_code}: {err_desc}",
                http_status=resp.status_code,
            )

        except Exception as exc:
            self._logger.error("discord_channel_post_exception", error=str(exc))
            return PostResult.fail(SocialPlatform.DISCORD, error=str(exc))

    def _load_credentials(self) -> tuple[str, str]:
        """
        Resolve (bot_token, channel_id): vault envelope → env vars.

        Returns:
            (bot_token, channel_id) - both empty strings on failure.
        """
        token = ""
        channel_id = ""

        if self._vault is not None and self._envelope is not None:
            with contextlib.suppress(Exception):
                token_set = self._vault.decrypt_token_set(self._envelope)
                token = token_set.get("access_token", "")
                channel_id = token_set.get("channel_id", "")

        # env var fallbacks
        if not token:
            token = os.getenv("ORGANISM_CONNECTORS__DISCORD__BOT_TOKEN", "")
        if not channel_id:
            channel_id = os.getenv("ORGANISM_DISCORD_CHANNEL_ID", "")

        return token, channel_id


# ── Helpers ────────────────────────────────────────────────────────────────


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    with contextlib.suppress(Exception):
        return resp.json()  # type: ignore[return-value]
    return {}
