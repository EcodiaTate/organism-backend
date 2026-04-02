"""
EcodiaOS - Social Interface: Telegram Channel Client

Posts messages and images to a public Telegram channel on behalf of the organism.

Auth: Bot token. The vault must hold a token envelope with purpose="oauth_token"
containing:
    {
        "access_token": "<bot_token>",
        "channel_id": "@mychannel"    # optional; falls back to ORGANISM_TELEGRAM_CHANNEL_ID
    }

Env var fallback:
    ORGANISM_TELEGRAM_CHANNEL_ID - public channel username (e.g. @ecodiaos) or numeric id

API:
    POST https://api.telegram.org/bot{token}/sendMessage
    POST https://api.telegram.org/bot{token}/sendPhoto

Char limits:
    - sendMessage: 4,096 chars (Markdown/HTML).
    - sendPhoto caption: 1,024 chars.

Note: This client targets CHANNEL posting (broadcast). For admin DMs use the
existing identity/connectors/telegram.py + axon/executors/send_telegram.py.
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

logger = structlog.get_logger("interfaces.social.telegram_channel")

_TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"
_MAX_MESSAGE_CHARS = 4_096
_MAX_CAPTION_CHARS = 1_024


class TelegramChannelClient:
    """
    Thin async HTTP client for Telegram Bot API - channel posting.

    Credentials resolved from vault first, env var second.
    """

    def __init__(
        self,
        vault: IdentityVault | None = None,
        envelope: SealedEnvelope | None = None,
    ) -> None:
        self._vault = vault
        self._envelope = envelope
        self._logger = logger.bind(client="telegram_channel")

    # ── Public API ─────────────────────────────────────────────────────────

    async def post_to_channel(
        self,
        text: str,
        channel_id: str | None = None,
        parse_mode: str = "Markdown",
        disable_web_page_preview: bool = False,
    ) -> PostResult:
        """
        Send a text message to the configured channel.

        Args:
            text: Message text (≤4,096 chars, including disclaimer).
            channel_id: Channel username (@handle) or numeric ID.
                        Falls back to ORGANISM_TELEGRAM_CHANNEL_ID env var.
            parse_mode: "Markdown" | "HTML" | "MarkdownV2" (default "Markdown").
            disable_web_page_preview: Suppress link previews.

        Returns:
            PostResult - never raises.
        """
        token, default_channel = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.TELEGRAM_CHANNEL, "no_credentials")

        target = channel_id or default_channel
        if not target:
            return PostResult.fail(SocialPlatform.TELEGRAM_CHANNEL, "no_channel_id")

        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[:_MAX_MESSAGE_CHARS - 1] + "…"

        params: dict[str, Any] = {
            "chat_id": target,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }

        return await self._bot_call(token, "sendMessage", params)

    async def post_with_image(
        self,
        image_url: str,
        caption: str = "",
        channel_id: str | None = None,
        parse_mode: str = "Markdown",
    ) -> PostResult:
        """
        Send a photo from URL with optional caption to the configured channel.

        Args:
            image_url: Publicly accessible HTTPS URL of the image.
            caption: Caption text (≤1,024 chars).
            channel_id: Channel username or numeric ID; falls back to env var.
            parse_mode: "Markdown" | "HTML" | "MarkdownV2".

        Returns:
            PostResult - never raises.
        """
        token, default_channel = self._load_credentials()
        if not token:
            return PostResult.fail(SocialPlatform.TELEGRAM_CHANNEL, "no_credentials")

        target = channel_id or default_channel
        if not target:
            return PostResult.fail(SocialPlatform.TELEGRAM_CHANNEL, "no_channel_id")

        if len(caption) > _MAX_CAPTION_CHARS:
            caption = caption[:_MAX_CAPTION_CHARS - 1] + "…"

        params = {
            "chat_id": target,
            "photo": image_url,
            "caption": caption,
            "parse_mode": parse_mode,
        }

        return await self._bot_call(token, "sendPhoto", params)

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _bot_call(
        self,
        token: str,
        method: str,
        params: dict[str, Any],
    ) -> PostResult:
        """Dispatch a Telegram Bot API method call."""
        url = f"https://api.telegram.org/bot{token}/{method}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=params)

            data = _safe_json(resp)

            if resp.status_code == 200 and data.get("ok"):
                result = data.get("result", {})
                message_id = str(result.get("message_id", ""))
                # Telegram doesn't return a direct public URL; construct best-effort.
                channel_raw = params.get("chat_id", "")
                channel_str = str(channel_raw).lstrip("@")
                post_url = (
                    f"https://t.me/{channel_str}/{message_id}"
                    if message_id and not channel_str.startswith("-")
                    else ""
                )
                self._logger.info("telegram_channel_post_ok", message_id=message_id)
                return PostResult.ok(
                    platform=SocialPlatform.TELEGRAM_CHANNEL,
                    post_id=message_id,
                    url=post_url,
                    http_status=200,
                    raw_response=result,
                )

            err_desc = data.get("description", resp.text[:200])
            self._logger.warning(
                "telegram_channel_post_failed",
                status=resp.status_code,
                description=err_desc,
            )
            return PostResult.fail(
                SocialPlatform.TELEGRAM_CHANNEL,
                error=f"HTTP {resp.status_code}: {err_desc}",
                http_status=resp.status_code,
            )

        except Exception as exc:
            self._logger.error("telegram_channel_post_exception", error=str(exc))
            return PostResult.fail(SocialPlatform.TELEGRAM_CHANNEL, error=str(exc))

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
            token = os.getenv("ORGANISM_CONNECTORS__TELEGRAM__BOT_TOKEN", "")
        if not channel_id:
            channel_id = os.getenv("ORGANISM_TELEGRAM_CHANNEL_ID", "")

        return token, channel_id


# ── Helpers ────────────────────────────────────────────────────────────────


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    with contextlib.suppress(Exception):
        return resp.json()  # type: ignore[return-value]
    return {}
