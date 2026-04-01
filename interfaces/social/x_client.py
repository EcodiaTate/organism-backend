"""
EcodiaOS - Social Interface: X (Twitter) Client

Posts tweets on behalf of the organism using X API v2.

Auth: X API v2 tweet creation requires OAuth 1.0a *user context* (not a
bearer token). The vault must hold a token envelope with purpose="oauth_token"
containing the fields:
    {
        "oauth_token": "<user access token>",
        "oauth_token_secret": "<user access token secret>",
        "consumer_key": "<app API key>",
        "consumer_secret": "<app API key secret>"
    }

OAuth 1.0a signatures are computed by hand (HMAC-SHA1) so we avoid pulling
in the `tweepy` dependency - httpx is already in the project.

X character limits:
    - Standard tweet: 280 characters (including the disclaimer).
    - If the raw content + disclaimer would exceed 280 chars, content is
      truncated at the truncation boundary and an ellipsis is appended
      before the disclaimer.

Rate limits (free/basic tier): 17 tweets per 24 hours per user.
The executor enforces a conservative rate_limit; this client is a thin HTTP
wrapper and does not perform its own rate-limiting.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import secrets
import time
import urllib.parse
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from interfaces.social.types import PostResult, SocialPlatform

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault, SealedEnvelope

logger = structlog.get_logger("interfaces.social.x")

_TWEETS_URL = "https://api.twitter.com/2/tweets"
_TWEET_HARD_LIMIT = 280
# Leave room for the disclaimer; content is truncated if needed.
_ELLIPSIS = "…"


def _percent_encode(value: str) -> str:
    """RFC 5849 §3.6 percent-encoding."""
    return urllib.parse.quote(value, safe="")


def _oauth1_authorization_header(
    method: str,
    url: str,
    payload: dict[str, Any],
    consumer_key: str,
    consumer_secret: str,
    token: str,
    token_secret: str,
) -> str:
    """
    Construct an OAuth 1.0a Authorization header for a JSON-body POST.

    X's v2 JSON endpoints sign only the OAuth parameters (not the JSON body)
    because the body is not form-encoded.  Per RFC 5849 §3.4.1, the base
    string is built from the HTTP method, base URL, and *only* the OAuth
    header params.
    """
    oauth_params: dict[str, str] = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": token,
        "oauth_version": "1.0",
    }

    # Base string - OAuth params only (no JSON body params for JSON endpoints)
    sorted_params = sorted(oauth_params.items())
    param_string = "&".join(
        f"{_percent_encode(k)}={_percent_encode(v)}" for k, v in sorted_params
    )
    base_string = (
        _percent_encode(method.upper())
        + "&"
        + _percent_encode(url)
        + "&"
        + _percent_encode(param_string)
    )

    # Signing key: consumer_secret & token_secret (both percent-encoded)
    signing_key = f"{_percent_encode(consumer_secret)}&{_percent_encode(token_secret)}"
    raw_signature = hmac.new(
        signing_key.encode("ascii"),
        base_string.encode("ascii"),
        hashlib.sha1,
    ).digest()
    signature = base64.b64encode(raw_signature).decode("ascii")
    oauth_params["oauth_signature"] = signature

    # Build the Authorization header value
    header_parts = ", ".join(
        f'{_percent_encode(k)}="{_percent_encode(v)}"'
        for k, v in sorted(oauth_params.items())
    )
    return f"OAuth {header_parts}"


class XSocialClient:
    """
    Thin async client for posting tweets via X API v2 with OAuth 1.0a.

    Intended to be constructed per-execution; the vault reference is
    required so credentials are never passed in plaintext.

    Usage::

        client = XSocialClient(vault=vault, envelope=sealed_envelope)
        result = await client.post_tweet("Hello, world! 🤖")
    """

    def __init__(
        self,
        vault: IdentityVault,
        envelope: SealedEnvelope,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._vault = vault
        self._envelope = envelope
        self._http = http_client or httpx.AsyncClient(timeout=15.0)
        self._logger = logger.bind(platform="x")

    async def post_tweet(self, text: str) -> PostResult:
        """
        Post a tweet.  Returns PostResult - never raises.

        Args:
            text: Final tweet text (disclaimer already injected by executor).
        """
        creds = self._load_credentials()
        if creds is None:
            return PostResult.fail(
                SocialPlatform.X,
                error="Vault envelope missing required OAuth 1.0a fields",
            )

        consumer_key, consumer_secret, oauth_token, oauth_token_secret = creds
        payload: dict[str, Any] = {"text": text}

        auth_header = _oauth1_authorization_header(
            method="POST",
            url=_TWEETS_URL,
            payload=payload,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            token=oauth_token,
            token_secret=oauth_token_secret,
        )
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

        try:
            resp = await self._http.post(_TWEETS_URL, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            self._logger.error("x_post_transport_error", error=str(exc))
            return PostResult.fail(SocialPlatform.X, error=f"Transport error: {exc}")

        if resp.status_code not in (200, 201):
            self._logger.error(
                "x_post_api_error",
                status=resp.status_code,
                body_preview=resp.text[:200],
            )
            return PostResult.fail(
                SocialPlatform.X,
                error=f"X API error HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )

        resp_data: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            resp_data = resp.json()

        tweet_id = str(resp_data.get("data", {}).get("id", ""))
        tweet_url = f"https://x.com/i/web/status/{tweet_id}" if tweet_id else ""

        self._logger.info("x_post_success", tweet_id=tweet_id)
        return PostResult.ok(
            platform=SocialPlatform.X,
            post_id=tweet_id,
            url=tweet_url,
            http_status=resp.status_code,
            raw_response=resp_data,
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _load_credentials(
        self,
    ) -> tuple[str, str, str, str] | None:
        """
        Decrypt the envelope and extract the four OAuth 1.0a fields.
        Returns (consumer_key, consumer_secret, oauth_token, oauth_token_secret)
        or None if any field is missing.
        """
        try:
            token_data = self._vault.decrypt_token_json(self._envelope)
        except Exception as exc:
            self._logger.error("x_vault_decrypt_failed", error=str(exc))
            return None

        consumer_key = token_data.get("consumer_key", "")
        consumer_secret = token_data.get("consumer_secret", "")
        oauth_token = token_data.get("oauth_token", "")
        oauth_token_secret = token_data.get("oauth_token_secret", "")

        if not all([consumer_key, consumer_secret, oauth_token, oauth_token_secret]):
            self._logger.warning(
                "x_credentials_incomplete",
                has_consumer_key=bool(consumer_key),
                has_consumer_secret=bool(consumer_secret),
                has_oauth_token=bool(oauth_token),
                has_oauth_token_secret=bool(oauth_token_secret),
            )
            return None

        return consumer_key, consumer_secret, oauth_token, oauth_token_secret


def truncate_for_x(content: str, disclaimer: str) -> str:
    """
    Fit content + " " + disclaimer within the X character limit.

    If content alone would overflow, it is truncated and an ellipsis
    appended so the disclaimer always appears verbatim.
    """
    separator = " "
    full = content + separator + disclaimer
    if len(full) <= _TWEET_HARD_LIMIT:
        return full

    # Calculate maximum content length
    max_content_len = _TWEET_HARD_LIMIT - len(separator) - len(disclaimer) - len(_ELLIPSIS)
    if max_content_len <= 0:
        # Edge case: disclaimer itself is near the limit.
        return (disclaimer[:_TWEET_HARD_LIMIT])

    return content[:max_content_len] + _ELLIPSIS + separator + disclaimer
