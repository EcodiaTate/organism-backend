"""
EcodiaOS -- Identity Communication Layer (Phase 16h: External Identity Layer)

Receives inbound verification messages from Twilio (SMS) and optionally IMAP
(email), extracts 2FA/OTP codes, and publishes them on the Synapse event bus
so the Axon executor that initiated the authentication flow can complete it.

Security notes:
  - Every inbound Twilio webhook is validated via X-Twilio-Signature (HMAC-SHA1)
    before any payload is processed. Invalid requests are rejected HTTP 403.
  - Body parsing MUST use urllib.parse.parse_qs(raw_body, keep_blank_values=True)
    so empty form fields are included in the HMAC computation.
    Omitting keep_blank_values=True causes signature mismatches on messages with
    fields like NumMedia whose value is an empty string.
  - The Twilio auth token is from config.identity_comm.twilio_auth_token
    (env: ORGANISM_IDENTITY_COMM__TWILIO_AUTH_TOKEN) and is never logged.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import imaplib
import re
import urllib.parse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from config import EcodiaOSConfig, IdentityCommConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.communication")

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

identity_comm_router = APIRouter(tags=["identity-comm"])

# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

# Standard 6-digit numeric OTP (Google, Apple, banks, Twilio Verify ...)
_PATTERN_NUMERIC = re.compile(r"\b(\d{6})\b")

# Alphanumeric codes: 4-8 uppercase chars containing at least one letter AND
# one digit (avoids false-positives on plain words or plain numbers).
_PATTERN_ALPHANUM = re.compile(r"\b([A-Z0-9]{4,8})\b")


def _validate_twilio_signature(
    auth_token: str,
    url: str,
    raw_body: bytes,
    signature_header: str,
) -> bool:
    """
    Validate X-Twilio-Signature per Twilio HMAC-SHA1 spec.

    Steps:
      1. Start with the full request URL.
      2. Sort POST parameters alphabetically.
      3. Append each param key+value (no separator) to the URL string.
      4. Sign with HMAC-SHA1 keyed on the auth token.
      5. Base64-encode and compare_digest against the header.

    CRITICAL: keep_blank_values=True is mandatory. Twilio includes ALL form
    fields in its signature -- including those with empty string values.
    Stripping empty fields silently breaks signature validation.
    """
    params: dict[str, list[str]] = urllib.parse.parse_qs(
        raw_body.decode("utf-8", errors="replace"),
        keep_blank_values=True,
    )

    signing_string = url
    for key in sorted(params.keys()):
        signing_string += key + (params[key][0] if params[key] else "")

    mac = hmac.new(
        auth_token.encode("utf-8"),
        signing_string.encode("utf-8"),
        hashlib.sha1,
    )
    expected = base64.b64encode(mac.digest()).decode("utf-8")
    return hmac.compare_digest(expected, signature_header)


def _extract_verification_code(body: str) -> str | None:
    """
    Extract the first verification code from an SMS or email body.

    Priority:
      1. 6-digit numeric OTP (most common).
      2. Alphanumeric code (4-8 chars, mixed letters+digits).

    Returns None when no recognisable code is present.
    """
    m = _PATTERN_NUMERIC.search(body)
    if m:
        return m.group(1)

    for m in _PATTERN_ALPHANUM.finditer(body.upper()):
        candidate = m.group(1)
        if any(c.isalpha() for c in candidate) and any(c.isdigit() for c in candidate):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Outbound SMS (HITL notifications)
# ---------------------------------------------------------------------------

_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"


# ---------------------------------------------------------------------------
# FastAPI dependency helpers
# ---------------------------------------------------------------------------


def _get_event_bus(request: Request) -> EventBus:
    bus = getattr(request.app.state, "event_bus", None)
    if bus is None:
        raise HTTPException(status_code=503, detail="event bus unavailable")
    return bus  # type: ignore[return-value]


def _get_config(request: Request) -> EcodiaOSConfig:
    cfg = getattr(request.app.state, "config", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="config unavailable")
    return cfg  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Twilio inbound SMS endpoint
# ---------------------------------------------------------------------------


@identity_comm_router.post("/api/v1/identity/webhook/twilio")
async def twilio_inbound_sms(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Receive an inbound Twilio SMS webhook.

    1. Validate X-Twilio-Signature -- HTTP 403 on failure.
    2. Parse TwiML form payload for From and Body fields.
    3. Extract a 6-digit or alphanumeric verification code from Body.
    4. Publish IDENTITY_VERIFICATION_RECEIVED on the Synapse event bus.

    Returns HTTP 200 immediately; publication runs in a background task
    so we never block Twilio waiting for Redis I/O.
    """
    cfg = _get_config(request)
    event_bus = _get_event_bus(request)

    auth_token = cfg.identity_comm.twilio_auth_token
    if not auth_token:
        logger.warning("twilio_webhook_rejected", reason="auth_token_not_configured")
        raise HTTPException(status_code=503, detail="Twilio auth token not configured")

    raw_body = await request.body()
    url = str(request.url)

    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        logger.warning("twilio_webhook_rejected", reason="missing_signature")
        raise HTTPException(status_code=403, detail="Missing X-Twilio-Signature")

    if not _validate_twilio_signature(auth_token, url, raw_body, signature):
        logger.warning("twilio_webhook_rejected", reason="invalid_signature", url=url)
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # Re-parse with keep_blank_values=True (same flag used in HMAC check above).
    params: dict[str, list[str]] = urllib.parse.parse_qs(
        raw_body.decode("utf-8", errors="replace"),
        keep_blank_values=True,
    )

    def _first(key: str) -> str:
        vals = params.get(key, [])
        return vals[0].strip() if vals else ""

    from_number = _first("From")
    body_text = _first("Body")
    log = logger.bind(from_number=from_number)

    # ── Admin-phone guard ────────────────────────────────────────
    # Only the configured admin number may drive HITL flows.
    # Messages from any other number are silently dropped after logging.
    admin_phone = cfg.identity_comm.admin_phone_number
    if admin_phone and from_number != admin_phone:
        log.warning(
            "twilio_webhook_unauthorized_sender",
            expected=admin_phone,
            received=from_number,
        )
        return {"status": "ok", "action": "unauthorized_sender_dropped"}

    if not body_text:
        log.info("twilio_webhook_empty_body")
        return {"status": "ok", "action": "ignored_empty"}

    # Try to extract a standard OTP code.  For HITL auth replies (e.g. "AUTH
    # 4821") there is no 6-digit OTP, but we still publish so Equor can apply
    # its own AUTH regex.  Always publish when the sender is the admin.
    code = _extract_verification_code(body_text)
    if code is None and not admin_phone:
        # No admin configured: require a recognisable OTP code.
        log.info("twilio_webhook_no_code_found", body_preview=body_text[:80])
        return {"status": "ok", "action": "no_code_found"}

    log.info(
        "twilio_verification_received",
        source=from_number,
        code_length=len(code) if code else 0,
        is_admin=bool(admin_phone),
    )

    async def _publish() -> None:
        event = SynapseEvent(
            event_type=SynapseEventType.IDENTITY_VERIFICATION_RECEIVED,
            source_system="identity",
            data={
                "source": from_number,
                "code": code or "",
                "channel": "sms",
                "raw_body": body_text,
            },
        )
        try:
            await event_bus.emit(event)
        except Exception as exc:
            logger.error("twilio_event_publish_failed", error=str(exc))

    background_tasks.add_task(_publish)
    return {"status": "ok", "action": "event_published"}


# ---------------------------------------------------------------------------
# IMAP background task - class wrapper for supervised_task scheduling
# ---------------------------------------------------------------------------


def _extract_email_subject(raw_email: str) -> str:
    """Extract the Subject header value from a raw RFC 822 email string."""
    for line in raw_email.splitlines():
        if line.lower().startswith("subject:"):
            return line[8:].strip()
    return ""


async def _do_imap_scan(comm: "IdentityCommConfig", event_bus: "EventBus") -> None:
    """
    Perform a single IMAP inbox scan for verification emails.

    For each unseen message containing a code, publishes both:
      - IDENTITY_VERIFICATION_RECEIVED  (generic; existing 2FA flow subscribers)
      - EMAIL_OTP_RECEIVED              (channel-specific; email-only handlers)

    imaplib is synchronous; runs in asyncio.to_thread() to avoid blocking the loop.
    """
    if not comm.imap_host or not comm.imap_username:
        logger.debug("imap_scan_skipped", reason="not_configured")
        return

    log = logger.bind(imap_host=comm.imap_host, mailbox=comm.imap_mailbox)
    log.info("imap_scan_starting")

    def _sync_scan() -> list[dict[str, str]]:
        """Run synchronous IMAP operations; returns list of {sender, code, subject}."""
        results: list[dict[str, str]] = []
        conn = imaplib.IMAP4_SSL(comm.imap_host, comm.imap_port)
        try:
            conn.login(comm.imap_username, comm.imap_password)
            conn.select(comm.imap_mailbox)
            _status, id_data = conn.search(None, "UNSEEN")
            message_ids: list[bytes] = id_data[0].split() if id_data[0] else []
            for msg_id in message_ids:
                try:
                    _fetch_status, msg_data = conn.fetch(msg_id, "(RFC822)")
                    if not msg_data or not msg_data[0]:
                        continue
                    raw_email: bytes = (
                        msg_data[0][1] if isinstance(msg_data[0], tuple) else b""
                    )
                    if not raw_email:
                        continue
                    email_text = raw_email.decode("utf-8", errors="replace")
                    code = _extract_verification_code(email_text)
                    if code is None:
                        continue
                    sender = _extract_email_sender(email_text)
                    subject = _extract_email_subject(email_text)
                    results.append({"sender": sender, "code": code, "subject": subject})
                except Exception as msg_exc:
                    logger.warning(
                        "imap_message_parse_failed",
                        msg_id=str(msg_id),
                        error=str(msg_exc),
                    )
        finally:
            try:
                conn.logout()
            except Exception:
                pass
        return results

    try:
        import asyncio as _asyncio
        found = await _asyncio.to_thread(_sync_scan)
        log.info("imap_scan_found_messages", count=len(found))
        for item in found:
            sender = item["sender"]
            code = item["code"]
            subject = item.get("subject", "")
            log.info("imap_verification_code_found", sender=sender, code_length=len(code))
            # Generic event - existing 2FA flow subscribers listen here
            await event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.IDENTITY_VERIFICATION_RECEIVED,
                source_system="identity",
                data={"source": sender, "code": code, "channel": "email"},
            ))
            # Channel-specific event - cleaner filtering without payload inspection
            await event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EMAIL_OTP_RECEIVED,
                source_system="identity",
                data={
                    "source": sender,
                    "code": code,
                    "channel": "email",
                    "raw_subject": subject,
                    "raw_from": sender,
                },
            ))
        log.info("imap_scan_complete")

    except imaplib.IMAP4.error as imap_exc:
        log.error("imap_connection_failed", error=str(imap_exc))
    except Exception as exc:
        log.error("imap_scan_failed", error=str(exc))


async def scan_imap_inbox(
    cfg: "EcodiaOSConfig",
    event_bus: "EventBus",
) -> None:
    """Backward-compatible single-shot IMAP scan (delegates to _do_imap_scan)."""
    await _do_imap_scan(cfg.identity_comm, event_bus)


class IMAPScanner:
    """
    Supervised background task that periodically scans the IMAP inbox.

    Wire into ``core/registry.py`` Phase 11::

        imap_scanner = IMAPScanner(config=config, event_bus=synapse.event_bus)
        supervised_task("imap_scanner", imap_scanner.run())

    The loop: sleep → scan → repeat.  Scan errors are logged; the loop continues.
    If IMAP is not configured (no host/username), the coroutine exits immediately
    and supervised_task will not restart it because it returns normally.
    """

    def __init__(self, config: "EcodiaOSConfig", event_bus: "EventBus") -> None:
        self._comm = config.identity_comm
        self._event_bus = event_bus
        self._interval_s: float = getattr(config.identity_comm, "imap_scan_interval_s", 60.0)

    async def run(self) -> None:
        """Infinite scan loop.  Exits only when cancelled or IMAP is unconfigured."""
        import asyncio as _asyncio

        if not self._comm.imap_host or not self._comm.imap_username:
            logger.info(
                "imap_scanner_not_started",
                reason="ORGANISM_IDENTITY_COMM__IMAP_HOST or __IMAP_USERNAME not set",
            )
            return

        logger.info(
            "imap_scanner_started",
            imap_host=self._comm.imap_host,
            interval_s=self._interval_s,
        )
        while True:
            await _asyncio.sleep(self._interval_s)
            try:
                await _do_imap_scan(self._comm, self._event_bus)
            except Exception as exc:
                logger.error("imap_scanner_cycle_failed", error=str(exc))


def _extract_email_sender(raw_email: str) -> str:
    """Extract the From header value from a raw RFC 822 email string."""
    for line in raw_email.splitlines():
        if line.lower().startswith("from:"):
            return line[5:].strip()
    return "unknown"


# ---------------------------------------------------------------------------
# Outbound SMS helper
# ---------------------------------------------------------------------------


async def send_admin_sms(comm: IdentityCommConfig, message: str) -> None:
    """
    Send an outbound SMS to the configured admin phone number via Twilio REST API.

    Requires comm.twilio_account_sid, comm.twilio_auth_token, and
    comm.twilio_from_number to be populated, and comm.admin_phone_number to
    be the destination.  Logs a warning and returns silently if any credential
    is missing so callers never raise on misconfiguration.
    """
    missing = [
        f for f in ("twilio_account_sid", "twilio_auth_token", "twilio_from_number", "admin_phone_number")
        if not getattr(comm, f, "")
    ]
    if missing:
        logger.warning(
            "send_admin_sms_skipped",
            reason="missing_credentials",
            fields=missing,
        )
        return

    url = (
        f"https://api.twilio.com/2010-04-01/Accounts/"
        f"{comm.twilio_account_sid}/Messages.json"
    )
    payload = {
        "To": comm.admin_phone_number,
        "From": comm.twilio_from_number,
        "Body": message,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                data=payload,
                auth=(comm.twilio_account_sid, comm.twilio_auth_token),
            )
        if resp.status_code not in (200, 201):
            logger.error(
                "send_admin_sms_failed",
                status=resp.status_code,
                body=resp.text[:200],
            )
        else:
            logger.info("send_admin_sms_sent", to=comm.admin_phone_number)
    except Exception as exc:
        logger.error("send_admin_sms_exception", error=str(exc))


# ---------------------------------------------------------------------------
# Typed payload helper for event subscribers
# ---------------------------------------------------------------------------


class VerificationPayload:
    """
    Typed view of an IDENTITY_VERIFICATION_RECEIVED event data dict.

    Usage::

        p = VerificationPayload.from_event(event.data)
        await submit_otp(p.code)
    """

    __slots__ = ("source", "code", "channel", "raw_body")

    def __init__(
        self,
        source: str,
        code: str,
        channel: str = "sms",
        raw_body: str = "",
    ) -> None:
        self.source = source
        self.code = code
        self.channel = channel
        self.raw_body = raw_body

    @classmethod
    def from_event(cls, data: dict[str, Any]) -> VerificationPayload:
        return cls(
            source=data.get("source", ""),
            code=data.get("code", ""),
            channel=data.get("channel", "sms"),
            raw_body=data.get("raw_body", ""),
        )


# ---------------------------------------------------------------------------
# Autonomous phone-number provisioning
# ---------------------------------------------------------------------------

_TWILIO_ERROR_CODES_INSUFFICIENT_FUNDS = {20003, 20005, 21211, 21606}
_TWILIO_ERROR_CODES_BAD_CREDENTIALS = {20003, 20008}

# Twilio country codes and their Twilio API country identifiers
_TWILIO_COUNTRY_CODES: dict[str, str] = {
    "US": "US",
    "AU": "AU",
    "GB": "GB",
    "CA": "CA",
    "NZ": "NZ",
    "IE": "IE",
    "ZA": "ZA",
    "IN": "IN",
    "SG": "SG",
    "DE": "DE",
    "FR": "FR",
    "JP": "JP",
    "BR": "BR",
    "MX": "MX",
    "SE": "SE",
    "NO": "NO",
    "FI": "FI",
    "DK": "DK",
    "NL": "NL",
    "BE": "BE",
    "CH": "CH",
    "AT": "AT",
    "IT": "IT",
    "ES": "ES",
    "PT": "PT",
    "PL": "PL",
    "CZ": "CZ",
}

# Area code patterns by country: digit count range (min, max)
_AREA_CODE_LENGTHS: dict[str, tuple[int, int]] = {
    "US": (3, 3),   # 3-digit NANP area codes
    "CA": (3, 3),   # 3-digit NANP area codes
    "AU": (1, 2),   # 1-digit (mobile prefix 4) or 2-digit (02, 03, 07, 08)
    "GB": (2, 5),   # UK area codes vary 2-5 digits
    "NZ": (1, 2),   # NZ area codes 1-2 digits
    "IE": (1, 3),   # Ireland 1-3 digits
}
_DEFAULT_AREA_CODE_LENGTHS: tuple[int, int] = (1, 6)


async def provision_new_phone_number(
    config: IdentityCommConfig,
    area_code: str = "415",
    country: str = "US",
) -> str:
    """
    Autonomously search for and purchase a Local phone number via Twilio.

    Supports any country that Twilio offers Local numbers in.  Pass the
    ISO 3166-1 alpha-2 country code (e.g. "US", "AU", "GB") in ``country``.

    Steps:
      1. Search AvailablePhoneNumbers/{country}/Local for a number matching
         ``area_code`` (semantics vary by country).
      2. Purchase the first result by POSTing to IncomingPhoneNumbers.
      3. Return the acquired number in E.164 format.

    Raises:
      RuntimeError: credentials not configured, unsupported country,
                    no numbers available, insufficient Twilio balance,
                    or any unexpected API error.
    """
    account_sid = config.twilio_account_sid
    auth_token = config.twilio_auth_token

    missing = [f for f, v in (("twilio_account_sid", account_sid), ("twilio_auth_token", auth_token)) if not v]
    if missing:
        raise RuntimeError(
            f"Twilio credentials not configured: {missing}. "
            "Set ORGANISM_IDENTITY_COMM__TWILIO_ACCOUNT_SID and __TWILIO_AUTH_TOKEN."
        )

    country_upper = country.upper().strip()
    twilio_country = _TWILIO_COUNTRY_CODES.get(country_upper)
    if twilio_country is None:
        raise RuntimeError(
            f"Country '{country}' is not in the supported list. "
            f"Supported: {sorted(_TWILIO_COUNTRY_CODES)}. "
            "You can manually provision a Twilio number for this country and set "
            "ORGANISM_IDENTITY_COMM__TWILIO_FROM_NUMBER directly."
        )

    log = logger.bind(area_code=area_code, country=twilio_country, account_sid=account_sid[:8] + "…")
    auth = (account_sid, auth_token)

    # Build search params - area code semantics differ by country
    search_params: dict[str, Any] = {"Limit": 1}
    if area_code:
        # AU uses AreaCode for 2-digit prefixes (02, 03, 07, 08) or InRegion.
        # Most Twilio countries accept AreaCode as a prefix filter; pass it
        # through and let Twilio return what's available.
        search_params["AreaCode"] = area_code

    async with httpx.AsyncClient(timeout=15.0) as client:
        # ── Step 1: search for available numbers ──────────────────────────
        search_url = (
            f"{_TWILIO_API_BASE}/Accounts/{account_sid}"
            f"/AvailablePhoneNumbers/{twilio_country}/Local.json"
        )
        search_resp = await client.get(search_url, params=search_params, auth=auth)

        if search_resp.status_code == 401:
            raise RuntimeError(
                "Twilio authentication failed during number search (HTTP 401). "
                "Verify TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."
            )
        if search_resp.status_code == 400:
            # Twilio rejects invalid area codes for the country - retry without it
            log.warning(
                "twilio_area_code_rejected_by_api",
                area_code=area_code,
                country=twilio_country,
                body=search_resp.text[:200],
            )
            search_resp = await client.get(
                search_url,
                params={"Limit": 1},
                auth=auth,
            )
        if search_resp.status_code != 200:
            _raise_twilio_error("number_search_failed", search_resp)

        search_data: dict[str, Any] = search_resp.json()
        available: list[dict[str, Any]] = search_data.get("available_phone_numbers", [])

        if not available:
            log.warning("twilio_no_numbers_available", area_code=area_code, country=twilio_country)
            raise RuntimeError(
                f"No Local numbers available for country={twilio_country}, "
                f"area_code={area_code!r}. Try a different area code or omit it."
            )

        candidate: str = available[0]["phone_number"]
        log.info("twilio_number_candidate_found", candidate=candidate, country=twilio_country)

        # ── Step 2: purchase the number ───────────────────────────────────
        purchase_url = (
            f"{_TWILIO_API_BASE}/Accounts/{account_sid}/IncomingPhoneNumbers.json"
        )
        purchase_resp = await client.post(
            purchase_url,
            data={"PhoneNumber": candidate},
            auth=auth,
        )

        if purchase_resp.status_code == 401:
            raise RuntimeError(
                "Twilio authentication failed during number purchase (HTTP 401). "
                "Verify TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."
            )
        if purchase_resp.status_code not in (200, 201):
            _raise_twilio_error("number_purchase_failed", purchase_resp)

        purchased: dict[str, Any] = purchase_resp.json()
        acquired_number: str = purchased["phone_number"]
        number_sid: str = purchased.get("sid", "")

        # ── Step 3: configure webhook URLs on the purchased number ────────
        # If webhook_base_url is set, point Twilio at this instance's SMS
        # endpoint so inbound messages arrive immediately without manual
        # Twilio Console configuration.
        webhook_base = getattr(config, "webhook_base_url", "").rstrip("/")
        if webhook_base and number_sid:
            sms_url = f"{webhook_base}/api/v1/identity/webhook/twilio"
            update_url = (
                f"{_TWILIO_API_BASE}/Accounts/{account_sid}"
                f"/IncomingPhoneNumbers/{number_sid}.json"
            )
            update_resp = await client.post(
                update_url,
                data={
                    "SmsUrl": sms_url,
                    "SmsMethod": "POST",
                    "SmsFallbackUrl": "",
                    "SmsFallbackMethod": "POST",
                },
                auth=auth,
            )
            if update_resp.status_code in (200, 201):
                log.info(
                    "twilio_number_webhook_configured",
                    number_sid=number_sid,
                    sms_url=sms_url,
                )
            else:
                # Non-fatal: number is live, but operator must set webhook manually
                log.warning(
                    "twilio_number_webhook_config_failed",
                    number_sid=number_sid,
                    sms_url=sms_url,
                    status=update_resp.status_code,
                    body=update_resp.text[:200],
                )
        elif not webhook_base:
            log.warning(
                "twilio_number_webhook_not_configured",
                reason="ORGANISM_IDENTITY_COMM__WEBHOOK_BASE_URL not set",
                action="set it or configure Twilio webhook manually in Console",
                sms_url_should_be="<your-public-url>/api/v1/identity/webhook/twilio",
            )

    logger.info(
        "twilio_number_provisioned",
        number=acquired_number,
        country=twilio_country,
        friendly_name=purchased.get("friendly_name", ""),
        sid=number_sid,
    )
    return acquired_number


# ---------------------------------------------------------------------------
# OTP coordination layer
# ---------------------------------------------------------------------------

# Map platform names to hints found in message text (case-insensitive).
PLATFORM_HINTS: dict[str, list[str]] = {
    "github": ["github", "git hub"],
    "google": ["google", "gmail"],
    "twitter": ["twitter", "x.com"],
    "telegram": ["telegram"],
    "coinbase": ["coinbase", "cbp"],
    "anthropic": ["anthropic", "claude"],
    "stripe": ["stripe"],
    "aws": ["amazon web services", "aws", "amazon"],
    "microsoft": ["microsoft", "azure", "outlook"],
    "apple": ["apple", "icloud"],
    "linkedin": ["linkedin"],
    "instagram": ["instagram"],
    "discord": ["discord"],
    "dropbox": ["dropbox"],
    "slack": ["slack"],
}


def extract_platform_hint(message: str) -> str | None:
    """
    Scan a message string for known platform names and return the first match.

    Returns None when no known platform is detected.
    """
    lower = message.lower()
    for platform, hints in PLATFORM_HINTS.items():
        if any(h in lower for h in hints):
            return platform
    return None


class _PendingOTPFlow:
    """State for a single waiting OTP flow."""

    __slots__ = ("platform", "expected_source", "future", "created_at", "timeout_seconds")

    def __init__(
        self,
        platform: str,
        expected_source: str,
        future: asyncio.Future[str],
        timeout_seconds: int,
    ) -> None:
        self.platform = platform
        self.expected_source = expected_source  # "sms" | "telegram" | "email" | "any"
        self.future = future
        self.created_at: datetime = datetime.now(timezone.utc)
        self.timeout_seconds = timeout_seconds


class OTPCoordinator:
    """
    Unified OTP coordination layer.

    Receives verification codes from any channel (SMS/Twilio, Telegram, Email)
    and resolves the matching pending authentication flow.

    Usage::

        coordinator = OTPCoordinator()
        coordinator.set_event_bus(bus)

        # In an authentication flow:
        code = await coordinator.wait_for_otp("github", source="email", timeout=300)

    Subscriptions (set up via set_event_bus):
        IDENTITY_VERIFICATION_RECEIVED  - Twilio SMS path (existing)
        TELEGRAM_OTP_RECEIVED           - Telegram bot path (I-1)
        EMAIL_OTP_RECEIVED              - IMAP email path (I-2)
    """

    def __init__(self) -> None:
        self._pending: dict[str, _PendingOTPFlow] = {}
        self._event_bus: Any = None
        self._log = structlog.get_logger("identity.otp_coordinator")

    def set_event_bus(self, event_bus: Any) -> None:
        """Wire the Synapse EventBus and register channel subscriptions."""
        self._event_bus = event_bus
        event_bus.subscribe(
            SynapseEventType.IDENTITY_VERIFICATION_RECEIVED,
            self._on_sms_otp,
        )
        event_bus.subscribe(
            SynapseEventType.TELEGRAM_OTP_RECEIVED,
            self._on_telegram_otp,
        )
        event_bus.subscribe(
            SynapseEventType.EMAIL_OTP_RECEIVED,
            self._on_email_otp,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    async def wait_for_otp(
        self,
        platform: str,
        source: str = "any",
        timeout: int = 300,
    ) -> str:
        """
        Register a pending OTP flow and block until a matching code arrives.

        Args:
            platform: Lowercase platform key (e.g. "github", "google").
            source:   Expected channel - "sms", "telegram", "email", or "any".
            timeout:  Seconds before asyncio.TimeoutError is raised.

        Returns:
            The verification code string.

        Raises:
            asyncio.TimeoutError: No code arrived within timeout seconds.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        flow = _PendingOTPFlow(
            platform=platform,
            expected_source=source,
            future=future,
            timeout_seconds=timeout,
        )
        # Allow only one pending flow per platform at a time; replace stale ones.
        existing = self._pending.get(platform)
        if existing is not None and not existing.future.done():
            existing.future.cancel()
        self._pending[platform] = flow

        self._log.info(
            "otp_flow_registered",
            platform=platform,
            source=source,
            timeout=timeout,
        )
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._pending.pop(platform, None)
            self._log.warning("otp_flow_timeout", platform=platform, source=source)
            raise asyncio.TimeoutError(
                f"OTP for platform={platform!r} (source={source!r}) timed out after {timeout}s"
            )

    def on_otp_received(self, platform_hint: str | None, code: str, source: str) -> None:
        """
        Called when any channel receives a verification code.

        Resolves the first pending flow whose platform matches *platform_hint*
        (substring match) or whose expected_source matches *source* (when the
        flow accepts "any" channel).

        Args:
            platform_hint: Platform name extracted from the message, or None.
            code:          The extracted OTP code.
            source:        Channel that delivered the code: "sms"|"telegram"|"email".
        """
        if not code:
            return

        hint_lower = (platform_hint or "").lower()

        for key, flow in list(self._pending.items()):
            if flow.future.done():
                del self._pending[key]
                continue

            source_matches = flow.expected_source in (source, "any")
            platform_matches = bool(hint_lower and (hint_lower in key.lower() or key.lower() in hint_lower))

            if platform_matches or (source_matches and not hint_lower):
                # Prefer explicit platform match; fall back to source-only match.
                flow.future.set_result(code)
                del self._pending[key]
                self._log.info(
                    "otp_flow_resolved",
                    platform=key,
                    source=source,
                    platform_hint=platform_hint,
                )
                self._emit_resolved(key, code, source)
                return

        self._log.debug(
            "otp_received_no_pending_flow",
            platform_hint=platform_hint,
            source=source,
            code_length=len(code),
        )

    # ── Channel handlers ────────────────────────────────────────────────────

    async def _on_sms_otp(self, event: Any) -> None:
        """Handler for IDENTITY_VERIFICATION_RECEIVED (Twilio SMS)."""
        data: dict[str, Any] = event.data or {}
        code = data.get("code", "")
        raw_body = data.get("raw_body", "") or data.get("source", "")
        platform_hint = extract_platform_hint(raw_body)
        self.on_otp_received(platform_hint, code, "sms")

    async def _on_telegram_otp(self, event: Any) -> None:
        """Handler for TELEGRAM_OTP_RECEIVED (I-1)."""
        data: dict[str, Any] = event.data or {}
        code = data.get("code", "")
        raw_text = data.get("raw_text", "") or data.get("sender_username", "")
        platform_hint = data.get("platform_hint") or extract_platform_hint(raw_text)
        self.on_otp_received(platform_hint, code, "telegram")

    async def _on_email_otp(self, event: Any) -> None:
        """Handler for EMAIL_OTP_RECEIVED (I-2)."""
        data: dict[str, Any] = event.data or {}
        code = data.get("code", "")
        subject = data.get("subject", "") or data.get("sender_address", "")
        platform_hint = data.get("platform_hint") or extract_platform_hint(subject)
        self.on_otp_received(platform_hint, code, "email")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _emit_resolved(self, platform: str, code: str, source: str) -> None:
        """Fire-and-forget OTP_FLOW_RESOLVED event for observability."""
        if self._event_bus is None:
            return
        event = SynapseEvent(
            event_type=SynapseEventType.OTP_FLOW_RESOLVED,
            source_system="identity",
            data={"platform": platform, "code": code, "source": source},
        )
        asyncio.ensure_future(self._event_bus.emit(event))


def _raise_twilio_error(event: str, response: httpx.Response) -> None:
    """
    Parse a Twilio error response and raise a descriptive RuntimeError.

    Twilio error bodies look like::

        {"code": 21606, "message": "...", "more_info": "...", "status": 400}

    Known code ranges:
      - 20003 / 20008: bad credentials
      - 20003 / 21606: insufficient funds
    """
    body_text = response.text[:500]
    twilio_code: int | None = None
    try:
        payload: dict[str, Any] = response.json()
        twilio_code = payload.get("code")
        message = payload.get("message", body_text)
    except Exception:
        message = body_text

    logger.error(
        event,
        http_status=response.status_code,
        twilio_code=twilio_code,
        message=message,
    )

    if twilio_code in _TWILIO_ERROR_CODES_INSUFFICIENT_FUNDS:
        raise RuntimeError(
            f"Twilio account has insufficient funds to purchase a phone number "
            f"(Twilio error {twilio_code}): {message}"
        )
    if twilio_code in _TWILIO_ERROR_CODES_BAD_CREDENTIALS:
        raise RuntimeError(
            f"Twilio credential error (Twilio error {twilio_code}): {message}"
        )

    raise RuntimeError(
        f"Twilio API error HTTP {response.status_code} "
        f"(Twilio code {twilio_code}): {message}"
    )


# ---------------------------------------------------------------------------
# Telegram inbound webhook endpoint
# ---------------------------------------------------------------------------


import json as _json
import os as _os


@identity_comm_router.post("/api/v1/identity/comm/telegram/webhook")
async def telegram_inbound_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Receive an inbound Telegram bot webhook Update.

    Security:
      1. Validate X-Telegram-Bot-Api-Secret-Token header (constant-time compare).
      2. Parse the Telegram Update JSON body.
      3. Extract OTP codes (6-digit numeric, 4-8 char alphanumeric).
      4. Publish events on Synapse bus:
         - TELEGRAM_MESSAGE_RECEIVED - all text messages from authorized chats
         - TELEGRAM_OTP_RECEIVED     - additionally emitted when an OTP is found

    Returns HTTP 200 immediately; publication runs in a background task.
    Invalid secret token → HTTP 403.
    """
    event_bus = _get_event_bus(request)
    cfg = _get_config(request)

    # ── Secret token validation ───────────────────────────────────────────
    webhook_secret = _os.environ.get("ORGANISM_TELEGRAM_WEBHOOK_SECRET", "")
    if not webhook_secret:
        logger.warning("telegram_webhook_rejected", reason="webhook_secret_not_configured")
        raise HTTPException(status_code=503, detail="Telegram webhook secret not configured")

    incoming_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(incoming_token.encode(), webhook_secret.encode()):
        logger.warning("telegram_webhook_rejected", reason="invalid_secret_token")
        raise HTTPException(status_code=403, detail="Invalid Telegram secret token")

    # ── Parse Update body ─────────────────────────────────────────────────
    try:
        raw_body = await request.body()
        update: dict[str, Any] = _json.loads(raw_body)
    except Exception as exc:
        logger.warning("telegram_webhook_parse_failed", error=str(exc))
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    message: dict[str, Any] | None = update.get("message") or update.get("channel_post")
    callback_query: dict[str, Any] | None = update.get("callback_query")

    # ── Admin chat guard ──────────────────────────────────────────────────
    admin_chat_id_str: str = getattr(cfg.identity_comm, "telegram_admin_chat_id", "")
    admin_chat_id: int | None = int(admin_chat_id_str) if admin_chat_id_str else None

    if message:
        chat: dict[str, Any] = message.get("chat", {})
        chat_id: int = chat.get("id", 0)
        from_user: dict[str, Any] = message.get("from", {})
        from_user_id: int = from_user.get("id", 0)
        sender_username: str = from_user.get("username", "")
        text: str = message.get("text", "")
        timestamp: int = message.get("date", 0)

        if admin_chat_id is not None and chat_id != admin_chat_id:
            logger.warning(
                "telegram_webhook_unauthorized_chat",
                expected=admin_chat_id,
                received=chat_id,
            )
            return {"status": "ok", "action": "unauthorized_chat_dropped"}

        if not text:
            return {"status": "ok", "action": "ignored_empty"}

        otp_code: str | None = _extract_verification_code(text)
        platform_hint: str | None = extract_platform_hint(text)

        log = logger.bind(chat_id=chat_id, from_user_id=from_user_id)
        log.info(
            "telegram_message_received",
            has_otp=otp_code is not None,
            text_length=len(text),
        )

        async def _publish_telegram() -> None:
            try:
                msg_event = SynapseEvent(
                    event_type=SynapseEventType.TELEGRAM_MESSAGE_RECEIVED,
                    source_system="identity",
                    data={
                        "chat_id": chat_id,
                        "message_text": text,
                        "from_user_id": from_user_id,
                        "sender_username": sender_username,
                        "timestamp": timestamp,
                        "otp_code": otp_code or "",
                    },
                )
                await event_bus.emit(msg_event)

                if otp_code:
                    otp_event = SynapseEvent(
                        event_type=SynapseEventType.TELEGRAM_OTP_RECEIVED,
                        source_system="identity",
                        data={
                            "chat_id": chat_id,
                            "code": otp_code,
                            "sender_username": sender_username,
                            "raw_text": text,
                            "from_user_id": from_user_id,
                            "platform_hint": platform_hint,
                        },
                    )
                    await event_bus.emit(otp_event)
            except Exception as exc:
                logger.error("telegram_event_publish_failed", error=str(exc))

        background_tasks.add_task(_publish_telegram)
        return {"status": "ok", "action": "event_published"}

    elif callback_query:
        cq_id: str = callback_query.get("id", "")
        cq_data: str = callback_query.get("data", "")
        cq_from: dict[str, Any] = callback_query.get("from", {})
        cq_msg: dict[str, Any] = callback_query.get("message") or {}
        cq_chat_id: int = cq_msg.get("chat", {}).get("id", 0)

        async def _publish_callback() -> None:
            try:
                cb_event = SynapseEvent(
                    event_type=SynapseEventType.TELEGRAM_MESSAGE_RECEIVED,
                    source_system="identity",
                    data={
                        "chat_id": cq_chat_id,
                        "callback_query_id": cq_id,
                        "callback_data": cq_data,
                        "from_user_id": cq_from.get("id", 0),
                        "sender_username": cq_from.get("username", ""),
                        "message_text": "",
                        "timestamp": 0,
                        "otp_code": "",
                    },
                )
                await event_bus.emit(cb_event)
            except Exception as exc:
                logger.error("telegram_callback_publish_failed", error=str(exc))

        background_tasks.add_task(_publish_callback)
        return {"status": "ok", "action": "callback_published"}

    return {"status": "ok", "action": "update_type_ignored"}


# ---------------------------------------------------------------------------
# Telegram command handler
# ---------------------------------------------------------------------------

_TELEGRAM_COMMANDS: dict[str, str] = {
    "/start": "start",
    "/help": "help",
    "/status": "status",
    "/ping": "ping",
}

_COMMAND_HELP_TEXT = (
    "*EcodiaOS Bot Commands*\n\n"
    "/ping - liveness check\n"
    "/status - organism health summary\n"
    "/help - show this message\n"
)


class TelegramCommandHandler:
    """
    Routes inbound Telegram bot commands (/ping, /status, /help) to live
    organism state and replies via the TelegramConnector.

    Subscribes to TELEGRAM_MESSAGE_RECEIVED on the Synapse bus. Non-command
    messages are ignored here (OTPs are handled by OTPCoordinator separately).

    Wiring::

        handler = TelegramCommandHandler(connector=tg_connector, synapse=synapse)
        handler.set_event_bus(event_bus)

    For /status to return metabolic data, also call::

        handler.set_oikos(oikos)
    """

    def __init__(self, connector: Any, synapse: Any) -> None:
        self._connector = connector
        self._synapse = synapse
        self._oikos: Any = None
        self._log = structlog.get_logger("identity.telegram_cmd")

    def set_oikos(self, oikos: Any) -> None:
        self._oikos = oikos

    def set_event_bus(self, event_bus: Any) -> None:
        event_bus.subscribe(
            SynapseEventType.TELEGRAM_MESSAGE_RECEIVED,
            self._on_message,
        )

    async def _on_message(self, event: Any) -> None:
        data: dict[str, Any] = event.data or {}
        text: str = (data.get("message_text") or "").strip()
        chat_id: int | str = data.get("chat_id", 0)

        if not text.startswith("/"):
            return

        # Extract command without arguments
        cmd = text.split()[0].lower().split("@")[0]

        self._log.info("telegram_command_received", command=cmd, chat_id=chat_id)

        if cmd == "/ping":
            await self._reply(chat_id, "pong")

        elif cmd in ("/help", "/start"):
            await self._reply(chat_id, _COMMAND_HELP_TEXT)

        elif cmd == "/status":
            reply = await self._build_status_reply()
            await self._reply(chat_id, reply)

        else:
            await self._reply(chat_id, f"Unknown command: `{cmd}`\nTry /help")

    async def _build_status_reply(self) -> str:
        lines: list[str] = ["*EcodiaOS Status*\n"]

        # Metabolic snapshot from Synapse
        if self._synapse is not None:
            try:
                snap = await self._synapse.metabolic_snapshot()
                if snap:
                    burn = snap.get("burn_rate_usd_per_day", 0)
                    runway = snap.get("runway_days", 0)
                    lines.append(f"Burn rate: ${burn:.2f}/day")
                    lines.append(f"Runway: {runway:.1f} days")
            except Exception as exc:
                self._log.warning("status_metabolic_failed", error=str(exc))

        # Economic state from Oikos
        if self._oikos is not None:
            try:
                econ = await self._oikos.economic_state()
                if econ:
                    liquid = econ.get("liquid_balance_usd", 0)
                    lines.append(f"Liquid: ${liquid:.4f}")
            except Exception as exc:
                self._log.warning("status_oikos_failed", error=str(exc))

        if len(lines) == 1:
            lines.append("_(no metabolic data available)_")

        return "\n".join(lines)

    async def _reply(self, chat_id: int | str, text: str) -> None:
        if self._connector is None:
            return
        try:
            await self._connector.send_message(
                chat_id=chat_id, text=text, parse_mode="Markdown"
            )
        except Exception as exc:
            self._log.warning("telegram_reply_failed", chat_id=chat_id, error=str(exc))


# ---------------------------------------------------------------------------
# Telegram getUpdates polling loop (fallback when no public URL)
# ---------------------------------------------------------------------------


class TelegramPollingLoop:
    """
    Long-polling fallback for Telegram when no public HTTPS URL is available
    for webhook registration.

    Uses the Telegram getUpdates API with long-polling (timeout=30s). Each
    update is processed and dispatched as Synapse events, exactly mirroring
    what the webhook endpoint does. The polling loop exits cleanly on
    asyncio.CancelledError.

    Wiring::

        poller = TelegramPollingLoop(connector=tg_connector, event_bus=bus)
        supervised_task(poller.run(), name="telegram_polling", ...)

    Do NOT start this when a webhook is already registered - Telegram does
    not allow simultaneous webhook + polling.
    """

    _POLL_TIMEOUT_S = 30
    _RETRY_SLEEP_S = 5

    def __init__(self, connector: Any, event_bus: Any) -> None:
        self._connector = connector
        self._event_bus = event_bus
        self._offset: int = 0
        self._log = structlog.get_logger("identity.telegram_polling")

    async def run(self) -> None:
        """Long-poll loop. Runs until cancelled."""
        self._log.info("telegram_polling_started")
        import asyncio as _asyncio

        while True:
            try:
                updates = await self._fetch_updates()
                for update in updates:
                    update_id: int = update.get("update_id", 0)
                    if update_id >= self._offset:
                        self._offset = update_id + 1
                    await self._dispatch_update(update)
            except _asyncio.CancelledError:
                self._log.info("telegram_polling_cancelled")
                raise
            except Exception as exc:
                self._log.warning("telegram_polling_error", error=str(exc))
                await _asyncio.sleep(self._RETRY_SLEEP_S)

    async def _fetch_updates(self) -> list[dict[str, Any]]:
        token = self._connector._resolve_token()
        if not token:
            import asyncio as _asyncio
            await _asyncio.sleep(self._RETRY_SLEEP_S)
            return []

        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=self._POLL_TIMEOUT_S + 5.0) as client:
            resp = await client.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={
                    "offset": self._offset,
                    "timeout": self._POLL_TIMEOUT_S,
                    "allowed_updates": ["message", "callback_query"],
                },
            )
        if resp.status_code != 200:
            self._log.warning("telegram_polling_http_error", status=resp.status_code)
            return []
        data = resp.json()
        if not data.get("ok"):
            self._log.warning("telegram_polling_api_error", description=data.get("description"))
            return []
        return data.get("result", [])  # type: ignore[no-any-return]

    async def _dispatch_update(self, update: dict[str, Any]) -> None:
        """Emit the same Synapse events the webhook endpoint would emit."""
        import os as _os

        message: dict[str, Any] | None = update.get("message") or update.get("channel_post")
        if not message:
            return

        chat: dict[str, Any] = message.get("chat", {})
        chat_id: int = chat.get("id", 0)
        from_user: dict[str, Any] = message.get("from", {})
        sender_username: str = from_user.get("username", "")
        from_user_id: int = from_user.get("id", 0)
        text: str = message.get("text", "")
        timestamp: int = message.get("date", 0)

        # Admin chat guard
        admin_chat_id_str = _os.environ.get("ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID", "")
        admin_chat_id: int | None = int(admin_chat_id_str) if admin_chat_id_str else None
        if admin_chat_id is not None and chat_id != admin_chat_id:
            return

        if not text:
            return

        otp_code: str | None = _extract_verification_code(text)
        platform_hint: str | None = extract_platform_hint(text)

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.TELEGRAM_MESSAGE_RECEIVED,
                source_system="identity",
                data={
                    "chat_id": chat_id,
                    "message_text": text,
                    "from_user_id": from_user_id,
                    "sender_username": sender_username,
                    "timestamp": timestamp,
                    "otp_code": otp_code or "",
                },
            ))

            if otp_code:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.TELEGRAM_OTP_RECEIVED,
                    source_system="identity",
                    data={
                        "chat_id": chat_id,
                        "code": otp_code,
                        "sender_username": sender_username,
                        "raw_text": text,
                        "from_user_id": from_user_id,
                        "platform_hint": platform_hint,
                    },
                ))
        except Exception as exc:
            self._log.error("telegram_polling_dispatch_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Telegram status broadcast helper
# ---------------------------------------------------------------------------


async def send_telegram_status(
    connector: Any,
    chat_id: int | str,
    message: str,
    parse_mode: str = "Markdown",
) -> None:
    """
    Send a plain-text status message to a Telegram chat via TelegramConnector.

    Best-effort - never raises; silently no-ops when the connector is unavailable.
    """
    if connector is None:
        return
    try:
        await connector.send_message(chat_id=chat_id, text=message, parse_mode=parse_mode)
        logger.info("telegram_status_sent", chat_id=chat_id, length=len(message))
    except Exception as exc:
        logger.warning("telegram_status_send_failed", chat_id=chat_id, error=str(exc))
