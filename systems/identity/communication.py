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
    (env: ECODIAOS_IDENTITY_COMM__TWILIO_AUTH_TOKEN) and is never logged.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import imaplib
import re
import urllib.parse
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
# IMAP background task stub
# ---------------------------------------------------------------------------


async def scan_imap_inbox(
    cfg: EcodiaOSConfig,
    event_bus: EventBus,
) -> None:
    """
    Scan the configured IMAP inbox for inbound verification emails.

    Wire to a scheduler with interval cfg.identity_comm.imap_scan_interval_s:

        scheduler.add_job(
            scan_imap_inbox,
            "interval",
            seconds=cfg.identity_comm.imap_scan_interval_s,
            args=[cfg, event_bus],
        )

    For each unseen message containing a code, publishes
    IDENTITY_VERIFICATION_RECEIVED on Synapse.

    imaplib is synchronous; wrap in asyncio.to_thread() for production, or
    replace with aioimaplib for native async.
    """
    comm = cfg.identity_comm
    if not comm.imap_host or not comm.imap_username:
        logger.debug("imap_scan_skipped", reason="not_configured")
        return

    log = logger.bind(imap_host=comm.imap_host, mailbox=comm.imap_mailbox)
    log.info("imap_scan_starting")

    try:
        conn = imaplib.IMAP4_SSL(comm.imap_host, comm.imap_port)
        conn.login(comm.imap_username, comm.imap_password)
        conn.select(comm.imap_mailbox)

        _status, id_data = conn.search(None, "UNSEEN")
        message_ids: list[bytes] = id_data[0].split() if id_data[0] else []
        log.info("imap_scan_found_messages", count=len(message_ids))

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
                log.info(
                    "imap_verification_code_found",
                    sender=sender,
                    code_length=len(code),
                )
                event = SynapseEvent(
                    event_type=SynapseEventType.IDENTITY_VERIFICATION_RECEIVED,
                    source_system="identity",
                    data={"source": sender, "code": code, "channel": "email"},
                )
                await event_bus.emit(event)

            except Exception as msg_exc:
                log.warning(
                    "imap_message_parse_failed",
                    msg_id=msg_id,
                    error=str(msg_exc),
                )

        conn.logout()
        log.info("imap_scan_complete")

    except imaplib.IMAP4.error as imap_exc:
        log.error("imap_connection_failed", error=str(imap_exc))
    except Exception as exc:
        log.error("imap_scan_failed", error=str(exc))


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
            "Set ECODIAOS_IDENTITY_COMM__TWILIO_ACCOUNT_SID and __TWILIO_AUTH_TOKEN."
        )

    country_upper = country.upper().strip()
    twilio_country = _TWILIO_COUNTRY_CODES.get(country_upper)
    if twilio_country is None:
        raise RuntimeError(
            f"Country '{country}' is not in the supported list. "
            f"Supported: {sorted(_TWILIO_COUNTRY_CODES)}. "
            "You can manually provision a Twilio number for this country and set "
            "ECODIAOS_IDENTITY_COMM__TWILIO_FROM_NUMBER directly."
        )

    log = logger.bind(area_code=area_code, country=twilio_country, account_sid=account_sid[:8] + "…")
    auth = (account_sid, auth_token)

    # Build search params — area code semantics differ by country
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
            # Twilio rejects invalid area codes for the country — retry without it
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
                reason="ECODIAOS_IDENTITY_COMM__WEBHOOK_BASE_URL not set",
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
