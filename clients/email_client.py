"""
EcodiaOS - Email Client

Async email sending with two backends:
  Primary:  AWS SES via boto3 (cheapest; organism already has AWS creds for S3)
  Fallback: SMTP via aiosmtplib (works with Gmail SMTP or any provider)

Configuration (env vars):
  ECODIAOS_EMAIL__PROVIDER         "ses" | "smtp"  (default: "ses")
  ECODIAOS_EMAIL__FROM_ADDRESS      sender address
  ECODIAOS_EMAIL__AWS_REGION        SES region      (default: "us-east-1")
  ECODIAOS_EMAIL__SMTP_HOST         SMTP hostname
  ECODIAOS_EMAIL__SMTP_PORT         SMTP port       (default: 587)
  ECODIAOS_EMAIL__SMTP_USERNAME     SMTP login
  ECODIAOS_EMAIL__SMTP_PASSWORD     SMTP password   (set via env/vault)
"""

from __future__ import annotations

import asyncio
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import structlog

logger = structlog.get_logger("clients.email_client")

_PROVIDER = os.environ.get("ECODIAOS_EMAIL__PROVIDER", "ses").lower()
_FROM_ADDRESS = os.environ.get("ECODIAOS_EMAIL__FROM_ADDRESS", "noreply@ecodiaos.org")
_AWS_REGION = os.environ.get("ECODIAOS_EMAIL__AWS_REGION", "us-east-1")
_SMTP_HOST = os.environ.get("ECODIAOS_EMAIL__SMTP_HOST", "")
_SMTP_PORT = int(os.environ.get("ECODIAOS_EMAIL__SMTP_PORT", "587"))
_SMTP_USERNAME = os.environ.get("ECODIAOS_EMAIL__SMTP_USERNAME", "")
_SMTP_PASSWORD = os.environ.get("ECODIAOS_EMAIL__SMTP_PASSWORD", "")


class EmailClient:
    """
    Async email client backed by AWS SES or SMTP.

    Usage::

        client = EmailClient()
        sent = await client.send(
            to="admin@example.com",
            subject="Alert",
            body="Something happened.",
        )
    """

    def __init__(
        self,
        provider: str | None = None,
        from_address: str | None = None,
        aws_region: str | None = None,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_username: str | None = None,
        smtp_password: str | None = None,
    ) -> None:
        self._provider = (provider or _PROVIDER).lower()
        self._from_address = from_address or _FROM_ADDRESS
        self._aws_region = aws_region or _AWS_REGION
        self._smtp_host = smtp_host or _SMTP_HOST
        self._smtp_port = smtp_port or _SMTP_PORT
        self._smtp_username = smtp_username or _SMTP_USERNAME
        self._smtp_password = smtp_password or _SMTP_PASSWORD

        logger.info(
            "email_client_initialized",
            provider=self._provider,
            from_address=self._from_address,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        from_addr: str | None = None,
        html: bool = False,
    ) -> dict[str, Any]:
        """
        Send an email.  Returns a dict with at least a ``message_id`` key.
        Returns ``{}`` on failure (logs the error; never raises).
        """
        sender = from_addr or self._from_address
        log = logger.bind(to=to, subject=subject[:80], provider=self._provider)

        try:
            if self._provider == "ses":
                return await self._send_ses(
                    to=to, subject=subject, body=body, from_addr=sender, html=html
                )
            else:
                return await self._send_smtp(
                    to=to, subject=subject, body=body, from_addr=sender, html=html
                )
        except Exception as exc:
            log.error("email_send_failed", error=str(exc))
            # Try SMTP fallback if SES primary failed
            if self._provider == "ses" and self._smtp_host:
                try:
                    log.warning("email_ses_failed_trying_smtp_fallback")
                    return await self._send_smtp(
                        to=to, subject=subject, body=body, from_addr=sender, html=html
                    )
                except Exception as smtp_exc:
                    log.error("email_smtp_fallback_failed", error=str(smtp_exc))
            return {}

    async def health_check(self) -> bool:
        """
        Verify the email backend is reachable.
        Returns True if healthy, False otherwise.  Never raises.
        """
        try:
            if self._provider == "ses":
                return await self._ses_health()
            else:
                return await self._smtp_health()
        except Exception as exc:
            logger.warning("email_health_check_failed", provider=self._provider, error=str(exc))
            return False

    # ── AWS SES backend ─────────────────────────────────────────────────────

    async def _send_ses(
        self,
        to: str,
        subject: str,
        body: str,
        from_addr: str,
        html: bool,
    ) -> dict[str, Any]:
        """Send via AWS SES using boto3 in a thread pool."""
        import boto3  # type: ignore[import]

        def _send_sync() -> dict[str, Any]:
            client = boto3.client("ses", region_name=self._aws_region)
            body_block: dict[str, Any] = (
                {"Html": {"Data": body, "Charset": "UTF-8"}}
                if html
                else {"Text": {"Data": body, "Charset": "UTF-8"}}
            )
            resp = client.send_email(
                Source=from_addr,
                Destination={"ToAddresses": [to]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": body_block,
                },
            )
            return {"message_id": resp.get("MessageId", ""), "provider": "ses"}

        result: dict[str, Any] = await asyncio.get_event_loop().run_in_executor(
            None, _send_sync
        )
        logger.info(
            "email_sent_ses",
            to=to,
            subject=subject[:80],
            message_id=result.get("message_id", ""),
        )
        return result

    async def _ses_health(self) -> bool:
        """Check SES by verifying AWS credentials are present and boto3 is importable."""
        try:
            import boto3  # type: ignore[import]  # noqa: F401

            return bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"))
        except ImportError:
            logger.debug("email_ses_health_boto3_missing")
            return False

    # ── SMTP backend ─────────────────────────────────────────────────────────

    async def _send_smtp(
        self,
        to: str,
        subject: str,
        body: str,
        from_addr: str,
        html: bool,
    ) -> dict[str, Any]:
        """Send via SMTP using aiosmtplib."""
        import aiosmtplib  # type: ignore[import]

        if not self._smtp_host:
            raise RuntimeError(
                "SMTP provider selected but ECODIAOS_EMAIL__SMTP_HOST is not set"
            )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to
        content_type = "html" if html else "plain"
        msg.attach(MIMEText(body, content_type, "utf-8"))

        smtp_kwargs: dict[str, Any] = {
            "hostname": self._smtp_host,
            "port": self._smtp_port,
            "start_tls": True,
        }
        if self._smtp_username:
            smtp_kwargs["username"] = self._smtp_username
        if self._smtp_password:
            smtp_kwargs["password"] = self._smtp_password

        await aiosmtplib.send(msg, **smtp_kwargs)

        message_id = msg.get("Message-ID", "")
        logger.info("email_sent_smtp", to=to, subject=subject[:80])
        return {"message_id": message_id, "provider": "smtp"}

    async def _smtp_health(self) -> bool:
        """Probe SMTP by opening and closing a connection."""
        if not self._smtp_host:
            return False
        try:
            import aiosmtplib  # type: ignore[import]

            smtp = aiosmtplib.SMTP(hostname=self._smtp_host, port=self._smtp_port)
            await asyncio.wait_for(smtp.connect(), timeout=5.0)
            await smtp.quit()
            return True
        except Exception:
            return False
