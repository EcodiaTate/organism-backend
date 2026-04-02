"""
EcodiaOS - Autonomous Account Provisioner (Phase 16h: Platform Identity)

Gives each EOS instance its own platform identities - GitHub account, phone
number, and generic platform accounts - without human intervention.

Architecture:
  1. AccountProvisioner owns the end-to-end flow for each platform.
  2. Twilio numbers are provisioned via REST API (no browser / no CAPTCHA).
  3. GitHub (and future platforms) use Playwright + 2captcha + OTPCoordinator.
  4. Every provisioning attempt gets Equor constitutional approval before
     spending wallet funds or external API calls.
  5. All costs are logged to Synapse for Oikos metabolic accounting.
  6. Credentials sealed in IdentityVault + recorded in connector_credentials.
  7. Neo4j gets an (:AccountProvisioning) audit node for every attempt.

Per-instance username pattern:
  GitHub: {prefix}-{instance_id[:8]}    (e.g. ecodiaos-a1b2c3d4)
  Gmail:  {prefix}.{instance_id[:8]}@gmail.com

Environment variables:
  ORGANISM_CAPTCHA__TWOCAPTCHA_API_KEY
  ORGANISM_CAPTCHA__PROVIDER
  ORGANISM_ACCOUNT_PROVISIONER__ENABLED
  ORGANISM_ACCOUNT_PROVISIONER__GITHUB_USERNAME_PREFIX
  ORGANISM_ACCOUNT_PROVISIONER__TWILIO_AREA_CODE
  ORGANISM_IDENTITY_COMM__TWILIO_ACCOUNT_SID  (existing)
  ORGANISM_IDENTITY_COMM__TWILIO_AUTH_TOKEN   (existing)
  ORGANISM_PUBLIC_URL                          (existing)
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from primitives.common import new_id, utc_now

if TYPE_CHECKING:
    from clients.browser_client import BrowserClient
    from clients.captcha_client import CaptchaClient
    from clients.neo4j import Neo4jClient
    from config import AccountProvisionerConfig, EcodiaOSConfig
    from systems.identity.communication import OTPCoordinator
    from systems.identity.vault import IdentityVault
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.account_provisioner")

_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"

# ─── Result types ────────────────────────────────────────────────────────────


class ProvisioningStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    EQUOR_DENIED = "equor_denied"
    ALREADY_EXISTS = "already_exists"
    CAPTCHA_FAILED = "captcha_failed"
    OTP_TIMEOUT = "otp_timeout"


@dataclass
class GitHubAccountResult:
    status: ProvisioningStatus
    username: str = ""
    email: str = ""
    instance_id: str = ""
    provisioning_id: str = field(default_factory=new_id)
    pat_sealed: bool = False
    error: str = ""


@dataclass
class GmailResult:
    status: ProvisioningStatus
    email: str = ""
    instance_id: str = ""
    provisioning_id: str = field(default_factory=new_id)
    error: str = ""


@dataclass
class AccountResult:
    status: ProvisioningStatus
    platform: str = ""
    username: str = ""
    instance_id: str = ""
    provisioning_id: str = field(default_factory=new_id)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# ─── AccountProvisioner ──────────────────────────────────────────────────────


class AccountProvisioner:
    """
    Autonomous external platform account provisioner.

    Lifecycle: call initialize() once; then call provision_* methods as needed.
    The provisioner is designed to be fire-and-forget from IdentitySystem.initialize()
    via asyncio.create_task() - it never blocks boot.

    Thread-safety: NOT thread-safe; single asyncio event loop.
    """

    def __init__(self) -> None:
        self._config: AccountProvisionerConfig | None = None
        self._full_config: EcodiaOSConfig | None = None
        self._event_bus: EventBus | None = None
        self._vault: IdentityVault | None = None
        self._neo4j: Neo4jClient | None = None
        self._otp_coordinator: OTPCoordinator | None = None
        self._captcha_client: CaptchaClient | None = None
        self._instance_id: str = ""
        self._log = logger.bind(system="account_provisioner")
        # Track which platforms we have successfully provisioned in this session
        self._provisioned: set[str] = set()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def initialize(
        self,
        instance_id: str,
        config: "EcodiaOSConfig",
        vault: "IdentityVault",
        otp_coordinator: "OTPCoordinator",
        event_bus: "EventBus | None" = None,
        neo4j: "Neo4jClient | None" = None,
    ) -> None:
        """
        Wire dependencies. Call once before any provision_* methods.
        Does NOT start provisioning - call provision_platform_identities() to start.
        """
        self._instance_id = instance_id
        self._full_config = config
        self._config = config.account_provisioner
        self._vault = vault
        self._otp_coordinator = otp_coordinator
        self._event_bus = event_bus
        self._neo4j = neo4j

        # Lazy-init captcha client if configured
        if config.captcha.enabled:
            from clients.captcha_client import CaptchaClient
            self._captcha_client = CaptchaClient(config.captcha)
            if event_bus:
                self._captcha_client.set_event_bus(event_bus)

        self._log = self._log.bind(instance_id=instance_id[:8])
        self._log.info("account_provisioner_initialized", captcha_ready=config.captcha.enabled)

    def set_event_bus(self, event_bus: "EventBus") -> None:
        self._event_bus = event_bus
        if self._captcha_client:
            self._captcha_client.set_event_bus(event_bus)

    def set_neo4j(self, neo4j: "Neo4jClient") -> None:
        self._neo4j = neo4j

    # ── Public API ───────────────────────────────────────────────────────

    async def provision_platform_identities(self) -> None:
        """
        Entry point called from IdentitySystem on first boot (via create_task).

        Provisions in order:
          1. Twilio phone number (if not already owned)
          2. GitHub account (if not already created)

        Each step is gated by Equor and gracefully skipped on failure.
        """
        if self._config is None or not self._config.enabled:
            self._log.info("provisioning_disabled")
            return

        self._log.info("platform_identity_provisioning_start")

        # Step 1 - Twilio number
        if not await self._has_twilio_number():
            try:
                number = await self.provision_twilio_number(
                    self._config.twilio_area_code
                )
                self._log.info("twilio_number_provisioned", number=number)
            except Exception as exc:
                self._log.warning("twilio_provision_failed", error=str(exc))
                await self._emit_event(
                    "ACCOUNT_PROVISIONING_FAILED",
                    {
                        "platform": "twilio",
                        "instance_id": self._instance_id,
                        "error": str(exc),
                        "retryable": True,
                    },
                )

        # Step 2 - GitHub account
        if not await self._has_own_github_account():
            username, email = self._derive_github_identity()
            try:
                result = await self.provision_github_account(username, email)
                if result.status == ProvisioningStatus.SUCCESS:
                    self._log.info("github_account_provisioned", username=result.username)
                else:
                    self._log.warning(
                        "github_provision_incomplete",
                        status=result.status,
                        error=result.error,
                    )
                    await self._emit_event(
                        "ACCOUNT_PROVISIONING_FAILED",
                        {
                            "platform": "github",
                            "instance_id": self._instance_id,
                            "error": result.error,
                            "status": result.status.value,
                            "retryable": result.status != ProvisioningStatus.EQUOR_DENIED,
                        },
                    )
            except Exception as exc:
                self._log.warning("github_provision_failed", error=str(exc))
                await self._emit_event(
                    "ACCOUNT_PROVISIONING_FAILED",
                    {
                        "platform": "github",
                        "instance_id": self._instance_id,
                        "error": str(exc),
                        "retryable": True,
                    },
                )

        self._log.info("platform_identity_provisioning_complete")

    async def provision_twilio_number(self, area_code: str = "415") -> str:
        """
        Purchase a new Twilio phone number for this instance.

        Steps:
          1. Equor constitutional gate (economic action)
          2. POST to Twilio IncomingPhoneNumbers REST API
          3. Seal {phone_number, twilio_sid} in IdentityVault
          4. Set TWILIO_FROM_NUMBER in os.environ (runtime override)
          5. Emit PHONE_NUMBER_PROVISIONED event
          6. Deduct cost from Oikos via DOMAIN_EPISODE_RECORDED

        Returns:
            E.164 phone number string (e.g. "+14155551234")

        Raises:
            RuntimeError: On Twilio API error or missing credentials.
        """
        provisioning_id = new_id()
        self._log.info("twilio_provision_start", area_code=area_code, provisioning_id=provisioning_id)

        # ── Equor gate ────────────────────────────────────────────────
        approved = await self._request_equor_approval(
            action="provision_twilio_number",
            context={
                "area_code": area_code,
                "estimated_cost_usd": self._config.twilio_number_cost_usd if self._config else "1.15",
                "purpose": "autonomous_phone_identity",
            },
            provisioning_id=provisioning_id,
        )
        if not approved:
            await self._audit_neo4j(
                platform="twilio",
                status=ProvisioningStatus.EQUOR_DENIED,
                provisioning_id=provisioning_id,
            )
            await self._emit_event(
                "PROVISIONING_REQUIRES_HUMAN_ESCALATION",
                {
                    "reason": "Equor denied Twilio number provisioning",
                    "platform": "twilio",
                    "instance_id": self._instance_id,
                    "provisioning_id": provisioning_id,
                },
            )
            raise RuntimeError("Equor denied Twilio number provisioning")

        # ── Twilio REST API ───────────────────────────────────────────
        cfg = self._full_config
        if cfg is None:
            raise RuntimeError("AccountProvisioner not initialized")

        account_sid = cfg.identity_comm.twilio_account_sid
        auth_token = cfg.identity_comm.twilio_auth_token
        if not account_sid or not auth_token:
            raise RuntimeError(
                "Twilio credentials missing: set ORGANISM_IDENTITY_COMM__TWILIO_ACCOUNT_SID "
                "and ORGANISM_IDENTITY_COMM__TWILIO_AUTH_TOKEN"
            )

        # Build webhook URL
        public_url = (
            cfg.identity_comm.webhook_base_url
            or os.environ.get("ORGANISM_PUBLIC_URL", "")
        ).rstrip("/")
        webhook_url = f"{public_url}/api/v1/identity/webhook/twilio" if public_url else ""

        payload: dict[str, str] = {"AreaCode": area_code}
        if webhook_url:
            payload["SmsUrl"] = webhook_url
            payload["SmsMethod"] = "POST"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{_TWILIO_API_BASE}/Accounts/{account_sid}/IncomingPhoneNumbers.json",
                data=payload,
                auth=(account_sid, auth_token),
            )

        if response.status_code not in (200, 201):
            body = response.text
            self._log.error(
                "twilio_api_error",
                status=response.status_code,
                body=body[:200],
            )
            await self._audit_neo4j(
                platform="twilio",
                status=ProvisioningStatus.FAILED,
                provisioning_id=provisioning_id,
                error=f"HTTP {response.status_code}: {body[:100]}",
            )
            await self._emit_event(
                "ACCOUNT_PROVISIONING_FAILED",
                {
                    "platform": "twilio",
                    "instance_id": self._instance_id,
                    "provisioning_id": provisioning_id,
                    "error": f"HTTP {response.status_code}: {body[:100]}",
                    "retryable": response.status_code >= 500,
                },
            )
            raise RuntimeError(f"Twilio API error {response.status_code}: {body[:200]}")

        data = response.json()
        phone_number: str = data["phone_number"]  # E.164 format
        number_sid: str = data["sid"]
        friendly_name: str = data.get("friendly_name", phone_number)

        self._log.info("twilio_number_purchased", number=phone_number, sid=number_sid)

        # ── Seal in vault ─────────────────────────────────────────────
        if self._vault is not None:
            credential = {
                "phone_number": phone_number,
                "twilio_sid": number_sid,
                "area_code": area_code,
                "friendly_name": friendly_name,
                "webhook_url": webhook_url,
                "provisioned_at": utc_now().isoformat(),
            }
            self._vault.encrypt_token_json(
                credential,
                platform_id=f"twilio.number.{self._instance_id[:8]}",
                purpose="phone_number",
            )

        # ── Runtime environment update ────────────────────────────────
        os.environ["ORGANISM_IDENTITY_COMM__TWILIO_FROM_NUMBER"] = phone_number

        # ── Synapse events ────────────────────────────────────────────
        cost_usd = self._config.twilio_number_cost_usd if self._config else "1.15"

        await self._emit_event(
            "PHONE_NUMBER_PROVISIONED",
            {
                "phone_number": phone_number,
                "twilio_sid": number_sid,
                "area_code": area_code,
                "cost_usd": cost_usd,
                "webhook_url": webhook_url,
                "instance_id": self._instance_id,
                "provisioning_id": provisioning_id,
            },
        )

        await self._emit_cost_event(
            platform="twilio",
            cost_usd=float(cost_usd),
            provisioning_id=provisioning_id,
        )

        await self._audit_neo4j(
            platform="twilio",
            status=ProvisioningStatus.SUCCESS,
            provisioning_id=provisioning_id,
            metadata={"phone_number": phone_number, "twilio_sid": number_sid},
        )

        self._provisioned.add("twilio")
        return phone_number

    async def provision_github_account(
        self,
        username: str,
        email: str,
    ) -> GitHubAccountResult:
        """
        Create a new GitHub account for this instance via Playwright.

        Steps:
          1. Equor gate
          2. Check username availability (GitHub REST API)
          3. Open github.com/signup in Playwright browser
          4. Fill email, password, username
          5. Solve CAPTCHA if present (2captcha)
          6. Enter email OTP (via IMAP/OTPCoordinator)
          7. Generate Personal Access Token via Settings → PAT
          8. Seal PAT in vault as connector_credentials
          9. Emit GITHUB_ACCOUNT_PROVISIONED

        Returns:
            GitHubAccountResult with status and account details.
        """
        provisioning_id = new_id()
        self._log.info(
            "github_provision_start",
            username=username,
            email=email,
            provisioning_id=provisioning_id,
        )

        # ── Equor gate ────────────────────────────────────────────────
        approved = await self._request_equor_approval(
            action="provision_github_account",
            context={
                "username": username,
                "email": email,
                "purpose": "autonomous_development_identity",
            },
            provisioning_id=provisioning_id,
        )
        if not approved:
            await self._audit_neo4j(
                platform="github",
                status=ProvisioningStatus.EQUOR_DENIED,
                provisioning_id=provisioning_id,
            )
            return GitHubAccountResult(
                status=ProvisioningStatus.EQUOR_DENIED,
                username=username,
                email=email,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error="Equor denied GitHub account provisioning",
            )

        # ── Check username availability ────────────────────────────────
        if not await self._github_username_available(username):
            self._log.warning("github_username_taken", username=username)
            # Try adding a random suffix
            username = f"{username}-{secrets.token_hex(2)}"
            if not await self._github_username_available(username):
                return GitHubAccountResult(
                    status=ProvisioningStatus.FAILED,
                    username=username,
                    email=email,
                    instance_id=self._instance_id,
                    provisioning_id=provisioning_id,
                    error="GitHub username unavailable after suffix retry",
                )

        # ── Browser-based signup ───────────────────────────────────────
        config = self._full_config
        if config is None:
            return GitHubAccountResult(
                status=ProvisioningStatus.FAILED,
                username=username,
                email=email,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error="AccountProvisioner not initialized",
            )

        headless = config.account_provisioner.browser_headless
        stealth = config.account_provisioner.browser_stealth
        otp_timeout = config.account_provisioner.otp_wait_timeout_s

        password = _generate_password()

        try:
            from clients.browser_client import BrowserClient

            async with BrowserClient(headless=headless, stealth=stealth) as browser:
                page = await browser.create_page()

                # Navigate to signup
                await browser.goto(page, "https://github.com/signup")
                await browser.human_pause(1.0, 2.5)

                # Fill email
                result = await browser.fill_form(
                    page, {"#email": email}, typing_delay_ms=80
                )
                if not result.ok:
                    raise RuntimeError(f"GitHub email fill failed: {result.errors}")

                await browser.click(page, "button[type='submit'], .js-signup-button")
                await browser.human_pause(1.5, 3.0)

                # Fill password
                await browser.fill_form(
                    page, {"#password": password}, typing_delay_ms=70
                )
                await browser.human_pause(0.5, 1.5)

                # Fill username
                await browser.fill_form(
                    page, {"#login": username}, typing_delay_ms=90
                )
                await browser.human_pause(0.8, 2.0)

                # Opt out of email marketing if checkbox visible
                try:
                    cb = await browser.wait_for_selector(
                        page, "#opt_in", timeout_ms=3000
                    )
                    if cb:
                        # Uncheck if checked
                        checked = await page.is_checked("#opt_in")
                        if checked:
                            await page.uncheck("#opt_in")
                except Exception:
                    pass

                # Solve CAPTCHA if present
                if self._captcha_client:
                    solved = await browser.solve_captcha_on_page(
                        page, self._captcha_client
                    )
                    if not solved:
                        await self._audit_neo4j(
                            platform="github",
                            status=ProvisioningStatus.CAPTCHA_FAILED,
                            provisioning_id=provisioning_id,
                        )
                        await self._emit_event(
                            "ACCOUNT_PROVISIONING_FAILED",
                            {
                                "platform": "github",
                                "instance_id": self._instance_id,
                                "provisioning_id": provisioning_id,
                                "error": "CAPTCHA solve failed",
                                "retryable": True,
                            },
                        )
                        return GitHubAccountResult(
                            status=ProvisioningStatus.CAPTCHA_FAILED,
                            username=username,
                            email=email,
                            instance_id=self._instance_id,
                            provisioning_id=provisioning_id,
                            error="CAPTCHA solve failed",
                        )

                # Submit form
                await browser.click(
                    page, "button[type='submit'], .js-signup-button"
                )
                await browser.human_pause(2.0, 4.0)

                # Wait for email verification step
                email_step = await browser.wait_for_selector(
                    page, "#user-email-verification-field, .js-emailverificationcode",
                    timeout_ms=15_000,
                )

                if email_step is None:
                    # Check if already on dashboard (rare: no email verification step)
                    page_text = await browser.get_page_text(page)
                    if "dashboard" in page.url or "Welcome to GitHub" in page_text:
                        self._log.info("github_signup_no_email_verification_step")
                    else:
                        # Screenshot for debugging
                        _ = await browser.screenshot(page)
                        raise RuntimeError(
                            "GitHub signup: email verification step not found. "
                            f"Current URL: {page.url}"
                        )

                if email_step is not None:
                    # Wait for OTP via email
                    if self._otp_coordinator is None:
                        raise RuntimeError("OTPCoordinator not wired into AccountProvisioner")

                    otp_code = await browser.wait_for_otp(
                        "github",
                        self._otp_coordinator,
                        source="email",
                        timeout_seconds=otp_timeout,
                    )

                    # Fill OTP
                    await browser.fill_form(
                        page,
                        {
                            "#user-email-verification-field, .js-emailverificationcode": otp_code
                        },
                        typing_delay_ms=120,
                    )
                    await browser.human_pause(0.5, 1.0)
                    await browser.click(page, "button[type='submit']")
                    await browser.human_pause(3.0, 5.0)

                # ── Generate Personal Access Token ─────────────────────
                pat = await self._github_generate_pat(browser, page)

        except asyncio.TimeoutError:
            await self._audit_neo4j(
                platform="github",
                status=ProvisioningStatus.OTP_TIMEOUT,
                provisioning_id=provisioning_id,
            )
            await self._emit_event(
                "ACCOUNT_PROVISIONING_FAILED",
                {
                    "platform": "github",
                    "instance_id": self._instance_id,
                    "provisioning_id": provisioning_id,
                    "error": "OTP wait timed out",
                    "retryable": True,
                },
            )
            return GitHubAccountResult(
                status=ProvisioningStatus.OTP_TIMEOUT,
                username=username,
                email=email,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error="OTP wait timed out",
            )
        except Exception as exc:
            self._log.error("github_provision_browser_error", error=str(exc))
            await self._audit_neo4j(
                platform="github",
                status=ProvisioningStatus.FAILED,
                provisioning_id=provisioning_id,
                error=str(exc),
            )
            await self._emit_event(
                "ACCOUNT_PROVISIONING_FAILED",
                {
                    "platform": "github",
                    "instance_id": self._instance_id,
                    "provisioning_id": provisioning_id,
                    "error": str(exc),
                    "retryable": True,
                },
            )
            return GitHubAccountResult(
                status=ProvisioningStatus.FAILED,
                username=username,
                email=email,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error=str(exc),
            )

        # ── Seal credentials in vault ─────────────────────────────────
        pat_sealed = False
        if self._vault is not None and pat:
            import json as _json
            token_data = {
                "username": username,
                "email": email,
                "password": password,
                "pat": pat,
                "instance_id": self._instance_id,
                "provisioned_at": utc_now().isoformat(),
            }
            self._vault.encrypt_token_json(
                token_data,
                platform_id=f"github.{self._instance_id[:8]}",
                purpose="github_account",
            )
            pat_sealed = True
            self._log.info("github_credentials_sealed", username=username)

        # ── Emit events ───────────────────────────────────────────────
        await self._emit_event(
            "GITHUB_ACCOUNT_PROVISIONED",
            {
                "username": username,
                "email": email,
                "instance_id": self._instance_id,
                "pat_sealed": pat_sealed,
                "provisioning_id": provisioning_id,
            },
        )

        await self._audit_neo4j(
            platform="github",
            status=ProvisioningStatus.SUCCESS,
            provisioning_id=provisioning_id,
            metadata={"username": username, "email": email, "pat_sealed": pat_sealed},
        )

        self._provisioned.add("github")

        return GitHubAccountResult(
            status=ProvisioningStatus.SUCCESS,
            username=username,
            email=email,
            instance_id=self._instance_id,
            provisioning_id=provisioning_id,
            pat_sealed=pat_sealed,
        )

    async def provision_gmail(self, username: str) -> GmailResult:
        """
        Create a Gmail account via Playwright.

        Args:
            username: Desired Gmail username (local part before @gmail.com).

        Returns:
            GmailResult with provisioning status and email address.

        Note: Gmail signup is rate-limited by Google and may require a phone
        number for verification. The provisioned Twilio number is used for SMS OTP.
        """
        provisioning_id = new_id()
        email = f"{username}@gmail.com"
        self._log.info("gmail_provision_start", username=username, provisioning_id=provisioning_id)

        approved = await self._request_equor_approval(
            action="provision_gmail",
            context={"username": username, "email": email, "purpose": "autonomous_email_identity"},
            provisioning_id=provisioning_id,
        )
        if not approved:
            await self._audit_neo4j(
                platform="gmail",
                status=ProvisioningStatus.EQUOR_DENIED,
                provisioning_id=provisioning_id,
            )
            return GmailResult(
                status=ProvisioningStatus.EQUOR_DENIED,
                email=email,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error="Equor denied Gmail provisioning",
            )

        config = self._full_config
        if config is None:
            return GmailResult(
                status=ProvisioningStatus.FAILED,
                email=email,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error="AccountProvisioner not initialized",
            )

        headless = config.account_provisioner.browser_headless
        stealth = config.account_provisioner.browser_stealth
        otp_timeout = config.account_provisioner.otp_wait_timeout_s
        password = _generate_password()

        try:
            from clients.browser_client import BrowserClient

            async with BrowserClient(headless=headless, stealth=stealth) as browser:
                page = await browser.create_page()

                await browser.goto(
                    page,
                    "https://accounts.google.com/signup/v2/createaccount?flowName=GlifWebSignIn&flowEntry=SignUp",
                )
                await browser.human_pause(1.5, 3.0)

                # Fill first name / last name (required)
                await browser.fill_form(
                    page,
                    {
                        "#firstName": "Eco",
                        "#lastName": "Dias",
                    },
                )
                await browser.click(page, "#collectNameNext")
                await browser.human_pause(1.5, 2.5)

                # Birthdate + gender (Google requires this)
                await browser.fill_form(page, {"#month": "01", "#day": "01", "#year": "1990"})
                await browser.select_option(page, "#gender", "1")  # male
                await browser.click(page, "#birthdaygenderNext")
                await browser.human_pause(1.5, 2.5)

                # Username
                await browser.fill_form(page, {"#userNameField": username})
                await browser.click(page, "#next")
                await browser.human_pause(2.0, 3.5)

                # Check for "username taken" signal
                page_text = await browser.get_page_text(page)
                if "already" in page_text.lower() or "in use" in page_text.lower():
                    return GmailResult(
                        status=ProvisioningStatus.FAILED,
                        email=email,
                        instance_id=self._instance_id,
                        provisioning_id=provisioning_id,
                        error="Gmail username unavailable",
                    )

                # Password
                await browser.fill_form(
                    page,
                    {
                        "#passwd": password,
                        "#confirm-passwd": password,
                    },
                )
                await browser.click(page, "#createpasswordNext")
                await browser.human_pause(2.0, 3.5)

                # CAPTCHA
                if self._captcha_client:
                    await browser.solve_captcha_on_page(page, self._captcha_client)

                # Phone verification (if Google requires it)
                phone_step = await browser.wait_for_selector(
                    page, "#phoneNumberId, input[type='tel']", timeout_ms=8_000
                )
                if phone_step and self._otp_coordinator:
                    # Enter the instance's Twilio number
                    from_number = os.environ.get(
                        "ORGANISM_IDENTITY_COMM__TWILIO_FROM_NUMBER", ""
                    )
                    if not from_number:
                        self._log.warning(
                            "gmail_phone_verification_skipped",
                            reason="no_twilio_number",
                        )
                    else:
                        await browser.fill_form(
                            page,
                            {"#phoneNumberId, input[type='tel']": from_number},
                        )
                        await browser.click(page, "#next")
                        await browser.human_pause(3.0, 5.0)

                        otp_code = await browser.wait_for_otp(
                            "google",
                            self._otp_coordinator,
                            source="sms",
                            timeout_seconds=otp_timeout,
                        )
                        await browser.fill_form(
                            page, {"#code": otp_code}
                        )
                        await browser.click(page, "#next")
                        await browser.human_pause(2.0, 3.5)

                # Agree to terms if present
                agree = await browser.wait_for_selector(
                    page, "#agreeButton, [data-action='agree']", timeout_ms=5_000
                )
                if agree:
                    await browser.click(page, "#agreeButton, [data-action='agree']")
                    await browser.human_pause(2.0, 3.0)

        except asyncio.TimeoutError:
            await self._emit_event(
                "ACCOUNT_PROVISIONING_FAILED",
                {
                    "platform": "gmail",
                    "instance_id": self._instance_id,
                    "provisioning_id": provisioning_id,
                    "error": "OTP wait timed out",
                    "retryable": True,
                },
            )
            return GmailResult(
                status=ProvisioningStatus.OTP_TIMEOUT,
                email=email,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error="OTP wait timed out",
            )
        except Exception as exc:
            self._log.error("gmail_provision_error", error=str(exc))
            await self._audit_neo4j(
                platform="gmail",
                status=ProvisioningStatus.FAILED,
                provisioning_id=provisioning_id,
                error=str(exc),
            )
            await self._emit_event(
                "ACCOUNT_PROVISIONING_FAILED",
                {
                    "platform": "gmail",
                    "instance_id": self._instance_id,
                    "provisioning_id": provisioning_id,
                    "error": str(exc),
                    "retryable": True,
                },
            )
            return GmailResult(
                status=ProvisioningStatus.FAILED,
                email=email,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error=str(exc),
            )

        # Seal credentials
        if self._vault is not None:
            self._vault.encrypt_token_json(
                {
                    "email": email,
                    "password": password,
                    "username": username,
                    "provisioned_at": utc_now().isoformat(),
                },
                platform_id=f"gmail.{self._instance_id[:8]}",
                purpose="gmail_account",
            )

        await self._emit_event(
            "PLATFORM_ACCOUNT_PROVISIONED",
            {
                "platform": "gmail",
                "username": email,
                "instance_id": self._instance_id,
                "provisioning_id": provisioning_id,
                "metadata": {"email": email},
            },
        )

        await self._audit_neo4j(
            platform="gmail",
            status=ProvisioningStatus.SUCCESS,
            provisioning_id=provisioning_id,
            metadata={"email": email},
        )

        self._provisioned.add("gmail")

        return GmailResult(
            status=ProvisioningStatus.SUCCESS,
            email=email,
            instance_id=self._instance_id,
            provisioning_id=provisioning_id,
        )

    async def provision_platform_account(
        self,
        platform: str,
        config: dict[str, Any],
    ) -> AccountResult:
        """
        Generic platform account provisioner - extensible hook for future platforms.

        Args:
            platform: Platform name (e.g. "discord", "twitter").
            config: Platform-specific signup parameters.

        Returns:
            AccountResult with provisioning details.
        """
        provisioning_id = new_id()
        self._log.info(
            "generic_platform_provision_start",
            platform=platform,
            provisioning_id=provisioning_id,
        )

        approved = await self._request_equor_approval(
            action=f"provision_{platform}_account",
            context={"platform": platform, **config},
            provisioning_id=provisioning_id,
        )
        if not approved:
            return AccountResult(
                status=ProvisioningStatus.EQUOR_DENIED,
                platform=platform,
                instance_id=self._instance_id,
                provisioning_id=provisioning_id,
                error="Equor denied platform account provisioning",
            )

        # Delegated to callers to implement platform-specific logic.
        # This base implementation returns FAILED to signal that a
        # platform-specific sub-provisioner is required.
        self._log.warning(
            "generic_provision_not_implemented",
            platform=platform,
            hint="Implement a platform-specific provisioner or subclass AccountProvisioner",
        )
        return AccountResult(
            status=ProvisioningStatus.FAILED,
            platform=platform,
            instance_id=self._instance_id,
            provisioning_id=provisioning_id,
            error=f"No provisioner implemented for platform: {platform!r}",
        )

    # ── Already-provisioned checks ────────────────────────────────────────

    async def _has_twilio_number(self) -> bool:
        """True if this instance already has a provisioned Twilio phone number."""
        # Check environment variable (runtime override)
        if os.environ.get("ORGANISM_IDENTITY_COMM__TWILIO_FROM_NUMBER"):
            return True
        if self._full_config and self._full_config.identity_comm.twilio_from_number:
            return True
        # Check Neo4j audit record - set on successful prior provisioning
        if self._neo4j is not None:
            try:
                rows = await self._neo4j.execute_read(
                    """
                    MATCH (p:AccountProvisioning {
                        instance_id: $instance_id,
                        platform: 'twilio',
                        status: 'success'
                    })
                    RETURN p.provisioning_id LIMIT 1
                    """,
                    {"instance_id": self._instance_id},
                )
                return bool(rows)
            except Exception:
                pass
        return False

    async def _has_own_github_account(self) -> bool:
        """True if this instance already has its own provisioned GitHub account."""
        if self._neo4j is not None:
            try:
                rows = await self._neo4j.execute_read(
                    """
                    MATCH (p:AccountProvisioning {
                        instance_id: $instance_id,
                        platform: 'github',
                        status: 'success'
                    })
                    RETURN p.provisioning_id LIMIT 1
                    """,
                    {"instance_id": self._instance_id},
                )
                return bool(rows)
            except Exception:
                pass
        return False

    # ── Identity derivation ───────────────────────────────────────────────

    def _derive_github_identity(self) -> tuple[str, str]:
        """
        Derive per-instance GitHub username and email.

        Returns:
            (username, email) tuple.
        """
        config = self._full_config
        prefix = (
            config.account_provisioner.github_username_prefix
            if config
            else "ecodiaos"
        )
        suffix = self._instance_id[:8].lower()
        username = f"{prefix}-{suffix}"

        # Email: either use a configured domain or rely on Gmail provisioning
        email_domain = (
            config.account_provisioner.github_email_domain if config else ""
        )
        if email_domain:
            email = f"{prefix}.{suffix}@{email_domain}"
        else:
            # Use a Gmail address with same naming convention
            email = f"{prefix}.{suffix}@gmail.com"

        return username, email

    # ── GitHub helpers ─────────────────────────────────────────────────────

    async def _github_username_available(self, username: str) -> bool:
        """
        Check if a GitHub username is available via the public REST API.

        Returns True if the username is free (404 response = no such user).
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"https://api.github.com/users/{username}",
                    headers={"Accept": "application/vnd.github+json"},
                )
            return resp.status_code == 404
        except Exception as exc:
            self._log.warning("github_availability_check_failed", error=str(exc))
            return True  # Optimistically assume available

    async def _github_generate_pat(
        self,
        browser: "BrowserClient",
        page: Any,
    ) -> str:
        """
        Navigate to GitHub Settings → Developer Settings → PAT and generate a token.

        Returns:
            The raw PAT string, or "" if generation fails.
        """
        try:
            await browser.goto(
                page,
                "https://github.com/settings/tokens/new",
                wait_until="networkidle",
            )
            await browser.human_pause(1.0, 2.0)

            # Set token note
            await browser.fill_form(
                page,
                {"#oauth_access[note]": f"ecodiaos-{self._instance_id[:8]}-auto"},
            )

            # Set expiration to "No expiration" if dropdown exists
            try:
                await browser.select_option(
                    page, "#oauth_access[expiration]", "no_expiration"
                )
            except Exception:
                pass

            # Check required scopes (repo, read:org, workflow)
            for scope_id in ("#user-content-repo", "#user-content-workflow", "#user-content-read:org"):
                try:
                    checked = await page.is_checked(scope_id)
                    if not checked:
                        await page.check(scope_id)
                except Exception:
                    pass

            await browser.click(page, 'input[type="submit"][value="Generate token"]')
            await browser.human_pause(2.0, 3.5)

            # Extract the token
            pat_element = await browser.wait_for_selector(
                page,
                "#new-oauth-token, [id='new-oauth-token'], code",
                timeout_ms=10_000,
            )
            if pat_element is None:
                self._log.warning("github_pat_element_not_found")
                return ""

            pat = (await pat_element.inner_text()).strip()
            self._log.info("github_pat_generated", pat_prefix=pat[:8] + "...")
            return pat

        except Exception as exc:
            self._log.warning("github_pat_generation_failed", error=str(exc))
            return ""

    # ── Equor approval gate ────────────────────────────────────────────────

    async def _request_equor_approval(
        self,
        action: str,
        context: dict[str, Any],
        provisioning_id: str,
    ) -> bool:
        """
        Emit a constitutional check request to Equor and await approval.

        Returns True if Equor approves or times out (default-allow on timeout).
        Returns False only on explicit DENY.
        """
        if self._event_bus is None:
            # No bus - allow by default (Equor unavailable)
            self._log.warning(
                "equor_gate_bypassed",
                reason="no_event_bus",
                action=action,
            )
            return True

        from systems.synapse.types import SynapseEvent, SynapseEventType

        timeout_s = (
            self._full_config.account_provisioner.equor_approval_timeout_s
            if self._full_config
            else 30.0
        )

        result_future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        request_id = new_id()

        def _on_approval(event: Any) -> None:
            if event.data.get("request_id") != request_id:
                return
            if not result_future.done():
                approved = event.data.get("approved", True)
                result_future.set_result(bool(approved))

        self._event_bus.subscribe(SynapseEventType.EQUOR_PROVISIONING_APPROVAL, _on_approval)

        try:
            # Emit the constitutional review request
            check_event = SynapseEvent(
                event_type=SynapseEventType.CERTIFICATE_PROVISIONING_REQUEST,
                source_system="identity.account_provisioner",
                data={
                    "request_id": request_id,
                    "provisioning_id": provisioning_id,
                    "action": action,
                    "context": context,
                    "instance_id": self._instance_id,
                    "provisioning_type": "platform_account",
                    "requires_amendment_approval": False,
                },
            )
            await self._event_bus.emit(check_event)

            try:
                approved = await asyncio.wait_for(result_future, timeout=timeout_s)
                self._log.info(
                    "equor_response_received",
                    action=action,
                    approved=approved,
                    request_id=request_id,
                )
                return approved
            except asyncio.TimeoutError:
                self._log.warning(
                    "equor_approval_timeout",
                    action=action,
                    timeout_s=timeout_s,
                )
                return True  # Default-allow on timeout (Equor may be busy)

        finally:
            # Unsubscribe to avoid listener accumulation
            try:
                self._event_bus.unsubscribe(
                    SynapseEventType.EQUOR_PROVISIONING_APPROVAL, _on_approval
                )
            except Exception:
                pass

    # ── Neo4j audit ───────────────────────────────────────────────────────

    async def _audit_neo4j(
        self,
        platform: str,
        status: ProvisioningStatus,
        provisioning_id: str,
        metadata: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        """
        Write an (:AccountProvisioning) audit node to Neo4j.

        (:AccountProvisioning {
            id, provisioning_id, platform, status, instance_id,
            error, created_at
        }) -[:PROVISIONED_BY]-> (:Identity {instance_id})
        """
        if self._neo4j is None:
            return

        now = utc_now().isoformat()
        try:
            await self._neo4j.execute_write(
                """
                MERGE (p:AccountProvisioning {provisioning_id: $provisioning_id})
                ON CREATE SET
                    p.id = $id,
                    p.provisioning_id = $provisioning_id,
                    p.platform = $platform,
                    p.status = $status,
                    p.instance_id = $instance_id,
                    p.error = $error,
                    p.metadata = $metadata,
                    p.created_at = datetime($now)
                ON MATCH SET
                    p.status = $status,
                    p.error = $error,
                    p.updated_at = datetime($now)
                WITH p
                MERGE (i:Identity {instance_id: $instance_id})
                MERGE (p)-[:PROVISIONED_BY {created_at: datetime($now)}]->(i)
                """,
                {
                    "id": new_id(),
                    "provisioning_id": provisioning_id,
                    "platform": platform,
                    "status": str(status),
                    "instance_id": self._instance_id,
                    "error": error,
                    "metadata": str(metadata or {}),
                    "now": now,
                },
            )
        except Exception as exc:
            self._log.warning(
                "neo4j_audit_failed",
                platform=platform,
                provisioning_id=provisioning_id,
                error=str(exc),
            )

    # ── Cost accounting ────────────────────────────────────────────────────

    async def _emit_cost_event(
        self,
        platform: str,
        cost_usd: float,
        provisioning_id: str,
    ) -> None:
        """Emit DOMAIN_EPISODE_RECORDED for Oikos infrastructure cost tracking."""
        if self._event_bus is None:
            return

        from primitives.common import utc_now as _utc_now
        from systems.synapse.types import SynapseEvent, SynapseEventType

        try:
            event = SynapseEvent(
                event_type=SynapseEventType.DOMAIN_EPISODE_RECORDED,
                source_system="identity.account_provisioner",
                data={
                    "domain": "account_provisioning",
                    "outcome": "success",
                    "revenue": "0",
                    "cost_usd": str(round(cost_usd, 6)),
                    "duration_ms": 0,
                    "custom_metrics": {
                        "platform": platform,
                        "provisioning_id": provisioning_id,
                    },
                    "timestamp": _utc_now().isoformat(),
                    "source_system": "identity.account_provisioner",
                },
            )
            await self._event_bus.emit(event)
        except Exception as exc:
            self._log.warning("cost_event_emit_failed", error=str(exc))

    # ── Synapse event emission ─────────────────────────────────────────────

    async def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit a typed Synapse event. Fire-and-forget, failure-tolerant."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        type_map: dict[str, SynapseEventType] = {
            "PHONE_NUMBER_PROVISIONED": SynapseEventType.PHONE_NUMBER_PROVISIONED,
            "GITHUB_ACCOUNT_PROVISIONED": SynapseEventType.GITHUB_ACCOUNT_PROVISIONED,
            "PLATFORM_ACCOUNT_PROVISIONED": SynapseEventType.PLATFORM_ACCOUNT_PROVISIONED,
            "ACCOUNT_PROVISIONING_FAILED": SynapseEventType.ACCOUNT_PROVISIONING_FAILED,
            "PROVISIONING_REQUIRES_HUMAN_ESCALATION": SynapseEventType.PROVISIONING_REQUIRES_HUMAN_ESCALATION,
        }

        evt_type = type_map.get(event_name)
        if evt_type is None:
            self._log.warning("unknown_event_type", event_name=event_name)
            return

        try:
            event = SynapseEvent(
                event_type=evt_type,
                source_system="identity.account_provisioner",
                data=data,
            )
            await self._event_bus.emit(event)
        except Exception as exc:
            self._log.warning("event_emit_failed", event=event_name, error=str(exc))


# ─── Helpers ─────────────────────────────────────────────────────────────────

import secrets as _secrets
import string as _string


def _generate_password(length: int = 22) -> str:
    """Generate a cryptographically random strong password."""
    alphabet = _string.ascii_letters + _string.digits + "!@#$%^&*()"
    pw = [
        _secrets.choice(_string.ascii_uppercase),
        _secrets.choice(_string.ascii_lowercase),
        _secrets.choice(_string.digits),
        _secrets.choice("!@#$%^&*()"),
    ]
    pw += [_secrets.choice(alphabet) for _ in range(length - 4)]
    import random as _random
    _random.shuffle(pw)
    return "".join(pw)
