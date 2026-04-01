"""
EcodiaOS - Playwright Browser Client

Async headless browser executor for form-based account creation flows.
Uses Playwright with optional playwright-stealth for bot detection evasion.

Key features:
  - Stealth mode: randomised viewport, user-agent, human-like typing delays
  - CAPTCHA integration: delegates to CaptchaClient on detected challenges
  - OTP integration: delegates to OTPCoordinator for code receipt
  - Screenshot audit trail: captures failures + completions for Neo4j storage

Dependencies (install once, lazy-imported):
  pip install playwright playwright-stealth
  playwright install chromium

Configuration:
  ECODIAOS_ACCOUNT_PROVISIONER__BROWSER_HEADLESS  - default True
  ECODIAOS_ACCOUNT_PROVISIONER__BROWSER_STEALTH   - default True

Thread-safety: NOT thread-safe. Single-threaded asyncio like all EOS clients.
"""

from __future__ import annotations

import asyncio
import random
import secrets
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.captcha_client import CaptchaClient
    from systems.identity.communication import OTPCoordinator

logger = structlog.get_logger("identity.browser_client")

# ─── Viewport pool (realistic monitor sizes) ────────────────────────────────

_VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 1280, "height": 800},
    {"width": 1536, "height": 864},
]

# ─── User-agent pool ────────────────────────────────────────────────────────

_USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

# ─── Result models ──────────────────────────────────────────────────────────


class PageLoadResult:
    """Result of navigating to a page."""

    def __init__(self, url: str, status: int, title: str) -> None:
        self.url = url
        self.status = status
        self.title = title

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400


class FormFillResult:
    """Result of a form fill operation."""

    def __init__(self, fields_filled: int, errors: list[str]) -> None:
        self.fields_filled = fields_filled
        self.errors = errors

    @property
    def ok(self) -> bool:
        return not self.errors


# ─── BrowserClient ──────────────────────────────────────────────────────────


class BrowserClient:
    """
    Async Playwright browser for form-based account creation.

    Usage pattern:
        async with BrowserClient(headless=True, stealth=True) as browser:
            page = await browser.create_page()
            await browser.goto(page, "https://github.com/signup")
            await browser.fill_form(page, {"email": "x@example.com"})
            solved = await browser.solve_captcha_on_page(page, captcha_client)
            code = await browser.wait_for_otp("github", otp_coordinator, timeout_s=300)
    """

    def __init__(self, headless: bool = True, stealth: bool = True) -> None:
        self._headless = headless
        self._stealth = stealth
        self._playwright: Any = None
        self._browser: Any = None
        self._log = logger.bind(headless=headless, stealth=stealth)

    # ── Context manager ────────────────────────────────────────────────────

    async def __aenter__(self) -> "BrowserClient":
        await self._launch()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._close()

    async def _launch(self) -> None:
        """Launch Playwright Chromium browser."""
        try:
            from playwright.async_api import async_playwright  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            ) from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )
        self._log.info("browser_launched")

    async def _close(self) -> None:
        """Close browser and Playwright instance."""
        try:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as exc:
            self._log.warning("browser_close_error", error=str(exc))

    # ── Page creation ──────────────────────────────────────────────────────

    async def create_page(self) -> Any:
        """
        Create a new browser page with stealth patches applied.

        Returns:
            Playwright Page object.
        """
        if self._browser is None:
            raise RuntimeError("BrowserClient must be used as async context manager")

        viewport = random.choice(_VIEWPORTS)
        user_agent = random.choice(_USER_AGENTS)

        context = await self._browser.new_context(
            viewport=viewport,
            user_agent=user_agent,
            locale="en-US",
            timezone_id="America/New_York",
            java_script_enabled=True,
            # Prevent navigator.webdriver leakage
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        # Inject stealth JS patches to mask automation signals
        await context.add_init_script("""
            // Override navigator.webdriver
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            // Mask Playwright-specific chrome.runtime.onConnect
            if (window.chrome && window.chrome.runtime) {
                Object.defineProperty(window.chrome, 'csi', {get: () => () => {}});
            }
            // Consistent plugin length
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """)

        page = await context.new_page()

        # Apply playwright-stealth if available and requested
        if self._stealth:
            try:
                from playwright_stealth import stealth_async  # type: ignore[import]
                await stealth_async(page)
                self._log.debug("stealth_applied")
            except ImportError:
                self._log.warning(
                    "playwright_stealth_unavailable",
                    hint="pip install playwright-stealth",
                )

        return page

    # ── Navigation ─────────────────────────────────────────────────────────

    async def goto(
        self,
        page: Any,
        url: str,
        wait_until: str = "networkidle",
        timeout_ms: int = 30_000,
    ) -> PageLoadResult:
        """Navigate to a URL and wait for the page to settle."""
        try:
            response = await page.goto(
                url,
                wait_until=wait_until,
                timeout=timeout_ms,
            )
            status = response.status if response else 200
            title = await page.title()
            self._log.info("page_loaded", url=url, status=status, title=title[:60])
            return PageLoadResult(url=url, status=status, title=title)
        except Exception as exc:
            self._log.error("page_load_failed", url=url, error=str(exc))
            raise

    # ── Form filling ───────────────────────────────────────────────────────

    async def fill_form(
        self,
        page: Any,
        fields: dict[str, str],
        typing_delay_ms: float = 50.0,
        field_pause_ms: float = 300.0,
    ) -> FormFillResult:
        """
        Fill form fields with human-like typing delays.

        Args:
            page: Playwright Page.
            fields: Mapping of CSS selector → value to type.
            typing_delay_ms: Milliseconds between keystrokes (jitter added).
            field_pause_ms: Milliseconds to pause between different fields.

        Returns:
            FormFillResult with count of filled fields and any errors.
        """
        errors: list[str] = []
        filled = 0

        for selector, value in fields.items():
            try:
                element = await page.wait_for_selector(selector, timeout=10_000)
                if element is None:
                    errors.append(f"selector not found: {selector!r}")
                    continue

                # Click to focus
                await element.click()
                await asyncio.sleep(random.uniform(0.1, 0.3))

                # Clear existing value
                await element.triple_click()
                await page.keyboard.press("Backspace")

                # Type with jitter
                for char in value:
                    await page.keyboard.type(char)
                    jitter = random.uniform(0.5, 1.8) * (typing_delay_ms / 1000.0)
                    await asyncio.sleep(jitter)

                filled += 1
                self._log.debug("field_filled", selector=selector, length=len(value))

                # Pause between fields
                await asyncio.sleep(random.uniform(0.2, 0.5) * (field_pause_ms / 1000.0))

            except Exception as exc:
                errors.append(f"{selector}: {exc}")
                self._log.warning("field_fill_error", selector=selector, error=str(exc))

        return FormFillResult(fields_filled=filled, errors=errors)

    async def click(
        self,
        page: Any,
        selector: str,
        timeout_ms: int = 10_000,
    ) -> bool:
        """
        Click an element. Returns True on success.
        """
        try:
            element = await page.wait_for_selector(selector, timeout=timeout_ms)
            if element is None:
                return False
            # Small random delay before clicking
            await asyncio.sleep(random.uniform(0.1, 0.4))
            await element.click()
            return True
        except Exception as exc:
            self._log.warning("click_failed", selector=selector, error=str(exc))
            return False

    async def wait_for_selector(
        self,
        page: Any,
        selector: str,
        timeout_ms: int = 10_000,
    ) -> Any | None:
        """Wait for a CSS selector to appear. Returns element or None."""
        try:
            return await page.wait_for_selector(selector, timeout=timeout_ms)
        except Exception:
            return None

    # ── CAPTCHA detection & solving ────────────────────────────────────────

    async def solve_captcha_on_page(
        self,
        page: Any,
        captcha_client: "CaptchaClient",
    ) -> bool:
        """
        Detect and solve any CAPTCHA present on the current page.

        Checks for reCAPTCHA v2/v3 and hCaptcha. Returns True if a CAPTCHA
        was found and solved (or if none was present). Returns False on failure.

        The solved token is injected back into the page's response textarea.
        """
        page_url = page.url
        self._log.info("captcha_scan_start", url=page_url)

        try:
            # ── reCAPTCHA v2 ────────────────────────────────────────────
            rc2 = await page.evaluate("""
                () => {
                    const el = document.querySelector('[data-sitekey]');
                    if (!el) return null;
                    // Distinguish hCaptcha from reCAPTCHA by script presence
                    const scripts = [...document.scripts].map(s => s.src);
                    const isHCaptcha = scripts.some(s => s.includes('hcaptcha'));
                    if (isHCaptcha) return null;
                    return el.getAttribute('data-sitekey');
                }
            """)
            if rc2:
                self._log.info("recaptcha_v2_detected", site_key=rc2[:20] + "...")
                token = await captcha_client.solve_recaptcha_v2(rc2, page_url)
                await self._inject_recaptcha_token(page, token)
                return True

            # ── hCaptcha ────────────────────────────────────────────────
            hc = await page.evaluate("""
                () => {
                    const scripts = [...document.scripts].map(s => s.src);
                    if (!scripts.some(s => s.includes('hcaptcha'))) return null;
                    const el = document.querySelector('[data-sitekey]');
                    return el ? el.getAttribute('data-sitekey') : null;
                }
            """)
            if hc:
                self._log.info("hcaptcha_detected", site_key=hc[:20] + "...")
                token = await captcha_client.solve_hcaptcha(hc, page_url)
                await self._inject_hcaptcha_token(page, token)
                return True

            self._log.debug("no_captcha_detected")
            return True  # No CAPTCHA present - success

        except Exception as exc:
            self._log.error("captcha_solve_failed", error=str(exc))
            return False

    async def _inject_recaptcha_token(self, page: Any, token: str) -> None:
        """Inject solved reCAPTCHA token into the page."""
        await page.evaluate(
            """(token) => {
                const el = document.getElementById('g-recaptcha-response');
                if (el) el.value = token;
                // Also trigger the callback if registered
                if (typeof ___grecaptcha_cfg !== 'undefined') {
                    const clients = ___grecaptcha_cfg.clients;
                    for (const c in clients) {
                        const client = clients[c];
                        for (const k in client) {
                            if (client[k] && typeof client[k].callback === 'function') {
                                client[k].callback(token);
                            }
                        }
                    }
                }
            }""",
            token,
        )

    async def _inject_hcaptcha_token(self, page: Any, token: str) -> None:
        """Inject solved hCaptcha token into the page."""
        await page.evaluate(
            """(token) => {
                const el = document.querySelector('[name="h-captcha-response"]');
                if (el) el.value = token;
                // Trigger hcaptcha callback
                if (typeof hcaptcha !== 'undefined') {
                    const wid = hcaptcha.getResponse ? 0 : null;
                    if (wid !== null && hcaptcha.execute) {
                        try { hcaptcha.execute(wid, {response: token}); } catch(e) {}
                    }
                }
            }""",
            token,
        )

    # ── OTP receipt ────────────────────────────────────────────────────────

    async def wait_for_otp(
        self,
        platform: str,
        otp_coordinator: "OTPCoordinator",
        source: str = "any",
        timeout_seconds: int = 300,
    ) -> str:
        """
        Wait for an OTP code to arrive via SMS, Telegram, or IMAP.

        Delegates to OTPCoordinator.wait_for_otp() which polls all configured
        channels (Twilio, Telegram, IMAP) and resolves when a code arrives.

        Args:
            platform: Platform name hint (e.g. "github", "gmail").
            otp_coordinator: Shared OTPCoordinator instance.
            source: Expected source channel ("sms" | "email" | "telegram" | "any").
            timeout_seconds: Seconds before giving up.

        Returns:
            The OTP code string.

        Raises:
            asyncio.TimeoutError: If no code arrives within timeout_seconds.
        """
        self._log.info(
            "waiting_for_otp",
            platform=platform,
            source=source,
            timeout_s=timeout_seconds,
        )
        code = await otp_coordinator.wait_for_otp(
            platform=platform,
            source=source,
            timeout=timeout_seconds,
        )
        self._log.info("otp_received", platform=platform, source=source)
        return code

    # ── Screenshot ─────────────────────────────────────────────────────────

    async def screenshot(self, page: Any, full_page: bool = False) -> bytes:
        """
        Take a screenshot of the current page.

        Args:
            page: Playwright Page.
            full_page: If True, capture the full scrollable page.

        Returns:
            PNG image bytes.
        """
        try:
            return await page.screenshot(type="png", full_page=full_page)
        except Exception as exc:
            self._log.warning("screenshot_failed", error=str(exc))
            return b""

    # ── Utilities ──────────────────────────────────────────────────────────

    async def get_page_text(self, page: Any) -> str:
        """Extract visible text content from the current page."""
        try:
            return await page.evaluate("() => document.body.innerText")
        except Exception:
            return ""

    async def get_input_value(self, page: Any, selector: str) -> str:
        """Read the current value of an input field."""
        try:
            return await page.input_value(selector)
        except Exception:
            return ""

    async def select_option(
        self, page: Any, selector: str, value: str
    ) -> bool:
        """Select a dropdown option by value. Returns True on success."""
        try:
            await page.select_option(selector, value=value)
            return True
        except Exception as exc:
            self._log.warning("select_option_failed", selector=selector, error=str(exc))
            return False

    async def human_pause(
        self, min_s: float = 0.5, max_s: float = 2.0
    ) -> None:
        """Pause for a human-like duration between actions."""
        await asyncio.sleep(random.uniform(min_s, max_s))

    @staticmethod
    def generate_password(length: int = 20) -> str:
        """
        Generate a cryptographically random strong password.

        Returns a password with mixed case, digits, and symbols that meets
        GitHub, Gmail, and most platform requirements.
        """
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*()"
        # Ensure at least one of each required character class
        pw = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
            secrets.choice("!@#$%^&*()"),
        ]
        pw += [secrets.choice(alphabet) for _ in range(length - 4)]
        random.shuffle(pw)
        return "".join(pw)
