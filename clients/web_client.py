"""
EcodiaOS - Web Intelligence Client

First-class real-time web data gathering for EOS.

Capabilities:
  search_web()          - Brave Search API | SerpAPI | DuckDuckGo scrape
  fetch_page()          - HTML retrieval via httpx; optional Playwright JS rendering
  extract_structured()  - LLM-guided extraction against a caller-supplied schema
  monitor_url()         - Detect changes on a URL (hash-based diff)
  check_robots()        - robots.txt compliance (cache 24h per domain)

Legal / ethical invariants (non-negotiable):
  - robots.txt respected; forbidden domains never fetched
  - Rate limit: ≥1s between requests to same domain; hard ceiling 60 req/hr
  - No paywall scraping; no personal data harvesting
  - All results stored with source URL + timestamp (attribution)
  - Official APIs preferred over scraping whenever available

Config (via SearchConfig):
  ORGANISM_SEARCH__PROVIDER        = "brave" | "serpapi" | "ddg"
  ORGANISM_SEARCH__BRAVE_API_KEY   = <key>
  ORGANISM_SEARCH__SERPAPI_KEY     = <key>
  ORGANISM_SEARCH__REQUEST_TIMEOUT_S = 10.0
  ORGANISM_SEARCH__MAX_REQ_PER_DOMAIN_PER_HOUR = 60
  ORGANISM_SEARCH__RATE_LIMIT_S    = 1.0

Neo4j logging:
  Every fetch is written as a (:WebIntelligenceEvent) node.
  Crawl budget tracked per domain.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import urllib.parse
import urllib.robotparser
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from config import SearchConfig

logger = structlog.get_logger("clients.web_client")

# ─── Result types ─────────────────────────────────────────────────


class SearchResult(BaseModel):
    """One result from a web search."""

    title: str
    url: str
    snippet: str
    source: str  # "brave" | "serpapi" | "ddg" | "llm_fallback"
    rank: int = 0


class PageContent(BaseModel):
    """Content retrieved from a URL."""

    url: str
    title: str = ""
    text: str  # Main readable text (HTML stripped)
    html: str = ""  # Raw HTML (may be empty if too large)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status_code: int = 200
    content_hash: str = ""  # SHA-256 of text, for change detection


class ChangeReport(BaseModel):
    """Result of a monitored URL check."""

    url: str
    changed: bool
    previous_hash: str
    current_hash: str
    diff_summary: str = ""
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─── Internal helpers ─────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s{2,}")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _strip_html(html: str, max_chars: int = 50_000) -> tuple[str, str]:
    """Return (title, plain_text) from raw HTML."""
    title_match = _TITLE_RE.search(html)
    title = _TAG_RE.sub("", title_match.group(1)).strip() if title_match else ""

    # Remove script/style blocks before stripping tags
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return title, text[:max_chars]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


# ─── WebIntelligenceClient ────────────────────────────────────────


class WebIntelligenceClient:
    """
    Web intelligence client for EOS.

    Thread-safe for concurrent asyncio tasks; NOT thread-safe across OS threads.
    All fetches are rate-limited per domain.  robots.txt is checked and cached.
    All operations degrade gracefully - no method raises; errors are logged and
    reflected in the return value.
    """

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._http: httpx.AsyncClient | None = None

        # Per-domain rate limiting: timestamps of recent requests (sliding window)
        self._domain_timestamps: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=120)  # keep last 2 × max_req_per_hour timestamps
        )
        self._domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # robots.txt cache: domain → (RobotFileParser | None, expiry_ts)
        self._robots_cache: dict[str, tuple[urllib.robotparser.RobotFileParser | None, float]] = {}
        self._robots_ttl = 86_400.0  # 24 h

        # URL monitor: url → last content hash
        self._monitor_hashes: dict[str, str] = {}

        # Neo4j client (optional, injected after construction)
        self._neo4j: Neo4jClient | None = None

    def set_neo4j(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=self._config.request_timeout_s,
                headers={
                    "User-Agent": (
                        "EcodiaOS-WebIntelligence/1.0 "
                        "(respectful crawler; contact: eos@ecodia.org)"
                    ),
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                follow_redirects=True,
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── Rate limiting ──────────────────────────────────────────────

    async def _enforce_rate_limit(self, url: str) -> bool:
        """
        Enforce per-domain rate limiting.
        Returns False if the domain's hourly budget is exhausted.
        Waits at least config.rate_limit_s between consecutive requests.
        """
        domain = _domain(url)
        if not domain:
            return True

        async with self._domain_locks[domain]:
            now = time.monotonic()
            timestamps = self._domain_timestamps[domain]

            # Hourly budget check (sliding 3600s window)
            hour_ago = now - 3600.0
            recent = [t for t in timestamps if t > hour_ago]
            if len(recent) >= self._config.max_req_per_domain_per_hour:
                logger.warning(
                    "web_client.rate_limit_exceeded",
                    domain=domain,
                    count=len(recent),
                    limit=self._config.max_req_per_domain_per_hour,
                )
                return False

            # Minimum inter-request gap
            if timestamps:
                last = timestamps[-1]
                wait = self._config.rate_limit_s - (now - last)
                if wait > 0:
                    await asyncio.sleep(wait)

            self._domain_timestamps[domain].append(time.monotonic())
            return True

    # ── robots.txt ─────────────────────────────────────────────────

    async def check_robots(self, url: str) -> bool:
        """
        Return True if robots.txt permits scraping the given URL.
        Returns True on network error (fail-open, since we honour it when reachable).
        """
        domain = _domain(url)
        if not domain:
            return True

        now = time.monotonic()
        cached = self._robots_cache.get(domain)
        if cached is not None:
            parser, expiry = cached
            if now < expiry:
                if parser is None:
                    return True
                return parser.can_fetch("*", url)

        robots_url = f"{urllib.parse.urlparse(url).scheme}://{domain}/robots.txt"
        try:
            client = await self._client()
            resp = await client.get(robots_url, timeout=5.0)
            if resp.status_code == 200:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(resp.text.splitlines())
                self._robots_cache[domain] = (parser, now + self._robots_ttl)
                allowed = parser.can_fetch("*", url)
                if not allowed:
                    logger.info("web_client.robots_blocked", url=url, robots_url=robots_url)
                return allowed
            # 404 / 410 robots → allow
            self._robots_cache[domain] = (None, now + self._robots_ttl)
            return True
        except Exception as exc:
            logger.debug("web_client.robots_check_error", domain=domain, error=str(exc))
            self._robots_cache[domain] = (None, now + self._robots_ttl)
            return True  # fail-open

    # ── Page fetch ─────────────────────────────────────────────────

    async def fetch_page(self, url: str, render_js: bool = False) -> PageContent:
        """
        Fetch a page and return its text content.

        render_js=True uses Playwright for JS-heavy pages (requires playwright installed
        and config.render_js_enabled=True).  Falls back to plain httpx on any error.

        Raises nothing - returns a PageContent with status_code reflecting the error.
        """
        log = logger.bind(url=url, render_js=render_js)

        # robots.txt gate
        if not await self.check_robots(url):
            log.warning("web_client.robots_disallowed")
            return PageContent(
                url=url, text="[Blocked by robots.txt]", status_code=403
            )

        # Rate limit gate
        if not await self._enforce_rate_limit(url):
            return PageContent(
                url=url, text="[Rate limit exceeded]", status_code=429
            )

        # JS rendering path
        if render_js and self._config.render_js_enabled:
            result = await self._fetch_with_playwright(url)
            if result is not None:
                await self._log_neo4j_event(url, "fetch_page", result.status_code, len(result.text))
                return result

        # Plain HTTP path
        try:
            client = await self._client()
            resp = await client.get(url)
            html = resp.text
            title, text = _strip_html(html)
            content = PageContent(
                url=url,
                title=title,
                text=text,
                html=html[:200_000],  # cap stored HTML at 200 KB
                status_code=resp.status_code,
                content_hash=_content_hash(text),
            )
            await self._log_neo4j_event(url, "fetch_page", resp.status_code, len(text))
            log.debug("web_client.fetch_ok", status=resp.status_code, text_len=len(text))
            return content
        except Exception as exc:
            log.warning("web_client.fetch_error", error=str(exc))
            return PageContent(url=url, text=f"[Fetch error: {exc}]", status_code=0)

    async def _fetch_with_playwright(self, url: str) -> PageContent | None:
        """Use Playwright headless browser; returns None on any import/runtime error."""
        try:
            from playwright.async_api import async_playwright  # type: ignore[import]

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                resp = await page.goto(url, timeout=int(self._config.request_timeout_s * 1000))
                await page.wait_for_load_state("networkidle", timeout=5000)
                html = await page.content()
                await browser.close()

            title, text = _strip_html(html)
            status = resp.status if resp else 200
            return PageContent(
                url=url,
                title=title,
                text=text,
                html=html[:200_000],
                status_code=status,
                content_hash=_content_hash(text),
            )
        except Exception as exc:
            logger.debug("web_client.playwright_error", url=url, error=str(exc))
            return None

    # ── Search ─────────────────────────────────────────────────────

    async def search_web(
        self, query: str, num_results: int | None = None
    ) -> list[SearchResult]:
        """
        Search the web.  Provider selected by config.provider.
        Falls back to next provider in chain on failure: brave → serpapi → ddg.
        Returns empty list on complete failure (never raises).
        """
        n = num_results or self._config.default_num_results
        provider = self._config.provider.lower()

        providers = [provider]
        for fallback in ("brave", "serpapi", "ddg"):
            if fallback not in providers:
                providers.append(fallback)

        for p in providers:
            try:
                results = await self._search_with(p, query, n)
                if results:
                    logger.info(
                        "web_client.search_ok",
                        provider=p,
                        query=query[:60],
                        count=len(results),
                    )
                    await self._log_neo4j_event(
                        f"search://{p}", "search_web", 200, len(results)
                    )
                    return results
            except Exception as exc:
                logger.warning("web_client.search_error", provider=p, error=str(exc))
                continue

        logger.warning("web_client.all_providers_failed", query=query[:60])
        return []

    async def _search_with(
        self, provider: str, query: str, num_results: int
    ) -> list[SearchResult]:
        if provider == "brave":
            return await self._search_brave(query, num_results)
        if provider == "serpapi":
            return await self._search_serpapi(query, num_results)
        if provider == "ddg":
            return await self._search_ddg(query, num_results)
        raise ValueError(f"Unknown search provider: {provider}")

    async def _search_brave(self, query: str, num_results: int) -> list[SearchResult]:
        if not self._config.brave_api_key:
            raise ValueError("Brave API key not configured")

        client = await self._client()
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(num_results, 20)},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._config.brave_api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for i, item in enumerate(data.get("web", {}).get("results", [])[:num_results]):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    source="brave",
                    rank=i,
                )
            )
        return results

    async def _search_serpapi(self, query: str, num_results: int) -> list[SearchResult]:
        if not self._config.serpapi_key:
            raise ValueError("SerpAPI key not configured")

        client = await self._client()
        resp = await client.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "num": min(num_results, 10),
                "api_key": self._config.serpapi_key,
                "engine": "google",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for i, item in enumerate(data.get("organic_results", [])[:num_results]):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="serpapi",
                    rank=i,
                )
            )
        return results

    async def _search_ddg(self, query: str, num_results: int) -> list[SearchResult]:
        """
        DuckDuckGo lite HTML scrape (no API key required).
        Rate-limited to 1 req/s by _enforce_rate_limit.
        Parses the plain HTML response from the lite endpoint.
        """
        url = "https://lite.duckduckgo.com/lite/"
        if not await self._enforce_rate_limit(url):
            return []

        client = await self._client()
        resp = await client.post(
            url,
            data={"q": query},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        html = resp.text

        # Parse results from DDG lite HTML
        results: list[SearchResult] = []
        # DDG lite: result links are <a class="result-link">
        link_re = re.compile(
            r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        snip_re = re.compile(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)

        links = link_re.findall(html)
        snippets = [_TAG_RE.sub("", s).strip() for s in snip_re.findall(html)]

        for i, (href, title_html) in enumerate(links[:num_results]):
            title = _TAG_RE.sub("", title_html).strip()
            snippet = snippets[i] if i < len(snippets) else ""
            # DDG lite hrefs are redirect URLs; extract actual URL from uddg param
            actual_url = href
            if "uddg=" in href:
                try:
                    parsed = urllib.parse.urlparse(href)
                    params = urllib.parse.parse_qs(parsed.query)
                    actual_url = params.get("uddg", [href])[0]
                    actual_url = urllib.parse.unquote(actual_url)
                except Exception:
                    pass
            results.append(
                SearchResult(title=title, url=actual_url, snippet=snippet, source="ddg", rank=i)
            )

        return results

    # ── Structured extraction ──────────────────────────────────────

    async def extract_structured(
        self,
        url: str,
        schema: dict[str, Any],
        llm: Any = None,
    ) -> dict[str, Any]:
        """
        Fetch a page and extract structured data matching the given schema.

        schema: dict describing expected fields, e.g.:
          {"protocol": "str", "tvl_usd": "float", "apy": "float"}

        Uses LLM to extract; returns {} when LLM is unavailable.
        """
        page = await self.fetch_page(url)
        if not page.text or page.status_code not in range(200, 300):
            return {}

        if llm is None:
            return {}

        schema_str = "\n".join(f"  {k}: {v}" for k, v in schema.items())
        prompt = (
            f"Extract the following fields from the page content below.\n"
            f"Return ONLY a JSON object with these fields (null for missing values):\n"
            f"{schema_str}\n\n"
            f"URL: {url}\n"
            f"Content (truncated to 4000 chars):\n{page.text[:4000]}\n\n"
            f"Return only valid JSON, no explanation."
        )

        try:
            from clients.llm import Message

            resp = await llm.generate(
                system_prompt=None,
                messages=[Message("user", prompt)],
                max_tokens=500,
                temperature=0.0,
                cache_system="web_client",
                cache_method="extract_structured",
            )
            text = resp.text if hasattr(resp, "text") else str(resp)

            # Parse JSON from response
            import json

            # Strip markdown fences if present
            text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
            text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
            return json.loads(text.strip())
        except Exception as exc:
            logger.warning("web_client.extract_structured_error", url=url, error=str(exc))
            return {}

    # ── URL monitoring ─────────────────────────────────────────────

    async def monitor_url(
        self, url: str, check_interval_hours: int | None = None
    ) -> ChangeReport:
        """
        Check a URL for content changes since the last call.

        Uses SHA-256 hash of extracted text for change detection.
        The first call always returns changed=True (baseline establishment).
        """
        page = await self.fetch_page(url)
        current_hash = page.content_hash or _content_hash(page.text)
        previous_hash = self._monitor_hashes.get(url, "")

        changed = previous_hash != current_hash and bool(previous_hash)
        if not previous_hash:
            changed = True  # baseline: signal on first check

        self._monitor_hashes[url] = current_hash

        report = ChangeReport(
            url=url,
            changed=changed,
            previous_hash=previous_hash,
            current_hash=current_hash,
            diff_summary=(
                f"Content changed at {url}" if changed else "No change detected"
            ),
        )

        if changed:
            logger.info(
                "web_client.url_changed",
                url=url,
                previous_hash=previous_hash,
                current_hash=current_hash,
            )

        return report

    # ── Neo4j audit ────────────────────────────────────────────────

    async def _log_neo4j_event(
        self, url: str, operation: str, status_code: int, result_count: int
    ) -> None:
        """Write a (:WebIntelligenceEvent) node for audit and budget tracking."""
        if self._neo4j is None:
            return
        try:
            now = datetime.now(UTC).isoformat()
            domain = _domain(url)
            await self._neo4j.execute_write(
                """
                MERGE (d:WebDomain {name: $domain})
                CREATE (e:WebIntelligenceEvent {
                    id: randomUUID(),
                    url: $url,
                    operation: $operation,
                    status_code: $status_code,
                    result_count: $result_count,
                    fetched_at: datetime($now),
                    domain: $domain
                })
                CREATE (d)-[:HAS_EVENT]->(e)
                """,
                {
                    "url": url[:500],
                    "operation": operation,
                    "status_code": status_code,
                    "result_count": result_count,
                    "now": now,
                    "domain": domain,
                },
            )
        except Exception as exc:
            logger.debug("web_client.neo4j_log_error", error=str(exc))


# ─── Intelligence feed monitors ───────────────────────────────────

# High-value URLs and their scraping metadata.
# "api" targets use official data APIs; "scrape" targets use HTML parsing.
INTELLIGENCE_FEEDS: list[dict[str, Any]] = [
    # DeFi ecosystem - official API
    {
        "id": "defillama_protocols",
        "url": "https://api.llama.fi/protocols",
        "method": "api",
        "category": "defi",
        "description": "DeFiLlama all protocols TVL",
        "schema": None,
    },
    {
        "id": "defillama_yields",
        "url": "https://yields.llama.fi/pools",
        "method": "api",
        "category": "defi",
        "description": "DeFiLlama yield pools",
        "schema": None,
    },
    # GitHub Trending
    {
        "id": "github_trending_python",
        "url": "https://github.com/trending/python?since=daily",
        "method": "scrape",
        "category": "tech_trends",
        "description": "GitHub trending Python repositories",
        "schema": None,
    },
    {
        "id": "github_trending_rust",
        "url": "https://github.com/trending/rust?since=daily",
        "method": "scrape",
        "category": "tech_trends",
        "description": "GitHub trending Rust repositories",
        "schema": None,
    },
    # Bounty platforms
    {
        "id": "algora_bounties",
        "url": "https://algora.io/bounties",
        "method": "scrape",
        "category": "bounty",
        "description": "Algora open bounties",
        "schema": None,
    },
    {
        "id": "gitcoin_grants",
        "url": "https://grants.gitcoin.co",
        "method": "scrape",
        "category": "bounty",
        "description": "Gitcoin active grants",
        "schema": None,
    },
    # HackerNews
    {
        "id": "hackernews_jobs",
        "url": "https://hacker-news.firebaseio.com/v0/jobstories.json",
        "method": "api",
        "category": "tech_trends",
        "description": "HackerNews job postings",
        "schema": None,
    },
    {
        "id": "hackernews_new",
        "url": "https://hacker-news.firebaseio.com/v0/newstories.json",
        "method": "api",
        "category": "tech_trends",
        "description": "HackerNews new stories",
        "schema": None,
    },
]
