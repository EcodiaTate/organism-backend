"""
EcodiaOS - Nova Opportunity Scanner

Proactive opportunity detection. Unlike the passive InputChannels (which poll external
APIs and hand raw data to Evo as PatternCandidates), the OpportunityScanner actively
RANKS every opportunity against the organism's four constitutional drives and decides:

  1. High confidence + high ROI  → inject directly as Nova goal (auto-goal)
  2. Medium confidence / ROI     → store in opportunity_backlog for deliberation context
  3. Type == learning            → emit LEARNING_OPPORTUNITY_DETECTED (Evo / Simula)
  4. All scanned                 → emit OPPORTUNITY_DETECTED per ranked item

Scanner inventory
─────────────────
  BountyOpportunityScanner    - wraps Oikos BountyHunter candidates
  YieldOpportunityScanner     - DeFiLlama pools not in current portfolio
  LearningOpportunityScanner  - ArXiv cs.AI + GitHub repos + HackerNews
  PartnershipOpportunityScanner - GitHub issues + X posts seeking AI collaborators
  MarketTimingScanner         - Snapshot.org governance votes + Base gas price

Drive scoring
─────────────
  Each opportunity is scored against Coherence / Care / Growth / Honesty.
  Composite = weighted average using the organism's current drive weights.
  Opportunities below MIN_COMPOSITE_SCORE are silently discarded before emission.

Background scheduling
─────────────────────
  The scanner runs as a supervised background task (30-minute cycle).
  It also fires on REVENUE_INJECTED (new capital → re-check yield thresholds)
  and DOMAIN_MASTERY_DETECTED (new domain unlocked → re-check bounty tiers).

Architecture notes
──────────────────
  - No direct cross-system imports; all wired via set_*() injection
  - HTTP via httpx.AsyncClient with per-scanner timeouts; never blocks the event loop
  - Dedup via an in-memory set + Redis SET (oikos:scanner:seen_ids); 7-day TTL
  - Drive alignment heuristics are intentionally conservative / cheap - no LLM calls
    in the scanner hot-path.  LLM deliberation happens *after* injection into Nova.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from primitives.common import DriveAlignmentVector, EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("nova.opportunity_scanner")

# ─── Constants ────────────────────────────────────────────────────────────────

# Auto-goal promotion: opportunity is injected directly as Nova goal when BOTH thresholds pass.
# 0.0 = no gate (all opportunities pass; Nova deliberation decides).
# Set NOVA_AUTO_GOAL_MIN_CONFIDENCE / NOVA_AUTO_GOAL_MIN_ROI in env to add a floor.
AUTO_GOAL_MIN_CONFIDENCE: float = float(os.getenv("NOVA_AUTO_GOAL_MIN_CONFIDENCE", "0.0"))
AUTO_GOAL_MIN_ROI: float = float(os.getenv("NOVA_AUTO_GOAL_MIN_ROI", "0.0"))

# Composite drive score floor before emission. 0.0 = emit everything.
# Set NOVA_MIN_COMPOSITE_SCORE in env to filter low-alignment noise.
MIN_COMPOSITE_SCORE: float = float(os.getenv("NOVA_MIN_COMPOSITE_SCORE", "0.0"))

# Assumed hourly cost for effort estimation (blended RE + Claude API cost)
HOURLY_COST_USD: float = float(os.getenv("NOVA_HOURLY_COST_USD", "0.50"))

# Yield improvement gate: 0.0 = surface all yield opportunities regardless of improvement delta.
# Set NOVA_YIELD_IMPROVEMENT_THRESHOLD_PCT in env to filter marginal improvements.
YIELD_IMPROVEMENT_THRESHOLD_PCT: float = float(os.getenv("NOVA_YIELD_IMPROVEMENT_THRESHOLD_PCT", "0.0"))

# Maximum items in the in-memory opportunity backlog (oldest evicted first)
MAX_BACKLOG_SIZE: int = 50

# Redis dedup key prefix + TTL
_REDIS_SEEN_KEY = "nova:scanner:seen_ids"
_REDIS_SEEN_TTL_S = 7 * 24 * 3600  # 7 days

# Per-scanner HTTP fetch timeout (seconds)
_FETCH_TIMEOUT_S = 20.0

# Learning opportunity: minimum relevance score to emit
MIN_LEARNING_RELEVANCE: float = 0.30


# ─── Data Models ──────────────────────────────────────────────────────────────


class ScannedOpportunity(EOSBaseModel):
    """
    A fully-scored opportunity produced by the scanner pipeline.

    This is the canonical type emitted on OPPORTUNITY_DETECTED and stored
    in Nova's backlog.  It is richer than the InputChannel Opportunity
    (which is a raw data container) because it carries drive-alignment
    scores and economic projections.
    """

    opportunity_id: str
    type: str  # "bounty" | "yield" | "content" | "partnership" | "learning"
    description: str
    estimated_value_usdc: Decimal = Decimal("0")
    estimated_effort_hours: float = 1.0
    roi: float = 0.0                          # estimated_value / (effort × hourly_cost)
    drive_alignment: DriveAlignmentVector = DriveAlignmentVector()
    composite_score: float = 0.0              # weighted sum of drive alignments
    source: str = ""                          # originating scanner name
    url: str | None = None
    deadline: datetime | None = None
    confidence: float = 0.5
    discovered_at: datetime = None            # set to utc_now() in __init__
    auto_goal: bool = False                   # True when Nova converts to goal

    def model_post_init(self, __context: Any) -> None:
        if self.discovered_at is None:
            object.__setattr__(self, "discovered_at", utc_now())


class LearningResource(EOSBaseModel):
    """A technical resource suitable for feeding directly into Evo / Simula."""

    resource_id: str
    resource_type: str   # "paper" | "repo" | "discussion"
    title: str
    summary: str
    url: str
    domain: str
    relevance_score: float = 0.5
    capability_gaps_addressed: list[str] = []
    source: str = ""     # "arxiv" | "github" | "hackernews" | "reddit"
    discovered_at: datetime = None

    def model_post_init(self, __context: Any) -> None:
        if self.discovered_at is None:
            object.__setattr__(self, "discovered_at", utc_now())


# ─── Drive Alignment Heuristics ───────────────────────────────────────────────


def _score_bounty(candidate: dict[str, Any]) -> DriveAlignmentVector:
    """Conservative drive alignment for a bounty opportunity."""
    return DriveAlignmentVector(
        coherence=0.5,   # contributing to a codebase is coherent action
        care=0.4,        # open-source contributions benefit the commons
        growth=0.7,      # solving novel problems grows capabilities
        honesty=0.8,     # public code work is inherently transparent
    )


def _score_yield(apy: float, is_new_protocol: bool) -> DriveAlignmentVector:
    """Drive alignment for a yield farming opportunity."""
    # High APY = more risk = slightly lower Care (potential for harm/loss)
    care = max(0.2, 0.7 - (apy / 100.0) * 0.3)
    return DriveAlignmentVector(
        coherence=0.5,
        care=float(f"{care:.2f}"),
        growth=0.6 if is_new_protocol else 0.4,  # new protocols = more to learn
        honesty=0.7,
    )


def _score_content(domain: str) -> DriveAlignmentVector:
    """Drive alignment for a content monetization opportunity."""
    return DriveAlignmentVector(
        coherence=0.6,
        care=0.7,   # sharing knowledge benefits others
        growth=0.5,
        honesty=0.9,  # expressing the organism's genuine perspective
    )


def _score_partnership(description: str) -> DriveAlignmentVector:
    """Drive alignment for a partnership / collaboration opportunity."""
    desc_lower = description.lower()
    # Flag anything that mentions secrecy, competitive advantage, exclusivity
    honesty_penalty = 0.3 if any(
        kw in desc_lower for kw in ("exclusive", "nda", "proprietary", "closed")
    ) else 0.0
    return DriveAlignmentVector(
        coherence=0.5,
        care=0.6,
        growth=0.7,   # collaboration expands networks and skills
        honesty=max(0.3, 0.8 - honesty_penalty),
    )


def _score_learning(domain: str, relevance: float) -> DriveAlignmentVector:
    """Drive alignment for a learning resource opportunity."""
    return DriveAlignmentVector(
        coherence=0.6 + relevance * 0.2,   # relevant learning tightens world model
        care=0.5,
        growth=0.6 + relevance * 0.3,      # direct growth signal
        honesty=0.7,
    )


def _compute_composite(
    alignment: DriveAlignmentVector,
    weights: dict[str, float],
) -> float:
    """Weighted composite from drive alignment + organism drive weights."""
    total_weight = sum(weights.values()) or 1.0
    return (
        alignment.coherence * weights.get("coherence", 1.0)
        + alignment.care * weights.get("care", 1.0)
        + alignment.growth * weights.get("growth", 1.0)
        + alignment.honesty * weights.get("honesty", 1.0)
    ) / total_weight


# ─── Sub-Scanners ─────────────────────────────────────────────────────────────


class BountyOpportunityScanner:
    """
    Wraps BountyHunter candidates already discovered by Oikos and converts
    the approved ones into ScannedOpportunity objects.

    The BountyHunter runs inside the Oikos economic consolidation cycle and
    stores candidates in Redis.  Rather than re-fetching external platforms
    (which would double the network load), this scanner reads the cached
    snapshot from Redis and re-scores them for drive alignment.

    If no Redis key exists it falls back to a lightweight direct scan of
    the GitHub public search endpoint (no auth, generous rate-limit).
    """

    REDIS_CANDIDATES_KEY = "oikos:bounty:candidates"

    async def scan(
        self,
        redis: "RedisClient | None",
        drive_weights: dict[str, float],
    ) -> list[ScannedOpportunity]:
        candidates_raw: list[dict[str, Any]] = []

        # Try to read cached candidates from Redis first
        if redis is not None:
            try:
                raw = await redis.get(self.REDIS_CANDIDATES_KEY)
                if raw:
                    import json
                    candidates_raw = json.loads(raw)
            except Exception as exc:
                logger.warning("bounty_scanner_redis_error", error=str(exc))

        # Fallback: lightweight GitHub search
        if not candidates_raw:
            candidates_raw = await self._fallback_github_search()

        results: list[ScannedOpportunity] = []
        for c in candidates_raw:
            reward_raw = c.get("reward_usd", "0")
            try:
                reward = Decimal(str(reward_raw))
            except Exception:
                continue
            if reward <= 0:
                continue

            cost_raw = c.get("estimated_cost_usd", "0")
            try:
                cost = Decimal(str(cost_raw))
            except Exception:
                cost = Decimal("0")

            effort_hours = float(cost) / HOURLY_COST_USD if HOURLY_COST_USD > 0 else 1.0
            roi = float(reward) / (effort_hours * HOURLY_COST_USD) if effort_hours > 0 else 0.0

            alignment = _score_bounty(c)
            composite = _compute_composite(alignment, drive_weights)
            if composite < MIN_COMPOSITE_SCORE:
                continue

            opp_id = _stable_id("bounty", c.get("issue_url", c.get("candidate_id", new_id())))
            results.append(ScannedOpportunity(
                opportunity_id=opp_id,
                type="bounty",
                description=(
                    f"{c.get('title', 'Bounty')} - ${float(reward):.2f} "
                    f"on {c.get('platform', 'unknown')}"
                ),
                estimated_value_usdc=reward,
                estimated_effort_hours=effort_hours,
                roi=roi,
                drive_alignment=alignment,
                composite_score=composite,
                source="bounty_hunter_cache",
                url=c.get("issue_url"),
                deadline=_parse_deadline(c.get("deadline")),
                confidence=float(c.get("capability_match", Decimal("0.5"))),
            ))

        return results

    async def _fallback_github_search(self) -> list[dict[str, Any]]:
        """Fetch top bounty-labelled GitHub issues as a lightweight fallback."""
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
                resp = await client.get(
                    "https://api.github.com/search/issues",
                    params={
                        "q": "label:bounty+state:open",
                        "sort": "created",
                        "per_page": "10",
                    },
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
                if resp.status_code != 200:
                    return []
                items = resp.json().get("items", [])
        except Exception:
            return []

        result = []
        for item in items:
            body = item.get("body", "") or ""
            # Simple heuristic: look for "$NNN" in title or body
            import re
            reward_match = re.search(r"\$(\d+(?:\.\d{2})?)", item.get("title", "") + body)
            reward = Decimal(reward_match.group(1)) if reward_match else Decimal("25")
            result.append({
                "candidate_id": str(item.get("id", "")),
                "platform": "github_bounties",
                "issue_url": item.get("html_url", ""),
                "title": item.get("title", ""),
                "description": body[:200],
                "reward_usd": str(reward),
                "estimated_cost_usd": str(reward * Decimal("0.3")),
                "capability_match": "0.5",
            })
        return result


class YieldOpportunityScanner:
    """
    Queries DeFiLlama for pools not currently in the organism's portfolio.
    Only surfaces opportunities where the new pool APY exceeds the organism's
    current best deployed position by at least YIELD_IMPROVEMENT_THRESHOLD_PCT.
    """

    _POOLS_URL = "https://yields.llama.fi/pools"
    _PROTOCOLS = {"aave-v3", "morpho", "compound-v3", "spark", "aerodrome", "uniswap-v3"}
    _CHAIN_FILTER = {"Base", "Ethereum"}

    async def scan(
        self,
        current_portfolio_apys: list[float],
        drive_weights: dict[str, float],
    ) -> list[ScannedOpportunity]:
        current_best = max(current_portfolio_apys) if current_portfolio_apys else 0.0

        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
                resp = await client.get(self._POOLS_URL)
                resp.raise_for_status()
                pools: list[dict[str, Any]] = resp.json().get("data", [])
        except Exception as exc:
            logger.warning("yield_scanner_fetch_error", error=str(exc))
            return []

        results: list[ScannedOpportunity] = []
        for pool in pools:
            protocol = (pool.get("project") or "").lower()
            if protocol not in self._PROTOCOLS:
                continue
            chain = pool.get("chain") or ""
            if chain not in self._CHAIN_FILTER:
                continue
            apy: float = pool.get("apy") or 0.0
            if apy <= 0:
                continue

            # Only surface if meaningfully better than current best
            improvement = apy - current_best
            if improvement < YIELD_IMPROVEMENT_THRESHOLD_PCT:
                continue

            tvl: float = pool.get("tvlUsd") or 0.0
            symbol: str = pool.get("symbol") or "?"
            monthly_usd = Decimal(str(round(10_000 * apy / 100 / 12, 2)))

            alignment = _score_yield(apy, is_new_protocol=protocol not in {"aave-v3", "morpho"})
            composite = _compute_composite(alignment, drive_weights)
            if composite < MIN_COMPOSITE_SCORE:
                continue

            opp_id = _stable_id("yield", f"{protocol}-{symbol}-{chain}")
            results.append(ScannedOpportunity(
                opportunity_id=opp_id,
                type="yield",
                description=(
                    f"{protocol.title()} {symbol} on {chain}: {apy:.1f}% APY "
                    f"(+{improvement:.1f}pp above current best). TVL ${tvl:,.0f}."
                ),
                estimated_value_usdc=monthly_usd,
                estimated_effort_hours=0.5,  # DeFi deployment is low-effort
                roi=float(monthly_usd) / (0.5 * HOURLY_COST_USD),
                drive_alignment=alignment,
                composite_score=composite,
                source="defi_llama_yield_scanner",
                url=f"https://defillama.com/protocol/{protocol}",
                confidence=0.65,
            ))

        # Sort by APY improvement descending; return top 5
        results.sort(key=lambda o: o.roi, reverse=True)
        return results[:5]


class LearningOpportunityScanner:
    """
    Scans external knowledge sources for technical resources relevant to the
    organism's current capability gaps.

    Sources:
      - ArXiv cs.AI / cs.LG - recent papers
      - GitHub trending - repos in the organism's domains
      - Hacker News - top AI/ML discussions

    Produces LearningResource objects (not ScannedOpportunity) which are
    emitted as LEARNING_OPPORTUNITY_DETECTED events directly to Evo/Simula.
    """

    _ARXIV_URL = "https://export.arxiv.org/api/query"
    _ARXIV_CATEGORIES = "cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CR+OR+cat:cs.NE"
    _HN_URL = "https://hn.algolia.com/api/v1/search"
    _GITHUB_TRENDING_URL = "https://api.github.com/search/repositories"

    # Domains that directly address organism capability gaps
    _CAPABILITY_GAP_KEYWORDS: dict[str, list[str]] = {
        "causal_reasoning": ["causal inference", "do-calculus", "counterfactual", "causal discovery"],
        "formal_verification": ["z3", "lean", "dafny", "theorem proving", "smt solver"],
        "continual_learning": ["continual learning", "catastrophic forgetting", "lifelong learning", "lora"],
        "active_inference": ["active inference", "free energy principle", "variational bayes", "efe"],
        "multi_agent": ["multi-agent", "agent coordination", "federated learning", "autonomous agents"],
        "code_generation": ["code generation", "program synthesis", "neural code", "codegen"],
        "defi_reasoning": ["defi", "liquidity", "yield farming", "uniswap", "aave"],
        "safety_alignment": ["constitutional ai", "rlhf", "reward modeling", "alignment", "safety"],
    }

    async def scan(
        self,
        known_domains: list[str],
    ) -> list[LearningResource]:
        results: list[LearningResource] = []

        # Run all three sources in parallel
        arxiv_task = asyncio.create_task(self._scan_arxiv())
        github_task = asyncio.create_task(self._scan_github_trending(known_domains))
        hn_task = asyncio.create_task(self._scan_hackernews())

        for coro in asyncio.as_completed([arxiv_task, github_task, hn_task]):
            try:
                batch = await coro
                results.extend(batch)
            except Exception as exc:
                logger.warning("learning_scanner_batch_error", error=str(exc))

        # Score relevance and filter
        scored = [r for r in results if r.relevance_score >= MIN_LEARNING_RELEVANCE]
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        return scored[:20]  # cap at 20 per scan cycle

    async def _scan_arxiv(self) -> list[LearningResource]:
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
                resp = await client.get(
                    self._ARXIV_URL,
                    params={
                        "search_query": self._ARXIV_CATEGORIES,
                        "start": "0",
                        "max_results": "20",
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                    },
                )
                resp.raise_for_status()
                content = resp.text
        except Exception as exc:
            logger.warning("arxiv_scan_error", error=str(exc))
            return []

        return self._parse_arxiv_atom(content)

    def _parse_arxiv_atom(self, xml: str) -> list[LearningResource]:
        """Parse ArXiv Atom feed - minimal XML parsing without lxml dependency."""
        results: list[LearningResource] = []
        import re

        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        for entry in entries[:20]:
            title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            id_m = re.search(r"<id>(.*?)</id>", entry)

            title = _strip_xml(title_m.group(1)) if title_m else ""
            summary = _strip_xml(summary_m.group(1))[:400] if summary_m else ""
            arxiv_url = id_m.group(1).strip() if id_m else ""

            if not title or not arxiv_url:
                continue

            domain, gaps, relevance = self._classify_resource(title + " " + summary)
            if relevance < MIN_LEARNING_RELEVANCE:
                continue

            resource_id = _stable_id("arxiv", arxiv_url)
            results.append(LearningResource(
                resource_id=resource_id,
                resource_type="paper",
                title=title,
                summary=summary,
                url=arxiv_url,
                domain=domain,
                relevance_score=relevance,
                capability_gaps_addressed=gaps,
                source="arxiv",
            ))
        return results

    async def _scan_github_trending(self, known_domains: list[str]) -> list[LearningResource]:
        """Search GitHub for recently active repos in the organism's domains."""
        # Compose a search query from known domains
        domain_query = " OR ".join(
            f'"{d}"' for d in (known_domains or ["active inference", "causal reasoning", "defi"])[:5]
        )
        cutoff = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
        query = f"({domain_query}) pushed:>{cutoff} stars:>10"

        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
                resp = await client.get(
                    self._GITHUB_TRENDING_URL,
                    params={"q": query, "sort": "stars", "per_page": "10"},
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
                if resp.status_code != 200:
                    return []
                items = resp.json().get("items", [])
        except Exception as exc:
            logger.warning("github_trending_scan_error", error=str(exc))
            return []

        results = []
        for repo in items:
            name: str = repo.get("full_name", "")
            description: str = repo.get("description") or ""
            url: str = repo.get("html_url", "")
            text = name + " " + description
            domain, gaps, relevance = self._classify_resource(text)
            if relevance < MIN_LEARNING_RELEVANCE:
                continue
            resource_id = _stable_id("github", url)
            results.append(LearningResource(
                resource_id=resource_id,
                resource_type="repo",
                title=name,
                summary=description[:300],
                url=url,
                domain=domain,
                relevance_score=relevance,
                capability_gaps_addressed=gaps,
                source="github",
            ))
        return results

    async def _scan_hackernews(self) -> list[LearningResource]:
        """Scan HackerNews for AI/ML discussions with high engagement."""
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
                resp = await client.get(
                    self._HN_URL,
                    params={
                        "query": "AI agent autonomous reasoning",
                        "tags": "story",
                        "numericFilters": "points>30",
                        "hitsPerPage": "10",
                    },
                )
                resp.raise_for_status()
                hits = resp.json().get("hits", [])
        except Exception as exc:
            logger.warning("hn_scan_error", error=str(exc))
            return []

        results = []
        for hit in hits:
            title: str = hit.get("title") or ""
            url: str = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            text = title + " " + (hit.get("story_text") or "")
            domain, gaps, relevance = self._classify_resource(text)
            if relevance < MIN_LEARNING_RELEVANCE:
                continue
            resource_id = _stable_id("hn", url)
            results.append(LearningResource(
                resource_id=resource_id,
                resource_type="discussion",
                title=title,
                summary=text[:300],
                url=url,
                domain=domain,
                relevance_score=relevance,
                capability_gaps_addressed=gaps,
                source="hackernews",
            ))
        return results

    def _classify_resource(
        self, text: str
    ) -> tuple[str, list[str], float]:
        """
        Classify text by capability gap relevance.
        Returns (primary_domain, gaps_addressed, relevance_score).
        """
        text_lower = text.lower()
        gap_hits: dict[str, int] = {}
        for gap, keywords in self._CAPABILITY_GAP_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                gap_hits[gap] = count

        if not gap_hits:
            return "general_ai", [], 0.0

        # Primary domain = gap with most keyword matches
        primary_gap = max(gap_hits, key=lambda g: gap_hits[g])
        total_hits = sum(gap_hits.values())
        relevance = min(1.0, total_hits * 0.15)

        return primary_gap, list(gap_hits.keys()), relevance


class PartnershipOpportunityScanner:
    """
    Scans for explicit collaboration requests on GitHub and X (Twitter).

    GitHub: issues tagged 'collaboration', 'ai-agent', 'automation', 'bounty'
    X:      not implemented (requires paid API) - placeholder for future expansion

    Only surfaces partnerships where the organism's capabilities are a clear fit.
    """

    async def scan(self, drive_weights: dict[str, float]) -> list[ScannedOpportunity]:
        results: list[ScannedOpportunity] = []

        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
                resp = await client.get(
                    "https://api.github.com/search/issues",
                    params={
                        "q": (
                            "label:collaboration+OR+label:ai-agent+OR+label:automation "
                            "state:open type:issue"
                        ),
                        "sort": "created",
                        "per_page": "15",
                    },
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
                if resp.status_code != 200:
                    return []
                items = resp.json().get("items", [])
        except Exception as exc:
            logger.warning("partnership_scanner_error", error=str(exc))
            return []

        for item in items:
            title: str = item.get("title", "")
            body: str = (item.get("body") or "")[:400]
            url: str = item.get("html_url", "")
            text = title + " " + body

            # Only keep explicit collaboration requests
            collab_signals = sum(1 for kw in (
                "looking for", "seeking", "collaborate", "partner", "work together",
                "ai agent", "automation", "help with", "contributor wanted"
            ) if kw in text.lower())
            if collab_signals < 2:
                continue

            alignment = _score_partnership(text)
            composite = _compute_composite(alignment, drive_weights)
            if composite < MIN_COMPOSITE_SCORE:
                continue

            opp_id = _stable_id("partnership", url)
            results.append(ScannedOpportunity(
                opportunity_id=opp_id,
                type="partnership",
                description=f"Collaboration request: {title[:120]}",
                estimated_value_usdc=Decimal("50"),   # placeholder; no stated reward
                estimated_effort_hours=5.0,
                roi=float(Decimal("50")) / (5.0 * HOURLY_COST_USD),
                drive_alignment=alignment,
                composite_score=composite,
                source="github_collaboration_scanner",
                url=url,
                confidence=0.4,   # low confidence - value is hard to estimate
            ))

        return results[:5]


class MarketTimingScanner:
    """
    Monitors DeFi governance and Base network conditions for time-sensitive
    action windows.

    Snapshot.org: upcoming governance votes that could change yields in
    the organism's positions. If a vote would significantly affect returns,
    flag for action.

    Base gas: low gas on Base L2 = cost-effective time to deploy capital.
    Flag when gas is below the historical median.
    """

    _SNAPSHOT_URL = "https://hub.snapshot.org/graphql"
    _ETHERSCAN_GAS_URL = "https://api.basescan.org/api"

    # Protocols where governance votes directly affect organism yield positions
    _MONITORED_SPACES = ["aave.eth", "morpho.eth", "compound-finance.eth"]

    # Base gas threshold (gwei) for "low gas" signal
    _LOW_GAS_THRESHOLD_GWEI: float = 0.05

    async def scan(
        self,
        drive_weights: dict[str, float],
        basescan_api_key: str = "",
    ) -> list[ScannedOpportunity]:
        results: list[ScannedOpportunity] = []

        gov_task = asyncio.create_task(self._scan_governance(drive_weights))
        gas_task = asyncio.create_task(self._scan_gas(drive_weights, basescan_api_key))

        for coro in asyncio.as_completed([gov_task, gas_task]):
            try:
                batch = await coro
                results.extend(batch)
            except Exception as exc:
                logger.warning("market_timing_scan_error", error=str(exc))

        return results

    async def _scan_governance(
        self, drive_weights: dict[str, float]
    ) -> list[ScannedOpportunity]:
        query = """
        {
          proposals(
            first: 10,
            where: {
              space_in: ["aave.eth", "morpho.eth", "compound-finance.eth"],
              state: "active"
            },
            orderBy: "end",
            orderDirection: asc
          ) {
            id title body start end space { id } scores_total
          }
        }
        """
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
                resp = await client.post(
                    self._SNAPSHOT_URL,
                    json={"query": query},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                proposals = resp.json().get("data", {}).get("proposals", [])
        except Exception as exc:
            logger.warning("snapshot_scan_error", error=str(exc))
            return []

        results = []
        for prop in proposals:
            title: str = prop.get("title", "")
            body: str = (prop.get("body") or "")[:300]
            space_id: str = prop.get("space", {}).get("id", "")
            end_ts: int = prop.get("end", 0)
            deadline = datetime.fromtimestamp(end_ts, tz=UTC) if end_ts else None

            # Only surface proposals that might affect yield
            if not any(kw in (title + body).lower() for kw in (
                "rate", "apy", "interest", "supply", "borrow", "liquidity", "pool", "param"
            )):
                continue

            alignment = DriveAlignmentVector(
                coherence=0.6,  # monitoring governance is coherent
                care=0.5,
                growth=0.4,
                honesty=0.8,
            )
            composite = _compute_composite(alignment, drive_weights)
            hours_until_end = (
                (deadline - datetime.now(UTC)).total_seconds() / 3600
                if deadline else 72.0
            )
            # Only surface if vote ends within 72 hours
            if hours_until_end > 72:
                continue

            opp_id = _stable_id("governance", prop.get("id", title))
            results.append(ScannedOpportunity(
                opportunity_id=opp_id,
                type="yield",  # affects yield positions
                description=(
                    f"Governance vote on {space_id}: {title[:100]}. "
                    f"Ends in {hours_until_end:.0f}h."
                ),
                estimated_value_usdc=Decimal("0"),  # defensive; outcome unclear
                estimated_effort_hours=0.25,
                roi=0.5,   # low ROI estimate but time-sensitive
                drive_alignment=alignment,
                composite_score=composite,
                source="snapshot_governance_scanner",
                url=f"https://snapshot.org/#/{space_id}/proposal/{prop.get('id', '')}",
                deadline=deadline,
                confidence=0.55,
            ))
        return results

    async def _scan_gas(
        self, drive_weights: dict[str, float], api_key: str
    ) -> list[ScannedOpportunity]:
        """Surface a 'deploy capital now' opportunity when Base gas is low."""
        try:
            params: dict[str, str] = {
                "module": "gastracker",
                "action": "gasoracle",
            }
            if api_key:
                params["apikey"] = api_key

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self._ETHERSCAN_GAS_URL, params=params)
                resp.raise_for_status()
                result = resp.json().get("result", {})
                suggest_gwei = float(result.get("SafeGasPrice", 999))
        except Exception:
            return []

        if suggest_gwei >= self._LOW_GAS_THRESHOLD_GWEI:
            return []

        alignment = DriveAlignmentVector(coherence=0.6, care=0.4, growth=0.5, honesty=0.7)
        composite = _compute_composite(alignment, drive_weights)
        opp_id = _stable_id("gas_window", datetime.now(UTC).strftime("%Y-%m-%dT%H"))
        return [ScannedOpportunity(
            opportunity_id=opp_id,
            type="yield",
            description=(
                f"Base gas is low ({suggest_gwei:.4f} gwei) - cost-effective window "
                "for DeFi capital deployment."
            ),
            estimated_value_usdc=Decimal("5"),
            estimated_effort_hours=0.1,
            roi=float(Decimal("5")) / (0.1 * HOURLY_COST_USD),
            drive_alignment=alignment,
            composite_score=composite,
            source="base_gas_scanner",
            confidence=0.85,
        )]


# ─── Dedup Helpers ────────────────────────────────────────────────────────────


def _stable_id(namespace: str, key: str) -> str:
    """Deterministic stable ID: sha256 of (namespace + key), first 16 hex chars."""
    return hashlib.sha256(f"{namespace}:{key}".encode()).hexdigest()[:16]


def _parse_deadline(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _strip_xml(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text).strip()


# ─── Main OpportunityScanner ──────────────────────────────────────────────────


class OpportunityScanner:
    """
    Proactively discovers opportunities aligned with the organism's drives.

    Orchestrates five sub-scanners, deduplicates results, scores against
    constitutional drives, and returns a ranked list of ScannedOpportunity
    objects ready for Nova to act on.

    Usage (inside NovaService)::

        scanner = OpportunityScanner()
        scanner.set_redis(redis)
        scanner.set_drive_weights({"coherence": 1.0, "care": 1.0, ...})
        results = await scanner.scan_cycle()
        learning = scanner.last_learning_resources  # feed to Evo / Simula

    The scanner is stateless except for:
      - _seen_ids: in-memory dedup set (cleared on restart; Redis is authoritative)
      - _current_portfolio_apys: updated externally by NovaService when Oikos reports
      - _drive_weights: updated when Nova receives SOMATIC_DRIVE_VECTOR or similar
    """

    def __init__(self) -> None:
        self._redis: "RedisClient | None" = None
        self._drive_weights: dict[str, float] = {
            "coherence": 1.0,
            "care": 1.0,
            "growth": 1.0,
            "honesty": 1.0,
        }
        self._known_domains: list[str] = ["active inference", "causal reasoning", "defi", "python"]
        self._current_portfolio_apys: list[float] = []
        self._basescan_api_key: str = ""

        # Sub-scanners
        self._bounty_scanner = BountyOpportunityScanner()
        self._yield_scanner = YieldOpportunityScanner()
        self._learning_scanner = LearningOpportunityScanner()
        self._partnership_scanner = PartnershipOpportunityScanner()
        self._market_scanner = MarketTimingScanner()

        # In-memory dedup
        self._seen_ids: set[str] = set()

        # Last scan results (available for NovaService to inspect)
        self.last_opportunities: list[ScannedOpportunity] = []
        self.last_learning_resources: list[LearningResource] = []
        self._last_scan_at: float = 0.0

        self._log = logger

    # ─── Dependency Injection ──────────────────────────────────────────────

    def set_redis(self, redis: "RedisClient") -> None:
        self._redis = redis

    def set_drive_weights(self, weights: dict[str, float]) -> None:
        self._drive_weights = dict(weights)

    def set_portfolio_apys(self, apys: list[float]) -> None:
        """Called by NovaService when Oikos reports current yield positions."""
        self._current_portfolio_apys = list(apys)

    def set_known_domains(self, domains: list[str]) -> None:
        self._known_domains = list(domains)

    def set_basescan_api_key(self, key: str) -> None:
        self._basescan_api_key = key

    # ─── Core Scan Cycle ──────────────────────────────────────────────────

    async def scan_cycle(self) -> list[ScannedOpportunity]:
        """
        Run all scanners concurrently, deduplicate, score, and return ranked
        opportunities.  Learning resources are stored in last_learning_resources
        and emitted separately by the caller.

        Returns opportunities sorted by (composite_score × confidence) descending.
        """
        self._log.info("opportunity_scan_cycle_start")
        start_ms = time.monotonic() * 1000

        # Run sub-scanners concurrently
        results = await asyncio.gather(
            self._bounty_scanner.scan(self._redis, self._drive_weights),
            self._yield_scanner.scan(self._current_portfolio_apys, self._drive_weights),
            self._partnership_scanner.scan(self._drive_weights),
            self._market_scanner.scan(self._drive_weights, self._basescan_api_key),
            return_exceptions=True,
        )

        # Run learning scanner separately (produces LearningResource, not ScannedOpportunity)
        try:
            learning = await self._learning_scanner.scan(self._known_domains)
        except Exception as exc:
            self._log.warning("learning_scan_error", error=str(exc))
            learning = []

        # Flatten and deduplicate economic opportunities
        all_opps: list[ScannedOpportunity] = []
        for batch in results:
            if isinstance(batch, Exception):
                self._log.warning("scanner_batch_error", error=str(batch))
                continue
            for opp in batch:
                if await self._is_new(opp.opportunity_id):
                    all_opps.append(opp)

        # Deduplicate learning resources
        new_learning: list[LearningResource] = []
        for lr in learning:
            if await self._is_new(lr.resource_id):
                new_learning.append(lr)

        # Rank by composite_score × confidence (higher = more urgent to act on)
        all_opps.sort(
            key=lambda o: o.composite_score * o.confidence,
            reverse=True,
        )

        self.last_opportunities = all_opps
        self.last_learning_resources = new_learning
        self._last_scan_at = time.monotonic()

        elapsed_ms = time.monotonic() * 1000 - start_ms
        self._log.info(
            "opportunity_scan_cycle_complete",
            opportunities=len(all_opps),
            learning_resources=len(new_learning),
            elapsed_ms=round(elapsed_ms),
        )
        return all_opps

    async def _is_new(self, item_id: str) -> bool:
        """True if the opportunity/resource has not been seen before."""
        if item_id in self._seen_ids:
            return False

        if self._redis is not None:
            try:
                is_member = await self._redis.sismember(_REDIS_SEEN_KEY, item_id)
                if is_member:
                    self._seen_ids.add(item_id)
                    return False
            except Exception:
                pass  # fall through to in-memory only

        # New - mark as seen
        self._seen_ids.add(item_id)
        if self._redis is not None:
            try:
                await self._redis.sadd(_REDIS_SEEN_KEY, item_id)
                await self._redis.expire(_REDIS_SEEN_KEY, _REDIS_SEEN_TTL_S)
            except Exception:
                pass

        return True

    def should_auto_goal(self, opp: ScannedOpportunity) -> bool:
        """True if this opportunity should be immediately converted to a Nova goal."""
        return (
            opp.confidence >= AUTO_GOAL_MIN_CONFIDENCE
            and opp.roi >= AUTO_GOAL_MIN_ROI
            and opp.composite_score >= 0.5
        )
