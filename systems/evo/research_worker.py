"""
EcodiaOS — ArXiv Research Worker

Autonomous daily scanner that queries arXiv for papers relevant to
EcodiaOS's capabilities, scores them for relevance using the organism's
LLM pipeline, and returns structured technique dictionaries suitable for
the ArxivProposalTranslator.

Pipeline:
  1. Query arXiv API for recent CS/AI papers matching EOS-relevant categories
  2. Filter by keyword relevance (zero LLM cost)
  3. Score remaining papers via LLM for technique-level relevance
  4. Return list of raw technique dicts ready for translation

This worker is stateless — deduplication and governance gating happen
downstream in the ArxivProposalTranslator → Simula pipeline.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(module="evo.research_worker")

# arXiv API endpoint — returns Atom XML
_ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Categories relevant to EcodiaOS subsystems
_ARXIV_CATEGORIES = [
    "cs.AI",   # Artificial Intelligence
    "cs.SE",   # Software Engineering
    "cs.PL",   # Programming Languages
    "cs.LO",   # Logic in Computer Science
    "cs.CR",   # Cryptography and Security
    "cs.MA",   # Multi-Agent Systems
    "cs.LG",   # Machine Learning
]

# Keywords that signal relevance to EOS capabilities
_RELEVANCE_KEYWORDS: list[str] = [
    "e-graph", "equality saturation", "program synthesis",
    "neurosymbolic", "code generation", "self-improving",
    "multi-agent", "autonomous", "formal verification",
    "proof", "smt", "z3", "constraint solving",
    "code repair", "automated debugging", "root cause",
    "reinforcement learning from human feedback", "rlhf", "grpo",
    "retrieval augmented", "rag", "code search",
    "vulnerability", "fuzzing", "exploit detection",
    "self-evolving", "meta-learning", "architecture search",
    "orchestration", "pipeline", "workflow",
    "adversarial", "robustness", "coevolution",
    "ebpf", "xdp", "network filter",
    "anomaly detection", "drift", "monitoring",
    "knowledge graph", "reasoning", "planning",
]

# Minimum LLM relevance score to emit as a technique
_MIN_RELEVANCE_SCORE = 0.6

# Max papers to fetch per scan
_MAX_RESULTS = 30

# Max papers to send through LLM scoring (after keyword filter)
_MAX_LLM_SCORED = 10


class ArxivScientist:
    """
    Daily arXiv scanner that discovers techniques relevant to EcodiaOS
    and produces raw technique dicts for the ArxivProposalTranslator.

    Uses the organism's rate-limited LLM pipeline for relevance scoring,
    so it respects existing token budgets and provider fallback logic.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._log = logger
        self._http = httpx.AsyncClient(timeout=30.0)
        self._total_scans: int = 0
        self._total_papers_found: int = 0
        self._total_techniques_emitted: int = 0

    async def run_daily_scan(self) -> list[dict[str, Any]]:
        """
        Execute a full scan cycle:
          1. Fetch recent papers from arXiv
          2. Keyword-filter for relevance
          3. LLM-score the top candidates
          4. Return technique dicts for papers above threshold

        Returns an empty list if no relevant papers are found or if the
        arXiv API is unreachable.
        """
        self._total_scans += 1
        self._log.info("arxiv_scan_started", scan_number=self._total_scans)

        # 1. Fetch papers
        papers = await self._fetch_recent_papers()
        if not papers:
            self._log.info("arxiv_scan_no_papers")
            return []

        self._total_papers_found += len(papers)

        # 2. Keyword filter
        candidates = self._keyword_filter(papers)
        self._log.info(
            "arxiv_keyword_filtered",
            total=len(papers),
            candidates=len(candidates),
        )
        if not candidates:
            return []

        # 3. LLM relevance scoring (capped to control token spend)
        scored = await self._score_candidates(candidates[:_MAX_LLM_SCORED])

        # 4. Filter by threshold
        techniques: list[dict[str, Any]] = [
            t for t in scored if t.get("relevance_score", 0.0) >= _MIN_RELEVANCE_SCORE
        ]

        self._total_techniques_emitted += len(techniques)
        self._log.info(
            "arxiv_scan_completed",
            papers_fetched=len(papers),
            keyword_candidates=len(candidates),
            llm_scored=len(scored),
            techniques_emitted=len(techniques),
        )
        return techniques

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_scans": self._total_scans,
            "total_papers_found": self._total_papers_found,
            "total_techniques_emitted": self._total_techniques_emitted,
        }

    # ── Internal ──────────────────────────────────────────────────

    async def _fetch_recent_papers(self) -> list[dict[str, str]]:
        """Query arXiv API for recent papers in relevant categories."""
        category_query = " OR ".join(f"cat:{c}" for c in _ARXIV_CATEGORIES)
        params = {
            "search_query": category_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(_MAX_RESULTS),
        }

        try:
            resp = await self._http.get(_ARXIV_API_URL, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            self._log.warning("arxiv_api_error", error=str(exc))
            return []

        return self._parse_atom_feed(resp.text)

    @staticmethod
    def _parse_atom_feed(xml_text: str) -> list[dict[str, str]]:
        """Parse arXiv Atom XML into a list of paper dicts."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[dict[str, str]] = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        for entry in root.findall("atom:entry", ns):
            paper_id_el = entry.find("atom:id", ns)
            title_el = entry.find("atom:title", ns)
            abstract_el = entry.find("atom:summary", ns)

            if paper_id_el is None or title_el is None:
                continue

            # Extract paper ID from URL: http://arxiv.org/abs/2401.12345v1 → 2401.12345
            raw_id = (paper_id_el.text or "").strip()
            paper_id = raw_id.rsplit("/", 1)[-1].split("v")[0]

            authors = [
                (a.find("atom:name", ns).text or "").strip()
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]

            # Extract arxiv URL from alternate link
            arxiv_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("rel") == "alternate":
                    arxiv_url = link.get("href", "")
                    break

            papers.append({
                "paper_id": paper_id,
                "title": (title_el.text or "").strip().replace("\n", " "),
                "abstract": (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else "",
                "authors": ", ".join(authors),
                "arxiv_url": arxiv_url or f"https://arxiv.org/abs/{paper_id}",
            })

        return papers

    def _keyword_filter(self, papers: list[dict[str, str]]) -> list[dict[str, str]]:
        """Fast keyword-based pre-filter. Zero LLM cost."""
        candidates: list[dict[str, str]] = []
        for paper in papers:
            searchable = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            matches = [kw for kw in _RELEVANCE_KEYWORDS if kw in searchable]
            if matches:
                paper["matched_keywords"] = ", ".join(matches)
                candidates.append(paper)
        return candidates

    async def _score_candidates(
        self,
        candidates: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Use the LLM to score each candidate's relevance to EcodiaOS."""

        scored: list[dict[str, Any]] = []

        for paper in candidates:
            try:
                score, technique_name = await self._score_single_paper(paper)
                if score > 0.0:
                    scored.append({
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "abstract": paper.get("abstract", ""),
                        "technique_name": technique_name,
                        "technique_keywords": [
                            kw.strip()
                            for kw in paper.get("matched_keywords", "").split(",")
                            if kw.strip()
                        ],
                        "authors": [
                            a.strip()
                            for a in paper.get("authors", "").split(",")
                            if a.strip()
                        ],
                        "arxiv_url": paper.get("arxiv_url", ""),
                        "relevance_score": score,
                    })
            except Exception as exc:
                self._log.warning(
                    "arxiv_score_failed",
                    paper_id=paper.get("paper_id"),
                    error=str(exc),
                )

        return scored

    async def _score_single_paper(
        self,
        paper: dict[str, str],
    ) -> tuple[float, str]:
        """
        Ask the LLM to evaluate a single paper's relevance to EcodiaOS.

        Returns (relevance_score, technique_name).
        Returns (0.0, "") if the paper is irrelevant or parsing fails.
        """

        prompt = (
            "You are evaluating an arXiv paper for relevance to EcodiaOS, "
            "a self-evolving AI organism with subsystems for: code generation "
            "(Simula), learning/hypothesis (Evo), perception (Atune), "
            "goal planning (Nova), ethics (Equor), memory, and embodiment (Soma).\n\n"
            f"Paper: {paper['title']}\n"
            f"Abstract: {paper.get('abstract', '')[:600]}\n\n"
            "Respond with ONLY a JSON object (no markdown):\n"
            '{"relevance_score": 0.0-1.0, "technique_name": "short name of the core technique"}\n\n'
            "Score 0.8+ if directly applicable to a subsystem. "
            "Score 0.5-0.7 if tangentially useful. "
            "Score below 0.5 if not relevant."
        )

        resp = await self._llm.evaluate(prompt, max_tokens=150, temperature=0.1)
        return self._parse_score_response(resp.text)

    @staticmethod
    def _parse_score_response(text: str) -> tuple[float, str]:
        """Parse LLM JSON response into (score, technique_name)."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        try:
            data = json.loads(cleaned)
            score = float(data.get("relevance_score", 0.0))
            technique = str(data.get("technique_name", ""))
            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            return (score, technique)
        except (json.JSONDecodeError, TypeError, ValueError):
            return (0.0, "")

    async def close(self) -> None:
        """Shut down the HTTP client."""
        await self._http.aclose()
