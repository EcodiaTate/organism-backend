"""
EcodiaOS - Belief Consolidation Scanner

Identifies high-confidence, low-volatility beliefs and hardens them into
read-only :ConsolidatedBelief reference nodes during Evo consolidation
Phase 2.75. Consolidated beliefs are faster to query and protected from
accidental mutation, while tentative beliefs remain in active, mutable form.

Integration:
    - Evo consolidation Phase 2.75 (after belief aging, before schema induction)
    - Foundation conflict detection in Phase 2 (hypothesis review)
    - Nova/Atune retrieval: prefer consolidated beliefs where available
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from primitives.common import new_id, utc_now
from systems.evo.types import (
    BeliefConsolidationResult,
    ConsolidatedBelief,
    FoundationConflict,
    Hypothesis,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# ─── Consolidation Thresholds ────────────────────────────────────────────────

# Minimum precision (confidence) for a belief to be eligible for consolidation
_MIN_PRECISION: float = 0.85

# Maximum volatility percentile - beliefs that change frequently stay mutable
_MAX_VOLATILITY: float = 0.2

# Minimum age in days before a belief can be consolidated
_MIN_AGE_DAYS: int = 30

# Negation patterns used in lightweight contradiction detection
_NEGATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bno longer\b", re.IGNORECASE),
    re.compile(r"\bincorrect\b", re.IGNORECASE),
    re.compile(r"\bfalse\b", re.IGNORECASE),
    re.compile(r"\bwrong\b", re.IGNORECASE),
    re.compile(r"\bdoes not\b", re.IGNORECASE),
    re.compile(r"\bdoesn't\b", re.IGNORECASE),
    re.compile(r"\bisn't\b", re.IGNORECASE),
    re.compile(r"\baren't\b", re.IGNORECASE),
    re.compile(r"\bwasn't\b", re.IGNORECASE),
    re.compile(r"\bwon't\b", re.IGNORECASE),
    re.compile(r"\bcan't\b", re.IGNORECASE),
    re.compile(r"\bcannot\b", re.IGNORECASE),
    re.compile(r"\bunlike\b", re.IGNORECASE),
    re.compile(r"\bcontrary\b", re.IGNORECASE),
]


class BeliefConsolidationScanner:
    """
    Scans the Neo4j belief graph for high-confidence, low-volatility beliefs
    and consolidates them into read-only :ConsolidatedBelief reference nodes.

    Run during Evo consolidation Phase 2.75 (after belief aging at 2.5,
    before schema induction at Phase 3).
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._logger = logger.bind(system="evo.belief_consolidation")

    async def scan_and_consolidate(
        self,
        min_precision: float = _MIN_PRECISION,
        max_volatility: float = _MAX_VOLATILITY,
        min_age_days: int = _MIN_AGE_DAYS,
    ) -> BeliefConsolidationResult:
        """
        Identify eligible beliefs and consolidate them into read-only nodes.

        Eligibility criteria:
            - precision >= min_precision (default 0.85)
            - volatility_percentile < max_volatility (default 0.2)
            - created_at >= min_age_days ago (default 30 days)
            - not already consolidated (no existing ConsolidatedBelief with same source_belief_id)

        For each eligible belief:
            1. Create a :ConsolidatedBelief node with mutable=false
            2. Create a (:ConsolidatedBelief)-[:SUPERSEDES]->(:Belief) relationship
            3. Mark the original :Belief with status="superseded_by_consolidated"

        Returns a BeliefConsolidationResult summary.
        """
        start = time.monotonic()
        now = utc_now()
        cutoff = now - __import__("datetime").timedelta(days=min_age_days)
        cutoff_iso = cutoff.isoformat()

        # Query eligible beliefs that haven't been consolidated yet
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (b:Belief)
                WHERE b.precision >= $min_precision
                  AND b.volatility_percentile < $max_volatility
                  AND b.created_at IS NOT NULL
                  AND b.created_at <= $cutoff
                  AND (b.status IS NULL OR b.status <> "superseded_by_consolidated")
                  AND NOT EXISTS {
                      MATCH (cb:ConsolidatedBelief {source_belief_id: b.id})
                  }
                RETURN b.id AS belief_id,
                       b.domain AS domain,
                       b.statement AS statement,
                       b.precision AS precision
                """,
                {
                    "min_precision": min_precision,
                    "max_volatility": max_volatility,
                    "cutoff": cutoff_iso,
                },
            )
        except Exception as exc:
            self._logger.error("belief_consolidation_scan_failed", error=str(exc))
            return BeliefConsolidationResult()

        total_scanned = len(records)
        consolidated_count = 0

        for record in records:
            belief_id = str(record.get("belief_id", ""))
            if not belief_id:
                continue

            cb_id = new_id()
            now_iso = now.isoformat()
            domain = str(record.get("domain", ""))
            statement = str(record.get("statement", ""))
            precision = float(record.get("precision", 0.0))

            # Check if a previous generation exists (for belief upgrades)
            generation = await self._get_next_generation(belief_id)

            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (cb:ConsolidatedBelief {
                        id: $cb_id,
                        source_belief_id: $belief_id,
                        domain: $domain,
                        statement: $statement,
                        precision: $precision,
                        consolidated_at: $now,
                        consolidation_generation: $generation,
                        mutable: false
                    })
                    WITH cb
                    MATCH (b:Belief {id: $belief_id})
                    SET b.status = "superseded_by_consolidated"
                    CREATE (cb)-[:SUPERSEDES]->(b)
                    """,
                    {
                        "cb_id": cb_id,
                        "belief_id": belief_id,
                        "domain": domain,
                        "statement": statement,
                        "precision": precision,
                        "now": now_iso,
                        "generation": generation,
                    },
                )
                consolidated_count += 1
            except Exception as exc:
                self._logger.warning(
                    "belief_consolidation_write_failed",
                    belief_id=belief_id,
                    error=str(exc),
                )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        result = BeliefConsolidationResult(
            beliefs_scanned=total_scanned,
            beliefs_consolidated=consolidated_count,
            duration_ms=elapsed_ms,
        )

        self._logger.info(
            "belief_consolidation_complete",
            scanned=total_scanned,
            consolidated=consolidated_count,
            duration_ms=elapsed_ms,
        )

        return result

    async def check_foundation_conflicts(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[FoundationConflict]:
        """
        Check whether any active hypotheses contradict consolidated beliefs.

        Uses a lightweight heuristic: same domain + negation pattern overlap.
        This is not LLM-based - designed to stay within the 60s consolidation budget.

        Returns a list of FoundationConflict entries (logged at high severity).
        """
        if not hypotheses:
            return []

        # Load all consolidated beliefs (compact - just id, domain, statement)
        try:
            cb_records = await self._neo4j.execute_read(
                """
                MATCH (cb:ConsolidatedBelief)
                WHERE cb.mutable = false
                RETURN cb.id AS id,
                       cb.domain AS domain,
                       cb.statement AS statement
                """,
            )
        except Exception as exc:
            self._logger.error(
                "foundation_conflict_load_failed",
                error=str(exc),
            )
            return []

        if not cb_records:
            return []

        # Build domain → consolidated beliefs index for fast lookup
        domain_index: dict[str, list[dict[str, str]]] = {}
        for r in cb_records:
            domain = str(r.get("domain", ""))
            entry = {
                "id": str(r.get("id", "")),
                "statement": str(r.get("statement", "")),
            }
            domain_index.setdefault(domain, []).append(entry)

        conflicts: list[FoundationConflict] = []

        for h in hypotheses:
            # Check consolidated beliefs in overlapping domains
            h_statement_lower = h.statement.lower()

            # Extract candidate domains: the hypothesis category maps to belief domains
            candidate_domains = _hypothesis_category_domains(h.category.value)

            for domain in candidate_domains:
                for cb in domain_index.get(domain, []):
                    if _detect_potential_contradiction(
                        h_statement_lower, cb["statement"]
                    ):
                        conflict = FoundationConflict(
                            hypothesis_id=h.id,
                            consolidated_belief_id=cb["id"],
                            hypothesis_statement=h.statement[:200],
                            consolidated_statement=cb["statement"][:200],
                        )
                        conflicts.append(conflict)

                        self._logger.warning(
                            "conflict_with_foundation",
                            hypothesis_id=h.id,
                            consolidated_belief_id=cb["id"],
                            severity="high",
                        )

        return conflicts

    async def query_consolidated_beliefs(
        self,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[ConsolidatedBelief]:
        """
        Query consolidated beliefs for retrieval (Nova, Atune).

        Prefer these over active :Belief nodes - they are read-only and
        represent high-confidence foundational knowledge.
        """
        try:
            if domain is not None:
                records = await self._neo4j.execute_read(
                    """
                    MATCH (cb:ConsolidatedBelief)
                    WHERE cb.domain = $domain
                    RETURN cb.id AS id,
                           cb.source_belief_id AS source_belief_id,
                           cb.domain AS domain,
                           cb.statement AS statement,
                           cb.precision AS precision,
                           cb.consolidated_at AS consolidated_at,
                           cb.consolidation_generation AS consolidation_generation
                    ORDER BY cb.precision DESC
                    LIMIT $limit
                    """,
                    {"domain": domain, "limit": limit},
                )
            else:
                records = await self._neo4j.execute_read(
                    """
                    MATCH (cb:ConsolidatedBelief)
                    RETURN cb.id AS id,
                           cb.source_belief_id AS source_belief_id,
                           cb.domain AS domain,
                           cb.statement AS statement,
                           cb.precision AS precision,
                           cb.consolidated_at AS consolidated_at,
                           cb.consolidation_generation AS consolidation_generation
                    ORDER BY cb.precision DESC
                    LIMIT $limit
                    """,
                    {"limit": limit},
                )
        except Exception as exc:
            self._logger.error(
                "consolidated_belief_query_failed",
                error=str(exc),
            )
            return []

        results: list[ConsolidatedBelief] = []
        for r in records:
            consolidated_at_raw = r.get("consolidated_at")
            if isinstance(consolidated_at_raw, str):
                try:
                    consolidated_at = datetime.fromisoformat(consolidated_at_raw)
                except ValueError:
                    consolidated_at = utc_now()
            elif isinstance(consolidated_at_raw, datetime):
                consolidated_at = consolidated_at_raw
            else:
                consolidated_at = utc_now()

            results.append(
                ConsolidatedBelief(
                    id=str(r.get("id", "")),
                    source_belief_id=str(r.get("source_belief_id", "")),
                    domain=str(r.get("domain", "")),
                    statement=str(r.get("statement", "")),
                    precision=float(r.get("precision", 0.0)),
                    consolidated_at=consolidated_at,
                    consolidation_generation=int(
                        r.get("consolidation_generation", 1)
                    ),
                )
            )

        return results

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _get_next_generation(self, source_belief_id: str) -> int:
        """
        Get the next consolidation generation for a belief.
        Returns 1 for first-time consolidation, or max(existing) + 1 for upgrades.
        """
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (cb:ConsolidatedBelief {source_belief_id: $belief_id})
                RETURN max(cb.consolidation_generation) AS max_gen
                """,
                {"belief_id": source_belief_id},
            )
            if records and records[0].get("max_gen") is not None:
                return int(records[0]["max_gen"]) + 1
        except Exception as exc:
            self._logger.warning(
                "consolidation_generation_lookup_failed",
                belief_id=source_belief_id,
                error=str(exc),
            )
        return 1


# ─── Heuristic Helpers ────────────────────────────────────────────────────────


def _hypothesis_category_domains(category: str) -> list[str]:
    """
    Map a hypothesis category to the belief domains it could conflict with.
    Returns a list of domain keys used in the consolidated belief index.
    """
    mapping: dict[str, list[str]] = {
        "world_model": ["capability", "process", "general", "technical_capability"],
        "self_model": ["identity", "personality", "capability"],
        "social": ["social", "relationship"],
        "procedural": ["process", "technical_capability"],
        "parameter": ["technical_capability", "capability"],
    }
    return mapping.get(category, ["general"])


def _detect_potential_contradiction(
    hypothesis_statement: str,
    consolidated_statement: str,
) -> bool:
    """
    Lightweight heuristic to detect whether a hypothesis potentially
    contradicts a consolidated belief.

    Strategy:
        1. Extract significant words (>3 chars) from both statements
        2. Require at least 2 shared significant words (topic overlap)
        3. Check if the hypothesis contains negation patterns

    This is intentionally conservative - false negatives are acceptable,
    false positives are logged for manual review.
    """
    hypothesis_lower = hypothesis_statement.lower()
    consolidated_lower = consolidated_statement.lower()

    # Extract significant words (>3 chars, no stopwords)
    _stopwords = {"that", "this", "with", "from", "they", "have", "been", "will",
                  "when", "what", "which", "their", "about", "more", "than", "also",
                  "very", "each", "does", "should", "would", "could"}
    h_words = {
        w for w in re.findall(r"\b\w{4,}\b", hypothesis_lower)
        if w not in _stopwords
    }
    c_words = {
        w for w in re.findall(r"\b\w{4,}\b", consolidated_lower)
        if w not in _stopwords
    }

    # Require topic overlap: at least 2 shared significant words
    shared = h_words & c_words
    if len(shared) < 2:
        return False

    # Check for negation patterns in the hypothesis
    return any(pattern.search(hypothesis_lower) for pattern in _NEGATION_PATTERNS)
