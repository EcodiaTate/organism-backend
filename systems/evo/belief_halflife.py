"""
EcodiaOS — Belief Half-Life System

Radioisotope decay model for knowledge freshness. Each belief is stamped
with a decay constant derived from its knowledge domain's historical
volatility, enabling the organism to automatically schedule re-verification
of beliefs as they "age".

Core formula:
    belief_age_factor = 2^(-elapsed_time / half_life)

Where:
    elapsed_time = now - last_verified
    half_life = domain-specific half-life in days
    decay_constant = ln(2) / half_life

When age_factor drops below 0.5 (one half-life has passed), the belief
is marked stale and queued for re-verification.

Domain half-lives are tunable and can be learned from historical
belief-change frequency (volatility_percentile).

Integration:
    - Evo consolidation Phase 2.5 (belief aging, between hypothesis review
      and schema induction)
    - Nova/Axon for epistemic action routing of stale beliefs
    - Belief creation/update in Nova and memory birth flows
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, utc_now

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# ─── Domain Half-Life Registry ───────────────────────────────────────────────

# Default half-life values per knowledge domain (days).
# These represent how quickly knowledge in each domain typically becomes
# unreliable. Tunable at runtime.
DEFAULT_DOMAIN_HALFLIFES: dict[str, float] = {
    # Fast-decaying (hours to days)
    "sentiment": 0.3,           # Mood/emotional state — ~7 hours
    "mood": 0.3,
    "emotional": 0.3,
    "engagement": 1.0,          # User engagement level — 1 day
    "attention": 0.5,           # What someone is focused on — 12 hours
    "availability": 0.5,        # Whether someone is available — 12 hours
    "schedule": 3.0,            # Near-term plans — 3 days
    "price": 0.04,              # Market price — ~1 hour
    "market_state": 0.08,       # Market conditions — ~2 hours

    # Medium-decaying (days to weeks)
    "preference": 14.0,         # User preferences — 2 weeks
    "opinion": 7.0,             # Stated opinions — 1 week
    "project_status": 5.0,      # Project progress — 5 days
    "relationship": 30.0,       # Interpersonal dynamics — 1 month
    "context": 2.0,             # Conversational context — 2 days
    "social": 14.0,             # Social dynamics — 2 weeks
    "request": 1.0,             # Active requests — 1 day
    "task_status": 3.0,         # Task completion state — 3 days

    # Slow-decaying (weeks to months)
    "capability": 90.0,         # Technical capabilities — 3 months
    "technical_capability": 90.0,
    "process": 60.0,            # Process/workflow knowledge — 2 months
    "policy": 90.0,             # Organisational policy — 3 months
    "skill": 120.0,             # Learned skills — 4 months
    "identity": 365.0,          # Core identity facts — 1 year
    "personality": 180.0,       # Personality traits — 6 months

    # Very slow (months to years)
    "physical_law": 36500.0,    # Physical constants — 100 years
    "mathematical": 36500.0,    # Mathematical truths — 100 years
    "definition": 3650.0,       # Definitions — 10 years
    "historical": 3650.0,       # Historical facts — 10 years
    "geographical": 1825.0,     # Geography — 5 years
}

# Fallback half-life for domains not in the registry
_DEFAULT_HALFLIFE_DAYS: float = 30.0

# Threshold below which a belief is considered stale
_STALE_AGE_FACTOR: float = 0.5  # One half-life has passed


# ─── Models ──────────────────────────────────────────────────────────────────


class BeliefHalfLife(EOSBaseModel):
    """
    Half-life metadata for a single belief node.

    Attached to beliefs in the Neo4j graph and used by the decay scorer
    to determine freshness.
    """

    domain: str = ""
    half_life_days: float = _DEFAULT_HALFLIFE_DAYS
    decay_constant: float = Field(default=0.0)  # Computed: ln(2) / half_life
    volatility_percentile: float = 0.5          # 0–1; computed from historical change frequency
    last_verified: datetime = Field(default_factory=utc_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute decay constant from half-life."""
        if self.decay_constant == 0.0 and self.half_life_days > 0:
            self.decay_constant = math.log(2) / self.half_life_days


class StaleBelief(EOSBaseModel):
    """A belief that has crossed its half-life threshold and needs re-verification."""

    belief_id: str
    domain: str = ""
    statement: str = ""
    age_factor: float = 0.0
    half_life_days: float = _DEFAULT_HALFLIFE_DAYS
    elapsed_days: float = 0.0
    last_verified: datetime = Field(default_factory=utc_now)
    priority: float = 0.0  # Higher = more urgent to re-verify


class BeliefAgingResult(EOSBaseModel):
    """Summary of one belief-aging pass during consolidation."""

    beliefs_scanned: int = 0
    beliefs_stale: int = 0
    beliefs_critical: int = 0       # age_factor < 0.25 (two half-lives)
    stale_beliefs: list[StaleBelief] = Field(default_factory=list)
    duration_ms: int = 0


# ─── Domain Half-Life Lookup ─────────────────────────────────────────────────


def get_halflife_for_domain(domain: str) -> float:
    """
    Look up the half-life for a belief domain.

    Checks exact match first, then prefix match, then falls back
    to the default.
    """
    normalised = domain.lower().strip()

    # Exact match
    if normalised in DEFAULT_DOMAIN_HALFLIFES:
        return DEFAULT_DOMAIN_HALFLIFES[normalised]

    # Prefix match (e.g. "user.emotional_state" → "emotional")
    for key, value in DEFAULT_DOMAIN_HALFLIFES.items():
        if key in normalised or normalised.startswith(key):
            return value

    return _DEFAULT_HALFLIFE_DAYS


def compute_halflife_metadata(domain: str) -> BeliefHalfLife:
    """Create a BeliefHalfLife for a given domain with defaults."""
    half_life = get_halflife_for_domain(domain)
    return BeliefHalfLife(
        domain=domain,
        half_life_days=half_life,
        last_verified=utc_now(),
    )


# ─── Decay Scorer ────────────────────────────────────────────────────────────


def compute_age_factor(
    half_life_days: float,
    last_verified: datetime,
    now: datetime | None = None,
) -> float:
    """
    Compute the belief age factor using radioisotope decay:
        age_factor = 2^(-elapsed_time / half_life)

    Returns a value in (0, 1]:
        1.0  = just verified
        0.5  = one half-life elapsed
        0.25 = two half-lives elapsed
        ...  → 0 as time → ∞
    """
    if half_life_days <= 0:
        return 0.0

    now = now or utc_now()
    elapsed = (now - last_verified).total_seconds() / 86400.0  # days

    if elapsed <= 0:
        return 1.0

    return math.pow(2, -elapsed / half_life_days)


def is_stale(
    half_life_days: float,
    last_verified: datetime,
    threshold: float = _STALE_AGE_FACTOR,
    now: datetime | None = None,
) -> bool:
    """Return True if the belief has crossed its staleness threshold."""
    return compute_age_factor(half_life_days, last_verified, now) < threshold


# ─── Belief Aging Scanner ────────────────────────────────────────────────────


class BeliefAgingScanner:
    """
    Scans the Neo4j belief graph for beliefs that have crossed their
    half-life threshold and need re-verification.

    Run during Evo consolidation Phase 2.5 (between hypothesis review
    and schema induction).
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._logger = logger.bind(system="evo.belief_halflife")

    async def scan_stale_beliefs(
        self,
        threshold: float = _STALE_AGE_FACTOR,
    ) -> BeliefAgingResult:
        """
        Query all beliefs with half-life metadata and identify stale ones.

        Returns a BeliefAgingResult with the list of stale beliefs sorted
        by priority (lowest age_factor = highest priority).
        """
        import time as _time

        start = _time.monotonic()

        now = utc_now()

        try:
            # Query beliefs that have half-life metadata
            records = await self._neo4j.execute_read(
                """
                MATCH (b:Belief)
                WHERE b.half_life_days IS NOT NULL
                  AND b.last_verified IS NOT NULL
                RETURN b.id AS belief_id,
                       b.domain AS domain,
                       b.statement AS statement,
                       b.half_life_days AS half_life_days,
                       b.last_verified AS last_verified,
                       b.precision AS precision
                """,
            )
        except Exception as exc:
            self._logger.error("belief_aging_scan_failed", error=str(exc))
            return BeliefAgingResult()

        stale_beliefs: list[StaleBelief] = []
        total_scanned = 0
        critical_count = 0

        for record in records:
            total_scanned += 1

            half_life = float(record.get("half_life_days", _DEFAULT_HALFLIFE_DAYS))
            last_verified_raw = record.get("last_verified")

            if last_verified_raw is None:
                continue

            # Parse last_verified — Neo4j returns datetime or ISO string
            if isinstance(last_verified_raw, str):
                try:
                    last_verified_dt = datetime.fromisoformat(last_verified_raw)
                except ValueError:
                    continue
            elif isinstance(last_verified_raw, datetime):
                last_verified_dt = last_verified_raw
            else:
                continue

            age_factor = compute_age_factor(half_life, last_verified_dt, now)
            elapsed_days = (now - last_verified_dt).total_seconds() / 86400.0

            if age_factor < threshold:
                precision = float(record.get("precision", 0.5))
                # Priority: lower age_factor + higher precision = more urgent
                # (high-confidence beliefs that are aging are more dangerous)
                priority = (1.0 - age_factor) * precision

                stale = StaleBelief(
                    belief_id=str(record.get("belief_id", "")),
                    domain=str(record.get("domain", "")),
                    statement=str(record.get("statement", ""))[:200],
                    age_factor=round(age_factor, 4),
                    half_life_days=half_life,
                    elapsed_days=round(elapsed_days, 2),
                    last_verified=last_verified_dt,
                    priority=round(priority, 4),
                )
                stale_beliefs.append(stale)

                if age_factor < 0.25:  # Two half-lives — critical
                    critical_count += 1

        # Sort by priority descending (most urgent first)
        stale_beliefs.sort(key=lambda s: s.priority, reverse=True)

        elapsed_ms = int((_time.monotonic() - start) * 1000)

        result = BeliefAgingResult(
            beliefs_scanned=total_scanned,
            beliefs_stale=len(stale_beliefs),
            beliefs_critical=critical_count,
            stale_beliefs=stale_beliefs,
            duration_ms=elapsed_ms,
        )

        self._logger.info(
            "belief_aging_scan_complete",
            scanned=total_scanned,
            stale=len(stale_beliefs),
            critical=critical_count,
            duration_ms=elapsed_ms,
        )

        return result

    async def mark_verified(self, belief_id: str) -> None:
        """
        Reset a belief's last_verified timestamp after successful re-verification.
        Called after Nova/Axon confirms the belief is still valid.
        """
        now_iso = utc_now().isoformat()
        try:
            await self._neo4j.execute_write(
                """
                MATCH (b:Belief {id: $belief_id})
                SET b.last_verified = $now
                """,
                {"belief_id": belief_id, "now": now_iso},
            )
            self._logger.debug("belief_reverified", belief_id=belief_id)
        except Exception as exc:
            self._logger.warning(
                "belief_reverification_failed",
                belief_id=belief_id,
                error=str(exc),
            )

    async def query_unreliable_in(
        self,
        hours: float,
        threshold: float = _STALE_AGE_FACTOR,
    ) -> list[StaleBelief]:
        """
        Dashboard query: "Which beliefs will be unreliable in N hours?"

        Projects the age factor forward by `hours` and returns beliefs
        that will cross the staleness threshold within that window.
        """
        future = utc_now() + timedelta(hours=hours)

        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (b:Belief)
                WHERE b.half_life_days IS NOT NULL
                  AND b.last_verified IS NOT NULL
                RETURN b.id AS belief_id,
                       b.domain AS domain,
                       b.statement AS statement,
                       b.half_life_days AS half_life_days,
                       b.last_verified AS last_verified,
                       b.precision AS precision
                """,
            )
        except Exception as exc:
            self._logger.error("unreliable_in_query_failed", error=str(exc))
            return []

        now = utc_now()
        will_be_stale: list[StaleBelief] = []

        for record in records:
            half_life = float(record.get("half_life_days", _DEFAULT_HALFLIFE_DAYS))
            last_verified_raw = record.get("last_verified")
            if last_verified_raw is None:
                continue

            if isinstance(last_verified_raw, str):
                try:
                    last_verified_dt = datetime.fromisoformat(last_verified_raw)
                except ValueError:
                    continue
            elif isinstance(last_verified_raw, datetime):
                last_verified_dt = last_verified_raw
            else:
                continue

            # Current age factor
            current_factor = compute_age_factor(half_life, last_verified_dt, now)
            # Projected age factor
            future_factor = compute_age_factor(half_life, last_verified_dt, future)

            # Include beliefs that are currently fine but will be stale by the deadline
            if current_factor >= threshold and future_factor < threshold:
                elapsed_days = (future - last_verified_dt).total_seconds() / 86400.0
                precision = float(record.get("precision", 0.5))

                will_be_stale.append(StaleBelief(
                    belief_id=str(record.get("belief_id", "")),
                    domain=str(record.get("domain", "")),
                    statement=str(record.get("statement", ""))[:200],
                    age_factor=round(future_factor, 4),
                    half_life_days=half_life,
                    elapsed_days=round(elapsed_days, 2),
                    last_verified=last_verified_dt,
                    priority=round((1.0 - future_factor) * precision, 4),
                ))

        will_be_stale.sort(key=lambda s: s.priority, reverse=True)

        self._logger.info(
            "unreliable_in_query_complete",
            hours=hours,
            beliefs_will_be_stale=len(will_be_stale),
        )

        return will_be_stale


# ─── Neo4j Belief Stamping ───────────────────────────────────────────────────


async def stamp_belief_halflife(
    neo4j: Neo4jClient,
    belief_id: str,
    domain: str,
    half_life_days: float | None = None,
) -> None:
    """
    Stamp a belief node in Neo4j with half-life metadata.

    Called during belief creation (Nova belief_updater, memory birth flows).
    If half_life_days is not provided, it is looked up from the domain registry.
    """
    if half_life_days is None:
        half_life_days = get_halflife_for_domain(domain)

    decay_constant = math.log(2) / half_life_days if half_life_days > 0 else 0.0
    now_iso = utc_now().isoformat()

    try:
        await neo4j.execute_write(
            """
            MATCH (b:Belief {id: $belief_id})
            SET b.domain = $domain,
                b.half_life_days = $half_life_days,
                b.decay_constant = $decay_constant,
                b.last_verified = $now,
                b.volatility_percentile = 0.5
            """,
            {
                "belief_id": belief_id,
                "domain": domain,
                "half_life_days": half_life_days,
                "decay_constant": decay_constant,
                "now": now_iso,
            },
        )
    except Exception as exc:
        logger.warning(
            "belief_halflife_stamp_failed",
            belief_id=belief_id,
            error=str(exc),
        )
