"""
EcodiaOS - Equor Constitutional Template Library

Read-only template library for the Arbitrage Reflex Arc.

Templates are ConstitutionalTemplate instances that Equor has pre-approved
as constitutionally safe. The library supports:
  1. O(1) lookup by template_id
  2. Approximate signature matching (for Atune's pattern detection)
  3. Circuit breaker per template (3 consecutive failures → disable)
  4. TTL-based staleness eviction (templates not re-evaluated in 60s expire)
  5. Metrics for monitoring

Storage: in-memory dict backed by Redis cache. The canonical source is
Equor's Neo4j governance records, but the hot path reads from memory only.

Thread safety: single-writer (Equor), multi-reader (Atune fast-path).
All mutations go through Equor; the library is effectively append-only
with soft deactivation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now

if TYPE_CHECKING:
    from primitives.fast_path import ConstitutionalTemplate

logger = structlog.get_logger()

# Circuit breaker: disable template after this many consecutive failures
_CIRCUIT_BREAKER_THRESHOLD = 3
# Templates not re-evaluated in this window (seconds) are considered stale
_STALENESS_WINDOW_S = 60.0
# Minimum approval confidence for fast-path eligibility
_MIN_APPROVAL_CONFIDENCE = 0.9


class TemplateLibrary:
    """
    In-memory constitutional template library.

    Atune queries this on the hot path (≤20ms budget). All reads are
    lock-free dict lookups. Writes happen only via Equor's governance
    flow (register/deactivate/record_outcome).
    """

    def __init__(self) -> None:
        # template_id → ConstitutionalTemplate
        self._templates: dict[str, ConstitutionalTemplate] = {}
        # signature_hash → template_id (for fast approximate lookup)
        self._hash_index: dict[str, str] = {}
        self._logger = logger.bind(system="equor.template_library")

        # Metrics
        self._total_lookups: int = 0
        self._total_matches: int = 0
        self._total_misses: int = 0

    # ─── Read Path (called by Atune, must be fast) ────────────────

    def find_match(
        self,
        candidate_signature: dict[str, Any],
        *,
        min_confidence: float = _MIN_APPROVAL_CONFIDENCE,
        tolerance: float = 0.1,
    ) -> ConstitutionalTemplate | None:
        """
        Find a pre-approved template matching the candidate signature.

        Returns the highest-confidence matching template, or None.
        Budget: ≤20ms for up to 100 templates.

        Filters:
          - active=True
          - approval_confidence >= min_confidence
          - not stale (last_approved_at within staleness window)
          - approximate signature match within tolerance
        """
        self._total_lookups += 1
        best: ConstitutionalTemplate | None = None

        for template in self._templates.values():
            # Skip inactive (circuit-breaker tripped)
            if not template.active:
                continue

            # Skip low-confidence templates
            if template.approval_confidence < min_confidence:
                continue

            # Skip stale templates (not re-evaluated recently)
            age_s = (utc_now() - template.last_approved_at).total_seconds()
            if age_s > _STALENESS_WINDOW_S:
                continue

            # Approximate signature match
            if not template.matches(candidate_signature, tolerance=tolerance):
                continue

            # Pick highest confidence
            if best is None or template.approval_confidence > best.approval_confidence:
                best = template

        if best is not None:
            self._total_matches += 1
            self._logger.debug(
                "template_matched",
                template_id=best.template_id,
                confidence=best.approval_confidence,
            )
        else:
            self._total_misses += 1

        return best

    def get(self, template_id: str) -> ConstitutionalTemplate | None:
        """O(1) lookup by template_id."""
        return self._templates.get(template_id)

    # ─── Write Path (called by Equor governance) ──────────────────

    def register(self, template: ConstitutionalTemplate) -> None:
        """
        Register a new template or update an existing one.

        Called by Equor after deliberative review approves a strategy.
        Rebuilds the hash index for the affected template.
        """
        self._templates[template.template_id] = template
        self._hash_index[template.signature_hash] = template.template_id
        self._logger.info(
            "template_registered",
            template_id=template.template_id,
            confidence=template.approval_confidence,
            max_capital=template.max_capital_per_execution,
        )

    def deactivate(self, template_id: str, reason: str = "") -> bool:
        """
        Soft-deactivate a template. Returns False if not found.

        Called by:
          - Circuit breaker (3 consecutive failures)
          - Equor governance (explicit revocation)
          - Staleness eviction
        """
        template = self._templates.get(template_id)
        if template is None:
            return False

        template.active = False
        self._logger.warning(
            "template_deactivated",
            template_id=template_id,
            reason=reason,
            consecutive_failures=template.consecutive_failures,
            total_executions=template.total_executions,
        )
        return True

    def reactivate(self, template_id: str) -> bool:
        """
        Reactivate a previously disabled template.

        Called by Nova after approving a recovery for a circuit-broken template.
        Resets the failure counter.
        """
        template = self._templates.get(template_id)
        if template is None:
            return False

        template.active = True
        template.consecutive_failures = 0
        template.last_approved_at = utc_now()
        self._logger.info(
            "template_reactivated",
            template_id=template_id,
        )
        return True

    # ─── Outcome Recording (called by Axon after fast-path execution) ─

    def record_success(self, template_id: str, capital_deployed: float) -> None:
        """
        Record a successful fast-path execution.
        Resets the consecutive failure counter.
        """
        template = self._templates.get(template_id)
        if template is None:
            return

        template.consecutive_failures = 0
        template.total_executions += 1
        template.total_capital_deployed += capital_deployed
        template.last_executed_at = utc_now()

    def record_failure(self, template_id: str) -> None:
        """
        Record a failed fast-path execution.

        Increments the consecutive failure counter. If it reaches the
        circuit breaker threshold, deactivates the template.
        """
        template = self._templates.get(template_id)
        if template is None:
            return

        template.consecutive_failures += 1
        template.total_executions += 1
        template.last_executed_at = utc_now()

        if template.consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            self.deactivate(
                template_id,
                reason=(
                    f"Circuit breaker: {template.consecutive_failures} "
                    f"consecutive failures"
                ),
            )

    # ─── Maintenance ──────────────────────────────────────────────

    def evict_stale(self) -> int:
        """
        Deactivate templates whose last_approved_at exceeds the staleness window.

        Called periodically by Atune (every 60 seconds) to forget templates
        that Equor hasn't re-evaluated.

        Returns the number of templates evicted.
        """
        evicted = 0
        now = utc_now()
        for template in self._templates.values():
            if not template.active:
                continue
            age_s = (now - template.last_approved_at).total_seconds()
            if age_s > _STALENESS_WINDOW_S:
                self.deactivate(template.template_id, reason="staleness_eviction")
                evicted += 1

        if evicted > 0:
            self._logger.info("stale_templates_evicted", count=evicted)
        return evicted

    # ─── Introspection ────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._templates.values() if t.active)

    @property
    def total_count(self) -> int:
        return len(self._templates)

    def list_active(self) -> list[ConstitutionalTemplate]:
        return [t for t in self._templates.values() if t.active]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_templates": len(self._templates),
            "active_templates": self.active_count,
            "total_lookups": self._total_lookups,
            "total_matches": self._total_matches,
            "total_misses": self._total_misses,
            "hit_rate": (
                self._total_matches / max(1, self._total_lookups)
            ),
        }
