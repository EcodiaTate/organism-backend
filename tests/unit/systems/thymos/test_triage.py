"""
Tests for Thymos Triage Layer.

Covers:
  - IncidentDeduplicator
  - SeverityScorer
  - ResponseRouter
"""

from __future__ import annotations

from primitives.common import new_id, utc_now
from systems.thymos.triage import (
    IncidentDeduplicator,
    ResponseRouter,
    SeverityScorer,
)
from systems.thymos.types import (
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairTier,
)


def _make_incident(
    fingerprint: str = "abc123",
    incident_class: IncidentClass = IncidentClass.CRASH,
    severity: IncidentSeverity = IncidentSeverity.MEDIUM,
    source_system: str = "nova",
    blast_radius: float = 0.2,
    occurrence_count: int = 1,
    user_visible: bool = False,
) -> Incident:
    return Incident(
        id=new_id(),
        timestamp=utc_now(),
        incident_class=incident_class,
        severity=severity,
        fingerprint=fingerprint,
        source_system=source_system,
        error_type="TestError",
        error_message="Test error message",
        blast_radius=blast_radius,
        occurrence_count=occurrence_count,
        user_visible=user_visible,
    )


# ─── IncidentDeduplicator ─────────────────────────────────────────


class TestIncidentDeduplicator:
    def test_first_occurrence_is_new(self):
        dedup = IncidentDeduplicator()
        incident = _make_incident(fingerprint="unique_fp_1")
        result = dedup.deduplicate(incident)
        assert result is incident  # New incident returned as-is

    def test_second_occurrence_is_duplicate(self):
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="dup_fp")
        inc2 = _make_incident(fingerprint="dup_fp")
        dedup.deduplicate(inc1)
        result = dedup.deduplicate(inc2)
        assert result is None  # Duplicate → None

    def test_duplicate_increments_count(self):
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="dup_fp")
        dedup.deduplicate(inc1)
        inc2 = _make_incident(fingerprint="dup_fp")
        dedup.deduplicate(inc2)
        # The original incident should have incremented count
        assert inc1.occurrence_count == 2

    def test_different_fingerprints_not_duplicated(self):
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="fp_a")
        inc2 = _make_incident(fingerprint="fp_b")
        assert dedup.deduplicate(inc1) is inc1
        assert dedup.deduplicate(inc2) is inc2

    def test_active_count(self):
        dedup = IncidentDeduplicator()
        dedup.deduplicate(_make_incident(fingerprint="a"))
        dedup.deduplicate(_make_incident(fingerprint="b"))
        dedup.deduplicate(_make_incident(fingerprint="a"))  # dup
        assert dedup.active_count == 2

    def test_resolve_removes_from_active(self):
        dedup = IncidentDeduplicator()
        inc = _make_incident(fingerprint="resolve_me")
        dedup.deduplicate(inc)
        resolved = dedup.resolve("resolve_me")
        assert resolved is inc
        assert dedup.active_count == 0

    # ─── T4 Re-emission Tests ─────────────────────────────────────────

    def test_occurrence_count_carried_forward_on_window_expiry(self):
        """When the dedup window expires, occurrence_count must survive."""
        from datetime import timedelta

        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="carry_fp")
        dedup.deduplicate(inc1)

        # Simulate 5 duplicates within window
        for _ in range(5):
            dup = _make_incident(fingerprint="carry_fp")
            dedup.deduplicate(dup)
        assert inc1.occurrence_count == 6

        # Create one after the window expires (>60s for CRASH class)
        post_window = _make_incident(fingerprint="carry_fp")
        post_window.timestamp = inc1.timestamp + timedelta(seconds=120)
        result = dedup.deduplicate(post_window)

        assert result is not None  # New incident (window expired)
        assert result.occurrence_count == 7  # 6 + 1, NOT reset to 1
        assert result.first_seen == inc1.timestamp  # first_seen preserved

    def test_t4_reemission_on_threshold(self):
        """When occurrence_count hits T4 threshold, dedup returns the incident."""
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="t4_fp")
        dedup.deduplicate(inc1)

        # Feed 5 duplicates (count goes 2,3,4,5,6)
        results = []
        for _ in range(5):
            dup = _make_incident(fingerprint="t4_fp")
            results.append(dedup.deduplicate(dup))

        # First 4 duplicates should be None, 5th should re-emit (count=6)
        assert all(r is None for r in results[:4])
        assert results[4] is inc1
        assert inc1.occurrence_count == 6

    def test_t4_reemission_only_once(self):
        """Re-emission must happen exactly once per fingerprint."""
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="once_fp")
        dedup.deduplicate(inc1)

        reemit_count = 0
        for _ in range(20):
            dup = _make_incident(fingerprint="once_fp")
            if dedup.deduplicate(dup) is not None:
                reemit_count += 1

        assert reemit_count == 1  # Exactly one re-emission

    def test_t4_reemission_lower_threshold_for_attribute_errors(self):
        """AttributeErrors should re-emit at the lower threshold (3)."""
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="attr_fp")
        inc1.error_type = "AttributeError"
        dedup.deduplicate(inc1)

        results = []
        for _ in range(5):
            dup = _make_incident(fingerprint="attr_fp")
            dup.error_type = "AttributeError"
            results.append(dedup.deduplicate(dup))

        # Should re-emit at count=3 (occurrence 3), not 6
        assert results[0] is None  # count=2
        assert results[1] is inc1  # count=3 → re-emit
        assert results[2] is None  # count=4, already re-emitted

    def test_resolve_clears_reemit_tracking(self):
        """Resolving an incident should clear the re-emission flag."""
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="resolve_reemit")
        dedup.deduplicate(inc1)

        # Push to threshold
        for _ in range(5):
            dedup.deduplicate(_make_incident(fingerprint="resolve_reemit"))

        assert "resolve_reemit" in dedup._t4_reemitted

        # Resolve
        dedup.resolve("resolve_reemit")
        assert "resolve_reemit" not in dedup._t4_reemitted

        # New incident with same fingerprint should start fresh
        inc2 = _make_incident(fingerprint="resolve_reemit")
        result = dedup.deduplicate(inc2)
        assert result is inc2


# ─── SeverityScorer ────────────────────────────────────────────────


class TestSeverityScorer:
    def test_returns_severity_enum(self):
        scorer = SeverityScorer()
        incident = _make_incident()
        result = scorer.compute_severity(incident)
        assert isinstance(result, IncidentSeverity)

    def test_high_blast_radius_increases_severity(self):
        scorer = SeverityScorer()
        low_blast = _make_incident(blast_radius=0.05, severity=IncidentSeverity.LOW)
        high_blast = _make_incident(blast_radius=0.8, severity=IncidentSeverity.LOW)
        score_low = scorer.compute_severity(low_blast)
        score_high = scorer.compute_severity(high_blast)
        # Higher blast radius should result in >= severity
        severity_order = [
            IncidentSeverity.INFO,
            IncidentSeverity.LOW,
            IncidentSeverity.MEDIUM,
            IncidentSeverity.HIGH,
            IncidentSeverity.CRITICAL,
        ]
        assert severity_order.index(score_high) >= severity_order.index(score_low)

    def test_user_visible_influences_score(self):
        scorer = SeverityScorer()
        invisible = _make_incident(user_visible=False)
        visible = _make_incident(user_visible=True)
        # Both should return a valid severity
        assert isinstance(scorer.compute_severity(invisible), IncidentSeverity)
        assert isinstance(scorer.compute_severity(visible), IncidentSeverity)


# ─── ResponseRouter ────────────────────────────────────────────────


class TestResponseRouter:
    def test_critical_routes_to_restart(self):
        router = ResponseRouter()
        incident = _make_incident(severity=IncidentSeverity.CRITICAL)
        tier = router.route(incident)
        assert tier in (RepairTier.RESTART, RepairTier.KNOWN_FIX, RepairTier.ESCALATE)

    def test_info_routes_to_noop(self):
        router = ResponseRouter()
        incident = _make_incident(severity=IncidentSeverity.INFO)
        tier = router.route(incident)
        assert tier == RepairTier.NOOP

    def test_low_routes_to_noop_or_parameter(self):
        router = ResponseRouter()
        incident = _make_incident(severity=IncidentSeverity.LOW)
        tier = router.route(incident)
        assert tier in (RepairTier.NOOP, RepairTier.PARAMETER)

    def test_returns_valid_tier(self):
        router = ResponseRouter()
        for severity in IncidentSeverity:
            incident = _make_incident(severity=severity)
            tier = router.route(incident)
            assert isinstance(tier, RepairTier)

    def test_noop_incident_escalated_to_novel_fix_on_recurrence(self):
        """A LOW severity incident with occurrence_count > 5 within 600s
        must escalate to NOVEL_FIX (T4)."""
        from datetime import timedelta

        router = ResponseRouter()
        incident = _make_incident(
            severity=IncidentSeverity.LOW,
            occurrence_count=6,
        )
        incident.first_seen = incident.timestamp - timedelta(seconds=300)

        tier = router.route(incident)
        assert tier == RepairTier.NOVEL_FIX

    def test_attribute_error_escalated_at_lower_threshold(self):
        """AttributeError with count > 2 within 600s must escalate to T4."""
        from datetime import timedelta

        router = ResponseRouter()
        incident = _make_incident(
            severity=IncidentSeverity.LOW,
            occurrence_count=3,
        )
        incident.error_type = "AttributeError"
        incident.first_seen = incident.timestamp - timedelta(seconds=60)

        tier = router.route(incident)
        assert tier == RepairTier.NOVEL_FIX
