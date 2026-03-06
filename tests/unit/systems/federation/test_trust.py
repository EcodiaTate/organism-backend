"""
Unit tests for the Federation Trust Model.

Tests trust scoring, level transitions, decay, and violation handling.
"""

from __future__ import annotations

from primitives.common import utc_now
from primitives.federation import (
    FederationInteraction,
    FederationLink,
    FederationLinkStatus,
    InteractionOutcome,
    TrustLevel,
    ViolationType,
)
from systems.federation.trust import TrustManager

# ─── Fixtures ────────────────────────────────────────────────────


def make_link(**kwargs) -> FederationLink:
    defaults = {
        "local_instance_id": "local-001",
        "remote_instance_id": "remote-001",
        "remote_endpoint": "https://remote:8002",
        "trust_level": TrustLevel.NONE,
        "trust_score": 0.0,
        "status": FederationLinkStatus.ACTIVE,
    }
    return FederationLink(**{**defaults, **kwargs})


def make_interaction(
    link: FederationLink,
    outcome: InteractionOutcome = InteractionOutcome.SUCCESSFUL,
    trust_value: float = 1.0,
    violation_type: ViolationType | None = None,
) -> FederationInteraction:
    return FederationInteraction(
        link_id=link.id,
        remote_instance_id=link.remote_instance_id,
        interaction_type="knowledge_request",
        direction="inbound",
        outcome=outcome,
        violation_type=violation_type,
        trust_value=trust_value,
    )


# ─── Trust Level Transitions ────────────────────────────────────


class TestTrustLevelTransitions:
    """Test that trust levels transition correctly at thresholds."""

    def test_starts_at_none(self):
        link = make_link()
        assert link.trust_level == TrustLevel.NONE
        assert link.trust_score == 0.0

    def test_reaches_acquaintance_at_5(self):
        manager = TrustManager()
        link = make_link()

        # 5 successful interactions of value 1.0
        for _ in range(5):
            interaction = make_interaction(link)
            manager.update_trust(link, interaction)

        assert link.trust_score == 5.0
        assert link.trust_level == TrustLevel.ACQUAINTANCE

    def test_reaches_colleague_at_20(self):
        manager = TrustManager()
        link = make_link(trust_score=19.0, trust_level=TrustLevel.ACQUAINTANCE)

        interaction = make_interaction(link)
        manager.update_trust(link, interaction)

        assert link.trust_score == 20.0
        assert link.trust_level == TrustLevel.COLLEAGUE

    def test_reaches_partner_at_50(self):
        manager = TrustManager()
        link = make_link(trust_score=49.0, trust_level=TrustLevel.COLLEAGUE)

        interaction = make_interaction(link)
        manager.update_trust(link, interaction)

        assert link.trust_score == 50.0
        assert link.trust_level == TrustLevel.PARTNER

    def test_reaches_ally_at_100(self):
        manager = TrustManager()
        link = make_link(trust_score=99.0, trust_level=TrustLevel.PARTNER)

        interaction = make_interaction(link)
        manager.update_trust(link, interaction)

        assert link.trust_score == 100.0
        assert link.trust_level == TrustLevel.ALLY

    def test_below_threshold_stays_at_none(self):
        manager = TrustManager()
        link = make_link()

        for _ in range(4):
            interaction = make_interaction(link)
            manager.update_trust(link, interaction)

        assert link.trust_score == 4.0
        assert link.trust_level == TrustLevel.NONE


# ─── Violation Handling ──────────────────────────────────────────


class TestViolationHandling:
    """Test that violations correctly penalize trust."""

    def test_violation_costs_3x(self):
        manager = TrustManager()
        link = make_link(trust_score=25.0, trust_level=TrustLevel.COLLEAGUE)

        interaction = make_interaction(
            link,
            outcome=InteractionOutcome.VIOLATION,
            violation_type=ViolationType.PROTOCOL_VIOLATION,
            trust_value=2.0,
        )
        manager.update_trust(link, interaction)

        # 25.0 - (2.0 * 3.0) = 19.0
        assert link.trust_score == 19.0
        assert link.trust_level == TrustLevel.ACQUAINTANCE

    def test_privacy_breach_resets_to_zero(self):
        manager = TrustManager()
        link = make_link(trust_score=80.0, trust_level=TrustLevel.PARTNER)

        interaction = make_interaction(
            link,
            outcome=InteractionOutcome.VIOLATION,
            violation_type=ViolationType.PRIVACY_BREACH,
            trust_value=1.0,
        )
        manager.update_trust(link, interaction)

        assert link.trust_score == 0.0
        assert link.trust_level == TrustLevel.NONE

    def test_trust_score_never_goes_below_zero(self):
        manager = TrustManager()
        link = make_link(trust_score=2.0)

        interaction = make_interaction(
            link,
            outcome=InteractionOutcome.VIOLATION,
            violation_type=ViolationType.DECEPTION,
            trust_value=5.0,
        )
        manager.update_trust(link, interaction)

        assert link.trust_score == 0.0

    def test_violation_count_increments(self):
        manager = TrustManager()
        link = make_link(trust_score=50.0, trust_level=TrustLevel.PARTNER)

        interaction = make_interaction(
            link,
            outcome=InteractionOutcome.VIOLATION,
            violation_type=ViolationType.CONSENT_VIOLATION,
        )
        manager.update_trust(link, interaction)

        assert link.violation_count == 1
        assert link.failed_interactions == 1

    def test_timeout_costs_half(self):
        manager = TrustManager()
        link = make_link(trust_score=10.0, trust_level=TrustLevel.ACQUAINTANCE)

        interaction = make_interaction(
            link,
            outcome=InteractionOutcome.TIMEOUT,
            trust_value=2.0,
        )
        manager.update_trust(link, interaction)

        # 10.0 - (2.0 * 0.5) = 9.0
        assert link.trust_score == 9.0


# ─── Trust Decay ─────────────────────────────────────────────────


class TestTrustDecay:
    """Test time-based trust decay for inactive links."""

    def test_no_decay_within_24_hours(self):
        manager = TrustManager(trust_decay_enabled=True, trust_decay_rate_per_day=1.0)
        link = make_link(trust_score=50.0, trust_level=TrustLevel.PARTNER)
        link.last_communication = utc_now()

        manager.apply_decay(link)
        assert link.trust_score == 50.0  # No change

    def test_decay_disabled(self):
        manager = TrustManager(trust_decay_enabled=False)
        link = make_link(trust_score=50.0, trust_level=TrustLevel.PARTNER)

        from datetime import timedelta
        link.last_communication = utc_now() - timedelta(days=30)

        manager.apply_decay(link)
        assert link.trust_score == 50.0  # No change

    def test_no_decay_without_last_communication(self):
        manager = TrustManager(trust_decay_enabled=True)
        link = make_link(trust_score=50.0)
        link.last_communication = None

        manager.apply_decay(link)
        assert link.trust_score == 50.0


# ─── Max Trust Level ─────────────────────────────────────────────


class TestMaxTrustLevel:
    """Test that trust level is capped at the configured maximum."""

    def test_max_trust_level_caps(self):
        manager = TrustManager(max_trust_level=TrustLevel.COLLEAGUE)
        link = make_link(trust_score=99.0)

        interaction = make_interaction(link)
        manager.update_trust(link, interaction)

        assert link.trust_score == 100.0
        assert link.trust_level == TrustLevel.COLLEAGUE  # Capped, not ALLY


# ─── Utility Methods ─────────────────────────────────────────────


class TestUtilityMethods:
    """Test helper methods on TrustManager."""

    def test_mean_trust_empty(self):
        manager = TrustManager()
        assert manager.mean_trust([]) == 0.0

    def test_mean_trust_calculated(self):
        manager = TrustManager()
        links = [
            make_link(trust_score=10.0),
            make_link(trust_score=30.0),
            make_link(trust_score=20.0),
        ]
        assert manager.mean_trust(links) == 20.0

    def test_can_coordinate(self):
        manager = TrustManager()
        link_none = make_link(trust_level=TrustLevel.NONE)
        link_acq = make_link(trust_level=TrustLevel.ACQUAINTANCE)
        link_col = make_link(trust_level=TrustLevel.COLLEAGUE)
        link_partner = make_link(trust_level=TrustLevel.PARTNER)

        assert not manager.can_coordinate(link_none)
        assert not manager.can_coordinate(link_acq)
        assert manager.can_coordinate(link_col)
        assert manager.can_coordinate(link_partner)

    def test_score_to_level(self):
        assert TrustManager._score_to_level(0.0) == TrustLevel.NONE
        assert TrustManager._score_to_level(4.9) == TrustLevel.NONE
        assert TrustManager._score_to_level(5.0) == TrustLevel.ACQUAINTANCE
        assert TrustManager._score_to_level(19.9) == TrustLevel.ACQUAINTANCE
        assert TrustManager._score_to_level(20.0) == TrustLevel.COLLEAGUE
        assert TrustManager._score_to_level(49.9) == TrustLevel.COLLEAGUE
        assert TrustManager._score_to_level(50.0) == TrustLevel.PARTNER
        assert TrustManager._score_to_level(99.9) == TrustLevel.PARTNER
        assert TrustManager._score_to_level(100.0) == TrustLevel.ALLY
        assert TrustManager._score_to_level(500.0) == TrustLevel.ALLY
