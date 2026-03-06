"""Unit tests for AudienceProfiler."""

from __future__ import annotations

import pytest

from systems.voxis.audience import AudienceProfiler
from systems.voxis.types import StrategyParams


@pytest.fixture
def profiler() -> AudienceProfiler:
    return AudienceProfiler()


class TestProfileBuilding:
    def test_builds_profile_for_new_individual(self, profiler: AudienceProfiler) -> None:
        profile = profiler.build_profile(
            addressee_id="user-1",
            addressee_name="Alice",
            interaction_count=0,
            memory_facts=[],
        )
        assert profile.individual_id == "user-1"
        assert profile.name == "Alice"
        assert profile.relationship_strength == 0.0
        assert profile.interaction_count == 0

    def test_relationship_strength_grows_with_interactions(self, profiler: AudienceProfiler) -> None:
        profile_new = profiler.build_profile(None, None, 0, [])
        profile_some = profiler.build_profile(None, None, 10, [])
        profile_established = profiler.build_profile(None, None, 100, [])
        assert profile_new.relationship_strength < profile_some.relationship_strength
        assert profile_some.relationship_strength < profile_established.relationship_strength

    def test_relationship_strength_plateaus_below_1(self, profiler: AudienceProfiler) -> None:
        profile = profiler.build_profile(None, None, 10000, [])
        assert profile.relationship_strength < 1.0

    def test_technical_level_extracted_from_memory_facts(self, profiler: AudienceProfiler) -> None:
        facts = [{"type": "technical_level", "value": 0.9}]
        profile = profiler.build_profile(None, None, 5, facts)
        assert profile.technical_level == pytest.approx(0.9)

    def test_comm_preferences_extracted_from_memory_facts(self, profiler: AudienceProfiler) -> None:
        facts = [{"type": "prefers_bullet_points", "value": True}]
        profile = profiler.build_profile(None, None, 5, facts)
        assert profile.communication_preferences.get("prefers_bullet_points") is True


class TestStrategyAdaptation:
    def test_non_technical_audience_sets_thorough_explanation(self, profiler: AudienceProfiler) -> None:
        profile = profiler.build_profile(None, None, 5, [{"type": "technical_level", "value": 0.1}])
        strategy = profiler.adapt(StrategyParams(), profile)
        assert strategy.explanation_depth == "thorough"
        assert strategy.jargon_level == "none"

    def test_technical_audience_sets_concise_explanation(self, profiler: AudienceProfiler) -> None:
        profile = profiler.build_profile(None, None, 5, [{"type": "technical_level", "value": 0.9}])
        strategy = profiler.adapt(StrategyParams(), profile)
        assert strategy.explanation_depth == "concise"
        assert strategy.assume_knowledge is True

    def test_distressed_audience_enables_empathy_first(self, profiler: AudienceProfiler) -> None:
        profile = profiler.build_profile(None, None, 5, [{"type": "emotional_distress", "value": 0.8}])
        strategy = profiler.adapt(StrategyParams(), profile)
        assert strategy.empathy_first is True
        assert strategy.information_density == "low"

    def test_frustrated_audience_reduces_target_length(self, profiler: AudienceProfiler) -> None:
        base = StrategyParams(target_length=300)
        profile = profiler.build_profile(None, None, 5, [{"type": "emotional_frustration", "value": 0.8}])
        strategy = profiler.adapt(base, profile)
        assert strategy.target_length < 300

    def test_group_audience_sets_collective_address(self, profiler: AudienceProfiler) -> None:
        profile = profiler.build_profile(None, None, 0, [], audience_type="group", group_size=10)
        strategy = profiler.adapt(StrategyParams(), profile)
        assert strategy.address_style == "collective"
        assert strategy.avoid_singling_out is True

    def test_first_interaction_introduces_self(self, profiler: AudienceProfiler) -> None:
        profile = profiler.build_profile(None, None, 0, [])
        strategy = profiler.adapt(StrategyParams(), profile)
        assert strategy.introduce_self_if_first is True

    def test_prefers_brief_reduces_target_length(self, profiler: AudienceProfiler) -> None:
        profile = profiler.build_profile(None, None, 5, [{"type": "prefers_brief", "value": True}])
        strategy = profiler.adapt(StrategyParams(target_length=500), profile)
        assert strategy.target_length <= 200
