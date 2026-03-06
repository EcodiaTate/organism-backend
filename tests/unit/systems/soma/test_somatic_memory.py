"""Tests for Soma Somatic Memory — marker creation and embodied reranking."""

from __future__ import annotations

import pytest

from systems.soma.somatic_memory import (
    MARKER_VECTOR_DIM,
    SomaticMemoryIntegration,
    _cosine_similarity,
    _vector_norm,
)
from systems.soma.types import (
    ALL_DIMENSIONS,
    InteroceptiveState,
    SomaticMarker,
)


def _make_smi(**kwargs) -> SomaticMemoryIntegration:
    return SomaticMemoryIntegration(
        rerank_boost=kwargs.get("rerank_boost", 0.3),
    )


def _state(value: float = 0.5) -> InteroceptiveState:
    return InteroceptiveState(
        sensed={d: value for d in ALL_DIMENSIONS},
        errors={"moment": {d: 0.1 for d in ALL_DIMENSIONS}},
        max_error_magnitude=0.3,
    )


class TestMarkerCreation:
    def test_creates_marker(self):
        smi = _make_smi()
        state = _state(0.5)
        marker = smi.create_marker(state, attractor_label="flow")
        assert isinstance(marker, SomaticMarker)
        assert marker.allostatic_context == "flow"
        assert len(marker.interoceptive_snapshot) == 9
        assert len(marker.allostatic_error_snapshot) == 9

    def test_marker_vector_is_19d(self):
        smi = _make_smi()
        state = _state(0.5)
        marker = smi.create_marker(state)
        vec = marker.to_vector()
        assert len(vec) == MARKER_VECTOR_DIM


class TestCosineUtilities:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=0.001)

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_norm(self):
        assert _vector_norm([3.0, 4.0]) == pytest.approx(5.0)


class TestSomaticReranking:
    def test_no_candidates(self):
        smi = _make_smi()
        result = smi.somatic_rerank([], _state())
        assert result == []

    def test_candidates_without_markers_unchanged(self):
        smi = _make_smi()
        candidates = [{"salience_score": 0.5}, {"salience_score": 0.3}]
        result = smi.somatic_rerank(candidates, _state())
        assert len(result) == 2

    def test_similar_marker_boosted(self):
        smi = _make_smi(rerank_boost=0.3)
        current_state = _state(0.5)

        # Candidate with identical somatic state should get boosted
        marker = SomaticMarker(
            interoceptive_snapshot={d: 0.5 for d in ALL_DIMENSIONS},
            allostatic_error_snapshot={d: 0.1 for d in ALL_DIMENSIONS},
            prediction_error_at_encoding=0.3,
        )

        class MockTrace:
            salience_score = 0.5
            somatic_marker = marker

        candidates = [MockTrace()]
        result = smi.somatic_rerank(candidates, current_state)
        assert len(result) == 1
        # Should be boosted
        assert result[0].salience_score > 0.5

    def test_boost_capped_at_max(self):
        smi = _make_smi(rerank_boost=0.3)
        current_state = _state(0.5)

        marker = SomaticMarker(
            interoceptive_snapshot={d: 0.5 for d in ALL_DIMENSIONS},
            allostatic_error_snapshot={d: 0.1 for d in ALL_DIMENSIONS},
            prediction_error_at_encoding=0.3,
        )

        class MockTrace:
            salience_score = 1.0
            somatic_marker = marker

        candidates = [MockTrace()]
        result = smi.somatic_rerank(candidates, current_state)
        # Max boost = 1.0 * (1 + 0.3 * similarity) where similarity <= 1.0
        assert result[0].salience_score <= 1.3
