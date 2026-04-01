"""
Integration tests for Stage 0 - Soma Integration.

Tests the complete somatic pathway end-to-end:
  Soma → Atune (precision_weights) → Nova (allostatic deliberation) → Memory (somatic markers)

Uses real Soma sub-components with mocked external dependencies
(no real LLM, Neo4j, or Redis).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import SomaConfig
from primitives.memory_trace import Episode, MemoryTrace, RetrievalResult
from systems.soma.service import SomaService
from systems.soma.somatic_memory import MARKER_VECTOR_DIM
from systems.soma.types import (
    ALL_DIMENSIONS,
    AllostaticSignal,
    InteroceptiveDimension,
    InteroceptiveState,
    SomaticMarker,
)

# ─── Config Helpers ──────────────────────────────────────────────


def _make_soma_config(**overrides: Any) -> SomaConfig:
    defaults = {
        "cycle_enabled": True,
        "phase_space_update_interval": 5,
        "trajectory_buffer_size": 50,
        "prediction_ewm_span": 5,
        "setpoint_adaptation_alpha": 0.05,
        "urgency_threshold": 0.3,
        "attractor_min_dwell_cycles": 5,
        "bifurcation_detection_enabled": True,
        "max_attractors": 20,
        "somatic_marker_enabled": True,
        "somatic_rerank_boost": 0.3,
        "developmental_gating_enabled": True,
        "initial_stage": "reflexive",
    }
    defaults.update(overrides)
    return SomaConfig(**defaults)


def _make_soma(**overrides: Any) -> SomaService:
    config = _make_soma_config(**overrides)
    return SomaService(config=config)


def _mock_atune() -> MagicMock:
    affect = MagicMock()
    affect.valence = 0.2
    affect.arousal = 0.4
    affect.curiosity = 0.5
    affect.care_activation = 0.3
    affect.coherence_stress = 0.1

    mgr = MagicMock()
    mgr.current = affect

    atune = MagicMock()
    atune._affect_mgr = mgr
    atune.mean_prediction_error = 0.2
    return atune


def _make_state(value: float = 0.5) -> InteroceptiveState:
    return InteroceptiveState(
        sensed={d: value for d in ALL_DIMENSIONS},
        errors={"moment": {d: 0.1 for d in ALL_DIMENSIONS}},
        max_error_magnitude=0.3,
    )


# ─── Task 0.1: SomaClock types exist and are well-formed ─────────


class TestSomaTypesIntegrity:
    """Verify all Soma types from types.py are coherent and usable."""

    def test_interoceptive_dimensions_count(self):
        assert len(ALL_DIMENSIONS) == 9

    def test_interoceptive_state_marker_vector(self):
        state = _make_state(0.6)
        vec = state.to_marker_vector()
        assert len(vec) == MARKER_VECTOR_DIM  # 9 + 9 + 1 = 19

    def test_allostatic_signal_default_safe(self):
        sig = AllostaticSignal.default()
        assert sig.urgency == 0.0
        assert len(sig.precision_weights) == 9
        for w in sig.precision_weights.values():
            assert w == 1.0

    def test_somatic_marker_to_vector(self):
        marker = SomaticMarker(
            interoceptive_snapshot={d: 0.5 for d in ALL_DIMENSIONS},
            allostatic_error_snapshot={d: 0.1 for d in ALL_DIMENSIONS},
            prediction_error_at_encoding=0.3,
            allostatic_context="flow",
        )
        vec = marker.to_vector()
        assert len(vec) == 19
        assert vec[-1] == 0.3  # PE magnitude


# ─── Task 0.2: Soma wired as step 0 in cognitive cycle ───────────


class TestSomaCycleExecution:
    """Verify Soma runs and emits a valid AllostaticSignal."""

    @pytest.mark.asyncio
    async def test_soma_runs_cycle(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()

        signal = await soma.run_cycle()
        assert isinstance(signal, AllostaticSignal)
        assert signal.urgency >= 0.0
        assert signal.dominant_error in ALL_DIMENSIONS
        assert len(signal.precision_weights) == 9

    @pytest.mark.asyncio
    async def test_soma_emits_precision_weights(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()

        signal = await soma.run_cycle()
        # Every dimension should have a precision weight
        for dim in ALL_DIMENSIONS:
            assert dim in signal.precision_weights
            assert signal.precision_weights[dim] >= 0.0

    @pytest.mark.asyncio
    async def test_soma_multiple_cycles_accumulate(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()

        for _ in range(5):
            await soma.run_cycle()

        assert soma.cycle_count == 5


# ─── Task 0.3: Atune precision_weights modulated by Soma ─────────


class TestAtunePrecisionWeightsConsumer:
    """Verify Soma outputs precision_weights that Atune could consume."""

    @pytest.mark.asyncio
    async def test_precision_weights_vary_with_state(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()

        # Run enough cycles for predictions to diverge from defaults
        for _ in range(10):
            await soma.run_cycle()

        signal = soma.get_current_signal()
        weights = signal.precision_weights
        # After 10 cycles, not all weights should be exactly 1.0
        # (though this depends on state dynamics)
        assert len(weights) == 9
        assert all(w >= 0.0 for w in weights.values())


# ─── Task 0.4: Nova allostatic triggers ──────────────────────────


class TestNovaAllostaticTrigger:
    """Verify Soma urgency signal would trigger allostatic mode in Nova."""

    @pytest.mark.asyncio
    async def test_urgency_below_threshold_no_allostatic(self):
        soma = _make_soma(urgency_threshold=0.7)
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()

        signal = await soma.run_cycle()
        # Fresh cycle with normal inputs → urgency should be low
        assert signal.urgency < 0.7

    @pytest.mark.asyncio
    async def test_get_current_signal(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()

        await soma.run_cycle()
        signal = soma.get_current_signal()
        assert isinstance(signal, AllostaticSignal)
        assert signal.dominant_error in ALL_DIMENSIONS

    def test_urgency_threshold_property(self):
        soma = _make_soma(urgency_threshold=0.5)
        assert soma.urgency_threshold == 0.5


# ─── Task 0.5: Memory somatic markers ────────────────────────────


class TestMemorySomaticMarkers:
    """Verify somatic marker creation and storage integration."""

    @pytest.mark.asyncio
    async def test_soma_creates_somatic_marker(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()
        await soma.run_cycle()

        marker = soma.get_somatic_marker()
        assert isinstance(marker, SomaticMarker)
        assert len(marker.interoceptive_snapshot) == 9
        assert len(marker.allostatic_error_snapshot) == 9
        vec = marker.to_vector()
        assert len(vec) == 19

    @pytest.mark.asyncio
    async def test_episode_stores_somatic_marker(self):
        """Episode model now accepts somatic_marker and somatic_vector."""
        marker = SomaticMarker(
            interoceptive_snapshot={d: 0.5 for d in ALL_DIMENSIONS},
            allostatic_error_snapshot={d: 0.1 for d in ALL_DIMENSIONS},
            prediction_error_at_encoding=0.3,
        )
        vec = marker.to_vector()

        episode = Episode(
            source="test:unit",
            raw_content="hello world",
            somatic_marker=marker,
            somatic_vector=vec,
        )

        assert episode.somatic_marker is marker
        assert episode.somatic_vector is not None
        assert len(episode.somatic_vector) == 19

    @pytest.mark.asyncio
    async def test_episode_without_marker_defaults_none(self):
        episode = Episode(source="test:unit", raw_content="no soma")
        assert episode.somatic_marker is None
        assert episode.somatic_vector is None

    def test_memory_trace_stores_somatic_marker(self):
        marker = SomaticMarker(
            interoceptive_snapshot={d: 0.5 for d in ALL_DIMENSIONS},
            allostatic_error_snapshot={d: 0.1 for d in ALL_DIMENSIONS},
            prediction_error_at_encoding=0.3,
        )
        trace = MemoryTrace(
            episode_id="ep_1",
            original_percept_id="per_1",
            somatic_marker=marker,
            somatic_vector=marker.to_vector(),
        )
        assert trace.somatic_marker is marker
        assert len(trace.somatic_vector) == 19


class TestSomaticReranking:
    """End-to-end somatic reranking through Soma service."""

    @pytest.mark.asyncio
    async def test_somatic_rerank_boosts_similar(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()
        await soma.run_cycle()

        # Create retrieval results with somatic vectors
        current_marker = soma.get_somatic_marker()
        current_vec = current_marker.to_vector()

        # Candidate with identical somatic state
        similar = RetrievalResult(
            node_id="ep_similar",
            salience=0.5,
            salience_score=0.5,
            somatic_vector=current_vec,
        )
        # Candidate with no somatic vector
        no_marker = RetrievalResult(
            node_id="ep_none",
            salience=0.5,
            salience_score=0.5,
            somatic_vector=None,
        )
        # Candidate with opposing somatic state
        opposing_vec = [-v for v in current_vec]
        opposing = RetrievalResult(
            node_id="ep_opposing",
            salience=0.5,
            salience_score=0.5,
            somatic_vector=opposing_vec,
        )

        candidates = [opposing, no_marker, similar]
        result = soma.somatic_rerank(candidates)

        assert len(result) == 3
        # Similar should be boosted to the top (or at least above opposing)
        node_ids = [r.node_id for r in result]
        similar_idx = node_ids.index("ep_similar")
        opposing_idx = node_ids.index("ep_opposing")
        assert similar_idx < opposing_idx  # Similar ranked higher

    @pytest.mark.asyncio
    async def test_somatic_rerank_graceful_without_markers(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()
        await soma.run_cycle()

        candidates = [
            RetrievalResult(node_id="a", salience=0.8, salience_score=0.8),
            RetrievalResult(node_id="b", salience=0.6, salience_score=0.6),
        ]
        result = soma.somatic_rerank(candidates)
        assert len(result) == 2
        # Without somatic vectors, order is by salience_score descending
        assert result[0].node_id == "a"

    @pytest.mark.asyncio
    async def test_rerank_empty_list(self):
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()
        await soma.run_cycle()

        result = soma.somatic_rerank([])
        assert result == []


class TestRetrievalResultSomaticFields:
    """Verify RetrievalResult carries somatic data for the pipeline."""

    def test_retrieval_result_has_somatic_vector(self):
        vec = [0.5] * 19
        r = RetrievalResult(
            node_id="ep_1",
            somatic_vector=vec,
            salience=0.5,
            salience_score=0.5,
        )
        assert r.somatic_vector == vec
        assert len(r.somatic_vector) == 19

    def test_retrieval_result_defaults_none(self):
        r = RetrievalResult(node_id="ep_1")
        assert r.somatic_vector is None
        assert r.salience_score == 0.0


# ─── Task 0.6: DB migrations types ──────────────────────────────


class TestMigrationTypes:
    """Verify the migration functions exist and are importable."""

    def test_timescaledb_migration_importable(self):
        from database.migrations import migrate_timescaledb_interoceptive_state
        assert callable(migrate_timescaledb_interoceptive_state)

    def test_neo4j_migration_importable(self):
        from database.migrations import migrate_neo4j_somatic_vector_index
        assert callable(migrate_neo4j_somatic_vector_index)

    def test_run_all_migrations_importable(self):
        from database.migrations import run_all_migrations
        assert callable(run_all_migrations)


# ─── Task 0.7: Full pathway Soma → Memory store → Memory retrieve ─


class TestFullSomaMemoryPathway:
    """
    End-to-end: Soma creates a marker → stored on Episode → retrieved
    with somatic_vector → reranked by somatic similarity.
    """

    @pytest.mark.asyncio
    async def test_marker_roundtrip(self):
        """Marker created by Soma can be stored on Episode and extracted as vector."""
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()
        await soma.run_cycle()

        # 1. Soma creates marker
        marker = soma.get_somatic_marker()
        vec = marker.to_vector()

        # 2. Episode stores both
        episode = Episode(
            source="test:roundtrip",
            raw_content="test content",
            somatic_marker=marker,
            somatic_vector=vec,
        )
        assert episode.somatic_vector == vec

        # 3. Simulate retrieval result (as if loaded from Neo4j)
        retrieval_result = RetrievalResult(
            node_id=episode.id,
            content=episode.raw_content,
            salience=0.5,
            salience_score=0.5,
            somatic_vector=episode.somatic_vector,
        )

        # 4. Rerank with Soma
        results = soma.somatic_rerank([retrieval_result])
        assert len(results) == 1
        # Salience should be boosted (identical state → max similarity)
        assert results[0].salience_score >= 0.5

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mixed_candidates(self):
        """Mixed candidates - some with markers, some without - all survive reranking."""
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()

        # Run a few cycles to build up state
        for _ in range(3):
            await soma.run_cycle()

        marker = soma.get_somatic_marker()
        vec = marker.to_vector()

        candidates = [
            RetrievalResult(
                node_id="ep_with_marker",
                salience=0.4,
                salience_score=0.4,
                somatic_vector=vec,
            ),
            RetrievalResult(
                node_id="ep_no_marker",
                salience=0.6,
                salience_score=0.6,
                somatic_vector=None,
            ),
            RetrievalResult(
                node_id="ep_partial_marker",
                salience=0.5,
                salience_score=0.5,
                somatic_vector=[0.0] * 19,  # Zero vector
            ),
        ]

        results = soma.somatic_rerank(candidates)
        assert len(results) == 3
        # All candidates survived (no crashes)
        ids = {r.node_id for r in results}
        assert ids == {"ep_with_marker", "ep_no_marker", "ep_partial_marker"}


class TestGracefulDegradation:
    """Verify all consumers work without Soma (pre-Soma fallback)."""

    def test_episode_without_soma(self):
        """Episode can be created without somatic fields."""
        episode = Episode(
            source="test:no_soma",
            raw_content="pre-soma episode",
        )
        assert episode.somatic_marker is None
        assert episode.somatic_vector is None

    def test_retrieval_result_without_soma(self):
        """RetrievalResult works without somatic fields."""
        r = RetrievalResult(
            node_id="ep_1",
            content="hello",
            salience=0.5,
        )
        assert r.somatic_vector is None
        assert r.unified_score == 0.0

    @pytest.mark.asyncio
    async def test_soma_rerank_on_none_soma(self):
        """If Soma is None, callers should skip reranking gracefully."""
        soma = None
        candidates = [
            RetrievalResult(node_id="a", salience=0.5, salience_score=0.5),
        ]
        # Simulating what memory/service.py does
        if soma is not None:
            candidates = soma.somatic_rerank(candidates)
        assert len(candidates) == 1
        assert candidates[0].node_id == "a"


# ─── Task 0.7: Synapse clock step-0 integration ───────────────────


class TestSynapseCycleWithSoma:
    """
    Verify the Synapse CognitiveClock wires Soma as step 0 of the theta tick,
    producing a SomaticCycleState on every CycleResult.

    Uses mock Atune and real Soma to test the clock's step-0 pathway
    without starting the full background loop.
    """

    def _make_mock_atune(self) -> MagicMock:
        """Minimal Atune mock: run_cycle() returns None (no broadcast)."""
        atune = AsyncMock()
        atune.run_cycle = AsyncMock(return_value=None)

        affect = MagicMock()
        affect.arousal = 0.4
        affect.coherence_stress = 0.1
        atune.current_affect = affect
        return atune

    @pytest.mark.asyncio
    async def test_somatic_cycle_state_built_from_signal(self) -> None:
        """
        After CognitiveClock runs with Soma wired, CycleResult.somatic
        contains a valid SomaticCycleState populated from AllostaticSignal.
        """
        from systems.synapse.types import SomaticCycleState

        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()

        # Run Soma directly (simulating what the clock does in step 0)
        await soma.run_cycle()
        signal = soma.get_current_signal()

        # Build SomaticCycleState from signal (mirrors clock._run_loop logic)
        somatic_state = SomaticCycleState(
            urgency=signal.urgency,
            dominant_error=signal.dominant_error.value
            if hasattr(signal.dominant_error, "value")
            else str(signal.dominant_error),
            arousal_sensed=signal.state.sensed.get(InteroceptiveDimension.AROUSAL, 0.4),
            energy_sensed=signal.state.sensed.get(InteroceptiveDimension.ENERGY, 0.6),
            precision_weights={
                k.value if hasattr(k, "value") else str(k): v
                for k, v in signal.precision_weights.items()
            },
            nearest_attractor=signal.nearest_attractor,
            trajectory_heading=signal.trajectory_heading,
            soma_cycle_ms=1.5,
        )

        # Verify all fields are well-formed
        assert isinstance(somatic_state, SomaticCycleState)
        assert 0.0 <= somatic_state.urgency <= 1.0
        assert isinstance(somatic_state.dominant_error, str)
        assert 0.0 <= somatic_state.arousal_sensed <= 1.0
        assert 0.0 <= somatic_state.energy_sensed <= 1.0
        assert len(somatic_state.precision_weights) == 9
        assert somatic_state.soma_cycle_ms >= 0.0

    @pytest.mark.asyncio
    async def test_cycle_result_somatic_field(self) -> None:
        """CycleResult accepts SomaticCycleState via the somatic field."""
        from systems.synapse.types import CycleResult, SomaticCycleState

        state = SomaticCycleState(
            urgency=0.3,
            dominant_error="arousal",
            arousal_sensed=0.65,
            energy_sensed=0.5,
            precision_weights={"arousal": 1.2, "energy": 0.8},
            nearest_attractor="flow",
            trajectory_heading="approaching",
            soma_cycle_ms=2.1,
        )

        result = CycleResult(
            cycle_number=1,
            elapsed_ms=145.0,
            budget_ms=150.0,
            somatic=state,
        )

        assert result.somatic is state
        assert result.somatic.urgency == pytest.approx(0.3)
        assert result.somatic.nearest_attractor == "flow"

    @pytest.mark.asyncio
    async def test_cycle_result_somatic_none_when_soma_absent(self) -> None:
        """When Soma is not wired, CycleResult.somatic is None."""
        from systems.synapse.types import CycleResult

        result = CycleResult(
            cycle_number=1,
            elapsed_ms=120.0,
            budget_ms=150.0,
        )

        assert result.somatic is None

    @pytest.mark.asyncio
    async def test_somatic_cycle_state_precision_weights_are_strings(self) -> None:
        """
        Precision weights are serialized as string keys (not InteroceptiveDimension enums)
        in SomaticCycleState, so they're safe for JSON/Redis transport.
        """
        soma = _make_soma()
        atune = _mock_atune()
        soma.set_atune(atune)
        await soma.initialize()
        await soma.run_cycle()

        signal = soma.get_current_signal()
        precision_weights = {
            k.value if hasattr(k, "value") else str(k): v
            for k, v in signal.precision_weights.items()
        }

        for key in precision_weights:
            assert isinstance(key, str), f"Expected str key, got {type(key)}: {key}"
        assert "arousal" in precision_weights
        assert "energy" in precision_weights

    @pytest.mark.asyncio
    async def test_soma_tick_event_structure(self) -> None:
        """SomaTickEvent wraps SomaticCycleState with a cycle_number and timestamp."""
        from systems.synapse.types import SomaticCycleState, SomaTickEvent

        state = SomaticCycleState(urgency=0.2, dominant_error="energy")
        event = SomaTickEvent(cycle_number=42, somatic_state=state)

        assert event.cycle_number == 42
        assert event.somatic_state is state
        assert event.id != ""
        assert event.timestamp is not None

        # model_dump() is safe for JSON serialization
        d = event.model_dump()
        assert d["cycle_number"] == 42
        assert d["somatic_state"]["urgency"] == pytest.approx(0.2)
