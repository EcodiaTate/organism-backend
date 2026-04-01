"""
Unit tests for AliveWebSocketServer (Spec 11a).

Covers:
  - Connection lifecycle (connect → initial state → disconnect)
  - Auth rejection (4401) when tokens configured, token missing or wrong
  - Auth pass-through when no tokens configured
  - Affect payload serialization - all fields present, dominance sourced correctly
  - System state snapshot serialization - all 13 sections present
  - Timeout protection - hung subsystem returns {"available": False, "error": "timeout"}
  - RE training stream write - xadd called per poll cycle
  - Health report reflects wired systems
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from systems.alive.ws_server import AliveWebSocketServer, _AUTH_TOKEN_PARAM, _GATHER_TIMEOUT
from systems.soma.types import InteroceptiveDimension


# ─── Fixtures ──────────────────────────────────────────────────────────────


def _make_soma_state(
    valence: float = 0.1,
    arousal: float = 0.4,
    social_charge: float = 0.3,
    coherence: float = 0.7,
    curiosity: float = 0.5,
    energy: float = 0.6,
    confidence: float = 0.65,
    integrity: float = 0.9,
    temporal_pressure: float = 0.2,
    urgency: float = 0.15,
) -> MagicMock:
    """Build a minimal SomaCurrentState mock."""
    state = MagicMock()
    state.sensed = {
        InteroceptiveDimension.VALENCE: valence,
        InteroceptiveDimension.AROUSAL: arousal,
        InteroceptiveDimension.SOCIAL_CHARGE: social_charge,
        InteroceptiveDimension.COHERENCE: coherence,
        InteroceptiveDimension.CURIOSITY_DRIVE: curiosity,
        InteroceptiveDimension.ENERGY: energy,
        InteroceptiveDimension.CONFIDENCE: confidence,
        InteroceptiveDimension.INTEGRITY: integrity,
        InteroceptiveDimension.TEMPORAL_PRESSURE: temporal_pressure,
    }
    state.urgency = urgency
    state.dominant_error = InteroceptiveDimension.ENERGY
    state.timestamp = None
    return state


def _make_soma(soma_state: MagicMock | None = None) -> MagicMock:
    soma = MagicMock()
    soma.get_current_state.return_value = soma_state or _make_soma_state()
    return soma


def _make_redis() -> MagicMock:
    redis = MagicMock()
    redis._config.prefix = "eos_test"
    redis.client = MagicMock()
    redis.client.xadd = AsyncMock(return_value=b"1-0")
    return redis


def _make_server(**kwargs: Any) -> AliveWebSocketServer:
    """Construct an AliveWebSocketServer with minimal dependencies."""
    defaults: dict[str, Any] = {
        "redis": _make_redis(),
        "soma": _make_soma(),
    }
    defaults.update(kwargs)
    return AliveWebSocketServer(**defaults)


# ─── Affect Payload ─────────────────────────────────────────────────────────


class TestBuildAffectPayload:
    def test_all_required_fields_present(self) -> None:
        server = _make_server()
        payload = server._build_affect_payload()

        required = {
            "valence", "arousal", "dominance", "curiosity",
            "care_activation", "coherence_stress",
            "energy", "confidence", "integrity", "temporal_pressure",
            "urgency", "dominant_error",
        }
        assert required.issubset(payload.keys()), (
            f"Missing fields: {required - payload.keys()}"
        )

    def test_dominance_from_atune_when_wired(self) -> None:
        atune = MagicMock()
        atune.current_affect.dominance = 0.77

        server = _make_server(atune=atune)
        payload = server._build_affect_payload()

        assert payload["dominance"] == 0.77

    def test_dominance_fallback_to_social_charge_when_atune_absent(self) -> None:
        soma_state = _make_soma_state(social_charge=0.42)
        server = _make_server(soma=_make_soma(soma_state))
        payload = server._build_affect_payload()

        # Without Atune, dominance should approximate SOCIAL_CHARGE
        assert payload["dominance"] == pytest.approx(0.42, abs=1e-4)

    def test_dominance_fallback_when_atune_raises(self) -> None:
        atune = MagicMock()
        atune.current_affect = MagicMock(side_effect=RuntimeError("atune down"))

        soma_state = _make_soma_state(social_charge=0.35)
        server = _make_server(atune=atune, soma=_make_soma(soma_state))
        payload = server._build_affect_payload()

        assert payload["dominance"] == pytest.approx(0.35, abs=1e-4)

    def test_soma_unavailable_returns_error_dict(self) -> None:
        server = _make_server(soma=None)
        payload = server._build_affect_payload()

        assert payload == {"available": False}

    def test_soma_raises_returns_error_dict(self) -> None:
        soma = MagicMock()
        soma.get_current_state.side_effect = RuntimeError("soma crashed")
        server = _make_server(soma=soma)
        payload = server._build_affect_payload()

        assert "available" in payload
        assert payload["available"] is False
        assert "error" in payload

    def test_coherence_stress_inverted(self) -> None:
        soma_state = _make_soma_state(coherence=0.6)
        server = _make_server(soma=_make_soma(soma_state))
        payload = server._build_affect_payload()

        assert payload["coherence_stress"] == pytest.approx(1.0 - 0.6, abs=1e-4)


# ─── System State Snapshot ──────────────────────────────────────────────────


class TestBuildSystemStatePayload:
    @pytest.mark.asyncio
    async def test_all_13_sections_present(self) -> None:
        server = _make_server()
        payload = await server._build_system_state_payload()

        expected_sections = {
            "cycle", "drives", "interoceptive", "attention",
            "immune", "goals", "actions", "economics",
            "mutations", "benchmarks", "causal", "compression", "sleep",
        }
        assert expected_sections == set(payload.keys())

    @pytest.mark.asyncio
    async def test_unavailable_subsystems_return_available_false(self) -> None:
        # No optional subsystems wired → all should return {"available": False}
        server = _make_server()
        payload = await server._build_system_state_payload()

        for section in ("cycle", "drives", "attention", "mutations", "benchmarks", "sleep"):
            assert payload[section].get("available") is False, (
                f"Section '{section}' should be unavailable but got {payload[section]}"
            )

    @pytest.mark.asyncio
    async def test_snapshot_is_json_serializable(self) -> None:
        server = _make_server()
        payload = await server._build_system_state_payload()

        # Must not raise
        serialized = orjson.dumps(payload)
        assert len(serialized) > 0


# ─── Timeout Protection ─────────────────────────────────────────────────────


class TestGatherTimeouts:
    @pytest.mark.asyncio
    async def test_immune_timeout_returns_error_payload(self) -> None:
        thymos = MagicMock()

        async def _slow_health() -> dict[str, Any]:
            await asyncio.sleep(10)  # will be cancelled by wait_for
            return {}

        thymos.health = _slow_health
        thymos.drive_state = MagicMock(
            equor_rejections=0,
            rejections_by_drive={},
        )

        server = _make_server(thymos=thymos)

        result = await asyncio.wait_for(
            server._gather_immune_safe(),
            timeout=_GATHER_TIMEOUT + 0.5,
        )
        assert result == {"available": False, "error": "timeout"}

    @pytest.mark.asyncio
    async def test_goals_timeout_returns_error_payload(self) -> None:
        nova = MagicMock()

        async def _slow_health() -> dict[str, Any]:
            await asyncio.sleep(10)
            return {}

        nova.health = _slow_health
        nova.active_goal_summaries = []

        server = _make_server(nova=nova)
        result = await asyncio.wait_for(
            server._gather_goals_safe(),
            timeout=_GATHER_TIMEOUT + 0.5,
        )
        assert result == {"available": False, "error": "timeout"}

    @pytest.mark.asyncio
    async def test_economics_timeout_returns_error_payload(self) -> None:
        oikos = MagicMock()

        async def _slow_health() -> dict[str, Any]:
            await asyncio.sleep(10)
            return {}

        oikos.health = _slow_health

        server = _make_server(oikos=oikos)
        result = await asyncio.wait_for(
            server._gather_economics_safe(),
            timeout=_GATHER_TIMEOUT + 0.5,
        )
        assert result == {"available": False, "error": "timeout"}

    @pytest.mark.asyncio
    async def test_timeout_in_one_section_does_not_block_others(self) -> None:
        """A hung immune gather must not prevent economics from responding."""
        thymos = MagicMock()

        async def _slow_health() -> dict[str, Any]:
            await asyncio.sleep(10)
            return {}

        thymos.health = _slow_health
        thymos.drive_state = MagicMock(equor_rejections=0, rejections_by_drive={})

        server = _make_server(thymos=thymos)

        # Run immune and economics gathers concurrently
        immune_result, economics_result = await asyncio.gather(
            server._gather_immune_safe(),
            server._gather_economics_safe(),
        )

        assert immune_result == {"available": False, "error": "timeout"}
        # economics has no oikos → available=False but NOT a timeout
        assert economics_result.get("error") != "timeout"


# ─── RE Training Stream Write ───────────────────────────────────────────────


class TestReTrainingStream:
    @pytest.mark.asyncio
    async def test_xadd_called_with_correct_stream_key(self) -> None:
        redis = _make_redis()
        server = _make_server(redis=redis)

        payload: dict[str, Any] = {"cycle": {"available": False}}
        await server._write_re_training_snapshot(payload)

        redis.client.xadd.assert_called_once()
        call_args = redis.client.xadd.call_args
        stream_key: str = call_args[0][0]
        assert stream_key == "eos_test:stream:alive_snapshots"

    @pytest.mark.asyncio
    async def test_xadd_snapshot_field_is_bytes(self) -> None:
        redis = _make_redis()
        server = _make_server(redis=redis)

        payload: dict[str, Any] = {"test": True}
        await server._write_re_training_snapshot(payload)

        fields: dict[str, Any] = redis.client.xadd.call_args[0][1]
        assert "snapshot" in fields
        # orjson.dumps → bytes
        assert isinstance(fields["snapshot"], bytes)

    @pytest.mark.asyncio
    async def test_xadd_failure_is_non_fatal(self) -> None:
        redis = _make_redis()
        redis.client.xadd = AsyncMock(side_effect=ConnectionError("redis down"))
        server = _make_server(redis=redis)

        # Must not raise
        await server._write_re_training_snapshot({"test": True})


# ─── WebSocket Authentication ───────────────────────────────────────────────


class TestWebSocketAuth:
    def _make_ws_mock(self, path: str = "/") -> MagicMock:
        ws = MagicMock()
        ws.remote_address = ("127.0.0.1", 9999)
        ws.request = MagicMock()
        ws.request.path = path
        ws.close = AsyncMock()
        ws.send = AsyncMock()
        # Simulate client that disconnects immediately after initial state
        ws.__aiter__ = MagicMock(return_value=iter([]))
        return ws

    @pytest.mark.asyncio
    async def test_no_auth_tokens_accepts_all_connections(self) -> None:
        server = _make_server()  # no auth_tokens
        assert not server._auth_tokens

        ws = self._make_ws_mock("/")
        # Handler calls _send_initial_state which calls _build_system_state_payload
        with patch.object(server, "_send_initial_state", new_callable=AsyncMock):
            await server._handler(ws)

        ws.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_token_accepted(self) -> None:
        token = "secret-token-abc"
        server = _make_server(auth_tokens={token})

        ws = self._make_ws_mock(f"/?{_AUTH_TOKEN_PARAM}={token}")
        with patch.object(server, "_send_initial_state", new_callable=AsyncMock):
            await server._handler(ws)

        ws.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_token_rejected_4401(self) -> None:
        server = _make_server(auth_tokens={"secret"})

        ws = self._make_ws_mock("/")  # no token query param
        await server._handler(ws)

        ws.close.assert_called_once_with(4401, "Unauthorized")

    @pytest.mark.asyncio
    async def test_wrong_token_rejected_4401(self) -> None:
        server = _make_server(auth_tokens={"correct-token"})

        ws = self._make_ws_mock(f"/?{_AUTH_TOKEN_PARAM}=wrong-token")
        await server._handler(ws)

        ws.close.assert_called_once_with(4401, "Unauthorized")

    @pytest.mark.asyncio
    async def test_client_added_to_set_on_connect(self) -> None:
        server = _make_server()
        assert len(server._clients) == 0

        ws = self._make_ws_mock("/")
        with patch.object(server, "_send_initial_state", new_callable=AsyncMock):
            await server._handler(ws)

        # After handler exits normally, client is removed
        assert ws not in server._clients

    @pytest.mark.asyncio
    async def test_rejected_client_not_added_to_set(self) -> None:
        server = _make_server(auth_tokens={"tok"})
        ws = self._make_ws_mock("/?token=bad")
        await server._handler(ws)

        assert ws not in server._clients


# ─── Health Report ──────────────────────────────────────────────────────────


class TestHealthReport:
    @pytest.mark.asyncio
    async def test_health_includes_auth_enabled_flag(self) -> None:
        server_open = _make_server()
        h_open = await server_open.health()
        assert h_open["auth_enabled"] is False

        server_auth = _make_server(auth_tokens={"tok"})
        h_auth = await server_auth.health()
        assert h_auth["auth_enabled"] is True

    @pytest.mark.asyncio
    async def test_health_wired_systems_atune_field(self) -> None:
        server = _make_server(atune=MagicMock())
        h = await server.health()
        assert h["systems_wired"]["atune"] is True

    @pytest.mark.asyncio
    async def test_health_wired_systems_atune_absent(self) -> None:
        server = _make_server()
        h = await server.health()
        assert h["systems_wired"]["atune"] is False

    @pytest.mark.asyncio
    async def test_health_status_running(self) -> None:
        server = _make_server()
        server._running = True
        h = await server.health()
        assert h["status"] == "running"

    @pytest.mark.asyncio
    async def test_health_status_stopped(self) -> None:
        server = _make_server()
        server._running = False
        h = await server.health()
        assert h["status"] == "stopped"
