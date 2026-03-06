"""
EcodiaOS — Alive: Visualization & Embodiment

WebSocket server that bridges Synapse telemetry, Atune affect state, and
aggregated real-time system state to the browser-based Three.js visualization.

Public API:
  AliveWebSocketServer — standalone WS server on port 8001
  BenchmarkProvider    — protocol for plugging in the future benchmarks system
"""

from systems.alive.ws_server import AliveWebSocketServer, BenchmarkProvider

__all__ = [
    "AliveWebSocketServer",
    "BenchmarkProvider",
]
