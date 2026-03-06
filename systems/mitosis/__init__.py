"""
EcodiaOS — Mitosis Infrastructure (Phase 16e: Speciation)

Container orchestration layer for child instance spawning.
The economic logic (fitness, niche selection, seed config) lives in Oikos;
this module handles the physical act of booting child containers.
"""

from systems.mitosis.spawner import LocalDockerSpawner, SpawnResult

__all__ = [
    "LocalDockerSpawner",
    "SpawnResult",
]
