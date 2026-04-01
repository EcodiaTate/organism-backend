"""
EcodiaOS - Kairos: Stage 4 Mechanism Extractor

Extracts the causal pathway between a confirmed cause–effect pair by
traversing the CausalNode graph stored in Memory.  The result is a
natural-language description of the intermediate steps (mechanism) that
is written into CausalRule.mechanism before Stage 5 context-invariance
testing begins.

Design principles
─────────────────
• Memory-only: queries go through MemoryService, no cross-system imports.
• Non-blocking: failures return an empty mechanism; the pipeline continues.
• Minimal: finds shortest path in the CausalNode graph up to max_depth hops.
  If no path is found it synthesises a direct-effect placeholder so
  downstream stages always receive a non-empty mechanism string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.kairos.types import CausalDirectionResult, CausalRule

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger("kairos.mechanism_extractor")

# Maximum number of intermediate hops to follow in the CausalNode graph.
_MAX_DEPTH = 4
# Maximum branching factor per hop (limits explosion for dense graphs).
_MAX_BRANCH = 8


class MechanismExtractor:
    """
    Stage 4 of the Kairos pipeline.

    For each (cause, effect) pair that passed confounder analysis, this
    stage queries Memory's CausalNode graph to find intermediate steps
    and populates CausalRule.mechanism with a human-readable pathway.
    """

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory

    def set_memory(self, memory: MemoryService) -> None:
        self._memory = memory

    async def extract(
        self,
        direction_result: CausalDirectionResult,
    ) -> str:
        """
        Extract the causal mechanism for the confirmed direction_result pair.

        Returns a natural-language mechanism string.  Never raises - returns
        a minimal placeholder on any error so Stage 5 can proceed.
        """
        cause = direction_result.cause
        effect = direction_result.effect
        if not cause or not effect:
            return f"{cause or '?'} directly influences {effect or '?'}"

        if self._memory is None:
            return self._direct_mechanism(cause, effect)

        try:
            path = await self._find_causal_path(cause, effect)
            if path:
                return self._path_to_mechanism(path)
        except Exception:
            logger.exception(
                "mechanism_extraction_failed",
                cause=cause,
                effect=effect,
            )

        return self._direct_mechanism(cause, effect)

    # ─── Internal helpers ──────────────────────────────────────────────

    async def _find_causal_path(
        self, cause: str, effect: str
    ) -> list[dict[str, Any]]:
        """
        BFS over the CausalNode graph in Memory to find the shortest path
        from `cause` to `effect` up to _MAX_DEPTH hops.

        Returns a list of node dicts (each with at least a 'name' key),
        or an empty list if no path is found.
        """
        # Each frontier entry: (current_node_name, path_so_far)
        frontier: list[tuple[str, list[dict[str, Any]]]] = [(cause, [])]
        visited: set[str] = {cause}

        for _ in range(_MAX_DEPTH):
            if not frontier:
                break
            next_frontier: list[tuple[str, list[dict[str, Any]]]] = []
            for current_name, path in frontier:
                neighbours = await self._get_causal_neighbours(current_name)
                for neighbour in neighbours[:_MAX_BRANCH]:
                    n_name = neighbour.get("name", "")
                    if not n_name or n_name in visited:
                        continue
                    new_path = path + [neighbour]
                    if n_name == effect:
                        # Prepend origin node for a complete path
                        origin = {"name": cause, "domain": neighbour.get("domain", "")}
                        return [origin] + new_path
                    visited.add(n_name)
                    next_frontier.append((n_name, new_path))
            frontier = next_frontier

        return []

    async def _get_causal_neighbours(
        self, node_name: str
    ) -> list[dict[str, Any]]:
        """
        Fetch direct CAUSES-successors of the named CausalNode from Memory.
        Returns an empty list when Memory is unavailable or the node doesn't exist.
        """
        if self._memory is None:
            return []
        try:
            # MemoryService exposes Neo4j via its _neo4j client.
            # We query through the service's internal client rather than
            # importing Neo4jClient directly (no cross-system import needed
            # because _memory IS the MemoryService instance).
            rows = await self._memory._neo4j.execute_read(  # type: ignore[attr-defined]
                """
                MATCH (c:CausalNode {name: $name})-[:CAUSES]->(e:CausalNode)
                RETURN e.name AS name, e.domain AS domain,
                       e.confidence AS confidence
                LIMIT $limit
                """,
                {"name": node_name, "limit": _MAX_BRANCH},
            )
            return [dict(r) for r in rows] if rows else []
        except Exception:
            logger.debug("causal_neighbours_query_failed", node_name=node_name)
            return []

    @staticmethod
    def _path_to_mechanism(path: list[dict[str, Any]]) -> str:
        """Convert a list of CausalNode dicts into a readable mechanism string."""
        steps = [n.get("name", "?") for n in path]
        if len(steps) == 2:
            return f"{steps[0]} directly causes {steps[-1]}"
        intermediate = " → ".join(steps[1:-1])
        return f"{steps[0]} causes {steps[-1]} via {intermediate}"

    @staticmethod
    def _direct_mechanism(cause: str, effect: str) -> str:
        return f"{cause} directly influences {effect} (no intermediate path found)"
