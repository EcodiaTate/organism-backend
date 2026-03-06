"""
EcodiaOS -- Simula Retrieval Subsystem (Stage 3B)

SWE-grep agentic retrieval: multi-hop code search that replaces
embedding-based find_similar with deterministic tool-based search.
"""

from systems.simula.retrieval.swe_grep import SweGrepRetriever

__all__ = [
    "SweGrepRetriever",
]
