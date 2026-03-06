"""
EcodiaOS — Simula Proposals

External proposal ingestion and translation into Simula's native
EvolutionProposal format.
"""

from systems.simula.proposals.adversarial_ingestion import (
    AdversarialProposalIngester,
    IngestionResult,
)

__all__ = [
    "AdversarialProposalIngester",
    "IngestionResult",
]
