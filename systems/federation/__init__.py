"""
EcodiaOS — Federation System

The Federation Protocol governs how EOS instances relate to each other —
as sovereign entities that can choose to share knowledge, coordinate action,
and build relationships.
"""

from systems.federation.exchange import ExchangeProtocol
from systems.federation.handshake import (
    HandshakeConfirmation,
    HandshakeProcessor,
    HandshakeRequest,
    HandshakeResponse,
    HandshakeResult,
)
from systems.federation.ingestion import IngestionPipeline
from systems.federation.service import FederationService

__all__ = [
    "ExchangeProtocol",
    "FederationService",
    "HandshakeConfirmation",
    "HandshakeProcessor",
    "HandshakeRequest",
    "HandshakeResponse",
    "HandshakeResult",
    "IngestionPipeline",
]
