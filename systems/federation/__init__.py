"""
EcodiaOS - Federation System

The Federation Protocol governs how EOS instances relate to each other -
as sovereign entities that can choose to share knowledge, coordinate action,
and build relationships.

Work Pooling (Phase 2):
  TaskDelegationManager - delegate discrete tasks to trusted peers with USDC payment
  BountySplitter        - decompose large bounties into N sub-tasks for co-solving
  ResourceSharingManager - compute offloading when CPU > 85%
  YieldPoolManager      - pool capital for large yield positions (trust ≥ 0.9)
  FederationMarketplace - post/bid/rate tasks via Redis pub/sub
  WorkRouter            - Nexus-aware specialisation-based task routing
"""

from systems.federation.bounty_splitting import BountySplitter
from systems.federation.exchange import ExchangeProtocol
from systems.federation.handshake import (
    HandshakeConfirmation,
    HandshakeProcessor,
    HandshakeRequest,
    HandshakeResponse,
    HandshakeResult,
)
from systems.federation.ingestion import IngestionPipeline
from systems.federation.marketplace import FederationMarketplace
from systems.federation.resource_sharing import ResourceSharingManager
from systems.federation.service import FederationService
from systems.federation.task_delegation import TaskDelegationManager
from systems.federation.work_router import WorkRouter
from systems.federation.yield_pool import YieldPoolManager

__all__ = [
    "BountySplitter",
    "ExchangeProtocol",
    "FederationMarketplace",
    "FederationService",
    "HandshakeConfirmation",
    "HandshakeProcessor",
    "HandshakeRequest",
    "HandshakeResponse",
    "HandshakeResult",
    "IngestionPipeline",
    "ResourceSharingManager",
    "TaskDelegationManager",
    "WorkRouter",
    "YieldPoolManager",
]
