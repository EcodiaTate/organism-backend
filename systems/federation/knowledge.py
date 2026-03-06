"""
EcodiaOS — Federation Knowledge Exchange

Orchestrates the consent-based sharing of knowledge between federated
instances. All sharing follows a request-response protocol:

1. Remote instance sends a KnowledgeRequest
2. This instance checks trust level permissions
3. Equor performs constitutional review
4. Privacy filter strips individual data
5. Filtered knowledge is returned

Sharing is NEVER automatic. Every request is individually evaluated
against trust level, constitutional alignment, and privacy constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from primitives.federation import (
    SHARING_PERMISSIONS,
    FederationInteraction,
    FederationLink,
    InteractionOutcome,
    KnowledgeItem,
    KnowledgeRequest,
    KnowledgeResponse,
    KnowledgeType,
    PrivacyLevel,
    TrustLevel,
)
from systems.federation.privacy import PrivacyFilter

if TYPE_CHECKING:
    from systems.federation.reputation_staking import ReputationStakingManager
    from systems.memory.service import MemoryService

logger = structlog.get_logger("systems.federation.knowledge")


class KnowledgeExchangeManager:
    """
    Manages knowledge exchange between federated instances.

    Responsibilities:
      - Validate knowledge requests against trust permissions
      - Retrieve requested knowledge from local memory
      - Apply privacy filtering before sharing
      - Track sharing statistics for audit
      - Prepare outbound knowledge requests
    """

    def __init__(
        self,
        memory: MemoryService | None = None,
        privacy_filter: PrivacyFilter | None = None,
        max_items_per_request: int = 100,
        staking_manager: ReputationStakingManager | None = None,
    ) -> None:
        self._memory = memory
        self._privacy_filter = privacy_filter or PrivacyFilter()
        self._max_items_per_request = max_items_per_request
        self._staking_manager: ReputationStakingManager | None = staking_manager
        self._logger = logger.bind(component="knowledge_exchange")

        # Counters
        self._requests_received: int = 0
        self._requests_granted: int = 0
        self._requests_denied: int = 0
        self._items_shared: int = 0

    # ─── Inbound: Handle Requests from Remote Instances ─────────────

    async def handle_request(
        self,
        request: KnowledgeRequest,
        link: FederationLink,
        equor_permitted: bool = True,
    ) -> tuple[KnowledgeResponse, FederationInteraction]:
        """
        Handle an inbound knowledge request from a federated instance.

        Steps:
          1. Check trust level permits this knowledge type
          2. Verify Equor constitutional clearance
          3. Retrieve matching knowledge from memory
          4. Apply privacy filter
          5. Return filtered response

        Returns both the response and an interaction record for trust scoring.
        """
        self._requests_received += 1
        start_time = utc_now()

        # Step 1: Check trust level
        permitted_types = SHARING_PERMISSIONS.get(link.trust_level, [])
        if request.knowledge_type not in permitted_types:
            self._requests_denied += 1

            response = KnowledgeResponse(
                request_id=request.id,
                granted=False,
                reason=(
                    f"Trust level {link.trust_level.name} does not permit "
                    f"sharing {request.knowledge_type.value}."
                ),
                trust_level_required=_min_trust_for_type(request.knowledge_type),
            )

            interaction = self._build_interaction(
                link=link,
                interaction_type="knowledge_request",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description=f"Denied: insufficient trust for {request.knowledge_type.value}",
                start_time=start_time,
            )

            self._logger.info(
                "knowledge_request_denied_trust",
                remote_id=link.remote_instance_id,
                knowledge_type=request.knowledge_type.value,
                trust_level=link.trust_level.name,
            )

            return response, interaction

        # Step 2: Equor constitutional review
        if not equor_permitted:
            self._requests_denied += 1

            response = KnowledgeResponse(
                request_id=request.id,
                granted=False,
                reason="Constitutional review denied this knowledge share.",
            )

            interaction = self._build_interaction(
                link=link,
                interaction_type="knowledge_request",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description="Denied: Equor constitutional review blocked",
                start_time=start_time,
            )

            return response, interaction

        # Step 3: Retrieve knowledge from memory
        raw_items = await self._retrieve_knowledge(
            knowledge_type=request.knowledge_type,
            query=request.query,
            query_embedding=request.query_embedding,
            domain=request.domain,
            max_results=min(request.max_results, self._max_items_per_request),
        )

        # Step 4: Apply privacy filter
        filtered = await self._privacy_filter.filter(raw_items, link.trust_level)

        # Step 5: Build response
        self._requests_granted += 1
        self._items_shared += len(filtered.items)
        link.shared_knowledge_count += len(filtered.items)

        response = KnowledgeResponse(
            request_id=request.id,
            granted=True,
            knowledge=filtered.items,
            attribution=link.local_instance_id,
        )

        interaction = self._build_interaction(
            link=link,
            interaction_type="knowledge_request",
            direction="inbound",
            outcome=InteractionOutcome.SUCCESSFUL,
            description=f"Shared {len(filtered.items)} items ({request.knowledge_type.value})",
            start_time=start_time,
            trust_value=1.0,
        )

        # Step 6: Create reputation bonds for shared knowledge items
        if self._staking_manager and filtered.items:
            for item in filtered.items:
                certainty = self._estimate_claim_certainty(item)
                bond = await self._staking_manager.create_bond(
                    claim=item,
                    link=link,
                    claim_certainty=certainty,
                )
                if bond:
                    interaction.metadata[f"bond:{item.item_id}"] = bond.id

        self._logger.info(
            "knowledge_request_granted",
            remote_id=link.remote_instance_id,
            knowledge_type=request.knowledge_type.value,
            raw_items=len(raw_items),
            filtered_items=len(filtered.items),
            removed_by_privacy=filtered.items_removed_by_privacy,
        )

        return response, interaction

    # ─── Outbound: Prepare Requests to Remote Instances ─────────────

    def build_request(
        self,
        knowledge_type: KnowledgeType,
        query: str = "",
        query_embedding: list[float] | None = None,
        domain: str = "",
        max_results: int = 10,
        local_instance_id: str = "",
    ) -> KnowledgeRequest:
        """Build a knowledge request to send to a remote instance."""
        return KnowledgeRequest(
            requesting_instance_id=local_instance_id,
            knowledge_type=knowledge_type,
            query=query,
            query_embedding=query_embedding,
            domain=domain,
            max_results=max_results,
        )

    async def ingest_response(
        self,
        response: KnowledgeResponse,
        link: FederationLink,
    ) -> FederationInteraction:
        """
        Process a knowledge response received from a remote instance.

        Ingests the shared knowledge into local memory (with federation
        provenance) and records the interaction for trust scoring.
        """
        if not response.granted:
            return self._build_interaction(
                link=link,
                interaction_type="knowledge_response",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description=f"Remote denied: {response.reason}",
                start_time=utc_now(),
            )

        # Store received knowledge in memory with federation provenance
        if self._memory and response.knowledge:
            for item in response.knowledge:
                item.source_instance_id = response.attribution
            link.received_knowledge_count += len(response.knowledge)

        # Check received knowledge against our bonded outbound claims
        if self._staking_manager and response.knowledge:
            for item in response.knowledge:
                contradictions = await self._staking_manager.check_contradiction(
                    inbound_item=item,
                    source_instance_id=response.attribution,
                )
                for bond, evidence in contradictions:
                    remote_address = self._get_remote_wallet_address(link)
                    if remote_address:
                        await self._staking_manager.forfeit_bond(
                            bond=bond,
                            evidence=evidence,
                            remote_wallet_address=remote_address,
                        )

        return self._build_interaction(
            link=link,
            interaction_type="knowledge_response",
            direction="inbound",
            outcome=InteractionOutcome.SUCCESSFUL,
            description=f"Received {len(response.knowledge)} items",
            start_time=utc_now(),
            trust_value=0.5,  # Receiving knowledge builds trust, but less than sharing
        )

    # ─── Knowledge Retrieval ─────────────────────────────────────────

    async def _retrieve_knowledge(
        self,
        knowledge_type: KnowledgeType,
        query: str,
        query_embedding: list[float] | None,
        domain: str,
        max_results: int,
    ) -> list[KnowledgeItem]:
        """
        Retrieve knowledge from local memory for federation sharing.

        Maps KnowledgeType to appropriate memory queries.
        """
        if self._memory is None:
            return []

        items: list[KnowledgeItem] = []

        try:
            if knowledge_type == KnowledgeType.PUBLIC_ENTITIES:
                items = await self._retrieve_public_entities(query, max_results)

            elif knowledge_type == KnowledgeType.COMMUNITY_DESCRIPTION:
                items = await self._retrieve_community_description()

            elif knowledge_type == KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE:
                items = await self._retrieve_community_knowledge(query, max_results)

            elif knowledge_type == KnowledgeType.PROCEDURES:
                items = await self._retrieve_procedures(domain, max_results)

            elif knowledge_type == KnowledgeType.HYPOTHESES:
                items = await self._retrieve_hypotheses(domain, max_results)

            elif knowledge_type == KnowledgeType.ANONYMISED_PATTERNS:
                items = await self._retrieve_patterns(domain, max_results)

            elif knowledge_type == KnowledgeType.SCHEMA_STRUCTURES:
                items = await self._retrieve_schema(max_results)

        except Exception as exc:
            self._logger.error(
                "knowledge_retrieval_failed",
                knowledge_type=knowledge_type.value,
                error=str(exc),
            )

        return items

    async def _retrieve_public_entities(
        self, query: str, max_results: int
    ) -> list[KnowledgeItem]:
        """Retrieve non-person entities from the knowledge graph."""
        # Query memory for entities, excluding person-type entities
        try:
            response = await self._memory.retrieve(  # type: ignore[union-attr]
                query_text=query or "public knowledge",
                max_results=max_results,
            )
            items = []
            for trace in response.traces:
                items.append(KnowledgeItem(
                    item_id=new_id(),
                    knowledge_type=KnowledgeType.PUBLIC_ENTITIES,
                    privacy_level=PrivacyLevel.PUBLIC,
                    content={
                        "summary": trace.summary,  # type: ignore[union-attr]
                        "entities": [
                            {"name": e.name, "type": e.type, "description": e.description}
                            for e in (trace.entities or [])  # type: ignore[union-attr]
                            if e.type.lower() not in ("person", "individual", "member")
                        ],
                    },
                ))
            return items
        except Exception:
            return []

    async def _retrieve_community_description(self) -> list[KnowledgeItem]:
        """Retrieve the community description from the Self node."""
        try:
            self_node = await self._memory.get_self()  # type: ignore[union-attr]
            if self_node is None:
                return []
            return [KnowledgeItem(
                item_id=new_id(),
                knowledge_type=KnowledgeType.COMMUNITY_DESCRIPTION,
                privacy_level=PrivacyLevel.PUBLIC,
                content={
                    "name": self_node.name,
                    "community_context": getattr(self_node, "community_context", ""),
                },
            )]
        except Exception:
            return []

    async def _retrieve_community_knowledge(
        self, query: str, max_results: int
    ) -> list[KnowledgeItem]:
        """Retrieve community-level aggregated knowledge."""
        try:
            response = await self._memory.retrieve(  # type: ignore[union-attr]
                query_text=query or "community patterns",
                max_results=max_results,
            )
            items = []
            for community in (response.communities or []):
                items.append(KnowledgeItem(
                    item_id=new_id(),
                    knowledge_type=KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
                    privacy_level=PrivacyLevel.COMMUNITY_ONLY,
                    content={
                        "summary": community.summary,
                        "coherence_score": community.coherence_score,
                    },
                ))
            return items
        except Exception:
            return []

    async def _retrieve_procedures(
        self, domain: str, max_results: int
    ) -> list[KnowledgeItem]:
        """Retrieve learned procedures from Evo."""
        # Procedures would come from Evo's procedure store
        # For now, return empty — will be wired when Evo exposes procedures
        return []

    async def _retrieve_hypotheses(
        self, domain: str, max_results: int
    ) -> list[KnowledgeItem]:
        """Retrieve active hypotheses from Evo."""
        return []

    async def _retrieve_patterns(
        self, domain: str, max_results: int
    ) -> list[KnowledgeItem]:
        """Retrieve anonymised patterns from memory communities."""
        return await self._retrieve_community_knowledge(
            domain or "detected patterns", max_results
        )

    async def _retrieve_schema(self, max_results: int) -> list[KnowledgeItem]:
        """Retrieve schema structures (community-level semantic graph)."""
        try:
            response = await self._memory.retrieve(  # type: ignore[union-attr]
                query_text="schema structure knowledge organization",
                max_results=max_results,
            )
            items = []
            for community in (response.communities or []):
                items.append(KnowledgeItem(
                    item_id=new_id(),
                    knowledge_type=KnowledgeType.SCHEMA_STRUCTURES,
                    privacy_level=PrivacyLevel.COMMUNITY_ONLY,
                    content={
                        "summary": community.summary,
                        "member_count": len(community.member_entity_ids)
                        if hasattr(community, "member_entity_ids")
                        else 0,
                    },
                ))
            return items
        except Exception:
            return []

    # ─── Reputation Staking Helpers ──────────────────────────────────

    @staticmethod
    def _estimate_claim_certainty(item: KnowledgeItem) -> float:
        """
        Derive a 0-1 certainty score from the knowledge item.

        Uses confidence metadata if present, otherwise defaults to 0.5
        (moderate certainty).
        """
        content = item.content
        if isinstance(content, dict):
            # Check for confidence or coherence_score in content
            if "confidence" in content:
                return max(0.0, min(1.0, float(content["confidence"])))
            if "coherence_score" in content:
                return max(0.0, min(1.0, float(content["coherence_score"])))
        return 0.5

    @staticmethod
    def _get_remote_wallet_address(link: FederationLink) -> str:
        """
        Get the remote instance's wallet address from its identity card.

        Returns empty string if no wallet address is available.
        """
        if link.remote_identity and link.remote_identity.wallet_address:
            return link.remote_identity.wallet_address
        return ""

    # ─── Helpers ─────────────────────────────────────────────────────

    def _build_interaction(
        self,
        link: FederationLink,
        interaction_type: str,
        direction: str,
        outcome: InteractionOutcome,
        description: str,
        start_time: Any,
        trust_value: float = 1.0,
    ) -> FederationInteraction:
        elapsed = utc_now() - start_time
        return FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type=interaction_type,
            direction=direction,
            outcome=outcome,
            description=description,
            trust_value=trust_value,
            latency_ms=int(elapsed.total_seconds() * 1000),
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "requests_received": self._requests_received,
            "requests_granted": self._requests_granted,
            "requests_denied": self._requests_denied,
            "items_shared": self._items_shared,
            "privacy_filter": self._privacy_filter.stats,
        }


# ─── Helpers ─────────────────────────────────────────────────────


def _min_trust_for_type(knowledge_type: KnowledgeType) -> TrustLevel:
    """Find the minimum trust level required for a knowledge type."""
    for level in TrustLevel:
        permitted = SHARING_PERMISSIONS.get(level, [])
        if knowledge_type in permitted:
            return level
    return TrustLevel.ALLY
