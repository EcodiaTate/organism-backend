"""
EcodiaOS -- Simula Proposal Intelligence

Smart proposal management: deduplication, prioritization, dependency
analysis, and cost estimation. Maximizes evolution quality per LLM
token by using cheap heuristics first and LLM only for ambiguous cases.

Key design:
  - Deduplication: 3-tier (exact prefix → category overlap → LLM similarity)
  - Prioritization: formula-based scoring, no LLM needed
  - Dependency analysis: rule-based ordering, no LLM needed
  - Cost estimation: heuristic lookup table, no LLM needed

Budget impact: Zero LLM tokens for normal operation.
LLM used only when >5 proposals need semantic deduplication (~300 tokens).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.simula.evolution_types import (
    ChangeCategory,
    EvolutionProposal,
    ProposalCluster,
    ProposalPriority,
    ProposalStatus,
    RiskLevel,
)

if TYPE_CHECKING:
    from clients.embedding import EmbeddingClient
    from clients.llm import LLMProvider
    from systems.simula.analytics import EvolutionAnalyticsEngine

logger = structlog.get_logger().bind(system="simula.intelligence")

# Cost heuristics by category (0.0-1.0 scale)
_CATEGORY_COST: dict[ChangeCategory, float] = {
    ChangeCategory.ADJUST_BUDGET: 0.1,
    ChangeCategory.ADD_EXECUTOR: 0.4,
    ChangeCategory.ADD_INPUT_CHANNEL: 0.4,
    ChangeCategory.ADD_PATTERN_DETECTOR: 0.4,
    ChangeCategory.MODIFY_CONTRACT: 0.7,
    ChangeCategory.MODIFY_CYCLE_TIMING: 0.5,
    ChangeCategory.CHANGE_CONSOLIDATION: 0.6,
    ChangeCategory.ADD_SYSTEM_CAPABILITY: 0.9,
}

# Impact heuristics by category (0.0-1.0 scale)
_CATEGORY_IMPACT: dict[ChangeCategory, float] = {
    ChangeCategory.ADJUST_BUDGET: 0.3,
    ChangeCategory.ADD_EXECUTOR: 0.6,
    ChangeCategory.ADD_INPUT_CHANNEL: 0.7,
    ChangeCategory.ADD_PATTERN_DETECTOR: 0.5,
    ChangeCategory.MODIFY_CONTRACT: 0.8,
    ChangeCategory.MODIFY_CYCLE_TIMING: 0.4,
    ChangeCategory.CHANGE_CONSOLIDATION: 0.5,
    ChangeCategory.ADD_SYSTEM_CAPABILITY: 0.9,
}

# Minimum description prefix length for exact dedup matching
_DEDUP_PREFIX_LEN: int = 50

# Minimum proposals before triggering LLM-based dedup
_LLM_DEDUP_THRESHOLD: int = 5


class ProposalIntelligence:
    """
    Smart proposal management for Simula.

    Provides deduplication, prioritization, dependency analysis,
    and cost estimation — all optimized for minimal token usage.

    Stage 1B upgrade: Tier 3 dedup now uses voyage-code-3 embeddings
    for cosine similarity instead of LLM-based text comparison.
    This is both cheaper (no LLM tokens) and more precise.
    """

    # Cosine similarity threshold for embedding-based dedup
    _EMBEDDING_DEDUP_THRESHOLD: float = 0.85

    def __init__(
        self,
        llm: LLMProvider | None = None,
        analytics: EvolutionAnalyticsEngine | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._llm = llm
        self._analytics = analytics
        self._embeddings = embedding_client
        self._log = logger
        # Dedup precision tracking (Stage 1B.5)
        self._dedup_stats = {
            "tier1_matches": 0,
            "tier2_matches": 0,
            "tier3_embedding_matches": 0,
            "tier3_llm_fallback_matches": 0,
            "embedding_dedup_calls": 0,
            "embedding_dedup_latency_ms": 0.0,
        }

    # ─── Prioritization ──────────────────────────────────────────────────────

    async def prioritize(
        self, proposals: list[EvolutionProposal],
    ) -> list[ProposalPriority]:
        """
        Score and rank proposals by:
          priority = evidence_strength * expected_impact / max(0.1, risk * cost)

        Pure heuristic scoring — zero LLM tokens.
        Proposals with higher scores should be processed first.
        """
        priorities: list[ProposalPriority] = []

        for proposal in proposals:
            evidence_strength = self._compute_evidence_strength(proposal)
            expected_impact = _CATEGORY_IMPACT.get(proposal.category, 0.5)
            estimated_risk = self._compute_risk_estimate(proposal)
            estimated_cost = self.estimate_cost(proposal)

            # Priority formula
            denominator = max(0.1, estimated_risk * estimated_cost)
            score = (evidence_strength * expected_impact) / denominator

            # Boost for proposals already partially processed
            if proposal.status == ProposalStatus.APPROVED:
                score *= 1.5

            reasoning = (
                f"evidence={evidence_strength:.2f}, impact={expected_impact:.2f}, "
                f"risk={estimated_risk:.2f}, cost={estimated_cost:.2f}"
            )

            priorities.append(ProposalPriority(
                proposal_id=proposal.id,
                priority_score=round(score, 3),
                evidence_strength=round(evidence_strength, 3),
                expected_impact=round(expected_impact, 3),
                estimated_risk=round(estimated_risk, 3),
                estimated_cost=round(estimated_cost, 3),
                reasoning=reasoning,
            ))

        # Sort by score descending
        priorities.sort(key=lambda p: p.priority_score, reverse=True)

        self._log.info(
            "proposals_prioritized",
            count=len(priorities),
            top_score=priorities[0].priority_score if priorities else 0.0,
        )
        return priorities

    def _compute_evidence_strength(self, proposal: EvolutionProposal) -> float:
        """
        Compute evidence strength from the proposal's evidence list.
        More evidence items = stronger signal. Capped at 1.0.
        """
        count = len(proposal.evidence)
        if count == 0:
            return 0.2  # minimal evidence
        # Logarithmic scaling: 1 item = 0.3, 5 items = 0.7, 10+ items = 0.9+
        import math
        return min(1.0, 0.2 + 0.3 * math.log1p(count))

    def _compute_risk_estimate(self, proposal: EvolutionProposal) -> float:
        """
        Estimate risk from simulation results and analytics history.
        Returns 0.0-1.0 scale.
        """
        # If simulation has run, use its risk level
        if proposal.simulation is not None:
            risk_map = {
                RiskLevel.LOW: 0.15,
                RiskLevel.MODERATE: 0.4,
                RiskLevel.HIGH: 0.7,
                RiskLevel.UNACCEPTABLE: 1.0,
            }
            return risk_map.get(proposal.simulation.risk_level, 0.4)

        # Otherwise, use category-based heuristic
        # Higher-impact categories carry more risk
        return _CATEGORY_COST.get(proposal.category, 0.5)

    # ─── Deduplication ───────────────────────────────────────────────────────

    async def deduplicate(
        self, proposals: list[EvolutionProposal],
    ) -> list[ProposalCluster]:
        """
        Detect semantically similar proposals in three tiers:
          Tier 1: Exact description prefix match (zero cost)
          Tier 2: Category + affected_systems overlap (zero cost)
          Tier 3: LLM similarity check (only if >5 proposals, ~300 tokens)

        Returns clusters where member proposals could be merged.
        """
        if len(proposals) < 2:
            return []

        clusters: list[ProposalCluster] = []
        clustered_ids: set[str] = set()

        # Tier 1: Exact description prefix match
        prefix_groups: dict[str, list[EvolutionProposal]] = {}
        for p in proposals:
            prefix = p.description[:_DEDUP_PREFIX_LEN].lower().strip()
            prefix_groups.setdefault(prefix, []).append(p)

        for prefix, group in prefix_groups.items():
            if len(group) < 2:
                continue
            rep = group[0]
            members = [p.id for p in group]
            clustered_ids.update(members)
            clusters.append(ProposalCluster(
                representative_id=rep.id,
                member_ids=members,
                similarity_scores=[1.0] * len(members),
                merge_recommendation=f"Identical prefix: '{prefix[:30]}...'",
            ))

        # Tier 2: Category + affected_systems overlap
        unclustered = [p for p in proposals if p.id not in clustered_ids]
        cat_system_groups: dict[str, list[EvolutionProposal]] = {}
        for p in unclustered:
            key = f"{p.category.value}::{','.join(sorted(p.change_spec.affected_systems))}"
            cat_system_groups.setdefault(key, []).append(p)

        for key, group in cat_system_groups.items():
            if len(group) < 2:
                continue
            rep = group[0]
            members = [p.id for p in group]
            clustered_ids.update(members)
            clusters.append(ProposalCluster(
                representative_id=rep.id,
                member_ids=members,
                similarity_scores=[0.7] * len(members),
                merge_recommendation=f"Same category and affected systems: {key}",
            ))

        # Tier 3: Embedding-based semantic similarity (preferred) or LLM fallback
        still_unclustered = [p for p in proposals if p.id not in clustered_ids]
        if len(still_unclustered) >= _LLM_DEDUP_THRESHOLD:
            if self._embeddings is not None:
                # Stage 1B: voyage-code-3 cosine similarity — cheaper and more precise
                embedding_clusters = await self._embedding_deduplicate(still_unclustered)
                clusters.extend(embedding_clusters)
            elif self._llm is not None:
                # Fallback: LLM-based semantic comparison (~300 tokens)
                llm_clusters = await self._llm_deduplicate(still_unclustered)
                clusters.extend(llm_clusters)

        if clusters:
            self._log.info(
                "dedup_complete",
                clusters=len(clusters),
                total_duplicates=sum(len(c.member_ids) for c in clusters),
                dedup_stats=self._dedup_stats,
            )
        return clusters

    async def _embedding_deduplicate(
        self, proposals: list[EvolutionProposal],
    ) -> list[ProposalCluster]:
        """
        Stage 1B: Embedding-based semantic dedup using voyage-code-3.

        Embeds all proposal descriptions, then finds pairs with cosine
        similarity above the threshold. Groups them into clusters.

        Zero LLM tokens. Cost: ~0.001 per proposal via Voyage API.
        """
        from clients.embedding import cosine_similarity

        assert self._embeddings is not None
        self._dedup_stats["embedding_dedup_calls"] += 1
        t_start = time.monotonic()

        # Build description texts for embedding
        texts = [
            f"{p.category.value}: {p.description[:200]}"
            for p in proposals[:20]  # Cap at 20 to control API cost
        ]

        try:
            import asyncio
            embeddings = await asyncio.wait_for(
                self._embeddings.embed_batch(texts),
                timeout=10.0,
            )
        except Exception as exc:
            self._log.warning("embedding_dedup_failed", error=str(exc))
            self._dedup_stats["embedding_dedup_latency_ms"] += (
                (time.monotonic() - t_start) * 1000
            )
            return []

        # Pairwise cosine similarity — find clusters above threshold
        clusters: list[ProposalCluster] = []
        clustered: set[int] = set()

        for i in range(len(embeddings)):
            if i in clustered:
                continue
            group = [i]
            for j in range(i + 1, len(embeddings)):
                if j in clustered:
                    continue
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._EMBEDDING_DEDUP_THRESHOLD:
                    group.append(j)
                    clustered.add(j)

            if len(group) >= 2:
                clustered.add(i)
                member_ids = [proposals[idx].id for idx in group]
                similarities = []
                for idx in group:
                    if idx == group[0]:
                        similarities.append(1.0)
                    else:
                        sim = cosine_similarity(embeddings[group[0]], embeddings[idx])
                        similarities.append(round(sim, 3))

                clusters.append(ProposalCluster(
                    representative_id=member_ids[0],
                    member_ids=member_ids,
                    similarity_scores=similarities,
                    merge_recommendation=(
                        f"Embedding similarity ≥{self._EMBEDDING_DEDUP_THRESHOLD} "
                        f"(voyage-code-3)"
                    ),
                ))
                self._dedup_stats["tier3_embedding_matches"] += len(group)

        latency_ms = (time.monotonic() - t_start) * 1000
        self._dedup_stats["embedding_dedup_latency_ms"] += latency_ms

        self._log.info(
            "embedding_dedup_complete",
            proposals_checked=len(proposals),
            clusters_found=len(clusters),
            latency_ms=round(latency_ms, 1),
        )
        return clusters

    async def _llm_deduplicate(
        self, proposals: list[EvolutionProposal],
    ) -> list[ProposalCluster]:
        """LLM-based semantic similarity check (fallback). ~300 tokens."""
        descriptions = "\n".join(
            f"{i+1}. [{p.id[:8]}] {p.category.value}: {p.description[:100]}"
            for i, p in enumerate(proposals[:10])
        )

        prompt = (
            "Below are evolution proposals for an AI system. "
            "Identify any that are semantically similar enough to be duplicates.\n\n"
            f"{descriptions}\n\n"
            "Reply with groups of similar proposals by their numbers.\n"
            "Format: GROUP: 1, 3 (reason)\n"
            "If no duplicates found, reply: NONE"
        )

        try:
            import asyncio
            response = await asyncio.wait_for(
                self._llm.evaluate(prompt=prompt, max_tokens=200, temperature=0.1),  # type: ignore[union-attr]
                timeout=8.0,
            )

            clusters: list[ProposalCluster] = []
            for line in response.text.strip().splitlines():
                line = line.strip()
                if line.upper() == "NONE" or "GROUP" not in line.upper():
                    continue
                try:
                    _, nums_part = line.split(":", 1)
                    reason_start = nums_part.find("(")
                    if reason_start > 0:
                        reason = nums_part[reason_start:].strip("() ")
                        nums_part = nums_part[:reason_start]
                    else:
                        reason = ""

                    indices = [
                        int(n.strip()) - 1 for n in nums_part.split(",") if n.strip().isdigit()
                    ]
                    valid = [i for i in indices if 0 <= i < len(proposals)]
                    if len(valid) >= 2:
                        members = [proposals[i].id for i in valid]
                        clusters.append(ProposalCluster(
                            representative_id=members[0],
                            member_ids=members,
                            similarity_scores=[0.6] * len(members),
                            merge_recommendation=reason or "LLM-detected similarity",
                        ))
                        self._dedup_stats["tier3_llm_fallback_matches"] += len(valid)
                except (ValueError, IndexError):
                    continue

            return clusters
        except Exception as exc:
            self._log.warning("llm_dedup_failed", error=str(exc))
            return []

    # ─── Dependency Analysis ─────────────────────────────────────────────────

    async def analyze_dependencies(
        self, proposals: list[EvolutionProposal],
    ) -> list[tuple[str, str, str]]:
        """
        Detect ordering dependencies between proposals.
        Returns list of (before_id, after_id, reason) tuples.

        Rule-based analysis — zero LLM tokens:
        - ADD_EXECUTOR should come before MODIFY_CONTRACT referencing axon
        - ADD_INPUT_CHANNEL before MODIFY_CONTRACT referencing atune
        - ADJUST_BUDGET after the thing it's budgeting for is added
        - ADD_SYSTEM_CAPABILITY is a superset that depends on components
        """
        if len(proposals) < 2:
            return []

        dependencies: list[tuple[str, str, str]] = []

        # Build lookup
        additive = [p for p in proposals if p.category in {
            ChangeCategory.ADD_EXECUTOR,
            ChangeCategory.ADD_INPUT_CHANNEL,
            ChangeCategory.ADD_PATTERN_DETECTOR,
        }]
        contracts = [p for p in proposals if p.category == ChangeCategory.MODIFY_CONTRACT]
        capabilities = [p for p in proposals if p.category == ChangeCategory.ADD_SYSTEM_CAPABILITY]
        budgets = [p for p in proposals if p.category == ChangeCategory.ADJUST_BUDGET]

        # Additive changes should come before contract modifications
        # that reference the same system
        for add_p in additive:
            add_systems = set(add_p.change_spec.affected_systems)
            for contract_p in contracts:
                contract_systems = set(contract_p.change_spec.affected_systems)
                overlap = add_systems & contract_systems
                if overlap:
                    dependencies.append((
                        add_p.id,
                        contract_p.id,
                        f"Add {add_p.category.value} before modifying contracts for {overlap}",
                    ))

        # Additive changes should come before capability additions
        for add_p in additive:
            for cap_p in capabilities:
                cap_systems = set(cap_p.change_spec.affected_systems)
                add_systems = set(add_p.change_spec.affected_systems)
                if cap_systems & add_systems:
                    dependencies.append((
                        add_p.id,
                        cap_p.id,
                        "Add component before adding system capability",
                    ))

        # Budget changes should come after the thing they budget for
        for budget_p in budgets:
            param = budget_p.change_spec.budget_parameter or ""
            for add_p in additive:
                # If the budget parameter references the additive system
                add_systems_list = add_p.change_spec.affected_systems
                for sys in add_systems_list:
                    if sys in param:
                        dependencies.append((
                            add_p.id,
                            budget_p.id,
                            f"Add component before adjusting its budget ({param})",
                        ))

        if dependencies:
            self._log.info(
                "dependencies_detected",
                count=len(dependencies),
            )
        return dependencies

    # ─── Cost Estimation ─────────────────────────────────────────────────────

    def estimate_cost(self, proposal: EvolutionProposal) -> float:
        """
        Heuristic cost estimation (0.0-1.0 scale).
        Zero LLM tokens — pure lookup + adjustment.
        """
        base_cost = _CATEGORY_COST.get(proposal.category, 0.5)

        # Adjust for complexity signals
        spec = proposal.change_spec
        if spec.affected_systems and len(spec.affected_systems) > 1:
            base_cost = min(1.0, base_cost + 0.1 * (len(spec.affected_systems) - 1))

        if spec.contract_changes and len(spec.contract_changes) > 2:
            base_cost = min(1.0, base_cost + 0.1)

        return round(base_cost, 2)

    # ─── Duplicate Detection Helper ──────────────────────────────────────────

    def get_dedup_stats(self) -> dict[str, Any]:
        """Return dedup precision benchmarking stats (Stage 1B.5)."""
        return dict(self._dedup_stats)

    def is_duplicate(
        self,
        proposal: EvolutionProposal,
        clusters: list[ProposalCluster],
    ) -> bool:
        """
        Check if a proposal appears in any cluster as a non-representative member.
        If it's in a cluster but not the representative, it's a duplicate.
        """
        for cluster in clusters:
            if proposal.id in cluster.member_ids and proposal.id != cluster.representative_id:
                return True
        return False
