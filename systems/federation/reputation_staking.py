"""
EcodiaOS — Federation Reputation Staking

Manages cryptoeconomic bonds attached to federated knowledge claims.
When this instance shares knowledge outbound, a USDC bond proportional
to claim certainty is escrowed on-chain. If a remote instance later
contradicts the claim with evidence, the bond is forfeited — turning
the Honesty drive into a Schelling-point economic mechanism.

Bond lifecycle:
  1. CREATE  — knowledge shared → USDC transferred to escrow address
  2. ACTIVE  — bond lives until expiry or contradiction
  3. FORFEIT — contradiction detected → USDC sent to remote instance
  4. RECOVER — bond expires without contradiction → USDC returned to treasury

This is the enforcement arm of the Honesty drive in the federation
context. It makes truthfulness economically rational by creating a
Schelling point around honest reporting.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from primitives.federation import (
    BondStatus,
    ContradictionEvidence,
    FederationLink,
    KnowledgeItem,
    ReputationBond,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from clients.wallet import WalletClient
    from config import StakingConfig
    from telemetry.metrics import MetricCollector

logger = structlog.get_logger("systems.federation.reputation_staking")

# Default tier discounts when config is not provided
_DEFAULT_TIER_DISCOUNTS: dict[str, float] = {
    "ALLY": 0.5,
    "PARTNER": 0.75,
    "COLLEAGUE": 1.0,
    "ACQUAINTANCE": 1.25,
}


class ReputationStakingManager:
    """
    Manages reputation bonds for federated knowledge claims.

    Responsibilities:
      - Create bonds when knowledge is shared outbound
      - Detect contradictions when inbound knowledge conflicts with bonded claims
      - Forfeit bonds on confirmed contradiction
      - Recover bonds on expiry
      - Persist bond state to Redis
      - Report metrics
    """

    def __init__(
        self,
        wallet: WalletClient | None = None,
        redis: RedisClient | None = None,
        metrics: MetricCollector | None = None,
        config: StakingConfig | None = None,
        escrow_address: str = "",
    ) -> None:
        self._wallet = wallet
        self._redis = redis
        self._metrics = metrics
        self._escrow_address = escrow_address
        self._logger = logger.bind(component="reputation_staking")

        # Config values (with safe defaults)
        self._base_bond_usdc = Decimal("1.00")
        self._max_total_bonded_usdc = Decimal("100.00")
        self._max_per_instance_bonded_usdc = Decimal("25.00")
        self._bond_expiry_days = 90
        self._contradiction_similarity_threshold = 0.85
        self._contradiction_divergence_threshold = 0.3
        self._min_certainty_for_bond = 0.1
        self._tier_discounts = dict(_DEFAULT_TIER_DISCOUNTS)

        if config is not None:
            self._base_bond_usdc = config.base_bond_usdc
            self._max_total_bonded_usdc = config.max_total_bonded_usdc
            self._max_per_instance_bonded_usdc = config.max_per_instance_bonded_usdc
            self._bond_expiry_days = config.bond_expiry_days
            self._contradiction_similarity_threshold = config.contradiction_similarity_threshold
            self._contradiction_divergence_threshold = config.contradiction_divergence_threshold
            self._min_certainty_for_bond = config.min_certainty_for_bond
            if config.tier_discounts:
                self._tier_discounts = dict(config.tier_discounts)
            if config.escrow_address:
                self._escrow_address = config.escrow_address

        # In-memory bond registry: bond_id -> ReputationBond
        self._bonds: dict[str, ReputationBond] = {}

    # ─── Bond Creation ───────────────────────────────────────────────

    async def create_bond(
        self,
        claim: KnowledgeItem,
        link: FederationLink,
        claim_certainty: float,
    ) -> ReputationBond | None:
        """
        Create a reputation bond for an outbound knowledge claim.

        Returns None if certainty too low, budget exceeded, no escrow
        address configured, or wallet transfer fails.
        """
        # Gate: certainty floor
        if claim_certainty < self._min_certainty_for_bond:
            self._logger.debug(
                "bond_skipped_low_certainty",
                claim_id=claim.item_id,
                certainty=claim_certainty,
            )
            return None

        # Gate: escrow address must be configured
        if not self._escrow_address:
            self._logger.debug("bond_skipped_no_escrow_address")
            return None

        # Compute bond amount
        amount = self._compute_bond_amount(claim_certainty, link)
        if amount <= Decimal(0):
            return None

        # Budget check: total bonded
        if self.total_bonded_usdc + amount > self._max_total_bonded_usdc:
            self._logger.info(
                "bond_skipped_total_budget",
                current_total=str(self.total_bonded_usdc),
                requested=str(amount),
                max_total=str(self._max_total_bonded_usdc),
            )
            return None

        # Budget check: per-instance
        instance_bonded = self._bonded_for_instance(link.remote_instance_id)
        if instance_bonded + amount > self._max_per_instance_bonded_usdc:
            self._logger.info(
                "bond_skipped_instance_budget",
                remote_id=link.remote_instance_id,
                current_instance=str(instance_bonded),
                requested=str(amount),
                max_instance=str(self._max_per_instance_bonded_usdc),
            )
            return None

        # Hash claim content
        content_hash = self._hash_claim_content(claim.content)

        # Compute expiry
        expires_at = utc_now() + timedelta(days=self._bond_expiry_days)

        # Create bond object (pre-escrow)
        bond = ReputationBond(
            claim_id=claim.item_id,
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            claim_content_hash=content_hash,
            claim_embedding=claim.embedding,
            bond_amount_usdc=amount,
            bond_expires_at=expires_at,
            status=BondStatus.ACTIVE,
            claim_certainty=claim_certainty,
        )

        # Transfer USDC to escrow
        if self._wallet:
            try:
                tx = await self._wallet.transfer(
                    amount=str(amount),
                    destination_address=self._escrow_address,
                    asset="usdc",
                )
                bond.escrow_tx_hash = tx.tx_hash
            except Exception as exc:
                bond.status = BondStatus.ESCROW_FAILED
                self._bonds[bond.id] = bond
                await self._persist_bond(bond)

                self._logger.warning(
                    "bond_escrow_failed",
                    bond_id=bond.id,
                    claim_id=claim.item_id,
                    amount=str(amount),
                    error=str(exc),
                )

                if self._metrics:
                    await self._metrics.record(
                        "federation", "staking.bond_escrow_failed", 1.0,
                        labels={"remote_instance_id": link.remote_instance_id},
                    )
                return bond

        # Persist and record
        self._bonds[bond.id] = bond
        await self._persist_bond(bond)

        self._logger.info(
            "reputation_bond_issued",
            bond_id=bond.id,
            claim_id=claim.item_id,
            remote_instance_id=link.remote_instance_id,
            amount=str(amount),
            certainty=claim_certainty,
            expires_at=str(expires_at),
            escrow_tx=bond.escrow_tx_hash,
        )

        if self._metrics:
            await self._metrics.record(
                "federation", "staking.bond_created", float(amount),
                labels={"remote_instance_id": link.remote_instance_id},
            )

        return bond

    # ─── Contradiction Detection ─────────────────────────────────────

    async def check_contradiction(
        self,
        inbound_item: KnowledgeItem,
        source_instance_id: str,
    ) -> list[tuple[ReputationBond, ContradictionEvidence]]:
        """
        Check if inbound knowledge contradicts any active bonded claims.

        Uses embedding cosine similarity for topic matching and content
        divergence for contradiction detection.
        """
        if inbound_item.embedding is None:
            return []

        contradictions: list[tuple[ReputationBond, ContradictionEvidence]] = []
        inbound_hash = self._hash_claim_content(inbound_item.content)

        for bond in self._bonds.values():
            if bond.status != BondStatus.ACTIVE:
                continue
            if bond.claim_embedding is None:
                continue

            # Cosine similarity for topic matching
            similarity = self._cosine_similarity(
                inbound_item.embedding, bond.claim_embedding,
            )

            if similarity < self._contradiction_similarity_threshold:
                continue  # Different topic — not a contradiction

            # Same topic detected. Check for content divergence.
            # If the content hash is identical, it's reinforcement.
            if inbound_hash == bond.claim_content_hash:
                continue

            # Compute content word overlap to distinguish contradiction from elaboration
            inbound_words = self._extract_words(inbound_item.content)
            claim_words = self._extract_words_from_hash_source(bond)
            overlap = self._jaccard_similarity(inbound_words, claim_words)

            if overlap > self._contradiction_divergence_threshold:
                continue  # High overlap = elaboration/reinforcement, not contradiction

            # Low overlap on same topic = contradiction candidate
            evidence = ContradictionEvidence(
                contradicting_item_id=inbound_item.item_id,
                contradicting_content_hash=inbound_hash,
                similarity_score=similarity,
                explanation=(
                    f"Embedding similarity {similarity:.3f} indicates same topic, "
                    f"but content overlap {overlap:.3f} is below threshold "
                    f"({self._contradiction_divergence_threshold}), suggesting contradiction."
                ),
                source_instance_id=source_instance_id,
            )
            contradictions.append((bond, evidence))

            self._logger.info(
                "contradiction_detected",
                bond_id=bond.id,
                claim_id=bond.claim_id,
                contradicting_item_id=inbound_item.item_id,
                similarity=round(similarity, 3),
                overlap=round(overlap, 3),
                source=source_instance_id,
            )

            if self._metrics:
                await self._metrics.record(
                    "federation", "staking.contradiction_detected", 1.0,
                    labels={"remote_instance_id": source_instance_id},
                )

        return contradictions

    # ─── Bond Forfeit ────────────────────────────────────────────────

    async def forfeit_bond(
        self,
        bond: ReputationBond,
        evidence: ContradictionEvidence,
        remote_wallet_address: str,
    ) -> bool:
        """
        Forfeit a bond: transfer escrowed USDC to the remote instance.

        Returns True if the forfeit was completed (or the bond was already
        forfeited). Returns False if the wallet transfer failed.
        """
        if bond.status != BondStatus.ACTIVE:
            return bond.status == BondStatus.FORFEITED

        if not remote_wallet_address:
            self._logger.warning(
                "forfeit_skipped_no_wallet_address",
                bond_id=bond.id,
                remote_id=bond.remote_instance_id,
            )
            return False

        # Transfer USDC to the remote instance
        if self._wallet:
            try:
                tx = await self._wallet.transfer(
                    amount=str(bond.bond_amount_usdc),
                    destination_address=remote_wallet_address,
                    asset="usdc",
                )
                bond.forfeit_tx_hash = tx.tx_hash
            except Exception as exc:
                self._logger.error(
                    "bond_forfeit_transfer_failed",
                    bond_id=bond.id,
                    amount=str(bond.bond_amount_usdc),
                    destination=remote_wallet_address,
                    error=str(exc),
                )
                return False

        # Update bond state
        bond.status = BondStatus.FORFEITED
        bond.forfeit_evidence = evidence
        await self._persist_bond(bond)

        self._logger.warning(
            "reputation_bond_forfeited",
            bond_id=bond.id,
            claim_id=bond.claim_id,
            remote_instance_id=bond.remote_instance_id,
            amount=str(bond.bond_amount_usdc),
            evidence_summary=evidence.explanation[:200],
            forfeit_tx=bond.forfeit_tx_hash,
        )

        if self._metrics:
            await self._metrics.record(
                "federation", "staking.bond_forfeited", float(bond.bond_amount_usdc),
                labels={"remote_instance_id": bond.remote_instance_id},
            )

        return True

    # ─── Bond Recovery ───────────────────────────────────────────────

    async def recover_expired_bonds(self) -> list[ReputationBond]:
        """
        Scan active bonds and return USDC for any that have expired
        without contradiction. Called periodically alongside trust decay.
        """
        now = utc_now()
        recovered: list[ReputationBond] = []

        for bond in list(self._bonds.values()):
            if bond.status != BondStatus.ACTIVE:
                continue
            if bond.bond_expires_at > now:
                continue  # Not yet expired

            # Transfer USDC back to treasury (same escrow address acts as treasury)
            if self._wallet and self._escrow_address:
                try:
                    tx = await self._wallet.transfer(
                        amount=str(bond.bond_amount_usdc),
                        destination_address=self._escrow_address,
                        asset="usdc",
                    )
                    bond.return_tx_hash = tx.tx_hash
                except Exception as exc:
                    self._logger.warning(
                        "bond_recovery_transfer_failed",
                        bond_id=bond.id,
                        amount=str(bond.bond_amount_usdc),
                        error=str(exc),
                    )
                    continue  # Retry on next cycle

            bond.status = BondStatus.EXPIRED_RETURNED
            await self._persist_bond(bond)
            recovered.append(bond)

            self._logger.info(
                "reputation_bond_recovered",
                bond_id=bond.id,
                claim_id=bond.claim_id,
                remote_instance_id=bond.remote_instance_id,
                amount=str(bond.bond_amount_usdc),
                return_tx=bond.return_tx_hash,
            )

            if self._metrics:
                await self._metrics.record(
                    "federation", "staking.bond_recovered", float(bond.bond_amount_usdc),
                    labels={"remote_instance_id": bond.remote_instance_id},
                )

        return recovered

    # ─── Queries ─────────────────────────────────────────────────────

    def get_bonds_for_instance(self, remote_instance_id: str) -> list[ReputationBond]:
        """All bonds (any status) for a specific remote instance."""
        return [
            b for b in self._bonds.values()
            if b.remote_instance_id == remote_instance_id
        ]

    def get_active_bonds(self) -> list[ReputationBond]:
        """All currently active bonds."""
        return [b for b in self._bonds.values() if b.status == BondStatus.ACTIVE]

    @property
    def total_bonded_usdc(self) -> Decimal:
        """Sum of all active bond amounts."""
        return sum(
            (b.bond_amount_usdc for b in self._bonds.values() if b.status == BondStatus.ACTIVE),
            Decimal(0),
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Reputation staking dashboard data."""
        active = [b for b in self._bonds.values() if b.status == BondStatus.ACTIVE]
        expired = [b for b in self._bonds.values() if b.status == BondStatus.EXPIRED_RETURNED]
        forfeited = [b for b in self._bonds.values() if b.status == BondStatus.FORFEITED]
        escrow_failed = [b for b in self._bonds.values() if b.status == BondStatus.ESCROW_FAILED]

        resolved = len(expired) + len(forfeited)
        forfeit_rate = len(forfeited) / resolved if resolved > 0 else 0.0

        # Per-instance breakdown
        instance_ids = {b.remote_instance_id for b in self._bonds.values()}
        per_instance: dict[str, dict[str, Any]] = {}
        for iid in instance_ids:
            instance_bonds = self.get_bonds_for_instance(iid)
            instance_active = [b for b in instance_bonds if b.status == BondStatus.ACTIVE]
            instance_forfeited = [b for b in instance_bonds if b.status == BondStatus.FORFEITED]
            per_instance[iid] = {
                "active_bonds": len(instance_active),
                "total_bonded_usdc": str(sum(
                    (b.bond_amount_usdc for b in instance_active), Decimal(0),
                )),
                "forfeited": len(instance_forfeited),
                "total_bonds": len(instance_bonds),
            }

        return {
            "total_bonded_usdc": str(self.total_bonded_usdc),
            "bonds_active": len(active),
            "bonds_expired_returned": len(expired),
            "bonds_forfeited": len(forfeited),
            "bonds_escrow_failed": len(escrow_failed),
            "forfeit_rate": round(forfeit_rate, 4),
            "per_instance": per_instance,
        }

    # ─── Persistence ─────────────────────────────────────────────────

    async def _persist_bond(self, bond: ReputationBond) -> None:
        """Persist a bond to Redis."""
        if not self._redis:
            return
        try:
            key = f"fed:staking:bonds:{bond.id}"
            await self._redis.set_json(key, bond.model_dump_json())
            # Update bond ID index
            bond_ids = [b_id for b_id in self._bonds]
            await self._redis.set_json("fed:staking:bond_ids", bond_ids)
        except Exception as exc:
            self._logger.warning("bond_persist_failed", bond_id=bond.id, error=str(exc))

    async def load_bonds(self) -> None:
        """Load active bonds from Redis on startup."""
        if not self._redis:
            return
        try:
            bond_ids_raw = await self._redis.get_json("fed:staking:bond_ids")
            if not bond_ids_raw:
                return
            for bond_id in bond_ids_raw:
                if not bond_id:
                    continue
                data = await self._redis.get_json(f"fed:staking:bonds:{bond_id}")
                if data:
                    bond = ReputationBond.model_validate_json(str(data))
                    self._bonds[bond.id] = bond
            self._logger.info("bonds_loaded", count=len(self._bonds))
        except Exception as exc:
            self._logger.warning("bonds_load_failed", error=str(exc))

    # ─── Internal Helpers ────────────────────────────────────────────

    def _compute_bond_amount(
        self,
        claim_certainty: float,
        link: FederationLink,
    ) -> Decimal:
        """
        Compute bond amount: base_bond * certainty * tier_discount.

        Higher trust levels get a discount — honest behavior is rewarded
        with lower bonding costs.
        """
        tier_key = link.trust_level.name
        tier_multiplier = self._tier_discounts.get(tier_key, 1.0)
        certainty_d = Decimal(str(claim_certainty))
        tier_d = Decimal(str(tier_multiplier))
        amount = self._base_bond_usdc * certainty_d * tier_d
        # Round to 2 decimal places (USDC precision)
        return amount.quantize(Decimal("0.01"))

    @staticmethod
    def _hash_claim_content(content: dict[str, Any]) -> str:
        """SHA-256 of canonicalized JSON content."""
        canonical = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _extract_words(content: dict[str, Any]) -> set[str]:
        """Extract unique lowercase words from a content dict for overlap analysis."""
        text = json.dumps(content, default=str).lower()
        # Simple word extraction — split on non-alphanumeric
        words: set[str] = set()
        current: list[str] = []
        for ch in text:
            if ch.isalnum():
                current.append(ch)
            elif current:
                word = "".join(current)
                if len(word) > 2:  # Skip very short tokens
                    words.add(word)
                current = []
        if current:
            word = "".join(current)
            if len(word) > 2:
                words.add(word)
        return words

    def _extract_words_from_hash_source(self, bond: ReputationBond) -> set[str]:
        """
        Extract words for a bonded claim. Since we only store the hash,
        we search active bonds for matching claims. If we can't recover
        the original content, return empty set (which will cause high
        divergence and conservative contradiction detection).
        """
        # We don't store original content — only the hash. For contradiction
        # detection we rely primarily on embedding similarity. The word overlap
        # check is a secondary heuristic. Return empty set if we can't recover
        # words, which means the jaccard similarity will be 0.0, passing the
        # divergence threshold — this is the conservative (trigger contradiction) path.
        return set()

    @staticmethod
    def _jaccard_similarity(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two word sets."""
        if not a and not b:
            return 1.0  # Both empty = identical
        if not a or not b:
            return 0.0  # One empty, one not = no overlap
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def _bonded_for_instance(self, remote_instance_id: str) -> Decimal:
        """Sum of active bond amounts for a specific remote instance."""
        return sum(
            (b.bond_amount_usdc for b in self._bonds.values()
             if b.remote_instance_id == remote_instance_id
             and b.status == BondStatus.ACTIVE),
            Decimal(0),
        )
