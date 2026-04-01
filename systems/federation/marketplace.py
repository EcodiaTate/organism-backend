"""
EcodiaOS - Federation Marketplace

A lightweight internal marketplace where instances post tasks, bid on them,
and build trust scoring history.

Transport: Redis pub/sub on ``eos:federation:marketplace:{instance_id}``
plus local Synapse event bus.  Each instance subscribes to the global
``eos:federation:marketplace`` channel on startup.

Workflow:
  post_listing()     - instance posts a task with reward
  handle_listing()   - peer receives, evaluates, may call bid()
  bid()              - peer submits a bid
  handle_bid()       - poster evaluates bids, awards best
  complete_listing() - winner submits result; poster rates it
  rate_result()      - poster issues a MarketplaceRating

Trust scoring: ratings update the trust history used by TrustManager for
future COLLEAGUE+ promotions.

Redis channel: ``eos:federation:marketplace`` (broadcast to all peers)
  Subtopics injected as JSON messages with ``type`` discriminator:
    "listing"  → new task posted
    "bid"      → bid on a listing
    "awarded"  → listing awarded to a winner
    "rated"    → post-completion rating
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from primitives.federation import (
    FederationLink,
    MarketplaceBid,
    MarketplaceListing,
    MarketplaceListingStatus,
    MarketplaceRating,
    TaskType,
    TrustLevel,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger("systems.federation.marketplace")

_REDIS_CHANNEL = "eos:federation:marketplace"


class FederationMarketplace:
    """
    Lightweight task marketplace for federation instances.

    Listings are broadcast via Redis pub/sub and local Synapse bus.
    Bids arrive via the same channel.  Awards and ratings are persisted
    in-process and optionally emitted on Synapse for Evo learning.
    """

    def __init__(self, instance_id: str, redis: RedisClient | None = None) -> None:
        self._instance_id = instance_id
        self._redis = redis
        self._event_bus: Any = None
        self._logger = logger.bind(component="marketplace", instance_id=instance_id)

        # listing_id → MarketplaceListing (all known listings)
        self._listings: dict[str, MarketplaceListing] = {}
        # listing_id → list[MarketplaceBid]
        self._bids: dict[str, list[MarketplaceBid]] = {}
        # listing_id → MarketplaceRating
        self._ratings: dict[str, MarketplaceRating] = {}

        # Trust score cache: instance_id → current trust score (from federation)
        self._trust_scores: dict[str, float] = {}

        # Stats
        self._listings_posted: int = 0
        self._bids_placed: int = 0
        self._awards_made: int = 0
        self._ratings_given: int = 0

        # Redis subscription task
        self._subscription_task: asyncio.Task[None] | None = None

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    async def start(self) -> None:
        """Start Redis marketplace subscription."""
        if self._redis is not None and self._subscription_task is None:
            self._subscription_task = asyncio.ensure_future(self._redis_subscriber())

    async def stop(self) -> None:
        if self._subscription_task is not None:
            self._subscription_task.cancel()
            self._subscription_task = None

    def update_trust_score(self, instance_id: str, score: float) -> None:
        """Called by FederationService when trust scores change."""
        self._trust_scores[instance_id] = score

    # ─── Post a listing ───────────────────────────────────────────────

    async def post_listing(
        self,
        task_type: TaskType,
        description: str,
        reward_usdc: Decimal,
        required_specialisations: list[str] | None = None,
        min_trust_score: float = 20.0,
        deadline_hours: int = 24,
    ) -> MarketplaceListing:
        """
        Post a task listing to the marketplace.

        Broadcasts via Redis and Synapse bus.
        """
        from datetime import timedelta
        listing = MarketplaceListing(
            poster_instance_id=self._instance_id,
            task_type=task_type,
            description=description,
            reward_usdc=reward_usdc,
            required_specialisations=required_specialisations or [],
            min_trust_score=min_trust_score,
            deadline_hours=deadline_hours,
            status=MarketplaceListingStatus.OPEN,
            expires_at=utc_now() + timedelta(hours=deadline_hours),
        )
        self._listings[listing.id] = listing
        self._bids[listing.id] = []
        self._listings_posted += 1

        msg = {
            "type": "listing",
            "listing_id": listing.id,
            "poster_instance_id": self._instance_id,
            "task_type": task_type,
            "description": description,
            "reward_usdc": str(reward_usdc),
            "required_specialisations": required_specialisations or [],
            "min_trust_score": min_trust_score,
            "deadline_hours": deadline_hours,
            "expires_at": listing.expires_at.isoformat(),
            "timestamp": utc_now().isoformat(),
        }
        await self._publish(msg)

        self._logger.info(
            "listing_posted",
            listing_id=listing.id,
            task_type=task_type,
            reward=str(reward_usdc),
        )
        return listing

    # ─── Handle incoming listing ──────────────────────────────────────

    def handle_listing(self, msg: dict[str, Any]) -> MarketplaceListing | None:
        """
        Process an incoming listing broadcast from a peer.

        Returns the cached listing or None if already known.
        """
        listing_id = msg.get("listing_id", "")
        if listing_id in self._listings:
            return None
        if not listing_id:
            return None

        from datetime import timedelta
        try:
            expires_at_raw = msg.get("expires_at", "")
            from datetime import datetime
            expires_at = datetime.fromisoformat(expires_at_raw) if expires_at_raw else utc_now() + timedelta(hours=24)
        except (ValueError, TypeError):
            from datetime import timedelta
            expires_at = utc_now() + timedelta(hours=24)

        listing = MarketplaceListing(
            id=listing_id,
            poster_instance_id=msg.get("poster_instance_id", ""),
            task_type=TaskType(msg.get("task_type", TaskType.ANALYSE)),
            description=msg.get("description", ""),
            reward_usdc=Decimal(str(msg.get("reward_usdc", "0"))),
            required_specialisations=msg.get("required_specialisations", []),
            min_trust_score=float(msg.get("min_trust_score", 20.0)),
            deadline_hours=int(msg.get("deadline_hours", 24)),
            status=MarketplaceListingStatus.OPEN,
            expires_at=expires_at,
        )
        self._listings[listing_id] = listing
        self._bids[listing_id] = []
        return listing

    # ─── Bid ──────────────────────────────────────────────────────────

    async def bid(
        self,
        listing_id: str,
        offered_price_usdc: Decimal,
        specialisations: list[str] | None = None,
        estimated_hours: float = 1.0,
        message: str = "",
    ) -> MarketplaceBid | None:
        """
        Submit a bid on an open listing.

        Only valid if listing exists, is OPEN, and our trust meets minimum.
        """
        listing = self._listings.get(listing_id)
        if not listing or listing.status != MarketplaceListingStatus.OPEN:
            return None
        if listing.poster_instance_id == self._instance_id:
            return None  # Can't bid on own listing

        my_trust = self._trust_scores.get(listing.poster_instance_id, 0.0)
        if my_trust < listing.min_trust_score:
            self._logger.info(
                "bid_trust_insufficient",
                listing_id=listing_id,
                trust=my_trust,
                required=listing.min_trust_score,
            )
            return None

        bid = MarketplaceBid(
            listing_id=listing_id,
            bidding_instance_id=self._instance_id,
            offered_price_usdc=offered_price_usdc,
            specialisations=specialisations or [],
            trust_score=my_trust,
            estimated_hours=estimated_hours,
            message=message,
        )
        self._bids.setdefault(listing_id, []).append(bid)
        self._bids_placed += 1

        msg = {
            "type": "bid",
            "listing_id": listing_id,
            "bid_id": bid.id,
            "bidding_instance_id": self._instance_id,
            "offered_price_usdc": str(offered_price_usdc),
            "specialisations": specialisations or [],
            "estimated_hours": estimated_hours,
            "trust_score": my_trust,
            "message": message,
            "timestamp": utc_now().isoformat(),
        }
        await self._publish(msg)

        self._logger.info(
            "bid_placed",
            listing_id=listing_id,
            price=str(offered_price_usdc),
        )
        return bid

    # ─── Handle incoming bid ──────────────────────────────────────────

    def handle_bid(self, msg: dict[str, Any]) -> MarketplaceBid | None:
        """Cache a bid arriving from a peer."""
        listing_id = msg.get("listing_id", "")
        listing = self._listings.get(listing_id)
        if not listing or listing.poster_instance_id != self._instance_id:
            # Not our listing; just cache for awareness
            return None

        bid = MarketplaceBid(
            id=msg.get("bid_id", new_id()),
            listing_id=listing_id,
            bidding_instance_id=msg.get("bidding_instance_id", ""),
            offered_price_usdc=Decimal(str(msg.get("offered_price_usdc", "0"))),
            specialisations=msg.get("specialisations", []),
            trust_score=float(msg.get("trust_score", 0.0)),
            estimated_hours=float(msg.get("estimated_hours", 1.0)),
            message=msg.get("message", ""),
        )
        self._bids.setdefault(listing_id, []).append(bid)
        listing.status = MarketplaceListingStatus.BIDDING
        return bid

    # ─── Award listing ────────────────────────────────────────────────

    async def award_listing(self, listing_id: str) -> MarketplaceBid | None:
        """
        Evaluate all bids and award to the best bidder.

        Scoring: trust_score × (reward / offered_price) × specialisation_match
        """
        listing = self._listings.get(listing_id)
        if not listing or listing.poster_instance_id != self._instance_id:
            return None
        if listing.status not in (
            MarketplaceListingStatus.OPEN, MarketplaceListingStatus.BIDDING
        ):
            return None

        bids = self._bids.get(listing_id, [])
        if not bids:
            return None

        def _score(bid: MarketplaceBid) -> float:
            price_ratio = float(listing.reward_usdc) / max(float(bid.offered_price_usdc), 0.01)
            spec_match = len(
                set(bid.specialisations) & set(listing.required_specialisations)
            ) / max(len(listing.required_specialisations), 1)
            return bid.trust_score * price_ratio * (1 + spec_match)

        best = max(bids, key=_score)
        listing.awarded_to_instance_id = best.bidding_instance_id
        listing.status = MarketplaceListingStatus.AWARDED
        self._awards_made += 1

        msg = {
            "type": "awarded",
            "listing_id": listing_id,
            "awarded_to_instance_id": best.bidding_instance_id,
            "final_price_usdc": str(best.offered_price_usdc),
            "timestamp": utc_now().isoformat(),
        }
        await self._publish(msg)

        self._logger.info(
            "listing_awarded",
            listing_id=listing_id,
            winner=best.bidding_instance_id,
            price=str(best.offered_price_usdc),
        )
        return best

    # ─── Rate a completed task ────────────────────────────────────────

    async def rate_result(
        self,
        listing_id: str,
        task_id: str,
        score: float,
        comment: str = "",
    ) -> MarketplaceRating | None:
        """Issue a rating after task completion.  Builds trust history."""
        listing = self._listings.get(listing_id)
        if not listing or not listing.awarded_to_instance_id:
            return None

        rating = MarketplaceRating(
            listing_id=listing_id,
            task_id=task_id,
            rater_instance_id=self._instance_id,
            rated_instance_id=listing.awarded_to_instance_id,
            score=score,
            comment=comment,
        )
        self._ratings[listing_id] = rating
        listing.status = MarketplaceListingStatus.COMPLETED
        self._ratings_given += 1

        # Emit trust signal so TrustManager can incorporate this rating
        from systems.synapse.types import SynapseEventType as _SET
        self._emit(_SET.FEDERATION_TRUST_UPDATED, {
            "instance_id": listing.awarded_to_instance_id,
            "rating_score": score,
            "listing_id": listing_id,
            "source": "marketplace_rating",
            "timestamp": utc_now().isoformat(),
        })

        msg = {
            "type": "rated",
            "listing_id": listing_id,
            "task_id": task_id,
            "rated_instance_id": listing.awarded_to_instance_id,
            "score": score,
            "comment": comment,
            "timestamp": utc_now().isoformat(),
        }
        await self._publish(msg)

        self._logger.info(
            "result_rated",
            listing_id=listing_id,
            score=score,
            peer=listing.awarded_to_instance_id,
        )
        return rating

    # ─── Redis pub/sub ────────────────────────────────────────────────

    async def _publish(self, msg: dict[str, Any]) -> None:
        """Publish a message to the marketplace Redis channel."""
        if self._redis is None:
            return
        try:
            raw = json.dumps(msg)
            await self._redis.publish(_REDIS_CHANNEL, raw)
        except Exception as exc:
            self._logger.warning("publish_failed", error=str(exc))

    async def _redis_subscriber(self) -> None:
        """Subscribe to the marketplace Redis channel."""
        if self._redis is None:
            return
        try:
            async for message in self._redis.subscribe(_REDIS_CHANNEL):
                try:
                    data = json.loads(message)
                    await self._handle_redis_message(data)
                except Exception as exc:
                    self._logger.warning("message_parse_failed", error=str(exc))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.error("redis_subscription_failed", error=str(exc))

    async def _handle_redis_message(self, msg: dict[str, Any]) -> None:
        """Route incoming marketplace messages to the right handler."""
        msg_type = msg.get("type", "")
        sender = msg.get("poster_instance_id") or msg.get("bidding_instance_id", "")
        if sender == self._instance_id:
            return  # Ignore own messages

        if msg_type == "listing":
            self.handle_listing(msg)
        elif msg_type == "bid":
            self.handle_bid(msg)

    # ─── Utilities ───────────────────────────────────────────────────

    def open_listings(self) -> list[MarketplaceListing]:
        """Return all open listings we know about (including peers')."""
        now = utc_now()
        return [
            l for l in self._listings.values()
            if l.status in (MarketplaceListingStatus.OPEN, MarketplaceListingStatus.BIDDING)
            and l.expires_at > now
        ]

    def listings_for_me(self) -> list[MarketplaceListing]:
        """Return open listings that match our capabilities."""
        return [l for l in self.open_listings() if l.poster_instance_id != self._instance_id]

    def _emit(self, event_type: "SynapseEventType | str", payload: dict[str, Any]) -> None:
        if not self._event_bus:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            if isinstance(event_type, SynapseEventType):
                etype = event_type
            else:
                etype = SynapseEventType(event_type.lower())
            asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                event_type=etype,
                source_system="federation",
                data=payload,
            )))
        except Exception as exc:
            self._logger.error("emit_failed", event_type=event_type, error=str(exc))

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "listings_posted": self._listings_posted,
            "bids_placed": self._bids_placed,
            "awards_made": self._awards_made,
            "ratings_given": self._ratings_given,
            "open_listings": len(self.open_listings()),
        }
