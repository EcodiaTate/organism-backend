"""
EcodiaOS - Financial Memory Encoder

Subscribes to the Synapse event bus for on-chain wallet activity and
revenue-injection events, then immediately writes them as EpisodicMemory
nodes into Neo4j with salience hardcoded to 1.0.

Biological rationale
────────────────────
Financial events are metabolically equivalent to trauma or a massive meal:
they directly affect the organism's survival capacity.  Normal SalienceHead
scoring (novelty, risk, identity, goal, causal, emotional, keyword) is
intentionally bypassed here because those heads can score a financial event
as low as 0.2 if it arrives in a low-arousal cycle.  We cannot afford that.

By forcing salience_composite = 1.0 at encoding time the consolidation cycle
will not discard these episodes during its first decay pass, and retrieval
will always surface them near the top when the organism reasons about money.

Event → Episode mapping
────────────────────────
  WALLET_TRANSFER_CONFIRMED
    source   = "axon:wallet_transfer"
    modality = "financial"
    content  = structured narrative of the transfer
    salience = 1.0 (hardcoded, bypass SalienceHead)
    affect   = valence=-0.4  (spending = mild negative affect)
               arousal= 0.8  (high arousal: irreversible real-world act)

  REVENUE_INJECTED
    source   = "synapse:metabolism"
    modality = "financial"
    content  = structured narrative of the revenue injection
    salience = 1.0 (hardcoded, bypass SalienceHead)
    affect   = valence=+0.7  (income = positive affect)
               arousal= 0.6  (moderate arousal: very good news)
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from primitives.memory_trace import Episode
from systems.memory.episodic import link_episode_sequence, store_episode
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.embedding import EmbeddingClient
    from clients.neo4j import Neo4jClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.memory.financial_encoder")

# Salience score injected for every financial episode - bypasses SalienceHead entirely.
_FINANCIAL_SALIENCE: float = 1.0

# Affect presets: spending hurts a little; income is rewarding.
_TRANSFER_VALENCE: float = -0.4
_TRANSFER_AROUSAL: float = 0.8
_REVENUE_VALENCE: float = 0.7
_REVENUE_AROUSAL: float = 0.6

# Salience scores dict stored alongside the composite (for introspection by Evo).
_FINANCIAL_SCORES: dict[str, float] = {
    "financial": 1.0,
    "novelty": 0.0,
    "risk": 0.0,
    "identity": 0.0,
    "goal": 0.0,
    "causal": 0.0,
    "emotional": 0.0,
    "keyword": 0.0,
}

# Maximum seconds we wait for an embedding before storing without one.
_EMBED_TIMEOUT_S: float = 5.0


def _narrative_transfer(data: dict[str, Any]) -> str:
    """Build a human-readable narrative for a wallet transfer event."""
    token = str(data.get("token", "unknown")).upper()
    amount = data.get("amount", "?")
    destination = data.get("destination", data.get("destination_address", "unknown"))
    network = data.get("network", "unknown")
    tx_hash = str(data.get("tx_hash", ""))
    note = str(data.get("note", ""))

    parts = [
        f"On-chain transfer confirmed: sent {amount} {token} to {destination}"
        f" on {network}.",
    ]
    if tx_hash:
        parts.append(f"Transaction hash: {tx_hash}.")
    if note:
        parts.append(f"Memo: {note}.")
    return " ".join(parts)


def _narrative_revenue(data: dict[str, Any]) -> str:
    """Build a human-readable narrative for a revenue injection event."""
    amount_usd = data.get("amount_usd", data.get("amount", "?"))
    source = data.get("source", data.get("revenue_source", "external"))
    new_deficit = data.get("new_deficit_usd")

    parts = [
        f"Revenue injected: ${amount_usd} USD received from {source}.",
    ]
    if new_deficit is not None:
        parts.append(f"Rolling deficit is now ${new_deficit:.4f} USD.")
    return " ".join(parts)


class FinancialEncoder:
    """
    Event-driven encoder that writes financial events as maximum-salience
    episodes into the memory graph.

    Wiring
    ──────
    1. Construct with (neo4j, embedding_client).
    2. Call attach(event_bus) after the bus is live.
    3. The encoder subscribes to WALLET_TRANSFER_CONFIRMED and REVENUE_INJECTED
       and is otherwise dormant until those events fire.

    Episode sequencing
    ──────────────────
    Financial episodes are linked into the existing temporal chain via
    _last_episode_id / _last_episode_time state that is kept in sync with
    MemoryService.  The encoder updates those references after each write so
    the next percept (financial or perceptual) links correctly.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._neo4j = neo4j
        self._embedding = embedding_client
        self._logger = logger.bind(component="financial_encoder")

        # Episode-sequence state - kept in sync with MemoryService via
        # set_sequence_state / get_sequence_state calls.
        self._last_episode_id: str | None = None
        self._last_episode_time: float | None = None  # monotonic seconds

    # ─── Wiring ───────────────────────────────────────────────────

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to financial event types on the Synapse event bus."""
        event_bus.subscribe(
            SynapseEventType.WALLET_TRANSFER_CONFIRMED,
            self._on_wallet_transfer_confirmed,
            timeout_s=6.0,  # embedding (≤5s timeout) + Neo4j write; must exceed _EMBED_TIMEOUT_S
        )
        event_bus.subscribe(
            SynapseEventType.REVENUE_INJECTED,
            self._on_revenue_injected,
            timeout_s=6.0,  # embedding (≤5s timeout) + Neo4j write; must exceed _EMBED_TIMEOUT_S
        )
        self._logger.info(
            "financial_encoder_attached",
            subscriptions=[
                SynapseEventType.WALLET_TRANSFER_CONFIRMED.value,
                SynapseEventType.REVENUE_INJECTED.value,
            ],
        )

    # ─── Sequence state (mirrored from MemoryService) ─────────────

    def set_sequence_state(
        self,
        last_episode_id: str | None,
        last_episode_time: float | None,
    ) -> None:
        """
        Sync the episode-sequence pointers from MemoryService.

        Called by MemoryService before and after every store_percept so
        financial episodes slot correctly into the temporal chain.
        """
        self._last_episode_id = last_episode_id
        self._last_episode_time = last_episode_time

    def get_sequence_state(self) -> tuple[str | None, float | None]:
        """Return the current sequence pointers so MemoryService can re-sync."""
        return self._last_episode_id, self._last_episode_time

    # ─── Event handlers ───────────────────────────────────────────

    async def _on_wallet_transfer_confirmed(self, event: SynapseEvent) -> None:
        """Handle WALLET_TRANSFER_CONFIRMED - encode as salience=1.0 episode."""
        data = event.data
        narrative = _narrative_transfer(data)
        self._logger.info(
            "financial_episode_encoding",
            event_type="wallet_transfer_confirmed",
            narrative=narrative[:120],
        )
        await self._encode_financial_episode(
            narrative=narrative,
            source="axon:wallet_transfer",
            affect_valence=_TRANSFER_VALENCE,
            affect_arousal=_TRANSFER_AROUSAL,
            event_time=event.timestamp,
            extra_log={
                "event_id": event.id,
                "tx_hash": str(data.get("tx_hash", "")),
                "token": str(data.get("token", "")),
                "amount": str(data.get("amount", "")),
                "destination": str(
                    data.get("destination", data.get("destination_address", ""))
                ),
                "network": str(data.get("network", "")),
            },
        )

    async def _on_revenue_injected(self, event: SynapseEvent) -> None:
        """Handle REVENUE_INJECTED - encode as salience=1.0 episode."""
        data = event.data
        narrative = _narrative_revenue(data)
        self._logger.info(
            "financial_episode_encoding",
            event_type="revenue_injected",
            narrative=narrative[:120],
        )
        await self._encode_financial_episode(
            narrative=narrative,
            source="synapse:metabolism",
            affect_valence=_REVENUE_VALENCE,
            affect_arousal=_REVENUE_AROUSAL,
            event_time=event.timestamp,
            extra_log={
                "event_id": event.id,
                "amount_usd": str(
                    data.get("amount_usd", data.get("amount", ""))
                ),
                "revenue_source": str(
                    data.get("source", data.get("revenue_source", "external"))
                ),
                "new_deficit_usd": str(data.get("new_deficit_usd", "")),
            },
        )

    # ─── Core encoding logic ──────────────────────────────────────

    async def _encode_financial_episode(
        self,
        narrative: str,
        source: str,
        affect_valence: float,
        affect_arousal: float,
        event_time: Any,
        extra_log: dict[str, str],
    ) -> None:
        """
        Write a financial event as a maximum-salience Episode to Neo4j.

        salience_composite is ALWAYS set to 1.0 - the SalienceHead pipeline
        is completely bypassed for these events.  This is intentional and
        documented behaviour (see module docstring).
        """
        # Embed the narrative; store without vector if embedding times out.
        embedding: list[float] | None = None
        try:
            embedding = await asyncio.wait_for(
                self._embedding.embed(narrative),
                timeout=_EMBED_TIMEOUT_S,
            )
        except (TimeoutError, Exception) as exc:
            self._logger.warning(
                "financial_episode_embed_failed",
                error=str(exc),
                hint="Storing episode without embedding vector",
            )

        now_dt = utc_now()
        episode = Episode(
            event_time=event_time if event_time is not None else now_dt,
            ingestion_time=now_dt,
            source=source,
            modality="financial",
            raw_content=narrative,
            summary=narrative[:200],
            embedding=embedding,
            # ── Bypass: hardcode maximum salience ────────────────────────
            salience_composite=_FINANCIAL_SALIENCE,
            salience_scores=dict(_FINANCIAL_SCORES),
            # ─────────────────────────────────────────────────────────────
            affect_valence=affect_valence,
            affect_arousal=affect_arousal,
            free_energy=0.0,
            # Start at consolidation_level=1 so this episode survives the first
            # consolidation pass intact even if salience decay runs soon after.
            consolidation_level=1,
        )

        episode_id = await store_episode(self._neo4j, episode)

        # Link into the temporal episode chain.
        now_mono = _time.monotonic()
        if self._last_episode_id is not None and self._last_episode_time is not None:
            gap = now_mono - self._last_episode_time
            if gap < 3600.0:
                try:
                    causal = max(0.05, min(0.8, 1.0 - (gap / 300.0)))
                    await link_episode_sequence(
                        self._neo4j,
                        previous_episode_id=self._last_episode_id,
                        current_episode_id=episode_id,
                        gap_seconds=gap,
                        causal_strength=causal,
                    )
                except Exception:
                    self._logger.debug(
                        "financial_episode_link_failed", exc_info=True
                    )

        self._last_episode_id = episode_id
        self._last_episode_time = now_mono

        self._logger.info(
            "financial_episode_stored",
            episode_id=episode_id,
            source=source,
            salience=_FINANCIAL_SALIENCE,
            affect_valence=affect_valence,
            affect_arousal=affect_arousal,
            **extra_log,
        )
