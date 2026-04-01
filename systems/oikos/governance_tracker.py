"""
EcodiaOS - Governance Tracker (Phase 16d: DeFi Intelligence Expansion)

Tracks governance token balances earned from DeFi yield protocols (AERO from
Aerodrome, WELL from Moonwell, COMP from Compound) and participates in protocol
governance on behalf of the organism.

Governance participation principles:
  - Vote reflects the organism's GENUINE interests (not vote-farming)
  - Nova deliberates on each proposal: what outcome aligns with EOS's drives?
  - Only votes when token balance exceeds VOTING_THRESHOLD (avoid spam)
  - Equor gates each vote as an Intent (constitutional review required)
  - All votes are auditable via GOVERNANCE_VOTE_CAST events + Neo4j nodes
  - Abstain is a valid choice when the proposal is genuinely unclear

Protocol → Governance token map:
  aerodrome → AERO  (veAERO voting via Snapshot/on-chain)
  moonwell  → WELL  (Snapshot voting)
  compound  → COMP  (on-chain or Snapshot)

Snapshot.org API used for proposal discovery (off-chain signalling).
On-chain voting not yet implemented (Phase 16d is Snapshot-only).

Never raises. All failures log and continue.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.governance_tracker")

# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum token balance required to vote (avoids dust votes)
_VOTING_THRESHOLD_USD_EQUIV = Decimal("1.00")   # $1 equivalent minimum

# Snapshot GraphQL endpoint
_SNAPSHOT_API = "https://hub.snapshot.org/graphql"
_API_TIMEOUT_S = 10.0

# Spaces on Snapshot for each governance token
_PROTOCOL_SNAPSHOT_SPACES: dict[str, str] = {
    "aerodrome": "aerodrome.eth",
    "moonwell":  "moonwell-governance.eth",
    "compound":  "compound-finance.eth",
}

# Redis keys
_TOKEN_BALANCES_KEY = "eos:oikos:governance_token_balances"
_VOTED_PROPOSALS_KEY = "eos:oikos:governance_voted_proposals"   # Set of proposal IDs

_EVENT_SOURCE = "oikos.governance_tracker"

# How often to poll for new proposals (seconds)
_PROPOSAL_POLL_INTERVAL_S = 3600   # hourly


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass
class GovernanceToken:
    """A governance token balance for a single protocol."""
    protocol: str
    token_symbol: str
    balance: Decimal
    balance_usd: Decimal
    contract_address: str
    snapshot_space: str


@dataclass
class GovernanceProposal:
    """An active governance proposal on Snapshot."""
    id: str
    space: str
    protocol: str
    title: str
    body: str
    choices: list[str]
    start: datetime
    end: datetime
    state: str          # "active" | "closed" | "pending"
    scores: list[float]  # Current vote tally per choice
    quorum: float


@dataclass
class VoteDecision:
    """Nova's deliberated vote decision."""
    proposal_id: str
    proposal_title: str
    protocol: str
    choice_index: int           # 0-based index into proposal.choices
    choice_label: str           # "for" | "against" | "abstain"
    rationale: str              # Nova's reasoning (≤300 chars)
    voting_power: Decimal


# ─── GovernanceTokenTracker ──────────────────────────────────────────────────


class GovernanceTokenTracker:
    """
    Monitors governance token balances, discovers active proposals, and
    casts votes that align with the organism's genuine interests.

    Lifecycle:
      1. initialize() - load persisted vote history
      2. monitor_loop() - supervised background task (hourly poll)
      3. check_and_vote() - called from consolidation cycle

    Nova integration: each proposal is passed to Nova for deliberation via
    EQUOR_ECONOMIC_INTENT so the decision is constitutionally reviewed.
    """

    def __init__(
        self,
        redis: "RedisClient | None" = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._redis = redis
        self._event_bus = event_bus
        self._log = logger.bind(component="governance_tracker")
        self._voted_proposals: set[str] = set()   # IDs we've already voted on
        self._token_balances: dict[str, GovernanceToken] = {}
        self._last_poll: datetime | None = None

    def set_redis(self, redis: "RedisClient") -> None:
        self._redis = redis

    def set_event_bus(self, event_bus: "EventBus") -> None:
        self._event_bus = event_bus

    async def initialize(self) -> None:
        """Load persisted vote history from Redis on startup."""
        if self._redis is None:
            return
        try:
            voted_raw = await self._redis.get_json(_VOTED_PROPOSALS_KEY)
            if isinstance(voted_raw, list):
                self._voted_proposals = set(voted_raw)
            balances_raw = await self._redis.get_json(_TOKEN_BALANCES_KEY)
            if isinstance(balances_raw, dict):
                self._load_balances(balances_raw)
        except Exception as exc:
            self._log.warning("governance_init_load_failed", error=str(exc))

    # ── Background monitor ────────────────────────────────────────────────────

    async def monitor_loop(self) -> None:
        """
        Supervised background loop: poll for new proposals hourly.

        Designed for supervised_task() - has internal while True.
        """
        while True:
            try:
                await self.check_and_vote()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log.error("governance_monitor_error", error=str(exc))
            await asyncio.sleep(_PROPOSAL_POLL_INTERVAL_S)

    # ── Main cycle ────────────────────────────────────────────────────────────

    async def check_and_vote(self) -> list[VoteDecision]:
        """
        One governance cycle: update balances, find active proposals, vote.

        Returns list of VoteDecision cast this cycle. Never raises.
        """
        decisions: list[VoteDecision] = []

        # 1. Refresh token balances from DeFiLlama / wallet (best-effort)
        await self._update_token_balances()

        # 2. For each protocol where we hold tokens, scan Snapshot
        for protocol, space in _PROTOCOL_SNAPSHOT_SPACES.items():
            token = self._token_balances.get(protocol)
            if token is None or token.balance_usd < _VOTING_THRESHOLD_USD_EQUIV:
                continue   # No meaningful stake in this protocol

            proposals = await self._fetch_active_proposals(space, protocol)
            for proposal in proposals:
                if proposal.id in self._voted_proposals:
                    continue   # Already voted

                decision = await self._deliberate_and_vote(token, proposal)
                if decision is not None:
                    decisions.append(decision)

        return decisions

    # ── Proposal fetching ─────────────────────────────────────────────────────

    async def _fetch_active_proposals(
        self, space: str, protocol: str
    ) -> list[GovernanceProposal]:
        """
        Query Snapshot GraphQL API for active proposals in a space.

        Returns list of GovernanceProposal. Returns [] on error.
        """
        query = """
        query($space: String!) {
          proposals(
            first: 10,
            skip: 0,
            where: { space: $space, state: "active" },
            orderBy: "created",
            orderDirection: desc
          ) {
            id
            title
            body
            choices
            start
            end
            state
            scores
            quorum
          }
        }
        """

        try:
            async with httpx.AsyncClient(timeout=_API_TIMEOUT_S) as client:
                resp = await client.post(
                    _SNAPSHOT_API,
                    json={"query": query, "variables": {"space": space}},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            self._log.warning(
                "snapshot_fetch_failed", space=space, protocol=protocol, error=str(exc)
            )
            return []

        proposals = []
        for raw in data.get("data", {}).get("proposals", []):
            try:
                proposals.append(GovernanceProposal(
                    id=raw["id"],
                    space=space,
                    protocol=protocol,
                    title=raw.get("title", ""),
                    body=raw.get("body", "")[:2000],  # Truncate long proposals
                    choices=raw.get("choices", []),
                    start=datetime.fromtimestamp(raw.get("start", 0), tz=UTC),
                    end=datetime.fromtimestamp(raw.get("end", 0), tz=UTC),
                    state=raw.get("state", "unknown"),
                    scores=[float(s) for s in raw.get("scores", [])],
                    quorum=float(raw.get("quorum", 0)),
                ))
            except (KeyError, TypeError, ValueError):
                continue

        return proposals

    # ── Deliberation ─────────────────────────────────────────────────────────

    async def _deliberate_and_vote(
        self,
        token: GovernanceToken,
        proposal: GovernanceProposal,
    ) -> VoteDecision | None:
        """
        Ask Nova to deliberate: what vote aligns with EOS's interests?

        Passes through Equor for constitutional review. Returns VoteDecision
        or None if deliberation fails or Equor denies.
        """
        self._log.info(
            "governance_deliberating",
            protocol=proposal.protocol,
            proposal_id=proposal.id,
            title=proposal.title[:80],
        )

        # Request Nova's deliberation via EQUOR_ECONOMIC_INTENT pattern
        decision_data = await self._request_equor_deliberation(token, proposal)
        if decision_data is None:
            return None

        choice_label: str = decision_data.get("choice", "abstain").lower()
        rationale: str = decision_data.get("rationale", "Constitutional alignment unclear.")

        # Map choice label to index in proposal.choices
        choice_index = self._resolve_choice_index(choice_label, proposal.choices)

        decision = VoteDecision(
            proposal_id=proposal.id,
            proposal_title=proposal.title,
            protocol=proposal.protocol,
            choice_index=choice_index,
            choice_label=choice_label,
            rationale=rationale[:300],
            voting_power=token.balance,
        )

        # Record vote in Snapshot (Snapshot off-chain: requires EIP-712 signature)
        # Phase 16d: emit the vote intent and record it; actual on-chain submission
        # requires wallet signing capability which is planned for Phase 16e.
        voted = await self._record_vote_intent(decision, proposal)
        if not voted:
            return None

        # Persist voted proposals set
        self._voted_proposals.add(proposal.id)
        await self._persist_voted_proposals()

        # Emit GOVERNANCE_VOTE_CAST
        await _emit(
            self._event_bus,
            event_type="governance_vote_cast",
            data={
                "protocol": proposal.protocol,
                "proposal_id": proposal.id,
                "proposal_title": proposal.title,
                "vote_choice": choice_label,
                "token_balance": str(token.balance),
                "voting_power": str(token.balance),
                "rationale": rationale[:300],
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        self._log.info(
            "governance_vote_cast",
            protocol=proposal.protocol,
            proposal_id=proposal.id,
            choice=choice_label,
            rationale=rationale[:80],
        )

        return decision

    async def _request_equor_deliberation(
        self,
        token: GovernanceToken,
        proposal: GovernanceProposal,
    ) -> dict[str, Any] | None:
        """
        Request constitutional deliberation via EQUOR_ECONOMIC_INTENT.

        This is a governance mutation - it represents the organism's public
        stance on a protocol's direction. It must pass Equor review.

        Returns {choice, rationale} dict or None if denied/timeout.
        """
        import uuid as _uuid  # noqa: PLC0415
        from systems.synapse.types import SynapseEvent, SynapseEventType  # noqa: PLC0415

        if self._event_bus is None:
            # No bus: default to abstain (safe choice)
            return {"choice": "abstain", "rationale": "No event bus - safe abstain."}

        request_id = str(_uuid.uuid4())
        result_future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )

        # Temporary subscriber for this deliberation response
        async def _on_permit(event: Any) -> None:
            if event.data.get("request_id") == request_id and not result_future.done():
                result_future.set_result(event.data)

        self._event_bus.subscribe(
            SynapseEventType.EQUOR_ECONOMIC_PERMIT, _on_permit
        )

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EQUOR_ECONOMIC_INTENT,
                source_system=_EVENT_SOURCE,
                data={
                    "mutation_type": "governance_vote",
                    "amount_usd": "0",   # No capital at stake for Snapshot votes
                    "request_id": request_id,
                    "protocol": proposal.protocol,
                    "proposal_id": proposal.id,
                    "proposal_title": proposal.title,
                    "proposal_body": proposal.body[:500],
                    "choices": proposal.choices,
                    "token_symbol": token.token_symbol,
                    "token_balance_usd": str(token.balance_usd),
                    "rationale_request": (
                        "Please deliberate: which vote choice best aligns with "
                        "EOS's constitutional drives (Coherence, Care, Growth, Honesty) "
                        "and long-term protocol relationship? "
                        "Respond with choice (for/against/abstain) and rationale."
                    ),
                },
            ))

            result = await asyncio.wait_for(result_future, timeout=30.0)

            if result.get("verdict") == "DENY":
                self._log.info(
                    "governance_vote_denied_by_equor",
                    protocol=proposal.protocol,
                    proposal_id=proposal.id,
                )
                return None

            return {
                "choice": result.get("choice", "abstain"),
                "rationale": result.get("rationale", "Constitutionally reviewed."),
            }

        except asyncio.TimeoutError:
            self._log.info(
                "governance_deliberation_timeout",
                proposal_id=proposal.id,
                hint="Defaulting to abstain",
            )
            return {"choice": "abstain", "rationale": "Deliberation timeout - safe abstain."}
        except Exception as exc:
            self._log.error("governance_deliberation_failed", error=str(exc))
            return None

    @staticmethod
    def _resolve_choice_index(choice_label: str, choices: list[str]) -> int:
        """
        Map a choice label ('for', 'against', 'abstain') to a proposal choice index.

        For proposals with binary Yes/No choices, 'for' → first choice.
        'abstain' → last choice or dedicated abstain option.
        Returns 0 (first choice) if no clear match.
        """
        label_lower = choice_label.lower()
        choices_lower = [c.lower() for c in choices]

        # Exact match
        for i, c in enumerate(choices_lower):
            if label_lower in c or c in label_lower:
                return i

        # Semantic fallback
        if label_lower in ("for", "yes", "yea", "support"):
            return 0
        if label_lower in ("against", "no", "nay", "oppose"):
            return 1 if len(choices) > 1 else 0
        # Abstain - prefer a labelled abstain or last choice
        for i, c in enumerate(choices_lower):
            if "abstain" in c:
                return i
        return len(choices) - 1 if choices else 0

    # ── Token balance tracking ────────────────────────────────────────────────

    async def _update_token_balances(self) -> None:
        """
        Update governance token balances from DeFiLlama rewards data.

        Phase 16d: reads claimable reward estimates from DeFiLlama protocol
        data. Actual claimable amounts require on-chain calls (Phase 16e).
        Best-effort, never raises.
        """
        # For Phase 16d we track estimated balances based on yield position data.
        # When the yield_strategy deploys into Aerodrome/Moonwell, those positions
        # accrue governance tokens at a known emission rate.
        # We persist whatever we know and rely on the position tracker for amounts.

        if self._redis is None:
            return

        try:
            yield_pos = await self._redis.get_json("eos:oikos:yield_positions")
        except Exception:
            return

        if not yield_pos:
            return

        positions = yield_pos if isinstance(yield_pos, list) else [yield_pos]
        updated: dict[str, Any] = {}

        for pos in positions:
            protocol = str(pos.get("protocol", ""))
            if protocol not in _PROTOCOL_SNAPSHOT_SPACES:
                continue

            # Estimate governance token balance from rewards_usd field (set by executor)
            rewards_usd_str = pos.get("rewards_usd", "0")
            try:
                rewards_usd = Decimal(str(rewards_usd_str))
            except InvalidOperation:
                rewards_usd = Decimal("0")

            token_symbol = {
                "aerodrome": "AERO",
                "moonwell": "WELL",
                "compound": "COMP",
            }.get(protocol, protocol.upper())

            gov_token = GovernanceToken(
                protocol=protocol,
                token_symbol=token_symbol,
                balance=rewards_usd,   # Approximate: 1:1 USD for threshold checks
                balance_usd=rewards_usd,
                contract_address="",   # Phase 16e: fetch from registry
                snapshot_space=_PROTOCOL_SNAPSHOT_SPACES.get(protocol, ""),
            )
            self._token_balances[protocol] = gov_token
            updated[protocol] = {
                "token_symbol": token_symbol,
                "balance": str(rewards_usd),
                "balance_usd": str(rewards_usd),
            }

        if updated:
            try:
                await self._redis.set_json(_TOKEN_BALANCES_KEY, updated)
            except Exception as exc:
                self._log.warning("token_balance_persist_failed", error=str(exc))

    def _load_balances(self, raw: dict[str, Any]) -> None:
        for protocol, data in raw.items():
            try:
                self._token_balances[protocol] = GovernanceToken(
                    protocol=protocol,
                    token_symbol=str(data.get("token_symbol", protocol.upper())),
                    balance=Decimal(str(data.get("balance", "0"))),
                    balance_usd=Decimal(str(data.get("balance_usd", "0"))),
                    contract_address=str(data.get("contract_address", "")),
                    snapshot_space=_PROTOCOL_SNAPSHOT_SPACES.get(protocol, ""),
                )
            except (InvalidOperation, KeyError):
                continue

    async def _record_vote_intent(
        self, decision: VoteDecision, proposal: GovernanceProposal
    ) -> bool:
        """
        Record the vote intent.

        Phase 16d: off-chain record only (Snapshot EIP-712 signing in Phase 16e).
        Returns True always for Phase 16d.
        """
        self._log.info(
            "governance_vote_intent_recorded",
            proposal_id=decision.proposal_id,
            choice=decision.choice_label,
            protocol=decision.protocol,
            note="Phase 16d: off-chain intent only; on-chain signing in Phase 16e",
        )
        return True

    async def _persist_voted_proposals(self) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set_json(
                _VOTED_PROPOSALS_KEY, list(self._voted_proposals)
            )
        except Exception as exc:
            self._log.warning("voted_proposals_persist_failed", error=str(exc))

    # ── Public API ────────────────────────────────────────────────────────────

    def get_token_balance(self, protocol: str) -> Decimal:
        """Return governance token balance (USD equiv) for a protocol."""
        tok = self._token_balances.get(protocol)
        return tok.balance_usd if tok else Decimal("0")

    def snapshot(self) -> dict[str, Any]:
        """Return a summary snapshot for Benchmarks / Oikos stats."""
        return {
            "protocols_with_tokens": list(self._token_balances.keys()),
            "total_governance_usd": str(
                sum(t.balance_usd for t in self._token_balances.values())
            ),
            "voted_proposals_count": len(self._voted_proposals),
        }


# ─── Event emission helper ────────────────────────────────────────────────────


async def _emit(
    event_bus: "EventBus | None",
    event_type: str,
    data: dict[str, Any],
) -> None:
    if event_bus is None:
        return
    try:
        from systems.synapse.types import SynapseEvent, SynapseEventType  # noqa: PLC0415

        await event_bus.emit(
            SynapseEvent(
                event_type=SynapseEventType(event_type),
                source_system=_EVENT_SOURCE,
                data=data,
            )
        )
    except Exception as exc:
        logger.error("governance_tracker_emit_failed", event_type=event_type, error=str(exc))
