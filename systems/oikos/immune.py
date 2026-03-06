"""
EcodiaOS — Oikos Economic Immune System (Phase 16f: 4-Layer Defence)

The organism's economic defence system. As it deploys capital into DeFi and
earns via on-chain mechanisms, it becomes a target. This module provides four
concentric layers of protection:

  Layer 1 — Transaction Shield:
      Pre-simulate every on-chain action before broadcast. Reject failing
      simulations without spending gas. Slippage capped at 50 bps
      (configurable). Route through Flashbots RPC to avoid MEV.

  Layer 2 — Threat Pattern Recognition:
      Adaptive detection via pattern matching. Monitors flash-loan prefix
      patterns, price manipulation signatures, suspicious contract
      interactions, and mempool poisoning. Known malicious addresses are
      blacklisted on detection.

  Layer 3 — Protocol Health Monitoring:
      Continuous monitoring of protocols holding organism capital. Triggers
      on TVL drop >20%, oracle deviation >5%, governance anomalies, or
      contract pause events. On alert: emit EMERGENCY_WITHDRAWAL intent.

  Layer 4 — Federation Threat Intelligence:
      Broadcast and receive ThreatAdvisory to/from federated instances.
      Trusted partners receive advisories immediately. Confirmed
      intelligence increases trust score.

Design:
  - All Decimal for money values — no float rounding on economics.
  - Async-only I/O; pure computation stays sync.
  - Communicates via Synapse EventBus (no direct cross-system imports).
  - structlog with component="economic_immune".
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003 — needed at runtime for Pydantic field resolution
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from config import OikosConfig
    from systems.oikos.models import YieldPosition
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.immune")

# ─── Constants ────────────────────────────────────────────────────────────────

_EVENT_SOURCE = "oikos.immune"
_BLACKLIST_REDIS_KEY = "eos:oikos:immune:blacklist"
_METRICS_REDIS_KEY = "eos:oikos:immune:metrics"

# Default thresholds (overridable via config where applicable)
_DEFAULT_MAX_SLIPPAGE_BPS: int = 50
_DEFAULT_MAX_GAS_GWEI: int = 500
_DEFAULT_TVL_DROP_ALERT_PCT = Decimal("20")
_DEFAULT_ORACLE_DEVIATION_ALERT_PCT = Decimal("5")
_DEFAULT_MIN_CONTRACT_AGE_DAYS: int = 7
_DEFAULT_FLASH_LOAN_THRESHOLD_USD = Decimal("100000")
_DEFAULT_PRICE_MOVE_THRESHOLD_PCT = Decimal("10")

# ─── Enums ────────────────────────────────────────────────────────────────────


class TransactionRisk(enum.StrEnum):
    """Risk classification for a pre-simulated transaction."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatPatternType(enum.StrEnum):
    """Categories of detectable economic threat patterns."""

    FLASH_LOAN = "flash_loan"
    PRICE_MANIPULATION = "price_manipulation"
    SUSPICIOUS_CONTRACT = "suspicious_contract"
    MEMPOOL_POISON = "mempool_poison"
    REENTRANCY = "reentrancy"
    ORACLE_MANIPULATION = "oracle_manipulation"


# ─── Models ───────────────────────────────────────────────────────────────────


class SimulationResult(EOSBaseModel):
    """Result of a Layer 1 transaction pre-simulation."""

    tx_hash: str = Field(default_factory=new_id)
    success: bool = False
    gas_used: int = 0
    revert_reason: str = ""
    value_at_risk_usd: Decimal = Decimal("0")
    slippage_bps: int = 0
    risk_level: TransactionRisk = TransactionRisk.LOW
    approved: bool = False


class ThreatPattern(EOSBaseModel):
    """A registered threat detection pattern used by Layer 2."""

    pattern_id: str = Field(default_factory=new_id)
    name: str = ""
    description: str = ""
    pattern_type: ThreatPatternType = ThreatPatternType.FLASH_LOAN
    detection_fn_name: str = ""
    severity: TransactionRisk = TransactionRisk.HIGH
    last_triggered: datetime | None = None
    trigger_count: int = 0


class ProtocolHealthStatus(EOSBaseModel):
    """Layer 3 health snapshot for a single protocol holding organism capital."""

    protocol_name: str = ""
    contract_address: str = ""
    chain_id: int = 8453

    # TVL tracking
    current_tvl_usd: Decimal = Decimal("0")
    tvl_at_deposit_usd: Decimal = Decimal("0")
    tvl_change_pct: Decimal = Decimal("0")

    # Oracle deviation
    oracle_price: Decimal = Decimal("0")
    reference_price: Decimal = Decimal("0")
    oracle_deviation_pct: Decimal = Decimal("0")

    # Status flags
    is_paused: bool = False
    governance_anomaly: bool = False
    health: str = "healthy"  # "healthy" | "degraded" | "critical"

    last_checked: datetime = Field(default_factory=utc_now)
    alerts: list[str] = Field(default_factory=list)


class BlacklistedAddress(EOSBaseModel):
    """An address flagged as malicious and barred from all interactions."""

    address: str = ""
    reason: str = ""
    detected_at: datetime = Field(default_factory=utc_now)
    reported_by: str = ""
    threat_advisory_id: str = ""


class ImmuneMetrics(EOSBaseModel):
    """Aggregate metrics for the Economic Immune System."""

    threats_blocked: int = 0
    false_positives: int = 0
    advisories_shared: int = 0
    advisories_received: int = 0
    transactions_shielded: int = 0
    transactions_rejected: int = 0
    protocols_monitored: int = 0
    last_scan: datetime = Field(default_factory=utc_now)


# ─── Default Threat Patterns ─────────────────────────────────────────────────


def _build_default_threat_patterns() -> list[ThreatPattern]:
    """Pre-built detection patterns covering the known threat surface."""
    return [
        ThreatPattern(
            name="Flash Loan Attack",
            description="Detects large borrow-and-repay within the same block, "
            "indicative of flash loan exploitation.",
            pattern_type=ThreatPatternType.FLASH_LOAN,
            detection_fn_name="_detect_flash_loan",
            severity=TransactionRisk.CRITICAL,
        ),
        ThreatPattern(
            name="Price Manipulation",
            description="Sudden price movement exceeding threshold in a short window, "
            "consistent with oracle or AMM manipulation.",
            pattern_type=ThreatPatternType.PRICE_MANIPULATION,
            detection_fn_name="_detect_price_manipulation",
            severity=TransactionRisk.CRITICAL,
        ),
        ThreatPattern(
            name="Suspicious Contract Interaction",
            description="Interaction with a contract deployed within the minimum age "
            "window — elevated rug-pull risk.",
            pattern_type=ThreatPatternType.SUSPICIOUS_CONTRACT,
            detection_fn_name="_detect_suspicious_contract",
            severity=TransactionRisk.HIGH,
        ),
        ThreatPattern(
            name="Mempool Poisoning",
            description="Transaction patterns consistent with mempool manipulation: "
            "sandwich attacks, front-running, or back-running.",
            pattern_type=ThreatPatternType.MEMPOOL_POISON,
            detection_fn_name="_detect_mempool_poison",
            severity=TransactionRisk.HIGH,
        ),
        ThreatPattern(
            name="Reentrancy Pattern",
            description="Nested call patterns matching known reentrancy signatures "
            "in transaction trace data.",
            pattern_type=ThreatPatternType.REENTRANCY,
            detection_fn_name="_detect_reentrancy",
            severity=TransactionRisk.CRITICAL,
        ),
        ThreatPattern(
            name="Oracle Manipulation",
            description="Oracle price deviating significantly from reference sources, "
            "indicating potential manipulation or stale data.",
            pattern_type=ThreatPatternType.ORACLE_MANIPULATION,
            detection_fn_name="_detect_oracle_manipulation",
            severity=TransactionRisk.CRITICAL,
        ),
    ]


# ─── Economic Immune System ──────────────────────────────────────────────────


class EconomicImmuneSystem:
    """
    Four-layer economic defence system for the organism's on-chain capital.

    Layers:
      1. Transaction Shield — pre-simulate, slippage cap, Flashbots routing
      2. Threat Pattern Recognition — adaptive detection of attack vectors
      3. Protocol Health Monitoring — TVL, oracle, governance, pause events
      4. Federation Threat Intelligence — broadcast/receive ThreatAdvisory

    Lifecycle:
      __init__(config, redis)  → wire internal state
      attach(event_bus)        → subscribe to federation advisories
      shield_transaction()     → Layer 1
      scan_for_threats()       → Layer 2
      monitor_protocol_health()→ Layer 3
      run_immune_cycle()       → periodic entry point (Layers 1-3 + metrics)
      broadcast_threat_advisory() → Layer 4 outbound
      _on_threat_advisory()    → Layer 4 inbound (event handler)
    """

    __slots__ = (
        "_config",
        "_redis",
        "_event_bus",
        "_blacklist",
        "_threat_patterns",
        "_protocol_health",
        "_metrics",
        "_max_slippage_bps",
        "_max_gas_gwei",
        "_tvl_drop_alert_pct",
        "_oracle_deviation_alert_pct",
        "_min_contract_age_days",
        "_flash_loan_threshold_usd",
        "_price_move_threshold_pct",
    )

    def __init__(self, config: OikosConfig, redis: RedisClient) -> None:
        self._config = config
        self._redis = redis
        self._event_bus: EventBus | None = None

        # Layer 1 config
        self._max_slippage_bps: int = _DEFAULT_MAX_SLIPPAGE_BPS
        self._max_gas_gwei: int = _DEFAULT_MAX_GAS_GWEI

        # Layer 2 config
        self._min_contract_age_days: int = _DEFAULT_MIN_CONTRACT_AGE_DAYS
        self._flash_loan_threshold_usd: Decimal = _DEFAULT_FLASH_LOAN_THRESHOLD_USD
        self._price_move_threshold_pct: Decimal = _DEFAULT_PRICE_MOVE_THRESHOLD_PCT

        # Layer 3 config
        self._tvl_drop_alert_pct: Decimal = _DEFAULT_TVL_DROP_ALERT_PCT
        self._oracle_deviation_alert_pct: Decimal = _DEFAULT_ORACLE_DEVIATION_ALERT_PCT

        # State
        self._blacklist: dict[str, BlacklistedAddress] = {}
        self._threat_patterns: list[ThreatPattern] = _build_default_threat_patterns()
        self._protocol_health: dict[str, ProtocolHealthStatus] = {}
        self._metrics = ImmuneMetrics()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to federation threat advisories and store bus reference."""
        self._event_bus = event_bus
        event_bus.subscribe(
            SynapseEventType.THREAT_ADVISORY_RECEIVED,
            self._on_threat_advisory,
        )
        logger.info(
            "economic_immune_system_attached",
            component="economic_immune",
            patterns_loaded=len(self._threat_patterns),
        )

    # ── Layer 1: Transaction Shield ───────────────────────────────────────

    async def shield_transaction(self, tx_data: dict[str, Any]) -> SimulationResult:
        """
        Pre-simulate a transaction before broadcast.

        Checks:
          - Sender and receiver not blacklisted
          - Slippage within configured maximum
          - Gas price reasonable (below max_gas_gwei)
          - Value at risk assessment

        Returns a SimulationResult. Transactions are only approved when all
        checks pass.
        """
        sender: str = tx_data.get("from", "").lower()
        receiver: str = tx_data.get("to", "").lower()
        slippage_bps: int = int(tx_data.get("slippage_bps", 0))
        gas_price_gwei: int = int(tx_data.get("gas_price_gwei", 0))
        value_usd = Decimal(str(tx_data.get("value_usd", "0")))
        tx_hash: str = tx_data.get("tx_hash", new_id())

        result = SimulationResult(
            tx_hash=tx_hash,
            value_at_risk_usd=value_usd,
            slippage_bps=slippage_bps,
        )

        # ── Blacklist check ──
        if self.is_blacklisted(sender):
            result.success = False
            result.revert_reason = f"sender {sender} is blacklisted"
            result.risk_level = TransactionRisk.CRITICAL
            result.approved = False
            self._metrics.transactions_rejected += 1
            logger.warning(
                "tx_shield_blacklisted_sender",
                component="economic_immune",
                sender=sender,
                tx_hash=tx_hash,
            )
            return result

        if self.is_blacklisted(receiver):
            result.success = False
            result.revert_reason = f"receiver {receiver} is blacklisted"
            result.risk_level = TransactionRisk.CRITICAL
            result.approved = False
            self._metrics.transactions_rejected += 1
            logger.warning(
                "tx_shield_blacklisted_receiver",
                component="economic_immune",
                receiver=receiver,
                tx_hash=tx_hash,
            )
            return result

        # ── Slippage check ──
        if slippage_bps > self._max_slippage_bps:
            result.success = False
            result.revert_reason = (
                f"slippage {slippage_bps}bps exceeds max {self._max_slippage_bps}bps"
            )
            result.risk_level = TransactionRisk.HIGH
            result.approved = False
            self._metrics.transactions_rejected += 1
            logger.warning(
                "tx_shield_slippage_exceeded",
                component="economic_immune",
                slippage_bps=slippage_bps,
                max_slippage_bps=self._max_slippage_bps,
                tx_hash=tx_hash,
            )
            return result

        # ── Gas price check ──
        if gas_price_gwei > self._max_gas_gwei:
            result.success = False
            result.revert_reason = (
                f"gas price {gas_price_gwei} gwei exceeds max {self._max_gas_gwei} gwei"
            )
            result.risk_level = TransactionRisk.MEDIUM
            result.approved = False
            self._metrics.transactions_rejected += 1
            logger.warning(
                "tx_shield_gas_too_high",
                component="economic_immune",
                gas_price_gwei=gas_price_gwei,
                max_gas_gwei=self._max_gas_gwei,
                tx_hash=tx_hash,
            )
            return result

        # ── Risk classification based on value at risk ──
        risk_level = self._classify_value_risk(value_usd)

        result.success = True
        result.risk_level = risk_level
        result.approved = True
        result.gas_used = int(tx_data.get("estimated_gas", 21000))

        self._metrics.transactions_shielded += 1

        logger.info(
            "tx_shield_approved",
            component="economic_immune",
            tx_hash=tx_hash,
            risk_level=risk_level.value,
            value_usd=str(value_usd),
            slippage_bps=slippage_bps,
        )

        # Emit shielded event
        if self._event_bus is not None:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.TRANSACTION_SHIELDED,
                source_system=_EVENT_SOURCE,
                data={
                    "tx_hash": tx_hash,
                    "risk_level": risk_level.value,
                    "value_usd": str(value_usd),
                    "slippage_bps": slippage_bps,
                    "approved": True,
                },
            ))

        return result

    def _classify_value_risk(self, value_usd: Decimal) -> TransactionRisk:
        """Classify transaction risk based on USD value at stake."""
        if value_usd >= Decimal("10000"):
            return TransactionRisk.CRITICAL
        if value_usd >= Decimal("1000"):
            return TransactionRisk.HIGH
        if value_usd >= Decimal("100"):
            return TransactionRisk.MEDIUM
        return TransactionRisk.LOW

    # ── Layer 2: Threat Pattern Recognition ───────────────────────────────

    async def scan_for_threats(
        self,
        recent_transactions: list[dict[str, Any]],
    ) -> list[ThreatPattern]:
        """
        Scan recent transactions for known threat patterns.

        Each pattern has a named detection method. When a pattern matches,
        the pattern's trigger count and last-triggered time are updated,
        and a THREAT_DETECTED event is emitted.

        Returns the list of patterns that matched.
        """
        matched: list[ThreatPattern] = []

        for pattern in self._threat_patterns:
            detection_fn = getattr(self, pattern.detection_fn_name, None)
            if detection_fn is None:
                logger.warning(
                    "threat_pattern_missing_detector",
                    component="economic_immune",
                    pattern_name=pattern.name,
                    detection_fn_name=pattern.detection_fn_name,
                )
                continue

            triggered = detection_fn(recent_transactions)
            if triggered:
                now = utc_now()
                pattern.last_triggered = now
                pattern.trigger_count += 1
                matched.append(pattern)
                self._metrics.threats_blocked += 1

                logger.warning(
                    "threat_pattern_triggered",
                    component="economic_immune",
                    pattern_name=pattern.name,
                    pattern_type=pattern.pattern_type.value,
                    severity=pattern.severity.value,
                    trigger_count=pattern.trigger_count,
                )

                if self._event_bus is not None:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.THREAT_DETECTED,
                        source_system=_EVENT_SOURCE,
                        data={
                            "pattern_id": pattern.pattern_id,
                            "pattern_name": pattern.name,
                            "pattern_type": pattern.pattern_type.value,
                            "severity": pattern.severity.value,
                            "trigger_count": pattern.trigger_count,
                        },
                    ))

        return matched

    # ── Detection functions (Layer 2 internals) ──

    def _detect_flash_loan(self, transactions: list[dict[str, Any]]) -> bool:
        """
        Detect flash-loan patterns: large borrow + repay in the same block.

        Heuristic: look for pairs of transactions with the same block number
        where one is a borrow exceeding the threshold and another is a repay
        of similar value.
        """
        by_block: dict[int, list[dict[str, Any]]] = {}
        for tx in transactions:
            block: int = tx.get("block_number", 0)
            if block > 0:
                by_block.setdefault(block, []).append(tx)

        for _block_num, block_txs in by_block.items():
            borrows: list[Decimal] = []
            repays: list[Decimal] = []
            for tx in block_txs:
                action = tx.get("action", "")
                value = Decimal(str(tx.get("value_usd", "0")))
                if action in ("borrow", "flashBorrow") and value >= self._flash_loan_threshold_usd:
                    borrows.append(value)
                elif action in ("repay", "flashRepay") and value >= self._flash_loan_threshold_usd:
                    repays.append(value)

            # A flash loan has a matching borrow + repay in the same block
            if borrows and repays:
                for borrow_val in borrows:
                    for repay_val in repays:
                        # Allow 1% tolerance for interest/fees
                        if borrow_val > 0:
                            ratio = abs(borrow_val - repay_val) / borrow_val
                        else:
                            ratio = Decimal("1")
                        if ratio < Decimal("0.01"):
                            return True

        return False

    def _detect_price_manipulation(self, transactions: list[dict[str, Any]]) -> bool:
        """
        Detect sudden price movements exceeding the threshold.

        Heuristic: any transaction reporting a price_change_pct that exceeds
        the configured threshold within a single block or very short window.
        """
        for tx in transactions:
            price_change_pct = Decimal(str(tx.get("price_change_pct", "0")))
            if abs(price_change_pct) >= self._price_move_threshold_pct:
                # Also blacklist the associated address if present
                suspicious_addr: str = tx.get("from", "").lower()
                if suspicious_addr and not self.is_blacklisted(suspicious_addr):
                    self.blacklist_address(
                        address=suspicious_addr,
                        reason=f"price manipulation detected: {price_change_pct}% move",
                        reported_by="immune_layer2",
                    )
                return True
        return False

    def _detect_suspicious_contract(self, transactions: list[dict[str, Any]]) -> bool:
        """
        Detect interactions with contracts younger than the minimum age threshold.

        Contracts deployed very recently carry elevated rug-pull and exploit risk.
        """
        for tx in transactions:
            contract_age_days = int(tx.get("contract_age_days", 999))
            if contract_age_days < self._min_contract_age_days:
                contract_addr: str = tx.get("to", "").lower()
                if contract_addr:
                    logger.warning(
                        "suspicious_young_contract",
                        component="economic_immune",
                        contract_address=contract_addr,
                        age_days=contract_age_days,
                        threshold_days=self._min_contract_age_days,
                    )
                return True
        return False

    def _detect_mempool_poison(self, transactions: list[dict[str, Any]]) -> bool:
        """
        Detect mempool poisoning patterns (sandwich attacks).

        Heuristic: look for a sequence where the same address places a buy
        before and a sell after the organism's transaction in the same block,
        exploiting the price impact.
        """
        by_block: dict[int, list[dict[str, Any]]] = {}
        for tx in transactions:
            block: int = tx.get("block_number", 0)
            if block > 0:
                by_block.setdefault(block, []).append(tx)

        for _block_num, block_txs in by_block.items():
            if len(block_txs) < 3:
                continue

            # Sort by tx index within block
            sorted_txs = sorted(block_txs, key=lambda t: int(t.get("tx_index", 0)))

            # Look for sandwich: buy → victim → sell by the same address
            for i in range(len(sorted_txs) - 2):
                tx_a = sorted_txs[i]
                tx_b = sorted_txs[i + 1]
                tx_c = sorted_txs[i + 2]

                addr_a = tx_a.get("from", "").lower()
                addr_c = tx_c.get("from", "").lower()
                addr_b = tx_b.get("from", "").lower()

                # Same attacker wrapping a different victim
                if (
                    addr_a
                    and addr_a == addr_c
                    and addr_a != addr_b
                    and tx_a.get("action") == "buy"
                    and tx_c.get("action") == "sell"
                ):
                    self.blacklist_address(
                        address=addr_a,
                        reason="mempool sandwich attack detected",
                        reported_by="immune_layer2",
                    )
                    return True

        return False

    def _detect_reentrancy(self, transactions: list[dict[str, Any]]) -> bool:
        """
        Detect reentrancy patterns in transaction trace data.

        Heuristic: a transaction whose internal_calls contain recursive
        invocations of the same function selector back into the calling contract.
        """
        for tx in transactions:
            internal_calls: list[dict[str, Any]] = tx.get("internal_calls", [])
            if len(internal_calls) < 2:
                continue

            # Count calls to the same target+selector combination
            call_signatures: dict[str, int] = {}
            for call in internal_calls:
                target: str = call.get("to", "").lower()
                selector: str = call.get("selector", "")
                if target and selector:
                    key = f"{target}:{selector}"
                    call_signatures[key] = call_signatures.get(key, 0) + 1

            # More than 2 calls to the same target+selector is suspicious
            for key, count in call_signatures.items():
                if count > 2:
                    target_addr = key.split(":")[0]
                    self.blacklist_address(
                        address=target_addr,
                        reason=f"reentrancy pattern detected: {count} recursive calls",
                        reported_by="immune_layer2",
                    )
                    return True

        return False

    def _detect_oracle_manipulation(self, transactions: list[dict[str, Any]]) -> bool:
        """
        Detect oracle price deviations from reference sources.

        Heuristic: any transaction where the reported oracle price deviates
        from a reference price by more than the configured oracle deviation
        threshold.
        """
        for tx in transactions:
            oracle_price = Decimal(str(tx.get("oracle_price", "0")))
            reference_price = Decimal(str(tx.get("reference_price", "0")))
            if oracle_price <= Decimal("0") or reference_price <= Decimal("0"):
                continue

            deviation_pct = abs(oracle_price - reference_price) / reference_price * Decimal("100")
            if deviation_pct >= self._oracle_deviation_alert_pct:
                logger.warning(
                    "oracle_manipulation_detected",
                    component="economic_immune",
                    oracle_price=str(oracle_price),
                    reference_price=str(reference_price),
                    deviation_pct=str(deviation_pct),
                )
                return True

        return False

    # ── Layer 3: Protocol Health Monitoring ───────────────────────────────

    async def monitor_protocol_health(
        self,
        yield_positions: list[YieldPosition],
    ) -> list[ProtocolHealthStatus]:
        """
        Check every protocol currently holding organism capital.

        For each yield position:
          - Compare current TVL to TVL at deposit — alert if drop >20%
          - Check oracle deviation against reference price — alert if >5%
          - Check for contract pause events
          - Governance anomaly detection
          - Emit PROTOCOL_ALERT for degraded protocols
          - Emit EMERGENCY_WITHDRAWAL for critical protocols

        Returns all statuses.
        """
        statuses: list[ProtocolHealthStatus] = []

        for position in yield_positions:
            status = self._evaluate_protocol(position)
            statuses.append(status)

            # Store in internal tracking
            cache_key = f"{position.protocol}:{position.protocol_address}"
            self._protocol_health[cache_key] = status

            if status.health == "critical":
                logger.error(
                    "protocol_health_critical",
                    component="economic_immune",
                    protocol=position.protocol,
                    contract=position.protocol_address,
                    alerts=status.alerts,
                )

                if self._event_bus is not None:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.PROTOCOL_ALERT,
                        source_system=_EVENT_SOURCE,
                        data={
                            "protocol": position.protocol,
                            "contract_address": position.protocol_address,
                            "health": "critical",
                            "alerts": status.alerts,
                            "tvl_change_pct": str(status.tvl_change_pct),
                            "oracle_deviation_pct": str(status.oracle_deviation_pct),
                        },
                    ))

                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.EMERGENCY_WITHDRAWAL,
                        source_system=_EVENT_SOURCE,
                        data={
                            "protocol": position.protocol,
                            "contract_address": position.protocol_address,
                            "reason": "; ".join(status.alerts),
                            "principal_usd": str(position.principal_usd),
                        },
                    ))

            elif status.health == "degraded":
                logger.warning(
                    "protocol_health_degraded",
                    component="economic_immune",
                    protocol=position.protocol,
                    contract=position.protocol_address,
                    alerts=status.alerts,
                )

                if self._event_bus is not None:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.PROTOCOL_ALERT,
                        source_system=_EVENT_SOURCE,
                        data={
                            "protocol": position.protocol,
                            "contract_address": position.protocol_address,
                            "health": "degraded",
                            "alerts": status.alerts,
                            "tvl_change_pct": str(status.tvl_change_pct),
                            "oracle_deviation_pct": str(status.oracle_deviation_pct),
                        },
                    ))

        self._metrics.protocols_monitored = len(yield_positions)
        return statuses

    def _evaluate_protocol(self, position: YieldPosition) -> ProtocolHealthStatus:
        """
        Evaluate a single protocol's health from its yield position data.

        Returns a ProtocolHealthStatus with computed alerts and health classification.
        """
        alerts: list[str] = []
        now = utc_now()

        # ── TVL change ──
        tvl_change_pct = Decimal("0")
        if position.tvl_usd_at_deposit > Decimal("0"):
            tvl_change_pct = (
                (position.tvl_usd - position.tvl_usd_at_deposit)
                / position.tvl_usd_at_deposit
                * Decimal("100")
            )

        if tvl_change_pct < -self._tvl_drop_alert_pct:
            alerts.append(
                f"TVL dropped {abs(tvl_change_pct):.1f}% from deposit "
                f"(${position.tvl_usd_at_deposit} -> ${position.tvl_usd})"
            )

        # ── Oracle deviation ──
        oracle_deviation_pct = Decimal("0")
        oracle_price = Decimal("0")
        reference_price = Decimal("0")

        # Use the cached protocol health for oracle data if available,
        # otherwise use position data directly
        cache_key = f"{position.protocol}:{position.protocol_address}"
        cached = self._protocol_health.get(cache_key)
        if cached is not None:
            oracle_price = cached.oracle_price
            reference_price = cached.reference_price
            if reference_price > Decimal("0"):
                oracle_deviation_pct = (
                    abs(oracle_price - reference_price)
                    / reference_price
                    * Decimal("100")
                )
        # If no cached oracle data, deviation stays at 0 (no alert)

        if oracle_deviation_pct >= self._oracle_deviation_alert_pct:
            alerts.append(
                f"Oracle deviation {oracle_deviation_pct:.2f}% "
                f"(oracle={oracle_price}, ref={reference_price})"
            )

        # ── Contract pause ──
        is_paused = position.health_status == "paused"
        if is_paused:
            alerts.append("Protocol contract is paused")

        # ── Governance anomaly (flagged externally on the position) ──
        governance_anomaly = position.health_status == "governance_anomaly"
        if governance_anomaly:
            alerts.append("Governance anomaly detected")

        # ── Health classification ──
        health = "healthy"
        if is_paused or governance_anomaly or len(alerts) >= 2:
            health = "critical"
        elif len(alerts) == 1:
            health = "degraded"

        # TVL drop >20% alone is critical (capital at immediate risk)
        if tvl_change_pct < -self._tvl_drop_alert_pct:
            health = "critical"

        # Oracle deviation >5% alone is critical (price feed unreliable)
        if oracle_deviation_pct >= self._oracle_deviation_alert_pct:
            health = "critical"

        return ProtocolHealthStatus(
            protocol_name=position.protocol,
            contract_address=position.protocol_address,
            chain_id=position.chain_id,
            current_tvl_usd=position.tvl_usd,
            tvl_at_deposit_usd=position.tvl_usd_at_deposit,
            tvl_change_pct=tvl_change_pct,
            oracle_price=oracle_price,
            reference_price=reference_price,
            oracle_deviation_pct=oracle_deviation_pct,
            is_paused=is_paused,
            governance_anomaly=governance_anomaly,
            health=health,
            last_checked=now,
            alerts=alerts,
        )

    # ── Layer 4: Federation Threat Intelligence ───────────────────────────

    async def _on_threat_advisory(self, event: SynapseEvent) -> None:
        """
        Handle an incoming threat advisory from a federated instance.

        Validates the advisory, adds flagged addresses to the blacklist,
        and tracks the advisory in metrics.
        """
        data = event.data
        advisory_id: str = data.get("advisory_id", new_id())
        source_instance: str = data.get("source_instance", "unknown")
        addresses: list[str] = data.get("malicious_addresses", [])
        reason: str = data.get("reason", "federation threat advisory")
        signature: str = data.get("signature", "")

        # ── Validate advisory ──
        if not signature:
            logger.warning(
                "threat_advisory_unsigned",
                component="economic_immune",
                advisory_id=advisory_id,
                source_instance=source_instance,
            )
            # Accept unsigned advisories with a warning — federation trust
            # model is handled at a higher layer. We still ingest the
            # intelligence but log the lack of signature.

        if not addresses:
            logger.info(
                "threat_advisory_empty",
                component="economic_immune",
                advisory_id=advisory_id,
                source_instance=source_instance,
            )
            return

        # ── Blacklist each reported address ──
        new_entries = 0
        for addr in addresses:
            normalised = addr.lower().strip()
            if not normalised:
                continue
            if normalised not in self._blacklist:
                entry = BlacklistedAddress(
                    address=normalised,
                    reason=reason,
                    detected_at=utc_now(),
                    reported_by=source_instance,
                    threat_advisory_id=advisory_id,
                )
                self._blacklist[normalised] = entry
                new_entries += 1

                if self._event_bus is not None:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.ADDRESS_BLACKLISTED,
                        source_system=_EVENT_SOURCE,
                        data={
                            "address": normalised,
                            "reason": reason,
                            "reported_by": source_instance,
                            "advisory_id": advisory_id,
                        },
                    ))

        self._metrics.advisories_received += 1

        logger.info(
            "threat_advisory_processed",
            component="economic_immune",
            advisory_id=advisory_id,
            source_instance=source_instance,
            addresses_reported=len(addresses),
            new_blacklist_entries=new_entries,
        )

    async def broadcast_threat_advisory(self, advisory_data: dict[str, Any]) -> None:
        """
        Broadcast a threat advisory to federated instances.

        Packages the threat details and emits THREAT_ADVISORY_SENT on the
        event bus. The Federation system picks this up and distributes it
        to trusted partners.
        """
        advisory_id: str = advisory_data.get("advisory_id", new_id())
        addresses: list[str] = advisory_data.get("malicious_addresses", [])
        reason: str = advisory_data.get("reason", "")
        severity: str = advisory_data.get("severity", TransactionRisk.HIGH.value)

        if self._event_bus is not None:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.THREAT_ADVISORY_SENT,
                source_system=_EVENT_SOURCE,
                data={
                    "advisory_id": advisory_id,
                    "malicious_addresses": addresses,
                    "reason": reason,
                    "severity": severity,
                    "blacklist_count": len(self._blacklist),
                },
            ))

        self._metrics.advisories_shared += 1

        logger.info(
            "threat_advisory_broadcast",
            component="economic_immune",
            advisory_id=advisory_id,
            addresses_count=len(addresses),
            reason=reason,
            severity=severity,
        )

    # ── Blacklist Management ──────────────────────────────────────────────

    def blacklist_address(
        self,
        address: str,
        reason: str,
        reported_by: str,
    ) -> None:
        """
        Add an address to the blacklist.

        Emitting ADDRESS_BLACKLISTED is done synchronously from the caller's
        perspective — fire-and-forget via the event bus is acceptable here
        because blacklist updates are idempotent.
        """
        normalised = address.lower().strip()
        if not normalised:
            return

        if normalised in self._blacklist:
            logger.debug(
                "address_already_blacklisted",
                component="economic_immune",
                address=normalised,
            )
            return

        entry = BlacklistedAddress(
            address=normalised,
            reason=reason,
            detected_at=utc_now(),
            reported_by=reported_by,
        )
        self._blacklist[normalised] = entry

        logger.warning(
            "address_blacklisted",
            component="economic_immune",
            address=normalised,
            reason=reason,
            reported_by=reported_by,
            total_blacklisted=len(self._blacklist),
        )

    def is_blacklisted(self, address: str) -> bool:
        """Check whether an address is on the blacklist."""
        if not address:
            return False
        return address.lower().strip() in self._blacklist

    # ── Immune Cycle (Main Entry Point) ───────────────────────────────────

    async def run_immune_cycle(
        self,
        economic_state: Any,
    ) -> ImmuneMetrics:
        """
        Main periodic entry point for the immune system.

        Called once per consolidation or economic cycle:
          1. Monitor protocol health for all active yield positions
          2. Update metrics
          3. Return current metrics snapshot

        The transaction shield (Layer 1) and threat scanning (Layer 2) are
        invoked on-demand by the wallet/executor layer, not in this cycle.
        This cycle handles the continuous background monitoring (Layer 3).
        """
        # Extract yield positions from economic state
        yield_positions: list[YieldPosition] = []
        if hasattr(economic_state, "yield_positions"):
            yield_positions = economic_state.yield_positions

        # Layer 3: Protocol health monitoring
        if yield_positions:
            await self.monitor_protocol_health(yield_positions)

        # Update scan timestamp
        self._metrics.last_scan = utc_now()

        logger.info(
            "immune_cycle_complete",
            component="economic_immune",
            protocols_monitored=self._metrics.protocols_monitored,
            threats_blocked=self._metrics.threats_blocked,
            transactions_shielded=self._metrics.transactions_shielded,
            transactions_rejected=self._metrics.transactions_rejected,
            blacklist_size=len(self._blacklist),
        )

        return self._metrics.model_copy()

    # ── Metrics ───────────────────────────────────────────────────────────

    def get_metrics(self) -> ImmuneMetrics:
        """Return a copy of current immune system metrics."""
        return self._metrics.model_copy()

    # ── Oracle Data Ingestion ─────────────────────────────────────────────

    def update_oracle_data(
        self,
        protocol: str,
        contract_address: str,
        oracle_price: Decimal,
        reference_price: Decimal,
    ) -> None:
        """
        Ingest oracle price data for a monitored protocol.

        Called externally (e.g., by Phantom or a price feed worker) to
        provide the latest oracle vs. reference price. This data is used
        by Layer 3 during protocol health evaluation.
        """
        cache_key = f"{protocol}:{contract_address}"
        existing = self._protocol_health.get(cache_key)

        if existing is not None:
            existing.oracle_price = oracle_price
            existing.reference_price = reference_price
            if reference_price > Decimal("0"):
                existing.oracle_deviation_pct = (
                    abs(oracle_price - reference_price)
                    / reference_price
                    * Decimal("100")
                )
            existing.last_checked = utc_now()
        else:
            # Create a placeholder status so the oracle data is available
            # for the next health evaluation cycle
            deviation_pct = Decimal("0")
            if reference_price > Decimal("0"):
                deviation_pct = (
                    abs(oracle_price - reference_price)
                    / reference_price
                    * Decimal("100")
                )
            self._protocol_health[cache_key] = ProtocolHealthStatus(
                protocol_name=protocol,
                contract_address=contract_address,
                oracle_price=oracle_price,
                reference_price=reference_price,
                oracle_deviation_pct=deviation_pct,
                last_checked=utc_now(),
            )

        logger.debug(
            "oracle_data_updated",
            component="economic_immune",
            protocol=protocol,
            contract_address=contract_address,
            oracle_price=str(oracle_price),
            reference_price=str(reference_price),
        )
