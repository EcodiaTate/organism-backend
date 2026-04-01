"""
EcodiaOS - Axon Market Pattern Detector

Extracts pattern signatures from market data percepts and queries Equor's
template library for fast-path eligibility.

This is the entry point of the Arbitrage Reflex Arc:
  1. Atune receives a market data percept (price feed, spread alert, etc.)
  2. MarketPatternDetector extracts a feature vector (pattern_signature)
  3. It queries the TemplateLibrary for a matching ConstitutionalTemplate
  4. If matched (confidence > 0.9), it emits a FastPathIntent
  5. The FastPathIntent is routed directly to Axon, bypassing Nova

Latency budget: ≤50ms total (30ms extraction + 20ms template lookup)

The pattern extraction strategy uses deterministic feature hashing - no LLM,
no embedding, no I/O. Pure CPU math on structured market data fields.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from primitives.fast_path import FastPathIntent

if TYPE_CHECKING:
    from systems.equor.template_library import TemplateLibrary

logger = structlog.get_logger()

# Market data percept metadata keys (populated by the market data ingestion channel)
_KEY_SPREAD_BPS = "spread_bps"
_KEY_POOLS = "pools"
_KEY_TOKEN_PAIR = "token_pair"
_KEY_PRICE = "price"
_KEY_VOLUME_24H = "volume_24h"
_KEY_PROTOCOL = "protocol"
_KEY_CHAIN = "chain"
_KEY_SPREAD_THRESHOLD = "spread_threshold"
_KEY_ARBITRAGE_TYPE = "arbitrage_type"


class MarketPatternDetector:
    """
    Extracts structured pattern signatures from market data percepts
    and matches them against Equor's pre-approved template library.

    Stateless - all state lives in the TemplateLibrary.
    """

    def __init__(self, template_library: TemplateLibrary) -> None:
        self._templates = template_library
        self._logger = logger.bind(system="axon.market_pattern")

        # Metrics
        self._percepts_processed: int = 0
        self._fast_path_intents_emitted: int = 0

    def detect(
        self,
        percept_id: str,
        content: str,
        metadata: dict[str, Any],
        percept_received_at: float,
    ) -> FastPathIntent | None:
        """
        Attempt to match a market data percept against pre-approved templates.

        Args:
            percept_id: The Percept's ULID.
            content: Raw text content of the percept.
            metadata: Structured metadata from the market data channel.
            percept_received_at: Monotonic timestamp when Atune received the percept.

        Returns:
            FastPathIntent if a matching template is found, None otherwise.
            Budget: ≤50ms total.
        """
        start = time.monotonic()
        self._percepts_processed += 1

        # Step 1: Extract pattern signature from metadata (≤30ms)
        signature = self._extract_signature(content, metadata)
        if not signature:
            return None

        extraction_ms = int((time.monotonic() - start) * 1000)

        # Step 2: Query template library for match (≤20ms)
        template = self._templates.find_match(signature)
        if template is None:
            return None

        match_ms = int((time.monotonic() - start) * 1000) - extraction_ms

        # Step 3: Build FastPathIntent
        now = utc_now()
        intent = FastPathIntent(
            template_id=template.template_id,
            pattern_signature=signature,
            executor_type=self._infer_executor(template, metadata),
            execution_params=self._build_execution_params(template, signature, metadata),
            max_capital=template.max_capital_per_execution,
            approval_confidence=template.approval_confidence,
            source_percept_id=percept_id,
            percept_received_at=now,
            template_matched_at=now,
        )

        total_ms = int((time.monotonic() - start) * 1000)
        self._fast_path_intents_emitted += 1

        self._logger.info(
            "fast_path_intent_emitted",
            template_id=template.template_id,
            percept_id=percept_id,
            extraction_ms=extraction_ms,
            match_ms=match_ms,
            total_ms=total_ms,
            executor=intent.executor_type,
            max_capital=template.max_capital_per_execution,
        )

        return intent

    def _extract_signature(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Extract a pattern signature from market data metadata.

        Returns None if the metadata doesn't contain enough market data
        to form a meaningful signature.
        """
        signature: dict[str, Any] = {}

        if _KEY_SPREAD_BPS in metadata:
            spread_bps = metadata[_KEY_SPREAD_BPS]
            if isinstance(spread_bps, (int, float)):
                signature["spread_threshold"] = spread_bps / 100.0

        if _KEY_POOLS in metadata:
            pools = metadata[_KEY_POOLS]
            if isinstance(pools, list) and pools:
                signature["pools"] = pools

        if _KEY_TOKEN_PAIR in metadata:
            signature["token_pair"] = metadata[_KEY_TOKEN_PAIR]

        if _KEY_PROTOCOL in metadata:
            signature["protocol"] = metadata[_KEY_PROTOCOL]

        if _KEY_CHAIN in metadata:
            signature["chain"] = metadata[_KEY_CHAIN]

        if _KEY_ARBITRAGE_TYPE in metadata:
            signature["arbitrage_type"] = metadata[_KEY_ARBITRAGE_TYPE]

        if _KEY_SPREAD_THRESHOLD in metadata:
            signature["spread_threshold"] = metadata[_KEY_SPREAD_THRESHOLD]

        if "spread_threshold" not in signature and "token_pair" not in signature:
            return None

        return signature

    def _infer_executor(
        self,
        template: Any,
        metadata: dict[str, Any],
    ) -> str:
        """Determine which Axon executor to invoke for this template."""
        explicit = template.pattern_signature.get("executor")
        if isinstance(explicit, str) and explicit:
            return explicit

        arb_type = metadata.get(_KEY_ARBITRAGE_TYPE, "")
        if arb_type in ("triangular", "cross_pool", "flash_loan"):
            return "defi_yield"
        if arb_type == "transfer":
            return "wallet_transfer"

        return "defi_yield"

    def _build_execution_params(
        self,
        template: Any,
        signature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the execution parameters dict for the Axon executor."""
        params: dict[str, Any] = {}

        params["protocol"] = signature.get("protocol", metadata.get(_KEY_PROTOCOL, "aave"))
        params["action"] = "deposit"

        spread = signature.get("spread_threshold", 0.0)
        if spread > 0 and template.max_capital_per_execution > 0:
            scale = min(1.0, spread / 1.0)
            params["amount"] = str(round(template.max_capital_per_execution * scale, 2))
        else:
            params["amount"] = str(round(template.max_capital_per_execution, 2))

        if "pools" in signature:
            params["pools"] = signature["pools"]
        if "chain" in signature:
            params["chain"] = signature["chain"]
        if "token_pair" in metadata:
            params["token_pair"] = metadata["token_pair"]

        params["execution_path"] = "fast_path"
        params["template_id"] = template.template_id

        return params

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "percepts_processed": self._percepts_processed,
            "fast_path_intents_emitted": self._fast_path_intents_emitted,
            "emission_rate": (
                self._fast_path_intents_emitted / max(1, self._percepts_processed)
            ),
        }
