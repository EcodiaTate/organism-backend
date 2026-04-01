"""
EcodiaOS - Exteroceptive Adapters

Concrete adapters that fetch raw data from external APIs and normalise
them into ExteroceptiveReadings. Each adapter follows the same contract:

  async def fetch() -> list[ExteroceptiveReading]

Adapters are fire-and-forget: any exception in a fetch is caught and
logged, and the adapter returns an empty list. The ExteroceptionService
tolerates partial failures gracefully.

Iron Rules (same as ExternalVolatilitySensor):
  - Never called from the theta cycle - runs on its own asyncio timer.
  - All HTTP calls bounded by configurable timeout.
  - No LLM calls, no DB writes, no side-effects beyond readings.
  - Any exception → swallowed, logged, retry at next poll.

Adapters:
  MarketDataAdapter   - CoinGecko (crypto), Alpha Vantage / Yahoo (equities)
  NewsSentimentAdapter - NewsAPI or similar sentiment firehose
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from abc import ABC, abstractmethod
from typing import Any

import structlog

from .types import (
    ExteroceptiveModality,
    ExteroceptiveReading,
    ReadingQuality,
)

logger = structlog.get_logger("systems.soma.exteroception.adapters")


# ─── Base Adapter ─────────────────────────────────────────────────


class BaseExteroceptiveAdapter(ABC):
    """Abstract base for all exteroceptive data adapters."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s

    @abstractmethod
    async def fetch(self) -> list[ExteroceptiveReading]:
        """Fetch and normalise external data into readings.

        Returns an empty list on any failure - never raises.
        """

    @property
    @abstractmethod
    def modalities(self) -> list[ExteroceptiveModality]:
        """Which modalities this adapter provides readings for."""


# ─── Market Data Adapter ─────────────────────────────────────────


# Sigmoid steepness for normalising percentage changes
_PRICE_SIGMOID_K: float = 0.15  # 5% change → ~0.5, 15% → ~0.9
_VOL_SIGMOID_K: float = 0.25    # Volatility normalisation

# CoinGecko public API (no auth, 30 req/min)
_COINGECKO_MARKET_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum,solana"
    "&vs_currencies=usd"
    "&include_24hr_change=true"
    "&include_24hr_vol=true"
    "&include_market_cap=true"
)

# Fear & Greed Index (alternative.me, public, no auth)
_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


class MarketDataAdapter(BaseExteroceptiveAdapter):
    """Fetches crypto market data and fear/greed index.

    Sources:
      1. CoinGecko - BTC, ETH, SOL price change + volume (no auth)
      2. Alternative.me Fear & Greed Index (no auth)

    Normalisation:
      - Price change %: sigmoid-normalised to [-1, 1] where:
          -15% → ~-0.9, -5% → ~-0.5, 0% → 0.0, +5% → +0.5, +15% → +0.9
      - Volatility: derived from abs(24h change) sigmoid to [0, 1]
      - Fear/Greed: linearly mapped from [0, 100] to [-1, +1]
    """

    def __init__(
        self,
        timeout_s: float = 10.0,
        include_fear_greed: bool = True,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        self._include_fear_greed = include_fear_greed

    @property
    def modalities(self) -> list[ExteroceptiveModality]:
        mods = [ExteroceptiveModality.CRYPTO_MARKET]
        if self._include_fear_greed:
            mods.append(ExteroceptiveModality.FEAR_GREED_INDEX)
        return mods

    async def fetch(self) -> list[ExteroceptiveReading]:
        """Fetch crypto prices and optionally fear/greed index."""
        readings: list[ExteroceptiveReading] = []

        # Run fetches concurrently
        tasks: list[asyncio.Task[list[ExteroceptiveReading]]] = [
            asyncio.create_task(self._fetch_crypto()),
        ]
        if self._include_fear_greed:
            tasks.append(asyncio.create_task(self._fetch_fear_greed()))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                readings.extend(result)
            elif isinstance(result, Exception):
                logger.debug("market_adapter_partial_failure", error=str(result))

        return readings

    async def _fetch_crypto(self) -> list[ExteroceptiveReading]:
        """Fetch BTC/ETH/SOL from CoinGecko and produce a single CRYPTO_MARKET reading."""
        data = await _http_get_json(_COINGECKO_MARKET_URL, self._timeout_s)
        if data is None:
            return []

        # Extract 24h changes
        changes: list[float] = []
        for coin in ("bitcoin", "ethereum", "solana"):
            coin_data = data.get(coin, {})
            change = coin_data.get("usd_24h_change")
            if change is not None:
                changes.append(float(change))

        if not changes:
            return []

        # Mean percentage change → normalised value
        mean_change = sum(changes) / len(changes)
        normalised_value = _signed_sigmoid(mean_change, k=_PRICE_SIGMOID_K)

        # Volatility: mean absolute change
        mean_abs = sum(abs(c) for c in changes) / len(changes)
        volatility = _unsigned_sigmoid(mean_abs, k=_VOL_SIGMOID_K)

        return [
            ExteroceptiveReading(
                modality=ExteroceptiveModality.CRYPTO_MARKET,
                value=normalised_value,
                volatility=volatility,
                quality=ReadingQuality.FRESH,
                raw_metadata={
                    "btc_24h_pct": changes[0] if len(changes) > 0 else 0.0,
                    "eth_24h_pct": changes[1] if len(changes) > 1 else 0.0,
                    "sol_24h_pct": changes[2] if len(changes) > 2 else 0.0,
                    "mean_change_pct": round(mean_change, 3),
                },
            )
        ]

    async def _fetch_fear_greed(self) -> list[ExteroceptiveReading]:
        """Fetch Fear & Greed Index from alternative.me."""
        data = await _http_get_json(_FEAR_GREED_URL, self._timeout_s)
        if data is None:
            return []

        fng_data = data.get("data", [])
        if not fng_data:
            return []

        raw_value = int(fng_data[0].get("value", 50))
        classification = fng_data[0].get("value_classification", "Neutral")

        # Map [0, 100] to [-1, +1]: 0=Extreme Fear → -1.0, 100=Extreme Greed → +1.0
        normalised = (raw_value - 50.0) / 50.0

        # Volatility: distance from neutral (50). Max at extremes.
        volatility = abs(raw_value - 50.0) / 50.0

        return [
            ExteroceptiveReading(
                modality=ExteroceptiveModality.FEAR_GREED_INDEX,
                value=normalised,
                volatility=volatility,
                quality=ReadingQuality.FRESH,
                raw_metadata={
                    "raw_index": raw_value,
                    "classification": classification,
                },
            )
        ]


# ─── News Sentiment Adapter ──────────────────────────────────────


class NewsSentimentAdapter(BaseExteroceptiveAdapter):
    """Fetches news sentiment data and produces NEWS_SENTIMENT readings.

    This adapter is designed to work with any sentiment API that returns
    a JSON payload with article-level sentiment scores. The default
    implementation uses a simple HTTP endpoint configurable via URL.

    Normalisation:
      - Aggregates per-article sentiment [-1, 1] into a mean score
      - Volatility derived from standard deviation of sentiment scores
      - Optionally filters by topic keywords (e.g., "AI", "crypto")

    For production, wire this to your preferred sentiment provider:
      - NewsAPI.org (free tier: 100 req/day)
      - Aylien News API
      - Event Registry
      - Custom RSS + local sentiment model
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        topic_keywords: list[str] | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        self._api_url = api_url
        self._api_key = api_key
        self._topic_keywords = topic_keywords or ["AI", "artificial intelligence", "crypto", "technology"]

    @property
    def modalities(self) -> list[ExteroceptiveModality]:
        return [ExteroceptiveModality.NEWS_SENTIMENT]

    async def fetch(self) -> list[ExteroceptiveReading]:
        """Fetch news sentiment. Returns empty if no API configured."""
        if not self._api_url:
            return []

        try:
            return await self._fetch_sentiment()
        except Exception as exc:
            logger.debug("news_sentiment_fetch_failed", error=str(exc))
            return []

    async def _fetch_sentiment(self) -> list[ExteroceptiveReading]:
        """Fetch from configured sentiment API and normalise."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["X-Api-Key"] = self._api_key

        data = await _http_get_json(
            self._api_url,
            self._timeout_s,
            extra_headers=headers,
        )
        if data is None:
            return []

        # Extract sentiment scores - adapt to your API's response format
        sentiments = self._extract_sentiments(data)
        if not sentiments:
            return [
                ExteroceptiveReading(
                    modality=ExteroceptiveModality.NEWS_SENTIMENT,
                    value=0.0,
                    volatility=0.0,
                    quality=ReadingQuality.DEGRADED,
                )
            ]

        # Aggregate
        mean_sentiment = sum(sentiments) / len(sentiments)
        # Volatility = standard deviation, normalised to [0, 1]
        if len(sentiments) > 1:
            variance = sum((s - mean_sentiment) ** 2 for s in sentiments) / len(sentiments)
            std_dev = math.sqrt(variance)
            # Std dev of [-1,1] values maxes out around 1.0
            volatility = min(std_dev, 1.0)
        else:
            volatility = 0.0

        return [
            ExteroceptiveReading(
                modality=ExteroceptiveModality.NEWS_SENTIMENT,
                value=max(-1.0, min(1.0, mean_sentiment)),
                volatility=volatility,
                quality=ReadingQuality.FRESH,
                raw_metadata={
                    "article_count": len(sentiments),
                    "mean_sentiment": round(mean_sentiment, 4),
                    "std_dev": round(volatility, 4),
                },
            )
        ]

    def _extract_sentiments(self, data: dict[str, Any]) -> list[float]:
        """Extract sentiment scores from API response.

        Supports common response formats:
          - {"articles": [{"sentiment": 0.5, ...}, ...]}
          - {"results": [{"score": 0.5, ...}, ...]}
          - {"sentiment": {"score": 0.5}}
          - {"data": [{"sentiment_score": 0.5, ...}, ...]}
        """
        sentiments: list[float] = []

        # Try common response shapes
        articles = (
            data.get("articles")
            or data.get("results")
            or data.get("data")
            or []
        )

        if isinstance(articles, list):
            for article in articles:
                if not isinstance(article, dict):
                    continue
                score = (
                    article.get("sentiment")
                    or article.get("score")
                    or article.get("sentiment_score")
                    or article.get("polarity")
                )
                if score is not None:
                    try:
                        sentiments.append(float(score))
                    except (ValueError, TypeError):
                        continue

        # Fallback: top-level aggregate score
        if not sentiments:
            top_sentiment = data.get("sentiment", {})
            if isinstance(top_sentiment, dict):
                score = top_sentiment.get("score") or top_sentiment.get("value")
                if score is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        sentiments.append(float(score))
            elif isinstance(top_sentiment, (int, float)):
                sentiments.append(float(top_sentiment))

        return sentiments


# ─── HTTP Utility ─────────────────────────────────────────────────


async def _http_get_json(
    url: str,
    timeout_s: float,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Fire-and-forget HTTP GET returning parsed JSON, or None on any error.

    Uses urllib (stdlib) in a thread executor to avoid blocking the event
    loop. Matches the pattern established by ExternalVolatilitySensor.
    """
    import json as _json
    import urllib.request

    loop = asyncio.get_running_loop()

    def _do_request() -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "EcodiaOS/1.0",
        }
        if extra_headers:
            headers.update(extra_headers)

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
        return _json.loads(body)  # type: ignore[no-any-return]

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _do_request),
            timeout=timeout_s + 2.0,
        )
    except Exception as exc:
        logger.debug("http_get_failed", url=url[:80], error=str(exc))
        return None


# ─── Normalisation Utilities ──────────────────────────────────────


def _signed_sigmoid(pct: float, k: float = 0.15) -> float:
    """Map a percentage change to [-1, +1] via a symmetric sigmoid.

    Preserves sign: positive % → positive output, negative % → negative.
    k controls steepness: k=0.15 → 5% ≈ ±0.5, 15% ≈ ±0.9.
    """
    # Use tanh which naturally maps to [-1, 1]
    return math.tanh(k * pct)


def _unsigned_sigmoid(abs_val: float, k: float = 0.25) -> float:
    """Map an absolute value to [0, 1] via logistic sigmoid.

    0 → 0.0, midpoint at 1/k, large values → ~1.0.
    """
    raw = 1.0 / (1.0 + math.exp(-k * abs_val))
    # Rescale from (0.5, 1.0) to (0.0, 1.0)
    return (raw - 0.5) * 2.0
