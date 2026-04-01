"""
EcodiaOS - Embedding Client Abstraction

Supports local sentence-transformers, API-based, sidecar, and
Voyage AI models (voyage-code-3 for code-aware embeddings).

Dimension is model-dependent:
  - sentence-transformers/all-mpnet-base-v2: 768
  - voyage-code-3: 1024
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import structlog

if TYPE_CHECKING:
    from config import EmbeddingConfig

logger = structlog.get_logger()


class EmbeddingClient(ABC):
    """Abstract interface for text embedding."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns a vector of configured dimension."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class LocalEmbeddingClient(EmbeddingClient):
    """
    Local embedding using sentence-transformers.
    Loaded in-process. Best for latency and privacy.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _load_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is None:
            import warnings

            from sentence_transformers import SentenceTransformer

            # Suppress HF Hub auth warning if HF_TOKEN is not set
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
                self._model = SentenceTransformer(self._model_name, device=self._device)

            logger.info(
                "embedding_model_loaded",
                model=self._model_name,
                device=self._device,
                dimension=self._model.get_sentence_embedding_dimension(),
            )

    async def embed(self, text: str) -> list[float]:
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()  # type: ignore[no-any-return]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()  # type: ignore[no-any-return]

    async def close(self) -> None:
        self._model = None


class SidecarEmbeddingClient(EmbeddingClient):
    """
    Embedding via HTTP sidecar service.
    For when you want the model in a separate process/container.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> list[float]:
        response = await self._client.post(
            f"{self._url}/embed",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]  # type: ignore[no-any-return]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            f"{self._url}/embed_batch",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]  # type: ignore[no-any-return]

    async def close(self) -> None:
        await self._client.aclose()


class VoyageEmbeddingClient(EmbeddingClient):
    """
    Voyage AI code embedding client (voyage-code-3).

    Optimized for code: captures semantic meaning of code snippets,
    function signatures, variable names, and documentation.

    voyage-code-3 produces 1024-dimensional embeddings.
    Rate limit: 10,000 tokens per request, 300 RPM.
    """

    # Retry configuration
    _MAX_RETRIES = 3
    _BASE_DELAY_S = 1.0
    _RETRYABLE_STATUS_CODES = {429, 503}

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-code-3",
        dimension: int = 1024,
        max_batch_size: int = 128,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._max_batch_size = max_batch_size
        clean_key = api_key.strip()
        self._client = httpx.AsyncClient(
            base_url="https://api.voyageai.com/v1",
            headers={
                "Authorization": f"Bearer {clean_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        logger.info(
            "voyage_embedding_client_initialized",
            model=model,
            dimension=dimension,
        )

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff on retryable status codes."""
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                response = await self._client.post("/embeddings", json=payload)
                if response.status_code in self._RETRYABLE_STATUS_CODES:
                    if attempt < self._MAX_RETRIES:
                        delay = self._BASE_DELAY_S * (2 ** attempt)
                        retry_after = response.headers.get("retry-after")
                        if retry_after:
                            with contextlib.suppress(ValueError):
                                delay = max(delay, float(retry_after))
                        logger.warning(
                            "voyage_retrying",
                            status=response.status_code,
                            attempt=attempt + 1,
                            delay_s=round(delay, 1),
                        )
                        await asyncio.sleep(delay)
                        continue
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(self._BASE_DELAY_S * (2 ** attempt))
                    continue
                raise
        raise last_exc or RuntimeError("Voyage embedding request failed after retries")

    async def embed(self, text: str) -> list[float]:
        payload = {
            "model": self._model,
            "input": [text],
            "input_type": "document",
        }
        data = await self._post_with_retry(payload)
        embeddings = data.get("data", [])
        if not embeddings:
            raise ValueError("Voyage API returned no embeddings")
        return embeddings[0]["embedding"]  # type: ignore[no-any-return]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), self._max_batch_size):
            batch = texts[i : i + self._max_batch_size]
            payload = {
                "model": self._model,
                "input": batch,
                "input_type": "document",
            }
            data = await self._post_with_retry(payload)
            batch_embeddings = data.get("data", [])

            # Sort by index to ensure order matches input
            batch_embeddings.sort(key=lambda x: x.get("index", 0))
            all_embeddings.extend(e["embedding"] for e in batch_embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query. Voyage recommends using input_type="query"
        for retrieval queries vs "document" for stored texts.
        """
        payload = {
            "model": self._model,
            "input": [query],
            "input_type": "query",
        }
        data = await self._post_with_retry(payload)
        embeddings = data.get("data", [])
        if not embeddings:
            raise ValueError("Voyage API returned no embeddings for query")
        return embeddings[0]["embedding"]  # type: ignore[no-any-return]

    async def close(self) -> None:
        await self._client.aclose()


class MockEmbeddingClient(EmbeddingClient):
    """
    Mock embedding client for testing and development.
    Returns random normalised vectors of the correct dimension.
    """

    def __init__(self, dimension: int = 768) -> None:
        self._dimension = dimension

    async def embed(self, text: str) -> list[float]:
        vec = np.random.randn(self._dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()  # type: ignore[no-any-return]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        for _ in texts:
            vec = np.random.randn(self._dimension).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            results.append(vec.tolist())
        return results

    async def close(self) -> None:
        pass


def create_embedding_client(config: EmbeddingConfig) -> EmbeddingClient:
    """Factory to create the configured embedding client."""
    if config.strategy == "local":
        return LocalEmbeddingClient(
            model_name=config.local_model,
            device=config.local_device,
        )
    elif config.strategy == "sidecar":
        if not config.sidecar_url:
            raise ValueError("Sidecar strategy requires sidecar_url in config")
        return SidecarEmbeddingClient(url=config.sidecar_url)
    elif config.strategy == "mock":
        return MockEmbeddingClient(dimension=config.dimension)
    else:
        raise ValueError(f"Unknown embedding strategy: {config.strategy}")


def create_voyage_client(api_key: str, model: str = "voyage-code-3") -> VoyageEmbeddingClient:
    """Create a Voyage AI embedding client for code-aware embeddings."""
    return VoyageEmbeddingClient(api_key=api_key, model=model)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))
