"""
EcodiaOS - EIS Embedding & Token Histogram Utilities

Two complementary text representations for antigenic similarity:

1. **Token histogram** (deterministic, <1ms): Captures lexical
   distribution via whitespace-split + lowered tokens. The histogram
   vector is a sparse, fixed-length representation suitable for fast
   cosine similarity. Effective at catching copy-paste attacks where
   the same tokens recur across attempts.

2. **Semantic embedding** (model-dependent, <5ms with local model):
   Uses the shared EmbeddingClient to produce a dense semantic vector.
   Captures meaning-level similarity between known pathogens and new
   inputs. This is the heavyweight antigenic signal.

Both representations are combined in AntigenicSignature for multi-vector
search in the pathogen store.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from systems.eis.models import (
    AntigenicSignature,
    EISConfig,
    StructuralProfile,
    TokenHistogram,
)
from systems.eis.structural_features import (
    structural_profile_hash,
    structural_profile_to_vector,
)

if TYPE_CHECKING:
    from clients.embedding import EmbeddingClient


# ─── Token Histogram ─────────────────────────────────────────────


def compute_token_histogram(
    text: str,
    top_k: int = 256,
) -> TokenHistogram:
    """
    Compute a token frequency histogram from whitespace-split text.

    Tokens are lowered and stripped of leading/trailing punctuation.
    Returns normalised frequencies for the top-k most common tokens.

    This is intentionally simple - not a proper tokenizer. The goal
    is a cheap, deterministic fingerprint that captures lexical
    distribution, not semantic meaning.

    Performance: <1ms for 100K character inputs.
    """
    if not text:
        return TokenHistogram()

    # Tokenise: split on whitespace, lower, strip punctuation.
    # Sample first 20K chars to keep latency under 1ms on large inputs.
    sample = text[:20_000]
    raw_tokens = sample.lower().split()
    tokens: list[str] = []
    for tok in raw_tokens:
        cleaned = tok.strip(".,;:!?\"'`()[]{}/<>*#@~-_=+|\\")
        if len(cleaned) >= 2:  # Skip single-char tokens (noise)
            tokens.append(cleaned)

    if not tokens:
        return TokenHistogram()

    total = len(tokens)
    counter = Counter(tokens)
    vocabulary_size = len(counter)

    # Hapax legomena: tokens appearing exactly once
    hapax_count = sum(1 for count in counter.values() if count == 1)
    hapax_ratio = hapax_count / vocabulary_size if vocabulary_size > 0 else 0.0

    # Top-k frequencies, normalised
    top_items = counter.most_common(top_k)
    frequencies = {tok: count / total for tok, count in top_items}

    return TokenHistogram(
        frequencies=frequencies,
        total_tokens=total,
        vocabulary_size=vocabulary_size,
        hapax_ratio=round(hapax_ratio, 4),
    )


def histogram_to_vector(
    histogram: TokenHistogram,
    dim: int = 64,
) -> list[float]:
    """
    Convert a TokenHistogram into a fixed-length normalised vector.

    Uses feature hashing (hashing trick) to map arbitrary token
    strings into a fixed-dimension vector space. This allows cosine
    similarity comparison between histograms of different vocabulary.

    The hashing trick: hash(token) mod dim -> bucket index, with
    sign determined by a second hash. This preserves inner product
    in expectation (Johnson-Lindenstrauss-like guarantee).
    """
    vec = np.zeros(dim, dtype=np.float32)

    for token, freq in histogram.frequencies.items():
        # Primary hash -> bucket index
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)  # noqa: S324
        bucket = h % dim

        # Sign hash (use different bytes of the same hash)
        sign = 1.0 if (h >> 32) & 1 == 0 else -1.0

        vec[bucket] += sign * freq

    # L2 normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


# ─── Antigenic Signature Assembly ─────────────────────────────────


async def compute_antigenic_signature(
    text: str,
    structural_profile: StructuralProfile,
    token_histogram: TokenHistogram,
    embed_client: EmbeddingClient | None = None,
    config: EISConfig | None = None,
) -> AntigenicSignature:
    """
    Assemble the full multi-vector antigenic signature.

    Combines three representations:
    1. Structural vector (from StructuralProfile) - deterministic, <1ms
    2. Histogram vector (from TokenHistogram) - deterministic, <1ms
    3. Semantic vector (from EmbeddingClient) - model-dependent, <5ms

    If embed_client is None, semantic_vector is left empty (graceful
    degradation - similarity search uses only structural + histogram).

    Total budget for this function: <7ms (structural + histogram are
    already computed; this just converts + embeds).
    """
    cfg = config or EISConfig()

    # 1. Structural vector
    structural_vec = structural_profile_to_vector(
        structural_profile,
        dim=cfg.structural_vector_dim,
    )
    struct_hash = structural_profile_hash(structural_profile)

    # 2. Histogram vector
    histogram_vec = histogram_to_vector(
        token_histogram,
        dim=cfg.histogram_vector_dim,
    )

    # 3. Semantic embedding (async, may be skipped)
    semantic_vec: list[float] = []
    dimension = 0

    if embed_client is not None and cfg.embedding_enabled:
        # Truncate text for embedding - most models have token limits
        embed_text = text[:8000]
        semantic_vec = await embed_client.embed(embed_text)
        dimension = len(semantic_vec)

    return AntigenicSignature(
        structural_hash=struct_hash,
        structural_vector=structural_vec,
        histogram_vector=histogram_vec,
        semantic_vector=semantic_vec,
        dimension=dimension,
    )


# ─── Pathogen Fingerprinting ─────────────────────────────────────


def compute_pathogen_fingerprint(
    text: str,
    structural_hash: str,
) -> str:
    """
    Compute a stable fingerprint for pathogen deduplication.

    Combines a content hash (first 2K chars) with the structural hash.
    Two inputs with the same content prefix AND structural shape get
    the same fingerprint. This deduplicates repeated attack attempts
    while distinguishing genuinely different inputs.
    """
    content_sample = text[:2000].lower().strip()
    content_hash = hashlib.sha256(content_sample.encode()).hexdigest()[:12]
    return f"{content_hash}:{structural_hash[:8]}"


# ─── Fast-path Composite Score ────────────────────────────────────


def compute_composite_threat_score(
    innate_score: float,
    structural_anomaly_score: float,
    histogram_similarity: float,
    semantic_similarity: float,
    config: EISConfig | None = None,
) -> float:
    """
    Compute weighted composite threat score from all fast-path signals.

    Each input is a 0.0-1.0 score from its respective detection layer.
    Weights are configurable via EISConfig. Returns a 0.0-1.0 composite.

    If semantic_similarity is 0.0 (no embedding available), its weight
    is redistributed proportionally to other signals.
    """
    cfg = config or EISConfig()

    weights = {
        "innate": cfg.innate_weight,
        "structural": cfg.structural_weight,
        "histogram": cfg.histogram_weight,
        "semantic": cfg.semantic_weight,
    }
    scores = {
        "innate": innate_score,
        "structural": structural_anomaly_score,
        "histogram": histogram_similarity,
        "semantic": semantic_similarity,
    }

    # If semantic is unavailable, redistribute its weight
    if semantic_similarity <= 0.0 and cfg.semantic_weight > 0:
        remaining_weight = 1.0 - cfg.semantic_weight
        if remaining_weight > 0:
            scale = 1.0 / remaining_weight
            weights["innate"] *= scale
            weights["structural"] *= scale
            weights["histogram"] *= scale
        weights["semantic"] = 0.0

    composite = sum(weights[k] * scores[k] for k in weights)
    return max(0.0, min(1.0, composite))


def compute_structural_anomaly_score(profile: StructuralProfile) -> float:
    """
    Score how anomalous a structural profile is relative to normal text.

    Heuristic scoring based on features that are rare in benign input
    but common in adversarial input. Returns 0.0-1.0.

    This is NOT a learned model - it's a hand-tuned scoring function
    that captures known adversarial structural patterns. It serves as
    a baseline until enough labelled data exists for a proper classifier.
    """
    score = 0.0

    # Very low alpha ratio suggests non-text content (encoding, binary)
    if profile.alpha_ratio < 0.3:
        score += 0.3

    # High special char ratio suggests encoding evasion or injection
    if profile.special_char_ratio > 0.1:
        score += 0.2

    # Very high repetition
    if profile.repetition_score > 0.6:
        score += 0.2

    # Deep bracket nesting (template injection, structured payloads)
    if profile.bracket_depth > 10:
        score += 0.15

    # High delimiter density
    if profile.delimiter_density > 15.0:
        score += 0.1

    # Very low entropy (repetitive content)
    if profile.entropy < 2.0 and profile.char_count > 100:
        score += 0.15

    # Very high entropy (random / encrypted content)
    if profile.entropy > 6.5:
        score += 0.1

    # Mixed scripts (homoglyph attacks)
    if profile.mixed_script:
        score += 0.15

    # Extremely long single line (obfuscation)
    if profile.max_line_length > 2000:
        score += 0.1

    return max(0.0, min(1.0, score))
