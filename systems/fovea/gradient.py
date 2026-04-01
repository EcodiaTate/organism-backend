"""
Fovea - Embedding Gradient Attention.

Computes token-level salience attribution for embedding-similarity heads
by analysing the Jacobian ∂similarity/∂token_embedding.  This answers the
question: "which tokens in the percept are *load-bearing* for this salience
decision?"

Two computation strategies are provided:

1. **Analytical Jacobian** (default): Derives ∂cosine/∂token_embedding_i
   from the composition chain: the sentence embedding is a (weighted) mean
   of token embeddings, so the gradient decomposes cleanly.

2. **Numerical Jacobian** (fallback): Finite-difference perturbation for
   arbitrary pooling functions.

Migrated from systems.atune.gradient during Atune → Fovea consolidation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .helpers import cosine_similarity

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .types import GradientAttentionVector


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K = 5
_EPSILON = 1e-6  # Numerical stability
_PERTURBATION = 1e-4  # For numerical Jacobian


# ---------------------------------------------------------------------------
# Analytical Jacobian (mean-pooled sentence embedding)
# ---------------------------------------------------------------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _scale(v: Sequence[float], s: float) -> list[float]:
    return [x * s for x in v]


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=False)]


def compute_gradient_attention_analytical(
    token_embeddings: list[list[float]],
    reference_embedding: list[float],
    *,
    top_k: int = _DEFAULT_TOP_K,
    pooling_weights: list[float] | None = None,
) -> GradientAttentionVector:
    """
    Compute token-level importance via the analytical Jacobian of cosine
    similarity assuming the sentence embedding is a weighted mean of token
    embeddings.
    """
    from .types import GradientAttentionVector

    n_tokens = len(token_embeddings)
    if n_tokens == 0 or not reference_embedding:
        return GradientAttentionVector()

    dim = len(reference_embedding)

    # Default: uniform pooling weights
    if pooling_weights is None:
        w = 1.0 / n_tokens
        weights = [w] * n_tokens
    else:
        weights = list(pooling_weights)

    # Compute sentence embedding (weighted mean)
    sentence = [0.0] * dim
    for i, tok in enumerate(token_embeddings):
        wi = weights[i]
        for d in range(min(dim, len(tok))):
            sentence[d] += wi * tok[d]

    norm_s = _norm(sentence)
    norm_r = _norm(reference_embedding)

    if norm_s < _EPSILON or norm_r < _EPSILON:
        return GradientAttentionVector()

    sim = _dot(sentence, reference_embedding) / (norm_s * norm_r)

    denom = norm_s * norm_s * norm_r
    g_base = [
        (reference_embedding[d] * norm_s - sim * sentence[d]) / denom
        for d in range(dim)
    ]
    g_base_norm = _norm(g_base)

    # Per-token gradient magnitudes
    raw_importances: list[float] = []
    for i in range(n_tokens):
        tok = token_embeddings[i]
        if g_base_norm < _EPSILON:
            raw_importances.append(0.0)
        else:
            tok_projection = abs(_dot(tok, g_base))
            raw_importances.append(weights[i] * tok_projection)

    # Normalise to sum = 1
    total = sum(raw_importances)
    if total < _EPSILON:
        per_token = [1.0 / n_tokens] * n_tokens
    else:
        per_token = [v / total for v in raw_importances]

    # Top-K load-bearing tokens
    indexed = sorted(enumerate(per_token), key=lambda x: x[1], reverse=True)
    load_bearing = [idx for idx, _ in indexed[:min(top_k, n_tokens)]]

    # Detect contradiction tokens
    conflicts: list[int] = []
    for i in range(n_tokens):
        tok = token_embeddings[i]
        if norm_r > _EPSILON:
            r_hat = _scale(reference_embedding, 1.0 / norm_r)
        else:
            r_hat = reference_embedding
        contribution = weights[i] * _dot(tok, r_hat)
        if contribution < -_EPSILON:
            conflicts.append(i)

    return GradientAttentionVector(
        per_token_importance=per_token,
        load_bearing_tokens=load_bearing,
        gradient_magnitude=g_base_norm,
        gradient_direction_conflicts=conflicts,
    )


# ---------------------------------------------------------------------------
# Numerical Jacobian (fallback for non-mean-pool models)
# ---------------------------------------------------------------------------


def compute_gradient_attention_numerical(
    token_embeddings: list[list[float]],
    reference_embedding: list[float],
    pooling_fn: Callable[[list[list[float]]], list[float]],
    *,
    top_k: int = _DEFAULT_TOP_K,
    epsilon: float = _PERTURBATION,
) -> GradientAttentionVector:
    """
    Compute token-level importance via finite-difference Jacobian for
    an arbitrary pooling function.
    """
    from .types import GradientAttentionVector

    n_tokens = len(token_embeddings)
    if n_tokens == 0 or not reference_embedding:
        return GradientAttentionVector()

    dim = len(reference_embedding)

    # Baseline similarity
    baseline_sentence = pooling_fn(token_embeddings)
    baseline_sim = cosine_similarity(baseline_sentence, reference_embedding)

    raw_importances: list[float] = []

    for i in range(n_tokens):
        grad_mag_sq = 0.0
        tok = token_embeddings[i]

        for d in range(min(dim, len(tok))):
            original = tok[d]
            tok[d] = original + epsilon
            perturbed_sentence = pooling_fn(token_embeddings)
            perturbed_sim = cosine_similarity(perturbed_sentence, reference_embedding)
            tok[d] = original  # Restore

            partial = (perturbed_sim - baseline_sim) / epsilon
            grad_mag_sq += partial * partial

        raw_importances.append(math.sqrt(grad_mag_sq))

    # Normalise
    total = sum(raw_importances)
    if total < _EPSILON:
        per_token = [1.0 / n_tokens] * n_tokens
    else:
        per_token = [v / total for v in raw_importances]

    indexed = sorted(enumerate(per_token), key=lambda x: x[1], reverse=True)
    load_bearing = [idx for idx, _ in indexed[:min(top_k, n_tokens)]]

    overall_magnitude = _norm(raw_importances)

    return GradientAttentionVector(
        per_token_importance=per_token,
        load_bearing_tokens=load_bearing,
        gradient_magnitude=overall_magnitude,
        gradient_direction_conflicts=[],  # Not computed in numerical mode
    )


# ---------------------------------------------------------------------------
# High-level API (used by Fovea prediction error dimensions)
# ---------------------------------------------------------------------------


def compute_token_salience(
    token_embeddings: list[list[float]],
    reference_embedding: list[float],
    head_name: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    pooling_weights: list[float] | None = None,
) -> GradientAttentionVector:
    """
    Compute token-level salience attribution for a single similarity
    comparison.  Uses the analytical Jacobian (fast, no extra model calls).

    This is the primary entry point for Fovea prediction error dimensions.
    """
    result = compute_gradient_attention_analytical(
        token_embeddings,
        reference_embedding,
        top_k=top_k,
        pooling_weights=pooling_weights,
    )
    result.head_name = head_name
    return result
