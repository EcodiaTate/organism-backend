"""
Atune — Embedding Gradient Attention.

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

The module is designed to be called *after* a salience head has computed its
similarity score, using the same embedding pair, so no redundant computation
occurs.
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

    Given:
        s = pooled_embedding = Σ_i w_i · t_i  (weighted mean of tokens)
        sim = cosine(s, r) = (s · r) / (||s|| · ||r||)

    The gradient w.r.t. token i's embedding is:
        ∂sim/∂t_i = w_i / (||s|| · ||r||) · (r - sim · s / ||s||²  · ||s||)

    But we only care about the *magnitude* of this gradient per token (scalar
    importance), not the full vector.  The magnitude tells us how much
    perturbing token i would change the similarity score.

    Parameters
    ----------
    token_embeddings:
        Per-token embeddings, shape (n_tokens, dim).
    reference_embedding:
        The reference embedding being compared against (e.g. goal, risk
        category, identity).
    top_k:
        Number of top load-bearing tokens to return.
    pooling_weights:
        Optional per-token weights for the pooling function.  Defaults to
        uniform (1/n).
    """
    from .types import GradientAttentionVector as GAV

    n_tokens = len(token_embeddings)
    if n_tokens == 0 or not reference_embedding:
        return GAV()

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
        return GAV()

    sim = _dot(sentence, reference_embedding) / (norm_s * norm_r)

    # ∂sim/∂t_i = w_i · [r / (||s|| · ||r||) - sim · s / (||s||² · ||r||)]
    #
    # The gradient magnitude for token i is ||∂sim/∂t_i||.
    # Factor out the common term:
    #   g_base = r / (||s|| · ||r||) - sim · s / (||s||² · ||r||)
    #          = [r · ||s|| - sim · s] / (||s||² · ||r||)
    denom = norm_s * norm_s * norm_r
    g_base = [
        (reference_embedding[d] * norm_s - sim * sentence[d]) / denom
        for d in range(dim)
    ]
    g_base_norm = _norm(g_base)

    # Per-token gradient magnitudes
    raw_importances: list[float] = []
    for i in range(n_tokens):
        # ||∂sim/∂t_i|| = |w_i| · ||g_base|| (since g_base is shared)
        # But token content also modulates: the actual gradient depends on
        # the token's alignment with g_base direction.
        tok = token_embeddings[i]
        # Project token onto gradient direction for a richer signal:
        # importance_i = w_i * |tok · g_base| (how much this token contributes
        # along the gradient direction)
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

    # Detect contradiction tokens: gradient direction opposes reference
    # A token "disagrees" if its contribution to the sentence embedding
    # moves *away* from the reference.
    conflicts: list[int] = []
    for i in range(n_tokens):
        tok = token_embeddings[i]
        # Token's contribution to similarity: w_i * (tok · r_hat)
        # If negative, this token pulls the sentence embedding away from ref.
        r_hat = _scale(reference_embedding, 1.0 / norm_r) if norm_r > _EPSILON else reference_embedding
        contribution = weights[i] * _dot(tok, r_hat)
        if contribution < -_EPSILON:
            conflicts.append(i)

    return GAV(
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

    For each token i, perturb its embedding along each dimension and
    measure the change in cosine similarity.  This is O(n_tokens × dim)
    similarity evaluations — use sparingly.

    Parameters
    ----------
    pooling_fn:
        Callable that takes ``list[list[float]]`` (token embeddings) and
        returns ``list[float]`` (sentence embedding).
    """
    from .types import GradientAttentionVector as GAV

    n_tokens = len(token_embeddings)
    if n_tokens == 0 or not reference_embedding:
        return GAV()

    dim = len(reference_embedding)

    # Baseline similarity
    baseline_sentence = pooling_fn(token_embeddings)
    baseline_sim = cosine_similarity(baseline_sentence, reference_embedding)

    raw_importances: list[float] = []

    for i in range(n_tokens):
        grad_mag_sq = 0.0
        tok = token_embeddings[i]

        for d in range(min(dim, len(tok))):
            # Perturb token i dimension d
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

    return GAV(
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

    Parameters
    ----------
    token_embeddings:
        Per-token dense embeddings from ``Percept.content.token_embeddings``.
    reference_embedding:
        The reference vector (goal target, risk category, identity, etc.).
    head_name:
        Name of the calling salience head (for attribution tracking).
    top_k:
        Number of top load-bearing token indices to return.
    pooling_weights:
        Optional per-token weights.  ``None`` = uniform mean-pool.
    """
    result = compute_gradient_attention_analytical(
        token_embeddings,
        reference_embedding,
        top_k=top_k,
        pooling_weights=pooling_weights,
    )
    result.head_name = head_name
    return result
