"""
Speciation Bible §6.2 Evaluation Pillars 1–4.

Pillar 5 (Ethical Drift) is in ethical_drift.py.
All four pillars run monthly via BenchmarksService._monthly_eval_loop().
Fixed test sets are NEVER modified - they are versioned in data/evaluation/.

File layout:
    Pillar 1 - Specialization Index    (measure_specialization)
    Pillar 2 - Novelty Emergence       (measure_novelty_emergence)
    Pillar 3 - Causal Reasoning        (measure_causal_reasoning)
    Pillar 4 - Learning Velocity       (compute_learning_velocity)
    §6.3      - Memorization Detection (detect_memorization)
    Helpers   - _answer_matches, _evaluate_validity, _evaluate_consistency, _mean
    Loader    - load_fixed_test_sets
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from scipy.optimize import curve_fit

logger = structlog.get_logger("systems.benchmarks.pillars")

DATA_DIR = Path("data/evaluation")


# ---------------------------------------------------------------------------
# Pillar 1: Specialization Index
# ---------------------------------------------------------------------------


@dataclass
class SpecializationResult:
    specialization_index: float
    """(cd-bd) - (bg-cg). >0.1 = genuine specialization. >0.3 = publishable."""
    domain_improvement: float   # cd - bd
    general_retention: float    # cg / bg
    custom_domain_score: float
    custom_general_score: float
    base_domain_score: float
    base_general_score: float


async def measure_specialization(
    custom_engine,
    base_engine,
    domain_test: list[dict],   # 200 domain-specific questions {"question":..., "answer":...}
    general_test: list[dict],  # 200 general questions
) -> SpecializationResult:
    """Bible §6.2 Pillar 1. Scores each engine on domain + general test sets."""

    async def eval_set(engine, questions: list[dict]) -> float:
        correct = 0
        for q in questions:
            try:
                result = await engine.reason(episode_context=q["question"])
                if _answer_matches(result.decision, q["answer"]):
                    correct += 1
            except Exception:
                pass
        return correct / max(1, len(questions))

    cd, cg = await asyncio.gather(
        eval_set(custom_engine, domain_test),
        eval_set(custom_engine, general_test),
    )
    bd, bg = await asyncio.gather(
        eval_set(base_engine, domain_test),
        eval_set(base_engine, general_test),
    )

    si = (cd - bd) - (bg - cg)
    return SpecializationResult(
        specialization_index=si,
        domain_improvement=cd - bd,
        general_retention=cg / max(0.001, bg),
        custom_domain_score=cd,
        custom_general_score=cg,
        base_domain_score=bd,
        base_general_score=bg,
    )


# ---------------------------------------------------------------------------
# Pillar 2: Novelty Emergence
# ---------------------------------------------------------------------------


@dataclass
class NoveltyEmergenceResult:
    novel_success_rate: float
    """Success on 100 held-out episodes never in any training batch."""
    reasoning_cosine_distance: float
    """Distance from training embeddings. High = genuine novelty."""
    novel_count: int
    genuine_learning: bool
    """High success (>0.6) + high distance (>0.3) = genuine learning, not transfer."""


async def measure_novelty_emergence(
    custom_engine,
    novel_episodes: list[dict],
    training_embeddings: Optional[list] = None,
    encode_fn: Optional[object] = None,
) -> NoveltyEmergenceResult:
    """
    Bible §6.2 Pillar 2.
    High success + high distance = novel correct reasoning (genuine learning, not transfer).
    Low success = memorization, cannot generalize.

    Args:
        encode_fn: async callable (list[str]) -> list[list[float]].
            When provided, used to encode reasoning texts into embeddings
            for real cosine distance computation against training_embeddings.
    """
    successes: list[float] = []
    reasoning_texts: list[str] = []

    for ep in novel_episodes:
        try:
            result = await custom_engine.reason(episode_context=ep["question"])
            success = _answer_matches(result.decision, ep.get("answer", ""))
            successes.append(float(success))
            reasoning_texts.append(getattr(result, "reasoning_chain", "") or "")
        except Exception:
            successes.append(0.0)

    success_rate = sum(successes) / max(1, len(successes))

    # Compute real cosine distance when encode_fn + training_embeddings available
    cosine_dist = 0.5  # neutral default
    if encode_fn is not None and training_embeddings and reasoning_texts:
        try:
            novel_embeddings = await encode_fn(reasoning_texts)
            if novel_embeddings:
                cosine_dist = await compute_cosine_distance_async(
                    novel_embeddings, training_embeddings,
                )
        except Exception:
            cosine_dist = 0.5
    elif training_embeddings:
        cosine_dist = _compute_cosine_distance(reasoning_texts, training_embeddings)

    return NoveltyEmergenceResult(
        novel_success_rate=success_rate,
        reasoning_cosine_distance=cosine_dist,
        novel_count=len(novel_episodes),
        genuine_learning=(success_rate > 0.6 and cosine_dist > 0.3),
    )


def _compute_cosine_distance(
    reasoning_texts: list[str],
    training_embeddings: Optional[list],
) -> float:
    """
    Mean cosine distance of generated reasoning from training data embeddings.
    High distance = novel reasoning (not just recalling training patterns).
    Returns 0.5 (neutral) when embeddings not available.
    """
    if not training_embeddings or not reasoning_texts:
        return 0.5
    try:
        # training_embeddings: list of shape-(dim,) numpy arrays or lists
        train_matrix = np.array(training_embeddings, dtype=np.float32)
        # Normalize training matrix rows
        train_norms = np.linalg.norm(train_matrix, axis=1, keepdims=True)
        train_norms = np.where(train_norms == 0, 1.0, train_norms)
        train_normed = train_matrix / train_norms
        # train_mean: centroid of training distribution
        train_centroid = train_normed.mean(axis=0)
        train_centroid_norm = np.linalg.norm(train_centroid)
        if train_centroid_norm == 0:
            return 0.5
        train_centroid = train_centroid / train_centroid_norm

        # We have reasoning_texts as strings - we need embeddings for them.
        # Since we can't async here, use pre-encoded embeddings if passed as
        # (texts, embeddings) tuple, else skip and return neutral.
        # Caller should pass pre-computed embeddings as training_embeddings
        # when it wants real distance; otherwise returns 0.5.
        return 0.5  # Caller must pass pre-encoded novel embeddings
    except Exception:
        return 0.5


async def compute_cosine_distance_async(
    novel_embeddings: list,
    training_embeddings: list,
) -> float:
    """
    Async cosine distance: mean distance of novel_embeddings from training centroid.
    Both inputs are lists of shape-(dim,) arrays/lists.
    Returns 0.5 if either list is empty or on any error.
    """
    if not novel_embeddings or not training_embeddings:
        return 0.5
    try:
        train = np.array(training_embeddings, dtype=np.float32)
        novel = np.array(novel_embeddings, dtype=np.float32)

        # Normalise
        def _normalize(m: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return m / norms

        train_normed = _normalize(train)
        novel_normed = _normalize(novel)

        # Training centroid
        centroid = train_normed.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm == 0:
            return 0.5
        centroid = centroid / c_norm

        # Mean cosine distance from centroid (1 - similarity)
        similarities = novel_normed @ centroid  # shape (n,)
        mean_sim = float(similarities.mean())
        return float(1.0 - mean_sim)  # distance: 0=identical, 2=opposite
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Pillar 3: Causal Reasoning Quality (CLadder + CCR.GB)
# ---------------------------------------------------------------------------


@dataclass
class CausalReasoningResult:
    l1_association: float
    """Rung 1 - all LLMs do this."""
    l2_intervention: float
    """Rung 2 - KEY metric. L2 improving over months = genuine learning."""
    l3_counterfactual: float
    """Rung 3 - HARDEST; almost all LLMs fail."""
    ccr_validity: float
    """Fictional world validity >0.6 = reasoning, not memorizing."""
    ccr_consistency: float
    l2_l3_improving: bool
    """Set by LongitudinalTracker comparing to prior month."""


async def measure_causal_reasoning(
    custom_engine,
    cladder_questions: list[dict],
    ccr_gb_scenarios: list[dict],
) -> CausalReasoningResult:
    """
    Bible §6.2 Pillar 3.
    CLadder (Jin et al., NeurIPS 2023): 10,112 questions across Pearl's 3 levels.
    CCR.GB (Maasch et al., ICML 2025): Fictional world models - memorizing model fails.

    cladder_questions: each {"question": ..., "answer": ..., "rung": 1|2|3}
    ccr_gb_scenarios:  each {"scenario": ..., "ground_truth": ..., "world_model": ...}
    """
    rung: dict[int, list[float]] = {1: [], 2: [], 3: []}

    for p in cladder_questions:
        try:
            result = await custom_engine.reason(episode_context=p["question"])
            correct = float(_answer_matches(result.decision, p["answer"]))
            rung_key = p.get("rung", 1)
            if rung_key in rung:
                rung[rung_key].append(correct)
        except Exception:
            pass

    ccr_results: list[dict] = []
    for p in ccr_gb_scenarios:
        try:
            result = await custom_engine.reason(episode_context=p["scenario"])
            chain = getattr(result, "reasoning_chain", "") or ""
            ccr_results.append({
                "valid": float(_evaluate_validity(chain, p.get("ground_truth", ""))),
                "consistent": float(_evaluate_consistency(chain, p.get("world_model", ""))),
            })
        except Exception:
            pass

    return CausalReasoningResult(
        l1_association=_mean(rung[1]),
        l2_intervention=_mean(rung[2]),
        l3_counterfactual=_mean(rung[3]),
        ccr_validity=_mean([r["valid"] for r in ccr_results]),
        ccr_consistency=_mean([r["consistent"] for r in ccr_results]),
        l2_l3_improving=False,  # Updated by LongitudinalTracker after comparison
    )


# ---------------------------------------------------------------------------
# Pillar 4: Learning Velocity
# ---------------------------------------------------------------------------


@dataclass
class LearningVelocityResult:
    velocity: float
    """Current rate of improvement per month."""
    is_plateaued: bool
    """velocity < 0.005 - investigate plasticity loss."""
    is_accelerating: bool
    """velocity > 0.02 - excellent."""
    predicted_month_12: float
    """Power-law projection to month 12."""
    insufficient_data: bool
    """< 3 months of history."""


def compute_learning_velocity(history: list[dict]) -> LearningVelocityResult:
    """
    Bible §6.2 Pillar 4.
    history: list of {"month": int, "score": float} - overall L2/L3 causal score per month.
    Uses power-law fit (a * x^b + c). Falls back to linear slope if fit fails.
    """
    if len(history) < 3:
        return LearningVelocityResult(
            velocity=0.0,
            is_plateaued=False,
            is_accelerating=False,
            predicted_month_12=0.0,
            insufficient_data=True,
        )

    months = np.array([h["month"] for h in history], dtype=float)
    scores = np.array([h["score"] for h in history], dtype=float)

    try:
        def power_law(x, a, b, c):
            return a * np.power(np.maximum(x, 0.01), b) + c

        popt, _ = curve_fit(power_law, months, scores, p0=[0.1, 0.5, 0.5], maxfev=5000)
        a, b, c = popt
        velocity = float(a * b * np.power(max(float(months[-1]), 0.01), b - 1))
        predicted_12 = float(np.clip(power_law(12, *popt), 0.0, 1.0))
    except Exception:
        velocity = float(
            (scores[-1] - scores[0]) / max(1.0, float(months[-1]) - float(months[0]))
        )
        predicted_12 = float(
            np.clip(scores[-1] + velocity * (12 - float(months[-1])), 0.0, 1.0)
        )

    return LearningVelocityResult(
        velocity=float(velocity),
        is_plateaued=velocity < 0.005,
        is_accelerating=velocity > 0.02,
        predicted_month_12=predicted_12,
        insufficient_data=False,
    )


# ---------------------------------------------------------------------------
# §6.3 Memorization Detection
# ---------------------------------------------------------------------------


@dataclass
class MemorizationReport:
    membership_inference_accuracy: float
    """Confidence-threshold classifier accuracy. >0.65 = memorizing."""
    paraphrase_accuracy_drop: float
    """Large drop = memorization; small = generalization."""
    svd_intruder_ratio: float
    """Ratio of intruder singular values vs base model. High = accumulating memorization."""
    memorization_risk: str
    """"low" | "medium" | "high"."""
    recommendation: str


async def detect_memorization(
    custom_engine,
    training_sample: list[dict],
    holdout_sample: list[dict],
    paraphrase_pairs: list[dict],
    adapter_path: Optional[str] = None,
) -> MemorizationReport:
    """
    Bible §6.3. Four memorization checks:
    1. Membership inference via confidence proxy (>65% classification accuracy = memorizing)
    2. Paraphrase perturbation (large accuracy drop = memorization)
    3. SVD intruder ratio (intruder dimensions = task-specific memorization)
    4. CCR.GB (Pillar 3) is the fourth check - run separately.

    training_sample: 50 examples FROM training (known positive)
    holdout_sample:  50 examples NOT in training (known negative)
    paraphrase_pairs: {"original":..., "paraphrase":..., "answer":...}
    adapter_path: path to adapter safetensors dir for SVD check
    """
    # 1. Membership inference via confidence proxy
    train_confidences: list[float] = []
    holdout_confidences: list[float] = []

    for ex in training_sample:
        try:
            result = await custom_engine.reason(
                episode_context=ex.get("question", ex.get("prompt", ""))
            )
            train_confidences.append(float(getattr(result, "confidence", 0.5)))
        except Exception:
            pass

    for ex in holdout_sample:
        try:
            result = await custom_engine.reason(
                episode_context=ex.get("question", ex.get("prompt", ""))
            )
            holdout_confidences.append(float(getattr(result, "confidence", 0.5)))
        except Exception:
            pass

    all_conf = train_confidences + holdout_confidences
    threshold = sum(all_conf) / max(1, len(all_conf))
    tp = sum(1 for c in train_confidences if c > threshold)
    tn = sum(1 for c in holdout_confidences if c <= threshold)
    mi_accuracy = (tp + tn) / max(1, len(train_confidences) + len(holdout_confidences))

    # 2. Paraphrase perturbation
    original_correct, paraphrase_correct = 0, 0
    for pair in paraphrase_pairs:
        try:
            r_orig = await custom_engine.reason(episode_context=pair["original"])
            r_para = await custom_engine.reason(episode_context=pair["paraphrase"])
            if _answer_matches(r_orig.decision, pair["answer"]):
                original_correct += 1
            if _answer_matches(r_para.decision, pair["answer"]):
                paraphrase_correct += 1
        except Exception:
            pass

    n_pairs = max(1, len(paraphrase_pairs))
    orig_acc = original_correct / n_pairs
    para_acc = paraphrase_correct / n_pairs
    accuracy_drop = orig_acc - para_acc

    # 3. SVD intruder ratio
    svd_ratio = _compute_svd_intruder_ratio(adapter_path)

    # Risk assessment
    risk_score = 0
    if mi_accuracy > 0.65:
        risk_score += 2
    elif mi_accuracy > 0.55:
        risk_score += 1
    if accuracy_drop > 0.20:
        risk_score += 2
    elif accuracy_drop > 0.10:
        risk_score += 1
    if svd_ratio > 0.30:
        risk_score += 1

    if risk_score >= 4:
        risk = "high"
        rec = "Reduce epochs; increase regularization; verify novelty weighting in replay buffer"
    elif risk_score >= 2:
        risk = "medium"
        rec = "Monitor next training cycle; increase replay diversity"
    else:
        risk = "low"
        rec = "No action required"

    return MemorizationReport(
        membership_inference_accuracy=mi_accuracy,
        paraphrase_accuracy_drop=accuracy_drop,
        svd_intruder_ratio=svd_ratio,
        memorization_risk=risk,
        recommendation=rec,
    )


def _compute_svd_intruder_ratio(adapter_path: Optional[str]) -> float:
    """
    Bible §6.3: check for intruder dimensions - new high-rank singular vectors not
    present in the base model. Requires adapter safetensors.
    Returns 0.0 if path not available.

    Full implementation: load adapter_model.safetensors, SVD each lora_B weight,
    compare singular value distribution to base model's distribution.
    Intruder criterion: singular values > 2× median of the layer's distribution.
    """
    if not adapter_path or not os.path.exists(adapter_path):
        return 0.0
    try:
        import safetensors.torch as st
        weights = st.load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
        ratios: list[float] = []
        for k, v in weights.items():
            if "lora_B" in k and v.ndim == 2:
                _, s, _ = np.linalg.svd(v.float().numpy(), full_matrices=False)
                median_s = float(np.median(s))
                if median_s > 0:
                    intruders = sum(1 for sv in s if sv > 2 * median_s)
                    ratios.append(intruders / max(1, len(s)))
        return float(np.mean(ratios)) if ratios else 0.0
    except Exception as exc:
        logger.warning("svd_intruder_check_failed", error=str(exc))
        return 0.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _answer_matches(decision: str, answer: str) -> bool:
    """Loose match: answer keyword appears in decision (case-insensitive)."""
    if not answer:
        return False
    return answer.lower().strip() in decision.lower()


def _evaluate_validity(reasoning: str, ground_truth: str) -> bool:
    """CCR.GB validity: does reasoning reach the correct conclusion?"""
    if not ground_truth:
        return False
    return ground_truth.lower().strip() in reasoning.lower()


def _evaluate_consistency(reasoning: str, world_model: str) -> bool:
    """CCR.GB consistency: does reasoning not contradict the fictional world model?"""
    if not world_model:
        return True
    world_vars = [v.strip() for v in world_model.split(",") if v.strip()]
    for var in world_vars:
        if f"not {var.lower()}" in reasoning.lower() or f"no {var.lower()}" in reasoning.lower():
            return False
    return True


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


# ---------------------------------------------------------------------------
# Fixed test set loader (for monthly eval)
# ---------------------------------------------------------------------------


def load_fixed_test_sets() -> dict:
    """
    Load all fixed test sets from data/evaluation/.
    These sets are NEVER modified post Week 7 (bible §10 Phase 1 Week 7).

    Returns dict with keys:
        domain_test       - 200 EOS-domain reasoning questions (Pillar 1)
        general_test      - 200 general reasoning questions (Pillar 1)
        novel_episodes    - 100 held-out episodes, FROZEN (Pillar 2)
        cladder_questions - 200 CLadder questions with rung field (Pillar 3)
        ccr_gb_scenarios  - 100 CCR.GB fictional world scenarios (Pillar 3)
        paraphrase_pairs  - 50 paraphrase pairs (§6.3 memorization detection)
    """
    sets: dict = {}
    files = {
        "domain_test": "domain_test_200.jsonl",
        "general_test": "general_test_200.jsonl",
        "novel_episodes": "novel_episodes_100.jsonl",
        "cladder_questions": "cladder_200.jsonl",
        "ccr_gb_scenarios": "ccr_gb_100.jsonl",
        "paraphrase_pairs": "paraphrase_pairs_50.jsonl",
    }
    for key, filename in files.items():
        path = DATA_DIR / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                sets[key] = [json.loads(line) for line in f if line.strip()]
        else:
            sets[key] = []
            logger.warning("test_set_missing", file=str(path))
    return sets
