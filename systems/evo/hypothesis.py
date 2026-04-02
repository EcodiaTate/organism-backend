"""
EcodiaOS - Evo Hypothesis Engine

Manages the full lifecycle of hypotheses:
  1. Generation  - LLM produces testable claims from detected patterns
  2. Testing     - each new episode is evaluated as evidence
  3. Integration - supported hypotheses have their mutations applied
  4. Archival    - refuted or stale hypotheses are stored and closed

Implements approximate Bayesian model comparison (spec Section IV.2):
  Evidence(H) = Σ log p(observation_i | H) - complexity(H)

Approximated as:
  evidence_score += strength × (1 - complexity_penalty × 0.1)  [for support]
  evidence_score -= strength                                     [for contradiction]

Integration requires (spec Section IX, VELOCITY_LIMITS):
  - evidence_score > 3.0
  - len(supporting_episodes) >= 10
  - hypothesis age >= 24 hours

Performance budget: evidence_evaluate ≤200ms per hypothesis (spec Section X).
"""

from __future__ import annotations

import contextlib
import json
import math
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import LLMProvider, Message
from clients.optimized_llm import OptimizedLLMProvider
from primitives.common import new_id, utc_now
from primitives.experimental import ExperimentDesign, ExperimentResult
from systems.evo.types import (
    VELOCITY_LIMITS,
    EvidenceDirection,
    EvidenceResult,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
    Mutation,
    MutationType,
    PatternCandidate,
)

if TYPE_CHECKING:
    from primitives.memory_trace import Episode
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

# Evidence thresholds (from VELOCITY_LIMITS)
_SUPPORT_SCORE_THRESHOLD: float = 3.0
_SUPPORT_EPISODE_THRESHOLD: int = VELOCITY_LIMITS["min_evidence_for_integration"]
_MIN_AGE_HOURS: int = VELOCITY_LIMITS["min_hypothesis_age_hours"]
_MAX_ACTIVE: int = VELOCITY_LIMITS["max_active_hypotheses"]

# LLM generation limits
_MAX_PER_BATCH: int = 3
_SYSTEM_PROMPT = (
    "You are the learning subsystem of a living digital organism. "
    "Your role is to generate precise, falsifiable hypotheses from observed patterns "
    "and evaluate evidence rigorously. Prefer simple explanations. "
    "Always respond with valid JSON matching the requested schema."
)


class HypothesisEngine:
    """
    Manages hypothesis generation, evidence accumulation, and lifecycle.

    Dependencies:
      llm     - LLM provider for generation and evaluation
      memory  - optional; used to persist hypotheses as :Hypothesis nodes
    """

    def __init__(
        self,
        llm: LLMProvider,
        instance_name: str = "EOS",
        memory: MemoryService | None = None,
        meta_learning: Any | None = None,
    ) -> None:
        self._llm = llm
        self._instance_name = instance_name
        self._memory = memory
        self._meta_learning = meta_learning
        self._logger = logger.bind(system="evo.hypothesis")
        self._optimized = isinstance(llm, OptimizedLLMProvider)

        # In-memory hypothesis registry (also persisted to Memory graph)
        self._active: dict[str, Hypothesis] = {}

        # Experiment tracking: hypothesis_id → ExperimentDesign
        self._experiments: dict[str, ExperimentDesign] = {}
        # Completed experiment results: hypothesis_id → ExperimentResult
        self._experiment_results: dict[str, ExperimentResult] = {}

        # Metrics
        self._total_proposed: int = 0
        self._total_supported: int = 0
        self._total_refuted: int = 0
        self._total_integrated: int = 0

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        """Look up a hypothesis by ID. Returns None if not found or archived."""
        return self._active.get(hypothesis_id)

    def _record_experiment_result(
        self,
        hypothesis: Hypothesis,
        outcome: str,
    ) -> None:
        """Record an ExperimentResult when a hypothesis reaches a terminal evidence state."""
        experiment = self._experiments.get(hypothesis.id)
        experiment_id = experiment.id if experiment else new_id()
        self._experiment_results[hypothesis.id] = ExperimentResult(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis.id,
            outcome=outcome,  # type: ignore[arg-type]
            metrics={
                "evidence_score": hypothesis.evidence_score,
                "supporting_count": float(len(hypothesis.supporting_episodes)),
                "contradicting_count": float(len(hypothesis.contradicting_episodes)),
            },
            confidence=max(0.0, min(1.0, hypothesis.evidence_score / 10.0)),
        )

    def get_experiment_results(self) -> dict[str, ExperimentResult]:
        """Return all completed experiment results."""
        return dict(self._experiment_results)

    def _compute_novelty_scores(self, new_hypotheses: list[Hypothesis]) -> None:
        """
        Compute novelty_score for each new hypothesis as
        1.0 - max_similarity against all confirmed (SUPPORTED/INTEGRATED) hypotheses.
        Uses word-set Jaccard as a cheap proxy for embedding similarity.
        """
        confirmed = [
            h for h in self._active.values()
            if h.status in (HypothesisStatus.SUPPORTED, HypothesisStatus.INTEGRATED)
        ]
        if not confirmed:
            for h in new_hypotheses:
                h.novelty_score = 1.0
            return

        confirmed_word_sets = [
            set(h.statement.lower().split()) for h in confirmed
        ]
        for h in new_hypotheses:
            words = set(h.statement.lower().split())
            max_sim = 0.0
            for cws in confirmed_word_sets:
                intersection = len(words & cws)
                union = len(words | cws)
                if union > 0:
                    max_sim = max(max_sim, intersection / union)
            h.novelty_score = round(1.0 - max_sim, 3)

    # ─── Generation ───────────────────────────────────────────────────────────

    async def generate_hypotheses(
        self,
        patterns: list[PatternCandidate],
        existing_summaries: list[str] | None = None,
    ) -> list[Hypothesis]:
        """
        Generate new hypotheses from a batch of pattern candidates.
        Uses LLM reasoning grounded in the pattern evidence.
        Respects MAX_ACTIVE_HYPOTHESES by skipping if at capacity.

        Returns up to _MAX_PER_BATCH new hypotheses.
        """
        if not patterns:
            return []

        if len(self._active) >= _MAX_ACTIVE:
            # Eviction policy (Spec §IV gap fix): rather than silently rejecting
            # new hypotheses, evict the lowest-fitness PROPOSED/TESTING hypothesis.
            # Fitness = evidence_score penalised by staleness (days since last evidence).
            evicted = self._evict_lowest_fitness()
            if evicted is None:
                # All active hypotheses are SUPPORTED/INTEGRATED - nothing to evict.
                self._logger.warning(
                    "hypothesis_capacity_reached_no_eviction",
                    active=len(self._active),
                    max=_MAX_ACTIVE,
                )
                return []
            self._logger.info(
                "hypothesis_evicted_for_capacity",
                evicted_id=evicted.id,
                evicted_score=round(evicted.evidence_score, 3),
                active=len(self._active),
            )

        prompt = _build_generation_prompt(
            instance_name=self._instance_name,
            patterns=patterns,
            existing_hypotheses=existing_summaries or list(self._active_summaries()),
            max_hypotheses=_MAX_PER_BATCH,
        )

        # Budget check: skip hypothesis generation in YELLOW/RED (low priority)
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not await self._llm.should_use_llm("evo.hypothesis", estimated_tokens=1200):
                self._logger.info("hypothesis_generation_skipped_budget")
                return []

        try:
            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=_SYSTEM_PROMPT,
                    messages=[Message("user", prompt)],
                    max_tokens=1200,
                    temperature=0.5,
                    output_format="json",
                    cache_system="evo.hypothesis",
                    cache_method="generate",
                )
            else:
                response = await self._llm.generate(
                    system_prompt=_SYSTEM_PROMPT,
                    messages=[Message("user", prompt)],
                    max_tokens=1200,
                    temperature=0.5,
                    output_format="json",
                )
            raw = _parse_json_safe(response.text)
        except Exception as exc:
            self._logger.error("hypothesis_generation_failed", error=str(exc))
            return []

        hypotheses: list[Hypothesis] = []
        for item in raw.get("hypotheses", [])[:_MAX_PER_BATCH]:
            try:
                h = _build_hypothesis(item)
                self._active[h.id] = h
                self._total_proposed += 1
                hypotheses.append(h)
                self._logger.info(
                    "hypothesis_proposed",
                    hypothesis_id=h.id,
                    category=h.category.value,
                    statement=h.statement[:80],
                )
            except (KeyError, ValueError) as exc:
                self._logger.warning("hypothesis_parse_failed", error=str(exc))
                continue

        # Compute novelty scores: 1.0 - max similarity to confirmed hypotheses
        self._compute_novelty_scores(hypotheses)

        # Persist to Memory if available
        if self._memory is not None:
            for h in hypotheses:
                await self._persist_hypothesis(h)

        return hypotheses

    # ─── Evidence Evaluation ──────────────────────────────────────────────────

    async def evaluate_evidence(
        self,
        hypothesis: Hypothesis,
        episode: Episode,
    ) -> EvidenceResult:
        """
        Evaluate whether this episode provides evidence for or against a hypothesis.
        LLM evaluates with temperature=0.3 for consistency.
        Updates hypothesis in-place and returns the result.
        Budget: ≤200ms.
        """
        prompt = _build_evidence_prompt(hypothesis, episode)

        # Budget check: skip evidence evaluation in YELLOW/RED
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not await self._llm.should_use_llm("evo.evidence", estimated_tokens=300):
                return EvidenceResult(
                    hypothesis_id=hypothesis.id,
                    episode_id=episode.id,
                    direction=EvidenceDirection.NEUTRAL,
                    strength=0.0,
                    reasoning="Skipped - budget constraints",
                )

        try:
            if self._optimized:
                response = await self._llm.evaluate(  # type: ignore[call-arg]
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.3,
                    cache_system="evo.evidence",
                    cache_method="evaluate",
                )
            else:
                response = await self._llm.evaluate(
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.3,
                )
            raw = _parse_json_safe(response.text)
        except Exception as exc:
            self._logger.error(
                "evidence_evaluation_failed",
                hypothesis_id=hypothesis.id,
                error=str(exc),
            )
            return EvidenceResult(
                hypothesis_id=hypothesis.id,
                episode_id=episode.id,
                direction=EvidenceDirection.NEUTRAL,
                strength=0.0,
                reasoning="evaluation failed",
                new_score=hypothesis.evidence_score,
                new_status=hypothesis.status,
            )

        direction_raw = raw.get("direction", "neutral")
        strength = float(raw.get("strength", 0.0))
        strength = max(0.0, min(1.0, strength))
        reasoning = str(raw.get("reasoning", ""))

        try:
            direction = EvidenceDirection(direction_raw)
        except ValueError:
            direction = EvidenceDirection.NEUTRAL

        # Update hypothesis evidence score (Bayesian accumulation with MDL Occam penalty)
        # MDL penalty: longer (more complex) hypotheses need proportionally more
        # evidence. log2(statement_length) gives bits of description complexity.
        mdl_complexity = math.log2(max(10, len(hypothesis.statement))) / 10.0
        effective_penalty = max(hypothesis.complexity_penalty, mdl_complexity)

        if direction == EvidenceDirection.SUPPORTS:
            hypothesis.supporting_episodes.append(episode.id)
            hypothesis.evidence_score += strength * (1.0 - effective_penalty * 0.1)
        elif direction == EvidenceDirection.CONTRADICTS:
            hypothesis.contradicting_episodes.append(episode.id)
            hypothesis.evidence_score -= strength

        hypothesis.last_evidence_at = utc_now()

        # Status transitions - use adaptive thresholds from meta-learning if available
        score_threshold = _SUPPORT_SCORE_THRESHOLD
        episode_threshold = _SUPPORT_EPISODE_THRESHOLD
        if self._meta_learning is not None:
            score_threshold = self._meta_learning.get_effective_evidence_threshold()
            episode_threshold = self._meta_learning.get_effective_min_episodes()

        if hypothesis.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING):
            old_status = hypothesis.status
            hypothesis.status = HypothesisStatus.TESTING

            # Create ExperimentDesign on first transition to TESTING
            if old_status == HypothesisStatus.PROPOSED and hypothesis.id not in self._experiments:
                self._experiments[hypothesis.id] = ExperimentDesign(
                    hypothesis_id=hypothesis.id,
                    experiment_type="before_after",
                    description=f"Test: {hypothesis.statement[:200]}",
                    success_criteria=hypothesis.formal_test or "evidence_score > threshold",
                )

            if (
                hypothesis.evidence_score > score_threshold
                and len(hypothesis.supporting_episodes) >= episode_threshold
            ):
                hypothesis.status = HypothesisStatus.SUPPORTED
                self._total_supported += 1
                self._record_experiment_result(hypothesis, "confirmed")
                self._logger.info(
                    "hypothesis_supported",
                    hypothesis_id=hypothesis.id,
                    evidence_score=round(hypothesis.evidence_score, 2),
                    supporting_count=len(hypothesis.supporting_episodes),
                )
            elif hypothesis.evidence_score < -2.0:
                hypothesis.status = HypothesisStatus.REFUTED
                self._total_refuted += 1
                self._record_experiment_result(hypothesis, "refuted")
                self._logger.info(
                    "hypothesis_refuted",
                    hypothesis_id=hypothesis.id,
                    evidence_score=round(hypothesis.evidence_score, 2),
                )

        return EvidenceResult(
            hypothesis_id=hypothesis.id,
            episode_id=episode.id,
            direction=direction,
            strength=strength,
            reasoning=reasoning,
            new_score=hypothesis.evidence_score,
            new_status=hypothesis.status,
        )

    # ─── Integration ──────────────────────────────────────────────────────────

    async def integrate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """
        Mark a supported hypothesis as INTEGRATED.
        The caller (ConsolidationOrchestrator) is responsible for applying
        the proposed_mutation - this method only closes the hypothesis lifecycle.

        Returns True if integration is valid and was applied.
        """
        if hypothesis.status != HypothesisStatus.SUPPORTED:
            return False

        # Age check - each hypothesis carries its own min_age_hours gate.
        # Repair hypotheses use 4 h; standard hypotheses use the spec default (24 h).
        age_hours = (utc_now() - hypothesis.created_at).total_seconds() / 3600
        if age_hours < hypothesis.min_age_hours:
            self._logger.info(
                "hypothesis_integration_deferred",
                hypothesis_id=hypothesis.id,
                age_hours=round(age_hours, 1),
                required_hours=hypothesis.min_age_hours,
            )
            return False

        hypothesis.status = HypothesisStatus.INTEGRATED
        self._total_integrated += 1

        # Remove from active registry
        self._active.pop(hypothesis.id, None)

        self._logger.info(
            "hypothesis_integrated",
            hypothesis_id=hypothesis.id,
            category=hypothesis.category.value,
            evidence_score=round(hypothesis.evidence_score, 2),
            mutation_type=(
                hypothesis.proposed_mutation.type.value
                if hypothesis.proposed_mutation else "none"
            ),
        )

        if self._memory is not None:
            await self._persist_hypothesis(hypothesis)

        return True

    async def archive_hypothesis(
        self,
        hypothesis: Hypothesis,
        reason: str = "",
    ) -> None:
        """Mark a hypothesis as ARCHIVED and remove from active registry."""
        hypothesis.status = HypothesisStatus.ARCHIVED
        self._active.pop(hypothesis.id, None)

        self._logger.info(
            "hypothesis_archived",
            hypothesis_id=hypothesis.id,
            reason=reason or "not specified",
            evidence_score=round(hypothesis.evidence_score, 2),
        )

        if self._memory is not None:
            await self._persist_hypothesis(hypothesis)

    def is_stale(self, hypothesis: Hypothesis, max_age_days: int = 7) -> bool:
        """
        Return True if this hypothesis has not received evidence in max_age_days.
        Stale hypotheses are archived during consolidation.
        """
        if hypothesis.status not in (
            HypothesisStatus.PROPOSED, HypothesisStatus.TESTING
        ):
            return False
        age = utc_now() - hypothesis.last_evidence_at
        return age > timedelta(days=max_age_days)

    # ─── Query Interface ──────────────────────────────────────────────────────

    def register_repair_hypothesis(
        self,
        statement: str,
        formal_test: str,
        endpoint: str,
        fix_type: str,
        incident_class: str,
        source_episode_id: str,
    ) -> tuple[Hypothesis, bool]:
        """
        Directly register a procedural hypothesis from a confirmed repair.

        Bypasses LLM generation because the pattern is already known: Thymos
        observed the fix succeed.  The hypothesis starts in TESTING state with
        one supporting episode and a small initial evidence bump so it can
        progress quickly if the same pattern recurs.

        Returns ``(hypothesis, is_new)`` where ``is_new=False`` means an
        existing hypothesis matched (caller should call evaluate_evidence on
        it with the new repair episode rather than discarding the observation).

        Returns ``(None, False)`` (as ``None`` cast) only when at capacity -
        callers must check for None before unpacking.
        """
        if len(self._active) >= _MAX_ACTIVE:
            evicted = self._evict_lowest_fitness()
            if evicted is None:
                self._logger.warning(
                    "hypothesis_capacity_reached_repair_no_eviction",
                    active=len(self._active),
                )
                return None  # type: ignore[return-value]
            self._logger.info(
                "hypothesis_evicted_for_repair_capacity",
                evicted_id=evicted.id,
                evicted_score=round(evicted.evidence_score, 3),
            )

        # Dedup on structured fields - exact match on normalised endpoint+fix_type pair.
        # Using repair_endpoint/repair_fix_type avoids false matches from statement text.
        endpoint_norm = endpoint.lower().strip()
        fix_type_norm = fix_type.lower().strip()
        if endpoint_norm and fix_type_norm:
            for existing in self._active.values():
                if (
                    existing.repair_endpoint == endpoint_norm
                    and existing.repair_fix_type == fix_type_norm
                ):
                    self._logger.debug(
                        "repair_hypothesis_duplicate",
                        endpoint=endpoint_norm,
                        fix_type=fix_type_norm,
                        existing_id=existing.id,
                    )
                    # Return the existing hypothesis so the caller can feed the
                    # new repair episode as evidence (Option B from design doc).
                    return existing, False

        h = Hypothesis(
            category=HypothesisCategory.PROCEDURAL,
            statement=statement,
            formal_test=formal_test,
            complexity_penalty=0.05,  # Direct repair evidence → low complexity
            min_age_hours=4.0,         # Repair hypotheses: 4 h gate (not 24 h)
            status=HypothesisStatus.TESTING,
            evidence_score=1.0,        # Bootstrap: one confirmed success
            supporting_episodes=[source_episode_id],
            repair_endpoint=endpoint_norm,
            repair_fix_type=fix_type_norm,
        )
        h.last_evidence_at = utc_now()
        self._active[h.id] = h
        self._total_proposed += 1
        self._logger.info(
            "repair_hypothesis_registered",
            hypothesis_id=h.id,
            endpoint=endpoint,
            fix_type=fix_type,
            incident_class=incident_class,
            statement=statement[:100],
        )
        # Persist immediately so the pattern survives restarts.
        if self._memory is not None:
            import asyncio
            with contextlib.suppress(RuntimeError):
                asyncio.get_event_loop().create_task(self._persist_hypothesis(h))
        return h, True

    def get_repair_hypotheses(self, endpoint: str) -> list[Hypothesis]:
        """
        Return active procedural hypotheses whose repair_endpoint contains the
        given endpoint as a substring (or vice versa).

        Uses the structured repair_endpoint field rather than statement text to
        avoid false matches from unrelated mentions of the same word.
        Falls back to statement substring for legacy hypotheses that pre-date
        the structured fields.
        """
        endpoint_lower = endpoint.lower().strip()
        results = []
        for h in self._active.values():
            if h.category != HypothesisCategory.PROCEDURAL:
                continue
            if h.status not in (HypothesisStatus.TESTING, HypothesisStatus.SUPPORTED):
                continue
            if h.repair_endpoint:
                # Structured match: either endpoint contains the stored one or
                # the stored one contains the queried endpoint (path prefix match).
                if endpoint_lower in h.repair_endpoint or h.repair_endpoint in endpoint_lower:
                    results.append(h)
            else:
                # Legacy fallback: substring in statement
                if endpoint_lower in h.statement.lower():
                    results.append(h)
        return results

    def get_active(self) -> list[Hypothesis]:
        """Return all currently active hypotheses (proposed or testing)."""
        return [
            h for h in self._active.values()
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)
        ]

    def get_supported(self) -> list[Hypothesis]:
        """Return all supported hypotheses ready for integration."""
        return [
            h for h in self._active.values()
            if h.status == HypothesisStatus.SUPPORTED
        ]

    def get_all_active(self) -> list[Hypothesis]:
        """Return all hypotheses still in the registry (not yet archived/integrated)."""
        return list(self._active.values())

    @property
    def stats(self) -> dict[str, int]:
        return {
            "active": len(self._active),
            "proposed": self._total_proposed,
            "supported": self._total_supported,
            "refuted": self._total_refuted,
            "integrated": self._total_integrated,
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    def _active_summaries(self) -> list[str]:
        """Return short statements of all active hypotheses (deduplication prompt)."""
        return [h.statement[:100] for h in self._active.values()]

    def _evict_lowest_fitness(self) -> Hypothesis | None:
        """Evict the lowest-fitness evictable hypothesis to free capacity.

        Eviction fitness = evidence_score − staleness_penalty, where
        staleness_penalty = days_since_last_evidence × 0.1 (max 2.0).

        Only PROPOSED and TESTING hypotheses are eviction candidates -
        SUPPORTED/INTEGRATED ones have already met evidence thresholds.

        Spec §IV gap fix: replaces silent rejection at capacity with LRU eviction.
        Returns the evicted hypothesis (already removed from _active), or None if
        no evictable candidates exist.
        """
        import time

        now = time.time()
        candidates = [
            h for h in self._active.values()
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)
        ]
        if not candidates:
            return None

        def _fitness(h: Hypothesis) -> float:
            try:
                last_evidence_ts = h.last_evidence_at.timestamp()
            except Exception:
                last_evidence_ts = h.created_at.timestamp() if hasattr(h, "created_at") else now
            days_stale = (now - last_evidence_ts) / 86400.0
            staleness_penalty = min(2.0, days_stale * 0.1)
            return h.evidence_score - staleness_penalty

        worst = min(candidates, key=_fitness)
        del self._active[worst.id]
        return worst

    async def _persist_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Store hypothesis as a governance record in Memory."""
        if self._memory is None:
            return
        try:
            await self._memory.execute_write(
                """
                MERGE (h:Hypothesis {hypothesis_id: $hypothesis_id})
                SET h.type = $type,
                    h.category = $category,
                    h.statement = $statement,
                    h.status = $status,
                    h.evidence_score = $evidence_score,
                    h.supporting_count = $supporting_count,
                    h.contradicting_count = $contradicting_count,
                    h.created_at = $created_at
                """,
                {
                    "hypothesis_id": hypothesis.id,
                    "type": "hypothesis",
                    "category": hypothesis.category.value,
                    "statement": hypothesis.statement,
                    "status": hypothesis.status.value,
                    "evidence_score": hypothesis.evidence_score,
                    "supporting_count": len(hypothesis.supporting_episodes),
                    "contradicting_count": len(hypothesis.contradicting_episodes),
                    "created_at": hypothesis.created_at.isoformat(),
                },
            )
        except Exception as exc:
            self._logger.warning(
                "hypothesis_persist_failed",
                hypothesis_id=hypothesis.id,
                error=str(exc),
            )


# ─── Prompt Builders ──────────────────────────────────────────────────────────


def _build_generation_prompt(
    instance_name: str,
    patterns: list[PatternCandidate],
    existing_hypotheses: list[str],
    max_hypotheses: int,
) -> str:
    pattern_lines = "\n".join(
        f"- [{p.type.value}] {', '.join(p.elements[:4])} "
        f"(count={p.count}, confidence={p.confidence:.2f})"
        for p in patterns[:10]
    )
    existing_block = (
        "\n".join(f"- {s}" for s in existing_hypotheses[:10])
        if existing_hypotheses
        else "(none)"
    )
    return f"""You are the learning system of {instance_name}, a living digital organism.

DETECTED PATTERNS:
{pattern_lines}

CURRENT ACTIVE HYPOTHESES (avoid duplicates):
{existing_block}

Generate up to {max_hypotheses} hypotheses that explain the patterns above.

Rules:
- Each hypothesis must be FALSIFIABLE - state exactly how it could be proven false
- Prefer SIMPLER explanations (Occam's razor) - penalise unnecessary complexity
- Do NOT duplicate existing hypotheses
- Max {max_hypotheses} hypotheses per batch

For each hypothesis respond in this exact JSON schema:
{{
  "hypotheses": [
    {{
      "category": "world_model|self_model|social|procedural|parameter",
      "statement": "A clear, testable claim (1-2 sentences)",
      "formal_test": "Specific observable condition that would confirm or refute this",
      "complexity": "low|medium|high",
      "proposed_mutation": {{
        "type": "parameter_adjustment|procedure_creation|schema_addition|evolution_proposal",
        "target": "parameter name, procedure name, or entity type",
        "value": 0.0,
        "description": "What specifically to change if confirmed"
      }}
    }}
  ]
}}

If no compelling hypotheses arise from these patterns, return {{"hypotheses": []}}."""


def _build_evidence_prompt(hypothesis: Hypothesis, episode: Episode) -> str:
    return f"""HYPOTHESIS: {hypothesis.statement}
FORMAL TEST: {hypothesis.formal_test}

EVIDENCE (episode):
Content: {episode.raw_content[:300] or episode.summary[:300]}
Source: {episode.source}
Time: {episode.event_time.isoformat()}
Affect: valence={episode.affect_valence:.2f}, arousal={episode.affect_arousal:.2f}
Salience: {episode.salience_composite:.2f}

Does this episode provide evidence FOR or AGAINST the hypothesis?

Respond in JSON:
{{
  "direction": "supports|contradicts|neutral",
  "strength": 0.0,
  "reasoning": "1-2 sentence explanation"
}}

Where strength (0.0–1.0) represents how strongly this evidence bears on the hypothesis."""


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _build_hypothesis(item: dict[str, Any]) -> Hypothesis:
    """Parse one hypothesis item from LLM JSON response."""
    category_raw = str(item.get("category", "world_model"))
    try:
        category = HypothesisCategory(category_raw)
    except ValueError:
        category = HypothesisCategory.WORLD_MODEL

    complexity_raw = str(item.get("complexity", "low"))
    complexity_map = {"low": 0.05, "medium": 0.15, "high": 0.30}
    complexity_penalty = complexity_map.get(complexity_raw, 0.10)

    mutation: Mutation | None = None
    mutation_data = item.get("proposed_mutation")
    if mutation_data and isinstance(mutation_data, dict):
        try:
            mutation = Mutation(
                type=MutationType(mutation_data.get("type", "parameter_adjustment")),
                target=str(mutation_data.get("target", "")),
                value=float(mutation_data.get("value", 0.0)),
                description=str(mutation_data.get("description", "")),
            )
        except (ValueError, KeyError):
            mutation = None

    return Hypothesis(
        category=category,
        statement=str(item["statement"]),
        formal_test=str(item["formal_test"]),
        complexity_penalty=complexity_penalty,
        proposed_mutation=mutation,
    )


def _parse_json_safe(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


# ─── Structural Hypothesis Generators ─────────────────────────────────────────
#
# These bypass LLM cost entirely. Hypotheses are computed from graph topology,
# prediction error signals, belief contradictions, and schema structural similarity.
# ──────────────────────────────────────────────────────────────────────────────


class StructuralHypothesisGenerator:
    """
    Generates hypotheses purely from graph topology, prediction errors,
    contradictions, and structural analogy - no LLM calls.

    Four generators:
      1. Graph-structural: betweenness centrality, cluster detection, temporal
         sequence analysis → latent relationship / entity type / intermediate step
      2. Prediction error-driven: Fovea PREDICTION_ERROR events → domain-specific
         hypotheses about why the world model is wrong
      3. Contradiction-driven: conflicting consolidated beliefs → scope boundary
         discovery hypotheses
      4. Analogy-driven: structurally similar schemas across domains → transfer
         hypotheses
    """

    def __init__(
        self,
        memory: MemoryService | None = None,
    ) -> None:
        self._memory = memory
        self._logger = logger.bind(system="evo.structural_hypothesis")
        self._total_generated: int = 0

    # ─── 1. Graph-Structural Generator ──────────────────────────────────────

    async def generate_from_graph_topology(
        self,
        existing_ids: set[str] | None = None,
    ) -> list[Hypothesis]:
        """
        Analyse the knowledge graph to find structural anomalies that
        suggest missing knowledge:

        - High-betweenness nodes with no direct edge → latent relationship
        - Dense clusters with no named schema → missing entity type
        - A→B→C frequent but A→C without B rare → B is necessary intermediate
        """
        if self._memory is None:
            return []

        existing_ids = existing_ids or set()
        hypotheses: list[Hypothesis] = []

        # --- Latent relationships: high betweenness bridging nodes ---
        try:
            bridging = await self._memory.execute_read(
                """
                MATCH (a)-[r1]-(bridge)-[r2]-(b)
                WHERE NOT (a)--(b)
                  AND a <> b AND a <> bridge AND b <> bridge
                WITH bridge, count(DISTINCT a) + count(DISTINCT b) AS reach,
                     collect(DISTINCT labels(a)[0]) AS a_types,
                     collect(DISTINCT labels(b)[0]) AS b_types
                WHERE reach > 5
                RETURN bridge.name AS bridge_name,
                       labels(bridge)[0] AS bridge_type,
                       reach,
                       a_types[..3] AS sample_a_types,
                       b_types[..3] AS sample_b_types
                ORDER BY reach DESC
                LIMIT 5
                """
            )
            for record in bridging:
                name = record.get("bridge_name", "unknown")
                h_id = f"structural_bridge_{name}"
                if h_id in existing_ids:
                    continue
                h = Hypothesis(
                    id=h_id,
                    category=HypothesisCategory.WORLD_MODEL,
                    statement=(
                        f"Node '{name}' (type: {record.get('bridge_type', '?')}) bridges "
                        f"{record.get('reach', 0)} disconnected nodes - there may be a "
                        f"latent direct relationship between types "
                        f"{record.get('sample_a_types', [])} and {record.get('sample_b_types', [])}"
                    ),
                    formal_test=(
                        f"If nodes connected through '{name}' also show direct "
                        f"co-occurrence or causal relationship in new episodes, "
                        f"a direct edge should be created"
                    ),
                    complexity_penalty=0.10,
                    status=HypothesisStatus.PROPOSED,
                )
                hypotheses.append(h)
        except Exception as exc:
            self._logger.warning("graph_bridging_query_failed", error=str(exc))

        # --- Dense clusters without named schema ---
        try:
            clusters = await self._memory.execute_read(
                """
                MATCH (a)-[r]-(b)
                WHERE NOT a:Schema AND NOT b:Schema
                WITH a, count(DISTINCT b) AS neighbors
                WHERE neighbors > 8
                WITH a, neighbors
                MATCH (a)--(c)--(d)--(a)
                WITH a, neighbors, count(DISTINCT c) + count(DISTINCT d) AS triangle_nodes
                WHERE triangle_nodes > 3
                RETURN a.name AS cluster_center,
                       labels(a)[0] AS center_type,
                       neighbors,
                       triangle_nodes
                ORDER BY triangle_nodes DESC
                LIMIT 3
                """
            )
            for record in clusters:
                center = record.get("cluster_center", "unknown")
                h_id = f"structural_cluster_{center}"
                if h_id in existing_ids:
                    continue
                h = Hypothesis(
                    id=h_id,
                    category=HypothesisCategory.WORLD_MODEL,
                    statement=(
                        f"Dense cluster around '{center}' ({record.get('neighbors', 0)} neighbors, "
                        f"{record.get('triangle_nodes', 0)} triangle participants) has no named "
                        f"schema - this may represent an undiscovered entity type or concept"
                    ),
                    formal_test=(
                        f"If elements in the cluster around '{center}' share >3 common "
                        f"properties, a schema should be induced to compress them"
                    ),
                    complexity_penalty=0.12,
                    status=HypothesisStatus.PROPOSED,
                )
                hypotheses.append(h)
        except Exception as exc:
            self._logger.warning("graph_cluster_query_failed", error=str(exc))

        # --- Temporal sequence intermediates: A→B→C common, A→C rare ---
        try:
            sequences = await self._memory.execute_read(
                """
                MATCH (a)-[:FOLLOWED_BY]->(b)-[:FOLLOWED_BY]->(c)
                WITH a.name AS step_a, b.name AS step_b, c.name AS step_c,
                     count(*) AS abc_count
                WHERE abc_count > 3
                OPTIONAL MATCH (a2)-[:FOLLOWED_BY]->(c2)
                WHERE a2.name = step_a AND c2.name = step_c
                WITH step_a, step_b, step_c, abc_count,
                     count(a2) AS direct_ac_count
                WHERE direct_ac_count < 2
                RETURN step_a, step_b, step_c, abc_count, direct_ac_count
                ORDER BY abc_count DESC
                LIMIT 3
                """
            )
            for record in sequences:
                a = record.get("step_a", "?")
                b = record.get("step_b", "?")
                c = record.get("step_c", "?")
                h_id = f"structural_intermediate_{a}_{b}_{c}"
                if h_id in existing_ids:
                    continue
                h = Hypothesis(
                    id=h_id,
                    category=HypothesisCategory.PROCEDURAL,
                    statement=(
                        f"'{b}' is a necessary intermediate step between '{a}' and '{c}': "
                        f"the sequence A→B→C occurs {record.get('abc_count', 0)} times "
                        f"but A→C directly only {record.get('direct_ac_count', 0)} times"
                    ),
                    formal_test=(
                        f"Attempting '{a}' → '{c}' directly (skipping '{b}') should fail "
                        f"or produce degraded results compared to the full A→B→C sequence"
                    ),
                    complexity_penalty=0.08,
                    status=HypothesisStatus.PROPOSED,
                )
                hypotheses.append(h)
        except Exception as exc:
            self._logger.warning("graph_sequence_query_failed", error=str(exc))

        self._total_generated += len(hypotheses)
        if hypotheses:
            self._logger.info(
                "structural_hypotheses_from_graph",
                count=len(hypotheses),
            )
        return hypotheses

    # ─── 2. Prediction Error-Driven Generator ──────────────────────────────

    def generate_from_prediction_error(
        self,
        domain: str,
        predicted_value: float,
        actual_value: float,
        context_description: str,
        existing_ids: set[str] | None = None,
    ) -> list[Hypothesis]:
        """
        When Fovea reports a high prediction error, generate hypotheses about
        why the world model is wrong in this domain.

        The error magnitude determines the hypothesis category:
          - Small errors (< 0.3) → PARAMETER hypothesis (calibration off)
          - Medium errors (0.3–0.7) → WORLD_MODEL hypothesis (model structure)
          - Large errors (> 0.7) → WORLD_MODEL hypothesis (fundamental gap)
        """
        existing_ids = existing_ids or set()
        error_magnitude = abs(actual_value - predicted_value)
        error_direction = "higher" if actual_value > predicted_value else "lower"
        hypotheses: list[Hypothesis] = []

        if error_magnitude < 0.3:
            # Parameter calibration hypothesis
            h_id = f"pred_error_param_{domain}_{int(error_magnitude * 100)}"
            if h_id not in existing_ids:
                h = Hypothesis(
                    id=h_id,
                    category=HypothesisCategory.PARAMETER,
                    statement=(
                        f"World model for domain '{domain}' has a systematic "
                        f"{error_direction} bias of ~{error_magnitude:.2f}: "
                        f"predicted {predicted_value:.3f}, actual {actual_value:.3f}. "
                        f"Context: {context_description[:150]}"
                    ),
                    formal_test=(
                        f"After adjusting the {domain} prediction parameter by "
                        f"{'-' if error_direction == 'higher' else '+'}{error_magnitude:.2f}, "
                        f"the next 5 predictions in this domain should have <50% of current error"
                    ),
                    complexity_penalty=0.05,
                    status=HypothesisStatus.PROPOSED,
                    proposed_mutation=Mutation(
                        type=MutationType.PARAMETER_ADJUSTMENT,
                        target=f"{domain}_prediction_bias",
                        value=error_magnitude if error_direction == "higher" else -error_magnitude,
                        description=(
                            f"Adjust {domain} prediction bias to correct "
                            f"systematic {error_direction} error"
                        ),
                    ),
                )
                hypotheses.append(h)
        elif error_magnitude < 0.7:
            # Structural world model hypothesis
            h_id = f"pred_error_structure_{domain}_{int(error_magnitude * 100)}"
            if h_id not in existing_ids:
                h = Hypothesis(
                    id=h_id,
                    category=HypothesisCategory.WORLD_MODEL,
                    statement=(
                        f"World model for domain '{domain}' is structurally incomplete: "
                        f"prediction error {error_magnitude:.2f} ({error_direction} than expected) "
                        f"suggests a missing variable or relationship. "
                        f"Context: {context_description[:150]}"
                    ),
                    formal_test=(
                        f"Identifying and adding the missing variable to the {domain} model "
                        f"should reduce prediction error below 0.15 within 10 observations"
                    ),
                    complexity_penalty=0.15,
                    status=HypothesisStatus.PROPOSED,
                )
                hypotheses.append(h)
        else:
            # Fundamental gap hypothesis
            h_id = f"pred_error_gap_{domain}_{int(error_magnitude * 100)}"
            if h_id not in existing_ids:
                h = Hypothesis(
                    id=h_id,
                    category=HypothesisCategory.WORLD_MODEL,
                    statement=(
                        f"Fundamental gap in world model for domain '{domain}': "
                        f"prediction error {error_magnitude:.2f} indicates the model's "
                        f"assumptions are wrong, not just miscalibrated. "
                        f"Predicted {predicted_value:.3f} vs actual {actual_value:.3f}. "
                        f"Context: {context_description[:150]}"
                    ),
                    formal_test=(
                        f"The existing {domain} model should be replaced or restructured - "
                        f"incremental parameter adjustment should NOT reduce error below 0.3"
                    ),
                    complexity_penalty=0.20,
                    status=HypothesisStatus.PROPOSED,
                )
                hypotheses.append(h)

        self._total_generated += len(hypotheses)
        if hypotheses:
            self._logger.info(
                "hypotheses_from_prediction_error",
                domain=domain,
                error_magnitude=round(error_magnitude, 3),
                count=len(hypotheses),
            )
        return hypotheses

    # ─── 3. Contradiction-Driven Generator ─────────────────────────────────

    async def generate_from_contradictions(
        self,
        beliefs: list[dict[str, Any]],
        existing_ids: set[str] | None = None,
    ) -> list[Hypothesis]:
        """
        When two consolidated beliefs appear to conflict, generate scope
        boundary discovery hypotheses.

        beliefs: list of dicts with keys: id, statement, domain, evidence_score, scope
        """
        existing_ids = existing_ids or set()
        hypotheses: list[Hypothesis] = []

        # Find pairs of beliefs that contradict (same domain, opposing statements)
        for i, b1 in enumerate(beliefs):
            for b2 in beliefs[i + 1:]:
                if b1.get("domain") != b2.get("domain"):
                    continue

                # Heuristic: contradiction if both have high evidence but one has
                # negative score direction vs the other
                s1 = b1.get("evidence_score", 0.0)
                s2 = b2.get("evidence_score", 0.0)
                if not (s1 > 1.0 and s2 > 1.0):
                    continue

                # Check for keyword-level contradiction signals
                stmt1 = b1.get("statement", "").lower()
                stmt2 = b2.get("statement", "").lower()
                contradiction_markers = [
                    ("increase", "decrease"), ("always", "never"),
                    ("more", "less"), ("higher", "lower"),
                    ("enable", "disable"), ("positive", "negative"),
                    ("fast", "slow"), ("before", "after"),
                ]
                found_contradiction = False
                for pos, neg in contradiction_markers:
                    if (pos in stmt1 and neg in stmt2) or (neg in stmt1 and pos in stmt2):
                        found_contradiction = True
                        break

                if not found_contradiction:
                    continue

                domain = b1.get("domain", "unknown")
                h_id = f"contradiction_{b1.get('id', '')}_{b2.get('id', '')}"
                if h_id in existing_ids:
                    continue

                scope1 = b1.get("scope", "unknown")
                scope2 = b2.get("scope", "unknown")

                h = Hypothesis(
                    id=h_id,
                    category=HypothesisCategory.WORLD_MODEL,
                    statement=(
                        f"Beliefs '{b1.get('statement', '')[:80]}' and "
                        f"'{b2.get('statement', '')[:80]}' appear to contradict in "
                        f"domain '{domain}'. They may both be true under different "
                        f"scope conditions: belief 1 applies to '{scope1}', "
                        f"belief 2 applies to '{scope2}'"
                    ),
                    formal_test=(
                        "Find the boundary condition separating the domains where "
                        "each belief holds. New observations should be classifiable "
                        "into one scope or the other with >80% accuracy"
                    ),
                    complexity_penalty=0.18,
                    status=HypothesisStatus.PROPOSED,
                )
                hypotheses.append(h)

                if len(hypotheses) >= 3:
                    break
            if len(hypotheses) >= 3:
                break

        self._total_generated += len(hypotheses)
        if hypotheses:
            self._logger.info(
                "hypotheses_from_contradictions",
                count=len(hypotheses),
                beliefs_examined=len(beliefs),
            )
        return hypotheses

    # ─── 4. Analogy-Driven Generator ───────────────────────────────────────

    def generate_from_schema_analogy(
        self,
        transfers: list[dict[str, Any]],
        existing_ids: set[str] | None = None,
    ) -> list[Hypothesis]:
        """
        When Schema A in domain 1 is structurally similar to Schema B in domain 2,
        hypothesize that the shared structure represents a deeper invariant.

        transfers: list of dicts with keys: source_schema, source_domain,
            target_domain, isomorphism_score, shared_elements
        """
        existing_ids = existing_ids or set()
        hypotheses: list[Hypothesis] = []

        for transfer in transfers:
            iso_score = transfer.get("isomorphism_score", 0.0)
            if iso_score < 0.5:
                continue

            source = transfer.get("source_schema", "unknown")
            source_domain = transfer.get("source_domain", "?")
            target_domain = transfer.get("target_domain", "?")
            shared = transfer.get("shared_elements", [])

            h_id = f"analogy_{source_domain}_{target_domain}_{source}"
            if h_id in existing_ids:
                continue

            h = Hypothesis(
                id=h_id,
                category=HypothesisCategory.WORLD_MODEL,
                statement=(
                    f"Schema '{source}' in domain '{source_domain}' is structurally "
                    f"isomorphic (score={iso_score:.2f}) to patterns in domain "
                    f"'{target_domain}'. Shared elements: {shared[:5]}. "
                    f"This suggests a domain-independent invariant that could "
                    f"transfer knowledge across domains"
                ),
                formal_test=(
                    f"Applying the structural template from '{source}' to domain "
                    f"'{target_domain}' should produce predictions with <0.3 error "
                    f"on at least 3 out of 5 test observations"
                ),
                complexity_penalty=0.15,
                status=HypothesisStatus.PROPOSED,
            )
            hypotheses.append(h)

            if len(hypotheses) >= 3:
                break

        self._total_generated += len(hypotheses)
        if hypotheses:
            self._logger.info(
                "hypotheses_from_analogy",
                count=len(hypotheses),
                transfers_examined=len(transfers),
            )
        return hypotheses

    # ─── Combined Structural Generation ────────────────────────────────────

    async def generate_all_structural(
        self,
        prediction_errors: list[dict[str, Any]] | None = None,
        beliefs: list[dict[str, Any]] | None = None,
        transfers: list[dict[str, Any]] | None = None,
        existing_ids: set[str] | None = None,
    ) -> list[Hypothesis]:
        """
        Run all structural generators and return combined results.
        Each generator is independent and failure-isolated.
        """
        existing_ids = existing_ids or set()
        all_hypotheses: list[Hypothesis] = []

        # 1. Graph topology
        try:
            graph_h = await self.generate_from_graph_topology(existing_ids)
            all_hypotheses.extend(graph_h)
            existing_ids.update(h.id for h in graph_h)
        except Exception as exc:
            self._logger.warning("graph_topology_generator_failed", error=str(exc))

        # 2. Prediction errors
        if prediction_errors:
            for pe in prediction_errors[:5]:
                try:
                    pe_h = self.generate_from_prediction_error(
                        domain=pe.get("domain", "unknown"),
                        predicted_value=pe.get("predicted", 0.0),
                        actual_value=pe.get("actual", 0.0),
                        context_description=pe.get("context", ""),
                        existing_ids=existing_ids,
                    )
                    all_hypotheses.extend(pe_h)
                    existing_ids.update(h.id for h in pe_h)
                except Exception as exc:
                    self._logger.warning(
                        "prediction_error_generator_failed", error=str(exc)
                    )

        # 3. Contradictions
        if beliefs:
            try:
                contra_h = await self.generate_from_contradictions(beliefs, existing_ids)
                all_hypotheses.extend(contra_h)
                existing_ids.update(h.id for h in contra_h)
            except Exception as exc:
                self._logger.warning("contradiction_generator_failed", error=str(exc))

        # 4. Schema analogies
        if transfers:
            try:
                analogy_h = self.generate_from_schema_analogy(transfers, existing_ids)
                all_hypotheses.extend(analogy_h)
            except Exception as exc:
                self._logger.warning("analogy_generator_failed", error=str(exc))

        if all_hypotheses:
            self._logger.info(
                "structural_generation_complete",
                total=len(all_hypotheses),
            )

        return all_hypotheses

    @property
    def stats(self) -> dict[str, int]:
        return {"total_structural_generated": self._total_generated}
