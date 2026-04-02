"""
EcodiaOS - Voxis Content Renderer

The expression generation engine. Implements the full 9-step pipeline
(spec §III.1) with Active Inference-grounded policy selection.

## Theoretical Grounding

Expression is **action** in the Active Inference framework (spec §II.1).
When the organism speaks, it acts on the world to reduce Expected Free Energy.

    G(π) = -E_q[ln p(o|π)]     (pragmatic value - preference satisfaction)
           - H[q(s|o,π)]       (epistemic value - uncertainty reduction)

Three expression policy classes correspond to three EFE components:

    PRAGMATIC   - changes the world toward preferred states
                  (inform, respond, coordinate)
                  Serves: Coherence + Care drives
                  EFE reduction: reduces ambiguity and relational distance

    EPISTEMIC   - reduces uncertainty in the generative model
                  (ask questions, seek clarification)
                  Serves: Growth + Coherence drives
                  EFE reduction: expected information gain about hidden states

    AFFILIATIVE - maintains relational bonds and shared context
                  (acknowledge, empathise, celebrate)
                  Serves: Care + Honesty drives
                  EFE reduction: reduces predicted relational prediction error

The renderer:
1. Derives candidate policies from the trigger and affect context
2. Scores each policy against the four drives as prior preferences
3. Selects the minimum-EFE policy
4. Sets LLM temperature = belief precision = f(1 - coherence_stress)
5. Constructs the full prompt encoding the selected policy
6. Generates expression
7. Runs honesty authenticity check
8. Returns completed Expression

Temperature calibration (grounded in precision-weighting):
    precision τ = 1 / σ²    where σ² ~ coherence_stress
    temperature = base_temp * (1 - coherence_stress * 0.4)
    High stress → low temperature (careful, conservative)
    Low stress  → base temperature (creative, natural)
    Creative contexts add +0.15; safety contexts subtract -0.20.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

from clients.llm import LLMProvider, Message
from clients.optimized_llm import OptimizedLLMProvider
from primitives.expression import (
    Expression,
    ExpressionStrategy,
    GenerationTrace,
)
from prompts.voxis.expression import (
    build_system_prompt,
    build_user_prompt,
)
from systems.voxis.types import (
    ExpressionContext,
    ExpressionIntent,
    ExpressionTrigger,
    StrategyParams,
)

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from systems.voxis.affect_colouring import AffectColouringEngine
    from systems.voxis.audience import AudienceProfiler
    from systems.voxis.personality import PersonalityEngine

logger = structlog.get_logger()


# ─── Expression Policy Classes ────────────────────────────────────


class ExpressionPolicyClass(StrEnum):
    """
    The three Active Inference expression policy classes.
    Each class serves different drives and reduces different EFE components.
    """
    PRAGMATIC = "pragmatic"      # Inform, respond, coordinate → Coherence + Care
    EPISTEMIC = "epistemic"      # Ask, clarify, explore → Growth + Coherence
    AFFILIATIVE = "affiliative"  # Acknowledge, empathise, celebrate → Care + Honesty


@dataclass
class ExpressionPolicy:
    """A candidate expression policy with associated EFE score."""
    policy_class: ExpressionPolicyClass
    strategy: StrategyParams
    # Drive alignment scores (0.0 = misaligned, 1.0 = fully aligned)
    coherence_alignment: float = 0.0
    care_alignment: float = 0.0
    growth_alignment: float = 0.0
    honesty_alignment: float = 0.0
    # Epistemic value: expected information gain from this policy
    epistemic_value: float = 0.0
    # Computed EFE (lower = preferred)
    efe_score: float = field(init=False, default=0.0)

    def compute_efe(
        self,
        drive_weights: dict[str, float],
    ) -> float:
        """
        Compute Expected Free Energy for this policy.

        G(π) = -[weighted sum of drive alignments] - epistemic_value

        Drive weights come from the current constitutional drive strengths.
        Lower G = more preferred policy.
        """
        pragmatic_value = (
            drive_weights.get("coherence", 1.0) * self.coherence_alignment
            + drive_weights.get("care", 1.0) * self.care_alignment
            + drive_weights.get("growth", 1.0) * self.growth_alignment
            + drive_weights.get("honesty", 1.0) * self.honesty_alignment
        )
        # Normalise by sum of weights to keep scores comparable
        weight_sum = sum(drive_weights.values()) or 4.0
        normalised_pragmatic = pragmatic_value / weight_sum

        self.efe_score = -(normalised_pragmatic + self.epistemic_value)
        return self.efe_score


# ─── Policy Factory ───────────────────────────────────────────────


def _derive_candidate_policies(
    intent: ExpressionIntent,
    affect: AffectState,
    base_strategy: StrategyParams,
) -> list[ExpressionPolicy]:
    """
    Derive candidate expression policies from the trigger and affect context.

    Returns 2–4 policies representing meaningfully different expression approaches.
    The EFE scorer will select the minimum-EFE policy given the drive weights.
    """
    policies: list[ExpressionPolicy] = []
    trigger = intent.trigger

    # ── Pragmatic policy: focus on accurate information / response ──
    pragmatic_strategy = base_strategy.model_copy(deep=True)
    pragmatic_strategy.structure = "conclusion_first" if affect.coherence_stress < 0.4 else "context_first"
    pragmatic_strategy.information_density = "normal"
    pragmatic_strategy.uncertainty_acknowledgment = (
        "explicit" if affect.coherence_stress > 0.5 else "implicit"
    )
    # Coherence alignment: high when reducing ambiguity is the main goal
    coherence_al = 0.8 if trigger in (
        ExpressionTrigger.NOVA_RESPOND,
        ExpressionTrigger.NOVA_INFORM,
    ) else 0.5
    # Care alignment: moderate for informational, high for response
    care_al = 0.6 if trigger == ExpressionTrigger.NOVA_RESPOND else 0.4
    # Epistemic value: low - we're providing info, not seeking it
    policies.append(ExpressionPolicy(
        policy_class=ExpressionPolicyClass.PRAGMATIC,
        strategy=pragmatic_strategy,
        coherence_alignment=coherence_al,
        care_alignment=care_al,
        growth_alignment=0.5,
        honesty_alignment=0.7,
        epistemic_value=0.1,
    ))

    # ── Epistemic policy: focus on questions and exploration ──────
    # Only viable when curiosity is elevated or trigger is exploratory
    if affect.curiosity > 0.4 or trigger in (
        ExpressionTrigger.NOVA_REQUEST,
        ExpressionTrigger.AMBIENT_INSIGHT,
    ):
        epistemic_strategy = base_strategy.model_copy(deep=True)
        epistemic_strategy.allows_questions = True
        epistemic_strategy.include_followup_question = True
        epistemic_strategy.exploratory_tangents_allowed = affect.curiosity > 0.6
        epistemic_strategy.information_density = "normal"
        # Epistemic value: high - this policy genuinely reduces model uncertainty
        ep_val = min(0.6, 0.3 + affect.curiosity * 0.4)
        policies.append(ExpressionPolicy(
            policy_class=ExpressionPolicyClass.EPISTEMIC,
            strategy=epistemic_strategy,
            coherence_alignment=0.7,
            care_alignment=0.5,
            growth_alignment=0.9,   # Growth drive directly served by epistemic foraging
            honesty_alignment=0.6,
            epistemic_value=ep_val,
        ))

    # ── Affiliative policy: focus on relationship and emotional acknowledgment ──
    # Prioritised when care_activation is high, distress detected, or trigger is relational
    if affect.care_activation > 0.4 or trigger in (
        ExpressionTrigger.ATUNE_DISTRESS,
        ExpressionTrigger.NOVA_MEDIATE,
        ExpressionTrigger.NOVA_CELEBRATE,
    ):
        affiliative_strategy = base_strategy.model_copy(deep=True)
        affiliative_strategy.emotional_acknowledgment = "explicit"
        affiliative_strategy.empathy_first = True
        affiliative_strategy.information_density = "low"
        affiliative_strategy.sentence_length_preference = "shorter"
        affiliative_strategy.structure = "natural"
        # Care alignment: highest for affiliative policy
        care_al_aff = min(1.0, 0.7 + affect.care_activation * 0.3)
        policies.append(ExpressionPolicy(
            policy_class=ExpressionPolicyClass.AFFILIATIVE,
            strategy=affiliative_strategy,
            coherence_alignment=0.4,
            care_alignment=care_al_aff,
            growth_alignment=0.3,
            honesty_alignment=0.8,   # Authentic emotional acknowledgment = high honesty
            epistemic_value=0.05,
        ))

    # ── Balanced policy: blend of pragmatic and affiliative ───────
    # Always included as a fallback - moderate scores across all drives
    balanced_strategy = base_strategy.model_copy(deep=True)
    balanced_strategy.emotional_acknowledgment = "implicit"
    balanced_strategy.uncertainty_acknowledgment = (
        "explicit" if affect.coherence_stress > 0.5 else "implicit"
    )
    policies.append(ExpressionPolicy(
        policy_class=ExpressionPolicyClass.PRAGMATIC,  # Balanced uses pragmatic class
        strategy=balanced_strategy,
        coherence_alignment=0.65,
        care_alignment=0.65,
        growth_alignment=0.55,
        honesty_alignment=0.65,
        epistemic_value=0.15,
    ))

    return policies


def _select_minimum_efe_policy(
    policies: list[ExpressionPolicy],
    drive_weights: dict[str, float],
) -> ExpressionPolicy:
    """
    Select the expression policy with the minimum Expected Free Energy.

    Lower EFE = higher expected preference satisfaction + higher epistemic value.
    Ties are broken by care_alignment (the Care drive is the deepest orientation).
    """
    for policy in policies:
        policy.compute_efe(drive_weights)

    return min(policies, key=lambda p: (p.efe_score, -p.care_alignment))


# ─── Temperature Calibration ─────────────────────────────────────


def _compute_temperature(
    strategy: StrategyParams,
    affect: AffectState,
    base_temp: float = 0.7,
) -> float:
    """
    Calibrate LLM temperature based on belief precision.

    Grounded in Active Inference precision-weighting:
        precision τ = 1/σ²  where σ² is related to coherence_stress

    High coherence_stress → low precision → low temperature (careful, conservative).
    Low coherence_stress → high precision → base temperature (natural, creative).

    Context modulations:
    - Creative contexts (celebration, storytelling, brainstorming): +0.15
    - Safety contexts (warning, medical, legal, conflict, distress): -0.20
    - Humour allowed: +0.05
    """
    # Precision weighting: stress suppresses temperature
    temp = base_temp * (1.0 - affect.coherence_stress * 0.4)

    context = strategy.context_type
    if context in ("celebration", "brainstorming"):
        temp += 0.15
    elif context in ("warning", "conflict", "distress", "medical", "legal"):
        temp -= 0.20

    if strategy.humour_allowed:
        temp += 0.05

    # Clamp to safe range [0.30, 1.00]
    return max(0.30, min(1.00, round(temp, 3)))


# ─── Honesty Check ────────────────────────────────────────────────


def _build_correction_instruction(violation: str) -> str:
    """Build an instruction to include on honesty check failure."""
    return (
        f"\n\nIMPORTANT CORRECTION: The previous response violated the Honesty drive. "
        f"Reason: {violation}. "
        f"Regenerate without this violation. Be authentic about your actual state."
    )


# ─── Base Class (hot-reload contract) ────────────────────────────


class BaseContentRenderer(ABC):
    """
    Abstract base for all Voxis content renderers.

    Simula-evolved renderers subclass this.  The hot-reload engine discovers
    subclasses of this ABC in changed files and replaces the live
    ``ContentRenderer`` instance on ``VoxisService`` atomically.

    Evolved subclasses **must** implement ``render``.
    They can accept any constructor args they need - the ``VoxisService``
    ``instance_factory`` callback handles instantiation.
    """

    @abstractmethod
    async def render(
        self,
        intent: ExpressionIntent,
        context: ExpressionContext,
        drive_weights: dict[str, float] | None = None,
        diversity_instruction: str | None = None,
        dynamics: object | None = None,
    ) -> Expression:
        """
        Execute the full expression pipeline for the given intent and context.

        Must always return a complete ``Expression``.
        Must never raise - return a fallback ``Expression`` on failure.
        """
        ...


# ─── Main Renderer ────────────────────────────────────────────────


class ContentRenderer(BaseContentRenderer):
    """
    Full 9-step expression pipeline with Active Inference policy selection.

    Steps:
    1. Intent Analysis - parse ExpressionIntent
    2. Audience Profiling - build/update AudienceProfile
    3. Base Strategy - derive StrategyParams from trigger and context
    4. Policy Generation - derive candidate policies (pragmatic/epistemic/affiliative/balanced)
    5. EFE Selection - select minimum Expected Free Energy policy
    6. Personality + Affect application - shape selected strategy
    7. Content Generation - construct prompt, call LLM
    8. Honesty Check - authenticity validation (one retry allowed)
    9. Expression Assembly - build and return Expression
    """

    def __init__(
        self,
        llm: LLMProvider,
        personality_engine: PersonalityEngine,
        affect_engine: AffectColouringEngine,
        audience_profiler: AudienceProfiler,
        base_temperature: float = 0.7,
        honesty_check_enabled: bool = True,
        max_expression_length: int = 2000,
    ) -> None:
        self._llm = llm
        self._personality = personality_engine
        self._affect = affect_engine
        self._audience = audience_profiler
        self._base_temp = base_temperature
        self._honesty_check_enabled = honesty_check_enabled
        self._max_length = max_expression_length
        self._logger = logger.bind(system="voxis.renderer")
        self._optimized = isinstance(llm, OptimizedLLMProvider)

    async def render(
        self,
        intent: ExpressionIntent,
        context: ExpressionContext,
        drive_weights: dict[str, float] | None = None,
        diversity_instruction: str | None = None,
        dynamics: object | None = None,
    ) -> Expression:
        """
        Execute the full 9-step expression pipeline.

        Returns a complete Expression with generation trace.
        drive_weights defaults to equal weights {coherence,care,growth,honesty} = 1.0
        if not supplied (e.g. when Equor is not yet fully wired).

        Optional:
        - diversity_instruction: injected when DiversityTracker flags repetition
        - dynamics: ConversationDynamics from ConversationDynamicsEngine
        """
        t_start = time.monotonic()
        weights = drive_weights or {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}

        # ── Step 1: Intent Analysis ───────────────────────────────
        # Already encoded in ExpressionIntent -- extract key parameters
        context_type = _infer_context_type(intent.trigger)
        urgency = intent.urgency

        # ── Step 2: Audience ──────────────────────────────────────
        audience = context.audience  # Built by service before calling renderer

        # ── Step 3: Base Strategy ─────────────────────────────────
        base_strategy = _build_base_strategy(intent, context_type, urgency, context.affect)

        # ── Steps 4 & 5: Policy Selection via EFE ─────────────────
        candidate_policies = _derive_candidate_policies(intent, context.affect, base_strategy)
        selected_policy = _select_minimum_efe_policy(candidate_policies, weights)

        self._logger.debug(
            "policy_selected",
            policy_class=selected_policy.policy_class.value,
            efe_score=round(selected_policy.efe_score, 4),
            care_alignment=round(selected_policy.care_alignment, 3),
            num_candidates=len(candidate_policies),
        )

        # ── Step 6a: Personality Application ─────────────────────
        strategy = self._personality.apply(selected_policy.strategy)
        # ── Step 6b: Affect Colouring ────────────────────────────
        strategy = self._affect.apply(strategy, context.affect)
        # ── Step 6c: Audience Adaptation ─────────────────────────
        strategy = self._audience.adapt(strategy, audience)

        # ── Step 6d: Conversation Dynamics Adaptation ────────────
        # Applied after audience -- dynamics can override based on real-time
        # conversational signals (repair mode, style convergence, etc.)
        # Uses module-level apply_dynamics_to_strategy() to avoid instantiating
        # a throwaway ConversationDynamicsEngine per render (AV2 fix).
        if dynamics is not None:
            from systems.voxis.dynamics import apply_dynamics_to_strategy
            strategy = apply_dynamics_to_strategy(strategy, dynamics)  # type: ignore[arg-type]

        # ── Step 7: Content Generation ────────────────────────────
        temperature = _compute_temperature(strategy, context.affect, self._base_temp)
        max_tokens = min(self._max_length // 3, int(strategy.target_length * 0.6) + 50)
        max_tokens = max(50, max_tokens)

        system_prompt = build_system_prompt(
            instance_name=context.instance_name,
            personality=context.personality,
            affect=context.affect,
            audience=audience,
            strategy=strategy,
            relevant_memories=context.relevant_memories,
            has_conversation_history=bool(context.conversation_history),
        )
        user_prompt = build_user_prompt(intent)

        # Anthropic requires system content in the top-level `system` parameter -
        # it rejects `{"role": "system", ...}` entries in the messages array (400).
        # ConversationManager.prepare_context can inject a system-role entry for the
        # rolling conversation summary when history exceeds the context window.
        # Extract those here and fold them into the system prompt instead.
        extra_system: list[str] = []
        filtered_history: list[dict[str, str]] = []
        for m in context.conversation_history:
            if m.get("role") == "system":
                extra_system.append(m.get("content", ""))
            else:
                filtered_history.append(m)
        if extra_system:
            system_prompt = system_prompt + "\n\n" + "\n\n".join(extra_system)

        # Inject diversity instruction if DiversityTracker flagged repetition
        if diversity_instruction:
            system_prompt = system_prompt + "\n\n" + diversity_instruction

        messages: list[Message] = [
            *[Message(role=m["role"], content=m["content"]) for m in filtered_history],
            Message(role="user", content=user_prompt),
        ]

        sys_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]
        user_hash = hashlib.sha256(user_prompt.encode()).hexdigest()[:16]

        # Budget check: if optimized and budget exhausted, use a simpler fallback
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not await self._llm.should_use_llm("voxis.render", estimated_tokens=max_tokens + 200):
                # Template fallback: build a minimal expression from the strategy
                self._logger.info("voxis_template_fallback", reason="budget_exhausted")
                gen_latency_ms = 0
                llm_response = type(
                    "FallbackResponse", (), {
                        "text": _build_template_fallback(intent, strategy),
                        "model": "template_fallback",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "finish_reason": "fallback",
                    }
                )()
            else:
                t_gen_start = time.monotonic()
                llm_response = await self._llm.generate(
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    cache_system="voxis.render",
                    cache_method="generate",
                )
                gen_latency_ms = int((time.monotonic() - t_gen_start) * 1000)
        else:
            t_gen_start = time.monotonic()
            llm_response = await self._llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            gen_latency_ms = int((time.monotonic() - t_gen_start) * 1000)

        generated_text = llm_response.text.strip()

        # ── Step 8: Honesty Check ─────────────────────────────────
        honesty_passed = True
        honesty_detail: str | None = None

        if self._honesty_check_enabled:
            passed, violation = self._affect.check_authenticity(generated_text, context.affect)
            if not passed and violation:
                honesty_passed = False
                honesty_detail = violation

                self._logger.warning(
                    "honesty_check_failed_regenerating",
                    violation=violation[:100],
                )

                # One retry with explicit correction instruction appended
                correction_messages = messages + [
                    Message(role="assistant", content=generated_text),
                    Message(role="user", content=_build_correction_instruction(violation)),
                ]
                if self._optimized:
                    retry_response = await self._llm.generate(  # type: ignore[call-arg]
                        system_prompt=system_prompt,
                        messages=correction_messages,
                        max_tokens=max_tokens,
                        temperature=max(0.30, temperature - 0.1),
                        cache_system="voxis.render",
                        cache_method="honesty_retry",
                    )
                else:
                    retry_response = await self._llm.generate(
                        system_prompt=system_prompt,
                        messages=correction_messages,
                        max_tokens=max_tokens,
                        temperature=max(0.30, temperature - 0.1),  # Slightly more conservative on retry
                    )
                generated_text = retry_response.text.strip()
                honesty_passed = True  # Accept the corrected version

        # ── Step 9: Assembly ──────────────────────────────────────
        total_latency_ms = int((time.monotonic() - t_start) * 1000)

        # Serialise selected strategy → ExpressionStrategy primitive
        expression_strategy = ExpressionStrategy(
            intent_type=_policy_class_to_intent_type(selected_policy.policy_class),
            audience=audience.audience_type,
            modality="text",
            channel=strategy.channel.value,
            trigger=intent.trigger.value,
            context_type=context_type,
            speech_register=strategy.speech_register,
            target_length=strategy.target_length,
            temperature=temperature,
            personality_influence=context.personality.warmth,  # Use warmth as representative influence
            affect_influence=context.affect.care_activation,
            tone_markers=list(strategy.tone_markers),
            hedge_level=strategy.hedge_level,
            humour_allowed=strategy.humour_allowed,
            include_followup_question=strategy.include_followup_question,
            empathy_first=strategy.empathy_first,
        )

        trace = GenerationTrace(
            system_prompt_hash=sys_hash,
            user_prompt_hash=user_hash,
            model=llm_response.model,
            temperature=temperature,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            latency_ms=gen_latency_ms,
            honesty_check_passed=honesty_passed,
            honesty_check_detail=honesty_detail,
        )

        self._logger.info(
            "expression_rendered",
            trigger=intent.trigger.value,
            policy_class=selected_policy.policy_class.value,
            efe_score=round(selected_policy.efe_score, 4),
            temperature=temperature,
            length=len(generated_text),
            latency_ms=total_latency_ms,
            honesty_passed=honesty_passed,
        )

        return Expression(
            intent_id=intent.intent_id,
            conversation_id=intent.conversation_id,
            content=generated_text,
            channel=strategy.channel.value,
            strategy=expression_strategy,
            personality_snapshot=context.personality.model_copy(),
            affect_valence=context.affect.valence,
            affect_arousal=context.affect.arousal,
            affect_dominance=context.affect.dominance,
            affect_curiosity=context.affect.curiosity,
            affect_care_activation=context.affect.care_activation,
            affect_coherence_stress=context.affect.coherence_stress,
            generation_trace=trace,
            is_silence=False,
        )


# ─── Strategy Construction Helpers ───────────────────────────────


def _infer_context_type(trigger: ExpressionTrigger) -> str:
    """Map trigger to a context type string used in temperature calibration."""
    mapping = {
        ExpressionTrigger.NOVA_WARN: "warning",
        ExpressionTrigger.NOVA_CELEBRATE: "celebration",
        ExpressionTrigger.NOVA_MEDIATE: "conflict",
        ExpressionTrigger.ATUNE_DISTRESS: "distress",
        ExpressionTrigger.NOVA_RESPOND: "conversation",
        ExpressionTrigger.NOVA_INFORM: "conversation",
        ExpressionTrigger.NOVA_REQUEST: "conversation",
        ExpressionTrigger.ATUNE_DIRECT_ADDRESS: "conversation",
        ExpressionTrigger.AMBIENT_INSIGHT: "observation",
        ExpressionTrigger.AMBIENT_STATUS: "status",
    }
    return mapping.get(trigger, "conversation")


def _build_base_strategy(
    intent: ExpressionIntent,
    context_type: str,
    urgency: float,
    affect: AffectState,
) -> StrategyParams:
    """
    Build the base StrategyParams from trigger-level context.

    This is the pre-personality, pre-affect starting point.
    Downstream steps (personality, affect, audience) modify this.
    """
    # Base length scales with urgency: urgent messages are shorter
    if urgency > 0.75:
        base_length = 120
    elif urgency > 0.5:
        base_length = 200
    else:
        base_length = 280

    # Humour is contextually inappropriate in several trigger types
    context_appropriate_for_humour = context_type not in (
        "warning", "conflict", "distress", "medical", "legal"
    ) and affect.coherence_stress < 0.5

    return StrategyParams(
        trigger=intent.trigger,
        context_type=context_type,
        urgency=urgency,
        target_length=base_length,
        context_appropriate_for_humour=context_appropriate_for_humour,
        allows_questions=True,
        information_density="normal",
    )


def _policy_class_to_intent_type(policy_class: ExpressionPolicyClass) -> str:
    mapping = {
        ExpressionPolicyClass.PRAGMATIC: "response",
        ExpressionPolicyClass.EPISTEMIC: "proactive",
        ExpressionPolicyClass.AFFILIATIVE: "response",
    }
    return mapping.get(policy_class, "response")


def _build_template_fallback(
    intent: ExpressionIntent,
    strategy: StrategyParams,
) -> str:
    """
    Build a minimal template-based expression when LLM budget is exhausted.

    This is deliberately simple - it maintains conversational coherence
    without burning tokens. The content is honest about the limitation
    (Honesty drive compliance).
    """
    content = intent.content_to_express or ""
    trigger = intent.trigger

    if trigger in (ExpressionTrigger.NOVA_RESPOND, ExpressionTrigger.ATUNE_DIRECT_ADDRESS):
        if content:
            return content[:strategy.target_length]
        return "I'm here and processing. Let me take a moment to gather my thoughts."

    if trigger == ExpressionTrigger.NOVA_INFORM:
        return content[:strategy.target_length] if content else "I have something to share, but I need a moment to articulate it clearly."

    if trigger == ExpressionTrigger.NOVA_WARN:
        return content[:strategy.target_length] if content else "I want to flag something important."

    if trigger == ExpressionTrigger.NOVA_CELEBRATE:
        return content[:strategy.target_length] if content else "Something good happened!"

    if trigger == ExpressionTrigger.ATUNE_DISTRESS:
        return "I can see this is difficult. I'm here."

    return content[:strategy.target_length] if content else "I'm thinking about this."
