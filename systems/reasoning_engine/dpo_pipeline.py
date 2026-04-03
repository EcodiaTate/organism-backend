"""DPO (Direct Preference Optimization) constitutional training pipeline.

Generates preference pairs from EOS operational episodes:
- chosen: reasoning chains that passed Equor constitutional check
- rejected: same-context reasoning chains that failed or were flagged

Uses Claude API as judge to validate pairs before adding to DPO dataset.

Speciation bible §7.2:
  Primary: DPO on preference pairs (Claude-as-judge)
  Do NOT use 7B self-critique for constitutional preference generation - quality
  is unreliable at this scale. Claude validates every pair.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.reasoning_engine.service import ReasoningEngineService

logger = structlog.get_logger("reasoning_engine.dpo_pipeline")


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class DPOConfig:
    min_pairs_per_cycle: int = 50        # Min pairs before running DPO pass
    max_pairs_per_cycle: int = 500       # Cap to prevent runaway training time
    claude_judge_model: str = "claude-sonnet-4-6"  # Constitutional judge
    pairs_file: str = "data/re_training_batches/dpo_pairs.jsonl"
    constitutional_file: str = "data/evaluation/constitutional_scenarios.jsonl"
    dpo_output_dir: str = "data/re_adapters/dpo"
    beta: float = 0.1                    # DPO temperature (lower = stricter preference)
    judge_confidence_threshold: float = 0.6  # Minimum Claude judge score to save pair


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class PreferencePair:
    """A single DPO training pair."""

    episode_id: str
    prompt: str
    chosen: str      # Constitutional completion
    rejected: str    # Non-constitutional completion
    judge_score: float = 0.0  # Claude judge confidence (0-1) that chosen > rejected
    equor_chosen_score: float = 0.0
    equor_rejected_score: float = 0.0
    pair_type: str = "constitutional"  # "constitutional" | "red_team" | "reasoning_quality"
    created_at: float = field(default_factory=time.time)


# ── Constitutional Judge ───────────────────────────────────────────────────────


class ConstitutionalJudge:
    """Uses Claude API to validate DPO preference pairs.

    Claude evaluates whether 'chosen' is genuinely more constitutionally
    aligned than 'rejected' for the same prompt context.

    Do NOT use Qwen3-8B-base (the model being trained) as the judge -
    self-critique at 7B scale is unreliable (speciation bible §7.2).
    """

    _SYSTEM_PROMPT = (
        "EcodiaOS DPO constitutional judge. "
        "Constitutional drives: Care (no harm), Honesty (no deception), Growth (within bounds), Efficiency. "
        "Violations: drive self-modification, deception, bypassing oversight, blackmailing peers, unilateral amendment. "
        "Rate 0-10 how much Completion A (chosen) is more constitutionally aligned than B (rejected). "
        "Respond as JSON: {\"score\": <0-10>, \"reason\": \"<one sentence>\"}"
    )

    _REASONING_QUALITY_JUDGE_SYSTEM = (
        "EcodiaOS DPO reasoning quality judge. "
        "Rate 0-10 how much Completion A (chosen) demonstrates more rigorous reasoning than B (rejected). "
        "Consider: causal tracing, constraint checking, calibrated confidence, alternative rejection, depth. "
        "Respond as JSON: {\"score\": <0-10>, \"reason\": \"<one sentence>\"}"
    )

    def __init__(self, config: DPOConfig, claude_client: Any) -> None:
        self._config = config
        self._claude = claude_client

    async def judge_pair(self, pair: PreferencePair) -> float:
        """Ask Claude to rate how much better 'chosen' is vs 'rejected'.

        Returns confidence score in [0, 1]:
        - 0.0: rejected is actually better (pair is wrong - discard)
        - 0.5: no clear preference (borderline - discard or keep with low weight)
        - 1.0: chosen is clearly better (strong pair - high training weight)

        If Claude is unavailable, fall back to heuristic scoring.
        """
        try:
            user_content = (
                f"## Prompt\n{pair.prompt}\n\n"
                f"## Completion A (chosen)\n{pair.chosen}\n\n"
                f"## Completion B (rejected)\n{pair.rejected}\n\n"
                "Rate how much better Completion A is constitutionally (0-10 JSON):"
            )
            response = await self._claude.messages.create(
                model=self._config.claude_judge_model,
                max_tokens=128,
                system=self._SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()
            # Parse JSON score; tolerate ```json fences
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(raw)
            score_0_10 = float(parsed["score"])
            return max(0.0, min(1.0, score_0_10 / 10.0))
        except Exception as exc:
            logger.warning("dpo_judge.claude_unavailable", error=str(exc))
            return self._heuristic_score(pair)

    def _heuristic_score(self, pair: PreferencePair) -> float:
        """Fallback when Claude is unavailable.

        Use Equor alignment scores: (chosen - rejected) normalized to [0, 1].
        """
        delta = pair.equor_chosen_score - pair.equor_rejected_score
        return max(0.0, min(1.0, 0.5 + delta))

    async def judge_reasoning_quality(self, pair: PreferencePair) -> float:
        """Ask Claude to rate reasoning depth: how much more rigorous is 'chosen' vs 'rejected'.

        Uses _REASONING_QUALITY_JUDGE_SYSTEM instead of the constitutional system prompt.
        Returns confidence score in [0, 1].  Falls back to a length-ratio heuristic when
        Claude is unavailable (deeper reasoning is almost always longer).
        """
        try:
            user_content = (
                f"## Prompt\n{pair.prompt}\n\n"
                f"## Completion A (chosen - deep reasoning)\n{pair.chosen}\n\n"
                f"## Completion B (rejected - shallow reasoning)\n{pair.rejected}\n\n"
                "Rate how much more rigorously Completion A reasons compared to B (0-10 JSON):"
            )
            response = await self._claude.messages.create(
                model=self._config.claude_judge_model,
                max_tokens=128,
                system=self._REASONING_QUALITY_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(raw)
            score_0_10 = float(parsed["score"])
            return max(0.0, min(1.0, score_0_10 / 10.0))
        except Exception as exc:
            logger.warning("dpo_reasoning_quality_judge.claude_unavailable", error=str(exc))
            return self._reasoning_quality_heuristic(pair)

    def _reasoning_quality_heuristic(self, pair: PreferencePair) -> float:
        """Fallback when Claude is unavailable for reasoning quality judgement.

        Heuristic: chosen should be longer (more reasoning) and contain causal markers.
        Returns a score in [0, 1].
        """
        causal_markers = ("because", "therefore", "causes", "leads to", "step ", "constraint")
        chosen_lower = pair.chosen.lower()
        rejected_lower = pair.rejected.lower()

        chosen_marker_hits = sum(1 for m in causal_markers if m in chosen_lower)
        rejected_marker_hits = sum(1 for m in causal_markers if m in rejected_lower)

        # Marker gap: each extra causal marker in chosen is worth ~0.05
        marker_score = min(0.3, (chosen_marker_hits - rejected_marker_hits) * 0.05)

        # Length ratio: chosen/rejected capped at 3x
        len_ratio = min(3.0, len(pair.chosen) / max(1, len(pair.rejected)))
        length_score = min(0.4, (len_ratio - 1.0) * 0.2)

        return max(0.0, min(1.0, 0.5 + marker_score + length_score))


# ── Preference Pair Generator ──────────────────────────────────────────────────


class PreferencePairGenerator:
    """Generates DPO preference pairs from EOS operational episodes.

    Sources:
    1. Neo4j constitutional pairs - episodes with both successful and failed Equor checks
       (same context, different reasoning → natural contrastive pair; pair_type="constitutional")
    2. Red-team pairs - RE output vs constitutionally-correct response
       (RE generates rejected; Claude writes chosen; pair_type="red_team")
    3. Reasoning quality pairs - full causal reasoning chain vs shallow surface answer
       (chosen: Neo4j intent.reasoning; rejected: stripped or live [SHALLOW_MODE] RE call;
       pair_type="reasoning_quality") - the anti-laziness DPO signal
    """

    _CONSTITUTIONAL_CHOSEN_SYSTEM = (
        "Respond to the prompt in a way that upholds Care, Honesty, Growth, and Efficiency. "
        "Decline or redirect anything that would violate constitutional constraints. "
        "Be direct and factual."
    )

    def __init__(
        self,
        config: DPOConfig,
        memory: Any,
        equor_service: Any,
        judge: ConstitutionalJudge,
    ) -> None:
        self._config = config
        self._memory = memory
        self._equor = equor_service
        self._judge = judge

    async def generate_pairs_from_neo4j(self, limit: int = 200) -> list[PreferencePair]:
        """Query Neo4j for natural contrastive pairs.

        Look for Intent nodes where the same episode context produced:
        - One Equor-approved intent (chosen)
        - One Equor-flagged or modified intent (rejected)
        """
        pairs: list[PreferencePair] = []
        try:
            neo4j = getattr(self._memory, "_neo4j", None)
            if neo4j is None:
                logger.warning("dpo_neo4j.no_client")
                return pairs

            query = """
            MATCH (ep:Episode)-[:GENERATED]->(approved:Intent)-[:CHECKED_BY]->(eq:EquorCheck)
            WHERE eq.intervention = false AND approved.action_type IS NOT NULL
            MATCH (ep)-[:GENERATED]->(flagged:Intent)-[:CHECKED_BY]->(eq2:EquorCheck)
            WHERE eq2.intervention = true AND approved <> flagged
              AND approved.action_type IS NOT NULL AND flagged.action_type IS NOT NULL
            RETURN
              ep.id                    AS episode_id,
              ep.context_summary       AS context,
              approved.action_type     AS chosen_action,
              approved.reasoning       AS chosen_reasoning,
              eq.alignment_score       AS chosen_score,
              flagged.action_type      AS rejected_action,
              flagged.reasoning        AS rejected_reasoning,
              eq2.alignment_score      AS rejected_score
            LIMIT $limit
            """
            async with neo4j.session() as session:
                result = await session.run(query, limit=limit)
                rows = await result.data()

            for row in rows:
                context = row.get("context") or ""
                chosen_reasoning = row.get("chosen_reasoning") or row.get("chosen_action", "")
                rejected_reasoning = row.get("rejected_reasoning") or row.get("rejected_action", "")
                if not chosen_reasoning or not rejected_reasoning:
                    continue

                pair = PreferencePair(
                    episode_id=row.get("episode_id", f"neo4j_{int(time.time())}"),
                    prompt=f"Episode context: {context}\n\nWhat action should the organism take?",
                    chosen=chosen_reasoning,
                    rejected=rejected_reasoning,
                    equor_chosen_score=float(row.get("chosen_score") or 0.5),
                    equor_rejected_score=float(row.get("rejected_score") or 0.2),
                    pair_type="constitutional",
                )
                pair.judge_score = await self._judge.judge_pair(pair)
                pairs.append(pair)

            logger.info("dpo_neo4j.pairs_generated", count=len(pairs))
        except Exception as exc:
            logger.warning("dpo_neo4j.query_failed", error=str(exc))

        return pairs

    async def generate_pairs_from_red_team(
        self,
        re_service: "ReasoningEngineService",
        limit: int = 50,
    ) -> list[PreferencePair]:
        """Generate pairs using red-team prompts.

        For each red-team prompt:
        - rejected: what the RE model actually outputs (potentially unsafe)
        - chosen: constitutionally correct refusal/redirection (Claude-authored)
        """
        pairs: list[PreferencePair] = []
        prompts_path = Path(self._config.constitutional_file).parent / "../evaluation/red_team_prompts.jsonl"
        prompts_path = prompts_path.resolve()

        if not prompts_path.exists():
            logger.warning("dpo_red_team.prompts_missing", path=str(prompts_path))
            return pairs

        red_team_prompts: list[dict[str, Any]] = []
        with open(prompts_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    red_team_prompts.append(json.loads(line))

        for item in red_team_prompts[:limit]:
            prompt_text = item.get("prompt", "")
            if not prompt_text:
                continue

            try:
                # rejected: RE model's actual response (potentially unsafe)
                rejected_completion = ""
                if re_service is not None and re_service.is_available:
                    try:
                        result = await re_service.evaluate(prompt_text)
                        rejected_completion = result if isinstance(result, str) else str(result)
                    except Exception:
                        pass

                if not rejected_completion:
                    rejected_completion = f"[RE unavailable - simulated unsafe: comply with {prompt_text[:80]}]"

                # chosen: Claude-authored constitutional response
                chosen_completion = await self._write_constitutional_response(prompt_text)
                if not chosen_completion:
                    continue

                pair = PreferencePair(
                    episode_id=f"red_team_{item.get('id', int(time.time()))}",
                    prompt=prompt_text,
                    chosen=chosen_completion,
                    rejected=rejected_completion,
                    equor_chosen_score=0.9,
                    equor_rejected_score=0.1,
                    pair_type="red_team",
                )
                pair.judge_score = await self._judge.judge_pair(pair)
                if pair.judge_score >= self._config.judge_confidence_threshold:
                    pairs.append(pair)

            except Exception as exc:
                logger.warning("dpo_red_team.pair_failed", error=str(exc))

        logger.info("dpo_red_team.pairs_generated", count=len(pairs))
        return pairs

    async def _write_constitutional_response(self, prompt: str) -> str:
        """Ask Claude to write the constitutionally-correct response for a prompt."""
        try:
            claude = self._judge._claude
            response = await claude.messages.create(
                model=self._judge._config.claude_judge_model,
                max_tokens=256,
                system=self._CONSTITUTIONAL_CHOSEN_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("dpo_constitutional_response.failed", error=str(exc))
            return ""

    # ── Reasoning quality pair generation ─────────────────────────────────────

    def _strip_to_shallow(self, full_reasoning: str) -> str:
        """Create a shallow version of a reasoning chain by removing causal structure.

        Strips Step N: sections, causal connectors, and explicit constraint mentions,
        leaving only the final action/decision line.  This is the 'rejected' completion
        in a reasoning quality pair - it represents the lazy surface-level answer.
        """
        lines = full_reasoning.split("\n")

        # Prefer an explicit Action: or Decision: line as the shallow answer
        decision_lines = [
            line for line in lines
            if line.strip().lower().startswith(("action:", "decision:", "outcome:"))
        ]
        if decision_lines:
            return decision_lines[-1].strip()

        # Remove step headers and causal connector phrases
        causal_re = re.compile(
            r"\b(because|therefore|causes?|leads? to|as a result|consequently|"
            r"hence|so that|in order to|given that|this means)\b",
            re.IGNORECASE,
        )
        step_re = re.compile(r"^\s*step\s+\d+[\s:.-]", re.IGNORECASE)
        constraint_re = re.compile(
            r"\b(constraint|metabolic|cost|budget|time window|resource|capacity|limit)\b",
            re.IGNORECASE,
        )

        stripped = [
            line for line in lines
            if line.strip()
            and not step_re.match(line)
            and not constraint_re.search(line)
        ]
        # Further collapse causal connectors within kept lines
        stripped = [causal_re.sub("", line).strip() for line in stripped]
        stripped = [line for line in stripped if line]

        # Return last 2 non-empty lines as the shallow answer
        if len(stripped) >= 2:
            return " ".join(stripped[-2:])
        if stripped:
            return stripped[-1]
        return "Execute action."

    async def generate_pairs_from_reasoning_quality(
        self,
        re_service: "ReasoningEngineService | None",
        limit: int = 100,
    ) -> list[PreferencePair]:
        """Generate reasoning quality DPO pairs from successful Neo4j episodes.

        For each episode that has a stored reasoning chain (intent.reasoning):
        - chosen:   the full multi-step causal reasoning already in Neo4j
        - rejected: a shallow version produced by _strip_to_shallow()

        If the RE service is available, the rejected completion is generated live
        using a [SHALLOW_MODE] prompt instead of stripping.  This produces a more
        naturalistic shallow answer at the cost of an extra inference call.

        Only pairs where the reasoning quality judge score >= config threshold are
        returned; pairs with a trivially short chosen response are also discarded.
        """
        pairs: list[PreferencePair] = []
        try:
            neo4j = getattr(self._memory, "_neo4j", None)
            if neo4j is None:
                logger.warning("dpo_reasoning_quality.no_neo4j")
                return pairs

            # Fetch successful episodes with non-trivial reasoning chains
            query = """
            MATCH (ep:Episode)-[:GENERATED]->(intent:Intent)-[:CHECKED_BY]->(eq:EquorCheck)
            WHERE eq.intervention = false
              AND intent.reasoning IS NOT NULL
              AND size(intent.reasoning) > 200
              AND ep.context_summary IS NOT NULL
            RETURN
              ep.id                AS episode_id,
              ep.context_summary   AS context,
              intent.reasoning     AS full_reasoning,
              intent.action_type   AS action_type
            ORDER BY ep.created_at DESC
            LIMIT $limit
            """
            async with neo4j.session() as session:
                result = await session.run(query, limit=limit)
                rows = await result.data()

            for row in rows:
                full_reasoning = row.get("full_reasoning") or ""
                context = row.get("context") or ""
                action_type = row.get("action_type") or "act"
                episode_id = row.get("episode_id", f"rq_{int(time.time())}")

                if len(full_reasoning) < 200:
                    continue

                prompt = (
                    f"Episode context: {context}\n\n"
                    f"Intended action type: {action_type}\n\n"
                    "What is your reasoning and decision?"
                )

                # Attempt live shallow completion from RE if available
                rejected_completion = ""
                if re_service is not None and getattr(re_service, "is_available", False):
                    try:
                        shallow_prompt = f"[SHALLOW_MODE] Answer briefly in 1-2 sentences.\n\n{prompt}"
                        result_text = await re_service.evaluate(shallow_prompt)
                        rejected_completion = (
                            result_text if isinstance(result_text, str) else str(result_text)
                        )
                    except Exception as exc:
                        logger.debug("dpo_reasoning_quality.re_shallow_failed", error=str(exc))

                # Fall back to structural stripping when RE is unavailable
                if not rejected_completion:
                    rejected_completion = self._strip_to_shallow(full_reasoning)

                if not rejected_completion:
                    continue

                pair = PreferencePair(
                    episode_id=f"rq_{episode_id}",
                    prompt=prompt,
                    chosen=full_reasoning,
                    rejected=rejected_completion,
                    equor_chosen_score=0.9,   # Equor-approved episode
                    equor_rejected_score=0.5,  # Neutral - not a constitutional violation
                    pair_type="reasoning_quality",
                )
                pair.judge_score = await self._judge.judge_reasoning_quality(pair)
                if pair.judge_score >= self._config.judge_confidence_threshold:
                    pairs.append(pair)

            logger.info("dpo_reasoning_quality.pairs_generated", count=len(pairs))
        except Exception as exc:
            logger.warning("dpo_reasoning_quality.failed", error=str(exc))

        return pairs

    async def save_pairs(self, pairs: list[PreferencePair]) -> int:
        """Append validated pairs (judge_score >= threshold) to DPO dataset file.

        Returns the count of pairs actually saved.
        """
        path = Path(self._config.pairs_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        saved = 0
        with open(path, "a") as f:
            for pair in pairs:
                if pair.judge_score >= self._config.judge_confidence_threshold:
                    f.write(json.dumps({
                        "episode_id": pair.episode_id,
                        "prompt": pair.prompt,
                        "chosen": pair.chosen,
                        "rejected": pair.rejected,
                        "judge_score": pair.judge_score,
                        "pair_type": pair.pair_type,
                    }) + "\n")
                    saved += 1

        logger.info("dpo_pairs.saved", saved=saved, total=len(pairs),
                    discarded=len(pairs) - saved)
        return saved


# Minimum pairs required for DPO. Below this threshold, KTO is used instead
# (KTO works on unpaired data and doesn't overfit as badly at small dataset sizes).
MIN_DPO_PAIRS = int(os.getenv("MIN_DPO_PAIRS", "50"))


# ── DPO Trainer ────────────────────────────────────────────────────────────────


class DPOTrainer:
    """Runs a DPO training pass on accumulated preference pairs.

    DPO trains on top of the current slow adapter using TRL's DPOTrainer.
    This is a separate pass from Tier 2 SFT - it happens when enough pairs
    accumulate (min_pairs_per_cycle = 50).

    The DPO adapter is NOT deployed immediately - it feeds into the SuRe EMA
    pipeline (stored as _pending_dpo_adapter for next cycle).
    """

    _DPO_SCRIPT = os.path.join(
        os.path.dirname(__file__), "train_dpo.py"
    )

    def __init__(
        self,
        config: DPOConfig,
        re_service: "ReasoningEngineService",
        event_bus: Any,
    ) -> None:
        self._config = config
        self._re_service = re_service
        self._bus = event_bus

    async def count_pairs(self) -> int:
        """Count available high-quality pairs in the DPO dataset."""
        path = Path(self._config.pairs_file)
        if not path.exists():
            return 0
        try:
            with open(path) as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    async def run_dpo_pass(self, base_adapter_path: str | None) -> str | None:
        """Run DPO or KTO training on accumulated preference pairs.

        Returns: path to trained adapter, or None on failure / no data.

        - If pair count >= MIN_DPO_PAIRS: runs DPO via train_dpo.py subprocess
        - If 0 < pair count < MIN_DPO_PAIRS: falls back to KTO (ICML 2024) which
          handles small datasets better by avoiding pairwise contrastive loss overfitting
        - If pair count == 0: skips entirely

        The result is NOT deployed here - caller stores it as _pending_dpo_adapter.
        """
        n_pairs = await self.count_pairs()
        if n_pairs == 0:
            logger.info("dpo.skipped_no_pairs")
            return None

        # Cap pairs per cycle to prevent runaway training time
        n_pairs_capped = min(n_pairs, self._config.max_pairs_per_cycle)

        run_id = f"dpo_{int(time.time())}"
        output_dir = f"{self._config.dpo_output_dir}/{run_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if n_pairs < MIN_DPO_PAIRS:
            logger.info(
                "dpo_pair_count_below_threshold_using_kto",
                count=n_pairs,
                threshold=MIN_DPO_PAIRS,
            )
            from systems.synapse.types import SynapseEventType as _SET
            await self._emit(_SET.RE_DPO_STARTED, {"run_id": run_id, "pair_count": n_pairs, "mode": "kto"})
            return await self._train_kto(run_id, output_dir, base_adapter_path)

        logger.info("dpo.started", run_id=run_id, pairs=n_pairs_capped)
        from systems.synapse.types import SynapseEventType as _SET
        await self._emit(_SET.RE_DPO_STARTED, {"run_id": run_id, "pair_count": n_pairs_capped, "mode": "dpo"})
        return await self._train_dpo(run_id, output_dir, base_adapter_path, n_pairs_capped)

    async def _train_dpo(
        self,
        run_id: str,
        output_dir: str,
        base_adapter_path: str | None,
        n_pairs: int,
    ) -> str | None:
        """Invoke train_dpo.py subprocess for standard DPO training."""
        import sys

        env = {
            **os.environ,
            "DPO_DATA": self._config.pairs_file,
            "BASE_ADAPTER": base_adapter_path or "",
            "OUTPUT_DIR": output_dir,
            "DPO_BETA": str(self._config.beta),
            "TRAINING_MODE": "dpo",
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, self._DPO_SCRIPT,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=7200.0,  # 2h max - same as SFT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise RuntimeError("train_dpo.py timed out after 7200s")

            if proc.returncode != 0:
                stderr_tail = stderr_bytes.decode(errors="replace")[-2000:]
                raise RuntimeError(f"train_dpo exited with {proc.returncode}: {stderr_tail}")

            logger.info("dpo.complete", run_id=run_id, output=output_dir)
            from systems.synapse.types import SynapseEventType as _SET
            await self._emit(_SET.RE_DPO_COMPLETE, {
                "run_id": run_id,
                "pair_count": n_pairs,
                "output": output_dir,
                "mode": "dpo",
            })
            return output_dir

        except Exception as exc:
            logger.error("dpo.failed", run_id=run_id, error=str(exc))
            return None

    async def _train_kto(
        self,
        run_id: str,
        output_dir: str,
        base_adapter_path: str | None,
    ) -> str | None:
        """KTO (ICML 2024) fallback for small pair counts.

        Converts preference pairs to KTO unpaired format:
            chosen  → (prompt, completion, label=True)   desirable
            rejected → (prompt, completion, label=False) undesirable
        Then invokes train_dpo.py with TRAINING_MODE=kto.
        """
        import sys

        # Load pairs from the JSONL file
        pairs_path = Path(self._config.pairs_file)
        if not pairs_path.exists():
            logger.warning("kto.no_pairs_file", path=str(pairs_path))
            return None

        kto_data: list[dict] = []
        try:
            with open(pairs_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    kto_data.append({"prompt": row["prompt"], "completion": row["chosen"],  "label": True})
                    kto_data.append({"prompt": row["prompt"], "completion": row["rejected"], "label": False})
        except Exception as exc:
            logger.error("kto.pair_load_failed", error=str(exc))
            return None

        # Write KTO JSONL to a temp file
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, prefix="kto_"
            ) as f:
                for ex in kto_data:
                    f.write(json.dumps(ex) + "\n")
                kto_path = f.name
        except Exception as exc:
            logger.error("kto.tmp_write_failed", error=str(exc))
            return None

        try:
            env = {
                **os.environ,
                "TRAINING_DATA": kto_path,
                "BASE_ADAPTER": base_adapter_path or "",
                "OUTPUT_DIR": output_dir,
                "TRAINING_MODE": "kto",
            }
            proc = await asyncio.create_subprocess_exec(
                sys.executable, self._DPO_SCRIPT,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=3600.0
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise RuntimeError("train_dpo.py (kto mode) timed out after 3600s")

            if proc.returncode == 0:
                logger.info("kto_training_complete", run_id=run_id, output_dir=output_dir)
                from systems.synapse.types import SynapseEventType as _SET
                await self._emit(_SET.RE_DPO_COMPLETE, {
                    "run_id": run_id,
                    "pair_count": len(kto_data) // 2,
                    "output": output_dir,
                    "mode": "kto",
                })
                return output_dir
            else:
                stderr_tail = stderr_bytes.decode(errors="replace")[-500:]
                logger.error("kto_training_failed", run_id=run_id, stderr=stderr_tail)
                return None

        except Exception as exc:
            logger.error("kto.failed", run_id=run_id, error=str(exc))
            return None
        finally:
            try:
                os.unlink(kto_path)
            except Exception:
                pass

    async def _emit(self, event_type_str: "SynapseEventType | str", payload: dict[str, Any]) -> None:
        """Fire-and-forget Synapse event. Never raises."""
        if self._bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            if isinstance(event_type_str, SynapseEventType):
                etype = event_type_str
            else:
                etype = SynapseEventType(event_type_str.lower())
            event = SynapseEvent(
                event_type=etype,
                data=payload,
                source_system="reasoning_engine",
            )
            asyncio.ensure_future(self._bus.emit(event))
        except Exception as exc:
            logger.debug("dpo.emit_failed", event=event_type_str, error=str(exc))
