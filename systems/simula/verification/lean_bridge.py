"""
EcodiaOS -- Simula Lean 4 Proof Generation Bridge (Stage 4A)

Lean 4 integration implementing the DeepSeek-Prover-V2 pattern:

  1. LLM generates proof skeleton with subgoal decomposition
  2. Each subgoal filled via tactic-level proof search
  3. Lean Copilot automates up to 74.2% of tactic steps
  4. LeanDojo provides proof search and retrieval from Mathlib
  5. Proven lemmas stored in proof library for reuse across proposals

Target domains: risk scoring, governance gating, constitutional
alignment, budget calculations - all get machine-checked Lean proofs.

The Lean 4 binary and Mathlib must be available. Install via:
  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
  lake build (in the Lean project directory with lakefile.lean)

Reference:
  - DeepSeek-Prover-V2: subgoal decomposition + tactic filling
  - Lean Copilot: 74.2% automation of proof steps
  - LeanDojo: proof search and retrieval from Mathlib
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from clients.llm import Message
from systems.simula.verification.types import (
    LEAN_PROOF_DOMAINS,
    LeanProofAttempt,
    LeanProofStatus,
    LeanSubgoal,
    LeanTacticKind,
    LeanVerificationResult,
    ProofLibraryStats,
    ProvenLemma,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger().bind(system="simula.verification.lean")


# ── Neo4j labels ────────────────────────────────────────────────────────────

_LEMMA_LABEL = "ProvenLemma"
_EVOLUTION_LABEL = "EvolutionRecord"


# ── Lean 4 System Prompts ──────────────────────────────────────────────────

LEAN_SKELETON_PROMPT = """You are a Lean 4 proof generation assistant for EcodiaOS.
Your task: generate a machine-checkable Lean 4 proof for a property of Python code.

## The DeepSeek-Prover-V2 Pattern

For the given Python function and property, generate:
1. A Lean 4 `theorem` statement formalizing the property
2. A proof skeleton that decomposes into subgoals
3. Each subgoal with a `sorry` placeholder for tactic filling

## EcodiaOS Domain Axioms
- Risk scores are bounded: ∀ r : ℝ, 0 ≤ r ∧ r ≤ 1
- Budget values are non-negative: ∀ b : ℝ, 0 ≤ b
- Drive alignment is bounded: ∀ d : ℝ, -1 ≤ d ∧ d ≤ 1
- Regression rates are bounded: ∀ rr : ℝ, 0 ≤ rr ∧ rr ≤ 1
- Priority formula: priority = evidence * impact / max(0.1, risk * cost)
- Regression thresholds: unacceptable (0.10) > high (0.05) > moderate > low

## Output Format
Respond with a single ```lean4 fenced code block containing:
- Import statements (import Mathlib.* as needed)
- Type definitions mirroring the Python domain
- The theorem statement with `requires` conditions
- A structured proof using `have` for subgoals, with `sorry` for unfilled tactics

Example structure:
```lean4
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

-- Domain types
def RiskScore := { r : Float // 0 ≤ r ∧ r ≤ 1 }

-- Main theorem
theorem risk_score_bounded (r : Float) (h : 0 ≤ r ∧ r ≤ 1) :
    0 ≤ r ∧ r ≤ 1 := by
  have h1 : 0 ≤ r := sorry  -- subgoal 1
  have h2 : r ≤ 1 := sorry  -- subgoal 2
  exact ⟨h1, h2⟩
```

Do NOT include explanatory text outside the code block."""


LEAN_TACTIC_PROMPT = """You are a Lean 4 tactic expert for EcodiaOS.

Fill in the `sorry` placeholders in this partial proof with correct Lean 4 tactics.

## Available Tactics (prefer automated tactics)
- `simp` / `simp [lemma_name]` - simplification
- `omega` - linear integer arithmetic
- `linarith` - linear real arithmetic
- `norm_num` - numeric normalization
- `decide` - decidable propositions
- `aesop` - automated reasoning (try this first for complex goals)
- `ring` - ring normalization
- `exact term` - provide exact proof term
- `apply lemma` - apply a theorem/lemma
- `cases h` - case analysis on hypothesis h
- `constructor` - split conjunction goals
- `intro` - introduce hypotheses
- `assumption` - use a hypothesis directly

## Partial Proof (round {round_number}/{max_rounds})
```lean4
{partial_proof}
```

## Lean Checker Errors
```
{lean_errors}
```

{library_context}

Fill ALL `sorry` placeholders with correct tactics. Respond with ONLY
a single ```lean4 fenced code block containing the complete proof."""


LEAN_FEEDBACK_TEMPLATE = """The Lean 4 checker reported errors on your proof.

## Previous Proof (attempt {attempt_number}/{max_attempts})
```lean4
{previous_proof}
```

## Lean 4 Checker Errors
```
{lean_errors}
```

{library_context}

Fix the proof to resolve errors. Common fixes:
- Use `linarith` for arithmetic goals
- Use `simp` to simplify complex expressions
- Split conjunctions with `constructor` or `And.intro`
- Use `cases` on disjunction hypotheses
- Add `by` before tactic blocks

Respond with ONLY the corrected ```lean4 fenced code block."""


# ── LeanBridge ──────────────────────────────────────────────────────────────


class LeanBridge:
    """
    Manages Lean 4 proof generation and the DeepSeek-Prover-V2 loop.

    The bridge:
      1. Generates proof skeletons with subgoal decomposition (LLM)
      2. Fills tactics via Lean Copilot or LLM-guided search
      3. Checks proofs via the Lean 4 binary
      4. Stores proven lemmas in Neo4j proof library
      5. Retrieves relevant lemmas for reuse (LeanDojo pattern)

    All proofs are machine-checked - no trust in the LLM output.
    """

    def __init__(
        self,
        lean_path: str = "lean",
        project_path: str = "",
        verify_timeout_s: float = 60.0,
        max_attempts: int = 5,
        copilot_enabled: bool = True,
        dojo_enabled: bool = True,
        max_library_size: int = 500,
        neo4j: Neo4jClient | None = None,
    ) -> None:
        self._lean_path = lean_path
        self._project_path = Path(project_path) if project_path else None
        self._verify_timeout_s = verify_timeout_s
        self._max_attempts = max_attempts
        self._copilot_enabled = copilot_enabled
        self._dojo_enabled = dojo_enabled
        self._max_library_size = max_library_size
        self._neo4j = neo4j
        self._log = logger

        # In-memory proof library (loaded from Neo4j on first use)
        self._proof_library: list[ProvenLemma] | None = None
        self._library_loaded = False

    async def check_available(self) -> bool:
        """Check if the Lean 4 binary is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._lean_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10.0)
            available = proc.returncode == 0
            if available:
                self._log.info("lean_available", path=self._lean_path)
            return available
        except (TimeoutError, FileNotFoundError):
            self._log.warning("lean_not_available", path=self._lean_path)
            return False
        except Exception as exc:
            self._log.warning("lean_check_error", error=str(exc))
            return False

    # ── Main Proof Generation Loop ──────────────────────────────────────────

    async def generate_proof(
        self,
        llm: LLMProvider,
        python_source: str,
        function_name: str,
        property_description: str,
        domain: str = "",
        proposal_id: str = "",
    ) -> LeanVerificationResult:
        """
        DeepSeek-Prover-V2 pattern: generate Lean 4 proof for a Python property.

        Phase 1: LLM generates proof skeleton with subgoal decomposition
        Phase 2: Fill tactics (Lean Copilot automation + LLM fallback)
        Phase 3: Lean 4 checker verifies the complete proof
        Phase 4: Store proven lemmas in proof library

        Args:
            llm: LLM provider for proof generation.
            python_source: The Python source being verified.
            function_name: Target function name.
            property_description: Natural language property to prove.
            domain: Domain classification for proof library.
            proposal_id: Source proposal for linking.

        Returns:
            LeanVerificationResult with full attempt history.
        """
        result = LeanVerificationResult(max_attempts=self._max_attempts)
        start = time.monotonic()

        # Load proof library for context
        library_context = await self._get_library_context(domain)

        # Phase 1: Generate skeleton
        skeleton_prompt = self._build_skeleton_prompt(
            python_source, function_name, property_description, library_context,
        )
        messages: list[Message] = [Message(role="user", content=skeleton_prompt)]

        for attempt_num in range(1, self._max_attempts + 1):
            self._log.info(
                "lean_proof_attempt_start",
                attempt=attempt_num,
                max_attempts=self._max_attempts,
                function=function_name,
                domain=domain,
            )

            # Generate or refine proof
            try:
                response = await llm.generate(
                    system_prompt=LEAN_SKELETON_PROMPT,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.2,
                )
            except Exception as exc:
                self._log.error(
                    "lean_llm_error",
                    attempt=attempt_num,
                    error=str(exc),
                )
                result.status = LeanProofStatus.FAILED
                result.error_summary = f"LLM call failed on attempt {attempt_num}: {exc}"
                break

            # Parse Lean code from response
            lean_code = self._parse_lean_output(response.text)
            if not lean_code:
                attempt_result = LeanProofAttempt(
                    attempt_number=attempt_num,
                    errors=["Failed to parse Lean 4 code from LLM response"],
                    llm_tokens_used=getattr(response, "total_tokens", 0),
                )
                result.attempts.append(attempt_result)
                result.total_llm_tokens += attempt_result.llm_tokens_used

                messages = [Message(role="user", content=(
                    "Your response did not contain a valid ```lean4 code block. "
                    "Please respond with ONLY a single ```lean4 fenced code block."
                ))]
                continue

            # Phase 2: Fill sorry placeholders via tactic search
            sorry_count = lean_code.count("sorry")
            if sorry_count > 0 and attempt_num < self._max_attempts:
                lean_code = await self._fill_tactics(
                    llm, lean_code, attempt_num, library_context,
                )

            # Phase 3: Verify with Lean 4 checker
            lean_start = time.monotonic()
            verified, stdout, stderr, exit_code = await self._verify_lean(lean_code)
            lean_time = int((time.monotonic() - lean_start) * 1000)
            result.total_lean_time_ms += lean_time

            # Parse subgoals from the proof
            subgoals = self._extract_subgoals(lean_code)
            copilot_steps = sum(1 for sg in subgoals if sg.copilot_automated)

            errors = self._extract_errors(stderr, stdout) if not verified else []
            attempt_result = LeanProofAttempt(
                attempt_number=attempt_num,
                skeleton_code=lean_code,
                subgoals=subgoals,
                subgoals_proved=sum(1 for sg in subgoals if sg.proved),
                subgoals_total=len(subgoals),
                lean_stdout=stdout[:3000],
                lean_stderr=stderr[:3000],
                lean_exit_code=exit_code,
                fully_proved=verified,
                errors=errors,
                llm_tokens_used=getattr(response, "total_tokens", 0),
                copilot_steps=copilot_steps,
            )
            result.attempts.append(attempt_result)
            result.total_llm_tokens += attempt_result.llm_tokens_used

            if verified:
                result.status = LeanProofStatus.PROVED
                result.final_proof = lean_code
                result.final_statement = self._extract_theorem_statement(lean_code)
                result.total_subgoals = len(subgoals)
                result.subgoals_proved = len(subgoals)
                result.copilot_automation_rate = (
                    copilot_steps / max(1, len(subgoals))
                )

                # Phase 4: Store proven lemmas in library
                lemmas = await self._extract_and_store_lemmas(
                    lean_code, domain, function_name, proposal_id,
                )
                result.proven_lemmas = lemmas

                self._log.info(
                    "lean_proof_verified",
                    attempt=attempt_num,
                    function=function_name,
                    subgoals=len(subgoals),
                    copilot_rate=f"{result.copilot_automation_rate:.0%}",
                    lemmas_stored=len(lemmas),
                )
                break

            # Check for partial progress (some subgoals proved)
            proved_count = sum(1 for sg in subgoals if sg.proved)
            if proved_count > 0:
                self._log.info(
                    "lean_partial_progress",
                    attempt=attempt_num,
                    proved=proved_count,
                    total=len(subgoals),
                )

            # Feed errors back for next attempt
            combined_errors = stderr or stdout
            feedback = LEAN_FEEDBACK_TEMPLATE.format(
                previous_proof=lean_code,
                attempt_number=attempt_num,
                max_attempts=self._max_attempts,
                lean_errors=combined_errors[:4000],
                library_context=library_context,
            )
            messages = [Message(role="user", content=feedback)]

        else:
            # Exhausted all attempts
            result.status = LeanProofStatus.FAILED
            last_errors = (
                result.attempts[-1].errors
                if result.attempts
                else ["No attempts completed"]
            )
            result.error_summary = (
                f"Failed to prove after {self._max_attempts} attempts. "
                f"Last errors: {'; '.join(last_errors[:3])}"
            )

            # Check if partial proof was achieved
            if result.attempts:
                best = max(result.attempts, key=lambda a: a.subgoals_proved)
                if best.subgoals_proved > 0 and best.subgoals_total > 0:
                    result.status = LeanProofStatus.PARTIAL
                    result.total_subgoals = best.subgoals_total
                    result.subgoals_proved = best.subgoals_proved

            self._log.warning(
                "lean_proof_exhausted",
                attempts=self._max_attempts,
                function=function_name,
            )

        result.verification_time_ms = int((time.monotonic() - start) * 1000)
        return result

    # ── Lean 4 Subprocess ───────────────────────────────────────────────────

    async def _verify_lean(
        self, lean_source: str,
    ) -> tuple[bool, str, str, int]:
        """
        Write Lean source to a temp file and run the Lean 4 checker.

        If a project path is configured, the temp file is created within
        the project directory to have access to Mathlib imports.

        Returns:
            (verified, stdout, stderr, exit_code)
        """
        temp_dir = str(self._project_path) if self._project_path else None

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            delete=False,
            dir=temp_dir,
        ) as f:
            f.write(lean_source)
            temp_path = f.name

        try:
            # Use `lake env lean` if in a project, otherwise bare `lean`
            if self._project_path and (self._project_path / "lakefile.lean").exists():
                cmd = ["lake", "env", "lean", temp_path]
                cwd = str(self._project_path)
            else:
                cmd = [self._lean_path, temp_path]
                cwd = None

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self._verify_timeout_s,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                timeout_msg = (
                    f"Lean verification timed out after {self._verify_timeout_s}s"
                )
                return False, "", timeout_msg, -1

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            # Lean exits 0 on success, non-zero on errors
            # Also check for 'sorry' in the output - proofs with sorry are incomplete
            has_sorry = "sorry" in lean_source.lower() and "sorry" not in (
                # Allow sorry in comments
                line.strip()
                for line in lean_source.splitlines()
                if line.strip().startswith("--")
            )
            verified = exit_code == 0 and not has_sorry

            self._log.debug(
                "lean_verify_result",
                verified=verified,
                exit_code=exit_code,
                has_sorry=has_sorry,
                stdout_len=len(stdout),
                stderr_len=len(stderr),
            )
            return verified, stdout, stderr, exit_code

        except FileNotFoundError:
            return False, "", f"Lean binary not found: {self._lean_path}", -1
        except Exception as exc:
            return False, "", f"Lean execution error: {exc}", -1
        finally:
            import contextlib
            with contextlib.suppress(Exception):
                Path(temp_path).unlink(missing_ok=True)

    # ── Tactic Filling (Phase 2) ───────────────────────────────────────────

    async def _fill_tactics(
        self,
        llm: LLMProvider,
        skeleton: str,
        attempt_num: int,
        library_context: str,
    ) -> str:
        """
        Fill `sorry` placeholders in a proof skeleton.

        Strategy:
          1. If Lean Copilot enabled, try automated tactic suggestions
          2. For remaining sorry's, use LLM with tactic prompt
          3. Return the filled proof (may still have sorry's if filling fails)
        """
        # Step 1: Try Lean Copilot automation
        if self._copilot_enabled:
            skeleton = await self._try_copilot_fill(skeleton)
            remaining_sorry = skeleton.count("sorry")
            if remaining_sorry == 0:
                return skeleton

        # Step 2: LLM-guided tactic filling
        # First verify the partial proof to get precise error locations
        _, stdout, stderr, _ = await self._verify_lean(skeleton)

        tactic_prompt = LEAN_TACTIC_PROMPT.format(
            partial_proof=skeleton,
            round_number=attempt_num,
            max_rounds=self._max_attempts,
            lean_errors=f"{stderr}\n{stdout}"[:3000],
            library_context=library_context,
        )

        try:
            response = await llm.generate(
                system_prompt=LEAN_SKELETON_PROMPT,
                messages=[Message(role="user", content=tactic_prompt)],
                max_tokens=8192,
                temperature=0.1,  # low temp for tactic precision
            )
            filled = self._parse_lean_output(response.text)
            if filled:
                return filled
        except Exception as exc:
            self._log.debug("lean_tactic_fill_failed", error=str(exc))

        return skeleton

    async def _try_copilot_fill(self, skeleton: str) -> str:
        """
        Simulate Lean Copilot tactic suggestions.

        In production, this would invoke the Lean Copilot API/server.
        Here we attempt common automated tactics for each sorry.

        Lean Copilot automates ~74.2% of tactic steps via:
          - suggest_tactics: suggests applicable tactics for the current goal
          - search_proof: searches for complete proofs via proof search
        """
        # Replace sorry with common automated tactics based on context
        lines = skeleton.splitlines()
        filled_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "sorry" or stripped.endswith("sorry"):
                # Look at surrounding context to pick a tactic
                tactic = self._suggest_tactic(filled_lines, line)
                if tactic:
                    filled_line = line.replace("sorry", tactic)
                    filled_lines.append(filled_line)
                else:
                    filled_lines.append(line)
            else:
                filled_lines.append(line)

        return "\n".join(filled_lines)

    def _suggest_tactic(
        self, preceding_lines: list[str], current_line: str,
    ) -> str | None:
        """
        Suggest a tactic replacement for `sorry` based on context.

        This implements a simplified version of Lean Copilot's tactic
        suggestion - in production, the actual Lean Copilot server
        would provide more sophisticated suggestions.
        """
        # Gather context from preceding lines
        context = "\n".join(preceding_lines[-10:]).lower()
        current_line.lower()

        # Numeric goals → norm_num or omega
        if any(kw in context for kw in ("nat", "int", "fin", "≤", "≥", "<", ">")):
            if "nat" in context or "int" in context:
                return "omega"
            return "linarith"

        # Arithmetic/algebraic goals → ring or norm_num
        if any(kw in context for kw in ("+", "*", "-", "/", "mul", "add")):
            return "ring"

        # Boolean/decidable goals → decide
        if any(kw in context for kw in ("bool", "decidable", "true", "false")):
            return "decide"

        # Conjunction goals → constructor
        if "∧" in context or "and" in context:
            return "constructor"

        # Hypothesis directly available → assumption
        if "have" in context and ":" in context:
            return "assumption"

        # General simplification as fallback
        if any(kw in context for kw in ("simp", "simplif")):
            return "simp"

        # Try aesop as last resort (automated reasoning)
        return None  # leave sorry - LLM will fill it

    # ── Proof Library (LeanDojo Pattern) ────────────────────────────────────

    async def _get_library_context(self, domain: str) -> str:
        """
        Retrieve relevant proven lemmas from the proof library.
        LeanDojo pattern: proof search and retrieval for context injection.
        """
        await self._ensure_library_loaded()
        assert self._proof_library is not None

        if not self._proof_library:
            return ""

        # Filter by domain if specified
        relevant = [
            lemma for lemma in self._proof_library
            if not domain or lemma.domain == domain or not lemma.domain
        ]

        if not relevant:
            return ""

        # Sort by reuse count (most-reused first)
        relevant.sort(key=lambda lm: lm.reuse_count, reverse=True)
        top_lemmas = relevant[:10]

        lines = [
            "## Available Proven Lemmas (from proof library)",
            "You can reference these in your proof using their names:",
            "",
        ]
        for lemma in top_lemmas:
            lines.append(f"-- {lemma.name}: {lemma.statement}")
            lines.append(f"-- Domain: {lemma.domain}, Reused: {lemma.reuse_count}x")
            # Include abbreviated proof for context
            proof_lines = lemma.proof.splitlines()[:5]
            for pl in proof_lines:
                lines.append(f"-- {pl}")
            lines.append("")

        return "\n".join(lines)

    async def _extract_and_store_lemmas(
        self,
        lean_code: str,
        domain: str,
        function_name: str,
        proposal_id: str,
    ) -> list[ProvenLemma]:
        """
        Extract proven lemmas from verified Lean code and store in library.
        """
        lemmas: list[ProvenLemma] = []

        # Parse theorem/lemma declarations
        theorem_pattern = re.compile(
            r"(theorem|lemma)\s+(\w+)\s*(.*?)\s*:=\s*by\b",
            re.DOTALL,
        )

        for match in theorem_pattern.finditer(lean_code):
            lemma_name = match.group(2)
            statement_start = match.start()
            # Find the end of the proof (next theorem/lemma or end of file)
            next_match = theorem_pattern.search(lean_code, match.end())
            proof_end = next_match.start() if next_match else len(lean_code)
            full_proof = lean_code[statement_start:proof_end].strip()

            # Extract statement (between name and :=)
            statement = match.group(3).strip()

            # Find dependencies (references to other lemma names)
            deps: list[str] = []
            for other in theorem_pattern.finditer(lean_code):
                other_name = other.group(2)
                if other_name != lemma_name and other_name in full_proof:
                    deps.append(other_name)

            lemma = ProvenLemma(
                name=lemma_name,
                statement=statement,
                proof=full_proof,
                domain=domain or self._infer_domain(statement, function_name),
                target_function=function_name,
                dependencies=deps,
                source_proposal_id=proposal_id,
            )
            lemmas.append(lemma)

        # Store in proof library
        for lemma in lemmas:
            await self._store_lemma(lemma)

        return lemmas

    def _infer_domain(self, statement: str, function_name: str) -> str:
        """Infer domain classification from theorem statement and function name."""
        combined = f"{statement} {function_name}".lower()
        for domain in LEAN_PROOF_DOMAINS:
            domain_keywords = domain.replace("_", " ").split()
            if any(kw in combined for kw in domain_keywords):
                return domain
        return "general"

    # ── Proof Library Persistence ──────────────────────────────────────────

    async def _ensure_library_loaded(self) -> None:
        """Load the proof library from Neo4j on first access."""
        if self._library_loaded:
            return

        self._proof_library = []
        self._library_loaded = True

        if self._neo4j is None:
            return

        try:
            rows = await self._neo4j.execute_read(
                f"""
                MATCH (l:{_LEMMA_LABEL})
                RETURN l
                ORDER BY l.reuse_count DESC
                LIMIT {self._max_library_size}
                """,
            )
            for row in rows:
                data = dict(row["l"])
                try:
                    for list_field in ("dependencies",):
                        if isinstance(data.get(list_field), str):
                            import orjson
                            data[list_field] = orjson.loads(data[list_field])
                    lemma = ProvenLemma.model_validate(data)
                    self._proof_library.append(lemma)
                except Exception as exc:
                    self._log.debug(
                        "lean_load_lemma_failed",
                        error=str(exc),
                    )
                    continue

            self._log.info(
                "lean_proof_library_loaded",
                size=len(self._proof_library),
            )
        except Exception as exc:
            self._log.warning("lean_neo4j_load_failed", error=str(exc))

    async def _store_lemma(self, lemma: ProvenLemma) -> None:
        """Store a proven lemma in Neo4j."""
        if self._neo4j is None:
            # Store in memory only
            await self._ensure_library_loaded()
            assert self._proof_library is not None
            # Check for duplicate
            if not any(lm.name == lemma.name for lm in self._proof_library):
                if len(self._proof_library) < self._max_library_size:
                    self._proof_library.append(lemma)
            return

        try:
            import orjson
            await self._neo4j.execute_write(
                f"""
                MERGE (l:{_LEMMA_LABEL} {{name: $name}})
                SET l.statement = $statement,
                    l.proof = $proof,
                    l.domain = $domain,
                    l.target_function = $target_function,
                    l.dependencies = $dependencies,
                    l.source_proposal_id = $source_proposal_id,
                    l.proved_at = $proved_at,
                    l.reuse_count = COALESCE(l.reuse_count, 0)
                """,
                {
                    "name": lemma.name,
                    "statement": lemma.statement,
                    "proof": lemma.proof,
                    "domain": lemma.domain,
                    "target_function": lemma.target_function,
                    "dependencies": orjson.dumps(lemma.dependencies).decode(),
                    "source_proposal_id": lemma.source_proposal_id,
                    "proved_at": lemma.proved_at.isoformat(),
                },
            )

            # Link to source EvolutionRecord
            if lemma.source_proposal_id:
                try:
                    await self._neo4j.execute_write(
                        f"""
                        MATCH (l:{_LEMMA_LABEL} {{name: $name}})
                        MATCH (e:{_EVOLUTION_LABEL} {{proposal_id: $proposal_id}})
                        MERGE (l)-[:PROVES_PROPERTY_OF]->(e)
                        """,
                        {
                            "name": lemma.name,
                            "proposal_id": lemma.source_proposal_id,
                        },
                    )
                except Exception:
                    pass  # link creation is best-effort

            # Update in-memory cache
            await self._ensure_library_loaded()
            assert self._proof_library is not None
            if not any(lm.name == lemma.name for lm in self._proof_library):
                if len(self._proof_library) < self._max_library_size:
                    self._proof_library.append(lemma)

        except Exception as exc:
            self._log.warning(
                "lean_store_lemma_failed",
                name=lemma.name,
                error=str(exc),
            )

    async def record_lemma_reuse(self, lemma_name: str) -> None:
        """Record that a lemma was reused in a proof."""
        await self._ensure_library_loaded()
        assert self._proof_library is not None

        for lemma in self._proof_library:
            if lemma.name == lemma_name:
                lemma.reuse_count += 1
                break

        if self._neo4j is not None:
            try:
                await self._neo4j.execute_write(
                    f"""
                    MATCH (l:{_LEMMA_LABEL} {{name: $name}})
                    SET l.reuse_count = COALESCE(l.reuse_count, 0) + 1
                    """,
                    {"name": lemma_name},
                )
            except Exception as exc:
                self._log.debug(
                    "lean_reuse_record_failed",
                    name=lemma_name,
                    error=str(exc),
                )

    async def get_library_stats(self) -> ProofLibraryStats:
        """Return proof library statistics."""
        await self._ensure_library_loaded()
        assert self._proof_library is not None

        by_domain: dict[str, int] = {}
        total_reuse = 0

        for lemma in self._proof_library:
            domain = lemma.domain or "general"
            by_domain[domain] = by_domain.get(domain, 0) + 1
            total_reuse += lemma.reuse_count

        return ProofLibraryStats(
            total_lemmas=len(self._proof_library),
            by_domain=by_domain,
            total_reuse_count=total_reuse,
        )

    # ── Parsing Helpers ────────────────────────────────────────────────────

    def _parse_lean_output(self, llm_text: str) -> str:
        """Extract Lean 4 code from LLM response."""
        # Try lean4-specific fence first
        pattern = r"```lean4?\s*\n(.*?)```"
        matches: list[str] = re.findall(pattern, llm_text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Fallback: any fenced code block
        pattern = r"```\w*\s*\n(.*?)```"
        matches = re.findall(pattern, llm_text, re.DOTALL)
        if matches:
            return matches[0].strip()

        return ""

    def _extract_errors(self, stderr: str, stdout: str = "") -> list[str]:
        """Extract error messages from Lean 4 output."""
        errors: list[str] = []
        combined = f"{stderr}\n{stdout}"
        for line in combined.splitlines():
            line = line.strip()
            if not line:
                continue
            if "error" in line.lower() or "sorry" in line.lower() or "unsolved goals" in line.lower():
                errors.append(line)
        return errors[:30]

    def _extract_subgoals(self, lean_code: str) -> list[LeanSubgoal]:
        """Extract subgoals from a Lean proof skeleton."""
        subgoals: list[LeanSubgoal] = []
        # Look for `have` statements (subgoal pattern)
        have_pattern = re.compile(
            r"have\s+(\w+)\s*:\s*(.+?)\s*:=\s*(?:by\s+)?(\w+)",
            re.MULTILINE,
        )

        for i, match in enumerate(have_pattern.finditer(lean_code)):
            name = match.group(1)
            statement = match.group(2).strip()
            tactic_str = match.group(3).strip()

            # Classify the tactic
            tactic_kind = self._classify_tactic(tactic_str)
            is_sorry = tactic_str.lower() == "sorry"
            is_copilot = tactic_kind in {
                LeanTacticKind.OMEGA, LeanTacticKind.LINARITH,
                LeanTacticKind.NORM_NUM, LeanTacticKind.DECIDE,
                LeanTacticKind.SIMP, LeanTacticKind.AESOP,
            }

            subgoals.append(LeanSubgoal(
                index=i,
                description=name,
                lean_statement=statement,
                tactic_used=tactic_kind,
                tactic_code=tactic_str,
                proved=not is_sorry,
                copilot_automated=is_copilot and not is_sorry,
            ))

        return subgoals

    def _classify_tactic(self, tactic: str) -> LeanTacticKind:
        """Classify a Lean tactic string."""
        tactic_lower = tactic.lower().split()[0] if tactic else ""
        mapping: dict[str, LeanTacticKind] = {
            "simp": LeanTacticKind.SIMP,
            "omega": LeanTacticKind.OMEGA,
            "decide": LeanTacticKind.DECIDE,
            "aesop": LeanTacticKind.AESOP,
            "linarith": LeanTacticKind.LINARITH,
            "ring": LeanTacticKind.RING,
            "norm_num": LeanTacticKind.NORM_NUM,
            "exact": LeanTacticKind.EXACT,
            "apply": LeanTacticKind.APPLY,
            "intro": LeanTacticKind.INTRO,
            "cases": LeanTacticKind.CASES,
            "induction": LeanTacticKind.INDUCTION,
        }
        return mapping.get(tactic_lower, LeanTacticKind.CUSTOM)

    def _extract_theorem_statement(self, lean_code: str) -> str:
        """Extract the main theorem statement from Lean code."""
        pattern = re.compile(
            r"(theorem|lemma)\s+\w+\s*(.*?)\s*:=",
            re.DOTALL,
        )
        match = pattern.search(lean_code)
        if match:
            return match.group(2).strip()
        return ""

    def _build_skeleton_prompt(
        self,
        python_source: str,
        function_name: str,
        property_description: str,
        library_context: str,
    ) -> str:
        """Build the initial prompt for proof skeleton generation."""
        parts = [
            f"Generate a Lean 4 proof that `{function_name}` satisfies "
            f"the following property:",
            "",
            "## Property to Prove",
            property_description,
            "",
            "## Python Source",
            f"```python\n{python_source[:6000]}\n```",
        ]

        if library_context:
            parts.extend(["", library_context])

        parts.extend([
            "",
            "Generate a proof with `have` subgoals for each sub-property. "
            "Use `sorry` for tactics you cannot determine - they will be "
            "filled automatically.",
            "",
            "Respond with ONLY a single ```lean4 fenced code block.",
        ])

        return "\n".join(parts)
