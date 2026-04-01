"""
EcodiaOS -- Simula LILO Library Learning (Stage 3C)

Extracts reusable code abstractions from successful evolution proposals.
Named after the LILO (Library Learning) pattern:

  1. LLM generates code for a proposal (standard code agent path)
  2. Stitch-like extraction: identify common sub-expressions across
     multiple successful proposals (lambda-abstraction extraction)
  3. AutoDoc-style naming: LLM names the extracted abstractions
  4. Store in Neo4j as :LibraryAbstraction nodes linked to :EvolutionRecord
  5. Feed the abstraction library into the code agent's system prompt

Over time, Simula builds a reusable library of patterns discovered
from its own evolution history. This reduces generation tokens (reuse
instead of regenerate) and increases code quality (proven patterns).

Periodic consolidation: merge similar abstractions, prune unused ones.
Metric: code reuse rate, reduction in generation tokens.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import re
import time
from collections import Counter
from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.simula.verification.types import (
    AbstractionExtractionResult,
    AbstractionKind,
    LibraryAbstraction,
    LibraryStats,
)

if TYPE_CHECKING:
    from pathlib import Path

    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger().bind(system="simula.lilo")


def _sanitize(value: str, max_len: int = 2000) -> str:
    """
    Sanitize a stored string before injecting it into an LLM prompt.

    Strips leading/trailing whitespace, truncates to max_len, and removes
    markdown heading sequences (e.g. "## IRON RULE") that could be parsed
    as prompt-level instructions by the LLM.

    This is a defence-in-depth measure: stored data (Neo4j abstractions,
    postmortems, log signals) must not be able to override prompt instructions.
    """
    if not isinstance(value, str):
        value = str(value)
    value = value[:max_len].strip()
    lines = value.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if re.match(r"^#{1,6}\s+", stripped):
            # Neutralise by prepending a zero-width non-breaking space so the
            # LLM still sees the content but the Markdown parser won't treat
            # it as a heading, and the model is less likely to act on it as
            # a directive.
            cleaned.append("\ufeff" + line)
        else:
            cleaned.append(line)
    return "\n".join(cleaned)

# Neo4j labels
_ABSTRACTION_LABEL = "LibraryAbstraction"
_EVOLUTION_LABEL = "EvolutionRecord"

# Minimum function body length to consider for extraction (lines)
_MIN_FUNCTION_LINES = 3

# Minimum occurrences across proposals for a pattern to qualify
_MIN_PATTERN_OCCURRENCES = 2

# Confidence decay rate per consolidation cycle for unused abstractions
_CONFIDENCE_DECAY = 0.05

# Prune threshold: abstractions below this confidence get removed
_PRUNE_THRESHOLD = 0.15

# Maximum abstractions in the library (prevents unbounded growth)
_MAX_LIBRARY_SIZE = 200


class LiloLibraryEngine:
    """
    LILO Library Learning for Simula.

    Builds and maintains a reusable abstraction library from successful
    evolution proposals. Extractions are stored in Neo4j and fed back
    into the code agent's system prompt for future proposals.

    Flow:
      extract_from_proposals() - after proposals succeed, extract patterns
      get_library_prompt()     - inject library into code agent prompt
      consolidate()            - merge/prune on idle cycles
      get_stats()              - library health metrics
    """

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        llm: LLMProvider | None = None,
        codebase_root: Path | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._root = codebase_root
        self._log = logger

        from systems.synapse.sentinel import ErrorSentinel
        self._sentinel = ErrorSentinel("simula.lilo")

        # In-memory library cache (loaded from Neo4j on first use)
        self._library: list[LibraryAbstraction] | None = None
        self._library_loaded: bool = False

    # ─── Public API ──────────────────────────────────────────────────────────

    async def extract_from_proposals(
        self,
        proposal_ids: list[str],
        files_changed: dict[str, list[str]],  # proposal_id -> files
    ) -> AbstractionExtractionResult:
        """
        Extract reusable abstractions from a batch of successful proposals.

        Steps:
          1. Parse all changed files, extract function ASTs
          2. Find common patterns (normalized sub-expressions)
          3. For qualifying patterns, use LLM to name and describe them
          4. Store in Neo4j, linked to source EvolutionRecords
          5. Merge into existing library if similar abstraction exists

        Called by SimulaService after proposals are successfully applied.
        """
        start = time.monotonic()

        # Step 1: Extract all functions from changed files
        all_functions: list[_ExtractedFunction] = []
        for proposal_id, files in files_changed.items():
            for fpath in files:
                functions = await self._extract_functions(fpath, proposal_id)
                all_functions.extend(functions)

        if not all_functions:
            return AbstractionExtractionResult(
                total_proposals_analyzed=len(proposal_ids),
                total_time_ms=int((time.monotonic() - start) * 1000),
            )

        # Step 2: Find common patterns via normalized body hashing
        patterns = self._find_common_patterns(all_functions)

        self._log.info(
            "lilo_patterns_found",
            total_functions=len(all_functions),
            common_patterns=len(patterns),
        )

        if not patterns:
            return AbstractionExtractionResult(
                total_proposals_analyzed=len(proposal_ids),
                total_time_ms=int((time.monotonic() - start) * 1000),
            )

        # Step 3: Name and describe qualifying patterns via LLM
        new_abstractions: list[LibraryAbstraction] = []
        for _pattern_hash, functions in patterns.items():
            # Use the first function as representative
            rep = functions[0]

            # Classify the abstraction kind
            kind = self._classify_kind(rep)

            # Build the abstraction
            abstraction = await self._build_abstraction(
                representative=rep,
                all_functions=functions,
                kind=kind,
            )

            if abstraction is not None:
                new_abstractions.append(abstraction)

        # Step 4: Merge into existing library
        merged_count = 0
        pruned_count = 0
        await self._ensure_library_loaded()
        assert self._library is not None

        for new_abs in new_abstractions:
            existing = self._find_similar_in_library(new_abs)
            if existing is not None:
                # Merge: update usage count and add source proposals
                existing.usage_count += 1
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.source_proposal_ids.extend(new_abs.source_proposal_ids)
                existing.last_used_at = utc_now()
                merged_count += 1
                await self._update_abstraction(existing)
            else:
                # New abstraction
                if len(self._library) < _MAX_LIBRARY_SIZE:
                    self._library.append(new_abs)
                    await self._store_abstraction(new_abs)
                else:
                    pruned_count += 1

        total_time_ms = int((time.monotonic() - start) * 1000)

        result = AbstractionExtractionResult(
            extracted=new_abstractions,
            merged_into_existing=merged_count,
            pruned=pruned_count,
            total_proposals_analyzed=len(proposal_ids),
            total_time_ms=total_time_ms,
        )

        self._log.info(
            "lilo_extraction_complete",
            new_abstractions=len(new_abstractions),
            merged=merged_count,
            pruned=pruned_count,
            library_size=len(self._library),
            time_ms=total_time_ms,
        )

        return result

    async def get_library_prompt(self, max_abstractions: int = 15) -> str:
        """
        Generate a prompt section describing the abstraction library
        for injection into the code agent's system prompt.

        Includes the top abstractions ranked by usage_count * confidence.
        """
        await self._ensure_library_loaded()
        assert self._library is not None

        if not self._library:
            return ""

        # Rank by usage * confidence
        ranked = sorted(
            self._library,
            key=lambda a: a.usage_count * a.confidence,
            reverse=True,
        )[:max_abstractions]

        lines: list[str] = [
            "# Reusable Abstractions Library",
            f"# {len(self._library)} total abstractions, showing top {len(ranked)}",
            "",
        ]

        for i, abs_ in enumerate(ranked, 1):
            safe_name = _sanitize(abs_.name, max_len=120)
            safe_desc = _sanitize(abs_.description, max_len=300)
            safe_sig = _sanitize(abs_.signature, max_len=300)
            lines.append(f"## {i}. {safe_name} ({abs_.kind.value})")
            lines.append(f"# {safe_desc}")
            lines.append(f"# Usage: {abs_.usage_count}x, Confidence: {abs_.confidence:.0%}")
            lines.append(safe_sig)
            # Include abbreviated source (first 10 lines) inside a fenced block
            # so the LLM treats it as data, not executable instructions.
            src_lines = _sanitize(abs_.source_code, max_len=4000).splitlines()[:10]
            lines.append("```python")
            lines.append("\n".join(src_lines))
            lines.append("```")
            if len(abs_.source_code.splitlines()) > 10:
                lines.append("    # ... (truncated)")
            lines.append("")

        return "\n".join(lines)

    async def consolidate(self) -> dict[str, int]:
        """
        Periodic consolidation of the abstraction library.

        Operations:
          1. Merge similar abstractions (by normalized source code)
          2. Decay confidence of unused abstractions
          3. Prune abstractions below confidence threshold
          4. Cap total library size

        Should be run on idle cycles.
        """
        await self._ensure_library_loaded()
        assert self._library is not None

        merged = 0
        pruned = 0
        decayed = 0

        # Step 1: Merge similar abstractions
        merged_indices: set[int] = set()
        for i in range(len(self._library)):
            if i in merged_indices:
                continue
            for j in range(i + 1, len(self._library)):
                if j in merged_indices:
                    continue
                if self._are_similar(self._library[i], self._library[j]):
                    # Merge j into i
                    self._library[i].usage_count += self._library[j].usage_count
                    self._library[i].confidence = min(
                        1.0,
                        self._library[i].confidence + self._library[j].confidence * 0.5,
                    )
                    self._library[i].source_proposal_ids.extend(
                        self._library[j].source_proposal_ids
                    )
                    merged_indices.add(j)
                    merged += 1
                    await self._delete_abstraction(self._library[j].name)

        # Remove merged
        self._library = [
            a for i, a in enumerate(self._library) if i not in merged_indices
        ]

        # Step 2: Decay confidence of unused abstractions
        for abs_ in self._library:
            if abs_.last_used_at is None:
                abs_.confidence = max(0.0, abs_.confidence - _CONFIDENCE_DECAY)
                decayed += 1
            elif (utc_now() - abs_.last_used_at).days > 30:
                abs_.confidence = max(0.0, abs_.confidence - _CONFIDENCE_DECAY * 0.5)
                decayed += 1

        # Step 3: Prune below threshold
        before = len(self._library)
        prune_names = [a.name for a in self._library if a.confidence < _PRUNE_THRESHOLD]
        self._library = [a for a in self._library if a.confidence >= _PRUNE_THRESHOLD]
        pruned = before - len(self._library)

        for name in prune_names:
            await self._delete_abstraction(name)

        # Step 4: Cap library size (keep highest confidence)
        if len(self._library) > _MAX_LIBRARY_SIZE:
            self._library.sort(
                key=lambda a: a.usage_count * a.confidence, reverse=True,
            )
            overflow = self._library[_MAX_LIBRARY_SIZE:]
            self._library = self._library[:_MAX_LIBRARY_SIZE]
            for abs_ in overflow:
                await self._delete_abstraction(abs_.name)
                pruned += 1

        self._log.info(
            "lilo_consolidated",
            merged=merged,
            decayed=decayed,
            pruned=pruned,
            library_size=len(self._library),
        )

        return {"merged": merged, "decayed": decayed, "pruned": pruned}

    async def get_stats(self) -> LibraryStats:
        """Return current library statistics."""
        await self._ensure_library_loaded()
        assert self._library is not None

        by_kind: Counter[str] = Counter()
        total_usage = 0
        total_confidence = 0.0

        for abs_ in self._library:
            by_kind[abs_.kind.value] += 1
            total_usage += abs_.usage_count
            total_confidence += abs_.confidence

        return LibraryStats(
            total_abstractions=len(self._library),
            by_kind=dict(by_kind),
            total_usage_count=total_usage,
            mean_confidence=round(
                total_confidence / max(1, len(self._library)), 3,
            ),
            last_consolidated=None,
        )

    async def record_usage(self, abstraction_name: str) -> None:
        """
        Record that an abstraction was used in a proposal.
        Called by the code agent when it reuses a library abstraction.
        """
        await self._ensure_library_loaded()
        assert self._library is not None

        for abs_ in self._library:
            if abs_.name == abstraction_name:
                abs_.usage_count += 1
                abs_.confidence = min(1.0, abs_.confidence + 0.05)
                abs_.last_used_at = utc_now()
                await self._update_abstraction(abs_)
                self._log.debug(
                    "lilo_usage_recorded",
                    name=abstraction_name,
                    count=abs_.usage_count,
                )
                return

    # ─── Function Extraction ────────────────────────────────────────────────

    async def _extract_functions(
        self, rel_path: str, proposal_id: str,
    ) -> list[_ExtractedFunction]:
        """Extract function definitions from a file."""
        if self._root is None:
            return []

        full_path = self._root / rel_path
        if not full_path.is_file() or full_path.suffix != ".py":
            return []

        try:
            source = full_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, OSError):
            return []

        functions: list[_ExtractedFunction] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            start = node.lineno
            end = node.end_lineno or node.lineno
            body_lines = source.splitlines()[start - 1:end]
            body = "\n".join(body_lines)

            if len(body_lines) < _MIN_FUNCTION_LINES:
                continue

            # Skip test functions and private helpers starting with __
            if node.name.startswith("test_") or node.name.startswith("__"):
                continue

            # Build signature
            args = [a.arg for a in node.args.args if a.arg != "self"]
            sig = f"def {node.name}({', '.join(args)})"
            if isinstance(node, ast.AsyncFunctionDef):
                sig = f"async {sig}"

            functions.append(_ExtractedFunction(
                name=node.name,
                file_path=rel_path,
                proposal_id=proposal_id,
                signature=sig,
                body=body,
                body_hash=hashlib.sha256(
                    self._normalize_body(body).encode()
                ).hexdigest()[:16],
                line_count=len(body_lines),
            ))

        return functions

    def _normalize_body(self, body: str) -> str:
        """
        Normalize function body for pattern matching.
        Remove whitespace variations, comments, docstrings.
        """
        lines: list[str] = []
        in_docstring = False

        for line in body.splitlines():
            stripped = line.strip()

            # Skip comments
            if stripped.startswith("#"):
                continue

            # Simple docstring detection
            if '"""' in stripped or "'''" in stripped:
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_docstring = not in_docstring
                continue
            if in_docstring:
                continue

            # Normalize whitespace
            normalized = re.sub(r"\s+", " ", stripped)
            if normalized:
                lines.append(normalized)

        return "\n".join(lines)

    # ─── Pattern Finding ────────────────────────────────────────────────────

    def _find_common_patterns(
        self, functions: list[_ExtractedFunction],
    ) -> dict[str, list[_ExtractedFunction]]:
        """
        Find common patterns across functions from different proposals.
        Groups by normalized body hash.
        """
        # Group by body hash
        hash_groups: dict[str, list[_ExtractedFunction]] = {}
        for func in functions:
            hash_groups.setdefault(func.body_hash, []).append(func)

        # Filter: must appear in >= MIN_PATTERN_OCCURRENCES different proposals
        patterns: dict[str, list[_ExtractedFunction]] = {}
        for body_hash, funcs in hash_groups.items():
            proposal_ids = set(f.proposal_id for f in funcs)
            if len(proposal_ids) >= _MIN_PATTERN_OCCURRENCES:
                patterns[body_hash] = funcs

        return patterns

    def _classify_kind(self, func: _ExtractedFunction) -> AbstractionKind:
        """Classify the abstraction kind from function characteristics."""
        name_lower = func.name.lower()
        body_lower = func.body.lower()

        if any(k in name_lower for k in ("validate", "check", "assert", "guard")):
            return AbstractionKind.VALIDATION_GUARD
        if any(k in name_lower for k in ("error", "handle", "catch", "except")):
            return AbstractionKind.ERROR_HANDLER
        if any(k in name_lower for k in ("transform", "convert", "parse", "serialize")):
            return AbstractionKind.DATA_TRANSFORM
        if any(k in name_lower for k in ("connect", "client", "adapter", "bridge")):
            return AbstractionKind.INTEGRATION_ADAPTER
        if any(k in body_lower for k in ("try:", "except ", "raise ")):
            return AbstractionKind.ERROR_HANDLER
        if any(k in body_lower for k in ("for ", "while ", "yield ")):
            return AbstractionKind.UTILITY_FUNCTION
        return AbstractionKind.PATTERN_TEMPLATE

    # ─── Abstraction Building ────────────────────────────────────────────────

    async def _build_abstraction(
        self,
        representative: _ExtractedFunction,
        all_functions: list[_ExtractedFunction],
        kind: AbstractionKind,
    ) -> LibraryAbstraction | None:
        """Build a LibraryAbstraction from a pattern."""
        # Use LLM to generate a good name and description if available
        name = representative.name
        description = f"Reusable {kind.value} pattern from {len(all_functions)} proposals"

        if self._llm is not None:
            try:
                llm_result = await asyncio.wait_for(
                    self._llm.evaluate(
                        prompt=(
                            "Name this reusable code pattern with a clear, descriptive "
                            "snake_case name, and provide a one-line description.\n\n"
                            f"```python\n{representative.body[:500]}\n```\n\n"
                            "Reply as:\nname: <snake_case_name>\n"
                            "description: <one-line description>"
                        ),
                        max_tokens=100,
                        temperature=0.2,
                    ),
                    timeout=5.0,
                )
                # Parse name and description from LLM output
                for line in llm_result.text.strip().splitlines():
                    line = line.strip()
                    if line.lower().startswith("name:"):
                        candidate = line.split(":", 1)[1].strip().strip("`'\"")
                        # Validate it's a valid Python identifier
                        if re.match(r"^[a-z_][a-z0-9_]*$", candidate):
                            name = candidate[:80]  # cap to prevent oversized identifiers
                    elif line.lower().startswith("description:"):
                        raw_desc = line.split(":", 1)[1].strip()
                        description = _sanitize(raw_desc, max_len=300)
            except Exception as exc:
                self._log.debug("lilo_llm_naming_failed", error=str(exc))

        proposal_ids = list(set(f.proposal_id for f in all_functions))

        return LibraryAbstraction(
            name=name,
            kind=kind,
            description=description,
            signature=representative.signature,
            source_code=representative.body,
            source_proposal_ids=proposal_ids,
            usage_count=len(all_functions),
            confidence=min(1.0, 0.3 + 0.1 * len(all_functions)),
            tags=self._extract_tags(representative),
        )

    def _extract_tags(self, func: _ExtractedFunction) -> list[str]:
        """Extract tags from function name and body."""
        tags: list[str] = []
        name_parts = func.name.lower().split("_")

        # Common tag keywords
        tag_keywords = {
            "async", "validate", "parse", "convert", "check", "build",
            "create", "update", "delete", "query", "fetch", "compute",
            "cache", "retry", "log", "format", "merge", "filter",
        }

        for part in name_parts:
            if part in tag_keywords:
                tags.append(part)

        # Check body for common patterns
        body_lower = func.body.lower()
        if "asyncio" in body_lower or "await " in body_lower:
            tags.append("async")
        if "redis" in body_lower:
            tags.append("redis")
        if "neo4j" in body_lower:
            tags.append("neo4j")
        if "pydantic" in body_lower or "basemodel" in body_lower:
            tags.append("pydantic")

        return list(set(tags))

    # ─── Library Similarity ──────────────────────────────────────────────────

    def _find_similar_in_library(
        self, new_abs: LibraryAbstraction,
    ) -> LibraryAbstraction | None:
        """Find an existing abstraction similar to the new one."""
        assert self._library is not None

        new_hash = hashlib.sha256(
            self._normalize_body(new_abs.source_code).encode()
        ).hexdigest()[:16]

        for existing in self._library:
            existing_hash = hashlib.sha256(
                self._normalize_body(existing.source_code).encode()
            ).hexdigest()[:16]

            if new_hash == existing_hash:
                return existing

            # Also check by name similarity
            if existing.name == new_abs.name and existing.kind == new_abs.kind:
                return existing

        return None

    def _are_similar(self, a: LibraryAbstraction, b: LibraryAbstraction) -> bool:
        """Check if two abstractions are similar enough to merge."""
        a_hash = hashlib.sha256(
            self._normalize_body(a.source_code).encode()
        ).hexdigest()[:16]
        b_hash = hashlib.sha256(
            self._normalize_body(b.source_code).encode()
        ).hexdigest()[:16]
        return a_hash == b_hash

    # ─── Neo4j Persistence ──────────────────────────────────────────────────

    async def _ensure_library_loaded(self) -> None:
        """Load the library from Neo4j on first access."""
        if self._library_loaded:
            return

        self._library = []
        self._library_loaded = True

        if self._neo4j is None:
            return

        try:
            rows = await self._neo4j.execute_read(
                f"""
                MATCH (a:{_ABSTRACTION_LABEL})
                RETURN a
                ORDER BY a.usage_count * a.confidence DESC
                LIMIT {_MAX_LIBRARY_SIZE}
                """
            )
            for row in rows:
                data = dict(row["a"])
                try:
                    # Handle list fields stored as strings
                    for list_field in ("source_proposal_ids", "tags"):
                        if isinstance(data.get(list_field), str):
                            import orjson
                            data[list_field] = orjson.loads(data[list_field])
                    abs_ = LibraryAbstraction.model_validate(data)
                    self._library.append(abs_)
                except Exception as exc:
                    self._log.debug(
                        "lilo_load_abstraction_failed",
                        error=str(exc),
                    )
                    continue

            self._log.info(
                "lilo_library_loaded",
                size=len(self._library),
            )
        except Exception as exc:
            self._log.warning("lilo_neo4j_load_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "lilo_library_load"},
            )

    async def _store_abstraction(self, abs_: LibraryAbstraction) -> None:
        """Store a new abstraction in Neo4j."""
        if self._neo4j is None:
            return

        try:
            import orjson
            await self._neo4j.execute_write(
                f"""
                CREATE (a:{_ABSTRACTION_LABEL} {{
                    name: $name,
                    kind: $kind,
                    description: $description,
                    signature: $signature,
                    source_code: $source_code,
                    source_proposal_ids: $source_proposal_ids,
                    usage_count: $usage_count,
                    confidence: $confidence,
                    tags: $tags,
                    created_at: $created_at
                }})
                """,
                {
                    "name": abs_.name,
                    "kind": abs_.kind.value,
                    "description": abs_.description,
                    "signature": abs_.signature,
                    "source_code": abs_.source_code,
                    "source_proposal_ids": orjson.dumps(abs_.source_proposal_ids).decode(),
                    "usage_count": abs_.usage_count,
                    "confidence": abs_.confidence,
                    "tags": orjson.dumps(abs_.tags).decode(),
                    "created_at": abs_.created_at.isoformat(),
                },
            )

            # Link to source EvolutionRecords
            for proposal_id in abs_.source_proposal_ids[:5]:
                try:
                    await self._neo4j.execute_write(
                        f"""
                        MATCH (a:{_ABSTRACTION_LABEL} {{name: $name}})
                        MATCH (e:{_EVOLUTION_LABEL} {{proposal_id: $proposal_id}})
                        MERGE (a)-[:EXTRACTED_FROM]->(e)
                        """,
                        {"name": abs_.name, "proposal_id": proposal_id},
                    )
                except Exception:
                    pass  # Link creation is best-effort

        except Exception as exc:
            self._log.warning(
                "lilo_store_failed",
                name=abs_.name,
                error=str(exc),
            )
            await self._sentinel.report(
                exc, context={"operation": "lilo_store", "abstraction": abs_.name},
            )

    async def _update_abstraction(self, abs_: LibraryAbstraction) -> None:
        """Update an existing abstraction in Neo4j."""
        if self._neo4j is None:
            return

        try:
            import orjson
            await self._neo4j.execute_write(
                f"""
                MATCH (a:{_ABSTRACTION_LABEL} {{name: $name}})
                SET a.usage_count = $usage_count,
                    a.confidence = $confidence,
                    a.source_proposal_ids = $source_proposal_ids,
                    a.last_used_at = $last_used_at
                """,
                {
                    "name": abs_.name,
                    "usage_count": abs_.usage_count,
                    "confidence": abs_.confidence,
                    "source_proposal_ids": orjson.dumps(abs_.source_proposal_ids).decode(),
                    "last_used_at": (abs_.last_used_at or utc_now()).isoformat(),
                },
            )
        except Exception as exc:
            self._log.debug("lilo_update_failed", name=abs_.name, error=str(exc))

    async def _delete_abstraction(self, name: str) -> None:
        """Delete an abstraction from Neo4j."""
        if self._neo4j is None:
            return

        try:
            await self._neo4j.execute_write(
                f"""
                MATCH (a:{_ABSTRACTION_LABEL} {{name: $name}})
                DETACH DELETE a
                """,
                {"name": name},
            )
        except Exception as exc:
            self._log.debug("lilo_delete_failed", name=name, error=str(exc))


# ─── Internal Data Class ─────────────────────────────────────────────────────


class _ExtractedFunction:
    """Internal representation of an extracted function (not persisted)."""

    __slots__ = (
        "name", "file_path", "proposal_id", "signature",
        "body", "body_hash", "line_count",
    )

    def __init__(
        self,
        name: str,
        file_path: str,
        proposal_id: str,
        signature: str,
        body: str,
        body_hash: str,
        line_count: int,
    ) -> None:
        self.name = name
        self.file_path = file_path
        self.proposal_id = proposal_id
        self.signature = signature
        self.body = body
        self.body_hash = body_hash
        self.line_count = line_count
