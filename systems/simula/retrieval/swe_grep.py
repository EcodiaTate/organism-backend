"""
EcodiaOS -- Simula SWE-grep Agentic Retrieval (Stage 3B)

RL-style multi-hop code search that replaces embedding-based find_similar.
Instead of computing vector similarity (which degrades performance 15% for
similar code per AllianceCoder), this agent performs targeted multi-hop
retrieval: grep → glob → read_file → AST query across 4 serial turns
with up to 8 parallel tool calls per turn.

Key insight (AllianceCoder finding): retrieving API/context docs instead
of similar code produces better results. SWE-grep focuses on:
  1. Interface contracts (type signatures, protocols, ABC definitions)
  2. Spec documents (the .claude/ specs for the target system)
  3. Test patterns (how similar things are tested)
  4. Import graph context (what the target module depends on)

Architecture:
  - 4 serial turns of retrieval (multi-hop reasoning)
  - 8 parallel tool calls per turn (grep/glob/read/ast_query)
  - Progressive refinement: each hop narrows scope based on prior results
  - Zero LLM tokens for the search itself (tools are deterministic)
  - One final LLM call to rank/filter results by relevance

Target: 20x faster than full agentic search, higher precision than embeddings.
"""

from __future__ import annotations

import ast
import asyncio
import fnmatch
import re
import time
from typing import TYPE_CHECKING, Any
from pathlib import Path

import structlog

from systems.simula.verification.types import (
    RetrievalHop,
    RetrievalToolKind,
    RetrievedContext,
    SweGrepResult,
)

if TYPE_CHECKING:

    from clients.llm import LLMProvider
logger = structlog.get_logger().bind(system="simula.retrieval")

# Maximum results per tool call
_MAX_GREP_RESULTS = 20
_MAX_GLOB_RESULTS = 30
_MAX_READ_LINES = 200
_MAX_AST_RESULTS = 15

# Parallel tool calls per hop
_PARALLEL_CALLS_PER_HOP = 8

# Maximum hops
_MAX_HOPS = 4


class SweGrepRetriever:
    """
    SWE-grep agentic retrieval engine for Simula.

    Performs multi-hop code search using 4 deterministic tools:
      - grep: regex pattern search across files
      - glob: file pattern matching
      - read_file: read specific file sections
      - ast_query: AST-level queries (class defs, function signatures, imports)

    Each hop can run up to 8 parallel tool calls. Results from each hop
    inform the next hop's queries (progressive refinement).

    Replaces embedding-based find_similar with higher precision and
    lower latency retrieval.
    """

    def __init__(
        self,
        codebase_root: Path,
        llm: LLMProvider | None = None,
        max_hops: int = _MAX_HOPS,
    ) -> None:
        self._root = codebase_root
        self._llm = llm
        self._max_hops = max_hops
        self._log = logger

    # ─── Public API ──────────────────────────────────────────────────────────

    async def retrieve_for_proposal(
        self,
        description: str,
        category: str,
        affected_systems: list[str],
        change_spec_context: str = "",
    ) -> SweGrepResult:
        """
        Retrieve relevant context for a proposal via multi-hop search.

        Strategy:
          Hop 1: Find affected system files + spec docs
          Hop 2: Find interfaces, contracts, and type definitions
          Hop 3: Find test patterns and existing implementations
          Hop 4: Find import dependencies and cross-system references
        """
        start = time.monotonic()
        all_contexts: list[RetrievedContext] = []
        all_hops: list[RetrievalHop] = []
        total_files_searched: set[str] = set()

        # Hop 1: System files and spec docs
        hop1 = await self._hop_system_and_specs(affected_systems)
        all_hops.append(hop1)
        hop1_contexts = await self._contexts_from_hop(hop1)
        all_contexts.extend(hop1_contexts)
        total_files_searched.update(hop1.files_found)

        # Hop 2: Interfaces and type definitions
        hop2 = await self._hop_interfaces_and_types(
            affected_systems, description, category,
        )
        all_hops.append(hop2)
        hop2_contexts = await self._contexts_from_hop(hop2)
        all_contexts.extend(hop2_contexts)
        total_files_searched.update(hop2.files_found)

        # Hop 3: Test patterns and similar implementations
        hop3 = await self._hop_tests_and_patterns(
            affected_systems, description, category,
        )
        all_hops.append(hop3)
        hop3_contexts = await self._contexts_from_hop(hop3)
        all_contexts.extend(hop3_contexts)
        total_files_searched.update(hop3.files_found)

        # Hop 4: Import graph and cross-system references
        hop4 = await self._hop_import_graph(
            affected_systems,
            [c.source for c in all_contexts if c.context_type == "code"],
        )
        all_hops.append(hop4)
        hop4_contexts = await self._contexts_from_hop(hop4)
        all_contexts.extend(hop4_contexts)
        total_files_searched.update(hop4.files_found)

        # Rank and deduplicate contexts
        ranked_contexts = self._rank_and_deduplicate(
            all_contexts, description, category,
        )

        total_time_ms = int((time.monotonic() - start) * 1000)
        total_tokens = sum(h.tokens_used for h in all_hops)

        result = SweGrepResult(
            contexts=ranked_contexts[:30],  # Cap at 30 most relevant
            hops=all_hops,
            total_hops=len(all_hops),
            total_files_searched=len(total_files_searched),
            total_snippets=len(ranked_contexts),
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
        )

        self._log.info(
            "swe_grep_complete",
            contexts=len(ranked_contexts),
            files_searched=len(total_files_searched),
            hops=len(all_hops),
            time_ms=total_time_ms,
        )

        return result

    async def retrieve_for_bridge(
        self,
        description: str,
        category: str = "",
        mutation_target: str = "",
    ) -> SweGrepResult:
        """
        Lightweight retrieval for the Evo→Simula bridge (3B.5).
        Focuses on finding the mutation target and category-relevant context.
        Single hop with up to 8 parallel searches.
        """
        start = time.monotonic()
        all_contexts: list[RetrievedContext] = []
        all_hops: list[RetrievalHop] = []
        all_files: list[str] = []

        tasks: list[asyncio.Task[list[str]]] = []

        # Search for mutation target if provided
        if mutation_target:
            tasks.append(asyncio.create_task(
                self._tool_grep(pattern=mutation_target, file_glob="**/*.py"),
            ))

        # Search for category-related keywords
        if category:
            keywords = self._category_to_keywords(category)
            for kw in keywords[:3]:
                tasks.append(asyncio.create_task(
                    self._tool_grep(pattern=kw, file_glob="**/*.py"),
                ))

        # Search for terms from the description
        desc_terms = [w for w in description.split() if len(w) > 4][:2]
        for term in desc_terms:
            tasks.append(asyncio.create_task(
                self._tool_grep(pattern=re.escape(term), file_glob="**/*.py"),
            ))

        results = await asyncio.gather(
            *tasks[:_PARALLEL_CALLS_PER_HOP],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, list):
                all_files.extend(result)

        # Deduplicate files
        unique_files = list(dict.fromkeys(all_files))

        hop = RetrievalHop(
            hop_number=1,
            tool_used=RetrievalToolKind.GREP,
            query=mutation_target or category or description[:50],
            files_found=unique_files,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
        all_hops.append(hop)

        # Read the top found files
        for fpath in unique_files[:5]:
            content = await self._tool_read_file(fpath, max_lines=100)
            if content:
                all_contexts.append(RetrievedContext(
                    source=fpath,
                    content=content,
                    context_type="code",
                    relevance_score=0.8,
                ))

        total_time_ms = int((time.monotonic() - start) * 1000)

        return SweGrepResult(
            contexts=all_contexts,
            hops=all_hops,
            total_hops=1,
            total_files_searched=len(unique_files),
            total_snippets=len(all_contexts),
            total_time_ms=total_time_ms,
        )

    # ─── Hop Implementations ────────────────────────────────────────────────

    async def _hop_system_and_specs(
        self, affected_systems: list[str],
    ) -> RetrievalHop:
        """Hop 1: Find affected system files and specification documents."""
        start = time.monotonic()
        all_files: list[str] = []

        # Parallel: glob for each system + spec docs
        tasks: list[asyncio.Task[list[str]]] = []

        for system in affected_systems[:_PARALLEL_CALLS_PER_HOP // 2]:
            tasks.append(asyncio.create_task(
                self._tool_glob(f"**/systems/{system}/**/*.py"),
            ))
            # Also find spec documents
            tasks.append(asyncio.create_task(
                self._tool_glob(f"**/.claude/**/{system}*"),
            ))

        # Fill remaining slots with general patterns
        remaining = _PARALLEL_CALLS_PER_HOP - len(tasks)
        if remaining > 0:
            tasks.append(asyncio.create_task(
                self._tool_glob("**/.claude/**/*.md"),
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_files.extend(result)

        latency_ms = int((time.monotonic() - start) * 1000)

        return RetrievalHop(
            hop_number=1,
            tool_used=RetrievalToolKind.GLOB,
            query=f"systems/{','.join(affected_systems)}",
            files_found=list(set(all_files)),
            snippets_collected=0,
            latency_ms=latency_ms,
        )

    async def _hop_interfaces_and_types(
        self,
        affected_systems: list[str],
        description: str,
        category: str,
    ) -> RetrievalHop:
        """Hop 2: Find interfaces, ABCs, protocols, and type definitions."""
        start = time.monotonic()
        all_files: list[str] = []

        tasks: list[asyncio.Task[list[str]]] = []

        # Grep for class definitions in affected systems
        for system in affected_systems[:3]:
            tasks.append(asyncio.create_task(
                self._tool_grep(
                    pattern=r"class\s+\w+.*(?:ABC|Protocol|BaseModel)",
                    file_glob=f"**/systems/{system}/**/*.py",
                ),
            ))

        # Grep for type definitions
        tasks.append(asyncio.create_task(
            self._tool_grep(
                pattern=r"class\s+\w+.*EOSBaseModel",
                file_glob="**/*.py",
            ),
        ))

        # Find primitives (shared types)
        tasks.append(asyncio.create_task(
            self._tool_glob("**/primitives/*.py"),
        ))

        # AST query: find function signatures matching the category
        category_keywords = self._category_to_keywords(category)
        for keyword in category_keywords[:2]:
            tasks.append(asyncio.create_task(
                self._tool_grep(
                    pattern=keyword,
                    file_glob="**/*.py",
                ),
            ))

        results = await asyncio.gather(
            *tasks[:_PARALLEL_CALLS_PER_HOP],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, list):
                all_files.extend(result)

        latency_ms = int((time.monotonic() - start) * 1000)

        return RetrievalHop(
            hop_number=2,
            tool_used=RetrievalToolKind.GREP,
            query=f"interfaces for {category}",
            files_found=list(set(all_files)),
            latency_ms=latency_ms,
        )

    async def _hop_tests_and_patterns(
        self,
        affected_systems: list[str],
        description: str,
        category: str,
    ) -> RetrievalHop:
        """Hop 3: Find test patterns and existing implementations."""
        start = time.monotonic()
        all_files: list[str] = []

        tasks: list[asyncio.Task[list[str]]] = []

        # Find tests for affected systems
        for system in affected_systems[:3]:
            tasks.append(asyncio.create_task(
                self._tool_glob(f"**/tests/**/test_{system}*.py"),
            ))
            tasks.append(asyncio.create_task(
                self._tool_glob(f"**/tests/**/{system}/**/*.py"),
            ))

        # Find config patterns
        tasks.append(asyncio.create_task(
            self._tool_glob("**/config/*.yaml"),
        ))

        # Find registry/router patterns (common for executors/channels)
        if category in ("add_executor", "add_input_channel", "add_pattern_detector"):
            tasks.append(asyncio.create_task(
                self._tool_grep(
                    pattern=r"register|registry|route",
                    file_glob="**/*.py",
                ),
            ))

        results = await asyncio.gather(
            *tasks[:_PARALLEL_CALLS_PER_HOP],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, list):
                all_files.extend(result)

        latency_ms = int((time.monotonic() - start) * 1000)

        return RetrievalHop(
            hop_number=3,
            tool_used=RetrievalToolKind.GLOB,
            query=f"tests and patterns for {category}",
            files_found=list(set(all_files)),
            latency_ms=latency_ms,
        )

    async def _hop_import_graph(
        self,
        affected_systems: list[str],
        discovered_files: list[str],
    ) -> RetrievalHop:
        """Hop 4: Follow import chains from discovered files."""
        start = time.monotonic()
        all_files: list[str] = []

        tasks: list[asyncio.Task[list[str]]] = []

        # Find files that import from affected systems
        for system in affected_systems[:3]:
            tasks.append(asyncio.create_task(
                self._tool_grep(
                    pattern=f"from ecodiaos\\.systems\\.{system}",
                    file_glob="**/*.py",
                ),
            ))

        # Find files that import from discovered key files
        for fpath in discovered_files[:3]:
            # Convert file path to module path
            module = fpath.replace("/", ".").replace(".py", "")
            if module.startswith("src."):
                module = module[4:]
            tasks.append(asyncio.create_task(
                self._tool_grep(
                    pattern=re.escape(module),
                    file_glob="**/*.py",
                ),
            ))

        results = await asyncio.gather(
            *tasks[:_PARALLEL_CALLS_PER_HOP],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, list):
                all_files.extend(result)

        latency_ms = int((time.monotonic() - start) * 1000)

        return RetrievalHop(
            hop_number=4,
            tool_used=RetrievalToolKind.GREP,
            query=f"import graph for {','.join(affected_systems)}",
            files_found=list(set(all_files)),
            latency_ms=latency_ms,
        )

    # ─── Tools (Deterministic, Zero LLM) ────────────────────────────────────

    async def _tool_grep(
        self,
        pattern: str,
        file_glob: str = "**/*.py",
    ) -> list[str]:
        """Search for a regex pattern in files matching the glob. Returns file paths."""
        results: list[str] = []
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return results

        # Determine search root
        search_root = self._root / "src"
        if not search_root.is_dir():
            search_root = self._root

        # Convert glob to a walkable pattern
        for py_file in search_root.rglob("*.py"):
            rel_path = str(py_file.relative_to(self._root))

            # Check if file matches the glob filter
            if not fnmatch.fnmatch(rel_path, file_glob) and not fnmatch.fnmatch(
                str(py_file), file_glob
            ):
                # Try with just the filename
                if not fnmatch.fnmatch(py_file.name, file_glob.split("/")[-1] if "/" in file_glob else file_glob):
                    continue

            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                if compiled.search(content):
                    results.append(rel_path)
                    if len(results) >= _MAX_GREP_RESULTS:
                        break
            except (OSError, UnicodeDecodeError):
                continue

        return results

    async def _tool_glob(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern. Returns relative paths."""
        results: list[str] = []

        # Try both with and without src/ prefix
        for base in [self._root / "src", self._root]:
            if not base.is_dir():
                continue
            for match in base.glob(pattern):
                if match.is_file():
                    try:
                        rel = str(match.relative_to(self._root))
                        if rel not in results:
                            results.append(rel)
                    except ValueError:
                        results.append(str(match))
                if len(results) >= _MAX_GLOB_RESULTS:
                    break
            if len(results) >= _MAX_GLOB_RESULTS:
                break

        return results

    async def _tool_read_file(
        self,
        rel_path: str,
        max_lines: int = _MAX_READ_LINES,
        start_line: int = 0,
    ) -> str:
        """Read a file from the codebase. Returns content string."""
        full_path = self._root / rel_path
        if not full_path.is_file():
            # Try with src/ prefix
            full_path = self._root / "src" / rel_path
            if not full_path.is_file():
                return ""

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            selected = lines[start_line:start_line + max_lines]
            return "\n".join(selected)
        except (OSError, UnicodeDecodeError):
            return ""

    async def _tool_ast_query(
        self,
        rel_path: str,
        query_type: str = "functions",
    ) -> list[dict[str, Any]]:
        """
        AST-level queries on a Python file.
        query_type: "functions" | "classes" | "imports"
        """
        full_path = self._root / rel_path
        if not full_path.is_file():
            full_path = self._root / "src" / rel_path
            if not full_path.is_file():
                return []

        try:
            source = full_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, OSError):
            return []

        results: list[dict[str, Any]] = []

        if query_type == "functions":
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    results.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": args,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "decorators": [
                            ast.dump(d) for d in node.decorator_list[:3]
                        ],
                    })
        elif query_type == "classes":
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(f"{ast.dump(base.value)}.{base.attr}")
                    results.append({
                        "name": node.name,
                        "line": node.lineno,
                        "bases": bases,
                    })
        elif query_type == "imports":
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        results.append({
                            "module": node.module,
                            "name": alias.name,
                            "alias": alias.asname,
                        })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        results.append({
                            "module": alias.name,
                            "name": alias.name,
                            "alias": alias.asname,
                        })

        return results[:_MAX_AST_RESULTS]

    # ─── Context Extraction ─────────────────────────────────────────────────

    async def _contexts_from_hop(
        self, hop: RetrievalHop,
    ) -> list[RetrievedContext]:
        """Extract RetrievedContext objects from hop results by reading files."""
        contexts: list[RetrievedContext] = []

        # Read up to 5 most relevant files from this hop
        for fpath in hop.files_found[:5]:
            content = await self._tool_read_file(fpath, max_lines=80)
            if not content:
                continue

            # Determine context type
            if fpath.endswith(".md"):
                ctx_type = "spec"
            elif "test" in fpath.lower():
                ctx_type = "test"
            elif fpath.endswith(".yaml") or fpath.endswith(".yml"):
                ctx_type = "api_doc"
            else:
                ctx_type = "code"

            contexts.append(RetrievedContext(
                source=fpath,
                content=content[:4000],  # Cap content size
                context_type=ctx_type,
                relevance_score=0.5,  # Will be refined in ranking
            ))

        hop.snippets_collected = len(contexts)
        return contexts

    # ─── Ranking and Deduplication ───────────────────────────────────────────

    def _rank_and_deduplicate(
        self,
        contexts: list[RetrievedContext],
        description: str,
        category: str,
    ) -> list[RetrievedContext]:
        """
        Rank contexts by relevance and remove duplicates.
        Uses keyword overlap scoring (zero LLM tokens).
        """
        # Deduplicate by source path
        seen: dict[str, RetrievedContext] = {}
        for ctx in contexts:
            if ctx.source not in seen or ctx.relevance_score > seen[ctx.source].relevance_score:
                seen[ctx.source] = ctx

        unique = list(seen.values())

        # Score each context by keyword overlap with description
        desc_words = set(description.lower().split())
        cat_keywords = set(self._category_to_keywords(category))
        combined_keywords = desc_words | cat_keywords

        for ctx in unique:
            content_words = set(ctx.content.lower().split()[:200])
            overlap = len(combined_keywords & content_words)
            # Boost by context type
            type_boost = {
                "spec": 1.5,
                "api_doc": 1.3,
                "code": 1.0,
                "test": 0.8,
            }.get(ctx.context_type, 1.0)
            ctx.relevance_score = round(
                min(1.0, (overlap / max(1, len(combined_keywords))) * type_boost),
                3,
            )

        # Sort by relevance descending
        unique.sort(key=lambda c: c.relevance_score, reverse=True)
        return unique

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _category_to_keywords(self, category: str) -> list[str]:
        """Map a change category to search keywords."""
        keyword_map: dict[str, list[str]] = {
            "add_executor": ["executor", "execute", "action_type", "axon", "registry"],
            "add_input_channel": ["channel", "input", "atune", "sensor", "ingest"],
            "add_pattern_detector": ["detector", "pattern", "detect", "evo", "scan"],
            "adjust_budget": ["budget", "parameter", "threshold", "weight", "config"],
            "modify_contract": ["contract", "interface", "protocol", "abc", "abstract"],
            "add_system_capability": ["capability", "system", "service", "feature"],
            "modify_cycle_timing": ["cycle", "timing", "theta", "synapse", "rhythm"],
            "change_consolidation": ["consolidation", "sleep", "schedule", "evo"],
        }
        return keyword_map.get(category, ["system", "change"])
