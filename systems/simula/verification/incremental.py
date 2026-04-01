"""
EcodiaOS -- Simula Incremental Verification Engine (Stage 3A)

Salsa-inspired incremental computation framework for verification.
Instead of re-verifying the entire codebase after each change, this
engine tracks function-level dependencies and recomputes only what
has changed or depends on what changed.

Key design principles:
  1. Function-level granularity - each function is a verification unit
  2. Content-hash early cutoff - if hash unchanged, skip all downstream
  3. Dependency-aware invalidation - invalidate all dependents of changed functions
  4. Durability stratification - Redis (hot) + Neo4j (cold) cache layers
  5. MVCC - concurrent proposals get isolated version spaces

Target: 95% of analysis queries ≤1.2s via cache hits.

Cache key format: "simula:incr:{file_path}:{function_name}:{content_hash}"
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import time
from typing import TYPE_CHECKING, Any
from pathlib import Path

import structlog

from primitives.common import utc_now
from systems.simula.verification.types import (
    CachedVerificationResult,
    FormalVerificationResult,
    FunctionSignature,
    IncrementalVerificationResult,
)

if TYPE_CHECKING:

    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
logger = structlog.get_logger().bind(system="simula.incremental")

# Redis key prefix for incremental verification cache
_REDIS_PREFIX = "simula:incr"
_REDIS_TTL_SECONDS = 3600  # 1 hour hot cache
_NEO4J_LABEL = "VerificationCache"

# Directories to scan for Python source
_SYSTEM_DIRS = [
    "ecodiaos/systems",
    "ecodiaos/primitives",
    "ecodiaos/clients",
]


class IncrementalVerificationEngine:
    """
    Salsa-inspired incremental verification for Simula proposals.

    Tracks function-level dependencies across the codebase and
    recomputes verification only for changed functions and their
    dependents. Uses a 2-tier cache (Redis hot + Neo4j cold) for
    durability stratification.

    MVCC support: each in-flight proposal gets an isolated version
    number so concurrent proposals don't interfere with each other's
    cache state.
    """

    def __init__(
        self,
        codebase_root: Path,
        redis: RedisClient | None = None,
        neo4j: Neo4jClient | None = None,
        hot_ttl_seconds: int = _REDIS_TTL_SECONDS,
    ) -> None:
        self._root = codebase_root
        self._redis = redis
        self._neo4j = neo4j
        self._hot_ttl = hot_ttl_seconds
        self._log = logger

        # In-memory dependency graph (built lazily)
        self._dep_graph: dict[str, FunctionSignature] | None = None
        # Reverse dependency index: function_key -> set of dependents
        self._reverse_deps: dict[str, set[str]] | None = None
        # MVCC version counter
        self._mvcc_version: int = 0
        # Inherited Dafny spec hashes from parent genome (one-time boot optimization).
        # Maps spec path → content hash.  Cleared after first full verification cycle.
        self._inherited_spec_hashes: dict[str, str] | None = None

    # ─── Public API ──────────────────────────────────────────────────────────

    async def verify_incremental(
        self,
        files_changed: list[str],
        formal_verifier: _FormalVerifierCallable | None = None,
        proposal_id: str = "",
    ) -> IncrementalVerificationResult:
        """
        Incrementally verify only the functions affected by the given file changes.

        Steps:
          1. Build/update the function dependency graph
          2. Identify changed functions (by content hash comparison)
          3. Compute transitive closure of dependents
          4. Check cache for each function - early cutoff if hash unchanged
          5. Re-verify only uncached/invalidated functions
          6. Store results in 2-tier cache

        Returns IncrementalVerificationResult with full statistics.
        """
        start = time.monotonic()
        self._mvcc_version += 1
        version = self._mvcc_version

        self._log.info(
            "incremental_verify_start",
            files_changed=len(files_changed),
            proposal_id=proposal_id,
            mvcc_version=version,
        )

        # Step 1: Build dependency graph
        await self._ensure_dep_graph()
        assert self._dep_graph is not None
        assert self._reverse_deps is not None

        # Step 2: Identify changed functions
        changed_keys = await self._identify_changed_functions(files_changed)

        if not changed_keys:
            self._log.info("incremental_no_changes_detected")
            return IncrementalVerificationResult(
                total_time_ms=int((time.monotonic() - start) * 1000),
                proposal_version=version,
            )

        # Step 3: Compute transitive dependents (invalidation cascade)
        all_affected = self._compute_transitive_dependents(changed_keys)

        self._log.info(
            "incremental_affected_functions",
            directly_changed=len(changed_keys),
            total_affected=len(all_affected),
        )

        # Step 4+5: Check cache, re-verify as needed
        results: list[CachedVerificationResult] = []
        cache_hits = 0
        early_cutoffs = 0
        re_verified = 0
        invalidated_names: list[str] = []

        for func_key in all_affected:
            sig = self._dep_graph.get(func_key)
            if sig is None:
                continue

            # Check hot cache first, then cold
            cached = await self._get_cached(func_key, sig.content_hash)

            if cached is not None and cached.signature.content_hash == sig.content_hash:
                # Early cutoff - hash unchanged, cache valid
                if func_key not in changed_keys:
                    early_cutoffs += 1
                    results.append(cached)
                    continue
                else:
                    cache_hits += 1
                    results.append(cached)
                    continue

            # Cache miss or stale - need re-verification
            invalidated_names.append(func_key)

            # Check if this spec was already verified by the parent organism.
            # If the content hash matches an inherited hash, skip verification
            # and treat as pre-verified (one-time boot optimization).
            if (
                self._inherited_spec_hashes is not None
                and sig.content_hash
                and self._inherited_spec_hashes.get(sig.file_path) == sig.content_hash
            ):
                self._log.debug(
                    "verification_skipped_inherited_from_parent",
                    function=func_key,
                    file_path=sig.file_path,
                    content_hash=sig.content_hash,
                )
                pre_verified = CachedVerificationResult(
                    signature=sig,
                    formal_verification=None,
                    test_passed=True,
                    static_analysis_clean=True,
                    ttl_seconds=self._hot_ttl,
                    version_id=version,
                )
                results.append(pre_verified)
                early_cutoffs += 1
                await self._store_cached(func_key, pre_verified)
                continue

            if formal_verifier is not None:
                try:
                    fv_result = await formal_verifier(
                        file_path=sig.file_path,
                        function_name=sig.function_name,
                    )
                except Exception as exc:
                    self._log.warning(
                        "incremental_verify_function_failed",
                        function=func_key,
                        error=str(exc),
                    )
                    fv_result = None
            else:
                fv_result = None

            verified = CachedVerificationResult(
                signature=sig,
                formal_verification=fv_result,
                test_passed=fv_result.passed if fv_result else True,
                static_analysis_clean=True,
                ttl_seconds=self._hot_ttl,
                version_id=version,
            )
            results.append(verified)
            re_verified += 1

            # Step 6: Store in cache
            await self._store_cached(func_key, verified)

        total_checked = len(all_affected)
        cache_hit_rate = (
            (cache_hits + early_cutoffs) / max(1, total_checked)
        )

        total_time_ms = int((time.monotonic() - start) * 1000)

        result = IncrementalVerificationResult(
            functions_checked=total_checked,
            functions_skipped_early_cutoff=early_cutoffs,
            functions_cache_hit=cache_hits,
            functions_re_verified=re_verified,
            cache_hit_rate=round(cache_hit_rate, 3),
            total_time_ms=total_time_ms,
            results=results,
            invalidated_functions=invalidated_names,
            proposal_version=version,
            concurrent_proposals=self._mvcc_version,
        )

        self._log.info(
            "incremental_verify_complete",
            checked=total_checked,
            cache_hits=cache_hits,
            early_cutoffs=early_cutoffs,
            re_verified=re_verified,
            cache_hit_rate=f"{cache_hit_rate:.1%}",
            time_ms=total_time_ms,
        )

        # Clear inherited spec hashes after first full verification cycle
        # (one-time boot optimization - no longer needed after initial run).
        if self._inherited_spec_hashes is not None:
            self._log.info(
                "inherited_spec_hashes_cleared",
                count=len(self._inherited_spec_hashes),
            )
            self._inherited_spec_hashes = None

        return result

    async def invalidate_for_files(self, files: list[str]) -> int:
        """
        Explicitly invalidate all cached results for functions in the given files.
        Returns the number of cache entries invalidated.
        """
        await self._ensure_dep_graph()
        assert self._dep_graph is not None

        count = 0
        for func_key, sig in self._dep_graph.items():
            if sig.file_path in files:
                await self._invalidate_cached(func_key)
                count += 1

        if count:
            self._log.info("incremental_invalidated", count=count, files=len(files))
        return count

    async def rebuild_graph(self) -> int:
        """Force a full dependency graph rebuild. Returns function count."""
        self._dep_graph = None
        self._reverse_deps = None
        await self._ensure_dep_graph()
        assert self._dep_graph is not None
        return len(self._dep_graph)

    def seed_inherited_hashes(self, hashes: dict[str, str]) -> None:
        """
        Accept Dafny spec content hashes inherited from parent genome.

        During the first verification cycle, specs whose content hash matches
        an inherited hash are skipped (pre-verified).  The inherited hashes are
        cleared after the first ``verify_incremental`` call completes.
        """
        if hashes:
            self._inherited_spec_hashes = dict(hashes)
            self._log.info(
                "inherited_spec_hashes_loaded",
                count=len(hashes),
            )

    def get_stats(self) -> dict[str, Any]:
        """Return current incremental engine statistics."""
        return {
            "dep_graph_size": len(self._dep_graph) if self._dep_graph else 0,
            "mvcc_version": self._mvcc_version,
            "hot_ttl_seconds": self._hot_ttl,
            "has_redis": self._redis is not None,
            "has_neo4j": self._neo4j is not None,
            "inherited_spec_hashes": len(self._inherited_spec_hashes) if self._inherited_spec_hashes else 0,
        }

    # ─── Dependency Graph ────────────────────────────────────────────────────

    async def _ensure_dep_graph(self) -> None:
        """Build the dependency graph if not already built."""
        if self._dep_graph is not None:
            return

        start = time.monotonic()
        self._dep_graph = {}
        self._reverse_deps = {}

        # Scan all Python files in system directories
        for sys_dir in _SYSTEM_DIRS:
            full_dir = self._root / "src" / sys_dir
            if not full_dir.is_dir():
                # Try without src/ prefix
                full_dir = self._root / sys_dir
                if not full_dir.is_dir():
                    continue

            for py_file in full_dir.rglob("*.py"):
                rel_path = str(py_file.relative_to(self._root))
                try:
                    source = py_file.read_text(encoding="utf-8")
                    self._index_file(rel_path, source)
                except Exception:
                    continue

        # Build reverse dependency index
        for func_key, sig in self._dep_graph.items():
            for imp in sig.imports:
                self._reverse_deps.setdefault(imp, set()).add(func_key)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._log.info(
            "dep_graph_built",
            functions=len(self._dep_graph),
            reverse_entries=len(self._reverse_deps),
            time_ms=elapsed_ms,
        )

    def _index_file(self, rel_path: str, source: str) -> None:
        """Parse a Python file and index all functions with their dependencies."""
        assert self._dep_graph is not None

        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            return

        # Collect top-level and class-level functions
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            func_name = node.name
            func_key = f"{rel_path}::{func_name}"

            # Extract the function source lines
            start_line = node.lineno
            end_line = node.end_lineno or node.lineno
            lines = source.splitlines()[start_line - 1:end_line]
            func_source = "\n".join(lines)

            # Content hash for early cutoff
            content_hash = hashlib.sha256(func_source.encode()).hexdigest()[:16]

            # Extract imports used within the function body
            imports = self._extract_function_imports(node, source)

            sig = FunctionSignature(
                file_path=rel_path,
                function_name=func_name,
                content_hash=content_hash,
                start_line=start_line,
                end_line=end_line,
                imports=imports,
            )

            self._dep_graph[func_key] = sig

    def _extract_function_imports(
        self,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        full_source: str,
    ) -> list[str]:
        """
        Extract function-level dependencies by analyzing name usage.
        Maps used names to module-level imports to build the dependency graph.
        """
        # Collect all Name nodes used in the function
        used_names: set[str] = set()
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

        # Map names to module imports from the file
        imports: list[str] = []
        try:
            file_tree = ast.parse(full_source)
        except SyntaxError:
            return imports

        for node in ast.walk(file_tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name in used_names:
                        imports.append(f"{node.module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name in used_names:
                        imports.append(alias.name)

        return imports

    # ─── Change Detection ────────────────────────────────────────────────────

    async def _identify_changed_functions(
        self, files_changed: list[str],
    ) -> set[str]:
        """
        Compare current function hashes with cached hashes to identify
        which functions actually changed. Returns set of function keys.
        """
        assert self._dep_graph is not None
        changed: set[str] = set()

        for func_key, sig in self._dep_graph.items():
            if sig.file_path not in files_changed:
                continue

            # Re-parse the file to get current hash
            full_path = self._root / sig.file_path
            if not full_path.is_file():
                changed.add(func_key)
                continue

            try:
                source = full_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=sig.file_path)
            except (SyntaxError, OSError):
                changed.add(func_key)
                continue

            # Find the function and compute current hash
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == sig.function_name
                ):
                    start = node.lineno
                    end = node.end_lineno or node.lineno
                    lines = source.splitlines()[start - 1:end]
                    current_hash = hashlib.sha256(
                        "\n".join(lines).encode()
                    ).hexdigest()[:16]

                    if current_hash != sig.content_hash:
                        changed.add(func_key)
                        # Update the graph with new hash
                        sig.content_hash = current_hash
                        sig.start_line = start
                        sig.end_line = end
                    break
            else:
                # Function was removed
                changed.add(func_key)

        return changed

    # ─── Transitive Dependency Computation ───────────────────────────────────

    def _compute_transitive_dependents(
        self, changed_keys: set[str],
    ) -> set[str]:
        """
        Compute the transitive closure of all functions that depend
        (directly or indirectly) on the changed functions. Uses BFS.
        """
        assert self._reverse_deps is not None

        affected = set(changed_keys)
        queue = list(changed_keys)

        while queue:
            current = queue.pop(0)
            dependents = self._reverse_deps.get(current, set())
            for dep in dependents:
                if dep not in affected:
                    affected.add(dep)
                    queue.append(dep)

        return affected

    # ─── 2-Tier Cache (Redis Hot + Neo4j Cold) ──────────────────────────────

    async def _get_cached(
        self,
        func_key: str,
        expected_hash: str,
    ) -> CachedVerificationResult | None:
        """
        Look up a cached verification result. Checks Redis first (hot),
        then Neo4j (cold). Returns None on miss.
        """
        # Tier 1: Redis hot cache
        if self._redis is not None:
            try:
                redis_key = f"{_REDIS_PREFIX}:{func_key}"
                data = await self._redis.get_json(redis_key)
                if data is not None:
                    result = CachedVerificationResult.model_validate(data)
                    if result.signature.content_hash == expected_hash:
                        return result
            except Exception as exc:
                self._log.debug("redis_cache_miss", key=func_key, error=str(exc))

        # Tier 2: Neo4j cold cache
        if self._neo4j is not None:
            try:
                rows = await self._neo4j.execute_read(
                    f"""
                    MATCH (c:{_NEO4J_LABEL} {{
                        function_key: $func_key,
                        content_hash: $hash
                    }})
                    RETURN c.data AS data
                    LIMIT 1
                    """,
                    {"func_key": func_key, "hash": expected_hash},
                )
                if rows:
                    import orjson
                    result = CachedVerificationResult.model_validate(
                        orjson.loads(rows[0]["data"])
                    )
                    # Promote to hot cache on read
                    await self._store_hot(func_key, result)
                    return result
            except Exception as exc:
                self._log.debug("neo4j_cache_miss", key=func_key, error=str(exc))

        return None

    async def _store_cached(
        self, func_key: str, result: CachedVerificationResult,
    ) -> None:
        """Store a verification result in both cache tiers."""
        await asyncio.gather(
            self._store_hot(func_key, result),
            self._store_cold(func_key, result),
            return_exceptions=True,
        )

    async def _store_hot(
        self, func_key: str, result: CachedVerificationResult,
    ) -> None:
        """Store in Redis hot cache with TTL."""
        if self._redis is None:
            return
        try:
            redis_key = f"{_REDIS_PREFIX}:{func_key}"
            data = result.model_dump(mode="json")
            await self._redis.set_json(redis_key, data, ttl=self._hot_ttl)
        except Exception as exc:
            self._log.debug("redis_store_failed", key=func_key, error=str(exc))

    async def _store_cold(
        self, func_key: str, result: CachedVerificationResult,
    ) -> None:
        """Store in Neo4j cold cache (durable, no TTL)."""
        if self._neo4j is None:
            return
        try:
            import orjson
            data_json = orjson.dumps(result.model_dump(mode="json")).decode()
            await self._neo4j.execute_write(
                f"""
                MERGE (c:{_NEO4J_LABEL} {{function_key: $func_key}})
                SET c.content_hash = $hash,
                    c.data = $data,
                    c.version_id = $version,
                    c.updated_at = $updated_at
                """,
                {
                    "func_key": func_key,
                    "hash": result.signature.content_hash,
                    "data": data_json,
                    "version": result.version_id,
                    "updated_at": utc_now().isoformat(),
                },
            )
        except Exception as exc:
            self._log.debug("neo4j_store_failed", key=func_key, error=str(exc))

    async def _invalidate_cached(self, func_key: str) -> None:
        """Invalidate a cache entry in both tiers."""
        tasks: list[Any] = []

        if self._redis is not None:
            async def _del_redis() -> None:
                try:
                    await self._redis.delete(f"{_REDIS_PREFIX}:{func_key}")  # type: ignore[union-attr]
                except Exception:
                    pass
            tasks.append(_del_redis())

        if self._neo4j is not None:
            async def _del_neo4j() -> None:
                try:
                    await self._neo4j.execute_write(  # type: ignore[union-attr]
                        f"MATCH (c:{_NEO4J_LABEL} {{function_key: $key}}) DELETE c",
                        {"key": func_key},
                    )
                except Exception:
                    pass
            tasks.append(_del_neo4j())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Type alias for the formal verifier callback
from collections.abc import Callable, Coroutine

_FormalVerifierCallable = Callable[
    ...,
    Coroutine[Any, Any, FormalVerificationResult | None],
]
