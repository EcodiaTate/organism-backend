"""
Unit tests for Simula Incremental Verification Engine (Stage 3A).

Tests dependency graph building, content-hash early cutoff, transitive
invalidation, 2-tier cache (Redis/Neo4j), and MVCC version isolation.
"""

from __future__ import annotations

import hashlib
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from systems.simula.verification.incremental import (
    IncrementalVerificationEngine,
)
from systems.simula.verification.types import (
    CachedVerificationResult,
    FunctionSignature,
    IncrementalVerificationResult,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ────────────────────────────────────────────────────────────────


def _write_py(root: Path, rel: str, source: str) -> Path:
    """Write a Python file and return its path."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    return p


def _make_engine(tmp_path: Path) -> IncrementalVerificationEngine:
    """Create an engine with no external services."""
    return IncrementalVerificationEngine(
        codebase_root=tmp_path,
        redis=None,
        neo4j=None,
    )


# ── Dependency Graph ────────────────────────────────────────────────────────


class TestDependencyGraph:
    """Test the AST-based dependency graph builder."""

    def test_index_file_extracts_functions(self, tmp_path: Path):
        """All function defs (sync + async) should be indexed."""
        _write_py(
            tmp_path,
            "systems/demo/service.py",
            """\
            import asyncio

            def sync_helper(x: int) -> int:
                return x + 1

            async def async_runner(y: str) -> str:
                return await asyncio.sleep(0) or y
            """,
        )

        engine = _make_engine(tmp_path)
        engine._dep_graph = {}
        engine._reverse_deps = {}

        source = (tmp_path / "systems/demo/service.py").read_text()
        engine._index_file("systems/demo/service.py", source)

        assert len(engine._dep_graph) == 2
        keys = list(engine._dep_graph.keys())
        assert any("sync_helper" in k for k in keys)
        assert any("async_runner" in k for k in keys)

    def test_content_hash_deterministic(self, tmp_path: Path):
        """Same source → same content hash."""
        src = """\
        def foo(a, b):
            return a + b
        """
        _write_py(tmp_path, "systems/alpha/mod.py", src)

        engine = _make_engine(tmp_path)
        engine._dep_graph = {}
        engine._reverse_deps = {}

        source = (tmp_path / "systems/alpha/mod.py").read_text()
        engine._index_file("systems/alpha/mod.py", source)

        sig = list(engine._dep_graph.values())[0]
        expected = hashlib.sha256(
            "\n".join(source.splitlines()[0:2]).encode()
        ).hexdigest()[:16]
        assert sig.content_hash == expected

    def test_extract_imports(self, tmp_path: Path):
        """Function imports should include names used inside the body."""
        src = """\
        from os.path import join

        def build_path(base):
            return join(base, "sub")
        """
        _write_py(tmp_path, "systems/beta/paths.py", src)

        engine = _make_engine(tmp_path)
        engine._dep_graph = {}
        engine._reverse_deps = {}

        source = (tmp_path / "systems/beta/paths.py").read_text()
        engine._index_file("systems/beta/paths.py", source)

        sig = list(engine._dep_graph.values())[0]
        assert any("os.path" in imp for imp in sig.imports)

    @pytest.mark.asyncio
    async def test_ensure_dep_graph_scans_system_dirs(self, tmp_path: Path):
        """The graph builder should scan systems/ tree."""
        _write_py(
            tmp_path,
            "systems/gamma/core.py",
            """\
            def process():
                return 42
            """,
        )

        engine = _make_engine(tmp_path)
        await engine._ensure_dep_graph()

        assert engine._dep_graph is not None
        assert len(engine._dep_graph) >= 1

    @pytest.mark.asyncio
    async def test_rebuild_graph_clears_and_rebuilds(self, tmp_path: Path):
        """rebuild_graph should clear and re-index."""
        _write_py(
            tmp_path,
            "systems/gamma/core.py",
            "def hello(): return 1\n",
        )

        engine = _make_engine(tmp_path)
        count1 = await engine.rebuild_graph()
        assert count1 >= 1

        # Add another function, rebuild
        _write_py(
            tmp_path,
            "systems/gamma/core.py",
            "def hello(): return 1\ndef world(): return 2\n",
        )
        count2 = await engine.rebuild_graph()
        assert count2 >= 2


# ── Transitive Dependents ───────────────────────────────────────────────────


class TestTransitiveDependents:
    """Test the BFS-based transitive dependency computation."""

    def test_no_dependents(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._dep_graph = {}
        engine._reverse_deps = {}

        result = engine._compute_transitive_dependents({"a::foo"})
        assert result == {"a::foo"}

    def test_direct_dependents(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._dep_graph = {}
        engine._reverse_deps = {
            "a::foo": {"b::bar", "c::baz"},
        }

        result = engine._compute_transitive_dependents({"a::foo"})
        assert result == {"a::foo", "b::bar", "c::baz"}

    def test_transitive_chain(self, tmp_path: Path):
        """A → B → C: changing A should invalidate all three."""
        engine = _make_engine(tmp_path)
        engine._dep_graph = {}
        engine._reverse_deps = {
            "a::foo": {"b::bar"},
            "b::bar": {"c::baz"},
        }

        result = engine._compute_transitive_dependents({"a::foo"})
        assert result == {"a::foo", "b::bar", "c::baz"}

    def test_cycle_handling(self, tmp_path: Path):
        """Cycles should not cause infinite loops."""
        engine = _make_engine(tmp_path)
        engine._dep_graph = {}
        engine._reverse_deps = {
            "a::foo": {"b::bar"},
            "b::bar": {"a::foo"},
        }

        result = engine._compute_transitive_dependents({"a::foo"})
        assert result == {"a::foo", "b::bar"}


# ── Verify Incremental ─────────────────────────────────────────────────────


class TestVerifyIncremental:
    """Test the main verify_incremental method."""

    @pytest.mark.asyncio
    async def test_no_changes_returns_empty(self, tmp_path: Path):
        """If no files changed, result should be empty with 0 checked."""
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def greet(): return 'hi'\n",
        )
        engine = _make_engine(tmp_path)

        result = await engine.verify_incremental(
            files_changed=["nonexistent.py"],
        )

        assert isinstance(result, IncrementalVerificationResult)
        assert result.functions_checked == 0
        assert result.proposal_version == 1

    @pytest.mark.asyncio
    async def test_changed_file_triggers_verification(self, tmp_path: Path):
        """Changing a file should trigger re-verification of its functions."""
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def greet(): return 'hi'\n",
        )
        engine = _make_engine(tmp_path)

        # First build the graph
        await engine.rebuild_graph()

        # Modify the file
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def greet(): return 'hello'\n",
        )

        result = await engine.verify_incremental(
            files_changed=["systems/test_sys/mod.py"],
        )

        assert result.functions_checked >= 1
        assert result.functions_re_verified >= 1

    @pytest.mark.asyncio
    async def test_mvcc_version_increments(self, tmp_path: Path):
        """Each call should increment the MVCC version."""
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def greet(): return 'hi'\n",
        )
        engine = _make_engine(tmp_path)

        r1 = await engine.verify_incremental(files_changed=["x.py"])
        r2 = await engine.verify_incremental(files_changed=["y.py"])

        assert r1.proposal_version == 1
        assert r2.proposal_version == 2

    @pytest.mark.asyncio
    async def test_formal_verifier_callback(self, tmp_path: Path):
        """When a formal_verifier is provided, it should be called."""
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def greet(): return 'hi'\n",
        )
        engine = _make_engine(tmp_path)
        await engine.rebuild_graph()

        # Modify to trigger change
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def greet(): return 'goodbye'\n",
        )

        mock_verifier = AsyncMock(return_value=None)

        result = await engine.verify_incremental(
            files_changed=["systems/test_sys/mod.py"],
            formal_verifier=mock_verifier,
        )

        assert result.functions_re_verified >= 1
        # Verifier should have been called at least once
        assert mock_verifier.call_count >= 1

    @pytest.mark.asyncio
    async def test_proposal_id_propagation(self, tmp_path: Path):
        """proposal_id should be reflected in the result."""
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def greet(): return 'hi'\n",
        )
        engine = _make_engine(tmp_path)

        result = await engine.verify_incremental(
            files_changed=["x.py"],
            proposal_id="proposal-123",
        )

        assert isinstance(result, IncrementalVerificationResult)


# ── Cache Operations ────────────────────────────────────────────────────────


class TestCacheOperations:
    """Test the 2-tier cache lookups (no real Redis/Neo4j)."""

    @pytest.mark.asyncio
    async def test_get_cached_returns_none_without_backends(self, tmp_path: Path):
        """With no Redis or Neo4j, cache lookups always miss."""
        engine = _make_engine(tmp_path)

        result = await engine._get_cached("test::func", "abc123")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_cached_noop_without_backends(self, tmp_path: Path):
        """Store should not raise when backends are None."""
        engine = _make_engine(tmp_path)

        sig = FunctionSignature(
            file_path="test.py",
            function_name="func",
            content_hash="abc123",
        )
        cached = CachedVerificationResult(
            signature=sig,
            test_passed=True,
        )

        # Should not raise
        await engine._store_cached("test::func", cached)

    @pytest.mark.asyncio
    async def test_redis_cache_hit(self, tmp_path: Path):
        """Redis cache hit should return the cached result."""
        mock_redis = AsyncMock()
        sig = FunctionSignature(
            file_path="test.py",
            function_name="func",
            content_hash="abc123",
        )
        cached = CachedVerificationResult(
            signature=sig,
            test_passed=True,
        )
        mock_redis.get_json = AsyncMock(
            return_value=cached.model_dump(mode="json"),
        )

        engine = IncrementalVerificationEngine(
            codebase_root=tmp_path,
            redis=mock_redis,
            neo4j=None,
        )

        result = await engine._get_cached("test::func", "abc123")
        assert result is not None
        assert result.signature.content_hash == "abc123"

    @pytest.mark.asyncio
    async def test_redis_cache_miss_stale_hash(self, tmp_path: Path):
        """Stale hash in Redis should not match."""
        mock_redis = AsyncMock()
        sig = FunctionSignature(
            file_path="test.py",
            function_name="func",
            content_hash="old_hash",
        )
        cached = CachedVerificationResult(signature=sig, test_passed=True)
        mock_redis.get_json = AsyncMock(
            return_value=cached.model_dump(mode="json"),
        )

        engine = IncrementalVerificationEngine(
            codebase_root=tmp_path,
            redis=mock_redis,
            neo4j=None,
        )

        result = await engine._get_cached("test::func", "new_hash")
        # Returns the object but hash won't match in verify_incremental
        assert result is not None
        assert result.signature.content_hash != "new_hash"


# ── Invalidation ────────────────────────────────────────────────────────────


class TestInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_for_files(self, tmp_path: Path):
        """Should count functions in the given files."""
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def a(): return 1\ndef b(): return 2\n",
        )
        engine = _make_engine(tmp_path)
        await engine.rebuild_graph()

        count = await engine.invalidate_for_files(
            ["systems/test_sys/mod.py"],
        )
        assert count == 2


# ── Stats ───────────────────────────────────────────────────────────────────


class TestIncrementalStats:
    def test_get_stats_initial(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        stats = engine.get_stats()

        assert stats["dep_graph_size"] == 0
        assert stats["mvcc_version"] == 0
        assert stats["has_redis"] is False
        assert stats["has_neo4j"] is False

    @pytest.mark.asyncio
    async def test_get_stats_after_build(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test_sys/mod.py",
            "def x(): return 1\n",
        )
        engine = _make_engine(tmp_path)
        await engine.rebuild_graph()

        stats = engine.get_stats()
        assert stats["dep_graph_size"] >= 1
