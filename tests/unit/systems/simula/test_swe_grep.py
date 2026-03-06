"""
Unit tests for Simula SWE-grep Agentic Retrieval (Stage 3B).

Tests multi-hop code search, each of the 4 deterministic tools
(grep, glob, read_file, ast_query), context ranking/dedup, and
the lightweight bridge retrieval path.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from systems.simula.retrieval.swe_grep import SweGrepRetriever
from systems.simula.verification.types import (
    RetrievalHop,
    RetrievalToolKind,
    RetrievedContext,
    SweGrepResult,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ────────────────────────────────────────────────────────────────


def _write_py(root: Path, rel: str, source: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    return p


def _make_retriever(tmp_path: Path) -> SweGrepRetriever:
    return SweGrepRetriever(
        codebase_root=tmp_path,
        llm=None,
        max_hops=4,
    )


# ── Tool: grep ──────────────────────────────────────────────────────────────


class TestToolGrep:
    """Test the deterministic grep tool."""

    @pytest.mark.asyncio
    async def test_grep_finds_pattern(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/axon/executor.py",
            "class EmailExecutor:\n    action_type = 'send_email'\n",
        )
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_grep(
            pattern="EmailExecutor",
            file_glob="**/*.py",
        )

        assert len(results) >= 1
        assert any("executor.py" in r for r in results)

    @pytest.mark.asyncio
    async def test_grep_regex_pattern(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/evo/detector.py",
            "class PatternDetector:\n    pass\nclass AnomalyDetector:\n    pass\n",
        )
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_grep(
            pattern=r"class\s+\w+Detector",
            file_glob="**/*.py",
        )

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/evo/core.py",
            "x = 1\n",
        )
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_grep(
            pattern="ZZZ_NONEXISTENT_PATTERN",
            file_glob="**/*.py",
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_grep_invalid_regex_returns_empty(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_grep(
            pattern="[invalid(regex",
            file_glob="**/*.py",
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_grep_respects_max_results(self, tmp_path: Path):
        """Should not return more than _MAX_GREP_RESULTS."""
        for i in range(25):
            _write_py(
                tmp_path,
                f"systems/test/file{i}.py",
                f"SEARCH_MARKER = {i}\n",
            )
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_grep(
            pattern="SEARCH_MARKER",
            file_glob="**/*.py",
        )

        assert len(results) <= 20  # _MAX_GREP_RESULTS


# ── Tool: glob ──────────────────────────────────────────────────────────────


class TestToolGlob:
    @pytest.mark.asyncio
    async def test_glob_finds_files(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/memory/service.py",
            "# memory service\n",
        )
        _write_py(
            tmp_path,
            "systems/memory/types.py",
            "# memory types\n",
        )
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_glob("**/systems/memory/**/*.py")
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_glob_no_match(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)
        results = await retriever._tool_glob("**/nonexistent/**/*.xyz")
        assert results == []


# ── Tool: read_file ─────────────────────────────────────────────────────────


class TestToolReadFile:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/nova/core.py",
            "# Nova core\ndef evaluate(): pass\n",
        )
        retriever = _make_retriever(tmp_path)

        content = await retriever._tool_read_file(
            "systems/nova/core.py",
        )

        assert "Nova core" in content
        assert "evaluate" in content

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)
        content = await retriever._tool_read_file("no/such/file.py")
        assert content == ""

    @pytest.mark.asyncio
    async def test_read_with_line_limit(self, tmp_path: Path):
        lines = "\n".join(f"line_{i}" for i in range(100))
        _write_py(tmp_path, "systems/test/big.py", lines)
        retriever = _make_retriever(tmp_path)

        content = await retriever._tool_read_file(
            "systems/test/big.py",
            max_lines=5,
        )

        assert len(content.splitlines()) == 5

    @pytest.mark.asyncio
    async def test_read_with_start_line(self, tmp_path: Path):
        lines = "\n".join(f"line_{i}" for i in range(20))
        _write_py(tmp_path, "systems/test/offset.py", lines)
        retriever = _make_retriever(tmp_path)

        content = await retriever._tool_read_file(
            "systems/test/offset.py",
            start_line=10,
            max_lines=3,
        )

        assert "line_10" in content
        assert "line_0" not in content


# ── Tool: ast_query ─────────────────────────────────────────────────────────


class TestToolAstQuery:
    @pytest.mark.asyncio
    async def test_query_functions(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test/funcs.py",
            """\
            def sync_fn(a, b):
                return a + b

            async def async_fn(x):
                return x
            """,
        )
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_ast_query(
            "systems/test/funcs.py",
            query_type="functions",
        )

        assert len(results) == 2
        names = {r["name"] for r in results}
        assert "sync_fn" in names
        assert "async_fn" in names
        # Check async detection
        async_fn = next(r for r in results if r["name"] == "async_fn")
        assert async_fn["is_async"] is True

    @pytest.mark.asyncio
    async def test_query_classes(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test/classes.py",
            """\
            class Base:
                pass

            class Child(Base):
                pass
            """,
        )
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_ast_query(
            "systems/test/classes.py",
            query_type="classes",
        )

        assert len(results) == 2
        child = next(r for r in results if r["name"] == "Child")
        assert "Base" in child["bases"]

    @pytest.mark.asyncio
    async def test_query_imports(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test/imports.py",
            """\
            import os
            from pathlib import Path
            from typing import Any, List
            """,
        )
        retriever = _make_retriever(tmp_path)

        results = await retriever._tool_ast_query(
            "systems/test/imports.py",
            query_type="imports",
        )

        modules = [r["module"] for r in results]
        assert "os" in modules
        assert "pathlib" in modules
        assert "typing" in modules

    @pytest.mark.asyncio
    async def test_query_nonexistent_file(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)
        results = await retriever._tool_ast_query("no/file.py")
        assert results == []


# ── Ranking & Deduplication ─────────────────────────────────────────────────


class TestRankingAndDedup:
    def test_deduplicates_by_source(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)

        contexts = [
            RetrievedContext(source="a.py", content="x", relevance_score=0.5),
            RetrievedContext(source="a.py", content="y", relevance_score=0.9),
            RetrievedContext(source="b.py", content="z", relevance_score=0.7),
        ]

        ranked = retriever._rank_and_deduplicate(
            contexts, description="test description", category="add_executor",
        )

        sources = [c.source for c in ranked]
        assert sources.count("a.py") == 1
        assert "b.py" in sources

    def test_higher_relevance_wins_dedup(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)

        contexts = [
            RetrievedContext(source="a.py", content="low", relevance_score=0.1),
            RetrievedContext(source="a.py", content="high", relevance_score=0.9),
        ]

        ranked = retriever._rank_and_deduplicate(
            contexts, description="test", category="",
        )

        assert len(ranked) == 1
        assert ranked[0].content == "high"

    def test_spec_type_gets_boost(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)

        contexts = [
            RetrievedContext(
                source="spec.md",
                content="executor action_type registry axon",
                context_type="spec",
            ),
            RetrievedContext(
                source="code.py",
                content="executor action_type registry axon",
                context_type="code",
            ),
        ]

        ranked = retriever._rank_and_deduplicate(
            contexts,
            description="executor action_type",
            category="add_executor",
        )

        # Spec should rank higher due to 1.5x boost
        assert ranked[0].context_type == "spec"

    def test_keyword_overlap_scoring(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)

        contexts = [
            RetrievedContext(
                source="relevant.py",
                content="executor registry action_type send_email",
                context_type="code",
            ),
            RetrievedContext(
                source="irrelevant.py",
                content="import os\nimport sys\nprint('hello')",
                context_type="code",
            ),
        ]

        ranked = retriever._rank_and_deduplicate(
            contexts,
            description="add email executor",
            category="add_executor",
        )

        assert ranked[0].source == "relevant.py"


# ── Category Keywords ───────────────────────────────────────────────────────


class TestCategoryKeywords:
    def test_known_category_returns_keywords(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)
        kws = retriever._category_to_keywords("add_executor")
        assert "executor" in kws
        assert "axon" in kws

    def test_unknown_category_returns_fallback(self, tmp_path: Path):
        retriever = _make_retriever(tmp_path)
        kws = retriever._category_to_keywords("unknown_category")
        assert "system" in kws


# ── Bridge Retrieval ────────────────────────────────────────────────────────


class TestBridgeRetrieval:
    @pytest.mark.asyncio
    async def test_bridge_retrieval_returns_result(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/axon/registry.py",
            "EXECUTOR_REGISTRY = {'send_email': EmailExecutor}\n",
        )
        retriever = _make_retriever(tmp_path)

        result = await retriever.retrieve_for_bridge(
            description="Add email sender executor",
            category="add_executor",
            mutation_target="email_sender",
        )

        assert isinstance(result, SweGrepResult)
        assert result.total_hops == 1
        assert len(result.hops) == 1

    @pytest.mark.asyncio
    async def test_bridge_retrieval_empty_codebase(self, tmp_path: Path):
        """Should gracefully handle no results."""
        retriever = _make_retriever(tmp_path)

        result = await retriever.retrieve_for_bridge(
            description="Add nonexistent capability",
        )

        assert isinstance(result, SweGrepResult)
        assert result.total_hops == 1
        assert result.contexts == []


# ── Full Proposal Retrieval ─────────────────────────────────────────────────


class TestProposalRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_for_proposal_runs_all_hops(self, tmp_path: Path):
        """All 4 hops should execute."""
        _write_py(
            tmp_path,
            "systems/atune/service.py",
            "class AtuneService:\n    pass\n",
        )
        _write_py(
            tmp_path,
            "ecodiaos/primitives/common.py",
            "from datetime import datetime\ndef utc_now(): pass\n",
        )
        retriever = _make_retriever(tmp_path)

        result = await retriever.retrieve_for_proposal(
            description="Add a new sensor input channel for temperature",
            category="add_input_channel",
            affected_systems=["atune"],
        )

        assert isinstance(result, SweGrepResult)
        assert result.total_hops == 4
        assert len(result.hops) == 4

        # Hop numbers should be 1..4
        hop_numbers = [h.hop_number for h in result.hops]
        assert hop_numbers == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_retrieve_caps_contexts(self, tmp_path: Path):
        """Results should be capped at 30 contexts."""
        # Create many files
        for i in range(40):
            _write_py(
                tmp_path,
                f"systems/axon/exec_{i}.py",
                f"class Executor{i}:\n    action_type = 'action_{i}'\n",
            )
        retriever = _make_retriever(tmp_path)

        result = await retriever.retrieve_for_proposal(
            description="Add executor",
            category="add_executor",
            affected_systems=["axon"],
        )

        assert len(result.contexts) <= 30


# ── Context Extraction ──────────────────────────────────────────────────────


class TestContextExtraction:
    @pytest.mark.asyncio
    async def test_contexts_from_hop_reads_files(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test/file1.py",
            "# test content\nfoo = 1\n",
        )
        retriever = _make_retriever(tmp_path)

        hop = RetrievalHop(
            hop_number=1,
            tool_used=RetrievalToolKind.GLOB,
            query="test",
            files_found=["systems/test/file1.py"],
        )

        contexts = await retriever._contexts_from_hop(hop)
        assert len(contexts) == 1
        assert "test content" in contexts[0].content

    @pytest.mark.asyncio
    async def test_context_type_detection(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test/test_foo.py",
            "# test file\n",
        )
        spec = tmp_path / ".claude" / "spec.md"
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text("# Spec doc\n", encoding="utf-8")

        retriever = _make_retriever(tmp_path)

        hop = RetrievalHop(
            hop_number=1,
            tool_used=RetrievalToolKind.GLOB,
            query="test",
            files_found=[
                "systems/test/test_foo.py",
                ".claude/spec.md",
            ],
        )

        contexts = await retriever._contexts_from_hop(hop)

        types = {c.context_type for c in contexts}
        assert "test" in types
        assert "spec" in types
