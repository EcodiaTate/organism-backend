"""
Unit tests for Simula LILO Library Learning (Stage 3C).

Tests abstraction extraction from proposals, pattern finding via
normalized body hashing, LLM-assisted naming, consolidation
(merge/decay/prune), library prompt generation, and usage tracking.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from systems.simula.learning.lilo import (
    LiloLibraryEngine,
    _ExtractedFunction,
)
from systems.simula.verification.types import (
    AbstractionExtractionResult,
    AbstractionKind,
    LibraryAbstraction,
    LibraryStats,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ────────────────────────────────────────────────────────────────


def _write_py(root: Path, rel: str, source: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    return p


def _make_engine(tmp_path: Path) -> LiloLibraryEngine:
    return LiloLibraryEngine(
        neo4j=None,
        llm=None,
        codebase_root=tmp_path,
    )


def _make_abstraction(
    name: str = "test_fn",
    kind: AbstractionKind = AbstractionKind.UTILITY_FUNCTION,
    source: str = "def test_fn():\n    return 1\n",
    confidence: float = 0.5,
    usage: int = 1,
) -> LibraryAbstraction:
    return LibraryAbstraction(
        name=name,
        kind=kind,
        description=f"Test abstraction: {name}",
        signature=f"def {name}()",
        source_code=source,
        source_proposal_ids=["p1"],
        usage_count=usage,
        confidence=confidence,
        tags=["test"],
    )


# ── Function Extraction ────────────────────────────────────────────────────


class TestFunctionExtraction:
    @pytest.mark.asyncio
    async def test_extracts_functions_from_file(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test/mod.py",
            """\
            def validate_input(data: dict) -> bool:
                if not data:
                    return False
                return "name" in data

            async def fetch_data(url: str) -> str:
                import httpx
                resp = await httpx.get(url)
                return resp.text
            """,
        )
        engine = _make_engine(tmp_path)

        functions = await engine._extract_functions(
            "systems/test/mod.py", "proposal-1",
        )

        assert len(functions) == 2
        names = {f.name for f in functions}
        assert "validate_input" in names
        assert "fetch_data" in names

    @pytest.mark.asyncio
    async def test_skips_test_functions(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test/mod.py",
            """\
            def test_something():
                assert True
                # body

            def real_function():
                x = 1
                return x
            """,
        )
        engine = _make_engine(tmp_path)

        functions = await engine._extract_functions(
            "systems/test/mod.py", "p1",
        )

        names = {f.name for f in functions}
        assert "test_something" not in names
        assert "real_function" in names

    @pytest.mark.asyncio
    async def test_skips_dunder_methods(self, tmp_path: Path):
        _write_py(
            tmp_path,
            "systems/test/mod.py",
            """\
            class Foo:
                def __init__(self):
                    self.x = 1
                    pass

                def process(self):
                    return self.x
                    # extra
            """,
        )
        engine = _make_engine(tmp_path)

        functions = await engine._extract_functions(
            "systems/test/mod.py", "p1",
        )

        names = {f.name for f in functions}
        assert "__init__" not in names
        assert "process" in names

    @pytest.mark.asyncio
    async def test_skips_short_functions(self, tmp_path: Path):
        """Functions shorter than _MIN_FUNCTION_LINES should be skipped."""
        _write_py(
            tmp_path,
            "systems/test/mod.py",
            "def tiny(): pass\n\ndef bigger():\n    x = 1\n    return x\n",
        )
        engine = _make_engine(tmp_path)

        functions = await engine._extract_functions(
            "systems/test/mod.py", "p1",
        )

        names = {f.name for f in functions}
        assert "tiny" not in names

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_empty(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        functions = await engine._extract_functions("no/such/file.py", "p1")
        assert functions == []

    @pytest.mark.asyncio
    async def test_non_python_file_returns_empty(self, tmp_path: Path):
        txt = tmp_path / "readme.txt"
        txt.write_text("Hello", encoding="utf-8")
        engine = _make_engine(tmp_path)
        functions = await engine._extract_functions("readme.txt", "p1")
        assert functions == []


# ── Body Normalization ──────────────────────────────────────────────────────


class TestBodyNormalization:
    def test_strips_comments(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        body = "def f():\n    # comment\n    return 1"
        normalized = engine._normalize_body(body)
        assert "# comment" not in normalized
        assert "return 1" in normalized

    def test_strips_docstrings(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        body = 'def f():\n    """docstring"""\n    return 1'
        normalized = engine._normalize_body(body)
        assert "docstring" not in normalized

    def test_normalizes_whitespace(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        body1 = "def f():\n    return   1"
        body2 = "def f():\n    return 1"
        assert engine._normalize_body(body1) == engine._normalize_body(body2)


# ── Pattern Finding ─────────────────────────────────────────────────────────


class TestPatternFinding:
    def test_finds_duplicate_patterns(self, tmp_path: Path):
        engine = _make_engine(tmp_path)

        func1 = _ExtractedFunction(
            name="validate_a",
            file_path="a.py",
            proposal_id="p1",
            signature="def validate_a(data)",
            body="def validate_a(data):\n    if not data:\n        return False\n    return True",
            body_hash="abc123",
            line_count=4,
        )
        func2 = _ExtractedFunction(
            name="validate_b",
            file_path="b.py",
            proposal_id="p2",
            signature="def validate_b(data)",
            body="def validate_b(data):\n    if not data:\n        return False\n    return True",
            body_hash="abc123",  # Same hash = same pattern
            line_count=4,
        )

        patterns = engine._find_common_patterns([func1, func2])
        assert len(patterns) == 1
        assert len(list(patterns.values())[0]) == 2

    def test_ignores_single_occurrence(self, tmp_path: Path):
        engine = _make_engine(tmp_path)

        func1 = _ExtractedFunction(
            name="unique_fn",
            file_path="a.py",
            proposal_id="p1",
            signature="def unique_fn()",
            body="def unique_fn():\n    return 42\n    # extra",
            body_hash="unique_hash",
            line_count=3,
        )

        patterns = engine._find_common_patterns([func1])
        assert len(patterns) == 0

    def test_requires_different_proposals(self, tmp_path: Path):
        """Same hash from the same proposal doesn't count."""
        engine = _make_engine(tmp_path)

        func1 = _ExtractedFunction(
            name="f1", file_path="a.py", proposal_id="p1",
            signature="def f1()", body="body\n#\n#", body_hash="hash1", line_count=3,
        )
        func2 = _ExtractedFunction(
            name="f2", file_path="b.py", proposal_id="p1",  # Same proposal
            signature="def f2()", body="body\n#\n#", body_hash="hash1", line_count=3,
        )

        patterns = engine._find_common_patterns([func1, func2])
        assert len(patterns) == 0  # Both from same proposal


# ── Kind Classification ─────────────────────────────────────────────────────


class TestKindClassification:
    def test_validation_guard(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        func = _ExtractedFunction(
            name="validate_input", file_path="a.py", proposal_id="p1",
            signature="def validate_input(data)",
            body="def validate_input(data):\n    return True\n    # end",
            body_hash="h", line_count=3,
        )
        assert engine._classify_kind(func) == AbstractionKind.VALIDATION_GUARD

    def test_error_handler(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        func = _ExtractedFunction(
            name="handle_error", file_path="a.py", proposal_id="p1",
            signature="def handle_error(exc)",
            body="def handle_error(exc):\n    log(exc)\n    return",
            body_hash="h", line_count=3,
        )
        assert engine._classify_kind(func) == AbstractionKind.ERROR_HANDLER

    def test_data_transform(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        func = _ExtractedFunction(
            name="transform_data", file_path="a.py", proposal_id="p1",
            signature="def transform_data(raw)",
            body="def transform_data(raw):\n    return dict(raw)\n    # end",
            body_hash="h", line_count=3,
        )
        assert engine._classify_kind(func) == AbstractionKind.DATA_TRANSFORM

    def test_integration_adapter(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        func = _ExtractedFunction(
            name="redis_client_adapter", file_path="a.py", proposal_id="p1",
            signature="def redis_client_adapter(cfg)",
            body="def redis_client_adapter(cfg):\n    return Client(cfg)\n    # extra",
            body_hash="h", line_count=3,
        )
        assert engine._classify_kind(func) == AbstractionKind.INTEGRATION_ADAPTER

    def test_fallback_pattern_template(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        func = _ExtractedFunction(
            name="do_stuff", file_path="a.py", proposal_id="p1",
            signature="def do_stuff()",
            body="def do_stuff():\n    x = 1\n    return x",
            body_hash="h", line_count=3,
        )
        assert engine._classify_kind(func) == AbstractionKind.PATTERN_TEMPLATE


# ── Tag Extraction ──────────────────────────────────────────────────────────


class TestTagExtraction:
    def test_extracts_name_tags(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        func = _ExtractedFunction(
            name="validate_cache_query", file_path="a.py", proposal_id="p1",
            signature="def validate_cache_query()",
            body="def validate_cache_query():\n    pass\n    #end",
            body_hash="h", line_count=3,
        )
        tags = engine._extract_tags(func)
        assert "validate" in tags
        assert "cache" in tags
        assert "query" in tags

    def test_extracts_body_tags(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        func = _ExtractedFunction(
            name="store", file_path="a.py", proposal_id="p1",
            signature="def store()",
            body="def store():\n    await redis.set(k, v)\n    neo4j.run(q)",
            body_hash="h", line_count=3,
        )
        tags = engine._extract_tags(func)
        assert "redis" in tags
        assert "neo4j" in tags


# ── Library Prompt Generation ───────────────────────────────────────────────


class TestLibraryPrompt:
    @pytest.mark.asyncio
    async def test_empty_library_returns_empty_string(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        prompt = await engine.get_library_prompt()
        assert prompt == ""

    @pytest.mark.asyncio
    async def test_library_prompt_includes_abstractions(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._library = [
            _make_abstraction("safe_merge", usage=5, confidence=0.9),
            _make_abstraction("retry_with_backoff", usage=3, confidence=0.8),
        ]
        engine._library_loaded = True

        prompt = await engine.get_library_prompt()

        assert "Reusable Abstractions Library" in prompt
        assert "safe_merge" in prompt
        assert "retry_with_backoff" in prompt
        assert "Usage: 5x" in prompt

    @pytest.mark.asyncio
    async def test_library_prompt_ranks_by_usage_x_confidence(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._library = [
            _make_abstraction("low_score", usage=1, confidence=0.1),
            _make_abstraction("high_score", usage=10, confidence=1.0),
        ]
        engine._library_loaded = True

        prompt = await engine.get_library_prompt()

        # high_score should appear before low_score
        assert prompt.index("high_score") < prompt.index("low_score")

    @pytest.mark.asyncio
    async def test_library_prompt_limits_count(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._library = [
            _make_abstraction(f"fn_{i}", usage=1, confidence=0.5)
            for i in range(20)
        ]
        engine._library_loaded = True

        # Default max_abstractions is 15
        prompt = await engine.get_library_prompt(max_abstractions=3)

        # Should only have 3 numbered entries
        assert prompt.count("## ") == 3


# ── Usage Tracking ──────────────────────────────────────────────────────────


class TestUsageTracking:
    @pytest.mark.asyncio
    async def test_record_usage_increments_count(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        abs_ = _make_abstraction("my_fn", usage=1, confidence=0.5)
        engine._library = [abs_]
        engine._library_loaded = True

        await engine.record_usage("my_fn")

        assert abs_.usage_count == 2
        assert abs_.confidence == 0.55  # +0.05
        assert abs_.last_used_at is not None

    @pytest.mark.asyncio
    async def test_record_usage_nonexistent_is_noop(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._library = [_make_abstraction("other_fn")]
        engine._library_loaded = True

        # Should not raise
        await engine.record_usage("nonexistent_fn")


# ── Consolidation ───────────────────────────────────────────────────────────


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_decay_unused_abstractions(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        abs_ = _make_abstraction("unused", confidence=0.5)
        abs_.last_used_at = None  # Never used
        engine._library = [abs_]
        engine._library_loaded = True

        result = await engine.consolidate()

        assert result["decayed"] >= 1
        assert abs_.confidence < 0.5

    @pytest.mark.asyncio
    async def test_prune_low_confidence(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._library = [
            _make_abstraction("good", confidence=0.8),
            _make_abstraction("bad", confidence=0.1),  # Below _PRUNE_THRESHOLD
        ]
        engine._library_loaded = True

        result = await engine.consolidate()

        assert result["pruned"] >= 1
        names = [a.name for a in engine._library]
        assert "good" in names
        assert "bad" not in names

    @pytest.mark.asyncio
    async def test_merge_similar_abstractions(self, tmp_path: Path):
        engine = _make_engine(tmp_path)

        # Two abstractions with identical normalized source
        source = "def merge_me():\n    return 42\n"
        engine._library = [
            _make_abstraction("fn_a", source=source, usage=3, confidence=0.6),
            _make_abstraction("fn_b", source=source, usage=2, confidence=0.4),
        ]
        engine._library_loaded = True

        result = await engine.consolidate()

        assert result["merged"] >= 1
        assert len(engine._library) == 1
        # Merged abstraction should have combined usage
        assert engine._library[0].usage_count == 5

    @pytest.mark.asyncio
    async def test_cap_library_size(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._library = [
            _make_abstraction(
                f"fn_{i}",
                source=f"def fn_{i}():\n    return {i}\n    # pad",
                usage=i,
                confidence=0.5,
            )
            for i in range(250)
        ]
        engine._library_loaded = True

        await engine.consolidate()

        assert len(engine._library) <= 200  # _MAX_LIBRARY_SIZE


# ── Similarity Detection ───────────────────────────────────────────────────


class TestSimilarity:
    def test_find_similar_exact_match(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        source = "def x():\n    return 1\n"
        existing = _make_abstraction("existing", source=source)
        engine._library = [existing]

        new = _make_abstraction("new_fn", source=source)
        result = engine._find_similar_in_library(new)

        assert result is existing

    def test_find_similar_name_and_kind_match(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        existing = _make_abstraction(
            "validate_input", kind=AbstractionKind.VALIDATION_GUARD,
        )
        engine._library = [existing]

        new = _make_abstraction(
            "validate_input",
            kind=AbstractionKind.VALIDATION_GUARD,
            source="def validate_input():\n    return True\n    # v2",
        )
        result = engine._find_similar_in_library(new)

        assert result is existing

    def test_find_similar_no_match(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._library = [_make_abstraction("unrelated")]

        new = _make_abstraction(
            "totally_different",
            source="def totally_different():\n    return 999\n    # pad",
        )
        result = engine._find_similar_in_library(new)

        assert result is None

    def test_are_similar_same_source(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        source = "def x():\n    return 1\n"
        a = _make_abstraction("a", source=source)
        b = _make_abstraction("b", source=source)
        assert engine._are_similar(a, b) is True

    def test_are_similar_different_source(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        a = _make_abstraction("a", source="def a():\n    return 1\n")
        b = _make_abstraction("b", source="def b():\n    return 2\n")
        assert engine._are_similar(a, b) is False


# ── Extract from Proposals (Integration) ─────────────────────────────────


class TestExtractFromProposals:
    @pytest.mark.asyncio
    async def test_no_files_returns_empty(self, tmp_path: Path):
        engine = _make_engine(tmp_path)

        result = await engine.extract_from_proposals(
            proposal_ids=["p1"],
            files_changed={"p1": []},
        )

        assert isinstance(result, AbstractionExtractionResult)
        assert result.total_proposals_analyzed == 1
        assert result.extracted == []

    @pytest.mark.asyncio
    async def test_extracts_patterns_across_proposals(self, tmp_path: Path):
        """Common code in two proposals should produce an abstraction."""
        # Both proposals write the same validation function
        common_body = """\
        def validate_config(cfg: dict) -> bool:
            if not isinstance(cfg, dict):
                return False
            if "name" not in cfg:
                return False
            return True
        """
        _write_py(
            tmp_path,
            "systems/alpha/validators.py",
            common_body,
        )
        _write_py(
            tmp_path,
            "systems/beta/validators.py",
            common_body,
        )

        engine = _make_engine(tmp_path)

        result = await engine.extract_from_proposals(
            proposal_ids=["p1", "p2"],
            files_changed={
                "p1": ["systems/alpha/validators.py"],
                "p2": ["systems/beta/validators.py"],
            },
        )

        assert isinstance(result, AbstractionExtractionResult)
        assert result.total_proposals_analyzed == 2
        # Should find the common validation pattern
        assert len(result.extracted) >= 1

    @pytest.mark.asyncio
    async def test_with_llm_naming(self, tmp_path: Path):
        """When LLM is available, it should name the abstraction."""
        common_body = """\
        def check_data(data):
            if not data:
                raise ValueError("empty")
            return data
        """
        _write_py(tmp_path, "systems/a/check.py", common_body)
        _write_py(tmp_path, "systems/b/check.py", common_body)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "name: safe_data_check\ndescription: Validates data is non-empty"
        mock_llm.evaluate = AsyncMock(return_value=mock_response)

        engine = LiloLibraryEngine(
            neo4j=None,
            llm=mock_llm,
            codebase_root=tmp_path,
        )

        result = await engine.extract_from_proposals(
            proposal_ids=["p1", "p2"],
            files_changed={
                "p1": ["systems/a/check.py"],
                "p2": ["systems/b/check.py"],
            },
        )

        if result.extracted:
            # LLM should have been used for naming
            assert any(
                a.name == "safe_data_check" or a.description == "Validates data is non-empty"
                for a in result.extracted
            )


# ── Stats ───────────────────────────────────────────────────────────────────


class TestLibraryStats:
    @pytest.mark.asyncio
    async def test_empty_stats(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        stats = await engine.get_stats()

        assert isinstance(stats, LibraryStats)
        assert stats.total_abstractions == 0
        assert stats.total_usage_count == 0

    @pytest.mark.asyncio
    async def test_stats_reflect_library(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine._library = [
            _make_abstraction("fn_a", kind=AbstractionKind.UTILITY_FUNCTION, usage=3, confidence=0.8),
            _make_abstraction("fn_b", kind=AbstractionKind.VALIDATION_GUARD, usage=5, confidence=0.6),
        ]
        engine._library_loaded = True

        stats = await engine.get_stats()

        assert stats.total_abstractions == 2
        assert stats.total_usage_count == 8
        assert stats.by_kind[AbstractionKind.UTILITY_FUNCTION.value] == 1
        assert stats.by_kind[AbstractionKind.VALIDATION_GUARD.value] == 1
        assert 0.6 <= stats.mean_confidence <= 0.8
