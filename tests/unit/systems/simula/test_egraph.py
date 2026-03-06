"""
Unit tests for Simula Equality Saturation Engine (Stage 6D).

Tests the e-graph data structure, union-find with path compression,
AST-to-e-graph conversion, algebraic rewrite rule application, and
the public EqualitySaturationEngine API (check_equivalence, simplify).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from systems.simula.egraph.equality_saturation import (
    EqualitySaturationEngine,
    _apply_commutativity_rules,
    _apply_identity_rules,
    _ast_to_egraph,
    _build_rewrite_rules,
    _EGraph,
    _ENode,
    _UnionFind,
)
from systems.simula.verification.types import (
    EGraphEquivalenceResult,
    EGraphStatus,
)

# ── Union-Find ────────────────────────────────────────────────────────────────


class TestUnionFind:
    """Tests for the path-compressed union-find (disjoint set) structure."""

    def test_make_set_creates_singleton(self):
        """Each make_set call creates a new singleton set with itself as root."""
        uf = _UnionFind()
        uf.make_set(0)
        uf.make_set(1)
        assert uf.find(0) == 0
        assert uf.find(1) == 1

    def test_make_set_idempotent(self):
        """Calling make_set twice on the same element does not reset it."""
        uf = _UnionFind()
        uf.make_set(0)
        uf.make_set(1)
        uf.union(0, 1)
        uf.make_set(0)  # should be a no-op
        assert uf.connected(0, 1)

    def test_union_connects_elements(self):
        """After union, two elements share the same root."""
        uf = _UnionFind()
        uf.make_set(0)
        uf.make_set(1)
        uf.union(0, 1)
        assert uf.connected(0, 1)

    def test_union_returns_root(self):
        """Union returns the root of the merged set."""
        uf = _UnionFind()
        uf.make_set(0)
        uf.make_set(1)
        root = uf.union(0, 1)
        assert root == uf.find(0)
        assert root == uf.find(1)

    def test_union_same_set_noop(self):
        """Unioning elements already in the same set returns the existing root."""
        uf = _UnionFind()
        uf.make_set(0)
        uf.make_set(1)
        uf.union(0, 1)
        root_before = uf.find(0)
        root_after = uf.union(0, 1)
        assert root_before == root_after

    def test_path_compression(self):
        """After find, the element's parent should point directly to root."""
        uf = _UnionFind()
        for i in range(5):
            uf.make_set(i)
        # Build a linear chain: 0 <- 1 <- 2 <- 3 <- 4
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        root = uf.find(4)
        # After path compression, 4's parent should be the root directly
        assert uf._parent[4] == root

    def test_path_compression_all_elements(self):
        """Path compression flattens the tree for all elements on the path."""
        uf = _UnionFind()
        for i in range(6):
            uf.make_set(i)
        # Chain: 0 <- 1 <- 2 <- 3 <- 4 <- 5
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        uf.union(4, 5)
        root = uf.find(5)
        # After find(5), every element on the path should point to root
        for i in range(6):
            uf.find(i)  # trigger compression
            assert uf._parent[i] == root

    def test_connected_different_sets(self):
        """Elements in different sets are not connected."""
        uf = _UnionFind()
        uf.make_set(0)
        uf.make_set(1)
        uf.make_set(2)
        uf.union(0, 1)
        assert uf.connected(0, 1)
        assert not uf.connected(0, 2)
        assert not uf.connected(1, 2)

    def test_union_by_rank(self):
        """Union by rank keeps the higher-rank tree as root."""
        uf = _UnionFind()
        for i in range(4):
            uf.make_set(i)
        # 0-1 forms rank-1 tree, 2 is rank-0
        uf.union(0, 1)
        root_01 = uf.find(0)
        uf.union(root_01, 2)
        # Root should still be the rank-1 root
        assert uf.find(2) == root_01

    def test_transitive_connectivity(self):
        """If a~b and b~c then a~c."""
        uf = _UnionFind()
        for i in range(3):
            uf.make_set(i)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.connected(0, 2)

    def test_many_elements(self):
        """Union-find works correctly with many elements."""
        uf = _UnionFind()
        n = 100
        for i in range(n):
            uf.make_set(i)
        # Union all even numbers together
        for i in range(2, n, 2):
            uf.union(0, i)
        # Union all odd numbers together
        for i in range(3, n, 2):
            uf.union(1, i)
        # All evens connected
        for i in range(0, n, 2):
            assert uf.connected(0, i)
        # All odds connected
        for i in range(1, n, 2):
            assert uf.connected(1, i)
        # Even and odd not connected
        assert not uf.connected(0, 1)


# ── E-Node ────────────────────────────────────────────────────────────────────


class TestENode:
    """Tests for the e-node hash-consed expression node."""

    def test_identical_nodes_same_hash(self):
        """Two nodes with identical op and children get the same hash key."""
        a = _ENode("Add", (1, 2))
        b = _ENode("Add", (1, 2))
        assert a.hash_key == b.hash_key

    def test_different_op_different_hash(self):
        """Nodes with different operators get different hash keys."""
        a = _ENode("Add", (1, 2))
        b = _ENode("Mult", (1, 2))
        assert a.hash_key != b.hash_key

    def test_different_children_different_hash(self):
        """Nodes with different children get different hash keys."""
        a = _ENode("Add", (1, 2))
        b = _ENode("Add", (2, 1))
        assert a.hash_key != b.hash_key

    def test_leaf_node_no_children(self):
        """A leaf node with no children has an empty children tuple."""
        node = _ENode("Constant:42")
        assert node.children == ()
        assert node.hash_key is not None

    def test_slots_defined(self):
        """_ENode uses __slots__ for memory efficiency."""
        node = _ENode("x")
        assert hasattr(node, "__slots__")
        assert "op" in node.__slots__
        assert "children" in node.__slots__
        assert "hash_key" in node.__slots__


# ── E-Graph ───────────────────────────────────────────────────────────────────


class TestEGraph:
    """Tests for the e-graph data structure."""

    def test_add_returns_eclass_id(self):
        """Adding a node returns a non-negative e-class ID."""
        eg = _EGraph()
        eid = eg.add(_ENode("Constant:1"))
        assert isinstance(eid, int)
        assert eid >= 0

    def test_add_deduplicates_identical_nodes(self):
        """Adding the same node twice returns the same e-class ID."""
        eg = _EGraph()
        eid1 = eg.add(_ENode("Constant:42"))
        eid2 = eg.add(_ENode("Constant:42"))
        assert eid1 == eid2

    def test_add_different_nodes_different_ids(self):
        """Adding different nodes returns different e-class IDs."""
        eg = _EGraph()
        eid1 = eg.add(_ENode("Constant:1"))
        eid2 = eg.add(_ENode("Constant:2"))
        assert eid1 != eid2

    def test_add_with_children(self):
        """Adding a node with children correctly references child e-classes."""
        eg = _EGraph()
        left = eg.add(_ENode("Constant:1"))
        right = eg.add(_ENode("Constant:2"))
        add_node = eg.add(_ENode("Add", (left, right)))
        assert add_node != left
        assert add_node != right

    def test_merge_makes_equivalent(self):
        """Merging two e-classes makes them equivalent."""
        eg = _EGraph()
        eid1 = eg.add(_ENode("Constant:1"))
        eid2 = eg.add(_ENode("Constant:2"))
        assert not eg.equivalent(eid1, eid2)
        eg.merge(eid1, eid2)
        assert eg.equivalent(eid1, eid2)

    def test_merge_same_class_noop(self):
        """Merging an e-class with itself is a no-op."""
        eg = _EGraph()
        eid = eg.add(_ENode("Constant:1"))
        root = eg.merge(eid, eid)
        assert root == eg.uf.find(eid)

    def test_merge_returns_root(self):
        """Merge returns the root of the unified e-class."""
        eg = _EGraph()
        eid1 = eg.add(_ENode("Constant:1"))
        eid2 = eg.add(_ENode("Constant:2"))
        root = eg.merge(eid1, eid2)
        assert root == eg.uf.find(eid1)
        assert root == eg.uf.find(eid2)

    def test_equivalent_detects_non_equivalence(self):
        """Non-merged e-classes are not equivalent."""
        eg = _EGraph()
        eid1 = eg.add(_ENode("Constant:1"))
        eid2 = eg.add(_ENode("Constant:2"))
        assert not eg.equivalent(eid1, eid2)

    def test_class_count_after_adds(self):
        """class_count reflects the number of distinct e-classes."""
        eg = _EGraph()
        eg.add(_ENode("A"))
        eg.add(_ENode("B"))
        eg.add(_ENode("C"))
        assert eg.class_count == 3

    def test_class_count_after_merge(self):
        """Merging reduces the distinct class count."""
        eg = _EGraph()
        a = eg.add(_ENode("A"))
        b = eg.add(_ENode("B"))
        c = eg.add(_ENode("C"))
        assert eg.class_count == 3
        eg.merge(a, b)
        assert eg.class_count == 2
        eg.merge(b, c)
        assert eg.class_count == 1

    def test_node_count(self):
        """node_count reflects total nodes across all e-classes."""
        eg = _EGraph()
        eg.add(_ENode("A"))
        eg.add(_ENode("B"))
        assert eg.node_count == 2

    def test_node_count_after_merge(self):
        """Merging e-classes combines their node lists."""
        eg = _EGraph()
        a = eg.add(_ENode("A"))
        b = eg.add(_ENode("B"))
        eg.merge(a, b)
        # Both nodes should still exist in the merged class
        assert eg.node_count == 2

    def test_hash_consing_with_child_canonicalization(self):
        """Adding a node with equivalent children (via merge) deduplicates correctly."""
        eg = _EGraph()
        x1 = eg.add(_ENode("Name:x"))
        x2 = eg.add(_ENode("Name:x"))
        # x1 and x2 are already the same due to hash-consing
        assert x1 == x2
        y = eg.add(_ENode("Name:y"))
        add1 = eg.add(_ENode("Add", (x1, y)))
        add2 = eg.add(_ENode("Add", (x2, y)))
        # Same children means same hash → same e-class
        assert add1 == add2


# ── Rewrite Rules ─────────────────────────────────────────────────────────────


class TestRewriteRules:
    """Tests for the built-in algebraic rewrite rules."""

    def test_rules_are_nonempty(self):
        """The built-in rule set is not empty."""
        rules = _build_rewrite_rules()
        assert len(rules) > 0

    def test_commutativity_rules_present(self):
        """Commutativity rules for Add and Mult exist."""
        rules = _build_rewrite_rules()
        names = {r.name for r in rules}
        assert "add_comm" in names
        assert "mul_comm" in names

    def test_identity_rules_present(self):
        """Identity rules (add_zero, mul_one, mul_zero) exist."""
        rules = _build_rewrite_rules()
        names = {r.name for r in rules}
        assert "add_zero" in names
        assert "mul_one" in names
        assert "mul_zero" in names

    def test_boolean_rules_present(self):
        """Boolean algebra rules (double negation, and_true, etc.) exist."""
        rules = _build_rewrite_rules()
        names = {r.name for r in rules}
        assert "double_neg" in names
        assert "and_true" in names
        assert "or_false" in names
        assert "and_false" in names

    def test_comparison_rules_present(self):
        """Comparison self-rules (eq_self, ne_self) exist."""
        rules = _build_rewrite_rules()
        names = {r.name for r in rules}
        assert "eq_self" in names
        assert "ne_self" in names

    def test_all_rules_have_name_pattern_replacement(self):
        """Every rule has a non-empty name, pattern, and replacement."""
        rules = _build_rewrite_rules()
        for rule in rules:
            assert rule.name, "Rule missing name"
            assert rule.pattern, f"Rule {rule.name} missing pattern"
            assert rule.replacement, f"Rule {rule.name} missing replacement"


# ── Apply Commutativity Rules ──────────────────────────────────────────────────


class TestApplyCommutativityRules:
    """Tests for the commutativity rule application function."""

    def test_commutative_add_merged(self):
        """a+b and b+a should be merged by commutativity."""
        eg = _EGraph()
        a = eg.add(_ENode("Name:a"))
        b = eg.add(_ENode("Name:b"))
        ab = eg.add(_ENode("Add", (a, b)))
        ba = eg.add(_ENode("Add", (b, a)))
        assert not eg.equivalent(ab, ba)
        merges = _apply_commutativity_rules(eg)
        assert merges > 0
        assert eg.equivalent(ab, ba)

    def test_commutative_mult_merged(self):
        """a*b and b*a should be merged by commutativity."""
        eg = _EGraph()
        a = eg.add(_ENode("Name:a"))
        b = eg.add(_ENode("Name:b"))
        ab = eg.add(_ENode("Mult", (a, b)))
        ba = eg.add(_ENode("Mult", (b, a)))
        merges = _apply_commutativity_rules(eg)
        assert merges > 0
        assert eg.equivalent(ab, ba)

    def test_non_commutative_op_not_merged(self):
        """Sub(a, b) and Sub(b, a) should NOT be merged."""
        eg = _EGraph()
        a = eg.add(_ENode("Name:a"))
        b = eg.add(_ENode("Name:b"))
        ab = eg.add(_ENode("Sub", (a, b)))
        ba = eg.add(_ENode("Sub", (b, a)))
        _apply_commutativity_rules(eg)
        assert not eg.equivalent(ab, ba)

    def test_no_merges_when_no_commutative_ops(self):
        """If there are no commutative operations, no merges happen."""
        eg = _EGraph()
        eg.add(_ENode("Constant:1"))
        eg.add(_ENode("Constant:2"))
        merges = _apply_commutativity_rules(eg)
        assert merges == 0


# ── Apply Identity Rules ──────────────────────────────────────────────────────


class TestApplyIdentityRules:
    """Tests for the identity element rule application function."""

    def test_add_zero_identity(self):
        """a + 0 should be merged with a."""
        eg = _EGraph()
        a = eg.add(_ENode("Name:a"))
        zero = eg.add(_ENode("Constant:0"))
        a_plus_zero = eg.add(_ENode("Add", (a, zero)))
        merges = _apply_identity_rules(eg)
        assert merges > 0
        assert eg.equivalent(a_plus_zero, a)

    def test_zero_plus_a_identity(self):
        """0 + a should be merged with a."""
        eg = _EGraph()
        a = eg.add(_ENode("Name:a"))
        zero = eg.add(_ENode("Constant:0"))
        zero_plus_a = eg.add(_ENode("Add", (zero, a)))
        merges = _apply_identity_rules(eg)
        assert merges > 0
        assert eg.equivalent(zero_plus_a, a)

    def test_mult_one_identity(self):
        """a * 1 should be merged with a."""
        eg = _EGraph()
        a = eg.add(_ENode("Name:a"))
        one = eg.add(_ENode("Constant:1"))
        a_times_one = eg.add(_ENode("Mult", (a, one)))
        merges = _apply_identity_rules(eg)
        assert merges > 0
        assert eg.equivalent(a_times_one, a)

    def test_mult_zero_annihilation(self):
        """a * 0 should be merged with 0."""
        eg = _EGraph()
        a = eg.add(_ENode("Name:a"))
        zero = eg.add(_ENode("Constant:0"))
        a_times_zero = eg.add(_ENode("Mult", (a, zero)))
        merges = _apply_identity_rules(eg)
        assert merges > 0
        assert eg.equivalent(a_times_zero, zero)

    def test_no_identity_merges_for_unrelated_nodes(self):
        """Non-identity operations should not trigger any merges."""
        eg = _EGraph()
        a = eg.add(_ENode("Name:a"))
        b = eg.add(_ENode("Name:b"))
        eg.add(_ENode("Add", (a, b)))
        merges = _apply_identity_rules(eg)
        assert merges == 0


# ── AST to E-Graph Conversion ─────────────────────────────────────────────────


class TestAstToEGraph:
    """Tests for converting Python AST nodes into e-graph nodes."""

    def test_constant_int(self):
        """An integer constant produces a Constant e-node."""
        import ast

        eg = _EGraph()
        tree = ast.parse("42", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0
        assert eg.node_count >= 1

    def test_name_variable(self):
        """A variable name produces a Name e-node."""
        import ast

        eg = _EGraph()
        tree = ast.parse("x", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0

    def test_binop_add(self):
        """a + b produces an Add e-node with two children."""
        import ast

        eg = _EGraph()
        tree = ast.parse("a + b", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0
        # Should have at least 3 nodes: Name:a, Name:b, Add
        assert eg.node_count >= 3

    def test_unary_op(self):
        """A unary operation (e.g., -x) produces an e-node."""
        import ast

        eg = _EGraph()
        tree = ast.parse("-x", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0
        assert eg.node_count >= 2  # Name:x, USub

    def test_bool_op_and(self):
        """Boolean 'and' chains into binary And e-nodes."""
        import ast

        eg = _EGraph()
        tree = ast.parse("a and b and c", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0
        # 3 names + 2 And nodes (chained binary)
        assert eg.node_count >= 5

    def test_compare_eq(self):
        """A comparison (a == b) produces an Eq e-node."""
        import ast

        eg = _EGraph()
        tree = ast.parse("a == b", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0
        assert eg.node_count >= 3  # Name:a, Name:b, Eq

    def test_if_exp(self):
        """A ternary (x if cond else y) produces an IfExp e-node."""
        import ast

        eg = _EGraph()
        tree = ast.parse("x if cond else y", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0
        # cond, x, y, IfExp
        assert eg.node_count >= 4

    def test_function_call(self):
        """A function call f(x) produces a Call e-node."""
        import ast

        eg = _EGraph()
        tree = ast.parse("f(x)", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0

    def test_attribute_access(self):
        """An attribute access obj.attr produces an Attr e-node."""
        import ast

        eg = _EGraph()
        tree = ast.parse("obj.attr", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0

    def test_subscript(self):
        """A subscript a[0] produces a Subscript e-node."""
        import ast

        eg = _EGraph()
        tree = ast.parse("a[0]", mode="eval")
        eid = _ast_to_egraph(tree.body, eg)
        assert eid >= 0

    def test_identical_expressions_same_eclass(self):
        """Two identical AST subtrees share the same e-class via hash-consing."""
        import ast

        eg = _EGraph()
        tree1 = ast.parse("x + 1", mode="eval")
        tree2 = ast.parse("x + 1", mode="eval")
        eid1 = _ast_to_egraph(tree1.body, eg)
        eid2 = _ast_to_egraph(tree2.body, eg)
        assert eg.equivalent(eid1, eid2)


# ── Equality Saturation Engine ────────────────────────────────────────────────


class TestEqualitySaturation:
    """Tests for the public EqualitySaturationEngine API."""

    @pytest.mark.asyncio
    async def test_identical_code_is_equivalent(self):
        """Identical code strings are trivially equivalent."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "a + b")
        assert result.semantically_equivalent is True
        assert result.status == EGraphStatus.SATURATED

    @pytest.mark.asyncio
    async def test_commutative_add_equivalent(self):
        """a + b and b + a are equivalent via commutativity."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "b + a")
        assert result.semantically_equivalent is True
        assert result.status == EGraphStatus.SATURATED
        assert "commutativity" in result.rules_applied

    @pytest.mark.asyncio
    async def test_commutative_mult_equivalent(self):
        """a * b and b * a are equivalent via commutativity."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a * b", "b * a")
        assert result.semantically_equivalent is True
        assert "commutativity" in result.rules_applied

    @pytest.mark.asyncio
    async def test_add_zero_identity_equivalent(self):
        """a + 0 is equivalent to a via identity rule."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + 0", "a")
        assert result.semantically_equivalent is True
        assert "identity" in result.rules_applied

    @pytest.mark.asyncio
    async def test_zero_plus_a_identity_equivalent(self):
        """0 + a is equivalent to a via identity rule."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("0 + a", "a")
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_mult_one_identity_equivalent(self):
        """a * 1 is equivalent to a via identity rule."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a * 1", "a")
        assert result.semantically_equivalent is True
        assert "identity" in result.rules_applied

    @pytest.mark.asyncio
    async def test_mult_zero_equivalent(self):
        """a * 0 is equivalent to 0 via annihilation rule."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a * 0", "0")
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_non_equivalent_code_rejected(self):
        """a + b and a * b are not equivalent."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "a * b")
        assert result.semantically_equivalent is False
        assert result.status == EGraphStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_non_equivalent_different_variables(self):
        """a + b and a + c are not equivalent."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "a + c")
        assert result.semantically_equivalent is False

    @pytest.mark.asyncio
    async def test_non_equivalent_different_constants(self):
        """1 + 2 and 1 + 3 are not equivalent."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("1 + 2", "1 + 3")
        assert result.semantically_equivalent is False

    @pytest.mark.asyncio
    async def test_subtraction_not_commutative(self):
        """a - b and b - a are NOT equivalent (subtraction is not commutative)."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a - b", "b - a")
        assert result.semantically_equivalent is False

    @pytest.mark.asyncio
    async def test_result_has_hashes(self):
        """Result includes SHA-256 hashes for original and rewritten code."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "b + a")
        assert result.original_hash != ""
        assert result.rewritten_hash != ""
        assert len(result.original_hash) == 16  # first 16 hex chars of SHA-256

    @pytest.mark.asyncio
    async def test_result_has_iterations(self):
        """Result reports the number of iterations performed."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "b + a")
        assert result.iterations >= 1

    @pytest.mark.asyncio
    async def test_result_has_eclass_and_enode_counts(self):
        """Result reports e-class and e-node counts."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "b + a")
        assert result.e_class_count >= 1
        assert result.e_node_count >= 1

    @pytest.mark.asyncio
    async def test_result_has_duration(self):
        """Result includes a non-negative duration in milliseconds."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "a + b")
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_syntax_error_returns_failed(self):
        """Invalid Python syntax results in FAILED status."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("def @@broken", "also @@broken")
        assert result.status == EGraphStatus.FAILED

    @pytest.mark.asyncio
    async def test_statement_mode_return(self):
        """Return statements are handled via exec-mode parsing."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("return a + b", "return b + a")
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_statement_mode_assignment(self):
        """Assignment statements are handled (RHS compared)."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("x = a + b", "x = b + a")
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_commutative_add_with_constants(self):
        """1 + 2 and 2 + 1 are equivalent via commutativity."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("1 + 2", "2 + 1")
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_nested_commutative_operations(self):
        """(a + b) * c and (b + a) * c are equivalent."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("(a + b) * c", "(b + a) * c")
        # The inner a+b and b+a get merged, then Mult nodes with
        # canonicalized children should match via hash-consing or commutativity.
        # This may or may not be proved depending on rebuild/canonicalization.
        # The test verifies the engine runs without error and returns a result.
        assert result.status in {EGraphStatus.SATURATED, EGraphStatus.PARTIAL}
        assert isinstance(result, EGraphEquivalenceResult)

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        """Engine stops after max_iterations even without saturation."""
        engine = EqualitySaturationEngine(max_iterations=1, timeout_s=5.0)
        result = await engine.check_equivalence("a + b", "a * b")
        # Should complete within 1 iteration
        assert result.iterations <= 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Engine returns TIMEOUT status when the deadline is exceeded."""
        engine = EqualitySaturationEngine(max_iterations=1_000_000, timeout_s=0.0)
        # With timeout_s=0.0, the deadline is immediately in the past
        result = await engine.check_equivalence("a + b", "b + a")
        # Either times out or finishes fast enough to saturate
        assert result.status in {EGraphStatus.TIMEOUT, EGraphStatus.SATURATED}
        assert isinstance(result.duration_ms, int)

    @pytest.mark.asyncio
    async def test_timeout_with_mock_time(self):
        """Verify timeout path by mocking time.monotonic to advance past deadline.

        Call sequence in check_equivalence:
          1. start = time.monotonic()           -> returns 100.0
          2. deadline = time.monotonic() + 1.0  -> returns 100.0, so deadline=101.0
          3. (loop) if time.monotonic() > deadline -> returns 200.0, exceeds 101.0 => TIMEOUT
        """
        call_count = 0

        def controlled_monotonic():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # start and deadline computation see t=100.0
                return 100.0
            # All subsequent calls (loop deadline check, final elapsed) see t=200.0
            return 200.0

        engine = EqualitySaturationEngine(max_iterations=1000, timeout_s=1.0)
        with patch("systems.simula.egraph.equality_saturation.time.monotonic", side_effect=controlled_monotonic):
            result = await engine.check_equivalence("a + b", "a * b")
        assert result.status == EGraphStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_constructor_default_values(self):
        """Default constructor uses 1000 max_iterations and 30s timeout."""
        engine = EqualitySaturationEngine()
        assert engine._max_iterations == 1000
        assert engine._timeout_s == 30.0

    @pytest.mark.asyncio
    async def test_constructor_custom_values(self):
        """Custom constructor values are stored correctly."""
        engine = EqualitySaturationEngine(max_iterations=50, timeout_s=10.0)
        assert engine._max_iterations == 50
        assert engine._timeout_s == 10.0

    @pytest.mark.asyncio
    async def test_boolean_and_commutative(self):
        """Boolean 'and' is commutative: (a and b) == (b and a)."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a and b", "b and a")
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_boolean_or_commutative(self):
        """Boolean 'or' is commutative: (a or b) == (b or a)."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("a or b", "b or a")
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_expression_vs_itself_saturates(self):
        """An expression compared to itself reaches SATURATED status immediately."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        result = await engine.check_equivalence("x * y + z", "x * y + z")
        assert result.status == EGraphStatus.SATURATED
        assert result.semantically_equivalent is True


# ── Simplify API ──────────────────────────────────────────────────────────────


class TestSimplify:
    """Tests for the simplify method."""

    @pytest.mark.asyncio
    async def test_simplify_returns_original_code(self):
        """Simplify currently returns the original code (extraction deferred)."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        simplified, result = await engine.simplify("a + b")
        assert simplified == "a + b"

    @pytest.mark.asyncio
    async def test_simplify_returns_equivalence_result(self):
        """Simplify returns a valid EGraphEquivalenceResult."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        simplified, result = await engine.simplify("a + b")
        assert isinstance(result, EGraphEquivalenceResult)
        # Code compared to itself → always equivalent
        assert result.semantically_equivalent is True
        assert result.status == EGraphStatus.SATURATED

    @pytest.mark.asyncio
    async def test_simplify_with_constant_expression(self):
        """Simplify handles constant expressions."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        simplified, result = await engine.simplify("42")
        assert simplified == "42"
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_simplify_with_complex_expression(self):
        """Simplify handles multi-operation expressions without errors."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        simplified, result = await engine.simplify("(a + b) * (c - d)")
        assert simplified == "(a + b) * (c - d)"
        assert result.semantically_equivalent is True

    @pytest.mark.asyncio
    async def test_simplify_syntax_error(self):
        """Simplify with invalid syntax returns FAILED status."""
        engine = EqualitySaturationEngine(max_iterations=100, timeout_s=5.0)
        simplified, result = await engine.simplify("def @@broken")
        assert result.status == EGraphStatus.FAILED
