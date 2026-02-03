"""
Unit tests for the MathViz Runtime Library.
"""

import pytest

from mathviz.runtime.math_ops import (
    set_union,
    set_intersection,
    set_difference,
    is_subset,
    is_superset,
    is_proper_subset,
    is_proper_superset,
    is_element,
    is_not_element,
    symmetric_difference,
    cartesian_product,
    power_set,
)


class TestSetUnion:
    """Tests for set union operation."""

    def test_basic_union(self):
        """Test basic set union."""
        assert set_union({1, 2}, {2, 3}) == {1, 2, 3}

    def test_disjoint_sets(self):
        """Test union of disjoint sets."""
        assert set_union({1, 2}, {3, 4}) == {1, 2, 3, 4}

    def test_identical_sets(self):
        """Test union of identical sets."""
        assert set_union({1, 2, 3}, {1, 2, 3}) == {1, 2, 3}

    def test_empty_set(self):
        """Test union with empty set."""
        assert set_union({1, 2}, set()) == {1, 2}
        assert set_union(set(), {1, 2}) == {1, 2}

    def test_with_list_input(self):
        """Test union accepts list input."""
        assert set_union([1, 2], [2, 3]) == {1, 2, 3}


class TestSetIntersection:
    """Tests for set intersection operation."""

    def test_basic_intersection(self):
        """Test basic set intersection."""
        assert set_intersection({1, 2, 3}, {2, 3, 4}) == {2, 3}

    def test_disjoint_sets(self):
        """Test intersection of disjoint sets."""
        assert set_intersection({1, 2}, {3, 4}) == set()

    def test_identical_sets(self):
        """Test intersection of identical sets."""
        assert set_intersection({1, 2, 3}, {1, 2, 3}) == {1, 2, 3}

    def test_empty_set(self):
        """Test intersection with empty set."""
        assert set_intersection({1, 2}, set()) == set()


class TestSetDifference:
    """Tests for set difference operation."""

    def test_basic_difference(self):
        """Test basic set difference."""
        assert set_difference({1, 2, 3}, {2, 3, 4}) == {1}

    def test_disjoint_sets(self):
        """Test difference of disjoint sets."""
        assert set_difference({1, 2}, {3, 4}) == {1, 2}

    def test_subset_difference(self):
        """Test difference when first is subset."""
        assert set_difference({1, 2}, {1, 2, 3}) == set()

    def test_empty_result(self):
        """Test difference resulting in empty set."""
        assert set_difference({1, 2}, {1, 2}) == set()


class TestSubsetSuperset:
    """Tests for subset/superset operations."""

    def test_is_subset_true(self):
        """Test true subset case."""
        assert is_subset({1, 2}, {1, 2, 3}) is True

    def test_is_subset_equal(self):
        """Test subset with equal sets."""
        assert is_subset({1, 2}, {1, 2}) is True

    def test_is_subset_false(self):
        """Test false subset case."""
        assert is_subset({1, 4}, {1, 2, 3}) is False

    def test_is_superset_true(self):
        """Test true superset case."""
        assert is_superset({1, 2, 3}, {1, 2}) is True

    def test_is_superset_equal(self):
        """Test superset with equal sets."""
        assert is_superset({1, 2}, {1, 2}) is True

    def test_is_superset_false(self):
        """Test false superset case."""
        assert is_superset({1, 2}, {1, 2, 3}) is False

    def test_is_proper_subset_true(self):
        """Test true proper subset case."""
        assert is_proper_subset({1, 2}, {1, 2, 3}) is True

    def test_is_proper_subset_equal(self):
        """Test proper subset with equal sets (should be false)."""
        assert is_proper_subset({1, 2}, {1, 2}) is False

    def test_is_proper_superset_true(self):
        """Test true proper superset case."""
        assert is_proper_superset({1, 2, 3}, {1, 2}) is True

    def test_is_proper_superset_equal(self):
        """Test proper superset with equal sets (should be false)."""
        assert is_proper_superset({1, 2}, {1, 2}) is False


class TestElementMembership:
    """Tests for element membership operations."""

    def test_is_element_true(self):
        """Test true element membership."""
        assert is_element(2, {1, 2, 3}) is True

    def test_is_element_false(self):
        """Test false element membership."""
        assert is_element(5, {1, 2, 3}) is False

    def test_is_element_list(self):
        """Test element membership in list."""
        assert is_element(2, [1, 2, 3]) is True

    def test_is_not_element_true(self):
        """Test true non-membership."""
        assert is_not_element(5, {1, 2, 3}) is True

    def test_is_not_element_false(self):
        """Test false non-membership."""
        assert is_not_element(2, {1, 2, 3}) is False


class TestSymmetricDifference:
    """Tests for symmetric difference operation."""

    def test_basic_symmetric_difference(self):
        """Test basic symmetric difference."""
        assert symmetric_difference({1, 2, 3}, {2, 3, 4}) == {1, 4}

    def test_identical_sets(self):
        """Test symmetric difference of identical sets."""
        assert symmetric_difference({1, 2}, {1, 2}) == set()

    def test_disjoint_sets(self):
        """Test symmetric difference of disjoint sets."""
        assert symmetric_difference({1, 2}, {3, 4}) == {1, 2, 3, 4}


class TestCartesianProduct:
    """Tests for Cartesian product operation."""

    def test_basic_product(self):
        """Test basic Cartesian product."""
        result = cartesian_product({1, 2}, {"a", "b"})
        expected = {(1, "a"), (1, "b"), (2, "a"), (2, "b")}
        assert result == expected

    def test_empty_set(self):
        """Test Cartesian product with empty set."""
        assert cartesian_product(set(), {1, 2}) == set()
        assert cartesian_product({1, 2}, set()) == set()


class TestPowerSet:
    """Tests for power set operation."""

    def test_small_set(self):
        """Test power set of small set."""
        result = power_set({1, 2})
        expected = {
            frozenset(),
            frozenset({1}),
            frozenset({2}),
            frozenset({1, 2}),
        }
        assert result == expected

    def test_empty_set(self):
        """Test power set of empty set."""
        result = power_set(set())
        assert result == {frozenset()}

    def test_singleton(self):
        """Test power set of singleton."""
        result = power_set({1})
        expected = {frozenset(), frozenset({1})}
        assert result == expected

    def test_power_set_size(self):
        """Test that power set has 2^n elements."""
        for n in range(5):
            s = set(range(n))
            ps = power_set(s)
            assert len(ps) == 2**n
