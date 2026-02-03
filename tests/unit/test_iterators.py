"""
Unit tests for MathViz iterator runtime functions.

Tests the runtime helper functions directly, independent of the
compilation pipeline, to ensure they behave correctly.
"""

from mathviz.runtime.iterators import (
    dict_filter_keys,
    dict_filter_values,
    dict_items,
    # Dict-specific
    dict_keys,
    dict_map_values,
    dict_values,
    iter_all,
    # Predicate
    iter_any,
    iter_average,
    iter_chain,
    iter_chunk,
    iter_collect_dict,
    # Collection
    iter_collect_list,
    iter_collect_set,
    iter_count,
    iter_count_if,
    iter_enumerate,
    iter_filter,
    iter_find,
    # Access
    iter_first,
    iter_flat_map,
    iter_flatten,
    iter_fold,
    iter_group_by,
    iter_join,
    iter_last,
    # Transformation
    iter_map,
    iter_max,
    iter_max_by,
    iter_min,
    iter_min_by,
    iter_none,
    iter_nth,
    iter_partition,
    iter_position,
    iter_product,
    iter_reduce,
    iter_reversed,
    iter_skip,
    iter_skip_while,
    # Ordering
    iter_sorted,
    iter_sorted_by,
    iter_sorted_by_desc,
    # Numeric
    iter_sum,
    # Slicing
    iter_take,
    iter_take_while,
    iter_unique,
    # Combination
    iter_zip,
)


class TestTransformationMethods:
    """Tests for transformation iterator methods."""

    def test_map_basic(self):
        """Test basic map functionality."""
        result = iter_map([1, 2, 3], lambda x: x * 2)
        assert result == [2, 4, 6]

    def test_map_empty(self):
        """Test map on empty list."""
        result = iter_map([], lambda x: x * 2)
        assert result == []

    def test_filter_basic(self):
        """Test basic filter functionality."""
        result = iter_filter([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        assert result == [2, 4]

    def test_filter_all_match(self):
        """Test filter when all elements match."""
        result = iter_filter([2, 4, 6], lambda x: x % 2 == 0)
        assert result == [2, 4, 6]

    def test_filter_none_match(self):
        """Test filter when no elements match."""
        result = iter_filter([1, 3, 5], lambda x: x % 2 == 0)
        assert result == []

    def test_reduce_sum(self):
        """Test reduce for summing."""
        result = iter_reduce([1, 2, 3, 4], 0, lambda acc, x: acc + x)
        assert result == 10

    def test_reduce_product(self):
        """Test reduce for product."""
        result = iter_reduce([1, 2, 3, 4], 1, lambda acc, x: acc * x)
        assert result == 24

    def test_fold_is_alias(self):
        """Test that fold is an alias for reduce."""
        result = iter_fold([1, 2, 3, 4], 0, lambda acc, x: acc + x)
        assert result == 10

    def test_flat_map(self):
        """Test flat_map functionality."""
        result = iter_flat_map([1, 2, 3], lambda x: [x, x * 10])
        assert result == [1, 10, 2, 20, 3, 30]

    def test_flatten(self):
        """Test flatten functionality."""
        result = iter_flatten([[1, 2], [3, 4], [5]])
        assert result == [1, 2, 3, 4, 5]


class TestAccessMethods:
    """Tests for access iterator methods."""

    def test_first_basic(self):
        """Test first on non-empty list."""
        assert iter_first([1, 2, 3]) == 1

    def test_first_empty(self):
        """Test first on empty list."""
        assert iter_first([]) is None

    def test_last_basic(self):
        """Test last on non-empty list."""
        assert iter_last([1, 2, 3]) == 3

    def test_last_empty(self):
        """Test last on empty list."""
        assert iter_last([]) is None

    def test_nth_valid(self):
        """Test nth with valid index."""
        assert iter_nth([1, 2, 3], 1) == 2

    def test_nth_out_of_bounds(self):
        """Test nth with out of bounds index."""
        assert iter_nth([1, 2, 3], 10) is None

    def test_find_exists(self):
        """Test find when element exists."""
        result = iter_find([1, 2, 3, 4], lambda x: x > 2)
        assert result == 3

    def test_find_not_exists(self):
        """Test find when element doesn't exist."""
        result = iter_find([1, 2, 3], lambda x: x > 10)
        assert result is None

    def test_position_exists(self):
        """Test position when element exists."""
        result = iter_position([1, 2, 3, 4], lambda x: x > 2)
        assert result == 2

    def test_position_not_exists(self):
        """Test position when element doesn't exist."""
        result = iter_position([1, 2, 3], lambda x: x > 10)
        assert result is None


class TestPredicateMethods:
    """Tests for predicate iterator methods."""

    def test_any_true(self):
        """Test any when some match."""
        assert iter_any([1, 2, 3], lambda x: x > 2) is True

    def test_any_false(self):
        """Test any when none match."""
        assert iter_any([1, 2, 3], lambda x: x > 10) is False

    def test_all_true(self):
        """Test all when all match."""
        assert iter_all([2, 4, 6], lambda x: x % 2 == 0) is True

    def test_all_false(self):
        """Test all when not all match."""
        assert iter_all([1, 2, 3], lambda x: x % 2 == 0) is False

    def test_none_true(self):
        """Test none when none match."""
        assert iter_none([1, 2, 3], lambda x: x > 10) is True

    def test_none_false(self):
        """Test none when some match."""
        assert iter_none([1, 2, 3], lambda x: x > 1) is False

    def test_count(self):
        """Test count functionality."""
        assert iter_count([1, 2, 3, 4, 5]) == 5

    def test_count_empty(self):
        """Test count on empty list."""
        assert iter_count([]) == 0

    def test_count_if(self):
        """Test count_if functionality."""
        assert iter_count_if([1, 2, 3, 4, 5], lambda x: x % 2 == 0) == 2


class TestNumericMethods:
    """Tests for numeric iterator methods."""

    def test_sum(self):
        """Test sum functionality."""
        assert iter_sum([1, 2, 3, 4, 5]) == 15

    def test_sum_empty(self):
        """Test sum on empty list."""
        assert iter_sum([]) == 0

    def test_product(self):
        """Test product functionality."""
        assert iter_product([1, 2, 3, 4]) == 24

    def test_product_empty(self):
        """Test product on empty list."""
        assert iter_product([]) == 1

    def test_min(self):
        """Test min functionality."""
        assert iter_min([3, 1, 4, 1, 5]) == 1

    def test_min_empty(self):
        """Test min on empty list."""
        assert iter_min([]) is None

    def test_max(self):
        """Test max functionality."""
        assert iter_max([3, 1, 4, 1, 5]) == 5

    def test_max_empty(self):
        """Test max on empty list."""
        assert iter_max([]) is None

    def test_average(self):
        """Test average functionality."""
        assert iter_average([1, 2, 3, 4, 5]) == 3.0

    def test_average_empty(self):
        """Test average on empty list."""
        assert iter_average([]) is None

    def test_min_by(self):
        """Test min_by functionality."""
        result = iter_min_by(["aaa", "b", "cc"], lambda x: len(x))
        assert result == "b"

    def test_max_by(self):
        """Test max_by functionality."""
        result = iter_max_by(["aaa", "b", "cc"], lambda x: len(x))
        assert result == "aaa"


class TestSlicingMethods:
    """Tests for slicing iterator methods."""

    def test_take(self):
        """Test take functionality."""
        assert iter_take([1, 2, 3, 4, 5], 3) == [1, 2, 3]

    def test_take_more_than_available(self):
        """Test take with n > length."""
        assert iter_take([1, 2], 5) == [1, 2]

    def test_skip(self):
        """Test skip functionality."""
        assert iter_skip([1, 2, 3, 4, 5], 2) == [3, 4, 5]

    def test_skip_more_than_available(self):
        """Test skip with n > length."""
        assert iter_skip([1, 2], 5) == []

    def test_take_while(self):
        """Test take_while functionality."""
        result = iter_take_while([1, 2, 3, 4, 1], lambda x: x < 4)
        assert result == [1, 2, 3]

    def test_skip_while(self):
        """Test skip_while functionality."""
        result = iter_skip_while([1, 2, 3, 4, 1], lambda x: x < 3)
        assert result == [3, 4, 1]


class TestOrderingMethods:
    """Tests for ordering iterator methods."""

    def test_sorted(self):
        """Test sorted functionality."""
        assert iter_sorted([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

    def test_sorted_by(self):
        """Test sorted_by functionality."""
        result = iter_sorted_by(["bbb", "a", "cc"], lambda x: len(x))
        assert result == ["a", "cc", "bbb"]

    def test_sorted_by_desc(self):
        """Test sorted_by_desc functionality."""
        result = iter_sorted_by_desc(["a", "bbb", "cc"], lambda x: len(x))
        assert result == ["bbb", "cc", "a"]

    def test_reversed(self):
        """Test reversed functionality."""
        assert iter_reversed([1, 2, 3]) == [3, 2, 1]


class TestCombinationMethods:
    """Tests for combination iterator methods."""

    def test_zip(self):
        """Test zip functionality."""
        result = iter_zip([1, 2, 3], ["a", "b", "c"])
        assert result == [(1, "a"), (2, "b"), (3, "c")]

    def test_enumerate(self):
        """Test enumerate functionality."""
        result = iter_enumerate(["a", "b", "c"])
        assert result == [(0, "a"), (1, "b"), (2, "c")]

    def test_chain(self):
        """Test chain functionality."""
        result = iter_chain([1, 2], [3, 4], [5])
        assert result == [1, 2, 3, 4, 5]

    def test_chunk(self):
        """Test chunk functionality."""
        result = iter_chunk([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_unique(self):
        """Test unique functionality."""
        result = iter_unique([1, 2, 2, 3, 1, 4])
        assert result == [1, 2, 3, 4]


class TestCollectionMethods:
    """Tests for collection iterator methods."""

    def test_collect_list(self):
        """Test collect_list functionality."""
        result = iter_collect_list(x for x in range(3))
        assert result == [0, 1, 2]

    def test_collect_set(self):
        """Test collect_set functionality."""
        result = iter_collect_set([1, 2, 2, 3])
        assert result == {1, 2, 3}

    def test_collect_dict(self):
        """Test collect_dict functionality."""
        result = iter_collect_dict([("a", 1), ("b", 2)])
        assert result == {"a": 1, "b": 2}

    def test_join(self):
        """Test join functionality."""
        result = iter_join(["a", "b", "c"], ", ")
        assert result == "a, b, c"

    def test_group_by(self):
        """Test group_by functionality."""
        result = iter_group_by([1, 2, 3, 4, 5], lambda x: x % 2)
        assert result == {0: [2, 4], 1: [1, 3, 5]}

    def test_partition(self):
        """Test partition functionality."""
        trues, falses = iter_partition([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        assert trues == [2, 4]
        assert falses == [1, 3, 5]


class TestDictMethods:
    """Tests for dict-specific iterator methods."""

    def test_dict_keys(self):
        """Test dict_keys functionality."""
        result = dict_keys({"a": 1, "b": 2, "c": 3})
        assert set(result) == {"a", "b", "c"}

    def test_dict_values(self):
        """Test dict_values functionality."""
        result = dict_values({"a": 1, "b": 2, "c": 3})
        assert set(result) == {1, 2, 3}

    def test_dict_items(self):
        """Test dict_items functionality."""
        result = dict_items({"a": 1, "b": 2})
        assert set(result) == {("a", 1), ("b", 2)}

    def test_dict_map_values(self):
        """Test dict_map_values functionality."""
        result = dict_map_values({"a": 1, "b": 2}, lambda x: x * 2)
        assert result == {"a": 2, "b": 4}

    def test_dict_filter_keys(self):
        """Test dict_filter_keys functionality."""
        result = dict_filter_keys({"a": 1, "b": 2, "c": 3}, lambda k: k != "b")
        assert result == {"a": 1, "c": 3}

    def test_dict_filter_values(self):
        """Test dict_filter_values functionality."""
        result = dict_filter_values({"a": 1, "b": 2, "c": 3}, lambda v: v > 1)
        assert result == {"b": 2, "c": 3}


class TestChainedOperations:
    """Tests for chaining multiple iterator operations."""

    def test_map_filter_chain(self):
        """Test chaining map and filter."""
        result = iter_filter(iter_map([1, 2, 3, 4, 5], lambda x: x * 2), lambda x: x > 5)
        assert result == [6, 8, 10]

    def test_filter_map_reduce(self):
        """Test chaining filter, map, and reduce."""
        filtered = iter_filter([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        mapped = iter_map(filtered, lambda x: x * 2)
        result = iter_reduce(mapped, 0, lambda acc, x: acc + x)
        assert result == 12  # (2*2) + (4*2) = 4 + 8 = 12

    def test_take_sorted_map(self):
        """Test chaining take, sorted, and map."""
        taken = iter_take([5, 3, 1, 4, 2], 3)
        sorted_vals = iter_sorted(taken)
        result = iter_map(sorted_vals, lambda x: x * 10)
        assert result == [10, 30, 50]
