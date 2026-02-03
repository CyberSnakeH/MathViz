"""
MathViz Iterator Runtime Support.

This module provides runtime helper functions for iterator methods
on collection types (List, Set, Dict). These enable functional-style
data transformations like map, filter, reduce, etc.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce as functools_reduce
from typing import (
    Any,
    TypeVar,
)

T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")
V = TypeVar("V")


# =============================================================================
# Transformation Methods
# =============================================================================


def iter_map(iterable: Iterable[T], func: Callable[[T], U]) -> list[U]:
    """
    Apply a function to each element and return a list of results.

    Example:
        iter_map([1, 2, 3], lambda x: x * 2) -> [2, 4, 6]
    """
    return list(map(func, iterable))


def iter_filter(iterable: Iterable[T], predicate: Callable[[T], bool]) -> list[T]:
    """
    Filter elements that satisfy the predicate.

    Example:
        iter_filter([1, 2, 3, 4], lambda x: x % 2 == 0) -> [2, 4]
    """
    return list(filter(predicate, iterable))


def iter_reduce(iterable: Iterable[T], initial: U, func: Callable[[U, T], U]) -> U:
    """
    Reduce the iterable to a single value using an accumulator function.

    Example:
        iter_reduce([1, 2, 3, 4], 0, lambda acc, x: acc + x) -> 10
    """
    return functools_reduce(func, iterable, initial)


def iter_fold(iterable: Iterable[T], initial: U, func: Callable[[U, T], U]) -> U:
    """
    Alias for reduce. Fold the iterable using an accumulator function.

    Example:
        iter_fold([1, 2, 3, 4], 0, lambda acc, x: acc + x) -> 10
    """
    return functools_reduce(func, iterable, initial)


def iter_flat_map(iterable: Iterable[T], func: Callable[[T], Iterable[U]]) -> list[U]:
    """
    Apply a function that returns an iterable to each element and flatten.

    Example:
        iter_flat_map([[1, 2], [3, 4]], lambda x: x) -> [1, 2, 3, 4]
    """
    result: list[U] = []
    for item in iterable:
        result.extend(func(item))
    return result


def iter_flatten[T](iterable: Iterable[Iterable[T]]) -> list[T]:
    """
    Flatten a nested iterable by one level.

    Example:
        iter_flatten([[1, 2], [3, 4]]) -> [1, 2, 3, 4]
    """
    result: list[T] = []
    for item in iterable:
        result.extend(item)
    return result


# =============================================================================
# Access Methods
# =============================================================================


def iter_first[T](iterable: Iterable[T]) -> T | None:
    """
    Return the first element, or None if empty.

    Example:
        iter_first([1, 2, 3]) -> 1
        iter_first([]) -> None
    """
    for item in iterable:
        return item
    return None


def iter_last[T](iterable: Iterable[T]) -> T | None:
    """
    Return the last element, or None if empty.

    Example:
        iter_last([1, 2, 3]) -> 3
        iter_last([]) -> None
    """
    result: T | None = None
    for item in iterable:
        result = item
    return result


def iter_nth[T](iterable: Iterable[T], n: int) -> T | None:
    """
    Return the nth element (0-indexed), or None if out of bounds.

    Example:
        iter_nth([1, 2, 3], 1) -> 2
        iter_nth([1, 2, 3], 5) -> None
    """
    for i, item in enumerate(iterable):
        if i == n:
            return item
    return None


def iter_find(iterable: Iterable[T], predicate: Callable[[T], bool]) -> T | None:
    """
    Find the first element that satisfies the predicate.

    Example:
        iter_find([1, 2, 3, 4], lambda x: x > 2) -> 3
    """
    for item in iterable:
        if predicate(item):
            return item
    return None


def iter_position(iterable: Iterable[T], predicate: Callable[[T], bool]) -> int | None:
    """
    Find the index of the first element that satisfies the predicate.

    Example:
        iter_position([1, 2, 3, 4], lambda x: x > 2) -> 2
    """
    for i, item in enumerate(iterable):
        if predicate(item):
            return i
    return None


# =============================================================================
# Predicate Methods
# =============================================================================


def iter_any(iterable: Iterable[T], predicate: Callable[[T], bool]) -> bool:
    """
    Check if any element satisfies the predicate.

    Example:
        iter_any([1, 2, 3], lambda x: x > 2) -> True
    """
    return any(predicate(item) for item in iterable)


def iter_all(iterable: Iterable[T], predicate: Callable[[T], bool]) -> bool:
    """
    Check if all elements satisfy the predicate.

    Example:
        iter_all([1, 2, 3], lambda x: x > 0) -> True
    """
    return all(predicate(item) for item in iterable)


def iter_none(iterable: Iterable[T], predicate: Callable[[T], bool]) -> bool:
    """
    Check if no element satisfies the predicate.

    Example:
        iter_none([1, 2, 3], lambda x: x > 5) -> True
    """
    return not any(predicate(item) for item in iterable)


def iter_count[T](iterable: Iterable[T]) -> int:
    """
    Count the number of elements.

    Example:
        iter_count([1, 2, 3]) -> 3
    """
    return sum(1 for _ in iterable)


def iter_count_if(iterable: Iterable[T], predicate: Callable[[T], bool]) -> int:
    """
    Count elements that satisfy the predicate.

    Example:
        iter_count_if([1, 2, 3, 4], lambda x: x % 2 == 0) -> 2
    """
    return sum(1 for item in iterable if predicate(item))


# =============================================================================
# Numeric Methods
# =============================================================================


def iter_sum(iterable: Iterable[Any]) -> Any:
    """
    Calculate the sum of all elements.

    Example:
        iter_sum([1, 2, 3, 4]) -> 10
    """
    return sum(iterable)


def iter_product(iterable: Iterable[Any]) -> Any:
    """
    Calculate the product of all elements.

    Example:
        iter_product([1, 2, 3, 4]) -> 24
    """
    result = 1
    for item in iterable:
        result *= item
    return result


def iter_min[T](iterable: Iterable[T]) -> T | None:
    """
    Find the minimum element.

    Example:
        iter_min([3, 1, 4, 1, 5]) -> 1
    """
    items = list(iterable)
    if not items:
        return None
    return min(items)


def iter_max[T](iterable: Iterable[T]) -> T | None:
    """
    Find the maximum element.

    Example:
        iter_max([3, 1, 4, 1, 5]) -> 5
    """
    items = list(iterable)
    if not items:
        return None
    return max(items)


def iter_average(iterable: Iterable[Any]) -> float | None:
    """
    Calculate the average of all elements.

    Example:
        iter_average([1, 2, 3, 4]) -> 2.5
    """
    items = list(iterable)
    if not items:
        return None
    return sum(items) / len(items)


def iter_min_by(iterable: Iterable[T], key: Callable[[T], Any]) -> T | None:
    """
    Find the minimum element by a key function.

    Example:
        iter_min_by(["a", "bbb", "cc"], lambda x: len(x)) -> "a"
    """
    items = list(iterable)
    if not items:
        return None
    return min(items, key=key)


def iter_max_by(iterable: Iterable[T], key: Callable[[T], Any]) -> T | None:
    """
    Find the maximum element by a key function.

    Example:
        iter_max_by(["a", "bbb", "cc"], lambda x: len(x)) -> "bbb"
    """
    items = list(iterable)
    if not items:
        return None
    return max(items, key=key)


# =============================================================================
# Slicing Methods
# =============================================================================


def iter_take[T](iterable: Iterable[T], n: int) -> list[T]:
    """
    Take the first n elements.

    Example:
        iter_take([1, 2, 3, 4, 5], 3) -> [1, 2, 3]
    """
    result: list[T] = []
    for i, item in enumerate(iterable):
        if i >= n:
            break
        result.append(item)
    return result


def iter_skip[T](iterable: Iterable[T], n: int) -> list[T]:
    """
    Skip the first n elements.

    Example:
        iter_skip([1, 2, 3, 4, 5], 2) -> [3, 4, 5]
    """
    result: list[T] = []
    for i, item in enumerate(iterable):
        if i >= n:
            result.append(item)
    return result


def iter_take_while(iterable: Iterable[T], predicate: Callable[[T], bool]) -> list[T]:
    """
    Take elements while the predicate is true.

    Example:
        iter_take_while([1, 2, 3, 4, 1], lambda x: x < 4) -> [1, 2, 3]
    """
    result: list[T] = []
    for item in iterable:
        if not predicate(item):
            break
        result.append(item)
    return result


def iter_skip_while(iterable: Iterable[T], predicate: Callable[[T], bool]) -> list[T]:
    """
    Skip elements while the predicate is true.

    Example:
        iter_skip_while([1, 2, 3, 4, 1], lambda x: x < 3) -> [3, 4, 1]
    """
    result: list[T] = []
    dropping = True
    for item in iterable:
        if dropping and predicate(item):
            continue
        dropping = False
        result.append(item)
    return result


# =============================================================================
# Ordering Methods
# =============================================================================


def iter_sorted[T](iterable: Iterable[T]) -> list[T]:
    """
    Return a sorted list of elements.

    Example:
        iter_sorted([3, 1, 4, 1, 5]) -> [1, 1, 3, 4, 5]
    """
    return sorted(iterable)  # type: ignore


def iter_sorted_by(iterable: Iterable[T], key: Callable[[T], Any]) -> list[T]:
    """
    Return a sorted list of elements by a key function.

    Example:
        iter_sorted_by(["bbb", "a", "cc"], lambda x: len(x)) -> ["a", "cc", "bbb"]
    """
    return sorted(iterable, key=key)


def iter_sorted_by_desc(iterable: Iterable[T], key: Callable[[T], Any]) -> list[T]:
    """
    Return a sorted list of elements by a key function in descending order.

    Example:
        iter_sorted_by_desc(["bbb", "a", "cc"], lambda x: len(x)) -> ["bbb", "cc", "a"]
    """
    return sorted(iterable, key=key, reverse=True)


def iter_reversed[T](iterable: Iterable[T]) -> list[T]:
    """
    Return a reversed list of elements.

    Example:
        iter_reversed([1, 2, 3]) -> [3, 2, 1]
    """
    items = list(iterable)
    return items[::-1]


# =============================================================================
# Combination Methods
# =============================================================================


def iter_zip[T, U](iterable1: Iterable[T], iterable2: Iterable[U]) -> list[tuple[T, U]]:
    """
    Zip two iterables together.

    Example:
        iter_zip([1, 2, 3], ["a", "b", "c"]) -> [(1, "a"), (2, "b"), (3, "c")]
    """
    return list(zip(iterable1, iterable2, strict=False))


def iter_enumerate[T](iterable: Iterable[T]) -> list[tuple[int, T]]:
    """
    Enumerate the elements with their indices.

    Example:
        iter_enumerate(["a", "b", "c"]) -> [(0, "a"), (1, "b"), (2, "c")]
    """
    return list(enumerate(iterable))


def iter_chain[T](*iterables: Iterable[T]) -> list[T]:
    """
    Chain multiple iterables together.

    Example:
        iter_chain([1, 2], [3, 4], [5, 6]) -> [1, 2, 3, 4, 5, 6]
    """
    result: list[T] = []
    for iterable in iterables:
        result.extend(iterable)
    return result


def iter_chunk[T](iterable: Iterable[T], size: int) -> list[list[T]]:
    """
    Split the iterable into chunks of the given size.

    Example:
        iter_chunk([1, 2, 3, 4, 5], 2) -> [[1, 2], [3, 4], [5]]
    """
    items = list(iterable)
    return [items[i : i + size] for i in range(0, len(items), size)]


def iter_unique[T](iterable: Iterable[T]) -> list[T]:
    """
    Return unique elements while preserving order.

    Example:
        iter_unique([1, 2, 2, 3, 1, 4]) -> [1, 2, 3, 4]
    """
    seen: set[T] = set()
    result: list[T] = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# =============================================================================
# Collection Methods
# =============================================================================


def iter_collect_list[T](iterable: Iterable[T]) -> list[T]:
    """
    Collect elements into a list.

    Example:
        iter_collect_list(x for x in range(3)) -> [0, 1, 2]
    """
    return list(iterable)


def iter_collect_set[T](iterable: Iterable[T]) -> set[T]:
    """
    Collect elements into a set.

    Example:
        iter_collect_set([1, 2, 2, 3]) -> {1, 2, 3}
    """
    return set(iterable)


def iter_collect_dict[K, V](iterable: Iterable[tuple[K, V]]) -> dict[K, V]:
    """
    Collect key-value pairs into a dict.

    Example:
        iter_collect_dict([("a", 1), ("b", 2)]) -> {"a": 1, "b": 2}
    """
    return dict(iterable)


def iter_join(iterable: Iterable[str], separator: str) -> str:
    """
    Join string elements with a separator.

    Example:
        iter_join(["a", "b", "c"], ", ") -> "a, b, c"
    """
    return separator.join(iterable)


def iter_group_by(iterable: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    """
    Group elements by a key function.

    Example:
        iter_group_by([1, 2, 3, 4], lambda x: x % 2) -> {0: [2, 4], 1: [1, 3]}
    """
    result: dict[K, list[T]] = {}
    for item in iterable:
        k = key(item)
        if k not in result:
            result[k] = []
        result[k].append(item)
    return result


def iter_partition(
    iterable: Iterable[T], predicate: Callable[[T], bool]
) -> tuple[list[T], list[T]]:
    """
    Partition elements into two lists based on a predicate.

    Returns (true_elements, false_elements).

    Example:
        iter_partition([1, 2, 3, 4], lambda x: x % 2 == 0) -> ([2, 4], [1, 3])
    """
    true_list: list[T] = []
    false_list: list[T] = []
    for item in iterable:
        if predicate(item):
            true_list.append(item)
        else:
            false_list.append(item)
    return (true_list, false_list)


# =============================================================================
# Dict-specific Methods
# =============================================================================


def dict_keys[K, V](d: dict[K, V]) -> list[K]:
    """Return the keys of a dictionary as a list."""
    return list(d.keys())


def dict_values[K, V](d: dict[K, V]) -> list[V]:
    """Return the values of a dictionary as a list."""
    return list(d.values())


def dict_items[K, V](d: dict[K, V]) -> list[tuple[K, V]]:
    """Return the key-value pairs of a dictionary as a list of tuples."""
    return list(d.items())


def dict_map_values(d: dict[K, V], func: Callable[[V], U]) -> dict[K, U]:
    """Apply a function to each value in the dictionary."""
    return {k: func(v) for k, v in d.items()}


def dict_filter_keys(d: dict[K, V], predicate: Callable[[K], bool]) -> dict[K, V]:
    """Filter dictionary entries by key predicate."""
    return {k: v for k, v in d.items() if predicate(k)}


def dict_filter_values(d: dict[K, V], predicate: Callable[[V], bool]) -> dict[K, V]:
    """Filter dictionary entries by value predicate."""
    return {k: v for k, v in d.items() if predicate(v)}
