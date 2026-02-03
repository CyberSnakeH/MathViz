"""
MathViz Runtime Mathematical Operations.

This module provides runtime support for mathematical set operations
that are translated from MathViz Unicode operators. These functions
are designed to work with Python sets while providing clear semantics
and good error messages.
"""

from collections.abc import Iterable
from typing import Any, TypeVar, Union

T = TypeVar("T")
SetLike = Union[set[T], frozenset[T], list[T], tuple[T, ...]]


def _to_set(value: Any) -> set:
    """
    Convert a value to a set if possible.

    Args:
        value: A set, frozenset, list, or tuple

    Returns:
        A set containing the elements

    Raises:
        TypeError: If the value cannot be converted to a set
    """
    if isinstance(value, set):
        return value
    if isinstance(value, (frozenset, list, tuple)):
        return set(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to set")


def set_union(a: SetLike[T], b: SetLike[T]) -> set[T]:
    """
    Compute the union of two sets (A ∪ B).

    The union contains all elements that are in A, in B, or in both.

    Args:
        a: First set
        b: Second set

    Returns:
        A new set containing elements from both sets

    Examples:
        >>> set_union({1, 2}, {2, 3})
        {1, 2, 3}
        >>> set_union([1, 2], [3, 4])
        {1, 2, 3, 4}
    """
    return _to_set(a) | _to_set(b)


def set_intersection(a: SetLike[T], b: SetLike[T]) -> set[T]:
    """
    Compute the intersection of two sets (A ∩ B).

    The intersection contains elements that are in both A and B.

    Args:
        a: First set
        b: Second set

    Returns:
        A new set containing only elements in both sets

    Examples:
        >>> set_intersection({1, 2, 3}, {2, 3, 4})
        {2, 3}
        >>> set_intersection({1, 2}, {3, 4})
        set()
    """
    return _to_set(a) & _to_set(b)


def set_difference(a: SetLike[T], b: SetLike[T]) -> set[T]:
    """
    Compute the set difference (A ∖ B or A - B).

    The difference contains elements that are in A but not in B.

    Args:
        a: First set (elements to keep)
        b: Second set (elements to remove)

    Returns:
        A new set containing elements in a but not in b

    Examples:
        >>> set_difference({1, 2, 3}, {2, 3, 4})
        {1}
        >>> set_difference({1, 2}, {3, 4})
        {1, 2}
    """
    return _to_set(a) - _to_set(b)


def is_subset(a: SetLike[T], b: SetLike[T]) -> bool:
    """
    Check if A is a subset of B (A ⊆ B).

    A is a subset of B if every element of A is also in B.
    Note: A set is always a subset of itself (A ⊆ A).

    Args:
        a: The potential subset
        b: The potential superset

    Returns:
        True if a is a subset of b

    Examples:
        >>> is_subset({1, 2}, {1, 2, 3})
        True
        >>> is_subset({1, 2}, {1, 2})
        True
        >>> is_subset({1, 4}, {1, 2, 3})
        False
    """
    return _to_set(a) <= _to_set(b)


def is_superset(a: SetLike[T], b: SetLike[T]) -> bool:
    """
    Check if A is a superset of B (A ⊇ B).

    A is a superset of B if every element of B is also in A.
    Note: A set is always a superset of itself (A ⊇ A).

    Args:
        a: The potential superset
        b: The potential subset

    Returns:
        True if a is a superset of b

    Examples:
        >>> is_superset({1, 2, 3}, {1, 2})
        True
        >>> is_superset({1, 2}, {1, 2})
        True
        >>> is_superset({1, 2}, {1, 2, 3})
        False
    """
    return _to_set(a) >= _to_set(b)


def is_proper_subset(a: SetLike[T], b: SetLike[T]) -> bool:
    """
    Check if A is a proper subset of B (A ⊂ B).

    A is a proper subset of B if A ⊆ B and A ≠ B.

    Args:
        a: The potential proper subset
        b: The potential proper superset

    Returns:
        True if a is a proper subset of b

    Examples:
        >>> is_proper_subset({1, 2}, {1, 2, 3})
        True
        >>> is_proper_subset({1, 2}, {1, 2})
        False
    """
    return _to_set(a) < _to_set(b)


def is_proper_superset(a: SetLike[T], b: SetLike[T]) -> bool:
    """
    Check if A is a proper superset of B (A ⊃ B).

    A is a proper superset of B if A ⊇ B and A ≠ B.

    Args:
        a: The potential proper superset
        b: The potential proper subset

    Returns:
        True if a is a proper superset of b

    Examples:
        >>> is_proper_superset({1, 2, 3}, {1, 2})
        True
        >>> is_proper_superset({1, 2}, {1, 2})
        False
    """
    return _to_set(a) > _to_set(b)


def is_element(element: T, collection: Iterable[T]) -> bool:
    """
    Check if an element is in a collection (x ∈ S).

    Args:
        element: The element to check for
        collection: The collection to check in

    Returns:
        True if element is in collection

    Examples:
        >>> is_element(2, {1, 2, 3})
        True
        >>> is_element(5, [1, 2, 3])
        False
    """
    return element in collection


def is_not_element(element: T, collection: Iterable[T]) -> bool:
    """
    Check if an element is not in a collection (x ∉ S).

    Args:
        element: The element to check for
        collection: The collection to check in

    Returns:
        True if element is not in collection

    Examples:
        >>> is_not_element(5, {1, 2, 3})
        True
        >>> is_not_element(2, [1, 2, 3])
        False
    """
    return element not in collection


def symmetric_difference(a: SetLike[T], b: SetLike[T]) -> set[T]:
    """
    Compute the symmetric difference of two sets (A △ B).

    The symmetric difference contains elements in A or B but not both.

    Args:
        a: First set
        b: Second set

    Returns:
        A new set containing elements in exactly one of the sets

    Examples:
        >>> symmetric_difference({1, 2, 3}, {2, 3, 4})
        {1, 4}
    """
    return _to_set(a) ^ _to_set(b)


def cartesian_product[T](a: Iterable[T], b: Iterable) -> set[tuple[T, Any]]:
    """
    Compute the Cartesian product of two collections (A × B).

    The Cartesian product is the set of all ordered pairs (x, y)
    where x ∈ A and y ∈ B.

    Args:
        a: First collection
        b: Second collection

    Returns:
        A set of tuples representing the Cartesian product

    Examples:
        >>> cartesian_product({1, 2}, {'a', 'b'})
        {(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')}
    """
    return {(x, y) for x in a for y in b}


def power_set[T](s: SetLike[T]) -> set[frozenset[T]]:
    """
    Compute the power set of a set (P(S) or 2^S).

    The power set is the set of all subsets, including the empty set
    and the set itself.

    Args:
        s: The input set

    Returns:
        A set of frozensets representing all subsets

    Examples:
        >>> power_set({1, 2})
        {frozenset(), frozenset({1}), frozenset({2}), frozenset({1, 2})}

    Note:
        The power set of a set with n elements has 2^n elements.
        Be cautious with large sets!
    """
    s_set = _to_set(s)
    result: set[frozenset[T]] = {frozenset()}

    for element in s_set:
        new_subsets = {subset | {element} for subset in result}
        result = result | {frozenset(s) for s in new_subsets}

    return result
