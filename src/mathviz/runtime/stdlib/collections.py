"""
MathViz Standard Library - Collections Module.

Provides collection manipulation functions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce as _reduce
from typing import Any, TypeVar

T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")
V = TypeVar("V")


# =============================================================================
# Basic Functions
# =============================================================================


def len(seq: Iterable) -> int:
    """Return length of sequence."""
    return __builtins__["len"](seq)


def range(*args) -> range:
    """Return range object."""
    return __builtins__["range"](*args)


def enumerate[T](seq: Iterable[T], start: int = 0) -> Iterable[tuple[int, T]]:
    """Enumerate items with index."""
    return __builtins__["enumerate"](seq, start)


def zip(*seqs: Iterable) -> Iterable[tuple]:
    """Zip multiple sequences together."""
    return __builtins__["zip"](*seqs)


def reversed[T](seq: Iterable[T]) -> list[T]:
    """Return reversed list."""
    return list(__builtins__["reversed"](list(seq)))


def sorted(
    seq: Iterable[T], key: Callable[[T], Any] | None = None, reverse: bool = False
) -> list[T]:
    """Return sorted list."""
    return __builtins__["sorted"](seq, key=key, reverse=reverse)


# =============================================================================
# Aggregation
# =============================================================================


def sum(seq: Iterable[T], start: T = 0) -> T:
    """Sum all elements."""
    return __builtins__["sum"](seq, start)


def prod[T](seq: Iterable[T]) -> T:
    """Product of all elements."""
    result = 1
    for x in seq:
        result *= x
    return result


def all(seq: Iterable[bool]) -> bool:
    """Return True if all elements are truthy."""
    return __builtins__["all"](seq)


def any(seq: Iterable[bool]) -> bool:
    """Return True if any element is truthy."""
    return __builtins__["any"](seq)


def none(seq: Iterable[bool]) -> bool:
    """Return True if no element is truthy."""
    return not any(seq)


# =============================================================================
# Transformation
# =============================================================================


def filter(pred: Callable[[T], bool], seq: Iterable[T]) -> list[T]:
    """Filter elements by predicate."""
    return list(__builtins__["filter"](pred, seq))


def map(func: Callable[[T], U], seq: Iterable[T]) -> list[U]:
    """Map function over elements."""
    return list(__builtins__["map"](func, seq))


def reduce(func: Callable[[T, T], T], seq: Iterable[T], initial: T | None = None) -> T:
    """Reduce sequence with function."""
    if initial is not None:
        return _reduce(func, seq, initial)
    return _reduce(func, seq)


def flat_map(func: Callable[[T], Iterable[U]], seq: Iterable[T]) -> list[U]:
    """Map and flatten results."""
    result = []
    for item in seq:
        result.extend(func(item))
    return result


def flatten[T](seq: Iterable[Iterable[T]]) -> list[T]:
    """Flatten one level of nesting."""
    result = []
    for item in seq:
        result.extend(item)
    return result


def deep_flatten(seq: Iterable) -> list:
    """Recursively flatten all levels."""
    result = []
    for item in seq:
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            result.extend(deep_flatten(item))
        else:
            result.append(item)
    return result


# =============================================================================
# Access
# =============================================================================


def first(seq: Iterable[T], default: T | None = None) -> T | None:
    """Return first element or default."""
    for item in seq:
        return item
    return default


def last(seq: Iterable[T], default: T | None = None) -> T | None:
    """Return last element or default."""
    result = default
    for item in seq:
        result = item
    return result


def nth(seq: Iterable[T], n: int, default: T | None = None) -> T | None:
    """Return nth element or default."""
    for i, item in enumerate(seq):
        if i == n:
            return item
    return default


def find(pred: Callable[[T], bool], seq: Iterable[T], default: T | None = None) -> T | None:
    """Find first element matching predicate."""
    for item in seq:
        if pred(item):
            return item
    return default


def position(pred: Callable[[T], bool], seq: Iterable[T]) -> int:
    """Find index of first element matching predicate, -1 if not found."""
    for i, item in enumerate(seq):
        if pred(item):
            return i
    return -1


def index_of(seq: Iterable[T], value: T) -> int:
    """Find index of value, -1 if not found."""
    for i, item in enumerate(seq):
        if item == value:
            return i
    return -1


# =============================================================================
# Slicing
# =============================================================================


def take[T](seq: Iterable[T], n: int) -> list[T]:
    """Take first n elements."""
    result = []
    for i, item in enumerate(seq):
        if i >= n:
            break
        result.append(item)
    return result


def drop[T](seq: Iterable[T], n: int) -> list[T]:
    """Drop first n elements."""
    result = []
    for i, item in enumerate(seq):
        if i >= n:
            result.append(item)
    return result


def take_while(pred: Callable[[T], bool], seq: Iterable[T]) -> list[T]:
    """Take elements while predicate is true."""
    result = []
    for item in seq:
        if not pred(item):
            break
        result.append(item)
    return result


def drop_while(pred: Callable[[T], bool], seq: Iterable[T]) -> list[T]:
    """Drop elements while predicate is true."""
    result = []
    dropping = True
    for item in seq:
        if dropping and pred(item):
            continue
        dropping = False
        result.append(item)
    return result


def slice[T](seq: Iterable[T], start: int, end: int | None = None, step: int = 1) -> list[T]:
    """Slice sequence with start, end, step."""
    seq_list = list(seq)
    if end is None:
        return seq_list[start::step]
    return seq_list[start:end:step]


# =============================================================================
# Grouping
# =============================================================================


def chunk[T](seq: Iterable[T], size: int) -> list[list[T]]:
    """Split into chunks of given size."""
    result = []
    current = []
    for item in seq:
        current.append(item)
        if len(current) == size:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def unique[T](seq: Iterable[T]) -> list[T]:
    """Return unique elements preserving order."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def group_by(key: Callable[[T], K], seq: Iterable[T]) -> dict[K, list[T]]:
    """Group elements by key function."""
    result: dict[K, list[T]] = {}
    for item in seq:
        k = key(item)
        if k not in result:
            result[k] = []
        result[k].append(item)
    return result


def partition(pred: Callable[[T], bool], seq: Iterable[T]) -> tuple[list[T], list[T]]:
    """Partition into (matching, not matching)."""
    yes, no = [], []
    for item in seq:
        if pred(item):
            yes.append(item)
        else:
            no.append(item)
    return yes, no


def frequency[T](seq: Iterable[T]) -> dict[T, int]:
    """Count frequency of each element."""
    result: dict[T, int] = {}
    for item in seq:
        result[item] = result.get(item, 0) + 1
    return result


# =============================================================================
# Combination
# =============================================================================


def zip_with(func: Callable, *seqs: Iterable) -> list:
    """Zip sequences and apply function to each tuple."""
    return [func(*items) for items in zip(*seqs)]


def interleave[T](*seqs: Iterable[T]) -> list[T]:
    """Interleave multiple sequences."""
    result = []
    iters = [iter(s) for s in seqs]
    while iters:
        next_iters = []
        for it in iters:
            try:
                result.append(next(it))
                next_iters.append(it)
            except StopIteration:
                pass
        iters = next_iters
    return result


def repeat[T](value: T, n: int) -> list[T]:
    """Repeat value n times."""
    return [value] * n


def cycle[T](seq: Iterable[T], n: int) -> list[T]:
    """Cycle through sequence n times."""
    seq_list = list(seq)
    return seq_list * n


# =============================================================================
# Dictionary Operations
# =============================================================================


def keys[K, V](d: dict[K, V]) -> list[K]:
    """Return list of keys."""
    return list(d.keys())


def values[K, V](d: dict[K, V]) -> list[V]:
    """Return list of values."""
    return list(d.values())


def items[K, V](d: dict[K, V]) -> list[tuple[K, V]]:
    """Return list of (key, value) pairs."""
    return list(d.items())


def get(d: dict[K, V], key: K, default: V | None = None) -> V | None:
    """Get value by key with default."""
    return d.get(key, default)


def merge(*dicts: dict) -> dict:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def invert[K, V](d: dict[K, V]) -> dict[V, K]:
    """Invert dictionary (swap keys and values)."""
    return {v: k for k, v in d.items()}


def select_keys(d: dict[K, V], keys: Iterable[K]) -> dict[K, V]:
    """Select only specified keys."""
    return {k: d[k] for k in keys if k in d}


def omit_keys(d: dict[K, V], keys: Iterable[K]) -> dict[K, V]:
    """Omit specified keys."""
    keys_set = set(keys)
    return {k: v for k, v in d.items() if k not in keys_set}
