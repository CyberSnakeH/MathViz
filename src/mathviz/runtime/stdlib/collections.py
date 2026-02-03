"""
MathViz Standard Library - Collections Module.

Provides collection manipulation functions.
"""

from __future__ import annotations
from typing import TypeVar, List, Set, Dict, Callable, Iterable, Optional, Any, Tuple
from functools import reduce as _reduce

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


def enumerate(seq: Iterable[T], start: int = 0) -> Iterable[Tuple[int, T]]:
    """Enumerate items with index."""
    return __builtins__["enumerate"](seq, start)


def zip(*seqs: Iterable) -> Iterable[Tuple]:
    """Zip multiple sequences together."""
    return __builtins__["zip"](*seqs)


def reversed(seq: Iterable[T]) -> List[T]:
    """Return reversed list."""
    return list(__builtins__["reversed"](list(seq)))


def sorted(
    seq: Iterable[T], key: Optional[Callable[[T], Any]] = None, reverse: bool = False
) -> List[T]:
    """Return sorted list."""
    return __builtins__["sorted"](seq, key=key, reverse=reverse)


# =============================================================================
# Aggregation
# =============================================================================


def sum(seq: Iterable[T], start: T = 0) -> T:
    """Sum all elements."""
    return __builtins__["sum"](seq, start)


def prod(seq: Iterable[T]) -> T:
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


def filter(pred: Callable[[T], bool], seq: Iterable[T]) -> List[T]:
    """Filter elements by predicate."""
    return list(__builtins__["filter"](pred, seq))


def map(func: Callable[[T], U], seq: Iterable[T]) -> List[U]:
    """Map function over elements."""
    return list(__builtins__["map"](func, seq))


def reduce(func: Callable[[T, T], T], seq: Iterable[T], initial: Optional[T] = None) -> T:
    """Reduce sequence with function."""
    if initial is not None:
        return _reduce(func, seq, initial)
    return _reduce(func, seq)


def flat_map(func: Callable[[T], Iterable[U]], seq: Iterable[T]) -> List[U]:
    """Map and flatten results."""
    result = []
    for item in seq:
        result.extend(func(item))
    return result


def flatten(seq: Iterable[Iterable[T]]) -> List[T]:
    """Flatten one level of nesting."""
    result = []
    for item in seq:
        result.extend(item)
    return result


def deep_flatten(seq: Iterable) -> List:
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


def first(seq: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    """Return first element or default."""
    for item in seq:
        return item
    return default


def last(seq: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    """Return last element or default."""
    result = default
    for item in seq:
        result = item
    return result


def nth(seq: Iterable[T], n: int, default: Optional[T] = None) -> Optional[T]:
    """Return nth element or default."""
    for i, item in enumerate(seq):
        if i == n:
            return item
    return default


def find(pred: Callable[[T], bool], seq: Iterable[T], default: Optional[T] = None) -> Optional[T]:
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


def take(seq: Iterable[T], n: int) -> List[T]:
    """Take first n elements."""
    result = []
    for i, item in enumerate(seq):
        if i >= n:
            break
        result.append(item)
    return result


def drop(seq: Iterable[T], n: int) -> List[T]:
    """Drop first n elements."""
    result = []
    for i, item in enumerate(seq):
        if i >= n:
            result.append(item)
    return result


def take_while(pred: Callable[[T], bool], seq: Iterable[T]) -> List[T]:
    """Take elements while predicate is true."""
    result = []
    for item in seq:
        if not pred(item):
            break
        result.append(item)
    return result


def drop_while(pred: Callable[[T], bool], seq: Iterable[T]) -> List[T]:
    """Drop elements while predicate is true."""
    result = []
    dropping = True
    for item in seq:
        if dropping and pred(item):
            continue
        dropping = False
        result.append(item)
    return result


def slice(seq: Iterable[T], start: int, end: Optional[int] = None, step: int = 1) -> List[T]:
    """Slice sequence with start, end, step."""
    seq_list = list(seq)
    if end is None:
        return seq_list[start::step]
    return seq_list[start:end:step]


# =============================================================================
# Grouping
# =============================================================================


def chunk(seq: Iterable[T], size: int) -> List[List[T]]:
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


def unique(seq: Iterable[T]) -> List[T]:
    """Return unique elements preserving order."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def group_by(key: Callable[[T], K], seq: Iterable[T]) -> Dict[K, List[T]]:
    """Group elements by key function."""
    result: Dict[K, List[T]] = {}
    for item in seq:
        k = key(item)
        if k not in result:
            result[k] = []
        result[k].append(item)
    return result


def partition(pred: Callable[[T], bool], seq: Iterable[T]) -> Tuple[List[T], List[T]]:
    """Partition into (matching, not matching)."""
    yes, no = [], []
    for item in seq:
        if pred(item):
            yes.append(item)
        else:
            no.append(item)
    return yes, no


def frequency(seq: Iterable[T]) -> Dict[T, int]:
    """Count frequency of each element."""
    result: Dict[T, int] = {}
    for item in seq:
        result[item] = result.get(item, 0) + 1
    return result


# =============================================================================
# Combination
# =============================================================================


def zip_with(func: Callable, *seqs: Iterable) -> List:
    """Zip sequences and apply function to each tuple."""
    return [func(*items) for items in zip(*seqs)]


def interleave(*seqs: Iterable[T]) -> List[T]:
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


def repeat(value: T, n: int) -> List[T]:
    """Repeat value n times."""
    return [value] * n


def cycle(seq: Iterable[T], n: int) -> List[T]:
    """Cycle through sequence n times."""
    seq_list = list(seq)
    return seq_list * n


# =============================================================================
# Dictionary Operations
# =============================================================================


def keys(d: Dict[K, V]) -> List[K]:
    """Return list of keys."""
    return list(d.keys())


def values(d: Dict[K, V]) -> List[V]:
    """Return list of values."""
    return list(d.values())


def items(d: Dict[K, V]) -> List[Tuple[K, V]]:
    """Return list of (key, value) pairs."""
    return list(d.items())


def get(d: Dict[K, V], key: K, default: Optional[V] = None) -> Optional[V]:
    """Get value by key with default."""
    return d.get(key, default)


def merge(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def invert(d: Dict[K, V]) -> Dict[V, K]:
    """Invert dictionary (swap keys and values)."""
    return {v: k for k, v in d.items()}


def select_keys(d: Dict[K, V], keys: Iterable[K]) -> Dict[K, V]:
    """Select only specified keys."""
    return {k: d[k] for k in keys if k in d}


def omit_keys(d: Dict[K, V], keys: Iterable[K]) -> Dict[K, V]:
    """Omit specified keys."""
    keys_set = set(keys)
    return {k: v for k, v in d.items() if k not in keys_set}
