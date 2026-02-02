"""
MathViz Standard Library - String Module.

Provides string manipulation functions.
"""

from __future__ import annotations
from typing import List, Optional


def strlen(s: str) -> int:
    """Return length of string."""
    return len(s)


def substr(s: str, start: int, end: Optional[int] = None) -> str:
    """Return substring from start to end (exclusive)."""
    if end is None:
        return s[start:]
    return s[start:end]


def split(s: str, sep: str = " ") -> List[str]:
    """Split string by separator."""
    return s.split(sep)


def join(items: List[str], sep: str = "") -> str:
    """Join list of strings with separator."""
    return sep.join(items)


def trim(s: str) -> str:
    """Remove leading and trailing whitespace."""
    return s.strip()


def trim_left(s: str) -> str:
    """Remove leading whitespace."""
    return s.lstrip()


def trim_right(s: str) -> str:
    """Remove trailing whitespace."""
    return s.rstrip()


def upper(s: str) -> str:
    """Convert to uppercase."""
    return s.upper()


def lower(s: str) -> str:
    """Convert to lowercase."""
    return s.lower()


def capitalize(s: str) -> str:
    """Capitalize first character."""
    return s.capitalize()


def title(s: str) -> str:
    """Convert to title case."""
    return s.title()


def starts_with(s: str, prefix: str) -> bool:
    """Check if string starts with prefix."""
    return s.startswith(prefix)


def ends_with(s: str, suffix: str) -> bool:
    """Check if string ends with suffix."""
    return s.endswith(suffix)


def contains(s: str, sub: str) -> bool:
    """Check if string contains substring."""
    return sub in s


def replace(s: str, old: str, new: str, count: int = -1) -> str:
    """Replace occurrences of old with new."""
    if count == -1:
        return s.replace(old, new)
    return s.replace(old, new, count)


def format(s: str, *args, **kwargs) -> str:
    """Format string with positional and keyword arguments."""
    return s.format(*args, **kwargs)


def char_at(s: str, index: int) -> str:
    """Get character at index."""
    if 0 <= index < len(s):
        return s[index]
    return ""


def index_of(s: str, sub: str, start: int = 0) -> int:
    """Find index of substring, -1 if not found."""
    return s.find(sub, start)


def last_index_of(s: str, sub: str) -> int:
    """Find last index of substring, -1 if not found."""
    return s.rfind(sub)


def reverse_str(s: str) -> str:
    """Reverse a string."""
    return s[::-1]


def repeat_str(s: str, n: int) -> str:
    """Repeat string n times."""
    return s * n


def pad_left(s: str, width: int, char: str = " ") -> str:
    """Pad string on left to specified width."""
    return s.rjust(width, char)


def pad_right(s: str, width: int, char: str = " ") -> str:
    """Pad string on right to specified width."""
    return s.ljust(width, char)


def center(s: str, width: int, char: str = " ") -> str:
    """Center string to specified width."""
    return s.center(width, char)


def is_digit(s: str) -> bool:
    """Check if string contains only digits."""
    return s.isdigit()


def is_alpha(s: str) -> bool:
    """Check if string contains only alphabetic characters."""
    return s.isalpha()


def is_alnum(s: str) -> bool:
    """Check if string contains only alphanumeric characters."""
    return s.isalnum()


def is_space(s: str) -> bool:
    """Check if string contains only whitespace."""
    return s.isspace()


def is_upper(s: str) -> bool:
    """Check if string is all uppercase."""
    return s.isupper()


def is_lower(s: str) -> bool:
    """Check if string is all lowercase."""
    return s.islower()


def count(s: str, sub: str) -> int:
    """Count occurrences of substring."""
    return s.count(sub)


def lines(s: str) -> List[str]:
    """Split string into lines."""
    return s.splitlines()


def words(s: str) -> List[str]:
    """Split string into words."""
    return s.split()


def chars(s: str) -> List[str]:
    """Convert string to list of characters."""
    return list(s)
