"""
MathViz Standard Library - Time Module.

Provides time-related functions.
"""

from __future__ import annotations

import time as _time
from datetime import datetime


def now() -> float:
    """Return current time as Unix timestamp (seconds)."""
    return _time.time()


def time_ns() -> int:
    """Return current time in nanoseconds."""
    return _time.time_ns()


def time_ms() -> int:
    """Return current time in milliseconds."""
    return int(_time.time() * 1000)


def sleep(seconds: float) -> None:
    """Sleep for given number of seconds."""
    _time.sleep(seconds)


def sleep_ms(milliseconds: int) -> None:
    """Sleep for given number of milliseconds."""
    _time.sleep(milliseconds / 1000)


class Stopwatch:
    """Simple stopwatch for measuring elapsed time."""

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed: float = 0.0
        self._running: bool = False

    def start(self) -> Stopwatch:
        """Start the stopwatch."""
        if not self._running:
            self._start = _time.perf_counter()
            self._running = True
        return self

    def stop(self) -> Stopwatch:
        """Stop the stopwatch."""
        if self._running:
            self._elapsed += _time.perf_counter() - self._start
            self._running = False
        return self

    def reset(self) -> Stopwatch:
        """Reset the stopwatch."""
        self._elapsed = 0.0
        if self._running:
            self._start = _time.perf_counter()
        return self

    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        if self._running:
            return self._elapsed + (_time.perf_counter() - self._start)
        return self._elapsed

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed() * 1000


def elapsed(func) -> float:
    """Measure execution time of a function."""
    start = _time.perf_counter()
    func()
    return _time.perf_counter() - start


def format_time(timestamp: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp as string."""
    return datetime.fromtimestamp(timestamp).strftime(fmt)


def parse_time(s: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> float:
    """Parse time string to timestamp."""
    return datetime.strptime(s, fmt).timestamp()


def today() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def year() -> int:
    """Return current year."""
    return datetime.now().year


def month() -> int:
    """Return current month (1-12)."""
    return datetime.now().month


def day() -> int:
    """Return current day of month."""
    return datetime.now().day


def hour() -> int:
    """Return current hour (0-23)."""
    return datetime.now().hour


def minute() -> int:
    """Return current minute (0-59)."""
    return datetime.now().minute


def second() -> int:
    """Return current second (0-59)."""
    return datetime.now().second


def weekday() -> int:
    """Return current weekday (0=Monday, 6=Sunday)."""
    return datetime.now().weekday()


def is_leap_year(year: int) -> bool:
    """Check if year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    """Return number of days in given month."""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    raise ValueError(f"Invalid month: {month}")


def add_days(timestamp: float, days: int) -> float:
    """Add days to timestamp."""
    return timestamp + days * 86400


def add_hours(timestamp: float, hours: int) -> float:
    """Add hours to timestamp."""
    return timestamp + hours * 3600


def add_minutes(timestamp: float, minutes: int) -> float:
    """Add minutes to timestamp."""
    return timestamp + minutes * 60


def diff_days(t1: float, t2: float) -> float:
    """Return difference between timestamps in days."""
    return (t2 - t1) / 86400


def diff_hours(t1: float, t2: float) -> float:
    """Return difference between timestamps in hours."""
    return (t2 - t1) / 3600


def diff_minutes(t1: float, t2: float) -> float:
    """Return difference between timestamps in minutes."""
    return (t2 - t1) / 60
