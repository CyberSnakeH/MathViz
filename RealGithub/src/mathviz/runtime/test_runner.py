"""
MathViz Test Runner.

Provides a test framework for MathViz programs with:
- Test discovery
- Test execution
- Assertions
- Result reporting
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Any, Dict
from enum import Enum
import time
import traceback
import sys


class TestStatus(Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    duration: float  # seconds
    message: Optional[str] = None
    location: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    duration: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.errors == 0


# =============================================================================
# Assertions
# =============================================================================

class AssertionError(Exception):
    """Raised when an assertion fails."""
    pass


def assert_true(condition: bool, message: str = "") -> None:
    """Assert that condition is true."""
    if not condition:
        raise AssertionError(message or "Expected true, got false")


def assert_false(condition: bool, message: str = "") -> None:
    """Assert that condition is false."""
    if condition:
        raise AssertionError(message or "Expected false, got true")


def assert_eq(actual: Any, expected: Any, message: str = "") -> None:
    """Assert that actual equals expected."""
    if actual != expected:
        raise AssertionError(
            message or f"Expected {expected!r}, got {actual!r}"
        )


def assert_ne(actual: Any, expected: Any, message: str = "") -> None:
    """Assert that actual does not equal expected."""
    if actual == expected:
        raise AssertionError(
            message or f"Expected {actual!r} to not equal {expected!r}"
        )


def assert_lt(actual: Any, expected: Any, message: str = "") -> None:
    """Assert that actual < expected."""
    if not actual < expected:
        raise AssertionError(
            message or f"Expected {actual!r} < {expected!r}"
        )


def assert_le(actual: Any, expected: Any, message: str = "") -> None:
    """Assert that actual <= expected."""
    if not actual <= expected:
        raise AssertionError(
            message or f"Expected {actual!r} <= {expected!r}"
        )


def assert_gt(actual: Any, expected: Any, message: str = "") -> None:
    """Assert that actual > expected."""
    if not actual > expected:
        raise AssertionError(
            message or f"Expected {actual!r} > {expected!r}"
        )


def assert_ge(actual: Any, expected: Any, message: str = "") -> None:
    """Assert that actual >= expected."""
    if not actual >= expected:
        raise AssertionError(
            message or f"Expected {actual!r} >= {expected!r}"
        )


def assert_in(item: Any, container: Any, message: str = "") -> None:
    """Assert that item is in container."""
    if item not in container:
        raise AssertionError(
            message or f"Expected {item!r} to be in {container!r}"
        )


def assert_not_in(item: Any, container: Any, message: str = "") -> None:
    """Assert that item is not in container."""
    if item in container:
        raise AssertionError(
            message or f"Expected {item!r} to not be in {container!r}"
        )


def assert_none(value: Any, message: str = "") -> None:
    """Assert that value is None."""
    if value is not None:
        raise AssertionError(
            message or f"Expected None, got {value!r}"
        )


def assert_not_none(value: Any, message: str = "") -> None:
    """Assert that value is not None."""
    if value is None:
        raise AssertionError(message or "Expected non-None value")


def assert_approx(actual: float, expected: float, tolerance: float = 1e-9, message: str = "") -> None:
    """Assert that actual is approximately equal to expected."""
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            message or f"Expected {expected} ± {tolerance}, got {actual}"
        )


def assert_raises(exception_type: type, func: Callable, *args, **kwargs) -> None:
    """Assert that function raises specified exception."""
    try:
        func(*args, **kwargs)
    except exception_type:
        return
    except Exception as e:
        raise AssertionError(
            f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"Expected {exception_type.__name__} to be raised")


def assert_type(value: Any, expected_type: type, message: str = "") -> None:
    """Assert that value is of expected type."""
    if not isinstance(value, expected_type):
        raise AssertionError(
            message or f"Expected type {expected_type.__name__}, got {type(value).__name__}"
        )


def assert_len(container: Any, expected_len: int, message: str = "") -> None:
    """Assert that container has expected length."""
    actual_len = len(container)
    if actual_len != expected_len:
        raise AssertionError(
            message or f"Expected length {expected_len}, got {actual_len}"
        )


def assert_empty(container: Any, message: str = "") -> None:
    """Assert that container is empty."""
    if len(container) != 0:
        raise AssertionError(
            message or f"Expected empty container, got {len(container)} items"
        )


def assert_not_empty(container: Any, message: str = "") -> None:
    """Assert that container is not empty."""
    if len(container) == 0:
        raise AssertionError(message or "Expected non-empty container")


# =============================================================================
# Test Runner
# =============================================================================

class TestRunner:
    """Runs tests and collects results."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._tests: List[tuple[str, Callable, Dict[str, Any]]] = []

    def add_test(self, name: str, func: Callable, **options) -> None:
        """Add a test to run."""
        self._tests.append((name, func, options))

    def run(self) -> TestSuite:
        """Run all registered tests."""
        suite = TestSuite(name="MathViz Tests")
        start_time = time.time()

        for name, func, options in self._tests:
            result = self._run_single_test(name, func, options)
            suite.results.append(result)

            if self.verbose:
                self._print_result(result)

        suite.duration = time.time() - start_time

        if self.verbose:
            self._print_summary(suite)

        return suite

    def _run_single_test(self, name: str, func: Callable, options: Dict[str, Any]) -> TestResult:
        """Run a single test and return result."""
        should_panic = options.get("should_panic", False)
        skip = options.get("skip", False)

        if skip:
            return TestResult(
                name=name,
                status=TestStatus.SKIPPED,
                duration=0.0,
                message=options.get("skip_reason", "Skipped"),
            )

        start_time = time.time()

        try:
            func()
            duration = time.time() - start_time

            if should_panic:
                return TestResult(
                    name=name,
                    status=TestStatus.FAILED,
                    duration=duration,
                    message="Expected panic, but test passed",
                )

            return TestResult(
                name=name,
                status=TestStatus.PASSED,
                duration=duration,
            )

        except AssertionError as e:
            duration = time.time() - start_time

            if should_panic:
                return TestResult(
                    name=name,
                    status=TestStatus.PASSED,
                    duration=duration,
                )

            return TestResult(
                name=name,
                status=TestStatus.FAILED,
                duration=duration,
                message=str(e),
                traceback=traceback.format_exc(),
            )

        except Exception as e:
            duration = time.time() - start_time

            if should_panic:
                return TestResult(
                    name=name,
                    status=TestStatus.PASSED,
                    duration=duration,
                )

            return TestResult(
                name=name,
                status=TestStatus.ERROR,
                duration=duration,
                message=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(),
            )

    def _print_result(self, result: TestResult) -> None:
        """Print a single test result."""
        status_symbols = {
            TestStatus.PASSED: "✓",
            TestStatus.FAILED: "✗",
            TestStatus.SKIPPED: "○",
            TestStatus.ERROR: "!",
        }
        status_colors = {
            TestStatus.PASSED: "\033[32m",  # Green
            TestStatus.FAILED: "\033[31m",  # Red
            TestStatus.SKIPPED: "\033[33m",  # Yellow
            TestStatus.ERROR: "\033[31m",   # Red
        }
        reset = "\033[0m"

        symbol = status_symbols[result.status]
        color = status_colors[result.status]
        duration_ms = result.duration * 1000

        print(f"{color}{symbol}{reset} {result.name} ({duration_ms:.1f}ms)")

        if result.message and result.status in (TestStatus.FAILED, TestStatus.ERROR):
            print(f"  {result.message}")

    def _print_summary(self, suite: TestSuite) -> None:
        """Print test suite summary."""
        print()
        print("=" * 60)
        print(f"Tests: {suite.total}, Passed: {suite.passed}, Failed: {suite.failed}, "
              f"Skipped: {suite.skipped}, Errors: {suite.errors}")
        print(f"Duration: {suite.duration:.2f}s")

        if suite.success:
            print("\033[32mAll tests passed!\033[0m")
        else:
            print("\033[31mSome tests failed.\033[0m")


def run_tests(test_module: Any, verbose: bool = True) -> TestSuite:
    """
    Run all tests in a module.

    Tests are functions starting with 'test_' or decorated with @test.
    """
    runner = TestRunner(verbose=verbose)

    for name in dir(test_module):
        if name.startswith("test_"):
            func = getattr(test_module, name)
            if callable(func):
                options = getattr(func, "_test_options", {})
                runner.add_test(name, func, **options)

    return runner.run()


def test(func: Optional[Callable] = None, *, should_panic: bool = False, skip: bool = False, skip_reason: str = ""):
    """Decorator to mark a function as a test."""
    def decorator(f: Callable) -> Callable:
        f._test_options = {
            "should_panic": should_panic,
            "skip": skip,
            "skip_reason": skip_reason,
        }
        return f

    if func is not None:
        return decorator(func)
    return decorator
