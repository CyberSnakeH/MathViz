"""
MathViz Standard Library - Async Helpers Module.

Provides async/await utilities and helpers.
"""

from __future__ import annotations
import asyncio
from typing import TypeVar, Callable, Awaitable, List, Tuple, Optional, Any
from functools import wraps

T = TypeVar("T")
U = TypeVar("U")


# =============================================================================
# Running Async Code
# =============================================================================


def run(coro: Awaitable[T]) -> T:
    """Run an async function synchronously."""
    return asyncio.run(coro)


def run_until_complete(coro: Awaitable[T]) -> T:
    """Run coroutine until complete in current event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# =============================================================================
# Concurrent Execution
# =============================================================================


async def gather(*coros: Awaitable[T]) -> List[T]:
    """Run multiple coroutines concurrently and gather results."""
    return list(await asyncio.gather(*coros))


async def gather_dict(coros: dict[str, Awaitable[T]]) -> dict[str, T]:
    """Run dict of coroutines and return dict of results."""
    keys = list(coros.keys())
    values = await asyncio.gather(*coros.values())
    return dict(zip(keys, values))


async def first_completed(*coros: Awaitable[T]) -> T:
    """Return result of first completed coroutine."""
    done, pending = await asyncio.wait(
        [asyncio.create_task(c) for c in coros], return_when=asyncio.FIRST_COMPLETED
    )
    # Cancel pending tasks
    for task in pending:
        task.cancel()
    # Return first result
    return done.pop().result()


async def race(*coros: Awaitable[T]) -> T:
    """Alias for first_completed."""
    return await first_completed(*coros)


async def all_settled(*coros: Awaitable[T]) -> List[Tuple[bool, Any]]:
    """
    Run all coroutines and return (success, value/exception) tuples.
    Never raises - captures all results/errors.
    """
    results = []
    tasks = [asyncio.create_task(c) for c in coros]
    await asyncio.wait(tasks)

    for task in tasks:
        try:
            results.append((True, task.result()))
        except Exception as e:
            results.append((False, e))

    return results


# =============================================================================
# Timing
# =============================================================================


async def sleep(seconds: float) -> None:
    """Async sleep for specified seconds."""
    await asyncio.sleep(seconds)


async def sleep_ms(milliseconds: int) -> None:
    """Async sleep for specified milliseconds."""
    await asyncio.sleep(milliseconds / 1000)


async def timeout(coro: Awaitable[T], seconds: float) -> Optional[T]:
    """Run coroutine with timeout, return None on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return None


async def timeout_or_raise(coro: Awaitable[T], seconds: float) -> T:
    """Run coroutine with timeout, raise TimeoutError on timeout."""
    return await asyncio.wait_for(coro, timeout=seconds)


# =============================================================================
# Retry Logic
# =============================================================================


async def retry(
    coro_factory: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> T:
    """
    Retry a coroutine with exponential backoff.

    Args:
        coro_factory: Function that creates a new coroutine
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each failure
        exceptions: Tuple of exception types to catch and retry
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff

    raise last_exception


# =============================================================================
# Queues and Channels
# =============================================================================


def create_queue(maxsize: int = 0) -> asyncio.Queue:
    """Create an async queue."""
    return asyncio.Queue(maxsize=maxsize)


async def send(queue: asyncio.Queue, item: T) -> None:
    """Send item to queue."""
    await queue.put(item)


async def receive(queue: asyncio.Queue) -> T:
    """Receive item from queue."""
    return await queue.get()


def try_send(queue: asyncio.Queue, item: T) -> bool:
    """Try to send item without blocking. Returns success."""
    try:
        queue.put_nowait(item)
        return True
    except asyncio.QueueFull:
        return False


def try_receive(queue: asyncio.Queue) -> Optional[T]:
    """Try to receive item without blocking. Returns None if empty."""
    try:
        return queue.get_nowait()
    except asyncio.QueueEmpty:
        return None


# =============================================================================
# Semaphores and Locks
# =============================================================================


def create_lock() -> asyncio.Lock:
    """Create an async lock."""
    return asyncio.Lock()


def create_semaphore(value: int = 1) -> asyncio.Semaphore:
    """Create an async semaphore."""
    return asyncio.Semaphore(value)


def create_event() -> asyncio.Event:
    """Create an async event."""
    return asyncio.Event()


# =============================================================================
# Iteration
# =============================================================================


async def async_map(func: Callable[[T], Awaitable[U]], items: List[T]) -> List[U]:
    """Map async function over items concurrently."""
    return await asyncio.gather(*[func(item) for item in items])


async def async_filter(pred: Callable[[T], Awaitable[bool]], items: List[T]) -> List[T]:
    """Filter items using async predicate."""
    results = await asyncio.gather(*[pred(item) for item in items])
    return [item for item, keep in zip(items, results) if keep]


async def async_reduce(
    func: Callable[[T, T], Awaitable[T]], items: List[T], initial: Optional[T] = None
) -> T:
    """Reduce items using async function (sequentially)."""
    if not items:
        if initial is not None:
            return initial
        raise ValueError("Empty sequence with no initial value")

    if initial is not None:
        result = initial
        start = 0
    else:
        result = items[0]
        start = 1

    for item in items[start:]:
        result = await func(result, item)

    return result


async def async_for_each(func: Callable[[T], Awaitable[None]], items: List[T]) -> None:
    """Apply async function to each item concurrently."""
    await asyncio.gather(*[func(item) for item in items])


async def async_for_each_sequential(func: Callable[[T], Awaitable[None]], items: List[T]) -> None:
    """Apply async function to each item sequentially."""
    for item in items:
        await func(item)


# =============================================================================
# Task Management
# =============================================================================


def create_task(coro: Awaitable[T]) -> asyncio.Task[T]:
    """Create a task from a coroutine."""
    return asyncio.create_task(coro)


async def cancel_task(task: asyncio.Task) -> None:
    """Cancel a task and wait for cancellation."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def is_task_done(task: asyncio.Task) -> bool:
    """Check if task is done."""
    return task.done()


def get_task_result(task: asyncio.Task) -> Any:
    """Get task result (raises if not done)."""
    return task.result()


# =============================================================================
# Decorators
# =============================================================================


def async_cached(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Cache results of async function."""
    cache = {}

    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = await func(*args, **kwargs)
        return cache[key]

    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def async_throttle(calls: int, period: float):
    """Limit async function to `calls` invocations per `period` seconds."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        semaphore = asyncio.Semaphore(calls)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                result = await func(*args, **kwargs)
                asyncio.get_event_loop().call_later(period, semaphore.release)
                try:
                    await semaphore.acquire()
                except:
                    pass
                return result

        return wrapper

    return decorator
