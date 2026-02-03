"""
MathViz Standard Library - Random Module.

Provides random number generation functions.
"""

from __future__ import annotations
import random as _random
from typing import TypeVar, Sequence, List, Optional

T = TypeVar("T")

# Global random generator
_rng = _random.Random()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    _rng.seed(seed)


def random() -> float:
    """Return random float in [0, 1)."""
    return _rng.random()


def random_int(a: int, b: int) -> int:
    """Return random integer in [a, b] inclusive."""
    return _rng.randint(a, b)


def random_float(a: float, b: float) -> float:
    """Return random float in [a, b]."""
    return _rng.uniform(a, b)


def random_bool(probability: float = 0.5) -> bool:
    """Return True with given probability."""
    return _rng.random() < probability


def random_choice(seq: Sequence[T]) -> T:
    """Return random element from sequence."""
    return _rng.choice(seq)


def random_choices(seq: Sequence[T], k: int) -> List[T]:
    """Return k random elements with replacement."""
    return _rng.choices(seq, k=k)


def random_sample(seq: Sequence[T], k: int) -> List[T]:
    """Return k unique random elements without replacement."""
    return _rng.sample(list(seq), k)


def random_shuffle(seq: List[T]) -> List[T]:
    """Return shuffled copy of list."""
    result = list(seq)
    _rng.shuffle(result)
    return result


def shuffle_in_place(seq: List[T]) -> None:
    """Shuffle list in place."""
    _rng.shuffle(seq)


def normal(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return random value from normal distribution."""
    return _rng.gauss(mu, sigma)


def uniform(a: float, b: float) -> float:
    """Return random value from uniform distribution [a, b]."""
    return _rng.uniform(a, b)


def exponential(lambd: float = 1.0) -> float:
    """Return random value from exponential distribution."""
    return _rng.expovariate(lambd)


def triangular(low: float = 0.0, high: float = 1.0, mode: Optional[float] = None) -> float:
    """Return random value from triangular distribution."""
    return _rng.triangular(low, high, mode)


def weighted_choice(weights: List[float]) -> int:
    """Return random index weighted by given weights."""
    return _rng.choices(range(len(weights)), weights=weights)[0]


def random_bytes(n: int) -> bytes:
    """Return n random bytes."""
    return _rng.randbytes(n)


def random_string(length: int, chars: str = "abcdefghijklmnopqrstuvwxyz") -> str:
    """Return random string of given length from given characters."""
    return "".join(_rng.choice(chars) for _ in range(length))


def random_hex(length: int) -> str:
    """Return random hexadecimal string of given length."""
    return "".join(_rng.choice("0123456789abcdef") for _ in range(length))


def random_color() -> str:
    """Return random hex color string."""
    return "#" + random_hex(6)


def random_unit_vector(dim: int = 3) -> List[float]:
    """Return random unit vector of given dimension."""
    import math

    vec = [_rng.gauss(0, 1) for _ in range(dim)]
    mag = math.sqrt(sum(x * x for x in vec))
    if mag == 0:
        return [1.0] + [0.0] * (dim - 1)
    return [x / mag for x in vec]


def random_angle() -> float:
    """Return random angle in [0, 2*pi)."""
    import math

    return _rng.random() * 2 * math.pi


def coin_flip() -> bool:
    """Return random boolean (50/50)."""
    return _rng.random() < 0.5


def dice(sides: int = 6) -> int:
    """Roll a dice with given number of sides."""
    return _rng.randint(1, sides)


def dice_roll(num: int = 1, sides: int = 6) -> List[int]:
    """Roll multiple dice."""
    return [dice(sides) for _ in range(num)]
