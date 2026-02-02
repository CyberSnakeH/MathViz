"""
MathViz Standard Library - Math Module.

Provides comprehensive mathematical functions including:
- Basic arithmetic
- Trigonometry
- Exponentials and logarithms
- Rounding and clamping
- Number theory
- Vector and matrix operations
"""

from __future__ import annotations
import math as _math
from typing import Union, Tuple, List, Sequence
import numpy as np

# =============================================================================
# Constants
# =============================================================================

PI = _math.pi
E = _math.e
TAU = _math.tau
PHI = (1 + _math.sqrt(5)) / 2  # Golden ratio
INF = float('inf')
NAN = float('nan')

# =============================================================================
# Basic Math Functions
# =============================================================================

def abs(x: Union[int, float]) -> Union[int, float]:
    """Return absolute value."""
    return x if x >= 0 else -x

def min(*args) -> Union[int, float]:
    """Return minimum value."""
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        return _math.inf if not args[0] else __builtins__['min'](args[0])
    return __builtins__['min'](args)

def max(*args) -> Union[int, float]:
    """Return maximum value."""
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        return -_math.inf if not args[0] else __builtins__['max'](args[0])
    return __builtins__['max'](args)

def clamp(x: float, low: float, high: float) -> float:
    """Clamp value between low and high."""
    return low if x < low else (high if x > high else x)

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by factor t."""
    return a + (b - a) * t

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def sign(x: float) -> int:
    """Return sign of x: -1, 0, or 1."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def copysign(x: float, y: float) -> float:
    """Return x with the sign of y."""
    return _math.copysign(x, y)

# =============================================================================
# Trigonometry
# =============================================================================

def sin(x: float) -> float:
    """Sine of x (radians)."""
    return _math.sin(x)

def cos(x: float) -> float:
    """Cosine of x (radians)."""
    return _math.cos(x)

def tan(x: float) -> float:
    """Tangent of x (radians)."""
    return _math.tan(x)

def asin(x: float) -> float:
    """Arc sine in radians."""
    return _math.asin(x)

def acos(x: float) -> float:
    """Arc cosine in radians."""
    return _math.acos(x)

def atan(x: float) -> float:
    """Arc tangent in radians."""
    return _math.atan(x)

def atan2(y: float, x: float) -> float:
    """Arc tangent of y/x in radians, handling quadrants."""
    return _math.atan2(y, x)

def sinh(x: float) -> float:
    """Hyperbolic sine."""
    return _math.sinh(x)

def cosh(x: float) -> float:
    """Hyperbolic cosine."""
    return _math.cosh(x)

def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    return _math.tanh(x)

def asinh(x: float) -> float:
    """Inverse hyperbolic sine."""
    return _math.asinh(x)

def acosh(x: float) -> float:
    """Inverse hyperbolic cosine."""
    return _math.acosh(x)

def atanh(x: float) -> float:
    """Inverse hyperbolic tangent."""
    return _math.atanh(x)

def degrees(x: float) -> float:
    """Convert radians to degrees."""
    return _math.degrees(x)

def radians(x: float) -> float:
    """Convert degrees to radians."""
    return _math.radians(x)

def hypot(*args) -> float:
    """Euclidean distance from origin."""
    return _math.hypot(*args)

# =============================================================================
# Powers and Logarithms
# =============================================================================

def sqrt(x: float) -> float:
    """Square root."""
    return _math.sqrt(x)

def cbrt(x: float) -> float:
    """Cube root."""
    return x ** (1/3) if x >= 0 else -((-x) ** (1/3))

def pow(base: float, exp: float) -> float:
    """Power function."""
    return _math.pow(base, exp)

def exp(x: float) -> float:
    """Exponential e^x."""
    return _math.exp(x)

def log(x: float, base: float = E) -> float:
    """Logarithm with optional base (default: natural log)."""
    if base == E:
        return _math.log(x)
    return _math.log(x, base)

def log10(x: float) -> float:
    """Base-10 logarithm."""
    return _math.log10(x)

def log2(x: float) -> float:
    """Base-2 logarithm."""
    return _math.log2(x)

# =============================================================================
# Rounding
# =============================================================================

def floor(x: float) -> int:
    """Round down to nearest integer."""
    return _math.floor(x)

def ceil(x: float) -> int:
    """Round up to nearest integer."""
    return _math.ceil(x)

def round(x: float, digits: int = 0) -> float:
    """Round to specified decimal places."""
    return __builtins__['round'](x, digits)

def trunc(x: float) -> int:
    """Truncate to integer (toward zero)."""
    return _math.trunc(x)

def frac(x: float) -> float:
    """Fractional part of x."""
    return x - floor(x)

# =============================================================================
# Number Theory
# =============================================================================

def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    return _math.gcd(a, b)

def lcm(a: int, b: int) -> int:
    """Least common multiple."""
    return abs(a * b) // gcd(a, b) if a and b else 0

def factorial(n: int) -> int:
    """Factorial of n."""
    return _math.factorial(n)

def comb(n: int, k: int) -> int:
    """Binomial coefficient (n choose k)."""
    return _math.comb(n, k)

def perm(n: int, k: int = None) -> int:
    """Permutations of n things taken k at a time."""
    return _math.perm(n, k)

def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(_math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def prime_factors(n: int) -> List[int]:
    """Return list of prime factors."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def fibonacci(n: int) -> int:
    """Return nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# =============================================================================
# Vector Operations
# =============================================================================

def vec2(x: float = 0, y: float = 0) -> np.ndarray:
    """Create a 2D vector."""
    return np.array([x, y], dtype=np.float64)

def vec3(x: float = 0, y: float = 0, z: float = 0) -> np.ndarray:
    """Create a 3D vector."""
    return np.array([x, y, z], dtype=np.float64)

def vec4(x: float = 0, y: float = 0, z: float = 0, w: float = 0) -> np.ndarray:
    """Create a 4D vector."""
    return np.array([x, y, z, w], dtype=np.float64)

def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two vectors."""
    return float(np.dot(a, b))

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3D vectors."""
    return np.cross(a, b)

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    mag = np.linalg.norm(v)
    if mag == 0:
        return v
    return v / mag

def magnitude(v: np.ndarray) -> float:
    """Return magnitude (length) of a vector."""
    return float(np.linalg.norm(v))

def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))

# =============================================================================
# Matrix Operations
# =============================================================================

def mat2(*values) -> np.ndarray:
    """Create a 2x2 matrix."""
    if len(values) == 0:
        return np.zeros((2, 2), dtype=np.float64)
    elif len(values) == 4:
        return np.array(values, dtype=np.float64).reshape(2, 2)
    else:
        raise ValueError("mat2 requires 0 or 4 values")

def mat3(*values) -> np.ndarray:
    """Create a 3x3 matrix."""
    if len(values) == 0:
        return np.zeros((3, 3), dtype=np.float64)
    elif len(values) == 9:
        return np.array(values, dtype=np.float64).reshape(3, 3)
    else:
        raise ValueError("mat3 requires 0 or 9 values")

def mat4(*values) -> np.ndarray:
    """Create a 4x4 matrix."""
    if len(values) == 0:
        return np.zeros((4, 4), dtype=np.float64)
    elif len(values) == 16:
        return np.array(values, dtype=np.float64).reshape(4, 4)
    else:
        raise ValueError("mat4 requires 0 or 16 values")

def identity(n: int) -> np.ndarray:
    """Create an n×n identity matrix."""
    return np.eye(n, dtype=np.float64)

def transpose(m: np.ndarray) -> np.ndarray:
    """Transpose a matrix."""
    return m.T

def determinant(m: np.ndarray) -> float:
    """Calculate matrix determinant."""
    return float(np.linalg.det(m))

def inverse(m: np.ndarray) -> np.ndarray:
    """Calculate matrix inverse."""
    return np.linalg.inv(m)
