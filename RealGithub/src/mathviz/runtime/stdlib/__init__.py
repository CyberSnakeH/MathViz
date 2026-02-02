"""
MathViz Standard Library.

Provides commonly used functions for math, strings, random, time, and collections.
"""

from mathviz.runtime.stdlib.math import *
from mathviz.runtime.stdlib.string import *
from mathviz.runtime.stdlib.random import *
from mathviz.runtime.stdlib.time import *
from mathviz.runtime.stdlib.collections import *

__all__ = [
    # Math
    "abs", "min", "max", "clamp", "lerp", "smoothstep",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "sqrt", "cbrt", "pow", "exp", "log", "log10", "log2",
    "floor", "ceil", "round", "trunc", "frac",
    "sign", "copysign", "hypot", "degrees", "radians",
    "gcd", "lcm", "factorial", "comb", "perm",
    "is_prime", "prime_factors", "fibonacci",
    "PI", "E", "TAU", "PHI", "INF", "NAN",
    # Vectors
    "vec2", "vec3", "vec4", "dot", "cross", "normalize", "magnitude", "distance",
    # Matrices
    "mat2", "mat3", "mat4", "identity", "transpose", "determinant", "inverse",
    # String
    "strlen", "substr", "split", "join", "trim", "upper", "lower",
    "starts_with", "ends_with", "contains", "replace", "format",
    "char_at", "index_of", "reverse_str", "repeat_str",
    # Random
    "random", "random_int", "random_float", "random_bool",
    "random_choice", "random_shuffle", "random_sample",
    "set_seed", "normal", "uniform",
    # Time
    "now", "sleep", "time_ns", "time_ms", "elapsed",
    "format_time", "parse_time",
    # Collections
    "len", "range", "enumerate", "zip", "reversed", "sorted",
    "sum", "prod", "all", "any", "filter", "map", "reduce",
    "first", "last", "nth", "find", "position",
    "take", "drop", "chunk", "flatten", "unique", "group_by",
]
